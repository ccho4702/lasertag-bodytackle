"""Fine-tune a VLA (SmolVLA by default; --pretrained lerobot/pi0_base | lerobot/pi05_base) to predict action chunks.

    cd baseline
    python -m vla.cache_images --session ../data/sample/*/
    python -m vla.train_smolvla --sessions ../data/sample/*/ --out runs_vla/smolvla_c16

Input  : upright RGB frame (3, 640, 480) + fixed task instruction (+ zero state)
Output : chunk of `--chunk` future commands [v, ω], normalized per dim by robots.<robot>.act_scale
         (the heatmap policy uses the same normalisation, so plans plug in directly)
Image  : the robot's own camera frame as stored (chaser 480x640 portrait, evader 320x240)

Only the action expert / projections are trained (SmolVLA default: frozen vision encoder and VLM).
Validation reports, for every chunk offset k, the 3-class accuracy of the predicted command at
t+k — i.e. how good a k-frame-old plan is. This is what the heatmap policy consumes when the VLA
is queried every `period` frames with `latency` frames of delay (k ∈ [latency, latency+period)).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import robots
from vla.data import TASK, VLAChunkDataset
from vla.policies import load_policy, load_finetuned, make_batch, tokenize






@torch.no_grad()
def predict_chunks(policy, loader, tokens, device, seed=0, max_batches=None):
    """Returns predicted chunks (N, chunk, 2) and targets (N, chunk, 2), normalized."""
    policy.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    P, Y = [], []
    for i, (img, act) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = make_batch(policy, img, tokens, device)
        noise = torch.randn(img.shape[0], policy.config.chunk_size, policy.config.max_action_dim,
                            device=device, generator=g)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            a = policy.predict_action_chunk(batch, noise=noise)
        P.append(a.float().cpu().numpy()); Y.append(act.numpy())
    return np.concatenate(P), np.concatenate(Y)


def chunk_metrics(pred, target):
    """Per-offset 3-class accuracy of snapped commands. pred/target normalized (N, chunk, 2)."""
    ps, ts = np.clip(np.round(pred), -1, 1), np.round(target)
    joint = (ps == ts).all(-1).mean(0)            # (chunk,)
    acc_w = (ps[..., 1] == ts[..., 1]).mean(0)
    cls, inv, cnt = np.unique(ts[:, 0], axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    hit0 = (ps[:, 0] == ts[:, 0]).all(-1)
    bacc0 = float(np.mean([hit0[inv == c].mean() for c in range(len(cls))]))
    return {'acc_joint_k': joint.tolist(), 'acc_w_k': acc_w.tolist(),
            'acc_joint_mean': float(joint.mean()), 'acc_joint_k0': float(joint[0]),
            'bacc_joint_k0': bacc0, 'majority_k0': float(cnt.max() / len(ts))}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--out', default='runs_vla/smolvla')
    ap.add_argument('--pretrained', default='lerobot/smolvla_base')
    ap.add_argument('--chunk', type=int, default=16, help='≥ latency + period (frames, ~10 Hz)')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--grad_ckpt', action='store_true', help='gradient checkpointing (π0: less memory)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--robot', default='chaser', choices=list(robots.ROBOTS))
    ap.add_argument('--label_source', default='requested', choices=['requested', 'executed'])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rb = robots.get(a.robot)
    kw = dict(chunk=a.chunk, test_frac=a.test_frac, val_frac=a.val_frac, robot=a.robot,
              label_source=a.label_source)
    tr = VLAChunkDataset(a.sessions, 'train', **kw)
    va = VLAChunkDataset(a.sessions, 'val', stride=2, **kw)
    te = VLAChunkDataset(a.sessions, 'test', stride=1, **kw)
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 4)), drop_last=True,
                    pin_memory=True, persistent_workers=int(os.environ.get('LSMA_NUM_WORKERS', 4)) > 0)
    vl = DataLoader(va, batch_size=32, shuffle=False, num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 2)))
    tel = DataLoader(te, batch_size=32, shuffle=False, num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 2))) if len(te) else None

    img_hw = tuple(np.load(os.path.join(a.sessions[0].rstrip('/'), 'images_vla.npy'), mmap_mode='r').shape[1:3])
    policy = load_policy(a.chunk, device, a.pretrained, img_hw)
    if a.grad_ckpt and hasattr(policy.config, 'gradient_checkpointing'):
        policy.config.gradient_checkpointing = True
    tokens = tokenize(policy, rb.task)
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f'trainable params: {sum(p.numel() for p in params)/1e6:.1f}M')
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    os.makedirs(a.out, exist_ok=True)
    meta = {**vars(a), 'task': rb.task, 'img_hw': list(img_hw), 'act_scale': rb.act_scale.tolist()}
    json.dump(meta, open(os.path.join(a.out, 'vla_config.json'), 'w'), indent=2)
    log = open(os.path.join(a.out, 'train_log.csv'), 'w')
    log.write('step,loss,val_acc_joint_mean,val_acc_joint_k0,lr,elapsed_s\n')

    best, step, t0, run_loss = -1.0, 0, time.time(), []
    it = iter(tl)
    while step < a.steps:
        policy.train()
        try:
            img, act = next(it)
        except StopIteration:
            it = iter(tl); img, act = next(it)
        batch = make_batch(policy, img, tokens, device)
        batch['action'] = act.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            loss, _ = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step(); sched.step(); step += 1
        run_loss.append(loss.item())

        if len(va) == 0:                 # sessions too short for a val chunk: keep the last weights
            if step == a.steps:
                policy.save_pretrained(os.path.join(a.out, 'best'))
                print(f'step {step}: no val chunks (val split < chunk) → saved last weights as best')
            continue
        if step % a.eval_every == 0 or step == a.steps:
            P, Y = predict_chunks(policy, vl, tokens, device)
            m = chunk_metrics(P, Y)
            el = time.time() - t0
            print(f'step {step:5d}  loss={np.mean(run_loss):.4f}  val acc_joint mean={m["acc_joint_mean"]:.3f} '
                  f'k0={m["acc_joint_k0"]:.3f} (majority {m["majority_k0"]:.3f})  t={el:.0f}s', flush=True)
            log.write(f'{step},{np.mean(run_loss):.5f},{m["acc_joint_mean"]:.4f},{m["acc_joint_k0"]:.4f},'
                      f'{sched.get_last_lr()[0]:.2e},{el:.1f}\n'); log.flush()
            run_loss = []
            if m['acc_joint_mean'] > best:
                best = m['acc_joint_mean']
                policy.save_pretrained(os.path.join(a.out, 'best'))
                json.dump(m, open(os.path.join(a.out, 'val_metrics.json'), 'w'), indent=2)
                print(f'  ↑ best saved ({best:.3f})')

    if len(te) == 0:     # e.g. --test_frac 0 (cross-fitting folds hold out whole sessions instead)
        return
    policy = load_finetuned(os.path.join(a.out, 'best'), device)
    P, Y = predict_chunks(policy, tel, tokens, device)
    m = chunk_metrics(P, Y)
    json.dump(m, open(os.path.join(a.out, 'test_metrics.json'), 'w'), indent=2)
    print('TEST acc_joint per offset k:', np.round(m['acc_joint_k'], 3).tolist())
    print(f'TEST acc_joint mean={m["acc_joint_mean"]:.3f}  k0={m["acc_joint_k0"]:.3f}  majority={m["majority_k0"]:.3f}')


if __name__ == '__main__':
    main()
