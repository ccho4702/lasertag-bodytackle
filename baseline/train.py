"""Train VisuomotorPolicy on observation sequences from real robot demonstrations.

Usage
    # paper student (arXiv 2603.11980): enemy+ally heatmaps + depth, ResNet18 → LSTM → MLP, L2
    python train.py --robot chaser --preset paper --sessions S1 S2 --test_sessions S3 --run_dir runs/paper

    # original repo baseline model (with leak-free block validation)
    python train.py --robot chaser --sessions S1 S2 S3 --channels enemy,ally --run_dir runs/base

    # all model-level improvements
    python train.py --sessions S1 S2 S3 --preset improved --run_dir runs/improved

    # single ablation, e.g. spatial-softmax encoder only
    python train.py --sessions S1 S2 S3 --encoder spatial --run_dir runs/abl_spatial

    # cross-session split
    python train.py --sessions S1 S2 --test_sessions S3 --run_dir runs/cross

    # resume / eval only
    python train.py --sessions ... --resume runs/improved/last.pt
    python train.py --sessions ... --resume runs/improved/best.pt --eval_only
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

import augment
from dataset import HeatmapDataset
from metrics import compute_metrics, format_metrics, label_stats
import robots
from model import ENCODERS, HEADS, build_model

DEFAULTS = dict(
    # model
    seq_len=8, feat_dim=64, lstm_hidden=128, lstm_layers=2,
    encoder='cnn', coordconv=False, head='tanh', n_bins=21,
    prev_action=False, prev_action_dropout=0.5,
    robot='chaser',          # robots.py profile: label units, command levels, act_scale
    channels='enemy,ally,depth',   # obs.npz channels by name (paper: enemy, ally heatmaps + depth)
    head_hidden=64, pretrained=True,
    # VLA conditioning (plans from vla/extract_plans.py); '' = off
    label_tag='',            # '' = operator labels (labels.csv); else actions_<tag>.npy
    label_shift=0,           # frame t ← command at t+shift; -2 = camera frames arrive ~0.2 s late (TASK.md)
    vla_tag='', vla_period=10, vla_latency=4, vla_latency_jitter=1, vla_dropout=0.2, vla_eval_phase=0,
    # loss
    loss='mse', huber_delta=0.1, w_ang=1.0, all_steps=False, label_smoothing=0.05,
    # augmentation
    aug_hflip=0.0, aug_jitter=0.0, aug_det_drop=0.0,
    # optimisation
    batch_size=64, epochs=100, lr=3e-4, lr_encoder=0.0, weight_decay=1e-4, grad_clip=1.0,
    ema=0.0,                 # EMA decay, 0 = off (e.g. 0.999)
    # data split
    val_frac=0.15, test_frac=0.20,
    # misc
    seed=0, run_dir='runs', resume=None, device=None, eval_only=False,
)

PRESETS = {
    'baseline': {},
    # arXiv 2603.11980 Sec. IV-A: ResNet18 (1024-d) → 1-layer LSTM(128) → 2-layer MLP(128),
    # sequence length 5, L2 loss on the action, lr 1e-6 (encoder) / 1e-5 (rest), 100 epochs
    'paper': dict(
        encoder='resnet18', feat_dim=1024, lstm_layers=1, lstm_hidden=128, seq_len=5,
        head='linear', head_hidden=128, loss='mse', lr=1e-5, lr_encoder=1e-6, epochs=100,
        channels='enemy,ally,depth',
    ),
    'improved': dict(
        encoder='spatial', coordconv=True, head='cls', n_bins=3, lstm_layers=1,
        channels='enemy,ally,obstacle,depth',
        w_ang=2.0, all_steps=True, lr=1e-3,
        aug_hflip=0.0, aug_jitter=1.5, aug_det_drop=0.1, ema=0.99,
    ),
}


# ── loss / eval ────────────────────────────────────────────────────────────────

def compute_loss(model, y, target, cfg):
    """y: raw head output (B, T, 2[, n_bins]);  target: (B, T, 2) normalized."""
    if not cfg.all_steps:
        y, target = y[:, -1:], target[:, -1:]
    w = target.new_tensor([1.0, cfg.w_ang])
    if cfg.head == 'cls':
        ce = F.cross_entropy(y.reshape(-1, cfg.n_bins), model.to_bins(target).reshape(-1),
                             reduction='none', label_smoothing=cfg.label_smoothing)
        per = ce.view(*target.shape)
    elif cfg.loss == 'huber':
        per = F.huber_loss(y, target, reduction='none', delta=cfg.huber_delta) / cfg.huber_delta
    else:
        per = (y - target) ** 2
    return (per * w).sum(-1).mean() / w.sum()


def AMP():
    """bf16 autocast when LSMA_AMP=1 (cuts activation memory ~2x on a shared GPU)"""
    on = os.environ.get('LSMA_AMP', '0') == '1' and torch.cuda.is_available()
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=on)


@torch.no_grad()
def predict_dataset(model, loader, device, return_vla=False):
    """Return last-step predictions and labels in robot units + enemy-visible mask.
    return_vla: also return the VLA plan alone at the last step (robot units; NaN where no plan)."""
    model.eval()
    preds, truths, vis, vla = [], [], [], []
    for batch in loader:
        seq, act = batch[0].to(device), batch[1]
        cond = batch[2].to(device) if len(batch) > 2 else None
        act_n = act.to(device) / model.act_scale
        with AMP():
            y = model(seq, act_n[:, :-1], cond)
        y = y.float() if torch.is_tensor(y) else type(y)(t.float() for t in y) if isinstance(y, (tuple, list)) else y
        preds.append((model.predict(y)[:, -1] * model.act_scale).cpu().numpy())
        truths.append(act[:, -1].numpy())
        vis.append((seq[:, -1, 0].amax(dim=(1, 2)) > 0.1).cpu().numpy())
        if cond is not None:
            v = (cond[:, -1, :2] * model.act_scale).cpu().numpy()
            v[cond[:, -1, 3].cpu().numpy() < 0.5] = np.nan
            vla.append(v)
    if not preds:
        out = (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,), bool))
        return out + (None,) if return_vla else out
    out = (np.concatenate(preds), np.concatenate(truths), np.concatenate(vis))
    if return_vla:
        return out + (np.concatenate(vla) if vla else None,)
    return out


def evaluate(model, loader, device, train_mean, robot):
    p, t, v, vla = predict_dataset(model, loader, device, return_vla=True)
    if len(p) == 0:
        return {'mse': float('nan'), 'ae': float('nan')}
    levels = robots.get(robot).levels
    m = compute_metrics(p, t, train_mean, v, levels=levels, scale=model.act_scale.cpu().numpy())
    if vla is not None:
        ok = ~np.isnan(vla[:, 0])
        if ok.any():
            mv = compute_metrics(vla[ok], t[ok], levels=levels)
            m['acc_joint_vla_only'] = mv['acc_joint']
            m['bacc_joint_vla_only'] = mv['bacc_joint']
            m['vla_coverage'] = float(ok.mean())
    return m


# ── main loop ──────────────────────────────────────────────────────────────────

def build_datasets(cfg):
    test_sessions = cfg.test_sessions or []
    vk = dict(channels=tuple(c.strip() for c in cfg.channels.split(',')), vla_tag=cfg.vla_tag or None,
              vla_period=cfg.vla_period, vla_latency=cfg.vla_latency,
              vla_latency_jitter=cfg.vla_latency_jitter, label_tag=cfg.label_tag or None, label_shift=cfg.label_shift)
    ev = dict(vla_phase=cfg.vla_eval_phase)
    if test_sessions:
        print(f'Split mode: cross-session (train={len(cfg.sessions)}, test={len(test_sessions)}); '
              f'val = last {cfg.val_frac:.0%} of each train session')
        kw = dict(seq_len=cfg.seq_len, test_frac=0.0, val_frac=cfg.val_frac, **vk)
        train_ds = HeatmapDataset(cfg.sessions, split='train', **kw)
        val_ds   = HeatmapDataset(cfg.sessions, split='val', **kw, **ev)
        test_ds  = HeatmapDataset(test_sessions, split='all', seq_len=cfg.seq_len, **vk, **ev)
    else:
        print(f'Split mode: temporal (test_frac={cfg.test_frac}, val_frac={cfg.val_frac}, block val)')
        kw = dict(seq_len=cfg.seq_len, test_frac=cfg.test_frac, val_frac=cfg.val_frac, **vk)
        train_ds = HeatmapDataset(cfg.sessions, split='train', **kw)
        val_ds   = HeatmapDataset(cfg.sessions, split='val', **kw, **ev)
        test_ds  = HeatmapDataset(cfg.sessions, split='test', **kw, **ev)
    return train_ds, val_ds, test_ds


def train(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    train_ds, val_ds, test_ds = build_datasets(cfg)
    if train_ds.robot != cfg.robot:
        raise ValueError(f'--robot {cfg.robot} but obs.npz was made for {train_ds.robot}')
    rb = robots.get(cfg.robot)
    label_stats(train_ds.window_actions, 'train', rb.action_names)
    train_mean = train_ds.window_actions.mean(0)
    cfg.act_scale = rb.act_scale.tolist()            # per-dim, robot units
    cfg.levels = [list(lv) for lv in rb.levels]
    print(f'robot = {rb.name} ({rb.platform}, {rb.units})  act_scale = {cfg.act_scale}  levels = {cfg.levels}')
    n_hm = sum(c.strip() != 'depth' for c in cfg.channels.split(','))

    pin = device.startswith('cuda')
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 2)), pin_memory=pin, drop_last=len(train_ds) > cfg.batch_size)
    val_loader  = DataLoader(val_ds,  batch_size=int(os.environ.get('LSMA_EVAL_BS', 256)), shuffle=False, num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 1)), pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=int(os.environ.get('LSMA_EVAL_BS', 256)), shuffle=False, num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 1)), pin_memory=pin)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model params = {n_params:,}')
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema), use_buffers=True) \
        if cfg.ema > 0 else None
    eval_model = lambda: ema.module if ema is not None else model

    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{'params': enc, 'lr': cfg.lr_encoder or cfg.lr}, {'params': rest, 'lr': cfg.lr}]
    opt   = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    start_epoch, best_val = 1, np.inf
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if ema is not None and ckpt.get('ema') is not None:
            ema.load_state_dict(ckpt['ema'])
        if not cfg.eval_only:
            opt.load_state_dict(ckpt['opt'])
            if ckpt.get('sched') is not None:
                sched.load_state_dict(ckpt['sched'])
            best_val = ckpt.get('best_val', np.inf)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'Resumed from {cfg.resume}  (next epoch {start_epoch}, best_val {best_val:.3f})')

    if cfg.eval_only:
        print(format_metrics(evaluate(eval_model(), val_loader, device, train_mean, cfg.robot), 'VAL   '))
        print(format_metrics(evaluate(eval_model(), test_loader, device, train_mean, cfg.robot), 'TEST  '))
        return model

    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, 'config.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2)
    log_path = os.path.join(cfg.run_dir, 'train_log.csv')
    if start_epoch == 1:
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_ae,val_mse_w,val_acc_joint,lr,elapsed_s\n')

    def save(path, epoch):
        torch.save({'model': model.state_dict(),
                    'ema': ema.state_dict() if ema is not None else None,
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': epoch, 'best_val': best_val, 'cfg': vars(cfg),
                    'train_mean': train_mean}, path)

    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            seq = batch[0].to(device, non_blocking=True)
            act = batch[1].to(device, non_blocking=True) / model.act_scale
            cond = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            w_lv = model.bin_centers[1] if cfg.head == 'cls' else \
                torch.tensor(rb.levels[1], device=device) / model.act_scale[1]
            if cond is None:
                seq, act = augment.hflip(seq, act, cfg.aug_hflip, w_levels=w_lv)
            else:
                seq, act, cond = augment.hflip(seq, act, cfg.aug_hflip, cond, w_levels=w_lv)
            seq = augment.jitter(seq, cfg.aug_jitter)
            seq = augment.det_dropout(seq, cfg.aug_det_drop, n_hm=n_hm)
            with AMP():
                y = model(seq, act[:, :-1], cond)
            y = y.float() if torch.is_tensor(y) else type(y)(t.float() for t in y) if isinstance(y, (tuple, list)) else y
            loss = compute_loss(model, y, act[:, 1:], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += loss.item() * len(seq); n += len(seq)
        sched.step()

        vm = evaluate(eval_model(), val_loader, device, train_mean, cfg.robot)
        lr_now, elapsed = sched.get_last_lr()[-1], time.time() - t0
        print(f'ep {epoch:3d}/{cfg.epochs}  loss={tot/max(n,1):.4f}  '
              f'val_ae={vm["ae"]:.4f} (zero={vm.get("ae_zero", float("nan")):.4f})  '
              f'acc v/w/joint={vm.get("acc_v", float("nan")):.2f}/{vm.get("acc_w", float("nan")):.2f}/'
              f'{vm.get("acc_joint", float("nan")):.2f}  lr={lr_now:.2e}  t={elapsed:.0f}s')
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{tot/max(n,1):.6f},{vm["ae"]:.5f},{vm.get("mse_w", float("nan")):.4f},'
                    f'{vm.get("acc_joint", float("nan")):.4f},{lr_now:.2e},{elapsed:.1f}\n')

        # model selection on the paper's Action Error (normalized MSE; units comparable across dims)
        if vm['ae'] < best_val:
            best_val = vm['ae']
            save(os.path.join(cfg.run_dir, 'best.pt'), epoch)
            print(f'  ↑ best saved  (val_ae={best_val:.4f})')
        save(os.path.join(cfg.run_dir, 'last.pt'), epoch)

    # ── final evaluation on held-out test set ─────────────────────────────
    best = torch.load(os.path.join(cfg.run_dir, 'best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(best['model'])
    if ema is not None:
        ema.load_state_dict(best['ema'])
    tm = evaluate(eval_model(), test_loader, device, train_mean, cfg.robot)
    print('\nDone.')
    print(f'  best val AE : {best_val:.4f}')
    print(format_metrics(tm, '  TEST  '))
    with open(os.path.join(cfg.run_dir, 'test_metrics.json'), 'w') as f:
        json.dump({'best_val_ae': best_val, **tm}, f, indent=2)
    return model


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--sessions', nargs='+', required=True,
                    help='train session directories (each must have obs.npz + labels.csv)')
    ap.add_argument('--test_sessions', nargs='+', default=[],
                    help='held-out sessions (cross-session split); omit for temporal split')
    ap.add_argument('--preset', choices=list(PRESETS), default='baseline',
                    help='set of defaults; explicit flags still override')
    choices = {'encoder': ENCODERS, 'head': HEADS, 'loss': ('mse', 'huber'), 'robot': tuple(robots.ROBOTS)}
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f'--{k}', action=argparse.BooleanOptionalAction, default=v)
        else:
            t = type(v) if v is not None else str
            ap.add_argument(f'--{k}', type=t, default=v, choices=choices.get(k))
    pre, _ = ap.parse_known_args()
    ap.set_defaults(**PRESETS[pre.preset])
    train(ap.parse_args())


if __name__ == '__main__':
    main()
