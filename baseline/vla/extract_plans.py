"""Run a fine-tuned SmolVLA on every frame of each session and store its action chunks.

    python -m vla.extract_plans --ckpt runs_vla/smolvla_c16/best --tag smolvla_c16 --sessions ../data/sample/*/

Writes <session>/vla_plans_<tag>.npz with
    plans      (N, chunk, 2) float32, normalized [-1, 1]: chunk predicted from frame t
    latency_ms measured batch-1 inference latency on this machine

Plans are computed for *every* frame so the heatmap policy can simulate any query period,
phase and latency at training time without re-running the VLA.
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import robots
from vla.data import prev_actions
from vla.policies import load_finetuned, make_batch, tokenize


class _Frames(Dataset):
    def __init__(self, path, hist=0, hist_stride=4, actions=None, state_hist=0):
        self.mm = np.load(path, mmap_mode='r')
        self.hist, self.hist_stride = hist, hist_stride
        self.actions, self.state_hist = actions, state_hist     # normalized commands, for the previous-command state

    def __len__(self):
        return len(self.mm)

    def __getitem__(self, i):
        img = self._img(i)
        return (img, prev_actions(self.actions, i, self.state_hist)) if self.state_hist else img

    def _img(self, i):
        if self.hist:      # same layout as VLAChunkDataset: (hist+1, 3, H, W), oldest first
            ts = [max(0, i - k * self.hist_stride) for k in range(self.hist, -1, -1)]
            return torch.from_numpy(np.stack([self.mm[u] for u in ts])).permute(0, 3, 1, 2)
        return torch.from_numpy(np.array(self.mm[i])).permute(2, 0, 1)


@torch.no_grad()
def measure_latency(policy, tokens, device, img, n=10, state=None):
    b = make_batch(policy, img[:1], tokens, device, state=None if state is None else state[:1])
    ts = []
    for _ in range(n + 2):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            policy.predict_action_chunk(b)
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    return 1000 * float(np.median(ts[2:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    device = 'cuda'
    policy = load_finetuned(a.ckpt, device).eval()
    import json
    meta = json.load(open(os.path.join(os.path.dirname(a.ckpt.rstrip('/')), 'vla_config.json')))
    tokens = tokenize(policy, meta['task'])          # same instruction as in training
    g = torch.Generator(device=device).manual_seed(a.seed)
    lat = None
    sh = meta.get('state_hist', 0)
    rb = robots.get(meta.get('robot', 'chaser'))
    for s in a.sessions:
        acts = rb.load_actions(s, meta.get('label_source', 'requested')) / rb.act_scale if sh else None
        loader = DataLoader(_Frames(os.path.join(s, 'images_vla.npy'), meta.get('hist', 0), meta.get('hist_stride', 4), acts, sh), batch_size=a.batch_size,
                            num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 2)))
        out = []
        for img in loader:
            img, st = img if sh else (img, None)
            if lat is None:
                lat = measure_latency(policy, tokens, device, img, state=st)
                print(f'batch-1 latency ≈ {lat:.0f} ms')
            noise = torch.randn(img.shape[0], policy.config.chunk_size, policy.config.max_action_dim,
                                device=device, generator=g)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(policy.predict_action_chunk(make_batch(policy, img, tokens, device, state=st),
                                                       noise=noise).float().cpu().numpy())
        plans = np.concatenate(out).astype(np.float32)
        path = os.path.join(s, f'vla_plans_{a.tag}.npz')
        np.savez_compressed(path, plans=plans, latency_ms=lat, ckpt=a.ckpt)
        print(f'{path}: plans {plans.shape}')


if __name__ == '__main__':
    main()
