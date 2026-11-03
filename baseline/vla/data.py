"""Dataset for SmolVLA fine-tuning: (upright RGB frame at t, action chunk t .. t+chunk-1).

Uses the same per-session time-range splits as the heatmap policy (dataset.split_range), so a
VLA trained on split='train' never sees val/test labels. A chunk never runs past the end of its
split range (LeRobot's SmolVLA ignores action padding masks, so we avoid padding altogether).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import robots
from dataset import split_range

TASK = robots.ROBOTS['chaser'].task   # default; per-robot instruction is robots.<robot>.task


class VLAChunkDataset(Dataset):
    def __init__(self, session_dirs, split='train', chunk=16, test_frac=0.2, val_frac=0.15,
                 robot='chaser', label_source='requested', stride=1):
        rb = robots.get(robot)
        act_scale = rb.act_scale                       # per-dim, same normalisation as the policy
        lo, hi = split_range(split, test_frac, val_frac)
        self.images, self.actions, self.index = [], [], []
        for si, sdir in enumerate(session_dirs):
            img_path = os.path.join(sdir, 'images_vla.npy')
            if not os.path.exists(img_path):
                raise FileNotFoundError(f'{img_path} missing — run python -m vla.cache_images first')
            imgs = np.load(img_path, mmap_mode='r')
            act = rb.load_actions(sdir, label_source)    # (N, 2) [v, ω] robot units
            assert len(imgs) == len(act)
            N = len(act)
            a, b = int(round(N * lo)), int(round(N * hi))
            self.images.append(imgs)
            self.actions.append(act / act_scale)
            self.index += [(si, t) for t in range(a, b - chunk + 1, stride)]
        self.chunk = chunk
        print(f'VLAChunkDataset[{split}]: {len(self.index):,} samples, {len(session_dirs)} sessions')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, t = self.index[i]
        img = torch.from_numpy(np.array(self.images[si][t])).permute(2, 0, 1)  # uint8 (3,H,W)
        act = torch.from_numpy(self.actions[si][t:t + self.chunk].copy())                   # (chunk, 2)
        return img, act


def make_batch(policy, img_u8, tokens, device, state_dim=2):
    """Build a SmolVLA batch dict directly (bypassing LeRobot processors: our actions are
    already normalized to [-1, 1] and there is no proprioceptive state)."""
    B = img_u8.shape[0]
    return {
        'observation.image': img_u8.to(device, non_blocking=True).float() / 255.0,
        'observation.state': torch.zeros(B, state_dim, device=device),
        'observation.language.tokens': tokens['input_ids'].to(device).expand(B, -1),
        'observation.language.attention_mask': tokens['attention_mask'].to(device).bool().expand(B, -1),
    }
