"""Dataset: sliding-window sequences of observation tensors + velocity labels.

Reads <session>/obs.npz written by preprocess.py (paper input: target heatmaps + depth).
Each sample is (seq, act_seq) where:
    seq     : (T, C, H, W)  float32 — the selected `channels` (by name, e.g. enemy, ally, depth),
              T consecutive frames [t-T+1 .. t]; 'enemy' must come first (used for metrics)
    act_seq : (T+1, 2)      float32 — actions for frames [t-T .. t]
              act_seq[1:]  are the per-step targets (act_seq[-1] is the label of the window)
              act_seq[:-1] are the previous actions (used only with --prev_action)
    Actions are the robot's body velocity [v, ω] in robot units (robots.py: % for Misty,
    mm/s for Cozmo).

Windows never cross session boundaries or split boundaries: every frame a window touches
(including the extra "previous action" frame) lies inside the selected range of its session.

Splits (per session, as fractions of its frames)
    all   : [0, 1)
    train : [0, (1-test_frac)(1-val_frac))
    val   : [(1-test_frac)(1-val_frac), 1-test_frac)     ← contiguous time block, no overlap with train
    test  : [1-test_frac, 1)
    trainval : [0, 1-test_frac)
For cross-session evaluation use test_frac=0 so train/val cover the whole session.

VLA conditioning (vla_tag set): a third item cond (T, 4) is returned, simulating a VLA that is
queried every `vla_period` frames (at frames phase + m·period of the session) and whose answer
arrives `vla_latency` frames later. At frame l the newest available plan was computed at
t0 = max{phase + m·period : t0 + latency <= l}; the conditioning is
    [plan_v, plan_w, age, has_plan]   plan = plans[t0, l - t0] (chunk index clamped), age = (l-t0)/(latency+period)
Plans come from <session>/vla_plans_<tag>.npz (vla/extract_plans.py), normalized to [-1, 1].
vla_phase=None draws a random phase per sample (training); an int fixes it (evaluation).
In training mode (vla_phase=None) the latency is also drawn per sample from
[latency - jitter, latency + jitter] (real VLA latency varies; median ~342 ms ≈ 3.4 frames on an RTX 3090).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAPER_CHANNELS = ('enemy', 'ally', 'depth')


def split_range(split, test_frac, val_frac):
    tv_end = 1.0 - test_frac
    tr_end = tv_end * (1.0 - val_frac)
    return {
        'all':      (0.0, 1.0),
        'trainval': (0.0, tv_end),
        'train':    (0.0, tr_end),
        'val':      (tr_end, tv_end),
        'test':     (tv_end, 1.0),
    }[split]


class HeatmapDataset(Dataset):
    """Load one or more session directories and select a contiguous time range of each."""

    def __init__(self, session_dirs, seq_len=8, split='all', test_frac=0.2, val_frac=0.15,
                 channels=PAPER_CHANNELS, vla_tag=None, vla_period=10, vla_latency=4, vla_phase=None,
                 vla_latency_jitter=0, verbose=True, label_tag=None, label_shift=0):
        super().__init__()
        lo, hi = split_range(split, test_frac, val_frac)
        hm_list, act_list, idx_list = [], [], []
        sid_list, local_list, self._plans = [], [], []
        offset = 0
        self.vla_tag, self.vla_period, self.vla_latency, self.vla_phase = \
            vla_tag, vla_period, vla_latency, vla_phase
        self.vla_latency_jitter = vla_latency_jitter

        channels = tuple(channels)
        assert channels[0] == 'enemy', 'first channel must be enemy (enemy-visible metrics)'
        self.channels, self.robot = channels, None
        for sdir in session_dirs:
            path = os.path.join(sdir, 'obs.npz')
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} not found — run preprocess.py --robot ... first.')
            npz = np.load(path)
            have = [str(c) for c in npz['channels']]
            miss = [c for c in channels if c not in have]
            if miss:
                raise ValueError(f'{path} lacks channels {miss} (has {have})')
            robot = str(npz['robot'])
            if self.robot is None:
                self.robot = robot
            assert robot == self.robot, f'mixed robots in one dataset: {robot} vs {self.robot}'
            obs = npz['obs']
            heatmaps = obs[:, [have.index(c) for c in channels]]          # (N, C, H, W) float16
            actions_all = npz['actions'].astype(np.float32)
            if label_tag:   # alternative labels, e.g. actions_teacher.npy from overhead/teacher.py
                actions_all = np.load(os.path.join(sdir, f'actions_{label_tag}.npy')).astype(np.float32)
            if label_shift:  # frame t gets the command issued at t+label_shift (<0: camera frames arrive late)
                actions_all = actions_all[np.clip(np.arange(len(actions_all)) + label_shift, 0, len(actions_all) - 1)]
            n_rows = len(pd.read_csv(os.path.join(sdir, 'labels.csv')))
            assert len(heatmaps) == len(actions_all) == n_rows, f'row mismatch in {sdir}'

            N = len(heatmaps)
            a, b = int(round(N * lo)), int(round(N * hi))
            heatmaps = heatmaps[a:b]
            actions = actions_all[a:b]

            # window ends at e; needs frames e-seq_len .. e (seq_len+1 frames incl. prev action)
            ends = np.arange(seq_len, len(heatmaps), dtype=np.int64) + offset
            hm_list.append(heatmaps)
            act_list.append(actions)
            idx_list.append(ends)
            sid_list.append(np.full(len(heatmaps), len(sid_list), np.int64))
            local_list.append(np.arange(a, b, dtype=np.int64))
            offset += len(heatmaps)
            if vla_tag:
                pp = os.path.join(sdir, f'vla_plans_{vla_tag}.npz')
                if not os.path.exists(pp):
                    raise FileNotFoundError(f'{pp} missing — run python -m vla.extract_plans')
                plans = np.load(pp)['plans']
                assert len(plans) == N, f'{pp}: {len(plans)} plans vs {N} frames'
                self._plans.append(plans.astype(np.float32))

        self._hm  = np.concatenate(hm_list,  axis=0) if hm_list else np.zeros((0, len(channels), 224, 224), np.float16)
        self._act = np.concatenate(act_list, axis=0) if act_list else np.zeros((0, 2), np.float32)
        self._idx = np.concatenate(idx_list, axis=0) if idx_list else np.zeros((0,), np.int64)
        self._sid = np.concatenate(sid_list) if sid_list else np.zeros((0,), np.int64)
        self._local = np.concatenate(local_list) if local_list else np.zeros((0,), np.int64)
        self._seq_len = seq_len
        if verbose:
            print(f'Dataset [{split}]: {len(self._idx):,} windows  '
                  f'({len(self._hm):,} frames, {len(session_dirs)} sessions)')

    def __len__(self):
        return len(self._idx)

    @property
    def window_actions(self):
        """(N, 2) label (last-step action, robot units) of every window."""
        return self._act[self._idx]

    @property
    def frame_actions(self):
        return self._act

    def __getitem__(self, i):
        end = self._idx[i]
        T = self._seq_len
        seq = self._hm[end - T + 1: end + 1].astype(np.float32)    # (T, C, H, W)
        act = self._act[end - T: end + 1].copy()       # (T+1, 2)
        if not self.vla_tag:
            return torch.from_numpy(seq), torch.from_numpy(act)
        return torch.from_numpy(seq), torch.from_numpy(act), torch.from_numpy(self.vla_cond(end))

    def vla_cond(self, end):
        T, P, L = self._seq_len, self.vla_period, self.vla_latency
        age_norm = L + P                      # nominal, so the age feature keeps one scale
        plans = self._plans[self._sid[end]]
        H = plans.shape[1]
        training = self.vla_phase is None
        phase = np.random.randint(P) if training else self.vla_phase % P
        jit = self.vla_latency_jitter
        if training and jit > 0:
            L = max(0, L + np.random.randint(-jit, jit + 1))
        cond = np.zeros((T, 4), np.float32)
        for j, l in enumerate(self._local[end - T + 1: end + 1]):
            m = (l - L - phase) // P
            if m < 0:
                continue                      # no VLA answer has arrived yet in this session
            t0 = phase + m * P
            k = l - t0                        # in [L, L+P)
            cond[j, :2] = plans[t0, min(k, H - 1)]
            cond[j, 2] = k / age_norm
            cond[j, 3] = 1.0
        return cond
