"""Live SmolVLA helper: query the VLA every `period` frames in a background thread and hand the
newest plan + its age to deploy.Agent (TASK.md: VLA is an auxiliary signal, not called every frame).

    from deploy import Agent
    from vla.runner import VLARunner, AsyncVLA
    agent = Agent('../checkpoints/policy/improved_ram_chaser.pt')       # a VLA-conditioned policy
    vla = AsyncVLA(VLARunner('../checkpoints/smolvla/chaser'), period=agent.cfg['vla_period'])
    for i, frame in enumerate(frames):
        vla.offer(i, frame)                                 # starts a query every `period` frames (non-blocking)
        plan, age = vla.latest(i)                           # newest finished plan, frames since its input frame
        cmd, raw = agent.step(frame, vla_plan=plan, vla_age_frames=age)

Self-test (latency):  python -m vla.runner --ckpt ../checkpoints/smolvla/chaser --session <session dir>
"""
import argparse
import collections
import json
import os
import threading
import time

import numpy as np
import torch

from vla.policies import load_finetuned, make_batch, tokenize


class VLARunner:
    def __init__(self, ckpt_dir, device=None):
        """ckpt_dir: folder with `best/` (lerobot policy) and `vla_config.json` (task prompt, chunk)."""
        # Apple GPU when there is no CUDA: one plan takes ~0.28 s on mps vs ~8 s on CPU (M5 MacBook, measured).
        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'mps' if torch.backends.mps.is_available() else 'cpu')
        best = os.path.join(ckpt_dir, 'best') if os.path.isdir(os.path.join(ckpt_dir, 'best')) else ckpt_dir
        self.meta = json.load(open(os.path.join(os.path.dirname(best.rstrip('/')), 'vla_config.json')))
        self.policy = load_finetuned(best, self.device).eval()
        self.default_task = self.task = self.meta['task']       # the instruction SmolVLA was fine-tuned with
        # hist > 0: the model also takes `hist` past frames, hist_stride frames apart, as extra camera images
        # (same layout as training, vla/data.py: oldest first, clamped to the first frame of the episode).
        self.hist, self.hist_stride = int(self.meta.get('hist', 0)), int(self.meta.get('hist_stride', 4))
        assert not self.meta.get('state_hist'), 'live runner does not pass previous commands (state_hist)'
        self.tokens = tokenize(self.policy, self.task)

    def set_task(self, task=None):
        """Change the instruction given to SmolVLA at run time (None or '' = back to the trained one).
        The model was fine-tuned with one fixed instruction, so any other text is outside its training."""
        self.task = (task or '').strip() or self.default_task
        self.tokens = tokenize(self.policy, self.task)
        return self.task

    @torch.no_grad()
    def plan(self, frame):
        """frame: HxWx3 uint8 RGB as recorded → (chunk, 2) plan, normalized [-1, 1] ([v, ω] / act_scale).
        For a model with past frames (hist > 0) pass the list [oldest, ..., current] of hist + 1 frames;
        a single frame is repeated (= an episode that has just started)."""
        if self.hist:
            frames = list(frame) if isinstance(frame, (list, tuple)) else [frame] * (self.hist + 1)
            assert len(frames) == self.hist + 1, f'need {self.hist + 1} frames, got {len(frames)}'
            img = torch.from_numpy(np.stack([np.ascontiguousarray(f) for f in frames])).permute(0, 3, 1, 2)[None]  # 1×(hist+1)×3×H×W
        else:
            img = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None]   # 1×3×H×W uint8
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device == 'cuda'):
            out = self.policy.predict_action_chunk(make_batch(self.policy, img, self.tokens, self.device))
        return out[0, :, :2].float().cpu().numpy()


class AsyncVLA:
    def __init__(self, runner, period=10):
        self.runner, self.period = runner, period
        self._plan, self._src, self._busy = None, None, False
        self._lock = threading.Lock()
        # every offered frame is kept as far back as the model looks (hist * hist_stride frames)
        self.hist, self.stride = getattr(runner, 'hist', 0), getattr(runner, 'hist_stride', 4)
        self._buf = collections.deque(maxlen=self.hist * self.stride + 1)
        self._gen = 0

    def reset(self):
        """New episode: forget the plan and the past frames."""
        with self._lock:
            self._plan = self._src = None
            self._gen += 1          # a plan still being computed belongs to the old episode: drop it when it arrives
        self._buf.clear()

    def offer(self, idx, frame):
        if self.hist:
            self._buf.append(frame)
        if idx % self.period or self._busy:
            return
        self._busy = True
        if self.hist:      # [t - hist*stride, ..., t - stride, t], clamped to the oldest frame of this episode
            buf = list(self._buf)
            frame = [buf[max(0, len(buf) - 1 - k * self.stride)] for k in range(self.hist, -1, -1)]

        gen = self._gen

        def work():
            try:
                p = self.runner.plan(frame)
                with self._lock:
                    if gen == self._gen:     # else: reset() ran meanwhile (its frame index would give a negative age)
                        self._plan, self._src = p, idx
            finally:
                self._busy = False
        threading.Thread(target=work, daemon=True).start()

    def latest(self, idx):
        with self._lock:
            return (None, None) if self._plan is None else (self._plan, idx - self._src)


def main():
    import pandas as pd
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--session', required=True)
    ap.add_argument('--n', type=int, default=10)
    a = ap.parse_args()
    r = VLARunner(a.ckpt)
    df = pd.read_csv(os.path.join(a.session, 'labels.csv'))
    ms = []
    for i in range(min(a.n, len(df))):
        f = np.array(Image.open(os.path.join(a.session, df.image[i])).convert('RGB'))
        if r.hist:         # past frames from the same session, as in training
            f = [np.array(Image.open(os.path.join(a.session, df.image[max(0, i - k * r.hist_stride)])).convert('RGB'))
                 for k in range(r.hist, 0, -1)] + [f]
        t = time.perf_counter()
        p = r.plan(f)
        if r.device == 'cuda':
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1e3)
        print(i, 'plan[0] =', p[0].round(2), f'{ms[-1]:.0f} ms')
    print(f'task: {r.meta["task"]}\nmedian latency {np.median(ms[2:]):.0f} ms (excl. 2 warm-up)')


if __name__ == '__main__':
    main()
