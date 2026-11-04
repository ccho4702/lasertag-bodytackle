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
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        best = os.path.join(ckpt_dir, 'best') if os.path.isdir(os.path.join(ckpt_dir, 'best')) else ckpt_dir
        self.meta = json.load(open(os.path.join(os.path.dirname(best.rstrip('/')), 'vla_config.json')))
        self.policy = load_finetuned(best, self.device).eval()
        self.tokens = tokenize(self.policy, self.meta['task'])

    @torch.no_grad()
    def plan(self, frame):
        """frame: HxWx3 uint8 RGB as recorded → (chunk, 2) plan, normalized [-1, 1] ([v, ω] / act_scale)."""
        img = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None]   # 1×3×H×W uint8
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device == 'cuda'):
            out = self.policy.predict_action_chunk(make_batch(self.policy, img, self.tokens, self.device))
        return out[0, :, :2].float().cpu().numpy()


class AsyncVLA:
    def __init__(self, runner, period=10):
        self.runner, self.period = runner, period
        self._plan, self._src, self._busy = None, None, False
        self._lock = threading.Lock()

    def offer(self, idx, frame):
        if idx % self.period or self._busy:
            return
        self._busy = True

        def work():
            p = self.runner.plan(frame)
            with self._lock:
                self._plan, self._src = p, idx
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
        t = time.perf_counter()
        p = r.plan(f)
        if r.device == 'cuda':
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1e3)
        print(i, 'plan[0] =', p[0].round(2), f'{ms[-1]:.0f} ms')
    print(f'task: {r.meta["task"]}\nmedian latency {np.median(ms[2:]):.0f} ms (excl. 2 warm-up)')


if __name__ == '__main__':
    main()
