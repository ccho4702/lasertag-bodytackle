"""Online inference: camera frame in → robot command out (same perception code as training).

    from deploy import Agent
    agent = Agent('runs/cv/paper_chaser/f0_s0/best.pt')       # robot comes from the checkpoint
    cmd = agent.step(rgb_uint8_hwc)                            # every camera frame
    # chaser (Misty):  requests.post(f'http://{ip}/api/drive', json=cmd)
    #                  cmd = {'LinearVelocity': %, 'AngularVelocity': %}
    # evader (Cozmo):  robot.drive_wheels(**cmd)
    #                  cmd = {'l_wheel_speed': mm/s, 'r_wheel_speed': mm/s}

The agent keeps the last seq_len observations (and previous actions / VLA conditioning when the
model uses them), runs the policy on that window and converts the last-step output with
robots.Robot.to_command. With a VLA (`vla_tag` models), pass the newest plan chunk and its age:
    agent.step(frame, vla_plan=plan_chunk_normalized, vla_age_frames=k)
Quick self-test on a recorded session:
    python deploy.py --ckpt runs/.../best.pt --session ../data_download/evader/misty_stop_obs1/<ts>
"""
import argparse
import collections
import os

import numpy as np
import torch

import robots
from model import build_model
from preprocess import Perception


class Agent:
    def __init__(self, ckpt, device=None, snap=False, use_ema=True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.cfg = cfg = ck['cfg']
        self.model = build_model({**cfg, 'pretrained': False}).to(self.device).eval()
        state = ck['model']
        if use_ema and ck.get('ema') is not None:
            state = {k[len('module.'):]: v for k, v in ck['ema'].items() if k.startswith('module.')}
        self.model.load_state_dict(state)
        self.robot = robots.get(cfg['robot'])
        self.snap = snap
        need_depth = 'depth' in self.model.channels
        self.perc = Perception(self.robot, depth=need_depth, device=self.device)
        self.idx = [self.perc.channels.index(c) for c in self.model.channels]
        T = cfg['seq_len']
        self.obs = collections.deque(maxlen=T)
        self.prev = collections.deque([np.zeros(2, np.float32)] * T, maxlen=T)
        self.cond = collections.deque([np.zeros(4, np.float32)] * T, maxlen=T)

    def reset(self):
        self.obs.clear()

    @torch.no_grad()
    def step(self, frame, vla_plan=None, vla_age_frames=None):
        """frame: HxWx3 uint8 RGB from the robot camera (as recorded: no rotation/flip/crop).
        Returns (command kwargs for the robot API, raw [v, ω] in robot units)."""
        o, _ = self.perc([frame])
        o = o[0, self.idx].astype(np.float32)
        if not self.obs:                                   # pad the history with the first frame
            self.obs.extend([o] * self.obs.maxlen)
        self.obs.append(o)
        c = np.zeros(4, np.float32)
        if vla_plan is not None and self.model.vla_cond:
            k = int(vla_age_frames or 0)
            plan = np.asarray(vla_plan, np.float32)
            c[:2] = plan[min(k, len(plan) - 1)]
            c[2] = k / (self.cfg['vla_latency'] + self.cfg['vla_period'])
            c[3] = 1.0
        self.cond.append(c)
        a = self.model.act(np.stack(self.obs), np.stack(self.prev) if self.model.prev_action else None,
                           np.stack(self.cond) if self.model.vla_cond else None)
        self.prev.append(a.astype(np.float32))
        return self.robot.to_command(a, snap=self.snap), a


def main():
    import pandas as pd
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--session', required=True)
    ap.add_argument('--n', type=int, default=20)
    a = ap.parse_args()
    agent = Agent(a.ckpt)
    df = pd.read_csv(os.path.join(a.session, 'labels.csv'))
    gt = agent.robot.load_actions(a.session)
    import time
    ms = []
    for i in range(min(a.n, len(df))):
        frame = np.array(Image.open(os.path.join(a.session, df.image[i])).convert('RGB'))
        t0 = time.perf_counter()
        cmd, raw = agent.step(frame)
        if agent.device == 'cuda':
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
        print(i, {k: round(v, 1) for k, v in cmd.items()}, 'raw', raw.round(1), 'label', gt[i], f'{ms[-1]:.0f} ms')
    if len(ms) > 3:
        print(f'step latency (excluding the first 3 warm-up frames): median {np.median(ms[3:]):.0f} ms '
              f'(budget ≈ {1000 / agent.robot.fps:.0f} ms at {agent.robot.fps} fps)')


if __name__ == '__main__':
    main()
