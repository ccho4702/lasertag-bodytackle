"""Minimal live loop: robot camera frame → policy → robot command (closed-loop test on the real robot).

    python live_example.py --ckpt ../checkpoints/policy/improved_chaser.pt --robot-ip 192.168.0.10
    python live_example.py --ckpt ../checkpoints/policy/improved_evader.pt          # Cozmo (SDK)
    # with the periodic SmolVLA (VLA-conditioned policy + matching robot VLA; scripts/download_smolvla.sh):
    python live_example.py --ckpt ../checkpoints/policy/improved_ram_chaser.pt --vla ../checkpoints/smolvla/chaser --robot-ip ...

!! get_frame() must return the camera image exactly as the recorder stored it
   (same resolution/orientation, RGB uint8 HxWx3) — fill it in with your recording code.
Stop with Ctrl-C (sends a stop command).
"""
import argparse
import time

import numpy as np

from deploy import Agent


def misty_sender(ip):
    import requests

    def send(cmd):
        requests.post(f'http://{ip}/api/drive', json=cmd, timeout=0.5)
    return send


def get_frame():
    """TODO: return the newest robot-camera frame (RGB uint8, as recorded in data_download/*/images)."""
    raise NotImplementedError('plug in the same camera capture code the recorder uses')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--robot-ip', help='Misty IP (chaser)')
    ap.add_argument('--snap', action='store_true', help='snap outputs to the demonstrated command levels')
    ap.add_argument('--max_s', type=float, default=60)
    ap.add_argument('--vla', help='SmolVLA folder (checkpoints/smolvla/<robot>); needs a VLA-conditioned policy')
    a = ap.parse_args()
    agent = Agent(a.ckpt, snap=a.snap)
    vla = None
    if a.vla:
        from vla.runner import AsyncVLA, VLARunner
        assert agent.model.vla_cond, 'this policy was trained without VLA conditioning (use improved_ram_chaser / improved_vla_evader)'
        vla = AsyncVLA(VLARunner(a.vla), period=agent.cfg['vla_period'])   # queried every Δ frames in a thread
    print('robot:', agent.robot.name, 'fps budget', round(1000 / agent.robot.fps), 'ms')
    if agent.robot.platform == 'misty':
        send = misty_sender(a.robot_ip)
        stop = {'LinearVelocity': 0, 'AngularVelocity': 0}
    else:
        import cozmo  # noqa: F401  (run the loop inside cozmo.run_program in your recorder instead)
        raise SystemExit('Cozmo: call agent.step(frame) inside your cozmo.run_program loop and '
                         'robot.drive_wheels(**cmd)')
    t0, i = time.time(), 0
    try:
        while time.time() - t0 < a.max_s:
            ts = time.perf_counter()
            frame = get_frame()
            plan = age = None
            if vla:
                vla.offer(i, frame)
                plan, age = vla.latest(i)
            cmd, raw = agent.step(frame, vla_plan=plan, vla_age_frames=age)
            i += 1
            send(cmd)
            print(f'{time.time() - t0:6.1f}s  {cmd}  ({(time.perf_counter() - ts) * 1e3:.0f} ms)')
    finally:
        send(stop)


if __name__ == '__main__':
    main()
