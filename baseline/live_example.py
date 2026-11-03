"""Minimal live loop: robot camera frame → policy → robot command (closed-loop test on the real robot).

    python live_example.py --ckpt ../checkpoints/policy/improved_chaser.pt --robot-ip 192.168.0.10
    python live_example.py --ckpt ../checkpoints/policy/improved_evader.pt          # Cozmo (SDK)

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
    a = ap.parse_args()
    agent = Agent(a.ckpt, snap=a.snap)
    print('robot:', agent.robot.name, 'fps budget', round(1000 / agent.robot.fps), 'ms')
    if agent.robot.platform == 'misty':
        send = misty_sender(a.robot_ip)
        stop = {'LinearVelocity': 0, 'AngularVelocity': 0}
    else:
        import cozmo  # noqa: F401  (run the loop inside cozmo.run_program in your recorder instead)
        raise SystemExit('Cozmo: call agent.step(frame) inside your cozmo.run_program loop and '
                         'robot.drive_wheels(**cmd)')
    t0 = time.time()
    try:
        while time.time() - t0 < a.max_s:
            ts = time.perf_counter()
            cmd, raw = agent.step(get_frame())
            send(cmd)
            print(f'{time.time() - t0:6.1f}s  {cmd}  ({(time.perf_counter() - ts) * 1e3:.0f} ms)')
    finally:
        send(stop)


if __name__ == '__main__':
    main()
