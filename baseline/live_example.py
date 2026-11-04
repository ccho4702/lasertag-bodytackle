"""Minimal in-process live loop (no HTTP), asynchronous like policy_server.py:
camera thread → newest frame → inference thread → newest command ← drive loop at a fixed rate.

    python live_example.py --ckpt ../checkpoints/policy/improved_cpu_chaser.pt --robot-ip 192.168.1.38

!! get_frame() must return the camera image exactly as LSMA-Laser-tag saves it
   (Misty: stream frame rotated 90° clockwise → 480×640 RGB). Fill it in with your capture code.
For the real controllers use robot_client/policy_misty.py / policy_cozmo.py + policy_server.py (HANDOFF §1-2).
"""
import argparse
import threading
import time

import numpy as np

from policy_server import Policy


def get_frame():
    """TODO: block until the next robot-camera frame and return it (RGB uint8 HxWx3, as recorded)."""
    raise NotImplementedError('plug in the same camera capture code the recorder uses')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--robot-ip', required=True, help='Misty IP')
    ap.add_argument('--device', default=None)
    ap.add_argument('--vote', type=int, default=1)
    ap.add_argument('--hz', type=float, default=10, help='drive command rate (independent of inference)')
    ap.add_argument('--max_s', type=float, default=60)
    a = ap.parse_args()
    import requests
    pol = Policy(a.ckpt, device=a.device, vote=a.vote)
    assert pol.robot.platform == 'misty', 'Cozmo: use robot_client/policy_cozmo.py'
    run = True

    def camera():
        while run:
            pol.put(np.asarray(get_frame()))
    threading.Thread(target=camera, daemon=True).start()
    http = requests.Session()
    t0 = time.time()
    try:
        while time.time() - t0 < a.max_s:
            c = pol.latest()
            if c['v'] == 0 and c['w'] == 0:
                http.post(f'http://{a.robot_ip}/api/drive/stop', json={'Hold': False}, timeout=0.5)
            else:   # timed drive like wasd_misty: keeps moving between updates, stops by itself if we die
                http.post(f'http://{a.robot_ip}/api/drive/time', json={'LinearVelocity': int(c['v']),
                          'AngularVelocity': int(c['w']), 'TimeMs': 500}, timeout=0.5)
            print(f"{time.time() - t0:6.1f}s {c['name']:14s} infer {c.get('infer_ms')} ms  frame age {c['frame_age_ms']} ms")
            time.sleep(1 / a.hz)
    finally:
        run = False
        http.post(f'http://{a.robot_ip}/api/drive/stop', json={'Hold': False}, timeout=1)


if __name__ == '__main__':
    main()
