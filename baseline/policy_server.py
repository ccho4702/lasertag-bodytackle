"""Policy server for the LSMA-Laser-tag controllers (github.com/gyeomhb/LSMA-Laser-tag).

The robot PC runs Windows + Python 3.8 (Cozmo SDK) without torch, so the policy runs here (GPU box or
the same PC in a separate Python ≥3.10 env) and the controllers talk to it over HTTP
(robot_client/policy_client.py, stdlib only).

    python policy_server.py --ckpt ../checkpoints/policy/improved_chaser_full.pt --port 8770
    python policy_server.py --ckpt ../checkpoints/policy/improved_evader_full.pt --port 8771
    # + periodic SmolVLA (VLA-conditioned policy only):
    python policy_server.py --ckpt ../checkpoints/policy/improved_ram_chaser.pt --vla ../checkpoints/smolvla/chaser

POST /step   body = raw RGB bytes (H*W*3, uint8), headers X-Width / X-Height
             — the frame exactly as LSMA-Laser-tag saves it (Misty: stream frame rotated 90° cw → 480×640,
               Cozmo: robot.world.latest_image.raw_image 320×240)
             → {"v": .., "w": .., "name": "forward_left", "api": {...}, "step": n, "ms": ..}
               Misty : v = LinearVelocity %, w = AngularVelocity % (send with POST /api/drive/time)
               Cozmo : api = {"l_wheel_speed": .., "r_wheel_speed": ..} for robot.drive_wheel_motors(**api)
POST /reset  start a new episode (clears the LSTM history and the VLA plan)
GET  /info   robot, command levels, fps budget
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from deploy import Agent


def command_name(v, w):
    """same names as LSMA-Laser-tag (wasd_misty.command_name / dataset_recorder.command_name)"""
    parts = []
    if v:
        parts.append('forward' if v > 0 else 'backward')
    if w:
        parts.append('left' if w > 0 else 'right')
    return '_'.join(parts) or 'stop'


class Policy:
    def __init__(self, ckpt, vla=None, snap=True, device=None):
        self.agent = Agent(ckpt, snap=snap, device=device)
        self.lock = threading.Lock()
        self.vla = None
        if vla:
            from vla.runner import AsyncVLA, VLARunner
            assert self.agent.model.vla_cond, 'policy was trained without VLA conditioning'
            self.vla = AsyncVLA(VLARunner(vla), period=self.agent.cfg['vla_period'])
        self.n = 0

    def reset(self):
        with self.lock:
            self.agent.reset()
            self.agent.prev.extend([np.zeros(2, np.float32)] * self.agent.prev.maxlen)
            self.agent.cond.extend([np.zeros(4, np.float32)] * self.agent.cond.maxlen)
            if self.vla:
                self.vla._plan = self.vla._src = None
            self.n = 0

    def step(self, frame):
        with self.lock:
            t0 = time.perf_counter()
            plan = age = None
            if self.vla:
                self.vla.offer(self.n, frame)
                plan, age = self.vla.latest(self.n)
            api, raw = self.agent.step(frame, vla_plan=plan, vla_age_frames=age)
            if self.agent.robot.platform == 'misty':     # misty API takes ints in %
                api = {k: int(round(v)) for k, v in api.items()}
            v, w = (float(x) for x in self.agent.robot.snap(raw))
            self.n += 1
            return {'v': v, 'w': w, 'name': command_name(v, w), 'api': api, 'raw': [float(x) for x in raw],
                    'step': self.n, 'vla_age': age, 'ms': round((time.perf_counter() - t0) * 1e3, 1)}


def make_handler(policy):
    class H(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith('/info'):
                r = policy.agent.robot
                self._json({'robot': r.name, 'platform': r.platform, 'levels': r.levels, 'fps': r.fps,
                            'budget_ms': round(1000 / r.fps), 'vla': policy.vla is not None,
                            'seq_len': policy.agent.cfg['seq_len']})
            else:
                self._json({'error': 'unknown path'}, 404)

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(n) if n else b''
            try:
                if self.path.startswith('/reset'):
                    policy.reset()
                    self._json({'ok': True})
                elif self.path.startswith('/step'):
                    w, h = int(self.headers['X-Width']), int(self.headers['X-Height'])
                    frame = np.frombuffer(body, np.uint8).reshape(h, w, 3)
                    self._json(policy.step(frame))
                else:
                    self._json({'error': 'unknown path'}, 404)
            except Exception as e:                       # the client stops the robot on any error
                self._json({'error': f'{type(e).__name__}: {e}'}, 500)

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--vla', default=None)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--device', default=None, help='cuda | mps | cpu (default: best available)')
    ap.add_argument('--no_snap', action='store_true', help='return the raw regression output (paper preset)')
    a = ap.parse_args()
    pol = Policy(a.ckpt, a.vla, snap=not a.no_snap, device=a.device)
    r = pol.agent.robot
    # warm-up (CUDA kernels, YOLO, depth) so the first real frame is not slow
    shape = (640, 480, 3) if r.platform == 'misty' else (240, 320, 3)
    for _ in range(3):
        pol.step(np.zeros(shape, np.uint8))
    pol.reset()
    print(f'policy server: {r.name} ({r.platform}, {pol.agent.device}, channels {pol.agent.model.channels}) on http://{a.host}:{a.port}  ckpt={a.ckpt}  vla={a.vla}', flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(pol)).serve_forever()


if __name__ == '__main__':
    main()
