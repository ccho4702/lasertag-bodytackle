"""Policy server for the LSMA-Laser-tag controllers (github.com/gyeomhb/LSMA-Laser-tag).

The robot PC runs Windows + Python 3.8 (Cozmo SDK) without torch, so the policy runs here (GPU box, MacBook,
or the same PC in a separate Python ≥3.10 env) and the controllers talk to it over HTTP
(robot_client/policy_client.py, stdlib only).

    python policy_server.py --ckpt ../checkpoints/policy/improved_cpu_chaser.pt --port 8770      # Misty
    python policy_server.py --ckpt ../checkpoints/policy/improved_cpu_evader.pt --port 8771      # Cozmo
    # + periodic SmolVLA (VLA-conditioned policy only):
    python policy_server.py --ckpt ../checkpoints/policy/improved_ram_chaser.pt --vla ../checkpoints/smolvla/chaser

Asynchronous by design — nothing on the robot side ever waits for inference:
    camera ──POST /frame──▶ [latest-frame slot] ──▶ inference thread (always on the newest frame,
          ◀── latest command ──────────────────────────  skips frames it cannot keep up with)
    robot drive loop re-sends the latest command at its own fixed rate.

POST /frame  body = raw RGB bytes (H*W*3 uint8), headers X-Width / X-Height — the frame exactly as
             LSMA-Laser-tag saves it (Misty: stream frame rotated 90° cw → 480×640, Cozmo: raw_image 320×240).
             Returns immediately with the latest command (not the one for this frame).
GET  /command  latest command
     command = {"v", "w", "name", "api", "seq", "frame_age_ms", "infer_ms", "fps", "stale"}
               Misty : v = LinearVelocity %, w = AngularVelocity % (send with POST /api/drive/time)
               Cozmo : api = {"l_wheel_speed", "r_wheel_speed"} for robot.drive_wheel_motors(**api)
               stale = true → no new camera frame for --stale_ms: the command is forced to stop
               frame_age_ms = age of the frame the command was computed from (≈ inference latency)
POST /step   synchronous variant (waits for the result for this frame) — for tests
POST /reset  new episode (clears LSTM history, smoothing, VLA plan)
GET  /info   robot, levels, budget
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: run unsupported ops on CPU instead of failing
import argparse
import collections
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
    def __init__(self, ckpt, vla=None, snap=True, device=None, vote=1, stale_ms=1000):
        self.agent = Agent(ckpt, snap=False, device=device)
        self.snap, self.vote, self.stale_ms = snap, vote, stale_ms
        self.robot = self.agent.robot
        self.lock = threading.Lock()          # guards agent state
        self.vla = None
        if self.agent.model.vla_cond and not vla:
            print('WARNING: this is an "ours" (VLA-conditioned) policy but --vla was not given — it will run '
                  'without plans (has_plan = 0). Pass --vla ../checkpoints/smolvla/<robot>.', flush=True)
        if vla:
            from vla.runner import AsyncVLA, VLARunner
            assert self.agent.model.vla_cond, 'policy was trained without VLA conditioning'
            self.vla = AsyncVLA(VLARunner(vla), period=self.agent.cfg['vla_period'])
        self.n = 0
        self.recent = collections.deque(maxlen=max(1, vote))
        # latest-frame slot + latest command
        self.cv = threading.Condition()
        self.frame, self.frame_t, self.frame_seq = None, 0.0, 0
        self.done_seq = 0
        self.cmd = self._make(0.0, 0.0, seq=0, frame_t=time.time(), infer_ms=0.0)
        self.times = collections.deque(maxlen=20)
        threading.Thread(target=self._worker, daemon=True).start()

    # ── command bookkeeping ───────────────────────────────────────────────────
    def _make(self, v, w, **extra):
        a = np.array([v, w], np.float32)
        api = self.robot.to_command(a, snap=False)
        if self.robot.platform == 'misty':
            api = {k: int(round(x)) for k, x in api.items()}
        return {'v': float(v), 'w': float(w), 'name': command_name(v, w), 'api': api, **extra}

    def _decide(self, raw):
        """snap to demonstrated levels; optional majority vote over the last `vote` inferences
        (hysteresis against one-frame flicker)."""
        a = self.robot.snap(raw) if self.snap else np.asarray(raw, np.float32)
        self.recent.append(tuple(float(x) for x in a))
        if self.vote > 1:
            cnt = collections.Counter(self.recent)
            best = max(cnt.values())
            a = next(c for c in reversed(self.recent) if cnt[c] == best)   # ties → most recent
        return a

    def latest(self):
        c = dict(self.cmd)
        now = time.time()
        c['frame_age_ms'] = round((now - c['frame_t']) * 1e3, 1)      # age of the frame this command came from
        c['fps'] = round(len(self.times) / max(sum(self.times), 1e-6), 1) if self.times else 0.0
        # stale = the camera stopped delivering (no new frame for stale_ms) → force stop
        c['stale'] = self.frame is not None and (now - self.frame_t) * 1e3 > self.stale_ms
        if c['stale']:
            c.update(self._make(0.0, 0.0))
        return c

    # ── frames in, inference thread ──────────────────────────────────────────
    def put(self, frame):
        with self.cv:
            self.frame, self.frame_t = frame, time.time()
            self.frame_seq += 1
            self.cv.notify_all()
            return self.frame_seq

    def _infer(self, frame, frame_t):
        with self.lock:
            t0 = time.perf_counter()
            plan = age = None
            if self.vla:
                self.vla.offer(self.n, frame)
                plan, age = self.vla.latest(self.n)
            _, raw = self.agent.step(frame, vla_plan=plan, vla_age_frames=age)
            v, w = self._decide(raw)
            self.n += 1
            ms = (time.perf_counter() - t0) * 1e3
            self.times.append(ms / 1e3)
            self.cmd = self._make(v, w, seq=self.n, frame_t=frame_t, infer_ms=round(ms, 1),
                                  raw=[float(x) for x in raw])

    def _worker(self):
        while True:
            with self.cv:
                while self.frame_seq == self.done_seq:
                    self.cv.wait()
                frame, frame_t, seq = self.frame, self.frame_t, self.frame_seq
            try:
                self._infer(frame, frame_t)
            except Exception as e:             # reported through the command; the client stops on error
                self.cmd = dict(self._make(0.0, 0.0, seq=self.n, frame_t=0.0, infer_ms=0.0),
                                error=f'{type(e).__name__}: {e}')
            with self.cv:
                self.done_seq = seq
                self.cv.notify_all()

    def step_sync(self, frame, timeout=60):
        seq = self.put(frame)
        with self.cv:
            self.cv.wait_for(lambda: self.done_seq >= seq, timeout=timeout)
        return self.latest()

    def reset(self):
        with self.lock:
            self.agent.reset()
            self.agent.prev.extend([np.zeros(2, np.float32)] * self.agent.prev.maxlen)
            self.agent.cond.extend([np.zeros(4, np.float32)] * self.agent.cond.maxlen)
            if self.vla:
                self.vla._plan = self.vla._src = None
            self.recent.clear()
            self.n = 0
            self.cmd = self._make(0.0, 0.0, seq=0, frame_t=time.time(), infer_ms=0.0)


def make_handler(policy):
    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'          # keep-alive: no new TCP connection per frame

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _frame(self, body):
            w, h = int(self.headers['X-Width']), int(self.headers['X-Height'])
            return np.frombuffer(body, np.uint8).reshape(h, w, 3).copy()

        def do_GET(self):
            if self.path.startswith('/command'):
                self._json(policy.latest())
            elif self.path.startswith('/info'):
                r = policy.robot
                self._json({'robot': r.name, 'platform': r.platform, 'levels': r.levels, 'fps': r.fps,
                            'budget_ms': round(1000 / r.fps), 'vla': policy.vla is not None,
                            'seq_len': policy.agent.cfg['seq_len'], 'device': policy.agent.device,
                            'vote': policy.vote, 'stale_ms': policy.stale_ms})
            else:
                self._json({'error': 'unknown path'}, 404)

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(n) if n else b''
            try:
                if self.path.startswith('/frame'):
                    policy.put(self._frame(body))
                    self._json(policy.latest())
                elif self.path.startswith('/step'):
                    self._json(policy.step_sync(self._frame(body)))
                elif self.path.startswith('/reset'):
                    policy.reset()
                    self._json({'ok': True})
                else:
                    self._json({'error': 'unknown path'}, 404)
            except Exception as e:
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
    ap.add_argument('--vote', type=int, default=1, help='majority vote over the last N inferences (3 = smoother, +1-2 steps lag)')
    ap.add_argument('--stale_ms', type=float, default=1000, help='stop when no new camera frame arrived for this long')
    ap.add_argument('--no_snap', action='store_true', help='return the raw regression output (paper preset)')
    a = ap.parse_args()
    pol = Policy(a.ckpt, a.vla, snap=not a.no_snap, device=a.device, vote=a.vote, stale_ms=a.stale_ms)
    r = pol.robot
    shape = (640, 480, 3) if r.platform == 'misty' else (240, 320, 3)
    for _ in range(3):                         # warm-up (kernels, YOLO, depth)
        pol.step_sync(np.zeros(shape, np.uint8))
    t = time.perf_counter()
    pol.step_sync(np.zeros(shape, np.uint8))
    ms = (time.perf_counter() - t) * 1e3
    pol.reset()
    pol.times.clear()
    print(f'policy server: {r.name} ({r.platform}, {pol.agent.device}) on http://{a.host}:{a.port}  '
          f'inference ≈ {ms:.0f} ms/frame (budget {1000 / r.fps:.0f} ms)  ckpt={a.ckpt}  vla={a.vla}', flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(pol)).serve_forever()


if __name__ == '__main__':
    main()
