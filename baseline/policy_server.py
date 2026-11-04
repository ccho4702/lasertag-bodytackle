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
import uuid
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from deploy import Agent
from frame_pipeline import OrderedFramePipeline


def command_name(v, w):
    """same names as LSMA-Laser-tag (wasd_misty.command_name / dataset_recorder.command_name)"""
    parts = []
    if v:
        parts.append('forward' if v > 0 else 'backward')
    if w:
        parts.append('left' if w > 0 else 'right')
    return '_'.join(parts) or 'stop'


class Policy:
    def __init__(self, ckpt, vla=None, snap=True, device=None, vote=1, stale_ms=1000, infer_fps=None,
                 command_ttl_ms=500, perception_workers=1):
        self.agent = Agent(ckpt, snap=False, device=device)
        self.snap, self.vote, self.stale_ms = snap, vote, stale_ms
        self.robot = self.agent.robot
        # run inference no faster than the frame rate the policy was trained on (Misty ≈ 9.9, Cozmo ≈ 8.3 fps
        # actually recorded): the seq_len-frame window then spans the same time as in training (8 frames ≈ 0.8 s)
        # even when the camera delivers 15 fps. 0 = every frame.
        self.infer_fps = self.robot.fps if infer_fps is None else infer_fps
        self._next_t = 0.0
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
        self.epoch = uuid.uuid4().hex
        self.command_ttl_ms = command_ttl_ms
        self.last_input_t = None
        self.perception_workers = perception_workers
        self.inflight_frames = 0
        self.recent = collections.deque(maxlen=max(1, vote))
        # latest-frame slot + latest command
        self.cv = threading.Condition()
        self.closed = False
        self.frame, self.frame_t, self.frame_seq = None, 0.0, 0
        self.done_seq = 0
        self.cmd = self._make(0.0, 0.0, seq=0, frame_t=time.time(), infer_ms=0.0)
        self.times = collections.deque(maxlen=20)
        self.completions = collections.deque(maxlen=40)
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.worker.join(timeout=2)

    # ── command bookkeeping ───────────────────────────────────────────────────
    def _make(self, v, w, **extra):
        a = np.array([v, w], np.float32)
        api = self.robot.to_command(a, snap=False)
        if self.robot.platform == 'misty':
            api = {k: int(round(x)) for k, x in api.items()}
        return {'v': float(v), 'w': float(w), 'name': command_name(v, w), 'api': api,
                'epoch': self.epoch, **extra}

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
        c['expires_at'] = c['frame_t'] + self.command_ttl_ms / 1000
        c['valid_for_ms'] = max(0.0, round((c['expires_at'] - now) * 1e3, 1))
        completed = tuple(self.completions)
        c['fps'] = round((len(completed) - 1) / max(completed[-1] - completed[0], 1e-6), 1) if len(completed) > 1 else 0.0
        # stale = the camera stopped delivering (no new frame for stale_ms) → force stop
        c['stale'] = (self.frame is None or (now - self.frame_t) * 1e3 > self.stale_ms
                      or c['valid_for_ms'] <= 0 or bool(c.get('error')))
        if c['stale']:
            c.update(self._make(0.0, 0.0))
        return c

    def wait_command(self, after, epoch=None, timeout=0.2):
        """Wait for publication, never for the robot's current movement to finish."""
        started = time.monotonic()
        with self.cv:
            self.cv.wait_for(lambda: self.cmd.get('seq', 0) > after or
                             (epoch is not None and epoch != self.epoch), timeout=timeout)
        command = self.latest()
        command['poll_wait_ms'] = round((time.monotonic() - started) * 1000, 3)
        return command

    # ── frames in, inference thread ──────────────────────────────────────────
    def put(self, frame):
        with self.cv:
            self.frame, self.frame_t = frame, time.time()
            self.frame_seq += 1
            self.cv.notify_all()
            return self.frame_seq

    def _infer(self, frame, frame_t, source_seq=None, epoch=None):
        with self.lock:
            if epoch is not None and epoch != self.epoch:
                return
            t0 = time.perf_counter()
            inference_started_at = time.time()
            plan = age = None
            if self.vla:
                self.vla.offer(self.n, frame)
                plan, age = self.vla.latest(self.n)
            _, raw = self.agent.step(frame, vla_plan=plan, vla_age_frames=age)
            ms = (time.perf_counter() - t0) * 1e3
            self._record(raw, frame_t, source_seq, inference_started_at, ms)

    def _record(self, raw, frame_t, source_seq, started_at, ms, **extra):
        """Publish only from the ordered policy thread while holding self.lock."""
        v, w = self._decide(raw)
        self.n += 1
        self.times.append(ms / 1e3)
        self.completions.append(time.perf_counter())
        self.cmd = self._make(v, w, seq=self.n, frame_t=frame_t, infer_ms=round(ms, 1),
                              raw=[float(x) for x in raw], source_frame_seq=source_seq,
                              inference_started_at=started_at, computed_at=time.time(),
                              input_gap_ms=None if self.last_input_t is None else
                              round((frame_t - self.last_input_t) * 1000, 2), **extra)
        self.last_input_t = frame_t

    def _parallel_worker(self):
        pipeline = OrderedFramePipeline(self.agent, workers=self.perception_workers)
        tickets = collections.deque()
        dispatched = 0
        next_dispatch = 0.0
        try:
            while True:
                if pipeline.pending and pipeline.pending[0][1].done():
                    ticket = tickets.popleft()
                    frame, frame_t, seq, epoch = ticket
                    try:
                        result = pipeline.pop(commit=False)
                        with self.lock:
                            if epoch == self.epoch:
                                t0 = time.perf_counter()
                                plan = age = None
                                if self.vla:
                                    self.vla.offer(self.n, frame)
                                    plan, age = self.vla.latest(self.n)
                                _, raw = self.agent.act_feature(result['feature'], vla_plan=plan, vla_age_frames=age)
                                temporal_ms = (time.perf_counter() - t0) * 1000
                                perception_ms = (result['encoding_finished'] - result['encoding_started']) * 1000
                                started_at = time.time() - (time.perf_counter() - result['encoding_started'])
                                self._record(raw, frame_t, seq, started_at, perception_ms + temporal_ms,
                                             perception_ms=round(perception_ms, 2), temporal_ms=round(temporal_ms, 2),
                                             commit_wait_ms=round((t0 - result['encoding_finished']) * 1000, 2))
                    except Exception as exc:
                        with self.lock:
                            if epoch == self.epoch:
                                self.cmd = self._make(0, 0, seq=self.n, frame_t=0, infer_ms=0,
                                                      error=f'{type(exc).__name__}: {exc}')
                    with self.cv:
                        self.done_seq = max(self.done_seq, seq)
                        self.cv.notify_all()
                with self.cv:
                    self.inflight_frames = len(pipeline.pending)
                    if self.closed:
                        return
                    now = time.monotonic()
                    if (len(pipeline.pending) < self.perception_workers and self.frame is not None
                            and self.frame_seq > dispatched and (not self.infer_fps or now >= next_dispatch)):
                        ticket = self.frame, self.frame_t, self.frame_seq, self.epoch
                        dispatched = self.frame_seq
                        next_dispatch = now + (1 / self.infer_fps if self.infer_fps else 0)
                    else:
                        self.cv.wait(timeout=0.002)
                        continue
                pipeline.submit(ticket[0], ticket[2])
                tickets.append(ticket)
        finally:
            pipeline.close()

    def _worker(self):
        if self.perception_workers > 1:
            return self._parallel_worker()
        while True:
            if self.infer_fps:                 # pace: wait until the next slot, then take the newest frame
                time.sleep(max(0.0, self._next_t - time.monotonic()))
            with self.cv:
                while self.frame_seq == self.done_seq and not self.closed:
                    self.cv.wait()
                if self.closed:
                    return
                frame, frame_t, seq = self.frame, self.frame_t, self.frame_seq
                epoch = self.epoch
            if self.infer_fps:
                self._next_t = max(self._next_t + 1.0 / self.infer_fps, time.monotonic() - 0.5 / self.infer_fps)
            try:
                self._infer(frame, frame_t, seq, epoch)
            except Exception as e:             # reported through the command; the client stops on error
                with self.lock:
                    if epoch == self.epoch:
                        self.cmd = dict(self._make(0.0, 0.0, seq=self.n, frame_t=0.0, infer_ms=0.0),
                                        error=f'{type(e).__name__}: {e}')
            with self.cv:
                self.done_seq = max(self.done_seq, seq)
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
            self.completions.clear()
            with self.cv:
                self.epoch = uuid.uuid4().hex
                self.last_input_t = None
                self.frame, self.frame_t = None, 0.0
                self.done_seq = self.frame_seq
                self.cmd = self._make(0.0, 0.0, seq=0, frame_t=time.time(), infer_ms=0.0)
                self.cv.notify_all()


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
                query = parse_qs(urlsplit(self.path).query)
                if 'after' in query:
                    self._json(policy.wait_command(int(query['after'][0]), query.get('epoch', [None])[0],
                               min(1.0, max(0.0, float(query.get('timeout_ms', ['200'])[0]) / 1000))))
                else:
                    self._json(policy.latest())
            elif self.path.startswith('/info'):
                r = policy.robot
                self._json({'robot': r.name, 'platform': r.platform, 'levels': r.levels, 'fps': r.fps,
                            'budget_ms': round(1000 / r.fps), 'vla': policy.vla is not None,
                            'seq_len': policy.agent.cfg['seq_len'], 'device': policy.agent.device,
                            'vote': policy.vote, 'stale_ms': policy.stale_ms, 'infer_fps': policy.infer_fps,
                            'epoch': policy.epoch, 'command_ttl_ms': policy.command_ttl_ms,
                            'action_horizon': 1, 'perception_workers': policy.perception_workers,
                            'inflight_frames': policy.inflight_frames})
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
                    self._json({'ok': True, 'epoch': policy.epoch})
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
    ap.add_argument('--infer_fps', type=float, default=None,
                    help='max inference rate (default: the training frame rate of the robot; 0 = every frame)')
    ap.add_argument('--no_snap', action='store_true', help='return the raw regression output (paper preset)')
    ap.add_argument('--perception_workers', type=int, choices=(1, 2), default=1,
                    help='Independent concurrent frame encoders; LSTM updates remain timestamp ordered')
    a = ap.parse_args()
    pol = Policy(a.ckpt, a.vla, snap=not a.no_snap, device=a.device, vote=a.vote, stale_ms=a.stale_ms,
                 infer_fps=a.infer_fps, perception_workers=a.perception_workers)
    r = pol.robot
    shape = (640, 480, 3) if r.platform == 'misty' else (240, 320, 3)
    paced, pol.infer_fps = pol.infer_fps, 0
    for _ in range(3):                         # warm-up (kernels, YOLO, depth)
        pol.step_sync(np.zeros(shape, np.uint8))
    t = time.perf_counter()
    pol.step_sync(np.zeros(shape, np.uint8))
    ms = (time.perf_counter() - t) * 1e3
    pol.reset()
    pol.times.clear()
    pol.infer_fps = paced
    print(f'policy server: infer ≤ {paced or "every"} fps | ' f' {r.name} ({r.platform}, {pol.agent.device}) on http://{a.host}:{a.port}  '
          f'inference ≈ {ms:.0f} ms/frame (budget {1000 / r.fps:.0f} ms)  ckpt={a.ckpt}  vla={a.vla}', flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(pol)).serve_forever()


if __name__ == '__main__':
    main()
