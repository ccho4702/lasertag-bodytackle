"""No-robot regression tests for overlapping inference and timestamped action delivery."""
from collections import deque
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baseline'))
sys.path.insert(0, str(ROOT / 'robot_client'))
from action_timing import ActionMailbox
from policy_client import PolicyClient
import policy_server


class ActionTimingTests(unittest.TestCase):
    def message(self, seq=1, epoch='episode', **extra):
        return dict(seq=seq, epoch=epoch, v=25, w=0, frame_age_ms=100,
                    valid_for_ms=400, stale=False, **extra)

    def test_out_of_order_and_old_episode_actions_are_rejected(self):
        box = ActionMailbox(epoch='episode')
        self.assertTrue(box.offer(self.message(seq=2), 10, 10.01))
        self.assertFalse(box.offer(self.message(seq=1), 10, 10.02))
        self.assertFalse(box.offer(self.message(seq=3, epoch='old'), 10, 10.02))
        self.assertEqual(box.latest(10.1).command['seq'], 2)

    def test_deadline_uses_monotonic_time_and_excludes_server_poll_wait(self):
        box = ActionMailbox(epoch='episode')
        self.assertTrue(box.offer(self.message(poll_wait_ms=200), 10, 10.21))
        self.assertAlmostEqual(box.current.expires_at, 10.6)
        self.assertIsNotNone(box.latest(10.59))
        self.assertIsNone(box.latest(10.61))
        self.assertFalse(box.offer(dict(self.message(seq=2), valid_for_ms=0), 10.7, 10.71))

    def test_server_error_with_repeated_sequence_is_not_ignored(self):
        box = ActionMailbox(epoch='episode')
        box.offer(self.message(), 10, 10.01)
        with self.assertRaises(RuntimeError):
            box.offer(self.message(error='inference failed'), 10.02, 10.03)


class FakeAgent:
    def __init__(self, *args, **kwargs):
        self.model = SimpleNamespace(vla_cond=False)
        self.cfg = {'seq_len': 8}
        self.robot = SimpleNamespace(name='chaser', platform='misty', fps=10, levels=[[-25, 0, 25], [-25, 0, 30]],
                                     snap=lambda x: x,
                                     to_command=lambda x, snap=False: {'LinearVelocity': x[0], 'AngularVelocity': x[1]})
        self.prev, self.cond = deque(maxlen=8), deque(maxlen=8)
        self.device = 'cpu'
        self.started = []
        self.finished = []
        self.delay = 0.08

    def step(self, frame, **kwargs):
        return self.act_feature(self.encode_frame(frame))

    def encode_frame(self, frame):
        self.started.append(time.monotonic())
        time.sleep(self.delay)
        self.finished.append(time.monotonic())
        return frame

    def act_feature(self, feature, **kwargs):
        return {}, np.array([25, 0], np.float32)

    def reset(self):
        pass


class PipelineOverlapTests(unittest.TestCase):
    workers = 1
    def setUp(self):
        with patch.object(policy_server, 'Agent', FakeAgent):
            self.policy = policy_server.Policy('fake', infer_fps=0, perception_workers=self.workers)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), policy_server.make_handler(self.policy))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = PolicyClient(f'http://127.0.0.1:{self.server.server_port}')

    def tearDown(self):
        self.client._conn().close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.policy.close()

    def test_next_action_finishes_before_previous_motor_lease_ends(self):
        epoch = self.client.reset()['epoch']
        self.client.send_frame(Image.new('RGB', (8, 8)))
        first = self.client.wait_command(0, epoch, 500)
        self.assertEqual(first['seq'], 1)
        self.assertLessEqual(first['inference_started_at'], first['computed_at'])
        motor_lease_end = time.monotonic() + 0.4  # simulated ongoing robot movement
        self.client.send_frame(Image.new('RGB', (8, 8), color='red'))
        second = self.client.wait_command(1, epoch, 500)
        self.assertEqual(second['seq'], 2)
        self.assertGreater(second['source_frame_seq'], first['source_frame_seq'])
        self.assertLess(self.policy.agent.finished[-1], motor_lease_end)
        self.assertGreater(second['valid_for_ms'], 0)
        self.assertFalse(second['stale'])

    def test_reset_wakes_waiter_and_prevents_replaying_old_camera_frame(self):
        self.client.send_frame(Image.new('RGB', (8, 8)))
        first = self.client.wait_command(0, timeout_ms=500)
        reset = self.client.reset()
        new = self.client.wait_command(first['seq'], first['epoch'], 500)
        self.assertEqual(new['epoch'], reset['epoch'])
        self.assertEqual(new['seq'], 0)
        self.assertTrue(new['stale'])
        time.sleep(0.1)
        self.assertIsNone(self.policy.frame)
        self.assertEqual(self.policy.latest()['seq'], 0)


class ConcurrentFrameServerTests(PipelineOverlapTests):
    workers = 2

    def test_two_different_camera_frames_are_in_flight_at_once(self):
        self.policy.agent.delay = 0.2
        epoch = self.client.reset()['epoch']
        self.client.send_frame(Image.new('RGB', (8, 8)))
        until = time.monotonic() + 1
        while not self.policy.agent.started and time.monotonic() < until:
            time.sleep(0.001)
        self.client.send_frame(Image.new('RGB', (8, 8), color='blue'))
        first = self.client.wait_command(0, epoch, 500)
        second = self.client.wait_command(first['seq'], epoch, 500)
        self.assertEqual(second['seq'], 2)
        self.assertLess(self.policy.agent.started[1], self.policy.agent.finished[0])
        self.assertLess(first['source_frame_seq'], second['source_frame_seq'])


if __name__ == '__main__':
    unittest.main()
