"""No-hardware test of policy_misty.py (run inside the LSMA-Laser-tag folder, like its own tests):
    .\\.venv\\Scripts\\python.exe -m unittest test_policy_misty
"""
import asyncio
import time
import unittest
from unittest.mock import Mock

from PIL import Image

from policy_misty import PolicyMistyOnly


def make_controller(step):
    c = PolicyMistyOnly.__new__(PolicyMistyOnly)
    c.keys, c.running, c.connected = set(), True, True
    c.drive_changed = asyncio.Event()
    c.ack_values, c.ack_monotonic, c.in_flight = (0, 0), 0, False
    c.last_frame_at = time.monotonic()
    c.frame = Image.new('RGB', (480, 640))
    c.connection = Mock()
    c.policy = Mock(step=step, reset=Mock())
    c.policy_on, c.policy_values = False, (0, 0)
    c.sent = []

    async def send_drive(values):
        c.sent.append(values)
        c.ack_values, c.ack_monotonic = values, time.monotonic()
    c.send_drive = send_drive
    return c


class PolicyMistyTest(unittest.IsolatedAsyncioTestCase):
    async def run_for(self, c, seconds, frames=True):
        async def new_frames():
            while frames:
                c.frame = Image.new('RGB', (480, 640))
                c.last_frame_at = time.monotonic()
                await asyncio.sleep(0.1)
        tasks = [asyncio.create_task(t) for t in (c.policy_loop(), c.drive_loop(), new_frames())]
        await asyncio.sleep(seconds)
        c.running = False
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_policy_drives_and_manual_overrides(self):
        c = make_controller(lambda img: {'v': 25.0, 'w': 30.0, 'name': 'forward_left', 'ms': 5})
        c.set_policy(True)
        c.policy.reset.assert_called_once()
        await self.run_for(c, 0.5)
        self.assertIn((25, 30), c.sent)
        c.running, c.keys = True, {'s'}
        self.assertEqual(c.wanted(), (-25, 0))          # held WASD wins over the policy
        c.keys = {'space'}
        self.assertEqual(c.wanted(), (0, 0))

    async def test_server_error_stops(self):
        def boom(img):
            raise RuntimeError('down')
        c = make_controller(boom)
        c.set_policy(True)
        await self.run_for(c, 0.4)
        self.assertFalse(c.policy_on)
        self.assertEqual(c.wanted(), (0, 0))
        self.assertNotIn((25, 30), c.sent)


if __name__ == '__main__':
    unittest.main()
