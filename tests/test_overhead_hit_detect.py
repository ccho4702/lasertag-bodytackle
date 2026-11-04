"""Bird's-eye first-hit timing and terminal-video-frame regressions, without robots."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from overhead import hit_detect


class FakeModel:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        # Separate first (arms hit detection), then remain in contact.
        other = [80, 10, 90, 20] if self.calls == 1 else [21, 10, 31, 20]
        boxes = SimpleNamespace(xyxy=np.array([[10, 10, 20, 20], other]),
                                cls=np.array([0, 1]), conf=np.array([0.9, 0.9]))
        return [SimpleNamespace(boxes=boxes)]


class FakeWriter:
    def __init__(self, *args):
        self.frames = []
        self.closed = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


class OverheadHitTests(unittest.TestCase):
    def run_case(self, no_video):
        args = SimpleNamespace(arm_gap_px=30, gap_px=10, dist_px=0, hit_frames=3,
                               hold=5, imgsz=960, conf=0.3, device='cpu', no_video=no_video,
                               fps=10, stop_on_hit=True)
        frames = [(str(i), np.zeros((100, 100, 3), np.uint8), i / 10) for i in range(10)]
        model, writer = FakeModel(), FakeWriter()
        with tempfile.TemporaryDirectory() as output, \
             patch.object(hit_detect, 'frames_of', return_value=iter(frames)), \
             patch.object(hit_detect, 'Writer', return_value=writer) as factory:
            result = hit_detect.run_stream(model, args, 'test', output)
        return model, writer, factory, result

    def test_stop_on_first_three_frame_confirmation(self):
        model, writer, factory, result = self.run_case(no_video=True)
        self.assertEqual(model.calls, 4)
        self.assertEqual(result['frames'], 4)
        self.assertEqual(result['first_hit']['frame_idx'], 1)
        self.assertAlmostEqual(result['time_to_first_hit_s'], 0.1)
        factory.assert_not_called()

    def test_video_includes_first_confirmed_hit_frame(self):
        model, writer, factory, result = self.run_case(no_video=False)
        self.assertEqual(len(writer.frames), result['frames'])
        self.assertEqual(len(writer.frames), 4)
        self.assertTrue(writer.closed)
        self.assertTrue(np.array_equal(writer.frames[-1][0, 0], [0, 0, 255]))

    def test_live_stream_maximum_duration(self):
        args = SimpleNamespace(session=None, video='http://example.test/stream.mjpg', fps=10, max_s=0.25)
        frame = np.zeros((10, 10, 3), np.uint8)
        cap = SimpleNamespace(get=lambda _: 10, read=lambda: (True, frame))
        with patch.object(hit_detect.cv2, 'VideoCapture', return_value=cap), \
             patch.object(hit_detect.time, 'time', side_effect=[100, 100.1, 100.2, 100.3]):
            frames = list(hit_detect.frames_of(args))
        self.assertEqual(len(frames), 2)
        self.assertAlmostEqual(frames[-1][2], 0.2)


if __name__ == '__main__':
    unittest.main()
