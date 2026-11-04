import sys
from pathlib import Path
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from frame_pipeline import OrderedFramePipeline


class FramePipelineTests(unittest.TestCase):
    def test_new_frame_computes_before_old_one_finishes_but_actions_stay_ordered(self):
        first_entered, second_entered, release = threading.Event(), threading.Event(), threading.Event()
        commits = []
        class Agent:
            def encode_frame(self, frame):
                if frame == 1:
                    first_entered.set()
                    if not release.wait(2):
                        raise RuntimeError('test release timeout')
                else:
                    second_entered.set()
                return frame
            def act_feature(self, feature):
                commits.append(feature)
                return feature, feature
        pipeline = OrderedFramePipeline(Agent(), workers=2)
        try:
            self.assertTrue(pipeline.submit(1, 100))
            self.assertTrue(first_entered.wait(1))
            self.assertTrue(pipeline.submit(2, 200))
            self.assertTrue(second_entered.wait(1))
            self.assertIsNone(pipeline.pop())
            self.assertFalse(pipeline.submit(3, 300))
            release.set()
            self.assertEqual(pipeline.pop(block=True)['timestamp'], 100)
            self.assertEqual(pipeline.pop(block=True)['timestamp'], 200)
            self.assertEqual(commits, [1, 2])
            with self.assertRaises(ValueError):
                pipeline.submit(0, 150)
        finally:
            release.set()
            pipeline.close()


if __name__ == '__main__':
    unittest.main()
