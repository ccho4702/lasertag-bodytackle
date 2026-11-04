"""Bounded concurrent frame encoding followed by strictly ordered policy updates."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time


class OrderedFramePipeline:
    def __init__(self, agent, workers=2):
        if workers < 1:
            raise ValueError('workers must be positive')
        self.agent = agent
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = deque()
        self.last_timestamp = None

    def submit(self, frame, timestamp):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Camera timestamps must increase')
        if len(self.pending) >= self.workers:
            return False  # caller retains/replaces its latest frame; never grow an old-image queue
        self.last_timestamp = timestamp
        def encode():
            start = time.perf_counter()
            feature = self.agent.encode_frame(frame)
            return feature, start, time.perf_counter()
        self.pending.append((timestamp, self.pool.submit(encode)))
        return True

    def pop(self, block=False, commit=True):
        if not self.pending:
            return None
        timestamp, future = self.pending[0]
        if not block and not future.done():
            return None
        self.pending.popleft()
        feature, started, finished = future.result()
        command, raw = self.agent.act_feature(feature) if commit else (None, None)
        return dict(timestamp=timestamp, command=command, raw=raw, feature=feature,
                    encoding_started=started, encoding_finished=finished,
                    committed_at=time.perf_counter())

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
