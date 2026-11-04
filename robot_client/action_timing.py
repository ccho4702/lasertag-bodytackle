"""Timestamped completed actions; future actions are not invented by this buffer."""
from collections import deque
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class TimedAction:
    command: dict
    received_at: float
    expires_at: float

    def remaining_ms(self, now):
        return max(0, int((self.expires_at - now) * 1000))


class ActionMailbox:
    """Newest completed action wins; all execution deadlines use the local monotonic clock."""
    def __init__(self, epoch=None, ttl_ms=500):
        self.epoch = epoch
        self.ttl_ms = ttl_ms
        self.current = None
        self.last_seq = -1
        self.history = deque(maxlen=64)

    def offer(self, command, request_started, received_at):
        if self.epoch is not None and command.get('epoch') != self.epoch:
            return False
        if command.get('error'):
            raise RuntimeError(command['error'])
        seq = int(command.get('seq', 0))
        if seq <= self.last_seq or seq <= 0:
            return False
        # Bound transport/serialization delay without counting time spent waiting on the server.
        transport_ms = max(0, (received_at - request_started) * 1000 - command.get('poll_wait_ms', 0))
        valid_ms = min(command.get('valid_for_ms', self.ttl_ms - command['frame_age_ms']),
                       self.ttl_ms - command['frame_age_ms']) - transport_ms
        if command.get('stale') or valid_ms <= 0:
            return False
        action = TimedAction(dict(command), received_at, received_at + valid_ms / 1000)
        self.current, self.last_seq = action, seq
        self.history.append(action)
        return True

    def latest(self, now=None):
        now = time.monotonic() if now is None else now
        if self.current is None or self.current.expires_at <= now:
            return None
        return self.current
