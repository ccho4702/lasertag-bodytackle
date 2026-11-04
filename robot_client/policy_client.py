"""Tiny client for baseline/policy_server.py — Python 3.8, stdlib + numpy + Pillow only
(the LSMA-Laser-tag .venv already has both). Copy robot_client/*.py into the LSMA-Laser-tag folder.

Asynchronous use (what policy_misty.py / policy_cozmo.py do):
    out = client.send_frame(img)   # returns at once with the server's *latest* command (inference runs in
                                   # the server's own thread on the newest frame; frames it can't keep up with are skipped)
    out = client.command()         # latest command without sending a frame
"""
import http.client
import json
import threading
import urllib.parse

import numpy as np

DEFAULT_POLICY_URL = 'http://127.0.0.1:8770'


class PolicyClient:
    def __init__(self, url=DEFAULT_POLICY_URL, timeout=2.0):
        u = urllib.parse.urlparse(url if '://' in url else 'http://' + url)
        self.host, self.port = u.hostname, u.port or 80
        self.url, self.timeout = url, timeout
        self._local = threading.local()        # one keep-alive connection per thread

    def _conn(self):
        c = getattr(self._local, 'conn', None)
        if c is None:
            c = self._local.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return c

    def _req(self, method, path, body=None, headers=None):
        for attempt in (0, 1):                 # one silent reconnect if the keep-alive socket was closed
            try:
                c = self._conn()
                c.request(method, path, body=body, headers=headers or {})
                out = json.loads(c.getresponse().read().decode())
                break
            except (http.client.HTTPException, ConnectionError, OSError):
                self._local.conn = None
                if attempt:
                    raise
        if 'error' in out:
            raise RuntimeError('policy server: ' + out['error'])
        return out

    @staticmethod
    def _encode(image):
        arr = np.ascontiguousarray(np.asarray(image.convert('RGB'), dtype=np.uint8))
        h, w = arr.shape[:2]
        return arr.tobytes(), {'X-Width': str(w), 'X-Height': str(h), 'Content-Type': 'application/octet-stream'}

    def info(self):
        return self._req('GET', '/info')

    def reset(self):
        return self._req('POST', '/reset', b'')

    def command(self):
        return self._req('GET', '/command')

    def wait_command(self, after, epoch=None, timeout_ms=200):
        query = {'after': after, 'timeout_ms': timeout_ms}
        if epoch is not None:
            query['epoch'] = epoch
        return self._req('GET', '/command?' + urllib.parse.urlencode(query))

    def send_frame(self, image):
        """image: PIL image exactly as LSMA-Laser-tag saves it (Misty: rotated 480x640, Cozmo: raw_image).
        Non-blocking w.r.t. inference: returns the latest command {"v","w","name","api","stale",...}."""
        body, headers = self._encode(image)
        return self._req('POST', '/frame', body, headers)

    def step(self, image):
        """synchronous: wait for the command computed from this very frame (tests only)."""
        body, headers = self._encode(image)
        return self._req('POST', '/step', body, headers)
