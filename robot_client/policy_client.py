"""Tiny client for baseline/policy_server.py — Python 3.8, stdlib + numpy + Pillow only
(the LSMA-Laser-tag .venv already has both). Copy robot_client/*.py into the LSMA-Laser-tag folder."""
import json
import urllib.request

import numpy as np

DEFAULT_POLICY_URL = 'http://127.0.0.1:8770'


class PolicyClient:
    def __init__(self, url=DEFAULT_POLICY_URL, timeout=2.0):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def _req(self, path, data=None, headers=None):
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {},
                                     method='POST' if data is not None else 'GET')
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode())
        if 'error' in out:
            raise RuntimeError('policy server: ' + out['error'])
        return out

    def info(self):
        return self._req('/info')

    def reset(self):
        return self._req('/reset', b'')

    def step(self, image):
        """image: PIL image exactly as LSMA-Laser-tag saves it (Misty: rotated 480x640, Cozmo: raw_image).
        Returns {"v", "w", "name", "api", "ms", ...}."""
        arr = np.ascontiguousarray(np.asarray(image.convert('RGB'), dtype=np.uint8))
        h, w = arr.shape[:2]
        return self._req('/step', arr.tobytes(), {'X-Width': str(w), 'X-Height': str(h),
                                                   'Content-Type': 'application/octet-stream'})
