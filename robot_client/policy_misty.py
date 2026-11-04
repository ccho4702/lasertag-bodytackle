"""Misty driven by the learned policy — drop-in variant of LSMA-Laser-tag's wasd_misty.py / wasd_misty_only.py.

Same camera stream (90° cw → 480×640), same drive API (POST /api/drive/time, 500 ms, re-sent every 0.15 s),
same recording (datasets_misty4/, + overhead images when tracking is on) — only the command source changes.

    .\\.venv\\Scripts\\python.exe .\\policy_misty.py --policy-url http://<gpu-host>:8770              # Misty only
    .\\.venv\\Scripts\\python.exe .\\policy_misty.py --policy-url http://<gpu-host>:8770 --with-cozmo # + Cozmo toggle, tracking, overhead

Keys:  P = policy ON/OFF (episode reset on ON)   W/A/S/D = manual override while held
       Space = stop everything (policy OFF)       Esc = quit
Any policy-server error or a camera gap > 2 s stops Misty.
"""
import argparse
import asyncio
import time

import wasd_misty
from policy_client import DEFAULT_POLICY_URL, PolicyClient
from wasd_misty import DEFAULT_IP, MistyController, command_name, velocities
from wasd_misty_only import MistyOnlyController


class PolicyMixin:
    def setup_policy(self, url):
        self.policy = PolicyClient(url)
        info = self.policy.info()
        if info.get('platform') != 'misty':
            raise SystemExit('policy server runs a {} model, not Misty'.format(info.get('robot')))
        self.policy_on = False
        self.policy_values = (0, 0)
        self.policy_status = 'Policy OFF (P to start) | server {} | budget {} ms'.format(url, info['budget_ms'])
        self.window.title('Misty 4 - POLICY + camera + dataset')
        print('policy server:', info, flush=True)

    # command source ------------------------------------------------------------------------------
    def wanted(self):
        manual = self.keys & {'w', 'a', 's', 'd'}
        if 'space' in self.keys:
            return 0, 0
        if self.policy_on and not manual:
            return self.policy_values
        return velocities(self.keys)

    def press(self, event):
        key = event.keysym.lower()
        if key == 'p':
            if 'p' not in self.keys:
                self.keys.add('p')
                self.set_policy(not self.policy_on)
            return
        if key == 'space':
            self.set_policy(False)
        super().press(event)

    def release(self, event):
        self.keys.discard(event.keysym.lower())
        self.drive_changed.set()

    def set_policy(self, on):
        if on and not self.connected:
            return
        self.policy_on = on
        self.policy_values = (0, 0)
        if on:
            try:
                self.policy.reset()
            except Exception as exc:
                self.policy_on = False
                self.connection.set('Policy server error: ' + str(exc))
        self.drive_changed.set()

    async def policy_loop(self):
        """Upload every new camera frame as soon as it arrives (no waiting for inference: the server answers
        with its newest command at once). drive_loop keeps re-sending that command every 0.15 s, so Misty
        moves continuously whatever the inference latency."""
        loop = asyncio.get_running_loop()
        last = None
        while self.running:
            frame = self.frame
            if self.policy_on and self.connected and frame is not None and frame is not last:
                last = frame
                try:
                    out = await loop.run_in_executor(None, self.policy.send_frame, frame)
                    if self.policy_on:
                        values = (0, 0) if out.get('stale') else (int(out['v']), int(out['w']))
                        if values != self.policy_values:
                            self.policy_values = values
                            self.drive_changed.set()
                        self.connection.set('POLICY ON: {} | infer {} ms | frame age {} ms | {} fps{}'.format(
                            out['name'], out.get('infer_ms'), out.get('frame_age_ms'), out.get('fps'),
                            ' | STALE' if out.get('stale') else ''))
                except Exception as exc:
                    self.policy_on = False
                    self.policy_values = (0, 0)
                    self.connection.set('Policy stopped: ' + str(exc))
                    self.drive_changed.set()
            await asyncio.sleep(0.005)

    # copies of wasd_misty.MistyController methods with velocities(self.keys) → self.wanted() ---------
    async def drive_loop(self):
        try:
            while self.running and self.connected:
                self.drive_changed.clear()
                wanted = self.wanted() if time.monotonic() - self.last_frame_at < 2 else (0, 0)
                if wanted != self.ack_values or (wanted != (0, 0) and time.monotonic() - self.ack_monotonic > 0.15):
                    await self.send_drive(wanted)
                    continue
                try:
                    await asyncio.wait_for(self.drive_changed.wait(), timeout=0.02)
                except asyncio.TimeoutError:
                    pass
        except Exception as exc:
            self.fail('Drive connection failed: ' + str(exc))

    def label(self, received):
        row = super().label(received)
        wanted = self.wanted()
        state = 'acknowledged'
        if self.in_flight or wanted != self.ack_values:
            state = 'transition'
        elif self.ack_values != (0, 0) and time.monotonic() - self.ack_monotonic > 0.5:
            state = 'expired'
        row.update({'requested_command': command_name(wanted), 'command_state': state,
                    'keys': ('policy' if self.policy_on and not (self.keys & {'w', 'a', 's', 'd'}) else row['keys'])})
        return row

    async def connect(self):
        await super().connect()
        asyncio.create_task(self.policy_loop())


class PolicyMistyOnly(PolicyMixin, MistyOnlyController):
    pass


class PolicyMisty(PolicyMixin, MistyController):
    pass


async def launch(args):
    if not args.with_cozmo:
        c = PolicyMistyOnly(args.ip)
        c.setup_policy(args.policy_url)
        await c.run()
        return
    # same as wasd_misty.launch, with the policy controller
    cozmo = None
    try:
        cozmo = await wasd_misty.preflight(args.ip, wasd_misty.decode_frame, wasd_misty.unwrap)
        c = PolicyMisty(args.ip, cozmo=cozmo)
        c.setup_policy(args.policy_url)
        await c.run()
    finally:
        if cozmo is not None:
            await cozmo.close()
            if getattr(cozmo, 'misty_socket', None) is not None:
                await cozmo.misty_socket.close()
            if getattr(cozmo, 'misty_session', None) is not None:
                await cozmo.misty_session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', default=DEFAULT_IP)
    parser.add_argument('--policy-url', default=DEFAULT_POLICY_URL)
    parser.add_argument('--with-cozmo', action='store_true',
                        help='full wasd_misty stack: Cozmo ON/OFF toggle, tracking, overhead images in recordings')
    args = parser.parse_args()
    try:
        asyncio.run(launch(args))
    except Exception as exc:
        print('Policy controller failed:', str(exc) or type(exc).__name__, flush=True)
        raise SystemExit(1)
