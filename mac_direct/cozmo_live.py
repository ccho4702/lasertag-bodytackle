"""Headless Cozmo policy run from the Mac (py3.8 + cozmo SDK): Cozmo camera -> policy server -> wheel motors.
usage: cozmo_live.py --seconds 30 [--dry] [--gate FILE]
  --gate FILE: wait in standby until FILE says "run" (see gate_open)"""
import argparse, asyncio, os, signal, sys, time
import cozmo
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'robot_client'))
from policy_client import PolicyClient

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8771'); ap.add_argument('--seconds', type=float, default=30)
ap.add_argument('--dry', action='store_true'); ap.add_argument('--save', default=None); ap.add_argument('--gate', default=None)
a = ap.parse_args()

def gate_open():
    """--gate FILE: drive only while FILE holds the word "run"; otherwise standby (connected, camera on, wheels stopped).
    The control panel uses this to start both robots and the overhead recording at the same moment."""
    if not a.gate:
        return True
    try:
        return open(a.gate).read().strip() == 'run'
    except OSError:
        return False

async def main(robot: cozmo.robot.Robot):
    pol = PolicyClient(a.url); info = pol.info(); print('policy:', info['robot'], flush=True)
    assert info['platform'] == 'cozmo'
    loop = asyncio.get_event_loop()
    try:
        robot.stop_all_motors()
        robot.camera.color_image_enabled = True
        robot.camera.image_stream_enabled = True
        await robot.set_head_angle(cozmo.util.degrees(-5)).wait_for_completed()
        pol.reset()
        t0 = time.monotonic(); last_img = None; pending = None; prev = None; speeds = (0, 0)
        out = {}; n = 0; last_print = 0; last_frame_t = time.monotonic()
        going = not a.gate; ready = False; gate_t = 0.0; go = going; seen_img = None; seen_t = time.monotonic()
        while time.monotonic() - t0 < a.seconds and not STOP['flag']:
            img = robot.world.latest_image
            # The SDK keeps this program alive after the tablet connection drops, and the robot then just sits
            # there. Leave instead, so the panel sees that Cozmo is gone (also checked while in standby).
            if img is not seen_img:
                seen_img = img; seen_t = time.monotonic()
            if not robot.conn.is_connected:
                LOST['why'] = '태블릿과의 연결이 끊겼습니다 (Cozmo 앱의 SDK 모드와 USB 연결을 확인하세요)'
                break
            if time.monotonic() - seen_t > 5:
                LOST['why'] = 'Cozmo 카메라 프레임이 5초 동안 들어오지 않았습니다 (Cozmo 전원과 앱 연결을 확인하세요)'
                break
            if time.monotonic() - gate_t > 0.05:
                gate_t = time.monotonic(); go = gate_open()
            if go and not going:                        # standby -> run: start from an empty history
                if pending is not None:
                    await pending
                pol.reset(); pending = None; out = {}; speeds = (0, 0); last_frame_t = time.monotonic()
                print('GO', flush=True)
            elif going and not go:                      # run -> standby: stop the wheels once
                robot.stop_all_motors(); speeds = (0, 0); prev = (0, 0)
                print('STANDBY (stop sent)', flush=True)
            going = go
            if not going:
                if img is not None and not ready:
                    ready = True; print('READY (standby: camera frames arriving)', flush=True)
                await asyncio.sleep(0.02)
                continue
            if pending is not None and pending.done():
                out = pending.result(); pending = None
                speeds = (0, 0) if out.get('stale') else (out['api']['l_wheel_speed'], out['api']['r_wheel_speed'])
            if pending is None and img is not None and img is not last_img:
                last_img = img; n += 1; last_frame_t = time.monotonic()
                if a.save and n == 5:
                    img.raw_image.save(a.save)
                pending = loop.run_in_executor(None, pol.send_frame, img.raw_image)
            if time.monotonic() - last_frame_t > 2:
                speeds = (0, 0)
            if speeds != prev and not a.dry:
                if speeds == (0, 0):
                    robot.stop_all_motors()
                else:
                    robot.drive_wheel_motors(*speeds)
                prev = speeds
            if time.monotonic() - last_print > 0.5:
                last_print = time.monotonic()
                print('%5.1fs  L=%+d R=%+d %-14s infer %s ms  age %s ms  frames %d%s' % (
                    time.monotonic() - t0, speeds[0], speeds[1], out.get('name', '-'), out.get('infer_ms'),
                    out.get('frame_age_ms'), n, '  [DRY]' if a.dry else ''), flush=True)
            await asyncio.sleep(0.01)
    finally:
        if LOST['why']:
            print('LOST: ' + LOST['why'], flush=True)
            os._exit(3)                   # the SDK would otherwise wait forever on the dead connection
        robot.stop_all_motors()
        await asyncio.sleep(0.3)          # give the stop message time to reach the robot before disconnecting
        print('STOP sent', flush=True)

# Ctrl+C / kill: leave the loop normally so the stop above is really sent. (With the SDK's default
# KeyboardInterrupt handling the program exited without stopping the wheels.)
STOP = {'flag': False}
LOST = {'why': ''}
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, lambda *_: STOP.update(flag=True))
cozmo.robot.Robot.drive_off_charger_on_connect = False
cozmo.run_program(main)
