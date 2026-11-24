"""Headless Cozmo policy run from the Mac (py3.8 + cozmo SDK): Cozmo camera -> policy server -> wheel motors.
usage: cozmo_live.py --seconds 30 [--dry]"""
import argparse, asyncio, os, signal, sys, time
import cozmo
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'robot_client'))
from policy_client import PolicyClient

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8771'); ap.add_argument('--seconds', type=float, default=30)
ap.add_argument('--dry', action='store_true'); ap.add_argument('--save', default=None)
a = ap.parse_args()

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
        while time.monotonic() - t0 < a.seconds and not STOP['flag']:
            img = robot.world.latest_image
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
        robot.stop_all_motors()
        await asyncio.sleep(0.3)          # give the stop message time to reach the robot before disconnecting
        print('STOP sent', flush=True)

# Ctrl+C / kill: leave the loop normally so the stop above is really sent. (With the SDK's default
# KeyboardInterrupt handling the program exited without stopping the wheels.)
STOP = {'flag': False}
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, lambda *_: STOP.update(flag=True))
cozmo.robot.Robot.drive_off_charger_on_connect = False
cozmo.run_program(main)
