"""Cozmo driven by the learned policy — drop-in variant of LSMA-Laser-tag's wasd_cozmo.py.

Same camera (color stream, head −5°, robot.world.latest_image.raw_image 320×240), same motor call
(robot.drive_wheel_motors(l, r) mm/s: ±90 straight, ±60 turn), same recording (dataset_cozmo/).

    .\\.venv\\Scripts\\python.exe .\\policy_cozmo.py --policy-url http://<gpu-host>:8771 [--record]

Keys (click the window first):  P = policy ON/OFF (episode reset on ON)   W/A/S/D = manual override
                                Space = stop (policy OFF)                 Esc = quit
Any policy-server error stops the motors.
"""
import argparse
import asyncio
import time
import tkinter as tk
from pathlib import Path

import cozmo
from PIL import Image, ImageTk

from dataset_recorder import COZMO_FIELDS, DatasetRecorder
from frame_rate import CAMERA_FPS, FrameLimiter
from policy_client import PolicyClient
from recording_position import POSITION_FIELDS, position_row
from tracking_panel import DEFAULT_TRACKING_URL, TrackingClient
from wasd_cozmo import wheel_speeds


def main_factory(args):
    async def main(robot: cozmo.robot.Robot):
        client = PolicyClient(args.policy_url)
        info = client.info()
        if info.get('platform') != 'cozmo':
            raise SystemExit('policy server runs a {} model, not Cozmo'.format(info.get('robot')))
        window = tk.Tk()
        window.title('Cozmo POLICY + camera')
        window.geometry('680x760')
        status = tk.StringVar(value='Policy OFF — click here, P to start, Space to stop')
        tk.Label(window, textvariable=status, font=('Segoe UI', 12), wraplength=640).pack(pady=12)
        view = tk.Label(window, text="Waiting for Cozmo's camera...", bg='black', fg='white')
        view.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        keys, state = set(), {'on': False, 'run': True, 'speeds': (0, 0)}
        tracker = TrackingClient(DEFAULT_TRACKING_URL)
        recorder = DatasetRecorder(Path(__file__).parent / 'dataset_cozmo', fps=CAMERA_FPS,
                                   fields=COZMO_FIELDS + POSITION_FIELDS) if args.record else None

        def set_policy(on):
            state['on'] = on
            state['speeds'] = (0, 0)
            if on:
                try:
                    client.reset()
                except Exception as exc:
                    state['on'] = False
                    status.set('Policy server error: ' + str(exc))

        def press(event):
            k = event.keysym.lower()
            if k == 'escape':
                state['run'] = False
            elif k == 'p':
                if 'p' not in keys:
                    keys.add('p')
                    set_policy(not state['on'])
            elif k in {'w', 'a', 's', 'd', 'space'}:
                if k == 'space':
                    keys.clear()
                    set_policy(False)
                keys.add(k)

        window.bind('<KeyPress>', press)
        window.bind('<KeyRelease>', lambda e: keys.discard(e.keysym.lower()))
        window.bind('<FocusOut>', lambda e: keys.clear())
        window.protocol('WM_DELETE_WINDOW', lambda: state.update(run=False))

        def on_camera_image(evt, *, image, **kwargs):
            if recorder is not None and state['run']:
                recorder.offer(image, {'policy'} if state['on'] else set(keys), state['speeds'],
                               robot.head_angle.degrees, position=position_row(tracker, 5))

        handler = robot.world.add_event_handler(cozmo.world.EvtNewCameraImage, on_camera_image)
        loop = asyncio.get_event_loop()
        last_img, shown, preview, previous, pending = None, None, FrameLimiter(), None, None
        try:
            robot.stop_all_motors()
            robot.camera.color_image_enabled = True
            robot.camera.image_stream_enabled = True
            await robot.set_head_angle(cozmo.util.degrees(-5)).wait_for_completed()
            while state['run']:
                window.update()
                img = robot.world.latest_image
                manual = keys & {'w', 'a', 's', 'd'}
                if 'space' in keys:
                    speeds = (0, 0)
                elif state['on'] and not manual:
                    # upload new frames without waiting for inference; apply whatever command is newest
                    if pending is not None and pending.done():
                        try:
                            out = pending.result()
                            state['speeds'] = (0, 0) if out.get('stale') else (
                                out['api']['l_wheel_speed'], out['api']['r_wheel_speed'])
                            status.set('POLICY ON: {} | infer {} ms | frame age {} ms | {} fps'.format(
                                out['name'], out.get('infer_ms'), out.get('frame_age_ms'), out.get('fps')))
                        except Exception as exc:
                            set_policy(False)
                            status.set('Policy stopped: ' + str(exc))
                        pending = None
                    if state['on'] and pending is None and img is not None and img is not last_img:
                        last_img = img
                        pending = loop.run_in_executor(None, client.send_frame, img.raw_image)
                    speeds = state['speeds'] if state['on'] else (0, 0)
                else:
                    speeds = wheel_speeds(keys)
                if speeds != previous:
                    if speeds == (0, 0):
                        robot.stop_all_motors()
                    else:
                        robot.drive_wheel_motors(*speeds)
                    previous = speeds
                if img is not None and img is not shown and preview.ready():
                    frame = img.raw_image.copy()
                    frame.thumbnail((640, 480), Image.Resampling.BILINEAR)
                    photo = ImageTk.PhotoImage(frame, master=window)
                    view.configure(image=photo, text='')
                    view.image = photo
                    shown = img
                await asyncio.sleep(0.01)
        finally:
            robot.stop_all_motors()
            handler.disable()
            if recorder is not None:
                recorder.close()
                print('saved', recorder.count, 'frames to', recorder.path, flush=True)
            tracker.close()
            window.destroy()
    return main


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-url', default='http://127.0.0.1:8771')
    p.add_argument('--record', action='store_true', help='save images + applied wheel speeds to dataset_cozmo/')
    a = p.parse_args()
    cozmo.robot.Robot.drive_off_charger_on_connect = False
    cozmo.run_program(main_factory(a))
