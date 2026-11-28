"""Headless Misty policy run from the Mac: Misty camera -> policy server (localhost) -> Misty drive API.
usage: misty_live.py --seconds 20 [--dry] [--gate FILE]
  --dry: camera + policy only, no drive commands
  --gate FILE: wait in standby until FILE says "run" (see gate_open)"""
import argparse, asyncio, base64, io, json, os, sys, time
import aiohttp
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'robot_client'))
from policy_client import PolicyClient

ap = argparse.ArgumentParser()
ap.add_argument('--ip', default='192.168.1.38'); ap.add_argument('--url', default='http://127.0.0.1:8770')
ap.add_argument('--seconds', type=float, default=20); ap.add_argument('--dry', action='store_true')
ap.add_argument('--save', default=None); ap.add_argument('--gate', default=None); ap.add_argument('--actions', default=None)
a = ap.parse_args()

# --actions FILE: one CSV row per policy output while driving (wall-clock time, command sent to the robot,
# what the policy said, inference time, frame age, stale flag). The panel cuts out each game's rows.
ACT = open(a.actions, 'w', buffering=1) if a.actions else None
if ACT:
    ACT.write('t_unix,cmd_a,cmd_b,name,infer_ms,frame_age_ms,stale\n')
def log_action(cmd, out):
    if ACT:
        ACT.write('%.3f,%s,%s,%s,%s,%s,%d\n' % (time.time(), cmd[0], cmd[1], out.get('name', ''), out.get('infer_ms', ''),
                                              out.get('frame_age_ms', ''), bool(out.get('stale'))))

def gate_open():
    """--gate FILE: drive only while FILE holds the word "run"; otherwise standby (connected, camera on, wheels stopped).
    The control panel uses this to start both robots and the overhead recording at the same moment."""
    if not a.gate:
        return True
    try:
        return open(a.gate).read().strip() == 'run'
    except OSError:
        return False

def decode(p):
    if isinstance(p, str):
        if p.lstrip().startswith(('{', '[', '"')):
            p = json.loads(p)
        if isinstance(p, dict):
            for k in ('base64', 'image', 'data', 'result'):
                if k in p:
                    return decode(p[k])
            raise ValueError('no image in message')
        p = base64.b64decode(p.split(',', 1)[-1])
    return Image.open(io.BytesIO(p)).convert('RGB')

async def main():
    pol = PolicyClient(a.url); print('policy:', pol.info()['robot'], flush=True)
    loop = asyncio.get_running_loop()
    state = {'cmd': (0, 0), 'name': 'stop', 'last_frame': 0.0, 'n': 0, 'run': True, 'out': {}, 'go': not a.gate, 'ready': False}
    async with aiohttp.ClientSession() as s:
        async def api(method, path, payload=None, t=3):
            async with s.request(method, f'http://{a.ip}/api/{path}', json=payload, timeout=aiohttp.ClientTimeout(total=t)) as r:
                return await r.json(content_type=None)
        # Always start with the head fully down (+26 deg pitch is Misty's limit), also in --dry runs.
        print('head down:', str(await api('POST', 'head', {'Pitch': 26, 'Roll': 0, 'Yaw': 0, 'Velocity': 60, 'Units': 'degrees'}, 5))[:80], flush=True)
        await asyncio.sleep(1.5)   # let the head settle before the policy sees frames
        pol.reset()

        async def camera():
            # Misty's video service often keeps a dead listener after a previous client left, so every
            # (re)connect restarts it on a fresh port; a stream that goes silent for 5 s is restarted too.
            import random
            while state['run']:
                port = random.randint(10000, 60000)
                try:
                    try:
                        await api('POST', 'videostreaming/stop', {}, 5)
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)
                    r = await api('POST', 'videostreaming/start', {'Port': port, 'Rotation': 0, 'Width': 640, 'Height': 480, 'Quality': 90, 'Overlay': False}, 20)
                    print(f'stream start on port {port}:', str(r)[:80], flush=True)
                    await asyncio.sleep(1.5)
                    async with s.ws_connect(f'ws://{a.ip}:{port}', max_msg_size=0, receive_timeout=5) as ws:
                        async for m in ws:
                            if not state['run']:
                                return
                            if m.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                                continue
                            if not state['go']:                 # standby: the camera is alive, nothing goes to the policy
                                state['last_frame'] = time.monotonic()
                                if not state['ready']:
                                    state['ready'] = True; print('READY (standby: camera frames arriving)', flush=True)
                                continue
                            fr = decode(m.data).transpose(Image.Transpose.ROTATE_270)
                            state['n'] += 1; state['last_frame'] = time.monotonic()
                            if state['n'] == 1:
                                print('first frame', fr.size, flush=True)
                            if a.save and state['n'] in (5,):
                                fr.save(a.save)
                            out = await loop.run_in_executor(None, pol.send_frame, fr)
                            state['out'] = out
                            state['cmd'] = (0, 0) if out.get('stale') else (int(out['v']), int(out['w']))
                            state['name'] = out['name']
                            log_action(state['cmd'], out)        # cmd = (v %, w %)
                    print('camera stream closed, restarting', flush=True)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print('camera error, restarting:', repr(e)[:120], flush=True)
                    await asyncio.sleep(1.0)

        async def drive():
            t0 = time.monotonic(); last_print = 0
            while time.monotonic() - t0 < a.seconds:
                go = gate_open()
                if go and not state['go']:              # standby -> run: start from an empty history
                    await loop.run_in_executor(None, pol.reset)
                    state.update(cmd=(0, 0), name='stop', out={}, n=0)   # n: frames sent in this game (display only)
                    t0 = time.monotonic(); print('GO', flush=True)       # the printed time also restarts per game
                elif state['go'] and not go:            # run -> standby: stop the wheels once
                    if not a.dry:
                        await api('POST', 'drive/stop', {'Hold': False}, 2)
                    print('STANDBY (stop sent)', flush=True)
                state['go'] = go
                if not go:
                    await asyncio.sleep(0.1)
                    continue
                v, w = state['cmd'] if time.monotonic() - state['last_frame'] < 2 else (0, 0)
                if not a.dry:
                    if (v, w) == (0, 0):
                        await api('POST', 'drive/stop', {'Hold': False}, 1)
                    else:
                        await api('POST', 'drive/time', {'LinearVelocity': v, 'AngularVelocity': w, 'TimeMs': 500}, 1)
                if time.monotonic() - last_print > 0.5:
                    last_print = time.monotonic(); o = state['out']
                    print(f"{time.monotonic()-t0:5.1f}s  v={v:+d} w={w:+d} {state['name']:14s} infer {o.get('infer_ms')} ms  age {o.get('frame_age_ms')} ms  frames {state['n']}{'  [DRY]' if a.dry else ''}", flush=True)
                await asyncio.sleep(0.15)
        cam = asyncio.create_task(camera())
        try:
            await drive()
        finally:
            state['run'] = False
            try:
                await api('POST', 'drive/stop', {'Hold': False}, 2); print('STOP sent', flush=True)
            finally:
                cam.cancel()
asyncio.run(main())
