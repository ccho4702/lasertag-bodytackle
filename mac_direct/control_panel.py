"""Local web panel to switch the Misty / Cozmo policy runs on and off.

    .venv/bin/python mac_direct/control_panel.py        ->  http://127.0.0.1:8780

ON  = make sure the policy server for the chosen model is up (starts or swaps it), then start the robot loop
      (misty_live.py / cozmo_live.py).
OFF = interrupt the robot loop (it sends the robot a stop) and, for Misty, send drive/stop once more.
Policy servers stay up between runs so the next ON is quick; "서버 끄기" stops them.

Listens on 127.0.0.1 only. Processes are found by port / script name, so runs started outside the panel
show up here too and can be switched off from it.
"""
import collections
import glob
import json
import os
import re
import shutil
import socket
import signal
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOGS = os.path.join(HERE, 'logs')
PORT = int(os.environ.get('PANEL_PORT', 8780))
MISTY_IP = '192.168.1.38'
V = '../checkpoints/policy/v2026.11.24/'

ROBOTS = {
    'misty': {
        'title': 'Misty', 'role': 'chaser (술래)', 'port': 8770,
        'loop': ['.venv/bin/python', 'mac_direct/misty_live.py', '--seconds', '86400'], 'loop_pat': 'misty_live.py',
        'models': {
            'baseline': {'label': 'baseline (T8K4, 11-24)', 'py': '.venv', 'ckpt': V + 'baseline_chaser_T8K4.pt',
                         'args': ['--device', 'mps']},
            'ours': {'label': 'ours = baseline + SmolVLA (T8K4, 11-24)', 'py': '.venv-vla', 'ckpt': V + 'ours_chaser_T8K4.pt',
                     'args': ['--vla', '../checkpoints/smolvla/chaser', '--device', 'mps'],
                     'needs': 'checkpoints/smolvla/chaser'},
        },
    },
    'cozmo': {
        'title': 'Cozmo', 'role': 'evader (도망자)', 'port': 8771,
        'loop': ['.venv-cozmo/bin/python', 'mac_direct/cozmo_live.py', '--seconds', '86400'], 'loop_pat': 'cozmo_live.py',
        'models': {
            'baseline': {'label': 'baseline (T8K4, 11-24)', 'py': '.venv', 'ckpt': V + 'baseline_evader_T8K4.pt',
                         'args': ['--device', 'cpu'], 'env': {'OMP_NUM_THREADS': '4'}},
            'old': {'label': 'baseline 구버전 (11-08, 간격 8)', 'py': '.venv', 'ckpt': '../checkpoints/policy/baseline_cpu_evader.pt',
                    'args': ['--device', 'cpu'], 'env': {'OMP_NUM_THREADS': '4'}},
        },
    },
}

LINE = re.compile(r'^\s*([\d.]+)s\s+(\S+) (\S+) (\S+)\s+infer (\S+) ms\s+age (\S+) ms\s+frames (\d+)')
state = {r: {'phase': '', 'error': '', 'busy': False} for r in ROBOTS}
# Health check: once a second the monitor asks each policy server for its newest command and keeps 60 s of samples.
#   bad  = the server is forcing a stop (command expired because inference is too slow, or no camera frames)
#   slow = inference takes longer than the robot's budget (the policy sees a different frame spacing than in training)
hist = {r: collections.deque(maxlen=60) for r in ROBOTS}
loop_since = {r: None for r in ROBOTS}
settings = {'autostop': False}
alert = {'msg': ''}
# Bird's-eye (overhead) camera: the Jetson serves /stream.mjpg and /snapshot.jpg (LSMA-Laser-tag jetson_camera_stream.py).
# Recording copies the MJPEG stream to a file without decoding it (ffmpeg -c copy), so it costs almost no CPU while
# the robots run. Hit detection (YOLO) runs on the saved file afterwards, and only while no robot loop is running.
OVERHEAD = os.environ.get('OVERHEAD_URL', 'http://192.168.1.37:8088').rstrip('/')
OH_DIR = os.path.join(HERE, 'overhead_records')
OH_MAX_S = 600
oh = {'proc': None, 'file': '', 'since': 0.0, 'reachable': False, 'checked': 0.0, 'analyzing': '', 'error': ''}
GRACE_S = 6          # ignore the start-up seconds of a run
AUTOSTOP_BAD_S, AUTOSTOP_SLOW_S = 3, 10
cache = {'robots': {}, 'misty_net': None, 'adb': '?', 't': 0}
lock = threading.Lock()


def sh(cmd, timeout=4):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ''


def http_json(url, data=None, timeout=1.5):
    try:
        req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def port_pid(port):
    """pid of the policy server started with `--port <port>`, from the process list.

    Do NOT use lsof here: the monitor calls this every second, and lsof walking every process's file
    descriptors twice a second slowed both policy servers about 5x (evader 72 ms -> 600 ms per frame,
    measured 2026-11-24 with replayed frames; with this panel stopped the servers were back to normal).
    """
    for ln in sh(['ps', '-axo', 'pid=,command=']).splitlines():
        parts = ln.split()
        if 'policy_server.py' in parts and '--port' in parts and parts[parts.index('--port') + 1:][:1] == [str(port)]:
            return int(parts[0])
    return None


def server_model(robot):
    """(pid, model key or '?') of the policy server listening on the robot's port."""
    pid = port_pid(ROBOTS[robot]['port'])
    if pid is None:
        return None, None
    cmd = sh(['ps', '-o', 'command=', '-p', str(pid)])
    for k, m in ROBOTS[robot]['models'].items():
        if os.path.basename(m['ckpt']) in cmd:
            return pid, k
    return pid, '?'


def loop_pids(robot):
    # Only real `python .../misty_live.py` processes: a shell whose command line merely mentions the
    # script name (which `pgrep -f` would also match) must never be reported or interrupted.
    pids = []
    for ln in sh(['ps', '-axo', 'pid=,command=']).splitlines():
        parts = ln.split()
        if len(parts) >= 3 and 'python' in os.path.basename(parts[1]).lower() \
                and os.path.basename(parts[2]) == ROBOTS[robot]['loop_pat']:
            pids.append(int(parts[0]))
    return pids


def log_path(robot, kind):
    return os.path.join(LOGS, '%s_%s.log' % (robot, kind))


def tail(path, n=4000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode(errors='replace')
    except OSError:
        return ''


def monitor():
    while True:
        robots = {}
        for r, cfg in ROBOTS.items():
            pid, model = server_model(r)
            info = http_json('http://127.0.0.1:%d/info' % cfg['port'], timeout=1) if pid else None
            lines = [ln for ln in tail(log_path(r, 'loop')).splitlines() if ln.strip() and 'LightCube' not in ln]
            last = None
            for ln in reversed(lines):
                m = LINE.match(ln)
                if m:
                    last = {'t': m.group(1), 'cmd': m.group(4), 'a': m.group(2), 'b': m.group(3),
                            'infer': m.group(5), 'age': m.group(6), 'frames': m.group(7)}
                    break
            running = bool(loop_pids(r))
            robots[r] = {'server_pid': pid, 'server_model': model, 'server_ready': bool(info),
                         'budget': (info or {}).get('budget_ms'), 'vla': (info or {}).get('vla'),
                         'loop': running, 'last': last, 'log': [ln[:150] for ln in lines[-6:]],
                         'health': health_of(r, running, info)}
        bat = http_json('http://%s/api/battery' % MISTY_IP, timeout=1.5)
        res = (bat or {}).get('result') or {}
        misty_net = {'pct': round(res.get('chargePercent', 0) * 100), 'charging': bool(res.get('isCharging'))} if bat else None
        adb = cache['adb']
        if time.time() - monitor.adb_t > 5:            # the tablet state changes rarely: ask adb every 5 s only
            monitor.adb_t, adb = time.time(), '없음'
            for ln in sh(['adb', 'devices']).splitlines()[1:]:
                if '\t' in ln:
                    adb = ln.split('\t')[1].strip()
        if time.time() - oh['checked'] > 3:
            oh['checked'] = time.time()
            host, _, port = OVERHEAD.split('://', 1)[1].partition(':')
            try:
                socket.create_connection((host, int(port or 80)), timeout=0.5).close()
                oh['reachable'] = True
            except OSError:
                oh['reachable'] = False
        if oh['proc'] is not None and oh['proc'].poll() is not None:      # ffmpeg ended by itself (stream lost / time limit)
            finish_recording()
        with lock:
            cache.update(robots=robots, misty_net=misty_net, adb=adb, t=time.time())
        if settings['autostop']:
            for r, info in robots.items():
                h = info['health']
                if h and ((h['level'] == 'bad' and h['streak_s'] >= AUTOSTOP_BAD_S)
                          or (h['level'] == 'slow' and h['streak_s'] >= AUTOSTOP_SLOW_S)):
                    alert['msg'] = '%s 자동 정지: %s — %s (%d초 지속). 이 경기는 무효로 보세요.' % (
                        time.strftime('%H:%M:%S'), ROBOTS[r]['title'], h['reason'], h['streak_s'])
                    for x in ROBOTS:
                        threading.Thread(target=stop_loop, args=(x,), daemon=True).start()
                    break
        time.sleep(1.0)


monitor.adb_t = 0.0


def health_of(robot, running, info):
    """Health of a running robot from the policy server's newest command; None when the robot is off or starting."""
    now = time.time()
    if not running:
        loop_since[robot] = None
        return None
    if loop_since[robot] is None:
        loop_since[robot] = now
        hist[robot].clear()
    if now - loop_since[robot] < GRACE_S:
        return None
    budget = (info or {}).get('budget_ms') or 0
    cmd = http_json('http://127.0.0.1:%d/command' % ROBOTS[robot]['port'], timeout=1) if info else None
    infer = (cmd or {}).get('infer_ms')
    if cmd is None:
        level, reason = 'bad', '정책 서버가 응답하지 않음'
    elif cmd.get('stale'):
        slow = infer is not None and budget and infer > budget
        level, reason = 'bad', ('추론이 너무 느려 명령이 만료됨 → 강제 정지 중' if slow
                                else '카메라 프레임이 안 들어옴 → 강제 정지 중')
    elif infer is not None and budget and infer > budget:
        level, reason = 'slow', '추론이 예산을 넘음 (명령은 나가지만 학습 때와 프레임 간격이 다름)'
    else:
        level, reason = 'ok', '정상'
    hist[robot].append({'t': now, 'level': level, 'infer': infer})
    samples = list(hist[robot])
    recent = [s for s in samples if s['t'] > now - 30]
    infs = sorted(s['infer'] for s in recent if s['infer'] is not None)
    first = now
    for s in reversed(samples):                 # how long the current level has lasted
        if s['level'] != level:
            break
        first = s['t']
    return {'level': level, 'reason': reason, 'streak_s': int(now - first), 'infer': infer, 'budget': budget,
            'age': (cmd or {}).get('frame_age_ms'), 'fps': (cmd or {}).get('fps'), 'target_fps': (info or {}).get('infer_fps'),
            'median30': infs[len(infs) // 2] if infs else None, 'max30': infs[-1] if infs else None,
            'bad_pct30': round(100 * sum(s['level'] == 'bad' for s in recent) / len(recent)),
            'slow_pct30': round(100 * sum(s['level'] == 'slow' for s in recent) / len(recent)), 'n30': len(recent)}


def stop_loop(robot):
    pids = loop_pids(robot)
    for p in pids:
        try:
            os.kill(p, signal.SIGINT)        # both loops send the robot a stop when interrupted
        except OSError:
            pass
    t0 = time.time()
    while loop_pids(robot) and time.time() - t0 < 6:
        time.sleep(0.2)
    for p in loop_pids(robot):
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    if robot == 'misty':                     # belt and braces: Misty also gets a direct stop
        http_json('http://%s/api/drive/stop' % MISTY_IP, {'Hold': False}, timeout=2)
    return bool(pids)


def stop_server(robot):
    pid = port_pid(ROBOTS[robot]['port'])
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        t0 = time.time()
        while port_pid(ROBOTS[robot]['port']) and time.time() - t0 < 10:
            time.sleep(0.3)


def turn_on(robot, model):
    st, cfg = state[robot], ROBOTS[robot]
    m = cfg['models'][model]
    try:
        if robot == 'misty' and http_json('http://%s/api/battery' % MISTY_IP, timeout=3) is None:
            raise RuntimeError('Misty(%s)에 연결되지 않습니다. 전원과 Wi-Fi를 확인하세요.' % MISTY_IP)
        if robot == 'cozmo' and cache['adb'] != 'device':
            raise RuntimeError('태블릿이 adb에 %s 상태입니다. USB 연결과 "USB 디버깅 허용"을 확인하세요.' % cache['adb'])
        if m.get('needs') and not os.path.isdir(os.path.join(REPO, m['needs'])):
            raise RuntimeError('%s 가 없습니다. bash scripts/download_smolvla.sh 로 먼저 받으세요.' % m['needs'])
        stop_loop(robot)
        pid, cur = server_model(robot)
        if cur != model:
            if pid:
                st['phase'] = '이전 서버 종료 중'
                stop_server(robot)
            st['phase'] = '정책 서버 시작 중 (20~60초)'
            cmd = [os.path.join(REPO, m['py'], 'bin', 'python'), 'policy_server.py', '--ckpt', m['ckpt'],
                   '--port', str(cfg['port'])] + m['args']
            if m.get('args') and '--vla' in m['args']:
                pass
            env = dict(os.environ, **m.get('env', {}))
            log = open(log_path(robot, 'server'), 'w')
            proc = subprocess.Popen(cmd, cwd=os.path.join(REPO, 'baseline'), env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            t0 = time.time()
            while http_json('http://127.0.0.1:%d/info' % cfg['port'], timeout=1) is None:
                if proc.poll() is not None:
                    raise RuntimeError('정책 서버가 시작 중 종료됐습니다: ' + tail(log_path(robot, 'server'), 300).strip().splitlines()[-1])
                if time.time() - t0 > 240:
                    raise RuntimeError('정책 서버가 240초 안에 준비되지 않았습니다.')
                time.sleep(1)
        st['phase'] = '로봇 루프 시작 중'
        log = open(log_path(robot, 'loop'), 'w')
        proc = subprocess.Popen([os.path.join(REPO, cfg['loop'][0])] + cfg['loop'][1:], cwd=REPO, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(4)
        if proc.poll() is not None:
            lines = tail(log_path(robot, 'loop'), 400).strip().splitlines()
            raise RuntimeError('로봇 루프가 바로 종료됐습니다: ' + (lines[-1] if lines else ''))
    except Exception as exc:
        st['error'] = str(exc)
    finally:
        st['phase'] = ''
        st['busy'] = False


def turn_off(robot):
    st = state[robot]
    try:
        st['phase'] = '정지 중'
        stop_loop(robot)
    except Exception as exc:
        st['error'] = str(exc)
    finally:
        st['phase'] = ''
        st['busy'] = False


def ffmpeg_exe():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    found = glob.glob(os.path.join(REPO, '.venv', 'lib', 'python*', 'site-packages', 'imageio_ffmpeg', 'binaries', 'ffmpeg*'))
    return found[0] if found else None


def start_recording():
    oh['error'] = ''
    if oh['proc'] is not None:
        return
    if not oh['reachable']:
        oh['error'] = '천장 카메라 스트림(%s)에 연결되지 않습니다. Jetson에서 스트림 프로그램이 켜져 있는지 확인하세요.' % OVERHEAD
        return
    exe = ffmpeg_exe()
    if not exe:
        oh['error'] = 'ffmpeg 을 찾지 못했습니다.'
        return
    os.makedirs(OH_DIR, exist_ok=True)
    # Raw MJPEG first (frames only, no timestamps); finish_recording() wraps it into an .mkv with the measured frame rate.
    path = os.path.join(OH_DIR, time.strftime('%Y%m%d_%H%M%S') + '.mjpeg')
    log = open(os.path.join(LOGS, 'overhead_record.log'), 'w')
    oh['proc'] = subprocess.Popen([exe, '-y', '-loglevel', 'error', '-use_wallclock_as_timestamps', '1', '-f', 'mjpeg',
                                   '-i', OVERHEAD + '/stream.mjpg', '-c', 'copy', '-t', str(OH_MAX_S), '-f', 'mjpeg', path],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    oh['file'], oh['since'] = path, time.time()


def finish_recording():
    proc, path = oh['proc'], oh['file']
    if proc is None:
        return
    oh['proc'] = None
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)          # lets ffmpeg close the file properly
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    elapsed = time.time() - oh['since']
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        oh['error'] = '녹화 파일이 비어 있습니다: ' + tail(os.path.join(LOGS, 'overhead_record.log'), 300).strip()[-200:]
        return
    # The stream carries no reliable timestamps, so the frame rate is measured: frames received / seconds recorded.
    # hit_detect then turns frame numbers into seconds with this rate (time to first hit).
    frames = 0
    with open(path, 'rb') as f:
        prev = b''
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            frames += (prev + chunk).count(b'\xff\xd8\xff')
            prev = chunk[-2:]
    fps = frames / max(elapsed, 0.1)
    mkv = os.path.splitext(path)[0] + '.mkv'
    rc = subprocess.call([ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '%.4f' % fps, '-f', 'mjpeg', '-i', path,
                          '-c', 'copy', mkv], stdin=subprocess.DEVNULL)
    if rc != 0 or not os.path.exists(mkv):
        oh['error'] = '녹화 파일 변환 실패. 원본은 남아 있습니다: ' + path
        return
    os.remove(path)
    json.dump({'frames': frames, 'seconds': round(elapsed, 2), 'fps': round(fps, 3), 'source': OVERHEAD + '/stream.mjpg',
               'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(oh['since']))},
              open(os.path.splitext(path)[0] + '.meta.json', 'w'))
    oh['file'] = mkv
    threading.Thread(target=analyze, args=(mkv,), daemon=True).start()


def analyze(path):
    """Overhead YOLO hit detection on a saved recording -> <name>.json (time to first hit) + annotated <name>.mp4."""
    name = os.path.basename(path)
    while any(loop_pids(r) for r in ROBOTS):       # never compete with the policies for the GPU/CPU
        oh['analyzing'] = name + ' (로봇이 꺼지면 분석 시작)'
        time.sleep(2)
    oh['analyzing'] = name + ' (히트 판정 중)'
    log = open(os.path.join(LOGS, 'overhead_analyze.log'), 'w')
    rc = subprocess.call([os.path.join(REPO, '.venv', 'bin', 'python'), '-m', 'overhead.hit_detect',
                          '--weights', '../checkpoints/yolo/overhead_yolo_v1.pt', '--video', path, '--fps', '10', '--out', OH_DIR],
                         cwd=os.path.join(REPO, 'baseline'), stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        oh['error'] = '히트 판정 실패 (%s): %s' % (name, tail(os.path.join(LOGS, 'overhead_analyze.log'), 300).strip()[-200:])
    oh['analyzing'] = ''


def overhead_status():
    recs = []
    for f in sorted(glob.glob(os.path.join(OH_DIR, '*.mkv')), reverse=True)[:8]:
        base = os.path.splitext(f)[0]
        item = {'file': f, 'mb': round(os.path.getsize(f) / 1e6, 1), 'analyzed': False, 'hit': None, 'duration': None,
                'annotated': base + '.mp4' if os.path.exists(base + '.mp4') else ''}
        if os.path.exists(base + '.json'):
            try:
                summ = json.load(open(base + '.json'))['summary']
                item.update(analyzed=True, hit=summ.get('time_to_first_hit_s'), duration=summ.get('duration_s'))
            except Exception:
                pass
        recs.append(item)
    return {'url': OVERHEAD, 'reachable': oh['reachable'], 'recording': oh['proc'] is not None, 'file': oh['file'],
            'elapsed': int(time.time() - oh['since']) if oh['proc'] is not None else 0, 'dir': OH_DIR,
            'analyzing': oh['analyzing'], 'error': oh['error'], 'recent': recs, 'max_s': OH_MAX_S}


def start_action(robot, fn, *args):
    st = state[robot]
    with lock:
        if st['busy']:
            return False
        st['busy'], st['error'] = True, ''
    threading.Thread(target=fn, args=(robot,) + args, daemon=True).start()
    return True


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>로봇 조작 패널</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#16181d;--mute:#667085;--line:#e3e6eb;--on:#12805c;--off:#b42318;--warn:#b54708;--chip:#eef1f5;--btn:#16181d;--btnink:#fff}
@media (prefers-color-scheme: dark){:root{--bg:#111318;--card:#1a1d24;--ink:#eceef2;--mute:#98a2b3;--line:#2a2f3a;--on:#3ccb93;--off:#f97066;--warn:#fdb022;--chip:#242833;--btn:#eceef2;--btnink:#111318}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 40px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:18px}
h1{font-size:20px;margin:0}h2{font-size:18px;margin:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.role{color:var(--mute);font-size:13px}
.pill{font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px;background:var(--chip);color:var(--mute)}
.pill.on{background:color-mix(in srgb,var(--on) 16%,transparent);color:var(--on)}
.pill.busy{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
label{display:block;font-size:12px;color:var(--mute);margin-bottom:4px}
select{width:100%;padding:9px 10px;border-radius:9px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit}
button{font:inherit;font-weight:600;border:0;border-radius:10px;padding:11px 16px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.toggle{width:100%;margin-top:12px;font-size:16px;padding:14px;background:var(--btn);color:var(--btnink)}
.toggle.stop{background:var(--off);color:#fff}
.ghost{background:var(--chip);color:var(--ink)}.danger{background:var(--off);color:#fff}
dl{display:grid;grid-template-columns:92px 1fr;gap:4px 10px;margin:14px 0 0;font-size:13.5px}
dt{color:var(--mute)}dd{margin:0;font-variant-numeric:tabular-nums}
.ok{color:var(--on)}.bad{color:var(--off)}.warn{color:var(--warn)}
.err{margin-top:12px;padding:10px 12px;border-radius:9px;background:color-mix(in srgb,var(--off) 12%,transparent);color:var(--off);font-size:13.5px}
pre{margin:12px 0 0;padding:10px;border-radius:9px;background:var(--chip);font:11.5px/1.45 ui-monospace,Menlo,monospace;overflow-x:auto;white-space:pre;min-height:96px}
.banner{margin-bottom:16px;padding:14px 16px;border-radius:12px;font-weight:600;display:flex;justify-content:space-between;align-items:center;gap:12px}
.banner.bad{background:var(--off);color:#fff}.banner.slow{background:color-mix(in srgb,var(--warn) 22%,var(--card));color:var(--ink);border:1px solid var(--warn)}
.banner button{padding:6px 12px;background:rgba(255,255,255,.25);color:inherit}
.health{margin-top:14px;padding:12px;border-radius:10px;background:var(--chip)}
.health.bad{background:color-mix(in srgb,var(--off) 14%,transparent)}.health.slow{background:color-mix(in srgb,var(--warn) 16%,transparent)}
.health .hl{font-weight:700}.health.ok .hl{color:var(--on)}.health.bad .hl{color:var(--off)}.health.slow .hl{color:var(--warn)}
.health .sub{font-size:12.5px;color:var(--mute);margin-top:4px;font-variant-numeric:tabular-nums}
.meter{position:relative;height:8px;border-radius:4px;background:var(--line);margin-top:10px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--on)}.meter.slow i{background:var(--warn)}.meter.bad i{background:var(--off)}
.meterwrap{position:relative}.mark{position:absolute;left:50%;top:6px;width:2px;height:16px;background:var(--ink);opacity:.55}
.scale{display:flex;justify-content:space-between;font-size:11px;color:var(--mute);margin-top:3px}
.wide{grid-column:1/-1}.oh{display:grid;grid-template-columns:minmax(260px,1.2fr) 1fr;gap:16px;align-items:start}
@media (max-width:760px){.oh{grid-template-columns:1fr}}
.view{background:#000;border-radius:10px;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:#98a2b3;font-size:13px;overflow:hidden;text-align:center;padding:0}
.view img{width:100%;height:100%;object-fit:contain;display:block}
.path{font:12px/1.4 ui-monospace,Menlo,monospace;word-break:break-all;background:var(--chip);padding:8px 10px;border-radius:8px;margin-top:6px;user-select:all}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600}td.f{font:11.5px/1.4 ui-monospace,Menlo,monospace;word-break:break-all;user-select:all}
.rec{background:var(--off);color:#fff}
.chk{display:flex;align-items:center;gap:6px;font-size:13.5px;color:var(--ink);margin:0}.chk input{width:16px;height:16px}
.hdr{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
footer{margin-top:18px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;color:var(--mute);font-size:13px}
</style></head><body><main>
<header><h1>로봇 조작 패널</h1><div class="hdr">
  <label class="chk" title="강제 정지가 3초, 예산 초과가 10초 이어지면 두 로봇을 모두 멈춥니다"><input type="checkbox" id="autostop">문제 생기면 자동 정지</label>
  <button class="danger" id="alloff">모두 정지</button></div></header>
<div class="banner" id="banner" hidden><span id="bannertext"></span><button id="bannerx" hidden>닫기</button></div>
<div class="grid" id="grid"></div>
<div class="grid" style="margin-top:16px"><section class="card wide">
  <div class="top"><div><h2>Bird's-eye view</h2><div class="role">천장 카메라 · <span id="oh-url"></span></div></div><span class="pill" id="oh-pill">–</span></div>
  <div class="oh"><div class="view" id="oh-view">연결 확인 중…</div>
  <div><button class="toggle" id="oh-btn" style="margin-top:0">녹화 시작</button>
    <div class="err" id="oh-err" hidden></div>
    <label style="margin-top:12px">저장 폴더</label><div class="path" id="oh-dir"></div>
    <label style="margin-top:10px">지금 / 마지막 녹화 파일</label><div class="path" id="oh-file">–</div>
    <div class="role" id="oh-note" style="margin-top:8px"></div></div></div>
  <table><thead><tr><th>녹화 파일 (원본)</th><th>크기</th><th>길이</th><th>첫 히트</th><th>판정 영상</th></tr></thead><tbody id="oh-rows"></tbody></table>
</section></div>
<footer><span id="foot">연결 중…</span><button class="ghost" id="srvoff">정책 서버 끄기</button></footer>
</main>
<script>
const ROBOTS = __ROBOTS__;
const grid = document.getElementById('grid');
for (const [key, r] of Object.entries(ROBOTS)) {
  const opts = Object.entries(r.models).map(([k, label]) => `<option value="${k}">${label}</option>`).join('');
  grid.insertAdjacentHTML('beforeend', `<section class="card" id="c-${key}">
    <div class="top"><div><h2>${r.title}</h2><div class="role">${r.role}</div></div><span class="pill" id="pill-${key}">꺼짐</span></div>
    <label for="sel-${key}">모델</label><select id="sel-${key}">${opts}</select>
    <button class="toggle" id="btn-${key}">켜기</button>
    <div class="err" id="err-${key}" hidden></div>
    <div class="health" id="h-${key}"><div class="hl" id="hl-${key}">꺼져 있음</div><div class="sub" id="hs-${key}">켜면 추론 속도를 1초마다 점검합니다</div>
      <div class="meterwrap"><div class="meter" id="hm-${key}"><i style="width:0"></i></div><div class="mark"></div></div>
      <div class="scale"><span>0</span><span id="hb-${key}">예산</span><span>예산의 2배</span></div></div>
    <dl><dt>로봇 연결</dt><dd id="net-${key}">–</dd><dt>정책 서버</dt><dd id="srv-${key}">–</dd>
        <dt>마지막 명령</dt><dd id="cmd-${key}">–</dd><dt>추론 / 지연</dt><dd id="lat-${key}">–</dd></dl>
    <pre id="log-${key}"></pre></section>`);
  document.getElementById('btn-' + key).onclick = () => {
    const on = document.getElementById('btn-' + key).dataset.on === '1';
    post(on ? '/api/off' : '/api/on', {robot: key, model: document.getElementById('sel-' + key).value});
  };
}
document.getElementById('alloff').onclick = () => post('/api/off_all', {});
document.getElementById('autostop').onchange = (e) => post('/api/settings', {autostop: e.target.checked});
document.getElementById('oh-btn').onclick = () => post(document.getElementById('oh-btn').dataset.on === '1' ? '/api/overhead/stop' : '/api/overhead/start', {});
let ohTimer = null, ohUrl = '';
function overhead(o) {
  ohUrl = o.url; document.getElementById('oh-url').textContent = o.url;
  const pill = document.getElementById('oh-pill'), btn = document.getElementById('oh-btn'), view = document.getElementById('oh-view');
  pill.textContent = o.recording ? `녹화 중 ${o.elapsed}초` : (o.reachable ? '연결됨' : '연결 안 됨');
  pill.className = 'pill' + (o.recording ? ' busy' : o.reachable ? ' on' : '');
  btn.dataset.on = o.recording ? '1' : '0'; btn.textContent = o.recording ? '녹화 중지' : '녹화 시작';
  btn.className = 'toggle' + (o.recording ? ' rec' : ''); btn.disabled = !o.recording && !o.reachable;
  const err = document.getElementById('oh-err'); err.hidden = !o.error; err.textContent = o.error;
  document.getElementById('oh-dir').textContent = o.dir; document.getElementById('oh-file').textContent = o.file || '–';
  document.getElementById('oh-note').textContent = o.analyzing ? '분석: ' + o.analyzing : `녹화는 최대 ${o.max_s}초. 중지하면 로봇이 꺼진 뒤 히트 판정을 자동으로 돌립니다.`;
  if (o.reachable) {
    if (!view.querySelector('img')) { view.innerHTML = '<img alt="천장 카메라 실시간 화면">'; }
    if (!ohTimer) ohTimer = setInterval(() => { const im = view.querySelector('img'); if (im && !document.hidden) im.src = ohUrl + '/snapshot.jpg?t=' + Date.now(); }, 400);
  } else {
    if (ohTimer) { clearInterval(ohTimer); ohTimer = null; }
    view.textContent = '천장 카메라 스트림에 연결되지 않습니다. Jetson에서 스트림 프로그램을 켜 주세요.';
  }
  document.getElementById('oh-rows').innerHTML = o.recent.length ? o.recent.map(r => `<tr><td class="f">${r.file}</td><td>${r.mb} MB</td>
    <td>${r.duration != null ? r.duration.toFixed(1) + '초' : '–'}</td>
    <td>${r.analyzed ? (r.hit != null ? r.hit.toFixed(2) + '초' : '히트 없음') : '미분석'}</td><td class="f">${r.annotated || '–'}</td></tr>`).join('')
    : '<tr><td colspan="5" style="color:var(--mute)">아직 녹화가 없습니다</td></tr>';
}
document.getElementById('bannerx').onclick = () => post('/api/clear_alert', {});
function health(key, r) {
  const h = r.health, box = document.getElementById('h-' + key), bar = document.getElementById('hm-' + key);
  if (!h) {
    box.className = 'health'; bar.className = 'meter'; bar.firstChild.style.width = '0';
    document.getElementById('hl-' + key).textContent = r.loop ? '시작 중 (점검 대기)' : '꺼져 있음';
    document.getElementById('hs-' + key).textContent = r.loop ? '시작 후 6초부터 점검합니다' : '켜면 추론 속도를 1초마다 점검합니다';
    return null;
  }
  box.className = 'health ' + h.level; bar.className = 'meter ' + h.level;
  bar.firstChild.style.width = Math.min(100, (h.infer || 0) / (h.budget || 1) * 50) + '%';
  document.getElementById('hb-' + key).textContent = '예산 ' + h.budget + ' ms';
  document.getElementById('hl-' + key).textContent = (h.level === 'ok' ? '정상' : h.reason) + (h.level !== 'ok' ? ` · ${h.streak_s}초째` : '');
  document.getElementById('hs-' + key).textContent =
    `지금 추론 ${h.infer ?? '–'} ms · 최근 30초 중앙값 ${h.median30 ?? '–'} ms, 최대 ${h.max30 ?? '–'} ms · 강제 정지 ${h.bad_pct30}% · 예산 초과 ${h.slow_pct30}% · 추론 속도 ${h.fps ?? '–'}/${h.target_fps ?? '–'} fps`;
  return h;
}
document.getElementById('srvoff').onclick = () => { if (confirm('정책 서버를 끄면 다음에 켤 때 20~60초가 걸립니다. 끌까요?')) post('/api/servers_off', {}); };
async function post(url, body) {
  try { await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}); } catch (e) {}
  refresh();
}
function set(id, text, cls) { const el = document.getElementById(id); el.textContent = text; el.className = cls || ''; }
async function refresh() {
  let s;
  try { s = await (await fetch('/api/status')).json(); }
  catch (e) { document.getElementById('foot').textContent = '패널 서버에 연결되지 않습니다 (control_panel.py 가 꺼졌는지 확인)'; return; }
  document.getElementById('foot').textContent = '패널 정상 · ' + new Date(s.t * 1000).toLocaleTimeString();
  document.getElementById('autostop').checked = s.autostop;
  if (s.overhead) overhead(s.overhead);
  const problems = [];
  for (const key of Object.keys(ROBOTS)) {
    const r = s.robots[key]; if (!r) continue;
    const h = health(key, r);
    if (h && h.level !== 'ok') problems.push({level: h.level, text: `${ROBOTS[key].title}: ${h.reason} (${h.streak_s}초째, 추론 ${h.infer ?? '–'} ms / 예산 ${h.budget} ms)`});
    const pill = document.getElementById('pill-' + key), btn = document.getElementById('btn-' + key), sel = document.getElementById('sel-' + key);
    pill.textContent = r.busy ? (r.phase || '처리 중') : (r.loop ? '주행 중' : '꺼짐');
    pill.className = 'pill' + (r.busy ? ' busy' : r.loop ? ' on' : '');
    btn.dataset.on = r.loop ? '1' : '0'; btn.textContent = r.loop ? '끄기' : '켜기';
    btn.className = 'toggle' + (r.loop ? ' stop' : ''); btn.disabled = r.busy;
    sel.disabled = r.busy || r.loop;
    if ((r.busy || r.loop) && r.server_model && r.server_model !== '?' && !r.phase.includes('서버')) sel.value = r.server_model;
    const err = document.getElementById('err-' + key); err.hidden = !r.error; err.textContent = r.error;
    if (key === 'misty') s.misty_net ? set('net-misty', `연결됨 · 배터리 ${s.misty_net.pct}%${s.misty_net.charging ? ' (충전 중 — 패드 위에서는 안 움직임)' : ''}`, s.misty_net.charging || s.misty_net.pct < 15 ? 'warn' : 'ok') : set('net-misty', '연결 안 됨', 'bad');
    else set('net-cozmo', s.adb === 'device' ? '태블릿 연결됨' : '태블릿 ' + s.adb, s.adb === 'device' ? 'ok' : 'bad');
    const name = r.server_model && ROBOTS[key].models[r.server_model] ? ROBOTS[key].models[r.server_model] : r.server_model;
    r.server_pid ? set('srv-' + key, (r.server_ready ? '준비됨 · ' : '시작 중 · ') + name, r.server_ready ? 'ok' : 'warn') : set('srv-' + key, '꺼짐', '');
    if (r.loop && r.last) { set('cmd-' + key, `${r.last.cmd}  (${r.last.a} ${r.last.b}) · ${r.last.t}s 경과`); set('lat-' + key, `${r.last.infer} ms / ${r.last.age} ms` + (r.budget ? ` · 예산 ${r.budget} ms` : '')); }
    else { set('cmd-' + key, '–'); set('lat-' + key, '–'); }
    document.getElementById('log-' + key).textContent = r.log.join('\n');
  }
  const banner = document.getElementById('banner'), bad = problems.some(p => p.level === 'bad');
  banner.hidden = !problems.length && !s.alert;
  banner.className = 'banner ' + (bad || (s.alert && !problems.length) ? 'bad' : 'slow');
  document.getElementById('bannertext').textContent = problems.length ? '⚠ ' + problems.map(p => p.text).join('  /  ') : s.alert;
  document.getElementById('bannerx').hidden = !!problems.length || !s.alert;
  document.title = (problems.length ? (bad ? '⛔ ' : '⚠ ') : '') + '로봇 조작 패널';
}
refresh(); setInterval(refresh, 1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/':
            robots = {k: {'title': v['title'], 'role': v['role'], 'models': {mk: mv['label'] for mk, mv in v['models'].items()}}
                      for k, v in ROBOTS.items()}
            self.send(200, PAGE.replace('__ROBOTS__', json.dumps(robots, ensure_ascii=False)), 'text/html')
        elif self.path == '/api/status':
            with lock:
                out = json.loads(json.dumps(cache))
            for r in ROBOTS:
                out['robots'].setdefault(r, {'loop': False, 'log': [], 'server_pid': None, 'server_model': None,
                                             'server_ready': False, 'last': None})
                out['robots'][r].setdefault('health', None)
                out['robots'][r].update(state[r])
            out.update(autostop=settings['autostop'], alert=alert['msg'], overhead=overhead_status())
            self.send(200, json.dumps(out))
        else:
            self.send(404, '{}')

    def do_POST(self):
        # Only this page may switch robots: same-origin JSON requests from a local browser tab.
        origin = self.headers.get('Origin', '')
        ok_origins = ('http://127.0.0.1:%d' % PORT, 'http://localhost:%d' % PORT)
        if origin not in ok_origins or 'application/json' not in self.headers.get('Content-Type', ''):
            return self.send(403, '{"error": "forbidden"}')
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0)) or 0) or b'{}')
        except ValueError:
            return self.send(400, '{"error": "bad json"}')
        robot, model = body.get('robot'), body.get('model')
        if self.path == '/api/on' and robot in ROBOTS and model in ROBOTS[robot]['models']:
            start_action(robot, turn_on, model)
        elif self.path == '/api/off' and robot in ROBOTS:
            start_action(robot, turn_off)
        elif self.path == '/api/overhead/start':
            start_recording()
        elif self.path == '/api/overhead/stop':
            threading.Thread(target=finish_recording, daemon=True).start()
        elif self.path == '/api/settings':
            settings['autostop'] = bool(body.get('autostop'))
        elif self.path == '/api/clear_alert':
            alert['msg'] = ''
        elif self.path == '/api/off_all':
            for r in ROBOTS:
                threading.Thread(target=stop_loop, args=(r,), daemon=True).start()
        elif self.path == '/api/servers_off':
            for r in ROBOTS:
                if not loop_pids(r):
                    threading.Thread(target=stop_server, args=(r,), daemon=True).start()
        else:
            return self.send(400, '{"error": "bad request"}')
        self.send(200, '{"ok": true}')


if __name__ == '__main__':
    os.makedirs(LOGS, exist_ok=True)
    if not os.path.exists(os.path.join(LOGS, '.gitignore')):
        open(os.path.join(LOGS, '.gitignore'), 'w').write('*\n')
    threading.Thread(target=monitor, daemon=True).start()
    print('control panel: http://127.0.0.1:%d' % PORT, flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
