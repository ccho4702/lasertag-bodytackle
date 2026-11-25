"""Local web panel to switch the Misty / Cozmo policy runs on and off.

    .venv/bin/python mac_direct/control_panel.py        ->  http://127.0.0.1:8780

ON  = make sure the policy server for the chosen model is up (starts or swaps it), then start the robot loop
      (misty_live.py / cozmo_live.py).
OFF = interrupt the robot loop (it sends the robot a stop) and, for Misty, send drive/stop once more.
Policy servers stay up between runs so the next ON is quick; "서버 끄기" stops them.

Game (one button): prepares both robots in standby (servers up, loops connected, camera frames arriving, wheels
stopped), and only when Misty, Cozmo and the overhead camera are all ready starts the recording and lets both
loops drive at the same moment (gate file, see misty_live.py --gate). Pause = robots back to standby + recording
closed; resume = new recording + drive at once. Full stop = loops and recording ended.

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
OH_DIR_DEFAULT = os.path.join(HERE, 'overhead_records')
OH_DIR_FILE = os.path.join(HERE, 'logs', 'overhead_dir.txt')     # folder chosen in the panel; survives a panel restart
OH_MAX_S = 600
# SmolVLA instruction typed in the panel ('' = the one it was trained with). Kept in a file so it is applied again
# whenever the "ours" server is started.
VLA_TASK_FILE = os.path.join(HERE, 'logs', 'vla_task.txt')
vla = {'task': '', 'error': ''}
oh = {'proc': None, 'file': '', 'since': 0.0, 'reachable': False, 'checked': 0.0, 'analyzing': '', 'error': '', 'dir': OH_DIR_DEFAULT, 'go_t': None}
# Game session. state: off -> preparing -> ready (standby) <-> running. The gate file is what the two robot loops
# watch: "run" = drive, anything else = standby.
GATE = os.path.join(LOGS, 'gate.txt')
DRY = ['--dry'] if os.environ.get('PANEL_DRY') else []       # rehearsal: loops run but send no drive commands
# robots: the ones that take part (only Misty when the evader is kept still); models: what they run.
sess = {'state': 'off', 'phase': '', 'error': '', 'busy': False, 'cancel': False, 'since': 0.0,
        'robots': list(ROBOTS), 'models': {}}
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
                         'vla_task': (info or {}).get('vla_task'), 'vla_task_default': (info or {}).get('vla_task_default'),
                         'loop': running, 'last': last, 'log': [ln[:150] for ln in lines[-6:]],
                         'health': health_of(r, running and sess['state'] not in ('preparing', 'ready'), info)}
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
            if sess['state'] == 'running' and not sess['busy']:
                start_session_action(session_pause, '천장 카메라 녹화가 끊겨(또는 최대 %d초에 도달해) 일시 정지했습니다.' % OH_MAX_S)
            else:
                finish_recording()
        with lock:
            cache.update(robots=robots, misty_net=misty_net, adb=adb, t=time.time())
        if sess['state'] in ('ready', 'running') and not sess['busy']:
            dead = [r for r in sess['robots'] if not robots[r]['loop']]
            if dead:
                start_session_action(session_stop, ' / '.join('%s 루프가 종료되어 경기를 종료했습니다: %s' % (
                    ROBOTS[r]['title'], last_log_line(r)[:160]) for r in dead))
        if settings['autostop']:
            for r, info in robots.items():
                h = info['health']
                if h and ((h['level'] == 'bad' and h['streak_s'] >= AUTOSTOP_BAD_S)
                          or (h['level'] == 'slow' and h['streak_s'] >= AUTOSTOP_SLOW_S)):
                    alert['msg'] = '%s 자동 정지: %s — %s (%d초 지속). 이 경기는 무효로 보세요.' % (
                        time.strftime('%H:%M:%S'), ROBOTS[r]['title'], h['reason'], h['streak_s'])
                    stop_everything()
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


def precheck(robot, model):
    m = ROBOTS[robot]['models'][model]
    if robot == 'misty' and http_json('http://%s/api/battery' % MISTY_IP, timeout=3) is None:
        raise RuntimeError('Misty(%s)에 연결되지 않습니다. 전원과 Wi-Fi를 확인하세요.' % MISTY_IP)
    if robot == 'cozmo' and cache['adb'] != 'device':
        raise RuntimeError('태블릿이 adb에 %s 상태입니다. USB 연결과 "USB 디버깅 허용"을 확인하세요.' % cache['adb'])
    if m.get('needs') and not os.path.isdir(os.path.join(REPO, m['needs'])):
        raise RuntimeError('%s 가 없습니다. bash scripts/download_smolvla.sh 로 먼저 받으세요.' % m['needs'])


def ensure_server(robot, model):
    """Make sure the policy server for `model` is up on the robot's port (starts or swaps it)."""
    st, cfg = state[robot], ROBOTS[robot]
    m = cfg['models'][model]
    pid, cur = server_model(robot)
    if cur == model and http_json('http://127.0.0.1:%d/info' % cfg['port'], timeout=2) is not None:
        if robot == 'misty':
            apply_vla_task()
        return
    if pid:
        st['phase'] = '이전 서버 종료 중'
        stop_server(robot)
    st['phase'] = '정책 서버 시작 중 (20~60초)'
    cmd = [os.path.join(REPO, m['py'], 'bin', 'python'), 'policy_server.py', '--ckpt', m['ckpt'],
           '--port', str(cfg['port'])] + m['args']
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
        if sess['cancel']:
            raise RuntimeError('취소됨')
        time.sleep(1)
    if robot == 'misty':
        apply_vla_task()


def apply_vla_task():
    """Send the panel's SmolVLA instruction to the Misty "ours" server (does nothing for a server without SmolVLA)."""
    url = 'http://127.0.0.1:%d' % ROBOTS['misty']['port']
    info = http_json(url + '/info', timeout=2)
    if not info or not info.get('vla'):
        return None
    if 'vla_task' not in info:
        vla['error'] = '지금 떠 있는 정책 서버는 지시문 변경을 지원하지 않는 이전 버전입니다. "정책 서버 끄기" 후 다시 켜세요.'
        return None
    res = http_json(url + '/vla_task', {'task': vla['task']}, timeout=10)
    vla['error'] = '' if res and res.get('ok') else '정책 서버에 지시문을 적용하지 못했습니다.'
    return res


def start_loop(robot, gated=False):
    cfg = ROBOTS[robot]
    log = open(log_path(robot, 'loop'), 'w')
    cmd = [os.path.join(REPO, cfg['loop'][0])] + cfg['loop'][1:] + DRY + (['--gate', GATE] if gated else [])
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def last_log_line(robot):
    lines = tail(log_path(robot, 'loop'), 400).strip().splitlines()
    return lines[-1] if lines else ''


def turn_on(robot, model):
    st = state[robot]
    try:
        precheck(robot, model)
        stop_loop(robot)
        ensure_server(robot, model)
        st['phase'] = '로봇 루프 시작 중'
        proc = start_loop(robot)
        time.sleep(4)
        if proc.poll() is not None:
            raise RuntimeError('로봇 루프가 바로 종료됐습니다: ' + last_log_line(robot))
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
    # Every recording gets its own numbered folder <save folder>/<num>/ that keeps the video, its .meta.json,
    # the hit-detection .json and the annotated .mp4 together.
    os.makedirs(oh['dir'], exist_ok=True)
    nums = [int(d) for d in os.listdir(oh['dir']) if d.isdigit() and os.path.isdir(os.path.join(oh['dir'], d))]
    folder = os.path.join(oh['dir'], '%03d' % (max(nums, default=0) + 1))
    os.makedirs(folder)
    # Raw MJPEG first (frames only, no timestamps); finish_recording() wraps it into an .mkv with the measured frame rate.
    path = os.path.join(folder, time.strftime('%Y%m%d_%H%M%S') + '.mjpeg')
    log = open(os.path.join(LOGS, 'overhead_record.log'), 'w')
    oh['proc'] = subprocess.Popen([exe, '-y', '-loglevel', 'error', '-use_wallclock_as_timestamps', '1', '-f', 'mjpeg',
                                   '-i', OVERHEAD + '/stream.mjpg', '-c', 'copy', '-t', str(OH_MAX_S), '-f', 'mjpeg', path],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    oh['file'], oh['since'], oh['go_t'], oh['game'] = path, time.time(), None, None


def drop_leading_frames(path, skip):
    """Rewrite a raw MJPEG file without its first `skip` frames (a frame starts at each JPEG start marker FF D8 FF)."""
    cut, seen, prev = path + '.cut', 0, b''
    with open(path, 'rb') as f, open(cut, 'wb') as out:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            if seen > skip:                              # past the cut point: copy the rest through
                out.write(chunk)
                continue
            buf, i = prev + chunk, 0                     # prev: last 2 bytes, for a marker split over two chunks
            while True:
                i = buf.find(b'\xff\xd8\xff', i)
                if i < 0:
                    break
                seen += 1
                if seen > skip:                          # first frame to keep
                    out.write(buf[i:])
                    break
                i += 1
            prev = chunk[-2:]
    os.replace(cut, path)


def finish_recording():
    proc, path, go_t, game = oh['proc'], oh['file'], oh['go_t'], oh.get('game')
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
    # Game recordings: ffmpeg has to be connected before the robots are released, so the file starts a moment
    # (about 0.8 s) early. Those leading frames are cut here, so the saved video begins when the robots start.
    lead = go_t - oh['since'] if go_t else 0.0
    skip = min(int(round(fps * lead)), max(frames - 1, 0)) if lead > 0 else 0
    if skip:
        drop_leading_frames(path, skip)
        frames -= skip
        elapsed -= skip / fps
    mkv = os.path.splitext(path)[0] + '.mkv'
    rc = subprocess.call([ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '%.4f' % fps, '-f', 'mjpeg', '-i', path,
                          '-c', 'copy', mkv], stdin=subprocess.DEVNULL)
    if rc != 0 or not os.path.exists(mkv):
        oh['error'] = '녹화 파일 변환 실패. 원본은 남아 있습니다: ' + path
        return
    os.remove(path)
    # robots_started_after_s: 0 for a game recording (leading frames cut above; trimmed_lead_s says how much),
    # None for a manual recording that is not tied to the robots.
    json.dump({'frames': frames, 'seconds': round(elapsed, 2), 'fps': round(fps, 3), 'source': OVERHEAD + '/stream.mjpg',
               'recorded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(oh['since'])),
               'robots_started_after_s': round(lead - skip / fps, 2) if go_t else None,
               'trimmed_lead_s': round(skip / fps, 2) if go_t else None, 'game': game},
              open(os.path.splitext(path)[0] + '.meta.json', 'w'))
    oh['file'] = mkv
    threading.Thread(target=analyze, args=(mkv,), daemon=True).start()


def set_overhead_dir(text):
    """Change the folder for new recordings (panel "수정" button). Returns an error message, or '' when done."""
    if oh['proc'] is not None:
        return '녹화 중에는 저장 폴더를 바꿀 수 없습니다. 녹화를 먼저 중지하세요.'
    text = (text or '').strip()
    if not text:
        return '폴더 경로가 비어 있습니다.'
    path = os.path.expanduser(text)
    if not os.path.isabs(path):
        path = os.path.join(REPO, path)                 # relative paths start at the repo root
    path = os.path.normpath(path)
    try:
        os.makedirs(path, exist_ok=True)
        if not os.access(path, os.W_OK):
            return '이 폴더에 쓸 수 없습니다: ' + path
        open(OH_DIR_FILE, 'w').write(path)
    except OSError as e:
        return '폴더를 만들 수 없습니다: %s (%s)' % (path, e.strerror)
    oh['dir'] = path
    return ''


def analyze(path):
    """Overhead YOLO hit detection on a saved recording -> <name>.json (time to first hit) + annotated <name>.mp4."""
    name = os.path.basename(path)
    # Starts as soon as the recording is saved, also while the robots are running (user's choice).
    oh['analyzing'] = name + ' (히트 판정 중)'
    log = open(os.path.join(LOGS, 'overhead_analyze.log'), 'w')
    # hit_detect writes the annotated video at --fps with every input frame, so it must be the recording's real
    # frame rate (about 30); with 10 the annotated video played 3x slower than real time.
    try:
        fps = float(json.load(open(os.path.splitext(path)[0] + '.meta.json'))['fps'])
    except Exception:
        fps = 30.0
    rc = subprocess.call([os.path.join(REPO, '.venv', 'bin', 'python'), '-m', 'overhead.hit_detect',
                          '--weights', '../checkpoints/yolo/overhead_yolo_v1.pt', '--video', path, '--fps', '%.3f' % fps, '--out', os.path.dirname(path)],
                         cwd=os.path.join(REPO, 'baseline'), stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        oh['error'] = '히트 판정 실패 (%s): %s' % (name, tail(os.path.join(LOGS, 'overhead_analyze.log'), 300).strip()[-200:])
    oh['analyzing'] = ''


def overhead_status():
    recs = []
    found = glob.glob(os.path.join(oh['dir'], '*', '*.mkv')) + glob.glob(os.path.join(oh['dir'], '*.mkv'))   # numbered folders + older flat files
    for f in sorted(found, key=os.path.getmtime, reverse=True)[:8]:
        base = os.path.splitext(f)[0]
        item = {'file': f, 'mb': round(os.path.getsize(f) / 1e6, 1), 'analyzed': False, 'hit': None, 'duration': None,
                'annotated': base + '.mp4' if os.path.exists(base + '.mp4') else '', 'offset': None}
        try:
            item['offset'] = json.load(open(base + '.meta.json')).get('robots_started_after_s')
        except Exception:
            pass
        if os.path.exists(base + '.json'):
            try:
                summ = json.load(open(base + '.json'))['summary']
                item.update(analyzed=True, hit=summ.get('time_to_first_hit_s'), duration=summ.get('duration_s'))
            except Exception:
                pass
        recs.append(item)
    return {'url': OVERHEAD, 'reachable': oh['reachable'], 'recording': oh['proc'] is not None, 'file': oh['file'],
            'elapsed': int(time.time() - oh['since']) if oh['proc'] is not None else 0, 'dir': oh['dir'], 'dir_default': OH_DIR_DEFAULT,
            'analyzing': oh['analyzing'], 'error': oh['error'], 'recent': recs, 'max_s': OH_MAX_S}


def set_gate(value):
    tmp = GATE + '.tmp'
    open(tmp, 'w').write(value)
    os.replace(tmp, GATE)


def prepare_robot(robot, model, errs):
    """Server up + loop connected and waiting in standby. Blocks until the loop reports READY."""
    st = state[robot]
    st['busy'], st['error'] = True, ''
    try:
        precheck(robot, model)
        stop_loop(robot)
        ensure_server(robot, model)
        st['phase'] = '로봇 연결 중 (카메라 대기)'
        proc = start_loop(robot, gated=True)
        t0 = time.time()
        while 'READY' not in tail(log_path(robot, 'loop'), 20000):
            if proc.poll() is not None:
                raise RuntimeError('로봇 루프가 종료됐습니다: ' + last_log_line(robot))
            if sess['cancel']:
                raise RuntimeError('취소됨')
            if time.time() - t0 > 90:
                raise RuntimeError('90초 안에 카메라 프레임이 들어오지 않았습니다.')
            time.sleep(0.3)
    except Exception as exc:
        errs[robot] = str(exc)
        st['error'] = '' if sess['cancel'] else str(exc)
    finally:
        st['phase'] = ''
        st['busy'] = False


def teardown():
    set_gate('standby')
    ts = [threading.Thread(target=stop_loop, args=(r,), daemon=True) for r in ROBOTS]
    [t.start() for t in ts]
    [t.join() for t in ts]
    finish_recording()
    sess.update(state='off', since=0.0)


def go():
    """Standby -> running: start the recording, wait until video really arrives, then release both robots at once."""
    sess['phase'] = '녹화 시작 중'
    for r in sess['robots']:
        if not loop_pids(r):
            raise RuntimeError('%s 루프가 꺼져 있습니다.' % ROBOTS[r]['title'])
    if oh['proc'] is not None:
        finish_recording()
    start_recording()
    if oh['proc'] is None:
        raise RuntimeError(oh['error'] or '녹화를 시작하지 못했습니다.')
    t0 = time.time()
    while time.time() - t0 < 5 and oh['proc'].poll() is None and not (os.path.exists(oh['file']) and os.path.getsize(oh['file']) > 0):
        time.sleep(0.05)
    if not (os.path.exists(oh['file']) and os.path.getsize(oh['file']) > 0):
        finish_recording()
        raise RuntimeError('천장 카메라에서 영상이 들어오지 않아 로봇을 출발시키지 않았습니다.')
    if sess['cancel']:
        finish_recording()
        raise RuntimeError('취소됨')
    set_gate('run')
    oh['go_t'] = time.time()
    # written into the recording's .meta.json: which model each robot ran, or "static" for a robot kept still
    oh['game'] = {r: sess['models'].get(r) if r in sess['robots'] else 'static' for r in ROBOTS}
    info = http_json('http://127.0.0.1:%d/info' % ROBOTS['misty']['port'], timeout=2) or {}
    if info.get('vla'):
        oh['game']['vla_task'] = info.get('vla_task')
    sess.update(state='running', since=time.time())


def session_start(models, autostart, robots):
    """off -> (preparing) -> ready [-> running]. robots: who takes part; the others are switched off and stay still."""
    try:
        sess.update(state='preparing', phase='준비 중', robots=list(robots), models=dict(models))
        set_gate('standby')
        for r in ROBOTS:
            if r not in robots:
                stop_loop(r)
        if not oh['reachable']:
            raise RuntimeError('천장 카메라 스트림(%s)에 연결되지 않습니다. Jetson에서 스트림 프로그램을 켜 주세요.' % OVERHEAD)
        if not ffmpeg_exe():
            raise RuntimeError('ffmpeg 을 찾지 못했습니다.')
        errs = {}
        ts = [threading.Thread(target=prepare_robot, args=(r, models[r], errs), daemon=True) for r in robots]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if errs:
            raise RuntimeError(' / '.join('%s: %s' % (ROBOTS[r]['title'], e) for r, e in errs.items()))
        if sess['cancel']:
            raise RuntimeError('취소됨')
        sess['state'] = 'ready'
    except Exception as exc:
        sess['error'] = '' if sess['cancel'] else '준비 실패 — ' + str(exc)
        teardown()
        return
    if autostart:
        session_go()


def session_go():
    if sess['state'] != 'ready':
        return
    try:
        go()
    except Exception as exc:
        if str(exc) != '취소됨':
            sess['error'] = '시작 실패 — ' + str(exc)


def session_pause(reason=''):
    """running -> ready: robots stop and wait, the recording is closed (and analysed)."""
    if sess['state'] != 'running':
        return
    set_gate('standby')
    sess['phase'] = '일시 정지 중 (녹화 저장)'
    time.sleep(0.4)
    http_json('http://%s/api/drive/stop' % MISTY_IP, {'Hold': False}, timeout=2)
    finish_recording()
    sess.update(state='ready', since=0.0)
    if reason:
        sess['error'] = reason


def session_stop(reason=''):
    sess['phase'] = '종료 중'
    teardown()
    if reason:
        sess['error'] = reason


def start_session_action(fn, *args):
    with lock:
        if sess['busy']:
            return False
        sess['busy'] = True
        if fn is not session_stop or not args:
            sess['error'] = ''

    def run():
        try:
            fn(*args)
        except Exception as exc:
            sess['error'] = str(exc)
        finally:
            sess.update(busy=False, phase='', cancel=False)
    threading.Thread(target=run, daemon=True).start()
    return True


def request_stop():
    """Full stop from any state. The robots are sent to standby at once; a running action is cancelled first."""
    set_gate('standby')
    sess['cancel'] = True

    def run():
        t0 = time.time()
        while sess['busy'] and time.time() - t0 < 300:
            time.sleep(0.1)
        if sess['state'] != 'off':
            start_session_action(session_stop)
        else:
            sess['cancel'] = False
    threading.Thread(target=run, daemon=True).start()


def stop_everything():
    """Header "모두 정지" and the automatic stop: ends a game if one is on, otherwise just stops the loops."""
    if sess['state'] != 'off' or sess['busy']:
        request_stop()
    else:
        for r in ROBOTS:
            threading.Thread(target=stop_loop, args=(r,), daemon=True).start()


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
.mini{padding:3px 10px;border-radius:7px;font-size:12px;background:var(--chip);color:var(--ink);margin-left:6px}
.row{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.row .mini{margin-left:0;padding:6px 12px}
textarea{width:100%;padding:9px 10px;border-radius:9px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:13px/1.4 inherit;font-family:inherit;resize:vertical}
input[type=text]{width:100%;padding:9px 10px;border-radius:9px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:12.5px ui-monospace,Menlo,monospace}
[hidden]{display:none!important}
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
.game{margin-bottom:16px}.game .btns{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.game .btns button{flex:1 1 220px;font-size:16px;padding:14px}.primary{background:var(--btn);color:var(--btnink)}
.steps{display:flex;gap:8px;flex-wrap:wrap}.step{font-size:13px;padding:5px 10px;border-radius:8px;background:var(--chip);color:var(--mute)}
.step.ok{background:color-mix(in srgb,var(--on) 14%,transparent);color:var(--on)}.step.wait{background:color-mix(in srgb,var(--warn) 16%,transparent);color:var(--warn)}
.seg{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:8px}.seg span{font-size:12px;color:var(--mute);min-width:64px}
.seg button{padding:7px 12px;border-radius:8px;font-size:13px;background:var(--chip);color:var(--ink);font-weight:500}
.seg button.sel{background:var(--btn);color:var(--btnink);font-weight:700}
.note{margin-top:8px;padding:8px 10px;border-radius:8px;font-size:13px;background:color-mix(in srgb,var(--warn) 16%,transparent);color:var(--warn)}
.step.bad{background:color-mix(in srgb,var(--off) 12%,transparent);color:var(--off)}
.chk{display:flex;align-items:center;gap:6px;font-size:13.5px;color:var(--ink);margin:0}.chk input{width:16px;height:16px}
.hdr{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
footer{margin-top:18px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;color:var(--mute);font-size:13px}
</style></head><body><main>
<header><h1>로봇 조작 패널</h1><div class="hdr">
  <label class="chk" title="강제 정지가 3초, 예산 초과가 10초 이어지면 두 로봇을 모두 멈춥니다"><input type="checkbox" id="autostop">문제 생기면 자동 정지</label>
  <button class="danger" id="alloff">모두 정지</button></div></header>
<div class="banner" id="banner" hidden><span id="bannertext"></span><button id="bannerx" hidden>닫기</button></div>
<section class="card game">
  <div class="top"><div><h2>경기 (Misty + Cozmo + 녹화 한 번에)</h2><div class="role" id="g-sub">모델과 조건을 고르면 Misty 모델, Cozmo 정지 여부, 저장 폴더가 함께 맞춰집니다</div></div><span class="pill" id="g-pill">꺼짐</span></div>
  <div class="steps"><span class="step" id="g-misty">Misty</span><span class="step" id="g-cozmo">Cozmo</span><span class="step" id="g-cam">천장 카메라</span></div>
  <div class="seg" style="margin-top:12px"><span>모델</span><span id="g-fm" style="display:contents"></span></div>
  <div class="seg"><span>조건</span><span id="g-fc" style="display:contents"></span></div>
  <div class="seg"><span>연습</span><button id="g-ftest" title="실험 결과에 넣지 않을 연습 판">test</button></div>
  <div class="path" id="g-dir"></div>
  <div class="btns"><button class="primary" id="g-main">시작</button><button class="ghost" id="g-pause" hidden>일시 정지 (녹화 종료)</button><button class="danger" id="g-stop" hidden>완전 종료</button></div>
  <div id="g-autowrap"><label class="chk" style="margin-top:10px"><input type="checkbox" id="g-auto" checked>모두 준비되면 바로 출발 (끄면 스탠바이에서 기다림)</label>
  <label class="chk" style="margin-top:6px"><input type="checkbox" id="g-static">Cozmo 정지 — Misty만 주행 (Cozmo는 켜지 않으므로 태블릿 연결이 필요 없음)</label></div>
  <div class="err" id="g-err" hidden></div>
  <div class="role" style="margin-top:8px">셋이 모두 준비될 때까지 로봇은 움직이지 않습니다. 일시 정지하면 로봇은 스탠바이로 서고 녹화는 저장됩니다. 다시 시작하면 바로 출발하고 새 파일로 녹화합니다.</div>
</section>
<div class="grid" id="grid"></div>
<div class="grid" style="margin-top:16px"><section class="card wide">
  <div class="top"><div><h2>Bird's-eye view</h2><div class="role">천장 카메라 · <span id="oh-url"></span></div></div><span class="pill" id="oh-pill">–</span></div>
  <div class="oh"><div class="view" id="oh-view">연결 확인 중…</div>
  <div><button class="toggle" id="oh-btn" style="margin-top:0">녹화 시작</button>
    <div class="err" id="oh-err" hidden></div>
    <label style="margin-top:12px">저장 폴더 <button class="mini" id="oh-dir-edit">수정</button></label>
    <div class="path" id="oh-dir"></div>
    <div id="oh-dir-form" hidden><input type="text" id="oh-dir-in" spellcheck="false" autocomplete="off" aria-label="저장 폴더 경로">
      <div class="row"><button class="mini" id="oh-dir-save">저장</button><button class="mini" id="oh-dir-cancel">취소</button><button class="mini" id="oh-dir-reset">기본 폴더로</button></div>
      <div class="err" id="oh-dir-err" hidden></div>
      <div class="role">녹화마다 이 폴더 안에 001, 002 … 폴더를 만들어 영상과 결과를 함께 넣습니다. 이후 녹화부터 적용됩니다. 없는 폴더는 새로 만들고, 상대 경로는 레포 폴더 기준입니다.</div></div>
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
    ${key === 'misty' ? `<label for="vt-in" style="margin-top:12px">SmolVLA 지시문 (ours 모델에만 쓰임)</label>
    <textarea id="vt-in" rows="3" spellcheck="false"></textarea>
    <div class="row"><button class="mini" id="vt-save">적용</button><button class="mini" id="vt-reset">학습 때 지시문으로</button></div>
    <div class="role" id="vt-note"></div>` : ''}
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
const models = () => Object.fromEntries(Object.keys(ROBOTS).map(k => [k, document.getElementById('sel-' + k).value]));
document.getElementById('g-main').onclick = () => post('/api/session/start', {models: models(), autostart: document.getElementById('g-auto').checked, cozmo_static: document.getElementById('g-static').checked});
document.getElementById('g-pause').onclick = () => post('/api/session/pause', {});
document.getElementById('g-stop').onclick = () => post('/api/session/stop', {});
// Save-folder picker: <default folder>/<model>/<condition>. obs = one obstacle, stop = the evader (Cozmo) stays still.
// One choice sets everything: the model button also picks the Misty model, the condition button also sets
// "Cozmo 정지"; changing the Misty model or that checkbox moves the folder the same way.
const F_MODELS = ['baseline', 'ours'], F_CONDS = ['nonobs_stop', 'obs_stop', 'nonobs_moving', 'obs_moving'];
let fPick = {m: null, c: null}, fSaving = false, fState = 'off', fGameStill = false;
const fStill = (c) => c.endsWith('_stop');
async function fApply() {
  if (!(fPick.m && fPick.c)) { folderUI(); return; }
  fSaving = true; try { await saveDir(ohDirDefault + '/' + fPick.m + '/' + fPick.c); } finally { fSaving = false; }
}
for (const [id, list, k] of [['g-fm', F_MODELS, 'm'], ['g-fc', F_CONDS, 'c']])
  for (const v of list) {
    const b = document.createElement('button'); b.textContent = v; b.dataset.v = v; b.dataset.k = k;
    b.onclick = () => {
      fPick[k] = v;
      if (k === 'm') document.getElementById('sel-misty').value = v; else document.getElementById('g-static').checked = fStill(v);
      fApply();
    };
    document.getElementById(id).appendChild(b);
  }
document.getElementById('g-ftest').onclick = () => { fPick = {m: null, c: null}; saveDir(ohDirDefault + '/test'); };
document.getElementById('sel-misty').addEventListener('change', (e) => { if (F_MODELS.includes(e.target.value)) { fPick.m = e.target.value; fApply(); } });
document.getElementById('g-static').addEventListener('change', (e) => {
  if (fPick.c) { fPick.c = fPick.c.split('_')[0] + (e.target.checked ? '_stop' : '_moving'); fApply(); }
});
function folderUI(o, s) {
  if (o) {
    folderUI.test = o.dir === o.dir_default + '/test';
    const rel = o.dir.startsWith(o.dir_default + '/') ? o.dir.slice(o.dir_default.length + 1).split('/') : [];
    const ok = rel.length === 2 && F_MODELS.includes(rel[0]) && F_CONDS.includes(rel[1]);
    fState = s.session.state; fGameStill = !s.session.robots.includes('cozmo');
    if (!fSaving) {
      if (ok) fPick = {m: rel[0], c: rel[1]}; else if (folderUI.test) fPick = {m: null, c: null};
      if (ok && fState === 'off' && !s.session.busy) {            // keep the model and the Cozmo option in step with the folder
        document.getElementById('sel-misty').value = rel[0]; document.getElementById('g-static').checked = fStill(rel[1]);
      }
    }
    document.getElementById('g-dir').textContent = o.dir + '/<번호>/';
    folderUI.rec = o.recording;
  }
  const inGame = fState !== 'off', curModel = document.getElementById('sel-misty').value;
  for (const b of document.querySelectorAll('#g-fm button, #g-fc button')) {
    b.className = fPick[b.dataset.k] === b.dataset.v ? 'sel' : '';
    // during a game the model and stop/moving are fixed; only the obstacle condition can still be switched
    b.disabled = !!folderUI.rec || (inGame && (b.dataset.k === 'm' ? b.dataset.v !== curModel : fStill(b.dataset.v) !== fGameStill));
  }
  const t = document.getElementById('g-ftest'); t.className = folderUI.test ? 'sel' : ''; t.disabled = !!folderUI.rec;
}
function game(s) {
  const g = s.session, st = g.state, main = document.getElementById('g-main'), pill = document.getElementById('g-pill');
  const label = {off: '꺼짐', preparing: '준비 중', ready: '스탠바이', running: `경기 중 ${g.elapsed}초`}[st];
  pill.textContent = g.busy && g.phase && st !== 'preparing' ? g.phase : label;
  pill.className = 'pill' + (st === 'running' ? ' on' : st === 'off' ? '' : ' busy');
  main.hidden = st === 'running'; main.disabled = g.busy;
  main.textContent = st === 'off' ? '시작' : st === 'preparing' ? '준비 중…' : (g.busy ? '출발 중…' : '출발 (새 녹화)');
  document.getElementById('g-pause').hidden = st !== 'running'; document.getElementById('g-pause').disabled = g.busy;
  document.getElementById('g-stop').hidden = st === 'off' && !g.busy;
  document.getElementById('g-stop').textContent = st === 'preparing' ? '취소' : '완전 종료';
  document.getElementById('g-autowrap').hidden = st !== 'off';
  const err = document.getElementById('g-err'); err.hidden = !g.error; err.textContent = g.error;
  document.getElementById('g-sub').textContent = (s.dry ? '[리허설 모드: 주행 명령을 보내지 않음] ' : '') + ({
    off: '모델과 조건을 고르면 Misty 모델, Cozmo 정지 여부, 저장 폴더가 함께 맞춰집니다', preparing: '모델과 로봇을 준비하는 중입니다. 로봇은 아직 움직이지 않습니다.',
    ready: '스탠바이: 모델이 떠 있고 로봇은 멈춰 있습니다. 위치를 잡고 출발을 누르세요.', running: (g.robots.length > 1 ? '두 로봇이' : 'Misty만') + ' 주행 중이고 천장 카메라를 녹화하고 있습니다.'}[st]);
  for (const k of Object.keys(ROBOTS)) {
    const r = s.robots[k], el = document.getElementById('g-' + k); if (!r) continue;
    if (st === 'off' ? (k === 'cozmo' && document.getElementById('g-static').checked) : !g.robots.includes(k)) {
      el.textContent = ROBOTS[k].title + ': 정지 (이번 경기에서 안 움직임)'; el.className = 'step'; continue; }
    const up = st !== 'off' && r.loop && !r.busy;
    el.textContent = ROBOTS[k].title + ': ' + (r.busy ? (r.phase || '처리 중') : up ? (st === 'running' ? '주행 중' : '준비됨') : r.error && st === 'off' ? '실패' : '꺼짐');
    el.className = 'step' + (r.busy ? ' wait' : up ? ' ok' : r.error && st === 'off' ? ' bad' : '');
  }
  const cam = document.getElementById('g-cam'), o = s.overhead;
  cam.textContent = '천장 카메라: ' + (o.recording ? `녹화 중 ${o.elapsed}초` : o.reachable ? '연결됨' : '연결 안 됨');
  cam.className = 'step' + (o.recording || o.reachable ? ' ok' : ' bad');
  return st !== 'off' || g.busy;
}
// SmolVLA instruction editor (Misty card)
const VT_TRAINED = 'Drive into the enemy robot and bump it with your body while avoiding obstacles.';
let vtDirty = false, vtMsg = '';
document.getElementById('vt-in').oninput = () => { vtDirty = true; };
async function vtSend(task) {
  try {
    const r = await (await fetch('/api/vla_task', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({task})})).json();
    vtDirty = false; vtMsg = r.ok ? '' : r.error;
  } catch (e) { vtMsg = '패널 서버에 연결되지 않습니다.'; }
  refresh();
}
document.getElementById('vt-save').onclick = () => vtSend(document.getElementById('vt-in').value);
document.getElementById('vt-reset').onclick = () => vtSend('');
function vlaTask(s) {
  const r = s.robots.misty || {}, v = s.vla_task, box = document.getElementById('vt-in'), note = document.getElementById('vt-note');
  const trained = r.vla_task_default || VT_TRAINED, want = v.saved || trained;
  if (!vtDirty && document.activeElement !== box) box.value = want;
  let t;
  if (v.error || vtMsg) t = '⚠ ' + (v.error || vtMsg);
  else if (r.vla_task != null) t = (r.vla_task === want ? '서버에 적용됨' : '서버에는 아직 다른 지시문이 들어 있음 — 적용을 누르세요') + (r.vla_task !== trained ? ' · 학습 때와 다른 지시문입니다 (학습 범위 밖)' : ' · 학습 때 지시문');
  else t = (r.server_ready ? '지금 서버는 SmolVLA 없이 떠 있습니다 (baseline).' : '서버가 꺼져 있습니다.') + ' ours를 켤 때 이 지시문이 적용됩니다.' + (want !== trained ? ' · 학습 때와 다른 지시문' : '');
  note.textContent = (vtDirty ? '(수정 중 — 적용을 눌러야 반영) ' : '') + t;
  note.className = 'role' + (v.error || vtMsg ? ' bad' : '');
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
  document.getElementById('oh-dir').textContent = o.dir; ohDir = o.dir; ohDirDefault = o.dir_default;
  document.getElementById('oh-dir-edit').disabled = o.recording; document.getElementById('oh-file').textContent = o.file || '–';
  document.getElementById('oh-note').textContent = o.analyzing ? '분석: ' + o.analyzing : `녹화는 최대 ${o.max_s}초. 중지하면 바로 히트 판정을 자동으로 돌립니다.`;
  if (o.reachable) {
    if (!view.querySelector('img')) { view.innerHTML = '<img alt="천장 카메라 실시간 화면">'; }
    if (!ohTimer) ohTimer = setInterval(() => { const im = view.querySelector('img'); if (im && !document.hidden) im.src = ohUrl + '/snapshot.jpg?t=' + Date.now(); }, 400);
  } else {
    if (ohTimer) { clearInterval(ohTimer); ohTimer = null; }
    view.textContent = '천장 카메라 스트림에 연결되지 않습니다. Jetson에서 스트림 프로그램을 켜 주세요.';
  }
  document.getElementById('oh-rows').innerHTML = o.recent.length ? o.recent.map(r => `<tr><td class="f">${r.file}</td><td>${r.mb} MB</td>
    <td>${r.duration != null ? r.duration.toFixed(1) + '초' : '–'}</td>
    <td>${r.analyzed ? (r.hit != null ? r.hit.toFixed(2) + '초' + (r.offset != null ? ' (출발 기준)' : '') : '히트 없음') : '미분석'}</td><td class="f">${r.annotated || '–'}</td></tr>`).join('')
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
let ohDir = '', ohDirDefault = '';
function dirForm(show) {
  document.getElementById('oh-dir-form').hidden = !show; document.getElementById('oh-dir').hidden = show;
  document.getElementById('oh-dir-err').hidden = true;
  if (show) { const i = document.getElementById('oh-dir-in'); i.value = ohDir; i.focus(); i.select(); }
}
async function saveDir(dir) {
  const err = document.getElementById('oh-dir-err');
  try {
    const r = await (await fetch('/api/overhead/dir', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({dir})})).json();
    if (r.ok) { dirForm(false); refresh(); } else { err.textContent = r.error; err.hidden = false; if (document.getElementById('oh-dir-form').hidden) alert(r.error); }
  } catch (e) { err.textContent = '패널 서버에 연결되지 않습니다.'; err.hidden = false; }
}
document.getElementById('oh-dir-edit').onclick = () => dirForm(true);
document.getElementById('oh-dir-cancel').onclick = () => dirForm(false);
document.getElementById('oh-dir-save').onclick = () => saveDir(document.getElementById('oh-dir-in').value);
document.getElementById('oh-dir-reset').onclick = () => saveDir(ohDirDefault);
document.getElementById('oh-dir-in').onkeydown = (e) => { if (e.key === 'Enter') saveDir(e.target.value); if (e.key === 'Escape') dirForm(false); };
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
  folderUI(s.overhead, s); vlaTask(s);
  const inGame = game(s), standby = ['preparing', 'ready'].includes(s.session.state);
  if (inGame) document.getElementById('oh-btn').disabled = true;       // the game owns the recording
  const problems = [];
  for (const key of Object.keys(ROBOTS)) {
    const r = s.robots[key]; if (!r) continue;
    const h = health(key, r);
    if (h && h.level !== 'ok') problems.push({level: h.level, text: `${ROBOTS[key].title}: ${h.reason} (${h.streak_s}초째, 추론 ${h.infer ?? '–'} ms / 예산 ${h.budget} ms)`});
    const pill = document.getElementById('pill-' + key), btn = document.getElementById('btn-' + key), sel = document.getElementById('sel-' + key);
    pill.textContent = r.busy ? (r.phase || '처리 중') : (r.loop ? (standby ? '스탠바이' : '주행 중') : '꺼짐');
    pill.className = 'pill' + (r.busy || (r.loop && standby) ? ' busy' : r.loop ? ' on' : '');
    btn.dataset.on = r.loop ? '1' : '0'; btn.textContent = inGame ? '경기 중에는 위 버튼으로 조작' : r.loop ? '끄기' : '켜기';
    btn.className = 'toggle' + (r.loop && !inGame ? ' stop' : ''); btn.disabled = r.busy || inGame;
    sel.disabled = r.busy || r.loop || inGame;
    if (r.loop && standby && !r.health) { document.getElementById('hl-' + key).textContent = '스탠바이 (정지 상태로 대기)'; document.getElementById('hs-' + key).textContent = '출발하면 6초 뒤부터 추론 속도를 점검합니다'; }
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
            out.update(autostop=settings['autostop'], alert=alert['msg'], overhead=overhead_status(), dry=bool(DRY),
                       vla_task={'saved': vla['task'], 'error': vla['error']},
                       session={'state': sess['state'], 'phase': sess['phase'], 'error': sess['error'], 'busy': sess['busy'],
                                'robots': sess['robots'],
                                'elapsed': int(time.time() - sess['since']) if sess['state'] == 'running' else 0})
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
        in_game = sess['state'] != 'off' or sess['busy']
        if self.path == '/api/session/start':
            models = body.get('models') or {}
            robots = ['misty'] if body.get('cozmo_static') else list(ROBOTS)     # evader kept still: Misty only
            if not all(models.get(r) in ROBOTS[r]['models'] for r in robots):
                return self.send(400, '{"error": "bad models"}')
            if sess['state'] == 'off':
                start_session_action(session_start, models, bool(body.get('autostart', True)), robots)
            elif sess['state'] == 'ready':
                start_session_action(session_go)
        elif self.path == '/api/session/pause':
            start_session_action(session_pause)
        elif self.path == '/api/session/stop':
            request_stop()
        elif in_game and self.path in ('/api/on', '/api/off', '/api/overhead/start', '/api/overhead/stop'):
            return self.send(409, '{"error": "game in progress"}')     # the game owns the robots and the recording
        elif self.path == '/api/on' and robot in ROBOTS and model in ROBOTS[robot]['models']:
            start_action(robot, turn_on, model)
        elif self.path == '/api/off' and robot in ROBOTS:
            start_action(robot, turn_off)
        elif self.path == '/api/overhead/start':
            start_recording()
        elif self.path == '/api/overhead/stop':
            threading.Thread(target=finish_recording, daemon=True).start()
        elif self.path == '/api/overhead/dir':
            err = set_overhead_dir(body.get('dir'))
            return self.send(200, json.dumps({'ok': not err, 'error': err, 'dir': oh['dir']}))
        elif self.path == '/api/vla_task':
            vla['task'] = str(body.get('task') or '').strip()[:500]
            open(VLA_TASK_FILE, 'w').write(vla['task'])
            vla['error'] = ''
            res = apply_vla_task()
            return self.send(200, json.dumps({'ok': not vla['error'], 'error': vla['error'], 'applied': bool(res)}))
        elif self.path == '/api/settings':
            settings['autostop'] = bool(body.get('autostop'))
        elif self.path == '/api/clear_alert':
            alert['msg'] = ''
        elif self.path == '/api/off_all':
            stop_everything()
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
    if os.path.exists(OH_DIR_FILE):
        oh['dir'] = open(OH_DIR_FILE).read().strip() or OH_DIR_DEFAULT
    if os.path.exists(VLA_TASK_FILE):
        vla['task'] = open(VLA_TASK_FILE).read().strip()
    set_gate('standby')          # a panel restart never leaves robots driving on a stale "run"
    procs = sh(['ps', '-axo', 'command='])
    waiting = [r for r, cfg in ROBOTS.items() if any(cfg['loop_pat'] in ln and '--gate' in ln.split() for ln in procs.splitlines())]
    if 'misty' in waiting:       # loops from an earlier panel are still waiting in standby (Misty alone = evader kept still)
        sess.update(state='ready', robots=waiting)
    threading.Thread(target=monitor, daemon=True).start()
    print('control panel: http://127.0.0.1:%d' % PORT, flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
