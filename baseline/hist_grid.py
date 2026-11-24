"""History ablation: how far back the policy looks (seconds) x how many frames it is given.

Grid cell = (seq_len T, frame_stride K); the window spans (T-1)*K frames (chaser ~9.8 fps, evader ~8.4 fps).
Every cell is trained with 3-fold cross-session CV (fold f tests on sessions[f::3], like cv.py) and the
final_v6 baseline recipe (improved preset, batch 32, depth Small@252 inputs).

    python hist_grid.py --gpu 1 [--wait_pids 123 456] [--robots chaser evader] [--dry]
    python hist_grid.py --summary

Runs are launched side by side on one shared GPU while they fit: a run starts only when the card's free
memory minus --reserve_gb covers its estimated need (estimates are replaced by the measured peak of earlier
runs with the same seq_len). Frames live on the GPU when that fits (LSMA_GPU_DATA=1), else in host memory
(LSMA_GPU_DATA=cpu). Resumable: cells with test_metrics.json are skipped; a failed run is retried twice.
Output: runs/cv_hist/<robot>/T<T>_K<K>/f<fold>/ and runs/cv_hist/summary.{json,md}.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

FPS = {'chaser': 9.8, 'evader': 8.4}
RECIPE = {'chaser': ['--epochs', '30', '--select', 'ae'], 'evader': ['--epochs', '20', '--select', 'bacc']}
# (T, K) in launch order: the T=8 row first (pure look-back sweep at the current frame count), then fewer frames,
# the single-frame reference, and the 16-frame row last (slowest). Spans ~ 14 / 28 / 56 / 112 frames.
GRID = [(8, 4), (8, 8), (8, 2), (8, 16),
        (4, 9), (4, 19), (4, 5), (4, 37),
        (1, 1),
        (16, 2), (16, 4), (16, 1), (16, 7)]
FOLDS = 3
ROOT = 'runs/cv_hist'
VLA_LATENCY = {'chaser': '4', 'evader': '3'}   # frames (see HANDOFF: ~340 ms)
KEYS = ['ae', 'ae_zero', 'acc_joint', 'acc_joint_major', 'bacc_joint', 'acc_w', 'w_sign_acc']


def log(m):
    print(time.strftime('%m-%d %H:%M:%S'), m, flush=True)


def sessions(robot):
    return sorted(os.path.dirname(p) for p in glob.glob(f'../data_download/{robot}/*/*/obs_dsmall252.npz'))


SEED = 0


def run_dir(robot, T, K, f):
    return os.path.join(ROOT, robot, f'T{T}_K{K}', f'f{f}' + (f'_s{SEED}' if SEED else ''))


def smi(gpu):
    """(free MiB on the card, {pid: used MiB})"""
    q = lambda *a: subprocess.run(['nvidia-smi', *a], capture_output=True, text=True).stdout
    used, total = map(int, q(f'--id={gpu}', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits').split(','))
    procs = {}
    for line in q('--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits').strip().splitlines():
        p, m = line.split(',')
        procs[int(p)] = int(m)
    return total - used, procs


def summary():
    out, lines = {}, []
    for robot in ('chaser', 'evader'):
        rows = []
        for T, K in sorted(set(GRID)):
            ms = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, robot, f'T{T}_K{K}', 'f*', 'test_metrics.json')))]
            if not ms:
                continue
            r = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in KEYS if k in ms[0]}
            r.update(T=T, K=K, span_frames=(T - 1) * K, span_s=round((T - 1) * K / FPS[robot], 1), folds=len(ms))
            rows.append(r)
        out[robot] = rows
        lines += [f'### {robot}', '', '| frames T | stride K | look-back (s) | folds | AE | acc_joint | bacc_joint | w_sign_acc | AE zero | acc majority |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        f2 = lambda r, k: f'{r[k][0]:.3f}±{r[k][1]:.3f}' if k in r else '-'
        for r in rows:
            lines.append(f"| {r['T']} | {r['K']} | {r['span_s']} | {r['folds']} | {f2(r,'ae')} | {f2(r,'acc_joint')} | {f2(r,'bacc_joint')} | "
                         f"{f2(r,'w_sign_acc')} | {f2(r,'ae_zero')} | {f2(r,'acc_joint_major')} |")
        lines.append('')
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, 'summary.json'), 'w'), indent=1)
    open(os.path.join(ROOT, 'summary.md'), 'w').write('\n'.join(lines))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--robots', nargs='+', default=['chaser', 'evader'])
    ap.add_argument('--reserve_gb', type=float, default=4.0, help='free memory always left on the card')
    ap.add_argument('--max_par', type=int, default=2)
    ap.add_argument('--wait_pids', nargs='*', type=int, default=[], help='start only after these processes exit')
    ap.add_argument('--yield_gb', type=float, default=0.0,
                    help='instead of waiting for --wait_pids: start now, but keep this much memory for the train.py runs '
                         'those chains launch one at a time (whatever such a run already holds counts towards it)')
    ap.add_argument('--yield_match', default='final_v6', help='cmdline substring of the chains\' train.py runs')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--store', default='auto', choices=['auto', 'cpu'], help='cpu: always keep frames in host memory (when other jobs on the card come and go)')
    ap.add_argument('--cells', nargs='*', default=[], help='only these grid cells, e.g. 8x4 4x5 (T x K)')
    ap.add_argument('--seed', type=int, default=0, help='seeds > 0 go to f<fold>_s<seed> next to the seed-0 runs')
    ap.add_argument('--name', default='', help='output folder runs/cv_vla/<name> (default: the vla tag)')
    ap.add_argument('--extra', default='', help='extra train.py arguments, e.g. "--vla_period 5"')
    ap.add_argument('--vla_tag', default='', help='condition on <session>/vla_plans_<tag>.npz ("ours"); results go to runs/cv_vla/<tag>/')
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    global ROOT, GRID, SEED
    SEED = a.seed
    if a.vla_tag or a.name:
        ROOT = f'runs/cv_vla/{a.name or a.vla_tag}'
    if a.cells:
        GRID = [tuple(map(int, c.split('x'))) for c in a.cells]
    if a.summary:
        print(summary()); return

    ses = {r: sessions(r) for r in a.robots}
    n_frames = {r: sum(len(np.load(os.path.join(s, 'obs_dsmall252.npz'))['valid']) for s in ses[r]) for r in a.robots}
    data_gb = {r: n_frames[r] * 4 * 224 * 224 * 2 / 2 ** 30 * 1.05 for r in a.robots}     # float16 frames on the GPU
    jobs = []        # interleave robots so both progress
    for T, K in GRID:
        for f in range(FOLDS):
            for r in a.robots:
                if not os.path.exists(os.path.join(run_dir(r, T, K, f), 'test_metrics.json')):
                    jobs.append(dict(robot=r, T=T, K=K, f=f, tries=0))
    log(f'{len(jobs)} runs to do; sessions {({r: len(ses[r]) for r in a.robots})}, frames {n_frames}, '
        f'data on GPU {({r: round(v, 1) for r, v in data_gb.items()})} GB')
    if a.dry:
        for j in jobs:
            print(j)
        return
    while not a.yield_gb and any(os.path.exists(f'/proc/{p}') for p in a.wait_pids):
        time.sleep(30)

    peak = {}                     # T -> measured compute peak GB (process memory minus on-GPU data)
    compute_gb = lambda T: peak[T] if T in peak else (peak[max(peak)] * T / max(peak) + 1 if peak else 0.9 * T + 1.5)
    running, last_launch, env0 = [], 0.0, os.environ.copy()
    env0.update(CUDA_VISIBLE_DEVICES=str(a.gpu), LSMA_NUM_WORKERS='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')     # the container is capped at 64 cores; default pools (112 threads each) get throttled
    while jobs or running:
        free, procs = smi(a.gpu)
        for j in running[:]:
            j['peak'] = max(j.get('peak', 0), procs.get(j['p'].pid, 0) / 1024)
            rc = j['p'].poll()
            if rc is None:
                continue
            running.remove(j)
            ok = os.path.exists(os.path.join(j['rd'], 'test_metrics.json'))
            log(f"{'done' if ok else 'FAILED rc=' + str(rc)}  {j['rd']}  {time.time() - j['t0']:.0f}s  peak {j['peak']:.1f} GB ({j['store']})")
            if ok:
                c = j['peak'] - (data_gb[j['robot']] if j['store'] == 'gpu' else 0)
                peak[j['T']] = max(peak.get(j['T'], 0), c)
            elif j['tries'] < 2:
                j['tries'] += 1; jobs.append({k: j[k] for k in ('robot', 'T', 'K', 'f', 'tries')})
        # one launch per tick, and only after the previous launch has had time to claim its memory
        if jobs and len(running) < a.max_par and time.time() - last_launch > 150:
            budget = free / 1024 - a.reserve_gb
            budget -= sum(max(0.0, j['est'] - procs.get(j['p'].pid, 0) / 1024) for j in running)   # memory running jobs have yet to claim
            if a.yield_gb and any(os.path.exists(f'/proc/{p}') for p in a.wait_pids):
                def other(pid):
                    try: return a.yield_match in open(f'/proc/{pid}/cmdline').read()
                    except OSError: return False
                budget += sum(m for pid, m in procs.items() if other(pid)) / 1024 - a.yield_gb
            for j in jobs:
                c = compute_gb(j['T'])
                store = 'gpu' if a.store == 'auto' and c + data_gb[j['robot']] <= budget else 'cpu' if c <= budget else None
                if store is None:
                    continue
                jobs.remove(j)
                r, T, K, f = j['robot'], j['T'], j['K'], j['f']
                rd = run_dir(r, T, K, f); os.makedirs(rd, exist_ok=True)
                test = ses[r][f::FOLDS]; train = [s for s in ses[r] if s not in test]
                env = dict(env0, LSMA_GPU_DATA='1' if store == 'gpu' else 'cpu', LSMA_EVAL_BS=str(max(8, 128 // T)))
                cmd = [sys.executable, 'train.py', '--preset', 'improved', '--robot', r, '--sessions', *train, '--test_sessions', *test,
                       '--seq_len', str(T), '--frame_stride', str(K), '--obs_name', 'obs_dsmall252.npz', '--batch_size', '32',
                       '--val_frac', '0.1', '--seed', str(a.seed), '--run_dir', rd, *RECIPE[r], *a.extra.split(),
                       *(['--vla_tag', a.vla_tag, '--vla_latency', VLA_LATENCY[r]] if a.vla_tag else [])]
                j.update(p=subprocess.Popen(cmd, stdout=open(os.path.join(rd, 'train.log'), 'w'), stderr=subprocess.STDOUT, env=env),
                         rd=rd, store=store, t0=time.time(), est=c + (data_gb[r] if store == 'gpu' else 0))
                running.append(j); last_launch = time.time()
                log(f'start {rd}  est {c:.1f}+{data_gb[r] if store == "gpu" else 0:.1f} GB ({store}), free {free / 1024:.1f} GB, {len(jobs)} left')
                break
        time.sleep(20)
    log('all done'); print(summary())


if __name__ == '__main__':
    main()
