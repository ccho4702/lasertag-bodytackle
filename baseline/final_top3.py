"""Deployment checkpoints for the best history config(s) of hist_grid.py (see runs/cv_hist/summary.md).

For every config (seq_len T, frame_stride K) and robot it trains, on ALL sessions (5 % of each session's tail kept
as a sanity test, like final_v6):
    baseline_<robot>_T<T>K<K>   camera -> policy
    ours_<robot>_T<T>K<K>       same config + periodic SmolVLA plans (out-of-fold plans, tag oof_v3)
    prevact_<robot>_T<T>K<K>    baseline + its own previous commands as input (experimental: offline numbers are
                                teacher-forced, so only a rollout shows whether it helps or just repeats itself)
Recipe = the CV recipe (improved preset, batch 32, depth Small@252; chaser 30 epochs / select ae, evader 20 / bacc).

    python final_top3.py --gpu 1                  # train (resumable), two runs side by side while they fit
    python final_top3.py --export v2026.11.24     # light checkpoints + selection.json -> ../checkpoints/policy/<ver>/
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import torch

from hist_grid import RECIPE, VLA_LATENCY, log, sessions, smi

CONFIGS = [(8, 4)]                           # --configs 8x4 4x5 16x1 = the top 3 (within noise of each other in CV)
ROOT = 'runs/final_v7'
VLA_TAG = 'oof_v3'
# earlier runs with the identical recipe and data (final_v6, 2026-11-14): export() keeps whichever run has the better
# validation score (the selection criterion; same seed -> same validation split)
SAME_RECIPE = {'baseline_chaser_T8K4': ['runs/final_v6/baseline_cpu_chaser'], 'ours_chaser_T8K4': ['runs/final_v6/ours_cpu_chaser']}
NEED_GB = {4: 5.5, 8: 9.0, 16: 17.5}         # measured peaks with frames in host memory (hist_grid logs) + margin


def jobs():
    for T, K in CONFIGS:
        for robot in ('chaser', 'evader'):
            for kind in ('baseline', 'ours', 'prevact'):
                yield dict(name=f'{kind}_{robot}_T{T}K{K}', robot=robot, kind=kind, T=T, K=K)


def train(a):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), LSMA_NUM_WORKERS='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', LSMA_GPU_DATA='cpu')
    todo = [j for j in jobs() if not os.path.exists(os.path.join(ROOT, j['name'], 'test_metrics.json'))]
    todo.sort(key=lambda j: -j['T'])           # big ones first, small ones fill the gaps
    log(f'{len(todo)} runs to do')
    running, last = [], 0.0
    while todo or running:
        free, procs = smi(a.gpu)
        for j in running[:]:
            if j['p'].poll() is None:
                continue
            running.remove(j)
            ok = os.path.exists(os.path.join(ROOT, j['name'], 'test_metrics.json'))
            log(f"{'done' if ok else 'FAILED rc=' + str(j['p'].returncode)}  {j['name']}  {time.time() - j['t0']:.0f}s")
            if not ok and j.setdefault('tries', 0) < 1:
                j['tries'] += 1; todo.append(j)
        if todo and len(running) < a.max_par and time.time() - last > 150:
            budget = free / 1024 - a.reserve_gb - sum(max(0.0, NEED_GB[j['T']] - procs.get(j['p'].pid, 0) / 1024) for j in running)
            for j in todo:
                if NEED_GB[j['T']] > budget:
                    continue
                todo.remove(j)
                rd = os.path.join(ROOT, j['name']); os.makedirs(rd, exist_ok=True)
                r = j['robot']
                cmd = [sys.executable, 'train.py', '--preset', 'improved', '--robot', r, '--sessions', *sessions(r),
                       '--seq_len', str(j['T']), '--frame_stride', str(j['K']), '--obs_name', 'obs_dsmall252.npz', '--batch_size', '32',
                       '--test_frac', '0.05', '--val_frac', '0.1', '--run_dir', rd, *RECIPE[r],
                       *(['--vla_tag', VLA_TAG, '--vla_latency', VLA_LATENCY[r]] if j['kind'] == 'ours' else []),
                       *(['--prev_action'] if j['kind'] == 'prevact' else [])]
                j.update(p=subprocess.Popen(cmd, stdout=open(os.path.join(rd, 'train.log'), 'w'), stderr=subprocess.STDOUT,
                                            env=dict(env, LSMA_EVAL_BS=str(max(8, 128 // j['T'])))), t0=time.time())
                running.append(j); last = time.time()
                log(f"start {j['name']}  need {NEED_GB[j['T']]} GB, free {free / 1024:.1f} GB, {len(todo)} left")
                break
        time.sleep(20)
    log('all done')


def export(ver):
    out = os.path.join('..', 'checkpoints', 'policy', ver); os.makedirs(out, exist_ok=True)
    cv = {}
    for robot in ('chaser', 'evader'):          # 3-fold CV numbers of the same config (hist_grid.py), all seeds pooled
        for T, K in CONFIGS:
            for kind, root in (('baseline', 'runs/cv_hist'), ('ours', f'runs/cv_vla/{VLA_TAG}'), ('prevact', 'runs/cv_vla/base_prevact')):
                ms = [json.load(open(p)) for p in glob.glob(os.path.join(root, robot, f'T{T}_K{K}', 'f*', 'test_metrics.json'))]
                if ms:
                    cv[f'{kind}_{robot}_T{T}K{K}'] = dict(runs=len(ms), **{k: round(sum(m[k] for m in ms) / len(ms), 3) for k in
                                                          ('ae', 'ae_zero', 'acc_joint', 'acc_joint_major', 'bacc_joint', 'w_sign_acc')})
    sel = {'_note': 'History config seq_len T x frame_stride K chosen on the 3-fold cross-session CV grid. Every checkpoint is trained on '
                    'all sessions; "cv_3fold" is the held-out estimate for that config, "tail_test" only a sanity check (last 5 % of '
                    'each training session). "candidates" = validation score (val AE, or val balanced accuracy for evader) of every '
                    'finished run with this recipe; the best one is exported. ours_* need SmolVLA release ' + ver + ' (--vla).'}
    for j in jobs():
        cands = []
        for rd in [os.path.join(ROOT, j['name']), *SAME_RECIPE.get(j['name'], [])]:
            if os.path.exists(os.path.join(rd, 'test_metrics.json')):      # trained and finished
                ck = torch.load(os.path.join(rd, 'best.pt'), map_location='cpu', weights_only=False)
                c = ck['cfg']
                assert (c['seq_len'], c['frame_stride'], c['robot'], c['obs_name']) == (j['T'], j['K'], j['robot'], 'obs_dsmall252.npz'), rd
                cands.append((ck['best_val'], rd, ck))           # train.py stores it so that lower is better (bacc negated)
        if not cands:
            print('skip', j['name']); continue
        _, rd, ck = min(cands, key=lambda x: x[0])
        torch.save({k: v for k, v in ck.items() if k not in ('opt', 'sched')}, os.path.join(out, j['name'] + '.pt'))
        cfg, tm = json.load(open(os.path.join(rd, 'config.json'))), json.load(open(os.path.join(rd, 'test_metrics.json')))
        sel[j['name']] = dict(run=rd, candidates={r: round(abs(v), 4) for v, r, _ in cands}, best_val=round(abs(ck['best_val']), 4), seq_len=j['T'], frame_stride=j['K'], sessions=len(cfg['sessions']), obs_name=cfg['obs_name'],
                              epochs=cfg['epochs'], select=cfg['select'], vla_tag=cfg.get('vla_tag') or None,
                              vla_latency=cfg['vla_latency'] if cfg.get('vla_tag') else None, best_epoch=ck.get('epoch'),
                              tail_test={k: round(tm[k], 3) for k in ('ae', 'acc_joint', 'bacc_joint', 'w_sign_acc') if k in tm},
                              cv_3fold=cv.get(j['name']))
    json.dump(sel, open(os.path.join(out, 'selection.json'), 'w'), indent=1)
    print(json.dumps(sel, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--reserve_gb', type=float, default=3.0)
    ap.add_argument('--max_par', type=int, default=2)
    ap.add_argument('--export', default='', help='version folder name, e.g. v2026.11.24')
    ap.add_argument('--configs', nargs='*', default=[], help='T x K, e.g. 8x4 4x5 16x1 (default: 8x4)')
    a = ap.parse_args()
    if a.configs:
        CONFIGS[:] = [tuple(map(int, c.split('x'))) for c in a.configs]
    export(a.export) if a.export else train(a)
