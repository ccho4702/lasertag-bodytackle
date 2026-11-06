"""How much does more data help?  Cross-session CV where the TEST sessions are fixed per fold and only the
amount of training data changes.  Sessions come in two groups (e.g. opponent static / opponent moving);
each fold's test set takes 1/folds of both groups, and every condition is evaluated on both groups separately.

    python data_scaling.py --robot chaser \
        --group static "../data_download/chaser/*/상대방 움직임 없음/*/" \
        --group moving "../data_download/chaser/*/상대방 움직임 있음/*/" \
        --conds static moving25 moving50 all --use_folds 0 1 2 --root runs/cv_moving

conditions:  static       train on the static group only
             movingNN     static group + NN % of the moving group's training sessions (fixed random subset)
             all          both groups
Results: <root>/<cond>/f<k>/{test_metrics,test_static,test_moving}.json and a summary table on stdout.
Run dirs that already have results are skipped, so the script can be re-run to add conditions.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np


def log(m):
    print(time.strftime('%H:%M'), m, flush=True)


def evaluate_subset(ckpt, sessions, out):
    import torch
    from torch.utils.data import DataLoader
    import train
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    b = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = argparse.Namespace(**b['cfg'])
    cfg.test_sessions = sessions
    _, _, test_ds = train.build_datasets(cfg)
    m = train.build_model(cfg).to(dev)
    if b.get('ema') is not None:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema), use_buffers=True)
        ema.load_state_dict(b['ema'])
        m = ema.module
    else:
        m.load_state_dict(b['model'])
    tm = train.evaluate(m, DataLoader(test_ds, batch_size=int(os.environ.get('LSMA_EVAL_BS', 64))), dev,
                        b['train_mean'], cfg.robot)
    with open(out, 'w') as f:
        json.dump({k: float(v) for k, v in tm.items() if np.isscalar(v)}, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--group', nargs=2, action='append', metavar=('NAME', 'GLOB'), required=True,
                    help='exactly two groups: first = base group, second = added group')
    ap.add_argument('--exclude', nargs='*', default=[], help='substrings of session paths to drop')
    ap.add_argument('--conds', nargs='+', default=['static', 'all'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--use_folds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--root', required=True)
    ap.add_argument('--max_par', type=int, default=1, help='concurrent training processes')
    ap.add_argument('--retries', type=int, default=3, help='re-run a failed (e.g. OOM) training this many times')
    ap.add_argument('--train_flags', default='--preset improved --obs_name obs_dsmall252.npz --batch_size 32 --epochs 60 --seed 0')
    ap.add_argument('--eval_only', nargs=3, metavar=('CKPT', 'OUT', 'SESSIONS_JSON'), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.eval_only:
        ckpt, out, sj = a.eval_only
        evaluate_subset(ckpt, json.loads(sj), out)
        return

    assert len(a.group) == 2, 'give exactly two --group'
    (g0, p0), (g1, p1) = a.group
    pick = lambda p: [s.rstrip('/') for s in sorted(glob.glob(p)) if not any(x in s for x in a.exclude)]
    S0, S1 = pick(p0), pick(p1)
    log(f'{a.robot}: {g0} {len(S0)} sessions, {g1} {len(S1)} sessions')

    jobs = []
    for f in a.use_folds:
        t0, t1 = S0[f::a.folds], S1[f::a.folds]
        tr0, tr1 = [s for s in S0 if s not in t0], [s for s in S1 if s not in t1]
        perm = np.random.RandomState(1000 + f).permutation(len(tr1))
        for cond in a.conds:
            if cond == g0 or cond == 'static':
                train = tr0
            elif cond == 'all':
                train = tr0 + tr1
            elif cond.startswith(g1) or cond.startswith('moving'):
                frac = int(cond[len(g1) if cond.startswith(g1) else len('moving'):]) / 100
                train = tr0 + [tr1[i] for i in sorted(perm[:max(1, round(frac * len(tr1)))])]
            else:
                raise SystemExit(f'unknown condition {cond}')
            rd = os.path.join(a.root, cond, f'f{f}')
            jobs.append((rd, train, t0, t1))

    def run(rd, train, t0, t1):
        if not os.path.exists(f'{rd}/test_metrics.json'):
            for attempt in range(a.retries):
                shutil.rmtree(rd, ignore_errors=True)
                os.makedirs(rd)
                cmd = [sys.executable, 'train.py', '--robot', a.robot, '--sessions', *train, '--test_sessions', *t0, *t1,
                       '--run_dir', rd, *a.train_flags.split()]
                with open(f'{rd}/train.log', 'w') as fo:
                    yield subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT)
                ok = os.path.exists(f'{rd}/test_metrics.json')
                log(f'{rd} ({len(train)} train sessions) {"done" if ok else "FAILED"}')
                if ok:
                    break
                time.sleep(120)
        for name, ss in [(g0, t0), (g1, t1)]:
            out = f'{rd}/test_{name}.json'
            if os.path.exists(f'{rd}/best.pt') and not os.path.exists(out):
                with open(f'{rd}/eval_{name}.log', 'w') as fo:
                    yield subprocess.Popen([sys.executable, __file__, '--robot', a.robot, '--group', g0, p0, '--group', g1, p1,
                                            '--root', a.root, '--eval_only', f'{rd}/best.pt', out, json.dumps(ss)],
                                           stdout=fo, stderr=subprocess.STDOUT)

    pending = [run(*j) for j in jobs]
    active = []                                   # (generator, current process)
    while pending or active:
        while pending and len(active) < a.max_par:
            g = pending.pop(0)
            p = next(g, None)
            if p is not None:
                active.append((g, p))
        for g, p in list(active):
            if p.poll() is not None:
                active.remove((g, p))
                q = next(g, None)
                if q is not None:
                    active.append((g, q))
        time.sleep(10)

    print(f'\n{"train":12s} {"test":8s} {"n":>2s} {"acc_joint":>13s} {"bacc_joint":>13s} {"majority":>13s} {"ae":>13s}')
    for cond in a.conds:
        for sub, fn in [('all', 'test_metrics.json'), (g0, f'test_{g0}.json'), (g1, f'test_{g1}.json')]:
            R = [json.load(open(p)) for p in sorted(glob.glob(f'{a.root}/{cond}/f*/{fn}'))
                 if int(p.split('/f')[-1].split('/')[0]) in a.use_folds]
            if not R:
                continue
            g = lambda k: f'{np.mean([r[k] for r in R]):.3f}±{np.std([r[k] for r in R]):.3f}'
            print(f'{cond:12s} {sub:8s} {len(R):2d} {g("acc_joint"):>13s} {g("bacc_joint"):>13s} '
                  f'{g("acc_joint_major"):>13s} {g("ae"):>13s}')


if __name__ == '__main__':
    main()
