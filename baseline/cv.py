"""Leave-one-session-out cross-validation runner for train.py.

    python cv.py --sessions ../data_download/evader/*/*/ --folds 5 --name paper_evader -- --robot evader --preset paper
    python cv.py --sessions ../data/sample/*/ --name improved_vla -- --preset improved --vla_tag oof

Fold f trains on every session except sessions[f::k] (val = last val_frac of each train session)
and tests on the held-out session(s). Results: runs/cv/<name>/f<fold>_s<seed>/test_metrics.json,
summary printed per run + pooled over folds/seeds. Resumable (finished folds are skipped).
With --vla_tag use OOF plans (vla/crossfit.py) with the same --folds, so no test session was
seen by the VLA that produced its plans.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

KEYS = ['ae', 'ae_zero', 'acc_joint', 'bacc_joint', 'acc_joint_major', 'acc_joint_vla_only', 'bacc_joint_vla_only']


def summarize(root):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, 'f*_s*', 'test_metrics.json')))]
    out = {}
    for k in KEYS:
        v = [r[k] for r in rows if k in r and r[k] == r[k]]
        if v:
            out[k] = (float(np.mean(v)), float(np.std(v)), len(v))
    return out


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    argv = argv[:argv.index('--')] if '--' in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--folds', type=int, default=0, help='0 = leave-one-session-out')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--root', default='runs/cv')
    a = ap.parse_args(argv)

    sessions = [s.rstrip('/') for s in a.sessions]
    k = a.folds or len(sessions)
    root = os.path.join(a.root, a.name)
    for f in range(k):
        test = sessions[f::k]
        train = [s for s in sessions if s not in test]
        for seed in a.seeds:
            rd = os.path.join(root, f'f{f}_s{seed}')
            if os.path.exists(os.path.join(rd, 'test_metrics.json')):
                continue
            os.makedirs(rd, exist_ok=True)
            print(f'[{a.name}] fold {f} seed {seed}', flush=True)
            with open(os.path.join(rd, 'train.log'), 'w') as log:
                subprocess.run([sys.executable, 'train.py', '--sessions', *train, '--test_sessions', *test,
                                '--run_dir', rd, '--seed', str(seed), *extra],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    s = summarize(root)
    print(f'[{a.name}] ' + '  '.join(f'{key}={m:.3f}±{sd:.3f}(n={n})' for key, (m, sd, n) in s.items()))
    json.dump(s, open(os.path.join(root, 'summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
