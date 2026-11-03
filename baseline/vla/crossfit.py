"""Out-of-fold (OOF) VLA plans via session-level cross-fitting.

Why: plans extracted by a VLA on frames it was trained on are near-perfect (in-sample), so a
heatmap policy trained on them learns to copy the VLA and collapses when the VLA is wrong at
test time. Here every session's plans come from a VLA that never saw that session.

    python -m vla.crossfit --sessions ../data/sample/*/ --tag oof --folds 3

For fold f: train SmolVLA on all sessions outside fold f (last val_frac of each used for model
selection), then extract plans for the sessions in fold f → <session>/vla_plans_<tag>.npz.
Resumable: a fold is skipped when all its sessions already have the plans file.
Then train the heatmap policy with  --vla_tag <tag>.
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--tag', default='oof')
    ap.add_argument('--folds', type=int, default=0, help='0 = leave-one-session-out')
    ap.add_argument('--out', default='runs_vla/crossfit')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--fold', type=int, default=-1, help='run only this fold (for parallel launchers)')
    a, extra = ap.parse_known_args()   # extra args are forwarded to train_smolvla

    sessions = [s.rstrip('/') for s in a.sessions]
    k = a.folds or len(sessions)
    folds = [sessions[i::k] for i in range(k)]
    for f, held in enumerate(folds):
        if a.fold >= 0 and f != a.fold:
            continue
        if all(os.path.exists(os.path.join(s, f'vla_plans_{a.tag}.npz')) for s in held):
            print(f'[fold {f}] done, skipping'); continue
        train = [s for s in sessions if s not in held]
        out = os.path.join(a.out, a.tag, f'fold{f}')
        print(f'[fold {f}] train on {len(train)} sessions, plans for {[os.path.basename(s) for s in held]}',
              flush=True)
        if not os.path.exists(os.path.join(out, 'best')):
            subprocess.run([sys.executable, '-m', 'vla.train_smolvla', '--sessions', *train,
                            '--out', out, '--test_frac', '0', '--steps', str(a.steps),
                            '--eval_every', str(a.eval_every), '--chunk', str(a.chunk), *extra],
                           check=True)
        subprocess.run([sys.executable, '-m', 'vla.extract_plans', '--ckpt', os.path.join(out, 'best'),
                        '--tag', a.tag, '--sessions', *held], check=True)


if __name__ == '__main__':
    main()
