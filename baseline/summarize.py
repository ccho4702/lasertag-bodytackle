"""Aggregate runs/<group>/<name>/s<seed>/test_metrics.json into a mean ± std table.

    python summarize.py runs/real
"""
import glob
import json
import os
import sys

import numpy as np

KEYS = ['ae', 'ae_zero', 'acc_joint', 'bacc_joint', 'acc_joint_major', 'acc_joint_vla_only', 'bacc_joint_vla_only',
        'acc_w_enemy', 'mse']


def main(root):
    rows = {}
    for f in sorted(glob.glob(os.path.join(root, '*', 's*', 'test_metrics.json'))):
        name = f.split(os.sep)[-3]
        rows.setdefault(name, []).append(json.load(open(f)))
    print(f'{"run":22s} n  ' + '  '.join(f'{k:>18s}' for k in KEYS))
    for name, ms in rows.items():
        cells = []
        for k in KEYS:
            v = [m[k] for m in ms if k in m and m[k] == m[k]]
            cells.append(f'{np.mean(v):.3f}±{np.std(v):.3f}'.rjust(18) if v else ' ' * 18)
        print(f'{name:22s} {len(ms)}  ' + '  '.join(cells))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/real')
