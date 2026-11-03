"""Summarise every CV experiment in runs/cv (mean ± std over folds × seeds) + learning curve + timing.

    python report.py [--root runs/cv] [--md out.md]
"""
import argparse, csv, glob, json, os, re
import numpy as np
from cv import summarize

KEYS = ['ae', 'ae_zero', 'acc_joint', 'bacc_joint', 'acc_joint_major', 'acc_joint_vla_only']


def run_time(d):
    t = []
    for f in glob.glob(os.path.join(d, 'f*_s*', 'train_log.csv')):
        rows = list(csv.DictReader(open(f)))
        if rows and os.path.exists(os.path.join(os.path.dirname(f), 'test_metrics.json')):
            t.append(float(rows[-1]['elapsed_s']) / 60)
    return (np.mean(t), len(t)) if t else (float('nan'), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='runs/cv')
    ap.add_argument('--md', default='')
    a = ap.parse_args()
    lines = []
    hdr = f'{"experiment":26s} {"n":>3s} {"min/run":>8s} ' + ' '.join(f'{k:>17s}' for k in KEYS)
    lines += [hdr, '-' * len(hdr)]
    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, '*/'))):
        name = os.path.basename(d.rstrip('/'))
        s = summarize(d)
        if not s:
            continue
        n = next(iter(s.values()))[2]
        tm, _ = run_time(d)
        rows.append((name, n, tm, s))
        lines.append(f'{name:26s} {n:3d} {tm:8.1f} ' + ' '.join(f'{s[k][0]:.3f}±{s[k][1]:.3f}'.rjust(17) if k in s else ' ' * 17 for k in KEYS))
    # learning curve table: frac → ae / acc_joint per preset & robot
    lc = {}
    for name, n, tm, s in rows:
        m = re.match(r'(paper|improved)_(chaser|evader)(?:_frac(\d+))?$', name)
        if m:
            lc.setdefault((m[1], m[2]), {})[int(m[3] or 100)] = (s['ae'][0], s['acc_joint'][0], n)
    if lc:
        lines += ['', 'learning curve (fraction of training sessions → test AE / acc_joint, n runs)']
        for (p, r), v in sorted(lc.items()):
            lines.append(f'  {p:9s} {r:7s} ' + '  '.join(f'{f:3d}%: {ae:.3f}/{ac:.3f} (n={n})' for f, (ae, ac, n) in sorted(v.items())))
    txt = '\n'.join(lines)
    print(txt)
    if a.md:
        open(a.md, 'w').write('```\n' + txt + '\n```\n')


if __name__ == '__main__':
    main()
