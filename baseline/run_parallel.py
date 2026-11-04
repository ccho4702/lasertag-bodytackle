"""Run the whole experiment grid of run_paper_cv.sh as many concurrent processes on ONE big GPU.

A single run (ResNet18, batch 64 / SmolVLA, batch 16) uses only a few GB and a small slice of a
B200, so runs are launched side by side with a dependency graph:

    cache_images(robot) → crossfit fold 0..4 (robot) → every *_vla_* run of that robot
    paper / improved runs (no VLA) start immediately.

    source ../env/activate.sh
    CUDA_VISIBLE_DEVICES=3 python run_parallel.py --max_par 24 --seeds 0 1 2

Outputs/layout are identical to run_paper_cv.sh / cv.py (runs/cv/<name>/f<fold>_s<seed>/,
runs_vla/crossfit_<robot>/oof/fold<f>/), resumable (finished runs are skipped), and at the end
cv.py is invoked per experiment to write runs/cv/<name>/summary.json.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

PY = sys.executable
LAT = {'chaser': 4, 'evader': 3}          # SmolVLA latency in frames (342 ms on the deployment RTX 3090)
FOLDS = 5
MEM = {'paper': 9, 'improved': 17, 'vla': 5}      # peak GB per process (LSMA_EVAL_BS=64, measured on B200)
# other VLA backbones (lerobot 0.4.4): tag → (pretrained, crossfit peak GB, latency frames per robot, train args)
VLA_MODELS = {
    # SmolVLA re-fine-tuned with the body-tackle prompt (robots.py task, TASK.md)
    'oof_ram':  ('lerobot/smolvla_base', 5, {'chaser': 4, 'evader': 3}, []),
    # π0 / π0.5 dropped (2026-11-03): no gain over SmolVLA, 4-5x the VRAM and ~3x the latency (TASK.md §7)
}
FRACS = (0.25, 0.5, 0.75)                          # learning curve: fraction of the training sessions


def sessions(robot):
    pat = '../data_download/chaser/*/*/*/' if robot == 'chaser' else '../data_download/evader/*/*/'
    return [s.rstrip('/') for s in sorted(glob.glob(pat))]       # same order as `ls -d ... | sort`


# experiment name → extra train.py flags (per robot).  Order = launch priority.
# body-tackle runs on the ~11 GB left next to vLLM on GPU3 (run with LSMA_AMP=1): batch 32, 60 epochs
SMALL = ['--batch_size', '32', '--epochs', '60']


def experiments(robot):
    L = str(LAT[robot])
    return {
        f'paper_{robot}':         ['--preset', 'paper'],
        f'paper_vla_{robot}':     ['--preset', 'paper', '--vla_tag', 'oof', '--vla_latency', L],
        f'paper_vla_p5_{robot}':  ['--preset', 'paper', '--vla_tag', 'oof', '--vla_period', '5', '--vla_latency', L],
        # HANDOFF §8.1: configuration sized to the data (3-class head, spatial-softmax encoder, aug, EMA)
        f'improved_{robot}':      ['--preset', 'improved'],
        f'improved_vla_{robot}':  ['--preset', 'improved', '--vla_tag', 'oof', '--vla_latency', L],
        # VLA query period sweep (Δ frames; chunk 16 ≥ Δ + L + jitter) and latency sensitivity
        # (L=4/3 frames = 342 ms on the deployment RTX 3090; L=2 ≈ 141 ms measured on the B200)
        f'paper_vla_p3_{robot}':      ['--preset', 'paper', '--vla_tag', 'oof', '--vla_period', '3', '--vla_latency', L],
        f'improved_vla_p5_{robot}':   ['--preset', 'improved', '--vla_tag', 'oof', '--vla_period', '5', '--vla_latency', L],
        f'improved_vla_p3_{robot}':   ['--preset', 'improved', '--vla_tag', 'oof', '--vla_period', '3', '--vla_latency', L],
        f'paper_vla_lat2_{robot}':    ['--preset', 'paper', '--vla_tag', 'oof', '--vla_latency', '2'],
        f'paper_vla_lat2_p5_{robot}': ['--preset', 'paper', '--vla_tag', 'oof', '--vla_period', '5', '--vla_latency', '2'],
        # TASK.md (body tackle): camera frames lag the commands by ~2 frames → label shift -2,
        # and the SmolVLA plans re-trained with the body-tackle prompt
        f'improved_amp_{robot}':        ['--preset', 'improved', *SMALL],   # reference for the runs below (same small-GPU settings)
        # CPU deployment: Depth-Anything-V2-Small at 252 px (≈40 ms on CPU vs 1.6 s for Large@518)
        f'improved_dsmall_{robot}':     ['--preset', 'improved', '--obs_name', 'obs_dsmall252.npz', *SMALL],
        f'improved_shift2_{robot}':     ['--preset', 'improved', '--label_shift', '-2', *SMALL],
        f'improved_ram_{robot}':        ['--preset', 'improved', '--vla_tag', 'oof_ram', '--vla_latency', L, *SMALL],
        f'improved_ram_shift2_{robot}': ['--preset', 'improved', '--vla_tag', 'oof_ram', '--vla_latency', L, '--label_shift', '-2', *SMALL],
    } | {
        # π0 / π0.5 as the periodic VLA (chunk 16 ≥ Δ + L + jitter → Δ ≤ 16 - L - 1)
        f'paper_{tag[4:]}_{robot}':       ['--preset', 'paper', '--vla_tag', tag, '--vla_latency', str(lat[robot]),
                                           '--vla_period', str(min(10, 16 - lat[robot] - 1))]
        for tag, (_, _, lat, _) in VLA_MODELS.items()
    } | {
        f'paper_{tag[4:]}_p5_{robot}':    ['--preset', 'paper', '--vla_tag', tag, '--vla_latency', str(lat[robot]),
                                           '--vla_period', '5']
        for tag, (_, _, lat, _) in VLA_MODELS.items()
    }


class Job:
    def __init__(self, name, cmd, log, done, deps=(), gpu=True, mem=0):
        self.name, self.cmd, self.log, self.done_fn, self.deps, self.gpu = name, cmd, log, done, list(deps), gpu
        self.mem = mem
        self.proc, self.status, self.t0 = None, 'pending', None

    def done(self):
        return self.done_fn()


def build(args):
    jobs = {}
    for R in args.robots:                 # pass 1: cache + crossfit (gate the VLA runs → highest priority)
        S = sessions(R)
        folds = [S[f::FOLDS] for f in range(FOLDS)]
        # 1) VLA image cache (CPU)
        cache = Job(f'cache_{R}', [PY, '-m', 'vla.cache_images', '--session', *S],
                    f'runs_vla/cache_{R}.log',
                    lambda S=S: all(os.path.exists(os.path.join(s, 'images_vla.npy')) for s in S), gpu=False)
        jobs[cache.name] = cache
        # 2) OOF SmolVLA plans, one process per fold
        for f, held in enumerate(folds):
            j = Job(f'crossfit_{R}_f{f}',
                    [PY, '-m', 'vla.crossfit', '--sessions', *S, '--tag', 'oof', '--folds', str(FOLDS),
                     '--out', f'runs_vla/crossfit_{R}', '--robot', R, '--fold', str(f), '--steps', str(args.vla_steps)],
                    f'runs_vla/crossfit_{R}/oof/fold{f}.log',
                    lambda held=held: all(os.path.exists(os.path.join(s, 'vla_plans_oof.npz')) for s in held),
                    deps=[cache.name], mem=MEM['vla'])
            jobs[j.name] = j
        for tag in args.vla_models:       # π0 / π0.5 OOF plans (patched-transformers interpreter)
            pre, mem, _, extra = VLA_MODELS[tag]
            for f, held in enumerate(folds):
                j = Job(f'crossfit_{tag}_{R}_f{f}',
                        [os.path.abspath(args.pi_python) if 'pi0' in pre else PY, '-m', 'vla.crossfit', '--sessions', *S, '--tag', tag,
                         '--folds', str(FOLDS), '--out', f'runs_vla/crossfit_{R}', '--robot', R, '--fold', str(f),
                         '--steps', str(args.vla_steps), '--pretrained', pre, *extra],
                        f'runs_vla/crossfit_{R}/{tag}/fold{f}.log',
                        lambda held=held, tag=tag: all(os.path.exists(os.path.join(s, f'vla_plans_{tag}.npz')) for s in held),
                        deps=[cache.name], mem=mem)
                jobs[j.name] = j
    for R in args.robots:                 # pass 2: heatmap policy runs (cv.py layout)
        S = sessions(R)
        folds = [S[f::FOLDS] for f in range(FOLDS)]
        vla_deps = [f'crossfit_{R}_f{f}' for f in range(FOLDS)]
        for name, flags in experiments(R).items():
            if args.only and not any(o in name for o in args.only):
                continue
            tag = flags[flags.index('--vla_tag') + 1] if '--vla_tag' in flags else None
            if tag in VLA_MODELS and tag not in args.vla_models:
                continue
            deps = [] if tag is None else vla_deps if tag == 'oof' else [f'crossfit_{tag}_{R}_f{f}' for f in range(FOLDS)]
            for f in range(FOLDS):
                test = folds[f]
                train = [s for s in S if s not in test]
                for seed in args.seeds:
                    rd = os.path.join('runs/cv', name, f'f{f}_s{seed}')
                    j = Job(f'{name}_f{f}_s{seed}',
                            [PY, 'train.py', '--sessions', *train, '--test_sessions', *test,
                             '--run_dir', rd, '--seed', str(seed), '--robot', R, *flags],
                            os.path.join(rd, 'train.log'),
                            lambda rd=rd: os.path.exists(os.path.join(rd, 'test_metrics.json')),
                            deps=deps, mem=MEM[flags[1]])
                    jobs[j.name] = j
    # pass 3: learning curve — same folds/test sessions, a random subset of the training sessions
    import numpy as np
    for R in ([] if args.no_lc else args.robots):
        S = sessions(R)
        folds = [S[f::FOLDS] for f in range(FOLDS)]
        for preset in args.lc_presets:
            for frac in args.lc_fracs:
                name = f'{preset}_{R}_frac{int(frac*100)}'
                if args.only and not any(o in name for o in args.only):
                    continue
                for f in range(FOLDS):
                    test = folds[f]
                    train = [s for s in S if s not in test]
                    perm = np.random.RandomState(1000 + f).permutation(len(train))
                    sub = [train[i] for i in sorted(perm[:max(2, int(round(frac * len(train))))])]
                    for seed in args.lc_seeds:
                        rd = os.path.join('runs/cv', name, f'f{f}_s{seed}')
                        j = Job(f'{name}_f{f}_s{seed}',
                                [PY, 'train.py', '--sessions', *sub, '--test_sessions', *test,
                                 '--run_dir', rd, '--seed', str(seed), '--robot', R, '--preset', preset],
                                os.path.join(rd, 'train.log'),
                                lambda rd=rd: os.path.exists(os.path.join(rd, 'test_metrics.json')), mem=MEM[preset])
                        jobs[j.name] = j
    if args.only:   # keep only the cache/crossfit jobs some selected run depends on
        needed = {d for j in jobs.values() if not j.name.startswith(('crossfit_', 'cache_')) for d in j.deps}
        needed |= {d for n in list(needed) for d in jobs[n].deps}
        jobs = {n: j for n, j in jobs.items() if not n.startswith(('crossfit_', 'cache_')) or n in needed}
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robots', nargs='+', default=['evader', 'chaser'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--max_par', type=int, default=20, help='max concurrent GPU processes')
    ap.add_argument('--mem_budget', type=float, default=165, help='GB of GPU memory to fill (sum of per-job peaks)')
    ap.add_argument('--lc_presets', nargs='+', default=['paper', 'improved'])
    ap.add_argument('--lc_fracs', nargs='+', type=float, default=list(FRACS))
    ap.add_argument('--lc_seeds', nargs='+', type=int, default=[0, 1], help='seeds for the learning-curve runs')
    ap.add_argument('--vla_steps', type=int, default=1000)
    ap.add_argument('--only', nargs='*', default=[], help='substrings of experiment names to run')
    ap.add_argument('--mem_cap', type=float, default=0, help='cap per-job memory estimates (GB; small eval batch)')
    ap.add_argument('--vla_models', nargs='*', default=[], choices=list(VLA_MODELS), help='extra VLA backbones (crossfit + runs)')
    ap.add_argument('--pi_python', default='../.venv-pi/bin/python',
                    help='interpreter with lerobot[pi] (patched transformers) used for the π0/π0.5 crossfit jobs')
    ap.add_argument('--no_lc', action='store_true', help='skip the learning-curve runs')
    ap.add_argument('--threads', type=int, default=8, help='OMP/MKL threads per process')
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    jobs = build(args)
    if args.mem_cap:
        for j in jobs.values():
            j.mem = min(j.mem, args.mem_cap)
    for j in jobs.values():
        if j.done():
            j.status = 'done'
    todo = [j for j in jobs.values() if j.status != 'done']
    print(f'{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run', flush=True)
    if args.dry:
        for j in todo:
            print(' ', j.name, '<-', j.deps)
        return

    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               PYTHONUNBUFFERED='1')
    t_start = time.time()
    while True:
        running = [j for j in jobs.values() if j.status == 'running']
        for j in running:
            rc = j.proc.poll()
            if rc is None:
                continue
            j.status = 'done' if rc == 0 and j.done() else 'failed'
            print(f'[{time.strftime("%H:%M:%S")}] {j.status.upper():6s} {j.name}  ({(time.time()-j.t0)/60:.1f} min)',
                  flush=True)
        running = [j for j in jobs.values() if j.status == 'running']
        n_gpu = sum(j.gpu for j in running)
        used = sum(j.mem for j in running)
        mp_file = os.environ.get('LSMA_MAXPAR_FILE', 'runs/.max_par')
        if os.path.exists(mp_file):                  # echo 32 > runs/.max_par  to change on the fly
            try:
                args.max_par = int(open(mp_file).read().strip())
            except ValueError:
                pass
        for j in jobs.values():
            if j.status != 'pending':
                continue
            if any(jobs[d].status == 'failed' for d in j.deps):
                j.status = 'skipped'; print(f'SKIP {j.name} (failed dependency)', flush=True); continue
            if not all(jobs[d].status == 'done' for d in j.deps):
                continue
            if j.gpu and (n_gpu >= args.max_par or used + j.mem > args.mem_budget):
                continue
            os.makedirs(os.path.dirname(j.log) or '.', exist_ok=True)
            j.proc = subprocess.Popen(j.cmd, stdout=open(j.log, 'w'), stderr=subprocess.STDOUT, env=env)
            j.status, j.t0 = 'running', time.time()
            n_gpu += j.gpu; used += j.mem
            print(f'[{time.strftime("%H:%M:%S")}] START  {j.name}  (gpu procs {n_gpu}, mem {used:.0f} GB)', flush=True)
        st = [j.status for j in jobs.values()]
        if 'running' not in st and 'pending' not in st:
            break
        if 'running' not in st:   # only pending jobs with unmet deps that can never be met
            for j in jobs.values():
                if j.status == 'pending':
                    j.status = 'skipped'
            break
        time.sleep(10)

    print(f'\nall jobs finished in {(time.time()-t_start)/3600:.2f} h', flush=True)
    bad = [j.name for j in jobs.values() if j.status in ('failed', 'skipped')]
    if bad:
        print('FAILED/SKIPPED:', *bad, sep='\n  ')
    # summaries (cv.py skips finished runs and writes summary.json)
    import json
    import cv as cvmod
    rows = []
    for d in sorted(glob.glob('runs/cv/*/')):
        name = os.path.basename(d.rstrip('/'))
        if not glob.glob(f'{d}/f*_s*/test_metrics.json'):
            continue
        s = cvmod.summarize(d)
        json.dump(s, open(os.path.join(d, 'summary.json'), 'w'), indent=2)
        rows.append((name, s))
    keys = ['ae', 'ae_zero', 'acc_joint', 'bacc_joint', 'acc_joint_major', 'acc_joint_vla_only']
    print(f'{"experiment":28s} n   ' + '  '.join(f'{k:>18s}' for k in keys))
    for name, s in rows:
        n = next(iter(s.values()))[2] if s else 0
        print(f'{name:28s} {n:<3d} ' + '  '.join(f'{s[k][0]:.3f}±{s[k][1]:.3f}'.rjust(18) if k in s else ' ' * 18 for k in keys))


if __name__ == '__main__':
    main()
