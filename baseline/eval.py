"""Evaluation and prediction plot for VisuomotorPolicy.

Usage
    python eval.py --session ../data/sample/S3 --ckpt runs/improved/best.pt
    python eval.py --session ../data/sample/S3 --ckpt runs/cross/best.pt --split all   # held-out session
"""
import argparse
import os

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import robots
from dataset import HeatmapDataset
from metrics import compute_metrics, format_metrics
from model import build_model
from train import predict_dataset


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get('cfg', {})
    model = build_model({**cfg, 'pretrained': False}).to(device)
    rb = robots.get(cfg.get('robot', 'chaser'))
    state = ckpt['model']
    if ckpt.get('ema') is not None and not args.no_ema:
        # AveragedModel state: 'module.' prefixed weights + 'n_averaged'
        state = {k[len('module.'):]: v for k, v in ckpt['ema'].items() if k.startswith('module.')}
    model.load_state_dict(state)
    model.eval()
    print(f'Loaded {args.ckpt}  (epoch {ckpt.get("epoch", "?")}, '
          f'encoder={cfg.get("encoder", "cnn")}, head={cfg.get("head", "tanh")})')

    ds = HeatmapDataset([args.session], seq_len=cfg.get('seq_len', 8), split=args.split,
                        test_frac=args.test_frac, val_frac=cfg.get('val_frac', 0.15),
                        channels=model.channels, vla_tag=cfg.get('vla_tag') or None,
                        vla_period=cfg.get('vla_period', 10), vla_latency=cfg.get('vla_latency', 4),
                        vla_phase=cfg.get('vla_eval_phase', 0))
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 2)))
    preds, truths, vis = predict_dataset(model, loader, device)
    from train import evaluate
    m = evaluate(model, loader, device, ckpt.get('train_mean'), rb.name)
    print(f'\n=== {os.path.basename(args.session)} [{args.split}]  n={len(preds)} ===')
    print(format_metrics(m))
    if cfg.get('prev_action'):
        print('NOTE: prev_action model evaluated with ground-truth previous actions (teacher forcing); '
              'closed-loop performance will be worse.')

    os.makedirs(os.path.dirname(os.path.abspath(args.plot_out)), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(truths))
    for i, (ax, label) in enumerate(zip(axes, [f'{n} [{rb.units}]' for n in rb.action_names])):
        ax.plot(t, truths[:, i], label='ground truth', lw=1, alpha=0.8)
        ax.plot(t, preds[:, i], label='predicted', lw=1, alpha=0.8)
        if i == 1:
            s_ = float(model.act_scale[1])
            ax.fill_between(t, -s_, s_, where=vis, color='C3', alpha=0.08, step='mid',
                            label='enemy visible')
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('window index')
    fig.suptitle(f'{os.path.basename(os.path.dirname(args.ckpt))}  —  '
                 f'AE {m["ae"]:.4f} (zero {m["ae_zero"]:.4f})')
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=100)
    print(f'Plot saved: {args.plot_out}')


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--session', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--plot_out', default='outputs/eval_plot.png')
    ap.add_argument('--split', default='test', choices=['all', 'train', 'val', 'test'],
                    help="temporal split to evaluate; use 'all' for a held-out session")
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--no_ema', action='store_true', help='use raw weights even if EMA exists')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
