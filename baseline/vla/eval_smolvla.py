"""Evaluate a fine-tuned SmolVLA checkpoint on the val/test time split.

    python -m vla.eval_smolvla --ckpt runs_vla/smolvla_c16/best --sessions ../data/sample/*/
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from vla.data import VLAChunkDataset
from vla.policies import load_finetuned
from vla.train_smolvla import chunk_metrics, predict_chunks, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--splits', nargs='+', default=['val', 'test'])
    a = ap.parse_args()
    meta = json.load(open(os.path.join(os.path.dirname(a.ckpt.rstrip('/')), 'vla_config.json')))
    device = 'cuda'
    policy = load_finetuned(a.ckpt, device).eval()
    tokens = tokenize(policy, meta['task'])
    out = {}
    for sp in a.splits:
        ds = VLAChunkDataset(a.sessions, sp, chunk=meta['chunk'], test_frac=meta['test_frac'],
                             val_frac=meta['val_frac'], robot=meta.get('robot', 'chaser'),
                             label_source=meta.get('label_source', 'requested'))
        P, Y = predict_chunks(policy, DataLoader(ds, batch_size=32, num_workers=int(os.environ.get('LSMA_NUM_WORKERS', 2))), tokens, device)
        m = chunk_metrics(P, Y)
        out[sp] = m
        print(f'{sp.upper()} acc_joint per k:', np.round(m['acc_joint_k'], 3).tolist())
        print(f'{sp.upper()} acc_joint mean={m["acc_joint_mean"]:.3f} k0={m["acc_joint_k0"]:.3f} '
              f'bacc_k0={m["bacc_joint_k0"]:.3f} majority={m["majority_k0"]:.3f}')
    json.dump(out, open(os.path.join(os.path.dirname(a.ckpt.rstrip('/')), 'test_metrics.json'), 'w'),
              indent=2)


if __name__ == '__main__':
    main()
