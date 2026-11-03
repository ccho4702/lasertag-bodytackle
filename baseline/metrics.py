"""Offline metrics shared by train.py and eval.py. Actions in robot units [v, ω] (robots.py).

ae : the paper's Action Error — MSE between student and expert action in the normalized
     action space (a / act_scale per dim), so v and ω (and % vs mm/s) are comparable.
acc_* / bacc_* : after snapping both prediction and label to the robot's command levels.
"""
import numpy as np


def label_stats(actions_pct, name='train', names=('v', 'w')):
    """Print a short distribution summary — helps decide between regression and --head cls."""
    lines = [f'Label stats [{name}]  (n={len(actions_pct):,})']
    for i, k in enumerate(names):
        a = actions_pct[:, i]
        uniq = np.unique(np.round(a, 2))
        q = np.percentile(a, [5, 25, 50, 75, 95])
        lines.append(
            f'  {k:11s} mean={a.mean():7.2f} std={a.std():6.2f}  '
            f'q5/25/50/75/95={q.round(1).tolist()}  '
            f'unique={len(uniq)}  zero={np.mean(np.abs(a) < 1e-3)*100:.0f}%  '
            f'|a|>=99={np.mean(np.abs(a) >= 99)*100:.0f}%')
    print('\n'.join(lines))


def snap(a, levels):
    """Map continuous actions (..., 2) to the nearest discrete command level of each dim.
    levels: one list for both dims, or a pair of per-dim lists (e.g. ω {-25, 0, 30})."""
    a = np.asarray(a, dtype=np.float64)
    if np.ndim(levels[0]) == 0:
        levels = (levels, levels)
    out = np.empty_like(a)
    for d in range(a.shape[-1]):
        lv = np.asarray(levels[d], dtype=np.float64)
        out[..., d] = lv[np.abs(a[..., d, None] - lv).argmin(-1)]
    return out


def compute_metrics(pred, truth, train_mean=None, enemy_visible=None, sign_thr=None, levels=None,
                    scale=None):
    """pred, truth: (N, 2) robot units.  enemy_visible: (N,) bool.  train_mean: (2,).
    levels: per-dim discrete command values → adds snapped accuracies, incl. the
    majority-class baseline acc_joint_major.  scale: (2,) act_scale → adds ae (normalized MSE)."""
    m = {}
    scale = np.ones(2) if scale is None else np.asarray(scale, np.float64)
    if sign_thr is None:
        sign_thr = 0.2 * scale[1]
    if len(pred):
        m['ae'] = float(np.mean(((pred - truth) / scale) ** 2))
        m['ae_zero'] = float(np.mean((truth / scale) ** 2))
    if levels is not None and len(pred):
        ps, ts = snap(pred, levels), snap(truth, levels)
        m['acc_v'] = float(np.mean(ps[:, 0] == ts[:, 0]))
        m['acc_w'] = float(np.mean(ps[:, 1] == ts[:, 1]))
        m['acc_joint'] = float(np.mean((ps == ts).all(1)))
        cls, inv, cnt = np.unique(ts, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        m['acc_joint_major'] = float(cnt.max() / len(ts))
        hit = (ps == ts).all(1)
        # balanced accuracy: mean per-class recall (robust to the train/test class shift); chance ≈ 1/#classes
        m['bacc_joint'] = float(np.mean([hit[inv == c].mean() for c in range(len(cls))]))
        if enemy_visible is not None and np.any(enemy_visible):
            ev = np.asarray(enemy_visible, bool)
            m['acc_w_enemy'] = float(np.mean(ps[ev, 1] == ts[ev, 1]))
    err = pred - truth
    m['mse'] = float(np.mean(err ** 2))
    m['mse_v'] = float(np.mean(err[:, 0] ** 2))
    m['mse_w'] = float(np.mean(err[:, 1] ** 2))
    m['mae_v'] = float(np.mean(np.abs(err[:, 0])))
    m['mae_w'] = float(np.mean(np.abs(err[:, 1])))
    m['mse_zero'] = float(np.mean(truth ** 2))
    if train_mean is not None:
        m['mse_mean'] = float(np.mean((truth - np.asarray(train_mean)[None]) ** 2))
    turning = np.abs(truth[:, 1]) > sign_thr
    m['w_sign_acc'] = float(np.mean(np.sign(pred[turning, 1]) == np.sign(truth[turning, 1]))) \
        if turning.any() else float('nan')
    if enemy_visible is not None:
        ev = np.asarray(enemy_visible, bool)
        m['enemy_frac'] = float(ev.mean())
        m['mse_enemy'] = float(np.mean(err[ev] ** 2)) if ev.any() else float('nan')
        m['mse_w_enemy'] = float(np.mean(err[ev, 1] ** 2)) if ev.any() else float('nan')
    return m


def format_metrics(m, prefix=''):
    keys = ['ae', 'ae_zero', 'acc_joint', 'acc_joint_major', 'bacc_joint', 'bacc_joint_vla_only',
            'acc_joint_vla_only', 'vla_coverage',
            'acc_v', 'acc_w', 'acc_w_enemy',
            'mse', 'mse_v', 'mse_w', 'mae_v', 'mae_w', 'w_sign_acc',
            'mse_enemy', 'mse_w_enemy', 'mse_zero', 'mse_mean']
    parts = [f'{k}={m[k]:.3f}' for k in keys if k in m]
    return prefix + '  '.join(parts)
