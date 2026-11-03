"""GPU batch augmentations for heatmap sequences.

seq: (B, T, C, H, W)   act: (B, T+1, 2) normalized ([:, :, 1] = angular velocity)
"""
import torch
import torch.nn.functional as F


def mirror(w, levels):
    """Mirror normalized angular commands through the robot's level set: the left command maps to
    the right one (e.g. Misty ω +30 % ↔ -25 %), so a flipped label is still a real command."""
    if levels is None:
        return -w
    lv = torch.as_tensor(levels, dtype=w.dtype, device=w.device).sort().values
    idx = (w[..., None] - lv).abs().argmin(-1)
    return lv[len(lv) - 1 - idx]


def hflip(seq, act, p=0.5, cond=None, w_levels=None):
    """Mirror the image horizontally and mirror the angular command (whole sequence).

    Only applied to sequences that contain at least one detection: with nothing in view the
    demonstrator's search behaviour (e.g. always turning left) need not be mirror-symmetric.
    """
    if p <= 0:
        return (seq, act) if cond is None else (seq, act, cond)
    has_det = seq.amax(dim=(1, 2, 3, 4)) > 0.1
    m = (torch.rand(seq.shape[0], device=seq.device) < p) & has_det
    if m.any():
        seq = seq.clone(); act = act.clone()
        seq[m] = seq[m].flip(-1)
        act[m, :, 1] = mirror(act[m, :, 1], w_levels)
        if cond is not None:
            cond = cond.clone()
            cond[m, :, 1] = -cond[m, :, 1]    # VLA plan's angular command mirrors too
    return (seq, act) if cond is None else (seq, act, cond)


def jitter(seq, max_px=2.0):
    """Independent small translation per frame — mimics YOLO box-centre noise.

    Per-frame (not per-sequence) so that the mean bearing, and thus the label, is unchanged.
    """
    if max_px <= 0:
        return seq
    B, T, C, H, W = seq.shape
    x = seq.reshape(B * T, C, H, W)
    shift = (torch.rand(B * T, 2, device=seq.device) * 2 - 1) * max_px
    theta = torch.zeros(B * T, 2, 3, device=seq.device, dtype=seq.dtype)
    theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    theta[:, 0, 2] = shift[:, 0] * 2 / W
    theta[:, 1, 2] = shift[:, 1] * 2 / H
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
    return x.view(B, T, C, H, W)


def det_dropout(seq, p=0.1, n_hm=None):
    """Zero a whole heatmap channel of a frame with prob p — mimics missed detections.
    Only the first n_hm (target heatmap) channels; the depth channel is left untouched."""
    if p <= 0:
        return seq
    B, T, C = seq.shape[:3]
    n_hm = C if n_hm is None else n_hm
    keep = (torch.rand(B, T, n_hm, 1, 1, device=seq.device) > p).to(seq.dtype)
    seq = seq.clone()
    seq[:, :, :n_hm] = seq[:, :, :n_hm] * keep
    return seq
