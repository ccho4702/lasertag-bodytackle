"""Preprocess (paper, Sec. III-D / Fig. 4): onboard RGB frame → student-policy input tensor.

Per frame (only the robot's own camera image is used):
    1. YOLO on the full upright frame (robot-specific checkpoint, see robots.py).
    2. Target heatmaps, one channel per class: a Gaussian kernel of size s = λ / D (output px)
       placed at each box centre (peak 1; std from the kernel size as in cv2.getGaussianKernel).
       D = f·W / w is the distance estimated from the class's known physical width W, the box
       width w (px) and the focal length f (from the FOV), clipped to [d_min, d_max]
       (paper: 0.5–3 m). Closer robots → larger kernels.
    3. Depth Anything V2 (Large) relative depth, clipped to its 5th/95th percentile and
       normalised to [0, 1].
    4. Resized to size×size (paper: 224) and stacked along the channel axis.

Writes <session>/obs.npz:
    obs      (N, C, S, S) float16   channels listed in `channels` (default enemy, ally, obstacle, depth;
                                    the paper's input is enemy + ally + depth, chosen at train time)
    channels (C,) str
    actions  (N, 2) float32         [v, ω] in robot units (robots.py), aligned with labels.csv rows
    valid    (N,) bool              False where the image could not be read (obs = 0)
    boxes    (N, K, 6) float32      raw detections [cls, conf, x0, y0, x1, y1] (normalised), K = max/frame
Rows always stay aligned with labels.csv (nothing is dropped or rewritten).

    python preprocess.py --robot chaser --session "../data_download/chaser/*/*/*/"
    python preprocess.py --robot evader --session ../data_download/evader/*/*/
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

import robots

ImageFile.LOAD_TRUNCATED_IMAGES = True

HM_CLASSES = ('enemy', 'ally', 'obstacle')
DEPTH_MODEL = 'depth-anything/Depth-Anything-V2-Large-hf'


def focal_px(width_px, hfov_deg):
    return (width_px / 2) / np.tan(np.deg2rad(hfov_deg) / 2)


def target_heatmaps(boxes, robot, img_w, size, lam, d_min, d_max, classes=HM_CLASSES):
    """boxes: (K, 6) [cls_name_idx, conf, x0, y0, x1, y1] normalised → (len(classes), size, size)."""
    hm = np.zeros((len(classes), size, size), np.float32)
    f = focal_px(img_w, robot.hfov_deg)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    for c, _, x0, y0, x1, y1 in boxes:
        c = int(c)
        w_px = max((x1 - x0) * img_w, 1.0)
        D = np.clip(f * robot.width_m[classes[c]] / w_px, d_min, d_max)
        k = max(lam / D, 3.0)                         # kernel size (output px)
        sig = 0.3 * ((k - 1) * 0.5 - 1) + 0.8         # cv2.getGaussianKernel convention
        cx, cy = (x0 + x1) / 2 * size, (y0 + y1) / 2 * size
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig * sig))
        win = (np.abs(xx - cx) <= k / 2) & (np.abs(yy - cy) <= k / 2)
        hm[c] = np.maximum(hm[c], np.where(win, g, 0.0))
    return hm


class Perception:
    """YOLO + DATv2 → obs tensor. Shared by preprocess (offline) and deploy (online) so the
    policy sees identical inputs in training and on the robot."""

    def __init__(self, robot, size=224, lam=40.0, d_min=0.3, d_max=3.0, conf=0.25, iou=0.45,
                 depth=True, yolo=None, device=None,
                 depth_model=None, depth_size=None):
        from ultralytics import YOLO
        self.robot = robots.get(robot)
        self.size, self.lam, self.d_min, self.d_max, self.conf, self.iou = size, lam, d_min, d_max, conf, iou
        self.device = device or robots.best_device()
        self.yolo = YOLO(yolo or self.robot.yolo)
        names = self.yolo.names
        missing = set(HM_CLASSES) - set(names.values())
        if missing:
            sys.exit(f'YOLO {yolo or self.robot.yolo} lacks classes {missing}; names={names}')
        self.cls_map = {k: HM_CLASSES.index(v) for k, v in names.items() if v in HM_CLASSES}
        self.depth = None
        if depth:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            # depth_model / depth_size must match training (stored in the checkpoint cfg):
            #   default = Large at the processor's 518 px; CPU option = Small at 252 px (≈40 ms on CPU)
            self.depth_model = depth_model or DEPTH_MODEL
            self.depth_size = int(depth_size) if depth_size else None
            self.dproc = AutoImageProcessor.from_pretrained(self.depth_model)
            self.depth = AutoModelForDepthEstimation.from_pretrained(
                self.depth_model, torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32).to(self.device).eval()
        self.channels = HM_CLASSES + (('depth',) if depth else ())

    @torch.no_grad()
    def __call__(self, frames):
        """frames: list of HxWx3 uint8 RGB → obs (B, C, S, S) float32, boxes list of (K, 6)."""
        # ultralytics treats numpy inputs as BGR (cv2 convention) — training images were read from
        # files the same way, so convert our RGB frames
        bgr = [np.ascontiguousarray(f[..., ::-1]) for f in frames]
        res = self.yolo.predict(bgr, imgsz=self.robot.yolo_imgsz, conf=self.conf, iou=self.iou,
                                verbose=False, device=self.device)
        boxes, hms = [], []
        for fr, r in zip(frames, res):
            b = np.zeros((0, 6), np.float32)
            if r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                keep = np.array([c in self.cls_map for c in cls], bool)
                if keep.any():
                    b = np.concatenate([np.array([self.cls_map[c] for c in cls[keep]], np.float32)[:, None],
                                        r.boxes.conf.cpu().numpy()[keep, None],
                                        r.boxes.xyxyn.cpu().numpy()[keep]], 1).astype(np.float32)
            boxes.append(b)
            hms.append(target_heatmaps(b, self.robot, fr.shape[1], self.size, self.lam, self.d_min, self.d_max))
        obs = torch.from_numpy(np.stack(hms))
        if self.depth is not None:
            kw = {'size': {'height': self.depth_size, 'width': self.depth_size}} if self.depth_size else {}
            inp = self.dproc(images=frames, return_tensors='pt', **kw).to(self.device)
            d = self.depth(pixel_values=inp['pixel_values'].to(self.depth.dtype)).predicted_depth.float()   # (B, h, w)
            d = F.interpolate(d[:, None], size=(self.size, self.size), mode='bilinear', align_corners=False)[:, 0]
            flat = d.flatten(1)
            lo = torch.quantile(flat, 0.05, dim=1)[:, None, None]
            hi = torch.quantile(flat, 0.95, dim=1)[:, None, None]
            d = ((d.clamp(min=lo, max=hi) - lo) / (hi - lo).clamp(min=1e-6)).cpu()
            obs = torch.cat([obs, d[:, None]], 1)
        return obs.numpy(), boxes


def load_rgb(path):
    try:
        return np.array(Image.open(path).convert('RGB')), True
    except Exception as e:  # keep alignment with labels.csv
        print(f'  [WARN] {path}: {e}')
        return None, False


def run_session(sdir, perc, label_source, batch, force, out_name='obs.npz'):
    import pandas as pd
    out = os.path.join(sdir, out_name)
    if os.path.exists(out) and not force:
        print(f'exists: {out}  (--force to redo)'); return
    df = pd.read_csv(os.path.join(sdir, 'labels.csv'))
    actions = perc.robot.load_actions(sdir, label_source)
    N = len(df)
    obs = np.zeros((N, len(perc.channels), perc.size, perc.size), np.float16)
    valid = np.zeros(N, bool)
    allboxes = [np.zeros((0, 6), np.float32)] * N
    for s in range(0, N, batch):
        idx, frames = [], []
        for i in range(s, min(N, s + batch)):
            fr, ok = load_rgb(os.path.join(sdir, df.image[i]))
            if ok:
                idx.append(i); frames.append(fr)
        if not frames:
            continue
        o, bx = perc(frames)
        obs[idx] = o.astype(np.float16); valid[idx] = True
        for i, b in zip(idx, bx):
            allboxes[i] = b
    K = max(1, max(len(b) for b in allboxes))
    boxes = np.full((N, K, 6), -1, np.float32)
    for i, b in enumerate(allboxes):
        boxes[i, :len(b)] = b
    np.savez_compressed(out, obs=obs, channels=np.array(perc.channels), actions=actions,
                        valid=valid, boxes=boxes, robot=perc.robot.name, label_source=label_source,
                        depth_model=getattr(perc, 'depth_model', ''), depth_size=getattr(perc, 'depth_size', None) or 0)
    vis = {c: float((obs[valid, k].reshape(valid.sum(), -1).max(1) > 0.1).mean())
           for k, c in enumerate(perc.channels) if c != 'depth'}
    print(f'{out}: {N} frames ({(~valid).sum()} unreadable)  visible ' +
          ' '.join(f'{c}={v:.0%}' for c, v in vis.items()))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--robot', required=True, choices=list(robots.ROBOTS))
    ap.add_argument('--session', nargs='+', required=True, help='session dirs (glob patterns allowed)')
    ap.add_argument('--yolo', default=None, help='override the robot profile checkpoint')
    ap.add_argument('--label_source', default='requested', choices=['requested', 'executed'])
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--lam', type=float, default=40.0, help='kernel scale λ (px·m): kernel size = λ / D')
    ap.add_argument('--d_min', type=float, default=0.3)
    ap.add_argument('--d_max', type=float, default=3.0)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--no_depth', action='store_true')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--depth_model', default=None, help=f'default {DEPTH_MODEL}; CPU: depth-anything/Depth-Anything-V2-Small-hf')
    ap.add_argument('--depth_size', type=int, default=None, help='depth input size (default: processor 518); CPU: 252')
    ap.add_argument('--out_name', default='obs.npz', help='e.g. obs_dsmall252.npz (train.py --obs_name)')
    a = ap.parse_args()
    sessions = sorted({p.rstrip('/') for s in a.session for p in (glob.glob(s) or [s])})
    perc = Perception(a.robot, a.size, a.lam, a.d_min, a.d_max, a.conf, a.iou, not a.no_depth, a.yolo,
                      depth_model=a.depth_model, depth_size=a.depth_size)
    print(f'{a.robot}: {len(sessions)} sessions  channels={perc.channels}  yolo={perc.robot.yolo}')
    for s in sessions:
        run_session(s, perc, a.label_source, a.batch, a.force, a.out_name)


if __name__ == '__main__':
    main()
