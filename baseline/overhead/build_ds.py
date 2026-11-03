"""Overhead (bird's-eye) detector dataset: filter Grounding DINO candidates with geometric rules and
per-session consistency, then write an Ultralytics YOLO dataset split by session.

    python -m overhead.build_ds --cands ../data_download/overhead_yolo/candidates.json \
        --out ../data_download/overhead_yolo/ds [--val_frac 0.2] [--review manual_review.json]

Classes: 0 misty, 1 cozmo, 2 obstacle (brick block).
Rules (tuned by eye on the 1280x720 overhead frames):
  arena ROI  : x < ROI_X (the desk/mat on the right is not part of the arena)
  misty      : label contains 'white robot', side 90–230 px, aspect in [0.5, 2]; the best-scoring box;
               if the ArUco centre (labels.csv position_*, tag 8) is valid it must lie inside the box
  cozmo      : label 'small toy robot' / 'toy robot' without 'white robot', side 30–95 px, not inside the
               misty box; best-scoring box
  obstacle   : NOT from DINO (blue bricks score < 0.3): the bricks are the only saturated objects on the
               white floor, so they are segmented by HSV saturation on the per-session median frame
               (static obstacles) → one box set per session, side 50–320 px
Session consistency: the robot that does not move in a session (Cozmo in chaser sessions, Misty in evader
sessions) gets the per-session median box when it was detected in >= 15% of the frames; frames whose box is
> 60 px from the median are replaced by it, frames without a detection get it too.
Frames where misty or cozmo is still unknown are left out of the dataset (no false 'absent' labels).
"""
import argparse, collections, json, os, random
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NAMES = ['misty', 'cozmo', 'obstacle']
ROI_X = 1060


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def inside(pt, b, m=0):
    return b[0] - m <= pt[0] <= b[2] + m and b[1] - m <= pt[1] <= b[3] + m


def side_ok(b, lo, hi, asp=(0.5, 2.0)):
    w, h = b[2] - b[0], b[3] - b[1]
    return lo <= w <= hi and lo <= h <= hi and asp[0] <= w / max(h, 1) <= asp[1]


def pick_frame(cands, aruco=None):
    """cands: [(label, score, xyxy)] → dict class -> [xyxy] (misty/cozmo at most one each)."""
    cands = [(l, s, b) for l, s, b in cands if b[0] < ROI_X and b[2] < ROI_X + 40]
    misty = [(s, b) for l, s, b in cands if 'white robot' in l and side_ok(b, 90, 230)]
    if aruco is not None:
        m2 = [(s, b) for s, b in misty if inside(aruco, b, 10)]
        misty = m2 if m2 else []          # ArUco says Misty is elsewhere → don't trust
    out = {}
    if misty:
        out['misty'] = max(misty)[1]
    cz = [(s, b) for l, s, b in cands if 'toy robot' in l and 'white robot' not in l and side_ok(b, 30, 95)]
    if 'misty' in out:
        cz = [(s, b) for s, b in cz if iou(b, out['misty']) < 0.05 and not inside(centre(b), out['misty'], 10)]
    if cz:
        out['cozmo'] = max(cz)[1]
    out['obstacle'] = []
    return out


def obstacles_by_color(sdir, frames, n=12, s_thr=100, v_thr=50, area_min=2500):
    """Static bricks of a session: HSV saturation mask on the median of n frames → boxes."""
    import cv2
    paths = [os.path.join(sdir, 'overhead_images', os.path.basename(f)) for f in frames]
    paths = paths[::max(1, len(paths) // n)][:n]
    ims = [cv2.imread(p) for p in paths]
    ims = [im for im in ims if im is not None]
    if not ims:
        return []
    med = np.median(np.stack(ims), 0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > s_thr) & (hsv[..., 2] > v_thr)).astype(np.uint8)
    mask[:, ROI_X:] = 0
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    out = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= area_min and 50 <= w <= 320 and 50 <= h <= 320:
            out.append([float(x), float(y), float(x + w), float(y + h)])
    return out


def median_box(boxes):
    return np.median(np.array(boxes), 0).tolist() if boxes else None


def session_of(rel):
    return os.path.dirname(os.path.dirname(rel))        # .../<session>/overhead_images/x.jpg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_frac', type=float, default=0.2, help='fraction of sessions held out for val')
    ap.add_argument('--review', default='', help='json {"reject": [[image, cls, xyxy]], "add": [[image, cls, xyxy]]}')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1, help='keep every k-th frame of each session (consecutive frames are near-duplicates)')
    a = ap.parse_args()
    cands = json.load(open(a.cands))
    by_sess = collections.defaultdict(list)
    for d in cands:
        by_sess[session_of(d['image'])].append(d)

    labels, stats = {}, collections.Counter()
    for sess, frames in sorted(by_sess.items()):
        sdir = os.path.join(ROOT, sess)
        df = pd.read_csv(os.path.join(sdir, 'labels.csv'))
        is_chaser = 'requested_command' in df.columns
        aruco = {}
        if 'position_valid' in df and is_chaser:        # chaser sessions: tag 8 = Misty
            for img, ok, x, y in zip(df['overhead_image'], df['position_valid'], df['position_x_px'], df['position_y_px']):
                if ok:
                    aruco[os.path.basename(img)] = (float(x), float(y))
        per = {}
        for d in sorted(frames, key=lambda d: d['image'])[::a.stride]:
            fn = os.path.basename(d['image'])
            per[d['image']] = pick_frame(d['cands'], aruco.get(fn))
        static = 'cozmo' if is_chaser else 'misty'       # the opponent does not move in these recordings
        found = [p[static] for p in per.values() if static in p]
        med = median_box(found) if len(found) >= 0.15 * len(per) else None
        if med is not None:
            for p in per.values():
                b = p.get(static)
                if b is None or np.hypot(*(np.subtract(centre(b), centre(med)))) > 60:
                    p[static] = med
                    stats[f'{static}_filled'] += 1
        else:
            stats[f'{static}_unknown_sessions'] += 1
        keep = obstacles_by_color(sdir, [os.path.basename(k) for k in per])
        for p in per.values():
            p['obstacle'] = [b for b in keep if all(iou(b, p[c]) < 0.3 for c in ('misty', 'cozmo') if c in p)]
        for img, p in per.items():
            if 'misty' not in p or 'cozmo' not in p:
                stats['dropped_frames'] += 1
                continue
            labels[img] = p
            for c in NAMES:
                stats[c] += len(p[c]) if c == 'obstacle' else int(c in p)
        stats['frames'] += len(per)
    print('label stats:', dict(stats))

    if a.review and os.path.exists(a.review):
        rv = json.load(open(a.review))
        for img, c, b in rv.get('reject', []):
            if c == 'obstacle':
                labels[img]['obstacle'] = [o for o in labels[img]['obstacle'] if iou(o, b) < 0.5]
            else:
                labels[img].pop(c, None)
        for img, c, b in rv.get('add', []):
            if c == 'obstacle':
                labels[img]['obstacle'].append(b)
            else:
                labels[img][c] = b

    sess = sorted(by_sess)
    random.Random(a.seed).shuffle(sess)
    nval = max(1, int(round(a.val_frac * len(sess))))
    val = set(sess[:nval])
    for split in ('train', 'val'):
        os.makedirs(os.path.join(a.out, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(a.out, split, 'labels'), exist_ok=True)
    n = collections.Counter()
    for img, p in labels.items():
        split = 'val' if session_of(img) in val else 'train'
        stem = img.replace('/', '__').replace(' ', '_')[:-4]
        src = os.path.join(ROOT, img)
        dst = os.path.join(a.out, split, 'images', stem + '.jpg')
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        W, H = 1280, 720
        lines = []
        for ci, c in enumerate(NAMES):
            boxes = p['obstacle'] if c == 'obstacle' else ([p[c]] if c in p else [])
            for x0, y0, x1, y1 in boxes:
                lines.append(f'{ci} {(x0+x1)/2/W:.5f} {(y0+y1)/2/H:.5f} {(x1-x0)/W:.5f} {(y1-y0)/H:.5f}')
        open(os.path.join(a.out, split, 'labels', stem + '.txt'), 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        n[split] += 1
    with open(os.path.join(a.out, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(a.out)}\ntrain: train/images\nval: val/images\nnames:\n' +
                ''.join(f'  {i}: {c}\n' for i, c in enumerate(NAMES)))
    json.dump(labels, open(os.path.join(a.out, 'labels.json'), 'w'))
    json.dump(sorted(val), open(os.path.join(a.out, 'val_sessions.json'), 'w'), indent=1)
    print(f'wrote {dict(n)} → {a.out}  (val sessions: {len(val)})')


if __name__ == '__main__':
    main()
