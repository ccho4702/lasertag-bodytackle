"""Filter auto labels (Grounding DINO boxes + CLIP crop verdicts) and write an Ultralytics YOLO
dataset split by session.

    python -m detector.verify_clip --labels ../data_download/evader_yolo/autolabels.json
    python detector/build_yolo_ds.py --labels ../data_download/evader_yolo/autolabels_clip.json \
        --out ../data_download/evader_yolo/ds --val_sessions <session> ...

Rules were tuned by visual QA of the Cozmo-view frames (see README in this folder):
  enemy    : CLIP 'white robot' prob >= 0.7 (any DINO score >= 0.25), or >= 0.45 with DINO >= 0.45,
             or CLIP argmax 'white plastic box' (close-up Misty base) with DINO >= 0.35;
             not inside an obstacle box (brick texture).
  obstacle : DINO >= 0.45 and CLIP argmax is a brick box / wooden surface, or DINO >= 0.3 with
             CLIP brick prob >= 0.5; not the long arena
             wall (aspect > 4) and not the 'wall + floor' full-width box.
  both     : not the dark strip artefact at the left image border.
"""
import argparse, json, os, collections

# image paths in the label jsons are relative to the projects root (parent of baseline/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

NAMES = {0: 'ally', 1: 'enemy', 2: 'obstacle'}
OBST_OK = {2, 3, 4}          # CLIP prompt ids: blue brick, yellow brick, plain wooden (near bricks)


def keep(c, x, y, w, h, s, j, pj, ok):
    if x - w / 2 < 0.01 and w < 0.07:                 # left-border dark strip artefact
        return False
    if c == 1:
        if (ok >= 0.7 and s >= 0.25) or (ok >= 0.45 and s >= 0.45):
            return True
        # close-up Misty base: CLIP calls it 'white plastic box'
        return j == 8 and pj >= 0.4 and s >= 0.35
    if c == 2:
        strong = s >= 0.45 and j in OBST_OK
        weak_brick = s >= 0.3 and j in (2, 3) and pj >= 0.5
        if not (strong or weak_brick):
            return False
        if (w * 320) / max(h * 240, 1) > 4.0:          # long flat = arena wall edge
            return False
        if w > 0.9 and y + h / 2 > 0.97:               # wall + floor box
            return False
        return True
    return False


def _xyxy(b):
    return b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2


def _inter(a, b):
    ax0, ay0, ax1, ay1 = _xyxy(a); bx0, by0, bx1, by1 = _xyxy(b)
    return max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))


def iou(a, b):
    i = _inter(a, b)
    return i / (a[3] * a[4] + b[3] * b[4] - i + 1e-9)


def filter_boxes(boxes):
    boxes = sorted([b for b in boxes if keep(*b)], key=lambda b: -b[5])
    boxes = [b for b in boxes
             if not (b[0] == 1 and any(o[0] == 2 and _inter(b, o) / (b[3] * b[4] + 1e-9) > 0.9 for o in boxes))]
    final = []
    for b in boxes:                                     # per-class NMS + drop boxes nested in a kept one
        if all(f[0] != b[0] or (iou(b, f) < 0.5 and _inter(b, f) / (b[3] * b[4] + 1e-9) < 0.85) for f in final):
            final.append(b)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='autolabels_clip.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_sessions', nargs='+', default=[])
    ap.add_argument('--review', default='', help='manual review json: {"reject": [[image, [c,x,y,w,h]]...], '
                                                 '"add": [[image, [c,x,y,w,h]]...]}')
    a = ap.parse_args()
    meta = json.load(open(a.labels))['meta']
    rev = json.load(open(a.review)) if a.review else {'reject': [], 'add': []}
    rej = {(p, tuple(round(v, 4) for v in b)) for p, b in rev['reject']}
    add = collections.defaultdict(list)
    for p, b in rev['add']:
        add[p].append(b)
    stats = collections.Counter()
    for split in ('train', 'val'):
        os.makedirs(os.path.join(a.out, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(a.out, 'labels', split), exist_ok=True)
    for m in meta:
        sess = m['image'].split(os.sep)[-3]
        split = 'val' if sess in a.val_sessions else 'train'
        final = [[round(v, 6) for v in b[:5]] for b in filter_boxes(m['boxes'])]
        n0 = len(final)
        final = [b for b in final if (m['image'], tuple(round(v, 4) for v in b)) not in rej] + add[m['image']]
        stats['manual_rejected'] += n0 - len(final) + len(add[m['image']]); stats['manual_added'] += len(add[m['image']])
        name = f'{sess}_{os.path.basename(m["image"])[:-4]}'
        dst = os.path.join(a.out, 'images', split, name + '.png')
        if not os.path.lexists(dst):
            os.symlink(resolve(m['image']), dst)
        with open(os.path.join(a.out, 'labels', split, name + '.txt'), 'w') as f:
            for c, x, y, w, h in final:
                f.write(f'{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
                stats[(split, NAMES[int(c)])] += 1
        stats[(split, 'images')] += 1
        stats[(split, 'empty')] += not final
    with open(os.path.join(a.out, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n')
        for k, v in NAMES.items():
            f.write(f'  {k}: {v}\n')
    for k in sorted(stats, key=str):
        print(k, stats[k])


if __name__ == '__main__':
    main()
