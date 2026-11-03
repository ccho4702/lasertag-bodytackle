"""Sheets of frames that ended up with NO label but had rejected candidate boxes that CLIP thinks
could be Misty / brick. Each candidate is numbered so it can be added via the review json.

    python detector/review_missed.py --labels ...autolabels_clip.json --review ...manual_review.json --out <dir>
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image, ImageDraw
from build_yolo_ds import filter_boxes, resolve

ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True); ap.add_argument('--review', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
meta = json.load(open(a.labels))['meta']; rev = json.load(open(a.review))
rej = {(p, tuple(round(v, 4) for v in b)) for p, b in rev['reject']}
cands = []
for m in meta:
    kept = [b for b in filter_boxes(m['boxes']) if (m['image'], tuple(round(v, 4) for v in [round(x, 6) for x in b[:5]])) not in rej]
    if kept:
        continue
    cs = [b for b in m['boxes'] if (b[0] == 1 and b[8] >= 0.3) or (b[0] == 2 and b[6] in (2, 3) and b[7] >= 0.3)
          or (b[0] == 1 and b[6] == 8)]
    for b in cs:
        cands.append((m['image'], [round(v, 6) for v in b[:5]]))
json.dump(cands, open(os.path.join(a.out, 'missed_index.json'), 'w'))
TW, TH, COLS, PER = 192, 156, 8, 64
col = {1: 'red', 2: 'cyan'}
for s in range(0, len(cands), PER):
    ch = cands[s:s + PER]
    sheet = Image.new('RGB', (TW * COLS, TH * ((len(ch) + COLS - 1) // COLS)), 'black')
    for k, (p, b) in enumerate(ch):
        im = Image.open(resolve(p)).convert('RGB').resize((TW, TH - 12)); d = ImageDraw.Draw(im)
        c, x, y, w, h = b
        d.rectangle([(x - w / 2) * TW, (y - h / 2) * (TH - 12), (x + w / 2) * TW, (y + h / 2) * (TH - 12)], outline=col[int(c)], width=2)
        sheet.paste(im, ((k % COLS) * TW, (k // COLS) * TH + 12))
        ImageDraw.Draw(sheet).text(((k % COLS) * TW + 2, (k // COLS) * TH), f'{s + k} {"E" if c == 1 else "O"}', fill='yellow')
    sheet.save(os.path.join(a.out, f'missed_sheet{s // PER}.jpg'), quality=90)
print(len(cands), 'candidates in', len({p for p, _ in cands}), 'frames')
