"""Render numbered crop sheets of the filtered boxes for manual review.

    python detector/review_sheet.py --labels ../data_download/evader_yolo/autolabels_clip.json --cls 1 --out <dir>
Writes <dir>/c<cls>_sheet<k>.jpg (80 crops each, index printed) and <dir>/c<cls>_index.json.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image, ImageDraw
from build_yolo_ds import filter_boxes, resolve

ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True); ap.add_argument('--cls', type=int, required=True)
ap.add_argument('--out', required=True); ap.add_argument('--per', type=int, default=80)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
meta = json.load(open(a.labels))['meta']
items = [(m['image'], [round(v, 6) for v in b[:5]]) for m in meta for b in filter_boxes(m['boxes']) if b[0] == a.cls]
json.dump(items, open(os.path.join(a.out, f'c{a.cls}_index.json'), 'w'))
TW, TH, COLS = 160, 120, 10
for s in range(0, len(items), a.per):
    chunk = items[s:s + a.per]
    sheet = Image.new('RGB', (TW * COLS, TH * ((len(chunk) + COLS - 1) // COLS)), 'black')
    for k, (p, b) in enumerate(chunk):
        im = Image.open(resolve(p)).convert('RGB'); W, H = im.size; _, x, y, w, h = b
        x0, y0, x1, y1 = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H
        cx0, cy0 = max(0, x0 - 0.3 * w * W - 10), max(0, y0 - 0.3 * h * H - 10)
        cx1, cy1 = min(W, x1 + 0.3 * w * W + 10), min(H, y1 + 0.3 * h * H + 10)
        t = im.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
        sc = min(TW / t.width, (TH - 12) / t.height); t = t.resize((max(1, int(t.width * sc)), max(1, int(t.height * sc))))
        d = ImageDraw.Draw(t)
        d.rectangle([(x0 - cx0) * sc, (y0 - cy0) * sc, (x1 - cx0) * sc - 1, (y1 - cy0) * sc - 1], outline='lime', width=2)
        sheet.paste(t, ((k % COLS) * TW, (k // COLS) * TH + 12))
        ImageDraw.Draw(sheet).text(((k % COLS) * TW + 2, (k // COLS) * TH), str(s + k), fill='yellow')
    sheet.save(os.path.join(a.out, f'c{a.cls}_sheet{s // a.per}.jpg'), quality=90)
print(len(items))
