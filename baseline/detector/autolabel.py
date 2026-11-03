"""Auto-label Cozmo-view frames with Grounding DINO → YOLO txt labels (classes as misty_yolo26s_v1).

    python -m detector.autolabel --sessions ../data_download/evader/*/*/ --out ../data_download/evader_yolo

Only the robot's own camera images are used. Classes: 0 ally, 1 enemy (Misty), 2 obstacle (brick blocks).
"""
import argparse, glob, json, os
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # projects root

# prompt phrase -> class id
PHRASES = {'white robot': 1, 'brick block': 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='IDEA-Research/grounding-dino-base')
    ap.add_argument('--box_thr', type=float, default=0.30)
    ap.add_argument('--text_thr', type=float, default=0.25)
    ap.add_argument('--max_area', type=float, default=0.6, help='drop boxes covering > this frac of image')
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda'
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).to(dev).eval()
    text = '. '.join(PHRASES) + '.'
    imgs = sorted(p for s in a.sessions for p in glob.glob(os.path.join(s, 'images', '*.png')))
    if a.limit: imgs = imgs[::max(1, len(imgs) // a.limit)][:a.limit]
    os.makedirs(a.out, exist_ok=True)
    meta = []
    for i in range(0, len(imgs), a.bs):
        batch = imgs[i:i + a.bs]
        ims = [Image.open(p).convert('RGB') for p in batch]
        inp = proc(images=ims, text=[text] * len(ims), return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=a.box_thr, text_threshold=a.text_thr,
            target_sizes=[im.size[::-1] for im in ims])
        for p, im, r in zip(batch, ims, res):
            W, H = im.size
            boxes = []
            for b, s, lab in zip(r['boxes'].tolist(), r['scores'].tolist(), r['text_labels'] if 'text_labels' in r else r['labels']):
                c = next((v for k, v in PHRASES.items() if k in lab or lab in k and lab), None)
                if c is None: continue
                x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
                if (x1 - x0) * (y1 - y0) > a.max_area * W * H: continue
                boxes.append((c, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H, s))
            meta.append({'image': os.path.relpath(os.path.abspath(p), ROOT), 'boxes': boxes})
        print(f'{i + len(batch)}/{len(imgs)}', flush=True)
    json.dump(meta, open(os.path.join(a.out, 'autolabels.json'), 'w'))


if __name__ == '__main__':
    main()
