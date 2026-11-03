"""Second-stage check of auto labels: classify each box crop with CLIP and store per-box verdicts.

    python -m detector.verify_clip --labels ../data_download/evader_yolo/autolabels.json
Writes autolabels_clip.json (each box gets an extra field: CLIP label index + prob).
"""
import argparse, json
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_yolo_ds import resolve  # noqa: E402

PROMPTS = ['a photo of a white humanoid robot', 'a photo of a red fire extinguisher',
           'a photo of a blue brick-patterned box', 'a photo of a yellow brick-patterned box',
           'a photo of a plain wooden wall', 'a photo of an empty white floor',
           'a photo of a person', 'a photo of lab equipment and cables', 'a photo of a white plastic box']
OK = {1: {0}, 2: {2, 3}}    # class id -> accepted prompt ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--model', default='openai/clip-vit-large-patch14')
    a = ap.parse_args()
    dev = 'cuda'
    model = CLIPModel.from_pretrained(a.model).to(dev).eval(); proc = CLIPProcessor.from_pretrained(a.model)
    with torch.no_grad():
        t = proc(text=PROMPTS, return_tensors='pt', padding=True).to(dev)
        tf = model.get_text_features(**t); tf = tf / tf.norm(dim=-1, keepdim=True)
    meta = json.load(open(a.labels))
    for i, m in enumerate(meta):
        if not m['boxes']:
            continue
        im = Image.open(resolve(m['image'])).convert('RGB'); W, H = im.size
        crops = []
        for c, x, y, w, h, s in m['boxes'][:]:
            pad = 0.1   # a little context
            x0, y0 = max(0, (x - w / 2 - pad * w) * W), max(0, (y - h / 2 - pad * h) * H)
            x1, y1 = min(W, (x + w / 2 + pad * w) * W), min(H, (y + h / 2 + pad * h) * H)
            crops.append(im.crop((int(x0), int(y0), max(int(x1), int(x0) + 2), max(int(y1), int(y0) + 2))))
        with torch.no_grad():
            f = model.get_image_features(**proc(images=crops, return_tensors='pt').to(dev))
            f = f / f.norm(dim=-1, keepdim=True)
            pr = (100 * f @ tf.T).softmax(-1).cpu().numpy()
        m['boxes'] = [list(b[:6]) + [int(p.argmax()), float(p.max()), float(sum(p[j] for j in OK.get(b[0], ())))]
                      for b, p in zip(m['boxes'], pr)]
        if i % 200 == 0:
            print(i, len(meta), flush=True)
    json.dump({'prompts': PROMPTS, 'meta': meta}, open(a.labels.replace('.json', '_clip.json'), 'w'))


if __name__ == '__main__':
    main()
