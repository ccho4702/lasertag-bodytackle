"""Grounding DINO candidates on the overhead (bird's-eye) frames of every session.

    cd baseline
    python -m overhead.autolabel --sessions ../data_download/chaser/*/*/*/ ../data_download/evader/*/*/ \
        --out ../data_download/overhead_yolo/candidates.json

Raw candidates only (label, score, xyxy px); rules + per-session consistency are applied in
overhead/build_ds.py so they can be re-tuned without re-running DINO.
Prompts: 'white robot' (Misty), 'small toy robot' (Cozmo), 'brick block' (obstacles).
"""
import argparse, glob, json, os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEXT = 'white robot. small toy robot. brick block.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='IDEA-Research/grounding-dino-base')
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--stride', type=int, default=1, help='use every k-th overhead frame')
    ap.add_argument('--bs', type=int, default=8)
    a = ap.parse_args()
    dev = 'cuda'
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).to(dev).eval()
    imgs = []
    for s in a.sessions:
        fr = sorted(glob.glob(os.path.join(s, 'overhead_images', '*.jpg')))[::a.stride]
        imgs += fr
    print(f'{len(imgs)} frames from {len(a.sessions)} sessions', flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = {}
    if os.path.exists(a.out):
        done = {d['image']: d for d in json.load(open(a.out))}
    todo = [p for p in imgs if os.path.relpath(os.path.abspath(p), ROOT) not in done]
    for i in range(0, len(todo), a.bs):
        batch = todo[i:i + a.bs]
        ims = [Image.open(p).convert('RGB') for p in batch]
        inp = proc(images=ims, text=[TEXT] * len(ims), return_tensors='pt').to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=a.thr, text_threshold=a.thr, target_sizes=[im.size[::-1] for im in ims])
        for p, im, r in zip(batch, ims, res):
            rel = os.path.relpath(os.path.abspath(p), ROOT)
            labs = r['text_labels'] if 'text_labels' in r else r['labels']
            done[rel] = {'image': rel, 'size': im.size,
                         'cands': [[lab, round(float(sc), 3), [round(v, 1) for v in b]]
                                   for b, sc, lab in zip(r['boxes'].tolist(), r['scores'].tolist(), labs)]}
        if (i // a.bs) % 20 == 0 or i + a.bs >= len(todo):
            print(f'{i + len(batch)}/{len(todo)}', flush=True)
            json.dump(list(done.values()), open(a.out, 'w'))
    json.dump(list(done.values()), open(a.out, 'w'))


if __name__ == '__main__':
    main()
