"""Cache upright RGB frames of a session as a uint8 memmap for fast VLA training.

    python -m vla.cache_images --session ../data/sample/<session>

Writes <session>/images_vla.npy  (N, H, W, 3) uint8, rows aligned with labels.csv.
Frames are stored as recorded (upright, 480×640 portrait, no crop / flip) — the YOLO
pipeline's 180° flip is *not* applied because the VLM backbone expects natural images.
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def cache_session(session, force=False):
    out = os.path.join(session, 'images_vla.npy')
    if os.path.exists(out) and not force:
        print(f'exists: {out}')
        return out
    df = pd.read_csv(os.path.join(session, 'labels.csv'))
    first = np.array(Image.open(os.path.join(session, df.image[0])).convert('RGB'))
    mm = np.lib.format.open_memmap(out, mode='w+', dtype=np.uint8, shape=(len(df), *first.shape))
    for i, rel in enumerate(df.image):
        try:
            mm[i] = np.array(Image.open(os.path.join(session, rel)).convert('RGB'))
        except Exception as e:   # keep alignment with labels.csv
            print(f'  [WARN] {rel}: {e}')
            mm[i] = 0
    mm.flush()
    print(f'cached {len(df)} frames {first.shape} → {out}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', nargs='+', required=True)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    for s in a.session:
        cache_session(s, a.force)


if __name__ == '__main__':
    main()
