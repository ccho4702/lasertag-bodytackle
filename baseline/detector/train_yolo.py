"""Train the Cozmo-view detector. Hyper-params copied from misty_yolo26s_v1.pt train_args,
except: init from the Misty checkpoint (same 3 classes) and imgsz 320 (Cozmo frames are 320x240).

    python detector/train_yolo.py --data ../data_download/evader_yolo/ds/data.yaml --name cozmo_v1
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--name', default='cozmo_v1')
ap.add_argument('--init', default='../yolo_extracted/release_v1/checkpoint/misty_yolo26s_v1.pt')
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--imgsz', type=int, default=320)
a = ap.parse_args()
YOLO(a.init).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=16, optimizer='AdamW', lr0=0.001429,
                   cache='ram', workers=4, seed=0, deterministic=True, close_mosaic=15, degrees=5,
                   fliplr=0.5, project='runs_det', name=a.name, exist_ok=True, plots=True)
