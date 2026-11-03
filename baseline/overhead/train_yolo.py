"""Fine-tune the overhead (bird's-eye) detector: classes misty / cozmo / obstacle.

    python -m overhead.train_yolo --data ../data_download/overhead_yolo/ds/data.yaml --name overhead_v1

Init from the Misty-view checkpoint (same yolo26s architecture; the head is re-initialised for the new
class names). imgsz 960 keeps Cozmo (~55 px in the 1280x720 frame) at ~40 px.
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--name', default='overhead_v1')
ap.add_argument('--init', default='../yolo_extracted/release_v1/checkpoint/misty_yolo26s_v1.pt')
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--imgsz', type=int, default=960)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--device', default='0')
a = ap.parse_args()
YOLO(a.init).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, optimizer='AdamW',
                   lr0=0.001, cache=False, workers=0, seed=0, close_mosaic=10, degrees=10, fliplr=0.5, flipud=0.5,
                   scale=0.3, project='runs_det', name=a.name, exist_ok=True, plots=True, patience=20)
