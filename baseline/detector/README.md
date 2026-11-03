# Cozmo-view detector (evader)

`misty_yolo26s_v1.pt` was trained on Misty-camera frames and does not detect Misty from the Cozmo
camera (320x240, low viewpoint, pink colour cast) — it boxes the floor as `obstacle`. This folder
builds a Cozmo-view dataset by auto-labelling + manual review and fine-tunes from the Misty model.
Only the robot's own camera images are used (overhead camera is never an input).

Classes are kept identical to the Misty model: `0 ally, 1 enemy (Misty), 2 obstacle (brick blocks)`.
Not obstacles: the arena wall, the cream/white box at the arena edge, lab equipment, people,
fire extinguishers (consistent with what the Misty model does on chaser frames).

## Pipeline

```bash
cd baseline
# 1. Grounding DINO candidates ('white robot', 'brick block'), low threshold
python -m detector.autolabel --sessions ../data_download/evader/*/*/ --out ../data_download/evader_yolo \
    --box_thr 0.25 --max_area 0.9
# 2. CLIP verdict per box crop
python -m detector.verify_clip --labels ../data_download/evader_yolo/autolabels.json
# 3. rule filter (tuned by visual QA) + manual review json -> YOLO dataset, split by session
python detector/build_yolo_ds.py --labels ../data_download/evader_yolo/autolabels_clip.json \
    --review ../data_download/evader_yolo/manual_review.json --out ../data_download/evader_yolo/ds \
    --val_sessions 20261103T035839_693734Z 20261103T040015_233619Z 20261103T042045_838328Z
# 4. fine-tune from the Misty checkpoint (hyper-params from its train_args, imgsz 320)
python detector/train_yolo.py --data ../data_download/evader_yolo/ds/data.yaml --name cozmo_v1
```

Review helpers: `review_sheet.py` (numbered crops of every kept box, per class) and
`review_missed.py` (rejected candidates in frames that ended up unlabeled). Decisions are stored
in `manual_review.json` (`reject` / `add` lists of `[image, [c,x,y,w,h]]`).

## Label QA log (v1)

- Every kept box was reviewed by eye: 357 enemy → 41 rejected (people, extinguishers, printer,
  lab equipment, a blue brick); 782 obstacle → 109 rejected (white box, arena wall segments).
- Unlabeled frames with candidates: 324 candidates reviewed, 18 boxes added (close-up Misty base,
  close blue bricks).
- Known gaps: bricks that fill the whole frame (>90% area) were never proposed; very distant
  Misty (< ~15 px) mostly unlabeled; frames that already had one label were not re-checked for a
  second missed object.
