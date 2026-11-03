#!/bin/bash
# Paper-aligned experiments on the new data, per robot (chaser = Misty shooter, evader = Cozmo).
#   1. preprocess (YOLO heatmaps + DATv2 depth)     2. OOF SmolVLA plans (5-fold cross-fitting)
#   3. 5-fold session CV x 3 seeds: paper student, paper student + VLA (period 5 / 10)
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
ROBOTS=${ROBOTS:-"chaser evader"}
SEEDS=${SEEDS:-"0 1 2"}
for R in $ROBOTS; do
  if [ $R = chaser ]; then mapfile -t S < <(ls -d ../data_download/chaser/*/*/*/ | sort); else mapfile -t S < <(ls -d ../data_download/evader/*/*/ | sort); fi
  # VLA latency ≈ 342 ms (measured, RTX 3090) in frames of this robot's camera rate: chaser 9.8 fps → 4, evader 8.4 fps → 3
  if [ $R = chaser ]; then LAT=4; else LAT=3; fi
  echo "== $R: ${#S[@]} sessions"
  $PY preprocess.py --robot $R --session "${S[@]}"
  $PY -m vla.cache_images --session "${S[@]}"
  $PY -m vla.crossfit --sessions "${S[@]}" --tag oof --folds 5 --out runs_vla/crossfit_$R --robot $R
  $PY cv.py --sessions "${S[@]}" --folds 5 --seeds $SEEDS --name paper_$R -- --robot $R --preset paper
  $PY cv.py --sessions "${S[@]}" --folds 5 --seeds $SEEDS --name paper_vla_$R -- --robot $R --preset paper --vla_tag oof --vla_latency $LAT
  $PY cv.py --sessions "${S[@]}" --folds 5 --seeds $SEEDS --name paper_vla_p5_$R -- --robot $R --preset paper --vla_tag oof --vla_period 5 --vla_latency $LAT
done
echo ALLDONE
