#!/bin/bash
# Fine-tuned SmolVLA (body-tackle prompts, trained on all sessions) → checkpoints/smolvla/{chaser,evader}/
# Files are > 100 MB, so they live in the GitHub release, not in git. Needs `gh auth login` (private repo).
#   bash scripts/download_smolvla.sh              # latest = v2026.11.24 (pairs with checkpoints/policy/v2026.11.24/ours_*)
#   bash scripts/download_smolvla.sh v0.2         # older release (pairs with checkpoints/policy/ours_{cpu,gpu}_*.pt)
# The installed version is written to checkpoints/smolvla/VERSION.
# Base model (only for re-training) downloads automatically from Hugging Face: lerobot/smolvla_base
set -e
VER=${1:-v2026.11.24}
cd "$(dirname "$0")/.."
mkdir -p checkpoints/smolvla
for r in chaser evader; do
  gh release download "$VER" -R ccho4702/lasertag-bodytackle -p "smolvla_${r}.tar" -D checkpoints/smolvla --clobber
  rm -rf "checkpoints/smolvla/${r}"
  tar -xf "checkpoints/smolvla/smolvla_${r}.tar" -C checkpoints/smolvla && rm "checkpoints/smolvla/smolvla_${r}.tar"
done
echo "$VER" > checkpoints/smolvla/VERSION
ls checkpoints/smolvla/*
