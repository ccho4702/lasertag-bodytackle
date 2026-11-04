#!/bin/bash
# Fine-tuned SmolVLA (body-tackle prompts, trained on all sessions) → checkpoints/smolvla/{chaser,evader}/
# Files are > 100 MB, so they live in the GitHub release, not in git. Needs `gh auth login` (private repo).
# Base model (only for re-training) downloads automatically from Hugging Face: lerobot/smolvla_base
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints/smolvla
for r in chaser evader; do
  gh release download v0.1 -R ccho4702/lasertag-bodytackle -p "smolvla_${r}.tar" -D checkpoints/smolvla --clobber
  tar -xf "checkpoints/smolvla/smolvla_${r}.tar" -C checkpoints/smolvla && rm "checkpoints/smolvla/smolvla_${r}.tar"
done
ls checkpoints/smolvla/*
