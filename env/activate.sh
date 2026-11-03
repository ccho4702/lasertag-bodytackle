# source env/activate.sh   (after: python3.10 -m venv .venv && .venv/bin/pip install -r env/requirements.txt)
_LT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$_LT/.venv/bin/activate"
export PY=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# training-server knobs (optional):
#   LSMA_NUM_WORKERS=0  when /dev/shm is tiny      LSMA_EVAL_BS=64  smaller eval batch
#   LSMA_AMP=1          bf16 autocast (half the activation memory)
