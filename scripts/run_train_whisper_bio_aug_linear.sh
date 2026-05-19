#!/bin/bash
#SBATCH --job-name=whisper-BIO
#SBATCH --partition=reservation_b6000_2
##SBATCH --partition=a100-al9
##SBATCH --reservation=klp_reservation
##SBATCH --qos=klp_a100_reservation
#SBATCH --time=32-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /ceph/work/KLP/peter1/whisper-bio/whisper_bio_slurm/%j.out
#SBATCH -e /ceph/work/KLP/peter1/whisper-bio/whisper_bio_slurm/%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ceph/work/KLP/peter1/whisper-bio}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi


export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_DIR="${TRAIN_DIR:-/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/train}"
VAL_DIR="${VAL_DIR:-/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/dev}"
LABEL_COL="${LABEL_COL:-wbd_aug_label}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/ceph/work/KLP/peter1/whisper-bio/checkpoints/whisper_aug_linear}"
RUN_NAME="${RUN_NAME:-whisper_bio_aug_linear}"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-6}"
EPOCHS="${EPOCHS:-1}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-8000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-0}"
VAL_STEPS="${VAL_STEPS:-0}"
LR="${LR:-1e-3}"
HEAD_TYPE="${HEAD_TYPE:-linear}"
mkdir -p "${CHECKPOINT_DIR}"

echo "[$(date --iso-8601=seconds)] starting ${SLURM_JOB_NAME:-local} job_id=${SLURM_JOB_ID:-none}"
echo "host=$(hostname)"
echo "train_dir=${TRAIN_DIR}"
echo "val_dir=${VAL_DIR}"
echo "label_col=${LABEL_COL}"
echo "head_type=${HEAD_TYPE}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "steps_per_epoch=${STEPS_PER_EPOCH} val_steps=${VAL_STEPS} val_every_steps=${VAL_EVERY_STEPS}"
echo "batch_size=${BATCH_SIZE}"
echo "num_workers=${NUM_WORKERS} prefetch_factor=${PREFETCH_FACTOR} shuffle_buffer=1000"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}"


python - <<'PY'
import importlib.util
import os
import sys

required = ["torch", "pyarrow", "scipy", "soundfile", "transformers"]
if os.environ.get("USE_WANDB", "1") == "1":
    required.append("wandb")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("Missing Python packages: " + ", ".join(missing) + ". Run: uv sync --no-default-groups")

import torch

version_parts = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
if version_parts < (2, 1):
    sys.exit(
        f"PyTorch {torch.__version__} is too old for the installed Transformers. "
        "Run: pip install --upgrade -r requirements.txt"
    )
PY



WANDB_ARG=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  WANDB_ARG=(
    --use-wandb
    --wandb-mode "${WANDB_MODE}"
    --wandb-name "${RUN_NAME}"
  )
fi

python -u "${PROJECT_ROOT}/train_whisper_wbd.py" \
  --train-dir "${TRAIN_DIR}" \
  --val-dir "${VAL_DIR}" \
  --label-col "${LABEL_COL}" \
  --loss ce \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --epochs "${EPOCHS}" \
  --val-every-steps "${VAL_EVERY_STEPS}" \
  --lr "${LR}" \
  --weight-decay 1e-2 \
  --max-grad-norm 1.0 \
  --encoder-layer 6 \
  --freeze-encoder \
  --head-type "${HEAD_TYPE}" \
  --dropout 0.1 \
  --boundary-tolerance 2 \
  --frame-selection mid \
  --shuffle-buffer 1000 \
  --log-every 200 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  "${WANDB_ARG[@]}"

echo "[$(date --iso-8601=seconds)] training finished"
