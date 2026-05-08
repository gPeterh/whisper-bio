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

CONDA_ENV_NAME="wbd"
conda env list
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
conda env list
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi


export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_DIR="${TRAIN_DIR:-/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/train}"
VAL_DIR="${VAL_DIR:-/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/dev}"
LABEL_COL="${LABEL_COL:-wbd_labels}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/ceph/work/KLP/peter1/whisper-bio/checkpoints/whisper_bio_clean_causal_conv1d}"
RUN_NAME="${RUN_NAME:-whisper-bio-clean-causal_conv1d}"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-3}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-8000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-0}"
VAL_STEPS="${VAL_STEPS:-0}"
LR="${LR:-1e-3}"
HEAD_TYPE="${HEAD_TYPE:-causal_conv1d}"
RESUME_ARG=()
if [[ -n "${RESUME:-}" && "${RESUME}" != "0" ]]; then
  RESUME_ARG=(--resume "${RESUME}")
fi

mkdir -p "${CHECKPOINT_DIR}"

echo "[$(date --iso-8601=seconds)] starting ${SLURM_JOB_NAME:-local} job_id=${SLURM_JOB_ID:-none}"
echo "host=$(hostname)"
echo "train_dir=${TRAIN_DIR}"
echo "val_dir=${VAL_DIR}"
echo "label_col=${LABEL_COL}"
echo "head_type=${HEAD_TYPE}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "steps_per_epoch=${STEPS_PER_EPOCH} val_steps=${VAL_STEPS} val_every_steps=${VAL_EVERY_STEPS}"


python - <<'PY'
import importlib.util
import os
import sys

required = ["torch", "torchaudio", "pyarrow", "transformers"]
if os.environ.get("USE_WANDB", "1") == "1":
    required.append("wandb")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("Missing Python packages: " + ", ".join(missing) + ". Run: pip install -r requirements_h100.txt or requirements_p6000.txt")

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

python -u /ceph/work/KLP/peter1/whisper-bio/train_whisper_wbd.py \
  --train-dir "${TRAIN_DIR}" \
  --val-dir "${VAL_DIR}" \
  --label-col "${LABEL_COL}" \
  --loss ce \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
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
  --frame-selection none \
  --shuffle-buffer 2048 \
  --log-every 200 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  "${RESUME_ARG[@]}" \
  "${WANDB_ARG[@]}"

echo "[$(date --iso-8601=seconds)] training finished"
