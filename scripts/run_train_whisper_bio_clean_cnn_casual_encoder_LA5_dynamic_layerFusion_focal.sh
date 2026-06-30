#!/bin/bash
#SBATCH --job-name=whisper-BIO
#SBATCH --partition=reservation_b6000_2
##SBATCH --partition=a100-al9
##SBATCH --reservation=klp_reservation
##SBATCH --qos=klp_a100_reservation
#SBATCH --time=32-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_DIR="${TRAIN_DIR:-/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/train}"
VAL_DIR="${VAL_DIR:-/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/dev}"
LABEL_COL="${LABEL_COL:-wbd_labels}"
CAUSAL_ENCODER="${CAUSAL_ENCODER:-1}"
ENCODER_LAYERS="${ENCODER_LAYERS:-8,16,24,32}"
LAYER_FUSION="${LAYER_FUSION:-dynamic_weighted_sum}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/whisper_bio_clean_cnn_causal_encoder_LA5_DynamicLayerFusion_focal}"
RUN_NAME="${RUN_NAME:-whisper_bio_clean_cnn_causal_encoder_LA5_DynamicLayerFusion_focal}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-1000}"
EPOCHS="${EPOCHS:-1}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-8000}"
VAL_STEPS="${VAL_STEPS:-0}"
LR="${LR:-1e-3}"
HEAD_TYPE="${HEAD_TYPE:-causal_conv1d}"

CAUSAL_ENCODER_ARG=()
if [[ "${CAUSAL_ENCODER}" == "1" || "${CAUSAL_ENCODER}" == "true" || "${CAUSAL_ENCODER}" == "TRUE" ]]; then
  CAUSAL_ENCODER_ARG=(--causal-encoder)
fi

mkdir -p "${CHECKPOINT_DIR}"

echo "[$(date --iso-8601=seconds)] starting ${SLURM_JOB_NAME:-local} job_id=${SLURM_JOB_ID:-none}"
echo "host=$(hostname)"
echo "train_dir=${TRAIN_DIR}"
echo "val_dir=${VAL_DIR}"
echo "label_col=${LABEL_COL}"
echo "head_type=${HEAD_TYPE}"
echo "causal_encoder=${CAUSAL_ENCODER}"
echo "encoder_layers=${ENCODER_LAYERS}"
echo "layer_fusion=${LAYER_FUSION}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "val_steps=${VAL_STEPS} val_every_steps=${VAL_EVERY_STEPS}"
echo "batch_size=${BATCH_SIZE}"




WANDB_ARG=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  WANDB_ARG=(
    --use-wandb
    --wandb-mode "${WANDB_MODE}"
    --wandb-name "${RUN_NAME}"
  )
fi

NUM_GPUS="${NUM_GPUS:-$(python - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)}"
if (( NUM_GPUS < 1 )); then
  echo "No visible CUDA GPUs." >&2
  exit 1
fi
echo "num_gpus=${NUM_GPUS}"

torchrun --standalone --nnodes=1 --nproc_per_node "${NUM_GPUS}" \
  "${PROJECT_ROOT}/train_whisper_wbd.py" \
  --train-dir "${TRAIN_DIR}" \
  --val-dir "${VAL_DIR}" \
  --label-col "${LABEL_COL}" \
  --loss focal \
  --focal-gamma 2.0 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --epochs "${EPOCHS}" \
  --val-every-steps "${VAL_EVERY_STEPS}" \
  --lr "${LR}" \
  --weight-decay 1e-2 \
  --max-grad-norm 1.0 \
  --encoder-layers "${ENCODER_LAYERS}" \
  --layer-fusion "${LAYER_FUSION}" \
  --freeze-encoder \
  "${CAUSAL_ENCODER_ARG[@]}" \
  --encoder-lookahead-frames 5 \
  --head-type "${HEAD_TYPE}" \
  --dropout 0.1 \
  --boundary-tolerance 2 \
  --frame-selection none \
  --shuffle-buffer "${SHUFFLE_BUFFER}" \
  --pin-memory \
  --log-every 200 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  "${WANDB_ARG[@]}"

echo "[$(date --iso-8601=seconds)] training finished"
