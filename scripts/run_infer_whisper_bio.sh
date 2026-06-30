#!/bin/bash
#SBATCH --job-name=infer_bio_test
#SBATCH --partition=reservation_b6000_2
##SBATCH --partition=a100-al9
##SBATCH --reservation=klp_reservation
##SBATCH --qos=klp_a100_reservation
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -o /ceph/work/KLP/peter1/whisper-bio/slurm-logs/%j.out
#SBATCH -e /ceph/work/KLP/peter1/whisper-bio/slurm-logs/%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ceph/work/KLP/peter1/whisper-bio}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"


export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

TEST_DIR="${TEST_DIR:-/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/test}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/whisper_bio_clean_cnn_causal_encoder_LA5_DynamicLayerFusion_focal/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/infer_result/CNN_CMask_LA5_DynamicLayerFusion_focal}"
PREDICTION_DIR="${PREDICTION_DIR:-${OUTPUT_DIR}/predictions}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
DEVICE="${DEVICE:-cuda}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
LABEL_COL="${LABEL_COL:-wbd_labels}"
FRAME_SELECTION="${FRAME_SELECTION:-none}"
BOUNDARY_TOLERANCE="${BOUNDARY_TOLERANCE:-2}" 
CAUSAL_ENCODER="${CAUSAL_ENCODER:-auto}"

if [[ ! -d "${TEST_DIR}" ]]; then
  echo "Test dir not found: ${TEST_DIR}" >&2
  echo "Set TEST_DIR=/path/to/test_dir" >&2
  exit 1
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT:-<empty>}" >&2
  echo "Set CHECKPOINT=/path/to/checkpoint.pt" >&2
  exit 1
fi

ATTN_ARG=()
if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  ATTN_ARG=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

CAUSAL_ENCODER_ARG=()
case "${CAUSAL_ENCODER}" in
  auto|"")
    CAUSAL_ENCODER_ARG=()
    ;;
  1|true|TRUE|yes|YES|on|ON)
    CAUSAL_ENCODER_ARG=(--causal-encoder)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    CAUSAL_ENCODER_ARG=(--no-causal-encoder)
    ;;
  *)
    echo "Invalid CAUSAL_ENCODER=${CAUSAL_ENCODER}; use auto, 1, or 0." >&2
    exit 1
    ;;
esac

AMP_ARG=(--amp)
if [[ "${USE_AMP:-1}" == "0" ]]; then
  AMP_ARG=(--no-amp)
fi

PREDICTION_ARG=()
if [[ -n "${PREDICTION_DIR}" ]]; then
  PREDICTION_ARG=(--prediction-dir "${PREDICTION_DIR}")
fi

echo "[$(date --iso-8601=seconds)] full test inference"
echo "test_dir=${TEST_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "prediction_dir=${PREDICTION_DIR:-<disabled>}"
echo "num_gpus=2"
echo "batch_size=${BATCH_SIZE}"
echo "num_workers=${NUM_WORKERS} prefetch_factor=${PREFETCH_FACTOR}"
echo "causal_encoder=${CAUSAL_ENCODER}"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m wbd.cli.infer \
  --test-dir "${TEST_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --device "${DEVICE}" \
  "${AMP_ARG[@]}" \
  --amp-dtype "${AMP_DTYPE}" \
  --label-col "${LABEL_COL}" \
  --frame-selection "${FRAME_SELECTION}" \
  --boundary-tolerance "${BOUNDARY_TOLERANCE}" \
  "${CAUSAL_ENCODER_ARG[@]}" \
  "${ATTN_ARG[@]}" \
  "${PREDICTION_ARG[@]}"

echo "[$(date --iso-8601=seconds)] done"
