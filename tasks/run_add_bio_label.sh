#!/bin/bash
#SBATCH --account ACD114137
#SBATCH --job-name=BIO
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -o /work/peter065/Word-Segmenter/bio_slurm-logs/%j.out
#SBATCH -e /work/peter065/Word-Segmenter/bio_slurm-logs/%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ceph/work/KLP/peter1/whisper-bio}"
source "${PROJECT_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH:-}"


python "${PROJECT_ROOT}/tasks/add_bio_label.py" \
  --input-dir /work/peter065/WALTZ_storage/waltz_datasets/test \
  --split test \
  --in-place \
  --overwrite
  
  
#--resume \
