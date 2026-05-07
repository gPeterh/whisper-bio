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

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wbd


python /work/peter065/Word-Segmenter/whisper-bio/add_bio_label.py \
  --input-dir /work/peter065/WALTZ_storage/waltz_datasets/test \
  --split test \
  --in-place \
  --overwrite
  
  
#--resume \