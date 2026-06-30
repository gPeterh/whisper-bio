# Whisper BIO Word Boundary Detection

This project trains a lightweight BIO frame classifier on top of frozen Whisper encoder hidden states for word boundary detection.

The input data is stored as WALTZ Arrow shards. Each row contains audio bytes and WhisperX word-level time alignment. The label preprocessing step converts the word time spans into BIO labels on the Whisper encoder time grid.

## TODO
1. [x] Add causal mask on whisper encoder
2. [x] Add inference code

## Project Layout

```text
whisper-bio/
├── docs/
├── run_half_data/                           # Half-data manifest lists
├── scripts/
│   ├── run_eval_whisper_bio_predictions.sh
│   ├── run_infer_whisper_bio.sh
│   ├── run_train_whisper_bio_aug_linear.sh
│   ├── run_train_whisper_bio_clean_linear.sh
│   └── run_train_whisper_bio_clean_cnn_casual_encoder_LA5_dynamic_layerFusion_focal.sh
├── src/wbd/
│   ├── cli/
│   │   ├── infer.py                         # Full test-set inference CLI
│   │   └── eval_predictions.py              # Offline prediction-cache evaluation CLI
│   ├── checkpoint.py
│   ├── data.py
│   ├── inference.py
│   ├── metrics.py
│   ├── model.py
│   └── trainer.py
├── tasks/
│   ├── add_bio_label.py                     # Add BIO labels to Arrow shards
│   └── run_add_bio_label.sh                 # Example label-building SLURM script
├── tests/
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
├── requirements_h100.txt
├── requirements_b6000.txt
├── train_whisper_wbd.py                     # Training entry point
└── uv.lock
```

## Environment

This project can run with either `uv` or the conda environment. The recommended project-managed setup is `uv`, which creates a `.venv/` from `pyproject.toml` and `uv.lock`.

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync --no-default-groups
source .venv/bin/activate
```

For development tools such as `pytest`, `ruff`, and `ipykernel`, include the dev group:

```bash
uv sync --group dev
```

Run project commands through uv:

```bash
uv run python train_whisper_wbd.py --help
```

Alternatively, activate the existing conda environment and install the legacy requirements file for the target machine:

```bash
module load anaconda3/2024.10-1
conda activate wbd
pip install -r requirements_b6000.txt
```

The B6000 uv setup currently pins Python 3.10 and routes `torch` to the PyTorch CUDA 12.8 wheel index in `pyproject.toml`. 

## Data

The training code expects WALTZ Arrow shards with audio and WBD label columns.

Default paths used by the scripts (nano5):

For dicos.grid
```text
/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/train
/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/dev
/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/test
```

For Nano5
```text
/work/peter065/WALTZ_storage/waltz_datasets/train
/work/peter065/WALTZ_storage/waltz_datasets/dev
/work/peter065/WALTZ_storage/waltz_datasets/test
```

Required label columns:

```text
wbd_labels      clean BIO labels
wbd_aug_label   augmented BIO labels for training experiments
wbd_label_len   valid frame length before padding
wbd_word_spans  word spans on the Whisper 20 ms frame grid
```

Label ids:

```text
0     O
1     B
2     I
-100  padding / ignored region
```

## Build BIO Labels

BIO labels can be added to Arrow files with:

```bash
python tasks/add_bio_label.py \
  --input-dir /work/peter065/WALTZ_storage/waltz_datasets/train \
  --split train \
  --in-place \
  --overwrite
```

Use `--split dev` or `--split test` for validation/test data. Train split can also create the augmented label column.

The labels are generated on the Whisper encoder frame grid, where one hidden-state frame corresponds to 20 ms. Audio beyond Whisper's 30 second window is ignored and padded with `-100`.

## Training

Clean-label baseline:

```bash
sbatch scripts/run_train_whisper_bio_clean_linear.sh
```

Augmented-label experiment:

```bash
sbatch scripts/run_train_whisper_bio_aug_linear.sh
```

Common environment overrides:

```bash
EPOCHS=3 \
BATCH_SIZE=32 \
GRAD_ACCUM_STEPS=2 \
LR=1e-3 \
HEAD_TYPE=causal_conv1d \
```

The training entry point supports resume through `--resume`
```bash
python train_whisper_wbd.py \
  --train-dir /path/to/train \
  --val-dir /path/to/dev \
  --checkpoint-dir /path/to/checkpoints \
  --resume /path/to/epoch_0_whole_resume.pt
```

Supported classifier heads:

```text
linear
mlp
causal_conv1d
```

Encoder layer selection:

```text
--encoder-layer N       single fixed encoder hidden-state layer
--encoder-layers A,B     multi-layer mode; use `all` for all encoder layers; requires --layer-fusion
```

Supported multi-layer fusion modes:

```text
learnable_weighted_sum  one global learnable weight per selected layer
dynamic_weighted_sum    per-frame layer weights predicted from the deepest selected layer
```

The current experiments freeze the Whisper encoder and train only the BIO classification head.

## Validation and Checkpoints

Validation runs every `VAL_EVERY_STEPS` optimizer steps and once again at the end of each epoch.

Checkpoint naming:

```text
epoch_0_step_8000.pt       step checkpoint after validation
epoch_0_whole.pt           epoch-end checkpoint
epoch_0_whole_resume.pt    epoch-end checkpoint intended for resume
best.pt                    best validation segmentation_r_value
```

Validation metrics include:

```text
token_acc
segmentation_precision
segmentation_recall
segmentation_f_value
segmentation_os
segmentation_r_value
boundary_hits
boundary_pred
boundary_ref
```

The best model is selected by `segmentation_r_value`.

## Inference

Full test-set inference writes only final metrics and run metadata:

```bash
CHECKPOINT=/path/to/best.pt \
OUTPUT_DIR=/path/to/infer_result/test \
sbatch scripts/run_infer_whisper_bio.sh
```

The output directory contains:

```text
metrics.json
metrics.csv
run_config.json
```

To also cache raw predicted BIO labels for later offline post-processing:

```bash
CHECKPOINT=/path/to/best.pt \
OUTPUT_DIR=/path/to/infer_result/test \
PREDICTION_DIR=/path/to/pred_cache/test \
sbatch scripts/run_infer_whisper_bio.sh
```

The prediction cache is written as one Parquet file per source Arrow shard plus a `manifest.json`.
Each row stores the source file, source row index, key, label length, and raw predicted labels
truncated to the valid label length.

Cached predictions can be evaluated later without rerunning Whisper:

```bash
wbd-eval-predictions \
  --prediction-dir /path/to/pred_cache/test \
  --output-dir /path/to/offline_eval/test \
  --frame-selection mid \
  --post-process
```
