# Whisper BIO Word Boundary Detection

This project trains a lightweight BIO frame classifier on top of frozen Whisper encoder hidden states for word boundary detection.

The input data is stored as WALTZ Arrow shards. Each row contains audio bytes and WhisperX word-level time alignment. The label preprocessing step converts the word time spans into BIO labels on the Whisper encoder time grid.

## Project Layout

```text
whisper-bio/
├── add_bio_label.py                         # Add BIO labels to Arrow shards
├── train_whisper_wbd.py                     # Training entry point
├── run_add_bio_label.sh                     # Example label-building SLURM script
├── requirements_h100.txt                    # Python environment for H100 machine
├── requirements_p6000.txt                   # Python environment for P6000 machine
├── experiments/
│   ├── run_train_whisper_bio_clean_linear.sh
│   └── run_train_whisper_bio_aug_linear.sh
└── wbd/
    ├── checkpoint.py
    ├── config.py
    ├── data.py
    ├── loss.py
    ├── metrics.py
    ├── model.py
    ├── trainer.py
    └── wandb_utils.py
```

## Environment

Activate the existing conda environment:

```bash
module load miniconda3
conda activate wbd
```

Install dependencies for the target machine:

```bash
pip install -r requirements_h100.txt
```

or:

```bash
pip install -r requirements_p6000.txt
```

## Data

The training code expects WALTZ Arrow shards with audio and WBD label columns.

Default paths used by the scripts:

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
python add_bio_label.py \
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
sbatch experiments/run_train_whisper_bio_clean_linear.sh
```

Augmented-label experiment:

```bash
sbatch experiments/run_train_whisper_bio_aug_linear.sh
```

Common environment overrides:

```bash
EPOCHS=3 \
BATCH_SIZE=32 \
GRAD_ACCUM_STEPS=2 \
LR=1e-3 \
HEAD_TYPE=linear \
CHECKPOINT_DIR=/path/to/checkpoints \
sbatch experiments/run_train_whisper_bio_clean_linear.sh
```

Resume from a checkpoint:

```bash
RESUME=/path/to/epoch_0_whole_resume.pt \
sbatch experiments/run_train_whisper_bio_clean_linear.sh
```

Supported classifier heads:

```text
linear
mlp
causal_conv1d
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

## Notes

Generated files are intentionally ignored by Git:

```text
__pycache__/
*.pyc
*.pt
*.pth
checkpoint/
wandb/
*.out
*.err
*.log
```

Large datasets, checkpoints, W&B logs, and SLURM logs should stay outside the GitHub repository.
