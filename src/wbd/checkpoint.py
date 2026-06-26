"""Checkpoint and local metrics CSV helpers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from .metrics import BatchMetrics


def unwrap_model(model):
    return getattr(model, "module", model)


def save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_segmentation_r_value: float,
    train_files: list[str],
    val_files: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    checkpoint = {
        "classifier": model_to_save.classifier.state_dict(),
        "model_trainable": model_to_save.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
        "best_segmentation_r_value": best_segmentation_r_value,
        "train_files": train_files,
        "val_files": val_files,
        "model_name_or_path": args.model_name_or_path,
        "encoder_layer": args.encoder_layer,
        "encoder_layers": args.encoder_layers,
        "encoder_lookahead_frames": args.encoder_lookahead_frames,
        "layer_fusion": model_to_save.layer_fusion,
        "freeze_encoder": args.freeze_encoder,
    }
    torch.save(checkpoint, path)


def append_metrics(path: Path, epoch: int, split: str, metrics: BatchMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"epoch": epoch, "split": split, **asdict(metrics)}
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
