#!/usr/bin/env python3
"""Train a Whisper encoder word-boundary detector on WALTZ Arrow shards.

The Arrow files should contain labels produced by add_bio_label.py:
  - wbd_labels or wbd_aug_label: BIO labels on Whisper encoder tokens.
  - wbd_label_len: valid token length before -100 padding.

Labels use 0=O, 1=B, 2=I, and -100=ignore.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wbd.checkpoint import append_metrics, save_checkpoint
from wbd.config import MAX_WHISPER_TOKENS
from wbd.data import WhisperCollator, WaltzArrowWBDDataset, resolve_files
from wbd.loss import make_criterion
from wbd.model import WhisperWBDClassifier, ensure_transformers_torch_backend
from wbd.trainer import autocast_context, run_epoch
from wbd.wandb_utils import init_wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen Whisper-large-v3 encoder plus BIO classification head."
    )
    parser.add_argument(
        "--train-dir",
        default="/work/peter065/WALTZ_storage/waltz_datasets/train",
        help="Directory containing WALTZ .arrow shards.",
    )
    parser.add_argument("--glob", default="*.arrow", help="Shard glob used with --train-dir.")
    parser.add_argument(
        "--val-dir",
        default="/work/peter065/WALTZ_storage/waltz_datasets/dev",
        help="Directory containing validation .arrow shards. Set empty to disable validation.",
    )
    parser.add_argument("--val-glob", default="*.arrow", help="Shard glob used with --val-dir.")
    parser.add_argument("--train-files", nargs="*", default=None, help="Explicit train Arrow files.")
    parser.add_argument("--train-manifest", default=None, help="Text file with one train Arrow file per line.")
    parser.add_argument("--val-files", nargs="*", default=None, help="Explicit validation Arrow files.")
    parser.add_argument("--val-manifest", default=None, help="Text file with one validation Arrow file per line.")
    parser.add_argument("--limit-shards", type=int, default=None, help="Use only the first N train shards.")
    parser.add_argument("--max-rows-per-shard", type=int, default=None, help="Debug limit per shard.")
    parser.add_argument("--shuffle-buffer", type=int, default=512, help="Streaming shuffle buffer size. 0 disables row shuffle.")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--model-name-or-path", default="openai/whisper-large-v3")
    parser.add_argument(
        "--encoder-layer",
        type=int,
        default=6,
        help="Hidden-state index to use. 6 means after the first 6 encoder blocks; 0 is conv+pos embedding.",
    )
    parser.add_argument("--attn-implementation", default="sdpa", choices=[None, "eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--head-type",
        choices=["linear", "mlp", "causal_conv1d"],
        default="linear",
        help="BIO classification head to put on top of Whisper encoder features.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--label-col", default="wbd_labels", help="Use wbd_labels for clean BIO or wbd_aug_label for augmentation.")
    parser.add_argument("--label-len-col", default="wbd_label_len")
    parser.add_argument("--max-tokens", type=int, default=MAX_WHISPER_TOKENS)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="Training batches per epoch. 0 means run through all resolved training shards.")
    parser.add_argument("--val-steps", type=int, default=0, help="Validation batches per epoch. 0 means run through all validation shards; negative disables validation.")
    parser.add_argument(
        "--val-every-steps",
        type=int,
        default=8000,
        help="Run validation and save a step checkpoint every N optimizer steps during training. 0 disables mid-epoch validation.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--class-weights",
        default=None,
        help="Comma-separated CE/focal class weights for O,B,I, for example 1,5,1.",
    )
    parser.add_argument("--boundary-tolerance", type=int, default=2, help="Boundary match tolerance in 20ms tokens.")
    parser.add_argument(
        "--frame-selection",
        choices=["none", "mid", "first", "last"],
        default="mid",
        help="Post-process consecutive predicted B-token clusters before paper-style segmentation metrics.",
    )

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="./checkpoint/whisper_wbd")
    parser.add_argument("--resume", default=None, help="Explicit checkpoint path to resume model/optimizer.")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-full-model", action="store_true", help="Also store frozen Whisper encoder weights.")
    parser.add_argument("--use-wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    parser.add_argument("--wandb-entity", default="ckip-Waltz")
    parser.add_argument("--wandb-project", default="whisper-WBD")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    parser.add_argument("--smoke-test", action="store_true", help="Run one batch through the model, then exit.")
    return parser.parse_args()


def make_loader(
    *,
    files: list[str],
    label_col: str,
    args: argparse.Namespace,
    collator: WhisperCollator,
    device: torch.device,
    shuffle_files: bool,
    shuffle_buffer: int,
    seed: int,
) -> tuple[WaltzArrowWBDDataset, DataLoader]:
    dataset = WaltzArrowWBDDataset(
        files=files,
        label_col=label_col,
        label_len_col=args.label_len_col,
        max_tokens=args.max_tokens,
        max_rows_per_shard=args.max_rows_per_shard,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        shuffle_files=shuffle_files,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    return dataset, loader


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ensure_transformers_torch_backend()

    train_files, val_files = resolve_files(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    with open(checkpoint_dir / "train_files.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_files) + "\n")
    if val_files:
        with open(checkpoint_dir / "val_files.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(val_files) + "\n")

    print(f"train shards: {len(train_files)}")
    print(f"val shards: {len(val_files)}")
    print(f"label column: {args.label_col}")
    print(f"model: {args.model_name_or_path}, encoder hidden layer: {args.encoder_layer}")
    wandb_run = init_wandb(args, train_files, val_files)

    device = torch.device(args.device)
    collator = WhisperCollator(args.model_name_or_path, args.max_tokens)
    train_dataset, train_loader = make_loader(
        files=train_files,
        label_col=args.label_col,
        args=args,
        collator=collator,
        device=device,
        shuffle_files=True,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )

    val_dataset = None
    val_loader = None
    if val_files:
        val_dataset, val_loader = make_loader(
            files=val_files,
            label_col="wbd_labels",
            args=args,
            collator=collator,
            device=device,
            shuffle_files=False,
            shuffle_buffer=0,
            seed=args.seed + 999,
        )

    model = WhisperWBDClassifier(
        model_name_or_path=args.model_name_or_path,
        encoder_layer=args.encoder_layer,
        freeze_encoder=args.freeze_encoder,
        head_type=args.head_type,
        dropout=args.dropout,
        attn_implementation=args.attn_implementation,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters: trainable={trainable:,} total={total:,}")

    criterion = make_criterion(args, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    global_step = 0
    best_score = -math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif "classifier" in ckpt:
            model.classifier.load_state_dict(ckpt["classifier"])
        else:
            raise ValueError(f"Checkpoint {args.resume} does not contain model or classifier weights.")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_score = float(ckpt.get("best_segmentation_r_value"))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    if args.smoke_test:
        batch = next(iter(train_loader))
        model.eval()
        with torch.no_grad(), autocast_context(args, device):
            logits = model(batch["input_features"].to(device))
        print(f"smoke input_features: {tuple(batch['input_features'].shape)}")
        print(f"smoke labels: {tuple(batch['labels'].shape)}")
        print(f"smoke logits: {tuple(logits.shape)}")
        print(f"smoke label_lens: {batch['label_lens'].tolist()}")
        return

    metrics_path = checkpoint_dir / "metrics.csv"
    best_state = {"score": best_score}

    def run_validation_and_checkpoint(
        *,
        epoch: int,
        global_step: int,
        metrics_split: str,
        checkpoint_name: str,
        checkpoint_epoch: int,
        resume_checkpoint_name: str | None = None,
    ) -> None:
        if val_loader is None or val_dataset is None or args.val_steps < 0:
            return
        val_dataset.set_epoch(epoch)
        val_metrics, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            args=args,
            steps=args.val_steps,
            desc=f"{metrics_split} val",
            epoch=epoch,
            global_step=global_step,
            wandb_run=None,
        )
        append_metrics(metrics_path, epoch, metrics_split, val_metrics)
        print(f"{metrics_split}: {asdict(val_metrics)}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "global_step": global_step,
                    "val/epoch": epoch,
                    "val/loss": val_metrics.loss,
                    "val/len": val_metrics.len,
                    "val/token_acc": val_metrics.token_acc,
                    "val/segmentation_precision": val_metrics.segmentation_precision,
                    "val/segmentation_recall": val_metrics.segmentation_recall,
                    "val/segmentation_f_value": val_metrics.segmentation_f_value,
                    "val/segmentation_os": val_metrics.segmentation_os,
                    "val/segmentation_r_value": val_metrics.segmentation_r_value,
                }
            )

        save_checkpoint(
            checkpoint_dir / checkpoint_name,
            model,
            optimizer,
            args,
            checkpoint_epoch,
            global_step,
            best_state["score"],
            train_files,
            val_files,
        )
        if resume_checkpoint_name is not None:
            save_checkpoint(
                checkpoint_dir / resume_checkpoint_name,
                model,
                optimizer,
                args,
                checkpoint_epoch,
                global_step,
                best_state["score"],
                train_files,
                val_files,
            )

        score = val_metrics.segmentation_r_value
        if score > best_state["score"]:
            best_state["score"] = score
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                args,
                checkpoint_epoch,
                global_step,
                best_state["score"],
                train_files,
                val_files,
            )
            print(f"saved best.pt with segmentation_r_value={best_state['score']:.4f}")

    def maybe_validate_on_step(epoch: int, current_global_step: int) -> None:
        if args.val_every_steps <= 0:
            return
        if current_global_step <= 0 or current_global_step % args.val_every_steps != 0:
            return
        run_validation_and_checkpoint(
            epoch=epoch,
            global_step=current_global_step,
            metrics_split=f"val_step_{current_global_step}",
            checkpoint_name=f"epoch_{epoch - 1}_step_{current_global_step}.pt",
            checkpoint_epoch=epoch - 1,
        )
        model.train(True)

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_dataset.set_epoch(epoch)
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            steps=args.steps_per_epoch,
            desc=f"epoch {epoch} train",
            epoch=epoch,
            global_step=global_step,
            wandb_run=wandb_run,
            step_callback=maybe_validate_on_step,
        )
        append_metrics(metrics_path, epoch, "train", train_metrics)
        print(f"epoch {epoch} train: {asdict(train_metrics)}")

        run_validation_and_checkpoint(
            epoch=epoch,
            global_step=global_step,
            metrics_split="val_epoch",
            checkpoint_name=f"epoch_{epoch - 1}_whole.pt",
            checkpoint_epoch=epoch,
            resume_checkpoint_name=f"epoch_{epoch - 1}_whole_resume.pt",
        )

        print(f"epoch {epoch} done in {(time.time() - start) / 60:.1f} min")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
