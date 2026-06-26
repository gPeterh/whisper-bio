"""Weights & Biases setup helpers."""

from __future__ import annotations

import argparse


def init_wandb(args: argparse.Namespace, train_files: list[str], val_files: list[str]):
    if not args.use_wandb:
        return None
    import wandb

    config = vars(args).copy()
    config.update(
        {
            "num_train_shards": len(train_files),
            "num_val_shards": len(val_files),
            "world_size": getattr(args, "world_size", 1),
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * getattr(args, "world_size", 1),
            "effective_batch_size": args.batch_size * getattr(args, "world_size", 1) * args.grad_accum_steps,
            "label_set": "augmented" if args.label_col == "wbd_aug_label" else "clean",
        }
    )
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        resume=args.wandb_resume,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=config,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    return run
