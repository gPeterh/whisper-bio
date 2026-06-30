#!/usr/bin/env python3
"""Run full WBD inference/evaluation on Arrow shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wbd.config import MAX_WHISPER_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WBD inference/evaluation on a test set.")
    parser.add_argument(
        "--test-dir",
        default="/ceph/work/KLP/peter1/WALTZ_storage/waltz_datasets/test",
        help="Directory containing test Arrow shards.",
    )
    parser.add_argument("--test-files", nargs="*", default=None, help="Explicit test Arrow files.")
    parser.add_argument("--test-manifest", default=None, help="Text file with one test Arrow file per line.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Model checkpoint .pt file.")
    parser.add_argument("--output-dir", type=Path, default=Path("infer_result/test"))
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Optional directory for raw predicted label Parquet cache.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=MAX_WHISPER_TOKENS)
    parser.add_argument("--label-col", default=None, help="Defaults to checkpoint label_col, then wbd_labels.")
    parser.add_argument("--label-len-col", default="wbd_label_len")
    parser.add_argument("--frame-selection", choices=["none", "mid", "first", "last"], default=None)
    parser.add_argument("--boundary-tolerance", type=int, default=None, help="Boundary tolerance in 20 ms frames.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument(
        "--causal-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint causal encoder setting. Defaults to the checkpoint value.",
    )
    parser.add_argument(
        "--encoder-lookahead-frames",
        type=int,
        default=None,
        help="Override checkpoint encoder lookahead. Defaults to the checkpoint value, or 0 for old checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from wbd.inference import (
        InferenceTotals,
        PredictionCacheWriter,
        checkpoint_arg,
        cleanup_distributed,
        finalize_totals,
        load_model_from_checkpoint,
        make_eval_loader,
        merge_prediction_manifests,
        metrics_dict,
        reduce_totals,
        resolve_eval_files,
        run_inference,
        setup_distributed,
        shard_files,
        write_json,
        write_metrics_csv,
    )
    from wbd.model import ensure_transformers_torch_backend

    ensure_transformers_torch_backend()
    device, dist_state = setup_distributed(args.device)

    try:
        files = resolve_eval_files(
            data_dir=args.test_dir,
            files=args.test_files,
            manifest=args.test_manifest,
        )
        local_files = shard_files(files, dist_state)
        if dist_state.is_main_process:
            print(
                f"distributed={dist_state.enabled} world_size={dist_state.world_size} "
                f"total_files={len(files)}"
            )
        print(f"rank={dist_state.rank} local_files={len(local_files)} device={device}", flush=True)

        model, ckpt = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=device,
            attn_implementation=args.attn_implementation,
            causal_encoder=args.causal_encoder,
            encoder_lookahead_frames=args.encoder_lookahead_frames,
        )

        model_name = ckpt.get("model_name_or_path") or checkpoint_arg(ckpt, "model_name_or_path")
        label_col = args.label_col or checkpoint_arg(ckpt, "label_col", "wbd_labels")
        frame_selection = args.frame_selection or checkpoint_arg(ckpt, "frame_selection", "none")
        boundary_tolerance = args.boundary_tolerance
        if boundary_tolerance is None:
            boundary_tolerance = int(checkpoint_arg(ckpt, "boundary_tolerance", 2))

        loader = None
        if local_files:
            loader = make_eval_loader(
                files=local_files,
                model_name_or_path=model_name,
                label_col=label_col,
                label_len_col=args.label_len_col,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=device.type == "cuda" if args.pin_memory is None else args.pin_memory,
            )

        prediction_dir = args.prediction_dir
        if prediction_dir is not None and dist_state.enabled:
            prediction_dir = prediction_dir / f"rank_{dist_state.rank:02d}"
        prediction_writer = None
        if prediction_dir is not None:
            prediction_writer = PredictionCacheWriter(
                prediction_dir,
                checkpoint=args.checkpoint,
                split=args.split,
                label_col=label_col,
                label_len_col=args.label_len_col,
                max_tokens=args.max_tokens,
                files=local_files,
            )
        totals = InferenceTotals()
        try:
            if loader is not None:
                _, totals = run_inference(
                    model=model,
                    loader=loader,
                    device=device,
                    amp=args.amp,
                    amp_dtype=args.amp_dtype,
                    boundary_tolerance=boundary_tolerance,
                    frame_selection=frame_selection,
                    desc=f"{args.split} inference rank {dist_state.rank}",
                    prediction_writer=prediction_writer,
                    show_progress=dist_state.is_main_process,
                )
        finally:
            if prediction_writer is not None:
                prediction_writer.close()

        totals = reduce_totals(totals, device, dist_state)
        if dist_state.enabled:
            torch.distributed.barrier()
        if args.prediction_dir is not None and dist_state.enabled and dist_state.is_main_process:
            merge_prediction_manifests(args.prediction_dir, world_size=dist_state.world_size)
        if dist_state.enabled:
            torch.distributed.barrier()

        if not dist_state.is_main_process:
            return

        metrics = finalize_totals(totals)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "checkpoint": str(args.checkpoint),
            "files": files,
            "label_col": label_col,
            "frame_selection": frame_selection,
            "boundary_tolerance": boundary_tolerance,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "prediction_dir": str(args.prediction_dir) if args.prediction_dir is not None else None,
            "encoder_lookahead_frames": model.encoder_lookahead_frames,
            "distributed": dist_state.enabled,
            "world_size": dist_state.world_size,
        }
        result = metrics_dict(
            metrics=metrics,
            totals=totals,
            checkpoint=args.checkpoint,
            label_col=label_col,
            frame_selection=frame_selection,
            boundary_tolerance=boundary_tolerance,
            files=files,
            split=args.split,
        )
        result.update(
            {
                "distributed": dist_state.enabled,
                "world_size": dist_state.world_size,
            }
        )
        write_json(args.output_dir / "run_config.json", config)
        write_json(args.output_dir / "metrics.json", result)
        write_metrics_csv(args.output_dir / "metrics.csv", result)

        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"metrics: {args.output_dir / 'metrics.json'}")
    finally:
        cleanup_distributed(dist_state)


if __name__ == "__main__":
    main()
