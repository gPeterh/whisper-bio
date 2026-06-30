"""Shared model loading and inference helpers for WBD evaluation."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import IGNORE_ID
from .data import WaltzArrowWBDDataset, WhisperCollator, read_manifest
from .metrics import BatchMetrics, boundary_positions, paper_segmentation_metrics, token_boundary_metrics
from .model import WhisperWBDClassifier
from .trainer import autocast_context


FRAME_SECONDS = 0.02
LABEL_NAMES = {0: "O", 1: "B", 2: "I", IGNORE_ID: "IGNORE"}


@dataclass
class InferenceTotals:
    examples: int = 0
    batches: int = 0
    frames: int = 0
    token_correct: int = 0
    boundary_hits: int = 0
    boundary_pred: int = 0
    boundary_ref: int = 0


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class PredictionCacheWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        checkpoint: Path,
        split: str,
        label_col: str,
        label_len_col: str,
        max_tokens: int,
        files: Iterable[str],
    ):
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.split = split
        self.label_col = label_col
        self.label_len_col = label_len_col
        self.max_tokens = max_tokens
        self.files = list(files)
        self.num_examples = 0
        self.schema = pa.schema(
            [
                pa.field("source_file", pa.string()),
                pa.field("source_index", pa.int32()),
                pa.field("key", pa.string()),
                pa.field("label_len", pa.int16()),
                pa.field("pred_labels", pa.list_(pa.int8())),
            ]
        )
        self.prediction_files = self._make_prediction_files(self.files)
        self.row_counts = {source_file: 0 for source_file in self.files}
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_stem(source_file: str) -> str:
        stem = Path(source_file).stem
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
        return cleaned or "source"

    def _make_prediction_files(self, files: list[str]) -> dict[str, str]:
        stems = [self._safe_stem(source_file) for source_file in files]
        duplicate_stems = {stem for stem in stems if stems.count(stem) > 1}
        prediction_files = {}
        for index, (source_file, stem) in enumerate(zip(files, stems)):
            if stem in duplicate_stems:
                prediction_files[source_file] = f"{index:05d}_{stem}.parquet"
            else:
                prediction_files[source_file] = f"{stem}.parquet"
        return prediction_files

    def add_batch(self, batch: dict, logits: torch.Tensor) -> None:
        pred = logits.argmax(dim=-1).detach().cpu()
        label_lens = batch["label_lens"].detach().cpu()
        source_indices = batch["source_indices"].detach().cpu()
        rows_by_source: dict[str, list[dict]] = {}
        for batch_index, key in enumerate(batch["keys"]):
            label_len = int(label_lens[batch_index].item())
            source_file = batch["source_files"][batch_index]
            rows_by_source.setdefault(source_file, []).append(
                {
                    "source_file": source_file,
                    "source_index": int(source_indices[batch_index].item()),
                    "key": key,
                    "label_len": label_len,
                    "pred_labels": pred[batch_index, :label_len].to(torch.int8).tolist(),
                }
            )
            self.num_examples += 1
        for source_file, rows in rows_by_source.items():
            self.write_rows(source_file, rows)

    def write_rows(self, source_file: str, rows: list[dict]) -> None:
        if not rows:
            return
        if source_file not in self.prediction_files:
            self.prediction_files[source_file] = f"{len(self.prediction_files):05d}_{self._safe_stem(source_file)}.parquet"
            self.row_counts[source_file] = 0

        table = self.table_from_rows(rows)
        if source_file not in self.writers:
            path = self.output_dir / self.prediction_files[source_file]
            self.writers[source_file] = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.writers[source_file].write_table(table)
        self.row_counts[source_file] += len(rows)

    def table_from_rows(self, rows: list[dict]) -> pa.Table:
        return pa.table(
            {
                "source_file": pa.array([row["source_file"] for row in rows], type=pa.string()),
                "source_index": pa.array([row["source_index"] for row in rows], type=pa.int32()),
                "key": pa.array([row["key"] for row in rows], type=pa.string()),
                "label_len": pa.array([row["label_len"] for row in rows], type=pa.int16()),
                "pred_labels": pa.array([row["pred_labels"] for row in rows], type=pa.list_(pa.int8())),
            },
            schema=self.schema,
        )

    def empty_table(self) -> pa.Table:
        return pa.table(
            {
                "source_file": pa.array([], type=pa.string()),
                "source_index": pa.array([], type=pa.int32()),
                "key": pa.array([], type=pa.string()),
                "label_len": pa.array([], type=pa.int16()),
                "pred_labels": pa.array([], type=pa.list_(pa.int8())),
            },
            schema=self.schema,
        )

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        for source_file, prediction_file in self.prediction_files.items():
            if self.row_counts.get(source_file, 0) == 0:
                pq.write_table(self.empty_table(), self.output_dir / prediction_file, compression="zstd")
        write_json(
            self.output_dir / "manifest.json",
            {
                "format": "wbd_prediction_cache_v1",
                "layout": "source_file",
                "checkpoint": str(self.checkpoint),
                "split": self.split,
                "label_col": self.label_col,
                "label_len_col": self.label_len_col,
                "max_tokens": self.max_tokens,
                "frame_seconds": FRAME_SECONDS,
                "files": [
                    {
                        "source_file": source_file,
                        "prediction_file": self.prediction_files[source_file],
                        "num_examples": self.row_counts.get(source_file, 0),
                    }
                    for source_file in self.files
                ],
                "columns": ["source_file", "source_index", "key", "label_len", "pred_labels"],
                "labels_truncated_to_label_len": True,
                "num_examples": self.num_examples,
                "num_files": len(self.files),
            },
        )


def checkpoint_arg(ckpt: dict, name: str, default=None):
    return (ckpt.get("args") or {}).get(name, default)


def setup_distributed(device_arg: str) -> tuple[torch.device, DistributedState]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return torch.device(device_arg), DistributedState(False, 0, 0, 1)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and str(device_arg).startswith("cuda")
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_arg)
    return device, DistributedState(True, rank, local_rank, world_size)


def cleanup_distributed(dist_state: DistributedState) -> None:
    if dist_state.enabled:
        dist.destroy_process_group()


def shard_files(files: list[str], dist_state: DistributedState) -> list[str]:
    if not dist_state.enabled:
        return files
    return files[dist_state.rank :: dist_state.world_size]


def frame_time(frame_index: int) -> float:
    return round(frame_index * FRAME_SECONDS, 6)


def label_name(label_id: int) -> str:
    return LABEL_NAMES.get(label_id, str(label_id))


def resolve_eval_files(
    *,
    data_dir: str | None,
    files: list[str] | None,
    manifest: str | None,
    glob_pattern: str = "*.arrow",
    limit_shards: int | None = None,
) -> list[str]:
    resolved: list[str] = []
    if files:
        resolved.extend(files)
    resolved.extend(read_manifest(manifest))
    if not resolved and data_dir:
        resolved = sorted(str(path) for path in Path(data_dir).glob(glob_pattern))
    if limit_shards is not None:
        resolved = resolved[:limit_shards]
    if not resolved:
        raise ValueError("No Arrow files resolved for inference.")
    return resolved


def merge_prediction_manifests(prediction_dir: Path, *, world_size: int) -> None:
    merged_files = []
    num_examples = 0
    base_manifest = None
    for rank in range(world_size):
        rank_dir = prediction_dir / f"rank_{rank:02d}"
        manifest_path = rank_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if base_manifest is None:
            base_manifest = dict(manifest)
        num_examples += int(manifest.get("num_examples", 0))
        for entry in manifest.get("files", []):
            merged_entry = dict(entry)
            merged_entry["prediction_file"] = f"rank_{rank:02d}/{entry['prediction_file']}"
            merged_files.append(merged_entry)

    if base_manifest is None:
        return

    base_manifest.update(
        {
            "distributed": True,
            "world_size": world_size,
            "files": merged_files,
            "num_examples": num_examples,
            "num_files": len(merged_files),
        }
    )
    write_json(prediction_dir / "manifest.json", base_manifest)


def load_classifier_state_compat(model: WhisperWBDClassifier, ckpt: dict, checkpoint_path: Path) -> None:
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        return
    if "model_trainable" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model_trainable"], strict=False)
        non_encoder_missing = [key for key in missing if not key.startswith("encoder.")]
        if unexpected or non_encoder_missing:
            raise ValueError(
                f"Checkpoint {checkpoint_path} did not match model. "
                f"missing={non_encoder_missing}, unexpected={unexpected}"
            )
        return
    if "classifier" not in ckpt:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain model or classifier weights.")

    state_dict = ckpt["classifier"]
    try:
        model.classifier.load_state_dict(state_dict)
        return
    except RuntimeError as original_error:
        if hasattr(model.classifier, "net") and all(not key.startswith("net.") for key in state_dict):
            remapped = {f"net.{key}": value for key, value in state_dict.items()}
            try:
                model.classifier.load_state_dict(remapped)
                return
            except RuntimeError:
                pass
        raise original_error


def load_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
    attn_implementation: str | None,
    causal_encoder: bool | None,
    encoder_lookahead_frames: int | None = None,
) -> tuple[WhisperWBDClassifier, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt.get("model_name_or_path") or checkpoint_arg(ckpt, "model_name_or_path")
    if model_name is None:
        raise KeyError("Checkpoint is missing model_name_or_path.")

    resolved_attn = attn_implementation or checkpoint_arg(ckpt, "attn_implementation", "sdpa")
    resolved_causal = causal_encoder
    if resolved_causal is None:
        resolved_causal = bool(checkpoint_arg(ckpt, "causal_encoder", False))
    resolved_lookahead = encoder_lookahead_frames
    if resolved_lookahead is None:
        resolved_lookahead = int(
            ckpt.get(
                "encoder_lookahead_frames",
                checkpoint_arg(ckpt, "encoder_lookahead_frames", 0),
            )
        )
    model = WhisperWBDClassifier(
        model_name_or_path=model_name,
        encoder_layer=int(ckpt.get("encoder_layer", checkpoint_arg(ckpt, "encoder_layer", 6))),
        freeze_encoder=bool(ckpt.get("freeze_encoder", checkpoint_arg(ckpt, "freeze_encoder", True))),
        head_type=checkpoint_arg(ckpt, "head_type", "linear"),
        dropout=float(checkpoint_arg(ckpt, "dropout", 0.1)),
        attn_implementation=resolved_attn,
        causal_encoder=resolved_causal,
        encoder_lookahead_frames=resolved_lookahead,
        encoder_layers=ckpt.get("encoder_layers", checkpoint_arg(ckpt, "encoder_layers", None)),
        layer_fusion=ckpt.get("layer_fusion", checkpoint_arg(ckpt, "layer_fusion", None)),
    ).to(device)
    load_classifier_state_compat(model, ckpt, checkpoint_path)
    model.eval()
    return model, ckpt


def make_eval_dataset(
    *,
    files: list[str],
    label_col: str,
    label_len_col: str,
    max_tokens: int,
) -> WaltzArrowWBDDataset:
    return WaltzArrowWBDDataset(
        files=files,
        label_col=label_col,
        label_len_col=label_len_col,
        max_tokens=max_tokens,
        shuffle_buffer=0,
        seed=0,
        shuffle_files=False,
    )


def make_eval_loader(
    *,
    files: list[str],
    model_name_or_path: str,
    label_col: str,
    label_len_col: str,
    max_tokens: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = make_eval_dataset(
        files=files,
        label_col=label_col,
        label_len_col=label_len_col,
        max_tokens=max_tokens,
    )
    collator = WhisperCollator(model_name_or_path, max_tokens)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collator,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def add_batch_to_totals(
    totals: InferenceTotals,
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_lens: torch.Tensor,
    boundary_tolerance: int,
    frame_selection: str,
) -> None:
    pred = logits.argmax(dim=-1)
    valid = labels != IGNORE_ID
    valid_count = int(valid.sum().item())
    if valid_count:
        totals.token_correct += int((pred[valid] == labels[valid]).sum().item())
        totals.frames += valid_count

    batch_metrics = token_boundary_metrics(
        logits.float(),
        labels,
        label_lens,
        boundary_tolerance,
        frame_selection,
    )
    totals.boundary_hits += int(batch_metrics.boundary_hits)
    totals.boundary_pred += int(batch_metrics.boundary_pred)
    totals.boundary_ref += int(batch_metrics.boundary_ref)
    totals.examples += int(labels.size(0))
    totals.batches += 1


def finalize_totals(totals: InferenceTotals) -> BatchMetrics:
    metrics = BatchMetrics()
    metrics.len = float(totals.frames)
    metrics.token_acc = totals.token_correct / totals.frames if totals.frames else 0.0
    metrics.boundary_hits = float(totals.boundary_hits)
    metrics.boundary_pred = float(totals.boundary_pred)
    metrics.boundary_ref = float(totals.boundary_ref)
    precision, recall, f_value, os_value, r_value = paper_segmentation_metrics(
        totals.boundary_hits,
        totals.boundary_pred,
        totals.boundary_ref,
    )
    metrics.segmentation_precision = precision
    metrics.segmentation_recall = recall
    metrics.segmentation_f_value = f_value
    metrics.segmentation_os = os_value
    metrics.segmentation_r_value = r_value
    return metrics


def reduce_totals(totals: InferenceTotals, device: torch.device, dist_state: DistributedState) -> InferenceTotals:
    if not dist_state.enabled:
        return totals
    values = torch.tensor(
        [
            totals.examples,
            totals.batches,
            totals.frames,
            totals.token_correct,
            totals.boundary_hits,
            totals.boundary_pred,
            totals.boundary_ref,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return InferenceTotals(
        examples=int(values[0].item()),
        batches=int(values[1].item()),
        frames=int(values[2].item()),
        token_correct=int(values[3].item()),
        boundary_hits=int(values[4].item()),
        boundary_pred=int(values[5].item()),
        boundary_ref=int(values[6].item()),
    )


def run_inference(
    *,
    model: WhisperWBDClassifier,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    boundary_tolerance: int,
    frame_selection: str,
    desc: str,
    prediction_writer: PredictionCacheWriter | None = None,
    show_progress: bool = True,
) -> tuple[BatchMetrics, InferenceTotals]:
    totals = InferenceTotals()
    autocast_args = SimpleNamespace(amp=amp, amp_dtype=amp_dtype)
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, dynamic_ncols=True, disable=not show_progress):
            input_features = batch["input_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            label_lens = batch["label_lens"].to(device, non_blocking=True)
            with autocast_context(autocast_args, device):
                logits = model(input_features)
            if prediction_writer is not None:
                prediction_writer.add_batch(batch, logits)
            add_batch_to_totals(
                totals,
                logits=logits,
                labels=labels,
                label_lens=label_lens,
                boundary_tolerance=boundary_tolerance,
                frame_selection=frame_selection,
            )
    return finalize_totals(totals), totals


def collect_examples(
    *,
    arrow_file: Path,
    label_col: str,
    label_len_col: str,
    max_tokens: int,
    num_examples: int,
) -> list[dict]:
    dataset = make_eval_dataset(
        files=[str(arrow_file)],
        label_col=label_col,
        label_len_col=label_len_col,
        max_tokens=max_tokens,
    )
    examples = []
    for example in dataset:
        examples.append(example)
        if len(examples) >= num_examples:
            break
    if not examples:
        raise RuntimeError(f"No valid labeled/audio examples found in {arrow_file}.")
    return examples


def metrics_dict(
    *,
    metrics: BatchMetrics,
    totals: InferenceTotals,
    checkpoint: Path,
    label_col: str,
    frame_selection: str,
    boundary_tolerance: int,
    files: Iterable[str],
    split: str,
) -> dict:
    metric_values = asdict(metrics)
    metric_values.pop("loss", None)
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "num_files": len(list(files)),
        "num_examples": totals.examples,
        "num_batches": totals.batches,
        "num_frames": totals.frames,
        "token_correct": totals.token_correct,
        "label_col": label_col,
        "frame_selection": frame_selection,
        "boundary_tolerance": boundary_tolerance,
        **metric_values,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_frame_csv(path: Path, gold: list[int], pred: list[int]) -> None:
    fieldnames = ["frame", "time_sec", "gold_id", "gold_label", "pred_id", "pred_label"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame, (gold_id, pred_id) in enumerate(zip(gold, pred)):
            row = {
                "frame": frame,
                "time_sec": frame_time(frame),
                "gold_id": gold_id,
                "gold_label": label_name(gold_id),
                "pred_id": pred_id,
                "pred_label": label_name(pred_id),
            }
            writer.writerow(row)


def debug_records_for_batch(
    *,
    batch: dict,
    logits: torch.Tensor,
    frame_selection: str,
) -> dict[str, dict]:
    pred = logits.argmax(dim=-1).cpu()
    labels = batch["labels"].cpu()
    label_lens = batch["label_lens"].cpu()
    records: dict[str, dict] = {}
    for batch_index, key in enumerate(batch["keys"]):
        length = int(label_lens[batch_index].item())
        gold_ids = labels[batch_index, :length].tolist()
        pred_ids = pred[batch_index, :length].tolist()
        gold_boundaries = boundary_positions(labels[batch_index, :length], frame_selection="none")
        pred_boundaries = boundary_positions(pred[batch_index, :length], frame_selection=frame_selection)
        records[key] = {
            "key": key,
            "label_len": length,
            "duration_sec": frame_time(length),
            "gold_labels": [label_name(x) for x in gold_ids],
            "pred_labels": [label_name(x) for x in pred_ids],
            "gold_label_ids": gold_ids,
            "pred_label_ids": pred_ids,
            "gold_boundary_frames": gold_boundaries,
            "gold_boundary_times": [frame_time(x) for x in gold_boundaries],
            "pred_boundary_frames": pred_boundaries,
            "pred_boundary_times": [frame_time(x) for x in pred_boundaries],
            "frame_selection": frame_selection,
        }
    return records
