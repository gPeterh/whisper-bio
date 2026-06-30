#!/usr/bin/env python3
"""Evaluate cached raw WBD predictions against source Arrow labels."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

from wbd.config import MAX_WHISPER_TOKENS
from wbd.metrics import BatchMetrics, paper_segmentation_metrics, prediction_boundary_metrics, repair_bio_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw prediction cache without running Whisper inference.")
    parser.add_argument("--prediction-dir", required=True, type=Path, help="Directory containing manifest.json.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for metrics.json/csv.")
    parser.add_argument("--label-col", default=None, help="Defaults to manifest label_col, then wbd_labels.")
    parser.add_argument("--label-len-col", default=None, help="Defaults to manifest label_len_col, then wbd_label_len.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Defaults to manifest max_tokens, then 1500.")
    parser.add_argument("--frame-selection", choices=["none", "mid", "first", "last"], default="none")
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument(
        "--post-process",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply BIO repair to cached predicted labels before scoring.",
    )
    parser.add_argument("--split", default=None, help="Defaults to manifest split.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def load_prediction_rows(path: Path) -> dict[int, dict]:
    table = pq.read_table(path)
    rows = {}
    for row in table.to_pylist():
        source_index = int(row["source_index"])
        if source_index in rows:
            raise ValueError(f"Duplicate source_index={source_index} in {path}")
        rows[source_index] = row
    return rows


def checked_prediction_file(prediction_dir: Path, entry: dict) -> Path:
    prediction_file = prediction_dir / entry["prediction_file"]
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
    return prediction_file


def source_label_rows(
    source_file: str,
    *,
    label_col: str,
    label_len_col: str,
    max_tokens: int,
    wanted_indices: set[int],
):
    with pa.memory_map(source_file, "r") as source:
        reader = ipc.open_stream(source)
        names = reader.schema.names
        for required in ("__key__", label_col, label_len_col):
            if required not in names:
                raise KeyError(f"{source_file} is missing required column {required!r}.")

        key_index = names.index("__key__")
        label_index = names.index(label_col)
        label_len_index = names.index(label_len_col)
        row_index = 0
        for batch in reader:
            key_column = batch.column(key_index)
            labels_column = batch.column(label_index)
            label_len_column = batch.column(label_len_index)
            for batch_index in range(batch.num_rows):
                current_index = row_index
                row_index += 1
                if current_index not in wanted_indices:
                    continue

                labels = labels_column[batch_index].as_py()
                if labels is None:
                    raise ValueError(f"{source_file}:{current_index} has null labels.")
                label_len = int(label_len_column[batch_index].as_py())
                label_len = max(0, min(label_len, max_tokens, len(labels)))
                yield {
                    "source_index": current_index,
                    "key": key_column[batch_index].as_py(),
                    "label_len": label_len,
                    "labels": labels[:label_len],
                }


def score_pair(
    *,
    pred_labels: list[int],
    gold_labels: list[int],
    label_len: int,
    boundary_tolerance: int,
    frame_selection: str,
    post_process: bool,
) -> BatchMetrics:
    pred = torch.tensor(pred_labels, dtype=torch.long).unsqueeze(0)
    labels = torch.tensor(gold_labels, dtype=torch.long).unsqueeze(0)
    label_lens = torch.tensor([label_len], dtype=torch.long)
    if post_process:
        pred = repair_bio_predictions(pred, label_lens)
    return prediction_boundary_metrics(pred, labels, label_lens, boundary_tolerance, frame_selection)


def add_metrics(totals: BatchMetrics, metrics: BatchMetrics) -> None:
    totals.len += metrics.len
    totals.token_acc += metrics.token_acc * metrics.len
    totals.boundary_hits += metrics.boundary_hits
    totals.boundary_pred += metrics.boundary_pred
    totals.boundary_ref += metrics.boundary_ref


def finalize(totals: BatchMetrics) -> BatchMetrics:
    if totals.len:
        totals.token_acc = totals.token_acc / totals.len
    precision, recall, f_value, os_value, r_value = paper_segmentation_metrics(
        totals.boundary_hits,
        totals.boundary_pred,
        totals.boundary_ref,
    )
    totals.segmentation_precision = precision
    totals.segmentation_recall = recall
    totals.segmentation_f_value = f_value
    totals.segmentation_os = os_value
    totals.segmentation_r_value = r_value
    return totals


def main() -> None:
    args = parse_args()
    manifest = read_json(args.prediction_dir / "manifest.json")
    if manifest.get("format") != "wbd_prediction_cache_v1":
        raise ValueError(f"Unsupported prediction cache format: {manifest.get('format')!r}")

    label_col = args.label_col or manifest.get("label_col") or "wbd_labels"
    label_len_col = args.label_len_col or manifest.get("label_len_col") or "wbd_label_len"
    max_tokens = args.max_tokens or int(manifest.get("max_tokens", MAX_WHISPER_TOKENS))
    split = args.split or manifest.get("split", "")

    totals = BatchMetrics()
    num_examples = 0
    num_files = 0
    for entry in tqdm(manifest["files"], desc="prediction files", dynamic_ncols=True):
        source_file = entry["source_file"]
        prediction_file = checked_prediction_file(args.prediction_dir, entry)
        prediction_rows = load_prediction_rows(prediction_file)
        seen_indices: set[int] = set()
        for gold in source_label_rows(
            source_file,
            label_col=label_col,
            label_len_col=label_len_col,
            max_tokens=max_tokens,
            wanted_indices=set(prediction_rows),
        ):
            source_index = gold["source_index"]
            pred_row = prediction_rows[source_index]
            seen_indices.add(source_index)

            if pred_row["source_file"] != source_file:
                raise ValueError(f"source_file mismatch at {source_file}:{source_index}")
            if pred_row["key"] != gold["key"]:
                raise ValueError(
                    f"key mismatch at {source_file}:{source_index}: pred={pred_row['key']!r} gold={gold['key']!r}"
                )
            if int(pred_row["label_len"]) != gold["label_len"]:
                raise ValueError(
                    f"label_len mismatch at {source_file}:{source_index}: "
                    f"pred={pred_row['label_len']} gold={gold['label_len']}"
                )

            metrics = score_pair(
                pred_labels=pred_row["pred_labels"],
                gold_labels=gold["labels"],
                label_len=gold["label_len"],
                boundary_tolerance=args.boundary_tolerance,
                frame_selection=args.frame_selection,
                post_process=args.post_process,
            )
            add_metrics(totals, metrics)
            num_examples += 1

        missing = set(prediction_rows) - seen_indices
        if missing:
            first_missing = min(missing)
            raise ValueError(f"{len(missing)} prediction rows did not match source labels; first={first_missing}")
        num_files += 1

    metrics = finalize(totals)
    metric_values = asdict(metrics)
    metric_values.pop("loss", None)
    result = {
        "prediction_dir": str(args.prediction_dir),
        "split": split,
        "num_files": num_files,
        "num_examples": num_examples,
        "label_col": label_col,
        "label_len_col": label_len_col,
        "max_tokens": max_tokens,
        "frame_selection": args.frame_selection,
        "post_process": args.post_process,
        "boundary_tolerance": args.boundary_tolerance,
        **metric_values,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", result)
    write_metrics_csv(args.output_dir / "metrics.csv", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
