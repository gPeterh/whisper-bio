#!/usr/bin/env python3
"""Add Whisper-grid BIO word-boundary labels to WALTZ Arrow shards.

This script appends four columns to existing HuggingFace Arrow stream files:

  - wbd_labels: fixed-length clean BIO labels on Whisper encoder tokens.
  - wbd_aug_label: optional begin-augmented labels for training.
  - wbd_label_len: valid token count before padding.
  - wbd_word_spans: word spans as [start_token, end_token] pairs.

Labels use 0=O, 1=B, 2=I, and -100=padding/ignore. The clean labels never
apply begin-label augmentation; when --split train is used, wbd_aug_label
contains the Word-Segmenter style B +/- 1 token augmentation.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm.auto import tqdm


BIO_O = 0
BIO_B = 1
BIO_I = 2
IGNORE_ID = -100
OBSOLETE_WBD_COLUMNS = ("wbd_train_labels",)


@dataclass
class FileStats:
    rows: int = 0
    words: int = 0
    truncated_words: int = 0
    invalid_words: int = 0
    empty_alignment_rows: int = 0
    o_count: int = 0
    b_count: int = 0
    i_count: int = 0
    ignore_count: int = 0

    def add_labels(self, labels: list[int]) -> None:
        self.o_count += labels.count(BIO_O)
        self.b_count += labels.count(BIO_B)
        self.i_count += labels.count(BIO_I)
        self.ignore_count += labels.count(IGNORE_ID)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append Whisper encoder BIO word-boundary labels to WALTZ Arrow shards."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Directory containing .arrow files.")
    source.add_argument("--manifest", type=Path, help="Text file with one .arrow path per line.")
    source.add_argument("--files", nargs="+", type=Path, help="Explicit .arrow files.")
    parser.add_argument("--glob", default="*.arrow", help="Glob used with --input-dir.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files.")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="train",
        help="Use train to apply begin-label augmentation; dev/test keep exact labels.",
    )
    parser.add_argument("--in-place", action="store_true", help="Rewrite each input Arrow file in place.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, --in-place is required.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing WBD columns if present.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files whose target already contains the required WBD columns.",
    )
    parser.add_argument(
        "--resume-check",
        choices=["schema", "full"],
        default="full",
        help="Use full to read record batches before skipping, catching truncated or corrupt outputs.",
    )
    parser.add_argument("--dt", type=float, default=0.02, help="Seconds per Whisper encoder token.")
    parser.add_argument("--max-tokens", type=int, default=1500, help="Padded label length.")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Whisper max audio duration. Words starting after this are ignored.",
    )
    parser.add_argument("--labels-col", default="wbd_labels")
    parser.add_argument("--aug-label-col", default="wbd_aug_label")
    parser.add_argument("--label-len-col", default="wbd_label_len")
    parser.add_argument("--spans-col", default="wbd_word_spans")
    parser.add_argument(
        "--no-aug-label",
        action="store_true",
        help="Do not add the augmented train-label column even when --split train.",
    )
    return parser.parse_args()


def iter_arrow_files(args: argparse.Namespace) -> list[Path]:
    if args.input_dir is not None:
        files = sorted(args.input_dir.glob(args.glob))
    elif args.manifest is not None:
        files = []
        with args.manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    files.append(Path(line.split()[0]))
    else:
        files = list(args.files)

    files = [p for p in files if p.suffix == ".arrow"]
    if args.limit is not None:
        files = files[: args.limit]
    return files


def read_arrow_stream(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_stream(source).read_all()


def open_arrow_stream(path: Path) -> tuple[pa.MemoryMappedFile, ipc.RecordBatchStreamReader]:
    source = pa.memory_map(str(path), "r")
    try:
        return source, ipc.open_stream(source)
    except Exception:
        source.close()
        raise


def write_arrow_stream(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def flatten_words(wx: Any) -> list[dict[str, Any]]:
    if not wx:
        return []
    words = wx.get("word_segments") or []
    if words:
        return words
    flattened = []
    for segment in wx.get("segments") or []:
        flattened.extend(segment.get("words") or [])
    return flattened


def build_bio(
    wx: Any,
    *,
    dt: float,
    max_tokens: int,
    max_seconds: float,
) -> tuple[list[int], int, list[list[int]], int, int]:
    words = flatten_words(wx)
    alignment_duration = 0.0
    for word in words:
        end = word.get("end")
        if end is not None:
            alignment_duration = max(alignment_duration, float(end))

    valid_len = min(max(0, math.ceil(min(alignment_duration, max_seconds) / dt)), max_tokens)
    labels = [IGNORE_ID] * max_tokens
    labels[:valid_len] = [BIO_O] * valid_len

    spans: list[list[int]] = []
    invalid_words = 0
    truncated_words = 0
    for word in words:
        start = word.get("start")
        end = word.get("end")
        if start is None or end is None:
            invalid_words += 1
            continue
        start_sec = float(start)
        end_sec = float(end)
        if start_sec >= max_seconds:
            truncated_words += 1
            continue
        token_start = max(0, min(valid_len, math.floor(start_sec / dt)))
        token_end = max(0, min(valid_len, math.ceil(min(end_sec, max_seconds) / dt)))
        if token_start >= token_end:
            invalid_words += 1
            continue

        labels[token_start:token_end] = [BIO_I] * (token_end - token_start)
        labels[token_start] = BIO_B
        spans.append([token_start, token_end])

    return labels, valid_len, spans, invalid_words, truncated_words


def augment_begin_labels(labels: list[int], spans: list[list[int]], valid_len: int) -> list[int]:
    augmented = labels.copy()
    for token_start, _ in spans:
        for idx in (token_start - 1, token_start, token_start + 1):
            if 0 <= idx < valid_len:
                augmented[idx] = BIO_B
    return augmented


def remove_existing_wbd_columns(table: pa.Table, args: argparse.Namespace) -> pa.Table:
    for name in (args.labels_col, args.aug_label_col, args.label_len_col, args.spans_col, *OBSOLETE_WBD_COLUMNS):
        if name in table.schema.names:
            table = table.drop([name])
    return table


def required_wbd_columns(args: argparse.Namespace) -> list[str]:
    columns = [args.labels_col, args.label_len_col, args.spans_col]
    if args.split == "train" and not args.no_aug_label:
        columns.append(args.aug_label_col)
    return columns


def has_resumable_wbd_columns(path: Path, args: argparse.Namespace) -> bool:
    if not path.exists():
        return False
    try:
        source, reader = open_arrow_stream(path)
    except Exception:
        return False

    try:
        schema = reader.schema
        names = set(schema.names)
        required_columns = required_wbd_columns(args)
        if not all(name in names for name in required_columns):
            return False

        for name in (args.labels_col, args.aug_label_col):
            if name not in names:
                continue
            field_type = schema.field(name).type
            if getattr(field_type, "list_size", None) != args.max_tokens:
                return False

        if args.resume_check == "schema":
            return True

        total_rows = 0
        required_indices = {name: schema.get_field_index(name) for name in required_columns}
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            except Exception:
                return False

            total_rows += batch.num_rows
            for idx in required_indices.values():
                column = batch.column(idx)
                if len(column) != batch.num_rows or column.null_count:
                    return False
                if hasattr(column, "values") and column.values.null_count:
                    return False
            label_lens = batch.column(required_indices[args.label_len_col]).to_pylist()
            if any(length is None or length < 0 or length > args.max_tokens for length in label_lens):
                return False
        return total_rows > 0
    finally:
        source.close()


def add_wbd_columns(table: pa.Table, args: argparse.Namespace) -> tuple[pa.Table, FileStats]:
    existing = [
        c
        for c in (args.labels_col, args.aug_label_col, args.label_len_col, args.spans_col, *OBSOLETE_WBD_COLUMNS)
        if c in table.schema.names
    ]
    if existing and not args.overwrite:
        raise ValueError(f"Existing WBD columns found: {existing}. Pass --overwrite to replace them.")
    if existing:
        table = remove_existing_wbd_columns(table, args)

    wx_values = table.column("whisperX_time_align").to_pylist()

    stats = FileStats(rows=table.num_rows)
    all_labels: list[list[int]] = []
    all_aug_labels: list[list[int]] = []
    all_label_lens: list[int] = []
    all_spans: list[list[list[int]]] = []
    add_aug_labels = args.split == "train" and not args.no_aug_label

    for wx in wx_values:
        if not flatten_words(wx):
            stats.empty_alignment_rows += 1
        labels, label_len, spans, invalid_words, truncated_words = build_bio(
            wx,
            dt=args.dt,
            max_tokens=args.max_tokens,
            max_seconds=args.max_seconds,
        )
        all_labels.append(labels)
        if add_aug_labels:
            all_aug_labels.append(augment_begin_labels(labels, spans, label_len))
        all_label_lens.append(label_len)
        all_spans.append(spans)
        stats.words += len(spans)
        stats.invalid_words += invalid_words
        stats.truncated_words += truncated_words
        stats.add_labels(labels)

    labels_type = pa.list_(pa.int16(), list_size=args.max_tokens)
    spans_type = pa.list_(pa.list_(pa.int16(), list_size=2))
    table = table.append_column(args.labels_col, pa.array(all_labels, type=labels_type))
    if add_aug_labels:
        table = table.append_column(args.aug_label_col, pa.array(all_aug_labels, type=labels_type))
    table = table.append_column(args.label_len_col, pa.array(all_label_lens, type=pa.int16()))
    table = table.append_column(args.spans_col, pa.array(all_spans, type=spans_type))
    return table, stats


def output_path_for(path: Path, args: argparse.Namespace) -> Path:
    if args.in_place:
        return path
    if args.output_dir is None:
        raise ValueError("Either --in-place or --output-dir is required.")
    return args.output_dir / path.name


def is_done_for_resume(path: Path, args: argparse.Namespace) -> bool:
    return args.resume and has_resumable_wbd_columns(output_path_for(path, args), args)


def rewrite_one(path: Path, args: argparse.Namespace) -> FileStats:
    table = read_arrow_stream(path)
    new_table, stats = add_wbd_columns(table, args)
    out_path = output_path_for(path, args)

    if args.in_place:
        original_mode = path.stat().st_mode
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            write_arrow_stream(new_table, tmp_path)
            os.chmod(tmp_path, original_mode)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        write_arrow_stream(new_table, out_path)

    return stats


def main() -> None:
    args = parse_args()
    if not args.in_place and args.output_dir is None:
        raise SystemExit("Refusing to write without --in-place or --output-dir.")

    files = iter_arrow_files(args)
    if not files:
        raise SystemExit("No .arrow files found.")

    print(
        f"Processing {len(files)} file(s), split={args.split}, "
        f"in_place={args.in_place}, resume={args.resume}"
    )
    totals = FileStats()
    skipped = 0
    for idx, path in enumerate(tqdm(files, desc="WBD Arrow shards", unit="file"), start=1):
        if is_done_for_resume(path, args):
            skipped += 1
            tqdm.write(f"[{idx}/{len(files)}] {path.name}: skipped existing WBD columns")
            continue

        stats = rewrite_one(path, args)
        for field in totals.__dataclass_fields__:
            setattr(totals, field, getattr(totals, field) + getattr(stats, field))
        tqdm.write(
            f"[{idx}/{len(files)}] {path.name}: rows={stats.rows} words={stats.words} "
            f"empty_rows={stats.empty_alignment_rows} invalid_words={stats.invalid_words} "
            f"truncated_words={stats.truncated_words} "
            f"labels(O/B/I/IGN)={stats.o_count}/{stats.b_count}/{stats.i_count}/{stats.ignore_count}"
        )

    print(
        "TOTAL: "
        f"processed_files={len(files) - skipped} skipped_files={skipped} "
        f"rows={totals.rows} words={totals.words} empty_rows={totals.empty_alignment_rows} "
        f"invalid_words={totals.invalid_words} truncated_words={totals.truncated_words} "
        f"labels(O/B/I/IGN)={totals.o_count}/{totals.b_count}/{totals.i_count}/{totals.ignore_count}"
    )


if __name__ == "__main__":
    main()
