"""WALTZ Arrow reading, audio decoding, and Whisper batch collation."""

from __future__ import annotations

import math
import random
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import IterableDataset, get_worker_info
from transformers import WhisperFeatureExtractor

from .config import IGNORE_ID, MAX_AUDIO_SECONDS, SAMPLE_RATE


def read_manifest(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]


def resolve_files(args) -> tuple[list[str], list[str]]:
    train_files = []
    if args.train_files:
        train_files.extend(args.train_files)
    train_files.extend(read_manifest(args.train_manifest))
    if not train_files:
        train_files = sorted(str(p) for p in Path(args.train_dir).glob(args.glob))
    if args.limit_shards is not None:
        train_files = train_files[: args.limit_shards]

    val_files = []
    if args.val_files:
        val_files.extend(args.val_files)
    val_files.extend(read_manifest(args.val_manifest))
    if not val_files and args.val_dir:
        val_files = sorted(str(p) for p in Path(args.val_dir).glob(args.val_glob))
        if not val_files:
            raise ValueError(f"No validation Arrow files resolved from --val-dir {args.val_dir!r}.")

    if not train_files:
        raise ValueError("No train Arrow files resolved.")
    return train_files, val_files


def decode_audio(mp3_field: dict) -> tuple[torch.Tensor, int]:
    audio_bytes = mp3_field.get("bytes")
    if not audio_bytes:
        raise ValueError("mp3.bytes is empty")
    with BytesIO(audio_bytes) as audio_file:
        audio, sample_rate = sf.read(audio_file, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        sample_rate = SAMPLE_RATE
    waveform = torch.from_numpy(audio)
    max_abs = waveform.abs().max()
    if max_abs > 1.0:
        waveform = waveform / max_abs
    max_samples = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)
    if waveform.numel() > max_samples:
        waveform = waveform[:max_samples]
    return waveform.contiguous(), sample_rate


def iter_arrow_raw_examples(
    path: str,
    *,
    label_col: str,
    label_len_col: str,
    max_tokens: int,
    max_rows_per_shard: int | None,
) -> Iterable[dict]:
    with pa.memory_map(path, "r") as source:
        reader = ipc.open_stream(source)
        row_count = 0
        for batch in reader:
            column_names = batch.schema.names
            if "mp3" not in column_names or label_col not in column_names or label_len_col not in column_names:
                continue
            key_column = batch.column(column_names.index("__key__")) if "__key__" in column_names else None
            mp3_column = batch.column(column_names.index("mp3"))
            labels_column = batch.column(column_names.index(label_col))
            label_len_column = batch.column(column_names.index(label_len_col))
            for idx in range(batch.num_rows):
                if max_rows_per_shard is not None and row_count >= max_rows_per_shard:
                    return
                row_count += 1
                labels = labels_column[idx].as_py()
                if labels is None:
                    continue
                label_len = int(label_len_column[idx].as_py())
                label_len = max(0, min(label_len, max_tokens, len(labels)))
                if label_len <= 0:
                    continue
                yield {
                    "key": key_column[idx].as_py() if key_column is not None else "",
                    "mp3": mp3_column[idx].as_py(),
                    "labels": labels,
                    "label_len": label_len,
                }


class WaltzArrowWBDDataset(IterableDataset):
    def __init__(
        self,
        files: list[str],
        label_col: str,
        label_len_col: str,
        max_tokens: int,
        max_rows_per_shard: int | None,
        shuffle_buffer: int,
        seed: int,
        shuffle_files: bool,
    ):
        super().__init__()
        self.files = files
        self.label_col = label_col
        self.label_len_col = label_len_col
        self.max_tokens = max_tokens
        self.max_rows_per_shard = max_rows_per_shard
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.shuffle_files = shuffle_files
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_files(self) -> list[str]:
        files = self.files[:]
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
            seed = self.seed + self.epoch * 100_003 + worker.id
        else:
            seed = self.seed + self.epoch * 100_003
        if self.shuffle_files:
            random.Random(seed).shuffle(files)
        return files

    def _iter_raw_examples(self) -> Iterable[dict]:
        for path in self._worker_files():
            yield from iter_arrow_raw_examples(
                path,
                label_col=self.label_col,
                label_len_col=self.label_len_col,
                max_tokens=self.max_tokens,
                max_rows_per_shard=self.max_rows_per_shard,
            )

    def _shuffle_raw_examples(self, raw_examples: Iterable[dict]) -> Iterable[dict]:
        rng = random.Random(self.seed + self.epoch * 100_003)
        buffer = []
        for example in raw_examples:
            if len(buffer) < self.shuffle_buffer:
                buffer.append(example)
                continue
            idx = rng.randrange(self.shuffle_buffer)
            yield buffer[idx]
            buffer[idx] = example
        rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def _materialize_example(self, raw: dict) -> dict | None:
        try:
            waveform, sample_rate = decode_audio(raw["mp3"])
        except Exception as exc:
            key = raw.get("key", "<unknown>")
            print(f"[warn] skip {key}: audio decode failed: {exc}", flush=True)
            return None
        label_len = raw["label_len"]
        labels = torch.tensor(raw["labels"][: self.max_tokens], dtype=torch.long)
        if labels.numel() < self.max_tokens:
            labels = F.pad(labels, (0, self.max_tokens - labels.numel()), value=IGNORE_ID)
        labels[label_len:] = IGNORE_ID
        return {
            "key": raw.get("key", ""),
            "waveform": waveform,
            "sample_rate": sample_rate,
            "labels": labels,
            "label_len": label_len,
        }

    def _iter_materialized_examples(self) -> Iterable[dict]:
        raw_examples = self._iter_raw_examples()
        if self.shuffle_buffer > 0:
            raw_examples = self._shuffle_raw_examples(raw_examples)
        for raw in raw_examples:
            example = self._materialize_example(raw)
            if example is not None:
                yield example

    def __iter__(self) -> Iterable[dict]:
        yield from self._iter_materialized_examples()


class WhisperCollator:
    def __init__(self, model_name_or_path: str, max_tokens: int):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
        self.max_tokens = max_tokens

    def __call__(self, examples: list[dict]) -> dict:
        waveforms = [ex["waveform"].numpy() for ex in examples]
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        labels = torch.stack([ex["labels"][: self.max_tokens] for ex in examples], dim=0)
        label_lens = torch.tensor([ex["label_len"] for ex in examples], dtype=torch.long)
        keys = [ex["key"] for ex in examples]
        return {
            "input_features": inputs.input_features,
            "labels": labels,
            "label_lens": label_lens,
            "keys": keys,
        }
