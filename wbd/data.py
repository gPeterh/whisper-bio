"""WALTZ Arrow reading, audio decoding, and Whisper batch collation."""

from __future__ import annotations

import random
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
import torchaudio
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
    waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.to(torch.float32)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
    max_abs = waveform.abs().max()
    if max_abs > 1.0:
        waveform = waveform / max_abs
    max_samples = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)
    if waveform.numel() > max_samples:
        waveform = waveform[:max_samples]
    return waveform.contiguous(), sample_rate


def iter_arrow_rows(path: str) -> Iterable[dict]:
    with pa.memory_map(path, "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            table = pa.Table.from_batches([batch])
            for row in table.to_pylist():
                yield row


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

    def _iter_examples(self) -> Iterable[dict]:
        for path in self._worker_files():
            row_count = 0
            for row in iter_arrow_rows(path):
                if self.max_rows_per_shard is not None and row_count >= self.max_rows_per_shard:
                    break
                row_count += 1
                if self.label_col not in row or self.label_len_col not in row:
                    continue
                labels = row[self.label_col]
                if labels is None:
                    continue
                label_len = int(row[self.label_len_col])
                label_len = max(0, min(label_len, self.max_tokens, len(labels)))
                try:
                    waveform, sample_rate = decode_audio(row["mp3"])
                except Exception as exc:
                    key = row.get("__key__", "<unknown>")
                    print(f"[warn] skip {key}: audio decode failed: {exc}", flush=True)
                    continue
                labels = torch.tensor(labels[: self.max_tokens], dtype=torch.long)
                if labels.numel() < self.max_tokens:
                    labels = F.pad(labels, (0, self.max_tokens - labels.numel()), value=IGNORE_ID)
                labels[label_len:] = IGNORE_ID
                yield {
                    "key": row.get("__key__", ""),
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "labels": labels,
                    "label_len": label_len,
                }

    def __iter__(self) -> Iterable[dict]:
        if self.shuffle_buffer <= 0:
            yield from self._iter_examples()
            return

        rng = random.Random(self.seed + self.epoch * 100_003)
        buffer: deque[dict] = deque()
        for example in self._iter_examples():
            buffer.append(example)
            if len(buffer) >= self.shuffle_buffer:
                idx = rng.randrange(len(buffer))
                buffer.rotate(-idx)
                yield buffer.popleft()
                buffer.rotate(idx)
        while buffer:
            idx = rng.randrange(len(buffer))
            buffer.rotate(-idx)
            yield buffer.popleft()
            buffer.rotate(idx)


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
