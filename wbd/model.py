"""Whisper encoder feature extraction and BIO classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel

from .config import NUM_CLASSES


HEAD_HIDDEN_SIZE = 256
CAUSAL_CONV_LAYERS = 3
CAUSAL_CONV_KERNEL_SIZE = 3


class LinearBIOHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_CLASSES),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class MLPBIOHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, HEAD_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HEAD_HIDDEN_SIZE, NUM_CLASSES),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class CausalConv1dBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        x = self.activation(x)
        return self.dropout(x)


class CausalConv1DBIOHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_proj = nn.Linear(hidden_size, HEAD_HIDDEN_SIZE)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv_stack = nn.Sequential(
            *[
                CausalConv1dBlock(
                    channels=HEAD_HIDDEN_SIZE,
                    kernel_size=CAUSAL_CONV_KERNEL_SIZE,
                    dropout=dropout,
                )
                for _ in range(CAUSAL_CONV_LAYERS)
            ]
        )
        self.output_proj = nn.Conv1d(HEAD_HIDDEN_SIZE, NUM_CLASSES, kernel_size=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(hidden)
        x = self.input_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.conv_stack(x)
        x = self.output_proj(x)
        return x.transpose(1, 2)


def make_bio_head(head_type: str, hidden_size: int, dropout: float) -> nn.Module:
    if head_type == "linear":
        return LinearBIOHead(hidden_size, dropout)
    if head_type == "mlp":
        return MLPBIOHead(hidden_size, dropout)
    if head_type == "causal_conv1d":
        return CausalConv1DBIOHead(hidden_size, dropout)
    raise ValueError(f"Unknown head type: {head_type}")


class WhisperWBDClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        encoder_layer: int,
        freeze_encoder: bool,
        head_type: str,
        dropout: float,
        attn_implementation: str | None,
    ):
        super().__init__()
        kwargs = {}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        whole_model = WhisperModel.from_pretrained(model_name_or_path, **kwargs)
        self.encoder = whole_model.get_encoder()
        self.encoder_layer = encoder_layer
        self.freeze_encoder = freeze_encoder
        self.head_type = head_type
        hidden_size = self.encoder.config.d_model

        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = make_bio_head(head_type, hidden_size, dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, input_features: torch.Tensor) -> torch.Tensor:
        if not (0 <= self.encoder_layer <= self.encoder.config.encoder_layers):
            raise ValueError(
                f"--encoder-layer must be in 0..{self.encoder.config.encoder_layers}, got {self.encoder_layer}"
            )
        hidden_states = F.gelu(self.encoder.conv1(input_features))
        hidden_states = F.gelu(self.encoder.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = hidden_states + self.encoder.embed_positions.weight[: hidden_states.size(1)]
        hidden_states = F.dropout(hidden_states, p=self.encoder.dropout, training=self.encoder.training)
        if self.encoder_layer == 0:
            return hidden_states

        # Layer 6 means the state after blocks 0..5, matching HF hidden_states[6].
        for idx, encoder_layer in enumerate(self.encoder.layers):
            if idx >= self.encoder_layer:
                break
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        if self.encoder_layer == self.encoder.config.encoder_layers:
            hidden_states = self.encoder.layer_norm(hidden_states)
        return hidden_states

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                hidden = self._encode(input_features)
        else:
            hidden = self._encode(input_features)
        return self.classifier(hidden)


def ensure_transformers_torch_backend() -> None:
    from transformers.utils import is_torch_available

    if is_torch_available():
        return
    raise SystemExit(
        "Transformers disabled its PyTorch backend. This usually means the installed "
        f"Transformers version requires a newer PyTorch than torch {torch.__version__}. "
        "Upgrade the environment with: pip install --upgrade -r requirements.txt"
    )
