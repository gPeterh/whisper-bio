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
DYNAMIC_FUSION_HIDDEN_SIZE = 128
LAYER_FUSION_SINGLE = "single"
LAYER_FUSION_WEIGHTED_SUM = "learnable_weighted_sum"
LAYER_FUSION_DYNAMIC_WEIGHTED_SUM = "dynamic_weighted_sum"
LAYER_FUSION_AUTO = "auto"
LAYER_FUSION_CHOICES = (
    LAYER_FUSION_AUTO,
    LAYER_FUSION_SINGLE,
    LAYER_FUSION_WEIGHTED_SUM,
    LAYER_FUSION_DYNAMIC_WEIGHTED_SUM,
)


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
        causal_encoder: bool = False,
        encoder_lookahead_frames: int = 0,
        encoder_layers: str | list[int] | None = None,
        layer_fusion: str | None = None,
    ):
        super().__init__()
        kwargs = {}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        whole_model = WhisperModel.from_pretrained(model_name_or_path, **kwargs)
        self.encoder = whole_model.get_encoder()
        self.encoder_layer = encoder_layer
        self.encoder_layers = self._resolve_encoder_layers(encoder_layer, encoder_layers)
        self.freeze_encoder = freeze_encoder
        self.head_type = head_type
        self.causal_encoder = causal_encoder
        if encoder_lookahead_frames < 0:
            raise ValueError("encoder_lookahead_frames must be >= 0.")
        self.encoder_lookahead_frames = int(encoder_lookahead_frames)
        self.layer_fusion = self._resolve_layer_fusion(layer_fusion)
        hidden_size = self.encoder.config.d_model

        if self.layer_fusion == LAYER_FUSION_WEIGHTED_SUM:
            self.layer_fusion_logits = nn.Parameter(torch.zeros(len(self.encoder_layers)))
        elif self.layer_fusion == LAYER_FUSION_DYNAMIC_WEIGHTED_SUM:
            self.layer_fusion_gate = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, DYNAMIC_FUSION_HIDDEN_SIZE),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(DYNAMIC_FUSION_HIDDEN_SIZE, len(self.encoder_layers)),
            )

        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = make_bio_head(head_type, hidden_size, dropout)

    def _resolve_layer_fusion(self, layer_fusion: str | None) -> str:
        if layer_fusion is None or layer_fusion == "" or layer_fusion == LAYER_FUSION_AUTO:
            return LAYER_FUSION_WEIGHTED_SUM if len(self.encoder_layers) > 1 else LAYER_FUSION_SINGLE
        if layer_fusion not in LAYER_FUSION_CHOICES:
            raise ValueError(f"Unknown layer fusion: {layer_fusion}")
        if layer_fusion == LAYER_FUSION_SINGLE and len(self.encoder_layers) != 1:
            raise ValueError("single layer fusion requires exactly one encoder layer.")
        if layer_fusion in (LAYER_FUSION_WEIGHTED_SUM, LAYER_FUSION_DYNAMIC_WEIGHTED_SUM):
            if len(self.encoder_layers) < 2:
                raise ValueError(f"{layer_fusion} requires at least two encoder layers.")
        return layer_fusion

    def _resolve_encoder_layers(self, encoder_layer: int, encoder_layers: str | list[int] | None) -> list[int]:
        max_layers = self.encoder.config.encoder_layers
        if encoder_layers is None or encoder_layers == "":
            layers = [encoder_layer]
        elif isinstance(encoder_layers, str):
            spec = encoder_layers.strip().lower()
            if spec == "all":
                layers = list(range(1, max_layers + 1))
            else:
                layers = [int(part.strip()) for part in encoder_layers.split(",") if part.strip()]
        else:
            layers = [int(layer) for layer in encoder_layers]

        if not layers:
            raise ValueError("--encoder-layers resolved to an empty layer list.")
        invalid = [layer for layer in layers if layer < 0 or layer > max_layers]
        if invalid:
            raise ValueError(f"Encoder layers must be in 0..{max_layers}, got {invalid}")
        return sorted(set(layers))

    @staticmethod
    def _causal_attention_mask(hidden_states: torch.Tensor, lookahead_frames: int = 0) -> torch.Tensor:
        seq_len = hidden_states.size(1)
        mask = torch.zeros((1, 1, seq_len, seq_len), device=hidden_states.device, dtype=hidden_states.dtype)
        future_positions = torch.triu(
            torch.ones((seq_len, seq_len), device=hidden_states.device, dtype=torch.bool),
            diagonal=1 + lookahead_frames,
        )
        return mask.masked_fill(future_positions.view(1, 1, seq_len, seq_len), torch.finfo(hidden_states.dtype).min)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("encoder.")
        }

    def _encode_selected_layers(self, input_features: torch.Tensor) -> list[torch.Tensor]:
        selected = set(self.encoder_layers)
        max_selected_layer = max(selected)
        selected_hidden_states: dict[int, torch.Tensor] = {}

        hidden_states = F.gelu(self.encoder.conv1(input_features))
        hidden_states = F.gelu(self.encoder.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = hidden_states + self.encoder.embed_positions.weight[: hidden_states.size(1)]
        hidden_states = F.dropout(hidden_states, p=self.encoder.dropout, training=self.encoder.training)
        if 0 in selected:
            selected_hidden_states[0] = hidden_states
        if max_selected_layer == 0:
            return [selected_hidden_states[layer] for layer in self.encoder_layers]

        attention_mask = (
            self._causal_attention_mask(hidden_states, self.encoder_lookahead_frames)
            if self.causal_encoder
            else None
        )

        # Layer N means the state after blocks 0..N-1, matching HF hidden_states[N].
        for idx, encoder_layer in enumerate(self.encoder.layers):
            if idx >= max_selected_layer:
                break
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            layer_index = idx + 1
            if layer_index == self.encoder.config.encoder_layers:
                hidden_states = self.encoder.layer_norm(hidden_states)
            if layer_index in selected:
                selected_hidden_states[layer_index] = hidden_states

        return [selected_hidden_states[layer] for layer in self.encoder_layers]

    def _fuse_selected_layers(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        if self.layer_fusion == LAYER_FUSION_SINGLE:
            return hidden_states[0]
        if self.layer_fusion == LAYER_FUSION_WEIGHTED_SUM:
            weights = torch.softmax(self.layer_fusion_logits, dim=0).to(
                device=hidden_states[0].device,
                dtype=hidden_states[0].dtype,
            )
            fused = hidden_states[0] * weights[0]
            for weight, hidden in zip(weights[1:], hidden_states[1:]):
                fused = fused + hidden * weight
            return fused
        if self.layer_fusion == LAYER_FUSION_DYNAMIC_WEIGHTED_SUM:
            gate_input = hidden_states[-1]
            weights = torch.softmax(self.layer_fusion_gate(gate_input), dim=-1).to(dtype=hidden_states[0].dtype)
            stacked_hidden = torch.stack(hidden_states, dim=2)
            return (stacked_hidden * weights.unsqueeze(-1)).sum(dim=2)
        raise ValueError(f"Unknown layer fusion: {self.layer_fusion}")

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                hidden_states = self._encode_selected_layers(input_features)
        else:
            hidden_states = self._encode_selected_layers(input_features)
        hidden = self._fuse_selected_layers(hidden_states)
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
