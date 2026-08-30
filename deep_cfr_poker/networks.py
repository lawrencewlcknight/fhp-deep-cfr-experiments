"""Neural network components used by the Deep CFR solver.

These layers preserve the truncated-normal initialisation of the original
notebook (a Sonnet-style linear layer with stddev = 1/sqrt(fan_in)) so that
results stay comparable across the notebook and the packaged code.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn


class SonnetLinear(nn.Module):
    """A Sonnet-style linear layer with optional ReLU activation."""

    def __init__(self, in_size: int, out_size: int, activate_relu: bool = True) -> None:
        super().__init__()
        self._activate_relu = bool(activate_relu)
        self._in_size = int(in_size)
        self._out_size = int(out_size)
        # Allocated by ``reset``; declared up front so static analysers are happy.
        self._weight: nn.Parameter = nn.Parameter(
            torch.empty(self._out_size, self._in_size), requires_grad=True
        )
        self._bias: nn.Parameter = nn.Parameter(
            torch.zeros(self._out_size), requires_grad=True
        )
        self.reset()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        y = F.linear(tensor, self._weight, self._bias)
        return F.relu(y) if self._activate_relu else y

    def reset(self) -> None:
        stddev = 1.0 / math.sqrt(self._in_size)
        # Truncated to ±2 standard deviations.
        sampled = stats.truncnorm.rvs(
            -2.0, 2.0, loc=0.0, scale=stddev, size=(self._out_size, self._in_size)
        )
        with torch.no_grad():
            self._weight.copy_(torch.from_numpy(sampled).to(self._weight.dtype))
            self._bias.zero_()


class MLP(nn.Module):
    """A simple feed-forward MLP with ``reset`` propagated to every layer."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activate_final: bool = False,
    ) -> None:
        super().__init__()
        layers: List[SonnetLinear] = []
        in_size = int(input_size)
        for size in hidden_sizes:
            layers.append(SonnetLinear(in_size=in_size, out_size=int(size)))
            in_size = int(size)
        layers.append(
            SonnetLinear(
                in_size=in_size,
                out_size=int(output_size),
                activate_relu=bool(activate_final),
            )
        )
        self.model = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        return x

    def reset(self) -> None:
        for layer in self.model:
            layer.reset()


class LayerNormMLP(nn.Module):
    """Feed-forward MLP with layer normalisation after each hidden layer."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activate_final: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        in_size = int(input_size)
        for size in hidden_sizes:
            size = int(size)
            self.hidden_layers.append(SonnetLinear(in_size=in_size, out_size=size))
            self.hidden_norms.append(nn.LayerNorm(size))
            in_size = size
        self.output_layer = SonnetLinear(
            in_size=in_size,
            out_size=int(output_size),
            activate_relu=bool(activate_final),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            x = norm(layer(x))
        return self.output_layer(x)

    def reset(self) -> None:
        for layer in self.hidden_layers:
            layer.reset()
        for norm in self.hidden_norms:
            norm.reset_parameters()
        self.output_layer.reset()


class DropoutMLP(nn.Module):
    """Feed-forward MLP with dropout after each hidden layer."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        dropout_probability: float,
        activate_final: bool = False,
    ) -> None:
        super().__init__()
        self.dropout_probability = float(dropout_probability)
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_size = int(input_size)
        for size in hidden_sizes:
            size = int(size)
            self.hidden_layers.append(SonnetLinear(in_size=in_size, out_size=size))
            self.dropouts.append(nn.Dropout(p=self.dropout_probability))
            in_size = size
        self.output_layer = SonnetLinear(
            in_size=in_size,
            out_size=int(output_size),
            activate_relu=bool(activate_final),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, dropout in zip(self.hidden_layers, self.dropouts):
            x = dropout(layer(x))
        return self.output_layer(x)

    def reset(self) -> None:
        for layer in self.hidden_layers:
            layer.reset()
        self.output_layer.reset()


class ResidualHiddenLayer(nn.Module):
    """One hidden layer with an optional same-width residual connection."""

    def __init__(self, in_size: int, out_size: int, *, layer_norm: bool = False) -> None:
        super().__init__()
        self.layer = SonnetLinear(in_size=in_size, out_size=out_size)
        self.use_residual = int(in_size) == int(out_size)
        self.norm = nn.LayerNorm(int(out_size)) if layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer(x)
        if self.use_residual:
            y = y + x
        if self.norm is not None:
            y = self.norm(y)
        return y

    def reset(self) -> None:
        self.layer.reset()
        if self.norm is not None:
            self.norm.reset_parameters()


class ResidualMLP(nn.Module):
    """MLP whose hidden layers after the first use same-width skip connections."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activate_final: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        in_size = int(input_size)
        for size in hidden_sizes:
            size = int(size)
            self.hidden_layers.append(
                ResidualHiddenLayer(in_size=in_size, out_size=size, layer_norm=layer_norm)
            )
            in_size = size
        self.output_layer = SonnetLinear(
            in_size=in_size,
            out_size=int(output_size),
            activate_relu=bool(activate_final),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def reset(self) -> None:
        for layer in self.hidden_layers:
            layer.reset()
        self.output_layer.reset()


class CenteredAdvantageMLP(nn.Module):
    """MLP head whose action outputs are centred to zero mean per state."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activate_final: bool = False,
    ) -> None:
        super().__init__()
        del activate_final
        self.hidden_layers = nn.ModuleList()
        in_size = int(input_size)
        for size in hidden_sizes:
            size = int(size)
            self.hidden_layers.append(SonnetLinear(in_size=in_size, out_size=size))
            in_size = size
        self.action_head = SonnetLinear(
            in_size=in_size,
            out_size=int(output_size),
            activate_relu=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        raw_advantages = self.action_head(x)
        return raw_advantages - raw_advantages.mean(dim=-1, keepdim=True)

    def reset(self) -> None:
        for layer in self.hidden_layers:
            layer.reset()
        self.action_head.reset()


class ResidualLayerNormCenteredAdvantageMLP(nn.Module):
    """Residual LayerNorm trunk with centred action-advantage outputs."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activate_final: bool = False,
    ) -> None:
        super().__init__()
        del activate_final
        self.hidden_layers = nn.ModuleList()
        in_size = int(input_size)
        for size in hidden_sizes:
            size = int(size)
            self.hidden_layers.append(
                ResidualHiddenLayer(in_size=in_size, out_size=size, layer_norm=True)
            )
            in_size = size
        self.action_head = SonnetLinear(
            in_size=in_size,
            out_size=int(output_size),
            activate_relu=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        raw_advantages = self.action_head(x)
        return raw_advantages - raw_advantages.mean(dim=-1, keepdim=True)

    def reset(self) -> None:
        for layer in self.hidden_layers:
            layer.reset()
        self.action_head.reset()


class DuelingMLP(nn.Module):
    """MLP with a scalar state head plus centred action-advantage head."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activate_final: bool = False,
    ) -> None:
        super().__init__()
        del activate_final
        self.hidden_layers = nn.ModuleList()
        in_size = int(input_size)
        for size in hidden_sizes:
            size = int(size)
            self.hidden_layers.append(SonnetLinear(in_size=in_size, out_size=size))
            in_size = size
        self.value_head = SonnetLinear(
            in_size=in_size,
            out_size=1,
            activate_relu=False,
        )
        self.action_head = SonnetLinear(
            in_size=in_size,
            out_size=int(output_size),
            activate_relu=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        state_value = self.value_head(x)
        raw_advantages = self.action_head(x)
        centred_advantages = raw_advantages - raw_advantages.mean(dim=-1, keepdim=True)
        return state_value + centred_advantages

    def reset(self) -> None:
        for layer in self.hidden_layers:
            layer.reset()
        self.value_head.reset()
        self.action_head.reset()


class SharedTrunk(nn.Module):
    """Shared hidden representation used by per-player output heads."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("SharedTrunk requires at least one hidden layer")
        self.layers = nn.ModuleList()
        in_size = int(input_size)
        for size in hidden_sizes:
            size = int(size)
            self.layers.append(SonnetLinear(in_size=in_size, out_size=size))
            in_size = size
        self.output_size = in_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


class PlayerActionHead(nn.Module):
    """Per-player action head backed by a shared representation trunk."""

    def __init__(self, trunk: SharedTrunk, output_size: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = SonnetLinear(
            in_size=trunk.output_size,
            out_size=int(output_size),
            activate_relu=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def reset(self) -> None:
        self.trunk.reset()
        self.head.reset()

    def reset_head(self) -> None:
        self.head.reset()


def build_shared_trunk_player_heads(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    num_players: int,
) -> list[PlayerActionHead]:
    """Builds one shared trunk with a distinct action-output head per player."""
    trunk = SharedTrunk(input_size=input_size, hidden_sizes=hidden_sizes)
    return [PlayerActionHead(trunk, output_size) for _ in range(int(num_players))]


def build_network(
    network_type: str,
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    activate_final: bool = False,
) -> nn.Module:
    """Constructs a supported MLP variant.

    ``"mlp"`` is the historical baseline and remains the default. Other
    variants are opt-in so existing experiments keep byte-for-byte equivalent
    model structure unless their config explicitly requests a new type.
    """
    network_type = str(network_type).lower()
    if network_type == "mlp":
        return MLP(input_size, hidden_sizes, output_size, activate_final)
    dropout_variants = {
        "dropout_mlp_p05": 0.05,
        "dropout_mlp_p10": 0.10,
        "dropout_mlp_p20": 0.20,
    }
    if network_type in dropout_variants:
        return DropoutMLP(
            input_size,
            hidden_sizes,
            output_size,
            dropout_variants[network_type],
            activate_final,
        )
    if network_type == "layer_norm_mlp":
        return LayerNormMLP(input_size, hidden_sizes, output_size, activate_final)
    if network_type == "residual_mlp":
        return ResidualMLP(
            input_size,
            hidden_sizes,
            output_size,
            activate_final,
            layer_norm=False,
        )
    if network_type == "residual_layer_norm_mlp":
        return ResidualMLP(
            input_size,
            hidden_sizes,
            output_size,
            activate_final,
            layer_norm=True,
        )
    if network_type == "centered_advantage_mlp":
        return CenteredAdvantageMLP(
            input_size,
            hidden_sizes,
            output_size,
            activate_final,
        )
    if network_type == "residual_layer_norm_centered_advantage_mlp":
        return ResidualLayerNormCenteredAdvantageMLP(
            input_size,
            hidden_sizes,
            output_size,
            activate_final,
        )
    if network_type == "dueling_mlp":
        return DuelingMLP(
            input_size,
            hidden_sizes,
            output_size,
            activate_final,
        )
    valid = (
        "mlp",
        "dropout_mlp_p05",
        "dropout_mlp_p10",
        "dropout_mlp_p20",
        "layer_norm_mlp",
        "residual_mlp",
        "residual_layer_norm_mlp",
        "centered_advantage_mlp",
        "residual_layer_norm_centered_advantage_mlp",
        "dueling_mlp",
    )
    raise ValueError(f"Unknown network_type={network_type!r}. Expected one of {valid}.")
