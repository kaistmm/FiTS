"""
Feedforward spiking networks used in the paper.

    x [B, T, C] → [spiking layer → Dropout] × L → pool over T → Linear → logits

Every hidden layer consumes the full spike sequence of the layer below, so the
temporal update of each layer can run as one fused Triton call.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from fits.neuron import FiTSNeuron, LIFNeuron

__all__ = ["FiTSNet", "build_model", "NEURON_TYPES"]

NEURON_TYPES = ("fits", "lif", "adaptive_lif")
_SURROGATE_IDX = {"triangle": 0, "atan": 1, "sigmoid": 2}


class FiTSNet(nn.Module):
    """
    Args:
        in_features  : input channels C.
        hidden_sizes : widths of the hidden layers, e.g. [512, 512].
        num_classes  : number of output classes.
        dropout      : dropout on the spikes of every hidden layer.
        readout      : "sum" (SHD/SSC/GSC) or "mean" (sMNIST) pooling over time.
        neuron       : "fits", "lif" (plain LIF) or "adaptive_lif".
        **neuron_kwargs : forwarded to :class:`fits.FiTSNeuron`
                          (order, tau_m, tau_a, dt, f_min, f_max, ...).
    """

    def __init__(self,
                 in_features: int,
                 hidden_sizes: list,
                 num_classes: int,
                 dropout: float = 0.0,
                 readout: str = "sum",
                 neuron: str = "fits",
                 **neuron_kwargs):
        super().__init__()
        if readout not in ("sum", "mean"):
            raise ValueError(f"readout must be 'sum' or 'mean', got {readout!r}")
        if neuron not in NEURON_TYPES:
            raise ValueError(f"neuron must be one of {NEURON_TYPES}, got {neuron!r}")

        self.in_features  = in_features
        self.hidden_sizes = list(hidden_sizes)
        self.num_classes  = num_classes
        self.readout      = readout
        self.neuron       = neuron

        neuron_layers, dropout_layers = [], []
        prev = in_features
        for hid in self.hidden_sizes:
            if neuron == "fits":
                layer = FiTSNeuron(prev, hid, **neuron_kwargs)
            else:
                layer = LIFNeuron(prev, hid, adaptive=(neuron == "adaptive_lif"), **neuron_kwargs)
            neuron_layers.append(layer)
            dropout_layers.append(nn.Dropout(p=dropout))
            prev = hid

        self.neuron_layers  = nn.ModuleList(neuron_layers)
        self.dropout_layers = nn.ModuleList(dropout_layers)
        self.output         = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_features] → logits [B, num_classes]."""
        out = x
        for layer, drop in zip(self.neuron_layers, self.dropout_layers):
            spikes, *_ = layer(out)
            out = drop(spikes)
        pooled = out.sum(dim=1) if self.readout == "sum" else out.mean(dim=1)
        return self.output(pooled)

    def get_frequencies(self) -> dict:
        """Learned target frequencies f* (Hz) of every hidden layer."""
        return {i: layer.get_freq().detach().clone()
                for i, layer in enumerate(self.neuron_layers)}

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, hidden={self.hidden_sizes}, "
                f"out={self.num_classes}, neuron={self.neuron}, readout={self.readout}")


def build_model(cfg, in_features: int, num_classes: int, readout: str = "sum") -> FiTSNet:
    """Build a :class:`FiTSNet` from a config namespace or dict (see ``config/``)."""
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)
    g = lambda key, default: getattr(cfg, key, default)
    return FiTSNet(
        in_features     = in_features,
        hidden_sizes    = list(cfg.hidden_dims),
        num_classes     = num_classes,
        dropout         = g("dropout", 0.0),
        readout         = readout,
        neuron          = g("neuron_type", "fits").lower(),
        order           = g("order", 1),
        chunk_size      = g("chunk_size", 32),
        tau_m           = cfg.tau_m,
        tau_a           = cfg.tau_a,
        dt              = cfg.dt,
        ratio_eta_gamma = g("ratio_eta_gamma", 1.0),
        f_min           = cfg.f_min,
        f_max           = cfg.f_max,
        v_threshold     = g("threshold", 1.0),
        surrogate_alpha = g("surrogate_alpha", 2.0),
        surrogate_type  = _SURROGATE_IDX[g("surrogate", "triangle")],
        lam_init        = g("lam_init", -3.0),
        implicit        = g("implicit", True),
        freq_init       = g("freq_init", "log"),
        learn_freq      = g("learn_freq", True),
        freq_param      = g("freq_param", "ct"),
        backend         = g("backend", "auto"),
    )
