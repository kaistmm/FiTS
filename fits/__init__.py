"""
FiTS: Interpretable Spiking Neurons via Frequency Selectivity and Temporal Shaping
(NeurIPS 2026).

    from fits import FiTSNeuron

    layer = FiTSNeuron(in_features=140, out_features=256, order=1).cuda()
    spikes, *_ = layer(x)          # x: [B, T, 140] -> spikes: [B, T, 256]
    layer.get_freq()               # learned target frequency of every neuron (Hz)
"""

from fits.frequency import (critical_freq, ct_peak_freq, dt_peak_freq,
                            eta_gamma_from_kappa, jury_bound, kappa_ct,
                            kappa_dt, kappa_from_freq)
from fits.models import FiTSNet, build_model
from fits.neuron import FiTSNeuron, LIFNeuron, init_log_freq
from fits.stability import FrequencyGuard

__version__ = "1.0.0"

__all__ = [
    "FiTSNeuron",
    "LIFNeuron",
    "FiTSNet",
    "build_model",
    "FrequencyGuard",
    "init_log_freq",
    "kappa_ct",
    "kappa_dt",
    "kappa_from_freq",
    "ct_peak_freq",
    "dt_peak_freq",
    "eta_gamma_from_kappa",
    "jury_bound",
    "critical_freq",
]
