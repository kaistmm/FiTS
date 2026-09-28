"""Shared FiTS and LIF neurons used by this benchmark.

The implementations live in fits/ so all datasets use the same dynamics.
"""

from fits.neuron import FiTSNeuron, LIFNeuron
from fits.stability import FrequencyGuard

__all__ = ["FiTSNeuron", "LIFNeuron", "FrequencyGuard"]
