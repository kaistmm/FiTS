"""Fully connected FiTS network for SHD."""

from fits.models import FiTSNet, build_model as _build_model

__all__ = ["FiTSNet", "build_model"]


def build_model(cfg):
    """Build the 20-class SHD network with sum pooling over time."""
    return _build_model(cfg, in_features=cfg.in_features,
                        num_classes=20, readout="sum")
