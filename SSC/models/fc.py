"""Fully connected FiTS network for SSC."""

from fits.models import FiTSNet, build_model as _build_model

__all__ = ["FiTSNet", "build_model"]


def build_model(cfg):
    """Build the 35-class SSC network with sum pooling over time."""
    return _build_model(cfg, in_features=cfg.in_features,
                        num_classes=35, readout="sum")
