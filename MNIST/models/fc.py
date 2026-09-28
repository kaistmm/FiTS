"""Fully connected FiTS network for MNIST."""

from fits.models import FiTSNet, build_model as _build_model

__all__ = ["FiTSNet", "build_model"]


def build_model(cfg):
    """Build the 10-class MNIST network with mean pooling over time."""
    return _build_model(cfg, in_features=1,
                        num_classes=10, readout="mean")
