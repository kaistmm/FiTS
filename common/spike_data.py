"""
SHD / SSC spike-train data (Cramer et al., 2020; https://zenkelab.org/datasets/).

Each HDF5 file stores spike times (s) and unit ids (0..699) per sample.  As in
the paper, the 700 channels are binned in groups of five (140 inputs) and the
first 1.0 s is binned into T = 250 steps of Δt = 4 ms.
"""

import os

import h5py
import numpy as np
import torch

ORIG_UNITS = 700
MAX_TIME = 1.0     # seconds; spikes after 1.0 s are dropped


class SpikeTrainDataset(torch.utils.data.Dataset):
    """Pre-binned (time_index, channel_index) events per sample."""

    def __init__(self, times, units, labels, nb_steps: int, in_features: int):
        self.labels = np.array(labels, dtype=np.int64)
        scale = in_features / ORIG_UNITS
        self.data = []
        for t, u in zip(times, units):
            t_idx = np.floor(t / MAX_TIME * nb_steps).astype(np.int64)
            u_idx = np.floor(u * scale).astype(np.int64)
            keep = (t_idx >= 0) & (t_idx < nb_steps) & (u_idx >= 0) & (u_idx < in_features)
            self.data.append((t_idx[keep], u_idx[keep]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.labels[idx]


def make_collate(nb_steps: int, in_features: int):
    """Collate events into a dense binary tensor [B, T, in_features]."""
    def collate_fn(batch):
        X = torch.zeros(len(batch), nb_steps, in_features)
        y = torch.zeros(len(batch), dtype=torch.long)
        for i, (ts, us, lbl) in enumerate(batch):
            if len(ts):
                X[i, torch.as_tensor(ts, dtype=torch.long),
                     torch.as_tensor(us, dtype=torch.long)] = 1.0
            y[i] = lbl
        return X, y
    return collate_fn


def read_h5(path: str):
    """Return (times, units, labels) arrays of one SHD/SSC HDF5 file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Download SHD/SSC from https://zenkelab.org/datasets/ "
            "and point --data-dir to the directory containing the .h5 files.")
    with h5py.File(path, "r") as f:
        return f["spikes"]["times"][:], f["spikes"]["units"][:], f["labels"][:]


def make_loader(dataset, batch_size, shuffle, workers, nb_steps, in_features):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=make_collate(nb_steps, in_features),
        pin_memory=torch.cuda.is_available(), num_workers=workers,
        persistent_workers=workers > 0)
