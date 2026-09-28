"""
sMNIST data — Data loading for sMNIST and psMNIST benchmarks.

Both tasks use the same MNIST pixel data (28×28=784 timesteps, in_features=1).
psMNIST applies a fixed random permutation to the temporal sequence, generated
deterministically from the training seed to ensure reproducibility.

Usage:
    from MNIST.core.data import make_loaders, make_permutation

    # sMNIST
    tr_ld, te_ld = make_loaders('smnist', data_dir=..., batch_size=128)

    # psMNIST (perm saved in config for reproducibility)
    perm = make_permutation(seed=0)
    tr_ld, te_ld = make_loaders('psmnist', data_dir=..., batch_size=128, perm=perm)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# Permutation
# ─────────────────────────────────────────────────────────────────────────────

def make_permutation(seed: int) -> Tensor:
    """
    Generate a deterministic permutation of 784 indices from a given seed.

    The permutation is intentionally separate from the global RNG state so
    that it is always reproducible regardless of training state.

    Args:
        seed: Integer seed used exclusively for generating the permutation.

    Returns:
        LongTensor of shape [784] — the fixed pixel permutation.
    """
    gen = torch.Generator()
    gen.manual_seed(seed + 98765)   # salted seed to avoid collisions with global seed
    return torch.randperm(784, generator=gen)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MNISTSequence(Dataset):
    """
    Wraps torchvision MNIST and exposes each image as a length-784 sequence.

    Each sample is returned as:
        x : FloatTensor [784, 1]  (pixel values ∈ [0, 1])
        y : int label

    For psMNIST, pass a permutation tensor; indices are applied at __getitem__
    so the DataLoader works the same way for both tasks.

    Args:
        root     : Root directory for MNIST data.
        train    : True for training set, False for test set.
        download : Download if not found.
        perm     : Optional [784] LongTensor permutation for psMNIST.
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = True,
                 perm: Optional[Tensor] = None):
        self._ds = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),   # [1, 28, 28]
        )
        self._perm = perm   # None → sMNIST, Tensor[784] → psMNIST

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        img, label = self._ds[idx]          # img: [1, 28, 28]
        x = img.view(784, 1)                 # [784, 1]
        if self._perm is not None:
            x = x[self._perm]               # apply permutation along time axis
        return x, label


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(
    task: str,
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 2,
    perm: Optional[Tensor] = None,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for sMNIST or psMNIST.

    No validation split is applied — training uses the full 60,000-sample
    MNIST train set.  Test accuracy on the 10,000-sample test set is used
    directly for model selection (best.pth).

    Args:
        task        : 'smnist' or 'psmnist'.
        data_dir    : Root directory where MNIST is (or will be) downloaded.
        batch_size  : Mini-batch size.
        num_workers : DataLoader worker count.
        perm        : Permutation tensor for psMNIST (ignored for smnist).
                      Use make_permutation(seed) to generate reproducibly.
        download    : Whether to auto-download if not present.

    Returns:
        (train_loader, test_loader)
    """
    task = task.lower()
    if task not in ('smnist', 'psmnist'):
        raise ValueError(f"task must be 'smnist' or 'psmnist', got {task!r}")

    active_perm = perm if task == 'psmnist' else None

    tr_ds = MNISTSequence(data_dir, train=True,  download=download, perm=active_perm)
    te_ds = MNISTSequence(data_dir, train=False, download=download, perm=active_perm)

    dl_kw: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        dl_kw['persistent_workers'] = True
        dl_kw['prefetch_factor']    = 2

    tr_ld = DataLoader(tr_ds, shuffle=True,  **dl_kw)
    te_ld = DataLoader(te_ds, shuffle=False, **dl_kw)

    return tr_ld, te_ld
