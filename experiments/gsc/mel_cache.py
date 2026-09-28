"""
Disk cache for GSC mel-spectrogram tensors (torch.save .pt per wav).

Layout under ``mel_cache_dir``::

    <mel_cache_dir>/<fp_key>/<split>/<class>/<id>.pt

``fp_key`` encodes ``mel_max_len``, ``audio_sr``, and ``normalize`` so changing
preprocessing uses a separate subdirectory without clobbering old caches.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import torch

from experiments.gsc.data import GSC_V02_CLASS_TO_IDX, _load_spectrogram, _make_dataset

log = logging.getLogger(__name__)


def mel_fp_key(mel_max_len: int, audio_sr: int, normalize: bool) -> str:
    """Subdirectory name for one preprocessing configuration."""
    return f"L{int(mel_max_len)}_sr{int(audio_sr)}_norm{int(bool(normalize))}"


def tensor_path_for_wav(
    split_root: str,
    mel_cache_dir: str,
    fp_key: str,
    wav_path: str,
) -> str:
    """Map a wav under ``split_root`` to its cached ``.pt`` path."""
    split_root = os.path.normpath(os.path.expanduser(split_root))
    mel_cache_dir = os.path.expanduser(mel_cache_dir)
    split_name = os.path.basename(split_root)
    rel = os.path.relpath(wav_path, split_root)
    rel_dir, base = os.path.split(rel)
    stem, ext = os.path.splitext(base)
    if ext.lower() not in (".wav",):
        raise ValueError(f"Expected .wav path, got: {wav_path}")
    pt_name = stem + ".pt"
    if rel_dir:
        sub = os.path.join(rel_dir, pt_name)
    else:
        sub = pt_name
    return os.path.join(mel_cache_dir, fp_key, split_name, sub)


def load_mel_tensor(path: str) -> torch.Tensor:
    """Load a single cached mel tensor from ``.pt``."""
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        t = torch.load(path, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected Tensor in {path}, got {type(t)}")
    return t


def ensure_mel_tensor_cache(
    split_root: str,
    mel_cache_dir: str,
    normalize: bool,
    mel_max_len: int,
    audio_sr: int,
) -> List[Tuple[str, int]]:
    """
    Ensure every wav under ``split_root`` has a ``.pt`` mel under ``mel_cache_dir``.

    Returns ``[(pt_path, label), ...]`` in the same order as ``_make_dataset``.
    """
    split_root = os.path.normpath(os.path.expanduser(split_root))
    mel_cache_dir = os.path.expanduser(mel_cache_dir)
    fp_key = mel_fp_key(mel_max_len, audio_sr, normalize)
    samples = _make_dataset(split_root, GSC_V02_CLASS_TO_IDX)
    if not samples:
        raise RuntimeError(f"No wav samples under {split_root}")

    n = len(samples)
    built = 0
    skipped = 0
    split_name = os.path.basename(split_root)
    log.info(
        "[GSC/mel-cache] Ensuring %d files for split=%s fp=%s under %s",
        n,
        split_name,
        fp_key,
        mel_cache_dir,
    )

    for i, (wav_path, label) in enumerate(samples):
        pt_path = tensor_path_for_wav(split_root, mel_cache_dir, fp_key, wav_path)
        if os.path.isfile(pt_path):
            skipped += 1
        else:
            spect = _load_spectrogram(wav_path, normalize, mel_max_len, audio_sr)
            os.makedirs(os.path.dirname(pt_path), exist_ok=True)
            torch.save(spect, pt_path)
            built += 1
        if (i + 1) % 10000 == 0 or (i + 1) == n:
            log.info(
                "[GSC/mel-cache]   %s: %d / %d  (new=%d, existing=%d)",
                split_name,
                i + 1,
                n,
                built,
                skipped,
            )

    log.info(
        "[GSC/mel-cache] Split %s finished: wrote %d new tensors, %d already present.",
        split_name,
        built,
        skipped,
    )
    return [
        (tensor_path_for_wav(split_root, mel_cache_dir, fp_key, w), lbl)
        for w, lbl in samples
    ]


def list_mel_tensor_paths(
    split_root: str,
    mel_cache_dir: str,
    normalize: bool,
    mel_max_len: int,
    audio_sr: int,
) -> List[Tuple[str, int]]:
    """Like ``ensure_mel_tensor_cache`` but do not create files; raise if any missing."""
    split_root = os.path.normpath(os.path.expanduser(split_root))
    mel_cache_dir = os.path.expanduser(mel_cache_dir)
    fp_key = mel_fp_key(mel_max_len, audio_sr, normalize)
    samples = _make_dataset(split_root, GSC_V02_CLASS_TO_IDX)
    out: List[Tuple[str, int]] = []
    for wav_path, label in samples:
        pt_path = tensor_path_for_wav(split_root, mel_cache_dir, fp_key, wav_path)
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(
                f"Missing mel tensor cache: {pt_path}\n"
                f"Run training with ensure_mel_cache=true or build the cache first."
            )
        out.append((pt_path, label))
    return out
