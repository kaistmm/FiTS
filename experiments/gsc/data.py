"""
GSC data — Google Speech Commands v0.02, 35-class mel-spectrogram loader.

Audio is resampled to 16 kHz and converted to 40-band mel-spectrograms
(n_fft=480, hop_length=160 → 10 ms per frame, fmin=40 Hz, fmax=4000 Hz),
yielding tensors of shape [1, 40, T] with T == mel_max_len (default 100).

Dataset:
    - 35 command classes (alphabetical label order, ``GSC_V02_CORE35``)
    - Input shape: [1, 40, T] per sample
    - Expected layout under ``<data_dir>/``:
        train/<class>/*.wav
        valid/<class>/*.wav
        test/<class>/*.wav

    Optional: set ``mel_cache_dir`` to use ``experiments.gsc.mel_cache`` —
    precomputed ``.pt`` tensors (created on demand when ``ensure_mel_cache``).

Use ``experiments.gsc.prepare.ensure_gsc_prepared`` to download v0.02 and build
splits under ``<data_dir>/`` when missing.
"""

import logging
import os

import numpy as np
import librosa
import torch
import torch.utils.data as data

AUDIO_EXTENSIONS = (".wav", ".WAV")

# Google Speech Commands v0.02 — 35 core words (label index = sort order).
GSC_V02_CORE35 = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
)

GSC_V02_CLASS_TO_IDX = {w: i for i, w in enumerate(GSC_V02_CORE35)}
N_CLASS = len(GSC_V02_CORE35)


def _is_audio_file(filename: str) -> bool:
    return filename.lower().endswith(".wav")


def _make_dataset(root: str, class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    root = os.path.expanduser(root)
    for target in sorted(os.listdir(root)):
        if target not in class_to_idx:
            continue
        label = class_to_idx[target]
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        for dirpath, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if _is_audio_file(fname):
                    path = os.path.join(dirpath, fname)
                    samples.append((path, label))
    return samples


def _load_spectrogram(
    path: str,
    normalize: bool = True,
    max_len: int = 100,
    sr: int = 16000,
) -> torch.Tensor:
    y, _ = librosa.load(path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=480,
        hop_length=160,
        power=1.0,
        n_mels=40,
        fmin=40.0,
        fmax=4000.0,
    )
    spect = librosa.power_to_db(mel, ref=np.max)

    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    else:
        spect = spect[:, :max_len]

    spect = torch.FloatTensor(spect).unsqueeze(0)

    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std > 0:
            spect = (spect - mean) / std

    return spect


class GCommandLoader(data.Dataset):
    """
    PyTorch Dataset for Google Speech Commands v0.02 mel-spectrograms (35-way).

    Returns (spectrogram, label) pairs where spectrogram has shape [1, 40, T].
    """

    def __init__(
        self,
        root: str,
        transform=None,
        normalize: bool = True,
        mel_max_len: int = 100,
        audio_sr: int = 16000,
        cache_mels: bool = False,
        mel_cache_dir: str | None = None,
        ensure_mel_cache: bool = True,
    ):
        self.class_to_idx = GSC_V02_CLASS_TO_IDX
        self.root = root
        self.transform = transform
        self.normalize = normalize
        self.mel_max_len = mel_max_len
        self.audio_sr = audio_sr
        self._cached_spects: list[torch.Tensor] | None = None
        self._cached_labels: list[int] | None = None
        self._from_pt = False
        self._load_mel_tensor = None

        if mel_cache_dir and str(mel_cache_dir).strip():
            from experiments.gsc.mel_cache import (
                ensure_mel_tensor_cache,
                list_mel_tensor_paths,
                load_mel_tensor,
            )

            self._load_mel_tensor = load_mel_tensor  # bound for RAM preload
            mdir = os.path.expanduser(str(mel_cache_dir).strip())
            if ensure_mel_cache:
                self.samples = ensure_mel_tensor_cache(
                    root,
                    mdir,
                    self.normalize,
                    self.mel_max_len,
                    self.audio_sr,
                )
            else:
                self.samples = list_mel_tensor_paths(
                    root,
                    mdir,
                    self.normalize,
                    self.mel_max_len,
                    self.audio_sr,
                )
            self._from_pt = True
        else:
            self.samples = _make_dataset(root, self.class_to_idx)

        if not self.samples:
            raise RuntimeError(
                f"No audio files found for the 35 GSC v0.02 classes under {root}. "
                "Run ensure_gsc_prepared() or check your processed split."
            )

        if cache_mels:
            log = logging.getLogger(__name__)
            n = len(self.samples)
            log.info("[GSC] Preloading %d mel tensors into RAM from %s …", n, root)
            spects: list[torch.Tensor] = []
            labels: list[int] = []
            for i, (path, label) in enumerate(self.samples):
                if self._from_pt:
                    spect = self._load_mel_tensor(path).clone()
                else:
                    spect = _load_spectrogram(
                        path, self.normalize, self.mel_max_len, self.audio_sr
                    )
                spects.append(spect)
                labels.append(label)
                if (i + 1) % 10000 == 0 or (i + 1) == n:
                    log.info("[GSC]   RAM cached %d / %d", i + 1, n)
            self._cached_spects = spects
            self._cached_labels = labels
            log.info("[GSC] Done RAM cache for %s.", root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self._cached_spects is not None:
            spect = self._cached_spects[idx]
            label = self._cached_labels[idx]
        else:
            path, label = self.samples[idx]
            if self._from_pt:
                assert self._load_mel_tensor is not None
                spect = self._load_mel_tensor(path)
            else:
                spect = _load_spectrogram(
                    path, self.normalize, self.mel_max_len, self.audio_sr
                )
        if self.transform is not None:
            spect = self.transform(spect)
        return spect, label
