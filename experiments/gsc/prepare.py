"""
Download and preprocess Google Speech Commands v0.02 (35 core classes).

Layout:
    <gsc_root>/
        raw/
            speech_commands_v0.02.tar.gz   (optional cache)
            ... unpacked files + validation_list.txt ...
        processed/
            train/<class>/*.wav
            valid/<class>/*.wav
            test/<class>/*.wav
"""

from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request

# Official TensorFlow mirror (same as used by tensorflow.org / tfds).
GSC_V02_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)
TAR_FILENAME = "speech_commands_v0.02.tar.gz"


def _find_unpacked_root(raw_dir: str) -> str:
    """Directory that contains validation_list.txt and testing_list.txt."""
    if os.path.isfile(os.path.join(raw_dir, "validation_list.txt")):
        return raw_dir
    for name in sorted(os.listdir(raw_dir)):
        p = os.path.join(raw_dir, name)
        if os.path.isdir(p) and os.path.isfile(
            os.path.join(p, "validation_list.txt")
        ):
            return p
    raise FileNotFoundError(
        f"Could not find validation_list.txt under {raw_dir} "
        "(extract the v0.02 tarball into this directory)."
    )


def _move_files(original_fold: str, data_fold: str, list_filename: str) -> None:
    with open(list_filename, encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            parts = rel.split("/")
            dest_sub = os.path.join(data_fold, parts[0])
            os.makedirs(dest_sub, exist_ok=True)
            src = os.path.join(original_fold, rel)
            dst = os.path.join(data_fold, rel)
            if os.path.isfile(src):
                shutil.move(src, dst)


def _create_train_fold(original_fold: str, train_fold: str) -> None:
    for name in list(os.listdir(original_fold)):
        src = os.path.join(original_fold, name)
        if os.path.isdir(src):
            dst = os.path.join(train_fold, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)


def _make_train_valid_test(gcommands_fold: str, out_path: str) -> None:
    validation_path = os.path.join(gcommands_fold, "validation_list.txt")
    test_path = os.path.join(gcommands_fold, "testing_list.txt")
    if not os.path.isfile(validation_path) or not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Missing validation_list.txt or testing_list.txt in {gcommands_fold}"
        )

    valid_fold = os.path.join(out_path, "valid")
    test_fold = os.path.join(out_path, "test")
    train_fold = os.path.join(out_path, "train")
    for d in (valid_fold, test_fold, train_fold):
        os.makedirs(d, exist_ok=True)

    _move_files(gcommands_fold, test_fold, test_path)
    _move_files(gcommands_fold, valid_fold, validation_path)
    _create_train_fold(gcommands_fold, train_fold)


def _tree_has_wavs(root: str) -> bool:
    for _, _, files in os.walk(root):
        if any(f.lower().endswith(".wav") for f in files):
            return True
    return False


def _processed_has_samples(processed_dir: str) -> bool:
    train = os.path.join(processed_dir, "train")
    if not os.path.isdir(train):
        return False
    n_wav = 0
    for _, _, files in os.walk(train):
        n_wav += sum(1 for f in files if f.lower().endswith(".wav"))
    return n_wav > 5000


def ensure_gsc_prepared(processed_dir: str, *, verbose: bool = True) -> None:
    """
    If ``processed_dir/train`` is missing or empty, download v0.02 under the
    parent of ``processed_dir`` (expected ``<gsc_root>/processed``) and build
    train/valid/test splits.
    """
    processed_dir = os.path.abspath(os.path.expanduser(processed_dir))
    if _processed_has_samples(processed_dir):
        if verbose:
            print(f"[GSC] Using existing processed data: {processed_dir}", flush=True)
        return

    gsc_root = os.path.dirname(processed_dir)
    raw_dir = os.path.join(gsc_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    tar_path = os.path.join(raw_dir, TAR_FILENAME)
    try:
        unpacked = _find_unpacked_root(raw_dir)
    except FileNotFoundError:
        unpacked = None

    if unpacked is None or not _tree_has_wavs(unpacked):
        if not os.path.isfile(tar_path):
            if verbose:
                print(f"[GSC] Downloading {GSC_V02_URL} → {tar_path}", flush=True)
            urllib.request.urlretrieve(GSC_V02_URL, tar_path)
        elif verbose:
            print(f"[GSC] Using cached archive {tar_path}", flush=True)
        if verbose:
            print(f"[GSC] Extracting {tar_path} → {raw_dir}", flush=True)
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=raw_dir)
        unpacked = _find_unpacked_root(raw_dir)
        if not _tree_has_wavs(unpacked):
            raise RuntimeError(
                f"Extracted archive under {raw_dir} contains no WAV files."
            )

    if verbose:
        print(f"[GSC] Building train/valid/test → {processed_dir}", flush=True)
    os.makedirs(processed_dir, exist_ok=True)
    for sub in ("train", "valid", "test"):
        p = os.path.join(processed_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    _make_train_valid_test(unpacked, processed_dir)

    if not _processed_has_samples(processed_dir):
        raise RuntimeError(
            f"Preprocessing failed: no WAVs found under {processed_dir}/train"
        )
    if verbose:
        print("[GSC] Dataset ready.", flush=True)


if __name__ == "__main__":
    # python -m experiments.gsc.prepare --data-dir /path/to/GSC/processed
    import argparse
    parser = argparse.ArgumentParser(description="Download GSC v0.02 and build train/valid/test splits")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="target directory for the processed splits (e.g. /path/to/GSC/processed); "
                             "the raw archive is stored in its sibling directory raw/")
    ensure_gsc_prepared(parser.parse_args().data_dir)
