"""
Google Speech Commands v0.02 (GSC, 35 classes), 40-band log-Mel inputs, T = 100.

Downloads and splits the dataset under ``data_dir`` on first use (disable with
``--skip-prepare true``).  The epoch is selected on the official validation
split and its test accuracy is reported.

    python -m GSC.scripts.train.train_chunked --config config/GSC/fits_o1_w512.yaml --data-dir /path/to/GSC/processed
"""

import os

import torch
from torch.utils.data import DataLoader

from GSC.core.utils import (build_parser, create_model, fit, log, merge_config, setup_run,
                                str2bool)
from GSC.core.data import N_CLASS, GCommandLoader
from GSC.core.prepare_gsc import ensure_gsc_prepared
from GSC.models.fc import build_model

IN_FEATURES = 40

DEFAULTS = dict(
    tau_m=0.1, tau_a=0.5, dt=0.01, f_min=1.0, f_max=30.0,
    mel_max_len=100, audio_sr=16000, skip_prepare=False,
    cache_mels=True, mel_cache_dir=None, ensure_mel_cache=True,
)


def to_time_major(x: torch.Tensor) -> torch.Tensor:
    """[B, 1, 40, T] mel batch -> [B, T, 40]."""
    return x.squeeze(1).permute(0, 2, 1).contiguous()


def main():
    p = build_parser("Train FiTS on GSC v0.02")
    p.add_argument("--mel-max-len", type=int, default=None)
    p.add_argument("--audio-sr", type=int, default=None)
    p.add_argument("--skip-prepare", type=str2bool, default=None,
                   help="do not download / split the dataset")
    p.add_argument("--cache-mels", type=str2bool, default=None,
                   help="keep all mel spectrograms in RAM")
    p.add_argument("--mel-cache-dir", type=str, default=None,
                   help="directory for precomputed per-file mel tensors (.pt)")
    p.add_argument("--ensure-mel-cache", type=str2bool, default=None,
                   help="build missing .pt files in mel_cache_dir before training")
    cfg = merge_config(DEFAULTS, p.parse_args())
    out_dir, device = setup_run(cfg)

    if not cfg.skip_prepare:
        ensure_gsc_prepared(cfg.data_dir, verbose=True)

    model, n_params = create_model(cfg, device, build_model)

    ds_kw = dict(normalize=True, mel_max_len=cfg.mel_max_len, audio_sr=cfg.audio_sr,
                 cache_mels=cfg.cache_mels, mel_cache_dir=cfg.mel_cache_dir or None,
                 ensure_mel_cache=cfg.ensure_mel_cache)
    log.info("building datasets from %s", cfg.data_dir)
    tr_ds = GCommandLoader(os.path.join(cfg.data_dir, "train"), **ds_kw)
    va_ds = GCommandLoader(os.path.join(cfg.data_dir, "valid"), **ds_kw)
    te_ds = GCommandLoader(os.path.join(cfg.data_dir, "test"), **ds_kw)

    dl_kw = dict(pin_memory=(device == "cuda"), num_workers=cfg.workers)
    if cfg.workers > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=2)
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, **dl_kw)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kw)
    te_ld = DataLoader(te_ds, batch_size=cfg.eval_batch_size, shuffle=False, **dl_kw)

    fit(model, cfg, out_dir, device, tr_ld, te_ld, valid_loader=va_ld, prep=to_time_major,
        extra_config=dict(dataset="GSC_v0.02_35", n_params=n_params, n_class=N_CLASS,
                          in_features=IN_FEATURES))


if __name__ == "__main__":
    main()
