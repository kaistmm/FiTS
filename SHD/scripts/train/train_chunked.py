"""
Spiking Heidelberg Digits (SHD, 20 classes).

Two protocols, as in the paper (Table 1):
    eval_mode: test_only    train on the full training set, report the best
                            test accuracy over training ("SHD")
    eval_mode: valid_split  hold out valid_frac of the training set (fixed
                            split seed), select the epoch on validation and
                            report its test accuracy ("SHD*")

    python -m SHD.scripts.train.train_chunked --config config/SHD/fits_o1_w128.yaml --data-dir /path/to/SHD
"""

import os

import numpy as np

from SHD.core.utils import build_parser, create_model, fit, log, merge_config, setup_run
from SHD.core.data import ORIG_UNITS, SpikeTrainDataset, make_loader, read_h5
from SHD.models.fc import build_model

N_CLASS = 20

DEFAULTS = dict(
    time_window=250, in_features=140,
    tau_m=0.04, tau_a=0.2, dt=0.004, f_min=1.0, f_max=50.0,
    eval_mode="test_only", valid_frac=0.2, valid_split_seed=0,
)


def main():
    p = build_parser("Train FiTS on SHD")
    p.add_argument("--eval-mode", type=str, default=None, choices=["test_only", "valid_split"])
    p.add_argument("--valid-frac", type=float, default=None)
    p.add_argument("--valid-split-seed", type=int, default=None)
    p.add_argument("--time-window", type=int, default=None)
    p.add_argument("--in-features", type=int, default=None)
    cfg = merge_config(DEFAULTS, p.parse_args())
    out_dir, device = setup_run(cfg)

    model, n_params = create_model(cfg, device, build_model)

    T, C = cfg.time_window, cfg.in_features
    times, units, labels = read_h5(os.path.join(cfg.data_dir, "shd_train.h5"))
    te_ds = SpikeTrainDataset(*read_h5(os.path.join(cfg.data_dir, "shd_test.h5")), T, C)

    va_ld = None
    if cfg.eval_mode == "valid_split":
        idx = np.random.RandomState(cfg.valid_split_seed).permutation(len(labels))
        n_valid = max(1, int(len(labels) * cfg.valid_frac))
        va_idx, tr_idx = idx[:n_valid], idx[n_valid:]
        tr_ds = SpikeTrainDataset(times[tr_idx], units[tr_idx], labels[tr_idx], T, C)
        va_ds = SpikeTrainDataset(times[va_idx], units[va_idx], labels[va_idx], T, C)
        va_ld = make_loader(va_ds, cfg.eval_batch_size, False, cfg.workers, T, C)
        log.info("SHD* protocol: train %d / valid %d (split seed %d)",
                 len(tr_ds), len(va_ds), cfg.valid_split_seed)
    elif cfg.eval_mode == "test_only":
        tr_ds = SpikeTrainDataset(times, units, labels, T, C)
        log.info("SHD best-test protocol: train %d", len(tr_ds))
    else:
        raise ValueError(f"unknown eval_mode {cfg.eval_mode!r}")

    tr_ld = make_loader(tr_ds, cfg.batch_size, True, cfg.workers, T, C)
    te_ld = make_loader(te_ds, cfg.eval_batch_size, False, cfg.workers, T, C)

    fit(model, cfg, out_dir, device, tr_ld, te_ld, valid_loader=va_ld,
        extra_config=dict(dataset="SHD", n_params=n_params, n_class=N_CLASS, orig_units=ORIG_UNITS))


if __name__ == "__main__":
    main()
