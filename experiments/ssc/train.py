"""
Spiking Speech Commands (SSC, 35 classes).

Uses the official train / valid / test split; the epoch is selected on the
validation set and its test accuracy is reported.

    python -m experiments.ssc.train --config configs/ssc/fits_o1_w512.yaml --data-dir /path/to/SSC
"""

import os

from experiments.common import build_parser, create_model, fit, merge_config, setup_run
from experiments.spike_data import ORIG_UNITS, SpikeTrainDataset, make_loader, read_h5

N_CLASS = 35

DEFAULTS = dict(
    time_window=250, in_features=140,
    tau_m=0.04, tau_a=0.2, dt=0.004, f_min=1.0, f_max=50.0,
)


def main():
    p = build_parser("Train FiTS on SSC")
    p.add_argument("--time-window", type=int, default=None)
    p.add_argument("--in-features", type=int, default=None)
    cfg = merge_config(DEFAULTS, p.parse_args())
    out_dir, device = setup_run(cfg)

    model, n_params = create_model(cfg, cfg.in_features, N_CLASS, device)

    T, C = cfg.time_window, cfg.in_features
    split = lambda name: SpikeTrainDataset(
        *read_h5(os.path.join(cfg.data_dir, f"ssc_{name}.h5")), T, C)
    tr_ld = make_loader(split("train"), cfg.batch_size, True, cfg.workers, T, C)
    va_ld = make_loader(split("valid"), cfg.eval_batch_size, False, cfg.workers, T, C)
    te_ld = make_loader(split("test"), cfg.eval_batch_size, False, cfg.workers, T, C)

    fit(model, cfg, out_dir, device, tr_ld, te_ld, valid_loader=va_ld,
        extra_config=dict(dataset="SSC", n_params=n_params, n_class=N_CLASS, orig_units=ORIG_UNITS))


if __name__ == "__main__":
    main()
