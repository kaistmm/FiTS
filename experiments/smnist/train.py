"""
Sequential MNIST (one pixel per step, T = 784) and permuted sMNIST.

Time is measured in pixel steps (Δt = 1), so τ_m, τ_a and f* are in steps and
cycles/step.  The spike trains are mean-pooled over time before the readout.
As in the paper's Table A.9 (and the prior results listed there), the best
test accuracy over training is reported.

    python -m experiments.smnist.train --config configs/smnist/fits_o1_w256.yaml --data-dir ./data/MNIST
"""

from experiments.common import build_parser, create_model, fit, log, merge_config, setup_run
from experiments.smnist.data import make_loaders, make_permutation

IN_FEATURES, N_CLASS = 1, 10

DEFAULTS = dict(
    task="smnist", batch_size=256,
    tau_m=20.0, tau_a=150.0, dt=1.0, f_min=0.001, f_max=0.1, freq_init="linear",
)


def main():
    p = build_parser("Train FiTS on sMNIST / psMNIST")
    p.add_argument("--task", type=str, default=None, choices=["smnist", "psmnist"])
    cfg = merge_config(DEFAULTS, p.parse_args())
    out_dir, device = setup_run(cfg)

    perm = make_permutation(cfg.seed) if cfg.task == "psmnist" else None
    tr_ld, te_ld = make_loaders(cfg.task, data_dir=cfg.data_dir, batch_size=cfg.batch_size,
                                num_workers=cfg.workers, perm=perm, download=True)
    log.info("%s: train %d / test %d", cfg.task, len(tr_ld.dataset), len(te_ld.dataset))

    model, n_params = create_model(cfg, IN_FEATURES, N_CLASS, device, readout="mean")

    extra = dict(dataset=cfg.task, n_params=n_params, n_class=N_CLASS)
    if perm is not None:
        extra["psmnist_perm"] = perm.tolist()
    fit(model, cfg, out_dir, device, tr_ld, te_ld, extra_config=extra)


if __name__ == "__main__":
    main()
