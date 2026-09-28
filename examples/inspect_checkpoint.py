"""
Read out what every neuron of a trained FiTS network learned.

    python examples/inspect_checkpoint.py runs/gsc/fits_o1_w512/best.pth [--csv neurons.csv] [--plot fstar.png]

Prints, per layer, the distribution of
    f*       target frequency (Hz), the learned FS coordinate
    f*_DT    peak realized by the discrete-time FS update (Hz)
    λ        TS mixing weights
    τ_TS     group delay added by the TS module at each neuron's f* (ms; < 0 = advance)
The checkpoint config is read from the checkpoint itself or from config.json
next to it.
"""

import argparse
import csv
import json
import os

import torch

from fits import build_model

_ALIASES = {"tau_mem": "tau_m", "tau_b": "tau_a"}   # key names used by older runs


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        with open(os.path.join(os.path.dirname(path), "config.json")) as f:
            cfg = json.load(f)
    cfg = {_ALIASES.get(k, k): v for k, v in cfg.items()}
    cfg.setdefault("neuron_type", "fits")
    state = ckpt["state"]
    in_features = state["neuron_layers.0.linear.weight"].shape[1]
    num_classes = state["output.weight"].shape[0]
    model = build_model(cfg, in_features, num_classes)
    model.load_state_dict(state)
    return model.eval()


def quantiles(x):
    x = x.double()
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return "   (no finite values)"
    q = torch.quantile(finite, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64))
    return "  ".join(f"{v:9.3f}" for v in q.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--csv", default=None, help="write one row per neuron")
    p.add_argument("--plot", default=None, help="save a histogram of f* per layer")
    args = p.parse_args()

    model = load_model(args.checkpoint)
    rows = []
    print(f"{'':22s}{'min':>9s}  {'q25':>9s}  {'median':>9s}  {'q75':>9s}  {'max':>9s}")
    for li, layer in enumerate(model.neuron_layers):
        f = layer.get_freq().detach()
        f_dt = layer.realized_freq() if layer.implicit else torch.full_like(f, float("nan"))
        delay_ms = layer.ts_group_delay() * 1e3
        lams = layer.get_lam_list()
        print(f"layer {li}  (N={layer.out_features}, M={layer.order}, freq_param={layer.freq_param})")
        print(f"  {'f* (Hz)':20s}{quantiles(f)}")
        print(f"  {'f*_DT (Hz)':20s}{quantiles(f_dt)}")
        for m, lam in enumerate(lams, 1):
            print(f"  {f'lambda_{m}':20s}{quantiles(lam.detach())}")
        if layer.order > 0:
            print(f"  {'TS delay @ f* (ms)':20s}{quantiles(delay_ms)}"
                  f"   ({(delay_ms < 0).float().mean() * 100:.1f}% advanced)")
        n_bad = int((~torch.isfinite(layer.get_kappa())).sum())
        if n_bad:
            print(f"  warning: {n_bad} neurons have a non-finite coupling (diverged during training)")
        for n in range(layer.out_features):
            rows.append(dict(layer=li, neuron=n, f_star_hz=float(f[n]), f_star_dt_hz=float(f_dt[n]),
                             ts_delay_ms=float(delay_ms[n]),
                             **{f"lambda_{m}": float(l[n]) for m, l in enumerate(lams, 1)}))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")

    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for li, layer in enumerate(model.neuron_layers):
            f = layer.get_freq().detach().numpy()
            f = f[np.isfinite(f)]
            ax.hist(f, bins=np.geomspace(f.min(), f.max(), 40), histtype="step", lw=1.5, label=f"layer {li}")
        ax.set_xscale("log")
        ax.set_xlabel("learned target frequency f* (Hz)")
        ax.set_ylabel("neurons")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
