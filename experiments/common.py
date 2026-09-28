"""
Shared utilities for the benchmark scripts: config handling, seeding, logging
and the training loop.

Configuration precedence: built-in defaults < YAML file (--config) < CLI flags.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from fits import FrequencyGuard, build_model

log = logging.getLogger("fits")

# Defaults shared by every benchmark; each script overrides the dataset constants.
COMMON_DEFAULTS = dict(
    seed=0, epochs=100, lr=1e-3, wd=0.0, batch_size=128, eval_batch_size=256, workers=2,
    results_dir=None, data_dir=None,
    neuron_type="fits", hidden_dims=[256, 256], dropout=0.0,
    order=1, chunk_size=32, implicit=True,
    tau_m=0.04, tau_a=0.2, dt=0.004, ratio_eta_gamma=1.0,
    f_min=1.0, f_max=50.0, freq_init="log", learn_freq=True, freq_param="ct",
    freq_guard=True, guard_safety=0.98, backend="auto",
    threshold=1.0, surrogate="triangle", surrogate_alpha=2.0, lam_init=-3.0,
    use_wandb=False, wandb_project="FiTS", wandb_run_name=None,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).strip().lower() in ("true", "1", "yes", "on"):
        return True
    if str(v).strip().lower() in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def build_parser(description: str) -> argparse.ArgumentParser:
    """Parser with the flags shared by all benchmarks (all default to None = not set)."""
    p = argparse.ArgumentParser(description=description)
    a = p.add_argument
    a("--config", type=str, default=None, help="YAML config file")
    a("--data-dir", type=str, default=None)
    a("--results-dir", type=str, default=None, help="output directory for logs and checkpoints")
    a("--seed", type=int, default=None)
    a("--epochs", type=int, default=None)
    a("--lr", type=float, default=None)
    a("--wd", type=float, default=None, help="weight decay")
    a("--batch-size", type=int, default=None)
    a("--workers", type=int, default=None)
    a("--dropout", type=float, default=None)

    a("--neuron-type", type=str, default=None, choices=["fits", "lif", "adaptive_lif"])
    a("--hidden-dims", type=int, nargs="+", default=None, help="e.g. --hidden-dims 512 512")
    a("--order", type=int, default=None, help="TS cascade order M (0 = FS only)")
    a("--chunk-size", type=int, default=None)
    a("--implicit", type=str2bool, default=None, help="semi-implicit (true) or explicit (false) Euler")
    a("--tau-m", type=float, default=None)
    a("--tau-a", type=float, default=None)
    a("--dt", type=float, default=None)
    a("--ratio-eta-gamma", type=float, default=None)
    a("--f-min", type=float, default=None)
    a("--f-max", type=float, default=None)
    a("--freq-init", type=str, default=None, choices=["log", "linear", "constant"])
    a("--learn-freq", type=str2bool, default=None, help="false freezes f* at initialization")
    a("--freq-param", type=str, default=None, choices=["ct", "dt"])
    a("--freq-guard", type=str2bool, default=None, help="project f* below the stability limit")
    a("--guard-safety", type=float, default=None)
    a("--backend", type=str, default=None, choices=["auto", "triton", "torch"])
    a("--threshold", type=float, default=None)
    a("--surrogate", type=str, default=None, choices=["triangle", "atan", "sigmoid"])
    a("--surrogate-alpha", type=float, default=None)
    a("--lam-init", type=float, default=None)

    a("--use-wandb", type=str2bool, default=None)
    a("--wandb-project", type=str, default=None)
    a("--wandb-run-name", type=str, default=None)
    return p


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data.get("hidden_dims"), str):
        data["hidden_dims"] = [int(x) for x in data["hidden_dims"].split()]
    return data


def merge_config(defaults: dict, args: argparse.Namespace) -> SimpleNamespace:
    cfg = dict(COMMON_DEFAULTS)
    cfg.update(defaults)
    if args.config:
        yaml_cfg = load_yaml(args.config)
        unknown = set(yaml_cfg) - set(cfg)
        if unknown:
            raise ValueError(f"unknown keys in {args.config}: {sorted(unknown)}")
        for key, value in yaml_cfg.items():
            # PyYAML reads "3e-3" (no dot) as a string
            if isinstance(cfg[key], float) and isinstance(value, str):
                yaml_cfg[key] = float(value)
        cfg.update(yaml_cfg)
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value
    if cfg["results_dir"] is None:
        raise ValueError("set results_dir in the config or pass --results-dir")
    if cfg["data_dir"] is None:
        raise ValueError("set data_dir in the config or pass --data-dir")
    cfg["neuron_type"] = cfg["neuron_type"].lower()
    return SimpleNamespace(**cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def setup_run(cfg) -> tuple[Path, str]:
    """Create the results directory, configure logging, seed everything."""
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    for h in (logging.FileHandler(out / "train.log", mode="w"), logging.StreamHandler()):
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)
    return out, device


def create_model(cfg, in_features: int, num_classes: int, device: str, readout: str = "sum"):
    model = build_model(cfg, in_features, num_classes, readout=readout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s", model)
    log.info("trainable parameters: %s | device: %s", f"{n_params:,}", device)
    return model, n_params


def create_optimizer(model, cfg):
    """Adam + cosine schedule, plus the f* stability guard when enabled."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0.0)
    guard = None
    if cfg.neuron_type == "fits" and cfg.freq_guard:
        guard = FrequencyGuard(model, optimizer, safety=cfg.guard_safety)
        if guard.caps:
            log.info("%s", guard)
    return optimizer, scheduler, guard


def init_wandb(cfg, config_dict):
    if not cfg.use_wandb:
        return None
    try:
        import wandb
        return wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                          dir=cfg.results_dir, config=config_dict)
    except Exception as exc:  # logging must never stop training
        log.warning("wandb init failed (%s); continuing without it", exc)
        return None


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, device, optimizer=None, prep=None):
    """One pass over ``loader``; trains when an optimizer is given. Returns (loss, acc)."""
    train = optimizer is not None
    model.train(train)
    total_loss, n_correct, n_total, n_batches = 0.0, 0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if prep is not None:
                x = prep(x)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += y.size(0)
    return total_loss / max(n_batches, 1), n_correct / max(n_total, 1)


def fit(model, cfg, out_dir, device, train_loader, test_loader, valid_loader=None,
        prep=None, extra_config=None):
    """
    Train for cfg.epochs and keep the best checkpoint.

    With ``valid_loader``: select the epoch on validation accuracy and report
    the test accuracy of that checkpoint (SSC, GSC, SHD*).
    Without it: select on test accuracy (the best-test protocol used by prior
    SHD and sMNIST work).
    """
    optimizer, scheduler, guard = create_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    config_dict = {**vars(cfg), **(extra_config or {})}
    write_json(out_dir / "config.json", config_dict)
    wandb_run = init_wandb(cfg, config_dict)

    history, best, best_epoch, test_at_best = [], float("-inf"), 0, None
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, prep)
        rec = {"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
               "lr": optimizer.param_groups[0]["lr"]}

        if valid_loader is not None:
            _, va_acc = run_epoch(model, valid_loader, criterion, device, prep=prep)
            rec["valid_acc"] = va_acc
            score = va_acc
        else:
            _, te_acc = run_epoch(model, test_loader, criterion, device, prep=prep)
            rec["test_acc"] = te_acc
            score = te_acc

        if score > best:
            best, best_epoch = score, ep
            if valid_loader is not None:
                _, te_acc = run_epoch(model, test_loader, criterion, device, prep=prep)
                rec["test_acc"] = te_acc
            test_at_best = rec["test_acc"]
            torch.save({"epoch": ep, "state": model.state_dict(), "config": config_dict,
                        "valid_acc": rec.get("valid_acc"), "test_acc": test_at_best},
                       out_dir / "best.pth")
        scheduler.step()

        if guard is not None:
            rec["freq_projected"] = guard.reset_counter()
        msg = f"Ep {ep:3d} | train {tr_acc * 100:5.2f}% loss {tr_loss:.4f}"
        if "valid_acc" in rec:
            msg += f" | valid {rec['valid_acc'] * 100:5.2f}%"
        if "test_acc" in rec:
            msg += f" | test {rec['test_acc'] * 100:5.2f}%"
        log.info(msg + (" *" if best_epoch == ep else ""))
        history.append(rec)
        if wandb_run is not None:
            wandb_run.log(rec, step=ep)

    selection = "valid" if valid_loader is not None else "test"
    summary = {"selection": selection, "best_epoch": best_epoch,
               f"best_{selection}_acc": best, "test_acc": test_at_best}
    write_json(out_dir / "metrics.json", {"history": history})
    write_json(out_dir / "summary.json", summary)
    log.info("[done] test acc %.2f%% at epoch %d (selected on %s)",
             test_at_best * 100, best_epoch, selection)
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    return summary
