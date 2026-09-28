"""
Tests for the FiTS package.  Run with ``python -m pytest tests``.

CPU tests cover the frequency maps, the PyTorch reference path, the baselines,
the stability guard and checkpoint layout.  The Triton tests compare the fused
kernels with the reference path and are skipped without CUDA.
"""

import math

import pytest
import torch

from fits import (FiTSNet, FiTSNeuron, FrequencyGuard, LIFNeuron, critical_freq,
                  ct_peak_freq, dt_peak_freq, jury_bound, kappa_ct, kappa_dt)

CONSTANTS = [(0.04, 0.2, 0.004), (0.1, 0.5, 0.01), (20.0, 150.0, 1.0)]  # SHD/SSC, GSC, sMNIST


# ─────────────────────────────────────────────────────────────────────────────
# Frequency parameterization
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("tau_m,tau_a,dt", CONSTANTS)
def test_ct_and_dt_maps_invert(tau_m, tau_a, dt):
    mu, rho = 1 / tau_m, 1 / tau_a
    f = torch.linspace(0.01, 0.45, 200, dtype=torch.float64) / dt
    assert torch.allclose(ct_peak_freq(kappa_ct(2 * math.pi * f, mu, rho), mu, rho), f, atol=1e-9)
    assert torch.allclose(dt_peak_freq(kappa_dt(2 * math.pi * f, mu, rho, dt), mu, rho, dt), f,
                          atol=1e-9)


@pytest.mark.parametrize("tau_m,tau_a,dt", CONSTANTS)
def test_dt_peak_matches_brute_force(tau_m, tau_a, dt):
    """dt_peak_freq equals the argmax of |H(e^{jω})| of the semi-implicit update."""
    mu, rho = 1 / tau_m, 1 / tau_a
    mu_b, rho_b = 1 - mu * dt, 1 - rho * dt
    w = torch.linspace(0, math.pi, 200_001, dtype=torch.float64)
    zi = torch.exp(-1j * w)
    for f in (0.02 / dt, 0.1 / dt, 0.2 / dt):
        kappa = kappa_ct(torch.tensor(2 * math.pi * f, dtype=torch.float64), mu, rho)
        kb = float(kappa) * dt * dt
        if kb >= jury_bound(mu, rho, dt):
            continue
        H = (1 - rho_b * zi) / ((1 - mu_b * zi) * (1 - rho_b * zi) + kb * zi)
        f_brute = float(w[H.abs().argmax()]) / (2 * math.pi * dt)
        grid = math.pi / 200_000 / (2 * math.pi * dt)
        assert abs(f_brute - float(dt_peak_freq(kappa, mu, rho, dt))) <= grid


def test_critical_frequency_values():
    # values quoted in the README
    assert critical_freq(1 / 0.1, 1 / 0.5, 0.01, "ct") == pytest.approx(30.88, abs=0.01)
    assert critical_freq(1 / 0.04, 1 / 0.2, 0.004, "ct") == pytest.approx(77.19, abs=0.01)
    # the exact-DT map is stable up to Nyquist for these constants
    assert critical_freq(1 / 0.1, 1 / 0.5, 0.01, "dt") == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
# Neuron (PyTorch backend)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("order", [0, 1, 2, 3, 6])
@pytest.mark.parametrize("freq_param", ["ct", "dt"])
def test_neuron_forward_backward_cpu(order, freq_param):
    torch.manual_seed(0)
    layer = FiTSNeuron(8, 16, order=order, freq_param=freq_param, backend="torch")
    x = (torch.rand(4, 30, 8) < 0.3).float()
    spikes, v, a, w1, w2 = layer(x)
    assert spikes.shape == (4, 30, 16) and v.shape == a.shape == w1.shape == w2.shape == (4, 16)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}
    (spikes.sum() + v.sum()).backward()
    assert layer.f_raw.grad is not None and torch.isfinite(layer.f_raw.grad).all()
    if order > 0:
        assert layer.ap_beta_raw.grad is not None


def test_state_is_carried_across_calls():
    """Splitting a sequence in two calls with carried state equals one call."""
    torch.manual_seed(0)
    layer = FiTSNeuron(8, 16, order=2, backend="torch")
    x = torch.randn(2, 40, 8)
    full = layer(x)[0]
    s1, v, a, w1, w2 = layer(x[:, :17])
    s2 = layer(x[:, 17:], v, a, w1, w2)[0]
    assert torch.equal(full, torch.cat([s1, s2], dim=1))


def test_frozen_frequencies():
    layer = FiTSNeuron(4, 8, order=1, learn_freq=False)
    names = [n for n, _ in layer.named_parameters()]
    assert "f_raw" not in names and "f_raw" in layer.state_dict()


def test_freq_init_modes():
    for mode in ("log", "linear", "constant"):
        f = FiTSNeuron(4, 5, freq_init=mode, f_min=1.0, f_max=64.0).get_freq()
        assert f.min() >= 1.0 - 1e-4 and f.max() <= 64.0 + 1e-3
    assert torch.allclose(FiTSNeuron(4, 5, freq_init="constant", f_min=1.0, f_max=64.0).get_freq(),
                          torch.full((5,), 8.0))


def test_realized_freq_dt_param_is_exact():
    layer = FiTSNeuron(4, 32, freq_param="dt", f_min=1.0, f_max=100.0)
    assert torch.allclose(layer.realized_freq(), layer.get_freq().double(), atol=1e-3)


def test_ts_group_delay():
    layer = FiTSNeuron(4, 3, order=1)
    with torch.no_grad():
        layer.ap_beta_raw.zero_()         # β = 0: the all-pass stage is a one-sample delay
        layer.lam1_raw.fill_(50.0)        # λ ≈ 1: output = all-pass branch only
    assert torch.allclose(layer.ts_group_delay(), torch.full((3,), layer.dt, dtype=torch.float64),
                          rtol=1e-4)
    with torch.no_grad():
        layer.lam1_raw.fill_(-50.0)       # λ ≈ 0: FS pathway only, no added delay
    assert layer.ts_group_delay().abs().max() < 1e-8


def test_plain_lif_matches_manual_loop():
    torch.manual_seed(0)
    layer = LIFNeuron(6, 5, adaptive=False, backend="torch")
    x = torch.randn(2, 25, 6)
    spikes = layer(x)[0]
    u = layer.linear(x)
    v = torch.zeros(2, 5)
    decay = 1 - layer.dt / layer.tau_m
    ref = []
    for t in range(25):
        v = decay * v + u[:, t]
        s = (v > 1.0).float()
        v = v - s
        ref.append(s)
    assert torch.equal(spikes, torch.stack(ref, 1))
    assert LIFNeuron(6, 5, adaptive=True).get_kappa().gt(0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Stability guard
# ─────────────────────────────────────────────────────────────────────────────
def test_frequency_guard_projects_and_releases():
    layer = FiTSNeuron(4, 6, order=0, tau_m=0.1, tau_a=0.5, dt=0.01, f_min=1.0, f_max=30.0)
    opt = torch.optim.Adam(layer.parameters(), lr=0.5)
    guard = FrequencyGuard(layer, opt)
    cap = next(iter(guard.caps.values()))
    for _ in range(5):                    # push f* upwards
        opt.zero_grad()
        (-layer.f_raw.sum()).backward()
        opt.step()
    assert layer.f_raw.max().item() <= cap + 1e-6
    assert guard.num_projected > 0
    pinned = layer.f_raw >= cap - 1e-6
    assert pinned.any()
    assert (opt.state[layer.f_raw]["exp_avg"][pinned] >= 0).all()   # no outward momentum left
    opt.zero_grad()
    layer.f_raw.sum().backward()          # pull f* downwards: pinned units must move
    opt.step()
    assert layer.f_raw.max().item() < cap


def test_guard_skips_frozen_frequencies():
    layer = FiTSNeuron(4, 6, learn_freq=False)
    guard = FrequencyGuard(layer, torch.optim.Adam(layer.parameters()))
    assert guard.caps == {}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint layout (must stay compatible with released checkpoints)
# ─────────────────────────────────────────────────────────────────────────────
def test_state_dict_keys():
    net = FiTSNet(140, [16, 16], 20, order=2)
    keys = set(net.state_dict())
    for i in (0, 1):
        for k in ("linear.weight", "linear.bias", "f_raw", "ap_beta_raw", "lam1_raw", "lam2_raw"):
            assert f"neuron_layers.{i}.{k}" in keys
    assert {"output.weight", "output.bias"} <= keys


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernels vs. reference (CUDA only)
# ─────────────────────────────────────────────────────────────────────────────
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("implicit", [True, False])
def test_triton_matches_reference(order, implicit):
    torch.manual_seed(0)
    # The explicit FS update is unstable at the default 50 Hz upper end
    # (spectral radius > 1). Compare backends within its stable range so
    # growing roundoff does not dominate the gradient comparison.
    kw = dict(order=order, implicit=implicit, chunk_size=16,
              f_max=50.0 if implicit else 10.0)
    fast = FiTSNeuron(20, 64, backend="triton", **kw).cuda()
    ref = FiTSNeuron(20, 64, backend="torch", **kw).cuda()
    ref.load_state_dict(fast.state_dict())
    with torch.no_grad():                 # non-trivial TS parameters
        for layer in (fast, ref):
            if order > 0:
                torch.manual_seed(1)
                layer.ap_beta_raw.normal_(0, 0.5)
                for name in ("lam1_raw", "lam2_raw", "ap_lam_raw"):
                    p = getattr(layer, name)
                    if p is not None:
                        p.normal_(0, 1.0)
    x = (torch.rand(8, 50, 20, device="cuda") < 0.2).float()

    out_f = fast(x)
    out_r = ref(x)
    mismatch = (out_f[0] != out_r[0]).float().mean().item()
    assert mismatch < 1e-3, f"spike mismatch {mismatch:.2e}"

    g = torch.randn_like(out_f[0])
    (out_f[0] * g).sum().backward()
    (out_r[0] * g).sum().backward()
    for (name, pf), (_, pr) in zip(fast.named_parameters(), ref.named_parameters()):
        assert torch.allclose(pf.grad, pr.grad, rtol=1e-3, atol=1e-4), name
