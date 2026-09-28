"""
FiTS neuron layer: linear projection + FS/TS temporal dynamics + spikes.

    x [B, T, in] ──Linear──▶ I [B, T, N] ──FiTS update over T──▶ spikes [B, T, N]

Learnable per-neuron parameters
    f_raw        [N]     log target frequency, f* = exp(f_raw) in Hz (unbounded)
    ap_beta_raw  [M, N]  all-pass coefficients, β_m = tanh(raw) ∈ (-1, 1)
    lam1_raw, lam2_raw   TS mixing weights λ = sigmoid(raw) ∈ (0, 1)   (M = 1, 2)
    ap_lam_raw   [M, N]  TS mixing weights for M > 2
(plus ``linear``).  M = ``order`` is the TS cascade order; M = 0 is FS only.

κ = η·γ is computed from f* in closed form on every forward pass
(``fits.frequency``), so f* is simultaneously the initialization, the
optimization coordinate and the post-training readout.

Backends
    "triton" : fused CUDA kernels (``fits/kernels``), M ≤ 5.  Requires the
               layer input for all T timesteps up front, i.e. a feedforward
               layer.
    "torch"  : plain PyTorch time loop (``fits/reference.py``); CPU/GPU, any M.
    "auto"   : Triton for CUDA inputs when available, otherwise PyTorch.
"""

import math
import warnings

import torch
import torch.nn as nn

from fits.frequency import (critical_freq, dt_peak_freq, eta_gamma_from_kappa,
                            jury_bound, kappa_ct, kappa_from_freq)
from fits.reference import fits_temporal_reference

try:
    from fits.kernels import fused_fits_temporal, fused_fits_temporal_high
    _TRITON_IMPORT_ERROR = None
except Exception as exc:  # triton missing, or no usable backend
    fused_fits_temporal = fused_fits_temporal_high = None
    _TRITON_IMPORT_ERROR = exc

__all__ = ["FiTSNeuron", "LIFNeuron", "init_log_freq"]


def init_log_freq(num_neurons: int, f_min: float, f_max: float,
                  freq_init: str = "log") -> torch.Tensor:
    """
    Initial f_raw = log(f*) for a layer.

    freq_init:
        "log"      log-uniform grid over [f_min, f_max]   (paper default)
        "linear"   linear grid over [f_min, f_max]
        "constant" every neuron at the log midpoint sqrt(f_min·f_max)
    """
    if freq_init == "log":
        f = torch.logspace(math.log10(f_min), math.log10(f_max), num_neurons)
    elif freq_init == "linear":
        f = torch.linspace(f_min, f_max, num_neurons)
    elif freq_init == "constant":
        f = torch.full((num_neurons,), math.sqrt(f_min * f_max))
    else:
        raise ValueError(f"freq_init must be 'log', 'linear' or 'constant', got {freq_init!r}")
    return torch.log(f)


class FiTSNeuron(nn.Module):
    """
    A layer of FiTS neurons.

    Args:
        in_features     : input dimension.
        out_features    : number of neurons N.
        order           : TS cascade order M (0 = FS only).
        chunk_size      : time-chunk length per Triton launch; a tiling choice
                          only (states are carried exactly across chunks).
        tau_m, tau_a    : membrane / adaptation time constants (s), μ = 1/τ_m, ρ = 1/τ_a.
        dt              : simulation timestep Δt (s).
        ratio_eta_gamma : η/γ (the paper uses 1.0).
        f_min, f_max    : range of the initial target frequencies (Hz).  f* is
                          not clamped to this range during training.
        v_threshold     : firing threshold V_th.
        surrogate_alpha : surrogate-gradient steepness.
        surrogate_type  : 0 = triangle, 1 = arctan, 2 = sigmoid.
        lam_init        : initial λ logit; sigmoid(-3) ≈ 0.047 starts close to FS only.
        implicit        : semi-implicit (True, paper) or explicit Euler FS update.
        freq_init       : "log" (paper) | "linear" | "constant", see ``init_log_freq``.
        learn_freq      : False freezes f* at its initialization (stored as a buffer).
        freq_param      : "ct" (paper, Theorem 1) or "dt" (exact discrete-time map).
        backend         : "auto" | "triton" | "torch".
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 order: int = 1,
                 chunk_size: int = 256,
                 tau_m: float = 0.04,
                 tau_a: float = 0.2,
                 dt: float = 0.004,
                 ratio_eta_gamma: float = 1.0,
                 f_min: float = 1.0,
                 f_max: float = 50.0,
                 v_threshold: float = 1.0,
                 surrogate_alpha: float = 2.0,
                 surrogate_type: int = 0,
                 lam_init: float = -3.0,
                 implicit: bool = True,
                 freq_init: str = "log",
                 learn_freq: bool = True,
                 freq_param: str = "ct",
                 backend: str = "auto"):
        super().__init__()
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")
        if freq_param not in ("ct", "dt"):
            raise ValueError(f"freq_param must be 'ct' or 'dt', got {freq_param!r}")
        if freq_param == "dt" and not implicit:
            raise ValueError("freq_param='dt' is derived for the semi-implicit update (implicit=True)")
        if backend not in ("auto", "triton", "torch"):
            raise ValueError(f"backend must be 'auto', 'triton' or 'torch', got {backend!r}")

        self.in_features     = in_features
        self.out_features    = out_features
        self.order           = order
        self.chunk_size      = chunk_size
        self.tau_m           = tau_m
        self.tau_a           = tau_a
        self.dt              = dt
        self.ratio_eta_gamma = ratio_eta_gamma
        self.f_min           = f_min
        self.f_max           = f_max
        self.v_threshold     = v_threshold
        self.surrogate_alpha = surrogate_alpha
        self.surrogate_type  = surrogate_type
        self.implicit        = implicit
        self.freq_init       = freq_init
        self.learn_freq      = learn_freq
        self.freq_param      = freq_param
        self.backend         = backend

        self.mu  = 1.0 / tau_m
        self.rho = 1.0 / tau_a

        self.linear = nn.Linear(int(in_features), int(out_features))

        f_raw = init_log_freq(out_features, f_min, f_max, freq_init)
        if learn_freq:
            self.f_raw = nn.Parameter(f_raw)
        else:
            self.register_buffer("f_raw", f_raw)

        if order > 0:
            self.ap_beta_raw = nn.Parameter(torch.zeros(order, out_features))
        else:
            self.ap_beta_raw = None

        if order == 1:
            self.lam1_raw   = nn.Parameter(torch.full((out_features,), float(lam_init)))
            self.lam2_raw   = None
            self.ap_lam_raw = None
        elif order == 2:
            self.lam1_raw   = nn.Parameter(torch.full((out_features,), float(lam_init)))
            self.lam2_raw   = nn.Parameter(torch.full((out_features,), float(lam_init)))
            self.ap_lam_raw = None
        elif order > 2:
            self.lam1_raw   = None
            self.lam2_raw   = None
            self.ap_lam_raw = nn.Parameter(torch.full((order, out_features), float(lam_init)))
        else:
            self.lam1_raw   = None
            self.lam2_raw   = None
            self.ap_lam_raw = None

        if implicit:
            f_crit = critical_freq(self.mu, self.rho, dt, freq_param)
            if f_max >= f_crit:
                warnings.warn(
                    f"f_max={f_max} Hz is at or above the stability limit {f_crit:.2f} Hz "
                    f"of the semi-implicit update (tau_m={tau_m}, tau_a={tau_a}, dt={dt}, "
                    f"freq_param={freq_param!r}); neurons initialized above it start unstable.",
                    stacklevel=2)

    # ── parameter accessors ────────────────────────────────────────────────
    def get_freq(self) -> torch.Tensor:
        """Target frequencies f* = exp(f_raw) in Hz, shape [N]."""
        return torch.exp(self.f_raw)

    def get_omega(self) -> torch.Tensor:
        """Target angular frequencies Ω* = 2π·f* in rad/s, shape [N]."""
        return 2.0 * math.pi * self.get_freq()

    def get_kappa(self) -> torch.Tensor:
        """Adaptation coupling κ = η·γ realizing f* under ``freq_param``, shape [N]."""
        return kappa_from_freq(self.get_freq(), self.mu, self.rho, self.dt, self.freq_param)

    def get_eta_gamma(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(η, γ) with η·γ = κ and η/γ = ratio_eta_gamma."""
        return eta_gamma_from_kappa(self.get_kappa(), self.ratio_eta_gamma)

    def get_beta_list(self) -> list[torch.Tensor]:
        """All-pass coefficients [β_1, ..., β_M], each [N]."""
        if self.order == 0 or self.ap_beta_raw is None:
            return []
        betas = torch.tanh(self.ap_beta_raw)
        return [betas[m] for m in range(self.order)]

    def get_lam1(self) -> torch.Tensor:
        """λ₁ ∈ (0, 1), shape [N] (zeros when M = 0)."""
        if self.lam1_raw is not None:
            return torch.sigmoid(self.lam1_raw)
        if self.ap_lam_raw is not None:
            return torch.sigmoid(self.ap_lam_raw[0])
        return torch.zeros(self.out_features, device=self.f_raw.device, dtype=self.f_raw.dtype)

    def get_lam2(self) -> torch.Tensor:
        """λ₂ ∈ (0, 1), shape [N] (zeros when M < 2)."""
        if self.lam2_raw is not None:
            return torch.sigmoid(self.lam2_raw)
        if self.ap_lam_raw is not None and self.order >= 2:
            return torch.sigmoid(self.ap_lam_raw[1])
        return torch.zeros(self.out_features, device=self.f_raw.device, dtype=self.f_raw.dtype)

    def get_lam_list(self) -> list[torch.Tensor]:
        """Mixing weights [λ_1, ..., λ_M], each [N]."""
        if self.order == 0:
            return []
        if self.ap_lam_raw is not None:
            return [torch.sigmoid(self.ap_lam_raw[m]) for m in range(self.order)]
        lams = [self.get_lam1()]
        if self.order >= 2:
            lams.append(self.get_lam2())
        return lams

    # ── interpretable readouts ─────────────────────────────────────────────
    @torch.no_grad()
    def realized_freq(self) -> torch.Tensor:
        """
        Peak frequency (Hz) actually realized by the discrete-time FS update,
        f*_DT (Theorem 2).  Equals ``get_freq()`` for freq_param="dt"; for "ct"
        it differs slightly (paper Sec. 5.3).  This is the response of the FS
        pathway Y₀; with M ≥ 1 the TS output is fed back through V, so the
        closed-loop peak of the full neuron can differ.  NaN for neurons outside
        the stable region (κΔt² ≥ Jury bound), which have no stable peak.
        """
        if not self.implicit:
            raise NotImplementedError("realized_freq is derived for the semi-implicit update")
        kappa = self.get_kappa().double()
        f = dt_peak_freq(kappa, self.mu, self.rho, self.dt)
        unstable = ~(kappa * self.dt ** 2 < jury_bound(self.mu, self.rho, self.dt))
        return f.masked_fill(unstable, float("nan"))

    @torch.no_grad()
    def ts_group_delay(self, freq_hz=None) -> torch.Tensor:
        """
        Group delay (seconds) that the TS module adds to the FS pathway, i.e.
        -d arg G(e^{jω})/dω · Δt for the λ-mixed all-pass cascade G, evaluated
        per neuron at ``freq_hz`` ([N] tensor or scalar; default: each neuron's
        own f*).  Negative values are group-delay advances (Proposition 1).
        Returns zeros for M = 0.
        """
        f = self.get_freq() if freq_hz is None else torch.as_tensor(freq_hz, device=self.f_raw.device)
        f = f.to(torch.float64).expand(self.out_features)
        if self.order == 0:
            return torch.zeros_like(f)
        betas = [b.to(torch.float64) for b in self.get_beta_list()]
        lams = [l.to(torch.float64) for l in self.get_lam_list()]

        def response(w):
            zi = torch.exp(-1j * w)
            stages = [torch.ones_like(zi)]
            for b in betas:                       # A_m(z) = (z⁻¹ - β_m) / (1 - β_m z⁻¹)
                stages.append(stages[-1] * (zi - b) / (1 - b * zi))
            blend = stages[-1]
            for m in range(self.order - 1, -1, -1):
                blend = (1 - lams[m]) * stages[m] + lams[m] * blend
            return blend

        w = 2 * math.pi * f * self.dt
        h = 1e-5
        dphi = torch.angle(response(w + h) * torch.conj(response(w - h)))
        return -dphi / (2 * h) * self.dt

    # ── forward ────────────────────────────────────────────────────────────
    def _use_triton(self, x: torch.Tensor) -> bool:
        if self.backend == "torch":
            return False
        available = fused_fits_temporal is not None and x.is_cuda
        if self.backend == "triton" and not available:
            reason = ("input is not on CUDA" if _TRITON_IMPORT_ERROR is None
                      else f"Triton import failed: {_TRITON_IMPORT_ERROR}")
            raise RuntimeError(f"backend='triton' requested but {reason}")
        return available and self.order <= 5

    def forward(self, x: torch.Tensor,
                v_init=None, a_init=None,
                w1_init=None, w2_init=None):
        """
        Args:
            x       : [B, T, in_features]
            v_init  : [B, N] initial membrane potential (zeros if None).
            a_init  : [B, N] initial adaptation state (zeros if None).
            w1_init : [B, N] initial state of all-pass stage 1 (zeros if None).
            w2_init : [B, N] initial state of all-pass stage 2 (zeros if None).

        Returns:
            spikes   : [B, T, N]
            v_final  : [B, N] final membrane potential
            a_final  : [B, N] final adaptation state
            w1_final : [B, N] final state of all-pass stage 1 (zeros if M < 1)
            w2_final : [B, N] final state of all-pass stage 2 (zeros if M < 2)
        """
        B, T, _ = x.shape
        N        = self.out_features
        dev, dt_ = x.device, x.dtype

        _z = lambda: torch.zeros(B, N, device=dev, dtype=dt_)
        if v_init is None: v_init = _z()
        if a_init is None: a_init = _z()

        u          = self.linear(x)
        eta, gamma = self.get_eta_gamma()
        beta_list  = self.get_beta_list()
        dyn = dict(tau_m=self.tau_m, tau_a=self.tau_a, dt=self.dt,
                   v_threshold=self.v_threshold,
                   surrogate_alpha=self.surrogate_alpha,
                   surrogate_type=self.surrogate_type,
                   implicit=self.implicit)

        if not self._use_triton(x):
            w_inits = [_z() for _ in range(self.order)]
            if self.order >= 1 and w1_init is not None: w_inits[0] = w1_init
            if self.order >= 2 and w2_init is not None: w_inits[1] = w2_init
            spikes, _, _, v_f, a_f, w_f = fits_temporal_reference(
                u, v_init, a_init, eta, gamma, beta_list, self.get_lam_list(), w_inits, **dyn)
            w1_f = w_f[0] if self.order >= 1 else _z()
            w2_f = w_f[1] if self.order >= 2 else _z()
            return spikes, v_f, a_f, w1_f, w2_f

        spikes_all = []
        chunk_v    = v_init
        chunk_a    = a_init

        if self.order <= 2:
            chunk_w1 = _z() if w1_init is None else w1_init
            chunk_w2 = _z() if w2_init is None else w2_init
            lam1     = self.get_lam1()
            lam2     = self.get_lam2()

            for i in range(0, T, self.chunk_size):
                u_chunk = u[:, i : i + self.chunk_size, :]
                (spike_c, _v_seq, _a_seq,
                 chunk_v, chunk_a,
                 chunk_w1, chunk_w2) = fused_fits_temporal(
                    u_chunk, chunk_v, chunk_a,
                    eta, gamma, beta_list, lam1, lam2,
                    y1_init=chunk_w1, y2_init=chunk_w2,
                    order=self.order, **dyn)
                spikes_all.append(spike_c)

            return torch.cat(spikes_all, dim=1), chunk_v, chunk_a, chunk_w1, chunk_w2

        lam_list = self.get_lam_list()
        chunk_ws = [_z() for _ in range(self.order)]
        if w1_init is not None: chunk_ws[0] = w1_init
        if w2_init is not None: chunk_ws[1] = w2_init

        for i in range(0, T, self.chunk_size):
            u_chunk = u[:, i : i + self.chunk_size, :]
            (spike_c, _v_seq, _a_seq,
             chunk_v, chunk_a,
             chunk_ws) = fused_fits_temporal_high(
                u_chunk, chunk_v, chunk_a,
                eta, gamma, beta_list, lam_list, chunk_ws,
                order=self.order, **dyn)
            spikes_all.append(spike_c)

        return torch.cat(spikes_all, dim=1), chunk_v, chunk_a, chunk_ws[0], chunk_ws[1]

    def extra_repr(self) -> str:
        freq = self.get_freq().detach()
        mode = "semi-implicit" if self.implicit else "explicit"
        s = (f"in={self.in_features}, out={self.out_features}, order={self.order}, "
             f"mode={mode}, freq_param={self.freq_param}, freq_init={self.freq_init}, "
             f"learn_freq={self.learn_freq}, f=[{freq.min():.4f},{freq.max():.4f}]")
        if self.order >= 1:
            lam1 = self.get_lam1().detach()
            s += f", lam1=[{lam1.min():.3f},{lam1.max():.3f}]"
        if self.order >= 2:
            lam2 = self.get_lam2().detach()
            s += f", lam2=[{lam2.min():.3f},{lam2.max():.3f}]"
        return s


class LIFNeuron(FiTSNeuron):
    """
    Baselines of the paper's ablations, run through the same kernel (M = 0) so
    that the discretization is identical to FiTS:

        adaptive=False : plain LIF, η = γ = 0      ("Plain LIF")
        adaptive=True  : adaptive LIF with one fixed, shared κ = κ(Ω* = 0)
                         for every neuron          ("Adapt. LIF")

    Nothing but the linear projection is learned.
    """

    def __init__(self, in_features: int, out_features: int, adaptive: bool = False, **kwargs):
        for key in ("order", "learn_freq", "freq_init", "freq_param"):
            kwargs.pop(key, None)
        super().__init__(in_features, out_features, order=0, learn_freq=False, **kwargs)
        del self.f_raw
        self.adaptive = adaptive
        k0 = kappa_ct(torch.zeros(out_features), self.mu, self.rho)
        self.register_buffer("kappa_fixed", k0 if adaptive else torch.zeros(out_features))

    def get_freq(self) -> torch.Tensor:
        return torch.zeros_like(self.kappa_fixed)

    def get_kappa(self) -> torch.Tensor:
        return self.kappa_fixed

    def get_lam1(self) -> torch.Tensor:
        return torch.zeros_like(self.kappa_fixed)

    def get_lam2(self) -> torch.Tensor:
        return torch.zeros_like(self.kappa_fixed)

    def extra_repr(self) -> str:
        kind = "adaptive LIF" if self.adaptive else "LIF"
        return f"in={self.in_features}, out={self.out_features}, {kind}"
