"""
Target-frequency parameterization of the FS module.

FiTS learns each neuron's target frequency f* (Hz) and maps it in closed form
to the adaptation coupling κ = η·γ.  Two maps are provided:

* Continuous time (paper, Theorem 1) -- the default ``freq_param="ct"``:

      Ω* = sqrt( sqrt(κ(2ρ² + 2ρμ + κ)) - ρ² )                         (Eq. 4)
      κ  = ρ(ρ+μ)·[ sqrt(1 + (1 + (Ω*/ρ)²)² / (1 + μ/ρ)²) - 1 ]       (Eq. 5)

  f* is the peak of the continuous-time response |H(jΩ)|.  The implemented
  semi-implicit Euler update realizes a slightly different peak
  (paper Sec. 5.3); ``dt_peak_freq`` returns that realized value.

* Exact discrete time -- ``freq_param="dt"``: the closed-form inverse of
  Theorem 2, so the implemented update peaks exactly at f*.  With
  μ̄ = 1-μΔt, ρ̄ = 1-ρΔt, κ̄ = κΔt² and B = (1-ρ̄²)(1-μ̄ρ̄), the stationarity
  condition of |H_d(e^{jω})| is a quadratic in κ̄,

      ρ̄κ̄² + Bκ̄ - μ̄(2ρ̄cos ω* - 1 - ρ̄²)² = 0,

  whose unique non-negative root gives κ̄*.

Stability (semi-implicit update, Jury criterion):  κ̄ < (1 + μ̄)(1 + ρ̄).
κ_CT grows like Ω*², so an unconstrained f* can cross this bound during
training; ``critical_freq`` gives the crossing point (see fits/stability.py).
κ_DT stays below the bound up to Nyquist for the constants used in the paper.

Units: μ = 1/τ_m and ρ = 1/τ_a in s⁻¹, Ω* = 2π·f* in rad/s, κ in s⁻², Δt in s.
The discrete-time helpers assume the semi-implicit update (``implicit=True``).
"""

import math

import torch

__all__ = [
    "kappa_ct",
    "ct_peak_freq",
    "kappa_dt",
    "dt_peak_freq",
    "kappa_from_freq",
    "eta_gamma_from_kappa",
    "jury_bound",
    "critical_freq",
]


def _as_tensor(x):
    return x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Continuous time (Theorem 1)
# ─────────────────────────────────────────────────────────────────────────────
def kappa_ct(omega, mu: float, rho: float):
    """κ realizing the continuous-time target Ω* (rad/s), Eq. (5)."""
    term1 = (1 + (omega / rho) ** 2) ** 2
    term2 = (1 + mu / rho) ** 2
    return rho * (rho + mu) * (torch.sqrt(1 + term1 / term2) - 1)


def ct_peak_freq(kappa, mu: float, rho: float):
    """Continuous-time target frequency (Hz) of a given κ, Eq. (4); 0 if the peak is at DC."""
    kappa = _as_tensor(kappa)
    inner = torch.sqrt(kappa * (2 * rho ** 2 + 2 * rho * mu + kappa)) - rho ** 2
    return torch.sqrt(inner.clamp(min=0.0)) / (2 * math.pi)


# ─────────────────────────────────────────────────────────────────────────────
# Exact discrete time (inverse of Theorem 2, semi-implicit update)
# ─────────────────────────────────────────────────────────────────────────────
def _bars(mu, rho, dt):
    mu_b, rho_b = 1.0 - mu * dt, 1.0 - rho * dt
    B = (1.0 - rho_b ** 2) * (1.0 - mu_b * rho_b)
    return mu_b, rho_b, B


def kappa_dt(omega, mu: float, rho: float, dt: float):
    """
    κ whose semi-implicit update peaks exactly at Ω* (rad/s).

    The per-sample frequency ω* = Ω*·Δt is kept inside (0, π).
    """
    mu_b, rho_b, B = _bars(mu, rho, dt)
    w = (omega * dt).clamp(min=1e-6, max=math.pi - 1e-6)
    C = mu_b * (2.0 * rho_b * torch.cos(w) - 1.0 - rho_b ** 2) ** 2
    kap_b = (-B + torch.sqrt(B * B + 4.0 * rho_b * C)) / (2.0 * rho_b)
    return kap_b / (dt * dt)


def dt_peak_freq(kappa, mu: float, rho: float, dt: float):
    """
    Realized discrete-time target frequency f*_DT (Hz) of the semi-implicit FS
    update with coupling κ (the forward map of Theorem 2).  0 when the peak is at DC.
    """
    kappa = _as_tensor(kappa)
    mu_b, rho_b, B = _bars(mu, rho, dt)
    kap_b = kappa * dt * dt
    x = ((1.0 + rho_b ** 2) - torch.sqrt(kap_b * (B + kap_b * rho_b) / mu_b)) / (2.0 * rho_b)
    return torch.arccos(x.clamp(-1.0, 1.0)) / (2 * math.pi * dt)


# ─────────────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────────────
def kappa_from_freq(freq_hz, mu: float, rho: float, dt: float, param: str = "ct"):
    """κ for target frequencies f* in Hz, using the "ct" (paper) or "dt" (exact) map."""
    omega = 2.0 * math.pi * freq_hz
    if param == "ct":
        return kappa_ct(omega, mu, rho)
    if param == "dt":
        return kappa_dt(omega, mu, rho, dt)
    raise ValueError(f"freq_param must be 'ct' or 'dt', got {param!r}")


def eta_gamma_from_kappa(kappa, ratio: float = 1.0):
    """Split κ = η·γ with η/γ = ratio (the paper uses ratio = 1)."""
    return torch.sqrt(kappa * ratio), torch.sqrt(kappa / ratio)


def jury_bound(mu: float, rho: float, dt: float) -> float:
    """Stability bound on κ̄ = κΔt² for the semi-implicit update: κ̄ < (1+μ̄)(1+ρ̄)."""
    mu_b, rho_b, _ = _bars(mu, rho, dt)
    return (1.0 + mu_b) * (1.0 + rho_b)


def critical_freq(mu: float, rho: float, dt: float, param: str = "ct",
                  safety: float = 1.0) -> float:
    """
    Largest f* (Hz) whose κ stays at or below ``safety`` × the Jury bound.

    Returns the Nyquist frequency 1/(2Δt) if the whole band is admissible.
    """
    nyquist = 1.0 / (2.0 * dt)
    kappa_max = safety * jury_bound(mu, rho, dt) / (dt * dt)
    if param == "ct":
        f = float(ct_peak_freq(kappa_max, mu, rho))
    elif param == "dt":
        if float(kappa_dt(torch.tensor(math.pi / dt, dtype=torch.float64), mu, rho, dt)) <= kappa_max:
            return nyquist
        f = float(dt_peak_freq(kappa_max, mu, rho, dt))
    else:
        raise ValueError(f"param must be 'ct' or 'dt', got {param!r}")
    return min(f, nyquist)
