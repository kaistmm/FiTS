"""
Pure-PyTorch implementation of the FiTS temporal update.

This is a plain time-step loop of Algorithm 1 for any cascade order M >= 0.
It is used

  * as the CPU fallback of :class:`fits.FiTSNeuron` (``backend="torch"``),
  * for orders M > 5, which the fused Triton kernels do not cover,
  * as the numerical reference the Triton kernels are tested against
    (``tests/test_fits.py``).

It computes exactly the same recurrence as ``fits/kernels/fits_kernel.py``;
only the execution strategy differs (autograd over T Python steps instead of a
fused, hand-written backward pass), so it is roughly 30-135x slower on GPU.
"""

import math

import torch

__all__ = ["spike_fn", "fits_temporal_reference"]


class _SpikeFunc(torch.autograd.Function):
    """Heaviside forward, surrogate-gradient backward."""

    @staticmethod
    def forward(ctx, v_out, thresh, alpha, stype):
        spike = (v_out > thresh).to(v_out.dtype)
        ctx.save_for_backward(v_out)
        ctx.thresh = thresh
        ctx.alpha = alpha
        ctx.stype = int(stype)
        return spike

    @staticmethod
    def backward(ctx, grad_out):
        (v_out,) = ctx.saved_tensors
        x = v_out - ctx.thresh
        alpha = ctx.alpha
        if ctx.stype == 0:          # Triangle
            surr = alpha * torch.clamp(1.0 - alpha * x.abs(), min=0.0)
        elif ctx.stype == 1:        # Arctan
            surr = alpha / (2.0 * (1.0 + (math.pi / 2.0 * alpha * x) ** 2))
        else:                       # Sigmoid
            s = torch.sigmoid(alpha * x)
            surr = alpha * s * (1.0 - s)
        return grad_out * surr, None, None, None


def spike_fn(v_out, v_threshold=1.0, surrogate_alpha=2.0, surrogate_type=0):
    """Spike S = Θ(v_out - V_th) with a surrogate gradient (0=Triangle, 1=Arctan, 2=Sigmoid)."""
    return _SpikeFunc.apply(v_out, v_threshold, surrogate_alpha, surrogate_type)


def fits_temporal_reference(u, v_init, a_init,
                            eta, gamma,
                            beta_list, lam_list, w_inits,
                            *, tau_m, tau_a, dt,
                            v_threshold, surrogate_alpha, surrogate_type,
                            implicit):
    """
    Run the FiTS update over a whole sequence with M = len(beta_list) all-pass stages.

    Args:
        u          : [B, T, N] input current (after the linear projection).
        v_init     : [B, N]    initial membrane potential V.
        a_init     : [B, N]    initial adaptation state a.
        eta, gamma : [N]       FS coupling coefficients.
        beta_list  : M tensors [N], all-pass coefficients β_m ∈ (-1, 1).
        lam_list   : M tensors [N], mixing weights λ_m ∈ (0, 1).
        w_inits    : M tensors [B, N], initial all-pass states W_m.

    Returns:
        (spikes, v_seq, a_seq, v_final, a_final, w_finals)
        spikes : [B, T, N];  v_seq / a_seq : [B, T, N] post-reset V and a;
        v_final, a_final : [B, N];  w_finals : list of M tensors [B, N].
    """
    B, T, N = u.shape
    M = len(beta_list)

    decay_v  = 1.0 - dt / tau_m
    decay_b  = 1.0 - dt / tau_a
    eta_dt   = eta   * dt   # [N], broadcasts over [B, N]
    gamma_dt = gamma * dt

    v      = v_init
    a      = a_init
    w_list = list(w_inits)

    spikes_t, v_t, a_t = [], [], []

    for step in range(T):
        u_step = u[:, step, :]          # [B, N]

        # ── FS module ──────────────────────────────────────────────────────────
        v_raw = decay_v * v + (-eta_dt) * a + u_step

        if implicit:
            a_new = gamma_dt * v_raw + decay_b * a
        else:
            a_new = gamma_dt * v      + decay_b * a

        # ── TS all-pass cascade (TDF-II) ───────────────────────────────────────
        # y_stages[0]=Y₀=v_raw, y_stages[m]=Y_m after m stages
        y_prev  = v_raw
        y_stages = [v_raw]
        new_w   = []
        for m in range(M):
            y_m = -beta_list[m] * y_prev + w_list[m]
            w_m =  y_prev + beta_list[m] * y_m
            new_w.append(w_m)
            y_stages.append(y_m)
            y_prev = y_m

        # ── TS mixing (nested, innermost → outermost) ──────────────────────────
        # M=0: Ỹ=Y₀; M=1: Ỹ=(1-λ₁)Y₀+λ₁Y₁
        # General: blend = Y_M; for m=M-1..0: blend=(1-λ_{m+1})Y_m+λ_{m+1}·blend
        if M == 0:
            v_out = v_raw
        else:
            blend = y_stages[M]
            for m in range(M - 1, -1, -1):
                blend = (1.0 - lam_list[m]) * y_stages[m] + lam_list[m] * blend
            v_out = blend

        # ── Spike + soft reset (only V is reset; a and W_m are carried) ────────
        spike = _SpikeFunc.apply(v_out, v_threshold, surrogate_alpha, surrogate_type)
        v_new = v_out - spike * v_threshold

        spikes_t.append(spike)
        v_t.append(v_new)
        a_t.append(a_new)

        v      = v_new
        a      = a_new
        w_list = new_w

    spikes = torch.stack(spikes_t, dim=1)
    v_seq  = torch.stack(v_t,      dim=1)
    a_seq  = torch.stack(a_t,      dim=1)

    return spikes, v_seq, a_seq, v, a, w_list
