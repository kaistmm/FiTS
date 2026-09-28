"""
Fused Triton forward/backward kernels for the FiTS temporal update.

Implements Algorithm 1 of the paper for one temporal chunk ``u: [B, T, N]``
(the input current I = W·x + b, already computed for every timestep):

FS module (Eq. 6):
    Y₀[k+1] = (1 - μΔt)·V[k] - ηΔt·a[k] + I[k]
    a[k+1]  = (1 - ρΔt)·a[k] + γΔt·Y₀[k+1]     (IMPLICIT=True, semi-implicit Euler)
    a[k+1]  = (1 - ρΔt)·a[k] + γΔt·V[k]        (IMPLICIT=False, explicit Euler)

TS module, first-order all-pass stages in transposed direct form II (Eq. 9):
    Y_m[k+1] = -β_m·Y_{m-1}[k+1] + W_m[k]
    W_m[k+1] =  Y_{m-1}[k+1] + β_m·Y_m[k+1]            m = 1..M

TS λ-mixing, nested from the innermost stage outwards:
    M = 1:  Ỹ = (1-λ₁)·Y₀ + λ₁·Y₁
    M = 2:  Ỹ = (1-λ₁)·Y₀ + λ₁·[(1-λ₂)·Y₁ + λ₂·Y₂]
    M ≥ 3:  blend = Y_M;  blend = (1-λ_m)·Y_{m-1} + λ_m·blend  for m = M..1

Spike and subtractive soft reset (only V is reset; a and W_m are carried):
    S[k+1] = Θ(Ỹ[k+1] - V_th)
    V[k+1] = Ỹ[k+1] - S[k+1]·V_th

Parallelisation: one Triton program owns BLOCK_SIZE (batch, neuron) pairs and
runs the whole time loop in registers.  This is valid because every state
update is element-wise per neuron -- the only coupling between neurons is the
input current, which a feedforward layer receives for all T timesteps at once.
A recurrent weight matrix (I[k] depending on this layer's spikes S[k-1]) would
break that assumption.

Kernels:
    fits_fwd_kernel / fits_bwd_kernel             ORDER ∈ {0, 1, 2}
    fits_fwd_kernel_high / fits_bwd_kernel_high   ORDER ∈ {3, 4, 5}
Compile-time flags: IMPLICIT, ORDER, BLOCK_SIZE (tl.constexpr).
"""

import torch
import triton
import triton.language as tl
from fits.kernels.surrogate_kernels import select_surrogate_grad


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD KERNEL
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def fits_fwd_kernel(
    U_ptr, V_init_ptr, A_init_ptr, W1_init_ptr, W2_init_ptr,
    eta_ptr, gamma_ptr, beta1_ptr, beta2_ptr,
    lam1_ptr,           # [N]  λ₁ = sigmoid(lam1_raw), ORDER >= 1
    lam2_ptr,           # [N]  λ₂ = sigmoid(lam2_raw), ORDER == 2
    decay_v, decay_b, dt, thresh,
    Spike_ptr, V_seq_ptr, A_seq_ptr,
    Vraw_seq_ptr,       # [B, T, N] pre-reset FS output Y₀, always stored
    W1_seq_ptr, W2_seq_ptr,
    V_final_ptr, A_final_ptr, W1_final_ptr, W2_final_ptr,
    B, T, N,
    stride_ub, stride_ut, stride_un,
    stride_sb, stride_st, stride_sn,
    stride_eta, stride_gamma,
    surrogate_type: tl.constexpr,
    IMPLICIT: tl.constexpr,
    ORDER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < (B * N)
    n     = offs % N
    b_idx = offs // N
    idx_n = b_idx * N + n

    v = tl.load(V_init_ptr + idx_n, mask=mask, other=0.0)
    a = tl.load(A_init_ptr + idx_n, mask=mask, other=0.0)

    eta      = tl.load(eta_ptr   + n * stride_eta,   mask=mask, other=0.0)
    gamma    = tl.load(gamma_ptr + n * stride_gamma, mask=mask, other=0.0)
    eta_dt   = eta   * dt
    gamma_dt = gamma * dt

    if ORDER >= 1:
        beta1 = tl.load(beta1_ptr + n, mask=mask, other=0.0)
        lam1  = tl.load(lam1_ptr  + n, mask=mask, other=0.0)
        w1    = tl.load(W1_init_ptr + idx_n, mask=mask, other=0.0)
    if ORDER >= 2:
        beta2 = tl.load(beta2_ptr + n, mask=mask, other=0.0)
        lam2  = tl.load(lam2_ptr  + n, mask=mask, other=0.0)
        w2    = tl.load(W2_init_ptr + idx_n, mask=mask, other=0.0)

    for t in range(T):
        u_idx = b_idx * stride_ub + t * stride_ut + n * stride_un
        u = tl.load(U_ptr + u_idx, mask=mask, other=0.0)

        # ── FS module ────────────────────────────────────────────────────
        v_k   = v
        v_raw = decay_v * v_k + (-eta_dt) * a + u

        if IMPLICIT:
            # Semi-implicit: adaptation uses updated Y₀
            a_new = gamma_dt * v_raw + decay_b * a
        else:
            # Explicit: adaptation uses previous V[k]
            a_new = gamma_dt * v_k + decay_b * a

        # ── TS module — all-pass cascade (TDF-II) ────────────────────────
        if ORDER == 0:
            v_out = v_raw

        if ORDER >= 1:
            w1_k   = w1
            y1_new = -beta1 * v_raw + w1_k
            w1_new =  v_raw + beta1 * y1_new

        if ORDER == 1:
            v_out = (1.0 - lam1) * v_raw + lam1 * y1_new

        if ORDER >= 2:
            w2_k   = w2
            y2_new = -beta2 * y1_new + w2_k
            w2_new =  y1_new + beta2 * y2_new

        if ORDER == 2:
            ap_blend = (1.0 - lam2) * y1_new + lam2 * y2_new
            v_out    = (1.0 - lam1) * v_raw   + lam1 * ap_blend

        # ── Spike generation and soft reset ──────────────────────────────
        spike = (v_out > thresh).to(tl.float32)
        v_new = v_out - spike * thresh

        # ── Store ─────────────────────────────────────────────────────────
        out_idx = b_idx * stride_sb + t * stride_st + n * stride_sn
        tl.store(Spike_ptr    + out_idx, spike,  mask=mask)
        tl.store(V_seq_ptr    + out_idx, v_new,  mask=mask)
        tl.store(A_seq_ptr    + out_idx, a_new,  mask=mask)
        tl.store(Vraw_seq_ptr + out_idx, v_raw,  mask=mask)
        if ORDER >= 1:
            tl.store(W1_seq_ptr + out_idx, w1_new, mask=mask)
        if ORDER >= 2:
            tl.store(W2_seq_ptr + out_idx, w2_new, mask=mask)

        v = v_new; a = a_new
        if ORDER >= 1:
            w1 = w1_new
        if ORDER >= 2:
            w2 = w2_new

    fin_idx = b_idx * N + n
    tl.store(V_final_ptr + fin_idx, v, mask=mask)
    tl.store(A_final_ptr + fin_idx, a, mask=mask)
    if ORDER >= 1:
        tl.store(W1_final_ptr + fin_idx, w1, mask=mask)
    if ORDER >= 2:
        tl.store(W2_final_ptr + fin_idx, w2, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD KERNEL
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def fits_bwd_kernel(
    Grad_Spike_ptr, Grad_V_seq_ptr, Grad_A_seq_ptr,
    Grad_V_final_ptr, Grad_A_final_ptr,
    Grad_W1_final_ptr, Grad_W2_final_ptr,
    V_init_ptr, A_init_ptr, W1_init_ptr, W2_init_ptr,
    Spike_ptr, V_seq_ptr, A_seq_ptr,
    Vraw_seq_ptr, W1_seq_ptr, W2_seq_ptr,
    eta_ptr, gamma_ptr, beta1_ptr, beta2_ptr,
    lam1_ptr, lam2_ptr,
    decay_v, decay_b, dt, thresh, alpha_surr,
    Grad_U_ptr,
    Grad_V_init_ptr, Grad_A_init_ptr,
    Grad_W1_init_ptr, Grad_W2_init_ptr,
    Grad_eta_ptr, Grad_gamma_ptr,
    Grad_beta1_ptr, Grad_beta2_ptr,
    Grad_lam1_ptr, Grad_lam2_ptr,
    B, T, N,
    stride_ub, stride_ut, stride_un,
    stride_sb, stride_st, stride_sn,
    stride_eta, stride_gamma,
    stride_geta, stride_ggamma,
    surrogate_type: tl.constexpr,
    IMPLICIT: tl.constexpr,
    ORDER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < (B * N)
    n     = offs % N
    b_idx = offs // N
    idx_n = b_idx * N + n

    eta      = tl.load(eta_ptr   + n * stride_eta,   mask=mask, other=0.0)
    gamma    = tl.load(gamma_ptr + n * stride_gamma, mask=mask, other=0.0)
    eta_dt   = eta   * dt
    gamma_dt = gamma * dt
    if ORDER >= 1:
        beta1 = tl.load(beta1_ptr + n, mask=mask, other=0.0)
        lam1  = tl.load(lam1_ptr  + n, mask=mask, other=0.0)
    if ORDER >= 2:
        beta2 = tl.load(beta2_ptr + n, mask=mask, other=0.0)
        lam2  = tl.load(lam2_ptr  + n, mask=mask, other=0.0)

    fin_idx = b_idx * N + n
    grad_v = tl.load(Grad_V_final_ptr + fin_idx, mask=mask, other=0.0)
    grad_a = tl.load(Grad_A_final_ptr + fin_idx, mask=mask, other=0.0)
    if ORDER >= 1:
        grad_w1 = tl.load(Grad_W1_final_ptr + fin_idx, mask=mask, other=0.0)
    else:
        grad_w1 = 0.0
    if ORDER >= 2:
        grad_w2 = tl.load(Grad_W2_final_ptr + fin_idx, mask=mask, other=0.0)
    else:
        grad_w2 = 0.0

    d_eta_accum   = 0.0 * eta
    d_gamma_accum = 0.0 * gamma
    if ORDER >= 1:
        d_beta1_accum = 0.0 * eta
        d_lam1_accum  = 0.0 * eta
    if ORDER >= 2:
        d_beta2_accum = 0.0 * eta
        d_lam2_accum  = 0.0 * eta

    for t in range(T - 1, -1, -1):
        out_idx = b_idx * stride_sb + t * stride_st + n * stride_sn

        spike      = tl.load(Spike_ptr    + out_idx, mask=mask, other=0.0)
        v_t        = tl.load(V_seq_ptr    + out_idx, mask=mask, other=0.0)
        a_t        = tl.load(A_seq_ptr    + out_idx, mask=mask, other=0.0)
        v_raw_t    = tl.load(Vraw_seq_ptr + out_idx, mask=mask, other=0.0)
        g_spike_in = tl.load(Grad_Spike_ptr + out_idx, mask=mask, other=0.0)
        g_v_in     = tl.load(Grad_V_seq_ptr + out_idx, mask=mask, other=0.0)
        g_a_in     = tl.load(Grad_A_seq_ptr + out_idx, mask=mask, other=0.0)

        grad_v += g_v_in
        grad_a += g_a_in

        v_out     = v_t + spike * thresh
        surr      = select_surrogate_grad(v_out - thresh, alpha_surr, surrogate_type)
        g_s_total = g_spike_in + grad_v * (-thresh)
        d_vmix    = grad_v + g_s_total * surr

        prev_idx = b_idx * stride_sb + (t - 1) * stride_st + n * stride_sn
        if t > 0:
            v_prev = tl.load(V_seq_ptr + prev_idx, mask=mask, other=0.0)
            a_prev = tl.load(A_seq_ptr + prev_idx, mask=mask, other=0.0)
        else:
            v_prev = tl.load(V_init_ptr + idx_n, mask=mask, other=0.0)
            a_prev = tl.load(A_init_ptr + idx_n, mask=mask, other=0.0)

        if ORDER >= 1:
            if t > 0:
                w1_prev = tl.load(W1_seq_ptr + prev_idx, mask=mask, other=0.0)
            else:
                w1_prev = tl.load(W1_init_ptr + idx_n, mask=mask, other=0.0)
            y1_t = -beta1 * v_raw_t + w1_prev

        if ORDER >= 2:
            if t > 0:
                w2_prev = tl.load(W2_seq_ptr + prev_idx, mask=mask, other=0.0)
            else:
                w2_prev = tl.load(W2_init_ptr + idx_n, mask=mask, other=0.0)
            y2_t = -beta2 * y1_t + w2_prev

        # ── Backward through TS mixing + AP cascade ───────────────────────
        if ORDER == 0:
            d_vraw      = d_vmix
            d_v_from_ap = 0.0

        if ORDER == 1:
            d_lam1_accum += d_vmix * (y1_t - v_raw_t)
            d_v_ap        = d_vmix * lam1
            d_vraw_mix    = d_vmix * (1.0 - lam1)

            dy1_tot = d_v_ap + beta1 * grad_w1
            d_vraw  = dy1_tot * (-beta1) + grad_w1 + d_vraw_mix
            d_beta1_accum += dy1_tot * (-v_raw_t) + grad_w1 * y1_t
            grad_w1 = dy1_tot
            d_v_from_ap = 0.0

        if ORDER == 2:
            ap_blend_t   = (1.0 - lam2) * y1_t + lam2 * y2_t

            d_lam1_accum += d_vmix * (ap_blend_t - v_raw_t)
            d_ap_blend    = d_vmix * lam1
            d_vraw_mix    = d_vmix * (1.0 - lam1)

            d_lam2_accum  += d_ap_blend * (y2_t - y1_t)
            d_y2_from_lam  = d_ap_blend * lam2
            d_y1_from_lam  = d_ap_blend * (1.0 - lam2)

            dy2_tot    = d_y2_from_lam + beta2 * grad_w2
            dy1_from_2 = dy2_tot * (-beta2) + grad_w2
            d_beta2_accum += dy2_tot * (-y1_t) + grad_w2 * y2_t
            grad_w2 = dy2_tot

            dy1_tot = dy1_from_2 + d_y1_from_lam + beta1 * grad_w1
            d_vraw  = dy1_tot * (-beta1) + grad_w1 + d_vraw_mix
            d_beta1_accum += dy1_tot * (-v_raw_t) + grad_w1 * y1_t
            grad_w1 = dy1_tot
            d_v_from_ap = 0.0

        # ── Backward through FS module ────────────────────────────────────
        if IMPLICIT:
            # Semi-implicit: a[k+1] = γΔt·Y₀[k+1] + decay_b·a[k]
            d_gamma_accum += grad_a * (dt * v_raw_t)
            d_vraw_tot     = d_vraw + grad_a * gamma_dt
            grad_a_tot     = grad_a * decay_b

            d_eta_accum += d_vraw_tot * (-dt * a_prev)
            u_idx = b_idx * stride_ub + t * stride_ut + n * stride_un
            tl.store(Grad_U_ptr + u_idx, d_vraw_tot, mask=mask)

            d_v_from_fs = d_vraw_tot * decay_v
            d_a_from_fs = grad_a_tot + d_vraw_tot * (-eta_dt)
        else:
            # Explicit: a[k+1] = γΔt·V[k] + decay_b·a[k]
            grad_a_tot    = grad_a * decay_b

            u_idx = b_idx * stride_ub + t * stride_ut + n * stride_un
            tl.store(Grad_U_ptr + u_idx, d_vraw, mask=mask)

            d_eta_accum   += d_vraw * (-dt * a_prev)
            d_gamma_accum += grad_a * (dt * v_prev)

            d_v_from_fs = d_vraw * decay_v + grad_a * gamma_dt
            d_a_from_fs = grad_a_tot       + d_vraw * (-eta_dt)

        grad_v = d_v_from_fs + (d_v_from_ap if ORDER >= 1 else 0.0 * d_v_from_fs)
        grad_a = d_a_from_fs

    tl.store(Grad_V_init_ptr + idx_n, grad_v, mask=mask)
    tl.store(Grad_A_init_ptr + idx_n, grad_a, mask=mask)
    if ORDER >= 1:
        tl.store(Grad_W1_init_ptr + idx_n, grad_w1, mask=mask)
    if ORDER >= 2:
        tl.store(Grad_W2_init_ptr + idx_n, grad_w2, mask=mask)

    tl.store(Grad_eta_ptr   + n * stride_geta   + b_idx, d_eta_accum,   mask=mask)
    tl.store(Grad_gamma_ptr + n * stride_ggamma + b_idx, d_gamma_accum, mask=mask)
    if ORDER >= 1:
        tl.store(Grad_beta1_ptr + n * stride_geta + b_idx, d_beta1_accum, mask=mask)
        tl.store(Grad_lam1_ptr  + n * stride_geta + b_idx, d_lam1_accum,  mask=mask)
    if ORDER >= 2:
        tl.store(Grad_beta2_ptr + n * stride_geta + b_idx, d_beta2_accum, mask=mask)
        tl.store(Grad_lam2_ptr  + n * stride_geta + b_idx, d_lam2_accum,  mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _dummy1(device, dtype):
    return torch.zeros(1, device=device, dtype=dtype)

def _zeros_n(N, device, dtype):
    return torch.zeros(N, device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Shared forward / backward implementation
# ─────────────────────────────────────────────────────────────────────────────
def _fits_forward(ctx, ORDER, IMPLICIT,
                  u, v_init, a_init, w1_init, w2_init,
                  eta, gamma, beta1, beta2, lam1, lam2,
                  tau_m, tau_a, dt, thresh,
                  surrogate_alpha, surrogate_type):
    B, T, N = u.shape
    decay_v = 1.0 - dt / tau_m
    decay_b = 1.0 - dt / tau_a

    for t in [u, v_init, a_init, w1_init, w2_init,
              eta, gamma, beta1, beta2, lam1, lam2]:
        t = t.contiguous() if hasattr(t, 'contiguous') else t

    u        = u.contiguous()
    v_init   = v_init.contiguous()
    a_init   = a_init.contiguous()
    w1_init  = w1_init.contiguous()
    w2_init  = w2_init.contiguous()
    eta      = eta.contiguous()
    gamma    = gamma.contiguous()
    beta1    = beta1.contiguous()
    beta2    = beta2.contiguous()
    lam1     = lam1.contiguous()
    lam2     = lam2.contiguous()

    spike    = torch.empty_like(u)
    v_seq    = torch.empty_like(u)
    a_seq    = torch.empty_like(u)
    vraw_seq = torch.empty_like(u)
    w1_seq   = torch.empty_like(u) if ORDER >= 1 else _dummy1(u.device, u.dtype)
    w2_seq   = torch.empty_like(u) if ORDER >= 2 else _dummy1(u.device, u.dtype)
    v_final  = torch.empty_like(v_init)
    a_final  = torch.empty_like(a_init)
    w1_final = torch.empty_like(w1_init)
    w2_final = torch.empty_like(w2_init)

    grid = lambda meta: (triton.cdiv(B * N, meta['BLOCK_SIZE']),)
    fits_fwd_kernel[grid](
        u, v_init, a_init, w1_init, w2_init,
        eta, gamma, beta1, beta2, lam1, lam2,
        decay_v, decay_b, dt, thresh,
        spike, v_seq, a_seq, vraw_seq, w1_seq, w2_seq,
        v_final, a_final, w1_final, w2_final,
        B, T, N,
        u.stride(0), u.stride(1), u.stride(2),
        spike.stride(0), spike.stride(1), spike.stride(2),
        eta.stride(0), gamma.stride(0),
        surrogate_type,
        IMPLICIT=IMPLICIT, ORDER=ORDER, BLOCK_SIZE=256,
    )

    saved = [spike, v_seq, a_seq, vraw_seq,
             v_init, a_init, w1_init, w2_init,
             eta, gamma, beta1, beta2, lam1, lam2]
    if ORDER >= 1: saved += [w1_seq]
    if ORDER >= 2: saved += [w2_seq]
    ctx.save_for_backward(*saved)
    ctx.params = (decay_v, decay_b, dt, thresh, surrogate_alpha,
                  surrogate_type, B, T, N, ORDER, IMPLICIT)

    return spike, v_seq, a_seq, v_final, a_final, w1_final, w2_final


def _fits_backward(ctx, ORDER, IMPLICIT,
                   grad_spike, grad_v_seq, grad_a_seq,
                   grad_v_final, grad_a_final,
                   grad_w1_final, grad_w2_final):
    (decay_v, decay_b, dt, thresh, surrogate_alpha,
     surrogate_type, B, T, N, _, _impl) = ctx.params

    saved = ctx.saved_tensors
    (spike, v_seq, a_seq, vraw_seq,
     v_init, a_init, w1_init, w2_init,
     eta, gamma, beta1, beta2, lam1, lam2) = saved[:14]
    idx = 14
    w1_seq = _dummy1(v_seq.device, v_seq.dtype)
    w2_seq = _dummy1(v_seq.device, v_seq.dtype)
    if ORDER >= 1: w1_seq = saved[idx]; idx += 1
    if ORDER >= 2: w2_seq = saved[idx]

    def _z(g, ref): return torch.zeros_like(ref) if g is None else g.contiguous()
    grad_spike    = _z(grad_spike,    spike)
    grad_v_seq    = _z(grad_v_seq,    v_seq)
    grad_a_seq    = _z(grad_a_seq,    a_seq)
    grad_v_final  = _z(grad_v_final,  v_init)
    grad_a_final  = _z(grad_a_final,  a_init)
    grad_w1_final = _z(grad_w1_final, w1_init)
    grad_w2_final = _z(grad_w2_final, w2_init)

    grad_u       = torch.empty_like(spike)
    grad_v_init  = torch.empty_like(v_init)
    grad_a_init  = torch.empty_like(a_init)
    grad_w1_init = torch.zeros_like(w1_init)
    grad_w2_init = torch.zeros_like(w2_init)

    grad_eta_raw   = torch.zeros(N, B, device=spike.device, dtype=spike.dtype)
    grad_gamma_raw = torch.zeros(N, B, device=spike.device, dtype=spike.dtype)
    grad_beta1_raw = torch.zeros(N, B, device=spike.device, dtype=spike.dtype)
    grad_beta2_raw = torch.zeros(N, B, device=spike.device, dtype=spike.dtype)
    grad_lam1_raw  = torch.zeros(N, B, device=spike.device, dtype=spike.dtype)
    grad_lam2_raw  = torch.zeros(N, B, device=spike.device, dtype=spike.dtype)

    grid = lambda meta: (triton.cdiv(B * N, meta['BLOCK_SIZE']),)
    fits_bwd_kernel[grid](
        grad_spike, grad_v_seq, grad_a_seq,
        grad_v_final, grad_a_final, grad_w1_final, grad_w2_final,
        v_init, a_init, w1_init, w2_init,
        spike, v_seq, a_seq, vraw_seq, w1_seq, w2_seq,
        eta, gamma, beta1, beta2, lam1, lam2,
        decay_v, decay_b, dt, thresh, surrogate_alpha,
        grad_u, grad_v_init, grad_a_init, grad_w1_init, grad_w2_init,
        grad_eta_raw, grad_gamma_raw, grad_beta1_raw, grad_beta2_raw,
        grad_lam1_raw, grad_lam2_raw,
        B, T, N,
        spike.stride(0), spike.stride(1), spike.stride(2),
        spike.stride(0), spike.stride(1), spike.stride(2),
        eta.stride(0), gamma.stride(0),
        grad_eta_raw.stride(0), grad_gamma_raw.stride(0),
        surrogate_type,
        IMPLICIT=IMPLICIT, ORDER=ORDER, BLOCK_SIZE=256,
    )

    grad_eta   = grad_eta_raw.sum(1)
    grad_gamma = grad_gamma_raw.sum(1)
    grad_beta1 = grad_beta1_raw.sum(1) if ORDER >= 1 else None
    grad_beta2 = grad_beta2_raw.sum(1) if ORDER >= 2 else None
    grad_lam1  = grad_lam1_raw.sum(1)  if ORDER >= 1 else None
    grad_lam2  = grad_lam2_raw.sum(1)  if ORDER >= 2 else None

    # Matches forward signature:
    # u, v_init, a_init, w1_init, w2_init,
    # eta, gamma, beta1, beta2, lam1, lam2,
    # tau_m, tau_a, dt, thresh, s_alpha, s_type
    return (grad_u, grad_v_init, grad_a_init, grad_w1_init, grad_w2_init,
            grad_eta, grad_gamma, grad_beta1, grad_beta2, grad_lam1, grad_lam2,
            None, None, None, None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Autograd Functions (one per (ORDER, IMPLICIT) combination)
# ─────────────────────────────────────────────────────────────────────────────
def _make_fits_func(order: int, implicit: bool):
    class _Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, v_init, a_init, w1_init, w2_init,
                    eta, gamma, beta1, beta2, lam1, lam2,
                    tau_m, tau_a, dt, thresh, surrogate_alpha, surrogate_type):
            return _fits_forward(
                ctx, order, implicit,
                u, v_init, a_init, w1_init, w2_init,
                eta, gamma, beta1, beta2, lam1, lam2,
                tau_m, tau_a, dt, thresh,
                surrogate_alpha, surrogate_type)

        @staticmethod
        def backward(ctx, grad_spike, grad_v_seq, grad_a_seq,
                     grad_v_final, grad_a_final, grad_w1_final, grad_w2_final):
            return _fits_backward(
                ctx, order, implicit,
                grad_spike, grad_v_seq, grad_a_seq,
                grad_v_final, grad_a_final,
                grad_w1_final, grad_w2_final)

    mode = "Implicit" if implicit else "Explicit"
    _Func.__name__ = f'FiTSFunc_{mode}_N{order}'
    return _Func


_FUNC_TABLE = {
    (order, implicit): _make_fits_func(order, implicit)
    for order in (0, 1, 2)
    for implicit in (True, False)
}


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-ORDER TRITON KERNELS  (ORDER = 3, 4, 5)
# M=0..2 kernels above are untouched; new kernels handle M=3,4,5.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def fits_fwd_kernel_high(
    U_ptr, V_init_ptr, A_init_ptr,
    W1_init_ptr, W2_init_ptr, W3_init_ptr, W4_init_ptr, W5_init_ptr,
    eta_ptr, gamma_ptr,
    beta1_ptr, beta2_ptr, beta3_ptr, beta4_ptr, beta5_ptr,
    lam1_ptr, lam2_ptr, lam3_ptr, lam4_ptr, lam5_ptr,
    decay_v, decay_b, dt, thresh,
    Spike_ptr, V_seq_ptr, A_seq_ptr, Vraw_seq_ptr,
    W1_seq_ptr, W2_seq_ptr, W3_seq_ptr, W4_seq_ptr, W5_seq_ptr,
    V_final_ptr, A_final_ptr,
    W1_final_ptr, W2_final_ptr, W3_final_ptr, W4_final_ptr, W5_final_ptr,
    B, T, N,
    stride_ub, stride_ut, stride_un,
    stride_sb, stride_st, stride_sn,
    stride_eta, stride_gamma,
    surrogate_type: tl.constexpr,
    IMPLICIT: tl.constexpr,
    ORDER: tl.constexpr,    # 3, 4, or 5
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < (B * N)
    n     = offs % N
    b_idx = offs // N
    idx_n = b_idx * N + n

    v = tl.load(V_init_ptr + idx_n, mask=mask, other=0.0)
    a = tl.load(A_init_ptr + idx_n, mask=mask, other=0.0)

    eta      = tl.load(eta_ptr   + n * stride_eta,   mask=mask, other=0.0)
    gamma    = tl.load(gamma_ptr + n * stride_gamma, mask=mask, other=0.0)
    eta_dt   = eta   * dt
    gamma_dt = gamma * dt

    beta1 = tl.load(beta1_ptr + n, mask=mask, other=0.0)
    lam1  = tl.load(lam1_ptr  + n, mask=mask, other=0.0)
    w1    = tl.load(W1_init_ptr + idx_n, mask=mask, other=0.0)
    beta2 = tl.load(beta2_ptr + n, mask=mask, other=0.0)
    lam2  = tl.load(lam2_ptr  + n, mask=mask, other=0.0)
    w2    = tl.load(W2_init_ptr + idx_n, mask=mask, other=0.0)
    beta3 = tl.load(beta3_ptr + n, mask=mask, other=0.0)
    lam3  = tl.load(lam3_ptr  + n, mask=mask, other=0.0)
    w3    = tl.load(W3_init_ptr + idx_n, mask=mask, other=0.0)
    if ORDER >= 4:
        beta4 = tl.load(beta4_ptr + n, mask=mask, other=0.0)
        lam4  = tl.load(lam4_ptr  + n, mask=mask, other=0.0)
        w4    = tl.load(W4_init_ptr + idx_n, mask=mask, other=0.0)
    if ORDER >= 5:
        beta5 = tl.load(beta5_ptr + n, mask=mask, other=0.0)
        lam5  = tl.load(lam5_ptr  + n, mask=mask, other=0.0)
        w5    = tl.load(W5_init_ptr + idx_n, mask=mask, other=0.0)

    for t in range(T):
        u_idx = b_idx * stride_ub + t * stride_ut + n * stride_un
        u = tl.load(U_ptr + u_idx, mask=mask, other=0.0)

        # ── FS module ────────────────────────────────────────────────────
        v_raw = decay_v * v + (-eta_dt) * a + u
        if IMPLICIT:
            a_new = gamma_dt * v_raw + decay_b * a
        else:
            a_new = gamma_dt * v      + decay_b * a

        # ── TS all-pass cascade (TDF-II) ─────────────────────────────────
        y1_new = -beta1 * v_raw  + w1;  w1_new = v_raw  + beta1 * y1_new
        y2_new = -beta2 * y1_new + w2;  w2_new = y1_new + beta2 * y2_new
        y3_new = -beta3 * y2_new + w3;  w3_new = y2_new + beta3 * y3_new
        if ORDER >= 4:
            y4_new = -beta4 * y3_new + w4;  w4_new = y3_new + beta4 * y4_new
        if ORDER >= 5:
            y5_new = -beta5 * y4_new + w5;  w5_new = y4_new + beta5 * y5_new

        # ── TS mixing (nested, innermost → outermost) ────────────────────
        if ORDER == 3:
            blend = (1.0 - lam3) * y2_new + lam3 * y3_new
            blend = (1.0 - lam2) * y1_new + lam2 * blend
            v_out = (1.0 - lam1) * v_raw  + lam1 * blend
        if ORDER == 4:
            blend = (1.0 - lam4) * y3_new + lam4 * y4_new
            blend = (1.0 - lam3) * y2_new + lam3 * blend
            blend = (1.0 - lam2) * y1_new + lam2 * blend
            v_out = (1.0 - lam1) * v_raw  + lam1 * blend
        if ORDER == 5:
            blend = (1.0 - lam5) * y4_new + lam5 * y5_new
            blend = (1.0 - lam4) * y3_new + lam4 * blend
            blend = (1.0 - lam3) * y2_new + lam3 * blend
            blend = (1.0 - lam2) * y1_new + lam2 * blend
            v_out = (1.0 - lam1) * v_raw  + lam1 * blend

        # ── Spike + soft reset ───────────────────────────────────────────
        spike = (v_out > thresh).to(tl.float32)
        v_new = v_out - spike * thresh

        out_idx = b_idx * stride_sb + t * stride_st + n * stride_sn
        tl.store(Spike_ptr    + out_idx, spike,  mask=mask)
        tl.store(V_seq_ptr    + out_idx, v_new,  mask=mask)
        tl.store(A_seq_ptr    + out_idx, a_new,  mask=mask)
        tl.store(Vraw_seq_ptr + out_idx, v_raw,  mask=mask)
        tl.store(W1_seq_ptr   + out_idx, w1_new, mask=mask)
        tl.store(W2_seq_ptr   + out_idx, w2_new, mask=mask)
        tl.store(W3_seq_ptr   + out_idx, w3_new, mask=mask)
        if ORDER >= 4:
            tl.store(W4_seq_ptr + out_idx, w4_new, mask=mask)
        if ORDER >= 5:
            tl.store(W5_seq_ptr + out_idx, w5_new, mask=mask)

        v = v_new; a = a_new
        w1 = w1_new; w2 = w2_new; w3 = w3_new
        if ORDER >= 4:
            w4 = w4_new
        if ORDER >= 5:
            w5 = w5_new

    fin_idx = b_idx * N + n
    tl.store(V_final_ptr  + fin_idx, v,  mask=mask)
    tl.store(A_final_ptr  + fin_idx, a,  mask=mask)
    tl.store(W1_final_ptr + fin_idx, w1, mask=mask)
    tl.store(W2_final_ptr + fin_idx, w2, mask=mask)
    tl.store(W3_final_ptr + fin_idx, w3, mask=mask)
    if ORDER >= 4:
        tl.store(W4_final_ptr + fin_idx, w4, mask=mask)
    if ORDER >= 5:
        tl.store(W5_final_ptr + fin_idx, w5, mask=mask)


@triton.jit
def fits_bwd_kernel_high(
    Grad_Spike_ptr, Grad_V_seq_ptr, Grad_A_seq_ptr,
    Grad_V_final_ptr, Grad_A_final_ptr,
    Grad_W1_final_ptr, Grad_W2_final_ptr,
    Grad_W3_final_ptr, Grad_W4_final_ptr, Grad_W5_final_ptr,
    V_init_ptr, A_init_ptr,
    W1_init_ptr, W2_init_ptr, W3_init_ptr, W4_init_ptr, W5_init_ptr,
    Spike_ptr, V_seq_ptr, A_seq_ptr, Vraw_seq_ptr,
    W1_seq_ptr, W2_seq_ptr, W3_seq_ptr, W4_seq_ptr, W5_seq_ptr,
    eta_ptr, gamma_ptr,
    beta1_ptr, beta2_ptr, beta3_ptr, beta4_ptr, beta5_ptr,
    lam1_ptr, lam2_ptr, lam3_ptr, lam4_ptr, lam5_ptr,
    decay_v, decay_b, dt, thresh, alpha_surr,
    Grad_U_ptr,
    Grad_V_init_ptr, Grad_A_init_ptr,
    Grad_W1_init_ptr, Grad_W2_init_ptr,
    Grad_W3_init_ptr, Grad_W4_init_ptr, Grad_W5_init_ptr,
    Grad_eta_ptr, Grad_gamma_ptr,
    Grad_beta1_ptr, Grad_beta2_ptr, Grad_beta3_ptr, Grad_beta4_ptr, Grad_beta5_ptr,
    Grad_lam1_ptr,  Grad_lam2_ptr,  Grad_lam3_ptr,  Grad_lam4_ptr,  Grad_lam5_ptr,
    B, T, N,
    stride_ub, stride_ut, stride_un,
    stride_sb, stride_st, stride_sn,
    stride_eta, stride_gamma,
    stride_geta, stride_ggamma,
    surrogate_type: tl.constexpr,
    IMPLICIT: tl.constexpr,
    ORDER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < (B * N)
    n     = offs % N
    b_idx = offs // N
    idx_n = b_idx * N + n

    eta      = tl.load(eta_ptr   + n * stride_eta,   mask=mask, other=0.0)
    gamma    = tl.load(gamma_ptr + n * stride_gamma, mask=mask, other=0.0)
    eta_dt   = eta   * dt
    gamma_dt = gamma * dt

    beta1 = tl.load(beta1_ptr + n, mask=mask, other=0.0)
    lam1  = tl.load(lam1_ptr  + n, mask=mask, other=0.0)
    beta2 = tl.load(beta2_ptr + n, mask=mask, other=0.0)
    lam2  = tl.load(lam2_ptr  + n, mask=mask, other=0.0)
    beta3 = tl.load(beta3_ptr + n, mask=mask, other=0.0)
    lam3  = tl.load(lam3_ptr  + n, mask=mask, other=0.0)
    if ORDER >= 4:
        beta4 = tl.load(beta4_ptr + n, mask=mask, other=0.0)
        lam4  = tl.load(lam4_ptr  + n, mask=mask, other=0.0)
    if ORDER >= 5:
        beta5 = tl.load(beta5_ptr + n, mask=mask, other=0.0)
        lam5  = tl.load(lam5_ptr  + n, mask=mask, other=0.0)

    fin_idx = b_idx * N + n
    grad_v  = tl.load(Grad_V_final_ptr  + fin_idx, mask=mask, other=0.0)
    grad_a  = tl.load(Grad_A_final_ptr  + fin_idx, mask=mask, other=0.0)
    grad_w1 = tl.load(Grad_W1_final_ptr + fin_idx, mask=mask, other=0.0)
    grad_w2 = tl.load(Grad_W2_final_ptr + fin_idx, mask=mask, other=0.0)
    grad_w3 = tl.load(Grad_W3_final_ptr + fin_idx, mask=mask, other=0.0)
    if ORDER >= 4:
        grad_w4 = tl.load(Grad_W4_final_ptr + fin_idx, mask=mask, other=0.0)
    else:
        grad_w4 = 0.0
    if ORDER >= 5:
        grad_w5 = tl.load(Grad_W5_final_ptr + fin_idx, mask=mask, other=0.0)
    else:
        grad_w5 = 0.0

    d_eta_accum   = 0.0 * eta
    d_gamma_accum = 0.0 * gamma
    d_beta1_accum = 0.0 * eta;  d_lam1_accum = 0.0 * eta
    d_beta2_accum = 0.0 * eta;  d_lam2_accum = 0.0 * eta
    d_beta3_accum = 0.0 * eta;  d_lam3_accum = 0.0 * eta
    if ORDER >= 4:
        d_beta4_accum = 0.0 * eta;  d_lam4_accum = 0.0 * eta
    if ORDER >= 5:
        d_beta5_accum = 0.0 * eta;  d_lam5_accum = 0.0 * eta

    for t in range(T - 1, -1, -1):
        out_idx = b_idx * stride_sb + t * stride_st + n * stride_sn

        spike      = tl.load(Spike_ptr    + out_idx, mask=mask, other=0.0)
        v_t        = tl.load(V_seq_ptr    + out_idx, mask=mask, other=0.0)
        a_t        = tl.load(A_seq_ptr    + out_idx, mask=mask, other=0.0)
        v_raw_t    = tl.load(Vraw_seq_ptr + out_idx, mask=mask, other=0.0)
        g_spike_in = tl.load(Grad_Spike_ptr + out_idx, mask=mask, other=0.0)
        g_v_in     = tl.load(Grad_V_seq_ptr + out_idx, mask=mask, other=0.0)
        g_a_in     = tl.load(Grad_A_seq_ptr + out_idx, mask=mask, other=0.0)

        grad_v += g_v_in
        grad_a += g_a_in

        v_out     = v_t + spike * thresh
        surr      = select_surrogate_grad(v_out - thresh, alpha_surr, surrogate_type)
        g_s_total = g_spike_in + grad_v * (-thresh)
        d_vmix    = grad_v + g_s_total * surr

        prev_idx = b_idx * stride_sb + (t - 1) * stride_st + n * stride_sn

        # Reconstruct previous W states and AP outputs
        if t > 0:
            w1_prev = tl.load(W1_seq_ptr + prev_idx, mask=mask, other=0.0)
            w2_prev = tl.load(W2_seq_ptr + prev_idx, mask=mask, other=0.0)
            w3_prev = tl.load(W3_seq_ptr + prev_idx, mask=mask, other=0.0)
        else:
            w1_prev = tl.load(W1_init_ptr + idx_n, mask=mask, other=0.0)
            w2_prev = tl.load(W2_init_ptr + idx_n, mask=mask, other=0.0)
            w3_prev = tl.load(W3_init_ptr + idx_n, mask=mask, other=0.0)

        y1_t = -beta1 * v_raw_t + w1_prev
        y2_t = -beta2 * y1_t    + w2_prev
        y3_t = -beta3 * y2_t    + w3_prev

        if ORDER >= 4:
            if t > 0:
                w4_prev = tl.load(W4_seq_ptr + prev_idx, mask=mask, other=0.0)
            else:
                w4_prev = tl.load(W4_init_ptr + idx_n, mask=mask, other=0.0)
            y4_t = -beta4 * y3_t + w4_prev

        if ORDER >= 5:
            if t > 0:
                w5_prev = tl.load(W5_seq_ptr + prev_idx, mask=mask, other=0.0)
            else:
                w5_prev = tl.load(W5_init_ptr + idx_n, mask=mask, other=0.0)
            y5_t = -beta5 * y4_t + w5_prev

        # ── Backward through TS mixing ────────────────────────────────────
        if ORDER == 3:
            blend3_t = (1.0 - lam3) * y2_t + lam3 * y3_t
            blend2_t = (1.0 - lam2) * y1_t + lam2 * blend3_t

            d_lam1_accum  += d_vmix * (blend2_t - v_raw_t)
            d_blend2       = d_vmix * lam1
            d_vraw_mix     = d_vmix * (1.0 - lam1)

            d_lam2_accum  += d_blend2 * (blend3_t - y1_t)
            d_blend3       = d_blend2 * lam2
            d_y1_from_lam  = d_blend2 * (1.0 - lam2)

            d_lam3_accum  += d_blend3 * (y3_t - y2_t)
            d_y3_from_lam  = d_blend3 * lam3
            d_y2_from_lam  = d_blend3 * (1.0 - lam3)

            dy3_tot        = d_y3_from_lam + beta3 * grad_w3
            dy2_from_3     = dy3_tot * (-beta3) + grad_w3
            d_beta3_accum += dy3_tot * (-y2_t) + grad_w3 * y3_t
            grad_w3        = dy3_tot

            dy2_tot        = dy2_from_3 + d_y2_from_lam + beta2 * grad_w2
            dy1_from_2     = dy2_tot * (-beta2) + grad_w2
            d_beta2_accum += dy2_tot * (-y1_t) + grad_w2 * y2_t
            grad_w2        = dy2_tot

            dy1_tot        = dy1_from_2 + d_y1_from_lam + beta1 * grad_w1
            d_vraw         = dy1_tot * (-beta1) + grad_w1 + d_vraw_mix
            d_beta1_accum += dy1_tot * (-v_raw_t) + grad_w1 * y1_t
            grad_w1        = dy1_tot

        if ORDER == 4:
            blend4_t = (1.0 - lam4) * y3_t + lam4 * y4_t
            blend3_t = (1.0 - lam3) * y2_t + lam3 * blend4_t
            blend2_t = (1.0 - lam2) * y1_t + lam2 * blend3_t

            d_lam1_accum  += d_vmix * (blend2_t - v_raw_t)
            d_blend2       = d_vmix * lam1
            d_vraw_mix     = d_vmix * (1.0 - lam1)

            d_lam2_accum  += d_blend2 * (blend3_t - y1_t)
            d_blend3       = d_blend2 * lam2
            d_y1_from_lam  = d_blend2 * (1.0 - lam2)

            d_lam3_accum  += d_blend3 * (blend4_t - y2_t)
            d_blend4       = d_blend3 * lam3
            d_y2_from_lam  = d_blend3 * (1.0 - lam3)

            d_lam4_accum  += d_blend4 * (y4_t - y3_t)
            d_y4_from_lam  = d_blend4 * lam4
            d_y3_from_lam  = d_blend4 * (1.0 - lam4)

            dy4_tot        = d_y4_from_lam + beta4 * grad_w4
            dy3_from_4     = dy4_tot * (-beta4) + grad_w4
            d_beta4_accum += dy4_tot * (-y3_t) + grad_w4 * y4_t
            grad_w4        = dy4_tot

            dy3_tot        = dy3_from_4 + d_y3_from_lam + beta3 * grad_w3
            dy2_from_3     = dy3_tot * (-beta3) + grad_w3
            d_beta3_accum += dy3_tot * (-y2_t) + grad_w3 * y3_t
            grad_w3        = dy3_tot

            dy2_tot        = dy2_from_3 + d_y2_from_lam + beta2 * grad_w2
            dy1_from_2     = dy2_tot * (-beta2) + grad_w2
            d_beta2_accum += dy2_tot * (-y1_t) + grad_w2 * y2_t
            grad_w2        = dy2_tot

            dy1_tot        = dy1_from_2 + d_y1_from_lam + beta1 * grad_w1
            d_vraw         = dy1_tot * (-beta1) + grad_w1 + d_vraw_mix
            d_beta1_accum += dy1_tot * (-v_raw_t) + grad_w1 * y1_t
            grad_w1        = dy1_tot

        if ORDER == 5:
            blend5_t = (1.0 - lam5) * y4_t + lam5 * y5_t
            blend4_t = (1.0 - lam4) * y3_t + lam4 * blend5_t
            blend3_t = (1.0 - lam3) * y2_t + lam3 * blend4_t
            blend2_t = (1.0 - lam2) * y1_t + lam2 * blend3_t

            d_lam1_accum  += d_vmix * (blend2_t - v_raw_t)
            d_blend2       = d_vmix * lam1
            d_vraw_mix     = d_vmix * (1.0 - lam1)

            d_lam2_accum  += d_blend2 * (blend3_t - y1_t)
            d_blend3       = d_blend2 * lam2
            d_y1_from_lam  = d_blend2 * (1.0 - lam2)

            d_lam3_accum  += d_blend3 * (blend4_t - y2_t)
            d_blend4       = d_blend3 * lam3
            d_y2_from_lam  = d_blend3 * (1.0 - lam3)

            d_lam4_accum  += d_blend4 * (blend5_t - y3_t)
            d_blend5       = d_blend4 * lam4
            d_y3_from_lam  = d_blend4 * (1.0 - lam4)

            d_lam5_accum  += d_blend5 * (y5_t - y4_t)
            d_y5_from_lam  = d_blend5 * lam5
            d_y4_from_lam  = d_blend5 * (1.0 - lam5)

            dy5_tot        = d_y5_from_lam + beta5 * grad_w5
            dy4_from_5     = dy5_tot * (-beta5) + grad_w5
            d_beta5_accum += dy5_tot * (-y4_t) + grad_w5 * y5_t
            grad_w5        = dy5_tot

            dy4_tot        = dy4_from_5 + d_y4_from_lam + beta4 * grad_w4
            dy3_from_4     = dy4_tot * (-beta4) + grad_w4
            d_beta4_accum += dy4_tot * (-y3_t) + grad_w4 * y4_t
            grad_w4        = dy4_tot

            dy3_tot        = dy3_from_4 + d_y3_from_lam + beta3 * grad_w3
            dy2_from_3     = dy3_tot * (-beta3) + grad_w3
            d_beta3_accum += dy3_tot * (-y2_t) + grad_w3 * y3_t
            grad_w3        = dy3_tot

            dy2_tot        = dy2_from_3 + d_y2_from_lam + beta2 * grad_w2
            dy1_from_2     = dy2_tot * (-beta2) + grad_w2
            d_beta2_accum += dy2_tot * (-y1_t) + grad_w2 * y2_t
            grad_w2        = dy2_tot

            dy1_tot        = dy1_from_2 + d_y1_from_lam + beta1 * grad_w1
            d_vraw         = dy1_tot * (-beta1) + grad_w1 + d_vraw_mix
            d_beta1_accum += dy1_tot * (-v_raw_t) + grad_w1 * y1_t
            grad_w1        = dy1_tot

        # ── Backward through FS module ────────────────────────────────────
        if t > 0:
            a_prev = tl.load(A_seq_ptr + prev_idx, mask=mask, other=0.0)
            v_prev = tl.load(V_seq_ptr + prev_idx, mask=mask, other=0.0)
        else:
            a_prev = tl.load(A_init_ptr + idx_n, mask=mask, other=0.0)
            v_prev = tl.load(V_init_ptr + idx_n, mask=mask, other=0.0)

        if IMPLICIT:
            d_gamma_accum += grad_a * (dt * v_raw_t)
            d_vraw_tot     = d_vraw + grad_a * gamma_dt
            grad_a_tot     = grad_a * decay_b
            d_eta_accum   += d_vraw_tot * (-dt * a_prev)
            u_idx = b_idx * stride_ub + t * stride_ut + n * stride_un
            tl.store(Grad_U_ptr + u_idx, d_vraw_tot, mask=mask)
            d_v_from_fs = d_vraw_tot * decay_v
            d_a_from_fs = grad_a_tot + d_vraw_tot * (-eta_dt)
        else:
            grad_a_tot    = grad_a * decay_b
            u_idx = b_idx * stride_ub + t * stride_ut + n * stride_un
            tl.store(Grad_U_ptr + u_idx, d_vraw, mask=mask)
            d_eta_accum   += d_vraw * (-dt * a_prev)
            d_gamma_accum += grad_a * (dt * v_prev)
            d_v_from_fs = d_vraw * decay_v + grad_a * gamma_dt
            d_a_from_fs = grad_a_tot       + d_vraw * (-eta_dt)

        grad_v = d_v_from_fs
        grad_a = d_a_from_fs

    tl.store(Grad_V_init_ptr  + idx_n, grad_v,  mask=mask)
    tl.store(Grad_A_init_ptr  + idx_n, grad_a,  mask=mask)
    tl.store(Grad_W1_init_ptr + idx_n, grad_w1, mask=mask)
    tl.store(Grad_W2_init_ptr + idx_n, grad_w2, mask=mask)
    tl.store(Grad_W3_init_ptr + idx_n, grad_w3, mask=mask)
    if ORDER >= 4:
        tl.store(Grad_W4_init_ptr + idx_n, grad_w4, mask=mask)
    if ORDER >= 5:
        tl.store(Grad_W5_init_ptr + idx_n, grad_w5, mask=mask)

    tl.store(Grad_eta_ptr   + n * stride_geta   + b_idx, d_eta_accum,   mask=mask)
    tl.store(Grad_gamma_ptr + n * stride_ggamma + b_idx, d_gamma_accum, mask=mask)
    tl.store(Grad_beta1_ptr + n * stride_geta + b_idx, d_beta1_accum, mask=mask)
    tl.store(Grad_lam1_ptr  + n * stride_geta + b_idx, d_lam1_accum,  mask=mask)
    tl.store(Grad_beta2_ptr + n * stride_geta + b_idx, d_beta2_accum, mask=mask)
    tl.store(Grad_lam2_ptr  + n * stride_geta + b_idx, d_lam2_accum,  mask=mask)
    tl.store(Grad_beta3_ptr + n * stride_geta + b_idx, d_beta3_accum, mask=mask)
    tl.store(Grad_lam3_ptr  + n * stride_geta + b_idx, d_lam3_accum,  mask=mask)
    if ORDER >= 4:
        tl.store(Grad_beta4_ptr + n * stride_geta + b_idx, d_beta4_accum, mask=mask)
        tl.store(Grad_lam4_ptr  + n * stride_geta + b_idx, d_lam4_accum,  mask=mask)
    if ORDER >= 5:
        tl.store(Grad_beta5_ptr + n * stride_geta + b_idx, d_beta5_accum, mask=mask)
        tl.store(Grad_lam5_ptr  + n * stride_geta + b_idx, d_lam5_accum,  mask=mask)


def _fits_forward_high(ctx, ORDER, IMPLICIT,
                       u, v_init, a_init,
                       w1_init, w2_init, w3_init, w4_init, w5_init,
                       eta, gamma,
                       beta1, beta2, beta3, beta4, beta5,
                       lam1, lam2, lam3, lam4, lam5,
                       tau_m, tau_a, dt, thresh,
                       surrogate_alpha, surrogate_type):
    B, T, N = u.shape
    decay_v  = 1.0 - dt / tau_m
    decay_b  = 1.0 - dt / tau_a
    dev, dt_ = u.device, u.dtype

    _c = lambda t: t.contiguous()
    u       = _c(u);      v_init = _c(v_init); a_init = _c(a_init)
    w1_init = _c(w1_init); w2_init = _c(w2_init); w3_init = _c(w3_init)
    w4_init = _c(w4_init); w5_init = _c(w5_init)
    eta     = _c(eta);    gamma = _c(gamma)
    beta1   = _c(beta1);  beta2 = _c(beta2);  beta3 = _c(beta3)
    beta4   = _c(beta4);  beta5 = _c(beta5)
    lam1    = _c(lam1);   lam2  = _c(lam2);   lam3  = _c(lam3)
    lam4    = _c(lam4);   lam5  = _c(lam5)

    spike    = torch.empty_like(u)
    v_seq    = torch.empty_like(u)
    a_seq    = torch.empty_like(u)
    vraw_seq = torch.empty_like(u)
    w1_seq   = torch.empty_like(u)
    w2_seq   = torch.empty_like(u)
    w3_seq   = torch.empty_like(u)
    w4_seq   = torch.empty_like(u) if ORDER >= 4 else _dummy1(dev, dt_)
    w5_seq   = torch.empty_like(u) if ORDER >= 5 else _dummy1(dev, dt_)
    v_final  = torch.empty_like(v_init)
    a_final  = torch.empty_like(a_init)
    w1_final = torch.empty_like(w1_init)
    w2_final = torch.empty_like(w2_init)
    w3_final = torch.empty_like(w3_init)
    w4_final = torch.empty_like(w4_init)
    w5_final = torch.empty_like(w5_init)

    grid = lambda meta: (triton.cdiv(B * N, meta['BLOCK_SIZE']),)
    fits_fwd_kernel_high[grid](
        u, v_init, a_init,
        w1_init, w2_init, w3_init, w4_init, w5_init,
        eta, gamma,
        beta1, beta2, beta3, beta4, beta5,
        lam1,  lam2,  lam3,  lam4,  lam5,
        decay_v, decay_b, dt, thresh,
        spike, v_seq, a_seq, vraw_seq,
        w1_seq, w2_seq, w3_seq, w4_seq, w5_seq,
        v_final, a_final,
        w1_final, w2_final, w3_final, w4_final, w5_final,
        B, T, N,
        u.stride(0), u.stride(1), u.stride(2),
        spike.stride(0), spike.stride(1), spike.stride(2),
        eta.stride(0), gamma.stride(0),
        surrogate_type,
        IMPLICIT=IMPLICIT, ORDER=ORDER, BLOCK_SIZE=256,
    )

    saved = [spike, v_seq, a_seq, vraw_seq,
             v_init, a_init,
             w1_init, w2_init, w3_init, w4_init, w5_init,
             eta, gamma,
             beta1, beta2, beta3, beta4, beta5,
             lam1,  lam2,  lam3,  lam4,  lam5,
             w1_seq, w2_seq, w3_seq]
    if ORDER >= 4: saved.append(w4_seq)
    if ORDER >= 5: saved.append(w5_seq)
    ctx.save_for_backward(*saved)
    ctx.params = (decay_v, decay_b, dt, thresh, surrogate_alpha,
                  surrogate_type, B, T, N, ORDER, IMPLICIT)

    return spike, v_seq, a_seq, v_final, a_final, w1_final, w2_final, w3_final, w4_final, w5_final


def _fits_backward_high(ctx, ORDER, IMPLICIT,
                        grad_spike, grad_v_seq, grad_a_seq,
                        grad_v_final, grad_a_final,
                        grad_w1_final, grad_w2_final,
                        grad_w3_final, grad_w4_final, grad_w5_final):
    (decay_v, decay_b, dt, thresh, surrogate_alpha,
     surrogate_type, B, T, N, _, _impl) = ctx.params

    saved = ctx.saved_tensors
    # Fixed section (26 tensors, always present regardless of ORDER):
    (spike, v_seq, a_seq, vraw_seq,
     v_init, a_init,
     w1_init, w2_init, w3_init, w4_init, w5_init,
     eta, gamma,
     beta1, beta2, beta3, beta4, beta5,
     lam1,  lam2,  lam3,  lam4,  lam5,
     w1_seq, w2_seq, w3_seq) = saved[:26]
    # Conditional section (w4_seq at idx=26, w5_seq at idx=27):
    idx = 26
    w4_seq = saved[idx] if ORDER >= 4 else _dummy1(v_seq.device, v_seq.dtype)
    if ORDER >= 4: idx += 1
    w5_seq = saved[idx] if ORDER >= 5 else _dummy1(v_seq.device, v_seq.dtype)

    def _z(g, ref): return torch.zeros_like(ref) if g is None else g.contiguous()
    grad_spike    = _z(grad_spike,    spike)
    grad_v_seq    = _z(grad_v_seq,    v_seq)
    grad_a_seq    = _z(grad_a_seq,    a_seq)
    grad_v_final  = _z(grad_v_final,  v_init)
    grad_a_final  = _z(grad_a_final,  a_init)
    grad_w1_final = _z(grad_w1_final, w1_init)
    grad_w2_final = _z(grad_w2_final, w2_init)
    grad_w3_final = _z(grad_w3_final, w3_init)
    grad_w4_final = _z(grad_w4_final, w4_init)
    grad_w5_final = _z(grad_w5_final, w5_init)

    grad_u        = torch.empty_like(spike)
    grad_v_init   = torch.empty_like(v_init)
    grad_a_init   = torch.empty_like(a_init)
    grad_w1_init  = torch.zeros_like(w1_init)
    grad_w2_init  = torch.zeros_like(w2_init)
    grad_w3_init  = torch.zeros_like(w3_init)
    grad_w4_init  = torch.zeros_like(w4_init)
    grad_w5_init  = torch.zeros_like(w5_init)

    dev, dt_ = spike.device, spike.dtype
    _gr = lambda: torch.zeros(N, B, device=dev, dtype=dt_)
    grad_eta_r   = _gr(); grad_gamma_r = _gr()
    grad_beta1_r = _gr(); grad_lam1_r  = _gr()
    grad_beta2_r = _gr(); grad_lam2_r  = _gr()
    grad_beta3_r = _gr(); grad_lam3_r  = _gr()
    grad_beta4_r = _gr(); grad_lam4_r  = _gr()
    grad_beta5_r = _gr(); grad_lam5_r  = _gr()

    grid = lambda meta: (triton.cdiv(B * N, meta['BLOCK_SIZE']),)
    fits_bwd_kernel_high[grid](
        grad_spike, grad_v_seq, grad_a_seq,
        grad_v_final, grad_a_final,
        grad_w1_final, grad_w2_final, grad_w3_final, grad_w4_final, grad_w5_final,
        v_init, a_init,
        w1_init, w2_init, w3_init, w4_init, w5_init,
        spike, v_seq, a_seq, vraw_seq,
        w1_seq, w2_seq, w3_seq, w4_seq, w5_seq,
        eta, gamma,
        beta1, beta2, beta3, beta4, beta5,
        lam1,  lam2,  lam3,  lam4,  lam5,
        decay_v, decay_b, dt, thresh, surrogate_alpha,
        grad_u, grad_v_init, grad_a_init,
        grad_w1_init, grad_w2_init, grad_w3_init, grad_w4_init, grad_w5_init,
        grad_eta_r,   grad_gamma_r,
        grad_beta1_r, grad_beta2_r, grad_beta3_r, grad_beta4_r, grad_beta5_r,
        grad_lam1_r,  grad_lam2_r,  grad_lam3_r,  grad_lam4_r,  grad_lam5_r,
        B, T, N,
        spike.stride(0), spike.stride(1), spike.stride(2),
        spike.stride(0), spike.stride(1), spike.stride(2),
        eta.stride(0), gamma.stride(0),
        grad_eta_r.stride(0), grad_gamma_r.stride(0),
        surrogate_type,
        IMPLICIT=IMPLICIT, ORDER=ORDER, BLOCK_SIZE=256,
    )

    # Matches forward signature (see _make_fits_func_high):
    # u, v_init, a_init, w1..5_init, eta, gamma, beta1..5, lam1..5,
    # tau_m, tau_a, dt, thresh, s_alpha, s_type
    return (grad_u, grad_v_init, grad_a_init,
            grad_w1_init, grad_w2_init, grad_w3_init, grad_w4_init, grad_w5_init,
            grad_eta_r.sum(1),   grad_gamma_r.sum(1),
            grad_beta1_r.sum(1), grad_beta2_r.sum(1), grad_beta3_r.sum(1),
            grad_beta4_r.sum(1) if ORDER >= 4 else None,
            grad_beta5_r.sum(1) if ORDER >= 5 else None,
            grad_lam1_r.sum(1),  grad_lam2_r.sum(1),  grad_lam3_r.sum(1),
            grad_lam4_r.sum(1)  if ORDER >= 4 else None,
            grad_lam5_r.sum(1)  if ORDER >= 5 else None,
            None, None, None, None, None, None)


def _make_fits_func_high(order: int, implicit: bool):
    class _Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx,
                    u, v_init, a_init,
                    w1_init, w2_init, w3_init, w4_init, w5_init,
                    eta, gamma,
                    beta1, beta2, beta3, beta4, beta5,
                    lam1, lam2, lam3, lam4, lam5,
                    tau_m, tau_a, dt, thresh,
                    surrogate_alpha, surrogate_type):
            return _fits_forward_high(
                ctx, order, implicit,
                u, v_init, a_init,
                w1_init, w2_init, w3_init, w4_init, w5_init,
                eta, gamma,
                beta1, beta2, beta3, beta4, beta5,
                lam1, lam2, lam3, lam4, lam5,
                tau_m, tau_a, dt, thresh,
                surrogate_alpha, surrogate_type)

        @staticmethod
        def backward(ctx,
                     grad_spike, grad_v_seq, grad_a_seq,
                     grad_v_final, grad_a_final,
                     grad_w1_final, grad_w2_final,
                     grad_w3_final, grad_w4_final, grad_w5_final):
            return _fits_backward_high(
                ctx, order, implicit,
                grad_spike, grad_v_seq, grad_a_seq,
                grad_v_final, grad_a_final,
                grad_w1_final, grad_w2_final,
                grad_w3_final, grad_w4_final, grad_w5_final)

    mode = "Implicit" if implicit else "Explicit"
    _Func.__name__ = f'FiTSFunc_{mode}_N{order}'
    return _Func


_FUNC_TABLE_HIGH = {
    (order, implicit): _make_fits_func_high(order, implicit)
    for order in (3, 4, 5)
    for implicit in (True, False)
}


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────
def fused_fits_temporal(u, v_init, a_init, eta, gamma, beta_list,
                        lam1, lam2, *,
                        y1_init=None, y2_init=None,
                        w1_init=None, w2_init=None,
                        order=1, tau_m=0.020, tau_a=0.1, dt=0.001,
                        v_threshold=1.0, surrogate_alpha=2.0, surrogate_type=0,
                        implicit=True):
    """
    Run one temporal chunk of the FiTS neuron update (Algorithm 1).

    Args:
        u             : [B, T, N] pre-synaptic input after linear projection.
        v_init        : [B, N]  initial membrane potential V.
        a_init        : [B, N]  initial adaptation state a.
        eta           : [N]     adaptation feedback η (computed from Ω*).
        gamma         : [N]     adaptation coupling γ (computed from Ω*).
        beta_list     : list of tensors [N] — all-pass coefficients β_m.
        lam1          : [N]     TS mixing weight λ₁ (ORDER ≥ 1).
        lam2          : [N]     TS mixing weight λ₂ (ORDER == 2).
        order         : int     number of all-pass stages (0, 1, or 2).
        tau_m       : float   membrane time constant τ_m (seconds).
        tau_a         : float   adaptation time constant τ_a (seconds).
        dt            : float   simulation timestep Δt (seconds).
        v_threshold   : float   spike threshold V_th.
        surrogate_alpha: float  surrogate gradient steepness.
        surrogate_type : int    surrogate type (0=Triangle, 1=Arctan, 2=Sigmoid).
        implicit      : bool    True → semi-implicit Euler; False → explicit Euler.

    Returns:
        (spike, v_seq, a_seq, v_final, a_final, w1_final, w2_final)
    """
    assert order in (0, 1, 2), f"order must be 0, 1, or 2; got {order}"
    Func = _FUNC_TABLE[(order, implicit)]
    N    = eta.shape[0]
    dev, dt_ = eta.device, eta.dtype

    b_size = v_init.shape[0]
    if w1_init is None and y1_init is not None: w1_init = y1_init
    if w2_init is None and y2_init is not None: w2_init = y2_init
    if w1_init is None: w1_init = torch.zeros(b_size, N, device=dev, dtype=dt_)
    if w2_init is None: w2_init = torch.zeros(b_size, N, device=dev, dtype=dt_)

    beta1 = beta_list[0] if order >= 1 else torch.zeros(N, device=dev, dtype=dt_)
    beta2 = beta_list[1] if order >= 2 else torch.zeros(N, device=dev, dtype=dt_)

    return Func.apply(u, v_init, a_init, w1_init, w2_init,
                      eta, gamma, beta1, beta2, lam1, lam2,
                      tau_m, tau_a, dt, v_threshold,
                      surrogate_alpha, surrogate_type)


def fused_fits_temporal_high(u, v_init, a_init, eta, gamma, beta_list, lam_list,
                              w_inits=None, *,
                              order, tau_m=0.020, tau_a=0.1, dt=0.001,
                              v_threshold=1.0, surrogate_alpha=2.0, surrogate_type=0,
                              implicit=True):
    """
    Triton-fused FiTS temporal chunk for ORDER = 3, 4, or 5.

    Mirrors the interface of fits.reference.fits_temporal_reference.

    Args:
        beta_list : list of ORDER tensors [N]
        lam_list  : list of ORDER tensors [N]
        w_inits   : list of ORDER tensors [B, N], or None → zeros

    Returns:
        (spikes, v_seq, a_seq, v_final, a_final, w_finals)
        w_finals : list of ORDER tensors [B, N]
    """
    assert order in (3, 4, 5), \
        f"fused_fits_temporal_high: order must be 3, 4, or 5; got {order}"
    Func = _FUNC_TABLE_HIGH[(order, implicit)]
    N        = eta.shape[0]
    dev, dt_ = eta.device, eta.dtype
    b_size   = v_init.shape[0]

    _z  = lambda: torch.zeros(b_size, N, device=dev, dtype=dt_)
    _zn = lambda: torch.zeros(N,      device=dev, dtype=dt_)

    if w_inits is None:
        w_inits = [_z() for _ in range(order)]
    while len(w_inits) < order:
        w_inits.append(_z())

    beta1 = beta_list[0]
    beta2 = beta_list[1]
    beta3 = beta_list[2]
    beta4 = beta_list[3] if order >= 4 else _zn()
    beta5 = beta_list[4] if order >= 5 else _zn()

    lam1 = lam_list[0]
    lam2 = lam_list[1]
    lam3 = lam_list[2]
    lam4 = lam_list[3] if order >= 4 else _zn()
    lam5 = lam_list[4] if order >= 5 else _zn()

    w1_init = w_inits[0]
    w2_init = w_inits[1]
    w3_init = w_inits[2]
    w4_init = w_inits[3] if order >= 4 else _z()
    w5_init = w_inits[4] if order >= 5 else _z()

    out = Func.apply(u, v_init, a_init,
                     w1_init, w2_init, w3_init, w4_init, w5_init,
                     eta, gamma,
                     beta1, beta2, beta3, beta4, beta5,
                     lam1, lam2, lam3, lam4, lam5,
                     tau_m, tau_a, dt, v_threshold,
                     surrogate_alpha, surrogate_type)

    # out = (spike, v_seq, a_seq, v_final, a_final, w1f, w2f, w3f, w4f, w5f)
    spike, v_seq, a_seq, v_final, a_final = out[0], out[1], out[2], out[3], out[4]
    w_finals = list(out[5 : 5 + order])

    return spike, v_seq, a_seq, v_final, a_final, w_finals
