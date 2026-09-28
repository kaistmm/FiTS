"""
Keeping learned target frequencies inside the stable region.

f* = exp(f_raw) is unconstrained during training.  Under the semi-implicit
update the FS recurrence is stable only while κΔt² < (1+μ̄)(1+ρ̄) (Jury
criterion).  With the continuous-time map κ_CT grows like f*², so there is a
critical frequency above which a neuron's state diverges (it then fires at
every step regardless of its input, and the surrogate gradient vanishes, so it
never recovers).  The limit depends only on (τ_m, τ_a, Δt), see
``fits.frequency.critical_freq``; e.g. 30.9 Hz for the GSC constants
(τ_m = 0.1 s, τ_a = 0.5 s, Δt = 10 ms), whose initial range already reaches
30 Hz, and 77.2 Hz for SHD/SSC (Δt = 4 ms).

:class:`FrequencyGuard` projects f* back below ``safety`` × that limit after
every optimizer step.  The forward pass is untouched, so below the cap the
model is exactly the unconstrained one.  The projection is momentum-aware for
Adam-type optimizers: outward gradient components of neurons sitting on the
cap are dropped before the step, and after projecting only the outward first
moment is cleared, so a neuron on the cap can always move back down.
"""

import math

import torch

from fits.frequency import critical_freq

__all__ = ["FrequencyGuard"]


class FrequencyGuard:
    """
    Projected optimization of f*:  min L  s.t.  f* ≤ f_safe  (per layer).

    Usage::

        optimizer = torch.optim.Adam(model.parameters(), lr=...)
        guard = FrequencyGuard(model, optimizer, safety=0.98)
        # ... ordinary training loop; the guard runs as optimizer step hooks.

    Args:
        model     : any module containing :class:`fits.FiTSNeuron` layers.
        optimizer : the optimizer that updates their ``f_raw``.
        safety    : fraction of the Jury bound on κΔt² to stay under.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 safety: float = 0.98):
        from fits.neuron import FiTSNeuron  # local import to avoid a cycle

        self.safety = safety
        self.caps = {}          # f_raw Parameter -> cap on f_raw (= log f_safe)
        self.cap_hz = []        # per guarded layer, for logging
        opt_params = {id(p) for g in optimizer.param_groups for p in g["params"]}
        for m in model.modules():
            if not isinstance(m, FiTSNeuron):
                continue
            p = getattr(m, "f_raw", None)
            if not isinstance(p, torch.nn.Parameter) or id(p) not in opt_params:
                continue            # frozen f* or not optimized by this optimizer
            if not m.implicit:
                continue            # the bound is derived for the semi-implicit update
            f_safe = critical_freq(m.mu, m.rho, m.dt, m.freq_param, safety)
            self.caps[p] = math.log(f_safe)
            self.cap_hz.append(f_safe)

        self.num_projected = 0  # elements projected since the last reset_counter()
        self._handles = [optimizer.register_step_pre_hook(self._pre_step),
                         optimizer.register_step_post_hook(self._post_step)]

    @torch.no_grad()
    def _pre_step(self, optimizer, args, kwargs):
        # On the cap, drop gradient components that would raise f_raw (g < 0).
        for p, cap in self.caps.items():
            if p.grad is not None:
                p.grad.masked_fill_((p >= cap) & (p.grad < 0), 0.0)

    @torch.no_grad()
    def _post_step(self, optimizer, args, kwargs):
        for p, cap in self.caps.items():
            over = p > cap
            n = int(over.sum())
            if n == 0:
                continue
            self.num_projected += n
            p.masked_fill_(over, cap)
            exp_avg = optimizer.state.get(p, {}).get("exp_avg")
            if exp_avg is not None:     # exp_avg < 0 would push f_raw up again
                exp_avg.masked_fill_(over & (exp_avg < 0), 0.0)

    def reset_counter(self) -> int:
        """Return and reset the number of projected elements."""
        n, self.num_projected = self.num_projected, 0
        return n

    def remove(self):
        """Detach the guard from the optimizer."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __repr__(self):
        caps = ", ".join(f"{c:.2f}" for c in self.cap_hz)
        return f"FrequencyGuard(safety={self.safety}, f_safe_hz=[{caps}])"
