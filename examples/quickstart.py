"""
Minimal example: FiTS neurons as a drop-in spiking layer in your own model.

    python examples/quickstart.py          # uses CUDA + Triton if available, else CPU
"""

import torch
import torch.nn as nn

from fits import FiTSNeuron, FrequencyGuard


class TinySNN(nn.Module):
    def __init__(self, in_features=140, hidden=128, num_classes=20):
        super().__init__()
        # SHD-like constants: τ_m = 40 ms, τ_a = 200 ms, Δt = 4 ms, f* ∈ [1, 50] Hz (log-spaced)
        common = dict(tau_m=0.04, tau_a=0.2, dt=0.004, f_min=1.0, f_max=50.0)
        self.layer1 = FiTSNeuron(in_features, hidden, order=1, **common)
        self.layer2 = FiTSNeuron(hidden, hidden, order=1, **common)
        self.readout = nn.Linear(hidden, num_classes)

    def forward(self, x):                      # x: [B, T, in_features]
        s1, *_ = self.layer1(x)                # spikes: [B, T, hidden]
        s2, *_ = self.layer2(s1)
        return self.readout(s2.sum(dim=1))     # sum over time -> logits


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = TinySNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    FrequencyGuard(model, optimizer)           # keeps every f* inside the stable region

    x = (torch.rand(16, 250, 140, device=device) < 0.05).float()   # random input spikes
    y = torch.randint(0, 20, (16,), device=device)
    for step in range(3):
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step}: loss {loss.item():.4f}")

    layer = model.layer1
    f = layer.get_freq().detach()
    print("\nper-neuron readouts of layer 1 (first 5 neurons)")
    print("  target frequency f*     (Hz):", f[:5].cpu().numpy().round(2))
    print("  realized DT peak f*_DT  (Hz):", layer.realized_freq()[:5].cpu().numpy().round(2))
    print("  TS group delay at f*    (ms):", (layer.ts_group_delay()[:5] * 1e3).cpu().numpy().round(3))
    print("  mixing weight lambda_1      :", layer.get_lam1()[:5].detach().cpu().numpy().round(3))


if __name__ == "__main__":
    main()
