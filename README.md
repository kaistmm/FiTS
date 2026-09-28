<div align="center">

# FiTS: Interpretable Spiking Neurons via<br>Frequency Selectivity and Temporal Shaping

[![arXiv](https://img.shields.io/badge/arXiv-2605.13071-b31b1b.svg)](https://arxiv.org/abs/2605.13071)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[**Jongmin Choi**](https://cjongmin.github.io)&nbsp;&nbsp;·&nbsp;&nbsp;**Joon Son Chung**

KAIST

</div>

## News

- **2026.09** FiTS is accepted to $`\color{red}\textbf{NeurIPS 2026}`$ as a $`\color{red}\textbf{Spotlight}`$ $`\color{red}\textbf{0.95\%}`$ (292 / 30,709) 🎖️

## Introduction

FiTS is a spiking neuron with two learnable parts. Frequency Selectivity (FS) gives each neuron a target frequency f\*, the peak of its subthreshold response, which is converted to the adaptation strength in closed form. Temporal Shaping (TS) is a small all-pass cascade that shifts when frequency components reach the threshold.

After training, every neuron can be read as its f\* and its TS delay. Details are in the [paper](https://arxiv.org/abs/2605.13071).

## Installation

```bash
git clone https://github.com/kaistmm/FiTS.git && cd FiTS
conda create -n fits python=3.10 -y && conda activate fits
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # match your CUDA
pip install -r requirements.txt
pip install -e .
```

We used Python 3.10, PyTorch 2.5.1 and CUDA 12.1 on an RTX A5000. The code also runs on CPU, more slowly.

## Data

| dataset | source | `--data-dir` contains |
|---|---|---|
| SHD | [zenkelab.org/datasets](https://zenkelab.org/datasets/) (gunzip the files) | `shd_train.h5`, `shd_test.h5` |
| SSC | [zenkelab.org/datasets](https://zenkelab.org/datasets/) (gunzip the files) | `ssc_train.h5`, `ssc_valid.h5`, `ssc_test.h5` |
| GSC v0.02 | downloaded and split on the first run | `train/`, `valid/`, `test/` |
| sMNIST | downloaded by torchvision | |

## Training

```bash
python -m experiments.shd.train    --config configs/shd/fits_o1_w128.yaml       --data-dir data/SHD
python -m experiments.shd.train    --config configs/shd/fits_o2_w256_valid.yaml --data-dir data/SHD
python -m experiments.ssc.train    --config configs/ssc/fits_o1_w512.yaml       --data-dir data/SSC
python -m experiments.gsc.train    --config configs/gsc/fits_o1_w512.yaml       --data-dir data/GSC/processed
python -m experiments.smnist.train --config configs/smnist/fits_o1_w256.yaml    --data-dir data/MNIST
```

| config | protocol | M | width | paper acc. (%) |
|---|---|:-:|:-:|:-:|
| [`shd/fits_o1_w128`](configs/shd/fits_o1_w128.yaml) | SHD, best test | 1 | 128 | 95.31 ± 0.21 |
| [`shd/fits_o2_w256_valid`](configs/shd/fits_o2_w256_valid.yaml) | SHD\*, 20% validation split | 2 | 256 | 94.38 ± 0.12 |
| [`ssc/fits_o1_w512`](configs/ssc/fits_o1_w512.yaml) | SSC | 1 | 512 | 78.23 ± 0.16 |
| [`gsc/fits_o1_w512`](configs/gsc/fits_o1_w512.yaml) | GSC | 1 | 512 | 94.48 ± 0.12 |
| [`smnist/fits_o1_w256`](configs/smnist/fits_o1_w256.yaml) | sMNIST, best test | 1 | 256 | 98.28 |

One run takes about 7 min on SHD, 24 min on GSC and 70 min on SSC (single RTX A5000). Checkpoints and logs are written to `results_dir`. Any config key can be overridden from the command line, e.g. `--order 0` for the FS-only model or `--neuron-type lif` for the LIF baseline. `scripts/run_seeds.sh` runs several seeds and reports mean ± std.

The scripts keep f\* below the frequency at which the discretized neuron becomes unstable (30.9 Hz for the GSC constants, 77.2 Hz for SHD/SSC). This has no effect unless a neuron reaches that limit; `--freq-guard false` turns it off.

## Using FiTS in your own model

```python
import torch
from fits import FiTSNeuron, FrequencyGuard

layer = FiTSNeuron(in_features=140, out_features=256, order=1,
                   tau_m=0.04, tau_a=0.2, dt=0.004, f_min=1.0, f_max=50.0)
x = (torch.rand(8, 250, 140) < 0.05).float()   # [batch, time, features]
spikes, *_ = layer(x)                          # [batch, time, 256]

optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
FrequencyGuard(layer, optimizer)       # optional, same guard as in the training scripts

layer.get_freq()                       # learned f* per neuron (Hz)
layer.ts_group_delay()                 # TS delay at each neuron's f* (s)
```

`dt` is the input time step in seconds, and `f_min`/`f_max` set the range of the initial f\*. [`examples/inspect_checkpoint.py`](examples/inspect_checkpoint.py) prints these quantities for a trained model.

## Code structure

```text
fits/          FiTS neuron, f* parameterization, models (pip install -e .)
experiments/   training scripts for SHD, SSC, GSC and sMNIST
configs/       one config per result in the table above
examples/      quickstart.py, inspect_checkpoint.py
scripts/       multi-seed runs
tests/         unit tests (python -m pytest tests)
```

## Citation

```bibtex
@inproceedings{choi2026fits,
  title     = {{FiTS}: Interpretable Spiking Neurons via Frequency Selectivity and Temporal Shaping},
  author    = {Choi, Jongmin and Chung, Joon Son},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Apache-2.0, see [LICENSE](LICENSE). The datasets keep their own licenses.
