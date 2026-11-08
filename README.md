# Implicit Variational Inference with Adversarial Density-Ratio Estimation

iCog Labs — Saturday AI training: *Implicit Variational Inference*.

Work in progress. The goal is to fit **implicit** variational posteriors (a neural
sampler `z = G(ε)` with no tractable density) to hard 2D posteriors, by estimating
the intractable `log q(z)` term with a learned classifier (adversarial
density-ratio estimation), and to compare against an explicit Gaussian VI baseline
and two alternative implicit methods (KIVI and SIVI).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
