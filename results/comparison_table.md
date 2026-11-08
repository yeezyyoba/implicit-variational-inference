# Comparison over 3 seeds (mean ± std)

KL(q‖p): binning-free kNN estimate of the reverse KL (the VI objective). KL(p‖q): forward KL on a 50×50 grid (punishes missing mass). TV: total variation on the same grid. Time: optimisation wall-clock on a 4-core CPU VM. Mixture: a mode counts if q puts ≥5% of its mass within 3 sd of it; 'mass between modes' exposes broad blobs that touch every mode without fitting any.

## Banana

| Method | KL(q‖p) | KL(p‖q) | TV | Time (s) |
|---|---|---|---|---|
| Gaussian VI | 0.470 ± 0.004 | 1.345 ± 0.013 | 0.351 ± 0.002 | 2 |
| Adversarial DRE | 0.007 ± 0.002 | 0.009 ± 0.002 | 0.033 ± 0.003 | 73 |
| KIVI | 0.159 ± 0.076 | 0.132 ± 0.071 | 0.162 ± 0.042 | 26 |
| SIVI | 0.013 ± 0.004 | 0.019 ± 0.004 | 0.046 ± 0.006 | 33 |

## X-shape

| Method | KL(q‖p) | KL(p‖q) | TV | Time (s) |
|---|---|---|---|---|
| Gaussian VI | 0.368 ± 0.002 | 1.069 ± 0.018 | 0.305 ± 0.001 | 2 |
| Adversarial DRE | 0.006 ± 0.005 | 0.015 ± 0.002 | 0.052 ± 0.007 | 74 |
| KIVI | 0.269 ± 0.051 | 0.468 ± 0.038 | 0.215 ± 0.025 | 28 |
| SIVI | 0.005 ± 0.002 | 0.014 ± 0.001 | 0.043 ± 0.001 | 32 |

## Gaussian mixture

| Method | KL(q‖p) | KL(p‖q) | TV | Time (s) | Modes with ≥5% mass | Mass between modes |
|---|---|---|---|---|---|---|
| Gaussian VI | 2.122 ± 0.266 | 1.986 ± 1.430 | 0.612 ± 0.034 | 2 | 2/4, 4/4, 4/4 | 18%, 40%, 40% |
| Adversarial DRE | 0.044 ± 0.018 | 0.035 ± 0.011 | 0.075 ± 0.018 | 180 | 4/4, 4/4, 4/4 | 3%, 3%, 3% |
| KIVI | 1.396 ± 0.026 | 3.768 ± 1.196 | 0.707 ± 0.034 | 65 | 2/4, 2/4, 1/4 | 9%, 9%, 3% |
| SIVI | 0.020 ± 0.001 | 0.023 ± 0.002 | 0.067 ± 0.005 | 72 | 4/4, 4/4, 4/4 | 2%, 2%, 2% |
