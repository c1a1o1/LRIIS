# LRIIS Code

## Overview

**LRIIS: Certified Low-Frequency Image Hiding with Cross-Band Leakage Theory and Minimax Embedding Shaping**

This directory contains the implementation code for the LRIIS framework, which hides full-size secret images (6.0 bpp) in the low-frequency wavelet subband with three certified guarantees:

- **C1 (Post-INN Projection):** Deterministic pixel-domain output certificate with 100% compliance
- **C2 (CLUB Leakage Suppression):** Conditional Likelihood Upper Bound MI minimization drives attacker recovery to zero-predictor baseline
- **C3 (MALS):** Minimax Allocation over Leakage Subbands via LP over 64 wavelet-packet subbands

## Structure

```
code/
├── models/
│   ├── __init__.py
│   ├── inn.py          # INN + DWT/IDWT + Post-INN Projection (C1)
│   ├── attacker.py     # HF Leakage Attacker + CLUB MI Loss (C2)
│   └── aem.py          # Blind AEM (ResNet-18 encoder + U-Net decoder + FiLM)
├── attacks.py           # 8-type differentiable attack layer
├── train.py             # Main training script (3-stage)
├── evaluate.py          # Evaluation + ablation + generalization tests
├── draw_all_paper_figures.py  # Generate all paper figures
├── draw_fig6_ablation.py      # Figure 6: Contribution-specific ablation chart
├── requirements.txt
└── README.md
```

## Key Results

| Method | Host PSNR (DIV2K) | Secret PSNR (DIV2K) | Host PSNR (COCO) | Secret PSNR (COCO) |
|--------|:-:|:-:|:-:|:-:|
| HiNet | 30.69 | 32.56 | 28.69 | 30.01 |
| RIIS | 35.43 | 36.48 | 33.08 | 34.31 |
| CrossNET | 38.19 | 39.58 | 36.49 | 37.39 |
| LIDS | 37.84 | 38.92 | 35.67 | 36.74 |
| **LRIIS (Ours)** | **41.99** | **42.82** | **39.52** | **40.51** |

### Ablation Study

| Variant | Secret PSNR | Drop |
|---------|:-:|:-:|
| w/o CLUB | 34.22 | −8.60 dB |
| w/o AEM | 38.15 | −4.67 dB |
| w/o Projection | 39.73 | −3.09 dB |
| w/o CSA | 41.08 | −1.74 dB |
| **Full LRIIS** | **42.82** | — |

### Contribution-Specific Ablations

**C1 - Projection Strategies:**
| Strategy | PSNR | Budget Viol. (%) |
|----------|:-:|:-:|
| Constant (λ=0.5) | 39.73 | 14.2 |
| Luminance masking | 40.51 | 8.7 |
| Gradient masking | 40.89 | 6.3 |
| **Post-INN Proj (Ours)** | **42.82** | **0.0** |

**C2 - Leakage Suppression:**
| Variant | Secret PSNR | Att. PSNR ↓ |
|---------|:-:|:-:|
| No loss | 34.22 | 12.4 |
| HF L1 only | 40.14 | 6.8 |
| Focal Freq | 41.07 | 4.1 |
| CLUB only | 41.53 | 1.2 |
| **CLUB + L1 (Ours)** | **42.82** | **0.3** |

**C3 - MALS Strategies:**
| Strategy | Secret PSNR | Leakage t̃ ↓ |
|----------|:-:|:-:|
| Uniform | 40.3 | 1.00 |
| Random | 40.1 | 0.87 |
| Single-atk (JPEG) | 41.8 | 0.41 |
| Average-atk | 41.5 | 0.52 |
| **MALS minimax (Ours)** | **42.8** | **0.31** |

## Training

```bash
# Stage 1: Offline leakage coefficient estimation (~2h)
python train.py --stage 1 --data_root ./data

# Stage 2: Joint end-to-end training (80 epochs, ~18h)
python train.py --stage 2 --data_root ./data --epochs 80 --batch_size 4 --seed 42

# Stage 3: Held-out attacker verification (10 epochs, frozen INN+AEM)
python train.py --stage 3 --checkpoint ./checkpoints/stage2.pth
```

Seeds for 5-run reproducibility: `{42, 123, 2024, 7777, 99999}`

## Evaluation

```bash
# Standard evaluation
python evaluate.py --checkpoint ./checkpoints/lriis_final.pth --mode eval

# Ablation study
python evaluate.py --mode ablation

# Out-of-range generalization
python evaluate.py --mode generalization
```

## Figure Generation

```bash
# Generate all paper figures
python draw_all_paper_figures.py
# Output: ../generated_figures/

# Generate Figure 6 (contribution-specific ablation)
python draw_fig6_ablation.py
# Output: ../generated_figures/fig6_contribution_ablation.png + ../f88.png
```

## Paper Correspondence

| Paper Section | Code File | Key Class/Function |
|---|---|---|
| §3.1 Post-INN Projection (C1) | `models/inn.py` | `PostINNProjection`, `LRIISEncoder` |
| §3.1 Budget Definition | `models/inn.py` | `compute_adaptive_budget()` |
| §3.2 Cross-Band Leakage (C2) | `models/attacker.py` | `CLUBLoss`, `HFLeakageAttacker` |
| §3.3 MALS LP (C3) | `train.py` | `solve_mals_lp()`, `MALSLoss` |
| §3.4 INN Architecture | `models/inn.py` | `AffineCouplingINN` (16 blocks, 48M params) |
| §3.4 AEM | `models/aem.py` | `BlindAEM`, `SeverityEncoder`, `UNetDecoder` |
| §3.4 Training | `train.py` | `three_stage_training()` |
| Eq.(total) | `train.py` | `l_total` (λ1=6, λ2=10, λ3=6, λ4=1.5, λ5=0.5) |

## Requirements

- Python 3.8+
- PyTorch 1.12+
- numpy, matplotlib, scipy
- pywt (PyWavelets)
- RTX 3090 Ti (24GB) recommended

## License

Code will be released under the MIT License upon publication.
