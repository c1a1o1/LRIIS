# LRIIS Code

## Structure

```
code/
├── models/
│   ├── __init__.py
│   ├── inn.py          # INN + DWT/IDWT + AmplitudeClamp + HFScaling (Contribution 1)
│   ├── attacker.py     # HF Leakage Attacker + LeakageLoss (Contribution 2)
│   └── aem.py          # Blind AEM + SeverityEncoder + OrdinalLoss (Contribution 3)
├── attacks.py           # 8-type differentiable attack layer
├── train.py             # Main training script
├── evaluate.py          # Evaluation + ablation + generalization tests
├── requirements.txt
└── README.md
```

## Training

```bash
python train.py --data_root ./data --epochs 80 --batch_size 4 --seed 42
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

## Paper Correspondence

| Paper Section | Code File | Key Class/Function |
|---|---|---|
| §3.1 Amplitude-Clipped Embedding | `models/inn.py` | `AmplitudeClamp`, `LRIISEncoder` |
| §3.2 Secret-Conditioned Leakage | `models/attacker.py` | `HFLeakageAttacker`, `LeakageLoss` |
| §3.3 Type-Severity Blind AEM | `models/aem.py` | `BlindAEM`, `SeverityEncoder`, `OrdinalSeverityLoss` |
| §3.3 Training attacks | `attacks.py` | `AttackLayer` (8 types) |
| Eq.(budget) | `models/inn.py` | `AmplitudeClamp.compute_budget()` |
| Eq.(clamp) | `models/inn.py` | `AmplitudeClamp.forward()` |
| Eq.(hf) + Eq.(att) | `models/attacker.py` | `LeakageLoss` |
| Eq.(blind_aem) | `models/aem.py` | `BlindAEM.forward()` |
| Eq.(ordinal) | `models/aem.py` | `OrdinalSeverityLoss.forward()` |
| Eq.(total) | `train.py` | `l_total` computation |
