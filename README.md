# LRIIS: Low-Frequency Robust Image Hiding via Wavelet Contrastive Learning

**Authors:** Zhiyi Cao, Lina Huo, Wei Wang, Shaozhang Niu

**Affiliation:** College of Computer and Cyberspace Security, Hebei Normal University / Beijing University of Posts and Telecommunications

---

## Abstract

Existing high-frequency image information hiding methods suffer from limited robustness against container image distortions and depend on predefined attack labels. We propose LRIIS, a low-frequency embedding framework that leverages an invertible neural network (INN) with dynamic adaptive strength for secret data concealment. To enhance robustness, we introduce an uncertainty-aware wavelet contrastive learning module (WC) that separates high-frequency subbands from low-frequency subbands, reducing cross-frequency artifacts. Furthermore, we design an unsupervised Attacked Image Enhancement Module (AEM) with multi-scale consistency constraints to generate attack-adaptive de-noised images. Unlike prior methods that require per-sample attack type labels for conditioning, AEM operates with only the feasible range of attack parameters known during training, enabling robust extraction under unknown attack severities at inference time. Experimental results on COCO and DIV2K demonstrate that LRIIS outperforms state-of-the-art methods by 3–5 dB in PSNR and 0.02–0.05 in SSIM for secret image extraction, while also generalizing to unseen distortions including JPEG compression, Gaussian noise, and geometric attacks without retraining.

---

## Key Contributions

1. **Low-frequency wavelet embedding** with dynamic adaptive strength balances imperceptibility and robustness.
2. **Uncertainty-aware wavelet contrastive loss** separates high/low-frequency subbands to suppress cross-frequency artifacts.
3. **Unsupervised Attacked Image Enhancement Module (AEM)** restores distorted images using attack parameter range, without per-sample labels.

---

## Code Availability

> ⚠️ **The source code and pre-trained models will be made publicly available upon paper acceptance.**

This repository currently contains the paper manuscript. The full implementation (training code, evaluation scripts, and pre-trained models) will be released after the paper is accepted for publication.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{cao2025lriis,
  title={LRIIS: Low-Frequency Robust Image Hiding via Wavelet Contrastive Learning},
  author={Cao, Zhiyi and Huo, Lina and Wang, Wei and Niu, Shaozhang},
  year={2025}
}
```

## License

Code will be released under the MIT License upon publication.
