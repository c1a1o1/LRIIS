# -*- coding: utf-8 -*-
"""
Differentiable attack simulation layer (8 attack types).
Training: randomly sample type + severity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import IntEnum


class AttackType(IntEnum):
    MOSAIC = 0
    JPEG = 1
    GAUSSIAN_NOISE = 2
    GAUSSIAN_BLUR = 3
    RESIZE = 4
    ROTATION = 5
    CROP = 6
    COMPOSITE = 7


class DifferentiableJPEG(nn.Module):
    """
    Differentiable JPEG approximation with DCT quantization + STE.
    Uses block-wise DCT, quantization with straight-through estimator,
    and inverse DCT -- the actual JPEG pipeline minus entropy coding.
    """
    def __init__(self):
        super().__init__()
        # Standard JPEG luminance quantization table (8x8)
        q_table = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ], dtype=torch.float32)
        self.register_buffer('q_table', q_table)

    def _dct_8x8(self, x):
        """Apply 8x8 block DCT (simplified via matrix multiply)."""
        B, C, H, W = x.shape
        # Pad to multiple of 8
        pH = (8 - H % 8) % 8
        pW = (8 - W % 8) % 8
        if pH > 0 or pW > 0:
            x = F.pad(x, (0, pW, 0, pH), mode='reflect')
        B, C, H, W = x.shape
        # Reshape into 8x8 blocks
        x = x.reshape(B, C, H // 8, 8, W // 8, 8)
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(-1, 8, 8)
        # DCT via cosine basis (approximate with learnable-free transform)
        n = torch.arange(8, device=x.device, dtype=x.dtype)
        basis = torch.cos(torch.pi * (2 * n.unsqueeze(1) + 1) * n.unsqueeze(0) / 16)
        basis[0] *= 1.0 / (2 ** 0.5)
        basis = basis * (2.0 / 8) ** 0.5
        dct = torch.matmul(basis.T, torch.matmul(x, basis))
        return dct, (B, C, H, W)

    def _idct_8x8(self, dct, shape):
        """Inverse 8x8 block DCT."""
        B, C, H, W = shape
        n = torch.arange(8, device=dct.device, dtype=dct.dtype)
        basis = torch.cos(torch.pi * (2 * n.unsqueeze(1) + 1) * n.unsqueeze(0) / 16)
        basis[0] *= 1.0 / (2 ** 0.5)
        basis = basis * (2.0 / 8) ** 0.5
        x = torch.matmul(basis, torch.matmul(dct, basis.T))
        x = x.reshape(B, C, H // 8, W // 8, 8, 8)
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        return x

    def forward(self, x, quality):
        """
        x: [B, C, H, W] in [0, 1]
        quality: int (1-100)
        Returns: JPEG-approximated image (differentiable via STE on quantization)
        """
        # Scale factor from quality
        if quality < 50:
            s = 5000.0 / quality
        else:
            s = 200.0 - 2.0 * quality
        q_scaled = torch.clamp(torch.floor((self.q_table * s + 50) / 100), 1, 255)

        # Shift to [-128, 128] range (JPEG convention)
        x_shifted = x * 255.0 - 128.0

        dct, shape = self._dct_8x8(x_shifted)

        # Quantization with STE
        q_scaled_flat = q_scaled.reshape(1, 8, 8).to(dct.device)
        quantized = torch.round(dct / q_scaled_flat)  # Forward: round
        # STE: backward treats round as identity
        dct_ste = dct + (quantized * q_scaled_flat - dct).detach()

        # IDCT
        x_recon = self._idct_8x8(dct_ste, shape)
        x_recon = (x_recon + 128.0) / 255.0
        return torch.clamp(x_recon[:, :, :x.shape[2], :x.shape[3]], 0, 1)


class AttackLayer(nn.Module):
    """
    Simulates 8 attack types with continuous severity.
    Returns: attacked image, attack_type (int), severity (float 0-1)
    """
    def __init__(self):
        super().__init__()
        self.diff_jpeg = DifferentiableJPEG()

    def mosaic(self, x, K):
        """Mosaic attack with K blocks."""
        B, C, H, W = x.shape
        j = max(1, int(H / np.sqrt(K)))
        out = x.clone()
        for bi in range(0, H, j):
            for bj in range(0, W, j):
                out[:, :, bi:bi+j, bj:bj+j] = x[:, :, bi:min(bi+1, H), bj:min(bj+1, W)]
        return out

    def gaussian_noise(self, x, sigma, noise=None):
        """Additive Gaussian noise. If noise is provided, use it (shared realization)."""
        if noise is None:
            noise = torch.randn_like(x)
        return x + noise * sigma

    def gaussian_blur(self, x, kernel_size):
        k = int(kernel_size) | 1  # ensure odd
        if k < 3: k = 3
        sigma = k / 3.0
        # Create Gaussian kernel
        ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        kernel = torch.exp(-ax**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel_2d = kernel.unsqueeze(0) * kernel.unsqueeze(1)
        kernel_2d = kernel_2d.expand(x.shape[1], 1, k, k)
        return F.conv2d(x, kernel_2d, padding=k//2, groups=x.shape[1])

    def resize_attack(self, x, scale):
        B, C, H, W = x.shape
        small = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        return F.interpolate(small, size=(H, W), mode='bilinear', align_corners=False)

    def rotation_attack(self, x, angle_deg):
        angle_rad = angle_deg * np.pi / 180.0
        B = x.shape[0]
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        theta = torch.tensor([[cos_a, -sin_a, 0],
                              [sin_a, cos_a, 0]], dtype=x.dtype, device=x.device)
        theta = theta.unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode='reflection')

    def crop_attack(self, x, ratio):
        B, C, H, W = x.shape
        margin = int(H * ratio)
        cropped = x[:, :, margin:H-margin, margin:W-margin]
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)

    def forward(self, x, attack_type=None, severity=None, shared_noise=None):
        """
        x: [B, 3, H, W]
        attack_type: int (0-7) or None (random)
        severity: float (0-1) or None (random)
        shared_noise: pre-generated noise tensor for coupled container/host attacks.
            When computing R_HF^A = DWT(A(c))_HF - DWT(A(h))_HF, the SAME noise
            realization must be used for both c and h to avoid noise-dominated residual.
        Returns: attacked, type_label, severity_score
        """
        B = x.shape[0]
        if attack_type is None:
            attack_type = np.random.randint(0, 8)
        if severity is None:
            severity = np.random.uniform(0.2, 1.0)

        if attack_type == AttackType.MOSAIC:
            K = int(200 + severity * 400)  # K in [200, 600]
            attacked = self.mosaic(x, K)
        elif attack_type == AttackType.JPEG:
            qf = int(90 - severity * 60)  # QF in [30, 90]
            attacked = self.diff_jpeg(x, qf)
        elif attack_type == AttackType.GAUSSIAN_NOISE:
            sigma = 0.01 + severity * 0.09  # sigma in [0.01, 0.1]
            attacked = self.gaussian_noise(x, sigma, noise=shared_noise)
        elif attack_type == AttackType.GAUSSIAN_BLUR:
            k = 3 + severity * 6  # kernel in [3, 9]
            attacked = self.gaussian_blur(x, k)
        elif attack_type == AttackType.RESIZE:
            scale = 1.0 - severity * 0.5  # scale in [0.5, 1.0]
            attacked = self.resize_attack(x, scale)
        elif attack_type == AttackType.ROTATION:
            angle = severity * 30.0  # angle in [0, 30]
            attacked = self.rotation_attack(x, angle)
        elif attack_type == AttackType.CROP:
            ratio = 0.05 + severity * 0.20  # crop in [0.05, 0.25]
            attacked = self.crop_attack(x, ratio)
        elif attack_type == AttackType.COMPOSITE:
            # JPEG + Noise (shared noise for the additive component)
            qf = int(90 - severity * 40)
            sigma = severity * 0.05
            attacked = self.diff_jpeg(x, qf)
            attacked = self.gaussian_noise(attacked, sigma, noise=shared_noise)
        else:
            attacked = x

        return attacked, attack_type, severity
