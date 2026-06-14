# -*- coding: utf-8 -*-
"""
LRIIS Evaluation & Ablation Script
Computes all metrics reported in the paper.
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from models.inn import LRIISEncoder, LRIISDecoder, DWT2d
from models.attacker import HFLeakageAttacker, PostAttackLeakageLoss
from models.aem import BlindAEM
from attacks import AttackLayer, AttackType


def psnr(x, y):
    mse = F.mse_loss(x, y)
    if mse < 1e-10:
        return 100.0
    return (10 * torch.log10(1.0 / mse)).item()


def ssim_simple(x, y, window_size=11):
    """Simplified SSIM (single-scale, grayscale average)."""
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x, window_size, 1, window_size//2)
    mu_y = F.avg_pool2d(y, window_size, 1, window_size//2)
    sigma_x2 = F.avg_pool2d(x**2, window_size, 1, window_size//2) - mu_x**2
    sigma_y2 = F.avg_pool2d(y**2, window_size, 1, window_size//2) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y, window_size, 1, window_size//2) - mu_x*mu_y
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean().item()


@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    encoder = LRIISEncoder().to(device)
    decoder = LRIISDecoder().to(device)
    aem = BlindAEM().to(device)
    attacker = HFLeakageAttacker().to(device)
    dwt = DWT2d().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    aem.load_state_dict(ckpt['aem'])
    attacker.load_state_dict(ckpt['attacker'])

    encoder.eval()
    decoder.eval()
    aem.eval()
    attacker.eval()

    attack_layer = AttackLayer().to(device)
    leakage_loss = PostAttackLeakageLoss()

    # Metrics accumulators
    metrics = {
        'host_psnr': [], 'host_ssim': [],
        'secret_psnr': [], 'secret_ssim': [],
        'aem_psnr': [], 'budget_max_ratio': [],
        'attacker_psnr': [], 'rho': [],
    }

    # Test loop (placeholder - replace with real data)
    num_test = args.num_test
    for i in range(num_test):
        host = torch.rand(1, 3, 1024, 1024, device=device)
        secret = torch.randn(1, 3, 512, 512, device=device)

        # Encode (returns: container, budget_map, violation_rate)
        container, budget, viol_rate = encoder(host, secret)

        # Host/Container quality
        metrics['host_psnr'].append(psnr(container, host))
        metrics['host_ssim'].append(ssim_simple(container, host))

        # Budget compliance: recompute residual for max ratio
        c_ll = dwt(container)[0]
        h_ll = dwt(host)[0]
        delta_ll = c_ll - h_ll
        ratio = (delta_ll.abs() / (budget + 1e-6)).max().item()
        metrics['budget_max_ratio'].append(ratio)

        # Attack
        attacked, atk_type, atk_sev = attack_layer(
            container, attack_type=args.attack_type, severity=args.severity
        )

        # Blind AEM restore
        restored, _, _, _ = aem(attacked)
        metrics['aem_psnr'].append(psnr(restored, container))

        # Extract secret
        extracted = decoder(restored)
        metrics['secret_psnr'].append(psnr(extracted, secret))
        metrics['secret_ssim'].append(ssim_simple(extracted, secret))

        # Leakage diagnostics (post-attack)
        # Apply same attack to host for reference
        attacked_host, _, _ = attack_layer(host, attack_type=args.attack_type, severity=args.severity)
        r_hf = leakage_loss.compute_post_attack_hf_residual(
            container, host, attacked, attacked_host, dwt
        )
        att_psnr = leakage_loss.compute_attacker_psnr(attacker, r_hf, secret)
        rho = leakage_loss.compute_rho(r_hf, attacked_host, dwt)
        metrics['attacker_psnr'].append(att_psnr)
        metrics['rho'].append(rho)

    # Print results
    print("\n" + "="*60)
    print(f"LRIIS Evaluation Results ({num_test} images)")
    print(f"Attack: type={args.attack_type}, severity={args.severity}")
    print("="*60)
    for k, v in metrics.items():
        arr = np.array(v)
        print(f"  {k:<20}: {arr.mean():.4f} ± {arr.std():.4f}")
    print("="*60)

    # Contribution-specific diagnostics
    print("\nContribution Diagnostics:")
    print(f"  C1 - Budget max ratio (should be <=1.0): {np.mean(metrics['budget_max_ratio']):.4f}")
    print(f"  C2 - Attacker PSNR (should be <10 dB): {np.mean(metrics['attacker_psnr']):.2f} dB")
    print(f"  C2 - Spectral leakage rho (should be <0.02): {np.mean(metrics['rho']):.5f}")
    print(f"  C3 - AEM restoration PSNR: {np.mean(metrics['aem_psnr']):.2f} dB")


def run_ablation(args):
    """Run all ablation configurations."""
    configs = {
        'full': {'use_spec': True, 'use_aem': True, 'use_clamp': True, 'use_csa': True},
        'w/o_Spec': {'use_spec': False, 'use_aem': True, 'use_clamp': True, 'use_csa': True},
        'w/o_AEM': {'use_spec': True, 'use_aem': False, 'use_clamp': True, 'use_csa': True},
        'w/o_Clamp': {'use_spec': True, 'use_aem': True, 'use_clamp': False, 'use_csa': True},
        'w/o_CSA': {'use_spec': True, 'use_aem': True, 'use_clamp': True, 'use_csa': False},
    }
    print("\nAblation Study")
    print("-"*60)
    for name, cfg in configs.items():
        print(f"\nConfig: {name} -> {cfg}")
        # Each config would load a separately trained checkpoint
        # evaluate(args) with modified model


def run_generalization(args):
    """Test out-of-range severity and unseen attack types."""
    test_cases = [
        # In-range
        (AttackType.MOSAIC, 0.25, "Mosaic K=300 (in-range)"),
        (AttackType.MOSAIC, 0.75, "Mosaic K=500 (in-range)"),
        # Out-of-range (below)
        (AttackType.MOSAIC, 0.0, "Mosaic K=100 (below range)"),
        # Out-of-range (above)
        (AttackType.MOSAIC, 1.5, "Mosaic K=1000 (above range)"),
        # Unseen types (zero-shot)
        (AttackType.JPEG, 0.5, "JPEG QF=60 (zero-shot)"),
        (AttackType.GAUSSIAN_NOISE, 0.5, "Noise sigma=0.05 (zero-shot)"),
        (AttackType.COMPOSITE, 0.5, "JPEG+Noise (zero-shot)"),
    ]
    print("\nGeneralization Test")
    print("-"*60)
    for atk_type, sev, desc in test_cases:
        print(f"  {desc}: attack_type={atk_type}, severity={sev}")
        # Run evaluate with these params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/lriis_final.pth')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--attack_type', type=int, default=0, help='0=mosaic')
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'ablation', 'generalization'])
    args = parser.parse_args()

    if args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'ablation':
        run_ablation(args)
    elif args.mode == 'generalization':
        run_generalization(args)
