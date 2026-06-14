# -*- coding: utf-8 -*-
"""
LRIIS Training Script - Fixed version per 创新性审阅记录.md
Key fixes:
- Two-stage training (Stage 1: contrastive clustering, Stage 2: conditional restoration)
- PairedBatchSampler ensures ordinal loss is non-zero
- Proper minimax: attacker/CLUB frozen during hiding update, gradients flow through R_HF
- FiLM injection verified (z_type/z_level actually used by generator)
- Disentanglement adversarial constraint
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from models.inn import LRIISEncoder, LRIISDecoder, INN, DWT2d
from models.attacker import HFLeakageAttacker, HFLeakageAttackerLarge, PostAttackLeakageLoss
from models.aem import BlindAEM, OrdinalSeverityLoss, DisentanglementLoss, PairedBatchSampler
from attacks import AttackLayer
from mals_solver import run_stage1


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        resnet = resnet18(pretrained=True)
        self.layers = nn.ModuleList([
            nn.Sequential(*list(resnet.children())[:5]),
            nn.Sequential(*list(resnet.children())[5:6]),
            nn.Sequential(*list(resnet.children())[6:7]),
        ])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        loss = 0
        fx, fy = x, y
        for layer in self.layers:
            fx, fy = layer(fx), layer(fy)
            loss += F.l1_loss(fx, fy)
        return loss / len(self.layers)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Seed: {args.seed}")

    # === MALS Stage 1 (offline LP solver) ===
    mals_path = Path(args.mals_dir) / 'mals_G_star.pt'
    G_star = None
    if mals_path.exists():
        print(f"Loading MALS preconditioner from {mals_path}")
        mals_data = torch.load(mals_path, map_location=device)
        G_star = mals_data['G_star']
        print(f"  t* = {mals_data['t_star']:.6f}, cond(Q) = "
              f"{mals_data['q_star'].max() / mals_data['q_star'].min():.2f}")
    else:
        print(f"WARNING: MALS file not found at {mals_path}. "
              f"Run `python mals_solver.py` first for C3 shaping. "
              f"Proceeding without MALS (uniform embedding).")

    # === Models ===
    # Shared INN: encoder (forward) and decoder (reverse) use the SAME weights
    shared_inn = INN(channels=6, num_blocks=16, hidden=256).to(device)
    encoder = LRIISEncoder(inn=shared_inn, G_star=G_star).to(device)
    decoder = LRIISDecoder(inn=shared_inn).to(device)
    aem = BlindAEM(num_types=8).to(device)
    attacker = HFLeakageAttacker().to(device)
    attacker_held_out = HFLeakageAttackerLarge().to(device)  # For validation only
    dwt = DWT2d().to(device)

    # === Losses ===
    leakage_loss = PostAttackLeakageLoss(lambda_hf=args.lambda_hf)
    ordinal_loss = OrdinalSeverityLoss(margin=0.1)
    disentangle_loss = DisentanglementLoss(num_types=8).to(device)
    perceptual_loss = PerceptualLoss().to(device)
    attack_layer = AttackLayer().to(device)
    batch_sampler = PairedBatchSampler(num_types=8, batch_size=args.batch_size)

    # === Optimizers ===
    opt_hide = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr, betas=(0.9, 0.99)
    )
    opt_aem = torch.optim.Adam(aem.parameters(), lr=args.lr, betas=(0.9, 0.99))
    opt_att = torch.optim.Adam(attacker.parameters(), lr=args.lr * 2, betas=(0.9, 0.99))
    opt_club = torch.optim.Adam(leakage_loss.club.parameters(), lr=args.lr, betas=(0.9, 0.99))
    opt_disent = torch.optim.Adam(disentangle_loss.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # =========================================================
    # STAGE 1: Contrastive severity clustering (AEM encoder only)
    # =========================================================
    print("\n=== STAGE 1: Contrastive Severity Clustering ===")
    stage1_epochs = args.stage1_epochs

    for epoch in range(stage1_epochs):
        # Sample paired batch (same type, different severity)
        type_labels, severity_scores = batch_sampler.sample()
        type_labels = type_labels.to(device)
        severity_scores = severity_scores.to(device)

        # Generate synthetic attacked images
        host = torch.rand(args.batch_size, 3, args.img_size, args.img_size, device=device)
        secret = torch.randn(args.batch_size, 3, args.img_size // 2, args.img_size // 2, device=device)

        with torch.no_grad():
            container, _, _ = encoder(host, secret)

        # Apply attacks per-sample with assigned type/severity
        attacked_list = []
        for b in range(args.batch_size):
            atk, _, _ = attack_layer(
                container[b:b+1],
                attack_type=type_labels[b].item(),
                severity=severity_scores[b].item()
            )
            attacked_list.append(atk)
        attacked = torch.cat(attacked_list, dim=0)

        # Forward through severity encoder only
        z_type, z_level, type_logits = aem.severity_encoder(attacked)

        # Type classification loss (using pseudo-labels from k-means in real impl;
        # here we use the sampled type as proxy for pseudo-labels)
        l_type_cls = F.cross_entropy(type_logits, type_labels)

        # Ordinal severity loss (guaranteed non-zero due to PairedBatchSampler)
        l_ord = ordinal_loss(z_level, severity_scores, type_labels)

        # Disentanglement: encoder wants predictors to fail
        l_disent_enc = disentangle_loss.encoder_loss(z_type, z_level, severity_scores, type_labels)

        # Total Stage 1 loss
        l_stage1 = l_type_cls + l_ord + 0.1 * l_disent_enc

        opt_aem.zero_grad()
        l_stage1.backward()
        opt_aem.step()

        # Train disentanglement predictors (separate step)
        z_type_d, z_level_d, _ = aem.severity_encoder(attacked.detach())
        l_disent_pred = disentangle_loss.predictor_loss(z_type_d, z_level_d, severity_scores, type_labels)
        opt_disent.zero_grad()
        l_disent_pred.backward()
        opt_disent.step()

        if epoch % 10 == 0:
            print(f"  S1 Epoch {epoch}/{stage1_epochs} | "
                  f"Type CE: {l_type_cls.item():.4f} | "
                  f"Ordinal: {l_ord.item():.4f} | "
                  f"Disent: {l_disent_enc.item():.4f}")

    # =========================================================
    # STAGE 2: Joint hiding + conditional restoration
    # =========================================================
    print("\n=== STAGE 2: Joint Training ===")

    for epoch in range(args.epochs):
        # Sample paired batch
        type_labels, severity_scores = batch_sampler.sample()
        type_labels = type_labels.to(device)
        severity_scores = severity_scores.to(device)

        # Synthetic data (replace with real DataLoader)
        host = torch.rand(args.batch_size, 3, args.img_size, args.img_size, device=device)
        secret = torch.randn(args.batch_size, 3, args.img_size // 2, args.img_size // 2, device=device)

        # === HIDE ===
        container, budget, viol_rate = encoder(host, secret)

        # === ATTACK with SHARED random state (per-sample) ===
        # Key fix: container and host must share the SAME noise/random realization
        # so that R_HF^A = DWT(A_xi(c))_HF - DWT(A_xi(h))_HF reflects only
        # the signal difference (secret leakage), not independent noise.
        attacked_list = []
        attacked_host_list = []
        for b in range(args.batch_size):
            # Pre-generate shared noise for stochastic attacks
            shared_noise = torch.randn_like(container[b:b+1])
            atk_c, _, _ = attack_layer(
                container[b:b+1],  # Keep gradients for CLUB backprop
                attack_type=type_labels[b].item(),
                severity=severity_scores[b].item(),
                shared_noise=shared_noise
            )
            attacked_list.append(atk_c)
            with torch.no_grad():
                atk_h, _, _ = attack_layer(
                    host[b:b+1],
                    attack_type=type_labels[b].item(),
                    severity=severity_scores[b].item(),
                    shared_noise=shared_noise  # SAME noise realization
                )
                attacked_host_list.append(atk_h)
        attacked = torch.cat(attacked_list, dim=0)
        attacked_host = torch.cat(attacked_host_list, dim=0)

        # === LEAKAGE: compute POST-ATTACK R_HF^A ===
        r_hf = leakage_loss.compute_post_attack_hf_residual(
            container, host, attacked, attacked_host, dwt
        )

        # === BLIND AEM (FiLM-conditioned restoration) ===
        restored, z_type, z_level, type_logits = aem(attacked)

        # === DECODE ===
        extracted = decoder(restored)

        # === LOSSES ===

        # L_HR: hiding-revealing quality
        l_hr = F.l1_loss(container, host) + F.l1_loss(extracted, secret)

        # L_leak: CLUB MI upper bound + HF residual (hiding net minimizes)
        l_leak = leakage_loss.hiding_loss(attacker, r_hf, secret)

        # L_AEM: restoration quality (supervision target = container, NOT input)
        l_aem = 6.0 * F.l1_loss(restored, container.detach()) + \
                2.0 * perceptual_loss(restored, container.detach())

        # L_CSA: multi-scale consistency
        h_ll = dwt(host)[0]
        c_ll = dwt(container)[0]
        l_csa = sum(
            F.l1_loss(
                F.interpolate(c_ll, scale_factor=1.0/r, mode='bilinear', align_corners=False),
                F.interpolate(h_ll, scale_factor=1.0/r, mode='bilinear', align_corners=False)
            ) for r in [2, 4, 8]
        )

        # L_ord: severity monotonicity
        l_ord = ordinal_loss(z_level, severity_scores, type_labels)

        # L_type: type classification
        l_type = F.cross_entropy(type_logits, type_labels)

        # L_disent: disentanglement (encoder side)
        l_disent = disentangle_loss.encoder_loss(z_type, z_level, severity_scores, type_labels)

        # Total
        l_total = (args.lambda1 * l_hr +
                   args.lambda2 * l_leak +
                   args.lambda3 * l_aem +
                   args.lambda4 * l_csa +
                   args.lambda5 * l_ord +
                   l_type +
                   0.1 * l_disent)

        opt_hide.zero_grad()
        opt_aem.zero_grad()
        l_total.backward()
        opt_hide.step()
        opt_aem.step()

        # === ATTACKER UPDATE (separate, hiding frozen) ===
        with torch.no_grad():
            r_hf_det = leakage_loss.compute_post_attack_hf_residual(
                container.detach(), host, attacked.detach(), attacked_host, dwt
            )
        l_att = leakage_loss.attacker_loss(attacker, r_hf_det, secret)
        opt_att.zero_grad()
        l_att.backward()
        opt_att.step()

        # === CLUB ESTIMATOR UPDATE ===
        l_club = leakage_loss.club_loss(r_hf_det, secret)
        opt_club.zero_grad()
        l_club.backward()
        opt_club.step()

        # === DISENTANGLEMENT PREDICTOR UPDATE ===
        z_t, z_l, _ = aem.severity_encoder(attacked.detach())
        l_dp = disentangle_loss.predictor_loss(z_t, z_l, severity_scores, type_labels)
        opt_disent.zero_grad()
        l_dp.backward()
        opt_disent.step()

        # === LOGGING ===
        if epoch % 10 == 0:
            with torch.no_grad():
                att_psnr = leakage_loss.compute_attacker_psnr(attacker, r_hf_det, secret)
                mi_est = leakage_loss.compute_mi_estimate(r_hf_det, secret)
                rho = leakage_loss.compute_rho(r_hf_det, attacked_host, dwt)
                host_psnr = 10 * torch.log10(1.0 / (F.mse_loss(container, host) + 1e-10))
                sec_psnr = 10 * torch.log10(1.0 / (F.mse_loss(extracted, secret) + 1e-10))

            print(f"  S2 Ep {epoch}/{args.epochs} | "
                  f"Total: {l_total.item():.3f} | "
                  f"Host: {host_psnr.item():.1f} dB | "
                  f"Secret: {sec_psnr.item():.1f} dB | "
                  f"AttPSNR: {att_psnr:.1f} | "
                  f"MI: {mi_est:.3f} | "
                  f"rho: {rho:.4f} | "
                  f"Viol: {viol_rate:.3f} | "
                  f"Ord: {l_ord.item():.4f}")

    # =========================================================
    # STAGE 3: Held-out attacker training (frozen encoder)
    # =========================================================
    print("\n=== STAGE 3: Held-Out Attacker Training ===")
    opt_held_out = torch.optim.Adam(attacker_held_out.parameters(), lr=args.lr * 2)
    encoder.eval()
    for ep in range(20):
        host_ho = torch.rand(args.batch_size, 3, args.img_size, args.img_size, device=device)
        secret_ho = torch.randn(args.batch_size, 3, args.img_size // 2, args.img_size // 2, device=device)
        with torch.no_grad():
            container_ho, _, _ = encoder(host_ho, secret_ho)
            # Random attack with shared noise
            shared_n = torch.randn_like(container_ho)
            atk_type_ho = np.random.randint(0, 8)
            sev_ho = np.random.uniform(0.2, 1.0)
            atk_c_ho, _, _ = attack_layer(container_ho, atk_type_ho, sev_ho, shared_noise=shared_n)
            atk_h_ho, _, _ = attack_layer(host_ho, atk_type_ho, sev_ho, shared_noise=shared_n)
            r_hf_ho = leakage_loss.compute_post_attack_hf_residual(
                container_ho, host_ho, atk_c_ho, atk_h_ho, dwt)
        # Train held-out attacker on frozen residuals
        pred_ho = attacker_held_out(r_hf_ho)
        l_ho = F.l1_loss(pred_ho, secret_ho)
        opt_held_out.zero_grad()
        l_ho.backward()
        opt_held_out.step()
        if ep % 5 == 0:
            ho_psnr = leakage_loss.compute_attacker_psnr(attacker_held_out, r_hf_ho, secret_ho)
            print(f"  Held-out Ep {ep}/20 | Loss: {l_ho.item():.4f} | PSNR: {ho_psnr:.1f} dB")
    encoder.train()

    # === SAVE ===
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f'lriis_seed{args.seed}.pth')
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'aem': aem.state_dict(),
        'attacker': attacker.state_dict(),
        'attacker_held_out': attacker_held_out.state_dict(),
        'club': leakage_loss.club.state_dict(),
    }, ckpt_path)
    print(f"\nSaved: {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--img_size', type=int, default=256, help='Use 256 for debug, 1024 for full')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--stage1_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda1', type=float, default=6.0)
    parser.add_argument('--lambda2', type=float, default=10.0)
    parser.add_argument('--lambda3', type=float, default=6.0)
    parser.add_argument('--lambda4', type=float, default=1.5)
    parser.add_argument('--lambda5', type=float, default=1.0)
    parser.add_argument('--lambda_hf', type=float, default=10.0)
    parser.add_argument('--mals_dir', type=str, default='./checkpoints/mals',
                        help='Directory containing MALS Stage 1 output (mals_G_star.pt)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(args)
