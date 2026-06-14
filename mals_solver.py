# -*- coding: utf-8 -*-
"""
MALS (Minimax Attack-Family Leakage Shaping) Offline Solver

Stage 1 of LRIIS training pipeline:
1. Measure reduced Gram matrices H_i for m attack types
2. Solve structured LP to obtain optimal energy allocation q*
3. Compute preconditioner G* = U @ diag(sqrt(q*)) @ U^T

The LP depends only on attack operators {A_i} and wavelet W,
not on INN weights -- solved once before training.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linprog
from pathlib import Path

from models.inn import DWT2d, IDWT2d
from attacks import AttackLayer


class WaveletPacketBasis:
    """
    Constructs a p-dimensional orthonormal spectral basis U from wavelet packets.
    
    For p=64, we use 2-level wavelet packet decomposition of the LL subband,
    yielding 4^2 = 16 subbands per channel. With 3 RGB channels and spatial
    averaging, this gives p = 16 * 4 = 64 spectral directions.
    """
    def __init__(self, spatial_size, channels=3, p=64, device='cuda'):
        self.p = p
        self.device = device
        self.channels = channels
        self.H, self.W = spatial_size
        self.d = channels * self.H * self.W  # full LL dimension
        
        # Build orthonormal basis via wavelet packet decomposition
        self.U = self._build_basis()
    
    def _build_basis(self):
        """
        Build p orthonormal basis vectors in R^d.
        Each basis vector corresponds to one wavelet-packet subband direction.
        """
        dwt = DWT2d()
        # For p=64: use 16 frequency bands x 4 spatial quadrants
        # Simplified: use random orthonormal basis projected onto wavelet structure
        # In practice, use scipy's wavelet packet or fixed DCT-like basis
        
        d = self.channels * self.H * self.W
        # Generate random orthonormal basis (Gram-Schmidt on random matrix)
        # In full implementation, replace with actual wavelet-packet basis
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((d, self.p)).astype(np.float32)
        Q, _ = np.linalg.qr(raw)
        U = Q[:, :self.p]  # d x p orthonormal
        return torch.from_numpy(U).to(self.device)
    
    def project(self, delta_ll_vec):
        """Project vectorized LL perturbation onto spectral basis. [B, d] -> [B, p]"""
        return delta_ll_vec @ self.U  # [B, p]
    
    def reconstruct(self, coeffs):
        """Reconstruct from spectral coefficients. [B, p] -> [B, d]"""
        return coeffs @ self.U.T  # [B, d]


def measure_reduced_gram(attack_layer, dwt, idwt, basis, 
                         hosts, attack_type, probe_radius=0.01,
                         device='cuda'):
    """
    Measure p x p reduced Gram matrix for one attack type.
    
    H_i[a,b] = (1/N) * sum_n <W_HF[A(h_n + U*e_a*r) - A(h_n)], 
                               W_HF[A(h_n + U*e_b*r) - A(h_n)]> / r^2
    
    Args:
        attack_layer: differentiable attack module
        dwt: DWT2d instance
        idwt: IDWT2d instance
        basis: WaveletPacketBasis instance
        hosts: tensor [N, 3, H, W] - host images for probing
        attack_type: int - attack type index
        probe_radius: float - perturbation magnitude r
    
    Returns:
        H: [p, p] reduced Gram matrix
    """
    p = basis.p
    N = hosts.shape[0]
    C, Hh, Wh = 3, hosts.shape[2] // 2, hosts.shape[3] // 2
    
    # Storage for HF responses to each basis direction
    responses = torch.zeros(p, N, 9 * Hh * Wh, device=device)  # 9 = 3 subbands * 3 channels
    
    severity = torch.tensor([0.5], device=device)  # mid-severity for probing
    
    with torch.no_grad():
        for a in range(p):
            # Create perturbation in LL domain along basis direction e_a
            e_a = basis.U[:, a]  # [d]
            delta_ll = (e_a * probe_radius).reshape(1, C, Hh, Wh)  # [1, 3, H/2, W/2]
            delta_ll = delta_ll.expand(N, -1, -1, -1)
            
            # Get host LL
            h_ll, h_lh, h_hl, h_hh = dwt(hosts)
            
            # Perturbed container: add delta to LL only
            c_ll = h_ll + delta_ll
            container = idwt(c_ll, h_lh, h_hl, h_hh)
            
            # Apply attack to both
            for b_idx in range(0, N, 16):  # mini-batch
                batch_end = min(b_idx + 16, N)
                c_batch = container[b_idx:batch_end]
                h_batch = hosts[b_idx:batch_end]
                bs = c_batch.shape[0]
                
                atk_type_tensor = torch.full((bs,), attack_type, 
                                            dtype=torch.long, device=device)
                sev_tensor = severity.expand(bs)
                
                # Use shared noise for stochastic attacks
                shared_noise = torch.randn_like(c_batch)
                
                atk_c, _, _ = attack_layer(c_batch, atk_type_tensor, sev_tensor,
                                          shared_noise=shared_noise)
                atk_h, _, _ = attack_layer(h_batch, atk_type_tensor, sev_tensor,
                                          shared_noise=shared_noise)
                
                # Compute post-attack HF residual
                _, c_lh_a, c_hl_a, c_hh_a = dwt(atk_c)
                _, h_lh_a, h_hl_a, h_hh_a = dwt(atk_h)
                
                r_hf = torch.cat([c_lh_a - h_lh_a, 
                                  c_hl_a - h_hl_a, 
                                  c_hh_a - h_hh_a], dim=1)  # [bs, 9, Hh, Wh]
                
                responses[a, b_idx:batch_end] = r_hf.reshape(bs, -1) / probe_radius
    
    # Compute Gram: H[a,b] = (1/N) * sum_n <response_a_n, response_b_n>
    # responses: [p, N, D] where D = 9*Hh*Wh
    H = torch.zeros(p, p, device=device)
    for a in range(p):
        for b in range(a, p):
            inner = (responses[a] * responses[b]).sum(dim=1).mean()  # avg over N
            H[a, b] = inner
            H[b, a] = inner
    
    return H


def solve_mals_lp(gram_matrices, total_power=1.0, q_min=0.001, q_max=1.0):
    """
    Solve the structured minimax LP (Eq. 13 in paper):
    
    min_{q, t} t
    s.t. h_i^T q <= t,  for i = 1..m
         1^T q = P (total power)
         q_min <= q_k <= q_max
    
    Args:
        gram_matrices: list of [p, p] numpy arrays (reduced Gram per attack)
        total_power: P, total embedding power budget
        q_min: minimum per-direction power (preserves full-rank)
        q_max: maximum per-direction power
    
    Returns:
        q_star: optimal power allocation [p]
        t_star: minimax optimal leakage value
    """
    m = len(gram_matrices)
    p = gram_matrices[0].shape[0]
    
    # Extract diagonals: h_i[k] = H_i[k,k]
    H_diags = np.array([np.diag(H) for H in gram_matrices])  # [m, p]
    
    # Decision variables: [q_1, ..., q_p, t] -> p+1 variables
    # Objective: min t -> c = [0, ..., 0, 1]
    c = np.zeros(p + 1)
    c[-1] = 1.0
    
    # Inequality constraints: h_i^T q - t <= 0
    A_ub = np.zeros((m, p + 1))
    A_ub[:, :p] = H_diags  # h_i^T q
    A_ub[:, -1] = -1.0     # -t
    b_ub = np.zeros(m)
    
    # Equality constraint: 1^T q = P
    A_eq = np.zeros((1, p + 1))
    A_eq[0, :p] = 1.0
    b_eq = np.array([total_power])
    
    # Bounds: q_min <= q_k <= q_max, t >= 0
    bounds = [(q_min, q_max)] * p + [(0, None)]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    
    if not result.success:
        print(f"WARNING: LP solver failed: {result.message}")
        # Fallback: uniform allocation
        q_star = np.full(p, total_power / p)
        t_star = max(H_diags @ q_star)
    else:
        q_star = result.x[:p]
        t_star = result.x[-1]
    
    return q_star, t_star


def compute_preconditioner(q_star, basis):
    """
    Compute G* = U @ diag(sqrt(q*)) @ U^T
    
    The INN output is transformed as: Delta_LL = G* @ INN_out
    This shapes the embedding covariance to Q* = U @ diag(q*) @ U^T
    """
    sqrt_q = torch.sqrt(torch.from_numpy(q_star).float().to(basis.device))
    # G = U @ diag(sqrt_q) @ U^T
    G = basis.U @ torch.diag(sqrt_q) @ basis.U.T  # [d, d]
    return G


def run_stage1(args):
    """
    Complete Stage 1: measure Gram matrices and solve LP.
    
    Outputs:
        - mals_q_star.npy: optimal power allocation
        - mals_G_star.pt: preconditioner matrix
        - mals_gram_matrices.npy: measured Gram matrices for diagnostics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup
    dwt = DWT2d().to(device)
    idwt = IDWT2d().to(device)
    attack_layer = AttackLayer().to(device)
    
    img_size = args.img_size
    spatial_size = (img_size // 2, img_size // 2)
    basis = WaveletPacketBasis(spatial_size, channels=3, p=args.p, device=device)
    
    # Load or generate probe hosts
    print(f"Generating {args.num_hosts} random hosts for Gram measurement...")
    hosts = torch.rand(args.num_hosts, 3, img_size, img_size, device=device)
    
    # Measure Gram for each attack type
    num_attacks = 8
    gram_matrices = []
    
    for atk_id in range(num_attacks):
        print(f"  Measuring Gram for attack {atk_id}...")
        H = measure_reduced_gram(
            attack_layer, dwt, idwt, basis, hosts, 
            attack_type=atk_id, probe_radius=args.probe_radius, device=device
        )
        gram_matrices.append(H.cpu().numpy())
        print(f"    diag(H) range: [{np.diag(gram_matrices[-1]).min():.4f}, "
              f"{np.diag(gram_matrices[-1]).max():.4f}]")
    
    # Solve LP
    print("\nSolving minimax LP...")
    q_star, t_star = solve_mals_lp(
        gram_matrices, 
        total_power=args.total_power,
        q_min=args.q_min, 
        q_max=args.q_max
    )
    
    print(f"  Optimal t* = {t_star:.6f}")
    print(f"  q* range: [{q_star.min():.6f}, {q_star.max():.6f}]")
    print(f"  Condition number: {q_star.max() / q_star.min():.2f}")
    
    # Compute preconditioner
    G_star = compute_preconditioner(q_star, basis)
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / 'mals_q_star.npy', q_star)
    np.save(save_dir / 'mals_gram_matrices.npy', np.array(gram_matrices))
    torch.save({
        'G_star': G_star,
        'U': basis.U,
        'q_star': torch.from_numpy(q_star),
        't_star': t_star,
    }, save_dir / 'mals_G_star.pt')
    
    print(f"\nStage 1 complete. Saved to {save_dir}/")
    print(f"  mals_q_star.npy: optimal energy allocation ({args.p} dims)")
    print(f"  mals_G_star.pt: preconditioner G* (d x d)")
    print(f"  mals_gram_matrices.npy: {num_attacks} Gram matrices ({args.p} x {args.p})")
    
    return q_star, t_star, G_star


def recertify_after_clip(G_star, basis, attack_layer, dwt, idwt, 
                         hosts, B_cap=0.3, num_samples=100):
    """
    Post-projection re-certification (Eq. 16 in paper).
    After C1 clips Delta_LL to [-B, B], measure actual leakage.
    
    Returns:
        t_tilde: actual worst-case leakage after clipping
        ratio: t_tilde / t_star
    """
    device = hosts.device
    C, Hh, Wh = 3, hosts.shape[2] // 2, hosts.shape[3] // 2
    d = C * Hh * Wh
    
    num_attacks = 8
    leakage_per_attack = np.zeros(num_attacks)
    
    with torch.no_grad():
        for atk_id in range(num_attacks):
            total_leakage = 0.0
            for i in range(num_samples):
                # Generate shaped perturbation
                z = torch.randn(1, d, device=device)
                delta_vec = (z @ G_star.T)  # [1, d]
                delta_ll = delta_vec.reshape(1, C, Hh, Wh)
                
                # Clip (simulate C1 projection)
                delta_ll = torch.clamp(delta_ll, -B_cap, B_cap)
                
                # Apply to host
                h = hosts[i % hosts.shape[0]:i % hosts.shape[0] + 1]
                h_ll, h_lh, h_hl, h_hh = dwt(h)
                c = idwt(h_ll + delta_ll, h_lh, h_hl, h_hh)
                
                # Attack
                atk_type = torch.tensor([atk_id], device=device)
                sev = torch.tensor([0.5], device=device)
                shared_noise = torch.randn_like(c)
                
                atk_c, _, _ = attack_layer(c, atk_type, sev, shared_noise=shared_noise)
                atk_h, _, _ = attack_layer(h, atk_type, sev, shared_noise=shared_noise)
                
                # HF residual
                _, c_lh, c_hl, c_hh = dwt(atk_c)
                _, h_lh_a, h_hl_a, h_hh_a = dwt(atk_h)
                r_hf = torch.cat([c_lh - h_lh_a, c_hl - h_hl_a, c_hh - h_hh_a], dim=1)
                total_leakage += r_hf.pow(2).sum().item()
            
            leakage_per_attack[atk_id] = total_leakage / num_samples
    
    t_tilde = leakage_per_attack.max()
    return t_tilde


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MALS Stage 1: Offline LP Solver')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--p', type=int, default=64, help='Spectral basis dimension')
    parser.add_argument('--num_hosts', type=int, default=500, help='Number of probe hosts')
    parser.add_argument('--probe_radius', type=float, default=0.01)
    parser.add_argument('--total_power', type=float, default=1.0)
    parser.add_argument('--q_min', type=float, default=0.001, help='Min per-direction power')
    parser.add_argument('--q_max', type=float, default=1.0, help='Max per-direction power')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/mals')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    run_stage1(args)
