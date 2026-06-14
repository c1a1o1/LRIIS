"""
Figure 6: Contribution-Specific Ablation Summary
基于论文 §4.3.2 Contribution-Specific Ablations 的三组消融实验数据
生成综合对比图，展示C1/C2/C3各贡献的细粒度消融结果
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ============ Panel (a): C1 Projection Strategies ============
ax1 = axes[0]
strategies = ['Constant\n($\\lambda$=0.5)', 'Luminance\nmasking', 'Gradient\nmasking', 'Post-INN\nProj (Ours)']
psnr_c1 = [39.73, 40.51, 40.89, 42.82]
budget_viol = [14.2, 8.7, 6.3, 0.0]

x = np.arange(len(strategies))
width = 0.38

bars1 = ax1.bar(x - width/2, psnr_c1, width, color='#4C72B0', alpha=0.85, label='Secret PSNR (dB)', edgecolor='black', linewidth=0.5)
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, budget_viol, width, color='#DD8452', alpha=0.85, label='Budget Viol. (%)', edgecolor='black', linewidth=0.5)

ax1.set_ylim(38.5, 43.5)
ax1_twin.set_ylim(0, 18)
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=8)
ax1.set_ylabel('Secret PSNR (dB)', fontsize=9, color='#4C72B0')
ax1_twin.set_ylabel('Budget Violation (%)', fontsize=9, color='#DD8452')
ax1.set_title('(a) C1: Projection Strategies', fontsize=10, fontweight='bold')

# 标注数值
for bar, val in zip(bars1, psnr_c1):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}', 
             ha='center', va='bottom', fontsize=7, color='#4C72B0', fontweight='bold')
for bar, val in zip(bars2, budget_viol):
    ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', 
                  ha='center', va='bottom', fontsize=7, color='#DD8452', fontweight='bold')

# 高亮 Ours
bars1[-1].set_edgecolor('red')
bars1[-1].set_linewidth(1.5)
bars2[-1].set_edgecolor('red')
bars2[-1].set_linewidth(1.5)

ax1.legend(loc='upper left', fontsize=7, framealpha=0.8)
ax1_twin.legend(loc='upper right', fontsize=7, framealpha=0.8)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ============ Panel (b): C2 Leakage Suppression Variants ============
ax2 = axes[1]
variants_c2 = ['No loss', 'HF L1\nonly', 'Focal\nFreq', 'CLUB\nonly', 'CLUB+L1\n(Ours)']
psnr_c2 = [34.22, 40.14, 41.07, 41.53, 42.82]
att_psnr = [12.4, 6.8, 4.1, 1.2, 0.3]

x2 = np.arange(len(variants_c2))

# PSNR 柱状图
bars_psnr = ax2.bar(x2 - width/2, psnr_c2, width, color='#55A868', alpha=0.85, 
                     label='Secret PSNR (dB)', edgecolor='black', linewidth=0.5)
ax2_twin = ax2.twinx()
# Attacker PSNR 折线图 + 散点
line = ax2_twin.plot(x2, att_psnr, 'rs-', markersize=7, linewidth=2, label='Att. PSNR (dB) ↓', alpha=0.9)
ax2_twin.fill_between(x2, 0, att_psnr, alpha=0.1, color='red')

ax2.set_ylim(32, 44)
ax2_twin.set_ylim(0, 15)
ax2.set_xticks(x2)
ax2.set_xticklabels(variants_c2, fontsize=8)
ax2.set_ylabel('Secret PSNR (dB)', fontsize=9, color='#55A868')
ax2_twin.set_ylabel('Attacker PSNR (dB) ↓', fontsize=9, color='red')
ax2.set_title('(b) C2: Leakage Suppression', fontsize=10, fontweight='bold')

for bar, val in zip(bars_psnr, psnr_c2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}', 
             ha='center', va='bottom', fontsize=7, color='#55A868', fontweight='bold')
for i, val in enumerate(att_psnr):
    ax2_twin.text(x2[i]+0.05, val + 0.4, f'{val:.1f}', ha='left', va='bottom', fontsize=7, color='red')

bars_psnr[-1].set_edgecolor('red')
bars_psnr[-1].set_linewidth(1.5)

ax2.legend(loc='upper left', fontsize=7, framealpha=0.8)
ax2_twin.legend(loc='upper right', fontsize=7, framealpha=0.8)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# ============ Panel (c): C3 MALS Strategies ============
ax3 = axes[2]
strategies_c3 = ['Uniform', 'Random', 'Single-atk\n(JPEG)', 'Average-\natk', 'MALS\n(Ours)']
secret_psnr_c3 = [40.3, 40.1, 41.8, 41.5, 42.8]
leakage_t = [1.00, 0.87, 0.41, 0.52, 0.31]

x3 = np.arange(len(strategies_c3))

bars_sec = ax3.bar(x3 - width/2, secret_psnr_c3, width, color='#8172B2', alpha=0.85, 
                    label='Secret PSNR (dB)', edgecolor='black', linewidth=0.5)
ax3_twin = ax3.twinx()
bars_leak = ax3_twin.bar(x3 + width/2, leakage_t, width, color='#CCB974', alpha=0.85, 
                          label='Leakage $\\tilde{t}$ ↓', edgecolor='black', linewidth=0.5)

ax3.set_ylim(39, 43.5)
ax3_twin.set_ylim(0, 1.2)
ax3.set_xticks(x3)
ax3.set_xticklabels(strategies_c3, fontsize=8)
ax3.set_ylabel('Secret PSNR (dB)', fontsize=9, color='#8172B2')
ax3_twin.set_ylabel('Empirical Leakage $\\tilde{t}$ ↓', fontsize=9, color='#CCB974')
ax3.set_title('(c) C3: MALS Strategies', fontsize=10, fontweight='bold')

for bar, val in zip(bars_sec, secret_psnr_c3):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.1f}', 
             ha='center', va='bottom', fontsize=7, color='#8172B2', fontweight='bold')
for bar, val in zip(bars_leak, leakage_t):
    ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                  ha='center', va='bottom', fontsize=7, color='#997A00', fontweight='bold')

bars_sec[-1].set_edgecolor('red')
bars_sec[-1].set_linewidth(1.5)
bars_leak[-1].set_edgecolor('red')
bars_leak[-1].set_linewidth(1.5)

ax3.legend(loc='upper left', fontsize=7, framealpha=0.8)
ax3_twin.legend(loc='upper right', fontsize=7, framealpha=0.8)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(pad=1.5)
plt.savefig('f:/BaiduSyncdisk/tongbu/视频信息隐藏2023/ELSE__LOW___IMAGE_steg/generated_figures/fig6_contribution_ablation.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('f:/BaiduSyncdisk/tongbu/视频信息隐藏2023/ELSE__LOW___IMAGE_steg/f88.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("Figure 6 saved: fig6_contribution_ablation.png & f88.png")
plt.close()
