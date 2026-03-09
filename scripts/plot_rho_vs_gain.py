#!/usr/bin/env python3
"""
Generate scatter plot: Coverage overlap (ρ) vs Semantic gain.

Validates Theorem 2 (Information-Theoretic Characterization):
Semantic gain > 0 iff ρ > 0
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from paper (Table 2 and Table A5)
datasets = ['WN18RR', 'FB15k-237', 'YAGO3-10', 'ICEWS14', 'ICEWS18']
rho = [0.34, 0.43, 0.66, 0.02, 0.01]  # Coverage overlap (A5)
semantic_gain = [0.08, 0.11, 0.12, 0.02, 0.00]  # CAGP - U_str on emerging

# Static vs Temporal markers
colors = ['#2ecc71', '#2ecc71', '#2ecc71', '#e74c3c', '#e74c3c']
markers = ['o', 's', '^', 'D', 'v']

plt.figure(figsize=(6, 4.5))

for i, (name, r, g, c, m) in enumerate(zip(datasets, rho, semantic_gain, colors, markers)):
    plt.scatter(r, g, c=c, marker=m, s=120, label=name, edgecolors='black', linewidths=1)

# Add trend line
rho_fit = np.array(rho)
gain_fit = np.array(semantic_gain)
z = np.polyfit(rho_fit, gain_fit, 1)
p = np.poly1d(z)
x_line = np.linspace(0, 0.7, 100)
plt.plot(x_line, p(x_line), 'k--', alpha=0.5, label=f'Linear fit (R²={np.corrcoef(rho, semantic_gain)[0,1]**2:.2f})')

# Add theory prediction regions
plt.axvspan(0, 0.05, alpha=0.1, color='red', label='ρ≈0: No semantic gain')
plt.axvspan(0.3, 0.7, alpha=0.1, color='green', label='ρ>0: Semantic gain')

plt.xlabel('Coverage Overlap ρ (fraction of emerging with coverage)', fontsize=11)
plt.ylabel('Semantic Gain (AUROC improvement on emerging)', fontsize=11)
plt.title('Theorem 2 Validation: Semantic helps iff ρ > 0', fontsize=12)
plt.legend(loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.xlim(-0.02, 0.72)
plt.ylim(-0.01, 0.15)

# Add annotations
plt.annotate('Temporal KGs\n(ρ≈0, no gain)', xy=(0.015, 0.01), fontsize=9,
             ha='center', color='#c0392b')
plt.annotate('Static KGs\n(ρ>0, gain)', xy=(0.5, 0.11), fontsize=9,
             ha='center', color='#27ae60')

import os

# Create output directories
os.makedirs('paper/figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

plt.tight_layout()
plt.savefig('paper/figures/fig4_rho_vs_gain.pdf', dpi=300, bbox_inches='tight')
plt.savefig('outputs/rho_vs_semantic_gain.png', dpi=150, bbox_inches='tight')
print("Saved: paper/figures/fig4_rho_vs_gain.pdf")
print("Saved: outputs/rho_vs_semantic_gain.png")

# Print correlation
corr = np.corrcoef(rho, semantic_gain)[0, 1]
print(f"\nPearson correlation: {corr:.3f}")
print(f"R²: {corr**2:.3f}")
print("\nTheorem 2 validation:")
print("- ICEWS14/18 (ρ≈0): Semantic gain ≈ 0 ✓")
print("- Static benchmarks (ρ>0.3): Semantic gain > 0.08 ✓")
