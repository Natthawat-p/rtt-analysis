"""
Schmidt 1986 validation plot in 'paper-style' format
====================================================
Header on the plot lists all test parameters explicitly,
following the reference format requested by the supervisor
(RCS plot of RIS with full parameter header).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aoa_estimators import (
    steering_vector_ula,
    sample_covariance,
    bartlett_spectrum,
    music_spectrum,
)

# ============================================================================
# Test parameters (these will appear in the header on the plot)
# ============================================================================
N_ANT       = 7
D_OVER_LAM  = 0.5
ANGLES_TRUE = np.array([-5.0, 10.0])
N_SNAPSHOTS = 100
THETA_GRID  = np.linspace(-90, 90, 3601)
SNR_DB      = 10.0
RNG_SEED    = 42


# ============================================================================
# Synthetic data
# ============================================================================
def generate_snapshots(angles_deg, snr_db, n_ant, n_snapshots,
                       d_over_lam=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    K = len(angles_deg)
    A = np.column_stack([
        steering_vector_ula(t, n_ant, d_over_lam) for t in angles_deg
    ])
    s = (rng.standard_normal((K, n_snapshots)) +
         1j * rng.standard_normal((K, n_snapshots))) / np.sqrt(2)
    sigma_n2 = 10 ** (-snr_db / 10)
    n = (rng.standard_normal((n_ant, n_snapshots)) +
         1j * rng.standard_normal((n_ant, n_snapshots))) * np.sqrt(sigma_n2 / 2)
    return A @ s + n


def music_reference(X, theta_grid, n_sources, d_over_lam=0.5):
    """Independent MUSIC implementation: SVD of X (vs eigh of R)."""
    N = X.shape[0]
    U, _, _ = np.linalg.svd(X, full_matrices=True)
    Un = U[:, n_sources:]
    A_grid = np.column_stack([
        steering_vector_ula(t, N, d_over_lam) for t in theta_grid
    ])
    Z = Un.conj().T @ A_grid
    return 1.0 / (np.sum(np.abs(Z) ** 2, axis=0) + 1e-12)


# ============================================================================
# Run experiment
# ============================================================================
rng = np.random.default_rng(RNG_SEED)
X = generate_snapshots(ANGLES_TRUE, SNR_DB, N_ANT, N_SNAPSHOTS,
                       D_OVER_LAM, rng=rng)
R = sample_covariance(X)

P_bart    = bartlett_spectrum(R, THETA_GRID, D_OVER_LAM)
P_music   = music_spectrum(R, THETA_GRID, n_sources=2,
                           d_over_lambda=D_OVER_LAM)
P_music_r = music_reference(X, THETA_GRID, n_sources=2,
                            d_over_lam=D_OVER_LAM)

# Normalise to 0 dB peak
P_bart_dB    = 10 * np.log10(P_bart    / P_bart.max())
P_music_dB   = 10 * np.log10(P_music   / P_music.max())
P_music_r_dB = 10 * np.log10(P_music_r / P_music_r.max())

# ============================================================================
# Plot 1: Main plot — Bartlett vs MUSIC, paper-style header
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6.5))

# Plot the spectra
ax.plot(THETA_GRID, P_bart_dB,  'C0-',  lw=1.8, label='Bartlett (Conventional)')
ax.plot(THETA_GRID, P_music_dB, 'C1--', lw=1.8, label='MUSIC (Schmidt 1986)')

# Mark true source angles
for theta in ANGLES_TRUE:
    ax.axvline(theta, color='k', linestyle=':', alpha=0.5, lw=0.8)

ax.set_xlim(-30, 30)
ax.set_ylim(-50, 5)
ax.set_xlabel('Observation Angle, θ (deg)', fontsize=12)
ax.set_ylabel('Normalised Spatial Spectrum (dB)', fontsize=12)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11, loc='lower right', framealpha=0.95)

# === Header with all parameters (paper style) ===
header = (
    f"Spatial Spectrum Comparison: Bartlett vs MUSIC\n"
    f"ULA with N = {N_ANT} elements, d = λ/2 spacing\n"
    f"Two uncorrelated sources at θ₁ = {ANGLES_TRUE[0]:+.0f}°, "
    f"θ₂ = {ANGLES_TRUE[1]:+.0f}° (SNR = {SNR_DB:.0f} dB per source)\n"
    f"L = {N_SNAPSHOTS} snapshots, θ-grid Δ = "
    f"{(THETA_GRID[1]-THETA_GRID[0])*1000:.1f} mdeg, seed = {RNG_SEED}"
)
ax.set_title(header, fontsize=11, loc='left', pad=12,
             fontfamily='serif', fontweight='normal')

fig.tight_layout()
out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "schmidt_validation_main.png")
fig.savefig(out1, dpi=140, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig)


# ============================================================================
# Plot 2: Overlay test — our MUSIC vs independent reference
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6.5))

ax.plot(THETA_GRID, P_music_dB,   'C1-',  lw=3.0,
        label='MUSIC (this work, eigh of R)', alpha=0.8)
ax.plot(THETA_GRID, P_music_r_dB, 'k--', lw=1.2,
        label='MUSIC (independent ref, SVD of X)')

for theta in ANGLES_TRUE:
    ax.axvline(theta, color='r', linestyle=':', alpha=0.5, lw=0.8)

ax.set_xlim(-30, 30)
ax.set_ylim(-50, 5)
ax.set_xlabel('Observation Angle, θ (deg)', fontsize=12)
ax.set_ylabel('Normalised Spatial Spectrum (dB)', fontsize=12)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=10, loc='lower center', framealpha=0.95)

# Error metrics
diff_dB = np.abs(P_music_dB - P_music_r_dB)
ax.text(0.02, 0.05,
        f"max |Δ| = {diff_dB.max():.4f} dB\n"
        f"median |Δ| = {np.median(diff_dB):.4f} dB\n"
        f"RMS |Δ| = {np.sqrt(np.mean(diff_dB**2)):.4f} dB",
        transform=ax.transAxes, fontsize=10,
        va='bottom', ha='left', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

header = (
    f"Verification by Independent Implementation (Overlay Test)\n"
    f"ULA with N = {N_ANT} elements, d = λ/2 spacing\n"
    f"Two sources at θ₁ = {ANGLES_TRUE[0]:+.0f}°, θ₂ = {ANGLES_TRUE[1]:+.0f}°, "
    f"SNR = {SNR_DB:.0f} dB, L = {N_SNAPSHOTS} snapshots\n"
    f"Two MUSIC implementations: Eigendecomposition vs SVD"
)
ax.set_title(header, fontsize=11, loc='left', pad=12,
             fontfamily='serif', fontweight='normal')

fig.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "schmidt_validation_overlay.png")
fig.savefig(out2, dpi=140, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig)


# ============================================================================
# Plot 3: RMSE vs SNR (Monte Carlo) -- bonus plot
# ============================================================================
print("\nRunning Monte Carlo RMSE vs SNR (200 trials/point)...")

SNR_LIST = np.arange(-10, 21, 5)
N_TRIALS = 200
TRUE_ANGLE_MC = 25.0
N_ANT_MC = 4
N_SNAP_MC = 50

rmse_music = []
for snr_db in SNR_LIST:
    errors = []
    for trial in range(N_TRIALS):
        rng_t = np.random.default_rng(abs(trial + int(snr_db * 100)) + 1)
        X_t = generate_snapshots([TRUE_ANGLE_MC], snr_db,
                                  N_ANT_MC, N_SNAP_MC,
                                  D_OVER_LAM, rng=rng_t)
        R_t = sample_covariance(X_t)
        P_t = music_spectrum(R_t, THETA_GRID, n_sources=1,
                             d_over_lambda=D_OVER_LAM)
        est_angle = THETA_GRID[np.argmax(P_t)]
        errors.append(est_angle - TRUE_ANGLE_MC)
    rmse_music.append(np.sqrt(np.mean(np.array(errors)**2)))

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.semilogy(SNR_LIST, rmse_music, 'C1o-', lw=2, markersize=8, label='MUSIC')

# CRLB approximation (proportional to 1/sqrt(SNR))
crlb = rmse_music[-1] * np.sqrt(10 ** (SNR_LIST[-1]/10)) / np.sqrt(10**(SNR_LIST/10))
ax.semilogy(SNR_LIST, crlb, 'k:', lw=1.5, label='CRLB asymptote (∝ 1/√SNR)')

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('RMSE of AoA estimate (deg)', fontsize=12)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(SNR_LIST[0]-1, SNR_LIST[-1]+1)

header = (
    f"Statistical Performance of MUSIC: RMSE vs SNR\n"
    f"ULA with N = {N_ANT_MC} elements, d = λ/2, single source at θ = {TRUE_ANGLE_MC:+.0f}°\n"
    f"L = {N_SNAP_MC} snapshots, {N_TRIALS} Monte Carlo trials per SNR point\n"
    f"Grid resolution Δθ = {(THETA_GRID[1]-THETA_GRID[0])*1000:.1f} mdeg"
)
ax.set_title(header, fontsize=11, loc='left', pad=12,
             fontfamily='serif', fontweight='normal')

fig.tight_layout()
out3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "schmidt_validation_rmse.png")
fig.savefig(out3, dpi=140, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig)

# ============================================================================
# Summary table
# ============================================================================
print("\n" + "="*70)
print("Test Results Summary")
print("="*70)
print(f"Test setup: ULA N={N_ANT}, d=λ/2, 2 sources at "
      f"{ANGLES_TRUE[0]:+.0f}° and {ANGLES_TRUE[1]:+.0f}°")
print(f"            L={N_SNAPSHOTS} snapshots, SNR={SNR_DB} dB/source")
print()
print(f"  max |Δ| (ours vs reference): {diff_dB.max():.4f} dB")
print(f"  median |Δ|:                  {np.median(diff_dB):.4f} dB")
print(f"  RMS |Δ|:                     {np.sqrt(np.mean(diff_dB**2)):.4f} dB")
print()
print(f"  RMSE at SNR=20 dB:   {rmse_music[-1]:.4f} deg")
print(f"  RMSE at SNR= 0 dB:   {rmse_music[len(SNR_LIST)//2]:.4f} deg")
print(f"  RMSE at SNR=-10 dB:  {rmse_music[0]:.4f} deg")