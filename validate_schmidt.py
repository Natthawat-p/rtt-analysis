"""
Validation: Reproduce key results from Schmidt (1986)
======================================================
Schmidt, R. O. "Multiple Emitter Location and Signal Parameter Estimation."
IEEE Trans. Antennas and Propagation, AP-34(3), 276-280, March 1986.

This script:
  1. Generates synthetic snapshots matching Schmidt's narrowband ULA model
  2. Runs OUR Bartlett and MUSIC implementations
  3. Runs an INDEPENDENT MUSIC reference (different code path: SVD of X
     instead of eigh of R) so we can overlay and check the curves match
  4. Plots side-by-side comparison

The overlay panels (right column) are the strict "plot two curves and see
if they overlap" test the user asked for.

Setup: N=7 ULA, d=lambda/2, 2 sources, 100 snapshots
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
# Schmidt 1986 setup
# ============================================================================
N_ANT       = 7
D_OVER_LAM  = 0.5
ANGLES_TRUE = np.array([-5.0, 10.0])
N_SNAPSHOTS = 100
THETA_GRID  = np.linspace(-90, 90, 3601)


# ============================================================================
# Synthetic data generator (Schmidt's narrowband ULA model)
# ============================================================================
def generate_snapshots(angles_deg, snr_db_per_source, n_ant, n_snapshots,
                       d_over_lam=0.5, rng=None):
    """
    Generate narrowband ULA snapshots: x(t) = A(theta) s(t) + n(t)
    where s and n are independent zero-mean complex Gaussian.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    K = len(angles_deg)
    A = np.column_stack([
        steering_vector_ula(theta, n_ant, d_over_lam) for theta in angles_deg
    ])

    s = (rng.standard_normal((K, n_snapshots)) +
         1j * rng.standard_normal((K, n_snapshots))) / np.sqrt(2)

    sigma_n2 = 10 ** (-snr_db_per_source / 10)
    n = (rng.standard_normal((n_ant, n_snapshots)) +
         1j * rng.standard_normal((n_ant, n_snapshots))) * np.sqrt(sigma_n2 / 2)

    return A @ s + n


# ============================================================================
# Independent MUSIC reference (SVD of X) — different from eigh(R) in ours
# ============================================================================
def music_reference(X, theta_grid_deg, n_sources, d_over_lam=0.5):
    """
    Textbook MUSIC via SVD of the data matrix.
    P(theta) = 1 / ||U_n^H a(theta)||^2
    """
    N = X.shape[0]
    U, _, _ = np.linalg.svd(X, full_matrices=True)
    Un = U[:, n_sources:]

    A_grid = np.column_stack([
        steering_vector_ula(t, N, d_over_lam) for t in theta_grid_deg
    ])
    Z = Un.conj().T @ A_grid
    return 1.0 / (np.sum(np.abs(Z) ** 2, axis=0) + 1e-12)


# ============================================================================
# Run one SNR scenario, plot two panels (compare + overlay)
# ============================================================================
def run_one_scenario(snr_db, ax_left, ax_right, title_suffix, rng_seed):
    rng = np.random.default_rng(rng_seed)
    X = generate_snapshots(ANGLES_TRUE, snr_db, N_ANT, N_SNAPSHOTS,
                           D_OVER_LAM, rng=rng)
    R = sample_covariance(X)

    P_b_ours = bartlett_spectrum(R, THETA_GRID, D_OVER_LAM)
    P_m_ours = music_spectrum(R, THETA_GRID, n_sources=2,
                              d_over_lambda=D_OVER_LAM)
    P_m_ref  = music_reference(X, THETA_GRID, n_sources=2,
                               d_over_lam=D_OVER_LAM)

    P_b_ours_dB = 10 * np.log10(P_b_ours / P_b_ours.max())
    P_m_ours_dB = 10 * np.log10(P_m_ours / P_m_ours.max())
    P_m_ref_dB  = 10 * np.log10(P_m_ref  / P_m_ref.max())

    # ----- Left: classical Schmidt Fig.2 style -----
    ax_left.plot(THETA_GRID, P_b_ours_dB, "C0-",  lw=1.6,
                 label="Bartlett (ours)")
    ax_left.plot(THETA_GRID, P_m_ours_dB, "C1--", lw=1.6,
                 label="MUSIC (ours)")
    for theta in ANGLES_TRUE:
        ax_left.axvline(theta, color="k", linestyle=":", alpha=0.6)
    ax_left.set_xlim(-30, 30)
    ax_left.set_ylim(-60, 5)
    ax_left.set_xlabel("Angle (deg)")
    ax_left.set_ylabel("Normalised spectrum (dB)")
    ax_left.set_title(f"Bartlett vs MUSIC  —  {title_suffix}")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=9, loc="lower center")
    ax_left.annotate(
        f"true angles: {ANGLES_TRUE[0]:+.0f}°, {ANGLES_TRUE[1]:+.0f}°",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # ----- Right: overlay our MUSIC vs reference -----
    ax_right.plot(THETA_GRID, P_m_ours_dB, "C1-", lw=3.0,
                  label="MUSIC (our code, eigh of R)", alpha=0.7)
    ax_right.plot(THETA_GRID, P_m_ref_dB,  "k--", lw=1.2,
                  label="MUSIC (independent ref, SVD of X)")
    for theta in ANGLES_TRUE:
        ax_right.axvline(theta, color="r", linestyle=":", alpha=0.5)
    ax_right.set_xlim(-30, 30)
    ax_right.set_ylim(-60, 5)
    ax_right.set_xlabel("Angle (deg)")
    ax_right.set_ylabel("Normalised spectrum (dB)")
    ax_right.set_title(f"OVERLAY: ours vs reference  —  {title_suffix}")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=8, loc="lower center")

    diff_db = np.abs(P_m_ours_dB - P_m_ref_dB)
    max_diff = diff_db.max()
    median_diff = np.median(diff_db)
    ax_right.text(0.02, 0.05,
                  f"max |Δ| = {max_diff:.4f} dB\n"
                  f"median |Δ| = {median_diff:.4f} dB",
                  transform=ax_right.transAxes, fontsize=9,
                  va="bottom", ha="left",
                  bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    # Estimate peak angles from our MUSIC (within ±5° of each true angle)
    peaks = []
    for theta_t in ANGLES_TRUE:
        mask = np.abs(THETA_GRID - theta_t) < 5
        local_max_idx = np.argmax(P_m_ours[mask])
        peaks.append(float(THETA_GRID[mask][local_max_idx]))

    return {"snr_db": snr_db,
            "max_diff_db": max_diff,
            "median_diff_db": median_diff,
            "music_peaks_ours": peaks}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    print("=" * 70)
    print("Schmidt 1986 reproduction — Bartlett vs MUSIC")
    print(f"N={N_ANT} ULA, d={D_OVER_LAM}λ, true angles {ANGLES_TRUE}°, "
          f"{N_SNAPSHOTS} snapshots")
    print("=" * 70)

    summary = []
    summary.append(run_one_scenario(10.0, axes[0, 0], axes[0, 1],
                                    "SNR = 10 dB per source", rng_seed=42))
    summary.append(run_one_scenario(0.0, axes[1, 0], axes[1, 1],
                                    "SNR = 0 dB per source", rng_seed=7))

    fig.suptitle("Validation against Schmidt (1986): MUSIC vs Bartlett, "
                 "with overlay of our MUSIC vs independent reference",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "schmidt_validation.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}\n")

    print(f"{'SNR (dB)':>10}  {'true 1':>8}  {'est 1':>8}  "
          f"{'true 2':>8}  {'est 2':>8}  {'max |Δ| dB':>12}  {'median |Δ| dB':>14}")
    print("-" * 82)
    for s in summary:
        p = s["music_peaks_ours"]
        print(f"{s['snr_db']:>10.1f}  "
              f"{ANGLES_TRUE[0]:>+8.2f}  {p[0]:>+8.2f}  "
              f"{ANGLES_TRUE[1]:>+8.2f}  {p[1]:>+8.2f}  "
              f"{s['max_diff_db']:>12.4f}  {s['median_diff_db']:>14.4f}")

    print()
    print("Interpretation:")
    print("  - Right panels overlay our MUSIC (thick orange) with an")
    print("    independent SVD-based reference (thin black dashed).")
    print("    If the two curves overlap, our implementation is correct.")
    print("  - Left panels reproduce Schmidt's Fig.2: MUSIC resolves the")
    print("    two close sources, Bartlett gives a single broad lobe.")


if __name__ == "__main__":
    main()
