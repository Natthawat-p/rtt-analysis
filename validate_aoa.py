"""
Validation script: Bartlett vs MUSIC on synthetic SRS-like channels.

Reproduces the qualitative behaviour reported in classical AoA literature
(Schmidt 1986, Van Trees Ch. 9):
  - Bartlett gives a broad lobe -> low resolution
  - MUSIC gives sharp peaks at true angles -> high resolution
  - Both estimators are unbiased at high SNR for a single source
  - MUSIC's advantage grows with N (antennas) and SNR

Output: /home/claude/aoa_validation.png  (4-panel figure)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aoa_estimators import (
    sample_covariance,
    bartlett_spectrum,
    music_spectrum,
    find_aoa_peaks,
)
from synthetic_srs import generate_srs_channel, generate_multipath_channel


# Common settings
THETA_GRID = np.linspace(-90, 90, 1801)   # 0.1 deg resolution
D_OVER_LAMBDA = 0.5


# -----------------------------------------------------------------------------
# Test 1: Single source, vary N (number of RX antennas)
# -----------------------------------------------------------------------------
def test_vary_N(ax):
    theta_true = 20.0
    snr_db = 20.0
    Ns = [2, 4, 8]
    colors = ["C0", "C1", "C2"]

    for N, c in zip(Ns, colors):
        H, _ = generate_srs_channel(
            theta_deg=theta_true, n_rx=N, n_sc=600,
            d_over_lambda=D_OVER_LAMBDA, snr_db=snr_db,
            rng=np.random.default_rng(42),
        )
        R = sample_covariance(H)
        P_b = bartlett_spectrum(R, THETA_GRID, D_OVER_LAMBDA)
        P_b_db = 10 * np.log10(P_b / P_b.max())
        ax.plot(THETA_GRID, P_b_db, color=c, linestyle="-",
                label=f"Bartlett, N={N}")

        if N > 1:
            P_m = music_spectrum(R, THETA_GRID, n_sources=1,
                                 d_over_lambda=D_OVER_LAMBDA)
            P_m_db = 10 * np.log10(P_m / P_m.max())
            ax.plot(THETA_GRID, P_m_db, color=c, linestyle="--",
                    label=f"MUSIC, N={N}")

    ax.axvline(theta_true, color="k", linestyle=":", alpha=0.6,
               label=f"True AoA = {theta_true}°")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Normalised spectrum (dB)")
    ax.set_title(f"Test 1: Single source @ {theta_true}°, SNR={snr_db} dB\n"
                 "Effect of number of antennas")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower center", ncol=3)


# -----------------------------------------------------------------------------
# Test 2: Single source, vary SNR (with fixed N=4)
# -----------------------------------------------------------------------------
def test_vary_SNR(ax):
    theta_true = -15.0
    N = 4
    snrs = [-5, 5, 20]
    colors = ["C3", "C4", "C5"]

    for snr, c in zip(snrs, colors):
        H, _ = generate_srs_channel(
            theta_deg=theta_true, n_rx=N, n_sc=600,
            d_over_lambda=D_OVER_LAMBDA, snr_db=snr,
            rng=np.random.default_rng(7),
        )
        R = sample_covariance(H)
        P_m = music_spectrum(R, THETA_GRID, n_sources=1,
                             d_over_lambda=D_OVER_LAMBDA)
        P_m_db = 10 * np.log10(P_m / P_m.max())
        ax.plot(THETA_GRID, P_m_db, color=c,
                label=f"MUSIC, SNR={snr} dB")

    ax.axvline(theta_true, color="k", linestyle=":", alpha=0.6,
               label=f"True AoA = {theta_true}°")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Normalised spectrum (dB)")
    ax.set_title(f"Test 2: Single source @ {theta_true}°, N={N}\n"
                 "Effect of SNR on MUSIC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")


# -----------------------------------------------------------------------------
# Test 3: Two closely-spaced sources -- the classical MUSIC vs Bartlett demo
# -----------------------------------------------------------------------------
def test_two_sources(ax):
    angles_true = [10.0, 18.0]   # 8 deg separation -- challenging for N=4
    N = 4
    snr = 20.0

    H, _ = generate_multipath_channel(
        angles_deg=angles_true, powers_db=[0, 0],
        n_rx=N, n_sc=600, d_over_lambda=D_OVER_LAMBDA, snr_db=snr,
        rng=np.random.default_rng(3),
    )
    R = sample_covariance(H)
    P_b = bartlett_spectrum(R, THETA_GRID, D_OVER_LAMBDA)
    P_m = music_spectrum(R, THETA_GRID, n_sources=2,
                         d_over_lambda=D_OVER_LAMBDA)

    P_b_db = 10 * np.log10(P_b / P_b.max())
    P_m_db = 10 * np.log10(P_m / P_m.max())

    ax.plot(THETA_GRID, P_b_db, "C0-", label="Bartlett")
    ax.plot(THETA_GRID, P_m_db, "C1--", label="MUSIC (n_src=2)")
    for th in angles_true:
        ax.axvline(th, color="k", linestyle=":", alpha=0.5)
    ax.set_xlim(-30, 50)
    ax.set_ylim(-60, 5)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Normalised spectrum (dB)")
    ax.set_title(f"Test 3: Two sources @ {angles_true}°, N={N}, SNR={snr} dB\n"
                 "MUSIC resolves; Bartlett may not")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


# -----------------------------------------------------------------------------
# Test 4: Estimation error vs SNR (Monte Carlo, single source)
# -----------------------------------------------------------------------------
def test_error_vs_snr(ax):
    theta_true = 25.0
    N = 4
    n_trials = 200
    snr_list = np.arange(-10, 31, 5)
    rmse_bartlett = []
    rmse_music = []

    for snr in snr_list:
        errs_b, errs_m = [], []
        for t in range(n_trials):
            H, _ = generate_srs_channel(
                theta_deg=theta_true, n_rx=N, n_sc=600,
                d_over_lambda=D_OVER_LAMBDA, snr_db=snr,
                rng=np.random.default_rng(1000 * t + int(snr) + 1000),
            )
            R = sample_covariance(H)
            P_b = bartlett_spectrum(R, THETA_GRID, D_OVER_LAMBDA)
            P_m = music_spectrum(R, THETA_GRID, n_sources=1,
                                 d_over_lambda=D_OVER_LAMBDA)
            th_b = THETA_GRID[np.argmax(P_b)]
            th_m = THETA_GRID[np.argmax(P_m)]
            errs_b.append((th_b - theta_true) ** 2)
            errs_m.append((th_m - theta_true) ** 2)
        rmse_bartlett.append(np.sqrt(np.mean(errs_b)))
        rmse_music.append(np.sqrt(np.mean(errs_m)))

    ax.semilogy(snr_list, rmse_bartlett, "C0-o", label="Bartlett")
    ax.semilogy(snr_list, rmse_music,    "C1-s", label="MUSIC")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE of AoA estimate (deg)")
    ax.set_title(f"Test 4: RMSE vs SNR, source @ {theta_true}°, N={N}\n"
                 f"{n_trials} Monte-Carlo trials per point")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    test_vary_N(axes[0, 0])
    test_vary_SNR(axes[0, 1])
    test_two_sources(axes[1, 0])
    test_error_vs_snr(axes[1, 1])

    fig.suptitle("AoA Validation: Bartlett vs MUSIC on synthetic SRS-like data",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "aoa_validation.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")

    # Print quick sanity-check numbers
    print("\n--- Sanity check: single-source point estimates ---")
    for theta_true in [-30, 0, 25, 60]:
        H, _ = generate_srs_channel(
            theta_deg=theta_true, n_rx=4, n_sc=600,
            snr_db=20, rng=np.random.default_rng(1),
        )
        R = sample_covariance(H)
        P_m = music_spectrum(R, THETA_GRID, n_sources=1,
                             d_over_lambda=D_OVER_LAMBDA)
        th_hat = find_aoa_peaks(P_m, THETA_GRID, n_peaks=1)[0]
        print(f"  true={theta_true:+6.1f}°   MUSIC estimate={th_hat:+6.2f}°   "
              f"error={th_hat - theta_true:+5.2f}°")


if __name__ == "__main__":
    main()
