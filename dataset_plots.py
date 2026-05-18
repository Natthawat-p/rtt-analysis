"""
Plotting helpers. Each function returns a matplotlib Figure.

Time-index convention matches main.m:
    x_time_index = array_index - N/2 + 1     (so array index N/2-1 = 0)
Peaks in srs_chT appear at small positive "time index" (e.g. +3 at d=10 m).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dataset_loader   import SYS_PARAMS, SRS_SUBCARRIER_IDX, NOISE_SUBCARRIER_IDX
from dataset_analysis import (
    estimate_snr_db, range_peak_detector, range_matched_filter,
    REF_INDEX, C, FS,
)


N_FFT = SYS_PARAMS["fft_size"]


# --- Direct ports of main.m's figures ----------------------------------------

def plot_impulse_response(srs_chT_col: np.ndarray,
                          title: str = "Impulse response (srs_chT frame 0)"):
    """
    Port of main.m:
        stem(-N/2+1:N/2, abs(srs_chT(:,1))/2^15)
    """
    N = srs_chT_col.size
    x = np.arange(-N // 2 + 1, N // 2 + 1)
    y = np.abs(srs_chT_col) / (2 ** 15)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stem(x, y, basefmt=" ")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Channel gain")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_channel_freq_response(srs_chF_col: np.ndarray,
                               title: str = "Frequency response (srs_chF frame 0)"):
    """
    Port of main.m:
        stem(-N/2+1:N/2, abs(srs_chF(:,1))/2^15)
    Expects an already-fftshifted column (load_measurement does the shift).
    """
    N = srs_chF_col.size
    x = np.arange(-N // 2 + 1, N // 2 + 1)
    y = np.abs(srs_chF_col) / (2 ** 15)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stem(x, y, basefmt=" ")
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("Channel gain")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_impulse_around_peak(srs_chT_col: np.ndarray,
                             window: int = 40,
                             title: str = "Impulse response (zoomed around peak)"):
    """
    Same data as main.m's stem, zoomed around the peak. The x-axis here is
    expressed as samples relative to the time-zero reference (N/2), so peak
    location directly translates to range:  d = x * c / (2 fs)
    """
    N = srs_chT_col.size
    p = int(np.argmax(np.abs(srs_chT_col)))
    x_all = np.arange(N) - REF_INDEX               # samples relative to ref
    y_all = np.abs(srs_chT_col) / (2 ** 15)
    mask = np.abs(x_all - (p - REF_INDEX)) <= window

    p_rel = p - REF_INDEX
    d_est = p_rel * C / (2.0 * FS)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stem(x_all[mask], y_all[mask], basefmt=" ")
    ax.axvline(p_rel, color="r", linestyle=":", alpha=0.7,
               label=f"peak @ sample {p_rel}  ->  d = {d_est:.2f} m")
    ax.axvline(0, color="k", linestyle="--", alpha=0.4, label="time-zero ref")
    ax.set_xlabel("Samples relative to time-zero reference")
    ax.set_ylabel("Channel gain")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_srs_vs_noise_subcarriers(srsrxdataF: np.ndarray,
                                  frame: int = 0,
                                  title: str = "SRS vs noise subcarriers"):
    """Overlay |srsrxdataF| at SRS pilots vs in-comb noise gaps for one frame."""
    col = srsrxdataF if srsrxdataF.ndim == 1 else srsrxdataF[:, frame]

    sig_mag = np.zeros_like(col, dtype=float)
    noi_mag = np.zeros_like(col, dtype=float)
    sig_mag[SRS_SUBCARRIER_IDX]   = np.abs(col[SRS_SUBCARRIER_IDX])
    noi_mag[NOISE_SUBCARRIER_IDX] = np.abs(col[NOISE_SUBCARRIER_IDX])

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(-len(col) // 2 + 1, len(col) // 2 + 1)
    ax.plot(x, sig_mag, label="SRS pilot subcarriers", lw=0.7)
    ax.plot(x, noi_mag, label="Noise-only subcarriers", lw=0.7, alpha=0.7)
    ax.set_xlabel("Subcarrier index (centred on DC)")
    ax.set_ylabel("|srsrxdataF|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_snr_per_frame(srsrxdataF: np.ndarray,
                       title: str = "Per-frame signal vs noise power"):
    """Plot the per-frame signal and noise powers used by estimate_snr_db."""
    snr = estimate_snr_db(srsrxdataF)
    sp  = snr["signal_power_per_frame"]
    npw = snr["noise_power_per_frame"]
    n_frames = len(sp)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(np.arange(n_frames), sp,  label="signal subcarriers")
    ax.semilogy(np.arange(n_frames), npw, label="noise subcarriers")
    ax.set_xlabel("Frame index (after dropping first 499)")
    ax.set_ylabel("Mean |X|^2 per RE (Q15 units)")
    ax.set_title(f"{title}  --  est. SNR = {snr['snr_db']:.1f} dB")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_iq_constellation(X: np.ndarray, max_points: int = 5000,
                          title: str = "IQ samples"):
    """Quick IQ scatter."""
    z = X.ravel()
    if z.size > max_points:
        rng = np.random.default_rng(0)
        z = z[rng.choice(z.size, max_points, replace=False)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(z.real, z.imag, s=2, alpha=0.3)
    lim = max(np.abs(z.real).max(), np.abs(z.imag).max()) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --- Cross-folder plots ------------------------------------------------------

def plot_snr_vs_gain(summaries: list[dict],
                     title: str = "Estimated SNR vs USRP TX gain"):
    """Scatter SNR_dB vs TX gain across the dataset."""
    rows = [s for s in summaries
            if s.get("tx_gain_db") is not None and s.get("snr_db_est") is not None]
    rows.sort(key=lambda r: r["tx_gain_db"])
    gains = [r["tx_gain_db"] for r in rows]
    snrs  = [r["snr_db_est"] for r in rows]
    dists = [r.get("distance_m") for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(gains, snrs, c=dists, cmap="viridis", s=40)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Distance (m)")
    ax.set_xlabel("USRP TX gain (dB)")
    ax.set_ylabel("Estimated SNR (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_range_error_cdf(errors_per_method: dict[str, np.ndarray],
                         title: str = "CDF of range estimation error"):
    """CDF plot like Fig. 6/7 of the paper. Pass {label: array_of_errors_m}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, errs in errors_per_method.items():
        e_sorted = np.sort(np.abs(np.asarray(errs)))
        cdf = np.arange(1, len(e_sorted) + 1) / max(len(e_sorted), 1)
        ax.plot(e_sorted, cdf, label=label, lw=1.5)
    ax.set_xlabel("Range estimation error (m)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_combining_gain_M(srs_chT: np.ndarray,
                          srs_chF_lin_interp: np.ndarray | None = None,
                          true_distance_m: float | None = None,
                          M_list: tuple[int, ...] = (1, 5, 10, 20, 40, 60),
                          fs: float = FS,
                          title: str = "Range estimate vs M (combined measurements)"):
    """Plot PD (and optionally MF) range estimates as a function of M."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pd_vals, M_actual = [], []
    for M in M_list:
        if M > srs_chT.shape[1]:
            break
        pd_vals.append(range_peak_detector(srs_chT, M=M, fs=fs))
        M_actual.append(M)
    ax.plot(M_actual, pd_vals, "o-", label="Peak Detector (Eq. 12)")

    if srs_chF_lin_interp is not None:
        H = np.fft.fftshift(srs_chF_lin_interp, axes=0)
        mf_vals, M_actual_mf = [], []
        for M in M_list:
            if M > H.shape[1]:
                break
            mf_vals.append(range_matched_filter(H, M=M, fs=fs)["range_m"])
            M_actual_mf.append(M)
        ax.plot(M_actual_mf, mf_vals, "s-", label="Matched Filter (Eq. 13)")

    if true_distance_m is not None:
        ax.axhline(true_distance_m, color="k", linestyle=":",
                   label=f"True distance = {true_distance_m} m")
    ax.set_xscale("log")
    ax.set_xlabel("Number of combined measurements M")
    ax.set_ylabel("Range estimate (m)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig