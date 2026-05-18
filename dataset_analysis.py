"""
Analysis routines for the OAI RTT dataset.

IMPORTANT: After verifying with the actual dataset, we know that `srs_chT`
uses array index N/2 (= 768) as the time-zero reference. The channel peak
at d = 10 m appears around array index 771 (= N/2 + 3), giving a delay of
~3 samples ~= 9.77 m physical range. The paper's reported 13.02 m suggests
an additional ~3.25 m processing/cable bias on top of that.

Range formula (corrected, Eq. 11):

        d = (peak_index - N/2) * c / (2 * fs)

This matches main.m's plotting convention where x = -N/2+1 : N/2 maps
array index k to "time index" (k - N/2 + 1). In Fig. 5 of the paper the
peak appears at "time index" around +5.
"""

from __future__ import annotations

import numpy as np

from dataset_loader import (
    SYS_PARAMS, SRS_SUBCARRIER_IDX, NOISE_SUBCARRIER_IDX,
)

C     = SYS_PARAMS["speed_of_light"]
FS    = SYS_PARAMS["sampling_rate_hz"]
N_FFT = SYS_PARAMS["fft_size"]
REF_INDEX = N_FFT // 2                # = 768; time-zero reference in srs_chT


# --- Power -------------------------------------------------------------------

def avg_power_per_re(X: np.ndarray) -> float:
    """Mean |X|^2 over all entries."""
    return float(np.mean(np.abs(X) ** 2))


# --- SNR estimation (main.m style) -------------------------------------------

def estimate_snr_db(srsrxdataF: np.ndarray) -> dict:
    """
    SNR from srsrxdataF using the SRS-pilot vs in-comb-gap split (Eq. 10):
        SNR_lin = (mean|sig|^2 - mean|noise|^2) / mean|noise|^2
    """
    if srsrxdataF is None:
        raise ValueError("srsrxdataF is None")
    if srsrxdataF.ndim != 2 or srsrxdataF.shape[0] != N_FFT:
        raise ValueError(f"Expected (N={N_FFT}, n_frames); got {srsrxdataF.shape}")

    sig = srsrxdataF[SRS_SUBCARRIER_IDX,   :]
    noi = srsrxdataF[NOISE_SUBCARRIER_IDX, :]

    sig_pow_pf = np.mean(np.abs(sig) ** 2, axis=0)
    noi_pow_pf = np.mean(np.abs(noi) ** 2, axis=0)
    sig_pow = float(np.mean(sig_pow_pf))
    noi_pow = float(np.mean(noi_pow_pf))

    if noi_pow <= 0:
        snr_lin = float("inf")
    else:
        snr_lin = max((sig_pow - noi_pow) / noi_pow, 1e-12)
    snr_db = 10.0 * np.log10(snr_lin) if snr_lin > 0 else float("nan")

    return {"snr_lin": snr_lin, "snr_db": snr_db,
            "signal_power_mean": sig_pow, "noise_power_mean": noi_pow,
            "signal_power_per_frame": sig_pow_pf,
            "noise_power_per_frame":  noi_pow_pf}


# --- Range estimation (CORRECTED) --------------------------------------------

def range_from_peak(peak_index: float,
                    ref_index: float = REF_INDEX,
                    fs: float = FS) -> float:
    """
    Convert a peak array-index to range in metres:
        d = (peak_index - ref_index) * c / (2 * fs)
    With ref_index = N/2 = 768 (the time-zero reference in srs_chT).
    """
    return (float(peak_index) - float(ref_index)) * C / (2.0 * fs)


def parabolic_refine(y: np.ndarray, p: int) -> float:
    """3-point parabolic refinement of an integer peak index."""
    if p <= 0 or p >= len(y) - 1:
        return float(p)
    y0, y1, y2 = y[p - 1], y[p], y[p + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0:
        return float(p)
    return p + 0.5 * (y0 - y2) / denom


def estimate_range_chT_single(srs_chT_col: np.ndarray,
                              fs: float = FS,
                              refine: bool = True) -> dict:
    """Single-frame range estimate from one column of srs_chT."""
    mag = np.abs(srs_chT_col)
    p_int = int(np.argmax(mag))
    p_ref = parabolic_refine(mag, p_int) if refine else float(p_int)
    return {"peak_index":         p_int,
            "peak_index_refined": p_ref,
            "range_m":            range_from_peak(p_ref, fs=fs)}


def impulse_response_from_chF(chF_col: np.ndarray,
                              fft_size: int = N_FFT) -> np.ndarray:
    """IFFT of one freq-domain column."""
    return np.fft.ifft(chF_col, n=fft_size)


# --- Coherent combiners (Eq. 12, 13, corrected) ------------------------------

def range_peak_detector(srs_chT: np.ndarray,
                        M: int | None = None,
                        fs: float = FS,
                        refine: bool = True) -> float:
    """
    Peak-detector combiner (Eq. 12), with N/2 reference correction:
        d_hat = c/(2 fs M) * sum_m (argmax|h_m| - N/2)
    """
    if srs_chT.ndim != 2:
        raise ValueError("srs_chT must be 2-D (N, n_frames)")
    n_total = srs_chT.shape[1]
    if M is None or M > n_total:
        M = n_total
    if M < 1:
        raise ValueError("Need at least one frame")

    H = srs_chT[:, :M]
    peaks = np.empty(M, dtype=float)
    for m in range(M):
        mag = np.abs(H[:, m])
        p_int = int(np.argmax(mag))
        peaks[m] = parabolic_refine(mag, p_int) if refine else p_int

    mean_peak = float(np.mean(peaks))
    return range_from_peak(mean_peak, fs=fs)


def range_matched_filter(srs_chF: np.ndarray,
                         M: int | None = None,
                         fs: float = FS,
                         fft_size: int = N_FFT,
                         tau_window_m: tuple[float, float] = (-5.0, 30.0),
                         tau_step_samples: float = 0.05,
                         use_srs_only: bool = True) -> dict:
    """
    Matched-filter combiner (Eq. 13).

        d_hat = (c / (2 fs)) * argmax_tau (1/M) sum_m |v(tau)^H H_m|^2

    The steering vector uses *centred* subcarrier indices (k - N/2), so tau
    is the delay relative to the array centre (which matches REF_INDEX).

    Parameters
    ----------
    srs_chF : (N, n_frames) complex
        Frequency-domain channel, fftshifted so SRS pilot indices align.
    tau_window_m : (d_min, d_max)
        Search range in metres. Default (-5, 30) covers everything in the
        paper plus a margin for negative-bias artefacts.
    tau_step_samples : float
        Grid step in samples (0.05 ≈ 0.16 m).
    """
    if srs_chF.ndim != 2:
        raise ValueError("srs_chF must be 2-D (N, n_frames)")
    n_total = srs_chF.shape[1]
    if M is None or M > n_total:
        M = n_total

    H = srs_chF[:, :M]
    if use_srs_only:
        idx = SRS_SUBCARRIER_IDX
        H_sub = H[idx, :]
        k = idx.astype(float)
    else:
        H_sub = H
        k = np.arange(fft_size, dtype=float)

    # Build tau grid (samples) from the metre window
    d_min, d_max = tau_window_m
    s_min = d_min * 2.0 * fs / C
    s_max = d_max * 2.0 * fs / C
    tau_grid = np.arange(s_min, s_max, tau_step_samples)

    k_centred = k - (fft_size / 2.0)
    phase = -2j * np.pi * np.outer(tau_grid, k_centred) / fft_size
    v = np.exp(phase)                                  # (G, K)

    corr = v.conj() @ H_sub                            # (G, M)
    cost = np.mean(np.abs(corr) ** 2, axis=1)          # (G,)

    g_hat   = int(np.argmax(cost))
    tau_hat = float(tau_grid[g_hat])
    return {"range_m":         tau_hat * C / (2.0 * fs),
            "tau_hat_samples": tau_hat,
            "cost_curve":      cost,
            "tau_grid":        tau_grid}


# --- Summary -----------------------------------------------------------------

def summarise_measurement(meas) -> dict:
    out = {
        "folder":            meas.meta.folder_name,
        "distance_m":        meas.meta.distance_m,
        "tx_gain_db":        meas.meta.tx_gain_db,
        "tx_attenuation_db": meas.meta.tx_attenuation_db,
        "timestamp":         meas.meta.timestamp,
    }

    for name, arr in [("srs_chT",            meas.srs_chT),
                      ("srs_chF",            meas.srs_chF),
                      ("srsrxdataF",         meas.srsrxdataF),
                      ("srs_chF_lin_interp", meas.srs_chF_lin_interp)]:
        if arr is not None:
            out[f"n_frames_{name}"] = arr.shape[1]
            out[f"P_avg_{name}"]    = avg_power_per_re(arr)

    if meas.srsrxdataF is not None:
        try:
            snr = estimate_snr_db(meas.srsrxdataF)
            out["snr_db_est"]  = snr["snr_db"]
            out["snr_lin_est"] = snr["snr_lin"]
        except Exception as e:
            out["snr_db_est"] = None
            out["snr_err"]    = str(e)

    if meas.srs_chT is not None and meas.srs_chT.shape[1] > 0:
        r = estimate_range_chT_single(meas.srs_chT[:, 0])
        out["range_m_frame0_chT"]  = r["range_m"]
        out["peak_idx_frame0_chT"] = r["peak_index"]

    if meas.srs_chT is not None and meas.srs_chT.shape[1] >= 2:
        out["range_m_PD_all"] = range_peak_detector(meas.srs_chT)

    if meas.srs_chF_lin_interp is not None and meas.srs_chF_lin_interp.shape[1] >= 2:
        H_shifted = np.fft.fftshift(meas.srs_chF_lin_interp, axes=0)
        out["range_m_MF_all"] = range_matched_filter(H_shifted)["range_m"]

    return out