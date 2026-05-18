"""
Periodicity diagnostic: find the period of the sawtooth in peak position.

Uses FFT and autocorrelation to identify the dominant period(s).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_loader import load_measurement
from dataset_analysis import REF_INDEX, parabolic_refine

# ---- Edit if needed ----
TARGET = "./ssb_measurements/scenario_ssb_10m_ue_att_0_20240405T133530"

meas = load_measurement(TARGET)
H = meas.srs_chT
N_frames = H.shape[1]

# Refined peak index per frame, relative to N/2
peaks = np.empty(N_frames, dtype=float)
for m in range(N_frames):
    mag = np.abs(H[:, m])
    p_int = int(np.argmax(mag))
    peaks[m] = parabolic_refine(mag, p_int)
peaks -= REF_INDEX

# Subtract mean to focus on the AC component
peaks_ac = peaks - peaks.mean()

# Autocorrelation (normalized)
def autocorr(x):
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]
acf = autocorr(peaks_ac)

# Find first significant peak in autocorrelation (excluding lag 0)
# A simple way: find the first local maximum above some threshold
lags = np.arange(N_frames)
# Look at lags 10..500 (rough search range)
search = acf[10:500]
peak_lag = 10 + int(np.argmax(search))
period_est = peak_lag

# Power spectrum
freqs = np.fft.rfftfreq(N_frames, d=1.0)   # cycles per frame
psd = np.abs(np.fft.rfft(peaks_ac)) ** 2
# Mask out DC and very low frequencies
psd[0] = 0
# Find dominant frequency
dom_bin = int(np.argmax(psd[1:100])) + 1
dom_freq = freqs[dom_bin]
period_fft = 1.0 / dom_freq if dom_freq > 0 else None

# Print summary
print(f"=== Periodicity of peak-position sawtooth ===")
print(f"Folder: {os.path.basename(TARGET)}")
print(f"N frames: {N_frames}")
print(f"Peak (relative to N/2): mean={peaks.mean():.3f}, std={peaks.std():.3f}")
print(f"Range over peaks: {peaks.min():.2f} to {peaks.max():.2f}")
print(f"Peak-to-peak: {peaks.max()-peaks.min():.2f} samples "
      f"= {(peaks.max()-peaks.min()) * 3e8 / (2*46.08e6):.2f} m")
print()
print(f"Autocorrelation: first major peak at lag {peak_lag}")
print(f"  -> period ~ {period_est} frames")
print(f"FFT: dominant freq = {dom_freq:.5f} cycles/frame")
print(f"  -> period ~ {period_fft:.1f} frames" if period_fft else "")
print()
print(f"Possible interpretations of {period_est} frames period:")
print(f"  If 1 frame = 10 ms (5G NR frame): period = {period_est*10/1000:.2f} sec")
print(f"  If 1 frame = 5 ms (half-frame):   period = {period_est*5/1000:.2f} sec")
print(f"  If 1 frame = 1 ms (subframe):     period = {period_est*1/1000:.2f} sec")
print(f"  If 1 frame = 0.5 ms (slot):       period = {period_est*0.5/1000:.2f} sec")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Top-left: zoomed time series (first 500 frames)
ax = axes[0, 0]
n_show = min(500, N_frames)
ax.plot(peaks[:n_show], ".-", markersize=2, alpha=0.7)
ax.set_xlabel("Frame index")
ax.set_ylabel("Peak − N/2 (samples)")
ax.set_title(f"Zoomed sawtooth (first {n_show} frames)")
ax.grid(True, alpha=0.3)
# annotate suspected period
ax.axvspan(0, period_est, alpha=0.2, color="orange",
           label=f"one period ≈ {period_est} frames")
ax.legend()

# Top-right: autocorrelation
ax = axes[0, 1]
n_plot_acf = min(500, N_frames // 2)
ax.plot(lags[:n_plot_acf], acf[:n_plot_acf])
ax.axvline(peak_lag, color="r", linestyle="--",
           label=f"first peak @ lag {peak_lag}")
ax.set_xlabel("Lag (frames)")
ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation of peak-position residual")
ax.grid(True, alpha=0.3)
ax.legend()

# Bottom-left: power spectrum (zoomed)
ax = axes[1, 0]
# Plot vs period (1/freq) on x for intuitive interpretation
periods = np.zeros_like(freqs)
periods[1:] = 1.0 / freqs[1:]
mask = (periods >= 10) & (periods <= 500)
ax.semilogy(periods[mask], psd[mask])
ax.axvline(period_est, color="r", linestyle="--",
           label=f"~{period_est} frames")
ax.set_xlabel("Period (frames)")
ax.set_ylabel("PSD")
ax.set_title("Power spectrum (showing dominant period)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()

# Bottom-right: phase-folded view at the estimated period
ax = axes[1, 1]
phase = np.arange(N_frames) % period_est
# Scatter plot
ax.plot(phase, peaks, ".", markersize=1, alpha=0.3)
# Compute average per phase bin
bins = np.linspace(0, period_est, 40)
mean_per_bin = []
for i in range(len(bins) - 1):
    mask = (phase >= bins[i]) & (phase < bins[i + 1])
    mean_per_bin.append(np.mean(peaks[mask]) if mask.any() else np.nan)
bin_centres = (bins[:-1] + bins[1:]) / 2
ax.plot(bin_centres, mean_per_bin, "r-", lw=2, label="mean per bin")
ax.set_xlabel(f"Phase within {period_est}-frame period")
ax.set_ylabel("Peak − N/2 (samples)")
ax.set_title(f"Phase-folded view (period = {period_est} frames)")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
out = "./analysis_output/_periodicity_analysis.png"
fig.savefig(out, dpi=110)
print(f"\nSaved {out}")
