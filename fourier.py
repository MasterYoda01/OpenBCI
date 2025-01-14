import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------
# 1) Read raw OpenBCI data & isolate columns for EXG 1–4
# --------------------------------------------------------------------
name = "SRP+Bias"
df = pd.read_csv(f'{name}.csv', sep='\t', header=None)

# Columns for EXG1–4 (assuming your data layout)
exg1_raw = df.iloc[:, 2].values  # EXG Ch.1
exg2_raw = df.iloc[:, 3].values  # EXG Ch.2
exg3_raw = df.iloc[:, 4].values  # EXG Ch.3
exg4_raw = df.iloc[:, 5].values  # EXG Ch.4

# --------------------------------------------------------------------
# 2) Convert from raw counts to microvolts
#    (Default Cyton: gain=24, Vref=4.5V, ~0.02235 µV/count)
# --------------------------------------------------------------------
SCALE_UV = 0.02235
exg1_uV = exg1_raw * SCALE_UV
exg2_uV = exg2_raw * SCALE_UV
exg3_uV = exg3_raw * SCALE_UV
exg4_uV = exg4_raw * SCALE_UV

# --------------------------------------------------------------------
# 3) Average the four channels
# --------------------------------------------------------------------
exg_avg_uV = (exg1_uV + exg2_uV + exg3_uV + exg4_uV) / 4.0

# --------------------------------------------------------------------
# 4) Compute FFT of the averaged signal
# --------------------------------------------------------------------
Fs = 250.0  # Default Cyton sampling rate
N  = len(exg_avg_uV)
t = np.arange(N) / Fs
X  = np.fft.rfft(exg_avg_uV)          # one-sided FFT
freqs = np.fft.rfftfreq(N, d=1/Fs)    # frequency axis
amp   = np.abs(X)                     # amplitude spectrum

# Optionally scale amplitude to match typical "true" amplitude
# e.g. if you prefer an amplitude scale ~ (2.0 / N) * abs(X).
# For demonstration, we'll leave amp = np.abs(X).

# --------------------------------------------------------------------
# 5) Single "zoomed" plot: amplitude from 500–2000 µV
# --------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(freqs, amp)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (µV)")
plt.title(f"FFT of Average of {name} EXG (Channels 1–4)")
plt.xlim([0, Fs/2])         # 0 to Nyquist
plt.ylim([0, 2500])       # Zoom y-axis from 500 to 2200 µV
plt.tight_layout()
plt.show()

# # --------------------------------------------------------------------
# # 6) (Optional) Plot in the time domain for a sanity check
# # --------------------------------------------------------------------
# plt.figure(figsize=(8, 4))
# plt.plot(t, exg_avg_uV, label="Average of 4 EXG Channels")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude (µV)")
# plt.title("Time-Domain Signal (Averaged)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # --------------------------------------------------------------------
# # 7) Generate and save plots in 0.5 Hz steps
# # --------------------------------------------------------------------
# # Create a folder for the output images
# output_folder = "my_plots_SRB+Ground"
# os.makedirs(output_folder, exist_ok=True)

# # Decide how high in frequency to go; up to Nyquist is Fs/2
# max_freq = Fs / 2

# # We'll make a series of mini-plots for intervals [f, f+0.5].
# # Adjust as needed (e.g. if you want 1 Hz intervals, or different ranges).
# freq_step = 0.5

# # Convert amplitude array to a "dB" scale or keep as is, your choice.
# # For demonstration, we'll keep linear amplitude.

# for start_freq in np.arange(0, max_freq, freq_step):
#     end_freq = start_freq + freq_step

#     # Create a mask for frequencies in [start_freq, end_freq]
#     mask = (freqs >= start_freq) & (freqs <= end_freq)

#     # If you want to skip empty segments (where mask might be no points),
#     # you can do something like:
#     if not np.any(mask):
#         continue

#     # Plot
#     plt.figure(figsize=(6, 4))
#     plt.plot(freqs[mask], amp[mask], marker='o')
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Amplitude (µV)")
#     plt.title(f"FFT from {start_freq:.1f} to {end_freq:.1f} Hz")
#     plt.xlim([start_freq, end_freq])
#     # Also zoom the y-axis for amplitude
#     plt.ylim([500, 2000])  # as requested
#     plt.tight_layout()
    
#     # Save figure
#     plot_filename = f"fft_{start_freq:.1f}-{end_freq:.1f}Hz.png"
#     plt.savefig(os.path.join(output_folder, plot_filename))
#     plt.close()

# print(f"Plots saved in folder: {output_folder}")
