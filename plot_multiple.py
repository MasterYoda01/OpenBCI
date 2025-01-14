import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------
# 1) Read raw OpenBCI data & isolate columns for EXG 1–4
# --------------------------------------------------------------------
df1 = pd.read_csv("data_SRP+Ground/SRP+Ground.csv", sep='\t', header=None)
df2 = pd.read_csv("data_SRP+Bias/SRP+Bias.csv", sep='\t', header=None)
df3 = pd.read_csv("data_SRP/SRP.csv", sep='\t', header=None)

# --------------------------------------------------------------------
# 2) Extract columns for EXG1–4 and average them
# --------------------------------------------------------------------
exg1_raw_1 = df1.iloc[:, 2].values
exg2_raw_1 = df1.iloc[:, 3].values
exg3_raw_1 = df1.iloc[:, 4].values
exg4_raw_1 = df1.iloc[:, 5].values
exg_avg_uV_1 = (exg1_raw_1 + exg2_raw_1 + exg3_raw_1 + exg4_raw_1) / 4.0

exg1_raw_2 = df2.iloc[:, 2].values
exg2_raw_2 = df2.iloc[:, 3].values
exg3_raw_2 = df2.iloc[:, 4].values
exg4_raw_2 = df2.iloc[:, 5].values
exg_avg_uV_2 = (exg1_raw_2 + exg2_raw_2 + exg3_raw_2 + exg4_raw_2) / 4.0

exg1_raw_3 = df3.iloc[:, 2].values
exg2_raw_3 = df3.iloc[:, 3].values
exg3_raw_3 = df3.iloc[:, 4].values
exg4_raw_3 = df3.iloc[:, 5].values
exg_avg_uV_3 = (exg1_raw_3 + exg2_raw_3 + exg3_raw_3 + exg4_raw_3) / 4.0

# --------------------------------------------------------------------
# 3) Convert from raw counts to microvolts (µV)
# --------------------------------------------------------------------
SCALE_UV = 0.02235  # Default Cyton scale factor
exg_avg_uV_1 *= SCALE_UV
exg_avg_uV_2 *= SCALE_UV
exg_avg_uV_3 *= SCALE_UV

# --------------------------------------------------------------------
# 4) Compute single-sided periodogram PSD for each dataset
# --------------------------------------------------------------------
Fs = 250.0  # sampling rate (Hz)
N1 = len(exg_avg_uV_1)
N2 = len(exg_avg_uV_2)
N3 = len(exg_avg_uV_3)

# -- FFT for dataset 1
X1 = np.fft.rfft(exg_avg_uV_1)
freqs1 = np.fft.rfftfreq(N1, d=1/Fs)
psd1 = (2.0 / (N1 * Fs)) * (np.abs(X1) ** 2)
psd1_dB = 10.0 * np.log10(psd1 + 1e-20)

# -- FFT for dataset 2
X2 = np.fft.rfft(exg_avg_uV_2)
freqs2 = np.fft.rfftfreq(N2, d=1/Fs)
psd2 = (2.0 / (N2 * Fs)) * (np.abs(X2) ** 2)
psd2_dB = 10.0 * np.log10(psd2 + 1e-20)

# -- FFT for dataset 3
X3 = np.fft.rfft(exg_avg_uV_3)
freqs3 = np.fft.rfftfreq(N3, d=1/Fs)
psd3 = (2.0 / (N3 * Fs)) * (np.abs(X3) ** 2)
psd3_dB = 10.0 * np.log10(psd3 + 1e-20)

# --------------------------------------------------------------------
# 5) Remove low-frequency bins (DC, etc.)
# --------------------------------------------------------------------
freqs1, psd1_dB = freqs1[40:], psd1_dB[40:]
freqs2, psd2_dB = freqs2[40:], psd2_dB[40:]
freqs3, psd3_dB = freqs3[40:], psd3_dB[40:]

# --------------------------------------------------------------------
# 6) Process data in 30 Hz chunks
# --------------------------------------------------------------------
# Create a folder for saving plots
output_folder = "chunks_30Hz_plots"
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(10, 6))

# Plot dataset 1
plt.plot(freqs1, psd1_dB, label="SRP+Ground", alpha=0.7)

# Plot dataset 2
plt.plot(freqs2, psd2_dB, label="SRP+Bias", alpha=0.7)

# Plot dataset 3
plt.plot(freqs3, psd3_dB, label="SRP", alpha=0.7)

# Add labels, legend, and title
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB, µV²/Hz)")
plt.title("Full Frequency Spectrum for Three Datasets")
plt.legend()

# Automatically scale axes to fit all data
plt.tight_layout()

# Show the plot
plt.show()

# Define the chunk size (30 Hz)
chunk_size = 30
max_freq = min(freqs1.max(), freqs2.max())

# Loop through the frequency range in chunks
for start_freq in range(0, int(max_freq), chunk_size):
    end_freq = start_freq + chunk_size

    # Mask for the current frequency range
    mask1 = (freqs1 >= start_freq) & (freqs1 < end_freq)
    mask2 = (freqs2 >= start_freq) & (freqs2 < end_freq)
    mask3 = (freqs3 >= start_freq) & (freqs3 < end_freq)

    # Create a plot for this chunk
    plt.figure(figsize=(8, 5))

    if np.any(mask1):  # Plot dataset 1 if it has data in this range
        plt.plot(freqs1[mask1], psd1_dB[mask1], label="SRP+Ground")

    if np.any(mask2):  # Plot dataset 2 if it has data in this range
        plt.plot(freqs2[mask2], psd2_dB[mask2], label="SRP+Bias")

    if np.any(mask3):
        plt.plot(freqs3[mask3], psd3_dB[mask3], label="SRP")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB, µV²/Hz)")
    plt.title(f"PSD from {start_freq} Hz to {end_freq} Hz")
    plt.legend()

    # Save the plot to the output folder
    plot_filename = f"{output_folder}/PSD_{start_freq}Hz_{end_freq}Hz.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

print(f"Plots saved in folder: {output_folder}")
