import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind, zscore
from scipy.signal import welch, butter, filtfilt
from matplotlib import pyplot as plt

# Function to calculate PSD
def calculate_psd(data_series, fs):
    f, psd = welch(data_series, fs=fs, nperseg=fs * 2)
    psd_db = 10 * np.log10(psd)
    return f, psd_db

# Function to calculate absolute deltas and identify higher values
def calculate_absolute_deltas(data1, data2):
    deltas = np.abs(data1 - data2)
    higher = np.where(data1 > data2, "Joe", "Mohammed")
    return deltas, higher

# Function to apply a filter
def apply_filter(data, fs, cutoff, filter_type):
    b, a = butter(4, cutoff / (fs / 2), btype=filter_type)
    return filtfilt(b, a, data)

# Function to process EEG data with filtering and averaging
def process_eeg_data(file_path, sampling_freq, filter_type=None, cutoff=None):
    data = pd.read_csv(file_path, delimiter="\t")
    eeg_data = data.iloc[:, 2:6]  # Columns EXG1–4

    # Average across channels
    avg_signal = eeg_data.mean(axis=1)

    # Apply filter if specified
    if filter_type and cutoff:
        avg_signal = apply_filter(avg_signal, sampling_freq, cutoff, filter_type)

    # Split into eyes-closed and eyes-open
    half_point = len(avg_signal) // 2
    eyes_closed = avg_signal[:half_point]
    eyes_open = avg_signal[half_point:]

    # Calculate PSD
    freq_closed, psd_closed = calculate_psd(eyes_closed, sampling_freq)
    freq_open, psd_open = calculate_psd(eyes_open, sampling_freq)

    # Focus on 8-12 Hz range
    focus_range = (freq_closed >= 0) & (freq_closed <= 30)
    return freq_closed[focus_range], psd_closed[focus_range], psd_open[focus_range]

# File paths for your data and Mohammed's data
file_yours = 'OpenBCI/joe_data/t4/SRP+Bias.csv'
file_mohammed = 'OpenBCI/moe_data/other/moh4_2.csv'

# Sampling frequency
sampling_freq = 250

# Define filters
filters = [
    {"type": None, "cutoff": None, "label": "No Filter"},
    {"type": "low", "cutoff": 30, "label": "Low-Pass Filter"},
    {"type": "high", "cutoff": 0.5, "label": "High-Pass Filter"}
]

# Initialize results
all_results = []

# Process data for each filter configuration
for f_config in filters:
    label = f_config["label"]
    f_type = f_config["type"]
    cutoff = f_config["cutoff"]

    # Process data
    freq_yours, psd_yours_closed, psd_yours_open = process_eeg_data(file_yours, sampling_freq, f_type, cutoff)
    freq_mohammed, psd_mohammed_closed, psd_mohammed_open = process_eeg_data(file_mohammed, sampling_freq, f_type, cutoff)

    # Perform ANOVA
    anova_closed = f_oneway(psd_yours_closed, psd_mohammed_closed)
    anova_open = f_oneway(psd_yours_open, psd_mohammed_open)

    # Point-by-point t-tests
    _, p_values_closed = ttest_ind(psd_yours_closed, psd_mohammed_closed)
    _, p_values_open = ttest_ind(psd_yours_open, psd_mohammed_open)

    # Calculate Z-Scores
    z_scores_yours_closed = zscore(psd_yours_closed)
    z_scores_yours_open = zscore(psd_yours_open)
    z_scores_mohammed_closed = zscore(psd_mohammed_closed)
    z_scores_mohammed_open = zscore(psd_mohammed_open)

    # Calculate Absolute Deltas
    deltas_closed, higher_closed = calculate_absolute_deltas(psd_yours_closed, psd_mohammed_closed)
    deltas_open, higher_open = calculate_absolute_deltas(psd_yours_open, psd_mohammed_open)

    # Create DataFrame for the current filter
    table_data = pd.DataFrame({
        "Filter": label,
        "Frequency (Hz)": freq_yours,
        "Joe Z-Score Closed": z_scores_yours_closed,
        "Joe Z-Score Open": z_scores_yours_open,
        "Mohammed Z-Score Closed": z_scores_mohammed_closed,
        "Mohammed Z-Score Open": z_scores_mohammed_open,
        "Absolute Delta Closed": deltas_closed,
        "Absolute Delta Open": deltas_open,
        "P-Value Closed": p_values_closed,
        "P-Value Open": p_values_open,
        "Higher Closed": higher_closed,
        "Higher Open": higher_open
    })

    # Append results
    all_results.append(table_data)

# Combine results for all filters
final_results = pd.concat(all_results, ignore_index=True)

# Save to CSV
output_csv_path = "OpenBCI/trial_results/results/trial-4_2_analysis_all_filters.csv"
final_results.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")


# === Visualization ===

# # Plot PSD Comparison for Both Participants (8-12 Hz)
# output_plot_path = 'results/SRP_PSD_8-12Hz.png'
# plt.figure(figsize=(12, 6))
# offset = 20
# plt.plot(freq_yours, psd_yours_closed + offset , label="Joe Eyes Closed", color='blue')
# plt.plot(freq_yours, psd_yours_open + offset, label="Joe Eyes Open", color='orange')
# plt.plot(freq_mohammed, psd_mohammed_closed + offset, label="Mohammed Eyes Closed", linestyle='--', color='green')
# plt.plot(freq_mohammed, psd_mohammed_open + offset, label="Mohammed Eyes Open", linestyle='--', color='red')
# plt.title("PSD Comparison (Joe's Data vs Mohammed's Data) (SRP + Bias)")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power Spectral Density (dB)")
# plt.legend()
# plt.grid()
# plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')  # Save the figure
# plt.show()

# print(f"Plot saved to {output_plot_path}")
