import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Load the data (assuming tab-delimited file)
file_path = 'moe_data/SRP.csv'  # Replace with your file path
data = pd.read_csv(file_path, delimiter="\t")

# Select relevant numeric columns for analysis (assuming all except the first column are EEG signals)
eeg_data = data.iloc[:, 1:]

# Split the data into two halves: Eyes Closed and Eyes Open
half_point = len(eeg_data) // 2
eyes_closed = eeg_data.iloc[:half_point]
eyes_open = eeg_data.iloc[half_point:]

# Define sampling frequency
sampling_freq = 250  # Replace with your actual sampling frequency if different

# Function to calculate PSD
def calculate_psd(data_series, fs):
    f, psd = welch(data_series, fs=fs, nperseg=fs*2)  # Welch's method
    return f, psd

# Calculate PSD for eyes-closed and eyes-open segments
frequency_closed, psd_closed = calculate_psd(eyes_closed.mean(axis=1), sampling_freq)
frequency_open, psd_open = calculate_psd(eyes_open.mean(axis=1), sampling_freq)

# Filter to focus on the 0-30 Hz range
focus_range = (frequency_closed >= 0) & (frequency_closed <= 30)
frequency_closed = frequency_closed[focus_range]
psd_closed = psd_closed[focus_range]
psd_open = psd_open[focus_range]

# Save the figure to a file
offset = 20
output_path = 'good_plots_moe/SRP_plot.png'  # Replace with your desired folder and file name
plt.figure(figsize=(10, 6))
plt.plot(frequency_closed, 10 * np.log10(psd_closed) + offset, label='Eyes Closed', color='blue')
plt.plot(frequency_closed, 10 * np.log10(psd_open) + offset, label='Eyes Open', color='orange')
plt.title('PSD of Eyes Open & Closed (0-30 Hz) (SRP)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.legend()
plt.grid()
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
plt.show()

print(f"Figure saved to {output_path}")
