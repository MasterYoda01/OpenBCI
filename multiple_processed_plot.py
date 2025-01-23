import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Define the base folder and subfolders
base_folder = 'OpenBCI/moe_data'
subfolders = ['t1', 't2', 't3', 't4']

# Define sampling frequency
sampling_freq = 250  # Replace with your actual sampling frequency if different

# Function to calculate PSD
def calculate_psd(data_series, fs):
    f, psd = welch(data_series, fs=fs, nperseg=fs*2)  # Welch's method
    return f, psd

# Loop through each subfolder and process the CSV files
for subfolder in subfolders:
    folder_path = os.path.join(base_folder, subfolder)
    output_folder = os.path.join('OpenBCI/trial_results', subfolder)  # Create output folder structure
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    # Iterate through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):  # Process only CSV files
            file_path = os.path.join(folder_path, file_name)
            
            # Load the data
            data = pd.read_csv(file_path, delimiter="\t")

            # Select relevant numeric columns for analysis
            eeg_data = data.iloc[:, 1:5]

            # Split the data into two halves: Eyes Closed and Eyes Open
            half_point = len(eeg_data) // 2
            eyes_closed = eeg_data.iloc[:half_point]
            eyes_open = eeg_data.iloc[half_point:]

            # Calculate PSD for eyes-closed and eyes-open segments
            frequency_closed, psd_closed = calculate_psd(eyes_closed.mean(axis=1), sampling_freq)
            frequency_open, psd_open = calculate_psd(eyes_open.mean(axis=1), sampling_freq)

            # Filter to focus on the 0-30 Hz range
            focus_range = (frequency_closed >= 0) & (frequency_closed <= 30)
            frequency_closed = frequency_closed[focus_range]
            psd_closed = psd_closed[focus_range]
            psd_open = psd_open[focus_range]

            # Create and save the plot
            output_file_name = f"{os.path.splitext(file_name)[0]}_plot.png"
            output_path = os.path.join(output_folder, output_file_name)
            offset = 20
            plt.figure(figsize=(10, 6))
            plt.plot(frequency_closed, 10 * np.log10(psd_closed) + offset, label='Eyes Closed', color='blue')
            plt.plot(frequency_closed, 10 * np.log10(psd_open) + offset, label='Eyes Open', color='orange')
            plt.title(f'PSD of Eyes Open & Closed (0-30 Hz) ({file_name})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (dB)')
            plt.legend()
            plt.grid()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save the figure
            plt.close()  # Close the plot to free memory

            print(f"Figure saved to {output_path}")
