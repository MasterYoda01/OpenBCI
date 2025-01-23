import pandas as pd
import matplotlib.pyplot as plt

def process_and_plot(file_path, filter_column, metrics):
    # Read the data from the PDF table (requires tabula-py or camelot for PDF tables)
    # Placeholder for CSV conversion (update the file path with actual preprocessed CSV if needed)
    # Assuming data is preprocessed into a CSV file
    data = pd.read_csv(file_path)

    for metric in metrics:
        # Calculate overall average for the metric
        overall_avg = data[metric].mean()
        print(f"Overall Average of {metric}: {overall_avg}")

        # Calculate average per filter for the metric
        per_filter_avg = data.groupby(filter_column)[metric].mean()

        # Plot overall average as a single bar
        plt.figure(figsize=(6, 4))
        bars = plt.bar(['Overall'], [overall_avg], color='blue')
        plt.title(f'Overall Average {metric}')
        plt.ylabel(metric)

        # Annotate bars with values
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                     ha='center', va='bottom')

        plt.savefig(f'OpenBCI/anova_history/new_trial_anova_figs/overall_average_{metric}.png')
        plt.show()

        # Plot per-filter averages
        plt.figure(figsize=(10, 6))
        bars = per_filter_avg.plot(kind='bar', color='orange')
        plt.title(f'Average {metric} Per Filter')
        plt.ylabel(metric)
        plt.xlabel('Filter')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Annotate bars with values
        for idx, bar in enumerate(bars.patches):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
                     ha='center', va='bottom')

        plt.savefig(f'OpenBCI/anova_history/new_trial_anova_figs/per_filter_average_{metric}.png')
        plt.show()

# Paths to your preprocessed CSVs (update these paths to your actual files)
file_paths = {
    'SRP_analysis': 'OpenBCI/trial_results/results/trial-1_2_analysis_all_filters.csv',
    'SRP+Bias_analysis': 'OpenBCI/trial_results/results/trial-3_2_analysis_all_filters.csv',
    'SRP+Ground_analysis': 'OpenBCI/trial_results/results/trial-4_2_analysis_all_filters.csv'
}

# Define the columns for filters and metrics (update as needed based on column names)
filter_column = 'Filter'
metrics = ['Joe Z-Score Closed', 'Joe Z-Score Open', 'Mohammed Z-Score Closed', 'Mohammed Z-Score Open', 'Absolute Delta Closed', 'Absolute Delta Open', 'P-Value Closed', 'P-Value Open']  # Add other metrics as needed

# Process and plot each file
for name, path in file_paths.items():
    print(f"Processing {name}...")
    process_and_plot(path, filter_column, metrics)
