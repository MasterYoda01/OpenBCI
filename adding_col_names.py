import pandas as pd

# File paths
input_file_path = '/path/to/your/input/file.csv'
filtered_output_file_path = '/path/to/your/output/file_filtered.csv'

# Columns to keep
columns_to_keep = [
    "Sample Index", "EXG Channel 0", "EXG Channel 1", "EXG Channel 2",
    "EXG Channel 3", "EXG Channel 4", "EXG Channel 5", "EXG Channel 6", "EXG Channel 7"
]

# Load the file as tab-separated
df = pd.read_csv(input_file_path, header=None, delimiter="\t")

# Filter to only include necessary columns
df_filtered = df.iloc[:, :len(columns_to_keep)]

# Assign column names
df_filtered.columns = columns_to_keep

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(filtered_output_file_path, index=False)
