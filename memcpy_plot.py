import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract image size from filename
def extract_image_size(filename):
    match = re.search(r'_img_(\d+)_', filename)
    return int(match.group(1)) if match else None

# Function to process CSV files matching a pattern
def process_csv_files(pattern):
    files = glob.glob(pattern)  # Find all matching files
    image_sizes = []
    transfer_times = []

    for file in files:
        try:
            # Read CSV
            df = pd.read_csv(file)

            # Filter for "CUDA memcpy Host-to-Device"
            transfer_row = df[df["Operation"] == "[CUDA memcpy Host-to-Device]"]

            if not transfer_row.empty:
                total_time_ns = transfer_row.iloc[0]["Total Time (ns)"]  # Get total time
                image_size = extract_image_size(file)  # Get image size from filename

                if image_size is not None:
                    image_sizes.append(image_size)
                    transfer_times.append(total_time_ns / 1e6)  # Convert ns to ms
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return image_sizes, transfer_times

# Main function to plot graph
def plot_transfer_time_vs_image_size(pattern):
    image_sizes, transfer_times = process_csv_files(pattern)

    if not image_sizes:
        print("No valid data found.")
        return

    # Sort data by image size
    sorted_data = sorted(zip(image_sizes, transfer_times))
    image_sizes, transfer_times = zip(*sorted_data)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(image_sizes, transfer_times, marker='o', linestyle='-')
    plt.xlabel("Image Size (px)")
    plt.ylabel("Transfer Time (ms)")
    plt.title("CUDA Host-to-Device memcpy Time vs. Image Size for v100, threads=256")
    plt.grid()
    plt.show()

# Example usage (modify regex pattern as needed)
pattern = "report/v100/v100_t_32_img_*_gpu_mem_time_sum.csv_cuda_gpu_mem_time_sum.csv"
# v100_t_4_img_256_gpu_mem_time_sum.csv_cuda_gpu_mem_time_sum
# v100_t_4_img_*_gpu_mem_time_sum.csv_cuda_gpu_mem_time_sum.csv
plot_transfer_time_vs_image_size(pattern)
