import pandas as pd
import matplotlib.pyplot as plt

# Given data (time in ns)
time_data = [45535, 452027, 458875, 451292,47488]
kernels = ["separateChannels","gaussianBlur(red)", "gaussianBlur(green)", "gaussianBlur(blue)", "recombineChannels"]

# Convert time from ns to ms
time_data_ms = [time / 1e6 for time in time_data]

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(kernels, time_data_ms, color=['orange', 'red', 'green', 'blue','pink'])

# Labels and title
plt.xlabel("Kernels")
plt.ylabel("Execution Time (ms)")
plt.title("Kernel Execution Time in ms for v100, threads=1024, image size=2048*2048")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()

# Show the plot
plt.show()
