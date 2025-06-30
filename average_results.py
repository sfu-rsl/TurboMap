import os
import re
import csv
import numpy as np
from collections import defaultdict

# Set the base directory to Results (relative to current working directory)
base_dir = os.path.join(os.getcwd(), "Results")
systems = ["FastMap"]
time_files = [
    "searchForTriangulation_time.txt",
    "MPCreation_time.txt",
    "searchInNeighbors_time.txt",
    "LBA_time.txt",
    "KFCulling_time.txt",
    "localMapping_time.txt"
]

results = defaultdict(lambda: defaultdict(dict))

for system in systems:
    system_path = os.path.join(base_dir, system, '1111')
    for sequence in os.listdir(system_path):
        sequence_path = os.path.join(system_path, sequence)
        if not os.path.isdir(sequence_path):
            continue

        rmse_vals = []
        component_avgs = {f: [] for f in time_files}
        component_stds = {f: [] for f in time_files}

        for run in os.listdir(sequence_path):
            run_path = os.path.join(sequence_path, run)
            if not os.path.isdir(run_path) or not run.startswith("v"):
                continue

            # Parse RMSE
            ostream_path = os.path.join(run_path, "ostream.txt")
            try:
                with open(ostream_path, "r") as f:
                    for line in f:
                        if "absolute_translational_error.rmse" in line:
                            match = re.search(r"rmse\s+([0-9.]+)", line)
                            if match:
                                rmse_vals.append(float(match.group(1)))
                            break
            except FileNotFoundError:
                continue

            # Parse time components
            data_dir = os.path.join(run_path, "LocalMapping", "data")
            for time_file in time_files:
                time_path = os.path.join(data_dir, time_file)
                try:
                    with open(time_path, "r") as f:
                        values = []
                        for line in f:
                            parts = line.strip().split(":")
                            if len(parts) == 2:
                                try:
                                    values.append(float(parts[1]))
                                except ValueError:
                                    continue
                        if values:
                            component_avgs[time_file].append(np.mean(values))
                            component_stds[time_file].append(np.std(values))
                except FileNotFoundError:
                    continue

        # Store average RMSE
        if rmse_vals:
            results[system][sequence]["rmse"] = (np.mean(rmse_vals), np.std(rmse_vals))

        # Store averaged means and stds for time files
        for time_file in time_files:
            avgs = component_avgs[time_file]
            stds = component_stds[time_file]
            if avgs:
                results[system][sequence][time_file] = (np.mean(avgs), np.mean(stds))

# Print to stdout and write to CSV
csv_filename = "average_times.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["System", "Sequence", "Metric", "Mean", "Std"])

    for system in systems:
        print(f"\n=== {system} ===")
        for sequence in sorted(results[system]):
            print(f"\n--- {sequence} ---")
            for key, (mean, std) in results[system][sequence].items():
                print(f"{key:35s} | Mean: {mean:.6f} | Std: {std:.6f}")
                writer.writerow([system, sequence, key, f"{mean:.6f}", f"{std:.6f}"])

print(f"\nResults also written to {csv_filename}")
