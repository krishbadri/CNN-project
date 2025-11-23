#!/usr/bin/env python3
"""
Plot benchmarking results (if you save timing data to CSV)
"""

import matplotlib.pyplot as plt
import csv
import sys

def plot_from_csv(csv_file):
    """Plot timing data from CSV file"""
    layers = []
    cpu_times = []
    gpu_times = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layers.append(row['layer'])
            cpu_times.append(float(row['cpu_ms']))
            gpu_times.append(float(row['gpu_ms']))
    
    x = range(len(layers))
    width = 0.35
    
    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], cpu_times, width, label='CPU', color='blue', alpha=0.7)
    ax.bar([i + width/2 for i in x], gpu_times, width, label='GPU', color='green', alpha=0.7)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Time (ms)')
    ax.set_title('CPU vs GPU Layer Timings')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark.png', dpi=150)
    print("Saved plot to benchmark.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_benchmarks.py <timings.csv>")
        sys.exit(1)
    
    plot_from_csv(sys.argv[1])

