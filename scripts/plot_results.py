#!/usr/bin/env python3
"""Generate speedup charts and summary tables from benchmark CSV results.

Reads the most recent CSV from results/tables/, computes speedup relative
to the sequential baseline, and optionally generates bar charts with matplotlib.
"""

import csv
import os
import sys
from collections import defaultdict

def load_csv(filepath):
    """Load benchmark CSV into a nested dict: {config: {impl_parallelism: time_sec}}."""
    data = defaultdict(dict)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['config']
            impl = row['implementation']
            par = int(row['parallelism'])
            time_sec = float(row['epoch_time_sec'])
            key = f"{impl}_{par}"
            data[config][key] = time_sec
    return data

def generate_text_table(data):
    """Print a text summary table showing times and speedup vs sequential baseline."""
    for config, results in sorted(data.items()):
        seq_time = results.get('sequential_1', None)
        if seq_time is None:
            continue

        print(f"\n{'='*60}")
        print(f"  Network: {config}")
        print(f"{'='*60}")
        print(f"  {'Implementation':<30} {'Time [s]':>10} {'Speedup':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Sequential':<30} {seq_time:>10.3f} {1.0:>10.2f}")

        for key, time_sec in sorted(results.items()):
            if key == 'sequential_1':
                continue
            parts = key.split('_')
            impl = parts[0].upper()
            par = parts[1]
            speedup = seq_time / time_sec if time_sec > 0 else 0
            label = f"{impl} ({par} {'threads' if impl == 'OPENMP' else 'procs'})"
            print(f"  {label:<30} {time_sec:>10.3f} {speedup:>10.2f}")

def generate_plots(data, output_dir):
    """Generate per-config speedup bar charts (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation.")
        print("Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    for config, results in sorted(data.items()):
        seq_time = results.get('sequential_1', None)
        if seq_time is None:
            continue

        labels = []
        speedups = []

        labels.append('Sequential')
        speedups.append(1.0)

        for key in sorted(results.keys()):
            if key == 'sequential_1':
                continue
            parts = key.split('_')
            impl = parts[0].upper()
            par = parts[1]
            speedup = seq_time / results[key] if results[key] > 0 else 0
            labels.append(f"{impl}\n({par})")
            speedups.append(speedup)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, speedups, color=['#2196F3' if 'OMP' in l else
                                                '#4CAF50' if 'MPI' in l else
                                                '#9E9E9E' for l in labels])
        ax.set_ylabel('Speedup')
        ax.set_title(f'Speedup — {config}')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        for bar, s in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{s:.2f}x', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'speedup_{config}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    # Find the most recent benchmark CSV in results/tables/
    tables_dir = 'results/tables'
    csvs = sorted([f for f in os.listdir(tables_dir) if f.endswith('.csv')])
    if not csvs:
        print("No CSV files found in results/tables/")
        sys.exit(1)

    csv_path = os.path.join(tables_dir, csvs[-1])
    print(f"Loading {csv_path}")

    data = load_csv(csv_path)
    generate_text_table(data)
    generate_plots(data, 'results/plots')
