#!/usr/bin/env python3
"""Generate speedup charts and summary tables from benchmark CSV results.

Reads the most recent CSV from results/tables/, computes speedup relative
to the sequential baseline, and optionally generates bar charts with matplotlib.
"""

import csv
import math
import os
import sys
from collections import defaultdict

# --- Logging helper: write to both stdout and log file ---
_log_fp = None

def _open_log():
    global _log_fp
    os.makedirs('results/logs', exist_ok=True)
    _log_fp = open('results/logs/plot_results.log', 'w')

def _close_log():
    global _log_fp
    if _log_fp:
        _log_fp.close()
        _log_fp = None

def log_print(msg=''):
    print(msg)
    if _log_fp:
        _log_fp.write(msg + '\n')
        _log_fp.flush()

def load_csv(filepath):
    """Load benchmark CSV, averaging over multiple runs.

    Returns
    -------
    data : dict
        {config: {impl_parallelism: mean_total_time_sec}}
    stats : dict
        {config: {impl_parallelism: {mean, stddev, min, max, n, accuracy}}}
    """
    raw_time = defaultdict(lambda: defaultdict(list))
    raw_acc  = defaultdict(lambda: defaultdict(list))
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['config']
            impl = row['implementation']
            par = int(row['parallelism'])
            key = f"{impl}_{par}"
            time_sec = float(row['total_time_sec'])
            raw_time[config][key].append(time_sec)
            accuracy = float(row.get('final_accuracy', 0))
            raw_acc[config][key].append(accuracy)

    data = defaultdict(dict)
    stats = defaultdict(dict)
    for config, results in raw_time.items():
        for key, times in results.items():
            n = len(times)
            mean = sum(times) / n
            variance = sum((t - mean) ** 2 for t in times) / n if n > 1 else 0.0
            stddev = math.sqrt(variance)
            accs = raw_acc[config][key]
            mean_acc = sum(accs) / len(accs) if accs else 0.0
            data[config][key] = mean
            stats[config][key] = {
                'mean': mean,
                'stddev': stddev,
                'min': min(times),
                'max': max(times),
                'n': n,
                'accuracy': mean_acc,
            }
    return data, stats

def generate_text_table(data, stats):
    """Print a text summary table showing times, accuracy, and speedup vs sequential baseline."""
    for config, results in sorted(data.items()):
        seq_time = results.get('sequential_1', None)
        if seq_time is None:
            continue

        cfg_stats = stats[config]
        multi_run = any(s['n'] > 1 for s in cfg_stats.values())

        log_print(f"\n{'='*80}")
        log_print(f"  Network: {config}")
        log_print(f"{'='*80}")
        if multi_run:
            log_print(f"  {'Implementation':<30} {'Time [s]':>16} {'Accuracy':>10} {'Speedup':>10} {'Runs':>6}")
            log_print(f"  {'-'*72}")
            seq_s = cfg_stats['sequential_1']
            log_print(f"  {'Sequential':<30} {seq_time:>8.3f} \u00b1 {seq_s['stddev']:<5.3f} {seq_s['accuracy']*100:>9.2f}% {1.0:>10.2f} {seq_s['n']:>5d}")
        else:
            log_print(f"  {'Implementation':<30} {'Time [s]':>10} {'Accuracy':>10} {'Speedup':>10}")
            log_print(f"  {'-'*60}")
            seq_s = cfg_stats['sequential_1']
            log_print(f"  {'Sequential':<30} {seq_time:>10.3f} {seq_s['accuracy']*100:>9.2f}% {1.0:>10.2f}")

        for key, time_sec in sorted(results.items()):
            if key == 'sequential_1':
                continue
            parts = key.split('_')
            impl = parts[0].upper()
            par = parts[1]
            speedup = seq_time / time_sec if time_sec > 0 else 0
            s = cfg_stats[key]
            label = f"{impl} ({par} {'threads' if impl == 'OPENMP' else 'procs'})"
            if multi_run:
                log_print(f"  {label:<30} {time_sec:>8.3f} \u00b1 {s['stddev']:<5.3f} {s['accuracy']*100:>9.2f}% {speedup:>10.2f} {s['n']:>5d}")
            else:
                log_print(f"  {label:<30} {time_sec:>10.3f} {s['accuracy']*100:>9.2f}% {speedup:>10.2f}")

def generate_plots(data, stats, output_dir):
    """Generate per-config speedup bar charts (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_print("matplotlib not installed — skipping plot generation.")
        log_print("Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    for config, results in sorted(data.items()):
        seq_time = results.get('sequential_1', None)
        if seq_time is None:
            continue

        cfg_stats = stats[config]
        seq_std = cfg_stats['sequential_1']['stddev']
        multi_run = any(s['n'] > 1 for s in cfg_stats.values())

        labels = []
        speedups = []
        speedup_errs = []

        labels.append('Sequential')
        speedups.append(1.0)
        speedup_errs.append(0.0)

        for key in sorted(results.keys()):
            if key == 'sequential_1':
                continue
            parts = key.split('_')
            impl = parts[0].upper()
            par = parts[1]
            t_mean = results[key]
            t_std = cfg_stats[key]['stddev']
            speedup = seq_time / t_mean if t_mean > 0 else 0
            # Error propagation: S = T_seq / T, dS = S * sqrt((dT_seq/T_seq)^2 + (dT/T)^2)
            if multi_run and t_mean > 0 and seq_time > 0:
                rel_seq = (seq_std / seq_time) ** 2
                rel_t = (t_std / t_mean) ** 2
                sp_err = speedup * math.sqrt(rel_seq + rel_t)
            else:
                sp_err = 0.0
            labels.append(f"{impl}\n({par})")
            speedups.append(speedup)
            speedup_errs.append(sp_err)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, speedups,
                      yerr=speedup_errs if multi_run else None,
                      capsize=4 if multi_run else 0,
                      color=['#2196F3' if 'OMP' in l else
                             '#4CAF50' if 'MPI' in l else
                             '#9E9E9E' for l in labels])
        ax.set_ylabel('Speedup')
        title = f'Speedup — {config}'
        if multi_run:
            n = cfg_stats['sequential_1']['n']
            title += f'  (avg of {n} runs)'
        ax.set_title(title)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        for bar, s in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{s:.2f}x', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'speedup_{config}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        log_print(f"Saved: {out_path}")

if __name__ == '__main__':
    _open_log()

    # Find the most recent benchmark CSV in results/tables/
    tables_dir = 'results/tables'
    csvs = [f for f in os.listdir(tables_dir) if f.endswith('.csv')]
    if not csvs:
        log_print("No CSV files found in results/tables/")
        _close_log()
        sys.exit(1)

    csv_path = os.path.join(tables_dir,
        max(csvs, key=lambda f: os.path.getmtime(os.path.join(tables_dir, f))))

    log_print(f"Loading {csv_path}")

    data, stats = load_csv(csv_path)
    generate_text_table(data, stats)
    generate_plots(data, stats, 'results/plots')
    _close_log()
