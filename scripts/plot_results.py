#!/usr/bin/env python3
"""Generate charts and summary tables from benchmark/training run results.

Usage:
    python scripts/plot_results.py [RUN_DIR]

If RUN_DIR is given, reads from:
    <RUN_DIR>/tables/summary.csv      — aggregated per-run metrics
    <RUN_DIR>/tables/all_epochs.csv   — consolidated per-epoch time series
and writes plots to:
    <RUN_DIR>/plots/

If no argument is given, finds the most recent results directory under results/.

Generates:
  From summary.csv:
    - Speedup bar charts (per config)
    - Execution time comparison (grouped bars)
    - Throughput scaling (line charts, OMP vs MPI)
  From all_epochs.csv (if present):
    - Training loss curves (per config)
    - Accuracy convergence vs wall-clock time (per config)
    - Epoch time distribution (box plots per config)
    - Epoch time scaling (mean epoch time vs parallelism)
"""

import csv
import math
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Logging helper: write to both stdout and log file
# ---------------------------------------------------------------------------
_log_fp = None


def _open_log(run_dir):
    global _log_fp
    log_path = os.path.join(run_dir, 'logs', 'plot_results.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log_fp = open(log_path, 'w')


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


# ---------------------------------------------------------------------------
# Resolve run directory
# ---------------------------------------------------------------------------
def find_run_dir():
    """Return an explicit directory from argv, or the most recent under results/."""
    if len(sys.argv) > 1:
        d = sys.argv[1]
        if os.path.isdir(d):
            return d
        print(f"Error: directory '{d}' does not exist.")
        sys.exit(1)

    # Fallback: scan results/ for the most recent run directory
    results_root = 'results'
    if not os.path.isdir(results_root):
        print(f"Error: '{results_root}/' not found.")
        sys.exit(1)

    candidates = []
    for name in os.listdir(results_root):
        full = os.path.join(results_root, name)
        summary = os.path.join(full, 'tables', 'summary.csv')
        if os.path.isdir(full) and os.path.isfile(summary):
            candidates.append(full)
    if not candidates:
        print("No run directories with tables/summary.csv found in results/")
        sys.exit(1)
    return max(candidates, key=lambda p: os.path.getmtime(
        os.path.join(p, 'tables', 'summary.csv')))


# ---------------------------------------------------------------------------
# Load summary CSV (aggregated per-run metrics)
# ---------------------------------------------------------------------------
def load_summary_csv(filepath):
    """Load summary CSV, averaging over multiple runs.

    Returns
    -------
    data : dict  {config: {impl_parallelism: mean_total_time_sec}}
    stats : dict {config: {impl_parallelism: {mean, stddev, min, max, n,
                   accuracy, throughput}}}
    """
    raw_time = defaultdict(lambda: defaultdict(list))
    raw_acc = defaultdict(lambda: defaultdict(list))
    raw_tput = defaultdict(lambda: defaultdict(list))
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
            tput = float(row.get('throughput_samples_per_sec', 0))
            raw_tput[config][key].append(tput)

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
            tputs = raw_tput[config][key]
            mean_tput = sum(tputs) / len(tputs) if tputs else 0.0
            data[config][key] = mean
            stats[config][key] = {
                'mean': mean,
                'stddev': stddev,
                'min': min(times),
                'max': max(times),
                'n': n,
                'accuracy': mean_acc,
                'throughput': mean_tput,
            }
    return data, stats


# ---------------------------------------------------------------------------
# Load consolidated epoch CSV
# ---------------------------------------------------------------------------
def load_epoch_csv(filepath):
    """Load all_epochs.csv into a structured dict.

    Returns
    -------
    epochs : dict
        {config: {impl_parallelism: {run: [row_dicts]}}}
        Each row_dict has: epoch (int), train_loss (float), test_accuracy (float),
        epoch_time_sec (float), cumulative_time_sec (float).
    """
    epochs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['config']
            impl = row['implementation']
            par = int(row['parallelism'])
            run = int(row['run'])
            key = f"{impl}_{par}"
            epochs[config][key][run].append({
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
                'test_accuracy': float(row['test_accuracy']),
                'epoch_time_sec': float(row['epoch_time_sec']),
                'cumulative_time_sec': float(row['cumulative_time_sec']),
            })
    return epochs


# ---------------------------------------------------------------------------
# Text summary table
# ---------------------------------------------------------------------------
def generate_text_table(data, stats):
    """Print a text summary table showing times, accuracy, and speedup."""
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


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
COLOR_SEQ = '#9E9E9E'
COLOR_OMP = '#2196F3'
COLOR_MPI = '#4CAF50'


def _impl_color(label):
    if 'openmp' in label.lower() or 'omp' in label.upper():
        return COLOR_OMP
    if 'mpi' in label.lower():
        return COLOR_MPI
    return COLOR_SEQ


# ---------------------------------------------------------------------------
# Plot 1: Speedup bar charts (existing, per config)
# ---------------------------------------------------------------------------
def plot_speedup(data, stats, output_dir):
    import matplotlib.pyplot as plt

    for config, results in sorted(data.items()):
        seq_time = results.get('sequential_1', None)
        if seq_time is None:
            continue

        cfg_stats = stats[config]
        seq_std = cfg_stats['sequential_1']['stddev']
        multi_run = any(s['n'] > 1 for s in cfg_stats.values())

        labels, speedups, speedup_errs, colors = [], [], [], []

        labels.append('Sequential')
        speedups.append(1.0)
        speedup_errs.append(0.0)
        colors.append(COLOR_SEQ)

        for key in sorted(results.keys()):
            if key == 'sequential_1':
                continue
            parts = key.split('_')
            impl = parts[0].upper()
            par = parts[1]
            t_mean = results[key]
            t_std = cfg_stats[key]['stddev']
            speedup = seq_time / t_mean if t_mean > 0 else 0
            if multi_run and t_mean > 0 and seq_time > 0:
                rel_seq = (seq_std / seq_time) ** 2
                rel_t = (t_std / t_mean) ** 2
                sp_err = speedup * math.sqrt(rel_seq + rel_t)
            else:
                sp_err = 0.0
            labels.append(f"{impl}\n({par})")
            speedups.append(speedup)
            speedup_errs.append(sp_err)
            colors.append(_impl_color(impl))

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 5))
        bars = ax.bar(labels, speedups,
                      yerr=speedup_errs if multi_run else None,
                      capsize=4 if multi_run else 0,
                      color=colors)
        ax.set_ylabel('Speedup')
        title = f'Speedup \u2014 {config}'
        if multi_run:
            n = cfg_stats['sequential_1']['n']
            title += f'  (avg of {n} runs)'
        ax.set_title(title)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        for bar, s in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{s:.2f}x', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        out = os.path.join(output_dir, f'speedup_{config}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log_print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Execution time comparison (grouped bars per config)
# ---------------------------------------------------------------------------
def plot_execution_time(data, stats, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    for config, results in sorted(data.items()):
        cfg_stats = stats[config]
        multi_run = any(s['n'] > 1 for s in cfg_stats.values())

        # Group by implementation
        impl_groups = defaultdict(list)  # impl -> [(par, mean, std)]
        for key in sorted(results.keys()):
            parts = key.split('_')
            impl = parts[0]
            par = int(parts[1])
            s = cfg_stats[key]
            impl_groups[impl].append((par, s['mean'], s['stddev']))

        if not impl_groups:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        impl_names = sorted(impl_groups.keys())
        # Get union of all parallelism levels
        all_pars = sorted(set(p for g in impl_groups.values() for p, _, _ in g))
        x = np.arange(len(all_pars))
        width = 0.8 / len(impl_names)

        for i, impl in enumerate(impl_names):
            par_map = {p: (m, s) for p, m, s in impl_groups[impl]}
            means = [par_map.get(p, (0, 0))[0] for p in all_pars]
            stds = [par_map.get(p, (0, 0))[1] for p in all_pars]
            offset = (i - len(impl_names) / 2 + 0.5) * width
            color = _impl_color(impl)
            ax.bar(x + offset, means, width,
                   yerr=stds if multi_run else None,
                   capsize=3 if multi_run else 0,
                   label=impl.upper(), color=color, alpha=0.85)

        ax.set_xlabel('Parallelism Level')
        ax.set_ylabel('Total Time (s)')
        ax.set_title(f'Execution Time \u2014 {config}')
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in all_pars])
        ax.legend()
        plt.tight_layout()
        out = os.path.join(output_dir, f'exec_time_{config}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log_print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Throughput scaling (line chart, OMP vs MPI per config)
# ---------------------------------------------------------------------------
def plot_throughput_scaling(data, stats, output_dir):
    import matplotlib.pyplot as plt

    for config, results in sorted(data.items()):
        cfg_stats = stats[config]

        # Collect throughput by impl
        impl_data = defaultdict(list)  # impl -> [(par, throughput)]
        for key in sorted(results.keys()):
            parts = key.split('_')
            impl = parts[0]
            par = int(parts[1])
            tput = cfg_stats[key].get('throughput', 0)
            if tput > 0:
                impl_data[impl].append((par, tput))

        if not impl_data:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for impl in sorted(impl_data.keys()):
            pts = sorted(impl_data[impl])
            pars = [p for p, _ in pts]
            tputs = [t for _, t in pts]
            color = _impl_color(impl)
            marker = 'o' if 'openmp' in impl else ('s' if 'mpi' in impl else '^')
            ax.plot(pars, tputs, marker=marker, color=color,
                    label=impl.upper(), linewidth=2, markersize=6)

        ax.set_xlabel('Parallelism Level')
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title(f'Throughput Scaling \u2014 {config}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(output_dir, f'throughput_{config}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log_print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4: Training loss curves (per config)
# ---------------------------------------------------------------------------
def plot_training_loss(epoch_data, output_dir):
    import matplotlib.pyplot as plt

    for config, impl_runs in sorted(epoch_data.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for key in sorted(impl_runs.keys()):
            runs = impl_runs[key]
            # Average across runs
            all_epochs_map = defaultdict(list)
            for run_id, rows in runs.items():
                for r in rows:
                    all_epochs_map[r['epoch']].append(r['train_loss'])

            epochs = sorted(all_epochs_map.keys())
            mean_loss = [sum(all_epochs_map[e]) / len(all_epochs_map[e]) for e in epochs]

            parts = key.split('_')
            impl = parts[0].upper()
            par = parts[1]
            label = f"{impl} ({par})" if impl != 'SEQUENTIAL' else 'Sequential'
            color = _impl_color(key)

            ax.plot(epochs, mean_loss, label=label, color=color,
                    alpha=0.7, linewidth=1.5)
            plotted = True

        if not plotted:
            plt.close()
            continue

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title(f'Training Loss \u2014 {config}')
        ax.legend(fontsize=8, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(output_dir, f'training_loss_{config}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log_print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 5: Accuracy convergence vs wall-clock time (per config)
# ---------------------------------------------------------------------------
def plot_accuracy_vs_time(epoch_data, output_dir):
    import matplotlib.pyplot as plt

    for config, impl_runs in sorted(epoch_data.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for key in sorted(impl_runs.keys()):
            runs = impl_runs[key]
            # Average across runs
            all_epochs_map = defaultdict(lambda: {'acc': [], 'time': []})
            for run_id, rows in runs.items():
                for r in rows:
                    all_epochs_map[r['epoch']]['acc'].append(r['test_accuracy'])
                    all_epochs_map[r['epoch']]['time'].append(r['cumulative_time_sec'])

            epochs = sorted(all_epochs_map.keys())
            mean_time = [sum(all_epochs_map[e]['time']) / len(all_epochs_map[e]['time']) for e in epochs]
            mean_acc = [sum(all_epochs_map[e]['acc']) / len(all_epochs_map[e]['acc']) * 100 for e in epochs]

            parts = key.split('_')
            impl = parts[0].upper()
            par = parts[1]
            label = f"{impl} ({par})" if impl != 'SEQUENTIAL' else 'Sequential'
            color = _impl_color(key)

            ax.plot(mean_time, mean_acc, label=label, color=color,
                    alpha=0.7, linewidth=1.5)
            plotted = True

        if not plotted:
            plt.close()
            continue

        ax.set_xlabel('Wall-Clock Time (s)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'Accuracy Convergence \u2014 {config}')
        ax.legend(fontsize=8, ncol=2, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(output_dir, f'accuracy_vs_time_{config}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log_print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 6: Epoch time distribution (box plots per config)
# ---------------------------------------------------------------------------
def plot_epoch_time_boxplot(epoch_data, output_dir):
    import matplotlib.pyplot as plt

    for config, impl_runs in sorted(epoch_data.items()):
        labels = []
        all_times = []
        colors = []

        for key in sorted(impl_runs.keys()):
            runs = impl_runs[key]
            times = []
            for run_id, rows in runs.items():
                for r in rows:
                    times.append(r['epoch_time_sec'])
            if not times:
                continue
            parts = key.split('_')
            impl = parts[0].upper()
            par = parts[1]
            label = f"{impl}\n({par})"
            labels.append(label)
            all_times.append(times)
            colors.append(_impl_color(key))

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 5))
        bp = ax.boxplot(all_times, labels=labels, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.5})
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Epoch Time (s)')
        ax.set_title(f'Epoch Time Distribution \u2014 {config}')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        out = os.path.join(output_dir, f'epoch_time_boxplot_{config}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log_print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 7: Epoch time scaling (mean epoch time vs parallelism)
# ---------------------------------------------------------------------------
def plot_epoch_time_scaling(epoch_data, output_dir):
    import matplotlib.pyplot as plt

    for config, impl_runs in sorted(epoch_data.items()):
        # Collect mean epoch time by impl and parallelism
        impl_data = defaultdict(list)  # impl -> [(par, mean_epoch_time)]
        for key in sorted(impl_runs.keys()):
            runs = impl_runs[key]
            all_times = []
            for run_id, rows in runs.items():
                for r in rows:
                    all_times.append(r['epoch_time_sec'])
            if not all_times:
                continue
            mean_t = sum(all_times) / len(all_times)
            parts = key.split('_')
            impl = parts[0]
            par = int(parts[1])
            impl_data[impl].append((par, mean_t))

        if not impl_data:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for impl in sorted(impl_data.keys()):
            pts = sorted(impl_data[impl])
            pars = [p for p, _ in pts]
            times = [t for _, t in pts]
            color = _impl_color(impl)
            marker = 'o' if 'openmp' in impl else ('s' if 'mpi' in impl else '^')
            ax.plot(pars, times, marker=marker, color=color,
                    label=impl.upper(), linewidth=2, markersize=6)

        ax.set_xlabel('Parallelism Level')
        ax.set_ylabel('Mean Epoch Time (s)')
        ax.set_title(f'Epoch Time Scaling \u2014 {config}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(output_dir, f'epoch_time_scaling_{config}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log_print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_dir = find_run_dir()
    _open_log(run_dir)

    summary_csv = os.path.join(run_dir, 'tables', 'summary.csv')
    epochs_csv = os.path.join(run_dir, 'tables', 'all_epochs.csv')
    plot_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    log_print(f"Run directory: {run_dir}")

    # --- Summary-based outputs ---
    if os.path.isfile(summary_csv):
        log_print(f"Loading {summary_csv}")
        data, stats = load_summary_csv(summary_csv)
        generate_text_table(data, stats)

        try:
            import matplotlib
            matplotlib.use('Agg')
            log_print("\nGenerating summary plots...")
            plot_speedup(data, stats, plot_dir)
            plot_execution_time(data, stats, plot_dir)
            plot_throughput_scaling(data, stats, plot_dir)
        except ImportError:
            log_print("matplotlib not installed \u2014 skipping plot generation.")
            log_print("Install with: pip install matplotlib")
    else:
        log_print(f"Warning: {summary_csv} not found \u2014 skipping summary analysis.")

    # --- Epoch-based plots ---
    if os.path.isfile(epochs_csv):
        log_print(f"\nLoading {epochs_csv}")
        epoch_data = load_epoch_csv(epochs_csv)

        try:
            import matplotlib
            matplotlib.use('Agg')
            log_print("Generating epoch plots...")
            plot_training_loss(epoch_data, plot_dir)
            plot_accuracy_vs_time(epoch_data, plot_dir)
            plot_epoch_time_boxplot(epoch_data, plot_dir)
            plot_epoch_time_scaling(epoch_data, plot_dir)
        except ImportError:
            log_print("matplotlib not installed \u2014 skipping epoch plots.")
    else:
        log_print(f"\nNote: {epochs_csv} not found \u2014 skipping epoch-based plots.")

    log_print("\nDone.")
    _close_log()
