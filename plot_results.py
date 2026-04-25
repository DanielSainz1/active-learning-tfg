import glob
import json
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_COLORS, STRATEGY_LABELS, CONFIG

ORIGINAL = ["random", "least_confidence", "margin", "entropy", "bald"]
TS_VARIANTS = ["least_confidence_ts", "margin_ts", "entropy_ts"]
MCD_VARIANTS = ["least_confidence_mcd", "margin_mcd", "entropy_mcd"]
ALL_STRATEGIES = ORIGINAL + TS_VARIANTS + MCD_VARIANTS

thresholds = CONFIG["label_efficiency_thresholds"]
budget = CONFIG["budget"]

# Group result files by exact strategy name (regex avoids substring collisions
# like least_confidence being matched inside least_confidence_ts).
_file_pattern = re.compile(r"^results_(.+?)_seed\d+\.json$")


def load_runs(strategies):
    runs_by_strategy = {s: [] for s in strategies}
    for file in sorted(glob.glob("results/results_*.json")):
        m = _file_pattern.match(os.path.basename(file))
        if not m:
            continue
        strat = m.group(1)
        if strat in runs_by_strategy:
            with open(file) as f:
                runs_by_strategy[strat].append(json.load(f))
    return runs_by_strategy


runs_by_strategy = load_runs(ALL_STRATEGIES)


def compute_metrics(runs):
    # AULC and final accuracy per seed, then mean ± std.
    aulcs, final_accs = [], []
    for run in runs:
        x = np.array([p["labeled_size"] for p in run])
        y = np.array([p["accuracy"] for p in run])
        aulcs.append(np.trapezoid(y, x))
        final_accs.append(run[-1]["accuracy"])
    aulc_mean = np.mean(aulcs)
    aulc_std = np.std(aulcs, ddof=1) if len(aulcs) > 1 else 0.0
    acc_mean = np.mean(final_accs)
    acc_std = np.std(final_accs, ddof=1) if len(final_accs) > 1 else 0.0

    le_cells = []
    for t in thresholds:
        le_values = [
            next((p["labeled_size"] for p in run if p["accuracy"] >= t), None)
            for run in runs
        ]
        reached = [v for v in le_values if v is not None]
        if len(reached) == 0:
            le_cells.append("n.a.".ljust(14))
        else:
            m_le = np.mean(reached)
            s_le = np.std(reached, ddof=1) if len(reached) > 1 else 0.0
            tag = "" if len(reached) == len(runs) else f" ({len(reached)}/{len(runs)})"
            le_cells.append(f"{m_le:.0f} ± {s_le:.0f}{tag}".ljust(14))

    stops = [r[-1]["labeled_size"] for r in runs]
    early = [s for s in stops if s < budget]
    if early:
        m_stop = np.mean(early)
        s_stop = np.std(early, ddof=1) if len(early) > 1 else 0.0
        tag = "" if len(early) == len(stops) else f" ({len(early)}/{len(stops)})"
        stop_cell = f"{m_stop:.0f} ± {s_stop:.0f}{tag}"
    else:
        stop_cell = f"no activado (llegan a {budget})"

    return {
        "aulc_mean": aulc_mean, "aulc_std": aulc_std,
        "acc_mean": acc_mean, "acc_std": acc_std,
        "le_cells": le_cells, "stop_cell": stop_cell,
    }


def print_table(strategies, title):
    print()
    print(f"=== {title} ===")
    header = (
        f"{'Strategy':<28} {'AULC (mean ± std)':<22} "
        f"{'Final Acc (mean ± std)':<26} "
        + "  ".join([f"LE@{int(t*100)}%".ljust(14) for t in thresholds])
    )
    print(header)
    print("-" * len(header))
    stop_rows = []
    for strategy in strategies:
        runs = runs_by_strategy.get(strategy, [])
        if not runs:
            continue
        m = compute_metrics(runs)
        name = STRATEGY_LABELS.get(strategy, strategy)
        aulc_cell = f"{m['aulc_mean']:.2f} ± {m['aulc_std']:.2f}"
        acc_cell = f"{m['acc_mean']:.4f} ± {m['acc_std']:.4f}"
        print(f"{name:<28} {aulc_cell:<22} {acc_cell:<26} " + "  ".join(m['le_cells']))
        stop_rows.append((name, m['stop_cell']))

    print()
    print(f"{'Strategy':<28} {'|L_t| al detenerse':<40}")
    print("-" * 70)
    for name, cell in stop_rows:
        print(f"{name:<28} {cell}")


def plot_curves(strategies, output_path, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy in strategies:
        runs = runs_by_strategy.get(strategy, [])
        if not runs:
            continue
        min_rounds = min(len(r) for r in runs)
        x = np.array([runs[0][i]["labeled_size"] for i in range(min_rounds)])
        accs = np.array([[r[i]["accuracy"] for i in range(min_rounds)] for r in runs])
        y_mean = accs.mean(axis=0)
        y_std = accs.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(min_rounds)

        linestyle = '-'
        if strategy.endswith("_ts"):
            linestyle = '--'
        elif strategy.endswith("_mcd"):
            linestyle = ':'

        color = PLOT_COLORS.get(strategy, "#000000")
        label = STRATEGY_LABELS.get(strategy, strategy)

        ax.plot(x, y_mean, label=label, color=color, linestyle=linestyle)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        color=color, alpha=0.15)

    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Labeled samples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# Original plot (kept identical to the previous pipeline)
print_table(ORIGINAL, "Original strategies")
plot_curves(ORIGINAL, "results.png",
            "Active Learning: Accuracy vs Labeled Samples")

# Temperature Scaling comparison: each base strategy vs its _ts variant
ts_comparison = ["random",
                 "least_confidence", "least_confidence_ts",
                 "margin",           "margin_ts",
                 "entropy",          "entropy_ts"]
print_table(ts_comparison, "Temperature Scaling comparison")
plot_curves(ts_comparison, "results_calibration_ts.png",
            "Temperature Scaling vs Uncalibrated")

# MC Dropout comparison: each base strategy vs its _mcd variant
mcd_comparison = ["random",
                  "least_confidence", "least_confidence_mcd",
                  "margin",           "margin_mcd",
                  "entropy",          "entropy_mcd"]
print_table(mcd_comparison, "MC Dropout comparison")
plot_curves(mcd_comparison, "results_calibration_mcd.png",
            "MC Dropout vs Uncalibrated (Point Softmax)")

# Global table with all 11 strategies
print_table(ALL_STRATEGIES, "All strategies")


# =============================================================================
# Truncated AULC analysis: fair comparison across all 11 strategies.
# Calibrated variants trigger early stopping before the budget is exhausted,
# so their raw AULC integrates over a shorter x range and looks artificially
# small. Truncating every run to a common |L_t| range removes that bias.
# =============================================================================

def common_max_x(strategies):
    # Smallest final |L_t| across all runs of the given strategies.
    xs = [run[-1]["labeled_size"]
          for s in strategies for run in runs_by_strategy.get(s, [])]
    return min(xs) if xs else None


def compute_aulc_truncated(run, x_max):
    # AULC integrated from |L_0| to x_max. If the run does not reach x_max
    # exactly, the accuracy at x_max is linearly interpolated.
    x = np.array([p["labeled_size"] for p in run], dtype=float)
    y = np.array([p["accuracy"] for p in run], dtype=float)
    if x[-1] < x_max:
        return None
    mask = x <= x_max
    x_t = x[mask].copy()
    y_t = y[mask].copy()
    if x_t[-1] < x_max:
        i = np.searchsorted(x, x_max)
        x_lo, x_hi = x[i - 1], x[i]
        y_lo, y_hi = y[i - 1], y[i]
        y_at = y_lo + (y_hi - y_lo) * (x_max - x_lo) / (x_hi - x_lo)
        x_t = np.append(x_t, x_max)
        y_t = np.append(y_t, y_at)
    return float(np.trapezoid(y_t, x_t))


def plot_curves_truncated(strategies, x_max, output_path, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy in strategies:
        runs = runs_by_strategy.get(strategy, [])
        if not runs:
            continue
        runs_trunc = [[p for p in r if p["labeled_size"] <= x_max] for r in runs]
        runs_trunc = [r for r in runs_trunc if r]
        if not runs_trunc:
            continue
        min_rounds = min(len(r) for r in runs_trunc)
        x = np.array([runs_trunc[0][i]["labeled_size"] for i in range(min_rounds)])
        accs = np.array([[r[i]["accuracy"] for i in range(min_rounds)] for r in runs_trunc])
        y_mean = accs.mean(axis=0)
        y_std = accs.std(axis=0, ddof=1) if len(runs_trunc) > 1 else np.zeros(min_rounds)

        linestyle = '-'
        if strategy.endswith("_ts"):
            linestyle = '--'
        elif strategy.endswith("_mcd"):
            linestyle = ':'
        color = PLOT_COLORS.get(strategy, "#000000")
        label = STRATEGY_LABELS.get(strategy, strategy)

        ax.plot(x, y_mean, label=label, color=color, linestyle=linestyle)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)

    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Labeled samples")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


x_max_global = common_max_x(ALL_STRATEGIES)
if x_max_global is not None:
    print()
    print(f"=== Truncated AULC: AULC integrated over [|L_0|, {x_max_global}] ===")
    print(f"Use this table for fair comparison. Raw AULC above is biased because")
    print(f"calibrated variants early-stop before the {budget}-label budget.")
    header = f"{'Strategy':<28} {'AULC truncated (mean ± std)':<30}"
    print(header)
    print("-" * len(header))
    for strategy in ALL_STRATEGIES:
        runs = runs_by_strategy.get(strategy, [])
        if not runs:
            continue
        aulcs = [compute_aulc_truncated(r, x_max_global) for r in runs]
        aulcs = [a for a in aulcs if a is not None]
        if not aulcs:
            continue
        m = np.mean(aulcs)
        s = np.std(aulcs, ddof=1) if len(aulcs) > 1 else 0.0
        tag = "" if len(aulcs) == len(runs) else f" ({len(aulcs)}/{len(runs)})"
        name = STRATEGY_LABELS.get(strategy, strategy)
        print(f"{name:<28} {m:.2f} ± {s:.2f}{tag}")

    # Truncated learning curves figure (all 11 strategies on the same x range)
    plot_curves_truncated(
        ALL_STRATEGIES, x_max_global, "results_truncated.png",
        f"Learning Curves (truncated at |L_t| = {x_max_global})"
    )
