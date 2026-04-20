import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_COLORS, STRATEGY_LABELS, CONFIG

strategies = ["random", "least_confidence", "margin", "entropy", "bald"]
thresholds = CONFIG["label_efficiency_thresholds"]
budget = CONFIG["budget"]

# Group result files by strategy (one list of runs per strategy)
runs_by_strategy = {s: [] for s in strategies}
for file in sorted(glob.glob("results/results_*.json")):
    for strategy in strategies:
        if f"_{strategy}_" in file:
            with open(file) as f:
                runs_by_strategy[strategy].append(json.load(f))
            break

stop_cells = {}

# Print AULC + Final Acc + LE header
header = (
    f"{'Strategy':<20} {'AULC (mean ± std)':<22} "
    f"{'Final Acc (mean ± std)':<26} "
    + "  ".join([f"LE@{int(t*100)}%".ljust(14) for t in thresholds])
)
print(header)
print("-" * len(header))

for strategy in strategies:
    runs = runs_by_strategy[strategy]
    if not runs:
        continue

    # AULC per seed, then mean ± std across seeds
    aulcs = []
    for run in runs:
        x = np.array([p["labeled_size"] for p in run])
        y = np.array([p["accuracy"] for p in run])
        aulcs.append(np.trapezoid(y, x))
    aulc_mean = np.mean(aulcs)
    aulc_std = np.std(aulcs, ddof=1) if len(aulcs) > 1 else 0.0

    # Final accuracy per seed (last round), then mean ± std
    final_accs = [r[-1]["accuracy"] for r in runs]
    acc_mean = np.mean(final_accs)
    acc_std = np.std(final_accs, ddof=1) if len(final_accs) > 1 else 0.0

    # Label Efficiency per seed per threshold
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
            mean = np.mean(reached)
            std = np.std(reached, ddof=1) if len(reached) > 1 else 0.0
            tag = "" if len(reached) == len(runs) else f" ({len(reached)}/{len(runs)})"
            le_cells.append(f"{mean:.0f} ± {std:.0f}{tag}".ljust(14))

    name = STRATEGY_LABELS[strategy]
    aulc_cell = f"{aulc_mean:.2f} ± {aulc_std:.2f}"
    acc_cell = f"{acc_mean:.4f} ± {acc_std:.4f}"
    print(f"{name:<20} {aulc_cell:<22} {acc_cell:<26} " + "  ".join(le_cells))

    # Stopping point per seed (last labeled_size recorded)
    stops = [r[-1]["labeled_size"] for r in runs]
    early = [s for s in stops if s < budget]
    if early:
        m = np.mean(early)
        s = np.std(early, ddof=1) if len(early) > 1 else 0.0
        tag = "" if len(early) == len(stops) else f" ({len(early)}/{len(stops)})"
        stop_cells[strategy] = f"{m:.0f} ± {s:.0f}{tag}"
    else:
        stop_cells[strategy] = f"no activado (llegan a {budget})"

    # Plot: align all seeds to the shortest run
    min_rounds = min(len(r) for r in runs)
    x = np.array([runs[0][i]["labeled_size"] for i in range(min_rounds)])
    accs = np.array([[r[i]["accuracy"] for i in range(min_rounds)] for r in runs])
    y_mean = accs.mean(axis=0)
    y_std = accs.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(min_rounds)

    plt.plot(x, y_mean, label=STRATEGY_LABELS[strategy], color=PLOT_COLORS[strategy])
    plt.fill_between(x, y_mean - y_std, y_mean + y_std,
                     color=PLOT_COLORS[strategy], alpha=0.2)

# Stopping-criterion table
print()
print(f"{'Strategy':<20} {'|L_t| al detenerse':<40}")
print("-" * 60)
for strategy in strategies:
    if strategy in stop_cells:
        print(f"{STRATEGY_LABELS[strategy]:<20} {stop_cells[strategy]}")

plt.grid(True, alpha=0.3)
plt.title("Active Learning: Accuracy vs Labeled Samples")
plt.xlabel("Labeled samples")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results.png")
