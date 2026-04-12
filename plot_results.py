import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_COLORS, STRATEGY_LABELS

#Strategies to compare
strategies = ["random", "least_confidence", "margin", "entropy", "bald"]

#Group result files by strategy
files_by_strategy = {s: [] for s in strategies}

for file in glob.glob("results/results_*.json"):
    for strategy in strategies:
        if strategy in file:
            files_by_strategy[strategy].append(file)
            break

averaged_by_strategy = {}

#Load all runs for each strategy
for strategy in strategies:
    all_runs = []
    for file in files_by_strategy[strategy]:
        with open(file) as f:
            data = json.load(f)
            all_runs.append(data)

#Average accuracy and std for every seed
    if all_runs:
        n_rounds = len(all_runs[0])
        averaged = []
        for i in range(n_rounds):
            labeled_size = all_runs[0][i]["labeled_size"]
            avg_accuracy = sum(run[i]["accuracy"] for run in all_runs) / len(all_runs)
            std_accuracy = np.std([run[i]["accuracy"] for run in all_runs])
            averaged.append({"labeled_size": labeled_size, "accuracy": avg_accuracy, "std": std_accuracy})
        averaged_by_strategy[strategy] = averaged

for strategy in averaged_by_strategy:       
    dataplot = averaged_by_strategy[strategy]
    x = [point["labeled_size"]for point in dataplot]
    y = [point["accuracy"] for point in dataplot]
    std = [point["std"] for point in dataplot]

    #Calculate AUC with trapezoid rule in numpy
    auc = np.trapezoid(y, x)
    print (f"{strategy}: AUC = {auc:.2f}")

    for threshold in [0.80, 0.85, 0.88, 0.90]:
      for size, acc in zip(x, y):
          if acc >= threshold:
              print(f"{strategy} @ {threshold}: {size} samples")
              break
      else:
        print(f"{strategy} @ {threshold}: never reached")

    plt.plot(x, y, label=STRATEGY_LABELS[strategy], color=PLOT_COLORS[strategy])
    plt.fill_between(x,
                 [a - s for a, s in zip(y,std)],
                 [a + s for a, s in zip(y,std)],
                 color = PLOT_COLORS[strategy],
                 alpha = 0.2)
plt.grid(True, alpha=0.3)
plt.title("Active Learning: Accuracy vs Labeled Samples")
plt.xlabel("Labeled samples")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results.png")

