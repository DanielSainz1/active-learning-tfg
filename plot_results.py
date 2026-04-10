import glob
import json
import matplotlib.pyplot as plt

#Strategies to compare
strategies = ["random", "least_confidence", "margin", "entropy", "bald"]

#Group result files by strategy
files_by_strategy = {s: [] for s in strategies}

for file in glob.glob("results_*json"):
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

#Average accuracy for every seed
    if all_runs:
        n_rounds = len(all_runs[0])
        averaged = []
        for i in range(n_rounds):
            labeled_size = all_runs[0][i]["labeled_size"]
            avg_accuracy = sum(run[i]["accuracy"] for run in all_runs) / len(all_runs)
            averaged.append({"labeled_size": labeled_size, "accuracy": avg_accuracy})
        averaged_by_strategy[strategy] = averaged

for strategy in averaged_by_strategy:       
    dataplot = averaged_by_strategy[strategy]
    x = [point["labeled_size"]for point in dataplot]
    y = [point["accuracy"] for point in dataplot]
    plt.plot(x, y, label=strategy)

plt.xlabel("Labeled samples")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results.png")

