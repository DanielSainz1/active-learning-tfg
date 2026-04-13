import subprocess
import sys

seeds = [42, 123, 456, 789, 1024]
strategies = ["random", "least_confidence", "margin", "entropy", "bald"]

for seed in seeds:
    for strategy in strategies:
        print(f"Running {strategy} seed {seed}...")
        subprocess.run([sys.executable, "run_experiment.py",
                        "--strategy", strategy, "--seed", str(seed)])

# Generate plots
subprocess.run([sys.executable, "plot_results.py"])
