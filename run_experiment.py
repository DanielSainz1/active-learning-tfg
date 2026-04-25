import argparse
import json
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from config import CONFIG
from models.cnn_fashion import create_model
from datasets.fashion_mnist import get_dataset
from engine.trainer import train, evaluate
from strategies.random_sampling import random_sampling
from strategies.margin import margin_sampling
from strategies.least_confidence import least_confidence
from strategies.entropy import entropy
from strategies.bald import bald
from strategies.least_confidence_ts import least_confidence_ts
from strategies.margin_ts import margin_ts
from strategies.entropy_ts import entropy_ts
from strategies.least_confidence_mcd import least_confidence_mcd
from strategies.margin_mcd import margin_mcd
from strategies.entropy_mcd import entropy_mcd
from calibration.temperature_scaling import optimize_temperature, make_val_loader

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", default="random")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
strategy = args.strategy
seed = args.seed

# Fix all random seeds for full reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Set up train data set and load dataset
labeled_indices, unlabeled_indices, train_dataset, test_dataset = get_dataset(
      seed,
      initial_pool_size=CONFIG["initial_pool_size"]
  )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []
patience = CONFIG["patience"]
min_improvement = CONFIG["min_improvement"]
round_idx = 0


# Active Learning loop
while len(labeled_indices) < CONFIG["budget"]:
    # Train model from scratch on current labeled pool
    model = create_model().to(device)
    labeled_subset = Subset(train_dataset, labeled_indices)
    train(model, labeled_subset, CONFIG, device)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["eval_batch_size"])
    # Evaluate on test set (always with the uncalibrated model)
    accuracy = evaluate(model, test_loader, device)
    unlabeled_subset = Subset(train_dataset, unlabeled_indices)

    # If the strategy uses Temperature Scaling, fit T on a stratified val split of L_t.
    # T is deterministic per (seed, round) thanks to a derived seed.
    T_value = None
    if strategy.endswith("_ts"):
        val_loader = make_val_loader(
            train_dataset,
            labeled_indices,
            CONFIG["calibration_val_fraction"],
            seed=seed * 10_000 + round_idx,
            batch_size=CONFIG["eval_batch_size"],
        )
        T_value = optimize_temperature(model, val_loader, device)
        print(f"Round {round_idx} | T = {T_value:.3f}")

    # Select new samples from unlabeled pool using the chosen strategy
    if strategy == "random":
        selected = random_sampling(unlabeled_indices, CONFIG["query_batch_size"], seed)
    elif strategy == "margin":
        positions = margin_sampling(model, unlabeled_subset, CONFIG["query_batch_size"], device)
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "least_confidence":
        positions = least_confidence(model, unlabeled_subset, CONFIG["query_batch_size"], device)
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "entropy":
        positions = entropy(model, unlabeled_subset, CONFIG["query_batch_size"], device)
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "bald":
        positions = bald(model, unlabeled_subset, CONFIG["query_batch_size"], device, CONFIG["mc_forward_passes"])
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "least_confidence_ts":
        positions = least_confidence_ts(model, unlabeled_subset, CONFIG["query_batch_size"], device, T_value)
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "margin_ts":
        positions = margin_ts(model, unlabeled_subset, CONFIG["query_batch_size"], device, T_value)
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "entropy_ts":
        positions = entropy_ts(model, unlabeled_subset, CONFIG["query_batch_size"], device, T_value)
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "least_confidence_mcd":
        positions = least_confidence_mcd(model, unlabeled_subset, CONFIG["query_batch_size"], device, CONFIG["mc_forward_passes"])
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "margin_mcd":
        positions = margin_mcd(model, unlabeled_subset, CONFIG["query_batch_size"], device, CONFIG["mc_forward_passes"])
        selected = [unlabeled_indices[pos] for pos in positions]
    elif strategy == "entropy_mcd":
        positions = entropy_mcd(model, unlabeled_subset, CONFIG["query_batch_size"], device, CONFIG["mc_forward_passes"])
        selected = [unlabeled_indices[pos] for pos in positions]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"Round | Labeled: {len(labeled_indices)}")

    # Update pools
    labeled_indices.extend(selected)
    unlabeled_indices = [i for i in unlabeled_indices if i not in selected]
    results.append({"labeled_size": len(labeled_indices), "accuracy": accuracy})
    round_idx += 1

    # Early stopping: stop if accuracy hasn't improved in the last N rounds
    if len(results) >= patience:
        recent = [r["accuracy"] for r in results[-patience:]]
        if max(recent) - min(recent) < min_improvement:
            print("Early stopping: accuracy has plateaued")
            break

#Save results
with open(f"results/results_{strategy}_seed{seed}.json", "w") as f:
    json.dump(results, f, indent=2)
