import argparse
import json
import os
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

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", default="random")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
strategy = args.strategy
seed = args.seed

#Set up train data set and load dataset
labeled_indices, unlabeled_indices, train_dataset, test_dataset = get_dataset(
      seed,
      initial_pool_size=CONFIG["initial_pool_size"]
  )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []
patience = CONFIG["patience"]
min_improvement = CONFIG["min_improvement"]


# Active Learning loop
while len(labeled_indices) < CONFIG["budget"]:
    # Train model from scratch on current labeled pool
    model = create_model().to(device)
    labeled_subset = Subset(train_dataset, labeled_indices)
    train(model, labeled_subset, CONFIG, device)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["eval_batch_size"])
    # Evaluate on test set
    accuracy = evaluate(model, test_loader, device)
    unlabeled_subset = Subset(train_dataset, unlabeled_indices)
    
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

    print(f"Round | Labeled: {len(labeled_indices)}")

    # Update pools
    labeled_indices.extend(selected)
    unlabeled_indices = [i for i in unlabeled_indices if i not in selected]
    results.append({"labeled_size": len(labeled_indices), "accuracy": accuracy})

    # Early stopping: stop if accuracy hasn't improved in the last N rounds
    if len(results) >= patience:
        recent = [r["accuracy"] for r in results[-patience:]]
        if max(recent) - min(recent) < min_improvement:
            print("Early stopping: accuracy has plateaued")
            break

#Save results
with open(f"results/results_{strategy}_seed{seed}.json", "w") as f:
    json.dump(results, f, indent=2)




