# =============================================================================
# config.py — Main configuration for the Active Learning Experiment.
# =============================================================================
# This file contains ALL of the experiments hyperparameters.
# =============================================================================

CONFIG = {
    # -------------------------------------------------------------------------
    # Active Learning cycle
    # -------------------------------------------------------------------------
    # n₀: Size of the initial labeled pool (seed L₀)
    "initial_pool_size": 100,

    # b: Number of instances acquired per iteration in the AL cycle
    "query_batch_size": 100,

    # B: Total budget for labels (max number of instances to label)
    # Cycle stops when |L| reaches that value
    "budget": 5000,

    # Number of repetitions with different seeds to estimate variability
    "n_seeds": 5,

    # -------------------------------------------------------------------------
    # CNN's training
    # -------------------------------------------------------------------------
    # Epochs of training in each AL cycle (model is trained from scratch)
    "train_epochs": 20,

    # Size of the mini-batch for the training DataLoader
    "train_batch_size": 64,

    # Adam's optimizer learning rate (like explained in Chapter 2)
    "learning_rate": 0.001,

    # Batch size for the evaluation / prediction phase (the bigger the quicker)
    # It doesn't affect to the model's quality, only inference's speed
    "eval_batch_size": 256,

    # -------------------------------------------------------------------------
    # MC Dropout (BALD and *_mcd strategies)
    # -------------------------------------------------------------------------
    # T: Number of stochastic forward passes to estimate uncertainty
    "mc_forward_passes": 5,

    # -------------------------------------------------------------------------
    # Temperature Scaling (*_ts strategies)
    # -------------------------------------------------------------------------
    # Fraction of the labeled pool used as validation set to optimize T.
    "calibration_val_fraction": 0.15,

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    # Fixed seeds for every of the n_seeds repetitions
    # Fixing the seed to make sure the experiments can be reproduced
    # This makes it so that if someone runs the code with the same seeds,
    # he'll obtain the exact same results.
    "base_seeds": [42, 123, 456, 789, 1024],

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    # Accuracy thresholds to calculate Label Efficiency (LE).
    # LE(a*) = Minimum number of labels to reach accuracy a*.
    "label_efficiency_thresholds": [0.80, 0.85, 0.88, 0.90],
    # Early stopping
    "patience": 5,
    "min_improvement": 0.005,
}

# Available acquisition strategies
STRATEGIES = [
    "random", "least_confidence", "margin", "entropy", "bald",
    # Temperature Scaling variants
    "least_confidence_ts", "margin_ts", "entropy_ts",
    # MC Dropout variants (reuse mc_forward_passes)
    "least_confidence_mcd", "margin_mcd", "entropy_mcd",
]

# Datasets available
DATASETS = ["fashion_mnist", "cifar10"]

# Colours for the graphics. *_ts and *_mcd share the color of their base
# strategy; they are distinguished by line style in plot_results.py.
PLOT_COLORS = {
    "random":                "#888888",  # gray
    "least_confidence":      "#E69F00",  # orange
    "margin":                "#56B4E9",  # light blue
    "entropy":               "#009E73",  # green
    "bald":                  "#CC79A7",  # pink/magenta
    "least_confidence_ts":   "#E69F00",
    "margin_ts":             "#56B4E9",
    "entropy_ts":            "#009E73",
    "least_confidence_mcd":  "#E69F00",
    "margin_mcd":            "#56B4E9",
    "entropy_mcd":           "#009E73",
}

# Readable names for the legends of the graphs
STRATEGY_LABELS = {
    "random":                "Random",
    "least_confidence":      "Least Confidence",
    "margin":                "Margin Sampling",
    "entropy":               "Entropy Sampling",
    "bald":                  "BALD (MC Dropout)",
    "least_confidence_ts":   "Least Confidence (TS)",
    "margin_ts":             "Margin Sampling (TS)",
    "entropy_ts":            "Entropy Sampling (TS)",
    "least_confidence_mcd":  "Least Confidence (MCD)",
    "margin_mcd":            "Margin Sampling (MCD)",
    "entropy_mcd":           "Entropy Sampling (MCD)",
}
