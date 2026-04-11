import numpy as np

def random_sampling(unlabeled_indices, n, seed=None):
    # Select n random samples from the unlabeled pool (baseline strategy)
    rng = np.random.default_rng(seed)
    selected = rng.choice(unlabeled_indices, size=n, replace=False)
    return selected

