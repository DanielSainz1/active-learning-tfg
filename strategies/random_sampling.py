import numpy as np

def random_sampling(unlabeled_indices, n, seed=None):
    rng = np.random.default_rng(seed)
    selected = rng.choice(unlabeled_indices, size=n, replace=False)
    return selected

