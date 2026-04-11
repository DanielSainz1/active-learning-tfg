# =============================================================================
# datasets/fashion_mnist.py — Load and prepare Fashion-MNIST
# =============================================================================
# Fashion-MNIST
# - 60.000 images to train / 10.000 to test
# - 10 different classes of clothes
# - 28×28 pixels, 1 channel (grayscale)
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# Normalization statistics from Fashion-MNIST (mean and std)
# These are calculated from the whole training set
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD  = (0.3530,)


def get_transform():
    """
    Returns standard transform for Fashion-MNIST.
    We only normalize: convert to tensor and adjust mean/std
    Data augmentation doesn't apply
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ])


def get_dataset(seed: int, initial_pool_size: int = 100):
    """
    Loads Fashion-MNIST and divides the training set into:
     - labeled_indices:   Indices for the 'inital_pool_size' first samples (L₀)
     - unlabeled_indices: Indices for the rest of the train (pool not labeled U₀)
     - train_dataset:     Whole train dataset (Necessary to access samples)
     - test_dataset:      Whole dataset (To evaluate accuracy)

    The initial seed L₀ is selected in a STRATIFIED way: you pick 
    (inital_pool_size // n_classes) samples from every class, ensuring that
    all classes are represented since the start

    Parameters
    ----------
    seed : int
        Seed to select L₀ randomly. Ensures reproducibility
    initial_pool_size : int
        Size of the initial seed. 

    Returns
    -------
    labeled_indices : list[int]
        Indices (from train_dataset) of the initially labelled pool.
    unlabeled_indices : list[int]
        Indices (from train_dataset) of the unlabelled pool.
    train_dataset : torchvision.datasets.FashionMNIST
        Whole training dataset with transformations applied
    test_dataset : torchvision.datasets.FashionMNIST
        Whole test dataset
    
    """
    transform = get_transform()

    # Download and load data
    # data_root='./data' saves the files in the local 'data' folder
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )

    # Get all of the labels from train to do the stratified selection
    all_labels = np.array(train_dataset.targets)
    n_classes = len(np.unique(all_labels))
    n_per_class = initial_pool_size // n_classes  # ex: 100 // 10 = 10 per class

    # Fix random seed to ensure reproducibility
    rng = np.random.default_rng(seed)

    labeled_indices = []
    for class_idx in range(n_classes):
        # Indices of all samples from this class
        class_indices = np.where(all_labels == class_idx)[0]
        # Select n_per_class randomly from this class
        selected = rng.choice(class_indices, size=n_per_class, replace=False)
        labeled_indices.extend(selected.tolist())

    # The non labeled pool is all of the train minus the indices already selected
    labeled_set = set(labeled_indices)
    unlabeled_indices = [i for i in range(len(train_dataset)) if i not in labeled_set]

    return labeled_indices, unlabeled_indices, train_dataset, test_dataset


# Names of the 10 different classes in Fashion-MNIST (useful for the plots)
FASHION_MNIST_CLASSES = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot",   # 9
]
