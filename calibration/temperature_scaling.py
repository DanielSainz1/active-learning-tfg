import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def stratified_val_indices(labeled_indices, targets, fraction, rng):
    # Pick a stratified subset of labeled_indices, keeping class balance.
    labeled_arr = np.array(labeled_indices)
    labels = np.array([int(targets[i]) for i in labeled_indices])
    val = []
    for c in np.unique(labels):
        positions = np.where(labels == c)[0]
        n_val = max(1, int(round(len(positions) * fraction)))
        n_val = min(n_val, len(positions))
        selected = rng.choice(positions, size=n_val, replace=False)
        val.extend(labeled_arr[selected].tolist())
    return val


def make_val_loader(train_dataset, labeled_indices, fraction, seed, batch_size):
    # Stratified validation DataLoader built from the current labeled pool.
    rng = np.random.default_rng(seed)
    val_idx = stratified_val_indices(
        labeled_indices, train_dataset.targets, fraction, rng
    )
    val_subset = Subset(train_dataset, val_idx)
    return DataLoader(val_subset, batch_size=batch_size, shuffle=False)


def optimize_temperature(model, val_loader, device, max_iter=50):
    # Optimize a single temperature scalar T > 0 that minimizes NLL on val_loader.
    # Guo et al. 2017, "On Calibration of Modern Neural Networks".
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            all_logits.append(model(images).cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    T = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return max(float(T.item()), 1e-3)


def apply_temperature(logits, T):
    # Divide logits by T and apply softmax to get calibrated probabilities.
    return F.softmax(logits / T, dim=1)
