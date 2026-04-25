import torch
from torch.utils.data import DataLoader


def least_confidence_mcd(model, unlabeled_dataset, n, device, T):
    # Least Confidence on MC Dropout mean probabilities
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)

    probabilities = model.get_probabilities_mc(loader, device, T)

    max_probs = torch.max(probabilities, dim=1).values
    uncertainty = 1 - max_probs

    selected_indices = torch.argsort(uncertainty, descending=True)[:n]
    return selected_indices.tolist()
