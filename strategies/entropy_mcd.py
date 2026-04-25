import torch
from torch.utils.data import DataLoader


def entropy_mcd(model, unlabeled_dataset, n, device, T):
    # Entropy Sampling on MC Dropout mean probabilities
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)

    probabilities = model.get_probabilities_mc(loader, device, T)

    uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

    selected_indices = torch.argsort(uncertainty, descending=True)[:n]
    return selected_indices.tolist()
