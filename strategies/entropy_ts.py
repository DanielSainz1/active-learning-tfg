import torch
from torch.utils.data import DataLoader
from calibration.temperature_scaling import apply_temperature


def entropy_ts(model, unlabeled_dataset, n, device, T):
    # Entropy Sampling on temperature-scaled probabilities
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)

    logits = model.get_logits(loader, device)
    probabilities = apply_temperature(logits, T)

    uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

    selected_indices = torch.argsort(uncertainty, descending=True)[:n]
    return selected_indices.tolist()
