import torch
from torch.utils.data import DataLoader
from calibration.temperature_scaling import apply_temperature


def least_confidence_ts(model, unlabeled_dataset, n, device, T):
    # Least Confidence on temperature-scaled probabilities (post-hoc calibration)
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)

    logits = model.get_logits(loader, device)
    probabilities = apply_temperature(logits, T)

    max_probs = torch.max(probabilities, dim=1).values
    uncertainty = 1 - max_probs

    selected_indices = torch.argsort(uncertainty, descending=True)[:n]
    return selected_indices.tolist()
