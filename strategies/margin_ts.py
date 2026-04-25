import torch
from torch.utils.data import DataLoader
from calibration.temperature_scaling import apply_temperature


def margin_ts(model, unlabeled_dataset, n, device, T):
    # Margin Sampling on temperature-scaled probabilities
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)

    logits = model.get_logits(loader, device)
    probabilities = apply_temperature(logits, T)

    probs = torch.topk(probabilities, k=2, dim=1).values
    top1 = probs[:, 0]
    top2 = probs[:, 1]
    uncertainty = 1 - (top1 - top2)

    selected_indices = torch.argsort(uncertainty, descending=True)[:n]
    return selected_indices.tolist()
