import torch
from torch.utils.data import DataLoader


def margin_mcd(model, unlabeled_dataset, n, device, T):
    # Margin Sampling on MC Dropout mean probabilities
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)

    probabilities = model.get_probabilities_mc(loader, device, T)

    probs = torch.topk(probabilities, k=2, dim=1).values
    top1 = probs[:, 0]
    top2 = probs[:, 1]
    uncertainty = 1 - (top1 - top2)

    selected_indices = torch.argsort(uncertainty, descending=True)[:n]
    return selected_indices.tolist()
