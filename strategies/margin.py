import torch
from torch.utils.data import DataLoader

def margin_sampling(model, unlabeled_dataset, n, device):
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)
    
    # 1. get probabilities for all images
    probabilities = model.get_probabilities(loader, device)

    # 2. get the second highest probability for each image
    probs = torch.topk(probabilities, k=2, dim=1).values
    top1 = probs[:, 0]
    top2 = probs[:, 1]
    
    # 3. uncertainty = 1 - max_prob
    uncertainty = 1 - (top1 - top2)
    
    # 4. select the n indices with highest uncertainty
    selected_indices = torch.argsort(uncertainty, descending=True)[:n]

    return selected_indices.tolist()