import torch
from torch.utils.data import DataLoader

def least_confidence(model, unlabeled_dataset, n, device):
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)
    
    # 1. get probabilities for all images
    probabilities = model.get_probabilities(loader, device)
    
    # 2. get the highest probability for each image
    max_probs = torch.max(probabilities, dim=1).values
    
    # 3. uncertainty = 1 - max_prob
    uncertainty = 1 - max_probs
    
    # 4. select the n indices with highest uncertainty
    selected_indices = torch.argsort(uncertainty, descending=True)[:n]

    return selected_indices.tolist()
