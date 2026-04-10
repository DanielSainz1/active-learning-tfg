import torch
from torch.utils.data import DataLoader

def entropy(model, unlabeled_dataset, n, device):
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)
    
    # 1. get probabilities for all images
    probabilities = model.get_probabilities(loader, device)
       
    # 2. entropy 
    uncertainty =  -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    
    # 4. select the n indices with highest uncertainty
    selected_indices = torch.argsort(uncertainty, descending=True)[:n]

    return selected_indices.tolist()