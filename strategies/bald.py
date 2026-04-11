import torch
from torch.utils.data import DataLoader


def bald(model, unlabeled_dataset, n, device, T):
    # BALD: Selects samples where the model disagrees most across MC Dropout passes
    # High disagreement between passes indicates high epimistic uncertainty
    loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False)

    model.eval()
    model.enable_dropout()

    # T forward passes → list of T matrices (n_samples, n_classes)
    all_passes = []
    for _ in range(T):
        probs = model.get_probabilities(loader, device)
        all_passes.append(probs)

    # all_passes is a list of T tensors → pile them in shape (T, n_samples, n_classes)
    all_passes = torch.stack(all_passes)  # (T, N, C)

    mean_probs = all_passes.mean(dim=0)
     #Entropy per pass
    entropy_per_pass =  -torch.sum(all_passes * torch.log(all_passes + 1e-10), dim=2)
   
    #average
    mean_entropy = entropy_per_pass.mean(dim=0)
 
 
    entropy_of_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)  # (N,)
    #score = entropy of mean - mean entropy
    score = entropy_of_mean - mean_entropy
    selected_indices = torch.argsort(score, descending=True)[:n]
    return selected_indices.tolist()