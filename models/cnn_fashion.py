import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN with two convolutional blocks (32 and 64 filters),
# a fully connected layer with ReLU, dropout (p=0.25),
# and an output layer returning logits (no softmax)
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # passes an image through the whole network and returns logits
        # Convolutional blocks
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten: (batch, 64, 7, 7) -> (batch, 3136)
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_probabilities(self, loader, device, mc_dropout=False):
        # Run inference and return softmax probabilities for all samples
        # When mc_dropout=True, keeps dropout active for stochastic forward passes (BALD)
        if not mc_dropout:
            self.eval()
        all_probs = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                probs = torch.softmax(self.forward(images), dim=1)
                all_probs.append(probs.cpu())
        return torch.cat(all_probs, dim=0)

    def get_logits(self, loader, device):
        # Return raw logits (no softmax). Used by Temperature Scaling.
        self.eval()
        all_logits = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                all_logits.append(self.forward(images).cpu())
        return torch.cat(all_logits, dim=0)

    def get_probabilities_mc(self, loader, device, T):
        # T stochastic forward passes with dropout active; returns the mean of the T softmax distributions.
        self.eval()
        self.enable_dropout()
        all_passes = []
        for _ in range(T):
            all_passes.append(self.get_probabilities(loader, device, mc_dropout=True))
        return torch.stack(all_passes).mean(dim=0)

    def enable_dropout(self):
      # Enable dropout during inference (used by BALD for MC Dropout)
      for module in self.modules():
          if isinstance(module, torch.nn.Dropout):
              module.train()


def create_model():
    return FashionMNISTCNN()

