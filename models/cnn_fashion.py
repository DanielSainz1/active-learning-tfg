import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        self.fc2 = nn.Linear(128, 10)



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))                               
        x = x.view(x.size(0), -1)           
        x = F.relu(self.fc1(x))                               
        x = self.dropout(x)                 
        x = self.fc2(x)
        return x

def crear_modelo():
    return FashionMNISTCNN()
