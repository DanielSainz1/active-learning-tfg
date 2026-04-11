import torch
import torch.nn as nn

def evaluate(model, loader, device):
    # Calculates accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            # 1. move to device
            images = images.to(device)
            labels = labels.to(device)
            # 2. forward pass
            logits = model(images)
            # 3. prediction is the highest index
            predictions = torch.argmax(logits, dim=1)
            # 4. count correct predictions
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total

def train_epoch(model, loader, optimizer, criterion, device):
    # Trains the model during 1 epoch. Runs through the images in the loader 
    # one time and updates weight. Returns average loss for that epoch.
    model.train()
    total_loss = 0

    for images, labels in loader:
        # 1. move to device
        images = images.to(device)
        labels = labels.to(device)
        # 2. zero gradients
        optimizer.zero_grad()
        # 3. forward pass
        logits = model(images)
        # 4. calculate loss
        loss = criterion(logits, labels)
        # 5. backpropagation
        loss.backward()
        # 6. update weights
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def train(model, labeled_dataset, config, device):
    # Calls train_epoch repeatedly during config["train_epochs"] epochs,
    loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=config["train_batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["train_epochs"]):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print (f" Epoch {epoch+1}/{config['train_epochs']} | Loss: {loss:.4f}")
