import os
import torch
import torch.optim as optim
import torch.nn.functional as F


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model


def train_model(model, num_epochs, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.3f}"
        )


def test_model(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy on images: {100 * correct / total}")
