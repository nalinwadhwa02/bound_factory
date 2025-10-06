from pathlib import Path
import torch
import os
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary

from models.model_utils import train_model, test_model, load_model, save_model

MODEL_PATH = Path(f"saved_model_checkpoints/saved_models/mnist_model.pt")
MNIST_DATASET = Path(f"data/mnist_data/")


class MNISTMODEL(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(28 * 28, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            # nn.ReLU(),
            # nn.Linear(in_features=100, out_features=10),
            # nn.Softmax(dim=-1),
        )


def get_trained_mnist_model_with_train_and_test(
    dataset_path=MNIST_DATASET, device="cpu"
):
    batch_size = 64
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # mean and std for MNIST
            transforms.Lambda(lambda x: x.view(-1)),  # flatten
        ]
    )

    train_dataset = datasets.MNIST(
        dataset_path,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        dataset_path,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = MNISTMODEL()

    model = model.to(device)

    # summary(model, (784,), device=str(device))

    model.train()

    if os.path.exists(MODEL_PATH):
        model = load_model(model, path=MODEL_PATH, device=device)
    else:
        # Train model if no saved weights
        train_model(model, num_epochs=10, train_loader=train_loader)
        save_model(model, path=MODEL_PATH)

    test_model(model, test_loader)

    return model, train_loader, test_loader
