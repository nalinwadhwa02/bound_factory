import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

from collections import defaultdict

from torchvision import datasets, transforms
from torchsummary import summary
# from tensorboardX import SummaryWriter

from bound_propagation import BoundModelFactory, HyperRectangle

from bound_factory import (
    BoundedModuleRegistry,
    get_input_bounds,
    get_bounded_module,
    backsubstitute_bounds,
    evaluate_bound_quality,
)

use_cuda = True

np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = f"data/saved_models/mnist_model.pt"


def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path=MODEL_PATH, device="cpu"):
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


def test_model_robustness_boundprop(model, eps, test_loader, device):
    """
    Evaluate robustness using bound-propagation library.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    eps : float
        Perturbation radius.
    test_loader : DataLoader
        MNIST test set loader.
    device : torch.device
        Device to run computations on.
    """
    # Build bound-propagation version of the model
    factory = BoundModelFactory()
    bound_model = factory.build(model.eval())

    total = 0
    correct = 0
    robustness_counts = defaultdict(int)

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # --- IBP bounds ---
        input_bounds = HyperRectangle.from_eps(images, eps)
        ibp_bounds = bound_model.ibp(input_bounds)

        # --- CROWN bounds ---
        crown_bounds = bound_model.crown(input_bounds).concretize()
        crown_ibp_bounds = bound_model.crown_ibp(input_bounds).concretize()

        # --- Compute nominal predictions ---
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += batch_size

        # --- Compute robustness based on bounds ---
        for bound_name, bound in [
            ("ibp", ibp_bounds),
            ("crown", crown_bounds),
            ("crown_ibp", crown_ibp_bounds),
        ]:
            lower = bound.lower
            upper = bound.upper

            # Robust if true class lower bound > max upper bound of others
            idx = torch.arange(batch_size, device=device)
            lower_true = lower[idx, labels]
            upper_masked = upper.clone()
            upper_masked[idx, labels] = float("-inf")
            robust_labels = (lower_true > upper_masked.max(dim=1)[0]).sum().item()
            robustness_counts[bound_name] += robust_labels

    print(f"Eps: {eps}")
    print(f"Total samples: {total}")
    print(f"Nominal Accuracy: {100 * correct / total:.2f}%")
    for name, count in robustness_counts.items():
        print(f"Robustness ({name}): {count} ({100 * count / total:.2f}%)")


def test_model_robustness(bounded_model, eps, test_loader):
    bounded_model.eval()
    device = next(bounded_model.model_module.parameters()).device
    with torch.no_grad():
        correct = 0
        robustness_count = defaultdict(int)
        collected_bounds = defaultdict(list)
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # base outputs
            outputs = bounded_model.forward(images)

            # perturbed_images
            perterbed_images_bounds = get_input_bounds(images, eps)

            # regular interval bounds
            output_bounds = bounded_model.interval_forward(perterbed_images_bounds)

            # poly interval bounds
            layer_poly_bounds = bounded_model.deeppoly_forward(perterbed_images_bounds)
            tight_bounds = backsubstitute_bounds(
                layer_poly_bounds, perterbed_images_bounds
            )

            # image correctness
            _, predicted = torch.max(outputs.data, 1)

            for bounds_label, bounds in [
                ("base_interval_bounds", output_bounds),
                ("poly_interval_bounds", layer_poly_bounds[-1][0]),
                ("poly_backsubst_bounds", tight_bounds),
            ]:
                collected_bounds[bounds_label].append(bounds)
                # check to see if any lower bound is larger than all the rest upper bounds
                idx = torch.arange(bounds.lower.size(0), device=device)
                lower_true = bounds.lower[idx, labels]
                upper_masked = bounds.upper.clone()
                upper_masked[idx, labels] = float("-inf")
                upper_masked_max, _ = upper_masked.max(dim=1)
                robust_labels = (lower_true > upper_masked_max).sum().item()
                robustness_count[bounds_label] += robust_labels

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Eps: {eps}")
        print(f"Total: {total}")
        print(f"Accuracy on images: {correct} ({100 * correct / total}%)")
        for bounds_label, rb in robustness_count.items():
            print(
                f"Bounds type: {bounds_label}\t robustness: {rb} ({100 * rb / total}%)"
            )


def main():
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 64

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # mean and std for MNIST
            transforms.Lambda(lambda x: x.view(-1)),  # flatten
        ]
    )

    train_dataset = datasets.MNIST(
        "mnist_data/",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        "mnist_data/",
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

    model = nn.Sequential(
        nn.Linear(28 * 28, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
        # nn.Softmax(dim=-1),
    )

    model = model.to(device)
    summary(model, (784,))
    model.train()

    if os.path.exists(MODEL_PATH):
        model = load_model(model, path=MODEL_PATH, device=device)
    else:
        # Train model if no saved weights
        train_model(model, num_epochs=10, train_loader=train_loader)
        save_model(model, path=MODEL_PATH)

    test_model(model, test_loader)

    ## you can use https://github.com/Zinoex/bound_propagation
    bounded_model = get_bounded_module(model)

    print(f"# Testing robustness accuracy from my bound_factory!!!")
    for eps in np.arange(0.01, 0.11, 0.01):
        test_model_robustness(bounded_model, eps, test_loader)
        print("-" * 50)

    print(f"# Testing robustness accuracy from bound_propogation library!!!")
    for eps in np.arange(0.01, 0.11, 0.01):
        test_model_robustness_boundprop(model, eps, test_loader, device)
        print("-" * 50)


if __name__ == "__main__":
    main()
