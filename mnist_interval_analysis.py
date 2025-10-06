import os
import torch
from torch._dynamo.config import verbose
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

from models.mnist_simple_model import (
    MNISTMODEL,
    get_trained_mnist_model_with_train_and_test,
)

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
    collected_bounds = defaultdict(list)

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
            collected_bounds[bound_name].append(bound)
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

    for bounds_label, rb in robustness_counts.items():
        mean_bound_width = evaluate_bound_quality(collected_bounds[bounds_label])
        print(
            f"Bounds type: {bounds_label}\t robustness: {rb} ({100 * rb / total}%)\t mean_bound_width: {mean_bound_width}"
        )
    return collected_bounds


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
            mean_bound_width = evaluate_bound_quality(collected_bounds[bounds_label])
            print(
                f"Bounds type: {bounds_label}\t robustness: {rb} ({100 * rb / total}%)\t mean_bound_width: {mean_bound_width}"
            )
        return collected_bounds


def main():
    device = torch.device("cuda" if use_cuda else "cpu")

    model, train_loader, test_loader = get_trained_mnist_model_with_train_and_test(
        device=device
    )
    ## you can use https://github.com/Zinoex/bound_propagation
    bounded_model = get_bounded_module(model)

    print(f"# Testing robustness accuracy from my bound_factory!!!")
    for eps in np.arange(0.01, 0.11, 0.01):
        bf_bounds = test_model_robustness(bounded_model, eps, test_loader)
        print("-" * 50)

    print(f"# Testing robustness accuracy from bound_propogation library!!!")
    for eps in np.arange(0.01, 0.11, 0.01):
        bf_bounds = test_model_robustness_boundprop(model, eps, test_loader, device)
        print("-" * 50)


if __name__ == "__main__":
    main()
