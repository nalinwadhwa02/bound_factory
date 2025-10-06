from bound_propagation import BoundModelFactory, HyperRectangle
import torch
from collections import defaultdict
from bound_factory import evaluate_bound_quality


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
        input_bounds = HyperRectangle.from_eps(images, eps / 0.3081)
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
