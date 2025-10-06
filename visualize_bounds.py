import torch
import matplotlib.pyplot as plt
import numpy as np

from bound_factory import get_bounded_module, get_input_bounds
from models.model_utils import get_best_device
from models.mnist_simple_model import get_trained_mnist_model_with_train_and_test


def print_layer_bounds(layer_bounds_list, input_bounds=None, detailed=False):
    """
    Print bounds information for each layer in the network.

    Args:
        layer_bounds_list: List of (Bounds, PolyBounds) tuples from deeppoly_forward
        input_bounds: Optional Bounds object for the input layer
        detailed: If True, prints min/max/mean for lower and upper bounds separately
    """
    print("=" * 80)
    print("LAYER-BY-LAYER BOUNDS VISUALIZATION")
    print("=" * 80)

    if input_bounds is not None:
        print("\n[INPUT LAYER]")
        print(f"  Shape: {input_bounds.lower.shape}")
        width = input_bounds.upper - input_bounds.lower
        print(f"  Mean bound width: {width.mean().item():.6f}")
        print(f"  Std bound width:  {width.std().item():.6f}")
        if detailed:
            print(
                f"  Lower bounds - min: {input_bounds.lower.min().item():.6f}, "
                f"max: {input_bounds.lower.max().item():.6f}, "
                f"mean: {input_bounds.lower.mean().item():.6f}"
            )
            print(
                f"  Upper bounds - min: {input_bounds.upper.min().item():.6f}, "
                f"max: {input_bounds.upper.max().item():.6f}, "
                f"mean: {input_bounds.upper.mean().item():.6f}"
            )
        print("-" * 80)

    for i, (interval_bounds, poly_bounds) in enumerate(layer_bounds_list):
        print(f"\n[LAYER {i + 1}]")
        print(f"  Shape: {interval_bounds.lower.shape}")

        width = interval_bounds.upper - interval_bounds.lower
        print(f"  Mean bound width: {width.mean().item():.6f}")
        print(f"  Std bound width:  {width.std().item():.6f}")
        print(f"  Max bound width:  {width.max().item():.6f}")
        print(f"  Min bound width:  {width.min().item():.6f}")

        if detailed:
            print(
                f"  Lower bounds - min: {interval_bounds.lower.min().item():.6f}, "
                f"max: {interval_bounds.lower.max().item():.6f}, "
                f"mean: {interval_bounds.lower.mean().item():.6f}"
            )
            print(
                f"  Upper bounds - min: {interval_bounds.upper.min().item():.6f}, "
                f"max: {interval_bounds.upper.max().item():.6f}, "
                f"mean: {interval_bounds.upper.mean().item():.6f}"
            )

        print("-" * 80)

    print("\n" + "=" * 80)


def visualize_bound_widths(layer_bounds_list, input_bounds=None, figsize=(12, 6)):
    """
    Create visualization of bound width statistics across layers.

    Args:
        layer_bounds_list: List of (Bounds, PolyBounds) tuples from deeppoly_forward
        input_bounds: Optional Bounds object for the input layer
        figsize: Tuple (width, height) for the figure size

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Extract statistics for each layer
    layer_names = []
    mean_widths = []
    std_widths = []
    max_widths = []
    min_widths = []

    # Add input layer if provided
    if input_bounds is not None:
        layer_names.append("Input")
        width = input_bounds.upper - input_bounds.lower
        mean_widths.append(width.mean().item())
        std_widths.append(width.std().item())
        max_widths.append(width.max().item())
        min_widths.append(width.min().item())

    # Add each computational layer
    for i, (interval_bounds, _) in enumerate(layer_bounds_list):
        layer_names.append(f"Layer {i + 1}")
        width = interval_bounds.upper - interval_bounds.lower
        mean_widths.append(width.mean().item())
        std_widths.append(width.std().item())
        max_widths.append(width.max().item())
        min_widths.append(width.min().item())

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Bound Width Analysis Across Layers", fontsize=14, fontweight="bold")

    x = np.arange(len(layer_names))

    # Plot 1: Mean bound width
    axes[0, 0].plot(x, mean_widths, "o-", linewidth=2, markersize=8, color="steelblue")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Mean Bound Width")
    axes[0, 0].set_title("Mean Bound Width per Layer")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(layer_names, rotation=45, ha="right")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Std bound width
    axes[0, 1].plot(x, std_widths, "o-", linewidth=2, markersize=8, color="coral")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Std Bound Width")
    axes[0, 1].set_title("Standard Deviation of Bound Width per Layer")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(layer_names, rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Min and Max bound widths
    axes[1, 0].plot(
        x, max_widths, "o-", linewidth=2, markersize=8, color="darkgreen", label="Max"
    )
    axes[1, 0].plot(
        x, min_widths, "s-", linewidth=2, markersize=8, color="darkred", label="Min"
    )
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Bound Width")
    axes[1, 0].set_title("Min and Max Bound Width per Layer")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(layer_names, rotation=45, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: All statistics combined
    axes[1, 1].plot(
        x,
        mean_widths,
        "o-",
        linewidth=2,
        markersize=6,
        color="steelblue",
        label="Mean",
        alpha=0.8,
    )
    axes[1, 1].fill_between(
        x,
        np.array(mean_widths) - np.array(std_widths),
        np.array(mean_widths) + np.array(std_widths),
        alpha=0.3,
        color="steelblue",
        label="±1 Std",
    )
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Bound Width")
    axes[1, 1].set_title("Mean ± Std Bound Width per Layer")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(layer_names, rotation=45, ha="right")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def compare_bounds_methods(layer_bounds_dict, input_bounds=None, figsize=(12, 8)):
    """
    Compare bounds from different methods (e.g., IBP vs DeepPoly).

    Args:
        layer_bounds_dict: Dict mapping method names to layer_bounds_list
                          e.g., {'IBP': ibp_bounds, 'DeepPoly': dp_bounds}
        input_bounds: Optional Bounds object for the input layer
        figsize: Tuple (width, height) for the figure size

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Comparison of Bound Propagation Methods", fontsize=14, fontweight="bold"
    )

    colors = ["steelblue", "coral", "darkgreen", "purple", "orange"]

    for method_idx, (method_name, layer_bounds_list) in enumerate(
        layer_bounds_dict.items()
    ):
        color = colors[method_idx % len(colors)]

        # Extract statistics
        layer_names = []
        mean_widths = []
        std_widths = []

        if input_bounds is not None:
            layer_names.append("Input")
            width = input_bounds.upper - input_bounds.lower
            mean_widths.append(width.mean().item())
            std_widths.append(width.std().item())

        for i, (interval_bounds, _) in enumerate(layer_bounds_list):
            layer_names.append(f"L{i + 1}")
            width = interval_bounds.upper - interval_bounds.lower
            mean_widths.append(width.mean().item())
            std_widths.append(width.std().item())

        x = np.arange(len(layer_names))

        # Plot mean widths
        axes[0, 0].plot(
            x,
            mean_widths,
            "o-",
            linewidth=2,
            markersize=6,
            color=color,
            label=method_name,
            alpha=0.8,
        )

        # Plot std widths
        axes[0, 1].plot(
            x,
            std_widths,
            "o-",
            linewidth=2,
            markersize=6,
            color=color,
            label=method_name,
            alpha=0.8,
        )

        # Plot mean with std bands
        axes[1, 0].plot(
            x,
            mean_widths,
            "o-",
            linewidth=2,
            markersize=6,
            color=color,
            label=method_name,
            alpha=0.8,
        )
        axes[1, 0].fill_between(
            x,
            np.array(mean_widths) - np.array(std_widths),
            np.array(mean_widths) + np.array(std_widths),
            alpha=0.2,
            color=color,
        )

        # Plot cumulative bound growth
        cumulative_width = np.cumsum(mean_widths)
        axes[1, 1].plot(
            x,
            cumulative_width,
            "o-",
            linewidth=2,
            markersize=6,
            color=color,
            label=method_name,
            alpha=0.8,
        )

    # Configure subplots
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Mean Bound Width")
    axes[0, 0].set_title("Mean Bound Width")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(layer_names, rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Std Bound Width")
    axes[0, 1].set_title("Std Bound Width")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(layer_names, rotation=45, ha="right")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Bound Width")
    axes[1, 0].set_title("Mean ± Std Bound Width")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(layer_names, rotation=45, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Cumulative Bound Width")
    axes[1, 1].set_title("Cumulative Bound Growth")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(layer_names, rotation=45, ha="right")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def get_bounds_summary(layer_bounds_list, input_bounds=None):
    """
    Get a summary dictionary of bounds statistics for programmatic access.

    Args:
        layer_bounds_list: List of (Bounds, PolyBounds) tuples from deeppoly_forward
        input_bounds: Optional Bounds object for the input layer

    Returns:
        Dictionary with statistics for each layer
    """
    summary = {}

    if input_bounds is not None:
        width = input_bounds.upper - input_bounds.lower
        summary["input"] = {
            "shape": tuple(input_bounds.lower.shape),
            "mean_width": width.mean().item(),
            "std_width": width.std().item(),
            "max_width": width.max().item(),
            "min_width": width.min().item(),
            "lower_mean": input_bounds.lower.mean().item(),
            "upper_mean": input_bounds.upper.mean().item(),
        }

    for i, (interval_bounds, _) in enumerate(layer_bounds_list):
        width = interval_bounds.upper - interval_bounds.lower
        summary[f"layer_{i + 1}"] = {
            "shape": tuple(interval_bounds.lower.shape),
            "mean_width": width.mean().item(),
            "std_width": width.std().item(),
            "max_width": width.max().item(),
            "min_width": width.min().item(),
            "lower_mean": interval_bounds.lower.mean().item(),
            "upper_mean": interval_bounds.upper.mean().item(),
        }

    return summary


if __name__ == "__main__":
    device = get_best_device()

    model, train_loader, test_loader = get_trained_mnist_model_with_train_and_test(
        device=device
    )
    ## you can use https://github.com/Zinoex/bound_propagation
    bounded_model = get_bounded_module(model)

    bounded_model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)

        input_bounds = get_input_bounds(images, 0.01)

        layer_bounds = bounded_model.deeppoly_forward(input_bounds)

        # Print text summary
        print_layer_bounds(layer_bounds, input_bounds, detailed=True)

        # Visualize with graphs
        fig, axes = visualize_bound_widths(layer_bounds, input_bounds)
        plt.show()

        # # Compare methods
        # fig, axes = compare_bounds_methods(
        #     {"IBP": ibp_bounds, "DeepPoly": deeppoly_bounds}, input_bounds
        # )
        # plt.show()

        # Get summary dict
        summary = get_bounds_summary(layer_bounds, input_bounds)
