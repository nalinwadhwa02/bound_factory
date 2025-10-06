import torch
import torch.nn as nn

from bound_factory import (
    get_bounded_module,
    get_input_bounds,
    backsubstitute_bounds,
    evaluate_bound_quality,
)


def main():
    # Create model
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    model.eval()

    bounded_model = get_bounded_module(model)

    # Create input bounds
    x = torch.randn(2, 10)
    eps = 0.1
    input_bounds = get_input_bounds(x, eps)

    # Forward pass with DeepPoly
    layer_poly_bounds = bounded_model.deeppoly_forward(input_bounds)
    tight_bounds = backsubstitute_bounds(layer_poly_bounds, input_bounds)

    print(
        "Forward pass bounds (loose):",
        layer_poly_bounds[-1][0].lower,
        layer_poly_bounds[-1][0].upper,
    )
    print("Backsubstituted bounds (tight):", tight_bounds.lower, tight_bounds.upper)

    evaluate_bound_quality(layer_poly_bounds[-1][0], verbose=True)
    evaluate_bound_quality(tight_bounds, verbose=True)

    return


if __name__ == "__main__":
    main()
