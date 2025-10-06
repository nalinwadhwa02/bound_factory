import torch
import torch.nn as nn
import torch.nn.functional as F

## you can use https://github.com/Zinoex/bound_propagation


class PolyBounds:
    def __init__(self, lower_coef, upper_coef, lower_bias, upper_bias):
        self.lower_coef = lower_coef
        self.upper_coef = upper_coef
        self.lower_bias = lower_bias
        self.upper_bias = upper_bias


class Bounds:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper


class BoundedModule(nn.Module):
    def __init__(self, model_module):
        super().__init__()
        self.model_module = model_module

    def forward(self, x):
        return self.model_module(x)

    def interval_forward(self, bounds: Bounds) -> Bounds:
        raise NotImplementedError

    def deeppoly_forward(self, bounds: Bounds) -> list[tuple[Bounds, PolyBounds]]:
        """
        Returns a list of (Bounds, PolyBounds) tuples representing the layer-wise bounds.
        Each element corresponds to a computational layer in the module.
        """
        raise NotImplementedError


class BoundedLinear(BoundedModule):
    def interval_forward(self, bounds):
        weights = self.model_module.weight
        bias = self.model_module.bias

        weight_pos = torch.clamp(weights, min=0)  # tensor of positive weights | 0
        weight_neg = torch.clamp(weights, max=0)  # tensor of negative weights | 0

        # Interval arithmetic for y = Wx + b
        lower = F.linear(bounds.lower, weight_pos, bias) + F.linear(
            bounds.upper, weight_neg, None
        )
        upper = F.linear(bounds.upper, weight_pos, bias) + F.linear(
            bounds.lower, weight_neg, None
        )

        return Bounds(lower, upper)

    def deeppoly_forward(self, bounds):
        interval_bounds = self.interval_forward(bounds)

        batch_size = bounds.lower.size(0)
        device = bounds.lower.device

        lower_coef = (
            self.model_module.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        )
        lower_bias = (
            self.model_module.bias.unsqueeze(0).expand(batch_size, -1).to(device)
            if self.model_module.bias is not None
            else torch.zeros(
                batch_size, self.model_module.weight.shape[0], device=device
            )
        )
        upper_coef = lower_coef  # same as lower (exact)
        upper_bias = lower_bias

        poly_bounds = PolyBounds(lower_coef, upper_coef, lower_bias, upper_bias)

        # Return single-element list containing this layer's bounds
        return [(interval_bounds, poly_bounds)]


class BoundedReLU(BoundedModule):
    def interval_forward(self, bounds):
        lower = torch.clamp(bounds.lower, min=0)
        upper = torch.clamp(bounds.upper, min=0)

        return Bounds(lower, upper)

    def deeppoly_forward(self, bounds):
        l = bounds.lower  # (batch, neurons)
        u = bounds.upper  # (batch, neurons)

        batch_size, num_neurons = l.shape
        device = l.device

        # Concrete bounds
        concrete_lower = torch.clamp(l, min=0)
        concrete_upper = torch.clamp(u, min=0)

        # Create identity matrix base
        identity = (
            torch.eye(num_neurons, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # Initialize coefficients
        lower_coef = torch.zeros(batch_size, num_neurons, num_neurons, device=device)
        lower_bias = torch.zeros(batch_size, num_neurons, device=device)
        upper_coef = torch.zeros(batch_size, num_neurons, num_neurons, device=device)
        upper_bias = torch.zeros(batch_size, num_neurons, device=device)

        # Case 1: Always active (l >= 0) -> y = x
        active_mask = l >= 0  # (batch, neurons)
        lower_coef = torch.where(active_mask.unsqueeze(-1), identity, lower_coef)
        upper_coef = torch.where(active_mask.unsqueeze(-1), identity, upper_coef)

        # Case 2: Always inactive (u <= 0) -> y = 0
        # Coefficients already zero, nothing to do

        # Case 3: Crossing (l < 0 < u)
        crossing_mask = (l < 0) & (u > 0)  # (batch, neurons)

        # Compute lambda and mu
        epsilon = 1e-8
        lambda_upper = torch.where(
            crossing_mask, u / (u - l + epsilon), torch.zeros_like(u)
        )
        mu_upper = torch.where(crossing_mask, -lambda_upper * l, torch.zeros_like(l))

        # Upper bound for crossing: y ≤ λ*x + μ
        # Set diagonal elements of upper_coef to lambda
        lambda_diag = torch.diag_embed(lambda_upper)  # (batch, neurons, neurons)
        upper_coef = torch.where(crossing_mask.unsqueeze(-1), lambda_diag, upper_coef)
        upper_bias = torch.where(crossing_mask, mu_upper, upper_bias)

        # Lower bound for crossing: y ≥ 0 (coefficients stay zero)

        poly_bounds = PolyBounds(lower_coef, upper_coef, lower_bias, upper_bias)

        # Return single-element list containing this layer's bounds
        return [(Bounds(concrete_lower, concrete_upper), poly_bounds)]


class BoundedSoftmax(BoundedModule):
    def interval_forward(self, bounds: Bounds) -> Bounds:
        """
        Tighter interval bounds for softmax outputs given input logit bounds.

        For each class i (vectorized across batch):
            upper_i = exp(u_i) / ( exp(u_i) + sum_{j != i} exp(l_j) )
            lower_i = exp(l_i) / ( exp(l_i) + sum_{j != i} exp(u_j) )

        Returns:
            Bounds(lower=(B,C), upper=(B,C))
        """
        l = bounds.lower  # (B, C) lower bound on logits
        u = bounds.upper  # (B, C) upper bound on logits
        eps = 1e-12
        device = l.device

        # Exponentiate endpoints
        exp_l = torch.exp(l)  # (B, C)
        exp_u = torch.exp(u)  # (B, C)

        # Sum across classes (keepdim so we can subtract the diagonal easily)
        sum_exp_l = exp_l.sum(dim=1, keepdim=True)  # (B, 1)
        sum_exp_u = exp_u.sum(dim=1, keepdim=True)  # (B, 1)

        # For each i we need sum_{j != i} exp(l_j) = sum_exp_l - exp_l[:, i]
        # Vectorized: denom_upper = exp_u + (sum_exp_l - exp_l)
        denom_upper = exp_u + (sum_exp_l - exp_l)  # (B, C)
        # Upper bound: maximize numerator (use u_i) and minimize others (use l_j for j != i)
        upper = exp_u / (denom_upper + eps)

        # For lower: denom_lower = exp_l + (sum_exp_u - exp_u)
        denom_lower = exp_l + (sum_exp_u - exp_u)  # (B, C)
        # Lower bound: minimize numerator (use l_i) and maximize others (use u_j for j != i)
        lower = exp_l / (denom_lower + eps)

        # Numerical safety: enforce bounds inside [0,1]
        lower = torch.clamp(lower, min=0.0, max=1.0)
        upper = torch.clamp(upper, min=0.0, max=1.0)

        return Bounds(lower, upper)

    def deeppoly_forward(self, bounds: Bounds):
        """
        Vectorized DeepPoly relaxation for softmax with:
          - exp upper bound: secant over [l_d, u_d]
          - exp lower bound: tangent at midpoint t = (l_d + u_d) / 2
        Returns: [(interval_bounds, poly_bounds)]
        - interval_bounds: Bounds(lower=(B,C), upper=(B,C))
        - poly_bounds: PolyBounds(lower_coef=(B,C,C), upper_coef=(B,C,C),
                                 lower_bias=(B,C), upper_bias=(B,C))
        """
        l = bounds.lower  # (B, C)
        u = bounds.upper  # (B, C)
        B, C = l.shape
        device = l.device
        eps = 1e-12

        # --- Pairwise differences d_{i,j} = x_j - x_i as tensors (B, C_out=i, C_in=j) ---
        # Use broadcasting: l_j = l.unsqueeze(1) -> (B,1,C) (j along last dim)
        #                   u_i = u.unsqueeze(2) -> (B,C,1) (i along second dim)
        l_j = l.unsqueeze(1)  # (B,1,C)
        u_j = u.unsqueeze(1)  # (B,1,C)
        l_i = l.unsqueeze(2)  # (B,C,1)
        u_i = u.unsqueeze(2)  # (B,C,1)

        l_d = l_j - u_i  # (B, C, C)  entry (b,i,j) = l_j(b,j) - u_i(b,i)
        u_d = u_j - l_i  # (B, C, C)  entry (b,i,j) = u_j(b,j) - l_i(b,i)

        # We will exclude diagonal (j==i) from S = sum_{j != i} exp(d).
        diag_idx = torch.arange(C, device=device)
        # Set diag bounds to zero temporarily so exp(diag) is not treated as a variable term.
        l_d[:, diag_idx, diag_idx] = 0.0
        u_d[:, diag_idx, diag_idx] = 0.0

        # --- Exponential relaxations ---
        # Precompute exp at endpoints
        exp_l = torch.exp(l_d)  # (B,C,C)
        exp_u = torch.exp(u_d)  # (B,C,C)

        # Upper bound: secant line over [l_d, u_d]
        denom = u_d - l_d
        a_u = (exp_u - exp_l) / (denom + eps)  # (B,C,C)
        # For extremely small denom, fallback slope ~ derivative (exp at midpoint-ish) via exp_u
        small_mask = denom.abs() < 1e-9
        if small_mask.any():
            a_u = torch.where(small_mask, exp_u, a_u)
        b_u = exp_l - a_u * l_d  # bias for secant (so exp(d) <= a_u * d + b_u)

        # Lower bound: tangent at midpoint t = (l_d + u_d) / 2
        t = 0.5 * (l_d + u_d)  # midpoint (B,C,C)
        a_l = torch.exp(t)  # slope = exp(t) (derivative at midpoint)
        b_l = torch.exp(t) - a_l * t  # bias so exp(d) >= a_l * d + b_l (tangent)

        # Zero-out diagonal contributions (excluded from S)
        a_u[:, diag_idx, diag_idx] = 0.0
        b_u[:, diag_idx, diag_idx] = 0.0
        a_l[:, diag_idx, diag_idx] = 0.0
        b_l[:, diag_idx, diag_idx] = 0.0
        exp_l[:, diag_idx, diag_idx] = 0.0
        exp_u[:, diag_idx, diag_idx] = 0.0

        # --- Build S_i = sum_{j != i} exp_{i,j} bounds and linear forms ---
        # Interval sums for S
        S_l = exp_l.sum(dim=2)  # (B, C)  sum over j
        S_u = exp_u.sum(dim=2)  # (B, C)

        # Each s_{i,j} ≈ a * d + b where d = x_j - x_i -> contributes:
        #   +a to coefficient of x_j (column j), and -a to coefficient of x_i (column i)
        # Build coefficient tensors for S (B, C_out, C_in)
        coef_S_upper = a_u.clone()  # currently +a placed at (i,j)
        coef_S_lower = a_l.clone()

        # subtract the sum over j at the input-position i to represent the "-a * x_i" term
        sum_a_u_over_j = a_u.sum(dim=2)  # (B, C)
        sum_a_l_over_j = a_l.sum(dim=2)  # (B, C)

        # subtract in the diagonal positions (i as input index)
        coef_S_upper[:, diag_idx, diag_idx] -= sum_a_u_over_j
        coef_S_lower[:, diag_idx, diag_idx] -= sum_a_l_over_j

        # bias terms for S
        bias_S_upper = b_u.sum(dim=2)  # (B, C)
        bias_S_lower = b_l.sum(dim=2)  # (B, C)

        # --- Relax g(S) = 1 / (1 + S) on [S_l, S_u] ---
        # g is convex and decreasing on S >= 0.
        g_at_S_l = 1.0 / (1.0 + S_l)  # (B, C)
        g_at_S_u = 1.0 / (1.0 + S_u)  # (B, C)
        # Upper via secant (convex -> secant is above)
        denom_g = S_u - S_l
        m_u = (g_at_S_u - g_at_S_l) / (denom_g + eps)  # (B, C)
        tiny_mask_g = denom_g.abs() < 1e-9
        if tiny_mask_g.any():
            # derivative at S_u fallback
            deriv_Su = -1.0 / ((1.0 + S_u) ** 2)
            m_u = torch.where(tiny_mask_g, deriv_Su, m_u)
        c_u = g_at_S_l - m_u * S_l  # (B, C)

        # Lower via tangent at S0 = S_u (common choice). You may also choose midpoint here.
        S0 = S_u
        m_l = -1.0 / ((1.0 + S0) ** 2)  # (B, C)
        c_l = (1.0 / (1.0 + S0)) - m_l * S0  # (B, C)

        # --- Compose: final linear coefficients on original logits ---
        # final_coef = m * coef_S (broadcast m over input dim)
        m_u_exp = m_u.unsqueeze(2)  # (B, C, 1)
        m_l_exp = m_l.unsqueeze(2)  # (B, C, 1)

        final_upper_coef = m_u_exp * coef_S_upper  # (B, C, C)
        final_lower_coef = m_l_exp * coef_S_lower  # (B, C, C)

        final_upper_bias = m_u * bias_S_upper + c_u  # (B, C)
        final_lower_bias = m_l * bias_S_lower + c_l  # (B, C)

        # --- Interval softmax per-output (remember g is decreasing) ---
        soft_lower = g_at_S_u  # (B, C)
        soft_upper = g_at_S_l  # (B, C)
        # clamp into [0,1] numerically
        soft_lower = torch.clamp(soft_lower, min=0.0, max=1.0)
        soft_upper = torch.clamp(soft_upper, min=0.0, max=1.0)
        interval_bounds = Bounds(soft_lower, soft_upper)

        # Build PolyBounds in the shape expected by backsubstitution
        poly_bounds = PolyBounds(
            lower_coef=final_lower_coef,  # (B, C_out, C_in)
            upper_coef=final_upper_coef,
            lower_bias=final_lower_bias,  # (B, C_out)
            upper_bias=final_upper_bias,
        )

        return [(interval_bounds, poly_bounds)]


class BoundedFlatten(BoundedModule):
    def interval_forward(self, bounds):
        batch_size = bounds.lower.size(0)
        lower = bounds.lower.view(batch_size, -1)
        upper = bounds.upper.view(batch_size, -1)
        return Bounds(lower, upper)

    def deeppoly_forward(self, bounds: Bounds) -> list[tuple[Bounds, PolyBounds]]:
        """
        DeepPoly forward pass for Flatten layer.
        Since flatten is a linear reshape, we only reshape the affine forms
        (no relaxation, no approximation).

        Returns:
            list[(Bounds, PolyBounds)] — single-element list representing
            the DeepPoly bounds for this layer.
        """
        batch_size = bounds.lower.size(0)
        device = bounds.lower.device

        # Flatten interval bounds
        lower = bounds.lower.view(batch_size, -1)
        upper = bounds.upper.view(batch_size, -1)
        interval_bounds = Bounds(lower, upper)

        # Create corresponding linear forms (identity mapping)
        # y = I * x, so coefficient = Identity, bias = 0
        input_dim = lower.numel() // batch_size
        identity = (
            torch.eye(input_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        )

        lower_coef = identity.clone()
        upper_coef = identity.clone()
        lower_bias = torch.zeros(batch_size, input_dim, device=device)
        upper_bias = torch.zeros(batch_size, input_dim, device=device)

        poly_bounds = PolyBounds(lower_coef, upper_coef, lower_bias, upper_bias)

        # Return as [(Bounds, PolyBounds)] list, consistent with framework
        return [(interval_bounds, poly_bounds)]


class BoundedMonotonic(BoundedModule):
    def interval_forward(self, bounds):
        # assuming that the generic module implements some sort of monotonic function
        lower = self.model_module.forward(bounds.lower)
        upper = self.model_module.forward(bounds.upper)
        return Bounds(lower, upper)


class BoundedSequential(BoundedModule):
    def interval_forward(self, bounds):
        self.model_module.eval()

        with torch.no_grad():
            bounded_modules = [
                get_bounded_module(child) for child in self.model_module.children()
            ]

            for bounded_module in bounded_modules:
                bounds = bounded_module.interval_forward(bounds)

        return bounds

    def deeppoly_forward(self, bounds):
        self.model_module.eval()

        with torch.no_grad():
            bounded_modules = [
                get_bounded_module(child) for child in self.model_module.children()
            ]

            # Collect all layer bounds from all modules
            all_layer_bounds = []

            current_bounds = bounds
            for bounded_module in bounded_modules:
                layer_bounds_list = bounded_module.deeppoly_forward(current_bounds)

                # Add all bounds from this module to our collection
                all_layer_bounds.extend(layer_bounds_list)

                # Update current bounds to the last layer's output
                current_bounds = layer_bounds_list[-1][0]

        # Return all collected layer bounds
        return all_layer_bounds


def backsubstitute_bounds(layer_bounds_list, input_bounds):
    """
    Compute tight bounds by backsubstituting through all layers.

    Args:
        layer_bounds_list: List of tuples (interval_bounds, poly_bounds) for each layer
                          layer_bounds_list[0] = first computational layer
                          layer_bounds_list[-1] = output layer
        input_bounds: Bounds object for the input layer

    Returns:
        Bounds: Tight concrete bounds on the output
    """
    with torch.no_grad():
        # Start from the last layer
        _, target_poly = layer_bounds_list[-1]

        current_lower_coef = (
            target_poly.lower_coef
        )  # (batch, out_neurons, prev_neurons)
        current_lower_bias = target_poly.lower_bias  # (batch, out_neurons)
        current_upper_coef = target_poly.upper_coef
        current_upper_bias = target_poly.upper_bias

        # Backsubstitute through each layer (from second-to-last down to first)
        for i in range(len(layer_bounds_list) - 2, -1, -1):
            _, prev_poly = layer_bounds_list[i]

            # For lower bound: substitute previous layer's symbolic forms
            # Positive coefficients use lower bound, negative use upper bound
            lower_coef_pos = torch.clamp(
                current_lower_coef, min=0
            )  # (batch, out, prev)
            lower_coef_neg = torch.clamp(current_lower_coef, max=0)

            # Compose: new_coef = current_coef @ prev_coef
            new_lower_coef = torch.bmm(
                lower_coef_pos, prev_poly.lower_coef
            ) + torch.bmm(lower_coef_neg, prev_poly.upper_coef)

            # Compose bias: new_bias = current_coef @ prev_bias + current_bias
            new_lower_bias = (
                torch.bmm(lower_coef_pos, prev_poly.lower_bias.unsqueeze(-1)).squeeze(
                    -1
                )
                + torch.bmm(lower_coef_neg, prev_poly.upper_bias.unsqueeze(-1)).squeeze(
                    -1
                )
                + current_lower_bias
            )

            # For upper bound: positive coefficients use upper bound, negative use lower bound
            upper_coef_pos = torch.clamp(current_upper_coef, min=0)
            upper_coef_neg = torch.clamp(current_upper_coef, max=0)

            new_upper_coef = torch.bmm(
                upper_coef_pos, prev_poly.upper_coef
            ) + torch.bmm(upper_coef_neg, prev_poly.lower_coef)

            new_upper_bias = (
                torch.bmm(upper_coef_pos, prev_poly.upper_bias.unsqueeze(-1)).squeeze(
                    -1
                )
                + torch.bmm(upper_coef_neg, prev_poly.lower_bias.unsqueeze(-1)).squeeze(
                    -1
                )
                + current_upper_bias
            )

            current_lower_coef = new_lower_coef
            current_lower_bias = new_lower_bias
            current_upper_coef = new_upper_coef
            current_upper_bias = new_upper_bias

        # Now we're in terms of input - compute concrete bounds
        input_lower = input_bounds.lower  # (batch, input_dim)
        input_upper = input_bounds.upper

        # Compute final lower bound
        lower_coef_pos = torch.clamp(current_lower_coef, min=0)
        lower_coef_neg = torch.clamp(current_lower_coef, max=0)
        final_lower = (
            torch.bmm(lower_coef_pos, input_lower.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(lower_coef_neg, input_upper.unsqueeze(-1)).squeeze(-1)
            + current_lower_bias
        )

        # Compute final upper bound
        upper_coef_pos = torch.clamp(current_upper_coef, min=0)
        upper_coef_neg = torch.clamp(current_upper_coef, max=0)
        final_upper = (
            torch.bmm(upper_coef_pos, input_upper.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(upper_coef_neg, input_lower.unsqueeze(-1)).squeeze(-1)
            + current_upper_bias
        )

        return Bounds(final_lower, final_upper)


BoundedModuleRegistry = {
    nn.Linear: BoundedLinear,
    nn.ReLU: BoundedReLU,
    nn.Sequential: BoundedSequential,
    nn.Flatten: BoundedFlatten,
    nn.Softmax: BoundedSoftmax,
}


def get_bounds_via_ibp(model, input_bounds) -> Bounds:
    return model.interval_forward(input_bounds)


def get_bounds_via_deeppoly(model, input_bounds) -> Bounds:
    layer_bounds_list = model.deeppoly_forward(input_bounds)
    tight_interval_bounds = backsubstitute_bounds(layer_bounds_list, input_bounds)
    return tight_interval_bounds


CertificationMethodsRegistry = {
    "ibp": get_bounds_via_ibp,
    "deeppoly": get_bounds_via_deeppoly,
}


def get_bounds_via_method(model: BoundedModule, method, input_bounds) -> Bounds:
    if method not in CertificationMethodsRegistry:
        raise Exception(
            f"Method {method} not found in the CertificationMethodsRegistry"
        )
    return CertificationMethodsRegistry[method](model, input_bounds)


def get_bounded_module(module: nn.Module) -> BoundedModule:
    module_type = type(module)
    if module_type not in BoundedModuleRegistry:
        raise Exception(f"Module {module_type} not found in BoundedModuleRegistry")
    return BoundedModuleRegistry[module_type](module)


def get_input_bounds(x, eps):
    lower = torch.clamp(x - eps, min=0, max=1)
    upper = torch.clamp(x + eps, min=0, max=1)
    return Bounds(lower, upper)


def evaluate_bound_quality(bounds: Bounds, verbose=False):
    """Comprehensive bound quality metrics"""
    width = bounds.upper - bounds.lower

    metrics = {
        "mean_width": width.mean().item(),
        "max_width": width.max().item(),
        "min_width": width.min().item(),
        "std_width": width.std().item(),
    }

    if verbose:
        print(f"Bound Quality Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    return metrics
