import torch
from typing import List, Tuple

from torch.types import Device
from torch.xpu import device
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


class GenerationTokenNode:
    def __init__(self, layer: int, token_id: int, probability: float, previous_tokens):
        self.layer = layer
        self.probability = probability
        self.previous_tokens = previous_tokens
        self.token_id = token_id
        self.children = []


class GenerationTree:
    def __init__(self, root: List[GenerationTokenNode]):
        self.root_children = root
        self.current_layer = 0

    def get_all_nodes(self):
        """Get all nodes in the tree organized by layer"""
        layers = {}

        def traverse(node):
            if node.layer not in layers:
                layers[node.layer] = []
            layers[node.layer].append(node)
            for child in node.children:
                traverse(child)

        for root_node in self.root_children:
            traverse(root_node)

        return layers

    def count_nodes(self):
        """Count total nodes in tree"""
        count = 0

        def traverse(node):
            nonlocal count
            count += 1
            for child in node.children:
                traverse(child)

        for root_node in self.root_children:
            traverse(root_node)
        return count


class TokenExplorer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.node_counter = 0

    def nucleus_sampling(
        self,
        logits: torch.Tensor,
        temp: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        if temp != 1.0:
            logits = logits / temp

        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Apply top_k if specified
        if top_k > 0:
            sorted_probs = sorted_probs[:top_k]
            sorted_indices = sorted_indices[:top_k]

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff index for nucleus sampling
        cutoff_mask = cumulative_probs <= top_p
        # Include at least one token and all tokens until we exceed top_p
        if cutoff_mask.any():
            cutoff_index = cutoff_mask.sum().item()
        else:
            cutoff_index = 1

        # Ensure we get at least one token
        cutoff_index = max(cutoff_index, 1)

        nucleus_tokens = []
        for i in range(cutoff_index):
            token_id = sorted_indices[i].item()
            probability = sorted_probs[i].item()
            nucleus_tokens.append((token_id, probability))

        return nucleus_tokens

    def _print_verbose(self, message, indent=0, verbose=False):
        """Helper function to print verbose messages"""
        if verbose:
            print("  " * indent + message)

    def _print_current_path(self, input_ids, verbose=False, indent=0):
        """Print the current token sequence"""
        if verbose:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
            token_str = " → ".join([f"'{t}'" for t in tokens])
            self._print_verbose(f"📍 Current path: {token_str}", indent, True)

    def generate(
        self,
        input_ids,
        max_length=10,
        layer=0,
        current_node=None,
        verbose=False,
        **generation_kwargs,
    ):
        """
        Recursively generate a token tree.

        Args:
            input_ids: Current token sequence
            max_length: Maximum depth of the tree
            layer: Current layer in the tree
            current_node: Current node being expanded
            verbose: If True, print detailed generation progress
            **generation_kwargs: Arguments for nucleus_sampling (temp, top_p, top_k)

        Returns:
            GenerationTree object (only on initial call)
        """
        # Base case: reached max depth
        if layer >= max_length:
            self._print_verbose(f"🛑 Reached max depth ({max_length})", layer, verbose)
            return None

        # Get model predictions
        self._print_verbose(
            f"🔍 Layer {layer}: Getting model predictions...", layer, verbose
        )
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Get nucleus tokens for current position
        nucleus_tokens = self.nucleus_sampling(
            outputs.logits[:, -1, :].squeeze(0), **generation_kwargs
        )

        self._print_verbose(
            f"💡 Found {len(nucleus_tokens)} candidate tokens (top_p={generation_kwargs.get('top_p', 0.9)}, top_k={generation_kwargs.get('top_k', 50)})",
            layer,
            verbose,
        )

        # Initial call - create root nodes
        if current_node is None:
            if verbose:
                print("\n" + "=" * 80)
                print("🌳 STARTING TREE GENERATION")
                print("=" * 80)
                self._print_current_path(input_ids, True, 0)
                print()

            root_nodes = []
            self.node_counter = 0

            for idx, (token_id, probability) in enumerate(nucleus_tokens):
                self.node_counter += 1
                token_text = self.tokenizer.decode([token_id])

                self._print_verbose(
                    f"🌱 Root node {idx + 1}/{len(nucleus_tokens)}: Creating node for token '{token_text}' (ID: {token_id}, p={probability:.4f})",
                    layer,
                    verbose,
                )

                # Create node for this token
                new_node = GenerationTokenNode(
                    layer=layer,
                    token_id=token_id,
                    probability=probability,
                    previous_tokens=input_ids.clone(),
                )

                # Create new input sequence with this token
                new_input_ids = torch.cat(
                    [input_ids, torch.tensor([[token_id]]).to(self.model.device)], dim=1
                ).to(self.model.device)

                if verbose:
                    print()
                    self._print_verbose(
                        f"⬇️  Expanding children of '{token_text}'...", layer, True
                    )

                # Recursively generate children
                self.generate(
                    new_input_ids,
                    max_length,
                    layer + 1,
                    new_node,
                    verbose,
                    **generation_kwargs,
                )

                root_nodes.append(new_node)

                if verbose:
                    print()
                    self._print_verbose(
                        f"✅ Completed root node {idx + 1}/{len(nucleus_tokens)}: '{token_text}' ({len(new_node.children)} children)",
                        layer,
                        True,
                    )
                    print()

            # Return the tree
            tree = GenerationTree(root_nodes)

            if verbose:
                print("\n" + "=" * 80)
                print(f"🎉 TREE GENERATION COMPLETE")
                print(f"📊 Total nodes created: {self.node_counter}")
                print(f"📊 Total nodes in tree: {tree.count_nodes()}")
                print(f"📊 Root nodes: {len(tree.root_children)}")
                print(f"📊 Max depth: {max_length}")
                print("=" * 80 + "\n")

            return tree

        # Recursive call - expand current node
        else:
            parent_token = self.tokenizer.decode([current_node.token_id])
            self._print_verbose(
                f"🔄 Expanding node '{parent_token}' (layer {current_node.layer})",
                layer,
                verbose,
            )
            self._print_current_path(input_ids, verbose, layer)

            for idx, (token_id, probability) in enumerate(nucleus_tokens):
                self.node_counter += 1
                token_text = self.tokenizer.decode([token_id])

                self._print_verbose(
                    f"  ├─ Child {idx + 1}/{len(nucleus_tokens)}: '{token_text}' (ID: {token_id}, p={probability:.4f})",
                    layer,
                    verbose,
                )

                # Create child node
                new_node = GenerationTokenNode(
                    layer=layer,
                    token_id=token_id,
                    probability=probability,
                    previous_tokens=input_ids.clone(),
                )

                # Add to current node's children
                current_node.children.append(new_node)

                # Create new input sequence
                new_input_ids = torch.cat(
                    [input_ids, torch.tensor([[token_id]]).to(self.model.device)], dim=1
                ).to(self.model.device)

                # Recursively generate children
                self.generate(
                    new_input_ids,
                    max_length,
                    layer + 1,
                    new_node,
                    verbose,
                    **generation_kwargs,
                )

            self._print_verbose(
                f"✓ Node '{parent_token}' expanded with {len(current_node.children)} children",
                layer,
                verbose,
            )

            return None


class TreeVisualizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def visualize_tree(
        self,
        tree: GenerationTree,
        show_probabilities=True,
        node_size=0.3,
        vertical_spacing=2.0,
        horizontal_spacing=1.0,
        max_nodes_per_layer=None,
    ):
        """
        Visualize the generation tree using matplotlib.

        Args:
            tree: GenerationTree object to visualize
            show_probabilities: Whether to show probabilities on edges
            node_size: Size of each node circle
            vertical_spacing: Vertical space between layers
            horizontal_spacing: Horizontal space between nodes
            max_nodes_per_layer: Maximum nodes to display per layer (for large trees)
        """
        # Get all nodes organized by layer
        layers = tree.get_all_nodes()
        max_layer = max(layers.keys()) if layers else 0

        # Sort nodes by probability within each layer and arrange high prob in center
        sorted_layers = {}
        for layer_idx in sorted(layers.keys()):
            nodes = layers[layer_idx]
            if max_nodes_per_layer and len(nodes) > max_nodes_per_layer:
                nodes = nodes[:max_nodes_per_layer]

            # Sort by probability descending
            nodes_sorted = sorted(nodes, key=lambda n: n.probability, reverse=True)

            # Arrange so highest prob are in center
            arranged_nodes = []
            left = []
            right = []
            for i, node in enumerate(nodes_sorted):
                if i % 2 == 0:
                    right.append(node)
                else:
                    left.append(node)
            arranged_nodes = left[::-1] + right
            sorted_layers[layer_idx] = arranged_nodes

        # Calculate figure size based on tree dimensions
        max_width = max(len(sorted_layers.get(i, [])) for i in range(max_layer + 1))
        fig_width = max(20, max_width * horizontal_spacing * 2)
        fig_height = max(12, (max_layer + 1) * vertical_spacing * 2)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Calculate positions
        node_positions = {}

        for layer_idx in sorted(sorted_layers.keys()):
            nodes = sorted_layers[layer_idx]
            y = fig_height - layer_idx * vertical_spacing - 1
            num_nodes = len(nodes)

            # Center the nodes horizontally
            total_width = (num_nodes - 1) * horizontal_spacing if num_nodes > 1 else 0
            start_x = (fig_width - total_width) / 2

            for i, node in enumerate(nodes):
                if num_nodes == 1:
                    x = fig_width / 2
                else:
                    x = start_x + i * horizontal_spacing
                node_positions[id(node)] = (x, y)

        # Set axis limits with padding
        ax.set_xlim(-1, fig_width + 1)
        ax.set_ylim(-1, fig_height + 1)
        ax.axis("off")

        # Create colormap for edges based on probability
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        cmap = cm.get_cmap("RdYlGn")  # Red (low) to Yellow to Green (high)
        norm = Normalize(vmin=0, vmax=1)

        # Draw edges first
        for layer_idx in sorted(sorted_layers.keys()):
            if layer_idx == 0:
                continue

            nodes = sorted_layers[layer_idx]

            for node in nodes:
                # Find parent
                parent = None
                for parent_layer_idx in sorted(sorted_layers.keys()):
                    if parent_layer_idx >= layer_idx:
                        break
                    for potential_parent in sorted_layers[parent_layer_idx]:
                        if node in potential_parent.children:
                            parent = potential_parent
                            break
                    if parent:
                        break

                if (
                    parent
                    and id(parent) in node_positions
                    and id(node) in node_positions
                ):
                    x1, y1 = node_positions[id(parent)]
                    x2, y2 = node_positions[id(node)]

                    # Color based on probability
                    edge_color = cmap(norm(node.probability))

                    # Draw edge with color based on probability
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=edge_color,
                        linewidth=2.5,
                        alpha=0.8,
                        zorder=1,
                    )

                    # Add probability label
                    if show_probabilities:
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.text(
                            mid_x,
                            mid_y,
                            f"{node.probability:.3f}",
                            fontsize=9,
                            ha="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                alpha=0.9,
                                edgecolor=edge_color,
                                linewidth=1.5,
                            ),
                            zorder=2,
                        )

        # Draw nodes
        for layer_idx in sorted(sorted_layers.keys()):
            nodes = sorted_layers[layer_idx]

            for node in nodes:
                if id(node) not in node_positions:
                    continue

                x, y = node_positions[id(node)]

                # Decode token
                token_text = self.tokenizer.decode([node.token_id])
                # Clean up token text
                token_text = token_text.replace("\n", "\\n").replace("\t", "\\t")
                if len(token_text) > 15:
                    token_text = token_text[:15] + "..."

                # Draw node with equal size
                circle = plt.Circle(
                    (x, y),
                    node_size,
                    color="lightblue",
                    ec="darkblue",
                    linewidth=2.5,
                    zorder=3,
                )
                ax.add_patch(circle)

                # Add token text
                ax.text(
                    x,
                    y,
                    token_text,
                    fontsize=10,
                    ha="center",
                    va="center",
                    weight="bold",
                    zorder=4,
                )

        # Add colorbar legend
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.01, shrink=0.6)
        cbar.set_label("Edge Probability", fontsize=12, weight="bold")

        plt.title("Token Generation Tree", fontsize=18, weight="bold", pad=20)
        plt.tight_layout()

        return fig

    def print_tree(self, tree: GenerationTree, max_depth=None):
        """Print tree structure in text format"""

        def print_node(node, indent=0, is_last=True):
            if max_depth and indent > max_depth:
                return

            prefix = "  " * indent
            if indent > 0:
                prefix += "└─ " if is_last else "├─ "

            token_text = self.tokenizer.decode([node.token_id])
            token_text = token_text.replace("\n", "\\n").replace("\t", "\\t")

            print(
                f"{prefix}[{token_text}] (p={node.probability:.4f}, layer={node.layer})"
            )

            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                print_node(child, indent + 1, is_last_child)

        print("Generation Tree:")
        print("=" * 50)
        for i, root_node in enumerate(tree.root_children):
            is_last = i == len(tree.root_children) - 1
            print_node(root_node, 0, is_last)
            if not is_last:
                print()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "openai-gpt"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    explorer = TokenExplorer(model, tokenizer)

    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate tree with verbose output
    tree = explorer.generate(
        input_ids,
        max_length=3,  # Depth of tree (reduced for demo)
        temp=1.0,
        top_p=0.9,
        top_k=5,  # Limit branches per node (reduced for demo)
        verbose=True,  # Enable verbose mode
    )

    # Visualize
    visualizer = TreeVisualizer(tokenizer)
    fig = visualizer.visualize_tree(
        tree,
        show_probabilities=True,
        node_size=0.3,  # Size of each node
        vertical_spacing=2.0,  # Space between layers
        horizontal_spacing=1.0,  # Space between nodes
    )
    plt.show()

    print("\n" + "=" * 80)
    print("FINAL TREE STRUCTURE:")
    print("=" * 80)
    visualizer.print_tree(tree)
