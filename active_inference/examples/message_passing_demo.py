#!/usr/bin/env python3
"""
Message Passing Algorithms Demonstration - Thin Orchestrator

This example demonstrates belief propagation and variational message passing by orchestrating
components from the active_inference framework. Shows GPU-accelerated inference on different
graph structures with comprehensive validation and visualization.

Active Inference Methods Used:
- Belief propagation (exact inference on trees)
- Variational message passing (approximate inference on loopy graphs)
- Message passing algorithms for probabilistic graphical models
- GPU-accelerated inference on different graph topologies

Generative Models:
- Chain-structured graphs (tree models)
- Grid-structured graphs (loopy models)
- Random graph models
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the active_inference package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up outputs directory
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "message_passing"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Import from the active_inference package using standardized imports
import active_inference as ai


class MessagePassingDemo:
    """Demonstration of message passing algorithms."""

    def __init__(self, feature_manager):
        self.feature_manager = feature_manager
        self.bp_engine = ai.BeliefPropagation(feature_manager)
        self.vmp_engine = ai.VariationalMessagePassing(feature_manager)
        self.gpu_accelerator = ai.GPUAccelerator(feature_manager)

    def create_chain_graph(self, num_nodes=5):
        """Create a chain-structured graph (tree)."""

        print(f"Creating chain graph with {num_nodes} nodes...")

        # Create adjacency matrix for chain: 0-1-2-3-4
        adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float)

        for i in range(num_nodes - 1):
            adjacency[i, i + 1] = 1.0
            adjacency[i + 1, i] = 1.0  # Undirected

        # Create node potentials (evidence)
        num_states = 3
        node_potentials = torch.softmax(torch.randn(num_nodes, num_states), dim=1)

        # Add strong evidence at endpoints
        node_potentials[0, 0] = 5.0  # Strong evidence for state 0 at node 0
        node_potentials[-1, 2] = 5.0  # Strong evidence for state 2 at last node
        node_potentials = torch.softmax(node_potentials, dim=1)

        return adjacency, node_potentials, num_states

    def create_grid_graph(self, size=3):
        """Create a 2D grid graph (loopy)."""

        print(f"Creating {size}x{size} grid graph...")

        num_nodes = size * size
        adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float)

        # Create grid connections
        for i in range(size):
            for j in range(size):
                node = i * size + j

                # Connect to right neighbor
                if j < size - 1:
                    right = i * size + (j + 1)
                    adjacency[node, right] = 1.0
                    adjacency[right, node] = 1.0

                # Connect to bottom neighbor
                if i < size - 1:
                    bottom = (i + 1) * size + j
                    adjacency[node, bottom] = 1.0
                    adjacency[bottom, node] = 1.0

        # Create node potentials
        num_states = 2
        node_potentials = torch.softmax(torch.randn(num_nodes, num_states), dim=1)

        # Add some structured evidence
        # Corner nodes have preferences
        corners = [0, size - 1, size * (size - 1), size * size - 1]
        for corner in corners:
            if corner < num_nodes:
                node_potentials[corner, 0] = 3.0  # Prefer state 0

        node_potentials = torch.softmax(node_potentials, dim=1)

        return adjacency, node_potentials, num_states

    def create_random_graph(self, num_nodes=6, edge_prob=0.3):
        """Create a random graph."""

        print(f"Creating random graph with {num_nodes} nodes (p={edge_prob})...")

        # Create random adjacency matrix
        adjacency = torch.bernoulli(
            torch.full((num_nodes, num_nodes), edge_prob, dtype=torch.float)
        )

        # Make symmetric (undirected)
        adjacency = torch.triu(adjacency, diagonal=1)
        adjacency = adjacency + adjacency.t()

        # Ensure graph is connected (add some guaranteed connections)
        for i in range(num_nodes - 1):
            adjacency[i, i + 1] = 1.0
            adjacency[i + 1, i] = 1.0

        # Create node potentials
        num_states = 3
        node_potentials = torch.softmax(torch.randn(num_nodes, num_states), dim=1)

        return adjacency, node_potentials, num_states

    def run_belief_propagation(self, adjacency, node_potentials, max_iterations=20):
        """Run belief propagation algorithm."""

        print("\n--- Running Belief Propagation ---")

        # Set up the graph
        self.bp_engine.set_graph(adjacency, node_potentials)

        # Run inference
        result = self.bp_engine.run(max_iterations=max_iterations, tolerance=1e-5)

        print(f"Converged: {result['converged']}")
        print(f"Final beliefs shape: {result['beliefs'].shape}")

        # Analyze results
        beliefs = result["beliefs"]
        most_probable_states = torch.argmax(beliefs, dim=1)

        print("Most probable states per node:")
        for i in range(len(beliefs)):
            state = most_probable_states[i].item()
            prob = beliefs[i, state].item()
            print(f"  Node {i}: state={state}, prob={prob:.4f}")

        # Compute belief entropy (uncertainty)
        entropy = torch.distributions.Categorical(probs=beliefs).entropy()
        # entropy printed below

        return result

    def run_variational_message_passing(
        self, adjacency, node_potentials, max_iterations=30
    ):
        """Run variational message passing algorithm."""

        print("\n--- Running Variational Message Passing ---")

        # Set up the graph
        self.vmp_engine.set_graph(adjacency, node_potentials)

        # Run inference
        result = self.vmp_engine.run(max_iterations=max_iterations, learning_rate=0.1)

        print(f"Final beliefs shape: {result['beliefs'].shape}")
        # monitoring print removed

        # Analyze results
        beliefs = result["beliefs"]
        most_probable_states = torch.argmax(beliefs, dim=1)

        print("Most probable states per node:")
        for i in range(len(beliefs)):
            state = most_probable_states[i].item()
            prob = beliefs[i, state].item()
            print(f"  Node {i}: state={state}, prob={prob:.4f}")

        # Compute belief entropy
        entropy = torch.distributions.Categorical(probs=beliefs).entropy()
        # monitoring print removed

        return result

    def compare_algorithms(self, adjacency, node_potentials):
        """Compare belief propagation and variational message passing."""

        print("\n" + "=" * 60)
        print("ALGORITHM COMPARISON")
        print("=" * 60)

        # Run both algorithms
        bp_result = self.run_belief_propagation(adjacency, node_potentials)
        vmp_result = self.run_variational_message_passing(adjacency, node_potentials)

        # Compare results
        bp_beliefs = bp_result["beliefs"]
        vmp_beliefs = vmp_result["beliefs"]

        # Compute differences
        belief_diff = torch.norm(bp_beliefs - vmp_beliefs)
        print(f"Belief difference norm: {belief_diff:.6f}")

        # Compare most probable states
        bp_states = torch.argmax(bp_beliefs, dim=1)
        vmp_states = torch.argmax(vmp_beliefs, dim=1)

        agreement = (bp_states == vmp_states).float().mean()
        print(f"Agreement: {agreement:.1%}")

        # For loopy graphs, BP may not converge exactly
        if not bp_result["converged"]:
            print("Note: BP did not converge exactly (expected for loopy graphs)")

        return bp_result, vmp_result


def analyze_graph_structures():
    """Analyze message passing on different graph structures."""

    print("=" * 80)
    print("MESSAGE PASSING ON DIFFERENT GRAPH STRUCTURES")
    print("=" * 80)

    feature_manager = TritonFeatureManager()
    demo = MessagePassingDemo(feature_manager)

    results = {}

    # Test 1: Chain graph (tree)
    print("\n" + "=" * 60)
    print("TEST 1: CHAIN GRAPH (TREE STRUCTURE)")
    print("=" * 60)

    adjacency, node_potentials, num_states = demo.create_chain_graph(6)
    bp_result, vmp_result = demo.compare_algorithms(adjacency, node_potentials)

    results["chain"] = {
        "adjacency": adjacency,
        "bp_result": bp_result,
        "vmp_result": vmp_result,
        "type": "tree",
    }

    # Test 2: Grid graph (loopy)
    print("\n" + "=" * 60)
    print("TEST 2: GRID GRAPH (LOOPY STRUCTURE)")
    print("=" * 60)

    adjacency, node_potentials, num_states = demo.create_grid_graph(3)
    bp_result, vmp_result = demo.compare_algorithms(adjacency, node_potentials)

    results["grid"] = {
        "adjacency": adjacency,
        "bp_result": bp_result,
        "vmp_result": vmp_result,
        "type": "loopy",
    }

    # Test 3: Random graph
    print("\n" + "=" * 60)
    print("TEST 3: RANDOM GRAPH")
    print("=" * 60)

    adjacency, node_potentials, num_states = demo.create_random_graph(8, 0.4)
    bp_result, vmp_result = demo.compare_algorithms(adjacency, node_potentials)

    results["random"] = {
        "adjacency": adjacency,
        "bp_result": bp_result,
        "vmp_result": vmp_result,
        "type": "random",
    }

    # Save results to JSON
    import json
    results_summary = {
        'timestamp': time.time(),
        'results': {k: {
            'type': v['type'],
            'bp_converged': v['bp_result']['converged'],
            'vmp_final_fe': v['vmp_result']['free_energy'].item() if hasattr(v['vmp_result']['free_energy'], 'item') else float(v['vmp_result']['free_energy'])
        } for k, v in results.items()}
    }
    
    results_path = OUTPUTS_DIR / 'message_passing_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"✓ Results saved to {results_path}")

    return results


def create_visualization(results):
    """Create comprehensive visualization of message passing results."""

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Message Passing Algorithm Comparison", fontsize=16)

        graph_types = list(results.keys())
        colors = ["blue", "red", "green"]

        # Plot 1: Graph structures
        for i, graph_type in enumerate(graph_types):
            ax = axes[0, i]
            adjacency = results[graph_type]["adjacency"]

            # Create networkx graph for visualization
            G = nx.from_numpy_array(adjacency.numpy())

            # Position nodes
            if graph_type == "chain":
                pos = {j: (j, 0) for j in range(len(G))}
            elif graph_type == "grid":
                size = int(np.sqrt(len(G)))
                pos = {
                    (i * size + j): (j, -i) for i in range(size) for j in range(size)
                }
            else:  # random
                pos = nx.spring_layout(G, seed=42)

            nx.draw(
                G,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightblue",
                node_size=300,
                font_size=10,
                font_weight="bold",
            )
            ax.set_title(f"{graph_type.capitalize()} Graph")
            ax.axis("off")

        # Plot 2: Belief comparison (BP vs VMP)
        ax = axes[1, 0]
        for i, graph_type in enumerate(graph_types):
            bp_beliefs = results[graph_type]["bp_result"]["beliefs"]
            vmp_beliefs = results[graph_type]["vmp_result"]["beliefs"]

            # Plot belief differences for first state
            bp_state0 = bp_beliefs[:, 0].cpu().numpy()
            vmp_state0 = vmp_beliefs[:, 0].cpu().numpy()

            x = np.arange(len(bp_state0))
            ax.scatter(
                x + i * 0.1,
                bp_state0,
                color=colors[i],
                marker="o",
                label=f"{graph_type} BP",
                alpha=0.7,
            )
            ax.scatter(
                x + i * 0.1 + 0.05,
                vmp_state0,
                color=colors[i],
                marker="s",
                label=f"{graph_type} VMP",
                alpha=0.7,
            )

        ax.set_xlabel("Node Index")
        ax.set_ylabel("Belief in State 0")
        ax.set_title("Belief Comparison (BP vs VMP)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        # Plot 3: Convergence analysis
        ax = axes[1, 1]
        convergence_data = []
        labels = []

        for graph_type in graph_types:
            bp_converged = results[graph_type]["bp_result"]["converged"]
            convergence_data.append(1 if bp_converged else 0)
            labels.append(f"{graph_type}\nBP")

        ax.bar(
            labels,
            convergence_data,
            color=["green" if x else "red" for x in convergence_data],
        )
        ax.set_ylabel("Converged (1=Yes, 0=No)")
        ax.set_title("Algorithm Convergence")
        ax.set_ylim(-0.1, 1.1)

        # Add value labels
        for i, v in enumerate(convergence_data):
            ax.text(i, v + 0.05, str(v), ha="center", va="bottom")

        # Plot 4: Free energy comparison (for VMP)
        ax = axes[1, 2]
        for i, graph_type in enumerate(graph_types):
            vmp_fe = results[graph_type]["vmp_result"]["free_energy"]

            ax.bar(
                i,
                vmp_fe.item(),
                color=colors[i],
                alpha=0.7,
                label=graph_type.capitalize(),
            )

        ax.set_xlabel("Graph Type")
        ax.set_ylabel("Variational Free Energy")
        ax.set_title("Final Free Energy (VMP)")
        ax.set_xticks(range(len(graph_types)))
        ax.set_xticklabels([t.capitalize() for t in graph_types])
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_path = OUTPUTS_DIR / "message_passing_comparison.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"Main comparison saved as {save_path}")

        # Create detailed belief evolution plots
        for graph_type in graph_types:
            plt.figure(figsize=(12, 8))

            bp_beliefs = results[graph_type]["bp_result"]["beliefs"]
            vmp_beliefs = results[graph_type]["vmp_result"]["beliefs"]

            num_nodes = bp_beliefs.shape[0]
            num_states = bp_beliefs.shape[1]

            fig, axes = plt.subplots(2, num_states, figsize=(15, 10))
            fig.suptitle(
                f"Belief Distributions - {graph_type.capitalize()} Graph", fontsize=14
            )

            for state in range(num_states):
                # BP beliefs
                axes[0, state].bar(
                    range(num_nodes),
                    bp_beliefs[:, state].cpu().numpy(),
                    color="blue",
                    alpha=0.7,
                )
                axes[0, state].set_title(f"BP - State {state}")
                axes[0, state].set_xlabel("Node")
                axes[0, state].set_ylabel("Belief")
                axes[0, state].grid(True, alpha=0.3)

                # VMP beliefs
                axes[1, state].bar(
                    range(num_nodes),
                    vmp_beliefs[:, state].cpu().numpy(),
                    color="red",
                    alpha=0.7,
                )
                axes[1, state].set_title(f"VMP - State {state}")
                axes[1, state].set_xlabel("Node")
                axes[1, state].set_ylabel("Belief")
                axes[1, state].grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = OUTPUTS_DIR / f"beliefs_{graph_type}.png"
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            print(f"Detailed beliefs plot saved as {save_path}")

    except ImportError as e:
        print(f"Visualization libraries not available: {e}")
        print("Install matplotlib and networkx for full visualization:")
        print("  pip install matplotlib networkx")
    except Exception as e:
        print(f"Visualization failed: {e}")


def run_performance_analysis():
    """Analyze performance characteristics of message passing."""

    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    import time

    feature_manager = TritonFeatureManager()
    demo = MessagePassingDemo(feature_manager)

    # Test different graph sizes
    sizes = [5, 10, 15, 20]

    print("Scaling analysis for chain graphs:")
    print("Nodes | BP Time (ms) | VMP Time (ms) | BP Converged")
    print("-" * 55)

    for num_nodes in sizes:
        adjacency, node_potentials, _ = demo.create_chain_graph(num_nodes)

        # Time BP
        start_time = time.time()
        bp_result = demo.run_belief_propagation(
            adjacency, node_potentials, max_iterations=10
        )
        bp_time = (time.time() - start_time) * 1000

        # Time VMP
        start_time = time.time()
        vmp_result = demo.run_variational_message_passing(
            adjacency, node_potentials, max_iterations=10
        )
        vmp_time = (time.time() - start_time) * 1000

        converged = "✓" if bp_result["converged"] else "✗"
        print("5d")

    # Memory analysis
    gpu_accelerator = GPUAccelerator(feature_manager)
    memory_stats = gpu_accelerator.get_memory_stats()

    if memory_stats:
        print("\nGPU Memory Usage:")
        for key, value in memory_stats.items():
            if "allocated" in key.lower():
                print(f"  {key}: {value:.2f} MB")
            else:
                print(f"{key}: {value}")


def main():
    """Run the complete message passing demonstration."""
    print("🚀 MESSAGE PASSING ALGORITHMS DEMONSTRATION")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run analysis
        results = analyze_graph_structures()
        
        # Create visualizations
        create_visualization(results)
        
        # Run performance analysis
        run_performance_analysis()
        
        print(f"\n🎉 DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {OUTPUTS_DIR}")
        print("  - message_passing_comparison.png")
        print("  - beliefs_*.png files")
        print("  - message_passing_results.json")
        
        return results
        
    except Exception as e:
        print(f"\n💥 Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("\n🏁 Demonstration finished successfully!")
        else:
            print("\n❌ Demonstration encountered errors.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
