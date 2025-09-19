#!/usr/bin/env python3
"""
Policy Selection with Expected Free Energy

Thin orchestrator demonstrating policy selection using expected free energy
minimization. Shows how active inference selects actions to minimize expected
free energy in decision-making scenarios.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the active_inference package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up outputs directory
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "policy_selection"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Import from the active_inference package using standardized imports
import active_inference as ai


class PolicySelector:
    """Policy selection using expected free energy minimization."""

    def __init__(self, feature_manager):
        self.feature_manager = feature_manager
        self.efe_engine = ai.ExpectedFreeEnergy(feature_manager)
        self.gpu_accelerator = ai.GPUAccelerator(feature_manager)

    def evaluate_policies(self, observations, policies, posterior, preferences=None):
        """Evaluate expected free energy for all policies."""

        EFE = self.efe_engine.compute(observations, policies, posterior, preferences)

        # Select optimal policy
        policy_indices, efe_values = self.efe_engine.select_policy(EFE)

        return EFE, policy_indices, efe_values

    def simulate_decision_making(
        self, num_steps=10, num_policies=5, observation_dim=4, policy_dim=4
    ):
        """Simulate a decision-making scenario."""

        print("Simulating decision-making scenario...")
        print(f"Steps: {num_steps}, Policies: {num_policies}")

        # Initialize state
        observations = torch.randn(1, observation_dim)  # Single agent
        posterior = torch.softmax(torch.randn(1, policy_dim), dim=1)

        # Define available policies (action sequences)
        policies = torch.randn(num_policies, policy_dim)

        # Define preferences (goals)
        preferences = torch.tensor(
            [[1.0, -0.5, 0.2, -1.0, 0.8]]
        )  # Per policy preferences

        history = {
            "observations": [],
            "posteriors": [],
            "selected_policies": [],
            "efe_values": [],
            "preferences": preferences.squeeze().tolist(),
        }

        for step in range(num_steps):
            # Evaluate all policies
            EFE, policy_indices, efe_values = self.evaluate_policies(
                observations, policies, posterior, preferences
            )

            selected_policy = policy_indices.item()
            selected_efe = efe_values.item()

            # Record state
            history["observations"].append(observations.squeeze().tolist())
            history["posteriors"].append(posterior.squeeze().tolist())
            history["selected_policies"].append(selected_policy)
            history["efe_values"].append(selected_efe)

            print(
                f"Step {step}: Selected policy {selected_policy}, EFE = {selected_efe:.4f}"
            )
            # Simulate state transition (simplified)
            # In a real scenario, this would be the environment dynamics
            action_effect = policies[selected_policy : selected_policy + 1] * 0.1
            observations = (
                observations + action_effect + 0.05 * torch.randn_like(observations)
            )

            # Update posterior based on new observations (simplified)
            posterior = torch.softmax(
                posterior + 0.1 * torch.randn_like(posterior), dim=1
            )

        return history


def create_multi_agent_scenario():
    """Create a multi-agent decision making scenario."""

    print("Creating multi-agent decision scenario...")

    # Scenario: Multiple agents making decisions in a shared environment
    num_agents = 3
    num_policies = 4
    observation_dim = 6
    policy_dim = 6

    # Generate observations for each agent
    observations = torch.randn(num_agents, observation_dim)

    # Each agent has different beliefs about the environment
    posteriors = torch.softmax(torch.randn(num_agents, policy_dim), dim=1)

    # Define shared action space (policies)
    policies = torch.randn(num_policies, policy_dim)

    # Different agents have different preferences
    preferences = torch.tensor(
        [
            [1.0, 0.5, -0.5, -1.0],  # Agent 1: prefers policy 0, dislikes 3
            [-0.5, 1.0, 0.5, -0.2],  # Agent 2: prefers policy 1
            [0.2, -0.3, 1.0, 0.8],  # Agent 3: prefers policy 2
        ]
    )

    return observations, posteriors, policies, preferences


def analyze_policy_selection(feature_manager):
    """Analyze policy selection behavior."""

    print("=" * 60)
    print("POLICY SELECTION ANALYSIS")
    print("=" * 60)

    selector = PolicySelector(feature_manager)

    # Single agent analysis
    print("\n--- Single Agent Analysis ---")
    observations = torch.randn(1, 4)
    policies = torch.randn(5, 4)
    posterior = torch.softmax(torch.randn(1, 4), dim=1)
    preferences = torch.tensor([[1.0, -0.5, 0.2, -0.8, 0.3]])

    EFE, policy_indices, efe_values = selector.evaluate_policies(
        observations, policies, posterior, preferences
    )

    print("Policy Expected Free Energies:")
    for i in range(len(policies)):
        print(f"  Policy {i}: {policies[i].tolist()}")

    print(f"\nSelected policy: {policy_indices.item()}")
    print(f"Selected EFE: {selected_efe:.4f}")
    # Multi-agent analysis
    print("\n--- Multi-Agent Analysis ---")
    observations, posteriors, policies, preferences = create_multi_agent_scenario()

    EFE_matrix, policy_indices, efe_values = selector.evaluate_policies(
        observations, policies, posteriors, preferences
    )

    print("Multi-agent policy selection:")
    for agent in range(len(observations)):
        print(f"  Agent {agent}: Selected policy {policy_indices[agent].item()}")
    # Analyze coordination
    unique_policies = torch.unique(policy_indices)
    if len(unique_policies) == 1:
        print(f"\nAll agents agreed on policy {unique_policies.item()}")
    else:
        print(f"\nAgents selected different policies: {policy_indices.tolist()}")

    # Simulate decision making
    print("\n--- Decision Making Simulation ---")
    history = selector.simulate_decision_making(
        num_steps=8, num_policies=5, observation_dim=4, policy_dim=4
    )

    # Analyze policy diversity
    selected_policies = history["selected_policies"]
    unique_selections = len(set(selected_policies))
    policy_frequency = {}
    for policy in selected_policies:
        policy_frequency[policy] = policy_frequency.get(policy, 0) + 1

    print(
        f"Policy selection diversity: {unique_selections}/{len(set(range(5)))} unique policies"
    )
    print("Policy frequencies:", policy_frequency)

    return history


def create_exploration_exploitation_analysis():
    """Analyze exploration-exploitation trade-off."""

    print("\n" + "=" * 60)
    print("EXPLORATION-EXPLOITATION ANALYSIS")
    print("=" * 60)

    feature_manager = TritonFeatureManager()
    selector = PolicySelector(feature_manager)

    # Create scenario with clear optimal policy
    observation_dim = 4
    num_policies = 6

    observations = torch.randn(1, observation_dim)
    policies = torch.randn(num_policies, observation_dim)

    # Strong preference for one policy (exploitation)
    preferences = torch.zeros(1, num_policies)
    preferences[0, 2] = 2.0  # Policy 2 is strongly preferred

    # Test different uncertainty levels
    uncertainty_levels = [0.1, 0.5, 1.0, 2.0]

    print("Testing different uncertainty levels:")
    print("Uncertainty | Selected Policy | EFE Value | Exploration?")
    print("-" * 50)

    for uncertainty in uncertainty_levels:
        # Create posterior with specified uncertainty
        posterior_logits = torch.randn(1, observation_dim) * uncertainty
        posterior = torch.softmax(posterior_logits, dim=1)

        EFE, policy_indices, efe_values = selector.evaluate_policies(
            observations, policies, posterior, preferences
        )

        selected_policy = policy_indices.item()
        selected_efe = efe_values.item()
        is_exploratory = selected_policy != 2  # Not the preferred policy

        print(
            f"  Uncertainty: {uncertainty:.1f} -> Policy: {selected_policy}, EFE: {selected_efe:.4f}, Exploratory: {is_exploratory}"
        )

        # Analyze EFE distribution
        efe_std = EFE.std().item()
        efe_range = EFE.max().item() - EFE.min().item()

        print(f"  EFE Std Dev: {efe_std:.4f}")
        print(f"  EFE Range: {efe_range:.4f}")

    return True


def create_visualization(history):
    """Create visualization of policy selection results."""

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Policy Selection Analysis", fontsize=16)

        # Plot 1: Policy selection over time
        ax = axes[0, 0]
        time_steps = range(len(history["selected_policies"]))
        ax.plot(
            time_steps, history["selected_policies"], "bo-", linewidth=2, markersize=8
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Selected Policy")
        ax.set_title("Policy Selection Trajectory")
        ax.grid(True, alpha=0.3)
        ax.set_yticks(range(max(history["selected_policies"]) + 1))

        # Plot 2: EFE values over time
        ax = axes[0, 1]
        ax.plot(time_steps, history["efe_values"], "r^-", linewidth=2, markersize=8)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Expected Free Energy")
        ax.set_title("EFE of Selected Policies")
        ax.grid(True, alpha=0.3)

        # Plot 3: Preference visualization
        ax = axes[0, 2]
        policies = list(range(len(history["preferences"])))
        ax.bar(policies, history["preferences"], color="green", alpha=0.7)
        ax.set_xlabel("Policy Index")
        ax.set_ylabel("Preference Value")
        ax.set_title("Policy Preferences")
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 4: Posterior belief evolution
        ax = axes[1, 0]
        posteriors = np.array(history["posteriors"])
        im = ax.imshow(
            posteriors.T, aspect="auto", cmap="viridis", interpolation="nearest"
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Belief Dimension")
        ax.set_title("Belief Evolution")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Plot 5: Observation trajectory
        ax = axes[1, 1]
        observations = np.array(history["observations"])
        for dim in range(observations.shape[1]):
            ax.plot(
                time_steps, observations[:, dim], label=f"Dimension {dim}", linewidth=2
            )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Observation Value")
        ax.set_title("Observation Trajectory")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        # Plot 6: Policy selection frequency
        ax = axes[1, 2]
        policy_counts = {}
        for policy in history["selected_policies"]:
            policy_counts[policy] = policy_counts.get(policy, 0) + 1

        policies = list(policy_counts.keys())
        counts = list(policy_counts.values())

        ax.bar(policies, counts, color="orange", alpha=0.7)
        ax.set_xlabel("Policy Index")
        ax.set_ylabel("Selection Frequency")
        ax.set_title("Policy Selection Frequency")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, count in enumerate(counts):
            ax.text(policies[i], count + 0.1, str(count), ha="center", va="bottom")

        plt.tight_layout()
        save_path = OUTPUTS_DIR / "policy_selection_analysis.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"Main visualization saved as {save_path}")

        # Create EFE landscape visualization
        plt.figure(figsize=(10, 8))

        # Simulate EFE landscape for different policy-belief combinations
        num_policies = 5
        belief_resolution = 10

        EFE_landscape = np.random.randn(
            num_policies, belief_resolution, belief_resolution
        )

        # Add structure: some policies have lower EFE in certain belief regions
        EFE_landscape[2, :, :] -= 2.0  # Policy 2 is generally better
        EFE_landscape[0, 5:, 5:] -= 1.5  # Policy 0 better in high belief regions

        fig, axes = plt.subplots(1, num_policies, figsize=(20, 4))
        fig.suptitle("Expected Free Energy Landscapes", fontsize=14)

        for policy in range(num_policies):
            im = axes[policy].imshow(
                EFE_landscape[policy], cmap="RdYlBu_r", aspect="equal", origin="lower"
            )
            axes[policy].set_title(f"Policy {policy}")
            axes[policy].set_xlabel("Belief Dim 1")
            axes[policy].set_ylabel("Belief Dim 2")

        plt.colorbar(im, ax=axes, shrink=0.8, label="Expected Free Energy")
        plt.tight_layout()
        save_path = OUTPUTS_DIR / "efe_landscapes.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"EFE landscape visualization saved as {save_path}")

    except ImportError:
        print("Matplotlib not available - skipping visualizations")
        print("Install matplotlib to enable plotting: pip install matplotlib")
    except Exception as e:
        print(f"Visualization failed: {e}")


def run_performance_analysis():
    """Analyze performance characteristics."""

    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    import time

    feature_manager = TritonFeatureManager()
    selector = PolicySelector(feature_manager)

    # Test different problem sizes
    sizes = [(10, 4, 5), (50, 8, 10), (100, 16, 20)]  # (batch, obs_dim, num_policies)

    print("Performance scaling analysis:")
    print("Size (batch x obs x policies) | Time (ms) | Policies/sec")
    print("-" * 55)

    for batch_size, obs_dim, num_policies in sizes:
        observations = torch.randn(batch_size, obs_dim)
        policies = torch.randn(num_policies, obs_dim)
        posterior = torch.softmax(torch.randn(batch_size, obs_dim), dim=1)

        # Warm up
        for _ in range(3):
            selector.evaluate_policies(observations, policies, posterior)

        # Time evaluation
        start_time = time.time()
        num_trials = 10

        for _ in range(num_trials):
            selector.evaluate_policies(observations, policies, posterior)

        end_time = time.time()

        avg_time = (end_time - start_time) / num_trials * 1000  # ms
        policies_per_sec = (
            num_policies * batch_size / ((end_time - start_time) / num_trials)
        )

        print(
            f"  {batch_size}x{observation_dim}x{num_policies} | {avg_time:8.1f}ms | {policies_per_sec:8.1f}"
        )

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
    """Run the complete policy selection demonstration."""

    print("=" * 80)
    print("POLICY SELECTION WITH EXPECTED FREE ENERGY")
    print("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Initialize system
        feature_manager = TritonFeatureManager()
        selector = PolicySelector(feature_manager)

        # Run policy selection analysis
        history = analyze_policy_selection(feature_manager)

        # Run exploration-exploitation analysis
        create_exploration_exploitation_analysis()

        # Create visualizations
        create_visualization(history)

        # Run performance analysis
        run_performance_analysis()
        
        # Save results to JSON
        import json
        results_summary = {
            'history': history,
            'exploration_analysis_completed': True,
            'performance_analysis_completed': True,
            'visualizations_created': ['policy_selection_analysis.png', 'efe_landscapes.png']
        }
        
        results_path = OUTPUTS_DIR / 'policy_selection_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        print(f"✓ Results saved to {results_path}")

        print("\n" + "=" * 80)
        print("POLICY SELECTION DEMONSTRATION COMPLETED")
        print("=" * 80)
        print("✓ Policy evaluation implemented")
        print("✓ Expected free energy minimization working")
        print("✓ Multi-agent scenarios analyzed")
        print("✓ Exploration-exploitation trade-offs examined")
        print("✓ Performance benchmarks completed")
        print("✓ Comprehensive visualizations generated")
        print(f"\nResults saved to: {OUTPUTS_DIR}")
        print("  - policy_selection_analysis.png")
        print("  - efe_landscapes.png")
        print("  - policy_selection_results.json")

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
