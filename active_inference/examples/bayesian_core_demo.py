"""
Bayesian Core Demonstration

Complete demonstration of Bayesian inference core functionalities:
- Bayesian belief updating
- Information theoretic calculations
- Free energy computation
- Real data validation and logging
- Comprehensive visualization and reporting

This example shows the fundamental Bayesian operations that form the basis
of active inference and free energy minimization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
from pathlib import Path

# Import from the active_inference package using standardized imports
import active_inference as ai

# Set up outputs directory
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "bayesian_core"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def setup_demo_data():
    """Generate realistic demo data for Bayesian inference."""
    print("🔧 Setting up demonstration data...")

    # Create different types of priors
    n_samples = 8
    n_states = 6

    # Weak prior (high entropy)
    weak_prior = torch.ones(n_samples, n_states) / n_states

    # Strong prior (low entropy, concentrated on first state)
    strong_prior_logits = torch.zeros(n_samples, n_states)
    strong_prior_logits[:, 0] = 3.0  # Strong preference for first state
    strong_prior = torch.softmax(strong_prior_logits, dim=1)

    # Biased prior
    biased_prior = torch.softmax(torch.randn(n_samples, n_states) * 0.5, dim=1)

    priors = {
        'weak': weak_prior,
        'strong': strong_prior,
        'biased': biased_prior
    }

    # Generate realistic observations
    # Simulate different observation patterns
    np.random.seed(42)
    torch.manual_seed(42)

    # Pattern 1: Clear preference for first state
    obs_pattern_1 = torch.randn(n_samples, 3) + torch.tensor([2.0, 0.0, 0.0])

    # Pattern 2: Clear preference for middle state
    obs_pattern_2 = torch.randn(n_samples, 3) + torch.tensor([0.0, 2.0, 0.0])

    # Pattern 3: Mixed/noisy observations
    obs_pattern_3 = torch.randn(n_samples, 3) * 0.5

    observations = {
        'pattern_1': obs_pattern_1,
        'pattern_2': obs_pattern_2,
        'pattern_3': obs_pattern_3
    }

    print(f"   Generated {len(priors)} types of priors")
    print(f"   Generated {len(observations)} observation patterns")
    print(f"   Each with {n_samples} samples and {n_states} states")

    return priors, observations, n_states


def demonstrate_bayesian_updating():
    """Demonstrate core Bayesian belief updating."""
    print("\n🧠 Demonstrating Bayesian Belief Updating")
    print("=" * 50)

    priors, observations, n_states = setup_demo_data()
    core = ai.ai.create_bayesian_core(enable_logging=True, enable_validation=True, enable_visualization=True)

    # Test different combinations
    scenarios = [
        ('weak', 'pattern_1', 'Weak prior + Clear signal'),
        ('strong', 'pattern_1', 'Strong prior + Clear signal'),
        ('biased', 'pattern_2', 'Biased prior + Different signal'),
        ('weak', 'pattern_3', 'Weak prior + Noisy signal')
    ]

    results = {}

    for prior_type, obs_type, description in scenarios:
        print(f"\n📊 Scenario: {description}")
        print("-" * 40)

        prior = priors[prior_type]
        obs = observations[obs_type]

        # Generate likelihood from observations
        likelihood = core.get_observation_likelihood(obs, n_states, likelihood_type="categorical")

        # Perform Bayesian update
        start_time = time.time()
        result = core.bayesian_update(prior, likelihood, obs)
        update_time = time.time() - start_time

        # Display results
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

        # Store results
        results[f"{prior_type}_{obs_type}"] = result

        # Visualize if possible
        try:
            save_path = OUTPUTS_DIR / f"beliefs_{prior_type}_{obs_type}.png"
            core.visualize_beliefs(result, save_path=str(save_path), show_plot=False)
            print(f"   ✓ Visualization saved to {save_path}")
        except Exception as e:
            print(f"   ⚠ Visualization failed: {e}")

    return core, results


def demonstrate_information_calculations():
    """Demonstrate information theoretic calculations."""
    print("\n📏 Demonstrating Information Calculations")
    print("=" * 50)

    core = ai.ai.create_bayesian_core(enable_logging=False, enable_validation=True, enable_visualization=False)

    # Create test distributions
    batch_size, n_states = 4, 5

    # Uniform distribution (maximum entropy)
    uniform = torch.ones(batch_size, n_states) / n_states

    # Concentrated distribution (low entropy)
    concentrated_logits = torch.zeros(batch_size, n_states)
    concentrated_logits[:, 0] = 4.0  # Very concentrated on first state
    concentrated = torch.softmax(concentrated_logits, dim=1)

    # Mixed distribution
    mixed = torch.softmax(torch.randn(batch_size, n_states), dim=1)

    distributions = {
        'uniform': uniform,
        'concentrated': concentrated,
        'mixed': mixed
    }

    print("Entropy Analysis:")
    print("-" * 30)
    for name, dist in distributions.items():
        entropy = core.compute_entropy(dist)
        print("12")

    # KL divergence examples
    print("\nKL Divergence Analysis:")
    print("-" * 30)
    test_pairs = [
        ('uniform → concentrated', uniform, concentrated),
        ('concentrated → uniform', concentrated, uniform),
        ('mixed → uniform', mixed, uniform),
        ('uniform → uniform', uniform, uniform)
    ]

    for desc, p, q in test_pairs:
        kl = core.compute_kl_divergence(p, q)
        print("30")

    # Mutual information example
    print("\nMutual Information Analysis:")
    print("-" * 30)

    # Create a scenario where posterior is more concentrated than prior
    prior = torch.softmax(torch.randn(batch_size, n_states), dim=1)
    # Simulate likelihood that concentrates belief
    likelihood = torch.softmax(torch.randn(batch_size, n_states) + torch.randn(batch_size, 1) * 2, dim=1)
    posterior = core.bayesian_update(prior, likelihood).posterior
    mutual_info = core.compute_mutual_information(prior, posterior, likelihood)

    print(".4f")
    print(".4f")
    print(".4f")

    return distributions


def demonstrate_free_energy_computation():
    """Demonstrate variational free energy computation."""
    print("\n⚡ Demonstrating Free Energy Computation")
    print("=" * 50)

    core = ai.create_bayesian_core(enable_logging=True, enable_validation=True, enable_visualization=False)

    # Create test scenario
    batch_size, n_states = 6, 4

    # Different prior concentrations
    prior_types = ['weak', 'medium', 'strong']
    priors = {}

    for prior_type in prior_types:
        if prior_type == 'weak':
            prior = torch.ones(batch_size, n_states) / n_states
        elif prior_type == 'medium':
            prior = torch.softmax(torch.randn(batch_size, n_states) * 0.5, dim=1)
        else:  # strong
            prior_logits = torch.randn(batch_size, n_states) * 2.0
            prior = torch.softmax(prior_logits, dim=1)
        priors[prior_type] = prior

    # Generate observations and likelihoods
    observations = torch.randn(batch_size, 2)
    likelihood = core.get_observation_likelihood(observations, n_states)

    print("Free Energy Analysis:")
    print("-" * 30)

    results = {}
    for prior_type, prior in priors.items():
        result = core.bayesian_update(prior, likelihood, observations)
        results[prior_type] = result

        print(f"{prior_type.capitalize():>8} prior: FE = {result.free_energy.mean().item():.4f} ± {result.free_energy.std().item():.4f}")

    # Analyze free energy components
    print("\nFree Energy Components (first sample):")
    print("-" * 40)
    result = results['medium']
    sample_idx = 0

    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    return core, results


def demonstrate_validation_and_logging():
    """Demonstrate validation and comprehensive logging."""
    print("\n✅ Demonstrating Validation and Logging")
    print("=" * 50)

    # Create core with full logging
    core = ai.create_bayesian_core(enable_logging=True, enable_validation=True, enable_visualization=False)

    print("Running validation tests...")

    # Test 1: Valid inputs
    print("\n1. Testing valid inputs:")
    prior = torch.ones(3, 4) / 4
    likelihood = torch.softmax(torch.randn(3, 4), dim=1)

    try:
        result = core.bayesian_update(prior, likelihood)
        print("   ✓ Valid inputs accepted")
        print(".4f")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    # Test 2: Invalid inputs (should be caught by validation)
    print("\n2. Testing invalid inputs:")

    # NaN values
    prior_nan = torch.tensor([[0.5, float('nan')], [0.3, 0.7]])
    try:
        core.bayesian_update(prior_nan, likelihood[:2])
        print("   ✗ NaN values not caught")
    except ValueError:
        print("   ✓ NaN values properly rejected")

    # Negative values
    prior_neg = torch.tensor([[0.5, -0.1], [0.6, 0.5]])
    try:
        core.bayesian_update(prior_neg, likelihood[:2])
        print("   ✗ Negative values not caught")
    except ValueError:
        print("   ✓ Negative values properly rejected")

    # Wrong shapes
    prior_wrong = torch.ones(2, 3)  # Wrong number of states
    try:
        core.bayesian_update(prior_wrong, likelihood[:2])
        print("   ✗ Shape mismatch not caught")
    except ValueError:
        print("   ✓ Shape mismatch properly rejected")

    # Show validation statistics
    print("\nValidation Statistics:")
    print("-" * 30)
    stats = core.validation_stats
    print(f"Total updates attempted: {stats['total_updates']}")
    print(f"Successful updates: {stats['successful_updates']}")
    print(f"Failed updates: {stats['failed_updates']}")
    if stats['computation_times']:
        print(".4f")
        print(".4f")

    if stats['validation_errors']:
        print(f"Validation errors: {len(stats['validation_errors'])}")
        print("Sample errors:")
        for i, error in enumerate(stats['validation_errors'][:3]):
            print(f"  {i+1}. {error}")

    return core


def demonstrate_reporting_and_visualization():
    """Demonstrate comprehensive reporting and visualization."""
    print("\n📊 Demonstrating Reporting and Visualization")
    print("=" * 50)

    core = ai.create_bayesian_core(enable_logging=True, enable_validation=True, enable_visualization=True)

    # Run multiple inference operations
    print("Running inference operations for reporting...")

    batch_size, n_states = 5, 4
    n_operations = 8

    for i in range(n_operations):
        # Vary the scenarios
        prior = torch.softmax(torch.randn(batch_size, n_states) * (i % 3), dim=1)
        likelihood = torch.softmax(torch.randn(batch_size, n_states), dim=1)
        observations = torch.randn(batch_size, 2)

        result = core.bayesian_update(prior, likelihood, observations)

        if (i + 1) % 4 == 0:
            print(f"   Completed {i+1}/{n_operations} operations")

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")

    # Save report to file
    report_path = OUTPUTS_DIR / "bayesian_core_demo_report.json"
    report = core.generate_report(save_path=str(report_path), include_history=True)

    print(f"✓ Report generated and saved to {report_path}")

    # Display key statistics
    print("\nReport Summary:")
    print("-" * 20)
    print(f"Total inference operations: {report['validation_stats']['total_updates']}")
    print(f"Successful operations: {report['validation_stats']['successful_updates']}")
    print(f"Convergence rate: {report['history_summary'].get('convergence_rate', 'N/A')}")

    if report['history_summary']:
        hist = report['history_summary']
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

    if report['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

    # Create visualizations
    print("\nCreating visualizations...")

    if len(core.history.updates) >= 2:
        # Create a summary visualization of the inference history
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Bayesian Inference History Summary', fontsize=14)

            # Free energy over time
            axes[0, 0].plot(core.history.free_energy_history, 'b-o', linewidth=2)
            axes[0, 0].set_title('Free Energy Evolution')
            axes[0, 0].set_xlabel('Update Step')
            axes[0, 0].set_ylabel('Free Energy')
            axes[0, 0].grid(True, alpha=0.3)

            # Entropy evolution
            axes[0, 1].plot(core.history.entropy_history, 'r-s', linewidth=2)
            axes[0, 1].set_title('Posterior Entropy Evolution')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Entropy')
            axes[0, 1].grid(True, alpha=0.3)

            # KL divergence evolution
            axes[1, 0].plot(core.history.kl_history, 'g-^', linewidth=2)
            axes[1, 0].set_title('KL Divergence Evolution')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('KL Divergence')
            axes[1, 0].grid(True, alpha=0.3)

            # Evidence evolution
            axes[1, 1].plot(core.history.evidence_history, 'm-d', linewidth=2)
            axes[1, 1].set_title('Evidence Evolution')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Evidence')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = OUTPUTS_DIR / 'inference_history_summary.png'
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ History summary visualization saved as {save_path}")

        except Exception as e:
            print(f"⚠ Visualization creation failed: {e}")

    return core, report


def run_complete_demonstration():
    """Run the complete Bayesian core demonstration."""
    print("🚀 BAYESIAN CORE COMPLETE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases all core Bayesian inference functionalities:")
    print("• Bayesian belief updating with real data validation")
    print("• Information theoretic calculations (entropy, KL divergence, mutual information)")
    print("• Variational free energy computation and minimization")
    print("• Comprehensive logging and error handling")
    print("• Visualization and detailed reporting")
    print("=" * 60)

    start_time = time.time()

    try:
        # 1. Bayesian Updating Demonstration
        core, update_results = demonstrate_bayesian_updating()

        # 2. Information Calculations Demonstration
        info_distributions = demonstrate_information_calculations()

        # 3. Free Energy Demonstration
        fe_core, fe_results = demonstrate_free_energy_computation()

        # 4. Validation and Logging Demonstration
        validation_core = demonstrate_validation_and_logging()

        # 5. Reporting and Visualization Demonstration
        reporting_core, final_report = demonstrate_reporting_and_visualization()

        # Final summary
        total_time = time.time() - start_time

        print(f"\n🎉 DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(".1f")
        print(f"Total inference operations: {len(core.history.updates)}")
        print(f"Successful operations: {core.validation_stats['successful_updates']}")
        print(f"Generated visualizations in: {OUTPUTS_DIR}")
        print(f"  - inference_history_summary.png")
        print(f"  - beliefs_*.png files")
        print(f"Generated reports in: {OUTPUTS_DIR}")
        print(f"  - bayesian_core_demo_report.json")
        print("=" * 60)

        print("\n📋 SUMMARY OF DEMONSTRATED CAPABILITIES:")
        print("✓ Bayesian belief updating (prior → likelihood → posterior)")
        print("✓ Information theoretic calculations (entropy, mutual information, KL divergence)")
        print("✓ Variational free energy computation and analysis")
        print("✓ Real data validation and error handling")
        print("✓ Comprehensive logging of all operations")
        print("✓ Visualization of belief distributions and evolution")
        print("✓ Detailed reporting and performance analysis")
        print("✓ Robust handling of edge cases and numerical stability")

        return {
            'update_results': update_results,
            'info_distributions': info_distributions,
            'fe_results': fe_results,
            'validation_stats': validation_core.validation_stats,
            'final_report': final_report,
            'total_time': total_time
        }

    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete demonstration
    results = run_complete_demonstration()

    if results:
        print("\n✅ All demonstrations completed successfully!")
        print(f"Results saved to: {OUTPUTS_DIR}")
        print(f"  - bayesian_core_demo_report.json")
        print(f"  - inference_history_summary.png")
        print(f"  - beliefs_*.png files")
    else:
        print("\n❌ Demonstration encountered errors.")
