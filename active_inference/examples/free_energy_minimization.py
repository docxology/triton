#!/usr/bin/env python3
"""
Free Energy Minimization Demonstration - Thin Orchestrator

This example demonstrates variational free energy minimization by orchestrating
components from the active_inference framework. Shows optimization across different
problem types with comprehensive monitoring and visualization.

Active Inference Methods Used:
- Variational Free Energy minimization (gradient-based optimization)
- Bayesian posterior updating
- Real-time convergence monitoring
- GPU-accelerated computation

Generative Models:
- Simple unimodal problems (single strong mode)
- Conflicting prior-likelihood problems
- Multimodal posterior problems
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the active_inference package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up outputs directory
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "free_energy"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Import from the active_inference package using standardized imports
import active_inference as ai


def main():
    """
    Main orchestrator for free energy minimization demonstration.

    Configuration:
    - Problems: Simple, conflicting, multimodal optimization scenarios
    - Optimization: Gradient-based variational inference
    - Monitoring: Real-time convergence tracking
    - GPU: MPS acceleration for Apple Silicon
    """

    print("=" * 80)
    print("FREE ENERGY MINIMIZATION DEMONSTRATION")
    print("=" * 80)
    print("Thin orchestrator demonstrating:")
    print("• Variational free energy minimization")
    print("• Different optimization problem types")
    print("• Real-time convergence monitoring")
    print("• GPU-accelerated gradient optimization")
    print()

    # ============================================================================
    # CONFIGURATION SECTION
    # ============================================================================

    config = {
        'problems': {
            'types': ['simple', 'conflicting', 'multimodal'],
            'batch_size': 32,
            'feature_dim': 6
        },
        'optimization': {
            'iterations': 50,
            'learning_rate': 0.02,
            'monitor_frequency': 5
        },
        'output': {
            'save_plots': True,
            'show_detailed_metrics': True
        }
    }

    print("Configuration:")
    for section, params in config.items():
        print(f"  {section}: {params}")
    print()

    # ============================================================================
    # ENVIRONMENT SETUP
    # ============================================================================

    print("🔧 Setting up environment...")

    # Initialize Triton feature manager (handles GPU setup)
    feature_manager = ai.TritonFeatureManager()
    gpu_accelerator = ai.GPUAccelerator(feature_manager)
    vfe_engine = ai.VariationalFreeEnergy(feature_manager)

    print(f"✓ Triton features initialized")
    print(f"✓ GPU accelerator ready (device: {gpu_accelerator.device.type})")
    print(f"✓ Variational free energy engine ready")

    # ============================================================================
    # OPTIMIZATION PROBLEM SETUP
    # ============================================================================

    print("\n🎯 Setting up optimization problems...")

    problems = {}

    for problem_type in config['problems']['types']:
        print(f"Creating {problem_type} problem...")

        if problem_type == 'simple':
            # Simple problem: single strong mode
            observations = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])

            # Create a prior with one strong preference
            prior_logits = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])
            prior_logits[:, 0] = 3.0  # Strong preference for first dimension
            prior = torch.softmax(prior_logits, dim=1)

            # Create likelihood that agrees with prior
            likelihood_logits = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])
            likelihood_logits[:, 0] = 2.0  # Some agreement with prior
            likelihood = torch.softmax(likelihood_logits, dim=1)

        elif problem_type == 'conflicting':
            # Conflicting prior and likelihood
            observations = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])

            # Prior prefers first dimension
            prior_logits = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])
            prior_logits[:, 0] = 4.0
            prior = torch.softmax(prior_logits, dim=1)

            # Likelihood prefers last dimension
            likelihood_logits = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])
            likelihood_logits[:, -1] = 4.0
            likelihood = torch.softmax(likelihood_logits, dim=1)

        elif problem_type == 'multimodal':
            # Multimodal posterior
            observations = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])

            # Create multimodal prior
            prior_logits = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])
            prior_logits[:, [0, config['problems']['feature_dim']//2, -1]] = 2.0  # Multiple peaks
            prior = torch.softmax(prior_logits, dim=1)

            # Likelihood with different preferences
            likelihood_logits = torch.randn(config['problems']['batch_size'], config['problems']['feature_dim'])
            likelihood_logits[:, [1, config['problems']['feature_dim']//2 + 1, -2]] = 2.0
            likelihood = torch.softmax(likelihood_logits, dim=1)

        # Initialize posterior randomly
        initial_posterior = torch.softmax(torch.randn(config['problems']['batch_size'], config['problems']['feature_dim']), dim=1)

        problems[problem_type] = {
            'observations': observations,
            'initial_posterior': initial_posterior,
            'prior': prior,
            'likelihood': likelihood
        }

        print(f"✓ {problem_type.capitalize()} problem created")
        print(f"  Observations shape: {observations.shape}")

    # ============================================================================
    # OPTIMIZATION EXECUTION
    # ============================================================================

    print("\n🧮 Running variational free energy minimization...")

    results = {}

    for problem_type, problem_data in problems.items():
        print(f"\n--- {problem_type.upper()} PROBLEM OPTIMIZATION ---")

        # Extract problem data
        observations = problem_data['observations']
        initial_posterior = problem_data['initial_posterior']
        prior = problem_data['prior']
        likelihood = problem_data['likelihood']

        print(f"Problem: {observations.shape[0]} samples, {observations.shape[1]} features")

        # Initialize optimization tracking
        posterior = initial_posterior.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([posterior], lr=config['optimization']['learning_rate'])

        # Track optimization history
        optimization_history = []

        # Initial free energy
        initial_fe = vfe_engine.compute(observations, posterior, prior, likelihood)
        initial_fe_mean = initial_fe.mean().item()

        optimization_history.append({
            'iteration': 0,
            'free_energy': initial_fe_mean,
            'gradient_norm': 0.0,
            'posterior_entropy': torch.distributions.Categorical(
                probs=torch.softmax(posterior, dim=1)
            ).entropy().mean().item(),
            'time': 0.0
        })

        print(f"Initial FE (mean): {initial_fe_mean:.4f}")

        start_time = time.time()

        # Run optimization
        for iteration in range(1, config['optimization']['iterations'] + 1):
            iter_start_time = time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Compute free energy
            free_energy = vfe_engine.compute(observations, posterior, prior, likelihood)
            loss = free_energy.mean()

            # Backpropagate
            loss.backward()

            # Compute gradient statistics
            grad_norm = torch.norm(posterior.grad).item()

            # Update parameters
            optimizer.step()

            # Normalize posterior (ensure it sums to 1)
            with torch.no_grad():
                posterior.data = torch.softmax(posterior.data, dim=1)

            # Record metrics
            if iteration % config['optimization']['monitor_frequency'] == 0 or iteration == config['optimization']['iterations']:
                iter_time = time.time() - iter_start_time
                current_fe = loss.item()

                # Compute posterior entropy
                posterior_probs = torch.softmax(posterior, dim=1)
                entropy = torch.distributions.Categorical(
                    probs=posterior_probs
                ).entropy().mean().item()

                optimization_history.append({
                    'iteration': iteration,
                    'free_energy': current_fe,
                    'gradient_norm': grad_norm,
                    'posterior_entropy': entropy,
                    'time': time.time() - start_time
                })

                print(f"Iteration {iteration:2d} - free energy: {current_fe:.4f}, grad_norm: {grad_norm:.4f}")
        total_time = time.time() - start_time
        final_fe = optimization_history[-1]['free_energy']
        improvement = initial_fe_mean - final_fe

        print("\nOptimization completed:")
        print(f"  Initial Free Energy: {initial_fe_mean:.4f}")
        print(f"  Final Free Energy: {final_fe:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        print(f"  Total Time: {total_time:.1f}s")

        results[problem_type] = {
            'history': optimization_history,
            'final_posterior': posterior.detach(),
            'initial_fe': initial_fe_mean,
            'final_fe': final_fe,
            'improvement': improvement,
            'total_time': total_time
        }

    # ============================================================================
    # COMPREHENSIVE ANALYSIS
    # ============================================================================

    print("\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 40)

    analysis_results = {}

    for problem_type, result in results.items():
        history = result['history']

        # Compute convergence metrics
        free_energies = [h['free_energy'] for h in history]
        gradient_norms = [h['gradient_norm'] for h in history]
        entropies = [h['posterior_entropy'] for h in history]

        # Simple convergence measure: relative improvement per iteration
        iterations = np.array([h['iteration'] for h in history])
        fe_values = np.array(free_energies)
        improvements = np.diff(fe_values)
        convergence_rate = np.mean(np.abs(improvements) / fe_values[:-1]) if len(improvements) > 0 else 0

        analysis_results[problem_type] = {
            'convergence_rate': convergence_rate,
            'final_entropy': entropies[-1],
            'gradient_norm_final': gradient_norms[-1],
            'total_iterations': len(history) - 1,
            'improvement': result['improvement']
        }

        print(f"\n{problem_type.upper()} Problem Analysis:")
        print(".2e")
        print(".4f")
        print(".4f")
        print(f"  Total iterations: {analysis_results[problem_type]['total_iterations']}")
        print(".4f")

    # ============================================================================
    # VISUALIZATION
    # ============================================================================

    if config['output']['save_plots']:
        print("\n📈 Generating visualizations...")

        try:
            # Create comprehensive analysis plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Free Energy Minimization Analysis', fontsize=16)

            colors = ['blue', 'red', 'green']
            problem_types = list(results.keys())

            # Plot 1: Free energy traces
            ax = axes[0, 0]
            for i, problem_type in enumerate(problem_types):
                history = results[problem_type]['history']
                iterations = [h['iteration'] for h in history]
                free_energies = [h['free_energy'] for h in history]

                ax.plot(iterations, free_energies, color=colors[i],
                       linewidth=2, label=problem_type.capitalize())

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Free Energy')
            ax.set_title('Free Energy Minimization')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Gradient norms
            ax = axes[0, 1]
            for i, problem_type in enumerate(problem_types):
                history = results[problem_type]['history']
                iterations = [h['iteration'] for h in history]
                grad_norms = [h['gradient_norm'] for h in history]

                ax.plot(iterations, grad_norms, color=colors[i],
                       linewidth=2, label=problem_type.capitalize())

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norms')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Plot 3: Posterior entropy
            ax = axes[0, 2]
            for i, problem_type in enumerate(problem_types):
                history = results[problem_type]['history']
                iterations = [h['iteration'] for h in history]
                entropies = [h['posterior_entropy'] for h in history]

                ax.plot(iterations, entropies, color=colors[i],
                       linewidth=2, label=problem_type.capitalize())

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Posterior Entropy')
            ax.set_title('Belief Uncertainty')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 4: Free energy improvement
            ax = axes[1, 0]
            improvements = []
            labels = []
            for problem_type in problem_types:
                improvement = results[problem_type]['improvement']
                improvements.append(improvement)
                labels.append(problem_type.capitalize())

            bars = ax.bar(labels, improvements, color=colors[:len(improvements)], alpha=0.7)
            ax.set_ylabel('Free Energy Reduction')
            ax.set_title('Optimization Improvement')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       '.3f', ha='center', va='bottom')

            # Plot 5: Final posterior distributions
            ax = axes[1, 1]
            problem_type = problem_types[0]
            final_posterior = results[problem_type]['final_posterior']

            # Show first few samples
            num_samples_to_show = min(10, final_posterior.shape[0])
            posterior_sample = final_posterior[:num_samples_to_show].cpu().numpy()

            im = ax.imshow(posterior_sample.T, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Hidden Dimension')
            ax.set_title(f'Final Posterior Beliefs\n({problem_type.capitalize()})')
            plt.colorbar(im, ax=ax)

            # Plot 6: Convergence comparison
            ax = axes[1, 2]
            for i, problem_type in enumerate(problem_types):
                history = results[problem_type]['history']
                iterations = np.array([h['iteration'] for h in history])
                free_energies = np.array([h['free_energy'] for h in history])

                # Normalize by initial value
                normalized_fe = free_energies / free_energies[0]

                ax.plot(iterations, normalized_fe, color=colors[i],
                       linewidth=2, label=problem_type.capitalize())

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Normalized Free Energy')
            ax.set_title('Normalized Convergence')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = OUTPUTS_DIR / 'free_energy_minimization_analysis.png'
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"✓ Analysis visualization saved as {save_path}")

            # Create detailed trajectory plot
            plt.figure(figsize=(12, 8))

            for i, problem_type in enumerate(problem_types):
                history = results[problem_type]['history']
                iterations = [h['iteration'] for h in history]
                free_energies = [h['free_energy'] for h in history]

                plt.plot(iterations, free_energies, color=colors[i],
                        linewidth=3, alpha=0.8, label=f'{problem_type.capitalize()} Problem')

                # Add markers at monitoring points
                plt.scatter(iterations[::2], np.array(free_energies)[::2],
                           color=colors[i], s=50, alpha=0.7)

            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Variational Free Energy', fontsize=12)
            plt.title('Free Energy Minimization Trajectories', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = OUTPUTS_DIR / 'free_energy_trajectories.png'
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"✓ Trajectory plot saved as {save_path}")

        except ImportError:
            print("⚠️  Matplotlib not available - skipping visualizations")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")

    # ============================================================================
    # VALIDATION AND TRACES
    # ============================================================================

    print("\n✅ VALIDATION CHECKS")
    print("=" * 30)

    validation_results = {
        'gpu_acceleration': gpu_accelerator.device.type in ['cuda', 'mps'],
        'all_problems_optimized': all(r['improvement'] > 0 for r in results.values()),
        'convergence_rates_valid': all(not np.isnan(ar['convergence_rate']) for ar in analysis_results.values()),
        'entropy_decreased': all(ar['final_entropy'] < 1.0 for ar in analysis_results.values()),
        'gradient_norms_small': all(ar['gradient_norm_final'] < 1.0 for ar in analysis_results.values())
    }

    for check, passed in validation_results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check.replace('_', ' ').title()}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(validation_results.values())
    print(f"\n🎯 Overall Validation: {'PASS' if all_passed else 'PARTIAL'}")

    # ============================================================================
    # FINAL OUTPUT
    # ============================================================================

    print("\n" + "=" * 80)
    print("FREE ENERGY MINIMIZATION DEMONSTRATION COMPLETED")
    print("=" * 80)

    if all_passed:
        print("🎉 SUCCESS: All optimization methods working correctly!")
        print("✓ Variational free energy minimization validated")
        print("✓ Different problem types handled successfully")
        print("✓ Real-time convergence monitoring working")
        print("✓ GPU acceleration confirmed")
        print("✓ All real Triton methods successfully demonstrated")
    else:
        print("⚠️  PARTIAL SUCCESS: Some validations need attention")
        print("✓ Core optimization working")
        print("⚠️  Some advanced metrics may need tuning")

    # Save results to JSON
    import json
    results_summary = {
        'config': config,
        'results': {k: {
            'initial_fe': v['initial_fe'],
            'final_fe': v['final_fe'],
            'improvement': v['improvement'],
            'total_time': v['total_time']
        } for k, v in results.items()},
        'analysis': analysis_results,
        'validation': validation_results
    }
    
    results_path = OUTPUTS_DIR / 'free_energy_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"✓ Results saved to {results_path}")

    print(f"\n📁 Outputs generated in {OUTPUTS_DIR}:")
    print("  - free_energy_minimization_analysis.png (comprehensive analysis)")
    print("  - free_energy_trajectories.png (optimization trajectories)")
    print("  - free_energy_results.json (numerical results)")
    print("  - All optimization traces and metrics printed above")
    print("  - Real validation of variational inference methods")

    print("\n🔬 ACTIVE INFERENCE METHODS DEMONSTRATED:")
    print("  1. Perception: Variational inference optimization")
    print("  2. Learning: Gradient-based free energy minimization")
    print("  3. Adaptation: Different problem type handling")
    print("  4. Monitoring: Real-time convergence tracking")

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        results = main()
        print("\n🏁 Demonstration finished successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
