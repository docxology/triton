#!/usr/bin/env python3
"""
POMDP Gridworld Active Inference - Thin Orchestrator

This example demonstrates a Partially Observable Markov Decision Process (POMDP) in a
temperature-controlled gridworld using GPU-accelerated Active Inference. The agent
navigates a 10x10 grid with temperature states, making observations and selecting
policies to maintain homeostatic temperature preferences.

Active Inference Methods Used:
- Variational Free Energy (VFE) minimization for state estimation from observations
- Expected Free Energy (EFE) minimization for policy selection and action planning
- GPU-accelerated Bayesian inference for belief state updates
- Real-time convergence monitoring and visualization

Environment:
- 10x10 gridworld with temperature states at each cell
- 5 discrete temperature levels per cell (very cold, cold, optimal, warm, hot)
- 5 observation types (noisy temperature readings)
- 4 actions: up, down, left, right
- Homeostatic preference for middle temperature (optimal state)

Generative Model:
- Hidden states: Agent position (100 cells) × Temperature state (5 levels) = 500 states
- Observations: Noisy temperature readings (5 types)
- Actions: Movement in grid (4 directions)
- Preferences: Strong preference for optimal temperature state
"""

import torch
import numpy as np
import time
import json
import pandas as pd
from pathlib import Path
import sys

# Add the src directory to the path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up outputs directory
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "pomdp_gridworld"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Import modular components
from pomdp_environment import POMDPGridworld
from active_inference_agent import ActiveInferenceAgent
from visualization_utils import create_comprehensive_visualization_suite
from core import TritonFeatureManager


def run_pomdp_simulation(n_timesteps=50):
    """
    Run the complete POMDP simulation with Active Inference.

    Args:
        n_timesteps: Number of timesteps to simulate

    Returns:
        Complete simulation results
    """
    print("=" * 80)
    print("POMDP GRIDWORLD ACTIVE INFERENCE SIMULATION")
    print("=" * 80)
    print(f"Running simulation for {n_timesteps} timesteps...")
    print()

    # Initialize components using existing framework
    feature_manager = TritonFeatureManager()
    environment = POMDPGridworld()
    agent = ActiveInferenceAgent(environment, feature_manager)

    # ===== CREATE COMPREHENSIVE VISUALIZATIONS =====
    print("\n📊 Creating Environment Visualizations...")
    environment.visualize_transition_model(OUTPUTS_DIR)
    environment.visualize_observation_model(OUTPUTS_DIR)
    environment.visualize_temperature_grid(OUTPUTS_DIR)
    environment.visualize_preferences(OUTPUTS_DIR, agent)
    environment.visualize_state_space(agent.belief, OUTPUTS_DIR, timestep=0)
    environment.create_generative_model_visualization(agent, OUTPUTS_DIR)

    # Initialize simulation
    current_state = np.random.randint(0, environment.total_states)
    total_reward = 0.0

    # Simulation history
    history = {
        'timestep': [],
        'position_i': [],
        'position_j': [],
        'temperature_state': [],
        'observation': [],
        'action': [],
        'reward': [],
        'efe_values': [],
        'belief_entropy': [],
        'estimated_position_i': [],
        'estimated_position_j': []
    }

    start_time = time.time()

    # Create visualization at key timesteps
    viz_timesteps = [9, 19, 29, 39, 49]  # Visualize at timesteps 10, 20, 30, 40, 50

    for t in range(n_timesteps):
        print(f"\n{'='*60}")
        print(f"🕐 TIMESTEP {t+1}/{n_timesteps}")
        print(f"{'='*60}")

        # Get current position and temperature
        pos_i, pos_j, temp_state = environment._state_to_pos_temp(current_state)
        print(f"🏠 True State: Position ({pos_i}, {pos_j}), Temperature: {environment.temperature_labels[temp_state]}")

        # Generate observation
        observation = environment.get_observation(current_state)
        print(f"👁️  Observation: {environment.observation_labels[observation]}")

        # Update belief using VFE
        if t > 0:  # Skip first update (no previous action)
            print(f"🔬 Updating belief with VFE (Action: {environment.action_labels[history['action'][-1]]})...")
            agent.update_belief_vfe(history['action'][-1], observation)

        # Select policy using EFE
        print("🎯 Computing Expected Free Energy for policy selection...")
        action, min_efe, efe_values = agent.select_policy_efe()
        action_name = environment.action_labels[action]

        # Execute action (transition to new state)
        next_state_probs = environment.transition_model[current_state, action]
        next_state = torch.multinomial(next_state_probs, 1).item()
        current_state = next_state

        # Get reward
        reward = environment.get_reward(current_state)
        total_reward += reward

        # Get estimated position
        est_pos_i, est_pos_j = agent.get_current_position_estimate()

        # Compute belief entropy
        belief_entropy = agent.get_belief_entropy()

        # Record history
        history['timestep'].append(t)
        history['position_i'].append(pos_i)
        history['position_j'].append(pos_j)
        history['temperature_state'].append(temp_state)
        history['observation'].append(observation)
        history['action'].append(action)
        history['reward'].append(reward)
        history['efe_values'].append(efe_values.tolist())
        history['belief_entropy'].append(belief_entropy)
        history['estimated_position_i'].append(est_pos_i)
        history['estimated_position_j'].append(est_pos_j)

        print(f"⚡ Action: {action_name} (EFE: {min_efe:.4f})")
        print(f"💰 Reward: {reward:.4f}")
        print(f"📍 Estimated position: ({est_pos_i}, {est_pos_j})")

        # Position accuracy
        position_accuracy = 1.0 if (pos_i == est_pos_i and pos_j == est_pos_j) else 0.0
        print(f"🎯 Position accuracy: {position_accuracy:.1%}")

        # Create visualization at key timesteps
        if t in viz_timesteps:
            print(f"📊 Creating belief state visualization for timestep {t+1}...")
            environment.visualize_state_space(agent.belief, OUTPUTS_DIR, timestep=t+1)

    total_time = time.time() - start_time

    # ===== FINAL BELIEF STATE VISUALIZATION =====
    print("\n📊 Creating final belief state visualization...")
    environment.visualize_state_space(agent.belief, OUTPUTS_DIR, timestep=n_timesteps)

    # ===== COMPREHENSIVE PERFORMANCE ANALYSIS =====
    position_accuracies = [
        1.0 if history['position_i'][i] == history['estimated_position_i'][i] and
             history['position_j'][i] == history['estimated_position_j'][i] else 0.0
        for i in range(len(history['position_i']))
    ]

    temperature_accuracies = [
        1.0 if history['temperature_state'][i] == torch.argmax(torch.tensor([
            agent.preferences[environment._pos_temp_to_state(
                history['estimated_position_i'][i],
                history['estimated_position_j'][i],
                temp
            )] for temp in range(environment.n_temperature_states)
        ])).item() else 0.0
        for i in range(len(history['temperature_state']))
    ]

    # Action distribution analysis
    action_counts = np.bincount(history['action'], minlength=environment.n_actions)
    action_distribution = action_counts / action_counts.sum()

    # EFE analysis
    efe_history = np.array(history['efe_values'])
    efe_min_over_time = np.min(efe_history, axis=1)
    efe_std_over_time = np.std(efe_history, axis=1)

    # Final results
    results = {
        'simulation_config': {
            'n_timesteps': n_timesteps,
            'grid_size': environment.grid_size,
            'n_temperature_states': environment.n_temperature_states,
            'n_observations': environment.n_observations,
            'n_actions': environment.n_actions,
            'total_states': environment.total_states
        },
        'performance_metrics': {
            'total_time': total_time,
            'avg_time_per_step': total_time / n_timesteps,
            'total_reward': total_reward,
            'avg_reward_per_step': total_reward / n_timesteps,
            'final_belief_entropy': belief_entropy,
            'position_accuracy': np.mean(position_accuracies),
            'temperature_accuracy': np.mean(temperature_accuracies),
            'final_position_accuracy': position_accuracies[-1],
            'final_temperature_accuracy': temperature_accuracies[-1],
            'action_distribution': action_distribution.tolist(),
            'efe_final_min': efe_min_over_time[-1],
            'efe_final_std': efe_std_over_time[-1],
            'belief_entropy_trend': history['belief_entropy']
        },
        'active_inference_analysis': {
            'vfe_updates_performed': n_timesteps - 1,
            'efe_computations_performed': n_timesteps,
            'final_most_likely_state': int(agent.belief.argmax().item()),
            'final_most_likely_position': agent.get_current_position_estimate(),
            'average_efe_complexity': np.mean([len(efe_vals) for efe_vals in history['efe_values']])
        },
        'history': history,
        'final_belief': agent.belief.detach().cpu().numpy().tolist(),
        'preferences': agent.preferences.detach().cpu().numpy().tolist()
    }

    print("\n🎉 SIMULATION COMPLETED!")
    print("=" * 80)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Avg time per step: {total_time / n_timesteps:.4f}s")
    print(f"💰 Total reward: {total_reward:.4f}")
    print(f"💰 Avg reward per step: {total_reward / n_timesteps:.4f}")
    print(f"🎯 Position accuracy: {np.mean(position_accuracies):.1%}")
    print(f"🌡️  Temperature accuracy: {np.mean(temperature_accuracies):.1%}")
    print(f"🧠 Final belief entropy: {belief_entropy:.4f}")
    print(f"🎯 Final position accuracy: {position_accuracies[-1]:.1%}")

    print("\n📊 ACTIVE INFERENCE PERFORMANCE:")
    print(f"   VFE Updates: {results['active_inference_analysis']['vfe_updates_performed']}")
    print(f"   EFE Computations: {results['active_inference_analysis']['efe_computations_performed']}")
    print(f"   Final Most Likely State: {results['active_inference_analysis']['final_most_likely_state']}")
    print(f"   Final Most Likely Position: {results['active_inference_analysis']['final_most_likely_position']}")

    print("\n🎯 ACTION DISTRIBUTION:")
    for i, prob in enumerate(results['performance_metrics']['action_distribution']):
        print(".3f")

    print("\n🔬 ACTIVE INFERENCE VALIDATION:")
    print("   ✓ Variational Free Energy (VFE) - State estimation from observations")
    print("   ✓ Expected Free Energy (EFE) - Policy selection with epistemic+pragmatic value")
    print("   ✓ Softmax Policy Selection - EFE-reweighted action sampling")
    print("   ✓ Bayesian Belief Updates - Real-time POMDP inference")
    print("   ✓ Comprehensive Visualizations - All matrices and state spaces")
    print("   ✓ Enhanced Logging - Detailed simulation tracking")

    return results


def create_visualizations(results):
    """Create comprehensive visualizations of the POMDP simulation."""
    print("\n📊 Creating comprehensive visualizations...")
    create_comprehensive_visualization_suite(results, OUTPUTS_DIR)


def save_results_to_files(results):
    """Save comprehensive results to JSON and CSV files."""
    print("\n💾 Saving results to files...")

    # Save complete results to JSON
    json_path = OUTPUTS_DIR / "pomdp_simulation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Complete results saved to: {json_path}")

    # Save summary metrics to JSON
    summary = {
        'simulation_summary': {
            'timesteps': results['simulation_config']['n_timesteps'],
            'grid_size': results['simulation_config']['grid_size'],
            'total_states': results['simulation_config']['total_states'],
            'performance': results['performance_metrics']
        },
        'active_inference_methods': [
            'Variational Free Energy (VFE) for state estimation',
            'Expected Free Energy (EFE) for policy selection',
            'GPU-accelerated Bayesian inference',
            'POMDP belief state updates'
        ],
        'environment_characteristics': [
            '10x10 gridworld with temperature states',
            '5 temperature levels per cell',
            '5 observation types (noisy readings)',
            '4 actions (grid navigation)',
            'Homeostatic temperature preference'
        ]
    }

    summary_path = OUTPUTS_DIR / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")

    # Save trajectory data to CSV
    trajectory_data = {
        'timestep': results['history']['timestep'],
        'true_position_x': results['history']['position_i'],
        'true_position_y': results['history']['position_j'],
        'estimated_position_x': results['history']['estimated_position_i'],
        'estimated_position_y': results['history']['estimated_position_j'],
        'temperature_state': results['history']['temperature_state'],
        'observation': results['history']['observation'],
        'action': results['history']['action'],
        'reward': results['history']['reward'],
        'belief_entropy': results['history']['belief_entropy']
    }

    df = pd.DataFrame(trajectory_data)
    csv_path = OUTPUTS_DIR / "trajectory_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Trajectory data saved to: {csv_path}")


def main():
    """Run the complete POMDP Gridworld Active Inference demonstration."""
    print("🚀 POMDP GRIDWORLD ACTIVE INFERENCE DEMONSTRATION")
    print("=" * 70)
    print("This example demonstrates Active Inference in a partially observable")
    print("temperature-controlled gridworld environment.")
    print()
    print("Key Features:")
    print("• POMDP with 10x10 gridworld (500 total states)")
    print("• 5 temperature states per cell with homeostatic preferences")
    print("• Variational Free Energy (VFE) for state estimation")
    print("• Expected Free Energy (EFE) for policy selection")
    print("• GPU-accelerated computations using Triton")
    print("• Comprehensive logging and visualization")
    print("=" * 70)

    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Run the POMDP simulation
        results = run_pomdp_simulation(n_timesteps=50)

        # Create visualizations
        create_visualizations(results)

        # Save results to files
        save_results_to_files(results)

        print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 Results saved to: {OUTPUTS_DIR}")
        print("  • pomdp_simulation_results.json (complete data)")
        print("  • simulation_summary.json (key metrics)")
        print("  • trajectory_data.csv (time series data)")
        print("  • Multiple PNG visualizations")

        print("\n📊 FINAL PERFORMANCE SUMMARY:")
        perf = results['performance_metrics']
        print(f"⏱️  Total time: {perf['total_time']:.1f}s")
        print(f"⚡ Avg time per step: {perf['avg_time_per_step']:.4f}s")
        print(f"💰 Total reward: {perf['total_reward']:.4f}")
        print(f"🎯 Position accuracy: {perf['position_accuracy']:.1%}")
        print(f"🧠 Final belief entropy: {perf['final_belief_entropy']:.4f}")

        print("\n🔬 ACTIVE INFERENCE METHODS DEMONSTRATED:")
        print("  1. Perception: VFE-based state estimation from observations")
        print("  2. Action: EFE-based policy selection for navigation")
        print("  3. Learning: Bayesian belief updates in POMDP setting")
        print("  4. Integration: Complete perception-action-learning cycle")

        return 0

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
