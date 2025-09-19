#!/usr/bin/env python3
"""
Enhanced Visualization Utilities for Active Inference POMDP

This module provides comprehensive visualization functions for Active Inference
in POMDP environments with enhanced data export and validation capabilities.

Key Features:
- Trajectory visualization with time progression
- Belief state evolution plots
- EFE value analysis over time
- Performance metrics dashboards
- Generative model matrix visualizations
- Animation-style trajectory plots
- Raw data export for investigation (_raw_data files)
- Validation reports for data integrity (_report files)

Functions:
    create_trajectory_plot: Plot agent trajectory with true vs estimated positions
    create_belief_evolution_plot: Show belief state changes over time
    create_efe_analysis_plot: Analyze expected free energy values
    create_performance_dashboard: Comprehensive performance metrics
    create_generative_model_animation: Animated generative model flow
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from datetime import datetime


def save_raw_data(data: Dict[str, Any], filename: str, output_dir: Path) -> None:
    """
    Save raw data to JSON file for investigation.

    Args:
        data: Data dictionary to save
        filename: Base filename (without extension)
        output_dir: Directory to save files
    """
    raw_data_path = output_dir / f"{filename}_raw_data.json"

    # Convert numpy/torch arrays to lists for JSON serialization
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            processed_data[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            processed_data[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(value, (list, tuple)):
            processed_data[key] = list(value)
        else:
            processed_data[key] = value

    with open(raw_data_path, 'w') as f:
        json.dump(processed_data, f, indent=2, default=str)

    print(f"✓ Raw data saved to: {raw_data_path}")


def create_validation_report(data: Dict[str, Any], report_type: str, output_dir: Path) -> None:
    """
    Create validation report for data integrity and analysis.

    Args:
        data: Data to validate and analyze
        report_type: Type of report (e.g., 'trajectory', 'belief', 'performance')
        output_dir: Directory to save report
    """
    report_path = output_dir / f"{report_type}_report.json"

    report = {
        "report_type": report_type,
        "timestamp": datetime.now().isoformat(),
        "validation_summary": {},
        "data_analysis": {},
        "quality_metrics": {},
        "recommendations": []
    }

    # Basic validation checks
    if report_type == "trajectory":
        report = validate_trajectory_data(data, report)
    elif report_type == "belief":
        report = validate_belief_data(data, report)
    elif report_type == "performance":
        report = validate_performance_data(data, report)

    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"✓ Validation report saved to: {report_path}")


def validate_trajectory_data(data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate trajectory data integrity."""
    history = data.get('history', {})
    config = data.get('config', {})

    # Data completeness checks
    required_fields = ['timestep', 'position_i', 'position_j', 'estimated_position_i', 'estimated_position_j']
    missing_fields = [field for field in required_fields if field not in history]

    report["validation_summary"]["data_completeness"] = {
        "status": "PASS" if not missing_fields else "FAIL",
        "missing_fields": missing_fields,
        "total_timesteps": len(history.get('timestep', []))
    }

    # Position bounds validation
    if 'position_i' in history and 'position_j' in history:
        grid_size = config.get('grid_size', 10)
        positions_i = history['position_i']
        positions_j = history['position_j']

        bounds_check = all(0 <= pos < grid_size for pos in positions_i + positions_j)
        report["validation_summary"]["position_bounds"] = {
            "status": "PASS" if bounds_check else "FAIL",
            "grid_size": grid_size,
            "min_position": min(positions_i + positions_j) if positions_i else None,
            "max_position": max(positions_i + positions_j) if positions_i else None
        }

    # Trajectory analysis
    if 'position_i' in history and 'estimated_position_i' in history:
        true_positions = list(zip(history['position_i'], history['position_j']))
        est_positions = list(zip(history['estimated_position_i'], history['estimated_position_j']))

        position_errors = []
        for true_pos, est_pos in zip(true_positions, est_positions):
            error = np.linalg.norm(np.array(true_pos) - np.array(est_pos))
            position_errors.append(error)

        report["data_analysis"]["trajectory_metrics"] = {
            "total_positions": len(true_positions),
            "unique_true_positions": len(set(true_positions)),
            "unique_estimated_positions": len(set(est_positions)),
            "mean_position_error": float(np.mean(position_errors)),
            "std_position_error": float(np.std(position_errors)),
            "max_position_error": float(np.max(position_errors))
        }

    # Quality recommendations
    if report["validation_summary"]["data_completeness"]["status"] == "PASS":
        report["recommendations"].append("Trajectory data is complete and valid")
    else:
        report["recommendations"].append("Complete missing trajectory fields for full analysis")

    return report


def validate_belief_data(data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate belief state data integrity."""
    history = data.get('history', {})

    # Entropy validation
    if 'belief_entropy' in history:
        entropy_values = history['belief_entropy']
        entropy_stats = {
            "mean_entropy": float(np.mean(entropy_values)),
            "std_entropy": float(np.std(entropy_values)),
            "min_entropy": float(np.min(entropy_values)),
            "max_entropy": float(np.max(entropy_values)),
            "entropy_trend": "decreasing" if entropy_values[-1] < entropy_values[0] else "stable"
        }
        report["data_analysis"]["entropy_analysis"] = entropy_stats

        # Validate entropy range (should be between 0 and log(total_states))
        total_states = data.get('config', {}).get('total_states', 500)
        max_entropy = np.log(total_states)
        valid_range = all(0 <= e <= max_entropy for e in entropy_values)
        report["validation_summary"]["entropy_range"] = {
            "status": "PASS" if valid_range else "FAIL",
            "expected_max": float(max_entropy),
            "actual_range": [float(np.min(entropy_values)), float(np.max(entropy_values))]
        }

    # EFE validation
    if 'efe_values' in history:
        efe_values = history['efe_values']
        all_efe = [val for timestep_efe in efe_values for val in timestep_efe]

        report["data_analysis"]["efe_analysis"] = {
            "total_efe_values": len(all_efe),
            "mean_efe": float(np.mean(all_efe)),
            "std_efe": float(np.std(all_efe)),
            "min_efe": float(np.min(all_efe)),
            "max_efe": float(np.max(all_efe))
        }

    return report


def validate_performance_data(data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """Validate performance metrics data integrity."""
    perf = data.get('performance_metrics', {})

    # Metric validation
    required_metrics = ['total_time', 'avg_time_per_step', 'total_reward', 'position_accuracy']
    missing_metrics = [metric for metric in required_metrics if metric not in perf]

    report["validation_summary"]["metrics_completeness"] = {
        "status": "PASS" if not missing_metrics else "FAIL",
        "missing_metrics": missing_metrics
    }

    # Performance analysis
    if 'position_accuracy' in perf:
        accuracy = perf['position_accuracy']
        report["data_analysis"]["performance_analysis"] = {
            "position_accuracy": accuracy,
            "performance_rating": "EXCELLENT" if accuracy > 0.8 else "GOOD" if accuracy > 0.6 else "NEEDS_IMPROVEMENT"
        }

    if 'action_distribution' in perf:
        action_dist = np.array(perf['action_distribution'])
        report["data_analysis"]["action_analysis"] = {
            "action_distribution": action_dist.tolist(),
            "most_frequent_action": int(np.argmax(action_dist)),
            "action_entropy": float(-np.sum(action_dist * np.log(action_dist + 1e-10))),
            "exploration_balance": "BALANCED" if np.std(action_dist) < 0.2 else "BIASED"
        }

    return report


def create_trajectory_plot(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create comprehensive trajectory visualization with raw data export and validation.

    Args:
        results: Simulation results dictionary
        output_dir: Directory to save plots
    """
    history = results['history']
    config = results['simulation_config']

    # Prepare raw data for export
    true_positions = list(zip(history['position_i'], history['position_j']))
    estimated_positions = list(zip(history['estimated_position_i'], history['estimated_position_j']))

    position_errors = []
    for i in range(len(true_positions)):
        true_pos = np.array(true_positions[i])
        est_pos = np.array(estimated_positions[i])
        error = np.linalg.norm(true_pos - est_pos)
        position_errors.append(error)

    actions = history['action']
    action_counts = np.bincount(actions, minlength=config['n_actions'])

    # Raw data for export
    trajectory_raw_data = {
        'config': config,
        'true_positions': true_positions,
        'estimated_positions': estimated_positions,
        'position_errors': position_errors,
        'temperature_states': history['temperature_state'],
        'actions': actions,
        'action_counts': action_counts.tolist(),
        'action_labels': ['Up', 'Down', 'Left', 'Right'],
        'timesteps': history['timestep']
    }

    # Save raw data
    save_raw_data(trajectory_raw_data, "trajectory_analysis", output_dir)

    # Create validation report
    validation_data = {
        'history': history,
        'config': config
    }
    create_validation_report(validation_data, "trajectory", output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Agent Trajectory and Position Estimation", fontsize=16, fontweight='bold')

    # Plot 1: True vs Estimated Trajectory
    ax = axes[0, 0]
    true_x, true_y = zip(*true_positions)
    est_x, est_y = zip(*estimated_positions)

    ax.plot(true_x, true_y, 'b-o', linewidth=2, markersize=4, label='True Position', alpha=0.8)
    ax.plot(est_x, est_y, 'r--s', linewidth=2, markersize=4, label='Estimated Position', alpha=0.8)

    ax.set_xlim(0, config['grid_size']-1)
    ax.set_ylim(0, config['grid_size']-1)
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_title('True vs Estimated Trajectory', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Position Estimation Error
    ax = axes[0, 1]
    ax.plot(history['timestep'], position_errors, 'orange', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position Error (Euclidean Distance)')
    ax.set_title('Position Estimation Error Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Temperature State Evolution
    ax = axes[1, 0]
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    temp_states = history['temperature_state']
    for temp_state in range(config['n_temperature_states']):
        indices = [i for i, state in enumerate(temp_states) if state == temp_state]
        if indices:
            ax.scatter(indices, [temp_state] * len(indices), c=colors[temp_state],
                      label=f'Temp {temp_state}', s=50, alpha=0.7)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Temperature State')
    ax.set_title('Temperature State Evolution', fontweight='bold')
    ax.set_yticks(range(config['n_temperature_states']))
    ax.set_yticklabels(['Very Cold', 'Cold', 'Optimal', 'Warm', 'Hot'])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Action Distribution
    ax = axes[1, 1]
    action_labels = ['Up', 'Down', 'Left', 'Right']

    bars = ax.bar(range(config['n_actions']), action_counts, color='skyblue', alpha=0.7)
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Selection Distribution', fontweight='bold')
    ax.set_xticks(range(config['n_actions']))
    ax.set_xticklabels(action_labels)

    # Add value labels on bars
    for bar, count in zip(bars, action_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Trajectory analysis saved to: {output_dir / 'trajectory_analysis.png'}")


def create_belief_evolution_plot(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create belief state evolution visualization with raw data export and validation.

    Args:
        results: Simulation results dictionary
        output_dir: Directory to save plots
    """
    history = results['history']
    config = results['simulation_config']

    # Prepare raw data for export
    entropy = history['belief_entropy']
    efe_values = history['efe_values']
    selected_actions = history['action']
    rewards = history['reward']
    cumulative_reward = np.cumsum(rewards)

    true_positions = list(zip(history['position_i'], history['position_j']))
    estimated_positions = list(zip(history['estimated_position_i'], history['estimated_position_j']))

    position_accuracy = []
    for true_pos, est_pos in zip(true_positions, estimated_positions):
        accuracy = 1.0 if true_pos == est_pos else 0.0
        position_accuracy.append(accuracy)

    # Process EFE data by action
    efe_by_action = {}
    for action in range(config['n_actions']):
        action_mask = np.array(selected_actions) == action
        if np.any(action_mask):
            action_indices = np.where(action_mask)[0]
            action_efe_values = []
            for t in action_indices:
                if t < len(efe_values) and action < len(efe_values[t]):
                    action_efe_values.append(efe_values[t][action])
            efe_by_action[f'action_{action}'] = {
                'indices': action_indices.tolist(),
                'values': action_efe_values
            }

    # Ensure selected_actions is a list for JSON serialization
    if isinstance(selected_actions, np.ndarray):
        selected_actions = selected_actions.tolist()
    elif not isinstance(selected_actions, list):
        selected_actions = list(selected_actions)

    # Raw data for export
    belief_raw_data = {
        'config': config,
        'belief_entropy': entropy,
        'efe_values': efe_values,
        'efe_by_action': efe_by_action,
        'cumulative_reward': cumulative_reward.tolist(),
        'position_accuracy': position_accuracy,
        'true_positions': true_positions,
        'estimated_positions': estimated_positions,
        'selected_actions': selected_actions,
        'timesteps': history['timestep']
    }

    # Ensure data is JSON serializable
    for key in ['true_positions', 'estimated_positions', 'timesteps', 'position_accuracy']:
        value = belief_raw_data[key]
        if isinstance(value, np.ndarray):
            belief_raw_data[key] = value.tolist()
        elif not isinstance(value, list):
            belief_raw_data[key] = list(value)

    # Save raw data
    save_raw_data(belief_raw_data, "belief_evolution", output_dir)

    # Create validation report
    validation_data = {
        'history': history,
        'config': config
    }
    create_validation_report(validation_data, "belief", output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Belief State Evolution Analysis", fontsize=16, fontweight='bold')

    # Plot 1: Belief Entropy Evolution
    ax = axes[0, 0]
    ax.plot(history['timestep'], entropy, 'purple', linewidth=2, marker='o', markersize=4)
    ax.fill_between(history['timestep'], entropy, alpha=0.3, color='purple')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Belief Entropy')
    ax.set_title('Belief Uncertainty Evolution', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: EFE Values Over Time
    ax = axes[0, 1]
    for action in range(config['n_actions']):
        action_mask = np.array(selected_actions) == action
        if np.any(action_mask):
            action_indices = np.where(action_mask)[0]
            action_efe_values = []
            for t in action_indices:
                if t < len(efe_values) and action < len(efe_values[t]):
                    action_efe_values.append(efe_values[t][action])

            if action_efe_values:  # Only plot if we have values
                ax.scatter(action_indices[:len(action_efe_values)], action_efe_values,
                          label=f'Action {action}', alpha=0.7, s=30)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Expected Free Energy')
    ax.set_title('EFE Values Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Reward Accumulation
    ax = axes[1, 0]
    ax.plot(history['timestep'], cumulative_reward, 'green', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Reward Accumulation', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Position Accuracy
    ax = axes[1, 1]
    ax.plot(history['timestep'], position_accuracy, 'orange', linewidth=2, marker='^', markersize=4)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position Accuracy')
    ax.set_title('Position Estimation Accuracy', fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "belief_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Belief evolution analysis saved to: {output_dir / 'belief_evolution.png'}")


def create_performance_dashboard(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create comprehensive performance metrics dashboard with raw data export and validation.

    Args:
        results: Simulation results dictionary
        output_dir: Directory to save plots
    """
    perf = results['performance_metrics']

    # Raw data for export
    performance_raw_data = {
        'performance_metrics': perf,
        'simulation_config': results['simulation_config'],
        'key_metrics': {
            'total_time': perf['total_time'],
            'avg_time_per_step': perf['avg_time_per_step'],
            'total_reward': perf['total_reward'],
            'avg_reward_per_step': perf['avg_reward_per_step'],
            'final_belief_entropy': perf['final_belief_entropy'],
            'position_accuracy': perf['position_accuracy']
        },
        'action_distribution': perf['action_distribution'],
        'action_labels': ['Up', 'Down', 'Left', 'Right'],
        'belief_entropy_trend': perf['belief_entropy_trend'],
        'accuracies': {
            'position_accuracy': perf['position_accuracy'],
            'temperature_accuracy': perf.get('temperature_accuracy', 0.0),
            'final_position_accuracy': perf['final_position_accuracy'],
            'final_temperature_accuracy': perf['final_temperature_accuracy']
        },
        'efe_statistics': {
            'efe_final_min': perf.get('efe_final_min', 0.0),
            'efe_final_std': perf.get('efe_final_std', 0.0)
        }
    }

    # Save raw data
    save_raw_data(performance_raw_data, "performance_dashboard", output_dir)

    # Create validation report
    create_validation_report(results, "performance", output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Active Inference Performance Dashboard", fontsize=16, fontweight='bold')

    # Plot 1: Key Metrics Summary
    ax = axes[0, 0]
    metrics = ['Total Time', 'Avg Time/Step', 'Total Reward', 'Avg Reward/Step', 'Final Entropy', 'Position Accuracy']
    values = [
        perf['total_time'],
        perf['avg_time_per_step'],
        perf['total_reward'],
        perf['avg_reward_per_step'],
        perf['final_belief_entropy'],
        perf['position_accuracy']
    ]

    bars = ax.barh(range(len(metrics)), values, color='lightcoral', alpha=0.7)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Value')
    ax.set_title('Performance Metrics', fontweight='bold')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + max(values) * 0.01, i, f'{value:.3f}',
               va='center', fontweight='bold')

    # Plot 2: Action Distribution
    ax = axes[0, 1]
    action_distribution = perf['action_distribution']
    action_labels = ['Up', 'Down', 'Left', 'Right']

    bars = ax.bar(range(len(action_distribution)), action_distribution, color='skyblue', alpha=0.7)
    ax.set_xticks(range(len(action_distribution)))
    ax.set_xticklabels(action_labels)
    ax.set_ylabel('Proportion')
    ax.set_title('Action Selection Distribution', fontweight='bold')

    # Add percentage labels
    for bar, prop in zip(bars, action_distribution):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{prop:.1%}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Belief Entropy Trend
    ax = axes[0, 2]
    entropy_trend = perf['belief_entropy_trend']
    timesteps = list(range(len(entropy_trend)))

    ax.plot(timesteps, entropy_trend, 'purple', linewidth=2, marker='o', markersize=3)
    ax.fill_between(timesteps, entropy_trend, alpha=0.3, color='purple')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Belief Entropy')
    ax.set_title('Belief Uncertainty Trend', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Accuracy Metrics
    ax = axes[1, 0]
    accuracies = ['Position\nAccuracy', 'Temperature\nAccuracy', 'Final Position\nAccuracy', 'Final Temperature\nAccuracy']
    acc_values = [
        perf['position_accuracy'],
        perf.get('temperature_accuracy', 0.0),
        perf['final_position_accuracy'],
        perf['final_temperature_accuracy']
    ]

    bars = ax.bar(range(len(accuracies)), acc_values, color='lightgreen', alpha=0.7)
    ax.set_xticks(range(len(accuracies)))
    ax.set_xticklabels(accuracies)
    ax.set_ylabel('Accuracy')
    ax.set_title('Estimation Accuracies', fontweight='bold')
    ax.set_ylim(0, 1)

    # Add percentage labels
    for bar, acc in zip(bars, acc_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

    # Plot 5: EFE Analysis
    ax = axes[1, 1]
    efe_min = perf.get('efe_final_min', 0.0)
    efe_std = perf.get('efe_final_std', 0.0)

    # Create a simple bar chart for EFE statistics
    ax.bar(['Min EFE', 'Std EFE'], [efe_min, efe_std], color=['orange', 'red'], alpha=0.7)
    ax.set_ylabel('EFE Value')
    ax.set_title('Final EFE Statistics', fontweight='bold')

    # Add value labels
    ax.text(0, efe_min + 0.01, f'{efe_min:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.text(1, efe_std + 0.01, f'{efe_std:.4f}', ha='center', va='bottom', fontweight='bold')

    # Plot 6: Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
    ACTIVE INFERENCE SUMMARY

    Simulation Duration: {perf['total_time']:.2f}s
    Average Step Time: {perf['avg_time_per_step']:.4f}s

    Final Performance:
    • Belief Entropy: {perf['final_belief_entropy']:.4f}
    • Position Accuracy: {perf['position_accuracy']:.1%}
    • Total Reward: {perf['total_reward']:.4f}

    Action Preferences:
    • Most Used: {np.argmax(action_distribution)}
    • Exploration Rate: {np.std(action_distribution):.3f}
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "performance_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Performance dashboard saved to: {output_dir / 'performance_dashboard.png'}")


def create_trajectory_animation(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create animated trajectory visualization with raw data export and validation.

    Args:
        results: Simulation results dictionary
        output_dir: Directory to save plots
    """
    history = results['history']
    config = results['simulation_config']

    # Prepare raw data for export
    true_positions = list(zip(history['position_i'], history['position_j']))
    estimated_positions = list(zip(history['estimated_position_i'], history['estimated_position_j']))

    # Calculate temperature preference grid
    grid_size = config['grid_size']
    temp_grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            center_dist = np.sqrt((i - grid_size//2)**2 + (j - grid_size//2)**2)
            temp_grid[i, j] = max(0, 2 - center_dist * 0.3)  # Higher in center

    # Raw data for export
    trajectory_animation_raw_data = {
        'config': config,
        'true_positions': true_positions,
        'estimated_positions': estimated_positions,
        'timesteps': history['timestep'],
        'temperature_preference_grid': temp_grid.tolist(),
        'grid_size': grid_size
    }

    # Ensure data is JSON serializable
    for key in ['true_positions', 'estimated_positions', 'timesteps']:
        value = trajectory_animation_raw_data[key]
        if isinstance(value, np.ndarray):
            trajectory_animation_raw_data[key] = value.tolist()
        elif not isinstance(value, list):
            trajectory_animation_raw_data[key] = list(value)

    # Save raw data
    save_raw_data(trajectory_animation_raw_data, "trajectory_evolution", output_dir)

    # Create validation report
    validation_data = {
        'history': history,
        'config': config
    }
    create_validation_report(validation_data, "trajectory", output_dir)

    plt.figure(figsize=(12, 8))

    plt.imshow(temp_grid, cmap='RdYlBu_r', alpha=0.3, extent=[0, grid_size-1, 0, grid_size-1])

    # Plot trajectory with time progression
    true_x, true_y = zip(*true_positions)
    est_x, est_y = zip(*estimated_positions)
    timesteps = history['timestep']

    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))

    for i, t in enumerate(timesteps):
        if i > 0:
            plt.plot([true_x[i-1], true_x[i]], [true_y[i-1], true_y[i]],
                    color=colors[i], linewidth=2, alpha=0.7)
            plt.plot([est_x[i-1], est_x[i]], [est_y[i-1], est_y[i]],
                    color=colors[i], linewidth=2, linestyle='--', alpha=0.7)

    # Mark start and end points
    plt.scatter(true_x[0], true_y[0], color='green', s=200, marker='*', label='Start', zorder=10)
    plt.scatter(true_x[-1], true_y[-1], color='red', s=200, marker='X', label='End', zorder=10)

    plt.xlabel('Grid X Position')
    plt.ylabel('Grid Y Position')
    plt.title('POMDP Agent Trajectory with Active Inference', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add colorbar for time progression
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(timesteps)-1))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Time Progression')

    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Trajectory evolution saved to: {output_dir / 'trajectory_evolution.png'}")


def create_comprehensive_visualization_suite(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create complete visualization suite for Active Inference results with comprehensive data export.

    Args:
        results: Simulation results dictionary
        output_dir: Directory to save plots
    """
    print("\n📊 Generating Comprehensive Visualization Suite...")

    create_trajectory_plot(results, output_dir)
    create_belief_evolution_plot(results, output_dir)
    create_performance_dashboard(results, output_dir)
    create_trajectory_animation(results, output_dir)

    print("\n📋 Data Export Summary:")
    print("=" * 60)
    print("📁 Raw Data Files (for investigation):")
    print(f"  • trajectory_analysis_raw_data.json")
    print(f"  • belief_evolution_raw_data.json")
    print(f"  • performance_dashboard_raw_data.json")
    print(f"  • trajectory_evolution_raw_data.json")
    print()
    print("📋 Validation Reports (data integrity):")
    print(f"  • trajectory_report.json")
    print(f"  • belief_report.json")
    print(f"  • performance_report.json")
    print()
    print("📊 Visualization Files:")
    print(f"  • trajectory_analysis.png")
    print(f"  • belief_evolution.png")
    print(f"  • performance_dashboard.png")
    print(f"  • trajectory_evolution.png")
    print("=" * 60)

    print("✅ Complete visualization suite generated with enhanced data export!")


# Export functions for external use
__all__ = [
    'create_trajectory_plot',
    'create_belief_evolution_plot',
    'create_performance_dashboard',
    'create_trajectory_animation',
    'create_comprehensive_visualization_suite'
]
