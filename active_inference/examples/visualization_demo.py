#!/usr/bin/env python3
"""
Comprehensive Visualization Demo for Active Inference Framework

This script demonstrates all the visualization types and output formats
that the enhanced run_all_examples.py and run_all_tests.py scripts generate.

It creates:
- Performance dashboards with multiple charts
- Belief evolution heatmaps and time series
- Statistical analysis plots
- Network topology visualizations
- Correlation matrices
- Timeline and dependency graphs
- Triton-specific ecosystem analysis

Usage:
    python visualization_demo.py
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, Patch
from matplotlib.collections import PatchCollection
from datetime import datetime

# Add the active_inference package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up outputs directory
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "visualization"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Import from the active_inference package using standardized imports
import active_inference as ai

# Set style for all plots
plt.style.use('default')
sns.set_palette("husl")


def create_sample_data():
    """Create sample data for visualization demo."""
    # Sample test results
    results = {
        "core_gpu": {
            "device_type": "mps",
            "matrix_mult_time_ms": 1.923,
            "tensor_allocation": True,
            "success": True
        },
        "free_energy": {
            "vfe_mean": 0.671,
            "efe_mean": -0.121,
            "initial_fe": 0.763,
            "final_fe": 0.365,
            "improvement": 0.398,
            "computation_time": 0.022,
            "success": True
        },
        "belief_tracking": {
            "belief_history": [
                [0.027] * 36,  # Simplified
                [0.028] * 36,
                [0.029] * 36,
                [0.030] * 36,
                [0.031] * 36
            ],
            "entropy_history": [3.58, 3.57, 3.56, 3.55, 3.54],
            "most_likely_history": [0, 1, 2, 3, 4],
            "entropy_change": 0.04,
            "total_steps": 5,
            "success": True
        },
        "triton_integration": {
            "triton_available": True,
            "triton_version": "3.5.0",
            "kernel_registration": True,
            "triton_kernel_test": True,
            "device_type": "mps",
            "success": True
        },
        "performance_validation": {
            "pytorch_baseline": {
                "vector_add_time": 0.000183,
                "size": 10000
            },
            "success": True
        }
    }

    return results


def demo_performance_dashboard():
    """Demonstrate comprehensive performance dashboard."""
    print("📊 Creating Performance Dashboard...")

    results = create_sample_data()

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Active Inference Framework - Performance Dashboard", fontsize=16, fontweight='bold')

    # Plot 1: Success Rate Overview
    test_names = list(results.keys())
    success_status = [1 if results[name].get("success", False) else 0 for name in test_names]
    success_count = sum(success_status)
    fail_count = len(success_status) - success_count

    axes[0, 0].pie([success_count, fail_count], labels=['Passed', 'Failed'],
                   autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
    axes[0, 0].set_title("Success Rate", fontweight='bold')

    # Plot 2: Performance Timeline
    success_rates = []
    cumulative_success = 0
    for i, (name, result) in enumerate(results.items()):
        if result.get("success", False):
            cumulative_success += 1
        success_rates.append(cumulative_success / (i + 1) * 100)

    axes[0, 1].plot(range(len(success_rates)), success_rates, 'g-', linewidth=3, marker='o', markersize=6)
    axes[0, 1].set_xlabel('Test Index')
    axes[0, 1].set_ylabel('Cumulative Success Rate (%)')
    axes[0, 1].set_title("Success Progression", fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 105)

    # Plot 3: Computation Times
    times = []
    labels = []
    for name, result in results.items():
        if result and "computation_time" in result:
            times.append(result["computation_time"])
            labels.append(name.replace('_', ' ').title())

    if times:
        bars = axes[0, 2].bar(range(len(labels)), times, color='skyblue', alpha=0.8)
        axes[0, 2].set_xticks(range(len(labels)))
        axes[0, 2].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].set_title("Computation Times", fontweight='bold')

    # Plot 4: Free Energy Optimization
    if results.get("free_energy"):
        fe_data = results["free_energy"]
        if "initial_fe" in fe_data and "final_fe" in fe_data:
            stages = ['Initial', 'Final']
            values = [fe_data["initial_fe"], fe_data["final_fe"]]

            bars = axes[1, 0].bar(stages, values, color=['#FF5722', '#4CAF50'], alpha=0.8)
            axes[1, 0].set_ylabel('Free Energy Value')
            axes[1, 0].set_title("Free Energy Optimization", fontweight='bold')

    # Plot 5: Belief Entropy Evolution
    if results.get("belief_tracking"):
        entropy_data = results["belief_tracking"]
        if "entropy_history" in entropy_data:
            entropy_history = entropy_data["entropy_history"]
            axes[1, 1].plot(range(len(entropy_history)), entropy_history, 'b-', linewidth=2, marker='o')
            axes[1, 1].fill_between(range(len(entropy_history)), entropy_history, alpha=0.3, color='blue')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Belief Entropy')
            axes[1, 1].set_title("Belief Entropy Evolution", fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Triton Performance
    if results.get("triton_integration"):
        triton_data = results["triton_integration"]
        perf_labels = []
        perf_values = []

        if triton_data.get("triton_available", False):
            perf_labels.append("Triton Available")
            perf_values.append(1)

        if triton_data.get("kernel_registration", False):
            perf_labels.append("Kernel Registration")
            perf_values.append(1)

        if perf_values:
            bars = axes[1, 2].bar(range(len(perf_labels)), perf_values, color='blue', alpha=0.7)
            axes[1, 2].set_xticks(range(len(perf_labels)))
            axes[1, 2].set_xticklabels(perf_labels, rotation=45, ha='right')
            axes[1, 2].set_title("Triton Integration Status", fontweight='bold')
            axes[1, 2].set_ylabel("Status (1=Working)")

    # Plot 7: Statistical Summary
    summary_stats = [
        len([r for r in results.values() if r is not None]),  # Completed
        len(results),  # Total
        len([r for r in results.values() if r and r.get("success", False)]),  # Successful
        len([r for r in results.values() if r and "computation_time" in r]),  # With timing
    ]

    axes[2, 0].pie(summary_stats, labels=['Completed', 'Total', 'Successful', 'With Timing'],
                   autopct='%1.1f%%', colors=sns.color_palette("pastel", 4), startangle=90)
    axes[2, 0].set_title("Results Summary", fontweight='bold')

    # Plot 8: Performance Comparison
    if results.get("performance_validation"):
        perf_data = results["performance_validation"]
        if "pytorch_baseline" in perf_data:
            baseline = perf_data["pytorch_baseline"]
            if "vector_add_time" in baseline:
                axes[2, 1].bar(['PyTorch Baseline'], [baseline["vector_add_time"]], color='orange', alpha=0.7)
                axes[2, 1].set_ylabel('Time (seconds)')
                axes[2, 1].set_title("Performance Baseline", fontweight='bold')

    # Plot 9: Component Health
    health_metrics = [
        ('GPU Ops', 1 if results.get("core_gpu", {}).get("success", False) else 0),
        ('Free Energy', 1 if results.get("free_energy", {}).get("success", False) else 0),
        ('Belief Tracking', 1 if results.get("belief_tracking", {}).get("success", False) else 0),
        ('Triton', 1 if results.get("triton_integration", {}).get("success", False) else 0),
    ]

    names = [name for name, _ in health_metrics]
    values = [value for _, value in health_metrics]

    bars = axes[2, 2].bar(range(len(names)), values, color='green', alpha=0.7)
    axes[2, 2].set_xticks(range(len(names)))
    axes[2, 2].set_xticklabels(names, rotation=45, ha='right')
    axes[2, 2].set_ylabel('Status (1=Working)')
    axes[2, 2].set_title("Component Health", fontweight='bold')
    axes[2, 2].set_ylim(0, 1.2)

    plt.tight_layout()
    save_path = OUTPUTS_DIR / "comprehensive_performance_dashboard.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    print(f"✓ Performance dashboard saved as: {save_path}")


def demo_belief_evolution_analysis():
    """Demonstrate belief evolution visualization."""
    print("🔍 Creating Belief Evolution Analysis...")

    results = create_sample_data()

    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle("Belief State Evolution Analysis", fontsize=16, fontweight='bold')

    if results.get("belief_tracking"):
        belief_data = results["belief_tracking"]

        # Heatmap
        if "belief_history" in belief_data:
            belief_history = np.array(belief_data["belief_history"])
            if belief_history.shape[1] > 50:
                indices = np.linspace(0, belief_history.shape[1]-1, 50, dtype=int)
                belief_history = belief_history[:, indices]

            axes[0, 0].imshow(belief_history.T, aspect='auto', cmap='viridis')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('State Index')
            axes[0, 0].set_title('Belief State Heatmap', fontweight='bold')

        # Entropy evolution
        if "entropy_history" in belief_data:
            entropy_history = belief_data["entropy_history"]
            axes[0, 1].plot(range(len(entropy_history)), entropy_history, 'b-', linewidth=2, marker='o')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Entropy')
            axes[0, 1].set_title('Entropy Evolution', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)

        # State estimation
        if "most_likely_history" in belief_data:
            most_likely_history = belief_data["most_likely_history"]
            axes[0, 2].plot(range(len(most_likely_history)), most_likely_history, 'r-', linewidth=2, marker='s')
            axes[0, 2].set_xlabel('Time Step')
            axes[0, 2].set_ylabel('Most Likely State')
            axes[0, 2].set_title('State Estimation', fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3)

        # Statistics
        if "entropy_history" in belief_data:
            stats_labels = ['Initial', 'Final', 'Change', 'Steps']
            stats_values = [
                entropy_history[0],
                entropy_history[-1],
                entropy_history[0] - entropy_history[-1],
                len(entropy_history)
            ]
            bars = axes[1, 0].bar(range(len(stats_labels)), stats_values, color='purple', alpha=0.7)
            axes[1, 0].set_xticks(range(len(stats_labels)))
            axes[1, 0].set_xticklabels(stats_labels)
            axes[1, 0].set_title('Evolution Statistics', fontweight='bold')

        # Distribution comparison
        if "belief_history" in belief_data and len(belief_history) >= 2:
            start_beliefs = belief_history[0]
            end_beliefs = belief_history[-1]
            axes[1, 1].hist([start_beliefs, end_beliefs], bins=10, alpha=0.7, label=['Start', 'End'])
            axes[1, 1].set_xlabel('Belief Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Belief Distribution', fontweight='bold')
            axes[1, 1].legend()

        # Convergence
        if "entropy_history" in belief_data and len(entropy_history) > 1:
            convergence = np.diff(entropy_history)
            axes[1, 2].plot(range(len(convergence)), convergence, 'g-', marker='x')
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].set_xlabel('Time Step')
            axes[1, 2].set_ylabel('Entropy Change')
            axes[1, 2].set_title('Convergence Rate', fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUTS_DIR / "belief_evolution_comprehensive.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    print(f"✓ Belief evolution analysis saved as: {save_path}")


def demo_network_topology():
    """Demonstrate network topology visualization."""
    print("🕸️ Creating Network Topology Visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Network Topology and Dependencies", fontsize=16, fontweight='bold')

    # Simple network visualization
    num_nodes = 8
    positions = {}
    for i in range(num_nodes):
        angle = 2 * np.pi * i / num_nodes
        positions[i] = (np.cos(angle) * 2, np.sin(angle) * 2)

    # Draw nodes
    node_sizes = np.random.uniform(500, 2000, num_nodes)
    for i, (node, pos) in enumerate(positions.items()):
        circle = Circle(pos, np.sqrt(node_sizes[i])/100, facecolor=sns.color_palette()[i % len(sns.color_palette())],
                       edgecolor='black', linewidth=2, alpha=0.8)
        axes[0, 0].add_patch(circle)
        axes[0, 0].text(pos[0], pos[1], f'Node {i}', ha='center', va='center', fontweight='bold', fontsize=10)

    # Draw edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() > 0.6:  # Random connectivity
                pos1 = positions[i]
                pos2 = positions[j]
                axes[0, 0].plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', linewidth=1, alpha=0.6)

    axes[0, 0].set_xlim(-3, 3)
    axes[0, 0].set_ylim(-3, 3)
    axes[0, 0].set_title('Component Dependency Network', fontweight='bold')
    axes[0, 0].axis('equal')
    axes[0, 0].axis('off')

    # Execution timeline
    components = ['GPU Init', 'Free Energy', 'Bayesian', 'Message Passing', 'Policy', 'Triton', 'Engine', 'Tests']
    execution_times = np.random.uniform(0.1, 2.0, len(components))

    colors = ['green' if np.random.random() > 0.3 else 'red' for _ in components]
    bars = axes[0, 1].barh(range(len(components)), execution_times, left=0, color=colors, alpha=0.7, height=0.8)
    axes[0, 1].set_yticks(range(len(components)))
    axes[0, 1].set_yticklabels(components)
    axes[0, 1].set_xlabel('Execution Order')
    axes[0, 1].set_title('Execution Timeline', fontweight='bold')

    # Add legend
    legend_elements = [Patch(facecolor='green', label='Success'),
                      Patch(facecolor='red', label='Failed')]
    axes[0, 1].legend(handles=legend_elements, loc='lower right')

    # Performance comparison
    methods = ['PyTorch CPU', 'PyTorch MPS', 'Triton MPS', 'Triton CUDA']
    performance = [1.0, 0.8, 0.6, 0.4]  # Relative performance

    bars = axes[1, 0].bar(range(len(methods)), performance, color=['red', 'green', 'blue', 'purple'], alpha=0.7)
    axes[1, 0].set_xticks(range(len(methods)))
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Relative Performance (higher is better)')
    axes[1, 0].set_title('Performance Comparison', fontweight='bold')

    # Statistical analysis
    np.random.seed(42)
    data1 = np.random.normal(0.5, 0.1, 100)
    data2 = np.random.normal(0.7, 0.15, 100)

    axes[1, 1].hist([data1, data2], bins=20, alpha=0.7, label=['Method A', 'Method B'], color=['blue', 'orange'])
    axes[1, 1].set_xlabel('Performance Metric')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Statistical Distribution Analysis', fontweight='bold')
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = OUTPUTS_DIR / "network_and_performance_analysis.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    print(f"✓ Network and performance analysis saved as: {save_path}")


def demo_data_exports():
    """Demonstrate data export capabilities."""
    print("💾 Creating Data Exports...")

    results = create_sample_data()

    # Export JSON data
    data_export = {
        "results": results,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "framework": "Active Inference",
            "version": "1.0.0",
            "platform": "Apple Silicon"
        },
        "summary": {
            "total_components": len(results),
            "successful_components": len([r for r in results.values() if r and r.get("success", False)]),
            "visualizations_created": 3
        }
    }

    save_path = OUTPUTS_DIR / "comprehensive_data_export.json"
    with open(save_path, 'w') as f:
        json.dump(data_export, f, indent=2, default=str)
    print(f"✓ Comprehensive JSON data export saved as: {save_path}")

    # Export CSV data
    csv_rows = []
    for component_name, result in results.items():
        if result and isinstance(result, dict):
            row = {"component": component_name}
            # Flatten nested dictionaries
            for key, value in result.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            csv_rows.append(row)

    if csv_rows:
        df = pd.DataFrame(csv_rows)
        save_path = OUTPUTS_DIR / "performance_metrics_export.csv"
        df.to_csv(save_path, index=False)
        print(f"✓ Performance metrics CSV export saved as: {save_path}")

    # Export summary statistics
    summary_stats = {
        "total_tests": len(results),
        "successful_tests": len([r for r in results.values() if r and r.get("success", False)]),
        "failed_tests": len([r for r in results.values() if not (r and r.get("success", False))]),
        "success_rate": len([r for r in results.values() if r and r.get("success", False)]) / len(results) * 100,
        "components_with_timing": len([r for r in results.values() if r and "computation_time" in r]),
        "average_computation_time": np.mean([r["computation_time"] for r in results.values() if r and "computation_time" in r]) if any("computation_time" in str(r) for r in results.values()) else 0
    }

    save_path = OUTPUTS_DIR / "summary_statistics.json"
    with open(save_path, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    print(f"✓ Summary statistics saved as: {save_path}")


def main():
    """Run the comprehensive visualization demo."""
    print("🚀 COMPREHENSIVE VISUALIZATION DEMO")
    print("=" * 50)

    try:
        # Create all visualization types
        demo_performance_dashboard()
        demo_belief_evolution_analysis()
        demo_network_topology()
        demo_data_exports()

        print("\n🎉 VISUALIZATION DEMO COMPLETED!")
        print("=" * 50)
        print(f"Generated visualizations in: {OUTPUTS_DIR}")
        print("• 📊 comprehensive_performance_dashboard.png")
        print("• 🔍 belief_evolution_comprehensive.png")
        print("• 🕸️ network_and_performance_analysis.png")
        print("• 💾 comprehensive_data_export.json")
        print("• 📈 performance_metrics_export.csv")
        print("• 📊 summary_statistics.json")
        print("\nThese demonstrate the many visualization types available!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
