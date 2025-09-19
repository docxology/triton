"""
Active Inference and Free Energy Principle Implementation

This module provides comprehensive GPU-accelerated implementations of active inference methods
using Triton for high-performance computation of variational free energy, expected free energy,
message passing algorithms, and complete POMDP active inference.

Features:
- Complete Triton kernel coverage with GPU acceleration
- Variational Free Energy (VFE) computation with gradient-based optimization
- Expected Free Energy (EFE) for policy selection
- Belief propagation and variational message passing
- POMDP active inference with state estimation and action selection
- Comprehensive Triton feature validation and performance benchmarking
- Real data analysis with validated implementations

All implementations use real Triton kernels with proper GPU memory management,
shared memory optimization, and parallel computation patterns.
"""

__version__ = "1.0.0"
__author__ = "Triton Active Inference Team"

import torch

# Core Triton functionality
from .core import (
    TritonFeatureManager,
    GPUAccelerator,
    TritonFeatureConfig,
    get_feature_manager,
    TRITON_AVAILABLE,
    # Triton kernel operations
    create_triton_kernel,
    launch_triton_kernel,
    optimize_memory_layout,
    # Device management
    get_optimal_device,
    synchronize_device,
    get_memory_stats,
    # Performance profiling
    profile_kernel_performance,
    benchmark_triton_kernels,
)

# Core Bayesian inference (fundamental operations)
from .bayesian_core import (
    BayesianInferenceCore,
    BayesianUpdateResult,
    InferenceHistory,
    create_bayesian_core,
    bayesian_update_with_reporting,
)

# Inference methods
from .inference import (
    ActiveInferenceEngine,
    BayesianInference,
    # Advanced inference
    VariationalInference,
    PredictiveCoding,
    # Optimization methods
    gradient_based_inference,
    natural_gradient_inference,
    # Sampling methods
    importance_sampling,
    metropolis_hastings,
)

# Free energy methods
from .free_energy import (
    VariationalFreeEnergy,
    ExpectedFreeEnergy,
    # Free energy computations
    compute_variational_free_energy,
    compute_expected_free_energy,
    minimize_free_energy,
    # Gradient computations
    compute_free_energy_gradient,
    compute_policy_gradient,
    # Optimization
    free_energy_minimization,
    policy_optimization,
)

# Message passing
from .message_passing import (
    MessagePassing,
    BeliefPropagation,
    VariationalMessagePassing,
    ExpectationPropagation,
    TreeReweightedMessagePassing,
    # Message passing algorithms
    belief_propagation,
    variational_message_passing,
    loopy_belief_propagation,
    # Graph operations
    create_factor_graph,
    create_markov_random_field,
    # Convergence methods
    check_convergence,
    accelerate_convergence,
)

# POMDP Active Inference
from .pomdp_active_inference import (
    POMDPActiveInference,
    GridWorld,
    # Environment models
    create_grid_world,
    create_continuous_pomdp,
    # State estimation
    variational_state_estimation,
    particle_filter_state_estimation,
    # Policy selection
    expected_free_energy_policy,
    kl_divergence_policy,
    pragmatic_value_policy,
    # Complete active inference loop
    active_inference_loop,
    adaptive_active_inference,
)

# Triton Coverage and Validation
from .triton_coverage import (
    TritonCoverageVerifier,
    run_triton_coverage_analysis,
    CoverageConfig,
    TritonFeature,
    # Validation methods
    validate_triton_features,
    benchmark_all_features,
    generate_coverage_report,
    # Performance analysis
    analyze_performance_bottlenecks,
    optimize_kernel_performance,
    # Feature detection
    detect_available_features,
    get_triton_capabilities,
)


# Convenience functions for common operations
def initialize_bayesian_core(device="auto", enable_logging=True, enable_validation=True, enable_visualization=True):
    """
    Initialize a Bayesian inference core for fundamental Bayesian operations.

    This provides the core Bayesian updating functionality with comprehensive
    logging, validation, and reporting capabilities.

    Args:
        device: Device to use ('cuda', 'cpu', 'mps', or 'auto')
        enable_logging: Enable comprehensive operation logging
        enable_validation: Enable real data input validation
        enable_visualization: Enable belief distribution visualization

    Returns:
        Configured BayesianInferenceCore instance

    Example:
        >>> core = initialize_bayesian_core(device="cuda")
        >>> prior = torch.ones(10, 5) / 5  # Uniform prior
        >>> likelihood = torch.softmax(torch.randn(10, 5), dim=1)
        >>> result = core.bayesian_update(prior, likelihood)
        >>> print(f"Free energy: {result.free_energy.mean().item():.4f}")
    """
    return create_bayesian_core(
        device=device,
        enable_logging=enable_logging,
        enable_validation=enable_validation,
        enable_visualization=enable_visualization
    )


def initialize_triton_active_inference(device="auto", precision="fp32"):
    """
    Initialize the complete Triton active inference framework.

    Args:
        device: Device to use ('cuda', 'cpu', 'mps', or 'auto')
        precision: Precision mode ('fp32', 'fp16', 'bf16', 'fp8')

    Returns:
        Configured TritonFeatureManager and GPUAccelerator
    """
    config = TritonFeatureConfig(device=device)

    if precision == "fp16":
        config.dtype = torch.float16
    elif precision == "bf16":
        config.dtype = torch.bfloat16
        config.enable_bf16 = True
    elif precision == "fp8":
        config.enable_fp8 = True

    feature_manager = TritonFeatureManager(config)
    gpu_accelerator = GPUAccelerator(feature_manager)

    return feature_manager, gpu_accelerator


def run_complete_active_inference_demo(
    environment_size=8, n_episodes=5, max_steps_per_episode=20, learning_rate=0.01
):
    """
    Run a complete active inference demonstration with POMDP.

    Args:
        environment_size: Size of the grid world
        n_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        learning_rate: Learning rate for optimization

    Returns:
        Dictionary containing results and performance metrics
    """
    from .pomdp_active_inference import create_pomdp_demo

    # Run the demonstration
    results = create_pomdp_demo()

    # Enhance results with additional metrics
    results.update(
        {
            "environment_size": environment_size,
            "n_episodes": n_episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "learning_rate": learning_rate,
            "triton_accelerated": TRITON_AVAILABLE,
        }
    )

    return results


def benchmark_triton_active_inference():
    """
    Run comprehensive benchmarks of all Triton active inference methods.

    Returns:
        Dictionary containing benchmark results for all methods
    """
    from .triton_coverage import run_triton_coverage_analysis

    # Run comprehensive coverage analysis
    coverage_report = run_triton_coverage_analysis()

    # Add specific active inference benchmarks
    benchmark_results = {
        "coverage_analysis": coverage_report,
        "vfe_performance": benchmark_vfe_computation(),
        "efe_performance": benchmark_efe_computation(),
        "message_passing_performance": benchmark_message_passing(),
        "pomdp_performance": benchmark_pomdp_active_inference(),
    }

    return benchmark_results


def benchmark_vfe_computation():
    """Benchmark variational free energy computation performance."""
    from .free_energy import VariationalFreeEnergy
    import time

    fm = TritonFeatureManager()
    vfe_engine = VariationalFreeEnergy(fm)

    sizes = [(100, 10), (500, 20), (1000, 30)]
    results = {}

    for batch_size, feature_dim in sizes:
        obs = torch.randn(batch_size, feature_dim)
        post = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)
        prior = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)
        lik = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)

        start_time = time.time()
        for _ in range(10):
            vfe = vfe_engine.compute(obs, post, prior, lik)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        results[f"batch_{batch_size}_dim_{feature_dim}"] = {
            "avg_time_ms": avg_time * 1000,
            "gflops": (2 * batch_size * feature_dim) / (avg_time * 1e9),
        }

    return results


def benchmark_efe_computation():
    """Benchmark expected free energy computation performance."""
    from .free_energy import ExpectedFreeEnergy
    import time

    fm = TritonFeatureManager()
    efe_engine = ExpectedFreeEnergy(fm)

    configs = [(50, 5, 10), (100, 8, 15), (200, 10, 20)]
    results = {}

    for batch_size, n_policies, feature_dim in configs:
        obs = torch.randn(batch_size, feature_dim)
        policies = torch.randn(n_policies, feature_dim)
        post = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)

        start_time = time.time()
        for _ in range(5):
            efe = efe_engine.compute(obs, policies, post)
        end_time = time.time()

        avg_time = (end_time - start_time) / 5
        results[f"batch_{batch_size}_policies_{n_policies}_dim_{feature_dim}"] = {
            "avg_time_ms": avg_time * 1000,
            "computations_per_second": batch_size * n_policies / avg_time,
        }

    return results


def benchmark_message_passing():
    """Benchmark message passing algorithms."""
    from .message_passing import BeliefPropagation
    import time

    fm = TritonFeatureManager()
    bp_engine = BeliefPropagation(fm)

    sizes = [5, 10, 15, 20]
    results = {}

    for n_nodes in sizes:
        # Create test graph
        adjacency = torch.rand(n_nodes, n_nodes) > 0.7
        adjacency = (adjacency | adjacency.t()).float()  # Make symmetric
        node_potentials = torch.softmax(torch.randn(n_nodes, 3), dim=1)

        bp_engine.set_graph(adjacency, node_potentials)

        start_time = time.time()
        result = bp_engine.run(max_iterations=10)
        end_time = time.time()

        results[f"nodes_{n_nodes}"] = {
            "total_time_ms": (end_time - start_time) * 1000,
            "converged": result["converged"],
            "final_entropy": torch.distributions.Categorical(probs=result["beliefs"])
            .entropy()
            .mean()
            .item(),
        }

    return results


def benchmark_pomdp_active_inference():
    """Benchmark complete POMDP active inference."""
    from .pomdp_active_inference import POMDPActiveInference
    import time

    sizes = [4, 6, 8]
    results = {}

    for grid_size in sizes:
        pomdp = POMDPActiveInference(grid_size=grid_size)

        start_time = time.time()
        for _ in range(5):
            action = pomdp.select_action()
            pomdp.step(action)
        end_time = time.time()

        results[f"grid_{grid_size}x{grid_size}"] = {
            "total_time_ms": (end_time - start_time) * 1000,
            "avg_entropy": pomdp.get_belief_entropy(),
            "final_state": pomdp.get_most_likely_state(),
        }

    return results


__all__ = [
    # Core functionality
    "TritonFeatureManager",
    "GPUAccelerator",
    "TritonFeatureConfig",
    "get_feature_manager",
    "TRITON_AVAILABLE",
    "create_triton_kernel",
    "launch_triton_kernel",
    "optimize_memory_layout",
    "get_optimal_device",
    "synchronize_device",
    "get_memory_stats",
    "profile_kernel_performance",
    "benchmark_triton_kernels",
    # Inference methods
    "ActiveInferenceEngine",
    "BayesianInference",
    "VariationalInference",
    "PredictiveCoding",
    "gradient_based_inference",
    "natural_gradient_inference",
    "importance_sampling",
    "metropolis_hastings",
    # Free energy methods
    "VariationalFreeEnergy",
    "ExpectedFreeEnergy",
    "compute_variational_free_energy",
    "compute_expected_free_energy",
    "minimize_free_energy",
    "compute_free_energy_gradient",
    "compute_policy_gradient",
    "free_energy_minimization",
    "policy_optimization",
    # Message passing
    "MessagePassing",
    "BeliefPropagation",
    "VariationalMessagePassing",
    "ExpectationPropagation",
    "TreeReweightedMessagePassing",
    "belief_propagation",
    "variational_message_passing",
    "loopy_belief_propagation",
    "create_factor_graph",
    "create_markov_random_field",
    "check_convergence",
    "accelerate_convergence",
    # POMDP Active Inference
    "POMDPActiveInference",
    "GridWorld",
    "create_grid_world",
    "create_continuous_pomdp",
    "variational_state_estimation",
    "particle_filter_state_estimation",
    "expected_free_energy_policy",
    "kl_divergence_policy",
    "pragmatic_value_policy",
    "active_inference_loop",
    "adaptive_active_inference",
    # Triton Coverage and Validation
    "TritonCoverageVerifier",
    "run_triton_coverage_analysis",
    "CoverageConfig",
    "TritonFeature",
    "validate_triton_features",
    "benchmark_all_features",
    "generate_coverage_report",
    "analyze_performance_bottlenecks",
    "optimize_kernel_performance",
    "detect_available_features",
    "get_triton_capabilities",
    # Convenience functions
    "initialize_triton_active_inference",
    "run_complete_active_inference_demo",
    "benchmark_triton_active_inference",
    "benchmark_vfe_computation",
    "benchmark_efe_computation",
    "benchmark_message_passing",
    "benchmark_pomdp_active_inference",
]
