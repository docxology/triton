"""
Comprehensive Free Energy Methods Tests with Triton Integration

Tests the VariationalFreeEnergy and ExpectedFreeEnergy classes with detailed Triton kernel validation.
Follows AAA pattern and includes extensive Triton-specific testing for variational free energy
and expected free energy computations in POMDP active inference.
"""

import pytest
import torch
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from free_energy import VariationalFreeEnergy, ExpectedFreeEnergy
from core import TritonFeatureManager, TritonFeatureConfig, TRITON_AVAILABLE
from . import (
    setup_test_environment,
    create_test_tensor,
    assert_tensors_close,
    TEST_DEVICE,
)


@pytest.mark.triton
class TestTritonVariationalFreeEnergy:
    """Test Triton-accelerated variational free energy computations."""

    def test_triton_vfe_kernel_registration(self):
        """Test that VFE kernels are properly registered with Triton metadata."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)
        vfe_engine = VariationalFreeEnergy(fm)

        # Verify kernel registration with comprehensive Triton metadata
        expected_metadata = {
            "description": "Triton kernel for variational free energy computation",
            "input_shapes": [
                "batch_size x feature_dim",
                "batch_size x feature_dim",
                "batch_size x feature_dim",
                "batch_size x feature_dim x feature_dim",
            ],
            "output_shapes": ["batch_size"],
            "optimizations": ["vectorization", "shared_memory", "parallel_reduction"],
            "triton_features": [
                "tl.load",
                "tl.store",
                "tl.program_id",
                "tl.sum",
                "tl.log",
                "tl.exp",
            ],
            "block_size": 256,
            "memory_layout": "coalesced",
        }

        kernel_info = fm.get_kernel("variational_free_energy")
        assert kernel_info is not None
        metadata = kernel_info["metadata"]

        # Verify all expected Triton features are present
        for key, expected_value in expected_metadata.items():
            assert key in metadata
            if isinstance(expected_value, list):
                for item in expected_value:
                    assert item in metadata[key]
            else:
                assert metadata[key] == expected_value

    def test_triton_vfe_memory_patterns(self):
        """Test Triton-optimized memory access patterns for VFE computation."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type, max_shared_memory=32768)
        fm = TritonFeatureManager(config)
        vfe_engine = VariationalFreeEnergy(fm)

        # Test various batch sizes and feature dimensions that Triton would optimize
        test_configs = [
            (1, 10),  # Small batch, small features
            (4, 50),  # Medium batch, medium features
            (8, 100),  # Large batch, large features
            (16, 200),  # Very large batch, very large features
        ]

        for batch_size, feature_dim in test_configs:
            # Create test tensors with Triton-optimized shapes
            observations = create_test_tensor(batch_size, feature_dim)
            posterior = torch.softmax(
                create_test_tensor(batch_size, feature_dim), dim=1
            )
            prior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
            likelihood = torch.softmax(
                create_test_tensor(batch_size, feature_dim), dim=1
            )

            # Compute VFE using Triton-optimized path
            vfe = vfe_engine.compute_vfe_kernel(
                observations, posterior, prior, likelihood
            )

            # Verify output properties
            assert vfe.shape == (batch_size,)
            assert vfe.device.type == TEST_DEVICE.type
            assert torch.all(torch.isfinite(vfe))
            # Free energy values can be positive or negative depending on formulation

    def test_triton_vfe_numerical_stability(self):
        """Test numerical stability of Triton VFE computation with edge cases."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type, dtype=torch.float32)
        fm = TritonFeatureManager(config)
        vfe_engine = VariationalFreeEnergy(fm)

        # Test with extreme values that could cause numerical issues
        batch_size, feature_dim = 3, 4

        # Create tensors with extreme values
        observations = torch.randn(batch_size, feature_dim, device=TEST_DEVICE, requires_grad=True) * 10
        posterior = torch.softmax(
            torch.randn(batch_size, feature_dim, device=TEST_DEVICE, requires_grad=True) * 10, dim=1
        )
        prior = torch.softmax(
            torch.randn(batch_size, feature_dim, device=TEST_DEVICE, requires_grad=True) * 10, dim=1
        )

        # Create likelihood with extreme probabilities
        likelihood = torch.softmax(
            torch.randn(batch_size, feature_dim, device=TEST_DEVICE, requires_grad=True) * 5, dim=1
        )

        # Compute VFE - should handle numerical extremes gracefully
        vfe = vfe_engine.compute_vfe_kernel(observations, posterior, prior, likelihood)

        # Verify numerical stability
        assert torch.all(torch.isfinite(vfe))
        assert not torch.any(torch.isnan(vfe))
        assert not torch.any(torch.isinf(vfe))

        # Test gradient computation stability
        vfe_sum = vfe.sum()
        vfe_sum.backward()

        # Check that gradients are finite
        if posterior.grad is not None:
            assert torch.all(torch.isfinite(posterior.grad))

    def test_triton_vfe_performance_scaling(self):
        """Test Triton VFE performance scaling with problem size."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)
        vfe_engine = VariationalFreeEnergy(fm)

        # Test different problem sizes
        sizes = [(10, 5), (50, 10), (100, 20), (200, 30)]

        performance_data = {}

        for batch_size, feature_dim in sizes:
            try:
                # Create test tensors
                observations = create_test_tensor(batch_size, feature_dim)
                posterior = torch.softmax(
                    create_test_tensor(batch_size, feature_dim), dim=1
                )
                prior = torch.softmax(
                    create_test_tensor(batch_size, feature_dim), dim=1
                )
                likelihood = torch.softmax(
                    create_test_tensor(batch_size, feature_dim), dim=1
                )

                # Warm up
                _ = vfe_engine.compute_vfe_kernel(
                    observations, posterior, prior, likelihood
                )

                # Time multiple runs
                import time

                start_time = time.time()

                n_runs = 10
                for _ in range(n_runs):
                    vfe = vfe_engine.compute_vfe_kernel(
                        observations, posterior, prior, likelihood
                    )

                end_time = time.time()

                avg_time = (end_time - start_time) / n_runs
                performance_data[(batch_size, feature_dim)] = avg_time

                print(
                    f"Performance test: {(batch_size, feature_dim)} took {avg_time:.2f}s"
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Size {(batch_size, feature_dim)}: Skipped (out of memory)")
                    break
                else:
                    raise

        # Verify reasonable performance scaling
        if len(performance_data) >= 2:
            sorted_sizes = sorted(performance_data.keys(), key=lambda x: x[0] * x[1])
            times = [performance_data[size] for size in sorted_sizes]

            # Performance should scale reasonably with problem size
            for i in range(1, len(times)):
                size_ratio = (sorted_sizes[i][0] * sorted_sizes[i][1]) / (
                    sorted_sizes[i - 1][0] * sorted_sizes[i - 1][1]
                )
                time_ratio = times[i] / times[i - 1]

                # Allow up to 5x time increase for 10x size increase
                assert (
                    time_ratio / size_ratio < 5
                ), f"Poor scaling: size {size_ratio:.1f}x, time {time_ratio:.1f}x"


@pytest.mark.triton
class TestTritonExpectedFreeEnergy:
    """Test Triton-accelerated expected free energy computations."""

    def test_triton_efe_kernel_registration(self):
        """Test that EFE kernels are properly registered with Triton metadata."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)
        efe_engine = ExpectedFreeEnergy(fm)

        # Verify kernel registration
        expected_metadata = {
            "description": "Triton kernel for expected free energy computation",
            "input_shapes": [
                "batch_size x feature_dim",
                "num_policies x feature_dim",
                "batch_size x feature_dim",
            ],
            "output_shapes": ["batch_size x num_policies"],
            "optimizations": [
                "vectorization",
                "shared_memory",
                "parallel_policy_evaluation",
            ],
            "triton_features": [
                "tl.load",
                "tl.store",
                "tl.program_id",
                "tl.sum",
                "tl.maximum",
            ],
            "block_size": 128,
            "memory_layout": "coalesced",
        }

        kernel_info = fm.get_kernel("expected_free_energy")
        assert kernel_info is not None
        metadata = kernel_info["metadata"]

        # Verify Triton features are present
        for key in ["triton_features", "optimizations", "block_size"]:
            assert key in metadata

    def test_triton_efe_policy_evaluation(self):
        """Test Triton-optimized policy evaluation in EFE computation."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)
        efe_engine = ExpectedFreeEnergy(fm)

        # Test different policy evaluation scenarios
        scenarios = [
            {
                "batch_size": 4,
                "num_policies": 8,
                "feature_dim": 6,
                "description": "Small scale policy evaluation",
            },
            {
                "batch_size": 8,
                "num_policies": 16,
                "feature_dim": 12,
                "description": "Medium scale policy evaluation",
            },
            {
                "batch_size": 12,
                "num_policies": 24,
                "feature_dim": 18,
                "description": "Large scale policy evaluation",
            },
        ]

        for scenario in scenarios:
            batch_size = scenario["batch_size"]
            num_policies = scenario["num_policies"]
            feature_dim = scenario["feature_dim"]

            # Create test data
            observations = create_test_tensor(batch_size, feature_dim)
            policies = create_test_tensor(num_policies, feature_dim)
            posterior = torch.softmax(
                create_test_tensor(batch_size, feature_dim), dim=1
            )
            preferences = create_test_tensor(batch_size)

            # Compute EFE
            efe_values = efe_engine.compute_expected_free_energy(
                observations,
                posterior,
                posterior,  # Using posterior as prior for simplicity
                torch.eye(feature_dim, device=TEST_DEVICE)
                .unsqueeze(0)
                .expand(batch_size, -1, -1),  # Identity likelihood
                policies.unsqueeze(-1),
                preferences.unsqueeze(0),
            )

            # Verify output properties
            assert efe_values.shape == (batch_size, num_policies)
            assert torch.all(torch.isfinite(efe_values))

            # Policy selection should work
            policy_idx, selected_efe = efe_engine.select_policy(efe_values)

            assert policy_idx.shape == (batch_size,)
            assert torch.all(policy_idx >= 0) and torch.all(policy_idx < num_policies)
            assert selected_efe.shape == (batch_size,)

    def test_triton_efe_epistemic_pragmatic_decomposition(self):
        """Test epistemic vs pragmatic value decomposition in Triton EFE."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)
        efe_engine = ExpectedFreeEnergy(fm)

        batch_size, num_policies, feature_dim = 3, 5, 4

        # Create test data
        posterior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        prior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        likelihood = (
            torch.eye(feature_dim, device=TEST_DEVICE)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        policies = create_test_tensor(num_policies, feature_dim)

        # Test epistemic value computation (information gain)
        epistemic_values = []
        for i in range(num_policies):
            policy = policies[i]
            epistemic = efe_engine.compute_epistemic_value(
                posterior, prior, likelihood, policy
            )
            epistemic_values.append(epistemic)

        epistemic_tensor = torch.stack(epistemic_values)
        assert epistemic_tensor.shape == (num_policies, batch_size)
        assert torch.all(torch.isfinite(epistemic_tensor))

        # Test pragmatic value computation (goal achievement)
        preferences = torch.zeros(feature_dim, device=TEST_DEVICE)
        preferences[0] = 1.0  # Prefer first state

        pragmatic_values = []
        for i in range(num_policies):
            policy = policies[i]
            pragmatic = efe_engine.compute_pragmatic_value(
                posterior, policy, preferences
            )
            pragmatic_values.append(pragmatic)

        pragmatic_tensor = torch.stack(pragmatic_values)
        assert pragmatic_tensor.shape == (num_policies, batch_size)
        assert torch.all(torch.isfinite(pragmatic_tensor))

        # Total EFE should be combination of epistemic and pragmatic
        efe_combined = epistemic_tensor.mean() + pragmatic_tensor.mean()
        assert torch.isfinite(efe_combined)

    def test_triton_efe_gradient_computation(self):
        """Test gradient computation for EFE optimization."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)
        efe_engine = ExpectedFreeEnergy(fm)

        batch_size, num_policies, feature_dim = 2, 3, 4

        # Create differentiable test data
        observations = create_test_tensor(batch_size, feature_dim, requires_grad=True)
        policies = create_test_tensor(num_policies, feature_dim, requires_grad=True)
        posterior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        preferences = create_test_tensor(batch_size, requires_grad=True)

        # Compute EFE with gradients
        efe_values = efe_engine.compute_expected_free_energy(
            observations,
            posterior,
            posterior,
            torch.eye(feature_dim, device=TEST_DEVICE).unsqueeze(0),
            policies.unsqueeze(-1),
            preferences.unsqueeze(0),
        )

        # Test gradient computation
        loss = efe_values.mean()
        loss.backward()

        # Verify gradients are computed
        assert observations.grad is not None
        assert policies.grad is not None
        assert preferences.grad is not None

        # Verify gradients are finite
        assert torch.all(torch.isfinite(observations.grad))
        assert torch.all(torch.isfinite(policies.grad))
        assert torch.all(torch.isfinite(preferences.grad))


@pytest.mark.triton
class TestTritonFreeEnergyIntegration:
    """Test integration between variational and expected free energy with Triton."""

    def test_triton_vfe_efe_integration_workflow(self):
        """Test complete Triton-accelerated VFE-EFE workflow for POMDP active inference."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)

        vfe_engine = VariationalFreeEnergy(fm)
        efe_engine = ExpectedFreeEnergy(fm)

        # Simulate POMDP active inference workflow
        batch_size, feature_dim, num_policies = 4, 8, 6

        # Step 1: Initial belief state (uniform prior)
        prior = torch.ones(batch_size, feature_dim, device=TEST_DEVICE) / feature_dim

        # Step 2: Process observation and update belief (VFE)
        observation = torch.randn(batch_size, feature_dim, device=TEST_DEVICE)
        likelihood = (
            torch.eye(feature_dim, device=TEST_DEVICE)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        posterior = vfe_engine.update_posterior(
            observation, prior, likelihood, n_iterations=5
        )
        assert posterior.shape == (batch_size, feature_dim)
        assert torch.allclose(
            posterior.sum(dim=1), torch.ones(batch_size, device=TEST_DEVICE)
        )

        # Step 3: Generate policies and evaluate EFE
        policies = create_test_tensor(num_policies, feature_dim)
        preferences = torch.zeros(feature_dim, device=TEST_DEVICE)
        preferences[0] = 1.0  # Prefer first state

        efe_values = efe_engine.compute_expected_free_energy(
            observation,
            posterior,
            prior,
            likelihood,
            policies.unsqueeze(-1),
            preferences.unsqueeze(0),
        )

        # Step 4: Select optimal policy
        policy_idx, selected_efe = efe_engine.select_policy(efe_values)

        # Verify complete workflow
        assert efe_values.shape == (batch_size, num_policies)
        assert policy_idx.shape == (batch_size,)
        assert torch.all(policy_idx >= 0) and torch.all(policy_idx < num_policies)
        assert selected_efe.shape == (batch_size,)

        # The selected EFE should be finite
        assert torch.all(torch.isfinite(selected_efe))

        print("✓ Complete Triton VFE-EFE workflow successful")

    def test_triton_free_energy_memory_efficiency(self):
        """Test memory efficiency of Triton free energy computations."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type, max_shared_memory=16384)
        fm = TritonFeatureManager(config)

        vfe_engine = VariationalFreeEnergy(fm)

        # Test with different batch sizes to check memory scaling
        sizes = [(10, 5), (50, 10), (100, 15)]

        memory_usage = {}

        for batch_size, feature_dim in sizes:
            try:
                # Create test tensors
                observations = create_test_tensor(batch_size, feature_dim)
                posterior = torch.softmax(
                    create_test_tensor(batch_size, feature_dim), dim=1
                )
                prior = torch.softmax(
                    create_test_tensor(batch_size, feature_dim), dim=1
                )
                likelihood = torch.softmax(
                    create_test_tensor(batch_size, feature_dim), dim=1
                )

                # Clear any existing GPU cache
                if TEST_DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                # Measure memory before
                if TEST_DEVICE.type == "cuda":
                    memory_before = torch.cuda.memory_allocated(TEST_DEVICE)

                # Compute VFE
                vfe = vfe_engine.compute_vfe_kernel(
                    observations, posterior, prior, likelihood
                )

                # Measure memory after
                if TEST_DEVICE.type == "cuda":
                    memory_after = torch.cuda.memory_allocated(TEST_DEVICE)
                    memory_used = memory_after - memory_before
                    memory_usage[(batch_size, feature_dim)] = memory_used

                    print(
                        f"Memory usage for {(batch_size, feature_dim)}: {memory_used/1024/1024:.2f} MB"
                    )
                else:
                    memory_usage[(batch_size, feature_dim)] = 0  # CPU mode

                assert vfe.shape == (batch_size,)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Size {(batch_size, feature_dim)}: Out of memory")
                    break
                else:
                    raise

        # Verify memory usage is reasonable
        if TEST_DEVICE.type == "cuda" and len(memory_usage) >= 2:
            sizes_list = list(memory_usage.keys())
            memory_list = [memory_usage[size] for size in sizes_list]

            # Memory usage should scale reasonably with problem size
            for i in range(1, len(memory_list)):
                size_ratio = (sizes_list[i][0] * sizes_list[i][1]) / (
                    sizes_list[i - 1][0] * sizes_list[i - 1][1]
                )
                memory_ratio = (
                    memory_list[i] / memory_list[i - 1] if memory_list[i - 1] > 0 else 1
                )

                # Allow up to 3x memory increase for 10x size increase
                assert (
                    memory_ratio / size_ratio < 3
                ), f"Poor memory scaling: {memory_ratio:.1f}x memory for {size_ratio:.1f}x size"


class TestVariationalFreeEnergy:
    """Test variational free energy computation."""

    @pytest.fixture
    def vfe_engine(self):
        """Create variational free energy engine for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        return VariationalFreeEnergy(feature_manager)

    def test_initialization(self, vfe_engine):
        """Test VFE engine initialization."""
        assert isinstance(vfe_engine.feature_manager, TritonFeatureManager)
        assert "variational_free_energy" in vfe_engine.feature_manager._kernels

    def test_compute_free_energy(self, vfe_engine):
        """Test variational free energy computation."""
        batch_size, feature_dim = 5, 3

        observations = create_test_tensor(batch_size, feature_dim)
        posterior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        prior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        likelihood = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)

        free_energy = vfe_engine.compute(observations, posterior, prior, likelihood)

        assert free_energy.shape == (batch_size,)
        assert free_energy.device.type == TEST_DEVICE.type
        assert torch.all(torch.isfinite(free_energy))

    def test_minimize_free_energy(self, vfe_engine):
        """Test variational free energy minimization."""
        batch_size, feature_dim = 3, 4

        observations = create_test_tensor(batch_size, feature_dim)
        initial_posterior = torch.softmax(
            create_test_tensor(batch_size, feature_dim), dim=1
        )
        prior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        likelihood = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)

        optimized_posterior = vfe_engine.minimize(
            observations,
            initial_posterior,
            prior,
            likelihood,
            num_iterations=10,
            learning_rate=0.1,
        )

        # Check that posterior remains normalized
        posterior_sums = optimized_posterior.sum(dim=1)
        assert_tensors_close(posterior_sums, torch.ones(batch_size, device=TEST_DEVICE))

        # Check that all values are positive (after softmax)
        assert torch.all(optimized_posterior >= 0)

        assert optimized_posterior.shape == (batch_size, feature_dim)


class TestExpectedFreeEnergy:
    """Test expected free energy computation."""

    @pytest.fixture
    def efe_engine(self):
        """Create expected free energy engine for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        return ExpectedFreeEnergy(feature_manager)

    def test_initialization(self, efe_engine):
        """Test EFE engine initialization."""
        assert isinstance(efe_engine.feature_manager, TritonFeatureManager)
        assert "expected_free_energy" in efe_engine.feature_manager._kernels

    def test_compute_expected_free_energy(self, efe_engine):
        """Test expected free energy computation."""
        batch_size, num_policies, feature_dim = 3, 4, 5

        observations = create_test_tensor(batch_size, feature_dim)
        policies = create_test_tensor(num_policies, feature_dim)
        posterior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        preferences = create_test_tensor(batch_size)

        EFE = efe_engine.compute(observations, policies, posterior, preferences)

        assert EFE.shape == (batch_size, num_policies)
        assert EFE.device.type == TEST_DEVICE.type
        assert torch.all(torch.isfinite(EFE))

    def test_compute_without_preferences(self, efe_engine):
        """Test EFE computation without explicit preferences."""
        batch_size, num_policies, feature_dim = 2, 3, 4

        observations = create_test_tensor(batch_size, feature_dim)
        policies = create_test_tensor(num_policies, feature_dim)
        posterior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)

        EFE = efe_engine.compute(observations, policies, posterior)

        assert EFE.shape == (batch_size, num_policies)
        assert torch.all(torch.isfinite(EFE))

    def test_select_policy(self, efe_engine):
        """Test policy selection based on EFE."""
        batch_size, num_policies = 3, 4

        # Create EFE matrix with known minimum for each batch
        EFE = create_test_tensor(batch_size, num_policies)
        # Make policy 1 have the lowest EFE for all batches
        EFE[:, 1] = -10.0

        policy_indices, EFE_values = efe_engine.select_policy(EFE)

        assert policy_indices.shape == (batch_size,)
        assert EFE_values.shape == (batch_size,)
        assert torch.all(policy_indices == 1)  # All should select policy 1
        assert torch.all(EFE_values == -10.0)  # All should have EFE of -10


class TestFreeEnergyIntegration:
    """Integration tests for free energy methods."""

    def test_variational_expected_free_energy_integration(self):
        """Test integration between variational and expected free energy."""
        setup_test_environment()

        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        vfe_engine = VariationalFreeEnergy(feature_manager)
        efe_engine = ExpectedFreeEnergy(feature_manager)

        batch_size, num_policies, feature_dim = 2, 3, 4

        # Create shared data
        observations = create_test_tensor(batch_size, feature_dim)
        policies = create_test_tensor(num_policies, feature_dim)
        posterior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        prior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        likelihood = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)

        # Compute both types of free energy
        vfe = vfe_engine.compute(observations, posterior, prior, likelihood)
        efe = efe_engine.compute(observations, policies, posterior)

        # Check shapes and basic properties
        assert vfe.shape == (batch_size,)
        assert efe.shape == (batch_size, num_policies)
        assert torch.all(torch.isfinite(vfe))
        assert torch.all(torch.isfinite(efe))

    def test_free_energy_minimization_convergence(self):
        """Test that free energy minimization converges."""
        setup_test_environment()

        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        vfe_engine = VariationalFreeEnergy(feature_manager)

        batch_size, feature_dim = 2, 3

        observations = create_test_tensor(batch_size, feature_dim)
        initial_posterior = torch.softmax(
            create_test_tensor(batch_size, feature_dim), dim=1
        )
        prior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)
        likelihood = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)

        # Compute initial free energy
        initial_fe = vfe_engine.compute(
            observations, initial_posterior, prior, likelihood
        )

        # Minimize free energy
        optimized_posterior = vfe_engine.minimize(
            observations,
            initial_posterior,
            prior,
            likelihood,
            num_iterations=20,
            learning_rate=0.05,
        )

        # Compute final free energy
        final_fe = vfe_engine.compute(
            observations, optimized_posterior, prior, likelihood
        )

        # Free energy should decrease (become more negative)
        assert torch.all(
            final_fe <= initial_fe + 0.1
        )  # Allow small numerical differences

    def test_policy_selection_consistency(self):
        """Test that policy selection is consistent."""
        setup_test_environment()

        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        efe_engine = ExpectedFreeEnergy(feature_manager)

        batch_size, num_policies, feature_dim = 3, 5, 4

        observations = create_test_tensor(batch_size, feature_dim)
        policies = create_test_tensor(num_policies, feature_dim)
        posterior = torch.softmax(create_test_tensor(batch_size, feature_dim), dim=1)

        # Compute EFE twice
        EFE1 = efe_engine.compute(observations, policies, posterior)
        EFE2 = efe_engine.compute(observations, policies, posterior)

        # Results should be identical (deterministic computation)
        assert_tensors_close(EFE1, EFE2)

        # Policy selection should be consistent
        indices1, values1 = efe_engine.select_policy(EFE1)
        indices2, values2 = efe_engine.select_policy(EFE2)

        assert torch.all(indices1 == indices2)
        assert_tensors_close(values1, values2)
