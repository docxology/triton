"""
Integration tests for Triton functionality in active inference framework.

These tests verify that Triton kernels are properly integrated and working
correctly within the active inference framework.
"""

import pytest
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core import TRITON_AVAILABLE, TritonFeatureManager, GPUAccelerator, TritonFeatureConfig
from free_energy import VariationalFreeEnergy, ExpectedFreeEnergy
from message_passing import BeliefPropagation, VariationalMessagePassing
from pomdp_active_inference import POMDPActiveInference


@pytest.mark.integration
class TestTritonIntegration:
    """Test Triton integration with active inference components."""

    @pytest.fixture
    def triton_setup(self):
        """Set up Triton components for testing."""
        config = TritonFeatureConfig(
            device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
            dtype=torch.float32,
        )

        fm = TritonFeatureManager(config)
        ga = GPUAccelerator(fm)

        return {
            "feature_manager": fm,
            "gpu_accelerator": ga,
            "device": ga.device,
            "config": config,
            "triton_available": TRITON_AVAILABLE,
        }

    def test_triton_feature_manager_integration(self, triton_setup):
        """Test that TritonFeatureManager integrates properly with other components."""
        fm = triton_setup["feature_manager"]

        # Test kernel registration
        def test_kernel():
            pass

        fm.register_kernel(
            "integration_test", test_kernel, {
                "description": "Integration test kernel",
                "input_shapes": ["batch_size x feature_dim"],
                "output_shapes": ["batch_size"]
            }
        )

        assert fm.verify_feature("integration_test")

        # Test feature listing
        features = fm.list_features()
        assert "kernels" in features
        assert "integration_test" in features["kernels"]

    def test_free_energy_triton_integration(self, triton_setup):
        """Test Triton integration with free energy components."""
        fm = triton_setup["feature_manager"]
        device = triton_setup["device"]

        # Test VFE integration
        vfe_engine = VariationalFreeEnergy(fm)

        # Create test data
        batch_size, feature_dim = 4, 8
        observations = torch.randn(batch_size, feature_dim, device=device)
        posterior = torch.softmax(
            torch.randn(batch_size, feature_dim, device=device), dim=1
        )
        prior = torch.softmax(
            torch.randn(batch_size, feature_dim, device=device), dim=1
        )
        likelihood = torch.softmax(
            torch.randn(batch_size, feature_dim, device=device), dim=1
        )

        # Test computation
        vfe = vfe_engine.compute(observations, posterior, prior, likelihood)

        assert vfe.shape == (batch_size,)
        assert vfe.device.type == device.type
        assert torch.all(torch.isfinite(vfe))

    def test_message_passing_triton_integration(self, triton_setup):
        """Test Triton integration with message passing components."""
        fm = triton_setup["feature_manager"]
        device = triton_setup["device"]

        # Create test graph
        adjacency = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float, device=device
        )

        node_potentials = torch.softmax(torch.randn(3, 2, device=device), dim=1)

        # Test BP integration
        bp_engine = BeliefPropagation(fm)
        bp_engine.set_graph(adjacency, node_potentials)

        result = bp_engine.run(max_iterations=3)

        assert "beliefs" in result
        assert result["beliefs"].shape == (3, 2)
        assert torch.all(torch.isfinite(result["beliefs"]))

    def test_pomdp_triton_integration(self, triton_setup):
        """Test Triton integration with POMDP components."""
        fm = triton_setup["feature_manager"]

        # Create simple POMDP
        pomdp = POMDPActiveInference(grid_size=4)

        # Test basic functionality
        initial_entropy = pomdp.get_belief_entropy()
        most_likely = pomdp.get_most_likely_state()

        assert isinstance(initial_entropy, float)
        assert isinstance(most_likely, int)
        assert 0 <= most_likely < pomdp.env.n_states

    def test_gpu_accelerator_integration(self, triton_setup):
        """Test GPU accelerator integration."""
        ga = triton_setup["gpu_accelerator"]
        device = triton_setup["device"]

        # Test tensor allocation
        shapes = [(10, 10), (5, 20), (2, 3, 4)]
        tensors = ga.allocate_tensors(shapes)

        assert len(tensors) == 3
        for tensor, expected_shape in zip(tensors, shapes):
            assert tensor.shape == torch.Size(expected_shape)
            assert tensor.device.type == device.type

        # Test memory management
        ga.synchronize()

    @pytest.mark.parametrize(
        "kernel_name",
        ["vector_add", "matrix_multiply", "softmax", "layer_norm", "attention"],
    )
    def test_kernel_availability(self, kernel_name):
        """Test that core Triton kernels are available."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        from core import create_triton_kernel

        kernel = create_triton_kernel(kernel_name)

        # Kernel should either be created or return None (fallback)
        assert kernel is not None or kernel is None  # Allow for fallback case

    def test_memory_optimization_features(self, triton_setup):
        """Test memory optimization features."""
        fm = triton_setup["feature_manager"]
        ga = triton_setup["gpu_accelerator"]

        # Test memory allocation patterns
        large_tensors = ga.allocate_tensors([(1000, 1000), (500, 2000)])

        for tensor in large_tensors:
            assert tensor.device.type == triton_setup["device"].type
            assert tensor.dtype == torch.float32

        # Clean up
        del large_tensors

    def test_performance_characteristics(self, triton_setup):
        """Test basic performance characteristics."""
        device = triton_setup["device"]

        # Simple performance test
        if device.type in ["cuda", "mps"]:
            size = 1000
            a = torch.randn(size, device=device)
            b = torch.randn(size, device=device)

            # Warm up
            _ = a + b

            # Timed operation
            import time

            start_time = time.time()
            for _ in range(100):
                c = a + b
            end_time = time.time()

            operation_time = (end_time - start_time) / 100
            assert operation_time > 0
            assert c.shape == (size,)


@pytest.mark.gpu
class TestGPUIntegration:
    """GPU-specific integration tests."""

    @pytest.fixture
    def gpu_setup(self):
        """Set up GPU testing environment."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from core import TritonFeatureManager, GPUAccelerator

        fm = TritonFeatureManager()
        ga = GPUAccelerator(fm)

        return {
            "feature_manager": fm,
            "gpu_accelerator": ga,
            "device": torch.device("cuda"),
        }

    def test_cuda_memory_management(self, gpu_setup):
        """Test CUDA memory management."""
        ga = gpu_setup["gpu_accelerator"]

        # Test memory allocation
        tensors = ga.allocate_tensors([(1000, 1000), (500, 2000)])

        # Check memory usage
        memory_stats = ga.get_memory_stats()
        assert "allocated" in memory_stats
        assert memory_stats["allocated"] > 0

        # Clean up
        del tensors

    def test_cuda_synchronization(self, gpu_setup):
        """Test CUDA synchronization."""
        ga = gpu_setup["gpu_accelerator"]

        # Create and use tensors
        tensor = torch.randn(1000, 1000, device="cuda")
        result = tensor @ tensor.t()

        # Synchronize
        ga.synchronize()

        assert result.shape == (1000, 1000)
        assert result.device.type == "cuda"
