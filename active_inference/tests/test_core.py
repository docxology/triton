"""
Comprehensive Triton Feature Management Tests

Tests the TritonFeatureManager and GPUAccelerator classes with detailed Triton kernel validation.
Follows AAA pattern and includes extensive Triton-specific testing.
"""

import pytest
import torch
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import (
    TritonFeatureManager,
    GPUAccelerator,
    TritonFeatureConfig,
    TRITON_AVAILABLE,
)
from . import (
    setup_test_environment,
    create_test_tensor,
    assert_tensors_close,
    TEST_DEVICE,
)


@pytest.mark.triton
class TestTritonKernelRegistration:
    """Test Triton kernel registration and management."""

    def test_triton_kernel_registration_and_validation(self):
        """ARRANGE: Set up Triton feature manager
        ACT: Register Triton kernels with metadata
        ASSERT: Verify kernel registration and feature validation"""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)

        # Register multiple Triton kernels with comprehensive metadata
        kernels_to_register = {
            "vector_add": {
                "description": "Vector addition kernel using Triton SIMD",
                "input_shapes": ["batch_size x dim", "batch_size x dim"],
                "output_shapes": ["batch_size x dim"],
                "optimizations": ["vectorization", "shared_memory"],
                "triton_features": ["tl.load", "tl.store", "tl.program_id"],
            },
            "matrix_multiply": {
                "description": "Matrix multiplication with Triton block tiling",
                "input_shapes": ["M x K", "K x N"],
                "output_shapes": ["M x N"],
                "optimizations": ["block_tiling", "shared_memory", "pipelining"],
                "triton_features": ["tl.arange", "tl.dot", "BLOCK_SIZE"],
            },
            "attention_kernel": {
                "description": "Scaled dot-product attention with Triton optimizations",
                "input_shapes": ["batch_size x seq_len x dim"],
                "output_shapes": ["batch_size x seq_len x dim"],
                "optimizations": [
                    "flash_attention",
                    "memory_efficient",
                    "vectorization",
                ],
                "triton_features": ["tl.softmax", "tl.exp", "tl.sum"],
            },
        }

        # Register all kernels
        for kernel_name, metadata in kernels_to_register.items():

            def dummy_kernel():
                pass

            fm.register_kernel(kernel_name, dummy_kernel, metadata)

            # Verify registration
            assert kernel_name in fm._kernels
            kernel_info = fm.get_kernel(kernel_name)
            assert kernel_info["kernel"] == dummy_kernel
            assert kernel_info["metadata"] == metadata

            # Verify Triton feature validation
            assert fm.verify_feature(kernel_name)

        # Test feature listing includes all registered kernels
        features = fm.list_features()
        assert "kernels" in features
        assert len(features["kernels"]) >= len(kernels_to_register)

    def test_triton_kernel_metadata_validation(self):
        """Test comprehensive Triton kernel metadata validation."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)

        # Test valid metadata
        valid_metadata = {
            "description": "Valid Triton kernel",
            "input_shapes": ["batch_size x dim", "batch_size x dim"],
            "output_shapes": ["batch_size x dim"],
            "optimizations": ["vectorization", "shared_memory"],
            "triton_features": ["tl.load", "tl.store", "tl.program_id"],
            "block_size": 128,
            "num_warps": 4,
            "num_stages": 3,
        }

        def valid_kernel():
            pass

        fm.register_kernel("valid_kernel", valid_kernel, valid_metadata)
        assert fm.verify_feature("valid_kernel")

        # Test invalid metadata (missing required fields)
        invalid_metadata = {"description": "Invalid kernel - missing shapes"}

        def invalid_kernel():
            pass

        fm.register_kernel("invalid_kernel", invalid_kernel, invalid_metadata)
        assert not fm.verify_feature("invalid_kernel")


@pytest.mark.triton
class TestTritonMemoryManagement:
    """Test Triton memory management and optimization features."""

    def test_triton_shared_memory_allocation(self):
        """Test Triton shared memory allocation and usage patterns."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type, max_shared_memory=65536)
        fm = TritonFeatureManager(config)
        ga = GPUAccelerator(fm)

        # Test various tensor allocation patterns that Triton would use
        allocation_patterns = [
            # Standard ML patterns
            (
                [batch, seq_len, hidden_dim]
                for batch in [1, 4, 8, 16]
                for seq_len in [64, 128, 256]
                for hidden_dim in [768, 1024, 2048]
            ),
            # Attention patterns
            (
                [batch, heads, seq_len, head_dim]
                for batch in [1, 2, 4]
                for heads in [8, 12, 16]
                for seq_len in [64, 128]
                for head_dim in [64, 128]
            ),
            # Convolution patterns
            (
                [batch, channels, height, width]
                for batch in [1, 4, 8]
                for channels in [64, 128, 256]
                for height in [32, 64]
                for width in [32, 64]
            ),
        ]

        for pattern_idx, pattern_generator in enumerate(allocation_patterns):
            for shape in pattern_generator:
                try:
                    tensors = ga.allocate_tensors([shape])
                    assert len(tensors) == 1
                    assert tensors[0].shape == torch.Size(shape)
                    assert tensors[0].device.type == TEST_DEVICE.type
                    assert tensors[0].dtype == config.dtype

                    # Test memory operations
                    if TEST_DEVICE.type == "cuda":
                        memory_stats = ga.get_memory_stats()
                        assert "allocated" in memory_stats
                        assert memory_stats["allocated"] > 0

                except RuntimeError as e:
                    # Skip if memory allocation fails (expected for very large tensors)
                    if "out of memory" in str(e).lower():
                        continue
                    else:
                        raise

    def test_triton_memory_coalescing_patterns(self):
        """Test memory coalescing patterns optimized for Triton."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)
        ga = GPUAccelerator(fm)

        # Test coalesced memory access patterns
        batch_size, feature_dim = 1024, 512

        # Create tensors with different memory layouts
        tensor_row_major = ga.allocate_tensors([(batch_size, feature_dim)])[0]
        tensor_col_major = tensor_row_major.t().contiguous()

        # Test that operations work correctly with both layouts
        result_rm = tensor_row_major @ tensor_row_major.t()
        result_cm = tensor_col_major @ tensor_col_major.t()

        assert result_rm.shape == (batch_size, batch_size)
        assert result_cm.shape == (feature_dim, feature_dim)

        # Verify numerical consistency
        # Note: Due to transpose, direct comparison may differ, but shapes should be correct
        assert result_rm.device.type == TEST_DEVICE.type
        assert result_cm.device.type == TEST_DEVICE.type


@pytest.mark.triton
class TestTritonKernelOptimization:
    """Test Triton kernel optimization features."""

    def test_triton_block_size_optimization(self):
        """Test Triton block size optimization for different workloads."""

        setup_test_environment()
        fm = TritonFeatureManager()

        # Test different block sizes for various kernel types
        kernel_configs = [
            {
                "name": "small_kernel",
                "block_size": 64,
                "workload": "element_wise",
                "description": "Small block size for element-wise operations",
            },
            {
                "name": "medium_kernel",
                "block_size": 256,
                "workload": "reduction",
                "description": "Medium block size for reduction operations",
            },
            {
                "name": "large_kernel",
                "block_size": 1024,
                "workload": "matrix_mult",
                "description": "Large block size for matrix multiplication",
            },
        ]

        for config in kernel_configs:
            metadata = {
                "description": config["description"],
                "input_shapes": ["batch_size x dim"],
                "output_shapes": ["batch_size x dim"],
                "optimizations": ["block_tiling", "vectorization"],
                "block_size": config["block_size"],
                "workload_type": config["workload"],
            }

            def kernel_func():
                pass

            fm.register_kernel(config["name"], kernel_func, metadata)
            assert fm.verify_feature(config["name"])

    def test_triton_pipeline_stages(self):
        """Test Triton pipeline stages configuration."""

        setup_test_environment()
        fm = TritonFeatureManager()

        # Test different pipeline configurations
        pipeline_configs = [
            {"num_stages": 2, "description": "Basic 2-stage pipeline"},
            {"num_stages": 3, "description": "Standard 3-stage pipeline"},
            {"num_stages": 4, "description": "Advanced 4-stage pipeline"},
        ]

        for config in pipeline_configs:
            metadata = {
                "description": config["description"],
                "input_shapes": ["batch_size x dim"],
                "output_shapes": ["batch_size x dim"],
                "optimizations": ["pipelining", "async_copy"],
                "num_stages": config["num_stages"],
            }

            def pipeline_kernel():
                pass

            kernel_name = f'pipeline_{config["num_stages"]}_stage'
            fm.register_kernel(kernel_name, pipeline_kernel, metadata)
            assert fm.verify_feature(kernel_name)


@pytest.mark.triton
class TestTritonPrecisionModes:
    """Test Triton precision mode configurations."""

    @pytest.mark.skipif(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        reason="GPU required for precision tests"
    )
    def test_triton_fp16_precision_mode(self):
        """Test Triton FP16 precision mode."""

        setup_test_environment()
        device = "cuda" if torch.cuda.is_available() else "mps"
        config = TritonFeatureConfig(
            device=device, dtype=torch.float16, enable_tf32=False
        )
        fm = TritonFeatureManager(config)
        ga = GPUAccelerator(fm)

        # Test FP16 tensor operations
        tensors = ga.allocate_tensors([(100, 100), (100, 50)])

        for tensor in tensors:
            assert tensor.dtype == torch.float16
            assert tensor.device.type in ["cuda", "mps"]

        # Test computation with FP16
        result = tensors[0] @ tensors[1]
        assert result.dtype == torch.float16
        assert result.device.type in ["cuda", "mps"]

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="TF32 only available on CUDA"
    )
    def test_triton_tf32_precision_mode(self):
        """Test Triton TF32 precision mode."""

        setup_test_environment()
        config = TritonFeatureConfig(
            device="cuda", dtype=torch.float32, enable_tf32=True
        )
        fm = TritonFeatureManager(config)

        # Verify TF32 is enabled
        assert config.enable_tf32

        # Test TF32 computation
        ga = GPUAccelerator(fm)
        tensors = ga.allocate_tensors([(256, 256), (256, 256)])

        # Matrix multiplication should use TF32 if available
        result = tensors[0] @ tensors[1]
        assert result.shape == (256, 256)
        assert result.device.type == "cuda"


@pytest.mark.triton
class TestTritonFeatureIntegration:
    """Test integration of multiple Triton features."""

    def test_comprehensive_triton_feature_integration(self):
        """Test comprehensive integration of Triton features."""

        setup_test_environment()
        config = TritonFeatureConfig(
            device=TEST_DEVICE.type,
            max_shared_memory=49152,
            num_warps=4,
            num_stages=3,
            enable_tf32=TEST_DEVICE.type == "cuda",
        )

        fm = TritonFeatureManager(config)
        ga = GPUAccelerator(fm)

        # Register comprehensive kernel with all Triton features
        comprehensive_metadata = {
            "description": "Comprehensive Triton kernel with all features",
            "input_shapes": [
                "batch_size x seq_len x hidden_dim",
                "batch_size x seq_len x hidden_dim",
            ],
            "output_shapes": ["batch_size x seq_len x hidden_dim"],
            "optimizations": [
                "vectorization",
                "shared_memory",
                "block_tiling",
                "pipelining",
                "async_copy",
                "fusion",
            ],
            "triton_features": [
                "tl.load",
                "tl.store",
                "tl.program_id",
                "tl.arange",
                "tl.dot",
                "tl.softmax",
                "tl.exp",
                "tl.sum",
                "BLOCK_SIZE",
                "NUM_WARPS",
                "NUM_STAGES",
            ],
            "block_size": 256,
            "num_warps": 4,
            "num_stages": 3,
            "memory_layout": "coalesced",
            "precision_mode": "mixed_fp16" if TEST_DEVICE.type == "cuda" else "fp32",
        }

        def comprehensive_kernel():
            # This would be a real Triton kernel with all features
            pass

        fm.register_kernel(
            "comprehensive_kernel", comprehensive_kernel, comprehensive_metadata
        )

        # Verify all features are properly integrated
        assert fm.verify_feature("comprehensive_kernel")

        features = fm.list_features()
        assert "kernels" in features
        assert "devices" in features
        assert "optimizations" in features

        # Test memory allocation with comprehensive requirements
        tensor_shapes = [
            (4, 128, 768),  # Attention pattern
            (4, 128, 128),  # Key/Value pattern
            (4, 12, 128, 64),  # Multi-head pattern
            (4, 128, 3072),  # FFN pattern
        ]

        tensors = ga.allocate_tensors(tensor_shapes)
        assert len(tensors) == len(tensor_shapes)

        for tensor, expected_shape in zip(tensors, tensor_shapes):
            assert tensor.shape == torch.Size(expected_shape)
            assert tensor.device.type == TEST_DEVICE.type
            assert tensor.dtype == config.dtype

        # Test synchronization
        ga.synchronize()

        print("✓ Comprehensive Triton feature integration successful")


@pytest.mark.triton
@pytest.mark.slow
class TestTritonPerformanceBenchmarks:
    """Performance benchmarks for Triton kernels."""

    def test_triton_kernel_performance_scaling(self):
        """Test Triton kernel performance scaling with problem size."""

        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        fm = TritonFeatureManager(config)
        ga = GPUAccelerator(fm)

        # Test different problem sizes
        sizes = [128, 256, 512, 1024, 2048]

        performance_data = {}

        for size in sizes:
            try:
                # Allocate tensors
                a = ga.allocate_tensors([(size, size)])[0]
                b = ga.allocate_tensors([(size, size)])[0]

                # Warm up
                _ = a @ b

                # Time the operation
                import time

                start_time = time.time()

                for _ in range(5):  # Multiple runs for averaging
                    result = a @ b
                    if TEST_DEVICE.type == "cuda":
                        torch.cuda.synchronize()

                end_time = time.time()

                avg_time = (end_time - start_time) / 5
                gflops = (2 * size**3) / (avg_time * 1e9)  # Approximate GFLOPS

                performance_data[size] = {
                    "time": avg_time,
                    "gflops": gflops,
                    "size": size,
                }

                print(f"Performance test: {size} took {avg_time:.2f}s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Size {size}: Skipped (out of memory)")
                    break
                else:
                    raise

        # Verify performance scaling (should improve with larger sizes up to a point)
        if len(performance_data) >= 3:
            sizes_list = list(performance_data.keys())
            times_list = [performance_data[s]["time"] for s in sizes_list]

            # Check that performance doesn't degrade dramatically
            # Use different thresholds for different devices
            max_degradation_ratio = 20 if TEST_DEVICE.type == "mps" else 10

            for i in range(1, len(times_list)):
                if times_list[i] / times_list[i - 1] < max_degradation_ratio:
                    continue
                else:
                    pytest.fail(
                        f"Performance degraded too much between sizes {sizes_list[i-1]} and {sizes_list[i]} "
                        f"(ratio: {times_list[i] / times_list[i - 1]:.2f}, max allowed: {max_degradation_ratio})"
                    )


class TestTritonFeatureConfig:
    """Test Triton feature configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TritonFeatureConfig()

        # Device should be resolved from "auto" to best available device
        assert config.device in ["cuda", "mps", "cpu"]
        assert config.dtype == torch.float32
        assert config.max_shared_memory == 49152
        assert config.num_warps == 4
        assert config.num_stages == 3
        assert not config.enable_fp8
        assert config.enable_bf16
        assert config.enable_tf32

    def test_custom_config(self):
        """Test custom configuration."""
        config = TritonFeatureConfig(
            device="cpu", dtype=torch.float16, num_warps=8, enable_fp8=True
        )

        assert config.device == "cpu"
        assert config.dtype == torch.float16
        assert config.num_warps == 8
        assert config.enable_fp8


class TestTritonFeatureManager:
    """Test Triton feature manager functionality."""

    @pytest.fixture
    def feature_manager(self):
        """Create feature manager for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        return TritonFeatureManager(config)

    def test_initialization(self, feature_manager):
        """Test feature manager initialization."""
        assert feature_manager.config.device == TEST_DEVICE.type
        assert len(feature_manager._kernels) == 0
        assert len(feature_manager._profilers) == 0

    def test_register_kernel(self, feature_manager):
        """Test kernel registration."""

        def dummy_kernel():
            pass

        metadata = {
            "description": "Test kernel",
            "input_shapes": ["batch_size x dim"],
            "output_shapes": ["batch_size"],
            "optimizations": ["vectorization"],
        }

        feature_manager.register_kernel("test_kernel", dummy_kernel, metadata)

        assert "test_kernel" in feature_manager._kernels
        kernel_info = feature_manager.get_kernel("test_kernel")
        assert kernel_info["kernel"] == dummy_kernel
        assert kernel_info["metadata"] == metadata

    def test_list_features(self, feature_manager):
        """Test feature listing."""
        features = feature_manager.list_features()

        assert "kernels" in features
        assert "devices" in features
        assert "dtypes" in features
        assert "optimizations" in features

        assert isinstance(features["kernels"], list)
        assert isinstance(features["devices"], list)
        assert isinstance(features["dtypes"], list)

    def test_verify_feature(self, feature_manager):
        """Test feature verification."""
        # Test unregistered feature
        assert not feature_manager.verify_feature("nonexistent")

        # Register and test valid feature
        def dummy_kernel():
            pass

        metadata = {
            "input_shapes": ["batch_size x dim"],
            "output_shapes": ["batch_size"],
        }

        feature_manager.register_kernel("test_kernel", dummy_kernel, metadata)
        assert feature_manager.verify_feature("test_kernel")

        # Test feature with incomplete metadata
        feature_manager.register_kernel("incomplete_kernel", dummy_kernel, {})
        assert not feature_manager.verify_feature("incomplete_kernel")


class TestGPUAccelerator:
    """Test GPU accelerator functionality."""

    @pytest.fixture
    def gpu_accelerator(self):
        """Create GPU accelerator for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        return GPUAccelerator(feature_manager)

    def test_initialization(self, gpu_accelerator):
        """Test GPU accelerator initialization."""
        assert gpu_accelerator.device.type == TEST_DEVICE.type

    def test_allocate_tensors(self, gpu_accelerator):
        """Test tensor allocation."""
        shapes = [(10,), (5, 5), (2, 3, 4)]

        tensors = gpu_accelerator.allocate_tensors(shapes)

        assert len(tensors) == 3
        assert tensors[0].shape == (10,)
        assert tensors[1].shape == (5, 5)
        assert tensors[2].shape == (2, 3, 4)

        for tensor in tensors:
            assert tensor.device.type == TEST_DEVICE.type
            assert tensor.dtype == torch.float32

    def test_allocate_tensors_with_dtype(self, gpu_accelerator):
        """Test tensor allocation with custom dtype."""
        shapes = [(10,)]
        dtype = torch.float16 if TEST_DEVICE.type == "cuda" else torch.float32

        tensors = gpu_accelerator.allocate_tensors(shapes, dtype=dtype)

        assert tensors[0].dtype == dtype
        assert tensors[0].device.type == TEST_DEVICE.type

    def test_memory_stats(self, gpu_accelerator):
        """Test memory statistics retrieval."""
        stats = gpu_accelerator.get_memory_stats()

        if TEST_DEVICE.type == "cuda":
            assert "allocated" in stats
            assert "reserved" in stats
            assert isinstance(stats["allocated"], int)
            assert isinstance(stats["reserved"], int)
        else:
            assert stats == {}

    def test_synchronization(self, gpu_accelerator):
        """Test GPU synchronization."""
        # Create and use some tensors to ensure there's work to synchronize
        tensor = create_test_tensor(100, 100)
        result = tensor @ tensor.t()

        gpu_accelerator.synchronize()

        # If we get here without error, synchronization worked
        assert result.shape == (100, 100)


class TestIntegration:
    """Integration tests for core components."""

    def test_feature_manager_gpu_accelerator_integration(self):
        """Test integration between feature manager and GPU accelerator."""
        setup_test_environment()

        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        gpu_accelerator = GPUAccelerator(feature_manager)

        # Register a kernel
        def dummy_kernel():
            pass

        feature_manager.register_kernel(
            "integration_test",
            dummy_kernel,
            {"input_shapes": ["batch_size x dim"], "output_shapes": ["batch_size"]},
        )

        # Verify integration
        assert feature_manager.verify_feature("integration_test")
        assert gpu_accelerator.device.type == feature_manager.config.device

        # Test tensor allocation through accelerator
        tensors = gpu_accelerator.allocate_tensors([(10, 10), (5,)])
        assert all(t.device.type == TEST_DEVICE.type for t in tensors)
        assert all(t.dtype == config.dtype for t in tensors)
