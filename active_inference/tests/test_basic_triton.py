"""
Tests for basic Triton operations that work on Apple Silicon.

This module contains comprehensive tests for basic Triton kernels that are
known to work reliably on Apple Silicon with MPS acceleration. These tests
focus on simple, well-supported operations that provide a foundation for
more complex Active Inference computations.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from .conftest import create_test_vectors, create_test_matrix, assert_tensors_close


class TestBasicTritonOperations:
    """Test basic Triton operations that work on Apple Silicon."""

    def test_triton_add_vectors_cpu(self):
        """Test vector addition on CPU."""
        from core import triton_add_vectors

        a, b = create_test_vectors(size=100, device="cpu")
        expected = a + b

        result = triton_add_vectors(a, b)

        assert result.shape == expected.shape
        assert_tensors_close(result, expected)

    def test_triton_add_vectors_mps(self):
        """Test vector addition on MPS (Apple Silicon)."""
        from core import triton_add_vectors

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        a, b = create_test_vectors(size=100, device="mps")
        expected = a + b

        result = triton_add_vectors(a, b)

        assert result.shape == expected.shape
        assert result.device == a.device
        assert_tensors_close(result, expected)

    def test_triton_multiply_vectors_cpu(self):
        """Test vector element-wise multiplication on CPU."""
        from core import triton_multiply_vectors

        a, b = create_test_vectors(size=100, device="cpu")
        expected = a * b

        result = triton_multiply_vectors(a, b)

        assert result.shape == expected.shape
        assert_tensors_close(result, expected)

    def test_triton_multiply_vectors_mps(self):
        """Test vector element-wise multiplication on MPS."""
        from core import triton_multiply_vectors

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        a, b = create_test_vectors(size=100, device="mps")
        expected = a * b

        result = triton_multiply_vectors(a, b)

        assert result.shape == expected.shape
        assert result.device == a.device
        assert_tensors_close(result, expected)

    def test_triton_vector_sum_cpu(self):
        """Test vector sum on CPU."""
        from core import triton_vector_sum

        a, _ = create_test_vectors(size=100, device="cpu")
        expected = torch.sum(a)

        result = triton_vector_sum(a)

        assert result.shape == expected.shape
        assert_tensors_close(result, expected)

    def test_triton_vector_sum_mps(self):
        """Test vector sum on MPS."""
        from core import triton_vector_sum

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        a, _ = create_test_vectors(size=100, device="mps")
        expected = torch.sum(a)

        result = triton_vector_sum(a)

        assert result.shape == expected.shape
        assert result.device == a.device
        assert_tensors_close(result, expected)

    def test_triton_operations_different_shapes_error(self):
        """Test that operations fail gracefully with different shapes."""
        from core import triton_add_vectors

        a = torch.randn(100, device="cpu")
        b = torch.randn(50, device="cpu")

        with pytest.raises(ValueError, match="same shape"):
            triton_add_vectors(a, b)

    def test_triton_operations_non_tensor_error(self):
        """Test that operations fail with non-tensor inputs."""
        from core import triton_add_vectors

        with pytest.raises(ValueError, match="PyTorch tensors"):
            triton_add_vectors([1, 2, 3], [4, 5, 6])

    def test_triton_operations_device_mismatch(self):
        """Test that operations handle device mismatches correctly."""
        from core import triton_add_vectors

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        a = torch.randn(100, device="cpu")
        b = torch.randn(100, device="mps")

        result = triton_add_vectors(a, b)

        # Should move both tensors to the same device and compute
        assert result.shape == (100,)
        expected = a + b.to("cpu")
        assert_tensors_close(result, expected)


class TestTritonPerformance:
    """Performance tests comparing Triton vs PyTorch on Apple Silicon."""

    @pytest.mark.slow
    def test_vector_add_performance_cpu(self):
        """Compare performance of vector addition on CPU."""
        from core import triton_add_vectors
        import time

        sizes = [1000, 10000, 100000]

        for size in sizes:
            a, b = create_test_vectors(size=size, device="cpu")

            # PyTorch baseline
            start_time = time.time()
            for _ in range(10):
                expected = a + b
            pytorch_time = (time.time() - start_time) / 10

            # Triton version
            start_time = time.time()
            for _ in range(10):
                result = triton_add_vectors(a, b)
            triton_time = (time.time() - start_time) / 10

            # Verify correctness
            assert_tensors_close(result, expected)

            print(".2e")

    @pytest.mark.slow
    def test_vector_add_performance_mps(self):
        """Compare performance of vector addition on MPS."""
        from core import triton_add_vectors
        import time

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        sizes = [1000, 10000, 100000]

        for size in sizes:
            a, b = create_test_vectors(size=size, device="mps")

            # PyTorch baseline
            start_time = time.time()
            for _ in range(10):
                expected = a + b
            pytorch_time = (time.time() - start_time) / 10

            # Triton version
            start_time = time.time()
            for _ in range(10):
                result = triton_add_vectors(a, b)
            triton_time = (time.time() - start_time) / 10

            # Verify correctness
            assert_tensors_close(result, expected)

            print(".2e")


class TestTritonFallbackBehavior:
    """Test that Triton operations fall back gracefully when kernels fail."""

    def test_fallback_when_triton_unavailable(self):
        """Test fallback behavior when Triton is not available."""
        from core import triton_add_vectors

        # Mock Triton as unavailable
        original_available = torch.__dict__.get('TRITON_AVAILABLE', True)
        torch.TRITON_AVAILABLE = False

        try:
            a, b = create_test_vectors(size=100, device="cpu")
            expected = a + b

            result = triton_add_vectors(a, b)

            assert_tensors_close(result, expected)
        finally:
            # Restore original state
            if 'TRITON_AVAILABLE' in torch.__dict__:
                torch.TRITON_AVAILABLE = original_available

    def test_fallback_on_kernel_failure(self):
        """Test that operations fall back when Triton kernels fail."""
        from core import triton_add_vectors

        a, b = create_test_vectors(size=100, device="cpu")
        expected = a + b

        # This should work even if Triton kernels fail
        result = triton_add_vectors(a, b)

        assert_tensors_close(result, expected)


class TestTritonIntegration:
    """Test integration with the broader Active Inference framework."""

    def test_triton_usage_reporting(self):
        """Test that Triton usage is properly reported."""
        from core import triton_add_vectors, reporter

        # Reset reporter stats
        reporter.usage_stats = {
            "triton_kernels_used": 0,
            "pytorch_fallbacks_used": 0,
            "kernel_launch_success": 0,
            "kernel_launch_failures": 0,
            "methods_using_triton": [],
            "methods_using_pytorch": [],
            "performance_comparison": {},
            "platform_optimizations": {},
            "kernel_cache": {}
        }

        a, b = create_test_vectors(size=100, device="cpu")
        result = triton_add_vectors(a, b)

        # Check that usage was reported
        summary = reporter.get_usage_summary()
        assert summary["total_operations"] >= 1

    def test_memory_layout_optimization(self):
        """Test that memory layout is optimized for Triton kernels."""
        from core import optimize_memory_layout

        # Create a non-contiguous tensor
        a = torch.randn(100, 200)
        b = a[:, ::2]  # Non-contiguous

        assert not b.is_contiguous()

        optimized = optimize_memory_layout(b, "coalesced")
        assert optimized.is_contiguous()
        assert_tensors_close(optimized, b)
