"""
Active Inference Test Suite

Comprehensive test suite for active inference and free energy principle implementations.
Follows test-driven development (TDD) principles with real data analysis.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Test configuration
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DTYPE = torch.float32
TEST_TOLERANCE = 1e-6

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def setup_test_environment():
    """Set up test environment with proper configurations."""
    if TEST_DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Set random seeds for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)


def create_test_tensor(*shape, dtype=None, device=None, requires_grad=False):
    """Create test tensor with standard configuration."""
    dtype = dtype or TEST_DTYPE
    device = device or TEST_DEVICE

    if len(shape) == 1:
        # 1D tensor
        data = torch.randn(*shape, dtype=dtype, device=device)
    else:
        # Multi-dimensional tensor
        data = torch.randn(*shape, dtype=dtype, device=device)

    if requires_grad:
        data.requires_grad_(True)

    return data


def assert_tensors_close(actual, expected, rtol=TEST_TOLERANCE, atol=TEST_TOLERANCE):
    """Assert that two tensors are close within tolerance."""
    assert torch.allclose(
        actual, expected, rtol=rtol, atol=atol
    ), f"Tensors not close:\nActual: {actual}\nExpected: {expected}"


def create_synthetic_data(num_samples=100, feature_dim=10, num_classes=5):
    """Create synthetic data for testing."""
    # Generate features
    features = torch.randn(
        num_samples, feature_dim, dtype=TEST_DTYPE, device=TEST_DEVICE
    )

    # Generate labels
    labels = torch.randint(0, num_classes, (num_samples,), device=TEST_DEVICE)

    # Generate some structure (clusters)
    centers = torch.randn(
        num_classes, feature_dim, dtype=TEST_DTYPE, device=TEST_DEVICE
    )
    for i in range(num_samples):
        features[i] += centers[labels[i]] * 0.5

    return features, labels, centers


# Export test utilities
__all__ = [
    "setup_test_environment",
    "create_test_tensor",
    "assert_tensors_close",
    "create_synthetic_data",
    "TEST_DEVICE",
    "TEST_DTYPE",
    "TEST_TOLERANCE",
]
