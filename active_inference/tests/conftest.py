"""
Pytest configuration for active inference tests.

Configures test environment and provides shared fixtures.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from . import setup_test_environment


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "slow: marks tests that are slow to run")
    config.addinivalue_line("markers", "integration: marks integration tests")


@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Set up test session environment."""
    setup_test_environment()


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def device(gpu_available):
    """Provide appropriate device for testing."""
    return torch.device("cuda" if gpu_available else "cpu")


@pytest.fixture
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_graph_data(device):
    """Provide sample graph data for message passing tests."""
    # Create a simple 4-node graph
    adjacency = torch.tensor(
        [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
        dtype=torch.float,
        device=device,
    )

    num_nodes, num_states = 4, 3
    node_potentials = torch.softmax(
        torch.randn(num_nodes, num_states, device=device), dim=1
    )

    return {
        "adjacency": adjacency,
        "node_potentials": node_potentials,
        "num_nodes": num_nodes,
        "num_states": num_states,
    }


@pytest.fixture
def sample_observations(device):
    """Provide sample observation data."""
    batch_size, feature_dim = 10, 5
    observations = torch.randn(batch_size, feature_dim, device=device)
    return observations


@pytest.fixture
def sample_policies(device):
    """Provide sample policy data."""
    num_policies, feature_dim = 5, 5
    policies = torch.randn(num_policies, feature_dim, device=device)
    return policies


@pytest.fixture
def sample_posterior(device):
    """Provide sample posterior distribution."""
    batch_size, feature_dim = 10, 5
    posterior = torch.softmax(
        torch.randn(batch_size, feature_dim, device=device), dim=1
    )
    return posterior


def assert_tensors_close(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Assert that two tensors are close within tolerance."""
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
        f"Tensors not close: max diff = {torch.max(torch.abs(tensor1 - tensor2))}"


def create_synthetic_data(num_samples=100, feature_dim=10, device="cpu"):
    """Create synthetic data for testing."""
    # Generate features
    features = torch.randn(num_samples, feature_dim, device=device)

    # Generate labels (simple classification)
    weights = torch.randn(feature_dim, 2, device=device)
    logits = torch.matmul(features, weights)
    labels = torch.argmax(logits, dim=1)

    return features, labels


def create_test_vectors(size=1000, device="cpu", dtype=torch.float32):
    """Create test vectors for basic operations."""
    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    return a, b


def create_test_matrix(rows=100, cols=50, device="cpu", dtype=torch.float32):
    """Create test matrices for matrix operations."""
    A = torch.randn(rows, cols, dtype=dtype, device=device)
    B = torch.randn(rows, cols, dtype=dtype, device=device)
    return A, B
