"""
Tests for Message Passing Algorithms

Tests the BeliefPropagation and VariationalMessagePassing classes.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from message_passing import BeliefPropagation, VariationalMessagePassing, MessagePassing
from core import TritonFeatureManager, TritonFeatureConfig
from . import (
    setup_test_environment,
    create_test_tensor,
    assert_tensors_close,
    TEST_DEVICE,
)


class TestMessagePassingBase:
    """Test base message passing functionality."""

    @pytest.fixture
    def message_passer(self):
        """Create message passing instance for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        return MessagePassing(feature_manager)

    def test_initialization(self, message_passer):
        """Test message passing initialization."""
        assert isinstance(message_passer.feature_manager, TritonFeatureManager)
        assert message_passer.graph_structure is None

    def test_set_graph(self, message_passer):
        """Test graph structure setup."""
        num_nodes = 4
        adjacency = torch.tensor(
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
            dtype=torch.float,
            device=TEST_DEVICE,
        )

        node_potentials = torch.softmax(create_test_tensor(num_nodes, 3), dim=1)

        message_passer.set_graph(adjacency, node_potentials)

        assert message_passer.adjacency is adjacency
        assert message_passer.node_potentials is node_potentials
        assert message_passer.num_nodes == num_nodes
        assert message_passer.num_states == 3

    def test_initialize_messages(self, message_passer):
        """Test message initialization."""
        num_nodes = 3
        adjacency = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float, device=TEST_DEVICE
        )

        node_potentials = torch.softmax(create_test_tensor(num_nodes, 2), dim=1)
        message_passer.set_graph(adjacency, node_potentials)

        messages = message_passer.initialize_messages()

        # Should have messages for each edge (both directions)
        expected_num_edges = 4  # (0,1), (1,0), (1,2), (2,1)
        assert messages.shape == (expected_num_edges, 2)

        # All messages should be uniform
        assert_tensors_close(messages, torch.ones_like(messages) / 2)


class TestBeliefPropagation:
    """Test belief propagation algorithm."""

    @pytest.fixture
    def bp_engine(self):
        """Create belief propagation engine for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        return BeliefPropagation(feature_manager)

    def test_initialization(self, bp_engine):
        """Test BP engine initialization."""
        assert isinstance(bp_engine.feature_manager, TritonFeatureManager)
        assert "belief_propagation" in bp_engine.feature_manager._kernels

    def test_run_tree_graph(self, bp_engine):
        """Test belief propagation on a tree-structured graph."""
        # Create a simple tree: 0 - 1 - 2
        adjacency = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float, device=TEST_DEVICE
        )

        # Strong evidence for state 0 at node 0
        node_potentials = torch.tensor(
            [
                [10.0, 0.1],  # Node 0 strongly favors state 0
                [1.0, 1.0],  # Node 1 neutral
                [0.1, 10.0],  # Node 2 strongly favors state 1
            ],
            dtype=torch.float,
            device=TEST_DEVICE,
        )

        bp_engine.set_graph(adjacency, node_potentials)

        result = bp_engine.run(max_iterations=10, tolerance=1e-4)

        assert "beliefs" in result
        assert "messages" in result
        assert "converged" in result

        beliefs = result["beliefs"]
        assert beliefs.shape == (3, 2)

        # Beliefs should be normalized
        belief_sums = beliefs.sum(dim=1)
        assert_tensors_close(belief_sums, torch.ones(3, device=TEST_DEVICE))

        # Node 0 should still strongly favor state 0
        assert beliefs[0, 0] > 0.8
        # Node 2 should still strongly favor state 1
        assert beliefs[2, 1] > 0.8

    def test_run_without_graph(self, bp_engine):
        """Test BP run without setting graph."""
        with pytest.raises(ValueError, match="Graph structure not set"):
            bp_engine.run()


class TestVariationalMessagePassing:
    """Test variational message passing."""

    @pytest.fixture
    def vmp_engine(self):
        """Create VMP engine for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        return VariationalMessagePassing(feature_manager)

    def test_initialization(self, vmp_engine):
        """Test VMP engine initialization."""
        assert isinstance(vmp_engine.feature_manager, TritonFeatureManager)
        assert "variational_message_passing" in vmp_engine.feature_manager._kernels

    def test_run_simple_graph(self, vmp_engine):
        """Test VMP on a simple graph."""
        # Create a small complete graph
        adjacency = torch.tensor(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float, device=TEST_DEVICE
        )

        node_potentials = torch.softmax(create_test_tensor(3, 2), dim=1)
        vmp_engine.set_graph(adjacency, node_potentials)

        result = vmp_engine.run(max_iterations=5)

        assert "beliefs" in result
        assert "free_energy" in result

        beliefs = result["beliefs"]
        free_energy = result["free_energy"]

        assert beliefs.shape == (3, 2)

        # Beliefs should be normalized
        belief_sums = beliefs.sum(dim=1)
        assert_tensors_close(belief_sums, torch.ones(3, device=TEST_DEVICE))

        # Free energy should be finite
        assert torch.isfinite(free_energy)


class TestMessagePassingIntegration:
    """Integration tests for message passing algorithms."""

    def test_bp_vmp_comparison(self):
        """Compare belief propagation and variational message passing."""
        setup_test_environment()

        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)

        # Create same graph for both algorithms
        adjacency = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float, device=TEST_DEVICE
        )

        node_potentials = torch.softmax(create_test_tensor(3, 2), dim=1)

        # Belief Propagation
        bp_engine = BeliefPropagation(feature_manager)
        bp_engine.set_graph(adjacency.clone(), node_potentials.clone())

        bp_result = bp_engine.run(max_iterations=10)

        # Variational Message Passing
        vmp_engine = VariationalMessagePassing(feature_manager)
        vmp_engine.set_graph(adjacency, node_potentials)

        vmp_result = vmp_engine.run(max_iterations=10)

        # Both should produce valid results
        assert bp_result["beliefs"].shape == vmp_result["beliefs"].shape
        assert torch.all(torch.isfinite(bp_result["beliefs"]))
        assert torch.all(torch.isfinite(vmp_result["beliefs"]))

        # Beliefs should be normalized for both
        for result in [bp_result, vmp_result]:
            belief_sums = result["beliefs"].sum(dim=1)
            assert_tensors_close(belief_sums, torch.ones(3, device=TEST_DEVICE))

    def test_message_passing_convergence(self):
        """Test convergence properties of message passing."""
        setup_test_environment()

        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        bp_engine = BeliefPropagation(feature_manager)

        # Create a chain graph: 0 - 1 - 2 - 3
        adjacency = torch.tensor(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            dtype=torch.float,
            device=TEST_DEVICE,
        )

        # Create smooth potentials (favoring consistent states)
        node_potentials = torch.tensor(
            [
                [2.0, 1.0],  # Slight preference for state 0
                [1.5, 1.5],  # Neutral
                [1.5, 1.5],  # Neutral
                [1.0, 2.0],  # Slight preference for state 1
            ],
            dtype=torch.float,
            device=TEST_DEVICE,
        )

        bp_engine.set_graph(adjacency, node_potentials)

        result = bp_engine.run(max_iterations=20, tolerance=1e-5)

        # Should converge
        assert result["converged"]

        beliefs = result["beliefs"]

        # Check that beliefs reflect the chain structure
        # Nodes should have correlated beliefs due to connectivity
        assert beliefs.shape == (4, 2)
        assert torch.all(beliefs >= 0)
        belief_sums = beliefs.sum(dim=1)
        assert_tensors_close(belief_sums, torch.ones(4, device=TEST_DEVICE))

    def test_loopy_graph_behavior(self):
        """Test message passing on graphs with loops."""
        setup_test_environment()

        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)

        # Create a loopy graph (triangle)
        adjacency = torch.tensor(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float, device=TEST_DEVICE
        )

        node_potentials = torch.softmax(create_test_tensor(3, 2), dim=1)

        # Test both BP and VMP on loopy graph
        bp_engine = BeliefPropagation(feature_manager)
        bp_engine.set_graph(adjacency.clone(), node_potentials.clone())

        vmp_engine = VariationalMessagePassing(feature_manager)
        vmp_engine.set_graph(adjacency, node_potentials)

        bp_result = bp_engine.run(max_iterations=15)
        vmp_result = vmp_engine.run(max_iterations=15)

        # Both should handle loopy graphs gracefully
        assert torch.all(torch.isfinite(bp_result["beliefs"]))
        assert torch.all(torch.isfinite(vmp_result["beliefs"]))
        assert torch.all(torch.isfinite(vmp_result["free_energy"]))

        # VMP might not converge in exact same way as BP on loopy graphs
        # but should still produce reasonable results
        for result in [bp_result, vmp_result]:
            beliefs = result["beliefs"]
            assert beliefs.shape == (3, 2)
            belief_sums = beliefs.sum(dim=1)
            assert_tensors_close(belief_sums, torch.ones(3, device=TEST_DEVICE))
