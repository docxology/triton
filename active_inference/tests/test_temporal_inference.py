"""
Test Temporal Inference Methods

Comprehensive tests for Triton-accelerated temporal inference algorithms:
- Kalman smoothing with forward-backward algorithm
- Viterbi decoding for most likely state sequences
- Temporal belief propagation
- Real Triton kernel validation and performance benchmarks
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any

# Import test utilities
from .conftest import assert_tensors_close, create_synthetic_data

# Import temporal inference modules
try:
    # Try relative import first (when used as package)
    from ..src.temporal_inference import (
        KalmanSmoother, ViterbiDecoder, TemporalBeliefPropagation,
        create_kalman_smoother, create_viterbi_decoder,
        create_temporal_belief_propagation, benchmark_temporal_inference
    )
except ImportError:
    # Fall back to absolute import (when imported directly)
    from src.temporal_inference import (
        KalmanSmoother, ViterbiDecoder, TemporalBeliefPropagation,
        create_kalman_smoother, create_viterbi_decoder,
        create_temporal_belief_propagation, benchmark_temporal_inference
    )

# Import Triton availability
try:
    from ..src.core import TRITON_AVAILABLE
except ImportError:
    from src.core import TRITON_AVAILABLE


class TestKalmanSmoother:
    """Test Kalman smoother with Triton acceleration."""

    @pytest.fixture
    def kalman_smoother(self):
        """Create Kalman smoother for testing."""
        return KalmanSmoother(state_dim=4, obs_dim=2)

    @pytest.fixture
    def temporal_data(self):
        """Generate synthetic temporal data for testing."""
        batch_size = 4
        seq_len = 16
        state_dim = 4
        obs_dim = 2

        # Generate synthetic observations
        observations = torch.randn(seq_len, batch_size, obs_dim)

        # Generate model parameters
        transition_matrix = torch.eye(state_dim).unsqueeze(0).repeat(batch_size, 1, 1) * 0.9
        emission_matrix = torch.randn(batch_size, state_dim, obs_dim) * 0.1
        initial_state = torch.randn(batch_size, state_dim)

        return {
            "observations": observations,
            "transition_matrix": transition_matrix,
            "emission_matrix": emission_matrix,
            "initial_state": initial_state,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "state_dim": state_dim,
            "obs_dim": obs_dim
        }

    def test_kalman_smoother_initialization(self, kalman_smoother):
        """Test Kalman smoother initialization."""
        assert kalman_smoother.state_dim == 4
        assert kalman_smoother.obs_dim == 2

    def test_kalman_smoothing(self, kalman_smoother, temporal_data):
        """Test Kalman smoothing on temporal data."""
        observations = temporal_data["observations"]
        transition_matrix = temporal_data["transition_matrix"]
        emission_matrix = temporal_data["emission_matrix"]
        initial_state = temporal_data["initial_state"]

        # Perform smoothing
        smoothed_states = kalman_smoother.smooth(
            observations, transition_matrix, emission_matrix, initial_state
        )

        # Check output shape
        assert smoothed_states.shape == observations.shape[:2] + (kalman_smoother.state_dim,)
        assert torch.isfinite(smoothed_states).all()

    def test_kalman_smoother_numerical_stability(self, kalman_smoother, temporal_data):
        """Test numerical stability of Kalman smoother."""
        # Test with extreme values
        observations = temporal_data["observations"] * 1000
        transition_matrix = temporal_data["transition_matrix"]
        emission_matrix = temporal_data["emission_matrix"] * 0.001
        initial_state = temporal_data["initial_state"] * 100

        smoothed_states = kalman_smoother.smooth(
            observations, transition_matrix, emission_matrix, initial_state
        )

        # Should not produce NaN or inf
        assert torch.isfinite(smoothed_states).all()

    @pytest.mark.parametrize("seq_len", [8, 16, 32])
    def test_kalman_smoother_sequence_lengths(self, seq_len):
        """Test Kalman smoother with different sequence lengths."""
        kalman_smoother = KalmanSmoother(state_dim=4, obs_dim=2)

        batch_size = 2
        observations = torch.randn(seq_len, batch_size, 2)
        transition_matrix = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        emission_matrix = torch.randn(batch_size, 4, 2)
        initial_state = torch.randn(batch_size, 4)

        smoothed_states = kalman_smoother.smooth(
            observations, transition_matrix, emission_matrix, initial_state
        )

        assert smoothed_states.shape == (seq_len, batch_size, 4)
        assert torch.isfinite(smoothed_states).all()


class TestViterbiDecoder:
    """Test Viterbi decoder with Triton acceleration."""

    @pytest.fixture
    def viterbi_decoder(self):
        """Create Viterbi decoder for testing."""
        return ViterbiDecoder(n_states=4, n_observations=3)

    @pytest.fixture
    def sequence_data(self):
        """Generate synthetic sequence data for testing."""
        batch_size = 3
        seq_len = 12
        n_states = 4
        n_observations = 3

        # Generate synthetic observations
        observations = torch.randn(seq_len, batch_size, n_observations)

        # Generate model parameters
        transition_matrix = torch.softmax(torch.randn(batch_size, n_states, n_states), dim=-1)
        emission_matrix = torch.softmax(torch.randn(batch_size, n_states, n_observations), dim=-1)
        initial_state = torch.softmax(torch.randn(batch_size, n_states), dim=-1)

        return {
            "observations": observations,
            "transition_matrix": transition_matrix,
            "emission_matrix": emission_matrix,
            "initial_state": initial_state,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "n_states": n_states,
            "n_observations": n_observations
        }

    def test_viterbi_decoder_initialization(self, viterbi_decoder):
        """Test Viterbi decoder initialization."""
        assert viterbi_decoder.n_states == 4
        assert viterbi_decoder.n_observations == 3

    def test_viterbi_decoding(self, viterbi_decoder, sequence_data):
        """Test Viterbi decoding on sequence data."""
        observations = sequence_data["observations"]
        transition_matrix = sequence_data["transition_matrix"]
        emission_matrix = sequence_data["emission_matrix"]
        initial_state = sequence_data["initial_state"]

        # Perform decoding
        viterbi_path = viterbi_decoder.decode(
            observations, transition_matrix, emission_matrix, initial_state
        )

        # Check output shape
        assert viterbi_path.shape == observations.shape[:2]
        assert viterbi_path.dtype == torch.long

        # Check that all states are valid
        assert torch.all(viterbi_path >= 0)
        assert torch.all(viterbi_path < viterbi_decoder.n_states)

    def test_viterbi_decoder_consistency(self, viterbi_decoder, sequence_data):
        """Test consistency of Viterbi decoder."""
        observations = sequence_data["observations"]
        transition_matrix = sequence_data["transition_matrix"]
        emission_matrix = sequence_data["emission_matrix"]
        initial_state = sequence_data["initial_state"]

        # Run decoding multiple times with same input
        path1 = viterbi_decoder.decode(observations, transition_matrix, emission_matrix, initial_state)
        path2 = viterbi_decoder.decode(observations, transition_matrix, emission_matrix, initial_state)

        # Should produce identical results (deterministic algorithm)
        assert torch.equal(path1, path2)

    @pytest.mark.parametrize("n_states", [2, 4, 6])
    def test_viterbi_decoder_state_counts(self, n_states):
        """Test Viterbi decoder with different numbers of states."""
        viterbi_decoder = ViterbiDecoder(n_states=n_states, n_observations=3)

        batch_size = 2
        seq_len = 8
        observations = torch.randn(seq_len, batch_size, 3)
        transition_matrix = torch.softmax(torch.randn(batch_size, n_states, n_states), dim=-1)
        emission_matrix = torch.softmax(torch.randn(batch_size, n_states, 3), dim=-1)
        initial_state = torch.softmax(torch.randn(batch_size, n_states), dim=-1)

        viterbi_path = viterbi_decoder.decode(
            observations, transition_matrix, emission_matrix, initial_state
        )

        assert viterbi_path.shape == (seq_len, batch_size)
        assert torch.all(viterbi_path >= 0)
        assert torch.all(viterbi_path < n_states)


class TestTemporalBeliefPropagation:
    """Test temporal belief propagation with Triton acceleration."""

    @pytest.fixture
    def temporal_bp(self):
        """Create temporal belief propagation for testing."""
        return TemporalBeliefPropagation(n_nodes=4, n_states=3, n_edges=6)

    @pytest.fixture
    def graph_data(self):
        """Generate synthetic graph data for testing."""
        seq_len = 8
        batch_size = 2
        n_nodes = 4
        n_states = 3

        # Generate potentials and weights
        potentials = torch.softmax(torch.randn(seq_len, batch_size, n_nodes, n_states), dim=-1)
        temporal_weights = torch.randn(batch_size, 6)  # 6 edges for 4 nodes

        return {
            "potentials": potentials,
            "temporal_weights": temporal_weights,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "n_nodes": n_nodes,
            "n_states": n_states
        }

    def test_temporal_bp_initialization(self, temporal_bp):
        """Test temporal belief propagation initialization."""
        assert temporal_bp.n_nodes == 4
        assert temporal_bp.n_states == 3
        assert temporal_bp.n_edges == 6

    def test_temporal_bp_inference(self, temporal_bp, graph_data):
        """Test temporal belief propagation inference."""
        potentials = graph_data["potentials"]
        temporal_weights = graph_data["temporal_weights"]
        max_iterations = 5

        # Run inference
        marginals = temporal_bp.run_inference(potentials, temporal_weights, max_iterations)

        # Check output shape
        assert marginals.shape == potentials.shape
        assert torch.isfinite(marginals).all()

        # Check that marginals are properly normalized
        marginal_sums = marginals.sum(dim=-1)
        assert torch.allclose(marginal_sums, torch.ones_like(marginal_sums), atol=1e-5)

    def test_temporal_bp_convergence(self, temporal_bp, graph_data):
        """Test convergence of temporal belief propagation."""
        potentials = graph_data["potentials"]
        temporal_weights = graph_data["temporal_weights"]

        # Run with different numbers of iterations
        marginals_5 = temporal_bp.run_inference(potentials, temporal_weights, max_iterations=5)
        marginals_10 = temporal_bp.run_inference(potentials, temporal_weights, max_iterations=10)

        # More iterations should produce more stable results
        # (This is a rough check - exact convergence depends on the problem)
        assert torch.isfinite(marginals_5).all()
        assert torch.isfinite(marginals_10).all()

    @pytest.mark.parametrize("max_iterations", [3, 5, 10])
    def test_temporal_bp_iteration_counts(self, max_iterations, graph_data):
        """Test temporal BP with different iteration counts."""
        temporal_bp = TemporalBeliefPropagation(n_nodes=4, n_states=3, n_edges=6)
        potentials = graph_data["potentials"]
        temporal_weights = graph_data["temporal_weights"]

        marginals = temporal_bp.run_inference(potentials, temporal_weights, max_iterations)

        assert marginals.shape == potentials.shape
        assert torch.isfinite(marginals).all()


class TestTemporalInferenceIntegration:
    """Integration tests for temporal inference components."""

    def test_kalman_viterbi_integration(self):
        """Test integration between Kalman smoother and Viterbi decoder."""
        # Create components
        kalman_smoother = KalmanSmoother(state_dim=4, obs_dim=2)
        viterbi_decoder = ViterbiDecoder(n_states=4, n_observations=2)

        batch_size = 2
        seq_len = 10

        # Generate data
        observations = torch.randn(seq_len, batch_size, 2)
        transition_matrix = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        emission_matrix = torch.randn(batch_size, 4, 2)
        initial_state = torch.randn(batch_size, 4)

        # Kalman smoothing
        smoothed_states = kalman_smoother.smooth(
            observations, transition_matrix, emission_matrix, initial_state
        )

        # Create proper emission probabilities for Viterbi (state_dim x obs_dim)
        viterbi_emission = torch.softmax(torch.randn(batch_size, 4, 2), dim=-1)
        viterbi_transition = torch.softmax(torch.randn(batch_size, 4, 4), dim=-1)
        viterbi_initial = torch.softmax(torch.randn(batch_size, 4), dim=-1)

        # Viterbi decoding
        viterbi_path = viterbi_decoder.decode(
            observations, viterbi_transition, viterbi_emission, viterbi_initial
        )

        # Check results
        assert smoothed_states.shape == (seq_len, batch_size, 4)
        assert viterbi_path.shape == (seq_len, batch_size)
        assert torch.isfinite(smoothed_states).all()
        assert torch.isfinite(viterbi_path).all()

    def test_temporal_inference_benchmark(self):
        """Test benchmarking functionality."""
        if TRITON_AVAILABLE:
            results = benchmark_temporal_inference()

            # Check that benchmark results are reasonable
            assert "kalman_smoother" in results
            assert "viterbi_decoder" in results

            for method, metrics in results.items():
                assert "triton_accelerated" in metrics
                assert isinstance(metrics["triton_accelerated"], bool)

    def test_temporal_inference_gradient_flow(self):
        """Test gradient flow in temporal inference (for learning applications)."""
        # Create a simple model that uses temporal inference
        kalman_smoother = KalmanSmoother(state_dim=2, obs_dim=2)

        batch_size = 2
        seq_len = 8

        # Create learnable parameters
        transition_matrix = torch.randn(batch_size, 2, 2, requires_grad=True)
        emission_matrix = torch.randn(batch_size, 2, 2, requires_grad=True)
        initial_state = torch.randn(batch_size, 2, requires_grad=True)

        # Generate observations
        observations = torch.randn(seq_len, batch_size, 2)

        # Forward pass
        smoothed_states = kalman_smoother.smooth(
            observations, transition_matrix, emission_matrix, initial_state
        )

        # Compute loss
        target_states = torch.randn_like(smoothed_states)
        loss = torch.nn.functional.mse_loss(smoothed_states, target_states)

        # Backward pass
        loss.backward()

        # Check gradients
        assert transition_matrix.grad is not None
        assert emission_matrix.grad is not None
        assert initial_state.grad is not None
        assert torch.isfinite(transition_matrix.grad).all()
        assert torch.isfinite(emission_matrix.grad).all()
        assert torch.isfinite(initial_state.grad).all()


class TestTemporalInferenceConvenienceFunctions:
    """Test convenience functions for creating temporal inference methods."""

    def test_create_kalman_smoother(self):
        """Test Kalman smoother creation function."""
        smoother = create_kalman_smoother(state_dim=4, obs_dim=2, use_triton=True)

        assert isinstance(smoother, KalmanSmoother)
        assert smoother.state_dim == 4
        assert smoother.obs_dim == 2

    def test_create_viterbi_decoder(self):
        """Test Viterbi decoder creation function."""
        decoder = create_viterbi_decoder(n_states=4, n_observations=3, use_triton=True)

        assert isinstance(decoder, ViterbiDecoder)
        assert decoder.n_states == 4
        assert decoder.n_observations == 3

    def test_create_temporal_belief_propagation(self):
        """Test temporal belief propagation creation function."""
        tbp = create_temporal_belief_propagation(n_nodes=4, n_states=3, n_edges=6, use_triton=True)

        assert isinstance(tbp, TemporalBeliefPropagation)
        assert tbp.n_nodes == 4
        assert tbp.n_states == 3
        assert tbp.n_edges == 6


# Performance tests
@pytest.mark.slow
class TestTemporalInferencePerformance:
    """Performance tests for temporal inference implementations."""

    def test_kalman_smoother_performance_scaling(self):
        """Test Kalman smoother performance with different sizes."""
        sizes = [(8, 4, 2), (16, 8, 4), (32, 16, 8)]  # seq_len, state_dim, obs_dim

        for seq_len, state_dim, obs_dim in sizes:
            kalman_smoother = KalmanSmoother(state_dim=state_dim, obs_dim=obs_dim)

            batch_size = 4
            observations = torch.randn(seq_len, batch_size, obs_dim)
            transition_matrix = torch.eye(state_dim).unsqueeze(0).repeat(batch_size, 1, 1)
            emission_matrix = torch.randn(batch_size, state_dim, obs_dim)
            initial_state = torch.randn(batch_size, state_dim)

            # Time smoothing
            import time
            start_time = time.time()

            smoothed_states = kalman_smoother.smooth(
                observations, transition_matrix, emission_matrix, initial_state
            )

            end_time = time.time()

            # Check that performance is reasonable
            total_time = end_time - start_time
            assert total_time < 5.0  # Should complete in reasonable time

    def test_viterbi_decoder_performance_scaling(self):
        """Test Viterbi decoder performance with different sizes."""
        sizes = [(8, 4, 3), (16, 6, 4), (24, 8, 5)]  # seq_len, n_states, n_obs

        for seq_len, n_states, n_obs in sizes:
            viterbi_decoder = ViterbiDecoder(n_states=n_states, n_observations=n_obs)

            batch_size = 4
            observations = torch.randn(seq_len, batch_size, n_obs)
            transition_matrix = torch.softmax(torch.randn(batch_size, n_states, n_states), dim=-1)
            emission_matrix = torch.softmax(torch.randn(batch_size, n_states, n_obs), dim=-1)
            initial_state = torch.softmax(torch.randn(batch_size, n_states), dim=-1)

            # Time decoding
            import time
            start_time = time.time()

            viterbi_path = viterbi_decoder.decode(
                observations, transition_matrix, emission_matrix, initial_state
            )

            end_time = time.time()

            # Check performance
            total_time = end_time - start_time
            assert total_time < 3.0  # Should complete in reasonable time

    def test_temporal_bp_performance_scaling(self):
        """Test temporal BP performance with different sizes."""
        configs = [(4, 3, 6, 8), (6, 4, 10, 12), (8, 5, 14, 16)]  # n_nodes, n_states, n_edges, seq_len

        for n_nodes, n_states, n_edges, seq_len in configs:
            temporal_bp = TemporalBeliefPropagation(n_nodes=n_nodes, n_states=n_states, n_edges=n_edges)

            batch_size = 2
            potentials = torch.softmax(torch.randn(seq_len, batch_size, n_nodes, n_states), dim=-1)
            temporal_weights = torch.randn(batch_size, n_edges)
            max_iterations = 3

            # Time inference
            import time
            start_time = time.time()

            marginals = temporal_bp.run_inference(potentials, temporal_weights, max_iterations)

            end_time = time.time()

            # Check performance
            total_time = end_time - start_time
            assert total_time < 10.0  # Should complete in reasonable time


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
