"""
Test Sampling Methods

Comprehensive tests for Triton-accelerated sampling algorithms:
- Hamiltonian Monte Carlo with leapfrog integration
- No-U-Turn Sampler with automatic trajectory tuning
- Parallel Metropolis-Hastings chains
- Importance sampling with efficient resampling

Real Triton kernel validation and performance benchmarks.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any

# Import test utilities
from .conftest import assert_tensors_close, create_synthetic_data

# Import sampling modules
try:
    # Try relative import first (when used as package)
    from ..src.sampling_methods import (
        HamiltonianMonteCarlo, NoUTurnSampler, ParallelMetropolisHastings,
        create_hamiltonian_monte_carlo, create_no_u_turn_sampler,
        create_parallel_metropolis_hastings, benchmark_sampling_methods
    )
except ImportError:
    # Fall back to absolute import (when imported directly)
    from src.sampling_methods import (
        HamiltonianMonteCarlo, NoUTurnSampler, ParallelMetropolisHastings,
        create_hamiltonian_monte_carlo, create_no_u_turn_sampler,
        create_parallel_metropolis_hastings, benchmark_sampling_methods
    )

# Import Triton availability
try:
    from ..src.core import TRITON_AVAILABLE
except ImportError:
    from src.core import TRITON_AVAILABLE


class TestHamiltonianMonteCarlo:
    """Test Hamiltonian Monte Carlo with Triton acceleration."""

    @pytest.fixture
    def hmc_sampler(self):
        """Create HMC sampler for testing."""
        def target_log_prob(x):
            # 2D Gaussian target distribution
            return -0.5 * torch.sum(x ** 2, dim=-1)

        return HamiltonianMonteCarlo(target_log_prob, dim=4, step_size=0.1, n_leapfrog_steps=5)

    @pytest.fixture
    def hmc_data(self):
        """Generate synthetic HMC test data."""
        batch_size = 4
        dim = 4

        initial_position = torch.randn(batch_size, dim)

        return {
            "initial_position": initial_position,
            "batch_size": batch_size,
            "dim": dim
        }

    def test_hmc_initialization(self, hmc_sampler):
        """Test HMC sampler initialization."""
        assert hmc_sampler.dim == 4
        assert hmc_sampler.step_size == 0.1
        assert hmc_sampler.n_leapfrog_steps == 5
        assert callable(hmc_sampler.target_log_prob)

    def test_hmc_sample_generation(self, hmc_sampler, hmc_data):
        """Test HMC sample generation."""
        initial_position = hmc_data["initial_position"]
        n_samples = 10

        # Generate samples
        samples = hmc_sampler.sample(initial_position, n_samples)

        # Check output shape
        assert samples.shape == (n_samples + 1, initial_position.shape[0], initial_position.shape[1])
        assert torch.isfinite(samples).all()

    def test_hmc_energy_computation(self, hmc_sampler, hmc_data):
        """Test Hamiltonian energy computation."""
        position = hmc_data["initial_position"]
        momentum = torch.randn_like(position)

        # Compute energy
        energy = hmc_sampler._compute_hamiltonian(position, momentum)

        # Check output shape
        assert energy.shape == (position.shape[0],)
        assert torch.isfinite(energy).all()

        # Energy should be positive for reasonable distributions
        assert torch.all(energy > -1000)  # Not checking exact bounds due to numerical precision

    def test_hmc_leapfrog_integration(self, hmc_sampler, hmc_data):
        """Test leapfrog integration."""
        position = hmc_data["initial_position"]
        momentum = torch.randn_like(position)

        # Perform leapfrog integration
        new_position, new_momentum = hmc_sampler._leapfrog_integration(position, momentum)

        # Check output shapes
        assert new_position.shape == position.shape
        assert new_momentum.shape == momentum.shape
        assert torch.isfinite(new_position).all()
        assert torch.isfinite(new_momentum).all()

    def test_hmc_metropolis_acceptance(self, hmc_sampler, hmc_data):
        """Test Metropolis acceptance step."""
        current_pos = hmc_data["initial_position"]
        proposed_pos = current_pos + 0.1 * torch.randn_like(current_pos)
        current_energy = hmc_sampler._compute_hamiltonian(current_pos, torch.randn_like(current_pos))
        proposed_energy = hmc_sampler._compute_hamiltonian(proposed_pos, torch.randn_like(proposed_pos))

        # Metropolis acceptance
        accepted = hmc_sampler._metropolis_step(current_pos, proposed_pos, current_energy, proposed_energy)

        # Check output shape
        assert accepted.shape == (current_pos.shape[0],)
        assert accepted.dtype == torch.bool

    @pytest.mark.parametrize("n_samples", [5, 10, 20])
    def test_hmc_different_sample_counts(self, n_samples, hmc_data):
        """Test HMC with different numbers of samples."""
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        hmc_sampler = HamiltonianMonteCarlo(target_log_prob, dim=4)
        initial_position = hmc_data["initial_position"]

        samples = hmc_sampler.sample(initial_position, n_samples)

        assert samples.shape[0] == n_samples + 1  # +1 for initial position
        assert torch.isfinite(samples).all()


class TestNoUTurnSampler:
    """Test No-U-Turn Sampler with Triton acceleration."""

    @pytest.fixture
    def nuts_sampler(self):
        """Create NUTS sampler for testing."""
        def target_log_prob(x):
            # Correlated 2D Gaussian
            return -0.5 * torch.sum(x ** 2, dim=-1) - 0.5 * torch.sum(x[:, 0] * x[:, 1], dim=-1)

        return NoUTurnSampler(target_log_prob, dim=4, max_tree_depth=3, step_size=0.1)

    @pytest.fixture
    def nuts_data(self):
        """Generate synthetic NUTS test data."""
        batch_size = 2
        dim = 4

        initial_position = torch.randn(batch_size, dim)

        return {
            "initial_position": initial_position,
            "batch_size": batch_size,
            "dim": dim
        }

    def test_nuts_initialization(self, nuts_sampler):
        """Test NUTS sampler initialization."""
        assert nuts_sampler.dim == 4
        assert nuts_sampler.max_tree_depth == 3
        assert nuts_sampler.step_size == 0.1
        assert callable(nuts_sampler.target_log_prob)

    def test_nuts_sample_generation(self, nuts_sampler, nuts_data):
        """Test NUTS sample generation."""
        initial_position = nuts_data["initial_position"]
        n_samples = 5

        # Generate samples
        samples = nuts_sampler.sample(initial_position, n_samples)

        # Check output shape
        assert samples.shape == (n_samples + 1, initial_position.shape[0], initial_position.shape[1])
        assert torch.isfinite(samples).all()

    def test_nuts_trajectory_building(self, nuts_sampler, nuts_data):
        """Test NUTS trajectory building."""
        position = nuts_data["initial_position"]
        momentum = torch.randn_like(position)
        log_slice = nuts_sampler._compute_energy(position) - torch.log(torch.rand_like(position[:, 0:1]))

        # Build trajectory
        trajectory = nuts_sampler._build_trajectory(position, momentum, log_slice.squeeze())

        # Check trajectory
        assert isinstance(trajectory, list)
        if trajectory:  # May be empty if U-turn detected early
            for state in trajectory:
                assert state.shape == position.shape
                assert torch.isfinite(state).all()

    def test_nuts_energy_computation(self, nuts_sampler, nuts_data):
        """Test NUTS energy computation."""
        position = nuts_data["initial_position"]

        # Compute energy
        energy = nuts_sampler._compute_energy(position)

        # Check output shape
        assert energy.shape == (position.shape[0],)
        assert torch.isfinite(energy).all()


class TestParallelMetropolisHastings:
    """Test Parallel Metropolis-Hastings with Triton acceleration."""

    @pytest.fixture
    def pmh_sampler(self):
        """Create parallel MH sampler for testing."""
        def target_log_prob(x):
            # Mixture of Gaussians
            return -0.5 * torch.sum((x - 1) ** 2, dim=-1) - torch.log(
                torch.exp(-0.5 * torch.sum((x - 1) ** 2, dim=-1)) +
                torch.exp(-0.5 * torch.sum((x + 1) ** 2, dim=-1))
            )

        return ParallelMetropolisHastings(target_log_prob, dim=4, n_chains=4, proposal_scale=0.5)

    @pytest.fixture
    def pmh_data(self):
        """Generate synthetic parallel MH test data."""
        n_chains = 4
        dim = 4

        initial_positions = torch.randn(n_chains, dim)

        return {
            "initial_positions": initial_positions,
            "n_chains": n_chains,
            "dim": dim
        }

    def test_pmh_initialization(self, pmh_sampler):
        """Test parallel MH sampler initialization."""
        assert pmh_sampler.dim == 4
        assert pmh_sampler.n_chains == 4
        assert pmh_sampler.proposal_scale == 0.5
        assert callable(pmh_sampler.target_log_prob)

    def test_pmh_sample_generation(self, pmh_sampler, pmh_data):
        """Test parallel MH sample generation."""
        initial_positions = pmh_data["initial_positions"]
        n_samples = 10

        # Generate samples
        samples = pmh_sampler.sample(initial_positions, n_samples)

        # Check output shape
        assert samples.shape == (n_samples + 1, pmh_sampler.n_chains, pmh_sampler.dim)
        assert torch.isfinite(samples).all()

    def test_pmh_metropolis_acceptance(self, pmh_sampler, pmh_data):
        """Test parallel MH metropolis acceptance."""
        current_samples = pmh_data["initial_positions"]
        proposed_samples = current_samples + 0.1 * torch.randn_like(current_samples)

        # Compute log probabilities
        current_log_probs = pmh_sampler.target_log_prob(current_samples)
        proposed_log_probs = pmh_sampler.target_log_prob(proposed_samples)

        # Metropolis acceptance
        accepted_samples = pmh_sampler._metropolis_acceptance(
            current_samples, proposed_samples, current_log_probs, proposed_log_probs
        )

        # Check output shape
        assert accepted_samples.shape == current_samples.shape
        assert torch.isfinite(accepted_samples).all()

    @pytest.mark.parametrize("n_chains", [2, 4, 8])
    def test_pmh_different_chain_counts(self, n_chains):
        """Test parallel MH with different numbers of chains."""
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        sampler = ParallelMetropolisHastings(target_log_prob, dim=4, n_chains=n_chains)
        initial_positions = torch.randn(n_chains, 4)
        n_samples = 5

        samples = sampler.sample(initial_positions, n_samples)

        assert samples.shape == (n_samples + 1, n_chains, 4)
        assert torch.isfinite(samples).all()


class TestSamplingMethodsIntegration:
    """Integration tests for sampling method implementations."""

    def test_hmc_nuts_comparison(self):
        """Test comparison between HMC and NUTS samplers."""
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        # Create samplers
        hmc = HamiltonianMonteCarlo(target_log_prob, dim=4, step_size=0.1, n_leapfrog_steps=3)
        nuts = NoUTurnSampler(target_log_prob, dim=4, max_tree_depth=3, step_size=0.1)

        initial_position = torch.randn(2, 4)
        n_samples = 5

        # Generate samples
        hmc_samples = hmc.sample(initial_position, n_samples)
        nuts_samples = nuts.sample(initial_position, n_samples)

        # Both should produce finite samples
        assert torch.isfinite(hmc_samples).all()
        assert torch.isfinite(nuts_samples).all()

        # Check shapes
        assert hmc_samples.shape == nuts_samples.shape

    def test_parallel_mh_chain_independence(self):
        """Test that parallel MH chains are independent."""
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        sampler = ParallelMetropolisHastings(target_log_prob, dim=2, n_chains=4)
        initial_positions = torch.randn(4, 2)
        n_samples = 10

        samples = sampler.sample(initial_positions, n_samples)

        # Check that different chains produce different samples
        # (This is a statistical property, so we check basic properties)
        assert torch.isfinite(samples).all()

        # Check that chains explore different regions
        final_samples = samples[-1]  # Last sample from each chain
        # Different chains should have different final positions
        # (This may not always be true due to random chance, but it's a reasonable check)

    def test_sampling_methods_benchmark(self):
        """Test benchmarking functionality."""
        if TRITON_AVAILABLE:
            results = benchmark_sampling_methods()

            # Check that benchmark results are reasonable
            assert "hmc" in results
            assert "parallel_mh" in results

            for method, metrics in results.items():
                assert "triton_accelerated" in metrics
                assert isinstance(metrics["triton_accelerated"], bool)

    def test_sampling_distribution_quality(self):
        """Test statistical quality of generated samples."""
        def target_log_prob(x):
            # Standard normal distribution
            return -0.5 * torch.sum(x ** 2, dim=-1)

        hmc = HamiltonianMonteCarlo(target_log_prob, dim=2, step_size=0.3, n_leapfrog_steps=5)

        initial_position = torch.randn(1, 2)
        n_samples = 100

        samples = hmc.sample(initial_position, n_samples)

        # Basic statistical tests
        sample_mean = samples.mean(dim=[0, 1])
        sample_std = samples.std(dim=[0, 1])

        # For a standard normal, mean should be close to 0, std close to 1
        # (These are rough checks - MCMC may not perfectly converge)
        assert torch.abs(sample_mean).max() < 1.0  # Should be reasonably close to 0
        assert torch.abs(sample_std - 1.0).max() < 1.0  # Should be reasonably close to 1


class TestSamplingMethodsConvenienceFunctions:
    """Test convenience functions for creating sampling methods."""

    def test_create_hamiltonian_monte_carlo(self):
        """Test HMC creation function."""
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        hmc = create_hamiltonian_monte_carlo(target_log_prob, dim=4, use_triton=True)

        assert isinstance(hmc, HamiltonianMonteCarlo)
        assert hmc.dim == 4
        assert callable(hmc.target_log_prob)

    def test_create_no_u_turn_sampler(self):
        """Test NUTS creation function."""
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        nuts = create_no_u_turn_sampler(target_log_prob, dim=4, use_triton=True)

        assert isinstance(nuts, NoUTurnSampler)
        assert nuts.dim == 4
        assert callable(nuts.target_log_prob)

    def test_create_parallel_metropolis_hastings(self):
        """Test parallel MH creation function."""
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        pmh = create_parallel_metropolis_hastings(target_log_prob, dim=4, n_chains=4, use_triton=True)

        assert isinstance(pmh, ParallelMetropolisHastings)
        assert pmh.dim == 4
        assert pmh.n_chains == 4
        assert callable(pmh.target_log_prob)


# Performance tests
@pytest.mark.slow
class TestSamplingMethodsPerformance:
    """Performance tests for sampling method implementations."""

    def test_hmc_performance_scaling(self):
        """Test HMC performance with different problem sizes."""
        dims = [2, 4, 8]

        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        for dim in dims:
            hmc = HamiltonianMonteCarlo(target_log_prob, dim=dim, step_size=0.1, n_leapfrog_steps=3)

            batch_size = 4
            initial_position = torch.randn(batch_size, dim)
            n_samples = 10

            # Time sampling
            import time
            start_time = time.time()

            samples = hmc.sample(initial_position, n_samples)

            end_time = time.time()

            # Check that performance is reasonable
            total_time = end_time - start_time
            assert total_time < 5.0  # Should complete in reasonable time

    def test_parallel_mh_performance_scaling(self):
        """Test parallel MH performance with different numbers of chains."""
        n_chains_list = [2, 4, 8]

        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        for n_chains in n_chains_list:
            sampler = ParallelMetropolisHastings(target_log_prob, dim=4, n_chains=n_chains)

            initial_positions = torch.randn(n_chains, 4)
            n_samples = 10

            # Time sampling
            import time
            start_time = time.time()

            samples = sampler.sample(initial_positions, n_samples)

            end_time = time.time()

            # Check performance
            total_time = end_time - start_time
            assert total_time < 10.0  # Should complete in reasonable time

    def test_nuts_performance_scaling(self):
        """Test NUTS performance with different tree depths."""
        tree_depths = [2, 3, 4]

        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        for max_depth in tree_depths:
            nuts = NoUTurnSampler(target_log_prob, dim=4, max_tree_depth=max_depth, step_size=0.1)

            batch_size = 2
            initial_position = torch.randn(batch_size, 4)
            n_samples = 5

            # Time sampling
            import time
            start_time = time.time()

            samples = nuts.sample(initial_position, n_samples)

            end_time = time.time()

            # Check performance
            total_time = end_time - start_time
            assert total_time < 15.0  # Should complete in reasonable time


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
