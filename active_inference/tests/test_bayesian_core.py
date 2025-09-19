"""
Comprehensive tests for Bayesian Inference Core

Tests all core functionalities: Bayesian updating, information calculations,
free energy computation, validation, logging, and visualization.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import json
import tempfile
import os
from datetime import datetime

from src.bayesian_core import (
    BayesianInferenceCore,
    BayesianUpdateResult,
    InferenceHistory,
    create_bayesian_core,
    bayesian_update_with_reporting
)


class TestBayesianUpdateResult:
    """Test BayesianUpdateResult data structure."""

    def test_result_creation(self):
        """Test creating a BayesianUpdateResult."""
        batch_size, n_states = 2, 4

        prior = torch.randn(batch_size, n_states)
        likelihood = torch.randn(batch_size, n_states)
        posterior = torch.randn(batch_size, n_states)
        evidence = torch.randn(batch_size)
        kl_div = torch.randn(batch_size)
        prior_ent = torch.randn(batch_size)
        post_ent = torch.randn(batch_size)
        mutual_info = torch.randn(batch_size)
        free_energy = torch.randn(batch_size)

        result = BayesianUpdateResult(
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
            kl_divergence=kl_div,
            prior_entropy=prior_ent,
            posterior_entropy=post_ent,
            mutual_information=mutual_info,
            free_energy=free_energy
        )

        assert result.prior.shape == (batch_size, n_states)
        assert result.update_method == "bayesian"
        assert result.converged == True
        assert result.iterations == 1
        assert isinstance(result.timestamp, datetime)

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        batch_size, n_states = 2, 3

        prior = torch.ones(batch_size, n_states) / n_states
        likelihood = torch.ones(batch_size, n_states) / n_states
        posterior = torch.ones(batch_size, n_states) / n_states
        evidence = torch.ones(batch_size)
        kl_div = torch.zeros(batch_size)
        prior_ent = torch.ones(batch_size)
        post_ent = torch.ones(batch_size)
        mutual_info = torch.zeros(batch_size)
        free_energy = torch.zeros(batch_size)

        result = BayesianUpdateResult(
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
            kl_divergence=kl_div,
            prior_entropy=prior_ent,
            posterior_entropy=post_ent,
            mutual_information=mutual_info,
            free_energy=free_energy
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'prior' in result_dict
        assert 'posterior' in result_dict
        assert 'free_energy' in result_dict
        assert 'timestamp' in result_dict

        # Check numpy array conversion
        assert isinstance(result_dict['prior'], np.ndarray)
        assert result_dict['prior'].shape == (batch_size, n_states)


class TestInferenceHistory:
    """Test InferenceHistory data structure."""

    def test_history_creation(self):
        """Test creating and using inference history."""
        history = InferenceHistory()

        assert len(history.updates) == 0
        assert len(history.free_energy_history) == 0

    def test_add_update(self):
        """Test adding updates to history."""
        history = InferenceHistory()

        # Create mock result
        batch_size, n_states = 2, 3
        result = BayesianUpdateResult(
            prior=torch.ones(batch_size, n_states) / n_states,
            likelihood=torch.ones(batch_size, n_states) / n_states,
            posterior=torch.ones(batch_size, n_states) / n_states,
            evidence=torch.ones(batch_size),
            kl_divergence=torch.zeros(batch_size),
            prior_entropy=torch.ones(batch_size),
            posterior_entropy=torch.ones(batch_size),
            mutual_information=torch.zeros(batch_size),
            free_energy=torch.zeros(batch_size)
        )

        history.add_update(result)

        assert len(history.updates) == 1
        assert len(history.free_energy_history) == 1
        assert len(history.entropy_history) == 1

    def test_summary_stats(self):
        """Test computing summary statistics."""
        history = InferenceHistory()

        # Add multiple results
        for i in range(3):
            result = BayesianUpdateResult(
                prior=torch.ones(2, 3) / 3,
                likelihood=torch.ones(2, 3) / 3,
                posterior=torch.ones(2, 3) / 3,
                evidence=torch.ones(2),
                kl_divergence=torch.full((2,), i * 0.1),
                prior_entropy=torch.ones(2),
                posterior_entropy=torch.ones(2),
                mutual_information=torch.zeros(2),
                free_energy=torch.full((2,), i * 0.5),
                computation_time=0.1 * (i + 1)
            )
            history.add_update(result)

        stats = history.get_summary_stats()

        assert stats['total_updates'] == 3
        assert 'avg_free_energy' in stats
        assert 'std_free_energy' in stats
        assert 'total_computation_time' in stats
        assert stats['convergence_rate'] == 1.0  # All converged

    def test_empty_history_stats(self):
        """Test summary stats with empty history."""
        history = InferenceHistory()
        stats = history.get_summary_stats()

        assert stats == {}


class TestBayesianInferenceCore:
    """Comprehensive tests for BayesianInferenceCore."""

    @pytest.fixture
    def core(self):
        """Create Bayesian inference core for testing."""
        return BayesianInferenceCore(
            enable_logging=False,  # Disable logging for tests
            enable_validation=True,
            enable_visualization=False  # Disable visualization for tests
        )

    def test_initialization(self, core):
        """Test core initialization."""
        assert isinstance(core.history, InferenceHistory)
        assert core.enable_validation == True
        assert core.enable_logging == False
        assert core.enable_visualization == False

        # Check validation stats initialized
        assert core.validation_stats['total_updates'] == 0
        assert core.validation_stats['successful_updates'] == 0

    def test_validate_inputs_valid(self, core):
        """Test input validation with valid inputs."""
        batch_size, n_states = 2, 4

        prior = torch.ones(batch_size, n_states) / n_states
        likelihood = torch.ones(batch_size, n_states) / n_states

        assert core.validate_inputs(prior, likelihood) == True

    def test_validate_inputs_invalid_shape(self, core):
        """Test input validation with invalid shapes."""
        prior = torch.ones(2, 3)  # Wrong shape
        likelihood = torch.ones(2, 4)

        assert core.validate_inputs(prior, likelihood) == False

    def test_validate_inputs_nan_values(self, core):
        """Test input validation with NaN values."""
        prior = torch.tensor([[0.5, float('nan')], [0.3, 0.7]])
        likelihood = torch.ones(2, 2) / 2

        assert core.validate_inputs(prior, likelihood) == False

    def test_validate_inputs_negative_values(self, core):
        """Test input validation with negative values."""
        prior = torch.tensor([[0.5, -0.1], [0.6, 0.5]])  # Negative probability
        likelihood = torch.ones(2, 2) / 2

        assert core.validate_inputs(prior, likelihood) == False

    def test_compute_evidence(self, core):
        """Test evidence computation."""
        batch_size, n_states = 2, 3

        prior = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        likelihood = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])

        evidence = core.compute_evidence(prior, likelihood)

        assert evidence.shape == (batch_size,)
        # Evidence should be positive
        assert torch.all(evidence > 0)

        # Manual calculation check
        expected_0 = 0.2 * 0.8 + 0.3 * 0.1 + 0.5 * 0.1  # 0.21
        expected_1 = 0.1 * 0.2 + 0.4 * 0.7 + 0.5 * 0.1  # 0.33

        assert torch.allclose(evidence, torch.tensor([expected_0, expected_1]), atol=1e-6)

    def test_compute_entropy(self, core):
        """Test entropy computation."""
        # Uniform distribution should have high entropy
        uniform = torch.ones(2, 4) / 4
        entropy_uniform = core.compute_entropy(uniform)

        # Deterministic distribution should have low entropy
        deterministic = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        entropy_det = core.compute_entropy(deterministic)

        assert entropy_uniform.shape == (2,)
        assert entropy_det.shape == (2,)
        assert torch.all(entropy_uniform > entropy_det)  # Uniform has higher entropy

    def test_compute_kl_divergence(self, core):
        """Test KL divergence computation."""
        # Same distributions should have zero KL divergence
        p = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        q = p.clone()

        kl = core.compute_kl_divergence(p, q)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

        # Different distributions should have positive KL
        p_diff = torch.tensor([[0.5, 0.3, 0.2]])
        q_diff = torch.tensor([[0.2, 0.3, 0.5]])
        kl_diff = core.compute_kl_divergence(p_diff, q_diff)

        assert kl_diff > 0

    def test_compute_mutual_information(self, core):
        """Test mutual information computation."""
        batch_size, n_states = 2, 3

        # Independent distributions should have zero mutual information
        prior = torch.ones(batch_size, n_states) / n_states
        posterior = prior.clone()
        likelihood = torch.ones(batch_size, n_states) / n_states

        mi = core.compute_mutual_information(prior, posterior, likelihood)
        assert mi.shape == (batch_size,)

    def test_bayesian_update_basic(self, core):
        """Test basic Bayesian update functionality."""
        batch_size, n_states = 2, 4

        # Create valid prior and likelihood
        prior = torch.ones(batch_size, n_states) / n_states
        likelihood = torch.softmax(torch.randn(batch_size, n_states), dim=1)

        result = core.bayesian_update(prior, likelihood)

        # Check result structure
        assert isinstance(result, BayesianUpdateResult)
        assert result.prior.shape == (batch_size, n_states)
        assert result.posterior.shape == (batch_size, n_states)
        assert result.free_energy.shape == (batch_size,)
        assert result.computation_time > 0

        # Check history updated
        assert len(core.history.updates) == 1
        assert core.validation_stats['total_updates'] == 1
        assert core.validation_stats['successful_updates'] == 1

    def test_bayesian_update_with_observations(self, core):
        """Test Bayesian update with observation data."""
        batch_size, n_states = 2, 3

        prior = torch.ones(batch_size, n_states) / n_states
        likelihood = torch.softmax(torch.randn(batch_size, n_states), dim=1)
        observations = torch.randn(batch_size, 2)

        result = core.bayesian_update(prior, likelihood, observations)

        assert isinstance(result, BayesianUpdateResult)
        assert result.posterior.shape == (batch_size, n_states)

    def test_get_prior_samples_uniform(self, core):
        """Test uniform prior generation."""
        n_samples, n_states = 5, 4

        prior = core.get_prior_samples(n_samples, n_states, prior_type="uniform")

        assert prior.shape == (n_samples, n_states)
        # Check normalization
        row_sums = prior.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n_samples), atol=1e-5)

    def test_get_prior_samples_dirichlet(self, core):
        """Test Dirichlet prior generation."""
        n_samples, n_states = 3, 4

        prior = core.get_prior_samples(n_samples, n_states, prior_type="dirichlet")

        assert prior.shape == (n_samples, n_states)
        # Check normalization
        row_sums = prior.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n_samples), atol=1e-5)
        # All values should be positive
        assert torch.all(prior > 0)

    def test_get_prior_samples_gaussian(self, core):
        """Test Gaussian prior generation."""
        n_samples, n_states = 4, 3

        prior = core.get_prior_samples(n_samples, n_states, prior_type="gaussian")

        assert prior.shape == (n_samples, n_states)
        # Check normalization
        row_sums = prior.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n_samples), atol=1e-5)

    def test_get_prior_samples_invalid_type(self, core):
        """Test invalid prior type."""
        with pytest.raises(ValueError, match="Unknown prior type"):
            core.get_prior_samples(5, 3, prior_type="invalid")

    def test_get_observation_likelihood_categorical(self, core):
        """Test categorical observation likelihood."""
        batch_size, obs_dim, n_states = 3, 2, 4

        observations = torch.randn(batch_size, obs_dim)
        likelihood = core.get_observation_likelihood(
            observations, n_states, likelihood_type="categorical"
        )

        assert likelihood.shape == (batch_size, n_states)
        # Check normalization
        row_sums = likelihood.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5)

    def test_get_observation_likelihood_gaussian(self, core):
        """Test Gaussian observation likelihood."""
        batch_size, obs_dim, n_states = 2, 3, 3

        observations = torch.randn(batch_size, obs_dim)
        likelihood = core.get_observation_likelihood(
            observations, n_states, likelihood_type="gaussian"
        )

        assert likelihood.shape == (batch_size, n_states)
        # Check normalization
        row_sums = likelihood.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5)

    def test_get_observation_likelihood_invalid_type(self, core):
        """Test invalid likelihood type."""
        observations = torch.randn(2, 3)
        with pytest.raises(ValueError, match="Unknown likelihood type"):
            core.get_observation_likelihood(observations, 4, likelihood_type="invalid")

    @patch('matplotlib.pyplot.show')
    def test_visualize_beliefs_disabled(self, mock_show, core):
        """Test visualization when disabled."""
        # Visualization is disabled in core fixture
        result = BayesianUpdateResult(
            prior=torch.ones(1, 3) / 3,
            likelihood=torch.ones(1, 3) / 3,
            posterior=torch.ones(1, 3) / 3,
            evidence=torch.ones(1),
            kl_divergence=torch.zeros(1),
            prior_entropy=torch.ones(1),
            posterior_entropy=torch.ones(1),
            mutual_information=torch.zeros(1),
            free_energy=torch.zeros(1)
        )

        # Should not crash when visualization is disabled
        core.visualize_beliefs(result)

        # plt.show should not be called
        mock_show.assert_not_called()

    def test_generate_report(self, core):
        """Test report generation."""
        # Add some fake data
        core.validation_stats['total_updates'] = 5
        core.validation_stats['successful_updates'] = 4
        core.validation_stats['computation_times'] = [0.1, 0.15, 0.08, 0.12]

        report = core.generate_report(include_history=False)

        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'validation_stats' in report
        assert 'history_summary' in report
        assert 'feature_manager_info' in report
        assert 'recommendations' in report

    def test_generate_report_with_save(self, core):
        """Test report generation with file saving."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            report = core.generate_report(save_path=temp_path, include_history=False)

            # Check file was created
            assert os.path.exists(temp_path)

            # Check file contents
            with open(temp_path, 'r') as f:
                saved_report = json.load(f)

            assert 'timestamp' in saved_report
            assert 'validation_stats' in saved_report

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_reset_history(self, core):
        """Test history reset functionality."""
        # Add some fake data
        core.validation_stats['total_updates'] = 10
        core.validation_stats['successful_updates'] = 8

        result = BayesianUpdateResult(
            prior=torch.ones(1, 2) / 2,
            likelihood=torch.ones(1, 2) / 2,
            posterior=torch.ones(1, 2) / 2,
            evidence=torch.ones(1),
            kl_divergence=torch.zeros(1),
            prior_entropy=torch.ones(1),
            posterior_entropy=torch.ones(1),
            mutual_information=torch.zeros(1),
            free_energy=torch.zeros(1)
        )
        core.history.add_update(result)

        # Verify data exists
        assert core.validation_stats['total_updates'] == 10
        assert len(core.history.updates) == 1

        # Reset
        core.reset_history()

        # Verify reset
        assert core.validation_stats['total_updates'] == 0
        assert core.validation_stats['successful_updates'] == 0
        assert len(core.history.updates) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_bayesian_core(self):
        """Test creating Bayesian core with convenience function."""
        core = create_bayesian_core(
            device="cpu",
            enable_logging=False,
            enable_validation=True,
            enable_visualization=False
        )

        assert isinstance(core, BayesianInferenceCore)
        assert core.enable_validation == True
        assert core.enable_logging == False

    def test_bayesian_update_with_reporting(self):
        """Test Bayesian update with automatic reporting."""
        batch_size, n_states = 2, 3

        prior = torch.ones(batch_size, n_states) / n_states
        likelihood = torch.softmax(torch.randn(batch_size, n_states), dim=1)

        result, report = bayesian_update_with_reporting(
            prior, likelihood,
            enable_logging=False,
            enable_validation=True,
            enable_visualization=False
        )

        assert isinstance(result, BayesianUpdateResult)
        assert isinstance(report, dict)
        assert 'validation_stats' in report
        assert 'history_summary' in report


class TestBayesianCoreIntegration:
    """Integration tests for Bayesian core functionality."""

    def test_complete_inference_workflow(self):
        """Test complete Bayesian inference workflow."""
        core = create_bayesian_core(enable_logging=False, enable_validation=True, enable_visualization=False)

        # Step 1: Generate priors
        priors = core.get_prior_samples(3, 4, prior_type="dirichlet")

        # Step 2: Generate observations and likelihoods
        observations = torch.randn(3, 2)
        likelihoods = core.get_observation_likelihood(observations, 4)

        # Step 3: Perform multiple Bayesian updates
        for i in range(3):
            result = core.bayesian_update(priors, likelihoods, observations)
            assert isinstance(result, BayesianUpdateResult)
            assert result.free_energy.shape == (3,)

        # Step 4: Check history
        assert len(core.history.updates) == 3
        assert len(core.history.free_energy_history) == 3

        # Step 5: Generate report
        report = core.generate_report(include_history=True)
        assert report['validation_stats']['total_updates'] == 3
        assert report['validation_stats']['successful_updates'] == 3

    def test_information_theory_consistency(self):
        """Test consistency of information theoretic calculations."""
        core = create_bayesian_core(enable_logging=False, enable_validation=True, enable_visualization=False)

        # Create deterministic posterior (should have zero entropy)
        batch_size, n_states = 2, 4
        prior = torch.softmax(torch.randn(batch_size, n_states), dim=1)
        posterior = torch.zeros(batch_size, n_states)
        posterior[:, 0] = 1.0  # Deterministic posterior

        # Create corresponding likelihood
        likelihood = torch.zeros(batch_size, n_states)
        likelihood[:, 0] = 1.0  # Likelihood favors first state

        result = core.bayesian_update(prior, likelihood)

        # Posterior entropy should be very low (close to 0)
        assert torch.all(result.posterior_entropy < 0.1)

        # KL divergence should be positive
        assert torch.all(result.kl_divergence >= 0)

        # Mutual information should be related to entropy reduction
        assert result.mutual_information.shape == (batch_size,)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
