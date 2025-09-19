"""
Tests for Active Inference Engine

Tests the ActiveInferenceEngine and BayesianInference classes.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference import ActiveInferenceEngine, BayesianInference
from core import TritonFeatureManager, TritonFeatureConfig
from . import (
    setup_test_environment,
    create_test_tensor,
    assert_tensors_close,
    TEST_DEVICE,
)


class TestActiveInferenceEngine:
    """Test active inference engine functionality."""

    @pytest.fixture
    def engine(self):
        """Create active inference engine for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        return ActiveInferenceEngine(feature_manager)

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert isinstance(engine.feature_manager, TritonFeatureManager)
        assert len(engine.models) == 0
        assert len(engine.posteriors) == 0

    def test_register_model(self, engine):
        """Test model registration."""
        model_spec = {
            "name": "test_model",
            "variables": {
                "hidden": {"shape": (10,), "type": "continuous"},
                "observed": {"shape": (5,), "type": "continuous"},
            },
            "likelihood": "gaussian",
            "prior": "normal",
        }

        engine.register_model("test_model", model_spec)

        assert "test_model" in engine.models
        assert engine.models["test_model"] == model_spec

    def test_compute_free_energy_unregistered_model(self, engine):
        """Test free energy computation with unregistered model."""
        observations = create_test_tensor(10, 5)

        with pytest.raises(RuntimeError, match="Free energy computation failed.*Model test_model not registered"):
            engine.compute_free_energy(observations, "test_model")

    def test_compute_free_energy_registered_model(self, engine):
        """Test free energy computation with registered model."""
        # Register a simple model
        model_spec = {"variables": {"hidden": {"shape": (10,)}}}
        engine.register_model("test_model", model_spec)

        observations = create_test_tensor(5, 10)

        # Should compute free energy using real implementation
        free_energy = engine.compute_free_energy(observations, "test_model")

        assert free_energy.shape == (5,)
        assert free_energy.device.type == TEST_DEVICE.type

    def test_update_posterior_no_posterior(self, engine):
        """Test posterior update when no posterior exists."""
        model_spec = {"variables": {"hidden": {"shape": (10,)}}}
        engine.register_model("test_model", model_spec)

        observations = create_test_tensor(5, 10)

        # Should initialize posterior automatically
        engine.update_posterior(observations, "test_model")

        assert "test_model" in engine.posteriors
        posterior = engine.posteriors["test_model"]
        assert "hidden" in posterior
        assert posterior["hidden"].shape == (10,)
        assert posterior["hidden"].requires_grad

    def test_predict_no_posterior(self, engine):
        """Test prediction when no posterior exists."""
        model_spec = {"variables": {"hidden": {"shape": (10,)}}}
        engine.register_model("test_model", model_spec)

        with pytest.raises(ValueError, match="No posterior available"):
            engine.predict("test_model")

    def test_predict_with_posterior(self, engine):
        """Test prediction with learned posterior."""
        # Set up model and posterior
        model_spec = {"variables": {"hidden": {"shape": (10,)}}}
        engine.register_model("test_model", model_spec)

        # Initialize posterior manually
        engine.posteriors["test_model"] = {
            "hidden": create_test_tensor(10, requires_grad=True)
        }

        predictions = engine.predict("test_model", num_samples=3)

        assert predictions.shape == (3, 10)  # num_samples x feature_dim
        assert predictions.device.type == TEST_DEVICE.type


class TestBayesianInference:
    """Test Bayesian inference methods."""

    @pytest.fixture
    def bayesian_engine(self):
        """Create Bayesian inference engine for testing."""
        setup_test_environment()
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        ai_engine = ActiveInferenceEngine(feature_manager)
        return BayesianInference(ai_engine)

    def test_compute_posterior(self, bayesian_engine):
        """Test posterior computation using Bayes rule."""
        batch_size, num_states = 5, 3

        prior = torch.softmax(create_test_tensor(batch_size, num_states), dim=1)
        likelihood = torch.softmax(create_test_tensor(batch_size, num_states), dim=1)

        posterior = bayesian_engine.compute_posterior(prior, likelihood)

        # Check normalization
        assert torch.allclose(
            posterior.sum(dim=1), torch.ones(batch_size, device=TEST_DEVICE)
        )

        # Check shape
        assert posterior.shape == (batch_size, num_states)

        # Posterior should be proportional to prior * likelihood
        expected = prior * likelihood
        expected = expected / expected.sum(dim=1, keepdim=True)

        assert_tensors_close(posterior, expected)

    def test_compute_evidence(self, bayesian_engine):
        """Test marginal likelihood (evidence) computation."""
        batch_size, num_states = 5, 3

        prior = torch.softmax(create_test_tensor(batch_size, num_states), dim=1)
        likelihood = torch.softmax(create_test_tensor(batch_size, num_states), dim=1)

        evidence = bayesian_engine.compute_evidence(prior, likelihood)

        assert evidence.shape == (batch_size,)
        assert torch.all(evidence > 0)  # Evidence should be positive

        # Evidence should equal sum of joint probabilities
        expected_evidence = (prior * likelihood).sum(dim=1)
        assert_tensors_close(evidence, expected_evidence)


class TestIntegration:
    """Integration tests for inference components."""

    def test_active_inference_bayesian_integration(self):
        """Test integration between active inference and Bayesian methods."""
        setup_test_environment()

        # Set up engine
        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        ai_engine = ActiveInferenceEngine(feature_manager)
        bayesian_engine = BayesianInference(ai_engine)

        # Register model
        model_spec = {"variables": {"hidden": {"shape": (5,)}}}
        ai_engine.register_model("integration_model", model_spec)

        # Create test data
        observations = create_test_tensor(10, 5)

        # Test free energy computation
        free_energy = ai_engine.compute_free_energy(observations, "integration_model")
        assert free_energy.shape == (10,)

        # Test posterior update
        ai_engine.update_posterior(observations, "integration_model")
        assert "integration_model" in ai_engine.posteriors

        # Test prediction
        predictions = ai_engine.predict("integration_model", num_samples=5)
        assert predictions.shape == (5, 5)

    def test_bayesian_inference_properties(self):
        """Test mathematical properties of Bayesian inference."""
        setup_test_environment()

        config = TritonFeatureConfig(device=TEST_DEVICE.type)
        feature_manager = TritonFeatureManager(config)
        ai_engine = ActiveInferenceEngine(feature_manager)
        bayesian_engine = BayesianInference(ai_engine)

        batch_size, num_states = 3, 4

        # Create uniform prior
        prior = torch.ones(batch_size, num_states, device=TEST_DEVICE) / num_states

        # Create likelihood that favors certain states
        likelihood = torch.zeros(batch_size, num_states, device=TEST_DEVICE)
        likelihood[:, 0] = 0.8  # Strong preference for state 0
        likelihood[:, 1:] = 0.2 / (num_states - 1)  # Uniform for others

        posterior = bayesian_engine.compute_posterior(prior, likelihood)

        # With uniform prior, posterior should equal normalized likelihood
        expected_posterior = likelihood / likelihood.sum(dim=1, keepdim=True)
        assert_tensors_close(posterior, expected_posterior)

        # Most probable state should be state 0
        most_probable = posterior.argmax(dim=1)
        assert torch.all(most_probable == 0)
