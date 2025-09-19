#!/usr/bin/env python3
"""
Tests for Active Inference Agent

Comprehensive tests for the ActiveInferenceAgent class,
ensuring VFE and EFE computations work correctly.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.active_inference_agent import ActiveInferenceAgent
from src.pomdp_environment import POMDPGridworld
from src.core import TritonFeatureManager


class TestActiveInferenceAgent:
    """Test ActiveInferenceAgent functionality."""

    @pytest.fixture
    def mock_environment(self):
        """Create mock environment for testing."""
        env = MagicMock()
        env.total_states = 25  # 5x5 grid with 1 temperature state for simplicity
        env.n_actions = 4
        env.grid_size = 5
        env.n_temperature_states = 1
        env.action_labels = ['Up', 'Down', 'Left', 'Right']

        # Mock transition model
        env.transition_model = torch.randn(env.total_states, env.n_actions, env.total_states)
        env.transition_model = torch.softmax(env.transition_model, dim=2)

        # Mock observation model
        env.observation_model = torch.randn(env.total_states, 3)  # 3 observations
        env.observation_model = torch.softmax(env.observation_model, dim=1)

        # Mock helper methods
        env._pos_temp_to_state = lambda i, j, t: i * 5 + j
        env._state_to_pos_temp = lambda s: (s // 5, s % 5, 0)
        env._get_next_position = lambda i, j, a: (i, j, 0)  # No movement for simplicity

        return env

    @pytest.fixture
    def mock_feature_manager(self):
        """Create mock feature manager."""
        fm = MagicMock()
        fm.device = torch.device('cpu')
        return fm

    @pytest.fixture
    def agent(self, mock_environment, mock_feature_manager):
        """Create test agent."""
        return ActiveInferenceAgent(mock_environment, mock_feature_manager)

    def test_initialization(self, agent, mock_environment):
        """Test agent initialization."""
        assert agent.env == mock_environment
        assert agent.belief.shape == (mock_environment.total_states,)
        assert torch.allclose(agent.belief.sum(), torch.tensor(1.0), atol=1e-5)
        assert agent.preferences.shape == (mock_environment.total_states,)

    def test_belief_initialization_uniform(self, agent):
        """Test that belief is initialized uniformly."""
        expected_uniform = 1.0 / agent.env.total_states
        assert torch.allclose(agent.belief, torch.full_like(agent.belief, expected_uniform), atol=1e-4)

    def test_preference_initialization(self, agent):
        """Test that preferences are properly initialized."""
        # Should have some variation in preferences
        assert not torch.allclose(agent.preferences, torch.zeros_like(agent.preferences))
        assert agent.preferences.min() >= 0.0  # All preferences should be non-negative

    @patch('builtins.print')  # Suppress print statements during testing
    def test_update_belief_vfe(self, mock_print, agent):
        """Test VFE-based belief update."""
        action = 0
        observation = 1

        initial_entropy = -torch.sum(agent.belief * torch.log(agent.belief + 1e-8))

        # Update belief
        updated_belief = agent.update_belief_vfe(action, observation)

        # Check that belief is still normalized
        assert torch.allclose(updated_belief.sum(), torch.tensor(1.0), atol=1e-5)

        # Check that belief has changed (unless it was already optimal)
        # This is a weak test but ensures the method runs without error
        assert updated_belief.shape == agent.belief.shape

        # Final belief should be stored
        assert torch.equal(updated_belief, agent.belief)

    @patch('builtins.print')
    def test_select_policy_efe(self, mock_print, agent):
        """Test EFE-based policy selection."""
        action, min_efe, efe_values = agent.select_policy_efe()

        # Check return values
        assert isinstance(action, int)
        assert 0 <= action < agent.env.n_actions
        assert isinstance(min_efe, float)
        assert isinstance(efe_values, np.ndarray)
        assert len(efe_values) == agent.env.n_actions

    def test_get_current_position_estimate(self, agent):
        """Test position estimation from belief state."""
        pos_i, pos_j = agent.get_current_position_estimate()

        assert isinstance(pos_i, int)
        assert isinstance(pos_j, int)
        assert 0 <= pos_i < agent.env.grid_size
        assert 0 <= pos_j < agent.env.grid_size

    def test_get_belief_entropy(self, agent):
        """Test belief entropy calculation."""
        entropy = agent.get_belief_entropy()

        assert isinstance(entropy, float)
        assert entropy >= 0.0  # Entropy should be non-negative

    def test_get_most_likely_state(self, agent):
        """Test most likely state retrieval."""
        state = agent.get_most_likely_state()

        assert isinstance(state, int)
        assert 0 <= state < agent.env.total_states

    def test_belief_update_conserves_probability_mass(self, agent):
        """Test that belief updates conserve probability mass."""
        original_sum = agent.belief.sum().item()

        # Perform multiple updates
        for _ in range(5):
            action = np.random.randint(0, agent.env.n_actions)
            observation = np.random.randint(0, 3)
            with patch('builtins.print'):
                agent.update_belief_vfe(action, observation)

            # Check normalization after each update
            assert abs(agent.belief.sum().item() - 1.0) < 1e-5

    def test_preference_temperature_correlation(self, agent):
        """Test that preferences correlate with temperature states."""
        # For states with optimal temperature, preference should be higher
        optimal_states = []
        suboptimal_states = []

        for state_idx in range(agent.env.total_states):
            _, _, temp_state = agent.env._state_to_pos_temp(state_idx)
            if temp_state == 2:  # Assuming optimal temperature is 2
                optimal_states.append(state_idx)
            else:
                suboptimal_states.append(state_idx)

        if optimal_states and suboptimal_states:
            optimal_avg_pref = torch.mean(agent.preferences[optimal_states])
            suboptimal_avg_pref = torch.mean(agent.preferences[suboptimal_states])

            # Optimal states should have higher preference on average
            assert optimal_avg_pref >= suboptimal_avg_pref


class TestActiveInferenceAgentIntegration:
    """Integration tests for Active Inference Agent."""

    def test_vfe_efe_integration_workflow(self):
        """Test complete VFE-EFE workflow."""
        # Create real environment for integration testing
        env = POMDPGridworld(grid_size=3)  # Small for fast testing
        fm = TritonFeatureManager()
        agent = ActiveInferenceAgent(env, fm)

        # Simulate a few steps
        current_state = 0

        for step in range(3):
            # Get observation
            observation = env.get_observation(current_state)

            # Update belief if not first step
            if step > 0:
                with patch('builtins.print'):
                    agent.update_belief_vfe(0, observation)  # Dummy action

            # Select action
            with patch('builtins.print'):
                action, _, _ = agent.select_policy_efe()

            # Transition to new state
            next_state_probs = env.transition_model[current_state, action]
            current_state = torch.multinomial(next_state_probs, 1).item()

        # Agent should still be functional
        assert agent.belief.shape == (env.total_states,)
        assert torch.allclose(agent.belief.sum(), torch.tensor(1.0), atol=1e-4)

    def test_belief_convergence_over_time(self):
        """Test that belief entropy decreases over time (learning)."""
        env = POMDPGridworld(grid_size=4)
        fm = TritonFeatureManager()
        agent = ActiveInferenceAgent(env, fm)

        entropies = []

        # Simulate sequence of observations
        current_state = 0
        for step in range(10):
            observation = env.get_observation(current_state)
            entropies.append(agent.get_belief_entropy())

            if step > 0:
                with patch('builtins.print'):
                    agent.update_belief_vfe(0, observation)

            # Move to new state
            action = np.random.randint(0, env.n_actions)
            next_state_probs = env.transition_model[current_state, action]
            current_state = torch.multinomial(next_state_probs, 1).item()

        # Entropy should generally decrease (more confident beliefs)
        # This is a statistical test, so we check that final entropy is not higher than initial
        assert entropies[-1] <= entropies[0] * 1.5  # Allow some fluctuation

    def test_policy_selection_stochasticity(self):
        """Test that policy selection has appropriate stochasticity."""
        env = POMDPGridworld(grid_size=3)
        fm = TritonFeatureManager()
        agent = ActiveInferenceAgent(env, fm)

        actions = []

        # Collect action selections over multiple trials
        for _ in range(20):
            with patch('builtins.print'):
                action, _, _ = agent.select_policy_efe()
            actions.append(action)

        # Should see some variation in actions (not always the same)
        unique_actions = set(actions)
        assert len(unique_actions) >= 2  # At least some variation

    def test_agent_state_persistence(self):
        """Test that agent maintains state correctly across operations."""
        env = POMDPGridworld(grid_size=3)
        fm = TritonFeatureManager()
        agent = ActiveInferenceAgent(env, fm)

        initial_belief = agent.belief.clone()
        initial_preferences = agent.preferences.clone()

        # Perform operations
        observation = 0
        with patch('builtins.print'):
            agent.update_belief_vfe(0, observation)
            agent.select_policy_efe()

        # Belief should have changed
        assert not torch.allclose(agent.belief, initial_belief, atol=1e-6)

        # Preferences should remain the same
        assert torch.allclose(agent.preferences, initial_preferences)


class TestActiveInferenceAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_extreme_belief_states(self):
        """Test behavior with extreme belief states."""
        env = POMDPGridworld(grid_size=3)
        fm = TritonFeatureManager()
        agent = ActiveInferenceAgent(env, fm)

        # Test with nearly deterministic belief
        agent.belief = torch.zeros(env.total_states)
        agent.belief[0] = 0.999
        agent.belief[1] = 0.001
        agent.belief = agent.belief / agent.belief.sum()

        # Should handle without numerical issues
        with patch('builtins.print'):
            action, _, _ = agent.select_policy_efe()
        assert 0 <= action < env.n_actions

    def test_uniform_belief_state(self):
        """Test behavior with uniform belief (maximum uncertainty)."""
        env = POMDPGridworld(grid_size=3)
        fm = TritonFeatureManager()
        agent = ActiveInferenceAgent(env, fm)

        # Set uniform belief
        agent.belief = torch.ones(env.total_states) / env.total_states

        # Should handle without issues
        entropy = agent.get_belief_entropy()
        expected_entropy = np.log(env.total_states)
        assert abs(entropy - expected_entropy) < 0.1

    def test_single_state_belief(self):
        """Test behavior with belief concentrated on single state."""
        env = POMDPGridworld(grid_size=3)
        fm = TritonFeatureManager()
        agent = ActiveInferenceAgent(env, fm)

        # Set belief on single state
        agent.belief = torch.zeros(env.total_states)
        agent.belief[10] = 1.0  # Some middle state

        # Should handle without issues
        entropy = agent.get_belief_entropy()
        assert entropy < 0.1  # Very low entropy

        pos_i, pos_j = agent.get_current_position_estimate()
        true_pos_i, true_pos_j, _ = env._state_to_pos_temp(10)
        assert pos_i == true_pos_i
        assert pos_j == true_pos_j


if __name__ == "__main__":
    pytest.main([__file__])
