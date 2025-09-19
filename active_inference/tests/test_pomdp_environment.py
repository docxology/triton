#!/usr/bin/env python3
"""
Tests for POMDP Environment

Comprehensive tests for the POMDPGridworld environment class,
ensuring all generative model components work correctly.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pomdp_environment import POMDPGridworld


class TestPOMDPGridworld:
    """Test POMDPGridworld environment functionality."""

    @pytest.fixture
    def environment(self):
        """Create test environment."""
        return POMDPGridworld(grid_size=5)  # Use defaults: 5 temp states, 5 observations

    def test_initialization(self, environment):
        """Test environment initialization."""
        assert environment.grid_size == 5
        assert environment.n_temperature_states == 5  # Default value
        assert environment.n_observations == 5  # Default value
        assert environment.total_states == 5 * 5 * 5  # 125 states
        assert environment.n_actions == 4  # up, down, left, right
        assert len(environment.temperature_labels) == 5
        assert len(environment.observation_labels) == 5
        assert len(environment.action_labels) == 4

    def test_temperature_grid_initialization(self, environment):
        """Test temperature grid is properly initialized."""
        assert environment.temperature_grid.shape == (5, 5, 5)
        # Check that probabilities are normalized
        for i in range(5):
            for j in range(5):
                probs = environment.temperature_grid[i, j]
                assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

    def test_transition_model_structure(self, environment):
        """Test transition model has correct structure."""
        expected_shape = (environment.total_states, environment.n_actions, environment.total_states)
        assert environment.transition_model.shape == expected_shape

    def test_observation_model_structure(self, environment):
        """Test observation model has correct structure."""
        expected_shape = (environment.total_states, 5)  # 5 observations by default
        assert environment.observation_model.shape == expected_shape

    def test_state_index_conversion(self, environment):
        """Test state to position/temperature and back conversion."""
        # Test a few random states
        test_states = [0, 10, 50, 124]  # First, middle, and last states (0-124 for 125 states)

        for state_idx in test_states:
            pos_i, pos_j, temp_state = environment._state_to_pos_temp(state_idx)
            reconstructed_idx = environment._pos_temp_to_state(pos_i, pos_j, temp_state)
            assert reconstructed_idx == state_idx

    def test_position_bounds_checking(self, environment):
        """Test position movement stays within bounds."""
        # Test all edge positions
        edge_positions = [
            (0, 0),    # Top-left
            (0, 4),    # Top-right
            (4, 0),    # Bottom-left
            (4, 4),    # Bottom-right
            (2, 2),    # Center
        ]

        for pos_i, pos_j in edge_positions:
            for action in range(environment.n_actions):
                next_i, next_j = environment._get_next_position(pos_i, pos_j, action)
                assert 0 <= next_i < environment.grid_size
                assert 0 <= next_j < environment.grid_size

    def test_observation_generation(self, environment):
        """Test observation generation from states."""
        # Test multiple states
        for state_idx in range(min(10, environment.total_states)):
            observation = environment.get_observation(state_idx)
            assert 0 <= observation < environment.n_observations

    def test_reward_calculation(self, environment):
        """Test reward calculation for different states."""
        # Test that rewards are reasonable
        for state_idx in range(min(20, environment.total_states)):
            reward = environment.get_reward(state_idx)
            assert isinstance(reward, float)
            assert -1.0 <= reward <= 1.0  # Reasonable reward range

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualization_methods(self, mock_show, mock_savefig, environment, tmp_path):
        """Test visualization methods don't crash."""
        output_dir = Path(tmp_path)

        # Test transition model visualization
        environment.visualize_transition_model(output_dir)
        mock_savefig.assert_called_with(output_dir / "transition_model.png", dpi=300, bbox_inches='tight')

        # Test observation model visualization
        mock_savefig.reset_mock()
        environment.visualize_observation_model(output_dir)
        mock_savefig.assert_called_with(output_dir / "observation_model.png", dpi=300, bbox_inches='tight')

        # Test temperature grid visualization
        mock_savefig.reset_mock()
        environment.visualize_temperature_grid(output_dir)
        mock_savefig.assert_called_with(output_dir / "temperature_grid.png", dpi=300, bbox_inches='tight')

    def test_visualize_preferences_without_agent(self, environment, tmp_path):
        """Test preference visualization without agent."""
        output_dir = Path(tmp_path)

        with patch('matplotlib.pyplot.savefig'):
            environment.visualize_preferences(output_dir, agent=None)
            # Should create default preferences internally

    def test_belief_state_visualization(self, environment, tmp_path):
        """Test belief state visualization."""
        output_dir = Path(tmp_path)

        # Create mock belief state
        belief_state = torch.randn(environment.total_states)
        belief_state = torch.softmax(belief_state, dim=0)

        with patch('matplotlib.pyplot.savefig'):
            environment.visualize_state_space(belief_state, output_dir, timestep=5)
            # Should not crash

    def test_generative_model_visualization(self, environment, tmp_path):
        """Test comprehensive generative model visualization."""
        output_dir = Path(tmp_path)

        # Create mock agent with required attributes
        mock_agent = MagicMock()
        mock_agent.preferences = torch.randn(environment.total_states)
        mock_agent.belief = torch.softmax(torch.randn(environment.total_states), dim=0)

        with patch('matplotlib.pyplot.savefig'), \
             patch('json.dump'), \
             patch('pathlib.Path.stat', return_value=MagicMock(st_size=1024)):
            environment.create_generative_model_visualization(mock_agent, output_dir)
            # Should not crash and should export JSON

    def test_different_grid_sizes(self):
        """Test environment works with different grid sizes."""
        for grid_size in [3, 5, 8]:
            env = POMDPGridworld(grid_size=grid_size)
            expected_states = grid_size * grid_size * 5  # 5 temperature states
            assert env.total_states == expected_states

    def test_temperature_state_preferences(self, environment):
        """Test that optimal temperature state gets highest preference."""
        # Find states with optimal temperature (state 2)
        optimal_states = []
        for state_idx in range(environment.total_states):
            _, _, temp_state = environment._state_to_pos_temp(state_idx)
            if temp_state == 2:  # Optimal temperature
                optimal_states.append(state_idx)

        # Get rewards for optimal vs non-optimal states
        optimal_rewards = [environment.get_reward(state) for state in optimal_states[:5]]
        non_optimal_rewards = []

        for state_idx in range(min(20, environment.total_states)):
            _, _, temp_state = environment._state_to_pos_temp(state_idx)
            if temp_state != 2:
                non_optimal_rewards.append(environment.get_reward(state_idx))

        # Optimal states should generally have higher rewards
        avg_optimal = np.mean(optimal_rewards)
        avg_non_optimal = np.mean(non_optimal_rewards)

        # This is a statistical test - optimal should be higher on average
        assert avg_optimal >= avg_non_optimal


class TestPOMDPEnvironmentIntegration:
    """Integration tests for POMDP environment."""

    def test_complete_state_space_coverage(self):
        """Test that all possible states are accessible."""
        env = POMDPGridworld(grid_size=3)  # Smaller for faster testing

        visited_states = set()

        # Try to visit many states through random transitions
        current_state = 0
        for _ in range(100):  # Try 100 transitions
            action = np.random.randint(0, env.n_actions)
            next_state_probs = env.transition_model[current_state, action]
            next_state = torch.multinomial(next_state_probs, 1).item()
            visited_states.add(next_state)
            current_state = next_state

        # Should visit a reasonable number of states
        assert len(visited_states) > 10

    def test_observation_likelihood_consistency(self):
        """Test that observation likelihoods are properly normalized."""
        env = POMDPGridworld(grid_size=3)
        for state_idx in range(min(10, env.total_states)):
            obs_probs = env.observation_model[state_idx]
            # Should sum to approximately 1 (allowing for numerical precision)
            assert torch.allclose(obs_probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_transition_matrix_normalization(self):
        """Test that transition probabilities are properly normalized."""
        env = POMDPGridworld(grid_size=3)
        for state_idx in range(min(5, env.total_states)):
            for action in range(env.n_actions):
                transition_probs = env.transition_model[state_idx, action]
                # Should sum to approximately 1
                assert torch.allclose(transition_probs.sum(), torch.tensor(1.0), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
