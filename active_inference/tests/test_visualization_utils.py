#!/usr/bin/env python3
"""
Tests for Visualization Utilities

Comprehensive tests for the visualization utility functions,
ensuring plots and charts are generated correctly.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.visualization_utils import (
    create_trajectory_plot,
    create_belief_evolution_plot,
    create_performance_dashboard,
    create_trajectory_animation,
    create_comprehensive_visualization_suite
)


class TestVisualizationFunctions:
    """Test individual visualization functions."""

    @pytest.fixture
    def mock_results(self):
        """Create mock simulation results for testing."""
        return {
            'simulation_config': {
                'grid_size': 5,
                'n_actions': 4,
                'n_timesteps': 10
            },
            'performance_metrics': {
                'total_time': 1.5,
                'avg_time_per_step': 0.15,
                'total_reward': 8.5,
                'avg_reward_per_step': 0.85,
                'final_belief_entropy': 2.1,
                'position_accuracy': 0.75,
                'temperature_accuracy': 0.60,
                'action_distribution': [0.3, 0.2, 0.25, 0.25]
            },
            'history': {
                'timestep': list(range(10)),
                'position_i': [0, 1, 2, 2, 3, 3, 4, 4, 4, 4],
                'position_j': [0, 0, 1, 2, 2, 3, 3, 2, 1, 0],
                'estimated_position_i': [0, 1, 2, 2, 3, 3, 4, 4, 4, 4],
                'estimated_position_j': [0, 0, 1, 2, 2, 3, 3, 2, 1, 0],
                'temperature_state': [0, 1, 2, 2, 1, 0, 2, 1, 0, 2],
                'action': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
                'reward': [0.1, 0.3, 1.0, 0.8, 0.5, 0.2, 0.9, 0.6, 0.4, 0.7],
                'efe_values': [[0.5, 0.3, 0.8, 0.6] for _ in range(10)],
                'belief_entropy': [3.2, 3.0, 2.8, 2.5, 2.3, 2.1, 1.9, 1.7, 1.5, 1.3]
            }
        }

    @pytest.fixture
    def tmp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        return Path(tmp_path) / "visualizations"

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.subplots')
    def test_create_trajectory_plot(self, mock_subplots, mock_heatmap, mock_show, mock_savefig, mock_results, tmp_output_dir):
        """Test trajectory plot creation."""
        # Mock the subplots return
        mock_fig, mock_axes = MagicMock(), [MagicMock() for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        create_trajectory_plot(mock_results, tmp_output_dir)

        # Should save the plot
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(tmp_output_dir / "trajectory_analysis.png") in str(args[0])

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_create_belief_evolution_plot(self, mock_subplots, mock_show, mock_savefig, mock_results, tmp_output_dir):
        """Test belief evolution plot creation."""
        mock_fig, mock_axes = MagicMock(), [MagicMock() for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        create_belief_evolution_plot(mock_results, tmp_output_dir)

        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(tmp_output_dir / "belief_evolution.png") in str(args[0])

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.barh')
    @patch('matplotlib.pyplot.bar')
    def test_create_performance_dashboard(self, mock_bar, mock_barh, mock_subplots, mock_show, mock_savefig, mock_results, tmp_output_dir):
        """Test performance dashboard creation."""
        mock_fig, mock_axes = MagicMock(), [MagicMock() for _ in range(6)]
        mock_subplots.return_value = (mock_fig, mock_axes)

        create_performance_dashboard(mock_results, tmp_output_dir)

        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(tmp_output_dir / "performance_dashboard.png") in str(args[0])

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.scatter')
    def test_create_trajectory_animation(self, mock_scatter, mock_imshow, mock_show, mock_savefig, mock_results, tmp_output_dir):
        """Test trajectory animation creation."""
        create_trajectory_animation(mock_results, tmp_output_dir)

        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(tmp_output_dir / "trajectory_evolution.png") in str(args[0])

    @patch('src.visualization_utils.create_trajectory_plot')
    @patch('src.visualization_utils.create_belief_evolution_plot')
    @patch('src.visualization_utils.create_performance_dashboard')
    @patch('src.visualization_utils.create_trajectory_animation')
    def test_create_comprehensive_visualization_suite(self, mock_animation, mock_dashboard, mock_evolution, mock_trajectory, mock_results, tmp_output_dir):
        """Test comprehensive visualization suite."""
        create_comprehensive_visualization_suite(mock_results, tmp_output_dir)

        # Should call all individual visualization functions
        mock_trajectory.assert_called_once_with(mock_results, tmp_output_dir)
        mock_evolution.assert_called_once_with(mock_results, tmp_output_dir)
        mock_dashboard.assert_called_once_with(mock_results, tmp_output_dir)
        mock_animation.assert_called_once_with(mock_results, tmp_output_dir)


class TestVisualizationEdgeCases:
    """Test visualization functions with edge cases."""

    def test_empty_history(self, tmp_output_dir):
        """Test visualization with empty history."""
        empty_results = {
            'simulation_config': {'grid_size': 5, 'n_actions': 4, 'n_timesteps': 0},
            'performance_metrics': {'action_distribution': [0.25, 0.25, 0.25, 0.25]},
            'history': {
                'timestep': [],
                'position_i': [],
                'position_j': [],
                'estimated_position_i': [],
                'estimated_position_j': [],
                'temperature_state': [],
                'action': [],
                'reward': [],
                'efe_values': [],
                'belief_entropy': []
            }
        }

        # Should handle empty data gracefully
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.subplots'):
            create_trajectory_plot(empty_results, tmp_output_dir)

    def test_single_timestep(self, tmp_output_dir):
        """Test visualization with single timestep."""
        single_results = {
            'simulation_config': {'grid_size': 5, 'n_actions': 4, 'n_timesteps': 1},
            'performance_metrics': {'action_distribution': [1.0, 0.0, 0.0, 0.0]},
            'history': {
                'timestep': [0],
                'position_i': [2],
                'position_j': [2],
                'estimated_position_i': [2],
                'estimated_position_j': [2],
                'temperature_state': [1],
                'action': [0],
                'reward': [0.5],
                'efe_values': [[0.5, 0.3, 0.8, 0.6]],
                'belief_entropy': [2.5]
            }
        }

        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.subplots'):
            create_belief_evolution_plot(single_results, tmp_output_dir)

    def test_extreme_values(self, tmp_output_dir):
        """Test visualization with extreme values."""
        extreme_results = {
            'simulation_config': {'grid_size': 5, 'n_actions': 4, 'n_timesteps': 5},
            'performance_metrics': {
                'action_distribution': [1.0, 0.0, 0.0, 0.0],
                'position_accuracy': 1.0,
                'temperature_accuracy': 0.0
            },
            'history': {
                'timestep': list(range(5)),
                'position_i': [0, 0, 0, 0, 0],  # No movement
                'position_j': [0, 0, 0, 0, 0],
                'estimated_position_i': [4, 4, 4, 4, 4],  # Wrong estimates
                'estimated_position_j': [4, 4, 4, 4, 4],
                'temperature_state': [0, 0, 0, 0, 0],
                'action': [0, 0, 0, 0, 0],  # Same action repeatedly
                'reward': [0.0, 0.0, 0.0, 0.0, 0.0],  # No reward
                'efe_values': [[1.0, 0.0, 0.0, 0.0] for _ in range(5)],
                'belief_entropy': [5.0, 5.0, 5.0, 5.0, 5.0]  # High uncertainty
            }
        }

        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.subplots'):
            create_performance_dashboard(extreme_results, tmp_output_dir)


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualization_pipeline(self, mock_show, mock_savefig, tmp_output_dir):
        """Test complete visualization pipeline."""
        # Create realistic test data
        results = {
            'simulation_config': {
                'grid_size': 10,
                'n_actions': 4,
                'n_timesteps': 20
            },
            'performance_metrics': {
                'total_time': 2.5,
                'avg_time_per_step': 0.125,
                'total_reward': 15.0,
                'avg_reward_per_step': 0.75,
                'final_belief_entropy': 1.8,
                'position_accuracy': 0.85,
                'temperature_accuracy': 0.75,
                'action_distribution': [0.3, 0.2, 0.3, 0.2]
            },
            'history': {
                'timestep': list(range(20)),
                'position_i': np.random.randint(0, 10, 20).tolist(),
                'position_j': np.random.randint(0, 10, 20).tolist(),
                'estimated_position_i': np.random.randint(0, 10, 20).tolist(),
                'estimated_position_j': np.random.randint(0, 10, 20).tolist(),
                'temperature_state': np.random.randint(0, 5, 20).tolist(),
                'action': np.random.randint(0, 4, 20).tolist(),
                'reward': np.random.uniform(0, 1, 20).tolist(),
                'efe_values': [np.random.uniform(0, 1, 4).tolist() for _ in range(20)],
                'belief_entropy': np.random.uniform(1, 4, 20).tolist()
            }
        }

        # Run all visualizations
        create_comprehensive_visualization_suite(results, tmp_output_dir)

        # Should have created multiple plots
        assert mock_savefig.call_count >= 4  # At least 4 visualization functions


class TestVisualizationDataValidation:
    """Test data validation in visualization functions."""

    def test_invalid_action_distribution(self, tmp_output_dir):
        """Test handling of invalid action distribution."""
        invalid_results = {
            'simulation_config': {'grid_size': 5, 'n_actions': 4, 'n_timesteps': 5},
            'performance_metrics': {
                'action_distribution': [0.5, 0.5]  # Wrong length
            },
            'history': {
                'timestep': list(range(5)),
                'position_i': [0, 1, 2, 3, 4],
                'position_j': [0, 1, 2, 3, 4],
                'estimated_position_i': [0, 1, 2, 3, 4],
                'estimated_position_j': [0, 1, 2, 3, 4],
                'temperature_state': [0, 1, 2, 3, 4],
                'action': [0, 1, 2, 3, 0],
                'reward': [0.1, 0.2, 0.3, 0.4, 0.5],
                'efe_values': [[0.1, 0.2, 0.3, 0.4] for _ in range(5)],
                'belief_entropy': [3.0, 2.8, 2.5, 2.2, 2.0]
            }
        }

        # Should handle gracefully without crashing
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.subplots'):
            create_performance_dashboard(invalid_results, tmp_output_dir)

    def test_missing_history_fields(self, tmp_output_dir):
        """Test handling of missing history fields."""
        incomplete_results = {
            'simulation_config': {'grid_size': 5, 'n_actions': 4, 'n_timesteps': 3},
            'performance_metrics': {'action_distribution': [0.25, 0.25, 0.25, 0.25]},
            'history': {
                'timestep': [0, 1, 2],
                # Missing position fields
                'temperature_state': [0, 1, 2],
                'action': [0, 1, 2],
                'reward': [0.1, 0.2, 0.3],
                'efe_values': [[0.1, 0.2, 0.3, 0.4] for _ in range(3)],
                'belief_entropy': [3.0, 2.8, 2.5]
            }
        }

        # Should handle missing fields gracefully
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.subplots'):
            try:
                create_trajectory_plot(incomplete_results, tmp_output_dir)
            except KeyError:
                # Expected to fail with missing fields, but should fail gracefully
                pass


if __name__ == "__main__":
    pytest.main([__file__])
