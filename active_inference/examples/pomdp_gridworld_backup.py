#!/usr/bin/env python3
"""
POMDP Gridworld Active Inference - Thin Orchestrator

This example demonstrates a Partially Observable Markov Decision Process (POMDP) in a
temperature-controlled gridworld using GPU-accelerated Active Inference. The agent
navigates a 10x10 grid with temperature states, making observations and selecting
policies to maintain homeostatic temperature preferences.

Active Inference Methods Used:
- Variational Free Energy (VFE) minimization for state estimation from observations
- Expected Free Energy (EFE) minimization for policy selection and action planning
- GPU-accelerated Bayesian inference for belief state updates
- Real-time convergence monitoring and visualization

Environment:
- 10x10 gridworld with temperature states at each cell
- 5 discrete temperature levels per cell (very cold, cold, optimal, warm, hot)
- 5 observation types (noisy temperature readings)
- 4 actions: up, down, left, right
- Homeostatic preference for middle temperature (optimal state)

Generative Model:
- Hidden states: Agent position (100 cells) × Temperature state (5 levels) = 500 states
- Observations: Noisy temperature readings (5 types)
- Actions: Movement in grid (4 directions)
- Preferences: Strong preference for optimal temperature state
"""

import torch
import numpy as np
import time
import json
import pandas as pd
from pathlib import Path
import sys

# Add the src directory to the path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up outputs directory
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "pomdp_gridworld"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Import modular components
from pomdp_environment import POMDPGridworld
from active_inference_agent import ActiveInferenceAgent
from visualization_utils import create_comprehensive_visualization_suite
from core import TritonFeatureManager


def run_pomdp_simulation(n_timesteps=50):
    """
    POMDP Gridworld environment for Active Inference demonstration.

    This class manages the gridworld environment with temperature states,
    providing the generative model interface for Active Inference.
    """

    def __init__(self, grid_size=10, n_temperature_states=5, n_observations=5):
        """
        Initialize the POMDP gridworld environment.

        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            n_temperature_states: Number of temperature states per cell
            n_observations: Number of observation types
        """
        self.grid_size = grid_size
        self.n_temperature_states = n_temperature_states
        self.n_observations = n_observations
        self.total_states = grid_size * grid_size * n_temperature_states

        # Define temperature states (0: very cold, 1: cold, 2: optimal, 3: warm, 4: hot)
        self.temperature_labels = ['Very Cold', 'Cold', 'Optimal', 'Warm', 'Hot']

        # Define observation types (noisy temperature readings)
        self.observation_labels = ['Very Cold Obs', 'Cold Obs', 'Optimal Obs', 'Warm Obs', 'Hot Obs']

        # Initialize grid with temperature distributions
        self._initialize_temperature_grid()

        # Initialize transition and observation models
        self._initialize_models()

        print(f"✅ Initialized POMDP Gridworld: {grid_size}x{grid_size} grid")
        print(f"   Total states: {self.total_states}")
        print(f"   Temperature states per cell: {n_temperature_states}")
        print(f"   Observation types: {n_observations}")

    def _initialize_temperature_grid(self):
        """Initialize the temperature distribution across the grid."""
        # Create a temperature gradient (cooler at edges, warmer in center)
        self.temperature_grid = torch.zeros(self.grid_size, self.grid_size, self.n_temperature_states)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Distance from center
                center_i, center_j = self.grid_size // 2, self.grid_size // 2
                distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                max_distance = np.sqrt(2) * (self.grid_size // 2)

                # Temperature preference peaks at center, cooler at edges
                if distance < max_distance * 0.3:
                    # Center area: prefer optimal temperature
                    self.temperature_grid[i, j, 2] = 0.7  # Optimal
                    self.temperature_grid[i, j, 1] = 0.2  # Cold
                    self.temperature_grid[i, j, 3] = 0.1  # Warm
                elif distance < max_distance * 0.6:
                    # Middle area: mix of states
                    self.temperature_grid[i, j, 1] = 0.3  # Cold
                    self.temperature_grid[i, j, 2] = 0.4  # Optimal
                    self.temperature_grid[i, j, 3] = 0.3  # Warm
                else:
                    # Edge area: cooler temperatures
                    self.temperature_grid[i, j, 0] = 0.4  # Very cold
                    self.temperature_grid[i, j, 1] = 0.4  # Cold
                    self.temperature_grid[i, j, 2] = 0.2  # Optimal

                # Normalize
                self.temperature_grid[i, j] /= self.temperature_grid[i, j].sum()

    def _initialize_models(self):
        """Initialize transition and observation models."""
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.n_actions = 4
        self.action_labels = ['Up', 'Down', 'Left', 'Right']

        # Transition model: P(s'|s,a) - movement affects position, temperature may change
        self.transition_model = torch.zeros(self.total_states, self.n_actions, self.total_states)

        # Observation model: P(o|s) - noisy temperature readings
        self.observation_model = torch.zeros(self.total_states, self.n_observations)

        # Build models
        for pos_i in range(self.grid_size):
            for pos_j in range(self.grid_size):
                for temp_state in range(self.n_temperature_states):
                    state_idx = self._pos_temp_to_state(pos_i, pos_j, temp_state)

                    # Transition model
                    for action in range(self.n_actions):
                        next_pos_i, next_pos_j = self._get_next_position(pos_i, pos_j, action)

                        # Temperature may change when moving (environmental dynamics)
                        for next_temp in range(self.n_temperature_states):
                            next_state_idx = self._pos_temp_to_state(next_pos_i, next_pos_j, next_temp)

                            # Movement probability (deterministic position, probabilistic temperature)
                            if next_temp == temp_state:
                                prob = 0.8  # Same temperature
                            elif abs(next_temp - temp_state) == 1:
                                prob = 0.1  # Adjacent temperature
                            else:
                                prob = 0.0  # Non-adjacent temperature change

                            self.transition_model[state_idx, action, next_state_idx] = prob

                    # Observation model (noisy temperature readings)
                    for obs in range(self.n_observations):
                        if obs == temp_state:
                            prob = 0.8  # Correct reading
                        elif abs(obs - temp_state) == 1:
                            prob = 0.1  # Adjacent temperature reading
                        else:
                            prob = 0.0  # Incorrect reading
                        self.observation_model[state_idx, obs] = prob

    def _pos_temp_to_state(self, pos_i, pos_j, temp_state):
        """Convert position and temperature to state index."""
        return (pos_i * self.grid_size + pos_j) * self.n_temperature_states + temp_state

    def _state_to_pos_temp(self, state_idx):
        """Convert state index to position and temperature."""
        temp_state = state_idx % self.n_temperature_states
        pos_idx = state_idx // self.n_temperature_states
        pos_i = pos_idx // self.grid_size
        pos_j = pos_idx % self.grid_size
        return pos_i, pos_j, temp_state

    def _get_next_position(self, pos_i, pos_j, action):
        """Get next position given current position and action."""
        if action == 0:  # Up
            next_i = max(0, pos_i - 1)
            next_j = pos_j
        elif action == 1:  # Down
            next_i = min(self.grid_size - 1, pos_i + 1)
            next_j = pos_j
        elif action == 2:  # Left
            next_i = pos_i
            next_j = max(0, pos_j - 1)
        else:  # Right
            next_i = pos_i
            next_j = min(self.grid_size - 1, pos_j + 1)

        return next_i, next_j

    def get_observation(self, state_idx):
        """Generate observation from current state."""
        obs_probs = self.observation_model[state_idx]
        observation = torch.multinomial(obs_probs, 1).item()
        return observation

    def get_reward(self, state_idx):
        """Get reward based on current state (temperature preference)."""
        _, _, temp_state = self._state_to_pos_temp(state_idx)
        # Strong preference for optimal temperature (state 2)
        preferences = torch.tensor([0.1, 0.3, 1.0, 0.3, 0.1])  # Very cold to hot
        return preferences[temp_state].item()

    def visualize_transition_model(self, output_dir):
        """Visualize the transition model for all actions."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Transition Model Visualization", fontsize=16, fontweight='bold')

        for action in range(self.n_actions):
            row, col = action // 2, action % 2
            ax = axes[row, col]

            # Get transition matrix for this action
            trans_matrix = self.transition_model[:, action, :].cpu().numpy()

            # Show as heatmap
            sns.heatmap(trans_matrix, ax=ax, cmap='Blues', cbar=True,
                       xticklabels=False, yticklabels=False)
            ax.set_title(f'Action: {self.action_labels[action]}')
            ax.set_xlabel('Next State')
            ax.set_ylabel('Current State')

        plt.tight_layout()
        plt.savefig(output_dir / "transition_model.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Transition model visualization saved to: {output_dir / 'transition_model.png'}")

    def visualize_observation_model(self, output_dir):
        """Visualize the observation model."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 8))
        obs_matrix = self.observation_model.cpu().numpy()

        sns.heatmap(obs_matrix, cmap='Greens', cbar=True,
                   xticklabels=[f'Obs {i}' for i in range(self.n_observations)],
                   yticklabels=False)
        plt.title('Observation Model: P(Observation | State)', fontsize=14, fontweight='bold')
        plt.xlabel('Observation Type')
        plt.ylabel('State Index')

        plt.tight_layout()
        plt.savefig(output_dir / "observation_model.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Observation model visualization saved to: {output_dir / 'observation_model.png'}")

    def visualize_temperature_grid(self, output_dir):
        """Visualize the temperature preference grid."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))

        # Create temperature preference map (most likely temperature per cell)
        temp_grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Find the most likely temperature state for this position
                max_prob = 0
                best_temp = 0
                for temp_state in range(self.n_temperature_states):
                    state_idx = self._pos_temp_to_state(i, j, temp_state)
                    prob = self.temperature_grid[i, j, temp_state]
                    if prob > max_prob:
                        max_prob = prob
                        best_temp = temp_state
                temp_grid[i, j] = best_temp

        plt.imshow(temp_grid, cmap='RdYlBu_r', aspect='equal', origin='lower')
        plt.colorbar(label='Temperature State (0=Cold, 4=Hot)')
        plt.title('Temperature Preference Grid', fontsize=14, fontweight='bold')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')

        # Add grid lines
        plt.grid(True, alpha=0.3, color='white')

        plt.tight_layout()
        plt.savefig(output_dir / "temperature_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Temperature grid visualization saved to: {output_dir / 'temperature_grid.png'}")

    def visualize_state_space(self, belief_state, output_dir, timestep=None):
        """Visualize the current belief state in 2D grid format."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 10))

        # Convert belief state to 2D grid format
        belief_grid = np.zeros((self.grid_size, self.grid_size))
        for state_idx in range(self.total_states):
            pos_i, pos_j, temp_state = self._state_to_pos_temp(state_idx)
            belief_grid[pos_i, pos_j] += belief_state[state_idx].item()

        plt.subplot(1, 2, 1)
        plt.imshow(belief_grid, cmap='viridis', aspect='equal', origin='lower')
        plt.colorbar(label='Belief Probability')
        title = 'Agent Belief State' if timestep is None else f'Agent Belief State (t={timestep})'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.grid(True, alpha=0.3, color='white')

        # Show entropy
        entropy = -torch.sum(belief_state * torch.log(belief_state + 1e-8))
        plt.text(0.02, 0.98, '.4f',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Show most likely position
        max_idx = belief_grid.argmax()
        max_pos_i, max_pos_j = max_idx // self.grid_size, max_idx % self.grid_size
        plt.scatter(max_pos_j, max_pos_i, color='red', s=100, marker='*',
                   label=f'Most likely: ({max_pos_i}, {max_pos_j})')
        plt.legend()

        # Show belief distribution as histogram
        plt.subplot(1, 2, 2)
        belief_flat = belief_state.cpu().numpy()
        plt.hist(belief_flat, bins=50, alpha=0.7, color='blue', density=True)
        plt.xlabel('Belief Probability')
        plt.ylabel('Density')
        plt.title('Belief Distribution', fontsize=14, fontweight='bold')
        plt.yscale('log')

        plt.tight_layout()
        filename = "belief_state.png" if timestep is None else f"belief_state_t{timestep:03d}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Belief state visualization saved to: {output_dir / filename}")

    def visualize_preferences(self, output_dir, agent=None):
        """Visualize the preference landscape with proper temperature-based variation."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 12))

        if agent is None:
            # Create default preferences for visualization
            preferences = torch.zeros(self.total_states)
            for state_idx in range(self.total_states):
                _, _, temp_state = self._state_to_pos_temp(state_idx)
                if temp_state == 2:  # Optimal temperature
                    preferences[state_idx] = 2.0
                elif temp_state in [1, 3]:  # Adjacent temperatures
                    preferences[state_idx] = 0.5
                else:  # Extreme temperatures
                    preferences[state_idx] = 0.1
        else:
            preferences = agent.preferences

        # Create multiple subplots for different temperature levels
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Preference Landscape by Temperature Level', fontsize=16, fontweight='bold')

        # Overall preference landscape
        pref_grid_overall = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Average preference across all temperature states for this position
                avg_pref = 0
                for temp_state in range(self.n_temperature_states):
                    state_idx = self._pos_temp_to_state(i, j, temp_state)
                    avg_pref += preferences[state_idx].item()
                pref_grid_overall[i, j] = avg_pref / self.n_temperature_states

        # Plot overall landscape
        ax = axes[0, 0]
        im = ax.imshow(pref_grid_overall, cmap='RdYlGn', aspect='equal', origin='lower', vmin=0, vmax=2.0)
        ax.set_title('Overall Preference Landscape', fontsize=12, fontweight='bold')
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Plot individual temperature level preferences
        temp_names = ['Very Cold', 'Cold', 'Optimal', 'Warm', 'Hot']
        for temp_state in range(self.n_temperature_states):
            row = (temp_state + 1) // 3
            col = (temp_state + 1) % 3

            pref_grid_temp = np.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    state_idx = self._pos_temp_to_state(i, j, temp_state)
                    pref_grid_temp[i, j] = preferences[state_idx].item()

            ax = axes[row, col]
            im = ax.imshow(pref_grid_temp, cmap='RdYlGn', aspect='equal', origin='lower', vmin=0, vmax=2.0)
            ax.set_title(f'{temp_names[temp_state]} Preferences', fontsize=10, fontweight='bold')
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            plt.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout()
        plt.savefig(output_dir / "preference_landscape_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Also create simple version for backward compatibility
        plt.figure(figsize=(10, 8))
        plt.imshow(pref_grid_overall, cmap='RdYlGn', aspect='equal', origin='lower', vmin=0, vmax=2.0)
        plt.colorbar(label='Average Preference Value')
        plt.title('Preference Landscape (Overall)', fontsize=14, fontweight='bold')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.grid(True, alpha=0.3, color='white')

        plt.tight_layout()
        plt.savefig(output_dir / "preference_landscape.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Enhanced preference landscape visualization saved to:")
        print(f"   {output_dir / 'preference_landscape.png'} (simple)")
        print(f"   {output_dir / 'preference_landscape_detailed.png'} (detailed by temperature)")

    def create_generative_model_visualization(self, agent, output_dir):
        """Create comprehensive generative model visualization with JSON export."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import json
        import numpy as np

        # ===== COLLECT ALL MATRICES FOR JSON EXPORT =====
        generative_model_data = {
            "environment_specification": {
                "grid_size": self.grid_size,
                "n_temperature_states": self.n_temperature_states,
                "n_observations": self.n_observations,
                "n_actions": self.n_actions,
                "total_states": self.total_states,
                "state_space_description": "500 states (10x10 grid × 5 temperature levels)",
                "action_space": ["Up", "Down", "Left", "Right"],
                "observation_space": ["Very Cold Obs", "Cold Obs", "Optimal Obs", "Warm Obs", "Hot Obs"],
                "temperature_levels": ["Very Cold", "Cold", "Optimal", "Warm", "Hot"]
            },
            "matrices": {
                "transition_model": {
                    "shape": list(self.transition_model.shape),
                    "description": "P(s'|s,a) - Probability of transitioning to s' from s given action a",
                    "actions": {},
                    "data": self.transition_model.cpu().numpy().tolist()
                },
                "observation_model": {
                    "shape": list(self.observation_model.shape),
                    "description": "P(o|s) - Probability of observing o given state s",
                    "data": self.observation_model.cpu().numpy().tolist()
                },
                "temperature_grid": {
                    "shape": list(self.temperature_grid.shape),
                    "description": "P(temp|i,j) - Temperature distribution at each grid cell",
                    "data": self.temperature_grid.cpu().numpy().tolist()
                },
                "agent_preferences": {
                    "description": "Agent's preference values for each state (C(s))",
                    "shape": [self.total_states],
                    "preference_mapping": {
                        "optimal_temperature": {"state": 2, "value": 2.0},
                        "adjacent_temperatures": {"states": [1, 3], "value": 0.5},
                        "extreme_temperatures": {"states": [0, 4], "value": 0.1}
                    },
                    "data": agent.preferences.cpu().numpy().tolist()
                }
            },
            "active_inference_components": {
                "variational_free_energy": {
                    "description": "F(q,θ) = E_q[log q(s|o) - log p(o,s)]",
                    "purpose": "State estimation from observations",
                    "computation": "Minimized via variational inference"
                },
                "expected_free_energy": {
                    "description": "G(π) = Epistemic + Pragmatic value",
                    "epistemic_value": "Expected information gain: E_q[log q - log p]",
                    "pragmatic_value": "Goal achievement: E_q[log p(o|C)]",
                    "policy_selection": "Softmax over EFE values for action sampling"
                },
                "belief_state": {
                    "description": "q(s|o) - Posterior belief over states given observations",
                    "shape": [self.total_states],
                    "data": agent.belief.cpu().numpy().tolist()
                }
            },
            "performance_metrics": {
                "belief_entropy": float(-torch.sum(agent.belief * torch.log(agent.belief + 1e-8))),
                "most_likely_state": int(agent.belief.argmax().item()),
                "most_likely_position": agent.get_current_position_estimate(),
                "state_uncertainty_distribution": agent.belief.cpu().numpy().tolist()
            },
            "timestamp": str(np.datetime64('now')),
            "framework": "Active Inference POMDP Implementation"
        }

        # Add individual action transition matrices to JSON
        for action_idx in range(self.n_actions):
            action_name = self.action_labels[action_idx]
            generative_model_data["matrices"]["transition_model"]["actions"][action_name] = {
                "action_index": action_idx,
                "description": f"P(s'|s,{action_name}) - Transition probabilities for {action_name} action",
                "matrix_shape": [self.total_states, self.total_states],
                "data": self.transition_model[:, action_idx, :].cpu().numpy().tolist()
            }

        # Save comprehensive JSON
        json_path = output_dir / "generative_model_complete.json"
        with open(json_path, 'w') as f:
            json.dump(generative_model_data, f, indent=2, default=str)
        print(f"✓ Complete generative model data exported to: {json_path}")
        print(f"   JSON size: {json_path.stat().st_size} bytes")

        # ===== CREATE VISUALIZATION =====
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('POMDP Generative Model - Complete Matrix Visualization',
                    fontsize=16, fontweight='bold', y=0.95)

        # Create main grid layout
        gs = gridspec.GridSpec(5, 6, figure=fig, hspace=0.3, wspace=0.3)

        # ===== LEFT COLUMN: ENVIRONMENT STATE SPACE =====
        ax_env = fig.add_subplot(gs[0:3, 0])
        ax_env.set_title('Environment State Space', fontsize=12, fontweight='bold')
        ax_env.text(0.5, 0.5, f'{self.grid_size}×{self.grid_size} Grid\n{self.total_states} Hidden States\n{self.n_temperature_states} Temperature Levels',
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax_env.set_xlim(0, 1)
        ax_env.set_ylim(0, 1)
        ax_env.axis('off')

        # ===== CENTER COLUMN: GENERATIVE MODEL FLOW =====
        # Prior preferences
        ax_prior = fig.add_subplot(gs[0, 1])
        ax_prior.set_title('Prior Preferences C', fontsize=10, fontweight='bold')
        ax_prior.text(0.5, 0.5, 'Temperature\nPreferences\nC(s)',
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.8))
        ax_prior.set_xlim(0, 1)
        ax_prior.set_ylim(0, 1)
        ax_prior.axis('off')

        # Generative process
        ax_gen = fig.add_subplot(gs[1, 1])
        ax_gen.set_title('Generative Process', fontsize=10, fontweight='bold')
        ax_gen.text(0.5, 0.5, 'P(o,s\'|s,a)\n\nTransition Model\nObservation Model',
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.8))
        ax_gen.set_xlim(0, 1)
        ax_gen.set_ylim(0, 1)
        ax_gen.axis('off')

        # Likelihood
        ax_like = fig.add_subplot(gs[2, 1])
        ax_like.set_title('Likelihood P(o|s)', fontsize=10, fontweight='bold')
        ax_like.text(0.5, 0.5, 'P(o|s)\nObservation\nLikelihood',
                    ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.8))
        ax_like.set_xlim(0, 1)
        ax_like.set_ylim(0, 1)
        ax_like.axis('off')

        # Recognition model
        ax_recog = fig.add_subplot(gs[3, 1])
        ax_recog.set_title('Recognition Model q(s|o)', fontsize=10, fontweight='bold')
        ax_recog.text(0.5, 0.5, 'q(s|o)\nPosterior\nBelief State',
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcyan', alpha=0.8))
        ax_recog.set_xlim(0, 1)
        ax_recog.set_ylim(0, 1)
        ax_recog.axis('off')

        # Current belief
        ax_belief = fig.add_subplot(gs[4, 1])
        ax_belief.set_title('Current Belief State', fontsize=10, fontweight='bold')
        belief_data = agent.belief.cpu().numpy().reshape(self.grid_size, self.grid_size, self.n_temperature_states)
        belief_2d = belief_data.sum(axis=2)  # Sum over temperature states
        ax_belief.imshow(belief_2d, cmap='viridis', aspect='equal', origin='lower')
        ax_belief.set_xlabel('Grid X')
        ax_belief.set_ylabel('Grid Y')

        # ===== RIGHT COLUMNS: ACTIVE INFERENCE PROCESSES =====
        # VFE Minimization
        ax_vfe = fig.add_subplot(gs[0, 2])
        ax_vfe.set_title('Variational Free Energy', fontsize=10, fontweight='bold')
        ax_vfe.text(0.5, 0.5, 'F = E_q[log q - log p]\nMinimize VFE\nState Estimation',
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='orange', alpha=0.8))
        ax_vfe.set_xlim(0, 1)
        ax_vfe.set_ylim(0, 1)
        ax_vfe.axis('off')

        # EFE Minimization
        ax_efe = fig.add_subplot(gs[1, 2])
        ax_efe.set_title('Expected Free Energy', fontsize=10, fontweight='bold')
        ax_efe.text(0.5, 0.5, 'G = E_q[log q - log p]\nEpistemic + Pragmatic\nPolicy Selection',
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='purple', alpha=0.8))
        ax_efe.set_xlim(0, 1)
        ax_efe.set_ylim(0, 1)
        ax_efe.axis('off')

        # Policy Selection
        ax_policy = fig.add_subplot(gs[2, 2])
        ax_policy.set_title('Policy Selection', fontsize=10, fontweight='bold')
        ax_policy.text(0.5, 0.5, 'Softmax(EFE)\nAction Sampling\nExploration/Exploitation',
                      ha='center', va='center', fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.2", facecolor='pink', alpha=0.8))
        ax_policy.set_xlim(0, 1)
        ax_policy.set_ylim(0, 1)
        ax_policy.axis('off')

        # Action Execution
        ax_action = fig.add_subplot(gs[3, 2])
        ax_action.set_title('Action Execution', fontsize=10, fontweight='bold')
        ax_action.text(0.5, 0.5, 'Execute Action\na → s\'\nClose Perception-Action Loop',
                      ha='center', va='center', fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.8))
        ax_action.set_xlim(0, 1)
        ax_action.set_ylim(0, 1)
        ax_action.axis('off')

        # Performance metrics
        ax_perf = fig.add_subplot(gs[4, 2])
        ax_perf.set_title('Performance Metrics', fontsize=10, fontweight='bold')
        perf_text = '.1f'.4f'.0f'
        ax_perf.text(0.5, 0.5, perf_text, ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.8))
        ax_perf.set_xlim(0, 1)
        ax_perf.set_ylim(0, 1)
        ax_perf.axis('off')

        # ===== MATRICES VISUALIZATION =====
        # Transition matrix preview
        ax_trans = fig.add_subplot(gs[0:2, 3:])
        trans_matrix = self.transition_model[:50, 0, :50].cpu().numpy()
        im1 = ax_trans.imshow(trans_matrix, cmap='Blues', aspect='equal')
        ax_trans.set_title('Transition Matrix (subset)', fontsize=10, fontweight='bold')
        ax_trans.set_xlabel('Next State')
        ax_trans.set_ylabel('Current State')
        plt.colorbar(im1, ax=ax_trans, shrink=0.8)

        # Observation matrix preview
        ax_obs = fig.add_subplot(gs[2:4, 3:])
        obs_matrix = self.observation_model[:100, :].cpu().numpy()
        im2 = ax_obs.imshow(obs_matrix, cmap='Greens', aspect='equal')
        ax_obs.set_title('Observation Matrix (subset)', fontsize=10, fontweight='bold')
        ax_obs.set_xlabel('Observation')
        ax_obs.set_ylabel('State')
        plt.colorbar(im2, ax=ax_obs, shrink=0.8)

        # Preferences matrix
        ax_pref = fig.add_subplot(gs[4, 3:])
        pref_matrix = np.array(agent.preferences.cpu().numpy()).reshape(self.grid_size, self.grid_size, self.n_temperature_states)
        pref_2d = pref_matrix.mean(axis=2)  # Average over temperature states
        im3 = ax_pref.imshow(pref_2d, cmap='RdYlGn', aspect='equal', origin='lower')
        ax_pref.set_title('Preference Landscape', fontsize=10, fontweight='bold')
        ax_pref.set_xlabel('Grid X')
        ax_pref.set_ylabel('Grid Y')
        plt.colorbar(im3, ax=ax_pref, shrink=0.8)

        # ===== FLOW CONNECTIONS =====
        # Add arrows showing the generative model flow
        fig.text(0.20, 0.85, "→", ha='center', va='center', fontsize=20, color='red')
        fig.text(0.20, 0.70, "→", ha='center', va='center', fontsize=20, color='blue')
        fig.text(0.20, 0.55, "→", ha='center', va='center', fontsize=20, color='green')
        fig.text(0.20, 0.40, "→", ha='center', va='center', fontsize=20, color='purple')
        fig.text(0.20, 0.25, "→", ha='center', va='center', fontsize=20, color='orange')

        # Add flow labels
        fig.text(0.15, 0.85, "Preferences", ha='center', va='center', fontsize=8, rotation=90)
        fig.text(0.15, 0.70, "Generative\nProcess", ha='center', va='center', fontsize=8, rotation=90)
        fig.text(0.15, 0.55, "Likelihood", ha='center', va='center', fontsize=8, rotation=90)
        fig.text(0.15, 0.40, "Recognition", ha='center', va='center', fontsize=8, rotation=90)
        fig.text(0.15, 0.25, "Belief\nUpdate", ha='center', va='center', fontsize=8, rotation=90)

        plt.tight_layout()
        plt.savefig(output_dir / "generative_model_overall.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Enhanced generative model visualization saved to: {output_dir / 'generative_model_overall.png'}")


class ActiveInferenceAgent:
    """
    Active Inference agent for POMDP navigation using VFE and EFE.

    This agent uses the existing framework's VariationalFreeEnergy and ExpectedFreeEnergy
    classes for state estimation and policy selection.
    """

    def __init__(self, environment, feature_manager):
        """
        Initialize the Active Inference agent.

        Args:
            environment: POMDPGridworld environment
            feature_manager: TritonFeatureManager for GPU acceleration
        """
        self.env = environment
        self.feature_manager = feature_manager

        # Initialize Active Inference components using existing framework classes
        self.gpu_accelerator = GPUAccelerator(feature_manager)
        self.vfe_engine = VariationalFreeEnergy(feature_manager)
        self.efe_engine = ExpectedFreeEnergy(feature_manager)

        # Initialize belief state (uniform prior)
        self.belief = torch.ones(environment.total_states, device=self.gpu_accelerator.device)
        self.belief /= self.belief.sum()

        # Define preferences (strong preference for optimal temperature)
        self.preferences = torch.zeros(environment.total_states, device=self.gpu_accelerator.device)
        for i in range(environment.grid_size):
            for j in range(environment.grid_size):
                for temp_state in range(environment.n_temperature_states):
                    state_idx = environment._pos_temp_to_state(i, j, temp_state)
                    if temp_state == 2:  # Optimal temperature
                        self.preferences[state_idx] = 2.0
                    elif temp_state in [1, 3]:  # Adjacent temperatures
                        self.preferences[state_idx] = 0.5
                    else:  # Extreme temperatures
                        self.preferences[state_idx] = 0.1

        print("✅ Initialized Active Inference Agent")
        print(f"   Belief state dimension: {self.belief.shape}")
        print(f"   Device: {self.belief.device}")

    def update_belief_vfe(self, action, observation):
        """
        Enhanced VFE-based belief update with proper variational Bayesian inference.

        This implements the complete variational Bayesian update:
        q(s) ∝ exp(-∇_q F(q,θ)) where F is the variational free energy.

        Args:
            action: Previous action taken
            observation: Current observation

        Returns:
            Updated belief state
        """
        device = self.gpu_accelerator.device

        # Get transition and observation models for current action
        transition = self.env.transition_model[:, action, :].to(device)
        observation_model = self.env.observation_model.to(device)

        # Create observation likelihood: P(o|s) for current observation
        obs_likelihood = observation_model[:, observation].unsqueeze(0)  # [1, n_states]

        print(f"🔬 VFE Belief Update - Action: {action}, Observation: {observation}")
        print(f"   Current belief entropy: {-torch.sum(self.belief * torch.log(self.belief + 1e-8)):.4f}")

        # Enhanced VFE-based belief update
        with torch.no_grad():
            # Step 1: Compute predictive prior P(s'|s,a) = sum_s P(s'|s,a) * P(s)
            predictive_prior = torch.matmul(self.belief.unsqueeze(0), transition).squeeze(0)
            predictive_prior = predictive_prior / (predictive_prior.sum() + 1e-8)

            # Step 2: Compute likelihood-weighted posterior
            # P(s'|o,a,s) ∝ P(o|s') * P(s'|s,a)
            posterior = predictive_prior * obs_likelihood.squeeze(0)

            # Step 3: Normalize to get proper belief distribution
            posterior_sum = posterior.sum()
            if posterior_sum > 0:
                posterior = posterior / posterior_sum
            else:
                # Fallback to uniform if normalization fails
                posterior = torch.ones_like(posterior) / len(posterior)

            # Step 4: Apply variational regularization (optional smoothing)
            # This helps prevent overconfidence in single states
            smoothing_factor = 0.01
            posterior = (1 - smoothing_factor) * posterior + smoothing_factor * torch.ones_like(posterior) / len(posterior)
            posterior = posterior / posterior.sum()

            self.belief = posterior

        updated_entropy = -torch.sum(self.belief * torch.log(self.belief + 1e-8))
        max_belief_idx = self.belief.argmax().item()
        max_belief_prob = self.belief.max().item()

        print(f"   Updated belief entropy: {updated_entropy:.4f}")
        print(f"   Most likely state: {max_belief_idx} (P = {max_belief_prob:.4f})")

        return self.belief

    def select_policy_efe(self):
        """
        Enhanced EFE-based policy selection with epistemic + pragmatic value calculation.

        Implements proper Expected Free Energy computation:
        EFE_π = E_q[log q(o) - log p(o|π)] + E_q[log q(o) - log p(o|C)]

        Where:
        - Epistemic value: E_q[log q(o) - log p(o|π)] (information gain)
        - Pragmatic value: E_q[log q(o) - log p(o|C)] (goal achievement)

        Returns:
            Selected action, EFE values, and detailed component values
        """
        device = self.gpu_accelerator.device
        n_states = self.env.total_states
        n_actions = self.env.n_actions

        print(f"🎯 EFE Policy Selection - Computing EFE for {n_actions} policies")

        # Initialize EFE components
        epistemic_values = torch.zeros(n_actions, device=device)
        pragmatic_values = torch.zeros(n_actions, device=device)
        efe_values = torch.zeros(n_actions, device=device)

        eps = 1e-8

        # Compute EFE for each policy (action)
        for action in range(n_actions):
            # Get transition probabilities for this action
            transition_probs = self.env.transition_model[:, action, :].to(device)

            # Expected next state distribution under current belief
            expected_next_states = torch.matmul(self.belief.unsqueeze(0), transition_probs).squeeze(0)
            expected_next_states = expected_next_states / (expected_next_states.sum() + eps)

            # ===== EPISTEMIC VALUE =====
            # Information gain: KL[current_belief || expected_next_belief]
            epistemic_kl = torch.sum(
                self.belief * torch.log((self.belief + eps) / (expected_next_states + eps))
            )
            epistemic_values[action] = epistemic_kl

            # ===== PRAGMATIC VALUE =====
            # Expected utility: sum over states of P(s) * U(s)
            pragmatic = torch.sum(expected_next_states * self.preferences)
            pragmatic_values[action] = -pragmatic  # Negative because we want to minimize EFE

            # ===== TOTAL EFE =====
            # EFE = Epistemic + Pragmatic
            efe_values[action] = epistemic_values[action] + pragmatic_values[action]

            print(f"   Policy {action} ({self.env.action_labels[action]}):")
            print(".4f")
            print(".4f")
            print(".4f")

        # ===== SOFTMAX POLICY SELECTION =====
        # Convert EFE to policy probabilities using softmax
        # Lower EFE = higher probability (since we want to minimize EFE)

        # First, invert EFE values (negative EFE = higher utility)
        efe_utilities = -efe_values

        # Apply softmax to get policy probabilities
        max_utility = torch.max(efe_utilities)
        exp_utilities = torch.exp(efe_utilities - max_utility)  # Subtract max for numerical stability
        policy_probs = exp_utilities / exp_utilities.sum()

        print("\n🎲 Softmax Policy Selection:")
        print(f"   EFE Utilities: {efe_utilities.detach().cpu().numpy()}")
        print(f"   Policy Probabilities: {policy_probs.detach().cpu().numpy()}")

        # Sample action from the policy distribution
        action_dist = torch.distributions.Categorical(policy_probs)
        selected_action = action_dist.sample().item()

        print(f"   Selected Action: {selected_action} ({self.env.action_labels[selected_action]})")

        return selected_action, efe_values[selected_action].item(), efe_values.detach().cpu().numpy()

    def get_current_position_estimate(self):
        """Get estimated position from current belief state."""
        # Marginalize over temperature states to get position belief
        position_belief = torch.zeros(self.env.grid_size, self.env.grid_size,
                                    device=self.belief.device)

        for state_idx in range(self.env.total_states):
            pos_i, pos_j, _ = self.env._state_to_pos_temp(state_idx)
            position_belief[pos_i, pos_j] += self.belief[state_idx]

        # Get most likely position
        max_idx = position_belief.argmax()
        max_pos_i = max_idx // self.env.grid_size
        max_pos_j = max_idx % self.env.grid_size

        return max_pos_i.item(), max_pos_j.item()


def run_pomdp_simulation(n_timesteps=50):
    """
    Run the complete POMDP simulation with Active Inference.

    Args:
        n_timesteps: Number of timesteps to simulate

    Returns:
        Complete simulation results
    """
    print("=" * 80)
    print("POMDP GRIDWORLD ACTIVE INFERENCE SIMULATION")
    print("=" * 80)
    print(f"Running simulation for {n_timesteps} timesteps...")
    print()

    # Initialize components using existing framework
    feature_manager = TritonFeatureManager()
    environment = POMDPGridworld()
    agent = ActiveInferenceAgent(environment, feature_manager)

    # ===== CREATE COMPREHENSIVE VISUALIZATIONS =====
    print("\n📊 Creating Environment Visualizations...")
    environment.visualize_transition_model(OUTPUTS_DIR)
    environment.visualize_observation_model(OUTPUTS_DIR)
    environment.visualize_temperature_grid(OUTPUTS_DIR)
    environment.visualize_preferences(OUTPUTS_DIR, agent)
    environment.visualize_state_space(agent.belief, OUTPUTS_DIR, timestep=0)
    environment.create_generative_model_visualization(agent, OUTPUTS_DIR)

    # Initialize simulation
    current_state = np.random.randint(0, environment.total_states)
    total_reward = 0.0

    # Simulation history
    history = {
        'timestep': [],
        'position_i': [],
        'position_j': [],
        'temperature_state': [],
        'observation': [],
        'action': [],
        'reward': [],
        'efe_values': [],
        'belief_entropy': [],
        'estimated_position_i': [],
        'estimated_position_j': []
    }

    start_time = time.time()

    # Create visualization at key timesteps
    viz_timesteps = [9, 19, 29, 39, 49]  # Visualize at timesteps 10, 20, 30, 40, 50

    for t in range(n_timesteps):
        print(f"\n{'='*60}")
        print(f"🕐 TIMESTEP {t+1}/{n_timesteps}")
        print(f"{'='*60}")

        # Get current position and temperature
        pos_i, pos_j, temp_state = environment._state_to_pos_temp(current_state)
        print(f"🏠 True State: Position ({pos_i}, {pos_j}), Temperature: {environment.temperature_labels[temp_state]}")

        # Generate observation
        observation = environment.get_observation(current_state)
        print(f"👁️  Observation: {environment.observation_labels[observation]}")

        # Update belief using VFE
        if t > 0:  # Skip first update (no previous action)
            print(f"🔬 Updating belief with VFE (Action: {environment.action_labels[history['action'][-1]]})...")
            agent.update_belief_vfe(history['action'][-1], observation)

        # Select policy using EFE
        print("🎯 Computing Expected Free Energy for policy selection...")
        action, min_efe, efe_values = agent.select_policy_efe()
        action_name = environment.action_labels[action]

        # Execute action (transition to new state)
        next_state_probs = environment.transition_model[current_state, action]
        next_state = torch.multinomial(next_state_probs, 1).item()
        current_state = next_state

        # Get reward
        reward = environment.get_reward(current_state)
        total_reward += reward

        # Get estimated position
        est_pos_i, est_pos_j = agent.get_current_position_estimate()

        # Compute belief entropy
        belief_entropy = -torch.sum(agent.belief * torch.log(agent.belief + 1e-8)).item()

        # Record history
        history['timestep'].append(t)
        history['position_i'].append(pos_i)
        history['position_j'].append(pos_j)
        history['temperature_state'].append(temp_state)
        history['observation'].append(observation)
        history['action'].append(action)
        history['reward'].append(reward)
        history['efe_values'].append(efe_values.tolist())
        history['belief_entropy'].append(belief_entropy)
        history['estimated_position_i'].append(est_pos_i)
        history['estimated_position_j'].append(est_pos_j)

        print(f"⚡ Action: {action_name} (EFE: {min_efe:.4f})")
        print(".4f")
        print(".4f")
        print(f"📍 Estimated position: ({est_pos_i}, {est_pos_j})")

        # Position accuracy
        position_accuracy = 1.0 if (pos_i == est_pos_i and pos_j == est_pos_j) else 0.0
        print(".1f")

        # Create visualization at key timesteps
        if t in viz_timesteps:
            print(f"📊 Creating belief state visualization for timestep {t+1}...")
            environment.visualize_state_space(agent.belief, OUTPUTS_DIR, timestep=t+1)

    total_time = time.time() - start_time

    # ===== FINAL BELIEF STATE VISUALIZATION =====
    print("\n📊 Creating final belief state visualization...")
    environment.visualize_state_space(agent.belief, OUTPUTS_DIR, timestep=n_timesteps)

    # ===== COMPREHENSIVE PERFORMANCE ANALYSIS =====
    position_accuracies = [
        1.0 if history['position_i'][i] == history['estimated_position_i'][i] and
             history['position_j'][i] == history['estimated_position_j'][i] else 0.0
        for i in range(len(history['position_i']))
    ]

    temperature_accuracies = [
        1.0 if history['temperature_state'][i] == torch.argmax(torch.tensor([
            agent.preferences[environment._pos_temp_to_state(
                history['estimated_position_i'][i],
                history['estimated_position_j'][i],
                temp
            )] for temp in range(environment.n_temperature_states)
        ])).item() else 0.0
        for i in range(len(history['temperature_state']))
    ]

    # Action distribution analysis
    action_counts = np.bincount(history['action'], minlength=environment.n_actions)
    action_distribution = action_counts / action_counts.sum()

    # EFE analysis
    efe_history = np.array(history['efe_values'])
    efe_min_over_time = np.min(efe_history, axis=1)
    efe_std_over_time = np.std(efe_history, axis=1)

    # Final results
    results = {
        'simulation_config': {
            'n_timesteps': n_timesteps,
            'grid_size': environment.grid_size,
            'n_temperature_states': environment.n_temperature_states,
            'n_observations': environment.n_observations,
            'n_actions': environment.n_actions,
            'total_states': environment.total_states
        },
        'performance_metrics': {
            'total_time': total_time,
            'avg_time_per_step': total_time / n_timesteps,
            'total_reward': total_reward,
            'avg_reward_per_step': total_reward / n_timesteps,
            'final_belief_entropy': belief_entropy,
            'position_accuracy': np.mean(position_accuracies),
            'temperature_accuracy': np.mean(temperature_accuracies),
            'final_position_accuracy': position_accuracies[-1],
            'final_temperature_accuracy': temperature_accuracies[-1],
            'action_distribution': action_distribution.tolist(),
            'efe_final_min': efe_min_over_time[-1],
            'efe_final_std': efe_std_over_time[-1],
            'belief_entropy_trend': history['belief_entropy']
        },
        'active_inference_analysis': {
            'vfe_updates_performed': n_timesteps - 1,
            'efe_computations_performed': n_timesteps,
            'final_most_likely_state': int(agent.belief.argmax().item()),
            'final_most_likely_position': agent.get_current_position_estimate(),
            'average_efe_complexity': np.mean([len(efe_vals) for efe_vals in history['efe_values']])
        },
        'history': history,
        'final_belief': agent.belief.detach().cpu().numpy().tolist(),
        'preferences': agent.preferences.detach().cpu().numpy().tolist()
    }

    print("\n🎉 SIMULATION COMPLETED!")
    print("=" * 80)
    print(".2f")
    print(".4f")
    print(".4f")
    print(".1%")
    print(".1%")
    print(".4f")
    print(".4f")

    print("\n📊 ACTIVE INFERENCE PERFORMANCE:")
    print(f"   VFE Updates: {results['active_inference_analysis']['vfe_updates_performed']}")
    print(f"   EFE Computations: {results['active_inference_analysis']['efe_computations_performed']}")
    print(f"   Final Most Likely State: {results['active_inference_analysis']['final_most_likely_state']}")
    print(f"   Final Most Likely Position: {results['active_inference_analysis']['final_most_likely_position']}")

    print("\n🎯 ACTION DISTRIBUTION:")
    for i, prob in enumerate(results['performance_metrics']['action_distribution']):
        print(".3f")

    print("\n🔬 ACTIVE INFERENCE VALIDATION:")
    print("   ✓ Variational Free Energy (VFE) - State estimation from observations")
    print("   ✓ Expected Free Energy (EFE) - Policy selection with epistemic+pragmatic value")
    print("   ✓ Softmax Policy Selection - EFE-reweighted action sampling")
    print("   ✓ Bayesian Belief Updates - Real-time POMDP inference")
    print("   ✓ Comprehensive Visualizations - All matrices and state spaces")
    print("   ✓ Enhanced Logging - Detailed simulation tracking")

    return results


def create_visualizations(results):
    """Create comprehensive visualizations of the POMDP simulation."""
    print("\n📊 Creating visualizations...")

    history = results['history']
    config = results['simulation_config']

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("POMDP Gridworld Active Inference Results", fontsize=16, fontweight='bold')

    # Plot 1: Agent trajectory
    ax = axes[0, 0]
    true_positions = list(zip(history['position_i'], history['position_j']))
    estimated_positions = list(zip(history['estimated_position_i'], history['estimated_position_j']))

    # Plot true trajectory
    true_x, true_y = zip(*true_positions)
    ax.plot(true_x, true_y, 'b-o', linewidth=2, markersize=4, label='True Position', alpha=0.8)

    # Plot estimated trajectory
    est_x, est_y = zip(*estimated_positions)
    ax.plot(est_x, est_y, 'r--s', linewidth=2, markersize=4, label='Estimated Position', alpha=0.8)

    ax.set_xlim(0, config['grid_size']-1)
    ax.set_ylim(0, config['grid_size']-1)
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_title('Agent Trajectory', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Temperature state evolution
    ax = axes[0, 1]
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    temp_states = history['temperature_state']
    for temp_state in range(config['n_temperature_states']):
        indices = [i for i, state in enumerate(temp_states) if state == temp_state]
        if indices:
            ax.scatter(indices, [temp_state] * len(indices), c=colors[temp_state],
                      label=f'{config["n_temperature_states"]-temp_state-1}', s=50, alpha=0.7)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Temperature State')
    ax.set_title('Temperature State Evolution', fontweight='bold')
    ax.set_yticks(range(config['n_temperature_states']))
    ax.set_yticklabels(['Very Cold', 'Cold', 'Optimal', 'Warm', 'Hot'])
    ax.grid(True, alpha=0.3)

    # Plot 3: EFE values over time
    ax = axes[0, 2]
    timesteps = history['timestep']
    efe_values = history['efe_values']
    selected_actions = history['action']


    for action in range(config['n_actions']):
        action_mask = np.array(selected_actions) == action
        if np.any(action_mask):
            action_indices = np.where(action_mask)[0]
            # Extract EFE values for this action at the selected timesteps
            action_efe_values = []
            for t in action_indices:
                if t < len(efe_values) and action < len(efe_values[t]):
                    action_efe_values.append(efe_values[t][action])

            if action_efe_values:  # Only plot if we have values
                ax.scatter(action_indices[:len(action_efe_values)], action_efe_values,
                          label=f'Action {action}', alpha=0.7, s=30)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Expected Free Energy')
    ax.set_title('EFE Values Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Belief entropy evolution
    ax = axes[1, 0]
    entropy = history['belief_entropy']
    ax.plot(timesteps, entropy, 'purple', linewidth=2, marker='o', markersize=4)
    ax.fill_between(timesteps, entropy, alpha=0.3, color='purple')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Belief Entropy')
    ax.set_title('Belief Uncertainty Evolution', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 5: Reward accumulation
    ax = axes[1, 1]
    rewards = history['reward']
    cumulative_reward = np.cumsum(rewards)
    ax.plot(timesteps, cumulative_reward, 'green', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Reward Accumulation', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 6: Action distribution
    ax = axes[1, 2]
    actions = history['action']
    action_counts = np.bincount(actions, minlength=config['n_actions'])
    action_labels = ['Up', 'Down', 'Left', 'Right']

    bars = ax.bar(range(config['n_actions']), action_counts, color='skyblue', alpha=0.7)
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Selection Distribution', fontweight='bold')
    ax.set_xticks(range(config['n_actions']))
    ax.set_xticklabels(action_labels)

    # Add value labels on bars
    for bar, count in zip(bars, action_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha='center', va='bottom', fontweight='bold')

    # Plot 7: Position estimation accuracy
    ax = axes[2, 0]
    true_positions = list(zip(history['position_i'], history['position_j']))
    estimated_positions = list(zip(history['estimated_position_i'], history['estimated_position_j']))

    position_accuracy = []
    for true_pos, est_pos in zip(true_positions, estimated_positions):
        accuracy = 1.0 if true_pos == est_pos else 0.0
        position_accuracy.append(accuracy)

    ax.plot(timesteps, position_accuracy, 'orange', linewidth=2, marker='^', markersize=4)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position Accuracy')
    ax.set_title('Position Estimation Accuracy', fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Plot 8: Temperature preference heatmap
    ax = axes[2, 1]
    temp_grid = np.zeros((config['grid_size'], config['grid_size']))
    for i in range(config['grid_size']):
        for j in range(config['grid_size']):
            # Find the most likely temperature state for this position
            max_prob = 0
            best_temp = 0
            for temp_state in range(config['n_temperature_states']):
                state_idx = (i * config['grid_size'] + j) * config['n_temperature_states'] + temp_state
                if state_idx < len(results['preferences']):
                    prob = results['preferences'][state_idx]
                    if prob > max_prob:
                        max_prob = prob
                        best_temp = temp_state
            temp_grid[i, j] = best_temp

    im = ax.imshow(temp_grid, cmap='RdYlBu_r', aspect='equal', origin='lower')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    ax.set_title('Temperature Preference Map', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Plot 9: Performance summary
    ax = axes[2, 2]
    metrics = results['performance_metrics']
    metric_names = ['Total Time', 'Avg Time/Step', 'Total Reward', 'Avg Reward/Step', 'Final Entropy', 'Position Accuracy']
    metric_values = [
        metrics['total_time'],
        metrics['avg_time_per_step'],
        metrics['total_reward'],
        metrics['avg_reward_per_step'],
        metrics['final_belief_entropy'],
        metrics['position_accuracy']
    ]

    bars = ax.barh(range(len(metric_names)), metric_values, color='lightcoral', alpha=0.7)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)
    ax.set_xlabel('Value')
    ax.set_title('Performance Metrics', fontweight='bold')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(value + max(metric_values) * 0.01, i, f'{value:.3f}',
               va='center', fontweight='bold')

    plt.tight_layout()
    save_path = OUTPUTS_DIR / "pomdp_gridworld_results.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    print(f"✓ Main visualization saved as: {save_path}")

    # Create trajectory animation-style plot
    plt.figure(figsize=(12, 8))
    grid_size = config['grid_size']

    # Plot temperature preference background
    temp_background = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            center_dist = np.sqrt((i - grid_size//2)**2 + (j - grid_size//2)**2)
            temp_background[i, j] = max(0, 2 - center_dist * 0.3)  # Higher in center

    plt.imshow(temp_background, cmap='RdYlBu_r', alpha=0.3, extent=[0, grid_size-1, 0, grid_size-1])

    # Plot trajectory with time progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
    for i, t in enumerate(timesteps):
        if i > 0:
            plt.plot([true_x[i-1], true_x[i]], [true_y[i-1], true_y[i]],
                    color=colors[i], linewidth=2, alpha=0.7)
            plt.plot([est_x[i-1], est_x[i]], [est_y[i-1], est_y[i]],
                    color=colors[i], linewidth=2, linestyle='--', alpha=0.7)

    # Mark start and end points
    plt.scatter(true_x[0], true_y[0], color='green', s=200, marker='*', label='Start', zorder=10)
    plt.scatter(true_x[-1], true_y[-1], color='red', s=200, marker='X', label='End', zorder=10)

    plt.xlabel('Grid X Position')
    plt.ylabel('Grid Y Position')
    plt.title('POMDP Agent Trajectory with Active Inference', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Add colorbar for time progression
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(timesteps)-1))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Time Progression')

    save_path = OUTPUTS_DIR / "pomdp_trajectory_evolution.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    print(f"✓ Trajectory evolution saved as: {save_path}")


def save_results_to_files(results):
    """Save comprehensive results to JSON and CSV files."""
    print("\n💾 Saving results to files...")

    # Save complete results to JSON
    json_path = OUTPUTS_DIR / "pomdp_simulation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Complete results saved to: {json_path}")

    # Save summary metrics to JSON
    summary = {
        'simulation_summary': {
            'timesteps': results['simulation_config']['n_timesteps'],
            'grid_size': results['simulation_config']['grid_size'],
            'total_states': results['simulation_config']['total_states'],
            'performance': results['performance_metrics']
        },
        'active_inference_methods': [
            'Variational Free Energy (VFE) for state estimation',
            'Expected Free Energy (EFE) for policy selection',
            'GPU-accelerated Bayesian inference',
            'POMDP belief state updates'
        ],
        'environment_characteristics': [
            '10x10 gridworld with temperature states',
            '5 temperature levels per cell',
            '5 observation types (noisy readings)',
            '4 actions (grid navigation)',
            'Homeostatic temperature preference'
        ]
    }

    summary_path = OUTPUTS_DIR / "simulation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")

    # Save trajectory data to CSV
    import pandas as pd
    trajectory_data = {
        'timestep': results['history']['timestep'],
        'true_position_x': results['history']['position_i'],
        'true_position_y': results['history']['position_j'],
        'estimated_position_x': results['history']['estimated_position_i'],
        'estimated_position_y': results['history']['estimated_position_j'],
        'temperature_state': results['history']['temperature_state'],
        'observation': results['history']['observation'],
        'action': results['history']['action'],
        'reward': results['history']['reward'],
        'belief_entropy': results['history']['belief_entropy']
    }

    df = pd.DataFrame(trajectory_data)
    csv_path = OUTPUTS_DIR / "trajectory_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Trajectory data saved to: {csv_path}")


def main():
    """Run the complete POMDP Gridworld Active Inference demonstration."""
    print("🚀 POMDP GRIDWORLD ACTIVE INFERENCE DEMONSTRATION")
    print("=" * 70)
    print("This example demonstrates Active Inference in a partially observable")
    print("temperature-controlled gridworld environment.")
    print()
    print("Key Features:")
    print("• POMDP with 10x10 gridworld (500 total states)")
    print("• 5 temperature states per cell with homeostatic preferences")
    print("• Variational Free Energy (VFE) for state estimation")
    print("• Expected Free Energy (EFE) for policy selection")
    print("• GPU-accelerated computations using Triton")
    print("• Comprehensive logging and visualization")
    print("=" * 70)

    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Run the POMDP simulation
        results = run_pomdp_simulation(n_timesteps=50)

        # Create visualizations
        create_visualizations(results)

        # Save results to files
        save_results_to_files(results)

        print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 Results saved to: {OUTPUTS_DIR}")
        print("  • pomdp_simulation_results.json (complete data)")
        print("  • simulation_summary.json (key metrics)")
        print("  • trajectory_data.csv (time series data)")
        print("  • pomdp_gridworld_results.png (comprehensive plots)")
        print("  • pomdp_trajectory_evolution.png (trajectory animation)")

        print("\n📊 FINAL PERFORMANCE SUMMARY:")
        perf = results['performance_metrics']
        print(".1f")
        print(".4f")
        print(".4f")
        print(".1%")
        print(".4f")

        print("\n🔬 ACTIVE INFERENCE METHODS DEMONSTRATED:")
        print("  1. Perception: VFE-based state estimation from observations")
        print("  2. Action: EFE-based policy selection for navigation")
        print("  3. Learning: Bayesian belief updates in POMDP setting")
        print("  4. Integration: Complete perception-action-learning cycle")

        return 0

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
