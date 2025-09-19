#!/usr/bin/env python3
"""
POMDP Gridworld Environment

This module provides a comprehensive POMDP gridworld environment for Active Inference
demonstrations. The environment models temperature states across a grid with partial
observability and provides all necessary methods for generative model specification.

Key Features:
- 10x10 grid with 5 temperature states per cell (500 total states)
- Partially observable temperature readings
- Movement dynamics with environmental temperature changes
- Comprehensive visualization methods
- JSON export of complete generative model matrices

Classes:
    POMDPGridworld: Main environment class with full generative model specification
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


class POMDPGridworld:
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
                        trans_probs = torch.zeros(self.n_temperature_states)
                        for next_temp in range(self.n_temperature_states):
                            # Movement probability (deterministic position, probabilistic temperature)
                            if next_temp == temp_state:
                                trans_probs[next_temp] = 0.8  # Same temperature
                            elif abs(next_temp - temp_state) == 1:
                                trans_probs[next_temp] = 0.1  # Adjacent temperature
                            else:
                                trans_probs[next_temp] = 0.0  # Non-adjacent temperature change

                        # Normalize transition probabilities
                        trans_probs_sum = trans_probs.sum()
                        if trans_probs_sum > 0:
                            trans_probs = trans_probs / trans_probs_sum

                        # Set transition probabilities for all next states with this position
                        for next_temp in range(self.n_temperature_states):
                            next_state_idx = self._pos_temp_to_state(next_pos_i, next_pos_j, next_temp)
                            self.transition_model[state_idx, action, next_state_idx] = trans_probs[next_temp]

                    # Observation model (noisy temperature readings)
                    obs_probs = torch.zeros(self.n_observations)
                    for obs in range(self.n_observations):
                        if obs == temp_state:
                            obs_probs[obs] = 0.8  # Correct reading
                        elif abs(obs - temp_state) == 1:
                            obs_probs[obs] = 0.1  # Adjacent temperature reading
                        else:
                            obs_probs[obs] = 0.0  # Incorrect reading

                    # Normalize observation probabilities
                    obs_probs_sum = obs_probs.sum()
                    if obs_probs_sum > 0:
                        obs_probs = obs_probs / obs_probs_sum

                    self.observation_model[state_idx] = obs_probs

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
        plt.text(0.02, 0.98, f'Entropy: {entropy:.4f}',
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
        perf_text = f'Entropy: {generative_model_data["performance_metrics"]["belief_entropy"]:.4f}\\nStates: {self.total_states}'
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


# Export the class for external use
__all__ = ['POMDPGridworld']
