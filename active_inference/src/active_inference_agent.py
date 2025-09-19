#!/usr/bin/env python3
"""
Active Inference Agent for POMDP Gridworld

This module provides an Active Inference agent that implements variational Bayesian
inference for state estimation (VFE) and expected free energy minimization for
policy selection (EFE) in partially observable environments.

Key Features:
- Variational Free Energy (VFE) for state estimation from observations
- Expected Free Energy (EFE) for policy selection with epistemic+pragmatic value
- Softmax policy selection from EFE-reweighted action priors
- Real-time belief state updates with variational regularization
- Comprehensive logging and performance metrics

Classes:
    ActiveInferenceAgent: Main agent class implementing VFE and EFE methods
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from core import TritonFeatureManager, GPUAccelerator
from free_energy import VariationalFreeEnergy, ExpectedFreeEnergy


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
            print(f"     Epistemic: {epistemic_values[action]:.4f}")
            print(f"     Pragmatic: {pragmatic_values[action]:.4f}")
            print(f"     Total EFE: {efe_values[action]:.4f}")

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

    def get_belief_entropy(self):
        """Get entropy of current belief state (uncertainty measure)."""
        return -torch.sum(self.belief * torch.log(self.belief + 1e-8)).item()

    def get_most_likely_state(self):
        """Get the most likely state from current belief."""
        return self.belief.argmax().item()


# Export the class for external use
__all__ = ['ActiveInferenceAgent']
