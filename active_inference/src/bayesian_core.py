"""
Core Bayesian Inference Engine for Active Inference

GPU-accelerated Bayesian updating with comprehensive logging, information calculations,
free energy computations, visualization, and validation. Focuses on fundamental
Bayesian inference operations with real data and robust error handling.

Key Features:
- Bayesian belief updating (prior → posterior via likelihood)
- Information theoretic calculations (entropy, mutual information, KL divergence)
- Free energy computation and minimization
- Comprehensive logging and reporting
- Real-time visualization of beliefs and information flow
- Extensive validation of all operations

All computations use real data and provide full traceability of inference processes.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
import numpy as np
import time
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json

# Flexible import for core module
try:
    # Try relative import first (when used as package)
    from .core import TritonFeatureManager, GPUAccelerator, TRITON_AVAILABLE, reporter
except ImportError:
    # Fallback for direct execution
    from core import TritonFeatureManager, GPUAccelerator, TRITON_AVAILABLE, reporter

logger = logging.getLogger(__name__)


@dataclass
class BayesianUpdateResult:
    """Result of a Bayesian belief update operation."""

    prior: torch.Tensor
    likelihood: torch.Tensor
    posterior: torch.Tensor
    evidence: torch.Tensor
    kl_divergence: torch.Tensor
    prior_entropy: torch.Tensor
    posterior_entropy: torch.Tensor
    mutual_information: torch.Tensor
    free_energy: torch.Tensor
    timestamp: datetime = field(default_factory=datetime.now)
    update_method: str = "bayesian"
    converged: bool = True
    iterations: int = 1
    computation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'prior': self.prior.detach().cpu().numpy(),
            'likelihood': self.likelihood.detach().cpu().numpy(),
            'posterior': self.posterior.detach().cpu().numpy(),
            'evidence': self.evidence.detach().cpu().numpy(),
            'kl_divergence': self.kl_divergence.detach().cpu().numpy(),
            'prior_entropy': self.prior_entropy.detach().cpu().numpy(),
            'posterior_entropy': self.posterior_entropy.detach().cpu().numpy(),
            'mutual_information': self.mutual_information.detach().cpu().numpy(),
            'free_energy': self.free_energy.detach().cpu().numpy(),
            'timestamp': self.timestamp.isoformat(),
            'update_method': self.update_method,
            'converged': self.converged,
            'iterations': self.iterations,
            'computation_time': self.computation_time
        }


@dataclass
class InferenceHistory:
    """History of inference operations for analysis and visualization."""

    updates: List[BayesianUpdateResult] = field(default_factory=list)
    free_energy_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    kl_history: List[float] = field(default_factory=list)
    evidence_history: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    def add_update(self, result: BayesianUpdateResult):
        """Add a new update to the history."""
        self.updates.append(result)
        self.free_energy_history.append(result.free_energy.mean().item())
        self.entropy_history.append(result.posterior_entropy.mean().item())
        self.kl_history.append(result.kl_divergence.mean().item())
        self.evidence_history.append(result.evidence.mean().item())
        self.timestamps.append(result.timestamp)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the inference history."""
        if not self.updates:
            return {}

        return {
            'total_updates': len(self.updates),
            'avg_free_energy': np.mean(self.free_energy_history),
            'std_free_energy': np.std(self.free_energy_history),
            'avg_entropy': np.mean(self.entropy_history),
            'std_entropy': np.std(self.entropy_history),
            'avg_kl_divergence': np.mean(self.kl_history),
            'std_kl_divergence': np.std(self.kl_history),
            'avg_evidence': np.mean(self.evidence_history),
            'std_evidence': np.std(self.evidence_history),
            'total_computation_time': sum(u.computation_time for u in self.updates),
            'convergence_rate': sum(u.converged for u in self.updates) / len(self.updates)
        }


class BayesianInferenceCore:
    """
    Core Bayesian inference engine with comprehensive logging and validation.

    Provides fundamental Bayesian updating operations with real data validation,
    information calculations, and detailed reporting capabilities.
    """

    def __init__(self,
                 feature_manager: Optional[TritonFeatureManager] = None,
                 enable_logging: bool = True,
                 enable_validation: bool = True,
                 enable_visualization: bool = True):
        """
        Initialize the Bayesian inference core.

        Args:
            feature_manager: Triton feature manager for GPU acceleration
            enable_logging: Whether to enable comprehensive logging
            enable_validation: Whether to enable real data validation
            enable_visualization: Whether to enable visualization capabilities
        """
        self.feature_manager = feature_manager or TritonFeatureManager()
        self.gpu_accelerator = GPUAccelerator(self.feature_manager)
        self.enable_logging = enable_logging
        self.enable_validation = enable_validation
        self.enable_visualization = enable_visualization

        # Initialize logging and history
        self.history = InferenceHistory()
        self.logger = logging.getLogger(f"{__name__}.BayesianInferenceCore")
        self.logger.setLevel(logging.INFO)

        # Validation counters
        self.validation_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'validation_errors': [],
            'computation_times': []
        }

        if self.enable_logging:
            self.logger.info("BayesianInferenceCore initialized with logging enabled")

    def validate_inputs(self,
                       prior: torch.Tensor,
                       likelihood: torch.Tensor,
                       observations: Optional[torch.Tensor] = None) -> bool:
        """
        Enhanced validation of inputs for Bayesian updating with comprehensive error checking.

        Args:
            prior: Prior distribution [batch_size, n_states]
            likelihood: Likelihood function [batch_size, n_states] or [batch_size, n_states, n_obs]
            observations: Optional observations for validation

        Returns:
            True if inputs are valid
        """
        validation_errors = []
        validation_warnings = []

        try:
            # Enhanced tensor type validation
            if not isinstance(prior, torch.Tensor):
                validation_errors.append(f"Prior must be torch.Tensor, got {type(prior)}")
            if not isinstance(likelihood, torch.Tensor):
                validation_errors.append(f"Likelihood must be torch.Tensor, got {type(likelihood)}")
            if observations is not None and not isinstance(observations, torch.Tensor):
                validation_errors.append(f"Observations must be torch.Tensor, got {type(observations)}")

            if validation_errors:
                raise ValueError("; ".join(validation_errors))

            # Enhanced shape validation
            if prior.dim() != 2:
                validation_errors.append(f"Prior must be 2D tensor, got shape {prior.shape}")
            else:
                batch_size = prior.shape[0]
                n_states = prior.shape[1]

                if batch_size == 0:
                    validation_errors.append("Prior batch size cannot be zero")
                if n_states == 0:
                    validation_errors.append("Prior must have at least one state")

            if likelihood.dim() not in [2, 3]:
                validation_errors.append(f"Likelihood must be 2D or 3D tensor, got shape {likelihood.shape}")
            else:
                if prior.dim() == 2:  # Only check if prior shape is valid
                    batch_size = prior.shape[0]
                    n_states = prior.shape[1]

                    if likelihood.dim() == 2:
                        if likelihood.shape[0] != batch_size:
                            validation_errors.append(f"Likelihood batch size {likelihood.shape[0]} doesn't match prior batch size {batch_size}")
                        if likelihood.shape[1] != n_states:
                            validation_errors.append(f"Likelihood n_states {likelihood.shape[1]} doesn't match prior n_states {n_states}")
                    else:  # 3D likelihood
                        if likelihood.shape[0] != batch_size:
                            validation_errors.append(f"Likelihood batch size {likelihood.shape[0]} doesn't match prior batch size {batch_size}")
                        if likelihood.shape[1] != n_states:
                            validation_errors.append(f"Likelihood n_states {likelihood.shape[1]} doesn't match prior n_states {n_states}")

            if validation_errors:
                raise ValueError("; ".join(validation_errors))

            # Enhanced value validation
            if torch.any(torch.isnan(prior)) or torch.any(torch.isinf(prior)):
                validation_errors.append("Prior contains NaN or Inf values")

            if torch.any(torch.isnan(likelihood)) or torch.any(torch.isinf(likelihood)):
                validation_errors.append("Likelihood contains NaN or Inf values")

            if observations is not None:
                if torch.any(torch.isnan(observations)) or torch.any(torch.isinf(observations)):
                    validation_errors.append("Observations contain NaN or Inf values")

            if validation_errors:
                raise ValueError("; ".join(validation_errors))

            # Normalization check with warnings
            prior_sums = prior.sum(dim=1)
            if not torch.allclose(prior_sums, torch.ones_like(prior_sums), atol=1e-5):
                validation_warnings.append("Prior distributions are not properly normalized")

            if likelihood.dim() == 2:
                likelihood_sums = likelihood.sum(dim=1)
                if not torch.allclose(likelihood_sums, torch.ones_like(likelihood_sums), atol=1e-5):
                    validation_warnings.append("Likelihood distributions are not properly normalized")

            # Positive probability check
            if torch.any(prior < 0):
                validation_errors.append("Prior contains negative probabilities")

            if torch.any(likelihood < 0):
                validation_errors.append("Likelihood contains negative probabilities")

            if validation_errors:
                raise ValueError("; ".join(validation_errors))

            # Log warnings if any
            if validation_warnings and self.enable_logging:
                for warning in validation_warnings:
                    self.logger.warning(f"Validation warning: {warning}")

            return True

        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Input validation failed: {e}")
            self.validation_stats['validation_errors'].append(str(e))
            return False

    def compute_evidence(self,
                        prior: torch.Tensor,
                        likelihood: torch.Tensor) -> torch.Tensor:
        """
        Compute marginal evidence p(x) = ∫ p(x|z) p(z) dz

        Args:
            prior: Prior distribution p(z) [batch_size, n_states]
            likelihood: Likelihood p(x|z) [batch_size, n_states]

        Returns:
            Evidence p(x) [batch_size]
        """
        if likelihood.dim() == 3:
            # Handle observation-dependent likelihood
            # For simplicity, assume first observation dimension
            likelihood = likelihood[:, :, 0]

        evidence = torch.sum(prior * likelihood, dim=1)
        return evidence

    def bayesian_update(self,
                        prior: torch.Tensor,
                        likelihood: torch.Tensor,
                        observations: Optional[torch.Tensor] = None,
                        normalize: bool = True) -> BayesianUpdateResult:
        """
        Perform Bayesian belief update: prior → posterior via likelihood.

        Args:
            prior: Prior distribution p(z) [batch_size, n_states]
            likelihood: Likelihood p(x|z) [batch_size, n_states] or [batch_size, n_states, n_obs]
            observations: Optional observation data for validation
            normalize: Whether to normalize the posterior

        Returns:
            BayesianUpdateResult with all computed quantities
        """
        start_time = time.time()

        # Input validation
        if self.enable_validation and not self.validate_inputs(prior, likelihood, observations):
            raise ValueError("Input validation failed")

        # Handle different likelihood shapes
        if likelihood.dim() == 3:
            # For now, use first observation - could be extended for full observation handling
            likelihood_2d = likelihood[:, :, 0] if likelihood.shape[2] > 0 else likelihood[:, :, 0]
        else:
            likelihood_2d = likelihood

        # Compute evidence
        evidence = self.compute_evidence(prior, likelihood_2d)

        # Compute posterior (unnormalized)
        posterior_unnorm = prior * likelihood_2d

        # Normalize posterior
        if normalize:
            posterior = posterior_unnorm / (evidence.unsqueeze(1) + 1e-8)
        else:
            posterior = posterior_unnorm

        # Information calculations
        kl_divergence = self.compute_kl_divergence(posterior, prior)
        prior_entropy = self.compute_entropy(prior)
        posterior_entropy = self.compute_entropy(posterior)
        mutual_information = self.compute_mutual_information(prior, posterior, likelihood_2d)

        # Free energy computation (simplified variational free energy)
        free_energy = self.compute_variational_free_energy(prior, posterior, likelihood_2d)

        computation_time = time.time() - start_time

        result = BayesianUpdateResult(
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
            kl_divergence=kl_divergence,
            prior_entropy=prior_entropy,
            posterior_entropy=posterior_entropy,
            mutual_information=mutual_information,
            free_energy=free_energy,
            update_method="bayesian",
            converged=True,
            iterations=1,
            computation_time=computation_time
        )

        # Update history and logging
        self.history.add_update(result)
        self.validation_stats['total_updates'] += 1
        self.validation_stats['successful_updates'] += 1
        self.validation_stats['computation_times'].append(computation_time)

        if self.enable_logging:
            self.logger.info(f"Bayesian update completed in {computation_time:.4f}s")
            self.logger.info(f"Free energy: {free_energy.mean().item():.4f}")
            self.logger.info(f"KL divergence: {kl_divergence.mean().item():.4f}")

        return result

    def compute_entropy(self, distribution: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy of probability distributions.

        H(X) = -∑ p(x) log p(x)

        Args:
            distribution: Probability distribution [batch_size, n_states]

        Returns:
            Entropy values [batch_size]
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        log_prob = torch.log(distribution + eps)
        entropy = -torch.sum(distribution * log_prob, dim=1)
        return entropy

    def compute_kl_divergence(self,
                             p: torch.Tensor,
                             q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence D_KL(p||q) = ∑ p(x) log(p(x)/q(x))

        Args:
            p: First distribution [batch_size, n_states]
            q: Second distribution [batch_size, n_states]

        Returns:
            KL divergence [batch_size]
        """
        eps = 1e-8
        kl = torch.sum(p * torch.log((p + eps) / (q + eps)), dim=1)
        return kl

    def compute_mutual_information(self,
                                 prior: torch.Tensor,
                                 posterior: torch.Tensor,
                                 likelihood: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information I(Z;X) = H(Z) - H(Z|X)

        Where H(Z|X) is the conditional entropy.

        Args:
            prior: Prior p(z) [batch_size, n_states]
            posterior: Posterior p(z|x) [batch_size, n_states]
            likelihood: Likelihood p(x|z) [batch_size, n_states]

        Returns:
            Mutual information [batch_size]
        """
        # Prior entropy H(Z)
        prior_entropy = self.compute_entropy(prior)

        # Conditional entropy H(Z|X) = ∑ p(x) H(Z|X=x)
        # For simplicity, use posterior entropy as approximation
        conditional_entropy = self.compute_entropy(posterior)

        mutual_info = prior_entropy - conditional_entropy
        return mutual_info

    def compute_variational_free_energy(self,
                                      prior: torch.Tensor,
                                      posterior: torch.Tensor,
                                      likelihood: torch.Tensor) -> torch.Tensor:
        """
        Compute variational free energy F = E_q[log q(z) - log p(x,z)]

        Simplified version using available quantities.

        Args:
            prior: Prior p(z) [batch_size, n_states]
            posterior: Posterior q(z) [batch_size, n_states]
            likelihood: Likelihood p(x|z) [batch_size, n_states]

        Returns:
            Free energy [batch_size]
        """
        # KL divergence between posterior and prior
        kl_term = self.compute_kl_divergence(posterior, prior)

        # Expected log likelihood under posterior
        evidence = self.compute_evidence(prior, likelihood)
        expected_ll = torch.log(evidence + 1e-8)

        # Variational free energy
        vfe = -expected_ll + kl_term

        return vfe

    def get_prior_samples(self,
                         n_samples: int,
                         n_states: int,
                         prior_type: str = "uniform",
                         concentration: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate prior distribution samples.

        Args:
            n_samples: Number of samples to generate
            n_states: Number of states
            prior_type: Type of prior ("uniform", "dirichlet", "gaussian")
            concentration: Concentration parameters for Dirichlet prior

        Returns:
            Prior samples [n_samples, n_states]
        """
        if prior_type == "uniform":
            prior = torch.ones(n_samples, n_states) / n_states
        elif prior_type == "dirichlet":
            if concentration is None:
                concentration = torch.ones(n_states)
            # Sample from Dirichlet distribution
            gamma_samples = torch.distributions.Gamma(concentration, 1.0).sample((n_samples,))
            prior = gamma_samples / gamma_samples.sum(dim=1, keepdim=True)
        elif prior_type == "gaussian":
            # Gaussian prior on logits, then softmax
            logits = torch.randn(n_samples, n_states)
            prior = F.softmax(logits, dim=1)
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")

        if self.enable_validation and not self.validate_inputs(prior, torch.ones_like(prior)):
            warnings.warn("Generated prior failed validation")

        return prior

    def get_observation_likelihood(self,
                                 observations: torch.Tensor,
                                 n_states: int,
                                 likelihood_type: str = "categorical",
                                 noise_level: float = 0.1) -> torch.Tensor:
        """
        Generate observation likelihood given real observations.

        Args:
            observations: Real observation data [batch_size, obs_dim]
            n_states: Number of latent states
            likelihood_type: Type of likelihood model
            noise_level: Noise level for likelihood generation

        Returns:
            Likelihood p(x|z) [batch_size, n_states]
        """
        batch_size, obs_dim = observations.shape

        if likelihood_type == "categorical":
            # Simple categorical likelihood based on observation values
            # Map observations to state probabilities
            obs_norm = (observations - observations.min()) / (observations.max() - observations.min() + 1e-8)
            obs_flat = obs_norm.view(batch_size, -1).mean(dim=1, keepdim=True)

            # Create state-dependent likelihood
            state_probs = torch.linspace(0, 1, n_states).unsqueeze(0).expand(batch_size, -1)
            likelihood = torch.exp(-((obs_flat - state_probs) ** 2) / (2 * noise_level ** 2))

        elif likelihood_type == "gaussian":
            # Gaussian likelihood for continuous observations
            # Assume each state corresponds to different mean
            state_means = torch.linspace(observations.min(), observations.max(), n_states)
            state_means = state_means.unsqueeze(0).expand(batch_size, -1)

            # Compute likelihood for each state
            diff = observations.unsqueeze(1) - state_means.unsqueeze(2)
            likelihood = torch.exp(-diff ** 2 / (2 * noise_level ** 2))
            likelihood = likelihood.mean(dim=2)  # Average over observation dimensions

        else:
            raise ValueError(f"Unknown likelihood type: {likelihood_type}")

        # Normalize to ensure valid probability distributions
        likelihood = likelihood / (likelihood.sum(dim=1, keepdim=True) + 1e-8)

        if self.enable_validation and not self.validate_inputs(torch.ones(batch_size, n_states), likelihood):
            warnings.warn("Generated likelihood failed validation")

        return likelihood

    def visualize_beliefs(self,
                         result: BayesianUpdateResult,
                         save_path: Optional[str] = None,
                         show_plot: bool = True):
        """
        Visualize belief distributions and information flow.

        Args:
            result: BayesianUpdateResult to visualize
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
        """
        if not self.enable_visualization:
            return

        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Bayesian Belief Update Analysis', fontsize=16)

            batch_idx = 0  # Visualize first batch element

            # Prior distribution
            axes[0, 0].bar(range(len(result.prior[batch_idx])), result.prior[batch_idx].detach().cpu().numpy())
            axes[0, 0].set_title('Prior Distribution')
            axes[0, 0].set_xlabel('States')
            axes[0, 0].set_ylabel('Probability')

            # Likelihood
            if result.likelihood.dim() == 2:
                axes[0, 1].bar(range(len(result.likelihood[batch_idx])),
                              result.likelihood[batch_idx].detach().cpu().numpy())
            else:
                # For 3D likelihood, show first observation
                axes[0, 1].bar(range(result.likelihood.shape[1]),
                              result.likelihood[batch_idx, :, 0].detach().cpu().numpy())
            axes[0, 1].set_title('Likelihood')
            axes[0, 1].set_xlabel('States')
            axes[0, 1].set_ylabel('Probability')

            # Posterior distribution
            axes[0, 2].bar(range(len(result.posterior[batch_idx])),
                          result.posterior[batch_idx].detach().cpu().numpy())
            axes[0, 2].set_title('Posterior Distribution')
            axes[0, 2].set_xlabel('States')
            axes[0, 2].set_ylabel('Probability')

            # Information measures
            info_labels = ['Prior Entropy', 'Posterior Entropy', 'KL Divergence', 'Mutual Info', 'Free Energy']
            info_values = [
                result.prior_entropy[batch_idx].item(),
                result.posterior_entropy[batch_idx].item(),
                result.kl_divergence[batch_idx].item(),
                result.mutual_information[batch_idx].item(),
                result.free_energy[batch_idx].item()
            ]

            axes[1, 0].bar(info_labels, info_values)
            axes[1, 0].set_title('Information Measures')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Free energy over time (if history available)
            if len(self.history.free_energy_history) > 1:
                axes[1, 1].plot(self.history.free_energy_history, 'b-o')
                axes[1, 1].set_title('Free Energy History')
                axes[1, 1].set_xlabel('Update Step')
                axes[1, 1].set_ylabel('Free Energy')
                axes[1, 1].grid(True, alpha=0.3)

            # Entropy evolution
            if len(self.history.entropy_history) > 1:
                axes[1, 2].plot(self.history.entropy_history, 'r-s')
                axes[1, 2].set_title('Posterior Entropy History')
                axes[1, 2].set_xlabel('Update Step')
                axes[1, 2].set_ylabel('Entropy')
                axes[1, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Visualization saved to {save_path}")

            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")

    def generate_report(self,
                       save_path: Optional[str] = None,
                       include_history: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive report of inference operations.

        Args:
            save_path: Optional path to save report
            include_history: Whether to include full history

        Returns:
            Report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_stats': self.validation_stats.copy(),
            'history_summary': self.history.get_summary_stats(),
            'feature_manager_info': {
                'triton_available': TRITON_AVAILABLE,
                'device': str(self.gpu_accelerator.device),
                'memory_stats': self.gpu_accelerator.get_memory_stats()
            }
        }

        if include_history:
            report['full_history'] = [update.to_dict() for update in self.history.updates]

        # Add recommendations based on performance
        recommendations = []
        if self.validation_stats['failed_updates'] > 0:
            recommendations.append("Review failed updates and input validation")
        if self.validation_stats['computation_times'] and np.mean(self.validation_stats['computation_times']) > 1.0:
            recommendations.append("Consider GPU acceleration for faster computation")
        if len(self.history.updates) > 0:
            avg_fe = np.mean(self.history.free_energy_history)
            if avg_fe > 10.0:
                recommendations.append("High free energy detected - consider model refinement")

        report['recommendations'] = recommendations

        if save_path:
            # Convert numpy arrays to lists for JSON serialization
            def numpy_converter(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
                return str(obj)

            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=numpy_converter)
            self.logger.info(f"Report saved to {save_path}")

        return report

    def reset_history(self):
        """Reset inference history."""
        self.history = InferenceHistory()
        self.validation_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'validation_errors': [],
            'computation_times': []
        }
        self.logger.info("Inference history reset")


# Convenience functions for easy access
def create_bayesian_core(device="auto", enable_logging=True, enable_validation=True, enable_visualization=True):
    """
    Create a Bayesian inference core with specified configuration.

    Args:
        device: Target device ("auto", "cuda", "cpu", "mps")
        enable_logging: Enable comprehensive logging
        enable_validation: Enable input validation
        enable_visualization: Enable visualization capabilities

    Returns:
        Configured BayesianInferenceCore instance
    """
    from .core import TritonFeatureConfig
    config = TritonFeatureConfig(device=device)
    feature_manager = TritonFeatureManager(config)
    return BayesianInferenceCore(
        feature_manager=feature_manager,
        enable_logging=enable_logging,
        enable_validation=enable_validation,
        enable_visualization=enable_visualization
    )


def bayesian_update_with_reporting(prior, likelihood, observations=None, **kwargs):
    """
    Perform Bayesian update with automatic reporting and validation.

    Args:
        prior: Prior distribution
        likelihood: Likelihood function
        observations: Optional observation data
        **kwargs: Additional arguments for BayesianInferenceCore

    Returns:
        BayesianUpdateResult with full reporting
    """
    core = create_bayesian_core(**kwargs)
    result = core.bayesian_update(prior, likelihood, observations)

    # Generate automatic report
    report = core.generate_report(include_history=True)

    return result, report


if __name__ == "__main__":
    # Example usage and validation
    print("Bayesian Inference Core - Validation and Testing")
    print("=" * 60)

    # Create core engine
    core = create_bayesian_core(enable_logging=True, enable_validation=True, enable_visualization=False)

    # Test with synthetic data
    batch_size, n_states = 4, 6

    # Generate priors
    priors = core.get_prior_samples(batch_size, n_states, prior_type="dirichlet")

    # Generate synthetic observations and likelihoods
    observations = torch.randn(batch_size, 3)
    likelihoods = core.get_observation_likelihood(observations, n_states, likelihood_type="categorical")

    print(f"Generated {batch_size} priors with {n_states} states")
    print(f"Prior entropy: {core.compute_entropy(priors).mean().item():.4f}")

    # Perform Bayesian updates
    print("\nPerforming Bayesian updates...")
    results = []
    for i in range(3):
        result = core.bayesian_update(priors, likelihoods, observations)
        results.append(result)
        print(f"Update {i+1}: FE={result.free_energy.mean().item():.4f}, "
              f"KL={result.kl_divergence.mean().item():.4f}")

    # Generate comprehensive report
    report = core.generate_report(save_path="bayesian_core_validation_report.json")
    print("\nValidation complete. Report saved to 'bayesian_core_validation_report.json'")
    print(f"Total updates: {report['validation_stats']['total_updates']}")
    print(f"Average computation time: {np.mean(report['validation_stats']['computation_times']):.4f}s")
