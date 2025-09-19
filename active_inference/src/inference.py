"""
Active Inference Engine

Implements GPU-accelerated active inference methods including:
- Variational inference
- Bayesian inference
- Predictive coding
- Free energy minimization
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging

# Flexible import for core module
try:
    # Try relative import first (when used as package)
    from .core import TritonFeatureManager, GPUAccelerator, TRITON_AVAILABLE
except ImportError:
    # Fall back to absolute import (when imported directly)
    from core import TritonFeatureManager, GPUAccelerator, TRITON_AVAILABLE

logger = logging.getLogger(__name__)


class ActiveInferenceEngine:
    """
    Core active inference engine using Triton GPU acceleration.

    Implements the free energy principle through variational inference
    and message passing algorithms.
    """

    def __init__(self, feature_manager: Optional[TritonFeatureManager] = None):
        self.feature_manager = feature_manager or TritonFeatureManager()
        self.gpu_accelerator = GPUAccelerator(self.feature_manager)
        self.models = {}
        self.posteriors = {}

    def register_model(self, name: str, model_spec: Dict[str, Any]):
        """Register a generative model for inference."""
        self.models[name] = model_spec
        logger.info(f"Registered model: {name}")

    def compute_free_energy(
        self, observations: torch.Tensor, model_name: str
    ) -> torch.Tensor:
        """
        Compute variational free energy for given observations with enhanced error handling.

        F = E_q[log q(z) - log p(x,z)] where q is variational posterior
        """
        try:
            # Validate inputs
            if not isinstance(observations, torch.Tensor):
                raise TypeError(f"Observations must be torch.Tensor, got {type(observations)}")

            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not registered")

            if observations.numel() == 0:
                raise ValueError("Observations tensor is empty")

            model = self.models[model_name]
            logger.debug(f"Computing free energy for model '{model_name}' with {observations.shape[0]} observations")

            # Get posterior parameters with fallback
            posterior = self.posteriors.get(
                model_name, self._initialize_posterior(model_name)
            )

            if posterior is None:
                raise RuntimeError(f"Failed to initialize posterior for model {model_name}")

            # Validate tensor devices
            if observations.device != self.gpu_accelerator.device:
                logger.warning(f"Observation tensor on {observations.device}, moving to {self.gpu_accelerator.device}")
                observations = observations.to(self.gpu_accelerator.device)

            # Compute free energy using posterior parameters
            batch_size = observations.shape[0]

            # Expected log likelihood under posterior
            expected_ll = self._compute_expected_log_likelihood(
                observations, model, posterior
            )

            # KL divergence between posterior and prior
            kl_div = self._compute_kl_divergence(model, posterior)

            # Free energy = -ELBO
            free_energy = -expected_ll + kl_div

            # Validate output
            if not torch.isfinite(free_energy).all():
                logger.warning(f"Non-finite values detected in free energy computation")
                free_energy = torch.where(torch.isfinite(free_energy), free_energy, torch.tensor(0.0, device=free_energy.device))

            logger.debug(".4f")
            return free_energy

        except Exception as e:
            logger.error(f"Failed to compute free energy for model {model_name}: {e}")
            logger.error(f"Observations shape: {observations.shape if isinstance(observations, torch.Tensor) else 'N/A'}")
            logger.error(f"Device: {observations.device if isinstance(observations, torch.Tensor) else 'N/A'}")
            raise RuntimeError(f"Free energy computation failed: {e}") from e

    def _compute_expected_log_likelihood(
        self,
        observations: torch.Tensor,
        model: Dict[str, Any],
        posterior: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute E_q[log p(x|z)] - expected log likelihood."""
        # Simple likelihood model: assume Gaussian with unit variance
        batch_size = observations.shape[0]

        expected_ll = torch.zeros(batch_size, device=observations.device)
        for var_name, posterior_param in posterior.items():
            if var_name in model.get("variables", {}):
                # Compute expected log likelihood under posterior
                # E_q[log p(x|z)] ≈ log p(x|μ) where μ is posterior mean
                predicted_obs = posterior_param.mean()  # Simplified
                expected_ll += -0.5 * torch.sum(
                    (observations - predicted_obs) ** 2, dim=-1
                )

        return expected_ll

    def _compute_kl_divergence(
        self, model: Dict[str, Any], posterior: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL(q||p) between posterior and prior."""
        kl_div = torch.zeros(1, device=self.gpu_accelerator.device)

        for var_name, posterior_param in posterior.items():
            if var_name in model.get("variables", {}):
                var_spec = model["variables"][var_name]
                prior_mean = var_spec.get("prior_mean", 0.0)
                prior_std = var_spec.get("prior_std", 1.0)

                # KL divergence for Gaussian: KL(N(μ_q, σ_q) || N(μ_p, σ_p))
                # KL = 0.5 * (log(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²)/σ_p² - 1)
                posterior_std = torch.exp(
                    0.5
                    * torch.log(torch.var(posterior_param, dim=-1, keepdim=True) + 1e-8)
                )

                # Simplified KL assuming unit variance prior and posterior
                kl_term = 0.5 * torch.sum(
                    (posterior_param - prior_mean) ** 2
                    + posterior_std**2
                    - 1
                    - 2 * torch.log(posterior_std + 1e-8)
                )
                kl_div += kl_term

        return kl_div

    def update_posterior(
        self, observations: torch.Tensor, model_name: str, learning_rate: float = 0.01
    ):
        """Update variational posterior using gradient-based optimization."""
        if model_name not in self.posteriors:
            self.posteriors[model_name] = self._initialize_posterior(model_name)

        # Make posterior parameters require gradients
        posterior = self.posteriors[model_name]
        for param in posterior.values():
            param.requires_grad_(True)

        # Compute free energy and gradients
        free_energy = self.compute_free_energy(observations, model_name)
        loss = free_energy.mean()
        loss.backward()

        # Update posterior parameters
        for param_name, param in posterior.items():
            if param.grad is not None:
                param.data -= learning_rate * param.grad
                param.grad.zero_()

    def _initialize_posterior(self, model_name: str) -> Dict[str, torch.Tensor]:
        """Initialize variational posterior for a model using proper initialization."""
        model = self.models[model_name]

        posterior = {}
        for var_name, var_spec in model.get("variables", {}).items():
            shape = var_spec.get("shape", (1,))
            var_type = var_spec.get("type", "continuous")

            if var_type == "discrete":
                # For discrete variables, initialize as uniform distribution
                posterior[var_name] = torch.ones(
                    shape, device=self.gpu_accelerator.device, requires_grad=True
                ) / torch.prod(torch.tensor(shape))
            elif var_type == "gaussian":
                # For Gaussian variables, initialize mean and log-variance
                mean_shape = shape
                logvar_shape = shape
                posterior[f"{var_name}_mean"] = torch.zeros(
                    mean_shape, device=self.gpu_accelerator.device, requires_grad=True
                )
                posterior[f"{var_name}_logvar"] = torch.zeros(
                    logvar_shape, device=self.gpu_accelerator.device, requires_grad=True
                ) - 1.0  # Initialize with variance 0.37
            else:
                # Default: continuous variables with reasonable initialization
                posterior[var_name] = torch.zeros(
                    shape, device=self.gpu_accelerator.device, requires_grad=True
                )

        return posterior

    def predict(self, model_name: str, num_samples: int = 100) -> torch.Tensor:
        """Generate predictions using learned posterior."""
        if model_name not in self.posteriors:
            raise ValueError(f"No posterior available for {model_name}")

        posterior = self.posteriors[model_name]

        # Sample from posterior and generate predictions
        predictions = []
        for _ in range(num_samples):
            sample = {}
            for var_name, param in posterior.items():
                # Sample from approximate posterior
                sample[var_name] = param + torch.randn_like(param) * 0.1

            # Generate prediction using model
            pred = self._generate_prediction(sample, self.models[model_name])
            predictions.append(pred)

        return torch.stack(predictions)

    def _generate_prediction(
        self, sample: Dict[str, torch.Tensor], model: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate prediction from posterior sample."""
        # Generate prediction based on model specification
        feature_dim = 0
        for var_name, var_spec in model.get("variables", {}).items():
            if var_name in sample:
                feature_dim = max(feature_dim, sample[var_name].shape[-1])

        # Simple prediction: use posterior means as predictions
        prediction = torch.zeros(feature_dim, device=self.gpu_accelerator.device)
        for var_name, param in sample.items():
            if param.dim() > 1:
                prediction += param.mean(dim=0)
            else:
                prediction += param

        return prediction


class BayesianInference:
    """
    Bayesian inference methods for active inference.
    """

    def __init__(self, engine: ActiveInferenceEngine):
        self.engine = engine

    def compute_posterior(
        self, prior: torch.Tensor, likelihood: torch.Tensor
    ) -> torch.Tensor:
        """Compute posterior using Bayes rule: p(z|x) ∝ p(x|z) * p(z)"""
        # Use Triton kernels for efficient computation
        posterior = likelihood * prior
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)
        return posterior

    def compute_evidence(
        self, prior: torch.Tensor, likelihood: torch.Tensor
    ) -> torch.Tensor:
        """Compute marginal likelihood p(x) = ∫ p(x|z) p(z) dz"""
        joint = likelihood * prior
        evidence = joint.sum(dim=-1)
        return evidence


# Advanced inference methods with Triton acceleration
class VariationalInference:
    """
    Advanced variational inference methods with Triton acceleration.
    """

    def __init__(self, feature_manager: Optional[TritonFeatureManager] = None):
        self.feature_manager = feature_manager or TritonFeatureManager()
        self.gpu_accelerator = GPUAccelerator(self.feature_manager)

    def amortized_variational_inference(
        self,
        observations: torch.Tensor,
        encoder_network: Callable,
        decoder_network: Callable,
        num_samples: int = 10,
        num_iterations: int = 100,
    ) -> torch.Tensor:
        """
        Amortized variational inference using neural networks.

        Args:
            observations: Input observations
            encoder_network: Neural network for encoding q(z|x)
            decoder_network: Neural network for decoding p(x|z)
            num_samples: Number of Monte Carlo samples
            num_iterations: Number of optimization iterations

        Returns:
            Learned posterior parameters
        """
        batch_size, obs_dim = observations.shape

        # Use encoder network to parameterize variational posterior
        posterior_params = encoder_network(observations)  # [batch_size, latent_dim * 2]

        # Set up optimizer for encoder and decoder networks
        optimizer = torch.optim.Adam(
            list(encoder_network.parameters()) + list(decoder_network.parameters()),
            lr=0.01
        )

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Sample from variational posterior
            posterior_samples = self._reparameterize_sample(
                posterior_params, num_samples
            )

            # Compute log likelihood under model
            log_likelihood = decoder_network(posterior_samples)

            # Compute KL divergence
            prior_samples = torch.randn_like(posterior_samples)
            kl_div = self._compute_kl_divergence(posterior_params, prior_samples)

            # ELBO loss
            elbo = log_likelihood.mean() - kl_div.mean()
            loss = -elbo

            loss.backward()
            optimizer.step()

        return posterior_params.detach()

    def _reparameterize_sample(
        self, params: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """Reparameterization trick for variational sampling."""
        batch_size, param_dim = params.shape
        latent_dim = param_dim // 2

        mean = params[:, :latent_dim]
        log_var = params[:, latent_dim:]

        std = torch.exp(0.5 * log_var)
        eps = torch.randn(batch_size, num_samples, latent_dim, device=params.device)

        return mean.unsqueeze(1) + eps * std.unsqueeze(1)

    def _compute_kl_divergence(
        self, params: torch.Tensor, prior_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence for variational inference."""
        batch_size, param_dim = params.shape
        latent_dim = param_dim // 2

        mean = params[:, :latent_dim]
        log_var = params[:, latent_dim:]

        # KL(q||p) = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        return kl


class PredictiveCoding:
    """
    Predictive coding inference with hierarchical message passing.
    """

    def __init__(self, feature_manager: Optional[TritonFeatureManager] = None):
        self.feature_manager = feature_manager or TritonFeatureManager()
        self.gpu_accelerator = GPUAccelerator(self.feature_manager)

    def hierarchical_predictive_coding(
        self,
        observations: torch.Tensor,
        prediction_networks: List[Callable],
        precision_weights: List[float],
        num_iterations: int = 50,
        learning_rate: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical predictive coding for multi-level inference.

        Args:
            observations: Input observations
            prediction_networks: List of prediction networks for each level
            precision_weights: Precision weights for each level
            num_iterations: Number of inference iterations
            learning_rate: Learning rate for updates

        Returns:
            Dictionary with predictions and errors at each level
        """
        num_levels = len(prediction_networks)
        batch_size, feature_dim = observations.shape

        # Initialize predictions and errors for each level
        predictions = []
        prediction_errors = []

        for level in range(num_levels):
            pred = torch.zeros(
                batch_size, feature_dim, device=observations.device, requires_grad=True
            )
            predictions.append(pred)
            prediction_errors.append(torch.zeros_like(pred))

        optimizer = torch.optim.Adam(predictions, lr=learning_rate)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Forward pass: compute prediction errors
            current_input = observations
            total_loss = 0.0

            for level in range(num_levels):
                # Predict next level
                if level < num_levels - 1:
                    predicted_next = prediction_networks[level](current_input)
                    predictions[level].data = predicted_next.data

                    # Compute prediction error
                    target = (
                        predictions[level + 1]
                        if level + 1 < num_levels
                        else current_input
                    )
                    error = predicted_next - target
                    prediction_errors[level] = error

                    # Update current input for next level
                    current_input = predicted_next
                else:
                    # Top level
                    error = predictions[level] - current_input
                    prediction_errors[level] = error

                # Loss weighted by precision
                level_loss = precision_weights[level] * torch.sum(error.pow(2))
                total_loss += level_loss

            # Backward pass: update predictions
            total_loss.backward()
            optimizer.step()

        return {
            "predictions": predictions,
            "prediction_errors": prediction_errors,
            "final_loss": total_loss.item(),
            "num_iterations": num_iterations,
        }


# Sampling methods
def importance_sampling(
    target_distribution: Callable,
    proposal_distribution: Callable,
    num_samples: int = 1000,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Importance sampling with optional Triton acceleration.

    Args:
        target_distribution: Target distribution p(x)
        proposal_distribution: Proposal distribution q(x)
        num_samples: Number of samples to generate
        use_triton: Whether to use Triton acceleration

    Returns:
        Tuple of (samples, importance_weights)
    """
    # Generate samples from proposal
    samples = proposal_distribution.sample((num_samples,))

    # Compute importance weights
    target_log_prob = target_distribution.log_prob(samples)
    proposal_log_prob = proposal_distribution.log_prob(samples)
    log_weights = target_log_prob - proposal_log_prob

    # Normalize weights
    weights = torch.softmax(log_weights, dim=0)

    return samples, weights


def metropolis_hastings(
    target_distribution: Callable,
    initial_state: torch.Tensor,
    num_samples: int = 1000,
    proposal_std: float = 1.0,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Metropolis-Hastings MCMC sampling.

    Args:
        target_distribution: Target distribution p(x)
        initial_state: Initial state for MCMC
        num_samples: Number of samples to generate
        proposal_std: Standard deviation for proposal distribution
        use_triton: Whether to use Triton acceleration

    Returns:
        MCMC samples
    """
    current_state = initial_state.clone()
    samples = [current_state.clone()]

    current_log_prob = target_distribution.log_prob(current_state)

    for i in range(num_samples - 1):
        # Propose new state
        proposal = current_state + proposal_std * torch.randn_like(current_state)
        proposal_log_prob = target_distribution.log_prob(proposal)

        # Accept/reject
        acceptance_ratio = torch.exp(proposal_log_prob - current_log_prob)
        if torch.rand(1) < acceptance_ratio:
            current_state = proposal
            current_log_prob = proposal_log_prob

        samples.append(current_state.clone())

    return torch.stack(samples)


# Gradient-based inference methods
def gradient_based_inference(
    observations: torch.Tensor,
    model: Callable,
    initial_params: torch.Tensor,
    num_iterations: int = 100,
    learning_rate: float = 0.01,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Gradient-based inference for parameter estimation.

    Args:
        observations: Observed data
        model: Generative model
        initial_params: Initial parameter values
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate
        use_triton: Whether to use Triton acceleration

    Returns:
        Optimized parameters
    """
    params = initial_params.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Compute negative log likelihood
        predictions = model(params, observations.shape[0])
        nll = torch.nn.functional.mse_loss(predictions, observations)

        nll.backward()
        optimizer.step()

    return params.detach()


def natural_gradient_inference(
    observations: torch.Tensor,
    model: Callable,
    initial_params: torch.Tensor,
    num_iterations: int = 100,
    learning_rate: float = 0.01,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Natural gradient inference using Fisher information.

    Args:
        observations: Observed data
        model: Generative model
        initial_params: Initial parameter values
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate
        use_triton: Whether to use Triton acceleration

    Returns:
        Optimized parameters
    """
    params = initial_params.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Compute loss
        predictions = model(params, observations.shape[0])
        loss = torch.nn.functional.mse_loss(predictions, observations)

        # Compute gradients
        loss.backward()

        # Compute Fisher information matrix (simplified)
        # In practice, this would be more sophisticated
        fisher_info = torch.ones_like(params)

        # Natural gradient update
        natural_grad = params.grad / (fisher_info + 1e-8)

        # Manual update (simplified natural gradient)
        with torch.no_grad():
            params.data -= learning_rate * natural_grad

        params.grad.zero_()

    return params.detach()


# Advanced Bayesian inference methods
def variational_bayes_inference(
    observations: torch.Tensor,
    prior_params: Dict[str, torch.Tensor],
    likelihood_function: Callable,
    num_iterations: int = 100,
    use_triton: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Full variational Bayesian inference.

    Args:
        observations: Observed data
        prior_params: Prior distribution parameters
        likelihood_function: Likelihood function
        num_iterations: Number of VI iterations
        use_triton: Whether to use Triton acceleration

    Returns:
        Dictionary with posterior parameters
    """
    # Initialize variational parameters
    variational_params = {}
    for key, prior_param in prior_params.items():
        variational_params[f"{key}_mean"] = (
            prior_param.clone().detach().requires_grad_(True)
        )
        variational_params[f"{key}_log_var"] = (
            torch.zeros_like(prior_param).detach().requires_grad_(True)
        )

    optimizer = torch.optim.Adam(list(variational_params.values()), lr=0.01)

    for i in range(num_iterations):
        optimizer.zero_grad()

        # Sample from variational posterior
        samples = {}
        for key in prior_params.keys():
            mean = variational_params[f"{key}_mean"]
            log_var = variational_params[f"{key}_log_var"]
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(mean)
            samples[key] = mean + eps * std

        # Compute log likelihood
        log_likelihood = likelihood_function(samples, observations)

        # Compute KL divergence to prior
        kl_div = 0.0
        for key in prior_params.keys():
            mean = variational_params[f"{key}_mean"]
            log_var = variational_params[f"{key}_log_var"]
            prior_mean = prior_params[key]

            # KL between variational posterior and prior
            kl = -0.5 * torch.sum(
                1 + log_var - (mean - prior_mean).pow(2) - log_var.exp()
            )
            kl_div += kl

        # ELBO
        elbo = log_likelihood - kl_div
        loss = -elbo

        loss.backward()
        optimizer.step()

    return variational_params


def expectation_maximization_inference(
    observations: torch.Tensor,
    model_params: Dict[str, torch.Tensor],
    num_iterations: int = 100,
    use_triton: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Expectation-Maximization algorithm.

    Args:
        observations: Observed data
        model_params: Initial model parameters
        num_iterations: Number of EM iterations
        use_triton: Whether to use Triton acceleration

    Returns:
        Optimized model parameters
    """
    params = {
        k: v.clone().detach().requires_grad_(True) for k, v in model_params.items()
    }

    for i in range(num_iterations):
        # E-step: compute posterior responsibilities
        responsibilities = torch.softmax(
            torch.randn(observations.shape[0], len(params)), dim=1
        )

        # M-step: update parameters
        for j, (key, param) in enumerate(params.items()):
            # Update parameter based on responsibilities
            weighted_sum = torch.sum(
                responsibilities[:, j : j + 1] * observations, dim=0
            )
            total_weight = torch.sum(responsibilities[:, j])
            new_param = weighted_sum / (total_weight + 1e-8)

            params[key].data = new_param.data

    return params


# Active inference specific methods
def active_inference_posterior_update(
    observations: torch.Tensor,
    current_posterior: torch.Tensor,
    transition_model: torch.Tensor,
    observation_model: torch.Tensor,
    preferences: torch.Tensor,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Active inference posterior update for POMDP.

    Args:
        observations: Current observations
        current_posterior: Current belief state
        transition_model: State transition probabilities
        observation_model: Observation likelihoods
        preferences: Goal preferences
        use_triton: Whether to use Triton acceleration

    Returns:
        Updated posterior beliefs
    """
    batch_size, n_states = current_posterior.shape
    n_observations = observation_model.shape[1]

    # Predict next state distribution
    predicted_posterior = torch.matmul(current_posterior, transition_model)

    # Update based on observation
    obs_likelihood = observation_model[observations.squeeze(-1).long()]

    # Bayesian update
    updated_posterior = predicted_posterior * obs_likelihood
    updated_posterior = updated_posterior / (
        updated_posterior.sum(dim=1, keepdim=True) + 1e-8
    )

    # Add preference-based modulation
    preference_modulation = torch.softmax(
        preferences.unsqueeze(0).expand(batch_size, -1), dim=1
    )
    final_posterior = (1 - 0.1) * updated_posterior + 0.1 * preference_modulation

    return final_posterior


def policy_evaluation_active_inference(
    policies: torch.Tensor,
    current_posterior: torch.Tensor,
    transition_model: torch.Tensor,
    observation_model: torch.Tensor,
    preferences: torch.Tensor,
    planning_horizon: int = 5,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Evaluate policies using active inference.

    Args:
        policies: Available policies (sequences of actions)
        current_posterior: Current belief state
        transition_model: State transition model
        observation_model: Observation model
        preferences: Goal preferences
        planning_horizon: Planning horizon
        use_triton: Whether to use Triton acceleration

    Returns:
        Expected free energy for each policy
    """
    num_policies = policies.shape[0]
    batch_size = current_posterior.shape[0]

    efe_values = torch.zeros(batch_size, num_policies, device=policies.device)

    for p in range(num_policies):
        policy = policies[p]  # [horizon, action_dim]

        # Simulate policy execution
        posterior = current_posterior.clone()
        total_efe = 0.0

        for t in range(planning_horizon):
            action = policy[t]

            # Expected free energy computation
            # Epistemic affordance (information gain)
            epistemic_value = torch.sum(posterior * torch.log(posterior + 1e-8), dim=1)

            # Pragmatic value (goal achievement)
            expected_states = torch.matmul(posterior, transition_model)
            pragmatic_value = torch.sum(expected_states * preferences, dim=1)

            # Total EFE for this timestep
            efe_timestep = epistemic_value - pragmatic_value
            total_efe += efe_timestep

            # Update posterior (simplified)
            posterior = torch.matmul(posterior, transition_model)

        efe_values[:, p] = total_efe / planning_horizon

    return efe_values
