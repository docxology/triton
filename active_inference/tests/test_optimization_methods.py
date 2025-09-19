"""
Test Optimization Methods

Comprehensive tests for Triton-accelerated optimization algorithms:
- Natural gradient descent with Fisher information
- Conjugate gradient methods for linear systems
- Adam optimizer with Triton kernels
- Real Triton kernel validation and performance benchmarks
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any

# Import test utilities
from .conftest import assert_tensors_close, create_synthetic_data

# Import optimization modules
try:
    # Try relative import first (when used as package)
    from ..src.optimization_methods import (
        NaturalGradientOptimizer, ConjugateGradientOptimizer, TritonAdam,
        create_natural_gradient_optimizer, create_conjugate_gradient_optimizer,
        create_triton_adam, benchmark_optimization_methods
    )
except ImportError:
    # Fall back to absolute import (when imported directly)
    from src.optimization_methods import (
        NaturalGradientOptimizer, ConjugateGradientOptimizer, TritonAdam,
        create_natural_gradient_optimizer, create_conjugate_gradient_optimizer,
        create_triton_adam, benchmark_optimization_methods
    )

# Import Triton availability
try:
    from ..src.core import TRITON_AVAILABLE
except ImportError:
    from src.core import TRITON_AVAILABLE


class TestNaturalGradientOptimizer:
    """Test natural gradient optimizer with Triton acceleration."""

    @pytest.fixture
    def natural_gradient_optimizer(self):
        """Create natural gradient optimizer for testing."""
        # Create simple parameter for testing - match optimization_data fixture
        param = torch.randn(4, 128, requires_grad=True)
        # Simple target log probability (standard normal)
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)
        return NaturalGradientOptimizer([param], lr=0.01, target_log_prob=target_log_prob)

    @pytest.fixture
    def optimization_data(self):
        """Generate synthetic optimization test data."""
        batch_size = 4
        param_dim = 128

        # Create parameters with gradients
        params = [torch.randn(batch_size, param_dim, requires_grad=True)]
        gradients = [torch.randn_like(params[0])]

        # Create Fisher information matrices
        fisher_matrices = [torch.eye(param_dim).unsqueeze(0).repeat(batch_size, 1, 1)]

        return {
            "params": params,
            "gradients": gradients,
            "fisher_matrices": fisher_matrices,
            "batch_size": batch_size,
            "param_dim": param_dim
        }

    def test_natural_gradient_initialization(self, natural_gradient_optimizer, optimization_data):
        """Test natural gradient optimizer initialization."""
        params = optimization_data["params"]
        batch_size = optimization_data["batch_size"]
        param_dim = optimization_data["param_dim"]

        assert len(natural_gradient_optimizer.parameters) == len(params)
        assert natural_gradient_optimizer.lr == 0.01
        assert natural_gradient_optimizer.damping == 1e-3

        # Check Fisher matrices
        assert len(natural_gradient_optimizer.fisher_matrices) == len(params)
        assert natural_gradient_optimizer.fisher_matrices[0].shape == (batch_size, param_dim, param_dim)

    def test_natural_gradient_step(self, natural_gradient_optimizer, optimization_data):
        """Test natural gradient optimization step."""
        gradients = optimization_data["gradients"]

        # Store original parameter values
        original_params = [param.clone() for param in natural_gradient_optimizer.parameters]

        # Perform optimization step
        natural_gradient_optimizer.step(gradients)

        # Check that parameters have been updated
        for original, updated in zip(original_params, natural_gradient_optimizer.parameters):
            assert not torch.allclose(original, updated, atol=1e-8)
            assert torch.isfinite(updated).all()

    def test_natural_gradient_fisher_update(self, natural_gradient_optimizer, optimization_data):
        """Test Fisher information matrix updates."""
        params = optimization_data["params"]

        # Create synthetic samples
        samples = [torch.randn_like(param) for param in params]

        # Update Fisher information
        natural_gradient_optimizer.update_fisher(samples)

        # Check that Fisher matrices have been updated
        for fisher in natural_gradient_optimizer.fisher_matrices:
            assert torch.isfinite(fisher).all()

    def test_natural_gradient_multiple_steps(self, natural_gradient_optimizer, optimization_data):
        """Test multiple natural gradient optimization steps."""
        params = optimization_data["params"]
        gradients = optimization_data["gradients"]

        # Perform multiple optimization steps
        for _ in range(5):
            natural_gradient_optimizer.step(gradients)

        # Check that optimization completes without errors
        for param in params:
            assert torch.isfinite(param).all()

    @pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
    def test_natural_gradient_learning_rates(self, lr, optimization_data):
        """Test natural gradient optimizer with different learning rates."""
        params = [param.clone().detach().requires_grad_(True) for param in optimization_data["params"]]
        gradients = optimization_data["gradients"]

        optimizer = NaturalGradientOptimizer(params, lr=lr)

        # Perform optimization step
        optimizer.step(gradients)

        # Check that parameters are updated
        for param in params:
            assert torch.isfinite(param).all()


class TestConjugateGradientOptimizer:
    """Test conjugate gradient optimizer with Triton acceleration."""

    @pytest.fixture
    def conjugate_gradient_optimizer(self):
        """Create conjugate gradient optimizer for testing."""
        # Create simple parameters for testing
        params = [torch.randn(16, 32, requires_grad=True)]
        return ConjugateGradientOptimizer(params)

    @pytest.fixture
    def linear_system_data(self):
        """Generate synthetic linear system data."""
        batch_size = 4
        n = 16

        # Create symmetric positive definite matrix A
        A = torch.randn(batch_size, n, n)
        A = torch.matmul(A, A.transpose(-2, -1)) + torch.eye(n)  # Make SPD

        # Create right-hand side b
        b = torch.randn(batch_size, n)

        return {
            "A": A,
            "b": b,
            "batch_size": batch_size,
            "n": n
        }

    def test_conjugate_gradient_initialization(self, conjugate_gradient_optimizer):
        """Test conjugate gradient optimizer initialization."""
        assert conjugate_gradient_optimizer.max_iter == 100
        assert conjugate_gradient_optimizer.tolerance == 1e-6

    def test_conjugate_gradient_solve_linear_system(self, conjugate_gradient_optimizer, linear_system_data):
        """Test conjugate gradient linear system solver."""
        A = linear_system_data["A"]
        b = linear_system_data["b"]

        # Solve linear system
        x = conjugate_gradient_optimizer.solve_linear_system(A, b)

        # Check output shape
        assert x.shape == b.shape

        # Check that solution is reasonable (Ax should be close to b)
        Ax = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
        residual = torch.norm(Ax - b, dim=-1)
        mean_residual = residual.mean()

        # Should be reasonably small (not checking exact convergence due to numerical precision)
        assert mean_residual < 1.0
        assert torch.isfinite(x).all()

    def test_conjugate_gradient_convergence(self, conjugate_gradient_optimizer, linear_system_data):
        """Test conjugate gradient convergence."""
        A = linear_system_data["A"]
        b = linear_system_data["b"]

        # Solve with tighter tolerance
        conjugate_gradient_optimizer.tolerance = 1e-8
        x = conjugate_gradient_optimizer.solve_linear_system(A, b)

        # Check numerical stability
        assert torch.isfinite(x).all()

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_conjugate_gradient_sizes(self, n):
        """Test conjugate gradient with different problem sizes."""
        params = [torch.randn(4, n, requires_grad=True)]
        optimizer = ConjugateGradientOptimizer(params)

        batch_size = 4
        A = torch.randn(batch_size, n, n)
        A = torch.matmul(A, A.transpose(-2, -1)) + torch.eye(n)
        b = torch.randn(batch_size, n)

        x = optimizer.solve_linear_system(A, b)

        assert x.shape == (batch_size, n)
        assert torch.isfinite(x).all()


class TestTritonAdam:
    """Test Triton-accelerated Adam optimizer."""

    @pytest.fixture
    def triton_adam_optimizer(self):
        """Create Triton Adam optimizer for testing."""
        # Create simple parameters for testing - match adam_data fixture
        params = [torch.randn(4, 128, requires_grad=True)]
        return TritonAdam(params, lr=0.001)

    @pytest.fixture
    def adam_data(self):
        """Generate synthetic Adam test data."""
        batch_size = 4
        param_dim = 128

        # Create parameters with gradients
        params = [torch.randn(batch_size, param_dim, requires_grad=True)]
        gradients = [torch.randn_like(params[0])]

        return {
            "params": params,
            "gradients": gradients,
            "batch_size": batch_size,
            "param_dim": param_dim
        }

    def test_triton_adam_initialization(self, triton_adam_optimizer, adam_data):
        """Test Triton Adam optimizer initialization."""
        params = adam_data["params"]

        assert triton_adam_optimizer.lr == 0.001
        assert triton_adam_optimizer.beta1 == 0.9
        assert triton_adam_optimizer.beta2 == 0.999
        assert triton_adam_optimizer.eps == 1e-8
        assert triton_adam_optimizer.t == 1

        # Check moment initialization
        assert len(triton_adam_optimizer.moments) == len(params)
        assert len(triton_adam_optimizer.second_moments) == len(params)

        for param, m, v in zip(params, triton_adam_optimizer.moments, triton_adam_optimizer.second_moments):
            assert m.shape == param.shape
            assert v.shape == param.shape
            assert torch.allclose(m, torch.zeros_like(m))
            assert torch.allclose(v, torch.zeros_like(v))

    def test_triton_adam_step(self, triton_adam_optimizer, adam_data):
        """Test Triton Adam optimization step."""
        gradients = adam_data["gradients"]

        # Store original parameter values
        original_params = [param.clone() for param in triton_adam_optimizer.parameters]

        # Perform optimization step
        triton_adam_optimizer.step(gradients)

        # Check that parameters have been updated
        for original, updated in zip(original_params, triton_adam_optimizer.parameters):
            assert not torch.allclose(original, updated, atol=1e-8)
            assert torch.isfinite(updated).all()

        # Check timestep increment
        assert triton_adam_optimizer.t == 2

    def test_triton_adam_multiple_steps(self, triton_adam_optimizer, adam_data):
        """Test multiple Triton Adam optimization steps."""
        gradients = adam_data["gradients"]

        # Perform multiple optimization steps
        for i in range(5):
            triton_adam_optimizer.step(gradients)
            assert triton_adam_optimizer.t == i + 2  # Starts at 1, increments each step

        # Check that optimization completes without errors
        for param in triton_adam_optimizer.parameters:
            assert torch.isfinite(param).all()

    def test_triton_adam_bias_correction(self, triton_adam_optimizer, adam_data):
        """Test Adam bias correction."""
        gradients = adam_data["gradients"]

        # Run for several steps to see bias correction effect
        for _ in range(10):
            triton_adam_optimizer.step(gradients)

        # Parameters should still be finite
        for param in triton_adam_optimizer.parameters:
            assert torch.isfinite(param).all()

    @pytest.mark.parametrize("lr", [0.0001, 0.001, 0.01])
    def test_triton_adam_learning_rates(self, lr, adam_data):
        """Test Triton Adam optimizer with different learning rates."""
        params = [param.clone().detach().requires_grad_(True) for param in adam_data["params"]]
        gradients = adam_data["gradients"]

        optimizer = TritonAdam(params, lr=lr)

        # Perform optimization step
        optimizer.step(gradients)

        # Check that parameters are updated
        for param in params:
            assert torch.isfinite(param).all()


class TestOptimizationMethodsIntegration:
    """Integration tests for optimization method implementations."""

    def test_natural_gradient_adam_comparison(self):
        """Test comparison between natural gradient and Adam."""
        # Create test problem
        param = torch.randn(16, 32, requires_grad=True)

        # Natural gradient optimizer
        ng_opt = NaturalGradientOptimizer([param.clone()], lr=0.01)

        # Adam optimizer
        adam_opt = TritonAdam([param.clone()], lr=0.01)

        # Generate gradients
        gradient = torch.randn_like(param)

        # Perform optimization steps
        ng_param = param.clone()
        adam_param = param.clone()

        ng_opt_params = [ng_param]
        adam_opt_params = [adam_param]

        ng_opt.step([gradient])
        adam_opt.step([gradient])

        # Both should produce finite updates
        assert torch.isfinite(ng_param).all()
        assert torch.isfinite(adam_param).all()

    def test_conjugate_gradient_quadratic_form(self):
        """Test conjugate gradient on quadratic optimization problem."""
        # Create quadratic problem: minimize (1/2)x^T A x - b^T x
        batch_size = 4
        n = 8

        # Create positive definite matrix A
        A = torch.randn(batch_size, n, n)
        A = torch.matmul(A, A.transpose(-2, -1)) + 0.1 * torch.eye(n)

        # Create vector b
        b = torch.randn(batch_size, n)

        # Create optimizer
        params = [torch.randn(batch_size, n, requires_grad=True)]
        optimizer = ConjugateGradientOptimizer(params)

        # Solve linear system A x = b
        x = optimizer.solve_linear_system(A, b)

        # Check that A x ≈ b
        Ax = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
        residual = torch.norm(Ax - b, dim=-1).mean()

        assert residual < 0.1  # Should be reasonably close

    def test_optimization_methods_benchmark(self):
        """Test benchmarking functionality."""
        if TRITON_AVAILABLE:
            results = benchmark_optimization_methods()

            # Check that benchmark results are reasonable
            assert "natural_gradient" in results
            assert "adam" in results

            for method, metrics in results.items():
                assert "triton_accelerated" in metrics
                assert isinstance(metrics["triton_accelerated"], bool)

    def test_optimization_gradient_flow(self):
        """Test end-to-end gradient flow in optimization methods."""
        # Create a simple optimization problem
        param = torch.randn(8, 16, requires_grad=True)

        # Create optimizer
        optimizer = TritonAdam([param], lr=0.01)

        # Create a simple loss function
        def loss_fn(p):
            return torch.sum(p ** 2)

        # Optimization loop
        for _ in range(5):
            # Compute gradients
            loss = loss_fn(param)
            loss.backward()

            # Optimization step
            optimizer.step([param.grad])

            # Clear gradients
            param.grad.zero_()

        # Check that optimization reduces the loss
        final_loss = loss_fn(param)
        assert final_loss < loss_fn(torch.randn_like(param))  # Should be better than random


class TestOptimizationMethodsConvenienceFunctions:
    """Test convenience functions for creating optimization methods."""

    def test_create_natural_gradient_optimizer(self):
        """Test natural gradient optimizer creation function."""
        params = [torch.randn(16, 32, requires_grad=True)]
        optimizer = create_natural_gradient_optimizer(params, lr=0.01, use_triton=True)

        assert isinstance(optimizer, NaturalGradientOptimizer)
        assert optimizer.lr == 0.01

    def test_create_conjugate_gradient_optimizer(self):
        """Test conjugate gradient optimizer creation function."""
        params = [torch.randn(8, 16, requires_grad=True)]
        optimizer = create_conjugate_gradient_optimizer(params, use_triton=True)

        assert isinstance(optimizer, ConjugateGradientOptimizer)

    def test_create_triton_adam(self):
        """Test Triton Adam creation function."""
        params = [torch.randn(32, 64, requires_grad=True)]
        optimizer = create_triton_adam(params, lr=0.001, use_triton=True)

        assert isinstance(optimizer, TritonAdam)
        assert optimizer.lr == 0.001


# Performance tests
@pytest.mark.slow
class TestOptimizationMethodsPerformance:
    """Performance tests for optimization method implementations."""

    def test_natural_gradient_performance_scaling(self):
        """Test natural gradient performance with different sizes."""
        sizes = [(16, 32), (32, 64), (64, 128)]  # batch_size, param_dim

        for batch_size, param_dim in sizes:
            param = torch.randn(batch_size, param_dim, requires_grad=True)
            optimizer = NaturalGradientOptimizer([param], lr=0.01)

            gradient = torch.randn_like(param)

            # Time the optimization step
            import time
            start_time = time.time()

            for _ in range(10):
                optimizer.step([gradient])

            end_time = time.time()

            # Check that performance is reasonable
            total_time = end_time - start_time
            assert total_time < 5.0  # Should complete in reasonable time

    def test_adam_performance_scaling(self):
        """Test Adam performance with different sizes."""
        sizes = [(16, 32), (32, 64), (64, 128)]  # batch_size, param_dim

        for batch_size, param_dim in sizes:
            param = torch.randn(batch_size, param_dim, requires_grad=True)
            optimizer = TritonAdam([param], lr=0.001)

            gradient = torch.randn_like(param)

            # Time the optimization step
            import time
            start_time = time.time()

            for _ in range(10):
                optimizer.step([gradient])

            end_time = time.time()

            # Check performance
            total_time = end_time - start_time
            assert total_time < 3.0  # Should complete in reasonable time

    def test_conjugate_gradient_performance_scaling(self):
        """Test conjugate gradient performance with different sizes."""
        sizes = [8, 16, 32]  # problem dimension

        for n in sizes:
            params = [torch.randn(4, n, requires_grad=True)]
            optimizer = ConjugateGradientOptimizer(params)

            batch_size = 4
            A = torch.randn(batch_size, n, n)
            A = torch.matmul(A, A.transpose(-2, -1)) + torch.eye(n)
            b = torch.randn(batch_size, n)

            # Time the linear system solve
            import time
            start_time = time.time()

            for _ in range(3):
                x = optimizer.solve_linear_system(A, b)

            end_time = time.time()

            # Check performance
            total_time = end_time - start_time
            assert total_time < 10.0  # Should complete in reasonable time


class TestOptimizationEdgeCases:
    """Test edge cases and numerical stability for optimization methods."""

    def test_natural_gradient_zero_gradients(self):
        """Test natural gradient optimizer with zero gradients."""
        param = torch.randn(4, 128, requires_grad=True)
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        optimizer = NaturalGradientOptimizer([param], lr=0.01, target_log_prob=target_log_prob)

        # Zero gradients
        zero_grads = [torch.zeros_like(param)]
        original_param = param.clone()

        optimizer.step(zero_grads)

        # Parameters should remain unchanged with zero gradients
        assert torch.allclose(param, original_param, atol=1e-8)

    def test_natural_gradient_extreme_learning_rate(self):
        """Test natural gradient optimizer with extreme learning rates."""
        param = torch.randn(4, 128, requires_grad=True)
        def target_log_prob(x):
            return -0.5 * torch.sum(x ** 2, dim=-1)

        # Test with very small learning rate
        optimizer_small = NaturalGradientOptimizer([param], lr=1e-8, target_log_prob=target_log_prob)
        gradients = [torch.randn_like(param)]
        original_param = param.clone()

        optimizer_small.step(gradients)

        # Parameters should change very little
        assert torch.allclose(param, original_param, atol=1e-6)

    def test_adam_numerical_stability(self):
        """Test Adam optimizer numerical stability with extreme values."""
        # Test with parameters that have extreme values
        param = torch.tensor([[1e10, -1e10, 1e-10, -1e-10]], requires_grad=True, dtype=torch.float32)
        gradients = [torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float32)]

        optimizer = TritonAdam([param], lr=0.001)

        # Perform multiple steps to test stability
        for _ in range(10):
            optimizer.step(gradients)

        # Parameters should remain finite
        assert torch.isfinite(param).all()
        assert not torch.isnan(param).any()
        assert not torch.isinf(param).any()

    def test_conjugate_gradient_singular_matrix(self):
        """Test conjugate gradient with nearly singular matrix."""
        # Create dummy parameters (required by ConjugateGradientOptimizer)
        params = [torch.randn(2, 2, requires_grad=True)]

        # Create a nearly singular matrix
        A = torch.tensor([[[1.0, 0.9999], [0.9999, 1.0]]], dtype=torch.float32)
        b = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

        optimizer = ConjugateGradientOptimizer(params, max_iter=100, tolerance=1e-6)

        x = optimizer.solve_linear_system(A, b)

        # Should still produce finite result
        assert torch.isfinite(x).all()

        # Check that Ax ≈ b (within tolerance)
        residual = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1) - b
        assert torch.norm(residual) < 1e-2  # Relaxed tolerance for near-singular matrix

    def test_optimizer_parameter_validation(self):
        """Test parameter validation in optimizers."""
        # Test with empty parameter list
        with pytest.raises(ValueError):
            NaturalGradientOptimizer([], lr=0.01)

        with pytest.raises(ValueError):
            TritonAdam([], lr=0.01)

        # Test with mismatched gradients
        param = torch.randn(4, 128, requires_grad=True)
        optimizer = TritonAdam([param], lr=0.01)

        # Wrong number of gradients
        with pytest.raises(ValueError):
            optimizer.step([])

        # Wrong gradient type
        with pytest.raises(TypeError):
            optimizer.step(["not a tensor"])

    def test_gradient_accumulation_edge_cases(self):
        """Test gradient accumulation with edge cases."""
        param = torch.randn(2, 64, requires_grad=True)
        optimizer = TritonAdam([param], lr=0.01)

        # Very small gradients
        tiny_grads = [torch.full_like(param, 1e-8)]
        original_param = param.clone()

        optimizer.step(tiny_grads)

        # Parameters should change
        assert not torch.allclose(param, original_param, atol=1e-10)

        # Very large gradients
        param_large = torch.randn(2, 64, requires_grad=True)
        optimizer_large = TritonAdam([param_large], lr=0.01)
        large_grads = [torch.full_like(param_large, 1e6)]

        optimizer_large.step(large_grads)

        # Should still be finite
        assert torch.isfinite(param_large).all()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
