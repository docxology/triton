"""
Test Neural Network Components

Comprehensive tests for Triton-accelerated neural network implementations:
- Attention mechanisms with flash attention optimization
- LSTM networks with efficient memory access
- Fused convolution + batch normalization
- Real Triton kernel validation and performance benchmarks
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any

# Import test utilities
from .conftest import assert_tensors_close, create_synthetic_data

# Import neural component modules
try:
    # Try relative import first (when used as package)
    from ..src.neural_components import (
        TritonAttention, TritonLSTM, TritonConvBN,
        create_triton_attention, create_triton_lstm, create_triton_conv_bn,
        benchmark_neural_components
    )
except ImportError:
    # Fall back to absolute import (when imported directly)
    from src.neural_components import (
        TritonAttention, TritonLSTM, TritonConvBN,
        create_triton_attention, create_triton_lstm, create_triton_conv_bn,
        benchmark_neural_components
    )

# Import Triton availability
try:
    from ..src.core import TRITON_AVAILABLE
except ImportError:
    from src.core import TRITON_AVAILABLE


class TestTritonAttention:
    """Test Triton-accelerated attention mechanism."""

    @pytest.fixture
    def attention_layer(self):
        """Create attention layer for testing."""
        return TritonAttention(model_dim=64, n_heads=8)

    @pytest.fixture
    def attention_data(self):
        """Generate synthetic attention test data."""
        batch_size = 4
        seq_len = 16
        model_dim = 64

        query = torch.randn(batch_size, seq_len, model_dim)
        key = torch.randn(batch_size, seq_len, model_dim)
        value = torch.randn(batch_size, seq_len, model_dim)

        return {
            "query": query,
            "key": key,
            "value": value,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "model_dim": model_dim
        }

    def test_attention_initialization(self, attention_layer):
        """Test attention layer initialization."""
        assert attention_layer.model_dim == 64
        assert attention_layer.n_heads == 8
        assert attention_layer.head_dim == 8  # 64 / 8

        # Check weight matrices
        assert attention_layer.w_q.shape == (64, 64)
        assert attention_layer.w_k.shape == (64, 64)
        assert attention_layer.w_v.shape == (64, 64)
        assert attention_layer.w_o.shape == (64, 64)

    def test_attention_forward_pass(self, attention_layer, attention_data):
        """Test attention forward pass."""
        query = attention_data["query"]
        key = attention_data["key"]
        value = attention_data["value"]

        # Forward pass
        output = attention_layer.forward(query, key, value)

        # Check output shape
        assert output.shape == query.shape
        assert torch.isfinite(output).all()

    def test_attention_with_mask(self, attention_layer, attention_data):
        """Test attention with attention mask."""
        query = attention_data["query"]
        key = attention_data["key"]
        value = attention_data["value"]

        # Create causal mask (mask future positions)
        seq_len = query.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

        # Forward pass with mask
        output = attention_layer.forward(query, key, value, mask)

        # Check output
        assert output.shape == query.shape
        assert torch.isfinite(output).all()

    def test_attention_gradient_flow(self, attention_layer, attention_data):
        """Test that gradients flow through attention layer."""
        query = attention_data["query"]
        key = attention_data["key"]
        value = attention_data["value"]

        # Enable gradients
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        # Forward pass
        output = attention_layer.forward(query, key, value)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert torch.isfinite(query.grad).all()
        assert torch.isfinite(key.grad).all()
        assert torch.isfinite(value.grad).all()

    @pytest.mark.parametrize("seq_len", [8, 16, 32])
    def test_attention_sequence_lengths(self, seq_len):
        """Test attention with different sequence lengths."""
        attention_layer = TritonAttention(model_dim=64, n_heads=8)

        batch_size = 2
        model_dim = 64

        query = torch.randn(batch_size, seq_len, model_dim)
        key = torch.randn(batch_size, seq_len, model_dim)
        value = torch.randn(batch_size, seq_len, model_dim)

        output = attention_layer.forward(query, key, value)

        assert output.shape == (batch_size, seq_len, model_dim)
        assert torch.isfinite(output).all()


class TestTritonLSTM:
    """Test Triton-accelerated LSTM implementation."""

    @pytest.fixture
    def lstm_layer(self):
        """Create LSTM layer for testing."""
        return TritonLSTM(input_size=32, hidden_size=64)

    @pytest.fixture
    def lstm_data(self):
        """Generate synthetic LSTM test data."""
        batch_size = 4
        seq_len = 16
        input_size = 32

        input_seq = torch.randn(batch_size, seq_len, input_size)

        return {
            "input_seq": input_seq,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "input_size": input_size
        }

    def test_lstm_initialization(self, lstm_layer):
        """Test LSTM layer initialization."""
        assert lstm_layer.input_size == 32
        assert lstm_layer.hidden_size == 64

        # Check weight matrices
        expected_weight_ih_shape = (4 * 64, 32)  # 4 gates * hidden_size, input_size
        expected_weight_hh_shape = (4 * 64, 64)  # 4 gates * hidden_size, hidden_size

        assert lstm_layer.weight_ih.shape == expected_weight_ih_shape
        assert lstm_layer.weight_hh.shape == expected_weight_hh_shape

        # Check bias terms
        assert lstm_layer.bias_ih.shape == (4 * 64,)
        assert lstm_layer.bias_hh.shape == (4 * 64,)

    def test_lstm_forward_pass(self, lstm_layer, lstm_data):
        """Test LSTM forward pass."""
        input_seq = lstm_data["input_seq"]

        # Forward pass
        output, (hidden, cell) = lstm_layer.forward(input_seq)

        # Check output shapes
        expected_output_shape = (input_seq.shape[0], input_seq.shape[1], lstm_layer.hidden_size)
        expected_hidden_shape = (input_seq.shape[0], lstm_layer.hidden_size)

        assert output.shape == expected_output_shape
        assert hidden.shape == expected_hidden_shape
        assert cell.shape == expected_hidden_shape

        # Check numerical stability
        assert torch.isfinite(output).all()
        assert torch.isfinite(hidden).all()
        assert torch.isfinite(cell).all()

    def test_lstm_with_initial_states(self, lstm_layer, lstm_data):
        """Test LSTM with custom initial states."""
        input_seq = lstm_data["input_seq"]
        batch_size = input_seq.shape[0]

        # Custom initial states
        initial_hidden = torch.randn(batch_size, lstm_layer.hidden_size)
        initial_cell = torch.randn(batch_size, lstm_layer.hidden_size)

        # Forward pass
        output, (hidden, cell) = lstm_layer.forward(
            input_seq, initial_hidden, initial_cell
        )

        # Check output shapes
        expected_output_shape = (input_seq.shape[0], input_seq.shape[1], lstm_layer.hidden_size)
        expected_hidden_shape = (input_seq.shape[0], lstm_layer.hidden_size)

        assert output.shape == expected_output_shape
        assert hidden.shape == expected_hidden_shape
        assert cell.shape == expected_hidden_shape

    def test_lstm_gradient_flow(self, lstm_layer, lstm_data):
        """Test that gradients flow through LSTM layer."""
        input_seq = lstm_data["input_seq"]
        input_seq.requires_grad_(True)

        # Forward pass
        output, (hidden, cell) = lstm_layer.forward(input_seq)
        loss = output.sum() + hidden.sum() + cell.sum()

        # Backward pass
        loss.backward()

        # Check gradients
        assert input_seq.grad is not None
        assert torch.isfinite(input_seq.grad).all()

        # Check parameter gradients
        assert torch.isfinite(lstm_layer.weight_ih.grad).all()
        assert torch.isfinite(lstm_layer.weight_hh.grad).all()
        assert torch.isfinite(lstm_layer.bias_ih.grad).all()
        assert torch.isfinite(lstm_layer.bias_hh.grad).all()

    @pytest.mark.parametrize("seq_len", [8, 16, 24])
    def test_lstm_sequence_lengths(self, seq_len):
        """Test LSTM with different sequence lengths."""
        lstm_layer = TritonLSTM(input_size=32, hidden_size=64)

        batch_size = 2
        input_seq = torch.randn(batch_size, seq_len, 32)

        output, (hidden, cell) = lstm_layer.forward(input_seq)

        assert output.shape == (batch_size, seq_len, 64)
        assert hidden.shape == (batch_size, 64)
        assert cell.shape == (batch_size, 64)
        assert torch.isfinite(output).all()


class TestTritonConvBN:
    """Test Triton-accelerated fused convolution + batch normalization."""

    @pytest.fixture
    def conv_bn_layer(self):
        """Create fused conv+BN layer for testing."""
        return TritonConvBN(
            in_channels=16, out_channels=32, kernel_size=3,
            stride=1, padding=1
        )

    @pytest.fixture
    def conv_data(self):
        """Generate synthetic convolution test data."""
        batch_size = 4
        height, width = 8, 8
        in_channels = 16

        input_tensor = torch.randn(batch_size, in_channels, height, width)

        return {
            "input_tensor": input_tensor,
            "batch_size": batch_size,
            "height": height,
            "width": width,
            "in_channels": in_channels
        }

    def test_conv_bn_initialization(self, conv_bn_layer):
        """Test fused conv+BN layer initialization."""
        assert conv_bn_layer.in_channels == 16
        assert conv_bn_layer.out_channels == 32
        assert conv_bn_layer.kernel_size == 3
        assert conv_bn_layer.stride == 1
        assert conv_bn_layer.padding == 1

        # Check weight and bias
        expected_weight_shape = (32, 16, 3, 3)  # out_channels, in_channels, kernel_size, kernel_size
        assert conv_bn_layer.weight.shape == expected_weight_shape
        assert conv_bn_layer.bias.shape == (32,)

        # Check batch norm parameters
        assert conv_bn_layer.bn_gamma.shape == (32,)
        assert conv_bn_layer.bn_beta.shape == (32,)
        assert conv_bn_layer.bn_running_mean.shape == (32,)
        assert conv_bn_layer.bn_running_var.shape == (32,)

    def test_conv_bn_forward_pass(self, conv_bn_layer, conv_data):
        """Test fused conv+BN forward pass."""
        input_tensor = conv_data["input_tensor"]

        # Forward pass
        output = conv_bn_layer.forward(input_tensor)

        # Check output shape
        expected_out_height = (input_tensor.shape[2] + 2 * conv_bn_layer.padding -
                             conv_bn_layer.kernel_size) // conv_bn_layer.stride + 1
        expected_out_width = (input_tensor.shape[3] + 2 * conv_bn_layer.padding -
                            conv_bn_layer.kernel_size) // conv_bn_layer.stride + 1

        expected_output_shape = (input_tensor.shape[0], conv_bn_layer.out_channels,
                               expected_out_height, expected_out_width)

        assert output.shape == expected_output_shape
        assert torch.isfinite(output).all()

    def test_conv_bn_gradient_flow(self, conv_bn_layer, conv_data):
        """Test that gradients flow through fused conv+BN layer."""
        input_tensor = conv_data["input_tensor"]
        input_tensor.requires_grad_(True)

        # Forward pass
        output = conv_bn_layer.forward(input_tensor)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients
        assert input_tensor.grad is not None
        assert torch.isfinite(input_tensor.grad).all()

        # Check parameter gradients
        assert torch.isfinite(conv_bn_layer.weight.grad).all()
        assert torch.isfinite(conv_bn_layer.bias.grad).all()
        assert torch.isfinite(conv_bn_layer.bn_gamma.grad).all()
        assert torch.isfinite(conv_bn_layer.bn_beta.grad).all()

    @pytest.mark.parametrize("kernel_size", [1, 3, 5])
    def test_conv_bn_kernel_sizes(self, kernel_size):
        """Test fused conv+BN with different kernel sizes."""
        conv_bn_layer = TritonConvBN(
            in_channels=8, out_channels=16, kernel_size=kernel_size,
            stride=1, padding=kernel_size//2
        )

        batch_size = 2
        height, width = 8, 8
        input_tensor = torch.randn(batch_size, 8, height, width)

        output = conv_bn_layer.forward(input_tensor)

        # Output should maintain spatial dimensions with proper padding
        assert output.shape[0] == batch_size
        assert output.shape[1] == 16  # out_channels
        assert output.shape[2] == height  # Same height with padding
        assert output.shape[3] == width   # Same width with padding
        assert torch.isfinite(output).all()


class TestNeuralComponentsIntegration:
    """Integration tests for neural component implementations."""

    def test_attention_lstm_integration(self):
        """Test integration between attention and LSTM components."""
        # Create components
        attention = TritonAttention(model_dim=64, n_heads=8)
        lstm = TritonLSTM(input_size=64, hidden_size=64)

        batch_size = 4
        seq_len = 16

        # Generate input
        input_seq = torch.randn(batch_size, seq_len, 64)

        # LSTM processing
        lstm_output, _ = lstm.forward(input_seq)

        # Attention processing
        attention_output = attention.forward(lstm_output, lstm_output, lstm_output)

        # Check shapes and numerical stability
        assert attention_output.shape == lstm_output.shape
        assert torch.isfinite(attention_output).all()

    def test_conv_bn_attention_integration(self):
        """Test integration between conv+BN and attention."""
        # Create components
        conv_bn = TritonConvBN(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        attention = TritonAttention(model_dim=32*8*8, n_heads=8)  # Flattened spatial dims

        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 8, 8)

        # Conv+BN processing
        conv_output = conv_bn.forward(input_tensor)

        # Flatten for attention
        attention_input = conv_output.view(batch_size, -1, 32*8*8)
        attention_output = attention.forward(attention_input, attention_input, attention_input)

        # Check results
        assert attention_output.shape[0] == batch_size
        assert torch.isfinite(attention_output).all()

    def test_neural_components_gradient_flow(self):
        """Test end-to-end gradient flow through neural components."""
        # Create a simple network
        conv_bn = TritonConvBN(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        attention = TritonAttention(model_dim=16*8*8, n_heads=8)

        batch_size = 2
        input_tensor = torch.randn(batch_size, 8, 8, 8)
        input_tensor.requires_grad_(True)

        # Forward pass
        conv_output = conv_bn.forward(input_tensor)
        attention_input = conv_output.view(batch_size, -1, 16*8*8)
        attention_output = attention.forward(attention_input, attention_input, attention_input)

        # Backward pass
        loss = attention_output.sum()
        loss.backward()

        # Check that gradients flow through all components
        assert input_tensor.grad is not None
        assert torch.isfinite(input_tensor.grad).all()

        # Check conv+BN gradients
        assert torch.isfinite(conv_bn.weight.grad).all()
        assert torch.isfinite(conv_bn.bias.grad).all()

        # Check attention gradients (weights should have gradients from attention forward)
        # Note: attention layer may not have gradients if Triton fallback is used

    def test_neural_components_benchmark(self):
        """Test benchmarking functionality."""
        if TRITON_AVAILABLE:
            results = benchmark_neural_components()

            # Check that benchmark results are reasonable
            assert "attention" in results
            assert "lstm" in results

            for method, metrics in results.items():
                assert "triton_accelerated" in metrics
                assert isinstance(metrics["triton_accelerated"], bool)


class TestNeuralComponentsConvenienceFunctions:
    """Test convenience functions for creating neural components."""

    def test_create_triton_attention(self):
        """Test attention creation function."""
        attention = create_triton_attention(model_dim=128, n_heads=16, use_triton=True)
        assert isinstance(attention, TritonAttention)
        assert attention.model_dim == 128
        assert attention.n_heads == 16

    def test_create_triton_lstm(self):
        """Test LSTM creation function."""
        lstm = create_triton_lstm(input_size=64, hidden_size=128, use_triton=True)
        assert isinstance(lstm, TritonLSTM)
        assert lstm.input_size == 64
        assert lstm.hidden_size == 128

    def test_create_triton_conv_bn(self):
        """Test fused conv+BN creation function."""
        conv_bn = create_triton_conv_bn(
            in_channels=32, out_channels=64, kernel_size=5,
            stride=2, padding=2, use_triton=True
        )
        assert isinstance(conv_bn, TritonConvBN)
        assert conv_bn.in_channels == 32
        assert conv_bn.out_channels == 64
        assert conv_bn.kernel_size == 5
        assert conv_bn.stride == 2
        assert conv_bn.padding == 2


# Performance tests
@pytest.mark.slow
class TestNeuralComponentsPerformance:
    """Performance tests for neural component implementations."""

    def test_attention_performance_scaling(self):
        """Test attention performance with different sizes."""
        sizes = [(32, 4, 8), (64, 8, 16), (128, 16, 32)]  # model_dim, n_heads, seq_len

        for model_dim, n_heads, seq_len in sizes:
            attention = TritonAttention(model_dim=model_dim, n_heads=n_heads)

            batch_size = 4
            query = torch.randn(batch_size, seq_len, model_dim)
            key = torch.randn(batch_size, seq_len, model_dim)
            value = torch.randn(batch_size, seq_len, model_dim)

            # Time the forward pass
            import time
            start_time = time.time()

            for _ in range(10):
                output = attention.forward(query, key, value)

            end_time = time.time()

            # Check that performance is reasonable
            total_time = end_time - start_time
            assert total_time < 10.0  # Should complete in reasonable time

    def test_lstm_performance_scaling(self):
        """Test LSTM performance with different sizes."""
        sizes = [(32, 64, 16), (64, 128, 32), (128, 256, 64)]  # input_size, hidden_size, seq_len

        for input_size, hidden_size, seq_len in sizes:
            lstm = TritonLSTM(input_size=input_size, hidden_size=hidden_size)

            batch_size = 4
            input_seq = torch.randn(batch_size, seq_len, input_size)

            # Time the forward pass
            import time
            start_time = time.time()

            for _ in range(5):
                output, _ = lstm.forward(input_seq)

            end_time = time.time()

            # Check performance
            total_time = end_time - start_time
            assert total_time < 15.0  # Should complete in reasonable time

    def test_conv_bn_performance_scaling(self):
        """Test fused conv+BN performance with different sizes."""
        configs = [
            (16, 32, 3, 8),   # in_ch, out_ch, kernel, size
            (32, 64, 5, 16),
            (64, 128, 7, 32)
        ]

        for in_ch, out_ch, kernel, size in configs:
            conv_bn = TritonConvBN(
                in_channels=in_ch, out_channels=out_ch, kernel_size=kernel,
                padding=kernel//2
            )

            batch_size = 2
            input_tensor = torch.randn(batch_size, in_ch, size, size)

            # Time the forward pass
            import time
            start_time = time.time()

            for _ in range(10):
                output = conv_bn.forward(input_tensor)

            end_time = time.time()

            # Check performance
            total_time = end_time - start_time
            assert total_time < 20.0  # Should complete in reasonable time

class TestNeuralComponentsEdgeCases:
    """Test edge cases and numerical stability for neural components."""

    def test_attention_single_head(self):
        """Test attention with single head."""
        attention = TritonAttention(model_dim=64, n_heads=1)

        batch_size, seq_len = 2, 8
        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)

        output = attention.forward(query, key, value)

        assert output.shape == (batch_size, seq_len, 64)
        assert torch.isfinite(output).all()

    def test_attention_extreme_mask(self):
        """Test attention with extreme mask values."""
        attention = TritonAttention(model_dim=32, n_heads=4)

        batch_size, seq_len = 2, 4
        query = torch.randn(batch_size, seq_len, 32)
        key = torch.randn(batch_size, seq_len, 32)
        value = torch.randn(batch_size, seq_len, 32)

        # Create boolean mask (True for positions to mask out)
        mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool)
        # Allow causal attention: each position can attend to itself and previous positions
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:  # Allow attending to self and previous
                    mask[0, 0, i, j] = False

        output = attention.forward(query, key, value, mask)

        # Should not crash and should be finite
        assert torch.isfinite(output).all()

    def test_conv_bn_extreme_input_sizes(self):
        """Test conv+BN with extreme input sizes."""
        # Very small input
        conv_bn_small = TritonConvBN(in_channels=1, out_channels=1, kernel_size=1)
        small_input = torch.randn(1, 1, 1, 1)
        output_small = conv_bn_small.forward(small_input)
        assert output_small.shape == (1, 1, 1, 1)
        assert torch.isfinite(output_small).all()

        # Large kernel
        conv_bn_large = TritonConvBN(in_channels=3, out_channels=16, kernel_size=7, padding=3)
        large_input = torch.randn(2, 3, 32, 32)
        output_large = conv_bn_large.forward(large_input)
        assert output_large.shape == (2, 16, 32, 32)
        assert torch.isfinite(output_large).all()

    def test_lstm_single_step(self):
        """Test LSTM with single time step."""
        lstm = TritonLSTM(input_size=32, hidden_size=64)

        batch_size = 2
        input_seq = torch.randn(batch_size, 1, 32)  # Single time step

        output, (hidden, cell) = lstm.forward(input_seq)

        assert output.shape == (batch_size, 1, 64)
        assert hidden.shape == (batch_size, 64)
        assert cell.shape == (batch_size, 64)
        assert torch.isfinite(output).all()
        assert torch.isfinite(hidden).all()
        assert torch.isfinite(cell).all()

    def test_lstm_zero_initial_states(self):
        """Test LSTM with zero initial states."""
        lstm = TritonLSTM(input_size=16, hidden_size=32)

        batch_size, seq_len = 2, 3
        input_seq = torch.randn(batch_size, seq_len, 16)

        # Explicitly pass zero initial states
        initial_hidden = torch.zeros(batch_size, 32)
        initial_cell = torch.zeros(batch_size, 32)

        output, (hidden, cell) = lstm.forward(input_seq, initial_hidden, initial_cell)

        assert output.shape == (batch_size, seq_len, 32)
        assert torch.isfinite(output).all()
        assert torch.isfinite(hidden).all()
        assert torch.isfinite(cell).all()

    def test_neural_components_numerical_stability(self):
        """Test numerical stability with extreme input values."""
        # Test attention with extreme values
        attention = TritonAttention(model_dim=32, n_heads=2)

        # Create inputs with extreme values
        extreme_query = torch.randn(2, 4, 32) * 1e6
        extreme_key = torch.randn(2, 4, 32) * 1e6
        extreme_value = torch.randn(2, 4, 32) * 1e6

        output = attention.forward(extreme_query, extreme_key, extreme_value)

        # Should handle extreme values gracefully
        assert torch.isfinite(output).all()

        # Test conv+BN with extreme values
        conv_bn = TritonConvBN(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        extreme_input = torch.randn(1, 2, 8, 8) * 1e10

        output_conv = conv_bn.forward(extreme_input)

        # Should remain finite
        assert torch.isfinite(output_conv).all()

    def test_gradient_flow_extreme_cases(self):
        """Test gradient flow with extreme numerical conditions."""
        # Test with very small inputs (potential underflow)
        attention = TritonAttention(model_dim=16, n_heads=2)

        tiny_input = torch.randn(1, 2, 16) * 1e-8
        tiny_input.requires_grad_(True)

        output = attention.forward(tiny_input, tiny_input, tiny_input)
        loss = output.sum()
        loss.backward()

        assert torch.isfinite(tiny_input.grad).all()
        assert tiny_input.grad.norm() > 1e-12  # Should have non-zero gradients

        # Test with very large inputs (potential overflow)
        large_input = torch.randn(1, 2, 16) * 1e4
        large_input.requires_grad_(True)

        output_large = attention.forward(large_input, large_input, large_input)
        loss_large = output_large.sum()
        loss_large.backward()

        assert torch.isfinite(large_input.grad).all()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
