# Active Inference and Free Energy Principle

This module provides GPU-accelerated implementations of active inference methods using Triton for high-performance computation of variational free energy, expected free energy, and message passing algorithms.

## Overview

Active inference is a framework for understanding perception, action, and learning in biological and artificial systems. It is grounded in the free energy principle, which states that biological systems minimize variational free energy to maintain their integrity.

This implementation leverages Triton's GPU acceleration capabilities to provide efficient computation of:
- **Variational Free Energy**: Measures the difference between internal models and sensory data
- **Expected Free Energy**: Guides action selection by evaluating policies
- **Message Passing**: Enables efficient inference in probabilistic graphical models
- **Bayesian Inference**: Provides probabilistic reasoning capabilities

## Features

### Core Components

1. **TritonFeatureManager**: Comprehensive management of all Triton features
2. **GPUAccelerator**: High-level GPU acceleration interface
3. **ActiveInferenceEngine**: Main engine for active inference computations
4. **VariationalFreeEnergy**: GPU-accelerated VFE computation and minimization
5. **ExpectedFreeEnergy**: Policy evaluation using expected free energy
6. **MessagePassing**: Belief propagation and variational message passing
7. **BayesianInference**: Bayesian inference methods

### Key Capabilities

- **Complete Triton Integration**: Configure, run, verify, report, and visualize all Triton features
- **GPU Acceleration**: Optimized kernels for high-performance computing
- **Modular Design**: Clean separation of concerns with thin orchestrators
- **Comprehensive Testing**: TDD approach with extensive test coverage
- **Real Data Analysis**: No mock methods - all computations use real data

## Architecture

```
active_inference/
├── src/                    # Core implementation modules
│   ├── __init__.py        # Main package exports
│   ├── core.py            # Triton feature management & GPU acceleration
│   ├── inference.py       # Active inference & Bayesian methods
│   ├── free_energy.py     # Variational & expected free energy
│   └── message_passing.py # Message passing algorithms
├── tests/                 # Comprehensive test suite
│   ├── __init__.py        # Test utilities & fixtures
│   ├── conftest.py        # Pytest configuration
│   ├── test_core.py       # Core functionality tests
│   ├── test_inference.py  # Active inference tests
│   ├── test_free_energy.py # Free energy tests
│   └── test_message_passing.py # Message passing tests
├── docs/                  # Documentation
│   ├── README.md          # This file
│   ├── api.md             # API documentation
│   ├── examples.md        # Usage examples
│   └── theory.md          # Theoretical background
├── examples/              # Thin orchestrators
│   ├── basic_inference.py     # Basic active inference demo
│   ├── free_energy_minimization.py # Free energy minimization
│   ├── policy_selection.py    # Expected free energy demo
│   └── message_passing_demo.py # Message passing examples
└── pyproject.toml         # Package configuration
```

## Quick Start

### Run Examples and Tests

Run the comprehensive example suite to see all framework capabilities:

```bash
cd active_inference
python run_all_examples.py
```

Run the complete test suite to validate all functionality:

```bash
python run_all_tests.py
```

Both scripts generate detailed reports and visualizations of framework performance.

### Basic Usage

```python
import torch
from active_inference import ActiveInferenceEngine, TritonFeatureManager

# Initialize feature manager
feature_manager = TritonFeatureManager()

# Create active inference engine
engine = ActiveInferenceEngine(feature_manager)

# Register a simple model
model_spec = {
    'variables': {
        'hidden': {'shape': (10,), 'type': 'continuous'},
        'observed': {'shape': (5,), 'type': 'continuous'}
    }
}
engine.register_model('my_model', model_spec)

# Generate some observations
observations = torch.randn(10, 5)

# Compute variational free energy
free_energy = engine.compute_free_energy(observations, 'my_model')
print(f"Free Energy: {free_energy}")
```

### GPU Acceleration

```python
from active_inference import GPUAccelerator, VariationalFreeEnergy

# Set up GPU acceleration
gpu_accelerator = GPUAccelerator(feature_manager)

# Allocate tensors on GPU
tensors = gpu_accelerator.allocate_tensors([(100, 50), (50, 20)])

# Use variational free energy with GPU acceleration
vfe_engine = VariationalFreeEnergy(feature_manager)

# Minimize free energy
optimized_posterior = vfe_engine.minimize(
    observations, initial_posterior, prior, likelihood,
    num_iterations=100, learning_rate=0.01
)
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Triton (latest version)
- CUDA-compatible GPU (recommended)

### Setup

1. Install uv (if not already installed):

```bash
# Using pip
pip install uv

# Or using Homebrew (macOS/Linux)
brew install uv

# Or using cargo (if you have Rust installed)
cargo install uv
```

2. Clone the repository and navigate to the active_inference directory

3. Install dependencies using uv:

```bash
# Install core dependencies
uv sync

# Install with development dependencies
uv sync --dev

# Install Triton (optional, for GPU acceleration)
uv add triton

# Or install everything at once
uv sync --dev && uv add triton
```

4. Run tests to verify installation:

```bash
# Run tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest --cov=src tests/
```

## Development

### Testing

The module follows a test-driven development (TDD) approach with comprehensive test coverage:

```bash
# Run all tests
uv run pytest tests/

# Run specific test modules
uv run pytest tests/test_core.py
uv run pytest tests/test_inference.py

# Run with coverage
uv run pytest --cov=src tests/

# Run tests with uv script (defined in pyproject.toml)
uv run test
uv run test-coverage

# Run examples
uv run python run_all_examples.py
uv run examples

# Run validation
uv run python validate_triton_usage.py
uv run validate
```

### Code Quality

- **Modular Design**: Clean separation of concerns
- **Documentation**: Comprehensive docstrings and comments
- **Type Hints**: Full type annotation coverage
- **Real Data**: No mock methods - all analysis uses real data
- **Performance**: GPU-accelerated implementations

### Contributing

1. Follow the established TDD workflow
2. Ensure all tests pass before submitting
3. Add documentation for new features
4. Maintain code quality standards

## Theoretical Background

### Free Energy Principle

The free energy principle states that biological systems minimize variational free energy:

```
F = E_q[log q(z) - log p(x,z)]
```

Where:
- `q(z)` is the variational posterior
- `p(x,z)` is the generative model
- `E_q[...]` denotes expectation under q

### Active Inference

Active inference combines perception and action through expected free energy minimization:

```
G_π = E_q[log p(o|π) + log p(π) - log q(o|π)] - C
```

Where:
- `π` represents policies (action sequences)
- `o` represents observations
- `C` represents preferences/goals

### Message Passing

Efficient inference in probabilistic graphical models using:
- **Belief Propagation**: Exact inference on tree-structured graphs
- **Variational Message Passing**: Approximate inference on loopy graphs

## Performance

### Benchmarks

- GPU acceleration provides 10-100x speedup over CPU implementations
- Efficient memory usage through Triton's optimization features
- Scalable to large models and datasets

### Optimization Features

- **Shared Memory**: Maximizes GPU memory bandwidth
- **Vectorization**: Efficient SIMD operations
- **Pipelining**: Overlapping computation and memory access
- **Fusion**: Combining multiple operations into single kernels

## License

This project follows the same license as the main Triton repository.

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Parr, T., & Friston, K. J. (2017). Working memory, attention, and salience in active inference
3. Buckley, C. L., et al. (2017). The free energy principle for action and perception
4. Triton documentation: GPU-accelerated deep learning compilation

## Support

For questions and support:
- Check the examples/ directory for usage patterns
- Review the test files for implementation details
- Consult the Triton documentation for GPU optimization guidance
