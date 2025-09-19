# Active Inference Framework Examples

This directory contains comprehensive examples demonstrating all capabilities of the Active Inference Framework. Each example showcases different aspects of active inference implementation, from basic concepts to advanced applications.

## Examples Overview

### Core Active Inference Demos

| Example | Description | Key Methods | Complexity |
|---------|-------------|-------------|------------|
| [**basic_inference.py**](basic_inference.py) | Fundamental active inference demonstration with variational free energy minimization | Variational Free Energy, Bayesian inference, Generative models | Beginner |
| [**basic_triton_demo.py**](basic_triton_demo.py) | Basic Triton operations for Apple Silicon with MPS acceleration | Triton kernels, Vector operations, GPU acceleration | Beginner |
| [**bayesian_core_demo.py**](bayesian_core_demo.py) | Complete Bayesian inference core functionalities with logging and validation | Bayesian updating, Information theory, Free energy computation | Intermediate |
| [**free_energy_minimization.py**](free_energy_minimization.py) | Variational free energy minimization across different problem types | Gradient optimization, Different problem types, Convergence monitoring | Intermediate |
| [**message_passing_demo.py**](message_passing_demo.py) | Belief propagation and variational message passing on graph structures | Belief propagation, Message passing, Graph algorithms | Advanced |
| [**policy_selection.py**](policy_selection.py) | Policy selection using expected free energy minimization | Expected free energy, Policy optimization, Decision making | Advanced |
| [**pomdp_gridworld.py**](pomdp_gridworld.py) | POMDP gridworld with Active Inference using VFE and EFE | POMDP, Gridworld navigation, State estimation, Policy selection | Advanced |
| [**triton_active_inference_demo.py**](triton_active_inference_demo.py) | Triton-accelerated active inference agent with belief updating | Triton integration, Active inference agent, Real-time inference | Intermediate |
| [**triton_benchmark.py**](triton_benchmark.py) | Performance benchmarking of Triton vs PyTorch operations | Performance analysis, Benchmarking, Optimization recommendations | Advanced |
| [**triton_improvements_demo.py**](triton_improvements_demo.py) | Showcase of all Triton integration improvements and enhancements | Enhanced kernels, Error handling, Performance monitoring | Advanced |
| [**visualization_demo.py**](visualization_demo.py) | Comprehensive visualization capabilities for all framework outputs | Data visualization, Performance dashboards, Belief evolution plots | Intermediate |

### Example Categories

#### 🎯 **Active Inference Fundamentals**
- `basic_inference.py` - Core active inference concepts
- `free_energy_minimization.py` - Free energy optimization
- `policy_selection.py` - Decision making and policy selection

#### 🧠 **Bayesian Inference & Learning**
- `bayesian_core_demo.py` - Bayesian inference operations
- `message_passing_demo.py` - Probabilistic graphical models
- `pomdp_gridworld.py` - POMDP with Active Inference
- `belief_tracking` - Belief state evolution

#### ⚡ **GPU Acceleration & Performance**
- `basic_triton_demo.py` - Basic Triton operations
- `triton_active_inference_demo.py` - Triton-accelerated active inference
- `triton_benchmark.py` - Performance benchmarking
- `triton_improvements_demo.py` - Advanced Triton features

#### 📊 **Visualization & Analysis**
- `visualization_demo.py` - Comprehensive visualization suite

## Quick Start

### Running Individual Examples

Each example can be run independently:

```bash
# Basic active inference demonstration
python basic_inference.py

# Triton performance benchmarking
python triton_benchmark.py

# Comprehensive visualization demo
python visualization_demo.py
```

### Running All Examples

For comprehensive testing and validation, use the main examples runner:

```bash
# From the active_inference directory
python run_all_examples.py
```

This will:
- Execute all examples in sequence
- Generate comprehensive reports
- Create performance visualizations
- Provide detailed logging and error analysis

## Example Details

### basic_inference.py
**Core Active Inference Demonstration**
- Demonstrates fundamental active inference workflow
- Variational free energy minimization
- Bayesian inference and belief updating
- Linear generative model with Gaussian likelihood
- GPU acceleration via MPS
- Comprehensive validation and visualization

### basic_triton_demo.py
**Basic Triton Operations for Apple Silicon**
- Essential Triton kernels for vector operations
- MPS acceleration on Apple Silicon
- Performance comparison with PyTorch
- Error handling and fallback mechanisms
- Integration with active inference computations

### bayesian_core_demo.py
**Bayesian Inference Core**
- Complete Bayesian belief updating
- Information theoretic calculations (entropy, KL divergence)
- Variational free energy computation
- Comprehensive logging and validation
- Real-time performance monitoring
- Detailed reporting and visualization

### free_energy_minimization.py
**Variational Free Energy Minimization**
- Optimization across different problem types:
  - Simple unimodal problems
  - Conflicting prior-likelihood scenarios
  - Multimodal posterior problems
- Real-time convergence monitoring
- Gradient-based optimization
- GPU-accelerated computation
- Performance analysis and visualization

### message_passing_demo.py
**Message Passing Algorithms**
- Belief propagation on tree structures
- Variational message passing on loopy graphs
- Different graph topologies (chain, grid, random)
- Algorithm comparison and validation
- GPU acceleration for large graphs
- Comprehensive performance analysis

### policy_selection.py
**Policy Selection with Expected Free Energy**
- Expected free energy computation for policy evaluation
- Multi-agent decision making scenarios
- Exploration-exploitation trade-off analysis
- Policy optimization and selection
- Real-time decision making simulation
- Performance benchmarking

### pomdp_gridworld.py
**POMDP Gridworld with Active Inference**
- Partially Observable Markov Decision Process (POMDP) implementation
- 10x10 gridworld with 5 temperature states per cell (500 total states)
- Variational Free Energy (VFE) for state estimation from noisy observations
- Expected Free Energy (EFE) for policy selection and navigation
- GPU-accelerated Bayesian belief updates using Triton
- Homeostatic temperature preference (optimal state = 2)
- 5 observation types representing noisy temperature readings
- 4 actions for grid navigation (up, down, left, right)
- Comprehensive trajectory tracking and performance analysis
- Real-time belief entropy monitoring and position estimation

### triton_active_inference_demo.py
**Triton-Accelerated Active Inference**
- Real-time active inference agent
- Triton-accelerated belief updating
- Policy selection with expected free energy
- Graceful fallback to PyTorch
- Performance monitoring and optimization
- Apple Silicon MPS acceleration

### triton_benchmark.py
**Performance Benchmarking**
- Comprehensive Triton vs PyTorch benchmarking
- Different problem sizes and configurations
- Performance analysis and recommendations
- Platform-specific optimizations
- Memory usage tracking
- Optimization suggestions

### triton_improvements_demo.py
**Triton Integration Enhancements**
- All Triton integration improvements
- Enhanced kernel implementations
- Advanced error handling and recovery
- Comprehensive logging system
- Performance analysis tools
- Apple Silicon optimizations
- Memory management improvements

### visualization_demo.py
**Comprehensive Visualization Suite**
- Performance dashboards
- Belief evolution heatmaps
- Statistical analysis plots
- Network topology visualizations
- Correlation matrices
- Timeline and dependency graphs
- Triton ecosystem analysis

## Dependencies

All examples require:
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib (for visualization)
- NetworkX (for graph visualizations)

Optional (for full functionality):
- Triton (for GPU acceleration)
- Seaborn (enhanced visualizations)

## Output Files

Examples generate various output files in the `outputs/` directory:

### Reports and Logs
- `comprehensive_examples_report.json` - Complete execution results
- `triton_benchmark_report.json` - Performance benchmarking data
- Various JSON logs for individual examples

### Visualizations
- `examples_performance_summary.png` - Overall performance dashboard
- `free_energy_optimization_examples.png` - Free energy optimization analysis
- `belief_evolution_analysis.png` - Belief state evolution
- Various PNG files for specific analyses

### Data Exports
- CSV files with performance metrics
- JSON data exports for further analysis

## Platform Support

### Apple Silicon (M1/M2/M3)
- Full MPS acceleration support
- Optimized Triton kernels
- Platform-specific error handling
- Memory management optimizations

### CUDA GPUs
- Full CUDA acceleration
- Advanced Triton kernel compilation
- Performance optimizations

### CPU-Only
- PyTorch CPU operations
- Graceful degradation
- Comprehensive fallback mechanisms

## Troubleshooting

### Common Issues

1. **Triton Not Available**
   - Examples will automatically fall back to PyTorch
   - Some GPU acceleration features will be unavailable
   - Performance may be reduced

2. **MPS Not Available (Intel Macs)**
   - Framework will use CPU operations
   - All functionality remains available
   - Performance will be CPU-limited

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify virtual environment activation

### Performance Optimization

- Use `triton_benchmark.py` to identify bottlenecks
- Enable Triton for maximum performance
- Use MPS on Apple Silicon for best results
- Monitor memory usage with logging features

## Development

When adding new examples:

1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Add error handling and logging
4. Generate appropriate visualizations
5. Update this README with new entries
6. Ensure compatibility with `run_all_examples.py`

## Contributing

Examples should demonstrate:
- Real active inference computations (no mocks)
- Comprehensive error handling
- Performance monitoring and logging
- Clear documentation and comments
- Platform compatibility
- GPU acceleration where applicable

See the main framework documentation for development guidelines.
