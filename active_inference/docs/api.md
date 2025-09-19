# API Documentation

Complete API reference for the Active Inference and Free Energy Principle module.

## Analysis and Visualization

### ComprehensiveAnalysisRunner

Main class for running comprehensive analysis of all framework components with extensive visualization and reporting.

#### Methods

**`__init__()`**
- Initializes the analysis runner with output directories
- Creates plots/, reports/, and data/ subdirectories
- Sets up comprehensive logging

**`run_core_gpu_analysis()`**
- Runs GPU performance benchmarking and memory analysis
- Generates performance scaling plots and memory usage histograms
- Returns: Performance metrics and capability analysis

**`run_free_energy_analysis()`**
- Analyzes variational and expected free energy methods
- Tests different problem scales and scenarios
- Generates convergence plots and performance comparisons

**`run_bayesian_inference_analysis()`**
- Evaluates Bayesian inference dynamics across scenarios
- Creates belief evolution visualizations and entropy analysis
- Returns: Inference performance metrics and dynamics data

**`run_message_passing_analysis()`**
- Tests message passing algorithms on different graph structures
- Generates convergence analysis and performance comparisons
- Returns: Algorithm performance data and graph analysis

**`run_pomdp_analysis()`**
- Evaluates POMDP active inference in grid world scenarios
- Creates belief tracking visualizations and policy analysis
- Returns: POMDP performance metrics and behavior analysis

**`run_policy_selection_analysis()`**
- Analyzes policy selection via expected free energy
- Generates EFE distribution plots and decision dynamics
- Returns: Policy evaluation results and selection analysis

**`run_all_analyses()`**
- Executes complete analysis suite
- Generates HTML reports and saves all visualizations
- Returns: Comprehensive results dictionary

**`save_visualization(fig: plt.Figure, name: str, description: str)`**
- Saves matplotlib figure with metadata
- Parameters:
  - `fig`: Matplotlib figure object
  - `name`: Filename (without extension)
  - `description`: Human-readable description

**`log_operation(operation: str, details: Dict[str, Any], success: bool = True)`**
- Logs analysis operations with timestamps
- Parameters:
  - `operation`: Operation name
  - `details`: Operation details dictionary
  - `success`: Whether operation succeeded

## Core Classes

### TritonFeatureManager

Comprehensive manager for all Triton features and GPU acceleration.

#### Methods

**`__init__(config: Optional[TritonFeatureConfig] = None)`**
- Initializes the feature manager with optional configuration
- Sets up Triton environment for optimal performance

**`register_kernel(name: str, kernel_fn: Any, metadata: Dict[str, Any])`**
- Registers a Triton kernel with metadata
- Parameters:
  - `name`: Unique kernel identifier
  - `kernel_fn`: Triton kernel function
  - `metadata`: Dictionary with 'input_shapes', 'output_shapes', 'optimizations'

**`get_kernel(name: str) -> Dict[str, Any]`**
- Retrieves registered kernel information
- Returns: Dictionary containing kernel function and metadata

**`list_features() -> Dict[str, List[str]]`**
- Lists all available Triton features and capabilities
- Returns: Dictionary with 'kernels', 'devices', 'dtypes', 'optimizations'

**`verify_feature(feature_name: str) -> bool`**
- Verifies that a Triton feature is available and functional
- Returns: True if feature is properly configured

**`profile_kernel(kernel_name: str, *args, **kwargs) -> Dict[str, Any]`**
- Profiles kernel performance with given arguments
- Returns: Profiling data including timing and metrics

### GPUAccelerator

High-level interface for GPU acceleration and memory management.

#### Methods

**`__init__(feature_manager: TritonFeatureManager)`**
- Initializes GPU accelerator with feature manager

**`allocate_tensors(shapes: List[torch.Size], dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]`**
- Allocates GPU tensors with optimal memory layout
- Parameters:
  - `shapes`: List of tensor shapes to allocate
  - `dtype`: Data type (defaults to feature manager config)
- Returns: List of allocated tensors

**`synchronize()`**
- Synchronizes GPU operations
- Blocks until all GPU operations complete

**`get_memory_stats() -> Dict[str, Any]`**
- Retrieves current GPU memory statistics
- Returns: Dictionary with 'allocated', 'reserved', 'max_allocated' (if CUDA available)

## Active Inference

### ActiveInferenceEngine

Core engine for active inference computations.

#### Methods

**`__init__(feature_manager: Optional[TritonFeatureManager] = None)`**
- Initializes active inference engine

**`register_model(name: str, model_spec: Dict[str, Any])`**
- Registers a generative model for inference
- Parameters:
  - `name`: Model identifier
  - `model_spec`: Model specification dictionary

**`compute_free_energy(observations: torch.Tensor, model_name: str) -> torch.Tensor`**
- Computes variational free energy for given observations
- Parameters:
  - `observations`: [batch_size, feature_dim] tensor
  - `model_name`: Registered model name
- Returns: Free energy values [batch_size]

**`update_posterior(observations: torch.Tensor, model_name: str, learning_rate: float = 0.01)`**
- Updates variational posterior using gradient descent
- Parameters:
  - `observations`: [batch_size, feature_dim] tensor
  - `model_name`: Registered model name
  - `learning_rate`: Gradient descent step size

**`predict(model_name: str, num_samples: int = 100) -> torch.Tensor`**
- Generates predictions using learned posterior
- Parameters:
  - `model_name`: Registered model name
  - `num_samples`: Number of prediction samples
- Returns: Predictions [num_samples, feature_dim]

### BayesianInference

Bayesian inference methods for active inference.

#### Methods

**`__init__(engine: ActiveInferenceEngine)`**
- Initializes Bayesian inference with active inference engine

**`compute_posterior(prior: torch.Tensor, likelihood: torch.Tensor) -> torch.Tensor`**
- Computes posterior using Bayes rule: p(z|x) ∝ p(x|z) * p(z)
- Parameters:
  - `prior`: Prior distribution [batch_size, num_states]
  - `likelihood`: Likelihood [batch_size, num_states]
- Returns: Posterior distribution [batch_size, num_states]

**`compute_evidence(prior: torch.Tensor, likelihood: torch.Tensor) -> torch.Tensor`**
- Computes marginal likelihood p(x) = ∫ p(x|z) p(z) dz
- Parameters:
  - `prior`: Prior distribution [batch_size, num_states]
  - `likelihood`: Likelihood [batch_size, num_states]
- Returns: Evidence values [batch_size]

## Free Energy Methods

### VariationalFreeEnergy

GPU-accelerated variational free energy computation and minimization.

#### Methods

**`__init__(feature_manager: Optional[TritonFeatureManager] = None)`**
- Initializes variational free energy engine

**`compute(observations: torch.Tensor, posterior: torch.Tensor, prior: torch.Tensor, likelihood: torch.Tensor) -> torch.Tensor`**
- Computes variational free energy F = E_q[log q(z) - log p(x,z)]
- Parameters:
  - `observations`: [batch_size, feature_dim]
  - `posterior`: Variational posterior q(z) [batch_size, feature_dim]
  - `prior`: Prior p(z) [batch_size, feature_dim]
  - `likelihood`: Likelihood p(x|z) [batch_size, feature_dim]
- Returns: Free energy [batch_size]

**`minimize(observations: torch.Tensor, initial_posterior: torch.Tensor, prior: torch.Tensor, likelihood: torch.Tensor, num_iterations: int = 100, learning_rate: float = 0.01) -> torch.Tensor`**
- Minimizes variational free energy using gradient descent
- Returns: Optimized posterior

### ExpectedFreeEnergy

Expected free energy computation for policy selection.

#### Methods

**`__init__(feature_manager: Optional[TritonFeatureManager] = None)`**
- Initializes expected free energy engine

**`compute(observations: torch.Tensor, policies: torch.Tensor, posterior: torch.Tensor, preferences: Optional[torch.Tensor] = None) -> torch.Tensor`**
- Computes expected free energy for policy evaluation
- Parameters:
  - `observations`: [batch_size, feature_dim]
  - `policies`: Available policies [num_policies, feature_dim]
  - `posterior`: Current beliefs [batch_size, feature_dim]
  - `preferences`: Preference values [batch_size] (optional)
- Returns: EFE for each policy [batch_size, num_policies]

**`select_policy(EFE: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`**
- Selects policy that minimizes expected free energy
- Parameters:
  - `EFE`: Expected free energy matrix [batch_size, num_policies]
- Returns: (policy_indices [batch_size], EFE_values [batch_size])

## Message Passing

### MessagePassing (Base Class)

Base class for message passing algorithms.

#### Methods

**`set_graph(adjacency_matrix: torch.Tensor, node_potentials: Optional[torch.Tensor] = None, edge_potentials: Optional[torch.Tensor] = None)`**
- Sets up the graphical model structure
- Parameters:
  - `adjacency_matrix`: [num_nodes, num_nodes] connectivity
  - `node_potentials`: [num_nodes, num_states] node potentials
  - `edge_potentials`: Edge potential matrices (optional)

**`initialize_messages() -> torch.Tensor`**
- Initializes messages uniformly
- Returns: Initial messages [num_edges, num_states]

### BeliefPropagation

Belief propagation for exact inference in tree-structured graphs.

#### Methods

**`run(max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[str, torch.Tensor]`**
- Runs belief propagation algorithm
- Returns: Dictionary with 'beliefs', 'messages', 'converged'

### VariationalMessagePassing

Variational message passing for approximate inference in loopy graphs.

#### Methods

**`run(max_iterations: int = 50, learning_rate: float = 0.1) -> Dict[str, torch.Tensor]`**
- Runs variational message passing
- Returns: Dictionary with 'beliefs', 'free_energy'

## Configuration

### TritonFeatureConfig

Configuration class for Triton feature usage.

#### Attributes

- `device: str = "cuda"` - Target device
- `dtype: torch.dtype = torch.float32` - Default data type
- `max_shared_memory: int = 49152` - Maximum shared memory (bytes)
- `num_warps: int = 4` - Number of warps per thread block
- `num_stages: int = 3` - Number of pipeline stages
- `enable_fp8: bool = False` - Enable FP8 precision
- `enable_bf16: bool = True` - Enable BFloat16 precision
- `enable_tf32: bool = True` - Enable TF32 precision

## Utility Functions

### Test Utilities (from tests/__init__.py)

**`setup_test_environment()`**
- Configures test environment with proper settings

**`create_test_tensor(*shape, dtype=None, device=None, requires_grad=False)`**
- Creates test tensor with standard configuration

**`assert_tensors_close(actual, expected, rtol=1e-6, atol=1e-6)`**
- Asserts that two tensors are close within tolerance

**`create_synthetic_data(num_samples=100, feature_dim=10, num_classes=5)`**
- Creates synthetic data for testing
- Returns: features, labels, centers

## Error Handling

All methods raise appropriate exceptions for invalid inputs:

- `ValueError`: Invalid parameters or unregistered components
- `RuntimeError`: GPU/CUDA related errors
- `AssertionError`: Test assertion failures

## Performance Considerations

### Memory Management
- Use `GPUAccelerator.allocate_tensors()` for optimal memory layout
- Call `GPUAccelerator.synchronize()` when timing is critical
- Monitor memory usage with `GPUAccelerator.get_memory_stats()`

### Kernel Optimization
- Triton kernels are automatically optimized for the target hardware
- Use appropriate block sizes and shared memory configurations
- Profile kernels with `TritonFeatureManager.profile_kernel()`

### Batch Processing
- Process data in optimal batch sizes for GPU utilization
- Use asynchronous operations when possible
- Consider memory constraints for large models

## Examples

See the `examples/` directory for complete usage examples:

- `basic_inference.py` - Basic active inference workflow
- `free_energy_minimization.py` - Free energy minimization
- `policy_selection.py` - Policy selection with expected free energy
- `message_passing_demo.py` - Message passing algorithms
