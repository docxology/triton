# Active Inference Examples

Comprehensive collection of thin orchestrators demonstrating complete Triton feature coverage for active inference and free energy principle methods. All examples follow the standardized pattern using the `active_inference` package.

## Setup

Before running the examples, ensure you have the active inference framework properly installed:

### Install uv

```bash
# Using pip
pip install uv

# Or using Homebrew (macOS/Linux)
brew install uv

# Or using cargo (if you have Rust installed)
cargo install uv
```

### Install Dependencies

```bash
# Navigate to the active_inference directory
cd active_inference

# Install core dependencies
uv sync

# Install with development dependencies
uv sync --dev

# Install Triton (optional, for GPU acceleration)
uv add triton

# Or install everything at once
uv sync --dev && uv add triton
```

### Verify Installation

```bash
# Run tests to verify installation
uv run pytest tests/ -v

# Run the comprehensive example suite
uv run python run_all_examples.py

# Or use the uv script
uv run examples
```

## Basic Active Inference

### Simple Generative Model

```python
import torch
import active_inference as ai

# Initialize components
feature_manager = ai.TritonFeatureManager()
engine = ai.ActiveInferenceEngine(feature_manager)

# Define a simple generative model
model_spec = {
    'name': 'perception_model',
    'variables': {
        'hidden_states': {
            'shape': (10,),
            'type': 'continuous',
            'description': 'Latent causes of observations'
        },
        'observations': {
            'shape': (5,),
            'type': 'continuous',
            'description': 'Sensory data'
        }
    },
    'likelihood': 'gaussian',
    'prior': 'normal',
    'precision': 1.0
}

# Register the model
engine.register_model('perception_model', model_spec)

# Generate synthetic observations
batch_size = 32
observations = torch.randn(batch_size, 5)

# Compute variational free energy
free_energy = engine.compute_free_energy(observations, 'perception_model')
print(f"Variational Free Energy: {free_energy.mean().item():.4f}")

# Update posterior beliefs
engine.update_posterior(observations, 'perception_model', learning_rate=0.01)

# Generate predictions
predictions = engine.predict('perception_model', num_samples=10)
print(f"Predictions shape: {predictions.shape}")
```

### Bayesian Inference Integration

```python
# Using ai prefix from previous import

# Create Bayesian inference engine
bayesian_engine = ai.BayesianInference(engine)

# Define prior and likelihood
prior = torch.softmax(torch.randn(batch_size, 10), dim=1)
likelihood = torch.softmax(torch.randn(batch_size, 10), dim=1)

# Compute posterior
posterior = bayesian_engine.compute_posterior(prior, likelihood)

# Compute evidence
evidence = bayesian_engine.compute_evidence(prior, likelihood)

print(f"Posterior entropy: {torch.distributions.Categorical(probs=posterior).entropy().mean():.4f}")
print(f"Log evidence: {evidence.mean():.4f}")
```

## Free Energy Minimization

### Variational Free Energy Optimization

```python
import torch
# Using ai prefix from previous import

# Set up GPU acceleration
gpu_accelerator = ai.GPUAccelerator(feature_manager)
vfe_engine = ai.VariationalFreeEnergy(feature_manager)

# Create synthetic data
batch_size, feature_dim = 64, 8
observations = torch.randn(batch_size, feature_dim)
prior = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)
likelihood = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)

# Initialize variational posterior
initial_posterior = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)

# Compute initial free energy
initial_fe = vfe_engine.compute(observations, initial_posterior, prior, likelihood)
print(f"Initial free energy: {initial_fe.mean().item():.4f}")

# Minimize free energy
optimized_posterior = vfe_engine.minimize(
    observations, initial_posterior, prior, likelihood,
    num_iterations=100, learning_rate=0.01
)

# Compute final free energy
final_fe = vfe_engine.compute(observations, optimized_posterior, prior, likelihood)
print(f"Final free energy: {final_fe.mean().item():.4f}")
print(f"Improvement: {initial_fe.mean().item() - final_fe.mean().item():.4f}")
```

### Gradient Flow Visualization

```python
import matplotlib.pyplot as plt

# Track free energy during optimization
free_energy_history = []

def track_free_energy(posterior):
    fe = vfe_engine.compute(observations, posterior, prior, likelihood)
    free_energy_history.append(fe.mean().item())
    return fe

# Minimize with tracking
optimized_posterior = vfe_engine.minimize(
    observations, initial_posterior, prior, likelihood,
    num_iterations=50, learning_rate=0.05
)

# Visualize optimization
plt.figure(figsize=(10, 6))
plt.plot(free_energy_history, 'b-', linewidth=2, label='Free Energy')
plt.xlabel('Iteration')
plt.ylabel('Free Energy')
plt.title('Variational Free Energy Minimization')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

## Policy Selection with Expected Free Energy

### Action Selection

```python
# Using ai prefix from previous import

# Initialize expected free energy engine
efe_engine = ai.ExpectedFreeEnergy(feature_manager)

# Define policies (possible actions)
num_policies = 5
feature_dim = 6
policies = torch.randn(num_policies, feature_dim)

# Current observations and beliefs
observations = torch.randn(batch_size, feature_dim)
posterior = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)

# Define preferences (goals)
preferences = torch.tensor([1.0, -1.0, 0.5, 0.0, -0.5])  # Preference for each policy

# Compute expected free energy for each policy
EFE = efe_engine.compute(observations, policies, posterior, preferences)

print(f"Expected Free Energy shape: {EFE.shape}")
print(f"Mean EFE per policy: {EFE.mean(dim=0)}")

# Select optimal policy
policy_indices, efe_values = efe_engine.select_policy(EFE)

print(f"Selected policies: {policy_indices[:10]}")  # First 10 samples
print(f"Corresponding EFE values: {efe_values[:10]}")
```

### Multi-step Policy Evaluation

```python
# Simulate multi-step decision making
horizon = 3  # Planning horizon
num_trajectories = 100

# Generate policy sequences
policy_sequences = []
for t in range(horizon):
    policies_t = torch.randn(num_policies, feature_dim)
    policy_sequences.append(policies_t)

# Evaluate cumulative expected free energy
total_EFE = torch.zeros(batch_size, num_policies)

for t in range(horizon):
    # Predict future observations under current policy
    future_obs = torch.randn(batch_size, feature_dim)  # Predicted observations

    # Update beliefs based on predictions
    updated_posterior = torch.softmax(torch.randn(batch_size, feature_dim), dim=1)

    # Compute EFE for this time step
    EFE_t = efe_engine.compute(future_obs, policy_sequences[t],
                              updated_posterior, preferences)

    # Discount future EFE
    discount_factor = 0.9 ** t
    total_EFE += discount_factor * EFE_t

# Select optimal policy sequence
optimal_policy_idx = total_EFE.argmin(dim=1)
print(f"Optimal policy indices: {optimal_policy_idx[:10]}")
```

## Message Passing on Graphical Models

### Belief Propagation on Trees

```python
# Using ai prefix from previous import

# Create belief propagation engine
bp_engine = ai.BeliefPropagation(feature_manager)

# Define a tree-structured graph (chain)
num_nodes = 5
adjacency = torch.zeros(num_nodes, num_nodes)

# Create chain: 0-1-2-3-4
for i in range(num_nodes - 1):
    adjacency[i, i+1] = 1
    adjacency[i+1, i] = 1  # Undirected

# Define node potentials (evidence)
num_states = 3
node_potentials = torch.softmax(torch.randn(num_nodes, num_states), dim=1)

# Add strong evidence at leaves
node_potentials[0, 0] = 10.0  # Strong evidence for state 0 at node 0
node_potentials[-1, 2] = 10.0  # Strong evidence for state 2 at node 4
node_potentials = torch.softmax(node_potentials, dim=1)

# Set up the graph
bp_engine.set_graph(adjacency, node_potentials)

# Run belief propagation
result = bp_engine.run(max_iterations=20, tolerance=1e-6)

print(f"Converged: {result['converged']}")
print(f"Final beliefs shape: {result['beliefs'].shape}")

# Visualize beliefs
beliefs = result['beliefs']
for i in range(num_nodes):
    print(f"Node {i} beliefs: {beliefs[i].detach().numpy()}")
```

### Variational Message Passing on Loopy Graphs

```python
# Using ai prefix from previous import

# Create variational message passing engine
vmp_engine = ai.VariationalMessagePassing(feature_manager)

# Define a loopy graph (triangle)
adjacency = torch.tensor([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=torch.float)

# Node potentials
node_potentials = torch.softmax(torch.randn(3, num_states), dim=1)

# Set up the graph
vmp_engine.set_graph(adjacency, node_potentials)

# Run variational message passing
result = vmp_engine.run(max_iterations=30, learning_rate=0.1)

print(f"Final beliefs: {result['beliefs'].detach().numpy()}")
print(f"Variational free energy: {result['free_energy'].item():.4f}")

# Compare with exact inference (if available)
print("Note: For loopy graphs, VMP provides approximate inference")
```

## Complete Workflow Integration

### Perception-Action Loop

```python
# Complete active inference loop
def active_inference_loop(observations, num_steps=10):
    """Demonstrate perception-action cycle."""

    # Initialize components
    feature_manager = TritonFeatureManager()
    engine = ActiveInferenceEngine(feature_manager)
    efe_engine = ExpectedFreeEnergy(feature_manager)

    # Register perception model
    perception_model = {
        'variables': {
            'hidden': {'shape': (10,)},
            'observed': {'shape': (observations.shape[1],)}
        }
    }
    engine.register_model('perception', perception_model)

    # Define action policies
    num_actions = 4
    action_policies = torch.randn(num_actions, 10)

    history = {
        'free_energy': [],
        'actions': [],
        'beliefs': []
    }

    for step in range(num_steps):
        # Perception: Update beliefs
        engine.update_posterior(observations, 'perception')
        posterior = engine.posteriors['perception']['hidden']

        # Compute current free energy
        fe = engine.compute_free_energy(observations, 'perception')
        history['free_energy'].append(fe.mean().item())

        # Action selection: Choose policy minimizing expected free energy
        EFE = efe_engine.compute(observations, action_policies, posterior)
        action_idx, _ = efe_engine.select_policy(EFE)

        history['actions'].append(action_idx.item())
        history['beliefs'].append(posterior.detach().clone())

        # Simulate action consequences (would be environment interaction)
        observations = observations + 0.1 * torch.randn_like(observations)

        print(f"Step {step}: FE={fe.mean().item():.4f}, Action={action_idx.item()}")

    return history

# Run the perception-action loop
observations = torch.randn(16, 5)
history = active_inference_loop(observations, num_steps=5)

# Visualize the loop
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['free_energy'], 'r-o')
plt.title('Variational Free Energy')
plt.xlabel('Time Step')
plt.ylabel('Free Energy')

plt.subplot(1, 3, 2)
plt.plot(history['actions'], 'b-s')
plt.title('Selected Actions')
plt.xlabel('Time Step')
plt.ylabel('Action Index')

plt.subplot(1, 3, 3)
beliefs_over_time = torch.stack(history['beliefs'])
plt.imshow(beliefs_over_time.T, aspect='auto', cmap='viridis')
plt.title('Belief Evolution')
plt.xlabel('Time Step')
plt.ylabel('Hidden State')
plt.colorbar()

plt.tight_layout()
plt.show()
```

## Performance Benchmarking

### Feature Coverage Verification

```python
from active_inference import TritonFeatureManager

def verify_triton_features():
    """Verify complete Triton feature coverage."""

    feature_manager = TritonFeatureManager()

    # List all available features
    features = feature_manager.list_features()
    print("Available Triton Features:")
    for category, items in features.items():
        print(f"  {category}: {len(items)} items")
        if len(items) <= 10:  # Show all if small list
            for item in items:
                print(f"    - {item}")

    # Verify specific kernels
    test_kernels = [
        'variational_free_energy',
        'expected_free_energy',
        'belief_propagation',
        'variational_message_passing'
    ]

    print("\nKernel Verification:")
    for kernel_name in test_kernels:
        verified = feature_manager.verify_feature(kernel_name)
        status = "✓" if verified else "✗"
        print(f"  {status} {kernel_name}")

    return features

# Run verification
features = verify_triton_features()
```

### Memory and Performance Profiling

```python
from active_inference import GPUAccelerator
import time

def profile_performance():
    """Profile memory usage and performance."""

    feature_manager = TritonFeatureManager()
    gpu_accelerator = GPUAccelerator(feature_manager)

    # Profile memory allocation
    print("Memory Profiling:")
    initial_stats = gpu_accelerator.get_memory_stats()
    print(f"Initial memory: {initial_stats}")

    # Allocate test tensors
    shapes = [(1000, 1000), (500, 500), (100, 100)]
    tensors = gpu_accelerator.allocate_tensors(shapes)

    allocation_stats = gpu_accelerator.get_memory_stats()
    print(f"After allocation: {allocation_stats}")

    # Profile computation
    print("\nPerformance Profiling:")

    # Test variational free energy computation
    from active_inference import VariationalFreeEnergy
    vfe_engine = VariationalFreeEnergy(feature_manager)

    observations = torch.randn(100, 50)
    posterior = torch.softmax(torch.randn(100, 50), dim=1)
    prior = torch.softmax(torch.randn(100, 50), dim=1)
    likelihood = torch.softmax(torch.randn(100, 50), dim=1)

    # Time the computation
    gpu_accelerator.synchronize()  # Ensure GPU is ready
    start_time = time.time()

    free_energy = vfe_engine.compute(observations, posterior, prior, likelihood)

    gpu_accelerator.synchronize()  # Wait for completion
    end_time = time.time()

    computation_time = end_time - start_time
    print(f"VFE computation time: {computation_time:.4f} seconds")
    print(f"Throughput: {100 / computation_time:.2f} samples/second")

    return {
        'memory_stats': allocation_stats,
        'computation_time': computation_time,
        'throughput': 100 / computation_time
    }

# Run profiling
profile_results = profile_performance()
```

## Advanced Usage Patterns

### Custom Kernel Registration

```python
@triton.jit
def custom_inference_kernel(
    data_ptr, weights_ptr, output_ptr,
    batch_size: tl.constexpr, feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256
):
    """Custom inference kernel example."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    # Load data
    data = tl.load(data_ptr + offsets * feature_dim, mask=mask)
    weights = tl.load(weights_ptr + offsets * feature_dim, mask=mask)

    # Custom computation (example: weighted sum with activation)
    result = tl.sum(data * weights, axis=0)
    result = tl.maximum(result, 0.0)  # ReLU activation

    tl.store(output_ptr + pid, result)

# Register custom kernel
feature_manager.register_kernel(
    'custom_inference',
    custom_inference_kernel,
    {
        'description': 'Custom inference with activation',
        'input_shapes': ['batch_size x feature_dim', 'batch_size x feature_dim'],
        'output_shapes': ['batch_size'],
        'optimizations': ['vectorization', 'shared_memory']
    }
)
```

### Multi-GPU Distribution

```python
def setup_multi_gpu():
    """Example of multi-GPU setup (conceptual)."""

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Create separate engines for each GPU
    engines = []
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            config = TritonFeatureConfig(device=f"cuda:{gpu_id}")
            feature_manager = TritonFeatureManager(config)
            engine = ActiveInferenceEngine(feature_manager)
            engines.append(engine)

    # Distribute workload across GPUs
    batch_size_per_gpu = 1000 // num_gpus

    results = []
    for gpu_id, engine in enumerate(engines):
        with torch.cuda.device(gpu_id):
            # Process data on this GPU
            observations = torch.randn(batch_size_per_gpu, 50)
            result = engine.compute_free_energy(observations, 'model')
            results.append(result.cpu())  # Move to CPU for aggregation

    # Aggregate results
    final_result = torch.cat(results, dim=0)
    return final_result

# Note: This is a conceptual example. Actual multi-GPU implementation
# would require careful consideration of data transfer and synchronization.
```

These examples demonstrate the thin orchestrator approach - each example focuses on a specific use case while leveraging the full power of the underlying Triton-accelerated active inference framework.
