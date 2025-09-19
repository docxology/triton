# Triton Usage Guide for Active Inference

## Overview

This guide provides comprehensive documentation for using Triton kernels in the Active Inference framework. It covers everything from basic kernel writing to advanced optimization techniques, with a focus on Apple Silicon compatibility and PyTorch fallback mechanisms.

## Table of Contents

1. [Introduction to Triton](#introduction-to-triton)
2. [Basic Kernel Structure](#basic-kernel-structure)
3. [Memory Management](#memory-management)
4. [Platform-Specific Optimizations](#platform-specific-optimizations)
5. [Active Inference Kernels](#active-inference-kernels)
6. [Performance Optimization](#performance-optimization)
7. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
8. [Best Practices](#best-practices)

## Introduction to Triton

### What is Triton?

Triton is a language and compiler for writing highly efficient GPU kernels using Python syntax. It enables:

- **Pythonic kernel writing**: Write GPU kernels in Python with automatic compilation
- **High performance**: Achieve near-optimal GPU utilization
- **Flexibility**: Support for complex algorithms and custom operations
- **Platform agnostic**: Works across different GPU architectures

### Key Concepts

- **Programs**: Functions decorated with `@triton.jit` that run on GPU
- **Blocks**: Groups of threads that execute together (typically 128-1024 threads)
- **Warps**: Groups of 32 threads within a block
- **Grids**: Arrays of blocks that cover the entire computation

### Why Triton for Active Inference?

Active Inference algorithms require:
- Complex mathematical operations
- Custom probability computations
- Message passing algorithms
- Real-time performance for decision making

Triton enables all these with GPU acceleration while maintaining algorithm readability.

## Basic Kernel Structure

### Simple Vector Addition Kernel

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,  # Pointers to input/output tensors
    n_elements,           # Number of elements to process
    BLOCK_SIZE: tl.constexpr = 1024,  # Compile-time constant
):
    """
    Triton kernel for vector addition.

    Args:
        a_ptr: Pointer to first input tensor
        b_ptr: Pointer to second input tensor
        c_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Number of threads per block
    """
    # Get thread index within the grid
    pid = tl.program_id(0)  # Program ID along dimension 0

    # Calculate indices for this thread block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for boundary conditions
    mask = offsets < n_elements

    # Load data with masking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Perform computation
    c = a + b

    # Store result with masking
    tl.store(c_ptr + offsets, c, mask=mask)
```

### Usage in Active Inference Framework

```python
from core import launch_triton_kernel, vector_add_kernel

# Create tensors
a = torch.randn(10000, device='cpu')
b = torch.randn(10000, device='cpu')
c = torch.zeros_like(a)

# Launch kernel
result = launch_triton_kernel(
    vector_add_kernel,
    (triton.cdiv(len(a), 1024),),  # Grid dimensions
    a, b, c, len(a)                # Kernel arguments
)

# Result contains the output tensor or None (if fallback used)
if result is not None:
    c = result
else:
    # Fallback already executed, c contains the result
    pass
```

## Memory Management

### Memory Layout Optimization

Triton requires careful memory layout for optimal performance:

```python
@triton.jit
def coalesced_memory_kernel(
    input_ptr, output_ptr,
    batch_size: tl.constexpr,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    """
    Demonstrates coalesced memory access patterns.
    """
    # Thread indices
    batch_idx = tl.program_id(0)
    feature_start = tl.program_id(1) * BLOCK_SIZE

    # Coalesced memory access: threads access consecutive elements
    offsets = feature_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_dim

    # Load entire row for this batch element
    input_start = batch_idx * feature_dim
    data = tl.load(input_ptr + input_start + offsets, mask=mask)

    # Process data
    result = data * 2.0  # Example operation

    # Store result
    tl.store(output_ptr + input_start + offsets, result, mask=mask)
```

### Shared Memory Usage

```python
@triton.jit
def shared_memory_kernel(
    input_ptr, output_ptr,
    batch_size: tl.constexpr,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    """
    Demonstrates shared memory usage for performance.
    """
    # Shared memory declaration
    shared_data = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Thread indices
    batch_idx = tl.program_id(0)
    local_idx = tl.arange(0, BLOCK_SIZE)

    # Load data into shared memory
    global_idx = batch_idx * feature_dim + local_idx
    data = tl.load(input_ptr + global_idx, mask=local_idx < feature_dim)
    shared_data[local_idx] = data

    # Synchronize threads
    tl.sync()

    # Process data in shared memory
    result = shared_data[local_idx] + 1.0

    # Store result
    tl.store(output_ptr + global_idx, result, mask=local_idx < feature_dim)
```

## Platform-Specific Optimizations

### Apple Silicon (MPS) Considerations

```python
def optimize_for_apple_silicon(kernel_fn, *args, **kwargs):
    """
    Platform-specific optimization for Apple Silicon.
    """
    import platform

    is_apple_silicon = (platform.system() == "Darwin" and
                       platform.machine() == "arm64")

    if is_apple_silicon:
        # Use larger block sizes for better MPS performance
        kwargs['BLOCK_SIZE'] = 512

        # Ensure tensors are contiguous for MPS
        optimized_args = []
        for arg in args:
            if hasattr(arg, 'is_contiguous') and not arg.is_contiguous():
                arg = arg.contiguous()
            optimized_args.append(arg)

        return kernel_fn(*optimized_args, **kwargs)
    else:
        return kernel_fn(*args, **kwargs)
```

### CUDA Optimizations

```python
@triton.jit
def cuda_optimized_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,  # Larger blocks for CUDA
):
    """
    CUDA-optimized kernel with larger block sizes.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    data = tl.load(input_ptr + offsets, mask=mask)

    # Process with CUDA-specific optimizations
    result = data * data  # Example computation

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)
```

## Active Inference Kernels

### Variational Free Energy Kernel

```python
@triton.jit
def variational_free_energy_kernel(
    observations_ptr,
    posterior_ptr,
    prior_ptr,
    likelihood_ptr,
    free_energy_ptr,
    batch_size: tl.constexpr,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 128,
):
    """
    Triton kernel for variational free energy computation.

    F(μ) = E_q[log q(z|μ) - log p(x,z|μ)]
    """
    batch_idx = tl.program_id(0)
    feature_start = batch_idx * BLOCK_SIZE

    offsets = feature_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * feature_dim

    # Load data
    obs = tl.load(observations_ptr + offsets, mask=mask)
    post = tl.load(posterior_ptr + offsets, mask=mask)
    prior = tl.load(prior_ptr + offsets, mask=mask)
    lik = tl.load(likelihood_ptr + offsets, mask=mask)

    # Compute expected log likelihood
    expected_ll = obs * post * lik

    # Compute KL divergence
    eps = 1e-8
    kl_term = post * tl.log((post + eps) / (prior + eps))

    # Combine terms (this is simplified - would need reduction)
    vfe_contrib = -expected_ll + kl_term

    # Store contribution
    tl.store(free_energy_ptr + offsets, vfe_contrib, mask=mask)
```

### Message Passing Kernel

```python
@triton.jit
def belief_propagation_kernel(
    messages_ptr,
    potentials_ptr,
    adjacency_ptr,
    new_messages_ptr,
    num_nodes: tl.constexpr,
    num_states: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 32,
):
    """
    Triton kernel for belief propagation message updates.
    """
    edge_idx = tl.program_id(0)

    # Calculate node indices for this edge
    node_i = edge_idx // num_nodes
    node_j = edge_idx % num_nodes

    # Skip if no edge exists
    edge_exists = tl.load(adjacency_ptr + node_i * num_nodes + node_j)
    if edge_exists == 0:
        return

    # Load node potentials
    node_i_potential = tl.load(
        potentials_ptr + node_i * num_states + tl.arange(0, num_states)
    )
    node_j_potential = tl.load(
        potentials_ptr + node_j * num_states + tl.arange(0, num_states)
    )

    # Load incoming messages to node i (excluding from j)
    incoming_message = tl.ones((num_states,), dtype=tl.float32)
    for k in range(num_nodes):
        if k != node_j:
            edge_k_i = k * num_nodes + node_i
            edge_exists_k = tl.load(adjacency_ptr + edge_k_i)
            if edge_exists_k > 0:
                msg_k_i = tl.load(
                    messages_ptr + edge_k_i * num_states + tl.arange(0, num_states)
                )
                incoming_message *= msg_k_i

    # Compute new message: normalize(ψ_i * ∏_{k≠j} m_ki)
    message_contrib = node_i_potential * incoming_message
    message_sum = tl.sum(message_contrib)
    new_message = message_contrib / (message_sum + 1e-8)

    # Store new message
    tl.store(
        new_messages_ptr + edge_idx * num_states + tl.arange(0, num_states),
        new_message
    )
```

## Performance Optimization

### Block Size Selection

```python
def optimal_block_size(problem_size: int, device_type: str) -> int:
    """
    Select optimal block size based on problem size and device.
    """
    if device_type == "cuda":
        if problem_size < 1024:
            return 256
        elif problem_size < 8192:
            return 512
        else:
            return 1024
    elif device_type == "mps":  # Apple Silicon
        if problem_size < 1024:
            return 128
        elif problem_size < 4096:
            return 256
        else:
            return 512
    else:  # CPU
        return 64
```

### Memory Access Patterns

```python
@triton.jit
def optimized_memory_access_kernel(
    input_ptr, output_ptr,
    batch_size: tl.constexpr,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    """
    Optimized memory access with coalesced loads/stores.
    """
    # Use 2D grid for better memory coalescing
    batch_idx = tl.program_id(0)
    feature_block = tl.program_id(1) * BLOCK_SIZE

    # Coalesced access: threads in a warp access consecutive memory
    offsets = feature_block + tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_dim

    # Load entire block of features for this batch
    input_start = batch_idx * feature_dim
    data = tl.load(input_ptr + input_start + offsets, mask=mask)

    # Process data
    result = data * 2.0 + 1.0  # Example fused operations

    # Store result (coalesced)
    tl.store(output_ptr + input_start + offsets, result, mask=mask)
```

### Fused Operations

```python
@triton.jit
def fused_operations_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size: tl.constexpr,
    input_dim: tl.constexpr,
    output_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 128,
):
    """
    Fused matrix multiplication with activation and bias addition.
    """
    batch_idx = tl.program_id(0)
    out_idx = tl.program_id(1)

    # Load weight vector for this output dimension
    weights = tl.load(
        weight_ptr + out_idx * input_dim + tl.arange(0, BLOCK_SIZE),
        mask=tl.arange(0, BLOCK_SIZE) < input_dim
    )

    # Initialize accumulator
    result = tl.zeros((1,), dtype=tl.float32)

    # Matrix multiplication
    for i in range(0, input_dim, BLOCK_SIZE):
        input_vals = tl.load(
            input_ptr + batch_idx * input_dim + i + tl.arange(0, BLOCK_SIZE),
            mask=(i + tl.arange(0, BLOCK_SIZE)) < input_dim
        )
        result += tl.sum(input_vals * weights)

    # Add bias
    bias = tl.load(bias_ptr + out_idx)
    result += bias

    # Apply activation (ReLU)
    result = tl.maximum(result, 0.0)

    # Store result
    tl.store(output_ptr + batch_idx * output_dim + out_idx, result)
```

## Debugging and Troubleshooting

### Common Issues and Solutions

#### 1. "0 active drivers" Error (Apple Silicon)

```python
# Solution: Check platform and use appropriate fallback
import platform

is_apple_silicon = (platform.system() == "Darwin" and
                   platform.machine() == "arm64")

if is_apple_silicon:
    # Use PyTorch fallback for Apple Silicon
    result = pytorch_fallback_function(*args, **kwargs)
else:
    # Try Triton kernel
    result = launch_triton_kernel(kernel_fn, grid, *args, **kwargs)
```

#### 2. Memory Access Violations

```python
# Solution: Always use masks for boundary conditions
@triton.jit
def safe_memory_access_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Always use masks!
    mask = offsets < n_elements

    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = data * 2.0
    tl.store(output_ptr + offsets, result, mask=mask)
```

#### 3. Performance Issues

```python
# Solution: Profile and optimize block sizes
def profile_kernel_performance(kernel_fn, *args, block_sizes=[128, 256, 512, 1024]):
    """Profile kernel performance across different block sizes."""
    results = {}

    for block_size in block_sizes:
        try:
            # Time kernel execution
            start_time = time.time()
            result = launch_triton_kernel(kernel_fn, grid, *args, BLOCK_SIZE=block_size)
            end_time = time.time()

            results[block_size] = end_time - start_time
        except:
            results[block_size] = float('inf')

    # Return best block size
    best_block_size = min(results, key=results.get)
    return best_block_size, results
```

## Best Practices

### 1. Memory Layout
- Always ensure tensors are contiguous before kernel launch
- Use coalesced memory access patterns
- Consider shared memory for frequently accessed data

### 2. Block Size Selection
- Choose block sizes that are multiples of 32 (warp size)
- Balance occupancy and memory usage
- Profile different sizes for optimal performance

### 3. Error Handling
- Always use masks for boundary conditions
- Implement comprehensive error handling
- Provide meaningful fallback mechanisms

### 4. Platform Awareness
- Detect platform and optimize accordingly
- Use appropriate block sizes for each platform
- Implement platform-specific optimizations

### 5. Performance Monitoring
- Profile kernels regularly
- Monitor memory usage
- Track kernel launch overhead

### 6. Code Organization
- Separate kernel definitions from host code
- Use meaningful variable names
- Document kernel parameters and behavior

## Example: Complete Active Inference Kernel

```python
@triton.jit
def complete_active_inference_kernel(
    observations_ptr,
    beliefs_ptr,
    policies_ptr,
    EFE_ptr,
    batch_size: tl.constexpr,
    n_states: tl.constexpr,
    n_policies: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 128,
):
    """
    Complete active inference kernel combining multiple operations.

    Computes expected free energy for policy selection.
    """
    batch_idx = tl.program_id(0)
    policy_idx = tl.program_id(1)

    # Load current beliefs
    beliefs = tl.load(
        beliefs_ptr + batch_idx * n_states + tl.arange(0, n_states),
        mask=tl.arange(0, n_states) < n_states
    )

    # Load policy
    policy = tl.load(
        policies_ptr + policy_idx * n_states + tl.arange(0, n_states),
        mask=tl.arange(0, n_states) < n_states
    )

    # Compute epistemic value (KL divergence)
    eps = 1e-8
    epistemic = tl.sum(
        beliefs * tl.log((beliefs + eps) / (policy + eps))
    )

    # Compute pragmatic value (expected reward)
    observations = tl.load(
        observations_ptr + batch_idx * n_states + tl.arange(0, n_states),
        mask=tl.arange(0, n_states) < n_states
    )
    pragmatic = tl.sum(beliefs * observations * policy)

    # Expected free energy (minimize this)
    efe = epistemic - pragmatic

    # Store result
    tl.store(
        EFE_ptr + batch_idx * n_policies + policy_idx,
        efe
    )
```

## Summary

This guide provides a comprehensive foundation for using Triton in active inference applications. Key takeaways:

1. **Triton enables Pythonic GPU kernel writing** with high performance
2. **Platform awareness is crucial** for optimal performance across devices
3. **Proper error handling and fallbacks** ensure robustness
4. **Memory layout and access patterns** significantly impact performance
5. **Active inference algorithms** benefit greatly from Triton optimization

With this foundation, you can develop high-performance active inference systems that work seamlessly across different platforms while maintaining algorithm clarity and correctness.
