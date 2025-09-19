# Theoretical Background

Mathematical foundations and theoretical underpinnings of the active inference and free energy principle implementations.

## Free Energy Principle

### Core Formulation

The free energy principle states that biological systems minimize variational free energy to maintain their integrity in the face of uncertainty. The variational free energy is defined as:

```
F = E_q[log q(z) - log p(x,z)]
```

Where:
- `q(z)` is the variational posterior (approximate posterior)
- `p(x,z)` is the generative model (joint distribution)
- `E_q[...]` denotes expectation under the variational distribution q

### Information-Theoretic Interpretation

The variational free energy can be decomposed as:

```
F = KL(q(z)||p(z)) - E_q[log p(x|z)] + log p(x)
```

This reveals that minimizing F is equivalent to:
1. Maximizing the likelihood of observations under the model
2. Minimizing the KL divergence between posterior and prior
3. Maximizing the marginal likelihood (evidence)

### Biological Relevance

In neuroscience, the free energy principle provides a unified account of:
- **Perception**: Inference about hidden causes of sensory data
- **Action**: Selection of actions that minimize expected free energy
- **Learning**: Updating of generative models based on experience

## Active Inference

### Expected Free Energy

Active inference extends the free energy principle to action by introducing expected free energy:

```
G_π = E_q[log p(o|π) + log p(π) - log q(o|π)] - C
```

Where:
- `π` represents policies (sequences of actions)
- `o` represents predicted observations
- `C` represents preferences or goals
- `q(o|π)` is the predicted observation distribution under policy π

### Policy Selection

Agents select policies that minimize expected free energy:

```
π* = argmin_π G_π
```

This leads to behavior that:
1. **Exploratory**: Prefers policies with high information gain
2. **Exploitative**: Prefers policies that achieve desired outcomes
3. **Risk-averse**: Avoids policies with high uncertainty

### Precision and Attention

The precision (inverse variance) of beliefs modulates the relative influence of different terms:

```
G_π = E_q[log p(o|π)] + γ * KL(q(o|π)||q(o)) - C
```

Where `γ` controls the relative weight of epistemic affordance (information gain).

## Variational Inference

### Variational Posterior

The variational posterior q(z) approximates the true posterior p(z|x):

```
q(z) ≈ p(z|x) = p(x|z) * p(z) / p(x)
```

Common variational families include:
- **Mean-field**: `q(z) = ∏ q_i(z_i)`
- **Structured**: `q(z)` respects dependencies in the generative model

### Evidence Lower Bound (ELBO)

The variational free energy provides a lower bound on the log evidence:

```
log p(x) ≥ E_q[log p(x,z)] - E_q[log q(z)] = -F
```

Maximizing the ELBO (minimizing F) improves the variational approximation.

### Coordinate Ascent

Many variational algorithms use coordinate ascent:

1. Fix q, optimize model parameters θ
2. Fix θ, optimize variational parameters φ

This leads to the variational EM algorithm.

## Message Passing Algorithms

### Belief Propagation

For tree-structured graphs, belief propagation provides exact inference:

**Message Update**:
```
m_ij(x_j) = ∑_{x_i} ψ_i(x_i) ψ_ij(x_i,x_j) ∏_{k≠j} m_ki(x_i)
```

**Belief Update**:
```
b_i(x_i) = ψ_i(x_i) ∏_{j∈N(i)} m_ji(x_i)
```

Where:
- `ψ_i(x_i)` are node potentials
- `ψ_ij(x_i,x_j)` are edge potentials
- `m_ij(x_j)` are messages from i to j
- `b_i(x_i)` are beliefs at node i

### Variational Message Passing

For loopy graphs, variational message passing approximates inference:

**Free Energy**:
```
F = ∑_i E_q[log q_i - log ψ_i] + ∑_{ij} E_q[log q_i q_j - log ψ_ij]
```

**Update Rules**:
```
q_i ∝ exp(E_{q_{\j}}[log ψ_i] + ∑_{j≠k} m_{ji})
```

Where `q_{\j}` denotes all q variables except q_j.

### Convergence and Accuracy

- **Trees**: Belief propagation converges in O(N) time with exact results
- **Loopy graphs**: Variational message passing provides approximate results
- **Guarantees**: Some loopy graphs admit exact fixed points

## Bayesian Inference

### Bayes Rule

The foundation of probabilistic inference:

```
p(z|x) = p(x|z) * p(z) / p(x)
```

Where:
- `p(z|x)` is the posterior
- `p(x|z)` is the likelihood
- `p(z)` is the prior
- `p(x)` is the evidence (normalizing constant)

### Predictive Distributions

**Posterior predictive**:
```
p(x'|x) = ∫ p(x'|z) p(z|x) dz
```

**Prior predictive**:
```
p(x') = ∫ p(x'|z) p(z) dz
```

### Bayesian Model Comparison

Compare models using marginal likelihood:

```
p(x|M) = ∫ p(x|z,M) p(z|M) dz
```

Models with higher marginal likelihood are preferred.

## GPU Acceleration with Triton

### Kernel Optimization

Triton provides several optimization opportunities:

**Shared Memory**:
```python
@triton.jit
def kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    # Load data into shared memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    result = some_computation(x, y)

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)
```

**Vectorization**:
```python
# Automatic vectorization of element-wise operations
z = x + y * tl.exp(w)
```

**Pipelining**:
```python
# Overlap memory access and computation
for stage in range(NUM_STAGES):
    # Load next batch
    # Compute current batch
    # Store previous batch
```

### Memory Layout Optimization

**Coalesced Access**:
- Arrange data to maximize memory bandwidth
- Use appropriate data types (float32, bfloat16, float16)

**Shared Memory Usage**:
- Cache frequently accessed data
- Reduce global memory traffic
- Enable fast data sharing within thread blocks

### Performance Characteristics

**Theoretical Peak Performance**:
- Modern GPUs: 10-20 TFLOPS
- Memory bandwidth: 1-2 TB/s
- Latency hiding through massive parallelism

**Achievable Performance**:
- Well-optimized kernels: 50-80% of peak
- Memory-bound operations: Limited by bandwidth
- Compute-bound operations: Limited by FLOPS

## Implementation Details

### Variational Free Energy Kernel

The variational free energy computation uses a custom Triton kernel:

```python
@triton.jit
def variational_free_energy_kernel(
    observations_ptr, posterior_ptr, prior_ptr, likelihood_ptr,
    free_energy_ptr, batch_size: tl.constexpr, feature_dim: tl.constexpr
):
    # Compute E_q[log p(x|z)]
    expected_ll = tl.sum(obs * post * lik, axis=0)

    # Compute KL(q||p)
    kl_div = tl.sum(post * tl.log(post / (prior + 1e-8)), axis=0)

    # Variational free energy
    vfe = -expected_ll + kl_div
```

### Message Passing Kernel

Belief propagation implemented with parallel message updates:

```python
@triton.jit
def belief_propagation_kernel(
    messages_ptr, potentials_ptr, adjacency_ptr,
    new_messages_ptr, num_nodes: tl.constexpr, num_states: tl.constexpr
):
    # Parallel update of all messages
    edge_id = tl.program_id(0)
    # ... message computation ...
```

### Numerical Stability

**Log-Sum-Exp Trick**:
```python
# Avoid numerical overflow in softmax
max_val = tl.max(logits)
log_sum = max_val + tl.log(tl.sum(tl.exp(logits - max_val)))
```

**Small Constants**:
```python
# Prevent log(0) and division by zero
epsilon = 1e-8
safe_log = tl.log(x + epsilon)
safe_div = tl.where(y != 0, x / y, 0.0)
```

## Applications

### Neuroscience

- **Predictive coding**: Hierarchical inference in cortical circuits
- **Active vision**: Eye movements and attention allocation
- **Motor control**: Optimal control under uncertainty

### Machine Learning

- **Variational autoencoders**: Generative modeling with latent variables
- **Bayesian neural networks**: Uncertainty quantification in deep learning
- **Reinforcement learning**: Exploration-exploitation trade-offs

### Artificial Intelligence

- **Autonomous agents**: Decision making under uncertainty
- **Cognitive architectures**: Unified perception-action-learning systems
- **Robotics**: Sensorimotor integration and control

## Future Directions

### Advanced Inference Methods

- **Amortized inference**: Neural network-based variational approximations
- **Particle-based methods**: Sequential Monte Carlo for complex distributions
- **Normalizing flows**: Flexible variational families

### Scalability Improvements

- **Distributed inference**: Multi-GPU and multi-node implementations
- **Sparse representations**: Efficient handling of high-dimensional data
- **Online learning**: Incremental updates for streaming data

### Integration with Deep Learning

- **Neural variational inference**: Differentiable inference in neural networks
- **Deep generative models**: Combining active inference with modern architectures
- **Meta-learning**: Learning to learn generative models

## References

### Foundational Papers

1. **Friston, K. (2010)**. The free-energy principle: a unified brain theory? Nature Reviews Neuroscience
2. **Parr, T., & Friston, K. J. (2017)**. Working memory, attention, and salience in active inference. Scientific Reports
3. **Buckley, C. L., et al. (2017)**. The free energy principle for action and perception: A mathematical review. Journal of Mathematical Psychology

### Technical References

4. **Titterington, D. M. (2004)**. Generalized linear models with functional form. Statistical Science
5. **Koller, D., & Friedman, N. (2009)**. Probabilistic Graphical Models. MIT Press
6. **Bishop, C. M. (2006)**. Pattern Recognition and Machine Learning. Springer

### Implementation References

7. **Triton documentation**: GPU-accelerated deep learning compilation
8. **PyTorch documentation**: Automatic differentiation and GPU computing
9. **CUDA programming guide**: GPU architecture and optimization
