# Understanding Gradient Descent Algorithms

## Introduction
Gradient Descent is a powerful optimization algorithm widely used in machine learning to minimize cost functions. It works by iteratively adjusting parameters in the direction of steepest descent of the loss surface.

## Types of Gradient Descent

### 1. Batch Gradient Descent (BGD)
- Computes gradient using entire dataset
- Characteristics:
    - Most stable convergence
    - Computationally expensive
    - High memory requirements
- Formula: θ = θ - α * ∇J(θ)
- Best for: Small to medium datasets

### 2. Stochastic Gradient Descent (SGD)
- Updates parameters using single training example
- Characteristics:
    - Fast computation
    - High variance in parameter updates
    - Less likely to get stuck in local minima
- Formula: θ = θ - α * ∇J(θ; x(i); y(i))
- Best for: Large datasets, online learning

### 3. Mini-batch Gradient Descent
- Uses small batches (typically 32-256 samples)
- Characteristics:
    - Balanced computation speed
    - Moderate parameter update variance
    - Good parallelization capability
- Formula: θ = θ - α * ∇J(θ; x(i:i+n); y(i:i+n))
- Best for: Most practical applications

## Mathematical Foundation

### Core Equations
```python
# Basic update rule
θ(t+1) = θ(t) - α * ∇J(θ(t))

# Learning rate scheduling
α(t) = α₀ / (1 + kt)

Where:
- θ: Model parameters
- α: Learning rate
- ∇J: Gradient of cost function
- t: Current iteration
- k: Decay rate
```

## Advanced Variations

### 1. Momentum
- Adds velocity term to updates
- v(t) = βv(t-1) + (1-β)∇J(θ)
- θ = θ - αv(t)

### 2. Adam
- Combines momentum and RMSprop
- Adaptive learning rates
- State-of-the-art performance

### 3. RMSprop
- Adaptive learning rates
- Handles non-stationary objectives
- Good for RNNs

## Optimization Tips

### Learning Rate Selection
- Start with α = 0.1 or 0.01
- Monitor loss curve
- Use learning rate schedules
- Consider adaptive methods

### Batch Size Guidelines
- Small (32-64): Better generalization
- Large (128-256): Faster training
- Very Large (512+): Distributed training

## Performance Comparison

| Aspect | BGD | SGD | Mini-batch |
|:-------|:----|:----|:-----------|
| Computation | O(n) | O(1) | O(b) |
| Memory | High | Minimal | Moderate |
| Convergence | Deterministic | Stochastic | Semi-stochastic |
| Parallelization | Limited | Poor | Excellent |

## Common Challenges
1. Vanishing/Exploding gradients
2. Saddle points
3. Poor conditioning
4. Local minima

## Best Practices
- Normalize input data
- Monitor gradient norms
- Use gradient clipping
- Implement early stopping
- Cross-validate hyperparameters
