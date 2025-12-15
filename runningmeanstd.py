import jax
import jax.numpy as jnp
import pickle


class RunningMeanStd:
    """Running mean and std normalization using pure JAX."""
    
    def __init__(self, shape):
        self.shape = shape
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = 0.0
    
    def update(self, x):
        """Update running statistics with new batch of data."""
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        
        # Welford's online algorithm for combining statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        # Update internal state
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x):
        """Normalize input using current statistics."""
        return (x - self.mean) / jnp.sqrt(self.var + 1e-8)
    
    def __call__(self, x, update_stats=True):
        """Normalize input, optionally updating statistics."""
        if update_stats:
            self.update(x)
        return self.normalize(x)
    
    def save(self, filepath):
        """Save statistics to file."""
        import numpy as np
        state = {
            'mean': np.array(self.mean),
            'var': np.array(self.var),
            'count': float(self.count),
            'shape': self.shape
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Statistics saved to {filepath}")
    
    def load(self, filepath):
        """Load statistics from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.mean = jnp.array(state['mean'])
        self.var = jnp.array(state['var'])
        self.count = state['count']
        self.shape = state['shape']
        print(f"Statistics loaded from {filepath}")
    
    def reset(self):
        """Reset statistics to initial values."""
        self.mean = jnp.zeros(self.shape)
        self.var = jnp.ones(self.shape)
        self.count = 0.0


# ============ USAGE EXAMPLES ============

# Create normalizer
normalizer = RunningMeanStd(shape=(4,))

# Training: normalize and update statistics
key = jax.random.PRNGKey(0)
x1 = jax.random.normal(key, (32, 4))
normalized1 = normalizer(x1, update_stats=True)

print("After batch 1:")
print("Mean:", normalizer.mean)
print("Var:", normalizer.var)
print("Count:", normalizer.count)

# Another batch
x2 = jax.random.normal(jax.random.PRNGKey(1), (32, 4))
normalized2 = normalizer(x2, update_stats=True)

print("\nAfter batch 2:")
print("Count:", normalizer.count)

# Inference: normalize without updating
x_test = jax.random.normal(jax.random.PRNGKey(2), (16, 4))
normalized_test = normalizer(x_test, update_stats=False)

print("\nTest normalized shape:", normalized_test.shape)

# Save and load
normalizer.save('running_stats.pkl')

new_normalizer = RunningMeanStd(shape=(4,))
new_normalizer.load('running_stats.pkl')
print("\nLoaded count:", new_normalizer.count)

# Reset if needed
normalizer.reset()
print("\nAfter reset, count:", normalizer.count)