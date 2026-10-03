"""Random number generation utilities for reproducibility."""

import numpy as np
from typing import Optional, Union


# Global random state
_global_rng: Optional[np.random.Generator] = None


def set_seed(seed: Optional[int] = None) -> np.random.Generator:
    """
    Set global random seed for reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        Random seed. If None, uses system randomness.
        
    Returns
    -------
    rng : Generator
        The global random number generator.
    """
    global _global_rng
    _global_rng = np.random.default_rng(seed)
    return _global_rng


def get_rng() -> np.random.Generator:
    """
    Get the global random number generator.
    Creates one with system entropy if not already set.
    
    Returns
    -------
    rng : Generator
        The global random number generator.
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = np.random.default_rng()
    return _global_rng


def shuffle_arrays(
    *arrays: np.ndarray,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> list[np.ndarray]:
    """
    Shuffle multiple arrays in the same order.
    
    Parameters
    ----------
    *arrays : array-like
        Arrays to shuffle (all must have same first dimension).
    random_state : int or Generator, optional
        Random seed or generator.
        
    Returns
    -------
    shuffled : list of ndarray
        Shuffled arrays.
    """
    if not arrays:
        return []
    
    if isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    elif random_state is None:
        rng = get_rng()
    else:
        rng = random_state
    
    n_samples = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != n_samples:
            raise ValueError("All arrays must have same first dimension")
    
    indices = rng.permutation(n_samples)
    return [arr[indices] for arr in arrays]


def train_test_split_indices(
    n_samples: int,
    test_size: float = 0.2,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate train/test split indices.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples.
    test_size : float, default=0.2
        Fraction of test samples.
    random_state : int or Generator, optional
        Random seed or generator.
        
    Returns
    -------
    train_idx, test_idx : tuple of ndarray
        Training and test indices.
    """
    if isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    elif random_state is None:
        rng = get_rng()
    else:
        rng = random_state
    
    n_test = int(n_samples * test_size)
    indices = rng.permutation(n_samples)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return train_idx, test_idx