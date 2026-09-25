"""Input validation utilities."""

import numpy as np
from typing import Optional


def check_array(
    array: np.ndarray,
    ensure_2d: bool = True,
    dtype: type = float,
    copy: bool = False,
) -> np.ndarray:
    """
    Validate and convert input to numpy array.
    
    Parameters
    ----------
    array : array-like
        Input array.
    ensure_2d : bool, default=True
        Ensure output is 2D.
    dtype : type, default=float
        Desired dtype.
    copy : bool, default=False
        Whether to force a copy.
        
    Returns
    -------
    validated : ndarray
        Validated array.
    """
    arr = np.asarray(array, dtype=dtype)
    
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1 and ensure_2d:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ValueError(f"Array must be 1D or 2D, got {arr.ndim}D")
    
    if copy:
        arr = arr.copy()
    
    return arr


def validate_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    ensure_2d: bool = True,
    dtype: type = float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Validate X and optionally y.
    
    Parameters
    ----------
    X : array-like
        Features.
    y : array-like, optional
        Targets.
    ensure_2d : bool, default=True
        Ensure X is 2D.
    dtype : type, default=float
        Desired dtype.
        
    Returns
    -------
    X_validated, y_validated : tuple
        Validated arrays.
    """
    X = check_array(X, ensure_2d=ensure_2d, dtype=dtype)
    
    if y is not None:
        y = np.asarray(y, dtype=dtype).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have different number of samples: "
                f"{X.shape[0]} vs {y.shape[0]}"
            )
    
    return X, y


def check_is_fitted(estimator, attributes: list[str]) -> None:
    """
    Check if estimator is fitted by verifying attributes exist.
    
    Parameters
    ----------
    estimator : object
        Estimator to check.
    attributes : list of str
        Attributes that should exist after fitting.
        
    Raises
    ------
    ValueError
        If estimator is not fitted.
    """
    if not all(hasattr(estimator, attr) for attr in attributes):
        raise ValueError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            f"Call 'fit' with appropriate arguments before using this estimator."
        )


def check_consistent_length(*arrays: np.ndarray) -> None:
    """
    Check that all arrays have consistent first dimension.
    
    Parameters
    ----------
    *arrays : array-like
        Arrays to check.
        
    Raises
    ------
    ValueError
        If lengths are inconsistent.
    """
    lengths = [np.asarray(arr).shape[0] for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent lengths: {lengths}")