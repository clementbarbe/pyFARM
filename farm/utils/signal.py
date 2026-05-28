"""Low-level signal processing helpers."""

import numpy as np


def standardize_rows(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation of each row.

    Parameters
    ----------
    x : ndarray, shape (n_rows, n_cols)

    Returns
    -------
    ndarray, same shape, dtype float64.
    """
    x = x.astype(np.float64, copy=True)
    x -= x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std < 1e-12] = 1.0
    return x / std


def corr_with_matrix(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation between *vector* and each row of *matrix*.

    Parameters
    ----------
    vector : 1-D array, length *n*.
    matrix : 2-D array, shape (k, n).

    Returns
    -------
    1-D array of length *k*.
    """
    vector = vector.astype(np.float64)
    matrix = matrix.astype(np.float64)
    vector = vector - vector.mean()
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    vnorm = np.sqrt(np.dot(vector, vector))
    if vnorm < 1e-12:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    mnorm = np.sqrt(np.sum(matrix * matrix, axis=1))
    mnorm[mnorm < 1e-12] = np.inf
    return (matrix @ vector) / (mnorm * vnorm)