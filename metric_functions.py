#!/usr/bin/env python3
"""
Helper functions for working with symmetric positive definite (SPD) matrices.

The main geometries included are:

1. Bures--Wasserstein (BW)
2. Affine-Invariant (AI)
3. Log-Euclidean (LogE)

Workflows will use these functions in three ways:

- Compute a distance between two SPD matrices.
- Compute a mean/reference SPD matrix for tangent-space projection.
- Compute a pairwise distance matrix for manifold UMAP with metric="precomputed".

Expected matrix formats
-----------------------
Individual SPD matrix:
    A.shape == (p, p)

Stack of SPD matrices in this file's mean functions:
    X.shape == (p, p, n)

However, many datasets are stored as:
    X.shape == (n, p, p)

If your data are stored as (n, p, p), convert them by using:
    X_ppn = np.moveaxis(X, 0, 2)

or use the helper:
    X_ppn = stack_npp_to_ppn(X)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm, logm, expm


# ============================================================
# Utilities
# ============================================================
# These are small numerical helper functions.
# SPD computations are sensitive to small numerical asymmetries and tiny
# negative eigenvalues caused by floating-point error. These utilities make the
# matrices safer to use before taking square roots, inverse square roots, logs,
# or exponentials.


def sym(A: np.ndarray) -> np.ndarray:
    """
    Return the symmetric part of a square matrix.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        A square matrix that should be symmetric, up to numerical error.

    Returns
    -------
    ndarray, shape (p, p)
        The matrix (A + A.T) / 2.
    """
    return 0.5 * (A + A.T)



def make_spd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Symmetrize a matrix and floor its eigenvalues to make it SPD.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        Input square matrix.
    eps : float, default=1e-12
        Minimum eigenvalue allowed. If eps <= 0, only symmetrization is done.

    Returns
    -------
    ndarray, shape (p, p)
        A symmetric positive definite matrix.
    """
    A = sym(A)
    if eps <= 0:
        return A

    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T



def sqrtm_spd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the symmetric matrix square root of an SPD matrix.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        SPD matrix.
    eps : float, default=1e-12
        Eigenvalue floor for numerical stability.

    Returns
    -------
    ndarray, shape (p, p)
        The matrix square root A^{1/2}.
    """
    A = make_spd(A, eps=eps)
    w, V = np.linalg.eigh(A)
    if eps > 0:
        w = np.maximum(w, eps)
    return (V * np.sqrt(w)) @ V.T



def invsqrtm_spd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the symmetric inverse square root of an SPD matrix.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        SPD matrix.
    eps : float, default=1e-12
        Eigenvalue floor for numerical stability.

    Returns
    -------
    ndarray, shape (p, p)
        The inverse square root A^{-1/2}.
    """
    A = make_spd(A, eps=eps)
    w, V = np.linalg.eigh(A)
    if eps > 0:
        w = np.maximum(w, eps)
    return (V * (1.0 / np.sqrt(w))) @ V.T



def logm_spd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the matrix logarithm of an SPD matrix using eigen-decomposition.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        SPD matrix.
    eps : float, default=1e-12
        Eigenvalue floor for numerical stability.

    Returns
    -------
    ndarray, shape (p, p)
        The symmetric matrix logarithm log(A).
    """
    A = make_spd(A, eps=eps)
    w, V = np.linalg.eigh(A)
    if eps > 0:
        w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T



def expm_sym(A: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential of a symmetric matrix.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        Symmetric matrix, usually a tangent vector.

    Returns
    -------
    ndarray, shape (p, p)
        The SPD matrix exp(A).
    """
    A = sym(A)
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)) @ V.T



def stack_npp_to_ppn(X: np.ndarray) -> np.ndarray:
    """
    Convert a stack from shape (n, p, p) to shape (p, p, n).

    Parameters
    ----------
    X : ndarray, shape (n, p, p)
        Dataset format commonly used in machine learning.

    Returns
    -------
    ndarray, shape (p, p, n)
        Dataset format expected by the mean routines in this file.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with 3 dimensions, got shape {X.shape}")
    if X.shape[1] != X.shape[2]:
        raise ValueError(f"Expected X with shape (n,p,p), got shape {X.shape}")
    return np.moveaxis(X, 0, 2)



def stack_ppn_to_npp(X: np.ndarray) -> np.ndarray:
    """
    Convert a stack from shape (p, p, n) to shape (n, p, p).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with 3 dimensions, got shape {X.shape}")
    if X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected X with shape (p,p,n), got shape {X.shape}")
    return np.moveaxis(X, 2, 0)


# ============================================================
# BW (Bures--Wasserstein)
# ============================================================

def BW_log(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Parameters
    ----------
    X : ndarray, shape (p, p)
        Current reference SPD matrix.
    Y : ndarray, shape (p, p)
        Target SPD matrix.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    ndarray, shape (p, p)
        A symmetric matrix used as a tangent-like update in the iterative BW
        mean calculation.
    """
    X = make_spd(X, eps=eps)
    Y = make_spd(Y, eps=eps)

    S = sqrtm_spd(X @ Y, eps=eps)
    return sym(S + S.T - 2.0 * X)



def BW_exp(A: np.ndarray, V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Parameters
    ----------
    A : ndarray, shape (p, p)
        Current reference SPD matrix.
    V : ndarray, shape (p, p)
        Symmetric update direction.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    ndarray, shape (p, p)
        Updated SPD matrix.
    """
    A = make_spd(A, eps=eps)
    V = sym(V)

    lam, U = np.linalg.eigh(A)
    if eps > 0:
        lam = np.maximum(lam, eps)

    Vu = U.T @ V @ U

    lam_i = lam[:, None]
    lam_j = lam[None, :]
    W = 1.0 / (lam_i + lam_j)

    C = (W * Vu) @ np.diag(lam) @ (W * Vu)
    out = A + V + U @ C @ U.T
    return make_spd(out, eps=eps)



def compute_BW_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the Bures--Wasserstein distance between two SPD matrices.

    Parameters
    ----------
    X, Y : ndarray, shape (p, p)
        SPD matrices.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    float
        BW distance between X and Y.
    """
    X = make_spd(X, eps=eps)
    Y = make_spd(Y, eps=eps)

    Xh = sqrtm_spd(X, eps=eps)
    mid = Xh @ Y @ Xh
    mid_sqrt = sqrtm_spd(mid, eps=eps)

    d2 = np.trace(X) + np.trace(Y) - 2.0 * np.trace(mid_sqrt)
    d2 = float(np.maximum(d2, 0.0))
    return float(np.sqrt(d2))


# ============================================================
# AI (Affine-Invariant)
# ============================================================

def compute_AI_distance(A: np.ndarray, B: np.ndarray, eps: float | None = None) -> float:
    """
    Compute the Affine-Invariant distance between two SPD matrices.

    Parameters
    ----------
    A, B : ndarray, shape (p, p)
        SPD matrices.
    eps : float or None
        Accepted for compatibility with other functions. This implementation
        does not use eps directly.

    Returns
    -------
    float
        Affine-Invariant Riemannian distance.
    """
    temp = np.dot(np.linalg.inv(A), B)
    lambdas, _ = np.linalg.eig(temp)

    # np.real handles tiny imaginary parts caused by floating-point error.
    out = np.real(np.sqrt(np.sum(np.log(np.real(lambdas)) ** 2) + 0j))
    return float(out)



def AI_log(A: np.ndarray, B: np.ndarray, eps: float | None = None) -> np.ndarray:
    """
    Compute the Affine-Invariant logarithm map of B at base point A.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        Base/reference SPD matrix.
    B : ndarray, shape (p, p)
        Target SPD matrix.
    eps : float or None
        Accepted for compatibility. This SciPy version does not use eps.

    Returns
    -------
    ndarray, shape (p, p)
        Tangent vector at A pointing toward B.
    """
    Ainv_sqrt = sqrtm(np.linalg.inv(A))
    temp = logm(Ainv_sqrt.dot(B).dot(Ainv_sqrt))
    out = sqrtm(A).dot(temp).dot(sqrtm(A))
    return np.real(sym(out))



def AI_exp(A: np.ndarray, B: np.ndarray, eps: float | None = None) -> np.ndarray:
    """
    Compute the Affine-Invariant exponential map at base point A.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        Base/reference SPD matrix.
    B : ndarray, shape (p, p)
        Tangent vector at A.
    eps : float or None
        Accepted for compatibility. This SciPy version does not use eps.

    Returns
    -------
    ndarray, shape (p, p)
        SPD matrix reached by moving from A in tangent direction B.
    """
    Ainv_sqrt = sqrtm(np.linalg.inv(A))
    temp = expm(Ainv_sqrt.dot(B).dot(Ainv_sqrt))
    out = sqrtm(A).dot(temp).dot(sqrtm(A))
    return make_spd(np.real(sym(out)), eps=1e-12)


# ============================================================
# Log-Euclidean (LogE)
# ============================================================

def LogE_mean(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Parameters
    ----------
    X : ndarray, shape (p, p, n)
        SPD matrices stacked along the third axis.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    ndarray, shape (p, p)
        Log-Euclidean mean.

    Project use
    -----------
    Use this to compute the reference matrix for LogE tangent projection.
    """
    n = X.shape[2]
    acc = np.zeros((X.shape[0], X.shape[1]), dtype=X.dtype)
    for i in range(n):
        acc += logm_spd(X[:, :, i], eps=eps)
    acc /= float(n)
    return make_spd(expm_sym(acc), eps=eps)



def LogE_log(M: np.ndarray, C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the Log-Euclidean log map of C at reference matrix M.

    Parameters
    ----------
    M : ndarray, shape (p, p)
        Reference SPD matrix, usually the LogE mean.
    C : ndarray, shape (p, p)
        SPD matrix to project.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    ndarray, shape (p, p)
        Tangent vector log(C) - log(M).
    """
    M = make_spd(M, eps=eps)
    C = make_spd(C, eps=eps)
    return sym(logm_spd(C, eps=eps) - logm_spd(M, eps=eps))



def LogE_exp(M: np.ndarray, V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the Log-Euclidean exponential map at reference matrix M.

    Parameters
    ----------
    M : ndarray, shape (p, p)
        Reference SPD matrix.
    V : ndarray, shape (p, p)
        Tangent vector.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    ndarray, shape (p, p)
        SPD matrix exp(log(M) + V).
    """
    M = make_spd(M, eps=eps)
    V = sym(V)
    return make_spd(expm_sym(logm_spd(M, eps=eps) + V), eps=eps)



def LogE_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the Log-Euclidean distance between two SPD matrices.

    Parameters
    ----------
    X, Y : ndarray, shape (p, p)
        SPD matrices.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    float
        Frobenius norm of log(X) - log(Y).
    """
    X = make_spd(X, eps=eps)
    Y = make_spd(Y, eps=eps)
    return float(np.linalg.norm(logm_spd(X, eps=eps) - logm_spd(Y, eps=eps), ord="fro"))


# ============================================================
# Backward compatibility of older function names
# ============================================================
# These aliases let older code use BW_dist() and AI_dist().


def BW_dist(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    """Alias for compute_BW_distance()."""
    return compute_BW_distance(A, B, eps=eps)



def AI_dist(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    """Alias for compute_AI_distance()."""
    return compute_AI_distance(A, B, eps=eps)


# ============================================================
# Barycenter for tangent-space projection
# ============================================================
# These functions compute the reference point/mean on the SPD manifold. For
# statistically correct cross-validation, compute the mean using the training
# fold only, then project both train and test matrices using that training mean.


def bw_projection_mean(
    X: np.ndarray,
    tol: float,
    verbose: bool = False,
    spd_eps: float = 1e-12,
    max_iter: int = 500,
):
    """
    Compute an iterative Bures--Wasserstein mean.

    Parameters
    ----------
    X : ndarray, shape (p, p, n)
        SPD matrices stacked on the third axis.
    tol : float
        Stop when the change between successive means is below this value.
    verbose : bool, default=False
        If True, print the convergence trace.
    spd_eps : float, default=1e-12
        Numerical eigenvalue floor.
    max_iter : int, default=500
        Maximum number of iterations.

    Returns
    -------
    mean : ndarray, shape (p, p)
        Estimated BW mean.
    iters : int
        Number of iterations used.
    final_dist : float
        Final distance between successive mean estimates.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (p,p,n), got {X.shape}")

    p, p2, n = X.shape
    if p != p2:
        raise ValueError(f"Expected square matrices, got {X.shape}")

    mean_new = np.mean(X, axis=2)
    dist_mean = float("inf")
    k = 0

    while dist_mean > tol and k < max_iter:
        U, s, VT = np.linalg.svd(mean_new, full_matrices=True)
        mean_old = VT.T @ np.diag(s) @ VT
        mean_old = sym(mean_old)
        mean_old = make_spd(mean_old, eps=spd_eps)

        acc = np.zeros((p, p), dtype=X.dtype)
        for i in range(n):
            acc += BW_log(mean_old, X[:, :, i], eps=spd_eps)
        mean_tangent = acc / float(n)

        mean_new = np.real(BW_exp(mean_old, mean_tangent, eps=spd_eps))
        mean_new = make_spd(mean_new, eps=spd_eps)

        dist_mean = BW_dist(mean_new, mean_old, eps=spd_eps)

        k += 1
        if verbose:
            print(f"Iter{k}, dist_mean={dist_mean:.7f}")

    return mean_new, k, float(dist_mean)



def ai_projection_mean(
    X: np.ndarray,
    tol: float,
    verbose: bool = False,
    spd_eps: float = 1e-12,
    max_iter: int = 500,
):
    """
    Compute an iterative Affine-Invariant mean.

    Parameters
    ----------
    X : ndarray, shape (p, p, n)
        SPD matrices stacked on the third axis.
    tol : float
        Convergence tolerance.
    verbose : bool, default=False
        If True, print the convergence trace.
    spd_eps : float, default=1e-12
        Numerical eigenvalue floor.
    max_iter : int, default=500
        Maximum number of iterations.

    Returns
    -------
    mean : ndarray, shape (p, p)
        Estimated AI mean.
    iters : int
        Number of iterations used.
    final_dist : float
        Final step distance.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (p,p,n), got {X.shape}")

    p, p2, n = X.shape
    if p != p2:
        raise ValueError(f"Expected square matrices, got {X.shape}")

    mean_new = np.mean(X, axis=2)
    dist_mean = float("inf")
    smaller_step = True
    k = 0
    last_dist = None

    mean_old = make_spd(sym(mean_new), eps=spd_eps)

    while dist_mean > tol and smaller_step and k < max_iter:
        U, s, VT = np.linalg.svd(mean_new, full_matrices=True)
        mean_old_candidate = VT.T @ np.diag(s) @ VT
        mean_old_candidate = sym(mean_old_candidate)
        mean_old_candidate = make_spd(mean_old_candidate, eps=spd_eps)

        acc = np.zeros((p, p), dtype=X.dtype)
        for i in range(n):
            acc += AI_log(mean_old_candidate, X[:, :, i], eps=spd_eps)
        mean_tangent = acc / float(n)

        mean_new_candidate = np.real(AI_exp(mean_old_candidate, mean_tangent, eps=spd_eps))
        mean_new_candidate = make_spd(mean_new_candidate, eps=spd_eps)

        dist_mean = AI_dist(mean_new_candidate, mean_old_candidate, eps=spd_eps)

        if k == 0:
            smaller_step = True
        else:
            smaller_step = (last_dist - dist_mean) > 1e-4

        if verbose:
            ss = "NA" if k == 0 else str(smaller_step)
            print(f"Iter{k}, Smaller step? {ss}, dist_mean={dist_mean:.7f}")

        if smaller_step:
            mean_old = mean_new_candidate
            mean_new = mean_new_candidate

        k += 1
        last_dist = dist_mean

    return mean_old, k, float(dist_mean)


# Alias expected by compute_mean_projection
ai_projection_mean_stable = ai_projection_mean


def compute_mean_projection(
    X: np.ndarray,
    metric: str,
    tol: float = 1e-6,
    spd_eps: float = 1e-12,
    max_iter: int = 500,
    verbose: bool = False,
):
    """
    Convenience wrapper to compute the mean/reference SPD matrix for one metric.

    Parameters
    ----------
    X : ndarray, shape (p, p, n)
        SPD matrices stacked on the third axis.
    metric : {'bw', 'ai', 'loge'}
        Geometry used for the mean.
    tol : float, default=1e-6
        Convergence tolerance for BW and AI.
    spd_eps : float, default=1e-12
        Numerical eigenvalue floor.
    max_iter : int, default=500
        Maximum iterations for BW and AI.
    verbose : bool, default=False
        If True, print convergence information.

    Returns
    -------
    mean : ndarray, shape (p, p)
        Mean/reference SPD matrix.
    info : dict
        Small dictionary with convergence information.

    Project use
    -----------
    This gives students one function to call regardless of metric.
    """
    metric = metric.lower()
    if metric in {"bw", "bures", "bures-wasserstein", "bures_wasserstein"}:
        M, iters, final = bw_projection_mean(
            X, tol=tol, verbose=verbose, spd_eps=spd_eps, max_iter=max_iter
        )
        return M, {"metric": "bw", "iters": iters, "final": final}

    if metric in {"ai", "airm", "affine", "affine-invariant", "affine_invariant"}:
        M, iters, final = ai_projection_mean_stable(
            X, tol=tol, max_iter=max_iter, verbose=verbose
        )
        return M, {"metric": "ai", "iters": iters, "final": final}

    if metric in {"loge", "log-euclidean", "log_euclidean"}:
        M = LogE_mean(X, eps=spd_eps)
        return M, {"metric": "loge", "iters": 0, "final": 0.0}

    raise ValueError("metric must be one of: 'bw', 'ai', or 'loge'")


# ============================================================
# Tangent projection helpers
# ============================================================

def project_to_tangent(M: np.ndarray, C: np.ndarray, metric: str, eps: float = 1e-12) -> np.ndarray:
    """
    Project one SPD matrix C to the tangent space at reference matrix M.

    Parameters
    ----------
    M : ndarray, shape (p, p)
        Reference/mean SPD matrix.
    C : ndarray, shape (p, p)
        SPD matrix to project.
    metric : {'bw', 'ai', 'loge'}
        Geometry used for the projection.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    ndarray, shape (p, p)
        Symmetric tangent matrix.
    """
    metric = metric.lower()
    if metric in {"bw", "bures", "bures-wasserstein", "bures_wasserstein"}:
        return BW_log(M, C, eps=eps)
    if metric in {"ai", "airm", "affine", "affine-invariant", "affine_invariant"}:
        return AI_log(M, C, eps=eps)
    if metric in {"loge", "log-euclidean", "log_euclidean"}:
        return LogE_log(M, C, eps=eps)
    raise ValueError("metric must be one of: 'bw', 'ai', or 'loge'")



def vectorize_symmetric(A: np.ndarray) -> np.ndarray:
    """
    Vectorize the upper triangle of a symmetric matrix.

    Parameters
    ----------
    A : ndarray, shape (p, p)
        Symmetric matrix.

    Returns
    -------
    ndarray, shape (p * (p + 1) / 2,)
        Vectorized upper triangle.
    """
    A = sym(A)
    p = A.shape[0]
    idx = np.triu_indices(p)
    v = A[idx].copy()
    return v



def project_stack_to_tangent_features(
    X: np.ndarray,
    M: np.ndarray,
    metric: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Project a stack of SPD matrices to vectorized tangent features.

    Parameters
    ----------
    X : ndarray, shape (n, p, p) or (p, p, n)
        Stack of SPD matrices.
    M : ndarray, shape (p, p)
        Reference/mean SPD matrix.
    metric : {'bw', 'ai', 'loge'}
        Geometry used for tangent projection.
    eps : float, default=1e-12
        Numerical eigenvalue floor.

    Returns
    -------
    ndarray, shape (n, p * (p + 1) / 2)
        Tangent feature matrix for scikit-learn.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {X.shape}")

    # Accept either common storage convention.
    if X.shape[1] == X.shape[2]:
        X_npp = X
    elif X.shape[0] == X.shape[1]:
        X_npp = stack_ppn_to_npp(X)
    else:
        raise ValueError(f"Could not identify matrix stack format from shape {X.shape}")

    features = []
    for C in X_npp:
        T = project_to_tangent(M, C, metric=metric, eps=eps)
        features.append(vectorize_symmetric(T))

    return np.vstack(features)


# ============================================================
# Pairwise distance matrix for manifold UMAP
# ============================================================
# Manifold UMAP with metric='precomputed' needs an n x n distance matrix. This
# function computes that matrix using AI, LogE, or BW distances directly on the
# SPD matrices.


def compute_pairwise_distance_matrix(
    X: np.ndarray,
    metric: str = "bw",
    eps: float = 1e-12,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute an n x n pairwise distance matrix for a stack of SPD matrices.

    Parameters
    ----------
    X : ndarray, shape (n, p, p) or (p, p, n)
        Stack of SPD matrices.
    metric : {'bw', 'ai', 'loge'}, default='bw'
        Distance metric to use.
    eps : float, default=1e-12
        Numerical eigenvalue floor.
    verbose : bool, default=True
        If True, print occasional progress updates.

    Returns
    -------
    D : ndarray, shape (n, n)
        Symmetric pairwise distance matrix with zeros on the diagonal.

    Notes
    -----
    This is O(n^2) in the number of matrices, so start with a smaller subset
    while debugging.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {X.shape}")

    if X.shape[1] == X.shape[2]:
        X_npp = X
    elif X.shape[0] == X.shape[1]:
        X_npp = stack_ppn_to_npp(X)
    else:
        raise ValueError(f"Could not identify matrix stack format from shape {X.shape}")

    metric = metric.lower()
    if metric in {"bw", "bures", "bures-wasserstein", "bures_wasserstein"}:
        dist_fun = lambda A, B: compute_BW_distance(A, B, eps=eps)
    elif metric in {"ai", "airm", "affine", "affine-invariant", "affine_invariant"}:
        dist_fun = lambda A, B: compute_AI_distance(A, B, eps=eps)
    elif metric in {"loge", "log-euclidean", "log_euclidean"}:
        dist_fun = lambda A, B: LogE_distance(A, B, eps=eps)
    else:
        raise ValueError("metric must be one of: 'bw', 'ai', or 'loge'")

    n = X_npp.shape[0]
    D = np.zeros((n, n), dtype=float)

    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fun(X_npp[i], X_npp[j])
            D[i, j] = d
            D[j, i] = d
            done += 1

        if verbose and (i % 25 == 0 or i == n - 1):
            print(f"Computed distances for row {i + 1}/{n} ({done}/{total_pairs} pairs)")

    return D
