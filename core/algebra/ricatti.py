
"""Finite- and infinite-horizon discrete Riccati solvers.

System:
    x[k + 1] = A @ x[k] + B @ u[k]

Cost:
    J = sum(x[k].T @ Q @ x[k] + u[k].T @ R @ u[k])
        + x[N].T @ S @ x[N]

Convention:
    u[k] = -K[k] @ x[k]

This module intentionally contains a scaffold rather than the solution.
"""

import numpy as np

def solve_finite_riccati(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    S: np.ndarray,
    horizon: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return value matrices P[k] and feedback gains K[k].

    Expected output sizes:
        P: horizon + 1 matrices, including P[horizon] = S
        K: horizon matrices, one gain for each control action

    Suggested workflow:
        1. Check matrix shapes and the horizon.
        2. Allocate P and K.
        3. Set the terminal condition P[horizon] = S.
        4. Iterate k backward from horizon - 1 to 0.
        5. Compute K[k] from the stationarity condition.
        6. Compute P[k] from the Bellman/Riccati recursion.
    """
    # check horizon
    if not isinstance(horizon, int):
        raise TypeError("horizon must be an integer")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    
    # Convert
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    Q = np.array(Q, dtype=float)
    R = np.array(R, dtype=float)
    S = np.array(S, dtype=float)

    # Validate inputs
    validate_shapes(A, B, Q, R, S)
    
    # Allocate P and K
    P = [None] * (horizon + 1)
    K = [None] * horizon

    # Terminal condition
    P[horizon] = S

    # Iterate backward in time
    for k in range(horizon-1, -1, -1):
        # Compute the Riccati recursion
        # K[k] = inv(R + B.T @ P[k + 1] @ B) @ B.T @ P[k + 1] @ A
        # P[k] = Q + A.T @ P[k + 1] @ A - A.T @ P[k + 1] @ B @ K[k]
        K[k] = np.linalg.solve(R + B.T @ P[k + 1] @ B, B.T @ P[k + 1] @ A)
        P[k] = Q + A.T @ P[k + 1] @ A - A.T @ P[k + 1] @ B @ K[k]

    return P, K

def solve_infinite_riccati(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the infinite-horizon discrete Riccati equation."""

    # Check iteration settings
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    
    if not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    # Convert
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    Q = np.array(Q, dtype=float)
    R = np.array(R, dtype=float)
    
    # Validate inputs
    validate_shapes(A, B, Q, R)

    # Initialize P
    P = Q.copy()

    for iteration in range(max_iterations):
        # Compute K
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        # Update P
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K

        # Check convergence
        if np.linalg.norm(P_next - P) < tolerance:
            return P_next, K

        P = P_next

    raise RuntimeError("Infinite-horizon Riccati recursion did not converge")

def validate_shapes(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    S: np.ndarray | None = None,
) -> None:
    """Validate dimensions before starting the recursion."""
    # A: (n, n)
    # B: (n, m)
    # Q: (n, n)
    # R: (m, m)
    # S: (n, n)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got {A.ndim}D")
    if B.ndim != 2:
        raise ValueError(f"B must be 2D, got {B.ndim}D")
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2D, got {Q.ndim}D")
    if R.ndim != 2:
        raise ValueError(f"R must be 2D, got {R.ndim}D")
    if S is not None and S.ndim != 2 :
        raise ValueError(f"S must be 2D, got {S.ndim}D")

    n = A.shape[0]
    m = B.shape[1]

    if A.shape[1] != n:
        raise ValueError(f"A must be square with shape ({n}, {n}), got {A.shape}")
    if B.shape[0] != n:
        raise ValueError(f"B must have shape ({n}, {m}), got {B.shape}")
    if Q.shape != (n, n):
        raise ValueError(f"Q must be square with shape ({n}, {n}), got {Q.shape}")
    if R.shape != (m, m):
        raise ValueError(f"R must be square with shape ({m}, {m}), got {R.shape}")
    if S is not None and S.shape != (n, n):
        raise ValueError(f"S must be square with shape ({n}, {n}), got {S.shape}")

    