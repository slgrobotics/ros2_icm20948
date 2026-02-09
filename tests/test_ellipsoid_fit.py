import numpy as np
import pytest

# Import your function under test
# Adjust the import path to wherever ellipsoid_fit lives.
# Example: from your_calibration_module import __ellipsoid_fit
#from your_calibration_module import __ellipsoid_fit


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a random 3x3 proper rotation matrix (det=+1)."""
    # Random orthonormal basis via QR
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    # Ensure right-handed (det=+1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _sample_unit_sphere_points(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform-ish points on unit sphere."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test__ellipsoid_fit_recovers_known_parameters():
    rng = np.random.default_rng(12345)

    n = 5000
    center_true = np.array([-12.5, -10.8, 23.7], dtype=float)  # µT bias
    radii_true = np.array([42.0, 33.0, 28.0], dtype=float)     # µT "soft-iron" radii
    R_true = _random_rotation_matrix(rng)

    # Generate ellipsoid surface points:
    U = _sample_unit_sphere_points(rng, n)              # on unit sphere
    X = (U * radii_true) @ R_true.T + center_true       # scale, rotate, translate

    # Add measurement noise (µT)
    noise_sigma = 0.3
    X_noisy = X + rng.normal(scale=noise_sigma, size=X.shape)

    center_fit, evecs_fit, radii_fit, v = ellipsoid_fit(X_noisy)

    print("center_fit:", center_fit)
    print("radii_fit:", radii_fit)

    # --- sanity checks ---
    assert np.all(np.isfinite(center_fit))
    assert np.all(np.isfinite(radii_fit))
    assert np.all(radii_fit > 0.0)

    # Eigenvectors should be orthonormal-ish
    # Note: your code returns evecs as evecs.T after eigh (rows are eigenvectors).
    # Either way, check orthonormality in a tolerant way.
    E = np.asarray(evecs_fit, dtype=float)
    # If eigenvectors are rows, E @ E.T ~ I; if columns, E.T @ E ~ I.
    err_rows = np.linalg.norm(E @ E.T - np.eye(3), ord="fro")
    err_cols = np.linalg.norm(E.T @ E - np.eye(3), ord="fro")
    assert min(err_rows, err_cols) < 1e-2

    # Center should be close (noise-limited). Tune tolerance as needed.
    assert np.linalg.norm(center_fit - center_true) < 2.0

    # Radii should match up to permutation (eigenvectors may reorder axes)
    r_fit = np.sort(np.asarray(radii_fit, dtype=float))
    r_true = np.sort(radii_true)
    # Relative error tolerance
    rel_err = np.max(np.abs(r_fit - r_true) / r_true)
    assert rel_err < 0.08  # 8% is conservative with noise; tighten if desired

    # Reject absurd solutions (your failure mode)
    assert r_fit[-1] < 200.0
    assert (r_fit[-1] / r_fit[0]) < 5.0


def test__ellipsoid_fit_detects_degenerate_planar_data():
    rng = np.random.default_rng(54321)

    n = 2000
    center_true = np.array([5.0, -3.0, 12.0], dtype=float)
    radii_true = np.array([45.0, 30.0, 25.0], dtype=float)
    R_true = _random_rotation_matrix(rng)

    U = _sample_unit_sphere_points(rng, n)
    X = (U * radii_true) @ R_true.T + center_true

    # Crush Z variation (degenerate)
    X[:, 2] = center_true[2] + rng.normal(scale=0.2, size=n)

    with pytest.raises(ValueError):
        ellipsoid_fit(X)


def ellipsoid_fit(X: np.ndarray):
    """
    General quadric least-squares ellipsoid fit.

    Returns:
        center (3,)
        evecs  (3,3) columns are principal axes
        radii  (3,)
        v      (10,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must be (N,3)")

    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    # Solve D @ v ≈ 1 (unconstrained; overall scale is arbitrary)
    D = np.column_stack([
        x * x, y * y, z * z,
        2 * x * y, 2 * x * z, 2 * y * z,
        2 * x, 2 * y, 2 * z,
        np.ones_like(x),
    ])
    rhs = np.ones((X.shape[0],), dtype=float)
    v, *_ = np.linalg.lstsq(D, rhs, rcond=None)

    # Build symmetric quadric matrix A (4x4)
    A = np.array([
        [v[0], v[3], v[4], v[6]],
        [v[3], v[1], v[5], v[7]],
        [v[4], v[5], v[2], v[8]],
        [v[6], v[7], v[8], v[9]],
    ], dtype=float)

    Q = A[:3, :3]
    b = A[:3, 3]

    # Q must be invertible to get a finite center
    if not np.all(np.isfinite(Q)) or np.linalg.cond(Q) > 1e12:
        raise ValueError("Degenerate quadric: Q ill-conditioned (insufficient 3D excitation).")

    center = -np.linalg.solve(Q, b)

    # Normalize using data: enforce (x-center)^T Qn (x-center) ≈ 1
    U = X - center
    s = np.einsum("ni,ij,nj->n", U, Q, U)

    # Overall quadric scale is arbitrary; s can be tiny — that's OK.
    # Use robust scale and handle sign.
    s0 = float(np.median(s))
    if not np.isfinite(s0) or abs(s0) < 1e-30:
        # fallback: use median absolute value
        s0 = float(np.median(np.abs(s)))
        if not np.isfinite(s0) or s0 < 1e-30:
            raise ValueError("Degenerate fit: quadratic form is numerically zero.")

    # Make scale positive (ellipsoid wants positive definite Qn)
    alpha = 1.0 / s0
    Qn = Q * alpha
    Qn = 0.5 * (Qn + Qn.T)

    evals, evecs = np.linalg.eigh(Qn)

    # For ellipsoid, all eigenvalues must be > 0
    if np.any(evals <= 1e-12):
        # Sometimes sign is flipped; try flipping once
        Qn2 = -Qn
        evals2, evecs2 = np.linalg.eigh(Qn2)
        if np.any(evals2 <= 1e-12):
            raise ValueError("Not an ellipsoid (non-positive eigenvalues). Likely degenerate / insufficient excitation.")
        evals, evecs = evals2, evecs2

    radii = 1.0 / np.sqrt(evals)
    return center, evecs, radii, v

