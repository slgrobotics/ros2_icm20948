import numpy as np
import pytest
from tests.ellipsoid_fit import ellipsoid_fit


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


