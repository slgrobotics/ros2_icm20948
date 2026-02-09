import numpy as np

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

    # Start from a real ellipsoid...
    U = _sample_unit_sphere_points(rng, n)
    X = (U * radii_true) @ R_true.T + center_true

    # ...then crush Z variation (simulate mostly yaw rotations / poor 3D excitation)
    X[:, 2] = center_true[2] + rng.normal(scale=0.2, size=n)

    center_fit, evecs_fit, radii_fit, v = ellipsoid_fit(X)

    r = np.sort(np.asarray(radii_fit, dtype=float))

    # In degenerate cases you typically see one radius explode (like 1e6)
    # or an extreme aspect ratio.
    assert (r[-1] > 500.0) or ((r[-1] / r[0]) > 20.0)

def ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                x * x + z * z - 2 * y * y,
                2 * x * y,
                2 * x * z,
                2 * y * z,
                2 * x,
                2 * y,
                2 * z,
                1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    M = D.dot(D.T)
    u = np.linalg.solve(M + 1e-9*np.eye(M.shape[0]), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                [v[3], v[1], v[5], v[7]],
                [v[4], v[5], v[2], v[8]],
                [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = center

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    S = R[:3, :3] / -R[3, 3]
    S = 0.5*(S + S.T)
    evals, evecs = np.linalg.eigh(S)  # evecs columns are eigenvectors
    evecs = evecs.T

    radii = np.sqrt(1.0 / np.abs(evals))

    return center, evecs, radii, v

"""
def ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                x * x + z * z - 2 * y * y,
                2 * x * y,
                2 * x * z,
                2 * y * z,
                2 * x,
                2 * y,
                2 * z,
                1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    M = D.dot(D.T)
    u = np.linalg.solve(M + 1e-9*np.eye(M.shape[0]), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                [v[3], v[1], v[5], v[7]],
                [v[4], v[5], v[2], v[8]],
                [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = center

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    S = R[:3, :3] / -R[3, 3]
    S = 0.5*(S + S.T)
    evals, evecs = np.linalg.eigh(S)
    evecs = evecs.T

    radii = np.sqrt(1.0 / np.abs(evals))

    return center, evecs, radii, v
"""
