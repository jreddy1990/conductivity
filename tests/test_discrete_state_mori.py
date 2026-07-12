import numpy as np

from conductivity.physical_library.projected_analytical_conductivity import (
    compute_continuous_mori_correction,
    compute_discrete_state_mori_matrices,
)


def _reversible_two_state_matrices(
    transition_rate_s_inv: float,
    displacement_m: float,
    coordinate_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    concentrations = np.ones(2, dtype=float)
    generator = transition_rate_s_inv * np.array([[-1.0, 1.0], [1.0, -1.0]])
    first_moments = displacement_m * np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    memory_values = coordinate_scale * np.array([[-1.0], [1.0]])
    return concentrations, generator, first_moments, memory_values


def test_reversible_two_state_discrete_mori_formula() -> None:
    transition_rate_s_inv = 3.0
    displacement_m = 2.0e-10
    inputs = _reversible_two_state_matrices(transition_rate_s_inv, displacement_m)

    memory_matrix, current_coupling = compute_discrete_state_mori_matrices(*inputs)

    np.testing.assert_allclose(memory_matrix, [[4.0 * transition_rate_s_inv]])
    np.testing.assert_allclose(
        current_coupling,
        [[-2.0 * transition_rate_s_inv * displacement_m, 0.0, 0.0]],
    )
    correction = compute_continuous_mori_correction(memory_matrix, current_coupling)
    np.testing.assert_allclose(
        correction[0, 0], transition_rate_s_inv * displacement_m**2
    )


def test_discrete_mori_tracks_transition_rate() -> None:
    slow = _reversible_two_state_matrices(2.0, 1.0e-10)
    fast = _reversible_two_state_matrices(10.0, 1.0e-10)
    slow_memory, slow_coupling = compute_discrete_state_mori_matrices(*slow)
    fast_memory, fast_coupling = compute_discrete_state_mori_matrices(*fast)

    np.testing.assert_allclose(fast_memory, 5.0 * slow_memory)
    np.testing.assert_allclose(fast_coupling, 5.0 * slow_coupling)
    np.testing.assert_allclose(
        compute_continuous_mori_correction(fast_memory, fast_coupling),
        5.0 * compute_continuous_mori_correction(slow_memory, slow_coupling),
    )


def test_discrete_mori_coordinate_rescaling_invariance() -> None:
    base = _reversible_two_state_matrices(4.0, 3.0e-10)
    rescaled = _reversible_two_state_matrices(4.0, 3.0e-10, coordinate_scale=7.0)
    base_memory, base_coupling = compute_discrete_state_mori_matrices(*base)
    scaled_memory, scaled_coupling = compute_discrete_state_mori_matrices(*rescaled)

    np.testing.assert_allclose(scaled_memory, 49.0 * base_memory)
    np.testing.assert_allclose(scaled_coupling, 7.0 * base_coupling)
    np.testing.assert_allclose(
        compute_continuous_mori_correction(scaled_memory, scaled_coupling),
        compute_continuous_mori_correction(base_memory, base_coupling),
    )


def test_discrete_state_memory_matrix_is_positive_semidefinite() -> None:
    concentrations = np.array([1.0, 2.0, 4.0])
    generator = np.array(
        [[-3.0, 2.0, 1.0], [1.0, -2.5, 1.5], [0.25, 0.75, -1.0]]
    )
    first_moments = np.zeros((3, 3, 3), dtype=float)
    memory_values = np.array([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]])

    memory_matrix, _current_coupling = compute_discrete_state_mori_matrices(
        concentrations, generator, first_moments, memory_values
    )

    assert np.min(np.linalg.eigvalsh(memory_matrix)) >= -1.0e-12

