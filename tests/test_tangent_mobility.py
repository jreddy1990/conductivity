"""Hostile tests for state-tangent self-current mobility."""

import numpy as np
import pytest

from conductivity.physical_library import projected_analytical_conductivity as model


def test_one_dimensional_transition_normal_removes_all_tangent_mobility() -> None:
    tangent = model.tangent_mobility(
        mobility_tensor_m2_s=np.asarray([[2.0]], dtype=float),
        transition_normal_gradient_matrix=np.asarray([[3.0]], dtype=float),
    )

    assert tangent == pytest.approx(np.zeros((1, 1)))
    assert np.linalg.matrix_rank(tangent) == 0


def test_rank_deficient_transition_normals_project_only_their_row_space() -> None:
    mobility = np.diag([2.0, 3.0, 5.0])
    redundant_normals = np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float
    )
    tangent = model.tangent_mobility(
        mobility_tensor_m2_s=mobility,
        transition_normal_gradient_matrix=redundant_normals,
    )

    assert tangent == pytest.approx(np.diag([0.0, 3.0, 5.0]))
    assert redundant_normals @ tangent == pytest.approx(np.zeros((2, 3)))
    assert np.linalg.matrix_rank(tangent) == 2


def test_self_current_tangent_tensor_is_symmetric_psd() -> None:
    self_current_tensor = model.compute_self_current_tangent_tensor(
        mobility_tensor_m2_s=np.asarray([[4.0, 1.0], [1.0, 2.0]], dtype=float),
        charge_polarization_gradient=np.asarray(
            [[1.0, 2.0], [0.0, 1.0], [1.0, -1.0]], dtype=float
        ),
        transition_normal_gradient_matrix=np.asarray([[1.0, 1.0]], dtype=float),
    )

    assert self_current_tensor == pytest.approx(self_current_tensor.T)
    assert np.linalg.eigvalsh(self_current_tensor).min() >= -model.PSD_TOL
    assert np.linalg.matrix_rank(self_current_tensor) == 1


def test_tangent_mobility_rejects_indefinite_source_mobility() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.tangent_mobility(
            mobility_tensor_m2_s=np.diag([1.0, -1.0]),
            transition_normal_gradient_matrix=np.asarray([[1.0, 0.0]], dtype=float),
        )


def test_transition_normal_matrix_selects_state_rows() -> None:
    state_normals = model.transition_normal_gradient_matrix_for_state(
        state_index=1,
        coordinate_dimension=2,
        transition_committor_gradients=(
            np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=float),
            np.asarray([[0.0, 1.0]], dtype=float),
        ),
        transition_surface_state_indices=(
            np.asarray([0, 1], dtype=int),
            np.asarray([1], dtype=int),
        ),
    )

    assert state_normals == pytest.approx(
        np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=float)
    )
