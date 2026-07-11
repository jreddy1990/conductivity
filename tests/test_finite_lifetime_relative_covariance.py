import numpy as np
import pytest

from constants import T_REF_K
from conductivity.physical_library.projected_analytical_conductivity import (
    apply_finite_lifetime_relative_covariance,
)


def _apply_diagonal_case(
    lifetime_rate_s_inv: float,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    covariance_diagonal_m2 = np.asarray([1.0e-20, 2.0e-20, 4.0e-20])
    relative_mobility_diagonal_m2_s = np.asarray([2.0e-10, 3.0e-10, 5.0e-10])
    weighted_fluctuations = np.diag(np.sqrt(covariance_diagonal_m2))
    relative_mobility = np.diag(relative_mobility_diagonal_m2_s)
    generator = np.asarray(
        [
            [-lifetime_rate_s_inv, lifetime_rate_s_inv],
            [lifetime_rate_s_inv, -lifetime_rate_s_inv],
        ],
        dtype=float,
    )
    capacity_fluxes = np.asarray(
        [[0.0, lifetime_rate_s_inv], [lifetime_rate_s_inv, 0.0]], dtype=float
    )
    self_currents = np.stack((relative_mobility, np.zeros((3, 3))), axis=0)
    return apply_finite_lifetime_relative_covariance(
        self_current_tensors_D_self_i_m2_s=self_currents,
        symmetric_capacity_fluxes_K_ij_mol_m3_s=capacity_fluxes,
        reversible_generator_Q_ij_s_inv=generator,
        transition_second_moments_M_ij_m2=np.zeros((2, 2, 3, 3)),
        state_concentrations_mol_m3=np.ones(2),
        state_relative_displacement_fluctuations_m=(
            weighted_fluctuations,
            np.empty((0, 3)),
        ),
        state_relative_displacement_mobilities_m2_s=(
            relative_mobility,
            np.empty((0, 0)),
        ),
        state_relative_center_charge_numbers=(
            np.asarray([1.0, -1.0]),
            np.empty(0),
        ),
        transition_displacement_edge_mask=capacity_fluxes > 0.0,
        temperature_K=T_REF_K,
    )


def test_finite_lifetime_covariance_matches_diagonal_closed_form() -> None:
    lifetime_rate_s_inv = 4.0e9
    adjusted_self_currents, adjusted_second_moments, diagnostics = _apply_diagonal_case(
        lifetime_rate_s_inv
    )
    covariance = np.asarray([1.0e-20, 2.0e-20, 4.0e-20])
    mobility = np.asarray([2.0e-10, 3.0e-10, 5.0e-10])
    expected = (
        lifetime_rate_s_inv
        * covariance
        * mobility
        / (lifetime_rate_s_inv * covariance + mobility)
    )
    np.testing.assert_allclose(adjusted_self_currents[0], np.zeros((3, 3)))
    np.testing.assert_allclose(
        lifetime_rate_s_inv * np.diag(adjusted_second_moments[0, 1]),
        expected,
    )
    np.testing.assert_allclose(
        adjusted_second_moments[1, 0], adjusted_second_moments[0, 1]
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].center_covariance_min_eigenvalue_m2_s >= -1.0e-25
    assert diagnostics[0].dc_self_trace_m2_s == 0.0
    assert diagnostics[0].transition_owned_trace_m2_s == pytest.approx(
        float(np.sum(expected))
    )
    assert diagnostics[0].short_trace_m2_s == pytest.approx(float(np.sum(mobility)))
    assert diagnostics[0].bounded_memory_trace_m2_s == pytest.approx(
        float(np.sum(mobility - expected))
    )


def test_finite_lifetime_covariance_uses_k_weighted_transition_ownership() -> None:
    outgoing_capacities = np.asarray([1.0e9, 3.0e9])
    capacity_fluxes = np.asarray(
        [
            [0.0, outgoing_capacities[0], outgoing_capacities[1]],
            [outgoing_capacities[0], 0.0, 0.0],
            [outgoing_capacities[1], 0.0, 0.0],
        ]
    )
    generator = capacity_fluxes.copy()
    np.fill_diagonal(generator, -np.sum(generator, axis=1))
    relative_mobility = np.diag([2.0e-10, 3.0e-10, 5.0e-10])
    covariance_diagonal_m2 = np.asarray([1.0e-20, 2.0e-20, 4.0e-20])

    adjusted_self, adjusted_moments, diagnostics = (
        apply_finite_lifetime_relative_covariance(
            self_current_tensors_D_self_i_m2_s=np.asarray(
                [relative_mobility, np.zeros((3, 3)), np.zeros((3, 3))]
            ),
            symmetric_capacity_fluxes_K_ij_mol_m3_s=capacity_fluxes,
            reversible_generator_Q_ij_s_inv=generator,
            transition_second_moments_M_ij_m2=np.zeros((3, 3, 3, 3)),
            state_concentrations_mol_m3=np.ones(3),
            state_relative_displacement_fluctuations_m=(
                np.diag(np.sqrt(covariance_diagonal_m2)),
                np.empty((0, 3)),
                np.empty((0, 3)),
            ),
            state_relative_displacement_mobilities_m2_s=(
                relative_mobility,
                np.empty((0, 0)),
                np.empty((0, 0)),
            ),
            state_relative_center_charge_numbers=(
                np.asarray([1.0, -1.0]),
                np.empty(0),
                np.empty(0),
            ),
            transition_displacement_edge_mask=capacity_fluxes > 0.0,
            temperature_K=T_REF_K,
        )
    )

    np.testing.assert_allclose(adjusted_self[0], np.zeros((3, 3)))
    np.testing.assert_allclose(adjusted_moments[0, 1], adjusted_moments[0, 2])
    transition_direct = 0.5 * np.einsum(
        "ij,ijab->ab", capacity_fluxes, adjusted_moments
    )
    assert diagnostics[0].transition_owned_trace_m2_s == pytest.approx(
        float(np.trace(transition_direct))
    )
    lifetime_rate = float(np.sum(outgoing_capacities))
    expected_transition_direct = np.diag(
        lifetime_rate
        * covariance_diagonal_m2
        * np.diag(relative_mobility)
        / (lifetime_rate * covariance_diagonal_m2 + np.diag(relative_mobility))
    )
    np.testing.assert_allclose(transition_direct, expected_transition_direct)
    np.testing.assert_allclose(
        capacity_fluxes[0, 2] * adjusted_moments[0, 2],
        3.0 * capacity_fluxes[0, 1] * adjusted_moments[0, 1],
    )


def test_finite_lifetime_covariance_has_correct_hostile_lifetime_limits() -> None:
    zero_lifetime_self_currents, zero_lifetime_moments, _ = _apply_diagonal_case(0.0)
    np.testing.assert_allclose(zero_lifetime_self_currents[0], np.zeros((3, 3)))
    np.testing.assert_allclose(zero_lifetime_moments, np.zeros((2, 2, 3, 3)))

    fast_lifetime_self_currents, fast_lifetime_moments, _ = _apply_diagonal_case(1.0e30)
    np.testing.assert_allclose(
        fast_lifetime_self_currents[0],
        np.zeros((3, 3)),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        1.0e30 * fast_lifetime_moments[0, 1],
        np.diag([2.0e-10, 3.0e-10, 5.0e-10]),
        rtol=1.0e-10,
    )


def test_finite_lifetime_covariance_rejects_non_psd_relative_mobility() -> None:
    with pytest.raises(ValueError, match="state_relative_displacement_mobility"):
        apply_finite_lifetime_relative_covariance(
            self_current_tensors_D_self_i_m2_s=np.zeros((1, 3, 3)),
            symmetric_capacity_fluxes_K_ij_mol_m3_s=np.zeros((1, 1)),
            reversible_generator_Q_ij_s_inv=np.zeros((1, 1)),
            transition_second_moments_M_ij_m2=np.zeros((1, 1, 3, 3)),
            state_concentrations_mol_m3=np.ones(1),
            state_relative_displacement_fluctuations_m=(np.eye(3),),
            state_relative_displacement_mobilities_m2_s=(np.diag([1.0, 1.0, -1.0]),),
            state_relative_center_charge_numbers=(np.asarray([1.0, -1.0]),),
            transition_displacement_edge_mask=np.zeros((1, 1), dtype=bool),
            temperature_K=T_REF_K,
        )


def test_finite_lifetime_covariance_preserves_psd_for_rotated_state_current() -> None:
    rotation = np.asarray(
        [[1.0, 1.0, 0.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]],
        dtype=float,
    )
    retained_state_current = (
        rotation @ np.diag([1.0e-10, 2.0e-10, 4.0e-10]) @ rotation.T
    )
    relative_mobility = np.diag([2.0e-10, 3.0e-10, 5.0e-10])
    adjusted, adjusted_moments, _diagnostics = (
        apply_finite_lifetime_relative_covariance(
            self_current_tensors_D_self_i_m2_s=np.asarray(
                [retained_state_current + relative_mobility, np.zeros((3, 3))]
            ),
            symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
                [[0.0, 2.0e9], [2.0e9, 0.0]]
            ),
            reversible_generator_Q_ij_s_inv=np.asarray(
                [[-2.0e9, 2.0e9], [2.0e9, -2.0e9]]
            ),
            transition_second_moments_M_ij_m2=np.zeros((2, 2, 3, 3)),
            state_concentrations_mol_m3=np.ones(2),
            state_relative_displacement_fluctuations_m=(
                np.diag(np.sqrt([1.0e-20, 2.0e-20, 3.0e-20])),
                np.empty((0, 3)),
            ),
            state_relative_displacement_mobilities_m2_s=(
                relative_mobility,
                np.empty((0, 0)),
            ),
            state_relative_center_charge_numbers=(
                np.asarray([1.0, -1.0]),
                np.empty(0),
            ),
            transition_displacement_edge_mask=np.asarray(
                [[False, True], [True, False]],
                dtype=bool,
            ),
            temperature_K=T_REF_K,
        )
    )

    np.testing.assert_allclose(adjusted[0], retained_state_current)
    assert np.min(np.linalg.eigvalsh(adjusted_moments[0, 1])) >= 0.0


def test_finite_lifetime_covariance_rejects_non_neutral_bound_centers() -> None:
    with pytest.raises(ValueError, match="neutral bound pair"):
        apply_finite_lifetime_relative_covariance(
            self_current_tensors_D_self_i_m2_s=np.asarray([np.eye(3) * 1.0e-10]),
            symmetric_capacity_fluxes_K_ij_mol_m3_s=np.zeros((1, 1)),
            reversible_generator_Q_ij_s_inv=np.asarray([[-1.0e9]]),
            transition_second_moments_M_ij_m2=np.zeros((1, 1, 3, 3)),
            state_concentrations_mol_m3=np.ones(1),
            state_relative_displacement_fluctuations_m=(np.eye(3) * 1.0e-10,),
            state_relative_displacement_mobilities_m2_s=(np.eye(3) * 1.0e-10,),
            state_relative_center_charge_numbers=(np.asarray([1.0, 1.0]),),
            transition_displacement_edge_mask=np.zeros((1, 1), dtype=bool),
            temperature_K=T_REF_K,
        )


def test_finite_lifetime_covariance_rejects_unowned_finite_mobility() -> None:
    with pytest.raises(ValueError, match="positive-K outgoing edge"):
        apply_finite_lifetime_relative_covariance(
            self_current_tensors_D_self_i_m2_s=np.asarray([np.eye(3) * 1.0e-10]),
            symmetric_capacity_fluxes_K_ij_mol_m3_s=np.zeros((1, 1)),
            reversible_generator_Q_ij_s_inv=np.asarray([[-1.0e9]]),
            transition_second_moments_M_ij_m2=np.zeros((1, 1, 3, 3)),
            state_concentrations_mol_m3=np.ones(1),
            state_relative_displacement_fluctuations_m=(np.eye(3) * 1.0e-10,),
            state_relative_displacement_mobilities_m2_s=(np.eye(3) * 1.0e-10,),
            state_relative_center_charge_numbers=(np.asarray([1.0, -1.0]),),
            transition_displacement_edge_mask=np.zeros((1, 1), dtype=bool),
            temperature_K=T_REF_K,
        )
