from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import conductivity.old.projected_mori_property_db_audit as projected_mori_audit
from conductivity.physical_library import projected_analytical_conductivity as model
from conductivity.old.projected_mori_property_db_audit import (
    audit_projected_mori_conductivity_against_property_db,
)
from constants import F, R, T_REF_K


def _zero_potential_J_mol(point: np.ndarray) -> float:
    return 0.0


def _unit_mobility_tensor_m2_s(point: np.ndarray) -> np.ndarray:
    return np.eye(point.size, dtype=float)


def _single_axis_charge_gradient(point: np.ndarray) -> np.ndarray:
    return np.asarray([[1.0], [0.0], [0.0]], dtype=float)


def _zero_charge_gradient(point: np.ndarray) -> np.ndarray:
    return np.zeros((3, point.size), dtype=float)


def _empty_memory_gradient(point: np.ndarray) -> np.ndarray:
    return np.zeros((0, point.size), dtype=float)


def _two_state_generator_inputs() -> dict:
    return {
        "basin_quadrature_points": (
            np.asarray([[-1.0]], dtype=float),
            np.asarray([[1.0]], dtype=float),
        ),
        "basin_quadrature_weights": (
            np.asarray([1.0], dtype=float),
            np.asarray([1.0], dtype=float),
        ),
        "transition_pair_indices": np.asarray([[0, 1]], dtype=int),
        "transition_quadrature_points": (np.asarray([[0.0]], dtype=float),),
        "transition_quadrature_weights": (np.asarray([1.0], dtype=float),),
        "transition_committor_gradients": (np.asarray([[0.5]], dtype=float),),
        "transition_surface_state_indices": (np.asarray([0], dtype=int),),
        "transition_path_weights": (np.asarray([1.0], dtype=float),),
        "total_component_concentrations_mol_m3": np.asarray([2.0], dtype=float),
        "basin_stoichiometry": np.asarray([[1.0], [1.0]], dtype=float),
        "self_current_coordinate_projectors": (
            np.eye(1, dtype=float),
            np.eye(1, dtype=float),
        ),
    }


def _one_state_generator_inputs() -> dict:
    return {
        "basin_quadrature_points": (np.asarray([[0.0]], dtype=float),),
        "basin_quadrature_weights": (np.asarray([1.0], dtype=float),),
        "transition_pair_indices": np.zeros((0, 2), dtype=int),
        "transition_quadrature_points": (),
        "transition_quadrature_weights": (),
        "transition_committor_gradients": (),
        "transition_surface_state_indices": (),
        "transition_path_displacements_m": (),
        "transition_path_weights": (),
        "total_component_concentrations_mol_m3": np.asarray([1.0], dtype=float),
        "basin_stoichiometry": np.asarray([[1.0]], dtype=float),
        "self_current_coordinate_projectors": (np.eye(1, dtype=float),),
    }


def _current_spanning_memory_gradient(point: np.ndarray) -> np.ndarray:
    return np.asarray([[1.0]], dtype=float)


def _duplicate_memory_gradient(point: np.ndarray) -> np.ndarray:
    return np.asarray([[1.0], [2.0]], dtype=float)


def _orthogonal_memory_gradient(point: np.ndarray) -> np.ndarray:
    return np.asarray([[0.0, 1.0]], dtype=float)


def _two_coordinate_charge_gradient(point: np.ndarray) -> np.ndarray:
    return np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)


def _offdiagonal_mobility_tensor_m2_s(point: np.ndarray) -> np.ndarray:
    return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)


def _null_energy_memory_gradient(point: np.ndarray) -> np.ndarray:
    return np.asarray([[1.0, 0.0]], dtype=float)


def _offdiagonal_charge_gradient(point: np.ndarray) -> np.ndarray:
    return np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)


def _odd_charge_density_memory_gradient(point: np.ndarray) -> np.ndarray:
    return np.asarray([[float(point[0])]], dtype=float)


def test_projected_module_contains_no_descriptor_or_recipe_closure() -> None:
    source = inspect.getsource(model)
    forbidden_terms = (
        "finite_markov_conductivity",
        "electrolyte_property_db",
        "compute_conductivity_from_composition",
        "build_projected_primitives_from_composition",
        "target_conductivity",
        "property_calibration_factor",
        "green_kubo",
        "einstein_helfand",
    )
    for forbidden_term in forbidden_terms:
        assert forbidden_term not in source


def test_chemical_potential_mass_balance_shares_conserved_components() -> None:
    component_totals_mol_m3 = np.asarray([1000.0, 700.0, 300.0], dtype=float)
    basin_stoichiometry = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    restricted_partitions = np.asarray([1.0, 0.9, 0.8, 0.25, 0.35], dtype=float)

    populations = model.compute_equilibrium_populations_from_stoichiometry(
        component_totals_mol_m3,
        basin_stoichiometry,
        restricted_partitions,
    )

    assert np.allclose(basin_stoichiometry.T @ populations, component_totals_mol_m3)
    assert populations[0] + populations[3] + populations[4] == pytest.approx(1000.0)


def test_density_weights_reproduce_basin_concentrations() -> None:
    basin_points = (
        np.asarray([[0.0]], dtype=float),
        np.asarray([[1.0]], dtype=float),
    )
    basin_weights = (
        np.asarray([1.0], dtype=float),
        np.asarray([2.0], dtype=float),
    )

    def potential_energy_J_mol(point: np.ndarray) -> float:
        return float(point[0] * R * 300.0)

    restricted_partitions = model.compute_restricted_partition_values(
        potential_energy_J_mol,
        basin_points,
        basin_weights,
        300.0,
    )
    density_result = model.compute_basin_density_weights(
        potential_energy_J_mol,
        basin_points,
        basin_weights,
        restricted_partitions,
        np.asarray([9.0], dtype=float),
        np.asarray([[1.0], [1.0]], dtype=float),
        300.0,
    )

    density_weights = density_result["basin_density_weights_mol_m3"]
    concentrations = density_result["basin_concentrations_mol_m3"]
    assert np.allclose(
        concentrations,
        np.asarray([float(np.sum(weights)) for weights in density_weights]),
    )
    assert np.isclose(float(np.sum(concentrations)), 9.0)


def test_free_brownian_charge_density_memory_does_not_cancel_ne() -> None:
    inputs = _one_state_generator_inputs()
    result = model.compute_projected_analytical_conductivity(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        _single_axis_charge_gradient,
        _current_spanning_memory_gradient,
        inputs["basin_quadrature_points"],
        inputs["basin_quadrature_weights"],
        inputs["transition_pair_indices"],
        inputs["transition_quadrature_points"],
        inputs["transition_quadrature_weights"],
        inputs["transition_committor_gradients"],
        inputs["transition_surface_state_indices"],
        inputs["transition_path_displacements_m"],
        inputs["transition_path_weights"],
        inputs["total_component_concentrations_mol_m3"],
        inputs["basin_stoichiometry"],
        300.0,
        1.0,
        inputs["self_current_coordinate_projectors"],
    )

    expected_projected_diffusivity_density = np.diag([1.0, 0.0, 0.0])
    expected_sigma_S_m = F * F / (R * 300.0) * (1.0 / 3.0)
    assert np.allclose(
        result.projected_diffusivity_tensor,
        expected_projected_diffusivity_density,
    )
    assert np.allclose(result.continuous_mori_correction_tensor, np.zeros((3, 3)))
    assert result.effect_attribution[
        "mori_filter_accepted_candidate_indices"
    ].size == 0
    assert np.array_equal(
        result.effect_attribution["mori_filter_rejected_candidate_indices"],
        np.asarray([0], dtype=int),
    )
    assert result.sigma_S_m == pytest.approx(expected_sigma_S_m)


def test_duplicate_memory_coordinate_discarded() -> None:
    filtered = model.filter_memory_basis_by_dirichlet_residual(
        _duplicate_memory_gradient,
        _unit_mobility_tensor_m2_s,
        _zero_charge_gradient,
        (np.asarray([[0.0]], dtype=float),),
        (np.asarray([1.0], dtype=float),),
        np.eye(3, dtype=float),
        model.MEMORY_NULLSPACE_RELATIVE_TOL,
        model.MEMORY_NULLSPACE_RELATIVE_TOL,
        model.PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL,
    )

    assert np.array_equal(filtered.accepted_candidate_indices, np.asarray([0]))
    assert np.array_equal(filtered.discarded_candidate_indices, np.asarray([1]))
    assert filtered.rejected_candidate_indices.size == 0


def test_current_spanning_memory_rejected_before_readout() -> None:
    filtered = model.filter_memory_basis_by_dirichlet_residual(
        _current_spanning_memory_gradient,
        _unit_mobility_tensor_m2_s,
        _single_axis_charge_gradient,
        (np.asarray([[0.0]], dtype=float),),
        (np.asarray([1.0], dtype=float),),
        np.diag([1.0, 0.0, 0.0]),
        model.MEMORY_NULLSPACE_RELATIVE_TOL,
        model.MEMORY_NULLSPACE_RELATIVE_TOL,
        model.PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL,
    )

    assert filtered.accepted_candidate_indices.size == 0
    assert np.array_equal(filtered.rejected_candidate_indices, np.asarray([0]))
    assert filtered.mori_memory_matrix_A.shape == (0, 0)
    assert filtered.mori_current_coupling_matrix_h.shape == (0, 3)


def test_incremental_memory_filter_preserves_psd_remaining_tensor() -> None:
    filtered = model.filter_memory_basis_by_dirichlet_residual(
        _orthogonal_memory_gradient,
        _unit_mobility_tensor_m2_s,
        _two_coordinate_charge_gradient,
        (np.asarray([[0.0, 0.0]], dtype=float),),
        (np.asarray([1.0], dtype=float),),
        np.diag([1.0, 1.0, 0.0]),
        model.MEMORY_NULLSPACE_RELATIVE_TOL,
        model.MEMORY_NULLSPACE_RELATIVE_TOL,
        model.PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL,
    )

    assert np.array_equal(filtered.accepted_candidate_indices, np.asarray([0]))
    correction = model.compute_continuous_mori_correction(
        filtered.mori_memory_matrix_A,
        filtered.mori_current_coupling_matrix_h,
    )
    remaining = np.diag([1.0, 1.0, 0.0]) - correction
    assert np.min(np.linalg.eigvalsh(remaining)) >= -1.0e-12


def test_null_current_memory_fails_in_dirichlet_filter() -> None:
    with pytest.raises(ValueError, match="zero residual Dirichlet energy"):
        model.filter_memory_basis_by_dirichlet_residual(
            _null_energy_memory_gradient,
            _offdiagonal_mobility_tensor_m2_s,
            _offdiagonal_charge_gradient,
            (np.asarray([[0.0, 0.0]], dtype=float),),
            (np.asarray([1.0], dtype=float),),
            np.diag([1.0, 0.0, 0.0]),
            model.MEMORY_NULLSPACE_RELATIVE_TOL,
            model.MEMORY_NULLSPACE_RELATIVE_TOL,
            model.PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL,
        )


def test_default_mori_memory_inventory_excludes_current_spanning_density_modes() -> None:
    memory_text = Path("conductivity/physical_library/memory.yaml").read_text()

    assert "charge_density_relaxation" not in memory_text
    assert "atmosphere_polarization" not in memory_text


def test_charge_density_mode_requires_phase_symmetric_quadrature() -> None:
    with pytest.raises(ValueError, match="aliases unbounded charge polarization"):
        model.validate_charge_density_translation_symmetry(
            _odd_charge_density_memory_gradient,
            _unit_mobility_tensor_m2_s,
            _single_axis_charge_gradient,
            (np.asarray([[1.0]], dtype=float),),
            (np.asarray([1.0], dtype=float),),
            model.MEMORY_NULLSPACE_RELATIVE_TOL,
        )

    model.validate_charge_density_translation_symmetry(
        _odd_charge_density_memory_gradient,
        _unit_mobility_tensor_m2_s,
        _single_axis_charge_gradient,
        (np.asarray([[-1.0], [1.0]], dtype=float),),
        (np.asarray([1.0, 1.0], dtype=float),),
        model.MEMORY_NULLSPACE_RELATIVE_TOL,
    )


def test_generator_path_uses_mass_balance_density_weights_for_conductivity() -> None:
    inputs = _two_state_generator_inputs()
    result = model.compute_projected_analytical_conductivity(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        _single_axis_charge_gradient,
        _empty_memory_gradient,
        inputs["basin_quadrature_points"],
        inputs["basin_quadrature_weights"],
        inputs["transition_pair_indices"],
        inputs["transition_quadrature_points"],
        inputs["transition_quadrature_weights"],
        inputs["transition_committor_gradients"],
        inputs["transition_surface_state_indices"],
        (np.asarray([[0.0, 0.0, 0.0]], dtype=float),),
        inputs["transition_path_weights"],
        inputs["total_component_concentrations_mol_m3"],
        inputs["basin_stoichiometry"],
        300.0,
        1.0,
        inputs["self_current_coordinate_projectors"],
    )

    expected_projected_diffusivity_density = np.diag([2.0, 0.0, 0.0])
    expected_sigma_S_m = F * F / (R * 300.0) * (2.0 / 3.0)
    assert np.allclose(
        result.projected_diffusivity_tensor,
        expected_projected_diffusivity_density,
    )
    assert result.sigma_S_m == pytest.approx(expected_sigma_S_m)


def test_full_generator_rejects_nonidentity_self_current_projector() -> None:
    inputs = _one_state_generator_inputs()
    with pytest.raises(ValueError, match="requires identity"):
        model.compute_projected_analytical_conductivity(
            _zero_potential_J_mol,
            _unit_mobility_tensor_m2_s,
            _single_axis_charge_gradient,
            _empty_memory_gradient,
            inputs["basin_quadrature_points"],
            inputs["basin_quadrature_weights"],
            inputs["transition_pair_indices"],
            inputs["transition_quadrature_points"],
            inputs["transition_quadrature_weights"],
            inputs["transition_committor_gradients"],
            inputs["transition_surface_state_indices"],
            inputs["transition_path_displacements_m"],
            inputs["transition_path_weights"],
            inputs["total_component_concentrations_mol_m3"],
            inputs["basin_stoichiometry"],
            300.0,
            1.0,
            (np.zeros((1, 1), dtype=float),),
        )


def test_zero_charge_gradient_produces_zero_conductivity() -> None:
    inputs = _two_state_generator_inputs()
    result = model.compute_projected_analytical_conductivity(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        _zero_charge_gradient,
        _empty_memory_gradient,
        inputs["basin_quadrature_points"],
        inputs["basin_quadrature_weights"],
        inputs["transition_pair_indices"],
        inputs["transition_quadrature_points"],
        inputs["transition_quadrature_weights"],
        inputs["transition_committor_gradients"],
        inputs["transition_surface_state_indices"],
        (np.asarray([[0.0, 0.0, 0.0]], dtype=float),),
        inputs["transition_path_weights"],
        inputs["total_component_concentrations_mol_m3"],
        inputs["basin_stoichiometry"],
        300.0,
        1.0,
        inputs["self_current_coordinate_projectors"],
    )

    assert result.sigma_mS_cm == 0.0


def test_poisson_memory_corrector_subtracts_correlated_jump_drift() -> None:
    concentrations = np.asarray([1.0, 1.0], dtype=float)
    capacity_fluxes = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    first_moments = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0e-9, 0.0, 0.0]],
            [[-1.0e-9, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    second_moments = np.asarray(
        [
            [
                np.zeros((3, 3), dtype=float),
                np.diag([1.0e-18, 0.0, 0.0]),
            ],
            [
                np.diag([1.0e-18, 0.0, 0.0]),
                np.zeros((3, 3), dtype=float),
            ],
        ],
        dtype=float,
    )
    self_current = np.zeros((2, 3, 3), dtype=float)

    result = model.compute_projected_analytical_conductivity_from_primitives(
        concentrations,
        capacity_fluxes,
        first_moments,
        second_moments,
        self_current,
        np.zeros((0, 0), dtype=float),
        np.zeros((0, 3), dtype=float),
        300.0,
    )

    assert result.direct_diffusivity_tensor[0, 0] > 0.0
    assert result.finite_state_memory_correction_tensor[0, 0] > 0.0
    assert result.projected_diffusivity_tensor[0, 0] == pytest.approx(0.0)


def test_three_state_transition_chain_has_nonzero_finite_state_drift() -> None:
    concentrations = np.ones(3, dtype=float)
    capacity_fluxes = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    transition_length_m = 1.0e-9
    first_moments = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [transition_length_m, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [-transition_length_m, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [transition_length_m, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [-transition_length_m, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ],
        dtype=float,
    )
    transition_second_moment = np.diag(
        [transition_length_m * transition_length_m, 0.0, 0.0]
    )
    zero_second_moment = np.zeros((3, 3), dtype=float)
    second_moments = np.asarray(
        [
            [
                zero_second_moment,
                transition_second_moment,
                zero_second_moment,
            ],
            [
                transition_second_moment,
                zero_second_moment,
                transition_second_moment,
            ],
            [
                zero_second_moment,
                transition_second_moment,
                zero_second_moment,
            ],
        ],
        dtype=float,
    )
    self_current = np.zeros((3, 3, 3), dtype=float)

    result = model.compute_projected_analytical_conductivity_from_primitives(
        concentrations,
        capacity_fluxes,
        first_moments,
        second_moments,
        self_current,
        np.zeros((0, 0), dtype=float),
        np.zeros((0, 3), dtype=float),
        300.0,
    )

    state_drift = np.einsum(
        "ij,ija->ia",
        result.reversible_generator_Q_ij_s_inv,
        result.transition_first_moments_d_ij_m,
    )
    assert np.any(state_drift != 0.0)
    assert result.finite_state_memory_correction_tensor[0, 0] > 0.0


def test_weighted_poisson_accepts_roundoff_scale_solvable_drift() -> None:
    generator = np.asarray(
        [
            [-1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=float,
    )
    concentrations = np.asarray([1.0e6, 1.0e6], dtype=float)
    drift = np.asarray([1.0e-4, -1.0e-4 + 1.0e-20], dtype=float)

    solution = model.solve_weighted_poisson(generator, concentrations, drift)

    assert np.all(np.isfinite(solution))
    assert concentrations @ solution == pytest.approx(0.0)


def test_mori_null_current_mode_fails_loudly() -> None:
    with pytest.raises(ValueError, match="null memory mode"):
        model.compute_continuous_mori_correction(
            np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=float),
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        )


def test_transition_displacement_scale_is_enforced_on_path_samples() -> None:
    with pytest.raises(ValueError, match="transition displacement exceeds"):
        model.compute_transition_path_displacement_moments(
            np.asarray([[0, 1]], dtype=int),
            (np.asarray([[2.0e-8, 0.0, 0.0]], dtype=float),),
            (np.asarray([1.0], dtype=float),),
            2,
            model.DEFAULT_MAX_TRANSITION_DISPLACEMENT_M,
        )


def test_multicenter_charge_covariance_uses_z_d_z() -> None:
    center_mobility = np.asarray([[1.0, -0.25], [-0.25, 0.5]], dtype=float)
    charge_numbers = np.asarray([1.0, -1.0], dtype=float)
    assert model.compute_state_charge_mobility_tensor(
        charge_numbers,
        center_mobility,
    ) == pytest.approx(2.0)


def test_memory_coordinate_state_means_use_density_weights() -> None:
    basin_points = (
        np.asarray([[0.0], [2.0]], dtype=float),
        np.asarray([[10.0]], dtype=float),
    )
    density_weights = (
        np.asarray([1.0, 3.0], dtype=float),
        np.asarray([2.0], dtype=float),
    )
    concentrations = np.asarray([4.0, 2.0], dtype=float)

    def memory_coordinates(point: np.ndarray) -> np.ndarray:
        return np.asarray([point[0], point[0] * point[0]], dtype=float)

    state_means = model.compute_state_memory_coordinate_means(
        memory_coordinates,
        basin_points,
        density_weights,
        concentrations,
    )

    assert np.allclose(state_means[0], np.asarray([1.5, 3.0]))
    assert np.allclose(state_means[1], np.asarray([10.0, 100.0]))


def test_candidate_mori_scoring_uses_spec_pseudoinverse() -> None:
    score_result = model.score_candidate_mori_coordinates(
        np.asarray([[2.0]], dtype=float),
        np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        np.asarray([3.0], dtype=float),
        np.asarray([[1.0]], dtype=float),
        np.asarray([[2.0, 0.0, 0.0]], dtype=float),
    )

    assert np.allclose(score_result["residual_coupling"], np.asarray([[1.5, 0.0, 0.0]]))
    assert score_result["residual_energy"][0] == pytest.approx(2.5)
    assert score_result["scores"][0] == pytest.approx(0.9)


def test_primitive_ownership_scores_assign_largest_response() -> None:
    base = model.compute_projected_analytical_conductivity_from_primitives(
        np.asarray([1.0], dtype=float),
        np.zeros((1, 1), dtype=float),
        np.zeros((1, 1, 3), dtype=float),
        np.zeros((1, 1, 3, 3), dtype=float),
        np.asarray([np.diag([1.0e-10, 0.0, 0.0])], dtype=float),
        np.zeros((0, 0), dtype=float),
        np.zeros((0, 3), dtype=float),
        300.0,
    )
    perturbed = model.compute_projected_analytical_conductivity_from_primitives(
        np.asarray([1.0], dtype=float),
        np.zeros((1, 1), dtype=float),
        np.zeros((1, 1, 3), dtype=float),
        np.zeros((1, 1, 3, 3), dtype=float),
        np.asarray([np.diag([3.0e-10, 0.0, 0.0])], dtype=float),
        np.zeros((0, 0), dtype=float),
        np.zeros((0, 3), dtype=float),
        300.0,
    )

    scores = model.compute_primitive_ownership_scores(perturbed, base, 1.0)

    assert scores["S_D"] > 0.0
    assert scores["largest_primitive_index"] == pytest.approx(3.0)


def test_basis_refinement_adds_missing_current_coordinate() -> None:
    refinement_result = model.refine_mori_basis_by_projected_residual(
        direct_diffusivity_tensor=np.diag([1.0, 0.0, 0.0]),
        initial_mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        initial_mori_current_coupling_matrix_h=np.zeros((0, 3), dtype=float),
        candidate_self_energies_A_gg=np.asarray([2.0, 1.0], dtype=float),
        candidate_cross_energies_A_gPhi=np.zeros((2, 0), dtype=float),
        candidate_cross_energy_matrix=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        candidate_current_couplings_h_g=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            dtype=float,
        ),
        temperature_K=300.0,
        residual_score_tolerance=0.001,
        conductivity_change_tolerance_S_m=1.0e-30,
        max_added_coordinates=1,
    )

    assert np.array_equal(
        refinement_result["selected_candidate_indices"],
        np.asarray([0], dtype=int),
    )
    assert refinement_result["convergence_status"] == "basis_residual_above_tolerance"
    assert refinement_result["hard_convergence_failure"]
    assert refinement_result["final_maximum_residual_score"] == pytest.approx(0.01)
    assert refinement_result["final_conductivity_change_abs_S_m"] > 0.0
    assert np.allclose(
        refinement_result["final_mori_memory_matrix_A"],
        np.asarray([[2.0]], dtype=float),
    )
    assert np.allclose(
        refinement_result["final_mori_current_coupling_matrix_h"],
        np.asarray([[1.0, 0.0, 0.0]], dtype=float),
    )
    conductivity_history = refinement_result["conductivity_history_S_m"]
    assert conductivity_history.size == 2
    assert conductivity_history[-1] < conductivity_history[0]


def test_basis_refinement_converges_with_residual_and_conductivity_diagnostics() -> None:
    refinement_result = model.refine_mori_basis_by_projected_residual(
        direct_diffusivity_tensor=np.diag([1.0, 1.0, 0.0]),
        initial_mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        initial_mori_current_coupling_matrix_h=np.zeros((0, 3), dtype=float),
        candidate_self_energies_A_gg=np.asarray([2.0], dtype=float),
        candidate_cross_energies_A_gPhi=np.zeros((1, 0), dtype=float),
        candidate_cross_energy_matrix=np.asarray([[2.0]], dtype=float),
        candidate_current_couplings_h_g=np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        temperature_K=300.0,
        residual_score_tolerance=0.1,
        conductivity_change_tolerance_S_m=1.0e9,
        max_added_coordinates=1,
    )

    assert refinement_result["convergence_status"] == "converged"
    assert not refinement_result["hard_convergence_failure"]
    assert np.array_equal(
        refinement_result["selected_candidate_indices"],
        np.asarray([0], dtype=int),
    )
    assert refinement_result["maximum_residual_score_history"][0] == pytest.approx(0.5)
    assert refinement_result["maximum_residual_score_history"][-1] == pytest.approx(0.0)
    assert refinement_result["conductivity_change_history_abs_S_m"].size == 1


def test_basis_refinement_rejects_duplicate_null_and_current_spanning_candidates() -> None:
    refinement_result = model.refine_mori_basis_by_projected_residual(
        direct_diffusivity_tensor=np.diag([1.0, 1.0, 0.0]),
        initial_mori_memory_matrix_A=np.asarray([[1.0]], dtype=float),
        initial_mori_current_coupling_matrix_h=np.asarray(
            [[0.0, 1.0, 0.0]],
            dtype=float,
        ),
        candidate_self_energies_A_gg=np.asarray([1.0, 0.0, 1.0], dtype=float),
        candidate_cross_energies_A_gPhi=np.asarray(
            [
                [1.0],
                [0.0],
                [0.0],
            ],
            dtype=float,
        ),
        candidate_cross_energy_matrix=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        candidate_current_couplings_h_g=np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        temperature_K=300.0,
        residual_score_tolerance=1.0e-12,
        conductivity_change_tolerance_S_m=1.0e-30,
        max_added_coordinates=3,
    )

    assert np.array_equal(
        refinement_result["selected_candidate_indices"],
        np.asarray([], dtype=int),
    )
    assert np.array_equal(
        refinement_result["discarded_candidate_indices"],
        np.asarray([0], dtype=int),
    )
    assert np.array_equal(
        refinement_result["rejected_null_energy_candidate_indices"],
        np.asarray([1], dtype=int),
    )
    assert np.array_equal(
        refinement_result["rejected_current_spanning_candidate_indices"],
        np.asarray([2], dtype=int),
    )


def test_basis_refinement_marks_hard_status_when_add_budget_exhausts() -> None:
    refinement_result = model.refine_mori_basis_by_projected_residual(
        direct_diffusivity_tensor=np.diag([1.0, 1.0, 0.0]),
        initial_mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        initial_mori_current_coupling_matrix_h=np.zeros((0, 3), dtype=float),
        candidate_self_energies_A_gg=np.asarray([4.0, 4.0], dtype=float),
        candidate_cross_energies_A_gPhi=np.zeros((2, 0), dtype=float),
        candidate_cross_energy_matrix=np.asarray(
            [
                [4.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=float,
        ),
        candidate_current_couplings_h_g=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        temperature_K=300.0,
        residual_score_tolerance=0.2,
        conductivity_change_tolerance_S_m=1.0e9,
        max_added_coordinates=1,
    )

    assert np.array_equal(
        refinement_result["selected_candidate_indices"],
        np.asarray([0], dtype=int),
    )
    assert refinement_result["convergence_status"] == "basis_residual_above_tolerance"
    assert refinement_result["hard_convergence_failure"]
    assert refinement_result["final_maximum_residual_score"] == pytest.approx(0.25)


def test_composition_only_conductivity_fails_without_full_physical_library() -> None:
    recipe = {
        "solvents": {"EC": 0.3, "DMC": 0.7},
        "salts": {"LiPF6": 1.0},
        "additives": {},
    }

    with pytest.raises(ValueError, match="populated full ConductivityPhysicalLibrary"):
        model.compute_projected_analytical_conductivity_from_composition(
            recipe,
            T_REF_K,
        )

    with pytest.raises(ValueError, match="populated full ConductivityPhysicalLibrary"):
        model.build_projected_primitives_from_electrolyte_composition(
            recipe,
            T_REF_K,
        )

    with pytest.raises(ValueError, match="populated full ConductivityPhysicalLibrary"):
        model.build_projected_generator_from_electrolyte_composition(
            recipe,
            T_REF_K,
        )

    assert not hasattr(model, "build_species_data_reduced_physical_library")


def test_property_db_validator_evaluates_all_labeled_compositions() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "conductivity.physical_library.library_io",
            "validate-property-db",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "source_labeled_rows=102" in completed.stdout
    assert "evaluated_rows=0" in completed.stdout


def test_projected_mori_property_db_audit_rejects_recipe_only_row() -> None:
    recipe_only_row = {
        "recipe": {
            "solvents": {"EC": 0.3, "DMC": 0.7},
            "salts": {"LiPF6": 1.0},
            "additives": {},
        },
        "properties": {"conductivity_mS_cm": 10.0},
    }

    with pytest.raises(
        ValueError,
        match="recipe-only conductivity validation requires",
    ):
        audit_projected_mori_conductivity_against_property_db(
            (recipe_only_row,),
            T_REF_K,
            "basis",
            "relaxation",
            "anion",
        )


def test_projected_mori_audit_requires_active_mixed_recipe_species_records() -> None:
    missing_species_names = projected_mori_audit._missing_active_recipe_species_names(
        {
            "solvents": {"EC": 0.3, "DMC": 0.7},
            "salts": {"LiPF6": 0.2, "LiFSI": 0.8},
            "additives": {"FEC": 0.05},
        },
        {
            "EC": {},
            "DMC": {},
            "Li+": {},
        },
    )

    assert missing_species_names == ("FEC", "FSI-", "PF6-")
