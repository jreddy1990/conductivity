from __future__ import annotations

import inspect

import numpy as np

from conductivity import projected_analytical_conductivity as model
from constants import F, R


def _zero_potential_J_mol(q: np.ndarray) -> float:
    return 0.0


def _unit_mobility_tensor_m2_s(q: np.ndarray) -> np.ndarray:
    return np.eye(1)


def _zero_memory_gradient(q: np.ndarray) -> np.ndarray:
    return np.zeros((1, 1))


def _unit_memory_gradient(q: np.ndarray) -> np.ndarray:
    return np.ones((1, 1))


def _empty_memory_gradient(q: np.ndarray) -> np.ndarray:
    return np.zeros((0, 1))


def _single_axis_charge_polarization_gradient(q: np.ndarray) -> np.ndarray:
    return np.asarray([[1.0], [0.0], [0.0]], dtype=float)


def _zero_charge_polarization_gradient(q: np.ndarray) -> np.ndarray:
    return np.zeros((3, 1))


def _one_dimensional_two_basin_inputs() -> dict:
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
        "transition_path_weights": (np.asarray([1.0], dtype=float),),
    }


def test_projected_analytical_estimator_has_no_gk_eh_acceptance_logic():
    source = inspect.getsource(model)
    forbidden_terms = (
        "green_kubo",
        "einstein_helfand",
        "acceptance_test",
        "current_autocorrelation",
    )
    for forbidden_term in forbidden_terms:
        assert forbidden_term not in source.lower()


def test_equilibrium_populations_are_restricted_boltzmann_integrals():
    basin_points = (
        np.asarray([[0.0]], dtype=float),
        np.asarray([[1.0]], dtype=float),
    )
    basin_weights = (
        np.asarray([1.0], dtype=float),
        np.asarray([2.0], dtype=float),
    )

    def potential_energy_J_mol(q: np.ndarray) -> float:
        return float(q[0] * R * 300.0)

    partitions = model.compute_restricted_partition_values(
        potential_energy_J_mol,
        basin_points,
        basin_weights,
        300.0,
    )
    populations = model.compute_equilibrium_populations(partitions, 9.0)

    expected_partitions = np.asarray([1.0, 2.0 * np.exp(-1.0)], dtype=float)
    expected_populations = 9.0 * expected_partitions / np.sum(expected_partitions)
    assert np.allclose(partitions, expected_partitions)
    assert np.allclose(populations, expected_populations)


def test_neutral_generator_returns_zero_conductivity():
    inputs = _one_dimensional_two_basin_inputs()
    result = model.compute_projected_analytical_conductivity(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        _zero_charge_polarization_gradient,
        _zero_memory_gradient,
        inputs["basin_quadrature_points"],
        inputs["basin_quadrature_weights"],
        inputs["transition_pair_indices"],
        inputs["transition_quadrature_points"],
        inputs["transition_quadrature_weights"],
        inputs["transition_committor_gradients"],
        (np.asarray([[0.0, 0.0, 0.0]], dtype=float),),
        inputs["transition_path_weights"],
        2.0,
        300.0,
        1.0,
    )

    assert result["sigma_S_m"] == 0.0
    assert np.allclose(result["projected_diffusivity_tensor"], np.zeros((3, 3)))


def test_direct_self_current_matches_single_axis_generator_limit_without_memory():
    inputs = _one_dimensional_two_basin_inputs()
    result = model.compute_projected_analytical_conductivity(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        _single_axis_charge_polarization_gradient,
        _zero_memory_gradient,
        inputs["basin_quadrature_points"],
        inputs["basin_quadrature_weights"],
        inputs["transition_pair_indices"],
        inputs["transition_quadrature_points"],
        inputs["transition_quadrature_weights"],
        inputs["transition_committor_gradients"],
        (np.asarray([[0.0, 0.0, 0.0]], dtype=float),),
        inputs["transition_path_weights"],
        2.0,
        300.0,
        1.0,
    )

    expected_sigma_S_m = F * F / (R * 300.0) * (2.0 / 3.0)
    assert np.isclose(result["sigma_S_m"], expected_sigma_S_m)


def test_empty_mori_basis_is_valid_and_has_zero_continuous_correction():
    inputs = _one_dimensional_two_basin_inputs()
    result = model.compute_projected_analytical_conductivity(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        _single_axis_charge_polarization_gradient,
        _empty_memory_gradient,
        inputs["basin_quadrature_points"],
        inputs["basin_quadrature_weights"],
        inputs["transition_pair_indices"],
        inputs["transition_quadrature_points"],
        inputs["transition_quadrature_weights"],
        inputs["transition_committor_gradients"],
        (np.asarray([[0.0, 0.0, 0.0]], dtype=float),),
        inputs["transition_path_weights"],
        2.0,
        300.0,
        1.0,
    )

    assert result["mori_memory_matrix_A"].shape == (0, 0)
    assert result["mori_current_coupling_matrix_h"].shape == (0, 3)
    assert np.allclose(result["continuous_mori_correction_tensor"], np.zeros((3, 3)))


def test_function_path_mori_matrices_are_density_scaled():
    basin_points = (np.asarray([[0.0]], dtype=float),)
    basin_weights = (np.asarray([1.0], dtype=float),)

    mori_matrix, mori_coupling = model.compute_mori_memory_matrices(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        _single_axis_charge_polarization_gradient,
        _unit_memory_gradient,
        basin_points,
        basin_weights,
        5.0,
        300.0,
        1.0,
    )
    mori_correction = model.compute_continuous_mori_correction(
        mori_matrix,
        mori_coupling,
    )

    assert np.allclose(mori_matrix, np.asarray([[5.0]], dtype=float))
    assert np.allclose(mori_coupling, np.asarray([[5.0, 0.0, 0.0]], dtype=float))
    assert np.allclose(
        mori_correction,
        np.diag(np.asarray([5.0, 0.0, 0.0], dtype=float)),
    )


def test_primitive_readout_matches_generator_projection_readout():
    inputs = _one_dimensional_two_basin_inputs()
    generator_result = model.compute_projected_analytical_conductivity(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        _single_axis_charge_polarization_gradient,
        _empty_memory_gradient,
        inputs["basin_quadrature_points"],
        inputs["basin_quadrature_weights"],
        inputs["transition_pair_indices"],
        inputs["transition_quadrature_points"],
        inputs["transition_quadrature_weights"],
        inputs["transition_committor_gradients"],
        (np.asarray([[0.0, 0.0, 0.0]], dtype=float),),
        inputs["transition_path_weights"],
        2.0,
        300.0,
        1.0,
    )
    primitive_result = model.compute_projected_analytical_conductivity_from_primitives(
        generator_result["state_concentrations_mol_m3"],
        generator_result["symmetric_capacity_fluxes_K_ij_mol_m3_s"],
        generator_result["transition_first_moments_d_ij_m"],
        generator_result["transition_second_moments_M_ij_m2"],
        generator_result["self_current_tensors_D_self_i_m2_s"],
        generator_result["mori_memory_matrix_A"],
        generator_result["mori_current_coupling_matrix_h"],
        300.0,
    )

    assert np.isclose(primitive_result["sigma_S_m"], generator_result["sigma_S_m"])
    assert np.allclose(
        primitive_result["projected_diffusivity_tensor"],
        generator_result["projected_diffusivity_tensor"],
    )


def test_callable_input_wrapper_matches_gradient_input_pipeline():
    inputs = _one_dimensional_two_basin_inputs()

    def charge_polarization(point: np.ndarray) -> np.ndarray:
        return np.asarray([point[0], 0.0, 0.0], dtype=float)

    def memory_coordinates(point: np.ndarray) -> np.ndarray:
        return np.asarray([0.0], dtype=float)

    result = model.compute_projected_analytical_conductivity_from_functions(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        charge_polarization,
        memory_coordinates,
        inputs["basin_quadrature_points"],
        inputs["basin_quadrature_weights"],
        inputs["transition_pair_indices"],
        inputs["transition_quadrature_points"],
        inputs["transition_quadrature_weights"],
        inputs["transition_committor_gradients"],
        (np.asarray([[0.0]], dtype=float),),
        (np.asarray([[0.0]], dtype=float),),
        inputs["transition_path_weights"],
        2.0,
        300.0,
        1.0,
    )

    expected_sigma_S_m = F * F / (R * 300.0) * (2.0 / 3.0)
    assert np.isclose(result["sigma_S_m"], expected_sigma_S_m)


def test_finite_state_backjump_correction_reduces_direct_diffusivity():
    state_concentrations = np.asarray([1.0, 1.0], dtype=float)
    capacity_fluxes = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    generator = model.compute_reversible_generator(capacity_fluxes, state_concentrations)
    first_moments = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )

    correction = model.compute_finite_state_memory_correction(
        state_concentrations,
        generator,
        first_moments,
    )

    assert correction[0, 0] > 0.0
    assert np.allclose(correction[1:, :], np.zeros((2, 3)))


def test_continuous_mori_memory_correction_matches_quadratic_form():
    memory_matrix = np.asarray([[2.0, 0.0], [0.0, 8.0]], dtype=float)
    current_coupling = np.asarray(
        [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        dtype=float,
    )

    correction = model.compute_continuous_mori_correction(
        memory_matrix,
        current_coupling,
    )

    assert np.allclose(
        correction,
        np.asarray([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_candidate_mori_coordinate_score_removes_existing_basis_projection():
    result = model.score_candidate_mori_coordinates(
        np.asarray([[2.0]], dtype=float),
        np.asarray([[2.0, 0.0, 0.0]], dtype=float),
        np.asarray([4.0], dtype=float),
        np.asarray([[1.0]], dtype=float),
        np.asarray([[3.0, 0.0, 0.0]], dtype=float),
    )

    assert np.allclose(result["residual_coupling"], np.asarray([[2.0, 0.0, 0.0]]))
    assert np.allclose(result["residual_energy"], np.asarray([3.5]))
    assert np.allclose(result["scores"], np.asarray([4.0 / 3.5]))


def test_transition_moments_can_be_computed_from_charge_polarization_callable():
    def charge_polarization(point: np.ndarray) -> np.ndarray:
        return np.asarray([point[0], 0.0, 0.0], dtype=float)

    first_moments, second_moments = (
        model.compute_transition_path_displacement_moments_from_polarization(
            np.asarray([[0, 1]], dtype=int),
            (np.asarray([[0.0], [1.0]], dtype=float),),
            (np.asarray([[2.0], [4.0]], dtype=float),),
            (np.asarray([1.0, 3.0], dtype=float),),
            charge_polarization,
            2,
        )
    )

    assert np.allclose(first_moments[0, 1], np.asarray([2.75, 0.0, 0.0]))
    assert np.allclose(first_moments[1, 0], np.asarray([-2.75, -0.0, -0.0]))
    assert np.isclose(second_moments[0, 1, 0, 0], 7.75)


def test_finite_difference_charge_polarization_gradient_uses_p_callable():
    def charge_polarization(point: np.ndarray) -> np.ndarray:
        return np.asarray([point[0] * point[0], point[1], 0.0], dtype=float)

    gradient = model.compute_charge_polarization_gradient_by_finite_difference(
        charge_polarization,
        np.asarray([2.0, 3.0], dtype=float),
        np.asarray([1.0e-5, 1.0e-5], dtype=float),
    )

    assert np.allclose(
        gradient,
        np.asarray([[4.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float),
        atol=1.0e-5,
    )


def test_finite_difference_memory_gradient_uses_memory_coordinate_callable():
    def memory_coordinates(point: np.ndarray) -> np.ndarray:
        return np.asarray([point[0] * point[1], point[1] * point[1]], dtype=float)

    gradient = model.compute_memory_coordinate_gradient_by_finite_difference(
        memory_coordinates,
        np.asarray([2.0, 3.0], dtype=float),
        np.asarray([1.0e-5, 1.0e-5], dtype=float),
    )

    assert np.allclose(
        gradient,
        np.asarray([[3.0, 2.0], [0.0, 6.0]], dtype=float),
        atol=1.0e-5,
    )


def test_one_dimensional_smoluchowski_capacity_flux_matches_constant_profile():
    capacity_fluxes = model.compute_one_dimensional_smoluchowski_capacity_fluxes(
        np.asarray([[0, 1]], dtype=int),
        (np.asarray([0.0, 1.0], dtype=float),),
        (np.asarray([0.0, 0.0], dtype=float),),
        (np.asarray([2.0, 2.0], dtype=float),),
        6.0,
        300.0,
        3.0,
        2,
    )

    assert np.allclose(capacity_fluxes, np.asarray([[0.0, 4.0], [4.0, 0.0]]))


def test_one_dimensional_committor_solves_constant_profile_boundary_values():
    committor_solution = model.solve_one_dimensional_committors(
        (np.asarray([0.0, 0.5, 1.0], dtype=float),),
        (np.asarray([0.0, 0.0, 0.0], dtype=float),),
        (np.asarray([2.0, 2.0, 2.0], dtype=float),),
        300.0,
    )

    assert np.allclose(
        committor_solution["committor_values"][0],
        np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    assert np.allclose(
        committor_solution["committor_gradients"][0],
        np.asarray([1.0, 1.0, 1.0], dtype=float),
    )
    assert np.allclose(
        committor_solution["smoluchowski_resistances"],
        np.asarray([0.5], dtype=float),
    )


def test_one_dimensional_reaction_coordinate_pipeline_computes_capacity_and_sigma():
    inputs = _one_dimensional_two_basin_inputs()

    def charge_polarization(point: np.ndarray) -> np.ndarray:
        return np.asarray([point[0], 0.0, 0.0], dtype=float)

    def memory_coordinates(point: np.ndarray) -> np.ndarray:
        return np.asarray([0.0], dtype=float)

    result = (
        model.compute_projected_analytical_conductivity_from_one_dimensional_reaction_coordinates(
            _zero_potential_J_mol,
            _unit_mobility_tensor_m2_s,
            charge_polarization,
            memory_coordinates,
            inputs["basin_quadrature_points"],
            inputs["basin_quadrature_weights"],
            inputs["transition_pair_indices"],
            (np.asarray([0.0, 1.0], dtype=float),),
            (np.asarray([0.0, 0.0], dtype=float),),
            (np.asarray([2.0, 2.0], dtype=float),),
            (np.asarray([[0.0]], dtype=float),),
            (np.asarray([[0.0]], dtype=float),),
            inputs["transition_path_weights"],
            2.0,
            300.0,
            1.0,
        )
    )

    expected_sigma_S_m = F * F / (R * 300.0) * (2.0 / 3.0)
    assert np.allclose(
        result["symmetric_capacity_fluxes_K_ij_mol_m3_s"],
        np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=float),
    )
    assert np.isclose(result["sigma_S_m"], expected_sigma_S_m)


def test_basis_refinement_adds_candidate_until_projected_delta_is_small():
    refinement = model.refine_mori_basis_by_projected_residual(
        np.eye(3) * 10.0,
        np.asarray([[2.0]], dtype=float),
        np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        np.asarray([4.0, 5.0], dtype=float),
        np.asarray([[0.0], [0.0]], dtype=float),
        np.asarray([[4.0, 0.0], [0.0, 5.0]], dtype=float),
        np.asarray([[2.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float),
        300.0,
        1.0e-12,
        2,
    )

    assert refinement["selected_candidate_indices"].shape[0] >= 1
    assert refinement["conductivity_history_S_m"].shape[0] >= 2
    assert refinement["final_sigma_S_m"] < refinement["conductivity_history_S_m"][0]


def test_effect_metadata_maps_claimed_effects_to_theorem_primitives():
    effect_map = model.conductivity_effect_primitive_locations()

    assert effect_map["free_ion_fraction"] == ("c_i",)
    assert "D_self_i" in effect_map["Li_anion_anticorrelation"]
    assert "M_ij" in effect_map["identity_diffusion"]
