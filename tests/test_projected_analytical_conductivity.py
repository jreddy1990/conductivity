from __future__ import annotations

import inspect
import subprocess
import sys

import numpy as np
import pytest

from conductivity import projected_analytical_conductivity as model
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
            "conductivity.validate_projected_analytical_property_db",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "source_labeled_rows=102" in completed.stdout
    assert "evaluated_rows=0" in completed.stdout
    assert "failed_rows=102" in completed.stdout
    assert "populated full ConductivityPhysicalLibrary" in completed.stdout
    assert "mae_mS_cm=" not in completed.stdout
