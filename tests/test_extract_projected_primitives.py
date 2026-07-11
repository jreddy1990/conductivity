import numpy as np
import pytest

from constants import T_REF_K
from conductivity.extract_projected_primitives import main
from conductivity.physical_library.extract_projected_primitives import (
    AssociationThresholds,
    ChargedCenterCatalog,
    ChargedCenterFrame,
    _charge_displacements_by_step_m,
    _primitive_arrays_from_projected_set,
    _projected_recipe_record_from_composition_record,
)
from conductivity.physical_library.trajectory_primitives import (
    TrajectoryMarkovAdditiveSampleInput,
    _self_current_tensors,
    project_sampled_trajectory_to_generator_primitives,
    refine_trajectory_basis_from_state_current_samples,
)


def test_cli_entrypoint_imports_physical_library_extractor() -> None:
    assert callable(main)


def test_source_recipe_salts_become_conserved_ionic_components() -> None:
    composition_record = {
        "temperature_K": T_REF_K,
        "source_recipe": {
            "solvents": {"EC": 0.34, "DMC": 0.66},
            "salts": {"LiPF6": 0.8, "LiFSI": 0.3},
            "additives": {"VC": 0.005},
        },
    }

    recipe_record = _projected_recipe_record_from_composition_record(
        composition_record,
    )

    assert recipe_record == {
        "temperature_K": T_REF_K,
        "solvents_vv": {"EC": 0.34, "DMC": 0.66},
        "salts_mol_l": {"Li+": 1.1, "PF6-": 0.8, "FSI-": 0.3},
        "additives_weight_fraction": {"VC": 0.005},
    }


def test_transition_displacement_uses_local_endpoint_charge_polarization() -> None:
    center_catalog = _two_center_catalog()
    center_frames = (
        _center_frame(((9.5, 0.0, 0.0), (3.0, 0.0, 0.0))),
        _center_frame(((10.5, 0.0, 0.0), (5.0, 0.0, 0.0))),
    )

    displacements_m = _charge_displacements_by_step_m(
        center_frames,
        center_catalog,
        state_index_by_frame_and_center=np.asarray(((0, 2), (1, 2)), dtype=int),
        counterion_index_by_frame_and_center=np.asarray(((1, 0), (1, 0)), dtype=int),
        thresholds=AssociationThresholds(3.0, 7.0),
    )

    assert displacements_m[0] == pytest.approx((-1.0e-10, 0.0, 0.0))
    assert displacements_m[1] == pytest.approx((-1.0e-10, 0.0, 0.0))


def test_transition_displacement_requires_unique_stable_center_identities() -> None:
    center_catalog = ChargedCenterCatalog(
        molecule_ids=np.asarray((7, 7), dtype=int),
        species_labels=("Li+", "PF6-"),
        roles=("cation", "anion"),
        formal_charges_e=np.asarray((1.0, -1.0), dtype=float),
    )

    with pytest.raises(ValueError, match="identities must be unique"):
        _charge_displacements_by_step_m(
            (_center_frame(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),) * 2,
            center_catalog,
            state_index_by_frame_and_center=np.asarray(((0, 0), (1, 0)), dtype=int),
            counterion_index_by_frame_and_center=np.asarray(((1, 0), (1, 0)), dtype=int),
            thresholds=AssociationThresholds(3.0, 7.0),
        )


def test_transition_second_moment_survives_primitive_tensor_assembly() -> None:
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("A", "B"),
        occupancy_state_index_by_observation=np.asarray((0, 1), dtype=int),
        from_state_index_by_step=np.asarray((0, 1), dtype=int),
        to_state_index_by_step=np.asarray((1, 0), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            ((1.0e-10, 2.0e-10, 0.0), (-3.0e-10, 1.0e-10, 0.0)),
            dtype=float,
        ),
        self_charge_polarization_by_frame_and_center_m=np.zeros((5, 2, 3)),
        state_index_by_frame_and_center=np.asarray(((0, 1),) * 5, dtype=int),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )

    primitive_set = project_sampled_trajectory_to_generator_primitives(sample_input)
    primitive_arrays = _primitive_arrays_from_projected_set(primitive_set)
    oriented_samples_m = np.asarray(
        ((1.0e-10, 2.0e-10, 0.0), (3.0e-10, -1.0e-10, 0.0)),
        dtype=float,
    )
    expected_second_moment_m2 = np.einsum(
        "sa,sb->ab", oriented_samples_m, oriented_samples_m
    ) / 2.0

    assert primitive_arrays["transition_second_moments_M_ij_m2"][0, 1] == (
        pytest.approx(expected_second_moment_m2)
    )
    assert primitive_arrays["transition_second_moments_M_ij_m2"][1, 0] == (
        pytest.approx(expected_second_moment_m2)
    )


def test_trajectory_basis_refinement_exhausts_measured_state_current_candidates() -> None:
    displacements_m = np.asarray(
        (
            (1.0e-10, 0.0, 0.0),
            (0.0, 2.0e-10, 0.0),
            (-1.0e-10, 0.0, 0.0),
            (0.0, -2.0e-10, 0.0),
            (1.0e-10, 0.0, 0.0),
            (0.0, 2.0e-10, 0.0),
        ),
        dtype=float,
    )
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("A", "B"),
        occupancy_state_index_by_observation=np.asarray((0, 1), dtype=int),
        from_state_index_by_step=np.asarray((0, 1, 0, 1, 0, 1), dtype=int),
        to_state_index_by_step=np.asarray((0, 1, 0, 1, 0, 1), dtype=int),
        charge_displacement_by_step_m=displacements_m,
        self_charge_polarization_by_frame_and_center_m=np.zeros((5, 2, 3)),
        state_index_by_frame_and_center=np.asarray(((0, 1),) * 5, dtype=int),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )

    refinement = refine_trajectory_basis_from_state_current_samples(
        sample_input=sample_input,
        state_labels=("A", "B"),
        state_index_by_step=sample_input.from_state_index_by_step,
        samples_per_frame=2,
        direct_diffusivity_tensor_m2_s=np.eye(3) * 1.0e-6,
        residual_score_tolerance_m2_s=1.0e-20,
        conductivity_change_tolerance_S_m=1.0e6,
    )

    assert refinement.candidate_sample_count == 4
    assert refinement.candidate_set_exhausted
    assert refinement.convergence_status == "converged"
    assert refinement.selected_candidate_indices
    assert np.max(np.abs(refinement.final_mori_memory_matrix_A)) > 0.0
    assert np.max(np.abs(refinement.final_mori_current_coupling_matrix_h)) > 0.0


def test_state_conditioned_long_time_covariance_recovers_brownian_diffusion() -> None:
    random_generator = np.random.default_rng(481516)
    frame_count = 96
    center_count = 192
    timestep_s = 2.0e-12
    diffusion_m2_s = 1.7e-10
    increment_standard_deviation_m = np.sqrt(2.0 * diffusion_m2_s * timestep_s)
    increments_m = random_generator.normal(
        scale=increment_standard_deviation_m,
        size=(frame_count - 1, center_count, 3),
    )
    polarizations_m = np.concatenate(
        (np.zeros((1, center_count, 3)), np.cumsum(increments_m, axis=0)), axis=0
    )

    tensors, diagnostics = _self_current_tensors(
        state_labels=("free",),
        state_index_by_frame_and_center=np.zeros((frame_count, center_count), dtype=int),
        charge_polarization_by_frame_and_center_m=polarizations_m,
        state_concentrations={"free": 1000.0},
        timestep_s=timestep_s,
    )

    assert diagnostics[0].convergence_status == "converged"
    assert diagnostics[0].lag_count >= 4
    assert diagnostics[0].log_log_exponent == pytest.approx(1.0, abs=0.2)
    assert np.trace(np.asarray(tensors[0].diffusion_tensor_m2_s)) / 3.0 == pytest.approx(
        diffusion_m2_s, rel=0.2
    )


def test_state_conditioned_long_time_covariance_rejects_ballistic_motion() -> None:
    random_generator = np.random.default_rng(8675309)
    frame_count = 48
    center_count = 128
    timestep_s = 1.0e-12
    velocities_m_s = random_generator.normal(scale=120.0, size=(center_count, 3))
    times_s = np.arange(frame_count, dtype=float)[:, np.newaxis, np.newaxis] * timestep_s
    polarizations_m = times_s * velocities_m_s[np.newaxis, :, :]

    tensors, diagnostics = _self_current_tensors(
        state_labels=("free",),
        state_index_by_frame_and_center=np.zeros((frame_count, center_count), dtype=int),
        charge_polarization_by_frame_and_center_m=polarizations_m,
        state_concentrations={"free": 1000.0},
        timestep_s=timestep_s,
    )

    assert tensors == ()
    assert diagnostics[0].convergence_status == "not_converged"
    assert "linear covariance growth" in diagnostics[0].not_complete_reason


def _two_center_catalog() -> ChargedCenterCatalog:
    return ChargedCenterCatalog(
        molecule_ids=np.asarray((7, 11), dtype=int),
        species_labels=("Li+", "PF6-"),
        roles=("cation", "anion"),
        formal_charges_e=np.asarray((1.0, -1.0), dtype=float),
    )


def _center_frame(positions_A) -> ChargedCenterFrame:
    positions = np.asarray(positions_A, dtype=float)
    return ChargedCenterFrame(
        positions_A=positions,
        wrapped_positions_A=np.mod(positions, 10.0),
        box_bounds_A=np.asarray(((0.0, 10.0),) * 3, dtype=float),
    )
