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
    _extract_committed_transition_events,
    _find_diffusive_covariance_window,
    _self_current_tensors,
    project_sampled_trajectory_to_generator_primitives,
    refine_trajectory_basis_from_state_current_samples,
)


def test_aba_recrossing_emits_no_committed_transition() -> None:
    state_indices = np.asarray(
        ((0,), (0,), (0,), (0,), (0,), (1,), (1,), (1,), (0,), (0,), (0,), (0,), (0,)),
        dtype=int,
    )
    polarizations_m = np.zeros((*state_indices.shape, 3), dtype=float)

    events = _extract_committed_transition_events(
        state_index_by_frame_and_center=state_indices,
        charge_polarization_by_frame_and_center_m=polarizations_m,
        timestep_s=1.0e-12,
        commitment_time_s=5.0e-12,
    )

    assert events == ()


def test_committed_transition_uses_persistent_endpoint_displacement() -> None:
    state_indices = np.asarray(
        ((0,), (0,), (0,), (0,), (0,), (1,), (1,), (1,), (1,), (1,), (1,)),
        dtype=int,
    )
    x_displacement_m = np.arange(state_indices.shape[0], dtype=float) * 1.0e-10
    polarizations_m = np.column_stack(
        (x_displacement_m, np.zeros_like(x_displacement_m), np.zeros_like(x_displacement_m))
    )[:, np.newaxis, :]

    events = _extract_committed_transition_events(
        state_index_by_frame_and_center=state_indices,
        charge_polarization_by_frame_and_center_m=polarizations_m,
        timestep_s=1.0e-12,
        commitment_time_s=5.0e-12,
    )

    assert len(events) == 1
    assert events[0].source_endpoint_frame == 4
    assert events[0].destination_commitment_frame == 9
    assert events[0].charge_displacement_m == pytest.approx((5.0e-10, 0.0, 0.0))


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
    random_generator = np.random.default_rng(413)
    frame_count = 96
    center_count_per_state = 96
    center_count = 2 * center_count_per_state
    timestep_s = 1.0e-12
    diffusion_m2_s = 1.0e-10
    increments_m = random_generator.normal(
        scale=np.sqrt(2.0 * diffusion_m2_s * timestep_s),
        size=(frame_count - 1, center_count, 3),
    )
    polarizations_m = np.concatenate(
        (np.zeros((1, center_count, 3)), np.cumsum(increments_m, axis=0)), axis=0
    )
    frame_indices = np.arange(frame_count)[:, np.newaxis]
    center_indices = np.arange(center_count)[np.newaxis, :]
    initial_state_indices = (center_indices >= center_count_per_state).astype(int)
    switching_centers = (center_indices == 0) | (
        center_indices == center_count_per_state
    )
    state_indices = np.where(
        (frame_indices >= frame_count // 2) & switching_centers,
        1 - initial_state_indices,
        initial_state_indices,
    )
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("A", "B"),
        occupancy_state_index_by_observation=np.asarray((0, 1), dtype=int),
        from_state_index_by_step=np.asarray((0, 1), dtype=int),
        to_state_index_by_step=np.asarray((1, 0), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            ((1.0e-10, 2.0e-10, 0.0), (-3.0e-10, 1.0e-10, 0.0)),
            dtype=float,
        ),
        self_charge_polarization_by_frame_and_center_m=polarizations_m,
        state_index_by_frame_and_center=state_indices,
        self_current_valid_step_by_center=np.ones(
            (frame_count - 1, center_count), dtype=bool
        ),
        transition_commitment_time_s=timestep_s,
        zero_frequency_integration_window_s=20.0e-12,
        zero_frequency_plateau_window_s=5.0e-12,
        dt_s=timestep_s,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )

    primitive_set = project_sampled_trajectory_to_generator_primitives(sample_input)
    primitive_arrays = _primitive_arrays_from_projected_set(primitive_set)
    oriented_samples_m = np.asarray(
        (
            increments_m[frame_count // 2 - 1, 0],
            -increments_m[frame_count // 2 - 1, center_count_per_state],
        )
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
        self_current_valid_step_by_center=np.ones((4, 2), dtype=bool),
        transition_commitment_time_s=1.0e-12,
        zero_frequency_integration_window_s=20.0e-12,
        zero_frequency_plateau_window_s=5.0e-12,
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
    center_count = 2048
    timestep_s = 1.0e-12
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
        self_current_valid_step_by_center=np.ones(
            (frame_count - 1, center_count), dtype=bool
        ),
        integration_window_s=20.0e-12,
        plateau_window_s=5.0e-12,
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
        state_index_by_frame_and_center=np.zeros(
            (frame_count, center_count), dtype=int
        ),
        charge_polarization_by_frame_and_center_m=polarizations_m,
        state_concentrations={"free": 1000.0},
        timestep_s=timestep_s,
        self_current_valid_step_by_center=np.ones(
            (frame_count - 1, center_count), dtype=bool
        ),
        integration_window_s=20.0e-12,
        plateau_window_s=5.0e-12,
    )

    assert tensors == ()
    assert diagnostics[0].convergence_status == "not_converged"
    assert "linear covariance growth" in diagnostics[0].not_complete_reason


def test_zero_frequency_diffusion_is_invariant_to_dump_stride() -> None:
    random_generator = np.random.default_rng(1207)
    timestep_s = 0.5e-12
    frame_count = 161
    center_count = 2048
    diffusion_m2_s = 1.3e-10
    increments_m = random_generator.normal(
        scale=np.sqrt(2.0 * diffusion_m2_s * timestep_s),
        size=(frame_count - 1, center_count, 3),
    )
    polarizations_m = np.concatenate(
        (np.zeros((1, center_count, 3)), np.cumsum(increments_m, axis=0)), axis=0
    )

    estimates_m2_s = tuple(
        np.trace(
            np.asarray(
                _self_current_tensors(
                    state_labels=("free",),
                    state_index_by_frame_and_center=np.zeros(
                        (polarizations_m[::dump_stride].shape[0], center_count),
                        dtype=int,
                    ),
                    charge_polarization_by_frame_and_center_m=polarizations_m[
                        ::dump_stride
                    ],
                    state_concentrations={"free": 1000.0},
                    timestep_s=timestep_s * dump_stride,
                    self_current_valid_step_by_center=np.ones(
                        (polarizations_m[::dump_stride].shape[0] - 1, center_count),
                        dtype=bool,
                    ),
                    integration_window_s=20.0e-12,
                    plateau_window_s=5.0e-12,
                )[0][0].diffusion_tensor_m2_s
            )
        )
        / 3.0
        for dump_stride in (1, 2)
    )

    assert estimates_m2_s[1] == pytest.approx(estimates_m2_s[0], rel=0.08)


def test_zero_frequency_plateau_rejects_terminal_transient() -> None:
    timestep_s = 1.0e-12
    lag_times_s = np.arange(1, 41, dtype=float) * timestep_s
    diffusion_m2_s = 1.0e-10
    terminal_transient_m2 = np.where(
        (lag_times_s >= 16.0e-12) & (lag_times_s <= 20.0e-12),
        (lag_times_s - 16.0e-12) * 4.5e-9,
        0.0,
    )
    trace_covariance_m2 = (
        6.0 * diffusion_m2_s * lag_times_s + terminal_transient_m2
    )
    covariance_by_lag_m2 = (
        trace_covariance_m2[:, np.newaxis, np.newaxis] * np.eye(3)[np.newaxis] / 3.0
    )

    convergence, _, converged = _find_diffusive_covariance_window(
        state_label="free",
        covariance_by_lag_m2=covariance_by_lag_m2,
        sample_count_by_lag=np.full(40, 1000, dtype=int),
        populated_lags=np.ones(40, dtype=bool),
        timestep_s=timestep_s,
        integration_window_s=20.0e-12,
        plateau_window_s=5.0e-12,
    )

    assert not converged
    assert convergence.convergence_status == "not_converged"
    assert "stable final plateau" in convergence.not_complete_reason


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
