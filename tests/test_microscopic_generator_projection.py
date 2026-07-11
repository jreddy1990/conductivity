import numpy as np

from constants import T_REF_K
from conductivity.physical_library.projected_analytical_conductivity import (
    ProjectedPrimitiveInput,
    compute_projected_analytical_conductivity_from_primitives,
    compute_reversible_generator,
)
from conductivity.physical_library.trajectory_primitives import (
    TrajectoryMarkovAdditiveSampleInput,
    compute_finite_process_component_drift_residuals,
    project_sampled_trajectory_to_generator_primitives,
)


def test_sampled_trajectory_projection_exposes_reversible_primitives() -> None:
    frame_count = 96
    state_index_by_frame_and_center = np.tile(
        np.asarray([[0, 1]], dtype=int),
        (frame_count, 1),
    )
    random_generator = np.random.default_rng(1742)
    polarization_increments_m = random_generator.normal(
        scale=1.0e-11,
        size=(frame_count - 1, 2, 3),
    )
    charge_polarization_by_frame_and_center_m = np.concatenate(
        (
            np.zeros((1, 2, 3), dtype=float),
            np.cumsum(polarization_increments_m, axis=0),
        ),
        axis=0,
    )
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("free_ion_center:Li+", "free_ion_center:PF6-"),
        occupancy_state_index_by_observation=state_index_by_frame_and_center.reshape(-1),
        from_state_index_by_step=np.asarray([0, 1], dtype=int),
        to_state_index_by_step=np.asarray([1, 0], dtype=int),
        charge_displacement_by_step_m=np.asarray(
            [[2.0e-10, 0.0, 0.0], [-2.0e-10, 0.0, 0.0]],
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
        self_charge_polarization_by_frame_and_center_m=(
            charge_polarization_by_frame_and_center_m
        ),
        state_index_by_frame_and_center=state_index_by_frame_and_center,
    )

    primitive_set = project_sampled_trajectory_to_generator_primitives(sample_input)

    assert primitive_set.state_labels == ("free_ion_center:Li+", "free_ion_center:PF6-")
    assert primitive_set.state_concentrations_mol_m3["free_ion_center:Li+"] == 500.0
    assert len(primitive_set.reactive_fluxes) == 1
    assert len(primitive_set.conditional_displacement_moments) == 1
    assert primitive_set.diagnostics.component_drift_residuals[0].weighted_drift_norm_mol_m2_s == 0.0


def test_projected_primitive_input_returns_c_k_q_m_and_self_current() -> None:
    primitive_input = ProjectedPrimitiveInput(
        state_concentrations_mol_m3=np.asarray([500.0, 500.0], dtype=float),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
            [[0.0, 2.0e12], [2.0e12, 0.0]],
            dtype=float,
        ),
        transition_first_moments_d_ij_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [2.0e-10, 0.0, 0.0]],
                [[-2.0e-10, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        transition_second_moments_M_ij_m2=np.asarray(
            [
                [np.zeros((3, 3), dtype=float), np.diag([4.0e-20, 0.0, 0.0])],
                [np.diag([4.0e-20, 0.0, 0.0]), np.zeros((3, 3), dtype=float)],
            ],
            dtype=float,
        ),
        self_current_tensors_D_self_i_m2_s=np.asarray(
            [np.eye(3) * 1.0e-10, np.eye(3) * 1.0e-10],
            dtype=float,
        ),
        mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        mori_current_coupling_matrix_h=np.zeros((0, 3), dtype=float),
        temperature_K=T_REF_K,
        volume_m3=1.0,
    )

    result = compute_projected_analytical_conductivity_from_primitives(
        primitive_input.state_concentrations_mol_m3,
        primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s,
        primitive_input.transition_first_moments_d_ij_m,
        primitive_input.transition_second_moments_M_ij_m2,
        primitive_input.self_current_tensors_D_self_i_m2_s,
        primitive_input.mori_memory_matrix_A,
        primitive_input.mori_current_coupling_matrix_h,
        primitive_input.temperature_K,
        primitive_input.volume_m3,
    )
    generator = compute_reversible_generator(
        result.symmetric_capacity_fluxes_K_ij_mol_m3_s,
        result.state_concentrations_mol_m3,
    )

    assert result.state_concentrations_mol_m3.shape == (2,)
    assert result.symmetric_capacity_fluxes_K_ij_mol_m3_s.shape == (2, 2)
    assert generator.shape == (2, 2)
    assert result.transition_second_moments_M_ij_m2.shape == (2, 2, 3, 3)
    assert result.self_current_tensors_D_self_i_m2_s.shape == (2, 3, 3)
    assert result.sigma_mS_cm >= 0.0


def test_component_drift_residual_identifies_missing_reverse_event() -> None:
    residual = compute_finite_process_component_drift_residuals(
        state_labels=("state_a", "state_b"),
        state_concentrations_mol_m3=np.asarray([500.0, 500.0], dtype=float),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
            [[0.0, 1.0e12], [1.0e12, 0.0]],
            dtype=float,
        ),
        transition_first_moments_d_ij_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [2.0e-10, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        directed_transition_sample_counts=np.asarray([[0, 3], [0, 0]], dtype=int),
    )[0]

    assert residual.weighted_drift_norm_mol_m2_s > 0.0
    assert residual.top_edge_contributions[0].missing_reverse_event_candidate
