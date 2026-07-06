import numpy as np

from constants import T_REF_K
from conductivity.analytical_conductivity_model import (
    PROJECTION_CLASS_SELF_CURRENT_CARRIER,
    MicroscopicGeneratorModel,
    ProjectedBasisAssignment,
    ProjectedBasisFunctionDefinition,
    ProjectedGeneratorBuilder,
    estimate_direct_green_kubo_conductivity,
    project_microscopic_generator,
)
from conductivity.fm_md.atomistic_io import MolecularTrajectory


def _synthetic_charged_pair_trajectory() -> MolecularTrajectory:
    frame_count = 6
    frame_positions_x_angstrom = np.arange(frame_count, dtype=float)
    positions_angstrom = np.stack(
        (
            np.column_stack(
                (
                    frame_positions_x_angstrom,
                    np.zeros(frame_count, dtype=float),
                    np.zeros(frame_count, dtype=float),
                )
            ),
            np.column_stack(
                (
                    -frame_positions_x_angstrom,
                    np.zeros(frame_count, dtype=float),
                    np.zeros(frame_count, dtype=float),
                )
            ),
        ),
        axis=1,
    )
    return MolecularTrajectory(
        com_positions=positions_angstrom,
        molecule_species=np.asarray((0, 1), dtype=np.int32),
        formal_charges=np.asarray((1.0, -1.0), dtype=float),
        box=np.asarray((20.0, 20.0, 20.0), dtype=float),
        dt_fs=1000.0,
        n_frames=frame_count,
        n_molecules=2,
        temperature_K=T_REF_K,
    )


def test_sampled_microscopic_generator_exposes_charge_observables():
    generator_model = MicroscopicGeneratorModel(
        configuration_space="synthetic_two_center_configuration_space",
        equilibrium_measure="sampled_equal_weight_equilibrium_measure",
        reversible_generator="sampled_reversible_shift_generator",
        charge_polarization_observable="P=sum_a z_a R_a",
        trajectory=_synthetic_charged_pair_trajectory(),
    )

    charge_polarization = generator_model.charge_polarization_charge_number_m()
    charge_current = generator_model.charge_current_charge_number_m_s()

    assert charge_polarization.shape == (6, 3)
    assert np.isclose(charge_polarization[-1, 0], 1.0e-9)
    assert charge_current.shape == (4, 3)
    assert np.allclose(charge_current[:, 0], 200.0)
    assert generator_model.charged_center_concentration_mol_m3() > 0.0


def test_project_generator_returns_c_k_m_and_self_current_from_basis_assignment():
    generator_model = MicroscopicGeneratorModel(
        configuration_space="synthetic_two_center_configuration_space",
        equilibrium_measure="sampled_equal_weight_equilibrium_measure",
        reversible_generator="sampled_reversible_shift_generator",
        charge_polarization_observable="P=sum_a z_a R_a",
        trajectory=_synthetic_charged_pair_trajectory(),
    )
    state_index_by_frame_and_molecule = np.asarray(
        (
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ),
        dtype=int,
    )
    projected_generator = project_microscopic_generator(
        generator_model=generator_model,
        basis_assignment=ProjectedBasisAssignment(
            basis_functions=(
                ProjectedBasisFunctionDefinition(
                    state_label="free_ion_center:Li+",
                    projection_class=PROJECTION_CLASS_SELF_CURRENT_CARRIER,
                ),
                ProjectedBasisFunctionDefinition(
                    state_label="free_ion_center:PF6-",
                    projection_class=PROJECTION_CLASS_SELF_CURRENT_CARRIER,
                ),
            ),
            state_index_by_frame_and_molecule=state_index_by_frame_and_molecule,
        ),
    )

    primitive_set = projected_generator.primitive_set

    assert primitive_set.state_labels == (
        "free_ion_center:Li+",
        "free_ion_center:PF6-",
    )
    assert len(primitive_set.reactive_fluxes) == 0
    assert len(primitive_set.self_current_tensors) == 2
    self_current_by_label = {
        self_current.state_label: np.asarray(
            self_current.diffusion_tensor_m2_s,
            dtype=float,
        )
        for self_current in primitive_set.self_current_tensors
    }
    assert np.isclose(self_current_by_label["free_ion_center:Li+"][0, 0], 5.0e-18)
    assert np.isclose(self_current_by_label["free_ion_center:PF6-"][0, 0], 5.0e-18)
    assert primitive_set.markov_conductivity_result.sigma_mS_cm >= 0.0


class _FixedTwoStateBasisAssigner:
    def assign_basis(self, trajectory: MolecularTrajectory) -> ProjectedBasisAssignment:
        frame_count = int(trajectory.com_positions.shape[0])
        state_index_by_frame_and_molecule = np.repeat(
            np.asarray(((0, 1),), dtype=int),
            frame_count,
            axis=0,
        )
        return ProjectedBasisAssignment(
            basis_functions=(
                ProjectedBasisFunctionDefinition(
                    state_label="free_ion_center:Li+",
                    projection_class=PROJECTION_CLASS_SELF_CURRENT_CARRIER,
                ),
                ProjectedBasisFunctionDefinition(
                    state_label="free_ion_center:PF6-",
                    projection_class=PROJECTION_CLASS_SELF_CURRENT_CARRIER,
                ),
            ),
            state_index_by_frame_and_molecule=state_index_by_frame_and_molecule,
        )


def test_projected_generator_builder_exposes_theorem_primitive_set():
    generator_model = MicroscopicGeneratorModel(
        configuration_space="synthetic_two_center_configuration_space",
        equilibrium_measure="sampled_equal_weight_equilibrium_measure",
        reversible_generator="sampled_reversible_shift_generator",
        charge_polarization_observable="P=sum_a z_a R_a",
        trajectory=_synthetic_charged_pair_trajectory(),
    )

    projected_generator = ProjectedGeneratorBuilder(
        generator_model=generator_model,
        basis_assigner=_FixedTwoStateBasisAssigner(),
    ).build_projected_generator()
    primitive_set = projected_generator.projected_primitives

    assert primitive_set.state_labels == (
        "free_ion_center:Li+",
        "free_ion_center:PF6-",
    )
    assert primitive_set.restricted_equilibrium_populations_c_i_mol_m3.shape == (2,)
    assert primitive_set.symmetric_reactive_fluxes_K_ij_mol_m3_s.shape == (2, 2)
    assert primitive_set.reversible_generator_Q_ij_s_inv.shape == (2, 2)
    assert primitive_set.conditional_displacement_first_moments_d_ij_m.shape == (
        2,
        2,
        3,
    )
    assert primitive_set.conditional_displacement_second_moments_M_ij_m2.shape == (
        2,
        2,
        3,
        3,
    )
    assert primitive_set.self_current_diffusion_tensors_D_self_i_m2_s.shape == (
        2,
        3,
        3,
    )
    assert primitive_set.mori_memory_energy_matrix_A.shape == (2, 2)
    assert primitive_set.mori_current_coupling_matrix_h.shape == (3, 2)
    assert np.all(
        primitive_set.restricted_equilibrium_populations_c_i_mol_m3 > 0.0
    )
    assert np.isclose(
        primitive_set.self_current_diffusion_tensors_D_self_i_m2_s[0, 0, 0],
        5.0e-18,
    )
    assert (
        primitive_set.markov_conductivity_result.sigma_mS_cm
        == projected_generator.primitive_set.markov_conductivity_result.sigma_mS_cm
    )


def test_finite_time_green_kubo_estimator_is_positive_for_charge_separation():
    generator_model = MicroscopicGeneratorModel(
        configuration_space="synthetic_two_center_configuration_space",
        equilibrium_measure="sampled_equal_weight_equilibrium_measure",
        reversible_generator="sampled_reversible_shift_generator",
        charge_polarization_observable="P=sum_a z_a R_a",
        trajectory=_synthetic_charged_pair_trajectory(),
    )

    green_kubo_estimate = estimate_direct_green_kubo_conductivity(
        charge_current_charge_number_m_s=(
            generator_model.charge_current_charge_number_m_s()
        ),
        dt_s=generator_model.dt_s,
        volume_m3=generator_model.volume_m3,
        temperature_K=generator_model.temperature_K,
        integration_stop_index=4,
    )

    assert green_kubo_estimate.sigma_mS_cm > 0.0
    assert green_kubo_estimate.integral_charge_number_m2_s > 0.0
