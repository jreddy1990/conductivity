import io
import math
import tarfile

import numpy as np
import pytest

from constants import T_REF_K
from conductivity.analytical_conductivity_model import (
    PROJECTION_CLASS_DIAGNOSTIC_ONLY,
    PROJECTION_CLASS_POPULATION_BASIN,
    PROJECTION_CLASS_SELF_CURRENT_CARRIER,
    ProjectedBasisAssignment,
    ProjectedBasisFunctionDefinition,
    ProjectedChargedCenter,
    ProjectedConstraintMode,
    ProjectedTransportState,
    MicroscopicGeneratorModel,
    MarkovAdditiveConductivityInput,
    MarkovAdditiveEvent,
    OverdampedSmoluchowskiGeneratorInput,
    compute_markov_additive_green_kubo_conductivity,
    compute_projected_transport_state_charge_diffusivity_m2_s,
    compute_first_principles_projected_conductivity,
    compute_first_principles_conductivity_from_overdamped_generator,
    estimate_direct_green_kubo_conductivity,
    project_microscopic_generator,
)
from conductivity.fm_md.atomistic_io import MolecularTrajectory
from conductivity.trajectory_primitive_targets import (
    PF6AssociationCutoffs,
    PF6TrajectoryPrimitiveTargetInput,
    PF6ZenodoTrajectoryLayout,
    TrajectoryPrimitiveTargetProcessInput,
    TrajectoryMarkovAdditiveSampleInput,
    build_trajectory_primitive_target_markov_input,
    build_sampled_trajectory_markov_additive_input,
    compute_pf6_trajectory_primitive_targets,
    compute_trajectory_primitive_target_conductivity,
    project_sampled_trajectory_to_generator_primitives,
)
from conductivity.fit_conductivity_primitive_parameters import (
    _uncertainty_normalized_mapping_loss,
    _uncertainty_normalized_scalar_loss,
    default_molecular_primitive_fit_configuration,
)


def _synthetic_projected_transport_state(
    label: str,
    concentration_mol_m3: float,
    charged_centers: tuple[ProjectedChargedCenter, ...],
    constraint_modes: tuple[ProjectedConstraintMode, ...],
    mobility_covariance_matrix_m2_s: tuple[tuple[float, ...], ...],
    ligand_shell_features: dict[str, float],
    pair_basin: str,
    residence_time_s: float,
    partner_switch_time_s: float,
) -> ProjectedTransportState:
    hydrodynamic_radius_A = 2.0
    charge_cloud_radius_A = 2.5
    molecular_volume_A3 = 50.0
    ligand_field_asymmetry = 1.0
    diffusion_m2_s = max(
        center.diffusion_m2_s for center in charged_centers
    )
    return ProjectedTransportState(
        label=label,
        concentration_mol_m3=concentration_mol_m3,
        charged_centers=charged_centers,
        constraint_modes=constraint_modes,
        atmosphere_resistance_matrix_kg_s=tuple(),
        mobility_covariance_matrix_m2_s=mobility_covariance_matrix_m2_s,
        ligand_shell_features=ligand_shell_features,
        pair_basin=pair_basin,
        residence_time_s=residence_time_s,
        partner_switch_time_s=partner_switch_time_s,
        parent_cluster_label=label,
        parent_cluster_kind=pair_basin,
        center_species_name="synthetic_projected_motif",
        center_charge_number=sum(center.charge_number for center in charged_centers),
        center_index=0,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        molecular_volume_A3=molecular_volume_A3,
        ligand_field_asymmetry=ligand_field_asymmetry,
        diffusion_m2_s=diffusion_m2_s,
        local_obstruction_factor=1.0,
        local_obstruction_diffusion_scale=1.0,
        transport_role=pair_basin,
    )


def _synthetic_charged_pair_trajectory() -> MolecularTrajectory:
    frame_count = 6
    frame_offsets_angstrom = np.arange(frame_count, dtype=float)
    zero_offsets_angstrom = np.zeros(frame_count, dtype=float)
    positive_positions_angstrom = np.column_stack(
        (
            frame_offsets_angstrom,
            zero_offsets_angstrom,
            zero_offsets_angstrom,
        )
    )
    negative_positions_angstrom = np.column_stack(
        (
            -frame_offsets_angstrom,
            zero_offsets_angstrom,
            zero_offsets_angstrom,
        )
    )
    positions_angstrom = np.stack(
        (positive_positions_angstrom, negative_positions_angstrom),
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


def _synthetic_microscopic_generator_model() -> MicroscopicGeneratorModel:
    return MicroscopicGeneratorModel(
        configuration_space="synthetic_unwrapped_two_center_configuration_space",
        equilibrium_measure="sampled_equal_weight_equilibrium_measure",
        reversible_generator="sampled_reversible_shift_generator",
        charge_polarization_observable="P=sum_a z_a R_a",
        trajectory=_synthetic_charged_pair_trajectory(),
    )


class _ZeroPotentialModel:
    def potential_energy_J(self, positions_m: np.ndarray) -> float:
        return float(np.sum(positions_m * 0.0))

    def forces_N(self, positions_m: np.ndarray) -> np.ndarray:
        return np.zeros_like(positions_m, dtype=float)


class _ChargeSignBasisAssigner:
    def assign_basis(self, trajectory: MolecularTrajectory) -> ProjectedBasisAssignment:
        state_index_by_molecule = np.where(
            np.asarray(trajectory.formal_charges, dtype=float) > 0.0,
            0,
            1,
        )
        state_index_by_frame_and_molecule = np.repeat(
            state_index_by_molecule[np.newaxis, :],
            int(trajectory.com_positions.shape[0]),
            axis=0,
        )
        return ProjectedBasisAssignment(
            basis_functions=(
                ProjectedBasisFunctionDefinition(
                    state_label="free_ion_center:positive",
                    projection_class=PROJECTION_CLASS_SELF_CURRENT_CARRIER,
                ),
                ProjectedBasisFunctionDefinition(
                    state_label="free_ion_center:negative",
                    projection_class=PROJECTION_CLASS_SELF_CURRENT_CARRIER,
                ),
            ),
            state_index_by_frame_and_molecule=state_index_by_frame_and_molecule,
        )


def test_overdamped_generator_runs_full_first_principles_projection_pipeline():
    target_absolute_error_mS_cm = 6.0e-2
    projected_model = compute_first_principles_conductivity_from_overdamped_generator(
        generator_input=OverdampedSmoluchowskiGeneratorInput(
            configuration_space="overdamped_two_ion_periodic_configuration_space",
            equilibrium_measure="boltzmann_measure_for_zero_test_potential",
            reversible_generator="overdamped_smoluchowski_langevin_generator",
            charge_polarization_observable="P=sum_a z_a R_a",
            potential_model=_ZeroPotentialModel(),
            initial_positions_m=np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (2.0e-10, 0.0, 0.0),
                ),
                dtype=float,
            ),
            molecule_species_indices=np.asarray((0, 1), dtype=int),
            formal_charge_numbers=np.asarray((1.0, -1.0), dtype=float),
            box_lengths_m=np.asarray((2.0e-9, 2.0e-9, 2.0e-9), dtype=float),
            diffusion_coefficients_m2_s=np.asarray((1.0e-11, 1.0e-11), dtype=float),
            temperature_K=T_REF_K,
            dt_s=1.0e-12,
            frame_count=8,
            rng_seed=7,
        ),
        basis_assigner=_ChargeSignBasisAssigner(),
        green_kubo_integration_stop_index=4,
        einstein_helfand_fit_start_index=0,
        einstein_helfand_fit_stop_index=8,
        target_absolute_error_mS_cm=target_absolute_error_mS_cm,
    )

    acceptance_test = projected_model.projection_acceptance_test

    assert projected_model.generator_model.configuration_space == (
        "overdamped_two_ion_periodic_configuration_space"
    )
    assert projected_model.projected_generator.primitive_set.state_labels == (
        "free_ion_center:positive",
        "free_ion_center:negative",
    )
    assert len(projected_model.projected_generator.primitive_set.self_current_tensors) == 2
    assert acceptance_test.raw_green_kubo_sigma_mS_cm >= 0.0
    assert acceptance_test.raw_einstein_helfand_sigma_mS_cm >= 0.0
    assert acceptance_test.projected_sigma_mS_cm >= 0.0
    assert (
        acceptance_test.maximum_acceptance_gap_mS_cm
        <= target_absolute_error_mS_cm
    )
    assert acceptance_test.passed


def test_sampled_microscopic_generator_exposes_charge_observables():
    generator_model = _synthetic_microscopic_generator_model()

    charge_polarization = generator_model.charge_polarization_charge_number_m()
    charge_current = generator_model.charge_current_charge_number_m_s()

    assert charge_polarization.shape == (6, 3)
    assert np.isclose(charge_polarization[-1, 0], 1.0e-9)
    assert charge_current.shape == (4, 3)
    assert np.allclose(charge_current[:, 0], 200.0)
    assert generator_model.charged_center_concentration_mol_m3() > 0.0


def test_project_microscopic_generator_returns_c_k_m_and_self_current():
    generator_model = _synthetic_microscopic_generator_model()
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
    assert projected_generator.mori_operator.energy_matrix.shape == (2, 2)
    assert projected_generator.mori_operator.current_coupling_matrix.shape == (3, 2)


def test_multicenter_state_charge_diffusivity_uses_center_covariance():
    projected_state = _synthetic_projected_transport_state(
        label="feature_keyed:ssip:ligand_shell:bulky_anion",
        concentration_mol_m3=1000.0,
        charged_centers=(
            ProjectedChargedCenter(
                label="cation_center",
                charge_number=1,
                diffusion_m2_s=1.0e-10,
            ),
            ProjectedChargedCenter(
                label="anion_center",
                charge_number=-1,
                diffusion_m2_s=2.0e-10,
            ),
        ),
        constraint_modes=tuple(),
        mobility_covariance_matrix_m2_s=(
            (1.0e-10, -0.5e-10),
            (-0.5e-10, 2.0e-10),
        ),
        ligand_shell_features={"neutral_ligand_site_occupancy": 1.0},
        pair_basin="solvent_separated_pair",
        residence_time_s=1.0e-9,
        partner_switch_time_s=2.0e-9,
    )

    charge_diffusivity_m2_s = (
        compute_projected_transport_state_charge_diffusivity_m2_s(
            projected_state,
            T_REF_K,
        )
    )

    assert charge_diffusivity_m2_s == pytest.approx(4.0e-10)


def test_neutral_pair_no_comotion_gives_sum_of_center_diffusivities():
    projected_state = _synthetic_projected_transport_state(
        label="feature_keyed:ssip:no_comotion",
        concentration_mol_m3=1000.0,
        charged_centers=(
            ProjectedChargedCenter(
                label="cation_center",
                charge_number=1,
                diffusion_m2_s=1.5e-10,
            ),
            ProjectedChargedCenter(
                label="anion_center",
                charge_number=-1,
                diffusion_m2_s=2.5e-10,
            ),
        ),
        constraint_modes=tuple(),
        mobility_covariance_matrix_m2_s=(
            (1.5e-10, 0.0),
            (0.0, 2.5e-10),
        ),
        ligand_shell_features={"neutral_ligand_site_occupancy": 0.0},
        pair_basin="solvent_separated_pair",
        residence_time_s=1.0e-9,
        partner_switch_time_s=math.inf,
    )

    assert compute_projected_transport_state_charge_diffusivity_m2_s(
        projected_state,
        T_REF_K,
    ) == pytest.approx(4.0e-10)


def test_neutral_pair_perfect_comotion_gives_zero_charge_diffusivity():
    projected_state = _synthetic_projected_transport_state(
        label="feature_keyed:contact_pair:perfect_comotion",
        concentration_mol_m3=1000.0,
        charged_centers=(
            ProjectedChargedCenter(
                label="cation_center",
                charge_number=1,
                diffusion_m2_s=2.0e-10,
            ),
            ProjectedChargedCenter(
                label="anion_center",
                charge_number=-1,
                diffusion_m2_s=2.0e-10,
            ),
        ),
        constraint_modes=tuple(),
        mobility_covariance_matrix_m2_s=(
            (2.0e-10, 2.0e-10),
            (2.0e-10, 2.0e-10),
        ),
        ligand_shell_features={"neutral_ligand_site_occupancy": 0.0},
        pair_basin="contact_pair",
        residence_time_s=1.0e-9,
        partner_switch_time_s=math.inf,
    )

    assert compute_projected_transport_state_charge_diffusivity_m2_s(
        projected_state,
        T_REF_K,
    ) == pytest.approx(0.0)


def test_negative_li_anion_covariance_increases_charge_diffusivity():
    independent_state = _synthetic_projected_transport_state(
        label="feature_keyed:ssip:independent",
        concentration_mol_m3=1000.0,
        charged_centers=(
            ProjectedChargedCenter(
                label="cation_center",
                charge_number=1,
                diffusion_m2_s=1.0e-10,
            ),
            ProjectedChargedCenter(
                label="anion_center",
                charge_number=-1,
                diffusion_m2_s=1.0e-10,
            ),
        ),
        constraint_modes=tuple(),
        mobility_covariance_matrix_m2_s=(
            (1.0e-10, 0.0),
            (0.0, 1.0e-10),
        ),
        ligand_shell_features={"neutral_ligand_site_occupancy": 1.0},
        pair_basin="additive_separated_solvent_separated_pair",
        residence_time_s=1.0e-9,
        partner_switch_time_s=1.0e-8,
    )
    anticorrelated_state = _synthetic_projected_transport_state(
        label="feature_keyed:ssip:ligand_shell:bulky_anion:anticorrelated",
        concentration_mol_m3=1000.0,
        charged_centers=independent_state.charged_centers,
        constraint_modes=tuple(),
        mobility_covariance_matrix_m2_s=(
            (1.0e-10, -0.4e-10),
            (-0.4e-10, 1.0e-10),
        ),
        ligand_shell_features={"neutral_ligand_site_occupancy": 1.0},
        pair_basin="additive_separated_solvent_separated_pair",
        residence_time_s=1.0e-9,
        partner_switch_time_s=1.0e-8,
    )

    independent_diffusivity_m2_s = (
        compute_projected_transport_state_charge_diffusivity_m2_s(
            independent_state,
            T_REF_K,
        )
    )
    anticorrelated_diffusivity_m2_s = (
        compute_projected_transport_state_charge_diffusivity_m2_s(
            anticorrelated_state,
            T_REF_K,
        )
    )

    assert independent_diffusivity_m2_s == pytest.approx(2.0e-10)
    assert anticorrelated_diffusivity_m2_s == pytest.approx(2.8e-10)
    assert anticorrelated_diffusivity_m2_s > independent_diffusivity_m2_s


def test_projected_constraint_mode_suppresses_neutral_pair_dc_self_current():
    weakly_constrained_state = _synthetic_projected_transport_state(
        label="neutral_pair:weak_constraint",
        concentration_mol_m3=1000.0,
        charged_centers=(
            ProjectedChargedCenter(
                label="cation_center",
                charge_number=1,
                diffusion_m2_s=1.0e-10,
            ),
            ProjectedChargedCenter(
                label="anion_center",
                charge_number=-1,
                diffusion_m2_s=1.0e-10,
            ),
        ),
        constraint_modes=(
            ProjectedConstraintMode(
                first_center_label="cation_center",
                second_center_label="anion_center",
                lifetime_s=1.0e-13,
                length_m=5.0e-10,
            ),
        ),
        mobility_covariance_matrix_m2_s=tuple(),
        ligand_shell_features={"neutral_ligand_site_occupancy": 0.0},
        pair_basin="contact_pair",
        residence_time_s=1.0e-13,
        partner_switch_time_s=math.inf,
    )
    strongly_constrained_state = _synthetic_projected_transport_state(
        label="neutral_pair:strong_constraint",
        concentration_mol_m3=1000.0,
        charged_centers=weakly_constrained_state.charged_centers,
        constraint_modes=(
            ProjectedConstraintMode(
                first_center_label="cation_center",
                second_center_label="anion_center",
                lifetime_s=1.0e-4,
                length_m=5.0e-10,
            ),
        ),
        mobility_covariance_matrix_m2_s=tuple(),
        ligand_shell_features={"neutral_ligand_site_occupancy": 0.0},
        pair_basin="contact_pair",
        residence_time_s=1.0e-4,
        partner_switch_time_s=math.inf,
    )

    weak_charge_diffusivity_m2_s = (
        compute_projected_transport_state_charge_diffusivity_m2_s(
            weakly_constrained_state,
            T_REF_K,
        )
    )
    strong_charge_diffusivity_m2_s = (
        compute_projected_transport_state_charge_diffusivity_m2_s(
            strongly_constrained_state,
            T_REF_K,
        )
    )

    assert strong_charge_diffusivity_m2_s < weak_charge_diffusivity_m2_s
    assert strong_charge_diffusivity_m2_s < 1.0e-14


def test_pure_motif_exchange_zero_displacement_adds_no_direct_conductivity():
    markov_result = compute_markov_additive_green_kubo_conductivity(
        MarkovAdditiveConductivityInput(
            state_labels=("free_ligand_shell", "coordinated_ligand_shell"),
            state_concentrations_mol_m3=np.asarray((500.0, 500.0), dtype=float),
            events=(
                MarkovAdditiveEvent(
                    from_state_index=0,
                    to_state_index=1,
                    rate_s_inv=1.0e9,
                    charge_displacement_m=(0.0, 0.0, 0.0),
                    charge_displacement_second_moment_m2=(
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    label="motif_exchange:forward",
                    family_label="pure_motif_exchange",
                ),
                MarkovAdditiveEvent(
                    from_state_index=1,
                    to_state_index=0,
                    rate_s_inv=1.0e9,
                    charge_displacement_m=(0.0, 0.0, 0.0),
                    charge_displacement_second_moment_m2=(
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    label="motif_exchange:reverse",
                    family_label="pure_motif_exchange",
                ),
            ),
            temperature_K=T_REF_K,
        )
    )

    assert markov_result.direct_sigma_mS_cm == pytest.approx(0.0)
    assert markov_result.corrector_sigma_mS_cm == pytest.approx(0.0)
    assert markov_result.sigma_mS_cm == pytest.approx(0.0)


def test_finite_time_green_kubo_estimator_is_positive_for_charge_separation():
    generator_model = _synthetic_microscopic_generator_model()

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


def test_first_principles_projected_conductivity_reports_acceptance_test_gaps():
    generator_model = _synthetic_microscopic_generator_model()
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

    projected_model = compute_first_principles_projected_conductivity(
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
        green_kubo_integration_stop_index=4,
        einstein_helfand_fit_start_index=0,
        einstein_helfand_fit_stop_index=6,
        target_absolute_error_mS_cm=1.0,
    )

    acceptance_test = projected_model.projection_acceptance_test

    assert acceptance_test.raw_green_kubo_sigma_mS_cm > 0.0
    assert acceptance_test.raw_einstein_helfand_sigma_mS_cm > 0.0
    assert acceptance_test.projected_sigma_mS_cm >= 0.0
    assert acceptance_test.green_kubo_einstein_helfand_gap_mS_cm == abs(
        acceptance_test.raw_green_kubo_sigma_mS_cm
        - acceptance_test.raw_einstein_helfand_sigma_mS_cm
    )
    assert acceptance_test.maximum_acceptance_gap_mS_cm == max(
        acceptance_test.green_kubo_projected_gap_mS_cm,
        acceptance_test.einstein_helfand_projected_gap_mS_cm,
        acceptance_test.green_kubo_einstein_helfand_gap_mS_cm,
    )
    assert acceptance_test.passed is (
        acceptance_test.maximum_acceptance_gap_mS_cm
        <= acceptance_test.target_absolute_error_mS_cm
    )


def test_diagnostic_basis_labels_do_not_contribute_to_dc_projection():
    generator_model = _synthetic_microscopic_generator_model()
    state_index_by_frame_and_molecule = np.asarray(
        (
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 2),
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
                    state_label="unused_population_basin",
                    projection_class=PROJECTION_CLASS_POPULATION_BASIN,
                ),
                ProjectedBasisFunctionDefinition(
                    state_label="diagnostic_only:PF6-",
                    projection_class=PROJECTION_CLASS_DIAGNOSTIC_ONLY,
                ),
            ),
            state_index_by_frame_and_molecule=state_index_by_frame_and_molecule,
        ),
    )

    primitive_set = projected_generator.primitive_set

    assert primitive_set.state_concentrations_mol_m3["free_ion_center:Li+"] > 0.0
    assert primitive_set.state_concentrations_mol_m3["unused_population_basin"] == 0.0
    assert primitive_set.state_concentrations_mol_m3["diagnostic_only:PF6-"] == 0.0
    assert primitive_set.markov_conductivity_result.sigma_mS_cm > 0.0


def _two_state_process_input() -> TrajectoryPrimitiveTargetProcessInput:
    return TrajectoryPrimitiveTargetProcessInput(
        state_labels=("A", "B"),
        state_index_by_frame=np.asarray((0, 1, 0, 1, 0), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            (
                (1.0e-10, 0.0, 0.0),
                (-1.0e-10, 0.0, 0.0),
                (1.0e-10, 0.0, 0.0),
                (-1.0e-10, 0.0, 0.0),
            ),
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )


def test_trajectory_target_process_builds_reversible_pair_events():
    markov_input, diagnostics, state_index_remap = (
        build_trajectory_primitive_target_markov_input(_two_state_process_input())
    )

    assert state_index_remap == {0: 0, 1: 1}
    assert diagnostics.transition_sample_count == 4
    assert diagnostics.generated_event_count == 8
    assert markov_input.state_labels == ("A", "B")
    assert np.all(markov_input.state_concentrations_mol_m3 > 0.0)

    for event in markov_input.events:
        reverse_matches = tuple(
            candidate
            for candidate in markov_input.events
            if candidate.from_state_index == event.to_state_index
            and candidate.to_state_index == event.from_state_index
            and np.allclose(
                np.asarray(candidate.charge_displacement_m),
                -np.asarray(event.charge_displacement_m),
            )
        )
        assert reverse_matches


def test_trajectory_target_process_computes_nonnegative_conductivity():
    result = compute_trajectory_primitive_target_conductivity(
        _two_state_process_input(),
    )

    assert result.conductivity_result.sigma_mS_cm >= 0.0
    assert result.conductivity_result.event_reversal_residual_mol_m3_s >= 0.0


def test_trajectory_target_process_keeps_self_displacement_as_direct_variance():
    process_input = TrajectoryPrimitiveTargetProcessInput(
        state_labels=("A",),
        state_index_by_frame=np.asarray((0, 0, 0), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            ((1.0e-10, 0.0, 0.0), (-1.0e-10, 0.0, 0.0)),
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )

    markov_input, diagnostics, state_index_remap = (
        build_trajectory_primitive_target_markov_input(process_input)
    )

    assert state_index_remap == {0: 0}
    assert diagnostics.self_displacement_sample_count == 2
    assert diagnostics.generated_event_count == 4
    assert all(
        event.from_state_index == event.to_state_index
        for event in markov_input.events
    )


def test_sampled_trajectory_target_builds_parallel_center_process():
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("Li:contact_pair", "Li:free"),
        occupancy_state_index_by_observation=np.asarray((0, 1, 0, 1), dtype=int),
        from_state_index_by_step=np.asarray((0, 1), dtype=int),
        to_state_index_by_step=np.asarray((1, 0), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            ((1.0e-10, 0.0, 0.0), (-1.0e-10, 0.0, 0.0)),
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )

    markov_input, diagnostics, state_index_remap = (
        build_sampled_trajectory_markov_additive_input(sample_input)
    )

    assert state_index_remap == {0: 0, 1: 1}
    assert diagnostics.transition_sample_count == 2
    assert diagnostics.generated_event_count == 4
    assert markov_input.state_labels == ("Li:contact_pair", "Li:free")


def test_projected_generator_primitives_include_c_k_m_and_self_current():
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("free_ion_center:Li+", "contact_pair_center:Li+"),
        occupancy_state_index_by_observation=np.asarray(
            (0, 0, 1, 1, 0, 1),
            dtype=int,
        ),
        from_state_index_by_step=np.asarray((0, 1, 0, 1), dtype=int),
        to_state_index_by_step=np.asarray((1, 0, 0, 1), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            (
                (2.0e-10, 0.0, 0.0),
                (-2.0e-10, 0.0, 0.0),
                (1.0e-10, 0.0, 0.0),
                (-1.0e-10, 0.0, 0.0),
            ),
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1200.0,
        temperature_K=T_REF_K,
    )

    primitive_set = project_sampled_trajectory_to_generator_primitives(sample_input)

    assert primitive_set.state_concentrations_mol_m3[
        "free_ion_center:Li+"
    ] == 600.0
    assert primitive_set.state_occupancy_fractions[
        "contact_pair_center:Li+"
    ] == 0.5
    assert len(primitive_set.reactive_fluxes) == 1
    reactive_flux = primitive_set.reactive_fluxes[0]
    assert reactive_flux.from_state_label == "free_ion_center:Li+"
    assert reactive_flux.to_state_label == "contact_pair_center:Li+"
    assert reactive_flux.symmetric_flux_mol_m3_s == 3.0e14
    assert reactive_flux.forward_rate_s_inv == 5.0e11
    assert reactive_flux.reverse_rate_s_inv == 5.0e11
    assert len(primitive_set.conditional_displacement_moments) == 2
    first_moment = primitive_set.conditional_displacement_moments[0]
    covariance = np.asarray(first_moment.covariance_m2, dtype=float)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-30
    assert len(primitive_set.self_current_tensors) == 2
    self_tensor_by_label = {
        self_current.state_label: np.asarray(
            self_current.diffusion_tensor_m2_s,
            dtype=float,
        )
        for self_current in primitive_set.self_current_tensors
    }
    assert np.isclose(self_tensor_by_label["free_ion_center:Li+"][0, 0], 5.0e-9)
    assert np.isclose(
        self_tensor_by_label["contact_pair_center:Li+"][0, 0],
        5.0e-9,
    )
    assert primitive_set.markov_conductivity_result.sigma_mS_cm >= 0.0


def test_pf6_zenodo_parser_builds_primitive_targets_from_streamed_xyz(tmp_path):
    archive_path = tmp_path / "traj_PF6.tar.gz"
    member_name = "traj_PF6.xyz"
    _write_synthetic_pf6_archive(archive_path, member_name)

    target_input = PF6TrajectoryPrimitiveTargetInput(
        system_id="synthetic_ec_emc_lipf6",
        archive_path=archive_path,
        member_name=member_name,
        layout=PF6ZenodoTrajectoryLayout(
            expected_atom_count=33,
            ec_molecule_count=1,
            ec_atoms_per_molecule=10,
            emc_molecule_count=1,
            emc_atoms_per_molecule=15,
            pf6_molecule_count=1,
            pf6_atoms_per_molecule=7,
            li_atom_count=1,
        ),
        association_cutoffs=PF6AssociationCutoffs(
            contact_pair_max_distance_A=2.5,
            solvent_separated_pair_max_distance_A=5.0,
            aggregate_counterion_count=2,
        ),
        max_frames=3,
        frame_stride=1,
        block_count=1,
        temperature_K=333.0,
        expected_frame_interval_ps=1.5,
    )

    result = compute_pf6_trajectory_primitive_targets(target_input)

    assert result.diagnostics.frame_count == 3
    assert result.diagnostics.raw_frame_interval_ps == 1.5
    assert result.diagnostics.transition_sample_count == 4
    assert result.diagnostics.li_state_counts == {
        "free_ion_center:Li+": 1,
        "contact_pair_center:Li+": 1,
        "solvent_separated_pair_center:Li+": 1,
        "internal_polarization_center:Li+": 0,
    }
    assert result.diagnostics.pf6_state_counts == {
        "free_ion_center:PF6-": 1,
        "contact_pair_center:PF6-": 1,
        "solvent_separated_pair_center:PF6-": 1,
        "internal_polarization_center:PF6-": 0,
    }
    assert result.sample_input.charge_displacement_by_step_m.shape == (4, 3)
    assert result.process_result.conductivity_result.sigma_mS_cm >= 0.0
    artifact = result.primitive_target_artifact
    assert artifact.system_id == "synthetic_ec_emc_lipf6"
    assert artifact.block_count == 1
    assert artifact.state_occupancy_fractions["contact_pair_center:Li+"] > 0.0
    assert artifact.transition_rates_s_inv
    assert artifact.transition_rate_targets_validated is True
    assert artifact.displacement_moments_by_family
    assert artifact.displacement_moment_targets_validated is True
    assert artifact.markov_additive_sigma_mS_cm >= 0.0
    assert artifact.markov_additive_sigma_validated is True


def test_trajectory_primitive_fit_uses_block_uncertainty_without_config_weights():
    fit_options = default_molecular_primitive_fit_configuration()

    assert not hasattr(fit_options, "trajectory_concentration_loss_weight")
    assert not hasattr(fit_options, "trajectory_transition_rate_loss_weight")
    assert not hasattr(fit_options, "trajectory_displacement_moment_loss_weight")
    assert not hasattr(fit_options, "trajectory_sigma_loss_weight")
    assert _uncertainty_normalized_mapping_loss(
        {"free_ion_center:Li+": 14.0},
        {"free_ion_center:Li+": 10.0},
        {"free_ion_center:Li+": 2.0},
        "synthetic.state_concentrations_mol_m3",
    ) == pytest.approx(4.0)
    assert _uncertainty_normalized_scalar_loss(
        7.0,
        5.0,
        2.0,
        "synthetic.markov_additive_sigma_mS_cm",
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="missing block standard error"):
        _uncertainty_normalized_mapping_loss(
            {"free_ion_center:Li+": 14.0},
            {"free_ion_center:Li+": 10.0},
            {},
            "synthetic.state_concentrations_mol_m3",
        )
    with pytest.raises(ValueError, match="standard_error"):
        _uncertainty_normalized_scalar_loss(
            7.0,
            5.0,
            0.0,
            "synthetic.markov_additive_sigma_mS_cm",
        )


def _write_synthetic_pf6_archive(archive_path, member_name):
    frame_text = "".join(
        (
            _synthetic_pf6_frame(0.0, 20.0, 0.0),
            _synthetic_pf6_frame(1.5, 20.0, 2.5),
            _synthetic_pf6_frame(3.0, 20.0, 6.5),
        ),
    )
    frame_bytes = frame_text.encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        tar_info = _tar_info_with_size(member_name, len(frame_bytes))
        archive.addfile(tar_info, io.BytesIO(frame_bytes))


def _tar_info_with_size(member_name, size_bytes):
    tar_info = tarfile.TarInfo(member_name)
    object.__setattr__(tar_info, "size", size_bytes)
    return tar_info


def _synthetic_pf6_frame(time_ps, box_length_A, li_x_offset_A):
    atoms = (
        *_ec_atoms(),
        *_emc_atoms(),
        *_pf6_atoms(),
        ("Li", 6.5 + li_x_offset_A, 5.0, 5.0),
    )
    atom_lines = tuple(
        f"{element} {x_position_A:.6f} {y_position_A:.6f} {z_position_A:.6f}"
        for element, x_position_A, y_position_A, z_position_A in atoms
    )
    lines = (f"{len(atoms)} {box_length_A:.1f} {time_ps:.1f}", "", *atom_lines)
    return "\n".join(lines) + "\n"


def _ec_atoms():
    return (
        ("C", 1.0, 1.0, 1.0),
        ("C", 1.5, 1.0, 1.0),
        ("O", 1.0, 1.5, 1.0),
        ("O", 1.5, 1.5, 1.0),
        ("C", 1.25, 1.25, 1.5),
        ("H", 0.5, 1.0, 1.0),
        ("H", 2.0, 1.0, 1.0),
        ("H", 1.0, 2.0, 1.0),
        ("H", 1.5, 2.0, 1.0),
        ("O", 1.25, 1.25, 2.0),
    )


def _emc_atoms():
    return (
        ("C", 10.0, 10.0, 10.0),
        ("O", 10.5, 10.0, 10.0),
        ("C", 11.0, 10.0, 10.0),
        ("O", 11.5, 10.0, 10.0),
        ("C", 12.0, 10.0, 10.0),
        ("C", 12.5, 10.0, 10.0),
        ("H", 9.5, 10.0, 10.0),
        ("H", 10.0, 9.5, 10.0),
        ("H", 10.0, 10.5, 10.0),
        ("O", 13.0, 10.0, 10.0),
        ("H", 11.0, 9.5, 10.0),
        ("H", 11.0, 10.5, 10.0),
        ("H", 12.5, 9.5, 10.0),
        ("H", 12.5, 10.5, 10.0),
        ("H", 13.0, 10.5, 10.0),
    )


def _pf6_atoms():
    return (
        ("P", 5.0, 5.0, 5.0),
        ("F", 4.5, 5.0, 5.0),
        ("F", 5.5, 5.0, 5.0),
        ("F", 5.0, 4.5, 5.0),
        ("F", 5.0, 5.5, 5.0),
        ("F", 5.0, 5.0, 4.5),
        ("F", 5.0, 5.0, 5.5),
    )
