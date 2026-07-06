import numpy as np
import pytest

from constants import T_REF_K
from conductivity.analytical_conductivity_model import (
    PROJECTION_CLASS_SELF_CURRENT_CARRIER,
    OverdampedSmoluchowskiGeneratorInput,
    ProjectedBasisAssignment,
    ProjectedBasisFunctionDefinition,
    compare_recipe_sigma_to_trajectory_projection,
    compute_first_principles_conductivity_from_overdamped_generator,
)
from conductivity.fm_md.atomistic_io import MolecularTrajectory


class ZeroPotentialModel:
    def potential_energy_J(self, positions_m: np.ndarray) -> float:
        return float(np.sum(positions_m * 0.0))

    def forces_N(self, positions_m: np.ndarray) -> np.ndarray:
        return np.zeros_like(positions_m, dtype=float)


class ChargeSignBasisAssigner:
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


def test_sampled_generator_projection_has_non_vacuous_gk_closure_tolerance():
    projection_tolerance_mS_cm = 5.0e-2
    projected_model = compute_first_principles_conductivity_from_overdamped_generator(
        generator_input=OverdampedSmoluchowskiGeneratorInput(
            configuration_space="overdamped_two_ion_periodic_configuration_space",
            equilibrium_measure="boltzmann_measure_for_zero_test_potential",
            reversible_generator="overdamped_smoluchowski_langevin_generator",
            charge_polarization_observable="P=sum_a z_a R_a",
            potential_model=ZeroPotentialModel(),
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
        basis_assigner=ChargeSignBasisAssigner(),
        green_kubo_integration_stop_index=4,
        einstein_helfand_fit_start_index=0,
        einstein_helfand_fit_stop_index=8,
        target_absolute_error_mS_cm=projection_tolerance_mS_cm,
    )
    acceptance_test = projected_model.projection_acceptance_test

    assert acceptance_test.raw_green_kubo_sigma_mS_cm > 0.0
    assert acceptance_test.raw_einstein_helfand_sigma_mS_cm > 0.0
    assert acceptance_test.projected_sigma_mS_cm > 0.0
    assert acceptance_test.maximum_acceptance_gap_mS_cm < projection_tolerance_mS_cm
    assert acceptance_test.passed


def test_recipe_gap_is_separate_from_projected_gk_closure_gap():
    projection_tolerance_mS_cm = 5.0e-2
    projected_model = compute_first_principles_conductivity_from_overdamped_generator(
        generator_input=OverdampedSmoluchowskiGeneratorInput(
            configuration_space="overdamped_two_ion_periodic_configuration_space",
            equilibrium_measure="boltzmann_measure_for_zero_test_potential",
            reversible_generator="overdamped_smoluchowski_langevin_generator",
            charge_polarization_observable="P=sum_a z_a R_a",
            potential_model=ZeroPotentialModel(),
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
        basis_assigner=ChargeSignBasisAssigner(),
        green_kubo_integration_stop_index=4,
        einstein_helfand_fit_start_index=0,
        einstein_helfand_fit_stop_index=8,
        target_absolute_error_mS_cm=projection_tolerance_mS_cm,
    )
    recipe_generated_sigma_mS_cm = (
        projected_model.projection_acceptance_test.projected_sigma_mS_cm + 0.25
    )

    gap_audit = compare_recipe_sigma_to_trajectory_projection(
        projected_model,
        recipe_generated_sigma_mS_cm,
    )

    assert gap_audit.green_kubo_projection_gap_mS_cm < projection_tolerance_mS_cm
    assert gap_audit.einstein_helfand_projection_gap_mS_cm < projection_tolerance_mS_cm
    assert np.isclose(gap_audit.recipe_projection_gap_mS_cm, 0.25)
    assert gap_audit.recipe_generated_sigma_mS_cm == recipe_generated_sigma_mS_cm


def test_first_principles_projection_fails_loudly_when_closure_gap_exceeds_target():
    with pytest.raises(
        ValueError,
        match="first-principles projected conductivity closure failed",
    ):
        compute_first_principles_conductivity_from_overdamped_generator(
            generator_input=OverdampedSmoluchowskiGeneratorInput(
                configuration_space="overdamped_two_ion_periodic_configuration_space",
                equilibrium_measure="boltzmann_measure_for_zero_test_potential",
                reversible_generator="overdamped_smoluchowski_langevin_generator",
                charge_polarization_observable="P=sum_a z_a R_a",
                potential_model=ZeroPotentialModel(),
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
            basis_assigner=ChargeSignBasisAssigner(),
            green_kubo_integration_stop_index=4,
            einstein_helfand_fit_start_index=0,
            einstein_helfand_fit_stop_index=8,
            target_absolute_error_mS_cm=np.finfo(float).tiny,
        )
