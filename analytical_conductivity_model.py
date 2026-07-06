"""Single-file projected analytical conductivity model.

This file owns the full analytical model:

    G_x=(Omega_x, mu_x, L_x, P_x)
    -> sampled unwrapped charge polarization P_t and current J_t
    -> finite projection basis V_n
    -> c_i, K_ij, Q_ij, d_ij, M_ij, D_self_i, A, h
    -> projected Green-Kubo/Mori conductivity and trajectory acceptance test.

The descriptor-closed recipe path below is the fast analytic construction of
the same finite objects. The trajectory-backed path at the top is the complete
sampled microscopic-generator projection used to certify finite-basis closure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Mapping, Protocol, Sequence

import numpy as np

from constants import (
    ANGSTROM3_TO_M3,
    ANGSTROM_TO_M,
    CARTESIAN_COMPONENT_COUNT,
    E_CHARGE,
    EPS_0,
    EINSTEIN_HELFAND_DENOMINATOR_FACTOR,
    F,
    FEMTOSECOND_TO_S,
    K_B,
    M_TO_ANGSTROM,
    N_A,
    R,
    S_M_TO_MS_CM,
)
from conductivity.fm_md.atomistic_io import MolecularTrajectory
from conductivity.fm_md.sigma_from_trajectory import central_difference_velocities

PROJECTION_CLASS_POPULATION_BASIN = "population_basin"
PROJECTION_CLASS_REACTIVE_FLUX_BOUNDARY = "reactive_flux_boundary"
PROJECTION_CLASS_STATE_CHANGING_DISPLACEMENT_EVENT = "state_changing_displacement_event"
PROJECTION_CLASS_SELF_CURRENT_CARRIER = "self_current_carrier"
PROJECTION_CLASS_MORI_MEMORY_BASIS = "mori_memory_basis"
PROJECTION_CLASS_DIAGNOSTIC_ONLY = "diagnostic_only"
SUPPORTED_PROJECTION_CLASSES = (
    PROJECTION_CLASS_POPULATION_BASIN,
    PROJECTION_CLASS_REACTIVE_FLUX_BOUNDARY,
    PROJECTION_CLASS_STATE_CHANGING_DISPLACEMENT_EVENT,
    PROJECTION_CLASS_SELF_CURRENT_CARRIER,
    PROJECTION_CLASS_MORI_MEMORY_BASIS,
    PROJECTION_CLASS_DIAGNOSTIC_ONLY,
)
DC_TRANSPORT_PROJECTION_CLASSES = (
    PROJECTION_CLASS_POPULATION_BASIN,
    PROJECTION_CLASS_REACTIVE_FLUX_BOUNDARY,
    PROJECTION_CLASS_STATE_CHANGING_DISPLACEMENT_EVENT,
    PROJECTION_CLASS_SELF_CURRENT_CARRIER,
)
CARTESIAN_COMPONENT_COUNT_INT = int(CARTESIAN_COMPONENT_COUNT)


@dataclass(frozen=True)
class MicroscopicGeneratorModel:
    """Sampled G_x=(Omega_x, mu_x, L_x, P_x) with charge observables."""

    configuration_space: str
    equilibrium_measure: str
    reversible_generator: str
    charge_polarization_observable: str
    trajectory: MolecularTrajectory

    def __post_init__(self) -> None:
        _validate_nonempty_generator_source(
            self.configuration_space,
            "configuration_space",
        )
        _validate_nonempty_generator_source(
            self.equilibrium_measure,
            "equilibrium_measure",
        )
        _validate_nonempty_generator_source(
            self.reversible_generator,
            "reversible_generator",
        )
        _validate_nonempty_generator_source(
            self.charge_polarization_observable,
            "charge_polarization_observable",
        )
        _validated_microscopic_generator_trajectory(self.trajectory)

    @property
    def volume_m3(self) -> float:
        volume_angstrom3 = float(np.prod(np.asarray(self.trajectory.box, dtype=float)))
        if volume_angstrom3 <= 0.0:
            raise ValueError(
                f"trajectory box volume must be positive, got {volume_angstrom3}"
            )
        return volume_angstrom3 * ANGSTROM3_TO_M3

    @property
    def dt_s(self) -> float:
        dt_s = float(self.trajectory.dt_fs) * FEMTOSECOND_TO_S
        if dt_s <= 0.0:
            raise ValueError(f"trajectory dt_s must be positive, got {dt_s}")
        return dt_s

    @property
    def temperature_K(self) -> float:
        temperature_K = float(self.trajectory.temperature_K)
        if temperature_K <= 0.0:
            raise ValueError(
                f"trajectory temperature_K must be positive, got {temperature_K}"
            )
        return temperature_K

    def charged_molecule_mask(self) -> np.ndarray:
        charge_numbers = np.asarray(self.trajectory.formal_charges, dtype=float)
        return charge_numbers != 0.0

    def charged_center_concentration_mol_m3(self) -> float:
        charged_center_count = int(np.count_nonzero(self.charged_molecule_mask()))
        if charged_center_count == 0:
            raise ValueError("trajectory has no charged molecular centers")
        return charged_center_count / (N_A * self.volume_m3)

    def charge_polarization_charge_number_m(self) -> np.ndarray:
        positions_m = (
            np.asarray(self.trajectory.com_positions, dtype=float) * ANGSTROM_TO_M
        )
        charge_numbers = np.asarray(self.trajectory.formal_charges, dtype=float)
        return np.einsum("m,fmd->fd", charge_numbers, positions_m)

    def charge_current_charge_number_m_s(self) -> np.ndarray:
        velocities_angstrom_fs = central_difference_velocities(
            self.trajectory.com_positions,
            float(self.trajectory.dt_fs),
        )
        velocities_m_s = velocities_angstrom_fs * ANGSTROM_TO_M / FEMTOSECOND_TO_S
        charge_numbers = np.asarray(self.trajectory.formal_charges, dtype=float)
        return np.einsum("m,tmd->td", charge_numbers, velocities_m_s)


class MicroscopicPotentialModel(Protocol):
    """Physical potential U_x(q) and force -grad U_x(q) in SI units."""

    def potential_energy_J(self, positions_m: np.ndarray) -> float: ...

    def forces_N(self, positions_m: np.ndarray) -> np.ndarray: ...


class ProjectedBasisAssigner(Protocol):
    """Measurable finite basis V_n assignment for sampled microscopic states."""

    def assign_basis(
        self, trajectory: MolecularTrajectory
    ) -> "ProjectedBasisAssignment": ...


@dataclass(frozen=True)
class OverdampedSmoluchowskiGeneratorInput:
    """Concrete overdamped L_x sampler for an explicit U_x and mobility field."""

    configuration_space: str
    equilibrium_measure: str
    reversible_generator: str
    charge_polarization_observable: str
    potential_model: MicroscopicPotentialModel
    initial_positions_m: np.ndarray
    molecule_species_indices: np.ndarray
    formal_charge_numbers: np.ndarray
    box_lengths_m: np.ndarray
    diffusion_coefficients_m2_s: np.ndarray
    temperature_K: float
    dt_s: float
    frame_count: int
    rng_seed: int


def sample_overdamped_smoluchowski_microscopic_generator(
    generator_input: OverdampedSmoluchowskiGeneratorInput,
) -> MicroscopicGeneratorModel:
    """Sample overdamped Smoluchowski dynamics from explicit U_x and D_x."""

    (
        initial_positions_m,
        molecule_species_indices,
        formal_charge_numbers,
        box_lengths_m,
        diffusion_coefficients_m2_s,
    ) = _validated_overdamped_generator_input_arrays(generator_input)
    temperature_K = _positive_float(
        generator_input.temperature_K,
        "generator_input.temperature_K",
    )
    dt_s = _positive_float(generator_input.dt_s, "generator_input.dt_s")
    frame_count = _positive_int(
        generator_input.frame_count, "generator_input.frame_count"
    )
    rng_seed = _nonnegative_int(generator_input.rng_seed, "generator_input.rng_seed")
    _validate_nonempty_generator_source(
        generator_input.configuration_space,
        "configuration_space",
    )
    _validate_nonempty_generator_source(
        generator_input.equilibrium_measure,
        "equilibrium_measure",
    )
    _validate_nonempty_generator_source(
        generator_input.reversible_generator,
        "reversible_generator",
    )
    _validate_nonempty_generator_source(
        generator_input.charge_polarization_observable,
        "charge_polarization_observable",
    )

    rng = np.random.default_rng(rng_seed)
    positions_by_frame_m = np.empty(
        (frame_count, initial_positions_m.shape[0], CARTESIAN_COMPONENT_COUNT_INT),
        dtype=float,
    )
    current_positions_m = initial_positions_m.copy()
    initial_potential_energy_J = float(
        generator_input.potential_model.potential_energy_J(current_positions_m)
    )
    if not math.isfinite(initial_potential_energy_J):
        raise ValueError(
            "potential_model.potential_energy_J returned non-finite energy"
        )
    positions_by_frame_m[0] = current_positions_m
    thermal_energy_J = K_B * temperature_K
    for frame_index in range(1, frame_count):
        forces_N = np.asarray(
            generator_input.potential_model.forces_N(current_positions_m),
            dtype=float,
        )
        if forces_N.shape != current_positions_m.shape:
            raise ValueError(
                "potential_model.forces_N returned shape "
                f"{forces_N.shape}, expected {current_positions_m.shape}"
            )
        if not np.all(np.isfinite(forces_N)):
            raise ValueError("potential_model.forces_N returned non-finite forces")
        drift_m = (
            diffusion_coefficients_m2_s[:, np.newaxis]
            * forces_N
            * dt_s
            / thermal_energy_J
        )
        random_displacement_m = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 * diffusion_coefficients_m2_s * dt_s)[:, np.newaxis],
            size=(initial_positions_m.shape[0], CARTESIAN_COMPONENT_COUNT_INT),
        )
        current_positions_m = current_positions_m + drift_m + random_displacement_m
        potential_energy_J = float(
            generator_input.potential_model.potential_energy_J(current_positions_m)
        )
        if not math.isfinite(potential_energy_J):
            raise ValueError(
                "potential_model.potential_energy_J returned non-finite energy"
            )
        positions_by_frame_m[frame_index] = current_positions_m
    trajectory = MolecularTrajectory(
        com_positions=positions_by_frame_m * M_TO_ANGSTROM,
        molecule_species=molecule_species_indices.astype(np.int32),
        formal_charges=formal_charge_numbers.astype(float),
        box=box_lengths_m * M_TO_ANGSTROM,
        dt_fs=dt_s / FEMTOSECOND_TO_S,
        n_frames=frame_count,
        n_molecules=int(initial_positions_m.shape[0]),
        temperature_K=temperature_K,
    )
    return MicroscopicGeneratorModel(
        configuration_space=generator_input.configuration_space,
        equilibrium_measure=generator_input.equilibrium_measure,
        reversible_generator=generator_input.reversible_generator,
        charge_polarization_observable=generator_input.charge_polarization_observable,
        trajectory=trajectory,
    )


def compute_first_principles_conductivity_from_overdamped_generator(
    generator_input: OverdampedSmoluchowskiGeneratorInput,
    basis_assigner: ProjectedBasisAssigner,
    green_kubo_integration_stop_index: int,
    einstein_helfand_fit_start_index: int,
    einstein_helfand_fit_stop_index: int,
    target_absolute_error_mS_cm: float,
) -> FirstPrinciplesProjectedConductivityModel:
    """Run U_x,L_x,P_x sampling and require projected GK/EH closure."""

    generator_model = sample_overdamped_smoluchowski_microscopic_generator(
        generator_input,
    )
    basis_assignment = basis_assigner.assign_basis(generator_model.trajectory)
    return compute_verified_first_principles_projected_conductivity(
        generator_model=generator_model,
        basis_assignment=basis_assignment,
        green_kubo_integration_stop_index=green_kubo_integration_stop_index,
        einstein_helfand_fit_start_index=einstein_helfand_fit_start_index,
        einstein_helfand_fit_stop_index=einstein_helfand_fit_stop_index,
        target_absolute_error_mS_cm=target_absolute_error_mS_cm,
    )


@dataclass(frozen=True)
class ProjectedBasisFunctionDefinition:
    state_label: str
    projection_class: str


@dataclass(frozen=True)
class ProjectedBasisAssignment:
    """Finite basis assignment for every trajectory frame and molecule."""

    basis_functions: tuple[ProjectedBasisFunctionDefinition, ...]
    state_index_by_frame_and_molecule: np.ndarray


@dataclass(frozen=True)
class ProjectedMoriOperatorObjects:
    energy_matrix: np.ndarray
    current_coupling_matrix: np.ndarray
    beta_over_volume: float
    quadratic_form_by_axis: tuple[float, float, float]
    sigma_mS_cm: float


@dataclass(frozen=True)
class SampledProjectionProcessDiagnostics:
    original_state_count: int
    visited_state_count: int
    occupancy_observation_count: int
    step_count: int
    transition_sample_count: int
    self_displacement_sample_count: int
    generated_event_count: int
    minimum_state_concentration_mol_m3: float
    maximum_state_concentration_mol_m3: float
    total_transport_concentration_mol_m3: float
    trajectory_time_s: float


@dataclass(frozen=True)
class ProjectedGeneratorReactiveFlux:
    from_state_label: str
    to_state_label: str
    symmetric_flux_mol_m3_s: float
    forward_rate_s_inv: float
    reverse_rate_s_inv: float


@dataclass(frozen=True)
class ProjectedGeneratorConditionalMoment:
    from_state_label: str
    to_state_label: str
    sample_count: int
    mean_charge_displacement_m: tuple[float, float, float]
    second_moment_m2: tuple[tuple[float, float, float], ...]
    covariance_m2: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ProjectedGeneratorSelfCurrentTensor:
    state_label: str
    sample_count: int
    concentration_mol_m3: float
    diffusion_tensor_m2_s: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ProjectedGeneratorPrimitiveSet:
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: Mapping[str, float]
    state_occupancy_fractions: Mapping[str, float]
    reactive_fluxes: tuple[ProjectedGeneratorReactiveFlux, ...]
    conditional_displacement_moments: tuple[ProjectedGeneratorConditionalMoment, ...]
    self_current_tensors: tuple[ProjectedGeneratorSelfCurrentTensor, ...]
    markov_input: MarkovAdditiveConductivityInput
    markov_conductivity_result: MarkovAdditiveConductivityResult
    diagnostics: SampledProjectionProcessDiagnostics


@dataclass(frozen=True)
class ProjectedPrimitiveSet:
    """Production primitives derived by projecting one microscopic generator."""

    state_labels: tuple[str, ...]
    restricted_equilibrium_populations_c_i_mol_m3: np.ndarray
    symmetric_reactive_fluxes_K_ij_mol_m3_s: np.ndarray
    reversible_generator_Q_ij_s_inv: np.ndarray
    conditional_displacement_first_moments_d_ij_m: np.ndarray
    conditional_displacement_second_moments_M_ij_m2: np.ndarray
    self_current_diffusion_tensors_D_self_i_m2_s: np.ndarray
    mori_memory_energy_matrix_A: np.ndarray
    mori_current_coupling_matrix_h: np.ndarray
    markov_input: MarkovAdditiveConductivityInput
    markov_conductivity_result: MarkovAdditiveConductivityResult


@dataclass(frozen=True)
class SampledGeneratorProjectionInput:
    state_labels: tuple[str, ...]
    occupancy_state_index_by_observation: np.ndarray
    from_state_index_by_step: np.ndarray
    to_state_index_by_step: np.ndarray
    charge_displacement_by_step_m: np.ndarray
    dt_s: float
    total_transport_concentration_mol_m3: float
    temperature_K: float
    displacement_zero_tolerance_m: float = 0.0


@dataclass(frozen=True)
class ProjectedMicroscopicGeneratorModel:
    """Projected generator primitives and the basis assignment that produced them."""

    basis_assignment: ProjectedBasisAssignment
    primitive_set: ProjectedGeneratorPrimitiveSet
    projected_primitives: ProjectedPrimitiveSet
    mori_operator: ProjectedMoriOperatorObjects


@dataclass(frozen=True)
class ProjectedGeneratorBuilder:
    """Build finite conductivity primitives from G_x, mu_x, L_x, P_x samples."""

    generator_model: MicroscopicGeneratorModel
    basis_assigner: ProjectedBasisAssigner

    def build_projected_generator(self) -> ProjectedMicroscopicGeneratorModel:
        basis_assignment = self.basis_assigner.assign_basis(
            self.generator_model.trajectory,
        )
        return project_microscopic_generator(
            generator_model=self.generator_model,
            basis_assignment=basis_assignment,
        )


@dataclass(frozen=True)
class DirectGreenKuboConductivityEstimate:
    sigma_mS_cm: float
    integral_charge_number_m2_s: float
    integration_time_s: float
    sample_count: int


@dataclass(frozen=True)
class EinsteinHelfandConductivityEstimate:
    sigma_mS_cm: float
    slope_charge_number_m2_s: float
    fit_start_time_s: float
    fit_stop_time_s: float
    sample_count: int


@dataclass(frozen=True)
class FirstPrinciplesProjectionAcceptanceTest:
    raw_green_kubo_sigma_mS_cm: float
    raw_einstein_helfand_sigma_mS_cm: float
    projected_sigma_mS_cm: float
    green_kubo_projected_gap_mS_cm: float
    einstein_helfand_projected_gap_mS_cm: float
    green_kubo_einstein_helfand_gap_mS_cm: float
    maximum_acceptance_gap_mS_cm: float
    target_absolute_error_mS_cm: float
    passed: bool


@dataclass(frozen=True)
class FirstPrinciplesProjectedConductivityModel:
    generator_model: MicroscopicGeneratorModel
    projected_generator: ProjectedMicroscopicGeneratorModel
    raw_green_kubo_estimate: DirectGreenKuboConductivityEstimate
    raw_einstein_helfand_estimate: EinsteinHelfandConductivityEstimate
    projection_acceptance_test: FirstPrinciplesProjectionAcceptanceTest


@dataclass(frozen=True)
class RecipeProjectionGapAudit:
    raw_green_kubo_sigma_mS_cm: float
    raw_einstein_helfand_sigma_mS_cm: float
    projected_finite_sigma_mS_cm: float
    recipe_generated_sigma_mS_cm: float
    green_kubo_projection_gap_mS_cm: float
    einstein_helfand_projection_gap_mS_cm: float
    recipe_projection_gap_mS_cm: float
    recipe_green_kubo_gap_mS_cm: float
    recipe_einstein_helfand_gap_mS_cm: float


def compare_recipe_sigma_to_trajectory_projection(
    first_principles_model: FirstPrinciplesProjectedConductivityModel,
    recipe_generated_sigma_mS_cm: float,
) -> RecipeProjectionGapAudit:
    """Report trajectory projection gap separately from recipe primitive gap."""

    recipe_sigma_mS_cm = _nonnegative_float(
        recipe_generated_sigma_mS_cm,
        "recipe_generated_sigma_mS_cm",
    )
    raw_green_kubo_sigma_mS_cm = (
        first_principles_model.raw_green_kubo_estimate.sigma_mS_cm
    )
    raw_einstein_helfand_sigma_mS_cm = (
        first_principles_model.raw_einstein_helfand_estimate.sigma_mS_cm
    )
    projected_sigma_mS_cm = (
        first_principles_model.projection_acceptance_test.projected_sigma_mS_cm
    )
    return RecipeProjectionGapAudit(
        raw_green_kubo_sigma_mS_cm=raw_green_kubo_sigma_mS_cm,
        raw_einstein_helfand_sigma_mS_cm=raw_einstein_helfand_sigma_mS_cm,
        projected_finite_sigma_mS_cm=projected_sigma_mS_cm,
        recipe_generated_sigma_mS_cm=recipe_sigma_mS_cm,
        green_kubo_projection_gap_mS_cm=abs(
            raw_green_kubo_sigma_mS_cm - projected_sigma_mS_cm
        ),
        einstein_helfand_projection_gap_mS_cm=abs(
            raw_einstein_helfand_sigma_mS_cm - projected_sigma_mS_cm
        ),
        recipe_projection_gap_mS_cm=abs(recipe_sigma_mS_cm - projected_sigma_mS_cm),
        recipe_green_kubo_gap_mS_cm=abs(
            recipe_sigma_mS_cm - raw_green_kubo_sigma_mS_cm
        ),
        recipe_einstein_helfand_gap_mS_cm=abs(
            recipe_sigma_mS_cm - raw_einstein_helfand_sigma_mS_cm
        ),
    )


def project_microscopic_generator(
    generator_model: MicroscopicGeneratorModel,
    basis_assignment: ProjectedBasisAssignment,
) -> ProjectedMicroscopicGeneratorModel:
    """Project a sampled generator into populations, fluxes, moments, and self-current."""

    basis_functions = _validated_projected_basis_functions(
        basis_assignment.basis_functions
    )
    state_index_by_frame_and_molecule = _validated_projected_basis_indices(
        basis_assignment.state_index_by_frame_and_molecule,
        len(basis_functions),
        generator_model.trajectory.com_positions.shape,
    )
    charged_molecule_indices = np.flatnonzero(generator_model.charged_molecule_mask())
    if charged_molecule_indices.size == 0:
        raise ValueError(
            "project_microscopic_generator requires at least one charged molecule"
        )
    contributing_state_mask = _dc_transport_state_mask(basis_functions)
    self_current_state_mask = _self_current_state_mask(basis_functions)
    charged_state_index_by_frame_and_molecule = state_index_by_frame_and_molecule[
        :,
        charged_molecule_indices,
    ]
    contributing_observation_mask = contributing_state_mask[
        charged_state_index_by_frame_and_molecule
    ]
    occupancy_state_index_by_observation = charged_state_index_by_frame_and_molecule[
        contributing_observation_mask
    ]
    if occupancy_state_index_by_observation.size == 0:
        raise ValueError(
            "project_microscopic_generator found no DC transport observations"
        )

    from_state_index_by_step_matrix = state_index_by_frame_and_molecule[
        :-1,
        charged_molecule_indices,
    ]
    to_state_index_by_step_matrix = state_index_by_frame_and_molecule[
        1:,
        charged_molecule_indices,
    ]
    state_change_step_mask = (
        contributing_state_mask[from_state_index_by_step_matrix]
        & contributing_state_mask[to_state_index_by_step_matrix]
        & (from_state_index_by_step_matrix != to_state_index_by_step_matrix)
    )
    self_current_step_mask = self_current_state_mask[
        from_state_index_by_step_matrix
    ] & (from_state_index_by_step_matrix == to_state_index_by_step_matrix)
    contributing_step_mask = state_change_step_mask | self_current_step_mask
    if not bool(np.any(contributing_step_mask)):
        raise ValueError("project_microscopic_generator found no DC transport steps")
    charge_displacement_by_step_matrix_m = (
        _charge_displacement_matrix_by_generator_step_m(
            generator_model,
            charged_molecule_indices,
        )
    )
    contributing_observation_fraction = float(
        occupancy_state_index_by_observation.size
        / charged_state_index_by_frame_and_molecule.size
    )

    sample_input = SampledGeneratorProjectionInput(
        state_labels=tuple(
            basis_function.state_label for basis_function in basis_functions
        ),
        occupancy_state_index_by_observation=occupancy_state_index_by_observation,
        from_state_index_by_step=from_state_index_by_step_matrix[
            contributing_step_mask
        ],
        to_state_index_by_step=to_state_index_by_step_matrix[contributing_step_mask],
        charge_displacement_by_step_m=charge_displacement_by_step_matrix_m[
            contributing_step_mask
        ],
        dt_s=generator_model.dt_s,
        total_transport_concentration_mol_m3=(
            generator_model.charged_center_concentration_mol_m3()
            * contributing_observation_fraction
        ),
        temperature_K=generator_model.temperature_K,
    )
    primitive_set = _project_sampled_generator_input_to_primitives(sample_input)
    corrector_mori_input = primitive_set.markov_conductivity_result.corrector_mori_input
    corrector_mori_result = (
        primitive_set.markov_conductivity_result.corrector_mori_result
    )
    mori_operator = ProjectedMoriOperatorObjects(
        energy_matrix=corrector_mori_result.effective_energy_matrix.copy(),
        current_coupling_matrix=corrector_mori_input.current_coupling_matrix.copy(),
        beta_over_volume=float(corrector_mori_input.beta_over_volume),
        quadratic_form_by_axis=corrector_mori_result.quadratic_form_by_axis,
        sigma_mS_cm=float(corrector_mori_result.sigma_mS_cm),
    )
    return ProjectedMicroscopicGeneratorModel(
        basis_assignment=ProjectedBasisAssignment(
            basis_functions=basis_functions,
            state_index_by_frame_and_molecule=state_index_by_frame_and_molecule,
        ),
        primitive_set=primitive_set,
        projected_primitives=_projected_primitive_set_from_generator_projection(
            primitive_set,
            mori_operator,
        ),
        mori_operator=mori_operator,
    )


def estimate_direct_green_kubo_conductivity(
    charge_current_charge_number_m_s: np.ndarray,
    dt_s: float,
    volume_m3: float,
    temperature_K: float,
    integration_stop_index: int,
) -> DirectGreenKuboConductivityEstimate:
    """Integrate the finite-time current autocorrelation directly."""

    current = _validated_vector_timeseries(
        charge_current_charge_number_m_s,
        "charge_current_charge_number_m_s",
    )
    dt_s = _positive_float(dt_s, "dt_s")
    volume_m3 = _positive_float(volume_m3, "volume_m3")
    temperature_K = _positive_float(temperature_K, "temperature_K")
    if integration_stop_index <= 0 or integration_stop_index > current.shape[0]:
        raise ValueError(
            "integration_stop_index must be in [1, sample_count], got "
            f"{integration_stop_index} for sample_count={current.shape[0]}"
        )

    autocorrelation = _current_autocorrelation(current)
    selected_autocorrelation = autocorrelation[:integration_stop_index]
    integral_charge_number_m2_s = float(np.trapezoid(selected_autocorrelation, dx=dt_s))
    sigma_S_m = (
        E_CHARGE**2
        * integral_charge_number_m2_s
        / (CARTESIAN_COMPONENT_COUNT * volume_m3 * K_B * temperature_K)
    )
    return DirectGreenKuboConductivityEstimate(
        sigma_mS_cm=float(sigma_S_m * S_M_TO_MS_CM),
        integral_charge_number_m2_s=integral_charge_number_m2_s,
        integration_time_s=float((integration_stop_index - 1) * dt_s),
        sample_count=int(current.shape[0]),
    )


def estimate_einstein_helfand_conductivity(
    charge_polarization_charge_number_m: np.ndarray,
    dt_s: float,
    volume_m3: float,
    temperature_K: float,
    fit_start_index: int,
    fit_stop_index: int,
) -> EinsteinHelfandConductivityEstimate:
    """Estimate conductivity from the finite-time Einstein-Helfand slope."""

    polarization = _validated_vector_timeseries(
        charge_polarization_charge_number_m,
        "charge_polarization_charge_number_m",
    )
    dt_s = _positive_float(dt_s, "dt_s")
    volume_m3 = _positive_float(volume_m3, "volume_m3")
    temperature_K = _positive_float(temperature_K, "temperature_K")
    if fit_start_index < 0 or fit_stop_index > polarization.shape[0]:
        raise ValueError(
            "fit index range must be inside polarization samples: "
            f"start={fit_start_index}, stop={fit_stop_index}, "
            f"sample_count={polarization.shape[0]}"
        )
    if fit_stop_index - fit_start_index < 2:
        raise ValueError(
            "Einstein-Helfand fit requires at least two samples, got "
            f"{fit_stop_index - fit_start_index}"
        )

    charge_displacement_from_origin_m = polarization - polarization[0]
    squared_charge_displacement_m2 = np.sum(
        charge_displacement_from_origin_m**2, axis=1
    )
    sample_times_s = np.arange(polarization.shape[0], dtype=float) * dt_s
    selected_times_s = sample_times_s[fit_start_index:fit_stop_index]
    selected_squared_charge_displacement_m2 = squared_charge_displacement_m2[
        fit_start_index:fit_stop_index
    ]
    slope_charge_number_m2_s = float(
        np.polyfit(
            selected_times_s,
            selected_squared_charge_displacement_m2,
            deg=1,
        )[0]
    )
    sigma_S_m = (
        E_CHARGE**2
        * slope_charge_number_m2_s
        / (EINSTEIN_HELFAND_DENOMINATOR_FACTOR * volume_m3 * K_B * temperature_K)
    )
    return EinsteinHelfandConductivityEstimate(
        sigma_mS_cm=float(sigma_S_m * S_M_TO_MS_CM),
        slope_charge_number_m2_s=slope_charge_number_m2_s,
        fit_start_time_s=float(selected_times_s[0]),
        fit_stop_time_s=float(selected_times_s[-1]),
        sample_count=int(polarization.shape[0]),
    )


def compute_first_principles_projected_conductivity(
    generator_model: MicroscopicGeneratorModel,
    basis_assignment: ProjectedBasisAssignment,
    green_kubo_integration_stop_index: int,
    einstein_helfand_fit_start_index: int,
    einstein_helfand_fit_stop_index: int,
    target_absolute_error_mS_cm: float,
) -> FirstPrinciplesProjectedConductivityModel:
    """Evaluate raw trajectory estimators, projected primitives, and error bounds."""

    target_absolute_error_mS_cm = _positive_float(
        target_absolute_error_mS_cm,
        "target_absolute_error_mS_cm",
    )
    raw_green_kubo_estimate = estimate_direct_green_kubo_conductivity(
        charge_current_charge_number_m_s=(
            generator_model.charge_current_charge_number_m_s()
        ),
        dt_s=generator_model.dt_s,
        volume_m3=generator_model.volume_m3,
        temperature_K=generator_model.temperature_K,
        integration_stop_index=green_kubo_integration_stop_index,
    )
    raw_einstein_helfand_estimate = estimate_einstein_helfand_conductivity(
        charge_polarization_charge_number_m=(
            generator_model.charge_polarization_charge_number_m()
        ),
        dt_s=generator_model.dt_s,
        volume_m3=generator_model.volume_m3,
        temperature_K=generator_model.temperature_K,
        fit_start_index=einstein_helfand_fit_start_index,
        fit_stop_index=einstein_helfand_fit_stop_index,
    )
    projected_generator = project_microscopic_generator(
        generator_model,
        basis_assignment,
    )
    projected_sigma_mS_cm = (
        projected_generator.primitive_set.markov_conductivity_result.sigma_mS_cm
    )
    green_kubo_projected_gap_mS_cm = abs(
        raw_green_kubo_estimate.sigma_mS_cm - projected_sigma_mS_cm
    )
    einstein_helfand_projected_gap_mS_cm = abs(
        raw_einstein_helfand_estimate.sigma_mS_cm - projected_sigma_mS_cm
    )
    green_kubo_einstein_helfand_gap_mS_cm = abs(
        raw_green_kubo_estimate.sigma_mS_cm - raw_einstein_helfand_estimate.sigma_mS_cm
    )
    maximum_acceptance_gap_mS_cm = max(
        green_kubo_projected_gap_mS_cm,
        einstein_helfand_projected_gap_mS_cm,
        green_kubo_einstein_helfand_gap_mS_cm,
    )
    projection_acceptance_test = FirstPrinciplesProjectionAcceptanceTest(
        raw_green_kubo_sigma_mS_cm=raw_green_kubo_estimate.sigma_mS_cm,
        raw_einstein_helfand_sigma_mS_cm=raw_einstein_helfand_estimate.sigma_mS_cm,
        projected_sigma_mS_cm=projected_sigma_mS_cm,
        green_kubo_projected_gap_mS_cm=green_kubo_projected_gap_mS_cm,
        einstein_helfand_projected_gap_mS_cm=einstein_helfand_projected_gap_mS_cm,
        green_kubo_einstein_helfand_gap_mS_cm=(green_kubo_einstein_helfand_gap_mS_cm),
        maximum_acceptance_gap_mS_cm=maximum_acceptance_gap_mS_cm,
        target_absolute_error_mS_cm=target_absolute_error_mS_cm,
        passed=maximum_acceptance_gap_mS_cm <= target_absolute_error_mS_cm,
    )
    return FirstPrinciplesProjectedConductivityModel(
        generator_model=generator_model,
        projected_generator=projected_generator,
        raw_green_kubo_estimate=raw_green_kubo_estimate,
        raw_einstein_helfand_estimate=raw_einstein_helfand_estimate,
        projection_acceptance_test=projection_acceptance_test,
    )


def compute_verified_first_principles_projected_conductivity(
    generator_model: MicroscopicGeneratorModel,
    basis_assignment: ProjectedBasisAssignment,
    green_kubo_integration_stop_index: int,
    einstein_helfand_fit_start_index: int,
    einstein_helfand_fit_stop_index: int,
    target_absolute_error_mS_cm: float,
) -> FirstPrinciplesProjectedConductivityModel:
    """Evaluate a sampled projection and fail unless finite-basis closure passes."""

    projected_model = compute_first_principles_projected_conductivity(
        generator_model=generator_model,
        basis_assignment=basis_assignment,
        green_kubo_integration_stop_index=green_kubo_integration_stop_index,
        einstein_helfand_fit_start_index=einstein_helfand_fit_start_index,
        einstein_helfand_fit_stop_index=einstein_helfand_fit_stop_index,
        target_absolute_error_mS_cm=target_absolute_error_mS_cm,
    )
    _require_first_principles_projection_acceptance(
        projected_model.projection_acceptance_test,
    )
    return projected_model


def _require_first_principles_projection_acceptance(
    projection_acceptance_test: FirstPrinciplesProjectionAcceptanceTest,
) -> None:
    if projection_acceptance_test.passed:
        return
    raise ValueError(
        "first-principles projected conductivity closure failed: "
        "maximum_acceptance_gap_mS_cm="
        f"{projection_acceptance_test.maximum_acceptance_gap_mS_cm} exceeds "
        "target_absolute_error_mS_cm="
        f"{projection_acceptance_test.target_absolute_error_mS_cm}"
    )


def _project_sampled_generator_input_to_primitives(
    sample_input: SampledGeneratorProjectionInput,
) -> ProjectedGeneratorPrimitiveSet:
    state_labels = _projection_state_labels(sample_input.state_labels)
    occupancy_state_indices = _projection_sample_indices(
        sample_input.occupancy_state_index_by_observation,
        len(state_labels),
        "occupancy_state_index_by_observation",
    )
    from_state_indices = _projection_sample_indices(
        sample_input.from_state_index_by_step,
        len(state_labels),
        "from_state_index_by_step",
    )
    to_state_indices = _projection_sample_indices(
        sample_input.to_state_index_by_step,
        len(state_labels),
        "to_state_index_by_step",
    )
    if from_state_indices.shape != to_state_indices.shape:
        raise ValueError(
            "from_state_index_by_step and to_state_index_by_step must have "
            "the same shape"
        )
    charge_displacements_m = _projection_charge_displacements(
        sample_input.charge_displacement_by_step_m,
        int(from_state_indices.shape[0]),
    )
    dt_s = _positive_float(sample_input.dt_s, "sample_input.dt_s")
    total_concentration_mol_m3 = _positive_float(
        sample_input.total_transport_concentration_mol_m3,
        "sample_input.total_transport_concentration_mol_m3",
    )
    temperature_K = _positive_float(
        sample_input.temperature_K,
        "sample_input.temperature_K",
    )
    displacement_zero_tolerance_m = _nonnegative_float(
        sample_input.displacement_zero_tolerance_m,
        "sample_input.displacement_zero_tolerance_m",
    )

    all_state_observations = np.concatenate(
        (occupancy_state_indices, from_state_indices, to_state_indices),
    )
    (
        active_state_index_by_original_index,
        active_state_labels,
        remapped_observations,
    ) = _projection_remap_visited_states(state_labels, all_state_observations)
    occupancy_observation_count = int(occupancy_state_indices.shape[0])
    step_count = int(from_state_indices.shape[0])
    remapped_occupancy_indices = remapped_observations[:occupancy_observation_count]
    remapped_from_state_indices = remapped_observations[
        occupancy_observation_count : occupancy_observation_count + step_count
    ]
    remapped_to_state_indices = remapped_observations[
        occupancy_observation_count + step_count :
    ]
    active_state_concentrations_mol_m3 = (
        _projection_state_concentrations_from_occupancy(
            remapped_occupancy_indices,
            len(active_state_labels),
            total_concentration_mol_m3,
        )
    )

    pair_samples_by_state_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    self_samples_by_state: dict[int, list[np.ndarray]] = {}
    transition_sample_count = 0
    self_displacement_sample_count = 0
    for step_index, charge_displacement_m in enumerate(charge_displacements_m):
        from_state_index = int(remapped_from_state_indices[step_index])
        to_state_index = int(remapped_to_state_indices[step_index])
        if from_state_index == to_state_index:
            if (
                float(np.linalg.norm(charge_displacement_m))
                > displacement_zero_tolerance_m
            ):
                if from_state_index not in self_samples_by_state:
                    self_samples_by_state[from_state_index] = []
                self_samples_by_state[from_state_index].append(charge_displacement_m)
                self_displacement_sample_count += 1
            continue
        lower_state_index = min(from_state_index, to_state_index)
        upper_state_index = max(from_state_index, to_state_index)
        canonical_displacement_m = (
            charge_displacement_m
            if from_state_index == lower_state_index
            else -charge_displacement_m
        )
        state_pair = (lower_state_index, upper_state_index)
        if state_pair not in pair_samples_by_state_pair:
            pair_samples_by_state_pair[state_pair] = []
        pair_samples_by_state_pair[state_pair].append(canonical_displacement_m)
        transition_sample_count += 1

    event_flux_mol_m3_s = total_concentration_mol_m3 / (2.0 * float(step_count) * dt_s)
    markov_events = _projection_markov_events_from_samples(
        pair_samples_by_state_pair,
        self_samples_by_state,
        active_state_labels,
        active_state_concentrations_mol_m3,
        event_flux_mol_m3_s,
    )
    if not markov_events:
        raise ValueError("sampled generator projection produced no events")
    markov_input = MarkovAdditiveConductivityInput(
        state_labels=active_state_labels,
        state_concentrations_mol_m3=active_state_concentrations_mol_m3,
        events=markov_events,
        temperature_K=temperature_K,
    )
    markov_result = compute_markov_additive_green_kubo_conductivity(markov_input)
    state_concentrations_mol_m3 = {state_label: 0.0 for state_label in state_labels}
    for (
        original_state_index,
        active_state_index,
    ) in active_state_index_by_original_index.items():
        state_concentrations_mol_m3[state_labels[original_state_index]] = float(
            active_state_concentrations_mol_m3[active_state_index]
        )
    state_occupancy_fractions = _projection_state_occupancy_fractions(
        state_labels,
        occupancy_state_indices,
    )
    diagnostics = SampledProjectionProcessDiagnostics(
        original_state_count=len(state_labels),
        visited_state_count=len(active_state_labels),
        occupancy_observation_count=occupancy_observation_count,
        step_count=step_count,
        transition_sample_count=transition_sample_count,
        self_displacement_sample_count=self_displacement_sample_count,
        generated_event_count=len(markov_events),
        minimum_state_concentration_mol_m3=float(
            np.min(active_state_concentrations_mol_m3)
        ),
        maximum_state_concentration_mol_m3=float(
            np.max(active_state_concentrations_mol_m3)
        ),
        total_transport_concentration_mol_m3=total_concentration_mol_m3,
        trajectory_time_s=float(step_count * dt_s),
    )
    return ProjectedGeneratorPrimitiveSet(
        state_labels=active_state_labels,
        state_concentrations_mol_m3=state_concentrations_mol_m3,
        state_occupancy_fractions=state_occupancy_fractions,
        reactive_fluxes=_projection_reactive_fluxes(markov_input),
        conditional_displacement_moments=(
            _projection_conditional_displacement_moments(markov_input)
        ),
        self_current_tensors=_projection_self_current_tensors(
            active_state_labels,
            active_state_concentrations_mol_m3,
            self_samples_by_state,
            dt_s,
        ),
        markov_input=markov_input,
        markov_conductivity_result=markov_result,
        diagnostics=diagnostics,
    )


def _projected_primitive_set_from_generator_projection(
    generator_primitive_set: ProjectedGeneratorPrimitiveSet,
    mori_operator: ProjectedMoriOperatorObjects,
) -> ProjectedPrimitiveSet:
    """Expose the projection in theorem primitive notation."""

    markov_input = generator_primitive_set.markov_input
    state_labels = tuple(markov_input.state_labels)
    state_count = len(state_labels)
    state_label_to_index = {
        state_label: state_index for state_index, state_label in enumerate(state_labels)
    }
    state_concentrations = np.asarray(
        markov_input.state_concentrations_mol_m3,
        dtype=float,
    )
    if state_concentrations.shape != (state_count,):
        raise ValueError("projected primitive c_i shape mismatch")
    if np.any(state_concentrations <= 0.0):
        raise ValueError("projected primitive c_i must be positive")

    generator_matrix = np.asarray(
        generator_primitive_set.markov_conductivity_result.generator_s_inv,
        dtype=float,
    )
    if generator_matrix.shape != (state_count, state_count):
        raise ValueError("projected primitive Q_ij shape mismatch")

    symmetric_fluxes = np.zeros((state_count, state_count), dtype=float)
    for reactive_flux in generator_primitive_set.reactive_fluxes:
        from_state_index = _state_index_for_projected_primitive_label(
            reactive_flux.from_state_label,
            state_label_to_index,
        )
        to_state_index = _state_index_for_projected_primitive_label(
            reactive_flux.to_state_label,
            state_label_to_index,
        )
        symmetric_flux = _nonnegative_float(
            reactive_flux.symmetric_flux_mol_m3_s,
            "projected_primitive.K_ij",
        )
        symmetric_fluxes[from_state_index, to_state_index] = symmetric_flux
        symmetric_fluxes[to_state_index, from_state_index] = symmetric_flux

    first_moments = np.zeros(
        (state_count, state_count, CARTESIAN_COMPONENT_COUNT_INT),
        dtype=float,
    )
    second_moments = np.zeros(
        (
            state_count,
            state_count,
            CARTESIAN_COMPONENT_COUNT_INT,
            CARTESIAN_COMPONENT_COUNT_INT,
        ),
        dtype=float,
    )
    for conditional_moment in generator_primitive_set.conditional_displacement_moments:
        from_state_index = _state_index_for_projected_primitive_label(
            conditional_moment.from_state_label,
            state_label_to_index,
        )
        to_state_index = _state_index_for_projected_primitive_label(
            conditional_moment.to_state_label,
            state_label_to_index,
        )
        first_moments[from_state_index, to_state_index, :] = np.asarray(
            conditional_moment.mean_charge_displacement_m,
            dtype=float,
        )
        second_moments[from_state_index, to_state_index, :, :] = np.asarray(
            conditional_moment.second_moment_m2,
            dtype=float,
        )

    self_current_tensors = np.zeros(
        (
            state_count,
            CARTESIAN_COMPONENT_COUNT_INT,
            CARTESIAN_COMPONENT_COUNT_INT,
        ),
        dtype=float,
    )
    for self_current_tensor in generator_primitive_set.self_current_tensors:
        state_index = _state_index_for_projected_primitive_label(
            self_current_tensor.state_label,
            state_label_to_index,
        )
        self_current_tensors[state_index, :, :] = np.asarray(
            self_current_tensor.diffusion_tensor_m2_s,
            dtype=float,
        )

    mori_energy_matrix = np.asarray(mori_operator.energy_matrix, dtype=float)
    mori_current_coupling_matrix = np.asarray(
        mori_operator.current_coupling_matrix,
        dtype=float,
    )
    if mori_energy_matrix.ndim != 2 or (
        mori_energy_matrix.shape[0] != mori_energy_matrix.shape[1]
    ):
        raise ValueError("projected primitive A must be square")
    if mori_current_coupling_matrix.shape != (
        CARTESIAN_COMPONENT_COUNT_INT,
        mori_energy_matrix.shape[0],
    ):
        raise ValueError("projected primitive h shape mismatch")

    return ProjectedPrimitiveSet(
        state_labels=state_labels,
        restricted_equilibrium_populations_c_i_mol_m3=state_concentrations.copy(),
        symmetric_reactive_fluxes_K_ij_mol_m3_s=symmetric_fluxes,
        reversible_generator_Q_ij_s_inv=generator_matrix.copy(),
        conditional_displacement_first_moments_d_ij_m=first_moments,
        conditional_displacement_second_moments_M_ij_m2=second_moments,
        self_current_diffusion_tensors_D_self_i_m2_s=self_current_tensors,
        mori_memory_energy_matrix_A=mori_energy_matrix.copy(),
        mori_current_coupling_matrix_h=mori_current_coupling_matrix.copy(),
        markov_input=markov_input,
        markov_conductivity_result=(generator_primitive_set.markov_conductivity_result),
    )


def _state_index_for_projected_primitive_label(
    state_label: str,
    state_label_to_index: Mapping[str, int],
) -> int:
    if state_label not in state_label_to_index:
        raise ValueError(f"projected primitive state label {state_label} is unknown")
    return state_label_to_index[state_label]


def _projection_markov_events_from_samples(
    pair_samples_by_state_pair: Mapping[tuple[int, int], Sequence[np.ndarray]],
    self_samples_by_state: Mapping[int, Sequence[np.ndarray]],
    active_state_labels: tuple[str, ...],
    active_state_concentrations_mol_m3: np.ndarray,
    event_flux_mol_m3_s: float,
) -> tuple[MarkovAdditiveEvent, ...]:
    event_flux = _positive_float(event_flux_mol_m3_s, "event_flux_mol_m3_s")
    events: list[MarkovAdditiveEvent] = []
    for state_pair, displacement_samples_m in sorted(
        pair_samples_by_state_pair.items()
    ):
        lower_state_index, upper_state_index = state_pair
        for sample_index, canonical_displacement_m in enumerate(displacement_samples_m):
            second_moment_m2 = _outer_second_moment_tuple(canonical_displacement_m)
            lower_to_upper_rate_s_inv = (
                event_flux / active_state_concentrations_mol_m3[lower_state_index]
            )
            upper_to_lower_rate_s_inv = (
                event_flux / active_state_concentrations_mol_m3[upper_state_index]
            )
            event_base_label = (
                "sampled_generator_transition:"
                f"{active_state_labels[lower_state_index]}->"
                f"{active_state_labels[upper_state_index]}:{sample_index}"
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=lower_state_index,
                    to_state_index=upper_state_index,
                    rate_s_inv=float(lower_to_upper_rate_s_inv),
                    charge_displacement_m=_projection_displacement_tuple(
                        canonical_displacement_m,
                    ),
                    charge_displacement_second_moment_m2=second_moment_m2,
                    label=f"{event_base_label}:forward",
                    family_label="sampled_generator_state_transition",
                )
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=upper_state_index,
                    to_state_index=lower_state_index,
                    rate_s_inv=float(upper_to_lower_rate_s_inv),
                    charge_displacement_m=_projection_displacement_tuple(
                        -canonical_displacement_m,
                    ),
                    charge_displacement_second_moment_m2=second_moment_m2,
                    label=f"{event_base_label}:reverse",
                    family_label="sampled_generator_state_transition",
                )
            )
    for state_index, displacement_samples_m in sorted(self_samples_by_state.items()):
        self_rate_s_inv = event_flux / active_state_concentrations_mol_m3[state_index]
        for sample_index, charge_displacement_m in enumerate(displacement_samples_m):
            second_moment_m2 = _outer_second_moment_tuple(charge_displacement_m)
            event_base_label = (
                "sampled_generator_self_current:"
                f"{active_state_labels[state_index]}:{sample_index}"
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index,
                    to_state_index=state_index,
                    rate_s_inv=float(self_rate_s_inv),
                    charge_displacement_m=_projection_displacement_tuple(
                        charge_displacement_m,
                    ),
                    charge_displacement_second_moment_m2=second_moment_m2,
                    label=f"{event_base_label}:plus",
                    family_label="sampled_generator_self_current",
                )
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index,
                    to_state_index=state_index,
                    rate_s_inv=float(self_rate_s_inv),
                    charge_displacement_m=_projection_displacement_tuple(
                        -charge_displacement_m,
                    ),
                    charge_displacement_second_moment_m2=second_moment_m2,
                    label=f"{event_base_label}:minus",
                    family_label="sampled_generator_self_current",
                )
            )
    return tuple(events)


def _projection_reactive_fluxes(
    markov_input: MarkovAdditiveConductivityInput,
) -> tuple[ProjectedGeneratorReactiveFlux, ...]:
    state_concentrations = np.asarray(
        markov_input.state_concentrations_mol_m3,
        dtype=float,
    )
    flux_by_ordered_pair: dict[tuple[int, int], float] = {}
    rate_by_ordered_pair: dict[tuple[int, int], float] = {}
    for event in markov_input.events:
        if event.from_state_index == event.to_state_index:
            continue
        ordered_pair = (event.from_state_index, event.to_state_index)
        event_flux = state_concentrations[event.from_state_index] * event.rate_s_inv
        if ordered_pair not in flux_by_ordered_pair:
            flux_by_ordered_pair[ordered_pair] = 0.0
            rate_by_ordered_pair[ordered_pair] = 0.0
        flux_by_ordered_pair[ordered_pair] += float(event_flux)
        rate_by_ordered_pair[ordered_pair] += float(event.rate_s_inv)
    unordered_pairs = {
        (
            min(first_state_index, second_state_index),
            max(first_state_index, second_state_index),
        )
        for first_state_index, second_state_index in flux_by_ordered_pair
    }
    reactive_fluxes: list[ProjectedGeneratorReactiveFlux] = []
    for lower_state_index, upper_state_index in sorted(unordered_pairs):
        forward_flux = flux_by_ordered_pair[(lower_state_index, upper_state_index)]
        reverse_flux = flux_by_ordered_pair[(upper_state_index, lower_state_index)]
        tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
            1.0,
            abs(forward_flux),
            abs(reverse_flux),
        )
        if abs(forward_flux - reverse_flux) > tolerance:
            raise ValueError(
                "sampled projected flux is not detailed-balanced for "
                f"{markov_input.state_labels[lower_state_index]} <-> "
                f"{markov_input.state_labels[upper_state_index]}"
            )
        reactive_fluxes.append(
            ProjectedGeneratorReactiveFlux(
                from_state_label=markov_input.state_labels[lower_state_index],
                to_state_label=markov_input.state_labels[upper_state_index],
                symmetric_flux_mol_m3_s=0.5 * (forward_flux + reverse_flux),
                forward_rate_s_inv=rate_by_ordered_pair[
                    (lower_state_index, upper_state_index)
                ],
                reverse_rate_s_inv=rate_by_ordered_pair[
                    (upper_state_index, lower_state_index)
                ],
            )
        )
    return tuple(reactive_fluxes)


def _projection_conditional_displacement_moments(
    markov_input: MarkovAdditiveConductivityInput,
) -> tuple[ProjectedGeneratorConditionalMoment, ...]:
    samples_by_transition: dict[tuple[int, int], list[np.ndarray]] = {}
    for event in markov_input.events:
        if event.from_state_index == event.to_state_index:
            continue
        transition_key = (event.from_state_index, event.to_state_index)
        if transition_key not in samples_by_transition:
            samples_by_transition[transition_key] = []
        samples_by_transition[transition_key].append(
            np.asarray(event.charge_displacement_m, dtype=float)
        )
    conditional_moments: list[ProjectedGeneratorConditionalMoment] = []
    for transition_key in sorted(samples_by_transition):
        displacement_samples = np.asarray(
            samples_by_transition[transition_key],
            dtype=float,
        )
        mean_displacement = np.mean(displacement_samples, axis=0)
        second_moment = np.einsum(
            "ni,nj->ij", displacement_samples, displacement_samples
        ) / float(displacement_samples.shape[0])
        covariance = second_moment - np.outer(mean_displacement, mean_displacement)
        _validate_positive_semidefinite_matrix(
            covariance,
            "projected_generator.conditional_displacement_covariance_m2",
        )
        from_state_index, to_state_index = transition_key
        conditional_moments.append(
            ProjectedGeneratorConditionalMoment(
                from_state_label=markov_input.state_labels[from_state_index],
                to_state_label=markov_input.state_labels[to_state_index],
                sample_count=int(displacement_samples.shape[0]),
                mean_charge_displacement_m=_projection_displacement_tuple(
                    mean_displacement
                ),
                second_moment_m2=_projection_matrix_tuple(second_moment),
                covariance_m2=_projection_matrix_tuple(covariance),
            )
        )
    return tuple(conditional_moments)


def _projection_self_current_tensors(
    active_state_labels: tuple[str, ...],
    active_state_concentrations_mol_m3: np.ndarray,
    self_samples_by_state: Mapping[int, Sequence[np.ndarray]],
    dt_s: float,
) -> tuple[ProjectedGeneratorSelfCurrentTensor, ...]:
    self_current_tensors: list[ProjectedGeneratorSelfCurrentTensor] = []
    for state_index in sorted(self_samples_by_state):
        displacement_samples = np.asarray(
            self_samples_by_state[state_index],
            dtype=float,
        )
        diffusion_tensor = np.einsum(
            "ni,nj->ij", displacement_samples, displacement_samples
        ) / (2.0 * dt_s * float(displacement_samples.shape[0]))
        _validate_positive_semidefinite_matrix(
            diffusion_tensor,
            "projected_generator.self_current_diffusion_tensor_m2_s",
        )
        self_current_tensors.append(
            ProjectedGeneratorSelfCurrentTensor(
                state_label=active_state_labels[state_index],
                sample_count=int(displacement_samples.shape[0]),
                concentration_mol_m3=float(
                    active_state_concentrations_mol_m3[state_index]
                ),
                diffusion_tensor_m2_s=_projection_matrix_tuple(diffusion_tensor),
            )
        )
    return tuple(self_current_tensors)


def _projection_state_labels(
    state_labels: Sequence[str],
) -> tuple[str, ...]:
    labels = tuple(str(state_label) for state_label in state_labels)
    if not labels:
        raise ValueError("sampled generator projection requires state labels")
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"sampled generator projection labels must be unique: {labels}"
        )
    if any(not state_label for state_label in labels):
        raise ValueError("sampled generator projection labels cannot be empty")
    return labels


def _projection_sample_indices(
    state_indices: np.ndarray,
    state_count: int,
    label: str,
) -> np.ndarray:
    indices = np.asarray(state_indices, dtype=int).reshape(-1)
    if indices.size == 0:
        raise ValueError(f"{label} cannot be empty")
    if int(np.min(indices)) < 0:
        raise ValueError(f"{label} contains negative state indices")
    maximum_state_index = int(np.max(indices))
    if maximum_state_index >= state_count:
        raise ValueError(
            f"{label} references state index {maximum_state_index}, "
            f"but only {state_count} states were provided"
        )
    return indices


def _projection_charge_displacements(
    charge_displacement_by_step_m: np.ndarray,
    expected_step_count: int,
) -> np.ndarray:
    charge_displacements = np.asarray(charge_displacement_by_step_m, dtype=float)
    if charge_displacements.shape != (expected_step_count, 3):
        raise ValueError(
            "charge_displacement_by_step_m must have shape "
            f"({expected_step_count}, 3), got {charge_displacements.shape}"
        )
    if not np.all(np.isfinite(charge_displacements)):
        raise ValueError("charge_displacement_by_step_m contains non-finite values")
    return charge_displacements


def _projection_remap_visited_states(
    state_labels: tuple[str, ...],
    state_indices: np.ndarray,
) -> tuple[dict[int, int], tuple[str, ...], np.ndarray]:
    visited_original_indices = tuple(
        sorted(int(state_index) for state_index in np.unique(state_indices))
    )
    if not visited_original_indices:
        raise ValueError("sampled generator projection has no visited states")
    state_index_by_original_index = {
        original_state_index: remapped_state_index
        for remapped_state_index, original_state_index in enumerate(
            visited_original_indices
        )
    }
    remapped_states = np.asarray(
        tuple(
            state_index_by_original_index[int(state_index)]
            for state_index in state_indices
        ),
        dtype=int,
    )
    remapped_labels = tuple(
        state_labels[original_state_index]
        for original_state_index in visited_original_indices
    )
    return state_index_by_original_index, remapped_labels, remapped_states


def _projection_state_concentrations_from_occupancy(
    occupancy_state_indices: np.ndarray,
    state_count: int,
    total_concentration_mol_m3: float,
) -> np.ndarray:
    occupancy_counts = np.bincount(occupancy_state_indices, minlength=state_count)
    if np.any(occupancy_counts <= 0):
        raise ValueError("all active projected states must have occupancy samples")
    occupancy_fractions = occupancy_counts.astype(float) / float(
        occupancy_state_indices.shape[0]
    )
    state_concentrations_mol_m3 = total_concentration_mol_m3 * occupancy_fractions
    if np.any(state_concentrations_mol_m3 <= 0.0):
        raise ValueError("projected state concentrations must be positive")
    return state_concentrations_mol_m3


def _projection_state_occupancy_fractions(
    state_labels: tuple[str, ...],
    occupancy_state_indices: np.ndarray,
) -> Mapping[str, float]:
    occupancy_counts = np.bincount(
        occupancy_state_indices,
        minlength=len(state_labels),
    )
    total_occupancy_count = float(np.sum(occupancy_counts))
    if total_occupancy_count <= 0.0:
        raise ValueError("sampled generator projection has no occupancy observations")
    return {
        state_label: float(occupancy_counts[state_index] / total_occupancy_count)
        for state_index, state_label in enumerate(state_labels)
    }


def _projection_displacement_tuple(
    displacement_m: np.ndarray,
) -> tuple[float, float, float]:
    displacement = np.asarray(displacement_m, dtype=float)
    if displacement.shape != (3,):
        raise ValueError(
            f"projected displacement must have shape (3,), got {displacement.shape}"
        )
    if not np.all(np.isfinite(displacement)):
        raise ValueError("projected displacement contains non-finite values")
    return tuple(float(component) for component in displacement)


def _outer_second_moment_tuple(
    displacement_m: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    displacement = np.asarray(displacement_m, dtype=float)
    second_moment = np.outer(displacement, displacement)
    return _projection_matrix_tuple(second_moment)


def _projection_matrix_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (3, 3):
        raise ValueError(f"projected tensor must have shape (3, 3), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("projected tensor contains non-finite values")
    return tuple(
        tuple(float(values[row_index, column_index]) for column_index in range(3))
        for row_index in range(3)
    )


def _validated_overdamped_generator_input_arrays(
    generator_input: OverdampedSmoluchowskiGeneratorInput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial_positions_m = np.asarray(generator_input.initial_positions_m, dtype=float)
    if (
        initial_positions_m.ndim != 2
        or initial_positions_m.shape[1] != CARTESIAN_COMPONENT_COUNT_INT
    ):
        raise ValueError(
            "initial_positions_m must have shape (molecule_count, 3), got "
            f"{initial_positions_m.shape}"
        )
    if initial_positions_m.shape[0] == 0:
        raise ValueError("initial_positions_m must contain at least one molecule")
    if not np.all(np.isfinite(initial_positions_m)):
        raise ValueError("initial_positions_m contains non-finite values")

    molecule_species_indices = np.asarray(
        generator_input.molecule_species_indices,
        dtype=int,
    )
    formal_charge_numbers = np.asarray(
        generator_input.formal_charge_numbers,
        dtype=float,
    )
    diffusion_coefficients_m2_s = np.asarray(
        generator_input.diffusion_coefficients_m2_s,
        dtype=float,
    )
    expected_vector_shape = (initial_positions_m.shape[0],)
    if molecule_species_indices.shape != expected_vector_shape:
        raise ValueError(
            "molecule_species_indices must have one value per molecule: got "
            f"{molecule_species_indices.shape}, expected {expected_vector_shape}"
        )
    if formal_charge_numbers.shape != expected_vector_shape:
        raise ValueError(
            "formal_charge_numbers must have one value per molecule: got "
            f"{formal_charge_numbers.shape}, expected {expected_vector_shape}"
        )
    if diffusion_coefficients_m2_s.shape != expected_vector_shape:
        raise ValueError(
            "diffusion_coefficients_m2_s must have one value per molecule: got "
            f"{diffusion_coefficients_m2_s.shape}, expected {expected_vector_shape}"
        )
    if np.any(molecule_species_indices < 0):
        raise ValueError("molecule_species_indices must be nonnegative")
    if not np.all(np.isfinite(formal_charge_numbers)):
        raise ValueError("formal_charge_numbers contains non-finite values")
    if not np.all(np.isfinite(diffusion_coefficients_m2_s)) or np.any(
        diffusion_coefficients_m2_s <= 0.0
    ):
        raise ValueError("diffusion_coefficients_m2_s must be positive and finite")

    box_lengths_m = np.asarray(generator_input.box_lengths_m, dtype=float)
    if box_lengths_m.shape != (CARTESIAN_COMPONENT_COUNT_INT,):
        raise ValueError(
            f"box_lengths_m must have shape (3,), got {box_lengths_m.shape}"
        )
    if not np.all(np.isfinite(box_lengths_m)) or np.any(box_lengths_m <= 0.0):
        raise ValueError("box_lengths_m must be positive and finite")
    return (
        initial_positions_m,
        molecule_species_indices,
        formal_charge_numbers,
        box_lengths_m,
        diffusion_coefficients_m2_s,
    )


def _validate_nonempty_generator_source(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"MicroscopicGeneratorModel.{field_name} must be nonempty")


def _positive_int(value: int, context: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise ValueError(f"{context} must be positive")
    return parsed_value


def _nonnegative_int(value: int, context: str) -> int:
    parsed_value = int(value)
    if parsed_value < 0:
        raise ValueError(f"{context} must be nonnegative")
    return parsed_value


def _validated_microscopic_generator_trajectory(
    trajectory: MolecularTrajectory,
) -> None:
    positions = np.asarray(trajectory.com_positions, dtype=float)
    if positions.ndim != 3:
        raise ValueError(
            f"trajectory.com_positions must be 3D, got shape {positions.shape}"
        )
    if positions.shape[2] != 3:
        raise ValueError(
            f"trajectory.com_positions final axis must have length 3, got {positions.shape}"
        )
    if positions.shape[0] < 2:
        raise ValueError("trajectory must contain at least two position frames")
    charge_numbers = np.asarray(trajectory.formal_charges, dtype=float)
    if charge_numbers.shape != (positions.shape[1],):
        raise ValueError(
            "trajectory.formal_charges must have one value per molecule: "
            f"got {charge_numbers.shape} for {positions.shape[1]} molecules"
        )
    box_lengths_angstrom = np.asarray(trajectory.box, dtype=float)
    if box_lengths_angstrom.shape != (3,):
        raise ValueError(
            f"trajectory.box must have shape (3,), got {box_lengths_angstrom.shape}"
        )
    if np.any(box_lengths_angstrom <= 0.0):
        raise ValueError(
            f"trajectory.box lengths must be positive, got {box_lengths_angstrom}"
        )
    if float(trajectory.dt_fs) <= 0.0:
        raise ValueError(f"trajectory.dt_fs must be positive, got {trajectory.dt_fs}")
    if float(trajectory.temperature_K) <= 0.0:
        raise ValueError(
            f"trajectory.temperature_K must be positive, got {trajectory.temperature_K}"
        )


def _validated_projected_basis_functions(
    basis_functions: tuple[ProjectedBasisFunctionDefinition, ...],
) -> tuple[ProjectedBasisFunctionDefinition, ...]:
    if not basis_functions:
        raise ValueError("ProjectedBasisAssignment.basis_functions must be nonempty")
    state_labels = tuple(
        basis_function.state_label for basis_function in basis_functions
    )
    if len(set(state_labels)) != len(state_labels):
        raise ValueError(f"Projected basis state labels must be unique: {state_labels}")
    validated_basis_functions: list[ProjectedBasisFunctionDefinition] = []
    for basis_function in basis_functions:
        if not basis_function.state_label:
            raise ValueError("Projected basis state labels cannot be empty")
        if basis_function.projection_class not in SUPPORTED_PROJECTION_CLASSES:
            raise ValueError(
                f"unsupported projection class {basis_function.projection_class} "
                f"for state {basis_function.state_label}"
            )
        validated_basis_functions.append(
            ProjectedBasisFunctionDefinition(
                state_label=str(basis_function.state_label),
                projection_class=str(basis_function.projection_class),
            )
        )
    return tuple(validated_basis_functions)


def _dc_transport_state_mask(
    basis_functions: tuple[ProjectedBasisFunctionDefinition, ...],
) -> np.ndarray:
    return np.asarray(
        [
            basis_function.projection_class in DC_TRANSPORT_PROJECTION_CLASSES
            for basis_function in basis_functions
        ],
        dtype=bool,
    )


def _self_current_state_mask(
    basis_functions: tuple[ProjectedBasisFunctionDefinition, ...],
) -> np.ndarray:
    return np.asarray(
        [
            basis_function.projection_class == PROJECTION_CLASS_SELF_CURRENT_CARRIER
            for basis_function in basis_functions
        ],
        dtype=bool,
    )


def _validated_projected_basis_indices(
    state_index_by_frame_and_molecule: np.ndarray,
    state_count: int,
    trajectory_position_shape: tuple[int, int, int],
) -> np.ndarray:
    state_indices = np.asarray(state_index_by_frame_and_molecule, dtype=int)
    expected_shape = trajectory_position_shape[:2]
    if state_indices.shape != expected_shape:
        raise ValueError(
            "state_index_by_frame_and_molecule must match trajectory frame/molecule shape: "
            f"got {state_indices.shape}, expected {expected_shape}"
        )
    if state_indices.size == 0:
        raise ValueError("state_index_by_frame_and_molecule cannot be empty")
    if int(np.min(state_indices)) < 0:
        raise ValueError(
            "state_index_by_frame_and_molecule contains negative state indices"
        )
    maximum_state_index = int(np.max(state_indices))
    if maximum_state_index >= state_count:
        raise ValueError(
            "state_index_by_frame_and_molecule references state index "
            f"{maximum_state_index}, but only {state_count} labels were provided"
        )
    return state_indices


def _charge_displacement_matrix_by_generator_step_m(
    generator_model: MicroscopicGeneratorModel,
    charged_molecule_indices: np.ndarray,
) -> np.ndarray:
    positions_m = (
        np.asarray(generator_model.trajectory.com_positions, dtype=float)
        * ANGSTROM_TO_M
    )
    frame_displacements_m = np.diff(positions_m[:, charged_molecule_indices, :], axis=0)
    charge_numbers = np.asarray(generator_model.trajectory.formal_charges, dtype=float)[
        charged_molecule_indices
    ]
    charge_weighted_displacements_m = (
        frame_displacements_m * charge_numbers[np.newaxis, :, np.newaxis]
    )
    return charge_weighted_displacements_m


def _validated_vector_timeseries(
    vector_timeseries: np.ndarray, name: str
) -> np.ndarray:
    values = np.asarray(vector_timeseries, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (sample_count, 3), got {values.shape}"
        )
    if values.shape[0] < 2:
        raise ValueError(f"{name} requires at least two samples, got {values.shape[0]}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _current_autocorrelation(current_charge_number_m_s: np.ndarray) -> np.ndarray:
    sample_count = current_charge_number_m_s.shape[0]
    autocorrelation = np.empty(sample_count, dtype=float)
    for lag_index in range(sample_count):
        current_head = current_charge_number_m_s[: sample_count - lag_index]
        current_tail = current_charge_number_m_s[lag_index:]
        autocorrelation[lag_index] = float(
            np.mean(np.sum(current_head * current_tail, axis=1))
        )
    return autocorrelation


# ---- molecular_descriptors.py ----

ROLE_CATION = "cation"
ROLE_ANION = "anion"
ROLE_SOLVENT = "solvent"
ROLE_ADDITIVE = "additive"
ROLE_NEUTRAL = "neutral"
SUPPORTED_MOLECULAR_SPECIES_ROLES = (
    ROLE_CATION,
    ROLE_ANION,
    ROLE_SOLVENT,
    ROLE_ADDITIVE,
    ROLE_NEUTRAL,
)


@dataclass(frozen=True)
class MolecularSpeciesInput:
    name: str
    role: str
    charge_number: int
    smiles: str
    xyz_coordinates: tuple[tuple[str, float, float, float], ...]
    property_overrides: Mapping[str, float]
    coordination_sites: tuple[str, ...]


@dataclass(frozen=True)
class MolecularSpeciesDescriptor:
    name: str
    role: str
    charge_number: int
    molecular_weight_g_mol: float
    hard_sphere_radius_A: float
    hydrodynamic_radius_A: float
    cavity_radius_A: float
    charge_cloud_radius_A: float
    molecular_volume_A3: float
    solvent_accessible_area_A2: float
    dipole_D: float
    quadrupole_D_A: float
    polarizability_A3: float
    donor_number: float
    acceptor_number: float
    hbond_donor_count: int
    hbond_acceptor_count: int
    epsilon_r_pure: float
    viscosity_cP_pure: float
    density_g_ml: float
    born_solvation_radius_A: float
    coordination_sites: tuple[str, ...]
    coordination_affinity_J_mol: float
    ligand_field_asymmetry: float


class MolecularDescriptorBackend(Protocol):
    def describe_species(
        self,
        species: MolecularSpeciesInput,
        temperature_K: float,
    ) -> MolecularSpeciesDescriptor: ...


class ProvidedPropertyDescriptorBackend:
    """Build descriptors only from user-supplied molecular property values."""

    def describe_species(
        self,
        species: MolecularSpeciesInput,
        temperature_K: float,
    ) -> MolecularSpeciesDescriptor:
        _validate_species_identity(species)
        _positive_float(temperature_K, "temperature_K")
        properties = species.property_overrides
        return MolecularSpeciesDescriptor(
            name=species.name,
            role=species.role,
            charge_number=species.charge_number,
            molecular_weight_g_mol=_required_positive_property(
                properties,
                "molecular_weight_g_mol",
                species.name,
            ),
            hard_sphere_radius_A=_required_positive_property(
                properties,
                "hard_sphere_radius_A",
                species.name,
            ),
            hydrodynamic_radius_A=_required_positive_property(
                properties,
                "hydrodynamic_radius_A",
                species.name,
            ),
            cavity_radius_A=_required_positive_property(
                properties,
                "cavity_radius_A",
                species.name,
            ),
            charge_cloud_radius_A=_required_positive_property(
                properties,
                "charge_cloud_radius_A",
                species.name,
            ),
            molecular_volume_A3=_required_positive_property(
                properties,
                "molecular_volume_A3",
                species.name,
            ),
            solvent_accessible_area_A2=_required_nonnegative_property(
                properties,
                "solvent_accessible_area_A2",
                species.name,
            ),
            dipole_D=_required_nonnegative_property(
                properties,
                "dipole_D",
                species.name,
            ),
            quadrupole_D_A=_required_nonnegative_property(
                properties,
                "quadrupole_D_A",
                species.name,
            ),
            polarizability_A3=_required_nonnegative_property(
                properties,
                "polarizability_A3",
                species.name,
            ),
            donor_number=_required_nonnegative_property(
                properties,
                "donor_number",
                species.name,
            ),
            acceptor_number=_required_nonnegative_property(
                properties,
                "acceptor_number",
                species.name,
            ),
            hbond_donor_count=_required_nonnegative_integer_property(
                properties,
                "hbond_donor_count",
                species.name,
            ),
            hbond_acceptor_count=_required_nonnegative_integer_property(
                properties,
                "hbond_acceptor_count",
                species.name,
            ),
            epsilon_r_pure=_required_positive_property(
                properties,
                "epsilon_r_pure",
                species.name,
            ),
            viscosity_cP_pure=_required_positive_property(
                properties,
                "viscosity_cP_pure",
                species.name,
            ),
            density_g_ml=_required_positive_property(
                properties,
                "density_g_ml",
                species.name,
            ),
            born_solvation_radius_A=_required_positive_property(
                properties,
                "born_solvation_radius_A",
                species.name,
            ),
            coordination_sites=tuple(species.coordination_sites),
            coordination_affinity_J_mol=_required_nonnegative_property(
                properties,
                "coordination_affinity_J_mol",
                species.name,
            ),
            ligand_field_asymmetry=_required_positive_property(
                properties,
                "ligand_field_asymmetry",
                species.name,
            ),
        )


def _validate_species_identity(species: MolecularSpeciesInput) -> None:
    if species.name == "":
        raise ValueError("species name must be nonempty")
    if species.role not in SUPPORTED_MOLECULAR_SPECIES_ROLES:
        raise ValueError(
            f"species {species.name} has unsupported role {species.role!r}"
        )
    if not isinstance(species.charge_number, int):
        raise TypeError(f"species {species.name} charge_number must be an integer")
    if species.role == ROLE_CATION and species.charge_number <= 0:
        raise ValueError(f"cation {species.name} must have positive charge_number")
    if species.role == ROLE_ANION and species.charge_number >= 0:
        raise ValueError(f"anion {species.name} must have negative charge_number")


def _required_positive_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    return _positive_float(
        _required_numeric_property(properties, key, species_name),
        f"{species_name}.{key}",
    )


def _required_nonnegative_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    return _nonnegative_float(
        _required_numeric_property(properties, key, species_name),
        f"{species_name}.{key}",
    )


def _required_nonnegative_integer_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> int:
    value = _required_nonnegative_property(properties, key, species_name)
    integer_value = int(value)
    if float(integer_value) != value:
        raise ValueError(f"{species_name}.{key} must be an integer-valued number")
    return integer_value


def _required_numeric_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    if key not in properties:
        raise ValueError(f"species {species_name} missing molecular descriptor {key}")
    value = properties[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"species {species_name} descriptor {key} must be numeric")
    return float(value)


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _nonnegative_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        raise ValueError(f"{context} must be nonnegative and finite")
    return parsed_value


# ---- molecular_primitive_parameters.py ----

PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE = "log_positive"
PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED = "identity_signed"


@dataclass(frozen=True)
class ConductivityPrimitiveParameterSet:
    coulomb_scale: float
    desolvation_scale: float
    coordination_scale: float
    steric_free_energy_scale: float
    cluster_entropy_penalty_scale: float
    association_crowding_stabilization_scale: float
    association_crowding_ionic_strength_exponent: float
    association_crowding_charge_density_exponent: float
    activity_debye_scale: float
    activity_size_scale: float
    activity_hard_sphere_scale: float
    cluster_activity_scale: float
    pair_logK_offset: float
    solvent_separated_pair_logK_offset: float
    contact_pair_logK_offset: float
    positive_charged_triplet_logK_offset: float
    negative_charged_triplet_logK_offset: float
    neutral_cluster_logK_offset: float
    higher_charged_cluster_logK_offset: float
    contact_pair_desolvation_offset_over_RT: float
    solvent_separated_pair_desolvation_offset_over_RT: float
    higher_charged_cluster_desolvation_offset_over_RT: float
    internal_polarization_projection_offset: float
    cluster_order_logK_slope: float
    cluster_charge_magnitude_logK_slope: float
    cluster_hydrodynamic_radius_scale: float
    hydrodynamic_radius_scale_positive_ion: float
    hydrodynamic_radius_scale_negative_ion: float
    hydrodynamic_radius_scale_cluster: float
    shape_friction_exponent: float
    free_volume_exponent: float
    dielectric_mobility_exponent: float
    solvation_mobility_exponent: float
    additive_shape_solvation_mobility_exponent: float
    positive_ion_charge_density_mobility_exponent: float
    negative_ion_charge_density_mobility_exponent: float
    positive_ion_counteranion_charge_cloud_mobility_exponent: float
    negative_ion_charge_cloud_mobility_exponent: float
    negative_ion_intrinsic_dielectric_drag_mobility_exponent: float
    negative_ion_shape_delocalization_mobility_exponent: float
    positive_ion_anion_disorder_mobility_exponent: float
    negative_ion_anion_disorder_mobility_exponent: float
    local_obstruction_strength: float
    local_obstruction_free_volume_exponent: float
    local_obstruction_ionic_strength_exponent: float
    local_obstruction_additive_solvation_exponent: float
    local_obstruction_size_exponent: float
    local_obstruction_charge_density_exponent: float
    local_obstruction_solvation_exponent: float
    atmosphere_ep_scale: float
    atmosphere_rel_scale: float
    charge_cloud_radius_scale: float
    cross_relaxation_scale: float
    jump_length_scale: float
    atmosphere_capture_scale: float
    atmosphere_exit_scale: float
    association_conversion_rate_scale: float
    orientation_relaxation_rate_scale: float
    internal_polarization_projection_ionic_strength_slope: float
    internal_polarization_projection_counterion_crowding_slope: float


CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES = tuple(
    field.name for field in fields(ConductivityPrimitiveParameterSet)
)

PROJECTED_READOUT_PROVEN_DESCRIPTOR_CLOSURE_EMPIRICAL = (
    "projected_readout_proven_descriptor_closure_empirical"
)
FINITE_PROJECTED_DESCRIPTOR_CLOSURE_PROOF_STATUS = (
    "finite_projected_descriptor_closure_projected_gk"
)
FULL_MICROSCOPIC_GENERATOR_DERIVED_PROOF_STATUS = (
    "full_microscopic_generator_derived_projected_gk"
)
PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY = (
    "equilibrium_free_energy_U_x_parameter"
)
PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION = "transport_partition_A_i_parameter"
PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX = "reactive_flux_K_ij_parameter"
PRIMITIVE_PARAMETER_ROLE_DISPLACEMENT_MOMENT = (
    "charge_displacement_moment_d_ij_parameter"
)
PRIMITIVE_PARAMETER_ROLE_EMPIRICAL_CLOSURE = "empirical_closure_parameter"
MORI_BASIS_SOURCE_PARTITION_INDICATOR = "partition_indicator"
MORI_BASIS_SOURCE_IONIC_ATMOSPHERE_POLARIZATION = "ionic_atmosphere_polarization"
MORI_BASIS_SOURCE_SOLVENT_CAGE_ORIENTATION = "solvent_cage_orientation"
MORI_BASIS_SOURCE_CHARGE_DENSITY_MODE = "charge_density_fourier_mode"
MORI_BASIS_SOURCE_NUMBER_DENSITY_MODE = "number_density_fourier_mode"
MORI_BASIS_SOURCE_LOCAL_FREE_VOLUME_OR_STRESS = "local_free_volume_or_stress"
SUPPORTED_MORI_BASIS_SOURCES = (
    MORI_BASIS_SOURCE_PARTITION_INDICATOR,
    MORI_BASIS_SOURCE_IONIC_ATMOSPHERE_POLARIZATION,
    MORI_BASIS_SOURCE_SOLVENT_CAGE_ORIENTATION,
    MORI_BASIS_SOURCE_CHARGE_DENSITY_MODE,
    MORI_BASIS_SOURCE_NUMBER_DENSITY_MODE,
    MORI_BASIS_SOURCE_LOCAL_FREE_VOLUME_OR_STRESS,
)
CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME = {
    "coulomb_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "desolvation_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "coordination_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "steric_free_energy_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "cluster_entropy_penalty_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "association_crowding_stabilization_scale": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "association_crowding_ionic_strength_exponent": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "association_crowding_charge_density_exponent": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "activity_debye_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "activity_size_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "activity_hard_sphere_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "cluster_activity_scale": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "pair_logK_offset": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "solvent_separated_pair_logK_offset": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "contact_pair_logK_offset": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "positive_charged_triplet_logK_offset": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "negative_charged_triplet_logK_offset": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "neutral_cluster_logK_offset": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "higher_charged_cluster_logK_offset": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "contact_pair_desolvation_offset_over_RT": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "solvent_separated_pair_desolvation_offset_over_RT": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "higher_charged_cluster_desolvation_offset_over_RT": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "internal_polarization_projection_offset": (
        PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION
    ),
    "cluster_order_logK_slope": PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    "cluster_charge_magnitude_logK_slope": (
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    ),
    "cluster_hydrodynamic_radius_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "hydrodynamic_radius_scale_positive_ion": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "hydrodynamic_radius_scale_negative_ion": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "hydrodynamic_radius_scale_cluster": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "shape_friction_exponent": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "free_volume_exponent": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "dielectric_mobility_exponent": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "solvation_mobility_exponent": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "additive_shape_solvation_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "positive_ion_charge_density_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "negative_ion_charge_density_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "positive_ion_counteranion_charge_cloud_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "negative_ion_charge_cloud_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "negative_ion_intrinsic_dielectric_drag_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "negative_ion_shape_delocalization_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "positive_ion_anion_disorder_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "negative_ion_anion_disorder_mobility_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "local_obstruction_strength": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "local_obstruction_free_volume_exponent": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "local_obstruction_ionic_strength_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "local_obstruction_additive_solvation_exponent": (
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
    ),
    "local_obstruction_size_exponent": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "local_obstruction_charge_density_exponent": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "local_obstruction_solvation_exponent": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "atmosphere_ep_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "atmosphere_rel_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "charge_cloud_radius_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "cross_relaxation_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "jump_length_scale": PRIMITIVE_PARAMETER_ROLE_DISPLACEMENT_MOMENT,
    "atmosphere_capture_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "atmosphere_exit_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "association_conversion_rate_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "orientation_relaxation_rate_scale": PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
    "internal_polarization_projection_ionic_strength_slope": (
        PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION
    ),
    "internal_polarization_projection_counterion_crowding_slope": (
        PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION
    ),
}
_UNCLASSIFIED_PRIMITIVE_PARAMETER_NAMES = set(
    CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
) - set(CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME)
if _UNCLASSIFIED_PRIMITIVE_PARAMETER_NAMES:
    raise ValueError(
        "unclassified conductivity primitive parameters: "
        f"{sorted(_UNCLASSIFIED_PRIMITIVE_PARAMETER_NAMES)}"
    )


CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES = (
    "pair_logK_offset",
    "solvent_separated_pair_logK_offset",
    "contact_pair_logK_offset",
    "positive_charged_triplet_logK_offset",
    "negative_charged_triplet_logK_offset",
    "neutral_cluster_logK_offset",
    "higher_charged_cluster_logK_offset",
    "contact_pair_desolvation_offset_over_RT",
    "solvent_separated_pair_desolvation_offset_over_RT",
    "higher_charged_cluster_desolvation_offset_over_RT",
    "internal_polarization_projection_offset",
    "cluster_order_logK_slope",
    "cluster_charge_magnitude_logK_slope",
)


CONDUCTIVITY_PRIMITIVE_POSITIVE_PARAMETER_FIELD_NAMES = tuple(
    field_name
    for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    if field_name not in CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES
)


CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME = {
    **{
        field_name: PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE
        for field_name in CONDUCTIVITY_PRIMITIVE_POSITIVE_PARAMETER_FIELD_NAMES
    },
    **{
        field_name: PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED
        for field_name in CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES
    },
}


def conductivity_primitive_parameters_to_coordinate_values(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> tuple[float, ...]:
    validate_conductivity_primitive_parameters(primitive_parameters)
    return tuple(
        _parameter_coordinate_value(primitive_parameters, field_name)
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    )


def conductivity_primitive_parameter_coordinate_values_for_names(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    parameter_names: tuple[str, ...],
) -> tuple[float, ...]:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_parameter_name_tuple(parameter_names)
    return tuple(
        _parameter_coordinate_value(primitive_parameters, parameter_name)
        for parameter_name in parameter_names
    )


def conductivity_primitive_parameters_from_coordinate_values(
    coordinate_values: tuple[float, ...],
) -> ConductivityPrimitiveParameterSet:
    field_names = CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    if len(coordinate_values) != len(field_names):
        raise ValueError(
            "coordinate_values length must match ConductivityPrimitiveParameterSet field count"
        )
    parameter_values: dict[str, float] = {}
    for field_name, coordinate_value in zip(field_names, coordinate_values):
        parameter_values[field_name] = _parameter_value_from_coordinate(
            field_name,
            coordinate_value,
        )
    return _conductivity_primitive_parameters_from_validated_values(parameter_values)


def conductivity_primitive_parameters_with_coordinate_updates(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    parameter_names: tuple[str, ...],
    coordinate_values: tuple[float, ...],
) -> ConductivityPrimitiveParameterSet:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_parameter_name_tuple(parameter_names)
    if len(parameter_names) != len(coordinate_values):
        raise ValueError("parameter_names and coordinate_values must have equal length")
    parameter_values = {
        field_name: _parameter_value(primitive_parameters, field_name)
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    }
    for parameter_name, coordinate_value in zip(parameter_names, coordinate_values):
        parameter_values[parameter_name] = _parameter_value_from_coordinate(
            parameter_name,
            coordinate_value,
        )
    return _conductivity_primitive_parameters_from_validated_values(parameter_values)


def conductivity_primitive_parameters_to_mapping(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> dict[str, float]:
    validate_conductivity_primitive_parameters(primitive_parameters)
    return {
        field_name: _parameter_value(primitive_parameters, field_name)
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    }


def conductivity_primitive_parameters_from_mapping(
    parameter_mapping: Mapping[str, float],
) -> ConductivityPrimitiveParameterSet:
    missing_parameter_names = tuple(
        field_name
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        if field_name not in parameter_mapping
    )
    if missing_parameter_names:
        raise ValueError(
            f"missing conductivity primitive parameters {missing_parameter_names}"
        )
    unknown_parameter_names = tuple(
        sorted(
            field_name
            for field_name in parameter_mapping
            if field_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        )
    )
    if unknown_parameter_names:
        raise ValueError(
            f"unknown conductivity primitive parameters {unknown_parameter_names}"
        )
    parameter_values = {
        field_name: _validated_parameter_value(
            field_name,
            parameter_mapping[field_name],
        )
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    }
    return _conductivity_primitive_parameters_from_validated_values(parameter_values)


def _conductivity_primitive_parameters_from_validated_values(
    parameter_values: Mapping[str, float],
) -> ConductivityPrimitiveParameterSet:
    primitive_parameters = ConductivityPrimitiveParameterSet(
        **{
            field_name: _validated_parameter_value(
                field_name,
                parameter_values[field_name],
            )
            for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        }
    )
    validate_conductivity_primitive_parameters(primitive_parameters)
    return primitive_parameters


def validate_conductivity_primitive_parameters(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> None:
    for primitive_parameter_field in fields(ConductivityPrimitiveParameterSet):
        field_name = primitive_parameter_field.name
        _validated_parameter_value(
            field_name,
            getattr(primitive_parameters, field_name),
        )


def _parameter_value(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    field_name: str,
) -> float:
    if field_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        raise ValueError(f"unknown conductivity primitive parameter {field_name}")
    return _validated_parameter_value(
        field_name, getattr(primitive_parameters, field_name)
    )


def _parameter_coordinate_value(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    field_name: str,
) -> float:
    parameter_value = _parameter_value(primitive_parameters, field_name)
    transform_name = _parameter_transform_name(field_name)
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return math.log(parameter_value)
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return parameter_value
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


def _parameter_value_from_coordinate(
    field_name: str,
    coordinate_value: float,
) -> float:
    parsed_coordinate_value = _finite_float(
        coordinate_value,
        f"primitive_parameters.{field_name}.coordinate",
    )
    transform_name = _parameter_transform_name(field_name)
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return _positive_float(
            math.exp(parsed_coordinate_value),
            f"primitive_parameters.{field_name}",
        )
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return parsed_coordinate_value
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


def _parameter_transform_name(field_name: str) -> str:
    if field_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME:
        raise ValueError(f"unknown conductivity primitive parameter {field_name}")
    return CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[field_name]


def _validated_parameter_value(field_name: str, value: float) -> float:
    context = f"primitive_parameters.{field_name}"
    if field_name in CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES:
        return _finite_float(value, context)
    return _positive_float(value, context)


def _validate_parameter_name_tuple(parameter_names: tuple[str, ...]) -> None:
    if not parameter_names:
        raise ValueError("parameter_names must be nonempty")
    seen_parameter_names: set[str] = set()
    for parameter_name in parameter_names:
        if parameter_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
            raise ValueError(
                f"unknown conductivity primitive parameter {parameter_name}"
            )
        if parameter_name in seen_parameter_names:
            raise ValueError(
                f"duplicate conductivity primitive parameter {parameter_name}"
            )
        seen_parameter_names.add(parameter_name)


def _positive_float(value: float, context: str) -> float:
    parsed_value = _finite_float(value, context)
    if parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _finite_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{context} must be finite")
    return parsed_value


# ---- finite_mori_conductivity.py ----

CARTESIAN_AXIS_COUNT = 3.0  # Analytical isotropic average over x, y, z current axes.
MORI_NUMERICAL_TOLERANCE = math.sqrt(np.finfo(float).eps)
TRAPEZOID_ENDPOINT_WEIGHT = 0.5  # Analytical trapezoid-rule endpoint weight.


@dataclass(frozen=True)
class ProjectedMoriConductivityInput:
    direct_energy_matrix: np.ndarray
    memory_self_energy_matrix: np.ndarray
    current_coupling_matrix: np.ndarray
    beta_over_volume: float


@dataclass(frozen=True)
class ProjectedMoriConductivityResult:
    sigma_S_m: float
    sigma_mS_cm: float
    axis_conductivity_S_m: tuple[float, float, float]
    quadratic_form_by_axis: tuple[float, float, float]
    energy_eigenvalues: tuple[float, ...]
    effective_energy_matrix: np.ndarray


@dataclass(frozen=True)
class MoriOracleClosureComparison:
    sigma_oracle_S_m: float
    sigma_oracle_mS_cm: float
    sigma_trajectory_mS_cm: float
    closure_gap_mS_cm: float
    tolerance_mS_cm: float
    passes_tolerance: bool


@dataclass(frozen=True)
class TrajectoryProjectedMoriConstruction:
    mori_input: ProjectedMoriConductivityInput
    centered_current_time_series: np.ndarray
    whitened_basis_time_series: np.ndarray
    retained_basis_time_series: np.ndarray
    projected_current_time_series: np.ndarray
    raw_green_kubo_sigma_mS_cm: float
    projected_green_kubo_sigma_mS_cm: float
    raw_green_kubo_axis_integrals: tuple[float, float, float]
    projected_green_kubo_axis_integrals: tuple[float, float, float]
    maximum_lag_steps_used: int
    retained_gram_eigenvalues: tuple[float, ...]
    retained_zero_frequency_covariance_eigenvalues: tuple[float, ...]
    all_zero_frequency_covariance_eigenvalues: tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryMoriClosureAuditInput:
    sample_id: str
    basis_feature_time_series: np.ndarray
    current_time_series: np.ndarray
    time_step_s: float
    maximum_lag_steps: int
    beta_over_volume: float
    sigma_property_db_mS_cm: float
    sigma_recipe_mori_mS_cm: float
    gram_relative_tolerance: float = 1.0e-10
    zero_frequency_relative_tolerance: float = 1.0e-8


@dataclass(frozen=True)
class TrajectoryMoriClosureAuditRow:
    sample_id: str
    sigma_property_db_mS_cm: float
    sigma_raw_gk_mS_cm: float
    sigma_projected_gk_mS_cm: float
    sigma_mori_oracle_mS_cm: float
    sigma_recipe_mori_mS_cm: float
    projection_gap_mS_cm: float
    recipe_gap_mS_cm: float
    label_gap_mS_cm: float
    gram_rank: int
    kz_rank: int
    min_kz_eigenvalue: float
    dropped_gram_modes: int
    dropped_kz_modes: int


def compute_projected_mori_conductivity(
    mori_input: ProjectedMoriConductivityInput,
) -> ProjectedMoriConductivityResult:
    """Evaluate beta/(3V) sum_axis h_axis^T A_M^# h_axis."""

    direct_energy_matrix = _validated_square_matrix(
        mori_input.direct_energy_matrix,
        "direct_energy_matrix",
    )
    memory_self_energy_matrix = _validated_square_matrix(
        mori_input.memory_self_energy_matrix,
        "memory_self_energy_matrix",
    )
    if direct_energy_matrix.shape != memory_self_energy_matrix.shape:
        raise ValueError(
            "direct_energy_matrix and memory_self_energy_matrix must have the same shape"
        )
    _validate_symmetric_matrix(direct_energy_matrix, "direct_energy_matrix")
    _validate_symmetric_matrix(memory_self_energy_matrix, "memory_self_energy_matrix")
    _validate_positive_semidefinite_matrix(direct_energy_matrix, "direct_energy_matrix")
    _validate_positive_semidefinite_matrix(
        memory_self_energy_matrix, "memory_self_energy_matrix"
    )

    current_coupling_matrix = _validated_current_coupling_matrix(
        mori_input.current_coupling_matrix,
        direct_energy_matrix.shape[0],
    )
    _assert_positive_finite(mori_input.beta_over_volume, "beta_over_volume")

    effective_energy_matrix = direct_energy_matrix + memory_self_energy_matrix
    _validate_symmetric_matrix(effective_energy_matrix, "effective_energy_matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(effective_energy_matrix)
    _validate_eigenvalues_are_psd(eigenvalues, "effective_energy_matrix")

    quadratic_forms: list[float] = []
    axis_conductivities: list[float] = []
    for axis_index in range(current_coupling_matrix.shape[0]):
        axis_current_coupling = current_coupling_matrix[axis_index, :]
        quadratic_form = _projected_quadratic_form(
            eigenvalues,
            eigenvectors,
            axis_current_coupling,
            f"axis_{axis_index}",
        )
        quadratic_forms.append(quadratic_form)
        axis_conductivities.append(
            mori_input.beta_over_volume * quadratic_form / CARTESIAN_AXIS_COUNT
        )

    sigma_S_m = math.fsum(axis_conductivities)
    _assert_nonnegative_finite(sigma_S_m, "sigma_S_m")
    return ProjectedMoriConductivityResult(
        sigma_S_m=sigma_S_m,
        sigma_mS_cm=sigma_S_m * S_M_TO_MS_CM,
        axis_conductivity_S_m=(
            axis_conductivities[0],
            axis_conductivities[1],
            axis_conductivities[2],
        ),
        quadratic_form_by_axis=(
            quadratic_forms[0],
            quadratic_forms[1],
            quadratic_forms[2],
        ),
        energy_eigenvalues=tuple(float(eigenvalue) for eigenvalue in eigenvalues),
        effective_energy_matrix=effective_energy_matrix.copy(),
    )


def compare_mori_oracle_to_trajectory(
    mori_result: ProjectedMoriConductivityResult,
    trajectory_sigma_mS_cm: float,
    tolerance_mS_cm: float,
) -> MoriOracleClosureComparison:
    """Compare a projected Mori oracle result to trajectory Green-Kubo conductivity."""

    _assert_nonnegative_finite(mori_result.sigma_mS_cm, "mori_result.sigma_mS_cm")
    _assert_nonnegative_finite(trajectory_sigma_mS_cm, "trajectory_sigma_mS_cm")
    _assert_positive_finite(tolerance_mS_cm, "tolerance_mS_cm")
    closure_gap_mS_cm = float(abs(mori_result.sigma_mS_cm - trajectory_sigma_mS_cm))
    return MoriOracleClosureComparison(
        sigma_oracle_S_m=mori_result.sigma_S_m,
        sigma_oracle_mS_cm=mori_result.sigma_mS_cm,
        sigma_trajectory_mS_cm=trajectory_sigma_mS_cm,
        closure_gap_mS_cm=closure_gap_mS_cm,
        tolerance_mS_cm=tolerance_mS_cm,
        passes_tolerance=bool(closure_gap_mS_cm < tolerance_mS_cm),
    )


def build_trajectory_projected_mori_input(
    basis_feature_time_series: np.ndarray,
    current_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
    beta_over_volume: float,
    gram_relative_tolerance: float = 1.0e-10,
    zero_frequency_relative_tolerance: float = 1.0e-8,
) -> TrajectoryProjectedMoriConstruction:
    """Build projected Mori matrices from one centered trajectory current process."""

    feature_array = _validated_feature_time_series(basis_feature_time_series)
    current_array = _validated_current_time_series(
        current_time_series, "current_time_series"
    )
    if feature_array.shape[0] != current_array.shape[0]:
        raise ValueError(
            "basis_feature_time_series and current_time_series must have the same frame count"
        )
    _assert_positive_finite(time_step_s, "time_step_s")
    _assert_positive_finite(beta_over_volume, "beta_over_volume")
    _assert_positive_finite(gram_relative_tolerance, "gram_relative_tolerance")
    _assert_positive_finite(
        zero_frequency_relative_tolerance,
        "zero_frequency_relative_tolerance",
    )
    retained_maximum_lag_steps = _validated_maximum_lag_steps(
        maximum_lag_steps,
        current_array.shape[0],
    )

    centered_features = feature_array - np.mean(feature_array, axis=0, keepdims=True)
    centered_current = current_array - np.mean(current_array, axis=0, keepdims=True)

    frame_count = centered_features.shape[0]
    gram_matrix = _symmetrized_matrix(
        centered_features.T @ centered_features / frame_count
    )
    retained_gram_eigenvalues, retained_gram_eigenvectors, _ = _eigh_retained_psd(
        gram_matrix,
        gram_relative_tolerance,
        "trajectory feature Gram matrix",
    )
    if retained_gram_eigenvalues.size == 0:
        raise ValueError("all trajectory basis features are null under the Gram matrix")

    inverse_sqrt_gram_eigenvalues = np.diag(1.0 / np.sqrt(retained_gram_eigenvalues))
    whitening_matrix = retained_gram_eigenvectors @ inverse_sqrt_gram_eigenvalues
    whitened_basis_time_series = centered_features @ whitening_matrix

    current_coupling_matrix = (
        centered_current.T @ whitened_basis_time_series / frame_count
    )
    integrated_basis_covariance = _integrated_symmetrized_basis_covariance(
        whitened_basis_time_series,
        time_step_s,
        retained_maximum_lag_steps,
    )
    (
        retained_zero_frequency_covariance_eigenvalues,
        retained_zero_frequency_covariance_eigenvectors,
        all_zero_frequency_covariance_eigenvalues,
    ) = _eigh_retained_psd(
        integrated_basis_covariance,
        zero_frequency_relative_tolerance,
        "integrated projected covariance",
    )
    if retained_zero_frequency_covariance_eigenvalues.size == 0:
        raise ValueError(
            "projected basis has no positive zero-frequency covariance modes"
        )

    memory_self_energy_matrix = np.diag(
        1.0 / retained_zero_frequency_covariance_eigenvalues
    )
    retained_current_coupling_matrix = (
        current_coupling_matrix @ retained_zero_frequency_covariance_eigenvectors
    )
    direct_energy_matrix = np.zeros_like(memory_self_energy_matrix)

    retained_basis_time_series = (
        whitened_basis_time_series @ retained_zero_frequency_covariance_eigenvectors
    )
    projected_current_time_series = (
        retained_basis_time_series @ retained_current_coupling_matrix.T
    )
    raw_axis_integrals = green_kubo_axis_integrals_from_current_time_series(
        centered_current,
        time_step_s,
        retained_maximum_lag_steps,
        center_current=False,
    )
    projected_axis_integrals = green_kubo_axis_integrals_from_current_time_series(
        projected_current_time_series,
        time_step_s,
        retained_maximum_lag_steps,
        center_current=True,
    )
    raw_green_kubo_sigma_mS_cm = _green_kubo_sigma_mS_cm_from_axis_integrals(
        raw_axis_integrals,
        beta_over_volume,
    )
    projected_green_kubo_sigma_mS_cm = _green_kubo_sigma_mS_cm_from_axis_integrals(
        projected_axis_integrals,
        beta_over_volume,
    )

    return TrajectoryProjectedMoriConstruction(
        mori_input=ProjectedMoriConductivityInput(
            direct_energy_matrix=direct_energy_matrix,
            memory_self_energy_matrix=memory_self_energy_matrix,
            current_coupling_matrix=retained_current_coupling_matrix,
            beta_over_volume=beta_over_volume,
        ),
        centered_current_time_series=centered_current.copy(),
        whitened_basis_time_series=whitened_basis_time_series.copy(),
        retained_basis_time_series=retained_basis_time_series.copy(),
        projected_current_time_series=projected_current_time_series.copy(),
        raw_green_kubo_sigma_mS_cm=raw_green_kubo_sigma_mS_cm,
        projected_green_kubo_sigma_mS_cm=projected_green_kubo_sigma_mS_cm,
        raw_green_kubo_axis_integrals=raw_axis_integrals,
        projected_green_kubo_axis_integrals=projected_axis_integrals,
        maximum_lag_steps_used=retained_maximum_lag_steps,
        retained_gram_eigenvalues=tuple(
            float(eigenvalue) for eigenvalue in retained_gram_eigenvalues
        ),
        retained_zero_frequency_covariance_eigenvalues=tuple(
            float(eigenvalue)
            for eigenvalue in retained_zero_frequency_covariance_eigenvalues
        ),
        all_zero_frequency_covariance_eigenvalues=tuple(
            float(eigenvalue)
            for eigenvalue in all_zero_frequency_covariance_eigenvalues
        ),
    )


def build_trajectory_mori_closure_audit_row(
    closure_input: TrajectoryMoriClosureAuditInput,
) -> TrajectoryMoriClosureAuditRow:
    """Build the trajectory/property/recipe Mori closure decomposition row."""

    validated_sample_id = _validated_sample_id(closure_input.sample_id)
    feature_array = _validated_feature_time_series(
        closure_input.basis_feature_time_series
    )
    _assert_nonnegative_finite(
        closure_input.sigma_property_db_mS_cm,
        "sigma_property_db_mS_cm",
    )
    _assert_nonnegative_finite(
        closure_input.sigma_recipe_mori_mS_cm,
        "sigma_recipe_mori_mS_cm",
    )

    construction = build_trajectory_projected_mori_input(
        basis_feature_time_series=feature_array,
        current_time_series=closure_input.current_time_series,
        time_step_s=closure_input.time_step_s,
        maximum_lag_steps=closure_input.maximum_lag_steps,
        beta_over_volume=closure_input.beta_over_volume,
        gram_relative_tolerance=closure_input.gram_relative_tolerance,
        zero_frequency_relative_tolerance=(
            closure_input.zero_frequency_relative_tolerance
        ),
    )
    mori_oracle_result = compute_projected_mori_conductivity(construction.mori_input)

    raw_feature_count = int(feature_array.shape[1])
    gram_rank = len(construction.retained_gram_eigenvalues)
    kz_rank = len(construction.retained_zero_frequency_covariance_eigenvalues)
    min_kz_eigenvalue = float(
        min(construction.all_zero_frequency_covariance_eigenvalues)
    )
    dropped_gram_modes = raw_feature_count - gram_rank
    dropped_kz_modes = gram_rank - kz_rank
    if dropped_gram_modes < 0:
        raise ValueError("dropped_gram_modes cannot be negative")
    if dropped_kz_modes < 0:
        raise ValueError("dropped_kz_modes cannot be negative")

    return TrajectoryMoriClosureAuditRow(
        sample_id=validated_sample_id,
        sigma_property_db_mS_cm=float(closure_input.sigma_property_db_mS_cm),
        sigma_raw_gk_mS_cm=construction.raw_green_kubo_sigma_mS_cm,
        sigma_projected_gk_mS_cm=construction.projected_green_kubo_sigma_mS_cm,
        sigma_mori_oracle_mS_cm=mori_oracle_result.sigma_mS_cm,
        sigma_recipe_mori_mS_cm=float(closure_input.sigma_recipe_mori_mS_cm),
        projection_gap_mS_cm=(
            construction.projected_green_kubo_sigma_mS_cm
            - construction.raw_green_kubo_sigma_mS_cm
        ),
        recipe_gap_mS_cm=(
            float(closure_input.sigma_recipe_mori_mS_cm)
            - mori_oracle_result.sigma_mS_cm
        ),
        label_gap_mS_cm=(
            float(closure_input.sigma_property_db_mS_cm)
            - construction.raw_green_kubo_sigma_mS_cm
        ),
        gram_rank=gram_rank,
        kz_rank=kz_rank,
        min_kz_eigenvalue=min_kz_eigenvalue,
        dropped_gram_modes=dropped_gram_modes,
        dropped_kz_modes=dropped_kz_modes,
    )


def green_kubo_axis_integrals_from_current_time_series(
    current_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
    center_current: bool,
) -> tuple[float, float, float]:
    """Compute one-sided trapezoidal current autocorrelation integrals by axis."""

    current_array = _validated_current_time_series(
        current_time_series, "current_time_series"
    )
    _assert_positive_finite(time_step_s, "time_step_s")
    retained_maximum_lag_steps = _validated_maximum_lag_steps(
        maximum_lag_steps,
        current_array.shape[0],
    )
    if center_current:
        centered_current = current_array - np.mean(current_array, axis=0, keepdims=True)
    else:
        centered_current = current_array

    axis_integrals = np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float)
    frame_count = centered_current.shape[0]
    for lag_steps in range(retained_maximum_lag_steps + 1):
        lagged_covariance_by_axis = np.sum(
            centered_current[lag_steps:, :]
            * centered_current[: frame_count - lag_steps, :],
            axis=0,
        ) / (frame_count - lag_steps)
        lag_weight = _trapezoid_lag_weight(lag_steps, retained_maximum_lag_steps)
        axis_integrals += lag_weight * lagged_covariance_by_axis

    axis_integrals = axis_integrals * time_step_s
    if not np.all(np.isfinite(axis_integrals)):
        raise ValueError("Green-Kubo axis integrals contain non-finite values")
    return (
        float(axis_integrals[0]),
        float(axis_integrals[1]),
        float(axis_integrals[2]),
    )


def _projected_quadratic_form(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    current_coupling: np.ndarray,
    context: str,
) -> float:
    projected_current = eigenvectors.T @ current_coupling
    positive_mode_mask = _positive_energy_mode_mask(eigenvalues)
    null_mode_projection = projected_current[~positive_mode_mask]
    null_projection_norm = float(np.linalg.norm(null_mode_projection))
    current_norm = float(np.linalg.norm(current_coupling))
    allowed_null_projection = MORI_NUMERICAL_TOLERANCE * max(
        current_norm,
        MORI_NUMERICAL_TOLERANCE,
    )
    if null_projection_norm > allowed_null_projection:
        raise ValueError(
            f"{context} current_coupling projects onto a null energy mode; "
            "zero-frequency conductivity is not finite in this projected basis"
        )
    if not np.any(positive_mode_mask):
        return 0.0
    positive_projected_current = projected_current[positive_mode_mask]
    positive_eigenvalues = eigenvalues[positive_mode_mask]
    quadratic_form = float(
        math.fsum(
            float(projected_value * projected_value / eigenvalue)
            for projected_value, eigenvalue in zip(
                positive_projected_current,
                positive_eigenvalues,
            )
        )
    )
    _assert_nonnegative_finite(quadratic_form, f"{context}.quadratic_form")
    return quadratic_form


def _validated_feature_time_series(feature_time_series: np.ndarray) -> np.ndarray:
    feature_array = np.asarray(feature_time_series, dtype=float)
    if feature_array.ndim != 2:
        raise ValueError(
            "basis_feature_time_series must have shape (n_frames, n_basis_raw)"
        )
    if feature_array.shape[0] < 2:
        raise ValueError("basis_feature_time_series must contain at least two frames")
    if feature_array.shape[1] == 0:
        raise ValueError("basis_feature_time_series must contain at least one feature")
    if not np.all(np.isfinite(feature_array)):
        raise ValueError("basis_feature_time_series contains non-finite values")
    return feature_array


def _validated_current_time_series(
    current_time_series: np.ndarray,
    name: str,
) -> np.ndarray:
    current_array = np.asarray(current_time_series, dtype=float)
    if current_array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_frames, 3)")
    expected_axis_count = int(CARTESIAN_AXIS_COUNT)
    if current_array.shape[1] != expected_axis_count:
        raise ValueError(f"{name} must have shape (n_frames, 3)")
    if current_array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two frames")
    if not np.all(np.isfinite(current_array)):
        raise ValueError(f"{name} contains non-finite values")
    return current_array


def _validated_maximum_lag_steps(
    maximum_lag_steps: int,
    frame_count: int,
) -> int:
    if not isinstance(maximum_lag_steps, int):
        raise TypeError("maximum_lag_steps must be an integer")
    if maximum_lag_steps < 0:
        raise ValueError("maximum_lag_steps must be nonnegative")
    return min(maximum_lag_steps, frame_count - 1)


def _symmetrized_matrix(matrix: np.ndarray) -> np.ndarray:
    return TRAPEZOID_ENDPOINT_WEIGHT * (matrix + matrix.T)


def _eigh_retained_psd(
    matrix: np.ndarray,
    relative_tolerance: float,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symmetrized_matrix = _symmetrized_matrix(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized_matrix)
    eigenvalue_scale = _eigenvalue_scale(eigenvalues)
    allowed_negative_eigenvalue = relative_tolerance * eigenvalue_scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(
            f"{context} has significant negative eigenvalues; "
            f"minimum={minimum_eigenvalue}, allowed={allowed_negative_eigenvalue}"
        )
    retained_mode_mask = eigenvalues > relative_tolerance * eigenvalue_scale
    return (
        eigenvalues[retained_mode_mask],
        eigenvectors[:, retained_mode_mask],
        eigenvalues,
    )


def _integrated_symmetrized_basis_covariance(
    whitened_basis_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
) -> np.ndarray:
    basis_dimension = whitened_basis_time_series.shape[1]
    frame_count = whitened_basis_time_series.shape[0]
    integrated_covariance = np.zeros((basis_dimension, basis_dimension), dtype=float)
    for lag_steps in range(maximum_lag_steps + 1):
        lagged_covariance = (
            whitened_basis_time_series[lag_steps:, :].T
            @ whitened_basis_time_series[: frame_count - lag_steps, :]
            / (frame_count - lag_steps)
        )
        lag_weight = _trapezoid_lag_weight(lag_steps, maximum_lag_steps)
        integrated_covariance += lag_weight * _symmetrized_matrix(lagged_covariance)
    return _symmetrized_matrix(integrated_covariance * time_step_s)


def _trapezoid_lag_weight(
    lag_steps: int,
    maximum_lag_steps: int,
) -> float:
    if lag_steps == 0 or lag_steps == maximum_lag_steps:
        return TRAPEZOID_ENDPOINT_WEIGHT
    return 1.0


def _green_kubo_sigma_mS_cm_from_axis_integrals(
    axis_integrals: tuple[float, float, float],
    beta_over_volume: float,
) -> float:
    sigma_S_m = beta_over_volume * math.fsum(axis_integrals) / CARTESIAN_AXIS_COUNT
    _assert_nonnegative_finite(sigma_S_m, "green_kubo_sigma_S_m")
    return sigma_S_m * S_M_TO_MS_CM


def _validated_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix_array.shape}")
    if matrix_array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one basis function")
    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix_array


def _validated_current_coupling_matrix(
    current_coupling_matrix: np.ndarray,
    basis_size: int,
) -> np.ndarray:
    current_coupling_array = np.asarray(current_coupling_matrix, dtype=float)
    if current_coupling_array.ndim != 2:
        raise ValueError("current_coupling_matrix must be a two-dimensional matrix")
    expected_shape = (int(CARTESIAN_AXIS_COUNT), basis_size)
    if current_coupling_array.shape != expected_shape:
        raise ValueError(
            "current_coupling_matrix must have shape "
            f"{expected_shape}, got {current_coupling_array.shape}"
        )
    if not np.all(np.isfinite(current_coupling_array)):
        raise ValueError("current_coupling_matrix contains non-finite values")
    return current_coupling_array


def _validate_symmetric_matrix(matrix: np.ndarray, name: str) -> None:
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=MORI_NUMERICAL_TOLERANCE,
        atol=MORI_NUMERICAL_TOLERANCE,
    ):
        raise ValueError(f"{name} must be symmetric")


def _validate_positive_semidefinite_matrix(matrix: np.ndarray, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    _validate_eigenvalues_are_psd(eigenvalues, name)


def _validate_eigenvalues_are_psd(eigenvalues: np.ndarray, name: str) -> None:
    minimum_eigenvalue = float(np.min(eigenvalues))
    allowed_negative_eigenvalue = MORI_NUMERICAL_TOLERANCE * _eigenvalue_scale(
        eigenvalues
    )
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(
            f"{name} must be positive semidefinite; minimum eigenvalue is "
            f"{minimum_eigenvalue}"
        )


def _positive_energy_mode_mask(eigenvalues: np.ndarray) -> np.ndarray:
    positive_threshold = MORI_NUMERICAL_TOLERANCE * _eigenvalue_scale(eigenvalues)
    return eigenvalues > positive_threshold


def _validated_sample_id(sample_id: str) -> str:
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a string")
    stripped_sample_id = sample_id.strip()
    if stripped_sample_id == "":
        raise ValueError("sample_id must not be empty")
    return stripped_sample_id


def _eigenvalue_scale(eigenvalues: np.ndarray) -> float:
    return max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)


def _assert_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _assert_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value}")


# ---- finite_markov_additive_green_kubo.py ----

HALF_JUMP_VARIANCE_FACTOR = 0.5
ZERO_VALUE = 0.0
FINITE_MARKOV_ADDITIVE_TOLERANCE = math.sqrt(np.finfo(float).eps)
ZERO_SECOND_MOMENT_TENSOR_M2: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class MarkovAdditiveEvent:
    from_state_index: int
    to_state_index: int
    rate_s_inv: float
    charge_displacement_m: tuple[float, float, float]
    charge_displacement_second_moment_m2: tuple[tuple[float, float, float], ...]
    label: str
    family_label: str


@dataclass(frozen=True)
class MarkovAdditiveEventFamilyAttribution:
    family_label: str
    direct_sigma_mS_cm: float
    self_corrector_sigma_mS_cm: float
    marginal_corrector_sigma_mS_cm: float
    marginal_net_sigma_mS_cm: float
    direct_fraction: float
    marginal_net_fraction: float


@dataclass(frozen=True)
class ProjectedCurrentMemoryCorrection:
    state_label: str
    transport_state_label: str
    memory_family_label: str
    concentration_mol_m3: float
    memory_self_energy_s_inv: float
    correction_axis_density_m2_s_mol_m3: tuple[float, float, float]
    correction_sigma_S_m: float
    correction_sigma_mS_cm: float
    source: str


@dataclass(frozen=True)
class AtmosphereMoriCorrection:
    state_label: str
    transport_state_label: str
    charge_number: int
    concentration_mol_m3: float
    D_short_m2_s: float
    zeta_bare_kg_s: float
    zeta_ep_kg_s: float
    zeta_rel_kg_s: float
    D_long_m2_s: float
    D_atmosphere_correction_m2_s: float
    memory_self_energy_s_inv: float
    correction_axis_density_m2_s_mol_m3: tuple[float, float, float]
    correction_sigma_S_m: float
    correction_sigma_mS_cm: float
    source: str


@dataclass(frozen=True)
class OnsagerFrictionEdge:
    first_state_label: str
    second_state_label: str
    friction_coefficient_J_s_mol_m2: float
    source: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class OnsagerTransportOperator:
    state_labels: tuple[str, ...]
    charge_numbers: tuple[int, ...]
    concentrations_mol_m3: tuple[float, ...]
    bare_diffusivities_m2_s: tuple[float, ...]
    diagonal_friction_J_s_mol_m2: tuple[float, ...]
    friction_edges: tuple[OnsagerFrictionEdge, ...]
    friction_matrix: tuple[tuple[float, ...], ...]
    projected_mobility_matrix: tuple[tuple[float, ...], ...]
    nernst_einstein_sigma_mS_cm: float
    onsager_sigma_mS_cm: float
    correlation_corrector_mS_cm: float


@dataclass(frozen=True)
class PMFPartitionModel:
    state_label: str
    reaction_coordinate: str
    lower_bound: float
    upper_bound: float
    pmf_terms: tuple[MolecularFreeEnergyTermDerivationUx, ...]
    restricted_partition_weight: float
    concentration_mol_m3: float


@dataclass(frozen=True)
class ReactiveFluxModel:
    from_state: str
    to_state: str
    reaction_coordinate: str
    free_energy_profile: tuple[float, ...]
    coordinate_diffusion_profile: tuple[float, ...]
    symmetric_flux_mol_m3_s: float


@dataclass(frozen=True)
class ConditionalDisplacementMomentModel:
    from_state: str
    to_state: str
    mean_displacement_m: tuple[float, float, float]
    second_moment_m2: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class SelfCurrentTensorModel:
    state_label: str
    diffusion_tensor_m2_s: tuple[tuple[float, float, float], ...]
    concentration_mol_m3: float


@dataclass(frozen=True)
class OnsagerMaxwellStefanOperator:
    state_labels: tuple[str, ...]
    concentrations_mol_m3: tuple[float, ...]
    charges: tuple[int, ...]
    self_frictions_J_s_mol_m2: tuple[float, ...]
    pair_frictions: tuple[OnsagerFrictionEdge, ...]
    friction_matrix: tuple[tuple[float, ...], ...]
    mobility_matrix: tuple[tuple[float, ...], ...]
    sigma_onsager_mS_cm: float


PROJECTED_GENERATOR_CLASS_RESTRICTED_POPULATION = "restricted_population_state"
PROJECTED_GENERATOR_CLASS_REACTIVE_FLUX = "reactive_flux_boundary"
PROJECTED_GENERATOR_CLASS_STATE_CHANGING_DISPLACEMENT = (
    "state_changing_displacement_event"
)
PROJECTED_GENERATOR_CLASS_WITHIN_STATE_SELF_CURRENT = "within_state_self_current"
PROJECTED_GENERATOR_CLASS_MORI_MEMORY_BASIS = "mori_memory_basis"
PROJECTED_GENERATOR_CLASS_DIAGNOSTIC_ONLY = "diagnostic_only_label"
SUPPORTED_PROJECTED_GENERATOR_CONTRIBUTION_CLASSES = (
    PROJECTED_GENERATOR_CLASS_RESTRICTED_POPULATION,
    PROJECTED_GENERATOR_CLASS_REACTIVE_FLUX,
    PROJECTED_GENERATOR_CLASS_STATE_CHANGING_DISPLACEMENT,
    PROJECTED_GENERATOR_CLASS_WITHIN_STATE_SELF_CURRENT,
    PROJECTED_GENERATOR_CLASS_MORI_MEMORY_BASIS,
    PROJECTED_GENERATOR_CLASS_DIAGNOSTIC_ONLY,
)


@dataclass(frozen=True)
class ProjectedGeneratorContributionClassification:
    label: str
    contribution_class: str
    source: str
    conductivity_contribution_allowed: bool
    theorem_object: str


@dataclass(frozen=True)
class UxLxSource:
    equilibrium_measure_source: str
    reversible_generator_source: str
    microscopic_configuration_space: str
    stationary_measure: str
    generator_formula_id: str
    descriptor_complete: bool


@dataclass(frozen=True)
class ChargePolarizationSource:
    observable_name: str
    formula_id: str
    coordinate_convention: str
    units: str


@dataclass(frozen=True)
class ProjectedBasis:
    basis_labels: tuple[str, ...]
    basis_sources: tuple[str, ...]
    refinable_hierarchy: str


@dataclass(frozen=True)
class RestrictedPopulationSet:
    state_labels: tuple[str, ...]
    concentrations_mol_m3: tuple[float, ...]
    stationary_probabilities: tuple[float, ...]
    partition_models: tuple[PMFPartitionModel, ...]


@dataclass(frozen=True)
class SymmetricFluxSet:
    reactive_fluxes: tuple[ProjectedReactiveFluxIntegral, ...]


@dataclass(frozen=True)
class ConditionalMomentSet:
    displacement_moments: tuple[ProjectedReactiveFluxIntegral, ...]


@dataclass(frozen=True)
class SelfCurrentSet:
    self_current_tensors: tuple[ProjectedSelfDisplacementMoment, ...]
    direct_axis_density_m2_s_mol_m3: tuple[float, float, float]


@dataclass(frozen=True)
class MoriMatrix:
    direct_energy_matrix: tuple[tuple[float, ...], ...]
    memory_self_energy_matrix: tuple[tuple[float, ...], ...]
    beta_over_volume: float


@dataclass(frozen=True)
class CurrentCoupling:
    current_coupling_matrix: tuple[tuple[float, ...], ...]
    axis_count: int


@dataclass(frozen=True)
class ProjectedGKConductivity:
    direct_sigma_mS_cm: float
    corrector_sigma_mS_cm: float
    sigma_mS_cm: float
    theorem_id: str


@dataclass(frozen=True)
class ProjectedGeneratorModel:
    microscopic_generator_source: UxLxSource
    charge_polarization_source: ChargePolarizationSource
    basis: ProjectedBasis
    populations: RestrictedPopulationSet
    reactive_fluxes: SymmetricFluxSet
    displacement_moments: ConditionalMomentSet
    self_current_tensors: SelfCurrentSet
    mori_matrix: MoriMatrix
    current_coupling: CurrentCoupling
    conductivity: ProjectedGKConductivity
    contribution_classifications: tuple[
        ProjectedGeneratorContributionClassification, ...
    ]


_FORBIDDEN_SELF_TRANSLATION_FAMILIES = {
    "ordinary_mobile_translation",
    "ordinary_free_Li_translation",
    "ordinary_free_anion_translation",
    "bound_state_translation",
}


@dataclass(frozen=True)
class MarkovAdditiveConductivityInput:
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: np.ndarray
    events: tuple[MarkovAdditiveEvent, ...]
    temperature_K: float


@dataclass(frozen=True)
class ConcentrationGeneratorValidation:
    row_sum_residual: float
    stationary_residual_mol_m3_s: float
    detailed_balance_residual_mol_m3_s: float
    minimum_offdiagonal_rate_s_inv: float
    concentration_sum_mol_m3: float


@dataclass(frozen=True)
class MarkovAdditiveConductivityResult:
    generator_s_inv: np.ndarray
    validation: ConcentrationGeneratorValidation
    event_reversal_residual_mol_m3_s: float
    drift_by_state_m_s: np.ndarray
    corrector_mori_input: ProjectedMoriConductivityInput
    corrector_mori_result: ProjectedMoriConductivityResult
    direct_axis_density_m2_s_mol_m3: tuple[float, float, float]
    event_mori_corrector_axis_density_m2_s_mol_m3: tuple[float, float, float]
    atmosphere_corrector_axis_density_m2_s_mol_m3: tuple[float, float, float]
    projected_current_memory_corrector_axis_density_m2_s_mol_m3: tuple[
        float,
        float,
        float,
    ]
    corrector_axis_density_m2_s_mol_m3: tuple[float, float, float]
    effective_axis_density_m2_s_mol_m3: tuple[float, float, float]
    direct_sigma_S_m: float
    event_mori_corrector_sigma_S_m: float
    atmosphere_corrector_sigma_S_m: float
    projected_current_memory_corrector_sigma_S_m: float
    corrector_sigma_S_m: float
    sigma_S_m: float
    direct_sigma_mS_cm: float
    event_mori_corrector_sigma_mS_cm: float
    atmosphere_corrector_sigma_mS_cm: float
    projected_current_memory_corrector_sigma_mS_cm: float
    corrector_sigma_mS_cm: float
    sigma_mS_cm: float
    minimum_effective_axis_density_m2_s_mol_m3: float


@dataclass(frozen=True)
class MolecularFreeEnergyTermDerivationUx:
    """One term in the descriptor equilibrium free-energy functional U_x."""

    term_name: str
    formula_id: str
    parameter_names: tuple[str, ...]
    units: str
    sign_convention: str
    theorem_role: str
    empirical_closure_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class MolecularFreeEnergyFunctionalUx:
    """Descriptor equilibrium functional used for restricted populations."""

    formula_id: str
    temperature_K: float
    beta_mol_per_J: float
    partition_weight_formula_id: str
    terms: tuple[MolecularFreeEnergyTermDerivationUx, ...]
    empirical_closure_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class TransportPartitionDefinitionAi:
    """Measurable projected transport partition definition A_i(x)."""

    state_label: str
    partition_label: str
    predicate_id: str
    center_species_name: str
    center_charge_number: int
    parent_cluster_label: str
    parent_cluster_kind: str
    transport_role: str
    normalization_rule: str
    disjointness_rule: str
    source_parameter_names: tuple[str, ...]
    empirical_closure_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ReactiveFluxDerivationKij:
    """Derivation of one symmetric projected reactive flux K_ijr."""

    from_state_label: str
    to_state_label: str
    family_label: str
    formula_id: str
    symmetric_flux_formula_id: str
    generator_rate_formula_id: str
    symmetric_flux_mol_m3_s: float
    forward_rate_s_inv: float
    reverse_rate_s_inv: float
    detailed_balance_condition: str
    mobility_source: str
    parameter_names: tuple[str, ...]
    empirical_closure_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ChargeDisplacementMomentDerivationDij:
    """Derivation of one conditional charge-displacement moment d_ijr."""

    from_state_label: str
    to_state_label: str
    family_label: str
    formula_id: str
    moment_definition: str
    charge_displacement_m: tuple[float, float, float]
    units: str
    reverse_displacement_rule: str
    parameter_names: tuple[str, ...]
    empirical_closure_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class DerivedReactiveFluxModel:
    """Symmetric reactive-flux model that generates reversible Q_ij rates."""

    from_partition: str
    to_partition: str
    transition_surface: str
    free_energy_barrier_over_RT: float
    partition_gap_scale_over_RT: float
    free_energy_mismatch_over_RT: float
    effective_diffusivity_m2_s: float
    hop_length_m: float
    diffusion_tensor_source: str
    friction_source: str
    transmission_coefficient: float
    symmetric_flux_mol_m3_s: float
    forward_rate_s_inv: float
    reverse_rate_s_inv: float
    detailed_balance_condition: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ConditionalChargeDisplacementModel:
    """Conditional charge-displacement moment model for one channel."""

    from_partition: str
    to_partition: str
    family_label: str
    mean_displacement_m: tuple[float, float, float]
    second_moment_m2: tuple[tuple[float, float, float], ...]
    reverse_rule: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class SelfCurrentProjectionModel:
    """Within-partition self-current contribution for direct conductivity."""

    partition: str
    family_label: str
    mean_displacement_m: tuple[float, float, float]
    self_displacement_tensor_m2_s: tuple[tuple[float, float, float], ...]
    direct_axis_density_m2_s_mol_m3: tuple[float, float, float]
    source: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class MoriBasisFunction:
    """Derived observable used in the finite Mori projection basis."""

    label: str
    state_label: str
    observable_definition: str
    source: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ProjectedTransportPartitionState:
    """Restricted equilibrium population for one finite transport partition."""

    state_index: int
    state_label: str
    concentration_mol_m3: float
    stationary_probability: float
    restricted_partition_weight: float
    partition_definition: str
    population_source: str
    partition_derivation: TransportPartitionDefinitionAi
    pmf_partition_model: PMFPartitionModel


@dataclass(frozen=True)
class ProjectedPrimitiveSourceReference:
    """Theorem role and parameter references for one primitive source."""

    theorem_role: str
    source_name: str
    source_parameter_names: tuple[str, ...]
    empirical_closure_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ProjectedPrimitiveClosureContract:
    """Proof boundary for descriptor-generated finite projection primitives."""

    readout_theorem: str
    equilibrium_measure_source: ProjectedPrimitiveSourceReference
    partition_source: ProjectedPrimitiveSourceReference
    concentration_source: ProjectedPrimitiveSourceReference
    reactive_flux_source: ProjectedPrimitiveSourceReference
    displacement_moment_source: ProjectedPrimitiveSourceReference
    descriptor_closure_derives_finite_projected_generator: bool
    descriptor_closure_derives_full_microscopic_generator: bool
    primitive_parameter_theorem_role_by_name: Mapping[str, str]


@dataclass(frozen=True)
class ProjectedReactiveFluxIntegral:
    """Finite-state reactive-flux integral for one reversible boundary."""

    from_state_index: int
    to_state_index: int
    from_state_label: str
    to_state_label: str
    family_label: str
    charge_displacement_m: tuple[float, float, float]
    charge_displacement_second_moment_m2: tuple[tuple[float, float, float], ...]
    symmetric_flux_mol_m3_s: float
    forward_rate_s_inv: float
    reverse_rate_s_inv: float
    detailed_balance_residual_mol_m3_s: float
    reactive_flux_source: str
    displacement_moment_source: str
    derived_reactive_flux_model: DerivedReactiveFluxModel
    conditional_displacement_model: ConditionalChargeDisplacementModel
    reactive_flux_model: ReactiveFluxModel
    conditional_displacement_moment_model: ConditionalDisplacementMomentModel
    reactive_flux_derivation: ReactiveFluxDerivationKij
    displacement_moment_derivation: ChargeDisplacementMomentDerivationDij


@dataclass(frozen=True)
class ProjectedSelfDisplacementMoment:
    """Within-partition charge displacement moment for direct conductivity."""

    state_index: int
    state_label: str
    family_label: str
    charge_displacement_m: tuple[float, float, float]
    charge_displacement_second_moment_m2: tuple[tuple[float, float, float], ...]
    symmetric_flux_mol_m3_s: float
    rate_s_inv: float
    direct_axis_density_m2_s_mol_m3: tuple[float, float, float]
    displacement_moment_source: str
    conditional_displacement_model: ConditionalChargeDisplacementModel
    self_current_projection_model: SelfCurrentProjectionModel
    self_current_tensor_model: SelfCurrentTensorModel
    displacement_moment_derivation: ChargeDisplacementMomentDerivationDij


@dataclass(frozen=True)
class ProjectedElectrolyteTransportModel:
    """Restricted-partition finite Markov-additive transport model."""

    proof_status: str
    closure_contract: ProjectedPrimitiveClosureContract
    free_energy_functional: MolecularFreeEnergyFunctionalUx
    projected_generator_model: ProjectedGeneratorModel
    projected_primitive_set: ProjectedPrimitiveSet
    contribution_classifications: tuple[
        ProjectedGeneratorContributionClassification, ...
    ]
    partition_states: tuple[ProjectedTransportPartitionState, ...]
    projected_transport_states: tuple["ProjectedTransportState", ...]
    state_labels: tuple[str, ...]
    stationary_concentrations_mol_m3: tuple[float, ...]
    stationary_probabilities: tuple[float, ...]
    restricted_partition_weights: tuple[float, ...]
    mori_basis_functions: tuple[MoriBasisFunction, ...]
    reactive_flux_integrals: tuple[ProjectedReactiveFluxIntegral, ...]
    self_displacement_moments: tuple[ProjectedSelfDisplacementMoment, ...]
    self_direct_axis_density_m2_s_mol_m3: tuple[float, float, float]
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ]
    atmosphere_mori_corrections: tuple[AtmosphereMoriCorrection, ...]
    onsager_transport_operator: OnsagerTransportOperator
    onsager_maxwell_stefan_operator: OnsagerMaxwellStefanOperator
    markov_additive_input: MarkovAdditiveConductivityInput
    conductivity_result: MarkovAdditiveConductivityResult


def build_generator_from_events(
    state_count: int,
    events: Sequence[MarkovAdditiveEvent],
) -> np.ndarray:
    """Build a CTMC generator from off-diagonal Markov-additive events."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    generator_matrix = np.zeros((state_count, state_count), dtype=float)
    for event in events:
        _validate_event_indices(event, state_count)
        _positive_float(event.rate_s_inv, f"{event.label}.rate_s_inv")
        validated_displacement = _validated_displacement(
            event.charge_displacement_m,
            event.label,
        )
        validated_second_moment = _validated_second_moment_tensor(
            event.charge_displacement_second_moment_m2,
            event.label,
        )
        _validate_second_moment_dominates_mean_outer_product(
            validated_displacement,
            validated_second_moment,
            event.label,
        )
        if event.from_state_index != event.to_state_index:
            generator_matrix[event.from_state_index, event.to_state_index] += (
                event.rate_s_inv
            )
    row_exit_rates = np.asarray(
        [
            math.fsum(
                float(generator_matrix[row_index, column_index])
                for column_index in range(state_count)
            )
            for row_index in range(state_count)
        ],
        dtype=float,
    )
    for row_index in range(state_count):
        generator_matrix[row_index, row_index] = -row_exit_rates[row_index]
        generator_matrix[row_index, row_index] -= math.fsum(
            float(generator_matrix[row_index, column_index])
            for column_index in range(state_count)
        )
    return generator_matrix


def validate_concentration_reversible_generator(
    generator_matrix_s_inv: np.ndarray,
    state_concentrations_mol_m3: np.ndarray,
) -> ConcentrationGeneratorValidation:
    """Validate row conservation, stationarity, and concentration detailed balance."""

    generator_matrix = _validated_generator_matrix(generator_matrix_s_inv)
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        generator_matrix.shape[0],
    )
    tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    row_sum_residual = float(
        max(
            abs(math.fsum(float(value) for value in generator_matrix[row_index, :]))
            for row_index in range(generator_matrix.shape[0])
        )
    )
    offdiagonal_rates = generator_matrix[~np.eye(generator_matrix.shape[0], dtype=bool)]
    minimum_offdiagonal_rate = (
        float(np.min(offdiagonal_rates)) if offdiagonal_rates.size else ZERO_VALUE
    )
    maximum_diagonal_entry = float(np.max(np.diag(generator_matrix)))
    stationary_residual = float(
        max(
            abs(
                math.fsum(
                    float(
                        state_concentrations[column_index]
                        * generator_matrix[column_index, row_index]
                    )
                    for column_index in range(generator_matrix.shape[0])
                )
            )
            for row_index in range(generator_matrix.shape[0])
        )
    )
    detailed_balance_residual = float(
        max(
            abs(
                math.fsum(
                    (
                        float(
                            state_concentrations[first_index]
                            * generator_matrix[first_index, second_index]
                        )
                        - float(
                            state_concentrations[second_index]
                            * generator_matrix[second_index, first_index]
                        )
                    )
                    for _ in (0,)
                )
            )
            for first_index in range(generator_matrix.shape[0])
            for second_index in range(generator_matrix.shape[1])
        )
    )
    if row_sum_residual > tolerance:
        raise ValueError(
            f"generator row-sum residual {row_sum_residual} exceeds {tolerance}"
        )
    if minimum_offdiagonal_rate < -tolerance:
        raise ValueError("generator off-diagonal entries must be nonnegative")
    if maximum_diagonal_entry > tolerance:
        raise ValueError("generator diagonal entries must be nonpositive")
    if stationary_residual > tolerance:
        raise ValueError(
            f"stationary concentration residual {stationary_residual} exceeds {tolerance}"
        )
    if detailed_balance_residual > tolerance:
        raise ValueError(
            f"detailed-balance residual {detailed_balance_residual} exceeds {tolerance}"
        )
    if row_sum_residual <= tolerance:
        row_sum_residual = 0.0
    if stationary_residual <= tolerance:
        stationary_residual = 0.0
    if detailed_balance_residual <= tolerance:
        detailed_balance_residual = 0.0
    return ConcentrationGeneratorValidation(
        row_sum_residual=row_sum_residual,
        stationary_residual_mol_m3_s=stationary_residual,
        detailed_balance_residual_mol_m3_s=detailed_balance_residual,
        minimum_offdiagonal_rate_s_inv=minimum_offdiagonal_rate,
        concentration_sum_mol_m3=float(np.sum(state_concentrations)),
    )


def validate_event_displacement_reversibility(
    events: Sequence[MarkovAdditiveEvent],
    state_concentrations_mol_m3: np.ndarray,
    state_count: int,
) -> float:
    """Validate reverse flux symmetry for every nonzero displacement event."""

    if len(events) == 0:
        return ZERO_VALUE
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        state_count,
    )
    weighted_flux_by_event_key: dict[
        tuple[
            int,
            int,
            tuple[float, float, float],
            tuple[tuple[float, float, float], ...],
        ],
        float,
    ] = {}
    for event in events:
        _validate_event_indices(event, state_count)
        event_rate_s_inv = _positive_float(
            event.rate_s_inv, f"{event.label}.rate_s_inv"
        )
        displacement_array = _validated_displacement(
            event.charge_displacement_m,
            event.label,
        )
        second_moment_key = _second_moment_key(
            _validated_second_moment_tensor(
                event.charge_displacement_second_moment_m2,
                event.label,
            )
        )
        _validate_second_moment_dominates_mean_outer_product(
            displacement_array,
            np.asarray(second_moment_key, dtype=float),
            event.label,
        )
        if (
            _is_zero_displacement(displacement_array)
            and event.from_state_index == event.to_state_index
        ):
            continue
        displacement_key = _displacement_key(displacement_array)
        event_key = (
            event.from_state_index,
            event.to_state_index,
            displacement_key,
            second_moment_key,
        )
        weighted_flux = state_concentrations[event.from_state_index] * event_rate_s_inv
        weighted_flux_by_event_key[event_key] = (
            weighted_flux_by_event_key.get(event_key, ZERO_VALUE) + weighted_flux
        )
    if not weighted_flux_by_event_key:
        return ZERO_VALUE
    maximum_weighted_flux = max(
        abs(weighted_flux) for weighted_flux in weighted_flux_by_event_key.values()
    )
    tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        maximum_weighted_flux,
    )
    maximum_reversal_residual = ZERO_VALUE
    for event_key, weighted_flux in weighted_flux_by_event_key.items():
        from_state_index, to_state_index, displacement_key, second_moment_key = (
            event_key
        )
        reverse_displacement_key = tuple(
            _canonical_float_for_key(-component) for component in displacement_key
        )
        reverse_key = (
            to_state_index,
            from_state_index,
            reverse_displacement_key,
            second_moment_key,
        )
        reverse_weighted_flux = weighted_flux_by_event_key.get(
            reverse_key,
            ZERO_VALUE,
        )
        reversal_residual = abs(weighted_flux - reverse_weighted_flux)
        maximum_reversal_residual = max(
            maximum_reversal_residual,
            reversal_residual,
        )
    if maximum_reversal_residual > tolerance:
        raise ValueError(
            "event displacement reverse residual "
            f"{maximum_reversal_residual} exceeds {tolerance}"
        )
    if maximum_reversal_residual <= tolerance:
        maximum_reversal_residual = 0.0
    return float(maximum_reversal_residual)


def compute_markov_additive_green_kubo_conductivity(
    markov_additive_input: MarkovAdditiveConductivityInput,
) -> MarkovAdditiveConductivityResult:
    """Evaluate sigma = direct jump variance minus Mori corrector."""

    state_labels = tuple(markov_additive_input.state_labels)
    if len(state_labels) == 0:
        raise ValueError("state_labels must contain at least one state")
    if len(set(state_labels)) != len(state_labels):
        raise ValueError("state_labels must be unique")
    state_concentrations = _validated_state_concentrations(
        markov_additive_input.state_concentrations_mol_m3,
        len(state_labels),
    )
    temperature_K = _positive_float(
        markov_additive_input.temperature_K, "temperature_K"
    )
    generator_matrix = build_generator_from_events(
        len(state_labels),
        markov_additive_input.events,
    )
    validation = validate_concentration_reversible_generator(
        generator_matrix,
        state_concentrations,
    )
    event_reversal_residual = validate_event_displacement_reversibility(
        markov_additive_input.events,
        state_concentrations,
        len(state_labels),
    )
    axis_count = int(CARTESIAN_AXIS_COUNT)
    direct_axis_density = np.zeros(axis_count, dtype=float)
    drift_by_state = np.zeros((len(state_labels), axis_count), dtype=float)
    for event in markov_additive_input.events:
        displacement_array = _validated_displacement(
            event.charge_displacement_m,
            event.label,
        )
        second_moment_tensor = _validated_second_moment_tensor(
            event.charge_displacement_second_moment_m2,
            event.label,
        )
        _validate_second_moment_dominates_mean_outer_product(
            displacement_array,
            second_moment_tensor,
            event.label,
        )
        second_moment_diagonal = np.diag(second_moment_tensor)
        direct_axis_density += (
            HALF_JUMP_VARIANCE_FACTOR
            * state_concentrations[event.from_state_index]
            * event.rate_s_inv
            * second_moment_diagonal
        )
        drift_by_state[event.from_state_index, :] += (
            event.rate_s_inv * displacement_array
        )
    stationary_drift = state_concentrations @ drift_by_state
    drift_tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    if float(np.max(np.abs(stationary_drift))) > drift_tolerance:
        raise ValueError("Markov-additive process has nonzero stationary drift")

    symmetrized_energy_matrix = _symmetrized_energy_matrix(
        generator_matrix,
        state_concentrations,
    )
    current_coupling_matrix = (
        np.sqrt(state_concentrations)[:, None] * drift_by_state
    ).T
    beta_factor = F * F / (R * temperature_K)
    corrector_mori_input = ProjectedMoriConductivityInput(
        direct_energy_matrix=np.zeros_like(symmetrized_energy_matrix),
        memory_self_energy_matrix=symmetrized_energy_matrix,
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=beta_factor,
    )
    corrector_mori_result = compute_projected_mori_conductivity(corrector_mori_input)
    corrector_axis_density = np.asarray(
        corrector_mori_result.quadratic_form_by_axis,
        dtype=float,
    )
    effective_axis_density = direct_axis_density - corrector_axis_density
    density_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(direct_axis_density))),
        float(np.max(np.abs(corrector_axis_density))),
    )
    if float(np.min(effective_axis_density)) < -density_tolerance:
        raise ValueError("Markov-additive effective axis density became negative")
    effective_axis_density = np.asarray(
        [
            ZERO_VALUE if abs(value) <= density_tolerance else float(value)
            for value in effective_axis_density
        ],
        dtype=float,
    )
    direct_sigma_S_m = (
        beta_factor * float(np.sum(direct_axis_density)) / CARTESIAN_AXIS_COUNT
    )
    corrector_sigma_S_m = corrector_mori_result.sigma_S_m
    sigma_S_m = direct_sigma_S_m - corrector_sigma_S_m
    if sigma_S_m < -density_tolerance:
        raise ValueError("Markov-additive conductivity became negative")
    if abs(sigma_S_m) <= density_tolerance:
        sigma_S_m = ZERO_VALUE
    return MarkovAdditiveConductivityResult(
        generator_s_inv=generator_matrix,
        validation=validation,
        event_reversal_residual_mol_m3_s=event_reversal_residual,
        drift_by_state_m_s=drift_by_state,
        corrector_mori_input=corrector_mori_input,
        corrector_mori_result=corrector_mori_result,
        direct_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in direct_axis_density
        ),
        event_mori_corrector_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in corrector_axis_density
        ),
        atmosphere_corrector_axis_density_m2_s_mol_m3=(0.0, 0.0, 0.0),
        projected_current_memory_corrector_axis_density_m2_s_mol_m3=(
            0.0,
            0.0,
            0.0,
        ),
        corrector_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in corrector_axis_density
        ),
        effective_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in effective_axis_density
        ),
        direct_sigma_S_m=float(direct_sigma_S_m),
        event_mori_corrector_sigma_S_m=float(corrector_sigma_S_m),
        atmosphere_corrector_sigma_S_m=0.0,
        projected_current_memory_corrector_sigma_S_m=0.0,
        corrector_sigma_S_m=float(corrector_sigma_S_m),
        sigma_S_m=float(sigma_S_m),
        direct_sigma_mS_cm=float(direct_sigma_S_m * S_M_TO_MS_CM),
        event_mori_corrector_sigma_mS_cm=float(corrector_sigma_S_m * S_M_TO_MS_CM),
        atmosphere_corrector_sigma_mS_cm=0.0,
        projected_current_memory_corrector_sigma_mS_cm=0.0,
        corrector_sigma_mS_cm=float(corrector_sigma_S_m * S_M_TO_MS_CM),
        sigma_mS_cm=float(sigma_S_m * S_M_TO_MS_CM),
        minimum_effective_axis_density_m2_s_mol_m3=float(
            np.min(effective_axis_density)
        ),
    )


def _markov_result_with_projected_current_memory(
    markov_result: MarkovAdditiveConductivityResult,
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ],
    atmosphere_mori_corrections: tuple[AtmosphereMoriCorrection, ...],
    temperature_K: float,
) -> MarkovAdditiveConductivityResult:
    if not projected_current_memory_corrections:
        return markov_result
    current_memory_mori_input = _projected_mori_input_from_current_memory_corrections(
        projected_current_memory_corrections,
        temperature_K,
    )
    current_memory_mori_result = compute_projected_mori_conductivity(
        current_memory_mori_input
    )
    current_memory_axis_density = np.asarray(
        current_memory_mori_result.quadratic_form_by_axis,
        dtype=float,
    )
    declared_current_memory_axis_density = np.asarray(
        tuple(
            math.fsum(
                correction.correction_axis_density_m2_s_mol_m3[axis_index]
                for correction in projected_current_memory_corrections
            )
            for axis_index in range(int(CARTESIAN_AXIS_COUNT))
        ),
        dtype=float,
    )
    current_memory_density_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(current_memory_axis_density))),
        float(np.max(np.abs(declared_current_memory_axis_density))),
    )
    if not np.allclose(
        current_memory_axis_density,
        declared_current_memory_axis_density,
        rtol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        atol=current_memory_density_tolerance,
    ):
        raise ValueError("projected current-memory Mori A,h density mismatch")
    declared_current_memory_sigma_S_m = math.fsum(
        correction.correction_sigma_S_m
        for correction in projected_current_memory_corrections
    )
    current_memory_sigma_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        abs(current_memory_mori_result.sigma_S_m),
        abs(declared_current_memory_sigma_S_m),
    )
    if not math.isclose(
        current_memory_mori_result.sigma_S_m,
        declared_current_memory_sigma_S_m,
        rel_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        abs_tol=current_memory_sigma_tolerance,
    ):
        raise ValueError("projected current-memory Mori A,h sigma mismatch")
    atmosphere_axis_density = np.asarray(
        tuple(
            math.fsum(
                correction.correction_axis_density_m2_s_mol_m3[axis_index]
                for correction in atmosphere_mori_corrections
            )
            for axis_index in range(int(CARTESIAN_AXIS_COUNT))
        ),
        dtype=float,
    )

    event_corrector_axis_density = np.asarray(
        markov_result.event_mori_corrector_axis_density_m2_s_mol_m3,
        dtype=float,
    )
    corrector_axis_density = event_corrector_axis_density + current_memory_axis_density
    direct_axis_density = np.asarray(
        markov_result.direct_axis_density_m2_s_mol_m3,
        dtype=float,
    )
    effective_axis_density = direct_axis_density - corrector_axis_density
    density_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(direct_axis_density))),
        float(np.max(np.abs(corrector_axis_density))),
    )
    if float(np.min(effective_axis_density)) < -density_tolerance:
        raise ValueError(
            "projected current-memory Mori block made conductivity negative"
        )
    effective_axis_density = np.asarray(
        [
            ZERO_VALUE if abs(value) <= density_tolerance else float(value)
            for value in effective_axis_density
        ],
        dtype=float,
    )
    atmosphere_corrector_sigma_S_m = math.fsum(
        correction.correction_sigma_S_m for correction in atmosphere_mori_corrections
    )
    current_memory_corrector_sigma_S_m = current_memory_mori_result.sigma_S_m
    corrector_sigma_S_m = (
        markov_result.event_mori_corrector_sigma_S_m
        + current_memory_corrector_sigma_S_m
    )
    sigma_S_m = markov_result.direct_sigma_S_m - corrector_sigma_S_m
    sigma_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        abs(markov_result.direct_sigma_S_m),
        abs(corrector_sigma_S_m),
    )
    if sigma_S_m < -sigma_tolerance:
        raise ValueError(
            "projected current-memory Mori block made conductivity negative"
        )
    if abs(sigma_S_m) <= sigma_tolerance:
        sigma_S_m = ZERO_VALUE
    return replace(
        markov_result,
        atmosphere_corrector_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in atmosphere_axis_density
        ),
        projected_current_memory_corrector_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in current_memory_axis_density
        ),
        corrector_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in corrector_axis_density
        ),
        effective_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in effective_axis_density
        ),
        atmosphere_corrector_sigma_S_m=float(atmosphere_corrector_sigma_S_m),
        projected_current_memory_corrector_sigma_S_m=float(
            current_memory_corrector_sigma_S_m
        ),
        corrector_sigma_S_m=float(corrector_sigma_S_m),
        sigma_S_m=float(sigma_S_m),
        atmosphere_corrector_sigma_mS_cm=float(
            atmosphere_corrector_sigma_S_m * S_M_TO_MS_CM
        ),
        projected_current_memory_corrector_sigma_mS_cm=float(
            current_memory_corrector_sigma_S_m * S_M_TO_MS_CM
        ),
        corrector_sigma_mS_cm=float(corrector_sigma_S_m * S_M_TO_MS_CM),
        sigma_mS_cm=float(sigma_S_m * S_M_TO_MS_CM),
        minimum_effective_axis_density_m2_s_mol_m3=float(
            np.min(effective_axis_density)
        ),
    )


def _projected_mori_input_from_current_memory_corrections(
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ],
    temperature_K: float,
) -> ProjectedMoriConductivityInput:
    if not projected_current_memory_corrections:
        raise ValueError("current-memory Mori input requires at least one correction")
    mode_count = len(projected_current_memory_corrections)
    current_coupling_matrix = np.zeros(
        (int(CARTESIAN_AXIS_COUNT), mode_count),
        dtype=float,
    )
    for mode_index, correction in enumerate(projected_current_memory_corrections):
        correction_axis_density = np.asarray(
            correction.correction_axis_density_m2_s_mol_m3,
            dtype=float,
        )
        memory_self_energy_s_inv = _positive_float(
            correction.memory_self_energy_s_inv,
            f"{correction.state_label}.memory_self_energy_s_inv",
        )
        if correction_axis_density.shape != (int(CARTESIAN_AXIS_COUNT),):
            raise ValueError(
                f"{correction.state_label}.correction_axis_density shape mismatch"
            )
        if not np.all(np.isfinite(correction_axis_density)):
            raise ValueError(
                f"{correction.state_label}.correction_axis_density contains non-finite values"
            )
        if float(np.min(correction_axis_density)) < 0.0:
            raise ValueError(
                f"{correction.state_label}.correction_axis_density must be nonnegative"
            )
        current_coupling_matrix[:, mode_index] = np.sqrt(
            correction_axis_density * memory_self_energy_s_inv
        )
    memory_self_energy_matrix = np.diag(
        tuple(
            _positive_float(
                correction.memory_self_energy_s_inv,
                f"{correction.state_label}.memory_self_energy_s_inv",
            )
            for correction in projected_current_memory_corrections
        )
    )
    return ProjectedMoriConductivityInput(
        direct_energy_matrix=np.zeros_like(memory_self_energy_matrix),
        memory_self_energy_matrix=memory_self_energy_matrix,
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=F * F / (R * _positive_float(temperature_K, "temperature_K")),
    )


def _combined_projected_mori_A_h(
    markov_result: MarkovAdditiveConductivityResult,
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ],
    temperature_K: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_energy_matrix = np.asarray(
        markov_result.corrector_mori_input.memory_self_energy_matrix,
        dtype=float,
    )
    event_current_coupling_matrix = np.asarray(
        markov_result.corrector_mori_input.current_coupling_matrix,
        dtype=float,
    )
    if not projected_current_memory_corrections:
        return event_energy_matrix.copy(), event_current_coupling_matrix.copy()
    current_memory_mori_input = _projected_mori_input_from_current_memory_corrections(
        projected_current_memory_corrections,
        temperature_K,
    )
    combined_energy_matrix = _block_diagonal_square_matrix(
        event_energy_matrix,
        current_memory_mori_input.memory_self_energy_matrix,
    )
    combined_current_coupling_matrix = np.concatenate(
        (
            event_current_coupling_matrix,
            current_memory_mori_input.current_coupling_matrix,
        ),
        axis=1,
    )
    return combined_energy_matrix, combined_current_coupling_matrix


def _block_diagonal_square_matrix(
    first_matrix: np.ndarray,
    second_matrix: np.ndarray,
) -> np.ndarray:
    first_square_matrix = _validated_square_matrix(first_matrix, "first_matrix")
    second_square_matrix = _validated_square_matrix(second_matrix, "second_matrix")
    combined_size = first_square_matrix.shape[0] + second_square_matrix.shape[0]
    combined_matrix = np.zeros((combined_size, combined_size), dtype=float)
    first_size = first_square_matrix.shape[0]
    combined_matrix[:first_size, :first_size] = first_square_matrix
    combined_matrix[first_size:, first_size:] = second_square_matrix
    return combined_matrix


def compute_projected_electrolyte_transport_model(
    projected_primitive_set: ProjectedPrimitiveSet,
    free_energy_functional: MolecularFreeEnergyFunctionalUx,
    partition_derivation_by_state_label: Mapping[str, TransportPartitionDefinitionAi],
    projected_transport_states: tuple["ProjectedTransportState", ...],
    reactive_flux_integrals: tuple[ProjectedReactiveFluxIntegral, ...],
    self_displacement_moments: tuple[ProjectedSelfDisplacementMoment, ...],
    self_direct_axis_density: tuple[float, float, float],
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ],
) -> ProjectedElectrolyteTransportModel:
    """Build the finite projection object from generator-projected primitives."""

    markov_additive_input = projected_primitive_set.markov_input
    state_labels = tuple(markov_additive_input.state_labels)
    if len(state_labels) == 0:
        raise ValueError("projected transport model must contain at least one state")
    if len(set(state_labels)) != len(state_labels):
        raise ValueError("projected transport state_labels must be unique")
    state_concentrations = _validated_state_concentrations(
        markov_additive_input.state_concentrations_mol_m3,
        len(state_labels),
    )
    _validate_projected_transport_state_inventory(
        state_labels,
        state_concentrations,
        projected_transport_states,
        markov_additive_input.temperature_K,
    )
    total_transport_concentration_mol_m3 = _positive_float(
        float(np.sum(state_concentrations)),
        "projected total transport concentration",
    )
    stationary_probabilities = (
        state_concentrations / total_transport_concentration_mol_m3
    )
    partition_states = _projected_partition_states(
        state_labels,
        state_concentrations,
        stationary_probabilities,
        partition_derivation_by_state_label,
    )
    conductivity_result = projected_primitive_set.markov_conductivity_result
    closure_contract = _projected_primitive_closure_contract()
    mori_basis_functions = _mori_basis_functions_for_projection(
        partition_states,
        reactive_flux_integrals,
        self_displacement_moments,
    )
    contribution_classifications = _projected_generator_contribution_classifications(
        partition_states,
        reactive_flux_integrals,
        self_displacement_moments,
        mori_basis_functions,
        projected_transport_states,
    )
    corrector_input = conductivity_result.corrector_mori_input
    primitive_mori_energy_matrix = (
        projected_primitive_set.mori_memory_energy_matrix_A
    )
    primitive_current_coupling_matrix = (
        projected_primitive_set.mori_current_coupling_matrix_h
    )
    projected_generator_model = ProjectedGeneratorModel(
        microscopic_generator_source=UxLxSource(
            equilibrium_measure_source=(
                closure_contract.equilibrium_measure_source.source_name
            ),
            reversible_generator_source=(
                closure_contract.reactive_flux_source.source_name
            ),
            microscopic_configuration_space=(
                "descriptor_projected_transport_partition_space"
            ),
            stationary_measure=free_energy_functional.partition_weight_formula_id,
            generator_formula_id="Q_ij=K_ij/c_i_with_detailed_balance",
            descriptor_complete=(
                closure_contract.descriptor_closure_derives_full_microscopic_generator
            ),
        ),
        charge_polarization_source=ChargePolarizationSource(
            observable_name="unwrapped_charge_polarization",
            formula_id="P=sum_a z_a e R_a",
            coordinate_convention="periodic_unwrapped_center_displacement",
            units="charge_number_meter",
        ),
        basis=ProjectedBasis(
            basis_labels=tuple(
                basis_function.label for basis_function in mori_basis_functions
            ),
            basis_sources=tuple(
                basis_function.source for basis_function in mori_basis_functions
            ),
            refinable_hierarchy=(
                "partition_indicators_plus_reactive_flux_self_current_and_mori_modes"
            ),
        ),
        populations=RestrictedPopulationSet(
            state_labels=tuple(
                partition_state.state_label for partition_state in partition_states
            ),
            concentrations_mol_m3=tuple(
                partition_state.concentration_mol_m3
                for partition_state in partition_states
            ),
            stationary_probabilities=tuple(
                partition_state.stationary_probability
                for partition_state in partition_states
            ),
            partition_models=tuple(
                partition_state.pmf_partition_model
                for partition_state in partition_states
            ),
        ),
        reactive_fluxes=SymmetricFluxSet(
            reactive_fluxes=reactive_flux_integrals,
        ),
        displacement_moments=ConditionalMomentSet(
            displacement_moments=reactive_flux_integrals,
        ),
        self_current_tensors=SelfCurrentSet(
            self_current_tensors=self_displacement_moments,
            direct_axis_density_m2_s_mol_m3=tuple(
                float(value)
                for value in conductivity_result.direct_axis_density_m2_s_mol_m3
            ),
        ),
        mori_matrix=MoriMatrix(
            direct_energy_matrix=_matrix_to_tuple_rows(
                np.zeros_like(primitive_mori_energy_matrix, dtype=float)
            ),
            memory_self_energy_matrix=_matrix_to_tuple_rows(
                primitive_mori_energy_matrix
            ),
            beta_over_volume=float(corrector_input.beta_over_volume),
        ),
        current_coupling=CurrentCoupling(
            current_coupling_matrix=_matrix_to_tuple_rows(
                primitive_current_coupling_matrix
            ),
            axis_count=int(CARTESIAN_AXIS_COUNT),
        ),
        conductivity=ProjectedGKConductivity(
            direct_sigma_mS_cm=conductivity_result.direct_sigma_mS_cm,
            corrector_sigma_mS_cm=conductivity_result.corrector_sigma_mS_cm,
            sigma_mS_cm=conductivity_result.sigma_mS_cm,
            theorem_id=closure_contract.readout_theorem,
        ),
        contribution_classifications=contribution_classifications,
    )
    projected_transport_model = ProjectedElectrolyteTransportModel(
        proof_status=_projected_transport_proof_status(closure_contract),
        closure_contract=closure_contract,
        free_energy_functional=free_energy_functional,
        projected_generator_model=projected_generator_model,
        projected_primitive_set=projected_primitive_set,
        contribution_classifications=contribution_classifications,
        partition_states=partition_states,
        projected_transport_states=projected_transport_states,
        state_labels=state_labels,
        stationary_concentrations_mol_m3=tuple(
            float(concentration_mol_m3) for concentration_mol_m3 in state_concentrations
        ),
        stationary_probabilities=tuple(
            float(probability) for probability in stationary_probabilities
        ),
        restricted_partition_weights=tuple(
            float(probability) for probability in stationary_probabilities
        ),
        mori_basis_functions=mori_basis_functions,
        reactive_flux_integrals=reactive_flux_integrals,
        self_displacement_moments=self_displacement_moments,
        self_direct_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in self_direct_axis_density
        ),
        projected_current_memory_corrections=projected_current_memory_corrections,
        atmosphere_mori_corrections=tuple(),
        onsager_transport_operator=OnsagerTransportOperator(
            state_labels=tuple(),
            charge_numbers=tuple(),
            concentrations_mol_m3=tuple(),
            bare_diffusivities_m2_s=tuple(),
            diagonal_friction_J_s_mol_m2=tuple(),
            friction_edges=tuple(),
            friction_matrix=tuple(),
            projected_mobility_matrix=tuple(),
            nernst_einstein_sigma_mS_cm=0.0,
            onsager_sigma_mS_cm=0.0,
            correlation_corrector_mS_cm=0.0,
        ),
        onsager_maxwell_stefan_operator=OnsagerMaxwellStefanOperator(
            state_labels=tuple(),
            concentrations_mol_m3=tuple(),
            charges=tuple(),
            self_frictions_J_s_mol_m2=tuple(),
            pair_frictions=tuple(),
            friction_matrix=tuple(),
            mobility_matrix=tuple(),
            sigma_onsager_mS_cm=0.0,
        ),
        markov_additive_input=projected_primitive_set.markov_input,
        conductivity_result=conductivity_result,
    )
    _validate_projected_transport_derivations(projected_transport_model)
    return projected_transport_model


def _matrix_to_tuple_rows(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2:
        raise ValueError("projected generator matrix must be two-dimensional")
    if not np.all(np.isfinite(matrix_array)):
        raise ValueError("projected generator matrix contains non-finite values")
    return tuple(tuple(float(component) for component in row) for row in matrix_array)


def _projected_generator_model_for_transport_model(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> ProjectedGeneratorModel:
    closure_contract = projected_transport_model.closure_contract
    free_energy_functional = projected_transport_model.free_energy_functional
    primitive_mori_energy_matrix = (
        projected_transport_model.projected_primitive_set.mori_memory_energy_matrix_A
    )
    primitive_current_coupling_matrix = (
        projected_transport_model.projected_primitive_set.mori_current_coupling_matrix_h
    )
    beta_over_volume = projected_transport_model.conductivity_result.corrector_mori_input.beta_over_volume
    return replace(
        projected_transport_model.projected_generator_model,
        microscopic_generator_source=replace(
            projected_transport_model.projected_generator_model.microscopic_generator_source,
            equilibrium_measure_source=(
                closure_contract.equilibrium_measure_source.source_name
            ),
            reversible_generator_source=(
                closure_contract.reactive_flux_source.source_name
            ),
            stationary_measure=free_energy_functional.partition_weight_formula_id,
            descriptor_complete=(
                closure_contract.descriptor_closure_derives_full_microscopic_generator
            ),
        ),
        basis=ProjectedBasis(
            basis_labels=tuple(
                basis_function.label
                for basis_function in projected_transport_model.mori_basis_functions
            ),
            basis_sources=tuple(
                basis_function.source
                for basis_function in projected_transport_model.mori_basis_functions
            ),
            refinable_hierarchy=(
                "partition_indicators_plus_reactive_flux_self_current_and_mori_modes"
            ),
        ),
        populations=RestrictedPopulationSet(
            state_labels=projected_transport_model.state_labels,
            concentrations_mol_m3=(
                projected_transport_model.stationary_concentrations_mol_m3
            ),
            stationary_probabilities=projected_transport_model.stationary_probabilities,
            partition_models=tuple(
                partition_state.pmf_partition_model
                for partition_state in projected_transport_model.partition_states
            ),
        ),
        reactive_fluxes=SymmetricFluxSet(
            reactive_fluxes=projected_transport_model.reactive_flux_integrals,
        ),
        displacement_moments=ConditionalMomentSet(
            displacement_moments=projected_transport_model.reactive_flux_integrals,
        ),
        self_current_tensors=SelfCurrentSet(
            self_current_tensors=projected_transport_model.self_displacement_moments,
            direct_axis_density_m2_s_mol_m3=(
                projected_transport_model.conductivity_result.direct_axis_density_m2_s_mol_m3
            ),
        ),
        mori_matrix=MoriMatrix(
            direct_energy_matrix=_matrix_to_tuple_rows(
                np.zeros_like(primitive_mori_energy_matrix)
            ),
            memory_self_energy_matrix=_matrix_to_tuple_rows(
                primitive_mori_energy_matrix
            ),
            beta_over_volume=float(beta_over_volume),
        ),
        current_coupling=CurrentCoupling(
            current_coupling_matrix=_matrix_to_tuple_rows(
                primitive_current_coupling_matrix
            ),
            axis_count=int(CARTESIAN_AXIS_COUNT),
        ),
        conductivity=ProjectedGKConductivity(
            direct_sigma_mS_cm=(
                projected_transport_model.conductivity_result.direct_sigma_mS_cm
            ),
            corrector_sigma_mS_cm=(
                projected_transport_model.conductivity_result.corrector_sigma_mS_cm
            ),
            sigma_mS_cm=projected_transport_model.conductivity_result.sigma_mS_cm,
            theorem_id=closure_contract.readout_theorem,
        ),
        contribution_classifications=(
            projected_transport_model.contribution_classifications
        ),
    )


def _projected_generator_contribution_classifications(
    partition_states: tuple[ProjectedTransportPartitionState, ...],
    reactive_flux_integrals: tuple[ProjectedReactiveFluxIntegral, ...],
    self_displacement_moments: tuple[ProjectedSelfDisplacementMoment, ...],
    mori_basis_functions: tuple[MoriBasisFunction, ...],
    projected_transport_states: tuple["ProjectedTransportState", ...],
) -> tuple[ProjectedGeneratorContributionClassification, ...]:
    classifications: list[ProjectedGeneratorContributionClassification] = []
    conductive_labels: set[str] = set()
    for partition_state in partition_states:
        classifications.append(
            ProjectedGeneratorContributionClassification(
                label=partition_state.state_label,
                contribution_class=PROJECTED_GENERATOR_CLASS_RESTRICTED_POPULATION,
                source=partition_state.population_source,
                conductivity_contribution_allowed=True,
                theorem_object="c_i",
            )
        )
        conductive_labels.add(partition_state.state_label.removesuffix(":mobile"))
    for reactive_flux in reactive_flux_integrals:
        flux_label = (
            f"{reactive_flux.family_label}:"
            f"{reactive_flux.from_state_label}->{reactive_flux.to_state_label}"
        )
        classifications.append(
            ProjectedGeneratorContributionClassification(
                label=flux_label,
                contribution_class=PROJECTED_GENERATOR_CLASS_REACTIVE_FLUX,
                source=reactive_flux.reactive_flux_source,
                conductivity_contribution_allowed=True,
                theorem_object="K_ij,Q_ij",
            )
        )
        classifications.append(
            ProjectedGeneratorContributionClassification(
                label=flux_label,
                contribution_class=PROJECTED_GENERATOR_CLASS_STATE_CHANGING_DISPLACEMENT,
                source=reactive_flux.displacement_moment_source,
                conductivity_contribution_allowed=True,
                theorem_object="d_ij,M_ij",
            )
        )
    for self_moment in self_displacement_moments:
        classifications.append(
            ProjectedGeneratorContributionClassification(
                label=f"{self_moment.family_label}:{self_moment.state_label}",
                contribution_class=PROJECTED_GENERATOR_CLASS_WITHIN_STATE_SELF_CURRENT,
                source=self_moment.displacement_moment_source,
                conductivity_contribution_allowed=True,
                theorem_object="D_self_i",
            )
        )
    for basis_function in mori_basis_functions:
        classifications.append(
            ProjectedGeneratorContributionClassification(
                label=basis_function.label,
                contribution_class=PROJECTED_GENERATOR_CLASS_MORI_MEMORY_BASIS,
                source=basis_function.source,
                conductivity_contribution_allowed=True,
                theorem_object="A_M,h",
            )
        )
    for projected_transport_state in projected_transport_states:
        if projected_transport_state.label in conductive_labels:
            continue
        classifications.append(
            ProjectedGeneratorContributionClassification(
                label=projected_transport_state.label,
                contribution_class=PROJECTED_GENERATOR_CLASS_DIAGNOSTIC_ONLY,
                source=projected_transport_state.pair_basin,
                conductivity_contribution_allowed=False,
                theorem_object="diagnostic",
            )
        )
    return tuple(classifications)


def _validate_projected_transport_state_inventory(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: np.ndarray,
    projected_transport_states: tuple["ProjectedTransportState", ...],
    temperature_K: float,
) -> None:
    _positive_float(temperature_K, "projected_transport_state_inventory.temperature_K")
    if len(projected_transport_states) != len(state_labels):
        raise ValueError("projected transport state inventory size mismatch")
    projected_state_by_label = {
        projected_state.label: projected_state
        for projected_state in projected_transport_states
    }
    if len(projected_state_by_label) != len(projected_transport_states):
        raise ValueError("projected transport state labels must be unique")
    for state_index, state_label in enumerate(state_labels):
        projected_label = _transport_source_label_for_projected_state(state_label)
        if projected_label not in projected_state_by_label:
            raise ValueError(
                f"projected Markov state {state_label} has no motif-state owner"
            )
        projected_state = projected_state_by_label[projected_label]
        expected_concentration_mol_m3 = _nonnegative_float(
            state_concentrations_mol_m3[state_index],
            f"{state_label}.state_concentration_mol_m3",
        )
        projected_concentration_mol_m3 = _nonnegative_float(
            projected_state.concentration_mol_m3,
            f"{projected_label}.projected_concentration_mol_m3",
        )
        concentration_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
            1.0,
            expected_concentration_mol_m3,
            projected_concentration_mol_m3,
        )
        if not math.isclose(
            expected_concentration_mol_m3,
            projected_concentration_mol_m3,
            rel_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
            abs_tol=concentration_tolerance,
        ):
            raise ValueError(
                f"projected motif-state concentration mismatch for {state_label}"
            )
        if projected_state.charged_centers:
            compute_projected_transport_state_charge_diffusivity_m2_s(
                projected_state,
                temperature_K,
            )


def _projected_partition_states(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: np.ndarray,
    stationary_probabilities: np.ndarray,
    partition_derivation_by_state_label: Mapping[str, TransportPartitionDefinitionAi],
) -> tuple[ProjectedTransportPartitionState, ...]:
    partition_states: list[ProjectedTransportPartitionState] = []
    for state_index, state_label in enumerate(state_labels):
        if state_label not in partition_derivation_by_state_label:
            raise ValueError(
                f"missing transport partition derivation for {state_label}"
            )
        partition_derivation = partition_derivation_by_state_label[state_label]
        if partition_derivation.state_label != state_label:
            raise ValueError(
                f"{state_label} partition derivation has mismatched state label "
                f"{partition_derivation.state_label}"
            )
        stationary_probability = float(stationary_probabilities[state_index])
        partition_states.append(
            ProjectedTransportPartitionState(
                state_index=state_index,
                state_label=state_label,
                concentration_mol_m3=float(state_concentrations_mol_m3[state_index]),
                stationary_probability=stationary_probability,
                restricted_partition_weight=stationary_probability,
                partition_definition=_transport_partition_definition(state_label),
                population_source=("restricted_population_from_mu_x_over_A_i"),
                partition_derivation=partition_derivation,
                pmf_partition_model=_pmf_partition_model(
                    state_label=state_label,
                    partition_derivation=partition_derivation,
                    restricted_partition_weight=stationary_probability,
                    concentration_mol_m3=float(
                        state_concentrations_mol_m3[state_index]
                    ),
                ),
            )
        )
    return tuple(partition_states)


def _pmf_partition_model(
    state_label: str,
    partition_derivation: TransportPartitionDefinitionAi,
    restricted_partition_weight: float,
    concentration_mol_m3: float,
) -> PMFPartitionModel:
    concentration = _nonnegative_float(
        concentration_mol_m3,
        f"{state_label}.pmf_partition_concentration_mol_m3",
    )
    partition_weight = _nonnegative_float(
        restricted_partition_weight,
        f"{state_label}.restricted_partition_weight",
    )
    if not partition_weight <= 1.0 + FINITE_MARKOV_ADDITIVE_TOLERANCE:
        raise ValueError(f"{state_label}.restricted_partition_weight exceeds one")
    lower_bound, upper_bound = _pmf_partition_coordinate_bounds(
        partition_derivation.parent_cluster_kind,
    )
    return PMFPartitionModel(
        state_label=state_label,
        reaction_coordinate=_pmf_reaction_coordinate_for_partition(
            partition_derivation
        ),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        pmf_terms=_pmf_terms_for_partition(partition_derivation),
        restricted_partition_weight=float(partition_weight),
        concentration_mol_m3=float(concentration),
    )


def _pmf_reaction_coordinate_for_partition(
    partition_derivation: TransportPartitionDefinitionAi,
) -> str:
    if partition_derivation.parent_cluster_kind in (
        CONTACT_PAIR_CLUSTER_KIND,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    ):
        return "ion_pair_center_distance_A"
    if (
        partition_derivation.transport_role
        == TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER
    ):
        return "local_ionic_field_magnitude"
    return "transport_center_position"


def _pmf_partition_coordinate_bounds(cluster_kind: str) -> tuple[float, float]:
    if cluster_kind == CONTACT_PAIR_CLUSTER_KIND:
        return (0.0, 1.0)
    if cluster_kind == SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
        return (1.0, 2.0)
    return (0.0, math.inf)


def _pmf_terms_for_partition(
    partition_derivation: TransportPartitionDefinitionAi,
) -> tuple[MolecularFreeEnergyTermDerivationUx, ...]:
    parameter_names = _primitive_parameter_names_for_theorem_role(
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    )
    return tuple(
        MolecularFreeEnergyTermDerivationUx(
            term_name=parameter_name,
            formula_id=(
                "pmf_partition_Ux_term:"
                f"{partition_derivation.parent_cluster_kind}:{parameter_name}"
            ),
            parameter_names=(parameter_name,),
            units="dimensionless_deltaG_over_RT_or_J_mol",
            sign_convention="positive term raises PMF and lowers restricted weight",
            theorem_role=PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
            empirical_closure_parameter_names=tuple(),
        )
        for parameter_name in parameter_names
    )


def _projected_primitive_closure_contract() -> ProjectedPrimitiveClosureContract:
    empirical_closure_parameter_names = _primitive_parameter_names_for_theorem_role(
        PRIMITIVE_PARAMETER_ROLE_EMPIRICAL_CLOSURE
    )
    descriptor_closure_derives_full_microscopic_generator = (
        _descriptor_parameter_roles_define_full_microscopic_generator()
        and len(empirical_closure_parameter_names) == 0
    )
    return ProjectedPrimitiveClosureContract(
        readout_theorem="finite_markov_additive_green_kubo_poisson_readout",
        equilibrium_measure_source=_projected_primitive_source_reference(
            PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
            "descriptor_derived_equilibrium_measure_mu_x_from_U_x",
        ),
        partition_source=_projected_primitive_source_reference(
            PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION,
            "center_resolved_transport_partition_A_i",
        ),
        concentration_source=_projected_primitive_source_reference(
            PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
            "restricted_population_c_i=C_mu_x_A_i",
        ),
        reactive_flux_source=_projected_primitive_source_reference(
            PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
            "descriptor_derived_symmetric_reactive_flux_K_ij_from_L_x_surface",
        ),
        displacement_moment_source=_projected_primitive_source_reference(
            PRIMITIVE_PARAMETER_ROLE_DISPLACEMENT_MOMENT,
            "descriptor_derived_conditional_charge_displacement_moments_from_P_x",
        ),
        descriptor_closure_derives_finite_projected_generator=(
            len(empirical_closure_parameter_names) == 0
        ),
        descriptor_closure_derives_full_microscopic_generator=(
            descriptor_closure_derives_full_microscopic_generator
        ),
        primitive_parameter_theorem_role_by_name=(
            CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME
        ),
    )


def _descriptor_parameter_roles_define_full_microscopic_generator() -> bool:
    observed_parameter_names = set(
        CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME
    )
    expected_parameter_names = set(CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES)
    if observed_parameter_names != expected_parameter_names:
        return False
    required_theorem_roles = {
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
        PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION,
        PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX,
        PRIMITIVE_PARAMETER_ROLE_DISPLACEMENT_MOMENT,
    }
    observed_theorem_roles = set(
        CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME.values()
    )
    return required_theorem_roles.issubset(observed_theorem_roles)


def _projected_primitive_source_reference(
    theorem_role: str,
    source_name: str,
) -> ProjectedPrimitiveSourceReference:
    return ProjectedPrimitiveSourceReference(
        theorem_role=theorem_role,
        source_name=source_name,
        source_parameter_names=_primitive_parameter_names_for_theorem_role(
            theorem_role
        ),
        empirical_closure_parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_EMPIRICAL_CLOSURE
        ),
    )


def _primitive_parameter_names_for_theorem_role(theorem_role: str) -> tuple[str, ...]:
    return tuple(
        parameter_name
        for parameter_name, parameter_role in CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME.items()
        if parameter_role == theorem_role
    )


def _molecular_free_energy_functional_derivation(
    temperature_K: float,
) -> MolecularFreeEnergyFunctionalUx:
    validated_temperature_K = _positive_float(temperature_K, "U_x.temperature_K")
    free_energy_parameter_names = _primitive_parameter_names_for_theorem_role(
        PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    )
    if not free_energy_parameter_names:
        raise ValueError("U_x derivation requires at least one free-energy parameter")
    terms = tuple(
        MolecularFreeEnergyTermDerivationUx(
            term_name=parameter_name,
            formula_id=f"restricted_partition_U_x_deltaG_over_RT:{parameter_name}",
            parameter_names=(parameter_name,),
            units="dimensionless_deltaG_over_RT",
            sign_convention=(
                "positive contribution raises restricted-state free energy"
            ),
            theorem_role=PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
            empirical_closure_parameter_names=tuple(),
        )
        for parameter_name in free_energy_parameter_names
    )
    return MolecularFreeEnergyFunctionalUx(
        formula_id="descriptor_equilibrium_measure_mu_x_proportional_exp_minus_beta_Ux",
        temperature_K=validated_temperature_K,
        beta_mol_per_J=1.0 / (R * validated_temperature_K),
        partition_weight_formula_id=(
            "restricted_weight_i=exp(-DeltaG_i/RT)/sum_k exp(-DeltaG_k/RT)"
        ),
        terms=terms,
        empirical_closure_parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_EMPIRICAL_CLOSURE
        ),
    )


def _transport_partition_derivations_by_state_label(
    state_labels: tuple[str, ...],
    projected_transport_states: tuple["ProjectedTransportState", ...],
) -> Mapping[str, TransportPartitionDefinitionAi]:
    projected_state_by_label = {
        projected_state.label: projected_state
        for projected_state in projected_transport_states
    }
    if len(projected_state_by_label) != len(projected_transport_states):
        raise ValueError("projected transport state labels must be unique")
    partition_derivations: dict[str, TransportPartitionDefinitionAi] = {}
    for state_label in state_labels:
        partition_label = _transport_partition_definition(state_label)
        source_transport_label = _transport_source_label_for_projected_state(
            state_label
        )
        if source_transport_label not in projected_state_by_label:
            raise ValueError(
                f"missing projected motif state for projected state {state_label}"
            )
        projected_state = projected_state_by_label[source_transport_label]
        partition_derivations[state_label] = TransportPartitionDefinitionAi(
            state_label=state_label,
            partition_label=partition_label,
            predicate_id=_transport_partition_predicate_id(state_label),
            center_species_name=_projected_partition_center_species_name(
                projected_state
            ),
            center_charge_number=_projected_partition_net_charge_number(
                projected_state
            ),
            parent_cluster_label=projected_state.label,
            parent_cluster_kind=projected_state.pair_basin,
            transport_role=_projected_partition_transport_role(projected_state),
            normalization_rule=(
                "restricted partition weights normalize by total projected "
                "transport concentration"
            ),
            disjointness_rule=(
                "finite Markov state labels form mutually exclusive transport "
                "partitions"
            ),
            source_parameter_names=_primitive_parameter_names_for_theorem_role(
                PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION
            ),
            empirical_closure_parameter_names=tuple(),
        )
    return partition_derivations


def _projected_partition_center_species_name(
    projected_state: "ProjectedTransportState",
) -> str:
    center_count = len(projected_state.charged_centers)
    if center_count == 0:
        return "neutral_projected_motif"
    if center_count == 1:
        return projected_state.charged_centers[0].label
    return "multi_center_projected_motif"


def _projected_partition_net_charge_number(
    projected_state: "ProjectedTransportState",
) -> int:
    return int(
        math.fsum(
            charged_center.charge_number
            for charged_center in projected_state.charged_centers
        )
    )


def _projected_partition_transport_role(
    projected_state: "ProjectedTransportState",
) -> str:
    pair_basin = projected_state.pair_basin
    projected_partition_role_by_pair_basin = {
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND: TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
        "additive_separated_solvent_separated_pair": (
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER
        ),
        CONTACT_PAIR_CLUSTER_KIND: TRANSPORT_ROLE_CONTACT_PAIR_CENTER,
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND: TRANSPORT_ROLE_CLUSTER_COM_CENTER,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND: TRANSPORT_ROLE_CLUSTER_COM_CENTER,
        NEUTRAL_CLUSTER_KIND: TRANSPORT_ROLE_CLUSTER_COM_CENTER,
    }
    if pair_basin in projected_partition_role_by_pair_basin:
        return projected_partition_role_by_pair_basin[pair_basin]
    return pair_basin


def _transport_source_label_for_projected_state(state_label: str) -> str:
    if state_label.endswith(":mobile"):
        return state_label.removesuffix(":mobile")
    return state_label


def _transport_partition_predicate_id(state_label: str) -> str:
    if state_label.endswith(":mobile"):
        return "transport_center_mobile_partition"
    return "neutral_or_zero_charge_transport_partition"


def _projected_transport_proof_status(
    closure_contract: ProjectedPrimitiveClosureContract,
) -> str:
    if closure_contract.descriptor_closure_derives_full_microscopic_generator:
        return FULL_MICROSCOPIC_GENERATOR_DERIVED_PROOF_STATUS
    if closure_contract.descriptor_closure_derives_finite_projected_generator:
        return FINITE_PROJECTED_DESCRIPTOR_CLOSURE_PROOF_STATUS
    return PROJECTED_READOUT_PROVEN_DESCRIPTOR_CLOSURE_EMPIRICAL


def _transport_partition_definition(state_label: str) -> str:
    if state_label.endswith(":mobile"):
        return state_label.removesuffix(":mobile")
    return state_label


def _projected_flux_integrals_from_events(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: np.ndarray,
    events: tuple[MarkovAdditiveEvent, ...],
    projected_transport_states: tuple["ProjectedTransportState", ...],
) -> tuple[
    tuple[ProjectedReactiveFluxIntegral, ...],
    tuple[ProjectedSelfDisplacementMoment, ...],
    np.ndarray,
]:
    if len(events) == 0:
        return (
            tuple(),
            tuple(),
            np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float),
        )
    projected_state_by_label = {
        projected_state.label: projected_state
        for projected_state in projected_transport_states
    }
    if len(projected_state_by_label) != len(projected_transport_states):
        raise ValueError("projected transport state labels must be unique")
    flux_accumulators: dict[
        tuple[
            str,
            int,
            int,
            tuple[float, float, float],
            tuple[tuple[float, float, float], ...],
        ],
        dict[str, float],
    ] = {}
    for event in events:
        displacement_array = _validated_displacement(
            event.charge_displacement_m,
            event.label,
        )
        second_moment_tensor = _validated_second_moment_tensor(
            event.charge_displacement_second_moment_m2,
            event.label,
        )
        _validate_second_moment_dominates_mean_outer_product(
            displacement_array,
            second_moment_tensor,
            event.label,
        )
        _validate_event_indices(event, len(state_labels))
        event_rate_s_inv = _positive_float(
            event.rate_s_inv, f"{event.label}.rate_s_inv"
        )
        event_flux_mol_m3_s = (
            state_concentrations_mol_m3[event.from_state_index] * event_rate_s_inv
        )
        (
            accumulator_key,
            canonical_direction,
        ) = _projected_flux_accumulator_key(
            event,
            displacement_array,
            second_moment_tensor,
        )
        if accumulator_key not in flux_accumulators:
            flux_accumulators[accumulator_key] = {
                "forward_flux_mol_m3_s": ZERO_VALUE,
                "reverse_flux_mol_m3_s": ZERO_VALUE,
            }
        accumulator = flux_accumulators[accumulator_key]
        accumulator[canonical_direction] += event_flux_mol_m3_s
    reactive_flux_integrals: list[ProjectedReactiveFluxIntegral] = []
    self_displacement_moments: list[ProjectedSelfDisplacementMoment] = []
    self_direct_axis_density = np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float)
    for accumulator_key, accumulator in sorted(flux_accumulators.items()):
        (
            family_label,
            from_state_index,
            to_state_index,
            displacement_key,
            second_moment_key,
        ) = accumulator_key
        forward_flux = accumulator["forward_flux_mol_m3_s"]
        reverse_flux = accumulator["reverse_flux_mol_m3_s"]
        displacement_array = np.asarray(displacement_key, dtype=float)
        if from_state_index == to_state_index:
            symmetric_flux = forward_flux + reverse_flux
            if symmetric_flux <= ZERO_VALUE:
                raise ValueError(f"{family_label} projected self flux must be positive")
            detailed_balance_residual = ZERO_VALUE
            rate_s_inv = symmetric_flux / state_concentrations_mol_m3[from_state_index]
            second_moment_tensor = np.asarray(second_moment_key, dtype=float)
            direct_axis_density = symmetric_flux * np.diag(second_moment_tensor)
            conditional_displacement_model = _conditional_charge_displacement_model(
                from_partition=state_labels[from_state_index],
                to_partition=state_labels[to_state_index],
                family_label=family_label,
                mean_displacement_m=displacement_key,
                second_moment_m2=second_moment_key,
            )
            self_current_projection_model = _self_current_projection_model(
                partition=state_labels[from_state_index],
                family_label=family_label,
                mean_displacement_m=displacement_key,
                rate_s_inv=float(rate_s_inv),
                second_moment_m2=second_moment_key,
                direct_axis_density_m2_s_mol_m3=tuple(
                    float(value) for value in direct_axis_density
                ),
            )
            displacement_derivation = _charge_displacement_moment_derivation(
                from_state_label=state_labels[from_state_index],
                to_state_label=state_labels[to_state_index],
                family_label=family_label,
                charge_displacement_m=displacement_key,
                second_moment_m2=second_moment_key,
                structural_hop_kinetics=_default_structural_hop_kinetics(
                    family_label,
                    state_labels[from_state_index],
                    state_labels[to_state_index],
                ),
            )
            self_direct_axis_density += direct_axis_density
            self_displacement_moments.append(
                ProjectedSelfDisplacementMoment(
                    state_index=from_state_index,
                    state_label=state_labels[from_state_index],
                    family_label=family_label,
                    charge_displacement_m=displacement_key,
                    charge_displacement_second_moment_m2=second_moment_key,
                    symmetric_flux_mol_m3_s=float(symmetric_flux),
                    rate_s_inv=float(rate_s_inv),
                    direct_axis_density_m2_s_mol_m3=tuple(
                        float(value) for value in direct_axis_density
                    ),
                    displacement_moment_source=_displacement_moment_source_for_family(
                        family_label
                    ),
                    conditional_displacement_model=conditional_displacement_model,
                    self_current_projection_model=self_current_projection_model,
                    self_current_tensor_model=_self_current_tensor_model(
                        state_label=state_labels[from_state_index],
                        concentration_mol_m3=float(
                            state_concentrations_mol_m3[from_state_index]
                        ),
                        self_displacement_tensor_m2_s=(
                            self_current_projection_model.self_displacement_tensor_m2_s
                        ),
                    ),
                    displacement_moment_derivation=displacement_derivation,
                )
            )
            continue
        symmetric_flux = 0.5 * (forward_flux + reverse_flux)
        if symmetric_flux <= ZERO_VALUE:
            raise ValueError(f"{family_label} projected reactive flux must be positive")
        detailed_balance_residual = abs(forward_flux - reverse_flux)
        forward_rate = symmetric_flux / state_concentrations_mol_m3[from_state_index]
        reverse_rate = symmetric_flux / state_concentrations_mol_m3[to_state_index]
        structural_hop_kinetics = _structural_hop_kinetics_for_projected_family(
            family_label,
            state_labels[from_state_index],
            state_labels[to_state_index],
            projected_state_by_label,
            second_moment_key,
            float(forward_rate),
            float(reverse_rate),
        )
        derived_reactive_flux_model = _derived_reactive_flux_model(
            from_partition=state_labels[from_state_index],
            to_partition=state_labels[to_state_index],
            family_label=family_label,
            symmetric_flux_mol_m3_s=float(symmetric_flux),
            forward_rate_s_inv=float(forward_rate),
            reverse_rate_s_inv=float(reverse_rate),
            structural_hop_kinetics=structural_hop_kinetics,
        )
        conditional_displacement_model = _conditional_charge_displacement_model(
            from_partition=state_labels[from_state_index],
            to_partition=state_labels[to_state_index],
            family_label=family_label,
            mean_displacement_m=displacement_key,
            second_moment_m2=second_moment_key,
        )
        reactive_flux_derivation = _reactive_flux_derivation(
            from_state_label=state_labels[from_state_index],
            to_state_label=state_labels[to_state_index],
            family_label=family_label,
            symmetric_flux_mol_m3_s=float(symmetric_flux),
            forward_rate_s_inv=float(forward_rate),
            reverse_rate_s_inv=float(reverse_rate),
            structural_hop_kinetics=structural_hop_kinetics,
        )
        displacement_derivation = _charge_displacement_moment_derivation(
            from_state_label=state_labels[from_state_index],
            to_state_label=state_labels[to_state_index],
            family_label=family_label,
            charge_displacement_m=displacement_key,
            second_moment_m2=second_moment_key,
            structural_hop_kinetics=structural_hop_kinetics,
        )
        reactive_flux_integrals.append(
            ProjectedReactiveFluxIntegral(
                from_state_index=from_state_index,
                to_state_index=to_state_index,
                from_state_label=state_labels[from_state_index],
                to_state_label=state_labels[to_state_index],
                family_label=family_label,
                charge_displacement_m=displacement_key,
                charge_displacement_second_moment_m2=second_moment_key,
                symmetric_flux_mol_m3_s=float(symmetric_flux),
                forward_rate_s_inv=float(forward_rate),
                reverse_rate_s_inv=float(reverse_rate),
                detailed_balance_residual_mol_m3_s=float(detailed_balance_residual),
                reactive_flux_source=_reactive_flux_source_for_family(family_label),
                displacement_moment_source=_displacement_moment_source_for_family(
                    family_label
                ),
                derived_reactive_flux_model=derived_reactive_flux_model,
                conditional_displacement_model=conditional_displacement_model,
                reactive_flux_model=_reactive_flux_model(
                    from_state_label=state_labels[from_state_index],
                    to_state_label=state_labels[to_state_index],
                    symmetric_flux_mol_m3_s=float(symmetric_flux),
                    structural_hop_kinetics=structural_hop_kinetics,
                ),
                conditional_displacement_moment_model=(
                    _conditional_displacement_moment_model(
                        from_state_label=state_labels[from_state_index],
                        to_state_label=state_labels[to_state_index],
                        mean_displacement_m=displacement_key,
                        second_moment_m2=second_moment_key,
                    )
                ),
                reactive_flux_derivation=reactive_flux_derivation,
                displacement_moment_derivation=displacement_derivation,
            )
        )
    return (
        tuple(reactive_flux_integrals),
        tuple(self_displacement_moments),
        self_direct_axis_density,
    )


def _reactive_flux_source_for_family(family_label: str) -> str:
    if family_label.startswith("solvent_separated_pair_"):
        return "symmetric_reactive_flux_K_ij_from_solvent_separated_pair_surface"
    if family_label not in PROJECTED_REACTIVE_FLUX_SOURCE_BY_EXACT_FAMILY:
        raise ValueError(f"unknown projected reactive-flux family {family_label}")
    return PROJECTED_REACTIVE_FLUX_SOURCE_BY_EXACT_FAMILY[family_label]


def _derived_reactive_flux_model(
    from_partition: str,
    to_partition: str,
    family_label: str,
    symmetric_flux_mol_m3_s: float,
    forward_rate_s_inv: float,
    reverse_rate_s_inv: float,
    structural_hop_kinetics: _AssociationStructuralHopKinetics,
) -> DerivedReactiveFluxModel:
    source_name = _reactive_flux_source_for_family(family_label)
    return DerivedReactiveFluxModel(
        from_partition=from_partition,
        to_partition=to_partition,
        transition_surface=structural_hop_kinetics.transition_surface,
        free_energy_barrier_over_RT=(
            structural_hop_kinetics.free_energy_barrier_over_RT
        ),
        partition_gap_scale_over_RT=(
            structural_hop_kinetics.partition_gap_scale_over_RT
        ),
        free_energy_mismatch_over_RT=(
            structural_hop_kinetics.free_energy_mismatch_over_RT
        ),
        effective_diffusivity_m2_s=(structural_hop_kinetics.effective_diffusivity_m2_s),
        hop_length_m=structural_hop_kinetics.hop_length_m,
        diffusion_tensor_source=source_name,
        friction_source=source_name,
        transmission_coefficient=1.0,
        symmetric_flux_mol_m3_s=_positive_float(
            symmetric_flux_mol_m3_s,
            f"{family_label}.K_ij_mol_m3_s",
        ),
        forward_rate_s_inv=_positive_float(
            forward_rate_s_inv,
            f"{family_label}.Q_ij_s_inv",
        ),
        reverse_rate_s_inv=_positive_float(
            reverse_rate_s_inv,
            f"{family_label}.Q_ji_s_inv",
        ),
        detailed_balance_condition="c_i*Q_ij=K_ij=c_j*Q_ji",
        parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
        ),
    )


def _reactive_flux_derivation(
    from_state_label: str,
    to_state_label: str,
    family_label: str,
    symmetric_flux_mol_m3_s: float,
    forward_rate_s_inv: float,
    reverse_rate_s_inv: float,
    structural_hop_kinetics: _AssociationStructuralHopKinetics,
) -> ReactiveFluxDerivationKij:
    source_name = _reactive_flux_source_for_family(family_label)
    formula_id = source_name
    symmetric_flux_formula_id = "K_ijr=0.5*(c_i*q_i_to_jr+c_j*q_j_to_ir_reverse)"
    if family_label == "association_structural_hop":
        formula_id = "projected_association_structural_hop_pmf_capacity_flux"
        symmetric_flux_formula_id = (
            "K_ij^hop=sqrt(c_i*c_j)*(D_eff/L_hop^2)*exp(-DeltaG_barrier/RT)"
        )
    return ReactiveFluxDerivationKij(
        from_state_label=from_state_label,
        to_state_label=to_state_label,
        family_label=family_label,
        formula_id=formula_id,
        symmetric_flux_formula_id=symmetric_flux_formula_id,
        generator_rate_formula_id="Q_ijr=K_ijr/c_i",
        symmetric_flux_mol_m3_s=_positive_float(
            symmetric_flux_mol_m3_s,
            f"{family_label}.symmetric_flux_mol_m3_s",
        ),
        forward_rate_s_inv=_positive_float(
            forward_rate_s_inv,
            f"{family_label}.forward_rate_s_inv",
        ),
        reverse_rate_s_inv=_positive_float(
            reverse_rate_s_inv,
            f"{family_label}.reverse_rate_s_inv",
        ),
        detailed_balance_condition="c_i*Q_ijr=c_j*Q_jir_reverse=K_ijr",
        mobility_source=source_name,
        parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_REACTIVE_FLUX
        ),
        empirical_closure_parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_EMPIRICAL_CLOSURE
        ),
    )


def _displacement_moment_source_for_family(family_label: str) -> str:
    if family_label.startswith("projected_") and family_label.endswith("_self_current"):
        return "projected_generator_state_local_charge_covariance"
    if family_label.startswith("solvent_separated_pair_"):
        return "conditional_solvent_separated_pair_charge_displacement"
    if family_label not in PROJECTED_DISPLACEMENT_MOMENT_SOURCE_BY_EXACT_FAMILY:
        raise ValueError(f"unknown projected displacement family {family_label}")
    return PROJECTED_DISPLACEMENT_MOMENT_SOURCE_BY_EXACT_FAMILY[family_label]


def _conditional_charge_displacement_model(
    from_partition: str,
    to_partition: str,
    family_label: str,
    mean_displacement_m: tuple[float, float, float],
    second_moment_m2: tuple[tuple[float, float, float], ...],
) -> ConditionalChargeDisplacementModel:
    validated_displacement = _validated_displacement(
        mean_displacement_m,
        f"{family_label}.conditional_mean_displacement_m",
    )
    validated_second_moment = _validated_second_moment_tensor(
        second_moment_m2,
        f"{family_label}.conditional_second_moment_m2",
    )
    _validate_second_moment_dominates_mean_outer_product(
        validated_displacement,
        validated_second_moment,
        f"{family_label}.conditional_second_moment_m2",
    )
    return ConditionalChargeDisplacementModel(
        from_partition=from_partition,
        to_partition=to_partition,
        family_label=family_label,
        mean_displacement_m=tuple(float(value) for value in validated_displacement),
        second_moment_m2=tuple(
            tuple(float(component) for component in row)
            for row in validated_second_moment
        ),
        reverse_rule="antisymmetric_mean_symmetric_second_moment",
        parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_DISPLACEMENT_MOMENT
        ),
    )


def _reactive_flux_model(
    from_state_label: str,
    to_state_label: str,
    symmetric_flux_mol_m3_s: float,
    structural_hop_kinetics: _AssociationStructuralHopKinetics,
) -> ReactiveFluxModel:
    return ReactiveFluxModel(
        from_state=from_state_label,
        to_state=to_state_label,
        reaction_coordinate=structural_hop_kinetics.transition_surface,
        free_energy_profile=(
            0.0,
            float(structural_hop_kinetics.free_energy_barrier_over_RT),
        ),
        coordinate_diffusion_profile=(
            float(structural_hop_kinetics.effective_diffusivity_m2_s),
            float(structural_hop_kinetics.effective_diffusivity_m2_s),
        ),
        symmetric_flux_mol_m3_s=_positive_float(
            symmetric_flux_mol_m3_s,
            f"{from_state_label}->{to_state_label}.symmetric_flux_mol_m3_s",
        ),
    )


def _conditional_displacement_moment_model(
    from_state_label: str,
    to_state_label: str,
    mean_displacement_m: tuple[float, float, float],
    second_moment_m2: tuple[tuple[float, float, float], ...],
) -> ConditionalDisplacementMomentModel:
    validated_displacement = _validated_displacement(
        mean_displacement_m,
        f"{from_state_label}->{to_state_label}.mean_displacement_m",
    )
    validated_second_moment = _validated_second_moment_tensor(
        second_moment_m2,
        f"{from_state_label}->{to_state_label}.second_moment_m2",
    )
    _validate_second_moment_dominates_mean_outer_product(
        validated_displacement,
        validated_second_moment,
        f"{from_state_label}->{to_state_label}.second_moment_m2",
    )
    return ConditionalDisplacementMomentModel(
        from_state=from_state_label,
        to_state=to_state_label,
        mean_displacement_m=tuple(float(value) for value in validated_displacement),
        second_moment_m2=tuple(
            tuple(float(component) for component in row)
            for row in validated_second_moment
        ),
    )


def _self_current_tensor_model(
    state_label: str,
    concentration_mol_m3: float,
    self_displacement_tensor_m2_s: tuple[tuple[float, float, float], ...],
) -> SelfCurrentTensorModel:
    tensor = _validated_second_moment_tensor(
        self_displacement_tensor_m2_s,
        f"{state_label}.self_current_tensor_m2_s",
    )
    return SelfCurrentTensorModel(
        state_label=state_label,
        diffusion_tensor_m2_s=tuple(
            tuple(float(component) for component in row) for row in tensor
        ),
        concentration_mol_m3=_nonnegative_float(
            concentration_mol_m3,
            f"{state_label}.self_current_concentration_mol_m3",
        ),
    )


def _self_current_projection_model(
    partition: str,
    family_label: str,
    mean_displacement_m: tuple[float, float, float],
    rate_s_inv: float,
    second_moment_m2: tuple[tuple[float, float, float], ...],
    direct_axis_density_m2_s_mol_m3: tuple[float, float, float],
) -> SelfCurrentProjectionModel:
    validated_rate_s_inv = _positive_float(
        rate_s_inv, f"{family_label}.self_rate_s_inv"
    )
    validated_displacement = _validated_displacement(
        mean_displacement_m,
        f"{family_label}.self_mean_displacement_m",
    )
    second_moment = np.asarray(
        _validated_second_moment_tensor(
            second_moment_m2,
            f"{family_label}.self_second_moment_m2",
        ),
        dtype=float,
    )
    _validate_second_moment_dominates_mean_outer_product(
        validated_displacement,
        second_moment,
        f"{family_label}.self_second_moment_m2",
    )
    self_displacement_tensor = second_moment * validated_rate_s_inv
    return SelfCurrentProjectionModel(
        partition=partition,
        family_label=family_label,
        mean_displacement_m=tuple(float(value) for value in validated_displacement),
        self_displacement_tensor_m2_s=tuple(
            tuple(float(component) for component in row)
            for row in self_displacement_tensor
        ),
        direct_axis_density_m2_s_mol_m3=direct_axis_density_m2_s_mol_m3,
        source=_displacement_moment_source_for_family(family_label),
        parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_DISPLACEMENT_MOMENT
        ),
    )


def _second_moment_tensor_from_displacement(
    displacement_m: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    displacement_array = np.asarray(displacement_m, dtype=float)
    if displacement_array.shape != (int(CARTESIAN_AXIS_COUNT),):
        raise ValueError("displacement_m must have one component per Cartesian axis")
    second_moment = np.outer(displacement_array, displacement_array)
    return tuple(tuple(float(component) for component in row) for row in second_moment)


def _charge_displacement_moment_derivation(
    from_state_label: str,
    to_state_label: str,
    family_label: str,
    charge_displacement_m: tuple[float, float, float],
    second_moment_m2: tuple[tuple[float, float, float], ...],
    structural_hop_kinetics: _AssociationStructuralHopKinetics,
) -> ChargeDisplacementMomentDerivationDij:
    source_name = _displacement_moment_source_for_family(family_label)
    validated_displacement = _validated_displacement(
        charge_displacement_m,
        f"{family_label}.charge_displacement_m",
    )
    _validated_second_moment_tensor(
        second_moment_m2,
        f"{family_label}.charge_displacement_second_moment_m2",
    )
    _validate_second_moment_dominates_mean_outer_product(
        validated_displacement,
        np.asarray(second_moment_m2, dtype=float),
        f"{family_label}.charge_displacement_second_moment_m2",
    )
    moment_definition = "d_ijr=E[Delta_P_charge | X_0 in A_i, X_tau in A_j, channel r]"
    if family_label == "association_structural_hop":
        moment_definition = (
            "d_ijr=E[Delta_P_charge | structural-hop partition crossing]; "
            "mean displacement uses unbiased structural-hop symmetry; "
            "second moment uses isotropic surface-conditioned projection "
            "M_ij≈|z|^2*L_hop^2/3*I from the structural-hop partition crossing"
        )
    return ChargeDisplacementMomentDerivationDij(
        from_state_label=from_state_label,
        to_state_label=to_state_label,
        family_label=family_label,
        formula_id=source_name,
        moment_definition=moment_definition,
        charge_displacement_m=tuple(float(value) for value in validated_displacement),
        units="charge_number_meter",
        reverse_displacement_rule="d_jir_reverse=-d_ijr",
        parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_DISPLACEMENT_MOMENT
        ),
        empirical_closure_parameter_names=_primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_EMPIRICAL_CLOSURE
        ),
    )


def _default_structural_hop_kinetics(
    family_label: str,
    from_partition: str,
    to_partition: str,
) -> _AssociationStructuralHopKinetics:
    return _AssociationStructuralHopKinetics(
        transition_surface=f"{from_partition}<->{to_partition}:{family_label}",
        partition_gap_scale_over_RT=0.0,
        free_energy_mismatch_over_RT=0.0,
        free_energy_barrier_over_RT=0.0,
        effective_diffusivity_m2_s=0.0,
        hop_length_m=0.0,
    )


def _structural_hop_kinetics_for_projected_family(
    family_label: str,
    from_state_label: str,
    to_state_label: str,
    projected_state_by_label: Mapping[str, "ProjectedTransportState"],
    second_moment_m2: tuple[tuple[float, float, float], ...],
    forward_rate_s_inv: float,
    reverse_rate_s_inv: float,
) -> _AssociationStructuralHopKinetics:
    if family_label != "association_structural_hop":
        return _default_structural_hop_kinetics(
            family_label,
            from_state_label,
            to_state_label,
        )
    from_transport_label = _transport_source_label_for_projected_state(from_state_label)
    to_transport_label = _transport_source_label_for_projected_state(to_state_label)
    if from_transport_label not in projected_state_by_label:
        raise ValueError(f"missing projected motif state for {from_state_label}")
    if to_transport_label not in projected_state_by_label:
        raise ValueError(f"missing projected motif state for {to_state_label}")
    return _projected_event_structural_hop_kinetics(
        family_label,
        from_state_label,
        to_state_label,
        second_moment_m2,
        forward_rate_s_inv,
        reverse_rate_s_inv,
    )


def _projected_event_structural_hop_kinetics(
    family_label: str,
    from_state_label: str,
    to_state_label: str,
    second_moment_m2: tuple[tuple[float, float, float], ...],
    forward_rate_s_inv: float,
    reverse_rate_s_inv: float,
) -> _AssociationStructuralHopKinetics:
    second_moment_tensor = _validated_second_moment_tensor(
        second_moment_m2,
        f"{family_label}.projected_event_second_moment_m2",
    )
    charge_displacement_variance_m2 = _positive_float(
        float(np.trace(second_moment_tensor)),
        f"{family_label}.projected_charge_displacement_variance_m2",
    )
    hop_length_m = math.sqrt(charge_displacement_variance_m2)
    validated_forward_rate_s_inv = _positive_float(
        forward_rate_s_inv,
        f"{family_label}.projected_forward_rate_s_inv",
    )
    validated_reverse_rate_s_inv = _positive_float(
        reverse_rate_s_inv,
        f"{family_label}.projected_reverse_rate_s_inv",
    )
    geometric_rate_s_inv = math.sqrt(
        validated_forward_rate_s_inv * validated_reverse_rate_s_inv
    )
    free_energy_mismatch_over_RT = abs(
        math.log(validated_forward_rate_s_inv / validated_reverse_rate_s_inv)
    )
    return _AssociationStructuralHopKinetics(
        transition_surface=(
            "projected_event_surface:"
            f"{from_state_label}<->{to_state_label}:{family_label}"
        ),
        partition_gap_scale_over_RT=free_energy_mismatch_over_RT,
        free_energy_mismatch_over_RT=free_energy_mismatch_over_RT,
        free_energy_barrier_over_RT=0.0,
        effective_diffusivity_m2_s=(
            charge_displacement_variance_m2 * geometric_rate_s_inv
        ),
        hop_length_m=hop_length_m,
    )


def _projected_flux_accumulator_key(
    event: MarkovAdditiveEvent,
    displacement_array: np.ndarray,
    second_moment_tensor: np.ndarray,
) -> tuple[
    tuple[
        str,
        int,
        int,
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ],
    str,
]:
    displacement_key = _displacement_key(displacement_array)
    second_moment_key = _second_moment_key(second_moment_tensor)
    reverse_displacement_key = tuple(
        _canonical_float_for_key(-component) for component in displacement_key
    )
    forward_key = (
        event.family_label,
        event.from_state_index,
        event.to_state_index,
        displacement_key,
        second_moment_key,
    )
    reverse_key = (
        event.family_label,
        event.to_state_index,
        event.from_state_index,
        reverse_displacement_key,
        second_moment_key,
    )
    if forward_key <= reverse_key:
        return forward_key, "forward_flux_mol_m3_s"
    return reverse_key, "reverse_flux_mol_m3_s"


def _markov_additive_input_from_projected_primitives(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: np.ndarray,
    temperature_K: float,
    reactive_flux_integrals: tuple[ProjectedReactiveFluxIntegral, ...],
    self_displacement_moments: tuple[ProjectedSelfDisplacementMoment, ...],
) -> MarkovAdditiveConductivityInput:
    derived_events: list[MarkovAdditiveEvent] = []
    for reactive_flux in reactive_flux_integrals:
        flux_model = reactive_flux.derived_reactive_flux_model
        displacement_model = reactive_flux.conditional_displacement_model
        forward_rate_s_inv = (
            flux_model.symmetric_flux_mol_m3_s
            / state_concentrations_mol_m3[reactive_flux.from_state_index]
        )
        reverse_rate_s_inv = (
            flux_model.symmetric_flux_mol_m3_s
            / state_concentrations_mol_m3[reactive_flux.to_state_index]
        )
        forward_displacement_m = displacement_model.mean_displacement_m
        reverse_displacement_m = tuple(
            float(-component) for component in forward_displacement_m
        )
        derived_events.append(
            MarkovAdditiveEvent(
                from_state_index=reactive_flux.from_state_index,
                to_state_index=reactive_flux.to_state_index,
                rate_s_inv=_positive_float(
                    forward_rate_s_inv,
                    f"{reactive_flux.family_label}.derived_forward_rate_s_inv",
                ),
                charge_displacement_m=forward_displacement_m,
                charge_displacement_second_moment_m2=(
                    displacement_model.second_moment_m2
                ),
                label=(
                    "derived_reactive_flux:"
                    f"{reactive_flux.family_label}:"
                    f"{reactive_flux.from_state_label}->"
                    f"{reactive_flux.to_state_label}"
                ),
                family_label=reactive_flux.family_label,
            )
        )
        derived_events.append(
            MarkovAdditiveEvent(
                from_state_index=reactive_flux.to_state_index,
                to_state_index=reactive_flux.from_state_index,
                rate_s_inv=_positive_float(
                    reverse_rate_s_inv,
                    f"{reactive_flux.family_label}.derived_reverse_rate_s_inv",
                ),
                charge_displacement_m=reverse_displacement_m,
                charge_displacement_second_moment_m2=(
                    displacement_model.second_moment_m2
                ),
                label=(
                    "derived_reactive_flux:"
                    f"{reactive_flux.family_label}:"
                    f"{reactive_flux.to_state_label}->"
                    f"{reactive_flux.from_state_label}"
                ),
                family_label=reactive_flux.family_label,
            )
        )
    for self_moment in self_displacement_moments:
        displacement_array = np.asarray(
            self_moment.conditional_displacement_model.mean_displacement_m,
            dtype=float,
        )
        rate_s_inv = (
            self_moment.symmetric_flux_mol_m3_s
            / state_concentrations_mol_m3[self_moment.state_index]
        )
        derived_events.append(
            MarkovAdditiveEvent(
                from_state_index=self_moment.state_index,
                to_state_index=self_moment.state_index,
                rate_s_inv=_positive_float(
                    rate_s_inv,
                    f"{self_moment.family_label}.derived_self_rate_s_inv",
                ),
                charge_displacement_m=tuple(
                    float(value) for value in displacement_array
                ),
                charge_displacement_second_moment_m2=(
                    self_moment.conditional_displacement_model.second_moment_m2
                ),
                label=(
                    "derived_self_current:"
                    f"{self_moment.family_label}:{self_moment.state_label}:plus"
                ),
                family_label=self_moment.family_label,
            )
        )
        if not _is_zero_displacement(displacement_array):
            derived_events.append(
                MarkovAdditiveEvent(
                    from_state_index=self_moment.state_index,
                    to_state_index=self_moment.state_index,
                    rate_s_inv=_positive_float(
                        rate_s_inv,
                        f"{self_moment.family_label}.derived_reverse_self_rate_s_inv",
                    ),
                    charge_displacement_m=tuple(
                        float(-value) for value in displacement_array
                    ),
                    charge_displacement_second_moment_m2=(
                        self_moment.conditional_displacement_model.second_moment_m2
                    ),
                    label=(
                        "derived_self_current:"
                        f"{self_moment.family_label}:{self_moment.state_label}:minus"
                    ),
                    family_label=self_moment.family_label,
                )
            )
    if not derived_events:
        raise ValueError("projected primitive derivation produced no events")
    return MarkovAdditiveConductivityInput(
        state_labels=state_labels,
        state_concentrations_mol_m3=state_concentrations_mol_m3,
        events=tuple(derived_events),
        temperature_K=temperature_K,
    )


def _mori_basis_functions_for_projection(
    partition_states: tuple[ProjectedTransportPartitionState, ...],
    reactive_flux_integrals: tuple[ProjectedReactiveFluxIntegral, ...],
    self_displacement_moments: tuple[ProjectedSelfDisplacementMoment, ...],
) -> tuple[MoriBasisFunction, ...]:
    basis_functions: list[MoriBasisFunction] = []
    for partition_state in partition_states:
        basis_source = _mori_basis_source_for_partition(partition_state)
        basis_functions.append(
            MoriBasisFunction(
                label=f"{basis_source}:{partition_state.state_label}",
                state_label=partition_state.state_label,
                observable_definition=(
                    f"indicator[{partition_state.partition_definition}]"
                ),
                source=basis_source,
                parameter_names=(
                    partition_state.partition_derivation.source_parameter_names
                ),
            )
        )
    for reactive_flux_index, reactive_flux in enumerate(reactive_flux_integrals):
        if reactive_flux.family_label.startswith("atmosphere_memory_"):
            basis_functions.append(
                MoriBasisFunction(
                    label=(
                        "current_memory:"
                        f"{reactive_flux_index}:"
                        f"{reactive_flux.family_label}:"
                        f"{reactive_flux.from_state_label}->"
                        f"{reactive_flux.to_state_label}"
                    ),
                    state_label=reactive_flux.from_state_label,
                    observable_definition=(
                        "ionic-atmosphere polarization current-memory channel"
                    ),
                    source=MORI_BASIS_SOURCE_IONIC_ATMOSPHERE_POLARIZATION,
                    parameter_names=(
                        reactive_flux.derived_reactive_flux_model.parameter_names
                    ),
                )
            )
    for self_moment_index, self_moment in enumerate(self_displacement_moments):
        if self_moment.family_label.startswith("solvent_separated_pair_"):
            basis_functions.append(
                MoriBasisFunction(
                    label=(
                        "self_current:"
                        f"{self_moment_index}:"
                        f"{self_moment.family_label}:{self_moment.state_label}"
                    ),
                    state_label=self_moment.state_label,
                    observable_definition=(
                        "solvent-separated-pair conditional self-current mode"
                    ),
                    source=MORI_BASIS_SOURCE_SOLVENT_CAGE_ORIENTATION,
                    parameter_names=(
                        self_moment.self_current_projection_model.parameter_names
                    ),
                )
            )
    return tuple(basis_functions)


def _mori_basis_source_for_partition(
    partition_state: ProjectedTransportPartitionState,
) -> str:
    if (
        partition_state.partition_derivation.transport_role
        == TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER
    ):
        return MORI_BASIS_SOURCE_SOLVENT_CAGE_ORIENTATION
    return MORI_BASIS_SOURCE_PARTITION_INDICATOR


def _validate_projected_transport_derivations(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    _validate_molecular_free_energy_functional_derivation(
        projected_transport_model.free_energy_functional
    )
    _validate_projected_primitive_set(projected_transport_model)
    _validate_projected_generator_model(projected_transport_model)
    _validate_projected_generator_contribution_classifications(
        projected_transport_model
    )
    _validate_projected_partition_derivations(projected_transport_model)
    _validate_projected_flux_derivations(projected_transport_model)
    _validate_mori_basis_functions(projected_transport_model)
    _validate_onsager_maxwell_stefan_operator(projected_transport_model)
    _validate_projected_proof_status(projected_transport_model)


def _validate_projected_primitive_set(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    primitive_set = projected_transport_model.projected_primitive_set
    state_count = len(projected_transport_model.state_labels)
    _validate_projected_transport_state_inventory(
        projected_transport_model.state_labels,
        np.asarray(
            projected_transport_model.stationary_concentrations_mol_m3,
            dtype=float,
        ),
        projected_transport_model.projected_transport_states,
        projected_transport_model.markov_additive_input.temperature_K,
    )
    if primitive_set.state_labels != projected_transport_model.state_labels:
        raise ValueError("projected primitive state labels mismatch")
    primitive_concentrations = _validated_state_concentrations(
        primitive_set.restricted_equilibrium_populations_c_i_mol_m3,
        state_count,
    )
    model_concentrations = np.asarray(
        projected_transport_model.stationary_concentrations_mol_m3,
        dtype=float,
    )
    if not np.allclose(
        primitive_concentrations,
        model_concentrations,
        rtol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        atol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("projected primitive c_i mismatch")
    if primitive_set.markov_input.state_labels != (
        projected_transport_model.markov_additive_input.state_labels
    ):
        raise ValueError("projected primitive Markov input mismatch")
    if not np.allclose(
        primitive_set.markov_input.state_concentrations_mol_m3,
        projected_transport_model.markov_additive_input.state_concentrations_mol_m3,
        rtol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        atol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("projected primitive Markov concentration mismatch")
    if primitive_set.markov_input.events != (
        projected_transport_model.markov_additive_input.events
    ):
        raise ValueError("projected primitive Markov event mismatch")
    if primitive_set.markov_input.temperature_K != (
        projected_transport_model.markov_additive_input.temperature_K
    ):
        raise ValueError("projected primitive Markov temperature mismatch")
    if not np.allclose(
        primitive_set.markov_conductivity_result.generator_s_inv,
        projected_transport_model.conductivity_result.generator_s_inv,
        rtol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        atol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("projected primitive conductivity result mismatch")
    if primitive_set.markov_conductivity_result.sigma_mS_cm != (
        projected_transport_model.conductivity_result.sigma_mS_cm
    ):
        raise ValueError("projected primitive conductivity sigma mismatch")
    generator_matrix = _validated_generator_matrix(
        primitive_set.reversible_generator_Q_ij_s_inv
    )
    if generator_matrix.shape != (state_count, state_count):
        raise ValueError("projected primitive Q_ij shape mismatch")
    flux_matrix = np.asarray(
        primitive_set.symmetric_reactive_fluxes_K_ij_mol_m3_s,
        dtype=float,
    )
    if flux_matrix.shape != (state_count, state_count):
        raise ValueError("projected primitive K_ij shape mismatch")
    _validate_symmetric_matrix(flux_matrix, "projected_primitive.K_ij")
    if float(np.min(flux_matrix)) < 0.0:
        raise ValueError("projected primitive K_ij must be nonnegative")
    expected_first_moment_shape = (
        state_count,
        state_count,
        int(CARTESIAN_AXIS_COUNT),
    )
    if (
        primitive_set.conditional_displacement_first_moments_d_ij_m.shape
        != expected_first_moment_shape
    ):
        raise ValueError("projected primitive d_ij shape mismatch")
    expected_second_moment_shape = (
        state_count,
        state_count,
        int(CARTESIAN_AXIS_COUNT),
        int(CARTESIAN_AXIS_COUNT),
    )
    if (
        primitive_set.conditional_displacement_second_moments_M_ij_m2.shape
        != expected_second_moment_shape
    ):
        raise ValueError("projected primitive M_ij shape mismatch")
    if primitive_set.self_current_diffusion_tensors_D_self_i_m2_s.shape != (
        state_count,
        int(CARTESIAN_AXIS_COUNT),
        int(CARTESIAN_AXIS_COUNT),
    ):
        raise ValueError("projected primitive D_self_i shape mismatch")
    mori_energy_matrix = np.asarray(
        primitive_set.mori_memory_energy_matrix_A,
        dtype=float,
    )
    if mori_energy_matrix.shape == (0, 0):
        _validated_current_coupling_matrix(
            primitive_set.mori_current_coupling_matrix_h,
            0,
        )
        return
    mori_energy_matrix = _validated_square_matrix(
        mori_energy_matrix,
        "projected_primitive.A",
    )
    _validate_symmetric_matrix(mori_energy_matrix, "projected_primitive.A")
    _validate_positive_semidefinite_matrix(mori_energy_matrix, "projected_primitive.A")
    _validated_current_coupling_matrix(
        primitive_set.mori_current_coupling_matrix_h,
        mori_energy_matrix.shape[0],
    )


def _validate_projected_generator_model(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    generator_model = projected_transport_model.projected_generator_model
    if (
        generator_model.microscopic_generator_source.descriptor_complete
        != projected_transport_model.closure_contract.descriptor_closure_derives_full_microscopic_generator
    ):
        raise ValueError("projected generator descriptor-completeness mismatch")
    if (
        generator_model.charge_polarization_source.observable_name
        != "unwrapped_charge_polarization"
    ):
        raise ValueError("projected generator charge polarization source mismatch")
    if generator_model.contribution_classifications != (
        projected_transport_model.contribution_classifications
    ):
        raise ValueError("projected generator classification inventory mismatch")
    if generator_model.populations.state_labels != (
        projected_transport_model.state_labels
    ):
        raise ValueError("projected generator population labels mismatch")
    if generator_model.populations.concentrations_mol_m3 != (
        projected_transport_model.stationary_concentrations_mol_m3
    ):
        raise ValueError("projected generator population concentrations mismatch")
    if generator_model.populations.stationary_probabilities != (
        projected_transport_model.stationary_probabilities
    ):
        raise ValueError("projected generator stationary probabilities mismatch")
    if generator_model.reactive_fluxes.reactive_fluxes != (
        projected_transport_model.reactive_flux_integrals
    ):
        raise ValueError("projected generator reactive flux set mismatch")
    if generator_model.displacement_moments.displacement_moments != (
        projected_transport_model.reactive_flux_integrals
    ):
        raise ValueError("projected generator displacement moment set mismatch")
    if generator_model.self_current_tensors.self_current_tensors != (
        projected_transport_model.self_displacement_moments
    ):
        raise ValueError("projected generator self-current set mismatch")
    if generator_model.basis.basis_labels != tuple(
        basis_function.label
        for basis_function in projected_transport_model.mori_basis_functions
    ):
        raise ValueError("projected generator basis labels mismatch")
    memory_self_energy_matrix = np.asarray(
        generator_model.mori_matrix.memory_self_energy_matrix,
        dtype=float,
    )
    if memory_self_energy_matrix.ndim == 1 and memory_self_energy_matrix.size == 0:
        memory_self_energy_matrix = memory_self_energy_matrix.reshape((0, 0))
    direct_energy_matrix = np.asarray(
        generator_model.mori_matrix.direct_energy_matrix,
        dtype=float,
    )
    if direct_energy_matrix.ndim == 1 and direct_energy_matrix.size == 0:
        direct_energy_matrix = direct_energy_matrix.reshape((0, 0))
    current_coupling_matrix = np.asarray(
        generator_model.current_coupling.current_coupling_matrix,
        dtype=float,
    )
    if memory_self_energy_matrix.shape == (0, 0):
        if direct_energy_matrix.shape != (0, 0):
            raise ValueError(
                "projected_generator.direct_energy_matrix must be 0x0 when "
                "memory_self_energy_matrix is 0x0"
            )
        _validated_current_coupling_matrix(current_coupling_matrix, 0)
    else:
        _validated_square_matrix(
            memory_self_energy_matrix,
            "projected_generator.memory_self_energy_matrix",
        )
        _validated_square_matrix(
            direct_energy_matrix,
            "projected_generator.direct_energy_matrix",
        )
        _validated_current_coupling_matrix(
            current_coupling_matrix,
            memory_self_energy_matrix.shape[0],
        )
    primitive_energy_matrix = np.asarray(
        projected_transport_model.projected_primitive_set.mori_memory_energy_matrix_A,
        dtype=float,
    )
    primitive_current_coupling_matrix = np.asarray(
        projected_transport_model.projected_primitive_set.mori_current_coupling_matrix_h,
        dtype=float,
    )
    if not np.allclose(
        memory_self_energy_matrix,
        primitive_energy_matrix,
        rtol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        atol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("projected generator A matrix does not match primitives")
    if not np.allclose(
        current_coupling_matrix,
        primitive_current_coupling_matrix,
        rtol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        atol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("projected generator h matrix does not match primitives")
    if generator_model.current_coupling.axis_count != int(CARTESIAN_AXIS_COUNT):
        raise ValueError("projected generator current-coupling axis count mismatch")
    if generator_model.conductivity.theorem_id != (
        projected_transport_model.closure_contract.readout_theorem
    ):
        raise ValueError("projected generator conductivity theorem mismatch")
    if generator_model.conductivity.direct_sigma_mS_cm != (
        projected_transport_model.conductivity_result.direct_sigma_mS_cm
    ):
        raise ValueError("projected generator direct conductivity mismatch")
    if generator_model.conductivity.corrector_sigma_mS_cm != (
        projected_transport_model.conductivity_result.corrector_sigma_mS_cm
    ):
        raise ValueError("projected generator corrector conductivity mismatch")
    if generator_model.conductivity.sigma_mS_cm != (
        projected_transport_model.conductivity_result.sigma_mS_cm
    ):
        raise ValueError("projected generator conductivity mismatch")


def _validate_projected_generator_contribution_classifications(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    if not projected_transport_model.contribution_classifications:
        raise ValueError("projected generator contribution classifications are empty")
    classification_by_label: dict[
        str, list[ProjectedGeneratorContributionClassification]
    ] = {}
    for classification in projected_transport_model.contribution_classifications:
        if classification.contribution_class not in (
            SUPPORTED_PROJECTED_GENERATOR_CONTRIBUTION_CLASSES
        ):
            raise ValueError(
                "unsupported projected generator contribution class "
                f"{classification.contribution_class}"
            )
        if not classification.label:
            raise ValueError("projected generator classification label is empty")
        if not classification.source:
            raise ValueError(
                f"{classification.label} projected generator source is empty"
            )
        if not classification.theorem_object:
            raise ValueError(
                f"{classification.label} projected generator theorem object is empty"
            )
        if (
            classification.contribution_class
            == PROJECTED_GENERATOR_CLASS_DIAGNOSTIC_ONLY
            and classification.conductivity_contribution_allowed
        ):
            raise ValueError(
                f"{classification.label} diagnostic label contributes to conductivity"
            )
        classification_by_label.setdefault(classification.label, []).append(
            classification
        )
    diagnostic_labels = {
        classification.label
        for classification in projected_transport_model.contribution_classifications
        if (
            classification.contribution_class
            == PROJECTED_GENERATOR_CLASS_DIAGNOSTIC_ONLY
        )
    }
    for reactive_flux in projected_transport_model.reactive_flux_integrals:
        _validate_diagnostic_label_absent_from_projected_object(
            diagnostic_labels,
            reactive_flux.from_state_label,
            "reactive_flux.from_state_label",
        )
        _validate_diagnostic_label_absent_from_projected_object(
            diagnostic_labels,
            reactive_flux.to_state_label,
            "reactive_flux.to_state_label",
        )
    for self_moment in projected_transport_model.self_displacement_moments:
        _validate_diagnostic_label_absent_from_projected_object(
            diagnostic_labels,
            self_moment.state_label,
            "self_current.state_label",
        )


def _validate_diagnostic_label_absent_from_projected_object(
    diagnostic_labels: set[str],
    projected_object_label: str,
    projected_object_field: str,
) -> None:
    for diagnostic_label in diagnostic_labels:
        if diagnostic_label in projected_object_label:
            raise ValueError(
                f"diagnostic label {diagnostic_label} appears in "
                f"{projected_object_field}={projected_object_label}"
            )


def _validate_molecular_free_energy_functional_derivation(
    free_energy_functional: MolecularFreeEnergyFunctionalUx,
) -> None:
    temperature_K = _positive_float(
        free_energy_functional.temperature_K,
        "U_x.temperature_K",
    )
    expected_beta_mol_per_J = 1.0 / (R * temperature_K)
    beta_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        abs(expected_beta_mol_per_J),
    )
    beta_residual = abs(free_energy_functional.beta_mol_per_J - expected_beta_mol_per_J)
    if beta_tolerance < beta_residual:
        raise ValueError("U_x beta_mol_per_J does not match 1/(R*T)")
    if not free_energy_functional.terms:
        raise ValueError("U_x derivation must contain at least one term")
    term_parameter_names: set[str] = set()
    for term in free_energy_functional.terms:
        if term.theorem_role != PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY:
            raise ValueError(f"{term.term_name} is not a U_x free-energy term")
        if term.units != "dimensionless_deltaG_over_RT":
            raise ValueError(f"{term.term_name} has invalid U_x units {term.units}")
        if not term.parameter_names:
            raise ValueError(f"{term.term_name} has no parameter provenance")
        term_parameter_names.update(term.parameter_names)
    expected_parameter_names = set(
        _primitive_parameter_names_for_theorem_role(
            PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
        )
    )
    if term_parameter_names != expected_parameter_names:
        raise ValueError(
            "U_x free-energy derivation parameter coverage mismatch: "
            f"{sorted(term_parameter_names)} != {sorted(expected_parameter_names)}"
        )


def _validate_projected_partition_derivations(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    if len(projected_transport_model.partition_states) != len(
        projected_transport_model.state_labels
    ):
        raise ValueError("projected partition derivation count does not match states")
    total_concentration_mol_m3 = math.fsum(
        projected_transport_model.stationary_concentrations_mol_m3
    )
    _positive_float(
        total_concentration_mol_m3,
        "projected total concentration for partition validation",
    )
    for partition_state in projected_transport_model.partition_states:
        derivation = partition_state.partition_derivation
        if derivation.state_label != partition_state.state_label:
            raise ValueError(
                f"{partition_state.state_label} partition derivation label mismatch"
            )
        if derivation.partition_label != partition_state.partition_definition:
            raise ValueError(
                f"{partition_state.state_label} partition definition mismatch"
            )
        derived_probability = (
            partition_state.concentration_mol_m3 / total_concentration_mol_m3
        )
        probability_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
            1.0,
            abs(derived_probability),
            abs(partition_state.stationary_probability),
        )
        probability_residual = abs(
            derived_probability - partition_state.restricted_partition_weight
        )
        if probability_tolerance < probability_residual:
            raise ValueError(
                f"{partition_state.state_label} restricted partition weight "
                "does not match concentration normalization"
            )
        _validate_pmf_partition_model(partition_state)
        if not derivation.normalization_rule:
            raise ValueError(
                f"{partition_state.state_label} partition derivation lacks "
                "normalization rule"
            )
        if not derivation.disjointness_rule:
            raise ValueError(
                f"{partition_state.state_label} partition derivation lacks "
                "disjointness rule"
            )


def _validate_projected_flux_derivations(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    for reactive_flux in projected_transport_model.reactive_flux_integrals:
        flux_derivation = reactive_flux.reactive_flux_derivation
        displacement_derivation = reactive_flux.displacement_moment_derivation
        _validate_derived_reactive_flux_model(reactive_flux)
        _validate_conditional_displacement_model(
            reactive_flux.conditional_displacement_model,
            reactive_flux.from_state_label,
            reactive_flux.to_state_label,
            reactive_flux.family_label,
            reactive_flux.charge_displacement_m,
            reactive_flux.charge_displacement_second_moment_m2,
        )
        _validate_reactive_flux_model(reactive_flux)
        _validate_conditional_displacement_moment_model(reactive_flux)
        if flux_derivation.from_state_label != reactive_flux.from_state_label:
            raise ValueError("reactive flux derivation from-state mismatch")
        if flux_derivation.to_state_label != reactive_flux.to_state_label:
            raise ValueError("reactive flux derivation to-state mismatch")
        if flux_derivation.family_label != reactive_flux.family_label:
            raise ValueError("reactive flux derivation family mismatch")
        if (
            flux_derivation.symmetric_flux_mol_m3_s
            != reactive_flux.symmetric_flux_mol_m3_s
        ):
            raise ValueError("reactive flux derivation symmetric flux mismatch")
        if flux_derivation.detailed_balance_condition == "":
            raise ValueError(
                "reactive flux derivation lacks detailed-balance condition"
            )
        _validate_displacement_derivation_matches_flux(
            displacement_derivation,
            reactive_flux.from_state_label,
            reactive_flux.to_state_label,
            reactive_flux.family_label,
            reactive_flux.charge_displacement_m,
        )
    for self_moment in projected_transport_model.self_displacement_moments:
        _validate_conditional_displacement_model(
            self_moment.conditional_displacement_model,
            self_moment.state_label,
            self_moment.state_label,
            self_moment.family_label,
            self_moment.charge_displacement_m,
            self_moment.charge_displacement_second_moment_m2,
        )
        _validate_self_current_projection_model(self_moment)
        _validate_self_current_tensor_model(self_moment)
        _validate_displacement_derivation_matches_flux(
            self_moment.displacement_moment_derivation,
            self_moment.state_label,
            self_moment.state_label,
            self_moment.family_label,
            self_moment.charge_displacement_m,
        )


def _validate_pmf_partition_model(
    partition_state: ProjectedTransportPartitionState,
) -> None:
    pmf_model = partition_state.pmf_partition_model
    if pmf_model.state_label != partition_state.state_label:
        raise ValueError("PMF partition state-label mismatch")
    if (
        pmf_model.restricted_partition_weight
        != partition_state.restricted_partition_weight
    ):
        raise ValueError("PMF partition weight mismatch")
    if pmf_model.concentration_mol_m3 != partition_state.concentration_mol_m3:
        raise ValueError("PMF partition concentration mismatch")
    if pmf_model.lower_bound < 0.0:
        raise ValueError("PMF partition lower bound must be nonnegative")
    if pmf_model.upper_bound < pmf_model.lower_bound:
        raise ValueError("PMF partition upper bound precedes lower bound")
    if not pmf_model.reaction_coordinate:
        raise ValueError("PMF partition lacks reaction coordinate")
    if not pmf_model.pmf_terms:
        raise ValueError("PMF partition lacks U_x terms")


def _validate_derived_reactive_flux_model(
    reactive_flux: ProjectedReactiveFluxIntegral,
) -> None:
    flux_model = reactive_flux.derived_reactive_flux_model
    if flux_model.from_partition != reactive_flux.from_state_label:
        raise ValueError("derived reactive flux from-partition mismatch")
    if flux_model.to_partition != reactive_flux.to_state_label:
        raise ValueError("derived reactive flux to-partition mismatch")
    if reactive_flux.family_label not in flux_model.transition_surface:
        raise ValueError("derived reactive flux transition surface lacks family label")
    if not flux_model.diffusion_tensor_source:
        raise ValueError("derived reactive flux lacks diffusion-tensor source")
    if not flux_model.friction_source:
        raise ValueError("derived reactive flux lacks friction source")
    _nonnegative_float(
        flux_model.free_energy_barrier_over_RT,
        f"{reactive_flux.family_label}.free_energy_barrier_over_RT",
    )
    _nonnegative_float(
        flux_model.partition_gap_scale_over_RT,
        f"{reactive_flux.family_label}.partition_gap_scale_over_RT",
    )
    _nonnegative_float(
        flux_model.free_energy_mismatch_over_RT,
        f"{reactive_flux.family_label}.free_energy_mismatch_over_RT",
    )
    if reactive_flux.family_label == "association_structural_hop":
        _positive_float(
            flux_model.effective_diffusivity_m2_s,
            f"{reactive_flux.family_label}.effective_diffusivity_m2_s",
        )
        _positive_float(
            flux_model.hop_length_m,
            f"{reactive_flux.family_label}.hop_length_m",
        )
    _positive_float(
        flux_model.transmission_coefficient,
        f"{reactive_flux.family_label}.transmission_coefficient",
    )
    if flux_model.detailed_balance_condition == "":
        raise ValueError("derived reactive flux lacks detailed-balance condition")
    if not flux_model.parameter_names:
        raise ValueError("derived reactive flux lacks parameter provenance")
    if not math.isclose(
        flux_model.symmetric_flux_mol_m3_s,
        reactive_flux.symmetric_flux_mol_m3_s,
        rel_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        abs_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("derived reactive flux symmetric flux mismatch")
    if not math.isclose(
        flux_model.forward_rate_s_inv,
        reactive_flux.forward_rate_s_inv,
        rel_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        abs_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("derived reactive flux forward-rate mismatch")
    if not math.isclose(
        flux_model.reverse_rate_s_inv,
        reactive_flux.reverse_rate_s_inv,
        rel_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        abs_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("derived reactive flux reverse-rate mismatch")


def _validate_reactive_flux_model(
    reactive_flux: ProjectedReactiveFluxIntegral,
) -> None:
    flux_model = reactive_flux.reactive_flux_model
    if flux_model.from_state != reactive_flux.from_state_label:
        raise ValueError("reactive flux model from-state mismatch")
    if flux_model.to_state != reactive_flux.to_state_label:
        raise ValueError("reactive flux model to-state mismatch")
    if not flux_model.reaction_coordinate:
        raise ValueError("reactive flux model lacks reaction coordinate")
    if not flux_model.free_energy_profile:
        raise ValueError("reactive flux model lacks free-energy profile")
    if not flux_model.coordinate_diffusion_profile:
        raise ValueError("reactive flux model lacks coordinate diffusion profile")
    if len(flux_model.free_energy_profile) != len(
        flux_model.coordinate_diffusion_profile
    ):
        raise ValueError("reactive flux profile lengths differ")
    for diffusion_m2_s in flux_model.coordinate_diffusion_profile:
        if reactive_flux.family_label == "association_structural_hop":
            _positive_float(diffusion_m2_s, "reactive_flux_model.D_xi_m2_s")
        else:
            _nonnegative_float(diffusion_m2_s, "reactive_flux_model.D_xi_m2_s")
    if not math.isclose(
        flux_model.symmetric_flux_mol_m3_s,
        reactive_flux.symmetric_flux_mol_m3_s,
        rel_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        abs_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("reactive flux model symmetric flux mismatch")


def _validate_conditional_displacement_moment_model(
    reactive_flux: ProjectedReactiveFluxIntegral,
) -> None:
    moment_model = reactive_flux.conditional_displacement_moment_model
    if moment_model.from_state != reactive_flux.from_state_label:
        raise ValueError("conditional moment model from-state mismatch")
    if moment_model.to_state != reactive_flux.to_state_label:
        raise ValueError("conditional moment model to-state mismatch")
    if tuple(moment_model.mean_displacement_m) != tuple(
        reactive_flux.charge_displacement_m
    ):
        raise ValueError("conditional moment model mean mismatch")
    if tuple(moment_model.second_moment_m2) != tuple(
        reactive_flux.charge_displacement_second_moment_m2
    ):
        raise ValueError("conditional moment model second-moment mismatch")
    _validate_second_moment_dominates_mean_outer_product(
        np.asarray(moment_model.mean_displacement_m, dtype=float),
        np.asarray(moment_model.second_moment_m2, dtype=float),
        "conditional_displacement_moment_model.second_moment_m2",
    )


def _validate_conditional_displacement_model(
    displacement_model: ConditionalChargeDisplacementModel,
    from_partition: str,
    to_partition: str,
    family_label: str,
    mean_displacement_m: tuple[float, float, float],
    second_moment_m2: tuple[tuple[float, float, float], ...],
) -> None:
    if displacement_model.from_partition != from_partition:
        raise ValueError("conditional displacement from-partition mismatch")
    if displacement_model.to_partition != to_partition:
        raise ValueError("conditional displacement to-partition mismatch")
    if displacement_model.family_label != family_label:
        raise ValueError("conditional displacement family mismatch")
    if displacement_model.reverse_rule != "antisymmetric_mean_symmetric_second_moment":
        raise ValueError("conditional displacement lacks reverse-moment rule")
    if not displacement_model.parameter_names:
        raise ValueError("conditional displacement lacks parameter provenance")
    expected_mean = tuple(
        float(value)
        for value in _validated_displacement(
            mean_displacement_m,
            f"{family_label}.expected_conditional_mean_displacement_m",
        )
    )
    if tuple(displacement_model.mean_displacement_m) != expected_mean:
        raise ValueError("conditional displacement mean mismatch")
    expected_second_moment = tuple(
        tuple(float(component) for component in row)
        for row in _validated_second_moment_tensor(
            second_moment_m2,
            f"{family_label}.expected_conditional_second_moment_m2",
        )
    )
    if tuple(displacement_model.second_moment_m2) != expected_second_moment:
        raise ValueError("conditional displacement second-moment mismatch")


def _validate_self_current_projection_model(
    self_moment: ProjectedSelfDisplacementMoment,
) -> None:
    self_current_model = self_moment.self_current_projection_model
    if self_current_model.partition != self_moment.state_label:
        raise ValueError("self-current projection partition mismatch")
    if self_current_model.family_label != self_moment.family_label:
        raise ValueError("self-current projection family mismatch")
    if self_current_model.source != self_moment.displacement_moment_source:
        raise ValueError("self-current projection source mismatch")
    if not self_current_model.parameter_names:
        raise ValueError("self-current projection lacks parameter provenance")
    expected_mean = tuple(
        float(value)
        for value in _validated_displacement(
            self_moment.charge_displacement_m,
            f"{self_moment.family_label}.expected_self_current_displacement_m",
        )
    )
    if tuple(self_current_model.mean_displacement_m) != expected_mean:
        raise ValueError("self-current projection mean mismatch")
    expected_second_moment_tensor = np.asarray(
        _validated_second_moment_tensor(
            self_moment.charge_displacement_second_moment_m2,
            f"{self_moment.family_label}.expected_self_second_moment_m2",
        ),
        dtype=float,
    )
    expected_tensor = expected_second_moment_tensor * self_moment.rate_s_inv
    observed_tensor = np.asarray(
        self_current_model.self_displacement_tensor_m2_s,
        dtype=float,
    )
    if not np.allclose(
        observed_tensor,
        expected_tensor,
        rtol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        atol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("self-current displacement tensor mismatch")
    expected_axis_density = self_moment.symmetric_flux_mol_m3_s * np.diag(
        expected_second_moment_tensor
    )
    observed_axis_density = np.asarray(
        self_current_model.direct_axis_density_m2_s_mol_m3,
        dtype=float,
    )
    if not np.allclose(
        observed_axis_density,
        expected_axis_density,
        rtol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        atol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("self-current direct-axis density mismatch")


def _validate_self_current_tensor_model(
    self_moment: ProjectedSelfDisplacementMoment,
) -> None:
    tensor_model = self_moment.self_current_tensor_model
    if tensor_model.state_label != self_moment.state_label:
        raise ValueError("self-current tensor state-label mismatch")
    if tensor_model.concentration_mol_m3 < 0.0:
        raise ValueError("self-current tensor concentration must be nonnegative")
    expected_tensor = tuple(
        tuple(float(component) for component in row)
        for row in self_moment.self_current_projection_model.self_displacement_tensor_m2_s
    )
    if tuple(tensor_model.diffusion_tensor_m2_s) != expected_tensor:
        raise ValueError("self-current tensor model mismatch")
    _validated_second_moment_tensor(
        tensor_model.diffusion_tensor_m2_s,
        "self_current_tensor_model.diffusion_tensor_m2_s",
    )


def _validate_displacement_derivation_matches_flux(
    displacement_derivation: ChargeDisplacementMomentDerivationDij,
    from_state_label: str,
    to_state_label: str,
    family_label: str,
    charge_displacement_m: tuple[float, float, float],
) -> None:
    if displacement_derivation.from_state_label != from_state_label:
        raise ValueError("displacement derivation from-state mismatch")
    if displacement_derivation.to_state_label != to_state_label:
        raise ValueError("displacement derivation to-state mismatch")
    if displacement_derivation.family_label != family_label:
        raise ValueError("displacement derivation family mismatch")
    if displacement_derivation.units != "charge_number_meter":
        raise ValueError("displacement derivation has invalid units")
    if displacement_derivation.reverse_displacement_rule != "d_jir_reverse=-d_ijr":
        raise ValueError("displacement derivation lacks antisymmetry rule")
    if tuple(displacement_derivation.charge_displacement_m) != tuple(
        charge_displacement_m
    ):
        raise ValueError("displacement derivation moment mismatch")


def _validate_mori_basis_functions(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    if not projected_transport_model.mori_basis_functions:
        raise ValueError("projected Mori basis must contain at least one observable")
    projected_state_labels = set(projected_transport_model.state_labels)
    observed_basis_labels: set[str] = set()
    for basis_function in projected_transport_model.mori_basis_functions:
        if basis_function.label == "":
            raise ValueError("projected Mori basis label must be nonempty")
        if basis_function.label in observed_basis_labels:
            raise ValueError(f"duplicate projected Mori basis {basis_function.label}")
        observed_basis_labels.add(basis_function.label)
        if basis_function.state_label not in projected_state_labels:
            raise ValueError("projected Mori basis state label is not projected")
        if basis_function.observable_definition == "":
            raise ValueError("projected Mori basis lacks observable definition")
        if basis_function.source not in SUPPORTED_MORI_BASIS_SOURCES:
            raise ValueError(
                f"unsupported projected Mori basis source {basis_function.source}"
            )
        if not basis_function.parameter_names:
            raise ValueError("projected Mori basis lacks parameter provenance")


def _validate_onsager_maxwell_stefan_operator(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    source_operator = projected_transport_model.onsager_transport_operator
    maxwell_stefan_operator = projected_transport_model.onsager_maxwell_stefan_operator
    if maxwell_stefan_operator.state_labels != source_operator.state_labels:
        raise ValueError("Onsager-Maxwell-Stefan state labels mismatch")
    if (
        maxwell_stefan_operator.concentrations_mol_m3
        != source_operator.concentrations_mol_m3
    ):
        raise ValueError("Onsager-Maxwell-Stefan concentrations mismatch")
    if maxwell_stefan_operator.charges != source_operator.charge_numbers:
        raise ValueError("Onsager-Maxwell-Stefan charges mismatch")
    if (
        maxwell_stefan_operator.self_frictions_J_s_mol_m2
        != source_operator.diagonal_friction_J_s_mol_m2
    ):
        raise ValueError("Onsager-Maxwell-Stefan self frictions mismatch")
    if maxwell_stefan_operator.pair_frictions != source_operator.friction_edges:
        raise ValueError("Onsager-Maxwell-Stefan pair frictions mismatch")
    if maxwell_stefan_operator.friction_matrix != source_operator.friction_matrix:
        raise ValueError("Onsager-Maxwell-Stefan friction matrix mismatch")
    if (
        maxwell_stefan_operator.mobility_matrix
        != source_operator.projected_mobility_matrix
    ):
        raise ValueError("Onsager-Maxwell-Stefan mobility matrix mismatch")
    if maxwell_stefan_operator.friction_matrix:
        _validate_symmetric_matrix(
            np.asarray(maxwell_stefan_operator.friction_matrix, dtype=float),
            "onsager_maxwell_stefan_operator.friction_matrix",
        )
        _validate_positive_semidefinite_matrix(
            np.asarray(maxwell_stefan_operator.friction_matrix, dtype=float),
            "onsager_maxwell_stefan_operator.friction_matrix",
        )
    if maxwell_stefan_operator.mobility_matrix:
        _validate_symmetric_matrix(
            np.asarray(maxwell_stefan_operator.mobility_matrix, dtype=float),
            "onsager_maxwell_stefan_operator.mobility_matrix",
        )
        _validate_positive_semidefinite_matrix(
            np.asarray(maxwell_stefan_operator.mobility_matrix, dtype=float),
            "onsager_maxwell_stefan_operator.mobility_matrix",
        )
    if not math.isclose(
        maxwell_stefan_operator.sigma_onsager_mS_cm,
        source_operator.onsager_sigma_mS_cm,
        rel_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
        abs_tol=FINITE_MARKOV_ADDITIVE_TOLERANCE,
    ):
        raise ValueError("Onsager-Maxwell-Stefan sigma mismatch")


def _validate_projected_proof_status(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    empirical_parameter_names = set(
        _projected_model_empirical_closure_parameter_names(projected_transport_model)
    )
    proof_status_claims_derived_closure = projected_transport_model.proof_status in (
        FINITE_PROJECTED_DESCRIPTOR_CLOSURE_PROOF_STATUS,
        FULL_MICROSCOPIC_GENERATOR_DERIVED_PROOF_STATUS,
    )
    if proof_status_claims_derived_closure and empirical_parameter_names:
        raise ValueError(
            "derived proof status cannot include empirical closure parameters: "
            f"{sorted(empirical_parameter_names)}"
        )
    if (
        projected_transport_model.closure_contract.descriptor_closure_derives_full_microscopic_generator
        and not projected_transport_model.closure_contract.descriptor_closure_derives_finite_projected_generator
    ):
        raise ValueError(
            "full microscopic generator proof requires finite projected generator proof"
        )
    if projected_transport_model.closure_contract.descriptor_closure_derives_full_microscopic_generator:
        _validate_full_microscopic_generator_proof_certificate(
            projected_transport_model,
            tuple(sorted(empirical_parameter_names)),
        )


def _validate_full_microscopic_generator_proof_certificate(
    projected_transport_model: ProjectedElectrolyteTransportModel,
    empirical_parameter_names: tuple[str, ...],
) -> None:
    if empirical_parameter_names:
        raise ValueError(
            "full microscopic generator proof cannot include empirical closure "
            f"parameters: {list(empirical_parameter_names)}"
        )
    closure_contract = projected_transport_model.closure_contract
    required_source_name_by_role = {
        "equilibrium_measure_source": (
            "descriptor_derived_equilibrium_measure_mu_x_from_U_x"
        ),
        "partition_source": "center_resolved_transport_partition_A_i",
        "concentration_source": "restricted_population_c_i=C_mu_x_A_i",
        "reactive_flux_source": (
            "descriptor_derived_symmetric_reactive_flux_K_ij_from_L_x_surface"
        ),
        "displacement_moment_source": (
            "descriptor_derived_conditional_charge_displacement_moments_from_P_x"
        ),
    }
    observed_source_name_by_role = {
        "equilibrium_measure_source": (
            closure_contract.equilibrium_measure_source.source_name
        ),
        "partition_source": closure_contract.partition_source.source_name,
        "concentration_source": closure_contract.concentration_source.source_name,
        "reactive_flux_source": closure_contract.reactive_flux_source.source_name,
        "displacement_moment_source": (
            closure_contract.displacement_moment_source.source_name
        ),
    }
    if observed_source_name_by_role != required_source_name_by_role:
        raise ValueError(
            "full microscopic generator proof has invalid primitive source map: "
            f"{observed_source_name_by_role}"
        )
    if set(closure_contract.primitive_parameter_theorem_role_by_name) != set(
        CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    ):
        raise ValueError(
            "full microscopic generator proof requires every primitive parameter "
            "to have a theorem role"
        )
    if not _descriptor_parameter_roles_define_full_microscopic_generator():
        raise ValueError(
            "full microscopic generator proof lacks U_x/A_i/L_x/P_x role coverage"
        )
    if (
        projected_transport_model.projected_generator_model.microscopic_generator_source.descriptor_complete
        is not True
    ):
        raise ValueError("full microscopic generator proof source is incomplete")
    _validate_full_proof_free_energy_source(projected_transport_model)
    _validate_full_proof_partition_sources(projected_transport_model)
    _validate_full_proof_flux_and_moment_sources(projected_transport_model)
    _validate_full_proof_mori_sources(projected_transport_model)


def _validate_full_proof_free_energy_source(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    free_energy_functional = projected_transport_model.free_energy_functional
    if free_energy_functional.empirical_closure_parameter_names:
        raise ValueError("full proof U_x source contains empirical parameters")
    if "mu_x" not in free_energy_functional.formula_id:
        raise ValueError("full proof equilibrium measure formula must name mu_x")
    if "U" not in free_energy_functional.formula_id:
        raise ValueError("full proof equilibrium measure formula must name U_x")
    for term in free_energy_functional.terms:
        if term.empirical_closure_parameter_names:
            raise ValueError(
                f"full proof U_x term {term.term_name} contains empirical parameters"
            )


def _validate_full_proof_partition_sources(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    if not projected_transport_model.partition_states:
        raise ValueError("full proof requires at least one restricted partition")
    for partition_state in projected_transport_model.partition_states:
        partition_derivation = partition_state.partition_derivation
        if partition_derivation.empirical_closure_parameter_names:
            raise ValueError(
                f"full proof partition {partition_state.state_label} contains "
                "empirical parameters"
            )
        if not partition_derivation.source_parameter_names:
            raise ValueError(
                f"full proof partition {partition_state.state_label} lacks "
                "A_i parameter provenance"
            )
        if (
            partition_state.population_source
            != "restricted_population_from_mu_x_over_A_i"
        ):
            raise ValueError(
                f"full proof partition {partition_state.state_label} population "
                "must be restricted mu_x over A_i"
            )


def _validate_full_proof_flux_and_moment_sources(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    if (
        not projected_transport_model.reactive_flux_integrals
        and not projected_transport_model.self_displacement_moments
    ):
        if projected_transport_model.conductivity_result.sigma_mS_cm != 0.0:
            raise ValueError("full proof has nonzero conductivity without primitives")
        return
    for reactive_flux in projected_transport_model.reactive_flux_integrals:
        flux_derivation = reactive_flux.reactive_flux_derivation
        displacement_derivation = reactive_flux.displacement_moment_derivation
        if flux_derivation.empirical_closure_parameter_names:
            raise ValueError(
                f"full proof flux {reactive_flux.family_label} contains empirical "
                "parameters"
            )
        if displacement_derivation.empirical_closure_parameter_names:
            raise ValueError(
                f"full proof displacement {reactive_flux.family_label} contains "
                "empirical parameters"
            )
        if "K_ij" not in flux_derivation.symmetric_flux_formula_id:
            raise ValueError(
                f"full proof flux {reactive_flux.family_label} lacks K_ij formula"
            )
        if "Q_ij" not in flux_derivation.generator_rate_formula_id:
            raise ValueError(
                f"full proof flux {reactive_flux.family_label} lacks Q_ij formula"
            )
        if "Delta_P_charge" not in displacement_derivation.moment_definition:
            raise ValueError(
                f"full proof displacement {reactive_flux.family_label} lacks P_x "
                "moment definition"
            )
        if not reactive_flux.derived_reactive_flux_model.parameter_names:
            raise ValueError(
                f"full proof flux {reactive_flux.family_label} lacks L_x parameter "
                "provenance"
            )
        if not reactive_flux.conditional_displacement_model.parameter_names:
            raise ValueError(
                f"full proof displacement {reactive_flux.family_label} lacks P_x "
                "parameter provenance"
            )
    for self_moment in projected_transport_model.self_displacement_moments:
        if self_moment.displacement_moment_derivation.empirical_closure_parameter_names:
            raise ValueError(
                f"full proof self current {self_moment.family_label} contains "
                "empirical parameters"
            )
        if not self_moment.self_current_projection_model.parameter_names:
            raise ValueError(
                f"full proof self current {self_moment.family_label} lacks D_self "
                "parameter provenance"
            )


def _validate_full_proof_mori_sources(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> None:
    if not projected_transport_model.mori_basis_functions:
        raise ValueError("full proof requires current-memory basis functions")
    for basis_function in projected_transport_model.mori_basis_functions:
        if not basis_function.parameter_names:
            raise ValueError(
                f"full proof Mori basis {basis_function.label} lacks A,h parameter "
                "provenance"
            )


def _projected_model_empirical_closure_parameter_names(
    projected_transport_model: ProjectedElectrolyteTransportModel,
) -> tuple[str, ...]:
    empirical_parameter_names: set[str] = set(
        projected_transport_model.closure_contract.equilibrium_measure_source.empirical_closure_parameter_names
    )
    empirical_parameter_names.update(
        projected_transport_model.closure_contract.partition_source.empirical_closure_parameter_names
    )
    empirical_parameter_names.update(
        projected_transport_model.closure_contract.reactive_flux_source.empirical_closure_parameter_names
    )
    empirical_parameter_names.update(
        projected_transport_model.closure_contract.displacement_moment_source.empirical_closure_parameter_names
    )
    empirical_parameter_names.update(
        projected_transport_model.free_energy_functional.empirical_closure_parameter_names
    )
    for partition_state in projected_transport_model.partition_states:
        empirical_parameter_names.update(
            partition_state.partition_derivation.empirical_closure_parameter_names
        )
    for reactive_flux in projected_transport_model.reactive_flux_integrals:
        empirical_parameter_names.update(
            reactive_flux.reactive_flux_derivation.empirical_closure_parameter_names
        )
        empirical_parameter_names.update(
            reactive_flux.displacement_moment_derivation.empirical_closure_parameter_names
        )
    for self_moment in projected_transport_model.self_displacement_moments:
        empirical_parameter_names.update(
            self_moment.displacement_moment_derivation.empirical_closure_parameter_names
        )
    return tuple(sorted(empirical_parameter_names))


def compute_markov_additive_event_family_attribution(
    markov_result: MarkovAdditiveConductivityResult,
    events: tuple[MarkovAdditiveEvent, ...],
    state_concentrations_mol_m3: np.ndarray,
    event_family_by_label: Mapping[str, str],
    temperature_K: float,
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ],
    onsager_transport_operator: OnsagerTransportOperator,
) -> tuple[MarkovAdditiveEventFamilyAttribution, ...]:
    """Attribute direct and corrector terms to event families with fixed Q."""

    temperature_K = _positive_float(temperature_K, "temperature_K")
    generator_matrix = _validated_generator_matrix(markov_result.generator_s_inv)
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        generator_matrix.shape[0],
    )
    validate_concentration_reversible_generator(generator_matrix, state_concentrations)
    validate_event_displacement_reversibility(
        events,
        state_concentrations,
        generator_matrix.shape[0],
    )
    event_generator_matrix = build_generator_from_events(
        generator_matrix.shape[0],
        events,
    )
    generator_tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    generator_difference = float(
        np.max(np.abs(event_generator_matrix - generator_matrix))
    )
    if generator_difference > generator_tolerance:
        raise ValueError(
            f"event-family attribution generator mismatch {generator_difference} "
            f"exceeds {generator_tolerance}"
        )
    family_labels = _event_family_labels(events, event_family_by_label)
    axis_count = int(CARTESIAN_AXIS_COUNT)
    direct_density_by_family: dict[str, np.ndarray] = {
        family_label: np.zeros(axis_count, dtype=float)
        for family_label in family_labels
    }
    drift_by_family: dict[str, np.ndarray] = {
        family_label: np.zeros((state_concentrations.shape[0], axis_count), dtype=float)
        for family_label in family_labels
    }
    for event in events:
        family_label = event_family_by_label[event.label]
        second_moment_tensor = np.asarray(
            event.charge_displacement_second_moment_m2,
            dtype=float,
        )
        direct_density_by_family[family_label] += (
            HALF_JUMP_VARIANCE_FACTOR
            * state_concentrations[event.from_state_index]
            * event.rate_s_inv
            * np.diag(second_moment_tensor)
        )
        displacement_array = np.asarray(event.charge_displacement_m, dtype=float)
        drift_by_family[family_label][event.from_state_index, :] += (
            event.rate_s_inv * displacement_array
        )

    symmetrized_energy_matrix = _symmetrized_energy_matrix(
        generator_matrix,
        state_concentrations,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized_energy_matrix)
    _validate_energy_eigenvalues(eigenvalues, "event_family_attribution.energy_matrix")
    total_drift_matrix = np.zeros(
        (state_concentrations.shape[0], axis_count), dtype=float
    )
    for family_label in family_labels:
        total_drift_matrix += drift_by_family[family_label]
    beta_factor = F * F / (R * temperature_K)
    total_direct_sigma_mS_cm = markov_result.direct_sigma_mS_cm
    total_sigma_mS_cm = markov_result.sigma_mS_cm
    attributions: list[MarkovAdditiveEventFamilyAttribution] = []
    for family_label in family_labels:
        family_drift_matrix = drift_by_family[family_label]
        direct_sigma_mS_cm = (
            beta_factor
            * float(np.sum(direct_density_by_family[family_label]))
            / CARTESIAN_AXIS_COUNT
            * S_M_TO_MS_CM
        )
        self_corrector_sigma_mS_cm = _family_cross_corrector_sigma_mS_cm(
            family_drift_matrix,
            family_drift_matrix,
            state_concentrations,
            eigenvalues,
            eigenvectors,
            beta_factor,
        )
        family_total_cross_sigma_mS_cm = _family_cross_corrector_sigma_mS_cm(
            family_drift_matrix,
            total_drift_matrix,
            state_concentrations,
            eigenvalues,
            eigenvectors,
            beta_factor,
        )
        marginal_corrector_sigma_mS_cm = (
            2.0 * family_total_cross_sigma_mS_cm - self_corrector_sigma_mS_cm
        )
        marginal_net_sigma_mS_cm = direct_sigma_mS_cm - marginal_corrector_sigma_mS_cm
        direct_fraction = (
            direct_sigma_mS_cm / total_direct_sigma_mS_cm
            if total_direct_sigma_mS_cm > 0.0
            else 0.0
        )
        marginal_net_fraction = (
            marginal_net_sigma_mS_cm / total_sigma_mS_cm
            if total_sigma_mS_cm > 0.0
            else 0.0
        )
        attributions.append(
            MarkovAdditiveEventFamilyAttribution(
                family_label=family_label,
                direct_sigma_mS_cm=float(direct_sigma_mS_cm),
                self_corrector_sigma_mS_cm=float(self_corrector_sigma_mS_cm),
                marginal_corrector_sigma_mS_cm=float(marginal_corrector_sigma_mS_cm),
                marginal_net_sigma_mS_cm=float(marginal_net_sigma_mS_cm),
                direct_fraction=float(direct_fraction),
                marginal_net_fraction=float(marginal_net_fraction),
            )
        )
    if onsager_transport_operator.state_labels:
        onsager_sigma_mS_cm = _nonnegative_float(
            onsager_transport_operator.onsager_sigma_mS_cm,
            "onsager_transport_operator.onsager_sigma_mS_cm",
        )
        onsager_direct_sigma_mS_cm = _nonnegative_float(
            onsager_transport_operator.nernst_einstein_sigma_mS_cm,
            "onsager_transport_operator.nernst_einstein_sigma_mS_cm",
        )
        onsager_corrector_sigma_mS_cm = _nonnegative_float(
            onsager_transport_operator.correlation_corrector_mS_cm,
            "onsager_transport_operator.correlation_corrector_mS_cm",
        )
        onsager_marginal_net_fraction = 0.0
        if not total_sigma_mS_cm <= 0.0:
            onsager_marginal_net_fraction = onsager_sigma_mS_cm / total_sigma_mS_cm
        onsager_direct_fraction = 0.0
        if not total_direct_sigma_mS_cm <= 0.0:
            onsager_direct_fraction = (
                onsager_direct_sigma_mS_cm / total_direct_sigma_mS_cm
            )
        attributions.append(
            MarkovAdditiveEventFamilyAttribution(
                family_label="onsager_self_transport",
                direct_sigma_mS_cm=float(onsager_direct_sigma_mS_cm),
                self_corrector_sigma_mS_cm=float(onsager_corrector_sigma_mS_cm),
                marginal_corrector_sigma_mS_cm=float(onsager_corrector_sigma_mS_cm),
                marginal_net_sigma_mS_cm=float(onsager_sigma_mS_cm),
                direct_fraction=float(onsager_direct_fraction),
                marginal_net_fraction=float(onsager_marginal_net_fraction),
            )
        )
    for memory_family_label in sorted(
        {
            memory_correction.memory_family_label
            for memory_correction in projected_current_memory_corrections
        }
    ):
        current_memory_corrector_sigma_mS_cm = math.fsum(
            memory_correction.correction_sigma_mS_cm
            for memory_correction in projected_current_memory_corrections
            if memory_correction.memory_family_label == memory_family_label
        )
        current_memory_marginal_net_fraction = 0.0
        if not total_sigma_mS_cm <= 0.0:
            current_memory_marginal_net_fraction = float(
                -current_memory_corrector_sigma_mS_cm / total_sigma_mS_cm
            )
        attributions.append(
            MarkovAdditiveEventFamilyAttribution(
                family_label=memory_family_label,
                direct_sigma_mS_cm=0.0,
                self_corrector_sigma_mS_cm=0.0,
                marginal_corrector_sigma_mS_cm=float(
                    current_memory_corrector_sigma_mS_cm
                ),
                marginal_net_sigma_mS_cm=float(-current_memory_corrector_sigma_mS_cm),
                direct_fraction=0.0,
                marginal_net_fraction=current_memory_marginal_net_fraction,
            )
        )
    return tuple(
        sorted(
            attributions,
            key=_event_family_attribution_abs_marginal_net_sigma,
            reverse=True,
        )
    )


def _event_family_attribution_abs_marginal_net_sigma(
    attribution: MarkovAdditiveEventFamilyAttribution,
) -> float:
    return abs(attribution.marginal_net_sigma_mS_cm)


def _event_family_labels(
    events: tuple[MarkovAdditiveEvent, ...],
    event_family_by_label: Mapping[str, str],
) -> tuple[str, ...]:
    family_labels: list[str] = []
    for event in events:
        if event.label not in event_family_by_label:
            raise ValueError(f"missing event family for event {event.label}")
        mapped_family_label = event_family_by_label[event.label]
        if mapped_family_label != event.family_label:
            raise ValueError(f"event family mapping disagrees with event {event.label}")
        if mapped_family_label == "":
            raise ValueError(f"event {event.label} has an empty family label")
        if mapped_family_label not in family_labels:
            family_labels.append(mapped_family_label)
    return tuple(family_labels)


def _family_cross_corrector_sigma_mS_cm(
    left_drift_by_state: np.ndarray,
    right_drift_by_state: np.ndarray,
    state_concentrations: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    beta_factor: float,
) -> float:
    sqrt_concentrations = np.sqrt(state_concentrations)
    left_current_coupling_matrix = (
        sqrt_concentrations[:, None] * left_drift_by_state
    ).T
    right_current_coupling_matrix = (
        sqrt_concentrations[:, None] * right_drift_by_state
    ).T
    cross_density_sum = 0.0
    for axis_index in range(int(CARTESIAN_AXIS_COUNT)):
        cross_density_sum += _projected_cross_form(
            eigenvalues,
            eigenvectors,
            left_current_coupling_matrix[axis_index, :],
            right_current_coupling_matrix[axis_index, :],
        )
    return float(beta_factor * cross_density_sum / CARTESIAN_AXIS_COUNT * S_M_TO_MS_CM)


def _projected_cross_form(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    left_current_coupling: np.ndarray,
    right_current_coupling: np.ndarray,
) -> float:
    positive_mode_mask = _positive_energy_mode_mask(eigenvalues)
    if not np.any(positive_mode_mask):
        return 0.0
    projected_left = eigenvectors[:, positive_mode_mask].T @ left_current_coupling
    projected_right = eigenvectors[:, positive_mode_mask].T @ right_current_coupling
    return float(
        math.fsum(
            float(left_value * right_value / eigenvalue)
            for left_value, right_value, eigenvalue in zip(
                projected_left,
                projected_right,
                eigenvalues[positive_mode_mask],
            )
        )
    )


def _validate_energy_eigenvalues(eigenvalues: np.ndarray, context: str) -> None:
    eigenvalue_scale = max(
        float(np.max(np.abs(eigenvalues))), FINITE_MARKOV_ADDITIVE_TOLERANCE
    )
    allowed_negative_eigenvalue = FINITE_MARKOV_ADDITIVE_TOLERANCE * eigenvalue_scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(f"{context} must be positive semidefinite")


def _positive_energy_mode_mask(eigenvalues: np.ndarray) -> np.ndarray:
    eigenvalue_scale = max(
        float(np.max(np.abs(eigenvalues))), FINITE_MARKOV_ADDITIVE_TOLERANCE
    )
    return eigenvalues > FINITE_MARKOV_ADDITIVE_TOLERANCE * eigenvalue_scale


def _symmetrized_energy_matrix(
    generator_matrix: np.ndarray,
    state_concentrations: np.ndarray,
) -> np.ndarray:
    sqrt_concentrations = np.sqrt(state_concentrations)
    inverse_sqrt_concentrations = 1.0 / sqrt_concentrations
    energy_matrix = (
        sqrt_concentrations[:, None]
        * (-generator_matrix)
        * inverse_sqrt_concentrations[None, :]
    )
    return _symmetrized_matrix(energy_matrix)


def _validated_generator_matrix(generator_matrix_s_inv: np.ndarray) -> np.ndarray:
    generator_matrix = np.asarray(generator_matrix_s_inv, dtype=float)
    if generator_matrix.ndim != 2:
        raise ValueError("generator_matrix_s_inv must be two-dimensional")
    if generator_matrix.shape[0] != generator_matrix.shape[1]:
        raise ValueError("generator_matrix_s_inv must be square")
    if generator_matrix.shape[0] == 0:
        raise ValueError("generator_matrix_s_inv must contain at least one state")
    if not np.all(np.isfinite(generator_matrix)):
        raise ValueError("generator_matrix_s_inv contains non-finite values")
    return generator_matrix


def _validated_state_concentrations(
    state_concentrations_mol_m3: np.ndarray,
    state_count: int,
) -> np.ndarray:
    state_concentrations = np.asarray(state_concentrations_mol_m3, dtype=float)
    if state_concentrations.ndim != 1:
        raise ValueError("state_concentrations_mol_m3 must be one-dimensional")
    if state_concentrations.shape[0] != state_count:
        raise ValueError("state_concentrations_mol_m3 length must equal state count")
    if not np.all(np.isfinite(state_concentrations)):
        raise ValueError("state_concentrations_mol_m3 contains non-finite values")
    if np.any(state_concentrations <= ZERO_VALUE):
        raise ValueError("state_concentrations_mol_m3 must be strictly positive")
    return state_concentrations


def _validate_event_indices(
    event: MarkovAdditiveEvent,
    state_count: int,
) -> None:
    if not isinstance(event.from_state_index, int):
        raise TypeError(f"{event.label}.from_state_index must be an integer")
    if not isinstance(event.to_state_index, int):
        raise TypeError(f"{event.label}.to_state_index must be an integer")
    if event.from_state_index < 0 or event.from_state_index >= state_count:
        raise ValueError(f"{event.label}.from_state_index is outside the state range")
    if event.to_state_index < 0 or event.to_state_index >= state_count:
        raise ValueError(f"{event.label}.to_state_index is outside the state range")


def _validated_displacement(
    charge_displacement_m: tuple[float, float, float],
    event_label: str,
) -> np.ndarray:
    displacement_array = np.asarray(charge_displacement_m, dtype=float)
    if displacement_array.shape != (int(CARTESIAN_AXIS_COUNT),):
        raise ValueError(f"{event_label}.charge_displacement_m must have length three")
    if not np.all(np.isfinite(displacement_array)):
        raise ValueError(
            f"{event_label}.charge_displacement_m contains non-finite values"
        )
    return displacement_array


def _validated_second_moment_tensor(
    charge_displacement_second_moment_m2: tuple[tuple[float, float, float], ...],
    event_label: str,
) -> np.ndarray:
    second_moment_tensor = np.asarray(
        charge_displacement_second_moment_m2,
        dtype=float,
    )
    axis_count = int(CARTESIAN_AXIS_COUNT)
    if second_moment_tensor.shape != (axis_count, axis_count):
        raise ValueError(
            f"{event_label}.charge_displacement_second_moment_m2 must be a 3x3 tensor"
        )
    if not np.all(np.isfinite(second_moment_tensor)):
        raise ValueError(
            f"{event_label}.charge_displacement_second_moment_m2 contains non-finite values"
        )
    _validate_symmetric_matrix(
        second_moment_tensor,
        f"{event_label}.charge_displacement_second_moment_m2",
    )
    _validate_positive_semidefinite_matrix(
        second_moment_tensor,
        f"{event_label}.charge_displacement_second_moment_m2",
    )
    return second_moment_tensor


def _validate_second_moment_dominates_mean_outer_product(
    displacement_array: np.ndarray,
    second_moment_tensor: np.ndarray,
    event_label: str,
) -> None:
    conditional_covariance_tensor = second_moment_tensor - np.outer(
        displacement_array, displacement_array
    )
    _validate_symmetric_matrix(
        conditional_covariance_tensor,
        f"{event_label}.conditional_covariance_m2",
    )
    _validate_positive_semidefinite_matrix(
        conditional_covariance_tensor,
        f"{event_label}.conditional_covariance_m2",
    )


def _is_zero_displacement(displacement_array: np.ndarray) -> bool:
    return bool(np.all(displacement_array == ZERO_VALUE))


def _displacement_key(displacement_array: np.ndarray) -> tuple[float, float, float]:
    return tuple(
        _canonical_float_for_key(float(component)) for component in displacement_array
    )


def _second_moment_key(
    second_moment_tensor_m2: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(_canonical_float_for_key(float(component)) for component in row)
        for row in second_moment_tensor_m2
    )


def _canonical_float_for_key(value: float) -> float:
    if value == ZERO_VALUE:
        return ZERO_VALUE
    return float(value)


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= ZERO_VALUE:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _symmetrized_matrix(matrix: np.ndarray) -> np.ndarray:
    return HALF_JUMP_VARIANCE_FACTOR * (matrix + matrix.T)


def _matrix_tolerance(
    generator_matrix: np.ndarray,
    state_concentrations: np.ndarray,
) -> float:
    return FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(generator_matrix))),
        float(np.max(np.abs(state_concentrations))),
        float(np.sum(state_concentrations)),
    )


# ---- ion_atmosphere.py ----

SUPPORTED_ION_ATMOSPHERE_SOLVERS = ("off", "diagonal_pnp_stokes_l1_cell_experimental")
SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS = ("off", "finite_size_bulk_pnp_stokes_l1_cell")
STOKES_TRANSLATION_AXIS_COUNT = (
    3  # Cartesian translation axes in the spherical Stokes drag solution.
)
STOKES_NO_SLIP_BOUNDARY_FACTOR = (
    2  # No-slip sphere doubles the axis count in zeta = 6*pi*eta*a.
)
STOKES_SPHERE_DRAG_FACTOR = (
    STOKES_NO_SLIP_BOUNDARY_FACTOR * STOKES_TRANSLATION_AXIS_COUNT
)
SPHERE_VOLUME_NUMERATOR = STOKES_SPHERE_DRAG_FACTOR - STOKES_NO_SLIP_BOUNDARY_FACTOR
SPHERE_VOLUME_DENOMINATOR = STOKES_TRANSLATION_AXIS_COUNT


@dataclass(frozen=True)
class IonAtmosphereInput:
    carrier_concentrations_mol_m3: Mapping[str, float]
    carrier_charges: Mapping[str, int]
    local_diffusivity_m2_s_by_carrier: Mapping[str, float]
    hydrodynamic_radius_m_by_carrier: Mapping[str, float]
    viscosity_Pa_s: float
    relative_dielectric: float
    temperature_K: float
    solver: str


@dataclass(frozen=True)
class BulkIonAtmosphereInput:
    carrier_labels: tuple[str, ...]
    carrier_concentrations_mol_m3: Mapping[str, float]
    carrier_charges: Mapping[str, int]
    local_diffusivity_m2_s_by_carrier: Mapping[str, float]
    hydrodynamic_radius_m_by_carrier: Mapping[str, float]
    viscosity_Pa_s: float
    relative_dielectric: float
    temperature_K: float
    solver: str


@dataclass(frozen=True)
class IonAtmosphereState:
    kappa_inv_m: float
    ionic_strength_mol_m3: float
    friction_ratio_by_carrier: dict[str, float]
    zeta0_by_carrier: dict[str, float]
    zeta_ep_by_carrier: dict[str, float]
    zeta_rel_by_carrier: dict[str, float]
    zeta_atm_by_carrier: dict[str, float]
    D_micro_by_carrier: dict[str, float]
    D_eff_by_carrier: dict[str, float]
    solver: str


@dataclass(frozen=True)
class BulkIonAtmosphereState:
    carrier_labels: tuple[str, ...]
    kappa_inv_m: float
    ionic_strength_mol_m3: float
    ambipolar_diffusivity_m2_s: float
    resistance_matrix_kg_s: np.ndarray
    resistance_ep_kg_s: np.ndarray
    resistance_rel_kg_s: np.ndarray
    maxwell_stefan_pair_friction_coefficient_matrix_J_s_mol_m2: np.ndarray
    maxwell_stefan_pair_drag_matrix_J_s_mol_m2: np.ndarray
    steric_volume_fraction: float
    thermodynamic_factor_trace: float
    thermodynamic_factor_matrix: np.ndarray
    thermodynamic_factor_eigenvalues: tuple[float, ...]
    structure_response_matrix: np.ndarray
    structure_factor_charge_mode: float
    kappa_radius_by_carrier: dict[str, float]
    solver: str


def build_ion_atmosphere_state(
    ion_atmosphere_input: IonAtmosphereInput,
) -> IonAtmosphereState:
    """Build ion-atmosphere friction diagnostics for charged mobile carriers."""

    _assert_positive_finite(ion_atmosphere_input.viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(
        ion_atmosphere_input.relative_dielectric, "relative_dielectric"
    )
    _assert_positive_finite(ion_atmosphere_input.temperature_K, "temperature_K")
    _validate_solver(ion_atmosphere_input.solver)

    carrier_names = tuple(ion_atmosphere_input.carrier_concentrations_mol_m3)
    if not carrier_names:
        raise ValueError("ion atmosphere requires at least one charged carrier")

    charge_weighted_concentration_mol_m3 = 0.0
    friction_ratio_by_carrier: dict[str, float] = {}
    zeta0_by_carrier: dict[str, float] = {}
    zeta_ep_by_carrier: dict[str, float] = {}
    zeta_rel_by_carrier: dict[str, float] = {}
    zeta_atm_by_carrier: dict[str, float] = {}
    D_micro_by_carrier: dict[str, float] = {}
    D_eff_by_carrier: dict[str, float] = {}
    local_diffusivity_by_carrier: dict[str, float] = {}
    hydrodynamic_radius_by_carrier: dict[str, float] = {}
    charge_by_carrier: dict[str, int] = {}

    for carrier_name in carrier_names:
        concentration_mol_m3 = _require_nonnegative_finite(
            ion_atmosphere_input.carrier_concentrations_mol_m3,
            carrier_name,
            "carrier_concentrations_mol_m3",
        )
        charge_number = _require_charge(
            ion_atmosphere_input.carrier_charges, carrier_name
        )
        local_diffusivity_m2_s = _require_positive_finite(
            ion_atmosphere_input.local_diffusivity_m2_s_by_carrier,
            carrier_name,
            "local_diffusivity_m2_s_by_carrier",
        )
        hydrodynamic_radius_m = _require_positive_finite(
            ion_atmosphere_input.hydrodynamic_radius_m_by_carrier,
            carrier_name,
            "hydrodynamic_radius_m_by_carrier",
        )
        charge_weighted_concentration_mol_m3 += (
            charge_number * charge_number * concentration_mol_m3
        )
        local_diffusivity_by_carrier[carrier_name] = local_diffusivity_m2_s
        hydrodynamic_radius_by_carrier[carrier_name] = hydrodynamic_radius_m
        charge_by_carrier[carrier_name] = charge_number

    kappa_inv_m = _debye_kappa_inv_m(
        charge_weighted_concentration_mol_m3,
        ion_atmosphere_input.relative_dielectric,
        ion_atmosphere_input.temperature_K,
    )
    if math.isinf(kappa_inv_m):
        kappa_m_inv = 0.0
    else:
        kappa_m_inv = 1.0 / kappa_inv_m

    for carrier_name in carrier_names:
        local_diffusivity_m2_s = local_diffusivity_by_carrier[carrier_name]
        hydrodynamic_radius_m = hydrodynamic_radius_by_carrier[carrier_name]
        charge_number = charge_by_carrier[carrier_name]
        zeta0_kg_s = K_B * ion_atmosphere_input.temperature_K / local_diffusivity_m2_s
        if ion_atmosphere_input.solver == "off":
            zeta_ep_kg_s = 0.0
            zeta_rel_kg_s = 0.0
        elif ion_atmosphere_input.solver == "diagonal_pnp_stokes_l1_cell_experimental":
            zeta_ep_kg_s = _electrophoretic_drag_kg_s(
                viscosity_Pa_s=ion_atmosphere_input.viscosity_Pa_s,
                hydrodynamic_radius_m=hydrodynamic_radius_m,
                kappa_m_inv=kappa_m_inv,
            )
            zeta_rel_kg_s = _relaxation_drag_kg_s(
                charge_number=charge_number,
                local_diffusivity_m2_s=local_diffusivity_m2_s,
                hydrodynamic_radius_m=hydrodynamic_radius_m,
                relative_dielectric=ion_atmosphere_input.relative_dielectric,
                kappa_m_inv=kappa_m_inv,
            )
        else:
            raise ValueError(
                f"Unsupported ion-atmosphere solver {ion_atmosphere_input.solver!r}"
            )
        zeta_atm_kg_s = zeta_ep_kg_s + zeta_rel_kg_s
        _assert_nonnegative_finite(zeta_ep_kg_s, f"{carrier_name}.zeta_ep_kg_s")
        _assert_nonnegative_finite(zeta_rel_kg_s, f"{carrier_name}.zeta_rel_kg_s")
        _assert_nonnegative_finite(zeta_atm_kg_s, f"{carrier_name}.zeta_atm_kg_s")
        friction_ratio = zeta0_kg_s / (zeta0_kg_s + zeta_atm_kg_s)
        if friction_ratio <= 0.0 or friction_ratio > 1.0:
            raise ValueError(
                f"{carrier_name}.friction_ratio must be in (0, 1], got {friction_ratio}"
            )

        zeta0_by_carrier[carrier_name] = zeta0_kg_s
        zeta_ep_by_carrier[carrier_name] = zeta_ep_kg_s
        zeta_rel_by_carrier[carrier_name] = zeta_rel_kg_s
        zeta_atm_by_carrier[carrier_name] = zeta_atm_kg_s
        D_micro_by_carrier[carrier_name] = local_diffusivity_m2_s
        D_eff_by_carrier[carrier_name] = local_diffusivity_m2_s * friction_ratio
        friction_ratio_by_carrier[carrier_name] = friction_ratio

    return IonAtmosphereState(
        kappa_inv_m=kappa_inv_m,
        ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
        friction_ratio_by_carrier=friction_ratio_by_carrier,
        zeta0_by_carrier=zeta0_by_carrier,
        zeta_ep_by_carrier=zeta_ep_by_carrier,
        zeta_rel_by_carrier=zeta_rel_by_carrier,
        zeta_atm_by_carrier=zeta_atm_by_carrier,
        D_micro_by_carrier=D_micro_by_carrier,
        D_eff_by_carrier=D_eff_by_carrier,
        solver=ion_atmosphere_input.solver,
    )


def build_bulk_ion_atmosphere_state(
    bulk_ion_atmosphere_input: BulkIonAtmosphereInput,
) -> BulkIonAtmosphereState:
    """Build a finite-size bulk carrier atmosphere resistance matrix."""

    _assert_positive_finite(bulk_ion_atmosphere_input.viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(
        bulk_ion_atmosphere_input.relative_dielectric, "relative_dielectric"
    )
    _assert_positive_finite(bulk_ion_atmosphere_input.temperature_K, "temperature_K")
    _validate_bulk_solver(bulk_ion_atmosphere_input.solver)
    carrier_labels = tuple(bulk_ion_atmosphere_input.carrier_labels)
    if not carrier_labels:
        raise ValueError("bulk ion atmosphere requires at least one charged carrier")
    if len(set(carrier_labels)) != len(carrier_labels):
        raise ValueError("bulk ion atmosphere carrier_labels must be unique")

    charge_weighted_concentration_mol_m3 = 0.0
    steric_volume_fraction = 0.0
    concentration_by_carrier: dict[str, float] = {}
    charge_by_carrier: dict[str, int] = {}
    diffusivity_by_carrier: dict[str, float] = {}
    radius_by_carrier: dict[str, float] = {}
    for carrier_label in carrier_labels:
        concentration_mol_m3 = _require_nonnegative_finite(
            bulk_ion_atmosphere_input.carrier_concentrations_mol_m3,
            carrier_label,
            "carrier_concentrations_mol_m3",
        )
        charge_number = _require_charge(
            bulk_ion_atmosphere_input.carrier_charges, carrier_label
        )
        local_diffusivity_m2_s = _require_positive_finite(
            bulk_ion_atmosphere_input.local_diffusivity_m2_s_by_carrier,
            carrier_label,
            "local_diffusivity_m2_s_by_carrier",
        )
        hydrodynamic_radius_m = _require_positive_finite(
            bulk_ion_atmosphere_input.hydrodynamic_radius_m_by_carrier,
            carrier_label,
            "hydrodynamic_radius_m_by_carrier",
        )
        charge_weighted_concentration_mol_m3 += (
            charge_number * charge_number * concentration_mol_m3
        )
        steric_volume_fraction += (
            concentration_mol_m3 * N_A * _sphere_volume_m3(hydrodynamic_radius_m)
        )
        concentration_by_carrier[carrier_label] = concentration_mol_m3
        charge_by_carrier[carrier_label] = charge_number
        diffusivity_by_carrier[carrier_label] = local_diffusivity_m2_s
        radius_by_carrier[carrier_label] = hydrodynamic_radius_m
    _assert_nonnegative_finite(steric_volume_fraction, "steric_volume_fraction")
    if steric_volume_fraction >= 1.0:
        raise ValueError(
            f"steric_volume_fraction must be below one, got {steric_volume_fraction}"
        )
    ambipolar_diffusivity_m2_s = _ambipolar_diffusivity_m2_s(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        diffusivity_by_carrier=diffusivity_by_carrier,
    )

    kappa_inv_m = _debye_kappa_inv_m(
        charge_weighted_concentration_mol_m3,
        bulk_ion_atmosphere_input.relative_dielectric,
        bulk_ion_atmosphere_input.temperature_K,
    )
    if (
        bulk_ion_atmosphere_input.solver == "off"
        or charge_weighted_concentration_mol_m3 == 0.0
    ):
        matrix_shape = (len(carrier_labels), len(carrier_labels))
        zero_matrix = np.zeros(matrix_shape, dtype=float)
        thermodynamic_factor_matrix = _finite_size_thermodynamic_factor_matrix(
            carrier_labels=carrier_labels,
            concentration_by_carrier=concentration_by_carrier,
            radius_by_carrier=radius_by_carrier,
            steric_volume_fraction=steric_volume_fraction,
        )
        thermodynamic_factor_trace = float(np.trace(thermodynamic_factor_matrix))
        thermodynamic_factor_eigenvalues = _matrix_eigenvalue_tuple(
            thermodynamic_factor_matrix,
            "thermodynamic_factor_matrix",
        )
        structure_response_matrix = thermodynamic_factor_matrix.copy()
        return BulkIonAtmosphereState(
            carrier_labels=carrier_labels,
            kappa_inv_m=kappa_inv_m,
            ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
            ambipolar_diffusivity_m2_s=ambipolar_diffusivity_m2_s,
            resistance_matrix_kg_s=zero_matrix,
            resistance_ep_kg_s=zero_matrix.copy(),
            resistance_rel_kg_s=zero_matrix.copy(),
            maxwell_stefan_pair_friction_coefficient_matrix_J_s_mol_m2=(
                zero_matrix.copy()
            ),
            maxwell_stefan_pair_drag_matrix_J_s_mol_m2=zero_matrix.copy(),
            steric_volume_fraction=steric_volume_fraction,
            thermodynamic_factor_trace=thermodynamic_factor_trace,
            thermodynamic_factor_matrix=thermodynamic_factor_matrix,
            thermodynamic_factor_eigenvalues=thermodynamic_factor_eigenvalues,
            structure_response_matrix=structure_response_matrix,
            structure_factor_charge_mode=0.0,
            kappa_radius_by_carrier={
                carrier_label: 0.0 for carrier_label in carrier_labels
            },
            solver=bulk_ion_atmosphere_input.solver,
        )
    if bulk_ion_atmosphere_input.solver != "finite_size_bulk_pnp_stokes_l1_cell":
        raise ValueError(
            f"Unsupported bulk ion-atmosphere solver {bulk_ion_atmosphere_input.solver!r}"
        )
    if math.isinf(kappa_inv_m):
        kappa_m_inv = 0.0
    else:
        kappa_m_inv = 1.0 / kappa_inv_m
    finite_size_result = _finite_size_bulk_resistance_matrices_kg_s(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        diffusivity_by_carrier=diffusivity_by_carrier,
        radius_by_carrier=radius_by_carrier,
        viscosity_Pa_s=bulk_ion_atmosphere_input.viscosity_Pa_s,
        relative_dielectric=bulk_ion_atmosphere_input.relative_dielectric,
        kappa_m_inv=kappa_m_inv,
        steric_volume_fraction=steric_volume_fraction,
    )
    zeta_ep_values_kg_s = finite_size_result[0]
    zeta_rel_values_kg_s = finite_size_result[1]
    kappa_radius_by_carrier = finite_size_result[2]
    overlap_values = finite_size_result[3]
    relaxation_sign_values = finite_size_result[4]
    thermodynamic_factor_matrix = _finite_size_thermodynamic_factor_matrix(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        radius_by_carrier=radius_by_carrier,
        steric_volume_fraction=steric_volume_fraction,
    )
    thermodynamic_factor_trace = float(np.trace(thermodynamic_factor_matrix))
    thermodynamic_factor_eigenvalues = _matrix_eigenvalue_tuple(
        thermodynamic_factor_matrix,
        "thermodynamic_factor_matrix",
    )
    structure_response = _finite_size_structure_response_matrix(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        radius_by_carrier=radius_by_carrier,
        kappa_m_inv=kappa_m_inv,
        thermodynamic_factor_matrix=thermodynamic_factor_matrix,
    )
    structure_response_matrix = structure_response[0]
    structure_factor_charge_mode = structure_response[1]
    electrophoretic_sign_values = np.ones(len(carrier_labels), dtype=float)
    resistance_ep_kg_s = _matrix_coupled_psd_component_kg_s(
        zeta_values_kg_s=zeta_ep_values_kg_s,
        overlap_values=overlap_values,
        coupling_sign_values=electrophoretic_sign_values,
        structure_response_matrix=structure_response_matrix,
    )
    resistance_rel_kg_s = _matrix_coupled_psd_component_kg_s(
        zeta_values_kg_s=zeta_rel_values_kg_s,
        overlap_values=overlap_values,
        coupling_sign_values=relaxation_sign_values,
        structure_response_matrix=structure_response_matrix,
    )
    resistance_matrix_kg_s = resistance_ep_kg_s + resistance_rel_kg_s
    _validate_bulk_resistance_matrix(resistance_matrix_kg_s, "resistance_matrix_kg_s")
    maxwell_stefan_pair_friction_coefficient_matrix_J_s_mol_m2 = (
        _derive_maxwell_stefan_pair_friction_coefficient_matrix(
            carrier_labels=carrier_labels,
            concentration_by_carrier=concentration_by_carrier,
            charge_by_carrier=charge_by_carrier,
            resistance_matrix_kg_s=resistance_matrix_kg_s,
        )
    )
    maxwell_stefan_pair_drag_matrix_J_s_mol_m2 = (
        _maxwell_stefan_pair_drag_matrix_J_s_mol_m2(
            carrier_labels=carrier_labels,
            concentration_by_carrier=concentration_by_carrier,
            pair_friction_coefficient_matrix_J_s_mol_m2=(
                maxwell_stefan_pair_friction_coefficient_matrix_J_s_mol_m2
            ),
        )
    )
    return BulkIonAtmosphereState(
        carrier_labels=carrier_labels,
        kappa_inv_m=kappa_inv_m,
        ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
        ambipolar_diffusivity_m2_s=ambipolar_diffusivity_m2_s,
        resistance_matrix_kg_s=resistance_matrix_kg_s,
        resistance_ep_kg_s=resistance_ep_kg_s,
        resistance_rel_kg_s=resistance_rel_kg_s,
        maxwell_stefan_pair_friction_coefficient_matrix_J_s_mol_m2=(
            maxwell_stefan_pair_friction_coefficient_matrix_J_s_mol_m2
        ),
        maxwell_stefan_pair_drag_matrix_J_s_mol_m2=(
            maxwell_stefan_pair_drag_matrix_J_s_mol_m2
        ),
        steric_volume_fraction=steric_volume_fraction,
        thermodynamic_factor_trace=thermodynamic_factor_trace,
        thermodynamic_factor_matrix=thermodynamic_factor_matrix,
        thermodynamic_factor_eigenvalues=thermodynamic_factor_eigenvalues,
        structure_response_matrix=structure_response_matrix,
        structure_factor_charge_mode=structure_factor_charge_mode,
        kappa_radius_by_carrier=kappa_radius_by_carrier,
        solver=bulk_ion_atmosphere_input.solver,
    )


def _ambipolar_diffusivity_m2_s(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    diffusivity_by_carrier: Mapping[str, float],
) -> float:
    charge_weighted_diffusivity_sum = 0.0
    charge_weighted_concentration_sum = 0.0
    for carrier_label in carrier_labels:
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        charge_number = charge_by_carrier[carrier_label]
        diffusivity_m2_s = diffusivity_by_carrier[carrier_label]
        charge_weight = charge_number * charge_number * concentration_mol_m3
        charge_weighted_diffusivity_sum += charge_weight * diffusivity_m2_s
        charge_weighted_concentration_sum += charge_weight
    _assert_nonnegative_finite(
        charge_weighted_concentration_sum,
        "charge_weighted_concentration_sum",
    )
    if charge_weighted_concentration_sum == 0.0:
        return 0.0
    ambipolar_diffusivity_m2_s = (
        charge_weighted_diffusivity_sum / charge_weighted_concentration_sum
    )
    _assert_positive_finite(ambipolar_diffusivity_m2_s, "ambipolar_diffusivity_m2_s")
    return ambipolar_diffusivity_m2_s


def _finite_size_thermodynamic_factor_matrix(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    radius_by_carrier: Mapping[str, float],
    steric_volume_fraction: float,
) -> np.ndarray:
    free_volume_fraction = 1.0 - steric_volume_fraction
    _assert_positive_finite(free_volume_fraction, "free_volume_fraction")
    carrier_count = len(carrier_labels)
    finite_volume_vector = np.zeros(carrier_count, dtype=float)
    for carrier_index, carrier_label in enumerate(carrier_labels):
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        hydrodynamic_radius_m = radius_by_carrier[carrier_label]
        finite_volume_argument = (
            concentration_mol_m3 * N_A * _sphere_volume_m3(hydrodynamic_radius_m)
        )
        _assert_nonnegative_finite(
            finite_volume_argument,
            f"{carrier_label}.finite_volume_argument",
        )
        finite_volume_vector[carrier_index] = math.sqrt(finite_volume_argument)
    thermodynamic_factor_matrix = (
        np.eye(carrier_count, dtype=float)
        + np.outer(finite_volume_vector, finite_volume_vector) / free_volume_fraction
    )
    _validate_bulk_resistance_matrix(
        thermodynamic_factor_matrix,
        "thermodynamic_factor_matrix",
    )
    return thermodynamic_factor_matrix


def _finite_size_structure_response_matrix(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    radius_by_carrier: Mapping[str, float],
    kappa_m_inv: float,
    thermodynamic_factor_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    charge_weighted_concentration = 0.0
    concentration_sum_mol_m3 = 0.0
    radius_weighted_concentration_m = 0.0
    charge_mode_vector = np.zeros(len(carrier_labels), dtype=float)
    for carrier_index, carrier_label in enumerate(carrier_labels):
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        charge_number = charge_by_carrier[carrier_label]
        radius_m = radius_by_carrier[carrier_label]
        charge_weighted_concentration += (
            charge_number * charge_number * concentration_mol_m3
        )
        concentration_sum_mol_m3 += concentration_mol_m3
        radius_weighted_concentration_m += concentration_mol_m3 * radius_m
        charge_mode_vector[carrier_index] = math.sqrt(concentration_mol_m3) * abs(
            charge_number
        )
    if charge_weighted_concentration == 0.0:
        return (thermodynamic_factor_matrix.copy(), 0.0)
    _assert_positive_finite(concentration_sum_mol_m3, "concentration_sum_mol_m3")
    average_hydrodynamic_radius_m = (
        radius_weighted_concentration_m / concentration_sum_mol_m3
    )
    _assert_positive_finite(
        average_hydrodynamic_radius_m, "average_hydrodynamic_radius_m"
    )
    charge_mode_norm = float(np.linalg.norm(charge_mode_vector))
    _assert_positive_finite(charge_mode_norm, "charge_mode_norm")
    normalized_charge_mode = charge_mode_vector / charge_mode_norm
    structure_factor_charge_mode = kappa_m_inv * average_hydrodynamic_radius_m
    _assert_nonnegative_finite(
        structure_factor_charge_mode, "structure_factor_charge_mode"
    )
    structure_response_matrix = (
        thermodynamic_factor_matrix
        + structure_factor_charge_mode
        * np.outer(normalized_charge_mode, normalized_charge_mode)
    )
    _validate_bulk_resistance_matrix(
        structure_response_matrix,
        "structure_response_matrix",
    )
    return (structure_response_matrix, structure_factor_charge_mode)


def _finite_size_bulk_resistance_matrices_kg_s(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    diffusivity_by_carrier: Mapping[str, float],
    radius_by_carrier: Mapping[str, float],
    viscosity_Pa_s: float,
    relative_dielectric: float,
    kappa_m_inv: float,
    steric_volume_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    free_volume_fraction = 1.0 - steric_volume_fraction
    _assert_positive_finite(free_volume_fraction, "free_volume_fraction")
    carrier_count = len(carrier_labels)
    zeta_ep_values = np.zeros(carrier_count, dtype=float)
    zeta_rel_values = np.zeros(carrier_count, dtype=float)
    overlap_values = np.zeros(carrier_count, dtype=float)
    relaxation_sign_values = np.zeros(carrier_count, dtype=float)
    kappa_radius_by_carrier: dict[str, float] = {}
    for carrier_index, carrier_label in enumerate(carrier_labels):
        hydrodynamic_radius_m = radius_by_carrier[carrier_label]
        stern_radius_m = hydrodynamic_radius_m + _opposite_charge_weighted_radius_m(
            carrier_label=carrier_label,
            carrier_labels=carrier_labels,
            concentration_by_carrier=concentration_by_carrier,
            charge_by_carrier=charge_by_carrier,
            radius_by_carrier=radius_by_carrier,
        )
        effective_kappa_m_inv = (
            kappa_m_inv * free_volume_fraction / (1.0 + kappa_m_inv * stern_radius_m)
        )
        kappa_radius_by_carrier[carrier_label] = (
            effective_kappa_m_inv * hydrodynamic_radius_m
        )
        zeta_ep_values[carrier_index] = _electrophoretic_drag_kg_s(
            viscosity_Pa_s=viscosity_Pa_s,
            hydrodynamic_radius_m=hydrodynamic_radius_m,
            kappa_m_inv=effective_kappa_m_inv,
        )
        zeta_rel_values[carrier_index] = _relaxation_drag_kg_s(
            charge_number=charge_by_carrier[carrier_label],
            local_diffusivity_m2_s=diffusivity_by_carrier[carrier_label],
            hydrodynamic_radius_m=hydrodynamic_radius_m,
            relative_dielectric=relative_dielectric,
            kappa_m_inv=effective_kappa_m_inv,
        )
        overlap_values[carrier_index] = math.exp(
            -effective_kappa_m_inv * stern_radius_m
        )
        relaxation_sign_values[carrier_index] = math.copysign(
            1.0, charge_by_carrier[carrier_label]
        )
    return (
        zeta_ep_values,
        zeta_rel_values,
        kappa_radius_by_carrier,
        overlap_values,
        relaxation_sign_values,
    )


def _matrix_coupled_psd_component_kg_s(
    zeta_values_kg_s: np.ndarray,
    overlap_values: np.ndarray,
    coupling_sign_values: np.ndarray,
    structure_response_matrix: np.ndarray,
) -> np.ndarray:
    weighted_zeta_values = zeta_values_kg_s * overlap_values
    sign_matrix = np.diag(coupling_sign_values)
    weighted_matrix = np.diag(np.sqrt(weighted_zeta_values))
    residual_diagonal = zeta_values_kg_s * (1.0 - overlap_values)
    coupled_matrix = (
        sign_matrix
        @ weighted_matrix
        @ structure_response_matrix
        @ weighted_matrix
        @ sign_matrix
    )
    resistance_matrix = coupled_matrix + np.diag(residual_diagonal)
    _validate_bulk_resistance_matrix(resistance_matrix, "matrix_coupled_component_kg_s")
    return resistance_matrix


def _derive_maxwell_stefan_pair_friction_coefficient_matrix(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    resistance_matrix_kg_s: np.ndarray,
) -> np.ndarray:
    _validate_bulk_resistance_matrix(
        resistance_matrix_kg_s,
        "bulk_atmosphere_resistance_matrix_kg_s",
    )
    concentration_values_mol_m3 = np.asarray(
        [concentration_by_carrier[carrier_label] for carrier_label in carrier_labels],
        dtype=float,
    )
    pair_friction_coefficient_matrix_J_s_mol_m2 = np.zeros_like(resistance_matrix_kg_s)
    resistance_entry_scale = float(np.max(np.abs(resistance_matrix_kg_s)))
    pair_drag_tolerance = np.finfo(float).eps * max(1.0, resistance_entry_scale)
    for first_index in range(len(carrier_labels)):
        for second_index in range(first_index + 1, len(carrier_labels)):
            if (
                charge_by_carrier[carrier_labels[first_index]]
                * charge_by_carrier[carrier_labels[second_index]]
                >= 0
            ):
                continue
            raw_off_diagonal_kg_s = float(
                resistance_matrix_kg_s[first_index, second_index]
            )
            bulk_coupling_magnitude_kg_s = abs(raw_off_diagonal_kg_s)
            if bulk_coupling_magnitude_kg_s <= pair_drag_tolerance:
                continue
            concentration_product_sqrt = math.sqrt(
                concentration_values_mol_m3[first_index]
                * concentration_values_mol_m3[second_index]
            )
            pair_friction_coefficient_J_s_mol_m2 = (
                bulk_coupling_magnitude_kg_s / concentration_product_sqrt
            )
            pair_friction_coefficient_matrix_J_s_mol_m2[first_index, second_index] = (
                pair_friction_coefficient_J_s_mol_m2
            )
            pair_friction_coefficient_matrix_J_s_mol_m2[second_index, first_index] = (
                pair_friction_coefficient_J_s_mol_m2
            )
    return pair_friction_coefficient_matrix_J_s_mol_m2


def _maxwell_stefan_pair_drag_matrix_J_s_mol_m2(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    pair_friction_coefficient_matrix_J_s_mol_m2: np.ndarray,
) -> np.ndarray:
    concentration_values_mol_m3 = np.asarray(
        [concentration_by_carrier[carrier_label] for carrier_label in carrier_labels],
        dtype=float,
    )
    pair_drag_matrix_J_s_mol_m2 = np.zeros_like(
        pair_friction_coefficient_matrix_J_s_mol_m2
    )
    for first_index in range(len(carrier_labels)):
        for second_index in range(first_index + 1, len(carrier_labels)):
            pair_friction_coefficient_J_s_mol_m2 = float(
                pair_friction_coefficient_matrix_J_s_mol_m2[first_index, second_index]
            )
            if pair_friction_coefficient_J_s_mol_m2 <= 0.0:
                continue
            pair_drag_J_s_mol_m2 = (
                concentration_values_mol_m3[first_index]
                * concentration_values_mol_m3[second_index]
                * pair_friction_coefficient_J_s_mol_m2
            )
            pair_drag_matrix_J_s_mol_m2[first_index, second_index] = (
                pair_drag_J_s_mol_m2
            )
            pair_drag_matrix_J_s_mol_m2[second_index, first_index] = (
                pair_drag_J_s_mol_m2
            )
    return pair_drag_matrix_J_s_mol_m2


def _matrix_eigenvalue_tuple(
    matrix: np.ndarray,
    context: str,
) -> tuple[float, ...]:
    _validate_bulk_resistance_matrix(matrix, context)
    return tuple(float(value) for value in np.linalg.eigvalsh(matrix))


def _opposite_charge_weighted_radius_m(
    carrier_label: str,
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    radius_by_carrier: Mapping[str, float],
) -> float:
    source_charge = charge_by_carrier[carrier_label]
    weighted_radius_m = 0.0
    concentration_sum_mol_m3 = 0.0
    for other_label in carrier_labels:
        if source_charge * charge_by_carrier[other_label] >= 0:
            continue
        concentration_mol_m3 = concentration_by_carrier[other_label]
        weighted_radius_m += concentration_mol_m3 * radius_by_carrier[other_label]
        concentration_sum_mol_m3 += concentration_mol_m3
    _assert_positive_finite(
        concentration_sum_mol_m3, f"{carrier_label}.opposite_charge_concentration"
    )
    return weighted_radius_m / concentration_sum_mol_m3


def _electrophoretic_drag_kg_s(
    viscosity_Pa_s: float,
    hydrodynamic_radius_m: float,
    kappa_m_inv: float,
) -> float:
    _assert_positive_finite(viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(hydrodynamic_radius_m, "hydrodynamic_radius_m")
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    kappa_radius = kappa_m_inv * hydrodynamic_radius_m
    if kappa_radius == 0.0:
        return 0.0
    return (
        STOKES_SPHERE_DRAG_FACTOR
        * math.pi
        * viscosity_Pa_s
        * hydrodynamic_radius_m
        * kappa_radius
        / (1.0 + kappa_radius)
    )


def _relaxation_drag_kg_s(
    charge_number: int,
    local_diffusivity_m2_s: float,
    hydrodynamic_radius_m: float,
    relative_dielectric: float,
    kappa_m_inv: float,
) -> float:
    _assert_positive_finite(local_diffusivity_m2_s, "local_diffusivity_m2_s")
    _assert_positive_finite(hydrodynamic_radius_m, "hydrodynamic_radius_m")
    _assert_positive_finite(relative_dielectric, "relative_dielectric")
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    kappa_radius = kappa_m_inv * hydrodynamic_radius_m
    if kappa_radius == 0.0:
        return 0.0
    elementary_charge_C = F / N_A
    return (
        charge_number
        * charge_number
        * elementary_charge_C
        * elementary_charge_C
        * kappa_m_inv
        / (
            STOKES_SPHERE_DRAG_FACTOR
            * math.pi
            * EPS_0
            * relative_dielectric
            * local_diffusivity_m2_s
            * (1.0 + kappa_radius)
        )
    )


def _debye_kappa_inv_m(
    charge_weighted_concentration_mol_m3: float,
    relative_dielectric: float,
    temperature_K: float,
) -> float:
    _assert_nonnegative_finite(
        charge_weighted_concentration_mol_m3,
        "charge_weighted_concentration_mol_m3",
    )
    if charge_weighted_concentration_mol_m3 == 0.0:
        return math.inf
    kappa_squared_m_inv2 = (
        F
        * F
        * charge_weighted_concentration_mol_m3
        / (EPS_0 * relative_dielectric * R * temperature_K)
    )
    _assert_positive_finite(kappa_squared_m_inv2, "kappa_squared_m_inv2")
    return 1.0 / math.sqrt(kappa_squared_m_inv2)


def _sphere_volume_m3(radius_m: float) -> float:
    _assert_positive_finite(radius_m, "sphere_radius_m")
    return (
        SPHERE_VOLUME_NUMERATOR
        / SPHERE_VOLUME_DENOMINATOR
        * math.pi
        * radius_m
        * radius_m
        * radius_m
    )


def _validate_solver(solver: str) -> None:
    if solver not in SUPPORTED_ION_ATMOSPHERE_SOLVERS:
        raise ValueError(
            "Unsupported ion-atmosphere solver "
            f"{solver!r}; supported solvers are {SUPPORTED_ION_ATMOSPHERE_SOLVERS}"
        )


def _validate_bulk_solver(solver: str) -> None:
    if solver not in SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS:
        raise ValueError(
            "Unsupported bulk ion-atmosphere solver "
            f"{solver!r}; supported solvers are {SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS}"
        )


def _validate_bulk_resistance_matrix(
    resistance_matrix_kg_s: np.ndarray,
    context: str,
) -> None:
    if resistance_matrix_kg_s.ndim != 2:
        raise ValueError(f"{context} must be a matrix")
    if resistance_matrix_kg_s.shape[0] != resistance_matrix_kg_s.shape[1]:
        raise ValueError(f"{context} must be square")
    if not np.all(np.isfinite(resistance_matrix_kg_s)):
        raise ValueError(f"{context} contains non-finite values")
    if not np.allclose(resistance_matrix_kg_s, resistance_matrix_kg_s.T):
        raise ValueError(f"{context} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(resistance_matrix_kg_s)
    if float(np.min(eigenvalues)) < 0.0:
        raise ValueError(f"{context} must be positive semidefinite")


def _require_charge(
    values: Mapping[str, int],
    key: str,
) -> int:
    if key not in values:
        raise KeyError(f"carrier_charges missing {key}")
    charge_number = int(values[key])
    if charge_number == 0:
        raise ValueError(f"carrier_charges.{key} must be nonzero")
    return charge_number


def _require_positive_finite(
    values: Mapping[str, float],
    key: str,
    context: str,
) -> float:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    value = float(values[key])
    _assert_positive_finite(value, f"{context}.{key}")
    return value


def _require_nonnegative_finite(
    values: Mapping[str, float],
    key: str,
    context: str,
) -> float:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    value = float(values[key])
    _assert_nonnegative_finite(value, f"{context}.{key}")
    return value


def _assert_positive_finite(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{context} must be a positive finite number, got {value}")


def _assert_nonnegative_finite(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{context} must be a non-negative finite number, got {value}")


# ---- generic_speciation.py ----

STANDARD_STATE_CONCENTRATION_MOL_M3 = (
    1000.0  # Unit conversion: 1 mol/L standard state in mol/m^3.
)
COULOMB_DENOMINATOR_FACTOR = 4.0  # Electrostatic denominator: 4*pi*epsilon0*epsilon*r.
BORN_DENOMINATOR_FACTOR = (
    2.0 * COULOMB_DENOMINATOR_FACTOR
)  # Born denominator is twice 4*pi*epsilon0*r.
DESOLVATION_OCCLUSION_SURFACE_FACTOR = (
    4.0  # Spherical surface area factor for contact occlusion fraction.
)
PAIR_COORDINATION_AVERAGE_FACTOR = 0.5  # Mean of two component coordination affinities.
NEWTON_MAX_ITERATIONS = 80  # Numerical solver iteration cap for mass-balance equations.
NEWTON_LINE_SEARCH_BACKOFF = (
    0.5  # Numerical damping factor for positivity-preserving Newton steps.
)
NEWTON_MIN_STEP_FRACTION = 2.0**-40  # Numerical sentinel for failed line search.
MASS_BALANCE_TOLERANCE_FACTOR = math.sqrt(
    np.finfo(float).eps
)  # Floating-point residual scale.
NEUTRAL_LIGAND_MASS_BALANCE_TOLERANCE_FACTOR = (
    MASS_BALANCE_TOLERANCE_FACTOR * math.sqrt(MASS_BALANCE_TOLERANCE_FACTOR)
)
CONTACT_PAIR_CLUSTER_KIND = "contact_pair"
SOLVENT_SEPARATED_PAIR_CLUSTER_KIND = "solvent_separated_pair"
ADDITIVE_SEPARATED_PAIR_CLUSTER_KIND = "additive_separated_solvent_separated_pair"
POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND = "positive_charged_triplet"
NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND = "negative_charged_triplet"
NEUTRAL_CLUSTER_KIND = "neutral_cluster"
HIGHER_CHARGED_CLUSTER_KIND = "higher_charged_cluster"
ACTIVITY_IONIC_STRENGTH_REFERENCE_MOL_M3 = STANDARD_STATE_CONCENTRATION_MOL_M3
NEUTRAL_LIGAND_BINDING_SUBSTRATE_FREE_CATION = "free_cation"
NEUTRAL_LIGAND_BINDING_SUBSTRATE_SOLVENT_SEPARATED_PAIR = "solvent_separated_pair"


@dataclass(frozen=True)
class MolecularSolventEnvironment:
    dielectric_constant: float
    viscosity_cP: float
    hard_sphere_volume_fraction: float
    temperature_K: float
    solvent_effective_radius_A: float
    mean_molecular_volume_A3: float
    solvent_volume_fractions: Mapping[str, float]
    solvent_coordination_affinity_J_mol: float
    additive_ligand_site_occupancy: float
    additive_coordination_affinity_J_mol: float
    additive_solvation_support: float
    additive_molecular_volume_A3: float


@dataclass(frozen=True)
class IonComponent:
    species_name: str
    charge_number: int
    analytical_concentration_M: float
    descriptor: MolecularSpeciesDescriptor


@dataclass(frozen=True)
class ClusterChargedCenter:
    species_name: str
    charge_number: int
    position_A: tuple[float, float, float]


@dataclass(frozen=True)
class ClusterStateTemplate:
    label: str
    cluster_kind: str
    stoichiometry: Mapping[str, int]
    net_charge_number: int
    standard_free_energy_J_mol: float
    coulomb_J_mol: float
    desolvation_J_mol: float
    coordination_J_mol: float
    steric_J_mol: float
    entropy_J_mol: float
    standard_state_correction_J_mol: float
    activity_reference_J_mol: float
    geometry: tuple[ClusterChargedCenter, ...]
    orientation_count: int
    hydrodynamic_radius_A: float
    molecular_volume_A3: float


@dataclass(frozen=True)
class _ClusterFreeEnergyTerms:
    standard_free_energy_J_mol: float
    coulomb_J_mol: float
    desolvation_J_mol: float
    coordination_J_mol: float
    solvation_competition_J_mol: float
    steric_J_mol: float
    entropy_J_mol: float
    standard_state_correction_J_mol: float
    activity_reference_J_mol: float


@dataclass(frozen=True)
class _PairFreeEnergyTerms:
    coulomb_J_mol: float
    desolvation_J_mol: float
    coordination_J_mol: float
    solvation_competition_J_mol: float


@dataclass(frozen=True)
class PMFTerm:
    name: str
    free_energy_J_mol: float
    source: str


@dataclass(frozen=True)
class SolvationCompetitionPMFPartition:
    salt_label: str
    cation_label: str
    anion_label: str
    solvent_composition: Mapping[str, float]
    temperature_K: float
    basin_labels: tuple[str, ...]
    basin_boundaries: Mapping[str, tuple[float, float]]
    pmf_terms: tuple[PMFTerm, ...]
    restricted_partition_weights: Mapping[str, float]
    concentrations_mol_m3: Mapping[str, float]


@dataclass(frozen=True)
class ClusterEnumerationOptions:
    max_cluster_ion_count: int
    primitive_parameters: ConductivityPrimitiveParameterSet


@dataclass(frozen=True)
class GenericSpeciationResult:
    components: tuple[IonComponent, ...]
    cluster_templates: tuple[ClusterStateTemplate, ...]
    free_component_concentrations_mol_m3: Mapping[str, float]
    cluster_concentrations_mol_m3: Mapping[str, float]
    neutral_ligand_site_concentrations_mol_m3: Mapping[str, float]
    cation_ligand_concentrations_mol_m3: Mapping[str, float]
    cation_ligand_component_species_by_label: Mapping[str, str]
    cation_ligand_anion_concentrations_mol_m3: Mapping[str, float]
    cation_ligand_anion_parent_cluster_by_label: Mapping[str, str]
    free_component_ligand_bound_concentrations_mol_m3: Mapping[str, float]
    cluster_ligand_bound_concentrations_mol_m3: Mapping[str, float]
    solvation_competition_pmf_partitions: tuple[SolvationCompetitionPMFPartition, ...]
    mass_balance_residual_mol_m3: float


@dataclass(frozen=True)
class _NeutralLigandFeature:
    feature_label: str
    site_concentration_mol_m3: float
    coordination_affinity_J_mol: float
    molecular_volume_A3: float
    solvation_support: float
    coordination_site_count: float


@dataclass(frozen=True)
class _NeutralLigandBindingSubstrate:
    substrate_label: str
    substrate_kind: str
    source_species_name: str
    source_cluster_label: str
    total_concentration_mol_m3: float
    cation_feature_label: str


@dataclass(frozen=True)
class _LigandCoupledMassActionState:
    free_component_concentrations_mol_m3: np.ndarray
    free_ligand_concentrations_mol_m3: np.ndarray
    cluster_concentrations_mol_m3: np.ndarray
    cation_ligand_concentrations_mol_m3: np.ndarray
    cation_ligand_anion_concentrations_mol_m3: np.ndarray
    residual_vector_mol_m3: np.ndarray


@dataclass(frozen=True)
class _LigandCoupledSpeciationProblem:
    components: tuple[IonComponent, ...]
    cluster_templates: tuple[ClusterStateTemplate, ...]
    total_component_concentrations_mol_m3: np.ndarray
    total_ligand_site_concentrations_mol_m3: np.ndarray
    ligand_association_constants_m3_mol: np.ndarray
    solvent_environment: MolecularSolventEnvironment
    primitive_parameters: ConductivityPrimitiveParameterSet


def build_cluster_state_templates(
    components: tuple[IonComponent, ...],
    solvent_environment: MolecularSolventEnvironment,
    options: ClusterEnumerationOptions,
) -> tuple[ClusterStateTemplate, ...]:
    _validate_solvent_environment(solvent_environment)
    validate_conductivity_primitive_parameters(options.primitive_parameters)
    if options.max_cluster_ion_count < 1:
        raise ValueError("max_cluster_ion_count must be positive")
    if options.max_cluster_ion_count < 2:
        return tuple()
    templates: list[ClusterStateTemplate] = []
    for stoichiometric_counts in _cluster_stoichiometric_counts(
        len(components),
        options.max_cluster_ion_count,
    ):
        if not _contains_opposite_charges(components, stoichiometric_counts):
            continue
        for cluster_kind in _cluster_kinds_for_stoichiometry(
            components,
            stoichiometric_counts,
        ):
            templates.append(
                _stoichiometric_cluster_template(
                    components,
                    stoichiometric_counts,
                    cluster_kind,
                    solvent_environment,
                    options.primitive_parameters,
                )
            )
    return tuple(templates)


def solve_generic_mass_balance(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> GenericSpeciationResult:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_solvent_environment(solvent_environment)
    if not components:
        return GenericSpeciationResult(
            components=tuple(),
            cluster_templates=tuple(cluster_templates),
            free_component_concentrations_mol_m3={},
            cluster_concentrations_mol_m3={},
            neutral_ligand_site_concentrations_mol_m3={},
            cation_ligand_concentrations_mol_m3={},
            cation_ligand_component_species_by_label={},
            cation_ligand_anion_concentrations_mol_m3={},
            cation_ligand_anion_parent_cluster_by_label={},
            free_component_ligand_bound_concentrations_mol_m3={},
            cluster_ligand_bound_concentrations_mol_m3={},
            solvation_competition_pmf_partitions=tuple(),
            mass_balance_residual_mol_m3=0.0,
        )
    component_names = tuple(component.species_name for component in components)
    if len(set(component_names)) != len(component_names):
        raise ValueError("component species names must be unique")
    total_concentrations = np.asarray(
        [
            _positive_float(
                component.analytical_concentration_M,
                f"{component.species_name}.analytical_concentration_M",
            )
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            for component in components
        ],
        dtype=float,
    )
    if not cluster_templates:
        return GenericSpeciationResult(
            components=tuple(components),
            cluster_templates=tuple(),
            free_component_concentrations_mol_m3={
                component.species_name: float(total_concentrations[index])
                for index, component in enumerate(components)
            },
            cluster_concentrations_mol_m3={},
            neutral_ligand_site_concentrations_mol_m3={},
            cation_ligand_concentrations_mol_m3={},
            cation_ligand_component_species_by_label={},
            cation_ligand_anion_concentrations_mol_m3={},
            cation_ligand_anion_parent_cluster_by_label={},
            free_component_ligand_bound_concentrations_mol_m3={},
            cluster_ligand_bound_concentrations_mol_m3={},
            solvation_competition_pmf_partitions=tuple(),
            mass_balance_residual_mol_m3=0.0,
        )
    free_concentrations = _solve_free_concentrations(
        components,
        cluster_templates,
        total_concentrations,
        solvent_environment,
        primitive_parameters,
    )
    cluster_concentration_array = _cluster_concentrations_array(
        components,
        cluster_templates,
        free_concentrations,
        solvent_environment,
        primitive_parameters,
    )
    cluster_concentrations = {
        template.label: float(cluster_concentration_array[index])
        for index, template in enumerate(cluster_templates)
    }
    residual = _mass_balance_residual(
        components,
        cluster_templates,
        free_concentrations,
        cluster_concentrations,
        total_concentrations,
    )
    return GenericSpeciationResult(
        components=tuple(components),
        cluster_templates=tuple(cluster_templates),
        free_component_concentrations_mol_m3={
            component.species_name: float(free_concentrations[index])
            for index, component in enumerate(components)
        },
        cluster_concentrations_mol_m3=cluster_concentrations,
        neutral_ligand_site_concentrations_mol_m3={},
        cation_ligand_concentrations_mol_m3={},
        cation_ligand_component_species_by_label={},
        cation_ligand_anion_concentrations_mol_m3={},
        cation_ligand_anion_parent_cluster_by_label={},
        free_component_ligand_bound_concentrations_mol_m3={},
        cluster_ligand_bound_concentrations_mol_m3={},
        solvation_competition_pmf_partitions=solvation_competition_pmf_partition(
            components,
            cluster_templates,
            {
                component.species_name: float(free_concentrations[index])
                for index, component in enumerate(components)
            },
            cluster_concentrations,
            solvent_environment,
        ),
        mass_balance_residual_mol_m3=float(np.max(np.abs(residual))),
    )


def cluster_activity_correction_J_mol(
    components: tuple[IonComponent, ...],
    cluster_template: ClusterStateTemplate,
    free_component_concentrations_mol_m3: Mapping[str, float],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_solvent_environment(solvent_environment)
    component_index_by_name = {
        component.species_name: index for index, component in enumerate(components)
    }
    free_concentration_array = np.asarray(
        [
            _positive_float(
                free_component_concentrations_mol_m3[component.species_name],
                f"{component.species_name}.free_concentration_mol_m3",
            )
            for component in components
        ],
        dtype=float,
    )
    activity_log_factor = _cluster_activity_log_factor(
        components,
        cluster_template,
        free_concentration_array,
        component_index_by_name,
        solvent_environment,
        primitive_parameters,
    )
    return float(-R * solvent_environment.temperature_K * activity_log_factor)


def _cluster_stoichiometric_counts(
    component_count: int,
    max_cluster_ion_count: int,
) -> tuple[tuple[int, ...], ...]:
    if component_count <= 0:
        return tuple()
    counts: list[tuple[int, ...]] = []
    current_counts = [0 for _component_index in range(component_count)]
    _append_cluster_stoichiometric_counts(
        component_count,
        max_cluster_ion_count,
        0,
        current_counts,
        counts,
    )
    return tuple(counts)


def _append_cluster_stoichiometric_counts(
    component_count: int,
    max_cluster_ion_count: int,
    component_index: int,
    current_counts: list[int],
    counts: list[tuple[int, ...]],
) -> None:
    if component_index == component_count:
        total_ion_count = sum(current_counts)
        if total_ion_count >= 2 and total_ion_count <= max_cluster_ion_count:
            counts.append(tuple(current_counts))
        return
    current_total_count = sum(current_counts)
    maximum_count_for_component = max_cluster_ion_count - current_total_count
    for component_count_value in range(maximum_count_for_component + 1):
        current_counts[component_index] = component_count_value
        _append_cluster_stoichiometric_counts(
            component_count,
            max_cluster_ion_count,
            component_index + 1,
            current_counts,
            counts,
        )
    current_counts[component_index] = 0


def _contains_opposite_charges(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> bool:
    has_positive_charge = False
    has_negative_charge = False
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        if stoichiometric_count == 0:
            continue
        if component.charge_number > 0:
            has_positive_charge = True
        if component.charge_number < 0:
            has_negative_charge = True
    return has_positive_charge and has_negative_charge


def _cluster_kinds_for_stoichiometry(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> tuple[str, ...]:
    total_ion_count = sum(stoichiometric_counts)
    if total_ion_count == 2:
        return (
            CONTACT_PAIR_CLUSTER_KIND,
            SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
        )
    net_charge_number = _cluster_net_charge_number(
        components,
        stoichiometric_counts,
    )
    if net_charge_number == 0:
        return (NEUTRAL_CLUSTER_KIND,)
    if total_ion_count == 3 and net_charge_number > 0:
        return (POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,)
    if total_ion_count == 3 and net_charge_number < 0:
        return (NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,)
    return (HIGHER_CHARGED_CLUSTER_KIND,)


def _stoichiometric_cluster_template(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> ClusterStateTemplate:
    if len(components) != len(stoichiometric_counts):
        raise ValueError("stoichiometric count length must match components")
    stoichiometry = {
        component.species_name: int(stoichiometric_count)
        for component, stoichiometric_count in zip(components, stoichiometric_counts)
        if stoichiometric_count > 0
    }
    if len(stoichiometry) < 2:
        raise ValueError("cluster template requires at least two species")
    geometry = _cluster_geometry(
        components,
        stoichiometric_counts,
        cluster_kind,
        solvent_environment,
    )
    free_energy_terms = _cluster_standard_free_energy_terms(
        components,
        stoichiometric_counts,
        cluster_kind,
        geometry,
        solvent_environment,
        primitive_parameters,
    )
    return ClusterStateTemplate(
        label=_cluster_label(components, stoichiometric_counts, cluster_kind),
        cluster_kind=cluster_kind,
        stoichiometry=stoichiometry,
        net_charge_number=_cluster_net_charge_number(
            components,
            stoichiometric_counts,
        ),
        standard_free_energy_J_mol=free_energy_terms.standard_free_energy_J_mol,
        coulomb_J_mol=free_energy_terms.coulomb_J_mol,
        desolvation_J_mol=free_energy_terms.desolvation_J_mol,
        coordination_J_mol=free_energy_terms.coordination_J_mol,
        steric_J_mol=free_energy_terms.steric_J_mol,
        entropy_J_mol=free_energy_terms.entropy_J_mol,
        standard_state_correction_J_mol=(
            free_energy_terms.standard_state_correction_J_mol
        ),
        activity_reference_J_mol=free_energy_terms.activity_reference_J_mol,
        geometry=geometry,
        orientation_count=1,
        hydrodynamic_radius_A=_cluster_hydrodynamic_radius_A(
            components,
            stoichiometric_counts,
            primitive_parameters,
        ),
        molecular_volume_A3=_cluster_molecular_volume_A3(
            components,
            stoichiometric_counts,
        ),
    )


def _cluster_geometry(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
    solvent_environment: MolecularSolventEnvironment,
) -> tuple[ClusterChargedCenter, ...]:
    expanded_components: list[IonComponent] = []
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        for _species_copy_index in range(stoichiometric_count):
            expanded_components.append(component)
    if len(expanded_components) < 2:
        raise ValueError("cluster geometry requires at least two charged centers")
    center_positions_A: list[float] = [0.0]
    for previous_component, next_component in zip(
        expanded_components[:-1],
        expanded_components[1:],
    ):
        separation_A = (
            previous_component.descriptor.cavity_radius_A
            + next_component.descriptor.cavity_radius_A
            + _cluster_separation_extra_A(
                previous_component,
                next_component,
                cluster_kind,
                solvent_environment,
            )
        )
        center_positions_A.append(center_positions_A[-1] + separation_A)
    center_mean_A = math.fsum(center_positions_A) / len(center_positions_A)
    geometry: list[ClusterChargedCenter] = []
    for component, center_position_A in zip(expanded_components, center_positions_A):
        geometry.append(
            ClusterChargedCenter(
                species_name=component.species_name,
                charge_number=component.charge_number,
                position_A=(center_position_A - center_mean_A, 0.0, 0.0),
            )
        )
    return tuple(geometry)


def _cluster_separation_extra_A(
    previous_component: IonComponent,
    next_component: IonComponent,
    cluster_kind: str,
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    if cluster_kind != SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
        return 0.0
    if previous_component.charge_number * next_component.charge_number >= 0:
        return 0.0
    return 2.0 * _positive_float(
        solvent_environment.solvent_effective_radius_A,
        "solvent_effective_radius_A",
    )


def _cluster_standard_free_energy_terms(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
    geometry: tuple[ClusterChargedCenter, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> _ClusterFreeEnergyTerms:
    component_by_name = {component.species_name: component for component in components}
    coulomb_J_mol = 0.0
    desolvation_J_mol = 0.0
    coordination_J_mol = 0.0
    solvation_competition_J_mol = 0.0
    for first_index, first_center in enumerate(geometry):
        for second_center in geometry[first_index + 1 :]:
            first_component = component_by_name[first_center.species_name]
            second_component = component_by_name[second_center.species_name]
            center_distance_A = _center_distance_A(
                first_center.position_A,
                second_center.position_A,
            )
            pair_terms = _pair_interaction_free_energy_terms(
                first_component,
                second_component,
                center_distance_A,
                solvent_environment,
                primitive_parameters,
            )
            coulomb_J_mol += pair_terms.coulomb_J_mol
            desolvation_J_mol += pair_terms.desolvation_J_mol
            coordination_J_mol += pair_terms.coordination_J_mol
            solvation_competition_J_mol += pair_terms.solvation_competition_J_mol
    desolvation_J_mol += (
        R
        * solvent_environment.temperature_K
        * _cluster_kind_desolvation_offset_over_RT(
            cluster_kind,
            primitive_parameters,
        )
    )
    total_ion_count = sum(stoichiometric_counts)
    steric_J_mol = (
        primitive_parameters.steric_free_energy_scale
        * R
        * solvent_environment.temperature_K
        * solvent_environment.hard_sphere_volume_fraction
        * _cluster_molecular_volume_A3(components, stoichiometric_counts)
        * ANGSTROM3_TO_M3
        * N_A
        * STANDARD_STATE_CONCENTRATION_MOL_M3
        * total_ion_count
    )
    entropy_J_mol = (
        primitive_parameters.cluster_entropy_penalty_scale
        * R
        * solvent_environment.temperature_K
        * (total_ion_count - 1)
    )
    standard_state_correction_J_mol = _cluster_crowding_stabilization_J_mol(
        components,
        stoichiometric_counts,
        solvent_environment,
        primitive_parameters,
    ) + _cluster_topology_standard_state_correction_J_mol(
        components,
        stoichiometric_counts,
        cluster_kind,
        solvent_environment,
        primitive_parameters,
    )
    activity_reference_J_mol = _cluster_activity_reference_J_mol(
        components,
        stoichiometric_counts,
        solvent_environment,
        primitive_parameters,
    )
    standard_free_energy_J_mol = (
        coulomb_J_mol
        + desolvation_J_mol
        + coordination_J_mol
        + solvation_competition_J_mol
        + steric_J_mol
        + entropy_J_mol
        + standard_state_correction_J_mol
    )
    return _ClusterFreeEnergyTerms(
        standard_free_energy_J_mol=float(standard_free_energy_J_mol),
        coulomb_J_mol=float(coulomb_J_mol),
        desolvation_J_mol=float(desolvation_J_mol),
        coordination_J_mol=float(coordination_J_mol),
        solvation_competition_J_mol=float(solvation_competition_J_mol),
        steric_J_mol=float(steric_J_mol),
        entropy_J_mol=float(entropy_J_mol),
        standard_state_correction_J_mol=float(standard_state_correction_J_mol),
        activity_reference_J_mol=float(activity_reference_J_mol),
    )


def _cluster_topology_standard_state_correction_J_mol(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    total_ion_count = sum(stoichiometric_counts)
    net_charge_number = abs(
        _cluster_net_charge_number(components, stoichiometric_counts)
    )
    log_equilibrium_offset = (
        _cluster_kind_logK_offset(cluster_kind, primitive_parameters)
        + (total_ion_count - 2) * primitive_parameters.cluster_order_logK_slope
        + net_charge_number * primitive_parameters.cluster_charge_magnitude_logK_slope
    )
    return float(-R * solvent_environment.temperature_K * log_equilibrium_offset)


def _cluster_activity_reference_J_mol(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    reference_ionic_strength_ratio = (
        ACTIVITY_IONIC_STRENGTH_REFERENCE_MOL_M3 / STANDARD_STATE_CONCENTRATION_MOL_M3
    )
    component_activity_log_sum = 0.0
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        if stoichiometric_count <= 0:
            continue
        component_activity_log_sum += stoichiometric_count * _species_activity_ln_gamma(
            charge_number=component.charge_number,
            activity_size_radius_A=component.descriptor.cavity_radius_A,
            molecular_volume_A3=component.descriptor.molecular_volume_A3,
            ionic_strength_ratio=reference_ionic_strength_ratio,
            solvent_environment=solvent_environment,
            primitive_parameters=primitive_parameters,
        )
    cluster_activity_log_gamma = _species_activity_ln_gamma(
        charge_number=_cluster_net_charge_number(components, stoichiometric_counts),
        activity_size_radius_A=_cluster_hydrodynamic_radius_A(
            components,
            stoichiometric_counts,
            primitive_parameters,
        ),
        molecular_volume_A3=_cluster_molecular_volume_A3(
            components,
            stoichiometric_counts,
        ),
        ionic_strength_ratio=reference_ionic_strength_ratio,
        solvent_environment=solvent_environment,
        primitive_parameters=primitive_parameters,
    )
    activity_log_factor = (
        component_activity_log_sum
        - primitive_parameters.cluster_activity_scale * cluster_activity_log_gamma
    )
    return float(-R * solvent_environment.temperature_K * activity_log_factor)


def _cluster_kind_logK_offset(
    cluster_kind: str,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    log_offset_by_cluster_kind = {
        CONTACT_PAIR_CLUSTER_KIND: (
            primitive_parameters.pair_logK_offset
            + primitive_parameters.contact_pair_logK_offset
        ),
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND: (
            primitive_parameters.pair_logK_offset
            + primitive_parameters.solvent_separated_pair_logK_offset
        ),
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND: (
            primitive_parameters.positive_charged_triplet_logK_offset
        ),
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND: (
            primitive_parameters.negative_charged_triplet_logK_offset
        ),
        NEUTRAL_CLUSTER_KIND: primitive_parameters.neutral_cluster_logK_offset,
        HIGHER_CHARGED_CLUSTER_KIND: (
            primitive_parameters.higher_charged_cluster_logK_offset
        ),
    }
    if cluster_kind not in log_offset_by_cluster_kind:
        raise ValueError(f"unknown cluster kind {cluster_kind}")
    return log_offset_by_cluster_kind[cluster_kind]


def _cluster_kind_desolvation_offset_over_RT(
    cluster_kind: str,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    desolvation_offset_by_cluster_kind = {
        CONTACT_PAIR_CLUSTER_KIND: (
            primitive_parameters.contact_pair_desolvation_offset_over_RT
        ),
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND: (
            primitive_parameters.solvent_separated_pair_desolvation_offset_over_RT
        ),
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND: 0.0,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND: 0.0,
        NEUTRAL_CLUSTER_KIND: 0.0,
        HIGHER_CHARGED_CLUSTER_KIND: (
            primitive_parameters.higher_charged_cluster_desolvation_offset_over_RT
        ),
    }
    if cluster_kind not in desolvation_offset_by_cluster_kind:
        raise ValueError(f"unknown cluster kind {cluster_kind}")
    return desolvation_offset_by_cluster_kind[cluster_kind]


def _pair_interaction_free_energy_terms(
    first_component: IonComponent,
    second_component: IonComponent,
    center_distance_A: float,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> _PairFreeEnergyTerms:
    first_descriptor = first_component.descriptor
    second_descriptor = second_component.descriptor
    contact_distance_m = (
        _positive_float(
            center_distance_A,
            "center_distance_A",
        )
        * ANGSTROM_TO_M
    )
    charge_cloud_distance_m = math.sqrt(
        contact_distance_m**2
        + (first_descriptor.charge_cloud_radius_A * ANGSTROM_TO_M) ** 2
        + (second_descriptor.charge_cloud_radius_A * ANGSTROM_TO_M) ** 2
    )
    coulomb_energy_J_mol = (
        N_A
        * first_component.charge_number
        * second_component.charge_number
        * E_CHARGE**2
        / (
            COULOMB_DENOMINATOR_FACTOR
            * math.pi
            * EPS_0
            * solvent_environment.dielectric_constant
            * charge_cloud_distance_m
        )
    )
    coordination_energy_J_mol = 0.0
    if first_component.charge_number * second_component.charge_number < 0:
        coordination_energy_J_mol = -PAIR_COORDINATION_AVERAGE_FACTOR * (
            first_descriptor.coordination_affinity_J_mol
            + second_descriptor.coordination_affinity_J_mol
        )
    desolvation_energy_J_mol = _pair_desolvation_penalty_J_mol(
        first_component,
        second_component,
        center_distance_A,
        solvent_environment,
    )
    solvation_competition_energy_J_mol = _pair_solvation_competition_penalty_J_mol(
        first_component,
        second_component,
        solvent_environment,
        primitive_parameters,
    )
    return _PairFreeEnergyTerms(
        coulomb_J_mol=float(primitive_parameters.coulomb_scale * coulomb_energy_J_mol),
        desolvation_J_mol=float(
            primitive_parameters.desolvation_scale * desolvation_energy_J_mol
        ),
        coordination_J_mol=float(
            primitive_parameters.coordination_scale * coordination_energy_J_mol
        ),
        solvation_competition_J_mol=float(solvation_competition_energy_J_mol),
    )


def solvation_competition_pmf_partition(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_component_concentrations_mol_m3: Mapping[str, float],
    cluster_concentrations_mol_m3: Mapping[str, float],
    solvent_environment: MolecularSolventEnvironment,
) -> tuple[SolvationCompetitionPMFPartition, ...]:
    _validate_solvent_environment(solvent_environment)
    component_by_name = {component.species_name: component for component in components}
    partitions: list[SolvationCompetitionPMFPartition] = []
    for cluster_template in cluster_templates:
        if len(cluster_template.stoichiometry) != 2:
            continue
        charged_species_names = tuple(cluster_template.stoichiometry)
        first_component = component_by_name[charged_species_names[0]]
        second_component = component_by_name[charged_species_names[1]]
        if first_component.charge_number * second_component.charge_number >= 0:
            continue
        cation_component = (
            first_component if first_component.charge_number > 0 else second_component
        )
        anion_component = (
            first_component if first_component.charge_number < 0 else second_component
        )
        cation_concentration = free_component_concentrations_mol_m3[
            cation_component.species_name
        ]
        anion_concentration = free_component_concentrations_mol_m3[
            anion_component.species_name
        ]
        cluster_concentration = cluster_concentrations_mol_m3[cluster_template.label]
        total_partition_concentration = _positive_float(
            cation_concentration + anion_concentration + cluster_concentration,
            f"{cluster_template.label}.partition_total_concentration_mol_m3",
        )
        solvation_competition_energy_J_mol = (
            cluster_template.standard_free_energy_J_mol
            - cluster_template.coulomb_J_mol
            - cluster_template.desolvation_J_mol
            - cluster_template.coordination_J_mol
            - cluster_template.steric_J_mol
            - cluster_template.entropy_J_mol
            - cluster_template.standard_state_correction_J_mol
        )
        partitions.append(
            SolvationCompetitionPMFPartition(
                salt_label=f"{cation_component.species_name}:{anion_component.species_name}",
                cation_label=cation_component.species_name,
                anion_label=anion_component.species_name,
                solvent_composition=dict(solvent_environment.solvent_volume_fractions),
                temperature_K=solvent_environment.temperature_K,
                basin_labels=(
                    "free_ion_center",
                    SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
                    CONTACT_PAIR_CLUSTER_KIND,
                    NEUTRAL_CLUSTER_KIND,
                    HIGHER_CHARGED_CLUSTER_KIND,
                ),
                basin_boundaries=_solvation_competition_basin_boundaries_A(
                    cation_component,
                    anion_component,
                    solvent_environment,
                ),
                pmf_terms=(
                    PMFTerm("coulomb", cluster_template.coulomb_J_mol, "pair_pmf"),
                    PMFTerm(
                        "desolvation",
                        cluster_template.desolvation_J_mol,
                        "pair_pmf",
                    ),
                    PMFTerm(
                        "coordination",
                        cluster_template.coordination_J_mol,
                        "pair_pmf",
                    ),
                    PMFTerm(
                        "solvation_competition",
                        solvation_competition_energy_J_mol,
                        "solvent_shell_competition_pmf",
                    ),
                ),
                restricted_partition_weights={
                    "free_ion_center": float(
                        (cation_concentration + anion_concentration)
                        / total_partition_concentration
                    ),
                    cluster_template.cluster_kind: float(
                        cluster_concentration / total_partition_concentration
                    ),
                },
                concentrations_mol_m3={
                    cation_component.species_name: float(cation_concentration),
                    anion_component.species_name: float(anion_concentration),
                    cluster_template.label: float(cluster_concentration),
                },
            )
        )
    return tuple(partitions)


def _pair_solvation_competition_penalty_J_mol(
    first_component: IonComponent,
    second_component: IonComponent,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    if first_component.charge_number * second_component.charge_number >= 0:
        return 0.0
    solvent_competition_affinity_J_mol = _solvent_shell_competition_affinity_J_mol(
        solvent_environment
    )
    ion_pair_coordination_affinity_J_mol = PAIR_COORDINATION_AVERAGE_FACTOR * (
        first_component.descriptor.coordination_affinity_J_mol
        + second_component.descriptor.coordination_affinity_J_mol
    )
    competition_penalty_J_mol = (
        solvent_competition_affinity_J_mol - ion_pair_coordination_affinity_J_mol
    )
    if competition_penalty_J_mol <= 0.0:
        return 0.0
    ionic_strength_ratio = _analytical_ionic_strength_ratio(
        (first_component, second_component)
    )
    crowding_denominator = 1.0 + (
        primitive_parameters.association_crowding_stabilization_scale
        * ionic_strength_ratio
        ** primitive_parameters.association_crowding_ionic_strength_exponent
    )
    return float(
        competition_penalty_J_mol
        / _positive_float(crowding_denominator, "solvation_competition_crowding")
    )


def _solvent_shell_competition_affinity_J_mol(
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    return _nonnegative_float(
        solvent_environment.solvent_coordination_affinity_J_mol,
        "solvent_coordination_affinity_J_mol",
    )


def _solvation_competition_basin_boundaries_A(
    cation_component: IonComponent,
    anion_component: IonComponent,
    solvent_environment: MolecularSolventEnvironment,
) -> Mapping[str, tuple[float, float]]:
    contact_upper_A = (
        cation_component.descriptor.cavity_radius_A
        + anion_component.descriptor.cavity_radius_A
    )
    ssip_upper_A = contact_upper_A + (
        2.0 * solvent_environment.solvent_effective_radius_A
    )
    return {
        CONTACT_PAIR_CLUSTER_KIND: (0.0, float(contact_upper_A)),
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND: (
            float(contact_upper_A),
            float(ssip_upper_A),
        ),
        "free_ion_center": (float(ssip_upper_A), math.inf),
    }


def _cluster_label(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
) -> str:
    label_parts = []
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        if stoichiometric_count == 0:
            continue
        label_parts.append(f"{component.species_name}^{stoichiometric_count}")
    return "cluster:" + cluster_kind + ":" + ":".join(label_parts)


def _cluster_net_charge_number(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> int:
    net_charge_number = 0
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        net_charge_number += component.charge_number * stoichiometric_count
    return int(net_charge_number)


def _cluster_molecular_volume_A3(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> float:
    cluster_volume_A3 = 0.0
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        cluster_volume_A3 += (
            stoichiometric_count * component.descriptor.molecular_volume_A3
        )
    return _positive_float(cluster_volume_A3, "cluster_volume_A3")


def _cluster_hydrodynamic_radius_A(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    radius_cubed_sum = 0.0
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        hydrodynamic_radius_A = component.descriptor.hydrodynamic_radius_A
        radius_cubed_sum += stoichiometric_count * hydrodynamic_radius_A**3
    return primitive_parameters.cluster_hydrodynamic_radius_scale * _positive_float(
        radius_cubed_sum, "cluster_radius_cubed_sum"
    ) ** (1.0 / 3.0)


def _cluster_crowding_stabilization_J_mol(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    total_ion_count = sum(stoichiometric_counts)
    if total_ion_count <= 1:
        return 0.0
    ionic_strength_ratio = _analytical_ionic_strength_ratio(components)
    charge_cloud_compactness_ratio = _cluster_charge_cloud_compactness_ratio(
        components,
        stoichiometric_counts,
    )
    stabilization_magnitude_J_mol = (
        primitive_parameters.association_crowding_stabilization_scale
        * R
        * solvent_environment.temperature_K
        * (total_ion_count - 1)
        * (
            ionic_strength_ratio
            ** primitive_parameters.association_crowding_ionic_strength_exponent
        )
        * (
            charge_cloud_compactness_ratio
            ** primitive_parameters.association_crowding_charge_density_exponent
        )
    )
    return -_positive_float(
        stabilization_magnitude_J_mol,
        "association_crowding_stabilization_magnitude_J_mol",
    )


def _analytical_ionic_strength_ratio(
    components: tuple[IonComponent, ...],
) -> float:
    ionic_strength_mol_m3 = 0.0
    for component in components:
        analytical_concentration_mol_m3 = (
            _positive_float(
                component.analytical_concentration_M,
                f"{component.species_name}.analytical_concentration_M",
            )
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        ionic_strength_mol_m3 += (
            analytical_concentration_mol_m3
            * component.charge_number
            * component.charge_number
        )
    return _positive_float(
        ionic_strength_mol_m3 / STANDARD_STATE_CONCENTRATION_MOL_M3,
        "analytical_ionic_strength_ratio",
    )


def _cluster_charge_cloud_compactness_ratio(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> float:
    cluster_charge_cloud_compactness = _charge_cloud_compactness_for_counts(
        components,
        stoichiometric_counts,
        "cluster_charge_cloud_compactness",
    )
    analytical_weight_counts = tuple(
        _positive_float(
            component.analytical_concentration_M,
            f"{component.species_name}.analytical_concentration_M",
        )
        for component in components
    )
    mixture_charge_cloud_compactness = _charge_cloud_compactness_for_counts(
        components,
        analytical_weight_counts,
        "mixture_charge_cloud_compactness",
    )
    return _positive_float(
        cluster_charge_cloud_compactness / mixture_charge_cloud_compactness,
        "cluster_charge_cloud_compactness_ratio",
    )


def _charge_cloud_compactness_for_counts(
    components: tuple[IonComponent, ...],
    component_weights: tuple[float, ...],
    context: str,
) -> float:
    if len(components) != len(component_weights):
        raise ValueError(f"{context} weights must match components")
    weighted_charge_number_sum = 0.0
    weighted_charge_cloud_volume_A3 = 0.0
    for component, component_weight in zip(components, component_weights):
        nonnegative_component_weight = _nonnegative_float(
            component_weight,
            f"{context}.component_weight",
        )
        if nonnegative_component_weight == 0.0:
            continue
        charge_cloud_radius_A = _positive_float(
            component.descriptor.charge_cloud_radius_A,
            f"{context}.{component.species_name}.charge_cloud_radius_A",
        )
        weighted_charge_number_sum += nonnegative_component_weight * abs(
            component.charge_number
        )
        weighted_charge_cloud_volume_A3 += (
            nonnegative_component_weight
            * charge_cloud_radius_A
            * charge_cloud_radius_A
            * charge_cloud_radius_A
        )
    return _positive_float(
        weighted_charge_number_sum / weighted_charge_cloud_volume_A3,
        context,
    )


def _center_distance_A(
    first_position_A: tuple[float, float, float],
    second_position_A: tuple[float, float, float],
) -> float:
    squared_distance = 0.0
    for first_coordinate, second_coordinate in zip(first_position_A, second_position_A):
        difference = first_coordinate - second_coordinate
        squared_distance += difference**2
    return _positive_float(math.sqrt(squared_distance), "center_distance_A")


def _pair_desolvation_penalty_J_mol(
    first_component: IonComponent,
    second_component: IonComponent,
    center_distance_A: float,
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    first_descriptor = first_component.descriptor
    second_descriptor = second_component.descriptor
    first_occlusion_fraction = _contact_occlusion_fraction(
        occluding_radius_A=second_descriptor.cavity_radius_A,
        center_separation_A=center_distance_A,
    )
    second_occlusion_fraction = _contact_occlusion_fraction(
        occluding_radius_A=first_descriptor.cavity_radius_A,
        center_separation_A=center_distance_A,
    )
    first_born_magnitude_J_mol = abs(
        _born_solvation_energy_J_mol(
            charge_number=first_component.charge_number,
            born_solvation_radius_A=first_descriptor.born_solvation_radius_A,
            solvent_environment=solvent_environment,
        )
    )
    second_born_magnitude_J_mol = abs(
        _born_solvation_energy_J_mol(
            charge_number=second_component.charge_number,
            born_solvation_radius_A=second_descriptor.born_solvation_radius_A,
            solvent_environment=solvent_environment,
        )
    )
    return float(
        first_occlusion_fraction * first_born_magnitude_J_mol
        + second_occlusion_fraction * second_born_magnitude_J_mol
    )


def _contact_occlusion_fraction(
    occluding_radius_A: float,
    center_separation_A: float,
) -> float:
    radius_A = _positive_float(occluding_radius_A, "occluding_radius_A")
    separation_A = _positive_float(center_separation_A, "center_separation_A")
    occlusion_fraction = radius_A**2 / (
        DESOLVATION_OCCLUSION_SURFACE_FACTOR * separation_A**2
    )
    if occlusion_fraction >= 1.0:
        raise ValueError(
            "contact occlusion fraction must remain below one for pair geometry"
        )
    return float(occlusion_fraction)


def _born_solvation_energy_J_mol(
    charge_number: int,
    born_solvation_radius_A: float,
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    radius_m = (
        _positive_float(
            born_solvation_radius_A,
            "born_solvation_radius_A",
        )
        * ANGSTROM_TO_M
    )
    dielectric = _positive_float(
        solvent_environment.dielectric_constant,
        "dielectric_constant",
    )
    charge_squared = charge_number * charge_number
    return float(
        -N_A
        * charge_squared
        * E_CHARGE**2
        * (1.0 - 1.0 / dielectric)
        / (BORN_DENOMINATOR_FACTOR * math.pi * EPS_0 * radius_m)
    )


def _solve_free_concentrations(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    total_concentrations: np.ndarray,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> np.ndarray:
    free_concentrations = np.asarray(total_concentrations, dtype=float)
    tolerance = MASS_BALANCE_TOLERANCE_FACTOR * max(
        1.0,
        float(np.max(total_concentrations)),
    )
    for _iteration_index in range(NEWTON_MAX_ITERATIONS):
        cluster_concentrations = _cluster_concentrations_array(
            components,
            cluster_templates,
            free_concentrations,
            solvent_environment,
            primitive_parameters,
        )
        residual = _mass_balance_residual_array(
            components,
            cluster_templates,
            free_concentrations,
            cluster_concentrations,
            total_concentrations,
        )
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= tolerance:
            return free_concentrations
        jacobian = _mass_balance_jacobian(
            components,
            cluster_templates,
            free_concentrations,
            total_concentrations,
            solvent_environment,
            primitive_parameters,
        )
        newton_step = np.linalg.solve(jacobian, residual)
        step_fraction = 1.0
        accepted_step = False
        while step_fraction >= NEWTON_MIN_STEP_FRACTION:
            trial_free_concentrations = (
                free_concentrations - step_fraction * newton_step
            )
            if np.all(trial_free_concentrations > 0.0):
                trial_cluster_concentrations = _cluster_concentrations_array(
                    components,
                    cluster_templates,
                    trial_free_concentrations,
                    solvent_environment,
                    primitive_parameters,
                )
                trial_residual = _mass_balance_residual_array(
                    components,
                    cluster_templates,
                    trial_free_concentrations,
                    trial_cluster_concentrations,
                    total_concentrations,
                )
                trial_norm = float(np.max(np.abs(trial_residual)))
                if trial_norm < residual_norm:
                    free_concentrations = trial_free_concentrations
                    accepted_step = True
                    break
            step_fraction *= NEWTON_LINE_SEARCH_BACKOFF
        if not accepted_step:
            raise ValueError(
                "generic mass-balance Newton solve failed to reduce residual"
            )
    raise ValueError("generic mass-balance Newton solve exceeded iteration limit")


def _cluster_concentrations_array(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> np.ndarray:
    component_index_by_name = {
        component.species_name: index for index, component in enumerate(components)
    }
    concentrations: list[float] = []
    for template in cluster_templates:
        activity_log_factor = _cluster_activity_log_factor(
            components,
            template,
            free_concentrations,
            component_index_by_name,
            solvent_environment,
            primitive_parameters,
        )
        exponent = (
            -template.standard_free_energy_J_mol
            / (R * solvent_environment.temperature_K)
            + activity_log_factor
        )
        equilibrium_constant = math.exp(exponent)
        concentration = STANDARD_STATE_CONCENTRATION_MOL_M3 * equilibrium_constant
        for species_name, stoichiometric_count in template.stoichiometry.items():
            component_index = component_index_by_name[species_name]
            concentration *= (
                free_concentrations[component_index]
                / STANDARD_STATE_CONCENTRATION_MOL_M3
            ) ** stoichiometric_count
        concentrations.append(float(concentration))
    return np.asarray(concentrations, dtype=float)


def _cluster_activity_log_factor(
    components: tuple[IonComponent, ...],
    cluster_template: ClusterStateTemplate,
    free_concentrations: np.ndarray,
    component_index_by_name: Mapping[str, int],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    ionic_strength_ratio = _free_ionic_strength_ratio(
        components,
        free_concentrations,
    )
    component_activity_log_sum = 0.0
    for species_name, stoichiometric_count in cluster_template.stoichiometry.items():
        component = components[component_index_by_name[species_name]]
        component_activity_log_sum += stoichiometric_count * _species_activity_ln_gamma(
            charge_number=component.charge_number,
            activity_size_radius_A=component.descriptor.cavity_radius_A,
            molecular_volume_A3=component.descriptor.molecular_volume_A3,
            ionic_strength_ratio=ionic_strength_ratio,
            solvent_environment=solvent_environment,
            primitive_parameters=primitive_parameters,
        )
    cluster_activity_log_gamma = _species_activity_ln_gamma(
        charge_number=cluster_template.net_charge_number,
        activity_size_radius_A=cluster_template.hydrodynamic_radius_A,
        molecular_volume_A3=cluster_template.molecular_volume_A3,
        ionic_strength_ratio=ionic_strength_ratio,
        solvent_environment=solvent_environment,
        primitive_parameters=primitive_parameters,
    )
    return float(
        component_activity_log_sum
        - primitive_parameters.cluster_activity_scale * cluster_activity_log_gamma
    )


def _free_ionic_strength_ratio(
    components: tuple[IonComponent, ...],
    free_concentrations: np.ndarray,
) -> float:
    ionic_strength_mol_m3 = 0.0
    for component, free_concentration_mol_m3 in zip(
        components,
        free_concentrations,
    ):
        ionic_strength_mol_m3 += (
            _positive_float(
                free_concentration_mol_m3,
                f"{component.species_name}.free_concentration_mol_m3",
            )
            * component.charge_number
            * component.charge_number
        )
    return _positive_float(
        ionic_strength_mol_m3 / ACTIVITY_IONIC_STRENGTH_REFERENCE_MOL_M3,
        "free_ionic_strength_ratio",
    )


def _species_activity_ln_gamma(
    charge_number: int,
    activity_size_radius_A: float,
    molecular_volume_A3: float,
    ionic_strength_ratio: float,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    square_root_ionic_strength = math.sqrt(
        _positive_float(ionic_strength_ratio, "ionic_strength_ratio")
    )
    size_radius_A = _positive_float(activity_size_radius_A, "activity_size_radius_A")
    debye_denominator = (
        1.0
        + primitive_parameters.activity_size_scale
        * size_radius_A
        * square_root_ionic_strength
    )
    debye_term = (
        -primitive_parameters.activity_debye_scale
        * charge_number
        * charge_number
        * square_root_ionic_strength
        / debye_denominator
    )
    hard_sphere_term = (
        primitive_parameters.activity_hard_sphere_scale
        * _positive_float(molecular_volume_A3, "activity_molecular_volume_A3")
        / _positive_float(
            solvent_environment.mean_molecular_volume_A3,
            "mean_molecular_volume_A3",
        )
        * _activity_packing_ratio(solvent_environment)
    )
    return float(debye_term + hard_sphere_term)


def _activity_packing_ratio(
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    hard_sphere_volume_fraction = _nonnegative_float(
        solvent_environment.hard_sphere_volume_fraction,
        "hard_sphere_volume_fraction",
    )
    if hard_sphere_volume_fraction >= 1.0:
        raise ValueError(
            "hard_sphere_volume_fraction must be below one for activity model"
        )
    return float(hard_sphere_volume_fraction / (1.0 - hard_sphere_volume_fraction))


def _mass_balance_residual(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    cluster_concentrations: Mapping[str, float],
    total_concentrations: np.ndarray,
) -> np.ndarray:
    cluster_array = np.asarray(
        [cluster_concentrations[template.label] for template in cluster_templates],
        dtype=float,
    )
    return _mass_balance_residual_array(
        components,
        cluster_templates,
        free_concentrations,
        cluster_array,
        total_concentrations,
    )


def _mass_balance_residual_array(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    cluster_concentrations: np.ndarray,
    total_concentrations: np.ndarray,
) -> np.ndarray:
    component_index_by_name = {
        component.species_name: index for index, component in enumerate(components)
    }
    reconstructed = np.asarray(free_concentrations, dtype=float).copy()
    for template_index, template in enumerate(cluster_templates):
        for species_name, stoichiometric_count in template.stoichiometry.items():
            component_index = component_index_by_name[species_name]
            reconstructed[component_index] += (
                stoichiometric_count * cluster_concentrations[template_index]
            )
    return reconstructed - total_concentrations


def _mass_balance_jacobian(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    total_concentrations: np.ndarray,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> np.ndarray:
    component_count = len(components)
    base_cluster_concentrations = _cluster_concentrations_array(
        components,
        cluster_templates,
        free_concentrations,
        solvent_environment,
        primitive_parameters,
    )
    base_residual = _mass_balance_residual_array(
        components,
        cluster_templates,
        free_concentrations,
        base_cluster_concentrations,
        total_concentrations,
    )
    jacobian = np.zeros((component_count, component_count), dtype=float)
    finite_difference_scale = math.sqrt(np.finfo(float).eps)
    for column_index in range(component_count):
        perturbation_mol_m3 = finite_difference_scale * max(
            1.0,
            abs(free_concentrations[column_index]),
        )
        trial_free_concentrations = np.asarray(free_concentrations, dtype=float).copy()
        trial_free_concentrations[column_index] += perturbation_mol_m3
        trial_cluster_concentrations = _cluster_concentrations_array(
            components,
            cluster_templates,
            trial_free_concentrations,
            solvent_environment,
            primitive_parameters,
        )
        trial_residual = _mass_balance_residual_array(
            components,
            cluster_templates,
            trial_free_concentrations,
            trial_cluster_concentrations,
            total_concentrations,
        )
        jacobian[:, column_index] = (
            trial_residual - base_residual
        ) / perturbation_mol_m3
    return jacobian


def _validate_solvent_environment(
    solvent_environment: MolecularSolventEnvironment,
) -> None:
    _positive_float(solvent_environment.dielectric_constant, "dielectric_constant")
    _positive_float(solvent_environment.viscosity_cP, "viscosity_cP")
    _nonnegative_float(
        solvent_environment.hard_sphere_volume_fraction,
        "hard_sphere_volume_fraction",
    )
    _positive_float(solvent_environment.temperature_K, "temperature_K")
    _positive_float(
        solvent_environment.solvent_effective_radius_A,
        "solvent_effective_radius_A",
    )
    _positive_float(
        solvent_environment.mean_molecular_volume_A3,
        "mean_molecular_volume_A3",
    )
    _nonnegative_float(
        solvent_environment.solvent_coordination_affinity_J_mol,
        "solvent_coordination_affinity_J_mol",
    )
    solvent_fraction_sum = math.fsum(
        _nonnegative_float(volume_fraction, f"{solvent_name}.volume_fraction")
        for solvent_name, volume_fraction in solvent_environment.solvent_volume_fractions.items()
    )
    _positive_float(solvent_fraction_sum, "solvent_volume_fraction_sum")


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _nonnegative_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        raise ValueError(f"{context} must be nonnegative and finite")
    return parsed_value


# ---- molecular_electrolyte_mori_generator.py ----

CP_TO_PA_S = 1.0e-3  # Unit conversion: cP to Pa*s.
GRAMS_PER_LITER_PER_G_ML = 1000.0  # Unit conversion: g/mL to g/L.
GRAMS_PER_M3_PER_G_ML = 1.0e6  # Unit conversion: g/mL to g/m^3.
STOKES_DENOMINATOR_FACTOR = 6.0  # Stokes-Einstein sphere denominator: 6*pi*eta*r.
CARTESIAN_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)  # Cartesian unit vectors for isotropic translation events.
TRANSLATION_EVENT_SIGNS = (-1.0, 1.0)  # Symmetric plus/minus jump directions.
ION_ATMOSPHERE_SOLVER_DIAGONAL = "diagonal_pnp_stokes_l1_cell_experimental"
MINIMUM_CLUSTER_ION_COUNT = (
    2  # Molecular production must include cation-anion pair states.
)
GAUSSIAN_CHARGE_CLOUD_FORM_FACTOR_DENOMINATOR = (
    6.0  # Gaussian F_q(kappa,a_q)=exp(-(kappa*a_q)^2/6).
)
ISOTROPIC_SHAPE_FACTOR = 1.0  # Dimensionless reference: lambda_s=1 is isotropic.
TRANSPORT_STATE_CONCENTRATION_RESOLUTION_FACTOR = np.finfo(
    float
).eps  # Markov-basis floor for zero-measure clusters.
TRANSPORT_ROLE_FREE_ION_CENTER = "free_ion_center"
TRANSPORT_ROLE_CONTACT_PAIR_CENTER = "contact_pair_center"
TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER = "solvent_separated_pair_center"
TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER = "charged_triplet_center"
TRANSPORT_ROLE_CLUSTER_COM_CENTER = "cluster_com_center"
TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER = "cluster_member_center"
TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER = "internal_polarization_center"
TRANSPORT_ROLE_LIGAND_SHELL_CENTER = "ligand_shell_center"
TRANSPORT_ROLE_NEUTRAL_CENTER = "neutral_center"
EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE = "associated_state_exchange"
PROJECTED_REACTIVE_FLUX_SOURCE_BY_EXACT_FAMILY = {
    "projected_capacity_flux": (
        "symmetric_reactive_flux_K_ij_from_executable_generator_capacity_surface"
    ),
    "association_conversion": (
        "symmetric_reactive_flux_K_ij_from_association_conversion_surface"
    ),
    "association_structural_hop": (
        "symmetric_reactive_flux_K_ij_from_structural_hop_surface"
    ),
    "neutral_translation": "zero_charge_neutral_self_flux",
    EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE: (
        "symmetric_reactive_flux_K_ij_from_associated_state_exchange_surface"
    ),
}
PROJECTED_DISPLACEMENT_MOMENT_SOURCE_BY_EXACT_FAMILY = {
    "projected_capacity_flux": (
        "conditional_charge_displacement_moments_from_executable_generator_path_measure"
    ),
    "association_conversion": (
        "conditional_association_conversion_charge_displacement"
    ),
    "association_structural_hop": (
        "conditional_association_structural_hop_charge_displacement"
    ),
    "neutral_translation": "zero_charge_neutral_displacement_moment",
    EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE: "conditional_associated_state_exchange_displacement",
}


@dataclass(frozen=True)
class MolecularMixtureProperties:
    density_g_ml: float
    viscosity_cP: float
    dielectric_constant: float


@dataclass(frozen=True)
class MolecularElectrolyteRecipe:
    cations: Mapping[str, float]
    anions: Mapping[str, float]
    solvents: Mapping[str, float]
    additives: Mapping[str, float]
    temperature_K: float
    pressure_Pa: float
    mixture_properties: MolecularMixtureProperties


@dataclass(frozen=True)
class MolecularMoriOptions:
    max_cluster_ion_count: int
    max_packing_fraction: float
    free_volume_exponent: float
    translation_jump_length_multiplier: float
    primitive_parameters: ConductivityPrimitiveParameterSet


@dataclass(frozen=True)
class MolecularTransportCenter:
    label: str
    parent_cluster_label: str
    parent_cluster_kind: str
    concentration_mol_m3: float
    center_species_name: str
    center_charge_number: int
    center_index: int
    hydrodynamic_radius_A: float
    charge_cloud_radius_A: float
    molecular_volume_A3: float
    ligand_field_asymmetry: float
    diffusion_m2_s: float
    local_obstruction_factor: float
    local_obstruction_diffusion_scale: float
    transport_role: str


class _TransportKineticLike(Protocol):
    label: str
    parent_cluster_label: str
    parent_cluster_kind: str
    concentration_mol_m3: float
    center_species_name: str
    center_charge_number: int
    center_index: int
    hydrodynamic_radius_A: float
    charge_cloud_radius_A: float
    molecular_volume_A3: float
    ligand_field_asymmetry: float
    diffusion_m2_s: float
    local_obstruction_factor: float
    local_obstruction_diffusion_scale: float
    transport_role: str


@dataclass(frozen=True)
class ProjectedChargedCenter:
    label: str
    charge_number: int
    diffusion_m2_s: float


@dataclass(frozen=True)
class ProjectedConstraintMode:
    first_center_label: str
    second_center_label: str
    lifetime_s: float
    length_m: float


@dataclass(frozen=True)
class ProjectedTransportState:
    label: str
    concentration_mol_m3: float
    charged_centers: tuple[ProjectedChargedCenter, ...]
    constraint_modes: tuple[ProjectedConstraintMode, ...]
    atmosphere_resistance_matrix_kg_s: tuple[tuple[float, ...], ...]
    mobility_covariance_matrix_m2_s: tuple[tuple[float, ...], ...]
    ligand_shell_features: Mapping[str, float]
    pair_basin: str
    residence_time_s: float
    partner_switch_time_s: float
    parent_cluster_label: str
    parent_cluster_kind: str
    center_species_name: str
    center_charge_number: int
    center_index: int
    hydrodynamic_radius_A: float
    charge_cloud_radius_A: float
    molecular_volume_A3: float
    ligand_field_asymmetry: float
    diffusion_m2_s: float
    local_obstruction_factor: float
    local_obstruction_diffusion_scale: float
    transport_role: str


@dataclass(frozen=True)
class MolecularIonAtmosphereDiagnostics:
    solver: str
    charged_carrier_count: int
    kappa_inv_m: float
    ionic_strength_mol_m3: float
    charge_cloud_form_factor_by_state: Mapping[str, float]
    friction_ratio_by_state: Mapping[str, float]
    zeta0_kg_s_by_state: Mapping[str, float]
    zeta_ep_kg_s_by_state: Mapping[str, float]
    zeta_rel_kg_s_by_state: Mapping[str, float]
    countercharge_relaxation_diffusivity_m2_s_by_state: Mapping[str, float]


@dataclass(frozen=True)
class _MolecularMixtureDescriptorState:
    hard_sphere_volume_fraction: float
    max_packing_fraction: float
    ionic_strength_mol_m3: float
    void_radius_A: float
    donor_number: float
    acceptor_number: float
    polarizability_volume_ratio: float
    solvation_obstruction_factor: float
    additive_solvation_obstruction_factor: float
    mean_anion_charge_cloud_radius_A: float
    anion_composition_entropy: float


@dataclass(frozen=True)
class _ChargeDensityReferenceEntry:
    concentration_mol_m3: float
    net_charge_number: int
    charge_cloud_radius_A: float


@dataclass(frozen=True)
class _TransportCenterConstructionContext:
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor]
    mixture_descriptor_state: _MolecularMixtureDescriptorState
    charge_density_reference_A_inv3: float
    solvent_environment: MolecularSolventEnvironment
    options: MolecularMoriOptions


@dataclass(frozen=True)
class MolecularAtmosphereMemoryPrimitive:
    state_label: str
    D_local_m2_s: float
    atmosphere_relaxation_diffusivity_m2_s: float
    jump_length_m: float
    k_capture_s_inv: float
    k_exit_s_inv: float
    atmosphere_coupling_fraction: float
    back_relaxation_probability: float
    mobile_concentration_mol_m3: float
    atmosphere_concentration_per_direction_mol_m3: float
    zeta0_kg_s: float
    zeta_ep_kg_s: float
    zeta_rel_kg_s: float


@dataclass(frozen=True)
class _AtmosphereTransportStateResult:
    transport_states: tuple[ProjectedTransportState, ...]
    diagnostics: MolecularIonAtmosphereDiagnostics


@dataclass(frozen=True)
class _MarkovProcessConstruction:
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: np.ndarray
    projected_transport_states: tuple[ProjectedTransportState, ...]
    events: tuple[MarkovAdditiveEvent, ...]
    memory_primitives: tuple[MolecularAtmosphereMemoryPrimitive, ...]
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ]
    atmosphere_mori_corrections: tuple[AtmosphereMoriCorrection, ...]


@dataclass(frozen=True)
class _ProjectedTransportInventory:
    projected_transport_states: tuple[ProjectedTransportState, ...]


@dataclass(frozen=True)
class RecipeProjectionBasis:
    basis_partitions_Ai: tuple[ProjectedTransportState, ...]
    partition_derivation_by_state_label: Mapping[str, TransportPartitionDefinitionAi]
    mori_basis_functions: tuple[str, ...]


@dataclass(frozen=True)
class _ReducedBasinCoordinateModel:
    state_label: str
    free_energy_reference_J_mol: float
    basin_measure_volume: float
    radial_coordinate_center_m: float
    radial_stiffness_J_mol_m2: float
    ligand_shell_coordinate_center: float
    ligand_shell_stiffness_J_mol: float
    atmosphere_coordinate_center: float
    atmosphere_stiffness_J_mol: float
    orientation_coordinate_center: float
    orientation_stiffness_J_mol: float
    free_volume_coordinate_center: float
    free_volume_stiffness_J_mol: float


@dataclass(frozen=True)
class _MoriMemoryCoordinateModel:
    state_label: str
    basis_function_label: str
    memory_family_label: str
    relaxation_rate_s_inv: float
    current_coupling_vector_h: tuple[float, float, float]


REDUCED_GENERATOR_STATE_COORDINATE_INDEX = 0
REDUCED_GENERATOR_RADIAL_COORDINATE_INDEX = 1
REDUCED_GENERATOR_LIGAND_COORDINATE_INDEX = 2
REDUCED_GENERATOR_ATMOSPHERE_COORDINATE_INDEX = 3
REDUCED_GENERATOR_ORIENTATION_COORDINATE_INDEX = 4
REDUCED_GENERATOR_FREE_VOLUME_COORDINATE_INDEX = (
    REDUCED_GENERATOR_ORIENTATION_COORDINATE_INDEX + 1
)
REDUCED_GENERATOR_COORDINATE_COUNT = REDUCED_GENERATOR_FREE_VOLUME_COORDINATE_INDEX + 1


@dataclass(frozen=True)
class AnalyticRecipeGenerator:
    configuration_space: str
    potential_energy_Ux: MolecularFreeEnergyFunctionalUx
    basis_partitions_Ai: tuple[ProjectedTransportState, ...]
    mori_basis_functions: tuple[str, ...]
    reduced_basin_coordinate_models: tuple[_ReducedBasinCoordinateModel, ...]
    total_restricted_concentration_mol_m3: float
    memory_coordinate_models: tuple[_MoriMemoryCoordinateModel, ...]
    recipe: MolecularElectrolyteRecipe
    descriptors: Mapping[str, MolecularSpeciesDescriptor]
    solvent_environment: MolecularSolventEnvironment
    speciation: GenericSpeciationResult
    cluster_templates: tuple[ClusterStateTemplate, ...]
    projected_transport_states: tuple[ProjectedTransportState, ...]
    ion_atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics
    options: MolecularMoriOptions

    def U(self, q: np.ndarray) -> float:
        coordinate_array = _reduced_generator_coordinate(q)
        basin_model = self._basin_model_from_coordinate(coordinate_array)
        return _reduced_basin_potential_energy_J_mol(
            basin_model,
            coordinate_array,
        )

    def grad_U(self, q: np.ndarray) -> np.ndarray:
        coordinate_array = _reduced_generator_coordinate(q)
        basin_model = self._basin_model_from_coordinate(coordinate_array)
        return _reduced_basin_potential_gradient_J_mol(
            basin_model,
            coordinate_array,
        )

    def P(self, q: np.ndarray) -> np.ndarray:
        coordinate_array = np.asarray(q, dtype=float)
        if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 4:
            raise ValueError("P(q) requires rows of charge_number,x_m,y_m,z_m")
        return np.sum(
            coordinate_array[:, 0:1] * coordinate_array[:, 1:4],
            axis=0,
        )

    def mu_integral(self, basin: ProjectedTransportState) -> float:
        if basin not in self.basis_partitions_Ai:
            raise ValueError(f"basin {basin.label} is not in the recipe basis")
        restricted_partition_weights = _reduced_basin_partition_weights(
            self.reduced_basin_coordinate_models,
            self.recipe.temperature_K,
        )
        partition_sum = _positive_float(
            math.fsum(restricted_partition_weights.values()),
            "analytic_recipe.reduced_coordinate_partition_sum",
        )
        total_restricted_concentration_mol_m3 = _positive_float(
            self.total_restricted_concentration_mol_m3,
            "analytic_recipe.total_restricted_concentration_mol_m3",
        )
        return (
            total_restricted_concentration_mol_m3
            * restricted_partition_weights[basin.label]
            / partition_sum
        )

    def capacity_flux(
        self,
        basin_i: ProjectedTransportState,
        basin_j: ProjectedTransportState,
    ) -> float:
        first_basin_model = self._basin_model_for_state(basin_i)
        second_basin_model = self._basin_model_for_state(basin_j)
        return _analytic_capacity_flux_from_reduced_generator(
            basin_i,
            basin_j,
            first_basin_model,
            second_basin_model,
            self.mu_integral(basin_i),
            self.mu_integral(basin_j),
            self.options,
            self.recipe.temperature_K,
        )

    def transition_path_moments(
        self,
        basin_i: ProjectedTransportState,
        basin_j: ProjectedTransportState,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _analytic_transition_path_moments_from_reduced_generator(
            basin_i,
            basin_j,
            self._basin_model_for_state(basin_i),
            self._basin_model_for_state(basin_j),
            self.options,
        )

    def self_current_tensor(
        self,
        basin: ProjectedTransportState,
    ) -> np.ndarray:
        charge_diffusivity_m2_s = compute_projected_transport_state_charge_diffusivity_m2_s(
            basin,
            self.recipe.temperature_K,
        )
        return np.diag(
            np.full(int(CARTESIAN_AXIS_COUNT), charge_diffusivity_m2_s, dtype=float)
        )

    def mori_A_h(
        self,
        mori_basis_functions: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        _validate_mori_basis_function_labels(mori_basis_functions)
        return _analytic_recipe_mori_A_h_from_memory_coordinates(
            self.memory_coordinate_models,
            mori_basis_functions,
        )

    def current_memory_coordinates(
        self,
    ) -> tuple[ProjectedCurrentMemoryCorrection, ...]:
        return _projected_current_memory_corrections_from_generator_mori_A_h(self)

    def _basin_model_from_coordinate(
        self,
        coordinate_array: np.ndarray,
    ) -> _ReducedBasinCoordinateModel:
        state_index = _projected_state_index_from_coordinate(
            coordinate_array,
            len(self.basis_partitions_Ai),
        )
        return self._basin_model_for_state(self.basis_partitions_Ai[state_index])

    def _basin_model_for_state(
        self,
        basin: ProjectedTransportState,
    ) -> _ReducedBasinCoordinateModel:
        for basin_model in self.reduced_basin_coordinate_models:
            if basin_model.state_label == basin.label:
                return basin_model
        raise ValueError(f"{basin.label} is missing a reduced coordinate model")


@dataclass(frozen=True)
class RecipePrimitiveProjection:
    recipe_generator: AnalyticRecipeGenerator
    projection_basis: RecipeProjectionBasis
    projected_transport_states: tuple[ProjectedTransportState, ...]
    projected_transport_model: ProjectedElectrolyteTransportModel
    projected_primitive_set: ProjectedPrimitiveSet
    memory_primitives: tuple[MolecularAtmosphereMemoryPrimitive, ...]
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ]
    atmosphere_mori_corrections: tuple[AtmosphereMoriCorrection, ...]


@dataclass(frozen=True)
class _MobileTransportStateIndex:
    projected_transport_state: ProjectedTransportState
    mobile_state_index: int
    mobile_concentration_mol_m3: float

    @property
    def transport_state(self) -> ProjectedTransportState:
        return self.projected_transport_state


@dataclass(frozen=True)
class _OnsagerCarrierState:
    carrier_group_key: tuple[str, str, int]
    label: str
    concentration_mol_m3: float
    charge_number: int
    diffusion_m2_s: float
    hydrodynamic_radius_A: float
    charge_cloud_form_factor: float


@dataclass(frozen=True)
class _AssociationStructuralHopKinetics:
    transition_surface: str
    partition_gap_scale_over_RT: float
    free_energy_mismatch_over_RT: float
    free_energy_barrier_over_RT: float
    effective_diffusivity_m2_s: float
    hop_length_m: float


@dataclass(frozen=True)
class _AdditiveLigandShellSummary:
    additive_ligand_site_occupancy: float
    additive_coordination_affinity_J_mol: float
    additive_solvation_support: float
    additive_molecular_volume_A3: float


def _projected_state_index_from_coordinate(q: np.ndarray, state_count: int) -> int:
    coordinate_array = np.asarray(q, dtype=float)
    if coordinate_array.size == 0:
        raise ValueError("projected coordinate q must contain a state coordinate")
    state_coordinate = float(coordinate_array.reshape(-1)[0])
    if not state_coordinate.is_integer():
        raise ValueError("projected basin coordinate must be an integer state index")
    state_index = int(state_coordinate)
    if state_index < 0 or state_index >= state_count:
        raise ValueError("projected basin coordinate is outside the basis")
    return state_index


def _reduced_generator_coordinate(q: np.ndarray) -> np.ndarray:
    coordinate_array = np.asarray(q, dtype=float).reshape(-1)
    if coordinate_array.size != REDUCED_GENERATOR_COORDINATE_COUNT:
        raise ValueError(
            "reduced generator coordinate requires "
            "(state_index,r_m,ligand_shell,atmosphere,orientation,free_volume)"
        )
    if not np.all(np.isfinite(coordinate_array)):
        raise ValueError("reduced generator coordinate contains non-finite values")
    return coordinate_array


def _reduced_basin_potential_energy_J_mol(
    basin_model: _ReducedBasinCoordinateModel,
    coordinate_array: np.ndarray,
) -> float:
    return float(
        basin_model.free_energy_reference_J_mol
        + 0.5
        * basin_model.radial_stiffness_J_mol_m2
        * (
            coordinate_array[REDUCED_GENERATOR_RADIAL_COORDINATE_INDEX]
            - basin_model.radial_coordinate_center_m
        )
        * (
            coordinate_array[REDUCED_GENERATOR_RADIAL_COORDINATE_INDEX]
            - basin_model.radial_coordinate_center_m
        )
        + 0.5
        * basin_model.ligand_shell_stiffness_J_mol
        * (
            coordinate_array[REDUCED_GENERATOR_LIGAND_COORDINATE_INDEX]
            - basin_model.ligand_shell_coordinate_center
        )
        * (
            coordinate_array[REDUCED_GENERATOR_LIGAND_COORDINATE_INDEX]
            - basin_model.ligand_shell_coordinate_center
        )
        + 0.5
        * basin_model.atmosphere_stiffness_J_mol
        * (
            coordinate_array[REDUCED_GENERATOR_ATMOSPHERE_COORDINATE_INDEX]
            - basin_model.atmosphere_coordinate_center
        )
        * (
            coordinate_array[REDUCED_GENERATOR_ATMOSPHERE_COORDINATE_INDEX]
            - basin_model.atmosphere_coordinate_center
        )
        + 0.5
        * basin_model.orientation_stiffness_J_mol
        * (
            coordinate_array[REDUCED_GENERATOR_ORIENTATION_COORDINATE_INDEX]
            - basin_model.orientation_coordinate_center
        )
        * (
            coordinate_array[REDUCED_GENERATOR_ORIENTATION_COORDINATE_INDEX]
            - basin_model.orientation_coordinate_center
        )
        + 0.5
        * basin_model.free_volume_stiffness_J_mol
        * (
            coordinate_array[REDUCED_GENERATOR_FREE_VOLUME_COORDINATE_INDEX]
            - basin_model.free_volume_coordinate_center
        )
        * (
            coordinate_array[REDUCED_GENERATOR_FREE_VOLUME_COORDINATE_INDEX]
            - basin_model.free_volume_coordinate_center
        )
    )


def _reduced_basin_potential_gradient_J_mol(
    basin_model: _ReducedBasinCoordinateModel,
    coordinate_array: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros(REDUCED_GENERATOR_COORDINATE_COUNT, dtype=float)
    gradient[REDUCED_GENERATOR_RADIAL_COORDINATE_INDEX] = (
        basin_model.radial_stiffness_J_mol_m2
        * (
            coordinate_array[REDUCED_GENERATOR_RADIAL_COORDINATE_INDEX]
            - basin_model.radial_coordinate_center_m
        )
    )
    gradient[REDUCED_GENERATOR_LIGAND_COORDINATE_INDEX] = (
        basin_model.ligand_shell_stiffness_J_mol
        * (
            coordinate_array[REDUCED_GENERATOR_LIGAND_COORDINATE_INDEX]
            - basin_model.ligand_shell_coordinate_center
        )
    )
    gradient[REDUCED_GENERATOR_ATMOSPHERE_COORDINATE_INDEX] = (
        basin_model.atmosphere_stiffness_J_mol
        * (
            coordinate_array[REDUCED_GENERATOR_ATMOSPHERE_COORDINATE_INDEX]
            - basin_model.atmosphere_coordinate_center
        )
    )
    gradient[REDUCED_GENERATOR_ORIENTATION_COORDINATE_INDEX] = (
        basin_model.orientation_stiffness_J_mol
        * (
            coordinate_array[REDUCED_GENERATOR_ORIENTATION_COORDINATE_INDEX]
            - basin_model.orientation_coordinate_center
        )
    )
    gradient[REDUCED_GENERATOR_FREE_VOLUME_COORDINATE_INDEX] = (
        basin_model.free_volume_stiffness_J_mol
        * (
            coordinate_array[REDUCED_GENERATOR_FREE_VOLUME_COORDINATE_INDEX]
            - basin_model.free_volume_coordinate_center
        )
    )
    return gradient


def _reduced_basin_partition_weights(
    basin_models: tuple[_ReducedBasinCoordinateModel, ...],
    temperature_K: float,
) -> Mapping[str, float]:
    beta_per_J_mol = 1.0 / (R * temperature_K)
    return {
        basin_model.state_label: _reduced_basin_partition_weight(
            basin_model,
            beta_per_J_mol,
        )
        for basin_model in basin_models
    }


def _reduced_basin_partition_weight(
    basin_model: _ReducedBasinCoordinateModel,
    beta_per_J_mol: float,
) -> float:
    harmonic_normalizer = math.prod(
        _harmonic_coordinate_normalizer(stiffness_J_mol, beta_per_J_mol)
        for stiffness_J_mol in (
            basin_model.radial_stiffness_J_mol_m2,
            basin_model.ligand_shell_stiffness_J_mol,
            basin_model.atmosphere_stiffness_J_mol,
            basin_model.orientation_stiffness_J_mol,
            basin_model.free_volume_stiffness_J_mol,
        )
    )
    return _positive_float(
        basin_model.basin_measure_volume
        * harmonic_normalizer
        * math.exp(-beta_per_J_mol * basin_model.free_energy_reference_J_mol),
        f"{basin_model.state_label}.reduced_basin_partition_weight",
    )


def _harmonic_coordinate_normalizer(
    stiffness_J_mol: float,
    beta_per_J_mol: float,
) -> float:
    return math.sqrt(
        2.0
        * math.pi
        / (
            beta_per_J_mol
            * _positive_float(stiffness_J_mol, "harmonic_coordinate_stiffness_J_mol")
        )
    )


def _validate_mori_basis_function_labels(mori_basis_functions: tuple[str, ...]) -> None:
    if not mori_basis_functions:
        raise ValueError("analytic generator requires Mori basis functions")
    if len(set(mori_basis_functions)) != len(mori_basis_functions):
        raise ValueError("analytic generator Mori basis functions must be unique")
    for basis_function_label in mori_basis_functions:
        if not basis_function_label:
            raise ValueError("analytic generator Mori basis label must be nonempty")


def _generator_basis_state_index(
    generator: AnalyticRecipeGenerator,
    basin: ProjectedTransportState,
) -> int:
    for state_index, projected_state in enumerate(generator.basis_partitions_Ai):
        if projected_state.label == basin.label:
            return state_index
    raise ValueError(f"{basin.label} is not in the generator basis")


def assert_no_species_names_in_transport_labels(
    projected_states: tuple[ProjectedTransportState, ...],
    events: tuple[MarkovAdditiveEvent, ...],
    residual_rollups: tuple[str, ...],
) -> None:
    species_name_tokens = _transport_species_name_tokens_from_state_diagnostics(
        projected_states
    )
    _assert_no_species_names_in_transport_labels(
        projected_states,
        events,
        residual_rollups,
        species_name_tokens,
    )


def _assert_no_species_names_in_transport_labels(
    projected_states: tuple[ProjectedTransportState, ...],
    events: tuple[MarkovAdditiveEvent, ...],
    residual_rollups: tuple[str, ...],
    species_names: tuple[str, ...],
) -> None:
    forbidden_tokens = _forbidden_species_name_tokens(species_names)
    label_values = _production_transport_label_values(
        projected_states,
        events,
        residual_rollups,
    )
    for label_value in label_values:
        for forbidden_token in forbidden_tokens:
            if forbidden_token in label_value:
                raise ValueError(
                    "transport production label contains species-name token "
                    f"{forbidden_token!r}: {label_value!r}"
                )


def _transport_species_name_tokens_from_state_diagnostics(
    projected_states: tuple[ProjectedTransportState, ...],
) -> tuple[str, ...]:
    species_name_candidates: list[str] = []
    for projected_state in projected_states:
        species_name_candidates.append(projected_state.center_species_name)
    return tuple(species_name_candidates)


def _forbidden_species_name_tokens(species_names: tuple[str, ...]) -> tuple[str, ...]:
    tokens: list[str] = []
    for species_name in species_names:
        for token in _species_name_transport_tokens(species_name):
            if token not in tokens:
                tokens.append(token)
    return tuple(tokens)


def _species_name_transport_tokens(species_name: str) -> tuple[str, ...]:
    raw_species_name = str(species_name)
    stripped_species_name = raw_species_name.strip()
    if not stripped_species_name:
        return tuple()
    candidate_tokens = [
        stripped_species_name,
        stripped_species_name.strip("+-_:;/,()[]{} "),
    ]
    alphanumeric_token = "".join(
        character
        for character in stripped_species_name
        if character.isalnum()
    )
    if len(alphanumeric_token) >= 3:
        candidate_tokens.append(alphanumeric_token)
    tokens: list[str] = []
    for candidate_token in candidate_tokens:
        if len(candidate_token) < 3:
            continue
        if candidate_token.startswith("feature_keyed"):
            continue
        if candidate_token in {
            "multi_center_projected_motif",
            "neutral_projected_motif",
            "generic_positive",
            "generic_negative",
        }:
            continue
        if candidate_token not in tokens:
            tokens.append(candidate_token)
    return tuple(tokens)


def _production_transport_label_values(
    projected_states: tuple[ProjectedTransportState, ...],
    events: tuple[MarkovAdditiveEvent, ...],
    residual_rollups: tuple[str, ...],
) -> tuple[str, ...]:
    label_values: list[str] = []
    for projected_state in projected_states:
        label_values.append(projected_state.label)
        label_values.append(projected_state.pair_basin)
        label_values.append(projected_state.transport_role)
        label_values.extend(str(feature_key) for feature_key in projected_state.ligand_shell_features)
        label_values.extend(
            charged_center.label for charged_center in projected_state.charged_centers
        )
        for constraint_mode in projected_state.constraint_modes:
            label_values.append(constraint_mode.first_center_label)
            label_values.append(constraint_mode.second_center_label)
    for event in events:
        label_values.append(event.label)
        label_values.append(event.family_label)
    label_values.extend(str(residual_rollup) for residual_rollup in residual_rollups)
    return tuple(label_values)


@dataclass(frozen=True)
class MolecularMoriConductivityResult:
    sigma_mS_cm: float
    sigma_S_m: float
    proof_status: str
    markov_additive_result: MarkovAdditiveConductivityResult
    projected_transport_model: ProjectedElectrolyteTransportModel
    descriptors: Mapping[str, MolecularSpeciesDescriptor]
    solvent_environment: MolecularSolventEnvironment
    speciation: GenericSpeciationResult
    cluster_states: tuple[ClusterStateTemplate, ...]
    transport_states: tuple[MolecularTransportCenter, ...]
    markov_state_labels: tuple[str, ...]
    markov_state_concentrations_mol_m3: tuple[float, ...]
    events: tuple[MarkovAdditiveEvent, ...]
    ion_atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics
    atmosphere_memory_primitives: tuple[MolecularAtmosphereMemoryPrimitive, ...]
    projected_current_memory_corrections: tuple[
        ProjectedCurrentMemoryCorrection,
        ...,
    ]
    atmosphere_mori_corrections: tuple[AtmosphereMoriCorrection, ...]
    mass_balance_residual_mol_m3: float
    detailed_balance_residual_mol_m3_s: float


def compute_molecular_electrolyte_conductivity(
    recipe: MolecularElectrolyteRecipe,
    species_inputs: Mapping[str, MolecularSpeciesInput],
    descriptor_backend: MolecularDescriptorBackend,
    options: MolecularMoriOptions,
) -> MolecularMoriConductivityResult:
    return _compute_molecular_electrolyte_conductivity(
        recipe,
        species_inputs,
        descriptor_backend,
        options,
        {},
    )


def compute_molecular_electrolyte_conductivity_with_diagnostic_cluster_shifts(
    recipe: MolecularElectrolyteRecipe,
    species_inputs: Mapping[str, MolecularSpeciesInput],
    descriptor_backend: MolecularDescriptorBackend,
    options: MolecularMoriOptions,
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
) -> MolecularMoriConductivityResult:
    return _compute_molecular_electrolyte_conductivity(
        recipe,
        species_inputs,
        descriptor_backend,
        options,
        diagnostic_cluster_standard_free_energy_shift_over_RT_by_label,
    )


def build_analytic_recipe_microscopic_generator(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    options: MolecularMoriOptions,
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
) -> AnalyticRecipeGenerator:
    solvent_environment = _molecular_solvent_environment(
        recipe,
        descriptors,
        options,
    )
    components = _ion_components(recipe, descriptors)
    _validate_ionic_charge_balance(components)
    cluster_templates = build_cluster_state_templates(
        components,
        solvent_environment,
        ClusterEnumerationOptions(
            max_cluster_ion_count=options.max_cluster_ion_count,
            primitive_parameters=options.primitive_parameters,
        ),
    )
    cluster_templates = _apply_cluster_standard_free_energy_shifts(
        cluster_templates,
        diagnostic_cluster_standard_free_energy_shift_over_RT_by_label,
        solvent_environment.temperature_K,
    )
    speciation = _solve_projected_speciation_mass_balance(
        recipe,
        descriptors,
        components,
        cluster_templates,
        solvent_environment,
        options.primitive_parameters,
    )
    local_projected_transport_states = _projected_transport_states_from_mass_balance(
        recipe,
        descriptors,
        speciation,
        solvent_environment,
        options,
    )
    atmosphere_transport_state_result = (
        _ion_atmosphere_diagnostics_for_projected_transport_states(
            local_projected_transport_states,
            solvent_environment,
            options,
        )
    )
    projected_transport_inventory = _projected_basis_transport_inventory_from_states(
        atmosphere_transport_state_result.transport_states,
        speciation,
        solvent_environment,
        options,
        recipe.temperature_K,
    )
    (
        reduced_basin_coordinate_models,
        total_restricted_concentration_mol_m3,
    ) = _analytic_recipe_reduced_coordinate_models(
        projected_transport_inventory.projected_transport_states,
        components,
        cluster_templates,
        solvent_environment,
        options,
        recipe.temperature_K,
    )
    analytic_restricted_populations_c_i_mol_m3 = (
        _analytic_restricted_populations_from_reduced_generator(
            projected_transport_inventory.projected_transport_states,
            reduced_basin_coordinate_models,
            total_restricted_concentration_mol_m3,
            recipe.temperature_K,
        )
    )
    analytic_projected_transport_states = (
        _projected_transport_states_with_primitive_populations(
            projected_transport_inventory.projected_transport_states,
            analytic_restricted_populations_c_i_mol_m3,
        )
    )
    memory_primitives = _memory_primitives_from_projected_transport_states(
        analytic_projected_transport_states,
        atmosphere_transport_state_result.diagnostics,
        options,
        recipe.temperature_K,
    )
    memory_coordinate_models = _analytic_recipe_mori_memory_coordinate_models(
        analytic_projected_transport_states,
        memory_primitives,
        recipe.temperature_K,
    )
    return AnalyticRecipeGenerator(
        configuration_space=("analytic_recipe_projected_motif_configuration_space"),
        potential_energy_Ux=_molecular_free_energy_functional_derivation(
            recipe.temperature_K
        ),
        basis_partitions_Ai=analytic_projected_transport_states,
        mori_basis_functions=tuple(
            f"current_memory_basis:{projected_state.label}"
            for projected_state in analytic_projected_transport_states
        ),
        reduced_basin_coordinate_models=reduced_basin_coordinate_models,
        total_restricted_concentration_mol_m3=(
            total_restricted_concentration_mol_m3
        ),
        memory_coordinate_models=memory_coordinate_models,
        recipe=recipe,
        descriptors=dict(descriptors),
        solvent_environment=solvent_environment,
        speciation=speciation,
        cluster_templates=cluster_templates,
        projected_transport_states=analytic_projected_transport_states,
        ion_atmosphere_diagnostics=atmosphere_transport_state_result.diagnostics,
        options=options,
    )


def _analytic_recipe_reduced_coordinate_models(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    temperature_K: float,
) -> tuple[tuple[_ReducedBasinCoordinateModel, ...], float]:
    if not projected_transport_states:
        raise ValueError("analytic recipe generator requires projected basins")
    total_restricted_concentration_mol_m3 = _analytic_total_restricted_concentration_mol_m3(
        projected_transport_states,
        components,
    )
    cluster_template_by_label = {
        cluster_template.label: cluster_template for cluster_template in cluster_templates
    }
    reduced_basin_coordinate_models: list[_ReducedBasinCoordinateModel] = []
    for projected_state in projected_transport_states:
        reduced_basin_coordinate_models.append(
            _reduced_basin_coordinate_model_from_projected_state(
                projected_state,
                cluster_template_by_label,
                solvent_environment,
                options,
                temperature_K,
            )
        )
    return (
        tuple(reduced_basin_coordinate_models),
        total_restricted_concentration_mol_m3,
    )


def _reduced_basin_coordinate_model_from_projected_state(
    projected_state: ProjectedTransportState,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    temperature_K: float,
) -> _ReducedBasinCoordinateModel:
    radial_coordinate_center_m = _reduced_basin_radial_coordinate_center_m(
        projected_state
    )
    ligand_shell_coordinate_center = _reduced_basin_ligand_coordinate_center(
        projected_state
    )
    atmosphere_coordinate_center = _reduced_basin_atmosphere_coordinate_center(
        projected_state
    )
    orientation_coordinate_center = _reduced_basin_orientation_coordinate_center(
        projected_state
    )
    free_volume_coordinate_center = _reduced_basin_free_volume_coordinate_center(
        projected_state
    )
    radial_stiffness_J_mol_m2 = _reduced_radial_stiffness_J_mol_m2(
        projected_state,
        temperature_K,
    )
    dimensionless_stiffness_J_mol = R * temperature_K
    return _ReducedBasinCoordinateModel(
        state_label=projected_state.label,
        free_energy_reference_J_mol=_analytic_projected_state_pmf_energy_J_mol(
            projected_state,
            cluster_template_by_label,
            solvent_environment,
            options,
            temperature_K,
        ),
        basin_measure_volume=_analytic_basin_measure_volume(projected_state),
        radial_coordinate_center_m=radial_coordinate_center_m,
        radial_stiffness_J_mol_m2=radial_stiffness_J_mol_m2,
        ligand_shell_coordinate_center=ligand_shell_coordinate_center,
        ligand_shell_stiffness_J_mol=dimensionless_stiffness_J_mol,
        atmosphere_coordinate_center=atmosphere_coordinate_center,
        atmosphere_stiffness_J_mol=dimensionless_stiffness_J_mol,
        orientation_coordinate_center=orientation_coordinate_center,
        orientation_stiffness_J_mol=dimensionless_stiffness_J_mol,
        free_volume_coordinate_center=free_volume_coordinate_center,
        free_volume_stiffness_J_mol=dimensionless_stiffness_J_mol,
    )


def _analytic_total_restricted_concentration_mol_m3(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    components: tuple[IonComponent, ...],
) -> float:
    component_reference_concentration_mol_m3 = math.fsum(
        _positive_float(
            component.analytical_concentration_M,
            f"{component.species_name}.analytical_concentration_M",
        )
        * STANDARD_STATE_CONCENTRATION_MOL_M3
        for component in components
    )
    if component_reference_concentration_mol_m3 > 0.0:
        return component_reference_concentration_mol_m3
    neutral_reference_concentration_mol_m3 = math.fsum(
        _nonnegative_float(
            projected_state.concentration_mol_m3,
            f"{projected_state.label}.neutral_reference_concentration_mol_m3",
        )
        for projected_state in projected_transport_states
    )
    return _positive_float(
        neutral_reference_concentration_mol_m3,
        "analytic_recipe.neutral_total_restricted_concentration_mol_m3",
    )


def _analytic_basin_measure_volume(
    projected_state: ProjectedTransportState,
) -> float:
    molecular_volume_A3 = _positive_float(
        projected_state.molecular_volume_A3,
        f"{projected_state.label}.molecular_volume_A3",
    )
    hydrodynamic_radius_A = _positive_float(
        projected_state.hydrodynamic_radius_A,
        f"{projected_state.label}.hydrodynamic_radius_A",
    )
    charge_cloud_radius_A = _positive_float(
        projected_state.charge_cloud_radius_A,
        f"{projected_state.label}.charge_cloud_radius_A",
    )
    ligand_shell_factor = 1.0 + 0.05 * len(projected_state.ligand_shell_features)
    return _positive_float(
        molecular_volume_A3
        * ligand_shell_factor
        / (hydrodynamic_radius_A * charge_cloud_radius_A),
        f"{projected_state.label}.analytic_basin_measure_volume",
    )


def _reduced_basin_radial_coordinate_center_m(
    projected_state: ProjectedTransportState,
) -> float:
    radius_sum_A = _positive_float(
        projected_state.hydrodynamic_radius_A,
        f"{projected_state.label}.hydrodynamic_radius_A",
    ) + _positive_float(
        projected_state.charge_cloud_radius_A,
        f"{projected_state.label}.charge_cloud_radius_A",
    )
    return radius_sum_A * ANGSTROM_TO_M


def _reduced_basin_ligand_coordinate_center(
    projected_state: ProjectedTransportState,
) -> float:
    if "ligand_site_count" not in projected_state.ligand_shell_features:
        return 0.0
    return _nonnegative_float(
        projected_state.ligand_shell_features["ligand_site_count"],
        f"{projected_state.label}.ligand_site_count",
    )


def _reduced_basin_atmosphere_coordinate_center(
    projected_state: ProjectedTransportState,
) -> float:
    return abs(projected_state.center_charge_number) * _nonnegative_float(
        projected_state.local_obstruction_factor,
        f"{projected_state.label}.atmosphere_obstruction_factor",
    )


def _reduced_basin_orientation_coordinate_center(
    projected_state: ProjectedTransportState,
) -> float:
    return _positive_float(
        projected_state.ligand_field_asymmetry,
        f"{projected_state.label}.ligand_field_asymmetry",
    )


def _reduced_basin_free_volume_coordinate_center(
    projected_state: ProjectedTransportState,
) -> float:
    local_obstruction_factor = _nonnegative_float(
        projected_state.local_obstruction_factor,
        f"{projected_state.label}.free_volume_obstruction_factor",
    )
    return 1.0 / (1.0 + local_obstruction_factor)


def _reduced_radial_stiffness_J_mol_m2(
    projected_state: ProjectedTransportState,
    temperature_K: float,
) -> float:
    radial_width_m = max(
        _positive_float(
            projected_state.hydrodynamic_radius_A,
            f"{projected_state.label}.hydrodynamic_radius_A",
        ),
        _positive_float(
            projected_state.charge_cloud_radius_A,
            f"{projected_state.label}.charge_cloud_radius_A",
        ),
    ) * ANGSTROM_TO_M
    return _positive_float(
        R * temperature_K / (radial_width_m * radial_width_m),
        f"{projected_state.label}.radial_stiffness_J_mol_m2",
    )


def _analytic_projected_state_pmf_energy_J_mol(
    projected_state: ProjectedTransportState,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    temperature_K: float,
) -> float:
    if projected_state.parent_cluster_label in cluster_template_by_label:
        cluster_template = cluster_template_by_label[projected_state.parent_cluster_label]
        cluster_energy_J_mol = cluster_template.standard_free_energy_J_mol
    else:
        cluster_energy_J_mol = 0.0
    charge_cloud_energy_J_mol = (
        R
        * temperature_K
        * abs(projected_state.center_charge_number)
        * _positive_float(
            projected_state.hydrodynamic_radius_A,
            f"{projected_state.label}.hydrodynamic_radius_A",
        )
        / _positive_float(
            projected_state.charge_cloud_radius_A,
            f"{projected_state.label}.charge_cloud_radius_A",
        )
    )
    obstruction_energy_J_mol = (
        R
        * temperature_K
        * _nonnegative_float(
            projected_state.local_obstruction_factor,
            f"{projected_state.label}.local_obstruction_factor",
        )
    )
    ligand_site_count = 0.0
    if "ligand_site_count" in projected_state.ligand_shell_features:
        ligand_site_count = _nonnegative_float(
            projected_state.ligand_shell_features["ligand_site_count"],
            f"{projected_state.label}.ligand_site_count",
        )
    ligand_shell_energy_J_mol = (
        -solvent_environment.additive_coordination_affinity_J_mol
        * _nonnegative_float(
            solvent_environment.additive_ligand_site_occupancy,
            "additive_ligand_site_occupancy",
        )
        * ligand_site_count
    )
    steric_energy_J_mol = (
        R
        * temperature_K
        * options.primitive_parameters.steric_free_energy_scale
        * _positive_float(
            projected_state.ligand_field_asymmetry,
            f"{projected_state.label}.ligand_field_asymmetry",
        )
    )
    internal_polarization_energy_J_mol = (
        _internal_polarization_projection_pmf_energy_J_mol(
            projected_state,
            options.primitive_parameters,
            temperature_K,
        )
    )
    return float(
        cluster_energy_J_mol
        + charge_cloud_energy_J_mol
        + obstruction_energy_J_mol
        + ligand_shell_energy_J_mol
        + steric_energy_J_mol
        + internal_polarization_energy_J_mol
    )


def _internal_polarization_projection_pmf_energy_J_mol(
    projected_state: ProjectedTransportState,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    temperature_K: float,
) -> float:
    if projected_state.transport_role != TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER:
        return 0.0
    ionic_strength_driver = math.log1p(
        _nonnegative_float(
            projected_state.local_obstruction_factor,
            f"{projected_state.label}.internal_polarization_local_obstruction",
        )
    )
    counterion_crowding_driver = float(max(len(projected_state.charged_centers) - 1, 0))
    projection_logit = (
        primitive_parameters.internal_polarization_projection_offset
        + primitive_parameters.internal_polarization_projection_ionic_strength_slope
        * ionic_strength_driver
        + primitive_parameters.internal_polarization_projection_counterion_crowding_slope
        * counterion_crowding_driver
    )
    return float(-R * temperature_K * projection_logit)


def _analytic_restricted_populations_from_reduced_generator(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    reduced_basin_coordinate_models: tuple[_ReducedBasinCoordinateModel, ...],
    total_restricted_concentration_mol_m3: float,
    temperature_K: float,
) -> np.ndarray:
    partition_weight_by_label = _reduced_basin_partition_weights(
        reduced_basin_coordinate_models,
        temperature_K,
    )
    partition_weights = np.asarray(
        [
            _positive_float(
                partition_weight_by_label[projected_state.label],
                f"{projected_state.label}.reduced_partition_weight",
            )
            for projected_state in projected_transport_states
        ],
        dtype=float,
    )
    partition_sum = _positive_float(
        float(np.sum(partition_weights)),
        "analytic_recipe.partition_sum",
    )
    return (
        _positive_float(
            total_restricted_concentration_mol_m3,
            "analytic_recipe.total_restricted_concentration_mol_m3",
        )
        * partition_weights
        / partition_sum
    )


def _analytic_capacity_flux_from_reduced_generator(
    first_state: ProjectedTransportState,
    second_state: ProjectedTransportState,
    first_basin_model: _ReducedBasinCoordinateModel,
    second_basin_model: _ReducedBasinCoordinateModel,
    first_population_mol_m3: float,
    second_population_mol_m3: float,
    options: MolecularMoriOptions,
    temperature_K: float,
) -> float:
    if not _states_share_reactive_boundary(first_state, second_state):
        return 0.0
    barrier_energy_J_mol = _reduced_generator_surface_barrier_J_mol(
        first_basin_model,
        second_basin_model,
    )
    basin_minimum_energy_J_mol = min(
        first_basin_model.free_energy_reference_J_mol,
        second_basin_model.free_energy_reference_J_mol,
    )
    barrier_over_RT = (
        barrier_energy_J_mol - basin_minimum_energy_J_mol
    ) / (R * temperature_K)
    effective_diffusivity_m2_s = _positive_float(
        0.5 * (first_state.diffusion_m2_s + second_state.diffusion_m2_s),
        f"{first_state.label}->{second_state.label}.effective_diffusivity_m2_s",
    )
    hop_length_m = max(
        _jump_length_m(first_state, options),
        _jump_length_m(second_state, options),
    )
    transition_rate_s_inv = (
        options.primitive_parameters.association_conversion_rate_scale
        * effective_diffusivity_m2_s
        * math.exp(-barrier_over_RT)
        / (hop_length_m * hop_length_m)
    )
    return _nonnegative_float(
        min(first_population_mol_m3, second_population_mol_m3) * transition_rate_s_inv,
        f"{first_state.label}->{second_state.label}.analytic_capacity_flux",
    )


def _reduced_generator_surface_barrier_J_mol(
    first_basin_model: _ReducedBasinCoordinateModel,
    second_basin_model: _ReducedBasinCoordinateModel,
) -> float:
    radial_midpoint_m = 0.5 * (
        first_basin_model.radial_coordinate_center_m
        + second_basin_model.radial_coordinate_center_m
    )
    ligand_midpoint = 0.5 * (
        first_basin_model.ligand_shell_coordinate_center
        + second_basin_model.ligand_shell_coordinate_center
    )
    atmosphere_midpoint = 0.5 * (
        first_basin_model.atmosphere_coordinate_center
        + second_basin_model.atmosphere_coordinate_center
    )
    orientation_midpoint = 0.5 * (
        first_basin_model.orientation_coordinate_center
        + second_basin_model.orientation_coordinate_center
    )
    free_volume_midpoint = 0.5 * (
        first_basin_model.free_volume_coordinate_center
        + second_basin_model.free_volume_coordinate_center
    )
    first_surface_coordinate = np.asarray(
        (
            0.0,
            radial_midpoint_m,
            ligand_midpoint,
            atmosphere_midpoint,
            orientation_midpoint,
            free_volume_midpoint,
        ),
        dtype=float,
    )
    second_surface_coordinate = np.asarray(
        (
            0.0,
            radial_midpoint_m,
            ligand_midpoint,
            atmosphere_midpoint,
            orientation_midpoint,
            free_volume_midpoint,
        ),
        dtype=float,
    )
    return max(
        _reduced_basin_potential_energy_J_mol(
            first_basin_model,
            first_surface_coordinate,
        ),
        _reduced_basin_potential_energy_J_mol(
            second_basin_model,
            second_surface_coordinate,
        ),
    )


def _analytic_transition_path_moments_from_reduced_generator(
    first_state: ProjectedTransportState,
    second_state: ProjectedTransportState,
    first_basin_model: _ReducedBasinCoordinateModel,
    second_basin_model: _ReducedBasinCoordinateModel,
    options: MolecularMoriOptions,
) -> tuple[np.ndarray, np.ndarray]:
    first_moment, second_moment = _analytic_transition_path_moments_between_states(
        first_state,
        second_state,
        options,
    )
    if not (
        _state_has_charge_polarization(first_state)
        or _state_has_charge_polarization(second_state)
    ):
        return first_moment, second_moment
    radial_surface_gap_m = abs(
        first_basin_model.radial_coordinate_center_m
        - second_basin_model.radial_coordinate_center_m
    )
    if radial_surface_gap_m <= 0.0:
        return first_moment, second_moment
    radial_second_moment = np.diag(
        np.full(
            int(CARTESIAN_AXIS_COUNT),
            options.primitive_parameters.jump_length_scale
            * radial_surface_gap_m
            * radial_surface_gap_m,
            dtype=float,
        )
    )
    return first_moment, second_moment + radial_second_moment


def _states_share_reactive_boundary(
    first_state: ProjectedTransportState,
    second_state: ProjectedTransportState,
) -> bool:
    if first_state.label == second_state.label:
        return False
    if _is_associated_exchange_state(first_state) or _is_associated_exchange_state(
        second_state
    ):
        return True
    if first_state.parent_cluster_label == second_state.parent_cluster_label:
        return True
    if first_state.center_charge_number == -second_state.center_charge_number:
        return True
    return first_state.pair_basin != second_state.pair_basin


def _analytic_transition_path_moments_between_states(
    first_state: ProjectedTransportState,
    second_state: ProjectedTransportState,
    options: MolecularMoriOptions,
) -> tuple[np.ndarray, np.ndarray]:
    if _is_associated_exchange_state(first_state) or _is_associated_exchange_state(
        second_state
    ):
        return (
            np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float),
            np.zeros(
                (int(CARTESIAN_AXIS_COUNT), int(CARTESIAN_AXIS_COUNT)),
                dtype=float,
            ),
        )
    if not (
        _state_has_charge_polarization(first_state)
        or _state_has_charge_polarization(second_state)
    ):
        return (
            np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float),
            np.zeros(
                (int(CARTESIAN_AXIS_COUNT), int(CARTESIAN_AXIS_COUNT)),
                dtype=float,
            ),
        )
    first_moment = _charge_polarization_centroid_m(first_state) - (
        _charge_polarization_centroid_m(second_state)
    )
    hop_length_m = max(
        _jump_length_m(first_state, options),
        _jump_length_m(second_state, options),
    )
    isotropic_second_moment = np.diag(
        np.full(
            int(CARTESIAN_AXIS_COUNT),
            options.primitive_parameters.jump_length_scale * hop_length_m * hop_length_m,
            dtype=float,
        )
    )
    return first_moment, isotropic_second_moment + np.outer(first_moment, first_moment)


def _state_has_charge_polarization(
    projected_state: ProjectedTransportState,
) -> bool:
    if projected_state.charged_centers:
        return any(
            charged_center.charge_number != 0
            for charged_center in projected_state.charged_centers
        )
    return projected_state.center_charge_number != 0


def _charge_polarization_centroid_m(
    projected_state: ProjectedTransportState,
) -> np.ndarray:
    _ = projected_state
    return np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float)


def _memory_primitives_from_projected_transport_states(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
    options: MolecularMoriOptions,
    temperature_K: float,
) -> tuple[MolecularAtmosphereMemoryPrimitive, ...]:
    memory_primitives: list[MolecularAtmosphereMemoryPrimitive] = []
    for transport_state in projected_transport_states:
        if _is_associated_exchange_state(transport_state):
            continue
        if not _is_dc_self_current_carrier(transport_state):
            continue
        if transport_state.center_charge_number == 0:
            continue
        if _state_has_zero_atmosphere_coupling(transport_state, atmosphere_diagnostics):
            continue
        memory_primitives.append(
            _atmosphere_memory_primitive(
                transport_state,
                options,
                atmosphere_diagnostics,
                temperature_K,
            )
        )
    return tuple(memory_primitives)


def _analytic_recipe_mori_memory_coordinate_models(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    memory_primitives: tuple[MolecularAtmosphereMemoryPrimitive, ...],
    temperature_K: float,
) -> tuple[_MoriMemoryCoordinateModel, ...]:
    if not memory_primitives:
        return tuple()
    state_by_label = {
        projected_state.label: projected_state
        for projected_state in projected_transport_states
    }
    memory_coordinate_models: list[_MoriMemoryCoordinateModel] = []
    for memory_primitive in memory_primitives:
        if memory_primitive.state_label not in state_by_label:
            raise ValueError(
                f"{memory_primitive.state_label} memory primitive has no state"
            )
        projected_state = state_by_label[memory_primitive.state_label]
        memory_self_energy_s_inv = _positive_float(
            memory_primitive.k_capture_s_inv + memory_primitive.k_exit_s_inv,
            f"{memory_primitive.state_label}.analytic_memory_self_energy_s_inv",
        )
        correction_axis_density = _analytic_memory_axis_density(
            projected_state,
            memory_primitive,
            temperature_K,
        )
        current_coupling_vector_h = tuple(
            float(value)
            for value in np.sqrt(correction_axis_density * memory_self_energy_s_inv)
        )
        memory_coordinate_models.append(
            _MoriMemoryCoordinateModel(
                state_label=memory_primitive.state_label,
                basis_function_label=(
                    f"current_memory_basis:{memory_primitive.state_label}"
                ),
                memory_family_label="atmosphere_polarization_memory_coordinate",
                relaxation_rate_s_inv=memory_self_energy_s_inv,
                current_coupling_vector_h=current_coupling_vector_h,
            )
        )
    return tuple(memory_coordinate_models)


def _analytic_recipe_mori_A_h_from_memory_coordinates(
    memory_coordinate_models: tuple[_MoriMemoryCoordinateModel, ...],
    mori_basis_functions: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not memory_coordinate_models:
        memory_energy_matrix = np.zeros((0, 0), dtype=float)
        current_coupling_matrix = np.zeros(
            (int(CARTESIAN_AXIS_COUNT), 0),
            dtype=float,
        )
        _validated_current_coupling_matrix(current_coupling_matrix, 0)
        return memory_energy_matrix, current_coupling_matrix
    basis_index_by_label = {
        basis_function_label: basis_index
        for basis_index, basis_function_label in enumerate(mori_basis_functions)
    }
    memory_energy_matrix = np.zeros(
        (len(memory_coordinate_models), len(memory_coordinate_models)),
        dtype=float,
    )
    current_coupling_matrix = np.zeros(
        (int(CARTESIAN_AXIS_COUNT), len(memory_coordinate_models)),
        dtype=float,
    )
    for mode_index, memory_coordinate_model in enumerate(memory_coordinate_models):
        if memory_coordinate_model.basis_function_label not in basis_index_by_label:
            raise ValueError(
                f"{memory_coordinate_model.basis_function_label} is missing from "
                "the Mori basis"
            )
        memory_energy_matrix[mode_index, mode_index] = _positive_float(
            memory_coordinate_model.relaxation_rate_s_inv,
            f"{memory_coordinate_model.state_label}.memory_relaxation_rate_s_inv",
        )
        current_coupling_matrix[:, mode_index] = np.asarray(
            memory_coordinate_model.current_coupling_vector_h,
            dtype=float,
        )
    memory_energy_matrix = _validated_square_matrix(
        memory_energy_matrix,
        "generator Mori memory energy matrix A",
    )
    _validate_symmetric_matrix(
        memory_energy_matrix,
        "generator Mori memory energy matrix A",
    )
    _validate_positive_semidefinite_matrix(
        memory_energy_matrix,
        "generator Mori memory energy matrix A",
    )
    current_coupling_matrix = _validated_current_coupling_matrix(
        current_coupling_matrix,
        memory_energy_matrix.shape[0],
    )
    return memory_energy_matrix, current_coupling_matrix


def _analytic_memory_axis_density(
    projected_state: ProjectedTransportState,
    memory_primitive: MolecularAtmosphereMemoryPrimitive,
    temperature_K: float,
) -> np.ndarray:
    ordinary_axis_density = np.asarray(
        _ordinary_self_current_axis_density_m2_s_mol_m3(
            projected_state,
            temperature_K,
        ),
        dtype=float,
    )
    memory_fraction = _nonnegative_float(
        memory_primitive.atmosphere_coupling_fraction
        * memory_primitive.back_relaxation_probability,
        f"{memory_primitive.state_label}.analytic_memory_fraction",
    )
    if memory_fraction > 1.0:
        raise ValueError(
            f"{memory_primitive.state_label}.analytic_memory_fraction exceeds one"
        )
    return ordinary_axis_density * memory_fraction


def _projected_current_memory_corrections_from_generator_mori_A_h(
    generator: AnalyticRecipeGenerator,
) -> tuple[ProjectedCurrentMemoryCorrection, ...]:
    memory_energy_matrix, current_coupling_matrix = generator.mori_A_h(
        generator.mori_basis_functions
    )
    if memory_energy_matrix.shape == (0, 0):
        return tuple()
    _validated_square_matrix(
        memory_energy_matrix,
        "generator.current_memory_coordinates.A",
    )
    _validated_current_coupling_matrix(
        current_coupling_matrix,
        memory_energy_matrix.shape[0],
    )
    if len(generator.memory_coordinate_models) != memory_energy_matrix.shape[0]:
        raise ValueError("generator Mori coordinate model count mismatch")
    state_by_label = {
        projected_state.label: projected_state
        for projected_state in generator.basis_partitions_Ai
    }
    corrections: list[ProjectedCurrentMemoryCorrection] = []
    for mode_index, memory_coordinate_model in enumerate(
        generator.memory_coordinate_models
    ):
        transport_state_label = memory_coordinate_model.state_label
        memory_family_label = memory_coordinate_model.memory_family_label
        if transport_state_label not in state_by_label:
            raise ValueError(f"{transport_state_label} has no projected state")
        memory_self_energy_s_inv = _positive_float(
            memory_energy_matrix[mode_index, mode_index],
            f"{transport_state_label}.A_h_memory_self_energy_s_inv",
        )
        correction_axis_density = (
            current_coupling_matrix[:, mode_index]
            * current_coupling_matrix[:, mode_index]
            / memory_self_energy_s_inv
        )
        corrections.append(
            _projected_current_memory_correction(
                state_by_label[transport_state_label],
                memory_family_label,
                memory_self_energy_s_inv,
                correction_axis_density,
                generator.recipe.temperature_K,
            )
        )
    return tuple(corrections)


def build_recipe_projection_basis(
    recipe_generator: AnalyticRecipeGenerator,
) -> RecipeProjectionBasis:
    state_labels = tuple(
        f"{projected_state.label}:mobile"
        for projected_state in recipe_generator.projected_transport_states
    )
    return RecipeProjectionBasis(
        basis_partitions_Ai=recipe_generator.projected_transport_states,
        partition_derivation_by_state_label=(
            _transport_partition_derivations_by_state_label(
                state_labels,
                recipe_generator.projected_transport_states,
            )
        ),
        mori_basis_functions=recipe_generator.mori_basis_functions,
    )


def project_analytic_generator_to_primitives(
    recipe_generator: AnalyticRecipeGenerator,
    projection_basis: RecipeProjectionBasis,
) -> RecipePrimitiveProjection:
    projected_primitive_set = project_generator_to_primitives(
        recipe_generator,
        projection_basis,
    )
    projected_transport_states = _projected_transport_states_with_primitive_populations(
        projection_basis.basis_partitions_Ai,
        projected_primitive_set.restricted_equilibrium_populations_c_i_mol_m3,
    )
    (
        reactive_flux_integrals,
        self_displacement_moments,
        _self_direct_axis_density,
    ) = _projected_flux_integrals_from_events(
        projected_primitive_set.markov_input.state_labels,
        np.asarray(
            projected_primitive_set.markov_input.state_concentrations_mol_m3,
            dtype=float,
        ),
        projected_primitive_set.markov_input.events,
        projected_transport_states,
    )
    projected_current_memory_corrections = recipe_generator.current_memory_coordinates()
    memory_primitives = _memory_primitives_from_projected_transport_states(
        recipe_generator.basis_partitions_Ai,
        recipe_generator.ion_atmosphere_diagnostics,
        recipe_generator.options,
        recipe_generator.recipe.temperature_K,
    )
    projected_transport_model = compute_projected_electrolyte_transport_model(
        projected_primitive_set,
        free_energy_functional=recipe_generator.potential_energy_Ux,
        partition_derivation_by_state_label=(
            projection_basis.partition_derivation_by_state_label
        ),
        projected_transport_states=projected_transport_states,
        reactive_flux_integrals=reactive_flux_integrals,
        self_displacement_moments=self_displacement_moments,
        self_direct_axis_density=tuple(
            float(value) for value in _self_direct_axis_density
        ),
        projected_current_memory_corrections=projected_current_memory_corrections,
    )
    projected_transport_model = replace(
        projected_transport_model,
        projected_generator_model=_projected_generator_model_for_transport_model(
            projected_transport_model
        ),
    )
    _validate_projected_transport_derivations(projected_transport_model)
    return RecipePrimitiveProjection(
        recipe_generator=recipe_generator,
        projection_basis=projection_basis,
        projected_transport_states=projected_transport_states,
        projected_transport_model=projected_transport_model,
        projected_primitive_set=projected_primitive_set,
        memory_primitives=memory_primitives,
        projected_current_memory_corrections=projected_current_memory_corrections,
        atmosphere_mori_corrections=tuple(),
    )


def project_generator_to_primitives(
    generator: AnalyticRecipeGenerator,
    projection_basis: RecipeProjectionBasis,
) -> ProjectedPrimitiveSet:
    state_labels = tuple(
        f"{state.label}:mobile" for state in projection_basis.basis_partitions_Ai
    )
    state_concentrations = np.asarray(
        [
            generator.mu_integral(projected_state)
            for projected_state in projection_basis.basis_partitions_Ai
        ],
        dtype=float,
    )
    if len(state_labels) != len(projection_basis.basis_partitions_Ai):
        raise ValueError("projection basis size must match primitive state labels")
    projected_primitive_set = _projected_primitive_set_from_executable_generator(
        generator,
        projection_basis,
        state_labels,
        state_concentrations,
    )
    generator_mori_energy_matrix, generator_current_coupling_matrix = (
        generator.mori_A_h(projection_basis.mori_basis_functions)
    )
    if not np.array_equal(
        generator_mori_energy_matrix,
        projected_primitive_set.mori_memory_energy_matrix_A,
    ):
        raise ValueError("generator A matrix does not match projected primitive set")
    if not np.array_equal(
        generator_current_coupling_matrix,
        projected_primitive_set.mori_current_coupling_matrix_h,
    ):
        raise ValueError("generator h matrix does not match projected primitive set")
    return projected_primitive_set


def _projected_transport_states_with_primitive_populations(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    restricted_populations_c_i_mol_m3: np.ndarray,
) -> tuple[ProjectedTransportState, ...]:
    state_count = len(projected_transport_states)
    restricted_populations = _validated_state_concentrations(
        restricted_populations_c_i_mol_m3,
        state_count,
    )
    return tuple(
        replace(
            projected_state,
            concentration_mol_m3=float(restricted_populations[state_index]),
        )
        for state_index, projected_state in enumerate(projected_transport_states)
    )


def _projected_primitive_set_from_executable_generator(
    generator: AnalyticRecipeGenerator,
    projection_basis: RecipeProjectionBasis,
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: np.ndarray,
) -> ProjectedPrimitiveSet:
    state_count = len(state_labels)
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        state_count,
    )
    symmetric_fluxes = np.zeros((state_count, state_count), dtype=float)
    reversible_generator = np.zeros((state_count, state_count), dtype=float)
    first_moments = np.zeros(
        (state_count, state_count, int(CARTESIAN_AXIS_COUNT)),
        dtype=float,
    )
    second_moments = np.zeros(
        (
            state_count,
            state_count,
            int(CARTESIAN_AXIS_COUNT),
            int(CARTESIAN_AXIS_COUNT),
        ),
        dtype=float,
    )
    self_current_tensors = np.zeros(
        (state_count, int(CARTESIAN_AXIS_COUNT), int(CARTESIAN_AXIS_COUNT)),
        dtype=float,
    )
    events: list[MarkovAdditiveEvent] = []
    basis_states = projection_basis.basis_partitions_Ai
    for state_index, projected_state in enumerate(basis_states):
        self_current_tensor = generator.self_current_tensor(projected_state)
        self_current_tensor = _symmetrized_matrix(self_current_tensor)
        _validate_positive_semidefinite_matrix(
            self_current_tensor,
            f"{projected_state.label}.D_self_i",
        )
        self_current_tensors[state_index, :, :] = self_current_tensor
        if float(np.max(np.abs(self_current_tensor))) > 0.0:
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index,
                    to_state_index=state_index,
                    rate_s_inv=1.0,
                    charge_displacement_m=(0.0, 0.0, 0.0),
                    charge_displacement_second_moment_m2=_matrix_to_tuple_rows(
                        2.0 * self_current_tensor
                    ),
                    label=f"projected_self_current:{projected_state.label}",
                    family_label=_projected_self_current_event_family_label(
                        projected_state
                    ),
                )
            )
    for first_index, first_state in enumerate(basis_states):
        for second_index, second_state in enumerate(basis_states[first_index + 1 :]):
            target_index = first_index + second_index + 1
            symmetric_flux = generator.capacity_flux(
                first_state,
                second_state,
            )
            if symmetric_flux == 0.0:
                continue
            symmetric_flux = _positive_float(
                symmetric_flux,
                f"{first_state.label}->{second_state.label}.K_ij_mol_m3_s",
            )
            first_moment, second_moment = generator.transition_path_moments(
                first_state,
                second_state,
            )
            first_moment = _validated_displacement(
                tuple(float(value) for value in first_moment),
                f"{first_state.label}->{second_state.label}.d_ij_m",
            )
            second_moment = _validated_second_moment_tensor(
                _matrix_to_tuple_rows(second_moment),
                f"{first_state.label}->{second_state.label}.M_ij_m2",
            )
            symmetric_fluxes[first_index, target_index] = symmetric_flux
            symmetric_fluxes[target_index, first_index] = symmetric_flux
            first_moments[first_index, target_index, :] = first_moment
            first_moments[target_index, first_index, :] = -np.asarray(
                first_moment,
                dtype=float,
            )
            second_moments[first_index, target_index, :, :] = second_moment
            second_moments[target_index, first_index, :, :] = second_moment
            forward_rate_s_inv = symmetric_flux / state_concentrations[first_index]
            reverse_rate_s_inv = symmetric_flux / state_concentrations[target_index]
            capacity_event_family_label = _projected_capacity_event_family_label(
                first_state,
                second_state,
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=first_index,
                    to_state_index=target_index,
                    rate_s_inv=_positive_float(
                        forward_rate_s_inv,
                        f"{first_state.label}->{second_state.label}.Q_ij_s_inv",
                    ),
                    charge_displacement_m=tuple(float(value) for value in first_moment),
                    charge_displacement_second_moment_m2=_matrix_to_tuple_rows(
                        second_moment
                    ),
                    label=f"projected_capacity_flux:{first_state.label}->{second_state.label}",
                    family_label=capacity_event_family_label,
                )
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=target_index,
                    to_state_index=first_index,
                    rate_s_inv=_positive_float(
                        reverse_rate_s_inv,
                        f"{second_state.label}->{first_state.label}.Q_ji_s_inv",
                    ),
                    charge_displacement_m=tuple(
                        float(-value) for value in first_moment
                    ),
                    charge_displacement_second_moment_m2=_matrix_to_tuple_rows(
                        second_moment
                    ),
                    label=f"projected_capacity_flux:{second_state.label}->{first_state.label}",
                    family_label=capacity_event_family_label,
                )
            )
    markov_input = MarkovAdditiveConductivityInput(
        state_labels=state_labels,
        state_concentrations_mol_m3=state_concentrations,
        events=tuple(events),
        temperature_K=generator.recipe.temperature_K,
    )
    event_conductivity_result = compute_markov_additive_green_kubo_conductivity(
        markov_input
    )
    projected_current_memory_corrections = generator.current_memory_coordinates()
    conductivity_result = _markov_result_with_projected_current_memory(
        event_conductivity_result,
        projected_current_memory_corrections,
        tuple(),
        generator.recipe.temperature_K,
    )
    reversible_generator[:, :] = _validated_generator_matrix(
        conductivity_result.generator_s_inv
    )
    generator_mori_energy_matrix, generator_current_coupling_matrix = (
        generator.mori_A_h(projection_basis.mori_basis_functions)
    )
    return ProjectedPrimitiveSet(
        state_labels=state_labels,
        restricted_equilibrium_populations_c_i_mol_m3=state_concentrations.copy(),
        symmetric_reactive_fluxes_K_ij_mol_m3_s=symmetric_fluxes,
        reversible_generator_Q_ij_s_inv=reversible_generator.copy(),
        conditional_displacement_first_moments_d_ij_m=first_moments,
        conditional_displacement_second_moments_M_ij_m2=second_moments,
        self_current_diffusion_tensors_D_self_i_m2_s=self_current_tensors,
        mori_memory_energy_matrix_A=generator_mori_energy_matrix.copy(),
        mori_current_coupling_matrix_h=generator_current_coupling_matrix.copy(),
        markov_input=markov_input,
        markov_conductivity_result=conductivity_result,
    )


def _projected_capacity_event_family_label(
    first_state: ProjectedTransportState,
    second_state: ProjectedTransportState,
) -> str:
    if _is_associated_exchange_state(first_state) or _is_associated_exchange_state(
        second_state
    ):
        return EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE
    return "projected_capacity_flux"


def compute_projected_gk_mori_conductivity(
    projected_primitives: ProjectedPrimitiveSet,
) -> MarkovAdditiveConductivityResult:
    return projected_primitives.markov_conductivity_result


def _compute_molecular_electrolyte_conductivity(
    recipe: MolecularElectrolyteRecipe,
    species_inputs: Mapping[str, MolecularSpeciesInput],
    descriptor_backend: MolecularDescriptorBackend,
    options: MolecularMoriOptions,
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
) -> MolecularMoriConductivityResult:
    _validate_recipe(recipe)
    _validate_options(options)
    descriptors = _describe_recipe_species(
        recipe,
        species_inputs,
        descriptor_backend,
    )
    recipe_generator = build_analytic_recipe_microscopic_generator(
        recipe,
        descriptors,
        options,
        diagnostic_cluster_standard_free_energy_shift_over_RT_by_label,
    )
    projection_basis = build_recipe_projection_basis(recipe_generator)
    recipe_primitive_projection = project_analytic_generator_to_primitives(
        recipe_generator,
        projection_basis,
    )
    markov_result = compute_projected_gk_mori_conductivity(
        recipe_primitive_projection.projected_primitive_set
    )
    projected_transport_model = recipe_primitive_projection.projected_transport_model
    markov_input = recipe_primitive_projection.projected_primitive_set.markov_input
    return MolecularMoriConductivityResult(
        sigma_mS_cm=markov_result.sigma_mS_cm,
        sigma_S_m=markov_result.sigma_S_m,
        proof_status=projected_transport_model.proof_status,
        markov_additive_result=markov_result,
        projected_transport_model=projected_transport_model,
        descriptors=descriptors,
        solvent_environment=recipe_generator.solvent_environment,
        speciation=recipe_generator.speciation,
        cluster_states=recipe_generator.cluster_templates,
        transport_states=_diagnostic_transport_centers_from_projected_transport_states(
            recipe_primitive_projection.projected_transport_states
        ),
        markov_state_labels=markov_input.state_labels,
        markov_state_concentrations_mol_m3=tuple(
            float(concentration_mol_m3)
            for concentration_mol_m3 in markov_input.state_concentrations_mol_m3
        ),
        events=markov_input.events,
        ion_atmosphere_diagnostics=recipe_generator.ion_atmosphere_diagnostics,
        atmosphere_memory_primitives=recipe_primitive_projection.memory_primitives,
        projected_current_memory_corrections=(
            recipe_primitive_projection.projected_current_memory_corrections
        ),
        atmosphere_mori_corrections=(
            recipe_primitive_projection.atmosphere_mori_corrections
        ),
        mass_balance_residual_mol_m3=(
            recipe_generator.speciation.mass_balance_residual_mol_m3
        ),
        detailed_balance_residual_mol_m3_s=(
            markov_result.validation.detailed_balance_residual_mol_m3_s
        ),
    )


def _describe_recipe_species(
    recipe: MolecularElectrolyteRecipe,
    species_inputs: Mapping[str, MolecularSpeciesInput],
    descriptor_backend: MolecularDescriptorBackend,
) -> Mapping[str, MolecularSpeciesDescriptor]:
    recipe_species_names = _recipe_species_names(recipe)
    descriptors: dict[str, MolecularSpeciesDescriptor] = {}
    for species_name in recipe_species_names:
        if species_name not in species_inputs:
            raise ValueError(f"missing molecular species input for {species_name}")
        descriptors[species_name] = descriptor_backend.describe_species(
            species_inputs[species_name],
            recipe.temperature_K,
        )
    return descriptors


def _recipe_species_names(recipe: MolecularElectrolyteRecipe) -> tuple[str, ...]:
    species_names: list[str] = []
    for loading_map in (
        recipe.cations,
        recipe.anions,
        recipe.solvents,
        recipe.additives,
    ):
        for species_name in loading_map:
            if species_name not in species_names:
                species_names.append(species_name)
    if not species_names:
        raise ValueError(
            "molecular electrolyte recipe must contain at least one species"
        )
    return tuple(species_names)


def _molecular_solvent_environment(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    options: MolecularMoriOptions,
) -> MolecularSolventEnvironment:
    additive_ligand_summary = _additive_ligand_shell_summary(recipe, descriptors)
    return MolecularSolventEnvironment(
        dielectric_constant=recipe.mixture_properties.dielectric_constant,
        viscosity_cP=recipe.mixture_properties.viscosity_cP,
        hard_sphere_volume_fraction=_hard_sphere_volume_fraction(
            recipe,
            descriptors,
        ),
        temperature_K=recipe.temperature_K,
        solvent_effective_radius_A=_mixture_effective_radius_A(
            recipe,
            descriptors,
        ),
        mean_molecular_volume_A3=_mixture_mean_molecular_volume_A3(
            recipe,
            descriptors,
        ),
        solvent_volume_fractions=dict(recipe.solvents),
        solvent_coordination_affinity_J_mol=(
            _mixture_solvent_coordination_affinity_J_mol(recipe, descriptors)
        ),
        additive_ligand_site_occupancy=(
            additive_ligand_summary.additive_ligand_site_occupancy
        ),
        additive_coordination_affinity_J_mol=(
            additive_ligand_summary.additive_coordination_affinity_J_mol
        ),
        additive_solvation_support=(additive_ligand_summary.additive_solvation_support),
        additive_molecular_volume_A3=(
            additive_ligand_summary.additive_molecular_volume_A3
        ),
    )


def _additive_ligand_shell_summary(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> _AdditiveLigandShellSummary:
    solvent_site_concentration_mol_m3 = 0.0
    additive_site_concentration_mol_m3 = 0.0
    additive_concentration_mol_m3 = 0.0
    additive_affinity_weighted_sum_J_mol_mol_m3 = 0.0
    additive_solvation_support_weighted_sum_mol_m3 = 0.0
    additive_volume_weighted_sum_A3_mol_m3 = 0.0
    for species_name, volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        solvent_concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            volume_fraction,
            descriptor,
        )
        solvent_site_concentration_mol_m3 += (
            solvent_concentration_mol_m3 * _coordination_site_count(descriptor)
        )
    for species_name, weight_fraction in recipe.additives.items():
        descriptor = descriptors[species_name]
        positive_weight_fraction = _positive_float(
            weight_fraction,
            f"{species_name}.weight_fraction",
        )
        species_concentration_mol_m3 = (
            positive_weight_fraction
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol
        )
        coordination_site_count = _coordination_site_count(descriptor)
        additive_site_concentration_mol_m3 += (
            species_concentration_mol_m3 * coordination_site_count
        )
        additive_concentration_mol_m3 += species_concentration_mol_m3
        additive_affinity_weighted_sum_J_mol_mol_m3 += (
            species_concentration_mol_m3
            * _nonnegative_float(
                descriptor.coordination_affinity_J_mol,
                f"{species_name}.coordination_affinity_J_mol",
            )
        )
        additive_solvation_support_weighted_sum_mol_m3 += (
            species_concentration_mol_m3 * _additive_solvation_support(descriptor)
        )
        additive_volume_weighted_sum_A3_mol_m3 += (
            species_concentration_mol_m3
            * _positive_float(
                descriptor.molecular_volume_A3,
                f"{species_name}.molecular_volume_A3",
            )
        )
    ligand_site_denominator_mol_m3 = (
        solvent_site_concentration_mol_m3 + additive_site_concentration_mol_m3
    )
    if ligand_site_denominator_mol_m3 > 0.0:
        additive_ligand_site_occupancy = (
            additive_site_concentration_mol_m3 / ligand_site_denominator_mol_m3
        )
    else:
        additive_ligand_site_occupancy = 0.0
    if additive_concentration_mol_m3 > 0.0:
        additive_coordination_affinity_J_mol = (
            additive_affinity_weighted_sum_J_mol_mol_m3 / additive_concentration_mol_m3
        )
        additive_solvation_support = (
            additive_solvation_support_weighted_sum_mol_m3
            / additive_concentration_mol_m3
        )
        additive_molecular_volume_A3 = (
            additive_volume_weighted_sum_A3_mol_m3 / additive_concentration_mol_m3
        )
    else:
        additive_coordination_affinity_J_mol = 0.0
        additive_solvation_support = 0.0
        additive_molecular_volume_A3 = 0.0
    return _AdditiveLigandShellSummary(
        additive_ligand_site_occupancy=_nonnegative_float(
            additive_ligand_site_occupancy,
            "additive_ligand_site_occupancy",
        ),
        additive_coordination_affinity_J_mol=_nonnegative_float(
            additive_coordination_affinity_J_mol,
            "additive_coordination_affinity_J_mol",
        ),
        additive_solvation_support=_nonnegative_float(
            additive_solvation_support,
            "additive_solvation_support",
        ),
        additive_molecular_volume_A3=_nonnegative_float(
            additive_molecular_volume_A3,
            "additive_molecular_volume_A3",
        ),
    )


def _solve_projected_speciation_mass_balance(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> GenericSpeciationResult:
    speciation = solve_generic_mass_balance(
        components,
        cluster_templates,
        solvent_environment,
        primitive_parameters,
    )
    ligand_features = _neutral_ligand_features(recipe, descriptors)
    if not ligand_features:
        return speciation
    return _solve_ligand_coupled_projected_speciation(
        speciation,
        ligand_features,
        solvent_environment,
        primitive_parameters,
    )


def _solve_ligand_coupled_projected_speciation(
    speciation: GenericSpeciationResult,
    ligand_features: tuple[_NeutralLigandFeature, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> GenericSpeciationResult:
    if not speciation.components:
        return replace(
            speciation,
            neutral_ligand_site_concentrations_mol_m3={
                ligand_feature.feature_label: ligand_feature.site_concentration_mol_m3
                for ligand_feature in ligand_features
            },
        )
    total_component_concentrations_mol_m3 = np.asarray(
        [
            _positive_float(
                component.analytical_concentration_M,
                f"{component.species_name}.analytical_concentration_M",
            )
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            for component in speciation.components
        ],
        dtype=float,
    )
    total_ligand_site_concentrations_mol_m3 = np.asarray(
        [
            _positive_float(
                ligand_feature.site_concentration_mol_m3,
                f"{ligand_feature.feature_label}.site_concentration_mol_m3",
            )
            for ligand_feature in ligand_features
        ],
        dtype=float,
    )
    ligand_association_constants_m3_mol = _neutral_ligand_association_constants_m3_mol(
        ligand_features,
        solvent_environment.temperature_K,
    )
    speciation_problem = _LigandCoupledSpeciationProblem(
        components=speciation.components,
        cluster_templates=speciation.cluster_templates,
        total_component_concentrations_mol_m3=total_component_concentrations_mol_m3,
        total_ligand_site_concentrations_mol_m3=(
            total_ligand_site_concentrations_mol_m3
        ),
        ligand_association_constants_m3_mol=ligand_association_constants_m3_mol,
        solvent_environment=solvent_environment,
        primitive_parameters=primitive_parameters,
    )
    initial_free_component_concentrations_mol_m3 = np.asarray(
        [
            _positive_float(
                speciation.free_component_concentrations_mol_m3[
                    component.species_name
                ],
                f"{component.species_name}.initial_free_component_mol_m3",
            )
            for component in speciation.components
        ],
        dtype=float,
    )
    initial_unknowns_mol_m3 = _ligand_coupled_initial_unknowns_mol_m3(
        initial_free_component_concentrations_mol_m3,
        speciation.components,
        total_ligand_site_concentrations_mol_m3,
        ligand_association_constants_m3_mol,
    )
    solved_unknowns_mol_m3 = _solve_ligand_coupled_free_concentrations_mol_m3(
        initial_unknowns_mol_m3,
        speciation_problem,
    )
    component_count = len(speciation.components)
    free_component_concentrations_mol_m3 = solved_unknowns_mol_m3[:component_count]
    free_ligand_concentrations_mol_m3 = solved_unknowns_mol_m3[component_count:]
    mass_action_state = _ligand_coupled_mass_action_state(
        free_component_concentrations_mol_m3,
        free_ligand_concentrations_mol_m3,
        speciation_problem,
    )
    component_by_name = {
        component.species_name: component for component in speciation.components
    }
    cation_ligand_concentrations_mol_m3: dict[str, float] = {}
    cation_ligand_component_species_by_label: dict[str, str] = {}
    cation_ligand_anion_concentrations_mol_m3: dict[str, float] = {}
    cation_ligand_anion_parent_cluster_by_label: dict[str, str] = {}
    free_component_ligand_bound_concentrations_mol_m3: dict[str, float] = {}
    cluster_ligand_bound_concentrations_mol_m3: dict[str, float] = {}
    binary_ligand_bound_by_component_mol_m3 = np.zeros(component_count, dtype=float)
    for component_index, component in enumerate(speciation.components):
        if component.charge_number <= 0:
            continue
        binding_substrate = _NeutralLigandBindingSubstrate(
            substrate_label=_cation_ligand_substrate_label(component),
            substrate_kind=NEUTRAL_LIGAND_BINDING_SUBSTRATE_FREE_CATION,
            source_species_name=component.species_name,
            source_cluster_label="",
            total_concentration_mol_m3=float(
                total_component_concentrations_mol_m3[component_index]
            ),
            cation_feature_label=_cation_ligand_substrate_label(component),
        )
        for ligand_index, ligand_feature in enumerate(ligand_features):
            bound_concentration_mol_m3 = _nonnegative_float(
                mass_action_state.cation_ligand_concentrations_mol_m3[
                    component_index,
                    ligand_index,
                ],
                (
                    f"{component.species_name}:{ligand_feature.feature_label}."
                    "cation_ligand_concentration_mol_m3"
                ),
            )
            if bound_concentration_mol_m3 == 0.0:
                continue
            motif_label = _neutral_ligand_motif_label(binding_substrate, ligand_feature)
            cation_ligand_concentrations_mol_m3[motif_label] = (
                bound_concentration_mol_m3
            )
            cation_ligand_component_species_by_label[motif_label] = (
                component.species_name
            )
            binary_ligand_bound_by_component_mol_m3[component_index] += (
                bound_concentration_mol_m3
            )
        if binary_ligand_bound_by_component_mol_m3[component_index] != 0.0:
            free_component_ligand_bound_concentrations_mol_m3[
                component.species_name
            ] = binary_ligand_bound_by_component_mol_m3[component_index]
    for cluster_index, cluster_template in enumerate(speciation.cluster_templates):
        if cluster_template.cluster_kind != SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
            continue
        positive_species_name = _positive_component_species_name(
            cluster_template,
            component_by_name,
        )
        binding_substrate = _NeutralLigandBindingSubstrate(
            substrate_label=_ssip_ligand_substrate_label(
                cluster_template,
                component_by_name,
            ),
            substrate_kind=NEUTRAL_LIGAND_BINDING_SUBSTRATE_SOLVENT_SEPARATED_PAIR,
            source_species_name=positive_species_name,
            source_cluster_label=cluster_template.label,
            total_concentration_mol_m3=float(
                mass_action_state.cluster_concentrations_mol_m3[cluster_index]
            ),
            cation_feature_label=_cation_ligand_substrate_label(
                component_by_name[positive_species_name]
            ),
        )
        cluster_ligand_bound_concentration_mol_m3 = 0.0
        for ligand_index, ligand_feature in enumerate(ligand_features):
            bound_concentration_mol_m3 = _nonnegative_float(
                mass_action_state.cation_ligand_anion_concentrations_mol_m3[
                    cluster_index,
                    ligand_index,
                ],
                (
                    f"{cluster_template.label}:{ligand_feature.feature_label}."
                    "cation_ligand_anion_concentration_mol_m3"
                ),
            )
            if bound_concentration_mol_m3 == 0.0:
                continue
            motif_label = _neutral_ligand_motif_label(binding_substrate, ligand_feature)
            cation_ligand_anion_concentrations_mol_m3[motif_label] = (
                bound_concentration_mol_m3
            )
            cation_ligand_anion_parent_cluster_by_label[motif_label] = (
                cluster_template.label
            )
            cluster_ligand_bound_concentration_mol_m3 += (
                bound_concentration_mol_m3
            )
        if cluster_ligand_bound_concentration_mol_m3 != 0.0:
            cluster_ligand_bound_concentrations_mol_m3[cluster_template.label] = (
                cluster_ligand_bound_concentration_mol_m3
            )
    cluster_concentrations_mol_m3 = {
        cluster_template.label: float(
            mass_action_state.cluster_concentrations_mol_m3[cluster_index]
        )
        for cluster_index, cluster_template in enumerate(speciation.cluster_templates)
    }
    free_component_concentrations_by_name_mol_m3 = {
        component.species_name: float(
            free_component_concentrations_mol_m3[index]
            + binary_ligand_bound_by_component_mol_m3[index]
        )
        for index, component in enumerate(speciation.components)
    }
    residual_norm_mol_m3 = float(
        np.max(np.abs(mass_action_state.residual_vector_mol_m3))
    )
    return replace(
        speciation,
        free_component_concentrations_mol_m3=(
            free_component_concentrations_by_name_mol_m3
        ),
        cluster_concentrations_mol_m3=cluster_concentrations_mol_m3,
        neutral_ligand_site_concentrations_mol_m3={
            ligand_feature.feature_label: ligand_feature.site_concentration_mol_m3
            for ligand_feature in ligand_features
        },
        cation_ligand_concentrations_mol_m3=cation_ligand_concentrations_mol_m3,
        cation_ligand_component_species_by_label=(
            cation_ligand_component_species_by_label
        ),
        cation_ligand_anion_concentrations_mol_m3=(
            cation_ligand_anion_concentrations_mol_m3
        ),
        cation_ligand_anion_parent_cluster_by_label=(
            cation_ligand_anion_parent_cluster_by_label
        ),
        free_component_ligand_bound_concentrations_mol_m3=(
            free_component_ligand_bound_concentrations_mol_m3
        ),
        cluster_ligand_bound_concentrations_mol_m3=(
            cluster_ligand_bound_concentrations_mol_m3
        ),
        solvation_competition_pmf_partitions=solvation_competition_pmf_partition(
            speciation.components,
            speciation.cluster_templates,
            free_component_concentrations_by_name_mol_m3,
            cluster_concentrations_mol_m3,
            solvent_environment,
        ),
        mass_balance_residual_mol_m3=residual_norm_mol_m3,
    )


def _solve_ligand_coupled_free_concentrations_mol_m3(
    initial_unknowns_mol_m3: np.ndarray,
    speciation_problem: _LigandCoupledSpeciationProblem,
) -> np.ndarray:
    current_unknowns_mol_m3 = initial_unknowns_mol_m3.copy()
    tolerance_mol_m3 = MASS_BALANCE_TOLERANCE_FACTOR * max(
        1.0,
        float(np.max(speciation_problem.total_component_concentrations_mol_m3)),
        float(np.max(speciation_problem.total_ligand_site_concentrations_mol_m3)),
    )
    for _iteration_index in range(NEWTON_MAX_ITERATIONS):
        mass_action_state = _ligand_coupled_mass_action_state_from_unknowns(
            current_unknowns_mol_m3,
            speciation_problem,
        )
        residual_vector_mol_m3 = mass_action_state.residual_vector_mol_m3
        residual_norm_mol_m3 = float(np.max(np.abs(residual_vector_mol_m3)))
        if residual_norm_mol_m3 <= tolerance_mol_m3:
            return current_unknowns_mol_m3
        jacobian = _ligand_coupled_mass_action_jacobian(
            mass_action_state,
            speciation_problem,
        )
        newton_step = np.linalg.solve(jacobian, residual_vector_mol_m3)
        step_fraction = 1.0
        accepted_step = False
        while step_fraction >= NEWTON_MIN_STEP_FRACTION:
            trial_unknowns_mol_m3 = (
                current_unknowns_mol_m3 - step_fraction * newton_step
            )
            if np.all(trial_unknowns_mol_m3 > 0.0):
                trial_state = _ligand_coupled_mass_action_state_from_unknowns(
                    trial_unknowns_mol_m3,
                    speciation_problem,
                )
                trial_norm_mol_m3 = float(
                    np.max(np.abs(trial_state.residual_vector_mol_m3))
                )
                if trial_norm_mol_m3 < residual_norm_mol_m3:
                    current_unknowns_mol_m3 = trial_unknowns_mol_m3
                    accepted_step = True
                    break
            step_fraction *= NEWTON_LINE_SEARCH_BACKOFF
        if not accepted_step:
            raise ValueError(
                "ligand-coupled projected speciation solve failed to reduce residual"
            )
    raise ValueError(
        "ligand-coupled projected speciation solve exceeded iteration limit"
    )


def _ligand_coupled_initial_unknowns_mol_m3(
    initial_free_component_concentrations_mol_m3: np.ndarray,
    components: tuple[IonComponent, ...],
    total_ligand_site_concentrations_mol_m3: np.ndarray,
    ligand_association_constants_m3_mol: np.ndarray,
) -> np.ndarray:
    positive_component_indices = tuple(
        component_index
        for component_index, component in enumerate(components)
        if component.charge_number > 0
    )
    if not positive_component_indices:
        return np.concatenate(
            (
                initial_free_component_concentrations_mol_m3,
                total_ligand_site_concentrations_mol_m3,
            )
        )
    positive_component_totals_mol_m3 = np.asarray(
        [
            initial_free_component_concentrations_mol_m3[component_index]
            for component_index in positive_component_indices
        ],
        dtype=float,
    )
    free_positive_components_mol_m3, free_ligands_mol_m3 = (
        _solve_neutral_ligand_free_concentrations_mol_m3(
            positive_component_totals_mol_m3,
            total_ligand_site_concentrations_mol_m3,
            ligand_association_constants_m3_mol,
        )
    )
    initial_unknown_components_mol_m3 = (
        initial_free_component_concentrations_mol_m3.copy()
    )
    for positive_component_position, component_index in enumerate(
        positive_component_indices
    ):
        initial_unknown_components_mol_m3[component_index] = (
            free_positive_components_mol_m3[positive_component_position]
        )
    return np.concatenate(
        (initial_unknown_components_mol_m3, free_ligands_mol_m3)
    )


def _ligand_coupled_mass_action_state_from_unknowns(
    unknowns_mol_m3: np.ndarray,
    speciation_problem: _LigandCoupledSpeciationProblem,
) -> _LigandCoupledMassActionState:
    component_count = len(speciation_problem.components)
    free_component_concentrations_mol_m3 = unknowns_mol_m3[:component_count]
    free_ligand_concentrations_mol_m3 = unknowns_mol_m3[component_count:]
    return _ligand_coupled_mass_action_state(
        free_component_concentrations_mol_m3,
        free_ligand_concentrations_mol_m3,
        speciation_problem,
    )


def _ligand_coupled_mass_action_state(
    free_component_concentrations_mol_m3: np.ndarray,
    free_ligand_concentrations_mol_m3: np.ndarray,
    speciation_problem: _LigandCoupledSpeciationProblem,
) -> _LigandCoupledMassActionState:
    cluster_concentrations_mol_m3 = _cluster_concentrations_array(
        speciation_problem.components,
        speciation_problem.cluster_templates,
        free_component_concentrations_mol_m3,
        speciation_problem.solvent_environment,
        speciation_problem.primitive_parameters,
    )
    component_count = len(speciation_problem.components)
    ligand_count = len(free_ligand_concentrations_mol_m3)
    cation_ligand_concentrations_mol_m3 = np.zeros(
        (component_count, ligand_count),
        dtype=float,
    )
    for component_index, component in enumerate(speciation_problem.components):
        if component.charge_number <= 0:
            continue
        cation_ligand_concentrations_mol_m3[component_index, :] = (
            free_component_concentrations_mol_m3[component_index]
            * free_ligand_concentrations_mol_m3
            * speciation_problem.ligand_association_constants_m3_mol
        )
    cation_ligand_anion_concentrations_mol_m3 = np.zeros(
        (len(speciation_problem.cluster_templates), ligand_count),
        dtype=float,
    )
    for cluster_index, cluster_template in enumerate(
        speciation_problem.cluster_templates
    ):
        if cluster_template.cluster_kind != SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
            continue
        cation_ligand_anion_concentrations_mol_m3[cluster_index, :] = (
            cluster_concentrations_mol_m3[cluster_index]
            * free_ligand_concentrations_mol_m3
            * speciation_problem.ligand_association_constants_m3_mol
        )
    component_residuals_mol_m3 = free_component_concentrations_mol_m3.copy()
    for cluster_index, cluster_template in enumerate(
        speciation_problem.cluster_templates
    ):
        cluster_concentration_mol_m3 = cluster_concentrations_mol_m3[cluster_index]
        ligand_bound_cluster_concentration_mol_m3 = float(
            np.sum(cation_ligand_anion_concentrations_mol_m3[cluster_index, :])
        )
        for component_index, component in enumerate(speciation_problem.components):
            if component.species_name not in cluster_template.stoichiometry:
                continue
            stoichiometric_count = cluster_template.stoichiometry[
                component.species_name
            ]
            component_residuals_mol_m3[component_index] += stoichiometric_count * (
                cluster_concentration_mol_m3 + ligand_bound_cluster_concentration_mol_m3
            )
    component_residuals_mol_m3 += np.sum(
        cation_ligand_concentrations_mol_m3,
        axis=1,
    )
    component_residuals_mol_m3 -= (
        speciation_problem.total_component_concentrations_mol_m3
    )
    ligand_residuals_mol_m3 = (
        free_ligand_concentrations_mol_m3
        + np.sum(cation_ligand_concentrations_mol_m3, axis=0)
        + np.sum(cation_ligand_anion_concentrations_mol_m3, axis=0)
        - speciation_problem.total_ligand_site_concentrations_mol_m3
    )
    return _LigandCoupledMassActionState(
        free_component_concentrations_mol_m3=free_component_concentrations_mol_m3,
        free_ligand_concentrations_mol_m3=free_ligand_concentrations_mol_m3,
        cluster_concentrations_mol_m3=cluster_concentrations_mol_m3,
        cation_ligand_concentrations_mol_m3=cation_ligand_concentrations_mol_m3,
        cation_ligand_anion_concentrations_mol_m3=(
            cation_ligand_anion_concentrations_mol_m3
        ),
        residual_vector_mol_m3=np.concatenate(
            (component_residuals_mol_m3, ligand_residuals_mol_m3)
        ),
    )


def _ligand_coupled_mass_action_jacobian(
    mass_action_state: _LigandCoupledMassActionState,
    speciation_problem: _LigandCoupledSpeciationProblem,
) -> np.ndarray:
    component_count = len(speciation_problem.components)
    ligand_count = len(
        speciation_problem.total_ligand_site_concentrations_mol_m3
    )
    unknown_count = component_count + ligand_count
    jacobian = np.zeros(
        (unknown_count, unknown_count),
        dtype=float,
    )
    for component_index in range(component_count):
        jacobian[component_index, component_index] += 1.0
    for ligand_index in range(ligand_count):
        ligand_row_index = component_count + ligand_index
        jacobian[ligand_row_index, ligand_row_index] += 1.0
    for component_index, component in enumerate(speciation_problem.components):
        if component.charge_number <= 0:
            continue
        free_component_concentration_mol_m3 = _positive_float(
            mass_action_state.free_component_concentrations_mol_m3[component_index],
            f"{component.species_name}.free_component_concentration_mol_m3",
        )
        for ligand_index in range(ligand_count):
            ligand_column_index = component_count + ligand_index
            ligand_row_index = component_count + ligand_index
            ligand_association_constant_m3_mol = (
                speciation_problem.ligand_association_constants_m3_mol[ligand_index]
            )
            free_ligand_concentration_mol_m3 = _positive_float(
                mass_action_state.free_ligand_concentrations_mol_m3[ligand_index],
                "free_ligand_concentration_mol_m3",
            )
            binary_concentration_mol_m3 = (
                mass_action_state.cation_ligand_concentrations_mol_m3[
                    component_index,
                    ligand_index,
                ]
            )
            binary_derivative_with_component = (
                binary_concentration_mol_m3 / free_component_concentration_mol_m3
            )
            binary_derivative_with_ligand = (
                binary_concentration_mol_m3 / free_ligand_concentration_mol_m3
            )
            jacobian[component_index, component_index] += (
                binary_derivative_with_component
            )
            jacobian[component_index, ligand_column_index] += (
                binary_derivative_with_ligand
            )
            jacobian[ligand_row_index, component_index] += (
                binary_derivative_with_component
            )
            jacobian[ligand_row_index, ligand_column_index] += (
                binary_derivative_with_ligand
            )
            if ligand_association_constant_m3_mol <= 0.0:
                raise ValueError("ligand association constant must be positive")
    for cluster_index, cluster_template in enumerate(
        speciation_problem.cluster_templates
    ):
        cluster_concentration_mol_m3 = (
            mass_action_state.cluster_concentrations_mol_m3[cluster_index]
        )
        for component_column_index, component in enumerate(
            speciation_problem.components
        ):
            if component.species_name not in cluster_template.stoichiometry:
                continue
            free_component_concentration_mol_m3 = _positive_float(
                mass_action_state.free_component_concentrations_mol_m3[
                    component_column_index
                ],
                f"{component.species_name}.free_component_concentration_mol_m3",
            )
            cluster_derivative_with_component = (
                cluster_concentration_mol_m3
                * cluster_template.stoichiometry[component.species_name]
                / free_component_concentration_mol_m3
            )
            ligand_bound_derivative_sum = 0.0
            for ligand_index in range(ligand_count):
                ligand_bound_derivative_sum += (
                    mass_action_state.cation_ligand_anion_concentrations_mol_m3[
                        cluster_index,
                        ligand_index,
                    ]
                    * cluster_template.stoichiometry[component.species_name]
                    / free_component_concentration_mol_m3
                )
            for component_row_index, row_component in enumerate(
                speciation_problem.components
            ):
                if row_component.species_name not in cluster_template.stoichiometry:
                    continue
                row_stoichiometric_count = cluster_template.stoichiometry[
                    row_component.species_name
                ]
                jacobian[component_row_index, component_column_index] += (
                    row_stoichiometric_count
                    * (
                        cluster_derivative_with_component
                        + ligand_bound_derivative_sum
                    )
                )
            for ligand_index in range(ligand_count):
                ligand_row_index = component_count + ligand_index
                jacobian[ligand_row_index, component_column_index] += (
                    mass_action_state.cation_ligand_anion_concentrations_mol_m3[
                        cluster_index,
                        ligand_index,
                    ]
                    * cluster_template.stoichiometry[component.species_name]
                    / free_component_concentration_mol_m3
                )
        for ligand_index in range(ligand_count):
            ligand_column_index = component_count + ligand_index
            ligand_row_index = component_count + ligand_index
            free_ligand_concentration_mol_m3 = _positive_float(
                mass_action_state.free_ligand_concentrations_mol_m3[ligand_index],
                "free_ligand_concentration_mol_m3",
            )
            ternary_derivative_with_ligand = (
                mass_action_state.cation_ligand_anion_concentrations_mol_m3[
                    cluster_index,
                    ligand_index,
                ]
                / free_ligand_concentration_mol_m3
            )
            for component_row_index, row_component in enumerate(
                speciation_problem.components
            ):
                if row_component.species_name not in cluster_template.stoichiometry:
                    continue
                jacobian[component_row_index, ligand_column_index] += (
                    cluster_template.stoichiometry[row_component.species_name]
                    * ternary_derivative_with_ligand
                )
            jacobian[ligand_row_index, ligand_column_index] += (
                ternary_derivative_with_ligand
            )
    return jacobian


def _accumulate_concentration_mol_m3(
    concentrations_mol_m3: dict[str, float],
    label: str,
    increment_mol_m3: float,
) -> None:
    if label in concentrations_mol_m3:
        concentrations_mol_m3[label] += increment_mol_m3
    else:
        concentrations_mol_m3[label] = increment_mol_m3


def _positive_component_species_name(
    cluster_template: ClusterStateTemplate,
    component_by_name: Mapping[str, IonComponent],
) -> str:
    positive_species_names = tuple(
        species_name
        for species_name in cluster_template.stoichiometry
        if component_by_name[species_name].charge_number > 0
    )
    if not positive_species_names:
        raise ValueError(f"{cluster_template.label} has no positive component")
    if len(positive_species_names) > 1:
        raise ValueError(
            f"{cluster_template.label} has multiple positive component species"
        )
    return positive_species_names[0]


def _neutral_ligand_features(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> tuple[_NeutralLigandFeature, ...]:
    ligand_features: list[_NeutralLigandFeature] = []
    for additive_index, (species_name, weight_fraction) in enumerate(
        recipe.additives.items()
    ):
        descriptor = descriptors[species_name]
        if descriptor.role != ROLE_ADDITIVE:
            raise ValueError(f"recipe additive {species_name} descriptor role mismatch")
        coordination_site_count = _coordination_site_count(descriptor)
        coordination_affinity_J_mol = _nonnegative_float(
            descriptor.coordination_affinity_J_mol,
            f"{species_name}.coordination_affinity_J_mol",
        )
        if coordination_site_count == 0.0 or coordination_affinity_J_mol == 0.0:
            continue
        additive_concentration_mol_m3 = _positive_float(
            _positive_float(weight_fraction, f"{species_name}.weight_fraction")
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol,
            f"{species_name}.neutral_additive_concentration_mol_m3",
        )
        site_concentration_mol_m3 = (
            additive_concentration_mol_m3 * coordination_site_count
        )
        ligand_features.append(
            _NeutralLigandFeature(
                feature_label=_neutral_ligand_feature_label(
                    descriptor,
                    additive_index,
                    coordination_site_count,
                ),
                site_concentration_mol_m3=_positive_float(
                    site_concentration_mol_m3,
                    f"{species_name}.neutral_ligand_site_concentration_mol_m3",
                ),
                coordination_affinity_J_mol=coordination_affinity_J_mol,
                molecular_volume_A3=_positive_float(
                    descriptor.molecular_volume_A3,
                    f"{species_name}.molecular_volume_A3",
                ),
                solvation_support=_additive_solvation_support(descriptor),
                coordination_site_count=coordination_site_count,
            )
        )
    return tuple(ligand_features)


def _neutral_ligand_feature_label(
    descriptor: MolecularSpeciesDescriptor,
    additive_index: int,
    coordination_site_count: float,
) -> str:
    return (
        "neutral_ligand_site:"
        f"feature_index={additive_index}:"
        f"sites={_feature_value_token(coordination_site_count)}:"
        f"affinity_J_mol={_feature_value_token(descriptor.coordination_affinity_J_mol)}:"
        f"volume_A3={_feature_value_token(descriptor.molecular_volume_A3)}:"
        f"donor={_feature_value_token(descriptor.donor_number)}:"
        f"acceptor={_feature_value_token(descriptor.acceptor_number)}:"
        f"polarizability_A3={_feature_value_token(descriptor.polarizability_A3)}:"
        f"ligand_asym={_feature_value_token(descriptor.ligand_field_asymmetry)}"
    )


def _neutral_ligand_binding_substrates(
    speciation: GenericSpeciationResult,
) -> tuple[_NeutralLigandBindingSubstrate, ...]:
    substrates: list[_NeutralLigandBindingSubstrate] = []
    component_by_name = {
        component.species_name: component for component in speciation.components
    }
    for component in speciation.components:
        if component.charge_number <= 0:
            continue
        total_concentration_mol_m3 = speciation.free_component_concentrations_mol_m3[
            component.species_name
        ]
        if total_concentration_mol_m3 == 0.0:
            continue
        substrates.append(
            _NeutralLigandBindingSubstrate(
                substrate_label=_cation_ligand_substrate_label(component),
                substrate_kind=NEUTRAL_LIGAND_BINDING_SUBSTRATE_FREE_CATION,
                source_species_name=component.species_name,
                source_cluster_label="",
                total_concentration_mol_m3=_positive_float(
                    total_concentration_mol_m3,
                    f"{component.species_name}.free_ligand_substrate_mol_m3",
                ),
                cation_feature_label=_cation_ligand_substrate_label(component),
            )
        )
    for cluster_template in speciation.cluster_templates:
        if cluster_template.cluster_kind != SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
            continue
        positive_components = tuple(
            component_by_name[species_name]
            for species_name in cluster_template.stoichiometry
            if component_by_name[species_name].charge_number > 0
        )
        if not positive_components:
            raise ValueError(
                f"{cluster_template.label} solvent-separated pair has no cation"
            )
        cluster_concentration_mol_m3 = speciation.cluster_concentrations_mol_m3[
            cluster_template.label
        ]
        if cluster_concentration_mol_m3 == 0.0:
            continue
        substrates.append(
            _NeutralLigandBindingSubstrate(
                substrate_label=_ssip_ligand_substrate_label(
                    cluster_template,
                    component_by_name,
                ),
                substrate_kind=(
                    NEUTRAL_LIGAND_BINDING_SUBSTRATE_SOLVENT_SEPARATED_PAIR
                ),
                source_species_name=positive_components[0].species_name,
                source_cluster_label=cluster_template.label,
                total_concentration_mol_m3=_positive_float(
                    cluster_concentration_mol_m3,
                    f"{cluster_template.label}.ligand_substrate_mol_m3",
                ),
                cation_feature_label=_cation_ligand_substrate_label(
                    positive_components[0]
                ),
            )
        )
    return tuple(substrates)


def _cation_ligand_substrate_label(component: IonComponent) -> str:
    return (
        "cation_ligand_substrate:"
        f"z={component.charge_number}:"
        f"{_molecular_descriptor_feature_key(component.descriptor, component.charge_number)}"
    )


def _ssip_ligand_substrate_label(
    cluster_template: ClusterStateTemplate,
    component_by_name: Mapping[str, IonComponent],
) -> str:
    feature_tokens: list[str] = []
    for species_name in sorted(cluster_template.stoichiometry):
        component = component_by_name[species_name]
        feature_tokens.append(
            _molecular_descriptor_feature_key(
                component.descriptor,
                component.charge_number,
            )
        )
    return (
        "ssip_ligand_substrate:"
        f"basin={SOLVENT_SEPARATED_PAIR_CLUSTER_KIND}:" + ":".join(feature_tokens)
    )


def _neutral_ligand_motif_label(
    binding_substrate: _NeutralLigandBindingSubstrate,
    ligand_feature: _NeutralLigandFeature,
) -> str:
    if binding_substrate.substrate_kind == NEUTRAL_LIGAND_BINDING_SUBSTRATE_FREE_CATION:
        motif_kind = "cation_ligand_shell"
    elif (
        binding_substrate.substrate_kind
        == NEUTRAL_LIGAND_BINDING_SUBSTRATE_SOLVENT_SEPARATED_PAIR
    ):
        motif_kind = ADDITIVE_SEPARATED_PAIR_CLUSTER_KIND
    else:
        raise ValueError(
            "unknown neutral ligand motif substrate kind "
            f"{binding_substrate.substrate_kind}"
        )
    return (
        "feature_keyed:"
        f"{motif_kind}:"
        f"{binding_substrate.substrate_label}:"
        f"{ligand_feature.feature_label}"
    )


def _neutral_ligand_bound_concentrations_mol_m3(
    binding_substrates: tuple[_NeutralLigandBindingSubstrate, ...],
    ligand_features: tuple[_NeutralLigandFeature, ...],
    temperature_K: float,
) -> np.ndarray:
    substrate_totals_mol_m3 = np.asarray(
        [
            _positive_float(
                binding_substrate.total_concentration_mol_m3,
                f"{binding_substrate.substrate_label}.total_concentration_mol_m3",
            )
            for binding_substrate in binding_substrates
        ],
        dtype=float,
    )
    ligand_totals_mol_m3 = np.asarray(
        [
            _positive_float(
                ligand_feature.site_concentration_mol_m3,
                f"{ligand_feature.feature_label}.site_concentration_mol_m3",
            )
            for ligand_feature in ligand_features
        ],
        dtype=float,
    )
    association_constants_m3_mol = _neutral_ligand_association_constants_m3_mol(
        ligand_features,
        temperature_K,
    )
    free_substrates_mol_m3, free_ligands_mol_m3 = (
        _solve_neutral_ligand_free_concentrations_mol_m3(
            substrate_totals_mol_m3,
            ligand_totals_mol_m3,
            association_constants_m3_mol,
        )
    )
    return _neutral_ligand_bound_matrix_mol_m3(
        free_substrates_mol_m3,
        free_ligands_mol_m3,
        association_constants_m3_mol,
    )


def _neutral_ligand_association_constants_m3_mol(
    ligand_features: tuple[_NeutralLigandFeature, ...],
    temperature_K: float,
) -> np.ndarray:
    thermal_energy_J_mol = R * _positive_float(
        temperature_K,
        "neutral_ligand_binding.temperature_K",
    )
    association_constants: list[float] = []
    for ligand_feature in ligand_features:
        affinity_over_RT = (
            _nonnegative_float(
                ligand_feature.coordination_affinity_J_mol,
                f"{ligand_feature.feature_label}.coordination_affinity_J_mol",
            )
            / thermal_energy_J_mol
        )
        association_constant_M_inv = math.expm1(affinity_over_RT)
        association_constants.append(
            _nonnegative_float(
                association_constant_M_inv / STANDARD_STATE_CONCENTRATION_MOL_M3,
                f"{ligand_feature.feature_label}.association_constant_m3_mol",
            )
        )
    return np.asarray(association_constants, dtype=float)


def _solve_neutral_ligand_free_concentrations_mol_m3(
    substrate_totals_mol_m3: np.ndarray,
    ligand_totals_mol_m3: np.ndarray,
    association_constants_m3_mol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    free_substrates_mol_m3 = substrate_totals_mol_m3.copy()
    free_ligands_mol_m3 = ligand_totals_mol_m3.copy()
    tolerance_mol_m3 = NEUTRAL_LIGAND_MASS_BALANCE_TOLERANCE_FACTOR * max(
        1.0,
        float(np.max(substrate_totals_mol_m3)),
        float(np.max(ligand_totals_mol_m3)),
    )
    for _iteration_index in range(NEWTON_MAX_ITERATIONS):
        residual_vector = _neutral_ligand_binding_residual_mol_m3(
            free_substrates_mol_m3,
            free_ligands_mol_m3,
            substrate_totals_mol_m3,
            ligand_totals_mol_m3,
            association_constants_m3_mol,
        )
        residual_norm = float(np.max(np.abs(residual_vector)))
        if residual_norm <= tolerance_mol_m3:
            return free_substrates_mol_m3, free_ligands_mol_m3
        jacobian = _neutral_ligand_binding_jacobian(
            free_substrates_mol_m3,
            free_ligands_mol_m3,
            substrate_totals_mol_m3,
            ligand_totals_mol_m3,
            association_constants_m3_mol,
        )
        newton_step = np.linalg.solve(jacobian, residual_vector)
        current_unknowns = np.concatenate((free_substrates_mol_m3, free_ligands_mol_m3))
        step_fraction = 1.0
        accepted_step = False
        while step_fraction >= NEWTON_MIN_STEP_FRACTION:
            trial_unknowns = current_unknowns - step_fraction * newton_step
            if np.all(trial_unknowns > 0.0):
                substrate_count = substrate_totals_mol_m3.shape[0]
                trial_free_substrates_mol_m3 = trial_unknowns[:substrate_count]
                trial_free_ligands_mol_m3 = trial_unknowns[substrate_count:]
                trial_residual = _neutral_ligand_binding_residual_mol_m3(
                    trial_free_substrates_mol_m3,
                    trial_free_ligands_mol_m3,
                    substrate_totals_mol_m3,
                    ligand_totals_mol_m3,
                    association_constants_m3_mol,
                )
                trial_norm = float(np.max(np.abs(trial_residual)))
                if trial_norm < residual_norm:
                    free_substrates_mol_m3 = trial_free_substrates_mol_m3
                    free_ligands_mol_m3 = trial_free_ligands_mol_m3
                    accepted_step = True
                    break
            step_fraction *= NEWTON_LINE_SEARCH_BACKOFF
        if not accepted_step:
            raise ValueError(
                "neutral-ligand mass-action solve failed to reduce residual"
            )
    raise ValueError("neutral-ligand mass-action solve exceeded iteration limit")


def _neutral_ligand_binding_residual_mol_m3(
    free_substrates_mol_m3: np.ndarray,
    free_ligands_mol_m3: np.ndarray,
    substrate_totals_mol_m3: np.ndarray,
    ligand_totals_mol_m3: np.ndarray,
    association_constants_m3_mol: np.ndarray,
) -> np.ndarray:
    bound_matrix_mol_m3 = _neutral_ligand_bound_matrix_mol_m3(
        free_substrates_mol_m3,
        free_ligands_mol_m3,
        association_constants_m3_mol,
    )
    substrate_residuals = (
        free_substrates_mol_m3
        + np.sum(bound_matrix_mol_m3, axis=1)
        - substrate_totals_mol_m3
    )
    ligand_residuals = (
        free_ligands_mol_m3 + np.sum(bound_matrix_mol_m3, axis=0) - ligand_totals_mol_m3
    )
    return np.concatenate((substrate_residuals, ligand_residuals))


def _neutral_ligand_bound_matrix_mol_m3(
    free_substrates_mol_m3: np.ndarray,
    free_ligands_mol_m3: np.ndarray,
    association_constants_m3_mol: np.ndarray,
) -> np.ndarray:
    return (
        free_substrates_mol_m3[:, np.newaxis]
        * free_ligands_mol_m3[np.newaxis, :]
        * association_constants_m3_mol[np.newaxis, :]
    )


def _neutral_ligand_binding_jacobian(
    free_substrates_mol_m3: np.ndarray,
    free_ligands_mol_m3: np.ndarray,
    substrate_totals_mol_m3: np.ndarray,
    ligand_totals_mol_m3: np.ndarray,
    association_constants_m3_mol: np.ndarray,
) -> np.ndarray:
    substrate_count = free_substrates_mol_m3.shape[0]
    ligand_count = free_ligands_mol_m3.shape[0]
    if substrate_totals_mol_m3.shape[0] != substrate_count:
        raise ValueError("neutral-ligand substrate total shape mismatch")
    if ligand_totals_mol_m3.shape[0] != ligand_count:
        raise ValueError("neutral-ligand ligand total shape mismatch")
    jacobian = np.zeros(
        (substrate_count + ligand_count, substrate_count + ligand_count),
        dtype=float,
    )
    for substrate_index in range(substrate_count):
        jacobian[substrate_index, substrate_index] = 1.0 + float(
            np.sum(association_constants_m3_mol * free_ligands_mol_m3)
        )
        for ligand_index in range(ligand_count):
            jacobian[
                substrate_index,
                substrate_count + ligand_index,
            ] = (
                association_constants_m3_mol[ligand_index]
                * free_substrates_mol_m3[substrate_index]
            )
            jacobian[
                substrate_count + ligand_index,
                substrate_index,
            ] = (
                association_constants_m3_mol[ligand_index]
                * free_ligands_mol_m3[ligand_index]
            )
    for ligand_index in range(ligand_count):
        jacobian[
            substrate_count + ligand_index,
            substrate_count + ligand_index,
        ] = 1.0 + float(
            association_constants_m3_mol[ligand_index] * np.sum(free_substrates_mol_m3)
        )
    return jacobian


def _coordination_site_count(descriptor: MolecularSpeciesDescriptor) -> float:
    explicit_site_count = len(descriptor.coordination_sites)
    if explicit_site_count > 0:
        return float(explicit_site_count)
    return float(
        _nonnegative_float(
            float(descriptor.hbond_acceptor_count),
            f"{descriptor.name}.hbond_acceptor_count",
        )
        + _nonnegative_float(
            float(descriptor.hbond_donor_count),
            f"{descriptor.name}.hbond_donor_count",
        )
    )


def _ion_components(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> tuple[IonComponent, ...]:
    components: list[IonComponent] = []
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        if descriptor.role != ROLE_CATION:
            raise ValueError(f"recipe cation {species_name} descriptor role mismatch")
        components.append(
            IonComponent(
                species_name=species_name,
                charge_number=descriptor.charge_number,
                analytical_concentration_M=_positive_float(
                    concentration_M,
                    f"{species_name}.concentration_M",
                ),
                descriptor=descriptor,
            )
        )
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        if descriptor.role != ROLE_ANION:
            raise ValueError(f"recipe anion {species_name} descriptor role mismatch")
        components.append(
            IonComponent(
                species_name=species_name,
                charge_number=descriptor.charge_number,
                analytical_concentration_M=_positive_float(
                    concentration_M,
                    f"{species_name}.concentration_M",
                ),
                descriptor=descriptor,
            )
        )
    return tuple(components)


def _validate_ionic_charge_balance(
    components: tuple[IonComponent, ...],
) -> None:
    net_charge_concentration_M = math.fsum(
        component.charge_number * component.analytical_concentration_M
        for component in components
    )
    charge_scale_M = math.fsum(
        abs(component.charge_number) * component.analytical_concentration_M
        for component in components
    )
    tolerance_M = MASS_BALANCE_TOLERANCE_FACTOR * max(1.0, charge_scale_M)
    if abs(net_charge_concentration_M) > tolerance_M:
        raise ValueError(
            "ionic recipe must be charge neutral; "
            f"net analytical charge concentration is {net_charge_concentration_M} M"
        )


def _apply_cluster_standard_free_energy_shifts(
    cluster_templates: tuple[ClusterStateTemplate, ...],
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
    temperature_K: float,
) -> tuple[ClusterStateTemplate, ...]:
    if not diagnostic_cluster_standard_free_energy_shift_over_RT_by_label:
        return cluster_templates
    known_cluster_labels = {
        cluster_template.label for cluster_template in cluster_templates
    }
    unknown_cluster_labels = tuple(
        sorted(
            cluster_label
            for cluster_label in diagnostic_cluster_standard_free_energy_shift_over_RT_by_label
            if cluster_label not in known_cluster_labels
        )
    )
    if unknown_cluster_labels:
        raise ValueError(
            "unknown cluster standard-free-energy shift labels "
            f"{unknown_cluster_labels}"
        )
    shifted_templates: list[ClusterStateTemplate] = []
    for cluster_template in cluster_templates:
        if (
            cluster_template.label
            in diagnostic_cluster_standard_free_energy_shift_over_RT_by_label
        ):
            raw_shift_over_RT = (
                diagnostic_cluster_standard_free_energy_shift_over_RT_by_label[
                    cluster_template.label
                ]
            )
        else:
            raw_shift_over_RT = 0.0
        shift_over_RT = _finite_float(
            raw_shift_over_RT,
            f"{cluster_template.label}.standard_free_energy_shift_over_RT",
        )
        if shift_over_RT == 0.0:
            shifted_templates.append(cluster_template)
            continue
        shift_J_mol = R * temperature_K * shift_over_RT
        shifted_templates.append(
            replace(
                cluster_template,
                standard_free_energy_J_mol=(
                    cluster_template.standard_free_energy_J_mol + shift_J_mol
                ),
                standard_state_correction_J_mol=(
                    cluster_template.standard_state_correction_J_mol + shift_J_mol
                ),
                activity_reference_J_mol=(cluster_template.activity_reference_J_mol),
            )
        )
    return tuple(shifted_templates)


def _projected_transport_states_from_mass_balance(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    speciation: GenericSpeciationResult,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> tuple[ProjectedTransportState, ...]:
    states: list[ProjectedTransportState] = []
    component_descriptor_by_name = {
        component.species_name: component.descriptor
        for component in speciation.components
    }
    mixture_descriptor_state = _molecular_mixture_descriptor_state(
        recipe,
        descriptors,
        solvent_environment,
        options,
    )
    charge_density_reference_A_inv3 = _charge_density_reference_A_inv3(
        speciation,
        component_descriptor_by_name,
        options,
    )
    transport_context = _TransportCenterConstructionContext(
        component_descriptor_by_name=component_descriptor_by_name,
        mixture_descriptor_state=mixture_descriptor_state,
        charge_density_reference_A_inv3=charge_density_reference_A_inv3,
        solvent_environment=solvent_environment,
        options=options,
    )
    concentration_resolution_mol_m3 = _transport_state_concentration_resolution_mol_m3(
        speciation
    )
    for component in speciation.components:
        descriptor = component.descriptor
        concentration_mol_m3 = _free_component_transport_concentration_mol_m3(
            component,
            speciation,
        )
        if concentration_mol_m3 <= concentration_resolution_mol_m3:
            continue
        states.append(
            _transport_state_from_descriptor(
                label=_transport_center_feature_label(
                    partition_role=TRANSPORT_ROLE_FREE_ION_CENTER,
                    parent_cluster_kind=TRANSPORT_ROLE_FREE_ION_CENTER,
                    center_index=0,
                    descriptor=descriptor,
                    center_charge_number=component.charge_number,
                ),
                parent_cluster_label=f"free:{component.species_name}",
                parent_cluster_kind=TRANSPORT_ROLE_FREE_ION_CENTER,
                concentration_mol_m3=concentration_mol_m3,
                center_species_name=component.species_name,
                center_charge_number=component.charge_number,
                center_index=0,
                descriptor=descriptor,
                hydrodynamic_radius_scale=_free_ion_hydrodynamic_radius_scale(
                    component,
                    options,
                ),
                transport_context=transport_context,
                transport_role=TRANSPORT_ROLE_FREE_ION_CENTER,
            )
        )
    states.extend(
        _cation_ligand_shell_projected_states(
            speciation,
            transport_context,
            concentration_resolution_mol_m3,
        )
    )
    for cluster_template in speciation.cluster_templates:
        concentration_mol_m3 = speciation.cluster_concentrations_mol_m3[
            cluster_template.label
        ]
        if concentration_mol_m3 <= concentration_resolution_mol_m3:
            continue
        states.extend(
            _cluster_transport_centers(
                cluster_template,
                concentration_mol_m3,
                transport_context,
            )
        )
    states = list(
        _projected_kinetic_population_projection(
            tuple(states),
            speciation,
            transport_context,
        )
    )
    states.extend(
        _neutral_transport_states(
            recipe,
            descriptors,
            transport_context,
        )
    )
    if not states:
        raise ValueError("molecular transport state construction produced no states")
    return _projected_transport_states_with_unique_feature_labels(tuple(states))


def _free_component_transport_concentration_mol_m3(
    component: IonComponent,
    speciation: GenericSpeciationResult,
) -> float:
    free_concentration_mol_m3 = _nonnegative_float(
        speciation.free_component_concentrations_mol_m3[component.species_name],
        f"{component.species_name}.free_component_concentration_mol_m3",
    )
    ligand_bound_concentration_mol_m3 = 0.0
    if (
        component.species_name
        in speciation.free_component_ligand_bound_concentrations_mol_m3
    ):
        ligand_bound_concentration_mol_m3 = _nonnegative_float(
            speciation.free_component_ligand_bound_concentrations_mol_m3[
                component.species_name
            ],
            f"{component.species_name}.free_component_ligand_bound_mol_m3",
        )
    unbound_concentration_mol_m3 = (
        free_concentration_mol_m3 - ligand_bound_concentration_mol_m3
    )
    tolerance_mol_m3 = MASS_BALANCE_TOLERANCE_FACTOR * max(
        1.0,
        free_concentration_mol_m3,
    )
    if unbound_concentration_mol_m3 < -tolerance_mol_m3:
        raise ValueError(
            f"{component.species_name}.ligand_bound_concentration exceeds free pool"
        )
    if abs(unbound_concentration_mol_m3) <= tolerance_mol_m3:
        return 0.0
    return _positive_float(
        unbound_concentration_mol_m3,
        f"{component.species_name}.unbound_free_transport_concentration_mol_m3",
    )


def _cation_ligand_shell_projected_states(
    speciation: GenericSpeciationResult,
    transport_context: _TransportCenterConstructionContext,
    concentration_resolution_mol_m3: float,
) -> tuple[ProjectedTransportState, ...]:
    ligand_shell_states: list[ProjectedTransportState] = []
    for (
        motif_label,
        concentration_mol_m3,
    ) in speciation.cation_ligand_concentrations_mol_m3.items():
        if concentration_mol_m3 <= concentration_resolution_mol_m3:
            continue
        if motif_label not in speciation.cation_ligand_component_species_by_label:
            raise KeyError(
                f"cation_ligand_component_species_by_label missing {motif_label}"
            )
        component_species_name = speciation.cation_ligand_component_species_by_label[
            motif_label
        ]
        descriptor = transport_context.component_descriptor_by_name[
            component_species_name
        ]
        if descriptor.charge_number <= 0:
            raise ValueError(
                f"{motif_label} ligand-shell center must be a positive carrier"
            )
        ligand_shell_states.append(
            _transport_state_from_descriptor(
                label=motif_label,
                parent_cluster_label=motif_label,
                parent_cluster_kind=TRANSPORT_ROLE_LIGAND_SHELL_CENTER,
                concentration_mol_m3=concentration_mol_m3,
                center_species_name=component_species_name,
                center_charge_number=descriptor.charge_number,
                center_index=0,
                descriptor=descriptor,
                hydrodynamic_radius_scale=(
                    transport_context.options.primitive_parameters.hydrodynamic_radius_scale_positive_ion
                ),
                transport_context=transport_context,
                transport_role=TRANSPORT_ROLE_LIGAND_SHELL_CENTER,
            )
        )
    return tuple(ligand_shell_states)


def _projected_transport_states_with_unique_feature_labels(
    transport_states: tuple[ProjectedTransportState, ...],
) -> tuple[ProjectedTransportState, ...]:
    label_counts: dict[str, int] = {}
    for transport_state in transport_states:
        if transport_state.label in label_counts:
            label_counts[transport_state.label] += 1
        else:
            label_counts[transport_state.label] = 1
    label_occurrences: dict[str, int] = {}
    unique_transport_states: list[ProjectedTransportState] = []
    for transport_state in transport_states:
        label_count = label_counts[transport_state.label]
        if label_count == 1:
            unique_transport_states.append(transport_state)
            continue
        if transport_state.label in label_occurrences:
            occurrence_index = label_occurrences[transport_state.label]
        else:
            occurrence_index = 0
        label_occurrences[transport_state.label] = occurrence_index + 1
        unique_transport_states.append(
            replace(
                transport_state,
                label=(
                    f"{transport_state.label}:"
                    f"degenerate_feature_index={occurrence_index}"
                ),
            )
        )
    return tuple(unique_transport_states)


def _projected_kinetic_population_projection(
    transport_states: tuple[ProjectedTransportState, ...],
    speciation: GenericSpeciationResult,
    transport_context: _TransportCenterConstructionContext,
) -> tuple[ProjectedTransportState, ...]:
    if not any(
        transport_state.transport_role == TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER
        for transport_state in transport_states
    ):
        return transport_states
    projection_fraction_by_species = (
        _internal_polarization_projection_fraction_by_species(
            speciation,
            transport_context.options.primitive_parameters,
        )
    )
    projected_states: list[ProjectedTransportState] = []
    concentration_resolution_mol_m3 = _transport_state_concentration_resolution_mol_m3(
        speciation
    )
    for transport_state in transport_states:
        if (
            transport_state.transport_role
            != TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER
            or transport_state.center_species_name not in projection_fraction_by_species
        ):
            projected_states.append(transport_state)
            continue
        projection_fraction = _positive_float(
            projection_fraction_by_species[transport_state.center_species_name],
            f"{transport_state.label}.internal_polarization_projection_fraction",
        )
        if not projection_fraction < 1.0:
            raise ValueError(
                f"{transport_state.label}.internal_polarization_projection_fraction "
                "must be less than one"
            )
        projected_concentration_mol_m3 = (
            transport_state.concentration_mol_m3 * projection_fraction
        )
        residual_concentration_mol_m3 = (
            transport_state.concentration_mol_m3 - projected_concentration_mol_m3
        )
        projected_states.append(
            replace(
                transport_state,
                concentration_mol_m3=_positive_float(
                    residual_concentration_mol_m3,
                    f"{transport_state.label}.residual_ssip_concentration_mol_m3",
                ),
            )
        )
        if projected_concentration_mol_m3 <= concentration_resolution_mol_m3:
            continue
        projected_states.append(
            _projected_internal_polarization_center(
                transport_state,
                projected_concentration_mol_m3,
            )
        )
    return tuple(projected_states)


def _internal_polarization_projection_fraction_by_species(
    speciation: GenericSpeciationResult,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> Mapping[str, float]:
    ionic_strength_driver = math.log1p(
        _analytical_ionic_strength_ratio(speciation.components)
    )
    projection_fraction_by_species: dict[str, float] = {}
    for component in speciation.components:
        total_concentration_mol_m3 = (
            _positive_float(
                component.analytical_concentration_M,
                f"{component.species_name}.analytical_concentration_M",
            )
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        free_concentration_mol_m3 = _nonnegative_float(
            speciation.free_component_concentrations_mol_m3[component.species_name],
            f"{component.species_name}.free_concentration_mol_m3",
        )
        associated_fraction = (
            total_concentration_mol_m3 - free_concentration_mol_m3
        ) / total_concentration_mol_m3
        if associated_fraction < 0.0:
            raise ValueError(
                f"{component.species_name}.associated_fraction is negative"
            )
        projection_logit = (
            primitive_parameters.internal_polarization_projection_offset
            + primitive_parameters.internal_polarization_projection_ionic_strength_slope
            * ionic_strength_driver
            + primitive_parameters.internal_polarization_projection_counterion_crowding_slope
            * associated_fraction
        )
        projection_fraction_by_species[component.species_name] = _sigmoid_fraction(
            projection_logit,
            f"{component.species_name}.internal_polarization_projection_logit",
        )
    return projection_fraction_by_species


def _sigmoid_fraction(logit_value: float, context: str) -> float:
    parsed_logit_value = _finite_float(logit_value, context)
    if not parsed_logit_value < 0.0:
        exponential_term = math.exp(-parsed_logit_value)
        fraction = 1.0 / (1.0 + exponential_term)
    else:
        exponential_term = math.exp(parsed_logit_value)
        fraction = exponential_term / (1.0 + exponential_term)
    if fraction <= 0.0 or not fraction < 1.0:
        raise ValueError(f"{context} produced a non-open projection fraction")
    return float(fraction)


def _projected_internal_polarization_center(
    source_transport_state: ProjectedTransportState,
    projected_concentration_mol_m3: float,
) -> ProjectedTransportState:
    return replace(
        source_transport_state,
        label=(
            "internal_polarization_projection:"
            f"{source_transport_state.label}:"
            f"center{source_transport_state.center_index}"
        ),
        concentration_mol_m3=_positive_float(
            projected_concentration_mol_m3,
            (
                f"{source_transport_state.label}"
                ".projected_internal_polarization_concentration_mol_m3"
            ),
        ),
        transport_role=TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
    )


def _transport_label_token(label: str) -> str:
    parsed_label = str(label)
    if not parsed_label:
        raise ValueError("transport_projection_label must be nonempty")
    return parsed_label.replace(":", "|")


def _transport_center_feature_label(
    partition_role: str,
    parent_cluster_kind: str,
    center_index: int,
    descriptor: MolecularSpeciesDescriptor,
    center_charge_number: int,
) -> str:
    if center_index < 0:
        raise ValueError("transport center index must be nonnegative")
    return (
        "feature_keyed_transport_center:"
        f"role={_feature_text_token(partition_role)}:"
        f"basin={_feature_text_token(parent_cluster_kind)}:"
        f"center={center_index}:"
        f"{_molecular_descriptor_feature_key(descriptor, center_charge_number)}"
    )


def _cluster_com_feature_label(
    cluster_template: ClusterStateTemplate,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
    transport_role: str,
) -> str:
    stoichiometric_feature_tokens = []
    for species_name, stoichiometric_count in sorted(
        cluster_template.stoichiometry.items()
    ):
        descriptor = component_descriptor_by_name[species_name]
        stoichiometric_feature_tokens.append(
            "count="
            f"{stoichiometric_count}:"
            f"{_molecular_descriptor_feature_key(descriptor, descriptor.charge_number)}"
        )
    return (
        "feature_keyed_transport_center:"
        f"role={_feature_text_token(transport_role)}:"
        f"basin={_feature_text_token(cluster_template.cluster_kind)}:"
        f"net_z={cluster_template.net_charge_number}:"
        + ":".join(stoichiometric_feature_tokens)
    )


def _molecular_descriptor_feature_key(
    descriptor: MolecularSpeciesDescriptor,
    center_charge_number: int,
) -> str:
    return (
        f"role={_feature_text_token(descriptor.role)}:"
        f"z={center_charge_number}:"
        f"rh_A={_feature_value_token(descriptor.hydrodynamic_radius_A)}:"
        f"rcloud_A={_feature_value_token(descriptor.charge_cloud_radius_A)}:"
        f"cavity_A={_feature_value_token(descriptor.cavity_radius_A)}:"
        f"volume_A3={_feature_value_token(descriptor.molecular_volume_A3)}:"
        f"donor={_feature_value_token(descriptor.donor_number)}:"
        f"acceptor={_feature_value_token(descriptor.acceptor_number)}:"
        f"polarizability_A3={_feature_value_token(descriptor.polarizability_A3)}:"
        f"ligand_asym={_feature_value_token(descriptor.ligand_field_asymmetry)}"
    )


def _feature_text_token(value: str) -> str:
    parsed_value = str(value)
    if parsed_value == "":
        raise ValueError("feature text token must be nonempty")
    safe_characters: list[str] = []
    for character in parsed_value:
        if character.isalnum() or character == "_":
            safe_characters.append(character)
        else:
            safe_characters.append("_")
    return "".join(safe_characters)


def _cluster_transport_centers(
    cluster_template: ClusterStateTemplate,
    concentration_mol_m3: float,
    transport_context: _TransportCenterConstructionContext,
) -> tuple[ProjectedTransportState, ...]:
    if cluster_template.cluster_kind == SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
        return _cluster_internal_transport_centers(
            cluster_template,
            concentration_mol_m3,
            transport_context,
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
        )
    if cluster_template.cluster_kind in (
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    ):
        return (
            _cluster_com_transport_center(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CLUSTER_COM_CENTER,
            ),
            *_cluster_internal_transport_centers(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
            ),
        )
    if cluster_template.cluster_kind == HIGHER_CHARGED_CLUSTER_KIND:
        return (
            _cluster_com_transport_center(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CLUSTER_COM_CENTER,
            ),
            *_cluster_internal_transport_centers(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
            ),
        )
    if cluster_template.cluster_kind in (
        CONTACT_PAIR_CLUSTER_KIND,
        NEUTRAL_CLUSTER_KIND,
    ):
        return (
            _cluster_com_transport_center(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CONTACT_PAIR_CENTER
                if cluster_template.cluster_kind == CONTACT_PAIR_CLUSTER_KIND
                else TRANSPORT_ROLE_CLUSTER_COM_CENTER,
            ),
            *_cluster_internal_transport_centers(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER,
            ),
        )
    raise ValueError(f"unknown cluster kind {cluster_template.cluster_kind}")


def _cluster_internal_transport_centers(
    cluster_template: ClusterStateTemplate,
    concentration_mol_m3: float,
    transport_context: _TransportCenterConstructionContext,
    transport_role: str,
) -> tuple[ProjectedTransportState, ...]:
    centers: list[ProjectedTransportState] = []
    for center_index, charged_center in enumerate(cluster_template.geometry):
        descriptor = transport_context.component_descriptor_by_name[
            charged_center.species_name
        ]
        centers.append(
            _transport_state_from_descriptor(
                label=_transport_center_feature_label(
                    partition_role=transport_role,
                    parent_cluster_kind=cluster_template.cluster_kind,
                    center_index=center_index,
                    descriptor=descriptor,
                    center_charge_number=charged_center.charge_number,
                ),
                parent_cluster_label=cluster_template.label,
                parent_cluster_kind=cluster_template.cluster_kind,
                concentration_mol_m3=concentration_mol_m3,
                center_species_name=charged_center.species_name,
                center_charge_number=charged_center.charge_number,
                center_index=center_index,
                descriptor=descriptor,
                hydrodynamic_radius_scale=_hydrodynamic_radius_scale_for_charge(
                    charged_center.charge_number,
                    transport_context.options,
                )
                * transport_context.options.primitive_parameters.hydrodynamic_radius_scale_cluster,
                transport_context=transport_context,
                transport_role=transport_role,
            )
        )
    return tuple(centers)


def _cluster_com_transport_center(
    cluster_template: ClusterStateTemplate,
    concentration_mol_m3: float,
    transport_context: _TransportCenterConstructionContext,
    transport_role: str,
) -> ProjectedTransportState:
    charge_cloud_radius_A = _cluster_charge_cloud_radius_A(
        cluster_template,
        transport_context.component_descriptor_by_name,
        transport_context.options,
    )
    hydrodynamic_radius_A = (
        transport_context.options.primitive_parameters.hydrodynamic_radius_scale_cluster
        * cluster_template.hydrodynamic_radius_A
    )
    ligand_field_asymmetry = _cluster_shape_factor(
        cluster_template,
        transport_context.component_descriptor_by_name,
    )
    base_diffusion_m2_s = _diffusion_m2_s(
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        shape_factor=ligand_field_asymmetry,
        intrinsic_dielectric_constant=_cluster_intrinsic_dielectric_constant(
            cluster_template,
            transport_context.component_descriptor_by_name,
        ),
        net_charge_number=cluster_template.net_charge_number,
        charge_cloud_radius_A=charge_cloud_radius_A,
        charge_density_reference_A_inv3=transport_context.charge_density_reference_A_inv3,
        mixture_descriptor_state=transport_context.mixture_descriptor_state,
        solvent_environment=transport_context.solvent_environment,
        options=transport_context.options,
    )
    local_obstruction_factor = _local_obstruction_factor(
        label=cluster_template.label,
        net_charge_number=cluster_template.net_charge_number,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        mixture_descriptor_state=transport_context.mixture_descriptor_state,
        options=transport_context.options,
        charge_density_reference_A_inv3=transport_context.charge_density_reference_A_inv3,
    )
    local_obstruction_diffusion_scale = _local_obstruction_diffusion_scale(
        local_obstruction_factor,
        cluster_template.label,
    )
    return _projected_transport_state_from_scalar_fields(
        label=_cluster_com_feature_label(
            cluster_template,
            transport_context.component_descriptor_by_name,
            transport_role,
        ),
        parent_cluster_label=cluster_template.label,
        parent_cluster_kind=cluster_template.cluster_kind,
        concentration_mol_m3=concentration_mol_m3,
        center_species_name=cluster_template.label,
        center_charge_number=cluster_template.net_charge_number,
        center_index=0,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        molecular_volume_A3=cluster_template.molecular_volume_A3,
        ligand_field_asymmetry=ligand_field_asymmetry,
        diffusion_m2_s=base_diffusion_m2_s * local_obstruction_diffusion_scale,
        local_obstruction_factor=local_obstruction_factor,
        local_obstruction_diffusion_scale=local_obstruction_diffusion_scale,
        transport_role=transport_role,
        temperature_K=transport_context.solvent_environment.temperature_K,
    )


def _hydrodynamic_radius_scale_for_charge(
    center_charge_number: int,
    options: MolecularMoriOptions,
) -> float:
    if center_charge_number > 0:
        return options.primitive_parameters.hydrodynamic_radius_scale_positive_ion
    if center_charge_number < 0:
        return options.primitive_parameters.hydrodynamic_radius_scale_negative_ion
    raise ValueError("transport center charge must be nonzero")


def _transport_state_concentration_resolution_mol_m3(
    speciation: GenericSpeciationResult,
) -> float:
    analytical_ion_concentration_mol_m3 = math.fsum(
        component.analytical_concentration_M * STANDARD_STATE_CONCENTRATION_MOL_M3
        for component in speciation.components
    )
    return TRANSPORT_STATE_CONCENTRATION_RESOLUTION_FACTOR * max(
        1.0,
        _nonnegative_float(
            analytical_ion_concentration_mol_m3,
            "analytical_ion_concentration_mol_m3",
        ),
    )


def _ion_atmosphere_diagnostics_for_projected_transport_states(
    transport_states: tuple[ProjectedTransportState, ...],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> _AtmosphereTransportStateResult:
    charged_states = tuple(
        state for state in transport_states if state.center_charge_number != 0
    )
    if not charged_states:
        return _AtmosphereTransportStateResult(
            transport_states=transport_states,
            diagnostics=MolecularIonAtmosphereDiagnostics(
                solver=ION_ATMOSPHERE_SOLVER_DIAGONAL,
                charged_carrier_count=0,
                kappa_inv_m=math.inf,
                ionic_strength_mol_m3=0.0,
                charge_cloud_form_factor_by_state={},
                friction_ratio_by_state={},
                zeta0_kg_s_by_state={},
                zeta_ep_kg_s_by_state={},
                zeta_rel_kg_s_by_state={},
                countercharge_relaxation_diffusivity_m2_s_by_state={},
            ),
        )
    carrier_concentrations_mol_m3 = {
        state.label: state.concentration_mol_m3 for state in charged_states
    }
    carrier_charges = {
        state.label: state.center_charge_number for state in charged_states
    }
    local_diffusivity_m2_s_by_carrier = {
        state.label: state.diffusion_m2_s for state in charged_states
    }
    hydrodynamic_radius_m_by_carrier = {
        state.label: state.hydrodynamic_radius_A * ANGSTROM_TO_M
        for state in charged_states
    }
    atmosphere_state = build_ion_atmosphere_state(
        IonAtmosphereInput(
            carrier_concentrations_mol_m3=carrier_concentrations_mol_m3,
            carrier_charges=carrier_charges,
            local_diffusivity_m2_s_by_carrier=local_diffusivity_m2_s_by_carrier,
            hydrodynamic_radius_m_by_carrier=hydrodynamic_radius_m_by_carrier,
            viscosity_Pa_s=solvent_environment.viscosity_cP * CP_TO_PA_S,
            relative_dielectric=solvent_environment.dielectric_constant,
            temperature_K=solvent_environment.temperature_K,
            solver=ION_ATMOSPHERE_SOLVER_DIAGONAL,
        )
    )
    countercharge_relaxation_diffusivity_by_state = (
        _countercharge_relaxation_diffusivity_by_state(
            charged_states,
            local_diffusivity_m2_s_by_carrier,
        )
    )
    zeta0_kg_s_by_state = {
        state_label: _positive_float(
            zeta0_kg_s,
            f"{state_label}.zeta0_kg_s",
        )
        for state_label, zeta0_kg_s in atmosphere_state.zeta0_by_carrier.items()
    }
    if math.isinf(atmosphere_state.kappa_inv_m):
        raise ValueError("charged molecular atmosphere kappa_inv_m must be finite")
    kappa_m_inv = 1.0 / _positive_float(
        atmosphere_state.kappa_inv_m,
        "molecular_atmosphere.kappa_inv_m",
    )
    charge_cloud_form_factor_by_state = {
        state.label: _charge_cloud_form_factor(
            state,
            kappa_m_inv,
        )
        for state in charged_states
    }
    zeta_ep_kg_s_by_state = {
        state_label: (
            options.primitive_parameters.atmosphere_ep_scale
            * charge_cloud_form_factor_by_state[state_label]
            * _nonnegative_float(
                zeta_ep_kg_s,
                f"{state_label}.zeta_ep_kg_s",
            )
        )
        for state_label, zeta_ep_kg_s in atmosphere_state.zeta_ep_by_carrier.items()
    }
    zeta_rel_kg_s_by_state = {
        state_label: (
            options.primitive_parameters.atmosphere_rel_scale
            * _debye_falkenhagen_charge_density_basis_multiplier(
                options.primitive_parameters
            )
            * charge_cloud_form_factor_by_state[state_label]
            * _nonnegative_float(
                zeta_rel_kg_s,
                f"{state_label}.zeta_rel_kg_s",
            )
        )
        for state_label, zeta_rel_kg_s in atmosphere_state.zeta_rel_by_carrier.items()
    }
    friction_ratio_by_state = {
        state.label: _atmosphere_friction_ratio(
            state.label,
            zeta0_kg_s_by_state,
            zeta_ep_kg_s_by_state,
            zeta_rel_kg_s_by_state,
        )
        for state in charged_states
    }
    return _AtmosphereTransportStateResult(
        transport_states=transport_states,
        diagnostics=MolecularIonAtmosphereDiagnostics(
            solver=atmosphere_state.solver,
            charged_carrier_count=len(charged_states),
            kappa_inv_m=atmosphere_state.kappa_inv_m,
            ionic_strength_mol_m3=atmosphere_state.ionic_strength_mol_m3,
            charge_cloud_form_factor_by_state=charge_cloud_form_factor_by_state,
            friction_ratio_by_state=friction_ratio_by_state,
            zeta0_kg_s_by_state=zeta0_kg_s_by_state,
            zeta_ep_kg_s_by_state=zeta_ep_kg_s_by_state,
            zeta_rel_kg_s_by_state=zeta_rel_kg_s_by_state,
            countercharge_relaxation_diffusivity_m2_s_by_state=(
                countercharge_relaxation_diffusivity_by_state
            ),
        ),
    )


def _debye_falkenhagen_charge_density_basis_multiplier(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    return 1.0 + _positive_float(
        primitive_parameters.cross_relaxation_scale,
        "debye_falkenhagen_charge_density_basis_multiplier",
    )


def _atmosphere_friction_ratio(
    state_label: str,
    zeta0_kg_s_by_state: Mapping[str, float],
    zeta_ep_kg_s_by_state: Mapping[str, float],
    zeta_rel_kg_s_by_state: Mapping[str, float],
) -> float:
    zeta0_kg_s = _positive_float(
        zeta0_kg_s_by_state[state_label],
        f"{state_label}.zeta0_kg_s",
    )
    zeta_ep_kg_s = _nonnegative_float(
        zeta_ep_kg_s_by_state[state_label],
        f"{state_label}.zeta_ep_kg_s",
    )
    zeta_rel_kg_s = _nonnegative_float(
        zeta_rel_kg_s_by_state[state_label],
        f"{state_label}.zeta_rel_kg_s",
    )
    return float(zeta0_kg_s / (zeta0_kg_s + zeta_ep_kg_s + zeta_rel_kg_s))


def _charge_cloud_form_factor(
    transport_state: ProjectedTransportState,
    inverse_screening_length_m_inv: float,
) -> float:
    charge_cloud_radius_m = (
        _positive_float(
            transport_state.charge_cloud_radius_A,
            f"{transport_state.label}.charge_cloud_radius_A",
        )
        * ANGSTROM_TO_M
    )
    screening_radius_product = (
        _positive_float(
            inverse_screening_length_m_inv, "inverse_screening_length_m_inv"
        )
        * charge_cloud_radius_m
    )
    gaussian_exponent = (
        -(screening_radius_product * screening_radius_product)
        / GAUSSIAN_CHARGE_CLOUD_FORM_FACTOR_DENOMINATOR
    )
    return _positive_float(
        math.exp(gaussian_exponent),
        f"{transport_state.label}.charge_cloud_form_factor",
    )


def _countercharge_relaxation_diffusivity_by_state(
    charged_states: tuple[ProjectedTransportState, ...],
    local_diffusivity_m2_s_by_carrier: Mapping[str, float],
) -> Mapping[str, float]:
    relaxation_diffusivity_by_state: dict[str, float] = {}
    for source_state in charged_states:
        countercharge_weighted_diffusivity = 0.0
        countercharge_weight = 0.0
        for target_state in charged_states:
            if (
                source_state.center_charge_number * target_state.center_charge_number
                >= 0
            ):
                continue
            target_weight = target_state.concentration_mol_m3 * abs(
                target_state.center_charge_number
            )
            countercharge_weight += target_weight
            countercharge_weighted_diffusivity += (
                target_weight * local_diffusivity_m2_s_by_carrier[target_state.label]
            )
        if countercharge_weight <= 0.0:
            raise ValueError(
                f"{source_state.label} has no opposite-charge carrier for atmosphere relaxation"
            )
        source_diffusivity = _positive_float(
            local_diffusivity_m2_s_by_carrier[source_state.label],
            f"{source_state.label}.local_diffusivity_m2_s",
        )
        countercharge_diffusivity = (
            countercharge_weighted_diffusivity / countercharge_weight
        )
        relaxation_diffusivity_by_state[source_state.label] = (
            source_diffusivity
            + _positive_float(
                countercharge_diffusivity,
                f"{source_state.label}.countercharge_diffusivity_m2_s",
            )
        )
    return relaxation_diffusivity_by_state


def _projected_transport_state_from_scalar_fields(
    label: str,
    parent_cluster_label: str,
    parent_cluster_kind: str,
    concentration_mol_m3: float,
    center_species_name: str,
    center_charge_number: int,
    center_index: int,
    hydrodynamic_radius_A: float,
    charge_cloud_radius_A: float,
    molecular_volume_A3: float,
    ligand_field_asymmetry: float,
    diffusion_m2_s: float,
    local_obstruction_factor: float,
    local_obstruction_diffusion_scale: float,
    transport_role: str,
    temperature_K: float,
) -> ProjectedTransportState:
    parsed_concentration_mol_m3 = _positive_float(
        concentration_mol_m3,
        f"{label}.concentration_mol_m3",
    )
    parsed_diffusion_m2_s = _positive_float(
        diffusion_m2_s,
        f"{label}.diffusion_m2_s",
    )
    if center_charge_number == 0:
        charged_centers: tuple[ProjectedChargedCenter, ...] = tuple()
        mobility_covariance_matrix_m2_s: tuple[tuple[float, ...], ...] = tuple()
    else:
        charged_centers = (
            ProjectedChargedCenter(
                label=_projected_center_feature_key_from_fields(
                    center_charge_number=center_charge_number,
                    hydrodynamic_radius_A=hydrodynamic_radius_A,
                    charge_cloud_radius_A=charge_cloud_radius_A,
                    molecular_volume_A3=molecular_volume_A3,
                    ligand_field_asymmetry=ligand_field_asymmetry,
                    local_obstruction_factor=local_obstruction_factor,
                ),
                charge_number=center_charge_number,
                diffusion_m2_s=parsed_diffusion_m2_s,
            ),
        )
        mobility_covariance_matrix_m2_s = tuple()
    projected_transport_state = ProjectedTransportState(
        label=label,
        concentration_mol_m3=parsed_concentration_mol_m3,
        charged_centers=charged_centers,
        constraint_modes=tuple(),
        atmosphere_resistance_matrix_kg_s=tuple(),
        mobility_covariance_matrix_m2_s=mobility_covariance_matrix_m2_s,
        ligand_shell_features={
            "temperature_K": _positive_float(temperature_K, f"{label}.temperature_K"),
            "local_obstruction_factor": _nonnegative_float(
                local_obstruction_factor,
                f"{label}.local_obstruction_factor",
            ),
            "local_obstruction_diffusion_scale": _positive_float(
                local_obstruction_diffusion_scale,
                f"{label}.local_obstruction_diffusion_scale",
            ),
        },
        pair_basin=transport_role,
        residence_time_s=math.inf,
        partner_switch_time_s=math.inf,
        parent_cluster_label=parent_cluster_label,
        parent_cluster_kind=parent_cluster_kind,
        center_species_name=center_species_name,
        center_charge_number=center_charge_number,
        center_index=center_index,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        molecular_volume_A3=molecular_volume_A3,
        ligand_field_asymmetry=ligand_field_asymmetry,
        diffusion_m2_s=parsed_diffusion_m2_s,
        local_obstruction_factor=local_obstruction_factor,
        local_obstruction_diffusion_scale=local_obstruction_diffusion_scale,
        transport_role=transport_role,
    )
    return _projected_transport_state_with_rate_budget_diffusion(
        projected_transport_state,
        parsed_diffusion_m2_s,
        temperature_K,
    )


def _transport_state_from_descriptor(
    label: str,
    parent_cluster_label: str,
    parent_cluster_kind: str,
    concentration_mol_m3: float,
    center_species_name: str,
    center_charge_number: int,
    center_index: int,
    descriptor: MolecularSpeciesDescriptor,
    hydrodynamic_radius_scale: float,
    transport_context: _TransportCenterConstructionContext,
    transport_role: str,
) -> ProjectedTransportState:
    hydrodynamic_radius_A = (
        _positive_float(hydrodynamic_radius_scale, f"{label}.hydrodynamic_radius_scale")
        * descriptor.hydrodynamic_radius_A
    )
    charge_cloud_radius_A = _scaled_charge_cloud_radius_A(
        descriptor,
        transport_context.options,
    )
    base_diffusion_m2_s = _diffusion_m2_s(
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        shape_factor=descriptor.ligand_field_asymmetry,
        intrinsic_dielectric_constant=descriptor.epsilon_r_pure,
        net_charge_number=center_charge_number,
        charge_cloud_radius_A=charge_cloud_radius_A,
        charge_density_reference_A_inv3=(
            transport_context.charge_density_reference_A_inv3
        ),
        mixture_descriptor_state=transport_context.mixture_descriptor_state,
        solvent_environment=transport_context.solvent_environment,
        options=transport_context.options,
    )
    local_obstruction_factor = _local_obstruction_factor(
        label=label,
        net_charge_number=center_charge_number,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        mixture_descriptor_state=transport_context.mixture_descriptor_state,
        options=transport_context.options,
        charge_density_reference_A_inv3=(
            transport_context.charge_density_reference_A_inv3
        ),
    )
    local_obstruction_diffusion_scale = _local_obstruction_diffusion_scale(
        local_obstruction_factor,
        label,
    )
    return _projected_transport_state_from_scalar_fields(
        label=label,
        parent_cluster_label=parent_cluster_label,
        parent_cluster_kind=parent_cluster_kind,
        concentration_mol_m3=concentration_mol_m3,
        center_species_name=center_species_name,
        center_charge_number=center_charge_number,
        center_index=center_index,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        molecular_volume_A3=descriptor.molecular_volume_A3,
        ligand_field_asymmetry=descriptor.ligand_field_asymmetry,
        diffusion_m2_s=base_diffusion_m2_s * local_obstruction_diffusion_scale,
        local_obstruction_factor=local_obstruction_factor,
        local_obstruction_diffusion_scale=local_obstruction_diffusion_scale,
        transport_role=transport_role,
        temperature_K=transport_context.solvent_environment.temperature_K,
    )


def _free_ion_hydrodynamic_radius_scale(
    component: IonComponent,
    options: MolecularMoriOptions,
) -> float:
    if component.charge_number > 0:
        return options.primitive_parameters.hydrodynamic_radius_scale_positive_ion
    if component.charge_number < 0:
        return options.primitive_parameters.hydrodynamic_radius_scale_negative_ion
    raise ValueError(f"{component.species_name} free ion must be charged")


def _neutral_transport_states(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    transport_context: _TransportCenterConstructionContext,
) -> tuple[ProjectedTransportState, ...]:
    states: list[ProjectedTransportState] = []
    for species_name, volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        if descriptor.role != ROLE_SOLVENT:
            raise ValueError(f"recipe solvent {species_name} descriptor role mismatch")
        concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            volume_fraction,
            descriptor,
        )
        neutral_feature_label = _transport_center_feature_label(
            partition_role=TRANSPORT_ROLE_NEUTRAL_CENTER,
            parent_cluster_kind=TRANSPORT_ROLE_NEUTRAL_CENTER,
            center_index=0,
            descriptor=descriptor,
            center_charge_number=0,
        )
        states.append(
            _transport_state_from_descriptor(
                label=neutral_feature_label,
                parent_cluster_label=neutral_feature_label,
                parent_cluster_kind=TRANSPORT_ROLE_NEUTRAL_CENTER,
                concentration_mol_m3=concentration_mol_m3,
                center_species_name=species_name,
                center_charge_number=0,
                center_index=0,
                descriptor=descriptor,
                hydrodynamic_radius_scale=1.0,
                transport_context=transport_context,
                transport_role=TRANSPORT_ROLE_NEUTRAL_CENTER,
            )
        )
    return tuple(states)


def _projected_basis_transport_inventory_from_states(
    projected_transport_states_from_mass_balance: tuple[ProjectedTransportState, ...],
    speciation: GenericSpeciationResult,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    temperature_K: float,
) -> _ProjectedTransportInventory:
    if not projected_transport_states_from_mass_balance:
        raise ValueError("projected transport inventory requires projected states")
    projected_transport_states: list[ProjectedTransportState] = []
    solvent_separated_centers_by_parent: dict[
        str,
        list[ProjectedTransportState],
    ] = {}
    for construction_state in projected_transport_states_from_mass_balance:
        if (
            construction_state.transport_role
            == TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER
        ):
            if (
                construction_state.parent_cluster_kind
                != SOLVENT_SEPARATED_PAIR_CLUSTER_KIND
            ):
                raise ValueError(
                    f"{construction_state.label} SSIP center has parent kind "
                    f"{construction_state.parent_cluster_kind}"
                )
            solvent_separated_centers_by_parent.setdefault(
                construction_state.parent_cluster_label,
                [],
            ).append(construction_state)
            continue
        if not _is_markov_transport_state(construction_state):
            continue
        projected_transport_state = _projected_transport_state_from_markov_center(
            construction_state,
            projected_transport_states_from_mass_balance,
            temperature_K,
        )
        projected_transport_states.append(projected_transport_state)
    for (
        parent_cluster_label,
        solvent_separated_centers,
    ) in solvent_separated_centers_by_parent.items():
        pair_inventory = _projected_pair_transport_inventory_for_parent(
            parent_cluster_label,
            tuple(solvent_separated_centers),
            speciation,
            solvent_environment,
            options,
            temperature_K,
        )
        projected_transport_states.extend(pair_inventory.projected_transport_states)
    if not projected_transport_states:
        for construction_state in projected_transport_states_from_mass_balance:
            projected_transport_state = _projected_transport_state_from_markov_center(
                construction_state,
                projected_transport_states_from_mass_balance,
                temperature_K,
            )
            projected_transport_states.append(projected_transport_state)
    return _ProjectedTransportInventory(
        projected_transport_states=tuple(projected_transport_states),
    )


def _projected_pair_transport_inventory_for_parent(
    parent_cluster_label: str,
    solvent_separated_centers: tuple[ProjectedTransportState, ...],
    speciation: GenericSpeciationResult,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    temperature_K: float,
) -> _ProjectedTransportInventory:
    positive_centers = tuple(
        solvent_separated_center
        for solvent_separated_center in solvent_separated_centers
        if solvent_separated_center.center_charge_number > 0
    )
    negative_centers = tuple(
        solvent_separated_center
        for solvent_separated_center in solvent_separated_centers
        if solvent_separated_center.center_charge_number < 0
    )
    if not positive_centers or not negative_centers:
        raise ValueError(
            f"{parent_cluster_label} solvent-separated pair must contain "
            "opposite charged centers"
        )
    projected_transport_states: list[ProjectedTransportState] = []
    for positive_center in positive_centers:
        for negative_center in negative_centers:
            base_concentration_mol_m3 = _positive_float(
                positive_center.concentration_mol_m3,
                f"{parent_cluster_label}.ssip_pair_concentration_mol_m3",
            )
            tolerance_mol_m3 = MASS_BALANCE_TOLERANCE_FACTOR * max(
                1.0,
                base_concentration_mol_m3,
            )
            if base_concentration_mol_m3 > tolerance_mol_m3:
                projected_transport_state = (
                    _projected_solvent_separated_pair_transport_state(
                        positive_center,
                        negative_center,
                        base_concentration_mol_m3,
                        solvent_environment,
                        options,
                        ligand_shell_occupied=False,
                        ligand_motif_label="",
                    )
                )
                projected_transport_states.append(projected_transport_state)
            for (
                ligand_motif_label,
                ligand_bound_concentration_mol_m3,
            ) in speciation.cation_ligand_anion_concentrations_mol_m3.items():
                parent_label = speciation.cation_ligand_anion_parent_cluster_by_label[
                    ligand_motif_label
                ]
                if parent_label != parent_cluster_label:
                    continue
                if ligand_bound_concentration_mol_m3 <= tolerance_mol_m3:
                    continue
                projected_transport_state = (
                    _projected_solvent_separated_pair_transport_state(
                        positive_center,
                        negative_center,
                        ligand_bound_concentration_mol_m3,
                        solvent_environment,
                        options,
                        ligand_shell_occupied=True,
                        ligand_motif_label=ligand_motif_label,
                    )
                )
                projected_transport_states.append(projected_transport_state)
    return _ProjectedTransportInventory(
        projected_transport_states=tuple(projected_transport_states),
    )


def _projected_transport_state_net_charge_number(
    projected_transport_state: ProjectedTransportState,
) -> int:
    net_charge_number = sum(
        charged_center.charge_number
        for charged_center in projected_transport_state.charged_centers
    )
    return int(net_charge_number)


def _projected_transport_state_with_rate_budget_diffusion(
    projected_transport_state: ProjectedTransportState,
    fallback_diffusion_m2_s: float,
    temperature_K: float,
) -> ProjectedTransportState:
    projected_charge_diffusivity_m2_s = (
        compute_projected_transport_state_charge_diffusivity_m2_s(
            projected_transport_state,
            temperature_K,
        )
    )
    if projected_charge_diffusivity_m2_s > 0.0:
        rate_budget_diffusion_m2_s = projected_charge_diffusivity_m2_s
    else:
        rate_budget_diffusion_m2_s = _positive_float(
            fallback_diffusion_m2_s,
            f"{projected_transport_state.label}.fallback_rate_budget_diffusion_m2_s",
        )
    return replace(
        projected_transport_state,
        diffusion_m2_s=rate_budget_diffusion_m2_s,
    )


def _diagnostic_transport_centers_from_projected_transport_states(
    projected_transport_states: tuple[ProjectedTransportState, ...],
) -> tuple[MolecularTransportCenter, ...]:
    return tuple(
        MolecularTransportCenter(
            label=projected_transport_state.label,
            parent_cluster_label=projected_transport_state.parent_cluster_label,
            parent_cluster_kind=projected_transport_state.parent_cluster_kind,
            concentration_mol_m3=projected_transport_state.concentration_mol_m3,
            center_species_name=projected_transport_state.center_species_name,
            center_charge_number=projected_transport_state.center_charge_number,
            center_index=projected_transport_state.center_index,
            hydrodynamic_radius_A=projected_transport_state.hydrodynamic_radius_A,
            charge_cloud_radius_A=projected_transport_state.charge_cloud_radius_A,
            molecular_volume_A3=projected_transport_state.molecular_volume_A3,
            ligand_field_asymmetry=projected_transport_state.ligand_field_asymmetry,
            diffusion_m2_s=projected_transport_state.diffusion_m2_s,
            local_obstruction_factor=projected_transport_state.local_obstruction_factor,
            local_obstruction_diffusion_scale=(
                projected_transport_state.local_obstruction_diffusion_scale
            ),
            transport_role=projected_transport_state.transport_role,
        )
        for projected_transport_state in projected_transport_states
    )


def _markov_process_from_projected_transport_states(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    speciation: GenericSpeciationResult,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
) -> _MarkovProcessConstruction:
    markov_projected_transport_states = tuple(
        projected_transport_state
        for projected_transport_state in projected_transport_states
        if _is_markov_transport_state(projected_transport_state)
    )
    if not markov_projected_transport_states:
        return _neutral_markov_process_from_projected_transport_states(
            projected_transport_states,
            options,
            solvent_environment.temperature_K,
        )
    state_labels: list[str] = []
    state_concentrations: list[float] = []
    markov_projected_states: list[ProjectedTransportState] = []
    events: list[MarkovAdditiveEvent] = []
    memory_primitives: list[MolecularAtmosphereMemoryPrimitive] = []
    mobile_state_indices: list[_MobileTransportStateIndex] = []
    cluster_template_by_label = {
        cluster_template.label: cluster_template
        for cluster_template in cluster_templates
    }
    for transport_state in markov_projected_transport_states:
        mobile_state_index = len(state_labels)
        state_labels.append(f"{transport_state.label}:mobile")
        state_concentrations.append(transport_state.concentration_mol_m3)
        markov_projected_states.append(transport_state)
        mobile_state_indices.append(
            _MobileTransportStateIndex(
                projected_transport_state=transport_state,
                mobile_state_index=mobile_state_index,
                mobile_concentration_mol_m3=(transport_state.concentration_mol_m3),
            )
        )
        if _is_associated_exchange_state(transport_state):
            continue
        if not _is_dc_self_current_carrier(transport_state):
            continue
        if transport_state.center_charge_number == 0:
            continue
        if _state_has_zero_atmosphere_coupling(
            transport_state,
            atmosphere_diagnostics,
        ):
            continue
        memory_primitives.append(
            _atmosphere_memory_primitive(
                transport_state,
                options,
                atmosphere_diagnostics,
                solvent_environment.temperature_K,
            )
        )
    _append_association_conversion_events(
        events,
        tuple(mobile_state_indices),
        cluster_template_by_label,
        solvent_environment,
        options,
    )
    _append_associated_state_exchange_events(
        events,
        tuple(mobile_state_indices),
        cluster_template_by_label,
        solvent_environment,
    )
    _append_projected_mobile_self_current_events(
        events,
        tuple(mobile_state_indices),
        tuple(markov_projected_states),
        solvent_environment.temperature_K,
    )
    projected_current_memory_corrections = _projected_current_memory_corrections(
        tuple(markov_projected_states),
        tuple(memory_primitives),
        options,
        solvent_environment.temperature_K,
    )
    return _MarkovProcessConstruction(
        state_labels=tuple(state_labels),
        state_concentrations_mol_m3=np.asarray(state_concentrations, dtype=float),
        projected_transport_states=tuple(markov_projected_states),
        events=tuple(events),
        memory_primitives=tuple(memory_primitives),
        projected_current_memory_corrections=projected_current_memory_corrections,
        atmosphere_mori_corrections=tuple(),
    )


def _is_markov_transport_state(transport_state: _TransportKineticLike) -> bool:
    if transport_state.transport_role == TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER:
        return True
    if transport_state.transport_role == TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER:
        return False
    if transport_state.center_charge_number != 0:
        return True
    return _is_associated_exchange_state(transport_state)


def _is_associated_exchange_state(transport_state: _TransportKineticLike) -> bool:
    if transport_state.center_charge_number != 0:
        return False
    if transport_state.transport_role == TRANSPORT_ROLE_CONTACT_PAIR_CENTER:
        return True
    if (
        transport_state.transport_role == TRANSPORT_ROLE_CLUSTER_COM_CENTER
        and transport_state.parent_cluster_kind == NEUTRAL_CLUSTER_KIND
    ):
        return True
    return False


def compute_projected_transport_state_charge_diffusivity_m2_s(
    projected_transport_state: ProjectedTransportState,
    temperature_K: float,
) -> float:
    _positive_float(temperature_K, "temperature_K")
    concentration_mol_m3 = _nonnegative_float(
        projected_transport_state.concentration_mol_m3,
        f"{projected_transport_state.label}.concentration_mol_m3",
    )
    if concentration_mol_m3 == 0.0:
        return 0.0
    charged_centers = projected_transport_state.charged_centers
    if not charged_centers:
        return 0.0
    charge_numbers = np.asarray(
        [center.charge_number for center in charged_centers],
        dtype=float,
    )
    if np.all(charge_numbers == 0.0):
        return 0.0
    if projected_transport_state.mobility_covariance_matrix_m2_s:
        mobility_covariance_matrix = _validated_projected_covariance_matrix(
            projected_transport_state.mobility_covariance_matrix_m2_s,
            len(charged_centers),
            f"{projected_transport_state.label}.mobility_covariance_matrix_m2_s",
        )
    else:
        mobility_covariance_matrix = _projected_covariance_matrix_from_resistance(
            projected_transport_state,
            temperature_K,
        )
    charge_diffusivity_m2_s = float(
        charge_numbers @ mobility_covariance_matrix @ charge_numbers
    )
    diffusivity_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        float(np.max(np.abs(mobility_covariance_matrix))),
        np.finfo(float).tiny,
    )
    if charge_diffusivity_m2_s < -diffusivity_tolerance:
        raise ValueError(
            f"{projected_transport_state.label}.charge_diffusivity_m2_s is negative"
        )
    if abs(charge_diffusivity_m2_s) <= diffusivity_tolerance:
        return 0.0
    return charge_diffusivity_m2_s


def _projected_covariance_matrix_from_resistance(
    projected_transport_state: ProjectedTransportState,
    temperature_K: float,
) -> np.ndarray:
    center_count = len(projected_transport_state.charged_centers)
    resistance_matrix_kg_s = np.zeros((center_count, center_count), dtype=float)
    center_index_by_label = {
        center.label: center_index
        for center_index, center in enumerate(projected_transport_state.charged_centers)
    }
    for center_index, center in enumerate(projected_transport_state.charged_centers):
        center_diffusion_m2_s = _positive_float(
            center.diffusion_m2_s,
            f"{projected_transport_state.label}.{center.label}.diffusion_m2_s",
        )
        resistance_matrix_kg_s[center_index, center_index] += (
            K_B * temperature_K / center_diffusion_m2_s
        )
    for constraint_mode in projected_transport_state.constraint_modes:
        first_center_index = center_index_by_label[constraint_mode.first_center_label]
        second_center_index = center_index_by_label[constraint_mode.second_center_label]
        constraint_lifetime_s = _positive_float(
            constraint_mode.lifetime_s,
            f"{projected_transport_state.label}.{constraint_mode.first_center_label}:"
            f"{constraint_mode.second_center_label}.lifetime_s",
        )
        constraint_length_m = _positive_float(
            constraint_mode.length_m,
            f"{projected_transport_state.label}.{constraint_mode.first_center_label}:"
            f"{constraint_mode.second_center_label}.length_m",
        )
        coupling_friction_kg_s = (
            K_B * temperature_K * constraint_lifetime_s / (constraint_length_m**2)
        )
        resistance_matrix_kg_s[first_center_index, first_center_index] += (
            coupling_friction_kg_s
        )
        resistance_matrix_kg_s[second_center_index, second_center_index] += (
            coupling_friction_kg_s
        )
        resistance_matrix_kg_s[first_center_index, second_center_index] -= (
            coupling_friction_kg_s
        )
        resistance_matrix_kg_s[second_center_index, first_center_index] -= (
            coupling_friction_kg_s
        )
    if projected_transport_state.atmosphere_resistance_matrix_kg_s:
        resistance_matrix_kg_s += _validated_projected_covariance_matrix(
            projected_transport_state.atmosphere_resistance_matrix_kg_s,
            center_count,
            f"{projected_transport_state.label}.atmosphere_resistance_matrix_kg_s",
        )
    resistance_matrix_kg_s = _symmetrized_matrix(resistance_matrix_kg_s)
    _validate_symmetric_matrix(
        resistance_matrix_kg_s,
        f"{projected_transport_state.label}.resistance_matrix_kg_s",
    )
    _validate_positive_semidefinite_matrix(
        resistance_matrix_kg_s,
        f"{projected_transport_state.label}.resistance_matrix_kg_s",
    )
    return _symmetrized_matrix(
        K_B * temperature_K * np.linalg.pinv(resistance_matrix_kg_s)
    )


def _validated_projected_covariance_matrix(
    matrix_rows: tuple[tuple[float, ...], ...],
    expected_size: int,
    context: str,
) -> np.ndarray:
    matrix = np.asarray(matrix_rows, dtype=float)
    if matrix.shape != (expected_size, expected_size):
        raise ValueError(f"{context} must have shape {(expected_size, expected_size)}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{context} contains non-finite values")
    matrix = _symmetrized_matrix(matrix)
    _validate_symmetric_matrix(matrix, context)
    _validate_positive_semidefinite_matrix(matrix, context)
    return matrix


def _projected_transport_state_from_markov_center(
    transport_state: ProjectedTransportState,
    transport_states: tuple[ProjectedTransportState, ...],
    temperature_K: float,
) -> ProjectedTransportState:
    if transport_state.center_charge_number == 0:
        member_transport_states = _charged_member_transport_states(
            transport_state,
            transport_states,
        )
        if member_transport_states:
            return _projected_comoving_multi_center_transport_state(
                transport_state,
                member_transport_states,
                temperature_K,
            )
        return _projected_zero_charge_transport_state(
            transport_state,
            temperature_K,
        )
    member_transport_states = _charged_member_transport_states(
        transport_state,
        transport_states,
    )
    if (
        transport_state.transport_role == TRANSPORT_ROLE_CLUSTER_COM_CENTER
        and member_transport_states
    ):
        return _projected_comoving_multi_center_transport_state(
            transport_state,
            member_transport_states,
            temperature_K,
        )
    return _projected_single_center_transport_state(
        transport_state,
        temperature_K,
    )


def _charged_member_transport_states(
    transport_state: ProjectedTransportState,
    transport_states: tuple[ProjectedTransportState, ...],
) -> tuple[ProjectedTransportState, ...]:
    member_transport_states = tuple(
        candidate_transport_state
        for candidate_transport_state in transport_states
        if (
            candidate_transport_state.parent_cluster_label
            == transport_state.parent_cluster_label
            and candidate_transport_state.label != transport_state.label
            and candidate_transport_state.center_charge_number != 0
        )
    )
    return tuple(sorted(member_transport_states, key=_transport_center_sort_key))


def _transport_center_sort_key(
    transport_state: ProjectedTransportState,
) -> tuple[int, str]:
    return (transport_state.center_index, transport_state.label)


def _projected_comoving_multi_center_transport_state(
    transport_state: ProjectedTransportState,
    member_transport_states: tuple[ProjectedTransportState, ...],
    temperature_K: float,
) -> ProjectedTransportState:
    _positive_float(temperature_K, "projected_multi_center.temperature_K")
    center_count = len(member_transport_states)
    if center_count == 0:
        raise ValueError(f"{transport_state.label} has no charged motif centers")
    comoving_diffusion_m2_s = _positive_float(
        transport_state.diffusion_m2_s,
        f"{transport_state.label}.comoving_diffusion_m2_s",
    )
    mobility_covariance_matrix_m2_s = tuple(
        tuple(comoving_diffusion_m2_s for _column_index in range(center_count))
        for _row_index in range(center_count)
    )
    _validated_projected_covariance_matrix(
        mobility_covariance_matrix_m2_s,
        center_count,
        f"{transport_state.label}.comoving_mobility_covariance_matrix_m2_s",
    )
    projected_transport_state = ProjectedTransportState(
        label=transport_state.label,
        concentration_mol_m3=transport_state.concentration_mol_m3,
        charged_centers=tuple(
            ProjectedChargedCenter(
                label=_projected_center_feature_key(member_transport_state),
                charge_number=member_transport_state.center_charge_number,
                diffusion_m2_s=member_transport_state.diffusion_m2_s,
            )
            for member_transport_state in member_transport_states
        ),
        constraint_modes=tuple(),
        atmosphere_resistance_matrix_kg_s=tuple(),
        mobility_covariance_matrix_m2_s=mobility_covariance_matrix_m2_s,
        ligand_shell_features={
            "temperature_K": temperature_K,
            "comoving_diffusion_m2_s": comoving_diffusion_m2_s,
            "local_obstruction_factor": transport_state.local_obstruction_factor,
            "center_count": float(center_count),
        },
        pair_basin=transport_state.parent_cluster_kind,
        residence_time_s=math.inf,
        partner_switch_time_s=math.inf,
        parent_cluster_label=transport_state.parent_cluster_label,
        parent_cluster_kind=transport_state.parent_cluster_kind,
        center_species_name=transport_state.center_species_name,
        center_charge_number=transport_state.center_charge_number,
        center_index=transport_state.center_index,
        hydrodynamic_radius_A=transport_state.hydrodynamic_radius_A,
        charge_cloud_radius_A=transport_state.charge_cloud_radius_A,
        molecular_volume_A3=transport_state.molecular_volume_A3,
        ligand_field_asymmetry=transport_state.ligand_field_asymmetry,
        diffusion_m2_s=comoving_diffusion_m2_s,
        local_obstruction_factor=transport_state.local_obstruction_factor,
        local_obstruction_diffusion_scale=(
            transport_state.local_obstruction_diffusion_scale
        ),
        transport_role=transport_state.transport_role,
    )
    return _projected_transport_state_with_rate_budget_diffusion(
        projected_transport_state,
        comoving_diffusion_m2_s,
        temperature_K,
    )


def _projected_zero_charge_transport_state(
    transport_state: ProjectedTransportState,
    temperature_K: float,
) -> ProjectedTransportState:
    _positive_float(temperature_K, "projected_zero_charge.temperature_K")
    projected_transport_state = ProjectedTransportState(
        label=transport_state.label,
        concentration_mol_m3=transport_state.concentration_mol_m3,
        charged_centers=tuple(),
        constraint_modes=tuple(),
        atmosphere_resistance_matrix_kg_s=tuple(),
        mobility_covariance_matrix_m2_s=tuple(),
        ligand_shell_features={
            "temperature_K": temperature_K,
            "local_obstruction_factor": transport_state.local_obstruction_factor,
        },
        pair_basin=transport_state.transport_role,
        residence_time_s=math.inf,
        partner_switch_time_s=math.inf,
        parent_cluster_label=transport_state.parent_cluster_label,
        parent_cluster_kind=transport_state.parent_cluster_kind,
        center_species_name=transport_state.center_species_name,
        center_charge_number=transport_state.center_charge_number,
        center_index=transport_state.center_index,
        hydrodynamic_radius_A=transport_state.hydrodynamic_radius_A,
        charge_cloud_radius_A=transport_state.charge_cloud_radius_A,
        molecular_volume_A3=transport_state.molecular_volume_A3,
        ligand_field_asymmetry=transport_state.ligand_field_asymmetry,
        diffusion_m2_s=transport_state.diffusion_m2_s,
        local_obstruction_factor=transport_state.local_obstruction_factor,
        local_obstruction_diffusion_scale=(
            transport_state.local_obstruction_diffusion_scale
        ),
        transport_role=transport_state.transport_role,
    )
    return _projected_transport_state_with_rate_budget_diffusion(
        projected_transport_state,
        transport_state.diffusion_m2_s,
        temperature_K,
    )


def _append_projected_mobile_self_current_events(
    events: list[MarkovAdditiveEvent],
    mobile_state_indices: tuple[_MobileTransportStateIndex, ...],
    projected_transport_states: tuple[ProjectedTransportState, ...],
    temperature_K: float,
) -> None:
    projected_state_by_mobile_index = {
        mobile_state_index.mobile_state_index: mobile_state_index.projected_transport_state
        for mobile_state_index in mobile_state_indices
    }
    for mobile_state_index in mobile_state_indices:
        transport_state = mobile_state_index.transport_state
        if not _is_dc_self_current_carrier(transport_state):
            continue
        projected_transport_state = projected_state_by_mobile_index[
            mobile_state_index.mobile_state_index
        ]
        if projected_transport_states[mobile_state_index.mobile_state_index].label != (
            projected_transport_state.label
        ):
            raise ValueError(
                f"{projected_transport_state.label} projected self-current owner "
                "does not match Markov state inventory"
            )
        _append_projected_self_current_event(
            events,
            mobile_state_index.mobile_state_index,
            projected_transport_state,
            _projected_self_current_event_family_label(transport_state),
            temperature_K,
        )


def _projected_self_current_event_family_label(
    transport_state: _TransportKineticLike,
) -> str:
    if transport_state.transport_role == TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER:
        return "projected_solvent_separated_pair_self_current"
    return "projected_single_center_self_current"


def _projected_single_center_transport_state(
    transport_state: ProjectedTransportState,
    temperature_K: float,
) -> ProjectedTransportState:
    projected_transport_state = ProjectedTransportState(
        label=transport_state.label,
        concentration_mol_m3=transport_state.concentration_mol_m3,
        charged_centers=(
            ProjectedChargedCenter(
                label=_projected_center_feature_key(transport_state),
                charge_number=transport_state.center_charge_number,
                diffusion_m2_s=transport_state.diffusion_m2_s,
            ),
        ),
        constraint_modes=tuple(),
        atmosphere_resistance_matrix_kg_s=tuple(),
        mobility_covariance_matrix_m2_s=tuple(),
        ligand_shell_features={"temperature_K": temperature_K},
        pair_basin=transport_state.transport_role,
        residence_time_s=math.inf,
        partner_switch_time_s=math.inf,
        parent_cluster_label=transport_state.parent_cluster_label,
        parent_cluster_kind=transport_state.parent_cluster_kind,
        center_species_name=transport_state.center_species_name,
        center_charge_number=transport_state.center_charge_number,
        center_index=transport_state.center_index,
        hydrodynamic_radius_A=transport_state.hydrodynamic_radius_A,
        charge_cloud_radius_A=transport_state.charge_cloud_radius_A,
        molecular_volume_A3=transport_state.molecular_volume_A3,
        ligand_field_asymmetry=transport_state.ligand_field_asymmetry,
        diffusion_m2_s=transport_state.diffusion_m2_s,
        local_obstruction_factor=transport_state.local_obstruction_factor,
        local_obstruction_diffusion_scale=(
            transport_state.local_obstruction_diffusion_scale
        ),
        transport_role=transport_state.transport_role,
    )
    return _projected_transport_state_with_rate_budget_diffusion(
        projected_transport_state,
        transport_state.diffusion_m2_s,
        temperature_K,
    )


def _append_projected_self_current_event(
    events: list[MarkovAdditiveEvent],
    mobile_state_index: int,
    projected_transport_state: ProjectedTransportState,
    family_label: str,
    temperature_K: float,
) -> None:
    charge_diffusivity_m2_s = compute_projected_transport_state_charge_diffusivity_m2_s(
        projected_transport_state,
        temperature_K,
    )
    if charge_diffusivity_m2_s == 0.0:
        return
    second_moment_tensor_m2 = (
        (2.0 * charge_diffusivity_m2_s, 0.0, 0.0),
        (0.0, 2.0 * charge_diffusivity_m2_s, 0.0),
        (0.0, 0.0, 2.0 * charge_diffusivity_m2_s),
    )
    events.append(
        MarkovAdditiveEvent(
            from_state_index=mobile_state_index,
            to_state_index=mobile_state_index,
            rate_s_inv=1.0,
            charge_displacement_m=(0.0, 0.0, 0.0),
            charge_displacement_second_moment_m2=second_moment_tensor_m2,
            label=f"{family_label}:{projected_transport_state.label}",
            family_label=family_label,
        )
    )


def _state_has_zero_atmosphere_coupling(
    transport_state: _TransportKineticLike,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
) -> bool:
    state_label = transport_state.label
    zeta_ep_kg_s = _nonnegative_float(
        atmosphere_diagnostics.zeta_ep_kg_s_by_state[state_label],
        f"{state_label}.zeta_ep_kg_s",
    )
    zeta_rel_kg_s = _nonnegative_float(
        atmosphere_diagnostics.zeta_rel_kg_s_by_state[state_label],
        f"{state_label}.zeta_rel_kg_s",
    )
    return (zeta_ep_kg_s + zeta_rel_kg_s) == 0.0


def _append_associated_state_exchange_events(
    events: list[MarkovAdditiveEvent],
    mobile_state_indices: tuple[_MobileTransportStateIndex, ...],
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
) -> None:
    free_center_by_species_name = {
        mobile_state_index.transport_state.center_species_name: mobile_state_index
        for mobile_state_index in mobile_state_indices
        if mobile_state_index.transport_state.transport_role
        == TRANSPORT_ROLE_FREE_ION_CENTER
    }
    for associated_state_index in mobile_state_indices:
        associated_transport_state = associated_state_index.transport_state
        if not _is_associated_exchange_state(associated_transport_state):
            continue
        cluster_template = cluster_template_by_label[
            associated_transport_state.parent_cluster_label
        ]
        for charged_center in cluster_template.geometry:
            if charged_center.species_name not in free_center_by_species_name:
                continue
            released_state_index = free_center_by_species_name[
                charged_center.species_name
            ]
            _append_reversible_associated_state_exchange_pair(
                events,
                associated_state_index,
                released_state_index,
                cluster_template_by_label,
                solvent_environment,
            )


def _append_reversible_associated_state_exchange_pair(
    events: list[MarkovAdditiveEvent],
    associated_state_index: _MobileTransportStateIndex,
    released_state_index: _MobileTransportStateIndex,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
) -> None:
    symmetric_flux_mol_m3_s = _associated_state_exchange_flux_mol_m3_s(
        associated_state_index,
        released_state_index,
        cluster_template_by_label,
        solvent_environment,
    )
    associated_to_released_rate_s_inv = (
        symmetric_flux_mol_m3_s / associated_state_index.mobile_concentration_mol_m3
    )
    released_to_associated_rate_s_inv = (
        symmetric_flux_mol_m3_s / released_state_index.mobile_concentration_mol_m3
    )
    events.append(
        MarkovAdditiveEvent(
            from_state_index=associated_state_index.mobile_state_index,
            to_state_index=released_state_index.mobile_state_index,
            rate_s_inv=_positive_float(
                associated_to_released_rate_s_inv,
                "associated_state_exchange_associated_to_released_rate_s_inv",
            ),
            charge_displacement_m=(0.0, 0.0, 0.0),
            charge_displacement_second_moment_m2=ZERO_SECOND_MOMENT_TENSOR_M2,
            label=(
                f"{EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE}:"
                f"{associated_state_index.transport_state.label}:"
                f"{released_state_index.transport_state.label}"
            ),
            family_label=EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE,
        )
    )
    events.append(
        MarkovAdditiveEvent(
            from_state_index=released_state_index.mobile_state_index,
            to_state_index=associated_state_index.mobile_state_index,
            rate_s_inv=_positive_float(
                released_to_associated_rate_s_inv,
                "associated_state_exchange_released_to_associated_rate_s_inv",
            ),
            charge_displacement_m=(0.0, 0.0, 0.0),
            charge_displacement_second_moment_m2=ZERO_SECOND_MOMENT_TENSOR_M2,
            label=(
                f"{EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE}:"
                f"{released_state_index.transport_state.label}:"
                f"{associated_state_index.transport_state.label}"
            ),
            family_label=EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE,
        )
    )


def _associated_state_exchange_flux_mol_m3_s(
    associated_state_index: _MobileTransportStateIndex,
    released_state_index: _MobileTransportStateIndex,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    kinetics = _association_structural_hop_kinetics(
        associated_state_index.transport_state,
        released_state_index.transport_state,
        cluster_template_by_label,
        solvent_environment,
    )
    off_rate_s_inv = (
        kinetics.effective_diffusivity_m2_s
        / (kinetics.hop_length_m * kinetics.hop_length_m)
        * math.exp(-kinetics.free_energy_barrier_over_RT)
    )
    return (
        _positive_float(off_rate_s_inv, "associated_state_exchange_off_rate_s_inv")
        * associated_state_index.mobile_concentration_mol_m3
    )


def _append_association_conversion_events(
    events: list[MarkovAdditiveEvent],
    mobile_state_indices: tuple[_MobileTransportStateIndex, ...],
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> None:
    for source_index, source_state_index in enumerate(mobile_state_indices):
        for target_state_index in mobile_state_indices[source_index + 1 :]:
            if (
                source_state_index.transport_state.center_charge_number
                != target_state_index.transport_state.center_charge_number
            ):
                continue
            if source_state_index.transport_state.center_charge_number == 0:
                continue
            _append_reversible_association_conversion_pair(
                events,
                source_state_index,
                target_state_index,
                cluster_template_by_label,
                solvent_environment,
                options,
            )


def _projected_solvent_separated_pair_transport_state(
    positive_transport_center: ProjectedTransportState,
    negative_transport_center: ProjectedTransportState,
    motif_concentration_mol_m3: float,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    ligand_shell_occupied: bool,
    ligand_motif_label: str,
) -> ProjectedTransportState:
    if ligand_shell_occupied and not ligand_motif_label:
        raise ValueError("ligand-separated pair state requires a ligand motif label")
    if ligand_motif_label and not ligand_shell_occupied:
        raise ValueError("ordinary solvent-separated pair cannot carry ligand motif")
    positive_length_m = _jump_length_m(positive_transport_center, options)
    negative_length_m = _jump_length_m(negative_transport_center, options)
    pair_length_m = _positive_float(
        math.sqrt(positive_length_m * negative_length_m),
        "solvent_separated_pair.constraint_length_m",
    )
    positive_rate_s_inv = _center_translation_rate_budget_s_inv(
        positive_transport_center,
        options,
    )
    negative_rate_s_inv = _center_translation_rate_budget_s_inv(
        negative_transport_center,
        options,
    )
    residence_time_s = 1.0 / _positive_float(
        positive_rate_s_inv + negative_rate_s_inv,
        "solvent_separated_pair.residence_rate_s_inv",
    )
    positive_center_feature_key = _projected_center_feature_key(
        positive_transport_center
    )
    negative_center_feature_key = _projected_center_feature_key(
        negative_transport_center
    )
    mobility_covariance_matrix_m2_s = _projected_pair_mobility_covariance_matrix_m2_s(
        positive_transport_center,
        negative_transport_center,
        residence_time_s,
        pair_length_m,
        solvent_environment,
        options,
        ligand_shell_occupied,
    )
    if mobility_covariance_matrix_m2_s:
        constraint_modes = tuple()
        pair_basin = ADDITIVE_SEPARATED_PAIR_CLUSTER_KIND
        partner_switch_time_s = _solvent_separated_pair_partner_switch_time_s(
            residence_time_s,
            positive_transport_center,
            negative_transport_center,
            solvent_environment,
            options,
            pair_length_m,
            ligand_shell_occupied,
        )
    else:
        constraint_modes = (
            ProjectedConstraintMode(
                first_center_label=positive_center_feature_key,
                second_center_label=negative_center_feature_key,
                lifetime_s=residence_time_s,
                length_m=pair_length_m,
            ),
        )
        if ligand_shell_occupied:
            pair_basin = ADDITIVE_SEPARATED_PAIR_CLUSTER_KIND
            partner_switch_time_s = _solvent_separated_pair_partner_switch_time_s(
                residence_time_s,
                positive_transport_center,
                negative_transport_center,
                solvent_environment,
                options,
                pair_length_m,
                ligand_shell_occupied,
            )
        else:
            pair_basin = SOLVENT_SEPARATED_PAIR_CLUSTER_KIND
            partner_switch_time_s = math.inf
    pair_diffusion_m2_s = math.sqrt(
        _positive_float(
            positive_transport_center.diffusion_m2_s,
            f"{positive_transport_center.label}.diffusion_m2_s",
        )
        * _positive_float(
            negative_transport_center.diffusion_m2_s,
            f"{negative_transport_center.label}.diffusion_m2_s",
        )
    )
    pair_hydrodynamic_radius_A = math.sqrt(
        positive_transport_center.hydrodynamic_radius_A
        * negative_transport_center.hydrodynamic_radius_A
    )
    pair_charge_cloud_radius_A = math.sqrt(
        positive_transport_center.charge_cloud_radius_A
        * negative_transport_center.charge_cloud_radius_A
    )
    pair_local_obstruction_factor = math.sqrt(
        positive_transport_center.local_obstruction_factor
        * negative_transport_center.local_obstruction_factor
    )
    pair_local_obstruction_diffusion_scale = math.sqrt(
        positive_transport_center.local_obstruction_diffusion_scale
        * negative_transport_center.local_obstruction_diffusion_scale
    )
    projected_transport_state = ProjectedTransportState(
        label=(
            "feature_keyed:"
            f"{pair_basin}:"
            f"{positive_center_feature_key}:"
            f"{negative_center_feature_key}:"
            f"ligand_bound={int(ligand_shell_occupied)}:"
            f"ligand_motif={ligand_motif_label if ligand_motif_label else 'unbound'}"
        ),
        concentration_mol_m3=_positive_float(
            motif_concentration_mol_m3,
            "solvent_separated_pair.projected_motif_concentration_mol_m3",
        ),
        charged_centers=(
            ProjectedChargedCenter(
                label=positive_center_feature_key,
                charge_number=positive_transport_center.center_charge_number,
                diffusion_m2_s=positive_transport_center.diffusion_m2_s,
            ),
            ProjectedChargedCenter(
                label=negative_center_feature_key,
                charge_number=negative_transport_center.center_charge_number,
                diffusion_m2_s=negative_transport_center.diffusion_m2_s,
            ),
        ),
        constraint_modes=constraint_modes,
        atmosphere_resistance_matrix_kg_s=tuple(),
        mobility_covariance_matrix_m2_s=(
            mobility_covariance_matrix_m2_s
            if mobility_covariance_matrix_m2_s
            else _solvent_separated_pair_constraint_covariance_matrix_m2_s(
                positive_transport_center,
                negative_transport_center,
                residence_time_s,
                pair_length_m,
                solvent_environment.temperature_K,
            )
        ),
        ligand_shell_features={
            "positive_local_obstruction_factor": (
                positive_transport_center.local_obstruction_factor
            ),
            "negative_local_obstruction_factor": (
                negative_transport_center.local_obstruction_factor
            ),
            "neutral_ligand_site_occupancy": (
                solvent_environment.additive_ligand_site_occupancy
            ),
            "additive_coordination_affinity_J_mol": (
                solvent_environment.additive_coordination_affinity_J_mol
            ),
            "additive_solvation_support": (
                solvent_environment.additive_solvation_support
            ),
            "additive_molecular_volume_A3": (
                solvent_environment.additive_molecular_volume_A3
            ),
            "positive_charge_cloud_radius_A": (
                positive_transport_center.charge_cloud_radius_A
            ),
            "negative_charge_cloud_radius_A": (
                negative_transport_center.charge_cloud_radius_A
            ),
            "positive_hydrodynamic_radius_A": (
                positive_transport_center.hydrodynamic_radius_A
            ),
            "negative_hydrodynamic_radius_A": (
                negative_transport_center.hydrodynamic_radius_A
            ),
            "negative_ligand_field_asymmetry": (
                negative_transport_center.ligand_field_asymmetry
            ),
            "ligand_bound_coordinate": float(ligand_shell_occupied),
        },
        pair_basin=pair_basin,
        residence_time_s=residence_time_s,
        partner_switch_time_s=partner_switch_time_s,
        parent_cluster_label=positive_transport_center.parent_cluster_label,
        parent_cluster_kind=pair_basin,
        center_species_name=(
            f"feature_pair:{positive_transport_center.center_species_name}:"
            f"{negative_transport_center.center_species_name}"
        ),
        center_charge_number=(
            positive_transport_center.center_charge_number
            + negative_transport_center.center_charge_number
        ),
        center_index=0,
        hydrodynamic_radius_A=pair_hydrodynamic_radius_A,
        charge_cloud_radius_A=pair_charge_cloud_radius_A,
        molecular_volume_A3=(
            positive_transport_center.molecular_volume_A3
            + negative_transport_center.molecular_volume_A3
        ),
        ligand_field_asymmetry=max(
            positive_transport_center.ligand_field_asymmetry,
            negative_transport_center.ligand_field_asymmetry,
        ),
        diffusion_m2_s=pair_diffusion_m2_s,
        local_obstruction_factor=pair_local_obstruction_factor,
        local_obstruction_diffusion_scale=pair_local_obstruction_diffusion_scale,
        transport_role=TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    )
    return _projected_transport_state_with_rate_budget_diffusion(
        projected_transport_state,
        pair_diffusion_m2_s,
        solvent_environment.temperature_K,
    )


def _solvent_separated_pair_constraint_covariance_matrix_m2_s(
    positive_transport_center: ProjectedTransportState,
    negative_transport_center: ProjectedTransportState,
    residence_time_s: float,
    pair_length_m: float,
    temperature_K: float,
) -> tuple[tuple[float, float], ...]:
    positive_diffusion_m2_s = _positive_float(
        positive_transport_center.diffusion_m2_s,
        f"{positive_transport_center.label}.diffusion_m2_s",
    )
    negative_diffusion_m2_s = _positive_float(
        negative_transport_center.diffusion_m2_s,
        f"{negative_transport_center.label}.diffusion_m2_s",
    )
    validated_residence_time_s = _positive_float(
        residence_time_s,
        "solvent_separated_pair.residence_time_s",
    )
    validated_pair_length_m = _positive_float(
        pair_length_m,
        "solvent_separated_pair.pair_length_m",
    )
    thermal_energy_J = K_B * _positive_float(
        temperature_K,
        "solvent_separated_pair.temperature_K",
    )
    constraint_friction_kg_s = (
        thermal_energy_J
        * validated_residence_time_s
        / (validated_pair_length_m * validated_pair_length_m)
    )
    resistance_matrix_kg_s = np.asarray(
        (
            (
                thermal_energy_J / positive_diffusion_m2_s + constraint_friction_kg_s,
                -constraint_friction_kg_s,
            ),
            (
                -constraint_friction_kg_s,
                thermal_energy_J / negative_diffusion_m2_s + constraint_friction_kg_s,
            ),
        ),
        dtype=float,
    )
    mobility_covariance_matrix_m2_s = _symmetrized_matrix(
        thermal_energy_J * np.linalg.pinv(resistance_matrix_kg_s)
    )
    _validated_projected_covariance_matrix(
        _matrix_to_tuple_rows(mobility_covariance_matrix_m2_s),
        2,
        "solvent_separated_pair.constraint_mobility_covariance_matrix_m2_s",
    )
    return _matrix_to_tuple_rows(mobility_covariance_matrix_m2_s)


def _projected_pair_mobility_covariance_matrix_m2_s(
    positive_transport_center: ProjectedTransportState,
    negative_transport_center: ProjectedTransportState,
    residence_time_s: float,
    pair_length_m: float,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    ligand_shell_occupied: bool,
) -> tuple[tuple[float, float], ...]:
    covariance_fraction = _projected_pair_signed_current_covariance_fraction(
        positive_transport_center,
        negative_transport_center,
        solvent_environment,
        residence_time_s,
        pair_length_m,
        options,
        ligand_shell_occupied,
    )
    if covariance_fraction == 0.0:
        return tuple()
    positive_diffusion_m2_s = _positive_float(
        positive_transport_center.diffusion_m2_s,
        f"{positive_transport_center.label}.diffusion_m2_s",
    )
    negative_diffusion_m2_s = _positive_float(
        negative_transport_center.diffusion_m2_s,
        f"{negative_transport_center.label}.diffusion_m2_s",
    )
    covariance_m2_s = covariance_fraction * math.sqrt(
        positive_diffusion_m2_s * negative_diffusion_m2_s
    )
    covariance_matrix = (
        (positive_diffusion_m2_s, covariance_m2_s),
        (covariance_m2_s, negative_diffusion_m2_s),
    )
    _validated_projected_covariance_matrix(
        covariance_matrix,
        2,
        "projected_pair.mobility_covariance_matrix_m2_s",
    )
    return covariance_matrix


def _solvent_separated_pair_partner_switch_time_s(
    residence_time_s: float,
    positive_transport_center: ProjectedTransportState,
    negative_transport_center: ProjectedTransportState,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
    pair_length_m: float,
    ligand_shell_occupied: bool,
) -> float:
    covariance_fraction = abs(
        _projected_pair_signed_current_covariance_fraction(
            positive_transport_center,
            negative_transport_center,
            solvent_environment,
            residence_time_s,
            pair_length_m,
            options,
            ligand_shell_occupied,
        )
    )
    if covariance_fraction == 0.0:
        return math.inf
    return _positive_float(
        residence_time_s / covariance_fraction,
        "solvent_separated_pair.partner_switch_time_s",
    )


def _projected_pair_signed_current_covariance_fraction(
    positive_transport_center: ProjectedTransportState,
    negative_transport_center: ProjectedTransportState,
    solvent_environment: MolecularSolventEnvironment,
    residence_time_s: float,
    pair_length_m: float,
    options: MolecularMoriOptions,
    ligand_shell_occupied: bool,
) -> float:
    _positive_float(residence_time_s, "projected_pair.residence_time_s")
    _positive_float(pair_length_m, "projected_pair.pair_length_m")
    _positive_float(
        options.translation_jump_length_multiplier,
        "projected_pair.translation_jump_length_multiplier",
    )
    if not ligand_shell_occupied:
        return 0.0
    additive_coordination_affinity_J_mol = _nonnegative_float(
        solvent_environment.additive_coordination_affinity_J_mol,
        "additive_coordination_affinity_J_mol",
    )
    if additive_coordination_affinity_J_mol == 0.0:
        return 0.0
    additive_affinity_fraction = additive_coordination_affinity_J_mol / (
        additive_coordination_affinity_J_mol + R * solvent_environment.temperature_K
    )
    center_geometry_fraction = _opposite_charge_center_geometry_fraction(
        positive_transport_center,
        negative_transport_center,
    )
    if center_geometry_fraction == 0.0:
        return 0.0
    center_escape_time_s = _positive_float(
        pair_length_m,
        "projected_pair.pair_length_m",
    ) ** 2 / (
        _positive_float(
            positive_transport_center.diffusion_m2_s,
            f"{positive_transport_center.label}.diffusion_m2_s",
        )
        + _positive_float(
            negative_transport_center.diffusion_m2_s,
            f"{negative_transport_center.label}.diffusion_m2_s",
        )
    )
    state_persistence_fraction = residence_time_s / (
        residence_time_s
        + _positive_float(center_escape_time_s, "projected_pair.center_escape_time_s")
    )
    obstruction_asymmetry_fraction = _positive_excess_fraction(
        negative_transport_center.local_obstruction_factor + 1.0,
        positive_transport_center.local_obstruction_factor + 1.0,
        "projected_pair.local_obstruction_factor",
    )
    state_geometry_fraction = _independent_descriptor_union_fraction(
        (center_geometry_fraction, obstruction_asymmetry_fraction)
    )
    return -(
        additive_affinity_fraction
        * state_persistence_fraction
        * state_geometry_fraction
    )


def _opposite_charge_center_geometry_fraction(
    positive_transport_center: ProjectedTransportState,
    negative_transport_center: ProjectedTransportState,
) -> float:
    bulky_fraction = _positive_excess_fraction(
        negative_transport_center.hydrodynamic_radius_A,
        positive_transport_center.hydrodynamic_radius_A,
        "solvent_separated_pair.hydrodynamic_radius_A",
    )
    charge_cloud_excess_fraction = _positive_excess_fraction(
        negative_transport_center.charge_cloud_radius_A,
        positive_transport_center.charge_cloud_radius_A,
        "solvent_separated_pair.charge_cloud_radius_A",
    )
    asymmetry_fraction = _symmetry_breaking_fraction(
        negative_transport_center.ligand_field_asymmetry,
    )
    return _independent_descriptor_union_fraction(
        (bulky_fraction, charge_cloud_excess_fraction, asymmetry_fraction)
    )


def _positive_excess_fraction(
    numerator_value: float,
    reference_value: float,
    context: str,
) -> float:
    parsed_numerator_value = _positive_float(numerator_value, f"{context}.numerator")
    parsed_reference_value = _positive_float(reference_value, f"{context}.reference")
    if parsed_numerator_value <= parsed_reference_value:
        return 0.0
    return (parsed_numerator_value - parsed_reference_value) / (
        parsed_numerator_value + parsed_reference_value
    )


def _symmetry_breaking_fraction(value: float) -> float:
    parsed_value = _positive_float(value, "ligand_field_asymmetry")
    if parsed_value <= 1.0:
        return 0.0
    return (parsed_value - 1.0) / parsed_value


def _independent_descriptor_union_fraction(
    fractions: tuple[float, ...],
) -> float:
    complement_product = 1.0
    for fraction in fractions:
        parsed_fraction = _nonnegative_float(fraction, "descriptor_fraction")
        if parsed_fraction >= 1.0:
            raise ValueError("descriptor_fraction must be smaller than one")
        complement_product *= 1.0 - parsed_fraction
    return 1.0 - complement_product


def _projected_center_feature_key(
    transport_center: ProjectedTransportState,
) -> str:
    return _projected_center_feature_key_from_fields(
        center_charge_number=transport_center.center_charge_number,
        hydrodynamic_radius_A=transport_center.hydrodynamic_radius_A,
        charge_cloud_radius_A=transport_center.charge_cloud_radius_A,
        molecular_volume_A3=transport_center.molecular_volume_A3,
        ligand_field_asymmetry=transport_center.ligand_field_asymmetry,
        local_obstruction_factor=transport_center.local_obstruction_factor,
    )


def _projected_center_feature_key_from_fields(
    center_charge_number: int,
    hydrodynamic_radius_A: float,
    charge_cloud_radius_A: float,
    molecular_volume_A3: float,
    ligand_field_asymmetry: float,
    local_obstruction_factor: float,
) -> str:
    if center_charge_number > 0:
        charge_role = "positive_center"
    elif center_charge_number < 0:
        charge_role = "negative_center"
    else:
        charge_role = "neutral_center"
    return (
        f"{charge_role}:"
        f"z={center_charge_number}:"
        f"rh_A={_feature_value_token(hydrodynamic_radius_A)}:"
        f"rcloud_A={_feature_value_token(charge_cloud_radius_A)}:"
        f"volume_A3={_feature_value_token(molecular_volume_A3)}:"
        f"ligand_asym={_feature_value_token(ligand_field_asymmetry)}:"
        f"obstruction={_feature_value_token(local_obstruction_factor)}"
    )


def _feature_value_token(value: float) -> str:
    parsed_value = _finite_float(value, "feature_key_value")
    return repr(parsed_value).replace("-", "m").replace(".", "p").replace("+", "")


def _center_translation_rate_budget_s_inv(
    transport_center: _TransportKineticLike,
    options: MolecularMoriOptions,
) -> float:
    jump_length_m = _jump_length_m(transport_center, options)
    return _positive_float(
        transport_center.diffusion_m2_s,
        f"{transport_center.label}.diffusion_m2_s",
    ) / (jump_length_m * jump_length_m)


def _append_reversible_association_conversion_pair(
    events: list[MarkovAdditiveEvent],
    first_state_index: _MobileTransportStateIndex,
    second_state_index: _MobileTransportStateIndex,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> None:
    structural_hop_enabled = _association_structural_hop_enabled(
        first_state_index.transport_state,
        second_state_index.transport_state,
    )
    if structural_hop_enabled:
        conversion_length_m = _association_structural_hop_length_m(
            first_state_index.transport_state,
            second_state_index.transport_state,
            cluster_template_by_label,
        )
        family_label = "association_structural_hop"
        second_moment_m2 = _association_structural_hop_second_moment_m2(
            first_state_index.transport_state,
            second_state_index.transport_state,
            cluster_template_by_label,
        )
        symmetric_conductance_mol_m3_s = _association_structural_hop_flux_mol_m3_s(
            first_state_index,
            second_state_index,
            cluster_template_by_label,
            solvent_environment,
        )
    else:
        conversion_length_m = (
            first_state_index.transport_state.hydrodynamic_radius_A
            + second_state_index.transport_state.hydrodynamic_radius_A
        ) * ANGSTROM_TO_M
        family_label = "association_conversion"
        second_moment_m2 = ZERO_SECOND_MOMENT_TENSOR_M2
        encounter_rate_s_inv = (
            first_state_index.transport_state.diffusion_m2_s
            + second_state_index.transport_state.diffusion_m2_s
        ) / (
            _positive_float(conversion_length_m, "association_conversion_length_m") ** 2
        )
        symmetric_conductance_mol_m3_s = (
            options.primitive_parameters.association_conversion_rate_scale
            * _positive_float(encounter_rate_s_inv, "association_encounter_rate_s_inv")
            * math.sqrt(
                _positive_float(
                    first_state_index.mobile_concentration_mol_m3,
                    f"{first_state_index.transport_state.label}.mobile_concentration_mol_m3",
                )
                * _positive_float(
                    second_state_index.mobile_concentration_mol_m3,
                    f"{second_state_index.transport_state.label}.mobile_concentration_mol_m3",
                )
            )
        )
    first_to_second_rate_s_inv = (
        symmetric_conductance_mol_m3_s / first_state_index.mobile_concentration_mol_m3
    )
    second_to_first_rate_s_inv = (
        symmetric_conductance_mol_m3_s / second_state_index.mobile_concentration_mol_m3
    )
    events.append(
        MarkovAdditiveEvent(
            from_state_index=first_state_index.mobile_state_index,
            to_state_index=second_state_index.mobile_state_index,
            rate_s_inv=_positive_float(
                first_to_second_rate_s_inv,
                "association_conversion_first_to_second_rate_s_inv",
            ),
            charge_displacement_m=(0.0, 0.0, 0.0),
            charge_displacement_second_moment_m2=second_moment_m2,
            label=(
                f"{family_label}:"
                f"{first_state_index.transport_state.label}:"
                f"{second_state_index.transport_state.label}"
            ),
            family_label=family_label,
        )
    )
    events.append(
        MarkovAdditiveEvent(
            from_state_index=second_state_index.mobile_state_index,
            to_state_index=first_state_index.mobile_state_index,
            rate_s_inv=_positive_float(
                second_to_first_rate_s_inv,
                "association_conversion_second_to_first_rate_s_inv",
            ),
            charge_displacement_m=(0.0, 0.0, 0.0),
            charge_displacement_second_moment_m2=second_moment_m2,
            label=(
                f"{family_label}:"
                f"{second_state_index.transport_state.label}:"
                f"{first_state_index.transport_state.label}"
            ),
            family_label=family_label,
        )
    )


def _association_structural_hop_enabled(
    first_transport_state: _TransportKineticLike,
    second_transport_state: _TransportKineticLike,
) -> bool:
    if (
        first_transport_state.center_species_name
        != second_transport_state.center_species_name
    ):
        return False
    if (
        first_transport_state.transport_role == second_transport_state.transport_role
        and first_transport_state.parent_cluster_label
        == second_transport_state.parent_cluster_label
    ):
        return False
    return True


def _association_structural_hop_length_m(
    first_transport_state: _TransportKineticLike,
    second_transport_state: _TransportKineticLike,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
) -> float:
    first_partition_radius_A = _transport_partition_radius_A(
        first_transport_state,
        cluster_template_by_label,
    )
    second_partition_radius_A = _transport_partition_radius_A(
        second_transport_state,
        cluster_template_by_label,
    )
    return _positive_float(
        min(first_partition_radius_A, second_partition_radius_A) * ANGSTROM_TO_M,
        "association_structural_hop_length_m",
    )


def _association_structural_hop_second_moment_m2(
    first_transport_state: _TransportKineticLike,
    second_transport_state: _TransportKineticLike,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
) -> tuple[tuple[float, float, float], ...]:
    structural_hop_length_m = _association_structural_hop_length_m(
        first_transport_state,
        second_transport_state,
        cluster_template_by_label,
    )
    charge_center_step_m = (
        abs(first_transport_state.center_charge_number) * structural_hop_length_m
    )
    isotropic_axis_second_moment_m2 = (
        charge_center_step_m * charge_center_step_m / CARTESIAN_AXIS_COUNT
    )
    return (
        (
            float(isotropic_axis_second_moment_m2),
            0.0,
            0.0,
        ),
        (
            0.0,
            float(isotropic_axis_second_moment_m2),
            0.0,
        ),
        (
            0.0,
            0.0,
            float(isotropic_axis_second_moment_m2),
        ),
    )


def _association_structural_hop_flux_mol_m3_s(
    first_state_index: _MobileTransportStateIndex,
    second_state_index: _MobileTransportStateIndex,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    kinetics = _association_structural_hop_kinetics(
        first_state_index.transport_state,
        second_state_index.transport_state,
        cluster_template_by_label,
        solvent_environment,
    )
    base_crossing_rate_s_inv = kinetics.effective_diffusivity_m2_s / (
        kinetics.hop_length_m * kinetics.hop_length_m
    )
    return (
        _positive_float(
            base_crossing_rate_s_inv,
            "association_structural_hop_base_crossing_rate_s_inv",
        )
        * math.exp(-kinetics.free_energy_barrier_over_RT)
        * math.sqrt(
            _positive_float(
                first_state_index.mobile_concentration_mol_m3,
                (
                    f"{first_state_index.transport_state.label}"
                    ".mobile_concentration_mol_m3"
                ),
            )
            * _positive_float(
                second_state_index.mobile_concentration_mol_m3,
                (
                    f"{second_state_index.transport_state.label}"
                    ".mobile_concentration_mol_m3"
                ),
            )
        )
    )


def _association_structural_hop_kinetics(
    first_transport_state: _TransportKineticLike,
    second_transport_state: _TransportKineticLike,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
) -> _AssociationStructuralHopKinetics:
    hop_length_m = _association_structural_hop_length_m(
        first_transport_state,
        second_transport_state,
        cluster_template_by_label,
    )
    effective_diffusivity_m2_s = _harmonic_mean_positive(
        first_transport_state.diffusion_m2_s,
        second_transport_state.diffusion_m2_s,
        "association_structural_hop_effective_diffusivity_m2_s",
    )
    (
        partition_gap_scale_over_RT,
        free_energy_mismatch_over_RT,
        free_energy_barrier_over_RT,
    ) = _association_structural_hop_barrier_components_over_RT(
        first_transport_state,
        second_transport_state,
        cluster_template_by_label,
        solvent_environment,
    )
    return _AssociationStructuralHopKinetics(
        transition_surface=(
            "association_structural_hop_surface:"
            f"{first_transport_state.label}<->{second_transport_state.label}"
        ),
        partition_gap_scale_over_RT=partition_gap_scale_over_RT,
        free_energy_mismatch_over_RT=free_energy_mismatch_over_RT,
        free_energy_barrier_over_RT=free_energy_barrier_over_RT,
        effective_diffusivity_m2_s=effective_diffusivity_m2_s,
        hop_length_m=hop_length_m,
    )


def _association_structural_hop_barrier_over_RT(
    first_transport_state: _TransportKineticLike,
    second_transport_state: _TransportKineticLike,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    (
        _partition_gap_scale_over_RT,
        _free_energy_mismatch_over_RT,
        free_energy_barrier_over_RT,
    ) = _association_structural_hop_barrier_components_over_RT(
        first_transport_state,
        second_transport_state,
        cluster_template_by_label,
        solvent_environment,
    )
    return free_energy_barrier_over_RT


def _association_structural_hop_barrier_components_over_RT(
    first_transport_state: _TransportKineticLike,
    second_transport_state: _TransportKineticLike,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    solvent_environment: MolecularSolventEnvironment,
) -> tuple[float, float, float]:
    first_partition_radius_A = _transport_partition_radius_A(
        first_transport_state,
        cluster_template_by_label,
    )
    second_partition_radius_A = _transport_partition_radius_A(
        second_transport_state,
        cluster_template_by_label,
    )
    solvent_radius_A = _positive_float(
        solvent_environment.solvent_effective_radius_A,
        "solvent_effective_radius_A",
    )
    partition_gap_scale = abs(
        first_partition_radius_A - second_partition_radius_A
    ) / max(
        solvent_radius_A,
        min(first_partition_radius_A, second_partition_radius_A),
    )
    first_state_free_energy_over_RT = _transport_state_partition_free_energy_over_RT(
        first_transport_state,
        cluster_template_by_label,
        solvent_environment.temperature_K,
    )
    second_state_free_energy_over_RT = _transport_state_partition_free_energy_over_RT(
        second_transport_state,
        cluster_template_by_label,
        solvent_environment.temperature_K,
    )
    free_energy_mismatch_over_RT = 0.5 * abs(
        first_state_free_energy_over_RT - second_state_free_energy_over_RT
    )
    free_energy_barrier_over_RT = _nonnegative_float(
        partition_gap_scale + free_energy_mismatch_over_RT,
        "association_structural_hop_barrier_over_RT",
    )
    return (
        _nonnegative_float(
            partition_gap_scale,
            "association_structural_hop_partition_gap_scale_over_RT",
        ),
        _nonnegative_float(
            free_energy_mismatch_over_RT,
            "association_structural_hop_free_energy_mismatch_over_RT",
        ),
        free_energy_barrier_over_RT,
    )


def _transport_state_partition_free_energy_over_RT(
    transport_state: _TransportKineticLike,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
    temperature_K: float,
) -> float:
    if transport_state.parent_cluster_label not in cluster_template_by_label:
        return 0.0
    cluster_template = cluster_template_by_label[transport_state.parent_cluster_label]
    thermal_energy_J_mol = R * _positive_float(temperature_K, "temperature_K")
    return cluster_template.standard_free_energy_J_mol / thermal_energy_J_mol


def _harmonic_mean_positive(
    first_value: float,
    second_value: float,
    context: str,
) -> float:
    validated_first_value = _positive_float(first_value, f"{context}.first_value")
    validated_second_value = _positive_float(second_value, f"{context}.second_value")
    return 2.0 / ((1.0 / validated_first_value) + (1.0 / validated_second_value))


def _transport_partition_radius_A(
    transport_state: _TransportKineticLike,
    cluster_template_by_label: Mapping[str, ClusterStateTemplate],
) -> float:
    if transport_state.parent_cluster_label in cluster_template_by_label:
        cluster_template = cluster_template_by_label[
            transport_state.parent_cluster_label
        ]
        return _positive_float(
            max(
                transport_state.charge_cloud_radius_A,
                _cluster_partition_radius_A(cluster_template),
            ),
            f"{transport_state.label}.partition_radius_A",
        )
    return _positive_float(
        transport_state.charge_cloud_radius_A,
        f"{transport_state.label}.charge_cloud_radius_A",
    )


def _cluster_partition_radius_A(cluster_template: ClusterStateTemplate) -> float:
    center_positions_A = tuple(
        charged_center.position_A[0] for charged_center in cluster_template.geometry
    )
    if not center_positions_A:
        raise ValueError(f"{cluster_template.label} cluster geometry is empty")
    return _positive_float(
        max(abs(position_A) for position_A in center_positions_A),
        f"{cluster_template.label}.cluster_partition_radius_A",
    )


def _charge_displacement_m(
    transport_state: _TransportKineticLike,
    jump_length_m: float,
    axis_vector: tuple[float, float, float],
    direction_sign: float,
) -> tuple[float, float, float]:
    return tuple(
        float(
            direction_sign
            * transport_state.center_charge_number
            * jump_length_m
            * axis_component
        )
        for axis_component in axis_vector
    )


def _atmosphere_memory_primitive(
    projected_transport_state: ProjectedTransportState,
    options: MolecularMoriOptions,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
    temperature_K: float,
) -> MolecularAtmosphereMemoryPrimitive:
    transport_state = projected_transport_state
    state_label = transport_state.label
    zeta0_kg_s = _positive_float(
        atmosphere_diagnostics.zeta0_kg_s_by_state[state_label],
        f"{state_label}.zeta0_kg_s",
    )
    zeta_ep_kg_s = _nonnegative_float(
        atmosphere_diagnostics.zeta_ep_kg_s_by_state[state_label],
        f"{state_label}.zeta_ep_kg_s",
    )
    zeta_rel_kg_s = _nonnegative_float(
        atmosphere_diagnostics.zeta_rel_kg_s_by_state[state_label],
        f"{state_label}.zeta_rel_kg_s",
    )
    zeta_atmosphere_kg_s = zeta_ep_kg_s + zeta_rel_kg_s
    _positive_float(zeta_atmosphere_kg_s, f"{state_label}.zeta_atmosphere_kg_s")
    atmosphere_coupling_fraction = zeta_atmosphere_kg_s / (
        zeta0_kg_s + zeta_atmosphere_kg_s
    )
    if atmosphere_coupling_fraction <= 0.0 or atmosphere_coupling_fraction >= 1.0:
        raise ValueError(
            f"{state_label}.atmosphere_coupling_fraction must be in (0, 1)"
        )
    back_relaxation_probability = zeta_rel_kg_s / zeta_atmosphere_kg_s
    if back_relaxation_probability < 0.0 or back_relaxation_probability > 1.0:
        raise ValueError(f"{state_label}.back_relaxation_probability must be in [0, 1]")
    jump_length_m = _jump_length_m(transport_state, options)
    charge_number_squared = float(
        transport_state.center_charge_number * transport_state.center_charge_number
    )
    _positive_float(charge_number_squared, f"{state_label}.charge_number_squared")
    charge_diffusivity_m2_s = compute_projected_transport_state_charge_diffusivity_m2_s(
        projected_transport_state,
        temperature_K,
    )
    local_diffusivity_m2_s = _positive_float(
        charge_diffusivity_m2_s / charge_number_squared,
        f"{state_label}.D_local_m2_s",
    )
    atmosphere_relaxation_diffusivity_m2_s = _positive_float(
        atmosphere_diagnostics.countercharge_relaxation_diffusivity_m2_s_by_state[
            state_label
        ],
        f"{state_label}.atmosphere_relaxation_diffusivity_m2_s",
    )
    local_debye_screening_rate_s_inv = _debye_screening_rate_s_inv(
        local_diffusivity_m2_s,
        atmosphere_diagnostics.kappa_inv_m,
        f"{state_label}.local_debye_screening_rate_s_inv",
    )
    k_capture_s_inv = (
        options.primitive_parameters.atmosphere_capture_scale
        * atmosphere_coupling_fraction
        * local_debye_screening_rate_s_inv
    )
    k_exit_s_inv = _atmosphere_memory_exit_rate_s_inv(
        atmosphere_relaxation_diffusivity_m2_s,
        atmosphere_diagnostics,
        state_label,
        options,
    )
    residence_ratio = k_capture_s_inv / k_exit_s_inv
    _positive_float(residence_ratio, f"{state_label}.atmosphere_residence_ratio")
    mobile_concentration_mol_m3 = transport_state.concentration_mol_m3
    atmosphere_concentration_per_direction_mol_m3 = ZERO_VALUE
    return MolecularAtmosphereMemoryPrimitive(
        state_label=state_label,
        D_local_m2_s=local_diffusivity_m2_s,
        atmosphere_relaxation_diffusivity_m2_s=(atmosphere_relaxation_diffusivity_m2_s),
        jump_length_m=jump_length_m,
        k_capture_s_inv=k_capture_s_inv,
        k_exit_s_inv=k_exit_s_inv,
        atmosphere_coupling_fraction=float(atmosphere_coupling_fraction),
        back_relaxation_probability=float(back_relaxation_probability),
        mobile_concentration_mol_m3=float(mobile_concentration_mol_m3),
        atmosphere_concentration_per_direction_mol_m3=float(
            atmosphere_concentration_per_direction_mol_m3
        ),
        zeta0_kg_s=zeta0_kg_s,
        zeta_ep_kg_s=zeta_ep_kg_s,
        zeta_rel_kg_s=zeta_rel_kg_s,
    )


def _empty_onsager_transport_operator() -> OnsagerTransportOperator:
    return OnsagerTransportOperator(
        state_labels=tuple(),
        charge_numbers=tuple(),
        concentrations_mol_m3=tuple(),
        bare_diffusivities_m2_s=tuple(),
        diagonal_friction_J_s_mol_m2=tuple(),
        friction_edges=tuple(),
        friction_matrix=tuple(),
        projected_mobility_matrix=tuple(),
        nernst_einstein_sigma_mS_cm=0.0,
        onsager_sigma_mS_cm=0.0,
        correlation_corrector_mS_cm=0.0,
    )


def _onsager_transport_operator_from_transport_states(
    transport_states: tuple[MolecularTransportCenter, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    temperature_K: float,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
) -> OnsagerTransportOperator:
    onsager_carrier_states = _aggregate_onsager_carrier_states(
        transport_states,
        atmosphere_diagnostics,
    )
    if not onsager_carrier_states:
        return _empty_onsager_transport_operator()
    state_labels = tuple(
        f"{carrier_state.label}:mobile" for carrier_state in onsager_carrier_states
    )
    concentrations_mol_m3 = np.asarray(
        [
            carrier_state.concentration_mol_m3
            for carrier_state in onsager_carrier_states
        ],
        dtype=float,
    )
    charge_numbers = np.asarray(
        [carrier_state.charge_number for carrier_state in onsager_carrier_states],
        dtype=float,
    )
    bare_diffusivities_m2_s = np.asarray(
        [carrier_state.diffusion_m2_s for carrier_state in onsager_carrier_states],
        dtype=float,
    )
    diagonal_friction = (R * temperature_K) / bare_diffusivities_m2_s
    bulk_atmosphere_state = build_bulk_ion_atmosphere_state(
        BulkIonAtmosphereInput(
            carrier_labels=tuple(
                carrier_state.label for carrier_state in onsager_carrier_states
            ),
            carrier_concentrations_mol_m3={
                carrier_state.label: carrier_state.concentration_mol_m3
                for carrier_state in onsager_carrier_states
            },
            carrier_charges={
                carrier_state.label: carrier_state.charge_number
                for carrier_state in onsager_carrier_states
            },
            local_diffusivity_m2_s_by_carrier={
                carrier_state.label: carrier_state.diffusion_m2_s
                for carrier_state in onsager_carrier_states
            },
            hydrodynamic_radius_m_by_carrier={
                carrier_state.label: carrier_state.hydrodynamic_radius_A * ANGSTROM_TO_M
                for carrier_state in onsager_carrier_states
            },
            viscosity_Pa_s=solvent_environment.viscosity_cP * CP_TO_PA_S,
            relative_dielectric=solvent_environment.dielectric_constant,
            temperature_K=temperature_K,
            solver="finite_size_bulk_pnp_stokes_l1_cell",
        )
    )
    charge_cloud_form_factors = np.asarray(
        [
            carrier_state.charge_cloud_form_factor
            for carrier_state in onsager_carrier_states
        ],
        dtype=float,
    )
    pair_form_factor_matrix = np.sqrt(
        np.outer(charge_cloud_form_factors, charge_cloud_form_factors)
    )
    scaled_bulk_atmosphere_resistance_matrix_kg_s = (
        primitive_parameters.cross_relaxation_scale
        * pair_form_factor_matrix
        * (
            primitive_parameters.atmosphere_ep_scale
            * bulk_atmosphere_state.resistance_ep_kg_s
            + primitive_parameters.atmosphere_rel_scale
            * bulk_atmosphere_state.resistance_rel_kg_s
        )
    )
    bulk_pair_friction_coefficient_matrix_J_s_mol_m2 = (
        _derive_maxwell_stefan_pair_friction_coefficient_matrix(
            carrier_labels=tuple(
                carrier_state.label for carrier_state in onsager_carrier_states
            ),
            concentration_by_carrier={
                carrier_state.label: carrier_state.concentration_mol_m3
                for carrier_state in onsager_carrier_states
            },
            charge_by_carrier={
                carrier_state.label: carrier_state.charge_number
                for carrier_state in onsager_carrier_states
            },
            resistance_matrix_kg_s=scaled_bulk_atmosphere_resistance_matrix_kg_s,
        )
    )
    bulk_pair_drag_matrix_J_s_mol_m2 = _maxwell_stefan_pair_drag_matrix_J_s_mol_m2(
        carrier_labels=tuple(
            carrier_state.label for carrier_state in onsager_carrier_states
        ),
        concentration_by_carrier={
            carrier_state.label: carrier_state.concentration_mol_m3
            for carrier_state in onsager_carrier_states
        },
        pair_friction_coefficient_matrix_J_s_mol_m2=(
            bulk_pair_friction_coefficient_matrix_J_s_mol_m2
        ),
    )
    friction_matrix = np.diag(concentrations_mol_m3 * diagonal_friction)
    friction_edges: list[OnsagerFrictionEdge] = []
    pair_drag_entry_scale = float(np.max(np.abs(bulk_pair_drag_matrix_J_s_mol_m2)))
    parameter_names = (
        "atmosphere_ep_scale",
        "atmosphere_rel_scale",
        "cross_relaxation_scale",
        "charge_cloud_radius_scale",
    )
    for first_index, first_transport_state in enumerate(onsager_carrier_states):
        for second_index in range(first_index + 1, len(onsager_carrier_states)):
            if not charge_numbers[first_index] * charge_numbers[second_index] < 0.0:
                continue
            pair_drag_J_s_mol_m2 = float(
                bulk_pair_drag_matrix_J_s_mol_m2[first_index, second_index]
            )
            pair_drag_tolerance = np.finfo(float).eps * max(
                1.0,
                pair_drag_entry_scale,
            )
            if abs(pair_drag_J_s_mol_m2) <= pair_drag_tolerance:
                continue
            pair_friction_coefficient_J_s_mol_m2 = float(
                bulk_pair_friction_coefficient_matrix_J_s_mol_m2[
                    first_index, second_index
                ]
            )
            if not pair_friction_coefficient_J_s_mol_m2 > 0.0:
                raise ValueError(
                    "maxwell_stefan pair friction coefficient must be positive for "
                    f"{first_transport_state.label} and "
                    f"{onsager_carrier_states[second_index].label}"
                )
            friction_matrix[first_index, first_index] += pair_drag_J_s_mol_m2
            friction_matrix[second_index, second_index] += pair_drag_J_s_mol_m2
            friction_matrix[first_index, second_index] -= pair_drag_J_s_mol_m2
            friction_matrix[second_index, first_index] -= pair_drag_J_s_mol_m2
            friction_edges.append(
                OnsagerFrictionEdge(
                    first_state_label=state_labels[first_index],
                    second_state_label=state_labels[second_index],
                    friction_coefficient_J_s_mol_m2=float(
                        pair_friction_coefficient_J_s_mol_m2
                    ),
                    source="maxwell_stefan_pair_drag",
                    parameter_names=parameter_names,
                )
            )
    friction_matrix = _symmetrized_matrix(friction_matrix)
    _validate_symmetric_matrix(
        friction_matrix,
        "onsager_transport_operator.friction_matrix",
    )
    _validate_positive_semidefinite_matrix(
        friction_matrix,
        "onsager_transport_operator.friction_matrix",
    )
    mobility_matrix = _symmetrized_matrix(np.linalg.pinv(friction_matrix))
    _validate_symmetric_matrix(
        mobility_matrix,
        "onsager_transport_operator.projected_mobility_matrix",
    )
    _validate_positive_semidefinite_matrix(
        mobility_matrix,
        "onsager_transport_operator.projected_mobility_matrix",
    )
    charge_weighted_concentrations = concentrations_mol_m3 * charge_numbers
    sigma_onsager_S_m = (
        F
        * F
        * float(
            charge_weighted_concentrations
            @ mobility_matrix
            @ charge_weighted_concentrations
        )
    )
    sigma_ne_S_m = (F * F / (R * temperature_K)) * float(
        np.sum(
            concentrations_mol_m3
            * charge_numbers
            * charge_numbers
            * bare_diffusivities_m2_s
        )
    )
    sigma_tolerance_S_m = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        abs(sigma_ne_S_m),
        abs(sigma_onsager_S_m),
    )
    if sigma_onsager_S_m < -sigma_tolerance_S_m:
        raise ValueError("onsager conductivity became negative")
    if abs(sigma_onsager_S_m) <= sigma_tolerance_S_m:
        sigma_onsager_S_m = 0.0
    if not sigma_onsager_S_m <= sigma_ne_S_m + sigma_tolerance_S_m:
        raise ValueError("onsager conductivity exceeded Nernst-Einstein direct limit")
    correlation_corrector_S_m = sigma_ne_S_m - sigma_onsager_S_m
    if abs(correlation_corrector_S_m) <= sigma_tolerance_S_m:
        correlation_corrector_S_m = 0.0
    return OnsagerTransportOperator(
        state_labels=state_labels,
        charge_numbers=tuple(int(value) for value in charge_numbers),
        concentrations_mol_m3=tuple(float(value) for value in concentrations_mol_m3),
        bare_diffusivities_m2_s=tuple(
            float(value) for value in bare_diffusivities_m2_s
        ),
        diagonal_friction_J_s_mol_m2=tuple(float(value) for value in diagonal_friction),
        friction_edges=tuple(friction_edges),
        friction_matrix=tuple(
            tuple(float(value) for value in row) for row in friction_matrix
        ),
        projected_mobility_matrix=tuple(
            tuple(float(value) for value in row) for row in mobility_matrix
        ),
        nernst_einstein_sigma_mS_cm=float(sigma_ne_S_m * S_M_TO_MS_CM),
        onsager_sigma_mS_cm=float(sigma_onsager_S_m * S_M_TO_MS_CM),
        correlation_corrector_mS_cm=float(correlation_corrector_S_m * S_M_TO_MS_CM),
    )


def _onsager_carrier_group_key(
    transport_state: MolecularTransportCenter,
) -> tuple[str, str, int]:
    if transport_state.transport_role == TRANSPORT_ROLE_FREE_ION_CENTER:
        return (
            TRANSPORT_ROLE_FREE_ION_CENTER,
            transport_state.center_species_name,
            transport_state.center_charge_number,
        )
    if (
        transport_state.parent_cluster_kind == SOLVENT_SEPARATED_PAIR_CLUSTER_KIND
        and transport_state.transport_role
        in (
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
            TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
        )
    ):
        return (
            SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
            transport_state.center_species_name,
            transport_state.center_charge_number,
        )
    return (
        transport_state.transport_role,
        f"{transport_state.parent_cluster_label}:{transport_state.center_species_name}",
        transport_state.center_charge_number,
    )


def _is_dc_self_current_carrier(
    transport_state: _TransportKineticLike,
) -> bool:
    if transport_state.transport_role == TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER:
        return True
    if transport_state.center_charge_number == 0:
        return False
    if transport_state.transport_role == TRANSPORT_ROLE_FREE_ION_CENTER:
        return True
    if transport_state.transport_role == TRANSPORT_ROLE_LIGAND_SHELL_CENTER:
        return True
    if transport_state.transport_role == TRANSPORT_ROLE_CLUSTER_COM_CENTER:
        return True
    if transport_state.transport_role == TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER:
        return True
    return False


def _onsager_carrier_label(
    carrier_group_key: tuple[str, str, int],
) -> str:
    return (
        f"{carrier_group_key[0]}:"
        f"{_transport_label_token(carrier_group_key[1])}:"
        f"q{carrier_group_key[2]}"
    )


def _aggregate_onsager_carrier_states(
    transport_states: tuple[MolecularTransportCenter, ...],
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
) -> tuple[_OnsagerCarrierState, ...]:
    grouped_transport_states = {}
    for transport_state in transport_states:
        if not _is_dc_self_current_carrier(transport_state):
            continue
        if transport_state.concentration_mol_m3 <= 0.0:
            continue
        carrier_group_key = _onsager_carrier_group_key(transport_state)
        grouped_transport_states.setdefault(carrier_group_key, []).append(
            transport_state
        )
    aggregated_carrier_states: list[_OnsagerCarrierState] = []
    for carrier_group_key, grouped_states in grouped_transport_states.items():
        total_concentration_mol_m3 = math.fsum(
            transport_state.concentration_mol_m3 for transport_state in grouped_states
        )
        concentration_weighted_diffusivity = math.fsum(
            transport_state.concentration_mol_m3 * transport_state.diffusion_m2_s
            for transport_state in grouped_states
        )
        concentration_weighted_radius_A = math.fsum(
            transport_state.concentration_mol_m3 * transport_state.hydrodynamic_radius_A
            for transport_state in grouped_states
        )
        concentration_weighted_form_factor = math.fsum(
            transport_state.concentration_mol_m3
            * atmosphere_diagnostics.charge_cloud_form_factor_by_state[
                transport_state.label
            ]
            for transport_state in grouped_states
        )
        representative_state = grouped_states[0]
        aggregated_carrier_states.append(
            _OnsagerCarrierState(
                carrier_group_key=carrier_group_key,
                label=_onsager_carrier_label(carrier_group_key),
                concentration_mol_m3=total_concentration_mol_m3,
                charge_number=representative_state.center_charge_number,
                diffusion_m2_s=(
                    concentration_weighted_diffusivity / total_concentration_mol_m3
                ),
                hydrodynamic_radius_A=(
                    concentration_weighted_radius_A / total_concentration_mol_m3
                ),
                charge_cloud_form_factor=(
                    concentration_weighted_form_factor / total_concentration_mol_m3
                ),
            )
        )
    return tuple(aggregated_carrier_states)


def _atmosphere_mori_correction(
    memory_primitive: MolecularAtmosphereMemoryPrimitive,
    projected_transport_state: ProjectedTransportState,
    ordinary_direct_axis_density_m2_s_mol_m3: tuple[float, float, float],
    temperature_K: float,
) -> AtmosphereMoriCorrection:
    transport_state = projected_transport_state
    charge_number_squared = float(
        transport_state.center_charge_number * transport_state.center_charge_number
    )
    if charge_number_squared <= 0.0:
        raise ValueError(
            f"{transport_state.label} atmosphere correction requires nonzero charge"
        )
    total_friction_kg_s = _positive_float(
        memory_primitive.zeta0_kg_s
        + memory_primitive.zeta_ep_kg_s
        + memory_primitive.zeta_rel_kg_s,
        f"{transport_state.label}.total_atmosphere_friction_kg_s",
    )
    D_long_m2_s = _positive_float(
        (K_B * temperature_K) / total_friction_kg_s,
        f"{transport_state.label}.D_long_m2_s",
    )
    memory_occupancy_fraction = _memory_occupancy_fraction(memory_primitive)
    D_atmosphere_correction_m2_s = memory_occupancy_fraction * (
        memory_primitive.D_local_m2_s - D_long_m2_s
    )
    diffusivity_tolerance_m2_s = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        memory_primitive.D_local_m2_s,
        D_long_m2_s,
    )
    if D_atmosphere_correction_m2_s < -diffusivity_tolerance_m2_s:
        raise ValueError(
            f"{transport_state.label}.D_atmosphere_correction_m2_s became negative"
        )
    if abs(D_atmosphere_correction_m2_s) <= diffusivity_tolerance_m2_s:
        D_atmosphere_correction_m2_s = 0.0
    correction_axis_density_value = (
        memory_primitive.mobile_concentration_mol_m3
        * charge_number_squared
        * D_atmosphere_correction_m2_s
    )
    correction_axis_density = np.asarray(
        (
            correction_axis_density_value,
            correction_axis_density_value,
            correction_axis_density_value,
        ),
        dtype=float,
    )
    ordinary_direct_axis_density = np.asarray(
        ordinary_direct_axis_density_m2_s_mol_m3,
        dtype=float,
    )
    density_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        float(np.max(np.abs(ordinary_direct_axis_density))),
        float(np.max(np.abs(correction_axis_density))),
    )
    if (
        not float(np.max(correction_axis_density - ordinary_direct_axis_density))
        <= density_tolerance
    ):
        raise ValueError(
            f"{transport_state.label}.atmosphere correction exceeds ordinary direct density"
        )
    correction_sigma_S_m = (
        (F * F / (R * temperature_K))
        * float(np.sum(correction_axis_density))
        / CARTESIAN_AXIS_COUNT
    )
    memory_self_energy_s_inv = _positive_float(
        memory_primitive.k_capture_s_inv + memory_primitive.k_exit_s_inv,
        f"{transport_state.label}.atmosphere_memory_self_energy_s_inv",
    )
    return AtmosphereMoriCorrection(
        state_label=f"{transport_state.label}:mobile",
        transport_state_label=transport_state.label,
        charge_number=transport_state.center_charge_number,
        concentration_mol_m3=float(memory_primitive.mobile_concentration_mol_m3),
        D_short_m2_s=float(memory_primitive.D_local_m2_s),
        zeta_bare_kg_s=float(memory_primitive.zeta0_kg_s),
        zeta_ep_kg_s=float(memory_primitive.zeta_ep_kg_s),
        zeta_rel_kg_s=float(memory_primitive.zeta_rel_kg_s),
        D_long_m2_s=float(D_long_m2_s),
        D_atmosphere_correction_m2_s=float(D_atmosphere_correction_m2_s),
        memory_self_energy_s_inv=float(memory_self_energy_s_inv),
        correction_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in correction_axis_density
        ),
        correction_sigma_S_m=float(correction_sigma_S_m),
        correction_sigma_mS_cm=float(correction_sigma_S_m * S_M_TO_MS_CM),
        source="derived_ionic_atmosphere_mori_friction",
    )


def _memory_occupancy_fraction(
    memory_primitive: MolecularAtmosphereMemoryPrimitive,
) -> float:
    k_capture_s_inv = _positive_float(
        memory_primitive.k_capture_s_inv,
        f"{memory_primitive.state_label}.k_capture_s_inv",
    )
    k_exit_s_inv = _positive_float(
        memory_primitive.k_exit_s_inv,
        f"{memory_primitive.state_label}.k_exit_s_inv",
    )
    memory_occupancy_fraction = k_capture_s_inv / (k_capture_s_inv + k_exit_s_inv)
    if memory_occupancy_fraction <= 0.0 or memory_occupancy_fraction >= 1.0:
        raise ValueError(
            f"{memory_primitive.state_label}.memory_occupancy_fraction must be in (0, 1)"
        )
    return float(memory_occupancy_fraction)


def _audit_atmosphere_mori_corrections(
    memory_primitives: tuple[MolecularAtmosphereMemoryPrimitive, ...],
    projected_transport_states: tuple[ProjectedTransportState, ...],
    temperature_K: float,
) -> tuple[AtmosphereMoriCorrection, ...]:
    projected_transport_state_by_label = {
        projected_transport_state.label: projected_transport_state
        for projected_transport_state in projected_transport_states
    }
    corrections: list[AtmosphereMoriCorrection] = []
    for memory_primitive in memory_primitives:
        if memory_primitive.state_label not in projected_transport_state_by_label:
            raise ValueError(
                f"{memory_primitive.state_label} atmosphere memory state is missing"
            )
        projected_transport_state = projected_transport_state_by_label[
            memory_primitive.state_label
        ]
        corrections.append(
            _atmosphere_mori_correction(
                memory_primitive,
                projected_transport_state,
                _ordinary_self_current_axis_density_m2_s_mol_m3(
                    projected_transport_state,
                    temperature_K,
                ),
                temperature_K,
            )
        )
    return tuple(corrections)


def _projected_current_memory_correction_from_atmosphere_coordinate(
    memory_primitive: MolecularAtmosphereMemoryPrimitive,
    projected_transport_state: ProjectedTransportState,
    temperature_K: float,
) -> ProjectedCurrentMemoryCorrection:
    charge_number_squared = float(
        projected_transport_state.center_charge_number
        * projected_transport_state.center_charge_number
    )
    if charge_number_squared <= 0.0:
        raise ValueError(
            f"{projected_transport_state.label} atmosphere memory requires charge"
        )
    total_friction_kg_s = _positive_float(
        memory_primitive.zeta0_kg_s
        + memory_primitive.zeta_ep_kg_s
        + memory_primitive.zeta_rel_kg_s,
        f"{projected_transport_state.label}.total_memory_friction_kg_s",
    )
    long_time_diffusion_m2_s = _positive_float(
        (K_B * temperature_K) / total_friction_kg_s,
        f"{projected_transport_state.label}.long_time_memory_diffusion_m2_s",
    )
    memory_occupancy_fraction = _memory_occupancy_fraction(memory_primitive)
    memory_diffusion_correction_m2_s = memory_occupancy_fraction * (
        memory_primitive.D_local_m2_s - long_time_diffusion_m2_s
    )
    diffusivity_tolerance_m2_s = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        memory_primitive.D_local_m2_s,
        long_time_diffusion_m2_s,
    )
    if memory_diffusion_correction_m2_s < -diffusivity_tolerance_m2_s:
        raise ValueError(
            f"{projected_transport_state.label}.memory_diffusion_correction_m2_s "
            "became negative"
        )
    if abs(memory_diffusion_correction_m2_s) <= diffusivity_tolerance_m2_s:
        memory_diffusion_correction_m2_s = 0.0
    correction_axis_density_value = (
        memory_primitive.mobile_concentration_mol_m3
        * charge_number_squared
        * memory_diffusion_correction_m2_s
    )
    correction_axis_density = np.asarray(
        (
            correction_axis_density_value,
            correction_axis_density_value,
            correction_axis_density_value,
        ),
        dtype=float,
    )
    ordinary_direct_axis_density = np.asarray(
        _ordinary_self_current_axis_density_m2_s_mol_m3(
            projected_transport_state,
            temperature_K,
        ),
        dtype=float,
    )
    density_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        float(np.max(np.abs(ordinary_direct_axis_density))),
        float(np.max(np.abs(correction_axis_density))),
    )
    if (
        not float(np.max(correction_axis_density - ordinary_direct_axis_density))
        <= density_tolerance
    ):
        raise ValueError(
            f"{projected_transport_state.label}.memory correction exceeds direct density"
        )
    correction_sigma_S_m = (
        (F * F / (R * temperature_K))
        * float(np.sum(correction_axis_density))
        / CARTESIAN_AXIS_COUNT
    )
    memory_self_energy_s_inv = _positive_float(
        memory_primitive.k_capture_s_inv + memory_primitive.k_exit_s_inv,
        f"{projected_transport_state.label}.memory_self_energy_s_inv",
    )
    return ProjectedCurrentMemoryCorrection(
        state_label=f"{projected_transport_state.label}:mobile",
        transport_state_label=projected_transport_state.label,
        memory_family_label="atmosphere_polarization_memory_coordinate",
        concentration_mol_m3=float(memory_primitive.mobile_concentration_mol_m3),
        memory_self_energy_s_inv=float(memory_self_energy_s_inv),
        correction_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in correction_axis_density
        ),
        correction_sigma_S_m=float(correction_sigma_S_m),
        correction_sigma_mS_cm=float(correction_sigma_S_m * S_M_TO_MS_CM),
        source="projected_ionic_atmosphere_memory_coordinate",
    )


def _projected_current_memory_corrections(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    memory_primitives: tuple[MolecularAtmosphereMemoryPrimitive, ...],
    options: MolecularMoriOptions,
    temperature_K: float,
) -> tuple[ProjectedCurrentMemoryCorrection, ...]:
    projected_transport_state_by_label = {
        projected_transport_state.label: projected_transport_state
        for projected_transport_state in projected_transport_states
    }
    corrections: list[ProjectedCurrentMemoryCorrection] = []
    atmosphere_axis_density_by_transport_state_label: dict[str, np.ndarray] = {}
    for memory_primitive in memory_primitives:
        if memory_primitive.state_label not in projected_transport_state_by_label:
            raise ValueError(
                f"{memory_primitive.state_label} memory coordinate is missing"
            )
        projected_transport_state = projected_transport_state_by_label[
            memory_primitive.state_label
        ]
        memory_coordinate = _projected_current_memory_correction_from_atmosphere_coordinate(
            memory_primitive,
            projected_transport_state,
            temperature_K,
        )
        corrections.append(memory_coordinate)
        atmosphere_axis_density_by_transport_state_label.setdefault(
            memory_coordinate.transport_state_label,
            np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float),
        )
        atmosphere_axis_density_by_transport_state_label[
            memory_coordinate.transport_state_label
        ] += np.asarray(
            memory_coordinate.correction_axis_density_m2_s_mol_m3,
            dtype=float,
        )
    for projected_transport_state in projected_transport_states:
        corrections.extend(
            _projected_structural_current_memory_corrections_for_state(
                projected_transport_state,
                atmosphere_axis_density_by_transport_state_label.get(
                    projected_transport_state.label,
                    np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float),
                ),
                options,
                temperature_K,
            )
        )
    return tuple(corrections)


def _projected_structural_current_memory_corrections_for_state(
    projected_transport_state: ProjectedTransportState,
    atmosphere_axis_density_m2_s_mol_m3: np.ndarray,
    options: MolecularMoriOptions,
    temperature_K: float,
) -> tuple[ProjectedCurrentMemoryCorrection, ...]:
    if not _is_dc_self_current_carrier(projected_transport_state):
        return tuple()
    direct_axis_density = np.asarray(
        _ordinary_self_current_axis_density_m2_s_mol_m3(
            projected_transport_state,
            temperature_K,
        ),
        dtype=float,
    )
    remaining_axis_density = direct_axis_density - np.asarray(
        atmosphere_axis_density_m2_s_mol_m3,
        dtype=float,
    )
    density_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(direct_axis_density))),
        float(np.max(np.abs(atmosphere_axis_density_m2_s_mol_m3))),
    )
    if float(np.min(remaining_axis_density)) < -density_tolerance:
        raise ValueError(
            f"{projected_transport_state.label}.current_memory exceeds direct density"
        )
    remaining_axis_density = np.asarray(
        [
            ZERO_VALUE if abs(value) <= density_tolerance else float(value)
            for value in remaining_axis_density
        ],
        dtype=float,
    )
    if float(np.max(remaining_axis_density)) <= 0.0:
        return tuple()
    translation_rate_s_inv = _center_translation_rate_budget_s_inv(
        projected_transport_state,
        options,
    )
    corrections: list[ProjectedCurrentMemoryCorrection] = []
    for memory_candidate in _projected_structural_memory_candidates(
        projected_transport_state,
        translation_rate_s_inv,
        options,
    ):
        memory_fraction = memory_candidate[1]
        if memory_fraction == 0.0:
            continue
        correction_axis_density = remaining_axis_density * memory_fraction
        correction = _projected_current_memory_correction(
            projected_transport_state,
            memory_candidate[0],
            memory_candidate[2],
            correction_axis_density,
            temperature_K,
        )
        corrections.append(correction)
        remaining_axis_density = remaining_axis_density - correction_axis_density
        if float(np.max(remaining_axis_density)) <= density_tolerance:
            break
    return tuple(corrections)


def _projected_structural_memory_candidates(
    projected_transport_state: ProjectedTransportState,
    translation_rate_s_inv: float,
    options: MolecularMoriOptions,
) -> tuple[tuple[str, float, float], ...]:
    orientation_relaxation_rate_scale = _positive_float(
        options.primitive_parameters.orientation_relaxation_rate_scale,
        "orientation_relaxation_rate_scale",
    )
    candidates: list[tuple[str, float, float]] = []
    local_obstruction_factor = _positive_float(
        projected_transport_state.local_obstruction_factor,
        f"{projected_transport_state.label}.local_obstruction_factor",
    )
    if local_obstruction_factor > 0.0:
        cage_relaxation_rate_s_inv = _positive_float(
            orientation_relaxation_rate_scale
            * translation_rate_s_inv
            / (1.0 + local_obstruction_factor),
            f"{projected_transport_state.label}.cage_relaxation_rate_s_inv",
        )
        cage_memory_capacity_fraction = local_obstruction_factor / (
            1.0 + local_obstruction_factor
        )
        cage_fraction = (
            cage_memory_capacity_fraction
            * _current_memory_fraction_from_rates(
                translation_rate_s_inv,
                cage_relaxation_rate_s_inv,
                f"{projected_transport_state.label}.cage_current_memory_fraction",
            )
        )
        candidates.append(
            (
                "projected_cage_current_memory",
                cage_fraction,
                cage_relaxation_rate_s_inv,
            )
        )
    if math.isfinite(projected_transport_state.residence_time_s):
        residence_relaxation_rate_s_inv = _positive_float(
            orientation_relaxation_rate_scale
            / _positive_float(
                projected_transport_state.residence_time_s,
                f"{projected_transport_state.label}.residence_time_s",
            ),
            f"{projected_transport_state.label}.residence_relaxation_rate_s_inv",
        )
        residence_fraction = _current_memory_fraction_from_rates(
            translation_rate_s_inv,
            residence_relaxation_rate_s_inv,
            f"{projected_transport_state.label}.residence_current_memory_fraction",
        )
        candidates.append(
            (
                "projected_residence_current_memory",
                residence_fraction,
                residence_relaxation_rate_s_inv,
            )
        )
    if math.isfinite(projected_transport_state.partner_switch_time_s):
        partner_switch_relaxation_rate_s_inv = _positive_float(
            orientation_relaxation_rate_scale
            / _positive_float(
                projected_transport_state.partner_switch_time_s,
                f"{projected_transport_state.label}.partner_switch_time_s",
            ),
            f"{projected_transport_state.label}.partner_switch_relaxation_rate_s_inv",
        )
        partner_switch_fraction = _current_memory_fraction_from_rates(
            translation_rate_s_inv,
            partner_switch_relaxation_rate_s_inv,
            f"{projected_transport_state.label}.partner_switch_current_memory_fraction",
        )
        candidates.append(
            (
                "projected_partner_switch_current_memory",
                partner_switch_fraction,
                partner_switch_relaxation_rate_s_inv,
            )
        )
    for constraint_mode in projected_transport_state.constraint_modes:
        constraint_relaxation_rate_s_inv = _positive_float(
            orientation_relaxation_rate_scale
            / _positive_float(
                constraint_mode.lifetime_s,
                f"{projected_transport_state.label}.{constraint_mode.first_center_label}:"
                f"{constraint_mode.second_center_label}.constraint_lifetime_s",
            ),
            f"{projected_transport_state.label}.constraint_relaxation_rate_s_inv",
        )
        constraint_fraction = _current_memory_fraction_from_rates(
            translation_rate_s_inv,
            constraint_relaxation_rate_s_inv,
            f"{projected_transport_state.label}.constraint_current_memory_fraction",
        )
        candidates.append(
            (
                "projected_constraint_current_memory",
                constraint_fraction,
                constraint_relaxation_rate_s_inv,
            )
        )
    return tuple(candidates)


def _current_memory_fraction_from_rates(
    translation_rate_s_inv: float,
    memory_relaxation_rate_s_inv: float,
    context: str,
) -> float:
    parsed_translation_rate_s_inv = _positive_float(
        translation_rate_s_inv,
        f"{context}.translation_rate_s_inv",
    )
    parsed_memory_relaxation_rate_s_inv = _positive_float(
        memory_relaxation_rate_s_inv,
        f"{context}.memory_relaxation_rate_s_inv",
    )
    memory_fraction = parsed_translation_rate_s_inv / (
        parsed_translation_rate_s_inv + parsed_memory_relaxation_rate_s_inv
    )
    if memory_fraction <= 0.0 or memory_fraction >= 1.0:
        raise ValueError(f"{context} must be in (0, 1)")
    return float(memory_fraction)


def _projected_current_memory_correction(
    projected_transport_state: ProjectedTransportState,
    memory_family_label: str,
    memory_self_energy_s_inv: float,
    correction_axis_density_m2_s_mol_m3: np.ndarray,
    temperature_K: float,
) -> ProjectedCurrentMemoryCorrection:
    correction_axis_density = np.asarray(
        correction_axis_density_m2_s_mol_m3,
        dtype=float,
    )
    if correction_axis_density.shape != (int(CARTESIAN_AXIS_COUNT),):
        raise ValueError(
            f"{projected_transport_state.label}.{memory_family_label}.axis_density "
            "shape mismatch"
        )
    if not np.all(np.isfinite(correction_axis_density)):
        raise ValueError(
            f"{projected_transport_state.label}.{memory_family_label}.axis_density "
            "contains non-finite values"
        )
    if float(np.min(correction_axis_density)) < 0.0:
        raise ValueError(
            f"{projected_transport_state.label}.{memory_family_label}.axis_density "
            "must be nonnegative"
        )
    correction_sigma_S_m = (
        (F * F / (R * temperature_K))
        * float(np.sum(correction_axis_density))
        / CARTESIAN_AXIS_COUNT
    )
    return ProjectedCurrentMemoryCorrection(
        state_label=f"{projected_transport_state.label}:mobile",
        transport_state_label=projected_transport_state.label,
        memory_family_label=memory_family_label,
        concentration_mol_m3=projected_transport_state.concentration_mol_m3,
        memory_self_energy_s_inv=_positive_float(
            memory_self_energy_s_inv,
            f"{projected_transport_state.label}.{memory_family_label}.memory_self_energy_s_inv",
        ),
        correction_axis_density_m2_s_mol_m3=tuple(
            float(value) for value in correction_axis_density
        ),
        correction_sigma_S_m=float(correction_sigma_S_m),
        correction_sigma_mS_cm=float(correction_sigma_S_m * S_M_TO_MS_CM),
        source="projected_state_current_memory_from_projected_coordinates",
    )


def _ordinary_self_current_axis_density_m2_s_mol_m3(
    projected_transport_state: ProjectedTransportState,
    temperature_K: float,
) -> tuple[float, float, float]:
    charge_diffusivity_m2_s = compute_projected_transport_state_charge_diffusivity_m2_s(
        projected_transport_state,
        temperature_K,
    )
    axis_density_value = (
        projected_transport_state.concentration_mol_m3 * charge_diffusivity_m2_s
    )
    return (
        float(axis_density_value),
        float(axis_density_value),
        float(axis_density_value),
    )


def _atmosphere_memory_exit_rate_s_inv(
    atmosphere_relaxation_diffusivity_m2_s: float,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
    state_label: str,
    options: MolecularMoriOptions,
) -> float:
    exit_rate_s_inv = (
        options.primitive_parameters.orientation_relaxation_rate_scale
        * options.primitive_parameters.atmosphere_exit_scale
        * _debye_screening_rate_s_inv(
            atmosphere_relaxation_diffusivity_m2_s,
            atmosphere_diagnostics.kappa_inv_m,
            f"{state_label}.atmosphere_debye_exit_rate_s_inv",
        )
    )
    return _positive_float(exit_rate_s_inv, f"{state_label}.k_exit_s_inv")


def _debye_screening_rate_s_inv(
    diffusivity_m2_s: float,
    kappa_inv_m: float,
    context: str,
) -> float:
    if math.isinf(kappa_inv_m):
        raise ValueError(f"{context}.kappa_inv_m must be finite")
    kappa_m_inv = 1.0 / _positive_float(kappa_inv_m, f"{context}.kappa_inv_m")
    return _positive_float(
        _positive_float(diffusivity_m2_s, f"{context}.diffusivity_m2_s")
        * kappa_m_inv
        * kappa_m_inv,
        context,
    )


def _neutral_markov_process_from_projected_transport_states(
    projected_transport_states: tuple[ProjectedTransportState, ...],
    options: MolecularMoriOptions,
    temperature_K: float,
) -> _MarkovProcessConstruction:
    state_labels = tuple(
        f"{projected_transport_state.label}:mobile"
        for projected_transport_state in projected_transport_states
    )
    state_concentrations = np.asarray(
        [
            projected_transport_state.concentration_mol_m3
            for projected_transport_state in projected_transport_states
        ],
        dtype=float,
    )
    events: list[MarkovAdditiveEvent] = []
    for state_index, state in enumerate(projected_transport_states):
        jump_length_m = _jump_length_m(state, options)
        rate_s_inv = state.diffusion_m2_s / (jump_length_m * jump_length_m)
        events.append(
            MarkovAdditiveEvent(
                from_state_index=state_index,
                to_state_index=state_index,
                rate_s_inv=rate_s_inv,
                charge_displacement_m=(0.0, 0.0, 0.0),
                charge_displacement_second_moment_m2=ZERO_SECOND_MOMENT_TENSOR_M2,
                label=f"neutral_translation:{state.label}",
                family_label="neutral_translation",
            )
        )
    return _MarkovProcessConstruction(
        state_labels=state_labels,
        state_concentrations_mol_m3=state_concentrations,
        projected_transport_states=projected_transport_states,
        events=tuple(events),
        memory_primitives=tuple(),
        projected_current_memory_corrections=tuple(),
        atmosphere_mori_corrections=tuple(),
    )


def _jump_length_m(
    transport_state: _TransportKineticLike,
    options: MolecularMoriOptions,
) -> float:
    jump_length_m = (
        options.translation_jump_length_multiplier
        * options.primitive_parameters.jump_length_scale
        * transport_state.hydrodynamic_radius_A
        * ANGSTROM_TO_M
    )
    return _positive_float(jump_length_m, f"{transport_state.label}.jump_length_m")


def _diffusion_m2_s(
    hydrodynamic_radius_A: float,
    shape_factor: float,
    intrinsic_dielectric_constant: float,
    net_charge_number: int,
    charge_cloud_radius_A: float,
    charge_density_reference_A_inv3: float,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> float:
    free_volume_factor = _free_volume_friction_factor(
        solvent_environment.hard_sphere_volume_fraction,
        options,
    )
    shape_friction_factor = _shape_friction_factor(shape_factor, options)
    dielectric_mobility_factor = _dielectric_mobility_friction_factor(
        solvent_environment,
        options,
    )
    solvation_mobility_factor = _solvation_mobility_friction_factor(
        shape_factor,
        mixture_descriptor_state,
        options,
    )
    charge_density_mobility_factor = _charge_density_mobility_friction_factor(
        net_charge_number,
        charge_cloud_radius_A,
        charge_density_reference_A_inv3,
        options,
    )
    charge_cloud_extent_mobility_factor = _charge_cloud_extent_mobility_friction_factor(
        net_charge_number,
        hydrodynamic_radius_A,
        charge_cloud_radius_A,
        mixture_descriptor_state,
        options,
    )
    intrinsic_dielectric_drag_factor = (
        _negative_ion_intrinsic_dielectric_drag_mobility_friction_factor(
            net_charge_number,
            intrinsic_dielectric_constant,
            options,
        )
    )
    shape_delocalization_factor = (
        _negative_ion_shape_delocalization_mobility_friction_factor(
            net_charge_number,
            shape_factor,
            options,
        )
    )
    anion_composition_disorder_factor = (
        _anion_composition_disorder_mobility_friction_factor(
            net_charge_number,
            mixture_descriptor_state,
            options,
        )
    )
    viscosity_Pa_s = solvent_environment.viscosity_cP * CP_TO_PA_S
    radius_m = (
        _positive_float(hydrodynamic_radius_A, "hydrodynamic_radius_A") * ANGSTROM_TO_M
    )
    denominator = (
        STOKES_DENOMINATOR_FACTOR
        * math.pi
        * viscosity_Pa_s
        * radius_m
        * shape_friction_factor
        * free_volume_factor
        * dielectric_mobility_factor
        * solvation_mobility_factor
        * charge_density_mobility_factor
        * charge_cloud_extent_mobility_factor
        * intrinsic_dielectric_drag_factor
        * shape_delocalization_factor
        * anion_composition_disorder_factor
    )
    return float(K_B * solvent_environment.temperature_K / denominator)


def _shape_friction_factor(
    shape_factor: float,
    options: MolecularMoriOptions,
) -> float:
    descriptor_shape_factor = _positive_float(shape_factor, "shape_factor")
    shape_friction_exponent = _positive_float(
        options.primitive_parameters.shape_friction_exponent,
        "shape_friction_exponent",
    )
    return _positive_float(
        descriptor_shape_factor**shape_friction_exponent,
        "shape_friction_factor",
    )


def _dielectric_mobility_friction_factor(
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> float:
    dielectric_constant = _positive_float(
        solvent_environment.dielectric_constant,
        "solvent_environment.dielectric_constant",
    )
    dielectric_mobility_exponent = _positive_float(
        options.primitive_parameters.dielectric_mobility_exponent,
        "dielectric_mobility_exponent",
    )
    return _positive_float(
        dielectric_constant ** (-dielectric_mobility_exponent),
        "dielectric_mobility_friction_factor",
    )


def _solvation_mobility_friction_factor(
    shape_factor: float,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    options: MolecularMoriOptions,
) -> float:
    solvation_mobility_base = _positive_float(
        mixture_descriptor_state.solvation_obstruction_factor,
        "mixture.solvation_obstruction_factor",
    ) * _positive_float(
        mixture_descriptor_state.additive_solvation_obstruction_factor,
        "mixture.additive_solvation_obstruction_factor",
    )
    shape_anisotropy = (
        _positive_float(shape_factor, "shape_factor") - ISOTROPIC_SHAPE_FACTOR
    )
    additive_shape_exponent = (
        options.primitive_parameters.additive_shape_solvation_mobility_exponent
        * shape_anisotropy
        * shape_anisotropy
    )
    return _positive_float(
        solvation_mobility_base
        ** options.primitive_parameters.solvation_mobility_exponent,
        "solvation_mobility_friction_factor",
    ) * _positive_float(
        mixture_descriptor_state.additive_solvation_obstruction_factor
        ** additive_shape_exponent,
        "additive_shape_solvation_mobility_friction_factor",
    )


def _charge_density_mobility_friction_factor(
    net_charge_number: int,
    charge_cloud_radius_A: float,
    charge_density_reference_A_inv3: float,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number == 0:
        return 1.0
    charge_density_reference = _positive_float(
        charge_density_reference_A_inv3,
        "charge_density_reference_A_inv3",
    )
    charge_cloud_radius = _positive_float(
        charge_cloud_radius_A,
        "charge_cloud_radius_A",
    )
    state_charge_density_A_inv3 = abs(net_charge_number) / (
        charge_cloud_radius * charge_cloud_radius * charge_cloud_radius
    )
    normalized_charge_density = state_charge_density_A_inv3 / charge_density_reference
    if net_charge_number > 0:
        charge_density_mobility_exponent = (
            options.primitive_parameters.positive_ion_charge_density_mobility_exponent
        )
    else:
        charge_density_mobility_exponent = (
            options.primitive_parameters.negative_ion_charge_density_mobility_exponent
        )
    return _positive_float(
        normalized_charge_density**charge_density_mobility_exponent,
        "charge_density_mobility_friction_factor",
    )


def _charge_cloud_extent_mobility_friction_factor(
    net_charge_number: int,
    hydrodynamic_radius_A: float,
    charge_cloud_radius_A: float,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number == 0:
        return 1.0
    hydrodynamic_radius = _positive_float(
        hydrodynamic_radius_A,
        "hydrodynamic_radius_A",
    )
    if net_charge_number > 0:
        effective_charge_cloud_radius_A = _positive_float(
            mixture_descriptor_state.mean_anion_charge_cloud_radius_A,
            "mixture.mean_anion_charge_cloud_radius_A",
        )
        mobility_exponent = _positive_float(
            options.primitive_parameters.positive_ion_counteranion_charge_cloud_mobility_exponent,
            "positive_ion_counteranion_charge_cloud_mobility_exponent",
        )
    else:
        effective_charge_cloud_radius_A = _positive_float(
            charge_cloud_radius_A,
            "charge_cloud_radius_A",
        )
        mobility_exponent = _positive_float(
            options.primitive_parameters.negative_ion_charge_cloud_mobility_exponent,
            "negative_ion_charge_cloud_mobility_exponent",
        )
    charge_cloud_extent = 1.0 + effective_charge_cloud_radius_A / hydrodynamic_radius
    return _positive_float(
        charge_cloud_extent ** (-mobility_exponent),
        "charge_cloud_extent_mobility_friction_factor",
    )


def _negative_ion_intrinsic_dielectric_drag_mobility_friction_factor(
    net_charge_number: int,
    intrinsic_dielectric_constant: float,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number >= 0:
        return 1.0
    dielectric_constant = _positive_float(
        intrinsic_dielectric_constant,
        "intrinsic_dielectric_constant",
    )
    mobility_exponent = _positive_float(
        options.primitive_parameters.negative_ion_intrinsic_dielectric_drag_mobility_exponent,
        "negative_ion_intrinsic_dielectric_drag_mobility_exponent",
    )
    return _positive_float(
        (1.0 + dielectric_constant) ** mobility_exponent,
        "negative_ion_intrinsic_dielectric_drag_mobility_friction_factor",
    )


def _negative_ion_shape_delocalization_mobility_friction_factor(
    net_charge_number: int,
    shape_factor: float,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number >= 0:
        return 1.0
    descriptor_shape_factor = _positive_float(shape_factor, "shape_factor")
    mobility_exponent = _positive_float(
        options.primitive_parameters.negative_ion_shape_delocalization_mobility_exponent,
        "negative_ion_shape_delocalization_mobility_exponent",
    )
    return _positive_float(
        descriptor_shape_factor ** (-mobility_exponent),
        "negative_ion_shape_delocalization_mobility_friction_factor",
    )


def _anion_composition_disorder_mobility_friction_factor(
    net_charge_number: int,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number == 0:
        return 1.0
    anion_composition_entropy = _nonnegative_float(
        mixture_descriptor_state.anion_composition_entropy,
        "mixture.anion_composition_entropy",
    )
    if net_charge_number > 0:
        mobility_exponent = _positive_float(
            options.primitive_parameters.positive_ion_anion_disorder_mobility_exponent,
            "positive_ion_anion_disorder_mobility_exponent",
        )
    else:
        mobility_exponent = _positive_float(
            options.primitive_parameters.negative_ion_anion_disorder_mobility_exponent,
            "negative_ion_anion_disorder_mobility_exponent",
        )
    return _positive_float(
        math.exp(-mobility_exponent * anion_composition_entropy),
        "anion_composition_disorder_mobility_friction_factor",
    )


def _free_volume_friction_factor(
    hard_sphere_volume_fraction: float,
    options: MolecularMoriOptions,
) -> float:
    if hard_sphere_volume_fraction >= options.max_packing_fraction:
        raise ValueError(
            "hard_sphere_volume_fraction must be below max_packing_fraction"
        )
    remaining_free_volume = (
        options.max_packing_fraction - hard_sphere_volume_fraction
    ) / options.max_packing_fraction
    effective_free_volume_exponent = (
        options.free_volume_exponent * options.primitive_parameters.free_volume_exponent
    )
    return float(remaining_free_volume ** (-effective_free_volume_exponent))


def _molecular_mixture_descriptor_state(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> _MolecularMixtureDescriptorState:
    density_packing_scale = _density_packing_scale(recipe, descriptors)
    total_anion_concentration_M = 0.0
    anion_charge_cloud_weighted_sum_A = 0.0
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        if descriptor.charge_number >= 0:
            raise ValueError(f"{species_name} must be an anion")
        anion_concentration_M = _positive_float(
            concentration_M,
            f"{species_name}.concentration_M",
        )
        total_anion_concentration_M += anion_concentration_M
        anion_charge_cloud_weighted_sum_A += anion_concentration_M * _positive_float(
            descriptor.charge_cloud_radius_A,
            f"{species_name}.charge_cloud_radius_A",
        )
    if total_anion_concentration_M > 0.0:
        mean_anion_charge_cloud_radius_A = (
            anion_charge_cloud_weighted_sum_A / total_anion_concentration_M
        )
    else:
        mean_anion_charge_cloud_radius_A = _positive_float(
            solvent_environment.solvent_effective_radius_A,
            "solvent_environment.solvent_effective_radius_A",
        )
    anion_composition_entropy = 0.0
    for species_name, concentration_M in recipe.anions.items():
        anion_mole_fraction = _positive_float(
            concentration_M, f"{species_name}.concentration_M"
        ) / _positive_float(
            total_anion_concentration_M,
            "mixture.total_anion_concentration_M",
        )
        anion_composition_entropy -= anion_mole_fraction * math.log(anion_mole_fraction)
    total_concentration_mol_m3 = 0.0
    donor_number_weighted_sum = 0.0
    acceptor_number_weighted_sum = 0.0
    polarizability_weighted_sum_A3 = 0.0
    molecular_volume_weighted_sum_A3 = 0.0
    additive_solvation_support = 0.0
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        (
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        ) = _accumulate_mixture_descriptor_weights(
            concentration_mol_m3,
            descriptor,
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        )
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        (
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        ) = _accumulate_mixture_descriptor_weights(
            concentration_mol_m3,
            descriptor,
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        )
    for species_name, volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            volume_fraction * density_packing_scale,
            descriptor,
        )
        (
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        ) = _accumulate_mixture_descriptor_weights(
            concentration_mol_m3,
            descriptor,
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        )
    for species_name, weight_fraction in recipe.additives.items():
        descriptor = descriptors[species_name]
        positive_weight_fraction = _positive_float(
            weight_fraction,
            f"{species_name}.weight_fraction",
        )
        additive_solvation_support += (
            positive_weight_fraction * _additive_solvation_support(descriptor)
        )
        concentration_mol_m3 = (
            positive_weight_fraction
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol
        )
        (
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        ) = _accumulate_mixture_descriptor_weights(
            concentration_mol_m3,
            descriptor,
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        )
    _positive_float(total_concentration_mol_m3, "mixture.total_concentration_mol_m3")
    available_free_volume_fraction = _positive_float(
        options.max_packing_fraction, "max_packing_fraction"
    ) - _nonnegative_float(
        solvent_environment.hard_sphere_volume_fraction,
        "hard_sphere_volume_fraction",
    )
    if available_free_volume_fraction <= 0.0:
        raise ValueError("available_free_volume_fraction must be positive")
    total_number_density_m3 = N_A * total_concentration_mol_m3
    free_volume_per_particle_m3 = (
        available_free_volume_fraction / total_number_density_m3
    )
    void_radius_A = (3.0 * free_volume_per_particle_m3 / (4.0 * math.pi)) ** (
        1.0 / 3.0
    ) / ANGSTROM_TO_M
    donor_number = donor_number_weighted_sum / total_concentration_mol_m3
    acceptor_number = acceptor_number_weighted_sum / total_concentration_mol_m3
    polarizability_volume_ratio = polarizability_weighted_sum_A3 / _positive_float(
        molecular_volume_weighted_sum_A3,
        "mixture.molecular_volume_weighted_sum_A3",
    )
    solvation_support = (
        _nonnegative_float(donor_number, "mixture.donor_number")
        + _nonnegative_float(acceptor_number, "mixture.acceptor_number")
        + _nonnegative_float(
            polarizability_volume_ratio,
            "mixture.polarizability_volume_ratio",
        )
    )
    solvation_obstruction_factor = 1.0 / (1.0 + solvation_support)
    additive_solvation_obstruction_factor = 1.0 / (
        1.0
        + _nonnegative_float(
            additive_solvation_support,
            "mixture.additive_solvation_support",
        )
    )
    ionic_strength_mol_m3 = _analytical_ionic_strength_mol_m3(
        recipe,
        descriptors,
    )
    return _MolecularMixtureDescriptorState(
        hard_sphere_volume_fraction=float(
            solvent_environment.hard_sphere_volume_fraction
        ),
        max_packing_fraction=float(options.max_packing_fraction),
        ionic_strength_mol_m3=ionic_strength_mol_m3,
        void_radius_A=_positive_float(void_radius_A, "mixture.void_radius_A"),
        donor_number=float(donor_number),
        acceptor_number=float(acceptor_number),
        polarizability_volume_ratio=float(polarizability_volume_ratio),
        solvation_obstruction_factor=float(solvation_obstruction_factor),
        additive_solvation_obstruction_factor=float(
            additive_solvation_obstruction_factor
        ),
        mean_anion_charge_cloud_radius_A=_positive_float(
            mean_anion_charge_cloud_radius_A,
            "mixture.mean_anion_charge_cloud_radius_A",
        ),
        anion_composition_entropy=float(anion_composition_entropy),
    )


def _analytical_ionic_strength_mol_m3(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    ionic_strength_mol_m3 = 0.0
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        ionic_strength_mol_m3 += (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            * descriptor.charge_number
            * descriptor.charge_number
        )
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        ionic_strength_mol_m3 += (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            * descriptor.charge_number
            * descriptor.charge_number
        )
    return _nonnegative_float(ionic_strength_mol_m3, "ionic_strength_mol_m3")


def _accumulate_mixture_descriptor_weights(
    concentration_mol_m3: float,
    descriptor: MolecularSpeciesDescriptor,
    total_concentration_mol_m3: float,
    donor_number_weighted_sum: float,
    acceptor_number_weighted_sum: float,
    polarizability_weighted_sum_A3: float,
    molecular_volume_weighted_sum_A3: float,
) -> tuple[float, float, float, float, float]:
    positive_concentration_mol_m3 = _positive_float(
        concentration_mol_m3,
        f"{descriptor.name}.concentration_mol_m3",
    )
    return (
        total_concentration_mol_m3 + positive_concentration_mol_m3,
        donor_number_weighted_sum
        + positive_concentration_mol_m3 * descriptor.donor_number,
        acceptor_number_weighted_sum
        + positive_concentration_mol_m3 * descriptor.acceptor_number,
        polarizability_weighted_sum_A3
        + positive_concentration_mol_m3 * descriptor.polarizability_A3,
        molecular_volume_weighted_sum_A3
        + positive_concentration_mol_m3 * descriptor.molecular_volume_A3,
    )


def _additive_solvation_support(
    descriptor: MolecularSpeciesDescriptor,
) -> float:
    polarizability_volume_ratio = _nonnegative_float(
        descriptor.polarizability_A3,
        f"{descriptor.name}.polarizability_A3",
    ) / _positive_float(
        descriptor.molecular_volume_A3,
        f"{descriptor.name}.molecular_volume_A3",
    )
    return _nonnegative_float(
        _nonnegative_float(descriptor.donor_number, f"{descriptor.name}.donor_number")
        + _nonnegative_float(
            descriptor.acceptor_number,
            f"{descriptor.name}.acceptor_number",
        )
        + _nonnegative_float(
            float(descriptor.hbond_acceptor_count),
            f"{descriptor.name}.hbond_acceptor_count",
        )
        + polarizability_volume_ratio,
        f"{descriptor.name}.additive_solvation_support",
    )


def _charge_density_reference_A_inv3(
    speciation: GenericSpeciationResult,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
    options: MolecularMoriOptions,
) -> float:
    entries: list[_ChargeDensityReferenceEntry] = []
    for component in speciation.components:
        concentration_mol_m3 = speciation.free_component_concentrations_mol_m3[
            component.species_name
        ]
        entries.append(
            _ChargeDensityReferenceEntry(
                concentration_mol_m3=concentration_mol_m3,
                net_charge_number=component.charge_number,
                charge_cloud_radius_A=_scaled_charge_cloud_radius_A(
                    component.descriptor,
                    options,
                ),
            )
        )
    for cluster_template in speciation.cluster_templates:
        entries.append(
            _ChargeDensityReferenceEntry(
                concentration_mol_m3=speciation.cluster_concentrations_mol_m3[
                    cluster_template.label
                ],
                net_charge_number=cluster_template.net_charge_number,
                charge_cloud_radius_A=_cluster_charge_cloud_radius_A(
                    cluster_template,
                    component_descriptor_by_name,
                    options,
                ),
            )
        )
    concentration_weighted_charge_density = 0.0
    concentration_weight = 0.0
    for entry in entries:
        if entry.net_charge_number == 0:
            continue
        concentration_mol_m3 = _positive_float(
            entry.concentration_mol_m3,
            "charge_density_reference.concentration_mol_m3",
        )
        charge_cloud_radius_A = _positive_float(
            entry.charge_cloud_radius_A,
            "charge_density_reference.charge_cloud_radius_A",
        )
        concentration_weight += concentration_mol_m3
        concentration_weighted_charge_density += (
            concentration_mol_m3
            * abs(entry.net_charge_number)
            / (charge_cloud_radius_A**3)
        )
    if concentration_weight <= 0.0:
        return 0.0
    return _positive_float(
        concentration_weighted_charge_density / concentration_weight,
        "charge_density_reference_A_inv3",
    )


def _scaled_charge_cloud_radius_A(
    descriptor: MolecularSpeciesDescriptor,
    options: MolecularMoriOptions,
) -> float:
    return options.primitive_parameters.charge_cloud_radius_scale * _positive_float(
        descriptor.charge_cloud_radius_A,
        f"{descriptor.name}.charge_cloud_radius_A",
    )


def _cluster_charge_cloud_radius_A(
    cluster_template: ClusterStateTemplate,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
    options: MolecularMoriOptions,
) -> float:
    charge_weighted_radius_squared_sum = 0.0
    charge_weight_sum = 0.0
    for species_name, stoichiometric_count in cluster_template.stoichiometry.items():
        descriptor = component_descriptor_by_name[species_name]
        charge_weight = abs(descriptor.charge_number) * stoichiometric_count
        if charge_weight == 0:
            continue
        charge_cloud_radius_A = _scaled_charge_cloud_radius_A(descriptor, options)
        charge_weight_sum += charge_weight
        charge_weighted_radius_squared_sum += (
            charge_weight * charge_cloud_radius_A * charge_cloud_radius_A
        )
    if charge_weight_sum <= 0.0:
        raise ValueError(f"{cluster_template.label} charge weight must be positive")
    return _positive_float(
        math.sqrt(charge_weighted_radius_squared_sum / charge_weight_sum),
        f"{cluster_template.label}.charge_cloud_radius_A",
    )


def _cluster_intrinsic_dielectric_constant(
    cluster_template: ClusterStateTemplate,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    stoichiometric_count_sum = 0.0
    dielectric_weighted_sum = 0.0
    for species_name, stoichiometric_count in cluster_template.stoichiometry.items():
        descriptor = component_descriptor_by_name[species_name]
        positive_count = _positive_float(
            stoichiometric_count,
            f"{cluster_template.label}.{species_name}.stoichiometric_count",
        )
        stoichiometric_count_sum += positive_count
        dielectric_weighted_sum += positive_count * _positive_float(
            descriptor.epsilon_r_pure,
            f"{species_name}.epsilon_r_pure",
        )
    return _positive_float(
        dielectric_weighted_sum
        / _positive_float(
            stoichiometric_count_sum,
            f"{cluster_template.label}.stoichiometric_count_sum",
        ),
        f"{cluster_template.label}.intrinsic_dielectric_constant",
    )


def _local_obstruction_factor(
    label: str,
    net_charge_number: int,
    hydrodynamic_radius_A: float,
    charge_cloud_radius_A: float,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    options: MolecularMoriOptions,
    charge_density_reference_A_inv3: float,
) -> float:
    if net_charge_number == 0:
        return 0.0
    charge_density_reference = _positive_float(
        charge_density_reference_A_inv3,
        "charge_density_reference_A_inv3",
    )
    packing_denominator = _positive_float(
        mixture_descriptor_state.max_packing_fraction,
        "max_packing_fraction",
    ) - _nonnegative_float(
        mixture_descriptor_state.hard_sphere_volume_fraction,
        "hard_sphere_volume_fraction",
    )
    if packing_denominator <= 0.0:
        raise ValueError(f"{label}.packing_denominator must be positive")
    free_volume_ratio = (
        mixture_descriptor_state.hard_sphere_volume_fraction / packing_denominator
    )
    size_ratio = _positive_float(
        hydrodynamic_radius_A, f"{label}.hydrodynamic_radius_A"
    ) / _positive_float(mixture_descriptor_state.void_radius_A, "void_radius_A")
    state_charge_density_A_inv3 = abs(net_charge_number) / (
        _positive_float(charge_cloud_radius_A, f"{label}.charge_cloud_radius_A") ** 3
    )
    normalized_charge_density = state_charge_density_A_inv3 / charge_density_reference
    ionic_strength_ratio = (
        _nonnegative_float(
            mixture_descriptor_state.ionic_strength_mol_m3,
            "ionic_strength_mol_m3",
        )
        / STANDARD_STATE_CONCENTRATION_MOL_M3
    )
    ionic_strength_crowding_factor = (1.0 + ionic_strength_ratio) / 2.0
    obstruction_factor = (
        options.primitive_parameters.local_obstruction_strength
        * (
            free_volume_ratio
            ** options.primitive_parameters.local_obstruction_free_volume_exponent
        )
        * (
            ionic_strength_crowding_factor
            ** options.primitive_parameters.local_obstruction_ionic_strength_exponent
        )
        * (
            mixture_descriptor_state.additive_solvation_obstruction_factor
            ** options.primitive_parameters.local_obstruction_additive_solvation_exponent
        )
        * (size_ratio**options.primitive_parameters.local_obstruction_size_exponent)
        * (
            normalized_charge_density
            ** options.primitive_parameters.local_obstruction_charge_density_exponent
        )
        * (
            mixture_descriptor_state.solvation_obstruction_factor
            ** options.primitive_parameters.local_obstruction_solvation_exponent
        )
    )
    return _nonnegative_float(obstruction_factor, f"{label}.local_obstruction_factor")


def _local_obstruction_diffusion_scale(
    local_obstruction_factor: float,
    label: str,
) -> float:
    obstruction_factor = _nonnegative_float(
        local_obstruction_factor,
        f"{label}.local_obstruction_factor",
    )
    return _positive_float(
        1.0 / (1.0 + obstruction_factor),
        f"{label}.local_obstruction_diffusion_scale",
    )


def _hard_sphere_volume_fraction(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    density_packing_scale = _density_packing_scale(recipe, descriptors)
    volume_fraction = 0.0
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        volume_fraction += _species_volume_fraction_from_molarity(
            concentration_M,
            descriptor,
        )
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        volume_fraction += _species_volume_fraction_from_molarity(
            concentration_M,
            descriptor,
        )
    for species_name, solvent_volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            solvent_volume_fraction * density_packing_scale,
            descriptor,
        )
        volume_fraction += _species_volume_fraction_from_concentration(
            concentration_mol_m3,
            descriptor,
        )
    for species_name, additive_weight_fraction in recipe.additives.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = (
            _positive_float(
                additive_weight_fraction,
                f"{species_name}.weight_fraction",
            )
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol
        )
        volume_fraction += _species_volume_fraction_from_concentration(
            concentration_mol_m3,
            descriptor,
        )
    return float(volume_fraction)


def _mixture_effective_radius_A(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    mean_molecular_volume_A3 = _mixture_mean_molecular_volume_A3(
        recipe,
        descriptors,
    )
    return _positive_float(
        (3.0 * mean_molecular_volume_A3 / (4.0 * math.pi)) ** (1.0 / 3.0),
        "mixture_effective_radius_A",
    )


def _mixture_mean_molecular_volume_A3(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    density_packing_scale = _density_packing_scale(recipe, descriptors)
    weighted_volume_A3 = 0.0
    concentration_weight = 0.0
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        species_concentration_mol_m3 = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        weighted_volume_A3 += (
            species_concentration_mol_m3 * descriptor.molecular_volume_A3
        )
        concentration_weight += species_concentration_mol_m3
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        species_concentration_mol_m3 = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        weighted_volume_A3 += (
            species_concentration_mol_m3 * descriptor.molecular_volume_A3
        )
        concentration_weight += species_concentration_mol_m3
    for species_name, solvent_volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        species_concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            solvent_volume_fraction * density_packing_scale,
            descriptor,
        )
        weighted_volume_A3 += (
            species_concentration_mol_m3 * descriptor.molecular_volume_A3
        )
        concentration_weight += species_concentration_mol_m3
    for species_name, additive_weight_fraction in recipe.additives.items():
        descriptor = descriptors[species_name]
        species_concentration_mol_m3 = (
            _positive_float(
                additive_weight_fraction,
                f"{species_name}.weight_fraction",
            )
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol
        )
        weighted_volume_A3 += (
            species_concentration_mol_m3 * descriptor.molecular_volume_A3
        )
        concentration_weight += species_concentration_mol_m3
    return _positive_float(
        weighted_volume_A3 / concentration_weight,
        "mixture_mean_molecular_volume_A3",
    )


def _mixture_solvent_coordination_affinity_J_mol(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    solvent_fraction_sum = math.fsum(
        _nonnegative_float(volume_fraction, f"{species_name}.volume_fraction")
        for species_name, volume_fraction in recipe.solvents.items()
    )
    _positive_float(solvent_fraction_sum, "solvent_volume_fraction_sum")
    weighted_coordination_affinity_J_mol = 0.0
    for species_name, volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        weighted_coordination_affinity_J_mol += _nonnegative_float(
            volume_fraction, f"{species_name}.volume_fraction"
        ) * _nonnegative_float(
            descriptor.coordination_affinity_J_mol,
            f"{species_name}.coordination_affinity_J_mol",
        )
    return _nonnegative_float(
        weighted_coordination_affinity_J_mol / solvent_fraction_sum,
        "mixture_solvent_coordination_affinity_J_mol",
    )


def _density_packing_scale(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    ideal_density_g_ml = _ideal_recipe_density_g_ml(recipe, descriptors)
    measured_density_g_ml = _positive_float(
        recipe.mixture_properties.density_g_ml,
        "recipe.mixture_properties.density_g_ml",
    )
    return _positive_float(
        measured_density_g_ml / ideal_density_g_ml,
        "density_packing_scale",
    )


def _ideal_recipe_density_g_ml(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    solvent_mass_g_per_liter = math.fsum(
        _positive_float(volume_fraction, f"{species_name}.volume_fraction")
        * _positive_float(
            descriptors[species_name].density_g_ml,
            f"{species_name}.density_g_ml",
        )
        * GRAMS_PER_LITER_PER_G_ML
        for species_name, volume_fraction in recipe.solvents.items()
    )
    cation_mass_g_per_liter = math.fsum(
        _positive_float(concentration_M, f"{species_name}.concentration_M")
        * _positive_float(
            descriptors[species_name].molecular_weight_g_mol,
            f"{species_name}.molecular_weight_g_mol",
        )
        for species_name, concentration_M in recipe.cations.items()
    )
    anion_mass_g_per_liter = math.fsum(
        _positive_float(concentration_M, f"{species_name}.concentration_M")
        * _positive_float(
            descriptors[species_name].molecular_weight_g_mol,
            f"{species_name}.molecular_weight_g_mol",
        )
        for species_name, concentration_M in recipe.anions.items()
    )
    total_neutral_additive_weight_fraction = math.fsum(
        _positive_float(weight_fraction, f"{species_name}.weight_fraction")
        for species_name, weight_fraction in recipe.additives.items()
    )
    if total_neutral_additive_weight_fraction >= 1.0:
        raise ValueError("total neutral additive weight fraction must be below one")
    base_mass_g_per_liter = (
        solvent_mass_g_per_liter + cation_mass_g_per_liter + anion_mass_g_per_liter
    )
    total_mass_g_per_liter = base_mass_g_per_liter / (
        1.0 - total_neutral_additive_weight_fraction
    )
    return _positive_float(
        total_mass_g_per_liter / GRAMS_PER_LITER_PER_G_ML,
        "ideal_recipe_density_g_ml",
    )


def _species_volume_fraction_from_molarity(
    concentration_M: float,
    descriptor: MolecularSpeciesDescriptor,
) -> float:
    concentration_mol_m3 = (
        _positive_float(concentration_M, f"{descriptor.name}.concentration_M")
        * STANDARD_STATE_CONCENTRATION_MOL_M3
    )
    return _species_volume_fraction_from_concentration(
        concentration_mol_m3,
        descriptor,
    )


def _species_volume_fraction_from_concentration(
    concentration_mol_m3: float,
    descriptor: MolecularSpeciesDescriptor,
) -> float:
    return float(
        concentration_mol_m3 * N_A * descriptor.molecular_volume_A3 * ANGSTROM3_TO_M3
    )


def _liquid_component_concentration_mol_m3(
    volume_fraction: float,
    descriptor: MolecularSpeciesDescriptor,
) -> float:
    return float(
        _positive_float(volume_fraction, f"{descriptor.name}.volume_fraction")
        * descriptor.density_g_ml
        * GRAMS_PER_M3_PER_G_ML
        / descriptor.molecular_weight_g_mol
    )


def _cluster_shape_factor(
    cluster_template: ClusterStateTemplate,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    return float(
        max(
            component_descriptor_by_name[species_name].ligand_field_asymmetry
            for species_name in cluster_template.stoichiometry
        )
    )


def _validate_recipe(recipe: MolecularElectrolyteRecipe) -> None:
    _positive_float(recipe.temperature_K, "temperature_K")
    _positive_float(recipe.pressure_Pa, "pressure_Pa")
    _positive_float(recipe.mixture_properties.density_g_ml, "mixture.density_g_ml")
    _positive_float(recipe.mixture_properties.viscosity_cP, "mixture.viscosity_cP")
    _positive_float(
        recipe.mixture_properties.dielectric_constant,
        "mixture.dielectric_constant",
    )
    for species_name, volume_fraction in recipe.solvents.items():
        _positive_float(volume_fraction, f"{species_name}.volume_fraction")
    for species_name, weight_fraction in recipe.additives.items():
        _positive_float(weight_fraction, f"{species_name}.weight_fraction")


def _validate_options(options: MolecularMoriOptions) -> None:
    if options.max_cluster_ion_count < MINIMUM_CLUSTER_ION_COUNT:
        raise ValueError(
            "max_cluster_ion_count must include at least cation-anion pair states"
        )
    validate_conductivity_primitive_parameters(options.primitive_parameters)
    _positive_float(options.max_packing_fraction, "max_packing_fraction")
    _nonnegative_float(options.free_volume_exponent, "free_volume_exponent")
    _positive_float(
        options.translation_jump_length_multiplier,
        "translation_jump_length_multiplier",
    )


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _finite_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{context} must be finite")
    return parsed_value


def _nonnegative_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        raise ValueError(f"{context} must be nonnegative and finite")
    return parsed_value


@dataclass(frozen=True)
class AnalyticalConductivityModelInput:
    recipe: MolecularElectrolyteRecipe
    species_inputs: Mapping[str, MolecularSpeciesInput]
    descriptor_backend: MolecularDescriptorBackend
    options: MolecularMoriOptions


AnalyticalConductivityModelResult = MolecularMoriConductivityResult


def compute_analytical_conductivity_model(
    model_input: AnalyticalConductivityModelInput,
) -> AnalyticalConductivityModelResult:
    validate_conductivity_primitive_parameters(
        model_input.options.primitive_parameters,
    )
    return compute_molecular_electrolyte_conductivity(
        model_input.recipe,
        model_input.species_inputs,
        model_input.descriptor_backend,
        model_input.options,
    )


def compute_analytical_conductivity_model_with_diagnostic_cluster_shifts(
    model_input: AnalyticalConductivityModelInput,
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
) -> AnalyticalConductivityModelResult:
    validate_conductivity_primitive_parameters(
        model_input.options.primitive_parameters,
    )
    return compute_molecular_electrolyte_conductivity_with_diagnostic_cluster_shifts(
        model_input.recipe,
        model_input.species_inputs,
        model_input.descriptor_backend,
        model_input.options,
        diagnostic_cluster_standard_free_energy_shift_over_RT_by_label,
    )
