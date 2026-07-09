"""Complete standalone projected analytical conductivity estimator.

Single production model:

    U(q), D(q), P(q), basins A_i, transition surfaces, transition-path dP,
    memory coordinates psi(q)
        -> c_i, K_ij, Q_ij, d_ij, M_ij, D_self_i, A, h
        -> sigma

The module contains no composition shortcut, no Green-Kubo/EH production
estimator, and no empirical correction path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from constants import (
    F,
    R,
    S_M_TO_MS_CM,
    T_REF_K,
)

F_C_PER_MOL = F
R_J_PER_MOL_K = R
CARTESIAN = 3
DEFAULT_MAX_TRANSITION_DISPLACEMENT_M = 1.0e-8
DEFAULT_FINITE_DIFFERENCE_REL_STEP = np.finfo(float).eps ** (1.0 / 3.0)
PSD_TOL = 1.0e-12
PROJECTED_REFERENCE_VOLUME_M3 = 1.0
STANDARD_CONCENTRATION_MOL_M3 = 1000.0
CHEMICAL_POTENTIAL_MASS_TOL = 1.0e-8
CHEMICAL_POTENTIAL_SOLVER_TOL = 1.0e-12
CHEMICAL_POTENTIAL_MAX_ITERATIONS = 1000
CHEMICAL_POTENTIAL_MIN_CONCENTRATION_MOL_M3 = 1.0e-30
CHEMICAL_POTENTIAL_TIKHONOV_RELATIVE_SCALE = 1.0e-12
CHEMICAL_POTENTIAL_MIN_LINESEARCH_ALPHA = float.fromhex(
    "0x1p-40"
)  # Numerical line-search sentinel from the projected-model specification.
FAILED_CHEMICAL_POTENTIAL_RESIDUAL = np.sqrt(np.finfo(float).max)
GENERATOR_BALANCE_TOL = 1.0e-10
PSEUDOINVERSE_RELATIVE_TOL = 1.0e-12
MEMORY_NULLSPACE_RELATIVE_TOL = 1.0e-8
POISSON_SOLVABILITY_ABS_TOL = 1.0e-18  # Numerical zero for c-weighted drift in one disconnected generator component.
POISSON_SOLVABILITY_EPSILON_FACTOR = (
    100.0  # Floating-point guard factor used by the Poisson component solvability test.
)
PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL = 1.0e-10
DIAGNOSTIC_TOP_RECORD_COUNT = (
    5  # Limit failure payload size while keeping dominant contributors visible.
)
PARTITION_RELATIVE_WEIGHT_FLOOR = np.finfo(float).tiny
LOG_FLOAT_MAX = np.log(np.finfo(float).max)
LOG_FLOAT_TINY = np.log(np.finfo(float).tiny)

Array = np.ndarray

FULL_LIBRARY_REQUIRED_SECTIONS = (
    "species",
    "interactions",
    "mixture",
    "mobility",
    "basis",
    "transition_memory",
    "projected_generator_inputs",
)

SPECIES_REQUIRED_FIELDS = (
    "role",
    "molecular_weight_kg_mol",
    "density_kg_m3",
    "partial_molar_volume_m3_mol",
    "sites",
    "bonds",
    "angles",
    "torsions",
    "constraints",
)

SITE_REQUIRED_FIELDS = (
    "site_id",
    "element",
    "mass_kg",
    "steric_radius_m",
    "hydrodynamic_radius_m",
    "volume_m3",
    "lj_sigma_m",
    "lj_epsilon_J",
    "charge_number",
    "charge_cloud_radius_m",
    "born_radius_m",
    "polarizability_SI",
    "donor_flag",
    "acceptor_flag",
)

PROJECTED_GENERATOR_REQUIRED_FIELDS = (
    "potential_energy_J_mol",
    "mobility_tensor_m2_s",
    "charge_polarization_gradient",
    "memory_coordinate_gradient",
    "basin_quadrature_points",
    "basin_quadrature_weights",
    "transition_pair_indices",
    "transition_quadrature_points",
    "transition_quadrature_weights",
    "transition_committor_gradients",
    "transition_surface_state_indices",
    "transition_path_displacements_m",
    "transition_path_weights",
    "transition_log_capacity_integrals",
    "transition_first_moments_d_ij_m",
    "transition_second_moments_M_ij_m2",
    "total_component_concentrations_mol_m3",
    "basin_stoichiometry",
    "volume_m3",
    "self_current_coordinate_projectors",
)

PROJECTED_PRIMITIVE_REQUIRED_FIELDS = (
    "state_concentrations_mol_m3",
    "symmetric_capacity_fluxes_K_ij_mol_m3_s",
    "transition_first_moments_d_ij_m",
    "transition_second_moments_M_ij_m2",
    "self_current_tensors_D_self_i_m2_s",
    "mori_memory_matrix_A",
    "mori_current_coupling_matrix_h",
    "temperature_K",
    "volume_m3",
)


class ProjectedGeneratorInput:
    def __init__(
        self,
        potential_energy_J_mol: Callable[[Array], float],
        mobility_tensor_m2_s: Callable[[Array], Array],
        charge_polarization_gradient: Callable[[Array], Array],
        memory_coordinate_gradient: Callable[[Array], Array],
        basin_quadrature_points: tuple[Array, ...],
        basin_quadrature_weights: tuple[Array, ...],
        transition_pair_indices: Array,
        transition_quadrature_points: tuple[Array, ...],
        transition_quadrature_weights: tuple[Array, ...],
        transition_committor_gradients: tuple[Array, ...],
        transition_surface_state_indices: tuple[Array, ...],
        transition_path_displacements_m: tuple[Array, ...],
        transition_path_weights: tuple[Array, ...],
        transition_log_capacity_integrals: Array,
        transition_uses_residence_rate_constants: Array,
        transition_residence_rate_constants_s_inv: Array,
        transition_first_moments_d_ij_m: Array,
        transition_second_moments_M_ij_m2: Array,
        total_component_concentrations_mol_m3: Array,
        basin_stoichiometry: Array,
        temperature_K: float,
        volume_m3: float,
        self_current_coordinate_projectors: tuple[Array, ...],
        max_transition_displacement_m: float = DEFAULT_MAX_TRANSITION_DISPLACEMENT_M,
    ) -> None:
        self.potential_energy_J_mol = potential_energy_J_mol
        self.mobility_tensor_m2_s = mobility_tensor_m2_s
        self.charge_polarization_gradient = charge_polarization_gradient
        self.memory_coordinate_gradient = memory_coordinate_gradient
        self.basin_quadrature_points = basin_quadrature_points
        self.basin_quadrature_weights = basin_quadrature_weights
        self.transition_pair_indices = transition_pair_indices
        self.transition_quadrature_points = transition_quadrature_points
        self.transition_quadrature_weights = transition_quadrature_weights
        self.transition_committor_gradients = transition_committor_gradients
        self.transition_surface_state_indices = transition_surface_state_indices
        self.transition_path_displacements_m = transition_path_displacements_m
        self.transition_path_weights = transition_path_weights
        self.transition_log_capacity_integrals = transition_log_capacity_integrals
        self.transition_uses_residence_rate_constants = (
            transition_uses_residence_rate_constants
        )
        self.transition_residence_rate_constants_s_inv = (
            transition_residence_rate_constants_s_inv
        )
        self.transition_first_moments_d_ij_m = transition_first_moments_d_ij_m
        self.transition_second_moments_M_ij_m2 = transition_second_moments_M_ij_m2
        self.total_component_concentrations_mol_m3 = (
            total_component_concentrations_mol_m3
        )
        self.basin_stoichiometry = basin_stoichiometry
        self.temperature_K = temperature_K
        self.volume_m3 = volume_m3
        self.self_current_coordinate_projectors = self_current_coordinate_projectors
        self.max_transition_displacement_m = max_transition_displacement_m


class FunctionGeneratorInput:
    def __init__(
        self,
        potential_energy_J_mol: Callable[[Array], float],
        mobility_tensor_m2_s: Callable[[Array], Array],
        charge_polarization: Callable[[Array], Array],
        memory_coordinates: Callable[[Array], Array],
        basin_quadrature_points: tuple[Array, ...],
        basin_quadrature_weights: tuple[Array, ...],
        transition_pair_indices: Array,
        transition_quadrature_points: tuple[Array, ...],
        transition_quadrature_weights: tuple[Array, ...],
        transition_committor_gradients: tuple[Array, ...],
        transition_surface_state_indices: tuple[Array, ...],
        transition_path_start_points: tuple[Array, ...],
        transition_path_end_points: tuple[Array, ...],
        transition_path_weights: tuple[Array, ...],
        total_component_concentrations_mol_m3: Array,
        basin_stoichiometry: Array,
        temperature_K: float,
        volume_m3: float,
        self_current_coordinate_projectors: tuple[Array, ...],
        finite_difference_relative_step: float = DEFAULT_FINITE_DIFFERENCE_REL_STEP,
        max_transition_displacement_m: float = DEFAULT_MAX_TRANSITION_DISPLACEMENT_M,
    ) -> None:
        self.potential_energy_J_mol = potential_energy_J_mol
        self.mobility_tensor_m2_s = mobility_tensor_m2_s
        self.charge_polarization = charge_polarization
        self.memory_coordinates = memory_coordinates
        self.basin_quadrature_points = basin_quadrature_points
        self.basin_quadrature_weights = basin_quadrature_weights
        self.transition_pair_indices = transition_pair_indices
        self.transition_quadrature_points = transition_quadrature_points
        self.transition_quadrature_weights = transition_quadrature_weights
        self.transition_committor_gradients = transition_committor_gradients
        self.transition_surface_state_indices = transition_surface_state_indices
        self.transition_path_start_points = transition_path_start_points
        self.transition_path_end_points = transition_path_end_points
        self.transition_path_weights = transition_path_weights
        self.total_component_concentrations_mol_m3 = (
            total_component_concentrations_mol_m3
        )
        self.basin_stoichiometry = basin_stoichiometry
        self.temperature_K = temperature_K
        self.volume_m3 = volume_m3
        self.self_current_coordinate_projectors = self_current_coordinate_projectors
        self.finite_difference_relative_step = finite_difference_relative_step
        self.max_transition_displacement_m = max_transition_displacement_m


class ConductivityPhysicalLibrary:
    def __init__(
        self,
        generator_input: ProjectedGeneratorInput,
        state_labels: tuple[str, ...],
        transition_labels: tuple[str, ...],
    ) -> None:
        self.generator_input = generator_input
        self.state_labels = state_labels
        self.transition_labels = transition_labels


class ProjectedPrimitiveInput:
    def __init__(
        self,
        state_concentrations_mol_m3: Array,
        symmetric_capacity_fluxes_K_ij_mol_m3_s: Array,
        transition_first_moments_d_ij_m: Array,
        transition_second_moments_M_ij_m2: Array,
        self_current_tensors_D_self_i_m2_s: Array,
        mori_memory_matrix_A: Array,
        mori_current_coupling_matrix_h: Array,
        temperature_K: float,
        volume_m3: float,
        max_transition_displacement_m: float = DEFAULT_MAX_TRANSITION_DISPLACEMENT_M,
    ) -> None:
        self.state_concentrations_mol_m3 = state_concentrations_mol_m3
        self.symmetric_capacity_fluxes_K_ij_mol_m3_s = (
            symmetric_capacity_fluxes_K_ij_mol_m3_s
        )
        self.transition_first_moments_d_ij_m = transition_first_moments_d_ij_m
        self.transition_second_moments_M_ij_m2 = transition_second_moments_M_ij_m2
        self.self_current_tensors_D_self_i_m2_s = self_current_tensors_D_self_i_m2_s
        self.mori_memory_matrix_A = mori_memory_matrix_A
        self.mori_current_coupling_matrix_h = mori_current_coupling_matrix_h
        self.temperature_K = temperature_K
        self.volume_m3 = volume_m3
        self.max_transition_displacement_m = max_transition_displacement_m


class ProjectedConductivityResult:
    def __init__(
        self,
        sigma_S_m: float,
        sigma_mS_cm: float,
        projected_diffusivity_tensor: Array,
        direct_diffusivity_tensor: Array,
        finite_state_memory_correction_tensor: Array,
        continuous_mori_correction_tensor: Array,
        state_concentrations_mol_m3: Array,
        symmetric_capacity_fluxes_K_ij_mol_m3_s: Array,
        reversible_generator_Q_ij_s_inv: Array,
        transition_first_moments_d_ij_m: Array,
        transition_second_moments_M_ij_m2: Array,
        self_current_tensors_D_self_i_m2_s: Array,
        mori_memory_matrix_A: Array,
        mori_current_coupling_matrix_h: Array,
        effect_attribution,
    ) -> None:
        self.sigma_S_m = sigma_S_m
        self.sigma_mS_cm = sigma_mS_cm
        self.projected_diffusivity_tensor = projected_diffusivity_tensor
        self.direct_diffusivity_tensor = direct_diffusivity_tensor
        self.finite_state_memory_correction_tensor = (
            finite_state_memory_correction_tensor
        )
        self.continuous_mori_correction_tensor = continuous_mori_correction_tensor
        self.state_concentrations_mol_m3 = state_concentrations_mol_m3
        self.symmetric_capacity_fluxes_K_ij_mol_m3_s = (
            symmetric_capacity_fluxes_K_ij_mol_m3_s
        )
        self.reversible_generator_Q_ij_s_inv = reversible_generator_Q_ij_s_inv
        self.transition_first_moments_d_ij_m = transition_first_moments_d_ij_m
        self.transition_second_moments_M_ij_m2 = transition_second_moments_M_ij_m2
        self.self_current_tensors_D_self_i_m2_s = self_current_tensors_D_self_i_m2_s
        self.mori_memory_matrix_A = mori_memory_matrix_A
        self.mori_current_coupling_matrix_h = mori_current_coupling_matrix_h
        self.effect_attribution = effect_attribution

    def __getitem__(self, key: str):
        return self.as_dict()[key]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigma_S_m": self.sigma_S_m,
            "sigma_mS_cm": self.sigma_mS_cm,
            "projected_diffusivity_tensor": self.projected_diffusivity_tensor,
            "direct_diffusivity_tensor": self.direct_diffusivity_tensor,
            "finite_state_memory_correction_tensor": self.finite_state_memory_correction_tensor,
            "continuous_mori_correction_tensor": self.continuous_mori_correction_tensor,
            "state_concentrations_mol_m3": self.state_concentrations_mol_m3,
            "symmetric_capacity_fluxes_K_ij_mol_m3_s": self.symmetric_capacity_fluxes_K_ij_mol_m3_s,
            "reversible_generator_Q_ij_s_inv": self.reversible_generator_Q_ij_s_inv,
            "transition_first_moments_d_ij_m": self.transition_first_moments_d_ij_m,
            "transition_second_moments_M_ij_m2": self.transition_second_moments_M_ij_m2,
            "self_current_tensors_D_self_i_m2_s": self.self_current_tensors_D_self_i_m2_s,
            "mori_memory_matrix_A": self.mori_memory_matrix_A,
            "mori_current_coupling_matrix_h": self.mori_current_coupling_matrix_h,
            "effect_attribution": dict(self.effect_attribution),
        }


class MemoryBasisFilterResult:
    def __init__(
        self,
        mori_memory_matrix_A: Array,
        mori_current_coupling_matrix_h: Array,
        accepted_candidate_indices: Array,
        discarded_candidate_indices: Array,
        rejected_candidate_indices: Array,
    ) -> None:
        self.mori_memory_matrix_A = mori_memory_matrix_A
        self.mori_current_coupling_matrix_h = mori_current_coupling_matrix_h
        self.accepted_candidate_indices = accepted_candidate_indices
        self.discarded_candidate_indices = discarded_candidate_indices
        self.rejected_candidate_indices = rejected_candidate_indices


class FiniteProcessReadoutDiagnostics:
    def __init__(
        self,
        readout_status: str,
        direct_only: bool,
        not_complete_reasons: tuple[str, ...],
        active_transition_capacity_flux_count: int,
        active_transition_first_moment_count: int,
        active_transition_second_moment_count: int,
    ) -> None:
        self.readout_status = readout_status
        self.direct_only = direct_only
        self.not_complete_reasons = not_complete_reasons
        self.active_transition_capacity_flux_count = (
            active_transition_capacity_flux_count
        )
        self.active_transition_first_moment_count = active_transition_first_moment_count
        self.active_transition_second_moment_count = (
            active_transition_second_moment_count
        )


class PrimitivePredictionReadinessDiagnostics:
    def __init__(
        self,
        readiness_status: str,
        scalar_label: str,
        not_complete_reasons: tuple[str, ...],
    ) -> None:
        self.readiness_status = readiness_status
        self.scalar_label = scalar_label
        self.not_complete_reasons = not_complete_reasons


def compute_projected_analytical_conductivity(
    potential_energy_J_mol: Callable[[Array], float],
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    memory_coordinate_gradient: Callable[[Array], Array],
    basin_quadrature_points: tuple[Array, ...],
    basin_quadrature_weights: tuple[Array, ...],
    transition_pair_indices: Array,
    transition_quadrature_points: tuple[Array, ...],
    transition_quadrature_weights: tuple[Array, ...],
    transition_committor_gradients: tuple[Array, ...],
    transition_surface_state_indices: tuple[Array, ...],
    transition_path_displacements_m: tuple[Array, ...],
    transition_path_weights: tuple[Array, ...],
    total_component_concentrations_mol_m3: Array,
    basin_stoichiometry: Array,
    temperature_K: float,
    volume_m3: float,
    self_current_coordinate_projectors: tuple[Array, ...],
) -> ProjectedConductivityResult:
    state_count = np.asarray(basin_stoichiometry, dtype=float).shape[0]
    transition_log_capacity_integrals = (
        compute_transition_log_capacity_integrals_from_quadrature(
            potential_energy_J_mol,
            mobility_tensor_m2_s,
            transition_quadrature_points,
            transition_quadrature_weights,
            transition_committor_gradients,
            temperature_K,
        )
    )
    transition_first_moments, transition_second_moments = (
        compute_transition_path_displacement_moments(
            transition_pair_indices,
            transition_path_displacements_m,
            transition_path_weights,
            state_count,
            max_displacement_m=DEFAULT_MAX_TRANSITION_DISPLACEMENT_M,
        )
    )
    return _compute_projected_analytical_conductivity_from_input(
        ProjectedGeneratorInput(
            potential_energy_J_mol=potential_energy_J_mol,
            mobility_tensor_m2_s=mobility_tensor_m2_s,
            charge_polarization_gradient=charge_polarization_gradient,
            memory_coordinate_gradient=memory_coordinate_gradient,
            basin_quadrature_points=basin_quadrature_points,
            basin_quadrature_weights=basin_quadrature_weights,
            transition_pair_indices=transition_pair_indices,
            transition_quadrature_points=transition_quadrature_points,
            transition_quadrature_weights=transition_quadrature_weights,
            transition_committor_gradients=transition_committor_gradients,
            transition_surface_state_indices=transition_surface_state_indices,
            transition_path_displacements_m=transition_path_displacements_m,
            transition_path_weights=transition_path_weights,
            transition_log_capacity_integrals=transition_log_capacity_integrals,
            transition_uses_residence_rate_constants=np.zeros(
                transition_log_capacity_integrals.shape,
                dtype=bool,
            ),
            transition_residence_rate_constants_s_inv=np.zeros(
                transition_log_capacity_integrals.shape,
                dtype=float,
            ),
            transition_first_moments_d_ij_m=transition_first_moments,
            transition_second_moments_M_ij_m2=transition_second_moments,
            total_component_concentrations_mol_m3=total_component_concentrations_mol_m3,
            basin_stoichiometry=basin_stoichiometry,
            temperature_K=temperature_K,
            volume_m3=volume_m3,
            self_current_coordinate_projectors=self_current_coordinate_projectors,
        )
    )


def _compute_projected_analytical_conductivity_from_input(
    model_input: ProjectedGeneratorInput,
) -> ProjectedConductivityResult:
    validate_generator_input(model_input)
    partitions = compute_restricted_partition_values(
        model_input.potential_energy_J_mol,
        model_input.basin_quadrature_points,
        model_input.basin_quadrature_weights,
        model_input.temperature_K,
    )
    density_result = compute_basin_density_weights(
        model_input.potential_energy_J_mol,
        model_input.basin_quadrature_points,
        model_input.basin_quadrature_weights,
        partitions,
        model_input.total_component_concentrations_mol_m3,
        model_input.basin_stoichiometry,
        model_input.temperature_K,
    )
    concentrations = np.asarray(
        density_result["basin_concentrations_mol_m3"], dtype=float
    )
    density_weights = tuple(density_result["basin_density_weights_mol_m3"])
    chemical_potentials = np.asarray(density_result["chemical_potentials"], dtype=float)
    K = compute_symmetric_capacity_fluxes(
        model_input.potential_energy_J_mol,
        model_input.mobility_tensor_m2_s,
        model_input.transition_pair_indices,
        model_input.transition_quadrature_points,
        model_input.transition_quadrature_weights,
        model_input.transition_committor_gradients,
        model_input.transition_surface_state_indices,
        model_input.basin_stoichiometry,
        chemical_potentials,
        model_input.temperature_K,
        len(partitions),
        model_input.transition_log_capacity_integrals,
        model_input.transition_uses_residence_rate_constants,
        model_input.transition_residence_rate_constants_s_inv,
        concentrations,
    )
    d, M = transition_moments_from_generator_input(model_input, len(partitions))
    Dself = compute_self_current_tensors(
        model_input.mobility_tensor_m2_s,
        model_input.charge_polarization_gradient,
        model_input.basin_quadrature_points,
        density_weights,
        concentrations,
        self_current_coordinate_projectors=model_input.self_current_coordinate_projectors,
    )
    Q_for_filter = compute_reversible_generator(K, concentrations)
    validate_reversible_generator(Q_for_filter, concentrations)
    direct_for_filter = compute_direct_diffusivity_tensor(concentrations, K, M, Dself)
    finite_correction_for_filter = compute_finite_state_memory_correction(
        concentrations,
        Q_for_filter,
        d,
    )
    filtered_memory = filter_memory_basis_by_dirichlet_residual(
        model_input.memory_coordinate_gradient,
        model_input.mobility_tensor_m2_s,
        model_input.charge_polarization_gradient,
        model_input.basin_quadrature_points,
        density_weights,
        _symmetrize(direct_for_filter - finite_correction_for_filter),
        MEMORY_NULLSPACE_RELATIVE_TOL,
        MEMORY_NULLSPACE_RELATIVE_TOL,
        PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL,
    )
    projector_ranks = tuple(
        int(np.linalg.matrix_rank(np.asarray(projector, dtype=float)))
        for projector in model_input.self_current_coordinate_projectors
    )
    conductivity_result = _compute_projected_analytical_conductivity_from_primitive_input(
        ProjectedPrimitiveInput(
            state_concentrations_mol_m3=concentrations,
            symmetric_capacity_fluxes_K_ij_mol_m3_s=K,
            transition_first_moments_d_ij_m=d,
            transition_second_moments_M_ij_m2=M,
            self_current_tensors_D_self_i_m2_s=Dself,
            mori_memory_matrix_A=filtered_memory.mori_memory_matrix_A,
            mori_current_coupling_matrix_h=filtered_memory.mori_current_coupling_matrix_h,
            temperature_K=model_input.temperature_K,
            volume_m3=model_input.volume_m3,
            max_transition_displacement_m=model_input.max_transition_displacement_m,
        ),
        self_current_projector_ranks=projector_ranks,
    )
    conductivity_result.effect_attribution.update(
        {
            "mori_filter_accepted_candidate_indices": (
                filtered_memory.accepted_candidate_indices
            ),
            "mori_filter_discarded_candidate_indices": (
                filtered_memory.discarded_candidate_indices
            ),
            "mori_filter_rejected_candidate_indices": (
                filtered_memory.rejected_candidate_indices
            ),
            "basis_refinement_convergence_status": "converged",
            "basis_refinement_not_complete_reasons": (),
            "basis_refinement_hard_convergence_failure": False,
            "basis_refinement_final_maximum_residual_score": np.asarray(
                [0.0],
                dtype=float,
            ),
            "basis_refinement_final_conductivity_change_S_m": np.asarray(
                [0.0],
                dtype=float,
            ),
        }
    )
    conductivity_result.effect_attribution.update(
        primitive_prediction_readiness_as_effect_attribution(
            conductivity_result.effect_attribution,
        )
    )
    return conductivity_result


def compute_projected_analytical_conductivity_from_functions(
    potential_energy_J_mol: Callable[[Array], float],
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization: Callable[[Array], Array],
    memory_coordinates: Callable[[Array], Array],
    basin_quadrature_points: tuple[Array, ...],
    basin_quadrature_weights: tuple[Array, ...],
    transition_pair_indices: Array,
    transition_quadrature_points: tuple[Array, ...],
    transition_quadrature_weights: tuple[Array, ...],
    transition_committor_gradients: tuple[Array, ...],
    transition_surface_state_indices: tuple[Array, ...],
    transition_path_start_points: tuple[Array, ...],
    transition_path_end_points: tuple[Array, ...],
    transition_path_weights: tuple[Array, ...],
    total_component_concentrations_mol_m3: Array,
    basin_stoichiometry: Array,
    temperature_K: float,
    volume_m3: float,
    self_current_coordinate_projectors: tuple[Array, ...],
) -> ProjectedConductivityResult:
    return _compute_projected_analytical_conductivity_from_functions_input(
        FunctionGeneratorInput(
            potential_energy_J_mol=potential_energy_J_mol,
            mobility_tensor_m2_s=mobility_tensor_m2_s,
            charge_polarization=charge_polarization,
            memory_coordinates=memory_coordinates,
            basin_quadrature_points=basin_quadrature_points,
            basin_quadrature_weights=basin_quadrature_weights,
            transition_pair_indices=transition_pair_indices,
            transition_quadrature_points=transition_quadrature_points,
            transition_quadrature_weights=transition_quadrature_weights,
            transition_committor_gradients=transition_committor_gradients,
            transition_surface_state_indices=transition_surface_state_indices,
            transition_path_start_points=transition_path_start_points,
            transition_path_end_points=transition_path_end_points,
            transition_path_weights=transition_path_weights,
            total_component_concentrations_mol_m3=total_component_concentrations_mol_m3,
            basin_stoichiometry=basin_stoichiometry,
            temperature_K=temperature_K,
            volume_m3=volume_m3,
            self_current_coordinate_projectors=self_current_coordinate_projectors,
        )
    )


def _compute_projected_analytical_conductivity_from_functions_input(
    model_input: FunctionGeneratorInput,
) -> ProjectedConductivityResult:
    validate_function_input(model_input)
    coordinate_dim = _infer_coordinate_dim(model_input.basin_quadrature_points)
    first_point = np.asarray(model_input.basin_quadrature_points[0], dtype=float)[0]
    charge_dim = int(
        np.asarray(model_input.charge_polarization(first_point), dtype=float).size
    )
    memory_dim = int(
        np.asarray(model_input.memory_coordinates(first_point), dtype=float).size
    )
    if charge_dim != CARTESIAN:
        raise ValueError("charge_polarization(q) must return shape (3,)")

    def gradP(point: Array) -> Array:
        return central_difference_jacobian(
            model_input.charge_polarization,
            point,
            output_count=CARTESIAN,
            coordinate_count=coordinate_dim,
            relative_step=model_input.finite_difference_relative_step,
        )

    def gradpsi(point: Array) -> Array:
        return central_difference_jacobian(
            model_input.memory_coordinates,
            point,
            output_count=memory_dim,
            coordinate_count=coordinate_dim,
            relative_step=model_input.finite_difference_relative_step,
        )

    displacements: list[Array] = []
    for starts, ends in zip(
        model_input.transition_path_start_points,
        model_input.transition_path_end_points,
        strict=True,
    ):
        starts2 = as_2d(starts, "transition_path_start_points[]")
        ends2 = as_2d(ends, "transition_path_end_points[]")
        if starts2.shape != ends2.shape:
            raise ValueError("transition path start/end shape mismatch")
        edge_dp = []
        for start, end in zip(starts2, ends2):
            edge_dp.append(
                np.asarray(model_input.charge_polarization(end), dtype=float)
                - np.asarray(model_input.charge_polarization(start), dtype=float)
            )
        displacements.append(np.asarray(edge_dp, dtype=float))
    transition_first_moments, transition_second_moments = (
        compute_transition_path_displacement_moments(
            model_input.transition_pair_indices,
            tuple(displacements),
            model_input.transition_path_weights,
            model_input.basin_stoichiometry.shape[0],
            max_displacement_m=model_input.max_transition_displacement_m,
        )
    )
    transition_log_capacity_integrals = (
        compute_transition_log_capacity_integrals_from_quadrature(
            model_input.potential_energy_J_mol,
            model_input.mobility_tensor_m2_s,
            model_input.transition_quadrature_points,
            model_input.transition_quadrature_weights,
            model_input.transition_committor_gradients,
            model_input.temperature_K,
        )
    )

    return _compute_projected_analytical_conductivity_from_input(
        ProjectedGeneratorInput(
            potential_energy_J_mol=model_input.potential_energy_J_mol,
            mobility_tensor_m2_s=model_input.mobility_tensor_m2_s,
            charge_polarization_gradient=gradP,
            memory_coordinate_gradient=gradpsi,
            basin_quadrature_points=model_input.basin_quadrature_points,
            basin_quadrature_weights=model_input.basin_quadrature_weights,
            transition_pair_indices=model_input.transition_pair_indices,
            transition_quadrature_points=model_input.transition_quadrature_points,
            transition_quadrature_weights=model_input.transition_quadrature_weights,
            transition_committor_gradients=model_input.transition_committor_gradients,
            transition_surface_state_indices=model_input.transition_surface_state_indices,
            transition_path_displacements_m=tuple(displacements),
            transition_path_weights=model_input.transition_path_weights,
            transition_log_capacity_integrals=transition_log_capacity_integrals,
            transition_uses_residence_rate_constants=np.zeros(
                transition_log_capacity_integrals.shape,
                dtype=bool,
            ),
            transition_residence_rate_constants_s_inv=np.zeros(
                transition_log_capacity_integrals.shape,
                dtype=float,
            ),
            transition_first_moments_d_ij_m=transition_first_moments,
            transition_second_moments_M_ij_m2=transition_second_moments,
            total_component_concentrations_mol_m3=model_input.total_component_concentrations_mol_m3,
            basin_stoichiometry=model_input.basin_stoichiometry,
            temperature_K=model_input.temperature_K,
            volume_m3=model_input.volume_m3,
            self_current_coordinate_projectors=model_input.self_current_coordinate_projectors,
            max_transition_displacement_m=model_input.max_transition_displacement_m,
        )
    )


def compute_projected_analytical_conductivity_from_primitives(
    state_concentrations_mol_m3: Array,
    symmetric_capacity_fluxes_K_ij_mol_m3_s: Array,
    transition_first_moments_d_ij_m: Array,
    transition_second_moments_M_ij_m2: Array,
    self_current_tensors_D_self_i_m2_s: Array,
    mori_memory_matrix_A: Array,
    mori_current_coupling_matrix_h: Array,
    temperature_K: float,
    volume_m3: float = PROJECTED_REFERENCE_VOLUME_M3,
) -> ProjectedConductivityResult:
    return _compute_projected_analytical_conductivity_from_primitive_input(
        ProjectedPrimitiveInput(
            state_concentrations_mol_m3=state_concentrations_mol_m3,
            symmetric_capacity_fluxes_K_ij_mol_m3_s=symmetric_capacity_fluxes_K_ij_mol_m3_s,
            transition_first_moments_d_ij_m=transition_first_moments_d_ij_m,
            transition_second_moments_M_ij_m2=transition_second_moments_M_ij_m2,
            self_current_tensors_D_self_i_m2_s=self_current_tensors_D_self_i_m2_s,
            mori_memory_matrix_A=mori_memory_matrix_A,
            mori_current_coupling_matrix_h=mori_current_coupling_matrix_h,
            temperature_K=temperature_K,
            volume_m3=volume_m3,
        ),
        self_current_projector_ranks=(),
    )


def compute_projected_analytical_conductivity_from_composition(
    recipe: Mapping[str, Mapping[str, float]],
    temperature_K: float,
) -> ProjectedConductivityResult:
    raise ValueError(
        "composition-only conductivity evaluation requires a populated full "
        "ConductivityPhysicalLibrary; call compute_projected_analytical_conductivity "
        "or build_projected_generator_from_physical_library with explicit U, D, P, "
        "basins, transition surfaces, transition moments, and memory coordinates"
    )


def build_projected_primitives_from_electrolyte_composition(
    recipe: Mapping[str, Mapping[str, float]],
    temperature_K: float,
) -> dict[str, Array]:
    raise ValueError(
        "composition-only primitive construction requires a populated full "
        "ConductivityPhysicalLibrary; recipe dictionaries do not determine U, D, P, "
        "basins, transition surfaces, transition moments, or memory coordinates"
    )


def build_projected_generator_from_electrolyte_composition(
    recipe: Mapping[str, Mapping[str, float]],
    temperature_K: float,
) -> ProjectedGeneratorInput:
    raise ValueError(
        "recipe-to-generator construction requires a populated full "
        "ConductivityPhysicalLibrary; composition species/loadings alone are not "
        "an executable physical library"
    )


def build_projected_generator_from_physical_library(
    physical_library: ConductivityPhysicalLibrary,
) -> ProjectedGeneratorInput:
    validate_physical_library(physical_library)
    return physical_library.generator_input


def _compute_projected_analytical_conductivity_from_primitive_input(
    primitive_input: ProjectedPrimitiveInput,
    self_current_projector_ranks: tuple[int, ...],
) -> ProjectedConductivityResult:
    c, K, d, M, Dself, A, h = validate_primitive_input(primitive_input)
    Q = compute_reversible_generator(K, c)
    validate_reversible_generator(Q, c)
    direct = compute_direct_diffusivity_tensor(c, K, M, Dself)
    finite_process_diagnostics = compute_finite_process_readout_diagnostics(K, d, M)
    finite_corr = compute_finite_state_memory_correction(c, Q, d)
    mori_corr = compute_continuous_mori_correction(A, h)
    validate_memory_schur_compatibility(
        direct_diffusivity_tensor=direct,
        finite_state_memory_correction_tensor=finite_corr,
        mori_memory_matrix_A=A,
        mori_current_coupling_matrix_h=h,
        self_current_tensors_D_self_i_m2_s=Dself,
        state_concentrations_mol_m3=c,
        self_current_projector_ranks=self_current_projector_ranks,
    )
    projected = project_diffusivity_tensor_to_psd_roundoff(
        _symmetrize(direct - finite_corr - mori_corr)
    )
    sigma_S_m = conductivity_from_projected_diffusivity(
        projected, primitive_input.temperature_K
    )
    attribution = compute_effect_attribution(
        c, K, M, Dself, finite_corr, mori_corr, projected
    )
    attribution.update(
        finite_process_readout_diagnostics_as_effect_attribution(
            finite_process_diagnostics
        )
    )
    attribution.update(primitive_prediction_readiness_as_effect_attribution(attribution))
    return ProjectedConductivityResult(
        sigma_S_m=sigma_S_m,
        sigma_mS_cm=sigma_S_m * S_M_TO_MS_CM,
        projected_diffusivity_tensor=projected,
        direct_diffusivity_tensor=direct,
        finite_state_memory_correction_tensor=finite_corr,
        continuous_mori_correction_tensor=mori_corr,
        state_concentrations_mol_m3=c,
        symmetric_capacity_fluxes_K_ij_mol_m3_s=K,
        reversible_generator_Q_ij_s_inv=Q,
        transition_first_moments_d_ij_m=d,
        transition_second_moments_M_ij_m2=M,
        self_current_tensors_D_self_i_m2_s=Dself,
        mori_memory_matrix_A=A,
        mori_current_coupling_matrix_h=h,
        effect_attribution=attribution,
    )


def compute_restricted_partition_values(
    potential_energy_J_mol: Callable[[Array], float],
    basin_quadrature_points: Sequence[Array],
    basin_quadrature_weights: Sequence[Array],
    temperature_K: float,
) -> Array:
    beta_mol = 1.0 / (R_J_PER_MOL_K * positive_float(temperature_K, "temperature_K"))
    log_values = []
    for points, weights in zip(
        basin_quadrature_points, basin_quadrature_weights, strict=True
    ):
        pts = as_2d(points, "basin_quadrature_points[]")
        w = as_1d(weights, "basin_quadrature_weights[]")
        if pts.shape[0] != w.size:
            raise ValueError("basin quadrature point/weight count mismatch")
        basin_log_value = -np.inf
        for point, weight in zip(pts, w):
            positive_weight = positive_float(float(weight), "basin_quadrature_weight")
            log_term = np.log(positive_weight) - beta_mol * float(
                potential_energy_J_mol(point)
            )
            basin_log_value = np.logaddexp(basin_log_value, log_term)
        log_values.append(basin_log_value)
    log_value_array = np.asarray(log_values, dtype=float)
    if not np.all(np.isfinite(log_value_array)):
        raise ValueError("restricted partition log-values must be finite")
    log_reference = float(np.max(log_value_array))
    relative_partition_values = np.maximum(
        np.exp(log_value_array - log_reference),
        PARTITION_RELATIVE_WEIGHT_FLOOR,
    )
    return positive_vector(relative_partition_values, "restricted_partition_values")


def solve_basin_chemical_potentials(
    total_component_concentrations_mol_m3: Array,
    basin_stoichiometry: Array,
    restricted_partition_values: Array,
) -> dict[str, Array]:
    component_totals = positive_vector(
        total_component_concentrations_mol_m3,
        "total_component_concentrations_mol_m3",
    )
    stoichiometry = np.asarray(basin_stoichiometry, dtype=float)
    if stoichiometry.ndim != 2 or not np.all(np.isfinite(stoichiometry)):
        raise ValueError("basin_stoichiometry must be a finite 2D matrix")
    basin_count, component_count = stoichiometry.shape
    if basin_count == 0 or component_count == 0:
        raise ValueError("basin_stoichiometry must contain basins and components")
    if component_count != component_totals.size:
        raise ValueError("basin_stoichiometry component count does not match totals")
    if np.any(stoichiometry < 0.0):
        raise ValueError("basin_stoichiometry entries must be nonnegative")
    if np.any(np.sum(stoichiometry, axis=0) <= 0.0):
        raise ValueError("each conserved component must appear in at least one basin")
    restricted_partitions = positive_vector(
        restricted_partition_values,
        "restricted_partition_values",
    )
    if restricted_partitions.size != basin_count:
        raise ValueError("restricted_partition_values length must match basin count")

    component_denominators = np.maximum(
        component_totals,
        CHEMICAL_POTENTIAL_MIN_CONCENTRATION_MOL_M3,
    )
    constraint_projection = _mass_constraint_projection_matrix(stoichiometry)
    component_partition_availability = stoichiometry.T @ restricted_partitions
    positive_vector(
        component_partition_availability,
        "component_partition_availability",
    )
    initial_chemical_potentials = np.log(
        np.maximum(component_totals, CHEMICAL_POTENTIAL_MIN_CONCENTRATION_MOL_M3)
        / (STANDARD_CONCENTRATION_MOL_M3 * component_partition_availability)
    )

    def normalized_component_residual(chemical_potentials: Array) -> Array:
        if not _chemical_potential_trial_is_representable(
            chemical_potentials,
            stoichiometry,
            restricted_partitions,
        ):
            return np.full(
                component_count,
                FAILED_CHEMICAL_POTENTIAL_RESIDUAL,
                dtype=float,
            )
        basin_concentrations = _basin_concentrations_from_chemical_potentials(
            chemical_potentials,
            stoichiometry,
            restricted_partitions,
        )
        residual = stoichiometry.T @ basin_concentrations - component_totals
        normalized_residual = residual / component_denominators
        return constraint_projection @ normalized_residual

    def normalized_component_jacobian(chemical_potentials: Array) -> Array:
        if not _chemical_potential_trial_is_representable(
            chemical_potentials,
            stoichiometry,
            restricted_partitions,
        ):
            return np.zeros(
                (constraint_projection.shape[0], component_count), dtype=float
            )
        basin_concentrations = _basin_concentrations_from_chemical_potentials(
            chemical_potentials,
            stoichiometry,
            restricted_partitions,
        )
        jacobian = stoichiometry.T @ (
            basin_concentrations[:, np.newaxis] * stoichiometry
        )
        normalized_jacobian = jacobian / component_denominators[:, np.newaxis]
        return constraint_projection @ normalized_jacobian

    least_squares_result = least_squares(
        normalized_component_residual,
        initial_chemical_potentials,
        jac=normalized_component_jacobian,
        xtol=CHEMICAL_POTENTIAL_SOLVER_TOL,
        ftol=CHEMICAL_POTENTIAL_SOLVER_TOL,
        gtol=CHEMICAL_POTENTIAL_SOLVER_TOL,
        max_nfev=CHEMICAL_POTENTIAL_MAX_ITERATIONS,
    )
    chemical_potentials = np.asarray(least_squares_result.x, dtype=float)
    basin_concentrations = _basin_concentrations_from_chemical_potentials(
        chemical_potentials,
        stoichiometry,
        restricted_partitions,
    )
    residual = stoichiometry.T @ basin_concentrations - component_totals
    normalized_residual = _normalized_mass_residual(residual, component_totals)
    if normalized_residual >= CHEMICAL_POTENTIAL_MASS_TOL:
        raise ValueError(
            "chemical potential mass-balance solve did not converge; "
            f"normalized_residual={normalized_residual:.9g}"
        )
    return {
        "chemical_potentials": chemical_potentials,
        "basin_concentrations_mol_m3": basin_concentrations,
        "residual_mol_m3": residual,
        "normalized_residual": np.asarray([normalized_residual], dtype=float),
        "iterations": np.asarray([least_squares_result.nfev], dtype=float),
    }


def _mass_constraint_projection_matrix(basin_stoichiometry: Array) -> Array:
    constraint_matrix = np.asarray(basin_stoichiometry, dtype=float).T
    left_singular_vectors, singular_values, _right_singular_vectors = np.linalg.svd(
        constraint_matrix,
        full_matrices=False,
    )
    if singular_values.size == 0:
        raise ValueError("mass-balance constraint matrix has no singular values")
    singular_value_threshold = PSEUDOINVERSE_RELATIVE_TOL * max(
        float(singular_values[0]), 1.0
    )
    independent_constraint_count = int(
        np.count_nonzero(singular_values > singular_value_threshold)
    )
    if independent_constraint_count == 0:
        raise ValueError(
            "mass-balance constraint matrix has no independent constraints"
        )
    return np.asarray(
        left_singular_vectors[:, :independent_constraint_count].T,
        dtype=float,
    )


def compute_equilibrium_populations_from_stoichiometry(
    total_component_concentrations_mol_m3: Array,
    basin_stoichiometry: Array,
    restricted_partition_values: Array,
) -> Array:
    solve_result = solve_basin_chemical_potentials(
        total_component_concentrations_mol_m3,
        basin_stoichiometry,
        restricted_partition_values,
    )
    return np.asarray(solve_result["basin_concentrations_mol_m3"], dtype=float)


def compute_basin_density_weights(
    potential_energy_J_mol: Callable[[Array], float],
    basin_quadrature_points: Sequence[Array],
    basin_quadrature_weights: Sequence[Array],
    restricted_partition_values: Array,
    total_component_concentrations_mol_m3: Array,
    basin_stoichiometry: Array,
    temperature_K: float,
) -> dict[str, Array | tuple[Array, ...]]:
    solve_result = solve_basin_chemical_potentials(
        total_component_concentrations_mol_m3,
        basin_stoichiometry,
        restricted_partition_values,
    )
    chemical_potentials = np.asarray(solve_result["chemical_potentials"], dtype=float)
    beta_mol = 1.0 / (R_J_PER_MOL_K * positive_float(temperature_K, "temperature_K"))
    solved_concentrations = np.asarray(
        solve_result["basin_concentrations_mol_m3"],
        dtype=float,
    )
    density_weights: list[Array] = []
    for basin_index, (points, weights) in enumerate(
        zip(basin_quadrature_points, basin_quadrature_weights, strict=True)
    ):
        quadrature_points = as_2d(points, "basin_quadrature_points[]")
        quadrature_weights = as_1d(weights, "basin_quadrature_weights[]")
        if quadrature_points.shape[0] != quadrature_weights.size:
            raise ValueError("basin quadrature point/weight count mismatch")
        log_terms = np.zeros(quadrature_weights.size, dtype=float)
        for point_index, point in enumerate(quadrature_points):
            positive_weight = positive_float(
                float(quadrature_weights[point_index]),
                "basin_quadrature_weight",
            )
            log_terms[point_index] = np.log(positive_weight) - beta_mol * float(
                potential_energy_J_mol(point)
            )
        log_normalizer = float(np.max(log_terms))
        normalized_weights = np.exp(log_terms - log_normalizer)
        normalized_weights /= positive_float(
            float(np.sum(normalized_weights)),
            "basin_density_normalized_weight_sum",
        )
        weights_mol_m3 = solved_concentrations[basin_index] * normalized_weights
        density_weights.append(weights_mol_m3)
    basin_concentrations = np.asarray(
        [float(np.sum(weights)) for weights in density_weights],
        dtype=float,
    )
    if not np.allclose(
        basin_concentrations,
        solved_concentrations,
        atol=CHEMICAL_POTENTIAL_MASS_TOL,
        rtol=CHEMICAL_POTENTIAL_MASS_TOL,
    ):
        raise ValueError(
            "density quadrature weights do not reproduce basin concentrations"
        )
    return {
        "chemical_potentials": chemical_potentials,
        "basin_concentrations_mol_m3": basin_concentrations,
        "basin_density_weights_mol_m3": tuple(density_weights),
        "residual_mol_m3": np.asarray(solve_result["residual_mol_m3"], dtype=float),
        "normalized_residual": np.asarray(
            solve_result["normalized_residual"],
            dtype=float,
        ),
        "iterations": np.asarray(solve_result["iterations"], dtype=float),
    }


def _basin_concentrations_from_chemical_potentials(
    chemical_potentials: Array,
    stoichiometry: Array,
    restricted_partition_values: Array,
) -> Array:
    log_concentrations = (
        np.log(STANDARD_CONCENTRATION_MOL_M3)
        + np.log(restricted_partition_values)
        + stoichiometry @ chemical_potentials
    )
    return positive_vector(
        np.exp(log_concentrations),
        "basin_concentrations_mol_m3",
    )


def _chemical_potential_trial_is_representable(
    chemical_potentials: Array,
    stoichiometry: Array,
    restricted_partition_values: Array,
) -> bool:
    log_concentrations = (
        np.log(STANDARD_CONCENTRATION_MOL_M3)
        + np.log(restricted_partition_values)
        + stoichiometry @ chemical_potentials
    )
    return bool(
        np.all(np.isfinite(log_concentrations))
        and np.max(log_concentrations) < LOG_FLOAT_MAX
    )


def _normalized_mass_residual(
    residual_mol_m3: Array,
    component_totals_mol_m3: Array,
) -> float:
    denominators = np.maximum(
        component_totals_mol_m3,
        CHEMICAL_POTENTIAL_MIN_CONCENTRATION_MOL_M3,
    )
    return float(np.max(np.abs(residual_mol_m3) / denominators))


def compute_transition_log_capacity_integrals_from_quadrature(
    potential_energy_J_mol: Callable[[Array], float],
    mobility_tensor_m2_s: Callable[[Array], Array],
    transition_quadrature_points: Sequence[Array],
    transition_quadrature_weights: Sequence[Array],
    transition_committor_gradients: Sequence[Array],
    temperature_K: float,
) -> Array:
    beta_mol = 1.0 / (R_J_PER_MOL_K * positive_float(temperature_K, "temperature_K"))
    validate_equal_lengths(
        len(transition_quadrature_points),
        (
            transition_quadrature_weights,
            transition_committor_gradients,
        ),
    )
    log_capacity_integrals = []
    for transition_index, points in enumerate(transition_quadrature_points):
        transition_points = as_2d(
            points,
            "transition_quadrature_points[]",
        )
        transition_weights = as_1d(
            transition_quadrature_weights[transition_index],
            "transition_quadrature_weights[]",
        )
        transition_gradients = as_2d(
            transition_committor_gradients[transition_index],
            "transition_committor_gradients[]",
        )
        if not (
            transition_points.shape[0]
            == transition_weights.size
            == transition_gradients.shape[0]
        ):
            raise ValueError(
                "transition quadrature point/weight/gradient count mismatch"
            )
        edge_log_capacity = -np.inf
        for point, quadrature_weight, committor_gradient in zip(
            transition_points,
            transition_weights,
            transition_gradients,
            strict=True,
        ):
            positive_weight = positive_float(
                float(quadrature_weight),
                "transition_quadrature_weight",
            )
            mobility_tensor = as_square(
                mobility_tensor_m2_s(point),
                point.size,
                "mobility_tensor",
            )
            dirichlet_density = float(
                committor_gradient @ mobility_tensor @ committor_gradient
            )
            positive_dirichlet_density = positive_float(
                dirichlet_density,
                "transition_dirichlet_density",
            )
            log_term = (
                np.log(positive_weight)
                - beta_mol * float(potential_energy_J_mol(point))
                + np.log(positive_dirichlet_density)
            )
            edge_log_capacity = np.logaddexp(edge_log_capacity, log_term)
        if not np.isfinite(edge_log_capacity):
            raise ValueError("transition log capacity integral must be finite")
        log_capacity_integrals.append(edge_log_capacity)
    return np.asarray(log_capacity_integrals, dtype=float)


def compute_symmetric_capacity_fluxes(
    potential_energy_J_mol: Callable[[Array], float],
    mobility_tensor_m2_s: Callable[[Array], Array],
    transition_pair_indices: Array,
    transition_quadrature_points: Sequence[Array],
    transition_quadrature_weights: Sequence[Array],
    transition_committor_gradients: Sequence[Array],
    transition_surface_state_indices: Sequence[Array],
    basin_stoichiometry: Array,
    chemical_potentials: Array,
    temperature_K: float,
    state_count: int,
    transition_log_capacity_integrals: Array,
    transition_uses_residence_rate_constants: Array,
    transition_residence_rate_constants_s_inv: Array,
    basin_concentrations_mol_m3: Array,
) -> Array:
    pairs = as_pairs(transition_pair_indices, state_count)
    validate_equal_lengths(
        pairs.shape[0],
        (
            transition_quadrature_points,
            transition_quadrature_weights,
            transition_committor_gradients,
            transition_surface_state_indices,
        ),
    )
    positive_float(temperature_K, "temperature_K")
    stoichiometry = np.asarray(basin_stoichiometry, dtype=float)
    if stoichiometry.ndim != 2 or stoichiometry.shape[0] != state_count:
        raise ValueError("basin_stoichiometry must have one row per state")
    potentials = as_1d(chemical_potentials, "chemical_potentials")
    if potentials.size != stoichiometry.shape[1]:
        raise ValueError("chemical_potentials length must match basin stoichiometry")
    log_capacity_integrals = as_1d(
        transition_log_capacity_integrals,
        "transition_log_capacity_integrals",
    )
    if log_capacity_integrals.size != pairs.shape[0]:
        raise ValueError(
            "transition_log_capacity_integrals length must equal transition count"
        )
    uses_residence_rate_constants = np.asarray(
        transition_uses_residence_rate_constants,
        dtype=bool,
    )
    if uses_residence_rate_constants.shape != (pairs.shape[0],):
        raise ValueError(
            "transition_uses_residence_rate_constants length must equal transition count"
        )
    residence_rate_constants = as_1d(
        transition_residence_rate_constants_s_inv,
        "transition_residence_rate_constants_s_inv",
    )
    if residence_rate_constants.size != pairs.shape[0]:
        raise ValueError(
            "transition_residence_rate_constants_s_inv length must equal transition count"
        )
    basin_concentrations = positive_vector(
        basin_concentrations_mol_m3,
        "basin_concentrations_mol_m3",
    )
    if basin_concentrations.size != state_count:
        raise ValueError("basin_concentrations_mol_m3 length must equal state_count")
    K = np.zeros((state_count, state_count), dtype=float)
    for e, (i, j) in enumerate(pairs):
        if (
            float(basin_concentrations[i])
            <= CHEMICAL_POTENTIAL_MIN_CONCENTRATION_MOL_M3
            or float(basin_concentrations[j])
            <= CHEMICAL_POTENTIAL_MIN_CONCENTRATION_MOL_M3
        ):
            continue
        pts = as_2d(transition_quadrature_points[e], "transition_quadrature_points[]")
        w = as_1d(transition_quadrature_weights[e], "transition_quadrature_weights[]")
        grads = as_2d(
            transition_committor_gradients[e], "transition_committor_gradients[]"
        )
        surface_states = np.asarray(transition_surface_state_indices[e], dtype=int)
        if surface_states.ndim != 1:
            raise ValueError("transition_surface_state_indices[] must be 1D")
        if not (pts.shape[0] == w.size == grads.shape[0] == surface_states.size):
            raise ValueError(
                "transition quadrature point/weight/gradient/state count mismatch"
            )
        if np.any(surface_states < 0) or np.any(surface_states >= state_count):
            raise ValueError("transition surface state index is out of range")
        unique_surface_states = np.unique(surface_states)
        if unique_surface_states.size != 1:
            raise ValueError(
                "stable transition capacity assembly requires one explicit "
                "capacity stoichiometry state per transition"
            )
        surface_state = int(unique_surface_states[0])
        if uses_residence_rate_constants[e]:
            rate_constant_s_inv = positive_float(
                float(residence_rate_constants[e]),
                "transition_residence_rate_constants_s_inv[]",
            )
            endpoint_limited_concentration = min(
                float(basin_concentrations[i]),
                float(basin_concentrations[j]),
            )
            edge_flux = endpoint_limited_concentration * rate_constant_s_inv
        else:
            if float(residence_rate_constants[e]) != 0.0:
                raise ValueError(
                    "capacity-integral transition must have zero residence rate constant"
                )
            exponent_shift = float(stoichiometry[surface_state] @ potentials)
            log_edge_flux = (
                np.log(STANDARD_CONCENTRATION_MOL_M3)
                + exponent_shift
                + float(log_capacity_integrals[e])
            )
            if log_edge_flux >= LOG_FLOAT_MAX:
                raise ValueError(
                    "transition capacity flux exceeds floating-point range"
                )
            if log_edge_flux <= LOG_FLOAT_TINY:
                edge_flux = 0.0
            else:
                edge_flux = float(np.exp(log_edge_flux))
        if edge_flux < -PSD_TOL:
            raise ValueError("capacity flux must be nonnegative")
        K[i, j] += edge_flux
        K[j, i] += edge_flux
    return K


def compute_reversible_generator(K: Array, c: Array) -> Array:
    K = as_square_any(K, "K")
    c = positive_vector(c, "c")
    if K.shape != (c.size, c.size):
        raise ValueError("K shape does not match c")
    Q = np.zeros_like(K, dtype=float)
    for i in range(c.size):
        for j in range(c.size):
            if i != j:
                Q[i, j] = K[i, j] / c[i]
        Q[i, i] = -float(np.sum(K[i])) / c[i]
    return Q


def validate_reversible_generator(Q: Array, c: Array) -> None:
    generator = as_square_any(Q, "Q")
    concentrations = positive_vector(c, "c")
    if generator.shape != (concentrations.size, concentrations.size):
        raise ValueError("Q shape does not match c")
    row_sum_residual = np.sum(generator, axis=1)
    row_sum_scale = max(float(np.max(np.abs(generator))), 1.0)
    if float(np.max(np.abs(row_sum_residual))) / row_sum_scale > GENERATOR_BALANCE_TOL:
        raise ValueError("finite generator row sums must be zero")
    concentration_flux = concentrations[:, np.newaxis] * generator
    if not np.allclose(
        concentration_flux,
        concentration_flux.T,
        atol=GENERATOR_BALANCE_TOL,
        rtol=GENERATOR_BALANCE_TOL,
    ):
        raise ValueError("finite generator violates detailed balance")
    stationarity_residual = concentrations @ generator
    stationarity_scale = max(float(np.max(np.abs(concentration_flux))), 1.0)
    if (
        float(np.max(np.abs(stationarity_residual))) / stationarity_scale
        > GENERATOR_BALANCE_TOL
    ):
        raise ValueError("finite generator violates stationarity")


def compute_transition_path_displacement_moments(
    transition_pair_indices: Array,
    transition_path_displacements_m: Sequence[Array],
    transition_path_weights: Sequence[Array],
    state_count: int,
    max_displacement_m: float,
) -> tuple[Array, Array]:
    pairs = as_pairs(transition_pair_indices, state_count)
    validate_equal_lengths(
        pairs.shape[0],
        (transition_path_displacements_m, transition_path_weights),
    )
    d = np.zeros((state_count, state_count, CARTESIAN), dtype=float)
    M = np.zeros((state_count, state_count, CARTESIAN, CARTESIAN), dtype=float)
    max_disp = positive_float(max_displacement_m, "max_transition_displacement_m")
    for e, (i, j) in enumerate(pairs):
        disps = as_2d(
            transition_path_displacements_m[e], "transition_path_displacements_m[]"
        )
        weights = as_1d(transition_path_weights[e], "transition_path_weights[]")
        if disps.shape[1] != CARTESIAN:
            raise ValueError("transition path displacements must have shape (n, 3)")
        if disps.shape[0] != weights.size:
            raise ValueError("transition path displacement/weight count mismatch")
        norms = np.linalg.norm(disps, axis=1)
        if np.any(norms > max_disp):
            raise ValueError(
                f"transition displacement exceeds max_transition_displacement_m={max_disp:g}"
            )
        wsum = positive_float(float(np.sum(weights)), "transition_path_weight_sum")
        mean = np.einsum("n,na->a", weights, disps) / wsum
        second = np.einsum("n,na,nb->ab", weights, disps, disps) / wsum
        d[i, j] = mean
        d[j, i] = -mean
        M[i, j] = _symmetrize(second)
        M[j, i] = _symmetrize(second)
    return d, M


def transition_moments_from_generator_input(
    model_input: ProjectedGeneratorInput,
    state_count: int,
) -> tuple[Array, Array]:
    d = np.asarray(model_input.transition_first_moments_d_ij_m, dtype=float)
    if d.shape != (state_count, state_count, CARTESIAN) or not np.all(np.isfinite(d)):
        raise ValueError("transition_first_moments_d_ij_m must have shape (n,n,3)")
    if not np.allclose(d + np.swapaxes(d, 0, 1), 0.0, atol=PSD_TOL, rtol=PSD_TOL):
        raise ValueError("transition_first_moments_d_ji must equal -d_ij")
    displacement_norms = np.linalg.norm(d.reshape(-1, CARTESIAN), axis=1)
    if np.any(displacement_norms > model_input.max_transition_displacement_m):
        raise ValueError("transition first moment exceeds max displacement")
    second_moments = np.asarray(
        model_input.transition_second_moments_M_ij_m2, dtype=float
    )
    if second_moments.shape != (
        state_count,
        state_count,
        CARTESIAN,
        CARTESIAN,
    ) or not np.all(np.isfinite(second_moments)):
        raise ValueError("transition_second_moments_M_ij_m2 must have shape (n,n,3,3)")
    if not np.allclose(
        second_moments,
        np.swapaxes(second_moments, 0, 1),
        atol=PSD_TOL,
        rtol=PSD_TOL,
    ):
        raise ValueError("transition_second_moments_M_ji must equal M_ij")
    transition_pair_indices = np.asarray(model_input.transition_pair_indices, dtype=int)
    for pair in transition_pair_indices:
        from_state_index = int(pair[0])
        to_state_index = int(pair[1])
        for first_index, second_index in (
            (from_state_index, to_state_index),
            (to_state_index, from_state_index),
        ):
            moment_matrix = second_moments[first_index, second_index]
            if not np.allclose(
                moment_matrix, moment_matrix.T, atol=PSD_TOL, rtol=PSD_TOL
            ):
                raise ValueError("each transition second moment must be symmetric")
            validate_psd(
                moment_matrix,
                f"transition_second_moment[{first_index},{second_index}]",
                allow_zero=True,
            )
    return d, second_moments


def compute_transition_path_displacement_moments_from_polarization(
    transition_pair_indices: Array,
    transition_path_start_points: tuple[Array, ...],
    transition_path_end_points: tuple[Array, ...],
    transition_path_weights: tuple[Array, ...],
    charge_polarization: Callable[[Array], Array],
    state_count: int,
) -> tuple[Array, Array]:
    transition_path_displacements = []
    for start_points, end_points in zip(
        transition_path_start_points,
        transition_path_end_points,
        strict=True,
    ):
        starts = as_2d(start_points, "transition_path_start_points[]")
        ends = as_2d(end_points, "transition_path_end_points[]")
        if starts.shape != ends.shape:
            raise ValueError("transition path start/end shape mismatch")
        displacements = []
        for start_point, end_point in zip(starts, ends, strict=True):
            displacements.append(
                np.asarray(charge_polarization(end_point), dtype=float)
                - np.asarray(charge_polarization(start_point), dtype=float)
            )
        transition_path_displacements.append(np.asarray(displacements, dtype=float))
    return compute_transition_path_displacement_moments(
        transition_pair_indices,
        tuple(transition_path_displacements),
        transition_path_weights,
        state_count,
        max_displacement_m=DEFAULT_MAX_TRANSITION_DISPLACEMENT_M,
    )


def compute_self_current_tensors(
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
    basin_concentrations_mol_m3: Array,
    self_current_coordinate_projectors: Sequence[Array],
) -> Array:
    concentrations = positive_vector(
        basin_concentrations_mol_m3,
        "basin_concentrations_mol_m3",
    )
    state_count = len(basin_quadrature_points)
    tensors = np.zeros((state_count, CARTESIAN, CARTESIAN), dtype=float)
    if len(basin_density_weights_mol_m3) != state_count:
        raise ValueError("basin_density_weights_mol_m3 length must equal state count")
    if concentrations.size != state_count:
        raise ValueError("basin_concentrations_mol_m3 length must equal state count")
    if len(self_current_coordinate_projectors) != state_count:
        raise ValueError(
            "self_current_coordinate_projectors length must equal state count"
        )
    for i, (points, density_weights) in enumerate(
        zip(basin_quadrature_points, basin_density_weights_mol_m3, strict=True)
    ):
        pts = as_2d(points, "basin_quadrature_points[]")
        W = as_1d(density_weights, "basin_density_weights_mol_m3[]")
        if pts.shape[0] != W.size:
            raise ValueError("basin quadrature point/density-weight count mismatch")
        numerator = np.zeros((CARTESIAN, CARTESIAN), dtype=float)
        for point, density_weight in zip(pts, W):
            D = as_square(mobility_tensor_m2_s(point), point.size, "mobility_tensor")
            gradP = as_matrix_shape(
                charge_polarization_gradient(point),
                (CARTESIAN, point.size),
                "charge_polarization_gradient",
            )
            numerator += float(density_weight) * (gradP @ D @ gradP.T)
        tensors[i] = _symmetrize(numerator / concentrations[i])
        validate_psd(tensors[i], f"D_self[{i}]", allow_zero=True)
    return tensors


def compute_mori_memory_matrices(
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    memory_coordinate_gradient: Callable[[Array], Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
) -> tuple[Array, Array]:
    first_point = np.asarray(basin_quadrature_points[0], dtype=float)[0]
    mem_dim = int(
        np.asarray(memory_coordinate_gradient(first_point), dtype=float).shape[0]
    )
    A = np.zeros((mem_dim, mem_dim), dtype=float)
    h = np.zeros((mem_dim, CARTESIAN), dtype=float)
    if mem_dim == 0:
        return A, h
    if len(basin_density_weights_mol_m3) != len(basin_quadrature_points):
        raise ValueError("basin_density_weights_mol_m3 length must equal state count")
    for points, density_weights in zip(
        basin_quadrature_points,
        basin_density_weights_mol_m3,
        strict=True,
    ):
        pts = as_2d(points, "basin_quadrature_points[]")
        W = as_1d(density_weights, "basin_density_weights_mol_m3[]")
        if pts.shape[0] != W.size:
            raise ValueError("basin quadrature point/density-weight count mismatch")
        for point, density_weight in zip(pts, W):
            D = as_square(mobility_tensor_m2_s(point), point.size, "mobility_tensor")
            gradpsi = as_2d(
                memory_coordinate_gradient(point), "memory_coordinate_gradient"
            )
            if gradpsi.shape[1] != point.size:
                raise ValueError(
                    "memory_coordinate_gradient has wrong coordinate dimension"
                )
            gradP = as_matrix_shape(
                charge_polarization_gradient(point),
                (CARTESIAN, point.size),
                "charge_polarization_gradient",
            )
            rho_w = float(density_weight)
            A += rho_w * (gradpsi @ D @ gradpsi.T)
            h += rho_w * (gradpsi @ D @ gradP.T)
    A = _symmetrize(A)
    validate_psd(A, "mori_memory_matrix_A", allow_zero=True)
    return A, h


def filter_memory_basis_by_dirichlet_residual(
    candidate_gradients: Callable[[Array], Array],
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
    direct_minus_finite_state_tensor: Array,
    energy_tol: float,
    null_current_tol: float,
    psd_tol: float,
) -> MemoryBasisFilterResult:
    raw_memory_matrix, raw_current_coupling = compute_mori_memory_matrices(
        mobility_tensor_m2_s,
        charge_polarization_gradient,
        candidate_gradients,
        basin_quadrature_points,
        basin_density_weights_mol_m3,
    )
    candidate_count = raw_memory_matrix.shape[0]
    remaining_tensor_base = _symmetrize(
        as_matrix_shape(
            direct_minus_finite_state_tensor,
            (CARTESIAN, CARTESIAN),
            "direct_minus_finite_state_tensor",
        )
    )
    energy_scale = max(_maximum_abs_entry(raw_memory_matrix), PSD_TOL)
    coupling_scale = max(_maximum_abs_entry(raw_current_coupling), PSD_TOL)
    energy_threshold = positive_float(energy_tol, "energy_tol") * energy_scale
    null_current_threshold = (
        positive_float(null_current_tol, "null_current_tol") * coupling_scale
    )
    psd_threshold = positive_float(psd_tol, "psd_tol")
    accepted_indices: list[int] = []
    discarded_indices: list[int] = []
    rejected_indices: list[int] = []
    current_memory_matrix = np.zeros((0, 0), dtype=float)
    current_coupling_matrix = np.zeros((0, CARTESIAN), dtype=float)
    for candidate_index in range(candidate_count):
        candidate_self_energy = float(
            raw_memory_matrix[candidate_index, candidate_index]
        )
        candidate_coupling = raw_current_coupling[candidate_index]
        if accepted_indices:
            cross_energy = raw_memory_matrix[
                candidate_index,
                np.asarray(accepted_indices, dtype=int),
            ].reshape((1, len(accepted_indices)))
            current_inverse = symmetric_psd_pseudoinverse(current_memory_matrix)
            residual_energy = float(
                candidate_self_energy
                - (cross_energy @ current_inverse @ cross_energy.T)[0, 0]
            )
            residual_coupling = candidate_coupling - (
                cross_energy @ current_inverse @ current_coupling_matrix
            ).reshape(CARTESIAN)
        if not accepted_indices:
            residual_energy = candidate_self_energy
            residual_coupling = candidate_coupling
        residual_coupling_norm = float(np.linalg.norm(residual_coupling))
        if residual_energy <= energy_threshold:
            if residual_coupling_norm > null_current_threshold:
                raise ValueError(
                    "Mori candidate has zero residual Dirichlet energy with nonzero "
                    "current coupling"
                )
            discarded_indices.append(candidate_index)
            continue
        incremental_correction = _symmetrize(
            np.outer(residual_coupling, residual_coupling) / residual_energy
        )
        current_correction = compute_continuous_mori_correction(
            current_memory_matrix,
            current_coupling_matrix,
        )
        remaining_tensor = _symmetrize(remaining_tensor_base - current_correction)
        candidate_remaining_tensor = _symmetrize(
            remaining_tensor - incremental_correction
        )
        if _memory_candidate_spans_current(
            incremental_correction,
            remaining_tensor,
            psd_threshold,
        ):
            rejected_indices.append(candidate_index)
            continue
        if _minimum_eigenvalue(candidate_remaining_tensor) < -psd_threshold * max(
            1.0,
            _maximum_abs_eigenvalue(remaining_tensor),
        ):
            rejected_indices.append(candidate_index)
            continue
        accepted_indices.append(candidate_index)
        accepted_array = np.asarray(accepted_indices, dtype=int)
        current_memory_matrix = raw_memory_matrix[
            np.ix_(accepted_array, accepted_array)
        ]
        current_coupling_matrix = raw_current_coupling[accepted_array]
    return MemoryBasisFilterResult(
        mori_memory_matrix_A=current_memory_matrix,
        mori_current_coupling_matrix_h=current_coupling_matrix,
        accepted_candidate_indices=np.asarray(accepted_indices, dtype=int),
        discarded_candidate_indices=np.asarray(discarded_indices, dtype=int),
        rejected_candidate_indices=np.asarray(rejected_indices, dtype=int),
    )


def validate_charge_density_translation_symmetry(
    memory_coordinate_gradient: Callable[[Array], Array],
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
    symmetry_tol: float,
) -> None:
    tolerance = positive_float(symmetry_tol, "symmetry_tol")
    if len(basin_quadrature_points) != len(basin_density_weights_mol_m3):
        raise ValueError("basin_density_weights_mol_m3 length must equal state count")
    for state_index, (points, density_weights) in enumerate(
        zip(basin_quadrature_points, basin_density_weights_mol_m3, strict=True)
    ):
        pts = as_2d(points, "basin_quadrature_points[]")
        W = as_1d(density_weights, "basin_density_weights_mol_m3[]")
        if pts.shape[0] != W.size:
            raise ValueError("basin quadrature point/density-weight count mismatch")
        state_coupling = None
        state_scale = 0.0
        for point, density_weight in zip(pts, W, strict=True):
            mobility = as_square(
                mobility_tensor_m2_s(point),
                point.size,
                "mobility_tensor",
            )
            memory_gradient = as_2d(
                memory_coordinate_gradient(point),
                "memory_coordinate_gradient",
            )
            if memory_gradient.shape[1] != point.size:
                raise ValueError(
                    "memory_coordinate_gradient has wrong coordinate dimension"
                )
            charge_gradient = as_matrix_shape(
                charge_polarization_gradient(point),
                (CARTESIAN, point.size),
                "charge_polarization_gradient",
            )
            weighted_coupling = (
                float(density_weight) * memory_gradient @ mobility @ charge_gradient.T
            )
            if state_coupling is None:
                state_coupling = np.zeros_like(weighted_coupling)
            state_coupling += weighted_coupling
            state_scale += float(np.linalg.norm(weighted_coupling))
        if state_coupling is None:
            raise ValueError("basin contains no quadrature points")
        residual_norm = float(np.linalg.norm(state_coupling))
        allowable_norm = tolerance * max(state_scale, PSD_TOL)
        if residual_norm > allowable_norm:
            raise ValueError(
                "charge-density memory quadrature aliases unbounded charge polarization "
                f"in state {state_index}: residual={residual_norm:.6e}, "
                f"allowable={allowable_norm:.6e}"
            )


def _memory_candidate_spans_current(
    incremental_correction: Array,
    remaining_tensor: Array,
    psd_tol: float,
) -> bool:
    remaining_trace = float(np.trace(remaining_tensor))
    if remaining_trace <= 0.0:
        return True
    correction_trace = float(np.trace(incremental_correction))
    return correction_trace >= (1.0 - psd_tol) * remaining_trace


def _minimum_eigenvalue(matrix: Array) -> float:
    eigenvalues = np.linalg.eigvalsh(_symmetrize(as_square_any(matrix, "matrix")))
    return float(np.min(eigenvalues))


def _maximum_abs_eigenvalue(matrix: Array) -> float:
    eigenvalues = np.linalg.eigvalsh(_symmetrize(as_square_any(matrix, "matrix")))
    return float(np.max(np.abs(eigenvalues)))


def _maximum_abs_entry(matrix: Array) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def compute_state_memory_coordinate_means(
    memory_coordinates: Callable[[Array], Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
    basin_concentrations_mol_m3: Array,
) -> Array:
    concentrations = positive_vector(
        basin_concentrations_mol_m3,
        "basin_concentrations_mol_m3",
    )
    if len(basin_quadrature_points) != concentrations.size:
        raise ValueError("basin_quadrature_points length must match concentrations")
    if len(basin_density_weights_mol_m3) != concentrations.size:
        raise ValueError(
            "basin_density_weights_mol_m3 length must match concentrations"
        )
    first_point = np.asarray(basin_quadrature_points[0], dtype=float)[0]
    memory_count = int(np.asarray(memory_coordinates(first_point), dtype=float).size)
    state_means = np.zeros((concentrations.size, memory_count), dtype=float)
    for basin_index, (points, density_weights) in enumerate(
        zip(basin_quadrature_points, basin_density_weights_mol_m3, strict=True)
    ):
        quadrature_points = as_2d(points, "basin_quadrature_points[]")
        quadrature_weights = as_1d(
            density_weights,
            "basin_density_weights_mol_m3[]",
        )
        if quadrature_points.shape[0] != quadrature_weights.size:
            raise ValueError("basin quadrature point/density-weight count mismatch")
        weighted_sum = np.zeros(memory_count, dtype=float)
        for point, density_weight in zip(quadrature_points, quadrature_weights):
            memory_value = as_1d(memory_coordinates(point), "memory_coordinates")
            if memory_value.size != memory_count:
                raise ValueError(
                    "memory coordinate dimension changed across quadrature"
                )
            weighted_sum += float(density_weight) * memory_value
        state_means[basin_index] = weighted_sum / concentrations[basin_index]
    return state_means


def compute_direct_diffusivity_tensor(
    c: Array, K: Array, M: Array, Dself: Array
) -> Array:
    direct = np.einsum("i,iab->ab", c, Dself)
    direct += 0.5 * np.einsum("ij,ijab->ab", K, M)
    return _symmetrize(direct)


def compute_finite_state_memory_correction(c: Array, Q: Array, d: Array) -> Array:
    if c.size == 1 or np.max(np.abs(d)) == 0.0 or np.max(np.abs(Q)) == 0.0:
        return np.zeros((CARTESIAN, CARTESIAN), dtype=float)
    b = np.einsum("ij,ija->ia", Q, d)
    correction = np.zeros((CARTESIAN, CARTESIAN), dtype=float)
    chis = []
    for axis in range(CARTESIAN):
        chis.append(solve_weighted_poisson(Q, c, b[:, axis]))
    for a in range(CARTESIAN):
        for b_axis in range(CARTESIAN):
            correction[a, b_axis] = float(np.sum(c * b[:, a] * chis[b_axis]))
    return _symmetrize(correction)


def compute_finite_process_readout_diagnostics(
    K: Array,
    d: Array,
    M: Array,
) -> FiniteProcessReadoutDiagnostics:
    capacity_fluxes = as_square_any(K, "symmetric_capacity_fluxes_K_ij_mol_m3_s")
    first_moments = np.asarray(d, dtype=float)
    second_moments = np.asarray(M, dtype=float)
    if first_moments.shape != (
        capacity_fluxes.shape[0],
        capacity_fluxes.shape[1],
        CARTESIAN,
    ):
        raise ValueError("d must have shape (n,n,3)")
    if second_moments.shape != (
        capacity_fluxes.shape[0],
        capacity_fluxes.shape[1],
        CARTESIAN,
        CARTESIAN,
    ):
        raise ValueError("M must have shape (n,n,3,3)")
    active_transition_mask = capacity_fluxes > PSD_TOL
    active_transition_count = int(
        np.count_nonzero(np.triu(active_transition_mask, k=1))
    )
    first_moment_norms = np.linalg.norm(
        first_moments.reshape(-1, CARTESIAN),
        axis=1,
    ).reshape(capacity_fluxes.shape)
    second_moment_traces = np.trace(second_moments, axis1=2, axis2=3)
    active_first_moment_count = int(
        np.count_nonzero(
            np.triu(active_transition_mask & (first_moment_norms > PSD_TOL), k=1)
        )
    )
    active_second_moment_count = int(
        np.count_nonzero(
            np.triu(active_transition_mask & (second_moment_traces > PSD_TOL), k=1)
        )
    )
    direct_only = active_transition_count == 0 or (
        active_first_moment_count == 0 and active_second_moment_count == 0
    )
    incomplete_reasons: list[str] = []
    if active_transition_count == 0:
        incomplete_reasons.append("no_active_transition_capacity_fluxes")
    if active_transition_count > 0 and active_first_moment_count == 0:
        incomplete_reasons.append("active_transitions_have_zero_first_moments")
    if active_transition_count > 0 and active_second_moment_count == 0:
        incomplete_reasons.append("active_transitions_have_zero_second_moments")
    readout_status = "projected"
    if direct_only:
        readout_status = "direct_only"
    return FiniteProcessReadoutDiagnostics(
        readout_status=readout_status,
        direct_only=direct_only,
        not_complete_reasons=tuple(incomplete_reasons),
        active_transition_capacity_flux_count=active_transition_count,
        active_transition_first_moment_count=active_first_moment_count,
        active_transition_second_moment_count=active_second_moment_count,
    )


def finite_process_readout_diagnostics_as_effect_attribution(
    diagnostics: FiniteProcessReadoutDiagnostics,
):
    return {
        "finite_process_readout_status": diagnostics.readout_status,
        "finite_process_direct_only": diagnostics.direct_only,
        "finite_process_not_complete_reasons": diagnostics.not_complete_reasons,
        "active_transition_capacity_flux_count": (
            diagnostics.active_transition_capacity_flux_count
        ),
        "active_transition_first_moment_count": (
            diagnostics.active_transition_first_moment_count
        ),
        "active_transition_second_moment_count": (
            diagnostics.active_transition_second_moment_count
        ),
    }


def compute_primitive_prediction_readiness_diagnostics(
    effect_attribution,
) -> PrimitivePredictionReadinessDiagnostics:
    not_complete_reasons: list[str] = []
    finite_process_status = str(effect_attribution["finite_process_readout_status"])
    finite_process_reasons = tuple(
        str(reason)
        for reason in effect_attribution["finite_process_not_complete_reasons"]
    )
    if finite_process_status != "projected":
        not_complete_reasons.append(f"finite_process_{finite_process_status}")
    not_complete_reasons.extend(finite_process_reasons)
    if "basis_refinement_convergence_status" not in effect_attribution:
        not_complete_reasons.append("basis_refinement_not_run")
    if "basis_refinement_convergence_status" in effect_attribution:
        basis_status = str(effect_attribution["basis_refinement_convergence_status"])
        if basis_status != "converged":
            not_complete_reasons.append(basis_status)
        basis_reasons = tuple(
            str(reason)
            for reason in effect_attribution["basis_refinement_not_complete_reasons"]
        )
        not_complete_reasons.extend(basis_reasons)
    unique_reasons = tuple(dict.fromkeys(not_complete_reasons))
    if unique_reasons:
        return PrimitivePredictionReadinessDiagnostics(
            readiness_status="incomplete",
            scalar_label="diagnostic",
            not_complete_reasons=unique_reasons,
        )
    return PrimitivePredictionReadinessDiagnostics(
        readiness_status="complete",
        scalar_label="primitive_prediction",
        not_complete_reasons=(),
    )


def primitive_prediction_readiness_as_effect_attribution(effect_attribution):
    diagnostics = compute_primitive_prediction_readiness_diagnostics(effect_attribution)
    return {
        "primitive_prediction_readiness_status": diagnostics.readiness_status,
        "primitive_prediction_scalar_label": diagnostics.scalar_label,
        "primitive_prediction_not_complete_reasons": diagnostics.not_complete_reasons,
    }


def basis_refinement_as_effect_attribution(refinement_result):
    return {
        "basis_refinement_convergence_status": str(
            refinement_result["convergence_status"]
        ),
        "basis_refinement_not_complete_reasons": tuple(
            str(reason) for reason in refinement_result["not_complete_reasons"]
        ),
        "basis_refinement_hard_convergence_failure": bool(
            refinement_result["hard_convergence_failure"]
        ),
        "basis_refinement_final_maximum_residual_score": np.asarray(
            refinement_result["final_maximum_residual_score"],
            dtype=float,
        ),
        "basis_refinement_final_conductivity_change_abs_S_m": np.asarray(
            refinement_result["final_conductivity_change_abs_S_m"],
            dtype=float,
        ),
    }


def compute_continuous_mori_correction(A: Array, h: Array) -> Array:
    A = np.asarray(A, dtype=float)
    h = np.asarray(h, dtype=float)
    if A.size == 0 or h.size == 0 or A.shape[0] == 0:
        return np.zeros((CARTESIAN, CARTESIAN), dtype=float)
    if np.max(np.abs(A)) == 0.0:
        if np.max(np.abs(h)) != 0.0:
            raise ValueError(
                "Mori coordinate has zero Dirichlet norm but nonzero current coupling"
            )
        return np.zeros((CARTESIAN, CARTESIAN), dtype=float)
    validate_psd(A, "mori_memory_matrix_A", allow_zero=True)
    memory_pseudoinverse = symmetric_psd_pseudoinverse(A)
    nullspace_projector = np.eye(A.shape[0]) - A @ memory_pseudoinverse
    for axis_index in range(CARTESIAN):
        coupling_vector = h[:, axis_index]
        coupling_norm = float(np.linalg.norm(coupling_vector))
        null_coupling_norm = float(
            np.linalg.norm(nullspace_projector @ coupling_vector)
        )
        if null_coupling_norm > MEMORY_NULLSPACE_RELATIVE_TOL * coupling_norm:
            raise ValueError(
                "Mori current coupling has nonzero projection into a null memory mode"
            )
    return _symmetrize(h.T @ memory_pseudoinverse @ h)


def validate_memory_schur_compatibility(
    direct_diffusivity_tensor: Array,
    finite_state_memory_correction_tensor: Array,
    mori_memory_matrix_A: Array,
    mori_current_coupling_matrix_h: Array,
    self_current_tensors_D_self_i_m2_s: Array,
    state_concentrations_mol_m3: Array,
    self_current_projector_ranks: tuple[int, ...],
) -> None:
    direct = as_matrix_shape(
        direct_diffusivity_tensor,
        (CARTESIAN, CARTESIAN),
        "direct_diffusivity_tensor",
    )
    finite_correction = as_matrix_shape(
        finite_state_memory_correction_tensor,
        (CARTESIAN, CARTESIAN),
        "finite_state_memory_correction_tensor",
    )
    memory_matrix = _symmetrize(
        as_square_any(mori_memory_matrix_A, "mori_memory_matrix_A")
    )
    memory_coupling = as_matrix_shape(
        mori_current_coupling_matrix_h,
        (memory_matrix.shape[0], CARTESIAN),
        "mori_current_coupling_matrix_h",
    )
    self_current_tensors = np.asarray(self_current_tensors_D_self_i_m2_s, dtype=float)
    concentrations = positive_vector(
        state_concentrations_mol_m3,
        "state_concentrations_mol_m3",
    )
    if self_current_tensors.shape != (concentrations.size, CARTESIAN, CARTESIAN):
        raise ValueError(
            "self_current_tensors_D_self_i_m2_s shape must match state count"
        )
    if memory_matrix.shape == (0, 0):
        if memory_coupling.shape != (0, CARTESIAN):
            raise ValueError("zero-memory Mori matrix requires zero-row coupling")
        return
    memory_pseudoinverse = symmetric_psd_pseudoinverse(memory_matrix)
    base_tensor = _symmetrize(direct - finite_correction)
    state_self_current_scores = concentrations * np.trace(
        self_current_tensors,
        axis1=1,
        axis2=2,
    )
    top_state_limit = min(DIAGNOSTIC_TOP_RECORD_COUNT, concentrations.size)
    top_state_order = np.argsort(-np.abs(state_self_current_scores))[:top_state_limit]
    top_states = tuple(
        {
            "state_index": int(state_index),
            "c_i_mol_m3": float(concentrations[state_index]),
            "c_i_trace_D_self": float(state_self_current_scores[state_index]),
        }
        for state_index in top_state_order
    )
    projector_rank_records = tuple(
        {
            "state_index": int(state_index),
            "projector_rank": int(projector_rank),
        }
        for state_index, projector_rank in enumerate(self_current_projector_ranks)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(memory_matrix)
    maximum_memory_eigenvalue = 0.0
    if eigenvalues.size > 0:
        maximum_memory_eigenvalue = float(np.max(eigenvalues))
    eigenvalue_tolerance = PSEUDOINVERSE_RELATIVE_TOL * maximum_memory_eigenvalue
    for axis_index in range(CARTESIAN):
        coupling_vector = memory_coupling[:, axis_index]
        memory_subtraction = float(
            coupling_vector @ memory_pseudoinverse @ coupling_vector
        )
        base_axis = float(base_tensor[axis_index, axis_index])
        schur_value = float(base_axis - memory_subtraction)
        schur_scale = max(1.0, abs(base_axis), abs(memory_subtraction))
        if schur_value < -PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL * schur_scale:
            projected_couplings = eigenvectors.T @ coupling_vector
            contributions = np.zeros_like(eigenvalues)
            active_eigenvalues = eigenvalues > eigenvalue_tolerance
            contributions[active_eigenvalues] = (
                projected_couplings[active_eigenvalues] ** 2
                / eigenvalues[active_eigenvalues]
            )
            top_memory_limit = min(DIAGNOSTIC_TOP_RECORD_COUNT, contributions.size)
            top_memory_order = np.argsort(-np.abs(contributions))[:top_memory_limit]
            top_memory_contributions = tuple(
                {
                    "memory_mode_index": int(mode_index),
                    "A_eigenvalue": float(eigenvalues[mode_index]),
                    "h_A_pinv_h_contribution": float(contributions[mode_index]),
                }
                for mode_index in top_memory_order
            )
            raise ValueError(
                "Mori Schur compatibility failed: "
                f"axis={axis_index}; "
                f"B_axis={float(direct[axis_index, axis_index])}; "
                f"C_Q_axis={float(finite_correction[axis_index, axis_index])}; "
                f"B_minus_C_Q_axis={base_axis}; "
                f"A_eigenvalues={tuple(float(value) for value in eigenvalues)}; "
                f"h_axis_norm={float(np.linalg.norm(coupling_vector))}; "
                f"h_T_A_pinv_h={memory_subtraction}; "
                f"schur_value={schur_value}; "
                f"top_memory_mode_contributions={top_memory_contributions}; "
                f"top_states_by_c_i_trace_D_self={top_states}; "
                f"projector_ranks_by_state={projector_rank_records}"
            )


def symmetric_psd_pseudoinverse(matrix: Array) -> Array:
    symmetric_matrix = _symmetrize(as_square_any(matrix, "symmetric_psd_matrix"))
    if symmetric_matrix.size == 0:
        return symmetric_matrix.copy()
    validate_psd(symmetric_matrix, "symmetric_psd_matrix", allow_zero=True)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    maximum_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    tolerance = PSEUDOINVERSE_RELATIVE_TOL * maximum_eigenvalue
    inverse_eigenvalues = np.asarray(
        [
            1.0 / eigenvalue if eigenvalue > tolerance else 0.0
            for eigenvalue in eigenvalues
        ],
        dtype=float,
    )
    return _symmetrize(eigenvectors @ np.diag(inverse_eigenvalues) @ eigenvectors.T)


def project_diffusivity_tensor_to_psd_roundoff(
    projected_diffusivity_tensor: Array,
) -> Array:
    projected = _symmetrize(
        as_matrix_shape(
            projected_diffusivity_tensor,
            (CARTESIAN, CARTESIAN),
            "projected_diffusivity_tensor",
        )
    )
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    minimum_eigenvalue = float(np.min(eigenvalues))
    maximum_abs_eigenvalue = float(np.max(np.abs(eigenvalues)))
    if minimum_eigenvalue >= 0.0:
        return projected
    if maximum_abs_eigenvalue == 0.0:
        return np.zeros((CARTESIAN, CARTESIAN), dtype=float)
    if (
        minimum_eigenvalue
        < -PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL * maximum_abs_eigenvalue
    ):
        raise ValueError(
            "projected diffusivity tensor has a physical negative eigenvalue"
        )
    clipped_eigenvalues = np.maximum(eigenvalues, 0.0)
    return _symmetrize(eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T)


def conductivity_from_projected_diffusivity(
    projected_diffusivity_tensor: Array, temperature_K: float
) -> float:
    D = as_matrix_shape(
        projected_diffusivity_tensor,
        (CARTESIAN, CARTESIAN),
        "projected_diffusivity_tensor",
    )
    return (
        F_C_PER_MOL
        * F_C_PER_MOL
        / (R_J_PER_MOL_K * positive_float(temperature_K, "temperature_K"))
        * float(np.trace(D))
        / CARTESIAN
    )


def compute_state_charge_mobility_tensor(
    charge_numbers: Array,
    charged_center_mobility_m2_s: Array,
) -> float:
    charges = as_1d(charge_numbers, "charge_numbers")
    mobility = as_square(
        charged_center_mobility_m2_s,
        charges.size,
        "charged_center_mobility_m2_s",
    )
    validate_psd(mobility, "charged_center_mobility_m2_s", allow_zero=True)
    return float(charges @ mobility @ charges)


def compute_charge_polarization_gradient_by_finite_difference(
    charge_polarization: Callable[[Array], Array],
    point: Array,
    coordinate_steps: Array,
) -> Array:
    return central_difference_jacobian_with_steps(
        charge_polarization,
        point,
        output_count=CARTESIAN,
        coordinate_steps=as_1d(coordinate_steps, "coordinate_steps"),
    )


def compute_memory_coordinate_gradient_by_finite_difference(
    memory_coordinates: Callable[[Array], Array],
    point: Array,
    coordinate_steps: Array,
) -> Array:
    point_array = np.asarray(point, dtype=float)
    memory_count = int(np.asarray(memory_coordinates(point_array), dtype=float).size)
    return central_difference_jacobian_with_steps(
        memory_coordinates,
        point_array,
        output_count=memory_count,
        coordinate_steps=as_1d(coordinate_steps, "coordinate_steps"),
    )


def score_candidate_mori_coordinates(
    current_mori_memory_matrix_A: Array,
    current_mori_current_coupling_matrix_h: Array,
    candidate_self_energies_A_gg: Array,
    candidate_cross_energies_A_gPhi: Array,
    candidate_current_couplings_h_g: Array,
) -> dict[str, Array]:
    current_memory = as_square_any(
        current_mori_memory_matrix_A,
        "current_mori_memory_matrix_A",
    )
    current_coupling = as_2d(
        current_mori_current_coupling_matrix_h,
        "current_mori_current_coupling_matrix_h",
    )
    candidate_self = as_1d(candidate_self_energies_A_gg, "candidate_self_energies")
    candidate_cross = as_2d(
        candidate_cross_energies_A_gPhi,
        "candidate_cross_energies",
    )
    candidate_coupling = as_2d(
        candidate_current_couplings_h_g,
        "candidate_current_couplings",
    )
    current_inverse = symmetric_psd_pseudoinverse(current_memory)
    residual_coupling = (
        candidate_coupling - candidate_cross @ current_inverse @ current_coupling
    )
    residual_energy = candidate_self - np.einsum(
        "ij,jk,ik->i",
        candidate_cross,
        current_inverse,
        candidate_cross,
    )
    coupling_norms = np.sum(residual_coupling * residual_coupling, axis=1)
    scores = np.zeros_like(residual_energy, dtype=float)
    valid_energy = residual_energy > PSD_TOL
    scores[valid_energy] = coupling_norms[valid_energy] / residual_energy[valid_energy]
    return {
        "residual_coupling": residual_coupling,
        "residual_energy": residual_energy,
        "scores": scores,
    }


def refine_mori_basis_by_projected_residual(
    direct_diffusivity_tensor: Array,
    initial_mori_memory_matrix_A: Array,
    initial_mori_current_coupling_matrix_h: Array,
    candidate_self_energies_A_gg: Array,
    candidate_cross_energies_A_gPhi: Array,
    candidate_cross_energy_matrix: Array,
    candidate_current_couplings_h_g: Array,
    temperature_K: float,
    residual_score_tolerance: float,
    conductivity_change_tolerance_S_m: float,
    max_added_coordinates: int,
) -> dict[str, Any]:
    selected_candidate_indices: list[int] = []
    current_memory = as_square_any(initial_mori_memory_matrix_A, "initial_A").copy()
    current_coupling = as_2d(initial_mori_current_coupling_matrix_h, "initial_h").copy()
    if current_coupling.shape != (current_memory.shape[0], CARTESIAN):
        raise ValueError(
            "initial_h must have one Cartesian coupling row per Mori basis"
        )
    if current_memory.size > 0:
        validate_psd(current_memory, "initial_A", allow_zero=True)
    candidate_self = as_1d(candidate_self_energies_A_gg, "candidate_A_gg")
    candidate_cross = as_2d(candidate_cross_energies_A_gPhi, "candidate_A_gPhi")
    candidate_matrix = as_square_any(candidate_cross_energy_matrix, "candidate_A")
    candidate_coupling = as_2d(candidate_current_couplings_h_g, "candidate_h")
    candidate_count = candidate_self.size
    if candidate_cross.shape != (candidate_count, current_memory.shape[0]):
        raise ValueError(
            "candidate_A_gPhi shape must match candidate count and initial basis"
        )
    if candidate_matrix.shape != (candidate_count, candidate_count):
        raise ValueError("candidate_A shape must match candidate count")
    if candidate_coupling.shape != (candidate_count, CARTESIAN):
        raise ValueError(
            "candidate_h must have one Cartesian coupling row per candidate"
        )
    validate_psd(candidate_matrix, "candidate_A", allow_zero=True)
    if max_added_coordinates < 0:
        raise ValueError("max_added_coordinates must be nonnegative")
    score_tolerance = positive_float(
        residual_score_tolerance,
        "residual_score_tolerance",
    )
    conductivity_tolerance = positive_float(
        conductivity_change_tolerance_S_m,
        "conductivity_change_tolerance_S_m",
    )
    direct_tensor = as_matrix_shape(
        direct_diffusivity_tensor,
        (CARTESIAN, CARTESIAN),
        "direct_diffusivity_tensor",
    )
    conductivity_history = [
        conductivity_from_projected_diffusivity(
            direct_tensor
            - compute_continuous_mori_correction(current_memory, current_coupling),
            temperature_K,
        )
    ]
    maximum_score_history: list[float] = []
    conductivity_change_history: list[float] = []
    selected_score_history: list[float] = []
    selected_residual_energy_history: list[float] = []
    selected_residual_coupling_norm_history: list[float] = []
    available_candidates = np.ones(candidate_self.size, dtype=bool)
    discarded_candidate_indices: list[int] = []
    rejected_null_energy_indices: list[int] = []
    rejected_current_spanning_indices: list[int] = []
    rejected_psd_indices: list[int] = []
    energy_scale = max(
        _maximum_abs_entry(candidate_matrix),
        _maximum_abs_entry(current_memory),
        PSD_TOL,
    )
    coupling_scale = max(
        _maximum_abs_entry(candidate_coupling),
        _maximum_abs_entry(current_coupling),
        PSD_TOL,
    )
    energy_threshold = MEMORY_NULLSPACE_RELATIVE_TOL * energy_scale
    null_current_threshold = MEMORY_NULLSPACE_RELATIVE_TOL * coupling_scale
    conductivity_change = 0.0
    final_maximum_score = np.inf
    for _iteration_index in range(max_added_coordinates + 1):
        dynamic_candidate_cross = np.hstack(
            [
                candidate_cross,
                candidate_matrix[:, selected_candidate_indices],
            ]
        )
        score_result = score_candidate_mori_coordinates(
            current_memory,
            current_coupling,
            candidate_self,
            dynamic_candidate_cross,
            candidate_coupling,
        )
        current_correction = compute_continuous_mori_correction(
            current_memory,
            current_coupling,
        )
        remaining_tensor = _symmetrize(direct_tensor - current_correction)
        eligible_candidates = available_candidates.copy()
        residual_energies = np.asarray(score_result["residual_energy"], dtype=float)
        residual_couplings = np.asarray(score_result["residual_coupling"], dtype=float)
        for candidate_index in np.flatnonzero(available_candidates):
            residual_energy = float(residual_energies[candidate_index])
            residual_coupling = residual_couplings[candidate_index]
            residual_coupling_norm = float(np.linalg.norm(residual_coupling))
            if residual_energy <= energy_threshold:
                available_candidates[candidate_index] = False
                eligible_candidates[candidate_index] = False
                if residual_coupling_norm <= null_current_threshold:
                    discarded_candidate_indices.append(int(candidate_index))
                else:
                    rejected_null_energy_indices.append(int(candidate_index))
                continue
            incremental_correction = _symmetrize(
                np.outer(residual_coupling, residual_coupling) / residual_energy
            )
            candidate_remaining_tensor = _symmetrize(
                remaining_tensor - incremental_correction
            )
            if _memory_candidate_spans_current(
                incremental_correction,
                remaining_tensor,
                PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL,
            ):
                available_candidates[candidate_index] = False
                eligible_candidates[candidate_index] = False
                rejected_current_spanning_indices.append(int(candidate_index))
                continue
            if _minimum_eigenvalue(
                candidate_remaining_tensor
            ) < -PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL * max(
                1.0,
                _maximum_abs_eigenvalue(remaining_tensor),
            ):
                available_candidates[candidate_index] = False
                eligible_candidates[candidate_index] = False
                rejected_psd_indices.append(int(candidate_index))
        scores = np.where(eligible_candidates, score_result["scores"], -np.inf)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            final_maximum_score = 0.0
        else:
            final_maximum_score = float(np.max(finite_scores))
        maximum_score_history.append(final_maximum_score)
        if (
            final_maximum_score <= score_tolerance
            and conductivity_change <= conductivity_tolerance
        ):
            break
        if _iteration_index == max_added_coordinates:
            break
        selected_index = int(np.argmax(scores))
        if not np.isfinite(scores[selected_index]):
            break
        selected_candidate_indices.append(selected_index)
        selected_score_history.append(float(scores[selected_index]))
        selected_residual_energy_history.append(
            float(residual_energies[selected_index])
        )
        selected_residual_coupling_norm_history.append(
            float(np.linalg.norm(residual_couplings[selected_index]))
        )
        available_candidates[selected_index] = False
        cross_row = dynamic_candidate_cross[selected_index : selected_index + 1]
        current_memory = np.block(
            [
                [current_memory, cross_row.T],
                [
                    cross_row,
                    candidate_matrix[
                        selected_index : selected_index + 1,
                        selected_index : selected_index + 1,
                    ],
                ],
            ]
        )
        current_coupling = np.vstack(
            [current_coupling, candidate_coupling[selected_index : selected_index + 1]]
        )
        conductivity_history.append(
            conductivity_from_projected_diffusivity(
                direct_tensor
                - compute_continuous_mori_correction(current_memory, current_coupling),
                temperature_K,
            )
        )
        conductivity_change = abs(conductivity_history[-1] - conductivity_history[-2])
        conductivity_change_history.append(float(conductivity_change))
    convergence_status = "converged"
    not_complete_reasons: list[str] = []
    if final_maximum_score > score_tolerance:
        convergence_status = "basis_residual_above_tolerance"
        not_complete_reasons.append("basis_residual_score_above_tolerance")
    elif conductivity_change > conductivity_tolerance:
        convergence_status = "conductivity_change_above_tolerance"
        not_complete_reasons.append("conductivity_change_above_tolerance")
    if (
        convergence_status != "converged"
        and len(selected_candidate_indices) >= max_added_coordinates
    ):
        not_complete_reasons.append("max_added_coordinates_exhausted")
    return {
        "selected_candidate_indices": np.asarray(
            selected_candidate_indices,
            dtype=int,
        ),
        "discarded_candidate_indices": np.asarray(
            discarded_candidate_indices,
            dtype=int,
        ),
        "rejected_null_energy_candidate_indices": np.asarray(
            rejected_null_energy_indices,
            dtype=int,
        ),
        "rejected_current_spanning_candidate_indices": np.asarray(
            rejected_current_spanning_indices,
            dtype=int,
        ),
        "rejected_psd_candidate_indices": np.asarray(
            rejected_psd_indices,
            dtype=int,
        ),
        "final_mori_memory_matrix_A": current_memory,
        "final_mori_current_coupling_matrix_h": current_coupling,
        "conductivity_history_S_m": np.asarray(conductivity_history, dtype=float),
        "maximum_residual_score_history": np.asarray(
            maximum_score_history,
            dtype=float,
        ),
        "conductivity_change_history_abs_S_m": np.asarray(
            conductivity_change_history,
            dtype=float,
        ),
        "selected_residual_score_history": np.asarray(
            selected_score_history,
            dtype=float,
        ),
        "selected_residual_energy_history": np.asarray(
            selected_residual_energy_history,
            dtype=float,
        ),
        "selected_residual_coupling_norm_history": np.asarray(
            selected_residual_coupling_norm_history,
            dtype=float,
        ),
        "final_maximum_residual_score": np.asarray(final_maximum_score, dtype=float),
        "final_conductivity_change_abs_S_m": np.asarray(
            conductivity_change,
            dtype=float,
        ),
        "final_sigma_S_m": np.asarray(conductivity_history[-1], dtype=float),
        "convergence_status": convergence_status,
        "not_complete_reasons": tuple(not_complete_reasons),
        "hard_convergence_failure": convergence_status != "converged",
    }


def conductivity_effect_primitive_locations() -> dict[str, tuple[str, ...]]:
    return {
        "free_ion_fraction": ("c_i",),
        "ion_association": ("c_i", "K_ij", "Q_ij"),
        "SSIP_CIP_balance": ("c_i", "K_ij", "D_self_i"),
        "aggregation": ("c_i", "K_ij", "D_self_i", "M_ij"),
        "neutral_ligand_coordination": ("c_i", "K_ij", "A", "h"),
        "Li_anion_anticorrelation": ("D_self_i", "M_ij", "A", "h"),
        "Li_anion_comotion": ("D_self_i", "M_ij", "A", "h"),
        "identity_diffusion": ("K_ij", "d_ij", "M_ij"),
        "partner_switching": ("K_ij", "d_ij", "M_ij", "A", "h"),
        "cage_backjump": ("C_Q", "A", "h"),
        "ion_atmosphere_relaxation": ("A", "h"),
    }


def compute_primitive_ownership_scores(
    plus_result: ProjectedConductivityResult,
    minus_result: ProjectedConductivityResult,
    perturbation_delta: float,
) -> dict[str, float]:
    denominator = 2.0 * positive_float(perturbation_delta, "perturbation_delta")
    derivative_c = (
        as_1d(plus_result.state_concentrations_mol_m3, "plus_c")
        - as_1d(minus_result.state_concentrations_mol_m3, "minus_c")
    ) / denominator
    derivative_K = (
        as_square_any(plus_result.symmetric_capacity_fluxes_K_ij_mol_m3_s, "plus_K")
        - as_square_any(minus_result.symmetric_capacity_fluxes_K_ij_mol_m3_s, "minus_K")
    ) / denominator
    derivative_d = (
        np.asarray(plus_result.transition_first_moments_d_ij_m, dtype=float)
        - np.asarray(minus_result.transition_first_moments_d_ij_m, dtype=float)
    ) / denominator
    derivative_M = (
        np.asarray(plus_result.transition_second_moments_M_ij_m2, dtype=float)
        - np.asarray(minus_result.transition_second_moments_M_ij_m2, dtype=float)
    ) / denominator
    derivative_D_self = (
        np.asarray(plus_result.self_current_tensors_D_self_i_m2_s, dtype=float)
        - np.asarray(minus_result.self_current_tensors_D_self_i_m2_s, dtype=float)
    ) / denominator
    derivative_A = (
        as_square_any(plus_result.mori_memory_matrix_A, "plus_A")
        - as_square_any(minus_result.mori_memory_matrix_A, "minus_A")
    ) / denominator
    derivative_h = (
        as_2d(plus_result.mori_current_coupling_matrix_h, "plus_h")
        - as_2d(minus_result.mori_current_coupling_matrix_h, "minus_h")
    ) / denominator
    derivative_sigma = (
        float(plus_result.sigma_mS_cm) - float(minus_result.sigma_mS_cm)
    ) / denominator
    scores = {
        "S_c": float(np.linalg.norm(derivative_c)),
        "S_K": float(np.linalg.norm(derivative_K)),
        "S_dM": float(np.linalg.norm(derivative_d) + np.linalg.norm(derivative_M)),
        "S_D": float(np.linalg.norm(derivative_D_self)),
        "S_Ah": float(np.linalg.norm(derivative_A) + np.linalg.norm(derivative_h)),
        "d_sigma_mS_cm": float(derivative_sigma),
    }
    primitive_scores = {
        primitive_name: score
        for primitive_name, score in scores.items()
        if primitive_name.startswith("S_")
    }
    largest_primitive = max(primitive_scores, key=primitive_scores.__getitem__)
    scores["largest_primitive_score"] = primitive_scores[largest_primitive]
    scores["largest_primitive_index"] = float(
        ("S_c", "S_K", "S_dM", "S_D", "S_Ah").index(largest_primitive)
    )
    return scores


def solve_weighted_poisson(Q: Array, c: Array, b: Array) -> Array:
    if np.max(np.abs(b)) == 0.0:
        return np.zeros_like(b, dtype=float)
    components = _generator_connected_components(Q)
    solution = np.zeros_like(b, dtype=float)
    for component_indices in components:
        component_drift = np.asarray(b[component_indices], dtype=float)
        if np.max(np.abs(component_drift)) == 0.0:
            continue
        component_concentrations = np.asarray(c[component_indices], dtype=float)
        weighted_drift = float(component_concentrations @ component_drift)
        weighted_drift_scale = float(
            np.sum(np.abs(component_concentrations * component_drift))
        )
        solvability_tolerance = max(
            POISSON_SOLVABILITY_ABS_TOL,
            POISSON_SOLVABILITY_EPSILON_FACTOR
            * np.finfo(float).eps
            * weighted_drift_scale,
        )
        if abs(weighted_drift) > solvability_tolerance:
            raise ValueError(
                "finite-state drift is not solvable on a generator component"
            )
        component_Q = np.asarray(
            Q[np.ix_(component_indices, component_indices)],
            dtype=float,
        )
        component_size = component_indices.size
        system = np.block(
            [
                [-component_Q, np.ones((component_size, 1))],
                [
                    component_concentrations[np.newaxis, :],
                    np.zeros((1, 1)),
                ],
            ]
        )
        rhs = np.concatenate((component_drift, np.asarray([0.0])))
        component_solution = np.linalg.solve(system, rhs)[:component_size]
        solution[component_indices] = component_solution
    return solution


def _generator_connected_components(Q: Array) -> tuple[Array, ...]:
    generator = np.asarray(Q, dtype=float)
    state_count = generator.shape[0]
    adjacency = (np.abs(generator) > 0.0) | (np.abs(generator.T) > 0.0)
    visited = np.zeros(state_count, dtype=bool)
    components = []
    for start_index in range(state_count):
        if visited[start_index]:
            continue
        stack = [start_index]
        visited[start_index] = True
        component = []
        while stack:
            state_index = stack.pop()
            component.append(state_index)
            neighbor_indices = np.flatnonzero(adjacency[state_index])
            for neighbor_index in neighbor_indices:
                if visited[neighbor_index]:
                    continue
                visited[neighbor_index] = True
                stack.append(int(neighbor_index))
        components.append(np.asarray(component, dtype=int))
    return tuple(components)


def compute_effect_attribution(
    c: Array, K: Array, M: Array, Dself: Array, Cq: Array, Cm: Array, Dproj: Array
) -> dict[str, Any]:
    state_self = np.einsum("i,iab->iab", c, Dself)
    edge_direct = 0.5 * np.einsum("ij,ijab->ijab", K, M)
    state_D_Q = np.trace(Dself, axis1=1, axis2=2) / CARTESIAN
    return {
        "trace_state_self_current_by_state": np.trace(state_self, axis1=1, axis2=2),
        "state_D_Q_zDz_m2_s": state_D_Q,
        "state_c_i_D_Q_mol_m_s": c * state_D_Q,
        "trace_transition_direct_by_edge": np.trace(edge_direct, axis1=2, axis2=3),
        "trace_direct_total": float(
            np.trace(np.sum(state_self, axis=0) + np.sum(edge_direct, axis=(0, 1)))
        ),
        "trace_finite_state_memory_correction": float(np.trace(Cq)),
        "trace_continuous_mori_correction": float(np.trace(Cm)),
        "trace_projected_diffusivity": float(np.trace(Dproj)),
    }


def validate_generator_input(x: ProjectedGeneratorInput) -> None:
    positive_float(x.temperature_K, "temperature_K")
    positive_vector(
        x.total_component_concentrations_mol_m3,
        "total_component_concentrations_mol_m3",
    )
    validate_basin_stoichiometry(
        x.basin_stoichiometry,
        len(x.basin_quadrature_points),
        np.asarray(x.total_component_concentrations_mol_m3, dtype=float).size,
    )
    positive_float(x.volume_m3, "volume_m3")
    validate_basin_quadrature(x.basin_quadrature_points, x.basin_quadrature_weights)
    state_count = len(x.basin_quadrature_points)
    validate_full_generator_projectors_are_identity(
        x.self_current_coordinate_projectors,
        state_count,
        _infer_coordinate_dim(x.basin_quadrature_points),
    )
    validate_transition_inputs(
        x.transition_pair_indices,
        (
            x.transition_quadrature_points,
            x.transition_quadrature_weights,
            x.transition_committor_gradients,
            x.transition_surface_state_indices,
            x.transition_path_displacements_m,
            x.transition_path_weights,
        ),
        state_count,
    )
    log_capacity_integrals = as_1d(
        x.transition_log_capacity_integrals,
        "transition_log_capacity_integrals",
    )
    transition_count = as_pairs(x.transition_pair_indices, state_count).shape[0]
    if log_capacity_integrals.size != transition_count:
        raise ValueError(
            "transition_log_capacity_integrals length must equal transition count"
        )
    residence_mode = np.asarray(x.transition_uses_residence_rate_constants, dtype=bool)
    if residence_mode.shape != (transition_count,):
        raise ValueError(
            "transition_uses_residence_rate_constants length must equal transition count"
        )
    residence_rates = as_1d(
        x.transition_residence_rate_constants_s_inv,
        "transition_residence_rate_constants_s_inv",
    )
    if residence_rates.size != transition_count:
        raise ValueError(
            "transition_residence_rate_constants_s_inv length must equal transition count"
        )
    if np.any(residence_rates[residence_mode] <= 0.0):
        raise ValueError("residence-rate transitions require positive rate constants")
    if np.any(residence_rates[~residence_mode] != 0.0):
        raise ValueError("capacity-integral transitions require zero residence rates")
    transition_moments_from_generator_input(x, state_count)


def validate_full_generator_projectors_are_identity(
    self_current_coordinate_projectors: Sequence[Array],
    state_count: int,
    coordinate_dimension: int,
) -> None:
    if len(self_current_coordinate_projectors) != state_count:
        raise ValueError(
            "self_current_coordinate_projectors length must equal state count"
        )
    identity_projector = np.eye(coordinate_dimension, dtype=float)
    for state_index, projector in enumerate(self_current_coordinate_projectors):
        projector_matrix = as_matrix_shape(
            projector,
            (coordinate_dimension, coordinate_dimension),
            f"self_current_coordinate_projectors[{state_index}]",
        )
        if not np.allclose(projector_matrix, identity_projector):
            raise ValueError(
                "full-generator conductivity path requires identity "
                f"self_current_coordinate_projectors[{state_index}]"
            )


def validate_physical_library(physical_library: ConductivityPhysicalLibrary) -> None:
    validate_generator_input(physical_library.generator_input)
    state_count = len(physical_library.generator_input.basin_quadrature_points)
    if len(physical_library.state_labels) != state_count:
        raise ValueError("physical library state labels do not match basin count")
    if len(set(physical_library.state_labels)) != len(physical_library.state_labels):
        raise ValueError("physical library state labels must be unique")
    transition_pairs = as_pairs(
        physical_library.generator_input.transition_pair_indices,
        state_count,
    )
    transition_count = int(transition_pairs.shape[0])
    if len(physical_library.transition_labels) != transition_count:
        raise ValueError(
            "physical library transition labels do not match transition count"
        )
    if len(set(physical_library.transition_labels)) != len(
        physical_library.transition_labels
    ):
        raise ValueError("physical library transition labels must be unique")


def validate_function_input(x: FunctionGeneratorInput) -> None:
    positive_float(x.temperature_K, "temperature_K")
    positive_vector(
        x.total_component_concentrations_mol_m3,
        "total_component_concentrations_mol_m3",
    )
    validate_basin_stoichiometry(
        x.basin_stoichiometry,
        len(x.basin_quadrature_points),
        np.asarray(x.total_component_concentrations_mol_m3, dtype=float).size,
    )
    positive_float(x.volume_m3, "volume_m3")
    positive_float(x.finite_difference_relative_step, "finite_difference_relative_step")
    validate_basin_quadrature(x.basin_quadrature_points, x.basin_quadrature_weights)
    pairs = as_pairs(x.transition_pair_indices, len(x.basin_quadrature_points))
    validate_equal_lengths(
        pairs.shape[0],
        (
            x.transition_quadrature_points,
            x.transition_quadrature_weights,
            x.transition_committor_gradients,
            x.transition_surface_state_indices,
            x.transition_path_start_points,
            x.transition_path_end_points,
            x.transition_path_weights,
        ),
    )


def validate_primitive_input(
    x: ProjectedPrimitiveInput,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    c = positive_vector(x.state_concentrations_mol_m3, "state_concentrations_mol_m3")
    n = c.size
    K = as_square(
        x.symmetric_capacity_fluxes_K_ij_mol_m3_s,
        n,
        "symmetric_capacity_fluxes_K_ij_mol_m3_s",
    )
    if np.any(K < -PSD_TOL):
        raise ValueError("K must be nonnegative")
    if not np.allclose(K, K.T, atol=1e-10, rtol=1e-10):
        raise ValueError("capacity flux matrix must be symmetric")
    if not np.allclose(np.diag(K), 0.0):
        raise ValueError("K diagonal must be zero")
    d = np.asarray(x.transition_first_moments_d_ij_m, dtype=float)
    if d.shape != (n, n, CARTESIAN) or not np.all(np.isfinite(d)):
        raise ValueError("d must have shape (n,n,3)")
    if not np.allclose(d + np.swapaxes(d, 0, 1), 0.0, atol=1e-12, rtol=1e-12):
        raise ValueError("d_ji must equal -d_ij")
    disp_norm = np.linalg.norm(d.reshape(-1, CARTESIAN), axis=1)
    if np.any(
        disp_norm
        > positive_float(
            x.max_transition_displacement_m, "max_transition_displacement_m"
        )
    ):
        raise ValueError("transition first moment exceeds max displacement")
    M = np.asarray(x.transition_second_moments_M_ij_m2, dtype=float)
    if M.shape != (n, n, CARTESIAN, CARTESIAN) or not np.all(np.isfinite(M)):
        raise ValueError("M must have shape (n,n,3,3)")
    if not np.allclose(M, np.swapaxes(M, 0, 1), atol=1e-12, rtol=1e-12):
        raise ValueError("M_ji must equal M_ij")
    for i in range(n):
        for j in range(n):
            if not np.allclose(M[i, j], M[i, j].T, atol=1e-12, rtol=1e-12):
                raise ValueError("each M_ij must be symmetric")
            validate_psd(M[i, j], f"M[{i},{j}]", allow_zero=True)
    Dself = np.asarray(x.self_current_tensors_D_self_i_m2_s, dtype=float)
    if Dself.shape != (n, CARTESIAN, CARTESIAN) or not np.all(np.isfinite(Dself)):
        raise ValueError("D_self must have shape (n,3,3)")
    for i in range(n):
        if not np.allclose(Dself[i], Dself[i].T, atol=1e-12, rtol=1e-12):
            raise ValueError("each D_self_i must be symmetric")
        validate_psd(Dself[i], f"D_self[{i}]", allow_zero=True)
    A = np.asarray(x.mori_memory_matrix_A, dtype=float)
    h = np.asarray(x.mori_current_coupling_matrix_h, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.all(np.isfinite(A)):
        raise ValueError("A must be finite square")
    if h.shape != (A.shape[0], CARTESIAN) or not np.all(np.isfinite(h)):
        raise ValueError("h must have shape (memory_count,3)")
    if A.size:
        if not np.allclose(A, A.T, atol=1e-12, rtol=1e-12):
            raise ValueError("A must be symmetric")
        validate_psd(A, "A", allow_zero=True)
    positive_float(x.temperature_K, "temperature_K")
    positive_float(x.volume_m3, "volume_m3")
    return c, K, d, M, Dself, A, h


def validate_basin_quadrature(
    points: Sequence[Array], weights: Sequence[Array]
) -> None:
    if len(points) == 0:
        raise ValueError("at least one basin is required")
    if len(points) != len(weights):
        raise ValueError("basin quadrature points/weights length mismatch")
    dim = None
    for p, w in zip(points, weights, strict=True):
        p2 = as_2d(p, "basin_quadrature_points[]")
        w1 = as_1d(w, "basin_quadrature_weights[]")
        if p2.shape[0] != w1.size:
            raise ValueError("basin quadrature point/weight count mismatch")
        if np.any(w1 < 0):
            raise ValueError("basin quadrature weights must be nonnegative")
        if dim is None:
            dim = p2.shape[1]
        elif p2.shape[1] != dim:
            raise ValueError(
                "all basin quadrature points must have same coordinate dimension"
            )


def validate_transition_inputs(
    pair_indices: Array,
    transition_input_sequences: tuple[Sequence[Array], ...],
    state_count: int,
) -> None:
    pairs = as_pairs(pair_indices, state_count)
    validate_equal_lengths(
        pairs.shape[0],
        transition_input_sequences,
    )


def validate_basin_stoichiometry(
    basin_stoichiometry: Array,
    state_count: int,
    component_count: int,
) -> None:
    stoichiometry = np.asarray(basin_stoichiometry, dtype=float)
    if stoichiometry.shape != (state_count, component_count):
        raise ValueError(
            f"basin_stoichiometry must have shape ({state_count},{component_count})"
        )
    if not np.all(np.isfinite(stoichiometry)):
        raise ValueError("basin_stoichiometry must be finite")
    if np.any(stoichiometry < 0.0):
        raise ValueError("basin_stoichiometry entries must be nonnegative")
    if np.any(np.sum(stoichiometry, axis=0) <= 0.0):
        raise ValueError("each conserved component must appear in at least one basin")


def central_difference_jacobian(
    function: Callable[[Array], Array],
    point: Array,
    output_count: int,
    coordinate_count: int,
    relative_step: float,
) -> Array:
    q = np.asarray(point, dtype=float)
    J = np.zeros((output_count, coordinate_count), dtype=float)
    for k in range(coordinate_count):
        step = positive_float(relative_step, "relative_step") * max(
            1.0, abs(float(q[k]))
        )
        q_plus = q.copy()
        q_plus[k] += step
        q_minus = q.copy()
        q_minus[k] -= step
        J[:, k] = (
            np.asarray(function(q_plus), dtype=float)
            - np.asarray(function(q_minus), dtype=float)
        ) / (2.0 * step)
    return J


def central_difference_jacobian_with_steps(
    function: Callable[[Array], Array],
    point: Array,
    output_count: int,
    coordinate_steps: Array,
) -> Array:
    q = np.asarray(point, dtype=float)
    steps = as_1d(coordinate_steps, "coordinate_steps")
    if q.ndim != 1 or q.size != steps.size:
        raise ValueError("point and coordinate_steps must be matching 1D arrays")
    jacobian = np.zeros((output_count, q.size), dtype=float)
    for coordinate_index in range(q.size):
        step = positive_float(float(steps[coordinate_index]), "coordinate_step")
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[coordinate_index] += step
        q_minus[coordinate_index] -= step
        jacobian[:, coordinate_index] = (
            np.asarray(function(q_plus), dtype=float)
            - np.asarray(function(q_minus), dtype=float)
        ) / (2.0 * step)
    return jacobian


def cumulative_trapezoid(coordinate_grid: Array, values: Array) -> Array:
    grid = as_1d(coordinate_grid, "coordinate_grid")
    integrand = as_1d(values, "values")
    if grid.size != integrand.size:
        raise ValueError("coordinate_grid and values must have the same length")
    cumulative = np.zeros_like(grid, dtype=float)
    for index in range(1, grid.size):
        spacing = grid[index] - grid[index - 1]
        cumulative[index] = cumulative[index - 1] + 0.5 * spacing * (
            integrand[index] + integrand[index - 1]
        )
    return cumulative


def as_pairs(value: Array, state_count: int) -> Array:
    arr = np.asarray(value, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("transition_pair_indices must have shape (edge_count,2)")
    for i, j in arr:
        if i < 0 or j < 0 or i >= state_count or j >= state_count or i == j:
            raise ValueError("invalid transition pair")
    return arr


def validate_equal_lengths(expected: int, items: tuple[Sequence[Array], ...]) -> None:
    for item in items:
        if len(item) != expected:
            raise ValueError("transition input tuple lengths must equal edge count")


def as_1d(value: Array, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1D array")
    return arr


def as_2d(value: Array, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 2D array")
    return arr


def as_square(value: Array, dimension: int, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (dimension, dimension) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must have shape ({dimension},{dimension})")
    return arr


def as_square_any(value: Array, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite square matrix")
    return arr


def as_matrix_shape(value: Array, shape: tuple[int, int], name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must have shape {shape}")
    return arr


def positive_float(value: float, name: str) -> float:
    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be positive")
    return x


def positive_vector(value: Array, name: str) -> Array:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be a finite positive vector")
    return arr


def validate_psd(matrix: Array, name: str, allow_zero: bool = False) -> None:
    M = _symmetrize(np.asarray(matrix, dtype=float))
    eig = np.linalg.eigvalsh(M)
    if np.min(eig) < -PSD_TOL * max(1.0, float(np.max(np.abs(eig)))):
        raise ValueError(f"{name} must be positive semidefinite")
    if not allow_zero and np.max(eig) <= PSD_TOL:
        raise ValueError(f"{name} must be nonzero positive semidefinite")


def _infer_coordinate_dim(points: Sequence[Array]) -> int:
    validate_basin_quadrature(
        points, tuple(np.ones(np.asarray(p).shape[0]) for p in points)
    )
    return int(np.asarray(points[0], dtype=float).shape[1])


def _symmetrize(matrix: Array) -> Array:
    M = np.asarray(matrix, dtype=float)
    return 0.5 * (M + M.T)


def run_self_tests() -> None:
    def U(q: Array) -> float:
        return 0.0

    def D(q: Array) -> Array:
        return np.eye(1)

    def gradP(q: Array) -> Array:
        return np.asarray([[1.0], [0.0], [0.0]], dtype=float)

    def gradP0(q: Array) -> Array:
        return np.zeros((3, 1), dtype=float)

    def gradpsi0(q: Array) -> Array:
        return np.zeros((0, 1), dtype=float)

    basin_points = (np.asarray([[-1.0]]), np.asarray([[1.0]]))
    basin_weights = (np.asarray([1.0]), np.asarray([1.0]))
    pair_idx = np.asarray([[0, 1]], dtype=int)
    trans_points = (np.asarray([[0.0]]),)
    trans_weights = (np.asarray([1.0]),)
    comm_grad_zero = (np.asarray([[0.0]]),)
    trans_surface_states = (np.asarray([0], dtype=int),)
    path_disp_zero = (np.asarray([[0.0, 0.0, 0.0]]),)
    path_weights = (np.asarray([1.0]),)
    component_totals = np.asarray([2.0], dtype=float)
    basin_stoichiometry = np.asarray([[1.0], [1.0]], dtype=float)
    projectors = (np.eye(1), np.eye(1))
    self_test_temperature_K = T_REF_K

    neutral = compute_projected_analytical_conductivity(
        U,
        D,
        gradP0,
        gradpsi0,
        basin_points,
        basin_weights,
        pair_idx,
        trans_points,
        trans_weights,
        comm_grad_zero,
        trans_surface_states,
        path_disp_zero,
        path_weights,
        component_totals,
        basin_stoichiometry,
        self_test_temperature_K,
        PROJECTED_REFERENCE_VOLUME_M3,
        projectors,
    )
    assert neutral.sigma_S_m == 0.0

    charged = compute_projected_analytical_conductivity(
        U,
        D,
        gradP,
        gradpsi0,
        basin_points,
        basin_weights,
        pair_idx,
        trans_points,
        trans_weights,
        comm_grad_zero,
        trans_surface_states,
        path_disp_zero,
        path_weights,
        component_totals,
        basin_stoichiometry,
        self_test_temperature_K,
        PROJECTED_REFERENCE_VOLUME_M3,
        projectors,
    )
    expected = (
        F_C_PER_MOL
        * F_C_PER_MOL
        / (R_J_PER_MOL_K * self_test_temperature_K)
        * (2.0 / 3.0)
    )
    assert np.isclose(charged.sigma_S_m, expected)

    c = np.asarray([1.0, 1.0])
    K = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    self_test_displacement_m = 1.0e-9
    d = np.zeros((2, 2, 3))
    d[0, 1, 0] = self_test_displacement_m
    d[1, 0, 0] = -self_test_displacement_m
    M = np.zeros((2, 2, 3, 3))
    M[0, 1, 0, 0] = self_test_displacement_m * self_test_displacement_m
    M[1, 0, 0, 0] = self_test_displacement_m * self_test_displacement_m
    Dself = np.zeros((2, 3, 3))
    res = compute_projected_analytical_conductivity_from_primitives(
        c,
        K,
        d,
        M,
        Dself,
        np.zeros((0, 0)),
        np.zeros((0, 3)),
        self_test_temperature_K,
    )
    assert np.isclose(res.projected_diffusivity_tensor[0, 0], 0.0)

    A = np.asarray([[2.0]])
    h = np.asarray([[3.0, 0.0, 0.0]])
    C = compute_continuous_mori_correction(A, h)
    assert np.isclose(C[0, 0], 4.5)
