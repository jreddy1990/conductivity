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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from constants import (
    F,
    R,
    S_M_TO_MS_CM,
    T_REF_K,
)
from utils.strict_validation import strict_nonnegative_finite_array

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
GENERATOR_BALANCE_TOL = 1.0e-10
PSEUDOINVERSE_RELATIVE_TOL = 1.0e-12
MEMORY_NULLSPACE_RELATIVE_TOL = 1.0e-8
POISSON_SOLVABILITY_ABS_TOL = 1.0e-18  # Numerical zero for c-weighted drift in one disconnected generator component.
POISSON_SOLVABILITY_EPSILON_FACTOR = (
    100.0  # Floating-point guard factor used by the Poisson component solvability test.
)
PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL = 1.0e-10
FINITE_PROCESS_NONZERO_EPSILON_FACTOR = 100.0
BASIS_REFINEMENT_RESIDUAL_RELATIVE_TOL = 1.0e-8
BASIS_REFINEMENT_CONDUCTIVITY_RELATIVE_TOL = 1.0e-8
DIAGNOSTIC_TOP_RECORD_COUNT = (
    5  # Limit failure payload size while keeping dominant contributors visible.
)
LOG_FLOAT_MAX = np.log(np.finfo(float).max)
LOG_FLOAT_TINY = np.log(np.finfo(float).tiny)

Array = np.ndarray


class TransportOwnership(str, Enum):
    DC_SELF = "dc_self"
    BOUNDED_MEMORY = "bounded_memory"
    TRANSITION_DISPLACEMENT = "transition_displacement"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class StateTransportOwnershipBasis:
    transition_displacement_gradients: Array
    transition_edge_indices: Array
    bounded_memory_gradients: Array
    bounded_memory_mode_indices: Array
    diagnostic_gradients: Array
    diagnostic_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        transition_gradients = _immutable_2d_array(
            self.transition_displacement_gradients,
            "transition_displacement_gradients",
        )
        bounded_memory_gradients = _immutable_2d_array(
            self.bounded_memory_gradients,
            "bounded_memory_gradients",
        )
        diagnostic_gradients = _immutable_2d_array(
            self.diagnostic_gradients,
            "diagnostic_gradients",
        )
        coordinate_dimensions = {
            transition_gradients.shape[1],
            bounded_memory_gradients.shape[1],
            diagnostic_gradients.shape[1],
        }
        if len(coordinate_dimensions) != 1:
            raise ValueError("transport ownership gradients must share one width")
        transition_edge_indices = _immutable_index_vector(
            self.transition_edge_indices,
            "transition_edge_indices",
        )
        bounded_memory_mode_indices = _immutable_index_vector(
            self.bounded_memory_mode_indices,
            "bounded_memory_mode_indices",
        )
        if transition_edge_indices.size != transition_gradients.shape[0]:
            raise ValueError("TRANSITION_OWNER_SOURCE_CARDINALITY_FAILED")
        if bounded_memory_mode_indices.size != bounded_memory_gradients.shape[0]:
            raise ValueError("MEMORY_OWNER_SOURCE_CARDINALITY_FAILED")
        if len(self.diagnostic_source_ids) != diagnostic_gradients.shape[0]:
            raise ValueError("DIAGNOSTIC_OWNER_SOURCE_CARDINALITY_FAILED")
        if any(not source_id.strip() for source_id in self.diagnostic_source_ids):
            raise ValueError("diagnostic_source_ids must not contain empty values")
        object.__setattr__(
            self,
            "transition_displacement_gradients",
            transition_gradients,
        )
        object.__setattr__(self, "transition_edge_indices", transition_edge_indices)
        object.__setattr__(self, "bounded_memory_gradients", bounded_memory_gradients)
        object.__setattr__(
            self,
            "bounded_memory_mode_indices",
            bounded_memory_mode_indices,
        )
        object.__setattr__(self, "diagnostic_gradients", diagnostic_gradients)


@dataclass(frozen=True)
class TransportOwnershipTensorSet:
    state_index: int
    quadrature_index: int
    full_short_time_tensor_m2_s: Array
    dc_self_tensor_m2_s: Array
    transition_displacement_tensor_m2_s: Array
    bounded_memory_tensor_m2_s: Array
    diagnostic_tensor_m2_s: Array
    closure_residual_tensor_m2_s: Array
    coordinate_support_rank: int
    transition_rank: int
    bounded_memory_rank: int
    diagnostic_rank: int


@dataclass(frozen=True)
class StateTransportOwnershipQuadrature:
    point_tensors: tuple[TransportOwnershipTensorSet, ...]
    density_weighted_full_tensor_m2_s: Array
    density_weighted_dc_self_tensor_m2_s: Array
    density_weighted_transition_displacement_tensor_m2_s: Array
    density_weighted_bounded_memory_tensor_m2_s: Array
    density_weighted_diagnostic_tensor_m2_s: Array
    maximum_closure_residual_m2_s: float


def _immutable_2d_array(values: Array, label: str) -> Array:
    array = np.asarray(values, dtype=float).copy()
    if array.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    array.setflags(write=False)
    return array


def _immutable_index_vector(values: Array, label: str) -> Array:
    array = np.asarray(values, dtype=int).copy()
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError(f"{label} must be a nonnegative one-dimensional vector")
    array.setflags(write=False)
    return array

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
    "state_transport_ownership_bases",
    "transition_transport_ownership",
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
        basin_energy_references_J_mol: Array,
        state_memory_active_mask: Array,
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
        state_transport_ownership_bases: tuple[
            tuple[StateTransportOwnershipBasis, ...], ...
        ],
        transition_transport_ownership: tuple[TransportOwnership, ...],
        state_relative_displacement_fluctuations_m: tuple[Array, ...],
        state_relative_displacement_mobilities_m2_s: tuple[Array, ...],
        state_relative_center_charge_numbers: tuple[Array, ...],
        state_memory_value_matrix: Array,
        max_transition_displacement_m: float = DEFAULT_MAX_TRANSITION_DISPLACEMENT_M,
    ) -> None:
        self.potential_energy_J_mol = potential_energy_J_mol
        self.mobility_tensor_m2_s = mobility_tensor_m2_s
        self.charge_polarization_gradient = charge_polarization_gradient
        self.memory_coordinate_gradient = memory_coordinate_gradient
        self.basin_quadrature_points = basin_quadrature_points
        self.basin_quadrature_weights = basin_quadrature_weights
        self.basin_energy_references_J_mol = basin_energy_references_J_mol
        self.state_memory_active_mask = state_memory_active_mask
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
        self.state_transport_ownership_bases = state_transport_ownership_bases
        self.transition_transport_ownership = transition_transport_ownership
        self.state_relative_displacement_fluctuations_m = (
            state_relative_displacement_fluctuations_m
        )
        self.state_relative_displacement_mobilities_m2_s = (
            state_relative_displacement_mobilities_m2_s
        )
        self.state_relative_center_charge_numbers = state_relative_center_charge_numbers
        self.state_memory_value_matrix = state_memory_value_matrix
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
        state_memory_value_matrix: Array,
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
        self.state_memory_value_matrix = state_memory_value_matrix
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
        discrete_state_memory_matrix_A_Q: Array,
        discrete_state_current_coupling_matrix_h_Q: Array,
        state_transport_ownership_quadratures: tuple[
            StateTransportOwnershipQuadrature, ...
        ],
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
        self.discrete_state_memory_matrix_A_Q = discrete_state_memory_matrix_A_Q
        self.discrete_state_current_coupling_matrix_h_Q = (
            discrete_state_current_coupling_matrix_h_Q
        )
        self.state_transport_ownership_quadratures = (
            state_transport_ownership_quadratures
        )
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
            "discrete_state_memory_matrix_A_Q": self.discrete_state_memory_matrix_A_Q,
            "discrete_state_current_coupling_matrix_h_Q": self.discrete_state_current_coupling_matrix_h_Q,
            "state_transport_ownership_quadratures": self.state_transport_ownership_quadratures,
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


@dataclass(frozen=True)
class FiniteLifetimeCovarianceDiagnostics:
    state_index: int
    lifetime_rate_s_inv: float
    relative_covariance_trace_m2: float
    instantaneous_relative_mobility_trace_m2_s: float
    finite_relative_mobility_trace_m2_s: float
    center_covariance_min_eigenvalue_m2_s: float
    short_trace_m2_s: float
    dc_self_trace_m2_s: float
    bounded_memory_trace_m2_s: float
    transition_owned_trace_m2_s: float


@dataclass(frozen=True)
class StateDriftComponentAudit:
    component_id: int
    state_indices: Array
    c_transpose_b_mol_m2_s: Array
    c_transpose_b_norm_mol_m2_s: float


@dataclass(frozen=True)
class DirectPrimitiveAuditLedger:
    B_self_full_tensor_mol_m_s: Array
    B_self_tangent_tensor_mol_m_s: Array
    B_transition_tensor_mol_m_s: Array
    B_overlap_removed_tensor_mol_m_s: Array
    B_total_tensor_mol_m_s: Array
    C_Q_contribution_tensor_mol_m_s: Array
    state_drift_b_i_m_s: Array
    state_exit_rates_s_inv: Array
    state_drift_b_i_norms_m_s: Array
    state_drift_components: tuple[StateDriftComponentAudit, ...]


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
            basin_energy_references_J_mol=_basin_energy_references_J_mol(
                potential_energy_J_mol,
                basin_quadrature_points,
            ),
            state_memory_active_mask=np.ones(
                (
                    state_count,
                    int(
                        np.asarray(
                            memory_coordinate_gradient(basin_quadrature_points[0][0]),
                            dtype=float,
                        ).shape[0]
                    ),
                ),
                dtype=bool,
            ),
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
            state_transport_ownership_bases=(
                _bounded_memory_state_transport_ownership_bases(
                    basin_quadrature_points,
                    memory_coordinate_gradient,
                )
            ),
            transition_transport_ownership=tuple(
                TransportOwnership.TRANSITION_DISPLACEMENT
                for _transition_index in range(
                    np.asarray(transition_pair_indices, dtype=int).shape[0]
                )
            ),
            state_relative_displacement_fluctuations_m=tuple(
                np.empty((0, CARTESIAN), dtype=float) for _ in range(state_count)
            ),
            state_relative_displacement_mobilities_m2_s=tuple(
                np.empty((0, 0), dtype=float) for _ in range(state_count)
            ),
            state_relative_center_charge_numbers=tuple(
                np.empty(0, dtype=float) for _ in range(state_count)
            ),
            state_memory_value_matrix=np.zeros((state_count, 0), dtype=float),
        )
    )


def _compute_projected_analytical_conductivity_from_input(
    model_input: ProjectedGeneratorInput,
) -> ProjectedConductivityResult:
    validate_generator_input(model_input)
    log_partitions = compute_restricted_log_partition_values(
        model_input.potential_energy_J_mol,
        model_input.basin_quadrature_points,
        model_input.basin_quadrature_weights,
        model_input.basin_energy_references_J_mol,
        model_input.temperature_K,
    )
    density_result = compute_basin_density_weights(
        model_input.potential_energy_J_mol,
        model_input.basin_quadrature_points,
        model_input.basin_quadrature_weights,
        model_input.basin_energy_references_J_mol,
        log_partitions,
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
        len(log_partitions),
        model_input.transition_log_capacity_integrals,
        model_input.transition_uses_residence_rate_constants,
        model_input.transition_residence_rate_constants_s_inv,
        concentrations,
    )
    d, M = transition_moments_from_generator_input(model_input, len(log_partitions))
    (
        Dself_full,
        Dself_tangent,
        transition_owned_self_current,
        bounded_memory_owned_self_current,
        diagnostic_owned_self_current,
        state_transport_ownership_quadratures,
    ) = compute_state_transport_ownership_quadratures(
        mobility_tensor_m2_s=model_input.mobility_tensor_m2_s,
        charge_polarization_gradient=model_input.charge_polarization_gradient,
        self_current_coordinate_projectors=(
            model_input.self_current_coordinate_projectors
        ),
        basin_quadrature_points=model_input.basin_quadrature_points,
        basin_density_weights_mol_m3=density_weights,
        basin_concentrations_mol_m3=concentrations,
        state_transport_ownership_bases=(
            _state_ownership_bases_with_memory_mode_indices(
                model_input.state_transport_ownership_bases,
                np.empty(0, dtype=int),
            )
        ),
    )
    Dself_continuous = Dself_tangent + bounded_memory_owned_self_current
    Q_for_filter = compute_reversible_generator(K, concentrations)
    validate_reversible_generator(Q_for_filter, concentrations)
    finite_lifetime_diagnostics = ()
    direct_for_filter = compute_direct_diffusivity_tensor(
        concentrations, K, M, Dself_tangent
    )
    finite_correction_for_filter = compute_finite_state_memory_correction(
        concentrations,
        Q_for_filter,
        d,
    )
    raw_memory_matrix, raw_current_coupling = compute_mori_memory_matrices(
        model_input.mobility_tensor_m2_s,
        model_input.charge_polarization_gradient,
        model_input.memory_coordinate_gradient,
        model_input.basin_quadrature_points,
        density_weights,
        model_input.state_memory_active_mask,
    )
    bounded_memory_candidate_indices = _declared_bounded_memory_mode_indices(
        model_input.state_transport_ownership_bases
    )
    if np.any(bounded_memory_candidate_indices >= raw_memory_matrix.shape[0]):
        raise ValueError("MORI_OWNER_INDEX_OUT_OF_RANGE")
    bounded_memory_matrix = raw_memory_matrix[np.ix_(
        bounded_memory_candidate_indices, bounded_memory_candidate_indices
    )]
    bounded_memory_current_coupling = raw_current_coupling[
        bounded_memory_candidate_indices
    ]
    remaining_direct_tensor = _symmetrize(
        direct_for_filter - finite_correction_for_filter
    )
    conductivity_scale_S_m = max(
        abs(
            conductivity_from_projected_diffusivity(
                remaining_direct_tensor,
                model_input.temperature_K,
            )
        ),
        np.finfo(float).tiny,
    )
    refinement_result = refine_mori_basis_by_projected_residual(
        direct_diffusivity_tensor=remaining_direct_tensor,
        initial_mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        initial_mori_current_coupling_matrix_h=np.zeros((0, CARTESIAN), dtype=float),
        candidate_self_energies_A_gg=np.diag(bounded_memory_matrix),
        candidate_cross_energies_A_gPhi=np.zeros(
            (bounded_memory_matrix.shape[0], 0), dtype=float
        ),
        candidate_cross_energy_matrix=bounded_memory_matrix,
        candidate_current_couplings_h_g=bounded_memory_current_coupling,
        temperature_K=model_input.temperature_K,
        residual_score_tolerance=BASIS_REFINEMENT_RESIDUAL_RELATIVE_TOL
        * max(_maximum_abs_eigenvalue(remaining_direct_tensor), np.finfo(float).tiny),
        conductivity_change_tolerance_S_m=(
            BASIS_REFINEMENT_CONDUCTIVITY_RELATIVE_TOL * conductivity_scale_S_m
        ),
        max_added_coordinates=bounded_memory_matrix.shape[0],
        require_candidate_set_exhaustion=False,
    )
    for candidate_index_field in (
        "selected_candidate_indices",
        "rejected_null_energy_candidate_indices",
        "rejected_current_spanning_candidate_indices",
        "rejected_psd_candidate_indices",
    ):
        local_candidate_indices = np.asarray(
            refinement_result[candidate_index_field], dtype=int
        )
        refinement_result[candidate_index_field] = bounded_memory_candidate_indices[
            local_candidate_indices
        ]
    accepted_bounded_memory_mode_indices = np.asarray(
        refinement_result["selected_candidate_indices"], dtype=int
    )
    _validate_bounded_memory_owner_consumer_closure(
        accepted_bounded_memory_mode_indices,
        refinement_result["final_mori_memory_matrix_A"],
        refinement_result["final_mori_current_coupling_matrix_h"],
    )
    if accepted_bounded_memory_mode_indices.size:
        (
            Dself_full,
            Dself_tangent,
            transition_owned_self_current,
            bounded_memory_owned_self_current,
            diagnostic_owned_self_current,
            state_transport_ownership_quadratures,
        ) = compute_state_transport_ownership_quadratures(
            mobility_tensor_m2_s=model_input.mobility_tensor_m2_s,
            charge_polarization_gradient=model_input.charge_polarization_gradient,
            self_current_coordinate_projectors=(
                model_input.self_current_coordinate_projectors
            ),
            basin_quadrature_points=model_input.basin_quadrature_points,
            basin_density_weights_mol_m3=density_weights,
            basin_concentrations_mol_m3=concentrations,
            state_transport_ownership_bases=(
                _state_ownership_bases_with_memory_mode_indices(
                    model_input.state_transport_ownership_bases,
                    accepted_bounded_memory_mode_indices,
                )
            ),
        )
        Dself_continuous = Dself_tangent + bounded_memory_owned_self_current
        post_ownership_direct_tensor = compute_direct_diffusivity_tensor(
            concentrations,
            K,
            M,
            Dself_continuous,
        )
        post_ownership_mori_correction = compute_continuous_mori_correction(
            refinement_result["final_mori_memory_matrix_A"],
            refinement_result["final_mori_current_coupling_matrix_h"],
        )
        post_ownership_projected_tensor = _symmetrize(
            post_ownership_direct_tensor
            - finite_correction_for_filter
            - post_ownership_mori_correction
        )
        post_ownership_eigenvalues = np.linalg.eigvalsh(post_ownership_projected_tensor)
        post_ownership_scale = max(
            _maximum_abs_eigenvalue(post_ownership_direct_tensor),
            np.finfo(float).tiny,
        )
        if float(np.min(post_ownership_eigenvalues)) < (
            -PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL * post_ownership_scale
        ):
            refinement_result["rejected_psd_candidate_indices"] = np.unique(
                np.concatenate(
                    (
                        np.asarray(
                            refinement_result["rejected_psd_candidate_indices"],
                            dtype=int,
                        ),
                        accepted_bounded_memory_mode_indices,
                    )
                )
            )
            refinement_result["selected_candidate_indices"] = np.empty(0, dtype=int)
            refinement_result["final_mori_memory_matrix_A"] = np.zeros(
                (0, 0), dtype=float
            )
            refinement_result["final_mori_current_coupling_matrix_h"] = np.zeros(
                (0, CARTESIAN), dtype=float
            )
            (
                Dself_full,
                Dself_tangent,
                transition_owned_self_current,
                bounded_memory_owned_self_current,
                diagnostic_owned_self_current,
                state_transport_ownership_quadratures,
            ) = compute_state_transport_ownership_quadratures(
                mobility_tensor_m2_s=model_input.mobility_tensor_m2_s,
                charge_polarization_gradient=model_input.charge_polarization_gradient,
                self_current_coordinate_projectors=(
                    model_input.self_current_coordinate_projectors
                ),
                basin_quadrature_points=model_input.basin_quadrature_points,
                basin_density_weights_mol_m3=density_weights,
                basin_concentrations_mol_m3=concentrations,
                state_transport_ownership_bases=(
                    _state_ownership_bases_with_memory_mode_indices(
                        model_input.state_transport_ownership_bases,
                        np.empty(0, dtype=int),
                    )
                ),
            )
            Dself_continuous = Dself_tangent + bounded_memory_owned_self_current
    projector_ranks = tuple(
        int(np.linalg.matrix_rank(np.asarray(projector, dtype=float)))
        for projector in model_input.self_current_coordinate_projectors
    )
    continuous_memory_matrix = np.asarray(
        refinement_result["final_mori_memory_matrix_A"], dtype=float
    )
    continuous_current_coupling = np.asarray(
        refinement_result["final_mori_current_coupling_matrix_h"], dtype=float
    )
    discrete_state_memory_values = as_2d(
        model_input.state_memory_value_matrix,
        "state_memory_value_matrix",
    )
    if discrete_state_memory_values.shape[0] != concentrations.size:
        raise ValueError(
            "state_memory_value_matrix row count must equal the state count"
        )
    if discrete_state_memory_values.shape[1]:
        provisional_discrete_memory, provisional_discrete_coupling = (
            compute_discrete_state_mori_matrices(
                concentrations,
                compute_reversible_generator(K, concentrations),
                d,
                discrete_state_memory_values,
            )
        )
        coupling_norms = np.linalg.norm(provisional_discrete_coupling, axis=1)
        coupling_scale = max(float(np.max(coupling_norms)), np.finfo(float).tiny)
        dirichlet_scale = max(
            float(np.max(np.abs(np.diag(provisional_discrete_memory)))),
            np.finfo(float).tiny,
        )
        retained_discrete_modes = np.flatnonzero(
            (coupling_norms > np.finfo(float).eps * coupling_scale)
            & (
                np.diag(provisional_discrete_memory)
                > np.finfo(float).eps * dirichlet_scale
            )
        )
        discrete_state_memory_values = discrete_state_memory_values[
            :, retained_discrete_modes
        ]
    else:
        retained_discrete_modes = np.empty(0, dtype=int)
    discrete_memory_count = discrete_state_memory_values.shape[1]
    continuous_memory_count = continuous_memory_matrix.shape[0]
    combined_memory_matrix = np.zeros(
        (
            continuous_memory_count + discrete_memory_count,
            continuous_memory_count + discrete_memory_count,
        ),
        dtype=float,
    )
    combined_memory_matrix[:continuous_memory_count, :continuous_memory_count] = (
        continuous_memory_matrix
    )
    combined_current_coupling = np.vstack(
        (
            continuous_current_coupling,
            np.zeros((discrete_memory_count, CARTESIAN), dtype=float),
        )
    )
    combined_state_memory_values = np.hstack(
        (
            np.asarray(
                model_input.state_memory_active_mask[
                    :, accepted_bounded_memory_mode_indices
                ],
                dtype=float,
            ),
            discrete_state_memory_values,
        )
    )
    conductivity_result = (
        _compute_projected_analytical_conductivity_from_primitive_input(
            ProjectedPrimitiveInput(
                state_concentrations_mol_m3=concentrations,
                symmetric_capacity_fluxes_K_ij_mol_m3_s=K,
                transition_first_moments_d_ij_m=d,
                transition_second_moments_M_ij_m2=M,
                self_current_tensors_D_self_i_m2_s=Dself_continuous,
                mori_memory_matrix_A=combined_memory_matrix,
                mori_current_coupling_matrix_h=combined_current_coupling,
                state_memory_value_matrix=combined_state_memory_values,
                temperature_K=model_input.temperature_K,
                volume_m3=model_input.volume_m3,
                max_transition_displacement_m=model_input.max_transition_displacement_m,
            ),
            self_current_projector_ranks=projector_ranks,
            full_self_current_tensors_D_self_i_m2_s=Dself_full,
        )
    )
    conductivity_result.state_transport_ownership_quadratures = (
        state_transport_ownership_quadratures
    )
    conductivity_result.effect_attribution.update(
        {
            "state_family_memory_retained_indices": retained_discrete_modes,
            "finite_lifetime_relative_covariance": finite_lifetime_diagnostics,
            "transport_ownership_state_tensors": tuple(
                {
                    "state_index": state_index,
                    "D_Q_short": Dself_full[state_index],
                    "D_Q_dc_self": Dself_tangent[state_index],
                    "D_Q_transition_owned": transition_owned_self_current[state_index],
                    "D_Q_bounded_memory": bounded_memory_owned_self_current[
                        state_index
                    ],
                    "D_Q_diagnostic": diagnostic_owned_self_current[state_index],
                    "D_Q_unowned": _symmetrize(
                        Dself_full[state_index]
                        - Dself_tangent[state_index]
                        - transition_owned_self_current[state_index]
                        - bounded_memory_owned_self_current[state_index]
                        - diagnostic_owned_self_current[state_index]
                    ),
                }
                for state_index in range(len(concentrations))
            ),
            "mori_filter_accepted_candidate_indices": refinement_result[
                "selected_candidate_indices"
            ],
            "mori_declared_bounded_memory_mode_indices": (
                bounded_memory_candidate_indices.copy()
            ),
            "mori_declared_bounded_memory_A_diagonal": np.diag(
                bounded_memory_matrix
            ).copy(),
            "mori_declared_bounded_memory_h_norms": np.linalg.norm(
                bounded_memory_current_coupling,
                axis=1,
            ),
            "mori_filter_discarded_candidate_indices": refinement_result[
                "discarded_candidate_indices"
            ],
            "mori_filter_rejected_candidate_indices": np.concatenate(
                (
                    refinement_result["rejected_null_energy_candidate_indices"],
                    refinement_result["rejected_current_spanning_candidate_indices"],
                    refinement_result["rejected_psd_candidate_indices"],
                )
            ),
        }
    )
    conductivity_result.effect_attribution.update(
        basis_refinement_as_effect_attribution(refinement_result)
    )
    conductivity_result.effect_attribution.update(
        primitive_prediction_readiness_as_effect_attribution(
            conductivity_result.effect_attribution,
        )
    )
    return conductivity_result


def _empty_state_transport_ownership_bases(
    basin_quadrature_points: Sequence[Array],
) -> tuple[tuple[StateTransportOwnershipBasis, ...], ...]:
    points_by_state = tuple(
        as_2d(state_points, "basin_quadrature_points[]")
        for state_points in basin_quadrature_points
    )
    return tuple(
        tuple(
            StateTransportOwnershipBasis(
                transition_displacement_gradients=np.empty(
                    (0, points.shape[1]),
                    dtype=float,
                ),
                transition_edge_indices=np.empty(0, dtype=int),
                bounded_memory_gradients=np.empty(
                    (0, points.shape[1]),
                    dtype=float,
                ),
                bounded_memory_mode_indices=np.empty(0, dtype=int),
                diagnostic_gradients=np.empty(
                    (0, points.shape[1]),
                    dtype=float,
                ),
                diagnostic_source_ids=(),
            )
            for _point in points
        )
        for points in points_by_state
    )


def _bounded_memory_state_transport_ownership_bases(
    basin_quadrature_points: Sequence[Array],
    memory_coordinate_gradient: Callable[[Array], Array],
) -> tuple[tuple[StateTransportOwnershipBasis, ...], ...]:
    points_by_state = tuple(
        as_2d(state_points, "basin_quadrature_points[]")
        for state_points in basin_quadrature_points
    )
    return tuple(
        tuple(
            _bounded_memory_ownership_basis_for_point(
                point,
                points.shape[1],
                memory_coordinate_gradient,
            )
            for point in points
        )
        for points in points_by_state
    )


def _bounded_memory_ownership_basis_for_point(
    point: Array,
    coordinate_dimension: int,
    memory_coordinate_gradient: Callable[[Array], Array],
) -> StateTransportOwnershipBasis:
    bounded_gradients = as_2d(
        memory_coordinate_gradient(point),
        "memory_coordinate_gradient(point)",
    )
    if bounded_gradients.shape[1] != coordinate_dimension:
        raise ValueError(
            "memory coordinate gradient width must match coordinate dimension"
        )
    return StateTransportOwnershipBasis(
        transition_displacement_gradients=np.empty(
            (0, coordinate_dimension),
            dtype=float,
        ),
        transition_edge_indices=np.empty(0, dtype=int),
        bounded_memory_gradients=bounded_gradients,
        bounded_memory_mode_indices=np.arange(
            bounded_gradients.shape[0],
            dtype=int,
        ),
        diagnostic_gradients=np.empty(
            (0, coordinate_dimension),
            dtype=float,
        ),
        diagnostic_source_ids=(),
    )


def _declared_bounded_memory_mode_indices(
    state_transport_ownership_bases: Sequence[
        Sequence[StateTransportOwnershipBasis]
    ],
) -> Array:
    declared_indices = {
        int(mode_index)
        for state_bases in state_transport_ownership_bases
        for ownership_basis in state_bases
        for mode_index in ownership_basis.bounded_memory_mode_indices
    }
    return np.asarray(sorted(declared_indices), dtype=int)


def _validate_bounded_memory_owner_consumer_closure(
    admitted_owner_mode_indices: Array,
    mori_memory_matrix_A: Array,
    mori_current_coupling_matrix_h: Array,
) -> None:
    owner_indices = np.asarray(admitted_owner_mode_indices, dtype=int)
    if owner_indices.ndim != 1 or np.unique(owner_indices).size != owner_indices.size:
        raise ValueError("MORI_OWNER_CONSUMER_CLOSURE_DUPLICATE_OWNER")
    memory_matrix = as_matrix_shape(
        mori_memory_matrix_A,
        (owner_indices.size, owner_indices.size),
        "mori_memory_matrix_A",
    )
    current_coupling = as_matrix_shape(
        mori_current_coupling_matrix_h,
        (owner_indices.size, CARTESIAN),
        "mori_current_coupling_matrix_h",
    )
    if not np.all(np.isfinite(memory_matrix)) or not np.all(
        np.isfinite(current_coupling)
    ):
        raise ValueError("MORI_OWNER_CONSUMER_CLOSURE_NONFINITE_CONSUMER")


def _state_ownership_bases_with_memory_mode_indices(
    state_transport_ownership_bases: Sequence[
        Sequence[StateTransportOwnershipBasis]
    ],
    selected_memory_mode_indices: Array,
) -> tuple[tuple[StateTransportOwnershipBasis, ...], ...]:
    selected_indices = np.asarray(selected_memory_mode_indices, dtype=int)
    if selected_indices.ndim != 1 or np.any(selected_indices < 0):
        raise ValueError("selected_memory_mode_indices must be nonnegative and 1D")
    selected_index_set = set(int(index) for index in selected_indices)
    return tuple(
        tuple(
            _ownership_basis_with_selected_memory_modes(
                ownership_basis,
                selected_index_set,
            )
            for ownership_basis in state_bases
        )
        for state_bases in state_transport_ownership_bases
    )


def _ownership_basis_with_selected_memory_modes(
    ownership_basis: StateTransportOwnershipBasis,
    selected_memory_mode_indices: set[int],
) -> StateTransportOwnershipBasis:
    retained_row_indices = np.asarray(
        [
            row_index
            for row_index, mode_index in enumerate(
                ownership_basis.bounded_memory_mode_indices
            )
            if int(mode_index) in selected_memory_mode_indices
        ],
        dtype=int,
    )
    coordinate_dimension = ownership_basis.bounded_memory_gradients.shape[1]
    retained_gradients = np.empty((0, coordinate_dimension), dtype=float)
    retained_mode_indices = np.empty(0, dtype=int)
    if retained_row_indices.size:
        retained_gradients = ownership_basis.bounded_memory_gradients[
            retained_row_indices
        ]
        retained_mode_indices = ownership_basis.bounded_memory_mode_indices[
            retained_row_indices
        ]
    return StateTransportOwnershipBasis(
        transition_displacement_gradients=(
            ownership_basis.transition_displacement_gradients
        ),
        transition_edge_indices=ownership_basis.transition_edge_indices,
        bounded_memory_gradients=retained_gradients,
        bounded_memory_mode_indices=retained_mode_indices,
        diagnostic_gradients=ownership_basis.diagnostic_gradients,
        diagnostic_source_ids=ownership_basis.diagnostic_source_ids,
    )


def apply_finite_lifetime_relative_covariance(
    self_current_tensors_D_self_i_m2_s: Array,
    symmetric_capacity_fluxes_K_ij_mol_m3_s: Array,
    reversible_generator_Q_ij_s_inv: Array,
    transition_second_moments_M_ij_m2: Array,
    state_concentrations_mol_m3: Array,
    state_relative_displacement_fluctuations_m: Sequence[Array],
    state_relative_displacement_mobilities_m2_s: Sequence[Array],
    state_relative_center_charge_numbers: Sequence[Array],
    transition_displacement_edge_mask: Array,
    temperature_K: float,
) -> tuple[Array, Array, tuple[FiniteLifetimeCovarianceDiagnostics, ...]]:
    self_currents = np.asarray(self_current_tensors_D_self_i_m2_s, dtype=float).copy()
    generator = as_square_any(
        reversible_generator_Q_ij_s_inv, "reversible_generator_Q_ij_s_inv"
    )
    state_count = generator.shape[0]
    capacity_fluxes = as_matrix_shape(
        symmetric_capacity_fluxes_K_ij_mol_m3_s,
        (state_count, state_count),
        "symmetric_capacity_fluxes_K_ij_mol_m3_s",
    )
    concentrations = positive_vector(
        state_concentrations_mol_m3, "state_concentrations_mol_m3"
    )
    second_moments = np.asarray(transition_second_moments_M_ij_m2, dtype=float).copy()
    displacement_edge_mask = np.asarray(transition_displacement_edge_mask, dtype=bool)
    if self_currents.shape != (state_count, CARTESIAN, CARTESIAN):
        raise ValueError("self-current tensor shape must match finite-lifetime states")
    if concentrations.shape != (state_count,):
        raise ValueError("state concentration count must match finite-lifetime states")
    if second_moments.shape != (state_count, state_count, CARTESIAN, CARTESIAN):
        raise ValueError("transition second moments must have shape (n,n,3,3)")
    if displacement_edge_mask.shape != (state_count, state_count):
        raise ValueError("transition displacement edge mask must have shape (n,n)")
    if not np.array_equal(displacement_edge_mask, displacement_edge_mask.T):
        raise ValueError("transition displacement edge mask must be symmetric")
    if not np.allclose(capacity_fluxes, capacity_fluxes.T):
        raise ValueError("finite-lifetime capacity fluxes must be symmetric")
    if np.any(capacity_fluxes < 0.0):
        raise ValueError("finite-lifetime capacity fluxes must be nonnegative")
    if not np.allclose(second_moments, np.swapaxes(second_moments, 0, 1)):
        raise ValueError("finite-lifetime transition moments require M_ji=M_ij")
    if (
        not len(state_relative_displacement_fluctuations_m)
        == len(state_relative_displacement_mobilities_m2_s)
        == len(state_relative_center_charge_numbers)
        == state_count
    ):
        raise ValueError("finite-lifetime descriptor count must match state count")
    diagnostics = []
    for state_index in range(state_count):
        fluctuations = np.asarray(
            state_relative_displacement_fluctuations_m[state_index], dtype=float
        )
        if fluctuations.shape == (0, CARTESIAN):
            continue
        if fluctuations.ndim != 2 or fluctuations.shape[1] != CARTESIAN:
            raise ValueError(
                "relative displacement fluctuations must have three columns"
            )
        instantaneous_relative_mobility = as_matrix_shape(
            state_relative_displacement_mobilities_m2_s[state_index],
            (CARTESIAN, CARTESIAN),
            "state_relative_displacement_mobility",
        )
        validate_psd(
            instantaneous_relative_mobility,
            "state_relative_displacement_mobility",
            allow_zero=True,
        )
        relative_covariance = _symmetrize(fluctuations.T @ fluctuations)
        validate_psd(
            relative_covariance, "relative_displacement_covariance", allow_zero=True
        )
        positive_float(temperature_K, "temperature_K")
        lifetime_rate = -float(generator[state_index, state_index])
        finite_relative_mobility = _finite_lifetime_relative_mobility(
            relative_covariance,
            instantaneous_relative_mobility,
            lifetime_rate,
        )
        validate_psd(
            finite_relative_mobility, "finite_relative_mobility", allow_zero=True
        )
        charges = as_1d(
            state_relative_center_charge_numbers[state_index],
            "state_relative_center_charge_numbers",
        )
        if charges.shape != (2,):
            raise ValueError("relative center charge vector must contain two centers")
        if not np.isclose(float(np.sum(charges)), 0.0):
            raise ValueError("finite relative covariance requires a neutral bound pair")
        self_currents[state_index] = _project_psd_numerical_roundoff(
            _symmetrize(self_currents[state_index] - instantaneous_relative_mobility),
            f"finite_lifetime_dc_self[{state_index}]",
        )
        positive_outgoing_indices = np.flatnonzero(
            (capacity_fluxes[state_index] > 0.0) & displacement_edge_mask[state_index]
        )
        transition_owned_trace = float(np.trace(finite_relative_mobility))
        short_trace = float(np.trace(instantaneous_relative_mobility))
        bounded_memory_trace = float(
            np.trace(instantaneous_relative_mobility - finite_relative_mobility)
        )
        ownership_scale = max(abs(short_trace), np.finfo(float).tiny)
        if not np.isclose(
            short_trace,
            bounded_memory_trace + transition_owned_trace,
            rtol=np.sqrt(np.finfo(float).eps),
            atol=np.finfo(float).eps * ownership_scale,
        ):
            raise ValueError(
                f"bound-state transport ownership does not reconstruct state {state_index}"
            )
        if transition_owned_trace > _scale_aware_nonzero_threshold(
            finite_relative_mobility
        ):
            if positive_outgoing_indices.size == 0:
                raise ValueError(
                    "bound state with nonzero finite relative mobility requires a positive-K outgoing edge"
                )
            outgoing_capacity = float(
                np.sum(capacity_fluxes[state_index, positive_outgoing_indices])
            )
            state_transition_moment = (
                concentrations[state_index]
                * finite_relative_mobility
                / outgoing_capacity
            )
            for outgoing_index in positive_outgoing_indices:
                second_moments[state_index, outgoing_index] = _symmetrize(
                    second_moments[state_index, outgoing_index]
                    + state_transition_moment
                )
                second_moments[state_index, outgoing_index] = (
                    _project_psd_numerical_roundoff(
                        second_moments[state_index, outgoing_index],
                        f"finite_M[{state_index},{outgoing_index}]",
                    )
                )
                second_moments[outgoing_index, state_index] = second_moments[
                    state_index, outgoing_index
                ]
                validate_psd(
                    second_moments[state_index, outgoing_index],
                    f"finite_M[{state_index},{outgoing_index}]",
                    allow_zero=True,
                )
        diagnostics.append(
            FiniteLifetimeCovarianceDiagnostics(
                state_index=state_index,
                lifetime_rate_s_inv=lifetime_rate,
                relative_covariance_trace_m2=float(np.trace(relative_covariance)),
                instantaneous_relative_mobility_trace_m2_s=float(
                    np.trace(instantaneous_relative_mobility)
                ),
                finite_relative_mobility_trace_m2_s=transition_owned_trace,
                center_covariance_min_eigenvalue_m2_s=float(
                    np.min(np.linalg.eigvalsh(self_currents[state_index]))
                ),
                short_trace_m2_s=short_trace,
                dc_self_trace_m2_s=float(np.trace(self_currents[state_index])),
                bounded_memory_trace_m2_s=bounded_memory_trace,
                transition_owned_trace_m2_s=transition_owned_trace,
            )
        )
    return self_currents, second_moments, tuple(diagnostics)


def _finite_lifetime_relative_mobility(
    relative_covariance_m2: Array,
    instantaneous_relative_mobility_m2_s: Array,
    lifetime_rate_s_inv: float,
) -> Array:
    covariance = _symmetrize(np.asarray(relative_covariance_m2, dtype=float))
    mobility = _symmetrize(
        np.asarray(instantaneous_relative_mobility_m2_s, dtype=float)
    )
    validate_psd(covariance, "relative_displacement_covariance", allow_zero=True)
    validate_psd(mobility, "state_relative_displacement_mobility", allow_zero=True)
    if lifetime_rate_s_inv < 0.0:
        raise ValueError("finite-lifetime exit rate must be nonnegative")
    if lifetime_rate_s_inv == 0.0:
        return np.zeros((CARTESIAN, CARTESIAN), dtype=float)
    covariance_square_root = symmetric_psd_square_root(
        covariance,
        "relative_displacement_covariance",
    )
    covariance_inverse_square_root = symmetric_psd_inverse_square_root(
        covariance,
        "relative_displacement_covariance",
    )
    mobility_square_root = symmetric_psd_square_root(
        mobility,
        "state_relative_displacement_mobility",
    )
    whitened_mobility_factor = covariance_inverse_square_root @ mobility_square_root
    whitened_relaxation = _symmetrize(
        whitened_mobility_factor @ whitened_mobility_factor.T
    )
    validate_psd(whitened_relaxation, "whitened_relative_relaxation", allow_zero=True)
    relaxation_rates_s_inv, relaxation_modes = np.linalg.eigh(whitened_relaxation)
    finite_response_rates_s_inv = (
        lifetime_rate_s_inv
        * relaxation_rates_s_inv
        / (lifetime_rate_s_inv + relaxation_rates_s_inv)
    )
    finite_mobility_factor = (
        covariance_square_root
        @ relaxation_modes
        @ np.diag(np.sqrt(finite_response_rates_s_inv))
    )
    finite_mobility = _symmetrize(finite_mobility_factor @ finite_mobility_factor.T)
    return _project_psd_numerical_roundoff(
        finite_mobility,
        "finite_relative_mobility",
    )


def _project_psd_numerical_roundoff(matrix: Array, name: str) -> Array:
    symmetric_matrix = _symmetrize(np.asarray(matrix, dtype=float))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    maximum_abs_eigenvalue = float(np.max(np.abs(eigenvalues)))
    if maximum_abs_eigenvalue == 0.0:
        return symmetric_matrix
    numerical_tolerance = np.sqrt(np.finfo(float).eps) * maximum_abs_eigenvalue
    if float(np.min(eigenvalues)) < -numerical_tolerance:
        raise ValueError(f"{name} has a physical negative eigenvalue")
    nonnegative_eigenvalues = np.where(eigenvalues > 0.0, eigenvalues, 0.0)
    return _symmetrize(eigenvectors @ np.diag(nonnegative_eigenvalues) @ eigenvectors.T)


def symmetric_psd_square_root(matrix: Array, name: str) -> Array:
    symmetric_matrix = _symmetrize(np.asarray(matrix, dtype=float))
    validate_psd(symmetric_matrix, name, allow_zero=True)
    left_vectors, singular_values, _right_vectors = np.linalg.svd(
        symmetric_matrix,
        hermitian=True,
    )
    return _symmetrize(
        (left_vectors * np.sqrt(singular_values)[None, :]) @ left_vectors.T
    )


def symmetric_psd_inverse_square_root(matrix: Array, name: str) -> Array:
    square_root = symmetric_psd_square_root(matrix, name)
    return np.linalg.pinv(square_root, hermitian=True)


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
            basin_energy_references_J_mol=_basin_energy_references_J_mol(
                model_input.potential_energy_J_mol,
                model_input.basin_quadrature_points,
            ),
            state_memory_active_mask=np.ones(
                (
                    model_input.basin_stoichiometry.shape[0],
                    int(np.asarray(gradpsi(model_input.basin_quadrature_points[0][0])).shape[0]),
                ),
                dtype=bool,
            ),
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
            state_transport_ownership_bases=(
                _bounded_memory_state_transport_ownership_bases(
                    model_input.basin_quadrature_points,
                    gradpsi,
                )
            ),
            transition_transport_ownership=tuple(
                TransportOwnership.TRANSITION_DISPLACEMENT
                for _transition_index in range(
                    np.asarray(model_input.transition_pair_indices, dtype=int).shape[0]
                )
            ),
            state_relative_displacement_fluctuations_m=tuple(
                np.empty((0, CARTESIAN), dtype=float)
                for _ in model_input.basin_quadrature_points
            ),
            state_relative_displacement_mobilities_m2_s=tuple(
                np.empty((0, 0), dtype=float)
                for _ in model_input.basin_quadrature_points
            ),
            state_relative_center_charge_numbers=tuple(
                np.empty(0, dtype=float) for _ in model_input.basin_quadrature_points
            ),
            state_memory_value_matrix=model_input.state_memory_value_matrix,
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
    state_memory_value_matrix: Array,
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
            state_memory_value_matrix=state_memory_value_matrix,
            temperature_K=temperature_K,
            volume_m3=volume_m3,
        ),
        self_current_projector_ranks=(),
        full_self_current_tensors_D_self_i_m2_s=self_current_tensors_D_self_i_m2_s,
    )


def _compute_projected_analytical_conductivity_from_primitive_input(
    primitive_input: ProjectedPrimitiveInput,
    self_current_projector_ranks: tuple[int, ...],
    full_self_current_tensors_D_self_i_m2_s: Array,
) -> ProjectedConductivityResult:
    c, K, d, M, Dself, A_D, h_D, state_memory_values = validate_primitive_input(
        primitive_input
    )
    Q = compute_reversible_generator(K, c)
    validate_reversible_generator(Q, c)
    direct = compute_direct_diffusivity_tensor(c, K, M, Dself)
    finite_process_diagnostics = compute_finite_process_readout_diagnostics(
        K,
        d,
        M,
        primitive_input.max_transition_displacement_m,
    )
    full_finite_corr = compute_finite_state_memory_correction(c, Q, d)
    A_Q, h_Q = compute_discrete_state_mori_matrices(
        c,
        Q,
        d,
        state_memory_values,
    )
    A = _symmetrize(A_D + A_Q)
    h = h_D + h_Q
    mori_corr = compute_continuous_mori_correction(A, h)
    discrete_state_mori_corr = compute_continuous_mori_correction(A_Q, h_Q)
    finite_corr = project_diffusivity_tensor_to_psd_roundoff(
        _symmetrize(full_finite_corr - discrete_state_mori_corr)
    )
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
        direct_primitive_audit_as_effect_attribution(
            compute_direct_primitive_audit_ledger(
                c,
                K,
                Q,
                d,
                M,
                full_self_current_tensors_D_self_i_m2_s,
                Dself,
                finite_corr,
            )
        )
    )
    attribution.update(
        finite_process_readout_diagnostics_as_effect_attribution(
            finite_process_diagnostics
        )
    )
    attribution.update(
        primitive_prediction_readiness_as_effect_attribution(attribution)
    )
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
        discrete_state_memory_matrix_A_Q=A_Q,
        discrete_state_current_coupling_matrix_h_Q=h_Q,
        state_transport_ownership_quadratures=(),
        effect_attribution=attribution,
    )


def _basin_energy_references_J_mol(
    potential_energy_J_mol: Callable[[Array], float],
    basin_quadrature_points: Sequence[Array],
) -> Array:
    references = np.asarray(
        [
            min(float(potential_energy_J_mol(point)) for point in as_2d(points, "basin"))
            for points in basin_quadrature_points
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(references)):
        raise ValueError("basin energy references must be finite")
    return references


def compute_restricted_log_partition_values(
    potential_energy_J_mol: Callable[[Array], float],
    basin_quadrature_points: Sequence[Array],
    basin_quadrature_weights: Sequence[Array],
    basin_energy_references_J_mol: Array,
    temperature_K: float,
) -> Array:
    beta_mol = 1.0 / (R_J_PER_MOL_K * positive_float(temperature_K, "temperature_K"))
    log_values = []
    energy_references = as_1d(
        basin_energy_references_J_mol,
        "basin_energy_references_J_mol",
    )
    if energy_references.size != len(basin_quadrature_points):
        raise ValueError("basin energy reference count must equal basin count")
    for points, weights, energy_reference_J_mol in zip(
        basin_quadrature_points,
        basin_quadrature_weights,
        energy_references,
        strict=True,
    ):
        pts = as_2d(points, "basin_quadrature_points[]")
        w = as_1d(weights, "basin_quadrature_weights[]")
        if pts.shape[0] != w.size:
            raise ValueError("basin quadrature point/weight count mismatch")
        basin_log_value = -np.inf
        for point, weight in zip(pts, w):
            positive_weight = positive_float(float(weight), "basin_quadrature_weight")
            relative_energy_J_mol = (
                float(potential_energy_J_mol(point))
                - float(energy_reference_J_mol)
            )
            log_term = np.log(positive_weight) - beta_mol * relative_energy_J_mol
            basin_log_value = np.logaddexp(basin_log_value, log_term)
        log_values.append(basin_log_value)
    log_value_array = np.asarray(log_values, dtype=float)
    if not np.all(np.isfinite(log_value_array)):
        raise ValueError("restricted partition log-values must be finite")
    return log_value_array


def solve_basin_chemical_potentials(
    total_component_concentrations_mol_m3: Array,
    basin_stoichiometry: Array,
    restricted_log_partition_values: Array,
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
    log_partitions = as_1d(
        restricted_log_partition_values,
        "restricted_log_partition_values",
    )
    if log_partitions.size != basin_count:
        raise ValueError(
            "restricted_log_partition_values length must match basin count"
        )

    def log_basin_concentrations(chemical_potentials: Array) -> Array:
        log_concentrations = (
            np.log(STANDARD_CONCENTRATION_MOL_M3)
            + log_partitions
            + stoichiometry @ chemical_potentials
        )
        if not np.all(np.isfinite(log_concentrations)):
            raise ValueError("chemical potential trial produced nonfinite log concentrations")
        return log_concentrations

    def component_log_totals(log_concentrations: Array) -> Array:
        return np.asarray(
            [
                _log_weighted_sum_exp(
                    log_concentrations,
                    stoichiometry[:, component_index],
                )
                for component_index in range(component_count)
            ],
            dtype=float,
        )

    initial_chemical_potentials = np.zeros(component_count, dtype=float)
    for _iteration_index in range(CHEMICAL_POTENTIAL_MAX_ITERATIONS):
        for component_index in range(component_count):
            current_log_concentrations = log_basin_concentrations(
                initial_chemical_potentials
            )
            predicted_component_log_total = _log_weighted_sum_exp(
                current_log_concentrations,
                stoichiometry[:, component_index],
            )
            maximum_component_stoichiometry = positive_float(
                float(np.max(stoichiometry[:, component_index])),
                "maximum_component_stoichiometry",
            )
            initial_chemical_potentials[component_index] += (
                (
                    np.log(component_totals[component_index])
                    - predicted_component_log_total
                )
                / maximum_component_stoichiometry
            )
        current_log_concentrations = log_basin_concentrations(
            initial_chemical_potentials
        )
        initialization_residual = float(
            np.max(
                np.abs(
                    component_log_totals(current_log_concentrations)
                    - np.log(component_totals)
                )
            )
        )
        if initialization_residual < CHEMICAL_POTENTIAL_MASS_TOL:
            break

    def normalized_component_residual(chemical_potentials: Array) -> Array:
        return component_log_totals(log_basin_concentrations(chemical_potentials)) - np.log(
            component_totals
        )

    def normalized_component_jacobian(chemical_potentials: Array) -> Array:
        log_concentrations = log_basin_concentrations(chemical_potentials)
        component_logs = component_log_totals(log_concentrations)
        jacobian = np.zeros((component_count, component_count), dtype=float)
        for component_index in range(component_count):
            component_stoichiometry = stoichiometry[:, component_index]
            active_mask = component_stoichiometry > 0.0
            normalized_component_weights = np.zeros(basin_count, dtype=float)
            normalized_component_weights[active_mask] = np.exp(
                log_concentrations[active_mask]
                + np.log(component_stoichiometry[active_mask])
                - component_logs[component_index]
            )
            jacobian[component_index] = (
                normalized_component_weights @ stoichiometry
            )
        return jacobian

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
        log_partitions,
    )
    residual = stoichiometry.T @ basin_concentrations - component_totals
    normalized_residual = _normalized_mass_residual(residual, component_totals)
    if normalized_residual >= CHEMICAL_POTENTIAL_MASS_TOL:
        raise ValueError(
            "chemical potential mass-balance solve did not converge; "
            f"normalized_residual={normalized_residual:.9g}; "
            f"target_totals={component_totals.tolist()}; "
            f"predicted_totals={(stoichiometry.T @ basin_concentrations).tolist()}; "
            f"residual={residual.tolist()}; "
            f"solver_status={least_squares_result.status}; "
            f"solver_message={least_squares_result.message}"
        )
    return {
        "chemical_potentials": chemical_potentials,
        "basin_concentrations_mol_m3": basin_concentrations,
        "residual_mol_m3": residual,
        "normalized_residual": np.asarray([normalized_residual], dtype=float),
        "iterations": np.asarray([least_squares_result.nfev], dtype=float),
    }


def _log_weighted_sum_exp(log_values: Array, weights: Array) -> float:
    values = as_1d(log_values, "log_values")
    positive_weights = as_1d(weights, "weights")
    if values.size != positive_weights.size:
        raise ValueError("log_values and weights must have equal length")
    if np.any(positive_weights < 0.0):
        raise ValueError("weights must be nonnegative")
    active_mask = positive_weights > 0.0
    if not np.any(active_mask):
        raise ValueError("weighted log sum requires a positive weight")
    active_terms = values[active_mask] + np.log(positive_weights[active_mask])
    maximum_term = float(np.max(active_terms))
    return maximum_term + float(np.log(np.sum(np.exp(active_terms - maximum_term))))


def _mass_constraint_projection_matrix(
    basin_stoichiometry: Array,
    component_denominators_mol_m3: Array,
) -> Array:
    constraint_matrix = np.asarray(basin_stoichiometry, dtype=float).T
    component_denominators = positive_vector(
        component_denominators_mol_m3,
        "component_denominators_mol_m3",
    )
    if component_denominators.size != constraint_matrix.shape[0]:
        raise ValueError(
            "component_denominators_mol_m3 length must match conserved components"
        )
    normalized_constraint_matrix = (
        constraint_matrix / component_denominators[:, np.newaxis]
    )
    left_singular_vectors, singular_values, _right_singular_vectors = np.linalg.svd(
        normalized_constraint_matrix,
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
    restricted_log_partition_values: Array,
) -> Array:
    solve_result = solve_basin_chemical_potentials(
        total_component_concentrations_mol_m3,
        basin_stoichiometry,
        restricted_log_partition_values,
    )
    return np.asarray(solve_result["basin_concentrations_mol_m3"], dtype=float)


def compute_basin_density_weights(
    potential_energy_J_mol: Callable[[Array], float],
    basin_quadrature_points: Sequence[Array],
    basin_quadrature_weights: Sequence[Array],
    basin_energy_references_J_mol: Array,
    restricted_log_partition_values: Array,
    total_component_concentrations_mol_m3: Array,
    basin_stoichiometry: Array,
    temperature_K: float,
) -> dict[str, Array | tuple[Array, ...]]:
    solve_result = solve_basin_chemical_potentials(
        total_component_concentrations_mol_m3,
        basin_stoichiometry,
        restricted_log_partition_values,
    )
    chemical_potentials = np.asarray(solve_result["chemical_potentials"], dtype=float)
    beta_mol = 1.0 / (R_J_PER_MOL_K * positive_float(temperature_K, "temperature_K"))
    solved_concentrations = np.asarray(
        solve_result["basin_concentrations_mol_m3"],
        dtype=float,
    )
    density_weights: list[Array] = []
    energy_references = as_1d(
        basin_energy_references_J_mol,
        "basin_energy_references_J_mol",
    )
    if energy_references.size != len(basin_quadrature_points):
        raise ValueError("basin energy reference count must equal basin count")
    for basin_index, (points, weights, energy_reference_J_mol) in enumerate(
        zip(
            basin_quadrature_points,
            basin_quadrature_weights,
            energy_references,
            strict=True,
        )
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
            relative_energy_J_mol = (
                float(potential_energy_J_mol(point))
                - float(energy_reference_J_mol)
            )
            log_terms[point_index] = (
                np.log(positive_weight) - beta_mol * relative_energy_J_mol
            )
        log_normalizer = float(np.max(log_terms))
        normalized_weights = np.exp(log_terms - log_normalizer)
        normalized_weights /= positive_float(
            float(np.sum(normalized_weights)),
            "basin_density_normalized_weight_sum",
        )
        weights_mol_m3 = solved_concentrations[basin_index] * normalized_weights
        density_weights.append(weights_mol_m3)
    quadrature_concentrations = np.asarray(
        [float(np.sum(weights)) for weights in density_weights],
        dtype=float,
    )
    if not np.allclose(
        quadrature_concentrations,
        solved_concentrations,
        atol=CHEMICAL_POTENTIAL_MASS_TOL,
        rtol=CHEMICAL_POTENTIAL_MASS_TOL,
    ):
        raise ValueError(
            "density quadrature weights do not reproduce basin concentrations"
        )
    return {
        "chemical_potentials": chemical_potentials,
        "basin_concentrations_mol_m3": solved_concentrations,
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
    restricted_log_partition_values: Array,
) -> Array:
    log_concentrations = (
        np.log(STANDARD_CONCENTRATION_MOL_M3)
        + restricted_log_partition_values
        + stoichiometry @ chemical_potentials
    )
    basin_concentrations = strict_nonnegative_finite_array(
        np.exp(np.maximum(log_concentrations, np.log(np.nextafter(0.0, 1.0)))),
        "basin_concentrations_mol_m3",
    )
    if basin_concentrations.ndim != 1:
        raise ValueError("basin_concentrations_mol_m3 must be a 1D array")
    return np.asarray(basin_concentrations, dtype=float)


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


def transition_normal_gradient_matrix_for_state(
    state_index: int,
    coordinate_dimension: int,
    transition_committor_gradients: Sequence[Array],
    transition_surface_state_indices: Sequence[Array],
) -> Array:
    """Collect the committor normals that constrain one state's local motion."""

    if state_index < 0:
        raise ValueError("state_index must be nonnegative")
    if coordinate_dimension <= 0:
        raise ValueError("coordinate_dimension must be positive")
    if len(transition_committor_gradients) != len(transition_surface_state_indices):
        raise ValueError("transition gradient/state collections must have equal length")
    state_normal_gradients = []
    for transition_index, (committor_gradients, surface_state_indices) in enumerate(
        zip(
            transition_committor_gradients,
            transition_surface_state_indices,
            strict=True,
        )
    ):
        gradient_matrix = as_2d(
            committor_gradients,
            f"transition_committor_gradients[{transition_index}]",
        )
        if gradient_matrix.shape[1] != coordinate_dimension:
            raise ValueError(
                "transition committor gradient has wrong coordinate dimension"
            )
        surface_states = np.asarray(surface_state_indices, dtype=int)
        if surface_states.shape != (gradient_matrix.shape[0],):
            raise ValueError("transition surface states must match gradient rows")
        if np.any(surface_states < 0):
            raise ValueError("transition surface state indices must be nonnegative")
        state_normal_gradients.extend(gradient_matrix[surface_states == state_index])
    if not state_normal_gradients:
        return np.empty((0, coordinate_dimension), dtype=float)
    normal_gradient_matrix = np.asarray(state_normal_gradients, dtype=float)
    if not np.all(np.isfinite(normal_gradient_matrix)):
        raise ValueError("transition normal gradients must be finite")
    return normal_gradient_matrix


def tangent_mobility(
    mobility_tensor_m2_s: Array,
    transition_normal_gradient_matrix: Array,
) -> Array:
    """Project a PSD mobility onto the tangent space of transition constraints."""

    mobility = as_square_any(mobility_tensor_m2_s, "mobility_tensor_m2_s")
    if not np.allclose(mobility, mobility.T, atol=PSD_TOL, rtol=PSD_TOL):
        raise ValueError("mobility_tensor_m2_s must be symmetric")
    validate_psd(mobility, "mobility_tensor_m2_s", allow_zero=True)
    normal_gradients = as_2d(
        transition_normal_gradient_matrix,
        "transition_normal_gradient_matrix",
    )
    if normal_gradients.shape[1] != mobility.shape[0]:
        raise ValueError("transition normals and mobility have incompatible dimensions")
    if not np.all(np.isfinite(normal_gradients)):
        raise ValueError("transition_normal_gradient_matrix must be finite")
    if normal_gradients.shape[0] == 0:
        return mobility.copy()

    mobility_eigenvalues, mobility_eigenvectors = np.linalg.eigh(mobility)
    maximum_mobility_eigenvalue = float(np.max(mobility_eigenvalues))
    positive_mobility_mask = mobility_eigenvalues > (
        PSEUDOINVERSE_RELATIVE_TOL * maximum_mobility_eigenvalue
    )
    if not np.any(positive_mobility_mask):
        return np.zeros_like(mobility)
    mobility_square_root = (
        mobility_eigenvectors[:, positive_mobility_mask]
        * np.sqrt(mobility_eigenvalues[positive_mobility_mask])[None, :]
    )
    whitened_normals = normal_gradients @ mobility_square_root
    _, singular_values, right_singular_vectors = np.linalg.svd(
        whitened_normals,
        full_matrices=False,
    )
    maximum_singular_value = float(np.max(singular_values))
    active_normal_mask = singular_values > (
        PSEUDOINVERSE_RELATIVE_TOL * maximum_singular_value
    )
    tangent_projector = np.eye(mobility_square_root.shape[1], dtype=float)
    if np.any(active_normal_mask):
        active_normal_rows = right_singular_vectors[active_normal_mask]
        tangent_projector -= active_normal_rows.T @ active_normal_rows
    tangent_mobility_tensor = _symmetrize(
        mobility_square_root @ tangent_projector @ mobility_square_root.T
    )
    validate_psd(tangent_mobility_tensor, "tangent_mobility", allow_zero=True)

    mobility_rank = np.linalg.matrix_rank(mobility, hermitian=True)
    tangent_rank = np.linalg.matrix_rank(tangent_mobility_tensor, hermitian=True)
    if tangent_rank > mobility_rank:
        raise ValueError("tangent mobility rank exceeds the source mobility rank")
    normal_residual = normal_gradients @ tangent_mobility_tensor
    residual_scale = max(
        _maximum_abs_entry(normal_gradients @ mobility),
        np.finfo(float).tiny,
    )
    if (
        _maximum_abs_entry(normal_residual)
        > PSEUDOINVERSE_RELATIVE_TOL * residual_scale
    ):
        raise ValueError("tangent mobility retains mobility along a transition normal")
    return tangent_mobility_tensor


def compute_self_current_tangent_tensor(
    mobility_tensor_m2_s: Array,
    charge_polarization_gradient: Array,
    transition_normal_gradient_matrix: Array,
) -> Array:
    """Compute the Cartesian self-current tensor from tangent state mobility."""

    mobility = as_square_any(mobility_tensor_m2_s, "mobility_tensor_m2_s")
    charge_gradient = as_matrix_shape(
        charge_polarization_gradient,
        (CARTESIAN, mobility.shape[0]),
        "charge_polarization_gradient",
    )
    tangent_mobility_tensor = tangent_mobility(
        mobility_tensor_m2_s=mobility,
        transition_normal_gradient_matrix=transition_normal_gradient_matrix,
    )
    self_current_tensor = _symmetrize(
        charge_gradient @ tangent_mobility_tensor @ charge_gradient.T
    )
    validate_psd(self_current_tensor, "self_current_tangent_tensor", allow_zero=True)
    return self_current_tensor


def compute_transport_ownership_tensor_set(
    state_index: int,
    quadrature_index: int,
    mobility_tensor_m2_s: Array,
    charge_polarization_gradient: Array,
    ownership_basis: StateTransportOwnershipBasis,
) -> TransportOwnershipTensorSet:
    mobility = as_square_any(mobility_tensor_m2_s, "mobility_tensor_m2_s")
    validate_psd(mobility, "mobility_tensor_m2_s", allow_zero=True)
    charge_gradient = as_matrix_shape(
        charge_polarization_gradient,
        (CARTESIAN, mobility.shape[0]),
        "charge_polarization_gradient",
    )
    transition_gradients = _validated_ownership_gradient_matrix(
        ownership_basis.transition_displacement_gradients,
        mobility.shape[0],
        "transition_displacement_gradients",
    )
    bounded_memory_gradients = _validated_ownership_gradient_matrix(
        ownership_basis.bounded_memory_gradients,
        mobility.shape[0],
        "bounded_memory_gradients",
    )
    diagnostic_gradients = _validated_ownership_gradient_matrix(
        ownership_basis.diagnostic_gradients,
        mobility.shape[0],
        "diagnostic_gradients",
    )
    _validate_ownership_source_cardinality(ownership_basis)

    mobility_eigenvalues, mobility_eigenvectors = np.linalg.eigh(mobility)
    maximum_eigenvalue = max(float(np.max(mobility_eigenvalues)), 0.0)
    support_mask = mobility_eigenvalues > (
        PSEUDOINVERSE_RELATIVE_TOL * maximum_eigenvalue
    )
    mobility_square_root = np.zeros_like(mobility)
    support_projector = np.zeros_like(mobility)
    if np.any(support_mask):
        support_vectors = mobility_eigenvectors[:, support_mask]
        mobility_square_root = _symmetrize(
            (support_vectors * np.sqrt(mobility_eigenvalues[support_mask])[None, :])
            @ support_vectors.T
        )
        support_projector = _symmetrize(support_vectors @ support_vectors.T)

    transition_projector = _row_space_projector_within_support(
        transition_gradients @ mobility_square_root,
        support_projector,
    )
    memory_residual_support = _orthogonal_complement_projector(
        support_projector,
        transition_projector,
    )
    bounded_memory_projector = _row_space_projector_within_support(
        bounded_memory_gradients @ mobility_square_root,
        memory_residual_support,
    )
    diagnostic_residual_support = _orthogonal_complement_projector(
        support_projector,
        transition_projector + bounded_memory_projector,
    )
    diagnostic_projector = _row_space_projector_within_support(
        diagnostic_gradients @ mobility_square_root,
        diagnostic_residual_support,
    )
    dc_self_projector = _orthogonal_complement_projector(
        support_projector,
        transition_projector + bounded_memory_projector + diagnostic_projector,
    )
    owner_projectors = (
        dc_self_projector,
        transition_projector,
        bounded_memory_projector,
        diagnostic_projector,
    )
    _validate_ownership_projectors(owner_projectors, support_projector)
    owner_mobilities = tuple(
        _symmetrize(mobility_square_root @ projector @ mobility_square_root)
        for projector in owner_projectors
    )
    owner_tensors = tuple(
        _symmetrize(charge_gradient @ owner_mobility @ charge_gradient.T)
        for owner_mobility in owner_mobilities
    )
    supported_mobility = _symmetrize(
        mobility_square_root @ support_projector @ mobility_square_root
    )
    full_short_time_tensor = _symmetrize(
        charge_gradient @ supported_mobility @ charge_gradient.T
    )
    closure_residual = _symmetrize(
        full_short_time_tensor - sum(owner_tensors, start=np.zeros((3, 3)))
    )
    closure_scale = max(
        _maximum_abs_eigenvalue(full_short_time_tensor),
        *(_maximum_abs_eigenvalue(owner_tensor) for owner_tensor in owner_tensors),
        np.finfo(float).tiny,
    )
    tensor_construction_scale = _maximum_abs_entry(
        np.abs(charge_gradient)
        @ np.abs(supported_mobility)
        @ np.abs(charge_gradient.T)
    )
    floating_point_tolerance = (
        np.finfo(float).eps
        * mobility.shape[0]
        * max(tensor_construction_scale, np.finfo(float).tiny)
    )
    closure_tolerance = max(
        PSEUDOINVERSE_RELATIVE_TOL * closure_scale,
        floating_point_tolerance,
    )
    closure_residual_magnitude = _maximum_abs_eigenvalue(closure_residual)
    if closure_residual_magnitude > closure_tolerance:
        raise ValueError(
            "TRANSPORT_OWNER_TENSOR_CLOSURE_FAILED: "
            f"state={state_index}; quadrature={quadrature_index}; "
            f"residual={closure_residual_magnitude:.16e}; "
            f"tolerance={closure_tolerance:.16e}; scale={closure_scale:.16e}"
        )
    diagnostic_tensor = owner_tensors[3]
    if _maximum_abs_eigenvalue(diagnostic_tensor) > closure_tolerance:
        raise ValueError(
            "DIAGNOSTIC_CURRENT_NONZERO: "
            f"state={state_index}; quadrature={quadrature_index}"
        )
    for owner_index, owner_tensor in enumerate(owner_tensors):
        validate_psd(
            owner_tensor,
            f"transport_owner_tensor[{owner_index}]",
            allow_zero=True,
        )
    return TransportOwnershipTensorSet(
        state_index=state_index,
        quadrature_index=quadrature_index,
        full_short_time_tensor_m2_s=full_short_time_tensor,
        dc_self_tensor_m2_s=owner_tensors[0],
        transition_displacement_tensor_m2_s=owner_tensors[1],
        bounded_memory_tensor_m2_s=owner_tensors[2],
        diagnostic_tensor_m2_s=diagnostic_tensor,
        closure_residual_tensor_m2_s=closure_residual,
        coordinate_support_rank=int(np.count_nonzero(support_mask)),
        transition_rank=int(np.linalg.matrix_rank(transition_projector)),
        bounded_memory_rank=int(np.linalg.matrix_rank(bounded_memory_projector)),
        diagnostic_rank=int(np.linalg.matrix_rank(diagnostic_projector)),
    )


def compute_state_transport_ownership_quadratures(
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    self_current_coordinate_projectors: Sequence[Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
    basin_concentrations_mol_m3: Array,
    state_transport_ownership_bases: Sequence[
        Sequence[StateTransportOwnershipBasis]
    ],
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    tuple[StateTransportOwnershipQuadrature, ...],
]:
    concentrations = positive_vector(
        basin_concentrations_mol_m3,
        "basin_concentrations_mol_m3",
    )
    state_count = len(basin_quadrature_points)
    if len(basin_density_weights_mol_m3) != state_count:
        raise ValueError("basin density-weight count must equal state count")
    if len(state_transport_ownership_bases) != state_count:
        raise ValueError("state ownership-basis count must equal state count")
    if concentrations.size != state_count:
        raise ValueError("basin concentration count must equal state count")
    if len(self_current_coordinate_projectors) != state_count:
        raise ValueError("self-current projector count must equal state count")
    quadratures = tuple(
        _compute_state_transport_ownership_quadrature(
            state_index=state_index,
            mobility_tensor_m2_s=mobility_tensor_m2_s,
            charge_polarization_gradient=charge_polarization_gradient,
            self_current_coordinate_projector=self_current_coordinate_projectors[
                state_index
            ],
            state_points=basin_quadrature_points[state_index],
            state_density_weights=basin_density_weights_mol_m3[state_index],
            state_concentration_mol_m3=float(concentrations[state_index]),
            state_bases=state_transport_ownership_bases[state_index],
        )
        for state_index in range(state_count)
    )
    return (
        np.asarray(
            [item.density_weighted_full_tensor_m2_s for item in quadratures]
        ),
        np.asarray(
            [item.density_weighted_dc_self_tensor_m2_s for item in quadratures]
        ),
        np.asarray(
            [
                item.density_weighted_transition_displacement_tensor_m2_s
                for item in quadratures
            ]
        ),
        np.asarray(
            [item.density_weighted_bounded_memory_tensor_m2_s for item in quadratures]
        ),
        np.asarray(
            [item.density_weighted_diagnostic_tensor_m2_s for item in quadratures]
        ),
        quadratures,
    )


def _compute_state_transport_ownership_quadrature(
    state_index: int,
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    self_current_coordinate_projector: Array,
    state_points: Array,
    state_density_weights: Array,
    state_concentration_mol_m3: float,
    state_bases: Sequence[StateTransportOwnershipBasis],
) -> StateTransportOwnershipQuadrature:
    points = as_2d(state_points, f"basin_quadrature_points[{state_index}]")
    density_weights = as_1d(
        state_density_weights,
        f"basin_density_weights_mol_m3[{state_index}]",
    )
    if points.shape[0] != density_weights.size:
        raise ValueError("basin point/density-weight count mismatch")
    if len(state_bases) != points.shape[0]:
        raise ValueError("state ownership bases must align with basin points")
    coordinate_projector = as_matrix_shape(
        self_current_coordinate_projector,
        (points.shape[1], points.shape[1]),
        f"self_current_coordinate_projectors[{state_index}]",
    )
    point_tensors = tuple(
        compute_transport_ownership_tensor_set(
            state_index=state_index,
            quadrature_index=quadrature_index,
            mobility_tensor_m2_s=_symmetrize(
                coordinate_projector
                @ mobility_tensor_m2_s(point)
                @ coordinate_projector.T
            ),
            charge_polarization_gradient=charge_polarization_gradient(point),
            ownership_basis=state_bases[quadrature_index],
        )
        for quadrature_index, point in enumerate(points)
    )
    return StateTransportOwnershipQuadrature(
        point_tensors=point_tensors,
        density_weighted_full_tensor_m2_s=_aggregate_ownership_tensor_field(
            point_tensors,
            density_weights,
            state_concentration_mol_m3,
            "full_short_time_tensor_m2_s",
        ),
        density_weighted_dc_self_tensor_m2_s=_aggregate_ownership_tensor_field(
            point_tensors,
            density_weights,
            state_concentration_mol_m3,
            "dc_self_tensor_m2_s",
        ),
        density_weighted_transition_displacement_tensor_m2_s=(
            _aggregate_ownership_tensor_field(
                point_tensors,
                density_weights,
                state_concentration_mol_m3,
                "transition_displacement_tensor_m2_s",
            )
        ),
        density_weighted_bounded_memory_tensor_m2_s=(
            _aggregate_ownership_tensor_field(
                point_tensors,
                density_weights,
                state_concentration_mol_m3,
                "bounded_memory_tensor_m2_s",
            )
        ),
        density_weighted_diagnostic_tensor_m2_s=_aggregate_ownership_tensor_field(
            point_tensors,
            density_weights,
            state_concentration_mol_m3,
            "diagnostic_tensor_m2_s",
        ),
        maximum_closure_residual_m2_s=max(
            _maximum_abs_eigenvalue(point_tensor.closure_residual_tensor_m2_s)
            for point_tensor in point_tensors
        ),
    )


def _aggregate_ownership_tensor_field(
    point_tensors: tuple[TransportOwnershipTensorSet, ...],
    density_weights: Array,
    state_concentration_mol_m3: float,
    field_name: str,
) -> Array:
    return _symmetrize(
        sum(
            (
                float(density_weight)
                * np.asarray(getattr(point_tensor, field_name), dtype=float)
                for density_weight, point_tensor in zip(
                    density_weights,
                    point_tensors,
                    strict=True,
                )
            ),
            start=np.zeros((CARTESIAN, CARTESIAN), dtype=float),
        )
        / positive_float(state_concentration_mol_m3, "state_concentration_mol_m3")
    )


def _validated_ownership_gradient_matrix(
    gradient_matrix: Array,
    coordinate_dimension: int,
    label: str,
) -> Array:
    gradients = as_2d(gradient_matrix, label)
    if gradients.shape[1] != coordinate_dimension:
        raise ValueError(f"{label} has wrong coordinate dimension")
    if not np.all(np.isfinite(gradients)):
        raise ValueError(f"{label} must be finite")
    return gradients


def _validate_ownership_source_cardinality(
    ownership_basis: StateTransportOwnershipBasis,
) -> None:
    transition_edge_indices = np.asarray(
        ownership_basis.transition_edge_indices,
        dtype=int,
    )
    bounded_memory_mode_indices = np.asarray(
        ownership_basis.bounded_memory_mode_indices,
        dtype=int,
    )
    if transition_edge_indices.shape != (
        ownership_basis.transition_displacement_gradients.shape[0],
    ):
        raise ValueError("TRANSITION_OWNER_SOURCE_CARDINALITY_FAILED")
    if bounded_memory_mode_indices.shape != (
        ownership_basis.bounded_memory_gradients.shape[0],
    ):
        raise ValueError("MEMORY_OWNER_SOURCE_CARDINALITY_FAILED")
    if len(ownership_basis.diagnostic_source_ids) != (
        ownership_basis.diagnostic_gradients.shape[0]
    ):
        raise ValueError("DIAGNOSTIC_OWNER_SOURCE_CARDINALITY_FAILED")
    if np.any(transition_edge_indices < 0) or np.any(bounded_memory_mode_indices < 0):
        raise ValueError("TRANSPORT_OWNER_SOURCE_INDEX_INVALID")


def _row_space_projector(row_matrix: Array) -> Array:
    rows = as_2d(row_matrix, "row_matrix")
    coordinate_dimension = rows.shape[1]
    if rows.shape[0] == 0:
        return np.zeros((coordinate_dimension, coordinate_dimension), dtype=float)
    _left_vectors, singular_values, right_vectors = np.linalg.svd(
        rows,
        full_matrices=False,
    )
    maximum_singular_value = max(float(np.max(singular_values)), 0.0)
    active_rows = singular_values > (
        PSEUDOINVERSE_RELATIVE_TOL * maximum_singular_value
    )
    if not np.any(active_rows):
        return np.zeros((coordinate_dimension, coordinate_dimension), dtype=float)
    basis = right_vectors[active_rows]
    return _symmetrize(basis.T @ basis)


def _row_space_projector_within_support(
    row_matrix: Array,
    support_projector: Array,
) -> Array:
    supported_rows = as_2d(row_matrix, "row_matrix") @ support_projector
    row_projector = _row_space_projector(supported_rows)
    return _spectral_projector(
        support_projector @ row_projector @ support_projector
    )


def _orthogonal_complement_projector(
    support_projector: Array,
    owned_projector: Array,
) -> Array:
    return _spectral_projector(
        support_projector - support_projector @ owned_projector @ support_projector
    )


def _spectral_projector(candidate_projector: Array) -> Array:
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(candidate_projector))
    active_mask = eigenvalues > 0.5
    if not np.any(active_mask):
        return np.zeros_like(candidate_projector)
    active_vectors = eigenvectors[:, active_mask]
    return _symmetrize(active_vectors @ active_vectors.T)


def _validate_ownership_projectors(
    owner_projectors: tuple[Array, Array, Array, Array],
    support_projector: Array,
) -> None:
    tolerance = PSEUDOINVERSE_RELATIVE_TOL * max(
        _maximum_abs_eigenvalue(support_projector),
        np.finfo(float).tiny,
    )
    for owner_index, projector in enumerate(owner_projectors):
        if _maximum_abs_entry(projector - projector.T) > tolerance:
            raise ValueError(
                f"TRANSPORT_OWNER_PROJECTOR_NOT_SYMMETRIC: owner={owner_index}"
            )
        if _maximum_abs_entry(projector @ projector - projector) > tolerance:
            raise ValueError(
                f"TRANSPORT_OWNER_PROJECTOR_NOT_IDEMPOTENT: owner={owner_index}"
            )
        validate_psd(
            projector,
            f"transport_owner_projector[{owner_index}]",
            allow_zero=True,
        )
    for first_index, first_projector in enumerate(owner_projectors):
        for second_index in range(first_index + 1, len(owner_projectors)):
            overlap = first_projector @ owner_projectors[second_index]
            if _maximum_abs_entry(overlap) > tolerance:
                raise ValueError(
                    "TRANSPORT_OWNER_PROJECTOR_OVERLAP: "
                    f"owners={first_index},{second_index}"
                )
    closure = _symmetrize(support_projector - sum(owner_projectors))
    if _maximum_abs_entry(closure) > tolerance:
        raise ValueError("TRANSPORT_OWNER_PROJECTOR_CLOSURE_FAILED")


def compute_self_current_tensors(
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
    basin_concentrations_mol_m3: Array,
    self_current_coordinate_projectors: Sequence[Array],
    self_current_ownership_normal_gradients_by_state: Sequence[Sequence[Array]],
    bounded_memory_coordinate_gradient: Callable[[Array], Array],
    bounded_memory_mode_indices: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    concentrations = positive_vector(
        basin_concentrations_mol_m3,
        "basin_concentrations_mol_m3",
    )
    bounded_mode_indices = np.asarray(bounded_memory_mode_indices, dtype=int)
    if bounded_mode_indices.ndim != 1 or np.any(bounded_mode_indices < 0):
        raise ValueError("bounded_memory_mode_indices must be nonnegative and 1D")
    state_count = len(basin_quadrature_points)
    full_tensors = np.zeros((state_count, CARTESIAN, CARTESIAN), dtype=float)
    tangent_tensors = np.zeros((state_count, CARTESIAN, CARTESIAN), dtype=float)
    transition_owned_tensors = np.zeros_like(full_tensors)
    bounded_memory_owned_tensors = np.zeros_like(full_tensors)
    diagnostic_owned_tensors = np.zeros_like(full_tensors)
    if len(basin_density_weights_mol_m3) != state_count:
        raise ValueError("basin_density_weights_mol_m3 length must equal state count")
    if concentrations.size != state_count:
        raise ValueError("basin_concentrations_mol_m3 length must equal state count")
    if len(self_current_coordinate_projectors) != state_count:
        raise ValueError(
            "self_current_coordinate_projectors length must equal state count"
        )
    if len(self_current_ownership_normal_gradients_by_state) != state_count:
        raise ValueError("self-current ownership normal count must equal state count")
    for i, (points, density_weights) in enumerate(
        zip(basin_quadrature_points, basin_density_weights_mol_m3, strict=True)
    ):
        pts = as_2d(points, "basin_quadrature_points[]")
        W = as_1d(density_weights, "basin_density_weights_mol_m3[]")
        if pts.shape[0] != W.size:
            raise ValueError("basin quadrature point/density-weight count mismatch")
        coordinate_projector = as_matrix_shape(
            self_current_coordinate_projectors[i],
            (pts.shape[1], pts.shape[1]),
            f"self_current_coordinate_projectors[{i}]",
        )
        state_point_normal_gradients = self_current_ownership_normal_gradients_by_state[
            i
        ]
        if len(state_point_normal_gradients) != pts.shape[0]:
            raise ValueError(
                "self-current ownership normals must have one matrix per basin point"
            )
        full_numerator = np.zeros((CARTESIAN, CARTESIAN), dtype=float)
        tangent_numerator = np.zeros((CARTESIAN, CARTESIAN), dtype=float)
        transition_owned_numerator = np.zeros((CARTESIAN, CARTESIAN), dtype=float)
        bounded_memory_owned_numerator = np.zeros((CARTESIAN, CARTESIAN), dtype=float)
        diagnostic_owned_numerator = np.zeros((CARTESIAN, CARTESIAN), dtype=float)
        for point_index, (point, density_weight) in enumerate(zip(pts, W)):
            ownership_normal_gradients = as_2d(
                state_point_normal_gradients[point_index],
                (
                    "self_current_ownership_normal_gradients_by_state"
                    f"[{i}][{point_index}]"
                ),
            )
            if ownership_normal_gradients.shape[1] != pts.shape[1]:
                raise ValueError(
                    "self-current ownership normals have wrong coordinate dimension"
                )
            mobility = as_square(
                mobility_tensor_m2_s(point), point.size, "mobility_tensor"
            )
            charge_gradient = as_matrix_shape(
                charge_polarization_gradient(point),
                (CARTESIAN, point.size),
                "charge_polarization_gradient",
            )
            bounded_memory_gradients = as_2d(
                bounded_memory_coordinate_gradient(point),
                "bounded_memory_coordinate_gradient",
            )
            if bounded_mode_indices.size:
                if (
                    int(np.max(bounded_mode_indices))
                    >= bounded_memory_gradients.shape[0]
                ):
                    raise ValueError("bounded memory mode index is out of range")
                bounded_memory_gradients = bounded_memory_gradients[
                    bounded_mode_indices
                ]
            else:
                bounded_memory_gradients = np.empty((0, point.size), dtype=float)
            if bounded_memory_gradients.shape[1] != point.size:
                raise ValueError(
                    "bounded memory gradient has wrong coordinate dimension"
                )
            owned_gradient_rows = np.vstack(
                (ownership_normal_gradients, bounded_memory_gradients)
            )
            base_mobility = _symmetrize(
                coordinate_projector @ mobility @ coordinate_projector.T
            )
            transition_tangent_mobility = tangent_mobility(
                mobility_tensor_m2_s=base_mobility,
                transition_normal_gradient_matrix=ownership_normal_gradients,
            )
            projected_mobility = tangent_mobility(
                mobility_tensor_m2_s=base_mobility,
                transition_normal_gradient_matrix=owned_gradient_rows,
            )
            full_numerator += float(density_weight) * _symmetrize(
                charge_gradient @ mobility @ charge_gradient.T
            )
            point_self_current_tensor = _symmetrize(
                charge_gradient @ projected_mobility @ charge_gradient.T
            )
            tangent_numerator += float(density_weight) * point_self_current_tensor
            transition_owned_numerator += float(density_weight) * _symmetrize(
                charge_gradient
                @ (base_mobility - transition_tangent_mobility)
                @ charge_gradient.T
            )
            bounded_memory_owned_numerator += float(density_weight) * _symmetrize(
                charge_gradient
                @ (transition_tangent_mobility - projected_mobility)
                @ charge_gradient.T
            )
            diagnostic_owned_numerator += float(density_weight) * _symmetrize(
                charge_gradient @ (mobility - base_mobility) @ charge_gradient.T
            )
        full_tensors[i] = _symmetrize(full_numerator / concentrations[i])
        tangent_tensors[i] = _symmetrize(tangent_numerator / concentrations[i])
        transition_owned_tensors[i] = _symmetrize(
            transition_owned_numerator / concentrations[i]
        )
        bounded_memory_owned_tensors[i] = _symmetrize(
            bounded_memory_owned_numerator / concentrations[i]
        )
        diagnostic_owned_tensors[i] = _symmetrize(
            diagnostic_owned_numerator / concentrations[i]
        )
        validate_psd(full_tensors[i], f"D_self_full[{i}]", allow_zero=True)
        validate_psd(tangent_tensors[i], f"D_self_tangent[{i}]", allow_zero=True)
        validate_psd(
            transition_owned_tensors[i],
            f"D_transition_owned[{i}]",
            allow_zero=True,
        )
        validate_psd(
            bounded_memory_owned_tensors[i],
            f"D_bounded_memory_owned[{i}]",
            allow_zero=True,
        )
        validate_psd(
            diagnostic_owned_tensors[i],
            f"D_diagnostic_owned[{i}]",
            allow_zero=True,
        )
        ownership_closure_residual = _symmetrize(
            full_tensors[i]
            - tangent_tensors[i]
            - transition_owned_tensors[i]
            - bounded_memory_owned_tensors[i]
            - diagnostic_owned_tensors[i]
        )
        closure_scale = max(
            _maximum_abs_eigenvalue(full_tensors[i]),
            np.finfo(float).tiny,
        )
        if _maximum_abs_eigenvalue(ownership_closure_residual) > (
            max(
                PROJECTED_DIFFUSIVITY_PSD_RELATIVE_TOL,
                pts.shape[1] * np.sqrt(np.finfo(float).eps),
            )
            * closure_scale
        ):
            raise ValueError(
                "STATE_OWNERSHIP_CLOSURE_FAILED: "
                f"state={i}; residual={_maximum_abs_eigenvalue(ownership_closure_residual)}; "
                f"scale={closure_scale}"
            )
    return (
        full_tensors,
        tangent_tensors,
        transition_owned_tensors,
        bounded_memory_owned_tensors,
        diagnostic_owned_tensors,
    )


def compute_mori_memory_matrices(
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    memory_coordinate_gradient: Callable[[Array], Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
    state_memory_active_mask: Array,
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
    active_mask = np.asarray(state_memory_active_mask, dtype=bool)
    if active_mask.shape != (len(basin_quadrature_points), mem_dim):
        raise ValueError("state memory active mask has wrong shape")
    for state_index, (points, density_weights) in enumerate(zip(
        basin_quadrature_points,
        basin_density_weights_mol_m3,
        strict=True,
    )):
        pts = as_2d(points, "basin_quadrature_points[]")
        W = as_1d(density_weights, "basin_density_weights_mol_m3[]")
        if pts.shape[0] != W.size:
            raise ValueError("basin quadrature point/density-weight count mismatch")
        for point, density_weight in zip(pts, W):
            D = as_square(mobility_tensor_m2_s(point), point.size, "mobility_tensor")
            gradpsi = as_2d(
                memory_coordinate_gradient(point), "memory_coordinate_gradient"
            )
            gradpsi = gradpsi * active_mask[state_index, :, np.newaxis]
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
        np.ones(
            (
                len(basin_quadrature_points),
                int(
                    np.asarray(
                        candidate_gradients(
                            np.asarray(basin_quadrature_points[0], dtype=float)[0]
                        )
                    ).shape[0]
                ),
            ),
            dtype=bool,
        ),
    )
    candidate_count = raw_memory_matrix.shape[0]
    remaining_tensor_base = _symmetrize(
        as_matrix_shape(
            direct_minus_finite_state_tensor,
            (CARTESIAN, CARTESIAN),
            "direct_minus_finite_state_tensor",
        )
    )
    energy_scale = max(_maximum_abs_entry(raw_memory_matrix), np.finfo(float).tiny)
    coupling_scale = max(_maximum_abs_entry(raw_current_coupling), np.finfo(float).tiny)
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
            _maximum_abs_eigenvalue(remaining_tensor),
            np.finfo(float).tiny,
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


def _scale_aware_nonzero_threshold(values: Array) -> float:
    magnitude_scale = max(_maximum_abs_entry(values), np.finfo(float).tiny)
    return FINITE_PROCESS_NONZERO_EPSILON_FACTOR * np.finfo(float).eps * magnitude_scale


def compute_state_memory_coordinate_means(
    memory_coordinates: Callable[[Array], Array],
    basin_quadrature_points: Sequence[Array],
    basin_density_weights_mol_m3: Sequence[Array],
    basin_concentrations_mol_m3: Array,
) -> tuple[Array, Array]:
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


def compute_discrete_state_mori_matrices(
    state_concentrations_mol_m3: Array,
    reversible_generator_Q_ij_s_inv: Array,
    transition_first_moments_d_ij_m: Array,
    state_memory_value_matrix: Array,
) -> tuple[Array, Array]:
    """Build finite-state Dirichlet energy and current coupling in value coordinates."""

    concentrations = positive_vector(
        state_concentrations_mol_m3, "state_concentrations_mol_m3"
    )
    generator = as_square(
        reversible_generator_Q_ij_s_inv,
        concentrations.size,
        "reversible_generator_Q_ij_s_inv",
    )
    memory_values = as_2d(state_memory_value_matrix, "state_memory_value_matrix")
    if memory_values.shape[0] != concentrations.size:
        raise ValueError("state_memory_value_matrix row count must match states")
    validate_reversible_generator(generator, concentrations)
    state_drift = compute_reversible_finite_state_drift(
        concentrations,
        generator,
        transition_first_moments_d_ij_m,
    )
    value_differences = memory_values[np.newaxis, :, :] - memory_values[:, np.newaxis, :]
    capacity_fluxes = concentrations[:, np.newaxis] * generator
    memory_matrix = 0.5 * np.einsum(
        "ij,ijm,ijn->mn",
        capacity_fluxes,
        value_differences,
        value_differences,
    )
    current_coupling = np.einsum(
        "i,im,ia->ma", concentrations, memory_values, state_drift
    )
    memory_matrix = _symmetrize(memory_matrix)
    if memory_matrix.size:
        validate_psd(
            memory_matrix,
            "discrete_state_memory_matrix_A_Q",
            allow_zero=True,
        )
        current_coupling = (
            memory_matrix
            @ symmetric_psd_pseudoinverse(memory_matrix)
            @ current_coupling
        )
    return memory_matrix, current_coupling


def compute_direct_diffusivity_tensor(
    c: Array, K: Array, M: Array, Dself: Array
) -> Array:
    direct = np.einsum("i,iab->ab", c, Dself)
    direct += 0.5 * np.einsum("ij,ijab->ab", K, M)
    return _symmetrize(direct)


def compute_direct_primitive_audit_ledger(
    c: Array,
    K: Array,
    Q: Array,
    d: Array,
    M: Array,
    Dself_full: Array,
    Dself_tangent: Array,
    finite_state_correction: Array,
) -> DirectPrimitiveAuditLedger:
    """Attribute the direct primitive tensor and finite-state overlap exactly."""

    concentrations = positive_vector(c, "c")
    generator = as_square_any(Q, "Q")
    capacity_fluxes = as_square_any(K, "K")
    first_moments = np.asarray(d, dtype=float)
    second_moments = np.asarray(M, dtype=float)
    full_self_tensors = np.asarray(Dself_full, dtype=float)
    tangent_self_tensors = np.asarray(Dself_tangent, dtype=float)
    finite_state_drift = compute_reversible_finite_state_drift(
        concentrations, generator, first_moments
    )
    self_full = _symmetrize(np.einsum("i,iab->ab", concentrations, full_self_tensors))
    self_tangent = _symmetrize(
        np.einsum("i,iab->ab", concentrations, tangent_self_tensors)
    )
    overlap_removed = _symmetrize(self_full - self_tangent)
    transition = _symmetrize(
        0.5 * np.einsum("ij,ijab->ab", capacity_fluxes, second_moments)
    )
    total = _symmetrize(self_tangent + transition)
    component_records: list[StateDriftComponentAudit] = []
    for component_id, component_indices in enumerate(
        _generator_connected_components(generator)
    ):
        component_weighted_drift = (
            concentrations[component_indices] @ finite_state_drift[component_indices]
        )
        component_records.append(
            StateDriftComponentAudit(
                component_id=component_id,
                state_indices=component_indices.copy(),
                c_transpose_b_mol_m2_s=component_weighted_drift,
                c_transpose_b_norm_mol_m2_s=float(
                    np.linalg.norm(component_weighted_drift)
                ),
            )
        )
    return DirectPrimitiveAuditLedger(
        B_self_full_tensor_mol_m_s=self_full,
        B_self_tangent_tensor_mol_m_s=self_tangent,
        B_transition_tensor_mol_m_s=transition,
        B_overlap_removed_tensor_mol_m_s=overlap_removed,
        B_total_tensor_mol_m_s=total,
        C_Q_contribution_tensor_mol_m_s=_symmetrize(finite_state_correction),
        state_drift_b_i_m_s=finite_state_drift,
        state_exit_rates_s_inv=-np.diag(generator),
        state_drift_b_i_norms_m_s=np.linalg.norm(finite_state_drift, axis=1),
        state_drift_components=tuple(component_records),
    )


def direct_primitive_audit_as_effect_attribution(
    ledger: DirectPrimitiveAuditLedger,
) -> dict:
    return {
        "B_self_full_tensor_mol_m_s": ledger.B_self_full_tensor_mol_m_s,
        "B_self_full_trace_mol_m_s": float(np.trace(ledger.B_self_full_tensor_mol_m_s)),
        "B_self_tangent_tensor_mol_m_s": ledger.B_self_tangent_tensor_mol_m_s,
        "B_self_tangent_trace_mol_m_s": float(
            np.trace(ledger.B_self_tangent_tensor_mol_m_s)
        ),
        "B_transition_tensor_mol_m_s": ledger.B_transition_tensor_mol_m_s,
        "B_transition_trace_mol_m_s": float(
            np.trace(ledger.B_transition_tensor_mol_m_s)
        ),
        "B_overlap_removed_tensor_mol_m_s": ledger.B_overlap_removed_tensor_mol_m_s,
        "B_overlap_removed_trace_mol_m_s": float(
            np.trace(ledger.B_overlap_removed_tensor_mol_m_s)
        ),
        "B_total_tensor_mol_m_s": ledger.B_total_tensor_mol_m_s,
        "B_total_trace_mol_m_s": float(np.trace(ledger.B_total_tensor_mol_m_s)),
        "state_drift_b_i_m_s": ledger.state_drift_b_i_m_s,
        "state_exit_rates_s_inv": ledger.state_exit_rates_s_inv,
        "state_drift_b_i_norms_m_s": ledger.state_drift_b_i_norms_m_s,
        "state_drift_components": tuple(
            {
                "component_id": component.component_id,
                "state_indices": component.state_indices,
                "c_transpose_b_mol_m2_s": component.c_transpose_b_mol_m2_s,
                "c_transpose_b_norm_mol_m2_s": component.c_transpose_b_norm_mol_m2_s,
            }
            for component in ledger.state_drift_components
        ),
        "C_Q_contribution_tensor_mol_m_s": ledger.C_Q_contribution_tensor_mol_m_s,
        "C_Q_contribution_trace_mol_m_s": float(
            np.trace(ledger.C_Q_contribution_tensor_mol_m_s)
        ),
    }


def compute_finite_state_memory_correction(c: Array, Q: Array, d: Array) -> Array:
    if c.size == 1 or np.max(np.abs(d)) == 0.0 or np.max(np.abs(Q)) == 0.0:
        return np.zeros((CARTESIAN, CARTESIAN), dtype=float)
    finite_state_drift = compute_reversible_finite_state_drift(c, Q, d)
    correction = np.zeros((CARTESIAN, CARTESIAN), dtype=float)
    chis = []
    for axis in range(CARTESIAN):
        chis.append(solve_weighted_poisson(Q, c, finite_state_drift[:, axis]))
    for a in range(CARTESIAN):
        for b_axis in range(CARTESIAN):
            correction[a, b_axis] = float(
                np.sum(c * finite_state_drift[:, a] * chis[b_axis])
            )
    return _symmetrize(correction)


def compute_reversible_finite_state_drift(c: Array, Q: Array, d: Array) -> Array:
    """Return state-conditioned jump drift after enforcing reversible solvability."""

    concentrations = positive_vector(c, "c")
    generator = as_square_any(Q, "Q")
    first_moments = np.asarray(d, dtype=float)
    state_count = concentrations.size
    if generator.shape != (state_count, state_count):
        raise ValueError("Q shape does not match c")
    if first_moments.shape != (state_count, state_count, CARTESIAN) or not np.all(
        np.isfinite(first_moments)
    ):
        raise ValueError("d must have shape (n,n,3)")
    validate_reversible_generator(generator, concentrations)
    if not np.allclose(
        first_moments,
        -np.swapaxes(first_moments, 0, 1),
        atol=PSD_TOL,
        rtol=PSD_TOL,
    ):
        raise ValueError("d_ji must equal -d_ij")
    finite_state_drift = np.einsum("ij,ija->ia", generator, first_moments)
    for component_indices in _generator_connected_components(generator):
        component_concentrations = concentrations[component_indices]
        component_drift = finite_state_drift[component_indices]
        weighted_drift = component_concentrations @ component_drift
        weighted_absolute_scale = np.sum(
            np.abs(component_concentrations[:, np.newaxis] * component_drift),
            axis=0,
        )
        solvability_tolerance = np.maximum(
            POISSON_SOLVABILITY_ABS_TOL,
            POISSON_SOLVABILITY_EPSILON_FACTOR
            * np.finfo(float).eps
            * weighted_absolute_scale,
        )
        if np.any(np.abs(weighted_drift) > solvability_tolerance):
            raise ValueError(
                "reversible finite-state drift violates componentwise c^T b = 0"
            )
    return finite_state_drift


def compute_finite_process_readout_diagnostics(
    K: Array,
    d: Array,
    M: Array,
    max_transition_displacement_m: float,
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
    capacity_threshold_mol_m3_s = _scale_aware_nonzero_threshold(capacity_fluxes)
    active_transition_mask = capacity_fluxes > capacity_threshold_mol_m3_s
    active_transition_count = int(
        np.count_nonzero(np.triu(active_transition_mask, k=1))
    )
    first_moment_norms = np.linalg.norm(
        first_moments.reshape(-1, CARTESIAN),
        axis=1,
    ).reshape(capacity_fluxes.shape)
    second_moment_traces = np.trace(second_moments, axis1=2, axis2=3)
    displacement_scale_m = positive_float(
        max_transition_displacement_m,
        "max_transition_displacement_m",
    )
    first_moment_threshold_m = (
        FINITE_PROCESS_NONZERO_EPSILON_FACTOR
        * np.finfo(float).eps
        * displacement_scale_m
    )
    second_moment_threshold_m2 = (
        FINITE_PROCESS_NONZERO_EPSILON_FACTOR
        * np.finfo(float).eps
        * displacement_scale_m**2
    )
    active_first_moment_count = int(
        np.count_nonzero(
            np.triu(
                active_transition_mask
                & (first_moment_norms > first_moment_threshold_m),
                k=1,
            )
        )
    )
    active_second_moment_count = int(
        np.count_nonzero(
            np.triu(
                active_transition_mask
                & (second_moment_traces > second_moment_threshold_m2),
                k=1,
            )
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
    not_complete_reasons.extend(
        str(reason)
        for reason in effect_attribution.get(
            "physical_library_not_complete_reasons",
            (),
        )
    )
    not_complete_reasons.extend(
        str(reason)
        for reason in effect_attribution.get(
            "primitive_estimator_not_complete_reasons",
            (),
        )
    )
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
        "basis_refinement_selected_residual_score_history": np.asarray(
            refinement_result["selected_residual_score_history"],
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
    energy_threshold = _scale_aware_nonzero_threshold(residual_energy)
    valid_energy = residual_energy > energy_threshold
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
    require_candidate_set_exhaustion: bool,
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
    if candidate_matrix.size > 0:
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
        np.finfo(float).tiny,
    )
    coupling_scale = max(
        _maximum_abs_entry(candidate_coupling),
        _maximum_abs_entry(current_coupling),
        np.finfo(float).tiny,
    )
    energy_threshold = MEMORY_NULLSPACE_RELATIVE_TOL * energy_scale
    null_current_threshold = MEMORY_NULLSPACE_RELATIVE_TOL * coupling_scale
    conductivity_change = 0.0
    final_maximum_score = np.inf
    candidate_set_exhausted = False
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
                _maximum_abs_eigenvalue(remaining_tensor),
                np.finfo(float).tiny,
            ):
                available_candidates[candidate_index] = False
                eligible_candidates[candidate_index] = False
                rejected_psd_indices.append(int(candidate_index))
        scores = np.where(eligible_candidates, score_result["scores"], -np.inf)
        negligible_candidate_indices = np.flatnonzero(
            eligible_candidates & (scores <= score_tolerance)
        )
        for candidate_index in negligible_candidate_indices:
            available_candidates[candidate_index] = False
            discarded_candidate_indices.append(int(candidate_index))
        scores[negligible_candidate_indices] = -np.inf
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            final_maximum_score = 0.0
            candidate_set_exhausted = True
        else:
            final_maximum_score = float(np.max(finite_scores))
        maximum_score_history.append(final_maximum_score)
        residual_is_converged = final_maximum_score <= score_tolerance
        if require_candidate_set_exhaustion:
            candidate_policy_is_satisfied = (
                candidate_set_exhausted
                and conductivity_change <= conductivity_tolerance
            )
        else:
            candidate_policy_is_satisfied = (
                candidate_set_exhausted or conductivity_change <= conductivity_tolerance
            )
        if residual_is_converged and candidate_policy_is_satisfied:
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
    if require_candidate_set_exhaustion and not candidate_set_exhausted:
        convergence_status = "candidate_set_not_exhausted"
        not_complete_reasons.append("candidate_set_not_exhausted")
    elif final_maximum_score > score_tolerance:
        convergence_status = "basis_residual_above_tolerance"
        not_complete_reasons.append("basis_residual_score_above_tolerance")
    elif conductivity_change > conductivity_tolerance and (
        require_candidate_set_exhaustion or not candidate_set_exhausted
    ):
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
        "candidate_set_exhausted": candidate_set_exhausted,
        "candidate_count": int(candidate_count),
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
    energy_references = as_1d(
        x.basin_energy_references_J_mol,
        "basin_energy_references_J_mol",
    )
    if energy_references.size != state_count or not np.all(
        np.isfinite(energy_references)
    ):
        raise ValueError("basin energy references must be finite and state-aligned")
    memory_active_mask = np.asarray(x.state_memory_active_mask, dtype=bool)
    memory_count = int(
        np.asarray(
            x.memory_coordinate_gradient(x.basin_quadrature_points[0][0]),
            dtype=float,
        ).shape[0]
    )
    if memory_active_mask.shape != (state_count, memory_count):
        raise ValueError("state memory active mask must be state and mode aligned")
    validate_self_current_coordinate_projectors(
        x.self_current_coordinate_projectors,
        state_count,
        _infer_coordinate_dim(x.basin_quadrature_points),
    )
    coordinate_dimension = _infer_coordinate_dim(x.basin_quadrature_points)
    if len(x.state_transport_ownership_bases) != state_count:
        raise ValueError("state transport ownership basis count must equal state count")
    for state_index, state_bases in enumerate(x.state_transport_ownership_bases):
        if len(state_bases) != len(x.basin_quadrature_points[state_index]):
            raise ValueError("state ownership bases must align with basin points")
        for ownership_basis in state_bases:
            for gradient_matrix in (
                ownership_basis.transition_displacement_gradients,
                ownership_basis.bounded_memory_gradients,
                ownership_basis.diagnostic_gradients,
            ):
                if gradient_matrix.shape[1] != coordinate_dimension:
                    raise ValueError(
                        "state ownership basis coordinate dimension mismatch"
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
    if len(x.transition_transport_ownership) != transition_count:
        raise ValueError("transition ownership count must equal transition count")
    if any(
        not isinstance(ownership, TransportOwnership)
        for ownership in x.transition_transport_ownership
    ):
        raise TypeError("transition ownership values must be TransportOwnership")
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


def validate_self_current_coordinate_projectors(
    self_current_coordinate_projectors: Sequence[Array],
    state_count: int,
    coordinate_dimension: int,
) -> None:
    if len(self_current_coordinate_projectors) != state_count:
        raise ValueError(
            "self_current_coordinate_projectors length must equal state count"
        )
    for state_index, projector in enumerate(self_current_coordinate_projectors):
        projector_matrix = as_matrix_shape(
            projector,
            (coordinate_dimension, coordinate_dimension),
            f"self_current_coordinate_projectors[{state_index}]",
        )
        if not np.allclose(projector_matrix, projector_matrix.T):
            raise ValueError(
                f"self_current_coordinate_projectors[{state_index}] must be symmetric"
            )
        if not np.allclose(projector_matrix @ projector_matrix, projector_matrix):
            raise ValueError(
                f"self_current_coordinate_projectors[{state_index}] must be idempotent"
            )


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
            transition_outer_moment_m2 = np.outer(d[i, j], d[i, j])
            transition_covariance_m2 = M[i, j] - transition_outer_moment_m2
            covariance_eigenvalues_m2 = np.linalg.eigvalsh(
                _symmetrize(transition_covariance_m2)
            )
            covariance_scale_m2 = max(
                _maximum_abs_entry(M[i, j]),
                _maximum_abs_entry(transition_outer_moment_m2),
                np.finfo(float).tiny,
            )
            if np.min(covariance_eigenvalues_m2) < -PSD_TOL * covariance_scale_m2:
                raise ValueError(
                    f"M[{i},{j}] - d[{i},{j}] d[{i},{j}]^T must be "
                    "positive semidefinite"
                )
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
    state_memory_values = as_2d(
        x.state_memory_value_matrix,
        "state_memory_value_matrix",
    )
    if state_memory_values.shape != (n, A.shape[0]):
        raise ValueError(
            "state_memory_value_matrix must have shape (state_count, memory_count)"
        )
    positive_float(x.temperature_K, "temperature_K")
    positive_float(x.volume_m3, "volume_m3")
    return c, K, d, M, Dself, A, h, state_memory_values


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
