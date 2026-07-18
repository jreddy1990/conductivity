"""Trajectory samples projected into finite-generator conductivity primitives."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from constants import K_B
from conductivity.physical_library.projected_analytical_conductivity import (
    CARTESIAN,
    refine_mori_basis_by_projected_residual,
)
from utils.time_series_statistics import linear_fit

ANGSTROM_TO_M = 1.0e-10
DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M = 0.0
TOP_COMPONENT_EDGE_CONTRIBUTION_COUNT = 5  # Compact diagnostic table, not physics.
MINIMUM_DIFFUSIVE_WINDOW_LAG_COUNT = 4  # Two-parameter regression plus residual DOF.
NORMAL_CONFIDENCE_MULTIPLIER_95_PERCENT = (
    1.959963984540054  # Standard normal 95% interval.
)
DIFFUSION_FROM_SYMMETRIZED_COVARIANCE_SLOPE = 0.25  # D=(slope+slope.T)/4 from Cov=2Dt.

Array = np.ndarray


@dataclass(frozen=True)
class TrajectoryMarkovAdditiveSampleInput:
    state_labels: tuple[str, ...]
    occupancy_state_index_by_observation: Array
    from_state_index_by_step: Array
    to_state_index_by_step: Array
    charge_displacement_by_step_m: Array
    self_charge_polarization_by_frame_and_center_m: Array
    state_index_by_frame_and_center: Array
    self_current_valid_step_by_center: Array
    transition_commitment_time_s: float
    zero_frequency_integration_window_s: float
    zero_frequency_plateau_window_s: float
    dt_s: float
    total_transport_concentration_mol_m3: float
    temperature_K: float
    displacement_zero_tolerance_m: float = DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M


@dataclass(frozen=True)
class ProjectedGeneratorPrimitiveDiagnostics:
    original_state_count: int
    visited_state_count: int
    observation_count: int
    step_count: int
    transition_sample_count: int
    self_displacement_sample_count: int
    generated_event_count: int
    minimum_state_concentration_mol_m3: float
    maximum_state_concentration_mol_m3: float
    total_transport_concentration_mol_m3: float
    trajectory_time_s: float
    self_diffusion_convergence: tuple["StateDiffusionConvergence", ...]
    component_drift_residuals: tuple["FiniteProcessComponentDriftResidual", ...]
    finite_process_legality: "FiniteProcessLegalityDiagnostic"


@dataclass(frozen=True)
class FiniteProcessEdgeDriftContribution:
    component_id: int
    from_state_label: str
    to_state_label: str
    contribution_mol_m2_s: tuple[float, float, float]
    contribution_norm_mol_m2_s: float
    capacity_flux_mol_m3_s: float
    first_moment_norm_m: float
    forward_sample_count: int
    reverse_sample_count: int
    missing_reverse_event_candidate: bool


@dataclass(frozen=True)
class FiniteProcessComponentDriftResidual:
    component_id: int
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: tuple[float, ...]
    exit_rates_s_inv: tuple[float, ...]
    concentration_sum_mol_m3: float
    weighted_drift_mol_m2_s: tuple[float, float, float]
    weighted_drift_norm_mol_m2_s: float
    weighted_absolute_drift_scale_mol_m2_s: float
    top_edge_contributions: tuple[FiniteProcessEdgeDriftContribution, ...]


@dataclass(frozen=True)
class FiniteProcessLegalityDiagnostic:
    state_labels: tuple[str, ...]
    maximum_detailed_balance_residual_mol_m3_s: float
    component_drift_residuals: tuple[FiniteProcessComponentDriftResidual, ...]


@dataclass(frozen=True)
class ProjectedGeneratorReactiveFlux:
    from_state_label: str
    to_state_label: str
    symmetric_flux_mol_m3_s: float
    forward_rate_s_inv: float
    reverse_rate_s_inv: float
    forward_sample_count: int
    reverse_sample_count: int


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
class StateDiffusionConvergence:
    state_label: str
    convergence_status: str
    not_complete_reason: str
    lag_start_frames: int
    lag_stop_frames: int
    lag_count: int
    minimum_samples_per_lag: int
    maximum_samples_per_lag: int
    trace_slope_m2_s: float
    trace_slope_standard_error_m2_s: float
    log_log_exponent: float
    log_log_exponent_standard_error: float


@dataclass(frozen=True)
class ProjectedGeneratorPrimitiveSet:
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: Mapping[str, float]
    state_occupancy_fractions: Mapping[str, float]
    reactive_fluxes: tuple[ProjectedGeneratorReactiveFlux, ...]
    conditional_displacement_moments: tuple[ProjectedGeneratorConditionalMoment, ...]
    self_current_tensors: tuple[ProjectedGeneratorSelfCurrentTensor, ...]
    diagnostics: ProjectedGeneratorPrimitiveDiagnostics


@dataclass(frozen=True)
class CommittedTransitionEvent:
    source_state_index: int
    destination_state_index: int
    source_endpoint_frame: int
    destination_commitment_frame: int
    center_index: int
    charge_displacement_m: tuple[float, float, float]


@dataclass(frozen=True)
class TrajectoryBasisRefinement:
    candidate_labels: tuple[str, ...]
    candidate_sample_count: int
    selected_candidate_indices: tuple[int, ...]
    conductivity_history_S_m: tuple[float, ...]
    selected_residual_score_history_m2_s: tuple[float, ...]
    candidate_set_exhausted: bool
    convergence_status: str
    not_complete_reasons: tuple[str, ...]
    final_maximum_residual_score_m2_s: float
    final_conductivity_change_abs_S_m: float
    residual_score_tolerance_m2_s: float
    conductivity_change_tolerance_S_m: float
    final_mori_memory_matrix_A: Array
    final_mori_current_coupling_matrix_h: Array


@dataclass(frozen=True)
class MolecularFamilyMemoryKernelEstimate:
    replica_count: int
    molecule_ids: tuple[int, ...]
    molecular_family_labels: tuple[str, ...]
    family_labels: tuple[str, ...]
    molecular_masses_kg: Array
    family_masses_kg: Array
    molecular_com_velocities_m_s: Array
    molecular_total_forces_N: Array
    family_com_velocities_m_s: Array
    family_total_forces_N: Array
    correlation_lag_times_s: Array
    memory_kernel_lag_times_s: Array
    velocity_autocorrelation_m2_s2: Array
    force_velocity_correlation_N_m_s: Array
    memory_kernel_kg_s2: Array
    cumulative_memory_integral_kg_s: Array
    integrated_memory_kernel_kg_s: Array
    diffusion_tensor_m2_s: Array
    markov_diffusion_available: bool
    plateau_relative_variation: float
    tail_relative_norm: float
    minimum_plateau_eigenvalue_kg_s: float
    plateau_gate_passed: bool
    tail_gate_passed: bool
    psd_gate_passed: bool
    not_complete_reasons: tuple[str, ...]


def estimate_molecular_family_memory_kernel(
    atomic_velocities_m_s: Array,
    atomic_forces_N: Array,
    atomic_masses_kg: Array,
    atom_molecule_ids: Array,
    molecular_family_labels: tuple[str, ...],
    temperature_K: float,
    timestep_s: float,
    maximum_lag_frames: int,
    plateau_window_frames: int,
    maximum_plateau_relative_variation: float,
    maximum_tail_relative_norm: float,
    psd_relative_tolerance: float,
    volterra_regularization: float,
    singular_value_relative_tolerance: float,
) -> MolecularFamilyMemoryKernelEstimate:
    """Estimate a molecular-family GLE kernel from force-complete trajectories."""

    velocities_m_s, forces_N, masses_kg, molecule_ids = (
        _validated_force_complete_atomic_trajectory(
            atomic_velocities_m_s,
            atomic_forces_N,
            atomic_masses_kg,
            atom_molecule_ids,
        )
    )
    physical_temperature_K = _positive_float(temperature_K, "temperature_K")
    physical_timestep_s = _positive_float(timestep_s, "timestep_s")
    validated_maximum_lag_frames = _memory_kernel_lag_count(
        maximum_lag_frames,
        velocities_m_s.shape[0],
    )
    validated_plateau_window_frames = _memory_kernel_plateau_count(
        plateau_window_frames,
        validated_maximum_lag_frames,
    )
    plateau_tolerance = _nonnegative_float(
        maximum_plateau_relative_variation,
        "maximum_plateau_relative_variation",
    )
    tail_tolerance = _nonnegative_float(
        maximum_tail_relative_norm,
        "maximum_tail_relative_norm",
    )
    psd_tolerance = _nonnegative_float(
        psd_relative_tolerance,
        "psd_relative_tolerance",
    )
    regularization = _nonnegative_float(
        volterra_regularization,
        "volterra_regularization",
    )
    singular_value_tolerance = _positive_float(
        singular_value_relative_tolerance,
        "singular_value_relative_tolerance",
    )
    if singular_value_tolerance >= 1.0:
        raise ValueError("singular_value_relative_tolerance must be less than one")

    ordered_molecule_ids = tuple(int(value) for value in np.unique(molecule_ids))
    family_labels_by_molecule = _validated_molecular_family_labels(
        molecular_family_labels,
        len(ordered_molecule_ids),
    )
    (
        molecular_masses_kg,
        molecular_com_velocities_m_s,
        molecular_total_forces_N,
    ) = _project_atoms_to_molecules(
        velocities_m_s,
        forces_N,
        masses_kg,
        molecule_ids,
        ordered_molecule_ids,
    )
    family_labels = tuple(dict.fromkeys(family_labels_by_molecule))
    (
        family_masses_kg,
        family_com_velocities_m_s,
        family_total_forces_N,
    ) = _project_molecules_to_families(
        molecular_masses_kg,
        molecular_com_velocities_m_s,
        molecular_total_forces_N,
        family_labels_by_molecule,
        family_labels,
    )

    centered_family_velocities_m_s = family_com_velocities_m_s - np.mean(
        family_com_velocities_m_s,
        axis=0,
        keepdims=True,
    )
    centered_family_forces_N = family_total_forces_N - np.mean(
        family_total_forces_N,
        axis=0,
        keepdims=True,
    )
    flattened_family_velocities_m_s = centered_family_velocities_m_s.reshape(
        centered_family_velocities_m_s.shape[0],
        -1,
    )
    flattened_family_forces_N = centered_family_forces_N.reshape(
        centered_family_forces_N.shape[0],
        -1,
    )
    coordinate_masses_kg = np.repeat(family_masses_kg, CARTESIAN)
    flattened_family_momenta_kg_m_s = (
        flattened_family_velocities_m_s * coordinate_masses_kg[np.newaxis, :]
    )
    velocity_autocorrelation_m2_s2 = _lagged_matrix_correlation(
        flattened_family_velocities_m_s,
        flattened_family_velocities_m_s,
        validated_maximum_lag_frames,
    )
    momentum_velocity_correlation_kg_m2_s2 = _lagged_matrix_correlation(
        flattened_family_momenta_kg_m_s,
        flattened_family_velocities_m_s,
        validated_maximum_lag_frames,
    )
    force_velocity_correlation_N_m_s = _lagged_matrix_correlation(
        flattened_family_forces_N,
        flattened_family_velocities_m_s,
        validated_maximum_lag_frames,
    )
    momentum_correlation_derivative_N_m_s = np.diff(
        momentum_velocity_correlation_kg_m2_s2,
        axis=0,
    ) / physical_timestep_s
    midpoint_force_velocity_correlation_N_m_s = 0.5 * (
        force_velocity_correlation_N_m_s[:-1]
        + force_velocity_correlation_N_m_s[1:]
    )
    volterra_convolution_residual_N_m_s = (
        midpoint_force_velocity_correlation_N_m_s
        - momentum_correlation_derivative_N_m_s
    )
    memory_kernel_kg_s2 = _invert_matrix_volterra_kernel(
        velocity_autocorrelation_m2_s2,
        volterra_convolution_residual_N_m_s,
        physical_timestep_s,
        regularization,
        singular_value_tolerance,
    )
    cumulative_memory_integral_kg_s = np.cumsum(
        memory_kernel_kg_s2 * physical_timestep_s,
        axis=0,
    )
    integrated_memory_kernel_kg_s = cumulative_memory_integral_kg_s[-1]
    (
        plateau_relative_variation,
        tail_relative_norm,
        minimum_plateau_eigenvalue_kg_s,
        plateau_gate_passed,
        tail_gate_passed,
        psd_gate_passed,
    ) = _memory_kernel_gate_diagnostics(
        memory_kernel_kg_s2,
        cumulative_memory_integral_kg_s,
        validated_plateau_window_frames,
        plateau_tolerance,
        tail_tolerance,
        psd_tolerance,
    )
    not_complete_reasons: list[str] = []
    if not plateau_gate_passed:
        not_complete_reasons.append("stable memory-integral plateau")
    if not tail_gate_passed:
        not_complete_reasons.append("decayed memory-kernel tail")
    if not psd_gate_passed:
        not_complete_reasons.append("positive semidefinite")
    markov_diffusion_available = not not_complete_reasons
    diffusion_tensor_m2_s = np.empty((0, 0), dtype=float)
    if markov_diffusion_available:
        diffusion_tensor_m2_s = (
            K_B
            * physical_temperature_K
            * _psd_pseudoinverse_with_relative_tolerance(
                integrated_memory_kernel_kg_s,
                psd_tolerance,
                singular_value_tolerance,
            )
        )

    return MolecularFamilyMemoryKernelEstimate(
        replica_count=1,
        molecule_ids=ordered_molecule_ids,
        molecular_family_labels=family_labels_by_molecule,
        family_labels=family_labels,
        molecular_masses_kg=molecular_masses_kg,
        family_masses_kg=family_masses_kg,
        molecular_com_velocities_m_s=molecular_com_velocities_m_s,
        molecular_total_forces_N=molecular_total_forces_N,
        family_com_velocities_m_s=family_com_velocities_m_s,
        family_total_forces_N=family_total_forces_N,
        correlation_lag_times_s=(
            np.arange(validated_maximum_lag_frames + 1, dtype=float)
            * physical_timestep_s
        ),
        memory_kernel_lag_times_s=(
            (np.arange(validated_maximum_lag_frames, dtype=float) + 0.5)
            * physical_timestep_s
        ),
        velocity_autocorrelation_m2_s2=velocity_autocorrelation_m2_s2,
        force_velocity_correlation_N_m_s=force_velocity_correlation_N_m_s,
        memory_kernel_kg_s2=memory_kernel_kg_s2,
        cumulative_memory_integral_kg_s=cumulative_memory_integral_kg_s,
        integrated_memory_kernel_kg_s=integrated_memory_kernel_kg_s,
        diffusion_tensor_m2_s=diffusion_tensor_m2_s,
        markov_diffusion_available=markov_diffusion_available,
        plateau_relative_variation=plateau_relative_variation,
        tail_relative_norm=tail_relative_norm,
        minimum_plateau_eigenvalue_kg_s=minimum_plateau_eigenvalue_kg_s,
        plateau_gate_passed=plateau_gate_passed,
        tail_gate_passed=tail_gate_passed,
        psd_gate_passed=psd_gate_passed,
        not_complete_reasons=tuple(not_complete_reasons),
    )


def estimate_molecular_family_memory_kernel_from_replicas(
    atomic_velocity_replicas_m_s: tuple[Array, ...],
    atomic_force_replicas_N: tuple[Array, ...],
    atomic_masses_kg: Array,
    atom_molecule_ids: Array,
    molecular_family_labels: tuple[str, ...],
    temperature_K: float,
    timestep_s: float,
    maximum_lag_frames: int,
    plateau_window_frames: int,
    maximum_plateau_relative_variation: float,
    maximum_tail_relative_norm: float,
    psd_relative_tolerance: float,
    volterra_regularization: float,
    singular_value_relative_tolerance: float,
) -> MolecularFamilyMemoryKernelEstimate:
    """Invert one kernel from independently averaged replica correlations."""

    if not atomic_velocity_replicas_m_s:
        raise ValueError("at least one velocity replica is required")
    if len(atomic_velocity_replicas_m_s) != len(atomic_force_replicas_N):
        raise ValueError("velocity and force replica counts must match")
    replica_estimates = tuple(
        estimate_molecular_family_memory_kernel(
            atomic_velocities_m_s=velocity_replica_m_s,
            atomic_forces_N=force_replica_N,
            atomic_masses_kg=atomic_masses_kg,
            atom_molecule_ids=atom_molecule_ids,
            molecular_family_labels=molecular_family_labels,
            temperature_K=temperature_K,
            timestep_s=timestep_s,
            maximum_lag_frames=maximum_lag_frames,
            plateau_window_frames=plateau_window_frames,
            maximum_plateau_relative_variation=maximum_plateau_relative_variation,
            maximum_tail_relative_norm=maximum_tail_relative_norm,
            psd_relative_tolerance=psd_relative_tolerance,
            volterra_regularization=volterra_regularization,
            singular_value_relative_tolerance=singular_value_relative_tolerance,
        )
        for velocity_replica_m_s, force_replica_N in zip(
            atomic_velocity_replicas_m_s,
            atomic_force_replicas_N,
            strict=True,
        )
    )
    reference_estimate = replica_estimates[0]
    for replica_estimate in replica_estimates[1:]:
        if replica_estimate.molecule_ids != reference_estimate.molecule_ids:
            raise ValueError("replica molecule identities must match")
        if replica_estimate.family_labels != reference_estimate.family_labels:
            raise ValueError("replica molecular families must match")
        if not np.array_equal(
            replica_estimate.molecular_masses_kg,
            reference_estimate.molecular_masses_kg,
        ):
            raise ValueError("replica molecular masses must match")

    velocity_autocorrelation_m2_s2 = np.mean(
        np.stack(
            tuple(
                replica_estimate.velocity_autocorrelation_m2_s2
                for replica_estimate in replica_estimates
            ),
            axis=0,
        ),
        axis=0,
    )
    force_velocity_correlation_N_m_s = np.mean(
        np.stack(
            tuple(
                replica_estimate.force_velocity_correlation_N_m_s
                for replica_estimate in replica_estimates
            ),
            axis=0,
        ),
        axis=0,
    )
    coordinate_masses_kg = np.repeat(reference_estimate.family_masses_kg, CARTESIAN)
    momentum_velocity_correlation_kg_m2_s2 = (
        velocity_autocorrelation_m2_s2 * coordinate_masses_kg[np.newaxis, :, np.newaxis]
    )
    physical_timestep_s = _positive_float(timestep_s, "timestep_s")
    momentum_correlation_derivative_N_m_s = np.diff(
        momentum_velocity_correlation_kg_m2_s2,
        axis=0,
    ) / physical_timestep_s
    midpoint_force_velocity_correlation_N_m_s = 0.5 * (
        force_velocity_correlation_N_m_s[:-1]
        + force_velocity_correlation_N_m_s[1:]
    )
    memory_kernel_kg_s2 = _invert_matrix_volterra_kernel(
        velocity_autocorrelation_m2_s2,
        midpoint_force_velocity_correlation_N_m_s
        - momentum_correlation_derivative_N_m_s,
        physical_timestep_s,
        _nonnegative_float(volterra_regularization, "volterra_regularization"),
        _positive_float(
            singular_value_relative_tolerance,
            "singular_value_relative_tolerance",
        ),
    )
    cumulative_memory_integral_kg_s = np.cumsum(
        memory_kernel_kg_s2 * physical_timestep_s,
        axis=0,
    )
    integrated_memory_kernel_kg_s = cumulative_memory_integral_kg_s[-1]
    plateau_tolerance = _nonnegative_float(
        maximum_plateau_relative_variation,
        "maximum_plateau_relative_variation",
    )
    tail_tolerance = _nonnegative_float(
        maximum_tail_relative_norm,
        "maximum_tail_relative_norm",
    )
    psd_tolerance = _nonnegative_float(
        psd_relative_tolerance,
        "psd_relative_tolerance",
    )
    (
        plateau_relative_variation,
        tail_relative_norm,
        minimum_plateau_eigenvalue_kg_s,
        plateau_gate_passed,
        tail_gate_passed,
        psd_gate_passed,
    ) = _memory_kernel_gate_diagnostics(
        memory_kernel_kg_s2,
        cumulative_memory_integral_kg_s,
        _memory_kernel_plateau_count(plateau_window_frames, maximum_lag_frames),
        plateau_tolerance,
        tail_tolerance,
        psd_tolerance,
    )
    not_complete_reasons = tuple(
        reason
        for passed, reason in (
            (plateau_gate_passed, "stable memory-integral plateau"),
            (tail_gate_passed, "decayed memory-kernel tail"),
            (psd_gate_passed, "positive semidefinite"),
        )
        if not passed
    )
    markov_diffusion_available = not not_complete_reasons
    diffusion_tensor_m2_s = np.empty((0, 0), dtype=float)
    if markov_diffusion_available:
        diffusion_tensor_m2_s = (
            K_B
            * _positive_float(temperature_K, "temperature_K")
            * _psd_pseudoinverse_with_relative_tolerance(
                integrated_memory_kernel_kg_s,
                psd_tolerance,
                _positive_float(
                    singular_value_relative_tolerance,
                    "singular_value_relative_tolerance",
                ),
            )
        )

    return MolecularFamilyMemoryKernelEstimate(
        replica_count=len(replica_estimates),
        molecule_ids=reference_estimate.molecule_ids,
        molecular_family_labels=reference_estimate.molecular_family_labels,
        family_labels=reference_estimate.family_labels,
        molecular_masses_kg=reference_estimate.molecular_masses_kg,
        family_masses_kg=reference_estimate.family_masses_kg,
        molecular_com_velocities_m_s=np.concatenate(
            tuple(
                replica_estimate.molecular_com_velocities_m_s
                for replica_estimate in replica_estimates
            ),
            axis=0,
        ),
        molecular_total_forces_N=np.concatenate(
            tuple(
                replica_estimate.molecular_total_forces_N
                for replica_estimate in replica_estimates
            ),
            axis=0,
        ),
        family_com_velocities_m_s=np.concatenate(
            tuple(
                replica_estimate.family_com_velocities_m_s
                for replica_estimate in replica_estimates
            ),
            axis=0,
        ),
        family_total_forces_N=np.concatenate(
            tuple(
                replica_estimate.family_total_forces_N
                for replica_estimate in replica_estimates
            ),
            axis=0,
        ),
        correlation_lag_times_s=reference_estimate.correlation_lag_times_s,
        memory_kernel_lag_times_s=reference_estimate.memory_kernel_lag_times_s,
        velocity_autocorrelation_m2_s2=velocity_autocorrelation_m2_s2,
        force_velocity_correlation_N_m_s=force_velocity_correlation_N_m_s,
        memory_kernel_kg_s2=memory_kernel_kg_s2,
        cumulative_memory_integral_kg_s=cumulative_memory_integral_kg_s,
        integrated_memory_kernel_kg_s=integrated_memory_kernel_kg_s,
        diffusion_tensor_m2_s=diffusion_tensor_m2_s,
        markov_diffusion_available=markov_diffusion_available,
        plateau_relative_variation=plateau_relative_variation,
        tail_relative_norm=tail_relative_norm,
        minimum_plateau_eigenvalue_kg_s=minimum_plateau_eigenvalue_kg_s,
        plateau_gate_passed=plateau_gate_passed,
        tail_gate_passed=tail_gate_passed,
        psd_gate_passed=psd_gate_passed,
        not_complete_reasons=not_complete_reasons,
    )


def _validated_force_complete_atomic_trajectory(
    atomic_velocities_m_s: Array,
    atomic_forces_N: Array,
    atomic_masses_kg: Array,
    atom_molecule_ids: Array,
) -> tuple[Array, Array, Array, Array]:
    velocities_m_s = np.asarray(atomic_velocities_m_s, dtype=float)
    forces_N = np.asarray(atomic_forces_N, dtype=float)
    masses_kg = np.asarray(atomic_masses_kg, dtype=float)
    unparsed_molecule_ids = np.asarray(atom_molecule_ids)
    molecule_ids = np.asarray(atom_molecule_ids, dtype=int)
    if velocities_m_s.ndim != 3 or velocities_m_s.shape[2] != CARTESIAN:
        raise ValueError("atomic_velocities_m_s must have shape (frames, atoms, 3)")
    if forces_N.shape != velocities_m_s.shape:
        raise ValueError("atomic_forces_N must match atomic_velocities_m_s")
    atom_count = velocities_m_s.shape[1]
    if masses_kg.shape != (atom_count,):
        raise ValueError("atomic_masses_kg must have one value per atom")
    if molecule_ids.shape != (atom_count,):
        raise ValueError("atom_molecule_ids must have one value per atom")
    if not np.array_equal(unparsed_molecule_ids, molecule_ids):
        raise ValueError("atom_molecule_ids must contain exact integers")
    if velocities_m_s.shape[0] < 3:
        raise ValueError("memory-kernel estimation requires at least three frames")
    if not np.all(np.isfinite(velocities_m_s)):
        raise ValueError("atomic_velocities_m_s must be finite")
    if not np.all(np.isfinite(forces_N)):
        raise ValueError("atomic_forces_N must be finite")
    if not np.all(np.isfinite(masses_kg)) or np.any(masses_kg <= 0.0):
        raise ValueError("atomic_masses_kg must be finite and positive")
    return velocities_m_s, forces_N, masses_kg, molecule_ids


def _memory_kernel_lag_count(maximum_lag_frames: int, frame_count: int) -> int:
    if isinstance(maximum_lag_frames, (bool, np.bool_)) or not isinstance(
        maximum_lag_frames,
        (int, np.integer),
    ):
        raise ValueError("maximum_lag_frames must be an integer")
    lag_count = int(maximum_lag_frames)
    if lag_count < 2 or lag_count >= frame_count:
        raise ValueError(
            "maximum_lag_frames must be at least two and less than frame count"
        )
    return lag_count


def _memory_kernel_plateau_count(
    plateau_window_frames: int,
    maximum_lag_frames: int,
) -> int:
    if isinstance(plateau_window_frames, (bool, np.bool_)) or not isinstance(
        plateau_window_frames,
        (int, np.integer),
    ):
        raise ValueError("plateau_window_frames must be an integer")
    plateau_count = int(plateau_window_frames)
    if plateau_count < 2 or plateau_count > maximum_lag_frames:
        raise ValueError(
            "plateau_window_frames must be at least two and no greater than maximum lag"
        )
    return plateau_count


def _validated_molecular_family_labels(
    molecular_family_labels: tuple[str, ...],
    molecule_count: int,
) -> tuple[str, ...]:
    if len(molecular_family_labels) != molecule_count:
        raise ValueError(
            "molecular_family_labels must align with sorted unique molecule IDs"
        )
    if any(not isinstance(label, str) or not label for label in molecular_family_labels):
        raise ValueError("molecular_family_labels must contain non-empty strings")
    return tuple(molecular_family_labels)


def _project_atoms_to_molecules(
    atomic_velocities_m_s: Array,
    atomic_forces_N: Array,
    atomic_masses_kg: Array,
    atom_molecule_ids: Array,
    ordered_molecule_ids: tuple[int, ...],
) -> tuple[Array, Array, Array]:
    molecular_masses_kg = np.asarray(
        tuple(
            float(np.sum(atomic_masses_kg[atom_molecule_ids == molecule_id]))
            for molecule_id in ordered_molecule_ids
        ),
        dtype=float,
    )
    molecular_com_velocities_m_s = np.stack(
        tuple(
            np.einsum(
                "a,fac->fc",
                atomic_masses_kg[atom_molecule_ids == molecule_id],
                atomic_velocities_m_s[:, atom_molecule_ids == molecule_id, :],
            )
            / molecular_masses_kg[molecule_index]
            for molecule_index, molecule_id in enumerate(ordered_molecule_ids)
        ),
        axis=1,
    )
    molecular_total_forces_N = np.stack(
        tuple(
            np.sum(
                atomic_forces_N[:, atom_molecule_ids == molecule_id, :],
                axis=1,
            )
            for molecule_id in ordered_molecule_ids
        ),
        axis=1,
    )
    return (
        molecular_masses_kg,
        molecular_com_velocities_m_s,
        molecular_total_forces_N,
    )


def _project_molecules_to_families(
    molecular_masses_kg: Array,
    molecular_com_velocities_m_s: Array,
    molecular_total_forces_N: Array,
    molecular_family_labels: tuple[str, ...],
    family_labels: tuple[str, ...],
) -> tuple[Array, Array, Array]:
    family_masks = tuple(
        np.asarray(
            tuple(label == family_label for label in molecular_family_labels),
            dtype=bool,
        )
        for family_label in family_labels
    )
    family_masses_kg = np.asarray(
        tuple(float(np.sum(molecular_masses_kg[mask])) for mask in family_masks),
        dtype=float,
    )
    family_com_velocities_m_s = np.stack(
        tuple(
            np.einsum(
                "m,fmc->fc",
                molecular_masses_kg[mask],
                molecular_com_velocities_m_s[:, mask, :],
            )
            / family_masses_kg[family_index]
            for family_index, mask in enumerate(family_masks)
        ),
        axis=1,
    )
    family_total_forces_N = np.stack(
        tuple(
            np.sum(molecular_total_forces_N[:, mask, :], axis=1)
            for mask in family_masks
        ),
        axis=1,
    )
    return family_masses_kg, family_com_velocities_m_s, family_total_forces_N


def _lagged_matrix_correlation(
    future_values: Array,
    origin_values: Array,
    maximum_lag_frames: int,
) -> Array:
    return np.asarray(
        tuple(
            np.einsum(
                "ti,tj->ij",
                future_values[lag_frames:],
                origin_values[: future_values.shape[0] - lag_frames],
            )
            / float(future_values.shape[0] - lag_frames)
            for lag_frames in range(maximum_lag_frames + 1)
        ),
        dtype=float,
    )


def _invert_matrix_volterra_kernel(
    velocity_autocorrelation_m2_s2: Array,
    volterra_convolution_residual_N_m_s: Array,
    timestep_s: float,
    volterra_regularization: float,
    singular_value_relative_tolerance: float,
) -> Array:
    lag_count = velocity_autocorrelation_m2_s2.shape[0] - 1
    midpoint_velocity_correlations_m2_s2 = 0.5 * (
        velocity_autocorrelation_m2_s2[:-1]
        + velocity_autocorrelation_m2_s2[1:]
    )
    zero_lag_correlation_m2_s2 = midpoint_velocity_correlations_m2_s2[0]
    left_vectors, singular_values, right_vectors_transpose = np.linalg.svd(
        zero_lag_correlation_m2_s2,
        full_matrices=False,
    )
    maximum_singular_value = float(np.max(singular_values))
    retained = singular_values > (
        singular_value_relative_tolerance * maximum_singular_value
    )
    inverse_weights = np.where(
        retained,
        singular_values
        / (
            singular_values * singular_values
            + volterra_regularization
            * maximum_singular_value
            * maximum_singular_value
        ),
        0.0,
    )
    zero_lag_pseudoinverse_s2_m2 = (
        right_vectors_transpose.T
        @ np.diag(inverse_weights)
        @ left_vectors.T
    )
    kernel_values: list[Array] = []
    for interval_index in range(lag_count):
        prior_convolution_N_m_s = sum(
            (
                kernel_values[kernel_index]
                @ midpoint_velocity_correlations_m2_s2[
                    interval_index - kernel_index
                ]
                * timestep_s
            )
            for kernel_index in range(interval_index)
        )
        remaining_convolution_N_m_s = (
            volterra_convolution_residual_N_m_s[interval_index]
            - prior_convolution_N_m_s
        )
        kernel_values.append(
            remaining_convolution_N_m_s
            @ zero_lag_pseudoinverse_s2_m2
            / timestep_s
        )
    unsymmetrized_kernel_kg_s2 = np.asarray(kernel_values, dtype=float)
    return 0.5 * (
        unsymmetrized_kernel_kg_s2
        + unsymmetrized_kernel_kg_s2.transpose(0, 2, 1)
    )


def _memory_kernel_gate_diagnostics(
    memory_kernel_kg_s2: Array,
    cumulative_memory_integral_kg_s: Array,
    plateau_window_frames: int,
    maximum_plateau_relative_variation: float,
    maximum_tail_relative_norm: float,
    psd_relative_tolerance: float,
) -> tuple[float, float, float, bool, bool, bool]:
    plateau_integrals_kg_s = cumulative_memory_integral_kg_s[-plateau_window_frames:]
    integrated_memory_kernel_kg_s = cumulative_memory_integral_kg_s[-1]
    integrated_scale_kg_s = max(
        float(np.linalg.norm(integrated_memory_kernel_kg_s, ord="fro")),
        np.finfo(float).tiny,
    )
    plateau_relative_variation = max(
        float(
            np.linalg.norm(
                plateau_value_kg_s - integrated_memory_kernel_kg_s,
                ord="fro",
            )
            / integrated_scale_kg_s
        )
        for plateau_value_kg_s in plateau_integrals_kg_s
    )
    kernel_norms_kg_s2 = np.linalg.norm(memory_kernel_kg_s2, axis=(1, 2))
    tail_relative_norm = float(
        np.max(kernel_norms_kg_s2[-plateau_window_frames:])
        / max(float(np.max(kernel_norms_kg_s2)), np.finfo(float).tiny)
    )
    plateau_eigenvalues_kg_s = tuple(
        np.linalg.eigvalsh(plateau_value_kg_s)
        for plateau_value_kg_s in plateau_integrals_kg_s
    )
    minimum_plateau_eigenvalue_kg_s = min(
        float(np.min(eigenvalues_kg_s))
        for eigenvalues_kg_s in plateau_eigenvalues_kg_s
    )
    psd_gate_passed = all(
        float(np.min(eigenvalues_kg_s))
        >= -psd_relative_tolerance
        * max(float(np.max(np.abs(eigenvalues_kg_s))), np.finfo(float).tiny)
        for eigenvalues_kg_s in plateau_eigenvalues_kg_s
    )
    return (
        plateau_relative_variation,
        tail_relative_norm,
        minimum_plateau_eigenvalue_kg_s,
        plateau_relative_variation <= maximum_plateau_relative_variation,
        tail_relative_norm <= maximum_tail_relative_norm,
        psd_gate_passed,
    )


def _psd_pseudoinverse_with_relative_tolerance(
    matrix: Array,
    psd_relative_tolerance: float,
    singular_value_relative_tolerance: float,
) -> Array:
    numeric_matrix = np.asarray(matrix, dtype=float)
    symmetric_matrix = 0.5 * (numeric_matrix + numeric_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    eigenvalue_scale = max(
        float(np.max(np.abs(eigenvalues))),
        np.finfo(float).tiny,
    )
    if float(np.min(eigenvalues)) < -psd_relative_tolerance * eigenvalue_scale:
        raise ValueError("integrated memory kernel must be positive semidefinite")
    retained_eigenvalues = eigenvalues > (
        singular_value_relative_tolerance * eigenvalue_scale
    )
    inverse_eigenvalues = np.where(
        retained_eigenvalues,
        1.0 / np.maximum(eigenvalues, np.finfo(float).tiny),
        0.0,
    )
    pseudoinverse = eigenvectors @ np.diag(inverse_eigenvalues) @ eigenvectors.T
    return 0.5 * (pseudoinverse + pseudoinverse.T)


def _extract_committed_transition_events(
    state_index_by_frame_and_center: Array,
    charge_polarization_by_frame_and_center_m: Array,
    timestep_s: float,
    commitment_time_s: float,
) -> tuple[CommittedTransitionEvent, ...]:
    state_indices = np.asarray(state_index_by_frame_and_center, dtype=int)
    polarizations_m = np.asarray(
        charge_polarization_by_frame_and_center_m,
        dtype=float,
    )
    if state_indices.ndim != 2:
        raise ValueError("state_index_by_frame_and_center must be two-dimensional")
    if polarizations_m.shape != (*state_indices.shape, CARTESIAN):
        raise ValueError("charge polarization must align with state frames and centers")
    physical_timestep_s = _positive_float(timestep_s, "timestep_s")
    physical_commitment_time_s = _positive_float(
        commitment_time_s,
        "commitment_time_s",
    )
    commitment_frame_count = int(
        np.ceil(physical_commitment_time_s / physical_timestep_s)
    )
    if commitment_frame_count >= state_indices.shape[0]:
        raise ValueError("commitment time is unresolved by the trajectory duration")

    events: list[CommittedTransitionEvent] = []
    for center_index in range(state_indices.shape[1]):
        center_states = state_indices[:, center_index]
        run_starts = np.concatenate(
            (np.asarray([0], dtype=int), np.flatnonzero(np.diff(center_states)) + 1)
        )
        run_stops = np.concatenate(
            (run_starts[1:], np.asarray([center_states.size], dtype=int))
        )
        committed_run_indices = np.flatnonzero(
            (run_stops - run_starts) >= commitment_frame_count
        )
        for committed_pair_index in range(committed_run_indices.size - 1):
            source_run_index = int(committed_run_indices[committed_pair_index])
            destination_run_index = int(
                committed_run_indices[committed_pair_index + 1]
            )
            source_state_index = int(center_states[run_starts[source_run_index]])
            destination_state_index = int(
                center_states[run_starts[destination_run_index]]
            )
            if source_state_index == destination_state_index:
                continue
            source_endpoint_frame = int(run_stops[source_run_index] - 1)
            destination_commitment_frame = int(
                run_starts[destination_run_index] + commitment_frame_count - 1
            )
            displacement_m = (
                polarizations_m[destination_commitment_frame, center_index]
                - polarizations_m[source_endpoint_frame, center_index]
            )
            events.append(
                CommittedTransitionEvent(
                    source_state_index=source_state_index,
                    destination_state_index=destination_state_index,
                    source_endpoint_frame=source_endpoint_frame,
                    destination_commitment_frame=destination_commitment_frame,
                    center_index=center_index,
                    charge_displacement_m=_vector_to_tuple(displacement_m),
                )
            )
    return tuple(events)


def refine_trajectory_basis_from_state_current_samples(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
    state_labels: tuple[str, ...],
    state_index_by_step: Array,
    samples_per_frame: int,
    direct_diffusivity_tensor_m2_s: Array,
    residual_score_tolerance_m2_s: float,
    conductivity_change_tolerance_S_m: float,
) -> TrajectoryBasisRefinement:
    """Refine lagged state-current coordinates from measured increments."""

    labels = _validated_state_labels(state_labels)
    state_indices = _state_indices(
        state_index_by_step, len(labels), "state_index_by_step"
    )
    displacements = _charge_displacements(
        sample_input.charge_displacement_by_step_m, state_indices.size
    )
    if samples_per_frame <= 0:
        raise ValueError("samples_per_frame must be positive")
    if displacements.shape[0] <= samples_per_frame:
        raise ValueError("trajectory basis refinement needs at least two step frames")
    timestep_s = _positive_float(sample_input.dt_s, "dt_s")
    candidate_labels = tuple(
        f"lagged_charge_displacement:{state_label}:{axis_label}"
        for state_label in labels
        for axis_label in ("x", "y", "z")
    )
    lagged_features_m = np.zeros(
        (displacements.shape[0] - samples_per_frame, len(candidate_labels)),
        dtype=float,
    )
    for sample_index, state_index in enumerate(state_indices[:-samples_per_frame]):
        feature_start = int(state_index) * CARTESIAN
        lagged_features_m[sample_index, feature_start : feature_start + CARTESIAN] = (
            displacements[sample_index]
        )
    centered_features_m = lagged_features_m - np.mean(
        lagged_features_m, axis=0, keepdims=True
    )
    next_displacements_m = displacements[samples_per_frame:]
    centered_next_displacements_m = next_displacements_m - np.mean(
        next_displacements_m, axis=0, keepdims=True
    )
    candidate_sample_count = centered_features_m.shape[0]
    candidate_memory_matrix = (
        centered_features_m.T
        @ centered_features_m
        / candidate_sample_count
        / timestep_s
    )
    candidate_memory_matrix = 0.5 * (
        candidate_memory_matrix + candidate_memory_matrix.T
    )
    candidate_current_coupling = (
        centered_features_m.T
        @ centered_next_displacements_m
        / candidate_sample_count
        / timestep_s
    )
    refinement_result = refine_mori_basis_by_projected_residual(
        direct_diffusivity_tensor=direct_diffusivity_tensor_m2_s,
        initial_mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        initial_mori_current_coupling_matrix_h=np.zeros((0, CARTESIAN), dtype=float),
        candidate_self_energies_A_gg=np.diag(candidate_memory_matrix),
        candidate_cross_energies_A_gPhi=np.zeros(
            (len(candidate_labels), 0), dtype=float
        ),
        candidate_cross_energy_matrix=candidate_memory_matrix,
        candidate_current_couplings_h_g=candidate_current_coupling,
        temperature_K=sample_input.temperature_K,
        residual_score_tolerance=residual_score_tolerance_m2_s,
        conductivity_change_tolerance_S_m=conductivity_change_tolerance_S_m,
        max_added_coordinates=len(candidate_labels),
        require_candidate_set_exhaustion=True,
    )
    return TrajectoryBasisRefinement(
        candidate_labels=candidate_labels,
        candidate_sample_count=candidate_sample_count,
        selected_candidate_indices=tuple(
            int(index) for index in refinement_result["selected_candidate_indices"]
        ),
        conductivity_history_S_m=tuple(
            float(value)
            for value in refinement_result["conductivity_history_S_m"]
        ),
        selected_residual_score_history_m2_s=tuple(
            float(value)
            for value in refinement_result["selected_residual_score_history"]
        ),
        candidate_set_exhausted=bool(refinement_result["candidate_set_exhausted"]),
        convergence_status=str(refinement_result["convergence_status"]),
        not_complete_reasons=tuple(refinement_result["not_complete_reasons"]),
        final_maximum_residual_score_m2_s=float(
            refinement_result["final_maximum_residual_score"]
        ),
        final_conductivity_change_abs_S_m=float(
            refinement_result["final_conductivity_change_abs_S_m"]
        ),
        residual_score_tolerance_m2_s=residual_score_tolerance_m2_s,
        conductivity_change_tolerance_S_m=conductivity_change_tolerance_S_m,
        final_mori_memory_matrix_A=np.asarray(
            refinement_result["final_mori_memory_matrix_A"], dtype=float
        ),
        final_mori_current_coupling_matrix_h=np.asarray(
            refinement_result["final_mori_current_coupling_matrix_h"], dtype=float
        ),
    )


def project_sampled_trajectory_to_generator_primitives(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
) -> ProjectedGeneratorPrimitiveSet:
    """Project observed state/displacement samples into primitive tensors."""

    state_labels = _validated_state_labels(sample_input.state_labels)
    occupancy_state_indices = _state_indices(
        sample_input.occupancy_state_index_by_observation,
        len(state_labels),
        "occupancy_state_index_by_observation",
    )
    from_state_indices = _state_indices(
        sample_input.from_state_index_by_step,
        len(state_labels),
        "from_state_index_by_step",
    )
    to_state_indices = _state_indices(
        sample_input.to_state_index_by_step,
        len(state_labels),
        "to_state_index_by_step",
    )
    if from_state_indices.size != to_state_indices.size:
        raise ValueError("from_state_index_by_step and to_state_index_by_step mismatch")
    charge_displacements = _charge_displacements(
        sample_input.charge_displacement_by_step_m,
        from_state_indices.size,
    )
    timestep_s = _positive_float(sample_input.dt_s, "dt_s")
    total_concentration_mol_m3 = _positive_float(
        sample_input.total_transport_concentration_mol_m3,
        "total_transport_concentration_mol_m3",
    )
    _positive_float(sample_input.temperature_K, "temperature_K")
    displacement_zero_tolerance_m = _nonnegative_float(
        sample_input.displacement_zero_tolerance_m,
        "displacement_zero_tolerance_m",
    )
    committed_events = _extract_committed_transition_events(
        state_index_by_frame_and_center=sample_input.state_index_by_frame_and_center,
        charge_polarization_by_frame_and_center_m=(
            sample_input.self_charge_polarization_by_frame_and_center_m
        ),
        timestep_s=timestep_s,
        commitment_time_s=sample_input.transition_commitment_time_s,
    )

    remap = _visited_state_remap(
        occupancy_state_indices,
        from_state_indices,
        to_state_indices,
    )
    remapped_labels = tuple(
        state_labels[original_index] for original_index in sorted(remap)
    )
    remapped_occupancy = np.asarray(
        [remap[int(index)] for index in occupancy_state_indices],
        dtype=int,
    )
    remapped_from = np.asarray(
        [remap[event.source_state_index] for event in committed_events],
        dtype=int,
    )
    remapped_to = np.asarray(
        [remap[event.destination_state_index] for event in committed_events],
        dtype=int,
    )
    committed_displacements = np.asarray(
        [event.charge_displacement_m for event in committed_events],
        dtype=float,
    ).reshape((-1, CARTESIAN))

    state_concentrations = _state_concentrations(
        remapped_labels,
        remapped_occupancy,
        total_concentration_mol_m3,
    )
    occupancy_fractions = _occupancy_fractions(remapped_labels, remapped_occupancy)
    reactive_fluxes = _reactive_fluxes(
        remapped_labels,
        remapped_from,
        remapped_to,
        state_concentrations,
        total_concentration_mol_m3,
        timestep_s,
        exposure_time_s=(
            (sample_input.state_index_by_frame_and_center.shape[0] - 1)
            * sample_input.state_index_by_frame_and_center.shape[1]
            * timestep_s
        ),
    )
    conditional_moments = _conditional_displacement_moments(
        remapped_labels,
        remapped_from,
        remapped_to,
        committed_displacements,
    )
    self_current_tensors, self_diffusion_convergence = _self_current_tensors(
        remapped_labels,
        np.asarray(sample_input.state_index_by_frame_and_center, dtype=int),
        np.asarray(
            sample_input.self_charge_polarization_by_frame_and_center_m, dtype=float
        ),
        state_concentrations,
        timestep_s,
        np.asarray(sample_input.self_current_valid_step_by_center, dtype=bool),
        sample_input.zero_frequency_integration_window_s,
        sample_input.zero_frequency_plateau_window_s,
    )
    component_drift_residuals = _component_drift_residuals_from_records(
        remapped_labels,
        state_concentrations,
        reactive_fluxes,
        conditional_moments,
    )
    finite_process_legality = _finite_process_legality_from_records(
        remapped_labels,
        state_concentrations,
        reactive_fluxes,
        conditional_moments,
    )
    diagnostics = ProjectedGeneratorPrimitiveDiagnostics(
        original_state_count=len(state_labels),
        visited_state_count=len(remapped_labels),
        observation_count=int(remapped_occupancy.size),
        step_count=int(from_state_indices.size),
        transition_sample_count=len(committed_events),
        self_displacement_sample_count=_self_displacement_sample_count(
            from_state_indices,
            to_state_indices,
            charge_displacements,
            displacement_zero_tolerance_m,
        ),
        generated_event_count=len(reactive_fluxes) + len(self_current_tensors),
        minimum_state_concentration_mol_m3=float(min(state_concentrations.values())),
        maximum_state_concentration_mol_m3=float(max(state_concentrations.values())),
        total_transport_concentration_mol_m3=total_concentration_mol_m3,
        trajectory_time_s=float(from_state_indices.size * timestep_s),
        self_diffusion_convergence=self_diffusion_convergence,
        component_drift_residuals=component_drift_residuals,
        finite_process_legality=finite_process_legality,
    )
    return ProjectedGeneratorPrimitiveSet(
        state_labels=remapped_labels,
        state_concentrations_mol_m3=state_concentrations,
        state_occupancy_fractions=occupancy_fractions,
        reactive_fluxes=reactive_fluxes,
        conditional_displacement_moments=conditional_moments,
        self_current_tensors=self_current_tensors,
        diagnostics=diagnostics,
    )


def _validated_state_labels(state_labels: tuple[str, ...]) -> tuple[str, ...]:
    labels = tuple(str(label) for label in state_labels)
    if not labels:
        raise ValueError("state_labels must not be empty")
    if len(set(labels)) != len(labels):
        raise ValueError("state_labels must be unique")
    return labels


def _state_indices(array: Array, state_count: int, label: str) -> Array:
    result = np.asarray(array, dtype=int)
    if result.ndim != 1:
        raise ValueError(f"{label} must be a 1D integer array")
    if np.any(result < 0) or np.any(result >= state_count):
        raise ValueError(f"{label} contains out-of-range state indices")
    return result


def _charge_displacements(array: Array, step_count: int) -> Array:
    result = np.asarray(array, dtype=float)
    if result.shape != (step_count, CARTESIAN) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"charge_displacement_by_step_m must have shape ({step_count}, 3)"
        )
    return result


def _visited_state_remap(
    occupancy_indices: Array,
    from_indices: Array,
    to_indices: Array,
) -> dict[int, int]:
    visited = sorted(
        set(int(index) for index in occupancy_indices)
        | set(int(index) for index in from_indices)
        | set(int(index) for index in to_indices)
    )
    if not visited:
        raise ValueError("no visited states found")
    return {
        original_index: remapped_index
        for remapped_index, original_index in enumerate(visited)
    }


def _state_concentrations(
    state_labels: tuple[str, ...],
    occupancy_indices: Array,
    total_concentration_mol_m3: float,
) -> dict[str, float]:
    counts = np.bincount(occupancy_indices, minlength=len(state_labels)).astype(float)
    count_sum = _positive_float(float(np.sum(counts)), "occupancy count sum")
    concentrations = counts / count_sum * total_concentration_mol_m3
    return {
        state_label: float(concentrations[state_index])
        for state_index, state_label in enumerate(state_labels)
    }


def _occupancy_fractions(
    state_labels: tuple[str, ...],
    occupancy_indices: Array,
) -> dict[str, float]:
    counts = np.bincount(occupancy_indices, minlength=len(state_labels)).astype(float)
    count_sum = _positive_float(float(np.sum(counts)), "occupancy count sum")
    fractions = counts / count_sum
    return {
        state_label: float(fractions[state_index])
        for state_index, state_label in enumerate(state_labels)
    }


def _reactive_fluxes(
    state_labels: tuple[str, ...],
    from_indices: Array,
    to_indices: Array,
    state_concentrations: Mapping[str, float],
    total_concentration_mol_m3: float,
    timestep_s: float,
    exposure_time_s: float,
) -> tuple[ProjectedGeneratorReactiveFlux, ...]:
    directed_counts: dict[tuple[int, int], int] = defaultdict(int)
    for sample_index, from_state_index in enumerate(from_indices):
        to_state_index = int(to_indices[sample_index])
        if int(from_state_index) == to_state_index:
            continue
        directed_counts[(int(from_state_index), to_state_index)] += 1
    _positive_float(timestep_s, "timestep_s")
    physical_exposure_time_s = _positive_float(exposure_time_s, "exposure_time_s")
    event_flux_per_sample_mol_m3_s = total_concentration_mol_m3 / (
        2.0 * physical_exposure_time_s
    )
    unordered_pairs = sorted(
        {
            (min(first_state, second_state), max(first_state, second_state))
            for first_state, second_state in directed_counts
        }
    )
    records: list[ProjectedGeneratorReactiveFlux] = []
    for lower_state_index, upper_state_index in unordered_pairs:
        forward_count = directed_counts[(lower_state_index, upper_state_index)]
        reverse_count = directed_counts[(upper_state_index, lower_state_index)]
        symmetric_flux = (
            0.5 * float(forward_count + reverse_count) * event_flux_per_sample_mol_m3_s
        )
        lower_label = state_labels[lower_state_index]
        upper_label = state_labels[upper_state_index]
        lower_concentration = _positive_float(
            state_concentrations[lower_label],
            f"state_concentration[{lower_label}]",
        )
        upper_concentration = _positive_float(
            state_concentrations[upper_label],
            f"state_concentration[{upper_label}]",
        )
        records.append(
            ProjectedGeneratorReactiveFlux(
                from_state_label=lower_label,
                to_state_label=upper_label,
                symmetric_flux_mol_m3_s=symmetric_flux,
                forward_rate_s_inv=symmetric_flux / lower_concentration,
                reverse_rate_s_inv=symmetric_flux / upper_concentration,
                forward_sample_count=int(forward_count),
                reverse_sample_count=int(reverse_count),
            )
        )
    return tuple(records)


def _conditional_displacement_moments(
    state_labels: tuple[str, ...],
    from_indices: Array,
    to_indices: Array,
    charge_displacements: Array,
) -> tuple[ProjectedGeneratorConditionalMoment, ...]:
    samples_by_transition: dict[tuple[int, int], list[Array]] = defaultdict(list)
    for sample_index, from_state_index in enumerate(from_indices):
        to_state_index = int(to_indices[sample_index])
        if int(from_state_index) == to_state_index:
            continue
        lower_state_index = min(int(from_state_index), to_state_index)
        upper_state_index = max(int(from_state_index), to_state_index)
        oriented_displacement = np.asarray(
            charge_displacements[sample_index], dtype=float
        )
        if int(from_state_index) == upper_state_index:
            oriented_displacement = -oriented_displacement
        samples_by_transition[(lower_state_index, upper_state_index)].append(
            oriented_displacement,
        )
    records: list[ProjectedGeneratorConditionalMoment] = []
    for transition_key in sorted(samples_by_transition):
        samples = np.asarray(samples_by_transition[transition_key], dtype=float)
        mean_displacement = np.mean(samples, axis=0)
        second_moment = np.einsum("sa,sb->ab", samples, samples) / float(
            samples.shape[0]
        )
        covariance = second_moment - np.outer(mean_displacement, mean_displacement)
        _validate_psd(covariance, "conditional displacement covariance")
        from_state_index, to_state_index = transition_key
        records.append(
            ProjectedGeneratorConditionalMoment(
                from_state_label=state_labels[from_state_index],
                to_state_label=state_labels[to_state_index],
                sample_count=int(samples.shape[0]),
                mean_charge_displacement_m=_vector_to_tuple(mean_displacement),
                second_moment_m2=_matrix_to_tuple(second_moment),
                covariance_m2=_matrix_to_tuple(covariance),
            )
        )
    return tuple(records)


def _self_current_tensors(
    state_labels: tuple[str, ...],
    state_index_by_frame_and_center: Array,
    charge_polarization_by_frame_and_center_m: Array,
    state_concentrations: Mapping[str, float],
    timestep_s: float,
    self_current_valid_step_by_center: Array,
    integration_window_s: float,
    plateau_window_s: float,
) -> tuple[
    tuple[ProjectedGeneratorSelfCurrentTensor, ...],
    tuple[StateDiffusionConvergence, ...],
]:
    state_indices = np.asarray(state_index_by_frame_and_center, dtype=int)
    polarizations_m = np.asarray(charge_polarization_by_frame_and_center_m, dtype=float)
    if state_indices.ndim != 2:
        raise ValueError(
            "state_index_by_frame_and_center must have shape (frames, centers)"
        )
    expected_shape = (*state_indices.shape, CARTESIAN)
    if polarizations_m.shape != expected_shape or not np.all(
        np.isfinite(polarizations_m)
    ):
        raise ValueError(
            "self_charge_polarization_by_frame_and_center_m must have shape "
            f"{expected_shape} and contain finite values"
        )
    if state_indices.shape[0] < 2:
        raise ValueError("self-current estimation needs at least two trajectory frames")
    valid_step_mask = np.asarray(self_current_valid_step_by_center, dtype=bool)
    if valid_step_mask.shape != (state_indices.shape[0] - 1, state_indices.shape[1]):
        raise ValueError(
            "self_current_valid_step_by_center must have shape (frames-1, centers)"
        )
    integration_window_s = _positive_float(
        integration_window_s, "zero_frequency_integration_window_s"
    )
    maximum_integration_lag_frames = min(
        state_indices.shape[0] - 1,
        int(np.floor(integration_window_s / timestep_s)),
    )
    records: list[ProjectedGeneratorSelfCurrentTensor] = []
    diagnostics: list[StateDiffusionConvergence] = []
    for state_index, state_label in enumerate(state_labels):
        covariance_by_lag_m2, sample_count_by_lag, populated_lags = (
            _state_conditioned_covariances_by_lag(
                state_indices=state_indices,
                polarizations_m=polarizations_m,
                valid_step_mask=valid_step_mask,
                target_state_index=state_index,
                maximum_lag_frames=maximum_integration_lag_frames,
            )
        )
        convergence, diffusion_tensor, converged = _find_diffusive_covariance_window(
            state_label=state_label,
            covariance_by_lag_m2=covariance_by_lag_m2,
            sample_count_by_lag=sample_count_by_lag,
            populated_lags=populated_lags,
            timestep_s=timestep_s,
            integration_window_s=integration_window_s,
            plateau_window_s=plateau_window_s,
        )
        diagnostics.append(convergence)
        if not converged:
            continue
        _validate_psd(diffusion_tensor, f"within-state diffusion {state_label}")
        records.append(
            ProjectedGeneratorSelfCurrentTensor(
                state_label=state_label,
                sample_count=convergence.minimum_samples_per_lag,
                concentration_mol_m3=float(state_concentrations[state_label]),
                diffusion_tensor_m2_s=_matrix_to_tuple(diffusion_tensor),
            )
        )
    return tuple(records), tuple(diagnostics)


def _state_conditioned_covariances_by_lag(
    state_indices: Array,
    polarizations_m: Array,
    valid_step_mask: Array,
    target_state_index: int,
    maximum_lag_frames: int,
) -> tuple[Array, Array, Array]:
    frame_count, center_count = state_indices.shape
    covariance_by_lag_m2 = np.zeros(
        (frame_count - 1, CARTESIAN, CARTESIAN), dtype=float
    )
    sample_count_by_lag = np.zeros(frame_count - 1, dtype=int)
    populated_lags = np.zeros(frame_count - 1, dtype=bool)
    non_target_state = state_indices != target_state_index
    non_target_prefix_count = np.vstack(
        (
            np.zeros((1, center_count), dtype=int),
            np.cumsum(non_target_state, axis=0, dtype=int),
        )
    )
    invalid_step_count = (~valid_step_mask).astype(int)
    invalid_step_prefix_count = np.vstack(
        (
            np.zeros((1, center_count), dtype=int),
            np.cumsum(invalid_step_count, axis=0, dtype=int),
        )
    )
    for lag_frames in range(1, maximum_lag_frames + 1):
        non_target_count_by_origin_and_center = (
            non_target_prefix_count[lag_frames + 1 :]
            - non_target_prefix_count[: frame_count - lag_frames]
        )
        invalid_count_by_origin_and_center = (
            invalid_step_prefix_count[lag_frames:]
            - invalid_step_prefix_count[: frame_count - lag_frames]
        )
        valid_residence = (non_target_count_by_origin_and_center == 0) & (
            invalid_count_by_origin_and_center == 0
        )
        displacement_by_origin_and_center = (
            polarizations_m[lag_frames:] - polarizations_m[:-lag_frames]
        )
        sample_array = displacement_by_origin_and_center[valid_residence]
        if sample_array.shape[0] < 2:
            continue
        centered_samples = sample_array - np.mean(sample_array, axis=0, keepdims=True)
        covariance_by_lag_m2[lag_frames - 1] = (
            centered_samples.T @ centered_samples / float(sample_array.shape[0] - 1)
        )
        sample_count_by_lag[lag_frames - 1] = sample_array.shape[0]
        populated_lags[lag_frames - 1] = True
    return covariance_by_lag_m2, sample_count_by_lag, populated_lags


def _find_diffusive_covariance_window(
    state_label: str,
    covariance_by_lag_m2: Array,
    sample_count_by_lag: Array,
    populated_lags: Array,
    timestep_s: float,
    integration_window_s: float,
    plateau_window_s: float,
) -> tuple[StateDiffusionConvergence, Array, bool]:
    integration_window_s = _positive_float(
        integration_window_s, "zero_frequency_integration_window_s"
    )
    plateau_window_s = _positive_float(
        plateau_window_s, "zero_frequency_plateau_window_s"
    )
    if plateau_window_s >= integration_window_s:
        raise ValueError(
            "zero_frequency_plateau_window_s must be shorter than "
            "zero_frequency_integration_window_s"
        )
    trace_by_lag_m2 = np.trace(covariance_by_lag_m2, axis1=1, axis2=2)
    lag_times_s = (np.arange(trace_by_lag_m2.size, dtype=float) + 1.0) * timestep_s
    finite_lag_indices = np.flatnonzero(
        populated_lags
        & (trace_by_lag_m2 > 0.0)
        & (lag_times_s <= integration_window_s)
    )
    if finite_lag_indices.size < MINIMUM_DIFFUSIVE_WINDOW_LAG_COUNT:
        return (
            _failed_diffusion_convergence(
                state_label, "fewer than four populated positive-covariance lags"
            ),
            np.zeros((CARTESIAN, CARTESIAN), dtype=float),
            False,
        )
    integration_start_s = max(timestep_s, integration_window_s - plateau_window_s)
    for window_start_offset in range(
        finite_lag_indices.size - MINIMUM_DIFFUSIVE_WINDOW_LAG_COUNT + 1
    ):
        window_indices = finite_lag_indices[window_start_offset:]
        window_lag_times_s = lag_times_s[window_indices]
        plateau_mask = window_lag_times_s >= integration_start_s
        if np.count_nonzero(plateau_mask) < MINIMUM_DIFFUSIVE_WINDOW_LAG_COUNT:
            continue
        covariance_window_m2 = covariance_by_lag_m2[window_indices]
        trace_covariance_m2 = trace_by_lag_m2[window_indices]
        trace_slope_m2_s, trace_slope_standard_error_m2_s = _linear_slope_and_error(
            window_lag_times_s, trace_covariance_m2
        )
        plateau_slope_m2_s, plateau_slope_standard_error_m2_s = (
            _linear_slope_and_error(
            window_lag_times_s[plateau_mask], trace_covariance_m2[plateau_mask]
            )
        )
        minimum_plateau_sample_count = int(
            np.min(sample_count_by_lag[window_indices[plateau_mask]])
        )
        integration_slope_uncertainty_m2_s = max(
            trace_slope_standard_error_m2_s,
            abs(trace_slope_m2_s) / np.sqrt(minimum_plateau_sample_count),
        )
        plateau_slope_uncertainty_m2_s = max(
            plateau_slope_standard_error_m2_s,
            abs(plateau_slope_m2_s) / np.sqrt(minimum_plateau_sample_count),
        )
        slope_difference_limit_m2_s = NORMAL_CONFIDENCE_MULTIPLIER_95_PERCENT * (
            integration_slope_uncertainty_m2_s + plateau_slope_uncertainty_m2_s
        )
        if (
            plateau_slope_m2_s <= 0.0
            or abs(plateau_slope_m2_s - trace_slope_m2_s)
            > slope_difference_limit_m2_s
        ):
            continue
        log_log_exponent, log_log_exponent_standard_error = _linear_slope_and_error(
            np.log(window_lag_times_s), np.log(trace_covariance_m2)
        )
        exponent_margin = (
            NORMAL_CONFIDENCE_MULTIPLIER_95_PERCENT
            * log_log_exponent_standard_error
        )
        if trace_slope_m2_s <= 0.0 or not (
            log_log_exponent - exponent_margin
            <= 1.0
            <= log_log_exponent + exponent_margin
        ):
            continue
        tensor_slopes_m2_s = np.empty((CARTESIAN, CARTESIAN), dtype=float)
        for first_axis in range(CARTESIAN):
            for second_axis in range(CARTESIAN):
                tensor_slopes_m2_s[first_axis, second_axis], _ = (
                    _linear_slope_and_error(
                        window_lag_times_s,
                        covariance_window_m2[:, first_axis, second_axis],
                    )
                )
        diffusion_tensor_m2_s = DIFFUSION_FROM_SYMMETRIZED_COVARIANCE_SLOPE * (
            tensor_slopes_m2_s + tensor_slopes_m2_s.T
        )
        if np.min(np.linalg.eigvalsh(diffusion_tensor_m2_s)) < 0.0:
            continue
        sample_counts = sample_count_by_lag[window_indices]
        return (
            StateDiffusionConvergence(
                state_label=state_label,
                convergence_status="converged",
                not_complete_reason="",
                lag_start_frames=int(window_indices[0] + 1),
                lag_stop_frames=int(window_indices[-1] + 1),
                lag_count=int(window_indices.size),
                minimum_samples_per_lag=int(np.min(sample_counts)),
                maximum_samples_per_lag=int(np.max(sample_counts)),
                trace_slope_m2_s=trace_slope_m2_s,
                trace_slope_standard_error_m2_s=trace_slope_standard_error_m2_s,
                log_log_exponent=log_log_exponent,
                log_log_exponent_standard_error=log_log_exponent_standard_error,
            ),
            diffusion_tensor_m2_s,
            True,
        )
    return (
        _failed_diffusion_convergence(
            state_label,
            "no physical integration window has linear covariance growth, a stable "
            "final plateau, and a PSD slope",
        ),
        np.zeros((CARTESIAN, CARTESIAN), dtype=float),
        False,
    )


def _linear_slope_and_error(x_values: Array, y_values: Array) -> tuple[float, float]:
    fit = linear_fit(x_values, y_values)
    return fit.slope, fit.slope_standard_error


def _failed_diffusion_convergence(
    state_label: str,
    reason: str,
) -> StateDiffusionConvergence:
    return StateDiffusionConvergence(
        state_label=state_label,
        convergence_status="not_converged",
        not_complete_reason=reason,
        lag_start_frames=0,
        lag_stop_frames=0,
        lag_count=0,
        minimum_samples_per_lag=0,
        maximum_samples_per_lag=0,
        trace_slope_m2_s=0.0,
        trace_slope_standard_error_m2_s=0.0,
        log_log_exponent=0.0,
        log_log_exponent_standard_error=0.0,
    )


def _self_displacement_sample_count(
    from_indices: Array,
    to_indices: Array,
    charge_displacements: Array,
    displacement_zero_tolerance_m: float,
) -> int:
    sample_count = 0
    for sample_index, from_state_index in enumerate(from_indices):
        if int(from_state_index) != int(to_indices[sample_index]):
            continue
        if (
            float(np.linalg.norm(charge_displacements[sample_index]))
            > displacement_zero_tolerance_m
        ):
            sample_count += 1
    return sample_count


def compute_finite_process_component_drift_residuals(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: Array,
    symmetric_capacity_fluxes_K_ij_mol_m3_s: Array,
    transition_first_moments_d_ij_m: Array,
    directed_transition_sample_counts: Array,
) -> tuple[FiniteProcessComponentDriftResidual, ...]:
    labels = _validated_state_labels(state_labels)
    state_count = len(labels)
    concentrations = np.asarray(state_concentrations_mol_m3, dtype=float)
    if concentrations.shape != (state_count,) or not np.all(
        np.isfinite(concentrations)
    ):
        raise ValueError("state_concentrations_mol_m3 must have shape (n,)")
    for state_index, concentration in enumerate(concentrations):
        _positive_float(
            float(concentration),
            f"state_concentrations_mol_m3[{state_index}]",
        )
    capacity_fluxes = np.asarray(symmetric_capacity_fluxes_K_ij_mol_m3_s, dtype=float)
    if capacity_fluxes.shape != (state_count, state_count) or not np.all(
        np.isfinite(capacity_fluxes)
    ):
        raise ValueError(
            "symmetric_capacity_fluxes_K_ij_mol_m3_s must have shape (n,n)"
        )
    if np.any(capacity_fluxes < 0.0):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must be nonnegative")
    if not np.allclose(capacity_fluxes, capacity_fluxes.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must be symmetric")
    if not np.allclose(np.diag(capacity_fluxes), 0.0, atol=1.0e-12, rtol=0.0):
        raise ValueError(
            "symmetric_capacity_fluxes_K_ij_mol_m3_s diagonal must be zero"
        )
    first_moments = np.asarray(transition_first_moments_d_ij_m, dtype=float)
    if first_moments.shape != (state_count, state_count, CARTESIAN) or not np.all(
        np.isfinite(first_moments)
    ):
        raise ValueError("transition_first_moments_d_ij_m must have shape (n,n,3)")
    directed_counts = np.asarray(directed_transition_sample_counts, dtype=int)
    if directed_counts.shape != (state_count, state_count):
        raise ValueError("directed_transition_sample_counts must have shape (n,n)")
    if np.any(directed_counts < 0):
        raise ValueError("directed_transition_sample_counts must be nonnegative")

    generator = np.zeros((state_count, state_count), dtype=float)
    for state_index, concentration in enumerate(concentrations):
        generator[state_index] = capacity_fluxes[state_index] / float(concentration)
    np.fill_diagonal(generator, 0.0)
    exit_rates = np.sum(generator, axis=1)
    finite_state_drift = np.einsum("ij,ija->ia", generator, first_moments)
    components = _capacity_flux_connected_components(capacity_fluxes)
    component_residuals: list[FiniteProcessComponentDriftResidual] = []
    for component_id, component_indices in enumerate(components):
        component_concentrations = concentrations[component_indices]
        component_drift = finite_state_drift[component_indices]
        weighted_drift = np.einsum("i,ia->a", component_concentrations, component_drift)
        weighted_absolute_drift_scale = float(
            np.sum(np.abs(component_concentrations[:, np.newaxis] * component_drift))
        )
        top_edge_contributions = _top_component_edge_drift_contributions(
            labels,
            capacity_fluxes,
            first_moments,
            directed_counts,
            component_indices,
            int(component_id),
        )
        component_residuals.append(
            FiniteProcessComponentDriftResidual(
                component_id=int(component_id),
                state_labels=tuple(labels[int(index)] for index in component_indices),
                state_concentrations_mol_m3=tuple(
                    float(concentration) for concentration in component_concentrations
                ),
                exit_rates_s_inv=tuple(
                    float(exit_rates[int(index)]) for index in component_indices
                ),
                concentration_sum_mol_m3=float(np.sum(component_concentrations)),
                weighted_drift_mol_m2_s=_vector_to_tuple(weighted_drift),
                weighted_drift_norm_mol_m2_s=float(np.linalg.norm(weighted_drift)),
                weighted_absolute_drift_scale_mol_m2_s=weighted_absolute_drift_scale,
                top_edge_contributions=top_edge_contributions,
            )
        )
    return tuple(component_residuals)


def diagnose_finite_process_legality(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: Array,
    symmetric_capacity_fluxes_K_ij_mol_m3_s: Array,
    transition_first_moments_d_ij_m: Array,
    transition_second_moments_M_ij_m2: Array,
    directed_transition_sample_counts: Array,
) -> FiniteProcessLegalityDiagnostic:
    """Validate reciprocal finite-process tensors and return c^T b diagnostics."""

    labels = _validated_state_labels(state_labels)
    state_count = len(labels)
    concentrations = np.asarray(state_concentrations_mol_m3, dtype=float)
    if concentrations.shape != (state_count,) or not np.all(
        np.isfinite(concentrations)
    ):
        raise ValueError("state_concentrations_mol_m3 must have shape (n,)")
    for state_index, concentration in enumerate(concentrations):
        _positive_float(
            float(concentration),
            f"state_concentrations_mol_m3[{state_index}]",
        )
    capacity_fluxes = np.asarray(symmetric_capacity_fluxes_K_ij_mol_m3_s, dtype=float)
    if capacity_fluxes.shape != (state_count, state_count) or not np.all(
        np.isfinite(capacity_fluxes)
    ):
        raise ValueError(
            "symmetric_capacity_fluxes_K_ij_mol_m3_s must have shape (n,n)"
        )
    if np.any(capacity_fluxes < 0.0):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must be nonnegative")
    if not np.allclose(capacity_fluxes, capacity_fluxes.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must be symmetric")
    if not np.allclose(np.diag(capacity_fluxes), 0.0, atol=1.0e-12, rtol=0.0):
        raise ValueError(
            "symmetric_capacity_fluxes_K_ij_mol_m3_s diagonal must be zero"
        )
    first_moments = np.asarray(transition_first_moments_d_ij_m, dtype=float)
    if first_moments.shape != (state_count, state_count, CARTESIAN) or not np.all(
        np.isfinite(first_moments)
    ):
        raise ValueError("transition_first_moments_d_ij_m must have shape (n,n,3)")
    if not _tensors_match_with_unit_scale(
        first_moments,
        -np.swapaxes(first_moments, 0, 1),
    ):
        raise ValueError("transition_first_moments_d_ji_m must equal -d_ij")
    second_moments = np.asarray(transition_second_moments_M_ij_m2, dtype=float)
    if second_moments.shape != (
        state_count,
        state_count,
        CARTESIAN,
        CARTESIAN,
    ) or not np.all(np.isfinite(second_moments)):
        raise ValueError("transition_second_moments_M_ij_m2 must have shape (n,n,3,3)")
    if not _tensors_match_with_unit_scale(
        second_moments,
        np.swapaxes(second_moments, 0, 1),
    ):
        raise ValueError("transition_second_moments_M_ji_m2 must equal M_ij")
    generator = np.zeros((state_count, state_count), dtype=float)
    for state_index, concentration in enumerate(concentrations):
        generator[state_index] = capacity_fluxes[state_index] / float(concentration)
    np.fill_diagonal(generator, 0.0)
    detailed_balance_residuals = np.abs(
        concentrations[:, np.newaxis] * generator
        - concentrations[np.newaxis, :] * generator.T
    )
    component_drift_residuals = compute_finite_process_component_drift_residuals(
        labels,
        concentrations,
        capacity_fluxes,
        first_moments,
        directed_transition_sample_counts,
    )
    return FiniteProcessLegalityDiagnostic(
        state_labels=labels,
        maximum_detailed_balance_residual_mol_m3_s=float(
            np.max(detailed_balance_residuals)
        ),
        component_drift_residuals=component_drift_residuals,
    )


def _component_drift_residuals_from_records(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: Mapping[str, float],
    reactive_fluxes: tuple[ProjectedGeneratorReactiveFlux, ...],
    conditional_displacement_moments: tuple[ProjectedGeneratorConditionalMoment, ...],
) -> tuple[FiniteProcessComponentDriftResidual, ...]:
    state_count = len(state_labels)
    state_index_by_label = {
        state_label: state_index for state_index, state_label in enumerate(state_labels)
    }
    concentrations = np.asarray(
        [state_concentrations_mol_m3[state_label] for state_label in state_labels],
        dtype=float,
    )
    capacity_fluxes = np.zeros((state_count, state_count), dtype=float)
    first_moments = np.zeros((state_count, state_count, CARTESIAN), dtype=float)
    directed_counts = np.zeros((state_count, state_count), dtype=int)
    for flux_record in reactive_fluxes:
        from_state_index = state_index_by_label[flux_record.from_state_label]
        to_state_index = state_index_by_label[flux_record.to_state_label]
        capacity_fluxes[from_state_index, to_state_index] = float(
            flux_record.symmetric_flux_mol_m3_s
        )
        capacity_fluxes[to_state_index, from_state_index] = float(
            flux_record.symmetric_flux_mol_m3_s
        )
        directed_counts[from_state_index, to_state_index] = int(
            flux_record.forward_sample_count
        )
        directed_counts[to_state_index, from_state_index] = int(
            flux_record.reverse_sample_count
        )
    for moment_record in conditional_displacement_moments:
        from_state_index = state_index_by_label[moment_record.from_state_label]
        to_state_index = state_index_by_label[moment_record.to_state_label]
        first_moment = np.asarray(moment_record.mean_charge_displacement_m, dtype=float)
        first_moments[from_state_index, to_state_index] = first_moment
        first_moments[to_state_index, from_state_index] = -first_moment
    return compute_finite_process_component_drift_residuals(
        state_labels,
        concentrations,
        capacity_fluxes,
        first_moments,
        directed_counts,
    )


def _finite_process_legality_from_records(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: Mapping[str, float],
    reactive_fluxes: tuple[ProjectedGeneratorReactiveFlux, ...],
    conditional_displacement_moments: tuple[ProjectedGeneratorConditionalMoment, ...],
) -> FiniteProcessLegalityDiagnostic:
    state_count = len(state_labels)
    state_index_by_label = {
        state_label: state_index for state_index, state_label in enumerate(state_labels)
    }
    concentrations = np.asarray(
        [state_concentrations_mol_m3[state_label] for state_label in state_labels],
        dtype=float,
    )
    capacity_fluxes = np.zeros((state_count, state_count), dtype=float)
    first_moments = np.zeros((state_count, state_count, CARTESIAN), dtype=float)
    second_moments = np.zeros(
        (state_count, state_count, CARTESIAN, CARTESIAN),
        dtype=float,
    )
    directed_counts = np.zeros((state_count, state_count), dtype=int)
    for flux_record in reactive_fluxes:
        from_state_index = state_index_by_label[flux_record.from_state_label]
        to_state_index = state_index_by_label[flux_record.to_state_label]
        capacity_fluxes[from_state_index, to_state_index] = float(
            flux_record.symmetric_flux_mol_m3_s
        )
        capacity_fluxes[to_state_index, from_state_index] = float(
            flux_record.symmetric_flux_mol_m3_s
        )
        directed_counts[from_state_index, to_state_index] = int(
            flux_record.forward_sample_count
        )
        directed_counts[to_state_index, from_state_index] = int(
            flux_record.reverse_sample_count
        )
    for moment_record in conditional_displacement_moments:
        from_state_index = state_index_by_label[moment_record.from_state_label]
        to_state_index = state_index_by_label[moment_record.to_state_label]
        first_moment = np.asarray(moment_record.mean_charge_displacement_m, dtype=float)
        second_moment = np.asarray(moment_record.second_moment_m2, dtype=float)
        first_moments[from_state_index, to_state_index] = first_moment
        first_moments[to_state_index, from_state_index] = -first_moment
        second_moments[from_state_index, to_state_index] = second_moment
        second_moments[to_state_index, from_state_index] = second_moment
    return diagnose_finite_process_legality(
        state_labels,
        concentrations,
        capacity_fluxes,
        first_moments,
        second_moments,
        directed_counts,
    )


def _capacity_flux_connected_components(capacity_fluxes: Array) -> tuple[Array, ...]:
    adjacency = (np.abs(capacity_fluxes) > 0.0) | (np.abs(capacity_fluxes.T) > 0.0)
    state_count = capacity_fluxes.shape[0]
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
            for neighbor_index in np.flatnonzero(adjacency[state_index]):
                if visited[int(neighbor_index)]:
                    continue
                visited[int(neighbor_index)] = True
                stack.append(int(neighbor_index))
        components.append(np.asarray(component, dtype=int))
    return tuple(components)


def _top_component_edge_drift_contributions(
    state_labels: tuple[str, ...],
    capacity_fluxes: Array,
    first_moments: Array,
    directed_transition_sample_counts: Array,
    component_indices: Array,
    component_id: int,
) -> tuple[FiniteProcessEdgeDriftContribution, ...]:
    edge_contributions: list[FiniteProcessEdgeDriftContribution] = []
    for from_state_index in component_indices:
        for to_state_index in component_indices:
            if int(from_state_index) == int(to_state_index):
                continue
            capacity_flux = float(
                capacity_fluxes[int(from_state_index), int(to_state_index)]
            )
            first_moment = np.asarray(
                first_moments[int(from_state_index), int(to_state_index)],
                dtype=float,
            )
            if capacity_flux == 0.0 and float(np.linalg.norm(first_moment)) == 0.0:
                continue
            contribution = capacity_flux * first_moment
            forward_sample_count = int(
                directed_transition_sample_counts[
                    int(from_state_index),
                    int(to_state_index),
                ]
            )
            reverse_sample_count = int(
                directed_transition_sample_counts[
                    int(to_state_index),
                    int(from_state_index),
                ]
            )
            edge_contributions.append(
                FiniteProcessEdgeDriftContribution(
                    component_id=component_id,
                    from_state_label=state_labels[int(from_state_index)],
                    to_state_label=state_labels[int(to_state_index)],
                    contribution_mol_m2_s=_vector_to_tuple(contribution),
                    contribution_norm_mol_m2_s=float(np.linalg.norm(contribution)),
                    capacity_flux_mol_m3_s=capacity_flux,
                    first_moment_norm_m=float(np.linalg.norm(first_moment)),
                    forward_sample_count=forward_sample_count,
                    reverse_sample_count=reverse_sample_count,
                    missing_reverse_event_candidate=(
                        forward_sample_count > 0 and reverse_sample_count == 0
                    ),
                )
            )
    sorted_edge_contributions = sorted(
        edge_contributions,
        key=_edge_drift_contribution_sort_key,
        reverse=True,
    )
    return tuple(sorted_edge_contributions[:TOP_COMPONENT_EDGE_CONTRIBUTION_COUNT])


def _edge_drift_contribution_sort_key(
    edge_contribution: FiniteProcessEdgeDriftContribution,
) -> float:
    return edge_contribution.contribution_norm_mol_m2_s


def _tensors_match_with_unit_scale(first_tensor: Array, second_tensor: Array) -> bool:
    difference = np.asarray(first_tensor, dtype=float) - np.asarray(
        second_tensor,
        dtype=float,
    )
    scale = max(
        float(np.max(np.abs(first_tensor))),
        float(np.max(np.abs(second_tensor))),
        np.finfo(float).tiny,
    )
    tolerance = 100.0 * np.finfo(float).eps * scale
    return bool(float(np.max(np.abs(difference))) <= tolerance)


def _validate_psd(matrix: Array, label: str) -> None:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(matrix))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{label} must be positive semidefinite")


def _matrix_to_tuple(matrix: Array) -> tuple[tuple[float, float, float], ...]:
    result = np.asarray(matrix, dtype=float)
    if result.shape != (CARTESIAN, CARTESIAN) or not np.all(np.isfinite(result)):
        raise ValueError("matrix must have shape (3, 3)")
    return tuple(
        tuple(
            float(result[row_index, column_index]) for column_index in range(CARTESIAN)
        )
        for row_index in range(CARTESIAN)
    )


def _vector_to_tuple(vector: Array) -> tuple[float, float, float]:
    result = np.asarray(vector, dtype=float)
    if result.shape != (CARTESIAN,) or not np.all(np.isfinite(result)):
        raise ValueError("vector must have shape (3,)")
    return tuple(float(component) for component in result)


def _tensor3_to_tuple(
    tensor: Array,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    result = np.asarray(tensor, dtype=float)
    if (
        result.ndim != 3
        or result.shape[2] != CARTESIAN
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("tensor must have shape (n, n, 3)")
    return tuple(
        tuple(
            _vector_to_tuple(result[first_index, second_index])
            for second_index in range(result.shape[1])
        )
        for first_index in range(result.shape[0])
    )


def _positive_float(value: float, label: str) -> float:
    numeric_value = float(value)
    if numeric_value <= 0.0 or not np.isfinite(numeric_value):
        raise ValueError(f"{label} must be positive and finite")
    return numeric_value


def _nonnegative_float(value: float, label: str) -> float:
    numeric_value = float(value)
    if numeric_value < 0.0 or not np.isfinite(numeric_value):
        raise ValueError(f"{label} must be nonnegative and finite")
    return numeric_value
