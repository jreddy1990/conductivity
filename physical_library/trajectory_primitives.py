"""Trajectory samples projected into finite-generator conductivity primitives."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from conductivity.physical_library.projected_analytical_conductivity import CARTESIAN

ANGSTROM_TO_M = 1.0e-10
DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M = 0.0
TOP_COMPONENT_EDGE_CONTRIBUTION_COUNT = 5  # Compact diagnostic table, not physics.

Array = np.ndarray


@dataclass(frozen=True)
class TrajectoryMarkovAdditiveSampleInput:
    state_labels: tuple[str, ...]
    occupancy_state_index_by_observation: Array
    from_state_index_by_step: Array
    to_state_index_by_step: Array
    charge_displacement_by_step_m: Array
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
    component_drift_residuals: tuple["FiniteProcessComponentDriftResidual", ...]
    component_solvable_projection: "FiniteProcessSolvableProjection"
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
class FiniteProcessSolvableProjection:
    projected_first_moments_d_ij_m: tuple[tuple[tuple[float, float, float], ...], ...]
    removed_first_moments_d_ij_m: tuple[tuple[tuple[float, float, float], ...], ...]
    maximum_removed_first_moment_norm_m: float
    projected_component_drift_residuals: tuple[FiniteProcessComponentDriftResidual, ...]


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
class ProjectedGeneratorPrimitiveSet:
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: Mapping[str, float]
    state_occupancy_fractions: Mapping[str, float]
    reactive_fluxes: tuple[ProjectedGeneratorReactiveFlux, ...]
    conditional_displacement_moments: tuple[ProjectedGeneratorConditionalMoment, ...]
    self_current_tensors: tuple[ProjectedGeneratorSelfCurrentTensor, ...]
    diagnostics: ProjectedGeneratorPrimitiveDiagnostics


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

    remap = _visited_state_remap(
        occupancy_state_indices,
        from_state_indices,
        to_state_indices,
    )
    remapped_labels = tuple(state_labels[original_index] for original_index in sorted(remap))
    remapped_occupancy = np.asarray(
        [remap[int(index)] for index in occupancy_state_indices],
        dtype=int,
    )
    remapped_from = np.asarray(
        [remap[int(index)] for index in from_state_indices],
        dtype=int,
    )
    remapped_to = np.asarray(
        [remap[int(index)] for index in to_state_indices],
        dtype=int,
    )

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
    )
    conditional_moments = _conditional_displacement_moments(
        remapped_labels,
        remapped_from,
        remapped_to,
        charge_displacements,
    )
    self_current_tensors = _self_current_tensors(
        remapped_labels,
        remapped_from,
        remapped_to,
        charge_displacements,
        state_concentrations,
        timestep_s,
        displacement_zero_tolerance_m,
    )
    component_drift_residuals = _component_drift_residuals_from_records(
        remapped_labels,
        state_concentrations,
        reactive_fluxes,
        conditional_moments,
    )
    component_solvable_projection = _component_solvable_projection_from_records(
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
        step_count=int(remapped_from.size),
        transition_sample_count=int(np.count_nonzero(remapped_from != remapped_to)),
        self_displacement_sample_count=_self_displacement_sample_count(
            remapped_from,
            remapped_to,
            charge_displacements,
            displacement_zero_tolerance_m,
        ),
        generated_event_count=len(reactive_fluxes) + len(self_current_tensors),
        minimum_state_concentration_mol_m3=float(min(state_concentrations.values())),
        maximum_state_concentration_mol_m3=float(max(state_concentrations.values())),
        total_transport_concentration_mol_m3=total_concentration_mol_m3,
        trajectory_time_s=float(remapped_from.size * timestep_s),
        component_drift_residuals=component_drift_residuals,
        component_solvable_projection=component_solvable_projection,
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
) -> tuple[ProjectedGeneratorReactiveFlux, ...]:
    directed_counts: dict[tuple[int, int], int] = defaultdict(int)
    for sample_index, from_state_index in enumerate(from_indices):
        to_state_index = int(to_indices[sample_index])
        if int(from_state_index) == to_state_index:
            continue
        directed_counts[(int(from_state_index), to_state_index)] += 1
    event_flux_per_sample_mol_m3_s = (
        total_concentration_mol_m3 / (2.0 * float(from_indices.size) * timestep_s)
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
            0.5
            * float(forward_count + reverse_count)
            * event_flux_per_sample_mol_m3_s
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
        oriented_displacement = np.asarray(charge_displacements[sample_index], dtype=float)
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
    from_indices: Array,
    to_indices: Array,
    charge_displacements: Array,
    state_concentrations: Mapping[str, float],
    timestep_s: float,
    displacement_zero_tolerance_m: float,
) -> tuple[ProjectedGeneratorSelfCurrentTensor, ...]:
    samples_by_state: dict[int, list[Array]] = defaultdict(list)
    for sample_index, from_state_index in enumerate(from_indices):
        if int(from_state_index) != int(to_indices[sample_index]):
            continue
        displacement = charge_displacements[sample_index]
        if float(np.linalg.norm(displacement)) <= displacement_zero_tolerance_m:
            continue
        samples_by_state[int(from_state_index)].append(displacement)
    records: list[ProjectedGeneratorSelfCurrentTensor] = []
    for state_index in sorted(samples_by_state):
        samples = np.asarray(samples_by_state[state_index], dtype=float)
        diffusion_tensor = np.einsum("sa,sb->ab", samples, samples) / (
            2.0 * timestep_s * float(samples.shape[0])
        )
        _validate_psd(diffusion_tensor, "self-current tensor")
        state_label = state_labels[state_index]
        records.append(
            ProjectedGeneratorSelfCurrentTensor(
                state_label=state_label,
                sample_count=int(samples.shape[0]),
                concentration_mol_m3=float(state_concentrations[state_label]),
                diffusion_tensor_m2_s=_matrix_to_tuple(diffusion_tensor),
            )
        )
    return tuple(records)


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
    if concentrations.shape != (state_count,) or not np.all(np.isfinite(concentrations)):
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
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must have shape (n,n)")
    if np.any(capacity_fluxes < 0.0):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must be nonnegative")
    if not np.allclose(capacity_fluxes, capacity_fluxes.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must be symmetric")
    if not np.allclose(np.diag(capacity_fluxes), 0.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s diagonal must be zero")
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
    if concentrations.shape != (state_count,) or not np.all(np.isfinite(concentrations)):
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
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must have shape (n,n)")
    if np.any(capacity_fluxes < 0.0):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must be nonnegative")
    if not np.allclose(capacity_fluxes, capacity_fluxes.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s must be symmetric")
    if not np.allclose(np.diag(capacity_fluxes), 0.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("symmetric_capacity_fluxes_K_ij_mol_m3_s diagonal must be zero")
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


def _component_solvable_projection_from_records(
    state_labels: tuple[str, ...],
    state_concentrations_mol_m3: Mapping[str, float],
    reactive_fluxes: tuple[ProjectedGeneratorReactiveFlux, ...],
    conditional_displacement_moments: tuple[ProjectedGeneratorConditionalMoment, ...],
) -> FiniteProcessSolvableProjection:
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
    projected_first_moments = project_first_moments_to_reversible_component_space(
        first_moments
    )
    removed_first_moments = first_moments - projected_first_moments
    projected_residuals = compute_finite_process_component_drift_residuals(
        state_labels,
        concentrations,
        capacity_fluxes,
        projected_first_moments,
        directed_counts,
    )
    removed_norms = np.linalg.norm(removed_first_moments, axis=2)
    return FiniteProcessSolvableProjection(
        projected_first_moments_d_ij_m=_tensor3_to_tuple(projected_first_moments),
        removed_first_moments_d_ij_m=_tensor3_to_tuple(removed_first_moments),
        maximum_removed_first_moment_norm_m=float(np.max(removed_norms)),
        projected_component_drift_residuals=projected_residuals,
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


def project_first_moments_to_reversible_component_space(
    transition_first_moments_d_ij_m: Array,
) -> Array:
    first_moments = np.asarray(transition_first_moments_d_ij_m, dtype=float)
    if first_moments.ndim != 3 or first_moments.shape[2] != CARTESIAN:
        raise ValueError("transition_first_moments_d_ij_m must have shape (n,n,3)")
    if first_moments.shape[0] != first_moments.shape[1]:
        raise ValueError("transition_first_moments_d_ij_m must be square in state axes")
    if not np.all(np.isfinite(first_moments)):
        raise ValueError("transition_first_moments_d_ij_m must be finite")
    projected = 0.5 * (first_moments - np.swapaxes(first_moments, 0, 1))
    for state_index in range(projected.shape[0]):
        projected[state_index, state_index] = 0.0
    return projected


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
        tuple(float(result[row_index, column_index]) for column_index in range(CARTESIAN))
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
    if result.ndim != 3 or result.shape[2] != CARTESIAN or not np.all(
        np.isfinite(result)
    ):
        raise ValueError("tensor must have shape (n, n, 3)")
    return tuple(
        tuple(_vector_to_tuple(result[first_index, second_index]) for second_index in range(result.shape[1]))
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
