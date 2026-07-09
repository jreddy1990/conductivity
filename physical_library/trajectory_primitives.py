"""Trajectory samples projected into finite-generator conductivity primitives."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from conductivity.physical_library.projected_analytical_conductivity import CARTESIAN

ANGSTROM_TO_M = 1.0e-10
DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M = 0.0

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
