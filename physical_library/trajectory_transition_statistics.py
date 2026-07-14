"""Empirical committor statistics from fixed-generator trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import CARTESIAN_COMPONENT_COUNT
from utils.time_series_statistics import mean_and_standard_error

Array = np.ndarray
SOURCE_BASIN_LABEL = 0
INTERIOR_BASIN_LABEL = 1
DESTINATION_BASIN_LABEL = 2


@dataclass(frozen=True)
class EmpiricalCommittorBin:
    lower_edge: float
    upper_edge: float
    sample_count: int
    destination_hit_count: int
    observed_committor: float
    generator_committor: float
    standard_error: float


@dataclass(frozen=True)
class EmpiricalCommittorEstimate:
    bins: tuple[EmpiricalCommittorBin, ...]
    maximum_standardized_residual: float
    symmetric_capacity_events_per_s: float


@dataclass(frozen=True)
class TransitionMomentEstimate:
    from_state_label: str
    to_state_label: str
    replica_count: int
    event_count: int
    symmetric_capacity_events_per_s: float
    capacity_standard_error_events_per_s: float
    first_moment_m: tuple[float, float, float]
    first_moment_standard_error_m: tuple[float, float, float]
    second_moment_m2: tuple[tuple[float, float, float], ...]
    second_moment_standard_error_m2: tuple[tuple[float, float, float], ...]


def estimate_empirical_committor(
    reaction_coordinate: Array,
    basin_labels: Array,
    frame_interval_s: float,
    bin_edges: Array,
) -> EmpiricalCommittorEstimate:
    coordinate = np.asarray(reaction_coordinate, dtype=float)
    labels = np.asarray(basin_labels, dtype=int)
    edges = np.asarray(bin_edges, dtype=float)
    _validate_committor_inputs(coordinate, labels, frame_interval_s, edges)
    destination_outcome = _next_basin_destination_outcome(labels)
    resolved_mask = (labels == INTERIOR_BASIN_LABEL) & (destination_outcome >= 0)
    interior_bins = np.digitize(coordinate[resolved_mask], edges[1:-1], right=False)
    interior_outcomes = destination_outcome[resolved_mask]
    bin_count = edges.size - 1
    observed_sample_counts = np.bincount(interior_bins, minlength=bin_count)
    destination_hit_counts = np.bincount(
        interior_bins,
        weights=interior_outcomes,
        minlength=bin_count,
    ).astype(int)
    observed_committor = np.divide(
        destination_hit_counts,
        observed_sample_counts,
        out=np.zeros(bin_count, dtype=float),
        where=observed_sample_counts > 0,
    )
    frame_states = _committor_frame_states(coordinate, labels, edges)
    symmetric_counts = _symmetric_transition_counts(frame_states, bin_count + 2)
    generator_committor = _solve_discrete_committor(symmetric_counts)[1:-1]
    standard_errors = np.sqrt(
        np.divide(
            observed_committor * (1.0 - observed_committor),
            observed_sample_counts,
            out=np.zeros(bin_count, dtype=float),
            where=observed_sample_counts > 0,
        )
    )
    populated = observed_sample_counts > 0
    if not np.any(populated):
        raise ValueError("empirical committor has no resolved interior samples")
    residual_denominator = np.maximum(
        standard_errors[populated],
        1.0 / np.sqrt(observed_sample_counts[populated]),
    )
    maximum_standardized_residual = float(
        np.max(
            np.abs(observed_committor[populated] - generator_committor[populated])
            / residual_denominator
        )
    )
    source_flux_count = float(np.sum(symmetric_counts[0, 1:-1]))
    trajectory_duration_s = frame_interval_s * (coordinate.size - 1)
    return EmpiricalCommittorEstimate(
        bins=tuple(
            EmpiricalCommittorBin(
                lower_edge=float(edges[bin_index]),
                upper_edge=float(edges[bin_index + 1]),
                sample_count=int(observed_sample_counts[bin_index]),
                destination_hit_count=int(destination_hit_counts[bin_index]),
                observed_committor=float(observed_committor[bin_index]),
                generator_committor=float(generator_committor[bin_index]),
                standard_error=float(standard_errors[bin_index]),
            )
            for bin_index in range(bin_count)
        ),
        maximum_standardized_residual=maximum_standardized_residual,
        symmetric_capacity_events_per_s=source_flux_count / trajectory_duration_s,
    )


def estimate_transition_moments_by_replica(
    from_state_labels: tuple[str, ...],
    to_state_labels: tuple[str, ...],
    replica_ids: tuple[str, ...],
    charge_displacements_m: Array,
    replica_durations_s: tuple[tuple[str, float], ...],
) -> tuple[TransitionMomentEstimate, ...]:
    displacements = np.asarray(charge_displacements_m, dtype=float)
    event_count = len(from_state_labels)
    cartesian_count = int(CARTESIAN_COMPONENT_COUNT)
    if (
        len(to_state_labels) != event_count
        or len(replica_ids) != event_count
        or displacements.shape != (event_count, cartesian_count)
        or not np.all(np.isfinite(displacements))
    ):
        raise ValueError("transition event fields must align with finite displacements")
    duration_by_replica = dict(replica_durations_s)
    if len(duration_by_replica) != len(replica_durations_s):
        raise ValueError("replica durations must have unique replica ids")
    if any(duration <= 0.0 for duration in duration_by_replica.values()):
        raise ValueError("replica durations must be positive")
    edge_keys = tuple(
        sorted(
            {
                tuple(sorted((from_label, to_label)))
                for from_label, to_label in zip(
                    from_state_labels, to_state_labels, strict=True
                )
                if from_label != to_label
            }
        )
    )
    return tuple(
        _estimate_edge_moments(
            edge_key=edge_key,
            from_state_labels=from_state_labels,
            to_state_labels=to_state_labels,
            replica_ids=replica_ids,
            displacements_m=displacements,
            duration_by_replica=duration_by_replica,
        )
        for edge_key in edge_keys
    )


def _estimate_edge_moments(
    edge_key: tuple[str, str],
    from_state_labels: tuple[str, ...],
    to_state_labels: tuple[str, ...],
    replica_ids: tuple[str, ...],
    displacements_m: Array,
    duration_by_replica: dict[str, float],
) -> TransitionMomentEstimate:
    normalized_by_replica: dict[str, list[Array]] = {
        replica_id: [] for replica_id in duration_by_replica
    }
    for event_index, from_label in enumerate(from_state_labels):
        to_label = to_state_labels[event_index]
        if tuple(sorted((from_label, to_label))) != edge_key:
            continue
        replica_id = replica_ids[event_index]
        if replica_id not in normalized_by_replica:
            raise ValueError(f"transition event has undeclared replica {replica_id}")
        displacement = displacements_m[event_index]
        if from_label != edge_key[0]:
            displacement = -displacement
        normalized_by_replica[replica_id].append(displacement)
    populated_replicas = tuple(
        replica_id
        for replica_id, samples in normalized_by_replica.items()
        if samples
    )
    if len(populated_replicas) < 2:
        raise ValueError(
            f"transition edge {edge_key} requires events in at least two replicas"
        )
    capacities = np.asarray(
        [
            len(normalized_by_replica[replica_id])
            / (2.0 * duration_by_replica[replica_id])
            for replica_id in populated_replicas
        ],
        dtype=float,
    )
    first_moments = np.asarray(
        [
            np.mean(normalized_by_replica[replica_id], axis=0)
            for replica_id in populated_replicas
        ]
    )
    second_moments = np.asarray(
        [
            np.mean(
                [
                    np.outer(displacement, displacement)
                    for displacement in normalized_by_replica[replica_id]
                ],
                axis=0,
            )
            for replica_id in populated_replicas
        ]
    )
    capacity, capacity_standard_error = mean_and_standard_error(capacities)
    first_mean = np.mean(first_moments, axis=0)
    first_standard_error = np.std(first_moments, axis=0, ddof=1) / np.sqrt(
        len(populated_replicas)
    )
    second_mean = np.mean(second_moments, axis=0)
    second_standard_error = np.std(second_moments, axis=0, ddof=1) / np.sqrt(
        len(populated_replicas)
    )
    sample_count = sum(
        len(normalized_by_replica[replica_id]) for replica_id in populated_replicas
    )
    return TransitionMomentEstimate(
        from_state_label=edge_key[0],
        to_state_label=edge_key[1],
        replica_count=len(populated_replicas),
        event_count=sample_count,
        symmetric_capacity_events_per_s=capacity,
        capacity_standard_error_events_per_s=capacity_standard_error,
        first_moment_m=tuple(float(value) for value in first_mean),
        first_moment_standard_error_m=tuple(
            float(value) for value in first_standard_error
        ),
        second_moment_m2=tuple(
            tuple(float(value) for value in row) for row in second_mean
        ),
        second_moment_standard_error_m2=tuple(
            tuple(float(value) for value in row) for row in second_standard_error
        ),
    )


def _validate_committor_inputs(
    coordinate: Array,
    labels: Array,
    frame_interval_s: float,
    edges: Array,
) -> None:
    if coordinate.ndim != 1 or labels.shape != coordinate.shape:
        raise ValueError("reaction coordinate and basin labels must be aligned vectors")
    if not np.all(np.isfinite(coordinate)) or frame_interval_s <= 0.0:
        raise ValueError("committor coordinate and frame interval must be physical")
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("committor bin edges must be a strictly increasing vector")
    legal_labels = {SOURCE_BASIN_LABEL, INTERIOR_BASIN_LABEL, DESTINATION_BASIN_LABEL}
    if not set(int(value) for value in np.unique(labels)).issubset(legal_labels):
        raise ValueError("basin labels must be source, interior, or destination")
    if not np.any(labels == SOURCE_BASIN_LABEL) or not np.any(
        labels == DESTINATION_BASIN_LABEL
    ):
        raise ValueError("committor trajectory must visit both boundary basins")


def _next_basin_destination_outcome(labels: Array) -> Array:
    outcomes = np.full(labels.size, -1, dtype=int)
    next_outcome = -1
    for frame_index in range(labels.size - 1, -1, -1):
        label = int(labels[frame_index])
        match label:
            case 0:
                next_outcome = 0
            case 1:
                pass
            case 2:
                next_outcome = 1
            case _:
                raise ValueError(f"unsupported basin label {label}")
        outcomes[frame_index] = next_outcome
    return outcomes


def _committor_frame_states(coordinate: Array, labels: Array, edges: Array) -> Array:
    bin_count = edges.size - 1
    states = np.digitize(coordinate, edges[1:-1], right=False).astype(int) + 1
    states[labels == SOURCE_BASIN_LABEL] = 0
    states[labels == DESTINATION_BASIN_LABEL] = bin_count + 1
    return states


def _symmetric_transition_counts(states: Array, state_count: int) -> Array:
    counts = np.zeros((state_count, state_count), dtype=float)
    for from_state, to_state in zip(states[:-1], states[1:], strict=True):
        if from_state == to_state:
            continue
        counts[int(from_state), int(to_state)] += 1.0
        counts[int(to_state), int(from_state)] += 1.0
    return 0.5 * counts


def _solve_discrete_committor(symmetric_counts: Array) -> Array:
    laplacian = np.diag(np.sum(symmetric_counts, axis=1)) - symmetric_counts
    interior_laplacian = laplacian[1:-1, 1:-1]
    destination_coupling = -laplacian[1:-1, -1]
    if np.linalg.matrix_rank(interior_laplacian) != interior_laplacian.shape[0]:
        raise ValueError("empirical committor transition graph is disconnected")
    interior_committor = np.linalg.solve(
        interior_laplacian, destination_coupling
    )
    return np.concatenate((np.asarray([0.0]), interior_committor, np.asarray([1.0])))
