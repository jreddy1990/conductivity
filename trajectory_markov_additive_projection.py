"""Trajectory-derived Markov-additive Green-Kubo projection.

This module is the trajectory bridge for the finite Markov-additive readout.
It takes a labeled trajectory, estimates the finite reversible event process
from observed occupancies, transition fluxes, and charge-displacement
increments, then delegates conductivity evaluation to the existing
Markov-additive Green-Kubo implementation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from conductivity.finite_markov_additive_green_kubo import (
    MarkovAdditiveConductivityInput,
    MarkovAdditiveConductivityResult,
    MarkovAdditiveEvent,
    compute_markov_additive_green_kubo_conductivity,
)


DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M = 0.0


@dataclass(frozen=True)
class TrajectoryMarkovAdditiveProjectionInput:
    """Input for a trajectory-derived finite event projection."""

    state_labels: tuple[str, ...]
    state_index_by_frame: np.ndarray
    charge_displacement_by_step_m: np.ndarray
    dt_s: float
    total_transport_concentration_mol_m3: float
    temperature_K: float
    displacement_zero_tolerance_m: float = DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M


@dataclass(frozen=True)
class TrajectoryMarkovAdditiveProjectionDiagnostics:
    original_state_count: int
    visited_state_count: int
    frame_count: int
    step_count: int
    transition_sample_count: int
    self_displacement_sample_count: int
    generated_event_count: int
    minimum_state_concentration_mol_m3: float
    maximum_state_concentration_mol_m3: float
    total_transport_concentration_mol_m3: float
    trajectory_time_s: float


@dataclass(frozen=True)
class TrajectoryMarkovAdditiveProjectionResult:
    markov_input: MarkovAdditiveConductivityInput
    conductivity_result: MarkovAdditiveConductivityResult
    diagnostics: TrajectoryMarkovAdditiveProjectionDiagnostics
    state_index_remap: Mapping[int, int]


def compute_trajectory_projected_markov_additive_conductivity(
    projection_input: TrajectoryMarkovAdditiveProjectionInput,
) -> TrajectoryMarkovAdditiveProjectionResult:
    """Estimate a reversible finite event process and evaluate conductivity."""

    markov_input, diagnostics, state_index_remap = (
        build_trajectory_markov_additive_input(projection_input)
    )
    conductivity_result = compute_markov_additive_green_kubo_conductivity(
        markov_input,
    )
    return TrajectoryMarkovAdditiveProjectionResult(
        markov_input=markov_input,
        conductivity_result=conductivity_result,
        diagnostics=diagnostics,
        state_index_remap=state_index_remap,
    )


def build_trajectory_markov_additive_input(
    projection_input: TrajectoryMarkovAdditiveProjectionInput,
) -> tuple[
    MarkovAdditiveConductivityInput,
    TrajectoryMarkovAdditiveProjectionDiagnostics,
    dict[int, int],
]:
    """Build a reversible finite Markov-additive input from trajectory data."""

    state_labels = _validated_state_labels(projection_input.state_labels)
    state_index_by_frame = _validated_state_index_by_frame(
        projection_input.state_index_by_frame,
        len(state_labels),
    )
    charge_displacements = _validated_charge_displacements(
        projection_input.charge_displacement_by_step_m,
        state_index_by_frame.shape[0] - 1,
    )
    dt_s = _positive_float(projection_input.dt_s, "dt_s")
    total_concentration_mol_m3 = _positive_float(
        projection_input.total_transport_concentration_mol_m3,
        "total_transport_concentration_mol_m3",
    )
    temperature_K = _positive_float(projection_input.temperature_K, "temperature_K")
    displacement_zero_tolerance_m = _nonnegative_float(
        projection_input.displacement_zero_tolerance_m,
        "displacement_zero_tolerance_m",
    )

    state_index_remap, remapped_labels, remapped_states = _remap_visited_states(
        state_labels,
        state_index_by_frame,
    )
    step_count = int(charge_displacements.shape[0])
    state_concentrations_mol_m3 = _state_concentrations_from_occupancy(
        remapped_states[:-1],
        len(remapped_labels),
        total_concentration_mol_m3,
    )

    pair_samples_by_state_pair: dict[tuple[int, int], list[np.ndarray]] = (
        defaultdict(list)
    )
    self_samples_by_state: dict[int, list[np.ndarray]] = defaultdict(list)
    transition_sample_count = 0
    self_displacement_sample_count = 0
    for step_index, charge_displacement_m in enumerate(charge_displacements):
        from_state_index = int(remapped_states[step_index])
        to_state_index = int(remapped_states[step_index + 1])
        if from_state_index == to_state_index:
            if (
                float(np.linalg.norm(charge_displacement_m))
                > displacement_zero_tolerance_m
            ):
                self_samples_by_state[from_state_index].append(
                    charge_displacement_m,
                )
                self_displacement_sample_count += 1
            continue
        lower_state_index = min(from_state_index, to_state_index)
        upper_state_index = max(from_state_index, to_state_index)
        canonical_displacement_m = (
            charge_displacement_m
            if from_state_index == lower_state_index
            else -charge_displacement_m
        )
        pair_samples_by_state_pair[
            (lower_state_index, upper_state_index)
        ].append(canonical_displacement_m)
        transition_sample_count += 1

    event_flux_mol_m3_s = total_concentration_mol_m3 / (
        2.0 * step_count * dt_s
    )
    events = _trajectory_events_from_samples(
        pair_samples_by_state_pair,
        self_samples_by_state,
        remapped_labels,
        state_concentrations_mol_m3,
        event_flux_mol_m3_s,
    )
    if not events:
        raise ValueError(
            "trajectory projection produced no nonzero Markov-additive events",
        )

    diagnostics = TrajectoryMarkovAdditiveProjectionDiagnostics(
        original_state_count=len(state_labels),
        visited_state_count=len(remapped_labels),
        frame_count=int(state_index_by_frame.shape[0]),
        step_count=step_count,
        transition_sample_count=transition_sample_count,
        self_displacement_sample_count=self_displacement_sample_count,
        generated_event_count=len(events),
        minimum_state_concentration_mol_m3=float(
            np.min(state_concentrations_mol_m3),
        ),
        maximum_state_concentration_mol_m3=float(
            np.max(state_concentrations_mol_m3),
        ),
        total_transport_concentration_mol_m3=total_concentration_mol_m3,
        trajectory_time_s=float(step_count * dt_s),
    )

    return (
        MarkovAdditiveConductivityInput(
            state_labels=tuple(remapped_labels),
            state_concentrations_mol_m3=state_concentrations_mol_m3,
            events=tuple(events),
            temperature_K=temperature_K,
        ),
        diagnostics,
        state_index_remap,
    )


def _trajectory_events_from_samples(
    pair_samples_by_state_pair: Mapping[tuple[int, int], Sequence[np.ndarray]],
    self_samples_by_state: Mapping[int, Sequence[np.ndarray]],
    remapped_labels: tuple[str, ...],
    state_concentrations_mol_m3: np.ndarray,
    event_flux_mol_m3_s: float,
) -> tuple[MarkovAdditiveEvent, ...]:
    event_flux = _positive_float(event_flux_mol_m3_s, "event_flux_mol_m3_s")
    events: list[MarkovAdditiveEvent] = []
    for state_pair, displacements_m in sorted(pair_samples_by_state_pair.items()):
        lower_state_index, upper_state_index = state_pair
        for sample_index, canonical_displacement_m in enumerate(displacements_m):
            lower_to_upper_rate_s_inv = (
                event_flux / state_concentrations_mol_m3[lower_state_index]
            )
            upper_to_lower_rate_s_inv = (
                event_flux / state_concentrations_mol_m3[upper_state_index]
            )
            event_base_label = (
                f"trajectory_pair:{remapped_labels[lower_state_index]}->"
                f"{remapped_labels[upper_state_index]}:{sample_index}"
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=lower_state_index,
                    to_state_index=upper_state_index,
                    rate_s_inv=float(lower_to_upper_rate_s_inv),
                    charge_displacement_m=_as_displacement_tuple(
                        canonical_displacement_m,
                    ),
                    label=f"{event_base_label}:forward",
                    family_label="trajectory_transition",
                ),
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=upper_state_index,
                    to_state_index=lower_state_index,
                    rate_s_inv=float(upper_to_lower_rate_s_inv),
                    charge_displacement_m=_as_displacement_tuple(
                        -canonical_displacement_m,
                    ),
                    label=f"{event_base_label}:reverse",
                    family_label="trajectory_transition",
                ),
            )

    for state_index, displacements_m in sorted(self_samples_by_state.items()):
        self_rate_s_inv = event_flux / state_concentrations_mol_m3[state_index]
        for sample_index, charge_displacement_m in enumerate(displacements_m):
            event_base_label = (
                f"trajectory_self:{remapped_labels[state_index]}:{sample_index}"
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index,
                    to_state_index=state_index,
                    rate_s_inv=float(self_rate_s_inv),
                    charge_displacement_m=_as_displacement_tuple(
                        charge_displacement_m,
                    ),
                    label=f"{event_base_label}:plus",
                    family_label="trajectory_self_displacement",
                ),
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index,
                    to_state_index=state_index,
                    rate_s_inv=float(self_rate_s_inv),
                    charge_displacement_m=_as_displacement_tuple(
                        -charge_displacement_m,
                    ),
                    label=f"{event_base_label}:minus",
                    family_label="trajectory_self_displacement",
                ),
            )
    return tuple(events)


def _validated_state_labels(state_labels: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(str(state_label) for state_label in state_labels)
    if not labels:
        raise ValueError("state_labels must be nonempty")
    if len(set(labels)) != len(labels):
        raise ValueError("state_labels must be unique")
    if any(not state_label for state_label in labels):
        raise ValueError("state_labels must not contain empty labels")
    return labels


def _validated_state_index_by_frame(
    state_index_by_frame: np.ndarray,
    state_count: int,
) -> np.ndarray:
    state_indices = np.asarray(state_index_by_frame, dtype=int)
    if state_indices.ndim != 1:
        raise ValueError("state_index_by_frame must be one-dimensional")
    if state_indices.shape[0] < 2:
        raise ValueError("state_index_by_frame must contain at least two frames")
    if int(np.min(state_indices)) < 0:
        raise ValueError("state_index_by_frame contains a negative state index")
    if int(np.max(state_indices)) >= state_count:
        raise ValueError("state_index_by_frame contains an invalid state index")
    return state_indices


def _validated_charge_displacements(
    charge_displacement_by_step_m: np.ndarray,
    expected_step_count: int,
) -> np.ndarray:
    charge_displacements = np.asarray(charge_displacement_by_step_m, dtype=float)
    if charge_displacements.shape != (expected_step_count, 3):
        raise ValueError(
            "charge_displacement_by_step_m must have shape "
            f"({expected_step_count}, 3)",
        )
    if not np.all(np.isfinite(charge_displacements)):
        raise ValueError("charge_displacement_by_step_m must be finite")
    return charge_displacements


def _remap_visited_states(
    state_labels: tuple[str, ...],
    state_indices: np.ndarray,
) -> tuple[dict[int, int], tuple[str, ...], np.ndarray]:
    visited_original_indices = tuple(
        sorted(int(state_index) for state_index in np.unique(state_indices)),
    )
    state_index_remap = {
        original_state_index: remapped_state_index
        for remapped_state_index, original_state_index in enumerate(
            visited_original_indices,
        )
    }
    remapped_states = np.asarray(
        tuple(
            state_index_remap[int(state_index)]
            for state_index in state_indices
        ),
        dtype=int,
    )
    remapped_labels = tuple(
        state_labels[original_state_index]
        for original_state_index in visited_original_indices
    )
    return state_index_remap, remapped_labels, remapped_states


def _state_concentrations_from_occupancy(
    residence_state_indices: np.ndarray,
    state_count: int,
    total_concentration_mol_m3: float,
) -> np.ndarray:
    occupancy_counts = np.bincount(
        residence_state_indices,
        minlength=state_count,
    )
    if np.any(occupancy_counts <= 0):
        raise ValueError("all remapped states must have positive residence count")
    occupancy_fractions = occupancy_counts.astype(float) / float(
        residence_state_indices.shape[0],
    )
    state_concentrations_mol_m3 = total_concentration_mol_m3 * occupancy_fractions
    if np.any(state_concentrations_mol_m3 <= 0.0):
        raise ValueError("all state concentrations must be positive")
    return state_concentrations_mol_m3


def _positive_float(value: float, label: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return parsed_value


def _nonnegative_float(value: float, label: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return parsed_value


def _as_displacement_tuple(
    displacement: np.ndarray,
) -> tuple[float, float, float]:
    displacement_array = np.asarray(displacement, dtype=float)
    if displacement_array.shape != (3,):
        raise ValueError("displacement must have shape (3,)")
    if not np.all(np.isfinite(displacement_array)):
        raise ValueError("displacement must be finite")
    return (
        float(displacement_array[0]),
        float(displacement_array[1]),
        float(displacement_array[2]),
    )
