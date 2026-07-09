"""Trajectory primitive projection and conductivity oracle extraction.

This module turns observed trajectories into finite projected conductivity
primitives: occupancies, reversible transition rates, residence times,
charge-displacement moments, self-current tensors, and the Markov-additive
readout.  Descriptor recipe predictions are evaluated against these primitive
objects by reporting projection gaps and recipe gaps separately.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import tarfile
from typing import Mapping, Sequence

import numpy as np

from constants import N_A
from conductivity.finite_markov_additive_green_kubo import (
    MarkovAdditiveConductivityInput,
    MarkovAdditiveConductivityResult,
    MarkovAdditiveEvent,
    compute_markov_additive_green_kubo_conductivity,
)
from conductivity.fm_md.atomistic_io import Element, _ATOMIC_MASS


DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M = 0.0
ANGSTROM_TO_M = 1.0e-10
PICSECOND_TO_S = 1.0e-12
LIPF6_CHARGED_CENTER_COUNT = 2.0  # LiPF6 contributes one Li+ and one PF6- center.
MINIMUM_AGGREGATE_COUNTERION_COUNT = 2  # Aggregate label requires multiple counterions.
FRAME_INTERVAL_RELATIVE_TOLERANCE = 1.0e-9  # Floating header-time comparison tolerance.
PF6_ANION_ELEMENT_SEQUENCE = ("P", "F", "F", "F", "F", "F", "F")


@dataclass(frozen=True)
class TrajectoryPrimitiveTargetProcessInput:
    """Input for a trajectory-derived finite process target."""

    state_labels: tuple[str, ...]
    state_index_by_frame: np.ndarray
    charge_displacement_by_step_m: np.ndarray
    dt_s: float
    total_transport_concentration_mol_m3: float
    temperature_K: float
    displacement_zero_tolerance_m: float = DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M


@dataclass(frozen=True)
class TrajectoryPrimitiveTargetProcessDiagnostics:
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
class TrajectoryPrimitiveTargetProcessResult:
    markov_input: MarkovAdditiveConductivityInput
    conductivity_result: MarkovAdditiveConductivityResult
    diagnostics: TrajectoryPrimitiveTargetProcessDiagnostics
    state_index_remap: Mapping[int, int]


@dataclass(frozen=True)
class TrajectoryMarkovAdditiveSampleInput:
    """Parallel-center trajectory samples for empirical finite-process targets.

    This form is for many independent charged transport centers observed in one
    trajectory.  Each center contributes frame occupancies and frame-to-frame
    transition/displacement samples.  The concentration scale is the total
    concentration of all sampled charged centers.
    """

    state_labels: tuple[str, ...]
    occupancy_state_index_by_observation: np.ndarray
    from_state_index_by_step: np.ndarray
    to_state_index_by_step: np.ndarray
    charge_displacement_by_step_m: np.ndarray
    dt_s: float
    total_transport_concentration_mol_m3: float
    temperature_K: float
    displacement_zero_tolerance_m: float = DEFAULT_DISPLACEMENT_ZERO_TOLERANCE_M


@dataclass(frozen=True)
class PF6ZenodoTrajectoryLayout:
    expected_atom_count: int
    ec_molecule_count: int
    ec_atoms_per_molecule: int
    emc_molecule_count: int
    emc_atoms_per_molecule: int
    pf6_molecule_count: int
    pf6_atoms_per_molecule: int
    li_atom_count: int


@dataclass(frozen=True)
class PF6AssociationCutoffs:
    contact_pair_max_distance_A: float
    solvent_separated_pair_max_distance_A: float
    aggregate_counterion_count: int


@dataclass(frozen=True)
class PF6TrajectoryPrimitiveTargetInput:
    system_id: str
    archive_path: Path
    member_name: str
    layout: PF6ZenodoTrajectoryLayout
    association_cutoffs: PF6AssociationCutoffs
    max_frames: int
    frame_stride: int
    block_count: int
    temperature_K: float
    expected_frame_interval_ps: float


@dataclass(frozen=True)
class PF6TrajectoryPrimitiveTargetDiagnostics:
    frame_count: int
    frame_stride: int
    raw_frame_interval_ps: float
    effective_frame_interval_ps: float
    first_time_ps: float
    last_time_ps: float
    mean_box_length_A: float
    salt_concentration_mol_m3: float
    charged_center_concentration_mol_m3: float
    li_state_counts: Mapping[str, int]
    pf6_state_counts: Mapping[str, int]
    transition_sample_count: int


@dataclass(frozen=True)
class PF6TrajectoryPrimitiveTargetResult:
    sample_input: TrajectoryMarkovAdditiveSampleInput
    process_result: TrajectoryPrimitiveTargetProcessResult
    diagnostics: PF6TrajectoryPrimitiveTargetDiagnostics
    primitive_target_artifact: "TrajectoryPrimitiveTargetArtifact"


@dataclass(frozen=True)
class TrajectoryDisplacementMomentTarget:
    sample_count: int
    mean_displacement_m: tuple[float, float, float]
    mean_squared_axis_displacement_m2: tuple[float, float, float]
    mean_squared_displacement_m2: float


@dataclass(frozen=True)
class TrajectoryBlockPrimitiveTarget:
    block_index: int
    frame_count: int
    step_count: int
    state_concentrations_mol_m3: Mapping[str, float]
    state_occupancy_fractions: Mapping[str, float]
    transition_rates_s_inv: Mapping[str, float]
    transition_rate_targets_validated: bool
    transition_fluxes_mol_m3_s: Mapping[str, float]
    residence_times_s: Mapping[str, float]
    displacement_moments_by_family: Mapping[str, TrajectoryDisplacementMomentTarget]
    markov_additive_sigma_mS_cm: float
    markov_direct_sigma_mS_cm: float
    markov_corrector_sigma_mS_cm: float


@dataclass(frozen=True)
class TrajectoryPrimitiveTargetArtifact:
    system_id: str
    frame_count: int
    dt_s: float
    frame_stride: int
    block_count: int
    state_concentrations_mol_m3: Mapping[str, float]
    state_occupancy_fractions: Mapping[str, float]
    transition_rates_s_inv: Mapping[str, float]
    transition_rate_targets_validated: bool
    transition_fluxes_mol_m3_s: Mapping[str, float]
    residence_times_s: Mapping[str, float]
    displacement_moments_by_family: Mapping[str, TrajectoryDisplacementMomentTarget]
    displacement_moment_targets_validated: bool
    markov_additive_sigma_mS_cm: float
    markov_direct_sigma_mS_cm: float
    markov_corrector_sigma_mS_cm: float
    markov_additive_sigma_validated: bool
    block_targets: tuple[TrajectoryBlockPrimitiveTarget, ...]
    block_state_concentration_standard_errors_mol_m3: Mapping[str, float]
    block_transition_rate_standard_errors_s_inv: Mapping[str, float]
    block_displacement_moment_standard_errors_m2: Mapping[str, float]
    block_sigma_standard_error_mS_cm: float


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
    diagnostics: TrajectoryPrimitiveTargetProcessDiagnostics


@dataclass(frozen=True)
class _PF6CenterFrame:
    li_centers_A: np.ndarray
    pf6_centers_A: np.ndarray
    box_length_A: float
    time_ps: float


def compute_trajectory_primitive_target_conductivity(
    process_input: TrajectoryPrimitiveTargetProcessInput,
) -> TrajectoryPrimitiveTargetProcessResult:
    """Estimate a reversible finite event process and evaluate conductivity."""

    markov_input, diagnostics, state_index_remap = (
        build_trajectory_primitive_target_markov_input(process_input)
    )
    conductivity_result = compute_markov_additive_green_kubo_conductivity(
        markov_input,
    )
    return TrajectoryPrimitiveTargetProcessResult(
        markov_input=markov_input,
        conductivity_result=conductivity_result,
        diagnostics=diagnostics,
        state_index_remap=state_index_remap,
    )


def compute_sampled_trajectory_markov_additive_conductivity(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
) -> TrajectoryPrimitiveTargetProcessResult:
    """Evaluate a reversible finite event target from parallel samples."""

    markov_input, diagnostics, state_index_remap = (
        build_sampled_trajectory_markov_additive_input(sample_input)
    )
    conductivity_result = compute_markov_additive_green_kubo_conductivity(
        markov_input,
    )
    return TrajectoryPrimitiveTargetProcessResult(
        markov_input=markov_input,
        conductivity_result=conductivity_result,
        diagnostics=diagnostics,
        state_index_remap=state_index_remap,
    )


def project_sampled_trajectory_to_generator_primitives(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
) -> ProjectedGeneratorPrimitiveSet:
    """Project sampled trajectory observations into GK generator primitives.

    The returned object is the generic projected-generator target:
    equilibrium populations, symmetric reactive fluxes, conditional
    charge-displacement moments, within-state self-current tensors, and the
    Markov-additive conductivity readout from the same samples.
    """

    process_result = compute_sampled_trajectory_markov_additive_conductivity(
        sample_input,
    )
    markov_input = process_result.markov_input
    state_concentrations = _state_concentration_targets(
        sample_input.state_labels,
        markov_input,
    )
    return ProjectedGeneratorPrimitiveSet(
        state_labels=markov_input.state_labels,
        state_concentrations_mol_m3=state_concentrations,
        state_occupancy_fractions=_state_occupancy_fraction_targets(sample_input),
        reactive_fluxes=_projected_reactive_fluxes(markov_input),
        conditional_displacement_moments=(
            _projected_conditional_displacement_moments(sample_input)
        ),
        self_current_tensors=_projected_self_current_tensors(
            sample_input,
            state_concentrations,
        ),
        markov_input=markov_input,
        markov_conductivity_result=process_result.conductivity_result,
        diagnostics=process_result.diagnostics,
    )


def build_trajectory_primitive_target_markov_input(
    process_input: TrajectoryPrimitiveTargetProcessInput,
) -> tuple[
    MarkovAdditiveConductivityInput,
    TrajectoryPrimitiveTargetProcessDiagnostics,
    dict[int, int],
]:
    """Build a reversible finite Markov-additive input from trajectory data."""

    state_labels = _validated_state_labels(process_input.state_labels)
    state_index_by_frame = _validated_state_index_by_frame(
        process_input.state_index_by_frame,
        len(state_labels),
    )
    charge_displacements = _validated_charge_displacements(
        process_input.charge_displacement_by_step_m,
        state_index_by_frame.shape[0] - 1,
    )
    dt_s = _positive_float(process_input.dt_s, "dt_s")
    total_concentration_mol_m3 = _positive_float(
        process_input.total_transport_concentration_mol_m3,
        "total_transport_concentration_mol_m3",
    )
    temperature_K = _positive_float(process_input.temperature_K, "temperature_K")
    displacement_zero_tolerance_m = _nonnegative_float(
        process_input.displacement_zero_tolerance_m,
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

    pair_samples_by_state_pair: dict[tuple[int, int], list[np.ndarray]] = defaultdict(
        list
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
        pair_samples_by_state_pair[(lower_state_index, upper_state_index)].append(
            canonical_displacement_m
        )
        transition_sample_count += 1

    event_flux_mol_m3_s = total_concentration_mol_m3 / (2.0 * step_count * dt_s)
    events = _trajectory_events_from_samples(
        pair_samples_by_state_pair,
        self_samples_by_state,
        remapped_labels,
        state_concentrations_mol_m3,
        event_flux_mol_m3_s,
    )
    if not events:
        raise ValueError(
            "trajectory target produced no nonzero Markov-additive events",
        )

    diagnostics = TrajectoryPrimitiveTargetProcessDiagnostics(
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


def build_sampled_trajectory_markov_additive_input(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
) -> tuple[
    MarkovAdditiveConductivityInput,
    TrajectoryPrimitiveTargetProcessDiagnostics,
    dict[int, int],
]:
    """Build a finite process from many center-wise trajectory samples."""

    state_labels = _validated_state_labels(sample_input.state_labels)
    occupancy_state_indices = _validated_state_index_by_frame(
        sample_input.occupancy_state_index_by_observation,
        len(state_labels),
    )
    from_state_indices = _validated_sample_state_indices(
        sample_input.from_state_index_by_step,
        len(state_labels),
        "from_state_index_by_step",
    )
    to_state_indices = _validated_sample_state_indices(
        sample_input.to_state_index_by_step,
        len(state_labels),
        "to_state_index_by_step",
    )
    if from_state_indices.shape != to_state_indices.shape:
        raise ValueError(
            "from_state_index_by_step and to_state_index_by_step must have "
            "the same shape",
        )
    charge_displacements = _validated_charge_displacements(
        sample_input.charge_displacement_by_step_m,
        int(from_state_indices.shape[0]),
    )
    dt_s = _positive_float(sample_input.dt_s, "dt_s")
    total_concentration_mol_m3 = _positive_float(
        sample_input.total_transport_concentration_mol_m3,
        "total_transport_concentration_mol_m3",
    )
    temperature_K = _positive_float(sample_input.temperature_K, "temperature_K")
    displacement_zero_tolerance_m = _nonnegative_float(
        sample_input.displacement_zero_tolerance_m,
        "displacement_zero_tolerance_m",
    )

    all_state_observations = np.concatenate(
        (occupancy_state_indices, from_state_indices, to_state_indices),
    )
    state_index_remap, remapped_labels, remapped_observations = _remap_visited_states(
        state_labels, all_state_observations
    )
    occupancy_count = int(occupancy_state_indices.shape[0])
    step_count = int(from_state_indices.shape[0])
    remapped_occupancy_state_indices = remapped_observations[:occupancy_count]
    remapped_from_state_indices = remapped_observations[
        occupancy_count : occupancy_count + step_count
    ]
    remapped_to_state_indices = remapped_observations[occupancy_count + step_count :]

    state_concentrations_mol_m3 = _state_concentrations_from_occupancy(
        remapped_occupancy_state_indices,
        len(remapped_labels),
        total_concentration_mol_m3,
    )
    pair_samples_by_state_pair: dict[tuple[int, int], list[np.ndarray]] = defaultdict(
        list
    )
    self_samples_by_state: dict[int, list[np.ndarray]] = defaultdict(list)
    transition_sample_count = 0
    self_displacement_sample_count = 0
    for step_index, charge_displacement_m in enumerate(charge_displacements):
        from_state_index = int(remapped_from_state_indices[step_index])
        to_state_index = int(remapped_to_state_indices[step_index])
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
        pair_samples_by_state_pair[(lower_state_index, upper_state_index)].append(
            canonical_displacement_m
        )
        transition_sample_count += 1

    event_flux_mol_m3_s = total_concentration_mol_m3 / (2.0 * step_count * dt_s)
    events = _trajectory_events_from_samples(
        pair_samples_by_state_pair,
        self_samples_by_state,
        remapped_labels,
        state_concentrations_mol_m3,
        event_flux_mol_m3_s,
    )
    if not events:
        raise ValueError(
            "sampled trajectory target produced no nonzero Markov-additive events",
        )

    diagnostics = TrajectoryPrimitiveTargetProcessDiagnostics(
        original_state_count=len(state_labels),
        visited_state_count=len(remapped_labels),
        frame_count=int(occupancy_state_indices.shape[0]),
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


def _projected_reactive_fluxes(
    markov_input: MarkovAdditiveConductivityInput,
) -> tuple[ProjectedGeneratorReactiveFlux, ...]:
    state_concentrations = np.asarray(
        markov_input.state_concentrations_mol_m3,
        dtype=float,
    )
    flux_by_ordered_pair: dict[tuple[int, int], float] = defaultdict(float)
    rate_by_ordered_pair: dict[tuple[int, int], float] = defaultdict(float)
    for event in markov_input.events:
        if event.from_state_index == event.to_state_index:
            continue
        ordered_pair = (event.from_state_index, event.to_state_index)
        event_flux = state_concentrations[event.from_state_index] * event.rate_s_inv
        flux_by_ordered_pair[ordered_pair] += float(event_flux)
        rate_by_ordered_pair[ordered_pair] += float(event.rate_s_inv)
    reactive_fluxes: list[ProjectedGeneratorReactiveFlux] = []
    unordered_pairs = {
        (
            min(first_state_index, second_state_index),
            max(first_state_index, second_state_index),
        )
        for first_state_index, second_state_index in flux_by_ordered_pair
    }
    for lower_state_index, upper_state_index in sorted(unordered_pairs):
        forward_flux = flux_by_ordered_pair[(lower_state_index, upper_state_index)]
        reverse_flux = flux_by_ordered_pair[(upper_state_index, lower_state_index)]
        tolerance = math.sqrt(np.finfo(float).eps) * max(
            1.0,
            abs(forward_flux),
            abs(reverse_flux),
        )
        if abs(forward_flux - reverse_flux) > tolerance:
            raise ValueError(
                "reactive flux samples are not detailed-balanced for "
                f"{markov_input.state_labels[lower_state_index]} <-> "
                f"{markov_input.state_labels[upper_state_index]}",
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
            ),
        )
    return tuple(reactive_fluxes)


def _projected_conditional_displacement_moments(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
) -> tuple[ProjectedGeneratorConditionalMoment, ...]:
    state_labels = _validated_state_labels(sample_input.state_labels)
    from_state_indices = _validated_sample_state_indices(
        sample_input.from_state_index_by_step,
        len(state_labels),
        "from_state_index_by_step",
    )
    to_state_indices = _validated_sample_state_indices(
        sample_input.to_state_index_by_step,
        len(state_labels),
        "to_state_index_by_step",
    )
    charge_displacements = _validated_charge_displacements(
        sample_input.charge_displacement_by_step_m,
        int(from_state_indices.shape[0]),
    )
    samples_by_transition: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for sample_index, charge_displacement_m in enumerate(charge_displacements):
        from_state_index = int(from_state_indices[sample_index])
        to_state_index = int(to_state_indices[sample_index])
        if from_state_index == to_state_index:
            continue
        samples_by_transition[(from_state_index, to_state_index)].append(
            charge_displacement_m,
        )
    conditional_moments: list[ProjectedGeneratorConditionalMoment] = []
    for transition_key in sorted(samples_by_transition):
        displacement_samples = np.asarray(
            samples_by_transition[transition_key],
            dtype=float,
        )
        mean_displacement = np.mean(displacement_samples, axis=0)
        second_moment = np.einsum(
            "ni,nj->ij",
            displacement_samples,
            displacement_samples,
        ) / float(displacement_samples.shape[0])
        covariance = second_moment - np.outer(mean_displacement, mean_displacement)
        _validate_positive_semidefinite_matrix(
            covariance,
            "conditional_displacement_moment.covariance_m2",
        )
        from_state_index, to_state_index = transition_key
        conditional_moments.append(
            ProjectedGeneratorConditionalMoment(
                from_state_label=state_labels[from_state_index],
                to_state_label=state_labels[to_state_index],
                sample_count=int(displacement_samples.shape[0]),
                mean_charge_displacement_m=tuple(
                    float(component) for component in mean_displacement
                ),
                second_moment_m2=_matrix_to_tuple(second_moment),
                covariance_m2=_matrix_to_tuple(covariance),
            ),
        )
    return tuple(conditional_moments)


def _projected_self_current_tensors(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
    state_concentrations_mol_m3: Mapping[str, float],
) -> tuple[ProjectedGeneratorSelfCurrentTensor, ...]:
    state_labels = _validated_state_labels(sample_input.state_labels)
    from_state_indices = _validated_sample_state_indices(
        sample_input.from_state_index_by_step,
        len(state_labels),
        "from_state_index_by_step",
    )
    to_state_indices = _validated_sample_state_indices(
        sample_input.to_state_index_by_step,
        len(state_labels),
        "to_state_index_by_step",
    )
    charge_displacements = _validated_charge_displacements(
        sample_input.charge_displacement_by_step_m,
        int(from_state_indices.shape[0]),
    )
    dt_s = _positive_float(sample_input.dt_s, "dt_s")
    displacement_zero_tolerance_m = _nonnegative_float(
        sample_input.displacement_zero_tolerance_m,
        "displacement_zero_tolerance_m",
    )
    self_samples_by_state: dict[int, list[np.ndarray]] = defaultdict(list)
    for sample_index, charge_displacement_m in enumerate(charge_displacements):
        from_state_index = int(from_state_indices[sample_index])
        to_state_index = int(to_state_indices[sample_index])
        if from_state_index != to_state_index:
            continue
        if (
            float(np.linalg.norm(charge_displacement_m))
            <= displacement_zero_tolerance_m
        ):
            continue
        self_samples_by_state[from_state_index].append(charge_displacement_m)

    self_current_tensors: list[ProjectedGeneratorSelfCurrentTensor] = []
    for state_index in sorted(self_samples_by_state):
        displacement_samples = np.asarray(
            self_samples_by_state[state_index],
            dtype=float,
        )
        diffusion_tensor = np.einsum(
            "ni,nj->ij",
            displacement_samples,
            displacement_samples,
        ) / (2.0 * dt_s * float(displacement_samples.shape[0]))
        _validate_positive_semidefinite_matrix(
            diffusion_tensor,
            "self_current_tensor.diffusion_tensor_m2_s",
        )
        state_label = state_labels[state_index]
        self_current_tensors.append(
            ProjectedGeneratorSelfCurrentTensor(
                state_label=state_label,
                sample_count=int(displacement_samples.shape[0]),
                concentration_mol_m3=float(
                    state_concentrations_mol_m3[state_label],
                ),
                diffusion_tensor_m2_s=_matrix_to_tuple(diffusion_tensor),
            ),
        )
    return tuple(self_current_tensors)


def _validate_positive_semidefinite_matrix(
    matrix: np.ndarray,
    label: str,
) -> None:
    if matrix.shape != (3, 3):
        raise ValueError(f"{label} must have shape (3, 3), got {matrix.shape}")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{label} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = math.sqrt(np.finfo(float).eps) * max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            f"{label} minimum eigenvalue {minimum_eigenvalue} is below "
            f"tolerance {-tolerance}",
        )


def _matrix_to_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {matrix.shape}")
    return tuple(
        tuple(float(matrix[row_index, column_index]) for column_index in range(3))
        for row_index in range(3)
    )


def compute_pf6_trajectory_primitive_targets(
    target_input: PF6TrajectoryPrimitiveTargetInput,
) -> PF6TrajectoryPrimitiveTargetResult:
    """Stream the Zenodo PF6 trajectory into empirical c, Q, d targets."""

    validated_input = _validated_pf6_target_input(target_input)
    state_labels = (
        "free_ion_center:Li+",
        "contact_pair_center:Li+",
        "solvent_separated_pair_center:Li+",
        "internal_polarization_center:Li+",
        "free_ion_center:PF6-",
        "contact_pair_center:PF6-",
        "solvent_separated_pair_center:PF6-",
        "internal_polarization_center:PF6-",
    )
    state_index_by_label = {
        state_label: state_index for state_index, state_label in enumerate(state_labels)
    }

    frame_stream = iter(_stream_pf6_center_frames(validated_input))
    first_frame = next(frame_stream, None)
    if first_frame is None:
        raise ValueError("PF6 trajectory stream produced no frames")

    previous_li_centers_unwrapped_A = first_frame.li_centers_A.copy()
    previous_pf6_centers_unwrapped_A = first_frame.pf6_centers_A.copy()
    previous_li_states = _pf6_li_state_indices(
        first_frame.li_centers_A,
        first_frame.pf6_centers_A,
        first_frame.box_length_A,
        validated_input.association_cutoffs,
        state_index_by_label,
    )
    previous_pf6_states = _pf6_anion_state_indices(
        first_frame.li_centers_A,
        first_frame.pf6_centers_A,
        first_frame.box_length_A,
        validated_input.association_cutoffs,
        state_index_by_label,
    )

    occupancy_state_indices: list[int] = []
    from_state_indices: list[int] = []
    to_state_indices: list[int] = []
    charge_displacements_m: list[np.ndarray] = []
    li_state_counts = {state_label: 0 for state_label in state_labels[:4]}
    pf6_state_counts = {state_label: 0 for state_label in state_labels[4:]}
    box_lengths_A = [first_frame.box_length_A]
    times_ps = [first_frame.time_ps]

    _append_center_occupancies(
        previous_li_states,
        previous_pf6_states,
        occupancy_state_indices,
        li_state_counts,
        pf6_state_counts,
        state_labels,
    )

    for current_frame in frame_stream:
        current_li_centers_unwrapped_A = _unwrap_center_positions_A(
            current_frame.li_centers_A,
            previous_li_centers_unwrapped_A,
            current_frame.box_length_A,
        )
        current_pf6_centers_unwrapped_A = _unwrap_center_positions_A(
            current_frame.pf6_centers_A,
            previous_pf6_centers_unwrapped_A,
            current_frame.box_length_A,
        )
        current_li_states = _pf6_li_state_indices(
            current_frame.li_centers_A,
            current_frame.pf6_centers_A,
            current_frame.box_length_A,
            validated_input.association_cutoffs,
            state_index_by_label,
        )
        current_pf6_states = _pf6_anion_state_indices(
            current_frame.li_centers_A,
            current_frame.pf6_centers_A,
            current_frame.box_length_A,
            validated_input.association_cutoffs,
            state_index_by_label,
        )

        _append_center_steps(
            previous_li_states,
            current_li_states,
            previous_li_centers_unwrapped_A,
            current_li_centers_unwrapped_A,
            1,
            from_state_indices,
            to_state_indices,
            charge_displacements_m,
        )
        _append_center_steps(
            previous_pf6_states,
            current_pf6_states,
            previous_pf6_centers_unwrapped_A,
            current_pf6_centers_unwrapped_A,
            -1,
            from_state_indices,
            to_state_indices,
            charge_displacements_m,
        )
        _append_center_occupancies(
            current_li_states,
            current_pf6_states,
            occupancy_state_indices,
            li_state_counts,
            pf6_state_counts,
            state_labels,
        )

        previous_li_centers_unwrapped_A = current_li_centers_unwrapped_A
        previous_pf6_centers_unwrapped_A = current_pf6_centers_unwrapped_A
        previous_li_states = current_li_states
        previous_pf6_states = current_pf6_states
        box_lengths_A.append(current_frame.box_length_A)
        times_ps.append(current_frame.time_ps)

    if len(times_ps) < 2:
        raise ValueError("PF6 trajectory target extraction needs at least two frames")

    raw_frame_interval_ps = _validated_frame_interval_ps(
        tuple(times_ps),
        validated_input.expected_frame_interval_ps,
        validated_input.frame_stride,
    )
    effective_frame_interval_ps = raw_frame_interval_ps * validated_input.frame_stride
    mean_box_length_A = float(np.mean(np.asarray(box_lengths_A, dtype=float)))
    mean_box_volume_m3 = (mean_box_length_A * ANGSTROM_TO_M) ** 3
    salt_concentration_mol_m3 = (
        validated_input.layout.li_atom_count / N_A / mean_box_volume_m3
    )
    charged_center_concentration_mol_m3 = (
        LIPF6_CHARGED_CENTER_COUNT
        * validated_input.layout.li_atom_count
        / N_A
        / mean_box_volume_m3
    )
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=state_labels,
        occupancy_state_index_by_observation=np.asarray(
            occupancy_state_indices,
            dtype=int,
        ),
        from_state_index_by_step=np.asarray(from_state_indices, dtype=int),
        to_state_index_by_step=np.asarray(to_state_indices, dtype=int),
        charge_displacement_by_step_m=np.asarray(
            charge_displacements_m,
            dtype=float,
        ),
        dt_s=effective_frame_interval_ps * PICSECOND_TO_S,
        total_transport_concentration_mol_m3=charged_center_concentration_mol_m3,
        temperature_K=validated_input.temperature_K,
    )
    process_result = compute_sampled_trajectory_markov_additive_conductivity(
        sample_input,
    )
    primitive_target_artifact = _primitive_target_artifact_from_sample_input(
        validated_input.system_id,
        sample_input,
        process_result,
        len(times_ps),
        int(validated_input.layout.li_atom_count * LIPF6_CHARGED_CENTER_COUNT),
        validated_input.block_count,
        validated_input.frame_stride,
    )
    diagnostics = PF6TrajectoryPrimitiveTargetDiagnostics(
        frame_count=len(times_ps),
        frame_stride=validated_input.frame_stride,
        raw_frame_interval_ps=raw_frame_interval_ps,
        effective_frame_interval_ps=effective_frame_interval_ps,
        first_time_ps=float(times_ps[0]),
        last_time_ps=float(times_ps[-1]),
        mean_box_length_A=mean_box_length_A,
        salt_concentration_mol_m3=float(salt_concentration_mol_m3),
        charged_center_concentration_mol_m3=float(
            charged_center_concentration_mol_m3,
        ),
        li_state_counts=li_state_counts,
        pf6_state_counts=pf6_state_counts,
        transition_sample_count=int(len(from_state_indices)),
    )
    return PF6TrajectoryPrimitiveTargetResult(
        sample_input=sample_input,
        process_result=process_result,
        diagnostics=diagnostics,
        primitive_target_artifact=primitive_target_artifact,
    )


def _validated_pf6_target_input(
    target_input: PF6TrajectoryPrimitiveTargetInput,
) -> PF6TrajectoryPrimitiveTargetInput:
    system_id = str(target_input.system_id)
    if not system_id:
        raise ValueError("system_id must be nonempty")
    archive_path = Path(target_input.archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(
            f"PF6 trajectory archive does not exist: {archive_path}"
        )
    member_name = str(target_input.member_name)
    if not member_name:
        raise ValueError("member_name must be nonempty")
    layout = target_input.layout
    _validate_positive_int(layout.expected_atom_count, "expected_atom_count")
    _validate_positive_int(layout.ec_molecule_count, "ec_molecule_count")
    _validate_positive_int(layout.ec_atoms_per_molecule, "ec_atoms_per_molecule")
    _validate_positive_int(layout.emc_molecule_count, "emc_molecule_count")
    _validate_positive_int(layout.emc_atoms_per_molecule, "emc_atoms_per_molecule")
    _validate_positive_int(layout.pf6_molecule_count, "pf6_molecule_count")
    _validate_positive_int(layout.pf6_atoms_per_molecule, "pf6_atoms_per_molecule")
    _validate_positive_int(layout.li_atom_count, "li_atom_count")
    expected_total_atom_count = (
        layout.ec_molecule_count * layout.ec_atoms_per_molecule
        + layout.emc_molecule_count * layout.emc_atoms_per_molecule
        + layout.pf6_molecule_count * layout.pf6_atoms_per_molecule
        + layout.li_atom_count
    )
    if layout.expected_atom_count != expected_total_atom_count:
        raise ValueError(
            "PF6 layout atom count mismatch: expected_atom_count="
            f"{layout.expected_atom_count}, block total={expected_total_atom_count}",
        )
    if layout.pf6_atoms_per_molecule != len(PF6_ANION_ELEMENT_SEQUENCE):
        raise ValueError(
            "PF6 layout requires "
            f"{len(PF6_ANION_ELEMENT_SEQUENCE)} atoms per PF6 molecule",
        )
    cutoffs = target_input.association_cutoffs
    contact_pair_max_distance_A = _positive_float(
        cutoffs.contact_pair_max_distance_A,
        "contact_pair_max_distance_A",
    )
    solvent_separated_pair_max_distance_A = _positive_float(
        cutoffs.solvent_separated_pair_max_distance_A,
        "solvent_separated_pair_max_distance_A",
    )
    if contact_pair_max_distance_A >= solvent_separated_pair_max_distance_A:
        raise ValueError(
            "contact_pair_max_distance_A must be smaller than "
            "solvent_separated_pair_max_distance_A",
        )
    if cutoffs.aggregate_counterion_count < MINIMUM_AGGREGATE_COUNTERION_COUNT:
        raise ValueError(
            "aggregate_counterion_count must be at least "
            f"{MINIMUM_AGGREGATE_COUNTERION_COUNT}",
        )
    max_frames = _validate_positive_int(target_input.max_frames, "max_frames")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    frame_stride = _validate_positive_int(target_input.frame_stride, "frame_stride")
    block_count = _validate_positive_int(target_input.block_count, "block_count")
    if block_count > max_frames - 1:
        raise ValueError("block_count cannot exceed max_frames - 1")
    temperature_K = _positive_float(target_input.temperature_K, "temperature_K")
    expected_frame_interval_ps = _positive_float(
        target_input.expected_frame_interval_ps,
        "expected_frame_interval_ps",
    )
    return PF6TrajectoryPrimitiveTargetInput(
        system_id=system_id,
        archive_path=archive_path,
        member_name=member_name,
        layout=layout,
        association_cutoffs=PF6AssociationCutoffs(
            contact_pair_max_distance_A=contact_pair_max_distance_A,
            solvent_separated_pair_max_distance_A=(
                solvent_separated_pair_max_distance_A
            ),
            aggregate_counterion_count=int(cutoffs.aggregate_counterion_count),
        ),
        max_frames=max_frames,
        frame_stride=frame_stride,
        block_count=block_count,
        temperature_K=temperature_K,
        expected_frame_interval_ps=expected_frame_interval_ps,
    )


def _stream_pf6_center_frames(
    target_input: PF6TrajectoryPrimitiveTargetInput,
) -> tuple[_PF6CenterFrame, ...]:
    frames: list[_PF6CenterFrame] = []
    with tarfile.open(target_input.archive_path, mode="r:gz") as archive:
        trajectory_member = archive.extractfile(target_input.member_name)
        if trajectory_member is None:
            raise ValueError(
                f"Archive member {target_input.member_name!r} is not a file",
            )
        raw_frame_index = 0
        kept_frame_count = 0
        while kept_frame_count < target_input.max_frames:
            header_line = trajectory_member.readline()
            if not header_line:
                break
            header_fields = header_line.decode("utf-8").split()
            if len(header_fields) < 3:
                raise ValueError(
                    f"Malformed PF6 XYZ header at raw frame {raw_frame_index}: "
                    f"{header_line!r}",
                )
            atom_count = int(header_fields[0])
            box_length_A = float(header_fields[1])
            time_ps = float(header_fields[2])
            trajectory_member.readline()
            if atom_count != target_input.layout.expected_atom_count:
                raise ValueError(
                    f"PF6 frame {raw_frame_index} has {atom_count} atoms; "
                    f"expected {target_input.layout.expected_atom_count}",
                )
            if raw_frame_index % target_input.frame_stride == 0:
                element_symbols: list[str] = []
                positions_A = np.empty((atom_count, 3), dtype=float)
                for atom_index in range(atom_count):
                    atom_fields = trajectory_member.readline().decode("utf-8").split()
                    if len(atom_fields) < 4:
                        raise ValueError(
                            "Malformed PF6 atom record at raw frame "
                            f"{raw_frame_index}, atom {atom_index}",
                        )
                    element_symbols.append(atom_fields[0])
                    positions_A[atom_index, 0] = float(atom_fields[1])
                    positions_A[atom_index, 1] = float(atom_fields[2])
                    positions_A[atom_index, 2] = float(atom_fields[3])
                _validate_pf6_frame_elements(element_symbols, target_input.layout)
                frames.append(
                    _pf6_center_frame_from_positions(
                        positions_A,
                        float(box_length_A),
                        float(time_ps),
                        target_input.layout,
                    ),
                )
                kept_frame_count += 1
            else:
                for atom_index in range(atom_count):
                    skipped_line = trajectory_member.readline()
                    if not skipped_line:
                        raise ValueError(
                            f"PF6 trajectory ended inside raw frame {raw_frame_index}",
                        )
            raw_frame_index += 1
    if not frames:
        raise ValueError("PF6 trajectory stream produced no kept frames")
    return tuple(frames)


def _pf6_center_frame_from_positions(
    positions_A: np.ndarray,
    box_length_A: float,
    time_ps: float,
    layout: PF6ZenodoTrajectoryLayout,
) -> _PF6CenterFrame:
    block_slices = _pf6_layout_slices(layout)
    pf6_slice = block_slices["pf6"]
    li_slice = block_slices["li"]
    li_centers_A = positions_A[li_slice].copy()
    pf6_positions_A = positions_A[pf6_slice].reshape(
        layout.pf6_molecule_count,
        layout.pf6_atoms_per_molecule,
        3,
    )
    pf6_centers_A = np.empty((layout.pf6_molecule_count, 3), dtype=float)
    for pf6_index in range(layout.pf6_molecule_count):
        pf6_centers_A[pf6_index] = _pf6_anion_center_A(
            pf6_positions_A[pf6_index],
            box_length_A,
        )
    return _PF6CenterFrame(
        li_centers_A=_wrap_positions_A(li_centers_A, box_length_A),
        pf6_centers_A=_wrap_positions_A(pf6_centers_A, box_length_A),
        box_length_A=box_length_A,
        time_ps=time_ps,
    )


def _pf6_anion_center_A(
    pf6_positions_A: np.ndarray,
    box_length_A: float,
) -> np.ndarray:
    if pf6_positions_A.shape != (len(PF6_ANION_ELEMENT_SEQUENCE), 3):
        raise ValueError(
            "PF6 position block must have shape "
            f"({len(PF6_ANION_ELEMENT_SEQUENCE)}, 3)",
        )
    p_position_A = pf6_positions_A[0]
    unfolded_positions_A = pf6_positions_A.copy()
    for atom_index in range(1, pf6_positions_A.shape[0]):
        unfolded_positions_A[atom_index] = p_position_A + _minimum_image_displacement_A(
            pf6_positions_A[atom_index] - p_position_A,
            box_length_A,
        )
    atom_masses_g_mol = np.asarray(
        (
            _ATOMIC_MASS[Element.P],
            _ATOMIC_MASS[Element.F],
            _ATOMIC_MASS[Element.F],
            _ATOMIC_MASS[Element.F],
            _ATOMIC_MASS[Element.F],
            _ATOMIC_MASS[Element.F],
            _ATOMIC_MASS[Element.F],
        ),
        dtype=float,
    )
    return np.sum(
        unfolded_positions_A * atom_masses_g_mol[:, None],
        axis=0,
    ) / float(np.sum(atom_masses_g_mol))


def _validate_pf6_frame_elements(
    element_symbols: Sequence[str],
    layout: PF6ZenodoTrajectoryLayout,
) -> None:
    block_slices = _pf6_layout_slices(layout)
    ec_symbols = tuple(element_symbols[block_slices["ec"]])
    emc_symbols = tuple(element_symbols[block_slices["emc"]])
    pf6_symbols = tuple(element_symbols[block_slices["pf6"]])
    li_symbols = tuple(element_symbols[block_slices["li"]])
    expected_ec_unit = ("C", "C", "O", "O", "C", "H", "H", "H", "H", "O")
    expected_emc_unit = (
        "C",
        "O",
        "C",
        "O",
        "C",
        "C",
        "H",
        "H",
        "H",
        "O",
        "H",
        "H",
        "H",
        "H",
        "H",
    )
    expected_pf6_unit = PF6_ANION_ELEMENT_SEQUENCE
    if ec_symbols != expected_ec_unit * layout.ec_molecule_count:
        raise ValueError("EC atom ordering does not match PF6 trajectory layout")
    if emc_symbols != expected_emc_unit * layout.emc_molecule_count:
        raise ValueError("EMC atom ordering does not match PF6 trajectory layout")
    if pf6_symbols != expected_pf6_unit * layout.pf6_molecule_count:
        raise ValueError("PF6 atom ordering does not match trajectory layout")
    if li_symbols != ("Li",) * layout.li_atom_count:
        raise ValueError("Li atom ordering does not match PF6 trajectory layout")


def _pf6_layout_slices(
    layout: PF6ZenodoTrajectoryLayout,
) -> Mapping[str, slice]:
    ec_start = 0
    ec_stop = layout.ec_molecule_count * layout.ec_atoms_per_molecule
    emc_start = ec_stop
    emc_stop = emc_start + layout.emc_molecule_count * layout.emc_atoms_per_molecule
    pf6_start = emc_stop
    pf6_stop = pf6_start + layout.pf6_molecule_count * layout.pf6_atoms_per_molecule
    li_start = pf6_stop
    li_stop = li_start + layout.li_atom_count
    if li_stop != layout.expected_atom_count:
        raise ValueError("PF6 layout slices do not cover expected atom count")
    return {
        "ec": slice(ec_start, ec_stop),
        "emc": slice(emc_start, emc_stop),
        "pf6": slice(pf6_start, pf6_stop),
        "li": slice(li_start, li_stop),
    }


def _pf6_li_state_indices(
    li_centers_A: np.ndarray,
    pf6_centers_A: np.ndarray,
    box_length_A: float,
    cutoffs: PF6AssociationCutoffs,
    state_index_by_label: Mapping[str, int],
) -> np.ndarray:
    distances_A = _pair_distance_matrix_A(li_centers_A, pf6_centers_A, box_length_A)
    nearby_counts = np.sum(
        distances_A <= cutoffs.solvent_separated_pair_max_distance_A,
        axis=1,
    )
    nearest_distances_A = np.min(distances_A, axis=1)
    state_indices = np.empty(li_centers_A.shape[0], dtype=int)
    for li_index, nearest_distance_A in enumerate(nearest_distances_A):
        if int(nearby_counts[li_index]) >= cutoffs.aggregate_counterion_count:
            state_label = "internal_polarization_center:Li+"
        elif nearest_distance_A <= cutoffs.contact_pair_max_distance_A:
            state_label = "contact_pair_center:Li+"
        elif nearest_distance_A <= cutoffs.solvent_separated_pair_max_distance_A:
            state_label = "solvent_separated_pair_center:Li+"
        else:
            state_label = "free_ion_center:Li+"
        state_indices[li_index] = state_index_by_label[state_label]
    return state_indices


def _pf6_anion_state_indices(
    li_centers_A: np.ndarray,
    pf6_centers_A: np.ndarray,
    box_length_A: float,
    cutoffs: PF6AssociationCutoffs,
    state_index_by_label: Mapping[str, int],
) -> np.ndarray:
    distances_A = _pair_distance_matrix_A(pf6_centers_A, li_centers_A, box_length_A)
    nearby_counts = np.sum(
        distances_A <= cutoffs.solvent_separated_pair_max_distance_A,
        axis=1,
    )
    nearest_distances_A = np.min(distances_A, axis=1)
    state_indices = np.empty(pf6_centers_A.shape[0], dtype=int)
    for pf6_index, nearest_distance_A in enumerate(nearest_distances_A):
        if int(nearby_counts[pf6_index]) >= cutoffs.aggregate_counterion_count:
            state_label = "internal_polarization_center:PF6-"
        elif nearest_distance_A <= cutoffs.contact_pair_max_distance_A:
            state_label = "contact_pair_center:PF6-"
        elif nearest_distance_A <= cutoffs.solvent_separated_pair_max_distance_A:
            state_label = "solvent_separated_pair_center:PF6-"
        else:
            state_label = "free_ion_center:PF6-"
        state_indices[pf6_index] = state_index_by_label[state_label]
    return state_indices


def _pair_distance_matrix_A(
    centers_a_A: np.ndarray,
    centers_b_A: np.ndarray,
    box_length_A: float,
) -> np.ndarray:
    displacement_A = centers_a_A[:, None, :] - centers_b_A[None, :, :]
    displacement_A = _minimum_image_displacement_A(displacement_A, box_length_A)
    return np.linalg.norm(displacement_A, axis=2)


def _minimum_image_displacement_A(
    displacement_A: np.ndarray,
    box_length_A: float,
) -> np.ndarray:
    box_length = _positive_float(box_length_A, "box_length_A")
    return displacement_A - box_length * np.round(displacement_A / box_length)


def _wrap_positions_A(
    positions_A: np.ndarray,
    box_length_A: float,
) -> np.ndarray:
    box_length = _positive_float(box_length_A, "box_length_A")
    return positions_A - box_length * np.floor(positions_A / box_length)


def _unwrap_center_positions_A(
    current_folded_centers_A: np.ndarray,
    previous_unwrapped_centers_A: np.ndarray,
    box_length_A: float,
) -> np.ndarray:
    previous_folded_centers_A = _wrap_positions_A(
        previous_unwrapped_centers_A,
        box_length_A,
    )
    folded_displacement_A = current_folded_centers_A - previous_folded_centers_A
    continuous_displacement_A = _minimum_image_displacement_A(
        folded_displacement_A,
        box_length_A,
    )
    return previous_unwrapped_centers_A + continuous_displacement_A


def _append_center_steps(
    previous_state_indices: np.ndarray,
    current_state_indices: np.ndarray,
    previous_centers_unwrapped_A: np.ndarray,
    current_centers_unwrapped_A: np.ndarray,
    center_charge_number: int,
    from_state_indices: list[int],
    to_state_indices: list[int],
    charge_displacements_m: list[np.ndarray],
) -> None:
    if previous_state_indices.shape != current_state_indices.shape:
        raise ValueError("previous and current state arrays must have same shape")
    center_displacements_m = (
        current_centers_unwrapped_A - previous_centers_unwrapped_A
    ) * ANGSTROM_TO_M
    for center_index in range(previous_state_indices.shape[0]):
        from_state_indices.append(int(previous_state_indices[center_index]))
        to_state_indices.append(int(current_state_indices[center_index]))
        charge_displacements_m.append(
            center_charge_number * center_displacements_m[center_index],
        )


def _append_center_occupancies(
    li_state_indices: np.ndarray,
    pf6_state_indices: np.ndarray,
    occupancy_state_indices: list[int],
    li_state_counts: dict[str, int],
    pf6_state_counts: dict[str, int],
    state_labels: tuple[str, ...],
) -> None:
    for state_index in li_state_indices:
        state_label = state_labels[int(state_index)]
        li_state_counts[state_label] += 1
        occupancy_state_indices.append(int(state_index))
    for state_index in pf6_state_indices:
        state_label = state_labels[int(state_index)]
        pf6_state_counts[state_label] += 1
        occupancy_state_indices.append(int(state_index))


def _validated_frame_interval_ps(
    kept_times_ps: tuple[float, ...],
    expected_raw_interval_ps: float,
    frame_stride: int,
) -> float:
    if len(kept_times_ps) < 2:
        raise ValueError("At least two kept times are required")
    kept_intervals_ps = np.diff(np.asarray(kept_times_ps, dtype=float))
    raw_intervals_ps = kept_intervals_ps / float(frame_stride)
    if not np.all(
        np.isclose(
            raw_intervals_ps,
            expected_raw_interval_ps,
            rtol=FRAME_INTERVAL_RELATIVE_TOLERANCE,
            atol=0.0,
        ),
    ):
        raise ValueError(
            "PF6 frame interval mismatch: observed raw intervals "
            f"{raw_intervals_ps}, expected {expected_raw_interval_ps} ps",
        )
    return float(expected_raw_interval_ps)


def _primitive_target_artifact_from_sample_input(
    system_id: str,
    sample_input: TrajectoryMarkovAdditiveSampleInput,
    process_result: TrajectoryPrimitiveTargetProcessResult,
    frame_count: int,
    charged_center_count: int,
    block_count: int,
    frame_stride: int,
) -> TrajectoryPrimitiveTargetArtifact:
    block_targets = _block_targets_from_sample_input(
        sample_input,
        frame_count,
        charged_center_count,
        block_count,
    )
    state_concentrations = _state_concentration_targets(
        sample_input.state_labels,
        process_result.markov_input,
    )
    state_occupancy_fractions = _state_occupancy_fraction_targets(sample_input)
    transition_rates = _transition_rate_targets(process_result.markov_input)
    transition_fluxes = _transition_flux_targets(process_result.markov_input)
    residence_times = _residence_time_targets(
        sample_input.state_labels,
        transition_rates,
    )
    displacement_moments = _displacement_moment_targets(sample_input)
    block_transition_rate_standard_errors = _block_mapping_standard_errors(
        block_targets,
        "transition_rates_s_inv",
        tuple(transition_rates),
    )
    block_displacement_moment_standard_errors = _block_displacement_standard_errors(
        block_targets,
        displacement_moments,
    )
    block_sigma_standard_error_mS_cm = _standard_error(
        tuple(block.markov_additive_sigma_mS_cm for block in block_targets),
    )

    return TrajectoryPrimitiveTargetArtifact(
        system_id=system_id,
        frame_count=frame_count,
        dt_s=sample_input.dt_s,
        frame_stride=frame_stride,
        block_count=block_count,
        state_concentrations_mol_m3=state_concentrations,
        state_occupancy_fractions=state_occupancy_fractions,
        transition_rates_s_inv=transition_rates,
        transition_rate_targets_validated=_mapping_targets_are_block_stable(
            transition_rates,
            block_transition_rate_standard_errors,
        ),
        transition_fluxes_mol_m3_s=transition_fluxes,
        residence_times_s=residence_times,
        displacement_moments_by_family=displacement_moments,
        displacement_moment_targets_validated=(
            _displacement_targets_are_block_stable(
                displacement_moments,
                block_displacement_moment_standard_errors,
            )
        ),
        markov_additive_sigma_mS_cm=(process_result.conductivity_result.sigma_mS_cm),
        markov_direct_sigma_mS_cm=(
            process_result.conductivity_result.direct_sigma_mS_cm
        ),
        markov_corrector_sigma_mS_cm=(
            process_result.conductivity_result.corrector_sigma_mS_cm
        ),
        markov_additive_sigma_validated=_scalar_target_is_block_stable(
            process_result.conductivity_result.sigma_mS_cm,
            block_sigma_standard_error_mS_cm,
        ),
        block_targets=block_targets,
        block_state_concentration_standard_errors_mol_m3=(
            _block_mapping_standard_errors(
                block_targets,
                "state_concentrations_mol_m3",
                tuple(state_concentrations),
            )
        ),
        block_transition_rate_standard_errors_s_inv=(
            block_transition_rate_standard_errors
        ),
        block_displacement_moment_standard_errors_m2=(
            block_displacement_moment_standard_errors
        ),
        block_sigma_standard_error_mS_cm=block_sigma_standard_error_mS_cm,
    )


def _block_targets_from_sample_input(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
    frame_count: int,
    charged_center_count: int,
    block_count: int,
) -> tuple[TrajectoryBlockPrimitiveTarget, ...]:
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    if charged_center_count <= 0:
        raise ValueError("charged_center_count must be positive")
    if block_count <= 0:
        raise ValueError("block_count must be positive")
    trajectory_interval_indices = np.arange(frame_count - 1, dtype=int)
    if block_count > trajectory_interval_indices.shape[0]:
        raise ValueError("block_count cannot exceed trajectory interval count")
    block_interval_groups = np.array_split(trajectory_interval_indices, block_count)
    block_targets: list[TrajectoryBlockPrimitiveTarget] = []
    for block_index, block_interval_indices in enumerate(block_interval_groups):
        first_interval_index = int(block_interval_indices[0])
        final_interval_index = int(block_interval_indices[-1])
        occupancy_start_index = first_interval_index * charged_center_count
        occupancy_stop_index = (final_interval_index + 2) * charged_center_count
        step_start_index = first_interval_index * charged_center_count
        step_stop_index = (final_interval_index + 1) * charged_center_count
        block_sample_input = TrajectoryMarkovAdditiveSampleInput(
            state_labels=sample_input.state_labels,
            occupancy_state_index_by_observation=(
                sample_input.occupancy_state_index_by_observation[
                    occupancy_start_index:occupancy_stop_index
                ]
            ),
            from_state_index_by_step=sample_input.from_state_index_by_step[
                step_start_index:step_stop_index
            ],
            to_state_index_by_step=sample_input.to_state_index_by_step[
                step_start_index:step_stop_index
            ],
            charge_displacement_by_step_m=(
                sample_input.charge_displacement_by_step_m[
                    step_start_index:step_stop_index
                ]
            ),
            dt_s=sample_input.dt_s,
            total_transport_concentration_mol_m3=(
                sample_input.total_transport_concentration_mol_m3
            ),
            temperature_K=sample_input.temperature_K,
            displacement_zero_tolerance_m=(sample_input.displacement_zero_tolerance_m),
        )
        block_process_result = compute_sampled_trajectory_markov_additive_conductivity(
            block_sample_input,
        )
        transition_rates = _transition_rate_targets(
            block_process_result.markov_input,
        )
        block_targets.append(
            TrajectoryBlockPrimitiveTarget(
                block_index=block_index,
                frame_count=int(final_interval_index - first_interval_index + 2),
                step_count=int(
                    block_sample_input.from_state_index_by_step.shape[0],
                ),
                state_concentrations_mol_m3=_state_concentration_targets(
                    sample_input.state_labels,
                    block_process_result.markov_input,
                ),
                state_occupancy_fractions=_state_occupancy_fraction_targets(
                    block_sample_input,
                ),
                transition_rates_s_inv=transition_rates,
                transition_rate_targets_validated=False,
                transition_fluxes_mol_m3_s=_transition_flux_targets(
                    block_process_result.markov_input,
                ),
                residence_times_s=_residence_time_targets(
                    sample_input.state_labels,
                    transition_rates,
                ),
                displacement_moments_by_family=_displacement_moment_targets(
                    block_sample_input,
                ),
                markov_additive_sigma_mS_cm=(
                    block_process_result.conductivity_result.sigma_mS_cm
                ),
                markov_direct_sigma_mS_cm=(
                    block_process_result.conductivity_result.direct_sigma_mS_cm
                ),
                markov_corrector_sigma_mS_cm=(
                    block_process_result.conductivity_result.corrector_sigma_mS_cm
                ),
            ),
        )
    return tuple(block_targets)


def _state_concentration_targets(
    requested_state_labels: tuple[str, ...],
    markov_input: MarkovAdditiveConductivityInput,
) -> Mapping[str, float]:
    concentrations = {state_label: 0.0 for state_label in requested_state_labels}
    for state_index, state_label in enumerate(markov_input.state_labels):
        concentrations[state_label] = float(
            markov_input.state_concentrations_mol_m3[state_index],
        )
    return concentrations


def _state_occupancy_fraction_targets(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
) -> Mapping[str, float]:
    occupancy_counts = np.bincount(
        sample_input.occupancy_state_index_by_observation,
        minlength=len(sample_input.state_labels),
    )
    total_occupancy_count = float(np.sum(occupancy_counts))
    if total_occupancy_count <= 0.0:
        raise ValueError("sample input has no occupancy observations")
    return {
        state_label: float(occupancy_counts[state_index] / total_occupancy_count)
        for state_index, state_label in enumerate(sample_input.state_labels)
    }


def _transition_rate_targets(
    markov_input: MarkovAdditiveConductivityInput,
) -> Mapping[str, float]:
    transition_rates: dict[str, float] = {}
    for event in markov_input.events:
        if event.from_state_index == event.to_state_index:
            continue
        transition_label = _transition_label(
            markov_input.state_labels[event.from_state_index],
            markov_input.state_labels[event.to_state_index],
        )
        if transition_label not in transition_rates:
            transition_rates[transition_label] = 0.0
        transition_rates[transition_label] += event.rate_s_inv
    return transition_rates


def _transition_flux_targets(
    markov_input: MarkovAdditiveConductivityInput,
) -> Mapping[str, float]:
    transition_fluxes: dict[str, float] = {}
    for event in markov_input.events:
        if event.from_state_index == event.to_state_index:
            continue
        transition_label = _transition_label(
            markov_input.state_labels[event.from_state_index],
            markov_input.state_labels[event.to_state_index],
        )
        event_flux = (
            markov_input.state_concentrations_mol_m3[event.from_state_index]
            * event.rate_s_inv
        )
        if transition_label not in transition_fluxes:
            transition_fluxes[transition_label] = 0.0
        transition_fluxes[transition_label] += float(event_flux)
    return transition_fluxes


def _residence_time_targets(
    state_labels: tuple[str, ...],
    transition_rates_s_inv: Mapping[str, float],
) -> Mapping[str, float]:
    outgoing_rates = {state_label: 0.0 for state_label in state_labels}
    for transition_label, transition_rate_s_inv in transition_rates_s_inv.items():
        from_state_label, _to_state_label = transition_label.split("->", maxsplit=1)
        outgoing_rates[from_state_label] += transition_rate_s_inv
    residence_times: dict[str, float] = {}
    for state_label, outgoing_rate_s_inv in outgoing_rates.items():
        if outgoing_rate_s_inv > 0.0:
            residence_times[state_label] = 1.0 / outgoing_rate_s_inv
        else:
            residence_times[state_label] = math.inf
    return residence_times


def _displacement_moment_targets(
    sample_input: TrajectoryMarkovAdditiveSampleInput,
) -> Mapping[str, TrajectoryDisplacementMomentTarget]:
    family_displacements: dict[str, list[np.ndarray]] = defaultdict(list)
    for step_index, charge_displacement_m in enumerate(
        sample_input.charge_displacement_by_step_m,
    ):
        from_state_label = sample_input.state_labels[
            int(sample_input.from_state_index_by_step[step_index])
        ]
        to_state_label = sample_input.state_labels[
            int(sample_input.to_state_index_by_step[step_index])
        ]
        transition_label = _transition_label(from_state_label, to_state_label)
        family_displacements[transition_label].append(charge_displacement_m)

    moment_targets: dict[str, TrajectoryDisplacementMomentTarget] = {}
    for transition_label, displacement_vectors_m in family_displacements.items():
        displacement_array_m = np.asarray(displacement_vectors_m, dtype=float)
        mean_displacement_m = np.mean(displacement_array_m, axis=0)
        mean_squared_axis_displacement_m2 = np.mean(
            displacement_array_m * displacement_array_m,
            axis=0,
        )
        mean_squared_displacement_m2 = float(
            np.mean(np.sum(displacement_array_m * displacement_array_m, axis=1)),
        )
        moment_targets[transition_label] = TrajectoryDisplacementMomentTarget(
            sample_count=int(displacement_array_m.shape[0]),
            mean_displacement_m=_as_displacement_tuple(mean_displacement_m),
            mean_squared_axis_displacement_m2=_as_displacement_tuple(
                mean_squared_axis_displacement_m2,
            ),
            mean_squared_displacement_m2=mean_squared_displacement_m2,
        )
    return moment_targets


def _transition_label(
    from_state_label: str,
    to_state_label: str,
) -> str:
    return f"{from_state_label}->{to_state_label}"


def _block_mapping_standard_errors(
    block_targets: tuple[TrajectoryBlockPrimitiveTarget, ...],
    mapping_attribute_name: str,
    target_labels: tuple[str, ...],
) -> Mapping[str, float]:
    standard_errors: dict[str, float] = {}
    for target_label in target_labels:
        values: list[float] = []
        for block in block_targets:
            block_mapping = getattr(block, mapping_attribute_name)
            if target_label in block_mapping:
                values.append(float(block_mapping[target_label]))
            else:
                values.append(0.0)
        standard_errors[target_label] = _standard_error(tuple(values))
    return standard_errors


def _block_displacement_standard_errors(
    block_targets: tuple[TrajectoryBlockPrimitiveTarget, ...],
    displacement_moments: Mapping[str, TrajectoryDisplacementMomentTarget],
) -> Mapping[str, float]:
    standard_errors: dict[str, float] = {}
    for transition_label in displacement_moments:
        values = tuple(
            block.displacement_moments_by_family[
                transition_label
            ].mean_squared_displacement_m2
            for block in block_targets
            if transition_label in block.displacement_moments_by_family
        )
        standard_errors[transition_label] = _standard_error(values)
    return standard_errors


def _standard_error(values: tuple[float, ...]) -> float:
    if len(values) <= 1:
        return 0.0
    value_array = np.asarray(values, dtype=float)
    return float(np.std(value_array, ddof=1) / math.sqrt(len(values)))


def _mapping_targets_are_block_stable(
    target_values: Mapping[str, float],
    standard_errors: Mapping[str, float],
) -> bool:
    if not target_values:
        return False
    for target_label, target_value in target_values.items():
        target_magnitude = abs(_finite_float(target_value, target_label))
        standard_error = _finite_float(
            standard_errors[target_label],
            f"{target_label}.standard_error",
        )
        if target_magnitude == 0.0:
            if standard_error != 0.0:
                return False
            continue
        if not standard_error <= target_magnitude:
            return False
    return True


def _scalar_target_is_block_stable(
    target_value: float,
    standard_error: float,
) -> bool:
    target_magnitude = abs(_finite_float(target_value, "scalar_target"))
    validated_standard_error = _finite_float(standard_error, "scalar_standard_error")
    if target_magnitude == 0.0:
        return validated_standard_error == 0.0
    return validated_standard_error <= target_magnitude


def _displacement_targets_are_block_stable(
    displacement_moments: Mapping[str, TrajectoryDisplacementMomentTarget],
    standard_errors: Mapping[str, float],
) -> bool:
    if not displacement_moments:
        return False
    second_moment_by_label = {
        label: moment.mean_squared_displacement_m2
        for label, moment in displacement_moments.items()
    }
    if not _mapping_targets_are_block_stable(
        second_moment_by_label,
        standard_errors,
    ):
        return False
    for transition_label, moment in displacement_moments.items():
        second_moment = _finite_float(
            moment.mean_squared_displacement_m2,
            f"{transition_label}.mean_squared_displacement_m2",
        )
        mean_norm_squared = math.fsum(
            component * component for component in moment.mean_displacement_m
        )
        if second_moment + np.finfo(float).eps < mean_norm_squared:
            return False
    return True


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


def _validated_sample_state_indices(
    state_indices_by_step: np.ndarray,
    state_count: int,
    label: str,
) -> np.ndarray:
    state_indices = np.asarray(state_indices_by_step, dtype=int)
    if state_indices.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if state_indices.shape[0] < 1:
        raise ValueError(f"{label} must contain at least one step")
    if int(np.min(state_indices)) < 0:
        raise ValueError(f"{label} contains a negative state index")
    if int(np.max(state_indices)) >= state_count:
        raise ValueError(f"{label} contains an invalid state index")
    return state_indices


def _validated_charge_displacements(
    charge_displacement_by_step_m: np.ndarray,
    expected_step_count: int,
) -> np.ndarray:
    charge_displacements = np.asarray(charge_displacement_by_step_m, dtype=float)
    if charge_displacements.shape != (expected_step_count, 3):
        raise ValueError(
            f"charge_displacement_by_step_m must have shape ({expected_step_count}, 3)",
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
        tuple(state_index_remap[int(state_index)] for state_index in state_indices),
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
    parsed_value = _finite_float(value, label)
    if parsed_value <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return parsed_value


def _nonnegative_float(value: float, label: str) -> float:
    parsed_value = _finite_float(value, label)
    if parsed_value < 0.0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return parsed_value


def _finite_float(value: float, label: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{label} must be finite")
    return parsed_value


def _validate_positive_int(value: int, label: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise ValueError(f"{label} must be positive")
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
