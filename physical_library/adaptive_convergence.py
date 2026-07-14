"""Adaptive experiment scheduling for microscopic conductivity convergence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import TypeAdapter

from conductivity.physical_library.microscopic_convergence import (
    FixedMicroscopicGenerator,
    MicroscopicConvergenceAudit,
)
from utils.strict_validation import (
    positive_finite_float,
    read_json_object,
    strict_positive_int,
    write_json_object,
)

LINEAR_REFINEMENT_FACTOR = 2
VOLUME_REFINEMENT_FACTOR = LINEAR_REFINEMENT_FACTOR**3  # Double all three box edges.


class RefinementAxis(str, Enum):
    PILOT = "pilot"
    DURATION = "duration"
    REPLICA = "replica"
    TIMESTEP = "timestep"
    BOX = "box"
    CURRENT_STRIDE = "current_stride"
    TRAJECTORY_STRIDE = "trajectory_stride"
    BASIS = "basis"
    TRANSITION = "transition"


@dataclass(frozen=True)
class AdaptiveConvergenceBudget:
    maximum_core_hours: float
    maximum_storage_bytes: int
    maximum_wall_time_s: float

    def validate(self) -> None:
        positive_finite_float(self.maximum_core_hours, "maximum_core_hours")
        strict_positive_int(self.maximum_storage_bytes, "maximum_storage_bytes")
        positive_finite_float(self.maximum_wall_time_s, "maximum_wall_time_s")


@dataclass(frozen=True)
class AdaptiveConvergenceAccuracy:
    conductivity_tolerance_S_m: float
    confidence_level: float
    minimum_effective_sample_size: float
    dirichlet_residual_tolerance_m2_s: float
    transition_relative_tolerance: float
    committor_residual_tolerance: float
    detailed_balance_residual_tolerance: float

    def validate(self) -> None:
        positive_finite_float(
            self.conductivity_tolerance_S_m, "conductivity_tolerance_S_m"
        )
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must be strictly between zero and one")
        positive_finite_float(
            self.minimum_effective_sample_size, "minimum_effective_sample_size"
        )
        positive_finite_float(
            self.dirichlet_residual_tolerance_m2_s, "dirichlet_residual_tolerance_m2_s"
        )
        positive_finite_float(
            self.transition_relative_tolerance, "transition_relative_tolerance"
        )
        positive_finite_float(
            self.committor_residual_tolerance, "committor_residual_tolerance"
        )
        positive_finite_float(
            self.detailed_balance_residual_tolerance,
            "detailed_balance_residual_tolerance",
        )


@dataclass(frozen=True)
class MicroscopicPilotPolicy:
    maximum_initial_timestep_fs: float
    minimum_periodic_edge_A: float
    minimum_production_duration_ps: float
    minimum_equilibration_duration_ps: float
    initial_trajectory_stride_steps: int
    initial_replica_count: int

    def validate(self) -> None:
        positive_finite_float(
            self.maximum_initial_timestep_fs, "maximum_initial_timestep_fs"
        )
        positive_finite_float(self.minimum_periodic_edge_A, "minimum_periodic_edge_A")
        positive_finite_float(
            self.minimum_production_duration_ps, "minimum_production_duration_ps"
        )
        positive_finite_float(
            self.minimum_equilibration_duration_ps, "minimum_equilibration_duration_ps"
        )
        strict_positive_int(
            self.initial_trajectory_stride_steps, "initial_trajectory_stride_steps"
        )
        if (
            strict_positive_int(self.initial_replica_count, "initial_replica_count")
            < LINEAR_REFINEMENT_FACTOR
        ):
            raise ValueError("initial_replica_count must be at least two")


@dataclass(frozen=True)
class AdaptiveConvergenceRequest:
    generator: FixedMicroscopicGenerator
    budget: AdaptiveConvergenceBudget
    accuracy: AdaptiveConvergenceAccuracy
    pilot_policy: MicroscopicPilotPolicy

    def validate(self) -> None:
        self.budget.validate()
        self.accuracy.validate()
        self.pilot_policy.validate()


@dataclass(frozen=True)
class AdaptiveRunDesign:
    refinement_axis: RefinementAxis
    refinement_level: int
    replica_id: str
    timestep_fs: float
    target_atom_count: int
    production_duration_ps: float
    equilibration_duration_ps: float
    trajectory_stride_steps: int
    velocity_seed: int
    thermostat_seed: int

    def validate(self) -> None:
        if self.refinement_level < 0:
            raise ValueError("refinement_level must be nonnegative")
        if not self.replica_id:
            raise ValueError("replica_id must be nonempty")
        positive_finite_float(self.timestep_fs, "timestep_fs")
        strict_positive_int(self.target_atom_count, "target_atom_count")
        positive_finite_float(self.production_duration_ps, "production_duration_ps")
        positive_finite_float(
            self.equilibration_duration_ps, "equilibration_duration_ps"
        )
        strict_positive_int(self.trajectory_stride_steps, "trajectory_stride_steps")
        strict_positive_int(self.velocity_seed, "velocity_seed")
        strict_positive_int(self.thermostat_seed, "thermostat_seed")


@dataclass(frozen=True)
class CompletedAdaptiveRun:
    design: AdaptiveRunDesign
    core_hours: float
    storage_bytes: int
    wall_time_s: float

    def validate(self) -> None:
        self.design.validate()
        positive_finite_float(self.core_hours, "core_hours")
        strict_positive_int(self.storage_bytes, "storage_bytes")
        positive_finite_float(self.wall_time_s, "wall_time_s")


@dataclass(frozen=True)
class AdaptiveExperimentState:
    generator: FixedMicroscopicGenerator
    budget: AdaptiveConvergenceBudget
    accuracy: AdaptiveConvergenceAccuracy
    pilot_policy: MicroscopicPilotPolicy
    completed_runs: tuple[CompletedAdaptiveRun, ...]
    pending_runs: tuple[AdaptiveRunDesign, ...]
    audits: tuple[MicroscopicConvergenceAudit, ...]

    def validate(self) -> None:
        self.budget.validate()
        self.accuracy.validate()
        self.pilot_policy.validate()
        for completed_run in self.completed_runs:
            completed_run.validate()
        for pending_run in self.pending_runs:
            pending_run.validate()


@dataclass(frozen=True)
class AdaptiveRefinementDecision:
    axis: RefinementAxis
    reason: str
    expected_error_reduction_per_core_hour: float


@dataclass(frozen=True)
class AdaptiveRefinementComplete:
    reason: str


@dataclass(frozen=True)
class AdaptiveRefinementPending:
    pending_run_count: int


AdaptiveRefinementResult = (
    AdaptiveRefinementDecision | AdaptiveRefinementComplete | AdaptiveRefinementPending
)

_AUDIT_TO_AXIS = {
    "equilibrium_stationarity": RefinementAxis.DURATION,
    "effective_sample_size": RefinementAxis.DURATION,
    "green_kubo_einstein_helfand_agreement": RefinementAxis.DURATION,
    "projected_zero_frequency_agreement": RefinementAxis.BASIS,
    "independent_replica_agreement": RefinementAxis.REPLICA,
    "nested_basis_conductivity": RefinementAxis.BASIS,
    "timestep_refinement": RefinementAxis.TIMESTEP,
    "box_size_refinement": RefinementAxis.BOX,
    "trajectory_duration_refinement": RefinementAxis.DURATION,
    "current_stride_refinement": RefinementAxis.CURRENT_STRIDE,
    "trajectory_stride_refinement": RefinementAxis.TRAJECTORY_STRIDE,
    "dirichlet_residual": RefinementAxis.BASIS,
    "transition_capacity_and_moments": RefinementAxis.TRANSITION,
    "committor_residual": RefinementAxis.TRANSITION,
    "detailed_balance_residual": RefinementAxis.TRANSITION,
}


def select_next_refinement(state: AdaptiveExperimentState) -> AdaptiveRefinementResult:
    """Choose the failed axis with the largest measured error reduction per cost."""
    state.validate()
    if state.pending_runs:
        return AdaptiveRefinementPending(len(state.pending_runs))
    if not state.completed_runs:
        return AdaptiveRefinementDecision(
            RefinementAxis.PILOT, "no trajectory evidence exists", float("inf")
        )
    if not state.audits:
        raise ValueError(
            "completed microscopic runs require measured convergence audits"
        )
    failed_audits = tuple(audit for audit in state.audits if not audit.passed)
    if not failed_audits:
        return AdaptiveRefinementComplete("all measured convergence audits pass")
    mean_core_hours = sum(run.core_hours for run in state.completed_runs) / len(
        state.completed_runs
    )
    decisions: list[AdaptiveRefinementDecision] = []
    for audit in failed_audits:
        if audit.name not in _AUDIT_TO_AXIS:
            raise ValueError(f"no adaptive refinement owner for audit {audit.name}")
        axis = _AUDIT_TO_AXIS[audit.name]
        decisions.append(
            AdaptiveRefinementDecision(
                axis,
                f"{audit.name}: {audit.measured_value} + {audit.uncertainty} versus {audit.tolerance}",
                _audit_severity(audit)
                / (mean_core_hours * _axis_cost_multiplier(axis)),
            )
        )
    return max(decisions, key=_decision_score)


def refine_timestep(
    parent: AdaptiveRunDesign, replica_id: str, velocity_seed: int, thermostat_seed: int
) -> AdaptiveRunDesign:
    return _validated_replacement(
        parent,
        RefinementAxis.TIMESTEP,
        replica_id,
        velocity_seed,
        thermostat_seed,
        parent.timestep_fs / LINEAR_REFINEMENT_FACTOR,
        parent.target_atom_count,
        parent.production_duration_ps,
        parent.trajectory_stride_steps,
    )


def refine_box_size(
    parent: AdaptiveRunDesign, replica_id: str, velocity_seed: int, thermostat_seed: int
) -> AdaptiveRunDesign:
    return _validated_replacement(
        parent,
        RefinementAxis.BOX,
        replica_id,
        velocity_seed,
        thermostat_seed,
        parent.timestep_fs,
        parent.target_atom_count * VOLUME_REFINEMENT_FACTOR,
        parent.production_duration_ps,
        parent.trajectory_stride_steps,
    )


def refine_duration(
    parent: AdaptiveRunDesign, replica_id: str, velocity_seed: int, thermostat_seed: int
) -> AdaptiveRunDesign:
    return _validated_replacement(
        parent,
        RefinementAxis.DURATION,
        replica_id,
        velocity_seed,
        thermostat_seed,
        parent.timestep_fs,
        parent.target_atom_count,
        parent.production_duration_ps * LINEAR_REFINEMENT_FACTOR,
        parent.trajectory_stride_steps,
    )


def refine_trajectory_stride(
    parent: AdaptiveRunDesign, replica_id: str, velocity_seed: int, thermostat_seed: int
) -> AdaptiveRunDesign:
    if parent.trajectory_stride_steps == 1:
        raise ValueError("trajectory stride is already one integration step")
    return _validated_replacement(
        parent,
        RefinementAxis.TRAJECTORY_STRIDE,
        replica_id,
        velocity_seed,
        thermostat_seed,
        parent.timestep_fs,
        parent.target_atom_count,
        parent.production_duration_ps,
        max(1, parent.trajectory_stride_steps // LINEAR_REFINEMENT_FACTOR),
    )


def budget_allows_run(
    state: AdaptiveExperimentState,
    proposed_core_hours: float,
    proposed_storage_bytes: int,
    proposed_wall_time_s: float,
) -> bool:
    state.validate()
    return (
        sum(run.core_hours for run in state.completed_runs) + proposed_core_hours
        <= state.budget.maximum_core_hours
        and sum(run.storage_bytes for run in state.completed_runs)
        + proposed_storage_bytes
        <= state.budget.maximum_storage_bytes
        and sum(run.wall_time_s for run in state.completed_runs) + proposed_wall_time_s
        <= state.budget.maximum_wall_time_s
    )


def initialize_pilot_designs(
    pilot_policy: MicroscopicPilotPolicy,
    target_atom_count: int,
    velocity_seeds: Sequence[int],
    thermostat_seeds: Sequence[int],
) -> tuple[AdaptiveRunDesign, ...]:
    pilot_policy.validate()
    strict_positive_int(target_atom_count, "target_atom_count")
    if (
        len(velocity_seeds) != pilot_policy.initial_replica_count
        or len(thermostat_seeds) != pilot_policy.initial_replica_count
    ):
        raise ValueError("seed counts must equal initial_replica_count")
    designs = tuple(
        AdaptiveRunDesign(
            RefinementAxis.PILOT,
            0,
            f"replica-{index + 1}",
            pilot_policy.maximum_initial_timestep_fs,
            target_atom_count,
            pilot_policy.minimum_production_duration_ps,
            pilot_policy.minimum_equilibration_duration_ps,
            pilot_policy.initial_trajectory_stride_steps,
            int(velocity_seeds[index]),
            int(thermostat_seeds[index]),
        )
        for index in range(pilot_policy.initial_replica_count)
    )
    for design in designs:
        design.validate()
    return designs


def read_adaptive_experiment_state(path: Path) -> AdaptiveExperimentState:
    state = TypeAdapter(AdaptiveExperimentState).validate_python(
        read_json_object(path, "adaptive convergence experiment")
    )
    state.validate()
    return state


def read_adaptive_convergence_request(path: Path) -> AdaptiveConvergenceRequest:
    request = TypeAdapter(AdaptiveConvergenceRequest).validate_python(
        read_json_object(path, "adaptive convergence request")
    )
    request.validate()
    return request


def write_adaptive_experiment_state(path: Path, state: AdaptiveExperimentState) -> None:
    state.validate()
    write_json_object(path, asdict(state), "adaptive convergence experiment")


def _validated_replacement(
    parent: AdaptiveRunDesign,
    refinement_axis: RefinementAxis,
    replica_id: str,
    velocity_seed: int,
    thermostat_seed: int,
    timestep_fs: float,
    target_atom_count: int,
    production_duration_ps: float,
    trajectory_stride_steps: int,
) -> AdaptiveRunDesign:
    parent.validate()
    refined = replace(
        parent,
        refinement_axis=refinement_axis,
        refinement_level=parent.refinement_level + 1,
        replica_id=replica_id,
        velocity_seed=velocity_seed,
        thermostat_seed=thermostat_seed,
        timestep_fs=timestep_fs,
        target_atom_count=target_atom_count,
        production_duration_ps=production_duration_ps,
        trajectory_stride_steps=trajectory_stride_steps,
    )
    refined.validate()
    return refined


def _decision_score(decision: AdaptiveRefinementDecision) -> float:
    return decision.expected_error_reduction_per_core_hour


def _audit_severity(audit: MicroscopicConvergenceAudit) -> float:
    if audit.tolerance == 0.0:
        return 1.0 + abs(audit.measured_value) + audit.uncertainty
    return (abs(audit.measured_value) + audit.uncertainty) / audit.tolerance


def _axis_cost_multiplier(axis: RefinementAxis) -> float:
    return {
        RefinementAxis.PILOT: 1.0,
        RefinementAxis.DURATION: float(LINEAR_REFINEMENT_FACTOR),
        RefinementAxis.REPLICA: 1.0,
        RefinementAxis.TIMESTEP: float(LINEAR_REFINEMENT_FACTOR),
        RefinementAxis.BOX: float(VOLUME_REFINEMENT_FACTOR),
        RefinementAxis.CURRENT_STRIDE: 1.0,
        RefinementAxis.TRAJECTORY_STRIDE: 1.0,
        RefinementAxis.BASIS: 1.0,
        RefinementAxis.TRANSITION: 1.0,
    }[axis]
