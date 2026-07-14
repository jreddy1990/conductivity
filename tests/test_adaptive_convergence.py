from __future__ import annotations

import pytest

from constants import MS_CM_TO_S_M, PA_PER_ATM
from conductivity.physical_library.adaptive_convergence import (
    AdaptiveConvergenceAccuracy,
    AdaptiveConvergenceBudget,
    AdaptiveExperimentState,
    AdaptiveRefinementComplete,
    AdaptiveRefinementDecision,
    CompletedAdaptiveRun,
    MicroscopicPilotPolicy,
    RefinementAxis,
    initialize_pilot_designs,
    refine_box_size,
    refine_duration,
    refine_timestep,
    select_next_refinement,
)
from conductivity.physical_library.microscopic_convergence import (
    FixedMicroscopicGenerator,
    MicroscopicConvergenceAudit,
    MicroscopicRecipeComposition,
)


def test_pilot_initialization_and_numerical_refinements_are_dyadic() -> None:
    pilot_policy = _pilot_policy()
    pilot_runs = initialize_pilot_designs(
        pilot_policy=pilot_policy,
        target_atom_count=2_000,
        velocity_seeds=(101, 103),
        thermostat_seeds=(107, 109),
    )

    assert len(pilot_runs) == pilot_policy.initial_replica_count
    assert pilot_runs[0].replica_id != pilot_runs[1].replica_id
    timestep_run = refine_timestep(pilot_runs[0], "replica-1", 113, 127)
    duration_run = refine_duration(pilot_runs[0], "replica-1", 131, 137)
    box_run = refine_box_size(pilot_runs[0], "replica-1", 139, 149)
    assert timestep_run.timestep_fs * 2 == pilot_runs[0].timestep_fs
    assert (
        duration_run.production_duration_ps == 2 * pilot_runs[0].production_duration_ps
    )
    assert box_run.target_atom_count == 8 * pilot_runs[0].target_atom_count


def test_scheduler_selects_largest_expected_error_reduction_per_cost() -> None:
    state = _state(
        audits=(
            MicroscopicConvergenceAudit(
                name="timestep_refinement",
                passed=False,
                measured_value=0.4,
                uncertainty=0.1,
                tolerance=0.1,
                compared_records=("dt",),
            ),
            MicroscopicConvergenceAudit(
                name="effective_sample_size",
                passed=False,
                measured_value=20.0,
                uncertainty=0.0,
                tolerance=100.0,
                compared_records=("duration",),
            ),
        )
    )

    decision = select_next_refinement(state)

    assert isinstance(decision, AdaptiveRefinementDecision)
    assert decision.axis is RefinementAxis.TIMESTEP


def test_scheduler_reports_completion_only_when_every_audit_passes() -> None:
    state = _state(
        audits=(
            MicroscopicConvergenceAudit(
                name="dirichlet_residual",
                passed=True,
                measured_value=0.01,
                uncertainty=0.0,
                tolerance=0.1,
                compared_records=("basis",),
            ),
        )
    )

    decision = select_next_refinement(state)

    assert isinstance(decision, AdaptiveRefinementComplete)


def test_scheduler_rejects_completed_runs_without_measured_audits() -> None:
    state = _state(audits=())

    with pytest.raises(
        ValueError,
        match="completed microscopic runs require measured convergence audits",
    ):
        select_next_refinement(state)


def _state(
    audits: tuple[MicroscopicConvergenceAudit, ...],
) -> AdaptiveExperimentState:
    pilot = initialize_pilot_designs(
        pilot_policy=_pilot_policy(),
        target_atom_count=2_000,
        velocity_seeds=(101, 103),
        thermostat_seeds=(107, 109),
    )[0]
    return AdaptiveExperimentState(
        generator=_generator(),
        budget=AdaptiveConvergenceBudget(
            maximum_core_hours=100.0,
            maximum_storage_bytes=1_000_000,
            maximum_wall_time_s=100_000.0,
        ),
        accuracy=AdaptiveConvergenceAccuracy(
            conductivity_tolerance_S_m=MS_CM_TO_S_M,
            confidence_level=0.95,
            minimum_effective_sample_size=100.0,
            dirichlet_residual_tolerance_m2_s=1.0e-10,
            transition_relative_tolerance=0.1,
            committor_residual_tolerance=0.1,
            detailed_balance_residual_tolerance=0.1,
        ),
        pilot_policy=_pilot_policy(),
        completed_runs=(
            CompletedAdaptiveRun(
                design=pilot,
                core_hours=1.0,
                storage_bytes=1_000,
                wall_time_s=1_000.0,
            ),
        ),
        pending_runs=(),
        audits=audits,
    )


def _pilot_policy() -> MicroscopicPilotPolicy:
    return MicroscopicPilotPolicy(
        maximum_initial_timestep_fs=1.0,
        minimum_periodic_edge_A=30.0,
        minimum_production_duration_ps=50.0,
        minimum_equilibration_duration_ps=20.0,
        initial_trajectory_stride_steps=10,
        initial_replica_count=2,
    )


def _generator() -> FixedMicroscopicGenerator:
    return FixedMicroscopicGenerator(
        recipe=MicroscopicRecipeComposition(
            solvents_vv=(("DMC", 0.7), ("EC", 0.3)),
            salts_mol_l=(("Li+", 1.0), ("PF6-", 1.0)),
            additives_weight_fraction=(),
        ),
        temperature_K=298.15,
        pressure_Pa=PA_PER_ATM,
        component_number_density_ratios=(
            ("DMC", 0.7),
            ("EC", 0.3),
            ("Li+", 0.01),
            ("PF6-", 0.01),
        ),
        force_field_name="OpenFF Sage 2.2.0",
        charge_model="AM1-BCC",
        charge_scale=1.0,
        topology_and_force_field_record=(("model", "sage"),),
        pair_style="lj/cut/coul/long",
        bond_style="harmonic",
        angle_style="harmonic",
        dihedral_style="fourier",
        improper_style="cvff",
        production_ensemble="NVE",
        equations_of_motion="Hamiltonian",
        thermostat_policy="none during production",
        long_range_interaction_model="PPPM tin-foil",
        periodic_boundary_convention="three-dimensional periodic",
    )
