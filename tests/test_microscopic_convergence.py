from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from constants import K_B
from conductivity.physical_library.generator_construction import (
    compute_conductivity_from_recipe,
)
from conductivity.physical_library.microscopic_convergence import (
    FixedMicroscopicGenerator,
    MicroscopicConvergenceThresholds,
    MicroscopicNumericalRealization,
    MicroscopicRecipeComposition,
    MicroscopicTrajectoryEvidence,
    build_microscopic_convergence_certificate,
    estimate_einstein_helfand_conductivity,
    read_microscopic_production_artifact,
    write_microscopic_production_artifact,
)


PHYSICAL_LIBRARY_ROOT = Path("conductivity/physical_library")


def test_einstein_helfand_estimator_recovers_brownian_charge_transport() -> None:
    random_generator = np.random.default_rng(17)
    frame_interval_s = 1.0e-12
    volume_m3 = 1.0e-25
    temperature_K = 300.0
    expected_conductivity_S_m = 1.2
    charge_moment_diffusivity = (
        volume_m3 * K_B * temperature_K * expected_conductivity_S_m
    )
    increment_standard_deviation = np.sqrt(
        2.0 * charge_moment_diffusivity * frame_interval_s
    )
    increments = random_generator.normal(
        scale=increment_standard_deviation,
        size=(40_000, 3),
    )
    helfand_moments = np.vstack(
        (np.zeros((1, 3), dtype=float), np.cumsum(increments, axis=0))
    )

    estimate = estimate_einstein_helfand_conductivity(
        helfand_moment_C_m=helfand_moments,
        frame_interval_s=frame_interval_s,
        volume_m3=volume_m3,
        temperature_K=temperature_K,
        fit_lag_start_frames=20,
        fit_lag_stop_frames=200,
        block_count=8,
    )

    assert estimate.conductivity_S_m == pytest.approx(
        expected_conductivity_S_m,
        rel=0.2,
    )
    assert estimate.standard_error_S_m > 0.0


def test_nested_galerkin_spaces_converge_monotonically_to_finite_gk_value() -> None:
    symmetric_rates = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    generator = symmetric_rates - np.diag(np.sum(symmetric_rates, axis=1))
    negative_generator = -generator
    microscopic_current = np.asarray((-1.5, -0.5, 0.5, 1.5), dtype=float)
    exact_corrector = np.linalg.pinv(negative_generator) @ microscopic_current
    exact_gk_value = float(microscopic_current @ exact_corrector)
    nested_bases = (
        np.asarray(((-1.0,), (-1.0,), (1.0,), (1.0,)), dtype=float),
        np.asarray(
            (
                (-1.0, 1.0),
                (-1.0, -1.0),
                (1.0, -1.0),
                (1.0, 1.0),
            ),
            dtype=float,
        ),
        np.column_stack(
            (
                np.eye(generator.shape[0], dtype=float)[:, :-1],
            )
        ),
    )
    projected_values = tuple(
        _finite_galerkin_gk_value(
            negative_generator=negative_generator,
            microscopic_current=microscopic_current,
            basis=basis,
        )
        for basis in nested_bases
    )

    assert projected_values[0] <= projected_values[1] <= projected_values[2]
    assert projected_values[-1] == pytest.approx(exact_gk_value)


def test_certificate_rejects_absent_numerical_refinement_axes() -> None:
    generator = _generator()
    realization = _realization()
    evidence = tuple(
        _evidence(
            generator=generator,
            realization=realization,
            replica_id=replica_id,
            basis_coordinates=basis_coordinates,
        )
        for replica_id in ("replica-a", "replica-b")
        for basis_coordinates in (("state",), ("state", "memory"))
    )

    certificate = build_microscopic_convergence_certificate(
        generator=generator,
        evidence=evidence,
        thresholds=_thresholds(),
    )

    assert "timestep_refinement" in certificate.failed_audits
    assert "box_size_refinement" in certificate.failed_audits
    assert "trajectory_duration_refinement" in certificate.failed_audits
    assert "current_stride_refinement" in certificate.failed_audits
    assert "trajectory_stride_refinement" in certificate.failed_audits
    with pytest.raises(ValueError, match="not converged"):
        certificate.require_converged()


def test_recipe_production_entrypoint_requires_microscopic_certificate(
    tmp_path: Path,
) -> None:
    recipe_path = PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_1m.yaml"
    with pytest.raises(ValueError, match="exactly one converged microscopic"):
        compute_conductivity_from_recipe(
            recipe=recipe_path,
            library_root=tmp_path,
        )


def test_complete_refinement_family_persists_only_after_recomputed_audits(
    tmp_path: Path,
) -> None:
    generator = _generator()
    finest_basis = ("state", "memory")
    production_realization = _realization_for_axes(
        timestep_fs=0.5,
        box_length_A=50.0,
        duration_ps=200.0,
        current_stride_steps=1,
        trajectory_stride_steps=5,
    )
    refinement_realizations = (
        production_realization,
        _realization_for_axes(1.0, 50.0, 200.0, 1, 5),
        _realization_for_axes(0.5, 40.0, 200.0, 1, 5),
        _realization_for_axes(0.5, 50.0, 100.0, 1, 5),
        _realization_for_axes(0.5, 50.0, 200.0, 2, 5),
        _realization_for_axes(0.5, 50.0, 200.0, 1, 10),
    )
    evidence = tuple(
        _evidence(generator, realization, replica_id, finest_basis)
        for realization in refinement_realizations
        for replica_id in ("replica-a", "replica-b")
    ) + tuple(
        _evidence(
            generator,
            production_realization,
            replica_id,
            ("state",),
        )
        for replica_id in ("replica-a", "replica-b")
    )
    certificate = build_microscopic_convergence_certificate(
        generator=generator,
        evidence=evidence,
        thresholds=_thresholds(),
    )
    assert certificate.converged

    artifact_path = tmp_path / "certificate.json"
    write_microscopic_production_artifact(
        path=artifact_path,
        generator=generator,
        evidence=evidence,
        thresholds=_thresholds(),
        projected_primitive_yaml_path=Path("projected.yaml"),
    )
    loaded = read_microscopic_production_artifact(artifact_path)
    assert loaded.certificate.converged
    assert loaded.evidence == evidence


def _generator() -> FixedMicroscopicGenerator:
    return FixedMicroscopicGenerator(
        recipe=MicroscopicRecipeComposition(
            solvents_vv=(("DMC", 0.7), ("EC", 0.3)),
            salts_mol_l=(("Li+", 1.0), ("PF6-", 1.0)),
            additives_weight_fraction=(),
        ),
        temperature_K=298.15,
        pressure_Pa=101_325.0,
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


def _realization() -> MicroscopicNumericalRealization:
    return MicroscopicNumericalRealization(
        molecule_counts=(("DMC", 70), ("EC", 30), ("Li+", 10), ("PF6-", 10)),
        box_vectors_A=(
            (40.0, 0.0, 0.0),
            (0.0, 40.0, 0.0),
            (0.0, 0.0, 40.0),
        ),
        timestep_fs=1.0,
        current_stride_steps=1,
        trajectory_stride_steps=10,
        duration_ps=100.0,
        discrete_integrator="velocity-Verlet",
        lammps_version="test",
        lammps_data_text="data",
        lammps_input_text="input",
    )


def _realization_for_axes(
    timestep_fs: float,
    box_length_A: float,
    duration_ps: float,
    current_stride_steps: int,
    trajectory_stride_steps: int,
) -> MicroscopicNumericalRealization:
    return MicroscopicNumericalRealization(
        molecule_counts=(("DMC", 70), ("EC", 30), ("Li+", 10), ("PF6-", 10)),
        box_vectors_A=(
            (box_length_A, 0.0, 0.0),
            (0.0, box_length_A, 0.0),
            (0.0, 0.0, box_length_A),
        ),
        timestep_fs=timestep_fs,
        current_stride_steps=current_stride_steps,
        trajectory_stride_steps=trajectory_stride_steps,
        duration_ps=duration_ps,
        discrete_integrator="velocity-Verlet",
        lammps_version="test",
        lammps_data_text="data",
        lammps_input_text="input",
    )


def _evidence(
    generator: FixedMicroscopicGenerator,
    realization: MicroscopicNumericalRealization,
    replica_id: str,
    basis_coordinates: tuple[str, ...],
) -> MicroscopicTrajectoryEvidence:
    return MicroscopicTrajectoryEvidence(
        generator=generator,
        numerical_realization=realization,
        replica_id=replica_id,
        conductivity_current_S_m=1.0,
        conductivity_helfand_S_m=1.0,
        projected_conductivity_S_m=1.0,
        basis_coordinates=basis_coordinates,
        maximum_dirichlet_residual_score_m2_s=1.0e-13,
        transition_capacity_relative_change=0.01,
        transition_first_moment_relative_change=0.01,
        transition_second_moment_relative_change=0.01,
        effective_sample_size=500.0,
        stationary=True,
        committor_residual=0.01,
        detailed_balance_residual=0.01,
    )


def _thresholds() -> MicroscopicConvergenceThresholds:
    return MicroscopicConvergenceThresholds(
        confidence_level=0.95,
        timestep_bias_tolerance_S_m=0.125,
        finite_size_tolerance_S_m=0.125,
        duration_tolerance_S_m=0.125,
        current_stride_tolerance_S_m=0.125,
        trajectory_stride_tolerance_S_m=0.125,
        estimator_agreement_tolerance_S_m=0.125,
        replica_tolerance_S_m=0.125,
        basis_tolerance_S_m=0.125,
        dirichlet_residual_tolerance_m2_s=1.0e-12,
        transition_relative_tolerance=0.125,
        committor_residual_tolerance=0.125,
        detailed_balance_residual_tolerance=0.125,
        minimum_effective_sample_size=100.0,
    )


def _finite_galerkin_gk_value(
    negative_generator: np.ndarray,
    microscopic_current: np.ndarray,
    basis: np.ndarray,
) -> float:
    dirichlet_matrix = basis.T @ negative_generator @ basis
    current_coupling = basis.T @ microscopic_current
    coefficients = np.linalg.pinv(dirichlet_matrix) @ current_coupling
    projected_corrector = basis @ coefficients
    return float(microscopic_current @ projected_corrector)
