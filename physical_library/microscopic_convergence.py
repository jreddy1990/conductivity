"""Admission gates for conductivity from one fixed microscopic model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np
from pydantic import TypeAdapter

from constants import (
    ANGSTROM_TO_M,
    CARTESIAN_COMPONENT_COUNT,
    E_CHARGE,
    EINSTEIN_HELFAND_DENOMINATOR_FACTOR,
    FEMTOSECOND_TO_S,
    K_B,
    S_M_TO_MS_CM,
)
from conductivity.physical_library.projected_primitives_io import (
    compute_conductivity_from_primitive_yaml,
)
from utils.strict_validation import (
    positive_finite_float,
    read_json_object,
    strict_nonnegative_float,
    write_json_object,
)
from utils.time_series_statistics import (
    MINIMUM_LINEAR_FIT_SAMPLE_COUNT,
    contiguous_blocks,
    dyadic_values,
    linear_fit,
    mean_and_standard_error,
)


Array = np.ndarray
MICROSCOPIC_PRODUCTION_SCHEMA = "microscopic_conductivity_production_v1"


@dataclass(frozen=True)
class MicroscopicRecipeComposition:
    solvents_vv: tuple[tuple[str, float], ...]
    salts_mol_l: tuple[tuple[str, float], ...]
    additives_weight_fraction: tuple[tuple[str, float], ...]

    @classmethod
    def from_record(cls, recipe_record: dict) -> MicroscopicRecipeComposition:
        return cls(
            solvents_vv=_positive_composition_items(
                recipe_record["solvents_vv"], "solvents_vv"
            ),
            salts_mol_l=_positive_composition_items(
                recipe_record["salts_mol_l"], "salts_mol_l"
            ),
            additives_weight_fraction=_nonnegative_composition_items(
                recipe_record["additives_weight_fraction"],
                "additives_weight_fraction",
            ),
        )


@dataclass(frozen=True)
class FixedMicroscopicGenerator:
    recipe: MicroscopicRecipeComposition
    temperature_K: float
    pressure_Pa: float
    component_number_density_ratios: tuple[tuple[str, float], ...]
    force_field_name: str
    charge_model: str
    charge_scale: float
    topology_and_force_field_record: tuple[tuple[str, str], ...]
    pair_style: str
    bond_style: str
    angle_style: str
    dihedral_style: str
    improper_style: str
    production_ensemble: str
    equations_of_motion: str
    thermostat_policy: str
    long_range_interaction_model: str
    periodic_boundary_convention: str


@dataclass(frozen=True)
class MicroscopicNumericalRealization:
    molecule_counts: tuple[tuple[str, int], ...]
    box_vectors_A: tuple[tuple[float, float, float], ...]
    timestep_fs: float
    current_stride_steps: int
    trajectory_stride_steps: int
    duration_ps: float
    discrete_integrator: str
    lammps_version: str
    lammps_data_text: str
    lammps_input_text: str


@dataclass(frozen=True)
class MicroscopicTrajectoryEvidence:
    generator: FixedMicroscopicGenerator
    numerical_realization: MicroscopicNumericalRealization
    replica_id: str
    conductivity_current_S_m: float
    conductivity_helfand_S_m: float
    projected_conductivity_S_m: float
    basis_coordinates: tuple[str, ...]
    maximum_dirichlet_residual_score_m2_s: float
    transition_capacity_relative_change: float
    transition_first_moment_relative_change: float
    transition_second_moment_relative_change: float
    effective_sample_size: float
    stationary: bool
    committor_residual: float
    detailed_balance_residual: float


@dataclass(frozen=True)
class MicroscopicConvergenceThresholds:
    confidence_level: float
    timestep_bias_tolerance_S_m: float
    finite_size_tolerance_S_m: float
    duration_tolerance_S_m: float
    current_stride_tolerance_S_m: float
    trajectory_stride_tolerance_S_m: float
    estimator_agreement_tolerance_S_m: float
    replica_tolerance_S_m: float
    basis_tolerance_S_m: float
    dirichlet_residual_tolerance_m2_s: float
    transition_relative_tolerance: float
    committor_residual_tolerance: float
    detailed_balance_residual_tolerance: float
    minimum_effective_sample_size: float


@dataclass(frozen=True)
class MicroscopicConvergenceAudit:
    name: str
    passed: bool
    measured_value: float
    uncertainty: float
    tolerance: float
    compared_records: tuple[str, ...]


@dataclass(frozen=True)
class MicroscopicConvergenceCertificate:
    generator: FixedMicroscopicGenerator
    production_conductivity_S_m: float
    production_standard_error_S_m: float
    audits: tuple[MicroscopicConvergenceAudit, ...]

    @property
    def failed_audits(self) -> tuple[str, ...]:
        return tuple(audit.name for audit in self.audits if not audit.passed)

    @property
    def converged(self) -> bool:
        return not self.failed_audits

    def require_converged(self) -> None:
        if self.failed_audits:
            raise ValueError(
                "microscopic conductivity evidence is not converged: "
                + ", ".join(self.failed_audits)
            )


@dataclass(frozen=True)
class MicroscopicProductionArtifact:
    schema: str
    generator: FixedMicroscopicGenerator
    evidence: tuple[MicroscopicTrajectoryEvidence, ...]
    thresholds: MicroscopicConvergenceThresholds
    projected_primitive_yaml_path: str


@dataclass(frozen=True)
class ValidatedMicroscopicProductionArtifact:
    schema: str
    certificate: MicroscopicConvergenceCertificate
    evidence: tuple[MicroscopicTrajectoryEvidence, ...]
    thresholds: MicroscopicConvergenceThresholds
    projected_primitive_yaml_path: str


def write_microscopic_production_artifact(
    path: Path,
    generator: FixedMicroscopicGenerator,
    evidence: Sequence[MicroscopicTrajectoryEvidence],
    thresholds: MicroscopicConvergenceThresholds,
    projected_primitive_yaml_path: Path,
) -> None:
    evidence_records = tuple(evidence)
    certificate = build_microscopic_convergence_certificate(
        generator=generator,
        evidence=evidence_records,
        thresholds=thresholds,
    )
    certificate.require_converged()
    artifact = MicroscopicProductionArtifact(
        schema=MICROSCOPIC_PRODUCTION_SCHEMA,
        generator=generator,
        evidence=evidence_records,
        thresholds=thresholds,
        projected_primitive_yaml_path=str(projected_primitive_yaml_path),
    )
    write_json_object(
        path,
        asdict(artifact),
        "microscopic production artifact",
    )


def read_microscopic_production_artifact(
    path: Path,
) -> ValidatedMicroscopicProductionArtifact:
    record = read_json_object(path, "microscopic production artifact")
    artifact = TypeAdapter(MicroscopicProductionArtifact).validate_python(record)
    if artifact.schema != MICROSCOPIC_PRODUCTION_SCHEMA:
        raise ValueError(
            f"{path} schema must be {MICROSCOPIC_PRODUCTION_SCHEMA}, "
            f"got {artifact.schema}"
        )
    certificate = build_microscopic_convergence_certificate(
        generator=artifact.generator,
        evidence=artifact.evidence,
        thresholds=artifact.thresholds,
    )
    certificate.require_converged()
    return ValidatedMicroscopicProductionArtifact(
        schema=artifact.schema,
        certificate=certificate,
        evidence=artifact.evidence,
        thresholds=artifact.thresholds,
        projected_primitive_yaml_path=artifact.projected_primitive_yaml_path,
    )


def load_converged_conductivity_for_recipe(
    recipe_record: dict,
    certificate_directory: Path,
):
    requested_recipe = MicroscopicRecipeComposition.from_record(recipe_record)
    requested_temperature_K = _positive_float(
        recipe_record["temperature_K"], "temperature_K"
    )
    matching_artifacts: list[
        tuple[Path, ValidatedMicroscopicProductionArtifact]
    ] = []
    for certificate_path in sorted(certificate_directory.glob("*.json")):
        artifact = read_microscopic_production_artifact(certificate_path)
        generator = artifact.certificate.generator
        if (
            generator.recipe == requested_recipe
            and generator.temperature_K == requested_temperature_K
        ):
            matching_artifacts.append((certificate_path, artifact))
    if len(matching_artifacts) != 1:
        raise ValueError(
            "production conductivity requires exactly one converged microscopic "
            f"certificate for the requested recipe; found {len(matching_artifacts)}"
        )
    certificate_path, artifact = matching_artifacts[0]
    primitive_path = Path(artifact.projected_primitive_yaml_path)
    if not primitive_path.is_absolute():
        primitive_path = certificate_path.parent / primitive_path
    conductivity_result = compute_conductivity_from_primitive_yaml(primitive_path)
    certificate_sigma_mS_cm = (
        artifact.certificate.production_conductivity_S_m * S_M_TO_MS_CM
    )
    if conductivity_result.sigma_mS_cm != certificate_sigma_mS_cm:
        raise ValueError(
            "convergence certificate conductivity does not equal the frozen "
            "projected primitive readout"
        )
    return conductivity_result


@dataclass(frozen=True)
class EinsteinHelfandEstimate:
    conductivity_S_m: float
    standard_error_S_m: float
    block_conductivities_S_m: tuple[float, ...]


@dataclass(frozen=True)
class EinsteinHelfandWindowEvidence:
    fit_lag_start_frames: int
    fit_lag_stop_frames: int
    block_count: int
    conductivity_S_m: float
    standard_error_S_m: float
    log_log_exponent: float
    log_log_exponent_standard_error: float


@dataclass(frozen=True)
class AdaptiveEinsteinHelfandEstimate:
    conductivity_S_m: float
    standard_error_S_m: float
    accepted_windows: tuple[EinsteinHelfandWindowEvidence, ...]


def estimate_adaptive_einstein_helfand_conductivity(
    helfand_moment_C_m: Array,
    frame_interval_s: float,
    volume_m3: float,
    temperature_K: float,
    confidence_level: float,
) -> AdaptiveEinsteinHelfandEstimate:
    moments = np.asarray(helfand_moment_C_m, dtype=float)
    if moments.ndim != 2 or moments.shape[1] != CARTESIAN_COMPONENT_COUNT:
        raise ValueError("helfand_moment_C_m must have shape (frames, 3)")
    confidence_multiplier = NormalDist().inv_cdf(
        0.5 + 0.5 * _confidence_level(confidence_level)
    )
    minimum_regression_point_count = MINIMUM_LINEAR_FIT_SAMPLE_COUNT
    maximum_block_count = moments.shape[0] // (
        2 * minimum_regression_point_count
    )
    if maximum_block_count < 2:
        raise ValueError("Helfand trajectory is too short for adaptive block analysis")
    accepted_windows: list[EinsteinHelfandWindowEvidence] = []
    for block_count in dyadic_values(2, maximum_block_count):
        frames_per_block = moments.shape[0] // block_count
        maximum_lag = frames_per_block - 1
        lag_boundaries = dyadic_values(1, maximum_lag)
        for stop_index in range(1, len(lag_boundaries)):
            stop_lag = lag_boundaries[stop_index]
            for start_index in range(stop_index):
                start_lag = lag_boundaries[start_index]
                if stop_lag - start_lag + 1 < minimum_regression_point_count:
                    continue
                if not _helfand_window_has_positive_block_slopes(
                    moments_C_m=moments,
                    frame_interval_s=frame_interval_s,
                    fit_lag_start_frames=start_lag,
                    fit_lag_stop_frames=stop_lag,
                    block_count=block_count,
                ):
                    continue
                estimate = estimate_einstein_helfand_conductivity(
                    helfand_moment_C_m=moments,
                    frame_interval_s=frame_interval_s,
                    volume_m3=volume_m3,
                    temperature_K=temperature_K,
                    fit_lag_start_frames=start_lag,
                    fit_lag_stop_frames=stop_lag,
                    block_count=block_count,
                )
                lag_frames, mean_square_displacements = _helfand_mean_square_trace(
                    moments_C_m=moments,
                    fit_lag_start_frames=start_lag,
                    fit_lag_stop_frames=stop_lag,
                )
                exponent_fit = linear_fit(
                    np.log(lag_frames.astype(float) * frame_interval_s),
                    np.log(mean_square_displacements),
                )
                exponent_distance = abs(exponent_fit.slope - 1.0)
                exponent_uncertainty = (
                    confidence_multiplier * exponent_fit.slope_standard_error
                )
                if exponent_distance > exponent_uncertainty:
                    continue
                accepted_windows.append(
                    EinsteinHelfandWindowEvidence(
                        fit_lag_start_frames=start_lag,
                        fit_lag_stop_frames=stop_lag,
                        block_count=block_count,
                        conductivity_S_m=estimate.conductivity_S_m,
                        standard_error_S_m=estimate.standard_error_S_m,
                        log_log_exponent=exponent_fit.slope,
                        log_log_exponent_standard_error=(
                            exponent_fit.slope_standard_error
                        ),
                    )
                )
    if len(accepted_windows) < 2:
        raise ValueError(
            "Helfand trajectory has fewer than two statistically diffusive windows"
        )
    conductivity_values = np.asarray(
        [window.conductivity_S_m for window in accepted_windows], dtype=float
    )
    within_window_variance = float(
        np.mean(
            np.square(
                [window.standard_error_S_m for window in accepted_windows]
            )
        )
    )
    between_window_variance = float(np.var(conductivity_values, ddof=1))
    return AdaptiveEinsteinHelfandEstimate(
        conductivity_S_m=float(np.mean(conductivity_values)),
        standard_error_S_m=float(
            np.sqrt(within_window_variance + between_window_variance)
        ),
        accepted_windows=tuple(accepted_windows),
    )


def estimate_einstein_helfand_conductivity(
    helfand_moment_C_m: Array,
    frame_interval_s: float,
    volume_m3: float,
    temperature_K: float,
    fit_lag_start_frames: int,
    fit_lag_stop_frames: int,
    block_count: int,
) -> EinsteinHelfandEstimate:
    moments = np.asarray(helfand_moment_C_m, dtype=float)
    if moments.ndim != 2 or moments.shape[1] != CARTESIAN_COMPONENT_COUNT:
        raise ValueError("helfand_moment_C_m must have shape (frames, 3)")
    if not np.all(np.isfinite(moments)):
        raise ValueError("helfand_moment_C_m must be finite")
    frame_interval = _positive_float(frame_interval_s, "frame_interval_s")
    volume = _positive_float(volume_m3, "volume_m3")
    temperature = _positive_float(temperature_K, "temperature_K")
    if block_count < 2:
        raise ValueError("block_count must be at least two")
    blocks = contiguous_blocks(moments, block_count)
    frames_per_block = blocks[0].shape[0]
    if fit_lag_start_frames < 1 or fit_lag_stop_frames <= fit_lag_start_frames:
        raise ValueError("Helfand fit lag bounds are invalid")
    if fit_lag_stop_frames >= frames_per_block:
        raise ValueError("Helfand fit lag stop must be shorter than each block")
    block_conductivities = tuple(
        _helfand_block_conductivity_S_m(
            block,
            frame_interval,
            volume,
            temperature,
            fit_lag_start_frames,
            fit_lag_stop_frames,
        )
        for block in blocks
    )
    conductivity, standard_error = mean_and_standard_error(block_conductivities)
    return EinsteinHelfandEstimate(
        conductivity_S_m=conductivity,
        standard_error_S_m=standard_error,
        block_conductivities_S_m=block_conductivities,
    )


def _helfand_block_conductivity_S_m(
    moments_C_m: Array,
    frame_interval_s: float,
    volume_m3: float,
    temperature_K: float,
    fit_lag_start_frames: int,
    fit_lag_stop_frames: int,
) -> float:
    lag_frames, mean_square_displacements = _helfand_mean_square_trace(
        moments_C_m=moments_C_m,
        fit_lag_start_frames=fit_lag_start_frames,
        fit_lag_stop_frames=fit_lag_stop_frames,
    )
    lag_times_s = lag_frames.astype(float) * frame_interval_s
    slope_C2_m2_s = linear_fit(
        lag_times_s, mean_square_displacements
    ).slope
    conductivity_S_m = slope_C2_m2_s / (
        EINSTEIN_HELFAND_DENOMINATOR_FACTOR * volume_m3 * K_B * temperature_K
    )
    if not np.isfinite(conductivity_S_m) or conductivity_S_m < 0.0:
        raise ValueError("Helfand diffusive slope must produce finite conductivity")
    return conductivity_S_m


def _helfand_window_has_positive_block_slopes(
    moments_C_m: Array,
    frame_interval_s: float,
    fit_lag_start_frames: int,
    fit_lag_stop_frames: int,
    block_count: int,
) -> bool:
    for block in contiguous_blocks(moments_C_m, block_count):
        lag_frames, mean_square_displacements = _helfand_mean_square_trace(
            moments_C_m=block,
            fit_lag_start_frames=fit_lag_start_frames,
            fit_lag_stop_frames=fit_lag_stop_frames,
        )
        fit = linear_fit(
            lag_frames.astype(float) * frame_interval_s,
            mean_square_displacements,
        )
        if fit.slope <= 0.0:
            return False
    return True


def _helfand_mean_square_trace(
    moments_C_m: Array,
    fit_lag_start_frames: int,
    fit_lag_stop_frames: int,
) -> tuple[Array, Array]:
    lag_frames = np.arange(
        fit_lag_start_frames,
        fit_lag_stop_frames + 1,
        dtype=int,
    )
    mean_square_displacements = np.asarray(
        [
            np.mean(
                np.sum(
                    (moments_C_m[lag_frame:] - moments_C_m[:-lag_frame]) ** 2,
                    axis=1,
                )
            )
            for lag_frame in lag_frames
        ],
        dtype=float,
    )
    if np.any(mean_square_displacements <= 0.0):
        raise ValueError("Helfand mean-square displacement must be positive")
    return lag_frames, mean_square_displacements


def microscopic_charge_current_C_m_s(
    atom_charges_e: Array,
    atom_velocities_A_fs: Array,
) -> Array:
    charges_e = np.asarray(atom_charges_e, dtype=float)
    velocities_A_fs = np.asarray(atom_velocities_A_fs, dtype=float)
    _validate_atom_vectors(charges_e, velocities_A_fs, "velocities")
    return E_CHARGE * ANGSTROM_TO_M / FEMTOSECOND_TO_S * np.einsum(
        "i,ij->j", charges_e, velocities_A_fs
    )


def microscopic_helfand_moment_C_m(
    atom_charges_e: Array,
    atom_unwrapped_positions_A: Array,
) -> Array:
    charges_e = np.asarray(atom_charges_e, dtype=float)
    positions_A = np.asarray(atom_unwrapped_positions_A, dtype=float)
    _validate_atom_vectors(charges_e, positions_A, "positions")
    return E_CHARGE * ANGSTROM_TO_M * np.einsum("i,ij->j", charges_e, positions_A)


def validate_nested_basis_evidence(
    parent: MicroscopicTrajectoryEvidence,
    child: MicroscopicTrajectoryEvidence,
) -> None:
    if child.generator != parent.generator:
        raise ValueError("microscopic_generator_identity_mismatch")
    parent_coordinate_count = len(parent.basis_coordinates)
    if child.basis_coordinates[:parent_coordinate_count] != parent.basis_coordinates:
        raise ValueError("child basis does not contain the parent basis exactly")
    if len(child.basis_coordinates) <= parent_coordinate_count:
        raise ValueError("child basis must add at least one coordinate")


def build_microscopic_convergence_certificate(
    generator: FixedMicroscopicGenerator,
    evidence: Sequence[MicroscopicTrajectoryEvidence],
    thresholds: MicroscopicConvergenceThresholds,
) -> MicroscopicConvergenceCertificate:
    records = tuple(evidence)
    _validate_evidence_structure(generator, records)
    confidence_level = _confidence_level(thresholds.confidence_level)
    confidence_multiplier = NormalDist().inv_cdf(0.5 + 0.5 * confidence_level)
    residual_tolerance = _positive_float(
        thresholds.dirichlet_residual_tolerance_m2_s,
        "dirichlet_residual_tolerance_m2_s",
    )
    transition_tolerance = _positive_float(
        thresholds.transition_relative_tolerance,
        "transition_relative_tolerance",
    )
    committor_tolerance = _positive_float(
        thresholds.committor_residual_tolerance,
        "committor_residual_tolerance",
    )
    detailed_balance_tolerance = _positive_float(
        thresholds.detailed_balance_residual_tolerance,
        "detailed_balance_residual_tolerance",
    )
    minimum_effective_sample_size = _positive_float(
        thresholds.minimum_effective_sample_size,
        "minimum_effective_sample_size",
    )
    finest_coordinate_count = max(len(record.basis_coordinates) for record in records)
    finest_basis_records = tuple(
        record
        for record in records
        if len(record.basis_coordinates) == finest_coordinate_count
    )
    finest = _production_realization_records(finest_basis_records)
    stationarity_failure_count = sum(not record.stationary for record in finest)
    minimum_observed_effective_sample_size = min(
        record.effective_sample_size for record in finest
    )
    maximum_dirichlet_residual = max(
        record.maximum_dirichlet_residual_score_m2_s for record in finest
    )
    maximum_transition_change = _maximum_transition_change(finest)
    maximum_committor_residual = max(record.committor_residual for record in finest)
    maximum_detailed_balance_residual = max(
        record.detailed_balance_residual for record in finest
    )
    audits = (
        _scalar_upper_bound_audit(
            name="equilibrium_stationarity",
            measured_value=float(stationarity_failure_count),
            tolerance=0.0,
            compared_records=_record_labels(finest),
        ),
        _scalar_lower_bound_audit(
            name="effective_sample_size",
            measured_value=minimum_observed_effective_sample_size,
            tolerance=minimum_effective_sample_size,
            compared_records=_record_labels(finest),
        ),
        _paired_estimator_audit(
            evidence=finest,
            left_name="current",
            right_name="helfand",
            name="green_kubo_einstein_helfand_agreement",
            tolerance=_positive_float(
                thresholds.estimator_agreement_tolerance_S_m,
                "estimator_agreement_tolerance_S_m",
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _paired_estimator_audit(
            evidence=finest,
            left_name="projected",
            right_name="helfand",
            name="projected_zero_frequency_agreement",
            tolerance=_positive_float(
                thresholds.estimator_agreement_tolerance_S_m,
                "estimator_agreement_tolerance_S_m",
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _replica_audit(
            evidence=finest,
            tolerance=_positive_float(
                thresholds.replica_tolerance_S_m, "replica_tolerance_S_m"
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _basis_audit(
            evidence=_production_axis_records(records),
            tolerance=_positive_float(
                thresholds.basis_tolerance_S_m, "basis_tolerance_S_m"
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _numerical_axis_audit(
            evidence=finest_basis_records,
            axis="timestep",
            name="timestep_refinement",
            tolerance=_positive_float(
                thresholds.timestep_bias_tolerance_S_m,
                "timestep_bias_tolerance_S_m",
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _numerical_axis_audit(
            evidence=finest_basis_records,
            axis="box",
            name="box_size_refinement",
            tolerance=_positive_float(
                thresholds.finite_size_tolerance_S_m,
                "finite_size_tolerance_S_m",
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _numerical_axis_audit(
            evidence=finest_basis_records,
            axis="duration",
            name="trajectory_duration_refinement",
            tolerance=_positive_float(
                thresholds.duration_tolerance_S_m, "duration_tolerance_S_m"
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _numerical_axis_audit(
            evidence=finest_basis_records,
            axis="current_stride",
            name="current_stride_refinement",
            tolerance=_positive_float(
                thresholds.current_stride_tolerance_S_m,
                "current_stride_tolerance_S_m",
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _numerical_axis_audit(
            evidence=finest_basis_records,
            axis="trajectory_stride",
            name="trajectory_stride_refinement",
            tolerance=_positive_float(
                thresholds.trajectory_stride_tolerance_S_m,
                "trajectory_stride_tolerance_S_m",
            ),
            confidence_multiplier=confidence_multiplier,
        ),
        _scalar_upper_bound_audit(
            name="dirichlet_residual",
            measured_value=maximum_dirichlet_residual,
            tolerance=residual_tolerance,
            compared_records=_record_labels(finest),
        ),
        _scalar_upper_bound_audit(
            name="transition_capacity_and_moments",
            measured_value=maximum_transition_change,
            tolerance=transition_tolerance,
            compared_records=_record_labels(finest),
        ),
        _scalar_upper_bound_audit(
            name="committor_residual",
            measured_value=maximum_committor_residual,
            tolerance=committor_tolerance,
            compared_records=_record_labels(finest),
        ),
        _scalar_upper_bound_audit(
            name="detailed_balance_residual",
            measured_value=maximum_detailed_balance_residual,
            tolerance=detailed_balance_tolerance,
            compared_records=_record_labels(finest),
        ),
    )
    projected_values = np.asarray([record.projected_conductivity_S_m for record in finest])
    conductivity, standard_error = mean_and_standard_error(projected_values)
    return MicroscopicConvergenceCertificate(
        generator=generator,
        production_conductivity_S_m=conductivity,
        production_standard_error_S_m=standard_error,
        audits=audits,
    )


def _paired_estimator_audit(
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
    left_name: str,
    right_name: str,
    name: str,
    tolerance: float,
    confidence_multiplier: float,
) -> MicroscopicConvergenceAudit:
    value_fields = {
        "current": "conductivity_current_S_m",
        "helfand": "conductivity_helfand_S_m",
        "projected": "projected_conductivity_S_m",
    }
    left_values = np.asarray(
        [getattr(record, value_fields[left_name]) for record in evidence]
    )
    right_values = np.asarray(
        [getattr(record, value_fields[right_name]) for record in evidence]
    )
    differences = left_values - right_values
    mean_difference, standard_error = mean_and_standard_error(differences)
    uncertainty = confidence_multiplier * standard_error
    measured_value = abs(mean_difference)
    return MicroscopicConvergenceAudit(
        name=name,
        passed=measured_value + uncertainty <= tolerance,
        measured_value=measured_value,
        uncertainty=uncertainty,
        tolerance=tolerance,
        compared_records=_record_labels(evidence),
    )


def _replica_audit(
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
    tolerance: float,
    confidence_multiplier: float,
) -> MicroscopicConvergenceAudit:
    values = np.asarray([record.projected_conductivity_S_m for record in evidence])
    _, standard_error = mean_and_standard_error(values)
    measured_value = float(np.max(values) - np.min(values))
    uncertainty = confidence_multiplier * standard_error
    return MicroscopicConvergenceAudit(
        name="independent_replica_agreement",
        passed=measured_value + uncertainty <= tolerance,
        measured_value=measured_value,
        uncertainty=uncertainty,
        tolerance=tolerance,
        compared_records=_record_labels(evidence),
    )


def _basis_audit(
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
    tolerance: float,
    confidence_multiplier: float,
) -> MicroscopicConvergenceAudit:
    coordinate_counts = tuple(
        sorted({len(record.basis_coordinates) for record in evidence})
    )
    previous_by_replica = {
        record.replica_id: record.projected_conductivity_S_m
        for record in evidence
        if len(record.basis_coordinates) == coordinate_counts[-2]
    }
    final_by_replica = {
        record.replica_id: record.projected_conductivity_S_m
        for record in evidence
        if len(record.basis_coordinates) == coordinate_counts[-1]
    }
    common_replicas = tuple(sorted(set(previous_by_replica) & set(final_by_replica)))
    if len(common_replicas) < 2:
        raise ValueError("basis convergence requires two paired replicas")
    differences = np.asarray(
        [final_by_replica[key] - previous_by_replica[key] for key in common_replicas]
    )
    mean_difference, standard_error = mean_and_standard_error(differences)
    uncertainty = confidence_multiplier * standard_error
    measured_value = abs(mean_difference)
    return MicroscopicConvergenceAudit(
        name="nested_basis_conductivity",
        passed=measured_value + uncertainty <= tolerance,
        measured_value=measured_value,
        uncertainty=uncertainty,
        tolerance=tolerance,
        compared_records=common_replicas,
    )


def _production_realization_records(
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
) -> tuple[MicroscopicTrajectoryEvidence, ...]:
    production_timestep_fs = min(
        record.numerical_realization.timestep_fs for record in evidence
    )
    production_volume_A3 = max(_box_volume_A3(record) for record in evidence)
    production_duration_ps = max(
        record.numerical_realization.duration_ps for record in evidence
    )
    production_current_stride = min(
        record.numerical_realization.current_stride_steps for record in evidence
    )
    production_trajectory_stride = min(
        record.numerical_realization.trajectory_stride_steps for record in evidence
    )
    records = tuple(
        record
        for record in evidence
        if record.numerical_realization.timestep_fs == production_timestep_fs
        and _box_volume_A3(record) == production_volume_A3
        and record.numerical_realization.duration_ps == production_duration_ps
        and record.numerical_realization.current_stride_steps
        == production_current_stride
        and record.numerical_realization.trajectory_stride_steps
        == production_trajectory_stride
    )
    if len({record.replica_id for record in records}) < 2:
        raise ValueError(
            "finest numerical realization requires two independent replicas"
        )
    return records


def _production_axis_records(
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
) -> tuple[MicroscopicTrajectoryEvidence, ...]:
    production_timestep_fs = min(
        record.numerical_realization.timestep_fs for record in evidence
    )
    production_volume_A3 = max(_box_volume_A3(record) for record in evidence)
    production_duration_ps = max(
        record.numerical_realization.duration_ps for record in evidence
    )
    production_current_stride = min(
        record.numerical_realization.current_stride_steps for record in evidence
    )
    production_trajectory_stride = min(
        record.numerical_realization.trajectory_stride_steps for record in evidence
    )
    return tuple(
        record
        for record in evidence
        if record.numerical_realization.timestep_fs == production_timestep_fs
        and _box_volume_A3(record) == production_volume_A3
        and record.numerical_realization.duration_ps == production_duration_ps
        and record.numerical_realization.current_stride_steps
        == production_current_stride
        and record.numerical_realization.trajectory_stride_steps
        == production_trajectory_stride
    )


def _numerical_axis_audit(
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
    axis: str,
    name: str,
    tolerance: float,
    confidence_multiplier: float,
) -> MicroscopicConvergenceAudit:
    grouped: dict[
        tuple, dict[float, dict[str, MicroscopicTrajectoryEvidence]]
    ] = {}
    for record in evidence:
        comparison_key = _numerical_comparison_key(record, axis)
        axis_value = _numerical_axis_value(record, axis)
        records_by_replica = grouped.setdefault(comparison_key, {}).setdefault(
            axis_value, {}
        )
        if record.replica_id in records_by_replica:
            raise ValueError("numerical refinement contains duplicate replica evidence")
        records_by_replica[record.replica_id] = record
    comparison_upper_bounds: list[float] = []
    comparison_labels: list[str] = []
    for values_by_axis in grouped.values():
        if len(values_by_axis) < 2:
            continue
        ordered_values = sorted(values_by_axis)
        if axis in {"box", "duration"}:
            coarse_value, refined_value = ordered_values[-2:]
        else:
            refined_value, coarse_value = ordered_values[:2]
        coarse_records = values_by_axis[coarse_value]
        refined_records = values_by_axis[refined_value]
        common_replicas = tuple(
            sorted(set(coarse_records) & set(refined_records))
        )
        if len(common_replicas) < 2:
            continue
        differences = np.asarray(
            [
                refined_records[replica_id].projected_conductivity_S_m
                - coarse_records[replica_id].projected_conductivity_S_m
                for replica_id in common_replicas
            ],
            dtype=float,
        )
        mean_difference, standard_error = mean_and_standard_error(differences)
        comparison_upper_bounds.append(
            abs(mean_difference) + confidence_multiplier * standard_error
        )
        comparison_labels.extend(
            f"{axis}:{coarse_value}->{refined_value}:{replica_id}"
            for replica_id in common_replicas
        )
    if not comparison_upper_bounds:
        return MicroscopicConvergenceAudit(
            name=name,
            passed=False,
            measured_value=0.0,
            uncertainty=0.0,
            tolerance=tolerance,
            compared_records=(),
        )
    maximum_upper_bound = max(comparison_upper_bounds)
    return MicroscopicConvergenceAudit(
        name=name,
        passed=maximum_upper_bound <= tolerance,
        measured_value=maximum_upper_bound,
        uncertainty=0.0,
        tolerance=tolerance,
        compared_records=tuple(comparison_labels),
    )


def _numerical_comparison_key(
    record: MicroscopicTrajectoryEvidence,
    axis: str,
) -> tuple:
    realization = record.numerical_realization
    common = (
        realization.discrete_integrator,
        realization.lammps_version,
        record.basis_coordinates,
    )
    comparison_keys = {
        "timestep": common + (
            realization.molecule_counts,
            realization.box_vectors_A,
            realization.current_stride_steps,
            realization.trajectory_stride_steps,
            realization.duration_ps,
        ),
        "box": common + (
            realization.timestep_fs,
            realization.current_stride_steps,
            realization.trajectory_stride_steps,
            realization.duration_ps,
        ),
        "duration": common + (
            realization.molecule_counts,
            realization.box_vectors_A,
            realization.timestep_fs,
            realization.current_stride_steps,
            realization.trajectory_stride_steps,
        ),
        "current_stride": common + (
            realization.molecule_counts,
            realization.box_vectors_A,
            realization.timestep_fs,
            realization.duration_ps,
            realization.trajectory_stride_steps,
        ),
        "trajectory_stride": common + (
            realization.molecule_counts,
            realization.box_vectors_A,
            realization.timestep_fs,
            realization.duration_ps,
            realization.current_stride_steps,
        ),
    }
    return comparison_keys[axis]


def _numerical_axis_value(
    record: MicroscopicTrajectoryEvidence,
    axis: str,
) -> float:
    realization = record.numerical_realization
    axis_values = {
        "timestep": realization.timestep_fs,
        "box": _box_volume_A3(record),
        "duration": realization.duration_ps,
        "current_stride": float(realization.current_stride_steps),
        "trajectory_stride": float(realization.trajectory_stride_steps),
    }
    return axis_values[axis]


def _box_volume_A3(record: MicroscopicTrajectoryEvidence) -> float:
    box_matrix = np.asarray(record.numerical_realization.box_vectors_A, dtype=float)
    if box_matrix.shape != (3, 3):
        raise ValueError("box_vectors_A must be a 3 by 3 matrix")
    volume_A3 = abs(float(np.linalg.det(box_matrix)))
    if not np.isfinite(volume_A3) or volume_A3 <= 0.0:
        raise ValueError("box_vectors_A must define a finite positive volume")
    return volume_A3


def _maximum_transition_change(
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
) -> float:
    return max(
        max(
            record.transition_capacity_relative_change,
            record.transition_first_moment_relative_change,
            record.transition_second_moment_relative_change,
        )
        for record in evidence
    )


def _validate_evidence_structure(
    generator: FixedMicroscopicGenerator,
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
) -> None:
    if len(evidence) < 4:
        raise ValueError("certificate requires two basis levels and two replicas")
    if any(record.generator != generator for record in evidence):
        raise ValueError("microscopic_generator_identity_mismatch")
    coordinate_counts = tuple(
        sorted({len(record.basis_coordinates) for record in evidence})
    )
    if len(coordinate_counts) < 2:
        raise ValueError("certificate requires nested basis levels")
    for coordinate_count in coordinate_counts:
        replicas = {
            record.replica_id
            for record in evidence
            if len(record.basis_coordinates) == coordinate_count
        }
        if len(replicas) < 2:
            raise ValueError("each basis level requires independent replicas")
    records_by_replica_and_realization: dict[
        tuple[str, MicroscopicNumericalRealization],
        list[MicroscopicTrajectoryEvidence],
    ] = {}
    for record in evidence:
        key = (record.replica_id, record.numerical_realization)
        records_by_replica_and_realization.setdefault(key, []).append(record)
    for replica_records in records_by_replica_and_realization.values():
        ordered = tuple(sorted(replica_records, key=_basis_coordinate_count))
        for parent, child in zip(ordered[:-1], ordered[1:], strict=True):
            validate_nested_basis_evidence(parent, child)


def _validate_atom_vectors(charges_e: Array, vectors: Array, label: str) -> None:
    if charges_e.ndim != 1 or vectors.shape != (charges_e.size, 3):
        raise ValueError(f"atom {label} must align with one-dimensional charges")
    if not np.all(np.isfinite(charges_e)) or not np.all(np.isfinite(vectors)):
        raise ValueError(f"microscopic {label} inputs must be finite")


def _basis_coordinate_count(record: MicroscopicTrajectoryEvidence) -> int:
    return len(record.basis_coordinates)


def _confidence_level(value: float) -> float:
    confidence_level = float(value)
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and strictly between zero and one")
    return confidence_level


def _record_labels(
    evidence: tuple[MicroscopicTrajectoryEvidence, ...],
) -> tuple[str, ...]:
    return tuple(
        (
            f"replica={record.replica_id};dt_fs="
            f"{record.numerical_realization.timestep_fs};duration_ps="
            f"{record.numerical_realization.duration_ps};basis="
            f"{len(record.basis_coordinates)}"
        )
        for record in evidence
    )


def _scalar_upper_bound_audit(
    name: str,
    measured_value: float,
    tolerance: float,
    compared_records: tuple[str, ...],
) -> MicroscopicConvergenceAudit:
    return MicroscopicConvergenceAudit(
        name=name,
        passed=measured_value <= tolerance,
        measured_value=measured_value,
        uncertainty=0.0,
        tolerance=tolerance,
        compared_records=compared_records,
    )


def _scalar_lower_bound_audit(
    name: str,
    measured_value: float,
    tolerance: float,
    compared_records: tuple[str, ...],
) -> MicroscopicConvergenceAudit:
    return MicroscopicConvergenceAudit(
        name=name,
        passed=measured_value >= tolerance,
        measured_value=measured_value,
        uncertainty=0.0,
        tolerance=tolerance,
        compared_records=compared_records,
    )


def _positive_float(value: float, label: str) -> float:
    return positive_finite_float(value, label)


def _positive_composition_items(
    record: dict,
    label: str,
) -> tuple[tuple[str, float], ...]:
    items = tuple(
        (str(species_name), _positive_float(value, f"{label}.{species_name}"))
        for species_name, value in sorted(record.items())
    )
    if not items:
        raise ValueError(f"{label} must not be empty")
    return items


def _nonnegative_composition_items(
    record: dict,
    label: str,
) -> tuple[tuple[str, float], ...]:
    items: list[tuple[str, float]] = []
    for species_name, value in sorted(record.items()):
        loading = strict_nonnegative_float(value, f"{label}.{species_name}")
        items.append((str(species_name), loading))
    return tuple(items)
