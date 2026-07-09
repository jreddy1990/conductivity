"""YAML I/O for projected conductivity primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from conductivity.physical_library.projected_analytical_conductivity import (
    ProjectedConductivityResult,
    ProjectedPrimitiveInput,
    compute_finite_state_memory_correction,
    compute_projected_analytical_conductivity_from_primitives,
    compute_reversible_generator,
    validate_primitive_input,
    validate_reversible_generator,
)
from conductivity.physical_library.trajectory_primitives import (
    diagnose_finite_process_legality,
)

Array = np.ndarray
PRIMITIVE_SCHEMA = "projected_primitives_v1"
PROJECTED_READOUT_SUCCEEDED = "succeeded"
PROJECTED_READOUT_DIRECT_ONLY = "direct_only"
PROJECTED_READOUT_FAILED = "failed"
VALID_PROJECTED_READOUT_STATUSES = (
    PROJECTED_READOUT_SUCCEEDED,
    PROJECTED_READOUT_DIRECT_ONLY,
    PROJECTED_READOUT_FAILED,
)


@dataclass(frozen=True)
class ProjectedPrimitiveArtifact:
    schema: str
    state_labels: tuple[str, ...]
    primitive_input: ProjectedPrimitiveInput


def read_projected_primitive_yaml(path: Path) -> ProjectedPrimitiveArtifact:
    """Read projected primitive tensors from YAML."""

    record = yaml.safe_load(path.read_text())
    if not isinstance(record, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    schema = str(record["schema"])
    if schema != PRIMITIVE_SCHEMA:
        raise ValueError(f"{path} schema must be {PRIMITIVE_SCHEMA}, got {schema}")
    projected_readout_status = str(record["projected_readout_status"])
    if projected_readout_status not in VALID_PROJECTED_READOUT_STATUSES:
        raise ValueError(
            f"{path} projected_readout_status must be one of "
            f"{VALID_PROJECTED_READOUT_STATUSES}, got {projected_readout_status}"
        )
    if projected_readout_status == PROJECTED_READOUT_FAILED:
        failure_reason = str(record["failure_reason"])
        diagnostics = record["diagnostics"]
        component_drift_residuals = diagnostics["component_drift_residuals"]
        raise ValueError(
            "projected primitive artifact is invalid for readout: "
            f"{projected_readout_status}; failure_reason={failure_reason}; "
            f"component_drift_residuals={component_drift_residuals}"
        )
    if projected_readout_status == PROJECTED_READOUT_DIRECT_ONLY:
        diagnostics = record["diagnostics"]
        direct_only_reasons = diagnostics["direct_only_reasons"]
        if not direct_only_reasons:
            raise ValueError("direct_only artifact requires direct_only_reasons")
    _projected_sigma_mS_cm(record, path)
    state_labels = tuple(str(label) for label in record["state_labels"])
    primitives = record["primitives"]
    memory_matrix_A = _array(
        primitives["mori_memory_matrix_A"],
        "mori_memory_matrix_A",
    )
    current_coupling_matrix_h = _array(
        primitives["mori_current_coupling_matrix_h"],
        "mori_current_coupling_matrix_h",
    )
    if memory_matrix_A.size == 0:
        memory_matrix_A = np.zeros((0, 0), dtype=float)
    if current_coupling_matrix_h.size == 0:
        current_coupling_matrix_h = np.zeros((0, 3), dtype=float)
    primitive_input = ProjectedPrimitiveInput(
        state_concentrations_mol_m3=_array(
            primitives["state_concentrations_mol_m3"],
            "state_concentrations_mol_m3",
        ),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=_array(
            primitives["symmetric_capacity_fluxes_K_ij_mol_m3_s"],
            "symmetric_capacity_fluxes_K_ij_mol_m3_s",
        ),
        transition_first_moments_d_ij_m=_array(
            primitives["transition_first_moments_d_ij_m"],
            "transition_first_moments_d_ij_m",
        ),
        transition_second_moments_M_ij_m2=_array(
            primitives["transition_second_moments_M_ij_m2"],
            "transition_second_moments_M_ij_m2",
        ),
        self_current_tensors_D_self_i_m2_s=_array(
            primitives["self_current_tensors_D_self_i_m2_s"],
            "self_current_tensors_D_self_i_m2_s",
        ),
        mori_memory_matrix_A=memory_matrix_A,
        mori_current_coupling_matrix_h=current_coupling_matrix_h,
        temperature_K=float(record["temperature_K"]),
        volume_m3=float(record["volume_m3"]),
    )
    validate_projected_primitive_artifact_input(primitive_input)
    return ProjectedPrimitiveArtifact(
        schema=schema,
        state_labels=state_labels,
        primitive_input=primitive_input,
    )


def validate_projected_primitive_artifact_input(
    primitive_input: ProjectedPrimitiveInput,
) -> None:
    """Validate that stored primitives form a legal finite projected process."""

    (
        state_concentrations_mol_m3,
        symmetric_capacity_fluxes_K_ij_mol_m3_s,
        transition_first_moments_d_ij_m,
        _transition_second_moments_M_ij_m2,
        _self_current_tensors_D_self_i_m2_s,
        _mori_memory_matrix_A,
        _mori_current_coupling_matrix_h,
    ) = validate_primitive_input(primitive_input)
    reversible_generator_Q_ij_s_inv = compute_reversible_generator(
        symmetric_capacity_fluxes_K_ij_mol_m3_s,
        state_concentrations_mol_m3,
    )
    validate_reversible_generator(
        reversible_generator_Q_ij_s_inv,
        state_concentrations_mol_m3,
    )
    diagnose_finite_process_legality(
        tuple(
            f"state_{state_index}"
            for state_index in range(state_concentrations_mol_m3.size)
        ),
        state_concentrations_mol_m3,
        symmetric_capacity_fluxes_K_ij_mol_m3_s,
        transition_first_moments_d_ij_m,
        _transition_second_moments_M_ij_m2,
        np.zeros(
            (
                state_concentrations_mol_m3.size,
                state_concentrations_mol_m3.size,
            ),
            dtype=int,
        ),
    )
    compute_finite_state_memory_correction(
        state_concentrations_mol_m3,
        reversible_generator_Q_ij_s_inv,
        transition_first_moments_d_ij_m,
    )


def write_projected_primitive_yaml(
    path: Path,
    state_labels: tuple[str, ...],
    primitive_input: ProjectedPrimitiveInput,
    conductivity_result: ProjectedConductivityResult,
) -> None:
    """Write projected primitive tensors and readout diagnostics to YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": PRIMITIVE_SCHEMA,
        "state_labels": list(state_labels),
        "temperature_K": float(primitive_input.temperature_K),
        "volume_m3": float(primitive_input.volume_m3),
        "primitives": {
            "state_concentrations_mol_m3": np.asarray(
                primitive_input.state_concentrations_mol_m3,
                dtype=float,
            ).tolist(),
            "symmetric_capacity_fluxes_K_ij_mol_m3_s": np.asarray(
                primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s,
                dtype=float,
            ).tolist(),
            "transition_first_moments_d_ij_m": np.asarray(
                primitive_input.transition_first_moments_d_ij_m,
                dtype=float,
            ).tolist(),
            "transition_second_moments_M_ij_m2": np.asarray(
                primitive_input.transition_second_moments_M_ij_m2,
                dtype=float,
            ).tolist(),
            "self_current_tensors_D_self_i_m2_s": np.asarray(
                primitive_input.self_current_tensors_D_self_i_m2_s,
                dtype=float,
            ).tolist(),
            "mori_memory_matrix_A": np.asarray(
                primitive_input.mori_memory_matrix_A,
                dtype=float,
            ).tolist(),
            "mori_current_coupling_matrix_h": np.asarray(
                primitive_input.mori_current_coupling_matrix_h,
                dtype=float,
            ).tolist(),
        },
        "projected_readout_status": _projected_readout_status_from_result(
            conductivity_result
        ),
        "sigma_mS_cm": float(conductivity_result.sigma_mS_cm),
        "sigma_S_m": float(conductivity_result.sigma_S_m),
    }
    if record["projected_readout_status"] == PROJECTED_READOUT_DIRECT_ONLY:
        record["diagnostics"] = {
            "direct_only_reasons": tuple(
                conductivity_result.effect_attribution[
                    "finite_process_not_complete_reasons"
                ]
            ),
            "active_transition_capacity_flux_count": int(
                conductivity_result.effect_attribution[
                    "active_transition_capacity_flux_count"
                ]
            ),
            "active_transition_first_moment_count": int(
                conductivity_result.effect_attribution[
                    "active_transition_first_moment_count"
                ]
            ),
            "active_transition_second_moment_count": int(
                conductivity_result.effect_attribution[
                    "active_transition_second_moment_count"
                ]
            ),
        }
    path.write_text(yaml.safe_dump(record, sort_keys=False))


def write_failed_projected_primitive_yaml(
    path: Path,
    schema: str,
    failure_reason: str,
    diagnostics: dict,
) -> None:
    """Write a failed primitive artifact that remains invalid for readout."""

    if not schema:
        raise ValueError("schema must not be empty")
    if not failure_reason:
        raise ValueError("failure_reason must not be empty")
    if "component_drift_residuals" not in diagnostics:
        raise KeyError("diagnostics must include component_drift_residuals")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": schema,
        "projected_readout_status": PROJECTED_READOUT_FAILED,
        "failure_reason": failure_reason,
        "diagnostics": diagnostics,
    }
    path.write_text(yaml.safe_dump(record, sort_keys=False))


def compute_conductivity_from_primitive_yaml(path: Path) -> ProjectedConductivityResult:
    """Read primitive YAML and run the projected conductivity readout."""

    artifact = read_projected_primitive_yaml(path)
    primitive_input = artifact.primitive_input
    return compute_projected_analytical_conductivity_from_primitives(
        primitive_input.state_concentrations_mol_m3,
        primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s,
        primitive_input.transition_first_moments_d_ij_m,
        primitive_input.transition_second_moments_M_ij_m2,
        primitive_input.self_current_tensors_D_self_i_m2_s,
        primitive_input.mori_memory_matrix_A,
        primitive_input.mori_current_coupling_matrix_h,
        primitive_input.temperature_K,
        primitive_input.volume_m3,
    )


@dataclass(frozen=True)
class PrimitiveClosureAnchor:
    feature_vector: Array
    artifact: ProjectedPrimitiveArtifact


@dataclass(frozen=True)
class PrimitiveClosureFit:
    anchors: tuple[PrimitiveClosureAnchor, ...]
    length_scales: Array


def load_closure_fit(
    primitive_yaml_paths: tuple[Path, ...],
    feature_vectors: tuple[Array, ...],
    length_scales: Array,
) -> PrimitiveClosureFit:
    """Load primitive anchors for deterministic kernel interpolation."""

    if len(primitive_yaml_paths) != len(feature_vectors):
        raise ValueError("primitive_yaml_paths and feature_vectors length mismatch")
    if not primitive_yaml_paths:
        raise ValueError("at least one primitive anchor is required")
    scales = _positive_vector(length_scales, "length_scales")
    anchors = tuple(
        _primitive_closure_anchor(
            primitive_yaml_path,
            feature_vectors[anchor_index],
            anchor_index,
            scales.size,
        )
        for anchor_index, primitive_yaml_path in enumerate(primitive_yaml_paths)
    )
    _validate_anchor_shapes(tuple(anchors))
    return PrimitiveClosureFit(anchors=anchors, length_scales=scales)


def interpolate_primitive_closure(
    closure_fit: PrimitiveClosureFit,
    query_feature_vector: Array,
) -> ProjectedPrimitiveInput:
    """Interpolate primitive tensors from populated anchor primitive YAMLs."""

    query = _finite_vector(query_feature_vector, "query_feature_vector")
    if query.size != closure_fit.length_scales.size:
        raise ValueError("query_feature_vector dimension must match length_scales")
    weights = _kernel_weights(closure_fit, query)
    primitive_inputs = tuple(anchor.artifact.primitive_input for anchor in closure_fit.anchors)
    return ProjectedPrimitiveInput(
        state_concentrations_mol_m3=_weighted_sum(
            weights,
            tuple(
                primitive_input.state_concentrations_mol_m3
                for primitive_input in primitive_inputs
            ),
        ),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=_weighted_sum(
            weights,
            tuple(
                primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s
                for primitive_input in primitive_inputs
            ),
        ),
        transition_first_moments_d_ij_m=_weighted_sum(
            weights,
            tuple(
                primitive_input.transition_first_moments_d_ij_m
                for primitive_input in primitive_inputs
            ),
        ),
        transition_second_moments_M_ij_m2=_weighted_sum(
            weights,
            tuple(
                primitive_input.transition_second_moments_M_ij_m2
                for primitive_input in primitive_inputs
            ),
        ),
        self_current_tensors_D_self_i_m2_s=_weighted_sum(
            weights,
            tuple(
                primitive_input.self_current_tensors_D_self_i_m2_s
                for primitive_input in primitive_inputs
            ),
        ),
        mori_memory_matrix_A=_weighted_sum(
            weights,
            tuple(primitive_input.mori_memory_matrix_A for primitive_input in primitive_inputs),
        ),
        mori_current_coupling_matrix_h=_weighted_sum(
            weights,
            tuple(
                primitive_input.mori_current_coupling_matrix_h
                for primitive_input in primitive_inputs
            ),
        ),
        temperature_K=float(
            np.dot(
                weights,
                np.asarray(
                    [primitive_input.temperature_K for primitive_input in primitive_inputs],
                    dtype=float,
                ),
            )
        ),
        volume_m3=float(
            np.dot(
                weights,
                np.asarray(
                    [primitive_input.volume_m3 for primitive_input in primitive_inputs],
                    dtype=float,
                ),
            )
        ),
    )


def _primitive_closure_anchor(
    primitive_yaml_path: Path,
    feature_vector: Array,
    anchor_index: int,
    feature_dimension: int,
) -> PrimitiveClosureAnchor:
    validated_feature_vector = _finite_vector(
        feature_vector,
        f"feature_vectors[{anchor_index}]",
    )
    if validated_feature_vector.size != feature_dimension:
        raise ValueError("feature_vector dimension must match length_scales")
    return PrimitiveClosureAnchor(
        feature_vector=validated_feature_vector,
        artifact=read_projected_primitive_yaml(primitive_yaml_path),
    )


@dataclass(frozen=True)
class PrimitiveTensorNorms:
    c_norm_mol_m3: float
    K_norm_mol_m3_s: float
    Q_norm_s_inv: float
    d_norm_m: float
    M_norm_m2: float
    D_self_norm_m2_s: float
    A_norm: float
    h_norm: float


@dataclass(frozen=True)
class PrimitiveTensorGaps:
    c_gap_mol_m3: float
    K_gap_mol_m3_s: float
    Q_gap_s_inv: float
    d_gap_m: float
    M_gap_m2: float
    D_self_gap_m2_s: float
    A_gap: float
    h_gap: float


@dataclass(frozen=True)
class PrimitiveScalarEstimateValue:
    sigma_mS_cm: float

    def validated(self, scalar_name: str) -> "PrimitiveScalarEstimateValue":
        sigma_mS_cm = float(self.sigma_mS_cm)
        if not np.isfinite(sigma_mS_cm):
            raise ValueError(f"{scalar_name} sigma_mS_cm must be finite")
        return PrimitiveScalarEstimateValue(sigma_mS_cm=sigma_mS_cm)


@dataclass(frozen=True)
class PrimitiveScalarEstimateNotProvided:
    scalar_name: str

    def validated(self, scalar_name: str) -> "PrimitiveScalarEstimateNotProvided":
        if self.scalar_name != scalar_name:
            raise ValueError(f"{scalar_name} scalar estimate name mismatch")
        return self


@dataclass(frozen=True)
class PrimitiveExternalScalarInput:
    green_kubo: PrimitiveScalarEstimateValue | PrimitiveScalarEstimateNotProvided
    einstein_helfand: PrimitiveScalarEstimateValue | PrimitiveScalarEstimateNotProvided


@dataclass(frozen=True)
class PrimitiveScalarReadout:
    finite_projected_sigma_mS_cm: float
    green_kubo: PrimitiveScalarEstimateValue | PrimitiveScalarEstimateNotProvided
    einstein_helfand: PrimitiveScalarEstimateValue | PrimitiveScalarEstimateNotProvided


@dataclass(frozen=True)
class PrimitiveScalarGapValue:
    gap_mS_cm: float


@dataclass(frozen=True)
class PrimitiveScalarGapNotComputed:
    scalar_name: str


@dataclass(frozen=True)
class PrimitiveScalarGaps:
    finite_projected_sigma_gap_mS_cm: float
    green_kubo: PrimitiveScalarGapValue | PrimitiveScalarGapNotComputed
    einstein_helfand: PrimitiveScalarGapValue | PrimitiveScalarGapNotComputed


@dataclass(frozen=True)
class PrimitiveOracleAuditReport:
    trajectory_norms: PrimitiveTensorNorms
    projected_norms: PrimitiveTensorNorms
    recipe_norms: PrimitiveTensorNorms
    projection_gap: PrimitiveTensorGaps
    recipe_primitive_gap: PrimitiveTensorGaps
    scalar_gap: PrimitiveScalarGaps
    trajectory_scalar_readout: PrimitiveScalarReadout
    projected_scalar_readout: PrimitiveScalarReadout
    recipe_scalar_readout: PrimitiveScalarReadout


def audit_primitive_oracle_closure(
    trajectory_primitives: ProjectedPrimitiveInput,
    projected_primitives: ProjectedPrimitiveInput,
    recipe_primitives: ProjectedPrimitiveInput,
    trajectory_scalar_input: PrimitiveExternalScalarInput,
    projected_scalar_input: PrimitiveExternalScalarInput,
    recipe_scalar_input: PrimitiveExternalScalarInput,
) -> PrimitiveOracleAuditReport:
    """Compare finite trajectory, projected, and recipe primitive closures."""

    validate_projected_primitive_artifact_input(trajectory_primitives)
    validate_projected_primitive_artifact_input(projected_primitives)
    validate_projected_primitive_artifact_input(recipe_primitives)
    trajectory_scalar_readout = _scalar_readout(
        trajectory_primitives,
        trajectory_scalar_input,
    )
    projected_scalar_readout = _scalar_readout(
        projected_primitives,
        projected_scalar_input,
    )
    recipe_scalar_readout = _scalar_readout(
        recipe_primitives,
        recipe_scalar_input,
    )
    return PrimitiveOracleAuditReport(
        trajectory_norms=_primitive_norms(trajectory_primitives),
        projected_norms=_primitive_norms(projected_primitives),
        recipe_norms=_primitive_norms(recipe_primitives),
        projection_gap=_primitive_gaps(trajectory_primitives, projected_primitives),
        recipe_primitive_gap=_primitive_gaps(projected_primitives, recipe_primitives),
        scalar_gap=PrimitiveScalarGaps(
            finite_projected_sigma_gap_mS_cm=abs(
                trajectory_scalar_readout.finite_projected_sigma_mS_cm
                - recipe_scalar_readout.finite_projected_sigma_mS_cm
            ),
            green_kubo=_scalar_gap(
                trajectory_scalar_readout.green_kubo,
                recipe_scalar_readout.green_kubo,
                "green_kubo",
            ),
            einstein_helfand=_scalar_gap(
                trajectory_scalar_readout.einstein_helfand,
                recipe_scalar_readout.einstein_helfand,
                "einstein_helfand",
            ),
        ),
        trajectory_scalar_readout=trajectory_scalar_readout,
        projected_scalar_readout=projected_scalar_readout,
        recipe_scalar_readout=recipe_scalar_readout,
    )


def audit_primitive_oracle_closure_from_yaml(
    trajectory_primitive_yaml_path: Path,
    projected_primitive_yaml_path: Path,
    recipe_primitive_yaml_path: Path,
    trajectory_scalar_input: PrimitiveExternalScalarInput,
    projected_scalar_input: PrimitiveExternalScalarInput,
    recipe_scalar_input: PrimitiveExternalScalarInput,
) -> PrimitiveOracleAuditReport:
    """Read validated primitive YAML anchors and run the closure audit."""

    trajectory_artifact = read_projected_primitive_yaml(trajectory_primitive_yaml_path)
    projected_artifact = read_projected_primitive_yaml(projected_primitive_yaml_path)
    recipe_artifact = read_projected_primitive_yaml(recipe_primitive_yaml_path)
    return audit_primitive_oracle_closure(
        trajectory_artifact.primitive_input,
        projected_artifact.primitive_input,
        recipe_artifact.primitive_input,
        trajectory_scalar_input,
        projected_scalar_input,
        recipe_scalar_input,
    )


def _kernel_weights(closure_fit: PrimitiveClosureFit, query: Array) -> Array:
    squared_distances = tuple(
        _scaled_squared_distance(query, anchor.feature_vector, closure_fit.length_scales)
        for anchor in closure_fit.anchors
    )
    unnormalized = np.exp(-0.5 * np.asarray(squared_distances, dtype=float))
    weight_sum = float(np.sum(unnormalized))
    if weight_sum <= 0.0:
        raise ValueError("primitive closure kernel weights underflowed")
    return unnormalized / weight_sum


def _scaled_squared_distance(
    query: Array,
    feature_vector: Array,
    length_scales: Array,
) -> float:
    scaled_difference = (query - feature_vector) / length_scales
    return float(scaled_difference @ scaled_difference)


def _weighted_sum(weights: Array, arrays: tuple[Array, ...]) -> Array:
    result = np.zeros_like(np.asarray(arrays[0], dtype=float), dtype=float)
    for anchor_index, array in enumerate(arrays):
        current = np.asarray(array, dtype=float)
        if current.shape != result.shape:
            raise ValueError("all primitive tensors must have matching shapes")
        result += float(weights[anchor_index]) * current
    return result


def _validate_anchor_shapes(anchors: tuple[PrimitiveClosureAnchor, ...]) -> None:
    reference_labels = anchors[0].artifact.state_labels
    reference_input = anchors[0].artifact.primitive_input
    reference_shapes = _primitive_shapes(reference_input)
    for anchor in anchors[1:]:
        if anchor.artifact.state_labels != reference_labels:
            raise ValueError("primitive anchors must have identical state labels")
        if _primitive_shapes(anchor.artifact.primitive_input) != reference_shapes:
            raise ValueError("primitive anchors must have matching tensor shapes")


def _primitive_shapes(primitive_input: ProjectedPrimitiveInput) -> tuple[tuple[int, ...], ...]:
    return (
        primitive_input.state_concentrations_mol_m3.shape,
        primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s.shape,
        primitive_input.transition_first_moments_d_ij_m.shape,
        primitive_input.transition_second_moments_M_ij_m2.shape,
        primitive_input.self_current_tensors_D_self_i_m2_s.shape,
        primitive_input.mori_memory_matrix_A.shape,
        primitive_input.mori_current_coupling_matrix_h.shape,
    )


def _primitive_norms(primitive_input: ProjectedPrimitiveInput) -> PrimitiveTensorNorms:
    (
        state_concentrations_mol_m3,
        symmetric_capacity_fluxes_K_ij_mol_m3_s,
        transition_first_moments_d_ij_m,
        transition_second_moments_M_ij_m2,
        self_current_tensors_D_self_i_m2_s,
        mori_memory_matrix_A,
        mori_current_coupling_matrix_h,
    ) = validate_primitive_input(primitive_input)
    reversible_generator_Q_ij_s_inv = compute_reversible_generator(
        symmetric_capacity_fluxes_K_ij_mol_m3_s,
        state_concentrations_mol_m3,
    )
    return PrimitiveTensorNorms(
        c_norm_mol_m3=float(np.linalg.norm(state_concentrations_mol_m3)),
        K_norm_mol_m3_s=float(
            np.linalg.norm(symmetric_capacity_fluxes_K_ij_mol_m3_s)
        ),
        Q_norm_s_inv=float(np.linalg.norm(reversible_generator_Q_ij_s_inv)),
        d_norm_m=float(np.linalg.norm(transition_first_moments_d_ij_m)),
        M_norm_m2=float(np.linalg.norm(transition_second_moments_M_ij_m2)),
        D_self_norm_m2_s=float(np.linalg.norm(self_current_tensors_D_self_i_m2_s)),
        A_norm=float(np.linalg.norm(mori_memory_matrix_A)),
        h_norm=float(np.linalg.norm(mori_current_coupling_matrix_h)),
    )


def _primitive_gaps(
    first_input: ProjectedPrimitiveInput,
    second_input: ProjectedPrimitiveInput,
) -> PrimitiveTensorGaps:
    _require_matching_primitive_shapes(first_input, second_input)
    (
        first_state_concentrations_mol_m3,
        first_symmetric_capacity_fluxes_K_ij_mol_m3_s,
        first_transition_first_moments_d_ij_m,
        first_transition_second_moments_M_ij_m2,
        first_self_current_tensors_D_self_i_m2_s,
        first_mori_memory_matrix_A,
        first_mori_current_coupling_matrix_h,
    ) = validate_primitive_input(first_input)
    (
        second_state_concentrations_mol_m3,
        second_symmetric_capacity_fluxes_K_ij_mol_m3_s,
        second_transition_first_moments_d_ij_m,
        second_transition_second_moments_M_ij_m2,
        second_self_current_tensors_D_self_i_m2_s,
        second_mori_memory_matrix_A,
        second_mori_current_coupling_matrix_h,
    ) = validate_primitive_input(second_input)
    first_generator_Q_ij_s_inv = compute_reversible_generator(
        first_symmetric_capacity_fluxes_K_ij_mol_m3_s,
        first_state_concentrations_mol_m3,
    )
    second_generator_Q_ij_s_inv = compute_reversible_generator(
        second_symmetric_capacity_fluxes_K_ij_mol_m3_s,
        second_state_concentrations_mol_m3,
    )
    return PrimitiveTensorGaps(
        c_gap_mol_m3=float(
            np.linalg.norm(
                first_state_concentrations_mol_m3 - second_state_concentrations_mol_m3
            )
        ),
        K_gap_mol_m3_s=float(
            np.linalg.norm(
                first_symmetric_capacity_fluxes_K_ij_mol_m3_s
                - second_symmetric_capacity_fluxes_K_ij_mol_m3_s
            )
        ),
        Q_gap_s_inv=float(
            np.linalg.norm(first_generator_Q_ij_s_inv - second_generator_Q_ij_s_inv)
        ),
        d_gap_m=float(
            np.linalg.norm(
                first_transition_first_moments_d_ij_m
                - second_transition_first_moments_d_ij_m
            )
        ),
        M_gap_m2=float(
            np.linalg.norm(
                first_transition_second_moments_M_ij_m2
                - second_transition_second_moments_M_ij_m2
            )
        ),
        D_self_gap_m2_s=float(
            np.linalg.norm(
                first_self_current_tensors_D_self_i_m2_s
                - second_self_current_tensors_D_self_i_m2_s
            )
        ),
        A_gap=float(
            np.linalg.norm(first_mori_memory_matrix_A - second_mori_memory_matrix_A)
        ),
        h_gap=float(
            np.linalg.norm(
                first_mori_current_coupling_matrix_h
                - second_mori_current_coupling_matrix_h
            )
        ),
    )


def _scalar_readout(
    primitive_input: ProjectedPrimitiveInput,
    scalar_input: PrimitiveExternalScalarInput,
) -> PrimitiveScalarReadout:
    finite_projected_sigma_mS_cm = float(
        compute_projected_analytical_conductivity_from_primitives(
            primitive_input.state_concentrations_mol_m3,
            primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s,
            primitive_input.transition_first_moments_d_ij_m,
            primitive_input.transition_second_moments_M_ij_m2,
            primitive_input.self_current_tensors_D_self_i_m2_s,
            primitive_input.mori_memory_matrix_A,
            primitive_input.mori_current_coupling_matrix_h,
            primitive_input.temperature_K,
            primitive_input.volume_m3,
        ).sigma_mS_cm
    )
    if not np.isfinite(finite_projected_sigma_mS_cm):
        raise ValueError("finite projected sigma_mS_cm must be finite")
    return PrimitiveScalarReadout(
        finite_projected_sigma_mS_cm=finite_projected_sigma_mS_cm,
        green_kubo=scalar_input.green_kubo.validated("green_kubo"),
        einstein_helfand=scalar_input.einstein_helfand.validated("einstein_helfand"),
    )


def _require_matching_primitive_shapes(
    first_input: ProjectedPrimitiveInput,
    second_input: ProjectedPrimitiveInput,
) -> None:
    first_shapes = tuple(
        primitive_array.shape for primitive_array in validate_primitive_input(first_input)
    )
    second_shapes = tuple(
        primitive_array.shape for primitive_array in validate_primitive_input(second_input)
    )
    if first_shapes != second_shapes:
        raise ValueError("primitive inputs must have matching tensor shapes")


def _scalar_gap(
    trajectory_estimate: PrimitiveScalarEstimateValue | PrimitiveScalarEstimateNotProvided,
    recipe_estimate: PrimitiveScalarEstimateValue | PrimitiveScalarEstimateNotProvided,
    scalar_name: str,
) -> PrimitiveScalarGapValue | PrimitiveScalarGapNotComputed:
    if isinstance(trajectory_estimate, PrimitiveScalarEstimateValue) and isinstance(
        recipe_estimate,
        PrimitiveScalarEstimateValue,
    ):
        return PrimitiveScalarGapValue(
            gap_mS_cm=abs(
                trajectory_estimate.sigma_mS_cm - recipe_estimate.sigma_mS_cm
            ),
        )
    return PrimitiveScalarGapNotComputed(scalar_name=scalar_name)


def _array(value, label: str) -> Array:
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _projected_sigma_mS_cm(record: dict, path: Path) -> float:
    sigma_field = _projected_sigma_field(record)
    projected_sigma_mS_cm = float(record[sigma_field])
    if not np.isfinite(projected_sigma_mS_cm):
        raise ValueError(f"{path} {sigma_field} must be finite")
    return projected_sigma_mS_cm


def _projected_sigma_field(record: dict) -> str:
    if "sigma_mS_cm" in record:
        return "sigma_mS_cm"
    if "projected_analytical_sigma_mS_cm" in record:
        return "projected_analytical_sigma_mS_cm"
    raise KeyError(
        "projected primitive artifact missing finite projected sigma field "
        "sigma_mS_cm or projected_analytical_sigma_mS_cm"
    )


def _projected_readout_status_from_result(
    conductivity_result: ProjectedConductivityResult,
) -> str:
    effect_attribution = conductivity_result.effect_attribution
    if "finite_process_readout_status" not in effect_attribution:
        raise KeyError("conductivity result missing finite_process_readout_status")
    status = str(effect_attribution["finite_process_readout_status"])
    if status == "projected":
        return PROJECTED_READOUT_SUCCEEDED
    if status == PROJECTED_READOUT_DIRECT_ONLY:
        return PROJECTED_READOUT_DIRECT_ONLY
    raise ValueError(f"unsupported finite_process_readout_status: {status}")


def _finite_vector(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 1D array")
    return result


def _positive_vector(array: Array, label: str) -> Array:
    result = _finite_vector(array, label)
    if np.any(result <= 0.0):
        raise ValueError(f"{label} entries must be positive")
    return result
