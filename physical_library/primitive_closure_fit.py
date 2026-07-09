"""Primitive closure interpolation over populated anchor YAMLs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from conductivity.physical_library.projected_primitives_io import (
    ProjectedPrimitiveArtifact,
    read_projected_primitive_yaml,
)
from conductivity.physical_library.projected_analytical_conductivity import (
    ProjectedPrimitiveInput,
)

Array = np.ndarray


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
    anchors = []
    for anchor_index, primitive_yaml_path in enumerate(primitive_yaml_paths):
        feature_vector = _finite_vector(
            feature_vectors[anchor_index],
            f"feature_vectors[{anchor_index}]",
        )
        if feature_vector.size != scales.size:
            raise ValueError("feature_vector dimension must match length_scales")
        anchors.append(
            PrimitiveClosureAnchor(
                feature_vector=feature_vector,
                artifact=read_projected_primitive_yaml(primitive_yaml_path),
            )
        )
    _validate_anchor_shapes(tuple(anchors))
    return PrimitiveClosureFit(anchors=tuple(anchors), length_scales=scales)


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


def _kernel_weights(closure_fit: PrimitiveClosureFit, query: Array) -> Array:
    squared_distances = []
    for anchor in closure_fit.anchors:
        scaled_difference = (query - anchor.feature_vector) / closure_fit.length_scales
        squared_distances.append(float(scaled_difference @ scaled_difference))
    unnormalized = np.exp(-0.5 * np.asarray(squared_distances, dtype=float))
    weight_sum = float(np.sum(unnormalized))
    if weight_sum <= 0.0:
        raise ValueError("primitive closure kernel weights underflowed")
    return unnormalized / weight_sum


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
