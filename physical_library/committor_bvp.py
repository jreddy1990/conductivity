"""Deterministic committor boundary-value solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from constants import R

Array = np.ndarray


@dataclass(frozen=True)
class OneDimensionalCommittorInput:
    grid_points: Array
    free_energy_J_mol: Array
    diffusivity_m2_s: Array
    temperature_K: float
    left_state_index: int
    right_state_index: int


@dataclass(frozen=True)
class OneDimensionalCommittorResult:
    committor: Array
    committor_gradient: Array
    capacity_integral: float
    log_capacity_integral: float
    quadrature_points: Array
    quadrature_weights: Array


def solve_one_dimensional_committor(
    committor_input: OneDimensionalCommittorInput,
) -> OneDimensionalCommittorResult:
    """Solve the 1D reversible Smoluchowski committor on a grid."""

    grid_points = _strictly_increasing_vector(
        committor_input.grid_points,
        "grid_points",
    )
    free_energy = _same_size_vector(
        committor_input.free_energy_J_mol,
        grid_points.size,
        "free_energy_J_mol",
    )
    diffusivity = _positive_same_size_vector(
        committor_input.diffusivity_m2_s,
        grid_points.size,
        "diffusivity_m2_s",
    )
    if committor_input.left_state_index < 0:
        raise ValueError("left_state_index must be nonnegative")
    if committor_input.right_state_index >= grid_points.size:
        raise ValueError("right_state_index is outside grid")
    if committor_input.left_state_index >= committor_input.right_state_index:
        raise ValueError("left_state_index must be less than right_state_index")
    beta_mol = 1.0 / (R * _positive_float(committor_input.temperature_K, "temperature_K"))
    log_interval_resistances = np.zeros(grid_points.size - 1, dtype=float)
    for interval_index in range(grid_points.size - 1):
        left = interval_index
        right = interval_index + 1
        interval_length = grid_points[right] - grid_points[left]
        midpoint_free_energy = 0.5 * (
            free_energy[left] + free_energy[right]
        )
        midpoint_diffusivity = 0.5 * (diffusivity[left] + diffusivity[right])
        log_interval_resistances[interval_index] = (
            np.log(interval_length)
            + beta_mol * midpoint_free_energy
            - np.log(midpoint_diffusivity)
        )
    active_log_resistance = log_interval_resistances[
        committor_input.left_state_index : committor_input.right_state_index
    ]
    log_total_resistance = _logsumexp(active_log_resistance)
    shifted_active_resistance = np.exp(active_log_resistance - log_total_resistance)
    committor = np.zeros(grid_points.size, dtype=float)
    committor[committor_input.right_state_index :] = 1.0
    cumulative_resistance = 0.0
    for grid_index in range(
        committor_input.left_state_index + 1,
        committor_input.right_state_index,
    ):
        active_interval_index = grid_index - 1 - committor_input.left_state_index
        cumulative_resistance += shifted_active_resistance[active_interval_index]
        committor[grid_index] = cumulative_resistance
    capacity_integral = _capacity_from_log_resistance(log_total_resistance)
    return OneDimensionalCommittorResult(
        committor=committor,
        committor_gradient=np.gradient(committor, grid_points),
        capacity_integral=capacity_integral,
        log_capacity_integral=-log_total_resistance,
        quadrature_points=grid_points,
        quadrature_weights=_trapezoid_weights(grid_points),
    )


def solve_dense_dirichlet_problem(
    stiffness_matrix: Array,
    right_hand_side: Array,
    dirichlet_indices: Array,
    dirichlet_values: Array,
) -> Array:
    """Solve a finite-dimensional Dirichlet linear system."""

    stiffness = _square_matrix(stiffness_matrix, "stiffness_matrix")
    rhs = _same_size_vector(right_hand_side, stiffness.shape[0], "right_hand_side")
    boundary_indices = np.asarray(dirichlet_indices, dtype=int)
    boundary_values = _same_size_vector(
        dirichlet_values,
        boundary_indices.size,
        "dirichlet_values",
    )
    if boundary_indices.ndim != 1:
        raise ValueError("dirichlet_indices must be 1D")
    if np.any(boundary_indices < 0) or np.any(boundary_indices >= stiffness.shape[0]):
        raise ValueError("dirichlet_indices contains out-of-range entries")
    solution = np.zeros(stiffness.shape[0], dtype=float)
    solution[boundary_indices] = boundary_values
    interior_mask = np.ones(stiffness.shape[0], dtype=bool)
    interior_mask[boundary_indices] = False
    interior_indices = np.nonzero(interior_mask)[0]
    if interior_indices.size == 0:
        return solution
    interior_matrix = stiffness[np.ix_(interior_indices, interior_indices)]
    boundary_matrix = stiffness[np.ix_(interior_indices, boundary_indices)]
    interior_rhs = rhs[interior_indices] - boundary_matrix @ boundary_values
    solution[interior_indices] = np.linalg.solve(interior_matrix, interior_rhs)
    return solution


def assemble_weighted_gradient_stiffness(
    points: Array,
    diffusivity: Callable[[float], float],
    density_weight: Callable[[float], float],
) -> Array:
    """Assemble a 1D linear finite-element stiffness matrix."""

    grid_points = _strictly_increasing_vector(points, "points")
    matrix = np.zeros((grid_points.size, grid_points.size), dtype=float)
    for interval_index in range(grid_points.size - 1):
        left_point = grid_points[interval_index]
        right_point = grid_points[interval_index + 1]
        interval_length = right_point - left_point
        midpoint = 0.5 * (left_point + right_point)
        coefficient = (
            _positive_float(diffusivity(midpoint), "diffusivity(midpoint)")
            * _positive_float(density_weight(midpoint), "density_weight(midpoint)")
            / interval_length
        )
        local = coefficient * np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        matrix[
            interval_index : interval_index + 2,
            interval_index : interval_index + 2,
        ] += local
    return matrix


def _trapezoid_weights(points: Array) -> Array:
    weights = np.zeros(points.size, dtype=float)
    weights[0] = 0.5 * (points[1] - points[0])
    weights[-1] = 0.5 * (points[-1] - points[-2])
    if points.size > 2:
        weights[1:-1] = 0.5 * (points[2:] - points[:-2])
    return weights


def _logsumexp(values: Array) -> float:
    maximum_value = float(np.max(values))
    if not np.isfinite(maximum_value):
        raise ValueError("log resistance entries must be finite")
    return maximum_value + float(np.log(np.sum(np.exp(values - maximum_value))))


def _capacity_from_log_resistance(log_total_resistance: float) -> float:
    if not np.isfinite(log_total_resistance):
        raise ValueError("log_total_resistance must be finite")
    minimum_log_capacity = np.log(np.finfo(float).tiny)
    if -log_total_resistance < minimum_log_capacity:
        return float(np.finfo(float).tiny)
    return _positive_float(float(np.exp(-log_total_resistance)), "capacity_integral")


def _strictly_increasing_vector(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.ndim != 1 or result.size < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 1D array with at least two entries")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{label} must be strictly increasing")
    return result


def _same_size_vector(array: Array, size: int, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must have shape ({size},)")
    return result


def _positive_same_size_vector(array: Array, size: int, label: str) -> Array:
    result = _same_size_vector(array, size, label)
    if np.any(result <= 0.0):
        raise ValueError(f"{label} entries must be positive")
    return result


def _square_matrix(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{label} must be square")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_float(value: float, label: str) -> float:
    numeric_value = float(value)
    if numeric_value <= 0.0 or not np.isfinite(numeric_value):
        raise ValueError(f"{label} must be positive and finite")
    return numeric_value
