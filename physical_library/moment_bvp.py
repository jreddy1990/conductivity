"""Deterministic transition-moment boundary-value solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import R
from conductivity.physical_library.projected_analytical_conductivity import CARTESIAN

Array = np.ndarray


@dataclass(frozen=True)
class MomentBoundaryValueInput:
    grid_points: Array
    free_energy_J_mol: Array
    diffusivity_m2_s: Array
    committor: Array
    left_boundary_index: int
    right_boundary_index: int
    charge_polarization_by_grid: Array
    reactive_exit_weights: Array
    temperature_K: float


@dataclass(frozen=True)
class MomentBoundaryValueResult:
    endpoint_mean_by_grid: Array
    endpoint_second_moment_by_grid: Array
    first_displacement_moment_m: Array
    second_displacement_moment_m2: Array


def solve_endpoint_moment_bvp(
    moment_input: MomentBoundaryValueInput,
) -> MomentBoundaryValueResult:
    """Solve endpoint moments for a 1D conditioned transition."""

    grid_points = _strictly_increasing_vector(moment_input.grid_points, "grid_points")
    _same_size_vector(
        moment_input.free_energy_J_mol,
        grid_points.size,
        "free_energy_J_mol",
    )
    _positive_same_size_vector(
        moment_input.diffusivity_m2_s,
        grid_points.size,
        "diffusivity_m2_s",
    )
    committor = _finite_vector(moment_input.committor, "committor")
    if committor.size != grid_points.size:
        raise ValueError("committor length must match grid_points")
    charge_polarization = _polarization_matrix(
        moment_input.charge_polarization_by_grid,
        committor.size,
    )
    left_boundary_index = _valid_index(
        moment_input.left_boundary_index,
        committor.size,
        "left_boundary_index",
    )
    right_boundary_index = _valid_index(
        moment_input.right_boundary_index,
        committor.size,
        "right_boundary_index",
    )
    if left_boundary_index >= right_boundary_index:
        raise ValueError("left_boundary_index must be less than right_boundary_index")
    exit_weights = _finite_vector(
        moment_input.reactive_exit_weights,
        "reactive_exit_weights",
    )
    if exit_weights.size != committor.size:
        raise ValueError("reactive_exit_weights length must match committor")
    if np.any(exit_weights < 0.0):
        raise ValueError("reactive_exit_weights must be nonnegative")
    weight_sum = float(np.sum(exit_weights))
    if weight_sum <= 0.0:
        raise ValueError("reactive_exit_weights sum must be positive")
    normalized_exit_weights = exit_weights / weight_sum

    endpoint_mean = np.zeros((committor.size, CARTESIAN), dtype=float)
    endpoint_second = np.zeros((committor.size, CARTESIAN, CARTESIAN), dtype=float)
    endpoint_mean[right_boundary_index] = charge_polarization[right_boundary_index]
    endpoint_second[right_boundary_index] = np.outer(
        charge_polarization[right_boundary_index],
        charge_polarization[right_boundary_index],
    )
    endpoint_mean[:] = endpoint_mean[right_boundary_index]
    endpoint_second[:] = endpoint_second[right_boundary_index]

    displacement_by_grid = endpoint_mean - charge_polarization
    first_moment = np.einsum(
        "n,na->a",
        normalized_exit_weights,
        displacement_by_grid,
    )
    second_moment = np.einsum(
        "n,nab->ab",
        normalized_exit_weights,
        endpoint_second
        - np.einsum("na,nb->nab", charge_polarization, endpoint_mean)
        - np.einsum("nb,na->nab", charge_polarization, endpoint_mean)
        + np.einsum("na,nb->nab", charge_polarization, charge_polarization),
    )
    return MomentBoundaryValueResult(
        endpoint_mean_by_grid=endpoint_mean,
        endpoint_second_moment_by_grid=endpoint_second,
        first_displacement_moment_m=first_moment,
        second_displacement_moment_m2=0.5 * (second_moment + second_moment.T),
    )


def build_path_moment_arrays(
    state_count: int,
    transition_pairs: Array,
    moment_results: tuple[MomentBoundaryValueResult, ...],
) -> tuple[Array, Array]:
    """Convert edge moment solves into full d_ij and M_ij arrays."""

    pairs = np.asarray(transition_pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("transition_pairs must have shape (edge_count, 2)")
    if pairs.shape[0] != len(moment_results):
        raise ValueError("transition_pairs and moment_results length mismatch")
    d = np.zeros((state_count, state_count, CARTESIAN), dtype=float)
    second = np.zeros((state_count, state_count, CARTESIAN, CARTESIAN), dtype=float)
    for edge_index, pair in enumerate(pairs):
        from_state = _valid_index(int(pair[0]), state_count, "from_state")
        to_state = _valid_index(int(pair[1]), state_count, "to_state")
        edge_result = moment_results[edge_index]
        first_moment = _cartesian_vector(
            edge_result.first_displacement_moment_m,
            "first_displacement_moment_m",
        )
        second_moment = _cartesian_matrix(
            edge_result.second_displacement_moment_m2,
            "second_displacement_moment_m2",
        )
        d[from_state, to_state] = first_moment
        d[to_state, from_state] = -first_moment
        second[from_state, to_state] = second_moment
        second[to_state, from_state] = second_moment
    return d, second


def _finite_vector(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 1D array")
    return result


def _strictly_increasing_vector(array: Array, label: str) -> Array:
    result = _finite_vector(array, label)
    if result.size < 2:
        raise ValueError(f"{label} must contain at least two entries")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{label} must be strictly increasing")
    return result


def _same_size_vector(array: Array, size: int, label: str) -> Array:
    result = _finite_vector(array, label)
    if result.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},)")
    return result


def _positive_same_size_vector(array: Array, size: int, label: str) -> Array:
    result = _same_size_vector(array, size, label)
    if np.any(result <= 0.0):
        raise ValueError(f"{label} entries must be positive")
    return result


def _conditioned_generator_1d(
    grid_points: Array,
    free_energy_J_mol: Array,
    diffusivity_m2_s: Array,
    committor: Array,
    left_boundary_index: int,
    right_boundary_index: int,
    temperature_K: float,
) -> Array:
    temperature = float(temperature_K)
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature_K must be positive and finite")
    positive_committor = _positive_representable_committor(
        committor,
        left_boundary_index,
        right_boundary_index,
    )
    if np.any(positive_committor[left_boundary_index + 1 : right_boundary_index + 1] <= 0.0):
        raise ValueError("committor must be positive inside the conditioned domain")
    beta_mol = 1.0 / (R * temperature)
    generator = np.zeros((grid_points.size, grid_points.size), dtype=float)
    row_log_rates: dict[int, list[tuple[int, float]]] = {}
    for grid_index in range(left_boundary_index, right_boundary_index):
        next_index = grid_index + 1
        spacing = grid_points[next_index] - grid_points[grid_index]
        midpoint_diffusivity = 0.5 * (
            diffusivity_m2_s[grid_index] + diffusivity_m2_s[next_index]
        )
        log_forward_rate = (
            np.log(midpoint_diffusivity)
            - 2.0 * np.log(spacing)
            - 0.5
            * beta_mol
            * (free_energy_J_mol[next_index] - free_energy_J_mol[grid_index])
        )
        log_reverse_rate = (
            np.log(midpoint_diffusivity)
            - 2.0 * np.log(spacing)
            - 0.5
            * beta_mol
            * (free_energy_J_mol[grid_index] - free_energy_J_mol[next_index])
        )
        if grid_index > left_boundary_index:
            row_log_rates.setdefault(grid_index, []).append(
                (
                    next_index,
                    log_forward_rate
                    + np.log(positive_committor[next_index])
                    - np.log(positive_committor[grid_index]),
                )
            )
        if grid_index > left_boundary_index and next_index < right_boundary_index:
            row_log_rates.setdefault(next_index, []).append(
                (
                    grid_index,
                    log_reverse_rate
                    + np.log(positive_committor[grid_index])
                    - np.log(positive_committor[next_index]),
                )
            )
    for grid_index in range(left_boundary_index + 1, right_boundary_index):
        entries = row_log_rates.get(grid_index)
        if entries is None:
            raise ValueError("conditioned generator row has no outgoing rates")
        row_log_scale = max(log_rate for _, log_rate in entries)
        row_sum = 0.0
        for target_index, log_rate in entries:
            scaled_rate = float(np.exp(log_rate - row_log_scale))
            generator[grid_index, target_index] = scaled_rate
            row_sum += scaled_rate
        generator[grid_index, grid_index] = -row_sum
    return generator


def _positive_representable_committor(
    committor: Array,
    left_boundary_index: int,
    right_boundary_index: int,
) -> Array:
    result = np.array(committor, dtype=float, copy=True)
    if np.any(result[left_boundary_index + 1 : right_boundary_index + 1] < 0.0):
        raise ValueError("committor must be nonnegative inside the conditioned domain")
    result[left_boundary_index + 1 : right_boundary_index] = np.maximum(
        result[left_boundary_index + 1 : right_boundary_index],
        np.finfo(float).tiny,
    )
    if result[right_boundary_index] <= 0.0:
        raise ValueError("right boundary committor must be positive")
    return result


def _polarization_matrix(array: Array, rows: int) -> Array:
    result = np.asarray(array, dtype=float)
    if result.shape != (rows, CARTESIAN) or not np.all(np.isfinite(result)):
        raise ValueError(f"charge_polarization_by_grid must have shape ({rows}, 3)")
    return result


def _cartesian_vector(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.shape != (CARTESIAN,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must have shape (3,)")
    return result


def _cartesian_matrix(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.shape != (CARTESIAN, CARTESIAN) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must have shape (3, 3)")
    return 0.5 * (result + result.T)


def _valid_index(index: int, size: int, label: str) -> int:
    if index < 0 or index >= size:
        raise ValueError(f"{label} is out of range")
    return index
