"""Transition-surface builders for physical projected-generator inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from constants import R
from conductivity.physical_library.physical_generator_builder import (
    PhysicalLocalFields,
    PhysicalTransitionQuadrature,
)
from conductivity.physical_library.physical_objects import SiteConfiguration
from conductivity.physical_library.projected_analytical_conductivity import CARTESIAN
from utils.strict_validation import (
    finite_vector,
    positive_float,
    positive_same_size_vector,
    same_size_vector,
    square_matrix,
    stable_logsumexp,
    strictly_increasing_vector,
    trapezoid_weights,
    valid_index,
)

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


@dataclass(frozen=True)
class OneDimensionalTransitionBuildInput:
    from_state_index: int
    to_state_index: int
    grid_configurations: tuple[SiteConfiguration, ...]
    local_fields: tuple[PhysicalLocalFields, ...]
    reaction_coordinate_values: Array
    reaction_coordinate_gradients: Array
    free_energy_J_mol: Array
    diffusivity_m2_s: Array
    temperature_K: float
    left_state_grid_index: int
    right_state_grid_index: int
    surface_state_indices: Array
    path_start_configurations: tuple[SiteConfiguration, ...]
    path_end_configurations: tuple[SiteConfiguration, ...]
    path_weights: Array
    first_displacement_moment_m: Array
    second_displacement_moment_m2: Array


@dataclass(frozen=True)
class TransitionSurfaceBuildResult:
    transition_quadrature: PhysicalTransitionQuadrature
    committor: Array
    capacity_integral: float
    log_capacity_integral: float


def solve_one_dimensional_committor(
    committor_input: OneDimensionalCommittorInput,
) -> OneDimensionalCommittorResult:
    """Solve the 1D reversible Smoluchowski committor on a grid."""

    grid_points = strictly_increasing_vector(
        committor_input.grid_points,
        "grid_points",
    )
    free_energy = same_size_vector(
        committor_input.free_energy_J_mol,
        grid_points.size,
        "free_energy_J_mol",
    )
    diffusivity = positive_same_size_vector(
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
    beta_mol = 1.0 / (R * positive_float(committor_input.temperature_K, "temperature_K"))
    grid_points_jax = jnp.asarray(grid_points)
    free_energy_jax = jnp.asarray(free_energy)
    diffusivity_jax = jnp.asarray(diffusivity)
    interval_lengths = grid_points_jax[1:] - grid_points_jax[:-1]
    midpoint_free_energy = 0.5 * (free_energy_jax[:-1] + free_energy_jax[1:])
    midpoint_diffusivity = 0.5 * (diffusivity_jax[:-1] + diffusivity_jax[1:])
    log_interval_resistances = np.asarray(
        jnp.log(interval_lengths)
        + beta_mol * midpoint_free_energy
        - jnp.log(midpoint_diffusivity),
        dtype=float,
    )
    active_log_resistance = log_interval_resistances[
        committor_input.left_state_index : committor_input.right_state_index
    ]
    log_total_resistance = stable_logsumexp(
        active_log_resistance,
        "log resistance",
    )
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
        quadrature_weights=trapezoid_weights(grid_points, "grid_points"),
    )


def solve_dense_dirichlet_problem(
    stiffness_matrix: Array,
    right_hand_side: Array,
    dirichlet_indices: Array,
    dirichlet_values: Array,
) -> Array:
    """Solve a finite-dimensional Dirichlet linear system."""

    stiffness = square_matrix(stiffness_matrix, "stiffness_matrix")
    rhs = same_size_vector(right_hand_side, stiffness.shape[0], "right_hand_side")
    boundary_indices = np.asarray(dirichlet_indices, dtype=int)
    boundary_values = same_size_vector(
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

    grid_points = strictly_increasing_vector(points, "points")
    matrix = np.zeros((grid_points.size, grid_points.size), dtype=float)
    for interval_index in range(grid_points.size - 1):
        left_point = grid_points[interval_index]
        right_point = grid_points[interval_index + 1]
        interval_length = right_point - left_point
        midpoint = 0.5 * (left_point + right_point)
        coefficient = (
            positive_float(diffusivity(midpoint), "diffusivity(midpoint)")
            * positive_float(density_weight(midpoint), "density_weight(midpoint)")
            / interval_length
        )
        local = coefficient * np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        matrix[
            interval_index : interval_index + 2,
            interval_index : interval_index + 2,
        ] += local
    return matrix


def solve_endpoint_moment_bvp(
    moment_input: MomentBoundaryValueInput,
) -> MomentBoundaryValueResult:
    """Solve endpoint moments for a 1D conditioned transition."""

    grid_points = strictly_increasing_vector(moment_input.grid_points, "grid_points")
    same_size_vector(
        moment_input.free_energy_J_mol,
        grid_points.size,
        "free_energy_J_mol",
    )
    positive_same_size_vector(
        moment_input.diffusivity_m2_s,
        grid_points.size,
        "diffusivity_m2_s",
    )
    committor = finite_vector(moment_input.committor, "committor")
    if committor.size != grid_points.size:
        raise ValueError("committor length must match grid_points")
    charge_polarization = _polarization_matrix(
        moment_input.charge_polarization_by_grid,
        committor.size,
    )
    left_boundary_index = valid_index(
        moment_input.left_boundary_index,
        committor.size,
        "left_boundary_index",
    )
    right_boundary_index = valid_index(
        moment_input.right_boundary_index,
        committor.size,
        "right_boundary_index",
    )
    if left_boundary_index >= right_boundary_index:
        raise ValueError("left_boundary_index must be less than right_boundary_index")
    exit_weights = finite_vector(
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
        from_state = valid_index(int(pair[0]), state_count, "from_state")
        to_state = valid_index(int(pair[1]), state_count, "to_state")
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


def build_one_dimensional_transition_surface(
    build_input: OneDimensionalTransitionBuildInput,
) -> TransitionSurfaceBuildResult:
    """Build transition quadrature from a deterministic 1D committor solve."""

    coordinate_values = np.asarray(build_input.reaction_coordinate_values, dtype=float)
    coordinate_gradients = np.asarray(build_input.reaction_coordinate_gradients, dtype=float)
    _validate_transition_grid(build_input, coordinate_values, coordinate_gradients)
    committor_result = solve_one_dimensional_committor(
        OneDimensionalCommittorInput(
            grid_points=coordinate_values,
            free_energy_J_mol=np.asarray(build_input.free_energy_J_mol, dtype=float),
            diffusivity_m2_s=np.asarray(build_input.diffusivity_m2_s, dtype=float),
            temperature_K=build_input.temperature_K,
            left_state_index=build_input.left_state_grid_index,
            right_state_index=build_input.right_state_grid_index,
        )
    )
    committor_gradient_along_coordinate = np.gradient(
        committor_result.committor,
        coordinate_values,
    )
    committor_gradients = (
        committor_gradient_along_coordinate[:, None] * coordinate_gradients
    )
    quadrature_weights = _trapezoid_grid_weights(coordinate_values)
    return TransitionSurfaceBuildResult(
        transition_quadrature=PhysicalTransitionQuadrature(
            from_state_index=build_input.from_state_index,
            to_state_index=build_input.to_state_index,
            configurations=build_input.grid_configurations,
            local_fields=build_input.local_fields,
            weights=quadrature_weights,
            committor_gradients=committor_gradients,
            surface_state_indices=np.asarray(build_input.surface_state_indices, dtype=int),
            path_start_configurations=build_input.path_start_configurations,
            path_end_configurations=build_input.path_end_configurations,
            path_weights=np.asarray(build_input.path_weights, dtype=float),
            first_displacement_moment_m=np.asarray(
                build_input.first_displacement_moment_m,
                dtype=float,
            ),
            second_displacement_moment_m2=np.asarray(
                build_input.second_displacement_moment_m2,
                dtype=float,
            ),
            log_capacity_integral=committor_result.log_capacity_integral,
            uses_residence_rate_constant=False,
            residence_rate_constant_s_inv=0.0,
        ),
        committor=committor_result.committor,
        capacity_integral=committor_result.capacity_integral,
        log_capacity_integral=committor_result.log_capacity_integral,
    )


def _validate_transition_grid(
    build_input: OneDimensionalTransitionBuildInput,
    coordinate_values: Array,
    coordinate_gradients: Array,
) -> None:
    grid_count = len(build_input.grid_configurations)
    if coordinate_values.shape != (grid_count,):
        raise ValueError("reaction_coordinate_values length must match grid_configurations")
    if len(build_input.local_fields) != grid_count:
        raise ValueError("local_fields length must match grid_configurations")
    if coordinate_gradients.ndim != 2 or coordinate_gradients.shape[0] != grid_count:
        raise ValueError("reaction_coordinate_gradients must have one row per grid point")
    if np.any(np.diff(coordinate_values) <= 0.0):
        raise ValueError("reaction_coordinate_values must be strictly increasing")
    if np.asarray(build_input.free_energy_J_mol, dtype=float).shape != (grid_count,):
        raise ValueError("free_energy_J_mol length must match grid_configurations")
    if np.asarray(build_input.diffusivity_m2_s, dtype=float).shape != (grid_count,):
        raise ValueError("diffusivity_m2_s length must match grid_configurations")
    if np.asarray(build_input.surface_state_indices, dtype=int).shape != (grid_count,):
        raise ValueError("surface_state_indices length must match grid_configurations")
    if len(build_input.path_start_configurations) != len(build_input.path_end_configurations):
        raise ValueError("path start and end configuration counts must match")
    if np.asarray(build_input.path_weights, dtype=float).shape != (
        len(build_input.path_start_configurations),
    ):
        raise ValueError("path_weights length must match path configurations")
    if np.asarray(build_input.first_displacement_moment_m, dtype=float).shape != (
        CARTESIAN,
    ):
        raise ValueError("first_displacement_moment_m must have shape (3,)")
    if np.asarray(build_input.second_displacement_moment_m2, dtype=float).shape != (
        CARTESIAN,
        CARTESIAN,
    ):
        raise ValueError("second_displacement_moment_m2 must have shape (3, 3)")


def _capacity_from_log_resistance(log_total_resistance: float) -> float:
    if not np.isfinite(log_total_resistance):
        raise ValueError("log_total_resistance must be finite")
    minimum_log_capacity = np.log(np.finfo(float).tiny)
    if -log_total_resistance < minimum_log_capacity:
        return float(np.finfo(float).tiny)
    return positive_float(float(np.exp(-log_total_resistance)), "capacity_integral")


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


def _trapezoid_grid_weights(coordinate_values: Array) -> Array:
    values = np.asarray(coordinate_values, dtype=float)
    if values.size < 2:
        raise ValueError("transition grid must contain at least two points")
    weights = np.zeros(values.size, dtype=float)
    spacing = np.diff(values)
    weights[0] = spacing[0] / 2.0
    weights[-1] = spacing[-1] / 2.0
    if values.size > 2:
        weights[1:-1] = (spacing[:-1] + spacing[1:]) / 2.0
    return weights
