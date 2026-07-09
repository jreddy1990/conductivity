"""Transition-surface builders for physical projected-generator inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductivity.physical_library.committor_bvp import (
    OneDimensionalCommittorInput,
    solve_one_dimensional_committor,
)
from conductivity.physical_library.physical_generator_builder import (
    PhysicalLocalFields,
    PhysicalTransitionQuadrature,
)
from conductivity.physical_library.physical_objects import SiteConfiguration
from conductivity.physical_library.projected_analytical_conductivity import CARTESIAN

Array = np.ndarray


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
