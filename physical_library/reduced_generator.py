"""Reduced-generator assembly for the projected conductivity model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from conductivity.physical_library.projected_analytical_conductivity import (
    CARTESIAN,
    ProjectedGeneratorInput,
)

Array = np.ndarray


@dataclass(frozen=True)
class ReducedStateQuadrature:
    points: Array
    weights: Array
    stoichiometry: Array
    self_current_projector: Array


@dataclass(frozen=True)
class ReducedTransitionQuadrature:
    from_state_index: int
    to_state_index: int
    points: Array
    weights: Array
    committor_gradients: Array
    surface_state_indices: Array
    path_displacements_m: Array
    path_weights: Array
    first_displacement_moment_m: Array
    second_displacement_moment_m2: Array
    log_capacity_integral: float
    uses_residence_rate_constant: bool
    residence_rate_constant_s_inv: float


@dataclass(frozen=True)
class ReducedGeneratorSpecification:
    potential_energy_J_mol: Callable[[Array], float]
    mobility_tensor_m2_s: Callable[[Array], Array]
    charge_polarization_gradient: Callable[[Array], Array]
    memory_coordinate_gradient: Callable[[Array], Array]
    state_quadratures: tuple[ReducedStateQuadrature, ...]
    transition_quadratures: tuple[ReducedTransitionQuadrature, ...]
    total_component_concentrations_mol_m3: Array
    temperature_K: float
    volume_m3: float


def build_projected_generator_input(
    specification: ReducedGeneratorSpecification,
) -> ProjectedGeneratorInput:
    """Convert reduced generator quadrature into the readout input object."""

    if not specification.state_quadratures:
        raise ValueError("state_quadratures must not be empty")
    state_count = len(specification.state_quadratures)
    basin_points = []
    basin_weights = []
    basin_stoichiometry = []
    self_projectors = []
    for state_index, state_quadrature in enumerate(specification.state_quadratures):
        points = _as_2d(state_quadrature.points, f"state[{state_index}].points")
        weights = _as_1d(state_quadrature.weights, f"state[{state_index}].weights")
        if points.shape[0] != weights.size:
            raise ValueError(f"state[{state_index}] point/weight count mismatch")
        projector = _as_square(
            state_quadrature.self_current_projector,
            points.shape[1],
            f"state[{state_index}].self_current_projector",
        )
        basin_points.append(points)
        basin_weights.append(weights)
        basin_stoichiometry.append(
            _as_1d(state_quadrature.stoichiometry, f"state[{state_index}].stoichiometry")
        )
        self_projectors.append(projector)

    transition_pair_indices = []
    transition_points = []
    transition_weights = []
    transition_gradients = []
    transition_surface_states = []
    transition_path_displacements = []
    transition_path_weights = []
    transition_log_capacity_integrals = []
    transition_uses_residence_rate_constants = []
    transition_residence_rate_constants = []
    transition_first_moments = np.zeros((state_count, state_count, CARTESIAN), dtype=float)
    transition_second_moments = np.zeros(
        (state_count, state_count, CARTESIAN, CARTESIAN),
        dtype=float,
    )
    for transition_index, transition_quadrature in enumerate(
        specification.transition_quadratures
    ):
        _validate_state_index(
            transition_quadrature.from_state_index,
            state_count,
            f"transition[{transition_index}].from_state_index",
        )
        _validate_state_index(
            transition_quadrature.to_state_index,
            state_count,
            f"transition[{transition_index}].to_state_index",
        )
        transition_pair_indices.append(
            [
                transition_quadrature.from_state_index,
                transition_quadrature.to_state_index,
            ]
        )
        points = _as_2d(
            transition_quadrature.points,
            f"transition[{transition_index}].points",
        )
        weights = _as_1d(
            transition_quadrature.weights,
            f"transition[{transition_index}].weights",
        )
        gradients = _as_2d(
            transition_quadrature.committor_gradients,
            f"transition[{transition_index}].committor_gradients",
        )
        surface_states = np.asarray(
            transition_quadrature.surface_state_indices,
            dtype=int,
        )
        if surface_states.ndim != 1:
            raise ValueError(
                f"transition[{transition_index}].surface_state_indices must be 1D"
            )
        if not (
            points.shape[0]
            == weights.size
            == gradients.shape[0]
            == surface_states.size
        ):
            raise ValueError(f"transition[{transition_index}] quadrature count mismatch")
        displacements = _as_2d(
            transition_quadrature.path_displacements_m,
            f"transition[{transition_index}].path_displacements_m",
        )
        if displacements.shape[1] != CARTESIAN:
            raise ValueError(
                f"transition[{transition_index}].path_displacements_m must have 3 columns"
            )
        path_weights = _as_1d(
            transition_quadrature.path_weights,
            f"transition[{transition_index}].path_weights",
        )
        if displacements.shape[0] != path_weights.size:
            raise ValueError(f"transition[{transition_index}] path count mismatch")
        first_moment = _as_1d(
            transition_quadrature.first_displacement_moment_m,
            f"transition[{transition_index}].first_displacement_moment_m",
        )
        if first_moment.shape != (CARTESIAN,):
            raise ValueError(
                f"transition[{transition_index}].first_displacement_moment_m must have shape (3,)"
            )
        second_moment = _as_square(
            transition_quadrature.second_displacement_moment_m2,
            CARTESIAN,
            f"transition[{transition_index}].second_displacement_moment_m2",
        )
        from_state_index = transition_quadrature.from_state_index
        to_state_index = transition_quadrature.to_state_index
        transition_first_moments[from_state_index, to_state_index] = first_moment
        transition_first_moments[to_state_index, from_state_index] = -first_moment
        transition_second_moments[from_state_index, to_state_index] = second_moment
        transition_second_moments[to_state_index, from_state_index] = second_moment
        transition_points.append(points)
        transition_weights.append(weights)
        transition_gradients.append(gradients)
        transition_surface_states.append(surface_states)
        transition_path_displacements.append(displacements)
        transition_path_weights.append(path_weights)
        transition_log_capacity_integrals.append(float(transition_quadrature.log_capacity_integral))
        transition_uses_residence_rate_constants.append(
            bool(transition_quadrature.uses_residence_rate_constant)
        )
        transition_residence_rate_constants.append(
            float(transition_quadrature.residence_rate_constant_s_inv)
        )

    if transition_pair_indices:
        transition_pair_index_array = np.asarray(transition_pair_indices, dtype=int)
    elif state_count == 1:
        transition_pair_index_array = np.empty((0, 2), dtype=int)
    else:
        raise ValueError(
            "multiple generated states require at least one transition quadrature"
        )

    return ProjectedGeneratorInput(
        potential_energy_J_mol=specification.potential_energy_J_mol,
        mobility_tensor_m2_s=specification.mobility_tensor_m2_s,
        charge_polarization_gradient=specification.charge_polarization_gradient,
        memory_coordinate_gradient=specification.memory_coordinate_gradient,
        basin_quadrature_points=tuple(basin_points),
        basin_quadrature_weights=tuple(basin_weights),
        transition_pair_indices=transition_pair_index_array,
        transition_quadrature_points=tuple(transition_points),
        transition_quadrature_weights=tuple(transition_weights),
        transition_committor_gradients=tuple(transition_gradients),
        transition_surface_state_indices=tuple(transition_surface_states),
        transition_path_displacements_m=tuple(transition_path_displacements),
        transition_path_weights=tuple(transition_path_weights),
        transition_log_capacity_integrals=np.asarray(
            transition_log_capacity_integrals,
            dtype=float,
        ),
        transition_uses_residence_rate_constants=np.asarray(
            transition_uses_residence_rate_constants,
            dtype=bool,
        ),
        transition_residence_rate_constants_s_inv=np.asarray(
            transition_residence_rate_constants,
            dtype=float,
        ),
        transition_first_moments_d_ij_m=transition_first_moments,
        transition_second_moments_M_ij_m2=transition_second_moments,
        total_component_concentrations_mol_m3=_as_1d(
            specification.total_component_concentrations_mol_m3,
            "total_component_concentrations_mol_m3",
        ),
        basin_stoichiometry=np.asarray(basin_stoichiometry, dtype=float),
        temperature_K=float(specification.temperature_K),
        volume_m3=float(specification.volume_m3),
        self_current_coordinate_projectors=tuple(self_projectors),
    )


def _as_1d(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 1D array")
    return result


def _as_2d(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 2D array")
    return result


def _as_square(array: Array, size: int, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must have shape ({size}, {size})")
    return result


def _validate_state_index(state_index: int, state_count: int, label: str) -> None:
    if state_index < 0 or state_index >= state_count:
        raise ValueError(f"{label} is out of range")
