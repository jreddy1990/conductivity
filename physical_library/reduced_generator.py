"""Reduced-generator assembly for the projected conductivity model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from conductivity.physical_library.projected_analytical_conductivity import (
    CARTESIAN,
    ProjectedGeneratorInput,
    StateTransportOwnershipBasis,
    TransportOwnership,
)

Array = np.ndarray


@dataclass(frozen=True)
class TransportOwnershipRecord:
    state: int
    label: str
    owner: TransportOwnership
    gradient: Array
    physical_basis: str

    def __post_init__(self) -> None:
        if isinstance(self.state, bool) or not isinstance(self.state, int):
            raise TypeError("state must be an integer")
        if self.state < 0:
            raise ValueError("state must be nonnegative")
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if not isinstance(self.owner, TransportOwnership):
            raise TypeError("owner must be a TransportOwnership")
        gradient = _as_1d(self.gradient, "gradient").copy()
        if gradient.size == 0:
            raise ValueError("gradient must not be empty")
        gradient.setflags(write=False)
        object.__setattr__(self, "gradient", gradient)
        if not isinstance(self.physical_basis, str):
            raise TypeError("physical_basis must be a string")
        if not self.physical_basis.strip():
            raise ValueError("physical_basis must not be empty")


@dataclass(frozen=True)
class ReducedStateQuadrature:
    points: Array
    weights: Array
    stoichiometry: Array
    self_current_projector: Array
    transport_ownership_bases: tuple[StateTransportOwnershipBasis, ...]
    relative_displacement_fluctuations_m: Array
    relative_displacement_mobility_m2_s: Array
    relative_center_charge_numbers: Array


@dataclass(frozen=True)
class ReducedTransitionQuadrature:
    from_state_index: int
    to_state_index: int
    transition_family: str
    transport_ownership: TransportOwnership
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
    state_continuous_memory_family_keys: tuple[tuple[str, ...], ...]
    state_quadratures: tuple[ReducedStateQuadrature, ...]
    transition_quadratures: tuple[ReducedTransitionQuadrature, ...]
    total_component_concentrations_mol_m3: Array
    state_memory_value_matrix: Array
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
    state_transport_ownership_bases = []
    relative_displacement_fluctuations = []
    relative_displacement_mobilities = []
    relative_center_charge_numbers = []
    for state_index, state_quadrature in enumerate(specification.state_quadratures):
        points = _as_2d(state_quadrature.points, f"state[{state_index}].points")
        weights = _as_1d(state_quadrature.weights, f"state[{state_index}].weights")
        if points.shape[0] != weights.size:
            raise ValueError(f"state[{state_index}] point/weight count mismatch")
        if len(state_quadrature.transport_ownership_bases) != points.shape[0]:
            raise ValueError(
                f"state[{state_index}].transport_ownership_bases count must match points"
            )
        for quadrature_index, ownership_basis in enumerate(
            state_quadrature.transport_ownership_bases
        ):
            _validate_transport_ownership_basis_width(
                ownership_basis,
                points.shape[1],
                state_index,
                quadrature_index,
            )
        projector = _as_square(
            state_quadrature.self_current_projector,
            points.shape[1],
            f"state[{state_index}].self_current_projector",
        )
        basin_points.append(points)
        basin_weights.append(weights)
        basin_stoichiometry.append(
            _as_1d(
                state_quadrature.stoichiometry, f"state[{state_index}].stoichiometry"
            )
        )
        self_projectors.append(projector)
        state_transport_ownership_bases.append(
            tuple(state_quadrature.transport_ownership_bases)
        )
        relative_displacement_fluctuations.append(
            np.asarray(
                state_quadrature.relative_displacement_fluctuations_m, dtype=float
            )
        )
        relative_displacement_mobilities.append(
            np.asarray(
                state_quadrature.relative_displacement_mobility_m2_s, dtype=float
            )
        )
        relative_center_charge_numbers.append(
            np.asarray(state_quadrature.relative_center_charge_numbers, dtype=float)
        )

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
    transition_transport_ownership = []
    transition_first_moments = np.zeros(
        (state_count, state_count, CARTESIAN), dtype=float
    )
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
        if not transition_quadrature.transition_family.strip():
            raise ValueError(
                f"transition[{transition_index}].transition_family must not be empty"
            )
        if not isinstance(
            transition_quadrature.transport_ownership,
            TransportOwnership,
        ):
            raise TypeError(
                f"transition[{transition_index}].transport_ownership must be a TransportOwnership"
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
            points.shape[0] == weights.size == gradients.shape[0] == surface_states.size
        ):
            raise ValueError(
                f"transition[{transition_index}] quadrature count mismatch"
            )
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
        transition_log_capacity_integrals.append(
            float(transition_quadrature.log_capacity_integral)
        )
        transition_uses_residence_rate_constants.append(
            bool(transition_quadrature.uses_residence_rate_constant)
        )
        transition_residence_rate_constants.append(
            float(transition_quadrature.residence_rate_constant_s_inv)
        )
        transition_transport_ownership.append(
            transition_quadrature.transport_ownership
        )

    if transition_pair_indices:
        transition_pair_index_array = np.asarray(transition_pair_indices, dtype=int)
    elif state_count == 1:
        transition_pair_index_array = np.empty((0, 2), dtype=int)
    else:
        raise ValueError(
            "multiple generated states require at least one transition quadrature"
        )

    (
        expanded_memory_gradient,
        expanded_memory_mask,
        expanded_ownership_bases,
    ) = _expand_continuous_memory_by_state_family(
        specification,
        tuple(state_transport_ownership_bases),
        tuple(basin_points),
    )

    return ProjectedGeneratorInput(
        potential_energy_J_mol=specification.potential_energy_J_mol,
        mobility_tensor_m2_s=specification.mobility_tensor_m2_s,
        charge_polarization_gradient=specification.charge_polarization_gradient,
        memory_coordinate_gradient=expanded_memory_gradient,
        basin_quadrature_points=tuple(basin_points),
        basin_quadrature_weights=tuple(basin_weights),
        basin_energy_references_J_mol=np.asarray(
            [
                min(
                    float(specification.potential_energy_J_mol(point))
                    for point in points
                )
                for points in basin_points
            ],
            dtype=float,
        ),
        state_memory_active_mask=expanded_memory_mask,
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
        state_transport_ownership_bases=expanded_ownership_bases,
        transition_transport_ownership=tuple(transition_transport_ownership),
        state_relative_displacement_fluctuations_m=tuple(
            relative_displacement_fluctuations
        ),
        state_relative_displacement_mobilities_m2_s=tuple(
            relative_displacement_mobilities
        ),
        state_relative_center_charge_numbers=tuple(relative_center_charge_numbers),
        state_memory_value_matrix=np.asarray(
            specification.state_memory_value_matrix, dtype=float
        ),
    )


def _expand_continuous_memory_by_state_family(
    specification: ReducedGeneratorSpecification,
    state_transport_ownership_bases: tuple[
        tuple[StateTransportOwnershipBasis, ...], ...
    ],
    basin_points: tuple[Array, ...],
) -> tuple[
    Callable[[Array], Array],
    Array,
    tuple[tuple[StateTransportOwnershipBasis, ...], ...],
]:
    family_keys = tuple(specification.state_continuous_memory_family_keys)
    if len(family_keys) != len(state_transport_ownership_bases):
        raise ValueError("continuous memory family key count must match states")
    base_mask = _state_memory_active_mask(
        specification,
        state_transport_ownership_bases,
        basin_points,
    )
    base_memory_count = base_mask.shape[1]
    expansion_pairs = tuple(
        (memory_index, family_key)
        for memory_index in range(base_memory_count)
        for family_key in dict.fromkeys(
            family_keys[state_index]
            for state_index in range(len(family_keys))
            if base_mask[state_index, memory_index]
        )
    )
    if not expansion_pairs:
        return specification.memory_coordinate_gradient, base_mask, state_transport_ownership_bases
    expanded_index_by_pair = {
        expansion_pair: expanded_index
        for expanded_index, expansion_pair in enumerate(expansion_pairs)
    }

    def expanded_memory_gradient(point: Array) -> Array:
        base_gradient = _as_2d(
            specification.memory_coordinate_gradient(point),
            "memory_coordinate_gradient",
        )
        if base_gradient.shape[0] != base_memory_count:
            raise ValueError("continuous memory gradient row count changed")
        return base_gradient[
            np.asarray(
                [memory_index for memory_index, _family_key in expansion_pairs],
                dtype=int,
            )
        ]

    expanded_mask = np.zeros(
        (len(family_keys), len(expansion_pairs)),
        dtype=bool,
    )
    expanded_bases = []
    for state_index, state_bases in enumerate(state_transport_ownership_bases):
        family_key = family_keys[state_index]
        for memory_index in np.flatnonzero(base_mask[state_index]):
            expanded_mask[
                state_index,
                expanded_index_by_pair[(int(memory_index), family_key)],
            ] = True
        expanded_state_bases = []
        for ownership_basis in state_bases:
            expanded_indices = np.asarray(
                [
                    expanded_index_by_pair[(int(memory_index), family_key)]
                    for memory_index in ownership_basis.bounded_memory_mode_indices
                ],
                dtype=int,
            )
            expanded_state_bases.append(
                StateTransportOwnershipBasis(
                    transition_displacement_gradients=(
                        ownership_basis.transition_displacement_gradients
                    ),
                    transition_edge_indices=ownership_basis.transition_edge_indices,
                    bounded_memory_gradients=ownership_basis.bounded_memory_gradients,
                    bounded_memory_mode_indices=expanded_indices,
                    diagnostic_gradients=ownership_basis.diagnostic_gradients,
                    diagnostic_source_ids=ownership_basis.diagnostic_source_ids,
                )
            )
        expanded_bases.append(tuple(expanded_state_bases))
    return expanded_memory_gradient, expanded_mask, tuple(expanded_bases)


def _state_memory_active_mask(
    specification: ReducedGeneratorSpecification,
    state_transport_ownership_bases: tuple[
        tuple[StateTransportOwnershipBasis, ...], ...
    ],
    basin_points: tuple[Array, ...],
) -> Array:
    memory_count = int(
        np.asarray(
            specification.memory_coordinate_gradient(basin_points[0][0]),
            dtype=float,
        ).shape[0]
    )
    active_mask = np.zeros(
        (len(state_transport_ownership_bases), memory_count),
        dtype=bool,
    )
    for state_index, state_bases in enumerate(state_transport_ownership_bases):
        for ownership_basis in state_bases:
            active_mask[
                state_index,
                np.asarray(ownership_basis.bounded_memory_mode_indices, dtype=int),
            ] = True
    return active_mask


def _as_1d(array: Array, label: str) -> Array:
    result = np.asarray(array, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 1D array")
    return result


def _validate_transport_ownership_basis_width(
    ownership_basis: StateTransportOwnershipBasis,
    coordinate_dimension: int,
    state_index: int,
    quadrature_index: int,
) -> None:
    gradient_groups = (
        ownership_basis.transition_displacement_gradients,
        ownership_basis.bounded_memory_gradients,
        ownership_basis.diagnostic_gradients,
    )
    if any(
        np.asarray(gradients, dtype=float).shape[1] != coordinate_dimension
        for gradients in gradient_groups
    ):
        raise ValueError(
            f"state[{state_index}].transport_ownership_bases[{quadrature_index}] "
            "gradient coordinate dimension mismatch"
        )


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
