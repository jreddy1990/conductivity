"""Memory-coordinate projection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from conductivity.physical_library.projected_analytical_conductivity import (
    compute_mori_memory_matrices,
    compute_state_memory_coordinate_means,
)

Array = np.ndarray


@dataclass(frozen=True)
class StateOrthogonalMemoryResult:
    state_means: Array
    mori_memory_matrix_A: Array
    mori_current_coupling_matrix_h: Array


def project_memory_coordinates(
    mobility_tensor_m2_s: Callable[[Array], Array],
    charge_polarization_gradient: Callable[[Array], Array],
    memory_coordinate_gradient: Callable[[Array], Array],
    memory_coordinates: Callable[[Array], Array],
    basin_quadrature_points: tuple[Array, ...],
    basin_density_weights_mol_m3: tuple[Array, ...],
    basin_concentrations_mol_m3: Array,
) -> StateOrthogonalMemoryResult:
    """Compute state means and Mori matrices for current-coupled memory modes."""

    state_means = compute_state_memory_coordinate_means(
        memory_coordinates,
        basin_quadrature_points,
        basin_density_weights_mol_m3,
        basin_concentrations_mol_m3,
    )
    mori_memory_matrix, mori_current_coupling = compute_mori_memory_matrices(
        mobility_tensor_m2_s,
        charge_polarization_gradient,
        memory_coordinate_gradient,
        basin_quadrature_points,
        basin_density_weights_mol_m3,
    )
    return StateOrthogonalMemoryResult(
        state_means=state_means,
        mori_memory_matrix_A=mori_memory_matrix,
        mori_current_coupling_matrix_h=mori_current_coupling,
    )


def make_state_orthogonal_memory_function(
    memory_coordinates: Callable[[Array], Array],
    state_classifier: Callable[[Array], int],
    state_means: Array,
) -> Callable[[Array], Array]:
    """Return psi(q)-E[psi|A_i] using an explicit state classifier."""

    means = np.asarray(state_means, dtype=float)
    if means.ndim != 2 or not np.all(np.isfinite(means)):
        raise ValueError("state_means must be a finite 2D array")

    def state_orthogonal_memory(point: Array) -> Array:
        state_index = int(state_classifier(point))
        if state_index < 0 or state_index >= means.shape[0]:
            raise ValueError("state_classifier returned an out-of-range state index")
        memory_value = np.asarray(memory_coordinates(point), dtype=float)
        if memory_value.shape != (means.shape[1],):
            raise ValueError("memory coordinate dimension does not match state_means")
        return memory_value - means[state_index]

    return state_orthogonal_memory
