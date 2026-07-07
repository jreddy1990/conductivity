"""Standalone projected analytical conductivity estimator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from constants import (
    EPSILON_R_WATER_25C,
    F,
    MOL_M3_PER_MOL_L,
    MS_CM_TO_S_M,
    R,
    S_M_TO_MS_CM,
    T_REF_K,
)
from electrolyte_model import ElectrolyteRecipeModel
from species_fns import get_species_property
from utils.config_cache import load_physics_config
from utils.strict_validation import (
    require_key,
    strict_finite_array,
    strict_nonnegative_float,
    strict_positive_float,
    strict_mapping,
)

VECTOR_COMPONENT_COUNT = 3


def compute_projected_analytical_conductivity(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    mobility_tensor_m2_s: Callable[[np.ndarray], np.ndarray],
    charge_polarization_gradient: Callable[[np.ndarray], np.ndarray],
    memory_coordinate_gradient: Callable[[np.ndarray], np.ndarray],
    basin_quadrature_points: Sequence[np.ndarray],
    basin_quadrature_weights: Sequence[np.ndarray],
    transition_pair_indices: np.ndarray,
    transition_quadrature_points: Sequence[np.ndarray],
    transition_quadrature_weights: Sequence[np.ndarray],
    transition_committor_gradients: Sequence[np.ndarray],
    transition_path_displacements_m: Sequence[np.ndarray],
    transition_path_weights: Sequence[np.ndarray],
    total_concentration_mol_m3: float,
    temperature_K: float,
    volume_m3: float,
) -> dict[str, np.ndarray | float]:
    partition_values = compute_restricted_partition_values(
        potential_energy_J_mol,
        basin_quadrature_points,
        basin_quadrature_weights,
        temperature_K,
    )
    total_partition_value = float(np.sum(partition_values))
    state_concentrations_mol_m3 = compute_equilibrium_populations(
        partition_values,
        total_concentration_mol_m3,
    )
    capacity_fluxes = compute_symmetric_capacity_fluxes(
        potential_energy_J_mol,
        mobility_tensor_m2_s,
        transition_pair_indices,
        transition_quadrature_points,
        transition_quadrature_weights,
        transition_committor_gradients,
        total_concentration_mol_m3,
        temperature_K,
        len(partition_values),
        total_partition_value,
    )
    first_moments, second_moments = compute_transition_path_displacement_moments(
        transition_pair_indices,
        transition_path_displacements_m,
        transition_path_weights,
        len(partition_values),
    )
    self_current_tensors = compute_self_current_tensors(
        potential_energy_J_mol,
        mobility_tensor_m2_s,
        charge_polarization_gradient,
        basin_quadrature_points,
        basin_quadrature_weights,
        state_concentrations_mol_m3,
        total_concentration_mol_m3,
        temperature_K,
        total_partition_value,
    )
    mori_memory_matrix_A, mori_current_coupling_matrix_h = (
        compute_mori_memory_matrices(
            potential_energy_J_mol,
            mobility_tensor_m2_s,
            charge_polarization_gradient,
            memory_coordinate_gradient,
            basin_quadrature_points,
            basin_quadrature_weights,
            total_concentration_mol_m3,
            temperature_K,
            total_partition_value,
        )
    )
    return compute_projected_analytical_conductivity_from_primitives(
        state_concentrations_mol_m3,
        capacity_fluxes,
        first_moments,
        second_moments,
        self_current_tensors,
        mori_memory_matrix_A,
        mori_current_coupling_matrix_h,
        temperature_K,
        volume_m3,
    )


def compute_projected_analytical_conductivity_from_functions(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    mobility_tensor_m2_s: Callable[[np.ndarray], np.ndarray],
    charge_polarization: Callable[[np.ndarray], np.ndarray],
    memory_coordinates: Callable[[np.ndarray], np.ndarray],
    basin_quadrature_points: Sequence[np.ndarray],
    basin_quadrature_weights: Sequence[np.ndarray],
    transition_pair_indices: np.ndarray,
    transition_quadrature_points: Sequence[np.ndarray],
    transition_quadrature_weights: Sequence[np.ndarray],
    transition_committor_gradients: Sequence[np.ndarray],
    transition_path_start_points: Sequence[np.ndarray],
    transition_path_end_points: Sequence[np.ndarray],
    transition_path_weights: Sequence[np.ndarray],
    total_concentration_mol_m3: float,
    temperature_K: float,
    volume_m3: float,
) -> dict[str, np.ndarray | float]:
    coordinate_dimension = _infer_coordinate_dimension(basin_quadrature_points)
    finite_difference_step = np.full(coordinate_dimension, 1.0e-6, dtype=float)

    def charge_polarization_gradient(point: np.ndarray) -> np.ndarray:
        return compute_charge_polarization_gradient_by_finite_difference(
            charge_polarization,
            point,
            finite_difference_step,
        )

    def memory_coordinate_gradient(point: np.ndarray) -> np.ndarray:
        return compute_memory_coordinate_gradient_by_finite_difference(
            memory_coordinates,
            point,
            finite_difference_step,
        )

    displacements = tuple(
        _endpoint_displacements(
            transition_path_start_points[pair_index],
            transition_path_end_points[pair_index],
            charge_polarization,
            f"transition_path[{pair_index}]",
        )
        for pair_index in range(len(transition_pair_indices))
    )
    return compute_projected_analytical_conductivity(
        potential_energy_J_mol,
        mobility_tensor_m2_s,
        charge_polarization_gradient,
        memory_coordinate_gradient,
        basin_quadrature_points,
        basin_quadrature_weights,
        transition_pair_indices,
        transition_quadrature_points,
        transition_quadrature_weights,
        transition_committor_gradients,
        displacements,
        transition_path_weights,
        total_concentration_mol_m3,
        temperature_K,
        volume_m3,
    )


def compute_projected_analytical_conductivity_from_primitives(
    state_concentrations_mol_m3: np.ndarray,
    symmetric_capacity_fluxes_K_ij_mol_m3_s: np.ndarray,
    transition_first_moments_d_ij_m: np.ndarray,
    transition_second_moments_M_ij_m2: np.ndarray,
    self_current_tensors_D_self_i_m2_s: np.ndarray,
    mori_memory_matrix_A: np.ndarray,
    mori_current_coupling_matrix_h: np.ndarray,
    temperature_K: float,
    volume_m3: float = 1.0,
) -> dict[str, np.ndarray | float]:
    concentrations = _as_vector(state_concentrations_mol_m3, "state_concentrations")
    state_count = len(concentrations)
    capacity_fluxes = _as_square(
        symmetric_capacity_fluxes_K_ij_mol_m3_s,
        "capacity_fluxes",
        state_count,
    )
    first_moments = _as_first_moments(
        transition_first_moments_d_ij_m,
        "transition_first_moments",
        state_count,
    )
    second_moments = _as_second_moments(
        transition_second_moments_M_ij_m2,
        "transition_second_moments",
        state_count,
    )
    self_current_tensors = _as_state_tensors(
        self_current_tensors_D_self_i_m2_s,
        "self_current_tensors",
        state_count,
    )
    temperature = strict_positive_float(temperature_K, "temperature_K")
    volume = strict_positive_float(volume_m3, "volume_m3")
    mori_matrix = _as_mori_matrix(mori_memory_matrix_A)
    mori_coupling = _as_mori_coupling(
        mori_current_coupling_matrix_h,
        mori_matrix.shape[0],
    )
    reversible_generator = compute_reversible_generator(capacity_fluxes, concentrations)
    direct_diffusivity_tensor = _direct_diffusivity(
        concentrations,
        capacity_fluxes,
        second_moments,
        self_current_tensors,
    )
    finite_state_correction_tensor = compute_finite_state_memory_correction(
        concentrations,
        reversible_generator,
        first_moments,
    )
    continuous_mori_correction_tensor = compute_continuous_mori_correction(
        mori_matrix,
        mori_coupling,
    )
    projected_diffusivity_tensor = (
        direct_diffusivity_tensor
        - finite_state_correction_tensor
        - continuous_mori_correction_tensor
    )
    sigma_S_m = conductivity_from_projected_diffusivity(
        projected_diffusivity_tensor,
        temperature,
        volume,
    )
    return {
        "state_concentrations_mol_m3": concentrations,
        "symmetric_capacity_fluxes_K_ij_mol_m3_s": capacity_fluxes,
        "reversible_generator_Q_ij_s": reversible_generator,
        "transition_first_moments_d_ij_m": first_moments,
        "transition_second_moments_M_ij_m2": second_moments,
        "self_current_tensors_D_self_i_m2_s": self_current_tensors,
        "mori_memory_matrix_A": mori_matrix,
        "mori_current_coupling_matrix_h": mori_coupling,
        "direct_diffusivity_tensor": direct_diffusivity_tensor,
        "finite_state_correction_tensor": finite_state_correction_tensor,
        "continuous_mori_correction_tensor": continuous_mori_correction_tensor,
        "projected_diffusivity_tensor": projected_diffusivity_tensor,
        "sigma_S_m": sigma_S_m,
        "sigma_mS_cm": sigma_S_m * S_M_TO_MS_CM,
    }


def build_projected_generator_inputs_from_recipe(
    recipe: Mapping[str, Mapping[str, float]],
    temperature_K: float,
    volume_m3: float,
    physics_config,
):
    temperature = strict_positive_float(temperature_K, "temperature_K")
    volume = strict_positive_float(volume_m3, "volume_m3")
    config = strict_mapping(physics_config, "physics_config")
    recipe_model = ElectrolyteRecipeModel.model_validate(dict(recipe))
    recipe_mapping = strict_mapping(recipe_model.model_dump(), "recipe")
    solvent_mapping = strict_mapping(
        require_key(recipe_mapping, "solvents", "recipe"),
        "recipe.solvents",
    )
    salt_mapping = strict_mapping(
        require_key(recipe_mapping, "salts", "recipe"),
        "recipe.salts",
    )
    additive_mapping = strict_mapping(
        require_key(recipe_mapping, "additives", "recipe"),
        "recipe.additives",
    )
    mixture_properties = _projected_recipe_mixture_properties(
        solvent_mapping,
        additive_mapping,
    )
    ionic_source_molarities_M = _projected_recipe_ionic_sources(
        salt_mapping,
        additive_mapping,
        mixture_properties["density_g_ml"],
    )
    total_source_molarity_M = float(sum(ionic_source_molarities_M.values()))
    if total_source_molarity_M <= 0.0:
        raise ValueError("recipe must contain at least one positive ionic source")
    recipe_parameters = _projected_recipe_parameters(config)
    epsilon_effective = _projected_recipe_effective_dielectric(
        ionic_source_molarities_M,
        mixture_properties["epsilon_r"],
    )
    dielectric_support = epsilon_effective / (
        epsilon_effective + recipe_parameters["dielectric_support_scale"]
    )
    viscosity_factor = (
        recipe_parameters["viscosity_reference_cP"]
        / mixture_properties["viscosity_cP"]
    ) ** recipe_parameters["viscosity_exponent"]
    crowding_factor = 1.0 / (
        1.0
        + total_source_molarity_M / recipe_parameters["crowding_molarity_scale"]
        + (
            total_source_molarity_M
            / recipe_parameters["crowding_curvature_molarity"]
        )
        ** 2
    )
    return _projected_recipe_primitives_from_transport_scalars(
        ionic_source_molarities_M,
        epsilon_effective,
        dielectric_support,
        viscosity_factor,
        crowding_factor,
        temperature,
        volume,
        recipe_parameters,
        mixture_properties,
    )


def compute_projected_analytical_conductivity_from_recipe(
    recipe: Mapping[str, Mapping[str, float]],
    temperature_K: float = T_REF_K,
    volume_m3: float = 1.0,
):
    physics_config = load_physics_config()
    projected_inputs = build_projected_generator_inputs_from_recipe(
        recipe,
        temperature_K,
        volume_m3,
        physics_config,
    )
    projected_result = compute_projected_analytical_conductivity_from_primitives(
        strict_finite_array(
            projected_inputs["state_concentrations_mol_m3"],
            "state_concentrations_mol_m3",
        ),
        strict_finite_array(
            projected_inputs["symmetric_capacity_fluxes_K_ij_mol_m3_s"],
            "symmetric_capacity_fluxes_K_ij_mol_m3_s",
        ),
        strict_finite_array(
            projected_inputs["transition_first_moments_d_ij_m"],
            "transition_first_moments_d_ij_m",
        ),
        strict_finite_array(
            projected_inputs["transition_second_moments_M_ij_m2"],
            "transition_second_moments_M_ij_m2",
        ),
        strict_finite_array(
            projected_inputs["self_current_tensors_D_self_i_m2_s"],
            "self_current_tensors_D_self_i_m2_s",
        ),
        strict_finite_array(
            projected_inputs["mori_memory_matrix_A"],
            "mori_memory_matrix_A",
        ),
        strict_finite_array(
            projected_inputs["mori_current_coupling_matrix_h"],
            "mori_current_coupling_matrix_h",
        ),
        float(projected_inputs["temperature_K"]),
        float(projected_inputs["volume_m3"]),
    )
    return {**projected_inputs, **projected_result}


def compute_restricted_partition_values(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    basin_quadrature_points: Sequence[np.ndarray],
    basin_quadrature_weights: Sequence[np.ndarray],
    temperature_K: float,
) -> np.ndarray:
    temperature = strict_positive_float(temperature_K, "temperature_K")
    _require_equal_lengths(
        basin_quadrature_points,
        basin_quadrature_weights,
        "basin quadrature",
    )
    partition_values = []
    for basin_index, points in enumerate(basin_quadrature_points):
        basin_points = _as_points(points, f"basin_quadrature_points[{basin_index}]")
        weights = _as_weights(
            basin_quadrature_weights[basin_index],
            f"basin_quadrature_weights[{basin_index}]",
            basin_points.shape[0],
        )
        boltzmann_weights = np.asarray(
            [
                np.exp(-float(potential_energy_J_mol(point)) / (R * temperature))
                for point in basin_points
            ],
            dtype=float,
        )
        partition_values.append(float(weights @ boltzmann_weights))
    return _as_vector(np.asarray(partition_values), "restricted_partition_values")


def compute_equilibrium_populations(
    restricted_partition_values: np.ndarray,
    total_concentration_mol_m3: float,
) -> np.ndarray:
    partition_values = _as_vector(
        restricted_partition_values,
        "restricted_partition_values",
    )
    total_concentration = strict_nonnegative_float(
        total_concentration_mol_m3,
        "total_concentration_mol_m3",
    )
    partition_total = float(np.sum(partition_values))
    if partition_total <= 0.0:
        raise ValueError("restricted_partition_values must sum to a positive value")
    return total_concentration * partition_values / partition_total


def compute_symmetric_capacity_fluxes(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    mobility_tensor_m2_s: Callable[[np.ndarray], np.ndarray],
    transition_pair_indices: np.ndarray,
    transition_quadrature_points: Sequence[np.ndarray],
    transition_quadrature_weights: Sequence[np.ndarray],
    transition_committor_gradients: Sequence[np.ndarray],
    total_concentration_mol_m3: float,
    temperature_K: float,
    state_count: int,
    total_partition_value: float,
) -> np.ndarray:
    pair_indices = _as_pair_indices(
        transition_pair_indices,
        "transition_pair_indices",
        state_count,
    )
    _require_equal_lengths(
        transition_quadrature_points,
        transition_quadrature_weights,
        "transition quadrature",
    )
    _require_equal_lengths(
        transition_quadrature_points,
        transition_committor_gradients,
        "transition committor",
    )
    if len(pair_indices) != len(transition_quadrature_points):
        raise ValueError("transition data count must match transition_pair_indices")
    total_concentration = strict_nonnegative_float(
        total_concentration_mol_m3,
        "total_concentration_mol_m3",
    )
    temperature = strict_positive_float(temperature_K, "temperature_K")
    partition_total = strict_positive_float(
        total_partition_value,
        "total_partition_value",
    )
    fluxes = np.zeros((state_count, state_count), dtype=float)
    for pair_index, (state_index_i, state_index_j) in enumerate(pair_indices):
        points = _as_points(
            transition_quadrature_points[pair_index],
            f"transition_quadrature_points[{pair_index}]",
        )
        weights = _as_weights(
            transition_quadrature_weights[pair_index],
            f"transition_quadrature_weights[{pair_index}]",
            points.shape[0],
        )
        gradients = _as_points(
            transition_committor_gradients[pair_index],
            f"transition_committor_gradients[{pair_index}]",
        )
        if gradients.shape != points.shape:
            raise ValueError("committor gradient shape must match transition points")
        integral = 0.0
        for sample_index, point in enumerate(points):
            mobility = _as_mobility(
                mobility_tensor_m2_s(point),
                points.shape[1],
                "mobility_tensor_m2_s",
            )
            gradient = gradients[sample_index]
            boltzmann_weight = np.exp(
                -float(potential_energy_J_mol(point)) / (R * temperature)
            )
            integral += float(
                weights[sample_index]
                * gradient.T
                @ mobility
                @ gradient
                * boltzmann_weight
            )
        value = total_concentration / partition_total * integral
        fluxes[state_index_i, state_index_j] = value
        fluxes[state_index_j, state_index_i] = value
    return 0.5 * (fluxes + fluxes.T)


def solve_one_dimensional_committors(
    reaction_coordinate_grids: Sequence[np.ndarray],
    free_energy_J_mol_profiles: Sequence[np.ndarray],
    diffusivity_m2_s_profiles: Sequence[np.ndarray],
    temperature_K: float,
) -> dict[str, tuple[np.ndarray, ...] | np.ndarray]:
    _require_equal_lengths(
        reaction_coordinate_grids,
        free_energy_J_mol_profiles,
        "free energy profiles",
    )
    _require_equal_lengths(
        reaction_coordinate_grids,
        diffusivity_m2_s_profiles,
        "diffusivity profiles",
    )
    temperature = strict_positive_float(temperature_K, "temperature_K")
    committor_values = []
    committor_gradients = []
    resistances = []
    for profile_index, coordinate_grid in enumerate(reaction_coordinate_grids):
        coordinates = _as_vector(coordinate_grid, "reaction_coordinate_grid")
        free_energy = _as_vector(
            free_energy_J_mol_profiles[profile_index],
            "free_energy_J_mol_profile",
        )
        diffusivity = _as_vector(
            diffusivity_m2_s_profiles[profile_index],
            "diffusivity_m2_s_profile",
        )
        if len(coordinates) < 2:
            raise ValueError("reaction coordinate grid must contain at least two nodes")
        if free_energy.shape != coordinates.shape or diffusivity.shape != coordinates.shape:
            raise ValueError("one-dimensional profiles must have matching shapes")
        if np.any(np.diff(coordinates) <= 0.0):
            raise ValueError("reaction coordinate grid must be strictly increasing")
        if np.any(diffusivity <= 0.0):
            raise ValueError("diffusivity profile must be positive")
        resistance_density = np.exp(free_energy / (R * temperature)) / diffusivity
        cumulative_resistance = _cumulative_trapezoid(coordinates, resistance_density)
        resistance = float(cumulative_resistance[-1])
        if resistance <= 0.0:
            raise ValueError("Smoluchowski resistance must be positive")
        committor_values.append(cumulative_resistance / resistance)
        committor_gradients.append(resistance_density / resistance)
        resistances.append(resistance)
    return {
        "committor_values": tuple(committor_values),
        "committor_gradients": tuple(committor_gradients),
        "smoluchowski_resistances": np.asarray(resistances, dtype=float),
    }


def compute_one_dimensional_smoluchowski_capacity_fluxes(
    transition_pair_indices: np.ndarray,
    reaction_coordinate_grids: Sequence[np.ndarray],
    free_energy_J_mol_profiles: Sequence[np.ndarray],
    diffusivity_m2_s_profiles: Sequence[np.ndarray],
    total_concentration_mol_m3: float,
    temperature_K: float,
    total_partition_value: float,
    state_count: int,
) -> np.ndarray:
    pair_indices = _as_pair_indices(
        transition_pair_indices,
        "transition_pair_indices",
        state_count,
    )
    solution = solve_one_dimensional_committors(
        reaction_coordinate_grids,
        free_energy_J_mol_profiles,
        diffusivity_m2_s_profiles,
        temperature_K,
    )
    resistances = _as_vector(
        solution["smoluchowski_resistances"],
        "smoluchowski_resistances",
    )
    if len(resistances) != len(pair_indices):
        raise ValueError("resistance count must match transition pair count")
    total_concentration = strict_nonnegative_float(
        total_concentration_mol_m3,
        "total_concentration_mol_m3",
    )
    partition_total = strict_positive_float(
        total_partition_value,
        "total_partition_value",
    )
    fluxes = np.zeros((state_count, state_count), dtype=float)
    for pair_index, (state_index_i, state_index_j) in enumerate(pair_indices):
        value = total_concentration / partition_total / resistances[pair_index]
        fluxes[state_index_i, state_index_j] = value
        fluxes[state_index_j, state_index_i] = value
    return fluxes


def compute_transition_path_displacement_moments(
    transition_pair_indices: np.ndarray,
    transition_path_displacements_m: Sequence[np.ndarray],
    transition_path_weights: Sequence[np.ndarray],
    state_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    pair_indices = _as_pair_indices(
        transition_pair_indices,
        "transition_pair_indices",
        state_count,
    )
    _require_equal_lengths(
        transition_path_displacements_m,
        transition_path_weights,
        "transition path samples",
    )
    if len(pair_indices) != len(transition_path_displacements_m):
        raise ValueError("transition samples must match transition_pair_indices")
    first_moments = np.zeros((state_count, state_count, VECTOR_COMPONENT_COUNT))
    second_moments = np.zeros(
        (state_count, state_count, VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT)
    )
    for pair_index, (state_index_i, state_index_j) in enumerate(pair_indices):
        displacements = _as_displacements(
            transition_path_displacements_m[pair_index],
            "transition_path_displacements_m",
        )
        weights = _as_weights(
            transition_path_weights[pair_index],
            "transition_path_weights",
            displacements.shape[0],
        )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("transition path weights must sum to a positive value")
        normalized_weights = weights / weight_sum
        first_moment = np.einsum("n,nk->k", normalized_weights, displacements)
        second_moment = np.einsum(
            "n,na,nb->ab",
            normalized_weights,
            displacements,
            displacements,
        )
        first_moments[state_index_i, state_index_j] = first_moment
        first_moments[state_index_j, state_index_i] = -first_moment
        second_moments[state_index_i, state_index_j] = second_moment
        second_moments[state_index_j, state_index_i] = second_moment
    return first_moments, second_moments


def compute_transition_path_displacement_moments_from_polarization(
    transition_pair_indices: np.ndarray,
    transition_path_start_points: Sequence[np.ndarray],
    transition_path_end_points: Sequence[np.ndarray],
    transition_path_weights: Sequence[np.ndarray],
    charge_polarization: Callable[[np.ndarray], np.ndarray],
    state_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    _require_equal_lengths(
        transition_path_start_points,
        transition_path_end_points,
        "transition path endpoints",
    )
    displacements = tuple(
        _endpoint_displacements(
            transition_path_start_points[pair_index],
            transition_path_end_points[pair_index],
            charge_polarization,
            f"transition_path[{pair_index}]",
        )
        for pair_index in range(len(transition_path_start_points))
    )
    return compute_transition_path_displacement_moments(
        transition_pair_indices,
        displacements,
        transition_path_weights,
        state_count,
    )


def compute_self_current_tensors(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    mobility_tensor_m2_s: Callable[[np.ndarray], np.ndarray],
    charge_polarization_gradient: Callable[[np.ndarray], np.ndarray],
    basin_quadrature_points: Sequence[np.ndarray],
    basin_quadrature_weights: Sequence[np.ndarray],
    state_concentrations_mol_m3: np.ndarray,
    total_concentration_mol_m3: float,
    temperature_K: float,
    total_partition_value: float,
) -> np.ndarray:
    concentrations = _as_vector(state_concentrations_mol_m3, "state_concentrations")
    total_concentration = strict_nonnegative_float(
        total_concentration_mol_m3,
        "total_concentration_mol_m3",
    )
    temperature = strict_positive_float(temperature_K, "temperature_K")
    partition_total = strict_positive_float(
        total_partition_value,
        "total_partition_value",
    )
    if len(concentrations) != len(basin_quadrature_points):
        raise ValueError("state concentration count must match basin count")
    tensors = np.zeros((len(concentrations), VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT))
    for basin_index, points in enumerate(basin_quadrature_points):
        if concentrations[basin_index] <= 0.0:
            continue
        basin_points = _as_points(points, f"basin_quadrature_points[{basin_index}]")
        weights = _as_weights(
            basin_quadrature_weights[basin_index],
            f"basin_quadrature_weights[{basin_index}]",
            basin_points.shape[0],
        )
        tensor_integral = np.zeros((VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT))
        for sample_index, point in enumerate(basin_points):
            mobility = _as_mobility(
                mobility_tensor_m2_s(point),
                basin_points.shape[1],
                "mobility_tensor_m2_s",
            )
            polarization_gradient = _as_polarization_gradient(
                charge_polarization_gradient(point),
                basin_points.shape[1],
            )
            boltzmann_weight = np.exp(
                -float(potential_energy_J_mol(point)) / (R * temperature)
            )
            tensor_integral += (
                weights[sample_index]
                * boltzmann_weight
                * polarization_gradient
                @ mobility
                @ polarization_gradient.T
            )
        tensors[basin_index] = (
            total_concentration
            / partition_total
            / concentrations[basin_index]
            * tensor_integral
        )
    return tensors


def compute_self_current_tensors_from_polarization(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    mobility_tensor_m2_s: Callable[[np.ndarray], np.ndarray],
    charge_polarization: Callable[[np.ndarray], np.ndarray],
    basin_quadrature_points: Sequence[np.ndarray],
    basin_quadrature_weights: Sequence[np.ndarray],
    state_concentrations_mol_m3: np.ndarray,
    total_concentration_mol_m3: float,
    temperature_K: float,
    total_partition_value: float,
) -> np.ndarray:
    coordinate_dimension = _infer_coordinate_dimension(basin_quadrature_points)
    finite_difference_step = np.full(coordinate_dimension, 1.0e-6)

    def charge_polarization_gradient(point: np.ndarray) -> np.ndarray:
        return compute_charge_polarization_gradient_by_finite_difference(
            charge_polarization,
            point,
            finite_difference_step,
        )

    return compute_self_current_tensors(
        potential_energy_J_mol,
        mobility_tensor_m2_s,
        charge_polarization_gradient,
        basin_quadrature_points,
        basin_quadrature_weights,
        state_concentrations_mol_m3,
        total_concentration_mol_m3,
        temperature_K,
        total_partition_value,
    )


def compute_mori_memory_matrices(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    mobility_tensor_m2_s: Callable[[np.ndarray], np.ndarray],
    charge_polarization_gradient: Callable[[np.ndarray], np.ndarray],
    memory_coordinate_gradient: Callable[[np.ndarray], np.ndarray],
    basin_quadrature_points: Sequence[np.ndarray],
    basin_quadrature_weights: Sequence[np.ndarray],
    total_concentration_mol_m3: float,
    temperature_K: float,
    total_partition_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    total_concentration = strict_nonnegative_float(
        total_concentration_mol_m3,
        "total_concentration_mol_m3",
    )
    temperature = strict_positive_float(temperature_K, "temperature_K")
    partition_total = strict_positive_float(
        total_partition_value,
        "total_partition_value",
    )
    memory_dimension = _infer_memory_dimension(
        memory_coordinate_gradient,
        basin_quadrature_points,
    )
    if memory_dimension == 0:
        return np.zeros((0, 0)), np.zeros((0, VECTOR_COMPONENT_COUNT))
    memory_matrix = np.zeros((memory_dimension, memory_dimension))
    current_coupling = np.zeros((memory_dimension, VECTOR_COMPONENT_COUNT))
    for basin_index, points in enumerate(basin_quadrature_points):
        basin_points = _as_points(points, f"basin_quadrature_points[{basin_index}]")
        weights = _as_weights(
            basin_quadrature_weights[basin_index],
            f"basin_quadrature_weights[{basin_index}]",
            basin_points.shape[0],
        )
        for sample_index, point in enumerate(basin_points):
            mobility = _as_mobility(
                mobility_tensor_m2_s(point),
                basin_points.shape[1],
                "mobility_tensor_m2_s",
            )
            memory_gradient = _as_memory_gradient(
                memory_coordinate_gradient(point),
                memory_dimension,
                basin_points.shape[1],
            )
            polarization_gradient = _as_polarization_gradient(
                charge_polarization_gradient(point),
                basin_points.shape[1],
            )
            density_weight = (
                weights[sample_index]
                * np.exp(-float(potential_energy_J_mol(point)) / (R * temperature))
                * total_concentration
                / partition_total
            )
            memory_matrix += (
                density_weight * memory_gradient @ mobility @ memory_gradient.T
            )
            current_coupling += (
                density_weight
                * memory_gradient
                @ mobility
                @ polarization_gradient.T
            )
    return 0.5 * (memory_matrix + memory_matrix.T), current_coupling


def compute_mori_memory_matrices_from_functions(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    mobility_tensor_m2_s: Callable[[np.ndarray], np.ndarray],
    charge_polarization: Callable[[np.ndarray], np.ndarray],
    memory_coordinates: Callable[[np.ndarray], np.ndarray],
    basin_quadrature_points: Sequence[np.ndarray],
    basin_quadrature_weights: Sequence[np.ndarray],
    total_concentration_mol_m3: float,
    temperature_K: float,
    total_partition_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate_dimension = _infer_coordinate_dimension(basin_quadrature_points)
    finite_difference_step = np.full(coordinate_dimension, 1.0e-6)

    def charge_polarization_gradient(point: np.ndarray) -> np.ndarray:
        return compute_charge_polarization_gradient_by_finite_difference(
            charge_polarization,
            point,
            finite_difference_step,
        )

    def memory_coordinate_gradient(point: np.ndarray) -> np.ndarray:
        return compute_memory_coordinate_gradient_by_finite_difference(
            memory_coordinates,
            point,
            finite_difference_step,
        )

    return compute_mori_memory_matrices(
        potential_energy_J_mol,
        mobility_tensor_m2_s,
        charge_polarization_gradient,
        memory_coordinate_gradient,
        basin_quadrature_points,
        basin_quadrature_weights,
        total_concentration_mol_m3,
        temperature_K,
        total_partition_value,
    )


def compute_reversible_generator(
    symmetric_capacity_fluxes_K_ij_mol_m3_s: np.ndarray,
    state_concentrations_mol_m3: np.ndarray,
) -> np.ndarray:
    concentrations = _as_vector(state_concentrations_mol_m3, "state_concentrations")
    fluxes = _as_square(
        symmetric_capacity_fluxes_K_ij_mol_m3_s,
        "capacity_fluxes",
        len(concentrations),
    )
    generator = np.zeros_like(fluxes)
    for state_index, concentration in enumerate(concentrations):
        if concentration <= 0.0:
            continue
        generator[state_index] = fluxes[state_index] / concentration
        generator[state_index, state_index] = 0.0
        generator[state_index, state_index] = -float(np.sum(generator[state_index]))
    return generator


def compute_finite_state_memory_correction(
    state_concentrations_mol_m3: np.ndarray,
    reversible_generator_Q_ij_s: np.ndarray,
    transition_first_moments_d_ij_m: np.ndarray,
) -> np.ndarray:
    concentrations = _as_vector(state_concentrations_mol_m3, "state_concentrations")
    state_count = len(concentrations)
    generator = _as_square(reversible_generator_Q_ij_s, "generator", state_count)
    first_moments = _as_first_moments(
        transition_first_moments_d_ij_m,
        "transition_first_moments",
        state_count,
    )
    state_drift = np.einsum("ij,ija->ia", generator, first_moments)
    block_matrix = np.zeros((state_count + 1, state_count + 1))
    block_matrix[:state_count, :state_count] = -generator
    block_matrix[:state_count, state_count] = 1.0
    block_matrix[state_count, :state_count] = concentrations
    correctors = np.column_stack(
        [
            _solve_poisson_axis(block_matrix, state_drift[:, axis_index])
            for axis_index in range(VECTOR_COMPONENT_COUNT)
        ]
    )
    correction = np.zeros((VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT))
    for axis_index_a in range(VECTOR_COMPONENT_COUNT):
        for axis_index_b in range(VECTOR_COMPONENT_COUNT):
            correction[axis_index_a, axis_index_b] = float(
                np.sum(
                    concentrations
                    * state_drift[:, axis_index_a]
                    * correctors[:, axis_index_b]
                )
            )
    return 0.5 * (correction + correction.T)


def compute_continuous_mori_correction(
    mori_memory_matrix_A: np.ndarray,
    mori_current_coupling_matrix_h: np.ndarray,
) -> np.ndarray:
    memory_matrix = _as_mori_matrix(mori_memory_matrix_A)
    coupling = _as_mori_coupling(mori_current_coupling_matrix_h, memory_matrix.shape[0])
    if memory_matrix.shape[0] == 0:
        return np.zeros((VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT))
    correction = coupling.T @ np.linalg.pinv(memory_matrix) @ coupling
    return 0.5 * (correction + correction.T)


def conductivity_from_projected_diffusivity(
    projected_diffusivity_tensor: np.ndarray,
    temperature_K: float,
    volume_m3: float = 1.0,
) -> float:
    diffusivity = _as_cartesian_matrix(
        projected_diffusivity_tensor,
        "projected_diffusivity_tensor",
    )
    temperature = strict_positive_float(temperature_K, "temperature_K")
    volume = strict_positive_float(volume_m3, "volume_m3")
    return float(F * F / (R * temperature * volume) * np.trace(diffusivity) / 3.0)


def score_candidate_mori_coordinates(
    current_mori_memory_matrix_A: np.ndarray,
    current_mori_current_coupling_matrix_h: np.ndarray,
    candidate_self_energies_A_gg: np.ndarray,
    candidate_cross_energies_A_gPhi: np.ndarray,
    candidate_current_couplings_h_g: np.ndarray,
) -> dict[str, np.ndarray]:
    current_matrix = _as_mori_matrix(current_mori_memory_matrix_A)
    current_coupling = _as_mori_coupling(
        current_mori_current_coupling_matrix_h,
        current_matrix.shape[0],
    )
    self_energies = _as_vector(candidate_self_energies_A_gg, "candidate_self_energies")
    cross_energies = _as_array(candidate_cross_energies_A_gPhi, "candidate_cross")
    candidate_couplings = _as_array(
        candidate_current_couplings_h_g,
        "candidate_couplings",
    )
    candidate_count = len(self_energies)
    if cross_energies.shape != (candidate_count, current_matrix.shape[0]):
        raise ValueError("candidate_cross_energies_A_gPhi has incompatible shape")
    if candidate_couplings.shape != (candidate_count, VECTOR_COMPONENT_COUNT):
        raise ValueError("candidate_current_couplings_h_g has incompatible shape")
    if current_matrix.shape[0] == 0:
        residual_coupling = candidate_couplings
        residual_energy = self_energies
    else:
        inverse_current = np.linalg.pinv(current_matrix)
        residual_coupling = (
            candidate_couplings - cross_energies @ inverse_current @ current_coupling
        )
        residual_energy = self_energies - np.einsum(
            "ga,ab,gb->g",
            cross_energies,
            inverse_current,
            cross_energies,
        )
    if np.any(residual_energy <= 0.0):
        raise ValueError("candidate residual energies must be positive")
    scores = np.einsum("ga,ga->g", residual_coupling, residual_coupling) / residual_energy
    return {
        "residual_coupling": residual_coupling,
        "residual_energy": residual_energy,
        "scores": scores,
    }


def refine_mori_basis_by_projected_residual(
    direct_minus_finite_state_diffusivity_tensor: np.ndarray,
    initial_mori_memory_matrix_A: np.ndarray,
    initial_mori_current_coupling_matrix_h: np.ndarray,
    candidate_self_energies_A_gg: np.ndarray,
    candidate_cross_energies_A_gPhi: np.ndarray,
    candidate_pair_cross_energies_A_gg: np.ndarray,
    candidate_current_couplings_h_g: np.ndarray,
    temperature_K: float,
    conductivity_delta_tolerance_S_m: float,
    maximum_added_coordinates: int,
) -> dict[str, np.ndarray | float]:
    base_diffusivity = _as_cartesian_matrix(
        direct_minus_finite_state_diffusivity_tensor,
        "direct_minus_finite_state_diffusivity_tensor",
    )
    current_matrix = _as_mori_matrix(initial_mori_memory_matrix_A)
    current_coupling = _as_mori_coupling(
        initial_mori_current_coupling_matrix_h,
        current_matrix.shape[0],
    )
    self_energies = _as_vector(candidate_self_energies_A_gg, "candidate_self_energies")
    original_cross = _as_array(candidate_cross_energies_A_gPhi, "candidate_cross")
    pair_cross = _as_array(candidate_pair_cross_energies_A_gg, "candidate_pair_cross")
    candidate_couplings = _as_array(candidate_current_couplings_h_g, "candidate_couplings")
    tolerance = strict_nonnegative_float(
        conductivity_delta_tolerance_S_m,
        "conductivity_delta_tolerance_S_m",
    )
    max_added = int(maximum_added_coordinates)
    if max_added < 0:
        raise ValueError("maximum_added_coordinates must be non-negative")
    selected_indices: list[int] = []
    remaining_indices = list(range(len(self_energies)))
    history = [
        conductivity_from_projected_diffusivity(
            base_diffusivity - compute_continuous_mori_correction(current_matrix, current_coupling),
            temperature_K,
        )
    ]
    for _iteration_index in range(max_added):
        if not remaining_indices:
            break
        cross_to_current = _candidate_cross_to_current(
            remaining_indices,
            selected_indices,
            original_cross,
            pair_cross,
        )
        scoring = score_candidate_mori_coordinates(
            current_matrix,
            current_coupling,
            self_energies[remaining_indices],
            cross_to_current,
            candidate_couplings[remaining_indices],
        )
        local_best_index = int(np.argmax(scoring["scores"]))
        best_index = remaining_indices[local_best_index]
        current_matrix = _augment_mori_matrix(
            current_matrix,
            cross_to_current[local_best_index],
            self_energies[best_index],
        )
        current_coupling = np.vstack([current_coupling, candidate_couplings[best_index]])
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)
        new_sigma = conductivity_from_projected_diffusivity(
            base_diffusivity - compute_continuous_mori_correction(current_matrix, current_coupling),
            temperature_K,
        )
        history.append(new_sigma)
        if abs(history[-1] - history[-2]) <= tolerance:
            break
    return {
        "selected_candidate_indices": np.asarray(selected_indices, dtype=int),
        "final_mori_memory_matrix_A": current_matrix,
        "final_mori_current_coupling_matrix_h": current_coupling,
        "conductivity_history_S_m": np.asarray(history, dtype=float),
        "final_sigma_S_m": float(history[-1]),
    }


def compute_charge_polarization_gradient_by_finite_difference(
    charge_polarization: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    finite_difference_step: np.ndarray,
) -> np.ndarray:
    return _finite_difference_jacobian(
        charge_polarization,
        point,
        finite_difference_step,
        VECTOR_COMPONENT_COUNT,
        "charge_polarization",
    )


def compute_memory_coordinate_gradient_by_finite_difference(
    memory_coordinates: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    finite_difference_step: np.ndarray,
) -> np.ndarray:
    point_array = _as_vector(point, "point")
    step = _as_vector(finite_difference_step, "finite_difference_step")
    baseline = _as_vector(memory_coordinates(point_array), "memory_coordinates")
    if len(baseline) == 0:
        return np.zeros((0, len(point_array)))
    return _finite_difference_jacobian(
        memory_coordinates,
        point_array,
        step,
        len(baseline),
        "memory_coordinates",
    )


def compute_projected_analytical_conductivity_from_one_dimensional_reaction_coordinates(
    potential_energy_J_mol: Callable[[np.ndarray], float],
    mobility_tensor_m2_s: Callable[[np.ndarray], np.ndarray],
    charge_polarization: Callable[[np.ndarray], np.ndarray],
    memory_coordinates: Callable[[np.ndarray], np.ndarray],
    basin_quadrature_points: Sequence[np.ndarray],
    basin_quadrature_weights: Sequence[np.ndarray],
    transition_pair_indices: np.ndarray,
    reaction_coordinate_grids: Sequence[np.ndarray],
    free_energy_J_mol_profiles: Sequence[np.ndarray],
    diffusivity_m2_s_profiles: Sequence[np.ndarray],
    transition_path_start_points: Sequence[np.ndarray],
    transition_path_end_points: Sequence[np.ndarray],
    transition_path_weights: Sequence[np.ndarray],
    total_concentration_mol_m3: float,
    temperature_K: float,
    volume_m3: float,
) -> dict[str, np.ndarray | float]:
    partition_values = compute_restricted_partition_values(
        potential_energy_J_mol,
        basin_quadrature_points,
        basin_quadrature_weights,
        temperature_K,
    )
    capacity_fluxes = compute_one_dimensional_smoluchowski_capacity_fluxes(
        transition_pair_indices,
        reaction_coordinate_grids,
        free_energy_J_mol_profiles,
        diffusivity_m2_s_profiles,
        total_concentration_mol_m3,
        temperature_K,
        float(np.sum(partition_values)),
        len(partition_values),
    )
    coordinate_dimension = _infer_coordinate_dimension(basin_quadrature_points)
    finite_difference_step = np.full(coordinate_dimension, 1.0e-6)

    def charge_polarization_gradient(point: np.ndarray) -> np.ndarray:
        return compute_charge_polarization_gradient_by_finite_difference(
            charge_polarization,
            point,
            finite_difference_step,
        )

    def memory_coordinate_gradient(point: np.ndarray) -> np.ndarray:
        return compute_memory_coordinate_gradient_by_finite_difference(
            memory_coordinates,
            point,
            finite_difference_step,
        )

    concentrations = compute_equilibrium_populations(
        partition_values,
        total_concentration_mol_m3,
    )
    first_moments, second_moments = (
        compute_transition_path_displacement_moments_from_polarization(
            transition_pair_indices,
            transition_path_start_points,
            transition_path_end_points,
            transition_path_weights,
            charge_polarization,
            len(partition_values),
        )
    )
    self_current_tensors = compute_self_current_tensors(
        potential_energy_J_mol,
        mobility_tensor_m2_s,
        charge_polarization_gradient,
        basin_quadrature_points,
        basin_quadrature_weights,
        concentrations,
        total_concentration_mol_m3,
        temperature_K,
        float(np.sum(partition_values)),
    )
    mori_matrix, mori_coupling = compute_mori_memory_matrices(
        potential_energy_J_mol,
        mobility_tensor_m2_s,
        charge_polarization_gradient,
        memory_coordinate_gradient,
        basin_quadrature_points,
        basin_quadrature_weights,
        total_concentration_mol_m3,
        temperature_K,
        float(np.sum(partition_values)),
    )
    result = compute_projected_analytical_conductivity_from_primitives(
        concentrations,
        capacity_fluxes,
        first_moments,
        second_moments,
        self_current_tensors,
        mori_matrix,
        mori_coupling,
        temperature_K,
        volume_m3,
    )
    result["symmetric_capacity_fluxes_K_ij_mol_m3_s"] = capacity_fluxes
    return result


def conductivity_effect_primitive_locations() -> Mapping[str, tuple[str, ...]]:
    return {
        "free_ion_fraction": ("c_i",),
        "ion_association": ("c_i", "K_ij"),
        "SSIP_CIP_balance": ("c_i", "K_ij", "D_self_i"),
        "aggregation": ("c_i", "K_ij", "D_self_i", "M_ij"),
        "neutral_ligand_coordination": ("c_i", "K_ij", "D_self_i", "A", "h"),
        "Li_anion_anticorrelation": ("D_self_i",),
        "partner_switching": ("K_ij", "d_ij", "M_ij", "A", "h"),
        "identity_diffusion": ("K_ij", "d_ij", "M_ij"),
        "structural_hopping": ("K_ij", "d_ij", "M_ij"),
        "cage_backjump": ("D_Q_corr", "A", "h"),
        "ion_atmosphere_relaxation": ("A", "h"),
    }


def _projected_recipe_parameters(config):
    ion_pairing_config = strict_mapping(
        require_key(config, "ion_pairing_model", "physics_config"),
        "physics_config.ion_pairing_model",
    )
    osm_transport_config = strict_mapping(
        require_key(config, "osm_transport_model", "physics_config"),
        "physics_config.osm_transport_model",
    )
    transport_arrhenius_config = strict_mapping(
        require_key(config, "transport_arrhenius", "physics_config"),
        "physics_config.transport_arrhenius",
    )
    concentration_transport_config = strict_mapping(
        require_key(config, "concentration_dependent_transport", "physics_config"),
        "physics_config.concentration_dependent_transport",
    )
    concentration_sweep_config = strict_mapping(
        require_key(
            concentration_transport_config,
            "concentration_sweep",
            "physics_config.concentration_dependent_transport",
        ),
        "physics_config.concentration_dependent_transport.concentration_sweep",
    )
    bjerrum_eps_ref = strict_positive_float(
        require_key(ion_pairing_config, "bjerrum_eps_ref", "ion_pairing_model"),
        "ion_pairing_model.bjerrum_eps_ref",
    )
    epsilon_water = strict_positive_float(
        EPSILON_R_WATER_25C,
        "EPSILON_R_WATER_25C",
    )
    crowding_molarity_scale = strict_positive_float(
        require_key(concentration_sweep_config, "c_max_M", "concentration_sweep"),
        "concentration_sweep.c_max_M",
    )
    aggregate_scale_M = strict_positive_float(
        require_key(ion_pairing_config, "aggregate_scale_mol_l", "ion_pairing_model"),
        "ion_pairing_model.aggregate_scale_mol_l",
    )
    return {
        "dielectric_support_scale": epsilon_water + bjerrum_eps_ref,
        "viscosity_reference_cP": strict_positive_float(
            require_key(
                transport_arrhenius_config,
                "reference_viscosity_cP",
                "transport_arrhenius",
            ),
            "transport_arrhenius.reference_viscosity_cP",
        ),
        "viscosity_exponent": strict_positive_float(
            require_key(osm_transport_config, "cation_se_alpha", "osm_transport_model"),
            "osm_transport_model.cation_se_alpha",
        ),
        "crowding_molarity_scale": crowding_molarity_scale,
        "crowding_curvature_molarity": crowding_molarity_scale + aggregate_scale_M,
        "bjerrum_eps_ref": bjerrum_eps_ref,
        "osm_transport_config": osm_transport_config,
    }


def _projected_recipe_mixture_properties(solvent_mapping, additive_mapping):
    component_fractions: dict[str, float] = {}
    for solvent_name, solvent_fraction in solvent_mapping.items():
        component_fractions[str(solvent_name)] = strict_nonnegative_float(
            solvent_fraction,
            f"recipe.solvents.{solvent_name}",
        )
    for additive_name, additive_fraction in additive_mapping.items():
        if not _projected_recipe_species_is_ionic_source(str(additive_name)):
            component_fractions[str(additive_name)] = strict_nonnegative_float(
                additive_fraction,
                f"recipe.additives.{additive_name}",
            )
    total_fraction = float(sum(component_fractions.values()))
    if total_fraction <= 0.0:
        raise ValueError("recipe must contain positive neutral liquid fraction")
    epsilon_r = 0.0
    log_viscosity = 0.0
    inverse_density = 0.0
    for species_name, fraction in component_fractions.items():
        normalized_fraction = fraction / total_fraction
        epsilon_r += normalized_fraction * _projected_recipe_species_float(
            species_name,
            "epsilon_r",
        )
        log_viscosity += normalized_fraction * np.log(
            _projected_recipe_species_float(species_name, "viscosity_cP")
        )
        inverse_density += normalized_fraction / _projected_recipe_species_float(
            species_name,
            "density_g_ml",
        )
    if inverse_density <= 0.0:
        raise ValueError("neutral liquid density mixing denominator is non-positive")
    return {
        "epsilon_r": epsilon_r,
        "viscosity_cP": float(np.exp(log_viscosity)),
        "density_g_ml": 1.0 / inverse_density,
    }


def _projected_recipe_ionic_sources(
    salt_mapping,
    additive_mapping,
    electrolyte_density_g_ml: float,
) -> dict[str, float]:
    density = strict_positive_float(electrolyte_density_g_ml, "electrolyte_density_g_ml")
    ionic_sources: dict[str, float] = {}
    for salt_name, salt_molarity in salt_mapping.items():
        ionic_sources[str(salt_name)] = strict_nonnegative_float(
            salt_molarity,
            f"recipe.salts.{salt_name}",
        )
    for additive_name, additive_fraction in additive_mapping.items():
        if _projected_recipe_species_is_ionic_source(str(additive_name)):
            source_name = str(additive_name)
            molecular_weight = _projected_recipe_species_float(
                source_name,
                "molecular_weight",
            )
            weight_fraction = strict_nonnegative_float(
                additive_fraction,
                f"recipe.additives.{additive_name}",
            )
            additive_molarity = (
                weight_fraction
                * density
                * MOL_M3_PER_MOL_L
                / molecular_weight
            )
            if source_name in ionic_sources:
                ionic_sources[source_name] += additive_molarity
            else:
                ionic_sources[source_name] = additive_molarity
    return ionic_sources


def _projected_recipe_effective_dielectric(
    ionic_source_molarities_M: Mapping[str, float],
    mixture_epsilon_r: float,
) -> float:
    epsilon_r = strict_positive_float(mixture_epsilon_r, "mixture_epsilon_r")
    decrement_fraction = 0.0
    for source_name, source_molarity in ionic_source_molarities_M.items():
        decrement_fraction += strict_nonnegative_float(
            source_molarity,
            f"ionic_source_molarities_M.{source_name}",
        ) * _projected_recipe_species_float(
            source_name,
            "dielectric_decrement_frac_per_M",
        )
    effective_epsilon = epsilon_r * (1.0 - decrement_fraction)
    if effective_epsilon <= 0.0:
        raise ValueError(
            "projected recipe effective dielectric became non-positive: "
            f"epsilon_r={epsilon_r}, decrement_fraction={decrement_fraction}"
        )
    return effective_epsilon


def _projected_recipe_free_ion_fraction(
    source_name: str,
    source_molarity_M: float,
    effective_epsilon_r: float,
    bjerrum_eps_ref: float,
) -> float:
    source_molarity = strict_positive_float(
        source_molarity_M,
        f"ionic_source_molarities_M.{source_name}",
    )
    epsilon_effective = strict_positive_float(
        effective_epsilon_r,
        "effective_epsilon_r",
    )
    epsilon_reference = strict_positive_float(bjerrum_eps_ref, "bjerrum_eps_ref")
    ion_pair_Kd_M = _projected_recipe_ion_pair_Kd_M(source_name)
    effective_Kd_M = ion_pair_Kd_M * epsilon_effective / epsilon_reference
    discriminant = effective_Kd_M * effective_Kd_M + (
        4.0 * effective_Kd_M * source_molarity
    )
    free_fraction = (
        -effective_Kd_M + float(np.sqrt(discriminant))
    ) / (2.0 * source_molarity)
    if free_fraction <= 0.0 or free_fraction > 1.0:
        raise ValueError(
            f"computed free-ion fraction for {source_name} is outside (0, 1]: "
            f"{free_fraction}"
        )
    return free_fraction


def _projected_recipe_ion_pair_Kd_M(source_name: str) -> float:
    ion_pair_Kd = get_species_property(source_name, "ion_pair_Kd_M")
    if ion_pair_Kd is not None:
        return strict_positive_float(ion_pair_Kd, f"{source_name}.ion_pair_Kd_M")
    ion_pair_binding = _projected_recipe_species_float(
        source_name,
        "ion_pair_binding_kj_mol",
    )
    reference_binding = _projected_recipe_species_float("LiPF6", "ion_pair_binding_kj_mol")
    reference_Kd = strict_positive_float(
        get_species_property("LiPF6", "ion_pair_Kd_M"),
        "LiPF6.ion_pair_Kd_M",
    )
    return reference_Kd * reference_binding / ion_pair_binding


def _projected_recipe_limited_molar_conductivity_S_cm2_mol(
    source_name: str,
    osm_transport_config,
) -> float:
    lambda0 = _projected_recipe_species_float(source_name, "Lambda_0")
    scale_key = f"lambda0_scale_{source_name}"
    if scale_key in osm_transport_config:
        lambda0 *= strict_positive_float(
            osm_transport_config[scale_key],
            f"osm_transport_model.{scale_key}",
        )
    return lambda0


def _projected_recipe_primitives_from_transport_scalars(
    ionic_source_molarities_M: Mapping[str, float],
    epsilon_effective: float,
    dielectric_support: float,
    viscosity_factor: float,
    crowding_factor: float,
    temperature_K: float,
    volume_m3: float,
    recipe_parameters: Mapping[str, float],
    mixture_properties: Mapping[str, float],
):
    direct_density_tensor = np.zeros((VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT))
    projected_density_tensor = np.zeros((VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT))
    state_concentrations: list[float] = []
    self_current_tensors: list[np.ndarray] = []
    free_fraction_by_source: dict[str, float] = {}
    projected_contribution_by_source_mS_cm: dict[str, float] = {}
    uncorrected_contribution_by_source_mS_cm: dict[str, float] = {}
    bjerrum_eps_ref = strict_positive_float(
        recipe_parameters["bjerrum_eps_ref"],
        "recipe_parameters.bjerrum_eps_ref",
    )
    osm_transport_config = strict_mapping(
        require_key(recipe_parameters, "osm_transport_config", "recipe_parameters"),
        "recipe_parameters.osm_transport_config",
    )

    for source_name, source_molarity_M in ionic_source_molarities_M.items():
        source_molarity = strict_nonnegative_float(
            source_molarity_M,
            f"ionic_source_molarities_M.{source_name}",
        )
        if source_molarity == 0.0:
            continue
        free_fraction = _projected_recipe_free_ion_fraction(
            source_name,
            source_molarity,
            epsilon_effective,
            bjerrum_eps_ref,
        )
        lambda0 = _projected_recipe_limited_molar_conductivity_S_cm2_mol(
            source_name,
            osm_transport_config,
        )
        uncorrected_sigma_mS_cm = source_molarity * lambda0 * free_fraction
        projected_sigma_mS_cm = (
            uncorrected_sigma_mS_cm
            * dielectric_support
            * viscosity_factor
            * crowding_factor
        )
        free_concentration_mol_m3 = (
            source_molarity * MOL_M3_PER_MOL_L * free_fraction
        )
        if free_concentration_mol_m3 <= 0.0:
            continue
        uncorrected_density = _sigma_mS_cm_to_diffusivity_density(
            uncorrected_sigma_mS_cm,
            temperature_K,
            volume_m3,
        )
        projected_density = _sigma_mS_cm_to_diffusivity_density(
            projected_sigma_mS_cm,
            temperature_K,
            volume_m3,
        )
        direct_density_tensor += np.eye(VECTOR_COMPONENT_COUNT) * uncorrected_density
        projected_density_tensor += np.eye(VECTOR_COMPONENT_COUNT) * projected_density
        state_concentrations.append(free_concentration_mol_m3)
        self_current_tensors.append(
            np.eye(VECTOR_COMPONENT_COUNT)
            * uncorrected_density
            / free_concentration_mol_m3
        )
        free_fraction_by_source[source_name] = free_fraction
        projected_contribution_by_source_mS_cm[source_name] = projected_sigma_mS_cm
        uncorrected_contribution_by_source_mS_cm[source_name] = uncorrected_sigma_mS_cm

    state_count = len(state_concentrations)
    if state_count == 0:
        raise ValueError("projected recipe builder produced no mobile carrier states")
    memory_density_tensor = direct_density_tensor - projected_density_tensor
    mori_memory_matrix_A, mori_current_coupling_matrix_h = (
        _isotropic_mori_primitives_from_density_tensor(memory_density_tensor)
    )
    return {
        "state_concentrations_mol_m3": np.asarray(state_concentrations, dtype=float),
        "symmetric_capacity_fluxes_K_ij_mol_m3_s": np.zeros((state_count, state_count)),
        "transition_first_moments_d_ij_m": np.zeros(
            (state_count, state_count, VECTOR_COMPONENT_COUNT)
        ),
        "transition_second_moments_M_ij_m2": np.zeros(
            (state_count, state_count, VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT)
        ),
        "self_current_tensors_D_self_i_m2_s": np.asarray(
            self_current_tensors,
            dtype=float,
        ),
        "mori_memory_matrix_A": mori_memory_matrix_A,
        "mori_current_coupling_matrix_h": mori_current_coupling_matrix_h,
        "temperature_K": temperature_K,
        "volume_m3": volume_m3,
        "mixture_epsilon_r": mixture_properties["epsilon_r"],
        "mixture_viscosity_cP": mixture_properties["viscosity_cP"],
        "effective_epsilon_r": epsilon_effective,
        "free_fraction_by_source": free_fraction_by_source,
        "projected_contribution_by_source_mS_cm": projected_contribution_by_source_mS_cm,
        "uncorrected_contribution_by_source_mS_cm": (
            uncorrected_contribution_by_source_mS_cm
        ),
    }


def _isotropic_mori_primitives_from_density_tensor(density_tensor: np.ndarray):
    density_matrix = _as_cartesian_matrix(density_tensor, "mori_density_tensor")
    diagonal_density = np.diag(density_matrix)
    if np.any(diagonal_density < 0.0):
        raise ValueError("Mori density tensor diagonal must be non-negative")
    active_indices = tuple(
        axis_index
        for axis_index, value in enumerate(diagonal_density)
        if value > 0.0
    )
    if not active_indices:
        return np.zeros((0, 0)), np.zeros((0, VECTOR_COMPONENT_COUNT))
    memory_matrix = np.diag(
        np.asarray([diagonal_density[axis_index] for axis_index in active_indices])
    )
    current_coupling = np.zeros((len(active_indices), VECTOR_COMPONENT_COUNT))
    for memory_index, axis_index in enumerate(active_indices):
        current_coupling[memory_index, axis_index] = diagonal_density[axis_index]
    return memory_matrix, current_coupling


def _sigma_mS_cm_to_diffusivity_density(
    sigma_mS_cm: float,
    temperature_K: float,
    volume_m3: float,
) -> float:
    sigma_S_m = strict_nonnegative_float(sigma_mS_cm, "sigma_mS_cm") * MS_CM_TO_S_M
    temperature = strict_positive_float(temperature_K, "temperature_K")
    volume = strict_positive_float(volume_m3, "volume_m3")
    return sigma_S_m * R * temperature * volume / (F * F)


def _projected_recipe_species_float(species_name: str, property_name: str) -> float:
    value = get_species_property(species_name, property_name)
    if value is None:
        raise ValueError(f"species {species_name} missing required {property_name}")
    return strict_positive_float(value, f"{species_name}.{property_name}")


def _projected_recipe_species_is_ionic_source(species_name: str) -> bool:
    configured_value = get_species_property(species_name, "provides_ionic_conductivity")
    if configured_value is not None:
        if not isinstance(configured_value, bool):
            raise ValueError(
                f"{species_name}.provides_ionic_conductivity must be boolean"
            )
        return configured_value
    has_limiting_conductivity = (
        get_species_property(species_name, "Lambda_0") is not None
    )
    has_anion_charge = get_species_property(species_name, "anion_charge") is not None
    return has_limiting_conductivity and has_anion_charge


def _direct_diffusivity(
    concentrations: np.ndarray,
    capacity_fluxes: np.ndarray,
    second_moments: np.ndarray,
    self_current_tensors: np.ndarray,
) -> np.ndarray:
    self_term = np.einsum("i,iab->ab", concentrations, self_current_tensors)
    transition_term = 0.5 * np.einsum("ij,ijab->ab", capacity_fluxes, second_moments)
    result = self_term + transition_term
    return 0.5 * (result + result.T)


def _solve_poisson_axis(block_matrix: np.ndarray, drift_axis: np.ndarray) -> np.ndarray:
    state_count = len(drift_axis)
    right_hand_side = np.zeros(state_count + 1)
    right_hand_side[:state_count] = drift_axis
    return (np.linalg.pinv(block_matrix) @ right_hand_side)[:state_count]


def _endpoint_displacements(
    start_points: np.ndarray,
    end_points: np.ndarray,
    charge_polarization: Callable[[np.ndarray], np.ndarray],
    context: str,
) -> np.ndarray:
    starts = _as_points(start_points, f"{context}.start_points")
    ends = _as_points(end_points, f"{context}.end_points")
    if starts.shape != ends.shape:
        raise ValueError(f"{context} start and end arrays must match")
    return np.asarray(
        [
            _as_cartesian_vector(charge_polarization(end), f"{context}.end")
            - _as_cartesian_vector(charge_polarization(start), f"{context}.start")
            for start, end in zip(starts, ends)
        ],
        dtype=float,
    )


def _finite_difference_jacobian(
    vector_function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    finite_difference_step: np.ndarray,
    output_dimension: int,
    context: str,
) -> np.ndarray:
    point_array = _as_vector(point, "point")
    step = _as_vector(finite_difference_step, "finite_difference_step")
    if step.shape != point_array.shape or np.any(step <= 0.0):
        raise ValueError("finite_difference_step must be positive and match point shape")
    jacobian = np.zeros((output_dimension, len(point_array)))
    for coordinate_index in range(len(point_array)):
        perturbation = np.zeros_like(point_array)
        perturbation[coordinate_index] = step[coordinate_index]
        forward = _as_vector(vector_function(point_array + perturbation), context)
        backward = _as_vector(vector_function(point_array - perturbation), context)
        if forward.shape != (output_dimension,) or backward.shape != (output_dimension,):
            raise ValueError(f"{context} output dimension must be {output_dimension}")
        jacobian[:, coordinate_index] = (
            forward - backward
        ) / (2.0 * step[coordinate_index])
    return jacobian


def _candidate_cross_to_current(
    remaining_indices: Sequence[int],
    selected_indices: Sequence[int],
    original_candidate_cross: np.ndarray,
    candidate_pair_cross: np.ndarray,
) -> np.ndarray:
    rows = []
    for candidate_index in remaining_indices:
        original = original_candidate_cross[candidate_index]
        selected = np.asarray(
            [candidate_pair_cross[candidate_index, selected_index] for selected_index in selected_indices],
            dtype=float,
        )
        rows.append(np.concatenate([original, selected]))
    if not rows:
        return np.zeros((0, original_candidate_cross.shape[1] + len(selected_indices)))
    return np.vstack(rows)


def _augment_mori_matrix(
    current_matrix: np.ndarray,
    cross_to_current: np.ndarray,
    candidate_self_energy: float,
) -> np.ndarray:
    augmented = np.zeros((current_matrix.shape[0] + 1, current_matrix.shape[1] + 1))
    augmented[:-1, :-1] = current_matrix
    augmented[-1, :-1] = cross_to_current
    augmented[:-1, -1] = cross_to_current
    augmented[-1, -1] = candidate_self_energy
    return augmented


def _cumulative_trapezoid(coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(coordinates)
    for coordinate_index in range(1, len(coordinates)):
        width = coordinates[coordinate_index] - coordinates[coordinate_index - 1]
        cumulative[coordinate_index] = (
            cumulative[coordinate_index - 1]
            + 0.5 * width * (values[coordinate_index - 1] + values[coordinate_index])
        )
    return cumulative


def _infer_coordinate_dimension(basin_quadrature_points: Sequence[np.ndarray]) -> int:
    if len(basin_quadrature_points) == 0:
        raise ValueError("at least one basin is required")
    return int(_as_points(basin_quadrature_points[0], "basin_quadrature_points[0]").shape[1])


def _infer_memory_dimension(
    memory_coordinate_gradient: Callable[[np.ndarray], np.ndarray],
    basin_quadrature_points: Sequence[np.ndarray],
) -> int:
    point = _as_points(basin_quadrature_points[0], "basin_quadrature_points[0]")[0]
    gradient = _as_array(memory_coordinate_gradient(point), "memory_coordinate_gradient")
    if gradient.ndim != 2:
        raise ValueError("memory_coordinate_gradient must be a 2-D array")
    return int(gradient.shape[0])


def _require_equal_lengths(
    left_sequence: Sequence[np.ndarray],
    right_sequence: Sequence[np.ndarray],
    context: str,
) -> None:
    if len(left_sequence) != len(right_sequence):
        raise ValueError(f"{context} sequences must have matching lengths")


def _as_array(values: np.ndarray, context: str) -> np.ndarray:
    return strict_finite_array(values, context).astype(float)


def _as_vector(values: np.ndarray, context: str) -> np.ndarray:
    array = _as_array(values, context)
    if array.ndim != 1:
        raise ValueError(f"{context} must be a 1-D array")
    return array


def _as_cartesian_vector(values: np.ndarray, context: str) -> np.ndarray:
    vector = _as_vector(values, context)
    if vector.shape != (VECTOR_COMPONENT_COUNT,):
        raise ValueError(f"{context} must have shape ({VECTOR_COMPONENT_COUNT},)")
    return vector


def _as_points(values: np.ndarray, context: str) -> np.ndarray:
    array = _as_array(values, context)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{context} must be a non-empty point matrix")
    return array


def _as_weights(values: np.ndarray, context: str, expected_length: int) -> np.ndarray:
    weights = _as_vector(values, context)
    if weights.shape != (expected_length,):
        raise ValueError(f"{context} must have length {expected_length}")
    if np.any(weights < 0.0):
        raise ValueError(f"{context} must be non-negative")
    return weights


def _as_pair_indices(values: np.ndarray, context: str, state_count: int) -> np.ndarray:
    array = strict_finite_array(values, context).astype(int)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{context} must have shape (n_pairs, 2)")
    if np.any(array < 0) or np.any(array >= state_count):
        raise ValueError(f"{context} contains an out-of-range state index")
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError(f"{context} cannot contain self transitions")
    return array


def _as_square(values: np.ndarray, context: str, expected_size: int) -> np.ndarray:
    array = _as_array(values, context)
    if array.shape != (expected_size, expected_size):
        raise ValueError(f"{context} must have shape ({expected_size}, {expected_size})")
    return array


def _as_cartesian_matrix(values: np.ndarray, context: str) -> np.ndarray:
    array = _as_array(values, context)
    if array.shape != (VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT):
        raise ValueError(f"{context} must have shape (3, 3)")
    return array


def _as_first_moments(values: np.ndarray, context: str, state_count: int) -> np.ndarray:
    array = _as_array(values, context)
    if array.shape != (state_count, state_count, VECTOR_COMPONENT_COUNT):
        raise ValueError(f"{context} must have shape ({state_count}, {state_count}, 3)")
    return array


def _as_second_moments(values: np.ndarray, context: str, state_count: int) -> np.ndarray:
    array = _as_array(values, context)
    expected_shape = (state_count, state_count, VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT)
    if array.shape != expected_shape:
        raise ValueError(f"{context} must have shape {expected_shape}")
    return array


def _as_state_tensors(values: np.ndarray, context: str, state_count: int) -> np.ndarray:
    array = _as_array(values, context)
    if array.shape != (state_count, VECTOR_COMPONENT_COUNT, VECTOR_COMPONENT_COUNT):
        raise ValueError(f"{context} must have shape ({state_count}, 3, 3)")
    return array


def _as_displacements(values: np.ndarray, context: str) -> np.ndarray:
    array = _as_array(values, context)
    if array.ndim == 1 and array.shape == (VECTOR_COMPONENT_COUNT,):
        array = array.reshape(1, VECTOR_COMPONENT_COUNT)
    if array.ndim != 2 or array.shape[1] != VECTOR_COMPONENT_COUNT:
        raise ValueError(f"{context} must have shape (n_samples, 3)")
    return array


def _as_mobility(
    values: np.ndarray,
    coordinate_dimension: int,
    context: str,
) -> np.ndarray:
    array = _as_array(values, context)
    if array.shape != (coordinate_dimension, coordinate_dimension):
        raise ValueError(f"{context} must have shape ({coordinate_dimension}, {coordinate_dimension})")
    return array


def _as_polarization_gradient(values: np.ndarray, coordinate_dimension: int) -> np.ndarray:
    array = _as_array(values, "charge_polarization_gradient")
    if array.shape != (VECTOR_COMPONENT_COUNT, coordinate_dimension):
        raise ValueError("charge_polarization_gradient has incompatible shape")
    return array


def _as_memory_gradient(
    values: np.ndarray,
    memory_dimension: int,
    coordinate_dimension: int,
) -> np.ndarray:
    array = _as_array(values, "memory_coordinate_gradient")
    if array.shape != (memory_dimension, coordinate_dimension):
        raise ValueError("memory_coordinate_gradient has incompatible shape")
    return array


def _as_mori_matrix(values: np.ndarray) -> np.ndarray:
    array = _as_array(values, "mori_memory_matrix_A")
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("mori_memory_matrix_A must be square")
    return array


def _as_mori_coupling(values: np.ndarray, memory_dimension: int) -> np.ndarray:
    array = _as_array(values, "mori_current_coupling_matrix_h")
    if array.shape != (memory_dimension, VECTOR_COMPONENT_COUNT):
        raise ValueError("mori_current_coupling_matrix_h has incompatible shape")
    return array
