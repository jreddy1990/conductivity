"""Build reduced-generator inputs from site-level physical configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from conductivity.physical_library.projected_analytical_conductivity import (
    StateTransportOwnershipBasis,
    TransportOwnership,
)
from conductivity.physical_library.physical_objects import (
    CARTESIAN_DIMENSION,
    PhysicalObjectBundle,
    SiteConfiguration,
    build_physical_objects,
    compute_charge_polarization_m,
)
from conductivity.physical_library.reduced_generator import (
    ReducedGeneratorSpecification,
    ReducedStateQuadrature,
    ReducedTransitionQuadrature,
)
from conductivity.physical_library.library_io import PhysicalLibraryRecords

Array = np.ndarray
LOCAL_FIELD_VECTOR_LENGTH = 4


@dataclass(frozen=True)
class PhysicalLocalFields:
    dielectric_constant: float
    viscosity_Pa_s: float
    ionic_strength_mol_m3: float
    local_packing_fraction: float


@dataclass(frozen=True)
class PhysicalStateQuadrature:
    label: str
    configurations: tuple[SiteConfiguration, ...]
    local_fields: tuple[PhysicalLocalFields, ...]
    weights: Array
    stoichiometry: Array
    self_current_projector: Array
    transport_ownership_bases: tuple[StateTransportOwnershipBasis, ...]
    relative_displacement_fluctuations_m: Array
    relative_displacement_mobility_m2_s: Array
    relative_center_charge_numbers: Array


@dataclass(frozen=True)
class PhysicalTransitionQuadrature:
    from_state_index: int
    to_state_index: int
    transition_family: str
    transport_ownership: TransportOwnership
    configurations: tuple[SiteConfiguration, ...]
    local_fields: tuple[PhysicalLocalFields, ...]
    weights: Array
    committor_gradients: Array
    surface_state_indices: Array
    path_start_configurations: tuple[SiteConfiguration, ...]
    path_end_configurations: tuple[SiteConfiguration, ...]
    path_weights: Array
    first_displacement_moment_m: Array
    second_displacement_moment_m2: Array
    log_capacity_integral: float
    uses_residence_rate_constant: bool
    residence_rate_constant_s_inv: float


@dataclass(frozen=True)
class PhysicalGeneratorBuildInput:
    records: PhysicalLibraryRecords
    template_configuration: SiteConfiguration
    state_quadratures: tuple[PhysicalStateQuadrature, ...]
    transition_quadratures: tuple[PhysicalTransitionQuadrature, ...]
    memory_coordinate_gradient_functions: tuple[Callable[[SiteConfiguration], Array], ...]
    total_component_concentrations_mol_m3: Array
    temperature_K: float
    volume_m3: float


def build_reduced_generator_specification_from_physical_objects(
    build_input: PhysicalGeneratorBuildInput,
) -> ReducedGeneratorSpecification:
    """Convert physical-library quadrature into deterministic generator functions."""

    configuration_identity_indices = _configuration_identity_indices(
        build_input.state_quadratures,
        build_input.transition_quadratures,
    )
    common_coordinate_count = _common_generator_coordinate_count(
        build_input.state_quadratures,
        build_input.transition_quadratures,
    )
    state_quadratures = tuple(
        _build_reduced_state_quadrature(
            state_quadrature,
            common_coordinate_count,
            configuration_identity_indices,
        )
        for state_quadrature in build_input.state_quadratures
    )
    transition_quadratures = tuple(
        _build_reduced_transition_quadrature(
            transition_quadrature,
            build_input.records,
            common_coordinate_count,
            configuration_identity_indices,
        )
        for transition_quadrature in build_input.transition_quadratures
    )
    point_registry = _generator_point_registry(
        build_input.state_quadratures,
        build_input.transition_quadratures,
        common_coordinate_count,
        configuration_identity_indices,
    )
    physical_object_at_point = _physical_object_function(build_input, point_registry)
    return ReducedGeneratorSpecification(
        potential_energy_J_mol=_physical_potential_function(physical_object_at_point),
        mobility_tensor_m2_s=_physical_mobility_function(physical_object_at_point),
        charge_polarization_gradient=_physical_polarization_gradient_function(
            physical_object_at_point
        ),
        memory_coordinate_gradient=_physical_memory_gradient_function(
            build_input,
            point_registry,
        ),
        state_quadratures=state_quadratures,
        transition_quadratures=transition_quadratures,
        total_component_concentrations_mol_m3=np.asarray(
            build_input.total_component_concentrations_mol_m3,
            dtype=float,
        ),
        temperature_K=build_input.temperature_K,
        volume_m3=build_input.volume_m3,
    )


def flatten_configuration_positions_m(configuration: SiteConfiguration) -> Array:
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    return positions_m.reshape(positions_m.size)


def flatten_configuration_with_local_fields(
    configuration: SiteConfiguration,
    local_fields: PhysicalLocalFields,
) -> Array:
    local_field_values = np.asarray(
        [
            local_fields.dielectric_constant,
            local_fields.viscosity_Pa_s,
            local_fields.ionic_strength_mol_m3,
            local_fields.local_packing_fraction,
        ],
        dtype=float,
    )
    return np.concatenate((flatten_configuration_positions_m(configuration), local_field_values))


def configuration_and_local_fields_from_generator_point(
    template_configuration: SiteConfiguration,
    generator_point: Array,
) -> tuple[SiteConfiguration, PhysicalLocalFields]:
    point = np.asarray(generator_point, dtype=float)
    site_count = len(template_configuration.species_names)
    position_size = site_count * CARTESIAN_DIMENSION
    expected_size = position_size + LOCAL_FIELD_VECTOR_LENGTH
    if point.shape != (expected_size,):
        raise ValueError(f"generator point must have shape ({expected_size},)")
    positions_m = point[:position_size].reshape((site_count, CARTESIAN_DIMENSION))
    local_field_values = point[position_size:]
    local_fields = PhysicalLocalFields(
        dielectric_constant=float(local_field_values[0]),
        viscosity_Pa_s=float(local_field_values[1]),
        ionic_strength_mol_m3=float(local_field_values[2]),
        local_packing_fraction=float(local_field_values[3]),
    )
    validate_local_fields(local_fields, "generator_point.local_fields")
    configuration = SiteConfiguration(
        species_names=template_configuration.species_names,
        molecule_ids=np.asarray(template_configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(template_configuration.site_ids, dtype=int),
        positions_m=positions_m,
        unwrapped_positions_m=positions_m,
        box_lengths_m=np.asarray(template_configuration.box_lengths_m, dtype=float),
    )
    return configuration, local_fields


def validate_site_configuration_family(
    template_configuration: SiteConfiguration,
    configurations: tuple[SiteConfiguration, ...],
) -> None:
    for configuration_index, configuration in enumerate(configurations):
        if configuration.species_names != template_configuration.species_names:
            raise ValueError(f"configuration[{configuration_index}] species_names mismatch")
        if not np.array_equal(configuration.molecule_ids, template_configuration.molecule_ids):
            raise ValueError(f"configuration[{configuration_index}] molecule_ids mismatch")
        if not np.array_equal(configuration.site_ids, template_configuration.site_ids):
            raise ValueError(f"configuration[{configuration_index}] site_ids mismatch")
        if not np.allclose(configuration.box_lengths_m, template_configuration.box_lengths_m):
            raise ValueError(f"configuration[{configuration_index}] box_lengths_m mismatch")


def _all_configurations(
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    transition_quadratures: tuple[PhysicalTransitionQuadrature, ...],
) -> tuple[SiteConfiguration, ...]:
    configurations = []
    for state_quadrature in state_quadratures:
        configurations.extend(state_quadrature.configurations)
    for transition_quadrature in transition_quadratures:
        configurations.extend(transition_quadrature.configurations)
        configurations.extend(transition_quadrature.path_start_configurations)
        configurations.extend(transition_quadrature.path_end_configurations)
    return tuple(configurations)


def _build_reduced_state_quadrature(
    state_quadrature: PhysicalStateQuadrature,
    common_coordinate_count: int,
    configuration_identity_indices: dict[tuple, int],
) -> ReducedStateQuadrature:
    validate_local_field_count(
        state_quadrature.local_fields,
        len(state_quadrature.configurations),
        state_quadrature.label,
    )
    points = np.vstack(
        tuple(
            _generator_point_with_configuration_identity(
                configuration,
                local_fields,
                common_coordinate_count,
                configuration_identity_indices,
            )
            for configuration, local_fields in zip(
                state_quadrature.configurations,
                state_quadrature.local_fields,
                strict=True,
            )
        )
    )
    projector = np.asarray(state_quadrature.self_current_projector, dtype=float)
    physical_coordinate_count = int(projector.shape[0])
    if projector.shape[0] != projector.shape[1]:
        raise ValueError(f"{state_quadrature.label}.self_current_projector must be square")
    if physical_coordinate_count > common_coordinate_count:
        raise ValueError(
            f"{state_quadrature.label}.self_current_projector must have "
            f"at most {common_coordinate_count} coordinates"
        )
    if not np.allclose(projector, np.eye(physical_coordinate_count, dtype=float)):
        raise ValueError(
            f"{state_quadrature.label}.self_current_projector must be identity "
            "in the full-generator path"
        )
    padded_projector = np.eye(common_coordinate_count, dtype=float)
    transport_ownership_bases = tuple(
        _pad_transport_ownership_basis(
            ownership_basis,
            common_coordinate_count,
            int(np.asarray(state_quadrature.configurations[0].positions_m).size),
        )
        for ownership_basis in state_quadrature.transport_ownership_bases
    )
    if len(transport_ownership_bases) != len(state_quadrature.configurations):
        raise ValueError(
            f"{state_quadrature.label}.transport_ownership_bases count must match configurations"
        )
    return ReducedStateQuadrature(
        points=points,
        weights=np.asarray(state_quadrature.weights, dtype=float),
        stoichiometry=np.asarray(state_quadrature.stoichiometry, dtype=float),
        self_current_projector=padded_projector,
        transport_ownership_bases=transport_ownership_bases,
        relative_displacement_fluctuations_m=np.asarray(
            state_quadrature.relative_displacement_fluctuations_m, dtype=float
        ),
        relative_displacement_mobility_m2_s=np.asarray(
            state_quadrature.relative_displacement_mobility_m2_s, dtype=float
        ),
        relative_center_charge_numbers=np.asarray(
            state_quadrature.relative_center_charge_numbers, dtype=float
        ),
    )


def _build_reduced_transition_quadrature(
    transition_quadrature: PhysicalTransitionQuadrature,
    records: PhysicalLibraryRecords,
    common_coordinate_count: int,
    configuration_identity_indices: dict[tuple, int],
) -> ReducedTransitionQuadrature:
    validate_local_field_count(
        transition_quadrature.local_fields,
        len(transition_quadrature.configurations),
        "transition",
    )
    points = np.vstack(
        tuple(
            _generator_point_with_configuration_identity(
                configuration,
                local_fields,
                common_coordinate_count,
                configuration_identity_indices,
            )
            for configuration, local_fields in zip(
                transition_quadrature.configurations,
                transition_quadrature.local_fields,
                strict=True,
            )
        )
    )
    committor_gradients = _extend_position_gradients_to_generator_points(
        transition_quadrature.committor_gradients,
        common_coordinate_count,
    )
    return ReducedTransitionQuadrature(
        from_state_index=transition_quadrature.from_state_index,
        to_state_index=transition_quadrature.to_state_index,
        transition_family=transition_quadrature.transition_family,
        transport_ownership=transition_quadrature.transport_ownership,
        points=points,
        weights=np.asarray(transition_quadrature.weights, dtype=float),
        committor_gradients=committor_gradients,
        surface_state_indices=np.asarray(
            transition_quadrature.surface_state_indices,
            dtype=int,
        ),
        path_displacements_m=_path_displacements_m(
            records,
            transition_quadrature.path_start_configurations,
            transition_quadrature.path_end_configurations,
        ),
        path_weights=np.asarray(transition_quadrature.path_weights, dtype=float),
        first_displacement_moment_m=np.asarray(
            transition_quadrature.first_displacement_moment_m,
            dtype=float,
        ),
        second_displacement_moment_m2=np.asarray(
            transition_quadrature.second_displacement_moment_m2,
            dtype=float,
        ),
        log_capacity_integral=float(transition_quadrature.log_capacity_integral),
        uses_residence_rate_constant=bool(
            transition_quadrature.uses_residence_rate_constant
        ),
        residence_rate_constant_s_inv=float(
            transition_quadrature.residence_rate_constant_s_inv
        ),
    )


def _path_displacements_m(
    records: PhysicalLibraryRecords,
    start_configurations: tuple[SiteConfiguration, ...],
    end_configurations: tuple[SiteConfiguration, ...],
) -> Array:
    if len(start_configurations) != len(end_configurations):
        raise ValueError("path_start_configurations and path_end_configurations mismatch")
    displacements = []
    for start_configuration, end_configuration in zip(start_configurations, end_configurations):
        start_polarization = compute_charge_polarization_m(records, start_configuration)
        end_polarization = compute_charge_polarization_m(records, end_configuration)
        displacements.append(end_polarization - start_polarization)
    return np.asarray(displacements, dtype=float)


def _common_generator_coordinate_count(
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    transition_quadratures: tuple[PhysicalTransitionQuadrature, ...],
) -> int:
    coordinate_counts = tuple(
        flatten_configuration_with_local_fields(configuration, local_fields).size
        for state_quadrature in state_quadratures
        for configuration, local_fields in zip(
            state_quadrature.configurations,
            state_quadrature.local_fields,
            strict=True,
        )
    ) + tuple(
        flatten_configuration_with_local_fields(configuration, local_fields).size
        for transition_quadrature in transition_quadratures
        for configuration, local_fields in zip(
            transition_quadrature.configurations,
            transition_quadrature.local_fields,
            strict=True,
        )
    )
    if not coordinate_counts:
        raise ValueError("physical generator build has no quadrature points")
    return int(max(coordinate_counts)) + 1


def _configuration_identity_indices(
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    transition_quadratures: tuple[PhysicalTransitionQuadrature, ...],
) -> dict[tuple, int]:
    signatures = {
        _configuration_identity_signature(configuration)
        for configuration in _all_configurations(state_quadratures, transition_quadratures)
    }
    return {
        signature: identity_index
        for identity_index, signature in enumerate(sorted(signatures), start=1)
    }


def _configuration_identity_signature(configuration: SiteConfiguration) -> tuple:
    return (
        configuration.species_names,
        tuple(int(value) for value in configuration.molecule_ids),
        tuple(int(value) for value in configuration.site_ids),
    )


def _generator_point_with_configuration_identity(
    configuration: SiteConfiguration,
    local_fields: PhysicalLocalFields,
    common_coordinate_count: int,
    configuration_identity_indices: dict[tuple, int],
) -> Array:
    generator_point = flatten_configuration_with_local_fields(
        configuration,
        local_fields,
    )
    padded_point = _pad_generator_point(generator_point, common_coordinate_count)
    signature = _configuration_identity_signature(configuration)
    if signature not in configuration_identity_indices:
        raise ValueError("configuration identity is not registered")
    padded_point[-1] = float(configuration_identity_indices[signature])
    return padded_point


def _pad_generator_point(generator_point: Array, common_coordinate_count: int) -> Array:
    point = np.asarray(generator_point, dtype=float)
    if point.ndim != 1:
        raise ValueError("generator point must be one-dimensional")
    if point.size > common_coordinate_count:
        raise ValueError("generator point exceeds common coordinate dimension")
    if point.size == common_coordinate_count:
        return point
    padded_point = np.zeros(common_coordinate_count, dtype=float)
    padded_point[: point.size] = point
    return padded_point


def _generator_point_registry(
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    transition_quadratures: tuple[PhysicalTransitionQuadrature, ...],
    common_coordinate_count: int,
    configuration_identity_indices: dict[tuple, int],
) -> dict[tuple[float, ...], tuple[SiteConfiguration, PhysicalLocalFields]]:
    registry: dict[tuple[float, ...], tuple[SiteConfiguration, PhysicalLocalFields]] = {}
    for state_quadrature in state_quadratures:
        _register_configuration_points(
            registry,
            state_quadrature.configurations,
            state_quadrature.local_fields,
            common_coordinate_count,
            configuration_identity_indices,
        )
    for transition_quadrature in transition_quadratures:
        _register_configuration_points(
            registry,
            transition_quadrature.configurations,
            transition_quadrature.local_fields,
            common_coordinate_count,
            configuration_identity_indices,
        )
    return registry


def _register_configuration_points(
    registry: dict[tuple[float, ...], tuple[SiteConfiguration, PhysicalLocalFields]],
    configurations: tuple[SiteConfiguration, ...],
    local_fields: tuple[PhysicalLocalFields, ...],
    common_coordinate_count: int,
    configuration_identity_indices: dict[tuple, int],
) -> None:
    validate_local_field_count(local_fields, len(configurations), "registered_points")
    for configuration, local_fields_at_configuration in zip(
        configurations,
        local_fields,
        strict=True,
    ):
        padded_generator_point = _generator_point_with_configuration_identity(
            configuration,
            local_fields_at_configuration,
            common_coordinate_count,
            configuration_identity_indices,
        )
        cache_key = _generator_point_cache_key(padded_generator_point)
        if cache_key in registry:
            registered_configuration, registered_local_fields = registry[cache_key]
            _validate_equivalent_registered_point(
                registered_configuration,
                registered_local_fields,
                configuration,
                local_fields_at_configuration,
            )
        registry[cache_key] = (
            configuration,
            local_fields_at_configuration,
        )


def _validate_equivalent_registered_point(
    registered_configuration: SiteConfiguration,
    registered_local_fields: PhysicalLocalFields,
    candidate_configuration: SiteConfiguration,
    candidate_local_fields: PhysicalLocalFields,
) -> None:
    same_configuration = (
        _configuration_identity_signature(registered_configuration)
        == _configuration_identity_signature(candidate_configuration)
        and np.array_equal(
            registered_configuration.positions_m,
            candidate_configuration.positions_m,
        )
        and np.array_equal(
            registered_configuration.unwrapped_positions_m,
            candidate_configuration.unwrapped_positions_m,
        )
        and np.array_equal(
            registered_configuration.box_lengths_m,
            candidate_configuration.box_lengths_m,
        )
    )
    if not same_configuration or registered_local_fields != candidate_local_fields:
        raise ValueError("generator point registry collision")


def _generator_point_cache_key(generator_point: Array) -> tuple[float, ...]:
    point = np.asarray(generator_point, dtype=float)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("generator point cache key requires a finite vector")
    return tuple(float(value) for value in point)


def _physical_object_function(
    build_input: PhysicalGeneratorBuildInput,
    point_registry: dict[tuple[float, ...], tuple[SiteConfiguration, PhysicalLocalFields]],
) -> Callable[[Array], PhysicalObjectBundle]:
    physical_object_cache: dict[tuple[float, ...], PhysicalObjectBundle] = {}

    def physical_object_at_point(generator_point: Array) -> PhysicalObjectBundle:
        cache_key = _generator_point_cache_key(generator_point)
        if cache_key in physical_object_cache:
            return physical_object_cache[cache_key]
        if cache_key not in point_registry:
            raise ValueError("generator point is not registered to a physical configuration")
        configuration, local_fields = point_registry[cache_key]
        physical_object = build_physical_objects(
            build_input.records,
            configuration,
            build_input.temperature_K,
            local_fields.dielectric_constant,
            local_fields.viscosity_Pa_s,
            local_fields.ionic_strength_mol_m3,
            local_fields.local_packing_fraction,
        )
        physical_object_cache[cache_key] = physical_object
        return physical_object

    return physical_object_at_point


def _physical_potential_function(
    physical_object_at_point: Callable[[Array], PhysicalObjectBundle],
) -> Callable[[Array], float]:
    def potential_energy_J_mol(generator_point: Array) -> float:
        return physical_object_at_point(generator_point).potential_energy_J_mol

    return potential_energy_J_mol


def _physical_mobility_function(
    physical_object_at_point: Callable[[Array], PhysicalObjectBundle],
) -> Callable[[Array], Array]:
    def mobility_tensor_m2_s(generator_point: Array) -> Array:
        physical_mobility = physical_object_at_point(generator_point).mobility_tensor_m2_s
        return _extend_square_matrix_to_dimension(
            physical_mobility,
            np.asarray(generator_point, dtype=float).size,
        )

    return mobility_tensor_m2_s


def _physical_polarization_gradient_function(
    physical_object_at_point: Callable[[Array], PhysicalObjectBundle],
) -> Callable[[Array], Array]:
    def charge_polarization_gradient(generator_point: Array) -> Array:
        physical_gradient = physical_object_at_point(
            generator_point
        ).charge_polarization_gradient
        return _extend_gradient_to_dimension(
            physical_gradient,
            np.asarray(generator_point, dtype=float).size,
        )

    return charge_polarization_gradient


def _physical_memory_gradient_function(
    build_input: PhysicalGeneratorBuildInput,
    point_registry: dict[tuple[float, ...], tuple[SiteConfiguration, PhysicalLocalFields]],
) -> Callable[[Array], Array]:
    def memory_coordinate_gradient(generator_point: Array) -> Array:
        cache_key = _generator_point_cache_key(generator_point)
        if cache_key not in point_registry:
            raise ValueError("generator point is not registered to a physical configuration")
        configuration, local_fields = point_registry[cache_key]
        generator_dimension = np.asarray(generator_point, dtype=float).size
        if not _configuration_has_role(build_input.records, configuration, "cation"):
            return np.zeros(
                (
                    len(build_input.memory_coordinate_gradient_functions),
                    generator_dimension,
                ),
                dtype=float,
            )
        gradients = []
        for gradient_function in build_input.memory_coordinate_gradient_functions:
            gradients.append(np.asarray(gradient_function(configuration), dtype=float))
        _ = local_fields
        if not gradients:
            return np.empty((0, generator_dimension), dtype=float)
        return np.vstack(
            tuple(
                _extend_gradient_to_dimension(
                    np.atleast_2d(gradient),
                    generator_dimension,
                )
                for gradient in gradients
            )
        )

    return memory_coordinate_gradient


def _configuration_has_role(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: str,
) -> bool:
    return any(
        records.species_records[species_name]["role"] == role
        for species_name in configuration.species_names
    )


def validate_local_field_count(
    local_fields: tuple[PhysicalLocalFields, ...],
    configuration_count: int,
    label: str,
) -> None:
    if len(local_fields) != configuration_count:
        raise ValueError(f"{label}.local_fields length must match configurations")
    for local_field_index, local_fields_at_node in enumerate(local_fields):
        validate_local_fields(local_fields_at_node, f"{label}.local_fields[{local_field_index}]")


def validate_local_fields(local_fields: PhysicalLocalFields, label: str) -> None:
    if local_fields.dielectric_constant <= 0.0:
        raise ValueError(f"{label}.dielectric_constant must be positive")
    if local_fields.viscosity_Pa_s <= 0.0:
        raise ValueError(f"{label}.viscosity_Pa_s must be positive")
    if local_fields.ionic_strength_mol_m3 < 0.0:
        raise ValueError(f"{label}.ionic_strength_mol_m3 must be nonnegative")
    if local_fields.local_packing_fraction < 0.0:
        raise ValueError(f"{label}.local_packing_fraction must be nonnegative")


def _extend_square_matrix_to_dimension(
    physical_matrix: Array,
    target_dimension: int,
) -> Array:
    matrix = np.asarray(physical_matrix, dtype=float)
    if matrix.shape[0] > target_dimension or matrix.shape[1] > target_dimension:
        raise ValueError("physical matrix exceeds target generator dimension")
    extended = np.zeros((target_dimension, target_dimension), dtype=float)
    extended[: matrix.shape[0], : matrix.shape[1]] = matrix
    return extended


def _extend_gradient_to_dimension(
    physical_gradient: Array,
    target_dimension: int,
) -> Array:
    gradient = np.asarray(physical_gradient, dtype=float)
    if gradient.shape[1] > target_dimension:
        raise ValueError("physical gradient exceeds target generator dimension")
    extended = np.zeros((gradient.shape[0], target_dimension), dtype=float)
    extended[:, : gradient.shape[1]] = gradient
    return extended


def _extend_position_gradients_to_generator_points(
    gradients: Array,
    generator_dimension: int,
) -> Array:
    gradient_array = np.asarray(gradients, dtype=float)
    if gradient_array.shape[1] == generator_dimension:
        return gradient_array
    if gradient_array.shape[1] > generator_dimension:
        raise ValueError("committor gradient dimension exceeds generator points")
    extended = np.zeros((gradient_array.shape[0], generator_dimension), dtype=float)
    extended[:, : gradient_array.shape[1]] = gradient_array
    return extended


def _pad_transport_ownership_basis(
    ownership_basis: StateTransportOwnershipBasis,
    generator_dimension: int,
    physical_position_coordinate_count: int,
) -> StateTransportOwnershipBasis:
    ownership_coordinate_count = ownership_basis.bounded_memory_gradients.shape[1]
    allowed_ownership_coordinate_counts = (
        physical_position_coordinate_count,
        physical_position_coordinate_count + LOCAL_FIELD_VECTOR_LENGTH,
    )
    if ownership_coordinate_count not in allowed_ownership_coordinate_counts:
        raise ValueError(
            "transport ownership width must contain positions with optional local fields"
        )
    bounded_memory_gradients = _extend_position_gradients_to_generator_points(
        ownership_basis.bounded_memory_gradients,
        generator_dimension,
    )
    bounded_memory_mode_indices = np.asarray(
        ownership_basis.bounded_memory_mode_indices,
        dtype=int,
    )
    return StateTransportOwnershipBasis(
        transition_displacement_gradients=_extend_position_gradients_to_generator_points(
            ownership_basis.transition_displacement_gradients,
            generator_dimension,
        ),
        transition_edge_indices=np.asarray(
            ownership_basis.transition_edge_indices,
            dtype=int,
        ),
        bounded_memory_gradients=bounded_memory_gradients,
        bounded_memory_mode_indices=bounded_memory_mode_indices,
        diagnostic_gradients=_extend_position_gradients_to_generator_points(
            ownership_basis.diagnostic_gradients,
            generator_dimension,
        ),
        diagnostic_source_ids=tuple(ownership_basis.diagnostic_source_ids),
    )
