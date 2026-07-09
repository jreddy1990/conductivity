"""Memory-coordinate functions and gradients for physical-generator inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from conductivity.physical_library.basin_builder import (
    OrientationBasin,
    assign_orientation_basin,
    compute_role_coordination_number,
)
from conductivity.physical_library.physical_objects import (
    CARTESIAN_DIMENSION,
    SiteConfiguration,
    compute_local_packing_fraction,
)
from conductivity.physical_library.library_io import PhysicalLibraryRecords

Array = np.ndarray

FINITE_DIFFERENCE_STEP_M = 1.0e-12  # Numerical epsilon: site-coordinate gradient step.
CENTRAL_DIFFERENCE_DENOMINATOR = 2.0  # Explicit constant: central difference width is 2h.
FULL_WAVE_FACTOR = 2.0  # Explicit constant: first reciprocal mode is 2*pi/L.
ORIENTATION_MEMORY_VALUES = {
    OrientationBasin.RADIAL: 1.0,
    OrientationBasin.BRIDGING: -1.0,
    OrientationBasin.TANGENTIAL: 0.0,
    OrientationBasin.UNASSIGNED: 0.0,
}


class MemoryCoordinateFamily(Enum):
    CHARGE_DENSITY_COSINE = "charge_density_cosine"
    CHARGE_DENSITY_SINE = "charge_density_sine"
    ATMOSPHERE_POLARIZATION = "atmosphere_polarization"
    CAGE_BACKJUMP = "cage_backjump"
    PARTNER_RESIDENCE = "partner_residence"
    LIGAND_SHELL = "ligand_shell"
    ANION_ORIENTATION = "anion_orientation"
    FREE_VOLUME_STRESS = "free_volume_stress"
    BOUNDED_INTERNAL_POLARIZATION = "bounded_internal_polarization"


@dataclass(frozen=True)
class MemoryCoordinate:
    family: MemoryCoordinateFamily
    wave_vector_m_inv: Array
    value_function: Callable[[SiteConfiguration], float]
    gradient_function: Callable[[SiteConfiguration], Array]


def build_default_memory_coordinates(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
) -> tuple[MemoryCoordinate, ...]:
    """Build default current-coupled memory coordinates from the record set."""

    _ = template_configuration
    coordinates = []
    coordinates.append(
        _analytic_coordinate(
            records,
            MemoryCoordinateFamily.CAGE_BACKJUMP,
            _cage_backjump_value,
            _pair_distance_gradient,
        )
    )
    coordinates.append(
        _analytic_coordinate(
            records,
            MemoryCoordinateFamily.PARTNER_RESIDENCE,
            _partner_residence_value,
            _li_anion_coordination_gradient,
        )
    )
    coordinates.append(
        _analytic_coordinate(
            records,
            MemoryCoordinateFamily.LIGAND_SHELL,
            _ligand_shell_value,
            _li_ligand_coordination_gradient,
        )
    )
    coordinates.append(
        _analytic_coordinate(
            records,
            MemoryCoordinateFamily.ANION_ORIENTATION,
            _anion_orientation_value,
            _zero_memory_gradient,
        )
    )
    coordinates.append(
        _analytic_coordinate(
            records,
            MemoryCoordinateFamily.FREE_VOLUME_STRESS,
            _free_volume_stress_value,
            _zero_memory_gradient,
        )
    )
    coordinates.append(
        _analytic_coordinate(
            records,
            MemoryCoordinateFamily.BOUNDED_INTERNAL_POLARIZATION,
            _bounded_internal_polarization_value,
            _bounded_internal_polarization_gradient,
        )
    )
    return tuple(coordinates)


def combine_memory_values(
    coordinates: tuple[MemoryCoordinate, ...],
    configuration: SiteConfiguration,
) -> Array:
    return np.asarray(
        [coordinate.value_function(configuration) for coordinate in coordinates],
        dtype=float,
    )


def combine_memory_gradients(
    coordinates: tuple[MemoryCoordinate, ...],
    configuration: SiteConfiguration,
) -> Array:
    if not coordinates:
        coordinate_count = len(configuration.species_names) * CARTESIAN_DIMENSION
        return np.zeros((0, coordinate_count), dtype=float)
    return np.vstack(
        tuple(
            np.asarray(coordinate.gradient_function(configuration), dtype=float)
            for coordinate in coordinates
        )
    )


def _charge_density_coordinate(
    records: PhysicalLibraryRecords,
    wave_vector_m_inv: Array,
    family: MemoryCoordinateFamily,
) -> MemoryCoordinate:
    wave_vector = np.asarray(wave_vector_m_inv, dtype=float)

    def value(configuration: SiteConfiguration) -> float:
        phases = np.asarray(configuration.positions_m, dtype=float) @ wave_vector
        charges = _configuration_charges(records, configuration)
        if family == MemoryCoordinateFamily.CHARGE_DENSITY_COSINE:
            return float(np.sum(charges * np.cos(phases)))
        if family == MemoryCoordinateFamily.CHARGE_DENSITY_SINE:
            return float(np.sum(charges * np.sin(phases)))
        raise ValueError(f"unsupported charge-density family {family}")

    def gradient(configuration: SiteConfiguration) -> Array:
        phases = np.asarray(configuration.positions_m, dtype=float) @ wave_vector
        charges = _configuration_charges(records, configuration)
        gradient_row = np.zeros(
            len(configuration.species_names) * CARTESIAN_DIMENSION,
            dtype=float,
        )
        for site_index, charge_number in enumerate(charges):
            start = site_index * CARTESIAN_DIMENSION
            stop = start + CARTESIAN_DIMENSION
            if family == MemoryCoordinateFamily.CHARGE_DENSITY_COSINE:
                gradient_row[start:stop] = (
                    -charge_number * np.sin(phases[site_index]) * wave_vector
                )
            if family == MemoryCoordinateFamily.CHARGE_DENSITY_SINE:
                gradient_row[start:stop] = (
                    charge_number * np.cos(phases[site_index]) * wave_vector
                )
        if (
            family != MemoryCoordinateFamily.CHARGE_DENSITY_COSINE
            and family != MemoryCoordinateFamily.CHARGE_DENSITY_SINE
        ):
            raise ValueError(f"unsupported charge-density family {family}")
        return gradient_row.reshape((1, gradient_row.size))

    return MemoryCoordinate(
        family=family,
        wave_vector_m_inv=wave_vector,
        value_function=value,
        gradient_function=gradient,
    )


def _finite_difference_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    family: MemoryCoordinateFamily,
    scalar_function: Callable[[PhysicalLibraryRecords, SiteConfiguration], float],
) -> MemoryCoordinate:
    def value(configuration: SiteConfiguration) -> float:
        return float(scalar_function(records, configuration))

    def gradient(configuration: SiteConfiguration) -> Array:
        coordinate_count = len(configuration.species_names) * CARTESIAN_DIMENSION
        gradient_row = np.zeros(coordinate_count, dtype=float)
        flat_positions = np.asarray(configuration.positions_m, dtype=float).reshape(
            coordinate_count
        )
        for coordinate_index in range(coordinate_count):
            plus_positions = flat_positions.copy()
            minus_positions = flat_positions.copy()
            plus_positions[coordinate_index] += FINITE_DIFFERENCE_STEP_M
            minus_positions[coordinate_index] -= FINITE_DIFFERENCE_STEP_M
            plus_configuration = _configuration_with_flat_positions(
                template_configuration,
                plus_positions,
            )
            minus_configuration = _configuration_with_flat_positions(
                template_configuration,
                minus_positions,
            )
            gradient_row[coordinate_index] = (
                scalar_function(records, plus_configuration)
                - scalar_function(records, minus_configuration)
            ) / (CENTRAL_DIFFERENCE_DENOMINATOR * FINITE_DIFFERENCE_STEP_M)
        return gradient_row.reshape((1, coordinate_count))

    return MemoryCoordinate(
        family=family,
        wave_vector_m_inv=np.zeros(CARTESIAN_DIMENSION, dtype=float),
        value_function=value,
        gradient_function=gradient,
    )


def _analytic_coordinate(
    records: PhysicalLibraryRecords,
    family: MemoryCoordinateFamily,
    scalar_function: Callable[[PhysicalLibraryRecords, SiteConfiguration], float],
    gradient_function: Callable[[PhysicalLibraryRecords, SiteConfiguration], Array],
) -> MemoryCoordinate:
    def value(configuration: SiteConfiguration) -> float:
        return float(scalar_function(records, configuration))

    def gradient(configuration: SiteConfiguration) -> Array:
        return np.asarray(gradient_function(records, configuration), dtype=float)

    return MemoryCoordinate(
        family=family,
        wave_vector_m_inv=np.zeros(CARTESIAN_DIMENSION, dtype=float),
        value_function=value,
        gradient_function=gradient,
    )


def _ligand_shell_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    return compute_role_coordination_number(
        records,
        configuration,
        center_role="cation",
        ligand_role="additive",
        switch_name="Li_ligand",
    )


def _li_ligand_coordination_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _coordination_switch_gradient(records, configuration, "Li_ligand")


def _atmosphere_polarization_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    charges = _configuration_charges(records, configuration)
    positions = np.asarray(configuration.positions_m, dtype=float)
    charge_center = np.einsum("n,na->a", charges, positions)
    return float(np.linalg.norm(charge_center))


def _cage_backjump_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    lithium_index = _first_role_index(records, configuration, "cation")
    anion_index = _first_role_index(records, configuration, "anion")
    pair_distance_m = float(
        np.linalg.norm(
            np.asarray(configuration.positions_m[anion_index], dtype=float)
            - np.asarray(configuration.positions_m[lithium_index], dtype=float)
        )
    )
    cage_reference_m = float(records.basis_record["pair_basins"]["r_SSIP_m"])
    return pair_distance_m - cage_reference_m


def _partner_residence_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    return compute_role_coordination_number(
        records,
        configuration,
        center_role="cation",
        ligand_role="anion",
        switch_name="Li_anion",
    )


def _li_anion_coordination_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _coordination_switch_gradient(records, configuration, "Li_anion")


def _bounded_internal_polarization_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    lithium_index = _first_role_index(records, configuration, "cation")
    anion_index = _first_role_index(records, configuration, "anion")
    pair_vector = (
        np.asarray(configuration.positions_m[anion_index], dtype=float)
        - np.asarray(configuration.positions_m[lithium_index], dtype=float)
    )
    charges = _configuration_charges(records, configuration)
    return float((charges[lithium_index] - charges[anion_index]) * pair_vector[0])


def _bounded_internal_polarization_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    lithium_index = _first_role_index(records, configuration, "cation")
    anion_index = _first_role_index(records, configuration, "anion")
    charges = _configuration_charges(records, configuration)
    charge_difference = charges[lithium_index] - charges[anion_index]
    gradient = np.zeros(
        (1, len(configuration.species_names) * CARTESIAN_DIMENSION),
        dtype=float,
    )
    gradient[0, lithium_index * CARTESIAN_DIMENSION] = -charge_difference
    gradient[0, anion_index * CARTESIAN_DIMENSION] = charge_difference
    return gradient


def _anion_orientation_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    lithium_index = _first_role_index(records, configuration, "cation")
    anion_index = _first_role_index(records, configuration, "anion")
    orientation = assign_orientation_basin(
        records,
        configuration,
        lithium_index,
        anion_index,
    )
    if orientation not in ORIENTATION_MEMORY_VALUES:
        raise ValueError(f"unsupported orientation basin {orientation}")
    return ORIENTATION_MEMORY_VALUES[orientation]


def _free_volume_stress_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    packing_fraction = compute_local_packing_fraction(records, configuration)
    phi_max = float(records.mixture_record["packing"]["phi_max"])
    if packing_fraction >= phi_max:
        raise ValueError("packing_fraction must be below phi_max")
    return packing_fraction / (phi_max - packing_fraction)


def _zero_memory_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    _ = records
    return np.zeros(
        (1, len(configuration.species_names) * CARTESIAN_DIMENSION),
        dtype=float,
    )


def _coordination_switch_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    switch_name: str,
) -> Array:
    switch_record = records.basis_record["coordination_switches"][switch_name]
    center_role = str(switch_record["center_role"])
    ligand_roles = tuple(str(role) for role in switch_record["ligand_roles"])
    switch_radius_m = float(switch_record["r0_m"])
    exponent = float(switch_record["exponent"])
    if switch_radius_m <= 0.0 or exponent <= 0.0:
        raise ValueError(f"{switch_name} switch radius and exponent must be positive")
    center_index = _first_role_index(records, configuration, center_role)
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    gradient = np.zeros(
        len(configuration.species_names) * CARTESIAN_DIMENSION,
        dtype=float,
    )
    center_gradient = np.zeros(CARTESIAN_DIMENSION, dtype=float)
    for ligand_index, species_name in enumerate(configuration.species_names):
        if ligand_index == center_index:
            continue
        if str(records.species_records[species_name]["role"]) not in ligand_roles:
            continue
        center_to_ligand_m = (
            np.asarray(positions_m[ligand_index], dtype=float)
            - np.asarray(positions_m[center_index], dtype=float)
        )
        distance_m = float(np.linalg.norm(center_to_ligand_m))
        if distance_m <= 0.0:
            raise ValueError("coordination memory distance must be positive")
        reduced_distance = distance_m / switch_radius_m
        denominator = 1.0 + reduced_distance**exponent
        derivative_wrt_distance = (
            -exponent
            * reduced_distance ** (exponent - 1.0)
            / switch_radius_m
            / (denominator * denominator)
        )
        ligand_gradient = derivative_wrt_distance * center_to_ligand_m / distance_m
        ligand_start = ligand_index * CARTESIAN_DIMENSION
        ligand_stop = ligand_start + CARTESIAN_DIMENSION
        gradient[ligand_start:ligand_stop] += ligand_gradient
        center_gradient -= ligand_gradient
    center_start = center_index * CARTESIAN_DIMENSION
    center_stop = center_start + CARTESIAN_DIMENSION
    gradient[center_start:center_stop] += center_gradient
    return gradient.reshape((1, gradient.size))


def _pair_distance_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    lithium_index = _first_role_index(records, configuration, "cation")
    anion_index = _first_role_index(records, configuration, "anion")
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    displacement_m = positions_m[anion_index] - positions_m[lithium_index]
    distance_m = float(np.linalg.norm(displacement_m))
    if distance_m <= 0.0:
        raise ValueError("pair distance memory coordinate must be positive")
    unit_vector = displacement_m / distance_m
    gradient = np.zeros(
        (1, len(configuration.species_names) * CARTESIAN_DIMENSION),
        dtype=float,
    )
    lithium_start = lithium_index * CARTESIAN_DIMENSION
    anion_start = anion_index * CARTESIAN_DIMENSION
    gradient[0, lithium_start : lithium_start + CARTESIAN_DIMENSION] = -unit_vector
    gradient[0, anion_start : anion_start + CARTESIAN_DIMENSION] = unit_vector
    return gradient


def _configuration_charges(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    charges = []
    for species_name, site_id in zip(configuration.species_names, configuration.site_ids):
        species_record = records.species_records[species_name]
        for site_record in species_record["sites"]:
            if int(site_record["site_id"]) == int(site_id):
                charges.append(float(site_record["charge_number"]))
                break
        else:
            raise KeyError(f"{species_name} has no site_id {int(site_id)}")
    return np.asarray(charges, dtype=float)


def _wave_vectors_from_box(box_lengths_m: Array) -> tuple[Array, ...]:
    lengths = np.asarray(box_lengths_m, dtype=float)
    wave_vectors = []
    for cartesian_index in range(CARTESIAN_DIMENSION):
        wave_vector = np.zeros(CARTESIAN_DIMENSION, dtype=float)
        wave_vector[cartesian_index] = FULL_WAVE_FACTOR * np.pi / lengths[cartesian_index]
        wave_vectors.append(wave_vector)
    return tuple(wave_vectors)


def _configuration_with_flat_positions(
    template_configuration: SiteConfiguration,
    flat_positions_m: Array,
) -> SiteConfiguration:
    positions = np.asarray(flat_positions_m, dtype=float).reshape(
        (len(template_configuration.species_names), CARTESIAN_DIMENSION)
    )
    return SiteConfiguration(
        species_names=template_configuration.species_names,
        molecule_ids=np.asarray(template_configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(template_configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=positions,
        box_lengths_m=np.asarray(template_configuration.box_lengths_m, dtype=float),
    )


def _first_role_index(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: str,
) -> int:
    for site_index, species_name in enumerate(configuration.species_names):
        if records.species_records[species_name]["role"] == role:
            return site_index
    raise ValueError(f"configuration has no species with role {role}")
