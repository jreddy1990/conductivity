"""Physical-object builders for the projected conductivity model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from constants import EPS_0, E_CHARGE, F, K_B, N_A, R
from conductivity.physical_library.library_io import PhysicalLibraryRecords
from utils.strict_validation import (
    symmetric_psd_numpy,
    symmetric_psd_pseudoinverse_numpy,
)

Array = np.ndarray
_BONDED_ENERGY_CACHE: dict[tuple, float] = {}

CARTESIAN_DIMENSION = 3
LJ_ATTRACTIVE_EXPONENT = 6  # Explicit constant: Lennard-Jones 12-6 attractive exponent.
LJ_REPULSIVE_EXPONENT_MULTIPLIER = 2  # Repulsive exponent is twice the attractive exponent.
BORN_DENOMINATOR_FACTOR = 8.0  # Explicit constant: Born charging free-energy denominator.
STOKES_SPHERE_DRAG_FACTOR = 6.0  # Explicit constant: Stokes sphere drag prefactor.
RPY_FAR_FIELD_DENOMINATOR = 8.0  # Analytical Oseen/Rotne-Prager denominator.
RPY_OVERLAP_CUBIC_NUMERATOR = 16.0  # Exact coefficient in the unequal-radius RPY partial-overlap tensor.
RPY_OVERLAP_DENOMINATOR = 32.0  # Exact denominator in the unequal-radius RPY partial-overlap tensor.
HARMONIC_PREFRACTOR = 0.5  # Harmonic bonded and packing quadratic prefactor.
UNITY = 1.0
CHARGE_CLOUD_RESPONSE_EXPONENT = (
    UNITY + 2.0 / CARTESIAN_DIMENSION
)  # Analytical screened cloud response exponent 1 + 2/d in d Cartesian dimensions.
ANGLE_COSINE_ROUNDOFF_TOLERANCE = 1.0e-12
ZERO_DISTANCE_TOLERANCE_M = 1.0e-30
CHARGE_NORMALIZATION_ROUNDOFF_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class SiteConfiguration:
    """Site-resolved molecular configuration used by the physical builders."""

    species_names: tuple[str, ...]
    molecule_ids: Array
    site_ids: Array
    positions_m: Array
    unwrapped_positions_m: Array
    box_lengths_m: Array


class PairBasin(Enum):
    CONTACT_ION_PAIR = "CIP"
    SOLVENT_SEPARATED_ION_PAIR = "SSIP"
    FREE = "free"
    TRANSITION = "transition"


@dataclass(frozen=True)
class PhysicalObjectBundle:
    """Computed physical objects at one site-resolved configuration."""

    potential_energy_J_mol: float
    mobility_tensor_m2_s: Array
    charge_polarization_m: Array
    charge_polarization_gradient: Array
    local_packing_fraction: float


@dataclass(frozen=True)
class AtmosphereResistanceDiagnostics:
    atmosphere_resistance_tensor_kg_s: Array
    electrophoretic_resistance_tensor_kg_s: Array
    relaxation_resistance_tensor_kg_s: Array
    cation_diagonal_resistance_trace_kg_s: float
    anion_diagonal_resistance_trace_kg_s: float
    cation_anion_cross_resistance_trace_kg_s: float
    mean_charge_cloud_form_factor: float
    mean_state_geometry_form_factor: float
    minimum_separation_over_debye_length: float
    debye_falkenhagen_time_s: float


@dataclass(frozen=True)
class ResistanceComponentDiagnostics:
    stokes_trace_kg_s: float
    free_volume_trace_kg_s: float
    charge_cloud_trace_kg_s: float
    atmosphere_trace_kg_s: float
    total_trace_kg_s: float


def build_physical_objects(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    temperature_K: float,
    dielectric_constant: float,
    viscosity_Pa_s: float,
    ionic_strength_mol_m3: float,
    local_packing_fraction: float,
) -> PhysicalObjectBundle:
    """Compute U, D, P, gradient(P), and local packing from site records."""

    validate_site_configuration(configuration)
    bonded_energy_J_mol = compute_bonded_energy_J_mol(records, configuration)
    lj_energy_J_mol = compute_lj_energy_J_mol(records, configuration)
    coulomb_energy_J_mol = compute_coulomb_energy_J_mol(
        records,
        configuration,
        dielectric_constant,
    )
    born_energy_J_mol = compute_born_energy_J_mol(
        records,
        configuration,
        dielectric_constant,
    )
    packing_energy_J_mol = compute_packing_energy_J_mol(
        records,
        temperature_K,
        local_packing_fraction,
    )
    coordination_energy_J_mol = compute_coordination_energy_J_mol(records, configuration)
    solvation_energy_J_mol = compute_solvation_competition_energy_J_mol(
        records,
        configuration,
    )
    activity_energy_J_mol = compute_activity_energy_J_mol(
        records,
        configuration,
        temperature_K,
        ionic_strength_mol_m3,
    )
    return PhysicalObjectBundle(
        potential_energy_J_mol=(
            bonded_energy_J_mol
            + lj_energy_J_mol
            + coulomb_energy_J_mol
            + born_energy_J_mol
            + packing_energy_J_mol
            + coordination_energy_J_mol
            + solvation_energy_J_mol
            + activity_energy_J_mol
        ),
        mobility_tensor_m2_s=compute_mobility_tensor_m2_s(
            records,
            configuration,
            temperature_K,
            viscosity_Pa_s,
            dielectric_constant,
            ionic_strength_mol_m3,
            local_packing_fraction,
        ),
        charge_polarization_m=compute_charge_polarization_m(records, configuration),
        charge_polarization_gradient=compute_charge_polarization_gradient(
            records,
            configuration,
        ),
        local_packing_fraction=local_packing_fraction,
    )


def validate_site_configuration(configuration: SiteConfiguration) -> None:
    site_count = len(configuration.species_names)
    molecule_ids = np.asarray(configuration.molecule_ids, dtype=int)
    site_ids = np.asarray(configuration.site_ids, dtype=int)
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    unwrapped_positions_m = np.asarray(configuration.unwrapped_positions_m, dtype=float)
    box_lengths_m = np.asarray(configuration.box_lengths_m, dtype=float)
    if molecule_ids.shape != (site_count,):
        raise ValueError("molecule_ids length must match species_names")
    if site_ids.shape != (site_count,):
        raise ValueError("site_ids length must match species_names")
    if positions_m.shape != (site_count, CARTESIAN_DIMENSION):
        raise ValueError("positions_m must have shape (site_count, 3)")
    if unwrapped_positions_m.shape != (site_count, CARTESIAN_DIMENSION):
        raise ValueError("unwrapped_positions_m must have shape (site_count, 3)")
    if box_lengths_m.shape != (CARTESIAN_DIMENSION,):
        raise ValueError("box_lengths_m must have shape (3,)")
    if not np.all(np.isfinite(positions_m)):
        raise ValueError("positions_m must be finite")
    if not np.all(np.isfinite(unwrapped_positions_m)):
        raise ValueError("unwrapped_positions_m must be finite")
    if not np.all(np.isfinite(box_lengths_m)) or np.any(box_lengths_m <= 0.0):
        raise ValueError("box_lengths_m must be finite and positive")


def compute_bonded_energy_J_mol(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    cache_key = _bonded_energy_cache_key(configuration)
    if cache_key in _BONDED_ENERGY_CACHE:
        return _BONDED_ENERGY_CACHE[cache_key]
    lookup = _configuration_site_lookup(configuration)
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    total_energy_J_mol = 0.0
    for species_name, species_record in records.species_records.items():
        molecule_ids = _molecule_ids_for_species(configuration, species_name)
        for molecule_id in molecule_ids:
            for bond_record in species_record["bonds"]:
                first_index = lookup[(species_name, molecule_id, int(bond_record["site_i"]))]
                second_index = lookup[
                    (species_name, molecule_id, int(bond_record["site_j"]))
                ]
                distance_m = _minimum_image_distance_m(
                    positions_m[first_index],
                    positions_m[second_index],
                    configuration.box_lengths_m,
                )
                displacement_m = distance_m - float(bond_record["r0_m"])
                total_energy_J_mol += (
                    HARMONIC_PREFRACTOR
                    * float(bond_record["k_J_m2_mol"])
                    * displacement_m
                    * displacement_m
                )
            for angle_record in species_record["angles"]:
                angle_rad = _angle_rad(
                    positions_m[
                        lookup[(species_name, molecule_id, int(angle_record["site_i"]))]
                    ],
                    positions_m[
                        lookup[(species_name, molecule_id, int(angle_record["site_j"]))]
                    ],
                    positions_m[
                        lookup[(species_name, molecule_id, int(angle_record["site_k"]))]
                    ],
                    configuration.box_lengths_m,
                )
                displacement_rad = angle_rad - float(angle_record["theta0_rad"])
                total_energy_J_mol += (
                    HARMONIC_PREFRACTOR
                    * float(angle_record["k_J_rad2_mol"])
                    * displacement_rad
                    * displacement_rad
                )
            for torsion_record in species_record["torsions"]:
                torsion_rad = _torsion_rad(
                    positions_m[
                        lookup[(species_name, molecule_id, int(torsion_record["site_i"]))]
                    ],
                    positions_m[
                        lookup[(species_name, molecule_id, int(torsion_record["site_j"]))]
                    ],
                    positions_m[
                        lookup[(species_name, molecule_id, int(torsion_record["site_k"]))]
                    ],
                    positions_m[
                        lookup[(species_name, molecule_id, int(torsion_record["site_l"]))]
                    ],
                    configuration.box_lengths_m,
                )
                for torsion_term in torsion_record["terms"]:
                    idivf = float(torsion_term["idivf"])
                    if idivf <= 0.0:
                        raise ValueError("torsion idivf must be positive")
                    total_energy_J_mol += (
                        float(torsion_term["Vn_J_mol"])
                        / idivf
                        * (
                            UNITY
                            + math.cos(
                                int(torsion_term["periodicity"]) * torsion_rad
                                - float(torsion_term["phase_rad"])
                            )
                        )
                    )
    _BONDED_ENERGY_CACHE[cache_key] = total_energy_J_mol
    return total_energy_J_mol


def _bonded_energy_cache_key(configuration: SiteConfiguration) -> tuple:
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    molecule_keys = []
    for species_name in sorted(set(configuration.species_names)):
        molecule_ids = _molecule_ids_for_species(configuration, species_name)
        for molecule_id in molecule_ids:
            site_indices = tuple(
                site_index
                for site_index, current_species_name in enumerate(
                    configuration.species_names
                )
                if current_species_name == species_name
                and int(configuration.molecule_ids[site_index]) == molecule_id
            )
            if not site_indices:
                raise ValueError("molecule has no sites")
            reference_position_m = positions_m[site_indices[0]]
            relative_positions_m = positions_m[list(site_indices)] - reference_position_m
            molecule_keys.append(
                (
                    species_name,
                    tuple(int(configuration.site_ids[site_index]) for site_index in site_indices),
                    tuple(np.round(relative_positions_m.reshape(-1), 18)),
                )
            )
    return tuple(molecule_keys)


def compute_coordination_energy_J_mol(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    coordination_record = records.basis_record["free_energy_terms"]["coordination_J_mol"]
    total_energy_J_mol = 0.0
    for switch_name, coefficient_J_mol in coordination_record.items():
        coordination_coefficient_J_mol = float(coefficient_J_mol)
        if switch_name == "Li_anion":
            coordination_coefficient_J_mol *= (
                li_anion_feature_coordination_energy_multiplier(records, configuration)
            )
        total_energy_J_mol += coordination_coefficient_J_mol * _coordination_number(
            records,
            configuration,
            switch_name,
        )
    return total_energy_J_mol


def li_anion_feature_coordination_energy_multiplier(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    anion_species_names = tuple(
        species_name
        for species_name in dict.fromkeys(configuration.species_names)
        if str(records.species_records[species_name]["role"]) == "anion"
    )
    if not anion_species_names:
        return UNITY
    feature_factors = tuple(
        anion_internal_charge_separation_factor(records, species_name)
        for species_name in anion_species_names
    )
    return UNITY + float(np.max(np.asarray(feature_factors, dtype=float)))


def anion_internal_charge_separation_factor(
    records: PhysicalLibraryRecords,
    species_name: str,
) -> float:
    species_record = records.species_records[species_name]
    if str(species_record["role"]) != "anion":
        return 0.0
    formal_charge_number = abs(float(species_record["formal_charge_e"]))
    if formal_charge_number == 0.0:
        return 0.0
    absolute_partial_charge_sum = _species_absolute_partial_charge_sum(species_record)
    if (
        absolute_partial_charge_sum + CHARGE_NORMALIZATION_ROUNDOFF_TOLERANCE
        < formal_charge_number
    ):
        raise ValueError(
            "anion absolute partial charge sum is smaller than formal charge magnitude"
        )
    normalized_partial_charge_sum = max(absolute_partial_charge_sum, formal_charge_number)
    return (
        normalized_partial_charge_sum - formal_charge_number
    ) / normalized_partial_charge_sum


def compute_solvation_competition_energy_J_mol(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    competition_record = records.basis_record["free_energy_terms"]["solvation_competition"]
    total_energy_J_mol = 0.0
    for switch_name, target_value in competition_record["targets"].items():
        coordination_number = _coordination_number(records, configuration, switch_name)
        displacement = coordination_number - float(target_value)
        stiffness_J_mol = float(competition_record["stiffness_J_mol"][switch_name])
        total_energy_J_mol += HARMONIC_PREFRACTOR * stiffness_J_mol * displacement * displacement
    return total_energy_J_mol


def compute_activity_energy_J_mol(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    temperature_K: float,
    ionic_strength_mol_m3: float,
) -> float:
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if ionic_strength_mol_m3 < 0.0:
        raise ValueError("ionic_strength_mol_m3 must be nonnegative")
    activity_record = records.basis_record["free_energy_terms"]["activity"]
    sqrt_ionic_strength = math.sqrt(ionic_strength_mol_m3)
    debye_a = float(activity_record["debye_huckel_A_sqrt_m3_mol"])
    debye_b = float(activity_record["debye_huckel_B_sqrt_m3_mol_per_m"])
    total_energy_J_mol = 0.0
    molecule_keys = set()
    for site_index, species_name in enumerate(configuration.species_names):
        molecule_key = (species_name, int(configuration.molecule_ids[site_index]))
        if molecule_key in molecule_keys:
            continue
        molecule_keys.add(molecule_key)
        species_record = records.species_records[species_name]
        charge_number = float(species_record["formal_charge_e"])
        if charge_number == 0.0:
            continue
        hard_radius_m = _species_mean_steric_radius_m(species_record)
        denominator = UNITY + debye_b * hard_radius_m * sqrt_ionic_strength
        log_gamma = -debye_a * charge_number * charge_number * sqrt_ionic_strength / denominator
        total_energy_J_mol += R * temperature_K * log_gamma
    return total_energy_J_mol


def compute_lj_energy_J_mol(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    site_records = _configuration_site_records(records, configuration)
    displacements_m = positions_m[None, :, :] - positions_m[:, None, :]
    box_lengths_m = np.asarray(configuration.box_lengths_m, dtype=float)
    displacements_m = displacements_m - box_lengths_m * np.rint(
        displacements_m / box_lengths_m
    )
    distances_m = np.linalg.norm(displacements_m, axis=2)
    sigma_values_m = np.asarray(
        [float(site_record["lj_sigma_m"]) for site_record in site_records],
        dtype=float,
    )
    epsilon_values_J = np.asarray(
        [float(site_record["lj_epsilon_J"]) for site_record in site_records],
        dtype=float,
    )
    sigma_matrix_m = HARMONIC_PREFRACTOR * (
        sigma_values_m[:, None] + sigma_values_m[None, :]
    )
    epsilon_matrix_J = np.sqrt(epsilon_values_J[:, None] * epsilon_values_J[None, :])
    molecule_ids = np.asarray(configuration.molecule_ids, dtype=int)
    species_names = np.asarray(configuration.species_names, dtype=str)
    same_molecule = (
        (molecule_ids[:, None] == molecule_ids[None, :])
        & (species_names[:, None] == species_names[None, :])
    )
    upper_mask = np.triu(np.ones(distances_m.shape, dtype=bool), k=1)
    pair_mask = upper_mask & ~same_molecule
    if np.any(distances_m[pair_mask] <= ZERO_DISTANCE_TOLERANCE_M):
        raise ValueError("LJ site distance is zero")
    sigma_over_distance = sigma_matrix_m[pair_mask] / distances_m[pair_mask]
    attractive_term = sigma_over_distance**LJ_ATTRACTIVE_EXPONENT
    repulsive_term = attractive_term**LJ_REPULSIVE_EXPONENT_MULTIPLIER
    return float(
        N_A
        * 4.0
        * np.sum(epsilon_matrix_J[pair_mask] * (repulsive_term - attractive_term))
    )


def compute_coulomb_energy_J_mol(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    dielectric_constant: float,
) -> float:
    if dielectric_constant <= 0.0:
        raise ValueError("dielectric_constant must be positive")
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    site_records = _configuration_site_records(records, configuration)
    prefactor_J_m_mol = N_A * E_CHARGE * E_CHARGE / (
        4.0 * math.pi * EPS_0 * dielectric_constant
    )
    charged_site_indices = tuple(
        site_index
        for site_index, site_record in enumerate(site_records)
        if float(site_record["charge_number"]) != 0.0
    )
    if not charged_site_indices:
        return 0.0
    charged_index_array = np.asarray(charged_site_indices, dtype=int)
    charged_positions_m = positions_m[charged_index_array]
    displacements_m = charged_positions_m[None, :, :] - charged_positions_m[:, None, :]
    box_lengths_m = np.asarray(configuration.box_lengths_m, dtype=float)
    displacements_m = displacements_m - box_lengths_m * np.rint(
        displacements_m / box_lengths_m
    )
    distances_m = np.linalg.norm(displacements_m, axis=2)
    charges = np.asarray(
        [float(site_records[site_index]["charge_number"]) for site_index in charged_site_indices],
        dtype=float,
    )
    same_molecule = (
        np.asarray(configuration.molecule_ids, dtype=int)[charged_index_array][:, None]
        == np.asarray(configuration.molecule_ids, dtype=int)[charged_index_array][None, :]
    )
    upper_mask = np.triu(np.ones(distances_m.shape, dtype=bool), k=1)
    pair_mask = upper_mask & ~same_molecule
    if np.any(distances_m[pair_mask] <= ZERO_DISTANCE_TOLERANCE_M):
        raise ValueError("Coulomb site distance is zero")
    charge_products = charges[:, None] * charges[None, :]
    return float(
        prefactor_J_m_mol
        * np.sum(charge_products[pair_mask] / distances_m[pair_mask])
    )


def compute_born_energy_J_mol(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    dielectric_constant: float,
) -> float:
    if dielectric_constant <= 0.0:
        raise ValueError("dielectric_constant must be positive")
    total_energy_J_mol = 0.0
    dielectric_factor = UNITY - UNITY / dielectric_constant
    site_records = _configuration_site_records(records, configuration)
    for site_index in range(len(configuration.species_names)):
        site_record = site_records[site_index]
        charge_number = float(site_record["charge_number"])
        born_radius_m = float(site_record["born_radius_m"])
        if born_radius_m <= 0.0:
            raise ValueError("born_radius_m must be positive")
        total_energy_J_mol -= (
            N_A
            * E_CHARGE
            * E_CHARGE
            * charge_number
            * charge_number
            * dielectric_factor
            / (BORN_DENOMINATOR_FACTOR * math.pi * EPS_0 * born_radius_m)
        )
    return total_energy_J_mol


def compute_local_packing_fraction(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    cell_volume_m3 = _cell_volume_m3(configuration.box_lengths_m)
    occupied_volume_m3 = 0.0
    for site_index in range(len(configuration.species_names)):
        occupied_volume_m3 += float(_site_record(records, configuration, site_index)["volume_m3"])
    return occupied_volume_m3 / cell_volume_m3


def compute_packing_energy_J_mol(
    records: PhysicalLibraryRecords,
    temperature_K: float,
    local_packing_fraction: float,
) -> float:
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    packing_record = records.mixture_record["packing"]
    phi_max = float(packing_record["phi_max"])
    if phi_max <= 0.0 or phi_max >= UNITY:
        raise ValueError("mixture.packing.phi_max must be between 0 and 1")
    if local_packing_fraction < 0.0:
        raise ValueError("local_packing_fraction must be nonnegative")
    if local_packing_fraction >= phi_max:
        return math.inf
    denominator = (UNITY - local_packing_fraction) * (UNITY - local_packing_fraction)
    return (
        R
        * temperature_K
        * local_packing_fraction
        * (4.0 - 3.0 * local_packing_fraction)
        / denominator
    )


def compute_stokes_mobility_tensor_m2_s(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> Array:
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if viscosity_Pa_s <= 0.0:
        raise ValueError("viscosity_Pa_s must be positive")
    return (
        K_B
        * temperature_K
        * _rpy_hydrodynamic_mobility_kg_inv_s(
            records,
            configuration,
            viscosity_Pa_s,
        )
    )


def _rpy_hydrodynamic_mobility_kg_inv_s(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    viscosity_Pa_s: float,
) -> Array:
    site_records = _configuration_site_records(records, configuration)
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    box_lengths_m = np.asarray(configuration.box_lengths_m, dtype=float)
    site_count = len(site_records)
    mobility = np.zeros(
        (CARTESIAN_DIMENSION * site_count, CARTESIAN_DIMENSION * site_count),
        dtype=float,
    )
    identity = np.eye(CARTESIAN_DIMENSION, dtype=float)
    radii_m = np.asarray(
        [
            _positive_site_hydrodynamic_radius_m(site_record)
            for site_record in site_records
        ],
        dtype=float,
    )
    for first_site_index, _first_site_record in enumerate(site_records):
        first_radius_m = radii_m[first_site_index]
        first_self_mobility = 1.0 / (
            STOKES_SPHERE_DRAG_FACTOR
            * math.pi
            * viscosity_Pa_s
            * first_radius_m
        )
        first_slice = _cartesian_site_slice(first_site_index)
        mobility[first_slice, first_slice] = first_self_mobility * identity
        for second_site_index in range(first_site_index + 1, site_count):
            second_radius_m = radii_m[second_site_index]
            displacement_m = positions_m[second_site_index] - positions_m[first_site_index]
            displacement_m -= box_lengths_m * np.rint(displacement_m / box_lengths_m)
            cross_mobility = _rpy_cross_mobility_block_kg_inv_s(
                displacement_m,
                first_radius_m,
                second_radius_m,
                viscosity_Pa_s,
            )
            second_slice = _cartesian_site_slice(second_site_index)
            mobility[first_slice, second_slice] = cross_mobility
            mobility[second_slice, first_slice] = cross_mobility.T
    symmetric_mobility = 0.5 * (mobility + mobility.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_mobility)
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -np.sqrt(np.finfo(float).eps) * eigenvalue_scale:
        raise ValueError("RPY hydrodynamic mobility is not positive semidefinite")
    return symmetric_mobility


def _rpy_cross_mobility_block_kg_inv_s(
    displacement_m: Array,
    first_radius_m: float,
    second_radius_m: float,
    viscosity_Pa_s: float,
) -> Array:
    separation_m = float(np.linalg.norm(displacement_m))
    radius_difference_m = abs(first_radius_m - second_radius_m)
    if separation_m <= radius_difference_m:
        return np.eye(CARTESIAN_DIMENSION, dtype=float) / (
            STOKES_SPHERE_DRAG_FACTOR
            * math.pi
            * viscosity_Pa_s
            * max(first_radius_m, second_radius_m)
        )
    direction = np.asarray(displacement_m, dtype=float) / separation_m
    direction_outer = np.outer(direction, direction)
    identity = np.eye(CARTESIAN_DIMENSION, dtype=float)
    contact_distance_m = first_radius_m + second_radius_m
    if separation_m >= contact_distance_m:
        radius_square_sum_m2 = first_radius_m**2 + second_radius_m**2
        finite_size_correction = radius_square_sum_m2 / (3.0 * separation_m**2)
        return (
            identity
            + direction_outer
            + finite_size_correction * (identity - 3.0 * direction_outer)
        ) / (RPY_FAR_FIELD_DENOMINATOR * math.pi * viscosity_Pa_s * separation_m)
    radius_difference_squared_m2 = (first_radius_m - second_radius_m) ** 2
    isotropic_numerator_m4 = (
        RPY_OVERLAP_CUBIC_NUMERATOR
        * separation_m**3
        * contact_distance_m
        - (radius_difference_squared_m2 + 3.0 * separation_m**2) ** 2
    )
    directional_numerator_m4 = (
        3.0 * (radius_difference_squared_m2 - separation_m**2) ** 2
    )
    overlap_denominator_m3 = RPY_OVERLAP_DENOMINATOR * separation_m**3
    isotropic_length_m = isotropic_numerator_m4 / overlap_denominator_m3
    directional_length_m = directional_numerator_m4 / overlap_denominator_m3
    return (
        isotropic_length_m * identity + directional_length_m * direction_outer
    ) / (
        STOKES_SPHERE_DRAG_FACTOR
        * math.pi
        * viscosity_Pa_s
        * first_radius_m
        * second_radius_m
    )


def _positive_site_hydrodynamic_radius_m(site_record: dict) -> float:
    hydrodynamic_radius_m = float(site_record["hydrodynamic_radius_m"])
    if hydrodynamic_radius_m <= 0.0:
        raise ValueError("hydrodynamic_radius_m must be positive")
    return hydrodynamic_radius_m


def _cartesian_site_slice(site_index: int) -> slice:
    start = CARTESIAN_DIMENSION * site_index
    return slice(start, start + CARTESIAN_DIMENSION)


def compute_mobility_tensor_m2_s(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    temperature_K: float,
    viscosity_Pa_s: float,
    dielectric_constant: float,
    ionic_strength_mol_m3: float,
    local_packing_fraction: float,
) -> Array:
    resistance = compute_resistance_tensor_kg_s(
        records,
        configuration,
        viscosity_Pa_s,
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
        local_packing_fraction,
    )
    constrained_mobility = _rigid_constrained_mobility_kg_inv_s(
        records,
        configuration,
        resistance,
    )
    return K_B * temperature_K * constrained_mobility


def _rigid_constrained_mobility_kg_inv_s(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    resistance_tensor_kg_s: Array,
) -> Array:
    resistance = np.asarray(resistance_tensor_kg_s, dtype=float)
    coordinate_count = CARTESIAN_DIMENSION * len(configuration.species_names)
    if resistance.shape != (coordinate_count, coordinate_count):
        raise ValueError("rigid resistance tensor shape does not match configuration")
    kinematic_map = _rigid_body_kinematic_map(records, configuration)
    generalized_resistance = 0.5 * (
        kinematic_map.T @ resistance @ kinematic_map
        + (kinematic_map.T @ resistance @ kinematic_map).T
    )
    generalized_mobility = symmetric_psd_pseudoinverse_numpy(
        generalized_resistance,
        "rigid generalized resistance",
    )
    constrained_mobility = kinematic_map @ generalized_mobility @ kinematic_map.T
    constrained_mobility = 0.5 * (constrained_mobility + constrained_mobility.T)
    eigenvalues = np.linalg.eigvalsh(constrained_mobility)
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -np.sqrt(np.finfo(float).eps) * eigenvalue_scale:
        raise ValueError("rigid constrained mobility is not positive semidefinite")
    return constrained_mobility


def _rigid_body_kinematic_map(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    molecule_keys = _configuration_molecule_keys(configuration)
    site_count = len(configuration.species_names)
    rigid_geometries = tuple(
        _rigid_molecule_geometry(
            records,
            configuration,
            species_name,
            molecule_id,
        )
        for species_name, molecule_id in molecule_keys
    )
    generalized_coordinate_count = sum(
        CARTESIAN_DIMENSION + active_rotation_axes.shape[1]
        for _site_indices, _center_m, _scale_m, active_rotation_axes in rigid_geometries
    )
    kinematic_map = np.zeros(
        (CARTESIAN_DIMENSION * site_count, generalized_coordinate_count),
        dtype=float,
    )
    unwrapped_positions_m = np.asarray(configuration.unwrapped_positions_m, dtype=float)
    generalized_offset = 0
    for site_indices, hydrodynamic_center_m, rotational_length_scale_m, active_axes in (
        rigid_geometries
    ):
        translation_slice = slice(
            generalized_offset,
            generalized_offset + CARTESIAN_DIMENSION,
        )
        rotation_slice = slice(
            translation_slice.stop,
            translation_slice.stop + active_axes.shape[1],
        )
        for site_index in site_indices:
            site_slice = _cartesian_site_slice(site_index)
            relative_position_m = (
                unwrapped_positions_m[site_index] - hydrodynamic_center_m
            )
            kinematic_map[site_slice, translation_slice] = np.eye(
                CARTESIAN_DIMENSION,
                dtype=float,
            )
            if active_axes.shape[1] > 0:
                kinematic_map[site_slice, rotation_slice] = -_cross_product_matrix(
                    relative_position_m / rotational_length_scale_m
                ) @ active_axes
        generalized_offset = rotation_slice.stop
    if np.linalg.matrix_rank(kinematic_map) != generalized_coordinate_count:
        raise ValueError("RIGID_KINEMATIC_RANK_INVALID")
    return kinematic_map


def _rigid_molecule_geometry(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    species_name: str,
    molecule_id: int,
) -> tuple[tuple[int, ...], Array, float, Array]:
    site_indices, _mass_fractions = molecule_site_indices_and_mass_fractions(
        records,
        configuration,
        species_name,
        molecule_id,
    )
    unwrapped_positions_m = np.asarray(configuration.unwrapped_positions_m, dtype=float)
    hydrodynamic_radii_m = np.asarray(
        [
            _positive_site_hydrodynamic_radius_m(
                _site_record(records, configuration, site_index)
            )
            for site_index in site_indices
        ],
        dtype=float,
    )
    normalized_radius_weights = hydrodynamic_radii_m / float(
        np.sum(hydrodynamic_radii_m)
    )
    molecule_positions_m = unwrapped_positions_m[np.asarray(site_indices, dtype=int)]
    hydrodynamic_center_m = np.sum(
        normalized_radius_weights[:, None] * molecule_positions_m,
        axis=0,
    )
    relative_positions_m = molecule_positions_m - hydrodynamic_center_m
    weighted_gyration_tensor_m2 = np.einsum(
        "i,ij,ik->jk",
        normalized_radius_weights,
        relative_positions_m,
        relative_positions_m,
    )
    rotational_resistance_geometry_m2 = (
        np.trace(weighted_gyration_tensor_m2)
        * np.eye(CARTESIAN_DIMENSION, dtype=float)
        - weighted_gyration_tensor_m2
    )
    rotational_eigenvalues_m2, rotational_eigenvectors = np.linalg.eigh(
        rotational_resistance_geometry_m2
    )
    rotational_scale_m2 = max(
        float(np.max(rotational_eigenvalues_m2)),
        np.finfo(float).tiny,
    )
    active_rotation_mask = rotational_eigenvalues_m2 > (
        np.sqrt(np.finfo(float).eps) * rotational_scale_m2
    )
    active_rotation_axes = rotational_eigenvectors[:, active_rotation_mask]
    if active_rotation_axes.shape[1] == 0:
        return site_indices, hydrodynamic_center_m, 1.0, active_rotation_axes
    rotational_length_scale_m = float(
        np.sqrt(np.trace(weighted_gyration_tensor_m2))
    )
    if not math.isfinite(rotational_length_scale_m) or rotational_length_scale_m <= 0.0:
        raise ValueError("RIGID_CLUSTER_GEOMETRY_INVALID")
    return (
        site_indices,
        hydrodynamic_center_m,
        rotational_length_scale_m,
        active_rotation_axes,
    )


def _cross_product_matrix(vector: Array) -> Array:
    vector_3d = np.asarray(vector, dtype=float)
    if vector_3d.shape != (CARTESIAN_DIMENSION,):
        raise ValueError("cross-product matrix requires a three-vector")
    first, second, third = vector_3d
    return np.asarray(
        [
            [0.0, -third, second],
            [third, 0.0, -first],
            [-second, first, 0.0],
        ],
        dtype=float,
    )


def compute_resistance_tensor_kg_s(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    viscosity_Pa_s: float,
    dielectric_constant: float,
    ionic_strength_mol_m3: float,
    temperature_K: float,
    local_packing_fraction: float,
) -> Array:
    if viscosity_Pa_s <= 0.0:
        raise ValueError("viscosity_Pa_s must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    site_count = len(configuration.species_names)
    hydrodynamic_mobility = _rpy_hydrodynamic_mobility_kg_inv_s(
        records,
        configuration,
        viscosity_Pa_s,
    )
    resistance = symmetric_psd_pseudoinverse_numpy(
        hydrodynamic_mobility,
        "RPY hydrodynamic mobility",
    )
    kappa_m_inv = _debye_kappa_m_inv(
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
    )
    site_records = _configuration_site_records(records, configuration)
    for site_index in range(site_count):
        site_record = site_records[site_index]
        stokes_drag_kg_s = _stokes_drag_kg_s(site_record, viscosity_Pa_s)
        additional_drag_kg_s = _free_volume_drag_kg_s(
            records,
            stokes_drag_kg_s,
            local_packing_fraction,
        )
        additional_drag_kg_s += _charge_cloud_drag_kg_s(
            float(site_record["hydrodynamic_radius_m"]),
            float(site_record["charge_cloud_radius_m"]),
            stokes_drag_kg_s,
            _effective_charge_cloud_number(records, configuration, site_index),
            kappa_m_inv,
            dielectric_constant,
            temperature_K,
            _short_range_wavevector_fraction(
                kappa_m_inv,
                float(site_record["charge_cloud_radius_m"]),
            ),
        )
        start = CARTESIAN_DIMENSION * site_index
        stop = start + CARTESIAN_DIMENSION
        resistance[start:stop, start:stop] += additional_drag_kg_s * np.eye(
            CARTESIAN_DIMENSION,
            dtype=float,
        )
    atmosphere_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        configuration,
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
        viscosity_Pa_s,
    )
    resistance += atmosphere_diagnostics.atmosphere_resistance_tensor_kg_s
    return resistance


def compute_resistance_component_diagnostics(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    viscosity_Pa_s: float,
    dielectric_constant: float,
    ionic_strength_mol_m3: float,
    temperature_K: float,
    local_packing_fraction: float,
) -> ResistanceComponentDiagnostics:
    if viscosity_Pa_s <= 0.0:
        raise ValueError("viscosity_Pa_s must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    kappa_m_inv = _debye_kappa_m_inv(
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
    )
    stokes_trace_kg_s = 0.0
    free_volume_trace_kg_s = 0.0
    charge_cloud_trace_kg_s = 0.0
    site_records = _configuration_site_records(records, configuration)
    for site_index, site_record in enumerate(site_records):
        stokes_drag_kg_s = _stokes_drag_kg_s(site_record, viscosity_Pa_s)
        stokes_trace_kg_s += CARTESIAN_DIMENSION * stokes_drag_kg_s
        free_volume_trace_kg_s += CARTESIAN_DIMENSION * _free_volume_drag_kg_s(
            records,
            stokes_drag_kg_s,
            local_packing_fraction,
        )
        charge_cloud_trace_kg_s += CARTESIAN_DIMENSION * _charge_cloud_drag_kg_s(
            float(site_record["hydrodynamic_radius_m"]),
            float(site_record["charge_cloud_radius_m"]),
            stokes_drag_kg_s,
            _effective_charge_cloud_number(records, configuration, site_index),
            kappa_m_inv,
            dielectric_constant,
            temperature_K,
            _short_range_wavevector_fraction(
                kappa_m_inv,
                float(site_record["charge_cloud_radius_m"]),
            ),
        )
    atmosphere_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        configuration,
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
        viscosity_Pa_s,
    )
    atmosphere_trace_kg_s = float(
        np.trace(atmosphere_diagnostics.atmosphere_resistance_tensor_kg_s)
    )
    return ResistanceComponentDiagnostics(
        stokes_trace_kg_s=stokes_trace_kg_s,
        free_volume_trace_kg_s=free_volume_trace_kg_s,
        charge_cloud_trace_kg_s=charge_cloud_trace_kg_s,
        atmosphere_trace_kg_s=atmosphere_trace_kg_s,
        total_trace_kg_s=(
            stokes_trace_kg_s
            + free_volume_trace_kg_s
            + charge_cloud_trace_kg_s
            + atmosphere_trace_kg_s
        ),
    )


def compute_atmosphere_resistance_diagnostics(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    dielectric_constant: float,
    ionic_strength_mol_m3: float,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> AtmosphereResistanceDiagnostics:
    validate_site_configuration(configuration)
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if dielectric_constant <= 0.0:
        raise ValueError("dielectric_constant must be positive")
    if ionic_strength_mol_m3 < 0.0:
        raise ValueError("ionic_strength_mol_m3 must be nonnegative")
    if viscosity_Pa_s <= 0.0:
        raise ValueError("viscosity_Pa_s must be positive")
    site_records = _configuration_site_records(records, configuration)
    site_count = len(site_records)
    matrix_shape = (CARTESIAN_DIMENSION * site_count, CARTESIAN_DIMENSION * site_count)
    zero_tensor = np.zeros(matrix_shape, dtype=float)
    kappa_m_inv = _debye_kappa_m_inv(
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
    )
    charged_site_indices = tuple(
        site_index
        for site_index in range(site_count)
        if _site_formal_charge_number(records, configuration, site_index) != 0.0
    )
    if kappa_m_inv == 0.0 or not charged_site_indices:
        return AtmosphereResistanceDiagnostics(
            atmosphere_resistance_tensor_kg_s=zero_tensor,
            electrophoretic_resistance_tensor_kg_s=zero_tensor.copy(),
            relaxation_resistance_tensor_kg_s=zero_tensor.copy(),
            cation_diagonal_resistance_trace_kg_s=0.0,
            anion_diagonal_resistance_trace_kg_s=0.0,
            cation_anion_cross_resistance_trace_kg_s=0.0,
            mean_charge_cloud_form_factor=0.0,
            mean_state_geometry_form_factor=0.0,
            minimum_separation_over_debye_length=math.inf,
            debye_falkenhagen_time_s=0.0,
        )
    atmosphere_record = records.mixture_record["atmosphere"]
    required_keys = (
        "response_model",
        "maximum_wavevector_integer",
        "provenance",
        "fitted_to_conductivity",
    )
    missing_keys = tuple(key for key in required_keys if key not in atmosphere_record)
    if missing_keys:
        raise KeyError(f"mixture.atmosphere missing required keys {missing_keys}")
    if atmosphere_record["response_model"] != "finite_wavevector_pnp_stokes":
        raise ValueError("unsupported atmosphere response model")
    if bool(atmosphere_record["fitted_to_conductivity"]):
        raise ValueError("atmosphere response may not be fitted to conductivity")
    maximum_wavevector_integer = int(atmosphere_record["maximum_wavevector_integer"])
    if maximum_wavevector_integer <= 0:
        raise ValueError("atmosphere.maximum_wavevector_integer must be positive")
    charged_index_array = np.asarray(charged_site_indices, dtype=int)
    positions_m = np.asarray(configuration.positions_m, dtype=float)[charged_index_array]
    charge_numbers = np.asarray(
        [float(site_records[index]["charge_number"]) for index in charged_site_indices],
        dtype=float,
    )
    cloud_radii_m = np.asarray(
        [float(site_records[index]["charge_cloud_radius_m"]) for index in charged_site_indices],
        dtype=float,
    )
    stokes_drags_kg_s = np.asarray(
        [_stokes_drag_kg_s(site_records[index], viscosity_Pa_s) for index in charged_site_indices],
        dtype=float,
    )
    site_diffusivities_m2_s = K_B * temperature_K / stokes_drags_kg_s
    charge_weights = charge_numbers * charge_numbers
    ambipolar_diffusivity_m2_s = float(
        np.dot(charge_weights, site_diffusivities_m2_s) / np.sum(charge_weights)
    )
    debye_falkenhagen_time_s = UNITY / (
        ambipolar_diffusivity_m2_s * kappa_m_inv * kappa_m_inv
    )
    charged_matrix_shape = (
        CARTESIAN_DIMENSION * len(charged_site_indices),
        CARTESIAN_DIMENSION * len(charged_site_indices),
    )
    charged_electrophoretic_tensor = np.zeros(charged_matrix_shape, dtype=float)
    charged_relaxation_tensor = np.zeros(charged_matrix_shape, dtype=float)
    cloud_form_factors: list[float] = []
    geometry_form_factors: list[float] = []
    wavevectors_m_inv = _periodic_wavevectors_m_inv(
        np.asarray(configuration.box_lengths_m, dtype=float),
        maximum_wavevector_integer,
    )
    screening_arguments = kappa_m_inv * cloud_radii_m
    atmosphere_formation_activation = float(
        np.mean(
            screening_arguments
            * screening_arguments
            / (UNITY + screening_arguments * screening_arguments)
        )
    )
    for wavevector_m_inv in wavevectors_m_inv:
        wavevector_norm_m_inv = float(np.linalg.norm(wavevector_m_inv))
        if wavevector_norm_m_inv > kappa_m_inv:
            continue
        direction = wavevector_m_inv / wavevector_norm_m_inv
        longitudinal_projector = np.outer(direction, direction)
        transverse_projector = np.eye(CARTESIAN_DIMENSION) - longitudinal_projector
        wavevector_squared_m_inv2 = wavevector_norm_m_inv * wavevector_norm_m_inv
        phases = positions_m @ wavevector_m_inv
        cloud_factors = np.exp(
            -HARMONIC_PREFRACTOR
            * (wavevector_squared_m_inv2 + kappa_m_inv * kappa_m_inv)
            * cloud_radii_m
            * cloud_radii_m
        )
        charge_mode = charge_numbers * cloud_factors * np.exp(1j * phases)
        form_factor_matrix = np.real(np.outer(charge_mode, np.conj(charge_mode)))
        drag_weighted_form_factor = (
            np.sqrt(stokes_drags_kg_s[:, None] * stokes_drags_kg_s[None, :])
            * form_factor_matrix
        )
        screened_activation = (
            kappa_m_inv
            * kappa_m_inv
            / (wavevector_squared_m_inv2 + kappa_m_inv * kappa_m_inv)
        )
        electrophoretic_kernel = atmosphere_formation_activation * screened_activation
        mode_relaxation_time_s = UNITY / (
            ambipolar_diffusivity_m2_s
            * (wavevector_squared_m_inv2 + kappa_m_inv * kappa_m_inv)
        )
        relaxation_kernel = (
            atmosphere_formation_activation
            * screened_activation
            * mode_relaxation_time_s
            / debye_falkenhagen_time_s
        )
        mode_weight = UNITY / float(len(wavevectors_m_inv))
        charged_electrophoretic_tensor += mode_weight * np.kron(
            drag_weighted_form_factor,
            electrophoretic_kernel * transverse_projector,
        )
        charged_relaxation_tensor += mode_weight * np.kron(
            drag_weighted_form_factor,
            relaxation_kernel * longitudinal_projector,
        )
        cloud_form_factors.append(float(np.mean(cloud_factors)))
        geometry_form_factors.append(float(np.mean(np.abs(charge_mode) ** 2)))
    charged_coordinate_indices = np.concatenate(
        tuple(
            np.arange(
                CARTESIAN_DIMENSION * site_index,
                CARTESIAN_DIMENSION * (site_index + 1),
                dtype=int,
            )
            for site_index in charged_site_indices
        )
    )
    electrophoretic_tensor = np.zeros(matrix_shape, dtype=float)
    relaxation_tensor = np.zeros(matrix_shape, dtype=float)
    electrophoretic_tensor[
        np.ix_(charged_coordinate_indices, charged_coordinate_indices)
    ] = charged_electrophoretic_tensor
    relaxation_tensor[
        np.ix_(charged_coordinate_indices, charged_coordinate_indices)
    ] = charged_relaxation_tensor
    electrophoretic_tensor = symmetric_psd_numpy(
        electrophoretic_tensor,
        "electrophoretic atmosphere resistance",
    )
    relaxation_tensor = symmetric_psd_numpy(
        relaxation_tensor,
        "relaxation atmosphere resistance",
    )
    total_tensor = symmetric_psd_numpy(
        electrophoretic_tensor + relaxation_tensor,
        "total atmosphere resistance",
    )
    cation_coordinate_indices = _charged_role_coordinate_indices(
        records, configuration, site_records, "cation"
    )
    anion_coordinate_indices = _charged_role_coordinate_indices(
        records, configuration, site_records, "anion"
    )
    charged_distances_m = _charged_site_distance_matrix_m(
        positions_m,
        np.asarray(configuration.box_lengths_m, dtype=float),
    )
    return AtmosphereResistanceDiagnostics(
        atmosphere_resistance_tensor_kg_s=total_tensor,
        electrophoretic_resistance_tensor_kg_s=electrophoretic_tensor,
        relaxation_resistance_tensor_kg_s=relaxation_tensor,
        cation_diagonal_resistance_trace_kg_s=_principal_trace(
            total_tensor, cation_coordinate_indices
        ),
        anion_diagonal_resistance_trace_kg_s=_principal_trace(
            total_tensor, anion_coordinate_indices
        ),
        cation_anion_cross_resistance_trace_kg_s=_cross_block_sum(
            total_tensor, cation_coordinate_indices, anion_coordinate_indices
        ),
        mean_charge_cloud_form_factor=(
            float(np.mean(cloud_form_factors)) if cloud_form_factors else 0.0
        ),
        mean_state_geometry_form_factor=(
            float(np.mean(geometry_form_factors)) if geometry_form_factors else 0.0
        ),
        minimum_separation_over_debye_length=_minimum_scaled_nonself_distance(
            charged_distances_m, kappa_m_inv
        ),
        debye_falkenhagen_time_s=debye_falkenhagen_time_s,
    )


def _periodic_wavevectors_m_inv(box_lengths_m: Array, maximum_integer: int) -> tuple[Array, ...]:
    wavevectors: list[Array] = []
    for first_index in range(-maximum_integer, maximum_integer + 1):
        for second_index in range(-maximum_integer, maximum_integer + 1):
            for third_index in range(-maximum_integer, maximum_integer + 1):
                integer_vector = np.asarray(
                    [first_index, second_index, third_index], dtype=float
                )
                if np.all(integer_vector == 0.0):
                    continue
                wavevectors.append(2.0 * math.pi * integer_vector / box_lengths_m)
    return tuple(wavevectors)


def _charged_site_distance_matrix_m(positions_m: Array, box_lengths_m: Array) -> Array:
    displacements_m = positions_m[None, :, :] - positions_m[:, None, :]
    displacements_m -= box_lengths_m * np.rint(displacements_m / box_lengths_m)
    return np.linalg.norm(displacements_m, axis=2)


def _principal_trace(tensor: Array, coordinate_indices: Array) -> float:
    if coordinate_indices.size == 0:
        return 0.0
    return float(np.trace(tensor[np.ix_(coordinate_indices, coordinate_indices)]))


def _cross_block_sum(tensor: Array, first_indices: Array, second_indices: Array) -> float:
    if first_indices.size == 0 or second_indices.size == 0:
        return 0.0
    return float(np.sum(tensor[np.ix_(first_indices, second_indices)]))


def _minimum_scaled_nonself_distance(distances_m: Array, kappa_m_inv: float) -> float:
    nonself_distances_m = distances_m[~np.eye(distances_m.shape[0], dtype=bool)]
    if nonself_distances_m.size == 0:
        return math.inf
    return float(np.min(nonself_distances_m) * kappa_m_inv)


def _charged_role_coordinate_indices(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_records: tuple[dict, ...],
    role: str,
) -> Array:
    coordinate_indices: list[int] = []
    for site_index, site_record in enumerate(site_records):
        if float(site_record["charge_number"]) == 0.0:
            continue
        species_name = configuration.species_names[site_index]
        species_role = str(records.species_records[species_name]["role"])
        if species_role != role:
            continue
        start = CARTESIAN_DIMENSION * site_index
        coordinate_indices.extend(range(start, start + CARTESIAN_DIMENSION))
    return np.asarray(coordinate_indices, dtype=int)


def _debye_falkenhagen_time_s(
    kappa_m_inv: float,
    temperature_K: float,
    site_records: tuple[dict, ...],
    charged_site_indices: tuple[int, ...],
    viscosity_Pa_s: float,
) -> float:
    if kappa_m_inv == 0.0:
        return 0.0
    mean_charged_drag_kg_s = _mean_charged_site_stokes_drag_kg_s(
        site_records,
        charged_site_indices,
        viscosity_Pa_s,
    )
    atmosphere_diffusivity_m2_s = K_B * temperature_K / mean_charged_drag_kg_s
    return 1.0 / (atmosphere_diffusivity_m2_s * kappa_m_inv * kappa_m_inv)


def compute_charge_polarization_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    """Return ionic charge polarization from formal molecule charge centers.

    Partial atomic charges belong to U(q): Coulomb, Born, LJ/coordination, and
    local structure.  Long-time ionic conductivity is driven by transported
    formal charge centers.  Neutral molecules therefore contribute no net
    polarization current even when their force-field sites carry nonzero
    partial charges.
    """

    polarization_m = np.zeros(CARTESIAN_DIMENSION, dtype=float)
    for molecule_key in _configuration_molecule_keys(configuration):
        species_name, molecule_id = molecule_key
        formal_charge_number = float(records.species_records[species_name]["formal_charge_e"])
        if formal_charge_number == 0.0:
            continue
        charge_center_m = molecule_center_of_mass_m(
            records,
            configuration,
            species_name,
            molecule_id,
        )
        polarization_m += formal_charge_number * charge_center_m
    return polarization_m


def compute_charge_polarization_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    """Return dP/dq for transported formal charge centers."""

    site_count = len(configuration.species_names)
    gradient = np.zeros(
        (CARTESIAN_DIMENSION, CARTESIAN_DIMENSION * site_count),
        dtype=float,
    )
    for molecule_key in _configuration_molecule_keys(configuration):
        species_name, molecule_id = molecule_key
        formal_charge_number = float(records.species_records[species_name]["formal_charge_e"])
        if formal_charge_number == 0.0:
            continue
        site_indices, mass_fractions = molecule_site_indices_and_mass_fractions(
            records,
            configuration,
            species_name,
            molecule_id,
        )
        for site_index, mass_fraction in zip(
            site_indices,
            mass_fractions,
            strict=True,
        ):
            site_slice = _cartesian_site_slice(site_index)
            gradient[:, site_slice] = (
                formal_charge_number * mass_fraction * np.eye(CARTESIAN_DIMENSION)
            )
    return gradient


def _configuration_molecule_keys(
    configuration: SiteConfiguration,
) -> tuple[tuple[str, int], ...]:
    molecule_keys = []
    for site_index, species_name in enumerate(configuration.species_names):
        molecule_keys.append((species_name, int(configuration.molecule_ids[site_index])))
    return tuple(dict.fromkeys(molecule_keys))


def _site_indices_for_molecule(
    configuration: SiteConfiguration,
    species_name: str,
    molecule_id: int,
) -> tuple[int, ...]:
    return tuple(
        site_index
        for site_index, current_species_name in enumerate(configuration.species_names)
        if current_species_name == species_name
        and int(configuration.molecule_ids[site_index]) == molecule_id
    )


def molecule_center_of_mass_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    species_name: str,
    molecule_id: int,
) -> Array:
    site_indices, mass_fractions = molecule_site_indices_and_mass_fractions(
        records,
        configuration,
        species_name,
        molecule_id,
    )
    unwrapped_positions_m = np.asarray(configuration.unwrapped_positions_m, dtype=float)
    return np.einsum(
        "i,ia->a",
        mass_fractions,
        unwrapped_positions_m[np.asarray(site_indices, dtype=int)],
    )


def molecule_site_indices_and_mass_fractions(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    species_name: str,
    molecule_id: int,
) -> tuple[tuple[int, ...], Array]:
    site_indices = _site_indices_for_molecule(configuration, species_name, molecule_id)
    if not site_indices:
        raise ValueError("molecular center needs at least one site")
    masses_kg = np.asarray(
        [
            float(_site_record(records, configuration, site_index)["mass_kg"])
            for site_index in site_indices
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(masses_kg)) or np.any(masses_kg <= 0.0):
        raise ValueError("molecular center requires positive finite site masses")
    return site_indices, masses_kg / float(np.sum(masses_kg))


def assign_pair_basin(
    pair_distance_m: float,
    basis_record: dict,
) -> PairBasin:
    pair_basins = basis_record["pair_basins"]
    cip_cutoff_m = float(pair_basins["r_CIP_m"])
    ssip_cutoff_m = float(pair_basins["r_SSIP_m"])
    free_cutoff_m = float(pair_basins["r_free_m"])
    if not (0.0 < cip_cutoff_m < ssip_cutoff_m <= free_cutoff_m):
        raise ValueError("pair basin cutoffs must satisfy 0 < r_CIP < r_SSIP <= r_free")
    if pair_distance_m < cip_cutoff_m:
        return PairBasin.CONTACT_ION_PAIR
    if pair_distance_m < ssip_cutoff_m:
        return PairBasin.SOLVENT_SEPARATED_ION_PAIR
    if pair_distance_m >= free_cutoff_m:
        return PairBasin.FREE
    return PairBasin.TRANSITION


def _configuration_site_lookup(configuration: SiteConfiguration) -> dict[tuple[str, int, int], int]:
    lookup = {}
    for site_index, species_name in enumerate(configuration.species_names):
        key = (
            species_name,
            int(configuration.molecule_ids[site_index]),
            int(configuration.site_ids[site_index]),
        )
        if key in lookup:
            raise ValueError(f"duplicate configuration site key {key}")
        lookup[key] = site_index
    return lookup


def _molecule_ids_for_species(
    configuration: SiteConfiguration,
    species_name: str,
) -> tuple[int, ...]:
    molecule_ids = []
    for site_index, current_species_name in enumerate(configuration.species_names):
        if current_species_name == species_name:
            molecule_ids.append(int(configuration.molecule_ids[site_index]))
    return tuple(sorted(set(molecule_ids)))


def _site_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
) -> dict:
    species_name = configuration.species_names[site_index]
    species_record = records.species_records[species_name]
    requested_site_id = int(configuration.site_ids[site_index])
    for site_record in species_record["sites"]:
        if int(site_record["site_id"]) == requested_site_id:
            return site_record
    raise KeyError(f"{species_name} has no site_id {requested_site_id}")


def _configuration_site_records(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> tuple[dict, ...]:
    return tuple(
        _site_record(records, configuration, site_index)
        for site_index in range(len(configuration.species_names))
    )


def _mixed_lj_parameters(first_site: dict, second_site: dict) -> tuple[float, float]:
    sigma_m = HARMONIC_PREFRACTOR * (
        float(first_site["lj_sigma_m"]) + float(second_site["lj_sigma_m"])
    )
    epsilon_J = math.sqrt(
        float(first_site["lj_epsilon_J"]) * float(second_site["lj_epsilon_J"])
    )
    if sigma_m <= 0.0:
        raise ValueError("mixed LJ sigma must be positive")
    if epsilon_J < 0.0:
        raise ValueError("mixed LJ epsilon must be non-negative")
    return sigma_m, epsilon_J


def _coordination_number(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    switch_name: str,
) -> float:
    switch_record = records.basis_record["coordination_switches"][switch_name]
    center_role = str(switch_record["center_role"])
    ligand_roles = tuple(str(ligand_role) for ligand_role in switch_record["ligand_roles"])
    switch_radius_m = float(switch_record["r0_m"])
    exponent = float(switch_record["exponent"])
    aggregation = str(switch_record["aggregation"])
    if not _has_role(records, configuration, center_role):
        return 0.0
    center_index = _first_role_index(records, configuration, center_role)
    site_switch_values_by_molecule: dict[tuple[str, int], list[float]] = {}
    for site_index, species_name in enumerate(configuration.species_names):
        if site_index == center_index:
            continue
        if records.species_records[species_name]["role"] not in ligand_roles:
            continue
        distance_m = _minimum_image_distance_m(
            configuration.positions_m[center_index],
            configuration.positions_m[site_index],
            configuration.box_lengths_m,
        )
        if distance_m <= ZERO_DISTANCE_TOLERANCE_M:
            raise ValueError("coordination distance is zero")
        molecule_key = (species_name, int(configuration.molecule_ids[site_index]))
        site_switch_values_by_molecule.setdefault(molecule_key, []).append(
            UNITY / (UNITY + (distance_m / switch_radius_m) ** exponent)
        )
    if aggregation == "site_sum":
        return float(
            sum(
                sum(site_switch_values)
                for site_switch_values in site_switch_values_by_molecule.values()
            )
        )
    if aggregation == "molecule_max_site_sum":
        return float(
            sum(
                max(site_switch_values)
                for site_switch_values in site_switch_values_by_molecule.values()
            )
        )
    raise ValueError(f"unsupported coordination aggregation {aggregation}")


def _first_role_index(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: str,
) -> int:
    for site_index, species_name in enumerate(configuration.species_names):
        if records.species_records[species_name]["role"] == role:
            return site_index
    raise ValueError(f"configuration has no species with role {role}")


def _has_role(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: str,
) -> bool:
    for species_name in configuration.species_names:
        if records.species_records[species_name]["role"] == role:
            return True
    return False


def _species_mean_steric_radius_m(species_record: dict) -> float:
    radii = [float(site_record["steric_radius_m"]) for site_record in species_record["sites"]]
    if not radii:
        raise ValueError("species has no steric radii")
    return float(np.mean(np.asarray(radii, dtype=float)))


def _stokes_drag_kg_s(site_record: dict, viscosity_Pa_s: float) -> float:
    hydrodynamic_radius_m = float(site_record["hydrodynamic_radius_m"])
    if hydrodynamic_radius_m <= 0.0:
        raise ValueError("hydrodynamic_radius_m must be positive")
    return (
        STOKES_SPHERE_DRAG_FACTOR
        * math.pi
        * viscosity_Pa_s
        * hydrodynamic_radius_m
    )


def _free_volume_drag_kg_s(
    records: PhysicalLibraryRecords,
    stokes_drag_kg_s: float,
    packing_fraction: float,
) -> float:
    phi_max = float(records.mixture_record["packing"]["phi_max"])
    if packing_fraction >= phi_max:
        return math.inf
    exponent = float(records.mixture_record["mobility"]["free_volume_exponent"])
    obstruction = (UNITY - packing_fraction / phi_max) ** (-exponent) - UNITY
    return obstruction * stokes_drag_kg_s


def _charge_cloud_drag_kg_s(
    hydrodynamic_radius_m: float,
    charge_cloud_radius_m: float,
    stokes_drag_kg_s: float,
    effective_charge_number: float,
    kappa_m_inv: float,
    dielectric_constant: float,
    temperature_K: float,
    short_range_wavevector_fraction: float,
) -> float:
    charge_number = abs(effective_charge_number)
    if charge_number == 0.0:
        return 0.0
    if stokes_drag_kg_s <= 0.0:
        raise ValueError("stokes_drag_kg_s must be positive")
    cloud_radius_m = float(charge_cloud_radius_m)
    if cloud_radius_m <= 0.0:
        raise ValueError("charge_cloud_radius_m must be positive")
    screen_argument = kappa_m_inv * cloud_radius_m
    screen_factor = screen_argument * screen_argument / (
        UNITY + screen_argument * screen_argument
    )
    if hydrodynamic_radius_m <= 0.0:
        raise ValueError("hydrodynamic_radius_m must be positive")
    if not 0.0 <= short_range_wavevector_fraction <= UNITY:
        raise ValueError("short_range_wavevector_fraction must be between zero and one")
    cloud_geometry_factor = hydrodynamic_radius_m / cloud_radius_m
    bjerrum_length_m = _bjerrum_length_m(dielectric_constant, temperature_K)
    electrostatic_coupling = bjerrum_length_m / cloud_radius_m
    electrostatic_cloud_factor = electrostatic_coupling**CHARGE_CLOUD_RESPONSE_EXPONENT
    return (
        stokes_drag_kg_s
        * charge_number
        * charge_number
        * cloud_geometry_factor
        * electrostatic_cloud_factor
        * screen_factor
        * short_range_wavevector_fraction
    )


def _short_range_wavevector_fraction(
    kappa_m_inv: float,
    charge_cloud_radius_m: float,
) -> float:
    if kappa_m_inv < 0.0:
        raise ValueError("kappa_m_inv must be nonnegative")
    if charge_cloud_radius_m <= 0.0:
        raise ValueError("charge_cloud_radius_m must be positive")
    if kappa_m_inv == 0.0:
        return UNITY
    scaled_cutoff = kappa_m_inv * charge_cloud_radius_m
    return float(
        math.erfc(scaled_cutoff)
        + 2.0
        * scaled_cutoff
        * math.exp(-scaled_cutoff * scaled_cutoff)
        / math.sqrt(math.pi)
    )


def _effective_charge_cloud_number(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
) -> float:
    species_name = configuration.species_names[site_index]
    species_record = records.species_records[species_name]
    formal_charge_number = float(species_record["formal_charge_e"])
    if formal_charge_number == 0.0:
        return 0.0
    molecule_site_indices = _site_indices_for_molecule(
        configuration,
        species_name,
        int(configuration.molecule_ids[site_index]),
    )
    absolute_partial_charges = np.asarray(
        [
            abs(
                float(
                    _site_record(records, configuration, molecule_site_index)[
                        "charge_number"
                    ]
                )
            )
            for molecule_site_index in molecule_site_indices
        ],
        dtype=float,
    )
    partial_charge_total = float(np.sum(absolute_partial_charges))
    if partial_charge_total == 0.0:
        raise ValueError("formally charged molecule has no charged force-field sites")
    site_partial_charge = abs(
        float(_site_record(records, configuration, site_index)["charge_number"])
    )
    return abs(formal_charge_number) * site_partial_charge / partial_charge_total


def _species_absolute_partial_charge_sum(species_record: dict) -> float:
    return float(
        np.sum(
            np.asarray(
                [
                    abs(float(site_record["charge_number"]))
                    for site_record in species_record["sites"]
                ],
                dtype=float,
            )
        )
    )


def _bjerrum_length_m(dielectric_constant: float, temperature_K: float) -> float:
    if dielectric_constant <= 0.0:
        raise ValueError("dielectric_constant must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    return (
        E_CHARGE
        * E_CHARGE
        / (4.0 * math.pi * EPS_0 * dielectric_constant * K_B * temperature_K)
    )


def _charged_site_stokes_drag_matrix_kg_s(
    site_records: tuple[dict, ...],
    charged_site_indices: tuple[int, ...],
    viscosity_Pa_s: float,
) -> Array:
    charged_site_drags_kg_s = np.asarray(
        [
            _stokes_drag_kg_s(site_records[site_index], viscosity_Pa_s)
            for site_index in charged_site_indices
        ],
        dtype=float,
    )
    return np.sqrt(charged_site_drags_kg_s[:, None] * charged_site_drags_kg_s[None, :])


def _mean_charged_site_stokes_drag_kg_s(
    site_records: tuple[dict, ...],
    charged_site_indices: tuple[int, ...],
    viscosity_Pa_s: float,
) -> float:
    if not charged_site_indices:
        raise ValueError("charged_site_indices must not be empty")
    charged_site_drags_kg_s = np.asarray(
        [
            _stokes_drag_kg_s(site_records[site_index], viscosity_Pa_s)
            for site_index in charged_site_indices
        ],
        dtype=float,
    )
    return float(np.mean(charged_site_drags_kg_s))


def _site_formal_charge_number(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
) -> float:
    species_name = configuration.species_names[site_index]
    return float(records.species_records[species_name]["formal_charge_e"])


def _debye_kappa_m_inv(
    dielectric_constant: float,
    ionic_strength_mol_m3: float,
    temperature_K: float,
) -> float:
    if dielectric_constant <= 0.0:
        raise ValueError("dielectric_constant must be positive")
    if ionic_strength_mol_m3 < 0.0:
        raise ValueError("ionic_strength_mol_m3 must be nonnegative")
    if ionic_strength_mol_m3 == 0.0:
        return 0.0
    return math.sqrt(
        F
        * F
        * 2.0
        * ionic_strength_mol_m3
        / (EPS_0 * dielectric_constant * R * temperature_K)
    )


def _same_molecule(
    configuration: SiteConfiguration,
    first_index: int,
    second_index: int,
) -> bool:
    return (
        configuration.species_names[first_index] == configuration.species_names[second_index]
        and int(configuration.molecule_ids[first_index])
        == int(configuration.molecule_ids[second_index])
    )


def _minimum_image_vector_m(first_position_m: Array, second_position_m: Array, box_lengths_m: Array) -> Array:
    vector_m = np.asarray(first_position_m, dtype=float) - np.asarray(second_position_m, dtype=float)
    lengths_m = np.asarray(box_lengths_m, dtype=float)
    return vector_m - lengths_m * np.rint(vector_m / lengths_m)


def _minimum_image_distance_m(
    first_position_m: Array,
    second_position_m: Array,
    box_lengths_m: Array,
) -> float:
    vector_m = _minimum_image_vector_m(first_position_m, second_position_m, box_lengths_m)
    return float(np.linalg.norm(vector_m))


def _angle_rad(
    first_position_m: Array,
    center_position_m: Array,
    third_position_m: Array,
    box_lengths_m: Array,
) -> float:
    first_vector_m = _minimum_image_vector_m(first_position_m, center_position_m, box_lengths_m)
    third_vector_m = _minimum_image_vector_m(third_position_m, center_position_m, box_lengths_m)
    first_norm_m = float(np.linalg.norm(first_vector_m))
    third_norm_m = float(np.linalg.norm(third_vector_m))
    if first_norm_m <= ZERO_DISTANCE_TOLERANCE_M or third_norm_m <= ZERO_DISTANCE_TOLERANCE_M:
        raise ValueError("angle contains zero-length vector")
    cosine = float(np.dot(first_vector_m, third_vector_m) / (first_norm_m * third_norm_m))
    if cosine < -UNITY:
        if cosine < -UNITY - ANGLE_COSINE_ROUNDOFF_TOLERANCE:
            raise ValueError("angle cosine is outside the valid range")
        cosine = -UNITY
    if cosine > UNITY:
        if cosine > UNITY + ANGLE_COSINE_ROUNDOFF_TOLERANCE:
            raise ValueError("angle cosine is outside the valid range")
        cosine = UNITY
    return math.acos(cosine)


def _torsion_rad(
    first_position_m: Array,
    second_position_m: Array,
    third_position_m: Array,
    fourth_position_m: Array,
    box_lengths_m: Array,
) -> float:
    first_bond = _minimum_image_vector_m(second_position_m, first_position_m, box_lengths_m)
    second_bond = _minimum_image_vector_m(third_position_m, second_position_m, box_lengths_m)
    third_bond = _minimum_image_vector_m(fourth_position_m, third_position_m, box_lengths_m)
    first_normal = np.cross(first_bond, second_bond)
    second_normal = np.cross(second_bond, third_bond)
    first_norm = float(np.linalg.norm(first_normal))
    second_norm = float(np.linalg.norm(second_normal))
    middle_norm = float(np.linalg.norm(second_bond))
    if (
        first_norm <= ZERO_DISTANCE_TOLERANCE_M
        or second_norm <= ZERO_DISTANCE_TOLERANCE_M
        or middle_norm <= ZERO_DISTANCE_TOLERANCE_M
    ):
        raise ValueError("torsion contains zero-length normal")
    first_unit = first_normal / first_norm
    second_unit = second_normal / second_norm
    middle_unit = second_bond / middle_norm
    torsion_x = float(np.dot(first_unit, second_unit))
    torsion_y = float(np.dot(np.cross(first_unit, middle_unit), second_unit))
    return math.atan2(torsion_y, torsion_x)


def _cell_volume_m3(box_lengths_m: Array) -> float:
    box_lengths = np.asarray(box_lengths_m, dtype=float)
    if box_lengths.shape != (CARTESIAN_DIMENSION,):
        raise ValueError("box_lengths_m must have shape (3,)")
    cell_volume_m3 = float(np.prod(box_lengths))
    if cell_volume_m3 <= 0.0:
        raise ValueError("cell volume must be positive")
    return cell_volume_m3
