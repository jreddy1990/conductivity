"""Physical-object builders for the projected conductivity model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from constants import EPS_0, E_CHARGE, F, K_B, N_A, R
from conductivity.physical_library.library_io import PhysicalLibraryRecords

Array = np.ndarray
_BONDED_ENERGY_CACHE: dict[tuple, float] = {}

CARTESIAN_DIMENSION = 3
LJ_ATTRACTIVE_EXPONENT = 6  # Explicit constant: Lennard-Jones 12-6 attractive exponent.
LJ_REPULSIVE_EXPONENT_MULTIPLIER = 2  # Repulsive exponent is twice the attractive exponent.
BORN_DENOMINATOR_FACTOR = 8.0  # Explicit constant: Born charging free-energy denominator.
STOKES_SPHERE_DRAG_FACTOR = 6.0  # Explicit constant: Stokes sphere drag prefactor.
HARMONIC_PREFRACTOR = 0.5  # Harmonic bonded and packing quadratic prefactor.
UNITY = 1.0
ANGLE_COSINE_ROUNDOFF_TOLERANCE = 1.0e-12
ZERO_DISTANCE_TOLERANCE_M = 1.0e-30


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
    shape_trace_kg_s: float
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
    cluster_energy_J_mol = compute_cluster_energy_J_mol(records, configuration)
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
            + cluster_energy_J_mol
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
        total_energy_J_mol += float(coefficient_J_mol) * _coordination_number(
            records,
            configuration,
            switch_name,
        )
    return total_energy_J_mol


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


def compute_cluster_energy_J_mol(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    if not _has_role(records, configuration, "cation"):
        return 0.0
    if not _has_role(records, configuration, "anion"):
        return 0.0
    lithium_index = _first_role_index(records, configuration, "cation")
    anion_index = _first_role_index(records, configuration, "anion")
    pair_distance_m = _minimum_image_distance_m(
        configuration.positions_m[lithium_index],
        configuration.positions_m[anion_index],
        configuration.box_lengths_m,
    )
    basin = assign_pair_basin(pair_distance_m, records.basis_record)
    basin_energies = records.basis_record["free_energy_terms"]["pair_basin_J_mol"]
    return float(basin_energies[basin.value])


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
    site_count = len(configuration.species_names)
    mobility = np.zeros(
        (CARTESIAN_DIMENSION * site_count, CARTESIAN_DIMENSION * site_count),
        dtype=float,
    )
    for site_index in range(site_count):
        site_record = _site_record(records, configuration, site_index)
        hydrodynamic_radius_m = float(site_record["hydrodynamic_radius_m"])
        if hydrodynamic_radius_m <= 0.0:
            raise ValueError("hydrodynamic_radius_m must be positive")
        drag_kg_s = (
            STOKES_SPHERE_DRAG_FACTOR
            * math.pi
            * viscosity_Pa_s
            * hydrodynamic_radius_m
        )
        diffusivity_m2_s = K_B * temperature_K / drag_kg_s
        start = CARTESIAN_DIMENSION * site_index
        stop = start + CARTESIAN_DIMENSION
        mobility[start:stop, start:stop] = diffusivity_m2_s * np.eye(
            CARTESIAN_DIMENSION,
            dtype=float,
        )
    return mobility


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
    mobility = np.zeros_like(resistance)
    charged_coordinate_indices = []
    site_records = _configuration_site_records(records, configuration)
    for site_index, site_record in enumerate(site_records):
        start = CARTESIAN_DIMENSION * site_index
        stop = start + CARTESIAN_DIMENSION
        if _site_formal_charge_number(records, configuration, site_index) == 0.0:
            mobility[start:stop, start:stop] = np.linalg.pinv(
                resistance[start:stop, start:stop],
                hermitian=True,
            )
        else:
            charged_coordinate_indices.extend(range(start, stop))
    if charged_coordinate_indices:
        charged_index_array = np.asarray(charged_coordinate_indices, dtype=int)
        charged_resistance = resistance[np.ix_(charged_index_array, charged_index_array)]
        mobility[np.ix_(charged_index_array, charged_index_array)] = np.linalg.pinv(
            charged_resistance,
            hermitian=True,
        )
    return K_B * temperature_K * mobility


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
    mobility_record = records.mixture_record["mobility"]
    site_count = len(configuration.species_names)
    resistance = np.zeros(
        (CARTESIAN_DIMENSION * site_count, CARTESIAN_DIMENSION * site_count),
        dtype=float,
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
        drag_kg_s = stokes_drag_kg_s
        drag_kg_s += _free_volume_drag_kg_s(
            records,
            stokes_drag_kg_s,
            local_packing_fraction,
        )
        drag_kg_s += _charge_cloud_drag_kg_s(
            mobility_record,
            site_record,
            stokes_drag_kg_s,
            _effective_charge_cloud_number(records, configuration, site_index),
            kappa_m_inv,
        )
        drag_kg_s += _shape_drag_kg_s(
            records,
            configuration,
            site_index,
            stokes_drag_kg_s,
        )
        start = CARTESIAN_DIMENSION * site_index
        stop = start + CARTESIAN_DIMENSION
        resistance[start:stop, start:stop] = drag_kg_s * np.eye(
            CARTESIAN_DIMENSION,
            dtype=float,
        )
    atmosphere_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        configuration,
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
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
    mobility_record = records.mixture_record["mobility"]
    kappa_m_inv = _debye_kappa_m_inv(
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
    )
    stokes_trace_kg_s = 0.0
    free_volume_trace_kg_s = 0.0
    charge_cloud_trace_kg_s = 0.0
    shape_trace_kg_s = 0.0
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
            mobility_record,
            site_record,
            stokes_drag_kg_s,
            _effective_charge_cloud_number(records, configuration, site_index),
            kappa_m_inv,
        )
        shape_trace_kg_s += CARTESIAN_DIMENSION * _shape_drag_kg_s(
            records,
            configuration,
            site_index,
            stokes_drag_kg_s,
        )
    atmosphere_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        configuration,
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
    )
    atmosphere_trace_kg_s = float(
        np.trace(atmosphere_diagnostics.atmosphere_resistance_tensor_kg_s)
    )
    return ResistanceComponentDiagnostics(
        stokes_trace_kg_s=stokes_trace_kg_s,
        free_volume_trace_kg_s=free_volume_trace_kg_s,
        charge_cloud_trace_kg_s=charge_cloud_trace_kg_s,
        shape_trace_kg_s=shape_trace_kg_s,
        atmosphere_trace_kg_s=atmosphere_trace_kg_s,
        total_trace_kg_s=(
            stokes_trace_kg_s
            + free_volume_trace_kg_s
            + charge_cloud_trace_kg_s
            + shape_trace_kg_s
            + atmosphere_trace_kg_s
        ),
    )


def compute_atmosphere_resistance_diagnostics(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    dielectric_constant: float,
    ionic_strength_mol_m3: float,
    temperature_K: float,
) -> AtmosphereResistanceDiagnostics:
    validate_site_configuration(configuration)
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if dielectric_constant <= 0.0:
        raise ValueError("dielectric_constant must be positive")
    if ionic_strength_mol_m3 < 0.0:
        raise ValueError("ionic_strength_mol_m3 must be nonnegative")
    mobility_record = records.mixture_record["mobility"]
    site_records = _configuration_site_records(records, configuration)
    kappa_m_inv = _debye_kappa_m_inv(
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
    )
    site_count = len(configuration.species_names)
    matrix_shape = (CARTESIAN_DIMENSION * site_count, CARTESIAN_DIMENSION * site_count)
    zero_tensor = np.zeros(matrix_shape, dtype=float)
    atmosphere_lambda_kg_s = float(mobility_record["atmosphere_lambda_kg_s"])
    if atmosphere_lambda_kg_s == 0.0 or kappa_m_inv == 0.0:
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
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    charged_site_indices = tuple(
        site_index
        for site_index, site_record in enumerate(site_records)
        if float(site_record["charge_number"]) != 0.0
    )
    if not charged_site_indices:
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
    charged_index_array = np.asarray(charged_site_indices, dtype=int)
    charged_positions_m = positions_m[charged_index_array]
    charged_displacements_m = (
        charged_positions_m[None, :, :] - charged_positions_m[:, None, :]
    )
    box_lengths_m = np.asarray(configuration.box_lengths_m, dtype=float)
    charged_displacements_m = charged_displacements_m - box_lengths_m * np.rint(
        charged_displacements_m / box_lengths_m
    )
    charged_distances_m = np.linalg.norm(charged_displacements_m, axis=2)
    charge_numbers = np.asarray(
        [float(site_records[site_index]["charge_number"]) for site_index in charged_site_indices],
        dtype=float,
    )
    cloud_radii_m = np.asarray(
        [
            float(site_records[site_index]["charge_cloud_radius_m"])
            for site_index in charged_site_indices
        ],
        dtype=float,
    )
    cloud_factor_matrix = np.exp(
        -HARMONIC_PREFRACTOR
        * kappa_m_inv
        * kappa_m_inv
        * (
            cloud_radii_m[:, None] * cloud_radii_m[:, None]
            + cloud_radii_m[None, :] * cloud_radii_m[None, :]
        )
    )
    screen_arguments = kappa_m_inv * cloud_radii_m
    screen_activation = screen_arguments * screen_arguments / (
        UNITY + screen_arguments * screen_arguments
    )
    atmosphere_activation_matrix = np.sqrt(
        screen_activation[:, None] * screen_activation[None, :]
    )
    geometry_factor_matrix = np.exp(
        -charged_distances_m
        * charged_distances_m
        * kappa_m_inv
        * kappa_m_inv
    )
    resistance_factor_matrix = (
        atmosphere_lambda_kg_s
        * charge_numbers[:, None]
        * charge_numbers[None, :]
        * atmosphere_activation_matrix
        * cloud_factor_matrix
        * geometry_factor_matrix
    )
    total_tensor = np.zeros(matrix_shape, dtype=float)
    electrophoretic_tensor = np.zeros(matrix_shape, dtype=float)
    relaxation_tensor = np.zeros(matrix_shape, dtype=float)
    for cartesian_axis in range(CARTESIAN_DIMENSION):
        coordinate_indices = charged_index_array * CARTESIAN_DIMENSION + cartesian_axis
        total_tensor[np.ix_(coordinate_indices, coordinate_indices)] += (
            resistance_factor_matrix
        )
        electrophoretic_tensor[np.ix_(coordinate_indices, coordinate_indices)] += (
            np.diag(np.diag(resistance_factor_matrix))
        )
        relaxation_tensor[np.ix_(coordinate_indices, coordinate_indices)] += (
            resistance_factor_matrix - np.diag(np.diag(resistance_factor_matrix))
        )
    cation_coordinate_indices = _charged_role_coordinate_indices(
        records,
        configuration,
        site_records,
        "cation",
    )
    anion_coordinate_indices = _charged_role_coordinate_indices(
        records,
        configuration,
        site_records,
        "anion",
    )
    cation_anion_cross_trace = 0.0
    if cation_coordinate_indices.size > 0 and anion_coordinate_indices.size > 0:
        cation_anion_cross_trace = float(
            np.sum(total_tensor[np.ix_(cation_coordinate_indices, anion_coordinate_indices)])
        )
    nonself_mask = ~np.eye(charged_distances_m.shape[0], dtype=bool)
    finite_nonself_distances_m = charged_distances_m[nonself_mask]
    if finite_nonself_distances_m.size == 0:
        minimum_separation_over_debye_length = math.inf
    else:
        minimum_separation_over_debye_length = float(
            np.min(finite_nonself_distances_m) * kappa_m_inv
        )
    return AtmosphereResistanceDiagnostics(
        atmosphere_resistance_tensor_kg_s=total_tensor,
        electrophoretic_resistance_tensor_kg_s=electrophoretic_tensor,
        relaxation_resistance_tensor_kg_s=relaxation_tensor,
        cation_diagonal_resistance_trace_kg_s=float(
            np.trace(total_tensor[np.ix_(cation_coordinate_indices, cation_coordinate_indices)])
        )
        if cation_coordinate_indices.size > 0
        else 0.0,
        anion_diagonal_resistance_trace_kg_s=float(
            np.trace(total_tensor[np.ix_(anion_coordinate_indices, anion_coordinate_indices)])
        )
        if anion_coordinate_indices.size > 0
        else 0.0,
        cation_anion_cross_resistance_trace_kg_s=cation_anion_cross_trace,
        mean_charge_cloud_form_factor=float(np.mean(cloud_factor_matrix)),
        mean_state_geometry_form_factor=float(np.mean(geometry_factor_matrix)),
        minimum_separation_over_debye_length=minimum_separation_over_debye_length,
        debye_falkenhagen_time_s=_debye_falkenhagen_time_s(
            kappa_m_inv,
            temperature_K,
            atmosphere_lambda_kg_s,
        ),
    )


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
    atmosphere_lambda_kg_s: float,
) -> float:
    if kappa_m_inv == 0.0 or atmosphere_lambda_kg_s == 0.0:
        return 0.0
    atmosphere_diffusivity_m2_s = K_B * temperature_K / atmosphere_lambda_kg_s
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
    unwrapped_positions_m = np.asarray(configuration.unwrapped_positions_m, dtype=float)
    for molecule_key in _configuration_molecule_keys(configuration):
        species_name, molecule_id = molecule_key
        formal_charge_number = float(records.species_records[species_name]["formal_charge_e"])
        if formal_charge_number == 0.0:
            continue
        charge_center_site_index = _charge_center_site_index_for_molecule(
            records,
            configuration,
            species_name,
            molecule_id,
        )
        charge_center_m = unwrapped_positions_m[charge_center_site_index]
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
        charge_center_site_index = _charge_center_site_index_for_molecule(
            records,
            configuration,
            species_name,
            molecule_id,
        )
        for cartesian_index in range(CARTESIAN_DIMENSION):
            gradient[
                cartesian_index,
                CARTESIAN_DIMENSION * charge_center_site_index + cartesian_index,
            ] = formal_charge_number
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


def _charge_center_site_index_for_molecule(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    species_name: str,
    molecule_id: int,
) -> int:
    site_indices = _site_indices_for_molecule(configuration, species_name, molecule_id)
    if not site_indices:
        raise ValueError("charge center needs at least one site")
    absolute_partial_charge_numbers = np.asarray(
        [
            abs(float(_site_record(records, configuration, site_index)["charge_number"]))
            for site_index in site_indices
        ],
        dtype=float,
    )
    if float(np.max(absolute_partial_charge_numbers)) == 0.0:
        raise ValueError("formally charged molecule has no charged force-field site")
    return int(site_indices[int(np.argmax(absolute_partial_charge_numbers))])


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
    if not _has_role(records, configuration, center_role):
        return 0.0
    center_index = _first_role_index(records, configuration, center_role)
    coordination_number = 0.0
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
        coordination_number += UNITY / (UNITY + (distance_m / switch_radius_m) ** exponent)
    return coordination_number


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
    mobility_record: dict,
    site_record: dict,
    stokes_drag_kg_s: float,
    effective_charge_number: float,
    kappa_m_inv: float,
) -> float:
    charge_number = abs(effective_charge_number)
    if charge_number == 0.0:
        return 0.0
    if stokes_drag_kg_s <= 0.0:
        raise ValueError("stokes_drag_kg_s must be positive")
    cloud_radius_m = float(site_record["charge_cloud_radius_m"])
    if cloud_radius_m <= 0.0:
        raise ValueError("charge_cloud_radius_m must be positive")
    screen_argument = kappa_m_inv * cloud_radius_m
    exponent = float(mobility_record["charge_cloud_exponent"])
    if exponent <= 0.0:
        raise ValueError("charge_cloud_exponent must be positive")
    screen_factor = screen_argument * screen_argument / (
        UNITY + screen_argument * screen_argument
    )
    return (
        float(mobility_record["charge_cloud_lambda_kg_m4_s"])
        * charge_number**exponent
        / (cloud_radius_m * cloud_radius_m * cloud_radius_m)
        * screen_factor
    )


def _site_formal_charge_number(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
) -> float:
    species_name = configuration.species_names[site_index]
    return float(records.species_records[species_name]["formal_charge_e"])


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
            abs(float(_site_record(records, configuration, molecule_site_index)["charge_number"]))
            for molecule_site_index in molecule_site_indices
        ],
        dtype=float,
    )
    partial_charge_total = float(np.sum(absolute_partial_charges))
    if partial_charge_total == 0.0:
        raise ValueError("formally charged molecule has no charged force-field sites")
    site_partial_charge = abs(float(_site_record(records, configuration, site_index)["charge_number"]))
    return abs(formal_charge_number) * site_partial_charge / partial_charge_total


def _shape_drag_kg_s(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
    stokes_drag_kg_s: float,
) -> float:
    asymmetry = _molecular_shape_asymmetry(records, configuration, site_index)
    exponent = float(records.mixture_record["mobility"]["shape_exponent"])
    return stokes_drag_kg_s * ((UNITY + asymmetry) ** exponent - UNITY)


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


def _molecular_shape_asymmetry(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
) -> float:
    species_name = configuration.species_names[site_index]
    molecule_id = int(configuration.molecule_ids[site_index])
    site_indices = [
        current_index
        for current_index, current_species_name in enumerate(configuration.species_names)
        if current_species_name == species_name
        and int(configuration.molecule_ids[current_index]) == molecule_id
    ]
    if len(site_indices) < 2:
        return 0.0
    positions = np.asarray(configuration.positions_m[site_indices], dtype=float)
    centered_positions = positions - np.mean(positions, axis=0)
    covariance = centered_positions.T @ centered_positions / float(len(site_indices))
    steric_radii_m = np.asarray(
        [
            float(_site_record(records, configuration, current_index)["steric_radius_m"])
            for current_index in site_indices
        ],
        dtype=float,
    )
    if np.any(steric_radii_m <= 0.0):
        raise ValueError("steric_radius_m must be positive")
    mean_steric_variance_m2 = float(np.mean(steric_radii_m * steric_radii_m))
    covariance += (
        mean_steric_variance_m2
        / float(CARTESIAN_DIMENSION)
        * np.eye(CARTESIAN_DIMENSION, dtype=float)
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive_eigenvalues = eigenvalues[eigenvalues > ZERO_DISTANCE_TOLERANCE_M**2]
    if positive_eigenvalues.size < 2:
        return 0.0
    return float(math.sqrt(np.max(positive_eigenvalues) / np.min(positive_eigenvalues)) - UNITY)


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
