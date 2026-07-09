"""Deterministic basin and feature builders for physical conductivity states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from conductivity.physical_library.mixture_closures import MixtureClosureResult
from conductivity.physical_library.physical_objects import (
    PairBasin,
    SiteConfiguration,
    assign_pair_basin,
    compute_local_packing_fraction,
)
from conductivity.physical_library.library_io import PhysicalLibraryRecords

Array = np.ndarray

COORDINATION_SWITCH_NUMERATOR = 1.0  # Explicit constant: logistic switch numerator.
LOW_BIN_INDEX = 0
MEDIUM_BIN_INDEX = 1
HIGH_BIN_INDEX = 2


class OrientationBasin(Enum):
    RADIAL = "radial"
    TANGENTIAL = "tangential"
    BRIDGING = "bridging"
    UNASSIGNED = "unassigned"


@dataclass(frozen=True)
class BasinFeatureVector:
    pair_basin: PairBasin
    lithium_solvent_coordination: float
    lithium_ligand_coordination: float
    lithium_anion_coordination: float
    orientation_basin: OrientationBasin
    packing_bin: int
    ionic_strength_bin: int
    dielectric_bin: int
    viscosity_bin: int


@dataclass(frozen=True)
class StateDefinition:
    state_key: tuple[str, ...]
    features: BasinFeatureVector
    stoichiometry: Array


def build_state_definition(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    mixture_closure: MixtureClosureResult,
    component_names: tuple[str, ...],
) -> StateDefinition:
    """Assign one configuration to a sparse state key and component stoichiometry."""

    features = compute_basin_features(records, configuration, mixture_closure)
    state_key = (
        features.pair_basin.value,
        features.orientation_basin.value,
        _bin_label(features.packing_bin),
        _bin_label(features.ionic_strength_bin),
        _bin_label(features.dielectric_bin),
        _bin_label(features.viscosity_bin),
    )
    return StateDefinition(
        state_key=state_key,
        features=features,
        stoichiometry=compute_component_stoichiometry(
            configuration,
            component_names,
        ),
    )


def compute_basin_features(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    mixture_closure: MixtureClosureResult,
) -> BasinFeatureVector:
    lithium_index = _first_site_index_with_role(records, configuration, "cation")
    anion_index = _first_site_index_with_role(records, configuration, "anion")
    lithium_anion_distance_m = float(
        np.linalg.norm(
            np.asarray(configuration.positions_m[lithium_index], dtype=float)
            - np.asarray(configuration.positions_m[anion_index], dtype=float)
        )
    )
    local_packing_fraction = compute_local_packing_fraction(records, configuration)
    return BasinFeatureVector(
        pair_basin=assign_pair_basin(
            lithium_anion_distance_m,
            records.basis_record,
        ),
        lithium_solvent_coordination=compute_role_coordination_number(
            records,
            configuration,
            center_role="cation",
            ligand_role="solvent",
            switch_name="Li_solvent",
        ),
        lithium_ligand_coordination=compute_role_coordination_number(
            records,
            configuration,
            center_role="cation",
            ligand_role="additive",
            switch_name="Li_ligand",
        ),
        lithium_anion_coordination=compute_role_coordination_number(
            records,
            configuration,
            center_role="cation",
            ligand_role="anion",
            switch_name="Li_anion",
        ),
        orientation_basin=assign_orientation_basin(
            records,
            configuration,
            lithium_index,
            anion_index,
        ),
        packing_bin=assign_threshold_bin(
            local_packing_fraction,
            records.basis_record["environment_bins"]["packing_fraction"],
        ),
        ionic_strength_bin=assign_threshold_bin(
            mixture_closure.ionic_strength_mol_m3,
            records.basis_record["environment_bins"]["ionic_strength_mol_m3"],
        ),
        dielectric_bin=assign_threshold_bin(
            mixture_closure.dielectric_constant,
            records.basis_record["environment_bins"]["dielectric"],
        ),
        viscosity_bin=assign_threshold_bin(
            mixture_closure.viscosity_Pa_s,
            records.basis_record["environment_bins"]["viscosity_Pa_s"],
        ),
    )


def compute_role_coordination_number(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    center_role: str,
    ligand_role: str,
    switch_name: str,
) -> float:
    switch_record = records.basis_record["coordination_switches"][switch_name]
    switch_radius_m = float(switch_record["r0_m"])
    exponent = float(switch_record["exponent"])
    center_index = _first_site_index_with_role(records, configuration, center_role)
    coordination_number = 0.0
    for site_index, species_name in enumerate(configuration.species_names):
        if site_index == center_index:
            continue
        if records.species_records[species_name]["role"] != ligand_role:
            continue
        distance_m = float(
            np.linalg.norm(
                np.asarray(configuration.positions_m[center_index], dtype=float)
                - np.asarray(configuration.positions_m[site_index], dtype=float)
            )
        )
        if distance_m <= 0.0:
            raise ValueError("coordination distance must be positive")
        coordination_number += COORDINATION_SWITCH_NUMERATOR / (
            COORDINATION_SWITCH_NUMERATOR + (distance_m / switch_radius_m) ** exponent
        )
    return coordination_number


def assign_orientation_basin(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    lithium_index: int,
    anion_index: int,
) -> OrientationBasin:
    bins = records.basis_record["orientation_bins"]
    anion_vector = _species_orientation_vector(records, configuration, anion_index)
    lithium_to_anion = (
        np.asarray(configuration.positions_m[anion_index], dtype=float)
        - np.asarray(configuration.positions_m[lithium_index], dtype=float)
    )
    anion_norm = float(np.linalg.norm(anion_vector))
    pair_norm = float(np.linalg.norm(lithium_to_anion))
    if anion_norm <= 0.0 or pair_norm <= 0.0:
        return OrientationBasin.UNASSIGNED
    orientation_cosine = float(np.dot(anion_vector, lithium_to_anion) / (anion_norm * pair_norm))
    if orientation_cosine >= float(bins["radial_min"]):
        return OrientationBasin.RADIAL
    if abs(orientation_cosine) <= float(bins["tangential_abs_max"]):
        return OrientationBasin.TANGENTIAL
    if orientation_cosine <= float(bins["bridging_max"]):
        return OrientationBasin.BRIDGING
    return OrientationBasin.UNASSIGNED


def assign_threshold_bin(value: float, thresholds: list[float]) -> int:
    if len(thresholds) != 2:
        raise ValueError("environment bin thresholds must contain two values")
    first_threshold = float(thresholds[0])
    second_threshold = float(thresholds[1])
    if not first_threshold < second_threshold:
        raise ValueError("environment bin thresholds must be increasing")
    if value < first_threshold:
        return LOW_BIN_INDEX
    if value < second_threshold:
        return MEDIUM_BIN_INDEX
    return HIGH_BIN_INDEX


def compute_component_stoichiometry(
    configuration: SiteConfiguration,
    component_names: tuple[str, ...],
) -> Array:
    stoichiometry = np.zeros(len(component_names), dtype=float)
    molecule_keys = set()
    for species_name, molecule_id in zip(configuration.species_names, configuration.molecule_ids):
        molecule_key = (species_name, int(molecule_id))
        if molecule_key in molecule_keys:
            continue
        molecule_keys.add(molecule_key)
        if species_name in component_names:
            stoichiometry[component_names.index(species_name)] += 1.0
    return stoichiometry


def _first_site_index_with_role(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: str,
) -> int:
    for site_index, species_name in enumerate(configuration.species_names):
        if records.species_records[species_name]["role"] == role:
            return site_index
    raise ValueError(f"configuration has no species with role {role}")


def _species_orientation_vector(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
) -> Array:
    species_name = configuration.species_names[site_index]
    molecule_id = int(configuration.molecule_ids[site_index])
    site_indices = [
        current_index
        for current_index, current_species_name in enumerate(configuration.species_names)
        if current_species_name == species_name
        and int(configuration.molecule_ids[current_index]) == molecule_id
    ]
    positions = np.asarray(configuration.positions_m[site_indices], dtype=float)
    charges = np.asarray(
        [
            _charge_number_for_site(
                records,
                species_name,
                int(configuration.site_ids[current_index]),
            )
            for current_index in site_indices
        ],
        dtype=float,
    )
    charge_magnitude_sum = float(np.sum(np.abs(charges)))
    if charge_magnitude_sum <= 0.0:
        return np.zeros(positions.shape[1], dtype=float)
    weighted_center = np.sum(positions * np.abs(charges)[:, None], axis=0) / charge_magnitude_sum
    geometric_center = np.mean(positions, axis=0)
    return weighted_center - geometric_center


def _charge_number_for_site(
    records: PhysicalLibraryRecords,
    species_name: str,
    site_id: int,
) -> float:
    for site_record in records.species_records[species_name]["sites"]:
        if int(site_record["site_id"]) == site_id:
            return float(site_record["charge_number"])
    raise KeyError(f"{species_name} has no site_id {site_id}")


def _bin_label(bin_index: int) -> str:
    if bin_index == LOW_BIN_INDEX:
        return "low"
    if bin_index == MEDIUM_BIN_INDEX:
        return "medium"
    if bin_index == HIGH_BIN_INDEX:
        return "high"
    raise ValueError(f"unsupported bin index {bin_index}")
