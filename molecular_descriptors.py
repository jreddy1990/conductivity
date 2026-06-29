"""Molecular descriptor inputs for arbitrary liquid-electrolyte conductivity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol


ROLE_CATION = "cation"
ROLE_ANION = "anion"
ROLE_SOLVENT = "solvent"
ROLE_ADDITIVE = "additive"
ROLE_NEUTRAL = "neutral"
SUPPORTED_MOLECULAR_SPECIES_ROLES = (
    ROLE_CATION,
    ROLE_ANION,
    ROLE_SOLVENT,
    ROLE_ADDITIVE,
    ROLE_NEUTRAL,
)


@dataclass(frozen=True)
class MolecularSpeciesInput:
    name: str
    role: str
    charge_number: int
    smiles: str
    xyz_coordinates: tuple[tuple[str, float, float, float], ...]
    property_overrides: Mapping[str, float]
    coordination_sites: tuple[str, ...]


@dataclass(frozen=True)
class MolecularSpeciesDescriptor:
    name: str
    role: str
    charge_number: int
    molecular_weight_g_mol: float
    hard_sphere_radius_A: float
    hydrodynamic_radius_A: float
    cavity_radius_A: float
    charge_cloud_radius_A: float
    molecular_volume_A3: float
    solvent_accessible_area_A2: float
    dipole_D: float
    quadrupole_D_A: float
    polarizability_A3: float
    donor_number: float
    acceptor_number: float
    hbond_donor_count: int
    hbond_acceptor_count: int
    epsilon_r_pure: float
    viscosity_cP_pure: float
    density_g_ml: float
    born_solvation_radius_A: float
    coordination_sites: tuple[str, ...]
    coordination_affinity_J_mol: float
    ligand_field_asymmetry: float


class MolecularDescriptorBackend(Protocol):
    def describe_species(
        self,
        species: MolecularSpeciesInput,
        temperature_K: float,
    ) -> MolecularSpeciesDescriptor:
        ...


class ProvidedPropertyDescriptorBackend:
    """Build descriptors only from user-supplied molecular property values."""

    def describe_species(
        self,
        species: MolecularSpeciesInput,
        temperature_K: float,
    ) -> MolecularSpeciesDescriptor:
        _validate_species_identity(species)
        _positive_float(temperature_K, "temperature_K")
        properties = species.property_overrides
        return MolecularSpeciesDescriptor(
            name=species.name,
            role=species.role,
            charge_number=species.charge_number,
            molecular_weight_g_mol=_required_positive_property(
                properties,
                "molecular_weight_g_mol",
                species.name,
            ),
            hard_sphere_radius_A=_required_positive_property(
                properties,
                "hard_sphere_radius_A",
                species.name,
            ),
            hydrodynamic_radius_A=_required_positive_property(
                properties,
                "hydrodynamic_radius_A",
                species.name,
            ),
            cavity_radius_A=_required_positive_property(
                properties,
                "cavity_radius_A",
                species.name,
            ),
            charge_cloud_radius_A=_required_positive_property(
                properties,
                "charge_cloud_radius_A",
                species.name,
            ),
            molecular_volume_A3=_required_positive_property(
                properties,
                "molecular_volume_A3",
                species.name,
            ),
            solvent_accessible_area_A2=_required_nonnegative_property(
                properties,
                "solvent_accessible_area_A2",
                species.name,
            ),
            dipole_D=_required_nonnegative_property(
                properties,
                "dipole_D",
                species.name,
            ),
            quadrupole_D_A=_required_nonnegative_property(
                properties,
                "quadrupole_D_A",
                species.name,
            ),
            polarizability_A3=_required_nonnegative_property(
                properties,
                "polarizability_A3",
                species.name,
            ),
            donor_number=_required_nonnegative_property(
                properties,
                "donor_number",
                species.name,
            ),
            acceptor_number=_required_nonnegative_property(
                properties,
                "acceptor_number",
                species.name,
            ),
            hbond_donor_count=_required_nonnegative_integer_property(
                properties,
                "hbond_donor_count",
                species.name,
            ),
            hbond_acceptor_count=_required_nonnegative_integer_property(
                properties,
                "hbond_acceptor_count",
                species.name,
            ),
            epsilon_r_pure=_required_positive_property(
                properties,
                "epsilon_r_pure",
                species.name,
            ),
            viscosity_cP_pure=_required_positive_property(
                properties,
                "viscosity_cP_pure",
                species.name,
            ),
            density_g_ml=_required_positive_property(
                properties,
                "density_g_ml",
                species.name,
            ),
            born_solvation_radius_A=_required_positive_property(
                properties,
                "born_solvation_radius_A",
                species.name,
            ),
            coordination_sites=tuple(species.coordination_sites),
            coordination_affinity_J_mol=_required_nonnegative_property(
                properties,
                "coordination_affinity_J_mol",
                species.name,
            ),
            ligand_field_asymmetry=_required_positive_property(
                properties,
                "ligand_field_asymmetry",
                species.name,
            ),
        )


def _validate_species_identity(species: MolecularSpeciesInput) -> None:
    if species.name == "":
        raise ValueError("species name must be nonempty")
    if species.role not in SUPPORTED_MOLECULAR_SPECIES_ROLES:
        raise ValueError(
            f"species {species.name} has unsupported role {species.role!r}"
        )
    if not isinstance(species.charge_number, int):
        raise TypeError(f"species {species.name} charge_number must be an integer")
    if species.role == ROLE_CATION and species.charge_number <= 0:
        raise ValueError(f"cation {species.name} must have positive charge_number")
    if species.role == ROLE_ANION and species.charge_number >= 0:
        raise ValueError(f"anion {species.name} must have negative charge_number")


def _required_positive_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    return _positive_float(
        _required_numeric_property(properties, key, species_name),
        f"{species_name}.{key}",
    )


def _required_nonnegative_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    return _nonnegative_float(
        _required_numeric_property(properties, key, species_name),
        f"{species_name}.{key}",
    )


def _required_nonnegative_integer_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> int:
    value = _required_nonnegative_property(properties, key, species_name)
    integer_value = int(value)
    if float(integer_value) != value:
        raise ValueError(f"{species_name}.{key} must be an integer-valued number")
    return integer_value


def _required_numeric_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    if key not in properties:
        raise ValueError(f"species {species_name} missing molecular descriptor {key}")
    value = properties[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"species {species_name} descriptor {key} must be numeric")
    return float(value)


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _nonnegative_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        raise ValueError(f"{context} must be nonnegative and finite")
    return parsed_value
