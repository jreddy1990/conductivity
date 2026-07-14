"""Mixture closure builders for physical conductivity objects."""

from __future__ import annotations

from dataclasses import dataclass
import math

from constants import EPS_0, F, N_A, R
from conductivity.physical_library.library_io import PhysicalLibraryRecords

IONIC_STRENGTH_PREFACTOR = 0.5  # Explicit constant: ionic strength is one half sum z_i^2 c_i.
UNORDERED_PAIR_PREFACTOR = 0.5  # Each symmetric unlike-additive encounter is one unordered pair.


@dataclass(frozen=True)
class MixtureComposition:
    solvent_volume_fractions: dict[str, float]
    ion_concentrations_mol_m3: dict[str, float]
    additive_weight_fractions: dict[str, float]


@dataclass(frozen=True)
class MixtureClosureResult:
    dielectric_constant: float
    viscosity_Pa_s: float
    ionic_strength_mol_m3: float
    debye_kappa_m_inv: float
    debye_length_m: float
    additive_collision_exposure: float


def compute_mixture_closures(
    records: PhysicalLibraryRecords,
    composition: MixtureComposition,
    temperature_K: float,
) -> MixtureClosureResult:
    """Compute dielectric, viscosity, ionic strength, and screening from records."""

    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    dielectric_constant = compute_bulk_dielectric_constant(records, composition)
    viscosity_Pa_s = compute_bulk_viscosity_Pa_s(records, composition)
    ionic_strength_mol_m3 = compute_ionic_strength_mol_m3(records, composition)
    debye_kappa_m_inv = compute_debye_kappa_m_inv(
        dielectric_constant,
        ionic_strength_mol_m3,
        temperature_K,
    )
    return MixtureClosureResult(
        dielectric_constant=dielectric_constant,
        viscosity_Pa_s=viscosity_Pa_s,
        ionic_strength_mol_m3=ionic_strength_mol_m3,
        debye_kappa_m_inv=debye_kappa_m_inv,
        debye_length_m=1.0 / debye_kappa_m_inv,
        additive_collision_exposure=compute_additive_collision_exposure(
            records,
            composition,
        ),
    )


def compute_additive_collision_exposure(
    records: PhysicalLibraryRecords,
    composition: MixtureComposition,
) -> float:
    additive_names = tuple(
        additive_name
        for additive_name, weight_fraction in composition.additive_weight_fractions.items()
        if weight_fraction > 0.0
    )
    reference_density_kg_m3 = float(records.mixture_record["reference_density_kg_m3"])
    additive_concentrations_mol_m3 = {
        additive_name: composition.additive_weight_fractions[additive_name]
        * reference_density_kg_m3
        / float(records.species_records[additive_name]["molecular_weight_kg_mol"])
        for additive_name in additive_names
    }
    collision_exposure = 0.0
    association_cell_radius_m = float(
        records.basis_record["pair_basins"]["r_free_m"]
    )
    for additive_name in additive_names:
        additive_record = records.species_records[additive_name]
        additive_hard_core_volume_m3 = float(
            additive_record["excluded_volume"]["hard_core_volume_m3"]
        )
        additive_coefficient = float(
            additive_record["local_microviscosity_coefficient"]
        )
        collision_exposure += (
            N_A
            * additive_concentrations_mol_m3[additive_name]
            * additive_hard_core_volume_m3
            * additive_coefficient
        )
    for additive_index, additive_name in enumerate(additive_names):
        additive_record = records.species_records[additive_name]
        additive_coefficient = float(
            additive_record["local_microviscosity_coefficient"]
        )
        for other_additive_name in additive_names[additive_index + 1 :]:
            other_additive_record = records.species_records[other_additive_name]
            other_additive_coefficient = float(
                other_additive_record["local_microviscosity_coefficient"]
            )
            additive_encounter_volume_m3 = _additive_encounter_volume_m3(
                additive_record,
                association_cell_radius_m,
            )
            other_additive_encounter_volume_m3 = _additive_encounter_volume_m3(
                other_additive_record,
                association_cell_radius_m,
            )
            collision_exposure += UNORDERED_PAIR_PREFACTOR * N_A * (
                composition.additive_weight_fractions[additive_name]
                * additive_coefficient
                * additive_concentrations_mol_m3[other_additive_name]
                * other_additive_encounter_volume_m3
                + composition.additive_weight_fractions[other_additive_name]
                * other_additive_coefficient
                * additive_concentrations_mol_m3[additive_name]
                * additive_encounter_volume_m3
            )
    return collision_exposure


def _additive_encounter_volume_m3(
    additive_record: dict,
    association_cell_radius_m: float,
) -> float:
    additive_radius_m = float(
        additive_record["excluded_volume"]["packing_radius_m"]
    )
    return (
        4.0
        * math.pi
        * (association_cell_radius_m + additive_radius_m) ** 3
        / 3.0
    )


def compute_bulk_dielectric_constant(
    records: PhysicalLibraryRecords,
    composition: MixtureComposition,
) -> float:
    """Return the solvent-matrix dielectric before pointwise local corrections."""

    dielectric_constant = 0.0
    volume_fraction_sum = 0.0
    for solvent_name, volume_fraction in composition.solvent_volume_fractions.items():
        species_record = records.species_records[solvent_name]
        dielectric_constant += volume_fraction * float(species_record["dielectric_constant"])
        volume_fraction_sum += volume_fraction
    if volume_fraction_sum <= 0.0:
        raise ValueError("solvent volume fractions must contain positive total volume")
    dielectric_constant /= volume_fraction_sum
    if dielectric_constant <= 0.0:
        raise ValueError("computed dielectric_constant must be positive")
    return dielectric_constant


def compute_bulk_viscosity_Pa_s(
    records: PhysicalLibraryRecords,
    composition: MixtureComposition,
) -> float:
    """Return the solvent-matrix viscosity before pointwise local corrections."""

    reciprocal_viscosity = 0.0
    volume_fraction_sum = 0.0
    for solvent_name, volume_fraction in composition.solvent_volume_fractions.items():
        species_record = records.species_records[solvent_name]
        viscosity_Pa_s = float(species_record["viscosity_Pa_s"])
        if viscosity_Pa_s <= 0.0:
            raise ValueError(f"{solvent_name}.viscosity_Pa_s must be positive")
        reciprocal_viscosity += volume_fraction / viscosity_Pa_s
        volume_fraction_sum += volume_fraction
    if reciprocal_viscosity <= 0.0 or volume_fraction_sum <= 0.0:
        raise ValueError("solvent volume fractions must define positive viscosity")
    return volume_fraction_sum / reciprocal_viscosity


def compute_ionic_strength_mol_m3(
    records: PhysicalLibraryRecords,
    composition: MixtureComposition,
) -> float:
    ionic_strength_mol_m3 = 0.0
    for species_name, concentration_mol_m3 in composition.ion_concentrations_mol_m3.items():
        if concentration_mol_m3 < 0.0:
            raise ValueError(f"{species_name} concentration must be nonnegative")
        species_record = records.species_records[species_name]
        charge_number = float(species_record["formal_charge_e"])
        ionic_strength_mol_m3 += (
            IONIC_STRENGTH_PREFACTOR
            * charge_number
            * charge_number
            * concentration_mol_m3
        )
    return ionic_strength_mol_m3


def compute_debye_kappa_m_inv(
    dielectric_constant: float,
    ionic_strength_mol_m3: float,
    temperature_K: float,
) -> float:
    if dielectric_constant <= 0.0:
        raise ValueError("dielectric_constant must be positive")
    if ionic_strength_mol_m3 <= 0.0:
        raise ValueError("ionic_strength_mol_m3 must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    kappa_squared_m2 = (
        F
        * F
        * 2.0
        * ionic_strength_mol_m3
        / (EPS_0 * dielectric_constant * R * temperature_K)
    )
    return math.sqrt(kappa_squared_m2)
