"""Physical-library YAML, recipe, and validation I/O for conductivity."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from constants import MOL_M3_PER_MOL_L
from data.electrolyte_property_db import DATA

SPECIES_REQUIRED_FIELDS = (
    "schema",
    "name",
    "role",
    "source",
    "formal_charge_e",
    "molecular_weight_kg_mol",
    "density_kg_m3",
    "partial_molar_volume_m3_mol",
    "sites",
    "bonds",
    "angles",
    "torsions",
    "constraints",
    "reference_conformer_coordinates_m",
    "coordinate_map",
)

SITE_REQUIRED_FIELDS = (
    "site_id",
    "element",
    "atomic_number",
    "mass_kg",
    "steric_radius_m",
    "hydrodynamic_radius_m",
    "volume_m3",
    "lj_sigma_m",
    "lj_epsilon_J",
    "charge_number",
    "charge_cloud_radius_m",
    "born_radius_m",
    "polarizability_SI",
    "donor_flag",
    "acceptor_flag",
    "hba_count_contribution",
    "hbd_count_contribution",
    "parameter_ids",
)

PAIR_REQUIRED_FIELDS = (
    "schema",
    "species_i",
    "species_j",
    "source",
    "lj_mixing_policy",
    "coulomb_policy",
    "site_pair_count",
)

TOP_LEVEL_REQUIRED_FILES = (
    "manifest.yaml",
    "mixture.yaml",
    "basis.yaml",
    "transitions.yaml",
    "memory.yaml",
)


@dataclass(frozen=True)
class PhysicalLibraryRecords:
    root: Path
    manifest: dict
    species_records: dict
    pair_records: dict
    mixture_record: dict
    basis_record: dict
    transition_record: dict
    memory_record: dict


@dataclass(frozen=True)
class RecipeComponentLoading:
    name: str
    concentration_mol_m3: float
    role: str


@dataclass(frozen=True)
class RecipeBuildResult:
    temperature_K: float
    components: tuple[RecipeComponentLoading, ...]
    solvent_volume_fractions: dict[str, float]
    additive_weight_fractions: dict[str, float]
    library_records: PhysicalLibraryRecords


def load_physical_library(root: Path) -> PhysicalLibraryRecords:
    library_root = Path(root)
    _validate_required_files_exist(library_root)
    manifest = _load_yaml_mapping(library_root / "manifest.yaml")
    records = PhysicalLibraryRecords(
        root=library_root,
        manifest=manifest,
        species_records=_load_species_records(library_root, manifest),
        pair_records=_load_pair_records(library_root),
        mixture_record=_load_yaml_mapping(library_root / "mixture.yaml"),
        basis_record=_load_yaml_mapping(library_root / "basis.yaml"),
        transition_record=_load_yaml_mapping(library_root / "transitions.yaml"),
        memory_record=_load_yaml_mapping(library_root / "memory.yaml"),
    )
    validate_physical_library_records(records)
    return records


def load_required_species_records(
    library_root: Path,
    required_species_names: tuple[str, ...],
) -> PhysicalLibraryRecords:
    records = load_physical_library(library_root)
    missing_species = [
        species_name
        for species_name in required_species_names
        if species_name not in records.species_records
    ]
    if missing_species:
        raise KeyError(f"physical library missing recipe species: {missing_species}")
    return records


def build_recipe_library_context(
    recipe_yaml_path: Path,
    library_root: Path,
) -> RecipeBuildResult:
    recipe_record = _load_recipe_mapping(recipe_yaml_path)
    library_records = load_physical_library(library_root)
    temperature_K = _positive_float(recipe_record["temperature_K"], "temperature_K")
    solvent_volume_fractions = _positive_float_mapping(
        recipe_record["solvents_vv"],
        "solvents_vv",
    )
    salt_molarities_mol_l = _positive_float_mapping(
        recipe_record["salts_mol_l"],
        "salts_mol_l",
    )
    additive_weight_fractions = _nonnegative_float_mapping(
        recipe_record["additives_weight_fraction"],
        "additives_weight_fraction",
    )
    for solvent_name in solvent_volume_fractions:
        _require_species(solvent_name, library_records)
    for salt_component_name in salt_molarities_mol_l:
        _require_species(salt_component_name, library_records)
    for additive_name in additive_weight_fractions:
        _require_species(additive_name, library_records)

    solvent_loadings = tuple(
        RecipeComponentLoading(
            name=solvent_name,
            concentration_mol_m3=_solvent_volume_fraction_to_concentration(
                solvent_name,
                volume_fraction,
                library_records,
            ),
            role="solvent",
        )
        for solvent_name, volume_fraction in solvent_volume_fractions.items()
    )
    salt_loadings = tuple(
        RecipeComponentLoading(
            name=salt_component_name,
            concentration_mol_m3=concentration_mol_l * MOL_M3_PER_MOL_L,
            role="salt_component",
        )
        for salt_component_name, concentration_mol_l in salt_molarities_mol_l.items()
    )
    additive_loadings = tuple(
        RecipeComponentLoading(
            name=additive_name,
            concentration_mol_m3=_additive_weight_fraction_to_concentration(
                additive_name,
                weight_fraction,
                library_records,
            ),
            role="additive",
        )
        for additive_name, weight_fraction in additive_weight_fractions.items()
    )
    return RecipeBuildResult(
        temperature_K=temperature_K,
        components=solvent_loadings + salt_loadings + additive_loadings,
        solvent_volume_fractions=solvent_volume_fractions,
        additive_weight_fractions=additive_weight_fractions,
        library_records=library_records,
    )


def validate_physical_library_records(records: PhysicalLibraryRecords) -> None:
    _validate_manifest(records.manifest)
    for species_name, species_record in records.species_records.items():
        _validate_species_record(species_name, species_record)
    for pair_key, pair_record in records.pair_records.items():
        _validate_pair_record(pair_key, pair_record, records.species_records)
    _require_mapping_keys(
        records.mixture_record,
        ("schema", "species", "dielectric", "viscosity", "packing", "atmosphere"),
        "mixture.yaml",
    )
    _require_mapping_keys(
        records.basis_record,
        ("schema", "coordinates", "threshold_source"),
        "basis.yaml",
    )
    _require_mapping_keys(
        records.transition_record,
        ("schema", "families", "numerical_method"),
        "transitions.yaml",
    )
    _require_mapping_keys(
        records.memory_record,
        ("schema", "families", "orthogonalization"),
        "memory.yaml",
    )


def validate_projected_property_db() -> dict[str, int]:
    source_labeled_rows = 0
    evaluated_rows = 0
    failed_rows = 0
    for row in DATA:
        properties = row["properties"]
        if "conductivity_mS_cm" not in properties:
            continue
        source_labeled_rows += 1
        if _has_projected_inputs(properties):
            evaluated_rows += 1
            continue
        failed_rows += 1
    return {
        "source_labeled_rows": source_labeled_rows,
        "evaluated_rows": evaluated_rows,
        "failed_rows": failed_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical-library validation tools.")
    parser.add_argument(
        "command",
        choices=("validate-library", "validate-property-db"),
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path("conductivity/physical_library"),
    )
    parsed_arguments = parser.parse_args()
    if parsed_arguments.command == "validate-library":
        return _main_validate_library(parsed_arguments.library_root)
    if parsed_arguments.command == "validate-property-db":
        return _main_validate_property_db()
    raise ValueError(f"unsupported command {parsed_arguments.command}")


def _main_validate_library(library_root: Path) -> int:
    records = load_physical_library(library_root)
    print(f"library_root={records.root}")
    print(f"species_count={len(records.species_records)}")
    print(f"pair_count={len(records.pair_records)}")
    print("species=" + ",".join(sorted(records.species_records)))
    return 0


def _main_validate_property_db() -> int:
    result = validate_projected_property_db()
    print(f"source_labeled_rows={result['source_labeled_rows']}")
    print(f"evaluated_rows={result['evaluated_rows']}")
    print(f"failed_rows={result['failed_rows']}")
    if result["failed_rows"] > 0:
        print("missing executable physical generator inputs")
    return 0


def _validate_required_files_exist(library_root: Path) -> None:
    for relative_path in TOP_LEVEL_REQUIRED_FILES:
        full_path = library_root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"physical library is missing {full_path}")


def _load_yaml_mapping(path: Path) -> dict:
    record = yaml.safe_load(path.read_text())
    if not isinstance(record, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return record


def _load_species_records(library_root: Path, manifest: dict) -> dict:
    _require_mapping_keys(manifest, ("species", "species_records"), "manifest.yaml")
    species_records = {}
    for relative_path in manifest["species_records"]:
        species_path = library_root / relative_path
        if not species_path.exists():
            raise FileNotFoundError(f"manifest references missing species file {species_path}")
        species_record = _load_yaml_mapping(species_path)
        species_records[species_record["name"]] = species_record
    return species_records


def _load_pair_records(library_root: Path) -> dict:
    pair_directory = library_root / "pairs"
    if not pair_directory.exists():
        raise FileNotFoundError(f"physical library is missing pair directory {pair_directory}")
    pair_records = {}
    for pair_path in sorted(pair_directory.glob("*.yaml")):
        pair_record = _load_yaml_mapping(pair_path)
        pair_key = f"{pair_record['species_i']}::{pair_record['species_j']}"
        pair_records[pair_key] = pair_record
    if not pair_records:
        raise ValueError(f"physical library pair directory has no pair records: {pair_directory}")
    return pair_records


def _validate_manifest(manifest: dict) -> None:
    _require_mapping_keys(
        manifest,
        (
            "schema",
            "force_field_source",
            "species",
            "species_records",
            "pair_records_glob",
            "mixture_record",
            "basis_record",
            "transition_record",
            "memory_record",
        ),
        "manifest.yaml",
    )
    if len(manifest["species"]) != len(manifest["species_records"]):
        raise ValueError("manifest species and species_records length mismatch")


def _validate_species_record(species_name: str, species_record: dict) -> None:
    _require_mapping_keys(species_record, SPECIES_REQUIRED_FIELDS, species_name)
    if species_name != species_record["name"]:
        raise ValueError(f"species key {species_name} does not match record name")
    if not species_record["sites"]:
        raise ValueError(f"{species_name} has no sites")
    total_charge = 0.0
    for site_record in species_record["sites"]:
        _require_mapping_keys(site_record, SITE_REQUIRED_FIELDS, f"{species_name}.sites")
        total_charge += float(site_record["charge_number"])
    charge_error = abs(total_charge - float(species_record["formal_charge_e"]))
    if charge_error > 1.0e-6:
        raise ValueError(
            f"{species_name} site charges sum to {total_charge}, "
            f"formal charge is {species_record['formal_charge_e']}"
        )
    coordinate_count = len(species_record["reference_conformer_coordinates_m"])
    if coordinate_count != len(species_record["sites"]):
        raise ValueError(f"{species_name} conformer coordinate count does not match sites")


def _validate_pair_record(
    pair_key: str,
    pair_record: dict,
    species_records: dict,
) -> None:
    _require_mapping_keys(pair_record, PAIR_REQUIRED_FIELDS, pair_key)
    if pair_record["species_i"] not in species_records:
        raise ValueError(f"{pair_key} species_i is absent from species records")
    if pair_record["species_j"] not in species_records:
        raise ValueError(f"{pair_key} species_j is absent from species records")
    expected_pair_count = len(species_records[pair_record["species_i"]]["sites"]) * len(
        species_records[pair_record["species_j"]]["sites"]
    )
    if int(pair_record["site_pair_count"]) != expected_pair_count:
        raise ValueError(
            f"{pair_key} site_pair_count {pair_record['site_pair_count']} "
            f"does not match {expected_pair_count}"
        )


def _load_recipe_mapping(path: Path) -> dict:
    record = yaml.safe_load(path.read_text())
    if not isinstance(record, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    required_keys = (
        "temperature_K",
        "solvents_vv",
        "salts_mol_l",
        "additives_weight_fraction",
    )
    for required_key in required_keys:
        if required_key not in record:
            raise KeyError(f"{path} missing required key {required_key}")
    return record


def _solvent_volume_fraction_to_concentration(
    species_name: str,
    volume_fraction: float,
    library_records: PhysicalLibraryRecords,
) -> float:
    species_record = library_records.species_records[species_name]
    density_kg_m3 = _positive_float(
        species_record["density_kg_m3"],
        f"{species_name}.density_kg_m3",
    )
    molecular_weight_kg_mol = _positive_float(
        species_record["molecular_weight_kg_mol"],
        f"{species_name}.molecular_weight_kg_mol",
    )
    return volume_fraction * density_kg_m3 / molecular_weight_kg_mol


def _additive_weight_fraction_to_concentration(
    species_name: str,
    weight_fraction: float,
    library_records: PhysicalLibraryRecords,
) -> float:
    species_record = library_records.species_records[species_name]
    molecular_weight_kg_mol = _positive_float(
        species_record["molecular_weight_kg_mol"],
        f"{species_name}.molecular_weight_kg_mol",
    )
    reference_density_kg_m3 = _positive_float(
        library_records.mixture_record["reference_density_kg_m3"],
        "mixture.reference_density_kg_m3",
    )
    return weight_fraction * reference_density_kg_m3 / molecular_weight_kg_mol


def _require_species(
    species_name: str,
    library_records: PhysicalLibraryRecords,
) -> None:
    if species_name not in library_records.species_records:
        raise KeyError(f"recipe species {species_name} is absent from physical library")


def _positive_float(value: float, label: str) -> float:
    numeric_value = float(value)
    if numeric_value <= 0.0:
        raise ValueError(f"{label} must be positive")
    return numeric_value


def _positive_float_mapping(value: dict, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    converted = {}
    for key, numeric_value in value.items():
        converted[str(key)] = _positive_float(numeric_value, f"{label}.{key}")
    return converted


def _nonnegative_float_mapping(value: dict, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    converted = {}
    for key, numeric_value in value.items():
        converted_value = float(numeric_value)
        if converted_value < 0.0:
            raise ValueError(f"{label}.{key} must be nonnegative")
        converted[str(key)] = converted_value
    return converted


def _require_mapping_keys(record: dict, required_keys: tuple[str, ...], label: str) -> None:
    missing = [required_key for required_key in required_keys if required_key not in record]
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")


def _has_projected_inputs(properties: dict) -> bool:
    if "projected_primitives" in properties:
        return True
    if "projected_generator_inputs" in properties:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
