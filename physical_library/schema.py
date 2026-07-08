"""Schema validation for the conductivity physical-library records."""

from __future__ import annotations

from pathlib import Path

import yaml


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


class PhysicalLibraryRecords:
    def __init__(
        self,
        root,
        manifest,
        species_records,
        pair_records,
        mixture_record,
        basis_record,
        transition_record,
        memory_record,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.species_records = species_records
        self.pair_records = pair_records
        self.mixture_record = mixture_record
        self.basis_record = basis_record
        self.transition_record = transition_record
        self.memory_record = memory_record


def load_physical_library(root) -> PhysicalLibraryRecords:
    library_root = Path(root)
    _validate_required_files_exist(library_root)
    manifest = _load_yaml_mapping(library_root / "manifest.yaml")
    species_records = _load_species_records(library_root, manifest)
    pair_records = _load_pair_records(library_root)
    records = PhysicalLibraryRecords(
        root=library_root,
        manifest=manifest,
        species_records=species_records,
        pair_records=pair_records,
        mixture_record=_load_yaml_mapping(library_root / "mixture.yaml"),
        basis_record=_load_yaml_mapping(library_root / "basis.yaml"),
        transition_record=_load_yaml_mapping(library_root / "transitions.yaml"),
        memory_record=_load_yaml_mapping(library_root / "memory.yaml"),
    )
    validate_physical_library_records(records)
    return records


def validate_physical_library_records(records) -> None:
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


def _validate_required_files_exist(library_root) -> None:
    for relative_path in TOP_LEVEL_REQUIRED_FILES:
        full_path = library_root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"physical library is missing {full_path}")


def _load_yaml_mapping(path):
    record = yaml.safe_load(path.read_text())
    if not isinstance(record, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return record


def _load_species_records(library_root, manifest):
    _require_mapping_keys(manifest, ("species", "species_records"), "manifest.yaml")
    species_records = {}
    for relative_path in manifest["species_records"]:
        species_path = library_root / relative_path
        if not species_path.exists():
            raise FileNotFoundError(f"manifest references missing species file {species_path}")
        species_record = _load_yaml_mapping(species_path)
        species_records[species_record["name"]] = species_record
    return species_records


def _load_pair_records(library_root):
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


def _validate_manifest(manifest) -> None:
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


def _validate_species_record(species_name, species_record) -> None:
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
    if len(species_record["reference_conformer_coordinates_m"]) != len(
        species_record["sites"]
    ):
        raise ValueError(f"{species_name} conformer coordinate count does not match sites")


def _validate_pair_record(pair_key, pair_record, species_records) -> None:
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


def _require_mapping_keys(record, required_keys, label) -> None:
    missing = [required_key for required_key in required_keys if required_key not in record]
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")
