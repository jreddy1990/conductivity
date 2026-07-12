"""Physical-library YAML, recipe, and validation I/O for conductivity."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from constants import MOL_M3_PER_MOL_L, R, T_REF_K
from conductivity.physical_library.speciation_equilibrium import (
    SpeciationEquilibriumResult,
    solve_speciation_equilibrium,
)

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
    "association.yaml",
    "equilibria.yaml",
)
FIRST_PRINCIPLES_ALLOWED_PARAMETER_PROVENANCE = (
    "universal_constant",
    "geometry_derived",
    "measured_pure_property",
    "measured_mixture_property",
    "continuum_theory",
    "fitted_to_primitive",
    "initialized_estimate",
)

ASSOCIATION_INITIALIZATION_MULTIPLIERS = {
    ("pair", "CIP"): -0.25,  # Plan-defined quarter-RT favorable CIP residual.
    ("pair", "SSIP"): -0.125,  # Plan-defined eighth-RT favorable SSIP residual.
    ("pair", "addSSIP"): -0.1875,  # Plan-defined three-sixteenths-RT addSSIP residual.
    ("cluster", "Li2A_positive"): 0.5,
    ("cluster", "LiA2_negative"): 0.5,
    ("cluster", "Li2A2_neutral"): 1.0,
    ("cluster", "bridge_network"): 5.0 / 4.0,
}
FIRST_PRINCIPLES_REJECTED_PARAMETER_PROVENANCE = "fitted_to_scalar_sigma"


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
    association_record: dict
    equilibria_record: dict


@dataclass(frozen=True)
class RecipeComponentLoading:
    name: str
    concentration_mol_m3: float
    role: str


@dataclass(frozen=True)
class RecipeBuildResult:
    temperature_K: float
    conserved_components: tuple[RecipeComponentLoading, ...]
    resolved_species: tuple[RecipeComponentLoading, ...]
    speciation_equilibrium: SpeciationEquilibriumResult
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
        association_record=_load_yaml_mapping(library_root / "association.yaml"),
        equilibria_record=_load_yaml_mapping(library_root / "equilibria.yaml"),
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
    return build_recipe_library_context_from_record(recipe_record, library_root)


def build_recipe_library_context_from_record(
    recipe_record: dict,
    library_root: Path,
) -> RecipeBuildResult:
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
        for solvent_name, volume_fraction in sorted(
            solvent_volume_fractions.items()
        )
    )
    salt_loadings = tuple(
        RecipeComponentLoading(
            name=salt_component_name,
            concentration_mol_m3=concentration_mol_l * MOL_M3_PER_MOL_L,
            role="salt_component",
        )
        for salt_component_name, concentration_mol_l in sorted(
            salt_molarities_mol_l.items(),
            key=_salt_component_sort_key(library_records),
        )
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
        for additive_name, weight_fraction in sorted(additive_weight_fractions.items())
    )
    conserved_components = solvent_loadings + salt_loadings + additive_loadings
    equilibrium_recipe_concentrations = {
        component.name: component.concentration_mol_m3
        for component in salt_loadings + additive_loadings
        if component.name in library_records.equilibria_record["recipe_component_formulas"]
    }
    speciation_equilibrium = solve_speciation_equilibrium(
        recipe_concentrations_mol_m3=equilibrium_recipe_concentrations,
        species_charges_e={
            name: float(record["formal_charge_e"])
            for name, record in library_records.species_records.items()
        },
        equilibrium_record=library_records.equilibria_record,
        temperature_K=temperature_K,
    )
    equilibrium_recipe_names = set(equilibrium_recipe_concentrations)
    inert_components = tuple(
        component
        for component in conserved_components
        if component.name not in equilibrium_recipe_names
    )
    resolved_equilibrium_species = tuple(
        RecipeComponentLoading(
            name=species.name,
            concentration_mol_m3=species.concentration_mol_m3,
            role=str(library_records.species_records[species.name]["role"]),
        )
        for species in speciation_equilibrium.species
    )
    return RecipeBuildResult(
        temperature_K=temperature_K,
        conserved_components=conserved_components,
        resolved_species=inert_components + resolved_equilibrium_species,
        speciation_equilibrium=speciation_equilibrium,
        solvent_volume_fractions=dict(sorted(solvent_volume_fractions.items())),
        additive_weight_fractions=dict(sorted(additive_weight_fractions.items())),
        library_records=library_records,
    )


def _salt_component_sort_key(records: PhysicalLibraryRecords):
    def component_key(item: tuple[str, float]) -> tuple[bool, str]:
        species_name, _concentration_mol_l = item
        species_role = str(records.species_records[species_name]["role"])
        return species_role != "cation", species_name

    return component_key


def validate_physical_library_records(records: PhysicalLibraryRecords) -> None:
    _validate_manifest(records.manifest)
    for species_name, species_record in records.species_records.items():
        _validate_species_record(species_name, species_record)
    for pair_key, pair_record in records.pair_records.items():
        _validate_pair_record(pair_key, pair_record, records.species_records)
    _require_mapping_keys(
        records.mixture_record,
        (
            "schema",
            "species",
            "dielectric",
            "viscosity",
            "packing",
            "atmosphere",
            "parameter_provenance",
        ),
        "mixture.yaml",
    )
    _validate_mixture_parameter_provenance(records.mixture_record)
    _require_mapping_keys(
        records.basis_record,
        ("schema", "coordinates", "threshold_source"),
        "basis.yaml",
    )
    _require_mapping_keys(
        records.transition_record,
        ("schema", "families", "numerical_method", "trajectory_projection"),
        "transitions.yaml",
    )
    _validate_trajectory_projection_record(
        records.transition_record["trajectory_projection"]
    )
    _require_mapping_keys(
        records.memory_record,
        ("schema", "memory_records", "orthogonalization"),
        "memory.yaml",
    )
    _validate_memory_records(records.memory_record, records.transition_record)
    _validate_association_record(records.association_record)
    _validate_equilibria_record(records.equilibria_record, records.species_records)


def _validate_equilibria_record(equilibria_record: dict, species_records: dict) -> None:
    _require_mapping_keys(
        equilibria_record,
        (
            "schema",
            "standard_concentration_mol_m3",
            "relative_residual_tolerance",
            "maximum_function_evaluations",
            "recipe_component_formulas",
            "equilibrium_species_formulas",
            "reactions",
        ),
        "equilibria.yaml",
    )
    for species_name in equilibria_record["equilibrium_species_formulas"]:
        if species_name not in species_records:
            raise KeyError(f"equilibria.yaml references missing species {species_name}")
    for reaction_record in equilibria_record["reactions"]:
        _require_mapping_keys(
            reaction_record,
            (
                "id",
                "stoichiometry",
                "equilibrium_constant_at_reference",
                "reference_temperature_K",
                "reaction_enthalpy_J_mol",
                "source",
                "parameter_provenance",
            ),
            "equilibria.yaml reaction",
        )
        if reaction_record["parameter_provenance"] not in FIRST_PRINCIPLES_ALLOWED_PARAMETER_PROVENANCE:
            raise ValueError(f"equilibria reaction {reaction_record['id']} has forbidden provenance")


def _validate_association_record(association_record: dict) -> None:
    _require_mapping_keys(
        association_record,
        ("schema", "association_residual", "state_resolved_born", "aggregate_topologies"),
        "association.yaml",
    )
    for operator_name in ("association_residual", "state_resolved_born"):
        operator_record = association_record[operator_name]
        _require_mapping_keys(
            operator_record,
            ("equation", "source", "parameter_provenance", "initialization_basis", "reference_temperature_K", "state_features"),
            f"association.yaml.{operator_name}",
        )
        provenance = str(operator_record["parameter_provenance"])
        if provenance not in FIRST_PRINCIPLES_ALLOWED_PARAMETER_PROVENANCE:
            raise ValueError(
                f"association.yaml.{operator_name} has forbidden provenance {provenance}"
            )
        reference_temperature_K = float(operator_record["reference_temperature_K"])
        if not np.isfinite(reference_temperature_K) or reference_temperature_K <= 0.0:
            raise ValueError(f"association.yaml.{operator_name}.reference_temperature_K must be positive")
        if provenance == "initialized_estimate":
            if operator_record["initialization_basis"] != "user_authorized_pre_validation_physical_initialization":
                raise ValueError(f"association.yaml.{operator_name} initialized_estimate requires user-authorized initialization_basis")
            if reference_temperature_K != T_REF_K:
                raise ValueError(f"association.yaml.{operator_name} initialized_estimate must use T_REF_K")
        state_features = operator_record["state_features"]
        if not isinstance(state_features, dict) or not state_features:
            raise ValueError(
                f"association.yaml.{operator_name}.state_features must be non-empty"
            )
        for feature_name, feature_values in state_features.items():
            if not isinstance(feature_values, dict) or not feature_values:
                raise ValueError(
                    f"association.yaml.{operator_name}.{feature_name} must be non-empty"
                )
            for state_value, coefficient in feature_values.items():
                coefficient_value = float(coefficient)
                if not np.isfinite(coefficient_value):
                    raise ValueError(
                        f"association.yaml.{operator_name}.{feature_name}.{state_value} "
                        "must be finite"
                    )
                if operator_name == "state_resolved_born" and not 0.0 <= coefficient_value <= 1.0:
                    raise ValueError(f"association.yaml.{operator_name}.{feature_name}.{state_value} must be in [0, 1]")

    residual_record = association_record["association_residual"]
    if residual_record["parameter_provenance"] == "initialized_estimate":
        residual_features = residual_record["state_features"]
        for feature_key, multiplier in ASSOCIATION_INITIALIZATION_MULTIPLIERS.items():
            feature_name, state_value = feature_key
            if feature_name not in residual_features or state_value not in residual_features[feature_name]:
                raise KeyError(f"missing_state_free_energy_operator: association_residual.{feature_name}.{state_value}")
            expected_J_mol = multiplier * R * T_REF_K
            if float(residual_features[feature_name][state_value]) != expected_J_mol:
                raise ValueError(f"association.yaml association_residual.{feature_name}.{state_value} must equal its exact R*T_REF_K initialization")
    _validate_aggregate_topologies(association_record["aggregate_topologies"])


def _validate_aggregate_topologies(aggregate_topologies: dict) -> None:
    expected_topologies = {
        "Li2A_positive": ("aggregate", {"Li": 2, "A": 1, "ligand": 0}, 1, (("Li0", "A0"), ("Li1", "A0"))),
        "LiA2_negative": ("aggregate", {"Li": 1, "A": 2, "ligand": 0}, -1, (("Li0", "A0"), ("Li0", "A1"))),
        "Li2A2_neutral": ("aggregate", {"Li": 2, "A": 2, "ligand": 0}, 0, (("Li0", "A0"), ("A0", "Li1"), ("Li1", "A1"))),
        "bridge_network": ("bridge_network", {"Li": 2, "A": 2, "ligand": 0}, 0, (("Li0", "A0"), ("A0", "Li1"), ("Li1", "A1"), ("A1", "Li0"))),
    }
    if not isinstance(aggregate_topologies, dict) or set(aggregate_topologies) != set(expected_topologies):
        raise ValueError("association.yaml.aggregate_topologies must contain the exact required topology inventory")
    required_fields = ("topology_id", "cluster_family", "graph_edges", "component_stoichiometry", "net_formal_charge_e", "minimum_cation_count", "minimum_anion_count", "minimum_ligand_count", "source", "parameter_provenance")
    for topology_id, expected in expected_topologies.items():
        topology_record = aggregate_topologies[topology_id]
        _require_mapping_keys(topology_record, required_fields, f"association.yaml.aggregate_topologies.{topology_id}")
        expected_family, expected_stoichiometry, expected_charge, expected_edges = expected
        actual_edges = tuple(tuple(edge) for edge in topology_record["graph_edges"])
        actual_identity = (topology_record["cluster_family"], topology_record["component_stoichiometry"], int(topology_record["net_formal_charge_e"]), actual_edges)
        if topology_record["topology_id"] != topology_id or actual_identity != expected:
            raise ValueError(f"association topology {topology_id} does not match the plan-defined record")
        minimum_counts = (int(topology_record["minimum_cation_count"]), int(topology_record["minimum_anion_count"]), int(topology_record["minimum_ligand_count"]))
        expected_counts = (expected_stoichiometry["Li"], expected_stoichiometry["A"], expected_stoichiometry["ligand"])
        if minimum_counts != expected_counts:
            raise ValueError(f"association topology {topology_id} has inconsistent minimum multiplicity")
        if topology_record["source"] != "user_authorized_initialized_topology" or topology_record["parameter_provenance"] != "initialized_estimate":
            raise ValueError(f"association topology {topology_id} requires initialized_estimate provenance")


def _validate_trajectory_projection_record(trajectory_projection_record: dict) -> None:
    required_fields = (
        "commitment_time_s",
        "recrossing_window_s",
        "endpoint_persistence_condition",
        "zero_frequency_integration_window_s",
        "zero_frequency_plateau_window_s",
    )
    _require_mapping_keys(
        trajectory_projection_record,
        required_fields,
        "transitions.yaml trajectory_projection",
    )
    for time_field in (
        "commitment_time_s",
        "recrossing_window_s",
        "zero_frequency_integration_window_s",
        "zero_frequency_plateau_window_s",
    ):
        time_value_s = float(trajectory_projection_record[time_field])
        if not np.isfinite(time_value_s) or time_value_s <= 0.0:
            raise ValueError(
                f"transitions.yaml trajectory_projection.{time_field} must be positive"
            )
    integration_window_s = float(
        trajectory_projection_record["zero_frequency_integration_window_s"]
    )
    plateau_window_s = float(
        trajectory_projection_record["zero_frequency_plateau_window_s"]
    )
    if plateau_window_s >= integration_window_s:
        raise ValueError(
            "zero-frequency plateau window must be shorter than integration window"
        )
    if (
        str(trajectory_projection_record["endpoint_persistence_condition"])
        != "uninterrupted_destination_residence"
    ):
        raise ValueError("unsupported transition endpoint persistence condition")


def _validate_memory_records(memory_record: dict, transition_record: dict) -> None:
    memory_records = memory_record["memory_records"]
    if not isinstance(memory_records, dict) or not memory_records:
        raise ValueError("memory.memory_records must be a non-empty mapping")
    transition_families = set(transition_record["families"])
    for family, family_record in memory_records.items():
        _require_mapping_keys(
            family_record,
            ("transport_ownership", "matching_transition_families"),
            f"memory.memory_records.{family}",
        )
        if family_record["transport_ownership"] not in (
            "bounded_memory",
            "diagnostic",
        ):
            raise ValueError(
                f"memory family {family} has invalid transport ownership"
            )
        matching_families = family_record["matching_transition_families"]
        if not isinstance(matching_families, list):
            raise TypeError(
                f"memory family {family} matching_transition_families must be a list"
            )
        missing_families = tuple(
            matching_family
            for matching_family in matching_families
            if matching_family not in transition_families
        )
        if missing_families:
            raise ValueError(
                f"memory family {family} references missing transitions {missing_families}"
            )


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
    parser.add_argument("--reference-box-length-m", type=float)
    parser.add_argument("--volume-m3", type=float)
    parser.add_argument("--state-quadrature-order", type=int)
    parser.add_argument("--transition-grid-count", type=int)
    parsed_arguments = parser.parse_args()
    if parsed_arguments.command == "validate-library":
        return _main_validate_library(parsed_arguments.library_root)
    if parsed_arguments.command == "validate-property-db":
        required_numerical_arguments = {
            "reference_box_length_m": parsed_arguments.reference_box_length_m,
            "volume_m3": parsed_arguments.volume_m3,
            "state_quadrature_order": parsed_arguments.state_quadrature_order,
            "transition_grid_count": parsed_arguments.transition_grid_count,
        }
        missing_argument_names = tuple(
            argument_name
            for argument_name, argument_value in required_numerical_arguments.items()
            if argument_value is None
        )
        if missing_argument_names:
            parser.error(
                "validate-property-db requires explicit numerical arguments: "
                + ", ".join(
                    f"--{argument_name.replace('_', '-')}"
                    for argument_name in missing_argument_names
                )
            )
        return _main_validate_property_db(
            library_root=parsed_arguments.library_root,
            reference_box_length_m=parsed_arguments.reference_box_length_m,
            volume_m3=parsed_arguments.volume_m3,
            state_quadrature_order=parsed_arguments.state_quadrature_order,
            transition_grid_count=parsed_arguments.transition_grid_count,
        )
    raise ValueError(f"unsupported command {parsed_arguments.command}")


def _main_validate_library(library_root: Path) -> int:
    records = load_physical_library(library_root)
    print(f"library_root={records.root}")
    print(f"species_count={len(records.species_records)}")
    print(f"pair_count={len(records.pair_records)}")
    print("species=" + ",".join(sorted(records.species_records)))
    return 0


def _main_validate_property_db(
    library_root: Path,
    reference_box_length_m: float,
    volume_m3: float,
    state_quadrature_order: int,
    transition_grid_count: int,
) -> int:
    from conductivity.physical_library.generator_construction import NumericalOptions
    from conductivity.physical_library.property_db_validation import (
        validate_property_db_supported_conductivity_rows,
    )
    from data.electrolyte_property_db import DATA

    numerical_options = NumericalOptions(
        reference_box_lengths_m=np.full(3, reference_box_length_m, dtype=float),
        volume_m3=volume_m3,
        state_quadrature_order=state_quadrature_order,
        transition_grid_count=transition_grid_count,
    )
    summary = validate_property_db_supported_conductivity_rows(
        property_db_entries=DATA,
        physical_library_root=library_root,
        numerical_options=numerical_options,
    )
    print(f"total_entry_count={summary.total_entry_count}")
    print(f"evaluated_entry_count={summary.evaluated_entry_count}")
    print(f"skipped_entry_count={summary.skipped_entry_count}")
    print(f"mean_error_mS_cm={summary.mean_error_mS_cm:.12g}")
    print(f"mean_absolute_error_mS_cm={summary.mean_absolute_error_mS_cm:.12g}")
    print(
        "root_mean_square_error_mS_cm="
        f"{summary.root_mean_square_error_mS_cm:.12g}"
    )
    print(
        "mean_absolute_percent_error="
        f"{summary.mean_absolute_percent_error:.12g}"
    )
    print(f"max_absolute_error_mS_cm={summary.max_absolute_error_mS_cm:.12g}")
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
            "association_record",
            "equilibria_record",
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


def _validate_mixture_parameter_provenance(mixture_record: dict) -> None:
    parameter_provenance = mixture_record["parameter_provenance"]
    if not isinstance(parameter_provenance, dict):
        raise TypeError("mixture.parameter_provenance must be a mapping")
    active_parameter_paths = _active_mixture_parameter_paths(mixture_record)
    missing_parameter_paths = tuple(
        parameter_path
        for parameter_path in active_parameter_paths
        if parameter_path not in parameter_provenance
    )
    if missing_parameter_paths:
        raise KeyError(
            "mixture.parameter_provenance missing active first-principles parameters: "
            f"{missing_parameter_paths}"
        )
    allowed_provenance = set(FIRST_PRINCIPLES_ALLOWED_PARAMETER_PROVENANCE)
    for parameter_path in active_parameter_paths:
        provenance_label = str(parameter_provenance[parameter_path])
        if provenance_label == FIRST_PRINCIPLES_REJECTED_PARAMETER_PROVENANCE:
            raise ValueError(
                "first-principles physical library rejects scalar-sigma-fitted "
                f"parameter {parameter_path}"
            )
        if provenance_label not in allowed_provenance:
            raise ValueError(
                f"mixture.parameter_provenance[{parameter_path}] has unsupported "
                f"label {provenance_label}"
            )


def _active_mixture_parameter_paths(mixture_record: dict) -> tuple[str, ...]:
    active_parameter_paths: list[str] = []
    mobility_record = mixture_record["mobility"]
    resistance_terms = tuple(str(term) for term in mobility_record["resistance_terms"])
    if "free_volume" in resistance_terms:
        active_parameter_paths.append("mobility.free_volume_exponent")
    if "free_volume" in resistance_terms:
        active_parameter_paths.append("packing.phi_max")
    if "local_fields" in mixture_record:
        local_field_record = mixture_record["local_fields"]
        for local_field_key in sorted(local_field_record):
            active_parameter_paths.append(f"local_fields.{local_field_key}")
    return tuple(active_parameter_paths)


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


if __name__ == "__main__":
    raise SystemExit(main())
