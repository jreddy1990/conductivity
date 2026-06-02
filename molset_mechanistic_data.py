"""Data assembly for the trained mechanistic MolSet prototype."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from constants import T_REF_K
from conductivity.mol_set_sigma_unit_aware_prototype import (
    D_INPUT,
    N_MAX_SPECIES,
    build_unit_aware_recipe_inputs,
    compute_unit_aware_normalization,
)
from data.electrolyte_calisol_db import DATA as CALISOL_DB
from data.electrolyte_electrolytomics_db import DATA as ELECTROLYTOMICS_DB
from data.electrolyte_property_db import DATA as PROPERTY_DB
from data.electrolyte_tplus_db import compute_deff_for_recipe, compute_tplus_for_all_recipes
from data.lehnert2025_db import load_lehnert2025
from data.logan2018_db import load_logan2018
from data.species_data import ADDITIVES, SALTS, SOLVENTS
from data.valoen2005_db import load_valoen2005
from utils.strict_validation import positive_finite_float, require_mapping


KNOWN_SPECIES = set(SOLVENTS) | set(SALTS) | set(ADDITIVES)
ROLE_SPECIES = {
    "solvents": set(SOLVENTS),
    "salts": set(SALTS),
    "additives": set(ADDITIVES),
}
MECHANISTIC_DATA_SOURCES = (
    "property_db",
    "logan2018",
    "valoen2005",
    "lehnert2025",
    "transport_targets2019",
    "electrolytomics",
    "calisol23_vv",
    "oedb_li_aux",
    "bamboo_mix_eis",
    "clean_oedb_li_aux",
)
FARADAY_C_MOL = 96485.33212
GAS_CONSTANT_J_MOL_K = 8.31446261815324
OEDB_ARROW = Path("conductivity/fm_data/oedb/electrolytes.arrow")
FM_CLEAN_DATASET = Path("conductivity/fm_data/clean_dataset.pkl")
FM_SPECIES_SMILES = Path("conductivity/fm_data/species_smiles.json")
OEDB_ANION_TO_SALT = {
    "BF4": "LiBF4",
    "ClO4": "LiClO4",
    "FSI": "LiFSI",
    "PF6": "LiPF6",
    "TFSI": "LiTFSI",
}
FM_ANION_TO_SALT = {
    "BF4-": "LiBF4",
    "ClO4-": "LiClO4",
    "FSI-": "LiFSI",
    "PF6-": "LiPF6",
    "TFSI-": "LiTFSI",
}
FM_REGISTERED_SMILES_ALIASES = {
    "CC#N": "AN",
    "F[B-](F)(F)F": "BF4-",
    "[B-](F)(F)(F)F": "BF4-",
    "[O-]Cl(=O)(=O)=O": "ClO4-",
    "[N-](S(=O)(=O)F)S(=O)(=O)F": "FSI-",
    "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F": "TFSI-",
    "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F": "TFSI-",
    "C1COC(=O)O1": "EC",
    "CC1COC(=O)O1": "PC",
    "C1C(OC(=O)O1)F": "FEC",
    "CCOC(=O)OC": "EMC",
    "CCOC(C)=O": "EA",
    "COP(=O)(OC)OC": "DMMP",
    "O=P(OCC(F)(F)F)(OCC(F)(F)F)OCC(F)(F)F": "TTFP",
}


@dataclass(frozen=True)
class DatasetManifestEntry:
    """Declared evidence class for one training source."""

    source: str
    evidence_class: str
    conductivity_role: str
    train_conductivity: bool
    auxiliary_labels: tuple[str, ...]
    source_type: str


DATASET_MANIFEST: Mapping[str, DatasetManifestEntry] = {
    "property_db": DatasetManifestEntry(
        source="property_db",
        evidence_class="curated_empirical_conductivity",
        conductivity_role="measured_label",
        train_conductivity=True,
        auxiliary_labels=("density",),
        source_type="manual_curated_rows",
    ),
    "logan2018": DatasetManifestEntry(
        source="logan2018",
        evidence_class="published_empirical_conductivity_viscosity",
        conductivity_role="measured_label",
        train_conductivity=True,
        auxiliary_labels=("viscosity",),
        source_type="tabulated_measurements",
    ),
    "valoen2005": DatasetManifestEntry(
        source="valoen2005",
        evidence_class="published_transport_correlation",
        conductivity_role="auxiliary_transport_correlation",
        train_conductivity=False,
        auxiliary_labels=("cation_self_current", "anion_self_current"),
        source_type="published_fit_grid",
    ),
    "lehnert2025": DatasetManifestEntry(
        source="lehnert2025",
        evidence_class="published_conductivity_correlation",
        conductivity_role="correlation_label",
        train_conductivity=True,
        auxiliary_labels=(),
        source_type="published_fit_grid",
    ),
    "transport_targets2019": DatasetManifestEntry(
        source="transport_targets2019",
        evidence_class="published_transport_auxiliary_targets",
        conductivity_role="not_a_conductivity_label",
        train_conductivity=False,
        auxiliary_labels=("cation_self_current", "anion_self_current"),
        source_type="matched_transport_coefficients",
    ),
    "electrolytomics": DatasetManifestEntry(
        source="electrolytomics",
        evidence_class="published_empirical_conductivity",
        conductivity_role="measured_label",
        train_conductivity=True,
        auxiliary_labels=(),
        source_type="converted_vv_measurements",
    ),
    "calisol23_vv": DatasetManifestEntry(
        source="calisol23_vv",
        evidence_class="published_empirical_conductivity",
        conductivity_role="measured_label",
        train_conductivity=True,
        auxiliary_labels=(),
        source_type="converted_vv_measurements",
    ),
    "oedb_li_aux": DatasetManifestEntry(
        source="oedb_li_aux",
        evidence_class="md_green_kubo_auxiliary_transport",
        conductivity_role="not_a_conductivity_training_label",
        train_conductivity=False,
        auxiliary_labels=(
            "density",
            "viscosity",
            "cation_self_current",
            "anion_self_current",
            "current_distinct",
            "association_fraction",
        ),
        source_type="single_salt_single_solvent_md_observables",
    ),
    "bamboo_mix_eis": DatasetManifestEntry(
        source="bamboo_mix_eis",
        evidence_class="cleaned_experimental_eis_transport_corpus",
        conductivity_role="measured_label",
        train_conductivity=True,
        auxiliary_labels=(),
        source_type="registered_species_mole_fraction_corpus",
    ),
    "clean_oedb_li_aux": DatasetManifestEntry(
        source="clean_oedb_li_aux",
        evidence_class="cleaned_md_green_kubo_auxiliary_transport_corpus",
        conductivity_role="not_a_conductivity_training_label",
        train_conductivity=False,
        auxiliary_labels=(
            "density",
            "viscosity",
            "cation_self_current",
            "anion_self_current",
            "current_distinct",
        ),
        source_type="registered_species_mole_fraction_md_observables",
    ),
}


@dataclass(frozen=True)
class DatasetAudit:
    """Source-level data-quality summary for training-gate reporting."""

    source: str
    rows: int
    conductivity_labels: int
    viscosity_labels: int
    cation_self_current_labels: int
    anion_self_current_labels: int
    current_distinct_labels: int
    association_fraction_labels: int
    multi_salt_rows: int
    additive_rows: int
    min_sigma_mS_cm: float
    max_sigma_mS_cm: float
    min_temperature_K: float
    max_temperature_K: float


@dataclass(frozen=True)
class MechanisticRow:
    """One empirical conductivity row with optional auxiliary labels."""

    source: str
    row_index: int
    recipe: Mapping[str, Any]
    conductivity_mS_cm: float
    has_conductivity: float
    density_g_ml: float
    has_density: float
    viscosity_cP: float
    has_viscosity: float
    dielectric: float
    has_dielectric: float
    cation_self_current_mS_cm: float
    has_cation_self_current: float
    anion_self_current_mS_cm: float
    has_anion_self_current: float
    cation_anion_distinct_mS_cm: float
    has_cation_anion_distinct: float
    current_distinct_mS_cm: float
    has_current_distinct: float
    association_fraction: float
    has_association_fraction: float
    temperature_K: float
    recipe_key: str


@dataclass(frozen=True)
class MechanisticBatch:
    """Padded arrays for training the mechanistic MolSet prototype."""

    species_props_norm: np.ndarray
    species_props_raw: np.ndarray
    solvent_volume_fraction: np.ndarray
    salt_molarity: np.ndarray
    additive_weight_fraction: np.ndarray
    mask: np.ndarray
    temperature_K: np.ndarray
    sigma_mS_cm: np.ndarray
    log_sigma: np.ndarray
    conductivity_mask: np.ndarray
    density_g_ml: np.ndarray
    density_mask: np.ndarray
    viscosity_cP: np.ndarray
    viscosity_mask: np.ndarray
    dielectric: np.ndarray
    dielectric_mask: np.ndarray
    cation_self_current_mS_cm: np.ndarray
    cation_self_current_mask: np.ndarray
    anion_self_current_mS_cm: np.ndarray
    anion_self_current_mask: np.ndarray
    cation_anion_distinct_mS_cm: np.ndarray
    cation_anion_distinct_mask: np.ndarray
    current_distinct_mS_cm: np.ndarray
    current_distinct_mask: np.ndarray
    association_fraction: np.ndarray
    association_fraction_mask: np.ndarray
    weights: np.ndarray
    row_indices: tuple[int, ...]
    sources: tuple[str, ...]
    recipe_keys: tuple[str, ...]


def all_known_species() -> tuple[str, ...]:
    """Return all registered electrolyte species usable by the prototype."""

    return tuple(sorted(KNOWN_SPECIES))


def load_property_db_rows() -> tuple[MechanisticRow, ...]:
    """Load room-temperature empirical conductivity rows from electrolyte_property_db."""

    rows: list[MechanisticRow] = []
    for row_index, entry in enumerate(PROPERTY_DB):
        recipe = require_mapping(entry, "recipe", f"DATA[{row_index}]")
        properties = require_mapping(entry, "properties", f"DATA[{row_index}]")
        rows.append(_row_from_recipe_properties("property_db", row_index, recipe, properties, T_REF_K))
    if not rows:
        raise ValueError("No conductivity rows loaded from electrolyte_property_db")
    return tuple(rows)


def load_mechanistic_rows(data_sources: Sequence[str]) -> tuple[MechanisticRow, ...]:
    """Load conductivity and available auxiliary labels from selected curated sources."""

    if not data_sources:
        raise ValueError("data_sources must not be empty")
    seen_sources: set[str] = set()
    rows: list[MechanisticRow] = []
    for source in data_sources:
        if source not in MECHANISTIC_DATA_SOURCES:
            raise ValueError(f"Unsupported mechanistic data source {source!r}")
        if source in seen_sources:
            raise ValueError(f"Duplicate mechanistic data source {source!r}")
        seen_sources.add(source)
        rows.extend(_load_source_rows(source))
    if not rows:
        raise ValueError(f"No rows loaded from data_sources={tuple(data_sources)!r}")
    _validate_loaded_manifest_sources(data_sources, rows)
    return tuple(rows)


def source_counts(rows: Sequence[MechanisticRow]) -> Mapping[str, int]:
    """Count loaded rows by source for audit reporting."""

    return dict(Counter(row.source for row in rows))


def audit_mechanistic_rows(rows: Sequence[MechanisticRow]) -> tuple[DatasetAudit, ...]:
    """Summarize source coverage and label availability before training."""

    if not rows:
        raise ValueError("rows must not be empty")
    audits: list[DatasetAudit] = []
    for source in sorted(source_counts(rows)):
        source_rows = [row for row in rows if row.source == source]
        sigma_rows = [row.conductivity_mS_cm for row in source_rows if row.has_conductivity > 0.0]
        if sigma_rows:
            min_sigma = float(np.min(np.asarray(sigma_rows, dtype=np.float64)))
            max_sigma = float(np.max(np.asarray(sigma_rows, dtype=np.float64)))
        else:
            min_sigma = 0.0
            max_sigma = 0.0
        temperatures = np.asarray([row.temperature_K for row in source_rows], dtype=np.float64)
        audits.append(
            DatasetAudit(
                source=source,
                rows=len(source_rows),
                conductivity_labels=sum(int(row.has_conductivity > 0.0) for row in source_rows),
                viscosity_labels=sum(int(row.has_viscosity > 0.0) for row in source_rows),
                cation_self_current_labels=sum(
                    int(row.has_cation_self_current > 0.0) for row in source_rows
                ),
                anion_self_current_labels=sum(
                    int(row.has_anion_self_current > 0.0) for row in source_rows
                ),
                current_distinct_labels=sum(
                    int(row.has_current_distinct > 0.0) for row in source_rows
                ),
                association_fraction_labels=sum(
                    int(row.has_association_fraction > 0.0) for row in source_rows
                ),
                multi_salt_rows=sum(
                    int(len(require_mapping(row.recipe, "salts", "recipe")) > 1) for row in source_rows
                ),
                additive_rows=sum(
                    int(bool(require_mapping(row.recipe, "additives", "recipe"))) for row in source_rows
                ),
                min_sigma_mS_cm=min_sigma,
                max_sigma_mS_cm=max_sigma,
                min_temperature_K=float(np.min(temperatures)),
                max_temperature_K=float(np.max(temperatures)),
            )
        )
    return tuple(audits)


def build_mechanistic_batch(
    rows: Sequence[MechanisticRow],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> MechanisticBatch:
    """Convert empirical rows into padded role-aware MolSet arrays."""

    if not rows:
        raise ValueError("rows must not be empty")
    n_rows = len(rows)
    props_norm = np.zeros((n_rows, N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    props_raw = np.zeros((n_rows, N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    solvent_vv = np.zeros((n_rows, N_MAX_SPECIES), dtype=np.float64)
    salt_m = np.zeros((n_rows, N_MAX_SPECIES), dtype=np.float64)
    additive_wt = np.zeros((n_rows, N_MAX_SPECIES), dtype=np.float64)
    mask = np.zeros((n_rows, N_MAX_SPECIES), dtype=np.float64)
    temperature = np.zeros(n_rows, dtype=np.float64)
    sigma = np.zeros(n_rows, dtype=np.float64)
    log_sigma = np.zeros(n_rows, dtype=np.float64)
    conductivity_mask = np.zeros(n_rows, dtype=np.float64)
    density = np.zeros(n_rows, dtype=np.float64)
    density_mask = np.zeros(n_rows, dtype=np.float64)
    viscosity = np.zeros(n_rows, dtype=np.float64)
    viscosity_mask = np.zeros(n_rows, dtype=np.float64)
    dielectric = np.zeros(n_rows, dtype=np.float64)
    dielectric_mask = np.zeros(n_rows, dtype=np.float64)
    cation_self_current = np.zeros(n_rows, dtype=np.float64)
    cation_self_current_mask = np.zeros(n_rows, dtype=np.float64)
    anion_self_current = np.zeros(n_rows, dtype=np.float64)
    anion_self_current_mask = np.zeros(n_rows, dtype=np.float64)
    cation_anion_distinct = np.zeros(n_rows, dtype=np.float64)
    cation_anion_distinct_mask = np.zeros(n_rows, dtype=np.float64)
    current_distinct = np.zeros(n_rows, dtype=np.float64)
    current_distinct_mask = np.zeros(n_rows, dtype=np.float64)
    association_fraction = np.zeros(n_rows, dtype=np.float64)
    association_fraction_mask = np.zeros(n_rows, dtype=np.float64)
    weights = np.ones(n_rows, dtype=np.float64)
    row_indices: list[int] = []
    sources: list[str] = []
    recipe_keys: list[str] = []

    for idx, row in enumerate(rows):
        inputs = build_unit_aware_recipe_inputs(row.recipe, norm_mean, norm_std)
        props_norm[idx] = inputs.species_props_norm
        props_raw[idx] = inputs.species_props_raw
        solvent_vv[idx] = inputs.solvent_volume_fraction
        salt_m[idx] = inputs.salt_molarity
        additive_wt[idx] = inputs.additive_weight_fraction
        mask[idx] = inputs.mask
        temperature[idx] = row.temperature_K
        sigma[idx] = row.conductivity_mS_cm
        log_sigma[idx] = np.log(row.conductivity_mS_cm)
        conductivity_mask[idx] = row.has_conductivity
        density[idx] = row.density_g_ml
        density_mask[idx] = row.has_density
        viscosity[idx] = row.viscosity_cP
        viscosity_mask[idx] = row.has_viscosity
        dielectric[idx] = row.dielectric
        dielectric_mask[idx] = row.has_dielectric
        cation_self_current[idx] = row.cation_self_current_mS_cm
        cation_self_current_mask[idx] = row.has_cation_self_current
        anion_self_current[idx] = row.anion_self_current_mS_cm
        anion_self_current_mask[idx] = row.has_anion_self_current
        cation_anion_distinct[idx] = row.cation_anion_distinct_mS_cm
        cation_anion_distinct_mask[idx] = row.has_cation_anion_distinct
        current_distinct[idx] = row.current_distinct_mS_cm
        current_distinct_mask[idx] = row.has_current_distinct
        association_fraction[idx] = row.association_fraction
        association_fraction_mask[idx] = row.has_association_fraction
        row_indices.append(row.row_index)
        sources.append(row.source)
        recipe_keys.append(row.recipe_key)

    return MechanisticBatch(
        species_props_norm=props_norm,
        species_props_raw=props_raw,
        solvent_volume_fraction=solvent_vv,
        salt_molarity=salt_m,
        additive_weight_fraction=additive_wt,
        mask=mask,
        temperature_K=temperature,
        sigma_mS_cm=sigma,
        log_sigma=log_sigma,
        conductivity_mask=conductivity_mask,
        density_g_ml=density,
        density_mask=density_mask,
        viscosity_cP=viscosity,
        viscosity_mask=viscosity_mask,
        dielectric=dielectric,
        dielectric_mask=dielectric_mask,
        cation_self_current_mS_cm=cation_self_current,
        cation_self_current_mask=cation_self_current_mask,
        anion_self_current_mS_cm=anion_self_current,
        anion_self_current_mask=anion_self_current_mask,
        cation_anion_distinct_mS_cm=cation_anion_distinct,
        cation_anion_distinct_mask=cation_anion_distinct_mask,
        current_distinct_mS_cm=current_distinct,
        current_distinct_mask=current_distinct_mask,
        association_fraction=association_fraction,
        association_fraction_mask=association_fraction_mask,
        weights=weights,
        row_indices=tuple(row_indices),
        sources=tuple(sources),
        recipe_keys=tuple(recipe_keys),
    )


def normalization_from_registered_species() -> tuple[np.ndarray, np.ndarray]:
    """Compute property normalization from registered species properties."""

    return compute_unit_aware_normalization(all_known_species())


def lifsi_dominant_fec_rows(rows: Sequence[MechanisticRow]) -> tuple[MechanisticRow, ...]:
    """Return empirical rows for LiFSI-dominant mixed-salt FEC behavior."""

    selected: list[MechanisticRow] = []
    for row in rows:
        salts = require_mapping(row.recipe, "salts", f"DATA[{row.row_index}].recipe")
        additives = require_mapping(row.recipe, "additives", f"DATA[{row.row_index}].recipe")
        if "LiFSI" not in salts or "LiPF6" not in salts:
            continue
        total_salt = sum(float(value) for value in salts.values())
        if total_salt <= 0.0:
            raise ValueError(f"DATA[{row.row_index}].recipe.salts total must be positive")
        lifsi_share = float(salts["LiFSI"]) / total_salt
        fec_loading = float(additives["FEC"]) if "FEC" in additives else 0.0
        if lifsi_share >= 0.75 and fec_loading <= 0.11:
            selected.append(row)
    selected.sort(key=lambda item: _fec_loading(item.recipe))
    return tuple(selected)


def recipe_key(recipe: Mapping[str, Any]) -> str:
    """Canonical recipe key for grouping and audit reports."""

    solvents = _format_component_mapping(require_mapping(recipe, "solvents", "recipe"))
    salts = _format_component_mapping(require_mapping(recipe, "salts", "recipe"))
    additives = _format_component_mapping(require_mapping(recipe, "additives", "recipe"))
    return f"solvents={solvents}|salts={salts}|additives={additives}"


def _format_component_mapping(mapping: Mapping[str, Any]) -> str:
    parts = [f"{name}:{float(value):.6g}" for name, value in sorted(mapping.items())]
    return ",".join(parts)


def _load_source_rows(source: str) -> tuple[MechanisticRow, ...]:
    if source == "property_db":
        return load_property_db_rows()
    if source == "logan2018":
        return _rows_from_entries(source, load_logan2018())
    if source == "valoen2005":
        return _rows_from_entries(source, load_valoen2005())
    if source == "lehnert2025":
        return _rows_from_entries(source, load_lehnert2025())
    if source == "transport_targets2019":
        return _transport_target_rows()
    if source == "electrolytomics":
        return _rows_from_dict_entries(source, ELECTROLYTOMICS_DB, convert_solvent_wt_to_vv=False)
    if source == "calisol23_vv":
        return _rows_from_dict_entries(source, CALISOL_DB, convert_solvent_wt_to_vv=True)
    if source == "oedb_li_aux":
        return _oedb_li_auxiliary_rows()
    if source == "bamboo_mix_eis":
        return _fm_clean_dataset_rows(source)
    if source == "clean_oedb_li_aux":
        return _fm_clean_dataset_rows(source)
    raise ValueError(f"Unsupported mechanistic data source {source!r}")


def _validate_loaded_manifest_sources(data_sources: Sequence[str], rows: Sequence[MechanisticRow]) -> None:
    for source in data_sources:
        if source not in DATASET_MANIFEST:
            raise ValueError(f"Dataset source {source!r} is missing from DATASET_MANIFEST")
        source_rows = [row for row in rows if row.source == source]
        if not source_rows:
            raise ValueError(f"Dataset source {source!r} loaded zero rows")
        manifest = DATASET_MANIFEST[source]
        if manifest.train_conductivity:
            n_sigma = sum(int(row.has_conductivity > 0.0) for row in source_rows)
            if n_sigma == 0:
                raise ValueError(f"Dataset source {source!r} declared conductivity labels but loaded none")
        for label_name in manifest.auxiliary_labels:
            n_label = _count_auxiliary_label(source_rows, label_name)
            if n_label == 0:
                raise ValueError(f"Dataset source {source!r} declared {label_name} labels but loaded none")


def _count_auxiliary_label(rows: Sequence[MechanisticRow], label_name: str) -> int:
    if label_name == "density":
        return sum(int(row.has_density > 0.0) for row in rows)
    if label_name == "viscosity":
        return sum(int(row.has_viscosity > 0.0) for row in rows)
    if label_name == "cation_self_current":
        return sum(int(row.has_cation_self_current > 0.0) for row in rows)
    if label_name == "anion_self_current":
        return sum(int(row.has_anion_self_current > 0.0) for row in rows)
    if label_name == "dielectric":
        return sum(int(row.has_dielectric > 0.0) for row in rows)
    if label_name == "cation_anion_distinct":
        return sum(int(row.has_cation_anion_distinct > 0.0) for row in rows)
    if label_name == "current_distinct":
        return sum(int(row.has_current_distinct > 0.0) for row in rows)
    if label_name == "association_fraction":
        return sum(int(row.has_association_fraction > 0.0) for row in rows)
    raise ValueError(f"Unsupported auxiliary label name {label_name!r}")


def _rows_from_dict_entries(
    source: str,
    entries: Sequence[Mapping[str, Any]],
    convert_solvent_wt_to_vv: bool,
) -> tuple[MechanisticRow, ...]:
    rows: list[MechanisticRow] = []
    skipped_role_conflict = 0
    for row_index, entry in enumerate(entries):
        recipe = require_mapping(entry, "recipe", f"{source}[{row_index}]")
        if not _recipe_uses_registered_roles(recipe):
            skipped_role_conflict += 1
            continue
        if convert_solvent_wt_to_vv:
            recipe = _recipe_with_solvent_weight_fractions_as_volume_fractions(recipe)
        properties = require_mapping(entry, "properties", f"{source}[{row_index}]")
        if DATASET_MANIFEST[source].train_conductivity:
            if "conductivity_mS_cm" not in properties:
                raise ValueError(f"{source}[{row_index}].properties.conductivity_mS_cm is required")
            sigma_raw = float(properties["conductivity_mS_cm"])
            if not np.isfinite(sigma_raw) or sigma_raw <= 0.0:
                continue
        if "temperature_K" in entry:
            temperature_raw = entry["temperature_K"]
        elif "T_K" in properties:
            temperature_raw = properties["T_K"]
        else:
            raise ValueError(f"{source}[{row_index}] requires temperature_K or properties.T_K")
        temperature_K = positive_finite_float(float(temperature_raw), f"{source}[{row_index}].temperature_K")
        rows.append(_row_from_recipe_properties(source, row_index, recipe, properties, temperature_K))
    if not rows:
        raise ValueError(
            f"No rows loaded from {source}; skipped_role_conflict={skipped_role_conflict}"
        )
    return tuple(rows)


def _recipe_with_solvent_weight_fractions_as_volume_fractions(
    recipe: Mapping[str, Any],
) -> Mapping[str, Any]:
    solvents = require_mapping(recipe, "solvents", "recipe")
    salts = require_mapping(recipe, "salts", "recipe")
    additives = require_mapping(recipe, "additives", "recipe")
    solvent_volumes = {}
    for name, weight_fraction in solvents.items():
        species = require_mapping(SOLVENTS, name, "SOLVENTS")
        density = positive_finite_float(float(species["density_g_ml"]), f"SOLVENTS.{name}.density_g_ml")
        weight = positive_finite_float(float(weight_fraction), f"recipe.solvents.{name}")
        solvent_volumes[name] = weight / density
    total_volume = sum(solvent_volumes.values())
    if total_volume <= 0.0:
        raise ValueError("Solvent volume total must be positive after weight-to-volume conversion")
    solvent_vv = {name: value / total_volume for name, value in solvent_volumes.items()}
    return {
        "solvents": solvent_vv,
        "salts": dict(salts),
        "additives": dict(additives),
    }


def _oedb_li_auxiliary_rows() -> tuple[MechanisticRow, ...]:
    if not OEDB_ARROW.exists():
        raise FileNotFoundError(f"OEDB Arrow file not found at {OEDB_ARROW}")
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ImportError("pyarrow is required to load oedb_li_aux") from exc

    table = feather.read_table(OEDB_ARROW)
    frame = table.to_pandas()
    rows: list[MechanisticRow] = []
    for row_index, entry in frame.iterrows():
        if entry["cation"] != "Li":
            continue
        anion_code = str(entry["anion"])
        if anion_code not in OEDB_ANION_TO_SALT:
            continue
        solvent_name = str(entry["solvent"])
        if solvent_name not in SOLVENTS:
            continue
        salt_name = OEDB_ANION_TO_SALT[anion_code]
        if salt_name not in SALTS:
            continue
        sigma = float(entry["Ionic Conductivity (mS/cm)"])
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue
        molality = positive_finite_float(
            float(entry["Concentration (mol/kg)"]),
            f"oedb_li_aux[{row_index}].Concentration (mol/kg)",
        )
        density_g_ml = positive_finite_float(
            float(entry["Density (g/cm³)"]),
            f"oedb_li_aux[{row_index}].Density (g/cm³)",
        )
        salt_molarity = _molality_to_molarity(molality, density_g_ml, salt_name)
        cation_self = _self_current_from_diffusivity(
            salt_molarity,
            float(entry["Cation's Diffusivity (m²/s)"]),
            T_REF_K,
            f"oedb_li_aux[{row_index}].Cation's Diffusivity (m²/s)",
        )
        anion_self = _self_current_from_diffusivity(
            salt_molarity,
            float(entry["Anion's Diffusivity (m²/s)"]),
            T_REF_K,
            f"oedb_li_aux[{row_index}].Anion's Diffusivity (m²/s)",
        )
        association_fraction = _association_fraction_from_coordination(
            float(entry["Coordination Number (Cation ← Anion)"]),
            float(entry["Coordination Number (Cation ← Solvent)"]),
            f"oedb_li_aux[{row_index}]",
        )
        recipe = {
            "solvents": {solvent_name: 1.0},
            "salts": {salt_name: salt_molarity},
            "additives": {},
        }
        properties = {
            "density": density_g_ml,
            "viscosity_cP": positive_finite_float(
                float(entry["Viscosity (mPa·s)"]),
                f"oedb_li_aux[{row_index}].Viscosity (mPa·s)",
            ),
            "cation_self_current_mS_cm": cation_self,
            "anion_self_current_mS_cm": anion_self,
            "current_distinct_mS_cm": sigma - cation_self - anion_self,
            "association_fraction": association_fraction,
        }
        rows.append(_row_from_recipe_properties("oedb_li_aux", row_index, recipe, properties, T_REF_K))
    if not rows:
        raise ValueError("No OEDB Li auxiliary rows mapped to registered MolSet roles")
    return tuple(rows)


def _fm_clean_dataset_rows(source: str) -> tuple[MechanisticRow, ...]:
    if not FM_CLEAN_DATASET.exists():
        raise FileNotFoundError(f"Clean FM dataset not found at {FM_CLEAN_DATASET}")
    rows: list[MechanisticRow] = []
    smiles_to_species = _fm_smiles_to_registered_species()
    with FM_CLEAN_DATASET.open("rb") as handle:
        clean_rows = pickle.load(handle)
    for row_index, entry in enumerate(clean_rows):
        if source == "bamboo_mix_eis":
            if entry.sigma_source != "BAMBOO-Mix":
                continue
            if str(entry.sigma_method) != "SigmaMethod.EXPERIMENTAL_EIS":
                continue
        elif source == "clean_oedb_li_aux":
            if entry.sigma_source != "OEDB-v1":
                continue
            if str(entry.sigma_method) != "SigmaMethod.GREEN_KUBO":
                continue
        else:
            raise ValueError(f"Unsupported clean FM dataset source {source!r}")
        recipe = _recipe_from_fm_clean_row(
            smiles_list=entry.smiles_list,
            mole_fractions=np.asarray(entry.mole_fractions, dtype=np.float64),
            density_g_ml=_fm_density_for_recipe_conversion(entry.aux_observables),
            smiles_to_species=smiles_to_species,
            context=f"{source}[{row_index}]",
        )
        if recipe is None:
            continue
        sigma = float(entry.sigma_mScm)
        if not np.isfinite(sigma) or sigma <= 0.0:
            continue
        properties: dict[str, float] = {}
        if DATASET_MANIFEST[source].train_conductivity:
            properties["conductivity_mS_cm"] = sigma
        else:
            if not isinstance(entry.aux_observables, Mapping):
                raise ValueError(f"{source}[{row_index}].aux_observables must be a mapping")
            aux = entry.aux_observables
            density = positive_finite_float(
                float(aux["density_gcm3"]),
                f"{source}[{row_index}].aux_observables.density_gcm3",
            )
            viscosity = positive_finite_float(
                float(aux["viscosity_mPas"]),
                f"{source}[{row_index}].aux_observables.viscosity_mPas",
            )
            salt_molarity = _single_salt_molarity(recipe, f"{source}[{row_index}].recipe")
            cation_self = _self_current_from_diffusivity(
                salt_molarity,
                float(aux["diffusion_cation_m2s"]),
                float(entry.temperature_K),
                f"{source}[{row_index}].aux_observables.diffusion_cation_m2s",
            )
            anion_self = _self_current_from_diffusivity(
                salt_molarity,
                float(aux["diffusion_anion_m2s"]),
                float(entry.temperature_K),
                f"{source}[{row_index}].aux_observables.diffusion_anion_m2s",
            )
            properties["density"] = density
            properties["viscosity_cP"] = viscosity
            properties["cation_self_current_mS_cm"] = cation_self
            properties["anion_self_current_mS_cm"] = anion_self
            properties["current_distinct_mS_cm"] = sigma - cation_self - anion_self
        temperature_K = positive_finite_float(
            float(entry.temperature_K),
            f"{source}[{row_index}].temperature_K",
        )
        rows.append(_row_from_recipe_properties(source, row_index, recipe, properties, temperature_K))
    if not rows:
        raise ValueError(f"No rows loaded from clean FM dataset source {source!r}")
    return tuple(rows)


def _fm_smiles_to_registered_species() -> Mapping[str, str]:
    if not FM_SPECIES_SMILES.exists():
        raise FileNotFoundError(f"FM species SMILES manifest not found at {FM_SPECIES_SMILES}")
    with FM_SPECIES_SMILES.open() as handle:
        raw_manifest = json.load(handle)
    smiles_to_species: dict[str, str] = {}
    for species_name, entry in raw_manifest.items():
        if not isinstance(entry, Mapping):
            continue
        if "smiles" not in entry:
            continue
        smiles = str(entry["smiles"])
        smiles_to_species[smiles] = species_name
    smiles_to_species.update(FM_REGISTERED_SMILES_ALIASES)
    return smiles_to_species


def _recipe_from_fm_clean_row(
    smiles_list: Sequence[str],
    mole_fractions: np.ndarray,
    density_g_ml: float,
    smiles_to_species: Mapping[str, str],
    context: str,
) -> Mapping[str, Any] | None:
    if len(smiles_list) != int(mole_fractions.shape[0]):
        raise ValueError(f"{context}.smiles_list and mole_fractions lengths differ")
    species_names: list[str] = []
    for smiles in smiles_list:
        if smiles not in smiles_to_species:
            return None
        species_names.append(smiles_to_species[smiles])
    if "Li+" not in species_names:
        return None
    anions = [name for name in species_names if name in FM_ANION_TO_SALT]
    if len(anions) != 1:
        return None
    anion_name = anions[0]
    salt_name = FM_ANION_TO_SALT[anion_name]
    if salt_name not in SALTS:
        return None
    neutral_names = [name for name in species_names if name != "Li+" and name != anion_name]
    if not neutral_names:
        return None
    if not all(name in SOLVENTS or name in ADDITIVES for name in neutral_names):
        return None
    mole_numbers = _normalized_nonnegative_vector(mole_fractions, f"{context}.mole_fractions")
    li_moles = _species_moles(species_names, mole_numbers, "Li+")
    anion_moles = _species_moles(species_names, mole_numbers, anion_name)
    if not np.isclose(li_moles, anion_moles):
        return None
    salt_moles = 0.5 * (li_moles + anion_moles)
    neutral_masses: dict[str, float] = {}
    solvent_volumes: dict[str, float] = {}
    additive_masses: dict[str, float] = {}
    for species_name, mole_number in zip(species_names, mole_numbers):
        if species_name == "Li+" or species_name == anion_name:
            continue
        role_data = _neutral_role_data(species_name, context)
        molecular_weight = positive_finite_float(
            float(role_data["molecular_weight"]),
            f"{context}.{species_name}.molecular_weight",
        )
        density = positive_finite_float(
            float(role_data["density_g_ml"]),
            f"{context}.{species_name}.density_g_ml",
        )
        mass_g = float(mole_number) * molecular_weight
        if species_name in neutral_masses:
            neutral_masses[species_name] = neutral_masses[species_name] + mass_g
        else:
            neutral_masses[species_name] = mass_g
        if species_name in SOLVENTS:
            if species_name in solvent_volumes:
                solvent_volumes[species_name] = solvent_volumes[species_name] + mass_g / density
            else:
                solvent_volumes[species_name] = mass_g / density
        elif species_name in ADDITIVES:
            if species_name in additive_masses:
                additive_masses[species_name] = additive_masses[species_name] + mass_g
            else:
                additive_masses[species_name] = mass_g
    if not solvent_volumes:
        return None
    total_solvent_volume = sum(solvent_volumes.values())
    if total_solvent_volume <= 0.0:
        raise ValueError(f"{context}.solvent volume total must be positive")
    solvent_vv = {name: volume / total_solvent_volume for name, volume in sorted(solvent_volumes.items())}
    salt_molarity = _fm_salt_molarity(
        salt_name=salt_name,
        salt_moles=salt_moles,
        neutral_masses=neutral_masses,
        density_g_ml=density_g_ml,
        context=context,
    )
    neutral_mass_total = sum(neutral_masses.values())
    if neutral_mass_total <= 0.0:
        raise ValueError(f"{context}.neutral mass total must be positive")
    salt_data = require_mapping(SALTS, salt_name, "SALTS")
    salt_mw = positive_finite_float(float(salt_data["molecular_weight"]), f"SALTS.{salt_name}.molecular_weight")
    total_solution_mass = neutral_mass_total + salt_moles * salt_mw
    if total_solution_mass <= 0.0:
        raise ValueError(f"{context}.solution mass total must be positive")
    additives = {
        name: mass / total_solution_mass
        for name, mass in sorted(additive_masses.items())
        if mass > 0.0
    }
    return {
        "solvents": solvent_vv,
        "salts": {salt_name: salt_molarity},
        "additives": additives,
    }


def _fm_density_for_recipe_conversion(aux_observables: Mapping[str, Any]) -> float:
    if "density_gcm3" in aux_observables:
        return positive_finite_float(float(aux_observables["density_gcm3"]), "aux_observables.density_gcm3")
    return 0.0


def _normalized_nonnegative_vector(values: np.ndarray, context: str) -> np.ndarray:
    parsed = np.asarray(values, dtype=np.float64)
    if parsed.ndim != 1:
        raise ValueError(f"{context} must be one-dimensional")
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{context} must be finite")
    if np.any(parsed < 0.0):
        raise ValueError(f"{context} must be nonnegative")
    total = float(np.sum(parsed))
    if total <= 0.0:
        raise ValueError(f"{context} total must be positive")
    return parsed / total


def _species_moles(
    species_names: Sequence[str],
    mole_numbers: np.ndarray,
    species_name: str,
) -> float:
    total = 0.0
    for name, mole_number in zip(species_names, mole_numbers):
        if name == species_name:
            total += float(mole_number)
    return total


def _neutral_role_data(species_name: str, context: str) -> Mapping[str, Any]:
    if species_name in SOLVENTS:
        return require_mapping(SOLVENTS, species_name, f"{context}.SOLVENTS")
    if species_name in ADDITIVES:
        return require_mapping(ADDITIVES, species_name, f"{context}.ADDITIVES")
    raise ValueError(f"{context}.{species_name} is not a registered neutral species")


def _fm_salt_molarity(
    salt_name: str,
    salt_moles: float,
    neutral_masses: Mapping[str, float],
    density_g_ml: float,
    context: str,
) -> float:
    salt_data = require_mapping(SALTS, salt_name, "SALTS")
    salt_mw = positive_finite_float(float(salt_data["molecular_weight"]), f"SALTS.{salt_name}.molecular_weight")
    salt_mass = salt_moles * salt_mw
    neutral_mass = sum(neutral_masses.values())
    if density_g_ml > 0.0:
        volume_l = (neutral_mass + salt_mass) / density_g_ml / 1000.0
    else:
        salt_density = positive_finite_float(float(salt_data["density_g_ml"]), f"SALTS.{salt_name}.density_g_ml")
        neutral_volume_ml = 0.0
        for species_name, mass in neutral_masses.items():
            role_data = _neutral_role_data(species_name, context)
            component_density = positive_finite_float(
                float(role_data["density_g_ml"]),
                f"{context}.{species_name}.density_g_ml",
            )
            neutral_volume_ml += mass / component_density
        volume_l = (neutral_volume_ml + salt_mass / salt_density) / 1000.0
    if volume_l <= 0.0:
        raise ValueError(f"{context}.solution volume must be positive")
    return positive_finite_float(salt_moles / volume_l, f"{context}.{salt_name}.molarity")


def _single_salt_molarity(recipe: Mapping[str, Any], context: str) -> float:
    salts = require_mapping(recipe, "salts", context)
    if len(salts) != 1:
        raise ValueError(f"{context}.salts must contain exactly one salt")
    return positive_finite_float(float(next(iter(salts.values()))), f"{context}.salt_molarity")


def _transport_target_rows() -> tuple[MechanisticRow, ...]:
    target_table = compute_tplus_for_all_recipes(T_REF_K)
    rows: list[MechanisticRow] = []
    for row_index, (recipe, tplus_target) in enumerate(zip(target_table["recipes"], target_table["targets"])):
        deff_target = compute_deff_for_recipe(recipe, T_REF_K)
        if deff_target is None:
            continue
        properties = {
            "D_salt_m2_s": deff_target.d_m2_s,
            "t_plus": tplus_target.t_plus,
        }
        row = _row_from_recipe_properties(
            "transport_targets2019",
            row_index,
            recipe,
            properties,
            T_REF_K,
        )
        if row.has_cation_self_current > 0.0 and row.has_anion_self_current > 0.0:
            rows.append(row)
    if not rows:
        raise ValueError("No transport_targets2019 rows loaded")
    return tuple(rows)


def _rows_from_entries(source: str, entries: Sequence[Any]) -> tuple[MechanisticRow, ...]:
    rows: list[MechanisticRow] = []
    for row_index, entry in enumerate(entries):
        recipe = require_mapping(entry._asdict(), "recipe", f"{source}[{row_index}]")
        properties = require_mapping(entry._asdict(), "properties", f"{source}[{row_index}]")
        if "T_K" not in properties:
            raise ValueError(f"{source}[{row_index}].properties.T_K is required")
        temperature_K = positive_finite_float(
            float(properties["T_K"]),
            f"{source}[{row_index}].properties.T_K",
        )
        rows.append(_row_from_recipe_properties(source, row_index, recipe, properties, temperature_K))
    if not rows:
        raise ValueError(f"No rows loaded from {source}")
    return tuple(rows)


def _row_from_recipe_properties(
    source: str,
    row_index: int,
    recipe: Mapping[str, Any],
    properties: Mapping[str, Any],
    temperature_K: float,
) -> MechanisticRow:
    _validate_recipe_species(recipe, f"{source}[{row_index}]")
    sigma = 1.0
    conductivity_mask = 0.0
    if "conductivity_mS_cm" in properties:
        sigma = positive_finite_float(
            float(properties["conductivity_mS_cm"]),
            f"{source}[{row_index}].properties.conductivity_mS_cm",
        )
        if DATASET_MANIFEST[source].train_conductivity:
            conductivity_mask = 1.0
    density = 1.0
    density_mask = 0.0
    if "density" in properties:
        density = positive_finite_float(
            float(properties["density"]),
            f"{source}[{row_index}].properties.density",
        )
        density_mask = 1.0
    viscosity, viscosity_mask = _optional_positive_property(
        properties,
        ("viscosity_cP", "viscosity_mPa_s", "viscosity_mPas"),
        f"{source}[{row_index}].properties",
    )
    dielectric, dielectric_mask = _optional_positive_property(
        properties,
        ("dielectric", "epsilon_r", "dielectric_constant"),
        f"{source}[{row_index}].properties",
    )
    cation_self, cation_self_mask = _optional_positive_property(
        properties,
        ("cation_self_current_mS_cm", "cation_self_conductivity_mS_cm"),
        f"{source}[{row_index}].properties",
    )
    anion_self, anion_self_mask = _optional_positive_property(
        properties,
        ("anion_self_current_mS_cm", "anion_self_conductivity_mS_cm"),
        f"{source}[{row_index}].properties",
    )
    if cation_self_mask == 0.0 and anion_self_mask == 0.0:
        cation_self, cation_self_mask, anion_self, anion_self_mask = _self_current_labels_from_diffusion(
            recipe,
            properties,
            temperature_K,
            f"{source}[{row_index}].properties",
        )
    cation_anion_distinct, cation_anion_distinct_mask = _optional_finite_property(
        properties,
        ("cation_anion_distinct_mS_cm", "cation_anion_cross_current_mS_cm"),
        f"{source}[{row_index}].properties",
    )
    current_distinct, current_distinct_mask = _optional_finite_property(
        properties,
        ("current_distinct_mS_cm", "distinct_current_mS_cm"),
        f"{source}[{row_index}].properties",
    )
    association_fraction, association_fraction_mask = _optional_fraction_property(
        properties,
        ("association_fraction", "contact_pair_fraction"),
        f"{source}[{row_index}].properties",
    )
    return MechanisticRow(
        source=source,
        row_index=row_index,
        recipe=recipe,
        conductivity_mS_cm=sigma,
        has_conductivity=conductivity_mask,
        density_g_ml=density,
        has_density=density_mask,
        viscosity_cP=viscosity,
        has_viscosity=viscosity_mask,
        dielectric=dielectric,
        has_dielectric=dielectric_mask,
        cation_self_current_mS_cm=cation_self,
        has_cation_self_current=cation_self_mask,
        anion_self_current_mS_cm=anion_self,
        has_anion_self_current=anion_self_mask,
        cation_anion_distinct_mS_cm=cation_anion_distinct,
        has_cation_anion_distinct=cation_anion_distinct_mask,
        current_distinct_mS_cm=current_distinct,
        has_current_distinct=current_distinct_mask,
        association_fraction=association_fraction,
        has_association_fraction=association_fraction_mask,
        temperature_K=temperature_K,
        recipe_key=recipe_key(recipe),
    )


def _validate_recipe_species(recipe: Mapping[str, Any], context: str) -> None:
    solvents = require_mapping(recipe, "solvents", f"{context}.recipe")
    salts = require_mapping(recipe, "salts", f"{context}.recipe")
    additives = require_mapping(recipe, "additives", f"{context}.recipe")
    for role_name, component_mapping in (
        ("solvents", solvents),
        ("salts", salts),
        ("additives", additives),
    ):
        for name in component_mapping:
            if name not in KNOWN_SPECIES:
                raise ValueError(f"{context}.recipe.{role_name}.{name} is not in species_data")


def _recipe_uses_registered_roles(recipe: Mapping[str, Any]) -> bool:
    solvents = require_mapping(recipe, "solvents", "recipe")
    salts = require_mapping(recipe, "salts", "recipe")
    additives = require_mapping(recipe, "additives", "recipe")
    for role_name, component_mapping in (
        ("solvents", solvents),
        ("salts", salts),
        ("additives", additives),
    ):
        for name in component_mapping:
            if name not in ROLE_SPECIES[role_name]:
                return False
    return True


def _molality_to_molarity(
    molality_mol_kg: float,
    solution_density_g_ml: float,
    salt_name: str,
) -> float:
    salt_data = require_mapping(SALTS, salt_name, "SALTS")
    salt_mw_g_mol = positive_finite_float(
        float(salt_data["molecular_weight"]),
        f"SALTS.{salt_name}.molecular_weight",
    )
    solution_mass_g = 1000.0 + molality_mol_kg * salt_mw_g_mol
    solution_volume_L = solution_mass_g / (solution_density_g_ml * 1000.0)
    return positive_finite_float(
        molality_mol_kg / solution_volume_L,
        f"oedb_li_aux.{salt_name}.molarity",
    )


def _self_current_from_diffusivity(
    salt_molarity_M: float,
    diffusivity_m2_s: float,
    temperature_K: float,
    context: str,
) -> float:
    diffusivity = positive_finite_float(diffusivity_m2_s, context)
    concentration_mol_m3 = salt_molarity_M * 1000.0
    prefactor = 10.0 * FARADAY_C_MOL * FARADAY_C_MOL * concentration_mol_m3 / (
        GAS_CONSTANT_J_MOL_K * temperature_K
    )
    return prefactor * diffusivity


def _association_fraction_from_coordination(
    cation_anion_coordination: float,
    cation_solvent_coordination: float,
    context: str,
) -> float:
    anion_cn = _nonnegative_finite_float(cation_anion_coordination, f"{context}.cation_anion_cn")
    solvent_cn = _nonnegative_finite_float(cation_solvent_coordination, f"{context}.cation_solvent_cn")
    total_cn = anion_cn + solvent_cn
    if total_cn <= 0.0:
        raise ValueError(f"{context}.coordination total must be positive")
    return anion_cn / total_cn


def _self_current_labels_from_diffusion(
    recipe: Mapping[str, Any],
    properties: Mapping[str, Any],
    temperature_K: float,
    context: str,
) -> tuple[float, float, float, float]:
    if "D_salt_m2_s" not in properties:
        return 0.0, 0.0, 0.0, 0.0
    if "t_plus" not in properties:
        return 0.0, 0.0, 0.0, 0.0
    salts = require_mapping(recipe, "salts", "recipe")
    if len(salts) != 1:
        return 0.0, 0.0, 0.0, 0.0
    salt_molarity_M = positive_finite_float(float(next(iter(salts.values()))), f"{context}.salt_molarity_M")
    d_salt = positive_finite_float(float(properties["D_salt_m2_s"]), f"{context}.D_salt_m2_s")
    t_plus = float(properties["t_plus"])
    if not np.isfinite(t_plus) or t_plus <= 0.0 or t_plus >= 1.0:
        raise ValueError(f"{context}.t_plus must be finite and between 0 and 1, got {t_plus!r}")
    c_mol_m3 = salt_molarity_M * 1000.0
    d_cation = 2.0 * t_plus * d_salt
    d_anion = 2.0 * (1.0 - t_plus) * d_salt
    prefactor = 10.0 * FARADAY_C_MOL * FARADAY_C_MOL * c_mol_m3 / (
        GAS_CONSTANT_J_MOL_K * temperature_K
    )
    return prefactor * d_cation, 1.0, prefactor * d_anion, 1.0


def _fec_loading(recipe: Mapping[str, Any]) -> float:
    additives = require_mapping(recipe, "additives", "recipe")
    if "FEC" not in additives:
        return 0.0
    return float(additives["FEC"])


def _optional_positive_property(
    properties: Mapping[str, Any],
    names: tuple[str, ...],
    context: str,
) -> tuple[float, float]:
    value, mask = _optional_finite_property(properties, names, context)
    if mask == 0.0:
        return value, mask
    parsed = positive_finite_float(value, f"{context}.{_matched_property_name(properties, names)}")
    return parsed, mask


def _optional_fraction_property(
    properties: Mapping[str, Any],
    names: tuple[str, ...],
    context: str,
) -> tuple[float, float]:
    value, mask = _optional_finite_property(properties, names, context)
    if mask == 0.0:
        return value, mask
    matched_name = _matched_property_name(properties, names)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{context}.{matched_name} must be between 0 and 1, got {value!r}")
    return value, mask


def _optional_finite_property(
    properties: Mapping[str, Any],
    names: tuple[str, ...],
    context: str,
) -> tuple[float, float]:
    for name in names:
        if name in properties:
            value = properties[name]
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}.{name} must be numeric, got {value!r}") from exc
            if not np.isfinite(parsed):
                raise ValueError(f"{context}.{name} must be finite, got {parsed!r}")
            return parsed, 1.0
    return 0.0, 0.0


def _nonnegative_finite_float(value: float, context: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{context} must be finite and nonnegative, got {parsed!r}")
    return parsed


def _matched_property_name(properties: Mapping[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        if name in properties:
            return name
    raise ValueError(f"None of {names!r} were present")
