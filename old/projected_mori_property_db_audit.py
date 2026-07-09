"""Empirical audit for the recipe-generated projected Mori conductivity readout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from conductivity.physical_library.library_io import load_physical_library
from data.species_data import SALTS
from utils.strict_validation import require_float, require_key, strict_mapping


PHYSICAL_LIBRARY_ROOT = Path("conductivity/physical_library")


@dataclass(frozen=True)
class ProjectedMoriPropertyDbRow:
    row_id: int
    empirical_sigma_mS_cm: float
    projected_mori_sigma_mS_cm: float
    residual_mS_cm: float
    projected_mori_sigma_S_m: float
    axis_conductivity_S_m: tuple[float, float, float]
    quadratic_form_by_axis: tuple[float, float, float]
    projected_basis_dimension: int
    energy_min_eigenvalue: float
    energy_max_eigenvalue: float


@dataclass(frozen=True)
class ProjectedMoriPropertyDbFailure:
    row_id: int
    error: str


@dataclass(frozen=True)
class ProjectedMoriPropertyDbAuditResult:
    labeled_rows: int
    evaluated_rows: int
    failed_rows: int
    mae_mS_cm: float
    rmse_mS_cm: float
    bias_mS_cm: float
    mape_percent: float
    r2: float
    pearson_r: float
    rows: tuple[ProjectedMoriPropertyDbRow, ...]
    failures: tuple[ProjectedMoriPropertyDbFailure, ...]


def audit_projected_mori_conductivity_against_property_db(
    entries,
    temperature_K: float,
    atmosphere_bath_basis: str,
    relaxation_dynamic_response: str,
    anion_diagonal_relaxation_form_factor: str,
) -> ProjectedMoriPropertyDbAuditResult:
    """Compare recipe-generated projected Mori values to empirical labels."""

    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be positive and finite, got {temperature_K}")

    rows: list[ProjectedMoriPropertyDbRow] = []
    labeled_rows = 0

    for row_id, entry in enumerate(entries):
        entry_sections = _require_projected_mori_entry(entry, row_id)
        properties = entry_sections["properties"]
        if "conductivity_mS_cm" not in properties:
            continue
        labeled_rows += 1
        empirical_sigma_mS_cm = require_float(
            properties,
            "conductivity_mS_cm",
            f"DATA[{row_id}].properties",
        )
        _require_projected_physical_library_payload(row_id, properties)
        recipe = strict_mapping(entry_sections["recipe"], f"DATA[{row_id}].recipe")
        _require_active_recipe_species_records(row_id, recipe)
        from conductivity.old.finite_markov_conductivity import (
            evaluate_finite_markov_conductivity,
        )

        finite_result = evaluate_finite_markov_conductivity(
            recipe,
            temperature_K,
            atmosphere_bath_basis,
            relaxation_dynamic_response,
            anion_diagonal_relaxation_form_factor,
        )
        projected_mori_result = finite_result.projected_mori_conductivity
        energy_eigenvalues = projected_mori_result.energy_eigenvalues
        if not energy_eigenvalues:
            raise ValueError("projected Mori result has no energy eigenvalues")
        projected_mori_sigma_mS_cm = projected_mori_result.sigma_mS_cm
        rows.append(
            ProjectedMoriPropertyDbRow(
                row_id=row_id,
                empirical_sigma_mS_cm=empirical_sigma_mS_cm,
                projected_mori_sigma_mS_cm=projected_mori_sigma_mS_cm,
                residual_mS_cm=projected_mori_sigma_mS_cm - empirical_sigma_mS_cm,
                projected_mori_sigma_S_m=projected_mori_result.sigma_S_m,
                axis_conductivity_S_m=projected_mori_result.axis_conductivity_S_m,
                quadratic_form_by_axis=projected_mori_result.quadratic_form_by_axis,
                projected_basis_dimension=len(energy_eigenvalues),
                energy_min_eigenvalue=float(min(energy_eigenvalues)),
                energy_max_eigenvalue=float(max(energy_eigenvalues)),
            )
        )

    metrics = _projected_mori_property_db_metrics(rows)
    return ProjectedMoriPropertyDbAuditResult(
        labeled_rows=labeled_rows,
        evaluated_rows=len(rows),
        failed_rows=0,
        mae_mS_cm=metrics["mae_mS_cm"],
        rmse_mS_cm=metrics["rmse_mS_cm"],
        bias_mS_cm=metrics["bias_mS_cm"],
        mape_percent=metrics["mape_percent"],
        r2=metrics["r2"],
        pearson_r=metrics["pearson_r"],
        rows=tuple(rows),
        failures=(),
    )


def _require_projected_mori_entry(entry, row_id: int) -> dict[str, dict]:
    entry_mapping = strict_mapping(entry, f"DATA[{row_id}]")
    recipe = strict_mapping(
        require_key(entry_mapping, "recipe", f"DATA[{row_id}]"),
        f"DATA[{row_id}].recipe",
    )
    properties = strict_mapping(
        require_key(entry_mapping, "properties", f"DATA[{row_id}]"),
        f"DATA[{row_id}].properties",
    )
    return {"recipe": recipe, "properties": properties}


def _require_projected_physical_library_payload(row_id: int, properties) -> None:
    properties_mapping = strict_mapping(properties, f"DATA[{row_id}].properties")
    if "projected_primitives" in properties_mapping:
        strict_mapping(
            properties_mapping["projected_primitives"],
            f"DATA[{row_id}].properties.projected_primitives",
        )
        return
    if "projected_generator_inputs" in properties_mapping:
        strict_mapping(
            properties_mapping["projected_generator_inputs"],
            f"DATA[{row_id}].properties.projected_generator_inputs",
        )
        return
    raise ValueError(
        f"DATA[{row_id}] is missing projected_primitives or "
        "projected_generator_inputs; recipe-only conductivity validation requires a "
        "populated full ConductivityPhysicalLibrary"
    )


def _require_active_recipe_species_records(
    row_id: int,
    recipe: dict[str, dict[str, float]],
) -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    missing_species_names = _missing_active_recipe_species_names(
        recipe,
        records.species_records,
    )
    if missing_species_names:
        raise KeyError(
            f"DATA[{row_id}] active recipe species missing from physical library: "
            f"{missing_species_names}"
        )


def _missing_active_recipe_species_names(
    recipe: dict[str, dict[str, float]],
    species_records: dict,
) -> tuple[str, ...]:
    active_species_names = _active_recipe_species_names(recipe)
    return tuple(
        species_name
        for species_name in active_species_names
        if species_name not in species_records
    )


def _active_recipe_species_names(recipe: dict[str, dict[str, float]]) -> tuple[str, ...]:
    active_species_names: list[str] = []
    for section_name in ("solvents", "salts", "additives"):
        section = strict_mapping(
            require_key(recipe, section_name, "canonical_recipe"),
            f"canonical_recipe.{section_name}",
        )
        for species_name, loading in section.items():
            loading_value = float(loading)
            if not math.isfinite(loading_value):
                raise ValueError(
                    f"canonical_recipe.{section_name}.{species_name} must be finite"
                )
            if loading_value > 0.0:
                active_species_names.extend(
                    _physical_library_species_names_for_recipe_loading(
                        section_name,
                        str(species_name),
                    )
                )
    return tuple(sorted(set(active_species_names)))


def _physical_library_species_names_for_recipe_loading(
    section_name: str,
    species_name: str,
) -> tuple[str, ...]:
    if section_name != "salts":
        return (species_name,)
    if species_name not in SALTS:
        return (species_name,)
    salt_record = strict_mapping(SALTS[species_name], f"SALTS.{species_name}")
    cation_name = str(require_key(salt_record, "cation", f"SALTS.{species_name}"))
    anion_name = str(require_key(salt_record, "anion", f"SALTS.{species_name}"))
    return (f"{cation_name}+", anion_name)


def _projected_mori_property_db_metrics(
    rows: Sequence[ProjectedMoriPropertyDbRow],
) -> dict[str, float]:
    if len(rows) < 2:
        raise ValueError(
            "projected Mori property-DB audit requires at least two evaluated rows"
        )

    empirical_values = np.asarray(
        [row.empirical_sigma_mS_cm for row in rows],
        dtype=float,
    )
    projected_mori_values = np.asarray(
        [row.projected_mori_sigma_mS_cm for row in rows],
        dtype=float,
    )
    residuals = projected_mori_values - empirical_values
    total_sum_squares = float(np.sum((empirical_values - float(np.mean(empirical_values))) ** 2))
    if total_sum_squares <= 0.0:
        raise ValueError("projected Mori property-DB audit empirical labels have zero variance")
    residual_sum_squares = float(np.sum(residuals * residuals))
    pearson_r = float(np.corrcoef(empirical_values, projected_mori_values)[0, 1])
    return {
        "mae_mS_cm": float(np.mean(np.abs(residuals))),
        "rmse_mS_cm": float(math.sqrt(float(np.mean(residuals * residuals)))),
        "bias_mS_cm": float(np.mean(residuals)),
        "mape_percent": float(np.mean(np.abs(residuals / empirical_values)) * 100.0),
        "r2": float(1.0 - residual_sum_squares / total_sum_squares),
        "pearson_r": pearson_r,
    }
