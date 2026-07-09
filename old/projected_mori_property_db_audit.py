"""Empirical audit for the recipe-generated projected Mori conductivity readout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from conductivity.finite_markov_conductivity import evaluate_finite_markov_conductivity
from conductivity.finite_markov_dataset_audit import (
    _require_entry,
    canonicalize_empirical_recipe,
)
from utils.strict_validation import require_float


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
    failures: list[ProjectedMoriPropertyDbFailure] = []
    labeled_rows = 0

    for row_id, entry in enumerate(entries):
        entry_sections = _require_entry(entry, row_id)
        properties = entry_sections["properties"]
        if "conductivity_mS_cm" not in properties:
            continue
        labeled_rows += 1
        try:
            empirical_sigma_mS_cm = require_float(
                properties,
                "conductivity_mS_cm",
                f"DATA[{row_id}].properties",
            )
            canonicalization = canonicalize_empirical_recipe(entry_sections["recipe"])
            finite_result = evaluate_finite_markov_conductivity(
                canonicalization.recipe,
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
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                ProjectedMoriPropertyDbFailure(
                    row_id=row_id,
                    error=str(exc),
                )
            )

    metrics = _projected_mori_property_db_metrics(rows)
    return ProjectedMoriPropertyDbAuditResult(
        labeled_rows=labeled_rows,
        evaluated_rows=len(rows),
        failed_rows=len(failures),
        mae_mS_cm=metrics["mae_mS_cm"],
        rmse_mS_cm=metrics["rmse_mS_cm"],
        bias_mS_cm=metrics["bias_mS_cm"],
        mape_percent=metrics["mape_percent"],
        r2=metrics["r2"],
        pearson_r=metrics["pearson_r"],
        rows=tuple(rows),
        failures=tuple(failures),
    )


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
