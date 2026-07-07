"""Validate projected analytical conductivity predictions against property DB rows."""

from __future__ import annotations

import json

import numpy as np

from conductivity.electrolyte_utils_features import get_component_fractions
from conductivity.projected_analytical_conductivity import (
    compute_projected_analytical_conductivity_from_recipe,
)
from data.electrolyte_property_db import DATA
from species_fns import ADDITIVES, SALTS, SOLVENTS, get_species_property
from utils.strict_validation import require_key, strict_mapping


def main() -> None:
    row_results = tuple(
        _evaluate_property_db_row(row_index, row_mapping)
        for row_index, row_mapping in enumerate(DATA)
        if "conductivity_mS_cm" in strict_mapping(
            require_key(row_mapping, "properties", f"DATA[{row_index}]"),
            f"DATA[{row_index}].properties",
        )
    )
    successful_results = tuple(
        row_result
        for row_result in row_results
        if row_result["predicted_mS_cm"] is not None
    )
    failed_results = tuple(
        row_result
        for row_result in row_results
        if row_result["predicted_mS_cm"] is None
    )
    print("projected_analytical_property_db_validation")
    print("model=conductivity.projected_analytical_conductivity")
    print(f"source_labeled_rows={len(row_results)}")
    print(
        "formulation_group_count="
        f"{len({str(row_result['recipe_key']) for row_result in row_results})}"
    )
    print(f"evaluated_rows={len(successful_results)}")
    print(f"failed_rows={len(failed_results)}")
    if successful_results:
        _print_metrics(successful_results, len(successful_results))
    if failed_results:
        print("failed_rows")
        for row_result in failed_results:
            print(
                f"row_id={row_result['row_id']} "
                f"failure={row_result['failure']}"
            )


def _evaluate_property_db_row(row_index: int, row_mapping):
    row = strict_mapping(row_mapping, f"DATA[{row_index}]")
    recipe = strict_mapping(
        require_key(row, "recipe", f"DATA[{row_index}]"),
        f"DATA[{row_index}].recipe",
    )
    properties = strict_mapping(
        require_key(row, "properties", f"DATA[{row_index}]"),
        f"DATA[{row_index}].properties",
    )
    empirical_mS_cm = float(
        require_key(
            properties,
            "conductivity_mS_cm",
            f"DATA[{row_index}].properties",
        )
    )
    try:
        canonical_recipe = _canonical_projected_recipe_from_property_db_row(
            recipe,
            properties,
        )
        projected_result = compute_projected_analytical_conductivity_from_recipe(
            canonical_recipe
        )
    except Exception as exc:
        return {
            "row_id": row_index,
            "recipe_key": json.dumps(recipe, sort_keys=True, separators=(",", ":")),
            "empirical_mS_cm": empirical_mS_cm,
            "predicted_mS_cm": None,
            "residual_mS_cm": None,
            "failure": str(exc),
        }
    predicted_mS_cm = float(projected_result["sigma_mS_cm"])
    return {
        "row_id": row_index,
        "recipe_key": json.dumps(canonical_recipe, sort_keys=True, separators=(",", ":")),
        "empirical_mS_cm": empirical_mS_cm,
        "predicted_mS_cm": predicted_mS_cm,
        "residual_mS_cm": predicted_mS_cm - empirical_mS_cm,
        "failure": None,
    }


def _canonical_projected_recipe_from_property_db_row(recipe, properties):
    raw_recipe = strict_mapping(recipe, "property_db.recipe")
    raw_properties = strict_mapping(properties, "property_db.properties")
    raw_solvents = strict_mapping(
        require_key(raw_recipe, "solvents", "property_db.recipe"),
        "property_db.recipe.solvents",
    )
    raw_salts = strict_mapping(
        require_key(raw_recipe, "salts", "property_db.recipe"),
        "property_db.recipe.salts",
    )
    raw_additives = strict_mapping(
        require_key(raw_recipe, "additives", "property_db.recipe"),
        "property_db.recipe.additives",
    )
    solvent_fractions: dict[str, float] = {}
    salt_molarities: dict[str, float] = {}
    additive_fractions: dict[str, float] = {}
    for species_name, raw_fraction in raw_solvents.items():
        fraction = float(raw_fraction)
        if species_name in SOLVENTS:
            solvent_fractions[str(species_name)] = fraction
        elif species_name in ADDITIVES:
            additive_fractions[str(species_name)] = (
                additive_fractions[str(species_name)] + fraction
                if str(species_name) in additive_fractions
                else fraction
            )
        else:
            raise ValueError(f"property DB solvent species {species_name} is not registered")
    for species_name, raw_molarity in raw_salts.items():
        molarity = float(raw_molarity)
        if species_name in SALTS:
            salt_molarities[str(species_name)] = molarity
        elif species_name in ADDITIVES and _species_is_projected_ionic_source(
            str(species_name)
        ):
            density_g_ml = _property_db_density_g_ml(raw_properties, raw_recipe)
            additive_fraction = _molarity_to_weight_fraction(
                str(species_name),
                molarity,
                density_g_ml,
            )
            additive_fractions[str(species_name)] = (
                additive_fractions[str(species_name)] + additive_fraction
                if str(species_name) in additive_fractions
                else additive_fraction
            )
        else:
            raise ValueError(f"property DB salt species {species_name} is not registered")
    for species_name, raw_fraction in raw_additives.items():
        fraction = float(raw_fraction)
        if species_name in ADDITIVES:
            additive_fractions[str(species_name)] = (
                additive_fractions[str(species_name)] + fraction
                if str(species_name) in additive_fractions
                else fraction
            )
        else:
            raise ValueError(
                f"property DB additive species {species_name} is not registered"
            )
    solvent_total = float(sum(solvent_fractions.values()))
    if solvent_total <= 0.0:
        raise ValueError("property DB recipe has no positive solvent fraction")
    normalized_solvents = {
        species_name: fraction / solvent_total
        for species_name, fraction in solvent_fractions.items()
    }
    return {
        "solvents": normalized_solvents,
        "salts": salt_molarities,
        "additives": additive_fractions,
    }


def _property_db_density_g_ml(properties, recipe) -> float:
    if "density" in properties:
        return float(properties["density"])
    component_fractions = get_component_fractions(dict(recipe))
    if not component_fractions:
        raise ValueError("property DB recipe has no liquid components for density")
    inverse_density = 0.0
    for species_name, fraction in component_fractions.items():
        if _species_is_projected_ionic_source(str(species_name)):
            continue
        density = get_species_property(str(species_name), "density_g_ml")
        if density is None:
            raise ValueError(f"species {species_name} missing density_g_ml")
        inverse_density += float(fraction) / float(density)
    if inverse_density <= 0.0:
        raise ValueError("property DB density estimate denominator is non-positive")
    return 1.0 / inverse_density


def _molarity_to_weight_fraction(
    species_name: str,
    molarity_mol_l: float,
    density_g_ml: float,
) -> float:
    molecular_weight_g_mol = get_species_property(species_name, "molecular_weight")
    if molecular_weight_g_mol is None:
        raise ValueError(f"species {species_name} missing molecular_weight")
    return float(molarity_mol_l) * float(molecular_weight_g_mol) / (
        float(density_g_ml) * 1000.0
    )


def _species_is_projected_ionic_source(species_name: str) -> bool:
    configured_value = get_species_property(species_name, "provides_ionic_conductivity")
    if configured_value is not None:
        if not isinstance(configured_value, bool):
            raise ValueError(
                f"{species_name}.provides_ionic_conductivity must be boolean"
            )
        return configured_value
    return (
        get_species_property(species_name, "Lambda_0") is not None
        and get_species_property(species_name, "anion_charge") is not None
    )


def _print_metrics(successful_results: tuple[dict, ...], worst_row_count: int) -> None:
    empirical = np.asarray(
        [float(row_result["empirical_mS_cm"]) for row_result in successful_results],
        dtype=float,
    )
    predicted = np.asarray(
        [float(row_result["predicted_mS_cm"]) for row_result in successful_results],
        dtype=float,
    )
    residual = predicted - empirical
    if len(empirical) > 1 and np.std(empirical) > 0.0 and np.std(predicted) > 0.0:
        pearson_r = float(np.corrcoef(empirical, predicted)[0, 1])
    else:
        pearson_r = 0.0
    print(f"mae_mS_cm={float(np.mean(np.abs(residual))):.9g}")
    print(f"rmse_mS_cm={float(np.sqrt(np.mean(residual * residual))):.9g}")
    print(f"bias_mS_cm={float(np.mean(residual)):.9g}")
    print(f"pearson_r={pearson_r:.9g}")
    print(f"maximum_abs_residual_mS_cm={float(np.max(np.abs(residual))):.9g}")
    print("worst_rows")
    for residual_index in np.argsort(-np.abs(residual))[:worst_row_count]:
        row_result = successful_results[int(residual_index)]
        print(
            f"row_id={row_result['row_id']} "
            f"empirical_mS_cm={float(row_result['empirical_mS_cm']):.9g} "
            f"predicted_mS_cm={float(row_result['predicted_mS_cm']):.9g} "
            f"residual_mS_cm={float(row_result['residual_mS_cm']):.9g}"
        )


if __name__ == "__main__":
    main()
