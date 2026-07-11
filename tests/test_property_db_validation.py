from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from data.electrolyte_property_db import DATA as PROPERTY_DB
from conductivity.physical_library.library_io import load_physical_library
from conductivity.physical_library.property_db_validation import (
    _UnsupportedRecipe,
    _supported_recipe_from_property_db_entry,
    _validation_row_from_result,
)

PHYSICAL_LIBRARY_ROOT = Path("conductivity/physical_library")


@pytest.mark.parametrize(
    ("readiness_status", "scalar_label"),
    (("incomplete", "diagnostic"), ("complete", "diagnostic")),
)
def test_property_db_validation_refuses_diagnostic_result(
    readiness_status: str,
    scalar_label: str,
) -> None:
    conductivity_result = SimpleNamespace(
        sigma_mS_cm=12.0,
        effect_attribution={
            "primitive_prediction_readiness_status": readiness_status,
            "primitive_prediction_scalar_label": scalar_label,
        },
    )
    supported_recipe = SimpleNamespace(
        solvents_vv={"EC": 1.0},
        salts_mol_l={"Li+": 1.0, "PF6-": 1.0},
        additives_weight_fraction={},
    )

    with pytest.raises(ValueError, match="requires a complete primitive prediction"):
        _validation_row_from_result(
            entry_index=0,
            supported_recipe=supported_recipe,
            measured_conductivity_mS_cm=10.0,
            conductivity_result=conductivity_result,
        )


def test_property_db_validation_refuses_missing_prediction() -> None:
    conductivity_result = SimpleNamespace(
        sigma_mS_cm=None,
        effect_attribution={
            "primitive_prediction_readiness_status": "complete",
            "primitive_prediction_scalar_label": "primitive_prediction",
        },
    )
    supported_recipe = SimpleNamespace(
        solvents_vv={"EC": 1.0},
        salts_mol_l={"Li+": 1.0, "PF6-": 1.0},
        additives_weight_fraction={},
    )

    with pytest.raises(ValueError, match="missing conductivity prediction"):
        _validation_row_from_result(
            entry_index=0,
            supported_recipe=supported_recipe,
            measured_conductivity_mS_cm=10.0,
            conductivity_result=conductivity_result,
        )


def test_every_property_db_species_has_a_physical_library_record() -> None:
    physical_library_records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    supported_species_names = frozenset(physical_library_records.species_records)
    supported_recipes = tuple(
        _supported_recipe_from_property_db_entry(
            entry_index=entry_index,
            recipe_record=property_db_entry["recipe"],
            supported_species_names=supported_species_names,
        )
        for entry_index, property_db_entry in enumerate(PROPERTY_DB)
    )
    unsupported_rows = [
        (entry_index, supported_recipe.unsupported_species_key)
        for entry_index, supported_recipe in enumerate(supported_recipes)
        if isinstance(supported_recipe, _UnsupportedRecipe)
    ]
    assert unsupported_rows == []
