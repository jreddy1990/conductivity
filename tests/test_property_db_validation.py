from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from data.electrolyte_property_db import DATA as PROPERTY_DB
from conductivity.physical_library.library_io import load_physical_library
from conductivity.physical_library import property_db_validation
from conductivity.physical_library.property_db_validation import (
    _UnsupportedRecipe,
    _property_validation_classification_from_result,
    _supported_recipe_from_property_db_entry,
    _validation_row_from_result,
    validate_property_db_supported_conductivity_rows,
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


@pytest.mark.parametrize(
    "classification",
    (
        "population_operator_missing",
        "state_mobility_operator_missing",
        "memory_dirichlet_operator_missing",
        "transition_moment_operator_missing",
        "dissociation_operator_missing",
        "state_resolved_born_missing",
        "unsupported_species",
    ),
)
def test_property_validation_classifies_production_primitive_owner_ledger(
    classification: str,
) -> None:
    conductivity_result = SimpleNamespace(
        effect_attribution={
            "primitive_prediction_readiness_status": "incomplete",
            "primitive_prediction_scalar_label": "diagnostic",
            "primitive_prediction_not_complete_reasons": (classification,),
            "final_conductivity_correction_mS_cm": 1000.0,
        },
    )

    assert (
        _property_validation_classification_from_result(conductivity_result)
        == classification
    )


def test_property_validation_classification_rejects_multiple_primitive_owners() -> None:
    conductivity_result = SimpleNamespace(
        effect_attribution={
            "primitive_prediction_readiness_status": "incomplete",
            "primitive_prediction_scalar_label": "diagnostic",
            "primitive_prediction_not_complete_reasons": (
                "population_operator_missing",
                "state_mobility_operator_missing",
            ),
        }
    )

    with pytest.raises(ValueError, match="multiple primitive owner failures"):
        _property_validation_classification_from_result(conductivity_result)


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


def test_supported_rows_execute_in_parallel_and_capture_row_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker_barrier = threading.Barrier(3, timeout=1.0)

    def fake_load_physical_library(physical_library_root: Path):
        return SimpleNamespace(species_records={"EC": {}, "Li+": {}, "PF6-": {}})

    def fake_compute_conductivity_from_recipe(
        recipe: Path,
        library_root: Path,
        numerical_options,
    ):
        entry_index = int(recipe.stem.removeprefix("property_db_"))
        worker_barrier.wait()
        if entry_index == 1:
            raise RuntimeError("row model failed")
        return SimpleNamespace(
            sigma_mS_cm=12.0,
            effect_attribution={
                "primitive_prediction_readiness_status": "complete",
                "primitive_prediction_scalar_label": "primitive_prediction",
            },
        )

    monkeypatch.setattr(
        property_db_validation, "load_physical_library", fake_load_physical_library
    )
    monkeypatch.setattr(
        property_db_validation,
        "compute_conductivity_from_recipe",
        fake_compute_conductivity_from_recipe,
    )
    supported_recipe = {
        "solvents": {"EC": 1.0},
        "salts": {"LiPF6": 1.0},
        "additives": {},
    }
    entries = [
        {"recipe": supported_recipe, "properties": {"conductivity_mS_cm": 10.0}},
        {"recipe": supported_recipe, "properties": {"conductivity_mS_cm": 11.0}},
        {"recipe": supported_recipe, "properties": {"conductivity_mS_cm": 9.0}},
        {"recipe": supported_recipe, "properties": {}},
        {
            "recipe": {
                "solvents": {"DMC": 1.0},
                "salts": {"LiPF6": 1.0},
                "additives": {},
            },
            "properties": {"conductivity_mS_cm": 8.0},
        },
    ]

    summary = validate_property_db_supported_conductivity_rows(
        property_db_entries=entries,
        physical_library_root=tmp_path,
        numerical_options=SimpleNamespace(),
    )

    assert [row.entry_index for row in summary.rows] == [0, 2]
    assert [progress.entry_index for progress in summary.progress] == [0, 1, 2, 3, 4]
    assert summary.evaluated_entry_count == 2
    assert summary.failed_entry_count == 1
    assert summary.skipped_entry_count == 2
    assert summary.classification_counts == {
        "evaluated": 2,
        "evaluation_failure": 1,
        "missing_conductivity": 1,
        "unsupported_species": 1,
    }
    assert summary.progress[1].detail == "RuntimeError: row model failed"
    assert summary.progress[0].detail == "primitive_residual_mS_cm=2"
