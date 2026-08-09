from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import threading
from types import SimpleNamespace

import pytest

from data.electrolyte_property_db import DATA as PROPERTY_DB
from data.species_data import (
    ADDITIVE_SPECIES,
    ADDITIVES,
    ELECTROLYTE_SPECIES,
    SALT_SPECIES,
    SALTS,
)
from conductivity.physical_library import generator_construction
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


@pytest.mark.parametrize("species_name", ("LiDFOB", "LiBOB"))
def test_dual_designation_species_use_one_canonical_record(species_name: str) -> None:
    assert species_name in SALT_SPECIES
    assert species_name in ADDITIVE_SPECIES
    assert SALTS[species_name] is ELECTROLYTE_SPECIES[species_name]
    assert ADDITIVES[species_name] is ELECTROLYTE_SPECIES[species_name]
    assert "type" not in ELECTROLYTE_SPECIES[species_name]


def test_every_additive_declares_current_collector_passivation_status() -> None:
    expected_passivating_additives = {
        "LiBOB",
        "LiDFOB",
        "LiPO2F2",
        "NaDFOB",
    }

    assert all(
        type(additive_record["passivation_film_former"]) is bool
        for additive_record in ADDITIVES.values()
    )
    assert {
        species_name
        for species_name, additive_record in ADDITIVES.items()
        if additive_record["passivation_film_former"]
    } == expected_passivating_additives


def test_conductivity_cache_serializes_identical_recipe_computation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_key = ("cache_stampede_test", str(tmp_path))
    computation_started = threading.Event()
    release_computation = threading.Event()
    duplicate_computation_started = threading.Event()
    computation_calls = Queue()
    result = SimpleNamespace(value=1)
    computed_results = Queue()

    def fake_cache_key(recipe, library_root, numerical_options):
        return cache_key

    def fake_cache_path(requested_cache_key):
        assert requested_cache_key == cache_key
        return tmp_path / "result.pkl"

    def fake_compute(recipe, library_root, numerical_options):
        computation_calls.put(1)
        if computation_calls.qsize() > 1:
            duplicate_computation_started.set()
        computation_started.set()
        assert release_computation.wait(timeout=1.0)
        return result

    def fake_validate(conductivity_result) -> None:
        assert conductivity_result.value == 1

    monkeypatch.setattr(
        generator_construction,
        "_conductivity_result_cache_key",
        fake_cache_key,
    )
    monkeypatch.setattr(
        generator_construction,
        "_persistent_conductivity_cache_path",
        fake_cache_path,
    )
    monkeypatch.setattr(
        generator_construction,
        "_compute_conductivity_from_recipe_uncached",
        fake_compute,
    )
    monkeypatch.setattr(
        generator_construction,
        "_validate_cached_conductivity_result",
        fake_validate,
    )
    with generator_construction._CONDUCTIVITY_RESULT_CACHE_LOCK:
        assert cache_key not in generator_construction._CONDUCTIVITY_RESULT_CACHE
        assert cache_key not in generator_construction._CONDUCTIVITY_RESULT_KEY_LOCKS

    def compute_result() -> None:
        computed_results.put(
            generator_construction.compute_analytical_conductivity_from_recipe(
                Path("recipe.yaml"),
                Path("library"),
                SimpleNamespace(),
            )
        )

    first_thread = threading.Thread(target=compute_result)
    second_thread = threading.Thread(target=compute_result)
    first_thread.start()
    assert computation_started.wait(timeout=1.0)
    second_thread.start()
    assert not duplicate_computation_started.wait(timeout=0.1)
    release_computation.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert computation_calls.qsize() == 1
    assert computed_results.get_nowait().value == 1
    assert computed_results.get_nowait().value == 1


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


def test_supported_rows_execute_unique_recipes_and_capture_recipe_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    computed_recipe_temperatures_K = Queue()
    submitted_work_count = Queue()

    class ImmediateProcessPoolExecutor(ThreadPoolExecutor):
        def submit(
            self,
            function,
            supported_recipe,
            pressure_Pa,
            molecule_count,
            dynamics,
            numerics,
            random_seed,
        ):
            submitted_work_count.put(1)
            return super().submit(
                function,
                supported_recipe,
                pressure_Pa,
                molecule_count,
                dynamics,
                numerics,
                random_seed,
            )

    def fake_load_physical_library(physical_library_root: Path):
        return SimpleNamespace(species_records={"EC": {}, "Li+": {}, "PF6-": {}})

    def fake_compute_first_principles_conductivity(
        recipe,
        temperature_K: float,
        pressure_Pa: float,
        molecule_count: int,
        dynamics,
        numerics,
        random_seed: int,
    ):
        assert recipe.salts == {"LiPF6": 1.0}
        assert pressure_Pa == 101325.0
        assert molecule_count == 8
        assert dynamics is dynamics_settings
        assert numerics is numerical_settings
        assert isinstance(random_seed, int)
        computed_recipe_temperatures_K.put(temperature_K)
        if temperature_K == 310.0:
            raise RuntimeError("row model failed")
        return SimpleNamespace(
            conductivity_S_m=1.2,
            effective_sample_size=480.0,
            maximum_split_rhat=1.004,
            conductivity_mcse_S_m=0.003,
        )

    monkeypatch.setattr(
        property_db_validation, "load_physical_library", fake_load_physical_library
    )
    monkeypatch.setattr(
        property_db_validation,
        "compute_first_principles_conductivity",
        fake_compute_first_principles_conductivity,
    )
    monkeypatch.setattr(
        property_db_validation,
        "ProcessPoolExecutor",
        ImmediateProcessPoolExecutor,
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
                **supported_recipe,
                "temperature_K": 310.0,
            },
            "properties": {"conductivity_mS_cm": 8.0},
        },
        {
            "recipe": {
                "solvents": {"DMC": 1.0},
                "salts": {"LiPF6": 1.0},
                "additives": {},
            },
            "properties": {"conductivity_mS_cm": 8.0},
        },
    ]

    observed_progress = []
    dynamics_settings = SimpleNamespace(name="dynamics")
    numerical_settings = SimpleNamespace(name="numerics")
    summary = validate_property_db_supported_conductivity_rows(
        property_db_entries=entries,
        physical_library_root=tmp_path,
        pressure_Pa=101325.0,
        molecule_count=8,
        dynamics=dynamics_settings,
        numerics=numerical_settings,
        random_seed=17,
        worker_count=2,
        progress_callback=observed_progress.append,
    )

    assert [row.entry_index for row in summary.rows] == [0, 1, 2]
    assert [row.error_mS_cm for row in summary.rows] == [2.0, 1.0, 3.0]
    assert [progress.entry_index for progress in summary.progress] == [0, 1, 2, 3, 4, 5]
    assert summary.evaluated_entry_count == 3
    assert summary.failed_entry_count == 1
    assert summary.skipped_entry_count == 2
    assert summary.classification_counts == {
        "evaluated": 3,
        "evaluation_failure": 1,
        "missing_conductivity": 1,
        "unsupported_species": 1,
    }
    assert summary.progress[4].detail == "RuntimeError: row model failed"
    assert summary.progress[0].detail == "primitive_residual_mS_cm=2"
    assert summary.rows[0].operator_effective_sample_size == 480.0
    assert summary.rows[0].maximum_split_rhat == 1.004
    assert summary.rows[0].conductivity_mcse_mS_cm == 0.03
    assert summary.minimum_operator_effective_sample_size == 480.0
    assert summary.maximum_split_rhat == 1.004
    assert summary.maximum_conductivity_mcse_mS_cm == 0.03
    assert submitted_work_count.qsize() == 2
    assert sorted(
        (
            computed_recipe_temperatures_K.get_nowait(),
            computed_recipe_temperatures_K.get_nowait(),
        )
    ) == [298.15, 310.0]
    assert {
        (progress.entry_index, progress.classification)
        for progress in observed_progress
    } == {
        (progress.entry_index, progress.classification)
        for progress in summary.progress
    }


def test_single_worker_validation_is_synchronous_and_reuses_canonical_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evaluated_temperatures_K = Queue()

    class ForbiddenProcessPoolExecutor:
        def __init__(self, max_workers: int) -> None:
            raise AssertionError(
                f"serial validation constructed an executor with {max_workers} workers"
            )

    def fake_load_physical_library(physical_library_root: Path):
        assert physical_library_root == tmp_path
        return SimpleNamespace(species_records={"EC": {}, "Li+": {}, "PF6-": {}})

    def fake_compute_first_principles_conductivity(
        recipe,
        temperature_K: float,
        pressure_Pa: float,
        molecule_count: int,
        dynamics,
        numerics,
        random_seed: int,
    ):
        assert recipe.salts == {"LiPF6": 1.0}
        assert pressure_Pa == 101325.0
        assert molecule_count == 64
        assert dynamics is dynamics_settings
        assert numerics is numerical_settings
        assert isinstance(random_seed, int)
        evaluated_temperatures_K.put(temperature_K)
        return SimpleNamespace(
            conductivity_S_m=1.2,
            effective_sample_size=510.0,
            maximum_split_rhat=1.003,
            conductivity_mcse_S_m=0.002,
        )

    def ignore_progress(progress) -> None:
        assert progress.entry_index >= 0

    monkeypatch.setattr(
        property_db_validation, "load_physical_library", fake_load_physical_library
    )
    monkeypatch.setattr(
        property_db_validation,
        "compute_first_principles_conductivity",
        fake_compute_first_principles_conductivity,
    )
    monkeypatch.setattr(
        property_db_validation,
        "ProcessPoolExecutor",
        ForbiddenProcessPoolExecutor,
    )
    recipe = {
        "solvents": {"EC": 1.0},
        "salts": {"LiPF6": 1.0},
        "additives": {},
    }
    entries = [
        {"recipe": recipe, "properties": {"conductivity_mS_cm": 10.0}},
        {"recipe": recipe, "properties": {"conductivity_mS_cm": 11.0}},
    ]
    dynamics_settings = SimpleNamespace(name="dynamics")
    numerical_settings = SimpleNamespace(name="numerics")

    summary = validate_property_db_supported_conductivity_rows(
        property_db_entries=entries,
        physical_library_root=tmp_path,
        pressure_Pa=101325.0,
        molecule_count=64,
        dynamics=dynamics_settings,
        numerics=numerical_settings,
        random_seed=17,
        worker_count=1,
        progress_callback=ignore_progress,
    )

    assert evaluated_temperatures_K.get_nowait() == 298.15
    assert evaluated_temperatures_K.empty()
    assert [row.predicted_conductivity_mS_cm for row in summary.rows] == [12.0, 12.0]
    assert summary.evaluated_entry_count == 2
    assert summary.failed_entry_count == 0
    assert summary.rows[0].operator_effective_sample_size == 510.0
    assert summary.rows[0].maximum_split_rhat == 1.003
    assert summary.rows[0].conductivity_mcse_mS_cm == 0.02
