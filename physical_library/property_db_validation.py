"""Reusable conductivity validation against supported property-DB rows."""

from __future__ import annotations

import hashlib
import math
from operator import attrgetter
import tempfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from constants import FRACTION_TO_PERCENT, T_REF_K
from conductivity.physical_library.generator_construction import (
    NumericalOptions,
    compute_conductivity_from_recipe,
)
from conductivity.physical_library.library_io import load_physical_library

MISSING_CONDUCTIVITY_KEY = "missing_conductivity"
EVALUATED_CLASSIFICATION = "evaluated"
MISSING_CONDUCTIVITY_CLASSIFICATION = "missing_conductivity"
UNSUPPORTED_SPECIES_CLASSIFICATION = "unsupported_species"
EVALUATION_FAILURE_CLASSIFICATION = "evaluation_failure"
PROPERTY_VALIDATION_OWNER_CLASSIFICATIONS = frozenset(
    {
        "population_operator_missing",
        "state_mobility_operator_missing",
        "memory_dirichlet_operator_missing",
        "transition_moment_operator_missing",
        "dissociation_operator_missing",
        "state_resolved_born_missing",
        UNSUPPORTED_SPECIES_CLASSIFICATION,
    }
)


@dataclass(frozen=True)
class PropertyDbConductivityValidationRow:
    entry_index: int
    solvents_vv: dict[str, float]
    salts_mol_l: dict[str, float]
    additives_weight_fraction: dict[str, float]
    measured_conductivity_mS_cm: float
    predicted_conductivity_mS_cm: float
    error_mS_cm: float
    absolute_error_mS_cm: float
    percent_error: float
    readiness_status: str
    scalar_label: str


@dataclass(frozen=True)
class PropertyDbConductivityValidationProgress:
    entry_index: int
    classification: str
    detail: str


@dataclass(frozen=True)
class _SupportedValidationRequest:
    entry_index: int
    measured_conductivity_mS_cm: float


@dataclass(frozen=True)
class _CanonicalRecipeWork:
    recipe_digest: str
    supported_recipe: _SupportedRecipe
    requests: tuple[_SupportedValidationRequest, ...]


@dataclass(frozen=True)
class _RecipePrediction:
    predicted_conductivity_mS_cm: float
    readiness_status: str
    scalar_label: str


@dataclass(frozen=True)
class _EvaluatedCanonicalRecipeOutcome:
    prediction: _RecipePrediction


@dataclass(frozen=True)
class _ClassifiedCanonicalRecipeOutcome:
    classification: str
    detail: str


@dataclass(frozen=True)
class PropertyDbConductivityValidationSummary:
    total_entry_count: int
    evaluated_entry_count: int
    skipped_entry_count: int
    skip_counts_by_unsupported_species: dict[str, int]
    failed_entry_count: int
    classification_counts: dict[str, int]
    progress: tuple[PropertyDbConductivityValidationProgress, ...]
    rows: tuple[PropertyDbConductivityValidationRow, ...]
    mean_error_mS_cm: float
    mean_absolute_error_mS_cm: float
    root_mean_square_error_mS_cm: float
    mean_absolute_percent_error: float
    max_absolute_error_mS_cm: float


@dataclass(frozen=True)
class PropertyDbPrimitiveStateOwnerRow:
    state_index: int
    state_label: str
    concentration_mol_m3: float
    self_current_trace_mol_m_s: float
    charged_center_D_Li_m2_s: float
    charged_center_D_anion_m2_s: float
    charged_center_D_Li_anion_m2_s: float
    charged_center_D_Q_m2_s: float
    resistance_stokes_trace_kg_s: float
    resistance_free_volume_trace_kg_s: float
    resistance_charge_cloud_trace_kg_s: float
    resistance_atmosphere_trace_kg_s: float
    resistance_total_trace_kg_s: float


@dataclass(frozen=True)
class PropertyDbPrimitiveTransitionOwnerRow:
    edge_index: int
    family: str
    from_state_index: int
    to_state_index: int
    capacity_flux_K_ij_mol_m3_s: float
    forward_rate_Q_ij_s_inv: float
    reverse_rate_Q_ji_s_inv: float
    first_moment_vector_m: np.ndarray
    first_moment_norm_m: float
    second_moment_trace_m2: float
    K_trace_M_mol_m5_s: float
    reaction_coordinate: str
    coordinate_span: float
    projected_diffusivity_min: float
    projected_diffusivity_max: float
    barrier_over_RT: float
    endpoint_displacement_length_m: float


@dataclass(frozen=True)
class PropertyDbPrimitiveOwnerDiagnostic:
    validation_row: PropertyDbConductivityValidationRow
    pair_concentration_totals_mol_m3: dict[str, float]
    top_self_current_states: tuple[PropertyDbPrimitiveStateOwnerRow, ...]
    top_concentration_states: tuple[PropertyDbPrimitiveStateOwnerRow, ...]
    top_resistance_states: tuple[PropertyDbPrimitiveStateOwnerRow, ...]
    top_transition_moment_edges: tuple[PropertyDbPrimitiveTransitionOwnerRow, ...]
    trace_direct_total: float
    trace_finite_state_memory_correction: float
    trace_continuous_mori_correction: float
    trace_projected_diffusivity: float
    direct_primitive_ledger: dict[str, float]
    state_drift_ledger: tuple[dict, ...]
    state_drift_component_ledger: tuple[dict, ...]
    mori_mode_ledger: tuple[dict, ...]


def validate_property_db_supported_conductivity_rows(
    property_db_entries,
    physical_library_root: Path,
    numerical_options: NumericalOptions,
    worker_count: int,
    progress_callback: Callable[[PropertyDbConductivityValidationProgress], None],
) -> PropertyDbConductivityValidationSummary:
    """Run current physical-library predictions for DB rows with supported species."""

    if worker_count <= 0:
        raise ValueError("property DB validation worker_count must be positive")

    physical_library_records = load_physical_library(physical_library_root)
    supported_species_names = frozenset(physical_library_records.species_records)
    validation_rows: list[PropertyDbConductivityValidationRow] = []
    progress_rows: list[PropertyDbConductivityValidationProgress] = []
    skip_counts_by_unsupported_species: Counter[str] = Counter()

    with tempfile.TemporaryDirectory(
        prefix="conductivity_property_db_validation_"
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        supported_work = []
        for entry_index, property_db_entry in enumerate(property_db_entries):
            if "recipe" not in property_db_entry:
                raise KeyError(f"property DB entry {entry_index} missing recipe")
            if "properties" not in property_db_entry:
                raise KeyError(f"property DB entry {entry_index} missing properties")
            recipe_record = _mapping_value(
                property_db_entry["recipe"],
                f"property DB entry {entry_index} recipe",
            )
            properties_record = _mapping_value(
                property_db_entry["properties"],
                f"property DB entry {entry_index} properties",
            )
            if "conductivity_mS_cm" not in properties_record:
                skip_counts_by_unsupported_species[MISSING_CONDUCTIVITY_KEY] += 1
                _record_validation_progress(
                    progress_rows,
                    PropertyDbConductivityValidationProgress(
                        entry_index, MISSING_CONDUCTIVITY_CLASSIFICATION, ""
                    ),
                    progress_callback,
                )
                continue

            supported_recipe = _supported_recipe_from_property_db_entry(
                entry_index=entry_index,
                recipe_record=recipe_record,
                supported_species_names=supported_species_names,
            )
            if isinstance(supported_recipe, _UnsupportedRecipe):
                skip_counts_by_unsupported_species[
                    supported_recipe.unsupported_species_key
                ] += 1
                _record_validation_progress(
                    progress_rows,
                    PropertyDbConductivityValidationProgress(
                        entry_index,
                        UNSUPPORTED_SPECIES_CLASSIFICATION,
                        supported_recipe.unsupported_species_key,
                    ),
                    progress_callback,
                )
                continue
            measured_conductivity_mS_cm = _finite_float(
                properties_record["conductivity_mS_cm"],
                f"property DB entry {entry_index} conductivity_mS_cm",
            )
            supported_work.append(
                (entry_index, supported_recipe, measured_conductivity_mS_cm)
            )

        canonical_recipe_work = _canonical_recipe_work(supported_work)
        recipe_paths = {
            work.recipe_digest: temporary_directory
            / f"property_db_recipe_{work.recipe_digest}.yaml"
            for work in canonical_recipe_work
        }
        for work in canonical_recipe_work:
            _write_recipe_yaml(
                recipe_paths[work.recipe_digest],
                work.supported_recipe,
            )

        active_worker_count = min(len(canonical_recipe_work), worker_count)
        with ProcessPoolExecutor(max_workers=active_worker_count) as executor:
            future_to_work = {
                executor.submit(
                    _evaluate_canonical_recipe,
                    recipe_paths[work.recipe_digest],
                    physical_library_root,
                    numerical_options,
                ): work
                for work in canonical_recipe_work
            }
            for future in as_completed(future_to_work):
                work = future_to_work[future]
                worker_error = future.exception()
                if worker_error is not None:
                    _record_canonical_recipe_failure(
                        work=work,
                        detail=f"{type(worker_error).__name__}: {worker_error}",
                        progress_rows=progress_rows,
                        progress_callback=progress_callback,
                    )
                    continue
                outcome = future.result()
                _record_canonical_recipe_outcome(
                    work=work,
                    outcome=outcome,
                    validation_rows=validation_rows,
                    progress_rows=progress_rows,
                    progress_callback=progress_callback,
                )

    validation_row_tuple = tuple(sorted(validation_rows, key=attrgetter("entry_index")))
    progress_tuple = tuple(sorted(progress_rows, key=attrgetter("entry_index")))
    if not validation_row_tuple:
        raise ValueError("property DB validation found no supported conductivity rows")

    classification_counts = Counter(
        progress.classification for progress in progress_tuple
    )

    errors_mS_cm = np.asarray(
        [validation_row.error_mS_cm for validation_row in validation_row_tuple],
        dtype=float,
    )
    absolute_errors_mS_cm = np.abs(errors_mS_cm)
    absolute_percent_errors = np.asarray(
        [abs(validation_row.percent_error) for validation_row in validation_row_tuple],
        dtype=float,
    )
    return PropertyDbConductivityValidationSummary(
        total_entry_count=len(property_db_entries),
        evaluated_entry_count=len(validation_row_tuple),
        skipped_entry_count=(
            classification_counts[MISSING_CONDUCTIVITY_CLASSIFICATION]
            + classification_counts[UNSUPPORTED_SPECIES_CLASSIFICATION]
        ),
        skip_counts_by_unsupported_species=dict(
            sorted(skip_counts_by_unsupported_species.items())
        ),
        failed_entry_count=classification_counts[EVALUATION_FAILURE_CLASSIFICATION],
        classification_counts=dict(sorted(classification_counts.items())),
        progress=progress_tuple,
        rows=validation_row_tuple,
        mean_error_mS_cm=float(np.mean(errors_mS_cm)),
        mean_absolute_error_mS_cm=float(np.mean(absolute_errors_mS_cm)),
        root_mean_square_error_mS_cm=float(
            np.sqrt(np.mean(errors_mS_cm * errors_mS_cm))
        ),
        mean_absolute_percent_error=float(np.mean(absolute_percent_errors)),
        max_absolute_error_mS_cm=float(np.max(absolute_errors_mS_cm)),
    )


def _record_validation_progress(
    progress_rows: list[PropertyDbConductivityValidationProgress],
    progress: PropertyDbConductivityValidationProgress,
    progress_callback: Callable[[PropertyDbConductivityValidationProgress], None],
) -> None:
    progress_rows.append(progress)
    progress_callback(progress)


def _canonical_recipe_work(
    supported_work: list[tuple[int, _SupportedRecipe, float]],
) -> tuple[_CanonicalRecipeWork, ...]:
    grouped_requests: dict[
        tuple,
        tuple[_SupportedRecipe, list[_SupportedValidationRequest]],
    ] = {}
    for entry_index, supported_recipe, measured_conductivity_mS_cm in supported_work:
        recipe_key = _canonical_recipe_key(supported_recipe)
        if recipe_key not in grouped_requests:
            grouped_requests[recipe_key] = (supported_recipe, [])
        grouped_requests[recipe_key][1].append(
            _SupportedValidationRequest(
                entry_index=entry_index,
                measured_conductivity_mS_cm=measured_conductivity_mS_cm,
            )
        )
    sorted_grouped_requests = sorted(
        grouped_requests.items(),
        key=_canonical_recipe_group_sort_key,
    )
    return tuple(
        _CanonicalRecipeWork(
            recipe_digest=_canonical_recipe_digest(recipe_key),
            supported_recipe=supported_recipe,
            requests=tuple(requests),
        )
        for recipe_key, (supported_recipe, requests) in sorted_grouped_requests
    )


def _canonical_recipe_group_sort_key(grouped_recipe_item: tuple) -> tuple:
    return grouped_recipe_item[0]


def _canonical_recipe_key(supported_recipe: _SupportedRecipe) -> tuple:
    return (
        supported_recipe.temperature_K,
        tuple(sorted(supported_recipe.solvents_vv.items())),
        tuple(sorted(supported_recipe.salts_mol_l.items())),
        tuple(sorted(supported_recipe.additives_weight_fraction.items())),
    )


def _canonical_recipe_digest(recipe_key: tuple) -> str:
    canonical_yaml = yaml.safe_dump(recipe_key, sort_keys=True)
    return hashlib.sha256(canonical_yaml.encode("utf-8")).hexdigest()


def _evaluate_canonical_recipe(
    recipe_yaml_path: Path,
    physical_library_root: Path,
    numerical_options: NumericalOptions,
) -> (
    _EvaluatedCanonicalRecipeOutcome | _ClassifiedCanonicalRecipeOutcome
):
    with threadpool_limits(limits=1):
        conductivity_result = compute_conductivity_from_recipe(
            recipe=recipe_yaml_path,
            library_root=physical_library_root,
        )
    classification = _property_validation_classification_from_result(
        conductivity_result=conductivity_result,
    )
    if classification != EVALUATED_CLASSIFICATION:
        reasons = tuple(
            str(reason)
            for reason in conductivity_result.effect_attribution[
                "primitive_prediction_not_complete_reasons"
            ]
        )
        return _ClassifiedCanonicalRecipeOutcome(
            classification=classification,
            detail=",".join(reasons),
        )
    if conductivity_result.sigma_mS_cm is None:
        raise ValueError("canonical recipe produced a missing conductivity prediction")
    predicted_conductivity_mS_cm = float(conductivity_result.sigma_mS_cm)
    if not math.isfinite(predicted_conductivity_mS_cm):
        raise ValueError("canonical recipe produced a non-finite conductivity prediction")
    effect_attribution = conductivity_result.effect_attribution
    return _EvaluatedCanonicalRecipeOutcome(
        prediction=_RecipePrediction(
            predicted_conductivity_mS_cm=predicted_conductivity_mS_cm,
            readiness_status=str(
                effect_attribution["primitive_prediction_readiness_status"]
            ),
            scalar_label=str(
                effect_attribution["primitive_prediction_scalar_label"]
            ),
        )
    )


def _record_canonical_recipe_failure(
    work: _CanonicalRecipeWork,
    detail: str,
    progress_rows: list[PropertyDbConductivityValidationProgress],
    progress_callback: Callable[[PropertyDbConductivityValidationProgress], None],
) -> None:
    for request in work.requests:
        _record_validation_progress(
            progress_rows,
            PropertyDbConductivityValidationProgress(
                entry_index=request.entry_index,
                classification=EVALUATION_FAILURE_CLASSIFICATION,
                detail=detail,
            ),
            progress_callback,
        )


def _record_canonical_recipe_outcome(
    work: _CanonicalRecipeWork,
    outcome: _EvaluatedCanonicalRecipeOutcome | _ClassifiedCanonicalRecipeOutcome,
    validation_rows: list[PropertyDbConductivityValidationRow],
    progress_rows: list[PropertyDbConductivityValidationProgress],
    progress_callback: Callable[[PropertyDbConductivityValidationProgress], None],
) -> None:
    if isinstance(outcome, _ClassifiedCanonicalRecipeOutcome):
        for request in work.requests:
            _record_validation_progress(
                progress_rows,
                PropertyDbConductivityValidationProgress(
                    entry_index=request.entry_index,
                    classification=outcome.classification,
                    detail=outcome.detail,
                ),
                progress_callback,
            )
        return

    for request in work.requests:
        validation_row = _validation_row_from_prediction(
            entry_index=request.entry_index,
            supported_recipe=work.supported_recipe,
            measured_conductivity_mS_cm=request.measured_conductivity_mS_cm,
            prediction=outcome.prediction,
        )
        validation_rows.append(validation_row)
        _record_validation_progress(
            progress_rows,
            PropertyDbConductivityValidationProgress(
                entry_index=request.entry_index,
                classification=EVALUATED_CLASSIFICATION,
                detail=f"primitive_residual_mS_cm={validation_row.error_mS_cm:.12g}",
            ),
            progress_callback,
        )


def _validation_row_from_prediction(
    entry_index: int,
    supported_recipe: _SupportedRecipe,
    measured_conductivity_mS_cm: float,
    prediction: _RecipePrediction,
) -> PropertyDbConductivityValidationRow:
    if measured_conductivity_mS_cm <= 0.0:
        raise ValueError(
            f"property DB entry {entry_index} conductivity_mS_cm must be positive"
        )
    error_mS_cm = (
        prediction.predicted_conductivity_mS_cm - measured_conductivity_mS_cm
    )
    return PropertyDbConductivityValidationRow(
        entry_index=entry_index,
        solvents_vv=dict(supported_recipe.solvents_vv),
        salts_mol_l=dict(supported_recipe.salts_mol_l),
        additives_weight_fraction=dict(supported_recipe.additives_weight_fraction),
        measured_conductivity_mS_cm=measured_conductivity_mS_cm,
        predicted_conductivity_mS_cm=prediction.predicted_conductivity_mS_cm,
        error_mS_cm=error_mS_cm,
        absolute_error_mS_cm=abs(error_mS_cm),
        percent_error=(
            error_mS_cm / measured_conductivity_mS_cm * FRACTION_TO_PERCENT
        ),
        readiness_status=prediction.readiness_status,
        scalar_label=prediction.scalar_label,
    )


def _property_validation_classification_from_result(conductivity_result) -> str:
    """Classify primitive ownership from the production readiness ledger."""

    effect_attribution = conductivity_result.effect_attribution
    readiness_status = str(effect_attribution["primitive_prediction_readiness_status"])
    scalar_label = str(effect_attribution["primitive_prediction_scalar_label"])
    if readiness_status == "complete" and scalar_label == "primitive_prediction":
        return EVALUATED_CLASSIFICATION
    reasons = tuple(
        str(reason)
        for reason in effect_attribution["primitive_prediction_not_complete_reasons"]
    )
    owner_classifications = tuple(
        reason
        for reason in reasons
        if reason in PROPERTY_VALIDATION_OWNER_CLASSIFICATIONS
    )
    if len(owner_classifications) > 1:
        raise ValueError(
            "property DB validation result has multiple primitive owner failures: "
            + ",".join(owner_classifications)
        )
    if owner_classifications:
        return owner_classifications[0]
    raise ValueError(
        "property DB validation requires a complete primitive prediction; "
        f"readiness_status={readiness_status}; scalar_label={scalar_label}; "
        f"reasons={reasons}"
    )


def diagnose_property_db_entry_primitives(
    property_db_entries,
    entry_index: int,
    physical_library_root: Path,
    numerical_options: NumericalOptions,
    top_count: int,
) -> PropertyDbPrimitiveOwnerDiagnostic:
    """Return primitive-owner diagnostics for one supported property-DB row."""

    if entry_index < 0 or entry_index >= len(property_db_entries):
        raise IndexError("entry_index out of range for property DB entries")
    if top_count <= 0:
        raise ValueError("top_count must be positive")

    physical_library_records = load_physical_library(physical_library_root)
    supported_species_names = frozenset(physical_library_records.species_records)
    property_db_entry = property_db_entries[entry_index]
    if "recipe" not in property_db_entry:
        raise KeyError(f"property DB entry {entry_index} missing recipe")
    if "properties" not in property_db_entry:
        raise KeyError(f"property DB entry {entry_index} missing properties")

    recipe_record = _mapping_value(
        property_db_entry["recipe"],
        f"property DB entry {entry_index} recipe",
    )
    properties_record = _mapping_value(
        property_db_entry["properties"],
        f"property DB entry {entry_index} properties",
    )
    if "conductivity_mS_cm" not in properties_record:
        raise KeyError(f"property DB entry {entry_index} missing conductivity_mS_cm")

    supported_recipe = _supported_recipe_from_property_db_entry(
        entry_index=entry_index,
        recipe_record=recipe_record,
        supported_species_names=supported_species_names,
    )
    if isinstance(supported_recipe, _UnsupportedRecipe):
        raise ValueError(
            f"property DB entry {entry_index} contains unsupported species "
            f"{supported_recipe.unsupported_species_key}"
        )

    with tempfile.TemporaryDirectory(
        prefix="conductivity_property_db_primitive_diagnostic_"
    ) as temporary_directory_name:
        recipe_yaml_path = (
            Path(temporary_directory_name) / f"property_db_{entry_index}.yaml"
        )
        _write_recipe_yaml(recipe_yaml_path, supported_recipe)
        conductivity_result = compute_conductivity_from_recipe(
            recipe=recipe_yaml_path,
            library_root=physical_library_root,
        )

    validation_row = _validation_row_from_result(
        entry_index=entry_index,
        supported_recipe=supported_recipe,
        measured_conductivity_mS_cm=_finite_float(
            properties_record["conductivity_mS_cm"],
            f"property DB entry {entry_index} conductivity_mS_cm",
        ),
        conductivity_result=conductivity_result,
    )
    state_owner_rows = _primitive_state_owner_rows(conductivity_result)
    transition_owner_rows = _primitive_transition_owner_rows(conductivity_result)
    return PropertyDbPrimitiveOwnerDiagnostic(
        validation_row=validation_row,
        pair_concentration_totals_mol_m3=_pair_concentration_totals(state_owner_rows),
        top_self_current_states=_top_state_owner_rows(
            state_owner_rows=state_owner_rows,
            field_name="self_current_trace_mol_m_s",
            top_count=top_count,
        ),
        top_concentration_states=_top_state_owner_rows(
            state_owner_rows=state_owner_rows,
            field_name="concentration_mol_m3",
            top_count=top_count,
        ),
        top_resistance_states=_top_state_owner_rows(
            state_owner_rows=state_owner_rows,
            field_name="resistance_total_trace_kg_s",
            top_count=top_count,
        ),
        top_transition_moment_edges=_top_transition_owner_rows(
            transition_owner_rows=transition_owner_rows,
            top_count=top_count,
        ),
        trace_direct_total=float(
            conductivity_result.effect_attribution["trace_direct_total"]
        ),
        trace_finite_state_memory_correction=float(
            conductivity_result.effect_attribution[
                "trace_finite_state_memory_correction"
            ]
        ),
        trace_continuous_mori_correction=float(
            conductivity_result.effect_attribution[
                "trace_continuous_mori_correction"
            ]
        ),
        trace_projected_diffusivity=float(
            conductivity_result.effect_attribution["trace_projected_diffusivity"]
        ),
        direct_primitive_ledger={
            ledger_name: float(conductivity_result.effect_attribution[ledger_name])
            for ledger_name in (
                "B_self_full_trace_mol_m_s",
                "B_self_tangent_trace_mol_m_s",
                "B_transition_trace_mol_m_s",
                "B_overlap_removed_trace_mol_m_s",
                "B_total_trace_mol_m_s",
                "C_Q_contribution_trace_mol_m_s",
            )
        },
        state_drift_ledger=tuple(
            {
                "state_index": state_index,
                "state_label": state_owner_rows[state_index].state_label,
                "exit_rate_s_inv": float(
                    conductivity_result.effect_attribution["state_exit_rates_s_inv"][
                        state_index
                    ]
                ),
                "drift_b_i_m_s": np.asarray(
                    conductivity_result.effect_attribution["state_drift_b_i_m_s"][
                        state_index
                    ],
                    dtype=float,
                ),
                "drift_b_i_norm_m_s": float(
                    conductivity_result.effect_attribution[
                        "state_drift_b_i_norms_m_s"
                    ][state_index]
                ),
            }
            for state_index in range(len(state_owner_rows))
        ),
        state_drift_component_ledger=tuple(
            conductivity_result.effect_attribution["state_drift_components"]
        ),
        mori_mode_ledger=tuple(
            conductivity_result.effect_attribution["mori_mode_ledger"]
        ),
    )


@dataclass(frozen=True)
class _SupportedRecipe:
    temperature_K: float
    solvents_vv: dict[str, float]
    salts_mol_l: dict[str, float]
    additives_weight_fraction: dict[str, float]


@dataclass(frozen=True)
class _UnsupportedRecipe:
    unsupported_species_key: str


@dataclass(frozen=True)
class _LithiumSaltComponents:
    lithium_component_name: str
    anion_component_name: str


@dataclass(frozen=True)
class _UnsupportedSaltFormula:
    salt_formula_name: str


def _supported_recipe_from_property_db_entry(
    entry_index: int,
    recipe_record,
    supported_species_names: frozenset[str],
) -> _SupportedRecipe | _UnsupportedRecipe:
    for required_key in ("solvents", "salts", "additives"):
        if required_key not in recipe_record:
            raise KeyError(f"property DB entry {entry_index} missing {required_key}")

    solvent_volume_fractions = _finite_float_mapping(
        recipe_record["solvents"],
        f"property DB entry {entry_index} solvents",
    )
    salt_formula_molarities = _finite_float_mapping(
        recipe_record["salts"],
        f"property DB entry {entry_index} salts",
    )
    additive_weight_fractions = _finite_float_mapping(
        recipe_record["additives"],
        f"property DB entry {entry_index} additives",
    )

    unsupported_species_names: set[str] = set()
    for solvent_name in solvent_volume_fractions:
        if solvent_name not in supported_species_names:
            unsupported_species_names.add(solvent_name)
    for additive_name in additive_weight_fractions:
        if additive_name not in supported_species_names:
            unsupported_species_names.add(additive_name)

    salt_component_molarities: Counter[str] = Counter()
    for salt_formula_name, salt_molarity_mol_l in salt_formula_molarities.items():
        salt_components = _lithium_salt_components(salt_formula_name)
        if isinstance(salt_components, _UnsupportedSaltFormula):
            unsupported_species_names.add(salt_components.salt_formula_name)
            continue
        lithium_component_name = salt_components.lithium_component_name
        anion_component_name = salt_components.anion_component_name
        salt_is_supported = True
        if lithium_component_name not in supported_species_names:
            salt_is_supported = False
        if anion_component_name not in supported_species_names:
            salt_is_supported = False
        if not salt_is_supported:
            unsupported_species_names.add(salt_formula_name)
            continue
        salt_component_molarities[lithium_component_name] += salt_molarity_mol_l
        salt_component_molarities[anion_component_name] += salt_molarity_mol_l

    if unsupported_species_names:
        return _UnsupportedRecipe(
            unsupported_species_key=",".join(sorted(unsupported_species_names))
        )

    temperature_key_count = int("temperature_K" in recipe_record) + int(
        "T_K" in recipe_record
    )
    if temperature_key_count > 1:
        raise KeyError(
            f"property DB entry {entry_index} has both temperature_K and T_K"
        )
    temperature_K = T_REF_K
    if "temperature_K" in recipe_record:
        temperature_K = _finite_float(
            recipe_record["temperature_K"],
            f"property DB entry {entry_index} temperature_K",
        )
    if "T_K" in recipe_record:
        temperature_K = _finite_float(
            recipe_record["T_K"],
            f"property DB entry {entry_index} T_K",
        )

    return _SupportedRecipe(
        temperature_K=temperature_K,
        solvents_vv=solvent_volume_fractions,
        salts_mol_l=dict(salt_component_molarities),
        additives_weight_fraction=additive_weight_fractions,
    )


def _lithium_salt_components(
    salt_formula_name: str,
) -> _LithiumSaltComponents | _UnsupportedSaltFormula:
    if not salt_formula_name.startswith("Li"):
        return _UnsupportedSaltFormula(salt_formula_name=salt_formula_name)
    anion_formula = salt_formula_name.removeprefix("Li")
    if not anion_formula:
        return _UnsupportedSaltFormula(salt_formula_name=salt_formula_name)
    return _LithiumSaltComponents(
        lithium_component_name="Li+",
        anion_component_name=f"{anion_formula}-",
    )


def _write_recipe_yaml(recipe_yaml_path: Path, supported_recipe: _SupportedRecipe) -> None:
    recipe_record = {
        "temperature_K": supported_recipe.temperature_K,
        "solvents_vv": supported_recipe.solvents_vv,
        "salts_mol_l": supported_recipe.salts_mol_l,
        "additives_weight_fraction": supported_recipe.additives_weight_fraction,
    }
    with recipe_yaml_path.open("w") as recipe_file:
        yaml.safe_dump(recipe_record, recipe_file, sort_keys=True)


def _validation_row_from_result(
    entry_index: int,
    supported_recipe: _SupportedRecipe,
    measured_conductivity_mS_cm: float,
    conductivity_result,
) -> PropertyDbConductivityValidationRow:
    readiness_status = str(
        conductivity_result.effect_attribution[
            "primitive_prediction_readiness_status"
        ]
    )
    scalar_label = str(
        conductivity_result.effect_attribution["primitive_prediction_scalar_label"]
    )
    if readiness_status != "complete" or scalar_label != "primitive_prediction":
        raise ValueError(
            "property DB validation requires a complete primitive prediction; "
            f"readiness_status={readiness_status}; scalar_label={scalar_label}"
        )
    if conductivity_result.sigma_mS_cm is None:
        raise ValueError(
            f"property DB entry {entry_index} produced a missing conductivity prediction"
        )
    predicted_conductivity_mS_cm = float(conductivity_result.sigma_mS_cm)
    if not math.isfinite(predicted_conductivity_mS_cm):
        raise ValueError(
            f"property DB entry {entry_index} produced a non-finite conductivity prediction"
        )
    if measured_conductivity_mS_cm <= 0.0:
        raise ValueError(
            f"property DB entry {entry_index} conductivity_mS_cm must be positive"
        )
    error_mS_cm = predicted_conductivity_mS_cm - measured_conductivity_mS_cm
    return PropertyDbConductivityValidationRow(
        entry_index=entry_index,
        solvents_vv=dict(supported_recipe.solvents_vv),
        salts_mol_l=dict(supported_recipe.salts_mol_l),
        additives_weight_fraction=dict(supported_recipe.additives_weight_fraction),
        measured_conductivity_mS_cm=measured_conductivity_mS_cm,
        predicted_conductivity_mS_cm=predicted_conductivity_mS_cm,
        error_mS_cm=error_mS_cm,
        absolute_error_mS_cm=abs(error_mS_cm),
        percent_error=(
            error_mS_cm / measured_conductivity_mS_cm * FRACTION_TO_PERCENT
        ),
        readiness_status=readiness_status,
        scalar_label=scalar_label,
    )


def _primitive_state_owner_rows(
    conductivity_result,
) -> tuple[PropertyDbPrimitiveStateOwnerRow, ...]:
    labels = tuple(
        str(label) for label in conductivity_result.effect_attribution["state_labels"]
    )
    concentrations = np.asarray(
        conductivity_result.state_concentrations_mol_m3,
        dtype=float,
    )
    self_current_traces = np.trace(
        np.einsum(
            "i,iab->iab",
            concentrations,
            np.asarray(
                conductivity_result.self_current_tensors_D_self_i_m2_s,
                dtype=float,
            ),
        ),
        axis1=1,
        axis2=2,
    )
    field_by_name = _state_owner_field_by_name(conductivity_result)
    _validate_state_owner_field_shapes(
        labels,
        concentrations,
        self_current_traces,
        field_by_name,
    )
    return tuple(
        PropertyDbPrimitiveStateOwnerRow(
            state_index=state_index,
            state_label=labels[state_index],
            concentration_mol_m3=float(concentrations[state_index]),
            self_current_trace_mol_m_s=float(self_current_traces[state_index]),
            charged_center_D_Li_m2_s=float(
                field_by_name["charged_center_D_Li_m2_s"][state_index]
            ),
            charged_center_D_anion_m2_s=float(
                field_by_name["charged_center_D_anion_m2_s"][state_index]
            ),
            charged_center_D_Li_anion_m2_s=float(
                field_by_name["charged_center_D_Li_anion_m2_s"][state_index]
            ),
            charged_center_D_Q_m2_s=float(
                field_by_name["charged_center_D_Q_m2_s"][state_index]
            ),
            resistance_stokes_trace_kg_s=float(
                field_by_name["resistance_stokes_trace_kg_s"][state_index]
            ),
            resistance_free_volume_trace_kg_s=float(
                field_by_name["resistance_free_volume_trace_kg_s"][state_index]
            ),
            resistance_charge_cloud_trace_kg_s=float(
                field_by_name["resistance_charge_cloud_trace_kg_s"][state_index]
            ),
            resistance_atmosphere_trace_kg_s=float(
                field_by_name["resistance_atmosphere_trace_kg_s"][state_index]
            ),
            resistance_total_trace_kg_s=float(
                field_by_name["resistance_total_trace_kg_s"][state_index]
            ),
        )
        for state_index in range(len(labels))
    )


def _state_owner_field_by_name(conductivity_result) -> dict[str, np.ndarray]:
    return {
        "charged_center_D_Li_m2_s": np.asarray(
            conductivity_result.effect_attribution["state_charged_center_D_Li_m2_s"],
            dtype=float,
        ),
        "charged_center_D_anion_m2_s": np.asarray(
            conductivity_result.effect_attribution[
                "state_charged_center_D_anion_m2_s"
            ],
            dtype=float,
        ),
        "charged_center_D_Li_anion_m2_s": np.asarray(
            conductivity_result.effect_attribution[
                "state_charged_center_D_Li_anion_m2_s"
            ],
            dtype=float,
        ),
        "charged_center_D_Q_m2_s": np.asarray(
            conductivity_result.effect_attribution[
                "state_charged_center_D_Q_zDz_m2_s"
            ],
            dtype=float,
        ),
        "resistance_stokes_trace_kg_s": np.asarray(
            conductivity_result.effect_attribution[
                "state_resistance_stokes_traces_kg_s"
            ],
            dtype=float,
        ),
        "resistance_free_volume_trace_kg_s": np.asarray(
            conductivity_result.effect_attribution[
                "state_resistance_free_volume_traces_kg_s"
            ],
            dtype=float,
        ),
        "resistance_charge_cloud_trace_kg_s": np.asarray(
            conductivity_result.effect_attribution[
                "state_resistance_charge_cloud_traces_kg_s"
            ],
            dtype=float,
        ),
        "resistance_atmosphere_trace_kg_s": np.asarray(
            conductivity_result.effect_attribution[
                "state_resistance_atmosphere_traces_kg_s"
            ],
            dtype=float,
        ),
        "resistance_total_trace_kg_s": np.asarray(
            conductivity_result.effect_attribution[
                "state_resistance_total_traces_kg_s"
            ],
            dtype=float,
        ),
    }


def _primitive_transition_owner_rows(
    conductivity_result,
) -> tuple[PropertyDbPrimitiveTransitionOwnerRow, ...]:
    edge_families = tuple(
        str(family)
        for family in conductivity_result.effect_attribution["transition_edge_families"]
    )
    from_state_indices = np.asarray(
        conductivity_result.effect_attribution["transition_edge_from_state_indices"],
        dtype=int,
    )
    to_state_indices = np.asarray(
        conductivity_result.effect_attribution["transition_edge_to_state_indices"],
        dtype=int,
    )
    capacity_fluxes = np.asarray(
        conductivity_result.effect_attribution[
            "transition_edge_capacity_fluxes_K_ij_mol_m3_s"
        ],
        dtype=float,
    )
    forward_rates = np.asarray(
        conductivity_result.effect_attribution[
            "transition_edge_forward_rates_Q_ij_s_inv"
        ],
        dtype=float,
    )
    reverse_rates = np.asarray(
        conductivity_result.effect_attribution[
            "transition_edge_reverse_rates_Q_ji_s_inv"
        ],
        dtype=float,
    )
    first_moment_vectors = np.asarray(
        conductivity_result.effect_attribution[
            "transition_edge_first_moment_vectors_m"
        ],
        dtype=float,
    )
    first_moment_norms = np.asarray(
        conductivity_result.effect_attribution["transition_edge_first_moment_norms_m"],
        dtype=float,
    )
    second_moment_traces = np.asarray(
        conductivity_result.effect_attribution[
            "transition_edge_second_moment_traces_m2"
        ],
        dtype=float,
    )
    direct_trace_contributions = np.asarray(
        conductivity_result.effect_attribution["transition_edge_K_trace_M_mol_m5_s"],
        dtype=float,
    )
    reaction_coordinates = tuple(
        str(coordinate)
        for coordinate in conductivity_result.effect_attribution[
            "transition_edge_reaction_coordinates"
        ]
    )
    coordinate_spans = np.asarray(
        conductivity_result.effect_attribution["transition_edge_coordinate_spans"],
        dtype=float,
    )
    projected_diffusivity_minima = np.asarray(
        conductivity_result.effect_attribution[
            "transition_edge_projected_diffusivity_min"
        ],
        dtype=float,
    )
    projected_diffusivity_maxima = np.asarray(
        conductivity_result.effect_attribution[
            "transition_edge_projected_diffusivity_max"
        ],
        dtype=float,
    )
    barriers_over_RT = np.asarray(
        conductivity_result.effect_attribution["transition_edge_barrier_over_RT"],
        dtype=float,
    )
    endpoint_displacement_lengths = np.asarray(
        conductivity_result.effect_attribution[
            "transition_edge_endpoint_displacement_lengths_m"
        ],
        dtype=float,
    )
    _validate_transition_owner_field_shapes(
        edge_families,
        {
            "from_state_indices": from_state_indices,
            "to_state_indices": to_state_indices,
            "capacity_fluxes": capacity_fluxes,
            "forward_rates": forward_rates,
            "reverse_rates": reverse_rates,
            "first_moment_norms": first_moment_norms,
            "second_moment_traces": second_moment_traces,
            "direct_trace_contributions": direct_trace_contributions,
            "coordinate_spans": coordinate_spans,
            "projected_diffusivity_minima": projected_diffusivity_minima,
            "projected_diffusivity_maxima": projected_diffusivity_maxima,
            "barriers_over_RT": barriers_over_RT,
            "endpoint_displacement_lengths": endpoint_displacement_lengths,
        },
    )
    if first_moment_vectors.shape != (len(edge_families), 3):
        raise ValueError("first moment vector count does not match transition edges")
    if len(reaction_coordinates) != len(edge_families):
        raise ValueError("reaction coordinate count does not match transition edges")
    return tuple(
        PropertyDbPrimitiveTransitionOwnerRow(
            edge_index=edge_index,
            family=edge_families[edge_index],
            from_state_index=int(from_state_indices[edge_index]),
            to_state_index=int(to_state_indices[edge_index]),
            capacity_flux_K_ij_mol_m3_s=float(capacity_fluxes[edge_index]),
            forward_rate_Q_ij_s_inv=float(forward_rates[edge_index]),
            reverse_rate_Q_ji_s_inv=float(reverse_rates[edge_index]),
            first_moment_vector_m=first_moment_vectors[edge_index].copy(),
            first_moment_norm_m=float(first_moment_norms[edge_index]),
            second_moment_trace_m2=float(second_moment_traces[edge_index]),
            K_trace_M_mol_m5_s=float(direct_trace_contributions[edge_index]),
            reaction_coordinate=reaction_coordinates[edge_index],
            coordinate_span=float(coordinate_spans[edge_index]),
            projected_diffusivity_min=float(
                projected_diffusivity_minima[edge_index]
            ),
            projected_diffusivity_max=float(
                projected_diffusivity_maxima[edge_index]
            ),
            barrier_over_RT=float(barriers_over_RT[edge_index]),
            endpoint_displacement_length_m=float(
                endpoint_displacement_lengths[edge_index]
            ),
        )
        for edge_index in range(len(edge_families))
    )


def _validate_state_owner_field_shapes(
    labels: tuple[str, ...],
    concentrations: np.ndarray,
    self_current_traces: np.ndarray,
    field_by_name: dict[str, np.ndarray],
) -> None:
    if concentrations.shape != (len(labels),):
        raise ValueError("state concentration count does not match state labels")
    if self_current_traces.shape != (len(labels),):
        raise ValueError("state self-current count does not match state labels")
    if not np.all(np.isfinite(concentrations)):
        raise ValueError("state concentrations contain non-finite values")
    if not np.all(np.isfinite(self_current_traces)):
        raise ValueError("state self-current traces contain non-finite values")
    for field_name, field_values in field_by_name.items():
        if field_values.shape != (len(labels),):
            raise ValueError(f"{field_name} count does not match state labels")
        if not np.all(np.isfinite(field_values)):
            raise ValueError(f"{field_name} contains non-finite values")


def _validate_transition_owner_field_shapes(
    edge_families: tuple[str, ...],
    field_by_name: dict[str, np.ndarray],
) -> None:
    edge_count = len(edge_families)
    for field_name, field_values in field_by_name.items():
        if field_values.shape != (edge_count,):
            raise ValueError(f"{field_name} count does not match transition edges")
        if not np.all(np.isfinite(field_values)):
            raise ValueError(f"{field_name} contains non-finite values")


def _pair_concentration_totals(
    state_owner_rows: tuple[PropertyDbPrimitiveStateOwnerRow, ...],
) -> dict[str, float]:
    concentration_totals: Counter[str] = Counter()
    for state_owner_row in state_owner_rows:
        concentration_totals[_state_pair_label(state_owner_row.state_label)] += (
            state_owner_row.concentration_mol_m3
        )
    return dict(sorted(concentration_totals.items()))


def _state_pair_label(state_label: str) -> str:
    pair_label = state_label.split("|", maxsplit=1)[0]
    if not pair_label:
        raise ValueError("state label has empty pair field")
    return pair_label


def _top_state_owner_rows(
    state_owner_rows: tuple[PropertyDbPrimitiveStateOwnerRow, ...],
    field_name: str,
    top_count: int,
) -> tuple[PropertyDbPrimitiveStateOwnerRow, ...]:
    ordered_rows = sorted(
        state_owner_rows,
        key=attrgetter(field_name),
        reverse=True,
    )
    return tuple(ordered_rows[:top_count])


def _top_transition_owner_rows(
    transition_owner_rows: tuple[PropertyDbPrimitiveTransitionOwnerRow, ...],
    top_count: int,
) -> tuple[PropertyDbPrimitiveTransitionOwnerRow, ...]:
    ordered_rows = sorted(
        transition_owner_rows,
        key=attrgetter("K_trace_M_mol_m5_s"),
        reverse=True,
    )
    return tuple(ordered_rows[:top_count])


def _finite_float_mapping(raw_mapping, context: str) -> dict[str, float]:
    mapping_value = _mapping_value(raw_mapping, context)
    return {
        str(species_name): _finite_float(value, f"{context}.{species_name}")
        for species_name, value in mapping_value.items()
    }


def _mapping_value(raw_mapping, context: str):
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"{context} must be a mapping")
    return raw_mapping


def _finite_float(value, context: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{context} must be finite")
    return numeric_value
