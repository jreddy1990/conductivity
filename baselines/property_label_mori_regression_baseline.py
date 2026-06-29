"""Property-calibrated latent Mori surrogate for electrolyte conductivity.

This module trains a recipe-to-latent-block surrogate from scalar conductivity
labels. The latent blocks are gauge-fixed because scalar labels identify the
quadratic form, not unique trajectory Mori operators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from constants import S_M_TO_MS_CM, T_REF_K
from conductivity.finite_markov_dataset_audit import (
    RecipeDict,
    _require_entry,
    canonicalize_empirical_recipe,
)
from conductivity.finite_mori_conductivity import (
    ProjectedMoriConductivityInput,
    ProjectedMoriConductivityResult,
    compute_projected_mori_conductivity,
)
from data.species_data import ADDITIVES, SALTS, SOLVENTS
from utils.strict_validation import require_float, require_mapping


LATENT_MORI_GAUGE = "identity_energy_nonnegative_current_power"
LATENT_MORI_MODEL_SOURCE = "property_calibrated_latent_mori_surrogate"
LATENT_MORI_BETA_OVER_VOLUME = 1.0
DEFAULT_LATENT_MORI_RIDGE_PENALTY = 1.0

LATENT_MORI_MODE_NAMES = (
    "free_ion_migration",
    "association_memory",
    "solvent_drag",
    "additive_environment",
    "residual_current_memory",
)
DEFAULT_LATENT_MORI_FOLD_COUNT = len(LATENT_MORI_MODE_NAMES)

LATENT_MORI_FEATURE_NAMES = (
    "temperature_ratio",
    "temperature_inverse_ratio",
    "total_salt_molarity_M",
    "log_total_salt_molarity",
    "total_salt_molarity_squared",
    "salt_mixing_entropy",
    "salt_Lambda_0_weighted",
    "salt_anion_radius_A_weighted",
    "salt_anion_volume_weighted",
    "salt_binding_energy_kJ_mol_weighted",
    "salt_jones_dole_B_weighted",
    "salt_dielectric_decrement_weighted",
    "salt_bjerrum_K_A_ref_weighted",
    "salt_stokes_alpha_weighted",
    "solvent_mixing_entropy",
    "solvent_epsilon_weighted",
    "solvent_log_viscosity_cP_weighted",
    "solvent_donor_number_weighted",
    "solvent_acceptor_number_weighted",
    "solvent_density_g_ml_weighted",
    "solvent_dipole_D_weighted",
    "solvent_coordination_log_weighted",
    "total_additive_weight_fraction",
    "additive_weight_fraction_squared",
    "additive_mixing_entropy",
    "additive_epsilon_weighted",
    "additive_log_viscosity_cP_weighted",
    "additive_donor_number_weighted",
    "additive_density_g_ml_weighted",
    "additive_binding_energy_kJ_mol_weighted",
    "additive_coordination_log_weighted",
    "salt_molarity_x_solvent_log_viscosity",
    "salt_molarity_x_additive_weight",
    "salt_molarity_x_salt_binding",
    "dielectric_over_viscosity_log_ratio",
)


@dataclass(frozen=True)
class LatentMoriFeatureVector:
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    mode_names: tuple[str, ...]
    mode_weight_inputs: tuple[float, ...]
    mode_weights: tuple[float, ...]


@dataclass(frozen=True)
class LatentMoriSurrogateMetrics:
    count: int
    mae_mS_cm: float
    rmse_mS_cm: float
    bias_mS_cm: float
    mape_percent: float
    r2: float
    pearson_r: float


@dataclass(frozen=True)
class LatentMoriSurrogateParameters:
    model_source: str
    gauge: str
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    standardized_coefficients: tuple[float, ...]
    intercept: float
    ridge_penalty: float
    temperature_K: float
    training_row_count: int
    mode_names: tuple[str, ...]
    training_metrics: LatentMoriSurrogateMetrics


@dataclass(frozen=True)
class LatentMoriPrediction:
    sigma_mS_cm: float
    log_sigma_mS_cm: float
    feature_vector: LatentMoriFeatureVector
    mode_contribution_mS_cm: tuple[float, ...]
    mori_input: ProjectedMoriConductivityInput
    mori_result: ProjectedMoriConductivityResult


@dataclass(frozen=True)
class LatentMoriSurrogateRow:
    row_id: int
    empirical_sigma_mS_cm: float
    full_fit_sigma_mS_cm: float
    cross_validated_sigma_mS_cm: float
    full_fit_residual_mS_cm: float
    cross_validated_residual_mS_cm: float
    projected_basis_dimension: int
    mode_contribution_mS_cm: tuple[float, ...]


@dataclass(frozen=True)
class LatentMoriSurrogateFailure:
    row_id: int
    error: str


@dataclass(frozen=True)
class LatentMoriSurrogateAuditResult:
    labeled_rows: int
    evaluated_rows: int
    failed_rows: int
    full_fit_metrics: LatentMoriSurrogateMetrics
    cross_validated_metrics: LatentMoriSurrogateMetrics
    parameters: LatentMoriSurrogateParameters
    rows: tuple[LatentMoriSurrogateRow, ...]
    failures: tuple[LatentMoriSurrogateFailure, ...]


@dataclass(frozen=True)
class _LatentMoriTrainingRecord:
    row_id: int
    recipe: RecipeDict
    empirical_sigma_mS_cm: float
    feature_vector: LatentMoriFeatureVector


def fit_latent_mori_surrogate(
    records: Sequence[_LatentMoriTrainingRecord],
    temperature_K: float,
    ridge_penalty: float,
) -> LatentMoriSurrogateParameters:
    """Fit a deterministic ridge model for log conductivity."""

    _assert_positive_finite(temperature_K, "temperature_K")
    _assert_nonnegative_finite(ridge_penalty, "ridge_penalty")
    if len(records) < 2:
        raise ValueError("latent Mori surrogate requires at least two training records")

    feature_matrix = _feature_matrix_from_records(records)
    target_log_sigma = np.asarray(
        [
            math.log(_positive_sigma(record.empirical_sigma_mS_cm, record.row_id))
            for record in records
        ],
        dtype=float,
    )
    feature_means = np.mean(feature_matrix, axis=0)
    feature_scales = _feature_scales(feature_matrix)
    standardized_feature_matrix = (feature_matrix - feature_means) / feature_scales
    design_matrix = np.column_stack(
        (np.ones(standardized_feature_matrix.shape[0]), standardized_feature_matrix)
    )
    penalty_matrix = np.eye(design_matrix.shape[1], dtype=float) * ridge_penalty
    penalty_matrix[0, 0] = 0.0
    normal_matrix = design_matrix.T @ design_matrix + penalty_matrix
    normal_rhs = design_matrix.T @ target_log_sigma
    fitted_coefficients = np.linalg.solve(normal_matrix, normal_rhs)

    provisional_parameters = LatentMoriSurrogateParameters(
        model_source=LATENT_MORI_MODEL_SOURCE,
        gauge=LATENT_MORI_GAUGE,
        feature_names=LATENT_MORI_FEATURE_NAMES,
        feature_means=tuple(float(value) for value in feature_means),
        feature_scales=tuple(float(value) for value in feature_scales),
        standardized_coefficients=tuple(
            float(value) for value in fitted_coefficients[1:]
        ),
        intercept=float(fitted_coefficients[0]),
        ridge_penalty=float(ridge_penalty),
        temperature_K=float(temperature_K),
        training_row_count=len(records),
        mode_names=LATENT_MORI_MODE_NAMES,
        training_metrics=LatentMoriSurrogateMetrics(
            count=0,
            mae_mS_cm=0.0,
            rmse_mS_cm=0.0,
            bias_mS_cm=0.0,
            mape_percent=0.0,
            r2=0.0,
            pearson_r=0.0,
        ),
    )
    training_predictions = [
        predict_latent_mori_conductivity(record.recipe, provisional_parameters).sigma_mS_cm
        for record in records
    ]
    training_targets = [record.empirical_sigma_mS_cm for record in records]
    training_metrics = _metrics_from_values(training_targets, training_predictions)
    return LatentMoriSurrogateParameters(
        model_source=LATENT_MORI_MODEL_SOURCE,
        gauge=LATENT_MORI_GAUGE,
        feature_names=LATENT_MORI_FEATURE_NAMES,
        feature_means=tuple(float(value) for value in feature_means),
        feature_scales=tuple(float(value) for value in feature_scales),
        standardized_coefficients=tuple(
            float(value) for value in fitted_coefficients[1:]
        ),
        intercept=float(fitted_coefficients[0]),
        ridge_penalty=float(ridge_penalty),
        temperature_K=float(temperature_K),
        training_row_count=len(records),
        mode_names=LATENT_MORI_MODE_NAMES,
        training_metrics=training_metrics,
    )


def predict_latent_mori_conductivity(
    recipe: RecipeDict,
    parameters: LatentMoriSurrogateParameters,
) -> LatentMoriPrediction:
    """Predict conductivity through gauge-fixed latent Mori blocks."""

    _validate_latent_parameters(parameters)
    feature_vector = featurize_latent_mori_recipe(recipe, parameters.temperature_K)
    feature_values = np.asarray(feature_vector.values, dtype=float)
    feature_means = np.asarray(parameters.feature_means, dtype=float)
    feature_scales = np.asarray(parameters.feature_scales, dtype=float)
    coefficients = np.asarray(parameters.standardized_coefficients, dtype=float)
    standardized_features = (feature_values - feature_means) / feature_scales
    log_sigma_mS_cm = float(parameters.intercept + standardized_features @ coefficients)
    if not math.isfinite(log_sigma_mS_cm):
        raise ValueError("latent Mori log conductivity prediction is non-finite")
    sigma_mS_cm = float(math.exp(log_sigma_mS_cm))
    _assert_positive_finite(sigma_mS_cm, "latent Mori sigma_mS_cm")
    mode_contribution_mS_cm = tuple(
        sigma_mS_cm * mode_weight for mode_weight in feature_vector.mode_weights
    )
    mori_input = _build_gauge_fixed_mori_input(mode_contribution_mS_cm)
    mori_result = compute_projected_mori_conductivity(mori_input)
    if abs(mori_result.sigma_mS_cm - sigma_mS_cm) > 1.0e-9 * max(1.0, sigma_mS_cm):
        raise ValueError(
            "latent Mori gauge construction does not reproduce predicted conductivity"
        )
    return LatentMoriPrediction(
        sigma_mS_cm=sigma_mS_cm,
        log_sigma_mS_cm=log_sigma_mS_cm,
        feature_vector=feature_vector,
        mode_contribution_mS_cm=mode_contribution_mS_cm,
        mori_input=mori_input,
        mori_result=mori_result,
    )


def featurize_latent_mori_recipe(
    recipe: RecipeDict,
    temperature_K: float,
) -> LatentMoriFeatureVector:
    """Build registry-derived numeric descriptors for a canonical recipe."""

    _assert_positive_finite(temperature_K, "temperature_K")
    solvents = _composition_section(recipe, "solvents", "recipe")
    salts = _composition_section(recipe, "salts", "recipe")
    additives = _composition_section(recipe, "additives", "recipe")
    total_salt_molarity_M = _section_total(salts, "recipe.salts")
    if total_salt_molarity_M <= 0.0:
        raise ValueError("latent Mori recipe requires positive total salt molarity")
    total_additive_weight_fraction = _section_total(additives, "recipe.additives")

    solvent_log_viscosity = _weighted_log_property(
        solvents,
        SOLVENTS,
        "viscosity_cP",
        "recipe.solvents",
    )
    solvent_epsilon = _weighted_property(
        solvents,
        SOLVENTS,
        "epsilon_r",
        "recipe.solvents",
    )
    salt_binding_energy = _weighted_property(
        salts,
        SALTS,
        "ion_pair_binding_kj_mol",
        "recipe.salts",
    )
    feature_values = {
        "temperature_ratio": temperature_K / T_REF_K,
        "temperature_inverse_ratio": T_REF_K / temperature_K,
        "total_salt_molarity_M": total_salt_molarity_M,
        "log_total_salt_molarity": math.log(total_salt_molarity_M),
        "total_salt_molarity_squared": total_salt_molarity_M * total_salt_molarity_M,
        "salt_mixing_entropy": _mixing_entropy(salts, "recipe.salts"),
        "salt_Lambda_0_weighted": _weighted_property(
            salts,
            SALTS,
            "Lambda_0",
            "recipe.salts",
        ),
        "salt_anion_radius_A_weighted": _weighted_property(
            salts,
            SALTS,
            "anion_radius",
            "recipe.salts",
        ),
        "salt_anion_volume_weighted": _weighted_property(
            salts,
            SALTS,
            "anion_volume",
            "recipe.salts",
        ),
        "salt_binding_energy_kJ_mol_weighted": salt_binding_energy,
        "salt_jones_dole_B_weighted": _weighted_property(
            salts,
            SALTS,
            "jones_dole_B",
            "recipe.salts",
        ),
        "salt_dielectric_decrement_weighted": _weighted_property(
            salts,
            SALTS,
            "dielectric_decrement_frac_per_M",
            "recipe.salts",
        ),
        "salt_bjerrum_K_A_ref_weighted": _weighted_property(
            salts,
            SALTS,
            "bjerrum_K_A_ref",
            "recipe.salts",
        ),
        "salt_stokes_alpha_weighted": _weighted_property(
            salts,
            SALTS,
            "stokes_einstein_alpha_anion",
            "recipe.salts",
        ),
        "solvent_mixing_entropy": _mixing_entropy(solvents, "recipe.solvents"),
        "solvent_epsilon_weighted": solvent_epsilon,
        "solvent_log_viscosity_cP_weighted": solvent_log_viscosity,
        "solvent_donor_number_weighted": _weighted_property(
            solvents,
            SOLVENTS,
            "donor_number",
            "recipe.solvents",
        ),
        "solvent_acceptor_number_weighted": _weighted_property(
            solvents,
            SOLVENTS,
            "acceptor_number",
            "recipe.solvents",
        ),
        "solvent_density_g_ml_weighted": _weighted_property(
            solvents,
            SOLVENTS,
            "density_g_ml",
            "recipe.solvents",
        ),
        "solvent_dipole_D_weighted": _weighted_property(
            solvents,
            SOLVENTS,
            "dipole_moment_D",
            "recipe.solvents",
        ),
        "solvent_coordination_log_weighted": _weighted_log_property(
            solvents,
            SOLVENTS,
            "coordination_affinity_M_inv",
            "recipe.solvents",
        ),
        "total_additive_weight_fraction": total_additive_weight_fraction,
        "additive_weight_fraction_squared": (
            total_additive_weight_fraction * total_additive_weight_fraction
        ),
        "additive_mixing_entropy": _mixing_entropy(additives, "recipe.additives"),
        "additive_epsilon_weighted": _weighted_property_allowing_empty(
            additives,
            ADDITIVES,
            "epsilon_r",
            "recipe.additives",
        ),
        "additive_log_viscosity_cP_weighted": _weighted_log_property_allowing_empty(
            additives,
            ADDITIVES,
            "viscosity_cP",
            "recipe.additives",
        ),
        "additive_donor_number_weighted": _weighted_property_allowing_empty(
            additives,
            ADDITIVES,
            "donor_number",
            "recipe.additives",
        ),
        "additive_density_g_ml_weighted": _weighted_property_allowing_empty(
            additives,
            ADDITIVES,
            "density_g_ml",
            "recipe.additives",
        ),
        "additive_binding_energy_kJ_mol_weighted": _weighted_property_allowing_empty(
            additives,
            ADDITIVES,
            "li_binding_energy_kJ_mol",
            "recipe.additives",
        ),
        "additive_coordination_log_weighted": _weighted_log_property_allowing_empty(
            additives,
            ADDITIVES,
            "coordination_affinity_M_inv",
            "recipe.additives",
        ),
        "salt_molarity_x_solvent_log_viscosity": (
            total_salt_molarity_M * solvent_log_viscosity
        ),
        "salt_molarity_x_additive_weight": (
            total_salt_molarity_M * total_additive_weight_fraction
        ),
        "salt_molarity_x_salt_binding": total_salt_molarity_M * salt_binding_energy,
        "dielectric_over_viscosity_log_ratio": math.log(solvent_epsilon)
        - solvent_log_viscosity,
    }
    feature_tuple = tuple(
        _finite_feature_value(feature_values[feature_name], feature_name)
        for feature_name in LATENT_MORI_FEATURE_NAMES
    )
    mode_weight_inputs = _latent_mode_weight_inputs(feature_values)
    mode_weights = _normalized_positive_weights(
        mode_weight_inputs,
        LATENT_MORI_MODE_NAMES,
    )
    return LatentMoriFeatureVector(
        feature_names=LATENT_MORI_FEATURE_NAMES,
        values=feature_tuple,
        mode_names=LATENT_MORI_MODE_NAMES,
        mode_weight_inputs=mode_weight_inputs,
        mode_weights=mode_weights,
    )


def fit_latent_mori_surrogate_from_property_db(
    entries,
    temperature_K: float,
    ridge_penalty: float,
) -> LatentMoriSurrogateParameters:
    """Fit reusable latent Mori parameters from labeled property-DB rows."""

    records, failures, labeled_rows = _training_records_from_property_db(
        entries,
        temperature_K,
    )
    if failures:
        failure_summary = "; ".join(
            f"row {failure.row_id}: {failure.error}" for failure in failures
        )
        raise ValueError(f"latent Mori training had failed rows: {failure_summary}")
    if len(records) != labeled_rows:
        raise ValueError("latent Mori training did not evaluate every labeled row")
    return fit_latent_mori_surrogate(records, temperature_K, ridge_penalty)


def audit_latent_mori_surrogate_against_property_db(
    entries,
    temperature_K: float,
    ridge_penalty: float,
    fold_count: int,
) -> LatentMoriSurrogateAuditResult:
    """Fit and audit the latent Mori surrogate against conductivity labels."""

    records, failures, labeled_rows = _training_records_from_property_db(
        entries,
        temperature_K,
    )
    if len(records) < 2:
        raise ValueError("latent Mori audit needs at least two evaluated labeled rows")
    parameters = fit_latent_mori_surrogate(records, temperature_K, ridge_penalty)
    full_fit_predictions = [
        predict_latent_mori_conductivity(record.recipe, parameters)
        for record in records
    ]
    cross_validated_sigmas = _cross_validated_predictions(
        records,
        temperature_K,
        ridge_penalty,
        fold_count,
    )
    rows = []
    for record, full_fit_prediction, cross_validated_sigma_mS_cm in zip(
        records,
        full_fit_predictions,
        cross_validated_sigmas,
    ):
        rows.append(
            LatentMoriSurrogateRow(
                row_id=record.row_id,
                empirical_sigma_mS_cm=record.empirical_sigma_mS_cm,
                full_fit_sigma_mS_cm=full_fit_prediction.sigma_mS_cm,
                cross_validated_sigma_mS_cm=cross_validated_sigma_mS_cm,
                full_fit_residual_mS_cm=(
                    full_fit_prediction.sigma_mS_cm - record.empirical_sigma_mS_cm
                ),
                cross_validated_residual_mS_cm=(
                    cross_validated_sigma_mS_cm - record.empirical_sigma_mS_cm
                ),
                projected_basis_dimension=len(full_fit_prediction.mori_result.energy_eigenvalues),
                mode_contribution_mS_cm=full_fit_prediction.mode_contribution_mS_cm,
            )
        )
    empirical_values = [record.empirical_sigma_mS_cm for record in records]
    full_fit_values = [prediction.sigma_mS_cm for prediction in full_fit_predictions]
    cross_validated_metrics = _metrics_from_values(
        empirical_values,
        cross_validated_sigmas,
    )
    return LatentMoriSurrogateAuditResult(
        labeled_rows=labeled_rows,
        evaluated_rows=len(records),
        failed_rows=len(failures),
        full_fit_metrics=_metrics_from_values(empirical_values, full_fit_values),
        cross_validated_metrics=cross_validated_metrics,
        parameters=parameters,
        rows=tuple(rows),
        failures=tuple(failures),
    )


def _training_records_from_property_db(
    entries,
    temperature_K: float,
) -> tuple[
    tuple[_LatentMoriTrainingRecord, ...],
    tuple[LatentMoriSurrogateFailure, ...],
    int,
]:
    records: list[_LatentMoriTrainingRecord] = []
    failures: list[LatentMoriSurrogateFailure] = []
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
            _positive_sigma(empirical_sigma_mS_cm, row_id)
            canonicalization = canonicalize_empirical_recipe(entry_sections["recipe"])
            feature_vector = featurize_latent_mori_recipe(
                canonicalization.recipe,
                temperature_K,
            )
            records.append(
                _LatentMoriTrainingRecord(
                    row_id=row_id,
                    recipe=canonicalization.recipe,
                    empirical_sigma_mS_cm=empirical_sigma_mS_cm,
                    feature_vector=feature_vector,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                LatentMoriSurrogateFailure(
                    row_id=row_id,
                    error=str(exc),
                )
            )
    return tuple(records), tuple(failures), labeled_rows


def _cross_validated_predictions(
    records: Sequence[_LatentMoriTrainingRecord],
    temperature_K: float,
    ridge_penalty: float,
    fold_count: int,
) -> tuple[float, ...]:
    if not isinstance(fold_count, int):
        raise TypeError("fold_count must be an integer")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if fold_count > len(records):
        raise ValueError("fold_count cannot exceed evaluated row count")

    predictions_by_position: list[float | None] = [None for _ in records]
    for fold_index in range(fold_count):
        training_records = tuple(
            record
            for record_index, record in enumerate(records)
            if record_index % fold_count != fold_index
        )
        validation_positions = tuple(
            record_index
            for record_index in range(len(records))
            if record_index % fold_count == fold_index
        )
        if len(training_records) < 2:
            raise ValueError("each latent Mori fold requires at least two training rows")
        fold_parameters = fit_latent_mori_surrogate(
            training_records,
            temperature_K,
            ridge_penalty,
        )
        for validation_position in validation_positions:
            validation_record = records[validation_position]
            validation_prediction = predict_latent_mori_conductivity(
                validation_record.recipe,
                fold_parameters,
            )
            predictions_by_position[validation_position] = validation_prediction.sigma_mS_cm

    if any(prediction is None for prediction in predictions_by_position):
        raise ValueError("cross validation did not produce every row prediction")
    return tuple(float(prediction) for prediction in predictions_by_position)


def _feature_matrix_from_records(
    records: Sequence[_LatentMoriTrainingRecord],
) -> np.ndarray:
    feature_rows = [record.feature_vector.values for record in records]
    feature_matrix = np.asarray(feature_rows, dtype=float)
    if feature_matrix.ndim != 2:
        raise ValueError("latent Mori feature matrix must be two-dimensional")
    if feature_matrix.shape[1] != len(LATENT_MORI_FEATURE_NAMES):
        raise ValueError("latent Mori feature matrix has unexpected feature count")
    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("latent Mori feature matrix contains non-finite values")
    return feature_matrix


def _feature_scales(feature_matrix: np.ndarray) -> np.ndarray:
    raw_scales = np.std(feature_matrix, axis=0)
    scale_floor = math.sqrt(float(np.finfo(float).eps))
    return np.asarray(
        [
            float(scale) if float(scale) > scale_floor else 1.0
            for scale in raw_scales
        ],
        dtype=float,
    )


def _build_gauge_fixed_mori_input(
    mode_contribution_mS_cm: Sequence[float],
) -> ProjectedMoriConductivityInput:
    mode_contributions = np.asarray(mode_contribution_mS_cm, dtype=float)
    if mode_contributions.shape != (len(LATENT_MORI_MODE_NAMES),):
        raise ValueError("mode_contribution_mS_cm has unexpected length")
    if not np.all(np.isfinite(mode_contributions)):
        raise ValueError("mode_contribution_mS_cm contains non-finite values")
    if np.any(mode_contributions < 0.0):
        raise ValueError("mode_contribution_mS_cm must be nonnegative")
    current_power_by_mode = mode_contributions / S_M_TO_MS_CM
    current_coupling_by_mode = np.sqrt(current_power_by_mode)
    current_coupling_matrix = np.vstack(
        (
            current_coupling_by_mode,
            current_coupling_by_mode,
            current_coupling_by_mode,
        )
    )
    mode_count = len(LATENT_MORI_MODE_NAMES)
    return ProjectedMoriConductivityInput(
        direct_energy_matrix=np.zeros((mode_count, mode_count), dtype=float),
        memory_self_energy_matrix=np.eye(mode_count, dtype=float),
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=LATENT_MORI_BETA_OVER_VOLUME,
    )


def _latent_mode_weight_inputs(
    feature_values: Mapping[str, float],
) -> tuple[float, ...]:
    total_salt_molarity_M = feature_values["total_salt_molarity_M"]
    salt_lambda = feature_values["salt_Lambda_0_weighted"]
    salt_binding = abs(feature_values["salt_binding_energy_kJ_mol_weighted"])
    solvent_log_viscosity = feature_values["solvent_log_viscosity_cP_weighted"]
    total_additive_weight_fraction = feature_values["total_additive_weight_fraction"]
    additive_donor = feature_values["additive_donor_number_weighted"]
    solvent_entropy = feature_values["solvent_mixing_entropy"]
    salt_additive_interaction = feature_values["salt_molarity_x_additive_weight"]
    return (
        total_salt_molarity_M * salt_lambda,
        total_salt_molarity_M * total_salt_molarity_M * (1.0 + salt_binding),
        math.exp(-solvent_log_viscosity),
        total_additive_weight_fraction * (1.0 + additive_donor),
        1.0 + solvent_entropy + salt_additive_interaction,
    )


def _normalized_positive_weights(
    positive_inputs: Sequence[float],
    mode_names: Sequence[str],
) -> tuple[float, ...]:
    if len(positive_inputs) != len(mode_names):
        raise ValueError("mode weight input count must match mode name count")
    validated_inputs = []
    for mode_name, positive_input in zip(mode_names, positive_inputs):
        if not math.isfinite(positive_input) or positive_input < 0.0:
            raise ValueError(
                f"mode weight input {mode_name} must be nonnegative and finite"
            )
        validated_inputs.append(float(positive_input))
    input_sum = math.fsum(validated_inputs)
    if input_sum <= 0.0:
        raise ValueError("at least one latent Mori mode weight input must be positive")
    return tuple(positive_input / input_sum for positive_input in validated_inputs)


def _composition_section(
    recipe: RecipeDict,
    section_name: str,
    context: str,
) -> Mapping[str, float]:
    raw_section = require_mapping(recipe, section_name, context)
    parsed_section: dict[str, float] = {}
    for species_name, raw_value in raw_section.items():
        if not isinstance(species_name, str) or species_name == "":
            raise ValueError(f"{context}.{section_name} species names must be non-empty strings")
        parsed_value = float(raw_value)
        if not math.isfinite(parsed_value) or parsed_value < 0.0:
            raise ValueError(
                f"{context}.{section_name}.{species_name} must be nonnegative and finite"
            )
        parsed_section[species_name] = parsed_value
    if section_name == "solvents" and math.fsum(parsed_section.values()) <= 0.0:
        raise ValueError("recipe.solvents must have positive total fraction")
    return parsed_section


def _section_total(
    section: Mapping[str, float],
    context: str,
) -> float:
    section_total = math.fsum(float(value) for value in section.values())
    if not math.isfinite(section_total) or section_total < 0.0:
        raise ValueError(f"{context} total must be nonnegative and finite")
    return section_total


def _weighted_property(
    section: Mapping[str, float],
    species_table,
    property_name: str,
    context: str,
) -> float:
    section_total = _section_total(section, context)
    if section_total <= 0.0:
        raise ValueError(f"{context} total must be positive")
    weighted_sum = 0.0
    for species_name, raw_weight in section.items():
        species_properties = _species_properties(species_table, species_name, context)
        property_value = require_float(
            species_properties,
            property_name,
            f"{context}.{species_name}",
        )
        weighted_sum += raw_weight * property_value
    return float(weighted_sum / section_total)


def _weighted_property_allowing_empty(
    section: Mapping[str, float],
    species_table,
    property_name: str,
    context: str,
) -> float:
    section_total = _section_total(section, context)
    if section_total == 0.0:
        return 0.0
    return _weighted_property(section, species_table, property_name, context)


def _weighted_log_property(
    section: Mapping[str, float],
    species_table,
    property_name: str,
    context: str,
) -> float:
    section_total = _section_total(section, context)
    if section_total <= 0.0:
        raise ValueError(f"{context} total must be positive")
    weighted_sum = 0.0
    for species_name, raw_weight in section.items():
        species_properties = _species_properties(species_table, species_name, context)
        property_value = require_float(
            species_properties,
            property_name,
            f"{context}.{species_name}",
        )
        if property_value <= 0.0:
            raise ValueError(f"{context}.{species_name}.{property_name} must be positive")
        weighted_sum += raw_weight * math.log(property_value)
    return float(weighted_sum / section_total)


def _weighted_log_property_allowing_empty(
    section: Mapping[str, float],
    species_table,
    property_name: str,
    context: str,
) -> float:
    section_total = _section_total(section, context)
    if section_total == 0.0:
        return 0.0
    return _weighted_log_property(section, species_table, property_name, context)


def _mixing_entropy(
    section: Mapping[str, float],
    context: str,
) -> float:
    section_total = _section_total(section, context)
    if section_total == 0.0:
        return 0.0
    entropy_terms = []
    for species_name, raw_value in section.items():
        if raw_value == 0.0:
            continue
        mole_like_fraction = raw_value / section_total
        if mole_like_fraction <= 0.0 or mole_like_fraction > 1.0:
            raise ValueError(f"{context}.{species_name} normalized fraction is invalid")
        entropy_terms.append(-mole_like_fraction * math.log(mole_like_fraction))
    return float(math.fsum(entropy_terms))


def _species_properties(
    species_table,
    species_name: str,
    context: str,
):
    if species_name not in species_table:
        raise ValueError(f"{context}.{species_name} is missing from the species registry")
    return species_table[species_name]


def _metrics_from_values(
    empirical_sigma_values_mS_cm: Sequence[float],
    predicted_sigma_values_mS_cm: Sequence[float],
) -> LatentMoriSurrogateMetrics:
    if len(empirical_sigma_values_mS_cm) != len(predicted_sigma_values_mS_cm):
        raise ValueError("empirical and predicted vectors must have the same length")
    if len(empirical_sigma_values_mS_cm) < 2:
        raise ValueError("latent Mori metrics require at least two values")
    empirical_values = np.asarray(empirical_sigma_values_mS_cm, dtype=float)
    predicted_values = np.asarray(predicted_sigma_values_mS_cm, dtype=float)
    if not np.all(np.isfinite(empirical_values)):
        raise ValueError("empirical conductivity vector contains non-finite values")
    if not np.all(np.isfinite(predicted_values)):
        raise ValueError("predicted conductivity vector contains non-finite values")
    if np.any(empirical_values <= 0.0):
        raise ValueError("empirical conductivity values must be positive")
    residuals = predicted_values - empirical_values
    total_sum_squares = float(
        np.sum((empirical_values - float(np.mean(empirical_values))) ** 2)
    )
    if total_sum_squares <= 0.0:
        raise ValueError("empirical conductivity values have zero variance")
    residual_sum_squares = float(np.sum(residuals * residuals))
    pearson_r = float(np.corrcoef(empirical_values, predicted_values)[0, 1])
    return LatentMoriSurrogateMetrics(
        count=len(empirical_sigma_values_mS_cm),
        mae_mS_cm=float(np.mean(np.abs(residuals))),
        rmse_mS_cm=float(math.sqrt(float(np.mean(residuals * residuals)))),
        bias_mS_cm=float(np.mean(residuals)),
        mape_percent=float(np.mean(np.abs(residuals / empirical_values)) * 100.0),
        r2=float(1.0 - residual_sum_squares / total_sum_squares),
        pearson_r=pearson_r,
    )


def _validate_latent_parameters(
    parameters: LatentMoriSurrogateParameters,
) -> None:
    if parameters.model_source != LATENT_MORI_MODEL_SOURCE:
        raise ValueError("latent Mori parameters have unexpected model_source")
    if parameters.gauge != LATENT_MORI_GAUGE:
        raise ValueError("latent Mori parameters have unexpected gauge")
    if parameters.feature_names != LATENT_MORI_FEATURE_NAMES:
        raise ValueError("latent Mori parameters have unexpected feature_names")
    if parameters.mode_names != LATENT_MORI_MODE_NAMES:
        raise ValueError("latent Mori parameters have unexpected mode_names")
    _assert_positive_finite(parameters.temperature_K, "parameters.temperature_K")
    _assert_nonnegative_finite(parameters.ridge_penalty, "parameters.ridge_penalty")
    if parameters.training_row_count < 2:
        raise ValueError("latent Mori parameters must record at least two training rows")
    expected_feature_count = len(LATENT_MORI_FEATURE_NAMES)
    if len(parameters.feature_means) != expected_feature_count:
        raise ValueError("latent Mori parameter feature mean count is invalid")
    if len(parameters.feature_scales) != expected_feature_count:
        raise ValueError("latent Mori parameter feature scale count is invalid")
    if len(parameters.standardized_coefficients) != expected_feature_count:
        raise ValueError("latent Mori parameter coefficient count is invalid")
    _assert_finite_sequence(parameters.feature_means, "parameters.feature_means")
    _assert_positive_sequence(parameters.feature_scales, "parameters.feature_scales")
    _assert_finite_sequence(
        parameters.standardized_coefficients,
        "parameters.standardized_coefficients",
    )
    _assert_finite(parameters.intercept, "parameters.intercept")


def _positive_sigma(
    sigma_mS_cm: float,
    row_id: int,
) -> float:
    if not math.isfinite(sigma_mS_cm) or sigma_mS_cm <= 0.0:
        raise ValueError(f"DATA[{row_id}].properties.conductivity_mS_cm must be positive")
    return float(sigma_mS_cm)


def _finite_feature_value(
    value: float,
    feature_name: str,
) -> float:
    if not math.isfinite(value):
        raise ValueError(f"latent Mori feature {feature_name} is non-finite")
    return float(value)


def _assert_positive_finite(
    value: float,
    context: str,
) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")


def _assert_nonnegative_finite(
    value: float,
    context: str,
) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{context} must be nonnegative and finite")


def _assert_finite(
    value: float,
    context: str,
) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")


def _assert_finite_sequence(
    values: Sequence[float],
    context: str,
) -> None:
    for value in values:
        _assert_finite(value, context)


def _assert_positive_sequence(
    values: Sequence[float],
    context: str,
) -> None:
    for value in values:
        _assert_positive_finite(value, context)
