"""Fit descriptor-neutral conductivity primitive parameters."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize

from conductivity.molecular_primitive_parameters import (
    CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES,
    ConductivityPrimitiveParameterSet,
    conductivity_primitive_parameter_log_values_for_names,
    conductivity_primitive_parameters_from_mapping,
    conductivity_primitive_parameters_to_mapping,
    conductivity_primitive_parameters_with_log_updates,
    validate_conductivity_primitive_parameters,
)

SPECIATION_FIT_PARAMETER_NAMES = (
    "coulomb_scale",
    "desolvation_scale",
    "coordination_scale",
    "steric_free_energy_scale",
    "cluster_entropy_penalty_scale",
    "association_crowding_stabilization_scale",
    "association_crowding_ionic_strength_exponent",
    "association_crowding_charge_density_exponent",
    "activity_debye_scale",
    "activity_size_scale",
    "activity_hard_sphere_scale",
    "cluster_activity_scale",
    "pair_logK_offset",
    "solvent_separated_pair_logK_offset",
    "contact_pair_logK_offset",
    "positive_charged_triplet_logK_offset",
    "negative_charged_triplet_logK_offset",
    "neutral_cluster_logK_offset",
    "higher_charged_cluster_logK_offset",
    "cluster_order_logK_slope",
    "cluster_charge_magnitude_logK_slope",
    "cluster_hydrodynamic_radius_scale",
    "hydrodynamic_radius_scale_cluster",
)


@dataclass(frozen=True)
class PrimitiveFitLogBound:
    parameter_name: str
    lower_log_value: float
    upper_log_value: float


@dataclass(frozen=True)
class PrimitiveFitOptions:
    huber_delta_mS_cm: float
    log_regularization_weight: float
    residual_tail_loss_weight: float
    residual_tail_count: int
    latin_hypercube_sample_count: int
    coordinate_search_rounds: int
    initial_coordinate_step_log: float
    coordinate_step_shrinkage: float
    minimum_coordinate_step_log: float
    powell_max_iterations: int
    powell_max_function_evaluations: int
    powell_xtol_log: float
    powell_ftol_objective: float
    random_seed: int
    maximum_failed_rows: int
    maximum_mass_balance_residual: float
    maximum_row_sum_residual: float
    maximum_stationary_residual: float
    maximum_detailed_balance_residual: float
    maximum_event_reversal_residual: float
    maximum_zero_charge_sigma_mS_cm: float
    candidate_output_path: str
    promotion_maximum_mae_mS_cm: float
    promotion_maximum_abs_bias_mS_cm: float
    promotion_maximum_worst_abs_residual_mS_cm: float
    promotion_require_mae_improvement: bool


@dataclass(frozen=True)
class PrimitiveFitDatasetEvaluation:
    empirical_sigmas_mS_cm: tuple[float, ...]
    predicted_sigmas_mS_cm: tuple[float, ...]
    failed_rows: int
    maximum_mass_balance_residual: float
    maximum_row_sum_residual: float
    maximum_stationary_residual: float
    maximum_detailed_balance_residual: float
    maximum_event_reversal_residual: float
    zero_charge_sigma_mS_cm: float
    higher_viscosity_lowers_dilute_conductivity: bool
    higher_packing_lowers_local_mobility: bool
    consumed_parameter_fields: tuple[str, ...]


class ConductivityPrimitiveParameterEvaluator(Protocol):
    def evaluate(
        self,
        primitive_parameters: ConductivityPrimitiveParameterSet,
    ) -> PrimitiveFitDatasetEvaluation:
        ...


@dataclass(frozen=True)
class PrimitiveFitCandidateResult:
    primitive_parameters: ConductivityPrimitiveParameterSet
    log_values: tuple[float, ...]
    objective_value: float
    mean_huber_loss_mS_cm: float
    tail_huber_loss_mS_cm: float
    log_regularization_loss: float
    mae_mS_cm: float
    bias_mS_cm: float
    pearson_r: float
    worst_abs_residual_mS_cm: float
    failed_rows: int
    rejected: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PrimitiveParameterFitResult:
    best_candidate: PrimitiveFitCandidateResult
    promotion_candidate: PrimitiveFitCandidateResult
    candidate_count: int
    accepted_candidate_count: int
    all_candidates: tuple[PrimitiveFitCandidateResult, ...]


@dataclass(frozen=True)
class PrimitivePromotionMetrics:
    mae_mS_cm: float
    bias_mS_cm: float
    pearson_r: float
    worst_abs_residual_mS_cm: float
    failed_rows: int
    maximum_mass_balance_residual: float
    maximum_row_sum_residual: float
    maximum_stationary_residual: float
    maximum_detailed_balance_residual: float
    maximum_event_reversal_residual: float
    zero_charge_sigma_mS_cm: float
    higher_viscosity_lowers_dilute_conductivity: bool
    higher_packing_lowers_local_mobility: bool


def fit_conductivity_primitive_parameters(
    initial_parameters: ConductivityPrimitiveParameterSet,
    regularization_reference_parameters: ConductivityPrimitiveParameterSet,
    log_bounds: tuple[PrimitiveFitLogBound, ...],
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitiveParameterFitResult:
    validate_conductivity_primitive_parameters(initial_parameters)
    validate_conductivity_primitive_parameters(regularization_reference_parameters)
    _validate_fit_options(options)
    ordered_bounds = _ordered_log_bounds(log_bounds)
    fitted_parameter_names = _ordered_bound_parameter_names(ordered_bounds)
    initial_log_values = _bounded_initial_log_values(
        conductivity_primitive_parameter_log_values_for_names(
            initial_parameters,
            fitted_parameter_names,
        ),
        ordered_bounds,
    )
    regularization_reference_log_values = (
        conductivity_primitive_parameter_log_values_for_names(
            regularization_reference_parameters,
            fitted_parameter_names,
        )
    )

    candidate_results: list[PrimitiveFitCandidateResult] = []
    candidate_results.append(
        evaluate_primitive_parameter_candidate(
            initial_log_values,
            initial_parameters,
            regularization_reference_log_values,
            ordered_bounds,
            evaluator,
            options,
        )
    )
    random_number_generator = random.Random(options.random_seed)
    for sample_log_values in _latin_hypercube_log_values(
        ordered_bounds,
        options.latin_hypercube_sample_count,
        random_number_generator,
    ):
        candidate_results.append(
            evaluate_primitive_parameter_candidate(
                sample_log_values,
                initial_parameters,
                regularization_reference_log_values,
                ordered_bounds,
                evaluator,
                options,
            )
        )

    current_best = _best_accepted_candidate(candidate_results)
    coordinate_step_log = options.initial_coordinate_step_log
    for search_round_index in range(options.coordinate_search_rounds):
        if coordinate_step_log < options.minimum_coordinate_step_log:
            break
        improved_this_round = False
        for parameter_index, log_bound in enumerate(ordered_bounds):
            for step_sign in (-1.0, 1.0):
                trial_log_values = _coordinate_trial_log_values(
                    current_best.log_values,
                    parameter_index,
                    step_sign * coordinate_step_log,
                    log_bound,
                )
                trial_result = evaluate_primitive_parameter_candidate(
                    trial_log_values,
                    initial_parameters,
                    regularization_reference_log_values,
                    ordered_bounds,
                    evaluator,
                    options,
                )
                candidate_results.append(trial_result)
                if _candidate_is_better(trial_result, current_best):
                    current_best = trial_result
                    improved_this_round = True
        if not improved_this_round:
            coordinate_step_log *= options.coordinate_step_shrinkage
        if search_round_index == options.coordinate_search_rounds - 1:
            break

    current_best = _run_powell_local_polish(
        current_best,
        candidate_results,
        initial_parameters,
        regularization_reference_log_values,
        ordered_bounds,
        evaluator,
        options,
    )

    accepted_candidate_count = sum(
        1 for candidate_result in candidate_results
        if not candidate_result.rejected
    )
    promotion_candidate = select_primitive_parameter_promotion_candidate(
        candidate_results,
        candidate_results[0],
        options,
    )
    return PrimitiveParameterFitResult(
        best_candidate=current_best,
        promotion_candidate=promotion_candidate,
        candidate_count=len(candidate_results),
        accepted_candidate_count=accepted_candidate_count,
        all_candidates=tuple(candidate_results),
    )


def evaluate_primitive_parameter_candidate(
    log_values: tuple[float, ...],
    base_parameters: ConductivityPrimitiveParameterSet,
    regularization_reference_log_values: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveFitLogBound, ...],
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitiveFitCandidateResult:
    _validate_fit_options(options)
    validate_conductivity_primitive_parameters(base_parameters)
    bounded_log_values = _bounded_initial_log_values(log_values, ordered_bounds)
    if len(regularization_reference_log_values) != len(ordered_bounds):
        raise ValueError(
            "regularization_reference_log_values length must match log bound count"
        )
    primitive_parameters = conductivity_primitive_parameters_with_log_updates(
        base_parameters,
        _ordered_bound_parameter_names(ordered_bounds),
        bounded_log_values,
    )
    try:
        evaluation = evaluator.evaluate(primitive_parameters)
    except (
        FloatingPointError,
        OverflowError,
        ValueError,
        np.linalg.LinAlgError,
    ) as evaluation_error:
        return _failed_candidate_result(
            primitive_parameters,
            bounded_log_values,
            regularization_reference_log_values,
            options,
            type(evaluation_error).__name__,
        )
    empirical_sigmas = _validated_sigma_tuple(
        evaluation.empirical_sigmas_mS_cm,
        "empirical_sigmas_mS_cm",
    )
    predicted_sigmas = _validated_sigma_tuple(
        evaluation.predicted_sigmas_mS_cm,
        "predicted_sigmas_mS_cm",
    )
    if len(empirical_sigmas) != len(predicted_sigmas):
        raise ValueError("empirical and predicted sigma tuples must have equal length")
    residuals = tuple(
        predicted_sigma_mS_cm - empirical_sigma_mS_cm
        for empirical_sigma_mS_cm, predicted_sigma_mS_cm in zip(
            empirical_sigmas,
            predicted_sigmas,
        )
    )
    rejection_reasons = list(_candidate_rejection_reasons(
        evaluation,
        predicted_sigmas,
        ordered_bounds,
        options,
    ))
    mae_mS_cm = _mean_absolute_residual(residuals)
    bias_mS_cm = float(math.fsum(residuals) / len(residuals))
    try:
        pearson_r = _pearson_correlation(empirical_sigmas, predicted_sigmas)
    except ValueError:
        pearson_r = 0.0
        rejection_reasons.append("undefined_pearson")
    worst_abs_residual_mS_cm = max(abs(residual) for residual in residuals)
    mean_huber_loss_mS_cm = float(
        math.fsum(
            _smooth_l1_loss_mS_cm(residual, options.huber_delta_mS_cm)
            for residual in residuals
        )
        / len(residuals)
    )
    tail_huber_loss_mS_cm = _tail_huber_loss_mS_cm(
        residuals,
        options.huber_delta_mS_cm,
        options.residual_tail_count,
    )
    log_regularization_loss = _log_regularization_loss(
        bounded_log_values,
        regularization_reference_log_values,
        options.log_regularization_weight,
    )
    rejected = bool(rejection_reasons)
    objective_value = (
        math.inf
        if rejected
        else (
            mean_huber_loss_mS_cm
            + options.residual_tail_loss_weight * tail_huber_loss_mS_cm
            + log_regularization_loss
        )
    )
    return PrimitiveFitCandidateResult(
        primitive_parameters=primitive_parameters,
        log_values=bounded_log_values,
        objective_value=objective_value,
        mean_huber_loss_mS_cm=mean_huber_loss_mS_cm,
        tail_huber_loss_mS_cm=tail_huber_loss_mS_cm,
        log_regularization_loss=log_regularization_loss,
        mae_mS_cm=mae_mS_cm,
        bias_mS_cm=bias_mS_cm,
        pearson_r=pearson_r,
        worst_abs_residual_mS_cm=worst_abs_residual_mS_cm,
        failed_rows=int(evaluation.failed_rows),
        rejected=rejected,
        rejection_reasons=tuple(rejection_reasons),
    )


def _failed_candidate_result(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    bounded_log_values: tuple[float, ...],
    regularization_reference_log_values: tuple[float, ...],
    options: PrimitiveFitOptions,
    error_type_name: str,
) -> PrimitiveFitCandidateResult:
    log_regularization_loss = _log_regularization_loss(
        bounded_log_values,
        regularization_reference_log_values,
        options.log_regularization_weight,
    )
    return PrimitiveFitCandidateResult(
        primitive_parameters=primitive_parameters,
        log_values=bounded_log_values,
        objective_value=math.inf,
        mean_huber_loss_mS_cm=math.inf,
        tail_huber_loss_mS_cm=math.inf,
        log_regularization_loss=log_regularization_loss,
        mae_mS_cm=math.inf,
        bias_mS_cm=math.inf,
        pearson_r=0.0,
        worst_abs_residual_mS_cm=math.inf,
        failed_rows=options.maximum_failed_rows + 1,
        rejected=True,
        rejection_reasons=(f"evaluation_error:{error_type_name}",),
    )


def _run_powell_local_polish(
    current_best: PrimitiveFitCandidateResult,
    candidate_results: list[PrimitiveFitCandidateResult],
    initial_parameters: ConductivityPrimitiveParameterSet,
    regularization_reference_log_values: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveFitLogBound, ...],
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitiveFitCandidateResult:
    if (
        options.powell_max_iterations == 0
        or options.powell_max_function_evaluations == 0
    ):
        return current_best

    evaluation_cache: dict[tuple[float, ...], PrimitiveFitCandidateResult] = {}

    def _objective_for_log_values(log_value_array: np.ndarray) -> float:
        log_values = tuple(float(log_value) for log_value in log_value_array)
        bounded_log_values = _bounded_initial_log_values(log_values, ordered_bounds)
        if bounded_log_values not in evaluation_cache:
            candidate_result = evaluate_primitive_parameter_candidate(
                bounded_log_values,
                initial_parameters,
                regularization_reference_log_values,
                ordered_bounds,
                evaluator,
                options,
            )
            evaluation_cache[bounded_log_values] = candidate_result
            candidate_results.append(candidate_result)
        return evaluation_cache[bounded_log_values].objective_value

    lower_log_values = np.asarray(
        [log_bound.lower_log_value for log_bound in ordered_bounds],
        dtype=float,
    )
    upper_log_values = np.asarray(
        [log_bound.upper_log_value for log_bound in ordered_bounds],
        dtype=float,
    )
    minimize(
        _objective_for_log_values,
        np.asarray(current_best.log_values, dtype=float),
        method="Powell",
        bounds=Bounds(lower_log_values, upper_log_values),
        options={
            "maxiter": options.powell_max_iterations,
            "maxfev": options.powell_max_function_evaluations,
            "xtol": options.powell_xtol_log,
            "ftol": options.powell_ftol_objective,
            "disp": False,
        },
    )
    for candidate_result in evaluation_cache.values():
        if _candidate_is_better(candidate_result, current_best):
            current_best = candidate_result
    return current_best


def _ordered_log_bounds(
    log_bounds: tuple[PrimitiveFitLogBound, ...],
) -> tuple[PrimitiveFitLogBound, ...]:
    if not log_bounds:
        raise ValueError("log_bounds must contain at least one primitive parameter")
    bound_by_parameter_name: dict[str, PrimitiveFitLogBound] = {}
    for log_bound in log_bounds:
        if log_bound.parameter_name in bound_by_parameter_name:
            raise ValueError(f"duplicate log bound for {log_bound.parameter_name}")
        _validate_log_bound(log_bound)
        bound_by_parameter_name[log_bound.parameter_name] = log_bound
    ordered_bounds: list[PrimitiveFitLogBound] = []
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        if parameter_name in bound_by_parameter_name:
            ordered_bounds.append(bound_by_parameter_name[parameter_name])
    return tuple(ordered_bounds)


def _ordered_bound_parameter_names(
    ordered_bounds: tuple[PrimitiveFitLogBound, ...],
) -> tuple[str, ...]:
    if not ordered_bounds:
        raise ValueError("ordered_bounds must contain at least one primitive parameter")
    return tuple(log_bound.parameter_name for log_bound in ordered_bounds)


def _log_bounds_for_parameter_names(
    log_bounds: tuple[PrimitiveFitLogBound, ...],
    parameter_names: tuple[str, ...],
) -> tuple[PrimitiveFitLogBound, ...]:
    requested_parameter_names = set(parameter_names)
    selected_bounds = tuple(
        log_bound for log_bound in log_bounds
        if log_bound.parameter_name in requested_parameter_names
    )
    if not selected_bounds:
        raise ValueError("selected log bounds must contain at least one parameter")
    return selected_bounds


def _validate_log_bound(log_bound: PrimitiveFitLogBound) -> None:
    if log_bound.parameter_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        raise ValueError(f"unknown primitive parameter {log_bound.parameter_name}")
    lower_log_value = _finite_float(
        log_bound.lower_log_value,
        f"{log_bound.parameter_name}.lower_log_value",
    )
    upper_log_value = _finite_float(
        log_bound.upper_log_value,
        f"{log_bound.parameter_name}.upper_log_value",
    )
    if lower_log_value >= upper_log_value:
        raise ValueError(f"{log_bound.parameter_name} lower bound must be below upper")


def _validate_fit_options(options: PrimitiveFitOptions) -> None:
    _positive_float(options.huber_delta_mS_cm, "huber_delta_mS_cm")
    _nonnegative_float(options.log_regularization_weight, "log_regularization_weight")
    _nonnegative_float(
        options.residual_tail_loss_weight,
        "residual_tail_loss_weight",
    )
    _positive_int(options.residual_tail_count, "residual_tail_count")
    _positive_int(
        options.latin_hypercube_sample_count,
        "latin_hypercube_sample_count",
    )
    _nonnegative_int(options.coordinate_search_rounds, "coordinate_search_rounds")
    _positive_float(options.initial_coordinate_step_log, "initial_coordinate_step_log")
    coordinate_step_shrinkage = _positive_float(
        options.coordinate_step_shrinkage,
        "coordinate_step_shrinkage",
    )
    if coordinate_step_shrinkage >= 1.0:
        raise ValueError("coordinate_step_shrinkage must be below 1")
    _positive_float(options.minimum_coordinate_step_log, "minimum_coordinate_step_log")
    _nonnegative_int(options.powell_max_iterations, "powell_max_iterations")
    _nonnegative_int(
        options.powell_max_function_evaluations,
        "powell_max_function_evaluations",
    )
    _positive_float(options.powell_xtol_log, "powell_xtol_log")
    _positive_float(options.powell_ftol_objective, "powell_ftol_objective")
    _nonnegative_int(options.maximum_failed_rows, "maximum_failed_rows")
    _nonnegative_float(
        options.maximum_mass_balance_residual,
        "maximum_mass_balance_residual",
    )
    _nonnegative_float(options.maximum_row_sum_residual, "maximum_row_sum_residual")
    _nonnegative_float(
        options.maximum_stationary_residual,
        "maximum_stationary_residual",
    )
    _nonnegative_float(
        options.maximum_detailed_balance_residual,
        "maximum_detailed_balance_residual",
    )
    _nonnegative_float(
        options.maximum_event_reversal_residual,
        "maximum_event_reversal_residual",
    )
    _nonnegative_float(
        options.maximum_zero_charge_sigma_mS_cm,
        "maximum_zero_charge_sigma_mS_cm",
    )
    _nonempty_string(options.candidate_output_path, "candidate_output_path")
    _positive_float(options.promotion_maximum_mae_mS_cm, "promotion_maximum_mae_mS_cm")
    _nonnegative_float(
        options.promotion_maximum_abs_bias_mS_cm,
        "promotion_maximum_abs_bias_mS_cm",
    )
    _nonnegative_float(
        options.promotion_maximum_worst_abs_residual_mS_cm,
        "promotion_maximum_worst_abs_residual_mS_cm",
    )
    if not isinstance(options.promotion_require_mae_improvement, bool):
        raise TypeError("promotion_require_mae_improvement must be a boolean")


def _bounded_initial_log_values(
    log_values: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveFitLogBound, ...],
) -> tuple[float, ...]:
    if len(log_values) != len(ordered_bounds):
        raise ValueError("log value count must match ordered bound count")
    bounded_values: list[float] = []
    for log_value, log_bound in zip(log_values, ordered_bounds):
        parsed_log_value = _finite_float(log_value, log_bound.parameter_name)
        if parsed_log_value < log_bound.lower_log_value:
            raise ValueError(f"{log_bound.parameter_name} is below its lower log bound")
        if parsed_log_value > log_bound.upper_log_value:
            raise ValueError(f"{log_bound.parameter_name} is above its upper log bound")
        bounded_values.append(parsed_log_value)
    return tuple(bounded_values)


def _latin_hypercube_log_values(
    ordered_bounds: tuple[PrimitiveFitLogBound, ...],
    sample_count: int,
    random_number_generator: random.Random,
) -> tuple[tuple[float, ...], ...]:
    _positive_int(sample_count, "latin_hypercube_sample_count")
    per_parameter_values: list[list[float]] = []
    for log_bound in ordered_bounds:
        parameter_values: list[float] = []
        log_span = log_bound.upper_log_value - log_bound.lower_log_value
        for sample_index in range(sample_count):
            unit_interval_value = (
                sample_index + random_number_generator.random()
            ) / sample_count
            parameter_values.append(
                log_bound.lower_log_value + unit_interval_value * log_span
            )
        random_number_generator.shuffle(parameter_values)
        per_parameter_values.append(parameter_values)
    samples: list[tuple[float, ...]] = []
    for sample_index in range(sample_count):
        samples.append(
            tuple(
                parameter_values[sample_index]
                for parameter_values in per_parameter_values
            )
        )
    return tuple(samples)


def _coordinate_trial_log_values(
    current_log_values: tuple[float, ...],
    parameter_index: int,
    step_log_value: float,
    log_bound: PrimitiveFitLogBound,
) -> tuple[float, ...]:
    trial_values = list(current_log_values)
    trial_value = current_log_values[parameter_index] + step_log_value
    if trial_value < log_bound.lower_log_value:
        trial_value = log_bound.lower_log_value
    if trial_value > log_bound.upper_log_value:
        trial_value = log_bound.upper_log_value
    trial_values[parameter_index] = trial_value
    return tuple(trial_values)


def _candidate_rejection_reasons(
    evaluation: PrimitiveFitDatasetEvaluation,
    predicted_sigmas_mS_cm: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveFitLogBound, ...],
    options: PrimitiveFitOptions,
) -> tuple[str, ...]:
    reasons: list[str] = []
    bounded_parameter_names = _ordered_bound_parameter_names(ordered_bounds)
    unconsumed_parameter_names = tuple(
        parameter_name for parameter_name in bounded_parameter_names
        if parameter_name not in evaluation.consumed_parameter_fields
    )
    if len(unconsumed_parameter_names) == len(bounded_parameter_names):
        reasons.append("no_consumed_primitive_parameters")
    elif unconsumed_parameter_names:
        reasons.append(
            "unconsumed_fit_parameters:"
            + ",".join(unconsumed_parameter_names)
        )
    if evaluation.failed_rows > options.maximum_failed_rows:
        reasons.append("failed_rows")
    if min(predicted_sigmas_mS_cm) < 0.0:
        reasons.append("negative_conductivity")
    _append_threshold_reason(
        reasons,
        evaluation.maximum_mass_balance_residual,
        options.maximum_mass_balance_residual,
        "mass_balance_residual",
    )
    _append_threshold_reason(
        reasons,
        evaluation.maximum_row_sum_residual,
        options.maximum_row_sum_residual,
        "row_sum_residual",
    )
    _append_threshold_reason(
        reasons,
        evaluation.maximum_stationary_residual,
        options.maximum_stationary_residual,
        "stationary_residual",
    )
    _append_threshold_reason(
        reasons,
        evaluation.maximum_detailed_balance_residual,
        options.maximum_detailed_balance_residual,
        "detailed_balance_residual",
    )
    _append_threshold_reason(
        reasons,
        evaluation.maximum_event_reversal_residual,
        options.maximum_event_reversal_residual,
        "event_reversal_residual",
    )
    _append_threshold_reason(
        reasons,
        abs(evaluation.zero_charge_sigma_mS_cm),
        options.maximum_zero_charge_sigma_mS_cm,
        "zero_charge_sigma",
    )
    if not evaluation.higher_viscosity_lowers_dilute_conductivity:
        reasons.append("viscosity_monotonicity")
    if not evaluation.higher_packing_lowers_local_mobility:
        reasons.append("packing_monotonicity")
    return tuple(reasons)


def _append_threshold_reason(
    reasons: list[str],
    observed_value: float,
    maximum_allowed_value: float,
    reason: str,
) -> None:
    observed = _nonnegative_float(observed_value, reason)
    maximum_allowed = _nonnegative_float(maximum_allowed_value, f"{reason}.maximum")
    if observed > maximum_allowed:
        reasons.append(reason)


def _smooth_l1_loss_mS_cm(
    residual_mS_cm: float,
    huber_delta_mS_cm: float,
) -> float:
    residual = _finite_float(residual_mS_cm, "residual_mS_cm")
    huber_delta = _positive_float(huber_delta_mS_cm, "huber_delta_mS_cm")
    abs_residual = abs(residual)
    if abs_residual <= huber_delta:
        return float(0.5 * residual * residual / huber_delta)
    return float(abs_residual - 0.5 * huber_delta)


def _tail_huber_loss_mS_cm(
    residuals_mS_cm: tuple[float, ...],
    huber_delta_mS_cm: float,
    residual_tail_count: int,
) -> float:
    if not residuals_mS_cm:
        raise ValueError("residuals_mS_cm must be nonempty")
    tail_count = _positive_int(residual_tail_count, "residual_tail_count")
    residual_rankings: list[tuple[float, float]] = []
    for residual_mS_cm in residuals_mS_cm:
        parsed_residual = _finite_float(residual_mS_cm, "residual_mS_cm")
        residual_rankings.append((abs(parsed_residual), parsed_residual))
    residual_rankings.sort(reverse=True)
    tail_residuals = tuple(
        parsed_residual
        for abs_residual, parsed_residual in residual_rankings[:tail_count]
    )
    return float(
        math.fsum(
            _smooth_l1_loss_mS_cm(residual_mS_cm, huber_delta_mS_cm)
            for residual_mS_cm in tail_residuals
        )
        / len(tail_residuals)
    )


def _log_regularization_loss(
    log_values: tuple[float, ...],
    reference_log_values: tuple[float, ...],
    log_regularization_weight: float,
) -> float:
    if len(log_values) != len(reference_log_values):
        raise ValueError("regularization log tuples must have equal length")
    regularization_weight = _nonnegative_float(
        log_regularization_weight,
        "log_regularization_weight",
    )
    squared_distance = math.fsum(
        (log_value - reference_log_value) * (log_value - reference_log_value)
        for log_value, reference_log_value in zip(log_values, reference_log_values)
    )
    return float(regularization_weight * squared_distance)


def _mean_absolute_residual(residuals_mS_cm: tuple[float, ...]) -> float:
    if not residuals_mS_cm:
        raise ValueError("residuals_mS_cm must be nonempty")
    return float(
        math.fsum(abs(_finite_float(residual, "residual_mS_cm")) for residual in residuals_mS_cm)
        / len(residuals_mS_cm)
    )


def _pearson_correlation(
    empirical_sigmas_mS_cm: tuple[float, ...],
    predicted_sigmas_mS_cm: tuple[float, ...],
) -> float:
    if len(empirical_sigmas_mS_cm) != len(predicted_sigmas_mS_cm):
        raise ValueError("Pearson inputs must have equal length")
    if len(empirical_sigmas_mS_cm) < 2:
        raise ValueError("Pearson correlation requires at least two rows")
    empirical_values = np.asarray(empirical_sigmas_mS_cm, dtype=float)
    predicted_values = np.asarray(predicted_sigmas_mS_cm, dtype=float)
    if not np.all(np.isfinite(empirical_values)):
        raise ValueError("empirical sigmas must be finite")
    if not np.all(np.isfinite(predicted_values)):
        raise ValueError("predicted sigmas must be finite")
    empirical_std = float(np.std(empirical_values))
    predicted_std = float(np.std(predicted_values))
    if empirical_std <= 0.0 or predicted_std <= 0.0:
        raise ValueError("Pearson correlation requires nonconstant sigma values")
    return float(np.corrcoef(empirical_values, predicted_values)[0, 1])


def _validated_sigma_tuple(
    sigmas_mS_cm: tuple[float, ...],
    context: str,
) -> tuple[float, ...]:
    if not sigmas_mS_cm:
        raise ValueError(f"{context} must be nonempty")
    return tuple(_finite_float(sigma_mS_cm, context) for sigma_mS_cm in sigmas_mS_cm)


def _best_accepted_candidate(
    candidate_results: list[PrimitiveFitCandidateResult],
) -> PrimitiveFitCandidateResult:
    accepted_candidates = tuple(
        candidate_result for candidate_result in candidate_results
        if not candidate_result.rejected
    )
    if not accepted_candidates:
        rejection_summary = "; ".join(
            ",".join(candidate_result.rejection_reasons)
            for candidate_result in candidate_results
        )
        raise ValueError(
            "no primitive parameter candidate satisfied physical invariants: "
            f"{rejection_summary}"
        )
    return min(
        accepted_candidates,
        key=lambda candidate_result: candidate_result.objective_value,
    )


def _candidate_is_better(
    trial_result: PrimitiveFitCandidateResult,
    current_best: PrimitiveFitCandidateResult,
) -> bool:
    if trial_result.rejected:
        return False
    return trial_result.objective_value < current_best.objective_value


def select_primitive_parameter_promotion_candidate(
    candidate_results: list[PrimitiveFitCandidateResult],
    baseline_candidate: PrimitiveFitCandidateResult,
    options: PrimitiveFitOptions,
) -> PrimitiveFitCandidateResult:
    _validate_fit_options(options)
    if baseline_candidate.rejected:
        raise ValueError("baseline_candidate must satisfy physical invariants")
    accepted_candidates = tuple(
        candidate_result for candidate_result in candidate_results
        if not candidate_result.rejected
    )
    if not accepted_candidates:
        raise ValueError("promotion candidate selection requires accepted candidates")
    return min(
        accepted_candidates,
        key=lambda candidate_result: _promotion_candidate_sort_key(
            candidate_result,
            baseline_candidate,
            options,
        ),
    )


def _promotion_candidate_sort_key(
    candidate_result: PrimitiveFitCandidateResult,
    baseline_candidate: PrimitiveFitCandidateResult,
    options: PrimitiveFitOptions,
) -> tuple[float, ...]:
    mae_improved = candidate_result.mae_mS_cm < baseline_candidate.mae_mS_cm
    bias_improved = abs(candidate_result.bias_mS_cm) < abs(baseline_candidate.bias_mS_cm)
    pearson_improved = candidate_result.pearson_r > baseline_candidate.pearson_r
    baseline_improvement_deficit = float(
        (0 if mae_improved else 1)
        + (0 if bias_improved else 1)
        + (0 if pearson_improved else 1)
    )
    target_distance = _promotion_target_distance(candidate_result, options)
    return (
        baseline_improvement_deficit,
        target_distance,
        candidate_result.mae_mS_cm,
        abs(candidate_result.bias_mS_cm),
        -candidate_result.pearson_r,
        candidate_result.worst_abs_residual_mS_cm,
        candidate_result.objective_value,
    )


def _promotion_target_distance(
    candidate_result: PrimitiveFitCandidateResult,
    options: PrimitiveFitOptions,
) -> float:
    mae_violation = _normalized_positive_violation(
        candidate_result.mae_mS_cm,
        options.promotion_maximum_mae_mS_cm,
    )
    bias_violation = _normalized_positive_violation(
        abs(candidate_result.bias_mS_cm),
        options.promotion_maximum_abs_bias_mS_cm,
    )
    worst_residual_violation = _normalized_positive_violation(
        candidate_result.worst_abs_residual_mS_cm,
        options.promotion_maximum_worst_abs_residual_mS_cm,
    )
    return float(
        mae_violation * mae_violation
        + bias_violation * bias_violation
        + worst_residual_violation * worst_residual_violation
    )


def _normalized_positive_violation(
    observed_value: float,
    maximum_allowed_value: float,
) -> float:
    observed = _nonnegative_float(observed_value, "observed_value")
    maximum_allowed = _positive_float(maximum_allowed_value, "maximum_allowed_value")
    return max(0.0, (observed - maximum_allowed) / maximum_allowed)


def primitive_parameter_promotion_rejection_reasons(
    baseline_metrics: PrimitivePromotionMetrics,
    candidate_metrics: PrimitivePromotionMetrics,
    options: PrimitiveFitOptions,
) -> tuple[str, ...]:
    _validate_fit_options(options)
    reasons: list[str] = []
    if (
        options.promotion_require_mae_improvement
        and candidate_metrics.mae_mS_cm >= baseline_metrics.mae_mS_cm
    ):
        reasons.append("mae_not_improved")
    if candidate_metrics.mae_mS_cm > options.promotion_maximum_mae_mS_cm:
        reasons.append("mae_target")
    if (
        abs(candidate_metrics.bias_mS_cm)
        > options.promotion_maximum_abs_bias_mS_cm
    ):
        reasons.append("bias_target")
    if (
        candidate_metrics.worst_abs_residual_mS_cm
        > options.promotion_maximum_worst_abs_residual_mS_cm
    ):
        reasons.append("worst_residual_target")
    if candidate_metrics.failed_rows > options.maximum_failed_rows:
        reasons.append("failed_rows")
    _append_threshold_reason(
        reasons,
        candidate_metrics.maximum_mass_balance_residual,
        options.maximum_mass_balance_residual,
        "mass_balance_residual",
    )
    _append_threshold_reason(
        reasons,
        candidate_metrics.maximum_row_sum_residual,
        options.maximum_row_sum_residual,
        "row_sum_residual",
    )
    _append_threshold_reason(
        reasons,
        candidate_metrics.maximum_stationary_residual,
        options.maximum_stationary_residual,
        "stationary_residual",
    )
    _append_threshold_reason(
        reasons,
        candidate_metrics.maximum_detailed_balance_residual,
        options.maximum_detailed_balance_residual,
        "detailed_balance_residual",
    )
    _append_threshold_reason(
        reasons,
        candidate_metrics.maximum_event_reversal_residual,
        options.maximum_event_reversal_residual,
        "event_reversal_residual",
    )
    _append_threshold_reason(
        reasons,
        abs(candidate_metrics.zero_charge_sigma_mS_cm),
        options.maximum_zero_charge_sigma_mS_cm,
        "zero_charge_sigma",
    )
    if not candidate_metrics.higher_viscosity_lowers_dilute_conductivity:
        reasons.append("viscosity_monotonicity")
    if not candidate_metrics.higher_packing_lowers_local_mobility:
        reasons.append("packing_monotonicity")
    return tuple(reasons)


def write_primitive_parameter_candidate_artifact(
    artifact_path: str,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    baseline_metrics: PrimitivePromotionMetrics,
    candidate_metrics: PrimitivePromotionMetrics,
    source_labeled_rows: int,
    promotion_rejection_reasons: tuple[str, ...],
) -> None:
    validate_conductivity_primitive_parameters(primitive_parameters)
    artifact_path_text = _nonempty_string(artifact_path, "artifact_path")
    _positive_int(source_labeled_rows, "source_labeled_rows")
    artifact_mapping = {
        "artifact_type": "molecular_conductivity_primitive_parameter_candidate",
        "source_labeled_rows": int(source_labeled_rows),
        "promotion": {
            "accepted": not promotion_rejection_reasons,
            "rejection_reasons": list(promotion_rejection_reasons),
        },
        "baseline": _primitive_promotion_metrics_mapping(baseline_metrics),
        "candidate": _primitive_promotion_metrics_mapping(candidate_metrics),
        "molecular_conductivity_primitive_parameters": (
            conductivity_primitive_parameters_to_mapping(primitive_parameters)
        ),
    }
    output_path = Path(artifact_path_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_primitive_parameter_candidate_config(
    artifact_path: str,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    source_labeled_rows: int,
) -> None:
    validate_conductivity_primitive_parameters(primitive_parameters)
    artifact_path_text = _nonempty_string(artifact_path, "artifact_path")
    _positive_int(source_labeled_rows, "source_labeled_rows")
    artifact_mapping = {
        "artifact_type": "molecular_conductivity_primitive_parameter_candidate_config",
        "source_labeled_rows": int(source_labeled_rows),
        "molecular_conductivity_primitive_parameters": (
            conductivity_primitive_parameters_to_mapping(primitive_parameters)
        ),
    }
    output_path = Path(artifact_path_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_primitive_parameters_from_candidate_config_artifact(
    artifact_path: str,
) -> ConductivityPrimitiveParameterSet:
    artifact_value = _primitive_candidate_artifact_mapping(
        artifact_path,
        "molecular_conductivity_primitive_parameter_candidate_config",
    )
    return _primitive_parameters_from_artifact_mapping(artifact_value)


def load_primitive_parameters_from_candidate_artifact(
    artifact_path: str,
) -> ConductivityPrimitiveParameterSet:
    artifact_value = _primitive_candidate_artifact_mapping(
        artifact_path,
        "molecular_conductivity_primitive_parameter_candidate",
    )
    return _primitive_parameters_from_artifact_mapping(artifact_value)


def load_primitive_parameters_from_promoted_candidate_artifact(
    artifact_path: str,
) -> ConductivityPrimitiveParameterSet:
    artifact_value = _primitive_candidate_artifact_mapping(
        artifact_path,
        "molecular_conductivity_primitive_parameter_candidate",
    )
    if "promotion" not in artifact_value:
        raise ValueError("promoted primitive artifact missing promotion object")
    promotion_mapping = artifact_value["promotion"]
    if not isinstance(promotion_mapping, dict):
        raise TypeError("promoted primitive artifact promotion must be an object")
    if "accepted" not in promotion_mapping:
        raise ValueError("promoted primitive artifact missing promotion.accepted")
    promotion_accepted = promotion_mapping["accepted"]
    if not isinstance(promotion_accepted, bool):
        raise TypeError("promoted primitive artifact promotion.accepted must be boolean")
    if not promotion_accepted:
        if "rejection_reasons" not in promotion_mapping:
            raise ValueError(
                "rejected primitive artifact missing promotion.rejection_reasons"
            )
        rejection_reasons = promotion_mapping["rejection_reasons"]
        if not isinstance(rejection_reasons, list):
            raise TypeError(
                "rejected primitive artifact rejection_reasons must be a list"
            )
        raise ValueError(
            "primitive candidate artifact was not promoted: "
            + ",".join(str(reason) for reason in rejection_reasons)
        )
    return _primitive_parameters_from_artifact_mapping(artifact_value)


def _primitive_candidate_artifact_mapping(
    artifact_path: str,
    expected_artifact_type: str,
) -> dict:
    artifact_path_text = _nonempty_string(artifact_path, "artifact_path")
    expected_type_text = _nonempty_string(
        expected_artifact_type,
        "expected_artifact_type",
    )
    artifact_text = Path(artifact_path_text).read_text(encoding="utf-8")
    artifact_value = json.loads(artifact_text)
    if not isinstance(artifact_value, dict):
        raise TypeError("primitive candidate artifact must contain a JSON object")
    if "artifact_type" not in artifact_value:
        raise ValueError("primitive candidate artifact missing artifact_type")
    artifact_type = artifact_value["artifact_type"]
    if artifact_type != expected_type_text:
        raise ValueError(
            "primitive candidate artifact_type must be "
            f"{expected_type_text}, found {artifact_type}"
        )
    return artifact_value


def _primitive_parameters_from_artifact_mapping(
    artifact_value: dict,
) -> ConductivityPrimitiveParameterSet:
    if "molecular_conductivity_primitive_parameters" not in artifact_value:
        raise ValueError(
            "primitive candidate artifact missing "
            "molecular_conductivity_primitive_parameters"
        )
    parameter_mapping = artifact_value["molecular_conductivity_primitive_parameters"]
    if not isinstance(parameter_mapping, dict):
        raise TypeError(
            "primitive candidate artifact missing "
            "molecular_conductivity_primitive_parameters object"
        )
    return conductivity_primitive_parameters_from_mapping(parameter_mapping)


def _primitive_promotion_metrics_mapping(
    metrics: PrimitivePromotionMetrics,
) -> dict:
    return {
        "mae_mS_cm": _nonnegative_float(metrics.mae_mS_cm, "metrics.mae_mS_cm"),
        "bias_mS_cm": _finite_float(metrics.bias_mS_cm, "metrics.bias_mS_cm"),
        "pearson_r": _finite_float(metrics.pearson_r, "metrics.pearson_r"),
        "worst_abs_residual_mS_cm": _nonnegative_float(
            metrics.worst_abs_residual_mS_cm,
            "metrics.worst_abs_residual_mS_cm",
        ),
        "failed_rows": int(_nonnegative_int(metrics.failed_rows, "metrics.failed_rows")),
        "maximum_mass_balance_residual": _nonnegative_float(
            metrics.maximum_mass_balance_residual,
            "metrics.maximum_mass_balance_residual",
        ),
        "maximum_row_sum_residual": _nonnegative_float(
            metrics.maximum_row_sum_residual,
            "metrics.maximum_row_sum_residual",
        ),
        "maximum_stationary_residual": _nonnegative_float(
            metrics.maximum_stationary_residual,
            "metrics.maximum_stationary_residual",
        ),
        "maximum_detailed_balance_residual": _nonnegative_float(
            metrics.maximum_detailed_balance_residual,
            "metrics.maximum_detailed_balance_residual",
        ),
        "maximum_event_reversal_residual": _nonnegative_float(
            metrics.maximum_event_reversal_residual,
            "metrics.maximum_event_reversal_residual",
        ),
        "zero_charge_sigma_mS_cm": _finite_float(
            metrics.zero_charge_sigma_mS_cm,
            "metrics.zero_charge_sigma_mS_cm",
        ),
        "higher_viscosity_lowers_dilute_conductivity": (
            metrics.higher_viscosity_lowers_dilute_conductivity
        ),
        "higher_packing_lowers_local_mobility": (
            metrics.higher_packing_lowers_local_mobility
        ),
    }


def _finite_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{context} must be finite")
    return parsed_value


def _positive_float(value: float, context: str) -> float:
    parsed_value = _finite_float(value, context)
    if parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive")
    return parsed_value


def _nonnegative_float(value: float, context: str) -> float:
    parsed_value = _finite_float(value, context)
    if parsed_value < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    return parsed_value


def _positive_int(value: int, context: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _nonnegative_int(value: int, context: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < 0:
        raise ValueError(f"{context} must be nonnegative")
    return value


def _nonempty_string(value: str, context: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{context} must be a nonempty string")
    return value


def main() -> None:
    from data.electrolyte_property_db import DATA
    from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
    from conductivity.molecular_property_db_audit import (
        MolecularPropertyDbPrimitiveEvaluator,
        MolecularPropertyDbRegistrySource,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        audit_molecular_property_db_cases,
        build_molecular_property_db_case_selection,
        configured_conductivity_primitive_parameters,
        default_molecular_primitive_fit_configuration,
        default_molecular_property_db_audit_options,
        validate_molecular_property_db_audit_result,
    )

    audit_options = default_molecular_property_db_audit_options()
    fit_options, log_bounds = default_molecular_primitive_fit_configuration()
    registry_source = MolecularPropertyDbRegistrySource(
        solvent_registry=SOLVENTS,
        salt_registry=SALTS,
        additive_registry=ADDITIVES,
        cation_registry=CATION_PROPERTIES,
    )
    case_selection = build_molecular_property_db_case_selection(
        tuple(DATA),
        registry_source,
        audit_options,
    )
    evaluator = MolecularPropertyDbPrimitiveEvaluator(
        case_selection.cases,
        audit_options,
    )
    configured_parameters = configured_conductivity_primitive_parameters()
    baseline_audit_result = audit_molecular_property_db_cases(
        case_selection.cases,
        configured_parameters,
        audit_options,
    )
    validate_molecular_property_db_audit_result(
        baseline_audit_result,
        fit_options,
    )
    speciation_log_bounds = _log_bounds_for_parameter_names(
        log_bounds,
        SPECIATION_FIT_PARAMETER_NAMES,
    )
    speciation_fit_result = fit_conductivity_primitive_parameters(
        configured_parameters,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        speciation_log_bounds,
        evaluator,
        fit_options,
    )
    speciation_candidate = speciation_fit_result.best_candidate
    fit_result = fit_conductivity_primitive_parameters(
        speciation_candidate.primitive_parameters,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        log_bounds,
        evaluator,
        fit_options,
    )
    best_candidate = fit_result.best_candidate
    promotion_candidate = fit_result.promotion_candidate
    write_primitive_parameter_candidate_config(
        fit_options.candidate_output_path,
        promotion_candidate.primitive_parameters,
        case_selection.source_labeled_rows,
    )
    loaded_candidate_parameters = (
        load_primitive_parameters_from_candidate_config_artifact(
            fit_options.candidate_output_path
        )
    )
    candidate_audit_result = audit_molecular_property_db_cases(
        case_selection.cases,
        loaded_candidate_parameters,
        audit_options,
    )
    validate_molecular_property_db_audit_result(
        candidate_audit_result,
        fit_options,
    )
    baseline_metrics = PrimitivePromotionMetrics(
        mae_mS_cm=baseline_audit_result.mae_mS_cm,
        bias_mS_cm=baseline_audit_result.bias_mS_cm,
        pearson_r=baseline_audit_result.pearson_r,
        worst_abs_residual_mS_cm=(
            baseline_audit_result.maximum_abs_residual_mS_cm
        ),
        failed_rows=baseline_audit_result.failed_rows,
        maximum_mass_balance_residual=(
            baseline_audit_result.maximum_mass_balance_residual
        ),
        maximum_row_sum_residual=baseline_audit_result.maximum_row_sum_residual,
        maximum_stationary_residual=(
            baseline_audit_result.maximum_stationary_residual
        ),
        maximum_detailed_balance_residual=(
            baseline_audit_result.maximum_detailed_balance_residual
        ),
        maximum_event_reversal_residual=(
            baseline_audit_result.maximum_event_reversal_residual
        ),
        zero_charge_sigma_mS_cm=baseline_audit_result.zero_charge_sigma_mS_cm,
        higher_viscosity_lowers_dilute_conductivity=(
            baseline_audit_result.higher_viscosity_lowers_dilute_conductivity
        ),
        higher_packing_lowers_local_mobility=(
            baseline_audit_result.higher_packing_lowers_local_mobility
        ),
    )
    candidate_metrics = PrimitivePromotionMetrics(
        mae_mS_cm=candidate_audit_result.mae_mS_cm,
        bias_mS_cm=candidate_audit_result.bias_mS_cm,
        pearson_r=candidate_audit_result.pearson_r,
        worst_abs_residual_mS_cm=(
            candidate_audit_result.maximum_abs_residual_mS_cm
        ),
        failed_rows=candidate_audit_result.failed_rows,
        maximum_mass_balance_residual=(
            candidate_audit_result.maximum_mass_balance_residual
        ),
        maximum_row_sum_residual=candidate_audit_result.maximum_row_sum_residual,
        maximum_stationary_residual=(
            candidate_audit_result.maximum_stationary_residual
        ),
        maximum_detailed_balance_residual=(
            candidate_audit_result.maximum_detailed_balance_residual
        ),
        maximum_event_reversal_residual=(
            candidate_audit_result.maximum_event_reversal_residual
        ),
        zero_charge_sigma_mS_cm=candidate_audit_result.zero_charge_sigma_mS_cm,
        higher_viscosity_lowers_dilute_conductivity=(
            candidate_audit_result.higher_viscosity_lowers_dilute_conductivity
        ),
        higher_packing_lowers_local_mobility=(
            candidate_audit_result.higher_packing_lowers_local_mobility
        ),
    )
    promotion_rejection_reasons = primitive_parameter_promotion_rejection_reasons(
        baseline_metrics,
        candidate_metrics,
        fit_options,
    )
    write_primitive_parameter_candidate_artifact(
        fit_options.candidate_output_path,
        loaded_candidate_parameters,
        baseline_metrics,
        candidate_metrics,
        case_selection.source_labeled_rows,
        promotion_rejection_reasons,
    )
    print("molecular_primitive_parameter_fit")
    print(f"source_labeled_rows={case_selection.source_labeled_rows}")
    print(f"candidate_count={fit_result.candidate_count}")
    print(f"accepted_candidate_count={fit_result.accepted_candidate_count}")
    print(
        "speciation_prefit_candidate_count="
        f"{speciation_fit_result.candidate_count}"
    )
    print(
        "speciation_prefit_accepted_candidate_count="
        f"{speciation_fit_result.accepted_candidate_count}"
    )
    print(f"speciation_prefit_mae_mS_cm={speciation_candidate.mae_mS_cm:.6f}")
    print(f"speciation_prefit_bias_mS_cm={speciation_candidate.bias_mS_cm:.6f}")
    print(f"speciation_prefit_pearson_r={speciation_candidate.pearson_r:.6f}")
    print(
        "speciation_prefit_worst_abs_residual_mS_cm="
        f"{speciation_candidate.worst_abs_residual_mS_cm:.6f}"
    )
    print(f"best_objective={best_candidate.objective_value:.6f}")
    print(f"best_mae_mS_cm={best_candidate.mae_mS_cm:.6f}")
    print(f"best_bias_mS_cm={best_candidate.bias_mS_cm:.6f}")
    print(f"best_pearson_r={best_candidate.pearson_r:.6f}")
    print(f"best_worst_abs_residual_mS_cm={best_candidate.worst_abs_residual_mS_cm:.6f}")
    print(f"best_failed_rows={best_candidate.failed_rows}")
    print(f"best_rejected={best_candidate.rejected}")
    print(f"best_rejection_reasons={','.join(best_candidate.rejection_reasons)}")
    print(
        "promotion_candidate_objective="
        f"{promotion_candidate.objective_value:.6f}"
    )
    print(f"promotion_candidate_mae_mS_cm={promotion_candidate.mae_mS_cm:.6f}")
    print(f"promotion_candidate_bias_mS_cm={promotion_candidate.bias_mS_cm:.6f}")
    print(f"promotion_candidate_pearson_r={promotion_candidate.pearson_r:.6f}")
    print(
        "promotion_candidate_worst_abs_residual_mS_cm="
        f"{promotion_candidate.worst_abs_residual_mS_cm:.6f}"
    )
    print(f"baseline_mae_mS_cm={baseline_metrics.mae_mS_cm:.6f}")
    print(f"baseline_bias_mS_cm={baseline_metrics.bias_mS_cm:.6f}")
    print(f"baseline_pearson_r={baseline_metrics.pearson_r:.6f}")
    print(
        "baseline_worst_abs_residual_mS_cm="
        f"{baseline_metrics.worst_abs_residual_mS_cm:.6f}"
    )
    print(f"candidate_output_path={fit_options.candidate_output_path}")
    print(f"verified_candidate_mae_mS_cm={candidate_metrics.mae_mS_cm:.6f}")
    print(f"verified_candidate_bias_mS_cm={candidate_metrics.bias_mS_cm:.6f}")
    print(f"verified_candidate_pearson_r={candidate_metrics.pearson_r:.6f}")
    print(
        "verified_candidate_worst_abs_residual_mS_cm="
        f"{candidate_metrics.worst_abs_residual_mS_cm:.6f}"
    )
    print(f"verified_candidate_failed_rows={candidate_metrics.failed_rows}")
    print(
        "verified_candidate_event_reversal_residual="
        f"{candidate_metrics.maximum_event_reversal_residual:.6e}"
    )
    print(
        "promotion_status="
        f"{'accepted' if not promotion_rejection_reasons else 'rejected'}"
    )
    print(
        "promotion_rejection_reasons="
        f"{','.join(promotion_rejection_reasons)}"
    )
    print("verified_candidate_worst_rows")
    for row_result in sorted(
        candidate_audit_result.rows,
        key=lambda row: abs(row.residual_mS_cm),
        reverse=True,
    )[:audit_options.audit_worst_row_count]:
        print(
            "row_id={row_id} empirical={empirical:.6f} predicted={predicted:.6f} "
            "residual={residual:.6f} direct={direct:.6f} corrector={corrector:.6f} "
            "failed={failed} reason={reason}".format(
                row_id=row_result.row_id,
                empirical=row_result.empirical_sigma_mS_cm,
                predicted=row_result.predicted_sigma_mS_cm,
                residual=row_result.residual_mS_cm,
                direct=row_result.direct_sigma_mS_cm,
                corrector=row_result.corrector_sigma_mS_cm,
                failed=row_result.failed,
                reason=row_result.failure_reason,
            )
        )
    print("best_parameters")
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        print(
            f"{parameter_name}="
            f"{getattr(best_candidate.primitive_parameters, parameter_name):.17g}"
        )
    print("promotion_candidate_parameters")
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        print(
            f"{parameter_name}="
            f"{getattr(promotion_candidate.primitive_parameters, parameter_name):.17g}"
        )


if __name__ == "__main__":
    main()
