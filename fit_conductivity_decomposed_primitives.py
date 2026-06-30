"""Decomposed conductivity primitive calibration by direct and corrector owners."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np

from data.electrolyte_property_db import DATA
from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
from conductivity.fit_conductivity_primitive_parameters import (
    CLUSTER_SENSITIVITY_PARAMETER_NAMES,
    MOBILITY_EVENT_FIT_PARAMETER_NAMES,
    SPECIATION_FIT_PARAMETER_NAMES,
    ConductivityPrimitiveParameterEvaluator,
    PrimitiveFitDatasetEvaluation,
    PrimitiveFitOptions,
    PrimitiveParameterFitResult,
    PrimitiveParameterTransform,
    PrimitivePredictionSensitivityDiagnostics,
    PrimitivePromotionMetrics,
    fit_conductivity_primitive_parameters,
    load_primitive_parameters_from_candidate_config_artifact,
    primitive_parameter_promotion_rejection_reasons,
    primitive_prediction_sensitivity_diagnostics,
    write_primitive_parameter_candidate_artifact,
    write_primitive_parameter_candidate_config,
)
from conductivity.molecular_primitive_parameters import (
    ConductivityPrimitiveParameterSet,
    validate_conductivity_primitive_parameters,
)
from conductivity.molecular_property_db_audit import (
    MolecularPropertyDbAuditOptions,
    MolecularPropertyDbAuditResult,
    MolecularPropertyDbCase,
    MolecularPropertyDbPrimitiveEvaluator,
    MolecularPropertyDbRegistrySource,
    audit_molecular_property_db_cases,
    build_molecular_property_db_case_selection,
    configured_conductivity_primitive_parameters,
    default_molecular_primitive_fit_configuration,
    default_molecular_property_db_audit_options,
    REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
    validate_molecular_property_db_audit_result,
)


DIRECT_CAPACITY_BLOCK_PARAMETER_NAMES = (
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
    "hydrodynamic_radius_scale_positive_ion",
    "hydrodynamic_radius_scale_negative_ion",
    "hydrodynamic_radius_scale_cluster",
    "shape_friction_exponent",
    "free_volume_exponent",
    "dielectric_mobility_exponent",
    "solvation_mobility_exponent",
    "additive_shape_solvation_mobility_exponent",
    "positive_ion_charge_density_mobility_exponent",
    "negative_ion_charge_density_mobility_exponent",
    "positive_ion_counteranion_charge_cloud_mobility_exponent",
    "negative_ion_charge_cloud_mobility_exponent",
    "negative_ion_intrinsic_dielectric_drag_mobility_exponent",
    "negative_ion_shape_delocalization_mobility_exponent",
    "positive_ion_anion_disorder_mobility_exponent",
    "negative_ion_anion_disorder_mobility_exponent",
    "local_obstruction_strength",
    "local_obstruction_free_volume_exponent",
    "local_obstruction_ionic_strength_exponent",
    "local_obstruction_additive_solvation_exponent",
    "local_obstruction_size_exponent",
    "local_obstruction_charge_density_exponent",
    "local_obstruction_solvation_exponent",
)

CORRELATION_BLOCK_PARAMETER_NAMES = (
    "atmosphere_ep_scale",
    "atmosphere_rel_scale",
    "charge_cloud_radius_scale",
    "cross_relaxation_scale",
    "jump_length_scale",
    "atmosphere_capture_scale",
    "atmosphere_exit_scale",
    "association_conversion_rate_scale",
    "orientation_relaxation_rate_scale",
)

CLUSTER_SINK_BLOCK_PARAMETER_NAMES = (
    "coulomb_scale",
    "desolvation_scale",
    "coordination_scale",
    "cluster_entropy_penalty_scale",
    "association_crowding_stabilization_scale",
    "association_crowding_ionic_strength_exponent",
    "association_crowding_charge_density_exponent",
    "activity_debye_scale",
    "activity_size_scale",
    "activity_hard_sphere_scale",
    "cluster_activity_scale",
    "cluster_hydrodynamic_radius_scale",
) + CLUSTER_SENSITIVITY_PARAMETER_NAMES


@dataclass(frozen=True)
class DecomposedFitBlockResult:
    block_name: str
    selected_row_count: int
    active_parameter_count: int
    sensitivity_rank: int
    accepted: bool
    accepted_parameters: ConductivityPrimitiveParameterSet
    fit_result: PrimitiveParameterFitResult
    audit_result: MolecularPropertyDbAuditResult


@dataclass(frozen=True)
class TransportRoleDirectScalingAudit:
    selected_row_count: int
    transport_roles: tuple[str, ...]
    scale_factors: tuple[float, ...]
    objective_value: float
    target_direct_sigmas_mS_cm: tuple[float, ...]
    fitted_direct_sigmas_mS_cm: tuple[float, ...]


@dataclass(frozen=True)
class DecomposedFitResult:
    direct_capacity_result: DecomposedFitBlockResult
    correlation_result: DecomposedFitBlockResult
    cluster_sink_result: DecomposedFitBlockResult
    final_result: DecomposedFitBlockResult
    baseline_audit_result: MolecularPropertyDbAuditResult
    candidate_audit_result: MolecularPropertyDbAuditResult
    baseline_role_scaling_audit: TransportRoleDirectScalingAudit
    candidate_role_scaling_audit: TransportRoleDirectScalingAudit
    promotion_rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DecomposedFitContext:
    cases: tuple[MolecularPropertyDbCase, ...]
    audit_options: MolecularPropertyDbAuditOptions
    fit_options: PrimitiveFitOptions
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...]
    full_evaluator: MolecularPropertyDbPrimitiveEvaluator


class RowFilteredPrimitiveEvaluator:
    def __init__(
        self,
        base_evaluator: ConductivityPrimitiveParameterEvaluator,
        selected_row_indices: tuple[int, ...],
    ) -> None:
        if not selected_row_indices:
            raise ValueError("row-filtered evaluator requires selected rows")
        self._base_evaluator = base_evaluator
        self._selected_row_indices = selected_row_indices

    def evaluate(
        self,
        primitive_parameters: ConductivityPrimitiveParameterSet,
    ) -> PrimitiveFitDatasetEvaluation:
        full_evaluation = self._base_evaluator.evaluate(primitive_parameters)
        return PrimitiveFitDatasetEvaluation(
            empirical_sigmas_mS_cm=_select_tuple_values(
                full_evaluation.empirical_sigmas_mS_cm,
                self._selected_row_indices,
            ),
            predicted_sigmas_mS_cm=_select_tuple_values(
                full_evaluation.predicted_sigmas_mS_cm,
                self._selected_row_indices,
            ),
            direct_sigmas_mS_cm=_select_tuple_values(
                full_evaluation.direct_sigmas_mS_cm,
                self._selected_row_indices,
            ),
            corrector_sigmas_mS_cm=_select_tuple_values(
                full_evaluation.corrector_sigmas_mS_cm,
                self._selected_row_indices,
            ),
            direct_capacity_gaps_mS_cm=_select_tuple_values(
                full_evaluation.direct_capacity_gaps_mS_cm,
                self._selected_row_indices,
            ),
            corrector_targets_mS_cm=_select_tuple_values(
                full_evaluation.corrector_targets_mS_cm,
                self._selected_row_indices,
            ),
            corrector_residuals_mS_cm=_select_tuple_values(
                full_evaluation.corrector_residuals_mS_cm,
                self._selected_row_indices,
            ),
            direct_capacity_failure_count=full_evaluation.direct_capacity_failure_count,
            corrector_too_strong_failure_count=(
                full_evaluation.corrector_too_strong_failure_count
            ),
            corrector_too_weak_failure_count=(
                full_evaluation.corrector_too_weak_failure_count
            ),
            empirical_sigma_spreads_mS_cm=_select_tuple_values(
                full_evaluation.empirical_sigma_spreads_mS_cm,
                self._selected_row_indices,
            ),
            cluster_activation_penalty=full_evaluation.cluster_activation_penalty,
            failed_rows=full_evaluation.failed_rows,
            maximum_mass_balance_residual=(
                full_evaluation.maximum_mass_balance_residual
            ),
            maximum_row_sum_residual=full_evaluation.maximum_row_sum_residual,
            maximum_stationary_residual=(
                full_evaluation.maximum_stationary_residual
            ),
            maximum_detailed_balance_residual=(
                full_evaluation.maximum_detailed_balance_residual
            ),
            maximum_event_reversal_residual=(
                full_evaluation.maximum_event_reversal_residual
            ),
            zero_charge_sigma_mS_cm=full_evaluation.zero_charge_sigma_mS_cm,
            higher_viscosity_lowers_dilute_conductivity=(
                full_evaluation.higher_viscosity_lowers_dilute_conductivity
            ),
            higher_packing_lowers_local_mobility=(
                full_evaluation.higher_packing_lowers_local_mobility
            ),
            consumed_parameter_fields=full_evaluation.consumed_parameter_fields,
        )


def fit_decomposed_conductivity_primitives() -> DecomposedFitResult:
    audit_options = default_molecular_property_db_audit_options()
    fit_options, coordinate_bounds = default_molecular_primitive_fit_configuration()
    block_fit_options = replace(
        fit_options,
        cluster_activation_loss_weight=(
            fit_options.decomposed_block_cluster_activation_loss_weight
        ),
        powell_max_iterations_per_parameter=(
            fit_options.decomposed_block_powell_max_iterations_per_parameter
        ),
        powell_max_function_evaluations_per_parameter=(
            fit_options.decomposed_block_powell_max_function_evaluations_per_parameter
        ),
    )
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
    block_evaluator = MolecularPropertyDbPrimitiveEvaluator(
        case_selection.cases,
        audit_options,
        block_fit_options,
    )
    full_evaluator = MolecularPropertyDbPrimitiveEvaluator(
        case_selection.cases,
        audit_options,
        fit_options,
    )
    decomposed_context = DecomposedFitContext(
        cases=case_selection.cases,
        audit_options=audit_options,
        fit_options=block_fit_options,
        coordinate_bounds=coordinate_bounds,
        full_evaluator=block_evaluator,
    )
    final_decomposed_context = DecomposedFitContext(
        cases=case_selection.cases,
        audit_options=audit_options,
        fit_options=fit_options,
        coordinate_bounds=coordinate_bounds,
        full_evaluator=full_evaluator,
    )
    current_parameters = configured_conductivity_primitive_parameters()
    baseline_audit_result = audit_molecular_property_db_cases(
        case_selection.cases,
        current_parameters,
        audit_options,
    )
    validate_molecular_property_db_audit_result(baseline_audit_result, fit_options)

    direct_capacity_result = _run_decomposed_block(
        "direct_capacity",
        current_parameters,
        baseline_audit_result,
        DIRECT_CAPACITY_BLOCK_PARAMETER_NAMES,
        tuple(
            row_index
            for row_index, row_result in enumerate(baseline_audit_result.rows)
            if row_result.direct_capacity_failure
        ),
        decomposed_context,
    )
    current_parameters = direct_capacity_result.accepted_parameters

    correlation_start_audit_result = direct_capacity_result.audit_result
    correlation_result = _run_decomposed_block(
        "corrector",
        current_parameters,
        correlation_start_audit_result,
        CORRELATION_BLOCK_PARAMETER_NAMES,
        tuple(
            row_index
            for row_index, row_result in enumerate(correlation_start_audit_result.rows)
            if (
                not row_result.direct_capacity_failure
                and (
                    row_result.corrector_too_strong_failure
                    or row_result.corrector_too_weak_failure
                )
            )
        ),
        decomposed_context,
    )
    current_parameters = correlation_result.accepted_parameters

    cluster_start_audit_result = correlation_result.audit_result
    cluster_sink_result = _run_decomposed_block(
        "cluster_sink",
        current_parameters,
        cluster_start_audit_result,
        CLUSTER_SINK_BLOCK_PARAMETER_NAMES,
        _cluster_sink_row_indices(cluster_start_audit_result, fit_options),
        decomposed_context,
    )
    current_parameters = cluster_sink_result.accepted_parameters

    final_start_audit_result = cluster_sink_result.audit_result
    final_result = _run_decomposed_block(
        "final_joint",
        current_parameters,
        final_start_audit_result,
        SPECIATION_FIT_PARAMETER_NAMES + MOBILITY_EVENT_FIT_PARAMETER_NAMES,
        tuple(range(len(final_start_audit_result.rows))),
        final_decomposed_context,
    )

    write_primitive_parameter_candidate_config(
        fit_options.candidate_output_path,
        final_result.accepted_parameters,
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
    validate_molecular_property_db_audit_result(candidate_audit_result, fit_options)
    baseline_role_scaling_audit = _transport_role_direct_scaling_audit(
        baseline_audit_result,
        fit_options,
    )
    candidate_role_scaling_audit = _transport_role_direct_scaling_audit(
        candidate_audit_result,
        fit_options,
    )
    baseline_metrics = PrimitivePromotionMetrics(
        mae_mS_cm=baseline_audit_result.mae_mS_cm,
        bias_mS_cm=baseline_audit_result.bias_mS_cm,
        pearson_r=baseline_audit_result.pearson_r,
        worst_abs_residual_mS_cm=baseline_audit_result.maximum_abs_residual_mS_cm,
        failed_rows=baseline_audit_result.failed_rows,
        maximum_mass_balance_residual=(
            baseline_audit_result.maximum_mass_balance_residual
        ),
        maximum_row_sum_residual=baseline_audit_result.maximum_row_sum_residual,
        maximum_stationary_residual=baseline_audit_result.maximum_stationary_residual,
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
        worst_abs_residual_mS_cm=candidate_audit_result.maximum_abs_residual_mS_cm,
        failed_rows=candidate_audit_result.failed_rows,
        maximum_mass_balance_residual=(
            candidate_audit_result.maximum_mass_balance_residual
        ),
        maximum_row_sum_residual=candidate_audit_result.maximum_row_sum_residual,
        maximum_stationary_residual=candidate_audit_result.maximum_stationary_residual,
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
    return DecomposedFitResult(
        direct_capacity_result=direct_capacity_result,
        correlation_result=correlation_result,
        cluster_sink_result=cluster_sink_result,
        final_result=final_result,
        baseline_audit_result=baseline_audit_result,
        candidate_audit_result=candidate_audit_result,
        baseline_role_scaling_audit=baseline_role_scaling_audit,
        candidate_role_scaling_audit=candidate_role_scaling_audit,
        promotion_rejection_reasons=promotion_rejection_reasons,
    )


def _run_decomposed_block(
    block_name: str,
    initial_parameters: ConductivityPrimitiveParameterSet,
    starting_audit_result: MolecularPropertyDbAuditResult,
    block_parameter_names: tuple[str, ...],
    selected_row_indices: tuple[int, ...],
    decomposed_context: DecomposedFitContext,
) -> DecomposedFitBlockResult:
    validate_conductivity_primitive_parameters(initial_parameters)
    if not selected_row_indices:
        raise ValueError(f"{block_name} block selected no calibration rows")
    base_block_coordinate_bounds = _coordinate_bounds_for_parameter_names(
        decomposed_context.coordinate_bounds,
        _unique_parameter_names(block_parameter_names),
    )
    block_evaluator = RowFilteredPrimitiveEvaluator(
        decomposed_context.full_evaluator,
        selected_row_indices,
    )
    sensitivity_diagnostics = primitive_prediction_sensitivity_diagnostics(
        initial_parameters,
        base_block_coordinate_bounds,
        block_evaluator,
        decomposed_context.fit_options,
    )
    active_coordinate_bounds = _coordinate_bounds_for_identifiable_parameters(
        base_block_coordinate_bounds,
        sensitivity_diagnostics,
    )
    fit_result = fit_conductivity_primitive_parameters(
        initial_parameters,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        active_coordinate_bounds,
        block_evaluator,
        decomposed_context.fit_options,
    )
    candidate_audit_result = audit_molecular_property_db_cases(
        decomposed_context.cases,
        fit_result.best_candidate.primitive_parameters,
        decomposed_context.audit_options,
    )
    validate_molecular_property_db_audit_result(
        candidate_audit_result,
        decomposed_context.fit_options,
    )
    accepted = _block_candidate_preserves_full_audit(
        starting_audit_result,
        candidate_audit_result,
    )
    accepted_parameters = (
        fit_result.best_candidate.primitive_parameters
        if accepted
        else initial_parameters
    )
    audit_result = candidate_audit_result if accepted else starting_audit_result
    _print_decomposed_block_result(
        block_name,
        selected_row_indices,
        sensitivity_diagnostics,
        active_coordinate_bounds,
        fit_result,
        starting_audit_result,
        candidate_audit_result,
        audit_result,
        accepted,
    )
    return DecomposedFitBlockResult(
        block_name=block_name,
        selected_row_count=len(selected_row_indices),
        active_parameter_count=len(active_coordinate_bounds),
        sensitivity_rank=sensitivity_diagnostics.rank,
        accepted=accepted,
        accepted_parameters=accepted_parameters,
        fit_result=fit_result,
        audit_result=audit_result,
    )


def _block_candidate_preserves_full_audit(
    starting_audit_result: MolecularPropertyDbAuditResult,
    candidate_audit_result: MolecularPropertyDbAuditResult,
) -> bool:
    return (
        candidate_audit_result.mae_mS_cm <= starting_audit_result.mae_mS_cm
        and candidate_audit_result.maximum_abs_residual_mS_cm
        <= starting_audit_result.maximum_abs_residual_mS_cm
    )


def _cluster_sink_row_indices(
    audit_result: MolecularPropertyDbAuditResult,
    fit_options: PrimitiveFitOptions,
) -> tuple[int, ...]:
    sorted_index_and_row = tuple(
        sorted(
            enumerate(audit_result.rows),
            key=lambda index_and_row: abs(index_and_row[1].residual_mS_cm),
            reverse=True,
        )
    )
    tail_index_and_row = sorted_index_and_row[:fit_options.residual_tail_count]
    return tuple(
        row_index
        for row_index, row_result in tail_index_and_row
        if (
            row_result.neutral_cluster_fraction > 0.0
            or row_result.charged_cluster_fraction > 0.0
            or row_result.charged_cluster_net_sigma_mS_cm != 0.0
        )
    )


def _transport_role_direct_scaling_audit(
    audit_result: MolecularPropertyDbAuditResult,
    fit_options: PrimitiveFitOptions,
) -> TransportRoleDirectScalingAudit:
    if not audit_result.rows:
        raise ValueError("role scaling audit requires audit rows")
    transport_roles = tuple(
        audit_result.rows[0].direct_sigma_by_transport_role_mS_cm.keys()
    )
    if not transport_roles:
        raise ValueError("role scaling audit requires transport-role fields")
    tail_rows = tuple(
        row_result
        for row_result in sorted(
            audit_result.rows,
            key=lambda row_result: abs(row_result.residual_mS_cm),
            reverse=True,
        )[: fit_options.residual_tail_count]
        if row_result.direct_capacity_failure
    )
    if not tail_rows:
        return TransportRoleDirectScalingAudit(
            selected_row_count=0,
            transport_roles=transport_roles,
            scale_factors=tuple(1.0 for transport_role in transport_roles),
            objective_value=0.0,
            target_direct_sigmas_mS_cm=tuple(),
            fitted_direct_sigmas_mS_cm=tuple(),
        )
    role_direct_matrix = np.asarray(
        tuple(
            tuple(
                row_result.direct_sigma_by_transport_role_mS_cm[transport_role]
                for transport_role in transport_roles
            )
            for row_result in tail_rows
        ),
        dtype=float,
    )
    target_direct_vector = np.asarray(
        tuple(row_result.empirical_sigma_mS_cm for row_result in tail_rows),
        dtype=float,
    )
    scale_factors = _solve_bounded_role_direct_scaling(
        role_direct_matrix,
        target_direct_vector,
        fit_options.role_direct_scaling_regularization_weight,
        fit_options.role_direct_scaling_lower_bound,
        fit_options.role_direct_scaling_upper_bound,
    )
    fitted_direct_vector = role_direct_matrix @ np.asarray(scale_factors, dtype=float)
    objective_value = _role_direct_scaling_objective(
        role_direct_matrix,
        target_direct_vector,
        np.asarray(scale_factors, dtype=float),
        fit_options.role_direct_scaling_regularization_weight,
    )
    return TransportRoleDirectScalingAudit(
        selected_row_count=len(tail_rows),
        transport_roles=transport_roles,
        scale_factors=scale_factors,
        objective_value=objective_value,
        target_direct_sigmas_mS_cm=tuple(float(value) for value in target_direct_vector),
        fitted_direct_sigmas_mS_cm=tuple(float(value) for value in fitted_direct_vector),
    )


def _solve_bounded_role_direct_scaling(
    role_direct_matrix: np.ndarray,
    target_direct_vector: np.ndarray,
    regularization_weight: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, ...]:
    if role_direct_matrix.ndim != 2:
        raise ValueError("role_direct_matrix must be two-dimensional")
    row_count, role_count = role_direct_matrix.shape
    if row_count == 0 or role_count == 0:
        raise ValueError("role_direct_matrix must be nonempty")
    if target_direct_vector.shape != (row_count,):
        raise ValueError("target_direct_vector shape must match role rows")
    if not np.all(np.isfinite(role_direct_matrix)):
        raise ValueError("role_direct_matrix values must be finite")
    if not np.all(np.isfinite(target_direct_vector)):
        raise ValueError("target_direct_vector values must be finite")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be below upper_bound")
    best_scale_vector = np.ones(role_count, dtype=float)
    best_objective_value = np.inf
    found_feasible_scale_vector = False
    active_set_status_values = (0, 1, 2)
    for active_set_statuses in itertools.product(
        active_set_status_values,
        repeat=role_count,
    ):
        candidate_scale_vector = np.ones(role_count, dtype=float)
        free_role_indices: list[int] = []
        for role_index, active_set_status in enumerate(active_set_statuses):
            if active_set_status == 0:
                free_role_indices.append(role_index)
            elif active_set_status == 1:
                candidate_scale_vector[role_index] = lower_bound
            else:
                candidate_scale_vector[role_index] = upper_bound
        if free_role_indices:
            fixed_role_indices = tuple(
                role_index
                for role_index in range(role_count)
                if role_index not in free_role_indices
            )
            fixed_direct_vector = np.zeros(row_count, dtype=float)
            if fixed_role_indices:
                fixed_direct_vector = (
                    role_direct_matrix[:, fixed_role_indices]
                    @ candidate_scale_vector[list(fixed_role_indices)]
                )
            free_role_matrix = role_direct_matrix[:, free_role_indices]
            adjusted_target_vector = target_direct_vector - fixed_direct_vector
            free_system_matrix = (
                free_role_matrix.T @ free_role_matrix
                + regularization_weight * np.eye(len(free_role_indices))
            )
            free_target_vector = (
                free_role_matrix.T @ adjusted_target_vector
                + regularization_weight * np.ones(len(free_role_indices))
            )
            free_scale_vector = np.linalg.solve(
                free_system_matrix,
                free_target_vector,
            )
            if np.any(free_scale_vector < lower_bound):
                continue
            if np.any(free_scale_vector > upper_bound):
                continue
            candidate_scale_vector[free_role_indices] = free_scale_vector
        objective_value = _role_direct_scaling_objective(
            role_direct_matrix,
            target_direct_vector,
            candidate_scale_vector,
            regularization_weight,
        )
        if objective_value < best_objective_value:
            best_objective_value = objective_value
            best_scale_vector = candidate_scale_vector
            found_feasible_scale_vector = True
    if not found_feasible_scale_vector:
        raise ValueError("bounded role direct-scaling solve found no feasible point")
    return tuple(float(value) for value in best_scale_vector)


def _role_direct_scaling_objective(
    role_direct_matrix: np.ndarray,
    target_direct_vector: np.ndarray,
    scale_vector: np.ndarray,
    regularization_weight: float,
) -> float:
    residual_vector = role_direct_matrix @ scale_vector - target_direct_vector
    regularization_vector = scale_vector - np.ones(scale_vector.shape[0], dtype=float)
    return float(
        residual_vector @ residual_vector
        + regularization_weight * (regularization_vector @ regularization_vector)
    )


def _select_tuple_values(
    values: tuple[float, ...],
    selected_row_indices: tuple[int, ...],
) -> tuple[float, ...]:
    if not values:
        raise ValueError("cannot select from an empty tuple")
    return tuple(values[row_index] for row_index in selected_row_indices)


def _coordinate_bounds_for_parameter_names(
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    parameter_names: tuple[str, ...],
) -> tuple[PrimitiveParameterTransform, ...]:
    requested_parameter_names = set(parameter_names)
    selected_coordinate_bounds = tuple(
        coordinate_bound
        for coordinate_bound in coordinate_bounds
        if coordinate_bound.name in requested_parameter_names
    )
    if not selected_coordinate_bounds:
        raise ValueError("selected coordinate bounds must contain at least one parameter")
    return selected_coordinate_bounds


def _coordinate_bounds_for_identifiable_parameters(
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    sensitivity_diagnostics: PrimitivePredictionSensitivityDiagnostics,
) -> tuple[PrimitiveParameterTransform, ...]:
    identifiable_parameter_names = set(
        sensitivity_diagnostics.identifiable_parameter_names
    )
    selected_coordinate_bounds = tuple(
        coordinate_bound
        for coordinate_bound in coordinate_bounds
        if coordinate_bound.name in identifiable_parameter_names
    )
    if not selected_coordinate_bounds:
        raise ValueError("identifiability analysis selected no active parameters")
    return selected_coordinate_bounds


def _unique_parameter_names(
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    ordered_parameter_names: list[str] = []
    seen_parameter_names: set[str] = set()
    for parameter_name in parameter_names:
        if parameter_name in seen_parameter_names:
            continue
        ordered_parameter_names.append(parameter_name)
        seen_parameter_names.add(parameter_name)
    return tuple(ordered_parameter_names)


def _print_decomposed_block_result(
    block_name: str,
    selected_row_indices: tuple[int, ...],
    sensitivity_diagnostics: PrimitivePredictionSensitivityDiagnostics,
    active_coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    fit_result: PrimitiveParameterFitResult,
    starting_audit_result: MolecularPropertyDbAuditResult,
    candidate_audit_result: MolecularPropertyDbAuditResult,
    audit_result: MolecularPropertyDbAuditResult,
    accepted: bool,
) -> None:
    print(f"block={block_name}")
    print(f"block_selected_row_count={len(selected_row_indices)}")
    print(f"block_prediction_sensitivity_rank={sensitivity_diagnostics.rank}")
    print(
        "block_prediction_sensitivity_invalid_trial_count="
        f"{len(sensitivity_diagnostics.invalid_trial_parameter_names)}"
    )
    print(f"block_active_parameter_count={len(active_coordinate_bounds)}")
    print(f"block_candidate_count={fit_result.candidate_count}")
    print(f"block_accepted_candidate_count={fit_result.accepted_candidate_count}")
    print(f"block_update_accepted={accepted}")
    print(f"block_start_mae_mS_cm={starting_audit_result.mae_mS_cm:.6f}")
    print(f"block_candidate_mae_mS_cm={candidate_audit_result.mae_mS_cm:.6f}")
    print(
        "block_candidate_worst_abs_residual_mS_cm="
        f"{candidate_audit_result.maximum_abs_residual_mS_cm:.6f}"
    )
    print(f"block_end_mae_mS_cm={audit_result.mae_mS_cm:.6f}")
    print(f"block_end_bias_mS_cm={audit_result.bias_mS_cm:.6f}")
    print(f"block_end_pearson_r={audit_result.pearson_r:.6f}")
    print(
        "block_end_worst_abs_residual_mS_cm="
        f"{audit_result.maximum_abs_residual_mS_cm:.6f}"
    )
    print(
        "block_end_direct_capacity_failures="
        f"{sum(row.direct_capacity_failure for row in audit_result.rows)}"
    )
    print(
        "block_end_corrector_too_strong_failures="
        f"{sum(row.corrector_too_strong_failure for row in audit_result.rows)}"
    )
    print(
        "block_end_corrector_too_weak_failures="
        f"{sum(row.corrector_too_weak_failure for row in audit_result.rows)}"
    )
    print(
        "block_active_parameters="
        f"{','.join(coordinate_bound.name for coordinate_bound in active_coordinate_bounds)}"
    )


def _print_decomposed_fit_result(result: DecomposedFitResult) -> None:
    candidate_audit_result = result.candidate_audit_result
    print("decomposed_molecular_primitive_parameter_fit")
    print(f"baseline_mae_mS_cm={result.baseline_audit_result.mae_mS_cm:.6f}")
    print(f"verified_candidate_mae_mS_cm={candidate_audit_result.mae_mS_cm:.6f}")
    print(f"verified_candidate_bias_mS_cm={candidate_audit_result.bias_mS_cm:.6f}")
    print(f"verified_candidate_pearson_r={candidate_audit_result.pearson_r:.6f}")
    print(
        "verified_candidate_worst_abs_residual_mS_cm="
        f"{candidate_audit_result.maximum_abs_residual_mS_cm:.6f}"
    )
    print(f"verified_candidate_failed_rows={candidate_audit_result.failed_rows}")
    print(
        "verified_candidate_event_reversal_residual="
        f"{candidate_audit_result.maximum_event_reversal_residual:.6e}"
    )
    print(
        "verified_candidate_direct_capacity_failures="
        f"{sum(row.direct_capacity_failure for row in candidate_audit_result.rows)}"
    )
    print(
        "verified_candidate_corrector_too_strong_failures="
        f"{sum(row.corrector_too_strong_failure for row in candidate_audit_result.rows)}"
    )
    print(
        "verified_candidate_corrector_too_weak_failures="
        f"{sum(row.corrector_too_weak_failure for row in candidate_audit_result.rows)}"
    )
    _print_transport_role_direct_scaling_audit(
        "baseline",
        result.baseline_role_scaling_audit,
    )
    _print_transport_role_direct_scaling_audit(
        "verified_candidate",
        result.candidate_role_scaling_audit,
    )
    print(
        "promotion_status="
        f"{'accepted' if not result.promotion_rejection_reasons else 'rejected'}"
    )
    print(
        "promotion_rejection_reasons="
        f"{','.join(result.promotion_rejection_reasons)}"
    )


def _print_transport_role_direct_scaling_audit(
    report_prefix: str,
    role_scaling_audit: TransportRoleDirectScalingAudit,
) -> None:
    print(
        f"{report_prefix}_role_direct_scaling_selected_row_count="
        f"{role_scaling_audit.selected_row_count}"
    )
    print(
        f"{report_prefix}_role_direct_scaling_objective="
        f"{role_scaling_audit.objective_value:.6f}"
    )
    for transport_role, scale_factor in zip(
        role_scaling_audit.transport_roles,
        role_scaling_audit.scale_factors,
    ):
        print(
            f"{report_prefix}_role_direct_scaling_factor_{transport_role}="
            f"{scale_factor:.6f}"
        )


def main() -> None:
    result = fit_decomposed_conductivity_primitives()
    _print_decomposed_fit_result(result)


if __name__ == "__main__":
    main()
