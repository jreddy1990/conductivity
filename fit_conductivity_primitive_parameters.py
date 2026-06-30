"""Fit descriptor-neutral conductivity primitive parameters."""

from __future__ import annotations

import json
import math
import random
import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Protocol, TYPE_CHECKING

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds
from scipy.optimize import minimize

from conductivity.molecular_primitive_parameters import (
    CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES,
    CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME,
    ConductivityPrimitiveParameterSet,
    PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED,
    PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE,
    conductivity_primitive_parameter_coordinate_values_for_names,
    conductivity_primitive_parameters_from_mapping,
    conductivity_primitive_parameters_to_mapping,
    conductivity_primitive_parameters_with_coordinate_updates,
    validate_conductivity_primitive_parameters,
)

if TYPE_CHECKING:
    from conductivity.molecular_property_db_audit import (
        MolecularPropertyDbAuditOptions,
        MolecularPropertyDbAuditResult,
        MolecularPropertyDbCase,
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
)

CLUSTER_SENSITIVITY_PARAMETER_NAMES = (
    "contact_pair_logK_offset",
    "solvent_separated_pair_logK_offset",
    "positive_charged_triplet_logK_offset",
    "negative_charged_triplet_logK_offset",
    "neutral_cluster_logK_offset",
    "higher_charged_cluster_logK_offset",
    "cluster_order_logK_slope",
    "cluster_charge_magnitude_logK_slope",
)

MOBILITY_EVENT_FIT_PARAMETER_NAMES = (
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

_CLUSTER_KIND_LOGK_PARAMETER_BY_KIND = {
    "contact_pair": "contact_pair_logK_offset",
    "solvent_separated_pair": "solvent_separated_pair_logK_offset",
    "positive_charged_triplet": "positive_charged_triplet_logK_offset",
    "negative_charged_triplet": "negative_charged_triplet_logK_offset",
    "neutral_cluster": "neutral_cluster_logK_offset",
    "higher_charged_cluster": "higher_charged_cluster_logK_offset",
}

PRIMITIVE_PARAMETER_FIT_CONFIG_KEY = "molecular_primitive_parameter_fit"


@dataclass(frozen=True)
class PrimitiveParameterTransform:
    name: str
    transform: Literal["log_positive", "identity_signed"]
    lower: float
    upper: float


@dataclass(frozen=True)
class PrimitiveFitOptions:
    huber_delta_mS_cm: float
    empirical_sigma_floor_mS_cm: float
    coordinate_regularization_weight: float
    residual_tail_loss_weight: float
    residual_tail_count: int
    cluster_activation_loss_weight: float
    cluster_activation_residual_threshold_mS_cm: float
    cluster_activation_min_charged_cluster_fraction: float
    cluster_activation_min_charged_cluster_net_sigma_mS_cm: float
    direct_capacity_loss_weight: float
    corrector_loss_weight: float
    role_direct_scaling_regularization_weight: float
    role_direct_scaling_lower_bound: float
    role_direct_scaling_upper_bound: float
    latin_hypercube_samples_per_parameter: float
    coordinate_search_rounds: int
    initial_coordinate_step: float
    coordinate_step_shrinkage: float
    minimum_coordinate_step: float
    powell_max_iterations_per_parameter: float
    powell_max_function_evaluations_per_parameter: float
    decomposed_block_powell_max_iterations_per_parameter: float
    decomposed_block_powell_max_function_evaluations_per_parameter: float
    decomposed_block_cluster_activation_loss_weight: float
    powell_xtol_coordinate: float
    powell_ftol_objective: float
    random_seed: int
    maximum_failed_rows: int
    maximum_mass_balance_residual: float
    maximum_row_sum_residual: float
    maximum_stationary_residual: float
    maximum_detailed_balance_residual: float
    maximum_event_reversal_residual: float
    maximum_zero_charge_sigma_mS_cm: float
    descriptor_matrix_high_correlation_threshold: float
    descriptor_matrix_condition_number_warn_threshold: float
    descriptor_matrix_reported_correlation_pair_count: int
    prediction_sensitivity_coordinate_step: float
    prediction_sensitivity_min_column_norm_mS_cm_per_coordinate: float
    prediction_sensitivity_relative_singular_value_threshold: float
    prediction_sensitivity_high_correlation_threshold: float
    prediction_sensitivity_reported_correlation_pair_count: int
    candidate_output_path: str
    promotion_maximum_mae_mS_cm: float
    promotion_maximum_abs_bias_mS_cm: float
    promotion_maximum_worst_abs_residual_mS_cm: float
    promotion_require_mae_improvement: bool


@dataclass(frozen=True)
class DescriptorCalibrationTarget:
    target_id: str
    source_row_ids: tuple[int, ...]
    descriptor_driver_values: tuple[tuple[str, float], ...]
    empirical_sigma_mS_cm: float
    empirical_sigma_spread_mS_cm: float
    residual_weight: float


@dataclass(frozen=True)
class PrimitiveFitDatasetEvaluation:
    descriptor_calibration_targets: tuple[DescriptorCalibrationTarget, ...]
    predicted_sigmas_mS_cm: tuple[float, ...]
    direct_sigmas_mS_cm: tuple[float, ...]
    corrector_sigmas_mS_cm: tuple[float, ...]
    direct_capacity_gaps_mS_cm: tuple[float, ...]
    corrector_targets_mS_cm: tuple[float, ...]
    corrector_residuals_mS_cm: tuple[float, ...]
    direct_capacity_failure_count: int
    corrector_too_strong_failure_count: int
    corrector_too_weak_failure_count: int
    cluster_activation_penalty: float
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


class MolecularPropertyDbPrimitiveEvaluator:
    def __init__(
        self,
        cases: tuple["MolecularPropertyDbCase", ...],
        audit_options: "MolecularPropertyDbAuditOptions",
        fit_options: PrimitiveFitOptions,
    ) -> None:
        if not cases:
            raise ValueError("molecular property-DB evaluator requires cases")
        self._cases = cases
        self._audit_options = audit_options
        self._fit_options = fit_options
        self._descriptor_calibration_targets = descriptor_calibration_targets_for_cases(
            cases,
            fit_options,
        )
        self._has_measured_consumed_parameter_fields = False
        self._consumed_parameter_fields: tuple[str, ...] = tuple()

    def evaluate(
        self,
        primitive_parameters: ConductivityPrimitiveParameterSet,
    ) -> PrimitiveFitDatasetEvaluation:
        from conductivity.molecular_property_db_audit import (
            audit_molecular_property_db_cases,
        )

        audit_result = audit_molecular_property_db_cases(
            self._cases,
            primitive_parameters,
            self._audit_options,
        )
        if not self._has_measured_consumed_parameter_fields:
            self._consumed_parameter_fields = _measured_consumed_parameter_fields(
                self._cases,
                primitive_parameters,
                self._audit_options,
                audit_result,
            )
            self._has_measured_consumed_parameter_fields = True
        return PrimitiveFitDatasetEvaluation(
            descriptor_calibration_targets=self._descriptor_calibration_targets,
            predicted_sigmas_mS_cm=tuple(
                row.predicted_sigma_mS_cm for row in audit_result.rows
            ),
            direct_sigmas_mS_cm=tuple(
                row.direct_sigma_mS_cm for row in audit_result.rows
            ),
            corrector_sigmas_mS_cm=tuple(
                row.corrector_sigma_mS_cm for row in audit_result.rows
            ),
            direct_capacity_gaps_mS_cm=tuple(
                row.direct_capacity_gap_mS_cm for row in audit_result.rows
            ),
            corrector_targets_mS_cm=tuple(
                row.corrector_target_mS_cm for row in audit_result.rows
            ),
            corrector_residuals_mS_cm=tuple(
                row.corrector_residual_mS_cm for row in audit_result.rows
            ),
            direct_capacity_failure_count=sum(
                1 for row in audit_result.rows if row.direct_capacity_failure
            ),
            corrector_too_strong_failure_count=sum(
                1 for row in audit_result.rows
                if row.corrector_too_strong_failure
            ),
            corrector_too_weak_failure_count=sum(
                1 for row in audit_result.rows
                if row.corrector_too_weak_failure
            ),
            cluster_activation_penalty=_cluster_activation_penalty(
                self._cases,
                primitive_parameters,
                self._audit_options,
                self._fit_options,
                audit_result,
            ),
            failed_rows=audit_result.failed_rows,
            maximum_mass_balance_residual=(
                audit_result.maximum_mass_balance_residual
            ),
            maximum_row_sum_residual=audit_result.maximum_row_sum_residual,
            maximum_stationary_residual=(
                audit_result.maximum_stationary_residual
            ),
            maximum_detailed_balance_residual=(
                audit_result.maximum_detailed_balance_residual
            ),
            maximum_event_reversal_residual=(
                audit_result.maximum_event_reversal_residual
            ),
            zero_charge_sigma_mS_cm=audit_result.zero_charge_sigma_mS_cm,
            higher_viscosity_lowers_dilute_conductivity=(
                audit_result.higher_viscosity_lowers_dilute_conductivity
            ),
            higher_packing_lowers_local_mobility=(
                audit_result.higher_packing_lowers_local_mobility
            ),
            consumed_parameter_fields=self._consumed_parameter_fields,
        )


def _cluster_activation_penalty(
    cases: tuple["MolecularPropertyDbCase", ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
    fit_options: PrimitiveFitOptions,
    audit_result: "MolecularPropertyDbAuditResult",
) -> float:
    from conductivity.molecular_property_db_audit import (
        cluster_sensitivity_diagnostics_for_row,
    )

    if fit_options.cluster_activation_loss_weight == 0.0:
        return 0.0
    case_by_row_id = {molecular_case.row_id: molecular_case for molecular_case in cases}
    selected_rows = tuple(
        sorted(
            audit_result.rows,
            key=lambda row_result: abs(row_result.residual_mS_cm),
            reverse=True,
        )[:fit_options.residual_tail_count]
    )
    penalty_terms: list[float] = []
    residual_threshold_mS_cm = fit_options.cluster_activation_residual_threshold_mS_cm
    minimum_charged_cluster_fraction = (
        fit_options.cluster_activation_min_charged_cluster_fraction
    )
    minimum_charged_cluster_net_sigma_mS_cm = (
        fit_options.cluster_activation_min_charged_cluster_net_sigma_mS_cm
    )
    for row_result in selected_rows:
        if abs(row_result.residual_mS_cm) <= residual_threshold_mS_cm:
            continue
        if row_result.row_id not in case_by_row_id:
            raise ValueError(f"missing molecular case for row {row_result.row_id}")
        if (
            row_result.charged_cluster_fraction
            >= minimum_charged_cluster_fraction
            and abs(row_result.charged_cluster_net_sigma_mS_cm)
            >= minimum_charged_cluster_net_sigma_mS_cm
        ):
            continue
        sensitivity_diagnostics = cluster_sensitivity_diagnostics_for_row(
            case_by_row_id[row_result.row_id],
            primitive_parameters,
            audit_options,
            row_result,
        )
        charged_cluster_sensitivity_weight = math.fsum(
            abs(sensitivity_diagnostic.sensitivity_mS_cm_per_logK)
            for sensitivity_diagnostic in sensitivity_diagnostics
            if (
                sensitivity_diagnostic.net_charge_number != 0
                and sensitivity_diagnostic.direction_needed == "increase_logK"
            )
        )
        if charged_cluster_sensitivity_weight <= 0.0:
            continue
        fraction_deficit = max(
            0.0,
            (
                minimum_charged_cluster_fraction
                - row_result.charged_cluster_fraction
            )
            / minimum_charged_cluster_fraction,
        )
        sigma_deficit = max(
            0.0,
            (
                minimum_charged_cluster_net_sigma_mS_cm
                - abs(row_result.charged_cluster_net_sigma_mS_cm)
            )
            / minimum_charged_cluster_net_sigma_mS_cm,
        )
        penalty_terms.append(
            charged_cluster_sensitivity_weight
            * (
                fraction_deficit * fraction_deficit
                + sigma_deficit * sigma_deficit
            )
        )
    return float(math.fsum(penalty_terms))


def _measured_consumed_parameter_fields(
    cases: tuple["MolecularPropertyDbCase", ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
    baseline_audit_result: "MolecularPropertyDbAuditResult",
) -> tuple[str, ...]:
    perturbation_scales = _positive_float_tuple(
        audit_options.parameter_consumption_perturbation_scales,
        "parameter_consumption_perturbation_scales",
    )
    if all(perturbation_scale == 1.0 for perturbation_scale in perturbation_scales):
        raise ValueError(
            "parameter_consumption_perturbation_scales must include a value "
            "that differs from one"
        )
    consumed_parameter_fields: list[str] = []
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        if _parameter_is_consumed_by_any_perturbation(
            cases,
            primitive_parameters,
            audit_options,
            baseline_audit_result,
            parameter_name,
            perturbation_scales,
        ):
            consumed_parameter_fields.append(parameter_name)
    return tuple(consumed_parameter_fields)


def _parameter_is_consumed_by_any_perturbation(
    cases: tuple["MolecularPropertyDbCase", ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
    baseline_audit_result: "MolecularPropertyDbAuditResult",
    parameter_name: str,
    perturbation_scales: tuple[float, ...],
) -> bool:
    from conductivity.molecular_property_db_audit import audit_molecular_property_db_cases

    baseline_parameter_value = getattr(primitive_parameters, parameter_name)
    for perturbation_scale in perturbation_scales:
        if perturbation_scale == 1.0:
            continue
        perturbed_parameter_value = _primitive_parameter_perturbed_value(
            parameter_name,
            baseline_parameter_value,
            perturbation_scale,
        )
        perturbed_parameters = replace(
            primitive_parameters,
            **{parameter_name: perturbed_parameter_value},
        )
        try:
            perturbed_audit_result = audit_molecular_property_db_cases(
                cases,
                perturbed_parameters,
                audit_options,
            )
        except (
            FloatingPointError,
            OverflowError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            return True
        if _audit_results_differ(
            baseline_audit_result,
            perturbed_audit_result,
        ):
            return True
    return False


def _primitive_parameter_perturbed_value(
    parameter_name: str,
    baseline_parameter_value: float,
    perturbation_scale: float,
) -> float:
    transform_name = CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[
        parameter_name
    ]
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return baseline_parameter_value * perturbation_scale
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return baseline_parameter_value + math.log(perturbation_scale)
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


def _audit_results_differ(
    baseline_audit_result: "MolecularPropertyDbAuditResult",
    perturbed_audit_result: "MolecularPropertyDbAuditResult",
) -> bool:
    if len(baseline_audit_result.rows) != len(perturbed_audit_result.rows):
        return True
    baseline_values = _audit_comparison_values(baseline_audit_result)
    perturbed_values = _audit_comparison_values(perturbed_audit_result)
    tolerance_factor = math.sqrt(np.finfo(float).eps)
    for baseline_value, perturbed_value in zip(
        baseline_values,
        perturbed_values,
    ):
        difference_scale = max(
            1.0,
            abs(baseline_value),
            abs(perturbed_value),
        )
        tolerance = tolerance_factor * difference_scale
        if abs(baseline_value - perturbed_value) > tolerance:
            return True
    return False


def _audit_comparison_values(
    audit_result: "MolecularPropertyDbAuditResult",
) -> tuple[float, ...]:
    values: list[float] = [
        audit_result.mae_mS_cm,
        audit_result.rmse_mS_cm,
        audit_result.bias_mS_cm,
        audit_result.pearson_r,
        audit_result.maximum_abs_residual_mS_cm,
        audit_result.maximum_mass_balance_residual,
        audit_result.maximum_row_sum_residual,
        audit_result.maximum_stationary_residual,
        audit_result.maximum_detailed_balance_residual,
        audit_result.maximum_event_reversal_residual,
        audit_result.zero_charge_sigma_mS_cm,
    ]
    for row_result in audit_result.rows:
        values.extend(
            (
                row_result.predicted_sigma_mS_cm,
                row_result.direct_sigma_mS_cm,
                row_result.corrector_sigma_mS_cm,
                row_result.direct_capacity_gap_mS_cm,
                row_result.corrector_target_mS_cm,
                row_result.corrector_residual_mS_cm,
                float(row_result.direct_capacity_failure),
                float(row_result.corrector_too_strong_failure),
                float(row_result.corrector_too_weak_failure),
                row_result.charge_weighted_transport_concentration_mol_m3,
                row_result.charged_cluster_direct_sigma_mS_cm,
                row_result.charged_cluster_corrector_sigma_mS_cm,
                row_result.charged_cluster_net_sigma_mS_cm,
                row_result.mass_balance_residual_mol_m3,
                row_result.row_sum_residual,
                row_result.stationary_residual_mol_m3_s,
                row_result.detailed_balance_residual_mol_m3_s,
                row_result.event_reversal_residual_mol_m3_s,
                row_result.free_ion_fraction,
                row_result.charged_cluster_fraction,
                row_result.neutral_cluster_fraction,
                row_result.cluster_transport_mobility_density_mol_m_s,
                row_result.charged_cluster_transport_mobility_density_mol_m_s,
                row_result.neutral_cluster_transport_mobility_density_mol_m_s,
            )
        )
        transport_roles = tuple(
            sorted(row_result.direct_sigma_by_transport_role_mS_cm)
        )
        for transport_role in transport_roles:
            values.extend(
                (
                    row_result.direct_sigma_by_transport_role_mS_cm[
                        transport_role
                    ],
                    row_result.corrector_sigma_by_transport_role_mS_cm[
                        transport_role
                    ],
                    row_result.net_sigma_by_transport_role_mS_cm[transport_role],
                )
            )
        for cluster_diagnostic in row_result.cluster_thermodynamic_diagnostics:
            values.extend(
                (
                    cluster_diagnostic.concentration_mol_m3,
                    cluster_diagnostic.concentration_fraction_of_total_ion,
                    cluster_diagnostic.standard_free_energy_J_mol,
                    cluster_diagnostic.standard_free_energy_over_RT,
                    cluster_diagnostic.log_equilibrium_constant,
                    cluster_diagnostic.coulomb_J_mol,
                    cluster_diagnostic.desolvation_J_mol,
                    cluster_diagnostic.coordination_J_mol,
                    cluster_diagnostic.steric_J_mol,
                    cluster_diagnostic.entropy_J_mol,
                    cluster_diagnostic.activity_correction_J_mol,
                    cluster_diagnostic.hydrodynamic_radius_A,
                    cluster_diagnostic.molecular_volume_A3,
                )
            )
    return tuple(_finite_float(value, "audit_comparison_value") for value in values)


def default_molecular_primitive_fit_configuration() -> tuple[
    PrimitiveFitOptions,
    tuple[PrimitiveParameterTransform, ...],
]:
    from conductivity.molecular_property_db_audit import (
        _load_optimization_config_section,
        _required_config_bool,
        _required_config_string,
        _required_mapping,
        _required_nonnegative_config_float,
        _required_nonnegative_config_int,
        _required_positive_config_float,
        _required_positive_config_int,
    )

    fit_config_mapping = _load_optimization_config_section(
        PRIMITIVE_PARAMETER_FIT_CONFIG_KEY
    )
    coordinate_bounds_mapping = _required_mapping(
        fit_config_mapping,
        "coordinate_bounds",
        "molecular_primitive_parameter_fit.coordinate_bounds",
    )
    unknown_coordinate_bound_names = tuple(
        sorted(
            parameter_name for parameter_name in coordinate_bounds_mapping
            if parameter_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        )
    )
    if unknown_coordinate_bound_names:
        raise ValueError(
            "unknown molecular primitive fit coordinate bounds: "
            f"{unknown_coordinate_bound_names}"
        )
    missing_coordinate_bound_names = tuple(
        parameter_name
        for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        if parameter_name not in coordinate_bounds_mapping
    )
    if missing_coordinate_bound_names:
        raise ValueError(
            "molecular primitive fit coordinate_bounds missing parameters "
            f"{missing_coordinate_bound_names}"
        )
    fit_options = PrimitiveFitOptions(
        huber_delta_mS_cm=_required_positive_config_float(
            fit_config_mapping,
            "huber_delta_mS_cm",
        ),
        empirical_sigma_floor_mS_cm=_required_positive_config_float(
            fit_config_mapping,
            "empirical_sigma_floor_mS_cm",
        ),
        coordinate_regularization_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "coordinate_regularization_weight",
        ),
        residual_tail_loss_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "residual_tail_loss_weight",
        ),
        residual_tail_count=_required_positive_config_int(
            fit_config_mapping,
            "residual_tail_count",
        ),
        cluster_activation_loss_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "cluster_activation_loss_weight",
        ),
        cluster_activation_residual_threshold_mS_cm=(
            _required_positive_config_float(
                fit_config_mapping,
                "cluster_activation_residual_threshold_mS_cm",
            )
        ),
        cluster_activation_min_charged_cluster_fraction=(
            _required_positive_config_float(
                fit_config_mapping,
                "cluster_activation_min_charged_cluster_fraction",
            )
        ),
        cluster_activation_min_charged_cluster_net_sigma_mS_cm=(
            _required_positive_config_float(
                fit_config_mapping,
                "cluster_activation_min_charged_cluster_net_sigma_mS_cm",
            )
        ),
        direct_capacity_loss_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "direct_capacity_loss_weight",
        ),
        corrector_loss_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "corrector_loss_weight",
        ),
        role_direct_scaling_regularization_weight=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "role_direct_scaling_regularization_weight",
            )
        ),
        role_direct_scaling_lower_bound=_required_positive_config_float(
            fit_config_mapping,
            "role_direct_scaling_lower_bound",
        ),
        role_direct_scaling_upper_bound=_required_positive_config_float(
            fit_config_mapping,
            "role_direct_scaling_upper_bound",
        ),
        latin_hypercube_samples_per_parameter=_required_positive_config_float(
            fit_config_mapping,
            "latin_hypercube_samples_per_parameter",
        ),
        coordinate_search_rounds=_required_nonnegative_config_int(
            fit_config_mapping,
            "coordinate_search_rounds",
        ),
        initial_coordinate_step=_required_positive_config_float(
            fit_config_mapping,
            "initial_coordinate_step",
        ),
        coordinate_step_shrinkage=_required_positive_config_float(
            fit_config_mapping,
            "coordinate_step_shrinkage",
        ),
        minimum_coordinate_step=_required_positive_config_float(
            fit_config_mapping,
            "minimum_coordinate_step",
        ),
        powell_max_iterations_per_parameter=_required_nonnegative_config_float(
            fit_config_mapping,
            "powell_max_iterations_per_parameter",
        ),
        powell_max_function_evaluations_per_parameter=_required_nonnegative_config_float(
            fit_config_mapping,
            "powell_max_function_evaluations_per_parameter",
        ),
        decomposed_block_powell_max_iterations_per_parameter=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "decomposed_block_powell_max_iterations_per_parameter",
            )
        ),
        decomposed_block_powell_max_function_evaluations_per_parameter=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "decomposed_block_powell_max_function_evaluations_per_parameter",
            )
        ),
        decomposed_block_cluster_activation_loss_weight=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "decomposed_block_cluster_activation_loss_weight",
            )
        ),
        powell_xtol_coordinate=_required_positive_config_float(
            fit_config_mapping,
            "powell_xtol_coordinate",
        ),
        powell_ftol_objective=_required_positive_config_float(
            fit_config_mapping,
            "powell_ftol_objective",
        ),
        random_seed=_required_nonnegative_config_int(
            fit_config_mapping,
            "random_seed",
        ),
        maximum_failed_rows=_required_nonnegative_config_int(
            fit_config_mapping,
            "maximum_failed_rows",
        ),
        maximum_mass_balance_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_mass_balance_residual",
        ),
        maximum_row_sum_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_row_sum_residual",
        ),
        maximum_stationary_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_stationary_residual",
        ),
        maximum_detailed_balance_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_detailed_balance_residual",
        ),
        maximum_event_reversal_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_event_reversal_residual",
        ),
        maximum_zero_charge_sigma_mS_cm=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_zero_charge_sigma_mS_cm",
        ),
        descriptor_matrix_high_correlation_threshold=(
            _required_positive_config_float(
                fit_config_mapping,
                "descriptor_matrix_high_correlation_threshold",
            )
        ),
        descriptor_matrix_condition_number_warn_threshold=(
            _required_positive_config_float(
                fit_config_mapping,
                "descriptor_matrix_condition_number_warn_threshold",
            )
        ),
        descriptor_matrix_reported_correlation_pair_count=(
            _required_nonnegative_config_int(
                fit_config_mapping,
                "descriptor_matrix_reported_correlation_pair_count",
            )
        ),
        prediction_sensitivity_coordinate_step=_required_positive_config_float(
            fit_config_mapping,
            "prediction_sensitivity_coordinate_step",
        ),
        prediction_sensitivity_min_column_norm_mS_cm_per_coordinate=(
            _required_positive_config_float(
                fit_config_mapping,
                "prediction_sensitivity_min_column_norm_mS_cm_per_coordinate",
            )
        ),
        prediction_sensitivity_relative_singular_value_threshold=(
            _required_positive_config_float(
                fit_config_mapping,
                "prediction_sensitivity_relative_singular_value_threshold",
            )
        ),
        prediction_sensitivity_high_correlation_threshold=(
            _required_positive_config_float(
                fit_config_mapping,
                "prediction_sensitivity_high_correlation_threshold",
            )
        ),
        prediction_sensitivity_reported_correlation_pair_count=(
            _required_nonnegative_config_int(
                fit_config_mapping,
                "prediction_sensitivity_reported_correlation_pair_count",
            )
        ),
        candidate_output_path=_required_config_string(
            fit_config_mapping,
            "candidate_output_path",
        ),
        promotion_maximum_mae_mS_cm=_required_positive_config_float(
            fit_config_mapping,
            "promotion_maximum_mae_mS_cm",
        ),
        promotion_maximum_abs_bias_mS_cm=_required_nonnegative_config_float(
            fit_config_mapping,
            "promotion_maximum_abs_bias_mS_cm",
        ),
        promotion_maximum_worst_abs_residual_mS_cm=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "promotion_maximum_worst_abs_residual_mS_cm",
            )
        ),
        promotion_require_mae_improvement=_required_config_bool(
            fit_config_mapping,
            "promotion_require_mae_improvement",
        ),
    )
    coordinate_bounds: list[PrimitiveParameterTransform] = []
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        coordinate_bounds.append(
            _primitive_fit_coordinate_bound_from_config(
                coordinate_bounds_mapping,
                parameter_name,
            )
        )
    return fit_options, tuple(coordinate_bounds)


def validate_molecular_property_db_audit_result(
    audit_result: "MolecularPropertyDbAuditResult",
    fit_options: PrimitiveFitOptions,
) -> None:
    if audit_result.evaluated_rows != audit_result.labeled_rows:
        raise ValueError(
            "molecular property-DB audit did not evaluate every labeled row: "
            f"{audit_result.evaluated_rows}/{audit_result.labeled_rows}"
        )
    if audit_result.failed_rows > fit_options.maximum_failed_rows:
        raise ValueError(
            "molecular property-DB audit failed row count "
            f"{audit_result.failed_rows} exceeds {fit_options.maximum_failed_rows}"
        )
    _threshold_or_raise(
        audit_result.maximum_mass_balance_residual,
        fit_options.maximum_mass_balance_residual,
        "maximum_mass_balance_residual",
    )
    _threshold_or_raise(
        audit_result.maximum_row_sum_residual,
        fit_options.maximum_row_sum_residual,
        "maximum_row_sum_residual",
    )
    _threshold_or_raise(
        audit_result.maximum_stationary_residual,
        fit_options.maximum_stationary_residual,
        "maximum_stationary_residual",
    )
    _threshold_or_raise(
        audit_result.maximum_detailed_balance_residual,
        fit_options.maximum_detailed_balance_residual,
        "maximum_detailed_balance_residual",
    )
    _threshold_or_raise(
        audit_result.maximum_event_reversal_residual,
        fit_options.maximum_event_reversal_residual,
        "maximum_event_reversal_residual",
    )
    _threshold_or_raise(
        abs(audit_result.zero_charge_sigma_mS_cm),
        fit_options.maximum_zero_charge_sigma_mS_cm,
        "zero_charge_sigma_mS_cm",
    )
    if not audit_result.higher_viscosity_lowers_dilute_conductivity:
        raise ValueError("higher-viscosity molecular invariant failed")
    if not audit_result.higher_packing_lowers_local_mobility:
        raise ValueError("higher-packing molecular invariant failed")


def _primitive_fit_coordinate_bound_from_config(
    coordinate_bounds_mapping: dict,
    parameter_name: str,
) -> PrimitiveParameterTransform:
    if parameter_name not in coordinate_bounds_mapping:
        raise ValueError(f"missing coordinate bound for {parameter_name}")
    bound_values = coordinate_bounds_mapping[parameter_name]
    if not isinstance(bound_values, list):
        raise TypeError(f"coordinate bound for {parameter_name} must be a list")
    expected_bound_count = 2
    if len(bound_values) != expected_bound_count:
        raise ValueError(
            f"coordinate bound for {parameter_name} must contain two values"
        )
    lower_value = _primitive_bound_endpoint_from_config(
        parameter_name,
        bound_values[0],
        "lower_bound",
    )
    upper_value = _primitive_bound_endpoint_from_config(
        parameter_name,
        bound_values[1],
        "upper_bound",
    )
    if lower_value >= upper_value:
        raise ValueError(
            f"coordinate bound for {parameter_name} lower must be below upper"
        )
    return PrimitiveParameterTransform(
        name=parameter_name,
        transform=CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[
            parameter_name
        ],
        lower=lower_value,
        upper=upper_value,
    )


def _primitive_bound_endpoint_from_config(
    parameter_name: str,
    endpoint_value: float,
    endpoint_name: str,
) -> float:
    context = f"{parameter_name}.{endpoint_name}"
    transform_name = CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[
        parameter_name
    ]
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return math.log(_positive_float(endpoint_value, context))
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return _finite_float(endpoint_value, context)
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


@dataclass(frozen=True)
class PrimitiveFitCandidateResult:
    primitive_parameters: ConductivityPrimitiveParameterSet
    coordinate_values: tuple[float, ...]
    objective_value: float
    mean_huber_loss_mS_cm: float
    tail_huber_loss_mS_cm: float
    direct_capacity_loss_mS_cm: float
    corrector_loss_mS_cm: float
    coordinate_regularization_loss: float
    cluster_activation_loss: float
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
class SpeciationSensitivityFitResult:
    candidate: PrimitiveFitCandidateResult
    sensitivity_row_count: int
    sensitivity_entry_count: int


@dataclass(frozen=True)
class PrimitiveDriverMatrixDiagnostics:
    row_count: int
    column_count: int
    rank: int
    condition_number: float
    zero_variance_columns: tuple[str, ...]
    high_correlation_pairs: tuple[tuple[str, str, float], ...]


@dataclass(frozen=True)
class PrimitivePredictionSensitivityDiagnostics:
    row_count: int
    parameter_count: int
    rank: int
    condition_number: float
    singular_values: tuple[float, ...]
    identifiable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    zero_sensitivity_parameter_names: tuple[str, ...]
    invalid_trial_parameter_names: tuple[str, ...]
    high_correlation_parameter_pairs: tuple[tuple[str, str, float], ...]


@dataclass(frozen=True)
class PrimitivePredictionSensitivityTrial:
    valid: bool
    predicted_sigmas_mS_cm: tuple[float, ...]


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
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitiveParameterFitResult:
    validate_conductivity_primitive_parameters(initial_parameters)
    validate_conductivity_primitive_parameters(regularization_reference_parameters)
    _validate_fit_options(options)
    ordered_bounds = _ordered_coordinate_bounds(coordinate_bounds)
    fitted_parameter_names = _ordered_bound_parameter_names(ordered_bounds)
    initial_coordinate_values = _bounded_initial_coordinate_values(
        conductivity_primitive_parameter_coordinate_values_for_names(
            initial_parameters,
            fitted_parameter_names,
        ),
        ordered_bounds,
    )
    regularization_reference_coordinate_values = (
        conductivity_primitive_parameter_coordinate_values_for_names(
            regularization_reference_parameters,
            fitted_parameter_names,
        )
    )

    candidate_results: list[PrimitiveFitCandidateResult] = []
    candidate_results.append(
        evaluate_primitive_parameter_candidate(
            initial_coordinate_values,
            initial_parameters,
            regularization_reference_coordinate_values,
            ordered_bounds,
            evaluator,
            options,
        )
    )
    random_number_generator = random.Random(options.random_seed)
    latin_hypercube_sample_count = _fit_budget_count_from_parameter_count(
        len(ordered_bounds),
        options.latin_hypercube_samples_per_parameter,
        "latin_hypercube_samples_per_parameter",
    )
    for sample_coordinate_values in _latin_hypercube_coordinate_values(
        ordered_bounds,
        latin_hypercube_sample_count,
        random_number_generator,
    ):
        candidate_results.append(
            evaluate_primitive_parameter_candidate(
                sample_coordinate_values,
                initial_parameters,
                regularization_reference_coordinate_values,
                ordered_bounds,
                evaluator,
                options,
            )
        )

    current_best = _best_accepted_candidate(candidate_results)
    coordinate_step_value = options.initial_coordinate_step
    for search_round_index in range(options.coordinate_search_rounds):
        if coordinate_step_value < options.minimum_coordinate_step:
            break
        improved_this_round = False
        for parameter_index, coordinate_bound in enumerate(ordered_bounds):
            for step_sign in (-1.0, 1.0):
                trial_coordinate_values = _coordinate_trial_coordinate_values(
                    current_best.coordinate_values,
                    parameter_index,
                    step_sign * coordinate_step_value,
                    coordinate_bound,
                )
                trial_result = evaluate_primitive_parameter_candidate(
                    trial_coordinate_values,
                    initial_parameters,
                    regularization_reference_coordinate_values,
                    ordered_bounds,
                    evaluator,
                    options,
                )
                candidate_results.append(trial_result)
                if _candidate_is_better(trial_result, current_best):
                    current_best = trial_result
                    improved_this_round = True
        if not improved_this_round:
            coordinate_step_value *= options.coordinate_step_shrinkage
        if search_round_index == options.coordinate_search_rounds - 1:
            break

    current_best = _run_powell_local_polish(
        current_best,
        candidate_results,
        initial_parameters,
        regularization_reference_coordinate_values,
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
    coordinate_values: tuple[float, ...],
    base_parameters: ConductivityPrimitiveParameterSet,
    regularization_reference_coordinate_values: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveParameterTransform, ...],
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitiveFitCandidateResult:
    _validate_fit_options(options)
    validate_conductivity_primitive_parameters(base_parameters)
    bounded_coordinate_values = _bounded_initial_coordinate_values(coordinate_values, ordered_bounds)
    if len(regularization_reference_coordinate_values) != len(ordered_bounds):
        raise ValueError(
            "regularization_reference_coordinate_values length must match coordinate bound count"
        )
    primitive_parameters = conductivity_primitive_parameters_with_coordinate_updates(
        base_parameters,
        _ordered_bound_parameter_names(ordered_bounds),
        bounded_coordinate_values,
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
            bounded_coordinate_values,
            regularization_reference_coordinate_values,
            options,
            type(evaluation_error).__name__,
        )
    descriptor_calibration_targets = _validated_descriptor_calibration_targets(
        evaluation.descriptor_calibration_targets,
    )
    empirical_sigmas = tuple(
        descriptor_target.empirical_sigma_mS_cm
        for descriptor_target in descriptor_calibration_targets
    )
    residual_weights = tuple(
        descriptor_target.residual_weight
        for descriptor_target in descriptor_calibration_targets
    )
    predicted_sigmas = _validated_sigma_tuple(
        evaluation.predicted_sigmas_mS_cm,
        "predicted_sigmas_mS_cm",
    )
    if len(empirical_sigmas) != len(predicted_sigmas):
        raise ValueError("empirical and predicted sigma tuples must have equal length")
    direct_sigmas = _validated_sigma_spread_tuple(
        evaluation.direct_sigmas_mS_cm,
        "direct_sigmas_mS_cm",
    )
    corrector_sigmas = _validated_sigma_spread_tuple(
        evaluation.corrector_sigmas_mS_cm,
        "corrector_sigmas_mS_cm",
    )
    direct_capacity_gaps = _validated_sigma_tuple(
        evaluation.direct_capacity_gaps_mS_cm,
        "direct_capacity_gaps_mS_cm",
    )
    corrector_targets = _validated_sigma_spread_tuple(
        evaluation.corrector_targets_mS_cm,
        "corrector_targets_mS_cm",
    )
    corrector_residuals = _validated_sigma_tuple(
        evaluation.corrector_residuals_mS_cm,
        "corrector_residuals_mS_cm",
    )
    if len(direct_sigmas) != len(empirical_sigmas):
        raise ValueError("direct sigma tuple length must match sigma tuple length")
    if len(corrector_sigmas) != len(empirical_sigmas):
        raise ValueError("corrector sigma tuple length must match sigma tuple length")
    if len(direct_capacity_gaps) != len(empirical_sigmas):
        raise ValueError("direct capacity gap tuple length must match sigma tuple length")
    if len(corrector_targets) != len(empirical_sigmas):
        raise ValueError("corrector target tuple length must match sigma tuple length")
    if len(corrector_residuals) != len(empirical_sigmas):
        raise ValueError("corrector residual tuple length must match sigma tuple length")
    _nonnegative_int(
        evaluation.direct_capacity_failure_count,
        "direct_capacity_failure_count",
    )
    _nonnegative_int(
        evaluation.corrector_too_strong_failure_count,
        "corrector_too_strong_failure_count",
    )
    _nonnegative_int(
        evaluation.corrector_too_weak_failure_count,
        "corrector_too_weak_failure_count",
    )
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
    mean_huber_loss_mS_cm = _weighted_mean_huber_loss_mS_cm(
        residuals,
        residual_weights,
        options.huber_delta_mS_cm,
    )
    tail_huber_loss_mS_cm = _tail_huber_loss_mS_cm(
        residuals,
        residual_weights,
        options.huber_delta_mS_cm,
        options.residual_tail_count,
    )
    direct_capacity_loss_mS_cm = _direct_capacity_gap_loss_mS_cm(
        direct_capacity_gaps,
        residual_weights,
        options.huber_delta_mS_cm,
    )
    corrector_loss_mS_cm = _corrector_residual_loss_mS_cm(
        direct_capacity_gaps,
        corrector_residuals,
        residual_weights,
        options.huber_delta_mS_cm,
    )
    coordinate_regularization_loss = _coordinate_regularization_loss(
        bounded_coordinate_values,
        regularization_reference_coordinate_values,
        options.coordinate_regularization_weight,
    )
    cluster_activation_loss = _cluster_activation_loss(
        evaluation.cluster_activation_penalty,
        options.cluster_activation_loss_weight,
    )
    rejected = bool(rejection_reasons)
    objective_value = (
        math.inf
        if rejected
        else (
            mean_huber_loss_mS_cm
            + options.residual_tail_loss_weight * tail_huber_loss_mS_cm
            + options.direct_capacity_loss_weight * direct_capacity_loss_mS_cm
            + options.corrector_loss_weight * corrector_loss_mS_cm
            + coordinate_regularization_loss
            + cluster_activation_loss
        )
    )
    return PrimitiveFitCandidateResult(
        primitive_parameters=primitive_parameters,
        coordinate_values=bounded_coordinate_values,
        objective_value=objective_value,
        mean_huber_loss_mS_cm=mean_huber_loss_mS_cm,
        tail_huber_loss_mS_cm=tail_huber_loss_mS_cm,
        direct_capacity_loss_mS_cm=direct_capacity_loss_mS_cm,
        corrector_loss_mS_cm=corrector_loss_mS_cm,
        coordinate_regularization_loss=coordinate_regularization_loss,
        cluster_activation_loss=cluster_activation_loss,
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
    bounded_coordinate_values: tuple[float, ...],
    regularization_reference_coordinate_values: tuple[float, ...],
    options: PrimitiveFitOptions,
    error_type_name: str,
) -> PrimitiveFitCandidateResult:
    coordinate_regularization_loss = _coordinate_regularization_loss(
        bounded_coordinate_values,
        regularization_reference_coordinate_values,
        options.coordinate_regularization_weight,
    )
    return PrimitiveFitCandidateResult(
        primitive_parameters=primitive_parameters,
        coordinate_values=bounded_coordinate_values,
        objective_value=math.inf,
        mean_huber_loss_mS_cm=math.inf,
        tail_huber_loss_mS_cm=math.inf,
        direct_capacity_loss_mS_cm=math.inf,
        corrector_loss_mS_cm=math.inf,
        coordinate_regularization_loss=coordinate_regularization_loss,
        cluster_activation_loss=math.inf,
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
    regularization_reference_coordinate_values: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveParameterTransform, ...],
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitiveFitCandidateResult:
    if (
        options.powell_max_iterations_per_parameter == 0.0
        or options.powell_max_function_evaluations_per_parameter == 0.0
    ):
        return current_best
    powell_max_iterations = _fit_budget_count_from_parameter_count_allowing_zero(
        len(ordered_bounds),
        options.powell_max_iterations_per_parameter,
        "powell_max_iterations_per_parameter",
    )
    powell_max_function_evaluations = (
        _fit_budget_count_from_parameter_count_allowing_zero(
            len(ordered_bounds),
            options.powell_max_function_evaluations_per_parameter,
            "powell_max_function_evaluations_per_parameter",
        )
    )

    evaluation_cache: dict[tuple[float, ...], PrimitiveFitCandidateResult] = {}

    def _objective_for_coordinate_values(coordinate_value_array: np.ndarray) -> float:
        coordinate_values = tuple(
            float(coordinate_value) for coordinate_value in coordinate_value_array
        )
        bounded_coordinate_values = _bounded_initial_coordinate_values(coordinate_values, ordered_bounds)
        if bounded_coordinate_values not in evaluation_cache:
            candidate_result = evaluate_primitive_parameter_candidate(
                bounded_coordinate_values,
                initial_parameters,
                regularization_reference_coordinate_values,
                ordered_bounds,
                evaluator,
                options,
            )
            evaluation_cache[bounded_coordinate_values] = candidate_result
            candidate_results.append(candidate_result)
        return evaluation_cache[bounded_coordinate_values].objective_value

    lower_coordinate_values = np.asarray(
        [coordinate_bound.lower for coordinate_bound in ordered_bounds],
        dtype=float,
    )
    upper_coordinate_values = np.asarray(
        [coordinate_bound.upper for coordinate_bound in ordered_bounds],
        dtype=float,
    )
    minimize(
        _objective_for_coordinate_values,
        np.asarray(current_best.coordinate_values, dtype=float),
        method="Powell",
        bounds=Bounds(lower_coordinate_values, upper_coordinate_values),
        options={
            "maxiter": powell_max_iterations,
            "maxfev": powell_max_function_evaluations,
            "xtol": options.powell_xtol_coordinate,
            "ftol": options.powell_ftol_objective,
            "disp": False,
        },
    )
    for candidate_result in evaluation_cache.values():
        if _candidate_is_better(candidate_result, current_best):
            current_best = candidate_result
    return current_best


def _ordered_coordinate_bounds(
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
) -> tuple[PrimitiveParameterTransform, ...]:
    if not coordinate_bounds:
        raise ValueError("coordinate_bounds must contain at least one primitive parameter")
    bound_by_parameter_name: dict[str, PrimitiveParameterTransform] = {}
    for coordinate_bound in coordinate_bounds:
        if coordinate_bound.name in bound_by_parameter_name:
            raise ValueError(
                f"duplicate coordinate bound for {coordinate_bound.name}"
            )
        _validate_coordinate_bound(coordinate_bound)
        bound_by_parameter_name[coordinate_bound.name] = coordinate_bound
    ordered_bounds: list[PrimitiveParameterTransform] = []
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        if parameter_name in bound_by_parameter_name:
            ordered_bounds.append(bound_by_parameter_name[parameter_name])
    return tuple(ordered_bounds)


def _ordered_bound_parameter_names(
    ordered_bounds: tuple[PrimitiveParameterTransform, ...],
) -> tuple[str, ...]:
    if not ordered_bounds:
        raise ValueError("ordered_bounds must contain at least one primitive parameter")
    return tuple(coordinate_bound.name for coordinate_bound in ordered_bounds)


def _coordinate_bounds_for_parameter_names(
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    parameter_names: tuple[str, ...],
) -> tuple[PrimitiveParameterTransform, ...]:
    requested_parameter_names = set(parameter_names)
    selected_bounds = tuple(
        coordinate_bound for coordinate_bound in coordinate_bounds
        if coordinate_bound.name in requested_parameter_names
    )
    if not selected_bounds:
        raise ValueError("selected coordinate bounds must contain at least one parameter")
    return selected_bounds


def fit_speciation_from_cluster_sensitivities(
    cases: tuple["MolecularPropertyDbCase", ...],
    baseline_parameters: ConductivityPrimitiveParameterSet,
    baseline_audit_result: "MolecularPropertyDbAuditResult",
    audit_options: "MolecularPropertyDbAuditOptions",
    fit_options: PrimitiveFitOptions,
    speciation_coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    regularization_reference_parameters: ConductivityPrimitiveParameterSet,
) -> SpeciationSensitivityFitResult:
    from conductivity.molecular_property_db_audit import (
        cluster_sensitivity_diagnostics_for_row,
    )

    validate_conductivity_primitive_parameters(baseline_parameters)
    validate_conductivity_primitive_parameters(regularization_reference_parameters)
    _validate_fit_options(fit_options)
    ordered_speciation_bounds = _ordered_coordinate_bounds(speciation_coordinate_bounds)
    evaluator = MolecularPropertyDbPrimitiveEvaluator(
        cases,
        audit_options,
        fit_options,
    )
    sensitivity_coordinate_bounds = tuple(
        coordinate_bound
        for coordinate_bound in ordered_speciation_bounds
        if coordinate_bound.name in set(CLUSTER_SENSITIVITY_PARAMETER_NAMES)
    )
    if not sensitivity_coordinate_bounds:
        speciation_parameter_names = _ordered_bound_parameter_names(
            ordered_speciation_bounds
        )
        candidate_coordinate_values = (
            conductivity_primitive_parameter_coordinate_values_for_names(
                baseline_parameters,
                speciation_parameter_names,
            )
        )
        regularization_reference_coordinate_values = (
            conductivity_primitive_parameter_coordinate_values_for_names(
                regularization_reference_parameters,
                speciation_parameter_names,
            )
        )
        candidate = evaluate_primitive_parameter_candidate(
            candidate_coordinate_values,
            baseline_parameters,
            regularization_reference_coordinate_values,
            ordered_speciation_bounds,
            evaluator,
            fit_options,
        )
        return SpeciationSensitivityFitResult(
            candidate=candidate,
            sensitivity_row_count=0,
            sensitivity_entry_count=0,
        )
    sensitivity_parameter_names = _ordered_bound_parameter_names(
        sensitivity_coordinate_bounds
    )
    case_by_row_id = {molecular_case.row_id: molecular_case for molecular_case in cases}
    selected_rows = tuple(
        sorted(
            baseline_audit_result.rows,
            key=lambda row_result: abs(row_result.residual_mS_cm),
            reverse=True,
        )[:fit_options.residual_tail_count]
    )

    row_residuals: list[float] = []
    row_sensitivity_vectors: list[tuple[float, ...]] = []
    sensitivity_entry_count = 0
    for row_result in selected_rows:
        if row_result.row_id not in case_by_row_id:
            raise ValueError(f"missing molecular audit case for row {row_result.row_id}")
        row_coefficients_by_parameter = {
            parameter_name: 0.0
            for parameter_name in sensitivity_parameter_names
        }
        stoichiometry_by_cluster_label = {
            cluster_diagnostic.cluster_label: cluster_diagnostic.stoichiometry
            for cluster_diagnostic in row_result.cluster_thermodynamic_diagnostics
        }
        for sensitivity_diagnostic in cluster_sensitivity_diagnostics_for_row(
            case_by_row_id[row_result.row_id],
            baseline_parameters,
            audit_options,
            row_result,
        ):
            sensitivity_mS_cm_per_logK = _finite_float(
                sensitivity_diagnostic.sensitivity_mS_cm_per_logK,
                "sensitivity_mS_cm_per_logK",
            )
            if sensitivity_mS_cm_per_logK == 0.0:
                continue
            kind_parameter_name = _cluster_kind_logK_parameter_name(
                sensitivity_diagnostic.cluster_kind
            )
            if kind_parameter_name in row_coefficients_by_parameter:
                row_coefficients_by_parameter[kind_parameter_name] += (
                    sensitivity_mS_cm_per_logK
                )
                sensitivity_entry_count += 1
            cluster_order = _cluster_order_from_stoichiometry(
                stoichiometry_by_cluster_label[
                    sensitivity_diagnostic.cluster_label
                ],
                sensitivity_diagnostic.cluster_label,
            )
            cluster_order_feature = max(0, cluster_order - 2)
            if (
                cluster_order_feature > 0
                and "cluster_order_logK_slope" in row_coefficients_by_parameter
            ):
                row_coefficients_by_parameter["cluster_order_logK_slope"] += (
                    sensitivity_mS_cm_per_logK * cluster_order_feature
                )
                sensitivity_entry_count += 1
            cluster_charge_feature = abs(sensitivity_diagnostic.net_charge_number)
            if (
                cluster_charge_feature > 0
                and "cluster_charge_magnitude_logK_slope"
                in row_coefficients_by_parameter
            ):
                row_coefficients_by_parameter[
                    "cluster_charge_magnitude_logK_slope"
                ] += sensitivity_mS_cm_per_logK * cluster_charge_feature
                sensitivity_entry_count += 1
        row_sensitivity_vector = tuple(
            row_coefficients_by_parameter[parameter_name]
            for parameter_name in sensitivity_parameter_names
        )
        if any(coefficient != 0.0 for coefficient in row_sensitivity_vector):
            row_residuals.append(row_result.residual_mS_cm)
            row_sensitivity_vectors.append(row_sensitivity_vector)

    if not row_sensitivity_vectors:
        candidate_coordinate_values = conductivity_primitive_parameter_coordinate_values_for_names(
            baseline_parameters,
            _ordered_bound_parameter_names(ordered_speciation_bounds),
        )
        regularization_reference_coordinate_values = (
            conductivity_primitive_parameter_coordinate_values_for_names(
                regularization_reference_parameters,
                _ordered_bound_parameter_names(ordered_speciation_bounds),
            )
        )
        candidate = evaluate_primitive_parameter_candidate(
            candidate_coordinate_values,
            baseline_parameters,
            regularization_reference_coordinate_values,
            ordered_speciation_bounds,
            evaluator,
            fit_options,
        )
        return SpeciationSensitivityFitResult(
            candidate=candidate,
            sensitivity_row_count=0,
            sensitivity_entry_count=0,
        )

    current_coordinate_values = conductivity_primitive_parameter_coordinate_values_for_names(
        baseline_parameters,
        sensitivity_parameter_names,
    )
    lower_coordinate_values = tuple(
        coordinate_bound.lower
        for coordinate_bound in sensitivity_coordinate_bounds
    )
    upper_coordinate_values = tuple(
        coordinate_bound.upper
        for coordinate_bound in sensitivity_coordinate_bounds
    )
    updated_coordinate_values = solve_bounded_huber_ridge_coordinate_values(
        residuals_mS_cm=tuple(row_residuals),
        sensitivities_mS_cm_per_coordinate=tuple(row_sensitivity_vectors),
        current_coordinate_values=current_coordinate_values,
        lower_coordinate_values=lower_coordinate_values,
        upper_coordinate_values=upper_coordinate_values,
        huber_delta_mS_cm=fit_options.huber_delta_mS_cm,
        ridge_weight=fit_options.coordinate_regularization_weight,
    )
    sensitivity_parameters = conductivity_primitive_parameters_with_coordinate_updates(
        baseline_parameters,
        sensitivity_parameter_names,
        updated_coordinate_values,
    )
    speciation_parameter_names = _ordered_bound_parameter_names(
        ordered_speciation_bounds
    )
    candidate_coordinate_values = conductivity_primitive_parameter_coordinate_values_for_names(
        sensitivity_parameters,
        speciation_parameter_names,
    )
    regularization_reference_coordinate_values = (
        conductivity_primitive_parameter_coordinate_values_for_names(
            regularization_reference_parameters,
            speciation_parameter_names,
        )
    )
    candidate = evaluate_primitive_parameter_candidate(
        candidate_coordinate_values,
        baseline_parameters,
        regularization_reference_coordinate_values,
        ordered_speciation_bounds,
        evaluator,
        fit_options,
    )
    return SpeciationSensitivityFitResult(
        candidate=candidate,
        sensitivity_row_count=len(row_sensitivity_vectors),
        sensitivity_entry_count=sensitivity_entry_count,
    )


def solve_bounded_huber_ridge_coordinate_values(
    residuals_mS_cm: tuple[float, ...],
    sensitivities_mS_cm_per_coordinate: tuple[tuple[float, ...], ...],
    current_coordinate_values: tuple[float, ...],
    lower_coordinate_values: tuple[float, ...],
    upper_coordinate_values: tuple[float, ...],
    huber_delta_mS_cm: float,
    ridge_weight: float,
) -> tuple[float, ...]:
    if not residuals_mS_cm:
        raise ValueError("residuals_mS_cm must be nonempty")
    if len(residuals_mS_cm) != len(sensitivities_mS_cm_per_coordinate):
        raise ValueError("residual and sensitivity row counts must match")
    parameter_count = len(current_coordinate_values)
    if parameter_count == 0:
        raise ValueError("current_coordinate_values must be nonempty")
    if (
        len(lower_coordinate_values) != parameter_count
        or len(upper_coordinate_values) != parameter_count
    ):
        raise ValueError("coordinate bound counts must match parameter count")
    parsed_current_coordinate_values = np.asarray(
        tuple(
            _finite_float(coordinate_value, "current_coordinate_value")
            for coordinate_value in current_coordinate_values
        ),
        dtype=float,
    )
    parsed_lower_coordinate_values = np.asarray(
        tuple(
            _finite_float(coordinate_value, "lower_coordinate_value")
            for coordinate_value in lower_coordinate_values
        ),
        dtype=float,
    )
    parsed_upper_coordinate_values = np.asarray(
        tuple(
            _finite_float(coordinate_value, "upper_coordinate_value")
            for coordinate_value in upper_coordinate_values
        ),
        dtype=float,
    )
    if np.any(parsed_lower_coordinate_values >= parsed_upper_coordinate_values):
        raise ValueError(
            "each lower coordinate value must be below its upper coordinate value"
        )
    parsed_current_coordinate_values = np.minimum(
        parsed_upper_coordinate_values,
        np.maximum(parsed_lower_coordinate_values, parsed_current_coordinate_values),
    )
    residual_vector = np.asarray(
        tuple(
            _finite_float(residual_mS_cm, "residual_mS_cm")
            for residual_mS_cm in residuals_mS_cm
        ),
        dtype=float,
    )
    sensitivity_matrix = np.asarray(
        tuple(
            tuple(
                _finite_float(
                    sensitivity_value,
                    "sensitivity_mS_cm_per_coordinate",
                )
                for sensitivity_value in sensitivity_row
            )
            for sensitivity_row in sensitivities_mS_cm_per_coordinate
        ),
        dtype=float,
    )
    if sensitivity_matrix.shape != (len(residual_vector), parameter_count):
        raise ValueError("sensitivity matrix shape must match residual and parameter counts")
    huber_delta = _positive_float(huber_delta_mS_cm, "huber_delta_mS_cm")
    regularization_weight = _nonnegative_float(ridge_weight, "ridge_weight")
    abs_residuals = np.abs(residual_vector)
    huber_weights = np.ones_like(abs_residuals)
    tail_mask = abs_residuals > huber_delta
    huber_weights[tail_mask] = huber_delta / abs_residuals[tail_mask]
    sqrt_huber_weights = np.sqrt(huber_weights)
    weighted_sensitivity_matrix = (
        sqrt_huber_weights[:, None] * sensitivity_matrix
    )
    weighted_target_vector = -sqrt_huber_weights * residual_vector
    if regularization_weight > 0.0:
        weighted_sensitivity_matrix = np.vstack(
            (
                weighted_sensitivity_matrix,
                math.sqrt(regularization_weight) * np.eye(parameter_count),
            )
        )
        weighted_target_vector = np.concatenate(
            (
                weighted_target_vector,
                np.zeros(parameter_count, dtype=float),
            )
        )
    coordinate_delta_values = np.linalg.lstsq(
        weighted_sensitivity_matrix,
        weighted_target_vector,
        rcond=None,
    )[0]
    updated_coordinate_values = parsed_current_coordinate_values + coordinate_delta_values
    bounded_coordinate_values = np.minimum(
        parsed_upper_coordinate_values,
        np.maximum(parsed_lower_coordinate_values, updated_coordinate_values),
    )
    return tuple(
        float(coordinate_value) for coordinate_value in bounded_coordinate_values
    )


def _cluster_kind_logK_parameter_name(cluster_kind: str) -> str:
    if cluster_kind not in _CLUSTER_KIND_LOGK_PARAMETER_BY_KIND:
        raise ValueError(f"unknown cluster kind {cluster_kind}")
    return _CLUSTER_KIND_LOGK_PARAMETER_BY_KIND[cluster_kind]


def _cluster_order_from_stoichiometry(
    stoichiometry: Mapping[str, int],
    cluster_label: str,
) -> int:
    if not stoichiometry:
        raise ValueError(f"cluster {cluster_label} has empty stoichiometry")
    cluster_order = 0
    for species_label, stoichiometric_count in stoichiometry.items():
        if not isinstance(stoichiometric_count, int):
            raise TypeError(
                f"cluster {cluster_label} stoichiometry for {species_label} "
                "must be an integer"
            )
        if stoichiometric_count <= 0:
            raise ValueError(
                f"cluster {cluster_label} stoichiometry for {species_label} "
                "must be positive"
            )
        cluster_order += stoichiometric_count
    return cluster_order


def _validate_coordinate_bound(coordinate_bound: PrimitiveParameterTransform) -> None:
    if coordinate_bound.name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        raise ValueError(f"unknown primitive parameter {coordinate_bound.name}")
    expected_transform = CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[
        coordinate_bound.name
    ]
    if coordinate_bound.transform != expected_transform:
        raise ValueError(
            f"{coordinate_bound.name} transform must be {expected_transform}"
        )
    if coordinate_bound.transform not in (
        PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE,
        PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED,
    ):
        raise ValueError(f"{coordinate_bound.name} has unknown transform")
    lower_coordinate_value = _finite_float(
        coordinate_bound.lower,
        f"{coordinate_bound.name}.lower",
    )
    upper_coordinate_value = _finite_float(
        coordinate_bound.upper,
        f"{coordinate_bound.name}.upper",
    )
    if lower_coordinate_value >= upper_coordinate_value:
        raise ValueError(f"{coordinate_bound.name} lower bound must be below upper")


def _validate_fit_options(options: PrimitiveFitOptions) -> None:
    _positive_float(options.huber_delta_mS_cm, "huber_delta_mS_cm")
    _positive_float(
        options.empirical_sigma_floor_mS_cm,
        "empirical_sigma_floor_mS_cm",
    )
    _nonnegative_float(options.coordinate_regularization_weight, "coordinate_regularization_weight")
    _nonnegative_float(
        options.residual_tail_loss_weight,
        "residual_tail_loss_weight",
    )
    _positive_int(options.residual_tail_count, "residual_tail_count")
    _nonnegative_float(
        options.cluster_activation_loss_weight,
        "cluster_activation_loss_weight",
    )
    _positive_float(
        options.cluster_activation_residual_threshold_mS_cm,
        "cluster_activation_residual_threshold_mS_cm",
    )
    _positive_float(
        options.cluster_activation_min_charged_cluster_fraction,
        "cluster_activation_min_charged_cluster_fraction",
    )
    _positive_float(
        options.cluster_activation_min_charged_cluster_net_sigma_mS_cm,
        "cluster_activation_min_charged_cluster_net_sigma_mS_cm",
    )
    _nonnegative_float(
        options.direct_capacity_loss_weight,
        "direct_capacity_loss_weight",
    )
    _nonnegative_float(
        options.corrector_loss_weight,
        "corrector_loss_weight",
    )
    _nonnegative_float(
        options.role_direct_scaling_regularization_weight,
        "role_direct_scaling_regularization_weight",
    )
    role_direct_scaling_lower_bound = _positive_float(
        options.role_direct_scaling_lower_bound,
        "role_direct_scaling_lower_bound",
    )
    role_direct_scaling_upper_bound = _positive_float(
        options.role_direct_scaling_upper_bound,
        "role_direct_scaling_upper_bound",
    )
    if role_direct_scaling_lower_bound >= role_direct_scaling_upper_bound:
        raise ValueError(
            "role_direct_scaling_lower_bound must be below "
            "role_direct_scaling_upper_bound"
        )
    _positive_float(
        options.latin_hypercube_samples_per_parameter,
        "latin_hypercube_samples_per_parameter",
    )
    _nonnegative_int(options.coordinate_search_rounds, "coordinate_search_rounds")
    _positive_float(options.initial_coordinate_step, "initial_coordinate_step")
    coordinate_step_shrinkage = _positive_float(
        options.coordinate_step_shrinkage,
        "coordinate_step_shrinkage",
    )
    if coordinate_step_shrinkage >= 1.0:
        raise ValueError("coordinate_step_shrinkage must be below 1")
    _positive_float(options.minimum_coordinate_step, "minimum_coordinate_step")
    _nonnegative_float(
        options.powell_max_iterations_per_parameter,
        "powell_max_iterations_per_parameter",
    )
    _nonnegative_float(
        options.powell_max_function_evaluations_per_parameter,
        "powell_max_function_evaluations_per_parameter",
    )
    _nonnegative_float(
        options.decomposed_block_powell_max_iterations_per_parameter,
        "decomposed_block_powell_max_iterations_per_parameter",
    )
    _nonnegative_float(
        options.decomposed_block_powell_max_function_evaluations_per_parameter,
        "decomposed_block_powell_max_function_evaluations_per_parameter",
    )
    _nonnegative_float(
        options.decomposed_block_cluster_activation_loss_weight,
        "decomposed_block_cluster_activation_loss_weight",
    )
    _positive_float(options.powell_xtol_coordinate, "powell_xtol_coordinate")
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
    high_correlation_threshold = _positive_float(
        options.descriptor_matrix_high_correlation_threshold,
        "descriptor_matrix_high_correlation_threshold",
    )
    if high_correlation_threshold >= 1.0:
        raise ValueError("descriptor_matrix_high_correlation_threshold must be below 1")
    _positive_float(
        options.descriptor_matrix_condition_number_warn_threshold,
        "descriptor_matrix_condition_number_warn_threshold",
    )
    _nonnegative_int(
        options.descriptor_matrix_reported_correlation_pair_count,
        "descriptor_matrix_reported_correlation_pair_count",
    )
    _positive_float(
        options.prediction_sensitivity_coordinate_step,
        "prediction_sensitivity_coordinate_step",
    )
    _positive_float(
        options.prediction_sensitivity_min_column_norm_mS_cm_per_coordinate,
        "prediction_sensitivity_min_column_norm_mS_cm_per_coordinate",
    )
    relative_singular_value_threshold = _positive_float(
        options.prediction_sensitivity_relative_singular_value_threshold,
        "prediction_sensitivity_relative_singular_value_threshold",
    )
    if relative_singular_value_threshold >= 1.0:
        raise ValueError(
            "prediction_sensitivity_relative_singular_value_threshold "
            "must be below 1"
        )
    prediction_correlation_threshold = _positive_float(
        options.prediction_sensitivity_high_correlation_threshold,
        "prediction_sensitivity_high_correlation_threshold",
    )
    if prediction_correlation_threshold >= 1.0:
        raise ValueError(
            "prediction_sensitivity_high_correlation_threshold must be below 1"
        )
    _nonnegative_int(
        options.prediction_sensitivity_reported_correlation_pair_count,
        "prediction_sensitivity_reported_correlation_pair_count",
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


def _bounded_initial_coordinate_values(
    coordinate_values: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveParameterTransform, ...],
) -> tuple[float, ...]:
    if len(coordinate_values) != len(ordered_bounds):
        raise ValueError("coordinate value count must match ordered bound count")
    bounded_values: list[float] = []
    for coordinate_value, coordinate_bound in zip(coordinate_values, ordered_bounds):
        parsed_coordinate_value = _finite_float(
            coordinate_value,
            coordinate_bound.name,
        )
        if parsed_coordinate_value < coordinate_bound.lower:
            raise ValueError(
                f"{coordinate_bound.name} is below its lower coordinate bound"
            )
        if parsed_coordinate_value > coordinate_bound.upper:
            raise ValueError(
                f"{coordinate_bound.name} is above its upper coordinate bound"
            )
        bounded_values.append(parsed_coordinate_value)
    return tuple(bounded_values)


def _fit_budget_count_from_parameter_count(
    parameter_count: int,
    evaluations_per_parameter: float,
    context: str,
) -> int:
    budget_count = _fit_budget_count_from_parameter_count_allowing_zero(
        parameter_count,
        evaluations_per_parameter,
        context,
    )
    if budget_count <= 0:
        raise ValueError(f"{context} produces zero evaluations")
    return budget_count


def _fit_budget_count_from_parameter_count_allowing_zero(
    parameter_count: int,
    evaluations_per_parameter: float,
    context: str,
) -> int:
    parsed_parameter_count = _positive_int(parameter_count, "parameter_count")
    parsed_evaluations_per_parameter = _nonnegative_float(
        evaluations_per_parameter,
        context,
    )
    return int(math.ceil(parsed_parameter_count * parsed_evaluations_per_parameter))


def _latin_hypercube_coordinate_values(
    ordered_bounds: tuple[PrimitiveParameterTransform, ...],
    sample_count: int,
    random_number_generator: random.Random,
) -> tuple[tuple[float, ...], ...]:
    _positive_int(sample_count, "latin_hypercube_derived_sample_count")
    per_parameter_values: list[list[float]] = []
    for coordinate_bound in ordered_bounds:
        parameter_values: list[float] = []
        coordinate_span = (
            coordinate_bound.upper
            - coordinate_bound.lower
        )
        for sample_index in range(sample_count):
            unit_interval_value = (
                sample_index + random_number_generator.random()
            ) / sample_count
            parameter_values.append(
                coordinate_bound.lower
                + unit_interval_value * coordinate_span
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


def _coordinate_trial_coordinate_values(
    current_coordinate_values: tuple[float, ...],
    parameter_index: int,
    step_coordinate_value: float,
    coordinate_bound: PrimitiveParameterTransform,
) -> tuple[float, ...]:
    trial_values = list(current_coordinate_values)
    trial_value = current_coordinate_values[parameter_index] + step_coordinate_value
    if trial_value < coordinate_bound.lower:
        trial_value = coordinate_bound.lower
    if trial_value > coordinate_bound.upper:
        trial_value = coordinate_bound.upper
    trial_values[parameter_index] = trial_value
    return tuple(trial_values)


def _candidate_rejection_reasons(
    evaluation: PrimitiveFitDatasetEvaluation,
    predicted_sigmas_mS_cm: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveParameterTransform, ...],
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


def _empirical_spread_residual_weights(
    empirical_sigma_spreads_mS_cm: tuple[float, ...],
    empirical_sigma_floor_mS_cm: float,
) -> tuple[float, ...]:
    if not empirical_sigma_spreads_mS_cm:
        raise ValueError("empirical_sigma_spreads_mS_cm must be nonempty")
    sigma_floor = _positive_float(
        empirical_sigma_floor_mS_cm,
        "empirical_sigma_floor_mS_cm",
    )
    raw_weights = tuple(
        1.0
        / (
            sigma_floor * sigma_floor
            + _nonnegative_float(
                empirical_sigma_spread_mS_cm,
                "empirical_sigma_spread_mS_cm",
            )
            * _nonnegative_float(
                empirical_sigma_spread_mS_cm,
                "empirical_sigma_spread_mS_cm",
            )
        )
        for empirical_sigma_spread_mS_cm in empirical_sigma_spreads_mS_cm
    )
    mean_weight = math.fsum(raw_weights) / len(raw_weights)
    _positive_float(mean_weight, "mean_empirical_spread_weight")
    return tuple(raw_weight / mean_weight for raw_weight in raw_weights)


def _weighted_mean_huber_loss_mS_cm(
    residuals_mS_cm: tuple[float, ...],
    residual_weights: tuple[float, ...],
    huber_delta_mS_cm: float,
) -> float:
    if not residuals_mS_cm:
        raise ValueError("residuals_mS_cm must be nonempty")
    if len(residuals_mS_cm) != len(residual_weights):
        raise ValueError("residual and weight counts must match")
    weight_sum = math.fsum(
        _positive_float(residual_weight, "residual_weight")
        for residual_weight in residual_weights
    )
    _positive_float(weight_sum, "residual_weight_sum")
    return float(
        math.fsum(
            _positive_float(residual_weight, "residual_weight")
            * _smooth_l1_loss_mS_cm(residual, huber_delta_mS_cm)
            for residual, residual_weight in zip(residuals_mS_cm, residual_weights)
        )
        / weight_sum
    )


def _tail_huber_loss_mS_cm(
    residuals_mS_cm: tuple[float, ...],
    residual_weights: tuple[float, ...],
    huber_delta_mS_cm: float,
    residual_tail_count: int,
) -> float:
    if not residuals_mS_cm:
        raise ValueError("residuals_mS_cm must be nonempty")
    if len(residuals_mS_cm) != len(residual_weights):
        raise ValueError("residual and weight counts must match")
    tail_count = _positive_int(residual_tail_count, "residual_tail_count")
    residual_rankings: list[tuple[float, float, float]] = []
    for residual_mS_cm, residual_weight in zip(residuals_mS_cm, residual_weights):
        parsed_residual = _finite_float(residual_mS_cm, "residual_mS_cm")
        residual_rankings.append((
            abs(parsed_residual),
            parsed_residual,
            _positive_float(residual_weight, "residual_weight"),
        ))
    residual_rankings.sort(reverse=True)
    tail_residuals_with_weights = tuple(
        (parsed_residual, parsed_weight)
        for abs_residual, parsed_residual, parsed_weight in residual_rankings[:tail_count]
    )
    tail_weight_sum = math.fsum(
        parsed_weight
        for parsed_residual, parsed_weight in tail_residuals_with_weights
    )
    _positive_float(tail_weight_sum, "tail_residual_weight_sum")
    return float(
        math.fsum(
            parsed_weight
            * _smooth_l1_loss_mS_cm(parsed_residual, huber_delta_mS_cm)
            for parsed_residual, parsed_weight in tail_residuals_with_weights
        )
        / tail_weight_sum
    )


def _direct_capacity_gap_loss_mS_cm(
    direct_capacity_gaps_mS_cm: tuple[float, ...],
    residual_weights: tuple[float, ...],
    huber_delta_mS_cm: float,
) -> float:
    if len(direct_capacity_gaps_mS_cm) != len(residual_weights):
        raise ValueError("direct-capacity gap and weight counts must match")
    positive_direct_capacity_gaps = tuple(
        max(0.0, _finite_float(gap_mS_cm, "direct_capacity_gap_mS_cm"))
        for gap_mS_cm in direct_capacity_gaps_mS_cm
    )
    return _weighted_mean_huber_loss_mS_cm(
        positive_direct_capacity_gaps,
        residual_weights,
        huber_delta_mS_cm,
    )


def _corrector_residual_loss_mS_cm(
    direct_capacity_gaps_mS_cm: tuple[float, ...],
    corrector_residuals_mS_cm: tuple[float, ...],
    residual_weights: tuple[float, ...],
    huber_delta_mS_cm: float,
) -> float:
    if len(direct_capacity_gaps_mS_cm) != len(corrector_residuals_mS_cm):
        raise ValueError("direct-capacity gap and corrector residual counts must match")
    if len(corrector_residuals_mS_cm) != len(residual_weights):
        raise ValueError("corrector residual and weight counts must match")
    selected_residuals: list[float] = []
    selected_weights: list[float] = []
    for direct_capacity_gap_mS_cm, corrector_residual_mS_cm, residual_weight in zip(
        direct_capacity_gaps_mS_cm,
        corrector_residuals_mS_cm,
        residual_weights,
    ):
        parsed_gap_mS_cm = _finite_float(
            direct_capacity_gap_mS_cm,
            "direct_capacity_gap_mS_cm",
        )
        if parsed_gap_mS_cm > 0.0:
            continue
        selected_residuals.append(
            _finite_float(corrector_residual_mS_cm, "corrector_residual_mS_cm")
        )
        selected_weights.append(_positive_float(residual_weight, "residual_weight"))
    if not selected_residuals:
        return 0.0
    return _weighted_mean_huber_loss_mS_cm(
        tuple(selected_residuals),
        tuple(selected_weights),
        huber_delta_mS_cm,
    )


def _coordinate_regularization_loss(
    coordinate_values: tuple[float, ...],
    reference_coordinate_values: tuple[float, ...],
    coordinate_regularization_weight: float,
) -> float:
    if len(coordinate_values) != len(reference_coordinate_values):
        raise ValueError("regularization coordinate tuples must have equal length")
    regularization_weight = _nonnegative_float(
        coordinate_regularization_weight,
        "coordinate_regularization_weight",
    )
    squared_distance = math.fsum(
        (coordinate_value - reference_coordinate_value)
        * (coordinate_value - reference_coordinate_value)
        for coordinate_value, reference_coordinate_value in zip(
            coordinate_values,
            reference_coordinate_values,
        )
    )
    return float(regularization_weight * squared_distance)


def _cluster_activation_loss(
    cluster_activation_penalty: float,
    cluster_activation_loss_weight: float,
) -> float:
    penalty_value = _nonnegative_float(
        cluster_activation_penalty,
        "cluster_activation_penalty",
    )
    loss_weight = _nonnegative_float(
        cluster_activation_loss_weight,
        "cluster_activation_loss_weight",
    )
    return float(loss_weight * penalty_value)


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


def _validated_sigma_spread_tuple(
    sigma_spreads_mS_cm: tuple[float, ...],
    context: str,
) -> tuple[float, ...]:
    if not sigma_spreads_mS_cm:
        raise ValueError(f"{context} must be nonempty")
    return tuple(
        _nonnegative_float(sigma_spread_mS_cm, context)
        for sigma_spread_mS_cm in sigma_spreads_mS_cm
    )


def _validated_descriptor_calibration_targets(
    descriptor_calibration_targets: tuple[DescriptorCalibrationTarget, ...],
) -> tuple[DescriptorCalibrationTarget, ...]:
    if not descriptor_calibration_targets:
        raise ValueError("descriptor_calibration_targets must be nonempty")
    first_driver_values = _validated_descriptor_driver_values(
        descriptor_calibration_targets[0].descriptor_driver_values,
        descriptor_calibration_targets[0].target_id,
    )
    reference_driver_names = tuple(
        driver_name for driver_name, driver_value in first_driver_values
    )
    seen_target_ids: set[str] = set()
    validated_targets: list[DescriptorCalibrationTarget] = []
    for target_index, descriptor_target in enumerate(descriptor_calibration_targets):
        target_id = _nonempty_string(descriptor_target.target_id, "target_id")
        if target_id in seen_target_ids:
            raise ValueError(f"duplicate descriptor calibration target {target_id}")
        seen_target_ids.add(target_id)
        source_row_ids = tuple(
            _nonnegative_int(source_row_id, "source_row_id")
            for source_row_id in descriptor_target.source_row_ids
        )
        if not source_row_ids:
            raise ValueError(f"descriptor target {target_id} has no source rows")
        descriptor_driver_values = (
            first_driver_values
            if target_index == 0
            else _validated_descriptor_driver_values(
                descriptor_target.descriptor_driver_values,
                target_id,
            )
        )
        driver_names = tuple(
            driver_name for driver_name, driver_value in descriptor_driver_values
        )
        if driver_names != reference_driver_names:
            raise ValueError(
                "descriptor calibration targets must share identical driver columns"
            )
        validated_targets.append(
            DescriptorCalibrationTarget(
                target_id=target_id,
                source_row_ids=source_row_ids,
                descriptor_driver_values=descriptor_driver_values,
                empirical_sigma_mS_cm=_nonnegative_float(
                    descriptor_target.empirical_sigma_mS_cm,
                    f"{target_id}.empirical_sigma_mS_cm",
                ),
                empirical_sigma_spread_mS_cm=_nonnegative_float(
                    descriptor_target.empirical_sigma_spread_mS_cm,
                    f"{target_id}.empirical_sigma_spread_mS_cm",
                ),
                residual_weight=_positive_float(
                    descriptor_target.residual_weight,
                    f"{target_id}.residual_weight",
                ),
            )
        )
    return tuple(validated_targets)


def _validated_descriptor_driver_values(
    descriptor_driver_values: tuple[tuple[str, float], ...],
    target_id: str,
) -> tuple[tuple[str, float], ...]:
    target_id_text = _nonempty_string(target_id, "target_id")
    if not descriptor_driver_values:
        raise ValueError(f"descriptor target {target_id_text} has no driver values")
    seen_driver_names: set[str] = set()
    validated_values: list[tuple[str, float]] = []
    for driver_name, driver_value in descriptor_driver_values:
        driver_name_text = _nonempty_string(driver_name, "descriptor_driver_name")
        if driver_name_text in seen_driver_names:
            raise ValueError(
                f"descriptor target {target_id_text} duplicate driver {driver_name_text}"
            )
        seen_driver_names.add(driver_name_text)
        validated_values.append(
            (
                driver_name_text,
                _finite_float(
                    driver_value,
                    f"{target_id_text}.{driver_name_text}",
                ),
            )
        )
    return tuple(validated_values)


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


def primitive_driver_matrix_diagnostics(
    cases: tuple["MolecularPropertyDbCase", ...],
    options: PrimitiveFitOptions,
) -> PrimitiveDriverMatrixDiagnostics:
    if not cases:
        raise ValueError("primitive driver matrix diagnostics require cases")
    _validate_fit_options(options)
    descriptor_targets = descriptor_calibration_targets_for_cases(cases, options)
    return primitive_driver_matrix_diagnostics_for_targets(
        descriptor_targets,
        options,
    )


def descriptor_calibration_targets_for_cases(
    cases: tuple["MolecularPropertyDbCase", ...],
    options: PrimitiveFitOptions,
) -> tuple[DescriptorCalibrationTarget, ...]:
    if not cases:
        raise ValueError("descriptor calibration targets require cases")
    _validate_fit_options(options)
    empirical_sigma_spreads = tuple(
        _nonnegative_float(
            molecular_case.empirical_sigma_spread_mS_cm,
            "empirical_sigma_spread_mS_cm",
        )
        for molecular_case in cases
    )
    residual_weights = _empirical_spread_residual_weights(
        empirical_sigma_spreads,
        options.empirical_sigma_floor_mS_cm,
    )
    descriptor_targets = tuple(
        DescriptorCalibrationTarget(
            target_id=f"formulation_group:{molecular_case.row_id}",
            source_row_ids=tuple(
                _nonnegative_int(source_row_id, "source_row_id")
                for source_row_id in molecular_case.source_row_ids
            ),
            descriptor_driver_values=_primitive_driver_feature_row(molecular_case),
            empirical_sigma_mS_cm=_nonnegative_float(
                molecular_case.empirical_sigma_mS_cm,
                "empirical_sigma_mS_cm",
            ),
            empirical_sigma_spread_mS_cm=empirical_sigma_spread_mS_cm,
            residual_weight=residual_weight,
        )
        for molecular_case, empirical_sigma_spread_mS_cm, residual_weight in zip(
            cases,
            empirical_sigma_spreads,
            residual_weights,
        )
    )
    return _validated_descriptor_calibration_targets(descriptor_targets)


def primitive_driver_matrix_diagnostics_for_targets(
    descriptor_calibration_targets: tuple[DescriptorCalibrationTarget, ...],
    options: PrimitiveFitOptions,
) -> PrimitiveDriverMatrixDiagnostics:
    descriptor_targets = _validated_descriptor_calibration_targets(
        descriptor_calibration_targets,
    )
    feature_rows = tuple(
        descriptor_target.descriptor_driver_values
        for descriptor_target in descriptor_targets
    )
    return primitive_driver_matrix_diagnostics_from_feature_rows(
        feature_rows,
        options,
    )


def primitive_driver_matrix_diagnostics_from_feature_rows(
    feature_rows: tuple[tuple[tuple[str, float], ...], ...],
    options: PrimitiveFitOptions,
) -> PrimitiveDriverMatrixDiagnostics:
    if not feature_rows:
        raise ValueError("primitive driver matrix diagnostics require feature rows")
    _validate_fit_options(options)
    feature_names = tuple(
        feature_name for feature_name, feature_value in feature_rows[0]
    )
    for feature_row in feature_rows:
        row_feature_names = tuple(
            feature_name for feature_name, feature_value in feature_row
        )
        if row_feature_names != feature_names:
            raise ValueError(
                "primitive driver feature names must be identical across rows"
            )
    driver_matrix = np.asarray(
        [
            [feature_value for feature_name, feature_value in feature_row]
            for feature_row in feature_rows
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(driver_matrix)):
        raise ValueError("primitive driver matrix must be finite")
    return _primitive_driver_matrix_diagnostics_from_matrix(
        driver_matrix,
        feature_names,
        options,
    )


def _primitive_driver_matrix_diagnostics_from_matrix(
    driver_matrix: np.ndarray,
    feature_names: tuple[str, ...],
    options: PrimitiveFitOptions,
) -> PrimitiveDriverMatrixDiagnostics:
    column_standard_deviations = np.std(driver_matrix, axis=0)
    zero_variance_columns = tuple(
        feature_name
        for feature_name, column_standard_deviation in zip(
            feature_names,
            column_standard_deviations,
        )
        if column_standard_deviation <= 0.0
    )
    active_column_indices = tuple(
        column_index
        for column_index, column_standard_deviation in enumerate(
            column_standard_deviations
        )
        if column_standard_deviation > 0.0
    )
    if not active_column_indices:
        raise ValueError("primitive driver matrix has no varying columns")
    active_matrix = driver_matrix[:, active_column_indices]
    centered_active_matrix = active_matrix - np.mean(active_matrix, axis=0)
    scaled_active_matrix = centered_active_matrix / np.std(active_matrix, axis=0)
    singular_values = np.linalg.svd(scaled_active_matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled_active_matrix))
    minimum_singular_value = float(np.min(singular_values))
    maximum_singular_value = float(np.max(singular_values))
    if minimum_singular_value <= np.finfo(float).eps:
        condition_number = math.inf
    else:
        condition_number = float(maximum_singular_value / minimum_singular_value)
    high_correlation_pairs = _primitive_driver_high_correlation_pairs(
        scaled_active_matrix,
        tuple(feature_names[column_index] for column_index in active_column_indices),
        options.descriptor_matrix_high_correlation_threshold,
    )
    return PrimitiveDriverMatrixDiagnostics(
        row_count=int(driver_matrix.shape[0]),
        column_count=len(feature_names),
        rank=rank,
        condition_number=condition_number,
        zero_variance_columns=zero_variance_columns,
        high_correlation_pairs=high_correlation_pairs,
    )


def _primitive_driver_high_correlation_pairs(
    scaled_active_matrix: np.ndarray,
    active_feature_names: tuple[str, ...],
    threshold: float,
) -> tuple[tuple[str, str, float], ...]:
    return _high_correlation_pairs_from_scaled_matrix(
        scaled_active_matrix,
        active_feature_names,
        threshold,
        "descriptor_matrix_high_correlation_threshold",
    )


def primitive_prediction_sensitivity_diagnostics(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitivePredictionSensitivityDiagnostics:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_fit_options(options)
    ordered_bounds = _ordered_coordinate_bounds(coordinate_bounds)
    parameter_names = _ordered_bound_parameter_names(ordered_bounds)
    baseline_evaluation = evaluator.evaluate(primitive_parameters)
    _validate_prediction_sensitivity_evaluation(
        baseline_evaluation,
        options,
        "baseline_prediction_sensitivity",
    )
    baseline_predicted_sigmas = _validated_sigma_tuple(
        baseline_evaluation.predicted_sigmas_mS_cm,
        "baseline_predicted_sigmas_mS_cm",
    )
    baseline_descriptor_targets = _validated_descriptor_calibration_targets(
        baseline_evaluation.descriptor_calibration_targets,
    )
    empirical_sigma_spreads = tuple(
        descriptor_target.empirical_sigma_spread_mS_cm
        for descriptor_target in baseline_descriptor_targets
    )
    if len(baseline_predicted_sigmas) != len(empirical_sigma_spreads):
        raise ValueError(
            "baseline predicted sigma and empirical spread counts must match"
        )
    residual_weights = tuple(
        descriptor_target.residual_weight
        for descriptor_target in baseline_descriptor_targets
    )
    current_coordinate_values = (
        conductivity_primitive_parameter_coordinate_values_for_names(
            primitive_parameters,
            parameter_names,
        )
    )
    sensitivity_columns: list[tuple[float, ...]] = []
    invalid_trial_parameter_names: list[str] = []
    coordinate_step = _positive_float(
        options.prediction_sensitivity_coordinate_step,
        "prediction_sensitivity_coordinate_step",
    )
    for parameter_index, coordinate_bound in enumerate(ordered_bounds):
        minus_coordinate_value, plus_coordinate_value = (
            _prediction_sensitivity_coordinate_pair(
                current_coordinate_values[parameter_index],
                coordinate_bound,
                coordinate_step,
            )
        )
        if minus_coordinate_value == plus_coordinate_value:
            sensitivity_columns.append(
                tuple(0.0 for baseline_sigma in baseline_predicted_sigmas)
            )
            continue
        minus_trial = _prediction_sensitivity_trial(
            primitive_parameters,
            parameter_names,
            current_coordinate_values,
            parameter_index,
            minus_coordinate_value,
            evaluator,
            options,
        )
        plus_trial = _prediction_sensitivity_trial(
            primitive_parameters,
            parameter_names,
            current_coordinate_values,
            parameter_index,
            plus_coordinate_value,
            evaluator,
            options,
        )
        if not minus_trial.valid or not plus_trial.valid:
            invalid_trial_parameter_names.append(coordinate_bound.name)
        _validate_prediction_sensitivity_trial_shape(
            minus_trial,
            baseline_predicted_sigmas,
            "minus_prediction_sensitivity",
        )
        _validate_prediction_sensitivity_trial_shape(
            plus_trial,
            baseline_predicted_sigmas,
            "plus_prediction_sensitivity",
        )
        if minus_trial.valid and plus_trial.valid:
            coordinate_delta = plus_coordinate_value - minus_coordinate_value
            _positive_float(
                coordinate_delta,
                "prediction_sensitivity_coordinate_delta",
            )
            sensitivity_columns.append(
                tuple(
                    (plus_sigma_mS_cm - minus_sigma_mS_cm) / coordinate_delta
                    for minus_sigma_mS_cm, plus_sigma_mS_cm in zip(
                        minus_trial.predicted_sigmas_mS_cm,
                        plus_trial.predicted_sigmas_mS_cm,
                    )
                )
            )
            continue
        if plus_trial.valid:
            coordinate_delta = plus_coordinate_value - current_coordinate_values[
                parameter_index
            ]
            _positive_float(
                coordinate_delta,
                "prediction_sensitivity_forward_coordinate_delta",
            )
            sensitivity_columns.append(
                tuple(
                    (plus_sigma_mS_cm - baseline_sigma_mS_cm) / coordinate_delta
                    for baseline_sigma_mS_cm, plus_sigma_mS_cm in zip(
                        baseline_predicted_sigmas,
                        plus_trial.predicted_sigmas_mS_cm,
                    )
                )
            )
            continue
        if minus_trial.valid:
            coordinate_delta = current_coordinate_values[
                parameter_index
            ] - minus_coordinate_value
            _positive_float(
                coordinate_delta,
                "prediction_sensitivity_backward_coordinate_delta",
            )
            sensitivity_columns.append(
                tuple(
                    (baseline_sigma_mS_cm - minus_sigma_mS_cm) / coordinate_delta
                    for baseline_sigma_mS_cm, minus_sigma_mS_cm in zip(
                        baseline_predicted_sigmas,
                        minus_trial.predicted_sigmas_mS_cm,
                    )
                )
            )
            continue
        sensitivity_columns.append(
            tuple(0.0 for baseline_sigma in baseline_predicted_sigmas)
        )
    sensitivity_matrix = np.asarray(sensitivity_columns, dtype=float).T
    if not np.all(np.isfinite(sensitivity_matrix)):
        raise ValueError("prediction sensitivity matrix must be finite")
    weighted_sensitivity_matrix = (
        np.sqrt(np.asarray(residual_weights, dtype=float))[:, None]
        * sensitivity_matrix
    )
    return _primitive_prediction_sensitivity_diagnostics_from_matrix(
        weighted_sensitivity_matrix,
        parameter_names,
        tuple(invalid_trial_parameter_names),
        options,
    )


def _prediction_sensitivity_coordinate_pair(
    current_coordinate_value: float,
    coordinate_bound: PrimitiveParameterTransform,
    coordinate_step: float,
) -> tuple[float, float]:
    current_coordinate = _finite_float(
        current_coordinate_value,
        f"{coordinate_bound.name}.current_coordinate",
    )
    parsed_coordinate_step = _positive_float(
        coordinate_step,
        "prediction_sensitivity_coordinate_step",
    )
    minus_coordinate_value = max(
        coordinate_bound.lower,
        current_coordinate - parsed_coordinate_step,
    )
    plus_coordinate_value = min(
        coordinate_bound.upper,
        current_coordinate + parsed_coordinate_step,
    )
    if minus_coordinate_value > plus_coordinate_value:
        raise ValueError(
            f"{coordinate_bound.name} sensitivity coordinate pair is invalid"
        )
    return minus_coordinate_value, plus_coordinate_value


def _prediction_sensitivity_trial(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    parameter_names: tuple[str, ...],
    current_coordinate_values: tuple[float, ...],
    parameter_index: int,
    trial_coordinate_value: float,
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitivePredictionSensitivityTrial:
    trial_coordinate_values = list(current_coordinate_values)
    trial_coordinate_values[parameter_index] = _finite_float(
        trial_coordinate_value,
        "trial_coordinate_value",
    )
    trial_parameters = conductivity_primitive_parameters_with_coordinate_updates(
        primitive_parameters,
        parameter_names,
        tuple(trial_coordinate_values),
    )
    try:
        trial_evaluation = evaluator.evaluate(trial_parameters)
        _validate_prediction_sensitivity_evaluation(
            trial_evaluation,
            options,
            "trial_prediction_sensitivity",
        )
        predicted_sigmas_mS_cm = _validated_sigma_tuple(
            trial_evaluation.predicted_sigmas_mS_cm,
            "trial_predicted_sigmas_mS_cm",
        )
    except (
        FloatingPointError,
        OverflowError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return PrimitivePredictionSensitivityTrial(
            valid=False,
            predicted_sigmas_mS_cm=tuple(),
        )
    return PrimitivePredictionSensitivityTrial(
        valid=True,
        predicted_sigmas_mS_cm=predicted_sigmas_mS_cm,
    )


def _validate_prediction_sensitivity_trial_shape(
    trial: PrimitivePredictionSensitivityTrial,
    baseline_predicted_sigmas_mS_cm: tuple[float, ...],
    context: str,
) -> None:
    if not trial.valid:
        return
    if len(trial.predicted_sigmas_mS_cm) != len(baseline_predicted_sigmas_mS_cm):
        raise ValueError(f"{context} prediction count must match baseline")


def _validate_prediction_sensitivity_evaluation(
    evaluation: PrimitiveFitDatasetEvaluation,
    options: PrimitiveFitOptions,
    context: str,
) -> None:
    context_text = _nonempty_string(context, "prediction_sensitivity_context")
    predicted_sigmas = _validated_sigma_tuple(
        evaluation.predicted_sigmas_mS_cm,
        f"{context_text}.predicted_sigmas_mS_cm",
    )
    if min(predicted_sigmas) < 0.0:
        raise ValueError(f"{context_text} produced negative conductivity")
    if evaluation.failed_rows > options.maximum_failed_rows:
        raise ValueError(f"{context_text} produced failed rows")
    _threshold_or_raise(
        evaluation.maximum_mass_balance_residual,
        options.maximum_mass_balance_residual,
        f"{context_text}.mass_balance_residual",
    )
    _threshold_or_raise(
        evaluation.maximum_row_sum_residual,
        options.maximum_row_sum_residual,
        f"{context_text}.row_sum_residual",
    )
    _threshold_or_raise(
        evaluation.maximum_stationary_residual,
        options.maximum_stationary_residual,
        f"{context_text}.stationarity_residual",
    )
    _threshold_or_raise(
        evaluation.maximum_detailed_balance_residual,
        options.maximum_detailed_balance_residual,
        f"{context_text}.detailed_balance_residual",
    )
    _threshold_or_raise(
        evaluation.maximum_event_reversal_residual,
        options.maximum_event_reversal_residual,
        f"{context_text}.event_reversal_residual",
    )
    _threshold_or_raise(
        abs(evaluation.zero_charge_sigma_mS_cm),
        options.maximum_zero_charge_sigma_mS_cm,
        f"{context_text}.zero_charge_sigma",
    )
    if not evaluation.higher_viscosity_lowers_dilute_conductivity:
        raise ValueError(f"{context_text} violates viscosity monotonicity")
    if not evaluation.higher_packing_lowers_local_mobility:
        raise ValueError(f"{context_text} violates packing monotonicity")


def _threshold_or_raise(
    observed_value: float,
    maximum_allowed_value: float,
    context: str,
) -> None:
    observed = _nonnegative_float(observed_value, context)
    maximum_allowed = _nonnegative_float(maximum_allowed_value, f"{context}.maximum")
    if observed > maximum_allowed:
        raise ValueError(
            f"{context} {observed:.6e} exceeds maximum {maximum_allowed:.6e}"
        )


def _primitive_prediction_sensitivity_diagnostics_from_matrix(
    weighted_sensitivity_matrix: np.ndarray,
    parameter_names: tuple[str, ...],
    invalid_trial_parameter_names: tuple[str, ...],
    options: PrimitiveFitOptions,
) -> PrimitivePredictionSensitivityDiagnostics:
    if weighted_sensitivity_matrix.ndim != 2:
        raise ValueError("prediction sensitivity matrix must be two-dimensional")
    row_count = _positive_int(
        int(weighted_sensitivity_matrix.shape[0]),
        "prediction_sensitivity_row_count",
    )
    parameter_count = _positive_int(
        int(weighted_sensitivity_matrix.shape[1]),
        "prediction_sensitivity_parameter_count",
    )
    if parameter_count != len(parameter_names):
        raise ValueError(
            "prediction sensitivity matrix columns must match parameter names"
        )
    if not np.all(np.isfinite(weighted_sensitivity_matrix)):
        raise ValueError("weighted prediction sensitivity matrix must be finite")
    column_norms = np.linalg.norm(weighted_sensitivity_matrix, axis=0)
    min_column_norm = _positive_float(
        options.prediction_sensitivity_min_column_norm_mS_cm_per_coordinate,
        "prediction_sensitivity_min_column_norm_mS_cm_per_coordinate",
    )
    active_column_indices = tuple(
        column_index
        for column_index, column_norm in enumerate(column_norms)
        if column_norm > min_column_norm
    )
    zero_sensitivity_parameter_names = tuple(
        parameter_name
        for parameter_name, column_norm in zip(parameter_names, column_norms)
        if column_norm <= min_column_norm
    )
    if not active_column_indices:
        return PrimitivePredictionSensitivityDiagnostics(
            row_count=row_count,
            parameter_count=parameter_count,
            rank=0,
            condition_number=math.inf,
            singular_values=tuple(),
            identifiable_parameter_names=tuple(),
            frozen_parameter_names=parameter_names,
            zero_sensitivity_parameter_names=zero_sensitivity_parameter_names,
            invalid_trial_parameter_names=invalid_trial_parameter_names,
            high_correlation_parameter_pairs=tuple(),
        )
    active_matrix = weighted_sensitivity_matrix[:, active_column_indices]
    singular_values_array = np.linalg.svd(active_matrix, compute_uv=False)
    if not np.all(np.isfinite(singular_values_array)):
        raise ValueError("prediction sensitivity singular values must be finite")
    maximum_singular_value = float(np.max(singular_values_array))
    if maximum_singular_value <= 0.0:
        rank = 0
        condition_number = math.inf
    else:
        singular_threshold = (
            maximum_singular_value
            * options.prediction_sensitivity_relative_singular_value_threshold
        )
        rank = int(
            sum(
                1
                for singular_value in singular_values_array
                if singular_value >= singular_threshold
            )
        )
        minimum_singular_value = float(np.min(singular_values_array))
        if minimum_singular_value <= np.finfo(float).eps:
            condition_number = math.inf
        else:
            condition_number = float(maximum_singular_value / minimum_singular_value)
    identifiable_parameter_names = _identifiable_parameter_names_from_qr(
        active_matrix,
        tuple(parameter_names[index] for index in active_column_indices),
        rank,
    )
    identifiable_parameter_name_set = set(identifiable_parameter_names)
    frozen_parameter_names = tuple(
        parameter_name
        for parameter_name in parameter_names
        if parameter_name not in identifiable_parameter_name_set
    )
    scaled_active_matrix = (
        math.sqrt(float(row_count))
        * active_matrix
        / column_norms[np.asarray(active_column_indices, dtype=int)][None, :]
    )
    high_correlation_pairs = _high_correlation_pairs_from_scaled_matrix(
        scaled_active_matrix,
        tuple(parameter_names[index] for index in active_column_indices),
        options.prediction_sensitivity_high_correlation_threshold,
        "prediction_sensitivity_high_correlation_threshold",
    )
    return PrimitivePredictionSensitivityDiagnostics(
        row_count=row_count,
        parameter_count=parameter_count,
        rank=rank,
        condition_number=condition_number,
        singular_values=tuple(float(value) for value in singular_values_array),
        identifiable_parameter_names=identifiable_parameter_names,
        frozen_parameter_names=frozen_parameter_names,
        zero_sensitivity_parameter_names=zero_sensitivity_parameter_names,
        invalid_trial_parameter_names=invalid_trial_parameter_names,
        high_correlation_parameter_pairs=high_correlation_pairs,
    )


def _identifiable_parameter_names_from_qr(
    active_matrix: np.ndarray,
    active_parameter_names: tuple[str, ...],
    rank: int,
) -> tuple[str, ...]:
    parsed_rank = _nonnegative_int(rank, "prediction_sensitivity_rank")
    if parsed_rank == 0:
        return tuple()
    if parsed_rank > len(active_parameter_names):
        raise ValueError("prediction sensitivity rank exceeds active parameter count")
    if active_matrix.shape[1] != len(active_parameter_names):
        raise ValueError("active sensitivity matrix must match active parameters")
    _orthogonal_matrix, _upper_matrix, pivot_indices = qr(
        active_matrix,
        pivoting=True,
        mode="economic",
    )
    selected_parameter_names = {
        active_parameter_names[int(pivot_index)]
        for pivot_index in pivot_indices[:parsed_rank]
    }
    return tuple(
        parameter_name
        for parameter_name in active_parameter_names
        if parameter_name in selected_parameter_names
    )


def _coordinate_bounds_for_identifiable_parameters(
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    prediction_sensitivity_diagnostics: PrimitivePredictionSensitivityDiagnostics,
) -> tuple[PrimitiveParameterTransform, ...]:
    identifiable_parameter_name_set = set(
        prediction_sensitivity_diagnostics.identifiable_parameter_names
    )
    selected_coordinate_bounds = tuple(
        coordinate_bound
        for coordinate_bound in _ordered_coordinate_bounds(coordinate_bounds)
        if coordinate_bound.name in identifiable_parameter_name_set
    )
    if not selected_coordinate_bounds:
        raise ValueError("prediction sensitivity analysis found no identifiable parameters")
    return selected_coordinate_bounds


def _coordinate_bounds_for_stage_and_full_identifiable_parameters(
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    stage_prediction_sensitivity_diagnostics: PrimitivePredictionSensitivityDiagnostics,
    full_prediction_sensitivity_diagnostics: PrimitivePredictionSensitivityDiagnostics,
) -> tuple[PrimitiveParameterTransform, ...]:
    stage_identifiable_parameter_name_set = set(
        stage_prediction_sensitivity_diagnostics.identifiable_parameter_names
    )
    full_identifiable_parameter_name_set = set(
        full_prediction_sensitivity_diagnostics.identifiable_parameter_names
    )
    selected_parameter_name_set = (
        stage_identifiable_parameter_name_set
        & full_identifiable_parameter_name_set
    )
    selected_coordinate_bounds = tuple(
        coordinate_bound
        for coordinate_bound in _ordered_coordinate_bounds(coordinate_bounds)
        if coordinate_bound.name in selected_parameter_name_set
    )
    if not selected_coordinate_bounds:
        raise ValueError(
            "stage and full prediction sensitivity analyses found no common "
            "identifiable parameters"
        )
    return selected_coordinate_bounds


def _high_correlation_pairs_from_scaled_matrix(
    scaled_active_matrix: np.ndarray,
    active_feature_names: tuple[str, ...],
    threshold: float,
    threshold_context: str,
) -> tuple[tuple[str, str, float], ...]:
    correlation_threshold = _positive_float(threshold, threshold_context)
    if correlation_threshold >= 1.0:
        raise ValueError(f"{threshold_context} must be below 1")
    if scaled_active_matrix.ndim != 2:
        raise ValueError("scaled matrix must be two-dimensional")
    row_count = _positive_int(
        int(scaled_active_matrix.shape[0]),
        "scaled_matrix_row_count",
    )
    column_count = _positive_int(
        int(scaled_active_matrix.shape[1]),
        "scaled_matrix_column_count",
    )
    if column_count != len(active_feature_names):
        raise ValueError(
            "scaled matrix columns must match active feature names"
        )
    if column_count < 2:
        return tuple()
    if not np.all(np.isfinite(scaled_active_matrix)):
        raise ValueError("scaled matrix must be finite")
    correlation_matrix = (
        scaled_active_matrix.T @ scaled_active_matrix
    ) / float(row_count)
    if not np.all(np.isfinite(correlation_matrix)):
        raise ValueError("correlation matrix must be finite")
    high_correlation_pairs: list[tuple[str, str, float]] = []
    for first_index, first_feature_name in enumerate(active_feature_names):
        for second_index in range(first_index + 1, len(active_feature_names)):
            correlation_value = float(correlation_matrix[first_index, second_index])
            if abs(correlation_value) >= correlation_threshold:
                high_correlation_pairs.append(
                    (
                        first_feature_name,
                        active_feature_names[second_index],
                        correlation_value,
                    )
                )
    return tuple(high_correlation_pairs)


def _primitive_driver_feature_row(
    molecular_case: "MolecularPropertyDbCase",
) -> tuple[tuple[str, float], ...]:
    recipe = molecular_case.recipe
    mixture_properties = recipe.mixture_properties
    return (
        ("temperature_K", _finite_float(recipe.temperature_K, "temperature_K")),
        (
            "mixture_density_g_ml",
            _finite_float(
                mixture_properties.density_g_ml,
                "mixture_density_g_ml",
            ),
        ),
        (
            "mixture_viscosity_cP",
            _finite_float(
                mixture_properties.viscosity_cP,
                "mixture_viscosity_cP",
            ),
        ),
        (
            "mixture_dielectric_constant",
            _finite_float(
                mixture_properties.dielectric_constant,
                "mixture_dielectric_constant",
            ),
        ),
        ("total_cation_molarity_M", _loading_sum(recipe.cations)),
        ("total_anion_molarity_M", _loading_sum(recipe.anions)),
        ("total_solvent_fraction", _loading_sum(recipe.solvents)),
        ("total_additive_weight_fraction", _loading_sum(recipe.additives)),
        ("ionic_strength_M", _ionic_strength_M(molecular_case)),
        (
            "cation_abs_charge_mean",
            _weighted_charge_mean(molecular_case, recipe.cations),
        ),
        (
            "anion_abs_charge_mean",
            _weighted_charge_mean(molecular_case, recipe.anions),
        ),
        (
            "cation_hydrodynamic_radius_A_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.cations,
                "hydrodynamic_radius_A",
            ),
        ),
        (
            "anion_hydrodynamic_radius_A_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.anions,
                "hydrodynamic_radius_A",
            ),
        ),
        (
            "cation_charge_cloud_radius_A_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.cations,
                "charge_cloud_radius_A",
            ),
        ),
        (
            "anion_charge_cloud_radius_A_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.anions,
                "charge_cloud_radius_A",
            ),
        ),
        (
            "solvent_donor_number_mean",
            _weighted_override_mean(molecular_case, recipe.solvents, "donor_number"),
        ),
        (
            "solvent_acceptor_number_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.solvents,
                "acceptor_number",
            ),
        ),
        (
            "solvent_polarizability_A3_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.solvents,
                "polarizability_A3",
            ),
        ),
        (
            "solvent_molecular_volume_A3_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.solvents,
                "molecular_volume_A3",
            ),
        ),
        (
            "additive_donor_number_mean",
            _weighted_override_mean(molecular_case, recipe.additives, "donor_number"),
        ),
        (
            "additive_acceptor_number_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.additives,
                "acceptor_number",
            ),
        ),
        (
            "additive_polarizability_A3_mean",
            _weighted_override_mean(
                molecular_case,
                recipe.additives,
                "polarizability_A3",
            ),
        ),
        (
            "empirical_sigma_spread_mS_cm",
            _nonnegative_float(
                molecular_case.empirical_sigma_spread_mS_cm,
                "empirical_sigma_spread_mS_cm",
            ),
        ),
    )


def _loading_sum(loadings: Mapping[str, float]) -> float:
    return float(
        math.fsum(
            _nonnegative_float(loading, "loading")
            for loading in loadings.values()
        )
    )


def _ionic_strength_M(molecular_case: "MolecularPropertyDbCase") -> float:
    recipe = molecular_case.recipe
    charge_weighted_sum = 0.0
    for loading_mapping in (recipe.cations, recipe.anions):
        for species_name, loading in loading_mapping.items():
            species_input = _case_species_input(molecular_case, species_name)
            charge_weighted_sum += (
                _nonnegative_float(loading, "ionic_strength_loading")
                * species_input.charge_number
                * species_input.charge_number
            )
    return float(0.5 * charge_weighted_sum)


def _weighted_charge_mean(
    molecular_case: "MolecularPropertyDbCase",
    loadings: Mapping[str, float],
) -> float:
    if not loadings:
        return 0.0
    weighted_sum = 0.0
    weight_sum = 0.0
    for species_name, loading in loadings.items():
        parsed_loading = _nonnegative_float(loading, "weighted_charge_loading")
        charge_number = _case_species_input(molecular_case, species_name).charge_number
        weighted_sum += parsed_loading * abs(charge_number)
        weight_sum += parsed_loading
    if weight_sum <= 0.0:
        return 0.0
    return float(weighted_sum / weight_sum)


def _weighted_override_mean(
    molecular_case: "MolecularPropertyDbCase",
    loadings: Mapping[str, float],
    property_name: str,
) -> float:
    if not loadings:
        return 0.0
    weighted_sum = 0.0
    weight_sum = 0.0
    for species_name, loading in loadings.items():
        parsed_loading = _nonnegative_float(loading, f"{property_name}.loading")
        property_value = _case_species_property(
            molecular_case,
            species_name,
            property_name,
        )
        weighted_sum += parsed_loading * property_value
        weight_sum += parsed_loading
    if weight_sum <= 0.0:
        return 0.0
    return float(weighted_sum / weight_sum)


def _case_species_property(
    molecular_case: "MolecularPropertyDbCase",
    species_name: str,
    property_name: str,
) -> float:
    species_input = _case_species_input(molecular_case, species_name)
    if property_name not in species_input.property_overrides:
        raise ValueError(f"{species_name} missing descriptor property {property_name}")
    return _finite_float(
        species_input.property_overrides[property_name],
        f"{species_name}.{property_name}",
    )


def _case_species_input(
    molecular_case: "MolecularPropertyDbCase",
    species_name: str,
):
    if species_name not in molecular_case.species_inputs:
        raise ValueError(
            f"case {molecular_case.row_id} missing species input {species_name}"
        )
    return molecular_case.species_inputs[species_name]


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


def _positive_float_tuple(values: tuple[float, ...], context: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{context} must be nonempty")
    return tuple(_positive_float(value, context) for value in values)


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


def _print_prediction_sensitivity_diagnostics(
    label: str,
    diagnostics: PrimitivePredictionSensitivityDiagnostics,
    options: PrimitiveFitOptions,
) -> None:
    label_text = _nonempty_string(label, "prediction_sensitivity_label")
    _validate_fit_options(options)
    print(
        f"{label_text}_prediction_sensitivity_row_count="
        f"{diagnostics.row_count}"
    )
    print(
        f"{label_text}_prediction_sensitivity_parameter_count="
        f"{diagnostics.parameter_count}"
    )
    print(f"{label_text}_prediction_sensitivity_rank={diagnostics.rank}")
    print(
        f"{label_text}_prediction_sensitivity_condition_number="
        f"{diagnostics.condition_number:.6e}"
    )
    print(
        f"{label_text}_prediction_sensitivity_identifiable_parameter_count="
        f"{len(diagnostics.identifiable_parameter_names)}"
    )
    print(
        f"{label_text}_prediction_sensitivity_identifiable_parameters="
        f"{','.join(diagnostics.identifiable_parameter_names)}"
    )
    print(
        f"{label_text}_prediction_sensitivity_frozen_parameter_count="
        f"{len(diagnostics.frozen_parameter_names)}"
    )
    print(
        f"{label_text}_prediction_sensitivity_frozen_parameters="
        f"{','.join(diagnostics.frozen_parameter_names)}"
    )
    print(
        f"{label_text}_prediction_sensitivity_zero_column_parameters="
        f"{','.join(diagnostics.zero_sensitivity_parameter_names)}"
    )
    print(
        f"{label_text}_prediction_sensitivity_high_correlation_pair_count="
        f"{len(diagnostics.high_correlation_parameter_pairs)}"
    )
    reported_pair_count = (
        options.prediction_sensitivity_reported_correlation_pair_count
    )
    for first_name, second_name, correlation_value in (
        diagnostics.high_correlation_parameter_pairs[:reported_pair_count]
    ):
        print(
            f"{label_text}_prediction_sensitivity_high_correlation_pair "
            f"first={first_name} second={second_name} "
            f"correlation={correlation_value:.6f}"
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

CORRECTOR_BLOCK_PARAMETER_NAMES = (
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
    audit_result: "MolecularPropertyDbAuditResult"


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
    corrector_result: DecomposedFitBlockResult
    cluster_sink_result: DecomposedFitBlockResult
    final_result: DecomposedFitBlockResult
    baseline_audit_result: "MolecularPropertyDbAuditResult"
    candidate_audit_result: "MolecularPropertyDbAuditResult"
    baseline_role_scaling_audit: TransportRoleDirectScalingAudit
    candidate_role_scaling_audit: TransportRoleDirectScalingAudit
    promotion_rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DecomposedFitContext:
    cases: tuple["MolecularPropertyDbCase", ...]
    audit_options: "MolecularPropertyDbAuditOptions"
    fit_options: PrimitiveFitOptions
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...]
    full_evaluator: ConductivityPrimitiveParameterEvaluator
    regularization_reference_parameters: ConductivityPrimitiveParameterSet


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
        selected_direct_capacity_gaps = _select_tuple_values(
            full_evaluation.direct_capacity_gaps_mS_cm,
            self._selected_row_indices,
        )
        selected_corrector_residuals = _select_tuple_values(
            full_evaluation.corrector_residuals_mS_cm,
            self._selected_row_indices,
        )
        return PrimitiveFitDatasetEvaluation(
            descriptor_calibration_targets=_select_descriptor_calibration_targets(
                full_evaluation.descriptor_calibration_targets,
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
            direct_capacity_gaps_mS_cm=selected_direct_capacity_gaps,
            corrector_targets_mS_cm=_select_tuple_values(
                full_evaluation.corrector_targets_mS_cm,
                self._selected_row_indices,
            ),
            corrector_residuals_mS_cm=selected_corrector_residuals,
            direct_capacity_failure_count=sum(
                direct_capacity_gap_mS_cm > 0.0
                for direct_capacity_gap_mS_cm in selected_direct_capacity_gaps
            ),
            corrector_too_strong_failure_count=sum(
                direct_capacity_gap_mS_cm <= 0.0 and corrector_residual_mS_cm > 0.0
                for direct_capacity_gap_mS_cm, corrector_residual_mS_cm in zip(
                    selected_direct_capacity_gaps,
                    selected_corrector_residuals,
                )
            ),
            corrector_too_weak_failure_count=sum(
                direct_capacity_gap_mS_cm <= 0.0 and corrector_residual_mS_cm < 0.0
                for direct_capacity_gap_mS_cm, corrector_residual_mS_cm in zip(
                    selected_direct_capacity_gaps,
                    selected_corrector_residuals,
                )
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


def _select_descriptor_calibration_targets(
    descriptor_targets: tuple[DescriptorCalibrationTarget, ...],
    selected_row_indices: tuple[int, ...],
) -> tuple[DescriptorCalibrationTarget, ...]:
    validated_targets = _validated_descriptor_calibration_targets(descriptor_targets)
    if not selected_row_indices:
        raise ValueError("selected_row_indices must be nonempty")
    return tuple(validated_targets[row_index] for row_index in selected_row_indices)


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


def fit_decomposed_conductivity_primitives() -> DecomposedFitResult:
    from data.electrolyte_property_db import DATA
    from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
    from conductivity.molecular_property_db_audit import (
        MolecularPropertyDbRegistrySource,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        audit_molecular_property_db_cases,
        build_molecular_property_db_case_selection,
        configured_conductivity_primitive_parameters,
        default_molecular_property_db_audit_options,
    )

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
        regularization_reference_parameters=REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
    )
    final_decomposed_context = DecomposedFitContext(
        cases=case_selection.cases,
        audit_options=audit_options,
        fit_options=fit_options,
        coordinate_bounds=coordinate_bounds,
        full_evaluator=full_evaluator,
        regularization_reference_parameters=REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
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

    corrector_start_audit_result = direct_capacity_result.audit_result
    corrector_result = _run_decomposed_block(
        "corrector",
        current_parameters,
        corrector_start_audit_result,
        CORRECTOR_BLOCK_PARAMETER_NAMES,
        tuple(
            row_index
            for row_index, row_result in enumerate(corrector_start_audit_result.rows)
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
    current_parameters = corrector_result.accepted_parameters

    cluster_start_audit_result = corrector_result.audit_result
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
        corrector_result=corrector_result,
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
    starting_audit_result: "MolecularPropertyDbAuditResult",
    block_parameter_names: tuple[str, ...],
    selected_row_indices: tuple[int, ...],
    decomposed_context: DecomposedFitContext,
) -> DecomposedFitBlockResult:
    from conductivity.molecular_property_db_audit import (
        audit_molecular_property_db_cases,
    )

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
        decomposed_context.regularization_reference_parameters,
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
    starting_audit_result: "MolecularPropertyDbAuditResult",
    candidate_audit_result: "MolecularPropertyDbAuditResult",
) -> bool:
    return (
        candidate_audit_result.mae_mS_cm <= starting_audit_result.mae_mS_cm
        and candidate_audit_result.maximum_abs_residual_mS_cm
        <= starting_audit_result.maximum_abs_residual_mS_cm
    )


def _cluster_sink_row_indices(
    audit_result: "MolecularPropertyDbAuditResult",
    fit_options: PrimitiveFitOptions,
) -> tuple[int, ...]:
    sorted_index_and_row = tuple(
        sorted(
            enumerate(audit_result.rows),
            key=lambda index_and_row: abs(index_and_row[1].residual_mS_cm),
            reverse=True,
        )
    )
    tail_index_and_row = sorted_index_and_row[: fit_options.residual_tail_count]
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
    audit_result: "MolecularPropertyDbAuditResult",
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


def print_decomposed_fit_result(result: DecomposedFitResult) -> None:
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
    from data.electrolyte_property_db import DATA
    from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
    from conductivity.molecular_property_db_audit import (
        MolecularPropertyDbRegistrySource,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        audit_molecular_property_db_cases,
        build_molecular_property_db_case_selection,
        configured_conductivity_primitive_parameters,
        default_molecular_property_db_audit_options,
    )

    audit_options = default_molecular_property_db_audit_options()
    fit_options, coordinate_bounds = default_molecular_primitive_fit_configuration()
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
    driver_matrix_diagnostics = primitive_driver_matrix_diagnostics(
        case_selection.cases,
        fit_options,
    )
    evaluator = MolecularPropertyDbPrimitiveEvaluator(
        case_selection.cases,
        audit_options,
        fit_options,
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
    base_speciation_coordinate_bounds = _coordinate_bounds_for_parameter_names(
        coordinate_bounds,
        SPECIATION_FIT_PARAMETER_NAMES,
    )
    base_mobility_event_coordinate_bounds = _coordinate_bounds_for_parameter_names(
        coordinate_bounds,
        MOBILITY_EVENT_FIT_PARAMETER_NAMES,
    )
    speciation_prediction_sensitivity_diagnostics = (
        primitive_prediction_sensitivity_diagnostics(
            configured_parameters,
            base_speciation_coordinate_bounds,
            evaluator,
            fit_options,
        )
    )
    mobility_event_prediction_sensitivity_diagnostics = (
        primitive_prediction_sensitivity_diagnostics(
            configured_parameters,
            base_mobility_event_coordinate_bounds,
            evaluator,
            fit_options,
        )
    )
    full_prediction_sensitivity_diagnostics = (
        primitive_prediction_sensitivity_diagnostics(
            configured_parameters,
            coordinate_bounds,
            evaluator,
            fit_options,
        )
    )
    speciation_coordinate_bounds = (
        _coordinate_bounds_for_stage_and_full_identifiable_parameters(
            base_speciation_coordinate_bounds,
            speciation_prediction_sensitivity_diagnostics,
            full_prediction_sensitivity_diagnostics,
        )
    )
    mobility_event_coordinate_bounds = (
        _coordinate_bounds_for_stage_and_full_identifiable_parameters(
            base_mobility_event_coordinate_bounds,
            mobility_event_prediction_sensitivity_diagnostics,
            full_prediction_sensitivity_diagnostics,
        )
    )
    identifiable_coordinate_bounds = _coordinate_bounds_for_identifiable_parameters(
        coordinate_bounds,
        full_prediction_sensitivity_diagnostics,
    )
    speciation_sensitivity_result = fit_speciation_from_cluster_sensitivities(
        case_selection.cases,
        configured_parameters,
        baseline_audit_result,
        audit_options,
        fit_options,
        speciation_coordinate_bounds,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
    )
    speciation_fit_result = fit_conductivity_primitive_parameters(
        speciation_sensitivity_result.candidate.primitive_parameters,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        speciation_coordinate_bounds,
        evaluator,
        fit_options,
    )
    speciation_candidate = speciation_fit_result.best_candidate
    mobility_event_fit_result = fit_conductivity_primitive_parameters(
        speciation_candidate.primitive_parameters,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        mobility_event_coordinate_bounds,
        evaluator,
        fit_options,
    )
    mobility_event_candidate = mobility_event_fit_result.best_candidate
    fit_result = fit_conductivity_primitive_parameters(
        mobility_event_candidate.primitive_parameters,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        identifiable_coordinate_bounds,
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
    print(f"descriptor_driver_row_count={driver_matrix_diagnostics.row_count}")
    print(f"descriptor_driver_column_count={driver_matrix_diagnostics.column_count}")
    print(f"descriptor_driver_rank={driver_matrix_diagnostics.rank}")
    print(
        "descriptor_driver_condition_number="
        f"{driver_matrix_diagnostics.condition_number:.6e}"
    )
    print(
        "descriptor_driver_zero_variance_columns="
        f"{','.join(driver_matrix_diagnostics.zero_variance_columns)}"
    )
    print(
        "descriptor_driver_high_correlation_pair_count="
        f"{len(driver_matrix_diagnostics.high_correlation_pairs)}"
    )
    reported_correlation_pair_count = (
        fit_options.descriptor_matrix_reported_correlation_pair_count
    )
    for first_name, second_name, correlation_value in (
        driver_matrix_diagnostics.high_correlation_pairs[
            :reported_correlation_pair_count
        ]
    ):
        print(
            "descriptor_driver_high_correlation_pair "
            f"first={first_name} second={second_name} "
            f"correlation={correlation_value:.6f}"
        )
    _print_prediction_sensitivity_diagnostics(
        "speciation",
        speciation_prediction_sensitivity_diagnostics,
        fit_options,
    )
    _print_prediction_sensitivity_diagnostics(
        "mobility_event",
        mobility_event_prediction_sensitivity_diagnostics,
        fit_options,
    )
    _print_prediction_sensitivity_diagnostics(
        "full",
        full_prediction_sensitivity_diagnostics,
        fit_options,
    )
    print(
        "configured_speciation_parameter_count="
        f"{len(base_speciation_coordinate_bounds)}"
    )
    print(
        "configured_mobility_event_parameter_count="
        f"{len(base_mobility_event_coordinate_bounds)}"
    )
    print(f"configured_full_parameter_count={len(coordinate_bounds)}")
    print(f"active_speciation_parameter_count={len(speciation_coordinate_bounds)}")
    print(
        "active_mobility_event_parameter_count="
        f"{len(mobility_event_coordinate_bounds)}"
    )
    print(f"active_full_parameter_count={len(identifiable_coordinate_bounds)}")
    print(
        "speciation_latin_hypercube_sample_count="
        f"{_fit_budget_count_from_parameter_count(
            len(speciation_coordinate_bounds),
            fit_options.latin_hypercube_samples_per_parameter,
            'latin_hypercube_samples_per_parameter',
        )}"
    )
    print(
        "full_latin_hypercube_sample_count="
        f"{_fit_budget_count_from_parameter_count(
            len(identifiable_coordinate_bounds),
            fit_options.latin_hypercube_samples_per_parameter,
            'latin_hypercube_samples_per_parameter',
        )}"
    )
    print(
        "full_powell_max_function_evaluations="
        f"{_fit_budget_count_from_parameter_count_allowing_zero(
            len(identifiable_coordinate_bounds),
            fit_options.powell_max_function_evaluations_per_parameter,
            'powell_max_function_evaluations_per_parameter',
        )}"
    )
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
    print(
        "mobility_event_fit_candidate_count="
        f"{mobility_event_fit_result.candidate_count}"
    )
    print(
        "mobility_event_fit_accepted_candidate_count="
        f"{mobility_event_fit_result.accepted_candidate_count}"
    )
    print(f"mobility_event_fit_mae_mS_cm={mobility_event_candidate.mae_mS_cm:.6f}")
    print(f"mobility_event_fit_bias_mS_cm={mobility_event_candidate.bias_mS_cm:.6f}")
    print(f"mobility_event_fit_pearson_r={mobility_event_candidate.pearson_r:.6f}")
    print(
        "mobility_event_fit_worst_abs_residual_mS_cm="
        f"{mobility_event_candidate.worst_abs_residual_mS_cm:.6f}"
    )
    print(
        "speciation_sensitivity_row_count="
        f"{speciation_sensitivity_result.sensitivity_row_count}"
    )
    print(
        "speciation_sensitivity_entry_count="
        f"{speciation_sensitivity_result.sensitivity_entry_count}"
    )
    print(
        "speciation_sensitivity_mae_mS_cm="
        f"{speciation_sensitivity_result.candidate.mae_mS_cm:.6f}"
    )
    print(
        "speciation_sensitivity_bias_mS_cm="
        f"{speciation_sensitivity_result.candidate.bias_mS_cm:.6f}"
    )
    print(
        "speciation_sensitivity_pearson_r="
        f"{speciation_sensitivity_result.candidate.pearson_r:.6f}"
    )
    print(
        "speciation_sensitivity_worst_abs_residual_mS_cm="
        f"{speciation_sensitivity_result.candidate.worst_abs_residual_mS_cm:.6f}"
    )
    print(
        "speciation_sensitivity_rejected="
        f"{speciation_sensitivity_result.candidate.rejected}"
    )
    print(
        "speciation_sensitivity_rejection_reasons="
        f"{','.join(speciation_sensitivity_result.candidate.rejection_reasons)}"
    )
    print(
        "speciation_sensitivity_cluster_activation_loss="
        f"{speciation_sensitivity_result.candidate.cluster_activation_loss:.6f}"
    )
    print(f"best_objective={best_candidate.objective_value:.6f}")
    print(f"best_mae_mS_cm={best_candidate.mae_mS_cm:.6f}")
    print(f"best_bias_mS_cm={best_candidate.bias_mS_cm:.6f}")
    print(f"best_pearson_r={best_candidate.pearson_r:.6f}")
    print(f"best_worst_abs_residual_mS_cm={best_candidate.worst_abs_residual_mS_cm:.6f}")
    print(f"best_failed_rows={best_candidate.failed_rows}")
    print(f"best_rejected={best_candidate.rejected}")
    print(f"best_rejection_reasons={','.join(best_candidate.rejection_reasons)}")
    print(f"best_cluster_activation_loss={best_candidate.cluster_activation_loss:.6f}")
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
    print(
        "promotion_candidate_cluster_activation_loss="
        f"{promotion_candidate.cluster_activation_loss:.6f}"
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
