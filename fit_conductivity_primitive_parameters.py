"""Fit descriptor-neutral conductivity primitive parameters."""

from __future__ import annotations

import argparse
import json
import math
import random
import itertools
import hashlib
from dataclasses import asdict, dataclass, replace
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
        MolecularClusterThermodynamicDiagnostic,
        MolecularPropertyDbAuditOptions,
        MolecularPropertyDbAuditResult,
        MolecularPropertyDbCase,
        MolecularPropertyDbRowResult,
    )
    from conductivity.molecular_descriptors import MolecularSpeciesInput
    from conductivity.analytical_conductivity_model import (
        AnalyticalConductivityModelResult,
        MolecularElectrolyteRecipe,
    )
    from conductivity.trajectory_primitive_targets import (
        TrajectoryBlockPrimitiveTarget,
        TrajectoryDisplacementMomentTarget,
        TrajectoryPrimitiveTargetArtifact,
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
    "contact_pair_desolvation_offset_over_RT",
    "solvent_separated_pair_desolvation_offset_over_RT",
    "higher_charged_cluster_desolvation_offset_over_RT",
    "internal_polarization_projection_offset",
    "internal_polarization_projection_ionic_strength_slope",
    "internal_polarization_projection_counterion_crowding_slope",
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
    "contact_pair_desolvation_offset_over_RT",
    "solvent_separated_pair_desolvation_offset_over_RT",
    "higher_charged_cluster_desolvation_offset_over_RT",
    "internal_polarization_projection_offset",
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
)

ONSAGER_OPERATOR_FIT_PARAMETER_NAMES = (
    "hydrodynamic_radius_scale_positive_ion",
    "hydrodynamic_radius_scale_negative_ion",
    "hydrodynamic_radius_scale_cluster",
    "shape_friction_exponent",
    "free_volume_exponent",
    "dielectric_mobility_exponent",
    "solvation_mobility_exponent",
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
    "charge_cloud_radius_scale",
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
TRAJECTORY_PRIMITIVE_CALIBRATION_ARTIFACT_TYPE = (
    "trajectory_primitive_calibration_target"
)
PRIMITIVE_FIT_PROGRESS_ARTIFACT_TYPE = (
    "molecular_conductivity_primitive_fit_progress"
)
TRAJECTORY_TARGET_LABEL_SEPARATOR = ":"
TRAJECTORY_CONTACT_PAIR_ROLE = "contact_pair_center"
CONTACT_PAIR_DESOLVATION_PARAMETER_NAME = "contact_pair_desolvation_offset_over_RT"
TRAJECTORY_POPULATION_PARAMETER_UPDATES_BY_ROLE = {
    TRAJECTORY_CONTACT_PAIR_ROLE: (
        (CONTACT_PAIR_DESOLVATION_PARAMETER_NAME, -1.0),
    ),
    "solvent_separated_pair_center": (
        ("solvent_separated_pair_logK_offset", 1.0),
    ),
    "internal_polarization_center": (
        ("internal_polarization_projection_offset", 1.0),
    ),
}


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
    trajectory_primitive_target_paths: tuple[str, ...]
    prediction_sensitivity_coordinate_step: float
    prediction_sensitivity_min_column_norm_mS_cm_per_coordinate: float
    prediction_sensitivity_relative_singular_value_threshold: float
    prediction_sensitivity_high_correlation_threshold: float
    prediction_sensitivity_reported_correlation_pair_count: int
    candidate_output_path: str
    progress_output_path: str
    decomposition_report_output_path: str
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
class TrajectoryPrimitiveCalibrationTarget:
    system_id: str
    recipe: "MolecularElectrolyteRecipe"
    species_inputs: Mapping[str, "MolecularSpeciesInput"]
    primitive_target_artifact: "TrajectoryPrimitiveTargetArtifact"


@dataclass(frozen=True)
class TrajectoryPrimitiveLossBreakdown:
    concentration_loss: float
    transition_rate_loss: float
    displacement_moment_loss: float
    sigma_loss_mS_cm: float


@dataclass(frozen=True)
class TrajectoryConcentrationTargetCoverage:
    system_id: str
    positive_target_count: int
    reachable_target_count: int
    unreachable_target_labels: tuple[str, ...]
    under_floor_target_labels: tuple[str, ...]
    predicted_target_rows: tuple[tuple[str, float, float, bool], ...]


@dataclass(frozen=True)
class PrimitiveFitDatasetEvaluation:
    descriptor_calibration_targets: tuple[DescriptorCalibrationTarget, ...]
    trajectory_primitive_calibration_targets: tuple[
        TrajectoryPrimitiveCalibrationTarget,
        ...,
    ]
    predicted_sigmas_mS_cm: tuple[float, ...]
    direct_sigmas_mS_cm: tuple[float, ...]
    corrector_sigmas_mS_cm: tuple[float, ...]
    direct_capacity_gaps_mS_cm: tuple[float, ...]
    corrector_targets_mS_cm: tuple[float, ...]
    corrector_residuals_mS_cm: tuple[float, ...]
    direct_capacity_failure_count: int
    corrector_too_strong_failure_count: int
    corrector_too_weak_failure_count: int
    trajectory_concentration_loss: float
    trajectory_transition_rate_loss: float
    trajectory_displacement_moment_loss: float
    trajectory_sigma_loss_mS_cm: float
    trajectory_concentration_unreachable_target_count: int
    trajectory_concentration_under_floor_target_count: int
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
        trajectory_primitive_calibration_targets: tuple[
            TrajectoryPrimitiveCalibrationTarget,
            ...,
        ],
    ) -> None:
        if not cases:
            raise ValueError("molecular property-DB evaluator requires cases")
        self._cases = cases
        self._audit_options = audit_options
        self._fit_options = fit_options
        self._trajectory_primitive_calibration_targets = (
            trajectory_primitive_calibration_targets
        )
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
        trajectory_losses = _trajectory_primitive_loss_breakdown(
            self._trajectory_primitive_calibration_targets,
            primitive_parameters,
            self._audit_options,
            self._fit_options,
        )
        trajectory_coverages = trajectory_concentration_target_coverage(
            self._trajectory_primitive_calibration_targets,
            primitive_parameters,
            self._audit_options,
        )
        return PrimitiveFitDatasetEvaluation(
            descriptor_calibration_targets=self._descriptor_calibration_targets,
            trajectory_primitive_calibration_targets=(
                self._trajectory_primitive_calibration_targets
            ),
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
            trajectory_concentration_loss=trajectory_losses.concentration_loss,
            trajectory_transition_rate_loss=trajectory_losses.transition_rate_loss,
            trajectory_displacement_moment_loss=(
                trajectory_losses.displacement_moment_loss
            ),
            trajectory_sigma_loss_mS_cm=trajectory_losses.sigma_loss_mS_cm,
            trajectory_concentration_unreachable_target_count=sum(
                len(coverage.unreachable_target_labels)
                for coverage in trajectory_coverages
            ),
            trajectory_concentration_under_floor_target_count=sum(
                len(coverage.under_floor_target_labels)
                for coverage in trajectory_coverages
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
            key=_absolute_residual_sort_key,
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


def _trajectory_primitive_loss_breakdown(
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
    fit_options: PrimitiveFitOptions,
) -> TrajectoryPrimitiveLossBreakdown:
    if not trajectory_targets:
        return TrajectoryPrimitiveLossBreakdown(
            concentration_loss=0.0,
            transition_rate_loss=0.0,
            displacement_moment_loss=0.0,
            sigma_loss_mS_cm=0.0,
        )
    concentration_losses: list[float] = []
    transition_rate_losses: list[float] = []
    displacement_moment_losses: list[float] = []
    sigma_losses_mS_cm: list[float] = []
    for trajectory_target in trajectory_targets:
        target_artifact = trajectory_target.primitive_target_artifact
        analytical_result = _trajectory_target_analytical_result(
            trajectory_target,
            primitive_parameters,
            audit_options,
        )
        concentration_losses.append(
            _uncertainty_normalized_mapping_loss(
                _transport_center_concentration_targets(analytical_result),
                target_artifact.state_concentrations_mol_m3,
                target_artifact.block_state_concentration_standard_errors_mol_m3,
                f"{trajectory_target.system_id}.state_concentrations_mol_m3",
            )
        )
        if target_artifact.transition_rate_targets_validated:
            transition_rate_losses.append(
                _uncertainty_normalized_mapping_loss(
                    _transition_rate_targets_from_analytical_result(
                        analytical_result
                    ),
                    target_artifact.transition_rates_s_inv,
                    target_artifact.block_transition_rate_standard_errors_s_inv,
                    f"{trajectory_target.system_id}.transition_rates_s_inv",
                )
            )
        if target_artifact.displacement_moment_targets_validated:
            displacement_moment_losses.append(
                _uncertainty_normalized_mapping_loss(
                    _displacement_moment_targets_from_analytical_result(
                        analytical_result,
                    ),
                    _mean_squared_displacement_targets(
                        target_artifact.displacement_moments_by_family,
                    ),
                    target_artifact.block_displacement_moment_standard_errors_m2,
                    f"{trajectory_target.system_id}.displacement_moments_by_family",
                )
            )
        if target_artifact.markov_additive_sigma_validated:
            sigma_losses_mS_cm.append(
                _uncertainty_normalized_scalar_loss(
                    analytical_result.sigma_mS_cm,
                    target_artifact.markov_additive_sigma_mS_cm,
                    target_artifact.block_sigma_standard_error_mS_cm,
                    f"{trajectory_target.system_id}.markov_additive_sigma_mS_cm",
                )
            )
    return TrajectoryPrimitiveLossBreakdown(
        concentration_loss=_mean_loss(concentration_losses, "trajectory_c_loss"),
        transition_rate_loss=_mean_loss(transition_rate_losses, "trajectory_Q_loss"),
        displacement_moment_loss=_mean_loss(
            displacement_moment_losses,
            "trajectory_d_loss",
        ),
        sigma_loss_mS_cm=_mean_loss(sigma_losses_mS_cm, "trajectory_sigma_loss"),
    )


def _trajectory_target_analytical_result(
    trajectory_target: TrajectoryPrimitiveCalibrationTarget,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
) -> "AnalyticalConductivityModelResult":
    from conductivity.analytical_conductivity_model import (
        AnalyticalConductivityModelInput,
        MolecularMoriOptions,
        compute_analytical_conductivity_model,
    )
    from conductivity.molecular_descriptors import ProvidedPropertyDescriptorBackend

    molecular_options = MolecularMoriOptions(
        max_cluster_ion_count=audit_options.max_cluster_ion_count,
        max_packing_fraction=audit_options.max_packing_fraction,
        free_volume_exponent=audit_options.free_volume_exponent,
        translation_jump_length_multiplier=(
            audit_options.translation_jump_length_multiplier
        ),
        primitive_parameters=primitive_parameters,
    )
    return compute_analytical_conductivity_model(
        AnalyticalConductivityModelInput(
            recipe=trajectory_target.recipe,
            species_inputs=trajectory_target.species_inputs,
            descriptor_backend=ProvidedPropertyDescriptorBackend(),
            options=molecular_options,
        )
    )


def _transport_center_concentration_targets(
    analytical_result: "AnalyticalConductivityModelResult",
) -> Mapping[str, float]:
    concentration_by_target_label: dict[str, float] = {}
    cluster_template_by_label = {
        cluster_template.label: cluster_template
        for cluster_template in analytical_result.cluster_states
    }
    descriptor_species_names = set(analytical_result.descriptors)
    for transport_state in analytical_result.transport_states:
        if transport_state.center_species_name in descriptor_species_names:
            _add_concentration_target(
                concentration_by_target_label,
                _transport_center_target_label(
                    transport_state.transport_role,
                    transport_state.center_species_name,
                ),
                transport_state.concentration_mol_m3,
            )
            continue
        if transport_state.parent_cluster_label not in cluster_template_by_label:
            continue
        cluster_template = cluster_template_by_label[
            transport_state.parent_cluster_label
        ]
        for species_name, stoichiometric_count in cluster_template.stoichiometry.items():
            _add_concentration_target(
                concentration_by_target_label,
                _transport_center_target_label(
                    transport_state.transport_role,
                    species_name,
                ),
                transport_state.concentration_mol_m3
                * _positive_int(
                    stoichiometric_count,
                    f"{cluster_template.label}.{species_name}.stoichiometric_count",
                ),
            )
    return concentration_by_target_label


def _add_concentration_target(
    concentration_by_target_label: dict[str, float],
    target_label: str,
    concentration_mol_m3: float,
) -> None:
    parsed_concentration_mol_m3 = _nonnegative_float(
        concentration_mol_m3,
        f"{target_label}.concentration_mol_m3",
    )
    if target_label not in concentration_by_target_label:
        concentration_by_target_label[target_label] = 0.0
    concentration_by_target_label[target_label] += parsed_concentration_mol_m3


def _transport_center_target_label(
    transport_role: str,
    center_species_name: str,
) -> str:
    role_name = _nonempty_string(transport_role, "transport_role")
    species_name = _nonempty_string(center_species_name, "center_species_name")
    return f"{role_name}:{species_name}"


def _transition_rate_targets_from_analytical_result(
    analytical_result: "AnalyticalConductivityModelResult",
) -> Mapping[str, float]:
    target_label_by_state_index = _target_label_by_markov_state_index(
        analytical_result,
    )
    transition_rates_s_inv: dict[str, float] = {}
    for event in analytical_result.events:
        if event.from_state_index == event.to_state_index:
            continue
        if event.from_state_index not in target_label_by_state_index:
            continue
        if event.to_state_index not in target_label_by_state_index:
            continue
        from_target_label = target_label_by_state_index[event.from_state_index]
        to_target_label = target_label_by_state_index[event.to_state_index]
        if from_target_label == to_target_label:
            continue
        transition_label = _transition_label(from_target_label, to_target_label)
        if transition_label not in transition_rates_s_inv:
            transition_rates_s_inv[transition_label] = 0.0
        transition_rates_s_inv[transition_label] += event.rate_s_inv
    return transition_rates_s_inv


def _displacement_moment_targets_from_analytical_result(
    analytical_result: "AnalyticalConductivityModelResult",
) -> Mapping[str, float]:
    target_label_by_state_index = _target_label_by_markov_state_index(
        analytical_result,
    )
    weighted_displacement_m2_by_transition: dict[str, float] = {}
    rate_by_transition: dict[str, float] = {}
    for event in analytical_result.events:
        if event.from_state_index not in target_label_by_state_index:
            continue
        if event.to_state_index not in target_label_by_state_index:
            continue
        from_target_label = target_label_by_state_index[event.from_state_index]
        to_target_label = target_label_by_state_index[event.to_state_index]
        transition_label = _transition_label(from_target_label, to_target_label)
        displacement_m2 = math.fsum(
            component_m * component_m
            for component_m in event.charge_displacement_m
        )
        if transition_label not in weighted_displacement_m2_by_transition:
            weighted_displacement_m2_by_transition[transition_label] = 0.0
            rate_by_transition[transition_label] = 0.0
        weighted_displacement_m2_by_transition[transition_label] += (
            event.rate_s_inv * displacement_m2
        )
        rate_by_transition[transition_label] += event.rate_s_inv
    displacement_moments_m2: dict[str, float] = {}
    for transition_label, weighted_displacement_m2 in (
        weighted_displacement_m2_by_transition.items()
    ):
        transition_rate_s_inv = _positive_float(
            rate_by_transition[transition_label],
            f"{transition_label}.transition_rate_s_inv",
        )
        displacement_moments_m2[transition_label] = (
            weighted_displacement_m2 / transition_rate_s_inv
        )
    return displacement_moments_m2


def _target_label_by_markov_state_index(
    analytical_result: "AnalyticalConductivityModelResult",
) -> Mapping[int, str]:
    target_label_by_transport_mobile_label = {
        f"{transport_state.label}:mobile": _transport_center_target_label(
            transport_state.transport_role,
            transport_state.center_species_name,
        )
        for transport_state in analytical_result.transport_states
    }
    target_label_by_state_index: dict[int, str] = {}
    for state_index, markov_state_label in enumerate(
        analytical_result.markov_state_labels,
    ):
        if markov_state_label in target_label_by_transport_mobile_label:
            target_label_by_state_index[state_index] = (
                target_label_by_transport_mobile_label[markov_state_label]
            )
    return target_label_by_state_index


def _transition_label(
    from_target_label: str,
    to_target_label: str,
) -> str:
    return (
        f"{_nonempty_string(from_target_label, 'from_target_label')}"
        f"->{_nonempty_string(to_target_label, 'to_target_label')}"
    )


def _mean_squared_displacement_targets(
    displacement_moments_by_family: Mapping[
        str,
        "TrajectoryDisplacementMomentTarget",
    ],
) -> Mapping[str, float]:
    displacement_targets_m2: dict[str, float] = {}
    for transition_label, moment_target in displacement_moments_by_family.items():
        displacement_targets_m2[
            _nonempty_string(transition_label, "transition_label")
        ] = _nonnegative_float(
            moment_target.mean_squared_displacement_m2,
            f"{transition_label}.mean_squared_displacement_m2",
        )
    return displacement_targets_m2


def trajectory_concentration_target_coverage(
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
) -> tuple[TrajectoryConcentrationTargetCoverage, ...]:
    coverages: list[TrajectoryConcentrationTargetCoverage] = []
    for trajectory_target in trajectory_targets:
        target_artifact = trajectory_target.primitive_target_artifact
        analytical_result = _trajectory_target_analytical_result(
            trajectory_target,
            primitive_parameters,
            audit_options,
        )
        predicted_concentrations = _transport_center_concentration_targets(
            analytical_result,
        )
        reachable_labels = _reachable_transport_center_target_labels(
            analytical_result,
        )
        concentration_floor_mol_m3 = _trajectory_concentration_floor_mol_m3(
            analytical_result,
        )
        positive_target_rows: list[tuple[str, float, float, bool]] = []
        unreachable_labels: list[str] = []
        under_floor_labels: list[str] = []
        for target_label, target_concentration_mol_m3 in sorted(
            target_artifact.state_concentrations_mol_m3.items()
        ):
            parsed_target_concentration_mol_m3 = _nonnegative_float(
                target_concentration_mol_m3,
                (
                    f"{trajectory_target.system_id}.{target_label}"
                    ".target_concentration_mol_m3"
                ),
            )
            if parsed_target_concentration_mol_m3 <= 0.0:
                continue
            reachable = target_label in reachable_labels
            predicted_concentration_mol_m3 = _predicted_mapping_value_or_zero(
                predicted_concentrations,
                target_label,
                f"{trajectory_target.system_id}.state_concentrations_mol_m3",
            )
            positive_target_rows.append(
                (
                    target_label,
                    parsed_target_concentration_mol_m3,
                    predicted_concentration_mol_m3,
                    reachable,
                )
            )
            if not reachable:
                unreachable_labels.append(target_label)
                continue
            if predicted_concentration_mol_m3 <= concentration_floor_mol_m3:
                under_floor_labels.append(target_label)
        coverages.append(
            TrajectoryConcentrationTargetCoverage(
                system_id=trajectory_target.system_id,
                positive_target_count=len(positive_target_rows),
                reachable_target_count=sum(
                    1 for target_row in positive_target_rows if target_row[3]
                ),
                unreachable_target_labels=tuple(unreachable_labels),
                under_floor_target_labels=tuple(under_floor_labels),
                predicted_target_rows=tuple(positive_target_rows),
            )
        )
    return tuple(coverages)


def validate_trajectory_concentration_target_coverage(
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
) -> tuple[TrajectoryConcentrationTargetCoverage, ...]:
    coverages = trajectory_concentration_target_coverage(
        trajectory_targets,
        primitive_parameters,
        audit_options,
    )
    unreachable_pairs: list[str] = []
    for coverage in coverages:
        unreachable_pairs.extend(
            f"{coverage.system_id}:{target_label}"
            for target_label in coverage.unreachable_target_labels
        )
    if unreachable_pairs:
        raise ValueError(
            "trajectory concentration targets are outside analytical target "
            f"namespace: {tuple(unreachable_pairs)}"
        )
    return coverages


def print_trajectory_concentration_target_coverage(
    report_prefix: str,
    coverages: tuple[TrajectoryConcentrationTargetCoverage, ...],
) -> None:
    prefix = _nonempty_string(report_prefix, "trajectory_coverage_report_prefix")
    for coverage in coverages:
        system_id = _nonempty_string(coverage.system_id, "coverage.system_id")
        print(
            f"{prefix}_trajectory_target_coverage_system={system_id} "
            f"positive_targets={coverage.positive_target_count} "
            f"reachable_targets={coverage.reachable_target_count} "
            f"unreachable_targets={len(coverage.unreachable_target_labels)} "
            f"under_floor_targets={len(coverage.under_floor_target_labels)}"
        )
        if coverage.unreachable_target_labels:
            print(
                f"{prefix}_trajectory_target_unreachable_labels_system={system_id} "
                f"labels={','.join(coverage.unreachable_target_labels)}"
            )
        if coverage.under_floor_target_labels:
            print(
                f"{prefix}_trajectory_target_under_floor_labels_system={system_id} "
                f"labels={','.join(coverage.under_floor_target_labels)}"
            )
        for (
            target_label,
            target_concentration_mol_m3,
            predicted_concentration_mol_m3,
            reachable,
        ) in coverage.predicted_target_rows:
            print(
                f"{prefix}_trajectory_target_row system={system_id} "
                f"label={target_label} "
                f"target_mol_m3={target_concentration_mol_m3:.6e} "
                f"predicted_mol_m3={predicted_concentration_mol_m3:.6e} "
                f"reachable={reachable}"
            )


def initialize_topology_logk_offsets_from_trajectory_concentrations(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    fit_options: PrimitiveFitOptions,
) -> ConductivityPrimitiveParameterSet:
    if not trajectory_targets:
        return primitive_parameters
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_fit_options(fit_options)
    coordinate_bound_by_name = {
        coordinate_bound.name: coordinate_bound
        for coordinate_bound in _ordered_coordinate_bounds(coordinate_bounds)
    }
    updated_parameter_mapping = conductivity_primitive_parameters_to_mapping(
        primitive_parameters,
    )
    numerator_by_parameter_name: dict[str, float] = {}
    denominator_by_parameter_name: dict[str, float] = {}
    for trajectory_target in trajectory_targets:
        analytical_result = _trajectory_target_analytical_result(
            trajectory_target,
            primitive_parameters,
            audit_options,
        )
        predicted_concentrations = _transport_center_concentration_targets(
            analytical_result,
        )
        reachable_labels = _reachable_transport_center_target_labels(
            analytical_result,
        )
        concentration_floor_mol_m3 = _trajectory_concentration_floor_mol_m3(
            analytical_result,
        )
        for target_label, target_concentration_mol_m3 in (
            trajectory_target.primitive_target_artifact.state_concentrations_mol_m3.items()
        ):
            parsed_target_concentration_mol_m3 = _nonnegative_float(
                target_concentration_mol_m3,
                (
                    f"{trajectory_target.system_id}.{target_label}"
                    ".target_concentration_mol_m3"
                ),
            )
            if parsed_target_concentration_mol_m3 <= 0.0:
                continue
            if target_label not in reachable_labels:
                continue
            target_role, _target_species_name = _trajectory_target_label_parts(
                target_label,
            )
            if target_role not in TRAJECTORY_POPULATION_PARAMETER_UPDATES_BY_ROLE:
                continue
            predicted_concentration_mol_m3 = max(
                concentration_floor_mol_m3,
                _predicted_mapping_value_or_zero(
                    predicted_concentrations,
                    target_label,
                    f"{trajectory_target.system_id}.state_concentrations_mol_m3",
                ),
            )
            log_concentration_ratio = (
                math.log(parsed_target_concentration_mol_m3)
                - math.log(predicted_concentration_mol_m3)
            )
            target_weight = max(
                parsed_target_concentration_mol_m3,
                concentration_floor_mol_m3,
            )
            for parameter_name, update_sign in (
                TRAJECTORY_POPULATION_PARAMETER_UPDATES_BY_ROLE[target_role]
            ):
                if parameter_name == CONTACT_PAIR_DESOLVATION_PARAMETER_NAME:
                    continue
                if parameter_name not in numerator_by_parameter_name:
                    numerator_by_parameter_name[parameter_name] = 0.0
                    denominator_by_parameter_name[parameter_name] = 0.0
                numerator_by_parameter_name[parameter_name] += (
                    target_weight * update_sign * log_concentration_ratio
                )
                denominator_by_parameter_name[parameter_name] += target_weight
    if numerator_by_parameter_name:
        for parameter_name, numerator_value in numerator_by_parameter_name.items():
            if parameter_name not in coordinate_bound_by_name:
                raise ValueError(f"missing coordinate bound for {parameter_name}")
            denominator_value = _positive_float(
                denominator_by_parameter_name[parameter_name],
                f"{parameter_name}.trajectory_initialization_weight",
            )
            current_coordinate = _finite_float(
                updated_parameter_mapping[parameter_name],
                f"{parameter_name}.current_coordinate",
            )
            proposed_coordinate = (
                current_coordinate + numerator_value / denominator_value
            )
            coordinate_bound = coordinate_bound_by_name[parameter_name]
            updated_parameter_mapping[parameter_name] = min(
                coordinate_bound.upper,
                max(coordinate_bound.lower, proposed_coordinate),
            )
    initialized_parameters = conductivity_primitive_parameters_from_mapping(
        updated_parameter_mapping,
    )
    validate_conductivity_primitive_parameters(initialized_parameters)
    return _initialize_contact_pair_desolvation_from_trajectory_response(
        initialized_parameters,
        trajectory_targets,
        audit_options,
        coordinate_bound_by_name,
        fit_options.minimum_coordinate_step,
    )


@dataclass(frozen=True)
class _ContactPairReachabilityPoint:
    coordinate_value: float
    log_concentration_residual: float


@dataclass(frozen=True)
class _ContactPairReachabilityProbe:
    successful: bool
    coordinate_value: float
    log_concentration_residual: float
    failure_reason: str


@dataclass(frozen=True)
class _ContactPairReachabilityBracket:
    bracketed: bool
    overprediction_point: _ContactPairReachabilityPoint
    underprediction_point: _ContactPairReachabilityPoint
    evaluated_points: tuple[_ContactPairReachabilityPoint, ...]


@dataclass(frozen=True)
class _ContactPairLogResidual:
    has_contact_pair_targets: bool
    log_concentration_residual: float


def _initialize_contact_pair_desolvation_from_trajectory_response(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    coordinate_bound_by_name: Mapping[str, PrimitiveParameterTransform],
    coordinate_tolerance: float,
) -> ConductivityPrimitiveParameterSet:
    validate_conductivity_primitive_parameters(primitive_parameters)
    if CONTACT_PAIR_DESOLVATION_PARAMETER_NAME not in coordinate_bound_by_name:
        raise ValueError(
            f"missing coordinate bound for {CONTACT_PAIR_DESOLVATION_PARAMETER_NAME}"
        )
    parameter_mapping = conductivity_primitive_parameters_to_mapping(
        primitive_parameters,
    )
    coordinate_bound = coordinate_bound_by_name[CONTACT_PAIR_DESOLVATION_PARAMETER_NAME]
    tolerance = _positive_float(
        coordinate_tolerance,
        "contact_pair_reachability_coordinate_tolerance",
    )
    current_probe = _contact_pair_reachability_probe(
        parameter_mapping,
        parameter_mapping[CONTACT_PAIR_DESOLVATION_PARAMETER_NAME],
        trajectory_targets,
        audit_options,
    )
    if not current_probe.successful:
        if current_probe.failure_reason == "no_contact_pair_targets":
            return primitive_parameters
        raise ValueError(
            "contact-pair reachability initialization failed at current "
            f"coordinate: {current_probe.failure_reason}"
        )
    current_point = _contact_pair_probe_point(current_probe)
    candidate_points = [current_point]
    if current_point.log_concentration_residual < 0.0:
        bracket_result = _contact_pair_overprediction_bracket_below_current(
            parameter_mapping,
            trajectory_targets,
            audit_options,
            coordinate_bound.lower,
            current_point,
            tolerance,
        )
    else:
        bracket_result = _contact_pair_underprediction_bracket_above_current(
            parameter_mapping,
            trajectory_targets,
            audit_options,
            current_point,
            coordinate_bound.upper,
            tolerance,
        )
    candidate_points.extend(bracket_result.evaluated_points)
    if bracket_result.bracketed:
        solved_point = _solve_contact_pair_response_bracket(
            parameter_mapping,
            trajectory_targets,
            audit_options,
            bracket_result.overprediction_point,
            bracket_result.underprediction_point,
            tolerance,
        )
        candidate_points.append(solved_point)
    best_point = _best_contact_pair_reachability_point(tuple(candidate_points))
    if best_point.coordinate_value == current_point.coordinate_value:
        return primitive_parameters
    updated_parameter_mapping = dict(parameter_mapping)
    updated_parameter_mapping[CONTACT_PAIR_DESOLVATION_PARAMETER_NAME] = (
        best_point.coordinate_value
    )
    initialized_parameters = conductivity_primitive_parameters_from_mapping(
        updated_parameter_mapping,
    )
    validate_conductivity_primitive_parameters(initialized_parameters)
    return initialized_parameters


def _contact_pair_probe_point(
    probe: _ContactPairReachabilityProbe,
) -> _ContactPairReachabilityPoint:
    if not probe.successful:
        raise ValueError(f"contact-pair probe failed: {probe.failure_reason}")
    return _ContactPairReachabilityPoint(
        coordinate_value=probe.coordinate_value,
        log_concentration_residual=probe.log_concentration_residual,
    )


def _contact_pair_overprediction_bracket_below_current(
    parameter_mapping: Mapping[str, float],
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    lower_coordinate: float,
    current_underprediction_point: _ContactPairReachabilityPoint,
    coordinate_tolerance: float,
) -> _ContactPairReachabilityBracket:
    underprediction_point = current_underprediction_point
    lower_boundary = _finite_float(
        lower_coordinate,
        "contact_pair_reachability_lower_coordinate",
    )
    evaluated_points: list[_ContactPairReachabilityPoint] = []
    iteration_count = _coordinate_bisection_iteration_count(
        underprediction_point.coordinate_value - lower_boundary,
        coordinate_tolerance,
    )
    for iteration_index in range(iteration_count):
        if not lower_boundary < underprediction_point.coordinate_value:
            break
        probe_coordinate = 0.5 * (
            lower_boundary + underprediction_point.coordinate_value
        )
        if not coordinate_tolerance < (
            underprediction_point.coordinate_value - probe_coordinate
        ):
            break
        probe = _contact_pair_reachability_probe(
            parameter_mapping,
            probe_coordinate,
            trajectory_targets,
            audit_options,
        )
        if not probe.successful:
            lower_boundary = probe_coordinate
            continue
        probe_point = _contact_pair_probe_point(probe)
        evaluated_points.append(probe_point)
        if 0.0 <= probe_point.log_concentration_residual:
            return _ContactPairReachabilityBracket(
                bracketed=True,
                overprediction_point=probe_point,
                underprediction_point=underprediction_point,
                evaluated_points=tuple(evaluated_points),
            )
        underprediction_point = probe_point
        _nonnegative_int(iteration_index, "contact_pair_bracket_iteration")
    return _ContactPairReachabilityBracket(
        bracketed=False,
        overprediction_point=underprediction_point,
        underprediction_point=underprediction_point,
        evaluated_points=tuple(evaluated_points),
    )


def _contact_pair_underprediction_bracket_above_current(
    parameter_mapping: Mapping[str, float],
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    current_overprediction_point: _ContactPairReachabilityPoint,
    upper_coordinate: float,
    coordinate_tolerance: float,
) -> _ContactPairReachabilityBracket:
    overprediction_point = current_overprediction_point
    upper_boundary = _finite_float(
        upper_coordinate,
        "contact_pair_reachability_upper_coordinate",
    )
    evaluated_points: list[_ContactPairReachabilityPoint] = []
    iteration_count = _coordinate_bisection_iteration_count(
        upper_boundary - overprediction_point.coordinate_value,
        coordinate_tolerance,
    )
    for iteration_index in range(iteration_count):
        if not overprediction_point.coordinate_value < upper_boundary:
            break
        probe_coordinate = 0.5 * (
            overprediction_point.coordinate_value + upper_boundary
        )
        if not coordinate_tolerance < (
            probe_coordinate - overprediction_point.coordinate_value
        ):
            break
        probe = _contact_pair_reachability_probe(
            parameter_mapping,
            probe_coordinate,
            trajectory_targets,
            audit_options,
        )
        if not probe.successful:
            upper_boundary = probe_coordinate
            continue
        probe_point = _contact_pair_probe_point(probe)
        evaluated_points.append(probe_point)
        if probe_point.log_concentration_residual <= 0.0:
            return _ContactPairReachabilityBracket(
                bracketed=True,
                overprediction_point=overprediction_point,
                underprediction_point=probe_point,
                evaluated_points=tuple(evaluated_points),
            )
        overprediction_point = probe_point
        _nonnegative_int(iteration_index, "contact_pair_bracket_iteration")
    return _ContactPairReachabilityBracket(
        bracketed=False,
        overprediction_point=overprediction_point,
        underprediction_point=overprediction_point,
        evaluated_points=tuple(evaluated_points),
    )


def _solve_contact_pair_response_bracket(
    parameter_mapping: Mapping[str, float],
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    overprediction_point: _ContactPairReachabilityPoint,
    underprediction_point: _ContactPairReachabilityPoint,
    coordinate_tolerance: float,
) -> _ContactPairReachabilityPoint:
    if not overprediction_point.coordinate_value < underprediction_point.coordinate_value:
        raise ValueError("contact-pair response bracket has inverted coordinates")
    if overprediction_point.log_concentration_residual < 0.0:
        raise ValueError("contact-pair response bracket missing overprediction side")
    if 0.0 < underprediction_point.log_concentration_residual:
        raise ValueError("contact-pair response bracket missing underprediction side")
    best_point = _best_contact_pair_reachability_point(
        (overprediction_point, underprediction_point)
    )
    iteration_count = _coordinate_bisection_iteration_count(
        underprediction_point.coordinate_value - overprediction_point.coordinate_value,
        coordinate_tolerance,
    )
    bracket_overprediction_point = overprediction_point
    bracket_underprediction_point = underprediction_point
    for iteration_index in range(iteration_count):
        if not coordinate_tolerance < (
            bracket_underprediction_point.coordinate_value
            - bracket_overprediction_point.coordinate_value
        ):
            break
        probe_coordinate = 0.5 * (
            bracket_overprediction_point.coordinate_value
            + bracket_underprediction_point.coordinate_value
        )
        probe = _contact_pair_reachability_probe(
            parameter_mapping,
            probe_coordinate,
            trajectory_targets,
            audit_options,
        )
        if not probe.successful:
            raise ValueError(
                "contact-pair response solve failed inside a feasible bracket: "
                f"{probe.failure_reason}"
            )
        probe_point = _contact_pair_probe_point(probe)
        best_point = _best_contact_pair_reachability_point(
            (best_point, probe_point)
        )
        if 0.0 <= probe_point.log_concentration_residual:
            bracket_overprediction_point = probe_point
        else:
            bracket_underprediction_point = probe_point
        _nonnegative_int(iteration_index, "contact_pair_solve_iteration")
    return best_point


def _coordinate_bisection_iteration_count(
    coordinate_span: float,
    coordinate_tolerance: float,
) -> int:
    span = _nonnegative_float(coordinate_span, "coordinate_bisection_span")
    tolerance = _positive_float(
        coordinate_tolerance,
        "coordinate_bisection_tolerance",
    )
    if span <= tolerance:
        return 1
    return int(math.ceil(math.log2(span / tolerance))) + 1


def _contact_pair_reachability_probe(
    parameter_mapping: Mapping[str, float],
    contact_pair_desolvation_offset_over_RT: float,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    audit_options: "MolecularPropertyDbAuditOptions",
) -> _ContactPairReachabilityProbe:
    coordinate_value = _finite_float(
        contact_pair_desolvation_offset_over_RT,
        CONTACT_PAIR_DESOLVATION_PARAMETER_NAME,
    )
    updated_parameter_mapping = dict(parameter_mapping)
    updated_parameter_mapping[CONTACT_PAIR_DESOLVATION_PARAMETER_NAME] = (
        coordinate_value
    )
    try:
        primitive_parameters = conductivity_primitive_parameters_from_mapping(
            updated_parameter_mapping,
        )
        residual = _contact_pair_log_concentration_residual(
            primitive_parameters,
            trajectory_targets,
            audit_options,
        )
    except (
        FloatingPointError,
        OverflowError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return _ContactPairReachabilityProbe(
            successful=False,
            coordinate_value=coordinate_value,
            log_concentration_residual=0.0,
            failure_reason=f"{type(exc).__name__}:{exc}",
        )
    if not residual.has_contact_pair_targets:
        return _ContactPairReachabilityProbe(
            successful=False,
            coordinate_value=coordinate_value,
            log_concentration_residual=0.0,
            failure_reason="no_contact_pair_targets",
        )
    return _ContactPairReachabilityProbe(
        successful=True,
        coordinate_value=coordinate_value,
        log_concentration_residual=residual.log_concentration_residual,
        failure_reason="",
    )


def _contact_pair_log_concentration_residual(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    audit_options: "MolecularPropertyDbAuditOptions",
) -> _ContactPairLogResidual:
    weighted_residual_sum = 0.0
    weight_sum = 0.0
    for trajectory_target in trajectory_targets:
        analytical_result = _trajectory_target_analytical_result(
            trajectory_target,
            primitive_parameters,
            audit_options,
        )
        predicted_concentrations = _transport_center_concentration_targets(
            analytical_result,
        )
        reachable_labels = _reachable_transport_center_target_labels(
            analytical_result,
        )
        concentration_floor_mol_m3 = _trajectory_concentration_floor_mol_m3(
            analytical_result,
        )
        for target_label, target_concentration_mol_m3 in (
            trajectory_target.primitive_target_artifact.state_concentrations_mol_m3.items()
        ):
            target_role, _target_species_name = _trajectory_target_label_parts(
                target_label,
            )
            if target_role != TRAJECTORY_CONTACT_PAIR_ROLE:
                continue
            if target_label not in reachable_labels:
                continue
            parsed_target_concentration_mol_m3 = _nonnegative_float(
                target_concentration_mol_m3,
                (
                    f"{trajectory_target.system_id}.{target_label}"
                    ".contact_pair_target_concentration_mol_m3"
                ),
            )
            if parsed_target_concentration_mol_m3 <= 0.0:
                continue
            predicted_concentration_mol_m3 = max(
                concentration_floor_mol_m3,
                _predicted_mapping_value_or_zero(
                    predicted_concentrations,
                    target_label,
                    (
                        f"{trajectory_target.system_id}."
                        "contact_pair_state_concentrations_mol_m3"
                    ),
                ),
            )
            target_weight = max(
                parsed_target_concentration_mol_m3,
                concentration_floor_mol_m3,
            )
            weighted_residual_sum += target_weight * (
                math.log(predicted_concentration_mol_m3)
                - math.log(parsed_target_concentration_mol_m3)
            )
            weight_sum += target_weight
    if weight_sum <= 0.0:
        return _ContactPairLogResidual(
            has_contact_pair_targets=False,
            log_concentration_residual=0.0,
        )
    return _ContactPairLogResidual(
        has_contact_pair_targets=True,
        log_concentration_residual=_finite_float(
            weighted_residual_sum / weight_sum,
            "contact_pair_log_concentration_residual",
        ),
    )


def _best_contact_pair_reachability_point(
    contact_pair_points: tuple[_ContactPairReachabilityPoint, ...],
) -> _ContactPairReachabilityPoint:
    if not contact_pair_points:
        raise ValueError("contact_pair_points must be nonempty")
    best_point = contact_pair_points[0]
    for candidate_point in contact_pair_points[1:]:
        if abs(candidate_point.log_concentration_residual) < abs(
            best_point.log_concentration_residual
        ):
            best_point = candidate_point
    return best_point


def _trajectory_target_label_parts(target_label: str) -> tuple[str, str]:
    label = _nonempty_string(target_label, "trajectory_target_label")
    separator_count = label.count(TRAJECTORY_TARGET_LABEL_SEPARATOR)
    if separator_count != 1:
        raise ValueError(
            "trajectory target label must be formatted as "
            f"transport_role{TRAJECTORY_TARGET_LABEL_SEPARATOR}species_name"
        )
    role_name, species_name = label.split(TRAJECTORY_TARGET_LABEL_SEPARATOR)
    return (
        _nonempty_string(role_name, "trajectory_target_role"),
        _nonempty_string(species_name, "trajectory_target_species_name"),
    )


def _reachable_transport_center_target_labels(
    analytical_result: "AnalyticalConductivityModelResult",
) -> frozenset[str]:
    from conductivity.analytical_conductivity_model import (
        CONTACT_PAIR_CLUSTER_KIND,
        HIGHER_CHARGED_CLUSTER_KIND,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
        NEUTRAL_CLUSTER_KIND,
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
        TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
        TRANSPORT_ROLE_CLUSTER_COM_CENTER,
        TRANSPORT_ROLE_CONTACT_PAIR_CENTER,
        TRANSPORT_ROLE_FREE_ION_CENTER,
        TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    )

    reachable_labels: set[str] = set(
        _transport_center_concentration_targets(analytical_result)
    )
    for species_name, descriptor in analytical_result.descriptors.items():
        if descriptor.charge_number != 0:
            reachable_labels.add(
                _transport_center_target_label(
                    TRANSPORT_ROLE_FREE_ION_CENTER,
                    species_name,
                )
            )
    transport_roles_by_cluster_kind = {
        CONTACT_PAIR_CLUSTER_KIND: (TRANSPORT_ROLE_CONTACT_PAIR_CENTER,),
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND: (
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
        ),
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND: (
            TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
            TRANSPORT_ROLE_CLUSTER_COM_CENTER,
        ),
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND: (
            TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
            TRANSPORT_ROLE_CLUSTER_COM_CENTER,
        ),
        HIGHER_CHARGED_CLUSTER_KIND: (
            TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
            TRANSPORT_ROLE_CLUSTER_COM_CENTER,
        ),
        NEUTRAL_CLUSTER_KIND: (TRANSPORT_ROLE_CLUSTER_COM_CENTER,),
    }
    for cluster_template in analytical_result.cluster_states:
        cluster_kind = _nonempty_string(
            cluster_template.cluster_kind,
            f"{cluster_template.label}.cluster_kind",
        )
        if cluster_kind not in transport_roles_by_cluster_kind:
            raise ValueError(f"unknown analytical cluster kind {cluster_kind}")
        for transport_role in transport_roles_by_cluster_kind[cluster_kind]:
            _add_reachable_stoichiometric_labels(
                reachable_labels,
                transport_role,
                cluster_template.stoichiometry,
            )
    return frozenset(reachable_labels)


def _add_reachable_stoichiometric_labels(
    reachable_labels: set[str],
    transport_role: str,
    stoichiometry: Mapping[str, int],
) -> None:
    for species_name, stoichiometric_count in stoichiometry.items():
        _positive_int(
            stoichiometric_count,
            f"{transport_role}:{species_name}.stoichiometric_count",
        )
        reachable_labels.add(
            _transport_center_target_label(transport_role, species_name)
        )


def _trajectory_concentration_floor_mol_m3(
    analytical_result: "AnalyticalConductivityModelResult",
) -> float:
    from conductivity.analytical_conductivity_model import (
        STANDARD_STATE_CONCENTRATION_MOL_M3,
        TRANSPORT_STATE_CONCENTRATION_RESOLUTION_FACTOR,
    )

    analytical_ion_concentration_mol_m3 = math.fsum(
        _positive_float(
            component.analytical_concentration_M,
            f"{component.species_name}.analytical_concentration_M",
        )
        * STANDARD_STATE_CONCENTRATION_MOL_M3
        for component in analytical_result.speciation.components
    )
    return TRANSPORT_STATE_CONCENTRATION_RESOLUTION_FACTOR * max(
        1.0,
        analytical_ion_concentration_mol_m3,
    )


def print_trajectory_topology_initialization_changes(
    baseline_parameters: ConductivityPrimitiveParameterSet,
    initialized_parameters: ConductivityPrimitiveParameterSet,
) -> None:
    baseline_mapping = conductivity_primitive_parameters_to_mapping(
        baseline_parameters,
    )
    initialized_mapping = conductivity_primitive_parameters_to_mapping(
        initialized_parameters,
    )
    reported_parameter_names = tuple(
        dict.fromkeys(
            parameter_name
            for update_tuple in TRAJECTORY_POPULATION_PARAMETER_UPDATES_BY_ROLE.values()
            for parameter_name, _update_sign in update_tuple
        )
    )
    for parameter_name in reported_parameter_names:
        if parameter_name not in baseline_mapping:
            raise ValueError(f"missing baseline primitive parameter {parameter_name}")
        if parameter_name not in initialized_mapping:
            raise ValueError(f"missing initialized primitive parameter {parameter_name}")
        baseline_value = _finite_float(
            baseline_mapping[parameter_name],
            f"{parameter_name}.baseline",
        )
        initialized_value = _finite_float(
            initialized_mapping[parameter_name],
            f"{parameter_name}.initialized",
        )
        print(
            "trajectory_topology_initialization "
            f"parameter={parameter_name} "
            f"baseline={baseline_value:.6f} "
            f"initialized={initialized_value:.6f}"
        )


def _log_mapping_loss(
    predicted_values_by_label: Mapping[str, float],
    target_values_by_label: Mapping[str, float],
    context: str,
) -> float:
    if not target_values_by_label:
        return 0.0
    positive_reference_values = tuple(
        _positive_float(target_value, f"{context}.{target_label}")
        for target_label, target_value in target_values_by_label.items()
        if target_value > 0.0
    )
    if not positive_reference_values:
        return 0.0
    log_epsilon = min(positive_reference_values) * math.sqrt(np.finfo(float).eps)
    residuals: list[float] = []
    for target_label, target_value in target_values_by_label.items():
        parsed_target_value = _nonnegative_float(
            target_value,
            f"{context}.{target_label}",
        )
        predicted_value = _predicted_mapping_value_or_zero(
            predicted_values_by_label,
            target_label,
            context,
        )
        residuals.append(
            math.log(predicted_value + log_epsilon)
            - math.log(parsed_target_value + log_epsilon)
        )
    return _mean_squared_dimensionless_loss(residuals, context)


def _uncertainty_normalized_mapping_loss(
    predicted_values_by_label: Mapping[str, float],
    target_values_by_label: Mapping[str, float],
    standard_errors_by_label: Mapping[str, float],
    context: str,
) -> float:
    if not target_values_by_label:
        return 0.0
    residuals: list[float] = []
    for target_label, target_value in target_values_by_label.items():
        label = _nonempty_string(target_label, f"{context}.target_label")
        if label not in standard_errors_by_label:
            raise ValueError(f"{context}.{label} is missing block standard error")
        parsed_target_value = _nonnegative_float(
            target_value,
            f"{context}.{label}.target",
        )
        standard_error = _positive_float(
            standard_errors_by_label[label],
            f"{context}.{label}.standard_error",
        )
        predicted_value = _predicted_mapping_value_or_zero(
            predicted_values_by_label,
            label,
            context,
        )
        residuals.append((predicted_value - parsed_target_value) / standard_error)
    return _mean_squared_dimensionless_loss(residuals, context)


def _uncertainty_normalized_scalar_loss(
    predicted_value: float,
    target_value: float,
    standard_error: float,
    context: str,
) -> float:
    parsed_predicted_value = _nonnegative_float(
        predicted_value,
        f"{context}.predicted",
    )
    parsed_target_value = _nonnegative_float(target_value, f"{context}.target")
    parsed_standard_error = _positive_float(
        standard_error,
        f"{context}.standard_error",
    )
    residual = (parsed_predicted_value - parsed_target_value) / parsed_standard_error
    return _mean_squared_dimensionless_loss([residual], context)


def _predicted_mapping_value_or_zero(
    predicted_values_by_label: Mapping[str, float],
    target_label: str,
    context: str,
) -> float:
    label = _nonempty_string(target_label, "target_label")
    if label not in predicted_values_by_label:
        return 0.0
    return _nonnegative_float(predicted_values_by_label[label], f"{context}.{label}")


def _mean_squared_dimensionless_loss(
    residuals: list[float],
    context: str,
) -> float:
    if not residuals:
        return 0.0
    return float(
        math.fsum(
            _finite_float(residual, context) * _finite_float(residual, context)
            for residual in residuals
        )
        / len(residuals)
    )


def _mean_loss(
    losses: list[float],
    context: str,
) -> float:
    if not losses:
        return 0.0
    return float(
        math.fsum(_nonnegative_float(loss, context) for loss in losses)
        / len(losses)
    )


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
    consumed_parameter_name_set = set(_projected_primitive_parameter_fields())
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        if _parameter_is_consumed_by_any_perturbation(
            cases,
            primitive_parameters,
            audit_options,
            baseline_audit_result,
            parameter_name,
            perturbation_scales,
        ):
            consumed_parameter_name_set.add(parameter_name)
    return tuple(
        parameter_name
        for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        if parameter_name in consumed_parameter_name_set
    )


def _projected_primitive_parameter_fields() -> tuple[str, ...]:
    from conductivity.analytical_conductivity_model import (
        CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME,
    )

    missing_parameter_names = tuple(
        parameter_name
        for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        if parameter_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME
    )
    if missing_parameter_names:
        raise ValueError(
            "primitive theorem-role map is missing parameter fields "
            f"{missing_parameter_names}"
        )
    return tuple(
        parameter_name
        for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        if parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME
    )


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
        trajectory_primitive_target_paths=_required_string_tuple_allow_empty(
            fit_config_mapping,
            "trajectory_primitive_target_paths",
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
        progress_output_path=_required_config_string(
            fit_config_mapping,
            "progress_output_path",
        ),
        decomposition_report_output_path=_required_config_string(
            fit_config_mapping,
            "decomposition_report_output_path",
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
    if audit_result.evaluated_rows and not audit_result.proof_statuses:
        raise ValueError("molecular property-DB audit missing proof_statuses")
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


def _required_string_tuple_allow_empty(
    mapping: dict,
    key: str,
) -> tuple[str, ...]:
    if key not in mapping:
        raise ValueError(f"molecular_primitive_parameter_fit.{key} missing")
    raw_values = mapping[key]
    if not isinstance(raw_values, list):
        raise TypeError(f"molecular_primitive_parameter_fit.{key} must be a list")
    parsed_values: list[str] = []
    for value_index, raw_value in enumerate(raw_values):
        value_context = f"molecular_primitive_parameter_fit.{key}[{value_index}]"
        if not isinstance(raw_value, str) or raw_value == "":
            raise ValueError(f"{value_context} must be a nonempty string")
        parsed_values.append(raw_value)
    duplicate_values = tuple(
        sorted(
            value
            for value in set(parsed_values)
            if parsed_values.count(value) > 1
        )
    )
    if duplicate_values:
        raise ValueError(
            f"molecular_primitive_parameter_fit.{key} duplicates {duplicate_values}"
        )
    return tuple(parsed_values)


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
    trajectory_concentration_loss: float
    trajectory_transition_rate_loss: float
    trajectory_displacement_moment_loss: float
    trajectory_sigma_loss_mS_cm: float
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
    sensitivity_values: tuple[float, ...]


@dataclass(frozen=True)
class PrimitivePromotionMetrics:
    mae_mS_cm: float
    bias_mS_cm: float
    pearson_r: float
    worst_abs_residual_mS_cm: float
    proof_statuses: tuple[str, ...]
    failed_rows: int
    trajectory_concentration_unreachable_target_count: int
    trajectory_concentration_under_floor_target_count: int
    maximum_mass_balance_residual: float
    maximum_row_sum_residual: float
    maximum_stationary_residual: float
    maximum_detailed_balance_residual: float
    maximum_event_reversal_residual: float
    zero_charge_sigma_mS_cm: float
    higher_viscosity_lowers_dilute_conductivity: bool
    higher_packing_lowers_local_mobility: bool


@dataclass(frozen=True)
class PrimitiveFitProgressPayload:
    active_coordinate_bounds: tuple[PrimitiveParameterTransform, ...]
    regularization_reference_parameters: ConductivityPrimitiveParameterSet
    latin_hypercube_sample_count: int
    coordinate_step_value: float
    improved_this_round: bool
    powell_cached_candidate_count: int
    promotion_candidate_is_available: bool
    promotion_candidate: PrimitiveFitCandidateResult


def _write_fit_progress_status(
    options: PrimitiveFitOptions,
    stage: str,
    stage_detail: str,
    candidate_results: list[PrimitiveFitCandidateResult],
    current_best: PrimitiveFitCandidateResult,
    progress_payload: PrimitiveFitProgressPayload,
) -> None:
    progress_path = Path(_nonempty_string(
        options.progress_output_path,
        "progress_output_path",
    ))
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    status_mapping = {
        "artifact_type": PRIMITIVE_FIT_PROGRESS_ARTIFACT_TYPE,
        "stage": _nonempty_string(stage, "fit_progress.stage"),
        "stage_detail": _nonempty_string(stage_detail, "fit_progress.stage_detail"),
        "candidate_count": len(candidate_results),
        "accepted_candidate_count": sum(
            1 for candidate_result in candidate_results
            if not candidate_result.rejected
        ),
        "current_best": _candidate_progress_mapping(
            current_best,
            progress_payload,
            options,
        ),
        "progress_values": _fit_progress_values_mapping(progress_payload, options),
    }
    temporary_progress_path = progress_path.with_name(progress_path.name + ".tmp")
    temporary_progress_path.write_text(
        json.dumps(status_mapping, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_progress_path.replace(progress_path)


def _fit_progress_values_mapping(
    progress_payload: PrimitiveFitProgressPayload,
    options: PrimitiveFitOptions,
):
    promotion_candidate = (
        _candidate_progress_mapping(
            progress_payload.promotion_candidate,
            progress_payload,
            options,
        )
        if progress_payload.promotion_candidate_is_available
        else None
    )
    active_parameter_names = _ordered_bound_parameter_names(
        progress_payload.active_coordinate_bounds,
    )
    return {
        "active_parameter_count": len(active_parameter_names),
        "active_parameter_names": active_parameter_names,
        "coordinate_bounds_digest": _coordinate_bounds_digest(
            progress_payload.active_coordinate_bounds,
        ),
        "fit_options_digest": _fit_options_digest(options),
        "regularization_reference_digest": (
            _primitive_parameters_digest(
                progress_payload.regularization_reference_parameters,
            )
        ),
        "latin_hypercube_sample_count": (
            progress_payload.latin_hypercube_sample_count
        ),
        "coordinate_step_value": progress_payload.coordinate_step_value,
        "improved_this_round": progress_payload.improved_this_round,
        "powell_cached_candidate_count": (
            progress_payload.powell_cached_candidate_count
        ),
        "promotion_candidate": promotion_candidate,
    }


def _progress_json_float(
    progress_value: float,
    context: str,
):
    parsed_value = float(progress_value)
    if math.isfinite(parsed_value):
        return parsed_value
    if math.isinf(parsed_value):
        return "inf" if parsed_value > 0.0 else "-inf"
    return "nan"


def _json_digest(mapping, context: str):
    _nonempty_string(context, "json_digest_context")
    digest_payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()


def _coordinate_bounds_digest(
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
) -> str:
    ordered_bounds = _ordered_coordinate_bounds(coordinate_bounds)
    coordinate_bound_mappings = tuple(
        asdict(coordinate_bound) for coordinate_bound in ordered_bounds
    )
    return _json_digest(
        {"coordinate_bounds": coordinate_bound_mappings},
        "coordinate_bounds_digest",
    )


def _fit_options_digest(options: PrimitiveFitOptions) -> str:
    _validate_fit_options(options)
    return _json_digest({"fit_options": asdict(options)}, "fit_options_digest")


def _primitive_parameters_digest(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> str:
    validate_conductivity_primitive_parameters(primitive_parameters)
    primitive_parameter_mapping = conductivity_primitive_parameters_to_mapping(
        primitive_parameters,
    )
    return _json_digest(
        {"primitive_parameters": primitive_parameter_mapping},
        "primitive_parameters_digest",
    )


def _candidate_progress_mapping(
    candidate_result: PrimitiveFitCandidateResult,
    progress_payload: PrimitiveFitProgressPayload,
    options: PrimitiveFitOptions,
):
    active_parameter_names = _ordered_bound_parameter_names(
        progress_payload.active_coordinate_bounds,
    )
    return {
        "objective_value": _progress_json_float(
            candidate_result.objective_value,
            "candidate.objective_value",
        ),
        "mae_mS_cm": _progress_json_float(
            candidate_result.mae_mS_cm,
            "candidate.mae_mS_cm",
        ),
        "bias_mS_cm": _progress_json_float(
            candidate_result.bias_mS_cm,
            "candidate.bias_mS_cm",
        ),
        "pearson_r": _progress_json_float(
            candidate_result.pearson_r,
            "candidate.pearson_r",
        ),
        "worst_abs_residual_mS_cm": _progress_json_float(
            candidate_result.worst_abs_residual_mS_cm,
            "candidate.worst_abs_residual_mS_cm",
        ),
        "trajectory_concentration_loss": _progress_json_float(
            candidate_result.trajectory_concentration_loss,
            "candidate.trajectory_concentration_loss",
        ),
        "trajectory_transition_rate_loss": _progress_json_float(
            candidate_result.trajectory_transition_rate_loss,
            "candidate.trajectory_transition_rate_loss",
        ),
        "trajectory_displacement_moment_loss": _progress_json_float(
            candidate_result.trajectory_displacement_moment_loss,
            "candidate.trajectory_displacement_moment_loss",
        ),
        "trajectory_sigma_loss_mS_cm": _progress_json_float(
            candidate_result.trajectory_sigma_loss_mS_cm,
            "candidate.trajectory_sigma_loss_mS_cm",
        ),
        "failed_rows": candidate_result.failed_rows,
        "rejected": candidate_result.rejected,
        "rejection_reasons": tuple(candidate_result.rejection_reasons),
        "primitive_parameters": conductivity_primitive_parameters_to_mapping(
            candidate_result.primitive_parameters,
        ),
        "coordinate_values": tuple(candidate_result.coordinate_values),
        "coordinate_parameter_names": active_parameter_names,
        "coordinate_bounds_digest": _coordinate_bounds_digest(
            progress_payload.active_coordinate_bounds,
        ),
        "fit_options_digest": _fit_options_digest(options),
        "regularization_reference_digest": (
            _primitive_parameters_digest(
                progress_payload.regularization_reference_parameters,
            )
        ),
    }


def _updated_current_best_candidate(
    candidate_result: PrimitiveFitCandidateResult,
    current_best: PrimitiveFitCandidateResult,
) -> PrimitiveFitCandidateResult:
    if candidate_result.rejected:
        return current_best
    if current_best.rejected:
        return candidate_result
    if _candidate_is_better(candidate_result, current_best):
        return candidate_result
    return current_best


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
    initial_candidate = evaluate_primitive_parameter_candidate(
        initial_coordinate_values,
        initial_parameters,
        regularization_reference_coordinate_values,
        ordered_bounds,
        evaluator,
        options,
    )
    candidate_results.append(initial_candidate)
    current_best_for_progress = initial_candidate
    _write_fit_progress_status(
        options,
        "initial_candidate",
        "initial configured primitive parameter evaluation",
        candidate_results,
        current_best_for_progress,
        PrimitiveFitProgressPayload(
            active_coordinate_bounds=ordered_bounds,
            regularization_reference_parameters=regularization_reference_parameters,
            latin_hypercube_sample_count=0,
            coordinate_step_value=0.0,
            improved_this_round=False,
            powell_cached_candidate_count=0,
            promotion_candidate_is_available=False,
            promotion_candidate=current_best_for_progress,
        ),
    )
    random_number_generator = random.Random(options.random_seed)
    latin_hypercube_sample_count = _fit_budget_count_from_parameter_count(
        len(ordered_bounds),
        options.latin_hypercube_samples_per_parameter,
        "latin_hypercube_samples_per_parameter",
    )
    for sample_index, sample_coordinate_values in enumerate(
        _latin_hypercube_coordinate_values(
            ordered_bounds,
            latin_hypercube_sample_count,
            random_number_generator,
        )
    ):
        sample_candidate = evaluate_primitive_parameter_candidate(
            sample_coordinate_values,
            initial_parameters,
            regularization_reference_coordinate_values,
            ordered_bounds,
            evaluator,
            options,
        )
        candidate_results.append(sample_candidate)
        current_best_for_progress = _updated_current_best_candidate(
            sample_candidate,
            current_best_for_progress,
        )
        _write_fit_progress_status(
            options,
            "latin_hypercube",
            f"sample {sample_index + 1} of {latin_hypercube_sample_count}",
            candidate_results,
            current_best_for_progress,
            PrimitiveFitProgressPayload(
                active_coordinate_bounds=ordered_bounds,
                regularization_reference_parameters=regularization_reference_parameters,
                latin_hypercube_sample_count=latin_hypercube_sample_count,
                coordinate_step_value=0.0,
                improved_this_round=False,
                powell_cached_candidate_count=0,
                promotion_candidate_is_available=False,
                promotion_candidate=current_best_for_progress,
            ),
        )

    current_best = _best_accepted_candidate(candidate_results)
    _write_fit_progress_status(
        options,
        "latin_hypercube_complete",
        "latin hypercube candidate generation complete",
        candidate_results,
        current_best,
        PrimitiveFitProgressPayload(
            active_coordinate_bounds=ordered_bounds,
            regularization_reference_parameters=regularization_reference_parameters,
            latin_hypercube_sample_count=latin_hypercube_sample_count,
            coordinate_step_value=0.0,
            improved_this_round=False,
            powell_cached_candidate_count=0,
            promotion_candidate_is_available=False,
            promotion_candidate=current_best,
        ),
    )
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
                _write_fit_progress_status(
                    options,
                    "coordinate_search",
                    (
                        f"round {search_round_index + 1} of "
                        f"{options.coordinate_search_rounds}, "
                        f"parameter {parameter_index + 1} of {len(ordered_bounds)}, "
                        f"step_sign {step_sign:.0f}"
                    ),
                    candidate_results,
                    current_best,
                    PrimitiveFitProgressPayload(
                        active_coordinate_bounds=ordered_bounds,
                        regularization_reference_parameters=(
                            regularization_reference_parameters
                        ),
                        latin_hypercube_sample_count=latin_hypercube_sample_count,
                        coordinate_step_value=coordinate_step_value,
                        improved_this_round=improved_this_round,
                        powell_cached_candidate_count=0,
                        promotion_candidate_is_available=False,
                        promotion_candidate=current_best,
                    ),
                )
        if not improved_this_round:
            coordinate_step_value *= options.coordinate_step_shrinkage
        _write_fit_progress_status(
            options,
            "coordinate_search_round_complete",
            f"round {search_round_index + 1} of {options.coordinate_search_rounds}",
            candidate_results,
            current_best,
            PrimitiveFitProgressPayload(
                active_coordinate_bounds=ordered_bounds,
                regularization_reference_parameters=regularization_reference_parameters,
                latin_hypercube_sample_count=latin_hypercube_sample_count,
                coordinate_step_value=coordinate_step_value,
                improved_this_round=improved_this_round,
                powell_cached_candidate_count=0,
                promotion_candidate_is_available=False,
                promotion_candidate=current_best,
            ),
        )
        if search_round_index == options.coordinate_search_rounds - 1:
            break

    _write_fit_progress_status(
        options,
        "powell_local_polish_start",
        "starting Powell local polish",
        candidate_results,
        current_best,
        PrimitiveFitProgressPayload(
            active_coordinate_bounds=ordered_bounds,
            regularization_reference_parameters=regularization_reference_parameters,
            latin_hypercube_sample_count=latin_hypercube_sample_count,
            coordinate_step_value=coordinate_step_value,
            improved_this_round=False,
            powell_cached_candidate_count=0,
            promotion_candidate_is_available=False,
            promotion_candidate=current_best,
        ),
    )
    current_best = _run_powell_local_polish(
        current_best,
        candidate_results,
        initial_parameters,
        regularization_reference_parameters,
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
    _write_fit_progress_status(
        options,
        "fit_complete",
        "primitive fit candidate search complete",
        candidate_results,
        current_best,
        PrimitiveFitProgressPayload(
            active_coordinate_bounds=ordered_bounds,
            regularization_reference_parameters=regularization_reference_parameters,
            latin_hypercube_sample_count=latin_hypercube_sample_count,
            coordinate_step_value=coordinate_step_value,
            improved_this_round=False,
            powell_cached_candidate_count=0,
            promotion_candidate_is_available=True,
            promotion_candidate=promotion_candidate,
        ),
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
    trajectory_concentration_loss = _nonnegative_float(
        evaluation.trajectory_concentration_loss,
        "trajectory_concentration_loss",
    )
    trajectory_transition_rate_loss = _nonnegative_float(
        evaluation.trajectory_transition_rate_loss,
        "trajectory_transition_rate_loss",
    )
    trajectory_displacement_moment_loss = _nonnegative_float(
        evaluation.trajectory_displacement_moment_loss,
        "trajectory_displacement_moment_loss",
    )
    trajectory_sigma_loss_mS_cm = _nonnegative_float(
        evaluation.trajectory_sigma_loss_mS_cm,
        "trajectory_sigma_loss_mS_cm",
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
            + trajectory_concentration_loss
            + trajectory_transition_rate_loss
            + trajectory_displacement_moment_loss
            + trajectory_sigma_loss_mS_cm
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
        trajectory_concentration_loss=trajectory_concentration_loss,
        trajectory_transition_rate_loss=trajectory_transition_rate_loss,
        trajectory_displacement_moment_loss=trajectory_displacement_moment_loss,
        trajectory_sigma_loss_mS_cm=trajectory_sigma_loss_mS_cm,
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
        trajectory_concentration_loss=math.inf,
        trajectory_transition_rate_loss=math.inf,
        trajectory_displacement_moment_loss=math.inf,
        trajectory_sigma_loss_mS_cm=math.inf,
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
    regularization_reference_parameters: ConductivityPrimitiveParameterSet,
    regularization_reference_coordinate_values: tuple[float, ...],
    ordered_bounds: tuple[PrimitiveParameterTransform, ...],
    evaluator: ConductivityPrimitiveParameterEvaluator,
    options: PrimitiveFitOptions,
) -> PrimitiveFitCandidateResult:
    validate_conductivity_primitive_parameters(regularization_reference_parameters)
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
        nonlocal current_best
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
            current_best = _updated_current_best_candidate(
                candidate_result,
                current_best,
            )
            _write_fit_progress_status(
                options,
                "powell_local_polish",
                f"objective evaluation {len(evaluation_cache)}",
                candidate_results,
                current_best,
                PrimitiveFitProgressPayload(
                    active_coordinate_bounds=ordered_bounds,
                    regularization_reference_parameters=regularization_reference_parameters,
                    latin_hypercube_sample_count=0,
                    coordinate_step_value=0.0,
                    improved_this_round=False,
                    powell_cached_candidate_count=len(evaluation_cache),
                    promotion_candidate_is_available=False,
                    promotion_candidate=current_best,
                ),
            )
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
    trajectory_primitive_calibration_targets: tuple[
        TrajectoryPrimitiveCalibrationTarget,
        ...,
    ],
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
        trajectory_primitive_calibration_targets,
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
            key=_absolute_residual_sort_key,
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
    for target_path in options.trajectory_primitive_target_paths:
        _nonempty_string(target_path, "trajectory_primitive_target_path")
    _nonempty_string(options.progress_output_path, "progress_output_path")
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
    if (
        0 < evaluation.trajectory_concentration_unreachable_target_count
        and options.trajectory_primitive_target_paths
    ):
        reasons.append("trajectory_c_unreachable")
    if (
        0 < evaluation.trajectory_concentration_under_floor_target_count
        and options.trajectory_primitive_target_paths
    ):
        reasons.append("trajectory_c_under_floor")
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


def _absolute_residual_sort_key(
    row_result: "MolecularPropertyDbRowResult",
) -> float:
    return abs(_finite_float(row_result.residual_mS_cm, "residual_mS_cm"))


def _indexed_absolute_residual_sort_key(
    index_and_row_result: tuple[int, "MolecularPropertyDbRowResult"],
) -> float:
    row_result = index_and_row_result[1]
    return _absolute_residual_sort_key(row_result)


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
        key=_candidate_objective_sort_key,
    )


def _candidate_is_better(
    trial_result: PrimitiveFitCandidateResult,
    current_best: PrimitiveFitCandidateResult,
) -> bool:
    if trial_result.rejected:
        return False
    return trial_result.objective_value < current_best.objective_value


def _candidate_objective_sort_key(
    candidate_result: PrimitiveFitCandidateResult,
) -> float:
    return candidate_result.objective_value


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
    return _minimum_promotion_candidate(
        accepted_candidates,
        baseline_candidate,
        options,
    )


def _minimum_promotion_candidate(
    accepted_candidates: tuple[PrimitiveFitCandidateResult, ...],
    baseline_candidate: PrimitiveFitCandidateResult,
    options: PrimitiveFitOptions,
) -> PrimitiveFitCandidateResult:
    best_candidate = accepted_candidates[0]
    best_sort_key = _promotion_candidate_sort_key(
        best_candidate,
        baseline_candidate,
        options,
    )
    for candidate_result in accepted_candidates[1:]:
        candidate_sort_key = _promotion_candidate_sort_key(
            candidate_result,
            baseline_candidate,
            options,
        )
        if candidate_sort_key < best_sort_key:
            best_candidate = candidate_result
            best_sort_key = candidate_sort_key
    return best_candidate


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
    baseline_descriptor_targets = _validated_descriptor_calibration_targets(
        baseline_evaluation.descriptor_calibration_targets,
    )
    baseline_sensitivity_values = _prediction_sensitivity_values(
        baseline_evaluation,
        options,
        "baseline_prediction_sensitivity",
    )
    sensitivity_weights = _prediction_sensitivity_weights(
        baseline_descriptor_targets,
        options,
        "baseline_prediction_sensitivity",
    )
    if len(baseline_sensitivity_values) != len(sensitivity_weights):
        raise ValueError("baseline sensitivity value and weight counts must match")
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
                tuple(0.0 for _value in baseline_sensitivity_values)
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
            baseline_sensitivity_values,
            "minus_prediction_sensitivity",
        )
        _validate_prediction_sensitivity_trial_shape(
            plus_trial,
            baseline_sensitivity_values,
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
                    (plus_value - minus_value) / coordinate_delta
                    for minus_value, plus_value in zip(
                        minus_trial.sensitivity_values,
                        plus_trial.sensitivity_values,
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
                    (plus_value - baseline_value) / coordinate_delta
                    for baseline_value, plus_value in zip(
                        baseline_sensitivity_values,
                        plus_trial.sensitivity_values,
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
                    (baseline_value - minus_value) / coordinate_delta
                    for baseline_value, minus_value in zip(
                        baseline_sensitivity_values,
                        minus_trial.sensitivity_values,
                    )
                )
            )
            continue
        sensitivity_columns.append(
            tuple(0.0 for _value in baseline_sensitivity_values)
        )
    sensitivity_matrix = np.asarray(sensitivity_columns, dtype=float).T
    if not np.all(np.isfinite(sensitivity_matrix)):
        raise ValueError("prediction sensitivity matrix must be finite")
    weighted_sensitivity_matrix = (
        np.sqrt(np.asarray(sensitivity_weights, dtype=float))[:, None]
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
        sensitivity_values = _prediction_sensitivity_values(
            trial_evaluation,
            options,
            "trial_prediction_sensitivity",
        )
    except (
        FloatingPointError,
        OverflowError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return PrimitivePredictionSensitivityTrial(
            valid=False,
            sensitivity_values=tuple(),
        )
    return PrimitivePredictionSensitivityTrial(
        valid=True,
        sensitivity_values=sensitivity_values,
    )


def _validate_prediction_sensitivity_trial_shape(
    trial: PrimitivePredictionSensitivityTrial,
    baseline_sensitivity_values: tuple[float, ...],
    context: str,
) -> None:
    if not trial.valid:
        return
    if len(trial.sensitivity_values) != len(baseline_sensitivity_values):
        raise ValueError(f"{context} sensitivity value count must match baseline")


def _prediction_sensitivity_values(
    evaluation: PrimitiveFitDatasetEvaluation,
    options: PrimitiveFitOptions,
    context: str,
) -> tuple[float, ...]:
    _validate_fit_options(options)
    context_text = _nonempty_string(context, "prediction_sensitivity_context")
    sensitivity_values: list[float] = list(
        _validated_sigma_tuple(
            evaluation.predicted_sigmas_mS_cm,
            f"{context_text}.predicted_sigmas_mS_cm",
        )
    )
    if options.trajectory_primitive_target_paths:
        sensitivity_values.append(
            _nonnegative_float(
                evaluation.trajectory_concentration_loss,
                f"{context_text}.trajectory_concentration_loss",
            )
        )
        sensitivity_values.append(
            _nonnegative_float(
                evaluation.trajectory_transition_rate_loss,
                f"{context_text}.trajectory_transition_rate_loss",
            )
        )
        sensitivity_values.append(
            _nonnegative_float(
                evaluation.trajectory_displacement_moment_loss,
                f"{context_text}.trajectory_displacement_moment_loss",
            )
        )
        sensitivity_values.append(
            _nonnegative_float(
                evaluation.trajectory_sigma_loss_mS_cm,
                f"{context_text}.trajectory_sigma_loss_mS_cm",
            )
        )
    return tuple(sensitivity_values)


def _prediction_sensitivity_weights(
    descriptor_targets: tuple[DescriptorCalibrationTarget, ...],
    options: PrimitiveFitOptions,
    context: str,
) -> tuple[float, ...]:
    _validate_fit_options(options)
    context_text = _nonempty_string(context, "prediction_sensitivity_context")
    sensitivity_weights: list[float] = [
        _positive_float(
            descriptor_target.residual_weight,
            f"{context_text}.descriptor_residual_weight",
        )
        for descriptor_target in descriptor_targets
    ]
    if options.trajectory_primitive_target_paths:
        sensitivity_weights.extend((1.0, 1.0, 1.0, 1.0))
    return tuple(
        _positive_float(sensitivity_weight, f"{context_text}.sensitivity_weight")
        for sensitivity_weight in sensitivity_weights
    )


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
    from conductivity.analytical_conductivity_model import (
        PROJECTED_READOUT_PROVEN_DESCRIPTOR_CLOSURE_EMPIRICAL,
    )

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
    if (
        PROJECTED_READOUT_PROVEN_DESCRIPTOR_CLOSURE_EMPIRICAL
        in candidate_metrics.proof_statuses
    ):
        reasons.append("descriptor_closure_empirical")
    if 0 < candidate_metrics.trajectory_concentration_unreachable_target_count:
        reasons.append("trajectory_c_unreachable")
    if 0 < candidate_metrics.trajectory_concentration_under_floor_target_count:
        reasons.append("trajectory_c_under_floor")
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


def write_default_calibration_error_decomposition_report(
    report_path: str,
) -> None:
    from data.electrolyte_property_db import DATA
    from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
    from conductivity.molecular_property_db_audit import (
        MolecularPropertyDbRegistrySource,
        build_molecular_property_db_case_selection,
        configured_conductivity_primitive_parameters,
        default_molecular_property_db_audit_options,
    )

    output_path = Path(_nonempty_string(report_path, "report_path"))
    audit_options = default_molecular_property_db_audit_options()
    fit_options, coordinate_bounds = default_molecular_primitive_fit_configuration()
    trajectory_targets = load_trajectory_primitive_calibration_targets(
        fit_options.trajectory_primitive_target_paths,
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
    configured_parameters = configured_conductivity_primitive_parameters()
    topology_initialized_parameters = (
        initialize_topology_logk_offsets_from_trajectory_concentrations(
            configured_parameters,
            trajectory_targets,
            audit_options,
            coordinate_bounds,
            fit_options,
        )
    )
    snapshot_parameter_sets = [
        ("configured_baseline", configured_parameters),
        ("topology_initialized", topology_initialized_parameters),
    ]
    candidate_path = Path(fit_options.candidate_output_path)
    if candidate_path.exists():
        snapshot_parameter_sets.append(
            (
                "completed_candidate",
                load_primitive_parameters_from_candidate_artifact(
                    fit_options.candidate_output_path,
                ),
            )
        )
    write_calibration_error_decomposition_report(
        str(output_path),
        tuple(snapshot_parameter_sets),
        case_selection.cases,
        audit_options,
        fit_options,
        trajectory_targets,
    )


def write_default_topology_reachability_sweep_report(
    report_path: str,
    contact_pair_logk_offsets: tuple[float, ...],
    solvent_separated_pair_logk_offsets: tuple[float, ...],
    higher_charged_cluster_logk_offsets: tuple[float, ...],
) -> None:
    from data.electrolyte_property_db import DATA
    from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
    from conductivity.molecular_property_db_audit import (
        MolecularPropertyDbRegistrySource,
        build_molecular_property_db_case_selection,
        configured_conductivity_primitive_parameters,
        default_molecular_property_db_audit_options,
    )

    output_path = Path(_nonempty_string(report_path, "report_path"))
    audit_options = default_molecular_property_db_audit_options()
    fit_options, _coordinate_bounds = default_molecular_primitive_fit_configuration()
    trajectory_targets = load_trajectory_primitive_calibration_targets(
        fit_options.trajectory_primitive_target_paths,
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
    report_mapping = topology_reachability_sweep_report_mapping(
        configured_conductivity_primitive_parameters(),
        case_selection.cases,
        audit_options,
        fit_options,
        trajectory_targets,
        contact_pair_logk_offsets,
        solvent_separated_pair_logk_offsets,
        higher_charged_cluster_logk_offsets,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def topology_reachability_sweep_report_mapping(
    base_parameters: ConductivityPrimitiveParameterSet,
    cases: tuple["MolecularPropertyDbCase", ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    fit_options: PrimitiveFitOptions,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    contact_pair_logk_offsets: tuple[float, ...],
    solvent_separated_pair_logk_offsets: tuple[float, ...],
    higher_charged_cluster_logk_offsets: tuple[float, ...],
) -> dict:
    contact_offsets = _nonnegative_float_tuple(
        contact_pair_logk_offsets,
        "contact_pair_logk_offsets",
    )
    ssip_offsets = _nonnegative_float_tuple(
        solvent_separated_pair_logk_offsets,
        "solvent_separated_pair_logk_offsets",
    )
    higher_offsets = _nonnegative_float_tuple(
        higher_charged_cluster_logk_offsets,
        "higher_charged_cluster_logk_offsets",
    )
    sweep_entries = []
    for contact_pair_logk_offset in contact_offsets:
        for ssip_logk_offset in ssip_offsets:
            for higher_charged_cluster_logk_offset in higher_offsets:
                sweep_parameters = _sweep_parameters_with_topology_offsets(
                    base_parameters,
                    contact_pair_logk_offset,
                    ssip_logk_offset,
                    higher_charged_cluster_logk_offset,
                )
                sweep_entries.append(
                    _topology_reachability_sweep_entry_mapping(
                        sweep_parameters,
                        cases,
                        audit_options,
                        fit_options,
                        trajectory_targets,
                        contact_pair_logk_offset,
                        ssip_logk_offset,
                        higher_charged_cluster_logk_offset,
                    )
                )
    return {
        "artifact_type": "molecular_conductivity_topology_reachability_sweep",
        "contact_pair_logk_offsets": contact_offsets,
        "solvent_separated_pair_logk_offsets": ssip_offsets,
        "higher_charged_cluster_logk_offsets": higher_offsets,
        "sweep_entry_count": len(sweep_entries),
        "sweep_entries": tuple(sweep_entries),
    }


def _sweep_parameters_with_topology_offsets(
    base_parameters: ConductivityPrimitiveParameterSet,
    contact_pair_logk_offset: float,
    solvent_separated_pair_logk_offset: float,
    higher_charged_cluster_logk_offset: float,
) -> ConductivityPrimitiveParameterSet:
    validate_conductivity_primitive_parameters(base_parameters)
    parameter_mapping = conductivity_primitive_parameters_to_mapping(base_parameters)
    parameter_mapping["contact_pair_logK_offset"] = _nonnegative_float(
        contact_pair_logk_offset,
        "contact_pair_logk_offset",
    )
    parameter_mapping["solvent_separated_pair_logK_offset"] = _nonnegative_float(
        solvent_separated_pair_logk_offset,
        "solvent_separated_pair_logk_offset",
    )
    parameter_mapping["higher_charged_cluster_logK_offset"] = _nonnegative_float(
        higher_charged_cluster_logk_offset,
        "higher_charged_cluster_logk_offset",
    )
    sweep_parameters = conductivity_primitive_parameters_from_mapping(
        parameter_mapping,
    )
    validate_conductivity_primitive_parameters(sweep_parameters)
    return sweep_parameters


def _topology_reachability_sweep_entry_mapping(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    cases: tuple["MolecularPropertyDbCase", ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    fit_options: PrimitiveFitOptions,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    contact_pair_logk_offset: float,
    solvent_separated_pair_logk_offset: float,
    higher_charged_cluster_logk_offset: float,
) -> dict:
    from conductivity.molecular_property_db_audit import (
        audit_molecular_property_db_cases,
    )

    audit_result = audit_molecular_property_db_cases(
        cases,
        primitive_parameters,
        audit_options,
    )
    trajectory_losses = _trajectory_primitive_loss_breakdown(
        trajectory_targets,
        primitive_parameters,
        audit_options,
        fit_options,
    )
    coverages = trajectory_concentration_target_coverage(
        trajectory_targets,
        primitive_parameters,
        audit_options,
    )
    trajectory_system_mappings = tuple(
        _topology_sweep_trajectory_system_mapping(
            trajectory_target,
            primitive_parameters,
            audit_options,
        )
        for trajectory_target in trajectory_targets
    )
    return {
        "sweep_parameters": {
            "contact_pair_logK_offset": contact_pair_logk_offset,
            "solvent_separated_pair_logK_offset": solvent_separated_pair_logk_offset,
            "higher_charged_cluster_logK_offset": higher_charged_cluster_logk_offset,
        },
        "property_metrics": {
            "mae_mS_cm": audit_result.mae_mS_cm,
            "bias_mS_cm": audit_result.bias_mS_cm,
            "pearson_r": audit_result.pearson_r,
            "worst_abs_residual_mS_cm": audit_result.maximum_abs_residual_mS_cm,
            "direct_sigma_mS_cm_mean": _mean_row_value(
                tuple(row.direct_sigma_mS_cm for row in audit_result.rows),
                "direct_sigma_mS_cm",
            ),
            "corrector_sigma_mS_cm_mean": _mean_row_value(
                tuple(row.corrector_sigma_mS_cm for row in audit_result.rows),
                "corrector_sigma_mS_cm",
            ),
        },
        "trajectory_losses": {
            "concentration_loss": trajectory_losses.concentration_loss,
            "transition_rate_loss": trajectory_losses.transition_rate_loss,
            "displacement_moment_loss": trajectory_losses.displacement_moment_loss,
            "sigma_loss_mS_cm": trajectory_losses.sigma_loss_mS_cm,
        },
        "direct_corrector_failure_counts": {
            "direct_capacity_failure_count": sum(
                1 for row in audit_result.rows if row.direct_capacity_failure
            ),
            "corrector_too_strong_failure_count": sum(
                1 for row in audit_result.rows if row.corrector_too_strong_failure
            ),
            "corrector_too_weak_failure_count": sum(
                1 for row in audit_result.rows if row.corrector_too_weak_failure
            ),
        },
        "trajectory_coverage": tuple(
            _trajectory_coverage_mapping(coverage) for coverage in coverages
        ),
        "trajectory_systems": trajectory_system_mappings,
    }


def _mean_row_value(values: tuple[float, ...], context: str) -> float:
    if not values:
        raise ValueError(f"{context} must contain at least one value")
    return float(
        math.fsum(_finite_float(value, context) for value in values)
        / len(values)
    )


def _topology_sweep_trajectory_system_mapping(
    trajectory_target: TrajectoryPrimitiveCalibrationTarget,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
) -> dict:
    analytical_result = _trajectory_target_analytical_result(
        trajectory_target,
        primitive_parameters,
        audit_options,
    )
    return {
        "system_id": trajectory_target.system_id,
        "sigma_mS_cm": analytical_result.sigma_mS_cm,
        "direct_sigma_mS_cm": (
            analytical_result.markov_additive_result.direct_sigma_mS_cm
        ),
        "corrector_sigma_mS_cm": (
            analytical_result.markov_additive_result.corrector_sigma_mS_cm
        ),
        "free_ion_fraction": _trajectory_free_ion_fraction(analytical_result),
        "charged_cluster_fraction": _trajectory_cluster_fraction_by_charge(
            analytical_result,
            "charged_cluster_fraction",
            True,
        ),
        "neutral_cluster_fraction": _trajectory_cluster_fraction_by_charge(
            analytical_result,
            "neutral_cluster_fraction",
            False,
        ),
        "top_clusters_by_concentration": tuple(
            _cluster_thermodynamic_mapping(cluster_mapping)
            for cluster_mapping in tuple(
                sorted(
                    _trajectory_cluster_thermodynamic_diagnostics(
                        trajectory_target,
                        analytical_result,
                        primitive_parameters,
                    ),
                    key=_cluster_concentration_sort_key,
                    reverse=True,
                )
            )[: audit_options.max_cluster_ion_count]
        ),
        "top_clusters_by_favorable_free_energy": tuple(
            _cluster_thermodynamic_mapping(cluster_mapping)
            for cluster_mapping in tuple(
                sorted(
                    _trajectory_cluster_thermodynamic_diagnostics(
                        trajectory_target,
                        analytical_result,
                        primitive_parameters,
                    ),
                    key=_cluster_free_energy_sort_key,
                )
            )[: audit_options.max_cluster_ion_count]
        ),
    }


def _trajectory_free_ion_fraction(
    analytical_result: "AnalyticalConductivityModelResult",
) -> float:
    total_ion_concentration_mol_m3 = _trajectory_total_ion_concentration_mol_m3(
        analytical_result,
    )
    free_ion_concentration_mol_m3 = math.fsum(
        analytical_result.speciation.free_component_concentrations_mol_m3[
            component.species_name
        ]
        for component in analytical_result.speciation.components
    )
    return free_ion_concentration_mol_m3 / total_ion_concentration_mol_m3


def _trajectory_cluster_fraction_by_charge(
    analytical_result: "AnalyticalConductivityModelResult",
    context: str,
    charged: bool,
) -> float:
    total_ion_concentration_mol_m3 = _trajectory_total_ion_concentration_mol_m3(
        analytical_result,
    )
    selected_concentration_mol_m3 = 0.0
    for cluster_template in analytical_result.speciation.cluster_templates:
        if (cluster_template.net_charge_number != 0) != charged:
            continue
        cluster_concentration_mol_m3 = (
            analytical_result.speciation.cluster_concentrations_mol_m3[
                cluster_template.label
            ]
        )
        selected_concentration_mol_m3 += (
            math.fsum(cluster_template.stoichiometry.values())
            * cluster_concentration_mol_m3
        )
    return _nonnegative_float(
        selected_concentration_mol_m3 / total_ion_concentration_mol_m3,
        context,
    )


def _trajectory_total_ion_concentration_mol_m3(
    analytical_result: "AnalyticalConductivityModelResult",
) -> float:
    from conductivity.analytical_conductivity_model import (
        STANDARD_STATE_CONCENTRATION_MOL_M3,
    )

    return _positive_float(
        math.fsum(
            component.analytical_concentration_M
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            for component in analytical_result.speciation.components
        ),
        "trajectory_total_ion_concentration_mol_m3",
    )


def _trajectory_cluster_thermodynamic_diagnostics(
    trajectory_target: TrajectoryPrimitiveCalibrationTarget,
    analytical_result: "AnalyticalConductivityModelResult",
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> tuple["MolecularClusterThermodynamicDiagnostic", ...]:
    from constants import R
    from conductivity.analytical_conductivity_model import (
        cluster_activity_correction_J_mol,
    )
    from conductivity.molecular_property_db_audit import (
        MolecularClusterThermodynamicDiagnostic,
    )

    total_ion_concentration_mol_m3 = _trajectory_total_ion_concentration_mol_m3(
        analytical_result,
    )
    diagnostics: list[MolecularClusterThermodynamicDiagnostic] = []
    for cluster_template in analytical_result.speciation.cluster_templates:
        cluster_concentration_mol_m3 = (
            analytical_result.speciation.cluster_concentrations_mol_m3[
                cluster_template.label
            ]
        )
        cluster_ion_count = math.fsum(cluster_template.stoichiometry.values())
        standard_free_energy_over_RT = (
            cluster_template.standard_free_energy_J_mol
            / (R * trajectory_target.recipe.temperature_K)
        )
        diagnostics.append(
            MolecularClusterThermodynamicDiagnostic(
                row_id=0,
                cluster_label=cluster_template.label,
                cluster_kind=cluster_template.cluster_kind,
                stoichiometry=dict(cluster_template.stoichiometry),
                net_charge_number=cluster_template.net_charge_number,
                concentration_mol_m3=cluster_concentration_mol_m3,
                concentration_fraction_of_total_ion=(
                    cluster_concentration_mol_m3
                    * cluster_ion_count
                    / total_ion_concentration_mol_m3
                ),
                standard_free_energy_J_mol=(
                    cluster_template.standard_free_energy_J_mol
                ),
                standard_free_energy_over_RT=standard_free_energy_over_RT,
                log_equilibrium_constant=-standard_free_energy_over_RT,
                coulomb_J_mol=cluster_template.coulomb_J_mol,
                desolvation_J_mol=cluster_template.desolvation_J_mol,
                coordination_J_mol=cluster_template.coordination_J_mol,
                steric_J_mol=cluster_template.steric_J_mol,
                entropy_J_mol=cluster_template.entropy_J_mol,
                activity_reference_J_mol=(
                    cluster_template.activity_reference_J_mol
                ),
                activity_correction_J_mol=cluster_activity_correction_J_mol(
                    analytical_result.speciation.components,
                    cluster_template,
                    analytical_result.speciation.free_component_concentrations_mol_m3,
                    analytical_result.solvent_environment,
                    primitive_parameters,
                ),
                hydrodynamic_radius_A=cluster_template.hydrodynamic_radius_A,
                molecular_volume_A3=cluster_template.molecular_volume_A3,
            )
        )
    return tuple(diagnostics)


def load_progress_checkpoint_primitive_parameters(
    progress_path: str,
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...],
    options: PrimitiveFitOptions,
    regularization_reference_parameters: ConductivityPrimitiveParameterSet,
) -> ConductivityPrimitiveParameterSet:
    progress_path_text = _nonempty_string(progress_path, "progress_path")
    checkpoint_mapping = json.loads(
        Path(progress_path_text).read_text(encoding="utf-8")
    )
    if not isinstance(checkpoint_mapping, dict):
        raise TypeError("fit progress checkpoint must contain a JSON object")
    _required_checkpoint_value(
        checkpoint_mapping,
        "artifact_type",
        "fit progress checkpoint",
    )
    if checkpoint_mapping["artifact_type"] != PRIMITIVE_FIT_PROGRESS_ARTIFACT_TYPE:
        raise ValueError(
            "fit progress checkpoint artifact_type must be "
            f"{PRIMITIVE_FIT_PROGRESS_ARTIFACT_TYPE}"
        )
    current_best_mapping = _required_checkpoint_mapping(
        checkpoint_mapping,
        "current_best",
        "fit progress checkpoint",
    )
    _validate_checkpoint_digest(
        current_best_mapping,
        "coordinate_bounds_digest",
        _coordinate_bounds_digest(coordinate_bounds),
    )
    _validate_checkpoint_digest(
        current_best_mapping,
        "fit_options_digest",
        _fit_options_digest(options),
    )
    _validate_checkpoint_digest(
        current_best_mapping,
        "regularization_reference_digest",
        _primitive_parameters_digest(regularization_reference_parameters),
    )
    primitive_parameter_mapping = _required_checkpoint_mapping(
        current_best_mapping,
        "primitive_parameters",
        "fit progress current_best",
    )
    primitive_parameters = conductivity_primitive_parameters_from_mapping(
        primitive_parameter_mapping,
    )
    validate_conductivity_primitive_parameters(primitive_parameters)
    return primitive_parameters


def _required_checkpoint_mapping(
    mapping: dict,
    key: str,
    context: str,
) -> dict:
    value = _required_checkpoint_value(mapping, key, context)
    if not isinstance(value, dict):
        raise TypeError(f"{context}.{key} must be an object")
    return value


def _required_checkpoint_value(
    mapping: dict,
    key: str,
    context: str,
):
    _nonempty_string(key, "checkpoint_key")
    _nonempty_string(context, "checkpoint_context")
    if key not in mapping:
        raise ValueError(f"{context} missing {key}")
    return mapping[key]


def _validate_checkpoint_digest(
    mapping: dict,
    key: str,
    expected_digest: str,
) -> None:
    observed_digest_value = _required_checkpoint_value(
        mapping,
        key,
        "fit progress current_best",
    )
    if not isinstance(observed_digest_value, str):
        raise TypeError(f"fit progress current_best.{key} must be a string")
    expected_digest_text = _nonempty_string(expected_digest, f"{key}.expected")
    if observed_digest_value != expected_digest_text:
        raise ValueError(
            f"fit progress current_best.{key} does not match current run"
        )


def calibration_error_decomposition_report_mapping(
    snapshot_parameter_sets: tuple[
        tuple[str, ConductivityPrimitiveParameterSet],
        ...,
    ],
    cases: tuple["MolecularPropertyDbCase", ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    fit_options: PrimitiveFitOptions,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
) -> dict:
    if not snapshot_parameter_sets:
        raise ValueError("calibration decomposition report requires snapshots")
    snapshots = []
    for snapshot_name, primitive_parameters in snapshot_parameter_sets:
        snapshots.append(
            _calibration_snapshot_decomposition_mapping(
                snapshot_name,
                primitive_parameters,
                cases,
                audit_options,
                fit_options,
                trajectory_targets,
            )
        )
    return {
        "artifact_type": "molecular_conductivity_calibration_error_decomposition",
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }


def write_calibration_error_decomposition_report(
    report_path: str,
    snapshot_parameter_sets: tuple[
        tuple[str, ConductivityPrimitiveParameterSet],
        ...,
    ],
    cases: tuple["MolecularPropertyDbCase", ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    fit_options: PrimitiveFitOptions,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
) -> None:
    output_path = Path(_nonempty_string(report_path, "report_path"))
    report_mapping = calibration_error_decomposition_report_mapping(
        snapshot_parameter_sets,
        cases,
        audit_options,
        fit_options,
        trajectory_targets,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _calibration_snapshot_decomposition_mapping(
    snapshot_name: str,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    cases: tuple["MolecularPropertyDbCase", ...],
    audit_options: "MolecularPropertyDbAuditOptions",
    fit_options: PrimitiveFitOptions,
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
) -> dict:
    from conductivity.molecular_property_db_audit import (
        audit_molecular_property_db_cases,
    )

    snapshot_name_text = _nonempty_string(snapshot_name, "snapshot_name")
    validate_conductivity_primitive_parameters(primitive_parameters)
    audit_result = audit_molecular_property_db_cases(
        cases,
        primitive_parameters,
        audit_options,
    )
    trajectory_losses = _trajectory_primitive_loss_breakdown(
        trajectory_targets,
        primitive_parameters,
        audit_options,
        fit_options,
    )
    coverages = trajectory_concentration_target_coverage(
        trajectory_targets,
        primitive_parameters,
        audit_options,
    )
    return {
        "snapshot_name": snapshot_name_text,
        "primitive_parameters_digest": _primitive_parameters_digest(
            primitive_parameters,
        ),
        "property_metrics": {
            "mae_mS_cm": audit_result.mae_mS_cm,
            "bias_mS_cm": audit_result.bias_mS_cm,
            "pearson_r": audit_result.pearson_r,
            "worst_abs_residual_mS_cm": audit_result.maximum_abs_residual_mS_cm,
            "failed_rows": audit_result.failed_rows,
        },
        "trajectory_losses": {
            "concentration_loss": trajectory_losses.concentration_loss,
            "transition_rate_loss": trajectory_losses.transition_rate_loss,
            "displacement_moment_loss": trajectory_losses.displacement_moment_loss,
            "sigma_loss_mS_cm": trajectory_losses.sigma_loss_mS_cm,
        },
        "trajectory_coverage": tuple(
            _trajectory_coverage_mapping(coverage) for coverage in coverages
        ),
        "direct_corrector_failure_counts": {
            "direct_capacity_failure_count": sum(
                1 for row in audit_result.rows if row.direct_capacity_failure
            ),
            "corrector_too_strong_failure_count": sum(
                1 for row in audit_result.rows if row.corrector_too_strong_failure
            ),
            "corrector_too_weak_failure_count": sum(
                1 for row in audit_result.rows if row.corrector_too_weak_failure
            ),
        },
        "row_buckets": _calibration_row_bucket_report_mapping(
            audit_result.rows,
            fit_options,
        ),
        "worst_rows": tuple(
            _calibration_worst_row_mapping(row_result, fit_options)
            for row_result in tuple(
                sorted(
                    audit_result.rows,
                    key=_absolute_residual_sort_key,
                    reverse=True,
                )
            )[: fit_options.residual_tail_count]
        ),
    }


def _calibration_row_bucket_report_mapping(
    row_results: tuple["MolecularPropertyDbRowResult", ...],
    fit_options: PrimitiveFitOptions,
) -> dict:
    if not row_results:
        raise ValueError("calibration row bucket report requires rows")
    return {
        "direct_capacity_failures": _calibration_row_bucket_mapping(
            "direct_capacity_failures",
            tuple(row for row in row_results if row.direct_capacity_failure),
            fit_options,
        ),
        "corrector_too_strong_failures": _calibration_row_bucket_mapping(
            "corrector_too_strong_failures",
            tuple(row for row in row_results if row.corrector_too_strong_failure),
            fit_options,
        ),
        "corrector_too_weak_failures": _calibration_row_bucket_mapping(
            "corrector_too_weak_failures",
            tuple(row for row in row_results if row.corrector_too_weak_failure),
            fit_options,
        ),
        "worst_residual_tail": _calibration_row_bucket_mapping(
            "worst_residual_tail",
            tuple(
                sorted(
                    row_results,
                    key=_absolute_residual_sort_key,
                    reverse=True,
                )
            )[: fit_options.residual_tail_count],
            fit_options,
        ),
    }


def _calibration_row_bucket_mapping(
    bucket_name: str,
    row_results: tuple["MolecularPropertyDbRowResult", ...],
    fit_options: PrimitiveFitOptions,
) -> dict:
    bucket_name_text = _nonempty_string(bucket_name, "bucket_name")
    sorted_rows = tuple(
        sorted(
            row_results,
            key=_absolute_residual_sort_key,
            reverse=True,
        )
    )
    return {
        "bucket_name": bucket_name_text,
        "row_count": len(sorted_rows),
        "mean_abs_residual_mS_cm": _mean_abs_row_residual_mS_cm(sorted_rows),
        "rows": tuple(
            _calibration_worst_row_mapping(row_result, fit_options)
            for row_result in sorted_rows
        ),
    }


def _mean_abs_row_residual_mS_cm(
    row_results: tuple["MolecularPropertyDbRowResult", ...],
) -> float:
    if not row_results:
        return 0.0
    return float(
        math.fsum(
            abs(_finite_float(row_result.residual_mS_cm, "row_residual_mS_cm"))
            for row_result in row_results
        )
        / len(row_results)
    )


def _trajectory_coverage_mapping(
    coverage: TrajectoryConcentrationTargetCoverage,
) -> dict:
    return {
        "system_id": coverage.system_id,
        "positive_target_count": coverage.positive_target_count,
        "reachable_target_count": coverage.reachable_target_count,
        "unreachable_target_labels": coverage.unreachable_target_labels,
        "under_floor_target_labels": coverage.under_floor_target_labels,
        "target_rows": tuple(
            {
                "target_label": target_label,
                "target_concentration_mol_m3": target_concentration_mol_m3,
                "predicted_concentration_mol_m3": predicted_concentration_mol_m3,
                "reachable": reachable,
            }
            for (
                target_label,
                target_concentration_mol_m3,
                predicted_concentration_mol_m3,
                reachable,
            ) in coverage.predicted_target_rows
        ),
    }


def _calibration_worst_row_mapping(
    row_result: "MolecularPropertyDbRowResult",
    fit_options: PrimitiveFitOptions,
) -> dict:
    return {
        "row_id": row_result.row_id,
        "source_row_ids": row_result.source_row_ids,
        "proof_status": row_result.proof_status,
        "empirical_sigma_mS_cm": row_result.empirical_sigma_mS_cm,
        "predicted_sigma_mS_cm": row_result.predicted_sigma_mS_cm,
        "residual_mS_cm": row_result.residual_mS_cm,
        "direct_sigma_mS_cm": row_result.direct_sigma_mS_cm,
        "corrector_sigma_mS_cm": row_result.corrector_sigma_mS_cm,
        "direct_capacity_gap_mS_cm": row_result.direct_capacity_gap_mS_cm,
        "corrector_target_mS_cm": row_result.corrector_target_mS_cm,
        "corrector_residual_mS_cm": row_result.corrector_residual_mS_cm,
        "direct_capacity_failure": row_result.direct_capacity_failure,
        "corrector_too_strong_failure": row_result.corrector_too_strong_failure,
        "corrector_too_weak_failure": row_result.corrector_too_weak_failure,
        "free_ion_fraction": row_result.free_ion_fraction,
        "charged_cluster_fraction": row_result.charged_cluster_fraction,
        "neutral_cluster_fraction": row_result.neutral_cluster_fraction,
        "charge_weighted_transport_concentration_mol_m3": (
            row_result.charge_weighted_transport_concentration_mol_m3
        ),
        "charged_cluster_direct_sigma_mS_cm": (
            row_result.charged_cluster_direct_sigma_mS_cm
        ),
        "charged_cluster_corrector_sigma_mS_cm": (
            row_result.charged_cluster_corrector_sigma_mS_cm
        ),
        "charged_cluster_net_sigma_mS_cm": (
            row_result.charged_cluster_net_sigma_mS_cm
        ),
        "direct_sigma_by_transport_role_mS_cm": dict(
            row_result.direct_sigma_by_transport_role_mS_cm
        ),
        "corrector_sigma_by_transport_role_mS_cm": dict(
            row_result.corrector_sigma_by_transport_role_mS_cm
        ),
        "net_sigma_by_transport_role_mS_cm": dict(
            row_result.net_sigma_by_transport_role_mS_cm
        ),
        "top_clusters_by_concentration": tuple(
            _cluster_thermodynamic_mapping(cluster_diagnostic)
            for cluster_diagnostic in tuple(
                sorted(
                    row_result.cluster_thermodynamic_diagnostics,
                    key=_cluster_concentration_sort_key,
                    reverse=True,
                )
            )[: fit_options.residual_tail_count]
        ),
        "top_clusters_by_favorable_free_energy": tuple(
            _cluster_thermodynamic_mapping(cluster_diagnostic)
            for cluster_diagnostic in tuple(
                sorted(
                    row_result.cluster_thermodynamic_diagnostics,
                    key=_cluster_free_energy_sort_key,
                )
            )[: fit_options.residual_tail_count]
        ),
    }


def _cluster_concentration_sort_key(
    cluster_diagnostic: "MolecularClusterThermodynamicDiagnostic",
) -> float:
    return cluster_diagnostic.concentration_mol_m3


def _cluster_free_energy_sort_key(
    cluster_diagnostic: "MolecularClusterThermodynamicDiagnostic",
) -> float:
    return cluster_diagnostic.standard_free_energy_over_RT


def _cluster_thermodynamic_mapping(
    cluster_diagnostic: "MolecularClusterThermodynamicDiagnostic",
) -> dict:
    return {
        "cluster_label": cluster_diagnostic.cluster_label,
        "cluster_kind": cluster_diagnostic.cluster_kind,
        "stoichiometry": dict(cluster_diagnostic.stoichiometry),
        "net_charge_number": cluster_diagnostic.net_charge_number,
        "concentration_mol_m3": cluster_diagnostic.concentration_mol_m3,
        "concentration_fraction_of_total_ion": (
            cluster_diagnostic.concentration_fraction_of_total_ion
        ),
        "standard_free_energy_over_RT": (
            cluster_diagnostic.standard_free_energy_over_RT
        ),
        "log_equilibrium_constant": cluster_diagnostic.log_equilibrium_constant,
        "coulomb_J_mol": cluster_diagnostic.coulomb_J_mol,
        "desolvation_J_mol": cluster_diagnostic.desolvation_J_mol,
        "coordination_J_mol": cluster_diagnostic.coordination_J_mol,
        "steric_J_mol": cluster_diagnostic.steric_J_mol,
        "entropy_J_mol": cluster_diagnostic.entropy_J_mol,
        "activity_correction_J_mol": (
            cluster_diagnostic.activity_correction_J_mol
        ),
        "hydrodynamic_radius_A": cluster_diagnostic.hydrodynamic_radius_A,
        "molecular_volume_A3": cluster_diagnostic.molecular_volume_A3,
    }


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


def load_trajectory_primitive_calibration_targets(
    target_paths: tuple[str, ...],
) -> tuple[TrajectoryPrimitiveCalibrationTarget, ...]:
    return tuple(
        _trajectory_primitive_calibration_target_from_path(target_path)
        for target_path in target_paths
    )


def _trajectory_primitive_calibration_target_from_path(
    target_path: str,
) -> TrajectoryPrimitiveCalibrationTarget:
    artifact_path = _nonempty_string(target_path, "trajectory_primitive_target_path")
    artifact_text = Path(artifact_path).read_text(encoding="utf-8")
    artifact_mapping = _json_mapping(json.loads(artifact_text), artifact_path)
    artifact_type = _required_json_string(
        artifact_mapping,
        "artifact_type",
        artifact_path,
    )
    if artifact_type != TRAJECTORY_PRIMITIVE_CALIBRATION_ARTIFACT_TYPE:
        raise ValueError(
            "trajectory primitive artifact_type must be "
            f"{TRAJECTORY_PRIMITIVE_CALIBRATION_ARTIFACT_TYPE}, found {artifact_type}"
        )
    system_id = _required_json_string(artifact_mapping, "system_id", artifact_path)
    return TrajectoryPrimitiveCalibrationTarget(
        system_id=system_id,
        recipe=_molecular_recipe_from_json_mapping(
            _required_json_mapping(artifact_mapping, "recipe", artifact_path),
            f"{artifact_path}.recipe",
        ),
        species_inputs=_molecular_species_inputs_from_json_mapping(
            _required_json_mapping(artifact_mapping, "species_inputs", artifact_path),
            f"{artifact_path}.species_inputs",
        ),
        primitive_target_artifact=_trajectory_primitive_target_artifact_from_mapping(
            system_id,
            _required_json_mapping(
                artifact_mapping,
                "primitive_targets",
                artifact_path,
            ),
            f"{artifact_path}.primitive_targets",
        ),
    )


def _json_mapping(
    value,
    context: str,
) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _required_json_mapping(
    mapping: dict,
    key: str,
    context: str,
) -> dict:
    if key not in mapping:
        raise ValueError(f"{context} missing {key}")
    return _json_mapping(mapping[key], f"{context}.{key}")


def _required_json_list(
    mapping: dict,
    key: str,
    context: str,
) -> list:
    if key not in mapping:
        raise ValueError(f"{context} missing {key}")
    value = mapping[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a list")
    return value


def _required_json_string(
    mapping: dict,
    key: str,
    context: str,
) -> str:
    if key not in mapping:
        raise ValueError(f"{context} missing {key}")
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{context}.{key} must be a nonempty string")
    return value


def _required_json_text(
    mapping: dict,
    key: str,
    context: str,
) -> str:
    if key not in mapping:
        raise ValueError(f"{context} missing {key}")
    value = mapping[key]
    if not isinstance(value, str):
        raise TypeError(f"{context}.{key} must be a string")
    return value


def _required_json_bool(
    mapping: dict,
    key: str,
    context: str,
) -> bool:
    if key not in mapping:
        raise ValueError(f"{context} missing {key}")
    value = mapping[key]
    if not isinstance(value, bool):
        raise TypeError(f"{context}.{key} must be a boolean")
    return value


def _required_json_int(
    mapping: dict,
    key: str,
    context: str,
) -> int:
    if key not in mapping:
        raise ValueError(f"{context} missing {key}")
    value = mapping[key]
    if not isinstance(value, int):
        raise TypeError(f"{context}.{key} must be an integer")
    return value


def _required_json_float(
    mapping: dict,
    key: str,
    context: str,
) -> float:
    if key not in mapping:
        raise ValueError(f"{context} missing {key}")
    value = mapping[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context}.{key} must be numeric")
    return _finite_float(float(value), f"{context}.{key}")


def _required_json_positive_float(
    mapping: dict,
    key: str,
    context: str,
) -> float:
    return _positive_float(
        _required_json_float(mapping, key, context),
        f"{context}.{key}",
    )


def _required_json_nonnegative_float(
    mapping: dict,
    key: str,
    context: str,
) -> float:
    return _nonnegative_float(
        _required_json_float(mapping, key, context),
        f"{context}.{key}",
    )


def _required_json_positive_int(
    mapping: dict,
    key: str,
    context: str,
) -> int:
    return _positive_int(
        _required_json_int(mapping, key, context),
        f"{context}.{key}",
    )


def _required_json_nonnegative_int(
    mapping: dict,
    key: str,
    context: str,
) -> int:
    return _nonnegative_int(
        _required_json_int(mapping, key, context),
        f"{context}.{key}",
    )


def _molecular_recipe_from_json_mapping(
    recipe_mapping: dict,
    context: str,
) -> "MolecularElectrolyteRecipe":
    from conductivity.analytical_conductivity_model import (
        MolecularElectrolyteRecipe,
        MolecularMixtureProperties,
    )

    mixture_mapping = _required_json_mapping(
        recipe_mapping,
        "mixture_properties",
        context,
    )
    return MolecularElectrolyteRecipe(
        cations=_required_float_mapping(recipe_mapping, "cations", context),
        anions=_required_float_mapping(recipe_mapping, "anions", context),
        solvents=_required_float_mapping(recipe_mapping, "solvents", context),
        additives=_required_float_mapping(recipe_mapping, "additives", context),
        temperature_K=_required_json_positive_float(
            recipe_mapping,
            "temperature_K",
            context,
        ),
        pressure_Pa=_required_json_positive_float(
            recipe_mapping,
            "pressure_Pa",
            context,
        ),
        mixture_properties=MolecularMixtureProperties(
            density_g_ml=_required_json_positive_float(
                mixture_mapping,
                "density_g_ml",
                f"{context}.mixture_properties",
            ),
            viscosity_cP=_required_json_positive_float(
                mixture_mapping,
                "viscosity_cP",
                f"{context}.mixture_properties",
            ),
            dielectric_constant=_required_json_positive_float(
                mixture_mapping,
                "dielectric_constant",
                f"{context}.mixture_properties",
            ),
        ),
    )


def _molecular_species_inputs_from_json_mapping(
    species_inputs_mapping: dict,
    context: str,
) -> Mapping[str, "MolecularSpeciesInput"]:
    from conductivity.analytical_conductivity_model import MolecularSpeciesInput

    species_inputs: dict[str, MolecularSpeciesInput] = {}
    for species_name, raw_species_mapping in species_inputs_mapping.items():
        species_context = f"{context}.{species_name}"
        species_mapping = _json_mapping(raw_species_mapping, species_context)
        species_inputs[species_name] = MolecularSpeciesInput(
            name=_required_json_string(species_mapping, "name", species_context),
            role=_required_json_string(species_mapping, "role", species_context),
            charge_number=_required_json_int(
                species_mapping,
                "charge_number",
                species_context,
            ),
            smiles=_required_json_text(species_mapping, "smiles", species_context),
            xyz_coordinates=_required_xyz_coordinates(
                species_mapping,
                "xyz_coordinates",
                species_context,
            ),
            property_overrides=_required_float_mapping(
                species_mapping,
                "property_overrides",
                species_context,
            ),
            coordination_sites=_required_string_tuple(
                species_mapping,
                "coordination_sites",
                species_context,
            ),
        )
    return species_inputs


def _required_float_mapping(
    mapping: dict,
    key: str,
    context: str,
) -> Mapping[str, float]:
    value_mapping = _required_json_mapping(mapping, key, context)
    parsed_mapping: dict[str, float] = {}
    for value_key, raw_value in value_mapping.items():
        if not isinstance(value_key, str) or value_key == "":
            raise ValueError(f"{context}.{key} contains an empty key")
        if not isinstance(raw_value, (int, float)):
            raise TypeError(f"{context}.{key}.{value_key} must be numeric")
        parsed_mapping[value_key] = _finite_float(
            float(raw_value),
            f"{context}.{key}.{value_key}",
        )
    return parsed_mapping


def _required_string_tuple(
    mapping: dict,
    key: str,
    context: str,
) -> tuple[str, ...]:
    values = _required_json_list(mapping, key, context)
    parsed_values: list[str] = []
    for value_index, raw_value in enumerate(values):
        value_context = f"{context}.{key}[{value_index}]"
        if not isinstance(raw_value, str) or raw_value == "":
            raise ValueError(f"{value_context} must be a nonempty string")
        parsed_values.append(raw_value)
    return tuple(parsed_values)


def _required_xyz_coordinates(
    mapping: dict,
    key: str,
    context: str,
) -> tuple[tuple[str, float, float, float], ...]:
    coordinate_rows = _required_json_list(mapping, key, context)
    parsed_coordinates: list[tuple[str, float, float, float]] = []
    for row_index, raw_coordinate_row in enumerate(coordinate_rows):
        row_context = f"{context}.{key}[{row_index}]"
        if not isinstance(raw_coordinate_row, list):
            raise TypeError(f"{row_context} must be a list")
        coordinate_row_length = 4
        if len(raw_coordinate_row) != coordinate_row_length:
            raise ValueError(f"{row_context} must contain element,x,y,z")
        element_symbol = raw_coordinate_row[0]
        if not isinstance(element_symbol, str) or element_symbol == "":
            raise ValueError(f"{row_context}.element must be a nonempty string")
        parsed_coordinates.append(
            (
                element_symbol,
                _finite_float(float(raw_coordinate_row[1]), f"{row_context}.x"),
                _finite_float(float(raw_coordinate_row[2]), f"{row_context}.y"),
                _finite_float(float(raw_coordinate_row[3]), f"{row_context}.z"),
            )
        )
    return tuple(parsed_coordinates)


def _trajectory_primitive_target_artifact_from_mapping(
    system_id: str,
    target_mapping: dict,
    context: str,
) -> "TrajectoryPrimitiveTargetArtifact":
    from conductivity.trajectory_primitive_targets import (
        TrajectoryPrimitiveTargetArtifact,
    )

    block_targets = tuple(
        _trajectory_block_target_from_mapping(
            _json_mapping(raw_block_target, f"{context}.block_targets[{block_index}]"),
            f"{context}.block_targets[{block_index}]",
        )
        for block_index, raw_block_target in enumerate(
            _required_json_list(target_mapping, "block_targets", context)
        )
    )
    return TrajectoryPrimitiveTargetArtifact(
        system_id=system_id,
        frame_count=_required_json_positive_int(target_mapping, "frame_count", context),
        dt_s=_required_json_positive_float(target_mapping, "dt_s", context),
        frame_stride=_required_json_positive_int(
            target_mapping,
            "frame_stride",
            context,
        ),
        block_count=_required_json_positive_int(target_mapping, "block_count", context),
        state_concentrations_mol_m3=_required_float_mapping(
            target_mapping,
            "state_concentrations_mol_m3",
            context,
        ),
        state_occupancy_fractions=_required_float_mapping(
            target_mapping,
            "state_occupancy_fractions",
            context,
        ),
        transition_rates_s_inv=_required_float_mapping(
            target_mapping,
            "transition_rates_s_inv",
            context,
        ),
        transition_rate_targets_validated=_required_json_bool(
            target_mapping,
            "transition_rate_targets_validated",
            context,
        ),
        transition_fluxes_mol_m3_s=_required_float_mapping(
            target_mapping,
            "transition_fluxes_mol_m3_s",
            context,
        ),
        residence_times_s=_required_float_mapping(
            target_mapping,
            "residence_times_s",
            context,
        ),
        displacement_moments_by_family=_displacement_moment_targets_from_json_mapping(
            _required_json_mapping(
                target_mapping,
                "displacement_moments_by_family",
                context,
            ),
            f"{context}.displacement_moments_by_family",
        ),
        displacement_moment_targets_validated=_required_json_bool(
            target_mapping,
            "displacement_moment_targets_validated",
            context,
        ),
        markov_additive_sigma_mS_cm=_required_json_nonnegative_float(
            target_mapping,
            "markov_additive_sigma_mS_cm",
            context,
        ),
        markov_direct_sigma_mS_cm=_required_json_nonnegative_float(
            target_mapping,
            "markov_direct_sigma_mS_cm",
            context,
        ),
        markov_corrector_sigma_mS_cm=_required_json_nonnegative_float(
            target_mapping,
            "markov_corrector_sigma_mS_cm",
            context,
        ),
        markov_additive_sigma_validated=_required_json_bool(
            target_mapping,
            "markov_additive_sigma_validated",
            context,
        ),
        block_targets=block_targets,
        block_state_concentration_standard_errors_mol_m3=_required_float_mapping(
            target_mapping,
            "block_state_concentration_standard_errors_mol_m3",
            context,
        ),
        block_transition_rate_standard_errors_s_inv=_required_float_mapping(
            target_mapping,
            "block_transition_rate_standard_errors_s_inv",
            context,
        ),
        block_displacement_moment_standard_errors_m2=_required_float_mapping(
            target_mapping,
            "block_displacement_moment_standard_errors_m2",
            context,
        ),
        block_sigma_standard_error_mS_cm=_required_json_nonnegative_float(
            target_mapping,
            "block_sigma_standard_error_mS_cm",
            context,
        ),
    )


def _trajectory_block_target_from_mapping(
    block_mapping: dict,
    context: str,
) -> "TrajectoryBlockPrimitiveTarget":
    from conductivity.trajectory_primitive_targets import TrajectoryBlockPrimitiveTarget

    return TrajectoryBlockPrimitiveTarget(
        block_index=_required_json_nonnegative_int(
            block_mapping,
            "block_index",
            context,
        ),
        frame_count=_required_json_positive_int(block_mapping, "frame_count", context),
        step_count=_required_json_positive_int(block_mapping, "step_count", context),
        state_concentrations_mol_m3=_required_float_mapping(
            block_mapping,
            "state_concentrations_mol_m3",
            context,
        ),
        state_occupancy_fractions=_required_float_mapping(
            block_mapping,
            "state_occupancy_fractions",
            context,
        ),
        transition_rates_s_inv=_required_float_mapping(
            block_mapping,
            "transition_rates_s_inv",
            context,
        ),
        transition_rate_targets_validated=_required_json_bool(
            block_mapping,
            "transition_rate_targets_validated",
            context,
        ),
        transition_fluxes_mol_m3_s=_required_float_mapping(
            block_mapping,
            "transition_fluxes_mol_m3_s",
            context,
        ),
        residence_times_s=_required_float_mapping(
            block_mapping,
            "residence_times_s",
            context,
        ),
        displacement_moments_by_family=_displacement_moment_targets_from_json_mapping(
            _required_json_mapping(
                block_mapping,
                "displacement_moments_by_family",
                context,
            ),
            f"{context}.displacement_moments_by_family",
        ),
        markov_additive_sigma_mS_cm=_required_json_nonnegative_float(
            block_mapping,
            "markov_additive_sigma_mS_cm",
            context,
        ),
        markov_direct_sigma_mS_cm=_required_json_nonnegative_float(
            block_mapping,
            "markov_direct_sigma_mS_cm",
            context,
        ),
        markov_corrector_sigma_mS_cm=_required_json_nonnegative_float(
            block_mapping,
            "markov_corrector_sigma_mS_cm",
            context,
        ),
    )


def _displacement_moment_targets_from_json_mapping(
    moment_mapping: dict,
    context: str,
) -> Mapping[str, "TrajectoryDisplacementMomentTarget"]:
    from conductivity.trajectory_primitive_targets import (
        TrajectoryDisplacementMomentTarget,
    )

    moment_targets: dict[str, TrajectoryDisplacementMomentTarget] = {}
    for family_label, raw_family_mapping in moment_mapping.items():
        family_context = f"{context}.{family_label}"
        family_mapping = _json_mapping(raw_family_mapping, family_context)
        moment_targets[family_label] = TrajectoryDisplacementMomentTarget(
            sample_count=_required_json_positive_int(
                family_mapping,
                "sample_count",
                family_context,
            ),
            mean_displacement_m=_required_float_triplet(
                family_mapping,
                "mean_displacement_m",
                family_context,
            ),
            mean_squared_axis_displacement_m2=_required_float_triplet(
                family_mapping,
                "mean_squared_axis_displacement_m2",
                family_context,
            ),
            mean_squared_displacement_m2=_required_json_nonnegative_float(
                family_mapping,
                "mean_squared_displacement_m2",
                family_context,
            ),
        )
    return moment_targets


def _required_float_triplet(
    mapping: dict,
    key: str,
    context: str,
) -> tuple[float, float, float]:
    values = _required_json_list(mapping, key, context)
    expected_value_count = 3
    if len(values) != expected_value_count:
        raise ValueError(f"{context}.{key} must contain three values")
    return (
        _finite_float(float(values[0]), f"{context}.{key}[0]"),
        _finite_float(float(values[1]), f"{context}.{key}[1]"),
        _finite_float(float(values[2]), f"{context}.{key}[2]"),
    )


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
        "proof_statuses": tuple(
            _nonempty_string(proof_status, "metrics.proof_status")
            for proof_status in metrics.proof_statuses
        ),
        "failed_rows": int(_nonnegative_int(metrics.failed_rows, "metrics.failed_rows")),
        "trajectory_concentration_unreachable_target_count": int(
            _nonnegative_int(
                metrics.trajectory_concentration_unreachable_target_count,
                "metrics.trajectory_concentration_unreachable_target_count",
            )
        ),
        "trajectory_concentration_under_floor_target_count": int(
            _nonnegative_int(
                metrics.trajectory_concentration_under_floor_target_count,
                "metrics.trajectory_concentration_under_floor_target_count",
            )
        ),
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


def _trajectory_concentration_coverage_counts(
    trajectory_targets: tuple[TrajectoryPrimitiveCalibrationTarget, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    audit_options: "MolecularPropertyDbAuditOptions",
) -> tuple[int, int]:
    if not trajectory_targets:
        return (0, 0)
    coverages = trajectory_concentration_target_coverage(
        trajectory_targets,
        primitive_parameters,
        audit_options,
    )
    unreachable_target_count = sum(
        len(coverage.unreachable_target_labels) for coverage in coverages
    )
    under_floor_target_count = sum(
        len(coverage.under_floor_target_labels) for coverage in coverages
    )
    return (unreachable_target_count, under_floor_target_count)


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


def _nonnegative_float_tuple(
    values: tuple[float, ...],
    context: str,
) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{context} must be nonempty")
    return tuple(_nonnegative_float(value, context) for value in values)


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


C_STABLE_DIRECT_CAPACITY_BLOCK_PARAMETER_NAMES = (
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
    trajectory_primitive_calibration_targets: tuple[
        TrajectoryPrimitiveCalibrationTarget,
        ...,
    ]
    coordinate_bounds: tuple[PrimitiveParameterTransform, ...]
    full_evaluator: ConductivityPrimitiveParameterEvaluator
    regularization_reference_parameters: ConductivityPrimitiveParameterSet


class SelectedRowPrimitiveEvaluator:
    def __init__(
        self,
        cases: tuple["MolecularPropertyDbCase", ...],
        audit_options: "MolecularPropertyDbAuditOptions",
        fit_options: PrimitiveFitOptions,
        trajectory_primitive_calibration_targets: tuple[
            TrajectoryPrimitiveCalibrationTarget,
            ...,
        ],
        selected_row_indices: tuple[int, ...],
        consumed_parameter_fields: tuple[str, ...],
    ) -> None:
        if not selected_row_indices:
            raise ValueError("selected-row evaluator requires selected rows")
        if not consumed_parameter_fields:
            raise ValueError("selected-row evaluator requires consumed parameter fields")
        selected_cases = tuple(cases[row_index] for row_index in selected_row_indices)
        selected_audit_options = replace(
            audit_options,
            include_event_family_attribution=False,
        )
        self._selected_evaluator = MolecularPropertyDbPrimitiveEvaluator(
            selected_cases,
            selected_audit_options,
            fit_options,
            trajectory_primitive_calibration_targets,
        )
        self._selected_evaluator._consumed_parameter_fields = consumed_parameter_fields
        self._selected_evaluator._has_measured_consumed_parameter_fields = True

    def evaluate(
        self,
        primitive_parameters: ConductivityPrimitiveParameterSet,
    ) -> PrimitiveFitDatasetEvaluation:
        return self._selected_evaluator.evaluate(primitive_parameters)

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
    trajectory_primitive_calibration_targets = (
        load_trajectory_primitive_calibration_targets(
            fit_options.trajectory_primitive_target_paths,
        )
    )
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
        trajectory_primitive_calibration_targets,
    )
    full_evaluator = MolecularPropertyDbPrimitiveEvaluator(
        case_selection.cases,
        audit_options,
        fit_options,
        trajectory_primitive_calibration_targets,
    )
    decomposed_context = DecomposedFitContext(
        cases=case_selection.cases,
        audit_options=audit_options,
        fit_options=block_fit_options,
        trajectory_primitive_calibration_targets=trajectory_primitive_calibration_targets,
        coordinate_bounds=coordinate_bounds,
        full_evaluator=block_evaluator,
        regularization_reference_parameters=REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
    )
    final_decomposed_context = DecomposedFitContext(
        cases=case_selection.cases,
        audit_options=audit_options,
        fit_options=fit_options,
        trajectory_primitive_calibration_targets=trajectory_primitive_calibration_targets,
        coordinate_bounds=coordinate_bounds,
        full_evaluator=full_evaluator,
        regularization_reference_parameters=REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
    )
    configured_parameters = configured_conductivity_primitive_parameters()
    validate_trajectory_concentration_target_coverage(
        trajectory_primitive_calibration_targets,
        configured_parameters,
        audit_options,
    )
    current_parameters = initialize_topology_logk_offsets_from_trajectory_concentrations(
        configured_parameters,
        trajectory_primitive_calibration_targets,
        audit_options,
        coordinate_bounds,
        fit_options,
    )
    initialized_parameters = current_parameters
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
        C_STABLE_DIRECT_CAPACITY_BLOCK_PARAMETER_NAMES,
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
    baseline_trajectory_coverage_counts = _trajectory_concentration_coverage_counts(
        trajectory_primitive_calibration_targets,
        current_parameters,
        audit_options,
    )
    candidate_trajectory_coverage_counts = _trajectory_concentration_coverage_counts(
        trajectory_primitive_calibration_targets,
        loaded_candidate_parameters,
        audit_options,
    )
    baseline_metrics = PrimitivePromotionMetrics(
        mae_mS_cm=baseline_audit_result.mae_mS_cm,
        bias_mS_cm=baseline_audit_result.bias_mS_cm,
        pearson_r=baseline_audit_result.pearson_r,
        worst_abs_residual_mS_cm=baseline_audit_result.maximum_abs_residual_mS_cm,
        proof_statuses=baseline_audit_result.proof_statuses,
        failed_rows=baseline_audit_result.failed_rows,
        trajectory_concentration_unreachable_target_count=(
            baseline_trajectory_coverage_counts[0]
        ),
        trajectory_concentration_under_floor_target_count=(
            baseline_trajectory_coverage_counts[1]
        ),
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
        proof_statuses=candidate_audit_result.proof_statuses,
        failed_rows=candidate_audit_result.failed_rows,
        trajectory_concentration_unreachable_target_count=(
            candidate_trajectory_coverage_counts[0]
        ),
        trajectory_concentration_under_floor_target_count=(
            candidate_trajectory_coverage_counts[1]
        ),
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
    write_calibration_error_decomposition_report(
        fit_options.decomposition_report_output_path,
        (
            ("c_initialized_baseline", initialized_parameters),
            (
                "after_direct_capacity_block",
                direct_capacity_result.accepted_parameters,
            ),
            ("after_corrector_block", corrector_result.accepted_parameters),
            ("after_cluster_sink_block", cluster_sink_result.accepted_parameters),
            ("verified_candidate", loaded_candidate_parameters),
        ),
        case_selection.cases,
        audit_options,
        fit_options,
        trajectory_primitive_calibration_targets,
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
    block_evaluator = SelectedRowPrimitiveEvaluator(
        decomposed_context.cases,
        decomposed_context.audit_options,
        decomposed_context.fit_options,
        decomposed_context.trajectory_primitive_calibration_targets,
        selected_row_indices,
        _unique_parameter_names(block_parameter_names),
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
    starting_evaluation = decomposed_context.full_evaluator.evaluate(
        initial_parameters,
    )
    candidate_evaluation = decomposed_context.full_evaluator.evaluate(
        fit_result.best_candidate.primitive_parameters,
    )
    accepted = _block_candidate_preserves_full_audit(
        starting_audit_result,
        candidate_audit_result,
        starting_evaluation,
        candidate_evaluation,
        decomposed_context.fit_options,
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


def _tail_and_failure_row_indices(audit_result, fit_options):
    worst_residual_row_indices = tuple(
        row_index
        for row_index, _row_result in sorted(
            enumerate(audit_result.rows),
            key=_indexed_absolute_residual_sort_key,
            reverse=True,
        )[: fit_options.residual_tail_count]
    )
    failure_row_indices = tuple(
        row_index
        for row_index, row_result in enumerate(audit_result.rows)
        if (
            row_result.direct_capacity_failure
            or row_result.corrector_too_strong_failure
            or row_result.corrector_too_weak_failure
        )
    )
    normal_row_indices = tuple(
        row_index
        for row_index, row_result in enumerate(audit_result.rows)
        if row_index not in set(worst_residual_row_indices + failure_row_indices)
        and not row_result.failed
    )
    stratified_normal_row_indices = []
    if normal_row_indices:
        normal_sample_count = min(
            fit_options.residual_tail_count,
            len(normal_row_indices),
        )
        for sample_index in range(normal_sample_count):
            if normal_sample_count == 1:
                stratified_normal_row_indices.append(normal_row_indices[0])
                continue
            row_position = int(
                round(
                    sample_index
                    * (len(normal_row_indices) - 1)
                    / (normal_sample_count - 1)
                )
            )
            stratified_normal_row_indices.append(normal_row_indices[row_position])
    return tuple(
        sorted(
            set(
                worst_residual_row_indices
                + failure_row_indices
                + tuple(stratified_normal_row_indices)
            )
        )
    )


def fit_onsager_operator_primitives(
    calibration_subset_mode: Literal[
        "full_property_db",
        "tail_and_failure_rows",
    ],
):
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

    if calibration_subset_mode not in {
        "full_property_db",
        "tail_and_failure_rows",
    }:
        raise ValueError(
            "calibration_subset_mode must be full_property_db or "
            "tail_and_failure_rows"
        )
    audit_options = default_molecular_property_db_audit_options()
    fit_options, coordinate_bounds = default_molecular_primitive_fit_configuration()
    fit_options = replace(
        fit_options,
        cluster_activation_loss_weight=0.0,
        latin_hypercube_samples_per_parameter=0.02,
        coordinate_search_rounds=0,
        powell_max_iterations_per_parameter=0.0,
        powell_max_function_evaluations_per_parameter=0.0,
    )
    trajectory_primitive_calibration_targets = (
        load_trajectory_primitive_calibration_targets(
            fit_options.trajectory_primitive_target_paths,
        )
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
    evaluator = MolecularPropertyDbPrimitiveEvaluator(
        case_selection.cases,
        audit_options,
        fit_options,
        trajectory_primitive_calibration_targets,
    )
    configured_parameters = configured_conductivity_primitive_parameters()
    initialized_parameters = initialize_topology_logk_offsets_from_trajectory_concentrations(
        configured_parameters,
        trajectory_primitive_calibration_targets,
        audit_options,
        coordinate_bounds,
        fit_options,
    )
    baseline_audit_result = audit_molecular_property_db_cases(
        case_selection.cases,
        initialized_parameters,
        audit_options,
    )
    validate_molecular_property_db_audit_result(
        baseline_audit_result,
        fit_options,
    )
    selected_row_indices = tuple(range(len(case_selection.cases)))
    if calibration_subset_mode == "tail_and_failure_rows":
        selected_row_indices = _tail_and_failure_row_indices(
            baseline_audit_result,
            fit_options,
        )
    block_evaluator = evaluator
    if calibration_subset_mode == "tail_and_failure_rows":
        block_evaluator = SelectedRowPrimitiveEvaluator(
            case_selection.cases,
            audit_options,
            fit_options,
            trajectory_primitive_calibration_targets,
            selected_row_indices,
            ONSAGER_OPERATOR_FIT_PARAMETER_NAMES,
        )
    onsager_coordinate_bounds = _coordinate_bounds_for_parameter_names(
        coordinate_bounds,
        ONSAGER_OPERATOR_FIT_PARAMETER_NAMES,
    )
    active_coordinate_bounds = onsager_coordinate_bounds
    fit_result = fit_conductivity_primitive_parameters(
        initialized_parameters,
        REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS,
        active_coordinate_bounds,
        block_evaluator,
        fit_options,
    )
    write_primitive_parameter_candidate_config(
        fit_options.candidate_output_path,
        fit_result.best_candidate.primitive_parameters,
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
    baseline_trajectory_coverage_counts = _trajectory_concentration_coverage_counts(
        trajectory_primitive_calibration_targets,
        initialized_parameters,
        audit_options,
    )
    candidate_trajectory_coverage_counts = _trajectory_concentration_coverage_counts(
        trajectory_primitive_calibration_targets,
        loaded_candidate_parameters,
        audit_options,
    )
    baseline_metrics = PrimitivePromotionMetrics(
        mae_mS_cm=baseline_audit_result.mae_mS_cm,
        bias_mS_cm=baseline_audit_result.bias_mS_cm,
        pearson_r=baseline_audit_result.pearson_r,
        worst_abs_residual_mS_cm=baseline_audit_result.maximum_abs_residual_mS_cm,
        proof_statuses=baseline_audit_result.proof_statuses,
        failed_rows=baseline_audit_result.failed_rows,
        trajectory_concentration_unreachable_target_count=(
            baseline_trajectory_coverage_counts[0]
        ),
        trajectory_concentration_under_floor_target_count=(
            baseline_trajectory_coverage_counts[1]
        ),
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
        proof_statuses=candidate_audit_result.proof_statuses,
        failed_rows=candidate_audit_result.failed_rows,
        trajectory_concentration_unreachable_target_count=(
            candidate_trajectory_coverage_counts[0]
        ),
        trajectory_concentration_under_floor_target_count=(
            candidate_trajectory_coverage_counts[1]
        ),
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
    return fit_result


def _block_candidate_preserves_full_audit(
    starting_audit_result: "MolecularPropertyDbAuditResult",
    candidate_audit_result: "MolecularPropertyDbAuditResult",
    starting_evaluation: PrimitiveFitDatasetEvaluation,
    candidate_evaluation: PrimitiveFitDatasetEvaluation,
    fit_options: PrimitiveFitOptions,
) -> bool:
    _validate_fit_options(fit_options)
    return (
        candidate_audit_result.mae_mS_cm <= starting_audit_result.mae_mS_cm
        and candidate_audit_result.maximum_abs_residual_mS_cm
        <= starting_audit_result.maximum_abs_residual_mS_cm
        and _candidate_preserves_trajectory_c_stability(
            starting_evaluation,
            candidate_evaluation,
            fit_options,
        )
    )


def _candidate_preserves_trajectory_c_stability(
    starting_evaluation: PrimitiveFitDatasetEvaluation,
    candidate_evaluation: PrimitiveFitDatasetEvaluation,
    fit_options: PrimitiveFitOptions,
) -> bool:
    _validate_fit_options(fit_options)
    if not fit_options.trajectory_primitive_target_paths:
        return True
    return (
        candidate_evaluation.trajectory_concentration_unreachable_target_count
        <= starting_evaluation.trajectory_concentration_unreachable_target_count
        and candidate_evaluation.trajectory_concentration_under_floor_target_count
        <= starting_evaluation.trajectory_concentration_under_floor_target_count
        and candidate_evaluation.trajectory_concentration_loss
        <= starting_evaluation.trajectory_concentration_loss
    )


def _cluster_sink_row_indices(
    audit_result: "MolecularPropertyDbAuditResult",
    fit_options: PrimitiveFitOptions,
) -> tuple[int, ...]:
    sorted_index_and_row = tuple(
        sorted(
            enumerate(audit_result.rows),
            key=_indexed_absolute_residual_sort_key,
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
            key=_absolute_residual_sort_key,
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


def _run_full_fit_main() -> None:
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
    trajectory_primitive_calibration_targets = (
        load_trajectory_primitive_calibration_targets(
            fit_options.trajectory_primitive_target_paths,
        )
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
    driver_matrix_diagnostics = primitive_driver_matrix_diagnostics(
        case_selection.cases,
        fit_options,
    )
    evaluator = MolecularPropertyDbPrimitiveEvaluator(
        case_selection.cases,
        audit_options,
        fit_options,
        trajectory_primitive_calibration_targets,
    )
    configured_parameters = configured_conductivity_primitive_parameters()
    configured_trajectory_coverages = validate_trajectory_concentration_target_coverage(
        trajectory_primitive_calibration_targets,
        configured_parameters,
        audit_options,
    )
    initialized_parameters = initialize_topology_logk_offsets_from_trajectory_concentrations(
        configured_parameters,
        trajectory_primitive_calibration_targets,
        audit_options,
        coordinate_bounds,
        fit_options,
    )
    initialized_trajectory_coverages = validate_trajectory_concentration_target_coverage(
        trajectory_primitive_calibration_targets,
        initialized_parameters,
        audit_options,
    )
    print_trajectory_topology_initialization_changes(
        configured_parameters,
        initialized_parameters,
    )
    print_trajectory_concentration_target_coverage(
        "configured",
        configured_trajectory_coverages,
    )
    print_trajectory_concentration_target_coverage(
        "initialized",
        initialized_trajectory_coverages,
    )
    configured_parameters = initialized_parameters
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
        trajectory_primitive_calibration_targets,
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
    baseline_trajectory_coverage_counts = _trajectory_concentration_coverage_counts(
        trajectory_primitive_calibration_targets,
        configured_parameters,
        audit_options,
    )
    candidate_trajectory_coverage_counts = _trajectory_concentration_coverage_counts(
        trajectory_primitive_calibration_targets,
        loaded_candidate_parameters,
        audit_options,
    )
    baseline_metrics = PrimitivePromotionMetrics(
        mae_mS_cm=baseline_audit_result.mae_mS_cm,
        bias_mS_cm=baseline_audit_result.bias_mS_cm,
        pearson_r=baseline_audit_result.pearson_r,
        worst_abs_residual_mS_cm=(
            baseline_audit_result.maximum_abs_residual_mS_cm
        ),
        proof_statuses=baseline_audit_result.proof_statuses,
        failed_rows=baseline_audit_result.failed_rows,
        trajectory_concentration_unreachable_target_count=(
            baseline_trajectory_coverage_counts[0]
        ),
        trajectory_concentration_under_floor_target_count=(
            baseline_trajectory_coverage_counts[1]
        ),
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
        proof_statuses=candidate_audit_result.proof_statuses,
        failed_rows=candidate_audit_result.failed_rows,
        trajectory_concentration_unreachable_target_count=(
            candidate_trajectory_coverage_counts[0]
        ),
        trajectory_concentration_under_floor_target_count=(
            candidate_trajectory_coverage_counts[1]
        ),
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
        key=_absolute_residual_sort_key,
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


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--block",
        choices=("full_decomposed", "onsager_operator"),
        default="full_decomposed",
    )
    argument_parser.add_argument(
        "--subset",
        choices=("full_property_db", "tail_and_failure_rows"),
        default="full_property_db",
    )
    parsed_arguments = argument_parser.parse_args()
    if parsed_arguments.block == "onsager_operator":
        fit_result = fit_onsager_operator_primitives(parsed_arguments.subset)
        print("molecular_primitive_parameter_fit")
        print("fit_block=onsager_operator")
        print(f"subset_mode={parsed_arguments.subset}")
        print(f"candidate_count={fit_result.candidate_count}")
        print(
            "accepted_candidate_count="
            f"{fit_result.accepted_candidate_count}"
        )
        print(
            "best_mae_mS_cm="
            f"{fit_result.best_candidate.mae_mS_cm:.6f}"
        )
        print(
            "best_bias_mS_cm="
            f"{fit_result.best_candidate.bias_mS_cm:.6f}"
        )
        print(
            "best_worst_abs_residual_mS_cm="
            f"{fit_result.best_candidate.worst_abs_residual_mS_cm:.6f}"
        )
        return
    if parsed_arguments.block == "full_decomposed":
        decomposed_fit_result = fit_decomposed_conductivity_primitives()
        print("molecular_primitive_parameter_fit")
        print("fit_block=full_decomposed")
        print(
            "direct_capacity_block_accepted="
            f"{decomposed_fit_result.direct_capacity_result.accepted}"
        )
        print(
            "corrector_block_accepted="
            f"{decomposed_fit_result.corrector_result.accepted}"
        )
        print(
            "cluster_sink_block_accepted="
            f"{decomposed_fit_result.cluster_sink_result.accepted}"
        )
        print(
            "final_joint_block_accepted="
            f"{decomposed_fit_result.final_result.accepted}"
        )
        print(
            "candidate_mae_mS_cm="
            f"{decomposed_fit_result.candidate_audit_result.mae_mS_cm:.6f}"
        )
        print(
            "candidate_bias_mS_cm="
            f"{decomposed_fit_result.candidate_audit_result.bias_mS_cm:.6f}"
        )
        print(
            "candidate_worst_abs_residual_mS_cm="
            f"{decomposed_fit_result.candidate_audit_result.maximum_abs_residual_mS_cm:.6f}"
        )
        print(
            "promotion_rejection_reasons="
            f"{','.join(decomposed_fit_result.promotion_rejection_reasons)}"
        )
        return


if __name__ == "__main__":
    main()
