"""Empirical validation for the analytic descriptor-to-Mori primitive generator."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from constants import F, R, S_M_TO_MS_CM
from conductivity.analytic_mori_primitive_generator import (
    AnalyticBackjumpCagePrimitive,
    AnalyticFreeCarrierObstructionPrimitive,
    AnalyticMoriRecipe,
    AnalyticMoriPrimitiveResult,
    AnalyticSelectiveCarrierCagePrimitive,
    AnalyticMoriSpeciesCatalog,
    ANALYTIC_MORI_ABLATION_ASSOCIATION_OFF,
    ANALYTIC_MORI_ABLATION_ATMOSPHERE_RESISTANCE_OFF,
    ANALYTIC_MORI_ABLATION_BACKJUMP_CAGE_MEMORY,
    ANALYTIC_MORI_ABLATION_BASELINE,
    ANALYTIC_MORI_ABLATION_BINDING_RESISTANCE_OFF,
    ANALYTIC_MORI_ABLATION_CARRIER_CAGE_POINT_SUBSTATES,
    ANALYTIC_MORI_ABLATION_DIELECTRIC_DECREMENT_OFF,
    ANALYTIC_MORI_ABLATION_FREE_ION_NE,
    ANALYTIC_MORI_ABLATION_FREE_LI_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_COMPACT_ANION_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_FREE_LI_PLUS_COMPACT_ANION_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CARRIER_CAGE_ONLY,
    ANALYTIC_MORI_ABLATION_DESCRIPTOR_ATMOSPHERE_RELAXATION_RELEASE_ONLY,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_RELEASE,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_REL_AND_EP_RELEASE,
    ANALYTIC_MORI_ABLATION_TIMESCALE_STRUCTURAL_CAGE_MEMORY,
    ANALYTIC_MORI_ABLATION_HIGHER_AGGREGATES_OFF,
    ANALYTIC_MORI_ABLATION_JONES_DOLE_VISCOSITY_OFF,
    EVENT_FAMILY_ORDINARY_FREE_ANION_TRANSLATION,
    EVENT_FAMILY_ORDINARY_FREE_LI_TRANSLATION,
    StructuralPrimitiveUncertaintyBudget,
    SUPPORTED_ANALYTIC_MORI_ABLATIONS,
    evaluate_analytic_mori_forced_free_carrier_obstruction_conductivity,
    evaluate_analytic_mori_conductivity,
    evaluate_analytic_mori_ablation_conductivity,
)
from conductivity.finite_markov_additive_green_kubo import (
    MarkovAdditiveEventFamilyAttribution,
)
from conductivity.finite_markov_dataset_audit import (
    _require_entry,
    canonicalize_empirical_recipe,
)
from utils.config_cache import load_physics_config, require_config
from utils.strict_validation import require_float


ACTIVE_VOLUME_M3_FOR_MOLAR_CONCENTRATION_READOUT = 1.0e-6
PERCENT = 100.0
ANGSTROM_PER_M = 1.0e10
MOLARITY_TO_MOL_M3 = 1000.0
REGISTRY_LAMBDA0_PRIMITIVE = "registry_lambda0"
ANALYTIC_MORI_AUDIT_ABLATION_MODES = (
    ANALYTIC_MORI_ABLATION_BASELINE,
    ANALYTIC_MORI_ABLATION_ASSOCIATION_OFF,
    ANALYTIC_MORI_ABLATION_HIGHER_AGGREGATES_OFF,
    ANALYTIC_MORI_ABLATION_BINDING_RESISTANCE_OFF,
    ANALYTIC_MORI_ABLATION_ATMOSPHERE_RESISTANCE_OFF,
    ANALYTIC_MORI_ABLATION_JONES_DOLE_VISCOSITY_OFF,
    ANALYTIC_MORI_ABLATION_DIELECTRIC_DECREMENT_OFF,
    ANALYTIC_MORI_ABLATION_FREE_ION_NE,
    ANALYTIC_MORI_ABLATION_CARRIER_CAGE_POINT_SUBSTATES,
    ANALYTIC_MORI_ABLATION_BACKJUMP_CAGE_MEMORY,
    ANALYTIC_MORI_ABLATION_FREE_LI_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_COMPACT_ANION_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_FREE_LI_PLUS_COMPACT_ANION_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CARRIER_CAGE_ONLY,
    ANALYTIC_MORI_ABLATION_DESCRIPTOR_ATMOSPHERE_RELAXATION_RELEASE_ONLY,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_RELEASE,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_REL_AND_EP_RELEASE,
    ANALYTIC_MORI_ABLATION_TIMESCALE_STRUCTURAL_CAGE_MEMORY,
    REGISTRY_LAMBDA0_PRIMITIVE,
)
STRUCTURAL_CERTIFICATE_ABLATION_MODES = (
    ANALYTIC_MORI_ABLATION_ASSOCIATION_OFF,
    ANALYTIC_MORI_ABLATION_HIGHER_AGGREGATES_OFF,
    ANALYTIC_MORI_ABLATION_ATMOSPHERE_RESISTANCE_OFF,
    ANALYTIC_MORI_ABLATION_JONES_DOLE_VISCOSITY_OFF,
    ANALYTIC_MORI_ABLATION_DIELECTRIC_DECREMENT_OFF,
    ANALYTIC_MORI_ABLATION_CARRIER_CAGE_POINT_SUBSTATES,
    ANALYTIC_MORI_ABLATION_BACKJUMP_CAGE_MEMORY,
    ANALYTIC_MORI_ABLATION_FREE_LI_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_COMPACT_ANION_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_FREE_LI_PLUS_COMPACT_ANION_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_TIMESCALE_STRUCTURAL_CAGE_MEMORY,
)
AGGREGATE_STATE_KINDS = ("LI2A_PLUS", "LIA2_MINUS", "LI2A2_NEUTRAL")
CHARGED_AGGREGATE_STATE_KINDS = ("LI2A_PLUS", "LIA2_MINUS")
OVER_ASSOCIATION_FREE_FRACTION_WARNING_THRESHOLD = 0.02  # Explicit user-declared population audit threshold.
OVER_ASSOCIATION_NEUTRAL_AGGREGATE_WARNING_THRESHOLD = 0.50  # Explicit user-declared population audit threshold.
LARGE_CANCELLATION_RATIO_WARNING_THRESHOLD = 0.90  # Explicit user-declared direct/corrector audit threshold.
DIELECTRIC_COLLAPSE_RATIO_WARNING_THRESHOLD = 0.60  # Explicit user-declared dielectric-head audit threshold.
INVERSE_TARGET_LOW_UNIT_BIN_MAX = 1.0 / 3.0  # Diagnostic unit-interval lower tercile boundary.
INVERSE_TARGET_HIGH_UNIT_BIN_MIN = 2.0 / 3.0  # Diagnostic unit-interval upper tercile boundary.
LOWER_QUARTILE_PROBABILITY = 0.25  # Standard IQR lower quantile probability for audit tables.
UPPER_QUARTILE_PROBABILITY = 0.75  # Standard IQR upper quantile probability for audit tables.
FREE_TRANSLATION_REQUIRED_SCALE_SINGLETON_THRESHOLD = 0.60  # Explicit inverse-audit singleton threshold from the row-52 diagnostic.
FREE_TRANSLATION_REQUIRED_SCALE_EXTREME_THRESHOLD = 0.45  # Explicit out-of-envelope threshold from the row-52 diagnostic.
FREE_TRANSLATION_CLUSTER_MIN_COUNT = 3  # Explicit minimum local-neighbor count for a systematic free-translation cluster.
FREE_TRANSLATION_CLUSTER_SCALE_THRESHOLD = 0.65  # Explicit median required-scale threshold for a systematic suppression cluster.
FREE_TRANSLATION_NEIGHBOR_MOLARITY_WINDOW_M = 0.25  # Explicit local-neighborhood molarity window.
FREE_TRANSLATION_NEIGHBOR_STERIC_WINDOW = 0.05  # Explicit local-neighborhood steric-volume window.
FREE_TRANSLATION_NEIGHBOR_DRIVER_WINDOW = 0.15  # Explicit local-neighborhood obstruction-driver window.
FREE_TRANSLATION_INVERSE_STATUS_GATE_SIGMA_MS_CM = 7.0  # Explicit row-52 diagnostic gate for row-status refinement.
PRIMITIVE_SENSITIVITY_CONFIG_SECTION = "analytic_mori_primitive_sensitivity_audit"
DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION = "analytic_mori_dense_free_volume_obstruction"
DENSE_FREE_LI_OBSTRUCTION_HEAD = "dense_free_volume.free_li_obstruction_strength"
DENSE_COMPACT_ANION_OBSTRUCTION_HEAD = "dense_free_volume.compact_anion_obstruction_strength"
DENSE_STERIC_REFERENCE_HEAD = "dense_free_volume.steric_reference_fraction"
DENSE_FREE_VOLUME_POWER_HEAD = "dense_free_volume.free_volume_power"
DIAGNOSTIC_PRIMITIVE_HEAD_BY_ABLATION_MODE = {
    ANALYTIC_MORI_ABLATION_ASSOCIATION_OFF: "diagnostic.association_off",
    ANALYTIC_MORI_ABLATION_ATMOSPHERE_RESISTANCE_OFF: "diagnostic.atmosphere_resistance_off",
    ANALYTIC_MORI_ABLATION_JONES_DOLE_VISCOSITY_OFF: "diagnostic.jones_dole_viscosity_off",
    ANALYTIC_MORI_ABLATION_DIELECTRIC_DECREMENT_OFF: "diagnostic.dielectric_decrement_off",
    ANALYTIC_MORI_ABLATION_CARRIER_CAGE_POINT_SUBSTATES: "diagnostic.carrier_cage_point_substates",
    ANALYTIC_MORI_ABLATION_BACKJUMP_CAGE_MEMORY: "diagnostic.Li_backjump_cage_memory",
    ANALYTIC_MORI_ABLATION_FREE_LI_LOCAL_OBSTRUCTION: "diagnostic.free_Li_local_obstruction",
    ANALYTIC_MORI_ABLATION_COMPACT_ANION_LOCAL_OBSTRUCTION: "diagnostic.compact_anion_local_obstruction",
    ANALYTIC_MORI_ABLATION_FREE_LI_PLUS_COMPACT_ANION_LOCAL_OBSTRUCTION: "diagnostic.free_Li_plus_compact_anion_local_obstruction",
    ANALYTIC_MORI_ABLATION_TIMESCALE_STRUCTURAL_CAGE_MEMORY: "diagnostic.timescale_structural_cage_memory",
}
DENSE_PRIMITIVE_CONFIG_KEY_BY_HEAD = {
    DENSE_FREE_LI_OBSTRUCTION_HEAD: "free_li_obstruction_strength",
    DENSE_COMPACT_ANION_OBSTRUCTION_HEAD: "compact_anion_obstruction_strength",
    DENSE_STERIC_REFERENCE_HEAD: "steric_reference_fraction",
    DENSE_FREE_VOLUME_POWER_HEAD: "free_volume_power",
}
PRIMITIVE_SENSITIVITY_HEADS = (
    DENSE_FREE_LI_OBSTRUCTION_HEAD,
    DENSE_COMPACT_ANION_OBSTRUCTION_HEAD,
    DENSE_STERIC_REFERENCE_HEAD,
    DENSE_FREE_VOLUME_POWER_HEAD,
)
PRIMITIVE_SENSITIVITY_STATUS_FINITE = "finite_sensitivity"
PRIMITIVE_SENSITIVITY_STATUS_ZERO = "zero_sensitivity"


class AnalyticMoriPredictionStatus(Enum):
    DESCRIPTOR_EQUATION_PREDICTION = "descriptor_equation_prediction"
    PRIMITIVE_WARNING = "primitive_warning"
    EQUATION_DOMAIN_VIOLATION = "equation_domain_violation"
    FREE_TRANSLATION_PRIMITIVE_FAILURE = "free_translation_primitive_failure"
    FREE_TRANSLATION_SINGLETON_RESIDUAL = "free_translation_singleton_residual"
    SYSTEMATIC_FREE_TRANSLATION_SUPPRESSION_CLUSTER = (
        "systematic_free_translation_suppression_cluster"
    )


EQUATION_DOMAIN_BLOCKING_STATUSES = frozenset(
    (AnalyticMoriPredictionStatus.EQUATION_DOMAIN_VIOLATION,)
)


@dataclass(frozen=True)
class AnalyticMoriPropertyDbRow:
    row_id: int
    empirical_sigma_mS_cm: float
    analytic_mori_sigma_mS_cm: float
    residual_mS_cm: float
    uncertainty_bound_mS_cm: float
    sigma_interval_min_mS_cm: float
    sigma_interval_max_mS_cm: float
    certificate_half_width_mS_cm: float
    dominant_uncertainty_head: str
    certificate_covers_empirical: bool
    certified_0p25_mS_cm: bool
    prediction_status: str
    salt_family: str
    solvent_family: str
    additive_basis: str
    salt_molarity_M: float
    epsilon_mixture: float
    epsilon_association: float
    epsilon_atmosphere: float
    effective_dielectric: float
    effective_viscosity_cP: float
    debye_kappa_inv_A: float
    steric_volume_fraction: float
    carrier_relaxation_form_factor_min: float
    carrier_charge_cloud_radius_A_max: float
    atmosphere_ep_trace_kg_s: float
    atmosphere_rel_trace_kg_s: float
    atmosphere_rel_li_li_trace_kg_s: float
    atmosphere_rel_anion_anion_trace_kg_s: float
    atmosphere_rel_li_anion_cross_frobenius_kg_s: float
    atmosphere_rel_anion_anion_cross_frobenius_kg_s: float
    lithium_form_factor_squared: float
    anion_form_factor_squared_min: float
    lithium_anion_cross_form_factor_min: float
    carrier_caged_fraction_max: float
    carrier_caged_diffusion_scale_min: float
    carrier_cage_exchange_rate_max_s_inv: float
    selective_cage_driver: float
    selective_caged_fraction_max: float
    selective_caged_diffusion_scale_min: float
    descriptor_release_driver: float
    atmosphere_relaxation_scale: float
    atmosphere_electrophoretic_scale: float
    backjump_cage_driver: float
    backjump_f_cage_Li: float
    backjump_g_attempt_Li: float
    backjump_p_back_Li: float
    backjump_exit_rate_s_inv: float
    backjump_length_A: float
    backjump_direct_sigma_mS_cm: float
    backjump_corrector_sigma_mS_cm: float
    backjump_net_sigma_delta_mS_cm: float
    ordinary_translation_fraction_Li: float
    free_li_obstruction_factor: float
    free_li_translation_diffusion_scale: float
    free_anion_obstruction_factor_max: float
    free_anion_translation_diffusion_scale_min: float
    obstruction_steric_driver: float
    obstruction_compact_anion_driver: float
    obstruction_carbonate_driver: float
    obstruction_high_salt_driver: float
    obstruction_low_donor_driver: float
    free_li_translation_marginal_net_mS_cm: float
    free_anion_translation_marginal_net_mS_cm: float
    free_li_fraction: float
    free_anion_fraction: float
    neutral_aggregate_fraction: float
    markov_corrector_over_direct: float
    over_association_warning: bool
    large_cancellation_warning: bool
    dielectric_collapse_warning: bool
    uncertified_population_warning: bool
    mass_balance_max_abs_residual_M: float
    row_sum_residual: float
    stationary_residual: float
    detailed_balance_residual: float
    event_reversal_residual_mol_m3_s: float
    direct_mori_sigma_mS_cm: float
    markov_direct_sigma_mS_cm: float
    markov_corrector_sigma_mS_cm: float
    markov_total_sigma_mS_cm: float
    minimum_effective_axis_density_m2_s_mol_m3: float
    markov_event_family_attributions: tuple[
        MarkovAdditiveEventFamilyAttribution,
        ...,
    ]


@dataclass(frozen=True)
class AnalyticMoriPropertyDbFailure:
    row_id: int
    error: str


@dataclass(frozen=True)
class AnalyticMoriFamilyMetrics:
    family_name: str
    count: int
    bias_mS_cm: float
    mae_mS_cm: float
    rmse_mS_cm: float


@dataclass(frozen=True)
class AnalyticMoriAblationPrediction:
    ablation_mode: str
    sigma_mS_cm: float
    residual_mS_cm: float


@dataclass(frozen=True)
class AnalyticMoriAblationMetric:
    ablation_mode: str
    evaluated_rows: int
    failed_rows: int
    mae_mS_cm: float
    rmse_mS_cm: float
    bias_mS_cm: float
    pearson_r: float


@dataclass(frozen=True)
class AnalyticMoriWorstRowPrimitiveDecomposition:
    row_id: int
    salt_family: str
    solvent_family: str
    additive_basis: str
    empirical_sigma_mS_cm: float
    baseline_sigma_mS_cm: float
    baseline_residual_mS_cm: float
    ablation_predictions: tuple[AnalyticMoriAblationPrediction, ...]
    free_li_fraction: float
    free_anion_fraction: float
    ssip_fraction: float
    cip_fraction: float
    charged_aggregate_fraction: float
    neutral_aggregate_fraction: float
    effective_dielectric: float
    effective_viscosity_cP: float
    debye_kappa_inv_A: float
    steric_volume_fraction: float
    carrier_relaxation_form_factor_min: float
    carrier_charge_cloud_radius_A_max: float
    selective_cage_driver: float
    selective_caged_fraction_max: float
    selective_caged_diffusion_scale_min: float
    descriptor_release_driver: float
    atmosphere_relaxation_scale: float
    atmosphere_electrophoretic_scale: float
    timescale_structural_cage_fraction_max: float
    timescale_structural_de_hop_structural_max: float
    timescale_structural_atmosphere_ratio_max: float
    timescale_structural_size_void_ratio_max: float
    timescale_structural_capture_fraction_max: float
    timescale_event_family_attributions: tuple[
        MarkovAdditiveEventFamilyAttribution, ...
    ]
    free_li_obstruction_factor: float
    free_li_translation_diffusion_scale: float
    free_anion_obstruction_factor_max: float
    free_anion_translation_diffusion_scale_min: float
    obstruction_steric_driver: float
    obstruction_compact_anion_driver: float
    obstruction_carbonate_driver: float
    obstruction_high_salt_driver: float
    obstruction_low_donor_driver: float
    free_li_translation_marginal_net_mS_cm: float
    free_anion_translation_marginal_net_mS_cm: float
    sigma_free_li_mS_cm: float
    sigma_free_anion_mS_cm: float
    sigma_ssip_mS_cm: float
    sigma_cip_mS_cm: float
    sigma_aggregates_mS_cm: float
    local_resistance_trace_kg_s: float
    binding_resistance_trace_kg_s: float
    atmosphere_resistance_trace_kg_s: float


@dataclass(frozen=True)
class AnalyticMoriAblationAuditResult:
    labeled_rows: int
    evaluated_rows: int
    failed_rows: int
    ablation_metrics: tuple[AnalyticMoriAblationMetric, ...]
    worst_row_decompositions: tuple[AnalyticMoriWorstRowPrimitiveDecomposition, ...]
    failures: tuple[AnalyticMoriPropertyDbFailure, ...]


@dataclass(frozen=True)
class AnalyticMoriObstructionReachabilityMetric:
    free_li_translation_scale: float
    compact_anion_translation_scale: float
    evaluated_rows: int
    failed_rows: int
    mae_mS_cm: float
    bias_mS_cm: float
    pearson_r: float
    row52_sigma_mS_cm: float
    row53_sigma_mS_cm: float
    lipf6_mae_mS_cm: float
    litfsi_mae_mS_cm: float
    lifsi_mae_mS_cm: float


@dataclass(frozen=True)
class FreeTranslationInverseTarget:
    row_id: int
    empirical_sigma_mS_cm: float
    predicted_sigma_mS_cm: float
    residual_mS_cm: float
    free_li_marginal_mS_cm: float
    free_anion_marginal_mS_cm: float
    free_translation_marginal_mS_cm: float
    required_common_free_scale_to_empirical: float
    required_common_free_scale_to_empirical_clipped: float
    required_common_free_scale_to_gate: float
    required_common_free_scale_to_gate_clipped: float
    required_li_scale_if_anion_fixed: float
    required_li_scale_if_anion_fixed_clipped: float
    required_anion_scale_if_li_fixed: float
    required_anion_scale_if_li_fixed_clipped: float
    salt_family: str
    solvent_family: str
    salt_molarity_M: float
    steric_volume_fraction: float
    obstruction_driver: float
    structural_interval_covers_empirical: bool
    prediction_status: str


@dataclass(frozen=True)
class FreeTranslationInverseNeighborhood:
    row_id: int
    neighbor_count: int
    median_neighbor_required_scale: float
    min_neighbor_required_scale: float
    median_neighbor_residual_mS_cm: float
    same_salt_family_count: int
    same_solvent_family_count: int
    has_systematic_cluster: bool


@dataclass(frozen=True)
class FreeTranslationInverseGroupMetric:
    group_kind: str
    group_label: str
    count: int
    median_required_common_free_scale: float
    iqr_required_common_free_scale: float
    median_required_common_free_scale_clipped: float
    mean_signed_error_mS_cm: float


@dataclass(frozen=True)
class FreeTranslationInverseAuditResult:
    gate_sigma_mS_cm: float
    targets: tuple[FreeTranslationInverseTarget, ...]
    neighborhoods: tuple[FreeTranslationInverseNeighborhood, ...]
    salt_family_metrics: tuple[FreeTranslationInverseGroupMetric, ...]
    solvent_family_metrics: tuple[FreeTranslationInverseGroupMetric, ...]
    steric_volume_bin_metrics: tuple[FreeTranslationInverseGroupMetric, ...]
    obstruction_driver_bin_metrics: tuple[FreeTranslationInverseGroupMetric, ...]


@dataclass(frozen=True)
class PrimitiveSensitivityRow:
    row_id: int
    empirical_sigma_mS_cm: float
    baseline_sigma_mS_cm: float
    residual_mS_cm: float
    primitive_head: str
    baseline_value: float
    sigma_minus_mS_cm: float
    sigma_plus_mS_cm: float
    sensitivity_mS_cm_per_log_unit: float
    required_change_status: str
    required_log_change: float
    required_scale: float
    positive_direction_improves_abs_residual: bool
    positive_direction_can_reduce_residual: bool


@dataclass(frozen=True)
class PrimitiveSensitivityGroup:
    primitive_head: str
    rows_improved_count: int
    rows_worsened_count: int
    mean_abs_residual_after_positive_step_mS_cm: float
    finite_required_scale_count: int
    median_required_scale: float


@dataclass(frozen=True)
class PrimitiveSensitivityAuditResult:
    log_parameter_step: float
    rows: tuple[PrimitiveSensitivityRow, ...]
    groups: tuple[PrimitiveSensitivityGroup, ...]
    failures: tuple[AnalyticMoriPropertyDbFailure, ...]


@dataclass(frozen=True)
class AnalyticMoriPropertyDbAuditResult:
    labeled_rows: int
    evaluated_rows: int
    failed_rows: int
    mae_mS_cm: float
    rmse_mS_cm: float
    bias_mS_cm: float
    mape_percent: float
    r2: float
    pearson_r: float
    certificate_coverage_fraction: float
    certified_0p25_count: int
    descriptor_complete_prediction_count: int
    equation_domain_violation_count: int
    max_mass_balance_residual_M: float
    max_row_sum_residual: float
    max_stationary_residual: float
    max_detailed_balance_residual: float
    max_event_reversal_residual_mol_m3_s: float
    over_association_warning_count: int
    large_cancellation_warning_count: int
    dielectric_collapse_warning_count: int
    uncertified_population_warning_count: int
    salt_family_metrics: tuple[AnalyticMoriFamilyMetrics, ...]
    rows: tuple[AnalyticMoriPropertyDbRow, ...]
    failures: tuple[AnalyticMoriPropertyDbFailure, ...]


@dataclass(frozen=True)
class _AblationEvaluationRecord:
    row_id: int
    empirical_sigma_mS_cm: float
    salt_family: str
    solvent_family: str
    additive_basis: str
    ablation_predictions: tuple[AnalyticMoriAblationPrediction, ...]
    baseline_result: AnalyticMoriPrimitiveResult
    primitive_result_by_mode: Mapping[str, AnalyticMoriPrimitiveResult]


@dataclass(frozen=True)
class _StructuralCertificateEvaluation:
    sigma_min_mS_cm: float
    sigma_max_mS_cm: float
    half_width_mS_cm: float
    dominant_uncertainty_head: str
    certified_0p25_mS_cm: bool


@dataclass(frozen=True)
class _ObstructionReachabilityRecord:
    row_id: int
    empirical_sigma_mS_cm: float
    predicted_sigma_mS_cm: float
    residual_mS_cm: float
    salt_family: str


def audit_analytic_mori_conductivity_against_property_db(
    entries,
    temperature_K: float,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> AnalyticMoriPropertyDbAuditResult:
    """Compare analytic Mori conductivity predictions to empirical conductivity labels."""

    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be positive and finite, got {temperature_K}")

    rows: list[AnalyticMoriPropertyDbRow] = []
    failures: list[AnalyticMoriPropertyDbFailure] = []
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
            canonicalization = canonicalize_empirical_recipe(entry_sections["recipe"])
            analytic_recipe = AnalyticMoriRecipe(
                solvent_volume_fractions=canonicalization.recipe["solvents"],
                salt_molarities_M=canonicalization.recipe["salts"],
                additive_weight_fractions=canonicalization.recipe["additives"],
                temperature_K=temperature_K,
                active_volume_m3=ACTIVE_VOLUME_M3_FOR_MOLAR_CONCENTRATION_READOUT,
            )
            analytic_result = evaluate_analytic_mori_conductivity(
                analytic_recipe,
                uncertainty_budget,
                species_catalog,
            )
            residual_mS_cm = analytic_result.sigma_mS_cm - empirical_sigma_mS_cm
            certificate = _structural_certificate_from_perturbation_scenarios(
                analytic_recipe,
                uncertainty_budget,
                species_catalog,
                analytic_result,
            )
            backjump_cage_result = evaluate_analytic_mori_ablation_conductivity(
                analytic_recipe,
                uncertainty_budget,
                species_catalog,
                ANALYTIC_MORI_ABLATION_BACKJUMP_CAGE_MEMORY,
            )
            uncertainty_bound_mS_cm = certificate.half_width_mS_cm
            certificate_covers_empirical = (
                certificate.sigma_min_mS_cm
                <= empirical_sigma_mS_cm
                <= certificate.sigma_max_mS_cm
            )
            free_li_fraction = _state_lithium_fraction(analytic_result, ("FREE_LI",))
            free_anion_fraction = _free_anion_fraction(analytic_result)
            neutral_aggregate_fraction = _state_lithium_fraction(
                analytic_result,
                ("LI2A2_NEUTRAL",),
            )
            markov_corrector_over_direct = _markov_corrector_over_direct(analytic_result)
            over_association_warning = _over_association_warning(
                free_li_fraction,
                free_anion_fraction,
                neutral_aggregate_fraction,
            )
            large_cancellation_warning = (
                markov_corrector_over_direct
                > LARGE_CANCELLATION_RATIO_WARNING_THRESHOLD
            )
            dielectric_collapse_warning = _dielectric_collapse_warning(analytic_result)
            prediction_status = _prediction_status(
                over_association_warning,
                large_cancellation_warning,
                dielectric_collapse_warning,
            )
            rows.append(
                AnalyticMoriPropertyDbRow(
                    row_id=row_id,
                    empirical_sigma_mS_cm=empirical_sigma_mS_cm,
                    analytic_mori_sigma_mS_cm=analytic_result.sigma_mS_cm,
                    residual_mS_cm=residual_mS_cm,
                    uncertainty_bound_mS_cm=uncertainty_bound_mS_cm,
                    sigma_interval_min_mS_cm=certificate.sigma_min_mS_cm,
                    sigma_interval_max_mS_cm=certificate.sigma_max_mS_cm,
                    certificate_half_width_mS_cm=certificate.half_width_mS_cm,
                    dominant_uncertainty_head=certificate.dominant_uncertainty_head,
                    certificate_covers_empirical=certificate_covers_empirical,
                    certified_0p25_mS_cm=(
                        certificate.certified_0p25_mS_cm
                        and certificate_covers_empirical
                    ),
                    prediction_status=prediction_status,
                    salt_family=_family_label(canonicalization.recipe["salts"]),
                    solvent_family=_family_label(canonicalization.recipe["solvents"]),
                    additive_basis=_family_label(canonicalization.recipe["additives"]),
                    salt_molarity_M=math.fsum(
                        canonicalization.recipe["salts"].values()
                    ),
                    epsilon_mixture=analytic_result.dielectric_heads.epsilon_mixture,
                    epsilon_association=analytic_result.dielectric_heads.epsilon_association,
                    epsilon_atmosphere=analytic_result.dielectric_heads.epsilon_atmosphere,
                    effective_dielectric=analytic_result.effective_dielectric,
                    effective_viscosity_cP=analytic_result.effective_viscosity_cP,
                    debye_kappa_inv_A=analytic_result.bulk_ion_atmosphere_state.kappa_inv_m
                    / 1.0e-10,
                    steric_volume_fraction=(
                        analytic_result.bulk_ion_atmosphere_state.steric_volume_fraction
                    ),
                    carrier_relaxation_form_factor_min=(
                        _minimum_carrier_relaxation_form_factor(analytic_result)
                    ),
                    carrier_charge_cloud_radius_A_max=(
                        _maximum_carrier_charge_cloud_radius_A(analytic_result)
                    ),
                    atmosphere_ep_trace_kg_s=(
                        analytic_result.atmosphere_block_diagnostics.electrophoretic_trace_kg_s
                    ),
                    atmosphere_rel_trace_kg_s=(
                        analytic_result.atmosphere_block_diagnostics.relaxation_trace_kg_s
                    ),
                    atmosphere_rel_li_li_trace_kg_s=(
                        analytic_result.atmosphere_block_diagnostics.relaxation_lithium_self_trace_kg_s
                    ),
                    atmosphere_rel_anion_anion_trace_kg_s=(
                        analytic_result.atmosphere_block_diagnostics.relaxation_anion_self_trace_kg_s
                    ),
                    atmosphere_rel_li_anion_cross_frobenius_kg_s=(
                        analytic_result.atmosphere_block_diagnostics.relaxation_lithium_anion_cross_frobenius_kg_s
                    ),
                    atmosphere_rel_anion_anion_cross_frobenius_kg_s=(
                        analytic_result.atmosphere_block_diagnostics.relaxation_anion_anion_cross_frobenius_kg_s
                    ),
                    lithium_form_factor_squared=(
                        analytic_result.atmosphere_block_diagnostics.lithium_form_factor_squared
                    ),
                    anion_form_factor_squared_min=(
                        analytic_result.atmosphere_block_diagnostics.minimum_anion_form_factor_squared
                    ),
                    lithium_anion_cross_form_factor_min=(
                        analytic_result.atmosphere_block_diagnostics.minimum_lithium_anion_cross_form_factor
                    ),
                    carrier_caged_fraction_max=_maximum_carrier_caged_fraction(
                        analytic_result
                    ),
                    carrier_caged_diffusion_scale_min=(
                        _minimum_carrier_caged_diffusion_scale(analytic_result)
                    ),
                    carrier_cage_exchange_rate_max_s_inv=(
                        _maximum_carrier_cage_exchange_rate_s_inv(analytic_result)
                    ),
                    selective_cage_driver=_maximum_selective_cage_driver(
                        analytic_result
                    ),
                    selective_caged_fraction_max=(
                        _maximum_selective_caged_fraction(analytic_result)
                    ),
                    selective_caged_diffusion_scale_min=(
                        _minimum_selective_caged_diffusion_scale(analytic_result)
                    ),
                    descriptor_release_driver=(
                        analytic_result.descriptor_atmosphere_release_primitive.release_driver
                    ),
                    atmosphere_relaxation_scale=(
                        analytic_result.descriptor_atmosphere_release_primitive.relaxation_scale
                    ),
                    atmosphere_electrophoretic_scale=(
                        analytic_result.descriptor_atmosphere_release_primitive.electrophoretic_scale
                    ),
                    backjump_cage_driver=_backjump_cage_driver(analytic_result),
                    backjump_f_cage_Li=_backjump_cage_occupancy_fraction(analytic_result),
                    backjump_g_attempt_Li=_backjump_attempt_fraction(analytic_result),
                    backjump_p_back_Li=_backjump_probability(analytic_result),
                    backjump_exit_rate_s_inv=_backjump_exit_rate_s_inv(analytic_result),
                    backjump_length_A=_backjump_length_A(analytic_result),
                    backjump_direct_sigma_mS_cm=_backjump_direct_sigma_mS_cm(
                        analytic_result
                    ),
                    backjump_corrector_sigma_mS_cm=(
                        backjump_cage_result.markov_additive_result.corrector_sigma_mS_cm
                        - analytic_result.markov_additive_result.corrector_sigma_mS_cm
                    ),
                    backjump_net_sigma_delta_mS_cm=(
                        backjump_cage_result.sigma_mS_cm - analytic_result.sigma_mS_cm
                    ),
                    ordinary_translation_fraction_Li=(
                        _backjump_ordinary_translation_fraction(analytic_result)
                    ),
                    free_li_obstruction_factor=(
                        _free_li_obstruction_primitive(
                            analytic_result
                        ).obstruction_factor
                    ),
                    free_li_translation_diffusion_scale=(
                        _free_li_obstruction_primitive(
                            analytic_result
                        ).diffusion_scale
                    ),
                    free_anion_obstruction_factor_max=(
                        _maximum_free_anion_obstruction_factor_from_primitives(
                            analytic_result
                        )
                    ),
                    free_anion_translation_diffusion_scale_min=(
                        _minimum_free_anion_translation_scale_from_primitives(
                            analytic_result
                        )
                    ),
                    obstruction_steric_driver=_maximum_obstruction_driver(
                        analytic_result,
                        "steric_driver",
                    ),
                    obstruction_compact_anion_driver=(
                        _maximum_obstruction_driver(
                            analytic_result,
                            "compact_anion_driver",
                        )
                    ),
                    obstruction_carbonate_driver=_maximum_obstruction_driver(
                        analytic_result,
                        "carbonate_driver",
                    ),
                    obstruction_high_salt_driver=_maximum_obstruction_driver(
                        analytic_result,
                        "high_salt_driver",
                    ),
                    obstruction_low_donor_driver=_maximum_obstruction_driver(
                        analytic_result,
                        "low_donor_driver",
                    ),
                    free_li_translation_marginal_net_mS_cm=(
                        _event_family_marginal_net_sigma_mS_cm(
                            analytic_result,
                            EVENT_FAMILY_ORDINARY_FREE_LI_TRANSLATION,
                        )
                    ),
                    free_anion_translation_marginal_net_mS_cm=(
                        _event_family_marginal_net_sigma_mS_cm(
                            analytic_result,
                            EVENT_FAMILY_ORDINARY_FREE_ANION_TRANSLATION,
                        )
                    ),
                    free_li_fraction=free_li_fraction,
                    free_anion_fraction=free_anion_fraction,
                    neutral_aggregate_fraction=neutral_aggregate_fraction,
                    markov_corrector_over_direct=markov_corrector_over_direct,
                    over_association_warning=over_association_warning,
                    large_cancellation_warning=large_cancellation_warning,
                    dielectric_collapse_warning=dielectric_collapse_warning,
                    uncertified_population_warning=(
                        over_association_warning or large_cancellation_warning
                    ),
                    mass_balance_max_abs_residual_M=(
                        analytic_result.mass_balance.max_abs_residual_M
                    ),
                    row_sum_residual=(
                        analytic_result.markov_additive_result.validation.row_sum_residual
                    ),
                    stationary_residual=(
                        analytic_result.markov_additive_result.validation.stationary_residual_mol_m3_s
                    ),
                    detailed_balance_residual=(
                        analytic_result.markov_additive_result.validation.detailed_balance_residual_mol_m3_s
                    ),
                    event_reversal_residual_mol_m3_s=(
                        analytic_result.markov_additive_result.event_reversal_residual_mol_m3_s
                    ),
                    direct_mori_sigma_mS_cm=analytic_result.direct_mori_result.sigma_mS_cm,
                    markov_direct_sigma_mS_cm=(
                        analytic_result.markov_additive_result.direct_sigma_mS_cm
                    ),
                    markov_corrector_sigma_mS_cm=(
                        analytic_result.markov_additive_result.corrector_sigma_mS_cm
                    ),
                    markov_total_sigma_mS_cm=(
                        analytic_result.markov_additive_result.sigma_mS_cm
                    ),
                    minimum_effective_axis_density_m2_s_mol_m3=(
                        analytic_result.markov_additive_result.minimum_effective_axis_density_m2_s_mol_m3
                    ),
                    markov_event_family_attributions=(
                        analytic_result.markov_event_family_attributions
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                AnalyticMoriPropertyDbFailure(
                    row_id=row_id,
                    error=str(exc),
                )
            )

    rows = _rows_with_free_translation_inverse_status(rows)
    metrics = _dataset_metrics(rows)
    return AnalyticMoriPropertyDbAuditResult(
        labeled_rows=labeled_rows,
        evaluated_rows=len(rows),
        failed_rows=len(failures),
        mae_mS_cm=metrics["mae_mS_cm"],
        rmse_mS_cm=metrics["rmse_mS_cm"],
        bias_mS_cm=metrics["bias_mS_cm"],
        mape_percent=metrics["mape_percent"],
        r2=metrics["r2"],
        pearson_r=metrics["pearson_r"],
        certificate_coverage_fraction=_certificate_coverage_fraction(rows),
        certified_0p25_count=sum(1 for row in rows if row.certified_0p25_mS_cm),
        descriptor_complete_prediction_count=sum(
            1
            for row in rows
            if _is_descriptor_complete_prediction_status(row.prediction_status)
        ),
        equation_domain_violation_count=sum(
            1
            for row in rows
            if _is_equation_domain_violation_status(row.prediction_status)
        ),
        max_mass_balance_residual_M=_maximum_row_value(
            rows,
            "mass_balance_max_abs_residual_M",
        ),
        max_row_sum_residual=_maximum_row_value(rows, "row_sum_residual"),
        max_stationary_residual=_maximum_row_value(rows, "stationary_residual"),
        max_detailed_balance_residual=_maximum_row_value(
            rows,
            "detailed_balance_residual",
        ),
        max_event_reversal_residual_mol_m3_s=_maximum_row_value(
            rows,
            "event_reversal_residual_mol_m3_s",
        ),
        over_association_warning_count=sum(
            1 for row in rows if row.over_association_warning
        ),
        large_cancellation_warning_count=sum(
            1 for row in rows if row.large_cancellation_warning
        ),
        dielectric_collapse_warning_count=sum(
            1 for row in rows if row.dielectric_collapse_warning
        ),
        uncertified_population_warning_count=sum(
            1 for row in rows if row.uncertified_population_warning
        ),
        salt_family_metrics=_salt_family_metrics(rows),
        rows=tuple(rows),
        failures=tuple(failures),
    )


def audit_analytic_mori_ablation_suite_against_property_db(
    entries,
    temperature_K: float,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    worst_row_count: int,
) -> AnalyticMoriAblationAuditResult:
    """Run primitive-head ablations against empirical conductivity labels."""

    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be positive and finite, got {temperature_K}")
    if worst_row_count <= 0:
        raise ValueError("worst_row_count must be positive")

    records: list[_AblationEvaluationRecord] = []
    failures: list[AnalyticMoriPropertyDbFailure] = []
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
            canonicalization = canonicalize_empirical_recipe(entry_sections["recipe"])
            analytic_recipe = AnalyticMoriRecipe(
                solvent_volume_fractions=canonicalization.recipe["solvents"],
                salt_molarities_M=canonicalization.recipe["salts"],
                additive_weight_fractions=canonicalization.recipe["additives"],
                temperature_K=temperature_K,
                active_volume_m3=ACTIVE_VOLUME_M3_FOR_MOLAR_CONCENTRATION_READOUT,
            )
            primitive_result_by_mode = _ablation_primitive_results(
                analytic_recipe,
                uncertainty_budget,
                species_catalog,
            )
            predictions = _ablation_predictions(
                empirical_sigma_mS_cm,
                analytic_recipe,
                primitive_result_by_mode,
                species_catalog,
            )
            records.append(
                _AblationEvaluationRecord(
                    row_id=row_id,
                    empirical_sigma_mS_cm=empirical_sigma_mS_cm,
                    salt_family=_family_label(canonicalization.recipe["salts"]),
                    solvent_family=_family_label(canonicalization.recipe["solvents"]),
                    additive_basis=_family_label(canonicalization.recipe["additives"]),
                    ablation_predictions=predictions,
                    baseline_result=primitive_result_by_mode[ANALYTIC_MORI_ABLATION_BASELINE],
                    primitive_result_by_mode=dict(primitive_result_by_mode),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                AnalyticMoriPropertyDbFailure(
                    row_id=row_id,
                    error=str(exc),
                )
            )

    return AnalyticMoriAblationAuditResult(
        labeled_rows=labeled_rows,
        evaluated_rows=len(records),
        failed_rows=len(failures),
        ablation_metrics=_ablation_metrics(records, failures),
        worst_row_decompositions=_worst_row_decompositions(records, worst_row_count, temperature_K),
        failures=tuple(failures),
    )


def audit_analytic_mori_obstruction_reachability_against_property_db(
    entries,
    temperature_K: float,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    free_li_translation_scales: tuple[float, ...],
    compact_anion_translation_scales: tuple[float, ...],
    tracked_row52_id: int,
    tracked_row53_id: int,
) -> tuple[AnalyticMoriObstructionReachabilityMetric, ...]:
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be positive and finite, got {temperature_K}")
    validated_free_li_scales = tuple(
        _reachability_translation_scale(
            free_li_translation_scale,
            "free_li_translation_scales",
        )
        for free_li_translation_scale in free_li_translation_scales
    )
    validated_compact_anion_scales = tuple(
        _reachability_translation_scale(
            compact_anion_translation_scale,
            "compact_anion_translation_scales",
        )
        for compact_anion_translation_scale in compact_anion_translation_scales
    )
    if not validated_free_li_scales:
        raise ValueError("free_li_translation_scales must contain at least one scale")
    if not validated_compact_anion_scales:
        raise ValueError("compact_anion_translation_scales must contain at least one scale")

    metrics: list[AnalyticMoriObstructionReachabilityMetric] = []
    for free_li_translation_scale in validated_free_li_scales:
        for compact_anion_translation_scale in validated_compact_anion_scales:
            records: list[_ObstructionReachabilityRecord] = []
            failures: list[AnalyticMoriPropertyDbFailure] = []
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
                    canonicalization = canonicalize_empirical_recipe(
                        entry_sections["recipe"]
                    )
                    analytic_recipe = AnalyticMoriRecipe(
                        solvent_volume_fractions=canonicalization.recipe["solvents"],
                        salt_molarities_M=canonicalization.recipe["salts"],
                        additive_weight_fractions=canonicalization.recipe["additives"],
                        temperature_K=temperature_K,
                        active_volume_m3=(
                            ACTIVE_VOLUME_M3_FOR_MOLAR_CONCENTRATION_READOUT
                        ),
                    )
                    analytic_result = (
                        evaluate_analytic_mori_forced_free_carrier_obstruction_conductivity(
                            analytic_recipe,
                            uncertainty_budget,
                            species_catalog,
                            free_li_translation_scale,
                            compact_anion_translation_scale,
                        )
                    )
                    predicted_sigma_mS_cm = analytic_result.sigma_mS_cm
                    records.append(
                        _ObstructionReachabilityRecord(
                            row_id=row_id,
                            empirical_sigma_mS_cm=empirical_sigma_mS_cm,
                            predicted_sigma_mS_cm=predicted_sigma_mS_cm,
                            residual_mS_cm=(
                                predicted_sigma_mS_cm - empirical_sigma_mS_cm
                            ),
                            salt_family=_family_label(
                                canonicalization.recipe["salts"]
                            ),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    failures.append(
                        AnalyticMoriPropertyDbFailure(
                            row_id=row_id,
                            error=str(exc),
                        )
                    )
            if len(records) < 2:
                raise ValueError(
                    "obstruction reachability audit requires at least two evaluated rows"
                )
            if labeled_rows <= 0:
                raise ValueError("obstruction reachability audit found no labeled rows")
            metrics.append(
                _obstruction_reachability_metric(
                    records,
                    failures,
                    free_li_translation_scale,
                    compact_anion_translation_scale,
                    tracked_row52_id,
                    tracked_row53_id,
                )
            )
    return tuple(metrics)


def compute_free_translation_inverse_target_audit(
    rows: Sequence[AnalyticMoriPropertyDbRow],
    gate_sigma_mS_cm: float,
) -> FreeTranslationInverseAuditResult:
    if not math.isfinite(gate_sigma_mS_cm) or gate_sigma_mS_cm <= 0.0:
        raise ValueError(
            f"gate_sigma_mS_cm must be positive and finite, got {gate_sigma_mS_cm}"
        )
    raw_targets = tuple(
        _free_translation_inverse_target(row, gate_sigma_mS_cm)
        for row in rows
    )
    if not raw_targets:
        raise ValueError("free-translation inverse target audit requires rows")
    neighborhoods = tuple(
        _free_translation_inverse_neighborhood(target, raw_targets)
        for target in raw_targets
    )
    neighborhood_by_row_id = {
        neighborhood.row_id: neighborhood
        for neighborhood in neighborhoods
    }
    targets = tuple(
        replace(
            target,
            prediction_status=_free_translation_inverse_status(
                target,
                neighborhood_by_row_id[target.row_id],
            ).value,
        )
        for target in raw_targets
    )
    return FreeTranslationInverseAuditResult(
        gate_sigma_mS_cm=float(gate_sigma_mS_cm),
        targets=targets,
        neighborhoods=neighborhoods,
        salt_family_metrics=_free_translation_group_metrics(
            targets,
            "salt_family",
            "salt_family",
        ),
        solvent_family_metrics=_free_translation_group_metrics(
            targets,
            "solvent_family",
            "solvent_family",
        ),
        steric_volume_bin_metrics=_free_translation_bin_group_metrics(
            targets,
            "steric_volume_bin",
            "steric_volume_fraction",
        ),
        obstruction_driver_bin_metrics=_free_translation_bin_group_metrics(
            targets,
            "obstruction_driver_bin",
            "obstruction_driver",
        ),
    )


def audit_analytic_mori_primitive_sensitivity_against_property_db(
    entries,
    temperature_K: float,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    row_ids: tuple[int, ...],
) -> PrimitiveSensitivityAuditResult:
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be positive and finite, got {temperature_K}")
    if not row_ids:
        raise ValueError("primitive sensitivity audit requires at least one row id")
    selected_row_ids = frozenset(row_ids)
    if len(selected_row_ids) != len(row_ids):
        raise ValueError("primitive sensitivity audit row ids must be unique")
    log_parameter_step = _primitive_sensitivity_log_parameter_step()
    rows: list[PrimitiveSensitivityRow] = []
    failures: list[AnalyticMoriPropertyDbFailure] = []
    seen_row_ids: set[int] = set()

    for row_id, entry in enumerate(entries):
        if row_id not in selected_row_ids:
            continue
        seen_row_ids.add(row_id)
        try:
            entry_sections = _require_entry(entry, row_id)
            properties = entry_sections["properties"]
            if "conductivity_mS_cm" not in properties:
                raise ValueError(f"DATA[{row_id}].properties missing conductivity_mS_cm")
            empirical_sigma_mS_cm = require_float(
                properties,
                "conductivity_mS_cm",
                f"DATA[{row_id}].properties",
            )
            canonicalization = canonicalize_empirical_recipe(entry_sections["recipe"])
            analytic_recipe = AnalyticMoriRecipe(
                solvent_volume_fractions=canonicalization.recipe["solvents"],
                salt_molarities_M=canonicalization.recipe["salts"],
                additive_weight_fractions=canonicalization.recipe["additives"],
                temperature_K=temperature_K,
                active_volume_m3=ACTIVE_VOLUME_M3_FOR_MOLAR_CONCENTRATION_READOUT,
            )
            baseline_result = evaluate_analytic_mori_conductivity(
                analytic_recipe,
                uncertainty_budget,
                species_catalog,
            )
            for primitive_head in PRIMITIVE_SENSITIVITY_HEADS:
                rows.append(
                    _primitive_sensitivity_row(
                        row_id,
                        empirical_sigma_mS_cm,
                        analytic_recipe,
                        uncertainty_budget,
                        species_catalog,
                        baseline_result,
                        primitive_head,
                        log_parameter_step,
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                AnalyticMoriPropertyDbFailure(
                    row_id=row_id,
                    error=str(exc),
                )
            )

    missing_row_ids = tuple(
        sorted(selected_row_ids.difference(seen_row_ids))
    )
    if missing_row_ids:
        raise ValueError(
            f"primitive sensitivity audit missing selected rows {missing_row_ids}"
        )
    if not rows:
        raise ValueError("primitive sensitivity audit produced no rows")
    return PrimitiveSensitivityAuditResult(
        log_parameter_step=log_parameter_step,
        rows=tuple(rows),
        groups=_primitive_sensitivity_groups(rows),
        failures=tuple(failures),
    )


def _primitive_sensitivity_log_parameter_step() -> float:
    physics_config = load_physics_config()
    sensitivity_config = require_config(
        physics_config,
        PRIMITIVE_SENSITIVITY_CONFIG_SECTION,
        context="_primitive_sensitivity_log_parameter_step",
    )
    if not isinstance(sensitivity_config, dict):
        raise TypeError(f"{PRIMITIVE_SENSITIVITY_CONFIG_SECTION} must be a config object")
    log_parameter_step = require_float(
        sensitivity_config,
        "log_parameter_step",
        PRIMITIVE_SENSITIVITY_CONFIG_SECTION,
    )
    if not math.isfinite(log_parameter_step) or log_parameter_step <= 0.0:
        raise ValueError(
            f"{PRIMITIVE_SENSITIVITY_CONFIG_SECTION}.log_parameter_step "
            "must be positive and finite"
        )
    return float(log_parameter_step)


def _primitive_sensitivity_row(
    row_id: int,
    empirical_sigma_mS_cm: float,
    analytic_recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    baseline_result: AnalyticMoriPrimitiveResult,
    primitive_head: str,
    log_parameter_step: float,
) -> PrimitiveSensitivityRow:
    baseline_sigma_mS_cm = baseline_result.sigma_mS_cm
    residual_mS_cm = baseline_sigma_mS_cm - empirical_sigma_mS_cm
    (
        baseline_value,
        sigma_minus_mS_cm,
        sigma_plus_mS_cm,
    ) = _primitive_sensitivity_sigmas(
        analytic_recipe,
        uncertainty_budget,
        species_catalog,
        baseline_result,
        primitive_head,
        log_parameter_step,
    )
    sensitivity_mS_cm_per_log_unit = (
        sigma_plus_mS_cm - sigma_minus_mS_cm
    ) / (2.0 * log_parameter_step)
    (
        required_change_status,
        required_log_change,
        required_scale,
    ) = _required_change_from_sensitivity(
        residual_mS_cm,
        sensitivity_mS_cm_per_log_unit,
    )
    positive_residual_mS_cm = sigma_plus_mS_cm - empirical_sigma_mS_cm
    return PrimitiveSensitivityRow(
        row_id=row_id,
        empirical_sigma_mS_cm=float(empirical_sigma_mS_cm),
        baseline_sigma_mS_cm=float(baseline_sigma_mS_cm),
        residual_mS_cm=float(residual_mS_cm),
        primitive_head=primitive_head,
        baseline_value=float(baseline_value),
        sigma_minus_mS_cm=float(sigma_minus_mS_cm),
        sigma_plus_mS_cm=float(sigma_plus_mS_cm),
        sensitivity_mS_cm_per_log_unit=float(sensitivity_mS_cm_per_log_unit),
        required_change_status=required_change_status,
        required_log_change=float(required_log_change),
        required_scale=float(required_scale),
        positive_direction_improves_abs_residual=(
            abs(positive_residual_mS_cm) < abs(residual_mS_cm)
        ),
        positive_direction_can_reduce_residual=(
            math.isfinite(sensitivity_mS_cm_per_log_unit)
            and residual_mS_cm * sensitivity_mS_cm_per_log_unit < 0.0
        ),
    )


def _primitive_sensitivity_sigmas(
    analytic_recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    baseline_result: AnalyticMoriPrimitiveResult,
    primitive_head: str,
    log_parameter_step: float,
) -> tuple[float, float, float]:
    if primitive_head in DENSE_PRIMITIVE_CONFIG_KEY_BY_HEAD:
        return _dense_config_parameter_sensitivity_sigmas(
            analytic_recipe,
            uncertainty_budget,
            species_catalog,
            primitive_head,
            log_parameter_step,
        )
    ablation_mode = _ablation_mode_for_diagnostic_primitive_head(primitive_head)
    ablation_result = evaluate_analytic_mori_ablation_conductivity(
        analytic_recipe,
        uncertainty_budget,
        species_catalog,
        ablation_mode,
    )
    return (
        1.0,
        baseline_result.sigma_mS_cm,
        ablation_result.sigma_mS_cm,
    )


def _dense_config_parameter_sensitivity_sigmas(
    analytic_recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    primitive_head: str,
    log_parameter_step: float,
) -> tuple[float, float, float]:
    config_key = DENSE_PRIMITIVE_CONFIG_KEY_BY_HEAD[primitive_head]
    baseline_value = _dense_config_parameter_value(config_key)
    sigma_minus_mS_cm = _evaluate_with_dense_config_parameter_scale(
        analytic_recipe,
        uncertainty_budget,
        species_catalog,
        config_key,
        math.exp(-log_parameter_step),
    )
    sigma_plus_mS_cm = _evaluate_with_dense_config_parameter_scale(
        analytic_recipe,
        uncertainty_budget,
        species_catalog,
        config_key,
        math.exp(log_parameter_step),
    )
    return (
        baseline_value,
        sigma_minus_mS_cm,
        sigma_plus_mS_cm,
    )


def _dense_config_parameter_value(config_key: str) -> float:
    dense_config = _dense_free_volume_config_section()
    return require_float(
        dense_config,
        config_key,
        DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION,
    )


def _evaluate_with_dense_config_parameter_scale(
    analytic_recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    config_key: str,
    parameter_scale: float,
) -> float:
    dense_config = _dense_free_volume_config_section()
    baseline_value = require_float(
        dense_config,
        config_key,
        DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION,
    )
    perturbed_value = baseline_value * parameter_scale
    if not math.isfinite(perturbed_value) or perturbed_value <= 0.0:
        raise ValueError(f"{config_key} perturbation must remain positive and finite")
    if config_key == "steric_reference_fraction" and perturbed_value >= 1.0:
        raise ValueError("steric_reference_fraction perturbation must stay below one")
    dense_config[config_key] = perturbed_value
    try:
        result = evaluate_analytic_mori_conductivity(
            analytic_recipe,
            uncertainty_budget,
            species_catalog,
        )
    finally:
        dense_config[config_key] = baseline_value
    return float(result.sigma_mS_cm)


def _dense_free_volume_config_section() -> dict:
    physics_config = load_physics_config()
    dense_config = require_config(
        physics_config,
        DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION,
        context="_dense_free_volume_config_section",
    )
    if not isinstance(dense_config, dict):
        raise TypeError(
            f"{DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION} must be a config object"
        )
    return dense_config


def _ablation_mode_for_diagnostic_primitive_head(primitive_head: str) -> str:
    for ablation_mode, diagnostic_primitive_head in (
        DIAGNOSTIC_PRIMITIVE_HEAD_BY_ABLATION_MODE.items()
    ):
        if diagnostic_primitive_head == primitive_head:
            return ablation_mode
    raise ValueError(f"unknown primitive sensitivity head {primitive_head}")


def _required_change_from_sensitivity(
    residual_mS_cm: float,
    sensitivity_mS_cm_per_log_unit: float,
) -> tuple[str, float, float]:
    if (
        not math.isfinite(sensitivity_mS_cm_per_log_unit)
        or sensitivity_mS_cm_per_log_unit == 0.0
    ):
        return (
            PRIMITIVE_SENSITIVITY_STATUS_ZERO,
            0.0,
            1.0,
        )
    required_log_change = -residual_mS_cm / sensitivity_mS_cm_per_log_unit
    return (
        PRIMITIVE_SENSITIVITY_STATUS_FINITE,
        float(required_log_change),
        float(math.exp(required_log_change)),
    )


def _primitive_sensitivity_groups(
    rows: Sequence[PrimitiveSensitivityRow],
) -> tuple[PrimitiveSensitivityGroup, ...]:
    groups: list[PrimitiveSensitivityGroup] = []
    for primitive_head in PRIMITIVE_SENSITIVITY_HEADS:
        head_rows = tuple(
            row
            for row in rows
            if row.primitive_head == primitive_head
        )
        if not head_rows:
            continue
        positive_abs_residuals = tuple(
            abs(row.sigma_plus_mS_cm - row.empirical_sigma_mS_cm)
            for row in head_rows
        )
        finite_required_scales = tuple(
            row.required_scale
            for row in head_rows
            if row.required_change_status == PRIMITIVE_SENSITIVITY_STATUS_FINITE
        )
        groups.append(
            PrimitiveSensitivityGroup(
                primitive_head=primitive_head,
                rows_improved_count=sum(
                    1 for row in head_rows if row.positive_direction_improves_abs_residual
                ),
                rows_worsened_count=sum(
                    1
                    for row in head_rows
                    if (
                        abs(row.sigma_plus_mS_cm - row.empirical_sigma_mS_cm)
                        > abs(row.residual_mS_cm)
                    )
                ),
                mean_abs_residual_after_positive_step_mS_cm=float(
                    np.mean(np.asarray(positive_abs_residuals, dtype=float))
                ),
                finite_required_scale_count=len(finite_required_scales),
                median_required_scale=(
                    float(np.median(np.asarray(finite_required_scales, dtype=float)))
                    if finite_required_scales
                    else 1.0
                ),
            )
        )
    return tuple(groups)


def _ablation_primitive_results(
    analytic_recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> dict[str, AnalyticMoriPrimitiveResult]:
    results: dict[str, AnalyticMoriPrimitiveResult] = {}
    for ablation_mode in SUPPORTED_ANALYTIC_MORI_ABLATIONS:
        results[ablation_mode] = evaluate_analytic_mori_ablation_conductivity(
            analytic_recipe,
            uncertainty_budget,
            species_catalog,
            ablation_mode,
        )
    return results


def _ablation_predictions(
    empirical_sigma_mS_cm: float,
    analytic_recipe: AnalyticMoriRecipe,
    primitive_result_by_mode: Mapping[str, AnalyticMoriPrimitiveResult],
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> tuple[AnalyticMoriAblationPrediction, ...]:
    predictions: list[AnalyticMoriAblationPrediction] = []
    for ablation_mode in ANALYTIC_MORI_AUDIT_ABLATION_MODES:
        if ablation_mode == REGISTRY_LAMBDA0_PRIMITIVE:
            sigma_mS_cm = _registry_lambda0_sigma_mS_cm(
                analytic_recipe.salt_molarities_M,
                species_catalog,
            )
        else:
            sigma_mS_cm = primitive_result_by_mode[ablation_mode].sigma_mS_cm
        predictions.append(
            AnalyticMoriAblationPrediction(
                ablation_mode=ablation_mode,
                sigma_mS_cm=float(sigma_mS_cm),
                residual_mS_cm=float(sigma_mS_cm - empirical_sigma_mS_cm),
            )
        )
    return tuple(predictions)


def _registry_lambda0_sigma_mS_cm(
    salt_molarities_M: Mapping[str, float],
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> float:
    conductivity_mS_cm = 0.0
    for salt_name, molarity_M in salt_molarities_M.items():
        salt_record = species_catalog.salts[salt_name]
        lambda_0_S_cm2_mol = salt_record["Lambda_0"]
        if not isinstance(lambda_0_S_cm2_mol, (int, float)):
            raise TypeError(f"salt.{salt_name}.Lambda_0 must be numeric")
        parsed_lambda_0 = float(lambda_0_S_cm2_mol)
        if not math.isfinite(parsed_lambda_0) or parsed_lambda_0 <= 0.0:
            raise ValueError(f"salt.{salt_name}.Lambda_0 must be positive and finite")
        conductivity_mS_cm += parsed_lambda_0 * float(molarity_M)
    return float(conductivity_mS_cm)


def _ablation_metrics(
    records: Sequence[_AblationEvaluationRecord],
    failures: Sequence[AnalyticMoriPropertyDbFailure],
) -> tuple[AnalyticMoriAblationMetric, ...]:
    metrics: list[AnalyticMoriAblationMetric] = []
    failed_rows = len(failures)
    for ablation_mode in ANALYTIC_MORI_AUDIT_ABLATION_MODES:
        empirical_values: list[float] = []
        predicted_values: list[float] = []
        for record in records:
            prediction = _prediction_for_mode(record.ablation_predictions, ablation_mode)
            empirical_values.append(record.empirical_sigma_mS_cm)
            predicted_values.append(prediction.sigma_mS_cm)
        if len(empirical_values) < 2:
            raise ValueError(f"ablation {ablation_mode} requires at least two evaluated rows")
        empirical_array = np.asarray(empirical_values, dtype=float)
        predicted_array = np.asarray(predicted_values, dtype=float)
        residual_array = predicted_array - empirical_array
        metrics.append(
            AnalyticMoriAblationMetric(
                ablation_mode=ablation_mode,
                evaluated_rows=len(records),
                failed_rows=failed_rows,
                mae_mS_cm=float(np.mean(np.abs(residual_array))),
                rmse_mS_cm=float(math.sqrt(float(np.mean(residual_array * residual_array)))),
                bias_mS_cm=float(np.mean(residual_array)),
                pearson_r=float(np.corrcoef(empirical_array, predicted_array)[0, 1]),
            )
        )
    return tuple(metrics)


def _worst_row_decompositions(
    records: Sequence[_AblationEvaluationRecord],
    worst_row_count: int,
    temperature_K: float,
) -> tuple[AnalyticMoriWorstRowPrimitiveDecomposition, ...]:
    worst_records = sorted(
        records,
        key=lambda record: abs(
            _prediction_for_mode(
                record.ablation_predictions,
                ANALYTIC_MORI_ABLATION_BASELINE,
            ).residual_mS_cm
        ),
        reverse=True,
    )[:worst_row_count]
    return tuple(
        _worst_row_decomposition(record, temperature_K)
        for record in worst_records
    )


def _worst_row_decomposition(
    record: _AblationEvaluationRecord,
    temperature_K: float,
) -> AnalyticMoriWorstRowPrimitiveDecomposition:
    baseline_prediction = _prediction_for_mode(
        record.ablation_predictions,
        ANALYTIC_MORI_ABLATION_BASELINE,
    )
    baseline_result = record.baseline_result
    candidate_diagnostic_result = record.primitive_result_by_mode[
        ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_REL_AND_EP_RELEASE
    ]
    timescale_diagnostic_result = record.primitive_result_by_mode[
        ANALYTIC_MORI_ABLATION_TIMESCALE_STRUCTURAL_CAGE_MEMORY
    ]
    return AnalyticMoriWorstRowPrimitiveDecomposition(
        row_id=record.row_id,
        salt_family=record.salt_family,
        solvent_family=record.solvent_family,
        additive_basis=record.additive_basis,
        empirical_sigma_mS_cm=record.empirical_sigma_mS_cm,
        baseline_sigma_mS_cm=baseline_prediction.sigma_mS_cm,
        baseline_residual_mS_cm=baseline_prediction.residual_mS_cm,
        ablation_predictions=record.ablation_predictions,
        free_li_fraction=_state_lithium_fraction(baseline_result, ("FREE_LI",)),
        free_anion_fraction=_free_anion_fraction(baseline_result),
        ssip_fraction=_state_lithium_fraction(baseline_result, ("SSIP",)),
        cip_fraction=_state_lithium_fraction(baseline_result, ("CIP",)),
        charged_aggregate_fraction=_state_lithium_fraction(
            baseline_result,
            CHARGED_AGGREGATE_STATE_KINDS,
        ),
        neutral_aggregate_fraction=_state_lithium_fraction(
            baseline_result,
            ("LI2A2_NEUTRAL",),
        ),
        effective_dielectric=baseline_result.effective_dielectric,
        effective_viscosity_cP=baseline_result.effective_viscosity_cP,
        debye_kappa_inv_A=baseline_result.bulk_ion_atmosphere_state.kappa_inv_m
        * ANGSTROM_PER_M,
        steric_volume_fraction=(
            baseline_result.bulk_ion_atmosphere_state.steric_volume_fraction
        ),
        carrier_relaxation_form_factor_min=_minimum_carrier_relaxation_form_factor(
            baseline_result
        ),
        carrier_charge_cloud_radius_A_max=_maximum_carrier_charge_cloud_radius_A(
            baseline_result
        ),
        selective_cage_driver=_maximum_selective_cage_driver(
            candidate_diagnostic_result
        ),
        selective_caged_fraction_max=_maximum_selective_caged_fraction(
            candidate_diagnostic_result
        ),
        selective_caged_diffusion_scale_min=(
            _minimum_selective_caged_diffusion_scale(candidate_diagnostic_result)
        ),
        descriptor_release_driver=(
            candidate_diagnostic_result.descriptor_atmosphere_release_primitive.release_driver
        ),
        atmosphere_relaxation_scale=(
            candidate_diagnostic_result.descriptor_atmosphere_release_primitive.relaxation_scale
        ),
        atmosphere_electrophoretic_scale=(
            candidate_diagnostic_result.descriptor_atmosphere_release_primitive.electrophoretic_scale
        ),
        timescale_structural_cage_fraction_max=(
            _maximum_timescale_structural_cage_fraction(timescale_diagnostic_result)
        ),
        timescale_structural_de_hop_structural_max=(
            _maximum_timescale_de_hop_structural(timescale_diagnostic_result)
        ),
        timescale_structural_atmosphere_ratio_max=(
            _maximum_timescale_atmosphere_ratio(timescale_diagnostic_result)
        ),
        timescale_structural_size_void_ratio_max=(
            _maximum_timescale_size_void_ratio(timescale_diagnostic_result)
        ),
        timescale_structural_capture_fraction_max=(
            _maximum_timescale_capture_fraction(timescale_diagnostic_result)
        ),
        timescale_event_family_attributions=(
            timescale_diagnostic_result.markov_event_family_attributions
        ),
        free_li_obstruction_factor=(
            _free_li_obstruction_primitive(baseline_result).obstruction_factor
        ),
        free_li_translation_diffusion_scale=(
            _free_li_obstruction_primitive(baseline_result).diffusion_scale
        ),
        free_anion_obstruction_factor_max=(
            _maximum_free_anion_obstruction_factor_from_primitives(baseline_result)
        ),
        free_anion_translation_diffusion_scale_min=(
            _minimum_free_anion_translation_scale_from_primitives(baseline_result)
        ),
        obstruction_steric_driver=_maximum_obstruction_driver(
            baseline_result,
            "steric_driver",
        ),
        obstruction_compact_anion_driver=(
            _maximum_obstruction_driver(
                baseline_result,
                "compact_anion_driver",
            )
        ),
        obstruction_carbonate_driver=_maximum_obstruction_driver(
            baseline_result,
            "carbonate_driver",
        ),
        obstruction_high_salt_driver=_maximum_obstruction_driver(
            baseline_result,
            "high_salt_driver",
        ),
        obstruction_low_donor_driver=_maximum_obstruction_driver(
            baseline_result,
            "low_donor_driver",
        ),
        free_li_translation_marginal_net_mS_cm=(
            _event_family_marginal_net_sigma_mS_cm(
                baseline_result,
                EVENT_FAMILY_ORDINARY_FREE_LI_TRANSLATION,
            )
        ),
        free_anion_translation_marginal_net_mS_cm=(
            _event_family_marginal_net_sigma_mS_cm(
                baseline_result,
                EVENT_FAMILY_ORDINARY_FREE_ANION_TRANSLATION,
            )
        ),
        sigma_free_li_mS_cm=_state_conductivity_mS_cm(
            baseline_result,
            temperature_K,
            ("FREE_LI",),
        ),
        sigma_free_anion_mS_cm=_state_conductivity_mS_cm(
            baseline_result,
            temperature_K,
            ("FREE_ANION",),
        ),
        sigma_ssip_mS_cm=_state_conductivity_mS_cm(
            baseline_result,
            temperature_K,
            ("SSIP",),
        ),
        sigma_cip_mS_cm=_state_conductivity_mS_cm(
            baseline_result,
            temperature_K,
            ("CIP",),
        ),
        sigma_aggregates_mS_cm=_state_conductivity_mS_cm(
            baseline_result,
            temperature_K,
            AGGREGATE_STATE_KINDS,
        ),
        local_resistance_trace_kg_s=_local_resistance_trace_kg_s(baseline_result),
        binding_resistance_trace_kg_s=_binding_resistance_trace_kg_s(baseline_result),
        atmosphere_resistance_trace_kg_s=_atmosphere_resistance_trace_kg_s(baseline_result),
    )


def _prediction_for_mode(
    predictions: Sequence[AnalyticMoriAblationPrediction],
    ablation_mode: str,
) -> AnalyticMoriAblationPrediction:
    for prediction in predictions:
        if prediction.ablation_mode == ablation_mode:
            return prediction
    raise ValueError(f"missing ablation prediction for mode {ablation_mode}")


def _free_translation_inverse_target(
    row: AnalyticMoriPropertyDbRow,
    gate_sigma_mS_cm: float,
) -> FreeTranslationInverseTarget:
    free_li_marginal_mS_cm = _positive_marginal(
        row.free_li_translation_marginal_net_mS_cm,
        row.row_id,
        "free_li_translation_marginal_net_mS_cm",
    )
    free_anion_marginal_mS_cm = _positive_marginal(
        row.free_anion_translation_marginal_net_mS_cm,
        row.row_id,
        "free_anion_translation_marginal_net_mS_cm",
    )
    free_translation_marginal_mS_cm = (
        free_li_marginal_mS_cm + free_anion_marginal_mS_cm
    )
    if free_translation_marginal_mS_cm <= 0.0:
        raise ValueError(f"row {row.row_id} free translation marginal must be positive")
    residual_mS_cm = row.residual_mS_cm
    required_common_free_scale_to_empirical = (
        1.0 - residual_mS_cm / free_translation_marginal_mS_cm
    )
    required_common_free_scale_to_gate = (
        1.0
        - (
            row.analytic_mori_sigma_mS_cm
            - gate_sigma_mS_cm
        )
        / free_translation_marginal_mS_cm
    )
    required_li_scale_if_anion_fixed = (
        1.0 - residual_mS_cm / free_li_marginal_mS_cm
    )
    required_anion_scale_if_li_fixed = (
        1.0 - residual_mS_cm / free_anion_marginal_mS_cm
    )
    obstruction_driver = (
        row.obstruction_steric_driver
        * row.obstruction_compact_anion_driver
        * row.obstruction_carbonate_driver
        * row.obstruction_high_salt_driver
        * row.obstruction_low_donor_driver
    )
    return FreeTranslationInverseTarget(
        row_id=row.row_id,
        empirical_sigma_mS_cm=row.empirical_sigma_mS_cm,
        predicted_sigma_mS_cm=row.analytic_mori_sigma_mS_cm,
        residual_mS_cm=row.residual_mS_cm,
        free_li_marginal_mS_cm=free_li_marginal_mS_cm,
        free_anion_marginal_mS_cm=free_anion_marginal_mS_cm,
        free_translation_marginal_mS_cm=free_translation_marginal_mS_cm,
        required_common_free_scale_to_empirical=float(
            required_common_free_scale_to_empirical
        ),
        required_common_free_scale_to_empirical_clipped=_clipped_unit_interval(
            required_common_free_scale_to_empirical
        ),
        required_common_free_scale_to_gate=float(required_common_free_scale_to_gate),
        required_common_free_scale_to_gate_clipped=_clipped_unit_interval(
            required_common_free_scale_to_gate
        ),
        required_li_scale_if_anion_fixed=float(required_li_scale_if_anion_fixed),
        required_li_scale_if_anion_fixed_clipped=_clipped_unit_interval(
            required_li_scale_if_anion_fixed
        ),
        required_anion_scale_if_li_fixed=float(required_anion_scale_if_li_fixed),
        required_anion_scale_if_li_fixed_clipped=_clipped_unit_interval(
            required_anion_scale_if_li_fixed
        ),
        salt_family=row.salt_family,
        solvent_family=row.solvent_family,
        salt_molarity_M=row.salt_molarity_M,
        steric_volume_fraction=row.steric_volume_fraction,
        obstruction_driver=float(obstruction_driver),
        structural_interval_covers_empirical=row.certificate_covers_empirical,
        prediction_status=row.prediction_status,
    )


def _positive_marginal(
    value: float,
    row_id: int,
    field_name: str,
) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"row {row_id} {field_name} must be positive and finite")
    return parsed_value


def _clipped_unit_interval(value: float) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"clipped inverse scale source must be finite, got {value}")
    return float(min(1.0, max(0.0, parsed_value)))


def _free_translation_inverse_neighborhood(
    target: FreeTranslationInverseTarget,
    targets: Sequence[FreeTranslationInverseTarget],
) -> FreeTranslationInverseNeighborhood:
    neighbors = tuple(
        candidate
        for candidate in targets
        if candidate.row_id != target.row_id
        and _is_free_translation_neighbor(target, candidate)
    )
    if not neighbors:
        return FreeTranslationInverseNeighborhood(
            row_id=target.row_id,
            neighbor_count=0,
            median_neighbor_required_scale=1.0,
            min_neighbor_required_scale=1.0,
            median_neighbor_residual_mS_cm=0.0,
            same_salt_family_count=0,
            same_solvent_family_count=0,
            has_systematic_cluster=False,
        )
    neighbor_required_scales = np.asarray(
        [
            neighbor.required_common_free_scale_to_empirical
            for neighbor in neighbors
        ],
        dtype=float,
    )
    neighbor_residuals = np.asarray(
        [neighbor.residual_mS_cm for neighbor in neighbors],
        dtype=float,
    )
    median_neighbor_required_scale = float(np.median(neighbor_required_scales))
    return FreeTranslationInverseNeighborhood(
        row_id=target.row_id,
        neighbor_count=len(neighbors),
        median_neighbor_required_scale=median_neighbor_required_scale,
        min_neighbor_required_scale=float(np.min(neighbor_required_scales)),
        median_neighbor_residual_mS_cm=float(np.median(neighbor_residuals)),
        same_salt_family_count=sum(
            1 for neighbor in neighbors if neighbor.salt_family == target.salt_family
        ),
        same_solvent_family_count=sum(
            1 for neighbor in neighbors if neighbor.solvent_family == target.solvent_family
        ),
        has_systematic_cluster=(
            len(neighbors) >= FREE_TRANSLATION_CLUSTER_MIN_COUNT
            and median_neighbor_required_scale
            <= FREE_TRANSLATION_CLUSTER_SCALE_THRESHOLD
        ),
    )


def _is_free_translation_neighbor(
    target: FreeTranslationInverseTarget,
    candidate: FreeTranslationInverseTarget,
) -> bool:
    if (
        target.salt_family != candidate.salt_family
        and target.solvent_family != candidate.solvent_family
    ):
        return False
    return (
        abs(target.salt_molarity_M - candidate.salt_molarity_M)
        <= FREE_TRANSLATION_NEIGHBOR_MOLARITY_WINDOW_M
        and abs(
            target.steric_volume_fraction
            - candidate.steric_volume_fraction
        )
        <= FREE_TRANSLATION_NEIGHBOR_STERIC_WINDOW
        and abs(target.obstruction_driver - candidate.obstruction_driver)
        <= FREE_TRANSLATION_NEIGHBOR_DRIVER_WINDOW
    )


def _free_translation_inverse_status(
    target: FreeTranslationInverseTarget,
    neighborhood: FreeTranslationInverseNeighborhood,
) -> AnalyticMoriPredictionStatus:
    if (
        target.required_common_free_scale_to_empirical
        < FREE_TRANSLATION_REQUIRED_SCALE_EXTREME_THRESHOLD
        and not target.structural_interval_covers_empirical
        and not neighborhood.has_systematic_cluster
    ):
        return AnalyticMoriPredictionStatus.FREE_TRANSLATION_PRIMITIVE_FAILURE
    if (
        target.required_common_free_scale_to_empirical
        < FREE_TRANSLATION_REQUIRED_SCALE_SINGLETON_THRESHOLD
        and not neighborhood.has_systematic_cluster
    ):
        return AnalyticMoriPredictionStatus.FREE_TRANSLATION_SINGLETON_RESIDUAL
    if neighborhood.has_systematic_cluster:
        return AnalyticMoriPredictionStatus.SYSTEMATIC_FREE_TRANSLATION_SUPPRESSION_CLUSTER
    return AnalyticMoriPredictionStatus.DESCRIPTOR_EQUATION_PREDICTION


def _rows_with_free_translation_inverse_status(
    rows: Sequence[AnalyticMoriPropertyDbRow],
) -> tuple[AnalyticMoriPropertyDbRow, ...]:
    if not rows:
        return tuple()
    inverse_audit = compute_free_translation_inverse_target_audit(
        rows,
        FREE_TRANSLATION_INVERSE_STATUS_GATE_SIGMA_MS_CM,
    )
    inverse_status_by_row_id = {
        target.row_id: AnalyticMoriPredictionStatus(target.prediction_status)
        for target in inverse_audit.targets
    }
    return tuple(
        replace(
            row,
            prediction_status=_combined_prediction_status(
                AnalyticMoriPredictionStatus(row.prediction_status),
                inverse_status_by_row_id[row.row_id],
            ).value,
        )
        for row in rows
    )


def _combined_prediction_status(
    baseline_prediction_status: AnalyticMoriPredictionStatus,
    free_translation_inverse_status: AnalyticMoriPredictionStatus,
) -> AnalyticMoriPredictionStatus:
    if (
        free_translation_inverse_status
        is AnalyticMoriPredictionStatus.FREE_TRANSLATION_PRIMITIVE_FAILURE
    ):
        return free_translation_inverse_status
    if (
        baseline_prediction_status
        is AnalyticMoriPredictionStatus.DESCRIPTOR_EQUATION_PREDICTION
    ):
        return free_translation_inverse_status
    return baseline_prediction_status


def _is_descriptor_complete_prediction_status(prediction_status: str) -> bool:
    parsed_prediction_status = AnalyticMoriPredictionStatus(prediction_status)
    return parsed_prediction_status not in EQUATION_DOMAIN_BLOCKING_STATUSES


def _is_equation_domain_violation_status(prediction_status: str) -> bool:
    return (
        AnalyticMoriPredictionStatus(prediction_status)
        in EQUATION_DOMAIN_BLOCKING_STATUSES
    )


def _free_translation_group_metrics(
    targets: Sequence[FreeTranslationInverseTarget],
    group_kind: str,
    field_name: str,
) -> tuple[FreeTranslationInverseGroupMetric, ...]:
    grouped_targets: defaultdict[str, list[FreeTranslationInverseTarget]] = defaultdict(list)
    for target in targets:
        grouped_targets[str(getattr(target, field_name))].append(target)
    return tuple(
        _free_translation_group_metric(group_kind, group_label, grouped_targets[group_label])
        for group_label in sorted(grouped_targets)
    )


def _free_translation_bin_group_metrics(
    targets: Sequence[FreeTranslationInverseTarget],
    group_kind: str,
    field_name: str,
) -> tuple[FreeTranslationInverseGroupMetric, ...]:
    grouped_targets: defaultdict[str, list[FreeTranslationInverseTarget]] = defaultdict(list)
    for target in targets:
        grouped_targets[
            _unit_interval_bin_label(float(getattr(target, field_name)))
        ].append(target)
    return tuple(
        _free_translation_group_metric(group_kind, group_label, grouped_targets[group_label])
        for group_label in sorted(grouped_targets)
    )


def _unit_interval_bin_label(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"unit-interval bin value must be finite, got {value}")
    if value < INVERSE_TARGET_LOW_UNIT_BIN_MAX:
        return "low"
    if value < INVERSE_TARGET_HIGH_UNIT_BIN_MIN:
        return "mid"
    return "high"


def _free_translation_group_metric(
    group_kind: str,
    group_label: str,
    targets: Sequence[FreeTranslationInverseTarget],
) -> FreeTranslationInverseGroupMetric:
    if not targets:
        raise ValueError(f"{group_kind} group {group_label} has no targets")
    required_scales = np.asarray(
        [
            target.required_common_free_scale_to_empirical
            for target in targets
        ],
        dtype=float,
    )
    clipped_required_scales = np.asarray(
        [
            target.required_common_free_scale_to_empirical_clipped
            for target in targets
        ],
        dtype=float,
    )
    residuals = np.asarray(
        [target.residual_mS_cm for target in targets],
        dtype=float,
    )
    lower_quartile = float(np.quantile(required_scales, LOWER_QUARTILE_PROBABILITY))
    upper_quartile = float(np.quantile(required_scales, UPPER_QUARTILE_PROBABILITY))
    return FreeTranslationInverseGroupMetric(
        group_kind=group_kind,
        group_label=group_label,
        count=len(targets),
        median_required_common_free_scale=float(np.median(required_scales)),
        iqr_required_common_free_scale=float(upper_quartile - lower_quartile),
        median_required_common_free_scale_clipped=float(
            np.median(clipped_required_scales)
        ),
        mean_signed_error_mS_cm=float(np.mean(residuals)),
    )


def _obstruction_reachability_metric(
    records: Sequence[_ObstructionReachabilityRecord],
    failures: Sequence[AnalyticMoriPropertyDbFailure],
    free_li_translation_scale: float,
    compact_anion_translation_scale: float,
    tracked_row52_id: int,
    tracked_row53_id: int,
) -> AnalyticMoriObstructionReachabilityMetric:
    empirical_values = np.asarray(
        [record.empirical_sigma_mS_cm for record in records],
        dtype=float,
    )
    predicted_values = np.asarray(
        [record.predicted_sigma_mS_cm for record in records],
        dtype=float,
    )
    residuals = predicted_values - empirical_values
    return AnalyticMoriObstructionReachabilityMetric(
        free_li_translation_scale=float(free_li_translation_scale),
        compact_anion_translation_scale=float(compact_anion_translation_scale),
        evaluated_rows=len(records),
        failed_rows=len(failures),
        mae_mS_cm=float(np.mean(np.abs(residuals))),
        bias_mS_cm=float(np.mean(residuals)),
        pearson_r=float(np.corrcoef(empirical_values, predicted_values)[0, 1]),
        row52_sigma_mS_cm=_tracked_row_sigma_mS_cm(records, tracked_row52_id),
        row53_sigma_mS_cm=_tracked_row_sigma_mS_cm(records, tracked_row53_id),
        lipf6_mae_mS_cm=_reachability_family_mae_mS_cm(records, "LiPF6"),
        litfsi_mae_mS_cm=_reachability_family_mae_mS_cm(records, "LiTFSI"),
        lifsi_mae_mS_cm=_reachability_family_mae_mS_cm(records, "LiFSI"),
    )


def _tracked_row_sigma_mS_cm(
    records: Sequence[_ObstructionReachabilityRecord],
    row_id: int,
) -> float:
    for record in records:
        if record.row_id == row_id:
            return float(record.predicted_sigma_mS_cm)
    raise ValueError(f"obstruction reachability missing tracked row {row_id}")


def _reachability_family_mae_mS_cm(
    records: Sequence[_ObstructionReachabilityRecord],
    family_name: str,
) -> float:
    residuals = [
        record.residual_mS_cm
        for record in records
        if record.salt_family == family_name
    ]
    if not residuals:
        raise ValueError(f"obstruction reachability missing family {family_name}")
    return float(np.mean(np.abs(np.asarray(residuals, dtype=float))))


def _reachability_translation_scale(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0 or parsed_value > 1.0:
        raise ValueError(f"{context} entries must be finite and in (0, 1]")
    return parsed_value


def _state_lithium_fraction(
    analytic_result: AnalyticMoriPrimitiveResult,
    state_kinds: tuple[str, ...],
) -> float:
    total_lithium_mol_m3 = (
        analytic_result.mass_balance.total_concentrations_M[0]
        * MOLARITY_TO_MOL_M3
    )
    if total_lithium_mol_m3 <= 0.0:
        raise ValueError("total lithium concentration must be positive")
    occupied_lithium_mol_m3 = math.fsum(
        _lithium_count_for_state_kind(state.state_kind) * state.concentration_mol_m3
        for state in analytic_result.transport_states
        if state.state_kind in state_kinds
    )
    return float(occupied_lithium_mol_m3 / total_lithium_mol_m3)


def _free_anion_fraction(analytic_result: AnalyticMoriPrimitiveResult) -> float:
    total_anion_mol_m3 = (
        math.fsum(analytic_result.mass_balance.total_concentrations_M[1:])
        * MOLARITY_TO_MOL_M3
    )
    if total_anion_mol_m3 <= 0.0:
        raise ValueError("total anion concentration must be positive")
    free_anion_mol_m3 = math.fsum(
        state.concentration_mol_m3
        for state in analytic_result.transport_states
        if state.state_kind == "FREE_ANION"
    )
    return float(free_anion_mol_m3 / total_anion_mol_m3)


def _markov_corrector_over_direct(analytic_result: AnalyticMoriPrimitiveResult) -> float:
    direct_sigma_mS_cm = analytic_result.markov_additive_result.direct_sigma_mS_cm
    if direct_sigma_mS_cm <= 0.0:
        return 0.0
    return float(
        analytic_result.markov_additive_result.corrector_sigma_mS_cm
        / direct_sigma_mS_cm
    )


def _minimum_carrier_relaxation_form_factor(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.atmosphere_carrier_relaxation_form_factors:
        return 1.0
    return float(
        min(
            form_factor.form_factor_squared
            for form_factor in analytic_result.atmosphere_carrier_relaxation_form_factors
        )
    )


def _maximum_carrier_charge_cloud_radius_A(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.atmosphere_carrier_relaxation_form_factors:
        return 0.0
    return float(
        max(
            form_factor.charge_cloud_radius_A
            for form_factor in analytic_result.atmosphere_carrier_relaxation_form_factors
        )
    )


def _maximum_carrier_caged_fraction(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.carrier_cage_primitives:
        return 0.0
    return float(
        max(
            cage_primitive.caged_fraction
            for cage_primitive in analytic_result.carrier_cage_primitives
        )
    )


def _minimum_carrier_caged_diffusion_scale(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.carrier_cage_primitives:
        return 1.0
    return float(
        min(
            cage_primitive.caged_diffusion_scale
            for cage_primitive in analytic_result.carrier_cage_primitives
        )
    )


def _maximum_carrier_cage_exchange_rate_s_inv(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.carrier_cage_primitives:
        return 0.0
    return float(
        max(
            cage_primitive.exchange_rate_s_inv
            for cage_primitive in analytic_result.carrier_cage_primitives
        )
    )


def _maximum_selective_cage_driver(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.selective_carrier_cage_primitives:
        return 0.0
    return float(
        max(
            selective_cage_primitive.selective_cage_driver
            for selective_cage_primitive in analytic_result.selective_carrier_cage_primitives
        )
    )


def _maximum_selective_caged_fraction(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.selective_carrier_cage_primitives:
        return 0.0
    return float(
        max(
            selective_cage_primitive.caged_fraction
            for selective_cage_primitive in analytic_result.selective_carrier_cage_primitives
        )
    )


def _minimum_selective_caged_diffusion_scale(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.selective_carrier_cage_primitives:
        return 1.0
    return float(
        min(
            selective_cage_primitive.caged_diffusion_scale
            for selective_cage_primitive in analytic_result.selective_carrier_cage_primitives
        )
    )


def _maximum_timescale_structural_cage_fraction(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.timescale_structural_memory_primitives:
        return 0.0
    return float(
        max(
            primitive.structural_cage_fraction
            for primitive in analytic_result.timescale_structural_memory_primitives
        )
    )


def _maximum_timescale_de_hop_structural(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.timescale_structural_memory_primitives:
        return 0.0
    return float(
        max(
            primitive.de_hop_structural
            for primitive in analytic_result.timescale_structural_memory_primitives
        )
    )


def _maximum_timescale_atmosphere_ratio(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.timescale_structural_memory_primitives:
        return 0.0
    return float(
        max(
            primitive.atmosphere_structural_ratio
            for primitive in analytic_result.timescale_structural_memory_primitives
        )
    )


def _maximum_timescale_size_void_ratio(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.timescale_structural_memory_primitives:
        return 0.0
    return float(
        max(
            primitive.size_void_ratio
            for primitive in analytic_result.timescale_structural_memory_primitives
        )
    )


def _maximum_timescale_capture_fraction(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    if not analytic_result.timescale_structural_memory_primitives:
        return 0.0
    return float(
        max(
            primitive.atmosphere_capture_fraction
            for primitive in analytic_result.timescale_structural_memory_primitives
        )
    )


def _lithium_backjump_cage_primitive(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> AnalyticBackjumpCagePrimitive | None:
    for backjump_primitive in analytic_result.backjump_cage_primitives:
        if backjump_primitive.carrier_kind == "FREE_LI":
            return backjump_primitive
    return None


def _backjump_cage_driver(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    backjump_primitive = _lithium_backjump_cage_primitive(analytic_result)
    if backjump_primitive is None:
        return 0.0
    return float(backjump_primitive.cage_driver)


def _backjump_cage_occupancy_fraction(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    backjump_primitive = _lithium_backjump_cage_primitive(analytic_result)
    if backjump_primitive is None:
        return 0.0
    return float(backjump_primitive.cage_occupancy_fraction)


def _backjump_attempt_fraction(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    backjump_primitive = _lithium_backjump_cage_primitive(analytic_result)
    if backjump_primitive is None:
        return 0.0
    return float(backjump_primitive.attempt_fraction)


def _backjump_probability(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    backjump_primitive = _lithium_backjump_cage_primitive(analytic_result)
    if backjump_primitive is None:
        return 0.0
    return float(backjump_primitive.backjump_probability)


def _backjump_exit_rate_s_inv(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    backjump_primitive = _lithium_backjump_cage_primitive(analytic_result)
    if backjump_primitive is None:
        return 0.0
    return float(backjump_primitive.exit_rate_s_inv)


def _backjump_length_A(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    backjump_primitive = _lithium_backjump_cage_primitive(analytic_result)
    if backjump_primitive is None:
        return 0.0
    return float(backjump_primitive.jump_length_m * ANGSTROM_PER_M)


def _backjump_direct_sigma_mS_cm(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    backjump_primitive = _lithium_backjump_cage_primitive(analytic_result)
    if backjump_primitive is None:
        return 0.0
    return float(backjump_primitive.direct_sigma_mS_cm)


def _backjump_ordinary_translation_fraction(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    backjump_primitive = _lithium_backjump_cage_primitive(analytic_result)
    if backjump_primitive is None:
        return 1.0
    return float(backjump_primitive.ordinary_translation_fraction)


def _free_li_obstruction_primitive(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> AnalyticFreeCarrierObstructionPrimitive:
    for obstruction_primitive in analytic_result.free_carrier_obstruction_primitives:
        if obstruction_primitive.carrier_kind == "lithium":
            return obstruction_primitive
    raise ValueError("missing free lithium obstruction primitive")


def _free_anion_obstruction_primitives(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> tuple[AnalyticFreeCarrierObstructionPrimitive, ...]:
    anion_primitives = tuple(
        obstruction_primitive
        for obstruction_primitive in analytic_result.free_carrier_obstruction_primitives
        if obstruction_primitive.carrier_kind == "anion"
    )
    if not anion_primitives:
        raise ValueError("missing free anion obstruction primitives")
    return anion_primitives


def _maximum_free_anion_obstruction_factor_from_primitives(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    return float(
        max(
            obstruction_primitive.obstruction_factor
            for obstruction_primitive in _free_anion_obstruction_primitives(
                analytic_result
            )
        )
    )


def _minimum_free_anion_translation_scale_from_primitives(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    return float(
        min(
            obstruction_primitive.diffusion_scale
            for obstruction_primitive in _free_anion_obstruction_primitives(
                analytic_result
            )
        )
    )


def _maximum_obstruction_driver(
    analytic_result: AnalyticMoriPrimitiveResult,
    field_name: str,
) -> float:
    if not analytic_result.free_carrier_obstruction_primitives:
        raise ValueError("missing free-carrier obstruction primitives")
    return float(
        max(
            float(getattr(obstruction_primitive, field_name))
            for obstruction_primitive in analytic_result.free_carrier_obstruction_primitives
        )
    )


def _event_family_marginal_net_sigma_mS_cm(
    analytic_result: AnalyticMoriPrimitiveResult,
    family_label: str,
) -> float:
    for attribution in analytic_result.markov_event_family_attributions:
        if attribution.family_label == family_label:
            return float(attribution.marginal_net_sigma_mS_cm)
    raise ValueError(f"missing event-family attribution for {family_label}")


def _prediction_status(
    over_association_warning: bool,
    large_cancellation_warning: bool,
    dielectric_collapse_warning: bool,
) -> str:
    if over_association_warning or large_cancellation_warning or dielectric_collapse_warning:
        return AnalyticMoriPredictionStatus.PRIMITIVE_WARNING.value
    return AnalyticMoriPredictionStatus.DESCRIPTOR_EQUATION_PREDICTION.value


def _structural_certificate_from_perturbation_scenarios(
    analytic_recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    baseline_result: AnalyticMoriPrimitiveResult,
) -> _StructuralCertificateEvaluation:
    scenario_sigmas: list[tuple[str, float]] = [
        (ANALYTIC_MORI_ABLATION_BASELINE, baseline_result.sigma_mS_cm)
    ]
    for ablation_mode in STRUCTURAL_CERTIFICATE_ABLATION_MODES:
        scenario_result = evaluate_analytic_mori_ablation_conductivity(
            analytic_recipe,
            uncertainty_budget,
            species_catalog,
            ablation_mode,
        )
        scenario_sigmas.append((ablation_mode, scenario_result.sigma_mS_cm))
    forced_free_li_translation_scale = _reachability_translation_scale(
        uncertainty_budget.forced_free_li_translation_scale_interval[0],
        "forced_free_li_translation_scale_interval.lower",
    )
    forced_compact_anion_translation_scale = _reachability_translation_scale(
        uncertainty_budget.forced_compact_anion_translation_scale_interval[0],
        "forced_compact_anion_translation_scale_interval.lower",
    )
    for scenario_label, free_li_translation_scale, compact_anion_translation_scale in (
        (
            "forced_free_li_translation_scale_lower",
            forced_free_li_translation_scale,
            1.0,
        ),
        (
            "forced_compact_anion_translation_scale_lower",
            1.0,
            forced_compact_anion_translation_scale,
        ),
        (
            "forced_free_li_plus_compact_anion_translation_scale_lower",
            forced_free_li_translation_scale,
            forced_compact_anion_translation_scale,
        ),
    ):
        scenario_result = (
            evaluate_analytic_mori_forced_free_carrier_obstruction_conductivity(
                analytic_recipe,
                uncertainty_budget,
                species_catalog,
                free_li_translation_scale,
                compact_anion_translation_scale,
            )
        )
        scenario_sigmas.append((scenario_label, scenario_result.sigma_mS_cm))
    sigma_values = tuple(sigma_mS_cm for _, sigma_mS_cm in scenario_sigmas)
    sigma_min_mS_cm = min(sigma_values)
    sigma_max_mS_cm = max(sigma_values)
    baseline_sigma_mS_cm = baseline_result.sigma_mS_cm
    half_width_mS_cm = max(
        abs(baseline_sigma_mS_cm - sigma_min_mS_cm),
        abs(sigma_max_mS_cm - baseline_sigma_mS_cm),
    )
    dominant_uncertainty_head = max(
        scenario_sigmas,
        key=lambda scenario: abs(scenario[1] - baseline_sigma_mS_cm),
    )[0]
    return _StructuralCertificateEvaluation(
        sigma_min_mS_cm=float(sigma_min_mS_cm),
        sigma_max_mS_cm=float(sigma_max_mS_cm),
        half_width_mS_cm=float(half_width_mS_cm),
        dominant_uncertainty_head=dominant_uncertainty_head,
        certified_0p25_mS_cm=bool(
            half_width_mS_cm <= uncertainty_budget.certificate_threshold_mS_cm
        ),
    )


def _over_association_warning(
    free_li_fraction: float,
    free_anion_fraction: float,
    neutral_aggregate_fraction: float,
) -> bool:
    return (
        free_li_fraction < OVER_ASSOCIATION_FREE_FRACTION_WARNING_THRESHOLD
        or free_anion_fraction < OVER_ASSOCIATION_FREE_FRACTION_WARNING_THRESHOLD
        or neutral_aggregate_fraction
        > OVER_ASSOCIATION_NEUTRAL_AGGREGATE_WARNING_THRESHOLD
    )


def _dielectric_collapse_warning(analytic_result: AnalyticMoriPrimitiveResult) -> bool:
    epsilon_mixture = analytic_result.dielectric_heads.epsilon_mixture
    if epsilon_mixture <= 0.0:
        raise ValueError("epsilon_mixture must be positive")
    return (
        analytic_result.dielectric_heads.epsilon_atmosphere / epsilon_mixture
        < DIELECTRIC_COLLAPSE_RATIO_WARNING_THRESHOLD
    )


def _lithium_count_for_state_kind(state_kind: str) -> int:
    if state_kind in ("FREE_LI", "SSIP", "CIP", "LIA2_MINUS"):
        return 1
    if state_kind in ("LI2A_PLUS", "LI2A2_NEUTRAL"):
        return 2
    if state_kind == "FREE_ANION":
        return 0
    raise ValueError(f"unsupported state_kind {state_kind}")


def _state_conductivity_mS_cm(
    analytic_result: AnalyticMoriPrimitiveResult,
    temperature_K: float,
    state_kinds: tuple[str, ...],
) -> float:
    conductivity_S_m = math.fsum(
        state.concentration_mol_m3 * state.charge_diffusivity_m2_s
        for state in analytic_result.transport_states
        if state.state_kind in state_kinds
    ) * F * F / (R * temperature_K)
    return float(conductivity_S_m * S_M_TO_MS_CM)


def _local_resistance_trace_kg_s(analytic_result: AnalyticMoriPrimitiveResult) -> float:
    weighted_trace = math.fsum(
        state.concentration_mol_m3 * float(np.trace(state.local_resistance_matrix_kg_s))
        for state in analytic_result.transport_states
    )
    return float(weighted_trace / _transport_concentration_sum_mol_m3(analytic_result))


def _binding_resistance_trace_kg_s(analytic_result: AnalyticMoriPrimitiveResult) -> float:
    weighted_trace = math.fsum(
        state.concentration_mol_m3 * float(np.trace(state.binding_resistance_matrix_kg_s))
        for state in analytic_result.transport_states
    )
    return float(weighted_trace / _transport_concentration_sum_mol_m3(analytic_result))


def _atmosphere_resistance_trace_kg_s(analytic_result: AnalyticMoriPrimitiveResult) -> float:
    weighted_trace = math.fsum(
        state.concentration_mol_m3 * float(np.trace(state.atmosphere_resistance_matrix_kg_s))
        for state in analytic_result.transport_states
    )
    return float(weighted_trace / _transport_concentration_sum_mol_m3(analytic_result))


def _transport_concentration_sum_mol_m3(
    analytic_result: AnalyticMoriPrimitiveResult,
) -> float:
    concentration_sum_mol_m3 = math.fsum(
        state.concentration_mol_m3 for state in analytic_result.transport_states
    )
    if concentration_sum_mol_m3 <= 0.0:
        raise ValueError("transport concentration sum must be positive")
    return float(concentration_sum_mol_m3)


def _dataset_metrics(rows: Sequence[AnalyticMoriPropertyDbRow]) -> dict[str, float]:
    if len(rows) < 2:
        raise ValueError("analytic Mori property-DB audit requires at least two evaluated rows")

    empirical_values = np.asarray(
        [row.empirical_sigma_mS_cm for row in rows],
        dtype=float,
    )
    predicted_values = np.asarray(
        [row.analytic_mori_sigma_mS_cm for row in rows],
        dtype=float,
    )
    residuals = predicted_values - empirical_values
    total_sum_squares = float(np.sum((empirical_values - float(np.mean(empirical_values))) ** 2))
    if total_sum_squares <= 0.0:
        raise ValueError("analytic Mori property-DB audit empirical labels have zero variance")
    residual_sum_squares = float(np.sum(residuals * residuals))
    pearson_r = float(np.corrcoef(empirical_values, predicted_values)[0, 1])
    return {
        "mae_mS_cm": float(np.mean(np.abs(residuals))),
        "rmse_mS_cm": float(math.sqrt(float(np.mean(residuals * residuals)))),
        "bias_mS_cm": float(np.mean(residuals)),
        "mape_percent": float(np.mean(np.abs(residuals / empirical_values)) * PERCENT),
        "r2": float(1.0 - residual_sum_squares / total_sum_squares),
        "pearson_r": pearson_r,
    }


def _certificate_coverage_fraction(rows: Sequence[AnalyticMoriPropertyDbRow]) -> float:
    if not rows:
        return 0.0
    covered_count = sum(1 for row in rows if row.certificate_covers_empirical)
    return float(covered_count / len(rows))


def _maximum_row_value(
    rows: Sequence[AnalyticMoriPropertyDbRow],
    field_name: str,
) -> float:
    if not rows:
        return 0.0
    return float(max(float(getattr(row, field_name)) for row in rows))


def _salt_family_metrics(
    rows: Sequence[AnalyticMoriPropertyDbRow],
) -> tuple[AnalyticMoriFamilyMetrics, ...]:
    residuals_by_family: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        residuals_by_family[row.salt_family].append(row.residual_mS_cm)
    metrics: list[AnalyticMoriFamilyMetrics] = []
    for family_name in sorted(residuals_by_family):
        residuals = np.asarray(residuals_by_family[family_name], dtype=float)
        metrics.append(
            AnalyticMoriFamilyMetrics(
                family_name=family_name,
                count=len(residuals),
                bias_mS_cm=float(np.mean(residuals)),
                mae_mS_cm=float(np.mean(np.abs(residuals))),
                rmse_mS_cm=float(math.sqrt(float(np.mean(residuals * residuals)))),
            )
        )
    return tuple(metrics)


def _family_label(species_loadings: Mapping[str, float]) -> str:
    if not species_loadings:
        return "none"
    return "+".join(species_name for species_name in species_loadings)
