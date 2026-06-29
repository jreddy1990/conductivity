"""Run the analytic descriptor-to-Mori generator against empirical conductivity labels."""

from __future__ import annotations

from constants import T_REF_K
from conductivity.analytic_mori_primitive_generator import (
    AnalyticMoriSpeciesCatalog,
    StructuralPrimitiveUncertaintyBudget,
)
from conductivity.analytic_mori_property_db_audit import (
    AnalyticMoriAblationAuditResult,
    FreeTranslationInverseAuditResult,
    FreeTranslationInverseGroupMetric,
    FreeTranslationInverseNeighborhood,
    FreeTranslationInverseTarget,
    PrimitiveSensitivityAuditResult,
    PrimitiveSensitivityGroup,
    PrimitiveSensitivityRow,
    AnalyticMoriObstructionReachabilityMetric,
    AnalyticMoriWorstRowPrimitiveDecomposition,
    AnalyticMoriPropertyDbAuditResult,
    AnalyticMoriPropertyDbRow,
    audit_analytic_mori_ablation_suite_against_property_db,
    audit_analytic_mori_conductivity_against_property_db,
    audit_analytic_mori_obstruction_reachability_against_property_db,
    audit_analytic_mori_primitive_sensitivity_against_property_db,
    compute_free_translation_inverse_target_audit,
)
from conductivity.run_finite_markov_dataset_audit import TOP_REPORTED_ROWS
from data.electrolyte_property_db import DATA
from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS


CERTIFICATE_THRESHOLD_MILLI_SIEMENS_PER_CM = float("0.25")  # Explicit user-declared audit threshold.
STRUCTURAL_LOGK_INTERVAL = (-1.0, 1.0)  # Explicit structural audit envelope: one natural-log unit around association constants.
STRUCTURAL_UNIT_SCALE_INTERVAL = (0.0, 1.0)  # Explicit structural audit envelope: disabled-to-baseline scale.
STRUCTURAL_HALF_TO_ONE_AND_HALF_SCALE_INTERVAL = (0.5, 1.5)  # Explicit structural audit envelope: factor-of-three centered scale span.
STRUCTURAL_RELAXATION_SCALE_INTERVAL = (0.25, 1.25)  # Explicit structural audit envelope: relaxation head can be strongly weakened or mildly strengthened.
STRUCTURAL_LITHIUM_CHARGE_CLOUD_RADIUS_INTERVAL_A = (1.0, 4.5)  # Explicit structural audit envelope: compact Li radius through registry solvated Li radius.
STRUCTURAL_ANION_CHARGE_CLOUD_RADIUS_INTERVAL_A = (0.1, 8.0)  # Explicit structural audit envelope: point-like compact anion through diffuse imide-like anion.
STRUCTURAL_CONVERSION_RATE_SCALE_INTERVAL = (0.25, 2.0)  # Explicit structural audit envelope: quarter-speed through double-speed conversion rates.
STRUCTURAL_FORCED_FREE_LI_TRANSLATION_SCALE_INTERVAL = (0.30, 1.00)  # Explicit rejected-obstruction diagnostic envelope.
STRUCTURAL_FORCED_COMPACT_ANION_TRANSLATION_SCALE_INTERVAL = (0.45, 1.00)  # Explicit rejected-obstruction diagnostic envelope.
ROW_FIFTY_TWO_ID_TEXT = "52"  # Explicit user-requested row-level event-family ownership diagnostic selector.
ROW_FIFTY_THREE_ID_TEXT = "53"  # Explicit row-level LiTFSI preservation diagnostic selector.
EVENT_ATTRIBUTION_REPORTED_ROW_IDS = (int(ROW_FIFTY_TWO_ID_TEXT),)
PRIMITIVE_SENSITIVITY_REPORTED_ROW_ID_TEXTS = ("75", "82", "72", "52", "53")
OBSTRUCTION_REACHABILITY_SCALE_TEXT_VALUES = ("1.00", "0.75", "0.60", "0.45", "0.30")
FREE_TRANSLATION_INVERSE_GATE_SIGMA_MS_CM = 7.0  # Explicit row-52 diagnostic gate from the rejected obstruction audit.


def main() -> None:
    species_catalog = AnalyticMoriSpeciesCatalog(
        solvents=SOLVENTS,
        salts=SALTS,
        additives=ADDITIVES,
        cations=CATION_PROPERTIES,
    )
    uncertainty_budget = StructuralPrimitiveUncertaintyBudget(
        association_logK_interval=STRUCTURAL_LOGK_INTERVAL,
        dielectric_decrement_scale_interval=STRUCTURAL_UNIT_SCALE_INTERVAL,
        jones_dole_scale_interval=STRUCTURAL_HALF_TO_ONE_AND_HALF_SCALE_INTERVAL,
        atmosphere_ep_scale_interval=STRUCTURAL_HALF_TO_ONE_AND_HALF_SCALE_INTERVAL,
        atmosphere_rel_scale_interval=STRUCTURAL_RELAXATION_SCALE_INTERVAL,
        lithium_charge_cloud_radius_interval_A=STRUCTURAL_LITHIUM_CHARGE_CLOUD_RADIUS_INTERVAL_A,
        anion_charge_cloud_radius_interval_A=STRUCTURAL_ANION_CHARGE_CLOUD_RADIUS_INTERVAL_A,
        cage_trapping_fraction_interval=STRUCTURAL_UNIT_SCALE_INTERVAL,
        jump_length_scale_interval=STRUCTURAL_HALF_TO_ONE_AND_HALF_SCALE_INTERVAL,
        conversion_rate_scale_interval=STRUCTURAL_CONVERSION_RATE_SCALE_INTERVAL,
        forced_free_li_translation_scale_interval=(
            STRUCTURAL_FORCED_FREE_LI_TRANSLATION_SCALE_INTERVAL
        ),
        forced_compact_anion_translation_scale_interval=(
            STRUCTURAL_FORCED_COMPACT_ANION_TRANSLATION_SCALE_INTERVAL
        ),
        certificate_threshold_mS_cm=CERTIFICATE_THRESHOLD_MILLI_SIEMENS_PER_CM,
    )
    audit = audit_analytic_mori_conductivity_against_property_db(
        DATA,
        T_REF_K,
        uncertainty_budget,
        species_catalog,
    )
    ablation_audit = audit_analytic_mori_ablation_suite_against_property_db(
        DATA,
        T_REF_K,
        uncertainty_budget,
        species_catalog,
        TOP_REPORTED_ROWS,
    )
    obstruction_reachability_scales = tuple(
        float(scale_text)
        for scale_text in OBSTRUCTION_REACHABILITY_SCALE_TEXT_VALUES
    )
    obstruction_reachability_metrics = (
        audit_analytic_mori_obstruction_reachability_against_property_db(
            DATA,
            T_REF_K,
            uncertainty_budget,
            species_catalog,
            obstruction_reachability_scales,
            obstruction_reachability_scales,
            int(ROW_FIFTY_TWO_ID_TEXT),
            int(ROW_FIFTY_THREE_ID_TEXT),
        )
    )
    inverse_target_audit = compute_free_translation_inverse_target_audit(
        audit.rows,
        FREE_TRANSLATION_INVERSE_GATE_SIGMA_MS_CM,
    )
    primitive_sensitivity_audit = (
        audit_analytic_mori_primitive_sensitivity_against_property_db(
            DATA,
            T_REF_K,
            uncertainty_budget,
            species_catalog,
            tuple(
                int(row_id_text)
                for row_id_text in PRIMITIVE_SENSITIVITY_REPORTED_ROW_ID_TEXTS
            ),
        )
    )

    print("analytic_mori_property_db_audit")
    _print_summary(audit)
    print("salt_family_metrics")
    for family_metrics in audit.salt_family_metrics:
        print(
            f"{family_metrics.family_name}: "
            f"count={family_metrics.count} "
            f"bias={family_metrics.bias_mS_cm:.6f} "
            f"mae={family_metrics.mae_mS_cm:.6f} "
            f"rmse={family_metrics.rmse_mS_cm:.6f}"
        )
    print("worst_analytic_mori_rows")
    for row in sorted(
        audit.rows,
        key=lambda audit_row: abs(audit_row.residual_mS_cm),
        reverse=True,
    )[:TOP_REPORTED_ROWS]:
        _print_row(row)
    print("primitive_sensitivity_worst_rows")
    _print_primitive_sensitivity_rows(primitive_sensitivity_audit)
    print("primitive_sensitivity_grouped_heads")
    _print_primitive_sensitivity_groups(primitive_sensitivity_audit)
    print("markov_event_family_attribution")
    _print_event_family_attributions(audit)
    print("analytic_mori_ablation_metrics")
    _print_ablation_summary(ablation_audit)
    for metric in ablation_audit.ablation_metrics:
        print(
            f"{metric.ablation_mode}: "
            f"evaluated={metric.evaluated_rows} "
            f"failed={metric.failed_rows} "
            f"mae={metric.mae_mS_cm:.6f} "
            f"rmse={metric.rmse_mS_cm:.6f} "
            f"bias={metric.bias_mS_cm:.6f} "
            f"pearson_r={metric.pearson_r:.6f}"
        )
    print("worst_analytic_mori_ablation_rows")
    for row_decomposition in ablation_audit.worst_row_decompositions:
        _print_ablation_row(row_decomposition)
    print("free_carrier_obstruction_reachability")
    _print_obstruction_reachability(obstruction_reachability_metrics)
    print("free_translation_inverse_targets")
    _print_free_translation_inverse_targets(inverse_target_audit)
    if audit.failures:
        print("analytic_mori_failures")
        for failure in audit.failures[:TOP_REPORTED_ROWS]:
            print(f"row_id={failure.row_id} error={failure.error}")


def _print_summary(audit: AnalyticMoriPropertyDbAuditResult) -> None:
    print(f"labeled_rows={audit.labeled_rows}")
    print(f"evaluated_rows={audit.evaluated_rows}")
    print(f"failed_rows={audit.failed_rows}")
    print(f"mae_mS_cm={audit.mae_mS_cm:.6f}")
    print(f"rmse_mS_cm={audit.rmse_mS_cm:.6f}")
    print(f"bias_mS_cm={audit.bias_mS_cm:.6f}")
    print(f"mape_percent={audit.mape_percent:.6f}")
    print(f"r2={audit.r2:.6f}")
    print(f"pearson_r={audit.pearson_r:.6f}")
    print(f"certificate_coverage_fraction={audit.certificate_coverage_fraction:.6f}")
    print(f"certified_0p25_count={audit.certified_0p25_count}")
    print(
        "descriptor_complete_prediction_count="
        f"{audit.descriptor_complete_prediction_count}"
    )
    print(
        "equation_domain_violation_count="
        f"{audit.equation_domain_violation_count}"
    )
    print(f"max_mass_balance_residual_M={audit.max_mass_balance_residual_M:.6e}")
    print(f"max_row_sum_residual={audit.max_row_sum_residual:.6e}")
    print(f"max_stationary_residual={audit.max_stationary_residual:.6e}")
    print(f"max_detailed_balance_residual={audit.max_detailed_balance_residual:.6e}")
    print(
        "max_event_reversal_residual_mol_m3_s="
        f"{audit.max_event_reversal_residual_mol_m3_s:.6e}"
    )
    print(f"over_association_warning_count={audit.over_association_warning_count}")
    print(f"large_cancellation_warning_count={audit.large_cancellation_warning_count}")
    print(f"dielectric_collapse_warning_count={audit.dielectric_collapse_warning_count}")
    print(f"uncertified_population_warning_count={audit.uncertified_population_warning_count}")


def _print_ablation_summary(audit: AnalyticMoriAblationAuditResult) -> None:
    print(f"ablation_labeled_rows={audit.labeled_rows}")
    print(f"ablation_evaluated_rows={audit.evaluated_rows}")
    print(f"ablation_failed_rows={audit.failed_rows}")


def _print_row(row: AnalyticMoriPropertyDbRow) -> None:
    print(
        f"row_id={row.row_id} "
        f"salt_family={row.salt_family} "
        f"solvent_family={row.solvent_family} "
        f"additive_basis={row.additive_basis} "
        f"salt_molarity_M={row.salt_molarity_M:.6f} "
        f"sigma_empirical={row.empirical_sigma_mS_cm:.6f} "
        f"sigma_analytic_mori={row.analytic_mori_sigma_mS_cm:.6f} "
        f"residual={row.residual_mS_cm:.6f} "
        f"uncertainty_bound={row.uncertainty_bound_mS_cm:.6f} "
        f"sigma_interval_min={row.sigma_interval_min_mS_cm:.6f} "
        f"sigma_interval_max={row.sigma_interval_max_mS_cm:.6f} "
        f"certificate_half_width={row.certificate_half_width_mS_cm:.6f} "
        f"dominant_uncertainty_head={row.dominant_uncertainty_head} "
        f"covers_empirical={row.certificate_covers_empirical} "
        f"certified_0p25={row.certified_0p25_mS_cm} "
        f"prediction_status={row.prediction_status} "
        f"epsilon_mixture={row.epsilon_mixture:.6f} "
        f"epsilon_association={row.epsilon_association:.6f} "
        f"epsilon_atmosphere={row.epsilon_atmosphere:.6f} "
        f"epsilon={row.effective_dielectric:.6f} "
        f"eta_cP={row.effective_viscosity_cP:.6f} "
        f"kappa_inv_A={row.debye_kappa_inv_A:.6f} "
        f"steric_volume_fraction={row.steric_volume_fraction:.6f} "
        f"carrier_relaxation_form_factor_min={row.carrier_relaxation_form_factor_min:.6f} "
        f"carrier_charge_cloud_radius_A_max={row.carrier_charge_cloud_radius_A_max:.6f} "
        f"R_ep_trace={row.atmosphere_ep_trace_kg_s:.6e} "
        f"R_rel_trace={row.atmosphere_rel_trace_kg_s:.6e} "
        f"R_rel_Li_Li_trace={row.atmosphere_rel_li_li_trace_kg_s:.6e} "
        f"R_rel_anion_anion_trace={row.atmosphere_rel_anion_anion_trace_kg_s:.6e} "
        f"R_rel_Li_anion_cross={row.atmosphere_rel_li_anion_cross_frobenius_kg_s:.6e} "
        f"R_rel_anion_anion_cross={row.atmosphere_rel_anion_anion_cross_frobenius_kg_s:.6e} "
        f"Li_form_factor_squared={row.lithium_form_factor_squared:.6f} "
        f"anion_form_factor_squared_min={row.anion_form_factor_squared_min:.6f} "
        f"Li_anion_cross_form_factor_min={row.lithium_anion_cross_form_factor_min:.6f} "
        f"carrier_caged_fraction_max={row.carrier_caged_fraction_max:.6f} "
        f"carrier_caged_diffusion_scale_min={row.carrier_caged_diffusion_scale_min:.6f} "
        f"carrier_cage_exchange_rate_max_s_inv={row.carrier_cage_exchange_rate_max_s_inv:.6e} "
        f"selective_cage_driver={row.selective_cage_driver:.6f} "
        f"selective_caged_fraction_max={row.selective_caged_fraction_max:.6f} "
        f"selective_caged_diffusion_scale_min={row.selective_caged_diffusion_scale_min:.6f} "
        f"descriptor_release_driver={row.descriptor_release_driver:.6f} "
        f"atmosphere_relaxation_scale={row.atmosphere_relaxation_scale:.6f} "
        f"atmosphere_electrophoretic_scale={row.atmosphere_electrophoretic_scale:.6f} "
        f"backjump_cage_driver={row.backjump_cage_driver:.6f} "
        f"backjump_f_cage_Li={row.backjump_f_cage_Li:.6f} "
        f"backjump_g_attempt_Li={row.backjump_g_attempt_Li:.6f} "
        f"backjump_p_back_Li={row.backjump_p_back_Li:.6f} "
        f"backjump_exit_rate_s_inv={row.backjump_exit_rate_s_inv:.6e} "
        f"backjump_length_A={row.backjump_length_A:.6f} "
        f"backjump_direct_sigma={row.backjump_direct_sigma_mS_cm:.6f} "
        f"backjump_corrector_sigma={row.backjump_corrector_sigma_mS_cm:.6f} "
        f"backjump_net_sigma_delta={row.backjump_net_sigma_delta_mS_cm:.6f} "
        f"ordinary_translation_fraction_Li={row.ordinary_translation_fraction_Li:.6f} "
        f"free_Li_obstruction_factor={row.free_li_obstruction_factor:.6f} "
        f"free_Li_translation_diffusion_scale={row.free_li_translation_diffusion_scale:.6f} "
        f"free_anion_obstruction_factor_max={row.free_anion_obstruction_factor_max:.6f} "
        f"free_anion_translation_diffusion_scale_min={row.free_anion_translation_diffusion_scale_min:.6f} "
        f"obstruction_steric_driver={row.obstruction_steric_driver:.6f} "
        f"obstruction_compact_anion_driver={row.obstruction_compact_anion_driver:.6f} "
        f"obstruction_carbonate_driver={row.obstruction_carbonate_driver:.6f} "
        f"obstruction_high_salt_driver={row.obstruction_high_salt_driver:.6f} "
        f"obstruction_low_donor_driver={row.obstruction_low_donor_driver:.6f} "
        f"free_Li_translation_marginal_net={row.free_li_translation_marginal_net_mS_cm:.6f} "
        f"free_anion_translation_marginal_net={row.free_anion_translation_marginal_net_mS_cm:.6f} "
        f"free_Li_fraction={row.free_li_fraction:.6f} "
        f"free_anion_fraction={row.free_anion_fraction:.6f} "
        f"neutral_aggregate_fraction={row.neutral_aggregate_fraction:.6f} "
        f"markov_corrector_over_direct={row.markov_corrector_over_direct:.6f} "
        f"over_association_warning={row.over_association_warning} "
        f"large_cancellation_warning={row.large_cancellation_warning} "
        f"dielectric_collapse_warning={row.dielectric_collapse_warning} "
        f"uncertified_population_warning={row.uncertified_population_warning} "
        f"mass_residual_M={row.mass_balance_max_abs_residual_M:.6e} "
        f"row_sum={row.row_sum_residual:.6e} "
        f"stationary={row.stationary_residual:.6e} "
        f"detailed_balance={row.detailed_balance_residual:.6e} "
        f"event_reversal={row.event_reversal_residual_mol_m3_s:.6e} "
        f"direct_mori_sigma={row.direct_mori_sigma_mS_cm:.6f} "
        f"markov_direct_sigma={row.markov_direct_sigma_mS_cm:.6f} "
        f"markov_corrector_sigma={row.markov_corrector_sigma_mS_cm:.6f} "
        f"markov_total_sigma={row.markov_total_sigma_mS_cm:.6f} "
        f"min_effective_axis_density={row.minimum_effective_axis_density_m2_s_mol_m3:.6e}"
    )


def _print_event_family_attributions(audit: AnalyticMoriPropertyDbAuditResult) -> None:
    row_by_id = {
        row.row_id: row
        for row in audit.rows
    }
    for row_id in EVENT_ATTRIBUTION_REPORTED_ROW_IDS:
        if row_id not in row_by_id:
            print(f"row_id={row_id} event_family_attribution_status=missing")
            continue
        row = row_by_id[row_id]
        for attribution in row.markov_event_family_attributions:
            print(
                f"row_id={row.row_id} "
                f"family_label={attribution.family_label} "
                f"direct_sigma_mS_cm={attribution.direct_sigma_mS_cm:.6f} "
                f"self_corrector_sigma_mS_cm={attribution.self_corrector_sigma_mS_cm:.6f} "
                f"marginal_corrector_sigma_mS_cm={attribution.marginal_corrector_sigma_mS_cm:.6f} "
                f"marginal_net_sigma_mS_cm={attribution.marginal_net_sigma_mS_cm:.6f} "
                f"direct_fraction={attribution.direct_fraction:.6f} "
                f"marginal_net_fraction={attribution.marginal_net_fraction:.6f}"
            )


def _print_primitive_sensitivity_rows(
    audit: PrimitiveSensitivityAuditResult,
) -> None:
    print(f"log_parameter_step={audit.log_parameter_step:.6f}")
    for row in audit.rows:
        print(
            f"row_id={row.row_id} "
            f"empirical={row.empirical_sigma_mS_cm:.6f} "
            f"baseline={row.baseline_sigma_mS_cm:.6f} "
            f"residual={row.residual_mS_cm:.6f} "
            f"primitive_head={row.primitive_head} "
            f"baseline_value={row.baseline_value:.6f} "
            f"sigma_minus={row.sigma_minus_mS_cm:.6f} "
            f"sigma_plus={row.sigma_plus_mS_cm:.6f} "
            f"sensitivity={row.sensitivity_mS_cm_per_log_unit:.6f} "
            f"required_change_status={row.required_change_status} "
            f"required_log_change={row.required_log_change:.6f} "
            f"required_scale={row.required_scale:.6f} "
            f"positive_improves_abs_residual={row.positive_direction_improves_abs_residual} "
            f"positive_can_reduce_residual={row.positive_direction_can_reduce_residual}"
        )


def _print_primitive_sensitivity_groups(
    audit: PrimitiveSensitivityAuditResult,
) -> None:
    for group in audit.groups:
        print(
            f"primitive_head={group.primitive_head} "
            f"rows_improved={group.rows_improved_count} "
            f"rows_worsened={group.rows_worsened_count} "
            f"mean_abs_residual_after_positive_step="
            f"{group.mean_abs_residual_after_positive_step_mS_cm:.6f} "
            f"finite_required_scale_count={group.finite_required_scale_count} "
            f"median_required_scale={group.median_required_scale:.6f}"
        )


def _print_ablation_row(row: AnalyticMoriWorstRowPrimitiveDecomposition) -> None:
    prediction_text = " ".join(
        f"{prediction.ablation_mode}={prediction.sigma_mS_cm:.6f}"
        for prediction in row.ablation_predictions
    )
    print(
        f"row_id={row.row_id} "
        f"salt_family={row.salt_family} "
        f"solvent_family={row.solvent_family} "
        f"additive_basis={row.additive_basis} "
        f"sigma_empirical={row.empirical_sigma_mS_cm:.6f} "
        f"sigma_baseline={row.baseline_sigma_mS_cm:.6f} "
        f"baseline_residual={row.baseline_residual_mS_cm:.6f} "
        f"free_Li_fraction={row.free_li_fraction:.6f} "
        f"free_anion_fraction={row.free_anion_fraction:.6f} "
        f"SSIP_fraction={row.ssip_fraction:.6f} "
        f"CIP_fraction={row.cip_fraction:.6f} "
        f"charged_aggregate_fraction={row.charged_aggregate_fraction:.6f} "
        f"neutral_aggregate_fraction={row.neutral_aggregate_fraction:.6f} "
        f"effective_dielectric={row.effective_dielectric:.6f} "
        f"effective_viscosity_cP={row.effective_viscosity_cP:.6f} "
        f"Debye_length_A={row.debye_kappa_inv_A:.6f} "
        f"steric_volume_fraction={row.steric_volume_fraction:.6f} "
        f"carrier_relaxation_form_factor_min={row.carrier_relaxation_form_factor_min:.6f} "
        f"carrier_charge_cloud_radius_A_max={row.carrier_charge_cloud_radius_A_max:.6f} "
        f"selective_cage_driver={row.selective_cage_driver:.6f} "
        f"selective_caged_fraction_max={row.selective_caged_fraction_max:.6f} "
        f"selective_caged_diffusion_scale_min={row.selective_caged_diffusion_scale_min:.6f} "
        f"descriptor_release_driver={row.descriptor_release_driver:.6f} "
        f"atmosphere_relaxation_scale={row.atmosphere_relaxation_scale:.6f} "
        f"atmosphere_electrophoretic_scale={row.atmosphere_electrophoretic_scale:.6f} "
        f"timescale_structural_cage_fraction_max={row.timescale_structural_cage_fraction_max:.6f} "
        f"timescale_structural_De_max={row.timescale_structural_de_hop_structural_max:.6f} "
        f"timescale_structural_atmosphere_ratio_max={row.timescale_structural_atmosphere_ratio_max:.6f} "
        f"timescale_structural_size_void_ratio_max={row.timescale_structural_size_void_ratio_max:.6f} "
        f"timescale_structural_capture_fraction_max={row.timescale_structural_capture_fraction_max:.6f} "
        f"free_Li_obstruction_factor={row.free_li_obstruction_factor:.6f} "
        f"free_Li_translation_diffusion_scale={row.free_li_translation_diffusion_scale:.6f} "
        f"free_anion_obstruction_factor_max={row.free_anion_obstruction_factor_max:.6f} "
        f"free_anion_translation_diffusion_scale_min={row.free_anion_translation_diffusion_scale_min:.6f} "
        f"obstruction_steric_driver={row.obstruction_steric_driver:.6f} "
        f"obstruction_compact_anion_driver={row.obstruction_compact_anion_driver:.6f} "
        f"obstruction_carbonate_driver={row.obstruction_carbonate_driver:.6f} "
        f"obstruction_high_salt_driver={row.obstruction_high_salt_driver:.6f} "
        f"obstruction_low_donor_driver={row.obstruction_low_donor_driver:.6f} "
        f"free_Li_translation_marginal_net={row.free_li_translation_marginal_net_mS_cm:.6f} "
        f"free_anion_translation_marginal_net={row.free_anion_translation_marginal_net_mS_cm:.6f} "
        f"sigma_free_Li={row.sigma_free_li_mS_cm:.6f} "
        f"sigma_free_anion={row.sigma_free_anion_mS_cm:.6f} "
        f"sigma_SSIP={row.sigma_ssip_mS_cm:.6f} "
        f"sigma_CIP={row.sigma_cip_mS_cm:.6f} "
        f"sigma_aggregates={row.sigma_aggregates_mS_cm:.6f} "
        f"local_resistance_trace={row.local_resistance_trace_kg_s:.6e} "
        f"binding_resistance_trace={row.binding_resistance_trace_kg_s:.6e} "
        f"atmosphere_resistance_trace={row.atmosphere_resistance_trace_kg_s:.6e} "
        f"{prediction_text}"
    )
    for attribution in row.timescale_event_family_attributions:
        if not attribution.family_label.startswith("timescale_"):
            continue
        print(
            f"row_id={row.row_id} "
            f"ablation_mode=timescale_structural_cage_memory "
            f"family_label={attribution.family_label} "
            f"direct_sigma_mS_cm={attribution.direct_sigma_mS_cm:.6f} "
            f"self_corrector_sigma_mS_cm={attribution.self_corrector_sigma_mS_cm:.6f} "
            f"marginal_corrector_sigma_mS_cm={attribution.marginal_corrector_sigma_mS_cm:.6f} "
            f"marginal_net_sigma_mS_cm={attribution.marginal_net_sigma_mS_cm:.6f} "
            f"direct_fraction={attribution.direct_fraction:.6f} "
            f"marginal_net_fraction={attribution.marginal_net_fraction:.6f}"
        )


def _print_obstruction_reachability(
    metrics: tuple[AnalyticMoriObstructionReachabilityMetric, ...],
) -> None:
    for metric in metrics:
        print(
            f"s_Li={metric.free_li_translation_scale:.2f} "
            f"s_anion={metric.compact_anion_translation_scale:.2f} "
            f"evaluated={metric.evaluated_rows} "
            f"failed={metric.failed_rows} "
            f"mae={metric.mae_mS_cm:.6f} "
            f"bias={metric.bias_mS_cm:.6f} "
            f"pearson_r={metric.pearson_r:.6f} "
            f"row52_sigma={metric.row52_sigma_mS_cm:.6f} "
            f"row53_sigma={metric.row53_sigma_mS_cm:.6f} "
            f"LiPF6_MAE={metric.lipf6_mae_mS_cm:.6f} "
            f"LiTFSI_MAE={metric.litfsi_mae_mS_cm:.6f} "
            f"LiFSI_MAE={metric.lifsi_mae_mS_cm:.6f}"
        )


def _print_free_translation_inverse_targets(
    audit: FreeTranslationInverseAuditResult,
) -> None:
    print(f"gate_sigma_mS_cm={audit.gate_sigma_mS_cm:.6f}")
    neighborhood_by_row_id = {
        neighborhood.row_id: neighborhood
        for neighborhood in audit.neighborhoods
    }
    print("top_free_translation_suppression_targets")
    suppression_targets = tuple(
        target
        for target in sorted(
            audit.targets,
            key=lambda inverse_target: inverse_target.required_common_free_scale_to_empirical,
        )
        if target.required_common_free_scale_to_empirical < 1.0
    )[:TOP_REPORTED_ROWS]
    for target in suppression_targets:
        _print_free_translation_inverse_target(
            target,
            neighborhood_by_row_id[target.row_id],
        )
    print("top_free_translation_enhancement_targets")
    enhancement_targets = tuple(
        target
        for target in sorted(
            audit.targets,
            key=lambda inverse_target: inverse_target.required_common_free_scale_to_empirical,
            reverse=True,
        )
        if target.required_common_free_scale_to_empirical > 1.0
    )[:TOP_REPORTED_ROWS]
    for target in enhancement_targets:
        _print_free_translation_inverse_target(
            target,
            neighborhood_by_row_id[target.row_id],
        )
    print("free_translation_inverse_by_salt_family")
    _print_free_translation_inverse_group_metrics(audit.salt_family_metrics)
    print("free_translation_inverse_by_solvent_family")
    _print_free_translation_inverse_group_metrics(audit.solvent_family_metrics)
    print("free_translation_inverse_by_steric_volume_bin")
    _print_free_translation_inverse_group_metrics(audit.steric_volume_bin_metrics)
    print("free_translation_inverse_by_obstruction_driver_bin")
    _print_free_translation_inverse_group_metrics(audit.obstruction_driver_bin_metrics)


def _print_free_translation_inverse_target(
    target: FreeTranslationInverseTarget,
    neighborhood: FreeTranslationInverseNeighborhood,
) -> None:
    print(
        f"row_id={target.row_id} "
        f"salt_family={target.salt_family} "
        f"solvent_family={target.solvent_family} "
        f"salt_molarity_M={target.salt_molarity_M:.6f} "
        f"sigma_empirical={target.empirical_sigma_mS_cm:.6f} "
        f"sigma_predicted={target.predicted_sigma_mS_cm:.6f} "
        f"residual={target.residual_mS_cm:.6f} "
        f"free_Li_marginal={target.free_li_marginal_mS_cm:.6f} "
        f"free_anion_marginal={target.free_anion_marginal_mS_cm:.6f} "
        f"free_translation_marginal={target.free_translation_marginal_mS_cm:.6f} "
        f"required_common_free_scale={target.required_common_free_scale_to_empirical:.6f} "
        f"required_common_free_scale_clipped={target.required_common_free_scale_to_empirical_clipped:.6f} "
        f"required_common_free_scale_to_gate={target.required_common_free_scale_to_gate:.6f} "
        f"required_common_free_scale_to_gate_clipped={target.required_common_free_scale_to_gate_clipped:.6f} "
        f"required_Li_scale_if_anion_fixed={target.required_li_scale_if_anion_fixed:.6f} "
        f"required_anion_scale_if_Li_fixed={target.required_anion_scale_if_li_fixed:.6f} "
        f"steric_volume_fraction={target.steric_volume_fraction:.6f} "
        f"obstruction_driver={target.obstruction_driver:.6f} "
        f"neighbor_count={neighborhood.neighbor_count} "
        f"neighbor_median_required_scale={neighborhood.median_neighbor_required_scale:.6f} "
        f"neighbor_min_required_scale={neighborhood.min_neighbor_required_scale:.6f} "
        f"neighbor_median_residual={neighborhood.median_neighbor_residual_mS_cm:.6f} "
        f"same_salt_family_count={neighborhood.same_salt_family_count} "
        f"same_solvent_family_count={neighborhood.same_solvent_family_count} "
        f"has_systematic_cluster={neighborhood.has_systematic_cluster} "
        f"prediction_status={target.prediction_status}"
    )


def _print_free_translation_inverse_group_metrics(
    metrics: tuple[FreeTranslationInverseGroupMetric, ...],
) -> None:
    for metric in metrics:
        print(
            f"group_kind={metric.group_kind} "
            f"group_label={metric.group_label} "
            f"count={metric.count} "
            f"median_required_common_free_scale={metric.median_required_common_free_scale:.6f} "
            f"iqr_required_common_free_scale={metric.iqr_required_common_free_scale:.6f} "
            f"median_required_common_free_scale_clipped={metric.median_required_common_free_scale_clipped:.6f} "
            f"mean_signed_error={metric.mean_signed_error_mS_cm:.6f}"
        )


if __name__ == "__main__":
    main()
