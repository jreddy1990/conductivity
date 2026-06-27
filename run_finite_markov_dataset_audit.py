"""Run the finite Markov conductivity generator against empirical rows."""

from __future__ import annotations

from constants import T_REF_K
from conductivity.finite_markov_conductivity import (
    ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF,
    ANION_DIAGONAL_RELAXATION_FORM_FACTOR_RESOLVED_STATE_FINITE_SIZE,
    ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
    RELAXATION_DYNAMIC_RESPONSE_OFF,
    RELAXATION_DYNAMIC_RESPONSE_STATE_LIFETIME,
)
from conductivity.finite_markov_dataset_audit import (
    ActiveLearningUtilityMetrics,
    BaseSpeciationInverseMetrics,
    BiasLedgerRow,
    DatasetAuditResult,
    FamilyAtmosphereMetrics,
    PredictiveValidationGroupMetrics,
    PromotionDecisionMetrics,
    RequiredAnionDiagFactorMetrics,
    audit_empirical_conductivity_dataset,
)
from data.electrolyte_property_db import DATA


TOP_REPORTED_ROWS = 12  # Explicit constant: compact but enough to expose family-level failures.


def main() -> None:
    audit = audit_empirical_conductivity_dataset(
        DATA,
        T_REF_K,
        ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
        RELAXATION_DYNAMIC_RESPONSE_OFF,
        ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF,
    )
    state_lifetime_audit = audit_empirical_conductivity_dataset(
        DATA,
        T_REF_K,
        ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
        RELAXATION_DYNAMIC_RESPONSE_STATE_LIFETIME,
        ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF,
    )
    finite_size_anion_audit = audit_empirical_conductivity_dataset(
        DATA,
        T_REF_K,
        ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
        RELAXATION_DYNAMIC_RESPONSE_OFF,
        ANION_DIAGONAL_RELAXATION_FORM_FACTOR_RESOLVED_STATE_FINITE_SIZE,
    )

    print("finite_markov_dataset_audit")
    _print_audit_summary("bath_basis=total_formal relaxation_dynamic_response=off", audit)
    _print_audit_summary(
        "bath_basis=total_formal relaxation_dynamic_response=state_lifetime "
        "anion_diagonal_relaxation_form_factor=off",
        state_lifetime_audit,
    )
    _print_audit_summary(
        "bath_basis=total_formal relaxation_dynamic_response=off "
        "anion_diagonal_relaxation_form_factor=resolved_state_finite_size",
        finite_size_anion_audit,
    )

    print("salt_family_metrics")
    for salt_family in sorted(audit.salt_family_metrics):
        family_metrics = audit.salt_family_metrics[salt_family]
        print(
            f"{salt_family}: count={family_metrics.count} "
            f"bias={family_metrics.bias_mS_cm:.6f} "
            f"mae={family_metrics.mae_mS_cm:.6f} "
            f"rmse={family_metrics.rmse_mS_cm:.6f}"
        )

    print("family_atmosphere_metrics")
    for family_metrics in audit.family_atmosphere_metrics:
        _print_family_atmosphere_metrics(family_metrics)

    print("required_anion_diag_factor_metrics")
    for required_factor_metrics in audit.required_anion_diag_factor_metrics:
        _print_required_anion_diag_factor_metrics(required_factor_metrics)

    print("base_speciation_inverse_metrics")
    for inverse_metrics in audit.base_speciation_inverse_metrics:
        _print_base_speciation_inverse_metrics(inverse_metrics)

    print("promotion_decision_metrics")
    for decision_metrics in audit.promotion_decision_metrics:
        _print_promotion_decision_metrics(decision_metrics)

    print("predictive_validation_group_metrics")
    for validation_metrics in audit.predictive_validation_group_metrics:
        _print_predictive_validation_group_metrics(validation_metrics)

    print("active_learning_utility_metrics")
    _print_active_learning_utility_metrics(audit.active_learning_utility_metrics)

    print("state_lifetime_grouped_deltas")
    _print_grouped_delta_metrics(audit, state_lifetime_audit)

    print("anion_diagonal_relaxation_form_factor_grouped_deltas")
    _print_grouped_delta_metrics(audit, finite_size_anion_audit)

    print("worst_rows")
    worst_rows = sorted(
        audit.ledger_rows,
        key=lambda ledger_row: ledger_row.absolute_error_mS_cm,
        reverse=True,
    )[:TOP_REPORTED_ROWS]
    for ledger_row in worst_rows:
        _print_worst_row(ledger_row)

    if audit.failures:
        print("failed_rows_detail")
        for failure in audit.failures:
            print(f"row_id={failure.row_id} error={failure.error}")
        raise SystemExit("finite Markov dataset audit had failed empirical rows")
    if state_lifetime_audit.failures:
        print("state_lifetime_failed_rows_detail")
        for failure in state_lifetime_audit.failures:
            print(f"row_id={failure.row_id} error={failure.error}")
        raise SystemExit("state-lifetime dataset audit had failed empirical rows")
    if finite_size_anion_audit.failures:
        print("finite_size_anion_failed_rows_detail")
        for failure in finite_size_anion_audit.failures:
            print(f"row_id={failure.row_id} error={failure.error}")
        raise SystemExit("finite-size anion dataset audit had failed empirical rows")


def _print_audit_summary(label: str, audit: DatasetAuditResult) -> None:
    print(label)
    print(f"labeled_rows={audit.labeled_rows}")
    print(f"evaluated_rows={audit.evaluated_rows}")
    print(f"failed_rows={audit.failed_rows}")
    print(f"mae_mS_cm={audit.mae_mS_cm:.6f}")
    print(f"rmse_mS_cm={audit.rmse_mS_cm:.6f}")
    print(f"bias_mS_cm={audit.bias_mS_cm:.6f}")
    print(f"mape_percent={audit.mape_percent:.6f}")
    print(f"r2={audit.r2:.6f}")
    print(f"pearson_r={audit.pearson_r:.6f}")
    print(f"max_row_sum_residual_s_inv={audit.max_row_sum_residual_s_inv:.6e}")
    print(f"max_stationary_residual_s_inv={audit.max_stationary_residual_s_inv:.6e}")
    print(f"max_detailed_balance_residual_s_inv={audit.max_detailed_balance_residual_s_inv:.6e}")


def _print_family_atmosphere_metrics(family_metrics: FamilyAtmosphereMetrics) -> None:
    print(
        f"group={family_metrics.group_name} "
        f"value={family_metrics.group_value} "
        f"count={family_metrics.count} "
        f"bias={family_metrics.bias_mS_cm:.6f} "
        f"mae={family_metrics.mae_mS_cm:.6f} "
        f"rmse={family_metrics.rmse_mS_cm:.6f} "
        f"mean_log_sigma_error={family_metrics.mean_log_sigma_error:.6f} "
        f"mean_log_base_error={family_metrics.mean_log_base_error:.6f} "
        f"mean_log_atmosphere_error={family_metrics.mean_log_atmosphere_error:.6f} "
        f"mean_log_ep_error={family_metrics.mean_log_ep_error:.6f} "
        f"mean_log_rel_error={family_metrics.mean_log_rel_error:.6f} "
        f"mean_log_rel_Li_error={family_metrics.mean_log_rel_Li_error:.6f} "
        f"mean_log_rel_anion_error={family_metrics.mean_log_rel_anion_error:.6f} "
        f"mean_log_rel_diag_error={family_metrics.mean_log_rel_diag_error:.6f} "
        f"mean_log_rel_cross_error={family_metrics.mean_log_rel_cross_error:.6f} "
        f"mean_log_rel_before_gate_error={family_metrics.mean_log_rel_before_gate_error:.6f} "
        f"mean_log_rel_after_gate_error={family_metrics.mean_log_rel_after_gate_error:.6f} "
        f"H={family_metrics.mean_H_atmosphere:.6f} "
        f"H_target={family_metrics.mean_H_atmosphere_target:.6f} "
        f"H_ratio={family_metrics.mean_H_ratio:.6f} "
        f"H_ep={family_metrics.mean_H_ep:.6f} "
        f"H_rel={family_metrics.mean_H_rel:.6f} "
        f"H_rel_Li={family_metrics.mean_H_rel_Li:.6f} "
        f"H_rel_anion={family_metrics.mean_H_rel_anion:.6f} "
        f"H_rel_diag={family_metrics.mean_H_rel_diag:.6f} "
        f"H_rel_cross={family_metrics.mean_H_rel_cross:.6f} "
        f"H_rel_before_gate={family_metrics.mean_H_rel_before_gate:.6f} "
        f"H_rel_after_gate={family_metrics.mean_H_rel_after_gate:.6f} "
        f"H_full={family_metrics.mean_H_full:.6f} "
        f"r_atm_current={family_metrics.mean_r_atm_current:.6f} "
        f"r_atm_target={family_metrics.mean_r_atm_target:.6f} "
        f"r_atm_current_over_target={family_metrics.mean_r_atm_current_over_target:.6f} "
        f"drag_ep_current_over_target={family_metrics.mean_drag_ep_current_over_target:.6f} "
        f"drag_rel_current_over_target={family_metrics.mean_drag_rel_current_over_target:.6f} "
        f"drag_rel_Li_current_over_target="
        f"{family_metrics.mean_drag_rel_Li_current_over_target:.6f} "
        f"drag_rel_anion_current_over_target="
        f"{family_metrics.mean_drag_rel_anion_current_over_target:.6f} "
        f"drag_rel_diag_current_over_target="
        f"{family_metrics.mean_drag_rel_diag_current_over_target:.6f} "
        f"drag_rel_cross_current_over_target="
        f"{family_metrics.mean_drag_rel_cross_current_over_target:.6f} "
        f"drag_rel_before_gate_current_over_target="
        f"{family_metrics.mean_drag_rel_before_gate_current_over_target:.6f} "
        f"drag_rel_after_gate_current_over_target="
        f"{family_metrics.mean_drag_rel_after_gate_current_over_target:.6f} "
        f"relaxation_gate={family_metrics.mean_relaxation_lifetime_gate:.6f} "
        f"g_anion_diag_required={family_metrics.mean_g_anion_diag_required:.6f} "
        f"anion_charge_cloud_radius_required_A="
        f"{family_metrics.mean_anion_charge_cloud_radius_required_A:.6f} "
        f"hydrodynamic_radius_A={family_metrics.mean_hydrodynamic_radius_A:.6f} "
        f"shape_factor={family_metrics.mean_shape_factor:.6f} "
        f"current_self_form_factor={family_metrics.mean_current_self_form_factor:.6f} "
        f"eta_rel={family_metrics.mean_eta_rel:.6f} "
        f"kappa_inv_A={family_metrics.mean_kappa_inv_A:.6f} "
        f"ionic_strength_mol_m3={family_metrics.mean_ionic_strength_mol_m3:.6f} "
        f"top_state={family_metrics.dominant_top_state} "
        f"owner={family_metrics.owner}"
    )


def _print_required_anion_diag_factor_metrics(
    required_factor_metrics: RequiredAnionDiagFactorMetrics,
) -> None:
    print(
        f"group={required_factor_metrics.group_name} "
        f"value={required_factor_metrics.group_value} "
        f"count={required_factor_metrics.count} "
        f"median_g_required={required_factor_metrics.median_g_required:.6f} "
        f"IQR_g_required={required_factor_metrics.iqr_g_required:.6f} "
        f"status_counts={_format_int_map(required_factor_metrics.status_counts)} "
        f"median_charge_cloud_radius_required_A="
        f"{required_factor_metrics.median_charge_cloud_radius_required_A:.6f} "
        f"median_hydrodynamic_radius_A="
        f"{required_factor_metrics.median_hydrodynamic_radius_A:.6f} "
        f"median_shape_factor={required_factor_metrics.median_shape_factor:.6f} "
        f"median_current_self_form_factor="
        f"{required_factor_metrics.median_current_self_form_factor:.6f} "
        f"charge_cloud_descriptor_covered_count="
        f"{required_factor_metrics.charge_cloud_descriptor_covered_count} "
        f"median_charge_cloud_descriptor_radius_A="
        f"{required_factor_metrics.median_charge_cloud_descriptor_radius_A:.6f} "
        f"charge_cloud_source_counts={_format_int_map(required_factor_metrics.charge_cloud_source_counts)} "
        f"mean_signed_error_mS_cm={required_factor_metrics.mean_signed_error_mS_cm:.6f}"
    )


def _print_base_speciation_inverse_metrics(
    inverse_metrics: BaseSpeciationInverseMetrics,
) -> None:
    print(
        f"group={inverse_metrics.group_name} "
        f"value={inverse_metrics.group_value} "
        f"count={inverse_metrics.count} "
        f"median_DeltaG_K_req_kJ_mol="
        f"{inverse_metrics.median_association_required_deltaG_kJ_mol:.6f} "
        f"IQR_DeltaG_K_req_kJ_mol="
        f"{inverse_metrics.iqr_association_required_deltaG_kJ_mol:.6f} "
        f"median_DeltaG_D_req_kJ_mol="
        f"{inverse_metrics.median_base_mobility_required_deltaG_kJ_mol:.6f} "
        f"IQR_DeltaG_D_req_kJ_mol="
        f"{inverse_metrics.iqr_base_mobility_required_deltaG_kJ_mol:.6f} "
        f"mean_signed_error_mS_cm={inverse_metrics.mean_signed_error_mS_cm:.6f}"
    )


def _print_promotion_decision_metrics(
    decision_metrics: PromotionDecisionMetrics,
) -> None:
    print(
        f"branch={decision_metrics.branch_name} "
        f"group={decision_metrics.group_name} "
        f"value={decision_metrics.group_value} "
        f"count={decision_metrics.count} "
        f"decision={decision_metrics.decision} "
        f"candidate={decision_metrics.candidate} "
        f"rationale={decision_metrics.rationale}"
    )


def _print_predictive_validation_group_metrics(
    validation_metrics: PredictiveValidationGroupMetrics,
) -> None:
    print(
        f"group={validation_metrics.group_name} "
        f"value={validation_metrics.group_value} "
        f"count={validation_metrics.count} "
        f"calibration_count={validation_metrics.calibration_count} "
        f"mae={validation_metrics.mae_mS_cm:.6f} "
        f"rmse={validation_metrics.rmse_mS_cm:.6f} "
        f"bias={validation_metrics.bias_mS_cm:.6f} "
        f"pearson_available={validation_metrics.pearson_r_available} "
        f"pearson_r={validation_metrics.pearson_r:.6f} "
        f"conformal_abs_error_80="
        f"{validation_metrics.conformal_abs_error_80_mS_cm:.6f} "
        f"conformal_coverage_80={validation_metrics.conformal_coverage_80:.6f} "
        f"conformal_abs_error_90="
        f"{validation_metrics.conformal_abs_error_90_mS_cm:.6f} "
        f"conformal_coverage_90={validation_metrics.conformal_coverage_90:.6f} "
        f"decision={validation_metrics.decision} "
        f"rationale={validation_metrics.rationale}"
    )


def _print_active_learning_utility_metrics(
    utility_metrics: ActiveLearningUtilityMetrics,
) -> None:
    print(
        f"candidate_count={utility_metrics.candidate_count} "
        f"selected_count={utility_metrics.selected_count} "
        f"hit_threshold_mS_cm={utility_metrics.hit_threshold_mS_cm:.6f} "
        f"selected_hit_count={utility_metrics.selected_hit_count} "
        f"true_hit_count={utility_metrics.true_hit_count} "
        f"selected_hit_rate={utility_metrics.selected_hit_rate:.6f} "
        f"random_hit_rate={utility_metrics.random_hit_rate:.6f} "
        f"enrichment_over_random={utility_metrics.enrichment_over_random:.6f} "
        f"best_measured_mS_cm={utility_metrics.best_measured_mS_cm:.6f} "
        f"best_selected_measured_mS_cm={utility_metrics.best_selected_measured_mS_cm:.6f} "
        f"regret_mS_cm={utility_metrics.regret_mS_cm:.6f} "
        f"decision={utility_metrics.decision} "
        f"rationale={utility_metrics.rationale}"
    )


def _print_grouped_delta_metrics(
    baseline_audit: DatasetAuditResult,
    comparison_audit: DatasetAuditResult,
) -> None:
    for group_name in ("additive_basis", "salt_family", "top_state_kind"):
        baseline_metrics_by_value = _family_metrics_by_value(baseline_audit, group_name)
        comparison_metrics_by_value = _family_metrics_by_value(comparison_audit, group_name)
        group_values = sorted(set(baseline_metrics_by_value) | set(comparison_metrics_by_value))
        for group_value in group_values:
            if group_value not in baseline_metrics_by_value or group_value not in comparison_metrics_by_value:
                raise ValueError(f"group {group_name}={group_value} missing from one relaxation audit")
            baseline_metrics = baseline_metrics_by_value[group_value]
            comparison_metrics = comparison_metrics_by_value[group_value]
            print(
                f"group={group_name} "
                f"value={group_value} "
                f"count_baseline={baseline_metrics.count} "
                f"count_comparison={comparison_metrics.count} "
                f"delta_bias={comparison_metrics.bias_mS_cm - baseline_metrics.bias_mS_cm:.6f} "
                f"bias_baseline={baseline_metrics.bias_mS_cm:.6f} "
                f"bias_comparison={comparison_metrics.bias_mS_cm:.6f} "
                f"delta_mae={comparison_metrics.mae_mS_cm - baseline_metrics.mae_mS_cm:.6f} "
                f"mae_baseline={baseline_metrics.mae_mS_cm:.6f} "
                f"mae_comparison={comparison_metrics.mae_mS_cm:.6f} "
                f"delta_log_atmosphere_error="
                f"{comparison_metrics.mean_log_atmosphere_error - baseline_metrics.mean_log_atmosphere_error:.6f} "
                f"log_atmosphere_error_baseline={baseline_metrics.mean_log_atmosphere_error:.6f} "
                f"log_atmosphere_error_comparison={comparison_metrics.mean_log_atmosphere_error:.6f} "
                f"delta_H_ratio={comparison_metrics.mean_H_ratio - baseline_metrics.mean_H_ratio:.6f} "
                f"H_ratio_baseline={baseline_metrics.mean_H_ratio:.6f} "
                f"H_ratio_comparison={comparison_metrics.mean_H_ratio:.6f} "
                f"delta_H={comparison_metrics.mean_H_atmosphere - baseline_metrics.mean_H_atmosphere:.6f} "
                f"H_baseline={baseline_metrics.mean_H_atmosphere:.6f} "
                f"H_comparison={comparison_metrics.mean_H_atmosphere:.6f} "
                f"H_target_baseline={baseline_metrics.mean_H_atmosphere_target:.6f} "
                f"H_target_comparison={comparison_metrics.mean_H_atmosphere_target:.6f} "
                f"delta_drag_rel_current_over_target="
                f"{comparison_metrics.mean_drag_rel_current_over_target - baseline_metrics.mean_drag_rel_current_over_target:.6f} "
                f"drag_rel_current_over_target_baseline="
                f"{baseline_metrics.mean_drag_rel_current_over_target:.6f} "
                f"drag_rel_current_over_target_comparison="
                f"{comparison_metrics.mean_drag_rel_current_over_target:.6f} "
                f"delta_drag_rel_Li_current_over_target="
                f"{comparison_metrics.mean_drag_rel_Li_current_over_target - baseline_metrics.mean_drag_rel_Li_current_over_target:.6f} "
                f"drag_rel_Li_current_over_target_baseline="
                f"{baseline_metrics.mean_drag_rel_Li_current_over_target:.6f} "
                f"drag_rel_Li_current_over_target_comparison="
                f"{comparison_metrics.mean_drag_rel_Li_current_over_target:.6f} "
                f"delta_drag_rel_anion_current_over_target="
                f"{comparison_metrics.mean_drag_rel_anion_current_over_target - baseline_metrics.mean_drag_rel_anion_current_over_target:.6f} "
                f"drag_rel_anion_current_over_target_baseline="
                f"{baseline_metrics.mean_drag_rel_anion_current_over_target:.6f} "
                f"drag_rel_anion_current_over_target_comparison="
                f"{comparison_metrics.mean_drag_rel_anion_current_over_target:.6f} "
                f"delta_drag_rel_diag_current_over_target="
                f"{comparison_metrics.mean_drag_rel_diag_current_over_target - baseline_metrics.mean_drag_rel_diag_current_over_target:.6f} "
                f"drag_rel_diag_current_over_target_baseline="
                f"{baseline_metrics.mean_drag_rel_diag_current_over_target:.6f} "
                f"drag_rel_diag_current_over_target_comparison="
                f"{comparison_metrics.mean_drag_rel_diag_current_over_target:.6f} "
                f"delta_drag_rel_cross_current_over_target="
                f"{comparison_metrics.mean_drag_rel_cross_current_over_target - baseline_metrics.mean_drag_rel_cross_current_over_target:.6f} "
                f"drag_rel_cross_current_over_target_baseline="
                f"{baseline_metrics.mean_drag_rel_cross_current_over_target:.6f} "
                f"drag_rel_cross_current_over_target_comparison="
                f"{comparison_metrics.mean_drag_rel_cross_current_over_target:.6f} "
                f"delta_drag_rel_after_gate_current_over_target="
                f"{comparison_metrics.mean_drag_rel_after_gate_current_over_target - baseline_metrics.mean_drag_rel_after_gate_current_over_target:.6f} "
                f"relaxation_gate_baseline={baseline_metrics.mean_relaxation_lifetime_gate:.6f} "
                f"relaxation_gate_comparison={comparison_metrics.mean_relaxation_lifetime_gate:.6f} "
                f"owner_baseline={baseline_metrics.owner} "
                f"owner_comparison={comparison_metrics.owner}"
            )


def _family_metrics_by_value(
    audit: DatasetAuditResult,
    group_name: str,
) -> dict[str, FamilyAtmosphereMetrics]:
    metrics_by_value: dict[str, FamilyAtmosphereMetrics] = {}
    for family_metrics in audit.family_atmosphere_metrics:
        if family_metrics.group_name != group_name:
            continue
        if family_metrics.group_value in metrics_by_value:
            raise ValueError(f"duplicate family atmosphere metric {group_name}={family_metrics.group_value}")
        metrics_by_value[family_metrics.group_value] = family_metrics
    return metrics_by_value


def _print_worst_row(ledger_row: BiasLedgerRow) -> None:
    uncorr_target_ratio = ledger_row.D_uncorr_m2_s / ledger_row.D_Q_target_m2_s
    print(
        f"row_id={ledger_row.row_id} "
        f"salt_family={ledger_row.salt_family} "
        f"solvent_family={ledger_row.solvent_family} "
        f"additives={ledger_row.additive_basis} "
        f"sigma_exp={ledger_row.sigma_exp_mS_cm:.6f} "
        f"sigma_pred={ledger_row.sigma_pred_mS_cm:.6f} "
        f"error={ledger_row.signed_error_mS_cm:.6f} "
        f"log_error={ledger_row.log_error:.6f} "
        f"D_uncorr_over_target={uncorr_target_ratio:.6f} "
        f"H_gen={ledger_row.H_gen:.6f} "
        f"D_NE_state={ledger_row.D_NE_state_m2_s:.6e} "
        f"D_after_binding={ledger_row.D_after_binding_m2_s:.6e} "
        f"D_after_atmosphere={ledger_row.D_after_atmosphere_m2_s:.6e} "
        f"D_state={ledger_row.D_state_m2_s:.6e} "
        f"H_binding={ledger_row.H_binding:.6f} "
        f"H_atmosphere={ledger_row.H_atmosphere:.6f} "
        f"H_atmosphere_target={ledger_row.H_atmosphere_target:.6f} "
        f"H_ep={ledger_row.H_ep:.6f} "
        f"H_rel={ledger_row.H_rel:.6f} "
        f"H_rel_Li={ledger_row.H_rel_Li:.6f} "
        f"H_rel_anion={ledger_row.H_rel_anion:.6f} "
        f"H_rel_diag={ledger_row.H_rel_diag:.6f} "
        f"H_rel_cross={ledger_row.H_rel_cross:.6f} "
        f"H_rel_before_gate={ledger_row.H_rel_before_gate:.6f} "
        f"H_rel_after_gate={ledger_row.H_rel_after_gate:.6f} "
        f"H_full={ledger_row.H_full:.6f} "
        f"drag_ep={ledger_row.drag_ep:.6f} "
        f"drag_rel={ledger_row.drag_rel:.6f} "
        f"drag_rel_Li={ledger_row.drag_rel_Li:.6f} "
        f"drag_rel_anion={ledger_row.drag_rel_anion:.6f} "
        f"drag_rel_diag={ledger_row.drag_rel_diag:.6f} "
        f"drag_rel_cross={ledger_row.drag_rel_cross:.6f} "
        f"drag_rel_before_gate={ledger_row.drag_rel_before_gate:.6f} "
        f"drag_rel_after_gate={ledger_row.drag_rel_after_gate:.6f} "
        f"drag_full={ledger_row.drag_full:.6f} "
        f"drag_ep_current_over_target={ledger_row.drag_ep_current_over_target:.6f} "
        f"drag_rel_current_over_target={ledger_row.drag_rel_current_over_target:.6f} "
        f"drag_rel_Li_current_over_target={ledger_row.drag_rel_Li_current_over_target:.6f} "
        f"drag_rel_anion_current_over_target={ledger_row.drag_rel_anion_current_over_target:.6f} "
        f"drag_rel_diag_current_over_target={ledger_row.drag_rel_diag_current_over_target:.6f} "
        f"drag_rel_cross_current_over_target={ledger_row.drag_rel_cross_current_over_target:.6f} "
        f"D_none_state={ledger_row.D_none_state_m2_s:.6e} "
        f"D_ep_state={ledger_row.D_ep_state_m2_s:.6e} "
        f"D_rel_state={ledger_row.D_rel_state_m2_s:.6e} "
        f"D_rel_Li_state={ledger_row.D_rel_Li_state_m2_s:.6e} "
        f"D_rel_anion_state={ledger_row.D_rel_anion_state_m2_s:.6e} "
        f"D_rel_diag_state={ledger_row.D_rel_diag_state_m2_s:.6e} "
        f"D_rel_full_state={ledger_row.D_rel_full_state_m2_s:.6e} "
        f"D_full_state={ledger_row.D_full_state_m2_s:.6e} "
        f"r_atm_current={ledger_row.r_atmosphere_current:.6f} "
        f"r_atm_target={ledger_row.r_atmosphere_target:.6f} "
        f"r_atm_current_over_target={ledger_row.r_atmosphere_current_over_target:.6f} "
        f"atmosphere_bath_basis={ledger_row.atmosphere_bath_basis} "
        f"ionic_strength_total={ledger_row.ionic_strength_total_mol_m3:.6f} "
        f"ionic_strength_external={ledger_row.ionic_strength_external_mol_m3:.6f} "
        f"external_over_total_ionic_strength={ledger_row.external_over_total_ionic_strength:.6f} "
        f"H_atmosphere_total_bath={ledger_row.H_atmosphere_total_bath:.6f} "
        f"H_atmosphere_total_bath_evaluated={ledger_row.H_atmosphere_total_bath_evaluated} "
        f"H_atmosphere_external_bath={ledger_row.H_atmosphere_external_bath:.6f} "
        f"H_atmosphere_external_bath_evaluated={ledger_row.H_atmosphere_external_bath_evaluated} "
        f"relaxation_dynamic_response={ledger_row.relaxation_dynamic_response} "
        f"anion_diagonal_relaxation_form_factor={ledger_row.anion_diagonal_relaxation_form_factor} "
        f"mean_relaxation_lifetime_gate={ledger_row.mean_relaxation_lifetime_gate:.6f} "
        f"g_anion_diag_required={ledger_row.g_anion_diag_required:.6f} "
        f"g_anion_diag_current={ledger_row.g_anion_diag_current:.6f} "
        f"g_anion_diag_required_status={ledger_row.g_anion_diag_required_status} "
        f"anion_charge_cloud_radius_required_A="
        f"{ledger_row.anion_charge_cloud_radius_required_A:.6f} "
        f"hydrodynamic_radius_A={ledger_row.hydrodynamic_radius_A:.6f} "
        f"top_state_shape_factor={ledger_row.shape_factor:.6f} "
        f"current_self_form_factor={ledger_row.current_self_form_factor:.6f} "
        f"charge_cloud_radius_available={ledger_row.charge_cloud_radius_available} "
        f"charge_cloud_radius_A={ledger_row.charge_cloud_radius_A:.6f} "
        f"charge_cloud_source={ledger_row.charge_cloud_source} "
        f"charge_cloud_site_count={ledger_row.charge_cloud_site_count} "
        f"top_state_resolved_charge_count={ledger_row.top_state_resolved_charge_count} "
        f"m_K={ledger_row.association_required_multiplier:.6f} "
        f"DeltaG_K_req_kJ_mol={ledger_row.association_required_deltaG_kJ_mol:.6f} "
        f"m_D={ledger_row.base_mobility_required_multiplier:.6f} "
        f"DeltaG_D_req_kJ_mol={ledger_row.base_mobility_required_deltaG_kJ_mol:.6f} "
        f"H_state={ledger_row.H_state:.6f} "
        f"D_jump={ledger_row.D_jump_m2_s:.6e} "
        f"H_jump={ledger_row.H_jump:.6f} "
        f"eta_rel={ledger_row.eta_rel:.6f} "
        f"phi_ion={ledger_row.ionic_occupied_volume_fraction:.6f} "
        f"f_crowd={ledger_row.crowding_factor:.6f} "
        f"D_uncorr_no_crowding={ledger_row.D_uncorr_no_crowding_m2_s:.6e} "
        f"D_uncorr_with_crowding={ledger_row.D_uncorr_with_crowding_m2_s:.6e} "
        f"shape_factor={_format_float_map(ledger_row.anion_shape_factor_by_feature)} "
        f"xi_cation={ledger_row.cation_microviscosity_coupling_exponent:.6f} "
        f"xi_anion={_format_float_map(ledger_row.anion_microviscosity_coupling_exponent_by_feature)} "
        f"carrier_strength_Li={ledger_row.carrier_strength_Li_mS_cm:.6f} "
        f"carrier_strength_anion={_format_float_map(ledger_row.carrier_strength_anion_by_feature_mS_cm)} "
        f"p_free={_motif_population_value(ledger_row, 'free_or_cage'):.6f} "
        f"p_CIP={_motif_population_value(ledger_row, 'CIP_total'):.6f} "
        f"p_AGG={_motif_population_value(ledger_row, 'aggregate'):.6f} "
        f"p_bridge_network={_motif_population_value(ledger_row, 'bridge_network'):.6f} "
        f"K_SSIP={_format_float_map(ledger_row.ssip_association_constant_by_feature_M_inv)} "
        f"K_CIP={_format_float_map(ledger_row.cip_association_constant_by_feature_M_inv)} "
        f"poisson_ratio={ledger_row.poisson_correction_ratio:.6f} "
        f"basis_adjustments={','.join(ledger_row.basis_adjustments)}"
    )
    if ledger_row.top_edge_contributions:
        edge = ledger_row.top_edge_contributions[0]
        print(
            f"  top_edge={edge.source_state}->{edge.target_state} "
            f"contribution_m2_s={edge.contribution_m2_s:.6e} "
            f"rate_s_inv={edge.rate_s_inv:.6e} "
            f"raw_delta2_m2={edge.raw_delta2_m2:.6e} "
            f"corrected_delta2_m2={edge.corrected_delta2_m2:.6e}"
        )
    if ledger_row.top_state_contributions:
        state = ledger_row.top_state_contributions[0]
        print(
            f"  top_state={state.state} "
            f"motif={state.motif} "
            f"top_state_kind={state.motif_kind} "
            f"state_concentration_M={state.state_concentration_M:.6e} "
            f"stoichiometry={_format_float_map(state.stoichiometry)} "
            f"contribution_m2_s={state.contribution_m2_s:.6e} "
            f"D_NE_alpha={state.D_NE_alpha_m2_s:.6e} "
            f"D_after_binding_alpha={state.D_after_binding_alpha_m2_s:.6e} "
            f"D_none_alpha={state.D_none_alpha_m2_s:.6e} "
            f"D_ep_alpha={state.D_ep_alpha_m2_s:.6e} "
            f"D_rel_alpha={state.D_rel_alpha_m2_s:.6e} "
            f"D_rel_Li_alpha={state.D_rel_Li_alpha_m2_s:.6e} "
            f"D_rel_anion_alpha={state.D_rel_anion_alpha_m2_s:.6e} "
            f"D_rel_diag_alpha={state.D_rel_diag_alpha_m2_s:.6e} "
            f"D_rel_full_alpha={state.D_rel_full_alpha_m2_s:.6e} "
            f"D_full_alpha={state.D_full_alpha_m2_s:.6e} "
            f"d_alpha={state.d_alpha_m2_s:.6e} "
            f"H_binding_alpha={state.H_binding_alpha:.6f} "
            f"H_atmosphere_alpha={state.H_atmosphere_alpha:.6f} "
            f"H_ep_alpha={state.H_ep_alpha:.6f} "
            f"H_rel_alpha={state.H_rel_alpha:.6f} "
            f"H_rel_Li_alpha={state.H_rel_Li_alpha:.6f} "
            f"H_rel_anion_alpha={state.H_rel_anion_alpha:.6f} "
            f"H_rel_diag_alpha={state.H_rel_diag_alpha:.6f} "
            f"H_rel_cross_alpha={state.H_rel_cross_alpha:.6f} "
            f"H_rel_before_gate_alpha={state.H_rel_before_gate_alpha:.6f} "
            f"H_rel_after_gate_alpha={state.H_rel_after_gate_alpha:.6f} "
            f"H_full_alpha={state.H_full_alpha:.6f} "
            f"drag_ep_alpha={state.drag_ep_alpha:.6f} "
            f"drag_rel_alpha={state.drag_rel_alpha:.6f} "
            f"drag_rel_Li_alpha={state.drag_rel_Li_alpha:.6f} "
            f"drag_rel_anion_alpha={state.drag_rel_anion_alpha:.6f} "
            f"drag_rel_diag_alpha={state.drag_rel_diag_alpha:.6f} "
            f"drag_rel_cross_alpha={state.drag_rel_cross_alpha:.6f} "
            f"drag_rel_before_gate_alpha={state.drag_rel_before_gate_alpha:.6f} "
            f"drag_rel_after_gate_alpha={state.drag_rel_after_gate_alpha:.6f} "
            f"drag_full_alpha={state.drag_full_alpha:.6f} "
            f"H_alpha={state.H_alpha:.6f} "
            f"constraint_tau_s={state.constraint_tau_s:.6e} "
            f"constraint_length_m={state.constraint_length_m:.6e} "
            f"constraint_mu={state.constraint_mu:.6e} "
            f"R_local_trace={state.local_resistance_trace_kg_s:.6e} "
            f"R_binding_trace={state.binding_resistance_trace_kg_s:.6e} "
            f"R_atmosphere_trace={state.atmosphere_resistance_trace_kg_s:.6e} "
            f"R_ep_trace={state.electrophoretic_resistance_trace_kg_s:.6e} "
            f"R_rel_trace={state.relaxation_resistance_trace_kg_s:.6e} "
            f"R_rel_Li_trace={state.relaxation_Li_resistance_trace_kg_s:.6e} "
            f"R_rel_anion_trace={state.relaxation_anion_resistance_trace_kg_s:.6e} "
            f"R_rel_diag_trace={state.relaxation_diag_resistance_trace_kg_s:.6e} "
            f"R_rel_cross_offdiag_norm={state.relaxation_cross_resistance_offdiag_norm_kg_s:.6e} "
            f"R_rel_before_gate={state.relaxation_resistance_before_gate_trace_kg_s:.6e} "
            f"R_rel_after_gate={state.relaxation_resistance_after_gate_trace_kg_s:.6e} "
            f"R_atmosphere_single_ion_trace={state.single_ion_atmosphere_trace_kg_s:.6e} "
            f"R_atmosphere_form_factor_trace={state.form_factor_atmosphere_trace_kg_s:.6e} "
            f"state_tau_s={state.atmosphere_state_lifetime_s:.6e} "
            f"tau_atm_s={state.atmosphere_relaxation_time_s:.6e} "
            f"atmosphere_lifetime_gate={state.atmosphere_lifetime_gate:.6f} "
            f"atmosphere_diagnostic_lifetime_gate={state.atmosphere_diagnostic_lifetime_gate:.6f} "
            f"relaxation_dynamic_response={state.relaxation_dynamic_response} "
            f"anion_diagonal_relaxation_form_factor={state.anion_diagonal_relaxation_form_factor} "
            f"relaxation_gate={state.relaxation_lifetime_gate:.6f} "
            f"g_anion_diag_required={state.g_anion_diag_required:.6f} "
            f"g_anion_diag_current={state.g_anion_diag_current:.6f} "
            f"g_anion_diag_required_status={state.g_anion_diag_required_status} "
            f"anion_charge_cloud_radius_required_A="
            f"{state.anion_charge_cloud_radius_required_A:.6f} "
            f"hydrodynamic_radius_A={state.hydrodynamic_radius_A:.6f} "
            f"anion_shape_factor={state.shape_factor:.6f} "
            f"current_self_form_factor={state.current_self_form_factor:.6f} "
            f"charge_cloud_radius_available={state.charge_cloud_radius_available} "
            f"charge_cloud_radius_A={state.charge_cloud_radius_A:.6f} "
            f"charge_cloud_source={state.charge_cloud_source} "
            f"charge_cloud_site_count={state.charge_cloud_site_count} "
            f"raw_form_factor={state.raw_atmosphere_form_factor:.6f} "
            f"effective_form_factor={state.effective_atmosphere_form_factor:.6f} "
            f"R_atmosphere_before_gate={state.atmosphere_resistance_before_lifetime_gate_trace_kg_s:.6e} "
            f"R_atmosphere_after_gate={state.atmosphere_resistance_after_lifetime_gate_trace_kg_s:.6e} "
            f"R_atmosphere_offdiag_norm={state.atmosphere_offdiag_norm_kg_s:.6e} "
            f"R_ep_offdiag_norm={state.electrophoretic_offdiag_norm_kg_s:.6e} "
            f"R_rel_offdiag_norm={state.relaxation_offdiag_norm_kg_s:.6e} "
            f"R_atmosphere_min_eig={state.atmosphere_min_eig_kg_s:.6e} "
            f"R_atmosphere_max_eig={state.atmosphere_max_eig_kg_s:.6e} "
            f"atmosphere_bath_basis={state.atmosphere_bath_basis} "
            f"ionic_strength_total={state.ionic_strength_total_mol_m3:.6f} "
            f"ionic_strength_external={state.ionic_strength_external_mol_m3:.6f} "
            f"external_over_total_ionic_strength={state.external_over_total_ionic_strength:.6f} "
            f"resolved_charge_center_count={state.resolved_charge_center_count} "
            f"anion_feature_id={state.anion_feature_id} "
            f"local_D_Li={state.local_D_Li_m2_s:.6e} "
            f"local_D_anion={state.local_D_anion_m2_s:.6e} "
            f"kappa_radius_Li={state.kappa_radius_Li:.6f} "
            f"kappa_radius_anion={state.kappa_radius_anion:.6f} "
            f"debye_kappa_inv_A={state.debye_kappa_inv_A:.6f} "
            f"separation_over_debye={state.separation_over_debye:.6f} "
            f"mean_center_separation_A={state.mean_charge_center_separation_A:.6f} "
            f"form_factor_cancellation={state.atmosphere_form_factor_cancellation:.6f} "
            f"thermodynamic_factor_trace={state.thermodynamic_factor_trace:.6f} "
            f"thermodynamic_factor_eigs={_format_float_sequence(state.thermodynamic_factor_eigenvalues)} "
            f"structure_factor_charge_mode={state.structure_factor_charge_mode:.6f} "
            f"pi={state.stationary_probability:.6e} "
            f"charge={state.charge:.6f}"
        )


def _motif_population_value(ledger_row: BiasLedgerRow, motif_key: str) -> float:
    if motif_key in ledger_row.motif_populations:
        return ledger_row.motif_populations[motif_key]
    return 0.0


def _format_float_map(values: dict[str, float]) -> str:
    if not values:
        return "none"
    return ",".join(f"{key}:{values[key]:.6f}" for key in sorted(values))


def _format_int_map(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ",".join(f"{key}:{values[key]}" for key in sorted(values))


def _format_float_sequence(values: tuple[float, ...]) -> str:
    if not values:
        return "none"
    return ",".join(f"{value:.6f}" for value in values)


if __name__ == "__main__":
    main()
