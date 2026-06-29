"""Run the property-label Mori regression baseline audit."""

from __future__ import annotations

from constants import T_REF_K
from conductivity.baselines.property_label_mori_regression_baseline import (
    DEFAULT_LATENT_MORI_FOLD_COUNT,
    DEFAULT_LATENT_MORI_RIDGE_PENALTY,
    LatentMoriSurrogateAuditResult,
    LatentMoriSurrogateMetrics,
    LatentMoriSurrogateRow,
    audit_latent_mori_surrogate_against_property_db,
)
from conductivity.run_finite_markov_dataset_audit import TOP_REPORTED_ROWS
from data.electrolyte_property_db import DATA


def main() -> None:
    audit = audit_latent_mori_surrogate_against_property_db(
        DATA,
        T_REF_K,
        DEFAULT_LATENT_MORI_RIDGE_PENALTY,
        DEFAULT_LATENT_MORI_FOLD_COUNT,
    )

    print("property_label_mori_regression_baseline_audit")
    _print_summary(audit)
    print("worst_cross_validated_rows")
    for row in sorted(
        audit.rows,
        key=lambda audit_row: abs(audit_row.cross_validated_residual_mS_cm),
        reverse=True,
    )[:TOP_REPORTED_ROWS]:
        _print_row(row)


def _print_summary(audit: LatentMoriSurrogateAuditResult) -> None:
    print(f"model_source={audit.parameters.model_source}")
    print(f"gauge={audit.parameters.gauge}")
    print(f"ridge_penalty={audit.parameters.ridge_penalty:.6e}")
    print(f"fold_count={DEFAULT_LATENT_MORI_FOLD_COUNT}")
    print(f"labeled_rows={audit.labeled_rows}")
    print(f"evaluated_rows={audit.evaluated_rows}")
    print(f"failed_rows={audit.failed_rows}")
    _print_metrics("full_fit", audit.full_fit_metrics)
    _print_metrics("cross_validated", audit.cross_validated_metrics)


def _print_metrics(
    label: str,
    metrics: LatentMoriSurrogateMetrics,
) -> None:
    print(
        f"{label}_count={metrics.count} "
        f"{label}_mae_mS_cm={metrics.mae_mS_cm:.6f} "
        f"{label}_rmse_mS_cm={metrics.rmse_mS_cm:.6f} "
        f"{label}_bias_mS_cm={metrics.bias_mS_cm:.6f} "
        f"{label}_mape_percent={metrics.mape_percent:.6f} "
        f"{label}_r2={metrics.r2:.6f} "
        f"{label}_pearson_r={metrics.pearson_r:.6f}"
    )


def _print_row(row: LatentMoriSurrogateRow) -> None:
    print(
        f"row_id={row.row_id} "
        f"sigma_empirical={row.empirical_sigma_mS_cm:.6f} "
        f"sigma_full_fit={row.full_fit_sigma_mS_cm:.6f} "
        f"sigma_cross_validated={row.cross_validated_sigma_mS_cm:.6f} "
        f"residual_full_fit={row.full_fit_residual_mS_cm:.6f} "
        f"residual_cross_validated={row.cross_validated_residual_mS_cm:.6f} "
        f"basis_dim={row.projected_basis_dimension} "
        f"mode_contributions={','.join(f'{value:.6f}' for value in row.mode_contribution_mS_cm)}"
    )


if __name__ == "__main__":
    main()
