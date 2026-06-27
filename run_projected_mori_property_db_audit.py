"""Run latent Mori conductivity readout directly against empirical labels."""

from __future__ import annotations

from constants import T_REF_K
from conductivity.finite_markov_conductivity import (
    ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF,
    ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
    RELAXATION_DYNAMIC_RESPONSE_OFF,
)
from conductivity.run_finite_markov_dataset_audit import TOP_REPORTED_ROWS
from conductivity.projected_mori_property_db_audit import (
    ProjectedMoriPropertyDbAuditResult,
    ProjectedMoriPropertyDbRow,
    audit_projected_mori_conductivity_against_property_db,
)
from data.electrolyte_property_db import DATA


def main() -> None:
    audit = audit_projected_mori_conductivity_against_property_db(
        DATA,
        T_REF_K,
        ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
        RELAXATION_DYNAMIC_RESPONSE_OFF,
        ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF,
    )

    print("projected_mori_property_db_audit")
    _print_summary(audit)
    print("worst_projected_mori_rows")
    for row in sorted(
        audit.rows,
        key=lambda audit_row: abs(audit_row.residual_mS_cm),
        reverse=True,
    )[:TOP_REPORTED_ROWS]:
        _print_row(row)


def _print_summary(audit: ProjectedMoriPropertyDbAuditResult) -> None:
    print(f"labeled_rows={audit.labeled_rows}")
    print(f"evaluated_rows={audit.evaluated_rows}")
    print(f"failed_rows={audit.failed_rows}")
    print(f"mae_mS_cm={audit.mae_mS_cm:.6f}")
    print(f"rmse_mS_cm={audit.rmse_mS_cm:.6f}")
    print(f"bias_mS_cm={audit.bias_mS_cm:.6f}")
    print(f"mape_percent={audit.mape_percent:.6f}")
    print(f"r2={audit.r2:.6f}")
    print(f"pearson_r={audit.pearson_r:.6f}")


def _print_row(row: ProjectedMoriPropertyDbRow) -> None:
    print(
        f"row_id={row.row_id} "
        f"sigma_empirical={row.empirical_sigma_mS_cm:.6f} "
        f"sigma_projected_mori={row.projected_mori_sigma_mS_cm:.6f} "
        f"residual={row.residual_mS_cm:.6f} "
        f"basis_dim={row.projected_basis_dimension} "
        f"energy_min={row.energy_min_eigenvalue:.6e} "
        f"energy_max={row.energy_max_eigenvalue:.6e} "
        f"axis_sigma_S_m={','.join(f'{axis_value:.6e}' for axis_value in row.axis_conductivity_S_m)} "
        f"quadratic_forms={','.join(f'{axis_value:.6e}' for axis_value in row.quadratic_form_by_axis)}"
    )


if __name__ == "__main__":
    main()
