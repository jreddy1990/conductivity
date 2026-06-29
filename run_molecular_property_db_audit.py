"""Run the molecular descriptor-neutral conductivity audit on the property DB."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from data.electrolyte_property_db import DATA
from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
from conductivity.fit_conductivity_primitive_parameters import (
    load_primitive_parameters_from_candidate_artifact,
    load_primitive_parameters_from_promoted_candidate_artifact,
)
from conductivity.molecular_primitive_parameters import (
    ConductivityPrimitiveParameterSet,
)
from conductivity.molecular_property_db_audit import (
    MolecularClusterSensitivityDiagnostic,
    MolecularClusterThermodynamicDiagnostic,
    MolecularPropertyDbRegistrySource,
    MolecularPropertyDbRowResult,
    audit_molecular_property_db_cases,
    build_molecular_property_db_case_selection,
    cluster_sensitivity_diagnostics_for_row,
    configured_conductivity_primitive_parameters,
    default_molecular_primitive_fit_configuration,
    default_molecular_property_db_audit_options,
    validate_molecular_property_db_audit_result,
)

CONFIGURED_BASELINE_CONFIG_NAME = "configured_baseline"
FITTED_CANDIDATE_CONFIG_NAME = "fitted_candidate"


def main() -> None:
    runner_arguments = _parse_runner_arguments()
    options = default_molecular_property_db_audit_options()
    fit_options, _log_bounds = default_molecular_primitive_fit_configuration()
    registry_source = MolecularPropertyDbRegistrySource(
        solvent_registry=SOLVENTS,
        salt_registry=SALTS,
        additive_registry=ADDITIVES,
        cation_registry=CATION_PROPERTIES,
    )
    case_selection = build_molecular_property_db_case_selection(
        tuple(DATA),
        registry_source,
        options,
    )
    primitive_parameters = load_primitive_parameters_for_audit_config(
        runner_arguments.config,
        fit_options.candidate_output_path,
    )
    result = audit_molecular_property_db_cases(
        case_selection.cases,
        primitive_parameters,
        options,
    )
    validate_molecular_property_db_audit_result(result, fit_options)
    print("molecular_property_db_audit")
    print(f"config={runner_arguments.config}")
    print(f"source_labeled_rows={case_selection.source_labeled_rows}")
    print(f"formulation_group_count={len(case_selection.formulation_groups)}")
    print(f"labeled_rows={result.labeled_rows}")
    print(f"evaluated_rows={result.evaluated_rows}")
    print(f"failed_rows={result.failed_rows}")
    print(f"mae_mS_cm={result.mae_mS_cm:.6f}")
    print(f"rmse_mS_cm={result.rmse_mS_cm:.6f}")
    print(f"bias_mS_cm={result.bias_mS_cm:.6f}")
    print(f"pearson_r={result.pearson_r:.6f}")
    print(f"maximum_abs_residual_mS_cm={result.maximum_abs_residual_mS_cm:.6f}")
    print(f"maximum_mass_balance_residual={result.maximum_mass_balance_residual:.6e}")
    print(f"maximum_row_sum_residual={result.maximum_row_sum_residual:.6e}")
    print(f"maximum_stationary_residual={result.maximum_stationary_residual:.6e}")
    print(
        "maximum_detailed_balance_residual="
        f"{result.maximum_detailed_balance_residual:.6e}"
    )
    print(
        "maximum_event_reversal_residual="
        f"{result.maximum_event_reversal_residual:.6e}"
    )
    print(f"zero_charge_sigma_mS_cm={result.zero_charge_sigma_mS_cm:.6e}")
    print(
        "higher_viscosity_lowers_dilute_conductivity="
        f"{result.higher_viscosity_lowers_dilute_conductivity}"
    )
    print(
        "higher_packing_lowers_local_mobility="
        f"{result.higher_packing_lowers_local_mobility}"
    )
    multi_row_formulation_groups = tuple(
        formulation_group for formulation_group in case_selection.formulation_groups
        if len(formulation_group.source_row_ids) > 1
    )
    if multi_row_formulation_groups:
        print("formulation_group_audit")
        for formulation_group in multi_row_formulation_groups:
            print(
                "representative_row_id={row_id} source_row_ids={source_rows} "
                "target_median={target:.6f} empirical_spread={spread:.6f} "
                "empirical_values={values} solvents={solvents} salts={salts} "
                "additives={additives}".format(
                    row_id=formulation_group.representative_row_id,
                    source_rows=formulation_group.source_row_ids,
                    target=formulation_group.target_sigma_mS_cm,
                    spread=formulation_group.empirical_sigma_spread_mS_cm,
                    values=formulation_group.empirical_sigmas_mS_cm,
                    solvents=dict(formulation_group.solvent_loadings),
                    salts=dict(formulation_group.salt_loadings_M),
                    additives=dict(formulation_group.additive_loadings),
                )
            )
    print("worst_rows")
    case_by_row_id = {
        molecular_case.row_id: molecular_case
        for molecular_case in case_selection.cases
    }
    for row_result in sorted(
        result.rows,
        key=lambda row: abs(row.residual_mS_cm),
        reverse=True,
    )[:options.audit_worst_row_count]:
        print(
            "row_id={row_id} source_row_ids={source_rows} "
            "empirical_target={empirical:.6f} empirical_spread={spread:.6f} "
            "predicted={predicted:.6f} "
            "residual={residual:.6f} direct={direct:.6f} corrector={corrector:.6f} "
            "free_ion_fraction={free_fraction:.6f} "
            "charged_cluster_fraction={charged_fraction:.6f} "
            "neutral_cluster_fraction={neutral_fraction:.6f} "
            "charged_cluster_direct={charged_direct:.6f} "
            "charged_cluster_corrector={charged_corrector:.6f} "
            "charged_cluster_net={charged_net:.6f} "
            "failed={failed} reason={reason}".format(
                row_id=row_result.row_id,
                source_rows=row_result.source_row_ids,
                empirical=row_result.empirical_sigma_mS_cm,
                spread=row_result.empirical_sigma_spread_mS_cm,
                predicted=row_result.predicted_sigma_mS_cm,
                residual=row_result.residual_mS_cm,
                direct=row_result.direct_sigma_mS_cm,
                corrector=row_result.corrector_sigma_mS_cm,
                free_fraction=row_result.free_ion_fraction,
                charged_fraction=row_result.charged_cluster_fraction,
                neutral_fraction=row_result.neutral_cluster_fraction,
                charged_direct=row_result.charged_cluster_direct_sigma_mS_cm,
                charged_corrector=(
                    row_result.charged_cluster_corrector_sigma_mS_cm
                ),
                charged_net=row_result.charged_cluster_net_sigma_mS_cm,
                failed=row_result.failed,
                reason=row_result.failure_reason,
            )
        )
        _print_top_cluster_diagnostics(
            row_result,
            options.audit_worst_row_count,
        )
        _print_cluster_sensitivity_diagnostics(
            cluster_sensitivity_diagnostics_for_row(
                case_by_row_id[row_result.row_id],
                primitive_parameters,
                options,
                row_result,
            ),
            options.audit_worst_row_count,
        )


@dataclass(frozen=True)
class MolecularPropertyDbAuditRunnerArguments:
    config: str


def _parse_runner_arguments() -> MolecularPropertyDbAuditRunnerArguments:
    argument_parser = argparse.ArgumentParser(
        description="Run the molecular descriptor-neutral conductivity property-DB audit."
    )
    argument_parser.add_argument(
        "--config",
        default=CONFIGURED_BASELINE_CONFIG_NAME,
        help=(
            "Primitive parameter source. Use configured_baseline, fitted_candidate, "
            "or a path to a molecular conductivity primitive candidate artifact."
        ),
    )
    parsed_arguments = argument_parser.parse_args()
    config_argument = parsed_arguments.config
    if not isinstance(config_argument, str) or not config_argument.strip():
        raise ValueError("--config must be a nonempty string")
    return MolecularPropertyDbAuditRunnerArguments(config=config_argument)


def load_primitive_parameters_for_audit_config(
    config_argument: str,
    fitted_candidate_artifact_path: str,
) -> ConductivityPrimitiveParameterSet:
    if not isinstance(config_argument, str) or not config_argument.strip():
        raise ValueError("config_argument must be a nonempty string")
    if (
        not isinstance(fitted_candidate_artifact_path, str)
        or not fitted_candidate_artifact_path.strip()
    ):
        raise ValueError("fitted_candidate_artifact_path must be a nonempty string")
    normalized_config_argument = config_argument.strip()
    if normalized_config_argument == CONFIGURED_BASELINE_CONFIG_NAME:
        return configured_conductivity_primitive_parameters()
    if normalized_config_argument == FITTED_CANDIDATE_CONFIG_NAME:
        return load_primitive_parameters_from_promoted_candidate_artifact(
            fitted_candidate_artifact_path
        )
    return load_primitive_parameters_from_candidate_artifact(normalized_config_argument)


def _print_top_cluster_diagnostics(
    row_result: MolecularPropertyDbRowResult,
    cluster_diagnostic_count: int,
) -> None:
    top_by_concentration = sorted(
        row_result.cluster_thermodynamic_diagnostics,
        key=lambda diagnostic: diagnostic.concentration_mol_m3,
        reverse=True,
    )[:cluster_diagnostic_count]
    top_by_free_energy = sorted(
        row_result.cluster_thermodynamic_diagnostics,
        key=lambda diagnostic: diagnostic.standard_free_energy_over_RT,
    )[:cluster_diagnostic_count]
    print("  top_clusters_by_concentration")
    for diagnostic in top_by_concentration:
        _print_cluster_diagnostic(diagnostic)
    print("  top_clusters_by_most_favorable_deltaG")
    for diagnostic in top_by_free_energy:
        _print_cluster_diagnostic(diagnostic)


def _print_cluster_sensitivity_diagnostics(
    sensitivity_diagnostics: tuple[MolecularClusterSensitivityDiagnostic, ...],
    cluster_diagnostic_count: int,
) -> None:
    print("  top_clusters_by_logK_sensitivity")
    for diagnostic in sensitivity_diagnostics[:cluster_diagnostic_count]:
        print(
            "    label={label} kind={kind} net_charge={charge} "
            "baseline_concentration={concentration:.6e} "
            "baseline_deltaG_over_RT={delta_g:.6f} "
            "sigma_lower_deltaG={sigma_lower:.6f} "
            "sigma_higher_deltaG={sigma_higher:.6f} "
            "sensitivity_mS_cm_per_logK={sensitivity:.6f} "
            "direction_needed={direction}".format(
                label=diagnostic.cluster_label,
                kind=diagnostic.cluster_kind,
                charge=diagnostic.net_charge_number,
                concentration=diagnostic.baseline_concentration_mol_m3,
                delta_g=diagnostic.baseline_deltaG_over_RT,
                sigma_lower=diagnostic.sigma_lower_deltaG_mS_cm,
                sigma_higher=diagnostic.sigma_higher_deltaG_mS_cm,
                sensitivity=diagnostic.sensitivity_mS_cm_per_logK,
                direction=diagnostic.direction_needed,
            )
        )


def _print_cluster_diagnostic(
    diagnostic: MolecularClusterThermodynamicDiagnostic,
) -> None:
    print(
        "    label={label} kind={kind} net_charge={charge} "
        "concentration={concentration:.6e} fraction={fraction:.6e} "
        "deltaG_over_RT={delta_g:.6f} logK={log_k:.6f} "
        "coulomb={coulomb:.6e} desolvation={desolvation:.6e} "
        "coordination={coordination:.6e} steric={steric:.6e} "
        "entropy={entropy:.6e} activity={activity:.6e}".format(
            label=diagnostic.cluster_label,
            kind=diagnostic.cluster_kind,
            charge=diagnostic.net_charge_number,
            concentration=diagnostic.concentration_mol_m3,
            fraction=diagnostic.concentration_fraction_of_total_ion,
            delta_g=diagnostic.standard_free_energy_over_RT,
            log_k=diagnostic.log_equilibrium_constant,
            coulomb=diagnostic.coulomb_J_mol,
            desolvation=diagnostic.desolvation_J_mol,
            coordination=diagnostic.coordination_J_mol,
            steric=diagnostic.steric_J_mol,
            entropy=diagnostic.entropy_J_mol,
            activity=diagnostic.activity_correction_J_mol,
        )
    )


if __name__ == "__main__":
    main()
