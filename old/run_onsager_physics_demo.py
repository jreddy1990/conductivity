"""Run the correlated-carrier conductivity prototype on audit recipes."""

from __future__ import annotations

import math

import numpy as np

from constants import T_REF_K
from data.electrolyte_property_db import DATA
from conductivity.onsager_physics_sigma import (
    evaluate_onsager_conductivity,
    fit_mechanism_params,
    fit_global_mobility_scale,
)


def main() -> None:
    calibration_entries, rejected = strict_calibration_entries(DATA)
    scalar_calibration = fit_global_mobility_scale(calibration_entries, temperature_K=T_REF_K)
    calibration = fit_mechanism_params(calibration_entries, temperature_K=T_REF_K)
    print(
        "calibration: "
        f"rows={calibration.n_rows} "
        f"rejected_strict_rows={len(rejected)} "
        f"mobility_scale={calibration.params.mobility_scale:.6f} "
        f"MAE={calibration.mae_mS_cm:.3f} mS/cm "
        f"RMSE={calibration.rmse_mS_cm:.3f} mS/cm"
    )
    print(
        "scalar-only baseline: "
        f"mobility_scale={scalar_calibration.params.mobility_scale:.6f} "
        f"MAE={scalar_calibration.mae_mS_cm:.3f} mS/cm "
        f"RMSE={scalar_calibration.rmse_mS_cm:.3f} mS/cm"
    )
    print(
        "mechanism params: "
        f"bjerrum_scale={calibration.params.bjerrum_dielectric_scale:.4f}, "
        f"visc_exp_scale={calibration.params.viscosity_exponent_scale:.4f}, "
        f"liquid_excess_visc_scale={calibration.params.liquid_excess_viscosity_scale:.4f}, "
        f"dimer_visc_scale={calibration.params.dimer_viscosity_scale:.4f}, "
        f"salt_visc_scale={calibration.params.salt_viscosity_scale:.4f}, "
        f"pair_gain={calibration.params.pair_correlation_gain:.4f}, "
        f"steric_gain={calibration.params.steric_anticorrelation_gain:.4f}, "
        f"aggregate_gain={calibration.params.aggregate_correlation_gain:.4f}"
    )
    if rejected:
        for idx, reason in rejected[:5]:
            print(f"rejected[{idx}]: {reason}")

    base = {
        "solvents": {"EC": 0.30, "DMC": 0.70},
        "salts": {"LiPF6": 1.0},
        "additives": {},
    }
    ttfp = {
        "solvents": {"EC": 0.30, "DMC": 0.70},
        "salts": {"LiPF6": 1.0},
        "additives": {"TTFP": 0.10},
    }

    print()
    print("audit cases at 298.15 K")
    print_result("base EC:DMC=30:70 v/v, LiPF6=1.0 M", base, calibration.params)
    print_result("TTFP=0.10 wt fraction", ttfp, calibration.params)

    base_result = evaluate_onsager_conductivity(base, T_REF_K, calibration.params)
    ttfp_result = evaluate_onsager_conductivity(ttfp, T_REF_K, calibration.params)
    if ttfp_result.sigma_mS_cm >= base_result.sigma_mS_cm:
        raise SystemExit(
            "TTFP audit failed: 10 wt% TTFP did not lower conductivity "
            f"({ttfp_result.sigma_mS_cm:.4f} >= {base_result.sigma_mS_cm:.4f} mS/cm)"
        )

    print()
    print("FEC loading sweep, EC:DMC=30:70 v/v, LiPF6=1.0 M")
    for loading in [0.0, 0.025, 0.05, 0.075, 0.10]:
        recipe = {
            "solvents": {"EC": 0.30, "DMC": 0.70},
            "salts": {"LiPF6": 1.0},
            "additives": {} if loading == 0.0 else {"FEC": loading},
        }
        result = evaluate_onsager_conductivity(recipe, T_REF_K, calibration.params)
        shell = _compact_shell(result.solvation.shell_fractions)
        rho = _compact_rho(result.correlation.li_anion_rho)
        free = _mean_free_fraction(result)
        print(
            f"FEC={loading:.3f} wt: "
            f"sigma={result.sigma_mS_cm:.4f} mS/cm, "
            f"sigma_NE={result.sigma_uncorrelated_mS_cm:.4f}, "
            f"eta={result.matrix.eta_solution_cP:.4f} cP, "
            f"eps_eff={result.matrix.epsilon_effective:.3f}, "
            f"free={free:.4f}, "
            f"rho={rho}, "
            f"shell={shell}"
        )

    eig_min = float(np.linalg.eigvalsh(base_result.correlation.matrix).min())
    print()
    print(f"base correlation PSD check: min_eigenvalue={eig_min:.8f}")


def strict_calibration_entries(entries: list[dict]) -> tuple[list[dict], list[tuple[int, str]]]:
    accepted = []
    rejected = []
    for idx, entry in enumerate(entries):
        try:
            evaluate_onsager_conductivity(entry["recipe"], T_REF_K)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append((idx, str(exc).splitlines()[0]))
        else:
            accepted.append(entry)
    if not accepted:
        raise ValueError("No empirical conductivity rows satisfy the strict recipe/model contract")
    return accepted, rejected


def print_result(label: str, recipe: dict, params) -> None:
    result = evaluate_onsager_conductivity(recipe, T_REF_K, params)
    print(
        f"{label}: "
        f"sigma={result.sigma_mS_cm:.4f} mS/cm, "
        f"sigma_NE={result.sigma_uncorrelated_mS_cm:.4f}, "
        f"eta={result.matrix.eta_solution_cP:.4f} cP, "
        f"eta_liq={result.matrix.eta_liquid_cP:.4f} cP, "
        f"eps_liq={result.matrix.epsilon_liquid:.3f}, "
        f"eps_eff={result.matrix.epsilon_effective:.3f}, "
        f"free={_mean_free_fraction(result):.4f}, "
        f"rho={_compact_rho(result.correlation.li_anion_rho)}, "
        f"shell={_compact_shell(result.solvation.shell_fractions)}"
    )


def _mean_free_fraction(result) -> float:
    total = sum(result.composition.ionic_source_molarities_M.values())
    free_lithium_allocation = math.fsum(
        concentration_M
        for motif_label, concentration_M in result.speciation.motif_concentrations_M.items()
        if motif_label.startswith("free_cation:")
    )
    return free_lithium_allocation / total


def _compact_shell(shell: dict[str, float]) -> str:
    return "{" + ", ".join(f"{name}:{value:.3f}" for name, value in sorted(shell.items())) + "}"


def _compact_rho(rho: dict[str, float]) -> str:
    return "{" + ", ".join(f"{name}:{value:.3f}" for name, value in sorted(rho.items())) + "}"


if __name__ == "__main__":
    main()
