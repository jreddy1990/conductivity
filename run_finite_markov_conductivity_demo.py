"""Run finite Markov-additive conductivity audits on benchmark recipes."""

from __future__ import annotations

from dataclasses import dataclass

from constants import T_REF_K
from conductivity.finite_markov_conductivity import (
    FiniteMarkovConductivityResult,
    TransitionAuditRow,
    evaluate_finite_markov_conductivity,
)


RecipeDict = dict[str, dict[str, float]]

EC_VOLUME_PARTS = 3.0
DMC_VOLUME_PARTS = 7.0  # Explicit constant: user-requested EC:DMC 30:70 v/v benchmark.
BASE_EC_VOLUME_FRACTION = EC_VOLUME_PARTS / (EC_VOLUME_PARTS + DMC_VOLUME_PARTS)
BASE_DMC_VOLUME_FRACTION = DMC_VOLUME_PARTS / (EC_VOLUME_PARTS + DMC_VOLUME_PARTS)
BASE_SALT_MOLARITY_M = 1.0
DUAL_LIPF6_MOLARITY_PARTS = 2.0
DUAL_LIFSI_MOLARITY_PARTS = 3.0
DUAL_SALT_TOTAL_MOLARITY_M = 1.0
DUAL_LIPF6_MOLARITY_M = (
    DUAL_SALT_TOTAL_MOLARITY_M
    * DUAL_LIPF6_MOLARITY_PARTS
    / (DUAL_LIPF6_MOLARITY_PARTS + DUAL_LIFSI_MOLARITY_PARTS)
)
DUAL_LIFSI_MOLARITY_M = (
    DUAL_SALT_TOTAL_MOLARITY_M
    * DUAL_LIFSI_MOLARITY_PARTS
    / (DUAL_LIPF6_MOLARITY_PARTS + DUAL_LIFSI_MOLARITY_PARTS)
)
FEC_MAX_WEIGHT_FRACTION = 1.0 / 10.0  # Explicit constant: requested 10 wt fraction FEC sweep endpoint.
FEC_SWEEP_INTERVALS = 4  # Explicit constant: requested five-point inclusive sweep from zero to endpoint.
FEC_SWEEP_WEIGHT_FRACTIONS = tuple(
    FEC_MAX_WEIGHT_FRACTION * sweep_index / FEC_SWEEP_INTERVALS
    for sweep_index in range(FEC_SWEEP_INTERVALS + 1)
)
TOP_MOTIF_COUNT = 6  # Explicit constant: compact audit table row count for motif populations.
TOP_RISK_ROW_COUNT = TOP_MOTIF_COUNT - 1


@dataclass(frozen=True)
class DemoCase:
    label: str
    recipe: RecipeDict


def main() -> None:
    print(f"finite Markov conductivity demo at T={T_REF_K:.2f} K")
    print("evaluation path: recipe -> transport kernel -> motif generator -> Poisson readout")
    for demo_case in _demo_cases():
        print()
        print(demo_case.label)
        print(_format_recipe(demo_case.recipe))
        result = evaluate_finite_markov_conductivity(demo_case.recipe, T_REF_K)
        _print_result(result)


def _demo_cases() -> tuple[DemoCase, ...]:
    cases = [
        DemoCase(
            label="baseline EC:DMC/LiPF6",
            recipe={
                "solvents": {"EC": BASE_EC_VOLUME_FRACTION, "DMC": BASE_DMC_VOLUME_FRACTION},
                "salts": {"LiPF6": BASE_SALT_MOLARITY_M},
                "additives": {},
            },
        )
    ]
    for fec_weight_fraction in FEC_SWEEP_WEIGHT_FRACTIONS:
        additives: dict[str, float] = {}
        if fec_weight_fraction > 0.0:
            additives["FEC"] = fec_weight_fraction
        cases.append(
            DemoCase(
                label=f"FEC sweep loading={fec_weight_fraction:.3f} wt fraction",
                recipe={
                    "solvents": {"EC": BASE_EC_VOLUME_FRACTION, "DMC": BASE_DMC_VOLUME_FRACTION},
                    "salts": {"LiPF6": BASE_SALT_MOLARITY_M},
                    "additives": additives,
                },
            )
        )
    cases.append(
        DemoCase(
            label="dual-salt EC:DMC/LiPF6/LiFSI",
            recipe={
                "solvents": {"EC": BASE_EC_VOLUME_FRACTION, "DMC": BASE_DMC_VOLUME_FRACTION},
                "salts": {
                    "LiPF6": DUAL_LIPF6_MOLARITY_M,
                    "LiFSI": DUAL_LIFSI_MOLARITY_M,
                },
                "additives": {},
            },
        )
    )
    return tuple(cases)


def _print_result(result: FiniteMarkovConductivityResult) -> None:
    model = result.generated_model
    if model is None:
        raise ValueError("Demo requires a recipe-generated finite Markov model")
    audit = model.mixture_audit
    print(
        "transport: "
        f"sigma={result.sigma_mS_cm:.6f} mS/cm, "
        f"D_Q={result.D_Q_m2_s:.6e} m^2/s, "
        f"D_v={result.vehicular_D_Q_m2_s:.6e} m^2/s, "
        f"D_jump={result.jump_D_Q_m2_s:.6e} m^2/s"
    )
    print(
        "kernel: "
        f"eta={audit.viscosity_cP:.6f} cP, "
        f"eps_bruggeman={audit.dielectric_bruggeman:.6f}, "
        f"eps_effective={audit.dielectric_effective:.6f}, "
        f"kappa_inv={audit.debye_kappa_inv_m:.6e} m, "
        f"C_cation={audit.cation_concentration_mol_m3:.6f} mol/m^3"
    )
    print(
        "residuals: "
        f"row_sum={result.row_sum_residual_s_inv:.6e} 1/s, "
        f"stationary={result.stationary_residual_s_inv:.6e} 1/s, "
        f"detailed_balance={result.detailed_balance_residual_s_inv:.6e} 1/s, "
        f"capacity={result.capacity_evaluation}"
    )
    print(f"free fractions: {_format_float_mapping(audit.free_fraction_by_feature)}")
    print(f"paired fractions: {_format_float_mapping(audit.paired_fraction_by_feature)}")
    print(f"aggregate fractions: {_format_float_mapping(audit.aggregate_fraction_by_feature)}")
    print(f"shell fractions: {_format_float_mapping(audit.shell_fractions)}")
    print(f"motif probabilities: {_format_top_motifs(model.chemical_motif_populations)}")
    print("co-motion risks:")
    for row in _top_comotion_risks(model.transition_audit):
        print(
            "  "
            f"{row.source_state}->{row.target_state}: "
            f"k={row.rate_s_inv:.6e} 1/s, "
            f"z_eff={row.effective_charge:.6f}, "
            f"|delta|={_displacement_norm(row):.6e} m"
        )


def _top_comotion_risks(
    transition_audit: tuple[TransitionAuditRow, ...],
) -> tuple[TransitionAuditRow, ...]:
    ranked = sorted(
        transition_audit,
        key=lambda row: row.rate_s_inv * (1.0 - min(abs(row.effective_charge), 1.0)),
        reverse=True,
    )
    return tuple(ranked[:TOP_RISK_ROW_COUNT])


def _format_recipe(recipe: RecipeDict) -> str:
    return (
        "recipe units: "
        f"solvents v/v={_format_float_mapping(recipe['solvents'])}, "
        f"salts M={_format_float_mapping(recipe['salts'])}, "
        f"additives wt fraction={_format_float_mapping(recipe['additives'])}"
    )


def _format_float_mapping(values: dict[str, float]) -> str:
    if not values:
        return "{}"
    return "{" + ", ".join(f"{name}:{value:.6f}" for name, value in sorted(values.items())) + "}"


def _format_top_motifs(values: dict[str, float]) -> str:
    ranked_items = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return "{" + ", ".join(
        f"{name}:{value:.6f}" for name, value in ranked_items[:TOP_MOTIF_COUNT]
    ) + "}"


def _displacement_norm(row: TransitionAuditRow) -> float:
    displacement = row.charge_displacement_m
    return (
        displacement[0] * displacement[0]
        + displacement[1] * displacement[1]
        + displacement[2] * displacement[2]
    ) ** 0.5


if __name__ == "__main__":
    main()
