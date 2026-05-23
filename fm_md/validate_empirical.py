"""Empirical validation of predict_sigma against the project's
data/electrolyte_property_db.py measurements (plan §0, §1a).

This is the ULTIMATE generalization gate. data/electrolyte_property_db.py
holds 125 (recipe, conductivity_mS_cm) rows assembled from literature
measurements on carbonate electrolytes (LiFSI/LiPF6/LiTFSI in EC/DMC/EMC
mixtures with various additives). Sigma here is experimental, not MD-derived,
so the comparison carries the structural classical-vs-experiment polarization
gap noted in [[loss3-f1-f2-f3-fixes]] (~25-50% systematic depression). That
gap is shared with BAMBOO and so cancels in the BAMBOO eval; here it remains,
which means the pass criterion is wider.

Method:
  1. Load DATA from data/electrolyte_property_db.py.
  2. Filter to rows whose salts AND solvents AND additives are all in
     SPECIES_CATALOGUE + SPECIES_SMILES.
  3. Convert each recipe to (species_list, mole_fractions) using fixed-density
     stoichiometric arithmetic on 1 L of solution.
  4. Sample --n-rows random recipes, run predict_sigma on each, compare to
     conductivity_mS_cm.
  5. Report log-MSE against a constant-mean baseline; pass criterion is
     log-MSE < baseline (which plan §1a documents the previous regression
     model fails by 24x).

Entry:
  python -m conductivity.fm_md.validate_empirical \\
      --model conductivity/fm_data/fm_md_acf_F1F2F3/propagator_final.pkl \\
      --n-rows 8 --sim-time-ps 300
"""

import control_framework.jax_m4_tuning  # noqa: F401  -- before any jax import

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from conductivity.fm_md.atomistic_io import SPECIES_CATALOGUE, _ATOMIC_MASS
from conductivity.fm_md.bamboo_mix_validation_set import SALT_TO_IONS
from conductivity.fm_md.infer import predict_sigma
from constants import T_REF_K
from data.electrolyte_property_db import DATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


# Bulk electrolyte density used for the 1 L basis. Same value as
# DEFAULT_DENSITY_G_CM3 in box_constructor.py (the Bytedance FSI box density
# at 333 K). Real electrolytes vary by ~10% across LiFSI/LiPF6, an additive
# systematic on the sigma comparison.
ELECTROLYTE_DENSITY_G_CM3 = 1.249

ALLOWED_SOLVENTS = {"EC", "EMC", "DMC", "DEC", "PC", "FEC", "VC", "DME"}
ALLOWED_ADDITIVES = {"FEC", "VC"}
ALLOWED_SALTS = set(SALT_TO_IONS.keys())   # {LiPF6, LiFSI, LiFSA}

# Default room temperature for the empirical DB rows (they do not carry T).
EMPIRICAL_DEFAULT_T_K = T_REF_K

OUTPUT_PATH = Path("conductivity/fm_data/validate_empirical_results.json")


def _species_mw(name: str) -> float:
    """Molecular weight in amu for a SPECIES_CATALOGUE entry, derived from
    atomistic_io._ATOMIC_MASS so there is one source of atomic masses."""
    sig = next(s for s in SPECIES_CATALOGUE if s.name == name)
    return float(sum(_ATOMIC_MASS[e] * c for e, c in sig.element_counts.items()))


def _salt_mw(salt_name: str) -> float:
    """MW of the SALT (cation + anion combined)."""
    cation, anion = SALT_TO_IONS[salt_name]
    return _species_mw(cation) + _species_mw(anion)


def _convert_recipe_to_composition(recipe: dict) -> tuple[list[str], list[float]] | None:
    """Convert a property-DB recipe to (species_list, mole_fractions). Returns
    None if any species is outside the catalogue; the row is skipped.

    Basis is 1 L of solution at ELECTROLYTE_DENSITY_G_CM3 with the salt
    contributing molarity * MW grams and the rest filled by solvents in their
    stated mass-fraction split. Output is a normalised mole-fraction vector
    across [cation, anion, solvent_1, solvent_2, ...].
    """
    solvents = recipe.get("solvents", {})
    salts = recipe.get("salts", {})
    additives = recipe.get("additives", {})

    for name in solvents:
        if name not in ALLOWED_SOLVENTS:
            return None
    if len(salts) != 1:
        return None
    salt_name = next(iter(salts.keys()))
    if salt_name not in ALLOWED_SALTS:
        return None
    for name in additives:
        if name in ALLOWED_ADDITIVES or name in ALLOWED_SOLVENTS:
            continue
        return None

    total_mass_g = ELECTROLYTE_DENSITY_G_CM3 * 1000.0   # 1 L of liquid
    salt_molarity_mol_per_L = float(salts[salt_name])
    salt_mass_g = salt_molarity_mol_per_L * _salt_mw(salt_name)
    solvent_mass_total_g = total_mass_g - salt_mass_g
    if solvent_mass_total_g <= 0:
        return None
    solvent_mass_frac_sum = sum(solvents.values())
    if solvent_mass_frac_sum <= 0:
        return None

    cation, anion = SALT_TO_IONS[salt_name]
    species_to_moles: dict[str, float] = {
        cation: salt_molarity_mol_per_L,
        anion: salt_molarity_mol_per_L,
    }
    for name, mfrac in solvents.items():
        mass_g = solvent_mass_total_g * (mfrac / solvent_mass_frac_sum)
        species_to_moles[name] = species_to_moles.get(name, 0.0) + mass_g / _species_mw(name)
    for name, frac in additives.items():
        if name not in ALLOWED_ADDITIVES and name not in ALLOWED_SOLVENTS:
            return None
        mass_g = solvent_mass_total_g * float(frac)
        species_to_moles[name] = species_to_moles.get(name, 0.0) + mass_g / _species_mw(name)

    species_list = list(species_to_moles.keys())
    mol_counts = np.array([species_to_moles[s] for s in species_list], dtype=np.float64)
    total = mol_counts.sum()
    if total <= 0:
        return None
    mole_fractions = (mol_counts / total).tolist()
    return species_list, mole_fractions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path,
                    help="Path to propagator checkpoint pickle.")
    ap.add_argument("--n-rows", type=int, default=8,
                    help="Number of validation rows to evaluate (random sample).")
    ap.add_argument("--sim-time-ps", type=float, default=2400.0,
                    help="Simulation time per row in ps. 2400 ps brings cepstral "
                         "P* to ~60 modes (plan F2 audit 2026-05-21).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    rows: list[dict] = []
    n_skip = 0
    for d in DATA:
        if "properties" not in d or "conductivity_mS_cm" not in d["properties"]:
            n_skip += 1
            continue
        conv = _convert_recipe_to_composition(d["recipe"])
        if conv is None:
            n_skip += 1
            continue
        species_list, mole_fractions = conv
        rows.append({
            "recipe": d["recipe"],
            "species_list": species_list,
            "mole_fractions": mole_fractions,
            "sigma_exp": float(d["properties"]["conductivity_mS_cm"]),
        })
    log.info("electrolyte_property_db: total=%d, kept=%d, skipped=%d",
             len(DATA), len(rows), n_skip)
    if not rows:
        log.error("no validation rows survived the catalogue filter")
        return 1

    sigmas = np.array([r["sigma_exp"] for r in rows])
    log.info("kept rows: sigma_exp range %.3f - %.3f mS/cm (median %.3f)",
             sigmas.min(), sigmas.max(), float(np.median(sigmas)))

    rng = np.random.default_rng(args.seed)
    idx_sample = rng.choice(len(rows), size=min(args.n_rows, len(rows)), replace=False)

    successes: list[dict] = []
    failures: list[dict] = []
    for i_pos, idx in enumerate(idx_sample):
        row = rows[int(idx)]
        log.info("=" * 70)
        log.info("row %d/%d (idx=%d): sigma_exp=%.3f mS/cm",
                 i_pos + 1, len(idx_sample), int(idx), row["sigma_exp"])
        log.info("  recipe: %s", row["recipe"])
        log.info("  species_list: %s", row["species_list"])
        log.info("  mole_fractions: %s", [round(x, 4) for x in row["mole_fractions"]])
        try:
            estimate = predict_sigma(
                model=args.model,
                species_list=row["species_list"],
                mole_fractions=row["mole_fractions"],
                temperature_K=EMPIRICAL_DEFAULT_T_K,
                sim_time_ps=args.sim_time_ps,
                seed=args.seed + i_pos,
            )
            sigma_pred = float(estimate.sigma_mS_cm)
            ratio = sigma_pred / row["sigma_exp"]
            log.info("  predicted: sigma=%.3f mS/cm CI=[%.3f, %.3f], ratio=%.3f",
                     sigma_pred, float(estimate.bootstrap_ci_low_mS_cm),
                     float(estimate.bootstrap_ci_high_mS_cm), ratio)
            successes.append({
                "row_idx": int(idx),
                "recipe": row["recipe"],
                "species_list": row["species_list"],
                "mole_fractions": row["mole_fractions"],
                "sigma_exp": row["sigma_exp"],
                "sigma_pred": sigma_pred,
                "sigma_pred_ci_low": float(estimate.bootstrap_ci_low_mS_cm),
                "sigma_pred_ci_high": float(estimate.bootstrap_ci_high_mS_cm),
                "ratio_pred_over_exp": ratio,
            })
        except Exception as e:
            failures.append({
                "row_idx": int(idx),
                "recipe": row["recipe"],
                "sigma_exp": row["sigma_exp"],
                "error": repr(e),
            })
            log.exception("row failed")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"successes": successes, "failures": failures}, f, indent=2)

    if successes:
        sigma_exp = np.array([r["sigma_exp"] for r in successes])
        sigma_pred = np.array([r["sigma_pred"] for r in successes])
        log_mse_model = float(np.mean((np.log(sigma_pred) - np.log(sigma_exp)) ** 2))
        log_mse_baseline = float(np.var(np.log(sigma_exp)))
        log.info("=" * 70)
        log.info("SUMMARY (%d/%d rows successful, %d failed):",
                 len(successes), len(idx_sample), len(failures))
        log.info("  log-MSE model    = %.5f", log_mse_model)
        log.info("  log-MSE baseline = %.5f (constant mean of log(sigma_exp))", log_mse_baseline)
        log.info("  model / baseline = %.3f  %s",
                 log_mse_model / max(log_mse_baseline, 1e-12),
                 "PASS" if log_mse_model < log_mse_baseline else "FAIL (worse than constant mean)")
    log.info("wrote per-row results to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
