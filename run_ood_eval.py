"""Run OOD evaluation only — uses existing species data, not saved model.

Each OOD eval retrains from scratch (leave-one-out), so no saved model needed.
"""
import sys
sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")

import control_framework.jax_m4_tuning  # noqa: F401

from conductivity.mol_set_sigma import (
    compute_normalization_stats, evaluate_species_ood,
    _DATA_ORIGINAL, _DATA_CALISOL, logger,
)

def main():
    all_species = set()
    for entry in _DATA_ORIGINAL + _DATA_CALISOL:
        if "conductivity_mS_cm" not in entry["properties"]:
            continue
        r = entry["recipe"]
        for k in ["salts", "solvents", "additives"]:
            all_species.update(r[k].keys())

    all_species = sorted(all_species)
    norm_mean, norm_std = compute_normalization_stats(all_species)

    import sys
    species_to_eval = sys.argv[1:]
    if not species_to_eval:
        raise ValueError("Usage: python -m conductivity.run_ood_eval SPECIES1 [SPECIES2 ...]")

    for sp in species_to_eval:
        logger.info(f"\n{'='*60}")
        result = evaluate_species_ood(sp, norm_mean, norm_std)
        logger.info(f"  {result['species']:8s}: OOD MAE = {result['ood_mae']:.3f} mS/cm "
                     f"(train MAE = {result['train_mae']:.3f}, n_ood = {result['n_ood']})")

if __name__ == "__main__":
    main()
