"""
Standalone OOD evaluation for MolSet v2.
Retrains from scratch without each held-out species, then evaluates.

Entry point: python -m conductivity.run_ood_v2
"""

import logging
import time
import sys

sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")

import control_framework.jax_m4_tuning  # noqa: F401 — must precede jax import

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from collections import defaultdict

from conductivity.mol_set_sigma import (
    N_MAX_SPECIES,
    D_INPUT,
    SEED_OOD,
    N_STEPS,
    LR_PEAK,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    MolSetBatch,
    get_raw_property_vector,
    get_normalized_property_vector,
    compute_mix_physics_stats,
    _compute_mixture_physics,
    _load_all_sources,
    _recipe_key,
    _extract_species_fracs,
)

from conductivity.mol_set_sigma_v2 import (
    N_BASE_FEATURES,
    MIX_IDX_LAMBDA0_AVG,
    MIX_IDX_EPS_MIX,
    MIX_IDX_ETA_MIX,
    MIX_IDX_BINDING_AVG,
    MIX_IDX_ANION_R_AVG,
    MIX_IDX_JONES_DOLE_B_AVG,
    MIX_IDX_IONIC_STRENGTH,
    init_params_v2,
    make_train_step_v2,
    compute_val_mae_v2,
    _compute_base_feature_stats,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CV_REJECT_THRESHOLD = 0.3  # Explicit constant: 30% CV cutoff (same as main training)
LOG_EVERY = 200  # Explicit constant: log progress every N steps during OOD retraining


def evaluate_species_ood_with_logging(species_name, norm_mean, norm_std):
    """Hold out all recipes containing species_name, retrain v2 from scratch, evaluate OOD."""
    logger.info(f"\n{'='*60}")
    logger.info(f"OOD EVALUATION (v2): holding out '{species_name}'")
    logger.info(f"{'='*60}")

    all_entries = _load_all_sources()

    recipe_groups: dict[tuple, list] = defaultdict(list)
    for recipe, sigma, temp, source in all_entries:
        key = (_recipe_key(recipe), round(temp, 0))
        recipe_groups[key].append((sigma, temp, recipe, source))

    train_rows = []
    ood_rows = []

    for (rkey, T_round), measurements in recipe_groups.items():
        sigmas = [m[0] for m in measurements]
        if len(sigmas) > 1:
            arr = np.array(sigmas)
            cv = arr.std() / max(arr.mean(), 1e-8)
            if cv > CV_REJECT_THRESHOLD:
                continue

        recipe = measurements[0][2]
        all_sp = (list(recipe["salts"].keys()) +
                  list(recipe["solvents"].keys()) +
                  list(recipe["additives"].keys()))

        avg_sigma = np.mean(sigmas)
        avg_temp = np.mean([m[1] for m in measurements])

        row = {"recipe": recipe, "sigma": avg_sigma, "temp": avg_temp, "species": all_sp, "key": rkey}
        if species_name in all_sp:
            ood_rows.append(row)
        else:
            train_rows.append(row)

    logger.info(f"Train recipes (no {species_name}): {len(train_rows)}")
    logger.info(f"OOD recipes (with {species_name}): {len(ood_rows)}")

    if len(ood_rows) < 5:
        logger.warning(f"Too few OOD recipes ({len(ood_rows)}), skipping")
        return {"species": species_name, "n_ood": len(ood_rows), "ood_mae": None, "train_mae": None}

    def rows_to_batch(rows_list) -> MolSetBatch:
        n = len(rows_list)
        props = np.zeros((n, N_MAX_SPECIES, D_INPUT), dtype=np.float64)
        raw = np.zeros((n, N_MAX_SPECIES, D_INPUT), dtype=np.float64)
        fracs_arr = np.zeros((n, N_MAX_SPECIES), dtype=np.float64)
        mask_arr = np.zeros((n, N_MAX_SPECIES), dtype=np.float64)
        temps_arr = np.zeros(n, dtype=np.float64)
        log_sigmas = np.zeros(n, dtype=np.float64)
        weights_arr = np.ones(n, dtype=np.float64)

        for i, row in enumerate(rows_list):
            species_fracs = _extract_species_fracs(row["recipe"])
            for j, (sp_name, frac) in enumerate(species_fracs[:N_MAX_SPECIES]):
                props[i, j] = get_normalized_property_vector(sp_name, norm_mean, norm_std)
                raw[i, j] = get_raw_property_vector(sp_name)
                fracs_arr[i, j] = frac
                mask_arr[i, j] = 1.0
            temps_arr[i] = row["temp"]
            log_sigmas[i] = np.log(row["sigma"])

        return MolSetBatch(
            species_props=props, raw_props=raw, fracs=fracs_arr, mask=mask_arr,
            temperature_K=temps_arr, log_sigma=log_sigmas,
            weights=weights_arr,
            recipe_keys=[r["key"] for r in rows_list],
        )

    train_batch = rows_to_batch(train_rows)
    ood_batch = rows_to_batch(ood_rows)

    ood_mix_mean, ood_mix_std = compute_mix_physics_stats(train_batch)

    base_feat_mean, base_feat_std = _compute_base_feature_stats(train_batch)

    params = init_params_v2(random.PRNGKey(SEED_OOD), ood_mix_mean, ood_mix_std)
    params["base_feat_mean"] = jnp.array(base_feat_mean)
    params["base_feat_std"] = jnp.array(base_feat_std)

    warmup_fraction = WARMUP_STEPS / N_STEPS
    ood_warmup = int(N_STEPS * warmup_fraction)
    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, ood_warmup)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, N_STEPS - ood_warmup)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [ood_warmup])
    opt = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = opt.init(params)
    step_fn = make_train_step_v2(opt)

    ja = train_batch.jax_arrays()
    batch_tuple = (ja["props"], ja["raw"], ja["fracs"], ja["mask"], ja["temps"],
                   ja["log_sigma"], ja["weights"])

    ood_rng = random.PRNGKey(SEED_OOD + 1)
    best_ood_mae = float("inf")
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        ood_rng, step_key = random.split(ood_rng)
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple, step_key)

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t0
            ood_mae = compute_val_mae_v2(params, ood_batch)
            train_mae = compute_val_mae_v2(params, train_batch)
            if ood_mae < best_ood_mae:
                best_ood_mae = ood_mae
                best_step = step
            logger.info(
                f"  [{species_name}] Step {step:5d} | loss={float(loss):.4f} | "
                f"train={train_mae:.3f} | OOD={ood_mae:.3f} mS/cm | "
                f"best_OOD={best_ood_mae:.3f}@{best_step} | {elapsed:.0f}s"
            )

    final_ood_mae = compute_val_mae_v2(params, ood_batch)
    final_train_mae = compute_val_mae_v2(params, train_batch)

    logger.info(f"\nOOD {species_name}: train MAE={final_train_mae:.3f}, OOD MAE={final_ood_mae:.3f} mS/cm")
    logger.info(f"  Best OOD MAE={best_ood_mae:.3f} at step {best_step}")
    return {
        "species": species_name, "n_ood": len(ood_rows),
        "ood_mae": final_ood_mae, "train_mae": final_train_mae,
        "best_ood_mae": best_ood_mae, "best_step": best_step,
    }


def main():
    from conductivity.mol_set_sigma import (
        compute_normalization_stats, prepare_molset_data,
        _DATA_ORIGINAL, _DATA_CALISOL,
    )

    logger.info("Computing normalization stats from full dataset...")
    all_species = set()
    for entry in _DATA_ORIGINAL + _DATA_CALISOL:
        if "conductivity_mS_cm" not in entry["properties"]:
            continue
        r = entry["recipe"]
        for k in ["salts", "solvents", "additives"]:
            all_species.update(r[k].keys())
    all_species = sorted(all_species)
    logger.info(f"All species ({len(all_species)}): {all_species}")

    norm_mean, norm_std = compute_normalization_stats(all_species)

    ood_species = ["FEC", "VC", "LiFSI"]
    ood_results = []
    for sp in ood_species:
        r = evaluate_species_ood_with_logging(sp, norm_mean, norm_std)
        ood_results.append(r)

    logger.info(f"\n{'='*60}")
    logger.info(f"OOD SUMMARY (v2 vs v1)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Species':<10} {'v2 OOD':>10} {'v1 OOD':>10} {'Improvement':>12}")
    logger.info(f"{'-'*44}")
    v1_baselines = {"FEC": 0.603, "VC": 0.462, "LiFSI": 2.496}
    for r in ood_results:
        if r["ood_mae"] is not None:
            v1 = v1_baselines[r["species"]]
            improvement = (v1 - r["ood_mae"]) / v1 * 100
            logger.info(f"{r['species']:<10} {r['ood_mae']:>10.3f} {v1:>10.3f} {improvement:>+11.1f}%")

    logger.info(f"\nDone.")


if __name__ == "__main__":
    main()
