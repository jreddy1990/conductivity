"""Full training run for flow_matching_conductivity. Entry: python -m conductivity.run_fm_training.

Loads clean_dataset.pkl (15,849 rows), splits by held-out species, trains with
optax cosine schedule, checkpoints every epoch to fm_data/checkpoints/.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")
import control_framework.jax_m4_tuning  # noqa: F401

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax

from conductivity.flow_matching_conductivity import (
    CLEAN_DATASET_PATH, DATA_DIR, HELDOUT_SOLVENTS, HELDOUT_ANIONS,
    SEED_SPLIT, SEED_TRAIN, BATCH_SIZE, N_EPOCHS, LR_PEAK, LR_FLOOR,
    WARMUP_STEPS, WEIGHT_DECAY, GRAD_CLIP_NORM,
    compute_normalization, smiles_to_graph, split_dataset,
    init_all_params, ModelBundle, pad_composition,
    composition_loss, model_forward,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


CHECKPOINT_DIR = DATA_DIR / "checkpoints"

VAL_FRACTION = 0.10                 # Explicit constant: held-out validation share of the in-distribution split (10%)
LOG_INTERVAL_STEPS = 10             # Explicit constant: emit a training-loss log line every 10 optimizer steps
PER_EPOCH_EVAL_MAX_ROWS = 100       # Explicit constant: subsample val/ood to 100 rows for per-epoch metric so logging stays fast

# Env-var overrides for fast iteration: SUBSET_SIZE=N caps the dataset to N
# rows (after split, before training), TRAIN_EPOCHS overrides N_EPOCHS,
# TRAIN_BATCH overrides BATCH_SIZE. Unset = full settings from the model file.
SUBSET_SIZE = int(os.environ.get("SUBSET_SIZE", "0"))
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", str(N_EPOCHS)))
TRAIN_BATCH = int(os.environ.get("TRAIN_BATCH", str(BATCH_SIZE)))


def evaluate_rows(model, rows, key, max_rows: int) -> float:
    """Log-MSE on up to max_rows compositions."""
    errs = []
    for r in rows[:max_rows]:
        comp = pad_composition(
            r.smiles_list, r.mole_fractions, r.temperature_K, smiles_to_graph,
        )
        ek, key = random.split(key)
        log_pred, _ = model_forward(model, comp, ek, n_samples=1)
        errs.append(float((log_pred - jnp.log(r.sigma_mScm)) ** 2))
    return float(np.mean(errs))


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading clean dataset from %s", CLEAN_DATASET_PATH)
    with open(CLEAN_DATASET_PATH, "rb") as f:
        rows = pickle.load(f)
    logger.info("Total rows loaded: %d", len(rows))
    if SUBSET_SIZE > 0 and SUBSET_SIZE < len(rows):
        rng_subset = np.random.default_rng(SEED_SPLIT)
        idx = rng_subset.permutation(len(rows))[:SUBSET_SIZE]
        rows = [rows[i] for i in idx]
        logger.info("Subset to %d rows (SUBSET_SIZE=%d)", len(rows), SUBSET_SIZE)

    train_rows, val_rows, ood_rows = split_dataset(
        rows, HELDOUT_SOLVENTS, HELDOUT_ANIONS, VAL_FRACTION, SEED_SPLIT,
    )
    norm = compute_normalization(train_rows)
    logger.info("Norm stats: T_mean=%.2fK T_std=%.2fK", norm.T_mean, norm.T_std)

    key = random.PRNGKey(SEED_TRAIN)
    init_key, key = random.split(key)
    params = init_all_params(init_key)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info("Model params: %d", n_params)

    n_steps = (len(train_rows) // TRAIN_BATCH) * TRAIN_EPOCHS
    logger.info(
        "Training plan: %d epochs x %d steps/epoch x batch %d = %d total steps",
        TRAIN_EPOCHS, len(train_rows) // TRAIN_BATCH, TRAIN_BATCH, n_steps,
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR_PEAK,
        warmup_steps=WARMUP_STEPS, decay_steps=n_steps, end_value=LR_FLOOR,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = optimizer.init(params)

    def row_loss(p, r, k_i):
        m = ModelBundle(params=p, norm_stats=norm)
        comp = pad_composition(
            r.smiles_list, r.mole_fractions, r.temperature_K, smiles_to_graph,
        )
        w = 1.0 / (1.0 + r.sigma_uncertainty_log ** 2)
        loss, _ = composition_loss(m, comp, jnp.log(r.sigma_mScm), w, k_i)
        return loss

    def batch_loss(p, batch, k):
        ks = random.split(k, len(batch))
        return jnp.mean(jnp.array([row_loss(p, r, ki) for r, ki in zip(batch, ks)]))

    grad_fn = value_and_grad(batch_loss)
    rng = np.random.default_rng(SEED_TRAIN)
    step = 0
    t_start = time.time()
    for epoch in range(TRAIN_EPOCHS):
        rng.shuffle(train_rows)
        for b in range(0, len(train_rows) - TRAIN_BATCH + 1, TRAIN_BATCH):
            bkey, key = random.split(key)
            loss_val, grads = grad_fn(params, train_rows[b:b + TRAIN_BATCH], bkey)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            step += 1
            if step % LOG_INTERVAL_STEPS == 0:
                logger.info(
                    "step %d epoch %d loss=%.4f elapsed=%.0fs",
                    step, epoch, float(loss_val), time.time() - t_start,
                )
        m = ModelBundle(params=params, norm_stats=norm)
        val_mse = evaluate_rows(m, val_rows, key, max_rows=PER_EPOCH_EVAL_MAX_ROWS)
        ood_mse = evaluate_rows(m, ood_rows, key, max_rows=PER_EPOCH_EVAL_MAX_ROWS)
        logger.info(
            "epoch %d done  val_log_mse=%.4f  ood_log_mse=%.4f  elapsed=%.0fs",
            epoch, val_mse, ood_mse, time.time() - t_start,
        )
        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump({"params": params, "norm_stats": norm, "epoch": epoch,
                         "val_log_mse": val_mse, "ood_log_mse": ood_mse}, f)
        logger.info("Saved %s", ckpt_path)

    final_path = DATA_DIR / "fm_conductivity_model.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({"params": params, "norm_stats": norm}, f)
    logger.info("Training complete. Final model: %s", final_path)


if __name__ == "__main__":
    main()
