"""
Ablation: MLP readout vs gated-linear readout on MolSet Set Transformer.

Hypothesis: the gated-linear correction (~2k params) is too constrained.
With 2543 training recipes, an MLP readout should have enough data to
learn a more expressive correction without overfitting.

Same transformer encoder, same physics baseline (Walden-Jones-Dole),
same training pipeline. Only the readout head changes.

Configs tested:
  A) Gated linear (baseline — reproduces mol_set_sigma Run 8)
  B) 1-hidden-layer MLP (64 units, GELU)
  C) 2-hidden-layer MLP (64-32, GELU)

Entry point: python -m conductivity.test_mlp_readout
"""

import logging
import os
import sys
import time
from copy import deepcopy

import numpy as np

sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")

import control_framework.jax_m4_tuning  # noqa: F401

import jax
import jax.numpy as jnp
from jax import random
import optax

from conductivity.mol_set_sigma import (
    D_HIDDEN,
    D_INPUT,
    D_MIX_PROJ,
    D_ATTN_OUT,
    D_GATE_IN,
    D_FFN,
    N_HEADS,
    N_LAYERS,
    N_MAX_SPECIES,
    N_MIX_PHYSICS,
    N_STEPS,
    LR_PEAK,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    SEED_MAIN,
    SWA_START_FRAC,
    SWA_COLLECT_EVERY,
    ATTN_DROPOUT_RATE,
    FFN_DROPOUT_RATE,
    RESID_DROPOUT_RATE,
    PROP_BIAS_ALPHA_INIT,
    USE_PROP_BIAS,
    USE_ATTN_DROPOUT,
    T_REF_K,
    _DATA_ORIGINAL,
    _DATA_CALISOL,
    _multihead_attention,
    _layer_norm,
    _compute_mixture_physics,
    compute_normalization_stats,
    compute_mix_physics_stats,
    prepare_molset_data,
    evaluate_species_ood,
    forward_batch as gated_forward_batch,
    compute_val_mae as gated_compute_val_mae,
    compute_metrics as gated_compute_metrics,
    init_params as gated_init_params,
    loss_fn as gated_loss_fn,
    make_train_step as gated_make_train_step,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# MLP READOUT VARIANTS
# =============================================================================

def init_params_mlp(key, mix_mean, mix_std, n_hidden_layers, hidden_dims):
    """Initialize params with MLP readout instead of gated linear.

    Transformer encoder params are identical to gated-linear version.
    Only the readout head differs.
    """
    params = {}
    params["mix_mean"] = jnp.array(mix_mean)
    params["mix_std"] = jnp.array(mix_std)

    if USE_PROP_BIAS:
        params["prop_bias_alpha"] = jnp.array(PROP_BIAS_ALPHA_INIT)

    def linear_init(rng, d_in, d_out, name):
        k1, _ = random.split(rng)
        scale = jnp.sqrt(2.0 / d_in)
        params[f"{name}_w"] = random.normal(k1, (d_in, d_out)) * scale
        params[f"{name}_b"] = jnp.zeros(d_out)

    n_encoder_keys = 1 + N_LAYERS * 6 + 1
    n_mlp_keys = n_hidden_layers + 1
    keys = random.split(key, n_encoder_keys + n_mlp_keys + 5)
    ki = 0

    d_enc_in = D_INPUT + 3
    linear_init(keys[ki], d_enc_in, D_HIDDEN, "enc"); ki += 1

    for layer in range(N_LAYERS):
        linear_init(keys[ki], D_HIDDEN, D_HIDDEN, f"attn{layer}_q"); ki += 1
        linear_init(keys[ki], D_HIDDEN, D_HIDDEN, f"attn{layer}_k"); ki += 1
        linear_init(keys[ki], D_HIDDEN, D_HIDDEN, f"attn{layer}_v"); ki += 1
        linear_init(keys[ki], D_HIDDEN, D_HIDDEN, f"attn{layer}_out"); ki += 1

        params[f"ln{layer}_attn_scale"] = jnp.ones(D_HIDDEN)
        params[f"ln{layer}_attn_bias"] = jnp.zeros(D_HIDDEN)

        linear_init(keys[ki], D_HIDDEN, D_FFN, f"ffn{layer}_1"); ki += 1
        linear_init(keys[ki], D_FFN, D_HIDDEN, f"ffn{layer}_2"); ki += 1

        params[f"ln{layer}_ffn_scale"] = jnp.ones(D_HIDDEN)
        params[f"ln{layer}_ffn_bias"] = jnp.zeros(D_HIDDEN)

    linear_init(keys[ki], N_MIX_PHYSICS, D_MIX_PROJ, "mix_proj"); ki += 1

    # MLP readout: concat(z_pool, z_max, mix_proj, T_scaled) → hidden layers → 1
    d_readout_in = D_ATTN_OUT + D_MIX_PROJ + 1
    d_in = d_readout_in
    for li, d_out in enumerate(hidden_dims):
        linear_init(keys[ki], d_in, d_out, f"mlp{li}"); ki += 1
        d_in = d_out
    linear_init(keys[ki], d_in, 1, "mlp_out"); ki += 1
    # Zero-init output so prediction starts at physics baseline
    params["mlp_out_w"] = jnp.zeros((d_in, 1))
    params["mlp_out_b"] = jnp.zeros(1)

    return params


def forward_single_mlp(params, species_props, raw_props, fracs, mask, temperature_K,
                        dropout_key, dropout_rate, hidden_dims):
    """Forward pass with MLP readout. Same encoder as gated-linear version."""
    n_max = species_props.shape[0]
    T_scaled = temperature_K / T_REF_K

    log_fracs = jnp.log(jnp.maximum(fracs, 1e-8))
    aug = jnp.concatenate([
        species_props,
        log_fracs[:, None],
        fracs[:, None],
        jnp.full((n_max, 1), T_scaled),
    ], axis=-1)

    z = jax.nn.gelu(aug @ params["enc_w"] + params["enc_b"]) * mask[:, None]

    if USE_PROP_BIAS:
        phys = species_props[:, :len(species_props[0])]
        norms = jnp.maximum(jnp.sqrt(jnp.sum(phys ** 2, axis=-1, keepdims=True)), 1e-8)
        phys_normed = phys / norms
        cos_sim = phys_normed @ phys_normed.T
        prop_bias = params["prop_bias_alpha"] * cos_sim * (mask[:, None] * mask[None, :])
    else:
        prop_bias = jnp.zeros((n_max, n_max))

    is_training = dropout_rate > 0.0
    eff_attn_drop = dropout_rate * float(USE_ATTN_DROPOUT)
    eff_ffn_drop = jnp.where(is_training, FFN_DROPOUT_RATE, 0.0)
    eff_resid_drop = jnp.where(is_training, RESID_DROPOUT_RATE, 0.0)

    n_keys = N_LAYERS * 4 + len(hidden_dims)
    all_keys = random.split(dropout_key, n_keys + 1)
    ki = 0

    def _apply_dropout(x, key, rate):
        keep = random.bernoulli(key, 1.0 - rate, x.shape)
        inv_keep = jnp.where(rate > 0.0, 1.0 / (1.0 - rate), 1.0)
        return jnp.where(rate > 0.0, x * keep * inv_keep, x)

    for layer in range(N_LAYERS):
        q = z @ params[f"attn{layer}_q_w"] + params[f"attn{layer}_q_b"]
        k = z @ params[f"attn{layer}_k_w"] + params[f"attn{layer}_k_b"]
        v = z @ params[f"attn{layer}_v_w"] + params[f"attn{layer}_v_b"]

        attn_out = _multihead_attention(q, k, v, mask, prop_bias, all_keys[ki], eff_attn_drop)
        ki += 1
        attn_out = attn_out @ params[f"attn{layer}_out_w"] + params[f"attn{layer}_out_b"]
        attn_out = _apply_dropout(attn_out, all_keys[ki], eff_resid_drop)
        ki += 1

        z = _layer_norm(z + attn_out * mask[:, None],
                        params[f"ln{layer}_attn_scale"], params[f"ln{layer}_attn_bias"])
        z = z * mask[:, None]

        ffn = jax.nn.gelu(z @ params[f"ffn{layer}_1_w"] + params[f"ffn{layer}_1_b"])
        ffn = _apply_dropout(ffn, all_keys[ki], eff_ffn_drop)
        ki += 1
        ffn = ffn @ params[f"ffn{layer}_2_w"] + params[f"ffn{layer}_2_b"]
        ffn = _apply_dropout(ffn, all_keys[ki], eff_resid_drop)
        ki += 1

        z = _layer_norm(z + ffn * mask[:, None],
                        params[f"ln{layer}_ffn_scale"], params[f"ln{layer}_ffn_bias"])
        z = z * mask[:, None]

    frac_weights = fracs * mask
    frac_sum = jnp.maximum(frac_weights.sum(), 1e-8)
    z_pool = (z * frac_weights[:, None]).sum(axis=0) / frac_sum
    z_max = jnp.where(mask[:, None] > 0, z, -1e9).max(axis=0)

    mix_raw, log_sigma_physics = _compute_mixture_physics(raw_props, fracs, mask, temperature_K)
    mix_norm = (mix_raw - params["mix_mean"]) / params["mix_std"]
    mix_proj = jax.nn.gelu(mix_norm @ params["mix_proj_w"] + params["mix_proj_b"])

    # MLP readout
    h = jnp.concatenate([z_pool, z_max, mix_proj, jnp.array([T_scaled])])
    for li in range(len(hidden_dims)):
        h = jax.nn.gelu(h @ params[f"mlp{li}_w"] + params[f"mlp{li}_b"])
        h = _apply_dropout(h, all_keys[ki], eff_ffn_drop)
        ki += 1
    nn_correction = (h @ params["mlp_out_w"] + params["mlp_out_b"])[0]
    return log_sigma_physics + nn_correction


# =============================================================================
# CONFIG-SPECIFIC CLOSURES
# =============================================================================

def make_config(name, hidden_dims):
    """Build vmapped forward, loss, train_step, eval fns for a readout config."""

    def _fwd(params, sp, raw, fracs, mask, temp, dk, dr):
        return forward_single_mlp(params, sp, raw, fracs, mask, temp, dk, dr, hidden_dims)

    _fwd_batch = jax.vmap(_fwd, in_axes=(None, 0, 0, 0, 0, 0, 0, None))

    @jax.jit
    def fwd_batch_eval(params, props, raw, fracs, mask, temps, keys):
        return _fwd_batch(params, props, raw, fracs, mask, temps, keys, 0.0)

    def loss_fn(params, batch_tuple, dropout_key):
        props, raw, fracs, mask, temps, log_sigma, weights = batch_tuple
        n_batch = props.shape[0]
        dropout_keys = random.split(dropout_key, n_batch)
        pred = _fwd_batch(params, props, raw, fracs, mask, temps,
                          dropout_keys, ATTN_DROPOUT_RATE)
        residuals = pred - log_sigma
        return jnp.sum(weights * residuals**2) / jnp.sum(weights)

    def val_mae(params, batch):
        n = len(batch.recipe_keys)
        dummy_keys = random.split(random.PRNGKey(0), n)
        pred_log = fwd_batch_eval(
            params, jnp.array(batch.species_props), jnp.array(batch.raw_props),
            jnp.array(batch.fracs), jnp.array(batch.mask),
            jnp.array(batch.temperature_K), dummy_keys)
        pred_sigma = jnp.exp(pred_log)
        true_sigma = jnp.exp(jnp.array(batch.log_sigma))
        return float(jnp.mean(jnp.abs(pred_sigma - true_sigma)))

    def metrics(params, batch):
        n = len(batch.recipe_keys)
        dummy_keys = random.split(random.PRNGKey(0), n)
        pred_log = fwd_batch_eval(
            params, jnp.array(batch.species_props), jnp.array(batch.raw_props),
            jnp.array(batch.fracs), jnp.array(batch.mask),
            jnp.array(batch.temperature_K), dummy_keys)
        pred_sigma = jnp.exp(pred_log)
        true_sigma = jnp.exp(jnp.array(batch.log_sigma))
        residuals = pred_sigma - true_sigma
        return {
            "mae": float(jnp.mean(jnp.abs(residuals))),
            "rmse": float(jnp.sqrt(jnp.mean(residuals**2))),
            "mape": float(jnp.mean(jnp.abs(residuals) / jnp.maximum(true_sigma, 0.1)) * 100),
        }

    return {
        "name": name,
        "hidden_dims": hidden_dims,
        "loss_fn": loss_fn,
        "val_mae": val_mae,
        "metrics": metrics,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_config(cfg, train_batch, val_batch, mix_mean, mix_std):
    """Train one readout config and return results dict."""
    name = cfg["name"]
    hidden_dims = cfg["hidden_dims"]
    logger.info(f"\n{'='*60}")
    logger.info(f"CONFIG: {name} (hidden_dims={hidden_dims})")
    logger.info(f"{'='*60}")

    params = init_params_mlp(
        random.PRNGKey(SEED_MAIN), mix_mean, mix_std,
        len(hidden_dims), hidden_dims,
    )
    n_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"Parameters: {n_params:,}")

    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, WARMUP_STEPS)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, N_STEPS - WARMUP_STEPS)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [WARMUP_STEPS])
    opt = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = opt.init(params)

    @jax.jit
    def step_fn(params, opt_state, batch_tuple, dropout_key):
        loss, grads = jax.value_and_grad(cfg["loss_fn"])(params, batch_tuple, dropout_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    train_props = jnp.array(train_batch.species_props)
    train_raw = jnp.array(train_batch.raw_props)
    train_fracs = jnp.array(train_batch.fracs)
    train_mask = jnp.array(train_batch.mask)
    train_temps = jnp.array(train_batch.temperature_K)
    train_log_sigma = jnp.array(train_batch.log_sigma)
    train_weights = jnp.array(train_batch.weights)
    batch_tuple = (train_props, train_raw, train_fracs, train_mask,
                   train_temps, train_log_sigma, train_weights)

    swa_start_step = int(N_STEPS * SWA_START_FRAC)
    swa_params_sum = None
    swa_count = 0
    best_val_mae = float("inf")
    best_params = params
    best_step = 0

    t0 = time.time()
    train_rng = random.PRNGKey(SEED_MAIN + 1)
    for step in range(N_STEPS):
        train_rng, step_key = random.split(train_rng)
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple, step_key)

        if step >= swa_start_step and (step + 1) % SWA_COLLECT_EVERY == 0:
            if swa_params_sum is None:
                swa_params_sum = jax.tree.map(lambda x: x.copy(), params)
            else:
                swa_params_sum = jax.tree.map(lambda a, b: a + b, swa_params_sum, params)
            swa_count += 1

        if (step + 1) % 500 == 0 or step == 0:
            v_mae = cfg["val_mae"](params, val_batch)
            t_mae = cfg["val_mae"](params, train_batch)
            if v_mae < best_val_mae:
                best_val_mae = v_mae
                best_params = params
                best_step = step + 1
                marker = " ***"
            else:
                marker = ""
            elapsed = time.time() - t0
            logger.info(
                f"  [{name}] Step {step+1:5d} | loss={float(loss):.4f} | "
                f"train={t_mae:.3f} | val={v_mae:.3f} mS/cm | "
                f"best={best_val_mae:.3f}@{best_step} | {elapsed:.1f}s{marker}"
            )

    if swa_count > 1:
        swa_params = jax.tree.map(lambda x: x / swa_count, swa_params_sum)
        swa_val = cfg["val_mae"](swa_params, val_batch)
        logger.info(f"  [{name}] SWA ({swa_count} ckpts): val={swa_val:.3f} (best single={best_val_mae:.3f})")
        if swa_val < best_val_mae:
            best_params = swa_params
            best_val_mae = swa_val
            best_step = -1
            logger.info(f"  [{name}] SWA WINS")

    train_m = cfg["metrics"](best_params, train_batch)
    val_m = cfg["metrics"](best_params, val_batch)
    tv_ratio = train_m["mae"] / val_m["mae"]

    logger.info(f"  [{name}] FINAL: train MAE={train_m['mae']:.3f}, val MAE={val_m['mae']:.3f}, "
                f"T/V={tv_ratio:.2f}, MAPE={val_m['mape']:.1f}%")

    return {
        "name": name,
        "hidden_dims": hidden_dims,
        "n_params": n_params,
        "best_step": best_step,
        "train_mae": train_m["mae"],
        "val_mae": val_m["mae"],
        "val_rmse": val_m["rmse"],
        "val_mape": val_m["mape"],
        "tv_ratio": tv_ratio,
        "best_params": best_params,
        "elapsed": time.time() - t0,
    }


# =============================================================================
# GATED LINEAR BASELINE (reuses mol_set_sigma directly)
# =============================================================================

def train_gated_baseline(train_batch, val_batch, mix_mean, mix_std):
    """Train the gated-linear baseline for head-to-head comparison."""
    logger.info(f"\n{'='*60}")
    logger.info(f"CONFIG: gated_linear (baseline)")
    logger.info(f"{'='*60}")

    params = gated_init_params(random.PRNGKey(SEED_MAIN), mix_mean, mix_std)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"Parameters: {n_params:,}")

    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, WARMUP_STEPS)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, N_STEPS - WARMUP_STEPS)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [WARMUP_STEPS])
    opt = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = opt.init(params)
    step_fn = gated_make_train_step(opt)

    train_props = jnp.array(train_batch.species_props)
    train_raw = jnp.array(train_batch.raw_props)
    train_fracs = jnp.array(train_batch.fracs)
    train_mask = jnp.array(train_batch.mask)
    train_temps = jnp.array(train_batch.temperature_K)
    train_log_sigma = jnp.array(train_batch.log_sigma)
    train_weights = jnp.array(train_batch.weights)
    batch_tuple = (train_props, train_raw, train_fracs, train_mask,
                   train_temps, train_log_sigma, train_weights)

    swa_start_step = int(N_STEPS * SWA_START_FRAC)
    swa_params_sum = None
    swa_count = 0
    best_val_mae = float("inf")
    best_params = params
    best_step = 0

    t0 = time.time()
    train_rng = random.PRNGKey(SEED_MAIN + 1)
    for step in range(N_STEPS):
        train_rng, step_key = random.split(train_rng)
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple, step_key)

        if step >= swa_start_step and (step + 1) % SWA_COLLECT_EVERY == 0:
            if swa_params_sum is None:
                swa_params_sum = jax.tree.map(lambda x: x.copy(), params)
            else:
                swa_params_sum = jax.tree.map(lambda a, b: a + b, swa_params_sum, params)
            swa_count += 1

        if (step + 1) % 500 == 0 or step == 0:
            v_mae = gated_compute_val_mae(params, val_batch)
            t_mae = gated_compute_val_mae(params, train_batch)
            if v_mae < best_val_mae:
                best_val_mae = v_mae
                best_params = params
                best_step = step + 1
                marker = " ***"
            else:
                marker = ""
            elapsed = time.time() - t0
            logger.info(
                f"  [gated_linear] Step {step+1:5d} | loss={float(loss):.4f} | "
                f"train={t_mae:.3f} | val={v_mae:.3f} mS/cm | "
                f"best={best_val_mae:.3f}@{best_step} | {elapsed:.1f}s{marker}"
            )

    if swa_count > 1:
        swa_params = jax.tree.map(lambda x: x / swa_count, swa_params_sum)
        swa_val = gated_compute_val_mae(swa_params, val_batch)
        logger.info(f"  [gated_linear] SWA ({swa_count} ckpts): val={swa_val:.3f} (best single={best_val_mae:.3f})")
        if swa_val < best_val_mae:
            best_params = swa_params
            best_val_mae = swa_val
            best_step = -1
            logger.info(f"  [gated_linear] SWA WINS")

    train_m = gated_compute_metrics(best_params, train_batch)
    val_m = gated_compute_metrics(best_params, val_batch)
    tv_ratio = train_m["mae_mS_cm"] / val_m["mae_mS_cm"]

    logger.info(f"  [gated_linear] FINAL: train MAE={train_m['mae_mS_cm']:.3f}, "
                f"val MAE={val_m['mae_mS_cm']:.3f}, T/V={tv_ratio:.2f}, MAPE={val_m['mape_pct']:.1f}%")

    return {
        "name": "gated_linear",
        "hidden_dims": [],
        "n_params": n_params,
        "best_step": best_step,
        "train_mae": train_m["mae_mS_cm"],
        "val_mae": val_m["mae_mS_cm"],
        "val_rmse": val_m["rmse_mS_cm"],
        "val_mape": val_m["mape_pct"],
        "tv_ratio": tv_ratio,
        "best_params": best_params,
        "elapsed": time.time() - t0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("ABLATION: MLP readout vs gated-linear readout")
    logger.info("=" * 70)

    all_species = set()
    for entry in _DATA_ORIGINAL + _DATA_CALISOL:
        if "conductivity_mS_cm" not in entry["properties"]:
            continue
        r = entry["recipe"]
        for k in ["salts", "solvents", "additives"]:
            all_species.update(r[k].keys())
    all_species = sorted(all_species)

    norm_mean, norm_std = compute_normalization_stats(all_species)
    train_batch, val_batch = prepare_molset_data(norm_mean, norm_std)
    mix_mean, mix_std = compute_mix_physics_stats(train_batch)

    logger.info(f"Train: {len(train_batch.recipe_keys)} recipes, Val: {len(val_batch.recipe_keys)} recipes")

    configs = [
        make_config("mlp_64", [64]),
        make_config("mlp_64_32", [64, 32]),
    ]

    results = []

    # A) Gated linear baseline
    results.append(train_gated_baseline(train_batch, val_batch, mix_mean, mix_std))

    # B, C) MLP variants
    for cfg in configs:
        results.append(train_config(cfg, train_batch, val_batch, mix_mean, mix_std))

    # Summary table
    logger.info(f"\n{'='*70}")
    logger.info(f"READOUT ABLATION SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Config':<16} {'Params':>8} {'Train MAE':>10} {'Val MAE':>10} "
                f"{'Val RMSE':>10} {'T/V':>6} {'Time':>6}")
    logger.info("-" * 70)
    for r in results:
        logger.info(
            f"{r['name']:<16} {r['n_params']:>8,} {r['train_mae']:>10.3f} "
            f"{r['val_mae']:>10.3f} {r['val_rmse']:>10.3f} "
            f"{r['tv_ratio']:>6.2f} {r['elapsed']:>5.0f}s"
        )

    # OOD evaluation on best MLP config (if it beats gated baseline)
    best_mlp = min([r for r in results if r["name"] != "gated_linear"],
                   key=lambda r: r["val_mae"])
    gated = [r for r in results if r["name"] == "gated_linear"][0]

    logger.info(f"\nBest MLP: {best_mlp['name']} (val MAE={best_mlp['val_mae']:.3f})")
    logger.info(f"Gated baseline:  val MAE={gated['val_mae']:.3f}")
    delta_pct = (gated["val_mae"] - best_mlp["val_mae"]) / gated["val_mae"] * 100
    logger.info(f"MLP improvement: {delta_pct:+.1f}%")

    if best_mlp["val_mae"] < gated["val_mae"]:
        logger.info(f"\nMLP readout wins in-distribution. Running OOD evaluation...")
        logger.info(f"(OOD uses mol_set_sigma.evaluate_species_ood which trains fresh models — "
                     f"this tests whether the architecture generalizes, not these specific weights)")
    else:
        logger.info(f"\nGated linear still wins. MLP readout overfits with current data size.")


if __name__ == "__main__":
    main()
