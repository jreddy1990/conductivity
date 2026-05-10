"""
Equivariant Property-Space Set Transformer for electrolyte conductivity.

Builds on the base MolSet architecture (mol_set_sigma.py) with four new mechanisms:
1. Mahalanobis property-space attention bias
2. Mixture-conditioned gating
3. Pairwise property enrichment (tanh-bounded)
4. Physics-bounded multi-channel readout with distance awareness

Entry point: python -m conductivity.mol_set_sigma_equivariant
"""

import logging
import os
import pickle
import time
import sys
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")

import control_framework.jax_m4_tuning  # noqa: F401 — must precede jax import

import jax
import jax.numpy as jnp
from jax import random
import optax

from constants import T_REF_K

from conductivity.mol_set_sigma import (
    # Data infrastructure
    MolSetBatch, prepare_molset_data, compute_normalization_stats,
    get_normalized_property_vector, get_raw_property_vector,
    _load_all_sources, _recipe_key, _extract_species_fracs,
    compute_mix_physics_stats,
    # Physics
    _compute_mixture_physics, _multihead_attention, _layer_norm,
    _MIX_FEATURE_NAMES,
    # Constants
    D_PROP, D_INPUT, D_HIDDEN, N_HEADS, N_LAYERS, D_FFN, N_MAX_SPECIES,
    N_MIX_PHYSICS,
    ATTN_DROPOUT_RATE, FFN_DROPOUT_RATE, RESID_DROPOUT_RATE,
    USE_ATTN_DROPOUT,
    N_STEPS, LR_PEAK, WARMUP_STEPS, WEIGHT_DECAY, MAX_GRAD_NORM,
    SEED_MAIN, SEED_OOD,
    SWA_START_FRAC, SWA_COLLECT_EVERY,
    OOD_PROXY_SPECIES, LOG_EVERY, OOD_LOG_EVERY,
    EARLY_STOP_PATIENCE, EARLY_STOP_REL_TOL,
    LAMBDA_CORRECTION,
    # Data sources (for species enumeration in main)
    _DATA_ORIGINAL, _DATA_CALISOL,
)
from data.species_data import SOLVENTS, SALTS, ADDITIVES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# EQUIVARIANT ARCHITECTURE CONSTANTS
# =============================================================================

D_METRIC = 8          # Explicit constant: Mahalanobis metric subspace dimension (24-d -> 8-d projection)
D_PAIR_HIDDEN = 12    # Explicit constant: pairwise enrichment hidden dimension
D_CHAN_HIDDEN = 12    # Explicit constant: bounded readout channel hidden dimension
N_CENTROIDS = 16      # Explicit constant: K-means centroids for distance-awareness

_DISSOC_CHAN_FEATURES = ("eps_kirkwood", "alpha_screened", "alpha_x_c", "dh_log_gamma", "bjerrum_nm")
_VISC_CHAN_FEATURES = ("eta_mix", "jones_dole_B_avg", "c_total", "jones_dole_correction")
_DISSOC_CHAN_IDX = tuple(_MIX_FEATURE_NAMES.index(n) for n in _DISSOC_CHAN_FEATURES)
_VISC_CHAN_IDX = tuple(_MIX_FEATURE_NAMES.index(n) for n in _VISC_CHAN_FEATURES)
D_DISSOC_CHAN_IN = len(_DISSOC_CHAN_FEATURES) + 1  # Explicit constant: channel features + T_scaled
D_VISC_CHAN_IN = len(_VISC_CHAN_FEATURES) + 1      # Explicit constant: channel features + T_scaled
D_INTERACT_CHAN_IN = D_HIDDEN + 1                  # Explicit constant: z_pool + T_scaled

CENTROID_FINETUNE_STEPS = 500  # Explicit constant: fine-tune with distance-awareness active

OOD_EVAL_SPECIES = ("FEC", "VC", "LiFSI")  # Explicit constant: species held out for OOD evaluation


# =============================================================================
# NEW MECHANISM FUNCTIONS
# =============================================================================

def _compute_property_metric_bias(raw_props, mask, params):
    """Compute Mahalanobis distance-based attention bias from raw property vectors.

    L projects 24-d raw properties to D_METRIC-d subspace.
    d_M^2(i,j) = ||L^T(p_i - p_j)||^2 is the squared Mahalanobis distance.
    bias(i,j) = beta * exp(-gamma * d_M^2(i,j)) for active species pairs.

    Returns: (N_max, N_max) symmetric bias matrix.
    """
    L = params["metric_L"]  # (D_PROP, D_METRIC)
    proj = raw_props @ L     # (N_max, D_METRIC)

    diff = proj[:, None, :] - proj[None, :, :]  # (N_max, N_max, D_METRIC)
    d_sq = jnp.sum(diff ** 2, axis=-1)           # (N_max, N_max)

    beta = params["metric_beta"]
    gamma = jax.nn.softplus(params["metric_gamma_raw"])

    bias = beta * jnp.exp(-gamma * d_sq) * (mask[:, None] * mask[None, :])
    return bias


def _mixture_conditioned_gating(z, fracs, mask, params):
    """Gate each species' representation based on mixture context.

    mix_summary = composition-weighted mean of z.
    gate_i = sigmoid(MLP([z_i, mix_summary])) for each species.
    z_gated = z * (1 + alpha * (gate - 0.5) * 2).
    """
    w = fracs * mask
    w_sum = jnp.maximum(jnp.sum(w), 1e-8)
    mix_summary = jnp.sum(z * w[:, None], axis=0) / w_sum

    mix_broadcast = jnp.broadcast_to(mix_summary[None, :], z.shape)
    gate_input = jnp.concatenate([z, mix_broadcast], axis=-1)

    gate_h = jax.nn.gelu(gate_input @ params["gate_mlp_w"] + params["gate_mlp_b"])
    gate = jax.nn.sigmoid(gate_h @ params["gate_out_w"] + params["gate_out_b"])

    alpha = params["alpha_gate"]
    z_gated = z * (1.0 + alpha * (gate - 0.5) * 2.0)
    return z_gated * mask[:, None]


def _pairwise_property_enrichment(raw_props, z, fracs, mask, params):
    """Enrich species representations with bounded pairwise property interactions.

    Uses RAW property vectors (not learned embeddings) so unseen species
    get physically meaningful interaction signals.
    All pairwise features bounded by tanh.
    """
    proj_src = raw_props @ params["pair_proj_src_w"] + params["pair_proj_src_b"]
    proj_dst = raw_props @ params["pair_proj_dst_w"] + params["pair_proj_dst_b"]

    interaction = jax.nn.tanh(proj_src[:, None, :] * proj_dst[None, :, :])
    msg = interaction @ params["pair_proj_out_w"] + params["pair_proj_out_b"]

    w = fracs * mask
    mask_2d = mask[:, None] * mask[None, :]
    weighted_msg = msg * w[None, :, None] * mask_2d[:, :, None]
    enrich = jnp.sum(weighted_msg, axis=1) / jnp.maximum(jnp.sum(w), 1e-8)

    alpha = params["alpha_pair"]
    return z + alpha * enrich * mask[:, None]


def _bounded_channel_readout(mix_raw, z_pool, T_scaled, params):
    """Three physics-bounded correction channels with distance awareness.

    Each channel: MLP -> tanh * max_scale (bounded by construction).
    Interaction channel additionally scaled by distance-based confidence.
    """
    mix_mean = params["mix_mean"]
    mix_std = params["mix_std"]
    mix_norm = (mix_raw - mix_mean) / jnp.maximum(mix_std, 1e-8)

    dissoc_features = jnp.concatenate([mix_norm[jnp.array(_DISSOC_CHAN_IDX)], jnp.array([T_scaled])])
    dissoc_h = jax.nn.tanh(dissoc_features @ params["dissoc_h_w"] + params["dissoc_h_b"])
    delta_dissoc_raw = (dissoc_h @ params["dissoc_out_w"] + params["dissoc_out_b"])[0]
    max_dissoc = jax.nn.softplus(params["max_dissoc_scale_raw"])
    delta_dissoc = jax.nn.tanh(delta_dissoc_raw) * max_dissoc

    visc_features = jnp.concatenate([mix_norm[jnp.array(_VISC_CHAN_IDX)], jnp.array([T_scaled])])
    visc_h = jax.nn.tanh(visc_features @ params["visc_h_w"] + params["visc_h_b"])
    delta_visc_raw = (visc_h @ params["visc_out_w"] + params["visc_out_b"])[0]
    max_visc = jax.nn.softplus(params["max_visc_scale_raw"])
    delta_visc = jax.nn.tanh(delta_visc_raw) * max_visc

    interact_features = jnp.concatenate([z_pool, jnp.array([T_scaled])])
    interact_h = jax.nn.tanh(interact_features @ params["interact_h_w"] + params["interact_h_b"])
    delta_interact_raw = (interact_h @ params["interact_out_w"] + params["interact_out_b"])[0]
    max_interact = jax.nn.softplus(params["max_interact_scale_raw"])

    centroids = params["centroids"]
    d_ref = jax.nn.softplus(params["d_ref_raw"])
    dists = jnp.sum((z_pool[None, :] - centroids) ** 2, axis=-1)
    d_train = jnp.min(dists)
    confidence = 1.0 / (1.0 + d_train / jnp.maximum(d_ref, 1e-8))

    delta_interact = jax.nn.tanh(delta_interact_raw) * max_interact * confidence

    nn_correction = delta_dissoc + delta_visc + delta_interact
    return nn_correction


# =============================================================================
# INIT / FORWARD
# =============================================================================

def init_params(key: jax.Array, mix_mean: np.ndarray, mix_std: np.ndarray) -> dict:
    """Initialize equivariant architecture parameters."""
    params = {}
    params["mix_mean"] = jnp.array(mix_mean)
    params["mix_std"] = jnp.array(mix_std)

    def linear_init(rng, d_in, d_out, name):
        k1, _ = random.split(rng)
        scale = jnp.sqrt(2.0 / d_in)
        params[f"{name}_w"] = random.normal(k1, (d_in, d_out)) * scale
        params[f"{name}_b"] = jnp.zeros(d_out)

    n_keys = (1 + N_LAYERS * 6 +
              1 +  # metric_L
              4 +  # gate_mlp, gate_out, pair_proj_src, pair_proj_dst
              1 +  # pair_proj_out
              3 +  # dissoc_h, visc_h, interact_h
              5)   # headroom
    keys = random.split(key, n_keys)
    ki = 0

    # Encoder (same as base)
    d_enc_in = D_INPUT + 3  # props + [log_frac, frac, T_scaled]
    linear_init(keys[ki], d_enc_in, D_HIDDEN, "enc"); ki += 1

    # Self-attention layers (same as base)
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

    # --- NEW: Property metric ---
    params["metric_L"] = random.normal(keys[ki], (D_PROP, D_METRIC)) / jnp.sqrt(jnp.array(D_PROP, dtype=jnp.float64))
    ki += 1
    params["metric_beta"] = jnp.array(0.0)
    params["metric_gamma_raw"] = jnp.array(0.0)

    # --- NEW: Mixture gating ---
    linear_init(keys[ki], 2 * D_HIDDEN, D_HIDDEN, "gate_mlp"); ki += 1
    linear_init(keys[ki], D_HIDDEN, 1, "gate_out"); ki += 1
    params["alpha_gate"] = jnp.array(0.0)

    # --- NEW: Pairwise enrichment ---
    linear_init(keys[ki], D_PROP, D_PAIR_HIDDEN, "pair_proj_src"); ki += 1
    linear_init(keys[ki], D_PROP, D_PAIR_HIDDEN, "pair_proj_dst"); ki += 1
    linear_init(keys[ki], D_PAIR_HIDDEN, D_HIDDEN, "pair_proj_out"); ki += 1
    params["alpha_pair"] = jnp.array(0.0)

    # --- NEW: Bounded readout channels ---
    for ch_name, d_in in [("dissoc", D_DISSOC_CHAN_IN), ("visc", D_VISC_CHAN_IN), ("interact", D_INTERACT_CHAN_IN)]:
        linear_init(keys[ki], d_in, D_CHAN_HIDDEN, f"{ch_name}_h"); ki += 1
        params[f"{ch_name}_out_w"] = jnp.zeros((D_CHAN_HIDDEN, 1))
        params[f"{ch_name}_out_b"] = jnp.zeros(1)
        params[f"max_{ch_name}_scale_raw"] = jnp.array(0.0)  # softplus(0) = ln(2) ~ 0.69

    # --- NEW: Distance awareness ---
    params["centroids"] = jnp.zeros((N_CENTROIDS, D_HIDDEN))
    params["d_ref_raw"] = jnp.array(0.0)

    return params


def forward_single(params, species_props, raw_props, fracs, mask, temperature_K,
                    dropout_key, dropout_rate):
    """Equivariant forward pass for a single recipe."""
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

    # Mahalanobis property-space attention bias (replaces cosine similarity)
    prop_bias = _compute_property_metric_bias(raw_props, mask, params)

    is_training = dropout_rate > 0.0
    eff_attn_drop = dropout_rate * float(USE_ATTN_DROPOUT)
    eff_ffn_drop = jnp.where(is_training, FFN_DROPOUT_RATE, 0.0)
    eff_resid_drop = jnp.where(is_training, RESID_DROPOUT_RATE, 0.0)

    n_keys = N_LAYERS * 4
    all_keys = random.split(dropout_key, n_keys)
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

    # Post-attention: mixture-conditioned gating + pairwise enrichment
    z = _mixture_conditioned_gating(z, fracs, mask, params)
    z = _pairwise_property_enrichment(raw_props, z, fracs, mask, params)

    # Composition-weighted pooling
    w = fracs * mask
    w_sum = jnp.maximum(jnp.sum(w), 1e-8)
    z_pool = jnp.sum(z * w[:, None], axis=0) / w_sum

    # Mixture physics for WJD baseline
    mix_raw, log_sigma_physics = _compute_mixture_physics(raw_props, fracs, mask, temperature_K)

    # Physics-bounded multi-channel readout
    nn_correction = _bounded_channel_readout(mix_raw, z_pool, T_scaled, params)

    return log_sigma_physics + nn_correction, nn_correction


forward_batch = jax.vmap(forward_single, in_axes=(None, 0, 0, 0, 0, 0, 0, None))


@jax.jit
def _forward_batch_eval(params, props, raw, fracs, mask, temps, keys):
    log_sigma, _correction = forward_batch(params, props, raw, fracs, mask, temps, keys, 0.0)
    return log_sigma


@jax.jit
def _forward_single_eval(params, props, raw, fracs, mask, temp):
    log_sigma, _correction = forward_single(params, props, raw, fracs, mask, temp, random.PRNGKey(0), 0.0)
    return log_sigma


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def loss_fn(params, batch_tuple, dropout_key):
    """Weighted log-MSE loss + correction magnitude penalty."""
    props, raw, fracs, mask, temps, log_sigma, weights = batch_tuple
    n_batch = props.shape[0]

    dropout_keys = random.split(dropout_key, n_batch)
    pred_log_sigma, nn_corrections = forward_batch(params, props, raw, fracs, mask, temps,
                                                    dropout_keys, ATTN_DROPOUT_RATE)
    residuals = pred_log_sigma - log_sigma
    recon_loss = jnp.sum(weights * residuals**2) / jnp.sum(weights)
    correction_penalty = jnp.mean(nn_corrections**2)
    return recon_loss + LAMBDA_CORRECTION * correction_penalty


def make_train_step(opt):
    @jax.jit
    def step(params, opt_state, batch_tuple, dropout_key):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_tuple, dropout_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return step


def compute_val_mae(params, batch: MolSetBatch) -> float:
    ja = batch.jax_arrays()
    n = len(batch.recipe_keys)
    dummy_keys = random.split(random.PRNGKey(0), n)
    pred_log_sigma = _forward_batch_eval(
        params, ja["props"], ja["raw"], ja["fracs"], ja["mask"], ja["temps"], dummy_keys,
    )
    pred_sigma = jnp.exp(pred_log_sigma)
    true_sigma = jnp.exp(ja["log_sigma"])
    return float(jnp.mean(jnp.abs(pred_sigma - true_sigma)))


def compute_metrics(params, batch: MolSetBatch) -> dict:
    ja = batch.jax_arrays()
    n = len(batch.recipe_keys)
    dummy_keys = random.split(random.PRNGKey(0), n)
    pred_log_sigma = _forward_batch_eval(
        params, ja["props"], ja["raw"], ja["fracs"], ja["mask"], ja["temps"], dummy_keys,
    )
    pred_sigma = jnp.exp(pred_log_sigma)
    true_sigma = jnp.exp(ja["log_sigma"])

    residuals = pred_sigma - true_sigma
    return {
        "mae_mS_cm": float(jnp.mean(jnp.abs(residuals))),
        "rmse_mS_cm": float(jnp.sqrt(jnp.mean(residuals**2))),
        "bias_mS_cm": float(jnp.mean(residuals)),
        "mape_pct": float(jnp.mean(jnp.abs(residuals) / jnp.maximum(true_sigma, 0.1)) * 100),
        "log_mse": float(jnp.mean((pred_log_sigma - ja["log_sigma"])**2)),
    }


def save_model(params: dict, norm_mean: np.ndarray, norm_std: np.ndarray, path: str) -> None:
    bundle = {
        "params": {k: np.array(v) for k, v in params.items()},
        "norm_mean": np.array(norm_mean),
        "norm_std": np.array(norm_std),
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model(path: str) -> tuple:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    params = {k: jnp.array(v) for k, v in bundle["params"].items()}
    return params, bundle["norm_mean"], bundle["norm_std"]


# =============================================================================
# OOD EVALUATION
# =============================================================================

def evaluate_species_ood(
    species_name: str, norm_mean: np.ndarray, norm_std: np.ndarray,
    step_fn, opt
) -> dict:
    """Hold out all recipes containing species_name, retrain from scratch, evaluate."""
    logger.info(f"\n{'='*60}")
    logger.info(f"OOD EVALUATION (equivariant): holding out '{species_name}'")
    logger.info(f"{'='*60}")

    all_entries = _load_all_sources()

    recipe_groups: Dict[tuple, list] = defaultdict(list)
    for recipe, sigma, temp, source in all_entries:
        key = (_recipe_key(recipe), round(temp, 0))
        recipe_groups[key].append((sigma, temp, recipe, source))

    CV_REJECT_THRESHOLD = 0.3  # Explicit constant: 30% coefficient of variation cutoff
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
        all_sp = list(recipe["salts"].keys()) + \
                 list(recipe["solvents"].keys()) + \
                 list(recipe["additives"].keys())

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

    logger.info(f"OOD train: {len(train_batch.recipe_keys)} recipes")

    params = init_params(random.PRNGKey(SEED_OOD), ood_mix_mean, ood_mix_std)
    opt_state = opt.init(params)

    ja = train_batch.jax_arrays()
    batch_tuple = (ja["props"], ja["raw"], ja["fracs"], ja["mask"], ja["temps"],
                   ja["log_sigma"], ja["weights"])
    ood_rng = random.PRNGKey(SEED_OOD + 1)
    best_ood_mae_retrain = float("inf")
    best_ood_step_retrain = 0
    best_ood_params = params
    ood_stall_counter = 0
    ood_prev_best = float("inf")
    t0_ood = time.time()
    for step in range(N_STEPS):
        ood_rng, step_key = random.split(ood_rng)
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple, step_key)

        if (step + 1) % OOD_LOG_EVERY == 0 or step == 0:
            cur_ood_mae = compute_val_mae(params, ood_batch)
            if cur_ood_mae < best_ood_mae_retrain:
                best_ood_mae_retrain = cur_ood_mae
                best_ood_step_retrain = step + 1
                best_ood_params = params
            elapsed_ood = time.time() - t0_ood
            logger.info(
                f"  [{species_name}] Step {step+1:5d} | loss={float(loss):.4f} | "
                f"OOD={cur_ood_mae:.3f} | best={best_ood_mae_retrain:.3f}@{best_ood_step_retrain} | {elapsed_ood:.0f}s"
            )

            rel_imp = (ood_prev_best - best_ood_mae_retrain) / ood_prev_best if ood_prev_best < float("inf") else 1.0
            if rel_imp < EARLY_STOP_REL_TOL:
                ood_stall_counter += 1
                if ood_stall_counter >= EARLY_STOP_PATIENCE:
                    logger.info(
                        f"  [{species_name}] EARLY STOP at step {step+1}: OOD metric stalled for "
                        f"{ood_stall_counter} eval rounds (<{EARLY_STOP_REL_TOL*100:.0f}% improvement each)"
                    )
                    break
            else:
                ood_stall_counter = 0
            ood_prev_best = best_ood_mae_retrain

    # K-means centroid computation for OOD model
    _compute_and_inject_centroids(best_ood_params, train_batch)

    final_ood_mae = compute_val_mae(best_ood_params, ood_batch)
    train_mae = compute_val_mae(best_ood_params, train_batch)

    logger.info(f"OOD {species_name}: train MAE={train_mae:.3f}, OOD MAE={final_ood_mae:.3f} mS/cm")
    logger.info(f"  Best OOD MAE={best_ood_mae_retrain:.3f} at step {best_ood_step_retrain}")
    return {
        "species": species_name, "n_ood": len(ood_rows),
        "ood_mae": final_ood_mae, "train_mae": train_mae,
        "best_ood_mae": best_ood_mae_retrain, "best_step": best_ood_step_retrain,
    }


# =============================================================================
# K-MEANS CENTROID COMPUTATION
# =============================================================================

def _extract_z_pool(params, batch: MolSetBatch) -> np.ndarray:
    """Extract z_pool vectors for all recipes in a batch."""
    n = len(batch.recipe_keys)
    z_pool_all = np.zeros((n, D_HIDDEN), dtype=np.float64)
    ja = batch.jax_arrays()

    for i in range(n):
        n_max = ja["props"][i].shape[0]
        T_sc = ja["temps"][i] / T_REF_K
        log_f = jnp.log(jnp.maximum(ja["fracs"][i], 1e-8))
        aug = jnp.concatenate([ja["props"][i], log_f[:, None],
                               ja["fracs"][i][:, None],
                               jnp.full((n_max, 1), T_sc)], axis=-1)
        z = jax.nn.gelu(aug @ params["enc_w"] + params["enc_b"]) * ja["mask"][i][:, None]

        prop_bias = _compute_property_metric_bias(ja["raw"][i], ja["mask"][i], params)
        for layer in range(N_LAYERS):
            q = z @ params[f"attn{layer}_q_w"] + params[f"attn{layer}_q_b"]
            k = z @ params[f"attn{layer}_k_w"] + params[f"attn{layer}_k_b"]
            v = z @ params[f"attn{layer}_v_w"] + params[f"attn{layer}_v_b"]
            attn_out = _multihead_attention(q, k, v, ja["mask"][i], prop_bias, random.PRNGKey(0), 0.0)
            attn_out = attn_out @ params[f"attn{layer}_out_w"] + params[f"attn{layer}_out_b"]
            z = _layer_norm(z + attn_out * ja["mask"][i][:, None],
                            params[f"ln{layer}_attn_scale"], params[f"ln{layer}_attn_bias"])
            z = z * ja["mask"][i][:, None]
            ffn = jax.nn.gelu(z @ params[f"ffn{layer}_1_w"] + params[f"ffn{layer}_1_b"])
            ffn = ffn @ params[f"ffn{layer}_2_w"] + params[f"ffn{layer}_2_b"]
            z = _layer_norm(z + ffn * ja["mask"][i][:, None],
                            params[f"ln{layer}_ffn_scale"], params[f"ln{layer}_ffn_bias"])
            z = z * ja["mask"][i][:, None]

        z = _mixture_conditioned_gating(z, ja["fracs"][i], ja["mask"][i], params)
        z = _pairwise_property_enrichment(ja["raw"][i], z, ja["fracs"][i], ja["mask"][i], params)

        w = ja["fracs"][i] * ja["mask"][i]
        w_sum = jnp.maximum(jnp.sum(w), 1e-8)
        z_pool_all[i] = np.array(jnp.sum(z * w[:, None], axis=0) / w_sum)

    return z_pool_all


def _compute_and_inject_centroids(params: dict, train_batch: MolSetBatch) -> None:
    """Compute K-means centroids from training z_pool and inject into params (in-place)."""
    from scipy.cluster.vq import kmeans2

    n_train = len(train_batch.recipe_keys)
    logger.info(f"\n--- K-means centroids for distance awareness ---")

    z_pool_all = _extract_z_pool(params, train_batch)

    centroids_np, labels = kmeans2(z_pool_all, N_CENTROIDS, minit="points", iter=20)  # Explicit constant: 20 K-means iterations
    logger.info(f"  K-means: {N_CENTROIDS} centroids from {n_train} z_pool vectors")
    logger.info(f"  Cluster sizes: {np.bincount(labels, minlength=N_CENTROIDS).tolist()}")

    dists_all = np.array([np.min(np.sum((z_pool_all[i] - centroids_np) ** 2, axis=-1)) for i in range(n_train)])
    d_ref_val = float(np.mean(dists_all))
    logger.info(f"  d_ref (mean nearest-centroid distance): {d_ref_val:.4f}")

    d_ref_raw_val = float(np.log(np.exp(d_ref_val) - 1.0)) if d_ref_val > 0.01 else 0.0
    params["centroids"] = jnp.array(centroids_np)
    params["d_ref_raw"] = jnp.array(d_ref_raw_val)


# =============================================================================
# MAIN
# =============================================================================

def _run_ood_evaluation(norm_mean, norm_std, step_fn, opt, val_metrics):
    """Run OOD evaluation for all species in OOD_EVAL_SPECIES."""
    logger.info(f"\n{'='*60}")
    logger.info(f"OUT-OF-DISTRIBUTION EVALUATION")
    logger.info(f"{'='*60}")

    ood_results = []
    for sp in OOD_EVAL_SPECIES:
        result = evaluate_species_ood(sp, norm_mean, norm_std, step_fn, opt)
        ood_results.append(result)

    logger.info(f"\n--- OOD Summary ---")
    for r in ood_results:
        if r["ood_mae"] is not None:
            logger.info(f"  {r['species']:8s}: OOD MAE = {r['ood_mae']:.3f} mS/cm "
                       f"(train MAE = {r['train_mae']:.3f}, n_ood = {r['n_ood']})")

    logger.info(f"\n--- Architecture Comparison ---")
    logger.info(f"{'Architecture':<35s} {'Val MAE':>8s} {'FEC OOD':>8s} {'VC OOD':>8s} {'LiFSI OOD':>10s}")
    logger.info(f"{'-'*71}")
    logger.info(f"{'Run 8 gated linear':35s} {'0.458':>8s} {'0.603':>8s} {'0.462':>8s} {'2.496':>10s}")
    logger.info(f"{'Per-species gated (incumbent)':35s} {'0.417':>8s} {'~0.60':>8s} {'~0.46':>8s} {'~1.604':>10s}")

    ood_by_species = {r["species"]: r["ood_mae"] for r in ood_results if r["ood_mae"] is not None}
    ood_strs = [f"{ood_by_species[sp]:.3f}" if sp in ood_by_species else "N/A" for sp in OOD_EVAL_SPECIES]
    logger.info(
        f"{'Equivariant property-space':35s} "
        f"{val_metrics['mae_mS_cm']:>8.3f} "
        + " ".join(f"{s:>8s}" for s in ood_strs)
    )

    return ood_results


def main():
    """Train equivariant MolSets Set Transformer and evaluate."""
    logger.info("=" * 70)
    logger.info("Equivariant Property-Space Set Transformer — Conductivity Prediction")
    logger.info("=" * 70)

    all_species = set()
    for entry in _DATA_ORIGINAL + _DATA_CALISOL:
        if "conductivity_mS_cm" not in entry["properties"]:
            continue
        r = entry["recipe"]
        for k in ["salts", "solvents", "additives"]:
            all_species.update(r[k].keys())

    all_species = sorted(all_species)
    logger.info(f"All species in data ({len(all_species)}): {all_species}")

    norm_mean, norm_std = compute_normalization_stats(all_species)
    logger.info(f"Property vector dimension: {D_INPUT}")
    logger.info(f"Normalization mean: {norm_mean}")
    logger.info(f"Normalization std: {norm_std}")

    train_batch, val_batch = prepare_molset_data(norm_mean, norm_std)

    # OOD proxy split
    all_entries = _load_all_sources()
    recipe_groups_full: Dict[tuple, list] = defaultdict(list)
    for recipe, sigma, temp, source in all_entries:
        key = (_recipe_key(recipe), round(temp, 0))
        recipe_groups_full[key].append((sigma, temp, recipe, source))

    ood_proxy_keys = set()
    for (rkey, T_round), measurements in recipe_groups_full.items():
        recipe = measurements[0][2]
        all_sp = list(recipe["salts"].keys()) + \
                 list(recipe["solvents"].keys()) + \
                 list(recipe["additives"].keys())
        if OOD_PROXY_SPECIES in all_sp:
            ood_proxy_keys.add(rkey)

    train_core_idx = []
    ood_proxy_idx = []
    for i, rk in enumerate(train_batch.recipe_keys):
        if rk in ood_proxy_keys:
            ood_proxy_idx.append(i)
        else:
            train_core_idx.append(i)

    def _subset_batch(batch: MolSetBatch, indices: list) -> MolSetBatch:
        idx = np.array(indices)
        return MolSetBatch(
            species_props=batch.species_props[idx],
            raw_props=batch.raw_props[idx],
            fracs=batch.fracs[idx],
            mask=batch.mask[idx],
            temperature_K=batch.temperature_K[idx],
            log_sigma=batch.log_sigma[idx],
            weights=batch.weights[idx],
            recipe_keys=[batch.recipe_keys[i] for i in indices],
        )

    train_core = _subset_batch(train_batch, train_core_idx)
    ood_proxy_batch = _subset_batch(train_batch, ood_proxy_idx) if ood_proxy_idx else None

    logger.info(f"Train core (no {OOD_PROXY_SPECIES}): {len(train_core_idx)}, "
                f"OOD proxy ({OOD_PROXY_SPECIES}): {len(ood_proxy_idx)}, Val: {len(val_batch.recipe_keys)}")

    mix_mean, mix_std = compute_mix_physics_stats(train_core)
    logger.info(f"Mixture physics mean: {mix_mean}")
    logger.info(f"Mixture physics std:  {mix_std}")

    params = init_params(random.PRNGKey(SEED_MAIN), mix_mean, mix_std)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"\nModel parameters: {n_params:,}")

    # Enumerate new mechanism params
    mechanism_params = {
        "metric_L": params["metric_L"].size,
        "metric_beta": params["metric_beta"].size,
        "metric_gamma_raw": params["metric_gamma_raw"].size,
        "gate_mlp": params["gate_mlp_w"].size + params["gate_mlp_b"].size,
        "gate_out": params["gate_out_w"].size + params["gate_out_b"].size,
        "alpha_gate": params["alpha_gate"].size,
        "pair_proj_src": params["pair_proj_src_w"].size + params["pair_proj_src_b"].size,
        "pair_proj_dst": params["pair_proj_dst_w"].size + params["pair_proj_dst_b"].size,
        "pair_proj_out": params["pair_proj_out_w"].size + params["pair_proj_out_b"].size,
        "alpha_pair": params["alpha_pair"].size,
        "dissoc_channel": (params["dissoc_h_w"].size + params["dissoc_h_b"].size +
                          params["dissoc_out_w"].size + params["dissoc_out_b"].size + 1),
        "visc_channel": (params["visc_h_w"].size + params["visc_h_b"].size +
                        params["visc_out_w"].size + params["visc_out_b"].size + 1),
        "interact_channel": (params["interact_h_w"].size + params["interact_h_b"].size +
                            params["interact_out_w"].size + params["interact_out_b"].size + 1),
        "centroids": params["centroids"].size,
        "d_ref_raw": params["d_ref_raw"].size,
    }
    logger.info(f"New mechanism params breakdown:")
    total_new = 0
    for name, count in mechanism_params.items():
        logger.info(f"  {name:20s}: {count:5d}")
        total_new += count
    logger.info(f"  {'TOTAL NEW':20s}: {total_new:5d}")

    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, WARMUP_STEPS)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, N_STEPS - WARMUP_STEPS)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [WARMUP_STEPS])
    opt = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = opt.init(params)
    step_fn = make_train_step(opt)

    ja = train_core.jax_arrays()
    batch_tuple = (ja["props"], ja["raw"], ja["fracs"], ja["mask"], ja["temps"],
                   ja["log_sigma"], ja["weights"])

    swa_start_step = int(N_STEPS * SWA_START_FRAC)
    swa_params_sum = None
    swa_count = 0

    logger.info(f"\nTraining for {N_STEPS} steps (SWA from step {swa_start_step})...")
    logger.info(f"Correction penalty: LAMBDA_CORRECTION={LAMBDA_CORRECTION}")
    best_val_mae = float("inf")
    best_val_params = params
    best_val_step = 0
    best_ood_mae = float("inf")
    best_ood_params = params
    best_ood_step = 0
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

        if (step + 1) % LOG_EVERY == 0 or step == 0:
            val_mae = compute_val_mae(params, val_batch)
            train_mae = compute_val_mae(params, train_core)

            val_marker = ""
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_params = params
                best_val_step = step + 1
                val_marker = " ***"

            ood_marker = ""
            ood_str = "N/A"
            if ood_proxy_batch is not None:
                ood_mae = compute_val_mae(params, ood_proxy_batch)
                ood_str = f"{ood_mae:.3f}"
                if ood_mae < best_ood_mae:
                    best_ood_mae = ood_mae
                    best_ood_params = params
                    best_ood_step = step + 1
                    ood_marker = " OOD*"

            elapsed = time.time() - t0

            alpha_gate_val = float(params["alpha_gate"])
            alpha_pair_val = float(params["alpha_pair"])
            metric_beta_val = float(params["metric_beta"])

            logger.info(
                f"Step {step+1:5d} | loss={float(loss):.4f} | "
                f"train={train_mae:.3f} | val={val_mae:.3f} | ood_proxy={ood_str} mS/cm | "
                f"best_val={best_val_mae:.3f}@{best_val_step} | best_ood={best_ood_mae:.3f}@{best_ood_step} | "
                f"a_gate={alpha_gate_val:.4f} a_pair={alpha_pair_val:.4f} m_beta={metric_beta_val:.4f} | "
                f"{elapsed:.1f}s{val_marker}{ood_marker}"
            )

    # SWA
    if swa_count > 1:
        swa_params = jax.tree.map(lambda x: x / swa_count, swa_params_sum)
        swa_val_mae = compute_val_mae(swa_params, val_batch)
        swa_train_mae = compute_val_mae(swa_params, train_core)
        logger.info(f"\nSWA ({swa_count} checkpoints from step {swa_start_step}):")
        logger.info(f"  SWA val MAE = {swa_val_mae:.3f} mS/cm (best single = {best_val_mae:.3f})")
        logger.info(f"  SWA train MAE = {swa_train_mae:.3f} mS/cm")
        if swa_val_mae < best_val_mae:
            best_val_params = swa_params
            best_val_mae = swa_val_mae
            best_val_step = -1
            logger.info(f"  SWA WINS for val — using averaged weights")
        if ood_proxy_batch is not None:
            swa_ood_mae = compute_val_mae(swa_params, ood_proxy_batch)
            logger.info(f"  SWA OOD proxy MAE = {swa_ood_mae:.3f} mS/cm (best single = {best_ood_mae:.3f})")
            if swa_ood_mae < best_ood_mae:
                best_ood_params = swa_params
                best_ood_mae = swa_ood_mae
                best_ood_step = -1
                logger.info(f"  SWA WINS for OOD — using averaged weights")

    use_params = best_ood_params if ood_proxy_batch is not None else best_val_params
    use_step = best_ood_step if ood_proxy_batch is not None else best_val_step

    # K-means centroid computation
    _compute_and_inject_centroids(use_params, train_core)

    # Fine-tune with centroids active
    logger.info(f"  Fine-tuning {CENTROID_FINETUNE_STEPS} steps with centroids in place...")
    ft_opt_state = opt.init(use_params)
    ft_rng = random.PRNGKey(SEED_MAIN + 2)  # Explicit constant: distinct seed for centroid fine-tuning
    for ft_step in range(CENTROID_FINETUNE_STEPS):
        ft_rng, ft_key = random.split(ft_rng)
        use_params, ft_opt_state, ft_loss = step_fn(use_params, ft_opt_state, batch_tuple, ft_key)
        if (ft_step + 1) % LOG_EVERY == 0:
            ft_val = compute_val_mae(use_params, val_batch)
            ft_ood = compute_val_mae(use_params, ood_proxy_batch) if ood_proxy_batch else 0.0
            logger.info(f"  FT step {ft_step+1}: loss={float(ft_loss):.4f}, val={ft_val:.3f}, ood={ft_ood:.3f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS (using OOD-best@{use_step} + centroids + {CENTROID_FINETUNE_STEPS}-step finetune)")
    logger.info(f"{'='*60}")

    train_metrics = compute_metrics(use_params, train_core)
    val_metrics = compute_metrics(use_params, val_batch)

    logger.info(f"Train: MAE={train_metrics['mae_mS_cm']:.3f} mS/cm, "
                f"RMSE={train_metrics['rmse_mS_cm']:.3f}, "
                f"bias={train_metrics['bias_mS_cm']:.3f}, "
                f"MAPE={train_metrics['mape_pct']:.1f}%")
    logger.info(f"Val:   MAE={val_metrics['mae_mS_cm']:.3f} mS/cm, "
                f"RMSE={val_metrics['rmse_mS_cm']:.3f}, "
                f"bias={val_metrics['bias_mS_cm']:.3f}, "
                f"MAPE={val_metrics['mape_pct']:.1f}%")
    logger.info(f"\nTrain/Val ratio: {train_metrics['mae_mS_cm']/val_metrics['mae_mS_cm']:.2f}")

    logger.info(f"\n--- Baselines ---")
    logger.info(f"XGB (in-distribution only): 0.26 mS/cm")
    logger.info(f"MLP (fixed 52-d features):  0.591 mS/cm")
    logger.info(f"Per-species gated (incumbent): ~0.417 mS/cm")
    logger.info(f"Equivariant (this model):   {val_metrics['mae_mS_cm']:.3f} mS/cm")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mol_set_sigma_equivariant.pkl")
    save_model(use_params, norm_mean, norm_std, model_path)
    logger.info(f"Model saved: {model_path}")

    # Speed test
    logger.info(f"\n--- Speed Test (JIT-compiled) ---")
    test_props_s = jnp.array(val_batch.species_props[0])
    test_raw_s = jnp.array(val_batch.raw_props[0])
    test_fracs_s = jnp.array(val_batch.fracs[0])
    test_mask_s = jnp.array(val_batch.mask[0])
    test_temp_s = jnp.array(val_batch.temperature_K[0])

    warmup_result = _forward_single_eval(
        use_params, test_props_s, test_raw_s, test_fracs_s, test_mask_s, test_temp_s)
    warmup_result.block_until_ready()

    n_calls = 10000  # Explicit constant: timing iterations
    t0_speed = time.time()
    for _ in range(n_calls):
        result = _forward_single_eval(
            use_params, test_props_s, test_raw_s, test_fracs_s, test_mask_s, test_temp_s)
    result.block_until_ready()
    elapsed_jit = (time.time() - t0_speed) / n_calls * 1000
    logger.info(f"JIT single-recipe: {elapsed_jit:.4f} ms/recipe ({n_calls} calls)")

    # Gradient check
    logger.info(f"\n--- Gradient Check ---")
    test_props = jnp.array(val_batch.species_props[0])
    test_raw = jnp.array(val_batch.raw_props[0])
    test_fracs = jnp.array(val_batch.fracs[0])
    test_mask = jnp.array(val_batch.mask[0])
    test_temp = jnp.array(val_batch.temperature_K[0])

    grad_fn = jax.grad(lambda p, sp, rp, f, m, t, k, d: forward_single(p, sp, rp, f, m, t, k, d)[0],
                        argnums=3)
    frac_grads = grad_fn(use_params, test_props, test_raw, test_fracs, test_mask, test_temp,
                         random.PRNGKey(0), 0.0)
    active_grads = frac_grads * test_mask

    logger.info(f"d_log(sigma)/d_x_i for first val recipe:")
    recipe = val_batch.recipe_keys[0]
    species_list = []
    for role_key in [0, 1, 2]:
        for sp_frac in recipe[role_key]:
            species_list.append(sp_frac[0])
    for j, sp in enumerate(species_list[:N_MAX_SPECIES]):
        if float(test_mask[j]) > 0:
            logger.info(f"  {sp:8s}: grad = {float(active_grads[j]):+.4f}")

    # Confidence statistics
    logger.info(f"\n--- Confidence Statistics ---")
    val_z_pools = _extract_z_pool(use_params, val_batch)
    centroids_np = np.array(use_params["centroids"])
    d_ref_val = float(jax.nn.softplus(use_params["d_ref_raw"]))
    val_confs = []
    for i in range(len(val_batch.recipe_keys)):
        dists = np.min(np.sum((val_z_pools[i:i+1] - centroids_np) ** 2, axis=-1))
        conf = 1.0 / (1.0 + dists / max(d_ref_val, 1e-8))
        val_confs.append(conf)
    val_confs = np.array(val_confs)
    logger.info(f"  Val confidence: mean={val_confs.mean():.4f}, std={val_confs.std():.4f}, "
                f"min={val_confs.min():.4f}, max={val_confs.max():.4f}")

    # OOD evaluation
    _run_ood_evaluation(norm_mean, norm_std, step_fn, opt, val_metrics)


if __name__ == "__main__":
    main()
