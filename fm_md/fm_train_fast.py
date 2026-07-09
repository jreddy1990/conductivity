"""Fast training entry point for the flow-matching conductivity model.

Builds a SMILES->MolecularGraph cache once, stacks each row into
fixed-shape JAX arrays, and runs vmap-over-batch + vmap-over-species
+ jit on the gradient step. Cuts per-step cost by roughly 10x versus
flow_matching_conductivity.train_model on a single CPU.

Entry: python -m conductivity.fm_train_fast
Env vars: SUBSET_SIZE, TRAIN_EPOCHS, TRAIN_BATCH (override defaults).
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")
import control_framework.jax_m4_tuning  # noqa: F401

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, random, value_and_grad, vmap
import optax

from conductivity.flow_matching_conductivity import (
    CLEAN_DATASET_PATH, DATA_DIR, HELDOUT_SOLVENTS, HELDOUT_ANIONS,
    SEED_SPLIT, SEED_TRAIN, BATCH_SIZE, N_EPOCHS, LR_PEAK, LR_FLOOR,
    WARMUP_STEPS, WEIGHT_DECAY, GRAD_CLIP_NORM,
    LAMBDA_FM, LAMBDA_SIGMA, N_FM_SAMPLES_PER_COMP,
    K_SPECTRUM, D_MOL, D_COMP, D_FM_TOKEN, D_HEAD, D_ATOM, D_BOND,
    N_GNN_LAYERS, N_ATTN_LAYERS, N_ATTN_HEADS, N_FM_LAYERS,
    N_FOURIER_FRACTION, N_FOURIER_TEMPERATURE, N_FOURIER_FLOWTIME,
    MAX_SPECIES, MAX_ATOMS, MAX_BONDS, ODE_STEPS,
    SPECTRUM_LAMBDA_BOUND, SPECTRUM_ALPHA_BOUND,
    LabeledRow, NormalizationStats, ModelBundle,
    compute_normalization, smiles_to_graph, split_dataset,
    init_all_params, fourier_features, EMPTY_GRAPH,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


CHECKPOINT_DIR = DATA_DIR / "checkpoints"
VAL_FRACTION = 0.10                   # Explicit constant: held-out validation share
LOG_INTERVAL_STEPS = 10               # Explicit constant: log every 10 optimizer steps
PER_EPOCH_EVAL_MAX_ROWS = 100         # Explicit constant: validation sample size per epoch

SUBSET_SIZE = int(os.environ.get("SUBSET_SIZE", "0"))
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", str(N_EPOCHS)))
TRAIN_BATCH = int(os.environ.get("TRAIN_BATCH", str(BATCH_SIZE)))


# =============================================================================
# BATCHED ARRAY LAYOUT (everything pre-stacked for vmap + jit)
# =============================================================================


class RowArrays(NamedTuple):
    """Per-row tensors as a pytree; vmap maps over the struct."""
    atom_features: jnp.ndarray         # (MAX_SPECIES, MAX_ATOMS, D_ATOM)
    bond_features: jnp.ndarray         # (MAX_SPECIES, MAX_BONDS, D_BOND)
    bond_src: jnp.ndarray              # (MAX_SPECIES, MAX_BONDS) int32
    bond_dst: jnp.ndarray              # (MAX_SPECIES, MAX_BONDS) int32
    atom_masks: jnp.ndarray            # (MAX_SPECIES, MAX_ATOMS)
    bond_masks: jnp.ndarray            # (MAX_SPECIES, MAX_BONDS)
    mole_fractions: jnp.ndarray        # (MAX_SPECIES,)
    species_mask: jnp.ndarray          # (MAX_SPECIES,)
    temperature: jnp.ndarray           # scalar


class BatchedArrays(NamedTuple):
    """Per-batch tensors. Convert to row-views via `RowArrays(*batch[:-2])`."""
    atom_features: jnp.ndarray
    bond_features: jnp.ndarray
    bond_src: jnp.ndarray
    bond_dst: jnp.ndarray
    atom_masks: jnp.ndarray
    bond_masks: jnp.ndarray
    mole_fractions: jnp.ndarray
    species_masks: jnp.ndarray
    temperatures: jnp.ndarray
    log_sigma_labels: jnp.ndarray
    sample_weights: jnp.ndarray


def build_smiles_cache(rows: List[LabeledRow]) -> Dict[str, Dict]:
    """Pre-parse every unique SMILES once into a dict of numpy arrays."""
    unique = set(s for r in rows for s in r.smiles_list)
    logger.info("Caching %d unique SMILES -> molecular graphs", len(unique))
    cache = {}
    for sm in unique:
        g = smiles_to_graph(sm)
        cache[sm] = {
            "atom_features": g.atom_features,
            "bond_features": g.bond_features,
            "bond_src": g.bond_src,
            "bond_dst": g.bond_dst,
            "atom_mask": g.atom_mask,
            "bond_mask": g.bond_mask,
        }
    # Add empty graph for padding slots
    cache[None] = {
        "atom_features": EMPTY_GRAPH.atom_features,
        "bond_features": EMPTY_GRAPH.bond_features,
        "bond_src": EMPTY_GRAPH.bond_src,
        "bond_dst": EMPTY_GRAPH.bond_dst,
        "atom_mask": EMPTY_GRAPH.atom_mask,
        "bond_mask": EMPTY_GRAPH.bond_mask,
    }
    return cache


def row_to_arrays(row: LabeledRow, cache: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """Pack one row into per-row arrays, padded to MAX_SPECIES."""
    n_sp = len(row.smiles_list)
    if n_sp > MAX_SPECIES:
        raise ValueError(f"Row has {n_sp} > MAX_SPECIES={MAX_SPECIES}")
    atom_feats = np.zeros((MAX_SPECIES, MAX_ATOMS, D_ATOM))
    bond_feats = np.zeros((MAX_SPECIES, MAX_BONDS, D_BOND))
    bond_src = np.zeros((MAX_SPECIES, MAX_BONDS), dtype=np.int32)
    bond_dst = np.zeros((MAX_SPECIES, MAX_BONDS), dtype=np.int32)
    atom_masks = np.zeros((MAX_SPECIES, MAX_ATOMS))
    bond_masks = np.zeros((MAX_SPECIES, MAX_BONDS))
    mole_fracs = np.zeros(MAX_SPECIES)
    species_mask = np.zeros(MAX_SPECIES)
    for i in range(MAX_SPECIES):
        if i < n_sp:
            sm = row.smiles_list[i]
            g = cache[sm]
            mole_fracs[i] = row.mole_fractions[i]
            species_mask[i] = 1.0
        else:
            g = cache[None]
        atom_feats[i] = g["atom_features"]
        bond_feats[i] = g["bond_features"]
        bond_src[i] = g["bond_src"]
        bond_dst[i] = g["bond_dst"]
        atom_masks[i] = g["atom_mask"]
        bond_masks[i] = g["bond_mask"]
    return {
        "atom_features": atom_feats,
        "bond_features": bond_feats,
        "bond_src": bond_src,
        "bond_dst": bond_dst,
        "atom_masks": atom_masks,
        "bond_masks": bond_masks,
        "mole_fractions": mole_fracs,
        "species_mask": species_mask,
        "temperature": row.temperature_K,
        "log_sigma": float(np.log(row.sigma_mScm)),
        "sample_weight": 1.0 / (1.0 + row.sigma_uncertainty_log ** 2),
    }


def stack_batch(per_row: List[Dict], norm: NormalizationStats) -> BatchedArrays:
    """Stack per-row dicts into a BatchedArrays (still numpy at this stage)."""
    def stk(key):
        return np.stack([r[key] for r in per_row], axis=0)
    return BatchedArrays(
        atom_features=jnp.asarray(stk("atom_features")),
        bond_features=jnp.asarray(stk("bond_features")),
        bond_src=jnp.asarray(stk("bond_src")),
        bond_dst=jnp.asarray(stk("bond_dst")),
        atom_masks=jnp.asarray(stk("atom_masks")),
        bond_masks=jnp.asarray(stk("bond_masks")),
        mole_fractions=jnp.asarray(stk("mole_fractions")),
        species_masks=jnp.asarray(stk("species_mask")),
        temperatures=jnp.asarray(np.array([r["temperature"] for r in per_row])),
        log_sigma_labels=jnp.asarray(np.array([r["log_sigma"] for r in per_row])),
        sample_weights=jnp.asarray(np.array([r["sample_weight"] for r in per_row])),
    )


# =============================================================================
# JAX-NATIVE MODEL FORWARD (pure tensors; vmap-friendly)
# =============================================================================


def gnn_forward_arrays(
    params: Dict,
    atom_feat: jnp.ndarray,        # (MAX_ATOMS, D_ATOM)
    bond_feat: jnp.ndarray,        # (MAX_BONDS, D_BOND)
    bond_src: jnp.ndarray,         # (MAX_BONDS,) int32
    bond_dst: jnp.ndarray,         # (MAX_BONDS,) int32
    atom_mask: jnp.ndarray,        # (MAX_ATOMS,)
    bond_mask: jnp.ndarray,        # (MAX_BONDS,)
) -> jnp.ndarray:
    """Molecular GNN forward on a single molecule; sum-pool to R^D_MOL."""
    h = jax.nn.silu(atom_feat @ params["atom_embed_w"] + params["atom_embed_b"])
    h = h * atom_mask[:, None]
    for L in range(N_GNN_LAYERS):
        edge_input = jnp.concatenate([h[bond_src], h[bond_dst], bond_feat], axis=-1)
        msg = jax.nn.silu(edge_input @ params[f"edge_w_{L}"] + params[f"edge_b_{L}"])
        msg = msg * bond_mask[:, None]
        agg = jnp.zeros((MAX_ATOMS, D_MOL)).at[bond_dst].add(msg)
        node_input = jnp.concatenate([h, agg], axis=-1)
        gated = jax.nn.silu(
            node_input @ params[f"node_w1_{L}"] + params[f"node_b1_{L}"]
        )
        delta = gated @ params[f"node_w2_{L}"] + params[f"node_b2_{L}"]
        h = h + delta * atom_mask[:, None]
    return jnp.sum(h * atom_mask[:, None], axis=0)


def composition_encoder_arrays(
    mol_gnn_params: Dict, attn_params: Dict,
    row: RowArrays, norm: NormalizationStats,
) -> jnp.ndarray:
    """Encode (composition, T) -> z in R^D_COMP."""
    m = vmap(gnn_forward_arrays, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        mol_gnn_params, row.atom_features, row.bond_features,
        row.bond_src, row.bond_dst, row.atom_masks, row.bond_masks,
    )                                              # (MAX_SPECIES, D_MOL)
    frac_features = fourier_features(row.mole_fractions, N_FOURIER_FRACTION)
    species_input = jnp.concatenate([m, frac_features], axis=-1)
    tokens = jax.nn.silu(
        species_input @ attn_params["token_in_w"] + attn_params["token_in_b"]
    )
    tokens = tokens * row.species_mask[:, None]
    for L in range(N_ATTN_LAYERS):
        qkv = tokens @ attn_params[f"attn_qkv_{L}"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(MAX_SPECIES, N_ATTN_HEADS, D_HEAD)
        k = k.reshape(MAX_SPECIES, N_ATTN_HEADS, D_HEAD)
        v = v.reshape(MAX_SPECIES, N_ATTN_HEADS, D_HEAD)
        attn = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(D_HEAD)
        attn = attn + (1.0 - row.species_mask[None, None, :]) * -1e9
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("hij,jhd->ihd", attn, v).reshape(MAX_SPECIES, D_COMP)
        tokens = tokens + (out @ attn_params[f"attn_proj_{L}"]) * row.species_mask[:, None]
        ff = jax.nn.silu(
            tokens @ attn_params[f"ff_w1_{L}"] + attn_params[f"ff_b1_{L}"]
        )
        ff = ff @ attn_params[f"ff_w2_{L}"] + attn_params[f"ff_b2_{L}"]
        tokens = (tokens + ff) * row.species_mask[:, None]
    scores = (tokens @ attn_params["pool_w"]) @ attn_params["pool_query"]
    scores = scores + (1.0 - row.species_mask) * -1e9
    alpha = jax.nn.softmax(scores)
    pooled = jnp.sum(tokens * alpha[:, None], axis=0)
    T_norm = (row.temperature - norm.T_mean) / max(norm.T_std, 1e-8)
    T_feat = fourier_features(T_norm[None], N_FOURIER_TEMPERATURE).flatten()
    z_raw = pooled + T_feat @ attn_params["T_proj_w"]
    # LayerNorm: ||z|| ~O(sqrt(D_COMP)), avoiding tanh saturation downstream.
    mu = jnp.mean(z_raw)
    sd = jnp.sqrt(jnp.var(z_raw) + 1e-6)
    return ((z_raw - mu) / sd) * attn_params["z_ln_scale"] + attn_params["z_ln_shift"]


def fm_velocity_arrays(
    params: Dict, xi: jnp.ndarray, s: jnp.ndarray, z: jnp.ndarray,
) -> jnp.ndarray:
    """FM velocity field u_theta(xi, s, z)."""
    tokens_in = xi.reshape(K_SPECTRUM, 2)
    tokens = jax.nn.silu(tokens_in @ params["token_in_w"] + params["token_in_b"])
    tokens = tokens + params["token_pos"]
    s_feat = fourier_features(s[None], N_FOURIER_FLOWTIME).flatten()
    tokens = tokens + (s_feat @ params["s_proj_w"])[None, :]
    z_token = z @ params["z_proj_w"]
    for L in range(N_FM_LAYERS):
        qkv = tokens @ params[f"fm_qkv_{L}"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(K_SPECTRUM, N_ATTN_HEADS, D_HEAD)
        k = k.reshape(K_SPECTRUM, N_ATTN_HEADS, D_HEAD)
        v = v.reshape(K_SPECTRUM, N_ATTN_HEADS, D_HEAD)
        attn = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(D_HEAD)
        attn = jax.nn.softmax(attn, axis=-1)
        self_out = jnp.einsum("hij,jhd->ihd", attn, v).reshape(K_SPECTRUM, D_FM_TOKEN)
        tokens = tokens + self_out @ params[f"fm_proj_{L}"]
        kv = z_token @ params[f"fm_xattn_kv_{L}"]
        k2, v2 = jnp.split(kv, 2, axis=-1)
        score = tokens @ k2 / jnp.sqrt(D_FM_TOKEN)
        weight = jax.nn.softmax(score[:, None], axis=-1)
        tokens = tokens + weight * v2[None, :]
        tokens = tokens + jax.nn.silu(tokens @ params[f"fm_ff_{L}"])
    return (tokens @ params["fm_out_w"] + params["fm_out_b"]).flatten()


def target_spectrum_arrays(params: Dict, z: jnp.ndarray) -> jnp.ndarray:
    raw = z @ params["target_w1"] + params["target_b1"]
    lambda_part = jnp.tanh(raw[:K_SPECTRUM]) * SPECTRUM_LAMBDA_BOUND
    alpha_part = jnp.tanh(raw[K_SPECTRUM:]) * SPECTRUM_ALPHA_BOUND
    return jnp.concatenate([lambda_part, alpha_part])


def green_kubo_arrays(xi: jnp.ndarray, offset: jnp.ndarray) -> jnp.ndarray:
    lambda_log = xi[:K_SPECTRUM]
    alpha_log = xi[K_SPECTRUM:]
    return jax.scipy.special.logsumexp(alpha_log - lambda_log) + offset


def integrate_fm_ode_arrays(
    fm_params: Dict, xi_0: jnp.ndarray, z: jnp.ndarray, n_steps: int,
) -> jnp.ndarray:
    ds = 1.0 / n_steps

    def body(step_idx, xi):
        s = step_idx * ds + ds / 2.0
        return xi + ds * fm_velocity_arrays(fm_params, xi, jnp.asarray(s), z)

    return lax.fori_loop(0, n_steps, body, xi_0)


# =============================================================================
# BATCHED LOSS (JIT-COMPILED)
# =============================================================================


def per_row_loss(
    params: Dict, norm: NormalizationStats, row: RowArrays,
    log_sigma_label: jnp.ndarray, weight: jnp.ndarray, key: jnp.ndarray,
) -> jnp.ndarray:
    """Loss for a single row."""
    z = composition_encoder_arrays(params["mol_gnn"], params["attn"], row, norm)
    xi_target = target_spectrum_arrays(params["fm"], z)
    log_sigma_pred = green_kubo_arrays(xi_target, params["fm"]["log_sigma_offset"])
    sigma_loss = weight * (log_sigma_pred - log_sigma_label) ** 2

    # FM regression loss averaged over N_FM_SAMPLES_PER_COMP draws.
    # stop_gradient on xi_target: the FM velocity net learns to chase the
    # target spectrum, but the L_sigma supervision is what shapes the target.
    # Without stop_gradient the FM loss can shrink xi_target toward zero
    # (mode collapse), wrecking sigma predictions.
    xi_target_sg = lax.stop_gradient(xi_target)
    keys = random.split(key, 2 * N_FM_SAMPLES_PER_COMP).reshape(
        N_FM_SAMPLES_PER_COMP, 2, -1,
    )

    def per_draw(pair):
        xi_0 = random.normal(pair[0], (2 * K_SPECTRUM,))
        s = random.uniform(pair[1], minval=0.0, maxval=1.0)
        xi_s = (1.0 - s) * xi_0 + s * xi_target_sg
        u_pred = fm_velocity_arrays(params["fm"], xi_s, jnp.asarray(s), z)
        u_true = xi_target_sg - xi_0
        # Normalize by the natural variance of the target velocity so the FM
        # loss is comparable in scale to L_sigma. Without this, L_FM dominates.
        var = (SPECTRUM_LAMBDA_BOUND ** 2 + SPECTRUM_ALPHA_BOUND ** 2)
        return jnp.mean((u_pred - u_true) ** 2) / var

    fm_loss = jnp.mean(vmap(per_draw)(keys))
    return LAMBDA_FM * fm_loss + LAMBDA_SIGMA * sigma_loss


LAMBDA_VAR_Z = 0.1                 # Explicit constant: variance regulariser weight on z (forces per-dim var ~1 across batch, prevents composition collapse onto a 1-D direction)
LAMBDA_VAR_OUT = 0.1               # Explicit constant: output-std regulariser weight; penalises predictions collapsing to the data mean


def batched_loss(
    params: Dict, norm: NormalizationStats, batch: BatchedArrays, key: jnp.ndarray,
) -> jnp.ndarray:
    B = batch.atom_features.shape[0]
    keys = random.split(key, B)
    rows = RowArrays(
        atom_features=batch.atom_features, bond_features=batch.bond_features,
        bond_src=batch.bond_src, bond_dst=batch.bond_dst,
        atom_masks=batch.atom_masks, bond_masks=batch.bond_masks,
        mole_fractions=batch.mole_fractions, species_mask=batch.species_masks,
        temperature=batch.temperatures,
    )

    # Per-row FM + sigma loss
    losses = vmap(per_row_loss, in_axes=(None, None, 0, 0, 0, 0))(
        params, norm, rows, batch.log_sigma_labels, batch.sample_weights, keys,
    )

    # Variance regulariser on z(c) across the batch. Penalises composition
    # collapse (audit found cosine similarity 0.96+ between random
    # compositions before this regulariser was added).
    zs = vmap(composition_encoder_arrays, in_axes=(None, None, 0, None))(
        params["mol_gnn"], params["attn"], rows, norm,
    )                                                     # (B, D_COMP)
    L_var_z = jnp.mean((jnp.std(zs, axis=0) - 1.0) ** 2)

    # Output-std regulariser: predicted log-sigma std should match label std.
    # Penalises the degenerate "predict the data mean" minimum.
    def _predict_log_sigma_from_z(z):
        xi = target_spectrum_arrays(params["fm"], z)
        return green_kubo_arrays(xi, params["fm"]["log_sigma_offset"])
    log_preds = vmap(_predict_log_sigma_from_z)(zs)
    L_var_out = (jnp.std(log_preds) - jnp.std(batch.log_sigma_labels)) ** 2

    return jnp.mean(losses) + LAMBDA_VAR_Z * L_var_z + LAMBDA_VAR_OUT * L_var_out


def make_step_fn(norm: NormalizationStats, optimizer: optax.GradientTransformation):
    """Build a jit-compiled training step closure over norm + optimizer."""
    grad_fn = value_and_grad(batched_loss, argnums=0)

    @jit
    def step(params, opt_state, batch, key):
        loss_val, grads = grad_fn(params, norm, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return step


def per_row_predict(params, norm, row: RowArrays, key) -> jnp.ndarray:
    z = composition_encoder_arrays(params["mol_gnn"], params["attn"], row, norm)
    xi_0 = random.normal(key, (2 * K_SPECTRUM,))
    xi_1 = integrate_fm_ode_arrays(params["fm"], xi_0, z, ODE_STEPS)
    return green_kubo_arrays(xi_1, params["fm"]["log_sigma_offset"])


def evaluate_batched(params, norm, batch: BatchedArrays, key) -> float:
    B = batch.atom_features.shape[0]
    keys = random.split(key, B)
    rows = RowArrays(
        atom_features=batch.atom_features, bond_features=batch.bond_features,
        bond_src=batch.bond_src, bond_dst=batch.bond_dst,
        atom_masks=batch.atom_masks, bond_masks=batch.bond_masks,
        mole_fractions=batch.mole_fractions, species_mask=batch.species_masks,
        temperature=batch.temperatures,
    )
    preds = vmap(per_row_predict, in_axes=(None, None, 0, 0))(
        params, norm, rows, keys,
    )
    return float(jnp.mean((preds - batch.log_sigma_labels) ** 2))


# =============================================================================
# OOD held-out by project taxonomy
# Held-out species are picked from data/species_data.py categories: one solvent,
# one salt, one additive that actually exist in the cleaned data.
# =============================================================================

# Reasonable picks: DEC (linear carbonate solvent, common but not dominant);
# LiTFSI (imide salt distinct from LiPF6/LiFSI in cluster behavior);
# VC (vinylene carbonate additive, important for SEI but a smaller subset).
HELDOUT_NAMES = ("DEC", "LiTFSI", "VC")


def heldout_smiles_set() -> set:
    from conductivity.flow_matching_conductivity import SMILES_BY_SPECIES
    out = set()
    for name in HELDOUT_NAMES:
        if name in SMILES_BY_SPECIES:
            out.add(SMILES_BY_SPECIES[name])
    return out


def split_by_smiles(rows, heldout_smiles, val_frac, seed):
    """Three-way split holding out any row containing any heldout SMILES."""
    rng = np.random.default_rng(seed)
    ood, in_dist = [], []
    for r in rows:
        if heldout_smiles & set(r.smiles_list):
            ood.append(r)
        else:
            in_dist.append(r)
    idx = rng.permutation(len(in_dist))
    n_val = int(len(in_dist) * val_frac)
    val = [in_dist[i] for i in idx[:n_val]]
    train = [in_dist[i] for i in idx[n_val:]]
    return train, val, ood


# =============================================================================
# MAIN
# =============================================================================


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading clean dataset")
    with open(CLEAN_DATASET_PATH, "rb") as f:
        rows = pickle.load(f)
    logger.info("Total rows: %d", len(rows))
    if SUBSET_SIZE > 0 and SUBSET_SIZE < len(rows):
        rng = np.random.default_rng(SEED_SPLIT)
        idx = rng.permutation(len(rows))[:SUBSET_SIZE]
        rows = [rows[i] for i in idx]
        logger.info("Subset to %d rows", len(rows))

    heldout = heldout_smiles_set()
    logger.info("Held-out species (by SMILES): %s -> %s", HELDOUT_NAMES, heldout)
    train_rows, val_rows, ood_rows = split_by_smiles(
        rows, heldout, VAL_FRACTION, SEED_SPLIT,
    )
    logger.info(
        "Split: train=%d val=%d ood=%d", len(train_rows), len(val_rows), len(ood_rows),
    )
    norm = compute_normalization(train_rows)
    logger.info("Norm stats: T_mean=%.2f T_std=%.2f", norm.T_mean, norm.T_std)

    cache = build_smiles_cache(train_rows + val_rows + ood_rows)
    logger.info("Pre-packing rows into arrays")
    t0 = time.time()
    train_per_row = [row_to_arrays(r, cache) for r in train_rows]
    val_per_row = [row_to_arrays(r, cache) for r in val_rows[:PER_EPOCH_EVAL_MAX_ROWS]]
    ood_per_row = [row_to_arrays(r, cache) for r in ood_rows[:PER_EPOCH_EVAL_MAX_ROWS]]
    logger.info("Pre-pack done in %.1fs", time.time() - t0)
    val_batch = stack_batch(val_per_row, norm) if val_per_row else None
    ood_batch = stack_batch(ood_per_row, norm) if ood_per_row else None

    key = random.PRNGKey(SEED_TRAIN)
    init_key, key = random.split(key)
    params = init_all_params(init_key)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info("Model params: %d", n_params)

    n_steps = (len(train_per_row) // TRAIN_BATCH) * TRAIN_EPOCHS
    warmup = min(WARMUP_STEPS, n_steps // 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR_PEAK,
        warmup_steps=warmup, decay_steps=n_steps, end_value=LR_FLOOR,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = optimizer.init(params)
    step_fn = make_step_fn(norm, optimizer)
    logger.info(
        "Training plan: %d epochs x %d steps/epoch x batch %d = %d total steps",
        TRAIN_EPOCHS, len(train_per_row) // TRAIN_BATCH, TRAIN_BATCH, n_steps,
    )

    rng = np.random.default_rng(SEED_TRAIN)
    step = 0
    t_start = time.time()
    for epoch in range(TRAIN_EPOCHS):
        order = rng.permutation(len(train_per_row))
        for b in range(0, len(order) - TRAIN_BATCH + 1, TRAIN_BATCH):
            idx = order[b:b + TRAIN_BATCH]
            batch = stack_batch([train_per_row[i] for i in idx], norm)
            bkey, key = random.split(key)
            params, opt_state, loss_val = step_fn(params, opt_state, batch, bkey)
            step += 1
            if step % LOG_INTERVAL_STEPS == 0 or step == 1:
                logger.info(
                    "step %d epoch %d loss=%.4f elapsed=%.0fs",
                    step, epoch, float(loss_val), time.time() - t_start,
                )
        val_mse = evaluate_batched(params, norm, val_batch, key) if val_batch else float("nan")
        ood_mse = evaluate_batched(params, norm, ood_batch, key) if ood_batch else float("nan")
        logger.info(
            "epoch %d done  val_log_mse=%.4f  ood_log_mse=%.4f  elapsed=%.0fs",
            epoch, val_mse, ood_mse, time.time() - t_start,
        )
        ckpt = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pkl"
        with open(ckpt, "wb") as f:
            pickle.dump(
                {"params": params, "norm_stats": norm, "epoch": epoch,
                 "val_log_mse": val_mse, "ood_log_mse": ood_mse}, f,
            )
        logger.info("Saved %s", ckpt)

    final = DATA_DIR / "fm_conductivity_model.pkl"
    with open(final, "wb") as f:
        pickle.dump({"params": params, "norm_stats": norm}, f)
    logger.info("Training complete. Final model: %s", final)


if __name__ == "__main__":
    main()
