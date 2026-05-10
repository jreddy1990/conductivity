"""
MolSet v2 — Set Transformer with three architectural improvements:

1. Bounded pairwise features (6 scores via tanh/sigmoid/Cauchy — no OOD explosion)
2. Hierarchical attention (intra-class blending + cross-class salt-solvent interaction)
3. Parameterized concentration-dome baseline (replaces WJD 3-5× overshoot)

Entry point: python -m conductivity.mol_set_sigma_v2
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

from constants import MS_CM_TO_S_M as _MS_CM_TO_S_M

from conductivity.mol_set_sigma import (
    PROPERTY_KEYS,
    D_INPUT,
    D_PROP,
    N_MAX_SPECIES,
    IDX_MW,
    IDX_DENSITY,
    IDX_EPSILON,
    IDX_VISCOSITY,
    IDX_DONOR,
    IDX_ACCEPTOR,
    IDX_LAMBDA0,
    IDX_ANION_R,
    IDX_ION_PAIR_BINDING,
    IDX_DIPOLE,
    IDX_COORD_AFFINITY,
    IDX_JONES_DOLE,
    IDX_CATION_R,
    SEED_MAIN,
    SEED_OOD,
    N_STEPS,
    LR_PEAK,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    SWA_START_FRAC,
    SWA_COLLECT_EVERY,
    ATTN_DROPOUT_RATE,
    FFN_DROPOUT_RATE,
    RESID_DROPOUT_RATE,
    PROP_BIAS_ALPHA_INIT,
    USE_PROP_BIAS,
    USE_ATTN_DROPOUT,
    N_MIX_PHYSICS,
    MolSetBatch,
    get_raw_property_vector,
    get_normalized_property_vector,
    compute_normalization_stats,
    prepare_molset_data,
    compute_mix_physics_stats,
    _compute_mixture_physics,
    _multihead_attention,
    _layer_norm,
    _load_all_sources,
    _recipe_key,
    _extract_species_fracs,
    _DATA_ORIGINAL,
    _DATA_CALISOL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# V2 ARCHITECTURE HYPERPARAMETERS
# =============================================================================

D_HIDDEN = 32       # Explicit constant: attention width (same as v1 — improvements come from structure not size)
N_HEADS = 4         # Explicit constant: d_head = D_HIDDEN/N_HEADS = 8
D_FFN = 2 * D_HIDDEN    # Explicit constant: reduced from 4× Vaswani to 2× for data-limited regime
D_ATTN_OUT = 2 * D_HIDDEN   # Explicit constant: z_pool + z_max concatenated
D_MIX_PROJ = D_HIDDEN        # Explicit constant: mixture physics projection bottleneck

N_PAIRWISE_V2 = 6   # Explicit constant: 6 bounded pairwise interaction scores
D_PAIR_PROJ = 8      # Explicit constant: pairwise projection dim (6 features → 8 via GELU)
D_BASE_HIDDEN = 16   # Explicit constant: baseline MLP hidden (7→16→3; ~179 params, cannot memorize 2543 recipes)
N_BASE_FEATURES = 7  # Explicit constant: physical inputs to dome baseline (Lambda0, eps, eta, binding, r_an, JDB, I)

# Gate input: mix_proj + pair_proj + T_scaled
D_GATE_IN_V2 = D_MIX_PROJ + D_PAIR_PROJ + 1

# Named indices into _compute_mixture_physics output vector.
# Positional: derived from group layout in mol_set_sigma.py _MIX_PHYSICS_GROUP_SIZES.
# mix_avg(13) | concentration(4) | transport(5) | kappa_proxy(3) | conc_poly(4) |
# cross_prop(7) | heterogeneity(7) | redox_gas(2) | ion_corr(11) | ...
MIX_IDX_EPS_MIX = 0            # Explicit constant: 1st element of mix_avg group (offset 0)
MIX_IDX_ETA_MIX = 1            # Explicit constant: 2nd element of mix_avg group (offset 0)
MIX_IDX_LAMBDA0_AVG = 8        # Explicit constant: 9th element of mix_avg group (offset 0)
MIX_IDX_BINDING_AVG = 9        # Explicit constant: 10th element of mix_avg group (offset 0)
MIX_IDX_ANION_R_AVG = 10       # Explicit constant: 11th element of mix_avg group (offset 0)
MIX_IDX_JONES_DOLE_B_AVG = 21  # Explicit constant: 5th element of transport group (offset 13+4=17)
MIX_IDX_IONIC_STRENGTH = 45    # Explicit constant: 1st element of ion_corr group (offset 13+4+5+3+4+7+7+2=45)

N_ATTN_LAYERS = 2   # Explicit constant: 2 hierarchical layers (intra-class + cross-class)
N_DROPOUT_KEYS_PER_LAYER = 4  # Explicit constant: attn, resid_attn, ffn, resid_ffn
SOFTPLUS_FLOOR = 0.1  # Explicit constant: positivity floor added after softplus to prevent degenerate zero values
IONIC_GATE_ALPHA_INIT = 0.05  # Explicit constant: sigmoid(0.05×Λ₀) maps salts(Λ₀≥70) to >0.97, solvents(Λ₀=0) to 0.5


# =============================================================================
# IMPROVEMENT 1: BOUNDED PAIRWISE FEATURES
# =============================================================================

def _compute_pairwise_v2(species_props, fracs, mask, pair_params):
    """Compute 6 bounded pairwise interaction features from species property vectors.

    All scores are bounded to [-1,1] or [0,1] by construction via tanh/sigmoid/Cauchy.
    Aggregated via composition-weighted sum: f_k = sum(x_i * x_j * s_k(i,j) * mask_ij).

    Args:
        species_props: (N_max, D_INPUT) raw property vectors
        fracs: (N_max,) composition fractions
        mask: (N_max,) active species mask
        pair_params: dict with learnable reference scales

    Returns:
        (N_PAIRWISE_V2,) array of bounded pairwise features
    """
    w = fracs * mask
    ww = w[:, None] * w[None, :]  # (N, N) outer product weights
    mask_2d = mask[:, None] * mask[None, :]

    dn = species_props[:, IDX_DONOR]
    an = species_props[:, IDX_ACCEPTOR]
    eps = species_props[:, IDX_EPSILON]
    eta = species_props[:, IDX_VISCOSITY]
    lam0 = species_props[:, IDX_LAMBDA0]
    binding = species_props[:, IDX_ION_PAIR_BINDING]
    coord = species_props[:, IDX_COORD_AFFINITY]

    # 1. Donor-Acceptor matching: tanh(DN_i * AN_j / (ref_dn * ref_an))
    ref_dn_an = jax.nn.softplus(pair_params["pair_ref_dn_an"]) + 1.0
    da_score = jnp.tanh(dn[:, None] * an[None, :] / ref_dn_an)
    f_da = jnp.sum(ww * mask_2d * da_score)

    # 2. Dielectric mismatch: 1 - |eps_i - eps_j| / (|eps_i - eps_j| + ref_eps)
    ref_eps = jax.nn.softplus(pair_params["pair_ref_eps"]) + 1.0
    eps_diff = jnp.abs(eps[:, None] - eps[None, :])
    eps_score = 1.0 - eps_diff / (eps_diff + ref_eps)
    f_eps = jnp.sum(ww * mask_2d * eps_score)

    # 3. Viscosity contrast: 1 - |eta_i - eta_j| / (|eta_i - eta_j| + ref_eta)
    ref_eta = jax.nn.softplus(pair_params["pair_ref_eta"]) + 1.0
    eta_diff = jnp.abs(eta[:, None] - eta[None, :])
    eta_score = 1.0 - eta_diff / (eta_diff + ref_eta)
    f_eta = jnp.sum(ww * mask_2d * eta_score)

    # 4. Ion pairing propensity: sigmoid(-(binding_i + binding_j) / ref_binding + offset)
    ref_binding = jax.nn.softplus(pair_params["pair_ref_binding"]) + 1.0
    ionic_mask_2d = jnp.where(lam0[:, None] + lam0[None, :] > 0, 1.0, 0.0)
    pair_score = jax.nn.sigmoid(-(binding[:, None] + binding[None, :]) / ref_binding + 1.0)
    f_pair = jnp.sum(ww * mask_2d * ionic_mask_2d * pair_score)

    # 5. Coordination competition: tanh(coord_i * coord_j / ref_coord^2)
    ref_coord = jax.nn.softplus(pair_params["pair_ref_coord"]) + 0.1  # floor prevents div-by-zero in coord^2
    coord_score = jnp.tanh(coord[:, None] * coord[None, :] / (ref_coord ** 2))
    f_coord = jnp.sum(ww * mask_2d * coord_score)

    # 6. Mobility complementarity: tanh(Lambda0_i * eps_j / (eta_j_safe * ref_mob))
    ref_mob = jax.nn.softplus(pair_params["pair_ref_mob"]) + 1.0
    eta_safe = jnp.maximum(eta, 0.1)
    mob_score = jnp.tanh(lam0[:, None] * eps[None, :] / (eta_safe[None, :] * ref_mob))
    f_mob = jnp.sum(ww * mask_2d * mob_score)

    return jnp.array([f_da, f_eps, f_eta, f_pair, f_coord, f_mob])


# =============================================================================
# IMPROVEMENT 2: HIERARCHICAL ATTENTION
# =============================================================================

def _hierarchical_attention(z, mask, params, raw_props, prop_bias, dropout_key, dropout_rate):
    """Two-stage hierarchical attention: intra-class then cross-class.

    Stage 1 (intra-class): solvent-solvent blending + salt-salt mixing
    Stage 2 (cross-class): salt-solvent solvation interaction

    Soft ionic classification via sigmoid(alpha * Lambda_0) — no hard role gating.
    Both stages use additive attention biases (not hard masks) so gradients flow everywhere.

    Args:
        z: (N_max, D_HIDDEN) encoded species representations
        mask: (N_max,) active species mask
        params: dict with attention layer params + ionic_gate_alpha
        raw_props: (N_max, D_INPUT) raw properties (for Lambda_0 classification)
        prop_bias: (N_max, N_max) property-similarity attention bias
        dropout_key: PRNG key for dropout
        dropout_rate: effective dropout rate (0 = eval mode)

    Returns:
        (N_max, D_HIDDEN) updated representations after hierarchical attention
    """
    n_max = z.shape[0]
    is_training = dropout_rate > 0.0
    eff_attn_drop = dropout_rate * float(USE_ATTN_DROPOUT)
    eff_ffn_drop = jnp.where(is_training, FFN_DROPOUT_RATE, 0.0)
    eff_resid_drop = jnp.where(is_training, RESID_DROPOUT_RATE, 0.0)

    all_keys = random.split(dropout_key, N_ATTN_LAYERS * N_DROPOUT_KEYS_PER_LAYER)
    ki = 0

    def _apply_dropout(x, key, rate):
        keep = random.bernoulli(key, 1.0 - rate, x.shape)
        inv_keep = jnp.where(rate > 0.0, 1.0 / (1.0 - rate), 1.0)
        return jnp.where(rate > 0.0, x * keep * inv_keep, x)

    # Soft ionic classification: p_ionic close to 1 for salts, 0 for solvents
    lam0 = raw_props[:, IDX_LAMBDA0]
    alpha = params["ionic_gate_alpha"]
    p_ionic = jax.nn.sigmoid(alpha * lam0) * mask

    # Intra-class bias: high when both species are same class
    # M_intra(i,j) = p_i*p_j + (1-p_i)*(1-p_j)
    intra_bias = (p_ionic[:, None] * p_ionic[None, :]
                  + (1.0 - p_ionic[:, None]) * (1.0 - p_ionic[None, :]))
    # Scale bias to be additive in attention logits (learned magnitude via prop_bias_alpha)
    intra_bias = intra_bias * mask[:, None] * mask[None, :]

    # Cross-class bias: high when species are different classes
    # M_cross(i,j) = 1 - M_intra(i,j) = p_i*(1-p_j) + (1-p_i)*p_j
    cross_bias = 1.0 - intra_bias
    cross_bias = cross_bias * mask[:, None] * mask[None, :]

    # --- Stage 1: Intra-class attention ---
    q = z @ params["attn0_q_w"] + params["attn0_q_b"]
    k = z @ params["attn0_k_w"] + params["attn0_k_b"]
    v = z @ params["attn0_v_w"] + params["attn0_v_b"]

    bias_0 = prop_bias * intra_bias
    attn_out = _multihead_attention(q, k, v, mask, bias_0, all_keys[ki], eff_attn_drop)
    ki += 1
    attn_out = attn_out @ params["attn0_out_w"] + params["attn0_out_b"]
    attn_out = _apply_dropout(attn_out, all_keys[ki], eff_resid_drop)
    ki += 1

    z = _layer_norm(z + attn_out * mask[:, None],
                    params["ln0_attn_scale"], params["ln0_attn_bias"])
    z = z * mask[:, None]

    ffn = jax.nn.gelu(z @ params["ffn0_1_w"] + params["ffn0_1_b"])
    ffn = _apply_dropout(ffn, all_keys[ki], eff_ffn_drop)
    ki += 1
    ffn = ffn @ params["ffn0_2_w"] + params["ffn0_2_b"]
    ffn = _apply_dropout(ffn, all_keys[ki], eff_resid_drop)
    ki += 1

    z = _layer_norm(z + ffn * mask[:, None],
                    params["ln0_ffn_scale"], params["ln0_ffn_bias"])
    z = z * mask[:, None]

    # --- Stage 2: Cross-class attention ---
    q = z @ params["attn1_q_w"] + params["attn1_q_b"]
    k = z @ params["attn1_k_w"] + params["attn1_k_b"]
    v = z @ params["attn1_v_w"] + params["attn1_v_b"]

    bias_1 = prop_bias * cross_bias
    attn_out = _multihead_attention(q, k, v, mask, bias_1, all_keys[ki], eff_attn_drop)
    ki += 1
    attn_out = attn_out @ params["attn1_out_w"] + params["attn1_out_b"]
    attn_out = _apply_dropout(attn_out, all_keys[ki], eff_resid_drop)
    ki += 1

    z = _layer_norm(z + attn_out * mask[:, None],
                    params["ln1_attn_scale"], params["ln1_attn_bias"])
    z = z * mask[:, None]

    ffn = jax.nn.gelu(z @ params["ffn1_1_w"] + params["ffn1_1_b"])
    ffn = _apply_dropout(ffn, all_keys[ki], eff_ffn_drop)
    ki += 1
    ffn = ffn @ params["ffn1_2_w"] + params["ffn1_2_b"]
    ffn = _apply_dropout(ffn, all_keys[ki], eff_resid_drop)
    ki += 1

    z = _layer_norm(z + ffn * mask[:, None],
                    params["ln1_ffn_scale"], params["ln1_ffn_bias"])
    z = z * mask[:, None]

    return z


# =============================================================================
# IMPROVEMENT 3: PARAMETERIZED CONCENTRATION-DOME BASELINE
# =============================================================================

def _baseline_conductivity_v2(mix_features_raw, fracs, mask, params):
    """Parameterized concentration-dome baseline replacing WJD.

    Predicts Casteel-Amis-like dome parameters from mixture physics features:
        sigma_base = kappa_max * (c/c_max) * exp(-(c - c_max)^2 / w^2)

    The dome parameters (kappa_max, c_max, w) are predicted by a small MLP from
    7 physically-motivated features, making this generalizable to unseen species.

    Args:
        mix_features_raw: (N_MIX_PHYSICS,) raw mixture physics features from _compute_mixture_physics
        fracs: (N_max,) composition fractions
        mask: (N_max,) active species mask
        params: dict with baseline MLP params (base_h_w, base_h_b, base_out_w, base_out_b)

    Returns:
        log(sigma_base) scalar — log-conductivity from analytical baseline
    """
    base_features = jnp.array([
        mix_features_raw[MIX_IDX_LAMBDA0_AVG],
        mix_features_raw[MIX_IDX_EPS_MIX],
        mix_features_raw[MIX_IDX_ETA_MIX],
        mix_features_raw[MIX_IDX_BINDING_AVG],
        mix_features_raw[MIX_IDX_ANION_R_AVG],
        mix_features_raw[MIX_IDX_JONES_DOLE_B_AVG],
        mix_features_raw[MIX_IDX_IONIC_STRENGTH],
    ])

    # Normalize base features with learned statistics
    base_norm = (base_features - params["base_feat_mean"]) / params["base_feat_std"]

    # Small MLP: (7) -> (16) -> (3)
    h = jax.nn.gelu(base_norm @ params["base_h_w"] + params["base_h_b"])
    raw_out = h @ params["base_out_w"] + params["base_out_b"]

    kappa_max = jax.nn.softplus(raw_out[0]) + SOFTPLUS_FLOOR
    c_max = jax.nn.softplus(raw_out[1]) + SOFTPLUS_FLOOR
    w = jax.nn.softplus(raw_out[2]) + SOFTPLUS_FLOOR

    ionic_strength = mix_features_raw[MIX_IDX_IONIC_STRENGTH]
    c_salt = jnp.maximum(ionic_strength, 1e-6)

    # Casteel-Amis dome: sigma = kappa_max * (c/c_max) * exp(-(c-c_max)^2 / w^2)
    c_ratio = c_salt / c_max
    exponent = -((c_salt - c_max) ** 2) / (w ** 2)
    sigma_base = kappa_max * c_ratio * jnp.exp(exponent)
    sigma_base = jnp.maximum(sigma_base, 1e-6)

    return jnp.log(sigma_base)


# =============================================================================
# PARAMETER INITIALIZATION
# =============================================================================

def init_params_v2(key, mix_mean, mix_std):
    """Initialize all v2 model parameters.

    Parameter groups:
    - Encoder (D_INPUT+3 -> D_HIDDEN)
    - Hierarchical attention (2 layers: intra-class + cross-class)
    - Pairwise reference scales (6 learnable scalars)
    - Pairwise projection (N_PAIRWISE_V2 -> D_PAIR_PROJ)
    - Baseline MLP (N_BASE_FEATURES -> D_BASE_HIDDEN -> 3)
    - Ionic gate (1 scalar)
    - Mix physics projection (N_MIX_PHYSICS -> D_MIX_PROJ)
    - Gated readout (gate + correction)
    - Normalization stats (frozen)
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

    n_keys = 1 + 12 + 1 + 1 + 2 + 1  # enc + 2 attn layers (6 each) + mix_proj + pair_proj + base MLP(2) + gate
    n_spare = 3  # Explicit constant: spare keys for future extensions without re-split
    keys = random.split(key, n_keys + n_spare)
    ki = 0

    # --- Encoder ---
    d_enc_in = D_INPUT + 3  # props + [log_frac, frac, T_scaled]
    linear_init(keys[ki], d_enc_in, D_HIDDEN, "enc"); ki += 1

    # --- Hierarchical attention: 2 layers ---
    for layer in range(2):
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

    # --- Mix physics projection ---
    linear_init(keys[ki], N_MIX_PHYSICS, D_MIX_PROJ, "mix_proj"); ki += 1

    # --- Pairwise reference scales (initialized from training-set-reasonable values) ---
    # Pairwise reference scale initializations — learnable, initialized so typical
    # training-set property magnitudes produce scores in the active region of tanh/sigmoid.
    # softplus(x)+floor ensures positivity. Values derived from species_data.py property ranges.
    params["pair_ref_dn_an"] = jnp.array(5.0)    # Explicit constant: softplus(5)+1≈6; DN~15×AN~20=300, /6→tanh active
    params["pair_ref_eps"] = jnp.array(3.0)       # Explicit constant: softplus(3)+1≈4; eps_diff≤90, /4→Cauchy active
    params["pair_ref_eta"] = jnp.array(1.0)       # Explicit constant: softplus(1)+1≈2.3; eta_diff≤2, /2.3→Cauchy active
    params["pair_ref_binding"] = jnp.array(3.0)   # Explicit constant: softplus(3)+1≈4; binding sum≤90, /4→sigmoid active
    params["pair_ref_coord"] = jnp.array(1.0)     # Explicit constant: softplus(1)+0.1≈1.4; coord product≤6, /1.96→tanh active
    params["pair_ref_mob"] = jnp.array(4.0)       # Explicit constant: softplus(4)+1≈5; Λ₀×ε/η≈1800, /5→tanh saturated

    # --- Pairwise projection ---
    linear_init(keys[ki], N_PAIRWISE_V2, D_PAIR_PROJ, "pair_proj"); ki += 1

    # --- Baseline MLP ---
    linear_init(keys[ki], N_BASE_FEATURES, D_BASE_HIDDEN, "base_h"); ki += 1
    linear_init(keys[ki], D_BASE_HIDDEN, 3, "base_out"); ki += 1
    # Baseline output bias: computed from inverse-softplus of desired physical defaults.
    # Target dome: kappa_max≈10 mS/cm, c_max≈1.1 M, width≈0.7 M (typical 1M LiPF6/EC:DMC).
    # softplus(x)+0.1=target → x=log(exp(target-0.1)-1)
    _kappa_target = 10.0   # Explicit constant: typical peak σ for Li-ion electrolytes (mS/cm)
    _cmax_target = 1.1     # Explicit constant: typical optimal salt concentration (M)
    _width_target = 0.7    # Explicit constant: typical dome half-width (M)
    params["base_out_b"] = jnp.array([
        jnp.log(jnp.exp(jnp.array(_kappa_target - SOFTPLUS_FLOOR)) - 1.0),
        jnp.log(jnp.exp(jnp.array(_cmax_target - SOFTPLUS_FLOOR)) - 1.0),
        jnp.log(jnp.exp(jnp.array(_width_target - SOFTPLUS_FLOOR)) - 1.0),
    ])

    # Baseline feature normalization stats (will be computed from training data)
    params["base_feat_mean"] = jnp.zeros(N_BASE_FEATURES)
    params["base_feat_std"] = jnp.ones(N_BASE_FEATURES)

    # --- Ionic gate (sigmoid scale for Lambda_0 classification) ---
    params["ionic_gate_alpha"] = jnp.array(IONIC_GATE_ALPHA_INIT)

    # --- Gated readout ---
    linear_init(keys[ki], D_GATE_IN_V2, D_ATTN_OUT, "gate"); ki += 1
    params["corr_w"] = jnp.zeros((D_ATTN_OUT, 1))
    params["corr_b"] = jnp.zeros(1)

    return params


# =============================================================================
# FORWARD PASS
# =============================================================================

def forward_single_v2(params, species_props, raw_props, fracs, mask, temperature_K,
                      dropout_key, dropout_rate):
    """Full v2 forward pass: encoder -> hierarchical attention -> readout.

    Integrates all three improvements:
    1. Bounded pairwise features (injected into gated readout)
    2. Hierarchical attention (intra-class + cross-class)
    3. Parameterized concentration-dome baseline (replaces WJD)

    Args:
        params: model parameter dict
        species_props: (N_max, D_INPUT) z-score normalized property vectors
        raw_props: (N_max, D_INPUT) raw unnormalized property vectors
        fracs: (N_max,) composition fractions
        mask: (N_max,) active species mask
        temperature_K: scalar temperature
        dropout_key: PRNG key
        dropout_rate: 0.0 for eval, ATTN_DROPOUT_RATE for training

    Returns:
        log(sigma_pred) scalar
    """
    n_max = species_props.shape[0]
    T_scaled = temperature_K / T_REF_K

    # --- Encoder: species properties + composition + temperature -> D_HIDDEN ---
    log_fracs = jnp.log(jnp.maximum(fracs, 1e-8))
    aug = jnp.concatenate([
        species_props,
        log_fracs[:, None],
        fracs[:, None],
        jnp.full((n_max, 1), T_scaled),
    ], axis=-1)

    z = jax.nn.gelu(aug @ params["enc_w"] + params["enc_b"]) * mask[:, None]

    # --- Property-distance attention bias ---
    if USE_PROP_BIAS:
        phys = species_props[:, :D_PROP]
        norms = jnp.maximum(jnp.sqrt(jnp.sum(phys ** 2, axis=-1, keepdims=True)), 1e-8)
        phys_normed = phys / norms
        cos_sim = phys_normed @ phys_normed.T
        prop_bias = params["prop_bias_alpha"] * cos_sim * (mask[:, None] * mask[None, :])
    else:
        prop_bias = jnp.zeros((n_max, n_max))

    # --- Hierarchical attention (improvement 2) ---
    key1, key2 = random.split(dropout_key)
    z = _hierarchical_attention(z, mask, params, raw_props, prop_bias, key1, dropout_rate)

    # --- Composition-weighted pooling ---
    frac_weights = fracs * mask
    frac_sum = jnp.maximum(frac_weights.sum(), 1e-8)
    z_pool = (z * frac_weights[:, None]).sum(axis=0) / frac_sum
    z_max = jnp.where(mask[:, None] > 0, z, -1e9).max(axis=0)

    # --- Mixture physics (reused from v1) ---
    mix_raw, _log_sigma_wjd = _compute_mixture_physics(raw_props, fracs, mask, temperature_K)
    mix_norm = (mix_raw - params["mix_mean"]) / params["mix_std"]
    mix_proj = jax.nn.gelu(mix_norm @ params["mix_proj_w"] + params["mix_proj_b"])

    # --- Bounded pairwise features (improvement 1) ---
    pair_params = {k: params[k] for k in params if k.startswith("pair_ref_")}
    pair_features = _compute_pairwise_v2(raw_props, fracs, mask, pair_params)
    pair_proj = jax.nn.gelu(pair_features @ params["pair_proj_w"] + params["pair_proj_b"])

    # --- Parameterized baseline (improvement 3) ---
    log_sigma_base = _baseline_conductivity_v2(mix_raw, fracs, mask, params)

    # --- Gated readout: physics + pairwise gate which attention features matter ---
    z_attn = jnp.concatenate([z_pool, z_max])
    gate_input = jnp.concatenate([mix_proj, pair_proj, jnp.array([T_scaled])])
    gates = jax.nn.sigmoid(gate_input @ params["gate_w"] + params["gate_b"])
    nn_correction = ((gates * z_attn) @ params["corr_w"] + params["corr_b"])[0]

    return log_sigma_base + nn_correction


forward_batch_v2 = jax.vmap(forward_single_v2, in_axes=(None, 0, 0, 0, 0, 0, 0, None))


@jax.jit
def _forward_batch_eval_v2(params, props, raw, fracs, mask, temps, keys):
    """JIT-compiled batch inference (eval mode)."""
    return forward_batch_v2(params, props, raw, fracs, mask, temps, keys, 0.0)


@jax.jit
def _forward_single_eval_v2(params, props, raw, fracs, mask, temp):
    """JIT-compiled single-recipe inference (eval mode)."""
    return forward_single_v2(params, props, raw, fracs, mask, temp, random.PRNGKey(0), 0.0)


# =============================================================================
# LOSS AND TRAINING
# =============================================================================

def loss_fn_v2(params, batch_tuple, dropout_key):
    """Weighted log-MSE loss for v2 model."""
    props, raw, fracs, mask, temps, log_sigma, weights = batch_tuple
    n_batch = props.shape[0]
    dropout_keys = random.split(dropout_key, n_batch)
    pred_log_sigma = forward_batch_v2(params, props, raw, fracs, mask, temps,
                                      dropout_keys, ATTN_DROPOUT_RATE)
    residuals = pred_log_sigma - log_sigma
    return jnp.sum(weights * residuals**2) / jnp.sum(weights)


def make_train_step_v2(opt):
    """Create JIT-compiled train step."""
    @jax.jit
    def step(params, opt_state, batch_tuple, dropout_key):
        loss, grads = jax.value_and_grad(loss_fn_v2)(params, batch_tuple, dropout_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return step


def compute_val_mae_v2(params, batch: MolSetBatch) -> float:
    """Compute validation MAE in mS/cm (eval mode)."""
    n = len(batch.recipe_keys)
    dummy_keys = random.split(random.PRNGKey(0), n)
    pred_log_sigma = _forward_batch_eval_v2(
        params,
        jnp.array(batch.species_props),
        jnp.array(batch.raw_props),
        jnp.array(batch.fracs),
        jnp.array(batch.mask),
        jnp.array(batch.temperature_K),
        dummy_keys,
    )
    pred_sigma = jnp.exp(pred_log_sigma)
    true_sigma = jnp.exp(jnp.array(batch.log_sigma))
    return float(jnp.mean(jnp.abs(pred_sigma - true_sigma)))


def compute_metrics_v2(params, batch: MolSetBatch) -> dict:
    """Compute MAE, RMSE, bias, MAPE."""
    n = len(batch.recipe_keys)
    dummy_keys = random.split(random.PRNGKey(0), n)
    pred_log_sigma = _forward_batch_eval_v2(
        params,
        jnp.array(batch.species_props),
        jnp.array(batch.raw_props),
        jnp.array(batch.fracs),
        jnp.array(batch.mask),
        jnp.array(batch.temperature_K),
        dummy_keys,
    )
    pred_sigma = jnp.exp(pred_log_sigma)
    true_sigma = jnp.exp(jnp.array(batch.log_sigma))
    residuals = pred_sigma - true_sigma
    return {
        "mae_mS_cm": float(jnp.mean(jnp.abs(residuals))),
        "rmse_mS_cm": float(jnp.sqrt(jnp.mean(residuals**2))),
        "bias_mS_cm": float(jnp.mean(residuals)),
        "mape_pct": float(jnp.mean(jnp.abs(residuals) / jnp.maximum(true_sigma, 0.1)) * 100),
        "log_mse": float(jnp.mean((pred_log_sigma - jnp.array(batch.log_sigma))**2)),
    }


# =============================================================================
# INFERENCE API
# =============================================================================

_PROP_CACHE_V2: Dict[str, np.ndarray] = {}


def _get_raw_cached_v2(name: str) -> np.ndarray:
    if name not in _PROP_CACHE_V2:
        _PROP_CACHE_V2[name] = get_raw_property_vector(name)
    return _PROP_CACHE_V2[name]


def predict_sigma_v2(params, norm_mean, norm_std, recipe, temperature_K):
    """Predict conductivity in mS/cm for a single recipe.

    Args:
        params: v2 model parameters
        norm_mean, norm_std: property normalization stats
        recipe: {"salts": {...}, "solvents": {...}, "additives": {...}}
        temperature_K: temperature in Kelvin

    Returns:
        sigma in mS/cm
    """
    props = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    raw = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    fracs = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    mask = np.zeros(N_MAX_SPECIES, dtype=np.float64)

    j = 0
    for role in ("salts", "solvents", "additives"):
        for sp_name, frac in sorted(recipe[role].items()):
            raw_vec = _get_raw_cached_v2(sp_name)
            props[j] = (raw_vec - norm_mean) / norm_std
            raw[j] = raw_vec
            fracs[j] = frac
            mask[j] = 1.0
            j += 1

    log_sigma = _forward_single_eval_v2(
        params,
        jnp.array(props),
        jnp.array(raw),
        jnp.array(fracs),
        jnp.array(mask),
        jnp.array(temperature_K),
    )
    return float(jnp.exp(log_sigma))


def molset_conductivity_s_m_v2(params, species_props_norm, species_props_raw, X, T_K):
    """Pure-JAX conductivity prediction for optimizer inner loop.

    All inputs are pre-built JAX arrays. Fully differentiable via JAX AD.

    Args:
        params: v2 model parameters
        species_props_norm: (n_design, D_INPUT) z-scored property vectors
        species_props_raw: (n_design, D_INPUT) raw property vectors
        X: (n_design,) mole fractions (design vector)
        T_K: scalar temperature in Kelvin

    Returns:
        sigma in S/m (scalar)
    """
    n_design = X.shape[0]
    props = jnp.zeros((N_MAX_SPECIES, D_INPUT))
    raw = jnp.zeros((N_MAX_SPECIES, D_INPUT))
    fracs = jnp.zeros(N_MAX_SPECIES)
    mask = jnp.zeros(N_MAX_SPECIES)

    props = props.at[:n_design].set(species_props_norm)
    raw = raw.at[:n_design].set(species_props_raw)
    fracs = fracs.at[:n_design].set(X)
    mask = mask.at[:n_design].set(jnp.where(X > 0.0, 1.0, 0.0))

    log_sigma = forward_single_v2(params, props, raw, fracs, mask, T_K,
                                  random.PRNGKey(0), 0.0)
    sigma_ms_cm = jnp.exp(log_sigma)
    return sigma_ms_cm * _MS_CM_TO_S_M


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_model_v2(params, norm_mean, norm_std, path):
    serializable = {k: np.array(v) for k, v in params.items()}
    bundle = {"params": serializable, "norm_mean": np.array(norm_mean), "norm_std": np.array(norm_std)}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"Model saved: {path}")


def load_model_v2(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    params = {k: jnp.array(v) for k, v in bundle["params"].items()}
    return params, bundle["norm_mean"], bundle["norm_std"]


# =============================================================================
# OOD EVALUATION
# =============================================================================

def evaluate_species_ood_v2(species_name, norm_mean, norm_std):
    """Hold out all recipes containing species_name, train v2 on rest, evaluate OOD."""
    logger.info(f"\n{'='*60}")
    logger.info(f"OOD EVALUATION (v2): holding out '{species_name}'")
    logger.info(f"{'='*60}")

    all_entries = _load_all_sources()

    recipe_groups: Dict[tuple, list] = defaultdict(list)
    for recipe, sigma, temp, source in all_entries:
        key = (_recipe_key(recipe), round(temp, 0))
        recipe_groups[key].append((sigma, temp, recipe, source))

    CV_REJECT_THRESHOLD = 0.3  # Explicit constant: 30% CV cutoff (same as main training)
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

    # Compute baseline feature stats from training data
    base_feat_mean, base_feat_std = _compute_base_feature_stats(train_batch)

    params = init_params_v2(random.PRNGKey(SEED_OOD), ood_mix_mean, ood_mix_std)
    params["base_feat_mean"] = jnp.array(base_feat_mean)
    params["base_feat_std"] = jnp.array(base_feat_std)

    n_ood_steps = N_STEPS  # Explicit constant: match main training budget
    warmup_fraction = WARMUP_STEPS / N_STEPS
    ood_warmup = int(n_ood_steps * warmup_fraction)
    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, ood_warmup)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, n_ood_steps - ood_warmup)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [ood_warmup])
    opt = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = opt.init(params)
    step_fn = make_train_step_v2(opt)

    jit_props = jnp.array(train_batch.species_props)
    jit_raw = jnp.array(train_batch.raw_props)
    jit_fracs = jnp.array(train_batch.fracs)
    jit_mask = jnp.array(train_batch.mask)
    jit_temps = jnp.array(train_batch.temperature_K)
    jit_log_sigma = jnp.array(train_batch.log_sigma)
    jit_weights = jnp.array(train_batch.weights)
    batch_tuple = (jit_props, jit_raw, jit_fracs, jit_mask, jit_temps, jit_log_sigma, jit_weights)

    ood_rng = random.PRNGKey(SEED_OOD + 1)
    for step in range(n_ood_steps):
        ood_rng, step_key = random.split(ood_rng)
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple, step_key)

    ood_mae = compute_val_mae_v2(params, ood_batch)
    train_mae = compute_val_mae_v2(params, train_batch)

    logger.info(f"OOD {species_name}: train MAE={train_mae:.3f}, OOD MAE={ood_mae:.3f} mS/cm")
    return {"species": species_name, "n_ood": len(ood_rows), "ood_mae": ood_mae, "train_mae": train_mae}


# =============================================================================
# UTILITIES
# =============================================================================

def _compute_base_feature_stats(batch: MolSetBatch) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std of the 7 baseline features across training batch."""
    n = len(batch.recipe_keys)
    all_feats = np.zeros((n, N_BASE_FEATURES), dtype=np.float64)
    for i in range(n):
        mix_raw, _ = _compute_mixture_physics(
            jnp.array(batch.raw_props[i]),
            jnp.array(batch.fracs[i]),
            jnp.array(batch.mask[i]),
        )
        all_feats[i] = np.array([
            mix_raw[MIX_IDX_LAMBDA0_AVG],
            mix_raw[MIX_IDX_EPS_MIX],
            mix_raw[MIX_IDX_ETA_MIX],
            mix_raw[MIX_IDX_BINDING_AVG],
            mix_raw[MIX_IDX_ANION_R_AVG],
            mix_raw[MIX_IDX_JONES_DOLE_B_AVG],
            mix_raw[MIX_IDX_IONIC_STRENGTH],
        ])
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    logger.info(f"Baseline feature mean: {mean}")
    logger.info(f"Baseline feature std:  {std}")
    return mean, std


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Train MolSet v2 and evaluate."""
    logger.info("=" * 70)
    logger.info("MolSet v2 — Hierarchical Attention + Bounded Pairwise + Dome Baseline")
    logger.info("=" * 70)

    # Species enumeration
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
    train_batch, val_batch = prepare_molset_data(norm_mean, norm_std)
    mix_mean, mix_std = compute_mix_physics_stats(train_batch)
    base_feat_mean, base_feat_std = _compute_base_feature_stats(train_batch)

    logger.info(f"Train: {len(train_batch.recipe_keys)}, Val: {len(val_batch.recipe_keys)}")

    # Initialize
    params = init_params_v2(random.PRNGKey(SEED_MAIN), mix_mean, mix_std)
    params["base_feat_mean"] = jnp.array(base_feat_mean)
    params["base_feat_std"] = jnp.array(base_feat_std)

    n_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer
    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, WARMUP_STEPS)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, N_STEPS - WARMUP_STEPS)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [WARMUP_STEPS])
    opt = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = opt.init(params)
    step_fn = make_train_step_v2(opt)

    # Prepare JAX arrays
    train_props = jnp.array(train_batch.species_props)
    train_raw = jnp.array(train_batch.raw_props)
    train_fracs = jnp.array(train_batch.fracs)
    train_mask = jnp.array(train_batch.mask)
    train_temps = jnp.array(train_batch.temperature_K)
    train_log_sigma = jnp.array(train_batch.log_sigma)
    train_weights = jnp.array(train_batch.weights)
    batch_tuple = (train_props, train_raw, train_fracs, train_mask, train_temps,
                   train_log_sigma, train_weights)

    # SWA setup
    swa_start_step = int(N_STEPS * SWA_START_FRAC)
    swa_params_sum = None
    swa_count = 0

    logger.info(f"\nTraining for {N_STEPS} steps (SWA from step {swa_start_step})...")
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

        if (step + 1) % 100 == 0 or step == 0:
            val_mae = compute_val_mae_v2(params, val_batch)
            train_mae = compute_val_mae_v2(params, train_batch)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = params
                best_step = step + 1
                marker = " ***"
            else:
                marker = ""

            elapsed = time.time() - t0
            logger.info(
                f"Step {step+1:5d} | loss={float(loss):.4f} | "
                f"train={train_mae:.3f} | val={val_mae:.3f} mS/cm | "
                f"best={best_val_mae:.3f}@{best_step} | {elapsed:.1f}s{marker}"
            )

    # SWA averaging
    if swa_count > 1:
        swa_params = jax.tree.map(lambda x: x / swa_count, swa_params_sum)
        swa_val_mae = compute_val_mae_v2(swa_params, val_batch)
        swa_train_mae = compute_val_mae_v2(swa_params, train_batch)
        logger.info(f"\nSWA ({swa_count} checkpoints from step {swa_start_step}):")
        logger.info(f"  SWA val MAE = {swa_val_mae:.3f} mS/cm (best single = {best_val_mae:.3f})")
        logger.info(f"  SWA train MAE = {swa_train_mae:.3f} mS/cm")
        if swa_val_mae < best_val_mae:
            best_params = swa_params
            best_val_mae = swa_val_mae
            best_step = -1
            logger.info(f"  SWA WINS — using averaged weights")

    # Final metrics
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS (step {best_step if best_step > 0 else 'SWA'})")
    logger.info(f"{'='*60}")

    train_metrics = compute_metrics_v2(best_params, train_batch)
    val_metrics = compute_metrics_v2(best_params, val_batch)

    logger.info(f"Train: MAE={train_metrics['mae_mS_cm']:.3f} mS/cm, "
                f"RMSE={train_metrics['rmse_mS_cm']:.3f}, "
                f"bias={train_metrics['bias_mS_cm']:.3f}, "
                f"MAPE={train_metrics['mape_pct']:.1f}%")
    logger.info(f"Val:   MAE={val_metrics['mae_mS_cm']:.3f} mS/cm, "
                f"RMSE={val_metrics['rmse_mS_cm']:.3f}, "
                f"bias={val_metrics['bias_mS_cm']:.3f}, "
                f"MAPE={val_metrics['mape_pct']:.1f}%")
    logger.info(f"Train/Val ratio: {train_metrics['mae_mS_cm']/val_metrics['mae_mS_cm']:.2f}")

    logger.info(f"\n--- Baselines ---")
    logger.info(f"XGB (in-distribution only): 0.26 mS/cm")
    logger.info(f"MLP (fixed 52-d features):  0.591 mS/cm")
    logger.info(f"MolSet v1 (Run 8):          0.458 mS/cm")
    logger.info(f"MolSet v1 (Run 9):          0.323 mS/cm")
    logger.info(f"MolSet v2 (this model):     {val_metrics['mae_mS_cm']:.3f} mS/cm")

    # Save model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mol_set_sigma_v2.pkl")
    save_model_v2(best_params, norm_mean, norm_std, model_path)

    # Gradient check
    logger.info(f"\n--- Gradient Check ---")
    test_props = jnp.array(val_batch.species_props[0])
    test_raw = jnp.array(val_batch.raw_props[0])
    test_fracs = jnp.array(val_batch.fracs[0])
    test_mask = jnp.array(val_batch.mask[0])
    test_temp = jnp.array(val_batch.temperature_K[0])

    grad_fn = jax.grad(forward_single_v2, argnums=3)
    frac_grads = grad_fn(best_params, test_props, test_raw, test_fracs, test_mask, test_temp,
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

    # Speed test
    logger.info(f"\n--- Speed Test ---")
    warmup_result = _forward_single_eval_v2(
        best_params, test_props, test_raw, test_fracs, test_mask, test_temp)
    warmup_result.block_until_ready()

    n_calls = 10000
    t0 = time.time()
    for _ in range(n_calls):
        result = _forward_single_eval_v2(
            best_params, test_props, test_raw, test_fracs, test_mask, test_temp)
    result.block_until_ready()
    elapsed_jit = (time.time() - t0) / n_calls * 1000
    logger.info(f"JIT single-recipe: {elapsed_jit:.4f} ms/recipe ({n_calls} calls)")

    # OOD evaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"OUT-OF-DISTRIBUTION EVALUATION (v2)")
    logger.info(f"{'='*60}")

    ood_species = ["FEC", "VC", "LiFSI"]
    ood_results = []
    for sp in ood_species:
        r = evaluate_species_ood_v2(sp, norm_mean, norm_std)
        ood_results.append(r)

    logger.info(f"\n{'='*60}")
    logger.info(f"OOD SUMMARY (v2 vs v1)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Species':<10} {'v2 OOD':>10} {'v1 OOD':>10} {'Improvement':>12}")
    logger.info(f"{'-'*44}")
    # Explicit constant: measured OOD MAE from v1 Run 8 (plan_molset_v2.md §5)
    v1_baselines = {"FEC": 0.603, "VC": 0.462, "LiFSI": 2.496}
    for r in ood_results:
        if r["ood_mae"] is not None:
            v1 = v1_baselines[r["species"]]
            improvement = (v1 - r["ood_mae"]) / v1 * 100
            logger.info(f"{r['species']:<10} {r['ood_mae']:>10.3f} {v1:>10.3f} {improvement:>+11.1f}%")

    # Pairwise feature boundedness verification
    logger.info(f"\n--- Pairwise Feature Boundedness Check ---")
    rng = np.random.default_rng(42)
    n_check = 100
    all_bounded = True
    for i in range(n_check):
        rand_props = jnp.array(rng.uniform(-5, 100, (N_MAX_SPECIES, D_INPUT)))
        rand_fracs = jnp.array(rng.dirichlet(np.ones(N_MAX_SPECIES)))
        rand_mask = jnp.array(rng.binomial(1, 0.5, N_MAX_SPECIES).astype(float))
        pair_p = {k: best_params[k] for k in best_params if k.startswith("pair_ref_")}
        feats = _compute_pairwise_v2(rand_props, rand_fracs, rand_mask, pair_p)
        if float(jnp.any(jnp.abs(feats) > 1.0)):
            all_bounded = False
            logger.warning(f"  Unbounded pairwise feature at check {i}: {feats}")
    if all_bounded:
        logger.info(f"  All {n_check} random checks PASSED — features bounded to [-1, 1]")


if __name__ == "__main__":
    main()
