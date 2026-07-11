"""
MolSet v3 — Set Transformer with physics-grounded MSA baseline.

Replaces the Casteel-Amis dome MLP (v2) with a composition-dependent MSA chain:
    Kirkwood-Fröhlich mixture dielectric → Fuoss ion pairing → MSA corrections → σ_baseline

The NN residual (hierarchical attention + bounded pairwise + gated linear readout)
learns everything the mean-field theories miss.

Entry point: python -m conductivity.mol_set_sigma_v3
"""

import logging
import os
import pickle
import time
import sys
from typing import Dict
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")

import control_framework.jax_m4_tuning  # noqa: F401 — must precede jax import

import jax
import jax.numpy as jnp
from jax import random
import optax

from constants import (
    T_REF_K,
    K_B,
    N_A,
    EPS_0,
    E_CHARGE,
    BJERRUM_LENGTH_NM,
    MS_CM_TO_S_M as _MS_CM_TO_S_M,
)

from conductivity.mol_set_sigma import (
    D_INPUT,
    D_PROP,
    N_MAX_SPECIES,
    IDX_MW,
    IDX_DENSITY,
    IDX_EPSILON,
    IDX_VISCOSITY,
    IDX_LAMBDA0,
    IDX_ANION_R,
    IDX_DIPOLE,
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
    PROP_BIAS_ALPHA_INIT,
    USE_PROP_BIAS,
    N_MIX_PHYSICS,
    EXP_OVERFLOW_GUARD,
    MolSetBatch,
    get_raw_property_vector,
    get_normalized_property_vector,
    compute_normalization_stats,
    prepare_molset_data,
    compute_mix_physics_stats,
    _compute_mixture_physics,
    _load_all_sources,
    _recipe_key,
    _extract_species_fracs,
    _DATA_ORIGINAL,
    _DATA_CALISOL,
)

from conductivity.mol_set_sigma_v2 import (
    D_HIDDEN,
    D_FFN,
    D_ATTN_OUT,
    D_MIX_PROJ,
    N_PAIRWISE_V2,
    D_PAIR_PROJ,
    IONIC_GATE_ALPHA_INIT,
    _compute_pairwise_v2,
    _hierarchical_attention,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# V3 ARCHITECTURE CONSTANTS
# =============================================================================

# Gate input: mix_proj + pair_proj + T_scaled (same as v2 — dome params were never in the gate)
D_GATE_IN_V3 = D_MIX_PROJ + D_PAIR_PROJ + 1

# Debye unit conversion: 1 Debye = 3.33564e-30 C·m (SI)
DEBYE_TO_CM = 3.33564e-30  # Explicit constant: NIST CODATA 2018

# Onsager limiting law prefactors (CGS-practical units, from Robinson & Stokes Table 6.1)
# Reused from v1 for consistency — these are universal electrolyte theory constants
ONSAGER_S1_PREFACTOR = (
    82.501  # Explicit constant: F²√2/(12π√(ε₀·R·Nₐ)), electrophoretic slope
)
ONSAGER_S2_Q_FACTOR = (
    0.2929  # Explicit constant: q/(1+√q) for q=0.5, symmetric 1:1 electrolyte
)
ONSAGER_S2_PREFACTOR = (
    8.2487e5  # Explicit constant: eF√2/(24π·ε₀·kT·√(ε₀·R·Nₐ)), relaxation slope
)
ONSAGER_EPS_T_EXPONENT = 1.5  # Explicit constant: (ε·T)^(3/2) from Onsager limiting law

LAMBDA0_SCALE = (
    1.0  # Explicit constant: Michaelis-Menten half-max for ionic weight (S·cm²/mol)
)
LAMBDA_CORRECTION = (
    0.01  # Explicit constant: L2 penalty weight on NN correction magnitude
)
OOD_PROXY_SPECIES = "LiTFSI"  # Explicit constant: held-out species for OOD early stopping (75 recipes, salt)


def _ionic_weight(lam0):
    """Continuous ionic weight from limiting molar conductivity.

    Λ₀/(Λ₀ + scale): exactly 0 for solvents (Λ₀=0), ~0.99 for salts (Λ₀≫scale).
    Replaces hard boolean jnp.where(Λ₀ > 0, 1, 0).
    """
    return lam0 / (lam0 + LAMBDA0_SCALE)


# =============================================================================
# MOLE FRACTION → MOLARITY CONVERSION
# =============================================================================


def _mole_frac_to_molarity(species_props, fracs, mask):
    """Convert salt mole fractions to molarity (mol/L) using mixture molar volume.

    c_salt [mol/L] = x_salt / V_mix [L/mol_mixture]
    V_mix = Σ(xᵢ · MWᵢ / ρᵢ) / 1000   [L/mol_mixture]

    where MW is g/mol, ρ is g/cm³, and the /1000 converts cm³ to L.
    """
    mw = species_props[:, IDX_MW]
    rho = jnp.maximum(species_props[:, IDX_DENSITY], 0.1)
    lam0 = species_props[:, IDX_LAMBDA0]

    w = fracs * mask
    iw = _ionic_weight(lam0)
    x_salt = jnp.sum(w * iw)

    # Mixture molar volume: V_mix = Σ(xᵢ · MWᵢ/ρᵢ) in cm³/mol_mixture
    v_mol_cm3 = jnp.sum(w * mw / rho)
    v_mol_L = v_mol_cm3 / 1000.0  # unit conversion: cm³ to L

    c_salt_mol_L = x_salt / jnp.maximum(v_mol_L, 1e-8)
    return c_salt_mol_L, x_salt, v_mol_cm3


# =============================================================================
# MSA PHYSICS BASELINE — MODEL 1: KIRKWOOD-FRÖHLICH MIXTURE DIELECTRIC
# =============================================================================


def _kirkwood_mixture_epsilon(species_props, fracs, mask, T_K):
    """Compute effective mixture dielectric constant with Kirkwood dipolar
    corrections and Born solvation shell salt-induced decrement.

    Replaces the linear mixing eps_mix = Σ(xᵢ·εᵢ) and the per-salt
    dielectric_decrement_frac_per_M heuristic with a composition-dependent
    model derived from species properties.

    Args:
        species_props: (N_max, D_INPUT) RAW property vectors
        fracs: (N_max,) composition fractions
        mask: (N_max,) active species mask
        T_K: scalar temperature in Kelvin

    Returns:
        eps_effective: scalar, effective mixture dielectric constant (dimensionless)
    """
    eps_per = species_props[:, IDX_EPSILON]
    mu_per = species_props[:, IDX_DIPOLE]
    mw_per = species_props[:, IDX_MW]
    rho_per = species_props[:, IDX_DENSITY]
    lam0 = species_props[:, IDX_LAMBDA0]
    r_cat = species_props[:, IDX_CATION_R]
    r_an = species_props[:, IDX_ANION_R]

    w = fracs * mask
    w_sum = jnp.maximum(jnp.sum(w), 1e-8)

    # --- Linear mixing baseline ---
    eps_linear = jnp.sum(w * eps_per) / w_sum

    # --- Kirkwood dipolar correction ---
    # Onsager cavity field: f_cav(εᵢ, εⱼ) = 3εⱼ/(2εⱼ + εᵢ) - 1
    # This captures how solvent j's cavity field enhances/suppresses solvent i's dipole contribution.
    # For i=j (same species), f_cav = 0, so self-terms vanish — only cross-terms contribute.
    eps_i = eps_per[:, None]  # (N, 1)
    eps_j = eps_per[None, :]  # (1, N)
    f_cav = 3.0 * eps_j / jnp.maximum(2.0 * eps_j + eps_i, 1.0) - 1.0

    # Dipole-dipole correlation: μᵢ·μⱼ / μ_ref²
    # μ_ref derived from mixture mean dipole (data-dependent, not hardcoded)
    mu_mean = jnp.sum(w * mu_per) / w_sum
    mu_ref_sq = jnp.maximum(mu_mean**2, 1.0)  # floor at 1 D² for zero-dipole mixtures
    mu_corr = mu_per[:, None] * mu_per[None, :] / mu_ref_sq

    # Composition-weighted cross-term sum
    ww = w[:, None] * w[None, :]
    mask_2d = mask[:, None] * mask[None, :]
    g_correction = jnp.sum(ww * mask_2d * mu_corr * f_cav) / jnp.maximum(w_sum**2, 1e-8)

    eps_dipolar = eps_linear * (1.0 + g_correction)

    # --- Dielectric decrement ---
    # The volume-exclusion model (Gavish & Promislow 2016) gives physically correct ε
    # but compounds with Fuoss to crush alpha at concentrated electrolytes. The Fuoss
    # model over-estimates ion pairing at 1-2M because it's a dilute theory, and
    # lowering ε makes K_A exponentially worse (K_A ∝ exp(λ_B/(ε·a))).
    #
    # For a physics BASELINE that the NN corrects, Kirkwood g-factor alone gives a
    # smooth, monotonic eps that's 10-30% too high at concentrated electrolytes.
    # This partially compensates Fuoss over-pairing, producing a baseline that's
    # systematically 2-5× low — much easier for the NN to correct than the 10-100×
    # underestimate from correct ε + Fuoss.
    eps_effective = eps_dipolar

    # Floor: vacuum permittivity is 1, real solvents are always > 2
    eps_effective = jnp.maximum(eps_effective, 2.0)

    return eps_effective


# =============================================================================
# MSA PHYSICS BASELINE — MODEL 2: FUOSS ION PAIRING
# =============================================================================


def _fuoss_ion_pairing(eps_eff, species_props, fracs, mask, T_K):
    """Compute free-ion fraction α from Fuoss association equilibrium.

    Uses the composition-dependent eps_eff from Kirkwood (not linear mixing).
    Multi-salt: salt-weighted average K_A.

    Args:
        eps_eff: scalar effective dielectric constant from _kirkwood_mixture_epsilon
        species_props: (N_max, D_INPUT) RAW property vectors
        fracs: (N_max,) composition fractions
        mask: (N_max,) active species mask
        T_K: scalar temperature

    Returns:
        alpha: scalar free-ion fraction (0 to 1)
        K_A_mix: scalar effective association constant (M⁻¹)
    """
    lam0 = species_props[:, IDX_LAMBDA0]
    r_cat = species_props[:, IDX_CATION_R]
    r_an = species_props[:, IDX_ANION_R]

    w = fracs * mask
    iw = _ionic_weight(lam0)
    sw = 1.0 - iw

    # Molarity (mol/L) — Fuoss equation is parameterized for molarity, not mole fractions
    c_mol_L, x_salt, _ = _mole_frac_to_molarity(species_props, fracs, mask)
    c_mol_L = jnp.maximum(c_mol_L, 1e-8)

    # Solvation-adjusted contact distance: bare ionic radii + solvent shell thickness
    # In solution, the closest approach includes at least one layer of coordinated solvent.
    # Effective radius from mean solvent molar volume: r_solv = (3·V_mol/(4π·Nₐ))^(1/3)
    mw_per = species_props[:, IDX_MW]
    rho_per = jnp.maximum(species_props[:, IDX_DENSITY], 0.1)
    molar_vol = mw_per / rho_per  # cm³/mol
    v_mol_solv = jnp.sum(w * sw * molar_vol) / jnp.maximum(jnp.sum(w * sw), 1e-8)
    r_solv_cm = (3.0 * v_mol_solv / (4.0 * jnp.pi * N_A)) ** (1.0 / 3.0)
    r_solv_A = r_solv_cm * 1e8  # unit conversion: cm → Å

    a_A = r_cat + r_an + r_solv_A  # Angstrom (solvation-adjusted)
    a_nm = a_A / 10.0  # unit conversion: Angstrom to nanometer
    a_m = a_A * 1e-10  # unit conversion: Angstrom to meter

    # Temperature-scaled Bjerrum length in nm (at ε_r=1)
    lambda_B_vac = BJERRUM_LENGTH_NM * (T_REF_K / T_K)

    # Fuoss equation: K_A [L/mol] = (4π·Nₐ/3)·a³·1000·exp(λ_B/(ε·a_nm))
    prefactor = (
        (4.0 * jnp.pi * N_A / 3.0) * (a_m**3) * 1000.0
    )  # unit conversion: m³ to L
    exponent = lambda_B_vac / (eps_eff * jnp.maximum(a_nm, 1e-3))
    exponent = jnp.minimum(exponent, EXP_OVERFLOW_GUARD)
    K_A_per = prefactor * jnp.exp(exponent)

    # Salt-weighted average K_A
    w_ionic = w * iw
    K_A_mix = jnp.sum(w_ionic * K_A_per) / jnp.maximum(jnp.sum(w_ionic), 1e-8)

    # Fuoss-Kraus quadratic with molarity: α = (-1 + √(1 + 4·K_A·c)) / (2·K_A·c)
    K_A_c = K_A_mix * c_mol_L
    discriminant = 1.0 + 4.0 * K_A_c
    alpha = (-1.0 + jnp.sqrt(discriminant)) / (2.0 * K_A_c)

    return alpha, K_A_mix


# =============================================================================
# MSA PHYSICS BASELINE — MODEL 3: MSA-CORRECTED CONDUCTIVITY
# =============================================================================


def _msa_corrected_conductivity(eps_eff, alpha, species_props, fracs, mask, T_K):
    """Compute MSA-corrected molar conductivity with electrophoretic and
    relaxation corrections, plus Jones-Dole viscosity coupling.

    The MSA improvement over Onsager: finite-ion-size screening parameter Γ
    dampens the √c corrections at high concentration where Onsager overshoots.

    Args:
        eps_eff: scalar effective dielectric constant
        alpha: scalar free-ion fraction from Fuoss
        species_props: (N_max, D_INPUT) RAW property vectors
        fracs: (N_max,) composition fractions
        mask: (N_max,) active species mask
        T_K: scalar temperature

    Returns:
        log_sigma_mS_cm: scalar log(conductivity in mS/cm)
    """
    lam0 = species_props[:, IDX_LAMBDA0]
    eta_per = species_props[:, IDX_VISCOSITY]
    jd_B = species_props[:, IDX_JONES_DOLE]
    r_cat = species_props[:, IDX_CATION_R]
    r_an = species_props[:, IDX_ANION_R]

    w = fracs * mask
    w_sum = jnp.maximum(jnp.sum(w), 1e-8)
    iw = _ionic_weight(lam0)
    sw = 1.0 - iw

    # Molarity (mol/L) for all concentration-dependent terms
    c_mol_L, x_salt, _ = _mole_frac_to_molarity(species_props, fracs, mask)
    c_mol_L = jnp.maximum(c_mol_L, 1e-8)
    sqrt_c = jnp.sqrt(c_mol_L)

    w_ionic = w * iw
    w_ionic_sum = jnp.maximum(jnp.sum(w_ionic), 1e-8)
    Lambda_0_avg = jnp.sum(w_ionic * lam0) / w_ionic_sum

    eta_mix = jnp.sum(w * sw * eta_per) / jnp.maximum(jnp.sum(w * sw), 1e-8)

    sigma_avg_A = jnp.sum(w_ionic * (r_cat + r_an)) / w_ionic_sum
    sigma_avg_m = sigma_avg_A * 1e-10  # Angstrom to meter

    # Debye screening: κ_D² = 2·e²·Nₐ·c_free·1000 / (ε₀·ε·k_B·T)
    # c_free = α·c in mol/L; factor 1000 converts L⁻¹ to m⁻³ (1 L = 10⁻³ m³)
    kappa_D_sq = (
        2.0
        * E_CHARGE**2
        * N_A
        * (alpha * c_mol_L * 1000.0)
        / (EPS_0 * eps_eff * K_B * T_K)
    )
    kappa_D = jnp.sqrt(jnp.maximum(kappa_D_sq, 0.0))

    # MSA screening parameter Γ
    sigma_kappa = sigma_avg_m * kappa_D
    Gamma = (-1.0 + jnp.sqrt(1.0 + 2.0 * sigma_kappa)) / (
        2.0 * jnp.maximum(sigma_avg_m, 1e-12)
    )

    f_MSA = kappa_D / jnp.maximum(kappa_D + 2.0 * Gamma, 1e-12)

    # Onsager electrophoretic: ΔΛ_e = S₁/(ε·T)^(1/2) · √c
    eps_T = jnp.maximum(eps_eff * T_K, 1.0)
    S_e = ONSAGER_S1_PREFACTOR / jnp.sqrt(eps_T)
    delta_Lambda_e = S_e * sqrt_c

    # Onsager relaxation: ΔΛ_r = S₂·Λ₀/(ε·T)^(3/2) · √c
    S_r = ONSAGER_S2_Q_FACTOR * ONSAGER_S2_PREFACTOR / eps_T**ONSAGER_EPS_T_EXPONENT
    delta_Lambda_r = S_r * Lambda_0_avg * sqrt_c

    Lambda_MSA = Lambda_0_avg - (delta_Lambda_e + delta_Lambda_r) * f_MSA
    Lambda_MSA = jnp.maximum(Lambda_MSA, 1.0)

    # Jones-Dole viscosity: η/η₀ = 1 + B√c + Bc (c in mol/L)
    B_avg = jnp.sum(w_ionic * jd_B) / w_ionic_sum
    JD_factor = 1.0 + B_avg * sqrt_c + B_avg * c_mol_L

    # σ [S/cm] = c_free [mol/cm³] · Λ [S·cm²/mol]
    # c_free [mol/cm³] = α · c [mol/L] / 1000
    sigma_S_cm = alpha * (c_mol_L / 1000.0) * Lambda_MSA / JD_factor
    sigma_mS_cm = sigma_S_cm * 1000.0  # S/cm to mS/cm

    SIGMA_FLOOR_MS_CM = 0.1  # Explicit constant: minimum physical conductivity of any liquid electrolyte
    sigma_mS_cm = jnp.maximum(sigma_mS_cm, SIGMA_FLOOR_MS_CM)
    return jnp.log(sigma_mS_cm)


# =============================================================================
# MSA PHYSICS BASELINE — TOP-LEVEL CHAIN
# =============================================================================


def _msa_baseline_conductivity(species_props, fracs, mask, T_K):
    """Kirkwood dielectric → full dissociation → MSA corrections → log(σ)."""
    eps_eff = _kirkwood_mixture_epsilon(species_props, fracs, mask, T_K)
    ALPHA_FULL_DISSOCIATION = 1.0  # Explicit constant: Fuoss removed — overestimates pairing at concentrated conditions
    return _msa_corrected_conductivity(
        eps_eff, ALPHA_FULL_DISSOCIATION, species_props, fracs, mask, T_K
    )


# =============================================================================
# FORWARD PASS
# =============================================================================


def forward_single_v3(
    params,
    species_props,
    raw_props,
    fracs,
    mask,
    temperature_K,
    dropout_key,
    dropout_rate,
):
    """Full v3 forward pass: encoder → hierarchical attention → MSA baseline → gated readout.

    Architecture identical to v2 except the baseline:
    - v2: Casteel-Amis dome MLP (179 learned params)
    - v3: MSA physics chain (0 learned params)

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

    # --- Encoder ---
    log_fracs = jnp.log(jnp.maximum(fracs, 1e-8))
    aug = jnp.concatenate(
        [
            species_props,
            log_fracs[:, None],
            fracs[:, None],
            jnp.full((n_max, 1), T_scaled),
        ],
        axis=-1,
    )
    z = jax.nn.gelu(aug @ params["enc_w"] + params["enc_b"]) * mask[:, None]

    # --- Property-distance attention bias ---
    if USE_PROP_BIAS:
        phys = species_props[:, :D_PROP]
        norms = jnp.maximum(jnp.sqrt(jnp.sum(phys**2, axis=-1, keepdims=True)), 1e-8)
        phys_normed = phys / norms
        cos_sim = phys_normed @ phys_normed.T
        prop_bias = (
            params["prop_bias_alpha"] * cos_sim * (mask[:, None] * mask[None, :])
        )
    else:
        prop_bias = jnp.zeros((n_max, n_max))

    # --- Hierarchical attention (from v2) ---
    key1, key2 = random.split(dropout_key)
    z = _hierarchical_attention(
        z, mask, params, raw_props, prop_bias, key1, dropout_rate
    )

    # --- Composition-weighted pooling ---
    frac_weights = fracs * mask
    frac_sum = jnp.maximum(frac_weights.sum(), 1e-8)
    z_pool = (z * frac_weights[:, None]).sum(axis=0) / frac_sum
    z_max = jnp.where(mask[:, None] > 0, z, -1e9).max(axis=0)

    # --- Mixture physics (from v1, for gate input) ---
    mix_raw, _log_sigma_wjd = _compute_mixture_physics(
        raw_props, fracs, mask, temperature_K
    )
    mix_norm = (mix_raw - params["mix_mean"]) / params["mix_std"]
    mix_proj = jax.nn.gelu(mix_norm @ params["mix_proj_w"] + params["mix_proj_b"])

    # --- Bounded pairwise features (from v2) ---
    pair_params = {k: params[k] for k in params if k.startswith("pair_ref_")}
    pair_features = _compute_pairwise_v2(raw_props, fracs, mask, pair_params)
    pair_proj = jax.nn.gelu(
        pair_features @ params["pair_proj_w"] + params["pair_proj_b"]
    )

    # --- MSA physics baseline (v3 replaces v2 dome) ---
    log_sigma_base = _msa_baseline_conductivity(raw_props, fracs, mask, temperature_K)

    # --- Gated readout ---
    z_attn = jnp.concatenate([z_pool, z_max])
    gate_input = jnp.concatenate([mix_proj, pair_proj, jnp.array([T_scaled])])
    gates = jax.nn.sigmoid(gate_input @ params["gate_w"] + params["gate_b"])
    nn_correction = ((gates * z_attn) @ params["corr_w"] + params["corr_b"])[0]

    return log_sigma_base + nn_correction, nn_correction


forward_batch_v3 = jax.vmap(forward_single_v3, in_axes=(None, 0, 0, 0, 0, 0, 0, None))


@jax.jit
def _forward_batch_eval_v3(params, props, raw, fracs, mask, temps, keys):
    log_sigma, _correction = forward_batch_v3(
        params, props, raw, fracs, mask, temps, keys, 0.0
    )
    return log_sigma


@jax.jit
def _forward_single_eval_v3(params, props, raw, fracs, mask, temp):
    log_sigma, _correction = forward_single_v3(
        params, props, raw, fracs, mask, temp, random.PRNGKey(0), 0.0
    )
    return log_sigma


# =============================================================================
# PARAMETER INITIALIZATION
# =============================================================================


def init_params_v3(key, mix_mean, mix_std):
    """Initialize v3 model parameters. Identical to v2 EXCEPT no dome MLP params."""
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

    n_keys = (
        1 + 12 + 1 + 1 + 1
    )  # enc + 2 attn layers (6 each) + mix_proj + pair_proj + gate
    n_spare = 3  # Explicit constant: spare keys for future extensions
    keys = random.split(key, n_keys + n_spare)
    ki = 0

    # --- Encoder ---
    d_enc_in = D_INPUT + 3
    linear_init(keys[ki], d_enc_in, D_HIDDEN, "enc")
    ki += 1

    # --- Hierarchical attention: 2 layers ---
    for layer in range(2):
        linear_init(keys[ki], D_HIDDEN, D_HIDDEN, f"attn{layer}_q")
        ki += 1
        linear_init(keys[ki], D_HIDDEN, D_HIDDEN, f"attn{layer}_k")
        ki += 1
        linear_init(keys[ki], D_HIDDEN, D_HIDDEN, f"attn{layer}_v")
        ki += 1
        linear_init(keys[ki], D_HIDDEN, D_HIDDEN, f"attn{layer}_out")
        ki += 1
        params[f"ln{layer}_attn_scale"] = jnp.ones(D_HIDDEN)
        params[f"ln{layer}_attn_bias"] = jnp.zeros(D_HIDDEN)
        linear_init(keys[ki], D_HIDDEN, D_FFN, f"ffn{layer}_1")
        ki += 1
        linear_init(keys[ki], D_FFN, D_HIDDEN, f"ffn{layer}_2")
        ki += 1
        params[f"ln{layer}_ffn_scale"] = jnp.ones(D_HIDDEN)
        params[f"ln{layer}_ffn_bias"] = jnp.zeros(D_HIDDEN)

    # --- Mix physics projection ---
    linear_init(keys[ki], N_MIX_PHYSICS, D_MIX_PROJ, "mix_proj")
    ki += 1

    # --- Pairwise reference scales ---
    params["pair_ref_dn_an"] = jnp.array(
        5.0
    )  # Explicit constant: softplus(5)+1≈6; DN~15×AN~20=300, /6→tanh active
    params["pair_ref_eps"] = jnp.array(
        3.0
    )  # Explicit constant: softplus(3)+1≈4; eps_diff≤90, /4→Cauchy active
    params["pair_ref_eta"] = jnp.array(
        1.0
    )  # Explicit constant: softplus(1)+1≈2.3; eta_diff≤2, /2.3→Cauchy active
    params["pair_ref_binding"] = jnp.array(
        3.0
    )  # Explicit constant: softplus(3)+1≈4; binding sum≤90, /4→sigmoid active
    params["pair_ref_coord"] = jnp.array(
        1.0
    )  # Explicit constant: softplus(1)+0.1≈1.4; coord product≤6, /1.96→tanh active
    params["pair_ref_mob"] = jnp.array(
        4.0
    )  # Explicit constant: softplus(4)+1≈5; Λ₀×ε/η≈1800, /5→tanh saturated

    # --- Pairwise projection ---
    linear_init(keys[ki], N_PAIRWISE_V2, D_PAIR_PROJ, "pair_proj")
    ki += 1

    # --- NO dome MLP params (v3: MSA baseline is parameter-free) ---

    # --- Ionic gate ---
    params["ionic_gate_alpha"] = jnp.array(IONIC_GATE_ALPHA_INIT)

    # --- Gated readout ---
    linear_init(keys[ki], D_GATE_IN_V3, D_ATTN_OUT, "gate")
    ki += 1
    params["corr_w"] = jnp.zeros((D_ATTN_OUT, 1))
    params["corr_b"] = jnp.zeros(1)

    return params


# =============================================================================
# LOSS AND TRAINING
# =============================================================================


def loss_fn_v3(params, batch_tuple, dropout_key):
    props, raw, fracs, mask, temps, log_sigma, weights = batch_tuple
    n_batch = props.shape[0]
    dropout_keys = random.split(dropout_key, n_batch)
    pred_log_sigma, nn_corrections = forward_batch_v3(
        params, props, raw, fracs, mask, temps, dropout_keys, ATTN_DROPOUT_RATE
    )
    residuals = pred_log_sigma - log_sigma
    recon_loss = jnp.sum(weights * residuals**2) / jnp.sum(weights)
    correction_penalty = jnp.mean(nn_corrections**2)
    return recon_loss + LAMBDA_CORRECTION * correction_penalty


def make_train_step_v3(opt):
    @jax.jit
    def step(params, opt_state, batch_tuple, dropout_key):
        loss, grads = jax.value_and_grad(loss_fn_v3)(params, batch_tuple, dropout_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return step


def compute_val_mae_v3(params, batch):
    n = len(batch.recipe_keys)
    props = jnp.array(batch.species_props)
    raw = jnp.array(batch.raw_props)
    fracs_j = jnp.array(batch.fracs)
    mask_j = jnp.array(batch.mask)
    temps_j = jnp.array(batch.temperature_K)
    log_sigma_true = jnp.array(batch.log_sigma)
    dummy_keys = random.split(random.PRNGKey(0), n)
    pred_log = _forward_batch_eval_v3(
        params, props, raw, fracs_j, mask_j, temps_j, dummy_keys
    )
    pred = jnp.exp(pred_log)
    true = jnp.exp(log_sigma_true)
    return float(jnp.mean(jnp.abs(pred - true)))


def compute_metrics_v3(params, batch):
    n = len(batch.recipe_keys)
    props = jnp.array(batch.species_props)
    raw = jnp.array(batch.raw_props)
    fracs_j = jnp.array(batch.fracs)
    mask_j = jnp.array(batch.mask)
    temps_j = jnp.array(batch.temperature_K)
    log_sigma_true = jnp.array(batch.log_sigma)
    dummy_keys = random.split(random.PRNGKey(0), n)
    pred_log = _forward_batch_eval_v3(
        params, props, raw, fracs_j, mask_j, temps_j, dummy_keys
    )
    pred = jnp.exp(pred_log)
    true = jnp.exp(log_sigma_true)
    errors = pred - true
    mae = float(jnp.mean(jnp.abs(errors)))
    rmse = float(jnp.sqrt(jnp.mean(errors**2)))
    bias = float(jnp.mean(errors))
    mape = float(jnp.mean(jnp.abs(errors) / jnp.maximum(true, 0.01)) * 100)
    return {"mae": mae, "rmse": rmse, "bias": bias, "mape": mape}


# =============================================================================
# OOD EVALUATION
# =============================================================================


def evaluate_species_ood_v3(species_name, norm_mean, norm_std):
    """Hold out all recipes containing species_name, retrain v3 from scratch, evaluate OOD."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"OOD EVALUATION (v3): holding out '{species_name}'")
    logger.info(f"{'=' * 60}")

    all_entries = _load_all_sources()
    recipe_groups: Dict[tuple, list] = defaultdict(list)
    for recipe, sigma, temp, source in all_entries:
        key = (_recipe_key(recipe), round(temp, 0))
        recipe_groups[key].append((sigma, temp, recipe, source))

    CV_REJECT_THRESHOLD = (
        0.3  # Explicit constant: 30% CV cutoff (same as main training)
    )
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
        all_sp = (
            list(recipe["salts"].keys())
            + list(recipe["solvents"].keys())
            + list(recipe["additives"].keys())
        )
        avg_sigma = np.mean(sigmas)
        avg_temp = np.mean([m[1] for m in measurements])
        row = {
            "recipe": recipe,
            "sigma": avg_sigma,
            "temp": avg_temp,
            "species": all_sp,
            "key": rkey,
        }
        if species_name in all_sp:
            ood_rows.append(row)
        else:
            train_rows.append(row)

    logger.info(f"Train recipes (no {species_name}): {len(train_rows)}")
    logger.info(f"OOD recipes (with {species_name}): {len(ood_rows)}")

    if len(ood_rows) < 5:
        logger.warning(f"Too few OOD recipes ({len(ood_rows)}), skipping")
        return {
            "species": species_name,
            "n_ood": len(ood_rows),
            "ood_mae": None,
            "train_mae": None,
        }

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
                props[i, j] = get_normalized_property_vector(
                    sp_name, norm_mean, norm_std
                )
                raw[i, j] = get_raw_property_vector(sp_name)
                fracs_arr[i, j] = frac
                mask_arr[i, j] = 1.0
            temps_arr[i] = row["temp"]
            log_sigmas[i] = np.log(row["sigma"])
        return MolSetBatch(
            species_props=props,
            raw_props=raw,
            fracs=fracs_arr,
            mask=mask_arr,
            temperature_K=temps_arr,
            log_sigma=log_sigmas,
            weights=weights_arr,
            recipe_keys=[r["key"] for r in rows_list],
        )

    train_batch = rows_to_batch(train_rows)
    ood_batch = rows_to_batch(ood_rows)
    ood_mix_mean, ood_mix_std = compute_mix_physics_stats(train_batch)

    params = init_params_v3(random.PRNGKey(SEED_OOD), ood_mix_mean, ood_mix_std)

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
    step_fn = make_train_step_v3(opt)

    jit_props = jnp.array(train_batch.species_props)
    jit_raw = jnp.array(train_batch.raw_props)
    jit_fracs = jnp.array(train_batch.fracs)
    jit_mask = jnp.array(train_batch.mask)
    jit_temps = jnp.array(train_batch.temperature_K)
    jit_log_sigma = jnp.array(train_batch.log_sigma)
    jit_weights = jnp.array(train_batch.weights)
    batch_tuple = (
        jit_props,
        jit_raw,
        jit_fracs,
        jit_mask,
        jit_temps,
        jit_log_sigma,
        jit_weights,
    )

    LOG_EVERY_OOD = 500  # Explicit constant: OOD training progress log interval
    ood_rng = random.PRNGKey(SEED_OOD + 1)
    best_ood_mae = float("inf")
    best_step = 0
    t0 = time.time()
    for step in range(1, N_STEPS + 1):
        ood_rng, step_key = random.split(ood_rng)
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple, step_key)
        if step % LOG_EVERY_OOD == 0 or step == 1:
            elapsed = time.time() - t0
            ood_mae = compute_val_mae_v3(params, ood_batch)
            if ood_mae < best_ood_mae:
                best_ood_mae = ood_mae
                best_step = step
            logger.info(
                f"  [{species_name}] Step {step:5d} | loss={float(loss):.4f} | "
                f"OOD={ood_mae:.3f} | best={best_ood_mae:.3f}@{best_step} | {elapsed:.0f}s"
            )

    final_ood_mae = compute_val_mae_v3(params, ood_batch)
    final_train_mae = compute_val_mae_v3(params, train_batch)
    logger.info(
        f"OOD {species_name}: train={final_train_mae:.3f}, OOD={final_ood_mae:.3f}, best={best_ood_mae:.3f}@{best_step}"
    )
    return {
        "species": species_name,
        "n_ood": len(ood_rows),
        "ood_mae": final_ood_mae,
        "train_mae": final_train_mae,
        "best_ood_mae": best_ood_mae,
        "best_step": best_step,
    }


# =============================================================================
# INFERENCE API
# =============================================================================


def predict_sigma_v3(params, norm_mean, norm_std, recipe, temperature_K):
    """Predict conductivity (mS/cm) for a single recipe dict."""
    species_fracs = _extract_species_fracs(recipe)
    props = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    raw = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    fracs = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    mask_arr = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    for j, (sp_name, frac) in enumerate(species_fracs[:N_MAX_SPECIES]):
        props[j] = get_normalized_property_vector(sp_name, norm_mean, norm_std)
        raw[j] = get_raw_property_vector(sp_name)
        fracs[j] = frac
        mask_arr[j] = 1.0
    log_sigma = _forward_single_eval_v3(
        params,
        jnp.array(props),
        jnp.array(raw),
        jnp.array(fracs),
        jnp.array(mask_arr),
        jnp.array(temperature_K),
    )
    return float(jnp.exp(log_sigma))


def molset_conductivity_s_m_v3(params, species_props_norm, species_props_raw, X, T_K):
    """Pure-JAX optimizer interface: conductivity in S/m."""
    log_sigma_mS_cm, _corr = forward_single_v3(
        params,
        species_props_norm,
        species_props_raw,
        X,
        jnp.ones(N_MAX_SPECIES),
        T_K,
        random.PRNGKey(0),
        0.0,
    )
    return jnp.exp(log_sigma_mS_cm) * _MS_CM_TO_S_M


# =============================================================================
# MAIN
# =============================================================================


def main():
    logger.info("=" * 70)
    logger.info(
        "MolSet v3 — MSA Physics Baseline + Hierarchical Attention + Bounded Pairwise"
    )
    logger.info("=" * 70)

    # --- Species enumeration ---
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

    # --- OOD Proxy Split: hold out OOD_PROXY_SPECIES from training ---
    all_entries_for_proxy = _load_all_sources()
    proxy_recipe_keys = set()
    recipe_groups_proxy: Dict[tuple, list] = defaultdict(list)
    for recipe, sigma, temp, source in all_entries_for_proxy:
        key = _recipe_key(recipe)
        recipe_groups_proxy[(key, round(temp, 0))].append(recipe)
    for (rkey, T_round), recipes in recipe_groups_proxy.items():
        recipe = recipes[0]
        all_sp = (
            list(recipe["salts"].keys())
            + list(recipe["solvents"].keys())
            + list(recipe["additives"].keys())
        )
        if OOD_PROXY_SPECIES in all_sp:
            proxy_recipe_keys.add(rkey)

    train_keys = train_batch.recipe_keys
    proxy_idx = np.array(
        [i for i, k in enumerate(train_keys) if k in proxy_recipe_keys]
    )
    core_idx = np.array(
        [i for i, k in enumerate(train_keys) if k not in proxy_recipe_keys]
    )

    def _subset_batch(batch, idx):
        return MolSetBatch(
            species_props=batch.species_props[idx],
            raw_props=batch.raw_props[idx],
            fracs=batch.fracs[idx],
            mask=batch.mask[idx],
            temperature_K=batch.temperature_K[idx],
            log_sigma=batch.log_sigma[idx],
            weights=batch.weights[idx],
            recipe_keys=[batch.recipe_keys[i] for i in idx],
        )

    train_core = _subset_batch(train_batch, core_idx)
    ood_proxy_batch = (
        _subset_batch(train_batch, proxy_idx) if len(proxy_idx) > 0 else None
    )
    mix_mean, mix_std = compute_mix_physics_stats(train_core)

    logger.info(
        f"Train core (no {OOD_PROXY_SPECIES}): {len(train_core.recipe_keys)}, "
        f"OOD proxy ({OOD_PROXY_SPECIES}): {len(proxy_idx)}, Val: {len(val_batch.recipe_keys)}"
    )

    # --- MSA baseline standalone evaluation (before any NN training) ---
    logger.info("\n--- MSA Baseline Standalone Evaluation ---")
    msa_errors_train = []
    for i in range(len(train_core.recipe_keys)):
        raw_i = jnp.array(train_core.raw_props[i])
        fracs_i = jnp.array(train_core.fracs[i])
        mask_i = jnp.array(train_core.mask[i])
        temp_i = jnp.array(train_core.temperature_K[i])
        log_sigma_msa = _msa_baseline_conductivity(raw_i, fracs_i, mask_i, temp_i)
        sigma_msa = float(jnp.exp(log_sigma_msa))
        sigma_true = float(jnp.exp(train_core.log_sigma[i]))
        msa_errors_train.append(abs(sigma_msa - sigma_true))

    msa_errors_val = []
    for i in range(len(val_batch.recipe_keys)):
        raw_i = jnp.array(val_batch.raw_props[i])
        fracs_i = jnp.array(val_batch.fracs[i])
        mask_i = jnp.array(val_batch.mask[i])
        temp_i = jnp.array(val_batch.temperature_K[i])
        log_sigma_msa = _msa_baseline_conductivity(raw_i, fracs_i, mask_i, temp_i)
        sigma_msa = float(jnp.exp(log_sigma_msa))
        sigma_true = float(jnp.exp(val_batch.log_sigma[i]))
        msa_errors_val.append(abs(sigma_msa - sigma_true))

    logger.info(f"MSA baseline train MAE: {np.mean(msa_errors_train):.3f} mS/cm")
    logger.info(f"MSA baseline val MAE:   {np.mean(msa_errors_val):.3f} mS/cm")
    logger.info(
        f"MSA baseline train median AE: {np.median(msa_errors_train):.3f} mS/cm"
    )
    logger.info(f"MSA baseline val median AE:   {np.median(msa_errors_val):.3f} mS/cm")

    # Sample some predictions for sanity check
    logger.info("\n--- MSA Baseline Sample Predictions ---")
    for i in range(min(10, len(val_batch.recipe_keys))):
        raw_i = jnp.array(val_batch.raw_props[i])
        fracs_i = jnp.array(val_batch.fracs[i])
        mask_i = jnp.array(val_batch.mask[i])
        temp_i = jnp.array(val_batch.temperature_K[i])

        eps_eff = _kirkwood_mixture_epsilon(raw_i, fracs_i, mask_i, temp_i)
        log_sigma_msa = _msa_corrected_conductivity(
            eps_eff, 1.0, raw_i, fracs_i, mask_i, temp_i
        )

        sigma_msa = float(jnp.exp(log_sigma_msa))
        sigma_true = float(jnp.exp(val_batch.log_sigma[i]))
        logger.info(
            f"  [{i}] eps_eff={float(eps_eff):.1f}, alpha=1.0 (full dissoc), "
            f"sigma_MSA={sigma_msa:.2f}, sigma_true={sigma_true:.2f} mS/cm"
        )

    # --- Initialize model ---
    params = init_params_v3(random.PRNGKey(SEED_MAIN), mix_mean, mix_std)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"\nModel parameters: {n_params:,}")

    # --- Optimizer ---
    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, WARMUP_STEPS)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, N_STEPS - WARMUP_STEPS)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [WARMUP_STEPS])
    opt = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = opt.init(params)
    step_fn = make_train_step_v3(opt)

    # --- Training (on core, excluding OOD proxy) ---
    jit_props = jnp.array(train_core.species_props)
    jit_raw = jnp.array(train_core.raw_props)
    jit_fracs = jnp.array(train_core.fracs)
    jit_mask = jnp.array(train_core.mask)
    jit_temps = jnp.array(train_core.temperature_K)
    jit_log_sigma = jnp.array(train_core.log_sigma)
    jit_weights = jnp.array(train_core.weights)
    batch_tuple = (
        jit_props,
        jit_raw,
        jit_fracs,
        jit_mask,
        jit_temps,
        jit_log_sigma,
        jit_weights,
    )

    rng = random.PRNGKey(SEED_MAIN + 1)
    best_val_mae = float("inf")
    best_params = params
    best_step = 0
    best_ood_proxy_mae = float("inf")
    best_ood_params = params
    best_ood_step = 0

    swa_start = int(N_STEPS * SWA_START_FRAC)
    swa_params_sum = None
    swa_count = 0

    LOG_EVERY = 100  # Explicit constant: training progress log interval
    logger.info(f"\nTraining for {N_STEPS} steps (SWA from step {swa_start})...")
    logger.info(f"Correction penalty: LAMBDA_CORRECTION={LAMBDA_CORRECTION}")
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        rng, step_key = random.split(rng)
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple, step_key)

        if step % LOG_EVERY == 0 or step == 1:
            train_mae = compute_val_mae_v3(params, train_core)
            val_mae = compute_val_mae_v3(params, val_batch)
            ood_proxy_mae = (
                compute_val_mae_v3(params, ood_proxy_batch)
                if ood_proxy_batch is not None
                else float("nan")
            )
            elapsed = time.time() - t0
            marker = ""
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = jax.tree.map(lambda x: x.copy(), params)
                best_step = step
                marker = " ***"
            ood_marker = ""
            if ood_proxy_batch is not None and ood_proxy_mae < best_ood_proxy_mae:
                best_ood_proxy_mae = ood_proxy_mae
                best_ood_params = jax.tree.map(lambda x: x.copy(), params)
                best_ood_step = step
                ood_marker = " OOD*"
            logger.info(
                f"Step {step:5d} | loss={float(loss):.4f} | "
                f"train={train_mae:.3f} | val={val_mae:.3f} | "
                f"ood_proxy={ood_proxy_mae:.3f} mS/cm | "
                f"best_val={best_val_mae:.3f}@{best_step} | "
                f"best_ood={best_ood_proxy_mae:.3f}@{best_ood_step} | "
                f"{elapsed:.1f}s{marker}{ood_marker}"
            )

        # SWA checkpoint collection
        if step >= swa_start and step % SWA_COLLECT_EVERY == 0:
            if swa_params_sum is None:
                swa_params_sum = jax.tree.map(lambda x: x.copy(), params)
            else:
                swa_params_sum = jax.tree.map(
                    lambda s, p: s + p, swa_params_sum, params
                )
            swa_count += 1

    # SWA averaging
    if swa_count > 0:
        swa_params = jax.tree.map(lambda s: s / swa_count, swa_params_sum)
        swa_val_mae = compute_val_mae_v3(swa_params, val_batch)
        swa_train_mae = compute_val_mae_v3(swa_params, train_core)
        logger.info(f"\nSWA ({swa_count} checkpoints from step {swa_start}):")
        logger.info(
            f"  SWA val MAE = {swa_val_mae:.3f} mS/cm (best single = {best_val_mae:.3f})"
        )
        logger.info(f"  SWA train MAE = {swa_train_mae:.3f} mS/cm")

    # --- Final metrics (best-val params) ---
    train_metrics = compute_metrics_v3(best_params, train_core)
    val_metrics = compute_metrics_v3(best_params, val_batch)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"FINAL RESULTS — best-val params (step {best_step})")
    logger.info(f"{'=' * 60}")
    logger.info(
        f"Train: MAE={train_metrics['mae']:.3f} mS/cm, RMSE={train_metrics['rmse']:.3f}, "
        f"bias={train_metrics['bias']:.3f}, MAPE={train_metrics['mape']:.1f}%"
    )
    logger.info(
        f"Val:   MAE={val_metrics['mae']:.3f} mS/cm, RMSE={val_metrics['rmse']:.3f}, "
        f"bias={val_metrics['bias']:.3f}, MAPE={val_metrics['mape']:.1f}%"
    )
    logger.info(
        f"Train/Val ratio: {train_metrics['mae'] / max(val_metrics['mae'], 0.001):.2f}"
    )

    # --- Final metrics (OOD-best params) ---
    if ood_proxy_batch is not None:
        ood_val_metrics = compute_metrics_v3(best_ood_params, val_batch)
        ood_proxy_metrics = compute_metrics_v3(best_ood_params, ood_proxy_batch)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FINAL RESULTS — OOD-proxy-best params (step {best_ood_step})")
        logger.info(f"{'=' * 60}")
        logger.info(f"Val:       MAE={ood_val_metrics['mae']:.3f} mS/cm")
        logger.info(
            f"OOD proxy: MAE={ood_proxy_metrics['mae']:.3f} mS/cm ({OOD_PROXY_SPECIES})"
        )

    # Baselines
    logger.info("\n--- Baselines ---")
    logger.info("XGB (in-distribution only): 0.26 mS/cm")
    logger.info("MLP (fixed 52-d features):  0.591 mS/cm")
    logger.info("MolSet v1 (Run 8):          0.458 mS/cm")
    logger.info("MolSet v2 (dome baseline):  0.333 mS/cm")
    logger.info(f"MolSet v3 (MSA+corr pen):   {val_metrics['mae']:.3f} mS/cm")

    # Save model (OOD-best params for OOD evaluation)
    use_params = best_ood_params if ood_proxy_batch is not None else best_params
    use_step = best_ood_step if ood_proxy_batch is not None else best_step
    model_path = os.path.join(os.path.dirname(__file__), "mol_set_sigma_v3.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "params": jax.tree.map(np.array, use_params),
                "norm_mean": np.array(norm_mean),
                "norm_std": np.array(norm_std),
                "mix_mean": np.array(mix_mean),
                "mix_std": np.array(mix_std),
                "val_mae": val_metrics["mae"],
                "best_step": use_step,
                "version": "v3_msa_corrpen_oodstop",
            },
            f,
        )
    logger.info(f"Model saved (OOD-best@{use_step}): {model_path}")

    # --- Gradient check ---
    logger.info("\n--- Gradient Check ---")
    test_idx = 0
    test_props = jnp.array(val_batch.species_props[test_idx])
    test_raw = jnp.array(val_batch.raw_props[test_idx])
    test_fracs = jnp.array(val_batch.fracs[test_idx])
    test_mask = jnp.array(val_batch.mask[test_idx])
    test_temp = jnp.array(val_batch.temperature_K[test_idx])

    grad_fn = jax.grad(
        lambda f: forward_single_v3(
            use_params,
            test_props,
            test_raw,
            f,
            test_mask,
            test_temp,
            random.PRNGKey(0),
            0.0,
        )[0]
    )
    grads = grad_fn(test_fracs)
    logger.info("d_log(sigma)/d_x_i for first val recipe:")
    species_fracs = _extract_species_fracs(
        {"salts": {}, "solvents": {}, "additives": {}}
    )
    # Reconstruct species names from val batch
    all_sp_names = sorted(
        set(
            sp
            for entry in _DATA_ORIGINAL + _DATA_CALISOL
            if "conductivity_mS_cm" in entry["properties"]
            for k in ["salts", "solvents", "additives"]
            for sp in entry["recipe"][k].keys()
        )
    )

    # Print gradient for each active species
    for j in range(N_MAX_SPECIES):
        if float(test_mask[j]) > 0:
            g = float(grads[j])
            logger.info(f"  species[{j}] : grad = {g:+.4f}")

    # --- Speed test ---
    logger.info("\n--- Speed Test ---")
    warmup_result = _forward_single_eval_v3(
        use_params, test_props, test_raw, test_fracs, test_mask, test_temp
    )
    warmup_result.block_until_ready()

    n_calls = 10000
    t0 = time.time()
    for _ in range(n_calls):
        result = _forward_single_eval_v3(
            use_params, test_props, test_raw, test_fracs, test_mask, test_temp
        )
    result.block_until_ready()
    elapsed_jit = (time.time() - t0) / n_calls * 1000
    logger.info(f"JIT single-recipe: {elapsed_jit:.4f} ms/recipe ({n_calls} calls)")

    # --- OOD Evaluation ---
    logger.info(f"\n{'=' * 60}")
    logger.info("OUT-OF-DISTRIBUTION EVALUATION (v3)")
    logger.info(f"{'=' * 60}")

    ood_species = ["FEC", "VC", "LiFSI"]
    ood_results = []
    for sp in ood_species:
        r = evaluate_species_ood_v3(sp, norm_mean, norm_std)
        ood_results.append(r)

    logger.info(f"\n{'=' * 60}")
    logger.info("OOD SUMMARY (v3 vs v2 vs v1)")
    logger.info(f"{'=' * 60}")
    logger.info(f"{'Species':<10} {'v3 OOD':>10} {'v2 OOD':>10} {'v1 OOD':>10}")
    logger.info(f"{'-' * 44}")
    # Explicit constants: measured OOD MAE from prior runs (plan_molset_v2.md §5)
    v1_baselines = {"FEC": 0.603, "VC": 0.462, "LiFSI": 2.496}
    v2_baselines = {
        "FEC": 0.845,
        "VC": None,
        "LiFSI": None,
    }  # pending v2 OOD completion
    for r in ood_results:
        if r["ood_mae"] is not None:
            v1 = v1_baselines[r["species"]]
            v2 = v2_baselines.get(r["species"])
            v2_str = f"{v2:.3f}" if v2 is not None else "pending"
            logger.info(
                f"{r['species']:<10} {r['ood_mae']:>10.3f} {v2_str:>10} {v1:>10.3f}"
            )


if __name__ == "__main__":
    main()
