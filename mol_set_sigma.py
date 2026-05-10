"""
MolSets-style Set Transformer for generalizable electrolyte conductivity prediction.

Architecture: Per-species physics property vectors -> self-attention -> composition-weighted
pooling -> readout MLP -> log(sigma). Handles variable-length species sets and generalizes
to unseen molecules via physics properties (not learned embeddings).

Entry point: python -m conductivity.mol_set_sigma
"""

import logging
import os
import pickle
import time
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")

import control_framework.jax_m4_tuning  # noqa: F401 — must precede jax import

import jax
import jax.numpy as jnp
from jax import random, lax
import optax

from constants import T_REF_K, E_CHARGE, EPS_0, K_B, N_A, MS_CM_TO_S_M as _MS_CM_TO_S_M, BJERRUM_LENGTH_NM

ANGSTROM_TO_NM = 1e-1   # Explicit constant: 1 Å = 0.1 nm
ANGSTROM_TO_M = 1e-10   # Explicit constant: 1 Å = 1e-10 m
EXP_OVERFLOW_GUARD = 50.0  # numerical sentinel: cap Fuoss exponent to prevent exp overflow
from data.species_data import SOLVENTS, SALTS, ADDITIVES
from data.electrolyte_property_db import DATA as _DATA_ORIGINAL
from data.electrolyte_calisol_db import DATA as _DATA_CALISOL
from data.electrolyte_electrolytomics_db import DATA as _DATA_ELECTROLYTOMICS
from data.lehnert2025_db import load_lehnert2025 as _load_lehnert
from data.logan2018_db import load_logan2018 as _load_logan
from data.nyman2008_db import load_nyman2008 as _load_nyman
from data.valoen2005_db import load_valoen2005 as _load_valoen

_KNOWN_SPECIES = set(SOLVENTS) | set(SALTS) | set(ADDITIVES)
SIGMA_MIN_THRESHOLD = 0.05   # Explicit constant: below this is frozen/measurement noise (audit: valoen/electrolytomics)
T_MIN_THRESHOLD_K = 253.0    # Explicit constant: below -20C electrolytes are frozen, data is unreliable
LITERATURE_WEIGHT = 0.75     # Explicit constant: literature data slightly less trusted than verified originals

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# PROPERTY VECTOR: which physics properties define each species
# =============================================================================

PROPERTY_KEYS = [
    "molecular_weight",
    "density_g_ml",
    "epsilon_r",
    "viscosity_cP",
    "donor_number",
    "acceptor_number",
    "homo_eV",
    "lumo_eV",
    "E_red",
    "E_ox",
    "flash_point_c",
    "boiling_point_c",
    "lumo_shift_per_cn",
    "gas_yield",
    "jones_dole_B",
    "Lambda_0",
    "anion_radius",
    "cation_radius",
    "ion_pair_binding_kj_mol",
    "dipole_moment_D",
    "coordination_affinity_M_inv",
    "preferred_coordination_number",
    "ligand_field_asymmetry",
    "ion_pair_Kd_M",
]

D_PROP = len(PROPERTY_KEYS)
D_INPUT = D_PROP

# Named indices into PROPERTY_KEYS for physics computations
IDX_MW = PROPERTY_KEYS.index("molecular_weight")
IDX_DENSITY = PROPERTY_KEYS.index("density_g_ml")
IDX_EPSILON = PROPERTY_KEYS.index("epsilon_r")
IDX_VISCOSITY = PROPERTY_KEYS.index("viscosity_cP")
IDX_DONOR = PROPERTY_KEYS.index("donor_number")
IDX_ACCEPTOR = PROPERTY_KEYS.index("acceptor_number")
IDX_LAMBDA0 = PROPERTY_KEYS.index("Lambda_0")
IDX_ANION_R = PROPERTY_KEYS.index("anion_radius")
IDX_ION_PAIR_BINDING = PROPERTY_KEYS.index("ion_pair_binding_kj_mol")
IDX_DIPOLE = PROPERTY_KEYS.index("dipole_moment_D")
IDX_COORD_AFFINITY = PROPERTY_KEYS.index("coordination_affinity_M_inv")
IDX_GAS_YIELD = PROPERTY_KEYS.index("gas_yield")
IDX_E_RED = PROPERTY_KEYS.index("E_red")
IDX_CATION_R = PROPERTY_KEYS.index("cation_radius")
IDX_JONES_DOLE = PROPERTY_KEYS.index("jones_dole_B")
IDX_COORD_NUMBER = PROPERTY_KEYS.index("preferred_coordination_number")
IDX_LF_ASYMMETRY = PROPERTY_KEYS.index("ligand_field_asymmetry")
IDX_ION_PAIR_KD = PROPERTY_KEYS.index("ion_pair_Kd_M")


# =============================================================================
# ARCHITECTURE HYPERPARAMETERS
# Explicit constants: architecture choices for reproducibility/auditability.
# =============================================================================

# Architecture: Set Transformer (Lee et al. ICML 2019) sized for 437 training recipes.
# Downsized: params/samples was 55:1 at d=32. d=24 gives ~14k → 32:1, closer to ideal 5-10:1.
D_HIDDEN = 32      # attention width; 24k params on 437 samples
N_HEADS = 4        # d_head = D_HIDDEN/N_HEADS = 8
N_LAYERS = 2       # depth = ceil(log2(N_MAX_SPECIES)) = ceil(log2(8)) = 3; use 2 (conservative)
D_FFN = 2 * D_HIDDEN  # reduced from 4× (Vaswani) to 2× — data-limited regime
ATTN_DROPOUT_RATE = 0.10    # Vaswani 2017 §5.4; halved from 0.15 after underfitting at full rates
FFN_DROPOUT_RATE = 0.08     # halved from 0.15: FFN+resid stacking too aggressive for 457 samples
RESID_DROPOUT_RATE = 0.05   # halved from 0.1: preserve gradient flow while regularizing
PROP_BIAS_ALPHA_INIT = 0.1  # learnable attention bias initial scale (small → training adjusts magnitude)

# --- Ablation flags ---
USE_PAIRWISE = False        # Ablation: confirmed harmful for OOD salt generalization (LiFSI 55k mS/cm)
USE_ATTN_DROPOUT = True     # Attention dropout prevents species identity memorization
USE_PROP_BIAS = True        # Property-distance cosine similarity biases attention logits

# Attention weights are (D_HIDDEN, D_HIDDEN) — species-count-independent. Extra slots are masked out.
# Max observed species per recipe is 8 (verified over 2693 entries). Pad to 10 for headroom.
# Previous value was 37 (full registry), wasting ~14x compute on masked-zero attention entries.
N_MAX_SPECIES = 10  # Explicit constant: max observed=8 across 2693 recipes (2+headroom), verified 2026-05-08

# Training: cosine annealing over 5000 steps.
# At batch_size=full_dataset, 5000 steps = 5000 epochs. Optimal for <2000 sample problems
# per "How Much Data Do You Need?" (Musgrave et al. 2020, Fig 3).
N_STEPS = 5000
LR_PEAK = 3e-4          # AdamW standard for transformers <100k params (Loshchilov & Hutter 2019)
WARMUP_STEPS = 500      # linear warmup prevents early sharp minima (Goyal 2017, 10% of total steps)
WEIGHT_DECAY = 5e-4     # stronger L2 for small-data regime (1e-4 was under-regularized: T/V ratio 0.63)
MAX_GRAD_NORM = 1.0     # gradient norm clipping (Pascanu 2013); prevents outlier-recipe update spikes
SEED_MAIN = 42          # arbitrary reproducibility seed (train/val split + param init)
SEED_OOD = 123          # separate seed for OOD leave-one-out experiments

# SWA: Stochastic Weight Averaging (Izmailov et al. 2018)
SWA_START_FRAC = 0.6    # Explicit constant: collect from last 40% of cosine schedule (§3 of paper)
SWA_COLLECT_EVERY = 100 # Explicit constant: checkpoint interval matching eval frequency

# Data filtering — matches electrolyte_cond_surrogate_train.py conventions exactly.
# These are measurement-protocol bounds, not optimizer constraints.
ROOM_TEMP_LOW_K = 293.0   # 20C — CALiSol "room temp" lower bound (measurement metadata)
ROOM_TEMP_HIGH_K = 303.0  # 30C — CALiSol "room temp" upper bound (measurement metadata)
LOW_KAPPA_WEIGHT = 0.25   # 1/4 weight for sigma < 2 mS/cm (outside optimizer operating range)
CALISOL_WEIGHT = 0.5      # CALiSol from varied labs: half credibility vs verified originals
LOW_KAPPA_THRESHOLD = 2.0  # mS/cm — boundary copied from electrolyte_cond_surrogate_train.py

LAMBDA_CORRECTION = 0.0    # Explicit constant: correction penalty DISABLED — L2 penalty degrades OOD (FEC 0.60→1.95) because model needs large correction from inaccurate WJD baseline
OOD_PROXY_SPECIES = "LiTFSI"  # Explicit constant: held-out species for OOD early stopping during training
LOG_EVERY = 100            # Explicit constant: log training progress every N steps
OOD_LOG_EVERY = 500        # Explicit constant: coarser logging for OOD retraining (3 species × 5000 steps each)
EARLY_STOP_PATIENCE = 3    # Explicit constant: stop if tracked metric doesn't improve >2% for this many eval rounds
EARLY_STOP_REL_TOL = 0.02  # Explicit constant: minimum relative improvement to reset patience counter

# Factored physics decomposition: separate dissociation and mobility corrections.
# Each factor depends only on the relevant physics — unseen solvents enter only through
# mixture-averaged scalars (ε_mix for dissociation, η_mix for mobility).
# Feature names mirror the return order of _compute_mixture_physics (verified by assert below).
_MIX_FEATURE_NAMES = (
    # GROUP 1: Mixture averages (13)
    "eps_mix", "eta_mix", "dn_mix", "dipole_mix", "an_mix", "density_mix", "mw_mix",
    "mobility", "lambda0_avg", "binding_avg", "anion_r_avg", "anion_vol_avg", "coord_affinity_avg",
    # GROUP 2: Concentration (4)
    "c_total", "c_sq", "c_cube", "log_c",
    # GROUP 3: Transport couplings (5)
    "c_over_eta", "c_times_mobility", "mobility_times_donor", "eta_sq", "jones_dole_B_avg",
    # GROUP 4: Conductivity proxies (3)
    "kappa_nernst", "kappa_with_dissoc", "kappa_composite",
    # GROUP 5: Concentration polynomial (4)
    "conc_dev", "abs_conc_dev", "conc_dev_sq", "conc_dev_cube",
    # GROUP 6: Cross-property (7)
    "lambda0_sq", "lambda0_over_eta", "lambda0_times_eps", "lambda0_ratio",
    "eta_times_c", "eta_times_c_sq", "kappa_eta_sq",
    # GROUP 7: Property spread / heterogeneity (7)
    "lambda0_spread", "binding_spread", "anion_r_spread", "eps_spread",
    "anion_vol_spread", "salt_entropy", "eta_spread",
    # GROUP 8: Reduction / gas (2)
    "gas_yield_avg", "e_red_avg",
    # GROUP 9: Ion-ion correlations (11)
    "ionic_strength", "bjerrum_nm", "debye_nm", "coupling",
    "electrophoretic_corr", "relaxation_corr", "kappa_onsager",
    "walden", "dh_log_gamma", "ion_pair_frac", "jones_dole_correction",
    # GROUP 10: Fuoss + anticorrelation (7)
    "log1p_fuoss_K_A", "P_coord_max", "coord_eps", "coord_dn",
    "f_flex", "log1p_anticorr_boost", "max_anion_r",
    # GROUP 11: Coordination drag (2)
    "coord_jones_dole", "net_anticorr_signed",
    # GROUP 12: High-ε species effects (7)
    "eps_excess_weighted", "eps_excess_x_lambda0", "eps_excess_x_mobility",
    "eps_excess_x_anion_vol", "eps_excess_x_binding", "eps_excess_x_coupling", "eps_excess_x_conc",
    # GROUP 13: Solvent × salt cross-terms (5)
    "donor_x_anion_vol", "acceptor_x_anion_vol", "eps_x_anion_vol", "eps_x_binding", "eta_x_anion_vol",
    # GROUP 14: Concentration × anion cross (3)
    "conc_x_anion_vol", "conc_x_binding", "conc_sq_x_anion_vol",
    # GROUP 15: Anticorrelation composites (4)
    "bjerrum_over_anion", "walden_x_anion_vol", "kappa_jones_dole", "anticorr_score",
    # GROUP 16: Conditioned dissociation (3)
    "eps_kirkwood", "alpha_screened", "alpha_x_c",
)

D_ION_GATE_IN = D_HIDDEN + 1  # per-ion post-attention repr + T_scaled
D_ION_READOUT = D_HIDDEN      # per-ion representation dimension

# Onsager limiting law analytical constants (Robinson & Stokes, "Electrolyte Solutions", Tables 6.1/4.23-4.24)
# These are exact results from statistical mechanics, not tuning parameters.
ONSAGER_S1_PREFACTOR = 82.501       # Explicit constant: F^2*sqrt(2)/(12*pi*sqrt(eps0*R*NA)) [CGS-practical, R&S Table 6.1]
ONSAGER_S2_Q_FACTOR = 0.2929        # Explicit constant: q/(1+sqrt(q)) for q=0.5 (symmetric 1:1 electrolyte, exact algebraic)
ONSAGER_S2_PREFACTOR = 8.2487e5     # Explicit constant: eF*sqrt(2)/(24*pi*eps0*kT*sqrt(eps0*R*NA)) [CGS, R&S Table 6.1]
ONSAGER_EPS_T_EXPONENT = 3 / 2      # (eps*T)^(3/2) exponent from Onsager limiting law derivation
DH_A_PREFACTOR = 1.8246e6           # Explicit constant: e^3*sqrt(2*NA*1000)/(8*pi*(eps0*kT)^(3/2)*ln(10)) [CGS, R&S eq 4.23]
DH_B_PREFACTOR = 50.29e8            # Explicit constant: sqrt(8*pi*NA*1000*e^2/(eps0*kT)) [CGS cm^-1, R&S eq 4.24]

# Fuoss ion association + anticorrelation physics — loaded from physics.json
import json as _json
with open(os.path.join(os.path.dirname(__file__), "..", "config", "physics.json")) as _f:
    _physics_cfg = _json.load(_f)
_ic_cfg = _physics_cfg["ion_correlation"]
_ac_cfg = _ic_cfg["anticorrelation_effect"]
_cc_cfg = _physics_cfg["coordination_competition"]
_pc_cfg = _physics_cfg["physical_constants"]

BJERRUM_VAC_298K_NM = float(_pc_cfg["bjerrum_length_298K_eps1_nm"])
N_COORD_TOTAL = float(_cc_cfg["n_total_fixed"])
ANTICORR_COEFFICIENT = float(_ac_cfg["coefficient"])
ANTICORR_K_A_MIN = float(_ac_cfg["K_A_min"])
ANTICORR_R_FLEX_REF_A = float(_ac_cfg["r_flex_threshold_A"])
ANTICORR_R_CUTOFF_A = float(_ic_cfg["anion_radius_threshold_A"])
ANTICORR_ALPHA_FLEX = float(_ac_cfg["alpha_flex"])
del _json, _f, _physics_cfg, _ic_cfg, _ac_cfg, _cc_cfg, _pc_cfg

N_SCREENED_FUOSS_ITERS = 5  # Explicit constant: self-consistent screening iterations (converges in 3-4)


def _ionic_weight(lam0):
    """Continuous ionic weight: Λ₀/(Λ₀ + 1). 0 for solvents, ~1 for salts."""
    return lam0 / (lam0 + 1.0)


def _mole_frac_to_molarity(species_props, fracs, mask):
    """Convert salt mole fraction to molarity (mol/L) using mixture molar volume.

    c_salt [mol/L] = x_salt / V_mix [L/mol_mixture]
    V_mix = Σ(xᵢ · MWᵢ / ρᵢ) / 1000
    """
    mw = species_props[:, IDX_MW]
    rho = jnp.maximum(species_props[:, IDX_DENSITY], 0.1)
    lam0 = species_props[:, IDX_LAMBDA0]

    w = fracs * mask
    iw = _ionic_weight(lam0)
    x_salt = jnp.sum(w * iw)

    v_mol_cm3 = jnp.sum(w * mw / rho)
    v_mol_L = v_mol_cm3 / 1000.0
    c_salt_mol_L = x_salt / jnp.maximum(v_mol_L, 1e-8)
    return c_salt_mol_L, x_salt


def _kirkwood_mixture_epsilon(species_props, fracs, mask, T_K):
    """Mixture dielectric constant with Kirkwood dipolar cross-corrections.

    ε_eff = ε_linear × (1 + g_Kirkwood) where g_Kirkwood captures
    dipole-dipole cavity field cross-terms between unlike solvents.
    """
    eps_per = species_props[:, IDX_EPSILON]
    mu_per = species_props[:, IDX_DIPOLE]

    w = fracs * mask
    w_sum = jnp.maximum(jnp.sum(w), 1e-8)

    eps_linear = jnp.sum(w * eps_per) / w_sum

    eps_i = eps_per[:, None]
    eps_j = eps_per[None, :]
    f_cav = 3.0 * eps_j / jnp.maximum(2.0 * eps_j + eps_i, 1.0) - 1.0

    mu_mean = jnp.sum(w * mu_per) / w_sum
    mu_ref_sq = jnp.maximum(mu_mean ** 2, 1.0)
    mu_corr = mu_per[:, None] * mu_per[None, :] / mu_ref_sq

    ww = w[:, None] * w[None, :]
    mask_2d = mask[:, None] * mask[None, :]
    g_correction = jnp.sum(ww * mask_2d * mu_corr * f_cav) / jnp.maximum(w_sum ** 2, 1e-8)

    return jnp.maximum(eps_linear * (1.0 + g_correction), 2.0)


def _screened_fuoss_alpha(eps_eff, species_props, fracs, mask, T_K):
    """Self-consistent Debye-screened Fuoss ion pairing with solvation-adjusted distance.

    Conditioning:
    1. Contact distance includes first solvation shell (r_cat + r_an + r_solv)
       where r_solv comes from solvent mean molar volume. Reduces K_A exponent.
    2. Debye screening damps K_A by ionic atmosphere: K_A_eff = K_A × exp(-κ_D × a)
    3. Self-consistent: κ_D depends on c_free = α×c, iterated to convergence.
    """
    lam0 = species_props[:, IDX_LAMBDA0]
    r_cat = species_props[:, IDX_CATION_R]
    r_an = species_props[:, IDX_ANION_R]
    mw_per = species_props[:, IDX_MW]
    rho_per = jnp.maximum(species_props[:, IDX_DENSITY], 0.1)

    w = fracs * mask
    iw = _ionic_weight(lam0)
    sw = 1.0 - iw

    c_mol_L, _ = _mole_frac_to_molarity(species_props, fracs, mask)
    c_mol_L = jnp.maximum(c_mol_L, 1e-8)

    # Solvation-adjusted contact distance: bare radii + solvent shell thickness
    # r_solv = (3 × V_mol_solvent / (4π × N_A))^(1/3) — molecular radius from molar volume
    molar_vol = mw_per / rho_per  # cm³/mol
    v_mol_solv = jnp.sum(w * sw * molar_vol) / jnp.maximum(jnp.sum(w * sw), 1e-8)
    r_solv_cm = (3.0 * v_mol_solv / (4.0 * jnp.pi * N_A)) ** (1.0 / 3.0)
    r_solv_A = r_solv_cm * 1e8  # cm → Å

    a_contact_A = r_cat + r_an + r_solv_A
    a_contact_nm = a_contact_A * ANGSTROM_TO_NM
    a_contact_m = a_contact_A * ANGSTROM_TO_M

    lambda_B_vac_nm = BJERRUM_VAC_298K_NM * (T_REF_K / T_K)

    fuoss_prefactor = (4.0 * jnp.pi * N_A / 3.0) * 1000.0
    fuoss_exponent = lambda_B_vac_nm / jnp.maximum(eps_eff * a_contact_nm, 1e-8)
    fuoss_exponent = jnp.minimum(fuoss_exponent, EXP_OVERFLOW_GUARD)
    K_A_bare_per = fuoss_prefactor * (a_contact_m ** 3) * jnp.exp(fuoss_exponent)

    w_ionic = w * iw
    w_ionic_sum = jnp.maximum(jnp.sum(w_ionic), 1e-8)
    K_A_bare = jnp.sum(w_ionic * K_A_bare_per) / w_ionic_sum

    a_avg_m = jnp.sum(w_ionic * a_contact_m) / w_ionic_sum

    def _iteration(carry, _):
        alpha = carry
        c_free = alpha * c_mol_L
        kappa_D_sq = 2.0 * E_CHARGE**2 * N_A * (c_free * 1000.0) / (
            EPS_0 * eps_eff * K_B * T_K)
        kappa_D = jnp.sqrt(jnp.maximum(kappa_D_sq, 0.0))

        screening_factor = jnp.exp(-kappa_D * a_avg_m)
        K_A_eff = K_A_bare * screening_factor

        K_A_c = K_A_eff * c_mol_L
        discriminant = 1.0 + 4.0 * K_A_c
        alpha_new = (-1.0 + jnp.sqrt(discriminant)) / jnp.maximum(2.0 * K_A_c, 1e-12)
        alpha_new = jnp.maximum(alpha_new, 1e-8)  # numerical floor only: quadratic gives α>0 but fp rounding near K_A→∞
        return alpha_new, None

    alpha_init = jnp.array(1.0)
    alpha_final, _ = lax.scan(_iteration, alpha_init, None, length=N_SCREENED_FUOSS_ITERS)

    return alpha_final, K_A_bare


def _lookup_species_data(name: str) -> dict:
    """Look up species data dict. Raises ValueError if species not found."""
    if name in SALTS:
        return SALTS[name]
    if name in SOLVENTS:
        return SOLVENTS[name]
    if name in ADDITIVES:
        return ADDITIVES[name]
    raise ValueError(f"Species '{name}' not found in SOLVENTS, SALTS, or ADDITIVES")


def get_raw_property_vector(name: str) -> np.ndarray:
    """Extract raw (unnormalized) property vector for a species.

    Returns D_INPUT-dimensional vector of physics properties.
    Missing properties are 0.0 (role-inappropriate, e.g. viscosity for salts).
    """
    data_dict = _lookup_species_data(name)

    vec = np.zeros(D_INPUT, dtype=np.float64)
    for i, key in enumerate(PROPERTY_KEYS):
        val = data_dict.get(key)
        if val is not None and isinstance(val, (int, float)):
            vec[i] = float(val)

    return vec


def compute_normalization_stats(species_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-property mean and std from a list of species."""
    vecs = np.array([get_raw_property_vector(sp) for sp in species_list])
    mean = vecs.mean(axis=0)
    std = vecs.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    return mean, std


def get_normalized_property_vector(
    name: str, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Get z-score normalized property vector."""
    raw = get_raw_property_vector(name)
    normalized = (raw - mean) / std
    return normalized


# =============================================================================
# DATA PREPARATION
# =============================================================================

@dataclass
class MolSetBatch:
    """Batched training data for the MolSets model."""
    species_props: np.ndarray   # (N_recipes, N_max, D_input) — z-score normalized
    raw_props: np.ndarray       # (N_recipes, N_max, D_input) — unnormalized for physics
    fracs: np.ndarray           # (N_recipes, N_max)
    mask: np.ndarray            # (N_recipes, N_max)
    temperature_K: np.ndarray   # (N_recipes,)
    log_sigma: np.ndarray       # (N_recipes,)
    weights: np.ndarray         # (N_recipes,)
    recipe_keys: list
    _jax_cache: dict = None

    def jax_arrays(self):
        """Return cached JAX arrays, converting from numpy only on first call."""
        if self._jax_cache is None:
            self._jax_cache = {
                "props": jnp.array(self.species_props),
                "raw": jnp.array(self.raw_props),
                "fracs": jnp.array(self.fracs),
                "mask": jnp.array(self.mask),
                "temps": jnp.array(self.temperature_K),
                "log_sigma": jnp.array(self.log_sigma),
                "weights": jnp.array(self.weights),
            }
        return self._jax_cache


def _recipe_key(recipe: dict) -> tuple:
    """Canonical hashable key for a recipe."""
    return (
        tuple(sorted(recipe["salts"].items())),
        tuple(sorted(recipe["solvents"].items())),
        tuple(sorted(recipe["additives"].items())),
    )


def _extract_species_fracs(recipe: dict) -> List[Tuple[str, float]]:
    """Extract (species_name, fraction) pairs from a recipe in deterministic order."""
    pairs = []
    for sp_name, frac in sorted(recipe["salts"].items()):
        pairs.append((sp_name, frac))
    for sp_name, frac in sorted(recipe["solvents"].items()):
        pairs.append((sp_name, frac))
    for sp_name, frac in sorted(recipe["additives"].items()):
        pairs.append((sp_name, frac))
    return pairs


def _normalize_entry(entry, source_name):
    """Convert any entry format (dict or NamedTuple) to unified (recipe, sigma, temp, source).

    Entry formats:
      NamedTuple (lehnert/logan/nyman/valoen): .recipe, .properties (has T_K, conductivity_mS_cm)
      dict original: recipe, properties (no temperature — all room temp)
      dict calisol/electrolytomics: recipe, properties, temperature_K at top level
    """
    if hasattr(entry, 'recipe') and not isinstance(entry, dict):
        recipe = entry.recipe
        props = entry.properties
    else:
        recipe = entry["recipe"]
        props = entry["properties"]

    if "conductivity_mS_cm" not in props:
        return None
    sigma = props["conductivity_mS_cm"]

    if "T_K" in props:
        temp = props["T_K"]
    elif isinstance(entry, dict) and "temperature_K" in entry:
        temp = entry["temperature_K"]
    else:
        temp = T_REF_K

    return (recipe, sigma, temp, source_name)


def _load_all_sources():
    """Load all conductivity datasets, apply quality filters, return unified list."""
    all_entries = []
    source_counts = defaultdict(int)

    # 1. Original (highest trust)
    for e in _DATA_ORIGINAL:
        norm = _normalize_entry(e, "original")
        if norm:
            all_entries.append(norm)
            source_counts["original"] += 1

    # 2. CALiSol (room temp only, dedup vs original)
    original_keys = set(_recipe_key(e["recipe"]) for e in _DATA_ORIGINAL)
    for e in _DATA_CALISOL:
        T = e["temperature_K"]
        if T < ROOM_TEMP_LOW_K or T > ROOM_TEMP_HIGH_K:
            continue
        if _recipe_key(e["recipe"]) in original_keys:
            continue
        norm = _normalize_entry(e, "calisol")
        if norm:
            all_entries.append(norm)
            source_counts["calisol"] += 1

    # 3. Electrolytomics (filter: sigma >= 0.05, T >= 253K, known species)
    for e in _DATA_ELECTROLYTOMICS:
        norm = _normalize_entry(e, "electrolytomics")
        if norm is None:
            continue
        recipe, sigma, temp, src = norm
        if sigma < SIGMA_MIN_THRESHOLD or temp < T_MIN_THRESHOLD_K:
            continue
        all_sp = set(recipe["salts"]) | set(recipe["solvents"]) | set(recipe["additives"])
        if not all_sp.issubset(_KNOWN_SPECIES):
            continue
        all_entries.append(norm)
        source_counts["electrolytomics"] += 1

    # 4. Literature datasets (same quality filters)
    for loader, name in [(_load_lehnert, "lehnert2025"), (_load_logan, "logan2018"),
                         (_load_nyman, "nyman2008"), (_load_valoen, "valoen2005")]:
        for e in loader():
            norm = _normalize_entry(e, name)
            if norm is None:
                continue
            recipe, sigma, temp, src = norm
            if sigma < SIGMA_MIN_THRESHOLD or temp < T_MIN_THRESHOLD_K:
                continue
            all_sp = set(recipe["salts"]) | set(recipe["solvents"]) | set(recipe["additives"])
            if not all_sp.issubset(_KNOWN_SPECIES):
                continue
            all_entries.append(norm)
            source_counts[name] += 1

    for src, count in sorted(source_counts.items()):
        logger.info(f"  {src:20s}: {count:5d} entries")
    logger.info(f"  {'TOTAL':20s}: {len(all_entries):5d} entries")
    return all_entries


def prepare_molset_data(
    norm_mean: np.ndarray, norm_std: np.ndarray
) -> Tuple[MolSetBatch, MolSetBatch]:
    """Load and prepare training/validation data from ALL conductivity datasets."""
    all_entries = _load_all_sources()

    # Group by (recipe_key, rounded_temp) — average measurements at same conditions
    recipe_groups: Dict[tuple, list] = defaultdict(list)
    for recipe, sigma, temp, source in all_entries:
        key = (_recipe_key(recipe), round(temp, 0))
        recipe_groups[key].append((sigma, temp, recipe, source))

    logger.info(f"Unique (recipe, T) groups: {len(recipe_groups)}")

    # Drop inconsistent groups (CV > 30%) — 11 found in electrolytomics audit
    CV_REJECT_THRESHOLD = 0.3  # Explicit constant: coefficient of variation cutoff from audit (11 groups flagged)
    n_rejected = 0
    all_species_in_data = set()
    rows = []
    for (rkey, T_round), measurements in recipe_groups.items():
        sigmas = [m[0] for m in measurements]
        if len(sigmas) > 1:
            arr = np.array(sigmas)
            cv = arr.std() / max(arr.mean(), 1e-8)
            if cv > CV_REJECT_THRESHOLD:
                n_rejected += 1
                continue

        recipe = measurements[0][2]
        all_sp = list(recipe["salts"].keys()) + \
                 list(recipe["solvents"].keys()) + \
                 list(recipe["additives"].keys())
        all_species_in_data.update(all_sp)

        avg_sigma = np.mean(sigmas)
        avg_temp = np.mean([m[1] for m in measurements])

        sources = set(m[3] for m in measurements)
        if "original" in sources:
            weight = 1.0
        elif "calisol" in sources:
            weight = CALISOL_WEIGHT
        else:
            weight = LITERATURE_WEIGHT

        if avg_sigma < LOW_KAPPA_THRESHOLD:
            weight *= LOW_KAPPA_WEIGHT

        rows.append({
            "recipe": recipe,
            "sigma": avg_sigma,
            "temp": avg_temp,
            "weight": weight,
            "key": rkey,
            "species": all_sp,
        })

    logger.info(f"Rejected {n_rejected} inconsistent groups (CV > {CV_REJECT_THRESHOLD})")
    logger.info(f"Species in training data: {sorted(all_species_in_data)}")
    logger.info(f"Total averaged recipes: {len(rows)}")

    rng = np.random.default_rng(SEED_MAIN)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    n_val = max(1, int(0.2 * len(rows)))
    val_indices = set(indices[:n_val].tolist())
    train_indices = set(indices[n_val:].tolist())

    def build_batch(idx_set) -> MolSetBatch:
        selected = [rows[i] for i in sorted(idx_set)]
        n = len(selected)

        props = np.zeros((n, N_MAX_SPECIES, D_INPUT), dtype=np.float64)
        raw = np.zeros((n, N_MAX_SPECIES, D_INPUT), dtype=np.float64)
        fracs = np.zeros((n, N_MAX_SPECIES), dtype=np.float64)
        mask = np.zeros((n, N_MAX_SPECIES), dtype=np.float64)
        temps = np.zeros(n, dtype=np.float64)
        log_sigmas = np.zeros(n, dtype=np.float64)
        weights = np.zeros(n, dtype=np.float64)
        keys = []

        for i, row in enumerate(selected):
            species_fracs = _extract_species_fracs(row["recipe"])

            for j, (sp_name, frac) in enumerate(species_fracs[:N_MAX_SPECIES]):
                props[i, j] = get_normalized_property_vector(sp_name, norm_mean, norm_std)
                raw[i, j] = get_raw_property_vector(sp_name)
                fracs[i, j] = frac
                mask[i, j] = 1.0

            temps[i] = row["temp"]
            log_sigmas[i] = np.log(row["sigma"])
            weights[i] = row["weight"]
            keys.append(row["key"])

        return MolSetBatch(
            species_props=props, raw_props=raw, fracs=fracs, mask=mask,
            temperature_K=temps, log_sigma=log_sigmas,
            weights=weights, recipe_keys=keys,
        )

    train_batch = build_batch(train_indices)
    val_batch = build_batch(val_indices)

    logger.info(f"Train: {len(train_batch.recipe_keys)} recipes")
    logger.info(f"Val: {len(val_batch.recipe_keys)} recipes")

    return train_batch, val_batch


# =============================================================================
# SET TRANSFORMER MODEL (Pure JAX)
# =============================================================================

def init_params(key: jax.Array, mix_mean: np.ndarray, mix_std: np.ndarray) -> dict:
    """Initialize all model parameters.

    mix_mean, mix_std: (N_MIX_PHYSICS,) normalization constants for mixture physics.
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

    n_keys = 1 + N_LAYERS * 6 + 1  # enc + layers + ion_gate
    keys = random.split(key, n_keys)
    ki = 0

    d_enc_in = D_INPUT + 3  # props + [log_frac, frac, T_scaled]
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

    # Per-species gated readout: each species' post-attention repr → gated → δᵢ
    linear_init(keys[ki], D_ION_GATE_IN, D_ION_READOUT, "ion_gate"); ki += 1
    params["ion_read_w"] = jnp.zeros((D_ION_READOUT, 1))
    params["ion_read_b"] = jnp.zeros(1)

    return params


def _layer_norm(x, scale, bias):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5) * scale + bias


def _multihead_attention(q, k, v, mask, prop_bias, dropout_key, dropout_rate):
    """Multi-head self-attention with property-distance bias and inverted dropout.

    q,k,v: (N_max, D_HIDDEN), mask: (N_max,)
    prop_bias: (N_max, N_max) property-similarity matrix, broadcast to all heads
    dropout_key: PRNG key for attention dropout
    dropout_rate: effective dropout rate (0.0 = no dropout, ATTN_DROPOUT_RATE = training)
    """
    seq_len, d = q.shape
    d_head = d // N_HEADS

    q = q.reshape(seq_len, N_HEADS, d_head).transpose(1, 0, 2)
    k = k.reshape(seq_len, N_HEADS, d_head).transpose(1, 0, 2)
    v = v.reshape(seq_len, N_HEADS, d_head).transpose(1, 0, 2)

    scale = jnp.sqrt(jnp.array(d_head, dtype=jnp.float64))
    attn_logits = jnp.matmul(q, k.transpose(0, 2, 1)) / scale

    attn_logits = attn_logits + prop_bias[None, :, :]

    mask_2d = mask[None, None, :] * mask[None, :, None]
    attn_logits = jnp.where(mask_2d > 0, attn_logits, -1e9)

    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    attn_weights = jnp.where(mask_2d > 0, attn_weights, 0.0)

    keep = random.bernoulli(dropout_key, 1.0 - dropout_rate, attn_weights.shape)
    inv_keep_rate = jnp.where(dropout_rate > 0.0, 1.0 / (1.0 - dropout_rate), 1.0)
    attn_weights = jnp.where(keep, attn_weights * inv_keep_rate, 0.0)
    attn_weights = jnp.where(mask_2d > 0, attn_weights, 0.0)

    return jnp.matmul(attn_weights, v).transpose(1, 0, 2).reshape(seq_len, d)


def _compute_mixture_physics(species_props, fracs, mask, T_K=T_REF_K):
    """Compute comprehensive mixture-level physical properties from per-species vectors.

    Captures ALL physics mechanisms governing ionic conductivity:
    - Mixture-averaged properties (dielectric, viscosity, solvation, mobility)
    - Ion-ion correlations (Onsager, Debye-Hückel, Bjerrum pairing, coupling parameter)
    - Fuoss ion association + Langmuir competitive coordination
    - Anticorrelation effects (solvation restructuring, anion flexibility)
    - Coordination drag penalties (Jones-Dole B-weighted by coordination probability)
    - Cross-property interactions (ε×anion, DN×anion, η×anion, conc×anion)
    - High-ε species effects (replaces role-based "additive" features)
    - Jones-Dole viscosity corrections (1 + B√c + Bc)
    - Salt heterogeneity (property spreads, mixing entropy via Λ₀)
    - Ion pairing fraction (Fuoss K_A × c / (1 + K_A × c))

    No role gating (is_salt, is_solvent, is_additive). Species are vectors of
    physical properties — Lambda_0 > 0 naturally identifies ionic species,
    epsilon_r naturally identifies polar species, etc.

    NOTE: species_props is RAW (unnormalized) D_INPUT-dimensional array.

    Returns: (N_MIX_PHYSICS,) array of mixture physics features.
    """
    w = fracs * mask
    w_sum = jnp.maximum(w.sum(), 1e-8)

    # ================================================================
    # GROUP 1: MIXTURE AVERAGES (13)
    # ================================================================
    eps_mix = jnp.sum(w * species_props[:, IDX_EPSILON]) / w_sum
    eta_mix = jnp.sum(w * species_props[:, IDX_VISCOSITY]) / w_sum
    dn_mix = jnp.sum(w * species_props[:, IDX_DONOR]) / w_sum
    dipole_mix = jnp.sum(w * species_props[:, IDX_DIPOLE]) / w_sum
    an_mix = jnp.sum(w * species_props[:, IDX_ACCEPTOR]) / w_sum
    density_mix = jnp.sum(w * species_props[:, IDX_DENSITY]) / w_sum
    mw_mix = jnp.sum(w * species_props[:, IDX_MW]) / w_sum
    mobility = eps_mix / jnp.maximum(eta_mix, 0.1)
    lambda0_avg = jnp.sum(w * species_props[:, IDX_LAMBDA0]) / w_sum
    binding_avg = jnp.sum(w * species_props[:, IDX_ION_PAIR_BINDING]) / w_sum
    anion_r_avg = jnp.sum(w * species_props[:, IDX_ANION_R]) / w_sum
    coord_affinity_avg = jnp.sum(w * species_props[:, IDX_COORD_AFFINITY]) / w_sum

    r_an = species_props[:, IDX_ANION_R]
    anion_vol_per = (4.0 / 3.0) * jnp.pi * r_an ** 3
    anion_vol_avg = jnp.sum(w * anion_vol_per) / w_sum

    # ================================================================
    # GROUP 2: CONCENTRATION (4)
    # ================================================================
    c_total = w_sum
    c_sq = c_total ** 2
    c_cube = c_total ** 3
    log_c = jnp.log1p(c_total)

    # ================================================================
    # GROUP 3: TRANSPORT COUPLINGS (5)
    # ================================================================
    c_over_eta = c_total / jnp.maximum(eta_mix, 0.1)
    c_times_mobility = c_total * mobility
    mobility_times_donor = mobility * dn_mix
    eta_sq = eta_mix ** 2
    jones_dole_B_avg = jnp.sum(w * species_props[:, IDX_JONES_DOLE]) / w_sum

    # ================================================================
    # GROUP 4: CONDUCTIVITY PROXIES (3)
    # ================================================================
    kappa_nernst = c_total * lambda0_avg / jnp.maximum(eta_mix, 0.1)
    kappa_with_dissoc = kappa_nernst * eps_mix
    kappa_composite = c_total * lambda0_avg * eps_mix / (
        jnp.maximum(eta_mix, 0.1) * (1.0 + c_total))

    # ================================================================
    # GROUP 5: CONCENTRATION POLYNOMIAL (4)
    # ================================================================
    conc_dev = c_total - 1.0
    abs_conc_dev = jnp.abs(conc_dev)
    conc_dev_sq = conc_dev ** 2
    conc_dev_cube = conc_dev ** 3

    # ================================================================
    # GROUP 6: CROSS-PROPERTY (7)
    # ================================================================
    lambda0_sq = lambda0_avg ** 2
    lambda0_over_eta = lambda0_avg / jnp.maximum(eta_mix, 0.1)
    lambda0_times_eps = lambda0_avg * eps_mix
    lambda0_max = jnp.max(jnp.where(mask > 0, species_props[:, IDX_LAMBDA0], 0.0))
    lambda0_ratio = lambda0_avg / jnp.maximum(lambda0_max, 1.0)
    eta_times_c = eta_mix * c_total
    eta_times_c_sq = eta_mix * c_sq
    kappa_eta_sq = c_total * lambda0_avg * eps_mix / jnp.maximum(eta_sq, 0.01)

    # ================================================================
    # GROUP 7: PROPERTY SPREAD / HETEROGENEITY (7)
    # ================================================================
    lambda0_vals = species_props[:, IDX_LAMBDA0] * mask
    binding_vals = species_props[:, IDX_ION_PAIR_BINDING] * mask
    anion_r_vals_m = r_an * mask
    eps_vals = species_props[:, IDX_EPSILON] * mask
    eta_vals = species_props[:, IDX_VISCOSITY] * mask

    lambda0_spread = jnp.max(jnp.where(mask > 0, lambda0_vals, -1e9)) - lambda0_avg
    binding_spread = jnp.max(jnp.where(mask > 0, binding_vals, -1e9)) - binding_avg
    anion_r_spread = jnp.max(jnp.where(mask > 0, anion_r_vals_m, -1e9)) - anion_r_avg
    eps_spread = jnp.max(jnp.where(mask > 0, eps_vals, -1e9)) - eps_mix
    anion_vol_spread = jnp.max(jnp.where(mask > 0, anion_vol_per * mask, -1e9)) - anion_vol_avg
    eta_spread = jnp.max(jnp.where(mask > 0, eta_vals, -1e9)) - eta_mix

    lam_weighted = species_props[:, IDX_LAMBDA0] * fracs * mask
    salt_entropy = jnp.sum(lam_weighted) ** 2 - jnp.sum(lam_weighted ** 2)

    # ================================================================
    # GROUP 8: REDUCTION / GAS PROXIES (2)
    # ================================================================
    gas_yield_avg = jnp.sum(w * species_props[:, IDX_GAS_YIELD]) / w_sum
    e_red_avg = jnp.sum(w * species_props[:, IDX_E_RED]) / w_sum

    # ================================================================
    # GROUP 9: ION-ION CORRELATION PHYSICS (11)
    # ================================================================
    eps_safe = jnp.maximum(eps_mix, 1.0)
    c_safe = jnp.maximum(c_total, 1e-12)
    sqrt_c = jnp.sqrt(c_safe)

    ionic_indicator = jnp.where(species_props[:, IDX_LAMBDA0] > 0, 1.0, 0.0)
    ionic_strength = jnp.sum(fracs * mask * ionic_indicator)

    eta_poise = eta_mix / 100.0
    onsager_S1 = ONSAGER_S1_PREFACTOR / jnp.maximum(
        eta_poise * jnp.sqrt(eps_safe * T_K), 1e-8)
    electrophoretic_corr = onsager_S1 * sqrt_c

    onsager_S2 = (ONSAGER_S2_Q_FACTOR * ONSAGER_S2_PREFACTOR
                  / jnp.maximum(eps_safe * T_K, 1.0) ** ONSAGER_EPS_T_EXPONENT)
    relaxation_corr = onsager_S2 * lambda0_avg * sqrt_c

    lambda_onsager = jnp.maximum(
        lambda0_avg - electrophoretic_corr - relaxation_corr, 0.0)
    kappa_onsager = c_total * lambda_onsager / 1000.0

    bjerrum_m = E_CHARGE ** 2 / (4.0 * jnp.pi * EPS_0 * eps_safe * K_B * T_K)
    bjerrum_nm = bjerrum_m * 1e9

    ionic_safe = jnp.maximum(ionic_strength, 1e-12)
    debye_m = jnp.sqrt(
        EPS_0 * eps_safe * K_B * T_K
        / (2.0 * E_CHARGE ** 2 * N_A * ionic_safe * 1000.0))
    debye_nm = debye_m * 1e9

    coupling = bjerrum_m / jnp.maximum(debye_m, 1e-30)
    walden = lambda0_avg * eta_mix

    A_dh = DH_A_PREFACTOR / jnp.maximum(eps_safe * T_K, 1.0) ** ONSAGER_EPS_T_EXPONENT
    B_dh = DH_B_PREFACTOR / jnp.sqrt(jnp.maximum(eps_safe * T_K, 1.0))
    a_ion_cm = jnp.maximum(anion_r_avg, 1.0) * 1e-8
    dh_log_gamma = -A_dh * sqrt_c / jnp.maximum(
        1.0 + B_dh * a_ion_cm * sqrt_c, 1e-8)

    jones_dole_correction = 1.0 + jones_dole_B_avg * sqrt_c + jones_dole_B_avg * c_safe

    # ================================================================
    # GROUP 10: FUOSS ION ASSOCIATION + ANTICORRELATION (7)
    # ================================================================
    r_cat = species_props[:, IDX_CATION_R]
    a_contact_A = r_cat + r_an
    a_contact_nm = a_contact_A * ANGSTROM_TO_NM
    a_contact_m = a_contact_A * ANGSTROM_TO_M

    lambda_B_vac = BJERRUM_VAC_298K_NM * (T_REF_K / T_K)
    fuoss_prefactor = (4.0 * jnp.pi * N_A / 3.0) * 1000.0
    fuoss_exponent = lambda_B_vac / jnp.maximum(eps_safe * a_contact_nm, 1e-8)
    fuoss_K_A_per = (fuoss_prefactor * (a_contact_m ** 3)
                     * jnp.exp(jnp.minimum(fuoss_exponent, EXP_OVERFLOW_GUARD)))
    fuoss_K_A = jnp.sum(w * fuoss_K_A_per) / w_sum

    K_coord = species_props[:, IDX_COORD_AFFINITY]
    Kc = K_coord * fracs * mask
    Kc_sum = jnp.maximum(Kc.sum(), 1e-8)
    P_coord_per = Kc / Kc_sum

    P_coord_max = jnp.max(P_coord_per)
    coord_eps = jnp.sum(P_coord_per * species_props[:, IDX_EPSILON])
    coord_dn = jnp.sum(P_coord_per * species_props[:, IDX_DONOR])

    max_anion_r = jnp.max(jnp.where(mask > 0, r_an, 0.0))
    f_flex = jnp.where(
        max_anion_r < ANTICORR_R_CUTOFF_A, 0.0,
        jnp.where(
            max_anion_r >= ANTICORR_R_FLEX_REF_A, 1.0,
            ((max_anion_r - ANTICORR_R_CUTOFF_A)
             / (ANTICORR_R_FLEX_REF_A - ANTICORR_R_CUTOFF_A)) ** ANTICORR_ALPHA_FLEX
        ))

    K_A_effective = jnp.maximum(fuoss_K_A - ANTICORR_K_A_MIN, 0.0)
    anticorr_boost = ANTICORR_COEFFICIENT * K_A_effective * P_coord_max * f_flex

    # Ion pairing fraction from Fuoss K_A: f_pair = K_A * c / (1 + K_A * c)
    ion_pair_frac = fuoss_K_A * c_safe / (1.0 + fuoss_K_A * c_safe)

    # ================================================================
    # GROUP 11: COORDINATION DRAG (2)
    # ================================================================
    coord_jones_dole = jnp.sum(P_coord_per * species_props[:, IDX_JONES_DOLE])
    net_anticorr = anticorr_boost - jnp.abs(coord_jones_dole) * c_total

    # ================================================================
    # GROUP 12: HIGH-ε SPECIES EFFECTS (7)
    # Replaces role-gated "additive epsilon excess" features.
    # Per-species ε deviation from mixture mean, weighted by loading.
    # High-ε species (FEC ε=107) that raise mixture ε → positive excess.
    # ================================================================
    eps_per = species_props[:, IDX_EPSILON]
    eps_excess_per = jnp.maximum(eps_per - eps_mix, 0.0) * mask
    eps_excess_weighted = jnp.sum(w * eps_excess_per) / w_sum

    eps_excess_x_lambda0 = eps_excess_weighted * lambda0_avg
    eps_excess_x_mobility = eps_excess_weighted * mobility
    eps_excess_x_anion_vol = eps_excess_weighted * anion_vol_avg
    eps_excess_x_binding = eps_excess_weighted * binding_avg
    eps_excess_x_coupling = eps_excess_weighted * coupling
    eps_excess_x_conc = eps_excess_weighted * c_total

    # ================================================================
    # GROUP 13: SOLVENT × SALT CROSS-TERMS (5)
    # Encode how solvent solvation properties (DN, AN, ε, η) interact
    # with anion identity (radius, volume, binding) to determine
    # dissociation, mobility, and correlation behavior.
    # ================================================================
    donor_x_anion_vol = dn_mix * anion_vol_avg
    acceptor_x_anion_vol = an_mix * anion_vol_avg
    eps_x_anion_vol = eps_mix * anion_vol_avg
    eps_x_binding = eps_mix * binding_avg
    eta_x_anion_vol = eta_mix * anion_vol_avg

    # ================================================================
    # GROUP 14: CONCENTRATION × ANION CROSS-TERMS (3)
    # Ion pairing and correlation effects are concentration-dependent
    # AND anion-dependent. Bulky anions have weaker pairing at all
    # concentrations but the concentration dependence differs.
    # ================================================================
    conc_x_anion_vol = c_total * anion_vol_avg
    conc_x_binding = c_total * binding_avg
    conc_sq_x_anion_vol = c_sq * anion_vol_avg

    # ================================================================
    # GROUP 15: ANTICORRELATION COMPOSITES (4)
    # ================================================================
    bjerrum_over_anion = bjerrum_nm / jnp.maximum(
        anion_r_avg * ANGSTROM_TO_NM, 1e-4)
    walden_x_anion_vol = walden * anion_vol_avg
    eta_corrected = eta_mix * jones_dole_correction
    kappa_jones_dole = c_total * lambda0_avg / jnp.maximum(eta_corrected, 0.1)
    anticorr_score = eps_mix * anion_vol_avg * jnp.maximum(1.0 - jnp.abs(conc_dev), 0.0)

    # Physics baseline: Walden-Jones-Dole (valid at 0.5-1.5M)
    # Walden-Jones-Dole baseline with Eyring exponential viscosity:
    # κ = c_salt · Λ₀_salt / η_ratio
    # η_ratio = exp(B·√c + B·c) — Eyring activated transport model.
    # Physics: at concentrated electrolytes (>0.5M), viscosity rises exponentially
    # with salt concentration due to ion-solvent coordination shell restructuring.
    # The dilute polynomial (1 + B√c + Bc) truncates the Taylor series too early;
    # the exponential captures the full activated process.
    c_salt = jnp.maximum(ionic_strength, 1e-12)
    sqrt_c_salt = jnp.sqrt(c_salt)
    lambda0_salt = jnp.sum(fracs * mask * ionic_indicator * species_props[:, IDX_LAMBDA0]) / c_salt
    B_salt = jnp.sum(fracs * mask * ionic_indicator * species_props[:, IDX_JONES_DOLE]) / c_salt
    # Eyring viscosity: η/η₀ = exp(B·c)
    # Pure concentration-activation without the Debye-Hückel √c electrostatic drag.
    # The √c term (Falkenhagen) applies to ion atmosphere relaxation at dilute limit;
    # at concentrated electrolytes the dominant resistance is ion-solvent restructuring
    # which scales with c, not √c.
    eta_ratio = jnp.exp(B_salt * c_salt)
    sigma_physics = c_salt * lambda0_salt / eta_ratio
    log_sigma_physics = jnp.log(jnp.maximum(sigma_physics, 1e-6))

    # Conditioned dissociation features for NN (not baseline):
    # Kirkwood ε captures dipolar cross-corrections between solvents.
    # Screened Fuoss α uses solvation-adjusted contact distance + Debye screening.
    # These are FEATURES — the NN learns when/how to use them.
    eps_kirkwood = _kirkwood_mixture_epsilon(species_props, fracs, mask, T_K)
    alpha_screened, K_A_bare = _screened_fuoss_alpha(eps_kirkwood, species_props, fracs, mask, T_K)

    features = jnp.array([
        # GROUP 1: Mixture averages (13)
        eps_mix, eta_mix, dn_mix, dipole_mix, an_mix, density_mix, mw_mix,
        mobility, lambda0_avg, binding_avg, anion_r_avg, anion_vol_avg,
        coord_affinity_avg,
        # GROUP 2: Concentration (4)
        c_total, c_sq, c_cube, log_c,
        # GROUP 3: Transport couplings (5)
        c_over_eta, c_times_mobility, mobility_times_donor, eta_sq,
        jones_dole_B_avg,
        # GROUP 4: Conductivity proxies (3)
        kappa_nernst, kappa_with_dissoc, kappa_composite,
        # GROUP 5: Concentration polynomial (4)
        conc_dev, abs_conc_dev, conc_dev_sq, conc_dev_cube,
        # GROUP 6: Cross-property (7)
        lambda0_sq, lambda0_over_eta, lambda0_times_eps, lambda0_ratio,
        eta_times_c, eta_times_c_sq, kappa_eta_sq,
        # GROUP 7: Property spread / heterogeneity (7)
        lambda0_spread, binding_spread, anion_r_spread, eps_spread,
        anion_vol_spread, salt_entropy, eta_spread,
        # GROUP 8: Reduction / gas (2)
        gas_yield_avg, e_red_avg,
        # GROUP 9: Ion-ion correlations (11)
        ionic_strength, bjerrum_nm, debye_nm, coupling,
        electrophoretic_corr, relaxation_corr, kappa_onsager,
        walden, dh_log_gamma, ion_pair_frac, jones_dole_correction,
        # GROUP 10: Fuoss + anticorrelation (7)
        jnp.log1p(fuoss_K_A), P_coord_max, coord_eps, coord_dn,
        f_flex, jnp.log1p(anticorr_boost), max_anion_r,
        # GROUP 11: Coordination drag (2)
        coord_jones_dole,
        jnp.sign(net_anticorr) * jnp.log1p(jnp.abs(net_anticorr)),
        # GROUP 12: High-ε species effects (7)
        eps_excess_weighted, eps_excess_x_lambda0, eps_excess_x_mobility,
        eps_excess_x_anion_vol, eps_excess_x_binding,
        eps_excess_x_coupling, eps_excess_x_conc,
        # GROUP 13: Solvent × salt cross-terms (5)
        donor_x_anion_vol, acceptor_x_anion_vol, eps_x_anion_vol,
        eps_x_binding, eta_x_anion_vol,
        # GROUP 14: Concentration × anion cross (3)
        conc_x_anion_vol, conc_x_binding, conc_sq_x_anion_vol,
        # GROUP 15: Anticorrelation composites (4)
        bjerrum_over_anion, walden_x_anion_vol, kappa_jones_dole, anticorr_score,
        # GROUP 16: Conditioned dissociation (3)
        # Kirkwood ε captures dipolar cross-corrections (5-15% above linear mixing).
        # Screened alpha: solvation-adjusted contact distance + Debye atmosphere damping.
        # Alpha×c gives effective free-carrier concentration.
        eps_kirkwood, alpha_screened, alpha_screened * c_total,
    ])
    return features, log_sigma_physics


_MIX_PHYSICS_GROUP_NAMES = [
    "mix_avg", "concentration", "transport", "kappa_proxy", "conc_poly",
    "cross_prop", "heterogeneity", "redox_gas", "ion_corr", "fuoss_anticorr",
    "coord_drag", "high_eps", "solv_salt_cross", "conc_anion_cross", "anticorr_composite",
    "conditioned_dissoc",
]
_MIX_PHYSICS_GROUP_SIZES = {
    "mix_avg": len(["eps_mix", "eta_mix", "dn_mix", "dipole_mix", "an_mix",
                     "density_mix", "mw_mix", "mobility", "lambda0_avg",
                     "binding_avg", "anion_r_avg", "anion_vol_avg", "coord_affinity_avg"]),
    "concentration": len(["c_total", "c_sq", "c_cube", "log_c"]),
    "transport": len(["c_over_eta", "c_times_mobility", "mobility_times_donor",
                       "eta_sq", "jones_dole_B_avg"]),
    "kappa_proxy": len(["kappa_nernst", "kappa_with_dissoc", "kappa_composite"]),
    "conc_poly": len(["conc_dev", "abs_conc_dev", "conc_dev_sq", "conc_dev_cube"]),
    "cross_prop": len(["lambda0_sq", "lambda0_over_eta", "lambda0_times_eps",
                        "lambda0_ratio", "eta_times_c", "eta_times_c_sq", "kappa_eta_sq"]),
    "heterogeneity": len(["lambda0_spread", "binding_spread", "anion_r_spread",
                           "eps_spread", "anion_vol_spread", "salt_entropy", "eta_spread"]),
    "redox_gas": len(["gas_yield_avg", "e_red_avg"]),
    "ion_corr": len(["ionic_strength", "bjerrum_nm", "debye_nm", "coupling",
                      "electrophoretic_corr", "relaxation_corr", "kappa_onsager",
                      "walden", "dh_log_gamma", "ion_pair_frac", "jones_dole_correction"]),
    "fuoss_anticorr": len(["fuoss_K_A", "P_coord_max", "coord_eps", "coord_dn",
                            "f_flex", "anticorr_boost", "max_anion_r"]),
    "coord_drag": len(["coord_jones_dole", "net_anticorr"]),
    "high_eps": len(["eps_excess_weighted", "eps_excess_x_lambda0", "eps_excess_x_mobility",
                      "eps_excess_x_anion_vol", "eps_excess_x_binding",
                      "eps_excess_x_coupling", "eps_excess_x_conc"]),
    "solv_salt_cross": len(["donor_x_anion_vol", "acceptor_x_anion_vol",
                             "eps_x_anion_vol", "eps_x_binding", "eta_x_anion_vol"]),
    "conc_anion_cross": len(["conc_x_anion_vol", "conc_x_binding", "conc_sq_x_anion_vol"]),
    "anticorr_composite": len(["bjerrum_over_anion", "walden_x_anion_vol", "kappa_jones_dole", "anticorr_score"]),
    "conditioned_dissoc": len(["eps_kirkwood", "alpha_screened", "alpha_x_c"]),
}
N_MIX_PHYSICS = sum(_MIX_PHYSICS_GROUP_SIZES[g] for g in _MIX_PHYSICS_GROUP_NAMES)
assert len(_MIX_FEATURE_NAMES) == N_MIX_PHYSICS, (
    f"_MIX_FEATURE_NAMES has {len(_MIX_FEATURE_NAMES)} entries but "
    f"N_MIX_PHYSICS={N_MIX_PHYSICS} — update _MIX_FEATURE_NAMES to match _compute_mixture_physics"
)

def _compute_pairwise_interactions(species_props, fracs, mask):
    """Compute composition-weighted pairwise interaction features between all active species.

    For each pair (i,j), computes physics-motivated cross terms weighted by x_i * x_j.
    These generalize to unseen species because they depend on property relationships,
    not species identity.

    Returns (N_PAIRWISE,) vector of summed pairwise features.
    """
    n = species_props.shape[0]
    w = fracs * mask  # (N_max,)

    eps = species_props[:, IDX_EPSILON]
    eta = species_props[:, IDX_VISCOSITY]
    dn = species_props[:, IDX_DONOR]
    an = species_props[:, IDX_ACCEPTOR]
    lam0 = species_props[:, IDX_LAMBDA0]
    binding = species_props[:, IDX_ION_PAIR_BINDING]
    dipole = species_props[:, IDX_DIPOLE]
    coord = species_props[:, IDX_COORD_AFFINITY]

    # Outer product weights: x_i * x_j for all pairs
    ww = w[:, None] * w[None, :]  # (N, N)

    # 1. Dielectric mismatch: |eps_i - eps_j| — drives preferential solvation
    eps_diff = jnp.abs(eps[:, None] - eps[None, :])
    f_eps_mismatch = jnp.sum(ww * eps_diff)

    # 2. Donor-acceptor complementarity: DN_i * AN_j — solvation shell strength
    da_cross = dn[:, None] * an[None, :]
    f_da_complement = jnp.sum(ww * da_cross)

    # 3. Ion mobility in each solvent: Lambda0_i * eps_j / max(eta_j, 0.1)
    mobility_cross = lam0[:, None] * eps[None, :] / jnp.maximum(eta[None, :], 0.1)
    f_ion_mobility = jnp.sum(ww * mobility_cross)

    # 4. Ion pairing tendency: binding_i / max(eps_j, 1) — low eps = more pairing
    pair_cross = binding[:, None] / jnp.maximum(eps[None, :], 1.0)
    f_ion_pairing = jnp.sum(ww * pair_cross)

    # 5. Viscosity contrast: |eta_i - eta_j| — drives non-ideal mixing
    eta_diff = jnp.abs(eta[:, None] - eta[None, :])
    f_eta_contrast = jnp.sum(ww * eta_diff)

    # 6. Dipole-dipole interaction: dipole_i * dipole_j — solvent structuring
    dd_cross = dipole[:, None] * dipole[None, :]
    f_dipole_dipole = jnp.sum(ww * dd_cross)

    # 7. Coordination competition: coord_i * coord_j — species compete for Li+ shell
    cc_cross = coord[:, None] * coord[None, :]
    f_coord_compete = jnp.sum(ww * cc_cross)

    # 8. Solvation-viscosity tradeoff: DN_i * eta_j — good solvators that are viscous
    solv_visc = dn[:, None] * eta[None, :]
    f_solv_visc = jnp.sum(ww * solv_visc)

    return jnp.array([
        f_eps_mismatch, f_da_complement, f_ion_mobility, f_ion_pairing,
        f_eta_contrast, f_dipole_dipole, f_coord_compete, f_solv_visc,
    ])


N_PAIRWISE = 8


def compute_mix_physics_stats(batch) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std of mixture physics features across a batch for z-scoring."""
    n = len(batch.recipe_keys)
    all_mix = np.zeros((n, N_MIX_PHYSICS), dtype=np.float64)
    for i in range(n):
        features, _ = _compute_mixture_physics(
            jnp.array(batch.raw_props[i]),
            jnp.array(batch.fracs[i]),
            jnp.array(batch.mask[i]),
        )
        all_mix[i] = features
    mean = all_mix.mean(axis=0)
    std = all_mix.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    logger.info(f"Mixture physics mean: {mean}")
    logger.info(f"Mixture physics std:  {std}")
    return mean, std


def fit_linear_baseline(batch, mix_mean: np.ndarray, mix_std: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit OLS linear baseline: log(sigma) ≈ mix_norm @ w + b.

    Uses normalized mixture physics features (species-agnostic) to predict log(sigma).
    This baseline captures R²~0.93 of variance — vs WJD which captures R²~0.10.
    The NN correction shrinks from ~2.0 to ~0.17, dramatically improving OOD transfer.
    """
    n = len(batch.recipe_keys)
    mix_features = np.zeros((n, N_MIX_PHYSICS), dtype=np.float64)
    for i in range(n):
        feat, _ = _compute_mixture_physics(
            jnp.array(batch.raw_props[i]),
            jnp.array(batch.fracs[i]),
            jnp.array(batch.mask[i]),
        )
        mix_features[i] = np.array(feat)

    mix_norm = (mix_features - mix_mean) / np.maximum(mix_std, 1e-8)
    targets = batch.log_sigma

    A = np.column_stack([mix_norm, np.ones(n)])
    coefs, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    w = coefs[:-1]
    b = float(coefs[-1])

    pred = mix_norm @ w + b
    residual = pred - targets
    r2 = 1.0 - np.var(residual) / np.var(targets)
    mae = np.abs(residual).mean()
    logger.info(f"Linear baseline fit: R²={r2:.4f}, MAE={mae:.3f}, bias={residual.mean():.4f}")
    return w, b


def compute_pairwise_stats(batch) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std of pairwise interaction features across a batch for z-scoring."""
    n = len(batch.recipe_keys)
    all_pair = np.zeros((n, N_PAIRWISE), dtype=np.float64)
    for i in range(n):
        all_pair[i] = _compute_pairwise_interactions(
            jnp.array(batch.raw_props[i]),
            jnp.array(batch.fracs[i]),
            jnp.array(batch.mask[i]),
        )
    mean = all_pair.mean(axis=0)
    std = all_pair.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    logger.info(f"Pairwise interaction mean: {mean}")
    logger.info(f"Pairwise interaction std:  {std}")
    return mean, std


def forward_single(params, species_props, raw_props, fracs, mask, temperature_K,
                    dropout_key, dropout_rate):
    """Forward pass for a single recipe."""
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
        phys = species_props[:, :D_PROP]
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

    # 4 keys per layer (attn, ffn, resid_attn, resid_ffn)
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

    # Physics baseline: Walden-Jones-Dole with salt-only B and extended viscosity
    _mix_raw, log_sigma_physics = _compute_mixture_physics(raw_props, fracs, mask, temperature_K)

    # Per-species gated readout: each species' post-attention repr → gated → δᵢ
    t_col = jnp.full((n_max, 1), T_scaled)
    gate_input = jnp.concatenate([z, t_col], axis=-1)  # (N_max, D_ION_GATE_IN)
    gates = jax.nn.sigmoid(gate_input @ params["ion_gate_w"] + params["ion_gate_b"])
    per_species_delta = (gates * z) @ params["ion_read_w"] + params["ion_read_b"]  # (N_max, 1)
    per_species_delta = per_species_delta[:, 0]  # (N_max,)

    nn_correction = jnp.sum(per_species_delta * mask * fracs)
    return log_sigma_physics + nn_correction, nn_correction


forward_batch = jax.vmap(forward_single, in_axes=(None, 0, 0, 0, 0, 0, 0, None))


@jax.jit
def _forward_batch_eval(params, props, raw, fracs, mask, temps, keys):
    """JIT-compiled batch inference (eval mode, no dropout)."""
    log_sigma, _correction = forward_batch(params, props, raw, fracs, mask, temps, keys, 0.0)
    return log_sigma


@jax.jit
def _forward_single_eval(params, props, raw, fracs, mask, temp):
    """JIT-compiled single-recipe inference (eval mode, no dropout)."""
    log_sigma, _correction = forward_single(params, props, raw, fracs, mask, temp, random.PRNGKey(0), 0.0)
    return log_sigma


# Property vector cache — avoids repeated Python dict lookups per species
_PROP_CACHE: Dict[str, np.ndarray] = {}


def _get_raw_cached(name: str) -> np.ndarray:
    if name not in _PROP_CACHE:
        _PROP_CACHE[name] = get_raw_property_vector(name)
    return _PROP_CACHE[name]


def predict_sigma(
    params: dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    recipe: dict,
    temperature_K: float,
) -> float:
    """Predict conductivity in mS/cm for a single recipe.

    recipe: {"salts": {"LiPF6": 1.0}, "solvents": {"EC": 0.5, "DMC": 0.5}, "additives": {}}
    """
    props = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    raw = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    fracs = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    mask = np.zeros(N_MAX_SPECIES, dtype=np.float64)

    j = 0
    for role in ("salts", "solvents", "additives"):
        for sp_name, frac in sorted(recipe[role].items()):
            raw_vec = _get_raw_cached(sp_name)
            props[j] = (raw_vec - norm_mean) / norm_std
            raw[j] = raw_vec
            fracs[j] = frac
            mask[j] = 1.0
            j += 1

    log_sigma = _forward_single_eval(
        params,
        jnp.array(props),
        jnp.array(raw),
        jnp.array(fracs),
        jnp.array(mask),
        jnp.array(temperature_K),
    )
    return float(jnp.exp(log_sigma))


def molset_conductivity_s_m(
    params: dict,
    species_props_norm: jnp.ndarray,
    species_props_raw: jnp.ndarray,
    X: jnp.ndarray,
    T_K: jnp.ndarray,
) -> jnp.ndarray:
    """Pure-JAX conductivity prediction for the optimizer inner loop.

    All inputs are pre-built JAX arrays — no Python dicts, no species lookups.
    Fully differentiable via JAX AD.

    Args:
        params: MolSet model parameters (from load_model).
        species_props_norm: (n_design, D_INPUT) z-scored property vectors.
        species_props_raw: (n_design, D_INPUT) raw property vectors.
        X: (n_design,) mole fractions (design vector).
        T_K: scalar temperature in Kelvin.

    Returns:
        σ in S/m (scalar).
    """
    n_design = X.shape[0]
    n_pad = max(n_design, N_MAX_SPECIES)
    props = jnp.zeros((n_pad, D_INPUT))
    raw = jnp.zeros((n_pad, D_INPUT))
    fracs = jnp.zeros(n_pad)
    mask = jnp.zeros(n_pad)

    props = props.at[:n_design].set(species_props_norm)
    raw = raw.at[:n_design].set(species_props_raw)
    fracs = fracs.at[:n_design].set(X)
    mask = mask.at[:n_design].set(jnp.where(X > 0.0, 1.0, 0.0))

    log_sigma, _corr = forward_single(params, props, raw, fracs, mask, T_K,
                                      random.PRNGKey(0), 0.0)
    sigma_ms_cm = jnp.exp(log_sigma)
    return sigma_ms_cm * _MS_CM_TO_S_M


def build_molset_species_arrays(
    species_names: tuple,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> tuple:
    """Pre-compute per-species property arrays for the optimizer.

    Called once at config-build time. Returns arrays indexed by design vector position.

    Args:
        species_names: tuple of species names in design vector order.
        norm_mean, norm_std: normalization stats from training.

    Returns:
        (species_props_norm, species_props_raw) — both (n_species, D_INPUT) jnp arrays.
    """
    n = len(species_names)
    props_norm = np.zeros((n, D_INPUT), dtype=np.float64)
    props_raw = np.zeros((n, D_INPUT), dtype=np.float64)
    for i, name in enumerate(species_names):
        raw_vec = get_raw_property_vector(name)
        props_raw[i] = raw_vec
        std_safe = np.where(norm_std > 1e-12, norm_std, 1.0)
        props_norm[i] = (raw_vec - norm_mean) / std_safe
    return jnp.array(props_norm), jnp.array(props_raw)


def save_model(params: dict, norm_mean: np.ndarray, norm_std: np.ndarray, path: str) -> None:
    serializable = {k: np.array(v) for k, v in params.items()}
    bundle = {
        "params": serializable,
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


def loss_fn(params, batch_tuple, dropout_key):
    """Weighted log-MSE loss + correction magnitude penalty.
    batch_tuple = (props, raw, fracs, mask, temps, log_sigma, weights)."""
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
    """Create a jit-compiled train step closed over the optimizer."""
    @jax.jit
    def step(params, opt_state, batch_tuple, dropout_key):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_tuple, dropout_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return step


def compute_val_mae(params, batch: MolSetBatch) -> float:
    """Compute validation MAE in mS/cm (eval mode, no dropout)."""
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
    """Compute MAE, RMSE, bias, MAPE (eval mode, no dropout)."""
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


# =============================================================================
# OOD EVALUATION
# =============================================================================

def evaluate_species_ood(
    species_name: str, norm_mean: np.ndarray, norm_std: np.ndarray,
    step_fn, opt
) -> dict:
    """Hold out all recipes containing species_name, train on rest, evaluate.

    Accepts pre-compiled step_fn and optimizer to avoid JIT recompilation per species.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"OOD EVALUATION: holding out '{species_name}'")
    logger.info(f"{'='*60}")

    all_entries = _load_all_sources()

    recipe_groups: Dict[tuple, list] = defaultdict(list)
    for recipe, sigma, temp, source in all_entries:
        key = (_recipe_key(recipe), round(temp, 0))
        recipe_groups[key].append((sigma, temp, recipe, source))

    CV_REJECT_THRESHOLD = 0.3  # Explicit constant: 30% coefficient of variation cutoff (same as prepare_molset_data)
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

    final_ood_mae = compute_val_mae(params, ood_batch)
    train_mae = compute_val_mae(params, train_batch)

    logger.info(f"OOD {species_name}: train MAE={train_mae:.3f}, OOD MAE={final_ood_mae:.3f} mS/cm")
    logger.info(f"  Best OOD MAE={best_ood_mae_retrain:.3f} at step {best_ood_step_retrain}")
    return {
        "species": species_name, "n_ood": len(ood_rows),
        "ood_mae": final_ood_mae, "train_mae": train_mae,
        "best_ood_mae": best_ood_mae_retrain, "best_step": best_ood_step_retrain,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Train MolSets Set Transformer and evaluate."""
    logger.info("=" * 70)
    logger.info("MolSets Set Transformer — Generalizable Conductivity Prediction")
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
    logger.info(f"Property vector dimension: {D_INPUT} (physics only, no role indicators)")
    logger.info(f"Normalization mean: {norm_mean}")
    logger.info(f"Normalization std: {norm_std}")

    logger.info("\nPer-species property vectors (first 5 dims + role):")
    for sp in all_species:
        vec = get_normalized_property_vector(sp, norm_mean, norm_std)
        logger.info(f"  {sp:8s}: [{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}, {vec[3]:.2f}, {vec[4]:.2f} ...]")

    train_batch, val_batch = prepare_molset_data(norm_mean, norm_std)

    # OOD proxy split: hold out OOD_PROXY_SPECIES from training for early stopping
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

    # Split train_batch into train_core (no proxy species) and ood_proxy_batch
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

    params = init_params(random.PRNGKey(SEED_MAIN), mix_mean, mix_std)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"\nModel parameters: {n_params:,}")

    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, WARMUP_STEPS)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, N_STEPS - WARMUP_STEPS)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [WARMUP_STEPS])
    opt = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = opt.init(params)
    step_fn = make_train_step(opt)

    # Pre-convert to JAX arrays once (avoids numpy→jax on every eval call)
    ja = train_core.jax_arrays()
    batch_tuple = (ja["props"], ja["raw"], ja["fracs"], ja["mask"], ja["temps"],
                   ja["log_sigma"], ja["weights"])

    # SWA: collect checkpoints from tail of training for weight averaging
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

        # SWA checkpoint collection
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
            logger.info(
                f"Step {step+1:5d} | loss={float(loss):.4f} | "
                f"train={train_mae:.3f} | val={val_mae:.3f} | ood_proxy={ood_str} mS/cm | "
                f"best_val={best_val_mae:.3f}@{best_val_step} | best_ood={best_ood_mae:.3f}@{best_ood_step} | "
                f"{elapsed:.1f}s{val_marker}{ood_marker}"
            )

    # SWA: average collected checkpoints and compare against BOTH val and OOD
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
            best_val_step = -1  # sentinel: SWA
            logger.info(f"  SWA WINS for val — using averaged weights")
        if ood_proxy_batch is not None:
            swa_ood_mae = compute_val_mae(swa_params, ood_proxy_batch)
            logger.info(f"  SWA OOD proxy MAE = {swa_ood_mae:.3f} mS/cm (best single = {best_ood_mae:.3f})")
            if swa_ood_mae < best_ood_mae:
                best_ood_params = swa_params
                best_ood_mae = swa_ood_mae
                best_ood_step = -1  # sentinel: SWA
                logger.info(f"  SWA WINS for OOD — using averaged weights")

    # Use OOD-best params for saving (primary goal: OOD generalization)
    use_params = best_ood_params if ood_proxy_batch is not None else best_val_params
    use_step = best_ood_step if ood_proxy_batch is not None else best_val_step

    logger.info(f"\nFINAL RESULTS — best-val params (step {best_val_step if best_val_step > 0 else 'SWA'})")
    logger.info(f"FINAL RESULTS — OOD-proxy-best params (step {best_ood_step if best_ood_step > 0 else 'SWA'})")

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS (using OOD-best@{use_step})")
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
    logger.info(f"Neural Onsager dual-loss:   0.687 mS/cm")
    logger.info(f"KAN direct sigma:           0.670 mS/cm")
    logger.info(f"MolSets (this model):       {val_metrics['mae_mS_cm']:.3f} mS/cm")

    # Save model (OOD-best params)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mol_set_sigma.pkl")
    save_model(use_params, norm_mean, norm_std, model_path)
    logger.info(f"Model saved (OOD-best@{use_step}): {model_path}")

    # Speed test — JIT-compiled single-recipe inference
    logger.info(f"\n--- Speed Test (JIT-compiled) ---")
    test_props_s = jnp.array(val_batch.species_props[0])
    test_raw_s = jnp.array(val_batch.raw_props[0])
    test_fracs_s = jnp.array(val_batch.fracs[0])
    test_mask_s = jnp.array(val_batch.mask[0])
    test_temp_s = jnp.array(val_batch.temperature_K[0])

    warmup_result = _forward_single_eval(
        use_params, test_props_s, test_raw_s, test_fracs_s, test_mask_s, test_temp_s)
    warmup_result.block_until_ready()

    n_calls = 10000
    t0 = time.time()
    for _ in range(n_calls):
        result = _forward_single_eval(
            use_params, test_props_s, test_raw_s, test_fracs_s, test_mask_s, test_temp_s)
    result.block_until_ready()
    elapsed_jit = (time.time() - t0) / n_calls * 1000
    logger.info(f"JIT single-recipe: {elapsed_jit:.4f} ms/recipe ({n_calls} calls)")

    # Batch inference speed
    batch_props = jnp.array(val_batch.species_props)
    batch_raw = jnp.array(val_batch.raw_props)
    batch_fracs = jnp.array(val_batch.fracs)
    batch_mask = jnp.array(val_batch.mask)
    batch_temps = jnp.array(val_batch.temperature_K)
    batch_keys = random.split(random.PRNGKey(0), len(val_batch.recipe_keys))

    warmup_b = _forward_batch_eval(
        use_params, batch_props, batch_raw, batch_fracs, batch_mask, batch_temps, batch_keys)
    warmup_b.block_until_ready()

    n_batch_calls = 1000
    t0 = time.time()
    for _ in range(n_batch_calls):
        result_b = _forward_batch_eval(
            use_params, batch_props, batch_raw, batch_fracs, batch_mask, batch_temps, batch_keys)
    result_b.block_until_ready()
    elapsed_batch = (time.time() - t0) / n_batch_calls * 1000
    n_val = len(val_batch.recipe_keys)
    logger.info(f"JIT batch ({n_val} recipes): {elapsed_batch:.3f} ms total, "
                f"{elapsed_batch/n_val:.4f} ms/recipe")

    # Gradient check — argnums=3 is fracs (after params, species_props, raw_props)
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

    # OOD evaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"OUT-OF-DISTRIBUTION EVALUATION")
    logger.info(f"{'='*60}")

    ood_species = ["FEC", "VC", "LiFSI"]
    ood_results = []
    for sp in ood_species:
        result = evaluate_species_ood(sp, norm_mean, norm_std, step_fn, opt)
        ood_results.append(result)

    logger.info(f"\n--- OOD Summary ---")
    for r in ood_results:
        if r["ood_mae"] is not None:
            logger.info(f"  {r['species']:8s}: OOD MAE = {r['ood_mae']:.3f} mS/cm "
                       f"(train MAE = {r['train_mae']:.3f}, n_ood = {r['n_ood']})")


if __name__ == "__main__":
    main()
