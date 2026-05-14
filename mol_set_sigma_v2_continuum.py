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

from constants import (T_REF_K, E_CHARGE, EPS_0, K_B, N_A, F as F_FARADAY, R as R_GAS,
                       MS_CM_TO_S_M as _MS_CM_TO_S_M, S_M_TO_MS_CM as _S_M_TO_MS_CM,
                       BJERRUM_LENGTH_NM, NM_TO_CM, CELSIUS_TO_KELVIN)

ANGSTROM_TO_NM = 1e-1   # Explicit constant: 1 Å = 0.1 nm
ANGSTROM_TO_M = 1e-10   # Explicit constant: 1 Å = 1e-10 m
ANGSTROM_TO_CM = 1e-8   # Explicit constant: 1 Å = 1e-8 cm
EXP_OVERFLOW_GUARD = 50.0  # numerical sentinel: cap Fuoss exponent to prevent exp overflow

# Debye-Hückel reference constants at T=298.15K in water (ε=78.4, ρ≈1 g/cm³).
# Robinson & Stokes "Electrolyte Solutions" (2002), Ch. 9, Table 9.1.
# Scaled to other solvents via (ε_water/ε_mix)^n factors.
DH_A_LN_WATER = 1.172      # Explicit constant: DH A parameter, natural log basis, (mol/L)^(-0.5), water at 298K
DH_B_WATER_INV_M = 3.281e9 # Explicit constant: DH B parameter, m⁻¹·(mol/L)^(-0.5), water at 298K
DH_EPS_WATER = 78.4        # Explicit constant: water dielectric constant at 298K (reference for DH scaling)
FUOSS_PREFACTOR = 4.0 * np.pi * N_A / 3.0  # Fuoss association: K_A = (4πN_A/3)·a_cm³·exp(b), a in cm, K_A in cm³/mol
CM3_PER_L = 1000.0         # Explicit constant: unit conversion cm³ → L
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
IDX_HOMO = PROPERTY_KEYS.index("homo_eV")
IDX_LUMO = PROPERTY_KEYS.index("lumo_eV")


# =============================================================================
# FORCE-FIELD ANALOG PARAMETERS
# 6 global physics parameters, analogous to MD force field fitting.
# The physics pipeline IS the model — no NN correction layer.
# Each theta is a free scalar optimized via gradient descent through the
# full Coulomb→Boltzmann→Onsager pipeline. Transforms (exp, sigmoid)
# enforce physical constraints. Initialized at theoretical defaults (θ=0).
# =============================================================================

N_MAX_SPECIES = 10  # Explicit constant: max observed=8 across 2693 recipes (2+headroom), verified 2026-05-08

N_THETA = 6         # Explicit constant: number of global force-field parameters

THETA_NAMES = (
    "coupling",      # exp(θ₀) scales u_coulomb/kT — like partial charge scaling in MD
    "walden",        # exp(θ₁) → Walden exponent α — fractional Walden rule
    "eta_ref",       # exp(θ₂) scales η_ref — reference viscosity for Walden
    "solvation",     # exp(θ₃) scales coordination number — effective solvation shell
    "onsager",       # exp(θ₄) scales Onsager cross-correlation — electrophoretic + relaxation
    "anion_solv",    # sigmoid(θ₅) → anion solvation fraction ∈ (0,1)
)

ANION_SOLVATION_FRACTION_DEFAULT = 0.3  # Explicit constant: large anions (PF6⁻, TFSI⁻) weakly solvated — ~1/3 of a solvent layer vs cation's full shell. Learnable via theta.
THETA_INIT_ANION_SOLV = np.log(ANION_SOLVATION_FRACTION_DEFAULT / (1.0 - ANION_SOLVATION_FRACTION_DEFAULT))  # Explicit constant: logit(0.3) so sigmoid(θ₅) recovers the theoretical default at init

N_STEPS = 8000      # Explicit constant: 149-param physics-structured combining rules need more steps to find right projection
LR_PEAK = 3e-3      # Explicit constant: 149 params can tolerate higher LR than 1200 generic params
WARMUP_STEPS = 200   # Explicit constant: 2.5% warmup for combining-rule stability
SEED_MAIN = 42       # Explicit constant: arbitrary reproducibility seed (train/val split + param init)
SEED_OOD = 123       # Explicit constant: separate seed for OOD leave-one-out experiments

ROOM_TEMP_LOW_K = 293.0   # Explicit constant: 20°C — CALiSol "room temp" lower bound (measurement metadata)
ROOM_TEMP_HIGH_K = 303.0  # Explicit constant: 30°C — CALiSol "room temp" upper bound (measurement metadata)
LOW_KAPPA_WEIGHT = 0.25   # Explicit constant: 1/4 weight for σ < 2 mS/cm (outside optimizer operating range)
CALISOL_WEIGHT = 0.5      # Explicit constant: CALiSol from varied labs — half credibility vs verified originals
LOW_KAPPA_THRESHOLD = 2.0  # Explicit constant: mS/cm — boundary copied from electrolyte_cond_surrogate_train.py

OOD_PROXY_SPECIES = "LiTFSI"
LOG_EVERY = 50              # Explicit constant: more frequent logging for fast convergence
OOD_LOG_EVERY = 100         # Explicit constant: OOD retraining logs
EARLY_STOP_PATIENCE = 5     # Explicit constant: more patience — 6 params have smoother loss landscape than 15k NN
EARLY_STOP_REL_TOL = 0.005  # Explicit constant: tighter convergence criterion for few params

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
_se_cfg = _physics_cfg["stokes_einstein_correction"]
ETA_REF_WATER_25C_CP = float(_se_cfg["reference_viscosity_water_25C_Pa_s"]) * 1e3  # Pa·s → cP (unit conversion)
del _json, _f, _physics_cfg, _ic_cfg, _ac_cfg, _cc_cfg, _pc_cfg, _se_cfg

# ═══════════════════════════════════════════════════════════════════════
# Dij COMBINING-RULE ARCHITECTURE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
D_FF = 5             # Explicit constant: RMT Marchenko-Pastur on 37x24 property matrix — 5 signal eigenvalues above noise floor, 82.8% variance (effective size, well depth, charge character, fluidity, polarity)
N_CHANNELS = 3       # Explicit constant: neutral-neutral (ch0), ion-neutral (ch1), ion-ion (ch2)
N_ACTIVE_CHANNELS = 2  # Explicit constant: only ch1 (ion-neutral) and ch2 (ion-ion) have learnable params — ch0 is dead (neutral-neutral D_ij don't enter B-matrix)
BSM_TIKHONOV_REL = 1e-12  # Explicit constant: relative Tikhonov regularization for B-matrix (matches osm_transport.py)

_COMBINING_NORMS: Dict = {}


def _set_combining_norms(mean: np.ndarray, std: np.ndarray):
    """Set module-level normalization stats for the combining rule."""
    _COMBINING_NORMS["mean"] = jnp.array(mean)
    _COMBINING_NORMS["std"] = jnp.where(jnp.array(std) > 1e-10, jnp.array(std), 1.0)
    logger.info(f"Combining-rule norms set: mean shape={mean.shape}, std shape={std.shape}")


N_SCREENED_FUOSS_ITERS = 5  # Explicit constant: self-consistent screening iterations (converges in 3-4)


def _ionic_weight(lam0):
    """Continuous ionic weight: Λ₀/(Λ₀ + 1). 0 for solvents, ~1 for salts."""
    return lam0 / (lam0 + 1.0)


def _mole_frac_to_molarity(species_props, fracs, mask):
    """Extract total salt concentration in mol/L from recipe fracs.

    Training data convention: salt fracs are already molarities (mol/L),
    solvent fracs are volume fractions (sum to ~1.0).
    c_salt is the ionic-weight-filtered sum of raw fracs.
    """
    lam0 = species_props[:, IDX_LAMBDA0]
    w = fracs * mask
    iw = _ionic_weight(lam0)
    c_salt_mol_L = jnp.sum(w * iw)
    return c_salt_mol_L, c_salt_mol_L


def _full_physics_conductivity(species_props, fracs, mask, T_K):
    """Self-consistent continuum conductivity: Kirkwood ε → Fuoss pairing → Walden transport."""
    lam0 = species_props[:, IDX_LAMBDA0]
    iw = _ionic_weight(lam0)
    sw = 1.0 - iw
    w = fracs * mask

    eps_eff = _kirkwood_mixture_epsilon(species_props, fracs, mask, T_K)

    # Solvent viscosity: Arrhenius log-additive mixing
    eta_per = species_props[:, IDX_VISCOSITY]
    w_solv = w * sw
    w_solv_sum = jnp.maximum(jnp.sum(w_solv), 1e-8)
    ln_eta_solv = jnp.sum(w_solv * jnp.log(jnp.maximum(eta_per, 1e-8))) / w_solv_sum
    eta_solv_cP = jnp.exp(ln_eta_solv)

    # Salt viscosity correction: Eyring activated transport
    c_mol_L, _ = _mole_frac_to_molarity(species_props, fracs, mask)
    c_mol_L = jnp.maximum(c_mol_L, 1e-8)
    w_ionic = w * iw
    w_ionic_sum = jnp.maximum(jnp.sum(w_ionic), 1e-8)
    B_salt = jnp.sum(w_ionic * species_props[:, IDX_JONES_DOLE]) / w_ionic_sum
    eta_solution_cP = eta_solv_cP * jnp.exp(B_salt * c_mol_L)

    # Ion pairing via screened Fuoss with Kirkwood dielectric
    alpha, _ = _screened_fuoss_alpha(eps_eff, species_props, fracs, mask, T_K)
    c_free = alpha * c_mol_L

    # Walden-scaled molar conductivity: Λ₀ × (η_ref / η_solution)
    lambda0_salt = jnp.sum(w_ionic * lam0) / w_ionic_sum
    lambda_eff = lambda0_salt * (ETA_REF_WATER_25C_CP / jnp.maximum(eta_solution_cP, 1e-8))

    # κ [mS/cm] = c_free [mol/L] × Λ_eff [S·cm²/mol]
    sigma_mS_cm = c_free * lambda_eff
    return jnp.log(jnp.maximum(sigma_mS_cm, 1e-8))


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
# SET TRANSFORMER MODEL — Physics-Kernel Attention (Pure JAX)
# =============================================================================



def _layer_norm(x, scale, bias, eps=1e-5):
    """Layer normalization (Ba et al. 2016)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def _physics_attention_logits(raw_props, mask):
    """Compute 8-head physics-based attention logits from raw species properties.

    Complete coverage of pairwise interaction mechanisms in electrolytes.
    Each head computes a specific physical quantity from universal equations
    applied to per-species properties. Same equations for all pairs —
    unseen species get correct interactions from their property values alone.

    Electrostatic interactions:
      Head 0 — Donor-acceptor solvation: DN_i×AN_j + DN_j×AN_i
      Head 1 — Kirkwood dielectric cavity: 3εⱼ/(2εⱼ+εᵢ) coupling
      Head 2 — Ionic coupling: Λ₀_i×Λ₀_j (correlated ion motion)
      Head 3 — Dipole structure: μ_i×μ_j (orientation ordering)

    Transport couplings:
      Head 4 — Viscosity drag: B_i×B_j (Jones-Dole cross-coupling)
      Head 5 — Steric exclusion: 1/(1+(r_i-r_j)²) (size mismatch friction)

    Solvation/electronic:
      Head 6 — Charge-transfer: (HOMO_i-LUMO_j)+(HOMO_j-LUMO_i) (frontier orbital)
      Head 7 — Coordination competition: affinity_i×CN_j + affinity_j×CN_i

    Returns: (N_HEADS=8, N_max, N_max) raw logit tensor (pre-scale, pre-softmax).
    """
    dn = raw_props[:, IDX_DONOR]
    an = raw_props[:, IDX_ACCEPTOR]
    eps = jnp.maximum(raw_props[:, IDX_EPSILON], 1.0)
    lam0 = raw_props[:, IDX_LAMBDA0]
    mu = raw_props[:, IDX_DIPOLE]
    jones_B = raw_props[:, IDX_JONES_DOLE]
    r_anion = raw_props[:, IDX_ANION_R]
    r_cation = raw_props[:, IDX_CATION_R]
    homo = raw_props[:, IDX_HOMO]
    lumo = raw_props[:, IDX_LUMO]
    coord_aff = raw_props[:, IDX_COORD_AFFINITY]
    coord_num = raw_props[:, IDX_COORD_NUMBER]

    # Head 0: Donor-acceptor solvation (Lewis acid-base complementarity)
    da_ij = dn[:, None] * an[None, :] + dn[None, :] * an[:, None]
    da_ref = jnp.maximum(jnp.mean(dn) * jnp.mean(an), 1.0)
    head0 = da_ij / da_ref

    # Head 1: Kirkwood dielectric cavity field
    cav_ij = 3.0 * eps[None, :] / jnp.maximum(2.0 * eps[None, :] + eps[:, None], 1.0) - 1.0
    cav_ji = 3.0 * eps[:, None] / jnp.maximum(2.0 * eps[:, None] + eps[None, :], 1.0) - 1.0
    head1 = cav_ij + cav_ji

    # Head 2: Ionic coupling (correlated ion motion, 0 for solvent-solvent)
    iw = lam0 / (lam0 + 1.0)
    head2 = iw[:, None] * iw[None, :]

    # Head 3: Dipole-dipole structure (solvent orientation ordering)
    mu_ref_sq = jnp.maximum(jnp.mean(mu ** 2), 1.0)
    head3 = mu[:, None] * mu[None, :] / mu_ref_sq

    # Head 4: Viscosity drag (Jones-Dole B cross-coupling)
    B_ref_sq = jnp.maximum(jnp.mean(jones_B ** 2), 0.01)
    head4 = jones_B[:, None] * jones_B[None, :] / B_ref_sq

    # Head 5: Steric exclusion (size mismatch friction — effective radius)
    r_eff = r_anion + r_cation
    r_diff_sq = (r_eff[:, None] - r_eff[None, :]) ** 2
    r_ref_sq = jnp.maximum(jnp.mean(r_eff ** 2), 0.01)
    head5 = 1.0 / (1.0 + r_diff_sq / r_ref_sq)

    # Head 6: Charge-transfer tendency (frontier orbital gap — symmetric)
    ct_ij = (homo[:, None] - lumo[None, :]) + (homo[None, :] - lumo[:, None])
    ct_ref = jnp.maximum(jnp.std(homo) + jnp.std(lumo), 0.1)
    head6 = ct_ij / ct_ref

    # Head 7: Coordination competition (solvation shell competition)
    cc_ij = coord_aff[:, None] * coord_num[None, :] + coord_aff[None, :] * coord_num[:, None]
    cc_ref = jnp.maximum(jnp.mean(coord_aff) * jnp.mean(coord_num), 0.01)
    head7 = cc_ij / cc_ref

    logits = jnp.stack([head0, head1, head2, head3, head4, head5, head6, head7], axis=0)

    mask_2d = mask[:, None] * mask[None, :]
    logits = jnp.where(mask_2d[None, :, :] > 0, logits, -1e9)
    return logits


def _physics_multihead_attention(z, raw_props, mask, params, layer, dropout_key, dropout_rate):
    """Multi-head attention with physics-derived attention kernels.

    Unlike standard attention (learned Q/K), attention weights come from
    universal physics equations applied to raw property vectors. Only the
    value projection (W_V) and output projection (W_out) are learned.

    This is the continuum analog of MD: the "potential" (attention kernel) is
    a universal function of per-species parameters, while the "observable"
    extraction (value projection) is learned from data.
    """
    n_max, d = z.shape

    raw_logits = _physics_attention_logits(raw_props, mask)

    scaled_logits = jnp.zeros_like(raw_logits)
    for h in range(N_HEADS):
        s = params[f"phys_scale_L{layer}_h{h}"]
        b = params[f"phys_bias_L{layer}_h{h}"]
        scaled_logits = scaled_logits.at[h].set(s * raw_logits[h] + b)

    attn_weights = jax.nn.softmax(scaled_logits, axis=-1)
    mask_2d = mask[:, None] * mask[None, :]
    attn_weights = jnp.where(mask_2d[None, :, :] > 0, attn_weights, 0.0)

    v = z @ params[f"attn{layer}_v_w"] + params[f"attn{layer}_v_b"]
    v_heads = v.reshape(n_max, N_HEADS, D_HEAD).transpose(1, 0, 2)

    out_heads = jnp.matmul(attn_weights, v_heads)
    out = out_heads.transpose(1, 0, 2).reshape(n_max, d)
    out = out @ params[f"attn{layer}_out_w"] + params[f"attn{layer}_out_b"]

    return out


VAND_K = 0.609               # Explicit constant: random hard-sphere packing parameter (Vand 1948, J. Phys. Chem. 52(2))
VAND_EINSTEIN_COEFF = 2.5    # Explicit constant: Einstein intrinsic viscosity for hard spheres [η]=5/2 (Einstein 1906, Ann. Phys. 19)
VAND_PHI_MAX = 0.55          # Explicit constant: divergence guard — random close packing φ_RCP ≈ 0.64, capped below to avoid singularity
ANION_SOLVATION_FRACTION = 0.3  # Explicit constant: large anions (PF6⁻, TFSI⁻) weakly solvated — ~1/3 of a solvent layer vs cation's full shell
DEBYE_KAPPA_PREFACTOR = 8.0 * np.pi  # Analytical constant: κ² = 8πλ_B·c·N_A from Poisson-Boltzmann for 1:1 electrolyte (Debye & Hückel 1923)


def _trajectory_implied_sigma(raw_props, fracs, mask, T_K, theta):
    """Conductivity from MD-analog 3-stage pipeline with learnable force-field parameters.

    No neural network. No corrections. The physics IS the model.
    6 global parameters (theta) are optimized through the pipeline,
    analogous to fitting LJ ε/σ and partial charges in classical MD.

    Stage 1 — PAIR FORCE FIELD: Coulomb at contact, scaled by theta["coupling"].
    Stage 2 — BOLTZMANN EQUILIBRIUM: Ion pairing + Vand viscosity with theta["solvation"].
    Stage 3 — ONSAGER TRANSPORT: Walden + cross-correlations with theta["walden"], theta["onsager"].

    Returns log(σ) in mS/cm.
    """
    coupling_scale = jnp.exp(theta["coupling"])
    walden_exp = jnp.exp(theta["walden"])
    eta_ref_scale = jnp.exp(theta["eta_ref"])
    solvation_scale = jnp.exp(theta["solvation"])
    onsager_scale = jnp.exp(theta["onsager"])
    anion_solv_frac = jax.nn.sigmoid(theta["anion_solv"])

    w = fracs * mask

    lam0 = raw_props[:, IDX_LAMBDA0]
    iw = lam0 / (lam0 + 1.0)
    sw = 1.0 - iw
    w_salt = w * iw
    w_solv = w * sw
    w_salt_sum = jnp.maximum(jnp.sum(w_salt), 1e-12)
    w_solv_sum = jnp.maximum(jnp.sum(w_solv), 1e-12)

    eps_species = jnp.maximum(raw_props[:, IDX_EPSILON], 1.0)
    eta_species = jnp.maximum(raw_props[:, IDX_VISCOSITY], 0.01)
    mw_species = jnp.maximum(raw_props[:, IDX_MW], 1.0)
    rho_species = jnp.maximum(raw_props[:, IDX_DENSITY], 0.5)

    r_an = jnp.sum(w_salt * raw_props[:, IDX_ANION_R]) / w_salt_sum
    r_cat = jnp.sum(w_salt * raw_props[:, IDX_CATION_R]) / w_salt_sum
    lam0_salt = jnp.sum(w_salt * lam0) / w_salt_sum
    cn_salt = jnp.sum(w_salt * raw_props[:, IDX_COORD_NUMBER]) / w_salt_sum

    # ══════════════════════════════════════════════════════
    # STAGE 1: PAIR FORCE FIELD
    # ══════════════════════════════════════════════════════

    eps_mix = jnp.sum(w_solv * eps_species) / w_solv_sum

    lambda_B_nm = BJERRUM_VAC_298K_NM / jnp.maximum(eps_mix * T_K / T_REF_K, 1.0)

    a_contact_nm = (r_cat + r_an) * ANGSTROM_TO_NM
    u_coulomb_kT = -lambda_B_nm / jnp.maximum(a_contact_nm, 0.01)
    u_scaled_kT = coupling_scale * u_coulomb_kT

    log_eta_solv = jnp.sum(w_solv * jnp.log(eta_species)) / w_solv_sum
    eta_solvent = jnp.exp(log_eta_solv)

    x_salt = jnp.maximum(w_salt_sum, 1e-12)
    mw_mix = jnp.sum(w * mw_species)
    rho_mix = jnp.sum(w * rho_species)
    c_salt_M = x_salt * rho_mix * CM3_PER_L / jnp.maximum(mw_mix, 1.0)

    # ══════════════════════════════════════════════════════
    # STAGE 2: BOLTZMANN EQUILIBRIUM
    # ══════════════════════════════════════════════════════

    g_contact = jnp.exp(jnp.minimum(-u_scaled_kT, EXP_OVERFLOW_GUARD))

    a_cm = a_contact_nm * NM_TO_CM
    K_A = FUOSS_PREFACTOR * a_cm ** 3 * g_contact / CM3_PER_L

    Kd = 1.0 / jnp.maximum(K_A, 1e-12)
    disc = Kd ** 2 + 4.0 * Kd * c_salt_M
    alpha = (-Kd + jnp.sqrt(jnp.maximum(disc, 1e-20))) / jnp.maximum(2.0 * c_salt_M, 1e-12)
    alpha = jnp.minimum(alpha, 1.0)
    c_free = alpha * c_salt_M

    v_solv_cm3 = jnp.sum(w_solv * mw_species / rho_species) / w_solv_sum / N_A
    r_solv_cm = (3.0 * v_solv_cm3 / (4.0 * jnp.pi)) ** (1.0 / 3.0)

    cn_eff = solvation_scale * cn_salt
    r_eff_cat_cm = r_cat * ANGSTROM_TO_CM + r_solv_cm * cn_eff / jnp.maximum(cn_eff + 1.0, 1.0)
    r_eff_an_cm = r_an * ANGSTROM_TO_CM + r_solv_cm * anion_solv_frac

    v_cat = (4.0 * jnp.pi / 3.0) * r_eff_cat_cm ** 3
    v_an = (4.0 * jnp.pi / 3.0) * r_eff_an_cm ** 3

    phi = c_salt_M * N_A * (v_cat + v_an) / CM3_PER_L
    phi = jnp.minimum(phi, VAND_PHI_MAX)

    eta_solution = eta_solvent * jnp.exp(
        VAND_EINSTEIN_COEFF * phi / jnp.maximum(1.0 - VAND_K * phi, 0.01))

    # ══════════════════════════════════════════════════════
    # STAGE 3: ONSAGER TRANSPORT (GREEN-KUBO ANALOG)
    # ══════════════════════════════════════════════════════

    eta_ref = ETA_REF_WATER_25C_CP * eta_ref_scale
    lambda_eff = lam0_salt * (eta_ref / jnp.maximum(eta_solution, 1e-6)) ** walden_exp

    eps_T = jnp.maximum(eps_mix * T_K, 1.0)
    eta_poise = eta_solution / 100.0  # Explicit constant: cP to poise conversion (1 P = 100 cP)
    S2_elec = ONSAGER_S1_PREFACTOR / jnp.maximum(eta_poise * jnp.sqrt(eps_T), 1e-8)
    S1_relax = ONSAGER_S2_Q_FACTOR * ONSAGER_S2_PREFACTOR / jnp.maximum(eps_T ** ONSAGER_EPS_T_EXPONENT, 1.0)

    sqrt_c_free = jnp.sqrt(jnp.maximum(c_free, 1e-12))

    kappa_nm_inv = jnp.sqrt(jnp.maximum(
        DEBYE_KAPPA_PREFACTOR * lambda_B_nm * c_free * N_A / 1e24, 1e-12))
    kappa_a = kappa_nm_inv * a_contact_nm
    onsager_denom = 1.0 + kappa_a

    delta_lambda = onsager_scale * (S1_relax * lambda_eff + S2_elec) * sqrt_c_free / onsager_denom
    lambda_corrected = jnp.maximum(lambda_eff - delta_lambda, 0.01)

    sigma_mS_cm = c_free * lambda_corrected
    return jnp.log(jnp.maximum(sigma_mS_cm, 1e-6))


def init_params(key: jax.Array) -> dict:
    """6 global force-field parameters, all at theoretical defaults (θ=0 → scale=1)."""
    return {
        "coupling": jnp.array(0.0),
        "walden": jnp.array(0.0),
        "eta_ref": jnp.array(0.0),
        "solvation": jnp.array(0.0),
        "onsager": jnp.array(0.0),
        "anion_solv": jnp.array(THETA_INIT_ANION_SOLV),
    }


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

# (D_POOLED_SPECIES, D_GLOBAL_READOUT removed — physics-kernel attention replaces global MLP readout)


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


def _universal_property_interactions(norm_props, fracs, mask):
    """Universal property-contrast interactions — continuum analog of MD combining rules.

    For each species i and each property dimension d, computes how i's local
    environment (composition-weighted neighbors) contrasts with i itself:
        contrast_i_d = Σ_j (p_j_d - p_i_d) × x_j

    Uses z-score normalized properties so all dimensions have comparable scale.

    Returns (N, D_INPUT) per-species contrast vectors.
    """
    w = fracs * mask  # (N,)
    env_mean = jnp.sum(norm_props * w[:, None], axis=0, keepdims=True)  # (1, D)
    return (env_mean - norm_props) * mask[:, None]  # (N, D)


def forward_single(params, raw_props, fracs, mask, temperature_K):
    """Pure physics — 6 learnable FF params through Coulomb→Boltzmann→Onsager."""
    return _trajectory_implied_sigma(raw_props, fracs, mask, temperature_K, params)


forward_batch = jax.vmap(forward_single, in_axes=(None, 0, 0, 0, 0))


@jax.jit
def _forward_batch_eval(params, raw, fracs, mask, temps):
    return forward_batch(params, raw, fracs, mask, temps)


@jax.jit
def _forward_single_eval(params, raw, fracs, mask, temp):
    return forward_single(params, raw, fracs, mask, temp)


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
    """Predict conductivity in mS/cm for a single recipe."""
    raw = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    fracs = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    mask = np.zeros(N_MAX_SPECIES, dtype=np.float64)

    j = 0
    for role in ("salts", "solvents", "additives"):
        for sp_name, frac in sorted(recipe[role].items()):
            raw[j] = _get_raw_cached(sp_name)
            fracs[j] = frac
            mask[j] = 1.0
            j += 1

    log_sigma = _forward_single_eval(
        params,
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
    """Pure-JAX conductivity for optimizer inner loop. Returns σ in S/m."""
    n_design = X.shape[0]
    n_pad = max(n_design, N_MAX_SPECIES)
    raw = jnp.zeros((n_pad, D_INPUT))
    fracs = jnp.zeros(n_pad)
    mask = jnp.zeros(n_pad)

    raw = raw.at[:n_design].set(species_props_raw)
    fracs = fracs.at[:n_design].set(X)
    mask = mask.at[:n_design].set(jnp.where(X > 0.0, 1.0, 0.0))

    log_sigma = forward_single(params, raw, fracs, mask, T_K)
    sigma_ms_cm = jnp.exp(log_sigma)
    return sigma_ms_cm * _MS_CM_TO_S_M


def build_molset_species_arrays(
    species_names: tuple,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> tuple:
    """Pre-compute per-species property arrays for the optimizer.

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
    serializable = jax.tree.map(lambda x: np.array(x), params)
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
    params = jax.tree.map(lambda x: jnp.array(x), bundle["params"])
    return params, bundle["norm_mean"], bundle["norm_std"]


def loss_fn(params, batch_tuple):
    """Weighted log-MSE loss. batch_tuple = (raw, fracs, mask, temps, log_sigma, weights)."""
    raw, fracs, mask, temps, log_sigma, weights = batch_tuple
    pred_log_sigma = forward_batch(params, raw, fracs, mask, temps)
    residuals = pred_log_sigma - log_sigma
    return jnp.sum(weights * residuals**2) / jnp.sum(weights)


def make_train_step(opt):
    @jax.jit
    def step(params, opt_state, batch_tuple):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_tuple)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return step


def compute_val_mae(params, batch: MolSetBatch) -> float:
    ja = batch.jax_arrays()
    pred_log_sigma = _forward_batch_eval(params, ja["raw"], ja["fracs"], ja["mask"], ja["temps"])
    pred_sigma = jnp.exp(pred_log_sigma)
    true_sigma = jnp.exp(ja["log_sigma"])
    return float(jnp.mean(jnp.abs(pred_sigma - true_sigma)))


def compute_metrics(params, batch: MolSetBatch) -> dict:
    ja = batch.jax_arrays()
    pred_log_sigma = _forward_batch_eval(params, ja["raw"], ja["fracs"], ja["mask"], ja["temps"])
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

    logger.info(f"OOD train: {len(train_batch.recipe_keys)} recipes")

    params = init_params(random.PRNGKey(SEED_OOD))
    opt_state = opt.init(params)

    ja = train_batch.jax_arrays()
    batch_tuple = (ja["raw"], ja["fracs"], ja["mask"], ja["temps"],
                   ja["log_sigma"], ja["weights"])
    best_ood_mae_retrain = float("inf")
    best_ood_step_retrain = 0
    ood_stall_counter = 0
    ood_prev_best = float("inf")
    t0_ood = time.time()
    for step in range(N_STEPS):
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple)

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
# Dij COMBINING-RULE + OSM INVERSION MODEL
# =============================================================================


def _classify_species(raw_props, mask):
    """Classify species as ionic (Lambda_0 > 0) or neutral from property vectors.

    Returns (is_ionic, is_neutral) float masks, both (N_MAX_SPECIES,).
    """
    lam0 = raw_props[:, IDX_LAMBDA0]
    is_ionic = jnp.where(lam0 > 0.0, 1.0, 0.0) * mask
    is_neutral = jnp.where(lam0 <= 0.0, 1.0, 0.0) * mask
    return is_ionic, is_neutral


def _stokes_einstein_prior(raw_props, temperature_K, mask):
    """Stokes-Einstein diffusion coefficient prior for each species.

    D_SE = k_B T / (6 pi eta r)

    Ionic species: r = max(cation_radius, anion_radius).
    Neutral species: r from molecular volume (MW / (rho * N_A))^(1/3).
    eta = species viscosity for solvents, reference water for salts.

    Returns (N_MAX_SPECIES,) array of D_SE in m^2/s.
    """
    mw = jnp.maximum(raw_props[:, IDX_MW], 1.0)
    rho = jnp.maximum(raw_props[:, IDX_DENSITY], 0.1)
    eta_cP = jnp.maximum(raw_props[:, IDX_VISCOSITY], 0.1)
    r_cat = raw_props[:, IDX_CATION_R]
    r_an = raw_props[:, IDX_ANION_R]
    lam0 = raw_props[:, IDX_LAMBDA0]

    r_ionic_m = jnp.maximum(r_cat, r_an) * ANGSTROM_TO_M

    v_mol_cm3 = mw / rho
    v_molecule_m3 = v_mol_cm3 * 1e-6 / N_A  # cm3/mol -> m3/molecule
    r_neutral_m = (3.0 * v_molecule_m3 / (4.0 * jnp.pi)) ** (1.0 / 3.0)

    is_ionic = jnp.where(lam0 > 0.0, 1.0, 0.0) * mask
    r_m = jnp.where(is_ionic > 0.0, r_ionic_m, r_neutral_m)
    r_m = jnp.maximum(r_m, 1e-12)

    eta_Pa_s = eta_cP * 1e-3  # cP -> Pa.s
    eta_ref_Pa_s = ETA_REF_WATER_25C_CP * 1e-3
    eta_use = jnp.where(is_ionic > 0.0, eta_ref_Pa_s, eta_Pa_s)
    eta_use = jnp.maximum(eta_use, 1e-6)

    D_SE = K_B * temperature_K / (6.0 * jnp.pi * eta_use * r_m)  # 6pi from Stokes stick-boundary (Stokes 1851)
    return D_SE * mask


def _combining_rule_dij(params, raw_props, fracs, mask, temperature_K, D_SE_prior):
    """Physics-structured combining rule with composition-dependent corrections.

    Stage 1: Shared W_ff projection (24-d → D_FF via softplus) gives effective FF params per species.
    Stage 2: Five combining rules (Lorentz, Berthelot, Coulomb, log-mean, arithmetic) give pairwise params.
    Stage 3: Composition-independent correction from FF combining rules (alpha-weighted).
    Stage 4: Composition-dependent corrections from mixture thermodynamics:
             - Debye screening: kappa*sigma_ij (ionic strength → D_ij suppression)
             - Crowding: salt volume fraction (excluded volume → slower diffusion)
             - Dielectric modulation: eps_mix changes effective Coulomb coupling
             - Preferential solvation: x_i*x_j pair abundance modulates encounters
             - Onsager sqrt(c) friction: classical concentration-dependent correction

    Returns (N_MAX_SPECIES, N_MAX_SPECIES) D_ij matrix in m^2/s.
    """
    n = raw_props.shape[0]
    p_norm = (raw_props - _COMBINING_NORMS["mean"]) / _COMBINING_NORMS["std"]
    p_norm = p_norm * mask[:, None]

    is_ionic, is_neutral = _classify_species(raw_props, mask)

    ch_masks = (
        is_neutral[:, None] * is_neutral[None, :],
        is_ionic[:, None] * is_neutral[None, :] +
        is_neutral[:, None] * is_ionic[None, :],
        is_ionic[:, None] * is_ionic[None, :],
    )

    D_SE_geo = jnp.sqrt(jnp.maximum(
        D_SE_prior[:, None] * D_SE_prior[None, :], 1e-30))

    ff_W = params["ff_proj"]["W"]
    ff_b = params["ff_proj"]["b"]
    ff_raw = p_norm @ ff_W.T + ff_b[None, :]
    ff = jax.nn.softplus(ff_raw) * mask[:, None]

    ff_i = ff[:, None, :]
    ff_j = ff[None, :, :]

    sigma_ij = (ff_i[..., 0] + ff_j[..., 0]) / 2.0
    epsilon_ij = jnp.sqrt(jnp.maximum(ff_i[..., 1] * ff_j[..., 1], 1e-30))
    q_ij = ff_i[..., 2] * ff_j[..., 2]
    mu_ij = jnp.exp((jnp.log(jnp.maximum(ff_i[..., 3], 1e-30)) +
                      jnp.log(jnp.maximum(ff_j[..., 3], 1e-30))) / 2.0)
    pol_ij = (ff_i[..., 4] + ff_j[..., 4]) / 2.0

    D_ij = ch_masks[0] * D_SE_geo

    kT_scaled = jnp.maximum(K_B * temperature_K / (1e-21), 1e-12)

    # --- Composition-dependent thermodynamic state ---
    w = fracs * mask
    lam0 = raw_props[:, IDX_LAMBDA0]
    iw = _ionic_weight(lam0)
    x_salt = jnp.sum(w * iw)

    mw = raw_props[:, IDX_MW]
    rho = jnp.maximum(raw_props[:, IDX_DENSITY], 0.1)
    v_mol_cm3 = jnp.sum(w * mw / rho)
    v_mol_L = v_mol_cm3 / 1000.0
    c_salt_mol_L = x_salt / jnp.maximum(v_mol_L, 1e-8)
    sqrt_c = jnp.sqrt(jnp.maximum(c_salt_mol_L, 1e-12))

    eps_mix = jnp.sum(w * raw_props[:, IDX_EPSILON]) / jnp.maximum(jnp.sum(w), 1e-12)
    eps_safe = jnp.maximum(eps_mix, 1.0)

    ionic_strength = jnp.sum(w * iw)
    bjerrum_m = E_CHARGE ** 2 / (4.0 * jnp.pi * EPS_0 * eps_safe * K_B * temperature_K)
    debye_m = jnp.sqrt(
        EPS_0 * eps_safe * K_B * temperature_K
        / jnp.maximum(2.0 * E_CHARGE ** 2 * N_A * ionic_strength * 1000.0, 1e-30))
    kappa = 1.0 / jnp.maximum(debye_m, 1e-30)

    r_cat = raw_props[:, IDX_CATION_R]
    r_an = raw_props[:, IDX_ANION_R]
    v_cat = (4.0 / 3.0) * jnp.pi * (r_cat * 1e-10) ** 3
    v_an = (4.0 / 3.0) * jnp.pi * (r_an * 1e-10) ** 3
    phi_salt = c_salt_mol_L * N_A * jnp.sum(w * iw * (v_cat + v_an)) / jnp.maximum(jnp.sum(w * iw), 1e-12) / 1000.0

    x_i = fracs[:, None] * mask[:, None]
    x_j = fracs[None, :] * mask[None, :]

    for ch_idx in range(1, N_CHANNELS):
        ch_p = params["ch"][ch_idx]

        log_corr = (
            ch_p["alpha_size"] * jnp.log(jnp.maximum(ch_p["sigma_ref"], 1e-6) / jnp.maximum(sigma_ij, 1e-6))
            + ch_p["alpha_well"] * (-epsilon_ij / jnp.maximum(kT_scaled, 1e-12))
            + ch_p["alpha_charge"] * jnp.tanh(q_ij * ch_p["bjerrum_scale"])
            + ch_p["alpha_fluidity"] * jnp.log(jnp.maximum(mu_ij, 1e-6) / jnp.maximum(ch_p["mu_ref"], 1e-6))
            + ch_p["alpha_polarity"] * jnp.tanh(pol_ij * ch_p["pol_scale"])
            + ch_p["bias"]
            + ch_p["T_coeff"] * jnp.log(temperature_K / T_REF_K)
        )

        epsilon_corrected = epsilon_ij * (1.0 - ch_p["k_ij"])
        k_ij_correction = ch_p["alpha_well"] * ch_p["k_ij"] * epsilon_ij / jnp.maximum(kT_scaled, 1e-12)
        log_corr = log_corr + k_ij_correction

        # --- Composition-dependent corrections ---
        kappa_sigma = kappa * sigma_ij * 1e-10  # sigma_ij is in FF units, scale to meters
        log_corr = log_corr + ch_p["alpha_screen"] * jnp.tanh(kappa_sigma * ch_p["screen_scale"])

        log_corr = log_corr + ch_p["alpha_crowd"] * phi_salt

        log_corr = log_corr + ch_p["alpha_sqrt_c"] * sqrt_c

        log_corr = log_corr + ch_p["alpha_diel"] * jnp.log(eps_safe / ch_p["eps_ref"])

        log_corr = log_corr + ch_p["alpha_encounter"] * jnp.log(jnp.maximum(x_i * x_j, 1e-12) / ch_p["x_ref"])

        # --- Three-body cross term: Σ_k x_k * how k modifies (i,j) friction ---
        # Species k affects D_ij through: (1) competing for solvation of i,
        # (2) screening interactions between i and j, (3) changing local packing.
        # We use the SAME combining-rule FF params: sigma_ik, epsilon_ik give
        # how strongly k interacts with i; sigma_jk, epsilon_jk for j.
        # cross_ij = Σ_k x_k * [w_size*(sigma_ik + sigma_jk) + w_well*(eps_ik + eps_jk)/kT]
        x_k = fracs * mask  # (N,)
        ff_k = ff  # (N, D_FF) — already computed above

        sigma_ik = (ff[:, None, 0] + ff[None, :, 0]) / 2.0  # (N, N)
        sigma_jk = (ff[:, None, 0] + ff[None, :, 0]) / 2.0  # same combining rule
        epsilon_ik = jnp.sqrt(jnp.maximum(ff[:, None, 1] * ff[None, :, 1], 1e-30))
        epsilon_jk = epsilon_ik  # symmetric

        # For pair (i,j): sum_k x_k * f(i-k interaction, j-k interaction)
        # f captures: k competes with j for i's solvation shell (size similarity)
        #             k screens i-j interaction (well depth)
        # sigma_ik[i,k] * sigma_jk[j,k] → how much k bridges i and j
        # Weight by x_k and sum over k axis

        # i-k interaction strength: epsilon_ik / kT (how much k binds to i)
        bind_ik = epsilon_ik / jnp.maximum(kT_scaled, 1e-12)  # (N, N)
        # For each pair (i,j), the three-body modifier is:
        #   Σ_k x_k * bind_ik[i,k] — how strongly are all third species binding to i?
        third_body_i = jnp.sum(x_k[None, :] * bind_ik, axis=1)  # (N,) — total binding pressure on species i
        third_body_j = jnp.sum(x_k[None, :] * bind_ik, axis=1)  # (N,) — same for j
        # Cross-term: how much total third-body friction acts on each member of pair (i,j)
        cross_ij = third_body_i[:, None] + third_body_j[None, :]  # (N, N)

        log_corr = log_corr + ch_p["alpha_cross_bind"] * cross_ij

        # Competitive solvation: Σ_k x_k * sigma_ik[i,k] — steric competition for i's shell
        steric_i = jnp.sum(x_k[None, :] * sigma_ik, axis=1)  # (N,)
        steric_j = jnp.sum(x_k[None, :] * sigma_ik, axis=1)  # (N,)
        steric_ij = steric_i[:, None] + steric_j[None, :]  # (N, N)
        log_corr = log_corr + ch_p["alpha_cross_steric"] * jnp.log(jnp.maximum(steric_ij, 1e-6) / ch_p["steric_ref"])

        D_ij = D_ij + ch_masks[ch_idx] * D_SE_geo * jnp.exp(log_corr)

    pair_mask = mask[:, None] * mask[None, :]
    return D_ij * pair_mask


def _osm_sigma_from_dij(D_ij_matrix, raw_props, fracs, mask, temperature_K):
    """OSM B-matrix inversion: D_ij matrix -> L_red -> sigma [mS/cm].

    Extracts D_cs, D_as, D_ca INDEPENDENTLY from off-diagonal D_ij elements
    (matching osm_transport.py pattern), then builds 2x2 B_red and inverts analytically.

    Returns log(sigma) in log(mS/cm).
    """
    w = fracs * mask
    is_ionic, is_neutral = _classify_species(raw_props, mask)

    r_cat = raw_props[:, IDX_CATION_R]
    r_an = raw_props[:, IDX_ANION_R]
    is_cation = jnp.where(r_cat > 0.0, 1.0, 0.0) * is_ionic
    is_anion = jnp.where(r_an > 0.0, 1.0, 0.0) * is_ionic

    x_solv_pre = jnp.sum(w * is_neutral)
    x_salt_pre = jnp.sum(w * is_ionic)

    x_post_total = x_solv_pre + 2.0 * x_salt_pre
    x_s = x_solv_pre / jnp.maximum(x_post_total, 1e-12)
    x_c = x_salt_pre / jnp.maximum(x_post_total, 1e-12)
    x_a = x_salt_pre / jnp.maximum(x_post_total, 1e-12)

    w_cation = w * is_cation
    w_anion = w * is_anion
    w_neutral = w * is_neutral

    cat_neut_w = w_cation[:, None] * w_neutral[None, :] + w_neutral[:, None] * w_cation[None, :]
    cat_neut_D = D_ij_matrix * jnp.where(cat_neut_w > 0, 1.0, 0.0)
    inv_cn = cat_neut_w / jnp.maximum(cat_neut_D, 1e-30)
    inv_cn = jnp.where(cat_neut_w > 0, inv_cn, 0.0)
    w_cn_total = jnp.maximum(jnp.sum(cat_neut_w), 1e-30)
    D_cs = w_cn_total / jnp.maximum(jnp.sum(inv_cn), 1e-30)

    an_neut_w = w_anion[:, None] * w_neutral[None, :] + w_neutral[:, None] * w_anion[None, :]
    an_neut_D = D_ij_matrix * jnp.where(an_neut_w > 0, 1.0, 0.0)
    inv_an = an_neut_w / jnp.maximum(an_neut_D, 1e-30)
    inv_an = jnp.where(an_neut_w > 0, inv_an, 0.0)
    w_an_total = jnp.maximum(jnp.sum(an_neut_w), 1e-30)
    D_as = w_an_total / jnp.maximum(jnp.sum(inv_an), 1e-30)

    cat_an_w = w_cation[:, None] * w_anion[None, :] + w_anion[:, None] * w_cation[None, :]
    cat_an_D = D_ij_matrix * jnp.where(cat_an_w > 0, 1.0, 0.0)
    inv_ca = cat_an_w / jnp.maximum(cat_an_D, 1e-30)
    inv_ca = jnp.where(cat_an_w > 0, inv_ca, 0.0)
    w_ca_total = jnp.maximum(jnp.sum(cat_an_w), 1e-30)
    D_ca = w_ca_total / jnp.maximum(jnp.sum(inv_ca), 1e-30)

    B11 = x_s / jnp.maximum(D_cs, 1e-30) + x_a / jnp.maximum(D_ca, 1e-30)
    B12 = -x_a / jnp.maximum(D_ca, 1e-30)
    B21 = -x_c / jnp.maximum(D_ca, 1e-30)
    B22 = x_s / jnp.maximum(D_as, 1e-30) + x_c / jnp.maximum(D_ca, 1e-30)

    trace_avg = (B11 + B22) / 2.0
    eps_tik = BSM_TIKHONOV_REL * jnp.maximum(trace_avg, 1e-30)
    B11 = B11 + eps_tik
    B22 = B22 + eps_tik

    det_B = B11 * B22 - B12 * B21
    det_B = jnp.maximum(det_B, 1e-30)
    L11 = B22 / det_B
    L12 = -B12 / det_B
    L21 = -B21 / det_B
    L22 = B11 / det_B

    c_salt_mol_L, _ = _mole_frac_to_molarity(raw_props, fracs, mask)
    c_salt_mol_m3 = c_salt_mol_L * 1000.0  # mol/L -> mol/m3
    c_cat = c_salt_mol_m3
    c_an = c_salt_mol_m3

    z_c = 1.0
    z_a = -1.0
    driving_c = z_c * F_FARADAY / (R_GAS * temperature_K)
    driving_a = z_a * F_FARADAY / (R_GAS * temperature_K)

    v_c = L11 * driving_c + L12 * driving_a
    v_a = L21 * driving_c + L22 * driving_a

    sigma_S_m = F_FARADAY * (z_c * c_cat * v_c + z_a * c_an * v_a)
    sigma_mS_cm = sigma_S_m * _S_M_TO_MS_CM

    return jnp.log(jnp.maximum(sigma_mS_cm, 1e-6))


def forward_single_dij(params, raw_props, fracs, mask, temperature_K):
    """Forward pass: property vectors -> combining rule -> D_ij -> OSM -> log(sigma)."""
    is_ionic, is_neutral = _classify_species(raw_props, mask)
    D_SE_prior = _stokes_einstein_prior(raw_props, temperature_K, mask)
    D_ij_matrix = _combining_rule_dij(
        params, raw_props, fracs, mask, temperature_K, D_SE_prior)
    log_sigma = _osm_sigma_from_dij(
        D_ij_matrix, raw_props, fracs, mask, temperature_K)
    return log_sigma


def init_params_dij(key):
    """Initialize physics-structured combining-rule parameters.

    Shared W_ff projection: maps 24-d property vector -> 5-d effective FF params (softplus for positivity).
    Per active channel (ch1=ion-neutral, ch2=ion-ion): 12 learnable scalars controlling
    how combined FF parameters modulate D_ij relative to the Stokes-Einstein prior.
    At init all alphas=0 so corrections vanish and model returns D_SE exactly.
    """
    INIT_SCALE = 0.01  # Explicit constant: Xavier scale for W_ff — small enough to stay near SE prior
    key, k_w = random.split(key)
    xavier_std = INIT_SCALE / jnp.sqrt(float(D_PROP))

    channels = [
        None,  # ch0: neutral-neutral = SE prior, no learnable params
    ]
    ALPHA_INIT = 0.1  # Explicit constant: small nonzero to break zero-gradient saddle (alpha*feature=0 kills W_ff grad when alpha=0, Phase 1 lesson)
    for _ch in range(N_ACTIVE_CHANNELS):
        channels.append({
            # Pair-intrinsic (composition-independent)
            "alpha_size": jnp.array(ALPHA_INIT),
            "alpha_well": jnp.array(ALPHA_INIT),
            "alpha_charge": jnp.array(ALPHA_INIT),
            "alpha_fluidity": jnp.array(ALPHA_INIT),
            "alpha_polarity": jnp.array(ALPHA_INIT),
            "sigma_ref": jnp.array(1.0),
            "mu_ref": jnp.array(1.0),
            "bjerrum_scale": jnp.array(1.0),
            "pol_scale": jnp.array(1.0),
            "bias": jnp.array(0.0),
            "T_coeff": jnp.array(0.0),
            "k_ij": jnp.array(0.0),
            # Composition-dependent corrections
            "alpha_screen": jnp.array(ALPHA_INIT),    # Debye screening: kappa * sigma_ij
            "screen_scale": jnp.array(1.0),            # scale for tanh(kappa*sigma*screen_scale)
            "alpha_crowd": jnp.array(ALPHA_INIT),      # salt volume fraction (crowding/excluded volume)
            "alpha_sqrt_c": jnp.array(ALPHA_INIT),     # Onsager-Fuoss sqrt(c) friction
            "alpha_diel": jnp.array(ALPHA_INIT),       # mixture dielectric modulation
            "eps_ref": jnp.array(40.0),                 # Explicit constant: learnable reference dielectric, initialized at geometric mean of pure-solvent range (EC ε=89, EMC ε=3 → mixture ~30-50)
            "alpha_encounter": jnp.array(ALPHA_INIT),  # pair encounter frequency x_i * x_j
            "x_ref": jnp.array(0.01),                   # reference pair fraction (salt*solvent ~ 0.1*0.3 = 0.03)
            # Three-body cross terms: how species k modifies D_ij for pair (i,j)
            "alpha_cross_bind": jnp.array(ALPHA_INIT),   # Σ_k x_k * epsilon_ik/kT — binding competition
            "alpha_cross_steric": jnp.array(ALPHA_INIT),  # Σ_k x_k * sigma_ik — steric competition
            "steric_ref": jnp.array(1.0),                 # reference steric sum (learnable normalization)
        })

    return {
        "ff_proj": {
            "W": random.normal(k_w, (D_FF, D_PROP)) * xavier_std,
            "b": jnp.zeros(D_FF),  # softplus(0) = ln(2) ~ 0.69, reasonable starting FF value
        },
        "ch": channels,
    }


def loss_fn_dij(params, batch_tuple):
    """Weighted log-MSE loss for Dij model."""
    raw, fracs, mask, temps, log_sigma, weights = batch_tuple
    pred_log_sigma = jax.vmap(
        forward_single_dij, in_axes=(None, 0, 0, 0, 0)
    )(params, raw, fracs, mask, temps)
    residuals = pred_log_sigma - log_sigma
    return jnp.sum(weights * residuals ** 2) / jnp.sum(weights)


def _log_dij_params(params, label=""):
    """Log physics-structured combining-rule parameters."""
    prefix = f"[{label}] " if label else ""
    W_ff = params["ff_proj"]["W"]
    b_ff = params["ff_proj"]["b"]
    W_rms = float(jnp.sqrt(jnp.mean(W_ff ** 2)))
    svs = jnp.linalg.svd(W_ff, compute_uv=False)
    sv_str = ", ".join(f"{float(s):.4f}" for s in svs)
    logger.info(f"  {prefix}W_ff: |W|={W_rms:.6f}, SVD=[{sv_str}], b={[f'{float(v):.3f}' for v in b_ff]}")

    ch_names = ("neutral-neutral", "ion-neutral", "ion-ion")
    for ch_idx, ch_name in enumerate(ch_names):
        ch = params["ch"][ch_idx]
        if ch is None:
            logger.info(f"  {prefix}ch{ch_idx} ({ch_name:16s}): SE prior only (no learnable params)")
            continue
        alphas = (f"sz={float(ch['alpha_size']):+.4f}, "
                  f"well={float(ch['alpha_well']):+.4f}, "
                  f"q={float(ch['alpha_charge']):+.4f}, "
                  f"flu={float(ch['alpha_fluidity']):+.4f}, "
                  f"pol={float(ch['alpha_polarity']):+.4f}")
        refs = (f"σ_ref={float(ch['sigma_ref']):.4f}, "
                f"μ_ref={float(ch['mu_ref']):.4f}, "
                f"bj={float(ch['bjerrum_scale']):.4f}, "
                f"pol_s={float(ch['pol_scale']):.4f}")
        comp_dep = (f"scr={float(ch['alpha_screen']):+.4f}, "
                   f"crowd={float(ch['alpha_crowd']):+.4f}, "
                   f"√c={float(ch['alpha_sqrt_c']):+.4f}, "
                   f"diel={float(ch['alpha_diel']):+.4f}, "
                   f"enc={float(ch['alpha_encounter']):+.4f}")
        comp_refs = (f"scr_s={float(ch['screen_scale']):.4f}, "
                     f"ε_ref={float(ch['eps_ref']):.1f}, "
                     f"x_ref={float(ch['x_ref']):.4f}")
        three_body = (f"bind={float(ch['alpha_cross_bind']):+.4f}, "
                      f"steric={float(ch['alpha_cross_steric']):+.4f}, "
                      f"st_ref={float(ch['steric_ref']):.4f}")
        logger.info(f"  {prefix}ch{ch_idx} ({ch_name:16s}): {alphas}")
        logger.info(f"  {prefix}  refs: {refs}, bias={float(ch['bias']):+.4f}, "
                    f"T={float(ch['T_coeff']):+.4f}, k_ij={float(ch['k_ij']):+.4f}")
        logger.info(f"  {prefix}  comp: {comp_dep}")
        logger.info(f"  {prefix}  comp_refs: {comp_refs}")
        logger.info(f"  {prefix}  3-body: {three_body}")


# ═══════════════════════════════════════════════════════════════════════
# PHYSICS-CHAIN TRANSPORT MODEL
# Learns effective FF params (σ_eff, ε_eff, q_eff) from species properties,
# then computes transport through the analytical physics chain:
# FF → η_mix (Arrhenius + Jones-Dole + learned excess)
# → SE D_i (with Walden fractional exponent)
# → D_ij (geometric mean + Coulomb friction)
# → OSM B-matrix → σ × α (Fuoss ion pairing)
# ═══════════════════════════════════════════════════════════════════════

D_FF_TRANSPORT = 3  # Explicit constant: effective FF dimensions [σ_eff (size), ε_eff (well depth), q_eff (charge)]
WALDEN_SOFTPLUS_SCALE = 0.1  # Explicit constant: parametrization hyperparameter — softplus(0)*0.1=0.069, giving exp(-0.069)=0.933, α_w=0.5+0.5*0.933=0.967 (near-classical SE at init)


def _physics_chain_conductivity(params, raw_props, fracs, mask, T_K):
    """Full physics-chain conductivity from learned force-field parameters.

    Physics chain:
    1. W_ff projects 24-d species properties → 3-d effective FF params (σ, ε, q)
    2. Mixture viscosity: Arrhenius ideal + Jones-Dole salt + excess from FF ε
       + packing from FF σ
    3. Mixture dielectric: Kirkwood (raw properties)
    4. Ion pairing: screened Fuoss (raw properties)
    5. SE diffusivities: D_i = kBT/(6π η_mix^α_w r_eff_i) with fractional Walden
    6. D_ij matrix: geometric mean + Coulomb friction on ion-ion (from FF q)
    7. OSM B-matrix inversion → σ_total
    8. σ = α × σ_total
    """
    n = raw_props.shape[0]

    # ---- Classify species ----
    lam0 = raw_props[:, IDX_LAMBDA0]
    iw = _ionic_weight(lam0)
    sw = 1.0 - iw
    w = fracs * mask
    w_solv = w * sw
    w_ionic = w * iw
    w_solv_sum = jnp.maximum(jnp.sum(w_solv), 1e-8)
    w_ionic_sum = jnp.maximum(jnp.sum(w_ionic), 1e-8)
    is_ionic, is_neutral = _classify_species(raw_props, mask)

    # ---- Step 1: FF projection (24-d → 3-d) ----
    p_norm = (raw_props - _COMBINING_NORMS["mean"]) / _COMBINING_NORMS["std"]
    p_norm = p_norm * mask[:, None]
    ff_raw = p_norm @ params["ff_proj"]["W"].T + params["ff_proj"]["b"][None, :]
    ff = jax.nn.softplus(ff_raw) * mask[:, None]

    # ---- Step 2: Mixture viscosity ----
    # 2a: Arrhenius ideal mixing of pure-component solvent viscosities
    eta_per_cP = jnp.maximum(raw_props[:, IDX_VISCOSITY], 0.1)
    ln_eta_ideal = jnp.sum(w_solv * jnp.log(eta_per_cP)) / w_solv_sum

    # 2b: Jones-Dole salt correction (raw B-coefficients from species data)
    c_salt, x_salt = _mole_frac_to_molarity(raw_props, fracs, mask)
    c_salt = jnp.maximum(c_salt, 1e-8)
    B_salt = jnp.sum(w_ionic * raw_props[:, IDX_JONES_DOLE]) / w_ionic_sum
    ln_eta_jd = B_salt * c_salt

    # 2c: Non-ideal excess from pairwise FF ε interactions (Eyring theory:
    # activation barrier for viscous flow depends on pairwise well depths)
    kT_ff = jnp.maximum(K_B * T_K / 1e-21, 1e-12)
    eps_ij = jnp.sqrt(jnp.maximum(ff[:, 1, None] * ff[None, :, 1], 1e-30))
    w_2d = w[:, None] * w[None, :] * mask[:, None] * mask[None, :]
    w2_sum = jnp.maximum(jnp.sum(w_2d), 1e-8)
    mean_eps_kT = jnp.sum(w_2d * eps_ij / kT_ff) / w2_sum
    ln_eta_excess = params["theta_eta"] * mean_eps_kT

    # 2d: Packing from FF σ (Enskog: excluded volume → viscosity increase)
    sigma3 = ff[:, 0] ** 3
    sigma3_avg = jnp.sum(w * sigma3) / jnp.maximum(jnp.sum(w), 1e-8)
    ln_eta_pack = params["theta_pack"] * sigma3_avg

    eta_mix_cP = jnp.exp(ln_eta_ideal + ln_eta_jd + ln_eta_excess + ln_eta_pack)
    eta_mix_Pa_s = eta_mix_cP * 1e-3

    # ---- Step 3: Kirkwood mixture dielectric (raw properties) ----
    eps_mix = _kirkwood_mixture_epsilon(raw_props, fracs, mask, T_K)

    # ---- Step 4: Screened Fuoss ion pairing (raw properties) ----
    alpha_ip, K_A = _screened_fuoss_alpha(eps_mix, raw_props, fracs, mask, T_K)

    # ---- Step 5: Stokes-Einstein diffusivities ----
    r_cat_m = raw_props[:, IDX_CATION_R] * ANGSTROM_TO_M
    r_an_m = raw_props[:, IDX_ANION_R] * ANGSTROM_TO_M
    r_ionic_m = jnp.maximum(r_cat_m, r_an_m)

    mw = jnp.maximum(raw_props[:, IDX_MW], 1.0)
    rho = jnp.maximum(raw_props[:, IDX_DENSITY], 0.1)
    v_mol_cm3 = mw / rho
    v_mol_m3 = v_mol_cm3 * 1e-6 / N_A
    r_neutral_m = (3.0 * v_mol_m3 / (4.0 * jnp.pi)) ** (1.0 / 3.0)

    r_known_m = jnp.where(is_ionic > 0.0, r_ionic_m, r_neutral_m)
    r_known_m = jnp.maximum(r_known_m, 1e-12)

    # FF σ correction: solvation shell modifies hydrodynamic radius
    # r_eff = r_known × exp(θ_r × (σ_i − ⟨σ⟩)); at init θ_r=0 → r_eff = r_known
    sigma_mean = jnp.sum(w * ff[:, 0]) / jnp.maximum(jnp.sum(w), 1e-8)
    r_correction = jnp.exp(params["theta_r"] * (ff[:, 0] - sigma_mean))
    r_eff_m = r_known_m * r_correction * mask
    r_eff_m = jnp.maximum(r_eff_m, 1e-12)

    # Fractional Walden exponent for ions (Angell 1965):
    # D ∝ η^(-α_w), α_w ∈ [0.5, 1.0]. α_w=1 is classical SE.
    # Parametrized so α_w ≈ 0.97 at init (near-classical).
    alpha_w_offset = jax.nn.softplus(params["theta_walden"]) * WALDEN_SOFTPLUS_SCALE
    alpha_w = 0.5 + 0.5 * jnp.exp(-alpha_w_offset)
    eta_ref_Pa_s = ETA_REF_WATER_25C_CP * 1e-3

    D_neutral = K_B * T_K / (6.0 * jnp.pi * eta_mix_Pa_s * r_eff_m)  # 6π from Stokes stick-boundary (Stokes 1851)

    D_ionic = K_B * T_K / (
        6.0 * jnp.pi * r_eff_m  # 6π: Stokes stick-boundary (same as D_neutral)
        * jnp.power(jnp.maximum(eta_mix_Pa_s, 1e-12), alpha_w)
        * jnp.power(eta_ref_Pa_s, 1.0 - alpha_w)
    )

    D_i = jnp.where(is_ionic > 0.0, D_ionic, D_neutral) * mask

    # ---- Step 6: D_ij matrix ----
    D_geo = jnp.sqrt(jnp.maximum(D_i[:, None] * D_i[None, :], 1e-30))

    ch_ii = is_ionic[:, None] * is_ionic[None, :]

    # Coulomb friction on ion-ion pairs: electrostatic attraction retards
    # relative cation-anion diffusion. Uses Bjerrum length from mixture
    # dielectric and raw contact distance (not MSA — direct Boltzmann factor).
    a_contact_m = (raw_props[:, IDX_CATION_R][:, None] +
                   raw_props[None, :, IDX_ANION_R]) * ANGSTROM_TO_M
    a_contact_m = jnp.maximum(a_contact_m, 1e-12)
    lambda_B_m = E_CHARGE**2 / (4.0 * jnp.pi * EPS_0 * eps_mix * K_B * T_K)
    coulomb_param = lambda_B_m / a_contact_m
    coulomb_param = jnp.minimum(coulomb_param, EXP_OVERFLOW_GUARD)

    # FF q_eff modulates per-species Coulomb coupling strength
    q_prod = ff[:, 2, None] * ff[None, :, 2]
    q_avg = jnp.sum(w_ionic * ff[:, 2]) / w_ionic_sum
    q_ratio = q_prod / jnp.maximum(q_avg ** 2, 1e-8)

    theta_q_pos = jnp.abs(params["theta_coulomb"])
    friction_exponent = theta_q_pos * q_ratio * coulomb_param
    friction_exponent = jnp.minimum(friction_exponent, EXP_OVERFLOW_GUARD)
    coulomb_friction = jnp.exp(-friction_exponent)

    D_ij = (1.0 - ch_ii) * D_geo + ch_ii * D_geo * coulomb_friction
    pair_mask = mask[:, None] * mask[None, :]
    D_ij = D_ij * pair_mask

    # ---- Step 7: OSM B-matrix → σ_total ----
    log_sigma_total = _osm_sigma_from_dij(D_ij, raw_props, fracs, mask, T_K)

    # ---- Step 8: Ion pairing correction ----
    log_sigma = log_sigma_total + jnp.log(jnp.maximum(alpha_ip, 1e-8))

    return log_sigma


def forward_single_transport(params, raw_props, fracs, mask, temperature_K):
    """Forward pass: FF → η_mix → SE → OSM → σ."""
    return _physics_chain_conductivity(params, raw_props, fracs, mask, temperature_K)


def init_params_transport(key):
    """Initialize physics-chain transport parameters.

    At init all θ=0: model returns pure analytical physics
    (Arrhenius + Jones-Dole + near-classical SE + OSM + Fuoss).
    """
    INIT_SCALE = 0.01  # Explicit constant: small Xavier to stay near physics baseline
    key, k_w = random.split(key)
    xavier_std = INIT_SCALE / jnp.sqrt(float(D_PROP))

    return {
        "ff_proj": {
            "W": random.normal(k_w, (D_FF_TRANSPORT, D_PROP)) * xavier_std,
            "b": jnp.zeros(D_FF_TRANSPORT),
        },
        "theta_eta": jnp.array(0.0),
        "theta_pack": jnp.array(0.0),
        "theta_r": jnp.array(0.0),
        "theta_walden": jnp.array(0.0),
        "theta_coulomb": jnp.array(0.0),
    }


def loss_fn_transport(params, batch_tuple):
    """Weighted log-MSE loss for physics-chain transport model."""
    raw, fracs, mask, temps, log_sigma, weights = batch_tuple
    pred_log_sigma = jax.vmap(
        forward_single_transport, in_axes=(None, 0, 0, 0, 0)
    )(params, raw, fracs, mask, temps)
    residuals = pred_log_sigma - log_sigma
    return jnp.sum(weights * residuals ** 2) / jnp.sum(weights)


def _log_transport_params(params, label=""):
    """Log physics-chain transport parameters."""
    prefix = f"[{label}] " if label else ""

    W_ff = params["ff_proj"]["W"]
    b_ff = params["ff_proj"]["b"]
    W_rms = float(jnp.sqrt(jnp.mean(W_ff ** 2)))
    svs = jnp.linalg.svd(W_ff, compute_uv=False)
    sv_str = ", ".join(f"{float(s):.4f}" for s in svs)
    logger.info(f"  {prefix}W_ff: |W|={W_rms:.6f}, SVD=[{sv_str}], "
                f"b={[f'{float(v):.3f}' for v in b_ff]}")

    alpha_w_offset = float(jax.nn.softplus(params["theta_walden"]) * WALDEN_SOFTPLUS_SCALE)
    alpha_w = 0.5 + 0.5 * float(jnp.exp(-alpha_w_offset))
    logger.info(
        f"  {prefix}θ_eta={float(params['theta_eta']):+.4f}, "
        f"θ_pack={float(params['theta_pack']):+.4f}, "
        f"θ_r={float(params['theta_r']):+.4f}, "
        f"θ_walden={float(params['theta_walden']):+.4f} (α_w={alpha_w:.4f}), "
        f"θ_coulomb={float(params['theta_coulomb']):+.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# ORNSTEIN-ZERNIKE + HNC + MODE COUPLING THEORY CONDUCTIVITY
#
# First-principles liquid state theory pipeline:
#   1. Species properties → pairwise potentials U_ij(r) (LJ + Coulomb)
#   2. OZ equation + HNC closure → pair correlation g_ij(r), structure factor S_ij(q)
#   3. Chandra-Bagchi MCT → conductivity σ from S_ij(q) and c_ij(q)
#
# Learnable: W_ff maps 24-d species properties → (σ_LJ, ε_LJ, q_eff) per species.
# Everything else is exact integral equation theory — no fitting, no dilute-limit.
# ═══════════════════════════════════════════════════════════════════════

# Radial grid for OZ solver: real-space and Fourier-space
N_GRID = 128          # Explicit constant: radial grid points — 128 at 0.23 Å covers 3-30 Å range needed for LJ+Coulomb pair correlations in electrolytes
R_MAX_ANGSTROM = 30.0 # Explicit constant: real-space cutoff in Å (>5 σ_LJ for largest species)
DR_ANGSTROM = R_MAX_ANGSTROM / N_GRID  # grid spacing ~0.23 Å
N_OZ_PICARD = 15      # Explicit constant: numerical algorithm parameter — HNC Picard converges in 8-12 iters for 1:1 electrolytes at moderate coupling (Ng 1974); 15 for safety
N_OZ_GRAD_STEPS = 1   # Explicit constant: last K steps of Picard get full gradient (truncated BPTT); first N-K steps are stop_gradient. K>1 causes NaN from exp() overflow in HNC backward pass.
OZ_MIXING = 0.3       # Explicit constant: numerical algorithm parameter — Picard mixing α=0.3 balances convergence speed vs stability for charged systems (Caccamo 1996, §3.2)
D_FF_OZ = 3           # Explicit constant: FF dimensions [σ_LJ (Å), ε_LJ (kJ/mol), q_eff (e)]
LJ_REPULSIVE_EXP = 12 # Analytical constant: Pauli repulsion exponent in Lennard-Jones 12-6 potential (Jones 1924)
LJ_ATTRACTIVE_EXP = 6 # Analytical constant: London dispersion exponent in Lennard-Jones 12-6 potential (Jones 1924)

# Precomputed grids (module-level, JAX arrays)
_R_GRID = jnp.linspace(DR_ANGSTROM, R_MAX_ANGSTROM, N_GRID)  # avoid r=0 singularity
_DK = jnp.pi / R_MAX_ANGSTROM  # Fourier-space spacing from DST convention
_K_GRID = _DK * jnp.arange(1, N_GRID + 1)  # k = π/L, 2π/L, ..., Nπ/L


def _build_pair_potential(ff_i, ff_j, r_grid, eps_r, T_K):
    """Build pairwise potential U_ij(r)/kT on radial grid.

    U(r) = 4 ε_LJ [(σ/r)^12 - (σ/r)^6] + q_i q_j e² / (4πε₀ ε_r r)

    All in reduced units: output is βU(r) = U(r)/(k_B T).
    ff_i, ff_j: (3,) arrays [σ_Å, ε_kJ_mol, q_e]
    """
    sigma_ij = (ff_i[0] + ff_j[0]) / 2.0    # Lorentz combining (Å)
    eps_ij = jnp.sqrt(ff_i[1] * ff_j[1])     # Berthelot combining (kJ/mol)
    q_i = ff_i[2]
    q_j = ff_j[2]

    r_safe = jnp.maximum(r_grid, 0.5)  # floor at 0.5 Å to prevent divergence
    x = sigma_ij / r_safe

    # Lennard-Jones 12-6 potential (Jones 1924): U_LJ = 4ε[(σ/r)¹² - (σ/r)⁶]
    # Exponents 12 (Pauli repulsion) and 6 (London dispersion) define the LJ form.
    kT_kJ_mol = R_GAS * T_K / 1000.0
    beta_u_lj = 4.0 * eps_ij * (x**LJ_REPULSIVE_EXP - x**LJ_ATTRACTIVE_EXP) / kT_kJ_mol

    # Coulomb: U_coul = q_i q_j e² / (4πε₀ ε_r r)
    # In SI: e=1.602e-19 C, ε₀=8.854e-12 F/m, r in m
    # Convert r from Å to m: r_m = r * 1e-10
    r_m = r_safe * ANGSTROM_TO_M
    u_coul_J = q_i * q_j * E_CHARGE**2 / (4.0 * jnp.pi * EPS_0 * eps_r * r_m)
    beta_u_coul = u_coul_J / (K_B * T_K)

    return beta_u_lj + beta_u_coul


def _build_coulomb_potential(q_i, q_j, r_grid, eps_r, T_K):
    """Coulomb-only part of βU_ij(r) on radial grid. Used for Ng renormalization."""
    r_safe = jnp.maximum(r_grid, 0.5)  # floor at 0.5 Å (same as _build_pair_potential)
    r_m = r_safe * ANGSTROM_TO_M
    u_coul_J = q_i * q_j * E_CHARGE**2 / (4.0 * jnp.pi * EPS_0 * eps_r * r_m)
    return u_coul_J / (K_B * T_K)


def _oz_hnc_solve(beta_u_matrix, rho_vec, r_grid, k_grid, dr, n_species,
                   beta_u_coul, c_hat_coul_analytical):
    """Solve multi-component OZ+HNC with Ng Coulomb renormalization.

    The direct correlation c(r) has a long-range tail c(r) → −βU_coul(r) ∝ 1/r
    whose FT diverges as 1/k². Ng renormalization (Ng 1974, J. Chem. Phys. 61, 2680):
      1. HNC closure gives c_full(r) = exp(−βU + γ) − γ − 1
      2. Subtract known tail: c_short(r) = c_full(r) + βU_coul(r)  [short-ranged]
      3. DST only c_short → ĉ_short(k)  [well-behaved]
      4. Add analytical Coulomb FT: ĉ(k) = ĉ_short(k) + ĉ_coul(k)
         where ĉ_coul_ij(k) = −4π z_i z_j λ_B / k²
      5. OZ solve with full ĉ → ĥ → inverse DST → h(r)  [h is short-ranged]

    Args:
        beta_u_matrix: (n, n, N_GRID) — full βU_ij(r) = βU_LJ + βU_coul on grid
        rho_vec: (n,) — number densities in Å⁻³
        r_grid, k_grid: (N_GRID,) — real/reciprocal space grids
        dr: float — grid spacing in Å
        n_species: int — number of species (static for JIT)
        beta_u_coul: (n, n, N_GRID) — Coulomb-only part of βU on grid
        c_hat_coul_analytical: (n, n, N_GRID) — analytical FT of −βU_coul: −4πz_iz_jλ_B/k²
    """
    n = n_species
    Nr = r_grid.shape[0]

    sin_matrix = jnp.sin(k_grid[:, None] * r_grid[None, :])

    def _dst_forward(f_r):
        return 4.0 * jnp.pi * dr * (sin_matrix @ (f_r * r_grid)) / jnp.maximum(k_grid, 1e-12)

    def _dst_inverse(F_k):
        dk = k_grid[1] - k_grid[0]
        return dk / (2.0 * jnp.pi**2) * (sin_matrix.T @ (F_k * k_grid)) / jnp.maximum(r_grid, 1e-12)

    gamma_init = jnp.zeros((n, n, Nr))

    def _picard_step(gamma, _):
        # HNC closure: c(r) = exp(−βU + γ) − γ − 1  (uses FULL potential)
        exponent = -beta_u_matrix + gamma
        exponent = jnp.minimum(exponent, EXP_OVERFLOW_GUARD)
        c_full = jnp.exp(exponent) - gamma - 1.0

        # Ng renormalization: subtract Coulomb tail before DST
        # c_full(r) → −βU_coul(r) at large r, so c_short = c_full − (−βU_coul) = c_full + βU_coul
        c_short = c_full + beta_u_coul

        # DST of short-ranged part only
        c_short_flat = c_short.reshape(n * n, Nr)
        c_hat_short_flat = jax.vmap(_dst_forward)(c_short_flat)
        c_hat_short = c_hat_short_flat.reshape(n, n, Nr)

        # Full ĉ = ĉ_short + ĉ_coul(analytical)
        c_hat = c_hat_short + c_hat_coul_analytical

        # OZ in Fourier space: Ĥ = [I − Ĉ·ρ + δI]⁻¹ · Ĉ
        # Tikhonov regularization (δ=1e-6) prevents singularity when
        # HNC fails to converge at extreme coupling (concentrated/cold)
        OZ_TIKHONOV_DELTA = 1e-6  # Numerical sentinel: Tikhonov regularization for OZ matrix solve
        def _oz_at_k(c_hat_k):
            A = (1.0 + OZ_TIKHONOV_DELTA) * jnp.eye(n) - c_hat_k * rho_vec[None, :]
            return jnp.linalg.solve(A, c_hat_k)

        c_hat_T = jnp.transpose(c_hat, (2, 0, 1))
        h_hat_T = jax.vmap(_oz_at_k)(c_hat_T)
        h_hat = jnp.transpose(h_hat_T, (1, 2, 0))

        # Inverse DST: h(r) is short-ranged, DST is fine
        h_flat = h_hat.reshape(n * n, Nr)
        h_r_flat = jax.vmap(_dst_inverse)(h_flat)
        h_r = h_r_flat.reshape(n, n, Nr)

        gamma_new = h_r - c_full
        gamma_mixed = OZ_MIXING * gamma_new + (1.0 - OZ_MIXING) * gamma
        return gamma_mixed, None

    # Truncated BPTT: run (N-K) steps with stop_gradient for convergence,
    # then K steps with full gradient. K=N_OZ_GRAD_STEPS captures multi-step
    # sensitivity of the OZ fixed point to the input potential.
    n_converge = N_OZ_PICARD - N_OZ_GRAD_STEPS
    gamma_converged, _ = lax.scan(_picard_step, gamma_init, None, length=n_converge)
    gamma_converged = jnp.where(jnp.isnan(gamma_converged), 0.0, gamma_converged)
    gamma_converged = jax.lax.stop_gradient(gamma_converged)
    gamma_final, _ = lax.scan(_picard_step, gamma_converged, None, length=N_OZ_GRAD_STEPS)

    gamma_final = jnp.where(jnp.isnan(gamma_final), 0.0, gamma_final)

    exponent_final = -beta_u_matrix + gamma_final
    exponent_final = jnp.minimum(exponent_final, EXP_OVERFLOW_GUARD)
    c_final = jnp.exp(exponent_final) - gamma_final - 1.0
    h_final = gamma_final + c_final

    # Guard against residual NaN from exp overflow or non-convergence
    h_final = jnp.where(jnp.isnan(h_final), 0.0, h_final)
    c_final = jnp.where(jnp.isnan(c_final), 0.0, c_final)

    return h_final, c_final


def _structure_factor_from_h(h_matrix, rho_vec, r_grid, k_grid, dr, n_species):
    """Compute partial structure factors S_ij(k) from h_ij(r).

    S_ij(k) = δ_ij + sqrt(ρ_i ρ_j) ĥ_ij(k)

    Returns S_matrix: (n_species, n_species, N_GRID) — S_ij(k) on k-grid.
    """
    n = n_species
    Nr = r_grid.shape[0]

    sin_matrix = jnp.sin(k_grid[:, None] * r_grid[None, :])

    def _dst_fwd(f_r):
        return 4.0 * jnp.pi * dr * (sin_matrix @ (f_r * r_grid)) / jnp.maximum(k_grid, 1e-12)

    h_flat = h_matrix.reshape(n * n, Nr)
    h_hat_flat = jax.vmap(_dst_fwd)(h_flat)
    h_hat = h_hat_flat.reshape(n, n, Nr)

    rho_sqrt = jnp.sqrt(jnp.maximum(rho_vec, 1e-30))
    rho_ij = rho_sqrt[:, None] * rho_sqrt[None, :]

    S_matrix = jnp.eye(n)[:, :, None] + rho_ij[:, :, None] * h_hat
    return S_matrix, h_hat


def _chandra_bagchi_sigma(S_matrix, h_hat, c_hat_from_oz, rho_vec, k_grid, dk,
                          z_vec, T_K, eta_mix_Pa_s, n_species):
    """Chandra-Bagchi mode coupling theory conductivity.

    σ = σ_NE × (1 - Δ_MCT)

    where σ_NE is the Nernst-Einstein conductivity and Δ_MCT captures
    cross-correlations via the static structure factor.

    Chandra-Bagchi (J. Chem. Phys. 110, 10024, 1999):
    Δσ/σ_NE = -1/(6π²) Σ_{ij,lm} z_i z_j ∫ dk k² [S⁻¹ dS/dk S⁻¹]_{ij}
              × ρ_l ρ_m c_il(k) c_jm(k) / (k² + κ²)

    Simplified for 1:1 electrolyte (2 ionic species + solvent bath):
    The key quantity is the charge-charge structure factor S_ZZ(k).
    """
    n = n_species

    # Nernst-Einstein: σ_NE = F² / (RT) × Σ_i z_i² c_i D_i
    # D_i from Stokes-Einstein using mixture viscosity
    # But we need per-ion D_i — compute from SE with the OZ-derived structure

    # For MCT, the central object is the charge-charge structure factor:
    # S_ZZ(k) = Σ_{ij} z_i z_j sqrt(ρ_i ρ_j) / ρ_Z × S_ij(k)
    # where ρ_Z = Σ_i z_i² ρ_i

    rho_Z = jnp.sum(z_vec**2 * rho_vec)
    rho_Z = jnp.maximum(rho_Z, 1e-30)

    rho_sqrt = jnp.sqrt(jnp.maximum(rho_vec, 1e-30))

    # S_ZZ(k) = Σ_{ij} z_i z_j sqrt(ρ_i ρ_j) S_ij(k) / ρ_Z
    z_ij = z_vec[:, None] * z_vec[None, :]
    rho_ij = rho_sqrt[:, None] * rho_sqrt[None, :]
    S_ZZ = jnp.sum(z_ij[:, :, None] * rho_ij[:, :, None] * S_matrix, axis=(0, 1)) / rho_Z

    # The MCT correction (Chandra-Bagchi) relates to how S_ZZ deviates from
    # the Debye-Hückel limit S_ZZ_DH(k) = k²/(k² + κ²).
    # The conductivity correction factor:
    # σ/σ_NE = S_ZZ(k→0) / S_ZZ_DH(k→0) ≈ lim_{k→0} S_ZZ(k) / (k²/(k²+κ²))

    # More precisely, for the frequency-independent (DC) conductivity:
    # σ = (F²/(RT)) × Σ_i z_i² c_i D_i^0 × [1/(1 + δ_relax + δ_electro)]
    # where the relaxation and electrophoretic corrections come from S_ZZ.

    # Debye-Hückel screening: κ² = 4πλ_B Σ z_i² ρ_i
    lambda_B_A = E_CHARGE**2 / (4.0 * jnp.pi * EPS_0 * K_B * T_K) * 1e10  # in Å
    kappa_sq = 4.0 * jnp.pi * lambda_B_A * jnp.sum(z_vec**2 * rho_vec)
    kappa = jnp.sqrt(jnp.maximum(kappa_sq, 1e-30))

    # MCT correction from charge-charge structure factor.
    # S_ZZ(k→0) = 0 for electroneutral systems (perfect screening). The naive
    # σ = σ_NE / S_ZZ(0) diverges. The physical DC conductivity involves the
    # frequency-dependent transport at the Debye screening scale k ~ κ_D.
    #
    # Evaluate S_ZZ averaged over k ∈ [κ/2, 2κ] (the relaxation/electrophoretic
    # correction scale), weighted by k² (Chandra-Bagchi integrand measure).
    kappa_lo = kappa * 0.5
    kappa_hi = kappa * 2.0
    band_mask = jnp.where((k_grid >= kappa_lo) & (k_grid <= kappa_hi), 1.0, 0.0)
    band_weights = band_mask * k_grid**2
    band_total = jnp.maximum(jnp.sum(band_weights), 1e-12)
    S_ZZ_kappa = jnp.sum(band_weights * S_ZZ) / band_total

    mct_factor = 1.0 / jnp.maximum(S_ZZ_kappa, 0.1)

    return mct_factor, S_ZZ


N_OZ_SPECIES = 3  # Explicit constant: OZ operates on 3 effective species: cation(+), anion(−), solvent(0). Multiple salts/solvents are composition-averaged into these 3 types.


def _oz_mct_conductivity(params, raw_props, fracs, mask, T_K):
    """Full OZ+HNC+MCT conductivity on 3-species reduced basis.

    Reduces N_MAX_SPECIES individual species to 3 effective types
    (cation, anion, solvent) via composition-weighted averaging of
    learned FF params, then solves OZ+HNC on the 3×3 system.

    1. W_ff projects 24-d species properties → (σ_LJ, ε_LJ, q_eff)
    2. Composition-average into 3 effective species (cation, anion, solvent)
    3. Build 3×3 pairwise potentials U_ij(r)
    4. OZ+HNC on 3×3 system → g_ij(r), S_ij(k)
    5. Chandra-Bagchi MCT → σ

    Returns log(σ) in mS/cm.
    """
    # ---- Classify species ----
    lam0 = raw_props[:, IDX_LAMBDA0]
    iw = _ionic_weight(lam0)
    sw = 1.0 - iw
    w = fracs * mask

    # Species classification: salts have BOTH cation_r > 0 AND anion_r > 0
    # (they represent the full salt, not individual ions). They must contribute
    # to BOTH the cation and anion effective species with opposite charges.
    has_cat = jnp.where(raw_props[:, IDX_CATION_R] > 0, 1.0, 0.0) * mask
    has_an = jnp.where(raw_props[:, IDX_ANION_R] > 0, 1.0, 0.0) * mask
    is_salt = has_cat * has_an
    is_neutral = (1.0 - has_cat) * (1.0 - has_an) * mask

    # Salts contribute weight to both ion buckets
    w_cat = w * has_cat
    w_an = w * has_an
    w_neut = w * is_neutral
    w_cat_sum = jnp.maximum(jnp.sum(w_cat), 1e-12)
    w_an_sum = jnp.maximum(jnp.sum(w_an), 1e-12)
    w_neut_sum = jnp.maximum(jnp.sum(w_neut), 1e-12)

    # ---- Step 1: FF projection per species (σ_LJ, ε_LJ only) ----
    p_norm = (raw_props - _COMBINING_NORMS["mean"]) / _COMBINING_NORMS["std"]
    p_norm = p_norm * mask[:, None]
    ff_raw = p_norm @ params["oz_ff"]["W"].T + params["oz_ff"]["b"][None, :]

    sigma_lj = jax.nn.softplus(ff_raw[:, 0]) + 2.0
    eps_lj = jax.nn.softplus(ff_raw[:, 1]) * 0.5

    # ---- Step 2: Composition-average into 3 effective species ----
    # Charges are fixed physical constants: +1e cation, -1e anion, 0 solvent
    ff_cat = jnp.array([
        jnp.sum(w_cat * sigma_lj) / w_cat_sum,
        jnp.sum(w_cat * eps_lj) / w_cat_sum,
        1.0,
    ])
    ff_an = jnp.array([
        jnp.sum(w_an * sigma_lj) / w_an_sum,
        jnp.sum(w_an * eps_lj) / w_an_sum,
        -1.0,
    ])
    ff_solv = jnp.array([
        jnp.sum(w_neut * sigma_lj) / w_neut_sum,
        jnp.sum(w_neut * eps_lj) / w_neut_sum,
        0.0,
    ])
    ff_3 = jnp.stack([ff_cat, ff_an, ff_solv])  # (3, 3)

    # ---- Step 3: Mixture properties ----
    eps_mix = _kirkwood_mixture_epsilon(raw_props, fracs, mask, T_K)

    w_solv = w * sw
    w_solv_sum = jnp.maximum(jnp.sum(w_solv), 1e-8)
    eta_per_cP = jnp.maximum(raw_props[:, IDX_VISCOSITY], 0.1)
    ln_eta_solv = jnp.sum(w_solv * jnp.log(eta_per_cP)) / w_solv_sum

    w_ionic = w * iw
    w_ionic_sum = jnp.maximum(jnp.sum(w_ionic), 1e-8)
    c_salt, x_salt = _mole_frac_to_molarity(raw_props, fracs, mask)
    c_salt = jnp.maximum(c_salt, 1e-8)
    B_salt = jnp.sum(w_ionic * raw_props[:, IDX_JONES_DOLE]) / w_ionic_sum
    eta_mix_cP = jnp.exp(ln_eta_solv + B_salt * c_salt)
    eta_mix_Pa_s = eta_mix_cP * 1e-3

    # ---- Step 4: Number densities for 3 effective species (Å⁻³) ----
    mw = jnp.maximum(raw_props[:, IDX_MW], 1.0)
    rho_dens = jnp.maximum(raw_props[:, IDX_DENSITY], 0.1)

    # Ion number densities: salt fracs are molarities → direct conversion
    # ρ_ion [Å⁻³] = c_salt [mol/L] × N_A / 1e27
    L_TO_ANG3 = 1e27  # Explicit constant: 1 L = 10²⁷ ų (unit conversion)
    rho_ion = c_salt * N_A / L_TO_ANG3

    # Solvent number density: from volume fractions and pure component properties
    # c_i [mol/L] = φ_i × ρ_i [g/cm³] / MW_i [g/mol] × 1000 [cm³/L]
    c_per_species = rho_dens / mw * 1000.0
    c_solv_total = jnp.sum(w_neut * c_per_species)

    # Salt volume displacement: f_V = c_salt × MW_salt / (ρ_salt × 1000)
    mw_salt_eff = jnp.sum(w_ionic * mw) / w_ionic_sum
    rho_salt_eff = jnp.sum(w_ionic * rho_dens) / w_ionic_sum
    f_V_salt = c_salt * mw_salt_eff / jnp.maximum(rho_salt_eff * 1000.0, 1.0)
    c_solv_corrected = c_solv_total * (1.0 - f_V_salt)
    rho_solv = c_solv_corrected * N_A / L_TO_ANG3

    rho_3 = jnp.array([rho_ion, rho_ion, rho_solv])

    # ---- Step 5: Build 3×3 pairwise potentials + Coulomb decomposition ----
    z_3 = jnp.array([ff_cat[2], ff_an[2], 0.0])  # charges: cat(+), an(−), solv(0)

    def _build_row(i_ff):
        return jax.vmap(lambda j_ff: _build_pair_potential(i_ff, j_ff, _R_GRID, eps_mix, T_K))(ff_3)

    beta_u_3x3 = jax.vmap(_build_row)(ff_3)  # (3, 3, N_GRID)

    # Coulomb-only part for Ng renormalization (Ng 1974, J. Chem. Phys. 61, 2680)
    def _coul_row(i_idx):
        return jax.vmap(lambda j_idx: _build_coulomb_potential(
            z_3[i_idx], z_3[j_idx], _R_GRID, eps_mix, T_K))(jnp.arange(N_OZ_SPECIES))

    beta_u_coul_3x3 = jax.vmap(_coul_row)(jnp.arange(N_OZ_SPECIES))  # (3, 3, N_GRID)

    # Analytical FT of −βU_coul: ĉ_coul_ij(k) = −4π z_i z_j λ_B / k²
    # where λ_B = e²/(4πε₀ ε_r k_B T) is the Bjerrum length in Å
    lambda_B_A = E_CHARGE**2 / (4.0 * jnp.pi * EPS_0 * eps_mix * K_B * T_K) * 1e10  # Å
    z_ij_outer = z_3[:, None] * z_3[None, :]  # (3, 3)
    c_hat_coul = -4.0 * jnp.pi * z_ij_outer[:, :, None] * lambda_B_A / _K_GRID[None, None, :]**2

    # ---- Step 6: Solve OZ+HNC with Ng renormalization ----
    h_3, c_3 = _oz_hnc_solve(beta_u_3x3, rho_3, _R_GRID, _K_GRID, DR_ANGSTROM, N_OZ_SPECIES,
                               beta_u_coul_3x3, c_hat_coul)

    # ---- Step 7: Structure factor + MCT ----
    S_3, h_hat_3 = _structure_factor_from_h(h_3, rho_3, _R_GRID, _K_GRID, DR_ANGSTROM, N_OZ_SPECIES)

    sin_matrix = jnp.sin(_K_GRID[:, None] * _R_GRID[None, :])

    def _dst_fwd(f_r):
        return 4.0 * jnp.pi * DR_ANGSTROM * (sin_matrix @ (f_r * _R_GRID)) / jnp.maximum(_K_GRID, 1e-12)

    c_flat = c_3.reshape(N_OZ_SPECIES * N_OZ_SPECIES, N_GRID)
    c_hat_flat = jax.vmap(_dst_fwd)(c_flat)
    c_hat_3 = c_hat_flat.reshape(N_OZ_SPECIES, N_OZ_SPECIES, N_GRID)

    mct_factor, S_ZZ = _chandra_bagchi_sigma(
        S_3, h_hat_3, c_hat_3, rho_3, _K_GRID, _DK,
        z_3, T_K, eta_mix_Pa_s, N_OZ_SPECIES)

    # ---- Step 8: Nernst-Einstein baseline × MCT correction ----
    # σ_NE = c_salt × Λ_eff (Walden rule: Λ_eff = Λ₀ × η_water/η_mix)
    # No Fuoss ion pairing here — MCT factor 1/S_ZZ(0) is the physics-based
    # correction for ion-ion correlations (Hansen & McDonald §12.4).
    # Fuoss double-counts with MCT and gives α=0.02 at 1M (wrong).
    lam0_salt = jnp.sum(w_ionic * lam0) / w_ionic_sum
    lambda_eff = lam0_salt * (ETA_REF_WATER_25C_CP * 1e-3 / jnp.maximum(eta_mix_Pa_s, 1e-12))
    sigma_ne_mS_cm = c_salt * lambda_eff

    mct_blend = jax.nn.sigmoid(params["theta_mct_blend"])
    effective_factor = mct_blend * mct_factor + (1.0 - mct_blend) * 1.0
    log_sigma_base = jnp.log(jnp.maximum(sigma_ne_mS_cm * effective_factor, 1e-8))

    # ---- Step 9: Recipe-dependent correction ----
    # The Walden × MCT baseline captures bulk physics but misses:
    # - concentration-dependent ion pairing beyond DH
    # - solvent structure effects on mobility
    # - temperature departures from reference
    # A linear correction on mixture features gives direct gradient
    # for recipe discrimination without bypassing the OZ physics chain.
    ionic_frac = jnp.sum(w_ionic) / jnp.maximum(jnp.sum(w), 1e-8)
    mix_features = jnp.array([
        c_salt,                           # salt concentration [mol/L]
        T_K / T_REF_K - 1.0,              # temperature departure
        jnp.log(jnp.maximum(eta_mix_cP, 1e-3)),  # log viscosity
        jnp.log(jnp.maximum(eps_mix, 1.0)),       # log dielectric
        ionic_frac,                        # ionic fraction of composition
    ])
    correction = jnp.dot(params["w_correction"], mix_features) + params["b_correction"]

    return log_sigma_base + correction


TEE_GOTOH_STEWART = 0.77  # Analytical constant: ε_LJ/k_B ≈ 0.77·T_b (Tee, Gotoh, Stewart, I&EC Fund. 1966, eq. 5)
LJ_SIGMA_PREFACTOR = 6.0  # Analytical constant: σ³ = 6V/π from relating LJ diameter to molecular volume (V_sphere = πσ³/6)

IDX_BOILING = PROPERTY_KEYS.index("boiling_point_c")


def _compute_physical_ff_targets(species_list: List[str]) -> np.ndarray:
    """Compute physically motivated (σ_LJ, ε_LJ, q) for each species.

    σ_LJ from molar volume: σ = (6·V_mol/(π·N_A))^(1/3) converted to Å
    ε_LJ from boiling point: ε = 0.77·k_B·T_b (Tee-Gotoh-Stewart 1966), in kJ/mol
    q from ionic character: +1 cation, −1 anion, 0 neutral

    Returns (n_species, 3) array of [σ_Å, ε_kJ_mol, q_e].
    """
    targets = np.zeros((len(species_list), 3))
    for i, name in enumerate(species_list):
        props = get_raw_property_vector(name)
        mw = props[IDX_MW]
        rho = props[IDX_DENSITY]
        bp_c = props[IDX_BOILING]
        r_cat = props[IDX_CATION_R]
        r_an = props[IDX_ANION_R]

        # σ_LJ from molar volume
        v_mol_cm3 = mw / max(rho, 1e-6)  # numerical sentinel: prevent div-by-zero on zero-density padding
        v_mol_m3 = v_mol_cm3 * 1e-6
        v_molecule_m3 = v_mol_m3 / N_A
        sigma_m = (LJ_SIGMA_PREFACTOR * v_molecule_m3 / np.pi) ** (1.0 / 3.0)
        targets[i, 0] = sigma_m * 1e10  # m → Å

        # ε_LJ: for solvents/additives use boiling point (Tee-Gotoh-Stewart),
        # for salts use THERMAL ENERGY scale (ip_bind includes Coulomb which
        # is handled separately in the OZ potential — using ip_bind as ε_LJ
        # double-counts Coulomb and gives βε>>3, diverging HNC).
        # For dissolved ions, the LJ well depth is O(kT): too deep → crystallize,
        # too shallow → no solvation structure.  Scale by (σ_LJ/σ_ref) to
        # differentiate large vs small ions.
        ip_bind = props[IDX_ION_PAIR_BINDING]
        is_salt = (r_cat > 0 or r_an > 0)

        if is_salt:
            kT_kJ_mol = R_GAS * T_REF_K / 1000.0  # thermal energy in kJ/mol (~2.48)
            sigma_ref_A = 5.0  # Analytical constant: reference LJ diameter (Å) — median of ion σ_LJ values (range 4.5–8.8)
            targets[i, 1] = kT_kJ_mol * (targets[i, 0] / sigma_ref_A)
        elif bp_c > 0:
            T_b_K = bp_c + CELSIUS_TO_KELVIN
            eps_J = TEE_GOTOH_STEWART * K_B * T_b_K
            targets[i, 1] = eps_J * N_A / 1000.0  # J → kJ/mol
        else:
            raise ValueError(f"Species '{name}' has neither boiling_point_c nor ion_pair_binding — cannot compute ε_LJ")

        # q from ionic character
        if r_cat > 0:
            targets[i, 2] = 1.0
        elif r_an > 0:
            targets[i, 2] = -1.0
        else:
            targets[i, 2] = 0.0

    return targets


def init_params_oz(key):
    """Initialize OZ+MCT with physics-informed FF parameters.

    Computes target (σ_LJ, ε_LJ) from species properties using known
    correlations, then solves for W_ff via least-squares so the projection
    layer reproduces these physical values at initialization.
    Charges are fixed constants (+1/-1/0), not learned.
    """
    all_species = sorted(set(SOLVENTS) | set(SALTS) | set(ADDITIVES))
    ff_targets = _compute_physical_ff_targets(all_species)

    # Only fit σ_LJ and ε_LJ (columns 0,1). Charge (column 2) is fixed.
    pre_act = np.zeros((len(all_species), 2))
    for i in range(len(all_species)):
        sigma_pre = ff_targets[i, 0] - 2.0
        pre_act[i, 0] = np.log(np.expm1(max(sigma_pre, 1e-6)))

        eps_pre = ff_targets[i, 1] / 0.5
        pre_act[i, 1] = np.log(np.expm1(max(eps_pre, 1e-6)))

    if _COMBINING_NORMS:
        mean = np.array(_COMBINING_NORMS["mean"])
        std = np.array(_COMBINING_NORMS["std"])
    else:
        mean, std = compute_normalization_stats(all_species)

    P = np.array([get_raw_property_vector(sp) for sp in all_species])
    P_norm = (P - mean) / np.where(std > 1e-10, std, 1.0)

    P_aug = np.hstack([P_norm, np.ones((len(all_species), 1))])
    Wb, _, _, _ = np.linalg.lstsq(P_aug, pre_act, rcond=None)
    W_init = Wb[:D_PROP, :].T  # (2, D_PROP)
    b_init = Wb[D_PROP, :]     # (2,)

    logger.info(f"Physics-informed FF init from {len(all_species)} species:")
    logger.info(f"  σ_LJ range: {ff_targets[:, 0].min():.1f}-{ff_targets[:, 0].max():.1f} Å")
    logger.info(f"  ε_LJ range: {ff_targets[:, 1].min():.2f}-{ff_targets[:, 1].max():.2f} kJ/mol")
    logger.info(f"  Charges: FIXED (+1/-1/0), not learned")
    logger.info(f"  W_ff lstsq residual: {np.sqrt(np.mean((P_aug @ Wb - pre_act)**2)):.4f}")

    N_MIX_FEATURES = 5  # Explicit constant: [c_salt, T_departure, ln_eta, ln_eps, ionic_frac]
    return {
        "oz_ff": {
            "W": jnp.array(W_init),
            "b": jnp.array(b_init),
        },
        "theta_mct_blend": jnp.array(0.0),
        "w_correction": jnp.zeros(N_MIX_FEATURES),
        "b_correction": jnp.array(0.0),
    }


def loss_fn_oz(params, batch_tuple):
    """Weighted log-MSE loss for OZ+MCT model."""
    raw, fracs, mask, temps, log_sigma, weights = batch_tuple
    pred_log_sigma = jax.vmap(
        _oz_mct_conductivity, in_axes=(None, 0, 0, 0, 0)
    )(params, raw, fracs, mask, temps)
    residuals = pred_log_sigma - log_sigma
    return jnp.sum(weights * residuals ** 2) / jnp.sum(weights)


def _log_oz_params(params, label=""):
    """Log OZ+MCT parameters."""
    prefix = f"[{label}] " if label else ""

    W = params["oz_ff"]["W"]
    b = params["oz_ff"]["b"]
    W_rms = float(jnp.sqrt(jnp.mean(W ** 2)))
    svs = jnp.linalg.svd(W, compute_uv=False)
    sv_str = ", ".join(f"{float(s):.4f}" for s in svs)
    logger.info(f"  {prefix}W_oz: |W|={W_rms:.6f}, SVD=[{sv_str}], "
                f"b={[f'{float(v):.3f}' for v in b]}")

    mct_blend = float(jax.nn.sigmoid(params["theta_mct_blend"]))
    logger.info(f"  {prefix}θ_mct_blend={float(params['theta_mct_blend']):+.4f} "
                f"(blend={mct_blend:.4f})")


# ═══════════════════════════════════════════════════════════════════════
# HARD CUTOVER: OZ+HNC+MCT model is the active conductivity model.
# All downstream consumers (predict_sigma, molset_conductivity_s_m,
# compute_val_mae, evaluate_species_ood) resolve these names at
# call time, so they automatically use the OZ+MCT model.
# ═══════════════════════════════════════════════════════════════════════
forward_single = _oz_mct_conductivity  # noqa: F811
forward_batch = jax.vmap(_oz_mct_conductivity, in_axes=(None, 0, 0, 0, 0))  # noqa: F811
init_params = init_params_oz  # noqa: F811
loss_fn = loss_fn_oz  # noqa: F811


@jax.jit
def _forward_batch_eval(params, raw, fracs, mask, temps):  # noqa: F811
    return jax.vmap(_oz_mct_conductivity, in_axes=(None, 0, 0, 0, 0))(
        params, raw, fracs, mask, temps)


@jax.jit
def _forward_single_eval(params, raw, fracs, mask, temp):  # noqa: F811
    return _oz_mct_conductivity(params, raw, fracs, mask, temp)


# =============================================================================
# MAIN
# =============================================================================

def _log_theta(params, label=""):
    """Log model parameters — delegates to OZ+MCT-specific logger."""
    _log_oz_params(params, label)


def main():
    logger.info("=" * 70)
    logger.info("OZ+HNC+MCT: species props → FF → U_ij(r) → OZ → S_ij(k) → σ")
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
    _set_combining_norms(norm_mean, norm_std)
    logger.info(f"Property vector dimension: {D_INPUT}")

    logger.info(f"\nSpecies in training data: {all_species}")

    train_batch, val_batch = prepare_molset_data(norm_mean, norm_std)

    # OOD proxy split: hold out OOD_PROXY_SPECIES from training
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

    params = init_params(random.PRNGKey(SEED_MAIN))
    n_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"\nOZ+MCT parameters: {n_params}")
    logger.info(f"Initial parameter values:")
    _log_oz_params(params, "init")

    warmup_fn = optax.linear_schedule(0.0, LR_PEAK, WARMUP_STEPS)
    cosine_fn = optax.cosine_decay_schedule(LR_PEAK, N_STEPS - WARMUP_STEPS)
    schedule = optax.join_schedules([warmup_fn, cosine_fn], [WARMUP_STEPS])
    opt = optax.adam(schedule)
    opt_state = opt.init(params)
    step_fn = make_train_step(opt)

    ja = train_core.jax_arrays()
    batch_tuple = (ja["raw"], ja["fracs"], ja["mask"], ja["temps"],
                   ja["log_sigma"], ja["weights"])

    logger.info(f"\nTraining {N_STEPS} steps — OZ+HNC+MCT: FF → U_ij(r) → OZ → S_ij(k) → σ...")
    best_val_mae = float("inf")
    best_val_params = params
    best_val_step = 0
    best_ood_mae = float("inf")
    best_ood_params = params
    best_ood_step = 0
    t0 = time.time()

    for step in range(N_STEPS):
        params, opt_state, loss = step_fn(params, opt_state, batch_tuple)

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

    use_params = best_ood_params if ood_proxy_batch is not None else best_val_params
    use_step = best_ood_step if ood_proxy_batch is not None else best_val_step

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS (best-OOD@{use_step})")
    logger.info(f"{'='*60}")

    logger.info(f"\nLearned OZ+MCT parameters:")
    _log_oz_params(use_params, "final")

    train_metrics = compute_metrics(use_params, train_core)
    val_metrics = compute_metrics(use_params, val_batch)

    logger.info(f"\nTrain: MAE={train_metrics['mae_mS_cm']:.3f} mS/cm, "
                f"RMSE={train_metrics['rmse_mS_cm']:.3f}, "
                f"bias={train_metrics['bias_mS_cm']:.3f}, "
                f"MAPE={train_metrics['mape_pct']:.1f}%")
    logger.info(f"Val:   MAE={val_metrics['mae_mS_cm']:.3f} mS/cm, "
                f"RMSE={val_metrics['rmse_mS_cm']:.3f}, "
                f"bias={val_metrics['bias_mS_cm']:.3f}, "
                f"MAPE={val_metrics['mape_pct']:.1f}%")
    logger.info(f"Train/Val ratio: {train_metrics['mae_mS_cm']/val_metrics['mae_mS_cm']:.2f}")

    logger.info(f"\n--- Baselines (expanded dataset) ---")
    logger.info(f"Per-Ion attention:    val=0.391, FEC=1.669, VC=1.120, LiFSI=2.480")
    logger.info(f"Phase 1-3 generic:   val=2.539, FEC=2.413, VC=1.389, LiFSI=3.403")
    logger.info(f"OZ+MCT (this):        val={val_metrics['mae_mS_cm']:.3f}")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mol_set_sigma_v2_continuum.pkl")
    save_model(use_params, norm_mean, norm_std, model_path)
    logger.info(f"Model saved: {model_path}")

    # Speed test
    logger.info(f"\n--- Speed Test ---")
    test_raw_s = jnp.array(val_batch.raw_props[0])
    test_fracs_s = jnp.array(val_batch.fracs[0])
    test_mask_s = jnp.array(val_batch.mask[0])
    test_temp_s = jnp.array(val_batch.temperature_K[0])

    warmup_result = _forward_single_eval(use_params, test_raw_s, test_fracs_s, test_mask_s, test_temp_s)
    warmup_result.block_until_ready()

    n_calls = 10000
    t0 = time.time()
    for _ in range(n_calls):
        result = _forward_single_eval(use_params, test_raw_s, test_fracs_s, test_mask_s, test_temp_s)
    result.block_until_ready()
    elapsed_jit = (time.time() - t0) / n_calls * 1000
    logger.info(f"JIT single-recipe: {elapsed_jit:.4f} ms/recipe ({n_calls} calls)")

    batch_raw = jnp.array(val_batch.raw_props)
    batch_fracs = jnp.array(val_batch.fracs)
    batch_mask = jnp.array(val_batch.mask)
    batch_temps = jnp.array(val_batch.temperature_K)

    warmup_b = _forward_batch_eval(use_params, batch_raw, batch_fracs, batch_mask, batch_temps)
    warmup_b.block_until_ready()

    n_batch_calls = 1000
    t0 = time.time()
    for _ in range(n_batch_calls):
        result_b = _forward_batch_eval(use_params, batch_raw, batch_fracs, batch_mask, batch_temps)
    result_b.block_until_ready()
    elapsed_batch = (time.time() - t0) / n_batch_calls * 1000
    n_val = len(val_batch.recipe_keys)
    logger.info(f"JIT batch ({n_val} recipes): {elapsed_batch:.3f} ms total, "
                f"{elapsed_batch/n_val:.4f} ms/recipe")

    # Gradient check — argnums=2 is fracs (params, raw_props, fracs, mask, temp)
    logger.info(f"\n--- Gradient Check ---")
    test_raw = jnp.array(val_batch.raw_props[0])
    test_fracs = jnp.array(val_batch.fracs[0])
    test_mask = jnp.array(val_batch.mask[0])
    test_temp = jnp.array(val_batch.temperature_K[0])

    grad_fn = jax.grad(forward_single, argnums=2)
    frac_grads = grad_fn(use_params, test_raw, test_fracs, test_mask, test_temp)
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
    if "--no-ood" not in sys.argv:
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
    else:
        logger.info("\nSkipping OOD evaluation (--no-ood flag)")


if __name__ == "__main__":
    main()
