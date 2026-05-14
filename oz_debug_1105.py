"""Debug: investigate recipe 1105 that produces NaN."""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "control_framework")
import jax_m4_tuning  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from conductivity.mol_set_sigma_v2_continuum import (
    init_params_oz, _oz_mct_conductivity, _oz_hnc_solve,
    _build_pair_potential, _build_coulomb_potential,
    _structure_factor_from_h, _chandra_bagchi_sigma,
    prepare_molset_data, compute_normalization_stats, _set_combining_norms,
    _COMBINING_NORMS, SOLVENTS, SALTS, ADDITIVES, N_MAX_SPECIES,
    IDX_MW, IDX_LAMBDA0, IDX_CATION_R, IDX_ANION_R, IDX_VISCOSITY,
    IDX_DENSITY, PROPERTY_KEYS,
    _R_GRID, _K_GRID, _DK, DR_ANGSTROM, N_OZ_SPECIES, N_GRID,
    _ionic_weight, _kirkwood_mixture_epsilon, _mole_frac_to_molarity,
    EXP_OVERFLOW_GUARD, ETA_REF_WATER_25C_CP, IDX_JONES_DOLE,
    _screened_fuoss_alpha,
)
from constants import E_CHARGE, EPS_0, K_B, N_A
from jax import random

all_species = sorted(set(SOLVENTS) | set(SALTS) | set(ADDITIVES))
mean, std = compute_normalization_stats(all_species)
_set_combining_norms(mean, std)
train_data, val_data = prepare_molset_data(mean, std)
key = random.PRNGKey(42)
params = init_params_oz(key)

idx = 1105
raw = jnp.array(train_data.raw_props[idx])
fracs = jnp.array(train_data.fracs[idx])
mask = jnp.array(train_data.mask[idx])
T_K = jnp.array(train_data.temperature_K[idx])
target = train_data.log_sigma[idx]

print(f"=== Recipe {idx} ===")
print(f"T={float(T_K):.1f} K, target σ={float(np.exp(target)):.3f} mS/cm")
n_active = int(mask.sum())
print(f"Active species: {n_active}")
for j in range(n_active):
    mw = float(raw[j, IDX_MW])
    lam0 = float(raw[j, IDX_LAMBDA0])
    cat_r = float(raw[j, IDX_CATION_R])
    an_r = float(raw[j, IDX_ANION_R])
    frac = float(fracs[j])
    role = "salt" if (cat_r > 0 or an_r > 0) else "solvent"
    print(f"  Slot {j}: MW={mw:.1f}, Λ₀={lam0:.1f}, cat_r={cat_r:.2f}, an_r={an_r:.2f}, frac={frac:.4f} ({role})")

# Check: is this a pure-solvent recipe (no salt)?
has_cat = jnp.where(raw[:, IDX_CATION_R] > 0, 1.0, 0.0) * mask
has_an = jnp.where(raw[:, IDX_ANION_R] > 0, 1.0, 0.0) * mask
print(f"\nhas_cat sum={float(jnp.sum(has_cat)):.1f}, has_an sum={float(jnp.sum(has_an)):.1f}")
if float(jnp.sum(has_cat)) == 0:
    print("*** THIS IS A PURE SOLVENT RECIPE — NO SALT! ***")

# Step through forward pass manually
lam0_vec = raw[:, IDX_LAMBDA0]
iw = _ionic_weight(lam0_vec)
w = fracs * mask
is_salt = has_cat * has_an
is_neutral = (1.0 - has_cat) * (1.0 - has_an) * mask

w_cat = w * has_cat
w_an = w * has_an
w_neut = w * is_neutral

print(f"\nw_cat_sum={float(jnp.sum(w_cat)):.6f}")
print(f"w_an_sum={float(jnp.sum(w_an)):.6f}")
print(f"w_neut_sum={float(jnp.sum(w_neut)):.6f}")

# FF projection
p_norm = (raw - _COMBINING_NORMS["mean"]) / _COMBINING_NORMS["std"]
p_norm = p_norm * mask[:, None]
ff_raw = p_norm @ params["oz_ff"]["W"].T + params["oz_ff"]["b"][None, :]

sigma_lj = jax.nn.softplus(ff_raw[:, 0]) + 2.0
eps_lj = jax.nn.softplus(ff_raw[:, 1]) * 0.5
q_raw = jnp.tanh(ff_raw[:, 2])
q_mag = jnp.abs(q_raw) + 0.5

w_cat_sum = jnp.maximum(jnp.sum(w_cat), 1e-12)
w_an_sum = jnp.maximum(jnp.sum(w_an), 1e-12)

ff_cat_q = float(jnp.sum(w_cat * q_mag) / w_cat_sum)
ff_an_q = float(-jnp.sum(w_an * q_mag) / w_an_sum)
print(f"\nEffective charges: cat_q={ff_cat_q:.4f}, an_q={ff_an_q:.4f}")

# Test the actual forward
result = _oz_mct_conductivity(params, raw, fracs, mask, T_K)
print(f"\nForward result: {float(result):.4f}, NaN={bool(jnp.isnan(result))}")
