"""Debug: trace NaN through _oz_mct_conductivity with real init params and data."""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "control_framework")
import jax_m4_tuning  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from conductivity.mol_set_sigma_v2_continuum import (
    init_params_oz, _oz_mct_conductivity,
    get_raw_property_vector, PROPERTY_KEYS, N_MAX_SPECIES,
    compute_normalization_stats, _COMBINING_NORMS,
    SOLVENTS, SALTS, ADDITIVES,
)
from jax import random

# Load training data to populate normalization stats
from conductivity.mol_set_sigma_v2_continuum import (
    prepare_molset_data, compute_normalization_stats, _set_combining_norms,
)
all_species = sorted(set(SOLVENTS) | set(SALTS) | set(ADDITIVES))
mean, std = compute_normalization_stats(all_species)
_set_combining_norms(mean, std)
train_data, val_data = prepare_molset_data(mean, std)

# Init params
key = random.PRNGKey(42)
params = init_params_oz(key)

# Take first 5 training recipes from MolSetBatch
raw_batch = jnp.array(train_data.raw_props[:5])
fracs_batch = jnp.array(train_data.fracs[:5])
mask_batch = jnp.array(train_data.mask[:5])
temps_batch = jnp.array(train_data.temperature_K[:5])
log_sigma_batch = jnp.array(train_data.log_sigma[:5])

print("=== Testing 5 training recipes through _oz_mct_conductivity (no JIT) ===")
for i in range(5):
    raw_i = raw_batch[i]
    fracs_i = fracs_batch[i]
    mask_i = mask_batch[i]
    T_i = temps_batch[i]
    target_i = log_sigma_batch[i]

    try:
        result = _oz_mct_conductivity(params, raw_i, fracs_i, mask_i, T_i)
        is_nan = bool(jnp.isnan(result))
        print(f"Recipe {i}: log(σ)={float(result):.4f}, σ={float(jnp.exp(result)):.3f} mS/cm, "
              f"target={float(jnp.exp(target_i)):.3f} mS/cm, NaN={is_nan}")
    except Exception as e:
        print(f"Recipe {i}: EXCEPTION: {e}")

# If all NaN, dig deeper into recipe 0
print("\n=== Detailed trace for recipe 0 ===")
raw_0 = raw_batch[0]
fracs_0 = fracs_batch[0]
mask_0 = mask_batch[0]
T_0 = temps_batch[0]

# Species in this recipe
from conductivity.mol_set_sigma_v2_continuum import IDX_LAMBDA0, IDX_CATION_R, IDX_ANION_R, IDX_MW
for j in range(int(mask_0.sum())):
    mw = float(raw_0[j, IDX_MW])
    lam0 = float(raw_0[j, IDX_LAMBDA0])
    cat_r = float(raw_0[j, IDX_CATION_R])
    an_r = float(raw_0[j, IDX_ANION_R])
    frac = float(fracs_0[j])
    role = "salt" if (cat_r > 0 or an_r > 0) else "solvent"
    print(f"  Slot {j}: MW={mw:.1f}, Λ₀={lam0:.1f}, cat_r={cat_r:.2f}, an_r={an_r:.2f}, frac={frac:.4f} ({role})")

# Step through the key intermediates
from conductivity.mol_set_sigma_v2_continuum import _ionic_weight, _kirkwood_mixture_epsilon, _mole_frac_to_molarity
lam0_vec = raw_0[:, IDX_LAMBDA0]
iw = _ionic_weight(lam0_vec)
print(f"\nionic_weight max: {float(jnp.max(iw * mask_0)):.4f}")

# FF projection
p_norm = (raw_0 - _COMBINING_NORMS["mean"]) / _COMBINING_NORMS["std"]
p_norm = p_norm * mask_0[:, None]
ff_raw = p_norm @ params["oz_ff"]["W"].T + params["oz_ff"]["b"][None, :]

sigma_lj = jax.nn.softplus(ff_raw[:, 0]) + 2.0
eps_lj = jax.nn.softplus(ff_raw[:, 1]) * 0.5
q_raw = jnp.tanh(ff_raw[:, 2])

is_cation = jnp.where(raw_0[:, IDX_CATION_R] > 0, 1.0, 0.0) * mask_0
is_anion = jnp.where(raw_0[:, IDX_ANION_R] > 0, 1.0, 0.0) * mask_0
is_neutral = (1.0 - is_cation) * (1.0 - is_anion) * mask_0

print(f"\nFF for active species:")
for j in range(int(mask_0.sum())):
    print(f"  Slot {j}: σ={float(sigma_lj[j]):.2f}Å, ε={float(eps_lj[j]):.2f} kJ/mol, "
          f"q_raw={float(q_raw[j]):.3f}, is_cat={float(is_cation[j]):.0f}, is_an={float(is_anion[j]):.0f}")

# Check: does LiPF6 get classified as BOTH cation and anion?
w = fracs_0 * mask_0
w_cat = w * is_cation
w_an = w * is_anion
w_neut = w * is_neutral
print(f"\nw_cat sum={float(jnp.sum(w_cat)):.4f}, w_an sum={float(jnp.sum(w_an)):.4f}, w_neut sum={float(jnp.sum(w_neut)):.4f}")

# If a salt has BOTH cation_r > 0 and anion_r > 0, it's classified as cation AND anion!
# That means is_neutral = 0 for salts, but the salt contributes to BOTH w_cat and w_an
# This is correct if salt species represent the FULL salt (cation+anion pair)
# But the 3-species reduction assumes distinct cation/anion species, not pairs

# Check q_biased
q_biased = (q_raw * is_neutral
            + (jnp.abs(q_raw) + 0.5) * is_cation
            - (jnp.abs(q_raw) + 0.5) * is_anion)
print(f"\nq_biased for active species:")
for j in range(int(mask_0.sum())):
    print(f"  Slot {j}: q_biased={float(q_biased[j]):.4f}")
