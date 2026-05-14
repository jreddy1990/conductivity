"""Debug: find which recipes produce NaN in vmap'd forward pass."""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "control_framework")
import jax_m4_tuning  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from conductivity.mol_set_sigma_v2_continuum import (
    init_params_oz, _oz_mct_conductivity, loss_fn_oz,
    prepare_molset_data, compute_normalization_stats, _set_combining_norms,
    SOLVENTS, SALTS, ADDITIVES, N_MAX_SPECIES,
)
from jax import random

all_species = sorted(set(SOLVENTS) | set(SALTS) | set(ADDITIVES))
mean, std = compute_normalization_stats(all_species)
_set_combining_norms(mean, std)
train_data, val_data = prepare_molset_data(mean, std)

key = random.PRNGKey(42)
params = init_params_oz(key)

# Test: single forward on first recipe (should be finite)
raw0 = jnp.array(train_data.raw_props[0])
frac0 = jnp.array(train_data.fracs[0])
mask0 = jnp.array(train_data.mask[0])
T0 = jnp.array(train_data.temperature_K[0])
result0 = _oz_mct_conductivity(params, raw0, frac0, mask0, T0)
print(f"Single forward (eager): {float(result0):.4f}, NaN={bool(jnp.isnan(result0))}")

# Test: JIT'd single forward
fwd_jit = jax.jit(_oz_mct_conductivity)
print("JIT compiling single forward...")
result0_jit = fwd_jit(params, raw0, frac0, mask0, T0)
print(f"Single forward (JIT): {float(result0_jit):.4f}, NaN={bool(jnp.isnan(result0_jit))}")

# Test: vmap'd forward on first 10 recipes
print("\nTesting vmap on 10 recipes...")
forward_batch = jax.vmap(_oz_mct_conductivity, in_axes=(None, 0, 0, 0, 0))
raw10 = jnp.array(train_data.raw_props[:10])
frac10 = jnp.array(train_data.fracs[:10])
mask10 = jnp.array(train_data.mask[:10])
T10 = jnp.array(train_data.temperature_K[:10])
results10 = forward_batch(params, raw10, frac10, mask10, T10)
nan_mask = jnp.isnan(results10)
print(f"vmap(10) NaN count: {int(jnp.sum(nan_mask))}/{len(results10)}")
for i in range(10):
    print(f"  Recipe {i}: {float(results10[i]):.4f} {'NaN!' if bool(nan_mask[i]) else ''}")

# Test: loss function with JIT
print("\nTesting loss_fn with JIT on 10 recipes...")
batch = (raw10, frac10, mask10, T10,
         jnp.array(train_data.log_sigma[:10]),
         jnp.array(train_data.weights[:10]))
loss_jit = jax.jit(loss_fn_oz)
loss_val = loss_jit(params, batch)
print(f"Loss (10 recipes): {float(loss_val):.4f}, NaN={bool(jnp.isnan(loss_val))}")

# Test: gradient with JIT on 10 recipes
print("\nTesting gradient on 10 recipes...")
grad_fn = jax.jit(jax.grad(loss_fn_oz))
grads = grad_fn(params, batch)
grad_w = grads["oz_ff"]["W"]
grad_b = grads["oz_ff"]["b"]
grad_blend = grads["theta_mct_blend"]
print(f"grad W: |grad|={float(jnp.sqrt(jnp.sum(grad_w**2))):.4f}, NaN={bool(jnp.any(jnp.isnan(grad_w)))}")
print(f"grad b: {[f'{float(g):.4f}' for g in grad_b]}, NaN={bool(jnp.any(jnp.isnan(grad_b)))}")
print(f"grad blend: {float(grad_blend):.4f}, NaN={bool(jnp.isnan(grad_blend))}")

# If gradients look OK, test on full batch
if not bool(jnp.any(jnp.isnan(grad_w))):
    print("\nGradients OK on 10 — testing full training batch...")
    raw_full = jnp.array(train_data.raw_props)
    frac_full = jnp.array(train_data.fracs)
    mask_full = jnp.array(train_data.mask)
    T_full = jnp.array(train_data.temperature_K)
    results_full = forward_batch(params, raw_full, frac_full, mask_full, T_full)
    n_nan = int(jnp.sum(jnp.isnan(results_full)))
    print(f"Full batch NaN: {n_nan}/{len(results_full)}")
    if n_nan > 0:
        nan_idxs = np.where(np.array(jnp.isnan(results_full)))[0]
        print(f"NaN at indices: {nan_idxs[:20]}")
else:
    print("\nGradients have NaN — skipping full batch test")
