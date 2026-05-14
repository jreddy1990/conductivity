"""Diagnostic: verify OZ+HNC+MCT convergence after Ng renormalization + salt ε fix."""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "control_framework")
import jax_m4_tuning  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from conductivity.mol_set_sigma_v2_continuum import (
    _compute_physical_ff_targets, _build_pair_potential, _build_coulomb_potential,
    _oz_hnc_solve, _structure_factor_from_h, _chandra_bagchi_sigma,
    _R_GRID, _K_GRID, _DK, DR_ANGSTROM, N_OZ_SPECIES, N_GRID,
    get_raw_property_vector, PROPERTY_KEYS,
    SOLVENTS, SALTS, ADDITIVES, EXP_OVERFLOW_GUARD,
)
from constants import T_REF_K, K_B, R as R_GAS, E_CHARGE, EPS_0, N_A

all_species = sorted(set(SOLVENTS) | set(SALTS) | set(ADDITIVES))
targets = _compute_physical_ff_targets(all_species)

print("=== Physical FF Targets (after salt ε fix) ===")
for i, sp in enumerate(all_species):
    beta_eps = targets[i, 1] / (R_GAS * T_REF_K / 1000.0)
    flag = " *** TOO HIGH ***" if beta_eps > 3.0 else ""
    print(f"  {sp:12s}: σ={targets[i,0]:6.2f} Å, ε={targets[i,1]:5.2f} kJ/mol (βε={beta_eps:5.2f}), q={targets[i,2]:+.0f}{flag}")

print(f"\n=== Testing OZ+HNC with Ng renormalization on 3-species system ===")
# 3 effective species from physical FF targets:
# cation: Li+ (use LiPF6-like), anion: PF6- (use LiPF6-like), solvent: EC:DMC avg
li_idx = all_species.index("LiPF6")
ec_idx = all_species.index("EC")
dmc_idx = all_species.index("DMC")

ff_cat = jnp.array([targets[li_idx, 0] * 0.3, targets[li_idx, 1] * 0.3, 1.0])
ff_an = jnp.array([targets[li_idx, 0] * 0.7, targets[li_idx, 1] * 0.7, -1.0])
ff_solv = jnp.array([(targets[ec_idx, 0] + targets[dmc_idx, 0]) / 2,
                      (targets[ec_idx, 1] + targets[dmc_idx, 1]) / 2, 0.0])
ff_3 = jnp.stack([ff_cat, ff_an, ff_solv])

print(f"ff_cat:  σ={float(ff_cat[0]):.2f}Å, ε={float(ff_cat[1]):.2f} kJ/mol, q={float(ff_cat[2]):+.1f}")
print(f"ff_an:   σ={float(ff_an[0]):.2f}Å, ε={float(ff_an[1]):.2f} kJ/mol, q={float(ff_an[2]):+.1f}")
print(f"ff_solv: σ={float(ff_solv[0]):.2f}Å, ε={float(ff_solv[1]):.2f} kJ/mol, q={float(ff_solv[2]):+.1f}")

eps_r = 40.0  # EC:DMC mixture dielectric
T_K = T_REF_K
z_3 = jnp.array([1.0, -1.0, 0.0])

# Full pairwise potentials
def build_row(i_ff):
    return jax.vmap(lambda j_ff: _build_pair_potential(i_ff, j_ff, _R_GRID, eps_r, T_K))(ff_3)
beta_u = jax.vmap(build_row)(ff_3)

# Coulomb-only part
def coul_row(i_idx):
    return jax.vmap(lambda j_idx: _build_coulomb_potential(
        z_3[i_idx], z_3[j_idx], _R_GRID, eps_r, T_K))(jnp.arange(N_OZ_SPECIES))
beta_u_coul = jax.vmap(coul_row)(jnp.arange(N_OZ_SPECIES))

# Analytical Coulomb FT
lambda_B_A = E_CHARGE**2 / (4.0 * jnp.pi * EPS_0 * eps_r * K_B * T_K) * 1e10
print(f"\nBjerrum length: {float(lambda_B_A):.2f} Å")
z_ij = z_3[:, None] * z_3[None, :]
c_hat_coul = -4.0 * jnp.pi * z_ij[:, :, None] * lambda_B_A / _K_GRID[None, None, :]**2

# Number densities: 1M LiPF6 in EC:DMC
L_TO_ANG3 = 1e27
rho_3 = jnp.array([
    0.6 * N_A / L_TO_ANG3,
    0.6 * N_A / L_TO_ANG3,
    12.0 * N_A / L_TO_ANG3,
])
print(f"Number densities (Å⁻³): cat={float(rho_3[0]):.6f}, an={float(rho_3[1]):.6f}, solv={float(rho_3[2]):.6f}")

# Solve with Ng renormalization
print(f"\nSolving OZ+HNC with Ng renormalization...")
h_3, c_3 = _oz_hnc_solve(beta_u, rho_3, _R_GRID, _K_GRID, DR_ANGSTROM, N_OZ_SPECIES,
                           beta_u_coul, c_hat_coul)

print(f"h_3 range: [{float(jnp.min(h_3)):.4f}, {float(jnp.max(h_3)):.4f}]")
print(f"c_3 range: [{float(jnp.min(c_3)):.4f}, {float(jnp.max(c_3)):.4f}]")
print(f"NaN: h={bool(jnp.any(jnp.isnan(h_3)))}, c={bool(jnp.any(jnp.isnan(c_3)))}")
print(f"Inf: h={bool(jnp.any(jnp.isinf(h_3)))}, c={bool(jnp.any(jnp.isinf(c_3)))}")

# g(r) = 1 + h(r)
names = ["cat", "an", "solv"]
for i in range(3):
    for j in range(i, 3):
        g_max = float(1.0 + jnp.max(h_3[i, j]))
        g_min = float(1.0 + jnp.min(h_3[i, j]))
        print(f"  g_{names[i]}-{names[j]}: min={g_min:.3f}, max={g_max:.3f}")

# Structure factor
S_3, h_hat_3 = _structure_factor_from_h(h_3, rho_3, _R_GRID, _K_GRID, DR_ANGSTROM, N_OZ_SPECIES)
print(f"\nS_3 range: [{float(jnp.min(S_3)):.4f}, {float(jnp.max(S_3)):.4f}]")
print(f"S_3 NaN: {bool(jnp.any(jnp.isnan(S_3)))}")

# S_ZZ(k→0)
rho_Z = jnp.sum(z_3**2 * rho_3)
rho_sqrt = jnp.sqrt(jnp.maximum(rho_3, 1e-30))
rho_ij = rho_sqrt[:, None] * rho_sqrt[None, :]
S_ZZ = jnp.sum(z_ij[:, :, None] * rho_ij[:, :, None] * S_3, axis=(0, 1)) / rho_Z
print(f"S_ZZ(k→0) = {float(S_ZZ[0]):.4f}")
print(f"MCT factor = 1/S_ZZ(0) = {1.0/float(S_ZZ[0]):.4f}")

# If S_ZZ(0) is in range 0.1-10, the MCT correction is physically meaningful
if 0.01 < float(S_ZZ[0]) < 100:
    print("✓ S_ZZ(k→0) in physically meaningful range")
else:
    print("✗ S_ZZ(k→0) out of range — OZ may not have converged")
