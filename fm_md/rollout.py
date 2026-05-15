"""Phase 5: roll the COM propagator forward to generate an MD-like trajectory.

One propagator step advances the molecular-COM box by `dt_fs`:
  1. Sample dr_0 ~ N(0, I) per molecule (the CFM Gaussian prior).
  2. Integrate the flow-matching ODE d(dr)/ds = u_theta(dr_s, s; x_t) from s=0 to
     s=1 with N_EULER_STEPS forward-Euler steps. The conditioning configuration
     x_t is fixed across the s-integration; only dr evolves.
  3. The endpoint dr is one sample of the per-molecule COM displacement; advance
     the box x_{t+1} = x_t + dr.

Chaining n_steps of this produces a molecular-COM trajectory. The COM positions
accumulate displacements without re-wrapping, so the rolled-out trajectory is
already unwrapped — exactly the continuous-path form the cepstral estimator
needs. The neighbor list is rebuilt once per propagator step.

Entry: `python -m conductivity.fm_md.rollout` loads the trained COM propagator,
rolls out from FSI MD frame 0, and reports the cepstral sigma against the
MD reference of 12.94 mS/cm.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import control_framework.jax_m4_tuning  # noqa: F401  -- MUST precede any jax import

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from conductivity.fm_md.atomistic_io import (
    MolecularConfiguration,
    MolecularTrajectory,
    load_com_cache,
)
from conductivity.fm_md.propagator import build_neighbor_arrays, propagator_core


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


N_EULER_STEPS = 20             # forward-Euler steps for the flow-matching ODE s: 0 -> 1 (plan §2.2)


def _fm_ode_endpoint(
    params: dict,
    dr0: jnp.ndarray,
    dr_prev: jnp.ndarray,
    molecule_species: jnp.ndarray,
    formal_charges: jnp.ndarray,
    species_graphs,
    nl,
) -> jnp.ndarray:
    """Integrate the flow-matching ODE from s=0 (dr0) to s=1. Jittable.

    `dr_prev` is the previous-step COM displacement (the momentum state) and is
    fixed across the s-integration. Forward Euler: dr <- dr + (1/N) * u_theta.
    """
    ds = 1.0 / N_EULER_STEPS

    def body(i, dr):
        s = i.astype(jnp.float32) * ds
        u = propagator_core(
            params, dr, s, dr_prev, molecule_species, formal_charges, species_graphs, nl,
        )
        return dr + ds * u

    return jax.lax.fori_loop(0, N_EULER_STEPS, body, dr0)


def rollout_trajectory(
    params: dict,
    x_init: MolecularConfiguration,
    n_steps: int,
    dt_fs: float,
    rng: jax.Array,
) -> MolecularTrajectory:
    """Chain n_steps COM-propagator advances from x_init.

    Inputs:
      params  : propagator parameter tree
      x_init  : MolecularConfiguration — starting molecular-COM box
      n_steps : number of propagator steps to take
      dt_fs   : physical time per propagator step
      rng     : JAX PRNG key
    Returns a MolecularTrajectory with n_steps + 1 frames; temperature_K is left
    as NaN here and set by the caller, which knows the simulation temperature.
    """
    box = np.asarray(x_init.box, dtype=np.float64)
    molecule_species = jnp.asarray(x_init.molecule_species)
    formal_charges = jnp.asarray(x_init.formal_charges, dtype=jnp.float32)
    species_graphs = jax.device_put(x_init.species_graphs)
    n_molecules = np.asarray(x_init.com_positions).shape[0]

    integrate = jax.jit(_fm_ode_endpoint)

    com_seq: list[np.ndarray] = [np.asarray(x_init.com_positions, dtype=np.float64)]
    cur_com = np.asarray(x_init.com_positions, dtype=np.float64)
    # Momentum state — carried across propagator steps (second-order rollout).
    dr_prev = jnp.asarray(x_init.prev_displacement, dtype=jnp.float32)

    t_start = time.time()
    LOG_EVERY_STEPS = 100   # Explicit constant: rollout progress log interval
    for step in range(n_steps):
        nl = jax.device_put(build_neighbor_arrays(cur_com, box))
        rng, noise_key = random.split(rng)
        dr0 = random.normal(noise_key, (n_molecules, 3))           # unit CFM prior
        dr = integrate(params, dr0, dr_prev, molecule_species, formal_charges,
                        species_graphs, nl)
        dr_prev = dr                                               # carry momentum forward
        cur_com = cur_com + np.asarray(dr, dtype=np.float64)       # accumulate -> unwrapped
        com_seq.append(cur_com.copy())
        if (step + 1) % LOG_EVERY_STEPS == 0:
            logger.info(
                "rollout step %d/%d  elapsed=%.0fs", step + 1, n_steps, time.time() - t_start,
            )

    com_positions = np.stack(com_seq, axis=0)
    return MolecularTrajectory(
        com_positions=com_positions,
        molecule_species=np.asarray(x_init.molecule_species),
        formal_charges=np.asarray(x_init.formal_charges),
        box=box,
        dt_fs=dt_fs,
        n_frames=com_positions.shape[0],
        n_molecules=n_molecules,
        temperature_K=float("nan"),
    )


# Reference MD conductivity for the FSI composition. The cepstral value is the
# Phase 6 result on the raw trajectory (operator_fm_phase_0_1_6_passed memory).
# The direct-ρ-sum value is what the short-segment estimator gives on the same
# MD trajectory; comparing the propagator's short-segment σ against THIS number
# cancels the estimator's common-mode finite-sample bias (plan §1q).
SIGMA_MD_REFERENCE_MSCM = 12.94
SIGMA_MD_DIRECT_REFERENCE_MSCM = 24.40
RHO1_MD = -0.253                       # MD charge-current autocorrelation at τ=1 (caging)
RHO_SUM_MAX_TAU = 3                    # highest lag in the direct ρ-sum; MD tail past τ=3 is negligible
DIVERGENCE_DISPLACEMENT_ANG = 20.0     # per-step |Δr| above which a burst is judged diverged
                                       # (well above any physical step, well below blow-up)


def rollout_bursts(
    params: dict,
    cached_com,
    n_bursts: int,
    burst_steps: int,
    rng: jax.Array,
) -> list[np.ndarray]:
    """Roll out many short, independent COM trajectories — the rollout-stable
    sigma path (plan §1q). Each burst starts from a different MD frame with the
    real previous-step displacement as its momentum state, and is short enough
    to stay well below the long-rollout divergence onset.

    Returns a list of (burst_steps + 1, n_molecules, 3) unwrapped COM arrays.
    """
    box = np.asarray(cached_com.box, dtype=np.float64)
    molecule_species = jnp.asarray(cached_com.molecule_species)
    formal_charges = jnp.asarray(cached_com.formal_charges, dtype=jnp.float32)
    species_graphs = jax.device_put(cached_com.species_graphs)
    n_molecules = cached_com.n_molecules
    integrate = jax.jit(_fm_ode_endpoint)   # compiled once, reused across all bursts

    np_rng = np.random.default_rng(0)
    bursts: list[np.ndarray] = []
    t_start = time.time()
    for b in range(n_bursts):
        f = int(np_rng.integers(1, cached_com.n_frames - 1))
        com_fm1 = np.asarray(cached_com.com_positions[f - 1], dtype=np.float64)
        cur_com = np.asarray(cached_com.com_positions[f], dtype=np.float64)
        dr_prev_np = cur_com - com_fm1
        dr_prev_np = dr_prev_np - box * np.round(dr_prev_np / box)   # minimum image
        dr_prev = jnp.asarray(dr_prev_np, dtype=jnp.float32)

        com_seq = [cur_com.copy()]
        for _ in range(burst_steps):
            nl = jax.device_put(build_neighbor_arrays(cur_com, box))
            rng, noise_key = random.split(rng)
            dr0 = random.normal(noise_key, (n_molecules, 3))
            dr = integrate(params, dr0, dr_prev, molecule_species, formal_charges,
                            species_graphs, nl)
            dr_prev = dr
            cur_com = cur_com + np.asarray(dr, dtype=np.float64)
            com_seq.append(cur_com.copy())
        bursts.append(np.stack(com_seq, axis=0))
        logger.info("burst %d/%d done (start frame %d)  elapsed=%.0fs",
                    b + 1, n_bursts, f, time.time() - t_start)
    return bursts


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 rollout + sigma validation.")
    parser.add_argument("--model", default="conductivity/fm_data/fm_md_model/propagator_final.pkl")
    parser.add_argument("--cache-dir", default="conductivity/fm_data/cache/traj_FSI_20k_com")
    parser.add_argument("--mode", choices=["bursts", "long"], default="bursts",
                        help="bursts: many short stable rollouts; long: one long rollout.")
    parser.add_argument("--n-bursts", type=int, default=256,
                        help="Number of short bursts (mode=bursts).")
    parser.add_argument("--burst-steps", type=int, default=RHO_SUM_MAX_TAU + 1,
                        help="Steps per burst (mode=bursts); a 4-step burst cannot diverge to overflow.")
    parser.add_argument("--n-steps", type=int, default=2000,
                        help="Steps for one long rollout (mode=long).")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    dt_fs = ckpt["dt_fs"] * ckpt["lag_frames"]
    logger.info("Loaded model %s (trained %d steps, dt per propagator step = %.1f fs)",
                args.model, ckpt["step"], dt_fs)

    cached = load_com_cache(Path(args.cache_dir))
    box = np.asarray(cached.box, dtype=np.float64)
    box_volume_ang3 = float(np.prod(box))
    from conductivity.fm_md.sigma_from_trajectory import (
        phase6_gate,
        sigma_direct_rho_sum,
        sigma_from_trajectory,
    )

    if args.mode == "bursts":
        logger.info("Rolling out %d bursts × %d steps from random FSI MD frames...",
                    args.n_bursts, args.burst_steps)
        bursts = rollout_bursts(params, cached, args.n_bursts, args.burst_steps, random.PRNGKey(0))
        rho_est = sigma_direct_rho_sum(
            com_segments=bursts,
            dt_fs=dt_fs,
            charges=np.asarray(cached.formal_charges, dtype=np.float64),
            box=box,
            temperature_K=cached.temperature_K,
            volume_ang3=box_volume_ang3,
            max_tau=RHO_SUM_MAX_TAU,
            divergence_displacement_ang=DIVERGENCE_DISPLACEMENT_ANG,
        )
        logger.info("=== Phase 5 rollout validation (direct ρ-sum, short bursts) ===")
        logger.info("propagator σ_direct:   %.3f mS/cm", rho_est.sigma_mS_cm)
        logger.info("MD σ_direct reference: %.3f mS/cm  (same short-segment estimator)",
                    SIGMA_MD_DIRECT_REFERENCE_MSCM)
        logger.info("ratio propagator/MD:   %.3f", rho_est.sigma_mS_cm / SIGMA_MD_DIRECT_REFERENCE_MSCM)
        logger.info("propagator ρ(1):       %+.3f   (MD ρ(1) = %+.3f — caging anti-correlation)",
                    float(rho_est.rho[1]), RHO1_MD)
        logger.info("propagator [1+2Σρ]:    %.3f", rho_est.correlation_factor)
        logger.info("diverged bursts:       %.1f%% of %d",
                    rho_est.diverged_fraction * 100.0, args.n_bursts)
        return 0

    com0 = np.asarray(cached.com_positions[0], dtype=np.float64)
    com1 = np.asarray(cached.com_positions[1], dtype=np.float64)
    dr_init = com1 - com0
    dr_init = dr_init - box * np.round(dr_init / box)
    x_init = MolecularConfiguration(
        com_positions=com1,
        prev_displacement=dr_init,
        molecule_species=cached.molecule_species,
        formal_charges=cached.formal_charges,
        box=cached.box,
        n_molecules=cached.n_molecules,
    )
    logger.info("Rolling out %d propagator steps from FSI MD frame 1...", args.n_steps)
    mol_traj = rollout_trajectory(params, x_init, args.n_steps, dt_fs, random.PRNGKey(0))
    mol_traj = mol_traj._replace(temperature_K=cached.temperature_K)
    estimate = sigma_from_trajectory(
        trajectory=mol_traj,
        charges=mol_traj.formal_charges.astype(np.float64),
        temperature_K=mol_traj.temperature_K,
        volume_ang3=box_volume_ang3,
    )
    gate = phase6_gate(estimate)
    logger.info("=== Phase 5 rollout validation (mode=long) ===")
    logger.info("rolled-out sigma:   %.3f mS/cm", gate.sigma_mS_cm)
    logger.info("MD reference sigma: %.3f mS/cm", SIGMA_MD_REFERENCE_MSCM)
    logger.info("ratio rolled/MD:    %.3f", gate.sigma_mS_cm / SIGMA_MD_REFERENCE_MSCM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
