"""Phase 4: flow-matching training loop for the MD-emulator propagator.

Trains `propagator_velocity` on time-lagged atomistic pairs (x_t, x_{t+lag})
from the cached FSI trajectory. The objective is the standard conditional
flow-matching regression on per-atom displacements:

  dr_target = minimum_image(positions[t+lag] - positions[t], box)   (MD truth)
  dr_0      ~ N(0, sigma_prior^2 I)                                 (Gaussian prior)
  s         ~ Uniform(0, 1)                                         (flow time)
  dr_s      = (1 - s) * dr_0 + s * dr_target                        (linear path)
  loss      = mean_atoms || u_theta(dr_s, s; x_t) - (dr_target - dr_0) ||^2

The FM target velocity of the linear interpolant path is the constant
(dr_target - dr_0). There is no sigma supervision anywhere — the propagator
learns dynamics, and conductivity is read off later from a rolled-out trajectory.

`sigma_prior` is set to the measured RMS per-atom displacement at the chosen
lag, so the Gaussian prior and the data distribution share a scale.

Entry: `python -m conductivity.fm_md.train`
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
import optax
from jax import random, value_and_grad

from conductivity.fm_md.atomistic_io import load_com_cache
from conductivity.fm_md.propagator import (
    NeighborArrays,
    build_neighbor_arrays,
    count_parameters,
    init_propagator_params,
    propagator_core,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Training constants (explicit, auditable)
# =============================================================================

LAG_FRAMES = 1                 # propagator step = 1 trajectory frame = 1.5 ps (native cadence)
N_TRAIN_STEPS = 3000           # first-pass training budget
LEARNING_RATE_PEAK = 3.0e-4    # AdamW peak LR
LEARNING_RATE_FLOOR = 1.0e-5   # cosine-decay floor
WARMUP_STEPS = 100             # linear LR warmup
WEIGHT_DECAY = 1.0e-5          # AdamW decoupled weight decay
GRAD_CLIP_NORM = 1.0           # global-norm gradient clip
CHECKPOINT_EVERY = 500         # steps between checkpoints
LOG_EVERY = 50                 # steps between training-loss log lines
COLLAPSE_CHECK_EVERY = 100     # steps between output-std collapse diagnostics
COLLAPSE_STD_FLOOR = 1.0e-3    # if propagator output std drops below this, halt (plan §1g)
NOISE_INJECT_FRAC = 0.3        # injected-noise std as a fraction of the measured one-step COM
                               # displacement RMS; trains the propagator to correct rollout
                               # errors in its Δr_{t-1} input (GNS-style noise injection,
                               # Sanchez-Gonzalez 2020). Plan §1q Problem 3.
SEED = 0


def _minimum_image(d: jnp.ndarray, box: jnp.ndarray) -> jnp.ndarray:
    """Minimum-image displacement under a cubic box."""
    return d - box * jnp.round(d / box)


def measure_displacement_scale(cached, lag_frames: int, n_sample_pairs: int) -> float:
    """Measure RMS per-coordinate minimum-image COM displacement at the given lag.

    Diagnostic only — the CFM prior is the unit Gaussian. Logged so the data
    scale is on record next to the unit-prior choice.
    """
    rng = np.random.default_rng(SEED)
    box = np.asarray(cached.box)
    max_start = cached.n_frames - lag_frames
    sq_sum = 0.0
    count = 0
    for _ in range(n_sample_pairs):
        t0 = int(rng.integers(0, max_start))
        d = np.asarray(cached.com_positions[t0 + lag_frames]) - np.asarray(cached.com_positions[t0])
        d = d - box * np.round(d / box)
        sq_sum += float(np.sum(d * d))
        count += d.size
    rms = float(np.sqrt(sq_sum / count))
    logger.info("Measured RMS per-coordinate COM displacement at lag=%d: %.4f Å", lag_frames, rms)
    return rms


def make_loss_fn(molecule_species, formal_charges, species_graphs, box):
    """Build the jitted FM loss + grad function for a fixed molecular topology.

    molecule_species / formal_charges / species_graphs / box are constant across
    frames, so they are closed over and baked into the compiled function. The
    neighbor list `nl` is built outside JAX (scipy cKDTree) and passed in per step.
    """
    box_j = jnp.asarray(box, dtype=jnp.float32)

    def fm_loss(params, pos_t, pos_tp1, dr_prev, dr0, s, nl):
        dr_target = _minimum_image(pos_tp1 - pos_t, box_j)          # (M, 3) MD truth
        dr_s = (1.0 - s) * dr0 + s * dr_target                     # linear interpolant
        u_pred = propagator_core(
            params, dr_s, s, dr_prev, molecule_species, formal_charges,
            species_graphs, nl,
        )                                                          # (M, 3)
        u_target = dr_target - dr0                                 # constant FM velocity
        per_node_sq = jnp.sum((u_pred - u_target) ** 2, axis=-1)   # (M,)
        return jnp.mean(per_node_sq)

    return jax.jit(value_and_grad(fm_loss))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 FM training on FSI pairs.")
    parser.add_argument("--cache-dir", default="conductivity/fm_data/cache/traj_FSI_20k_com")
    parser.add_argument("--out-dir", default="conductivity/fm_data/fm_md_model")
    parser.add_argument("--steps", type=int, default=N_TRAIN_STEPS)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cached = load_com_cache(cache_dir)
    logger.info(
        "Loaded COM cache: %d frames, %d molecules, dt=%.1f fs",
        cached.n_frames, cached.n_molecules, cached.dt_fs,
    )

    # The CFM prior is the unit Gaussian N(0, I); rollout uses the same unit
    # prior. The measured one-step COM displacement RMS sets the noise-injection
    # scale that stabilises the autoregressive rollout (plan §1q).
    disp_rms = measure_displacement_scale(cached, LAG_FRAMES, n_sample_pairs=200)
    noise_inject_std = NOISE_INJECT_FRAC * disp_rms
    logger.info("Noise-injection std on Δr_prev: %.4f Å (%.2f × displacement RMS)",
                noise_inject_std, NOISE_INJECT_FRAC)

    molecule_species = jnp.asarray(cached.molecule_species)
    formal_charges = jnp.asarray(cached.formal_charges, dtype=jnp.float32)
    species_graphs = jax.device_put(cached.species_graphs)
    box = np.asarray(cached.box)

    loss_and_grad = make_loss_fn(molecule_species, formal_charges, species_graphs, box)
    velocity_core = jax.jit(propagator_core)

    rng = random.PRNGKey(SEED)
    init_key, rng = random.split(rng)
    params = init_propagator_params(init_key)
    logger.info("Propagator parameters: %d", count_parameters(params))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LEARNING_RATE_PEAK,
        warmup_steps=WARMUP_STEPS, decay_steps=args.steps, end_value=LEARNING_RATE_FLOOR,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY),
    )
    opt_state = optimizer.init(params)

    np_rng = np.random.default_rng(SEED)
    box_j = jnp.asarray(box, dtype=jnp.float32)
    # Second-order propagator needs frame triples: t0-1, t0, t0+LAG. Sample t0
    # so both the previous frame and the target frame exist.
    max_start = cached.n_frames - LAG_FRAMES

    loss_history: list[float] = []
    t_start = time.time()
    for step in range(args.steps):
        t0 = int(np_rng.integers(LAG_FRAMES, max_start))
        pos_tm1_np = np.asarray(cached.com_positions[t0 - LAG_FRAMES], dtype=np.float64)
        pos_t_np = np.asarray(cached.com_positions[t0], dtype=np.float64)
        pos_t = jnp.asarray(pos_t_np, dtype=jnp.float32)
        pos_tp1 = jnp.asarray(cached.com_positions[t0 + LAG_FRAMES], dtype=jnp.float32)
        # previous-step displacement (the momentum state), minimum-image
        dr_prev = _minimum_image(
            jnp.asarray(pos_t_np - pos_tm1_np, dtype=jnp.float32), box_j,
        )

        # Neighbor list built once per step outside JAX (scipy cKDTree).
        nl = jax.device_put(build_neighbor_arrays(pos_t_np, box))

        inject_key, noise_key, s_key, rng = random.split(rng, 4)
        # Noise-injection: condition on a perturbed Δr_prev so the propagator
        # learns to correct rollout errors in its momentum input (plan §1q).
        dr_prev_noised = dr_prev + noise_inject_std * random.normal(
            inject_key, (cached.n_molecules, 3),
        )
        dr0 = random.normal(noise_key, (cached.n_molecules, 3))    # unit CFM prior
        s = random.uniform(s_key, ())

        loss_val, grads = loss_and_grad(params, pos_t, pos_tp1, dr_prev_noised, dr0, s, nl)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(float(loss_val))

        if (step + 1) % LOG_EVERY == 0:
            recent = np.mean(loss_history[-LOG_EVERY:])
            logger.info(
                "step %d/%d  loss=%.5f (recent-mean %.5f)  elapsed=%.0fs",
                step + 1, args.steps, float(loss_val), recent, time.time() - t_start,
            )

        if (step + 1) % COLLAPSE_CHECK_EVERY == 0:
            # Output-collapse diagnostic (plan §1g): the propagator velocity must
            # not degenerate to a constant across molecules.
            probe_key, rng = random.split(rng)
            probe_dr = random.normal(probe_key, (cached.n_molecules, 3))
            probe_vel = velocity_core(
                params, probe_dr, 0.5, dr_prev, molecule_species, formal_charges,
                species_graphs, nl,
            )
            probe_std = float(jnp.std(probe_vel))
            logger.info("  [collapse-check] propagator output std = %.5f", probe_std)
            if probe_std < COLLAPSE_STD_FLOOR:
                logger.error(
                    "OUTPUT COLLAPSE: std %.2e < %.2e at step %d. Halting.",
                    probe_std, COLLAPSE_STD_FLOOR, step + 1,
                )
                return 1

        if (step + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = out_dir / f"propagator_step_{step + 1:05d}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "params": jax.device_get(params),
                    "step": step + 1,
                    "lag_frames": LAG_FRAMES,
                    "dt_fs": cached.dt_fs,
                }, f)
            logger.info("  saved checkpoint %s", ckpt_path)

    first_window = float(np.mean(loss_history[:LOG_EVERY]))
    last_window = float(np.mean(loss_history[-LOG_EVERY:]))
    logger.info(
        "Training done. loss first-%d-mean=%.5f  last-%d-mean=%.5f  drop=%.2fx",
        LOG_EVERY, first_window, LOG_EVERY, last_window,
        first_window / max(last_window, 1e-12),
    )

    final_path = out_dir / "propagator_final.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({
            "params": jax.device_get(params),
            "step": args.steps,
            "lag_frames": LAG_FRAMES,
            "dt_fs": cached.dt_fs,
            "loss_first_window": first_window,
            "loss_last_window": last_window,
        }, f)
    logger.info("Saved final model to %s", final_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
