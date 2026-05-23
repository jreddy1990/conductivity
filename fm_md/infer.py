"""End-to-end inference for arbitrary electrolyte compositions (plan §2).

`predict_sigma(model, species_list, mole_fractions, temperature_K, sim_time_ps, seed)`
is the single-call entry point: it builds an initial atomistic box from the
composition, rolls the trained propagator forward for `sim_time_ps` of physical
time, and reads sigma off the resulting trajectory with the cepstral estimator.

The implementation composes three pieces, all from this package:
  - `build_initial_configuration` (box_constructor.py): composition → MolecularConfiguration
  - `rollout_trajectory` (rollout.py): propagator advance to a MolecularTrajectory
  - `sigma_from_trajectory` (sigma_from_trajectory.py): cepstral sigma readout

Per plan §3 PLAN CONTRACT, predict_sigma MUST call build_initial_configuration
AND rollout_trajectory. Both are explicit in this module's call graph.

Inference is FF-free (plan §0): no atom-typed pair potentials, no FF-based
equilibration. The propagator handles relaxation through ACF_ROLL_BURN-style
burn-in inside the rollout; we discard the first DEFAULT_BURN_PS picoseconds
of the rolled trajectory before reading sigma so the chain settles on
pi_theta first.
"""

from __future__ import annotations

import control_framework.jax_m4_tuning  # noqa: F401  -- MUST precede any jax import
                                         # (sets JAX_PLATFORMS=cpu so we never
                                         # touch the Metal backend, per CLAUDE.md)

import logging
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import jax
from jax import random

from conductivity.fm_md.box_constructor import build_initial_configuration
from conductivity.fm_md.rollout import rollout_trajectory
from conductivity.fm_md.sigma_from_trajectory import (
    CepstralConductivityEstimate,
    sigma_from_trajectory,
)

logger = logging.getLogger(__name__)


# Explicit constants for the inference pipeline.
# Atom-count target for the inference box. 2000 atoms gives ~150-200 molecules
# at typical electrolyte stoichiometries, large enough that the per-step
# charge-current variance is small but not so large that a 12 h per-composition
# budget is blown. The FSI training box is 8010 atoms; smaller is fine because
# the molecular encoder is per-molecule.
DEFAULT_TARGET_N_ATOMS = 2000

# Burn-in (picoseconds) to drop from the rolled trajectory before reading sigma.
# The propagator starts from a uniformly-random COM layout with zero momentum.
# A first attempt at DEFAULT_BURN_PS = 30 (20 steps at dt=1.5 ps) produced
# sigma = 65 mS/cm on the FSI composition with the step-50 checkpoint, against
# the audit's 22 mS/cm from a 2000-step rollout starting from an MD-equilibrated
# frame -- a 3x inflation attributable to the cold-start transient that 30 ps
# did not wash out. Bumped to 150 ps (100 propagator steps) so a uniform-random
# Li/FSI/EC/EMC box has time to develop a physical solvation structure before
# the cepstral window opens.
DEFAULT_BURN_PS = 150.0


def _load_model(model: Path | str | dict) -> tuple[dict, float, float]:
    """Resolve the `model` argument to (params, dt_fs, sigma_prior).

    Accepts either a pickle path (str/Path) or an already-loaded dict
    containing 'params', 'dt_fs', and 'sigma_prior' keys. The sigma_prior is
    the per-component std of the CFM Gaussian prior the model was trained
    with — every checkpoint produced by train.py since plan learning 1s
    carries it explicitly, and the OOD eval requires it (plan §3 contract
    update 2026-05-21).
    """
    if isinstance(model, (str, Path)):
        with open(model, "rb") as f:
            ckpt = pickle.load(f)
        if "sigma_prior" not in ckpt:
            raise ValueError(
                f"checkpoint {model} missing required 'sigma_prior' key; "
                f"every train.py checkpoint since plan learning 1s carries it"
            )
        return ckpt["params"], float(ckpt["dt_fs"]), float(ckpt["sigma_prior"])
    if isinstance(model, dict):
        for k in ("params", "dt_fs", "sigma_prior"):
            if k not in model:
                raise ValueError(f"model dict missing required {k!r} key")
        return model["params"], float(model["dt_fs"]), float(model["sigma_prior"])
    raise TypeError(f"model must be a Path/str/dict; got {type(model)}")


def predict_sigma(
    model,
    species_list: Sequence[str],
    mole_fractions: Sequence[float],
    temperature_K: float,
    sim_time_ps: float,
    seed: int,
) -> CepstralConductivityEstimate:
    """End-to-end sigma prediction for an arbitrary electrolyte composition.

    Inputs:
      model           : pickle path to a trained propagator checkpoint, or a
                        dict with 'params' and 'dt_fs' keys.
      species_list    : species names from SPECIES_CATALOGUE.
      mole_fractions  : mole-fraction vector matching species_list.
      temperature_K   : simulation temperature.
      sim_time_ps     : total simulated time. The first DEFAULT_BURN_PS ps are
                        discarded as burn-in; sigma is read from the remaining
                        trajectory.
      seed            : RNG seed for the box layout and the rollout noise.

    Returns the cepstral conductivity estimate (in mS/cm with bootstrap and
    asymptotic CIs).
    """
    params, dt_fs, sigma_prior = _load_model(model)

    # 1) Initial configuration from composition.
    x_init = build_initial_configuration(
        species_list=species_list,
        mole_fractions=mole_fractions,
        temperature_K=temperature_K,
        target_n_atoms=DEFAULT_TARGET_N_ATOMS,
        seed=seed,
    )
    logger.info("predict_sigma: built initial box with %d molecules, box_edge=%.3f Å, "
                "dt_fs=%.1f, sigma_prior=%.4f Å",
                x_init.n_molecules, float(x_init.box[0]), dt_fs, sigma_prior)

    # 2) Rollout. n_steps from sim_time_ps and dt_fs.
    sim_time_fs = sim_time_ps * 1000.0
    n_steps_total = int(round(sim_time_fs / dt_fs))
    n_burn_steps = int(round(DEFAULT_BURN_PS * 1000.0 / dt_fs))
    if n_steps_total <= n_burn_steps + 64:
        raise ValueError(
            f"sim_time_ps={sim_time_ps} gives n_steps={n_steps_total} <= "
            f"burn+64={n_burn_steps + 64}. Increase sim_time_ps."
        )
    logger.info("predict_sigma: rolling %d propagator steps (%.1f ps); burn=%d steps",
                n_steps_total, sim_time_ps, n_burn_steps)
    trajectory = rollout_trajectory(
        params=params,
        x_init=x_init,
        n_steps=n_steps_total,
        dt_fs=dt_fs,
        rng=random.PRNGKey(seed),
        sigma_prior=sigma_prior,
    )

    # 3) Drop burn-in and recompute trajectory metadata.
    trimmed_positions = np.asarray(trajectory.com_positions)[n_burn_steps:]
    from conductivity.fm_md.atomistic_io import MolecularTrajectory
    trimmed = MolecularTrajectory(
        com_positions=trimmed_positions,
        molecule_species=trajectory.molecule_species,
        formal_charges=trajectory.formal_charges,
        box=trajectory.box,
        dt_fs=trajectory.dt_fs,
        n_frames=int(trimmed_positions.shape[0]),
        n_molecules=trajectory.n_molecules,
        temperature_K=temperature_K,
    )
    logger.info("predict_sigma: kept %d frames after burn-in", trimmed.n_frames)

    # 4) Cepstral sigma readout.
    volume_ang3 = float(np.prod(trajectory.box))
    estimate = sigma_from_trajectory(
        trajectory=trimmed,
        charges=np.asarray(x_init.formal_charges),
        temperature_K=temperature_K,
        volume_ang3=volume_ang3,
    )
    logger.info("predict_sigma: sigma = %.3f mS/cm (bootstrap CI [%.3f, %.3f], asymptotic CI [%.3f, %.3f])",
                estimate.sigma_mS_cm, estimate.bootstrap_ci_low_mS_cm, estimate.bootstrap_ci_high_mS_cm,
                estimate.asymptotic_ci_low_mS_cm, estimate.asymptotic_ci_high_mS_cm)
    return estimate


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    parser = argparse.ArgumentParser(description="Predict sigma for a composition.")
    parser.add_argument("--model", required=True, help="Path to propagator checkpoint pickle.")
    parser.add_argument("--sim-time-ps", type=float, default=2400.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--species", nargs="+",
                        default=["Li+", "FSI-", "EC", "EMC"])
    parser.add_argument("--mole-fractions", nargs="+", type=float,
                        default=[52/653, 52/653, 149/653, 400/653])
    parser.add_argument("--temperature-K", type=float, default=333.0)
    args = parser.parse_args()
    estimate = predict_sigma(
        model=args.model,
        species_list=args.species,
        mole_fractions=args.mole_fractions,
        temperature_K=args.temperature_K,
        sim_time_ps=args.sim_time_ps,
        seed=args.seed,
    )
    print(f"sigma = {estimate.sigma_mS_cm:.3f} mS/cm")
