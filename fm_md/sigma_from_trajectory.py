"""Phase 6: ionic conductivity from a molecular-COM trajectory via cepstral analysis.

Consumes a `MolecularTrajectory` (PBC-unwrapped molecular center-of-mass paths
with per-molecule formal charges), builds the molecular charge current
`J(t) = Σ_mol q_mol · v_COM,mol(t)` by central finite difference, and routes it
through the project's existing cepstral conductivity estimator (Ercole 2017).

This is FF-free: charges are molecular formal integers anchored at the COM, no
atomic partial charges enter, no Nernst-Einstein assumption is made.

Reuses the existing pipeline rather than reimplementing:
  - foundation_model.current_spectrum_estimator.cation_anion_block_currents_from_velocities
  - foundation_model.cepstral_conductivity.estimate_conductivity_cepstral

Entry: `python -m conductivity.fm_md.sigma_from_trajectory`
loads the FSI trajectory, builds the COM time series, computes σ, and reports
whether it lands within the literature range for 1 m LiFSI in EC:EMC 3:7 at 333 K
(13-18 mS/cm).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from foundation_model.current_spectrum_estimator import (
    CONV_FACTOR,
    KB_EV_PER_K,
    CurrentBurstSet,
    cation_anion_block_currents_from_velocities,
)
from foundation_model.cepstral_conductivity import (
    CepstralConductivityEstimate,
    SegmentationConfig,
    TruncationAIC,
    estimate_conductivity_cepstral,
)

from conductivity.fm_md.atomistic_io import (
    MolecularTrajectory,
    TRAJ_FSI_DESCRIPTOR,
    load_molecular_trajectory,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Segmentation / truncation defaults
# =============================================================================


N_BURSTS_DEFAULT = 4           # Explicit constant: minimum for bootstrap CI; gives 4-way burst replicas
DPSS_TIME_BANDWIDTH = 2.5      # Explicit constant: standard multi-taper choice (Ercole 2017 §IV uses 2-4)
DPSS_N_TAPERS = 2              # Explicit constant: K=2 < 2·NW = 5 for orthogonality (Thomson 1982)
N_BOOTSTRAP = 200              # Explicit constant: bootstrap resamples for the burst-level CI
RANDOM_SEED = 42               # Explicit constant: reproducible bootstrap


# Literature reference for the FSI trajectory's composition: 1 m LiFSI in EC:EMC
# 3:7 at 60°C is reported in 13-18 mS/cm across multiple measurements (Logan
# et al. JES 2018; Hall et al. JES 2018). Used by the Phase 6 sanity gate only.
LIT_SIGMA_LOW_MSCM = 13.0
LIT_SIGMA_HIGH_MSCM = 18.0


# =============================================================================
# Velocity construction
# =============================================================================


def central_difference_velocities(com_positions: np.ndarray, dt_fs: float) -> np.ndarray:
    """Central-difference COM velocities, second-order accurate.

    Input:  com_positions shape (n_frames, n_molecules, 3) — must be unwrapped
            so consecutive frames are spatially continuous (no PBC jumps).
    Output: velocities shape (n_frames - 2, n_molecules, 3), units Å/fs.

    v_t = (r_{t+1} - r_{t-1}) / (2 dt). Drops the first and last frames since
    they have no neighbour on one side; this is the price of second-order
    accuracy and is acceptable given trajectories of thousands of frames.
    """
    if com_positions.ndim != 3:
        raise ValueError(f"com_positions must be 3D, got shape {com_positions.shape}")
    if com_positions.shape[0] < 3:
        raise ValueError(f"Need at least 3 frames for central differences, got {com_positions.shape[0]}")
    if dt_fs <= 0:
        raise ValueError(f"dt_fs must be > 0, got {dt_fs}")
    return (com_positions[2:] - com_positions[:-2]) / (2.0 * dt_fs)


# =============================================================================
# Phase 6 entrypoint (contracted signature)
# =============================================================================


def sigma_from_trajectory(
    trajectory: MolecularTrajectory,
    charges: np.ndarray,
    temperature_K: float,
    volume_ang3: float,
) -> CepstralConductivityEstimate:
    """Estimate ionic conductivity from a molecular-COM trajectory.

    Builds J(t) = Σ_mol q_mol · v_COM,mol(t) by central-difference velocities,
    splits the resulting time series into N_BURSTS_DEFAULT equal bursts, routes
    cation and anion charge currents through the project's existing cepstral
    estimator (Ercole 2017), and returns a `CepstralConductivityEstimate`
    carrying both bootstrap and asymptotic CIs.

    Inputs:
      trajectory:    MolecularTrajectory with `.com_positions` shape
                     (n_frames, n_molecules, 3) UNWRAPPED, and `.dt_fs`.
      charges:       (n_molecules,) per-molecule formal integer charges
                     (Li+ → +1, FSI- → -1, etc.). MUST contain both cations
                     (charges > 0) and anions (charges < 0).
      temperature_K: scalar K, used in σ = (1/3VkT)·S(ω→0).
      volume_ang3:   scalar Å³, the simulation box volume.

    Returns the cepstral estimate; σ is in mS/cm.
    """
    com_positions = trajectory.com_positions
    if com_positions.shape[1] != charges.shape[0]:
        raise ValueError(
            f"trajectory has {com_positions.shape[1]} molecules but charges has {charges.shape[0]}"
        )
    if temperature_K <= 0:
        raise ValueError(f"temperature_K must be > 0, got {temperature_K}")
    if volume_ang3 <= 0:
        raise ValueError(f"volume_ang3 must be > 0, got {volume_ang3}")

    velocities = central_difference_velocities(com_positions, trajectory.dt_fs)
    n_velocity_frames, n_molecules, _ = velocities.shape
    logger.info(
        "Built %d velocity frames from %d position frames (dt=%.2f fs); n_molecules=%d",
        n_velocity_frames, com_positions.shape[0], trajectory.dt_fs, n_molecules,
    )

    n_bursts = N_BURSTS_DEFAULT
    n_steps_per_burst = n_velocity_frames // n_bursts
    if n_steps_per_burst < 32:
        raise ValueError(
            f"Each burst would have {n_steps_per_burst} steps (n_velocity_frames={n_velocity_frames}, "
            f"n_bursts={n_bursts}); need at least 32 for a meaningful FFT"
        )
    n_total = n_bursts * n_steps_per_burst
    velocities_burst = velocities[:n_total].reshape(n_bursts, n_steps_per_burst, n_molecules, 3)
    logger.info(
        "Split into %d bursts × %d steps (each burst %.2f ps)",
        n_bursts, n_steps_per_burst, n_steps_per_burst * trajectory.dt_fs / 1000.0,
    )

    block_currents_list = []
    labels_canonical: list[str] | None = None
    for b_idx in range(n_bursts):
        block_b, labels = cation_anion_block_currents_from_velocities(
            velocities_burst[b_idx], charges.astype(np.float64),
        )
        block_currents_list.append(block_b)
        if labels_canonical is None:
            labels_canonical = labels
        elif labels != labels_canonical:
            raise RuntimeError(f"Block labels drift across bursts: {labels_canonical} vs {labels}")
    if labels_canonical is None:
        raise RuntimeError("No bursts processed; cannot determine block labels")
    block_currents = np.stack(block_currents_list, axis=0)
    logger.info(
        "Block currents shape %s (axes: bursts, steps, blocks=[cation, anion], xyz)",
        block_currents.shape,
    )

    burst_set = CurrentBurstSet(
        block_currents=block_currents,
        dt_fs=trajectory.dt_fs,
        volume_ang3=volume_ang3,
        temperature_K=temperature_K,
        block_labels=labels_canonical,
    )

    segmentation = SegmentationConfig(
        segment_length=n_steps_per_burst,   # one segment = whole burst (no Welch sub-segmentation)
        overlap=0,
        n_tapers=DPSS_N_TAPERS,
        time_bandwidth=DPSS_TIME_BANDWIDTH,
    )

    estimate = estimate_conductivity_cepstral(
        burst_set=burst_set,
        truncation=TruncationAIC(),
        segmentation=segmentation,
        n_bootstrap=N_BOOTSTRAP,
        random_seed=RANDOM_SEED,
    )
    logger.info(
        "σ = %.3f mS/cm  (bootstrap CI [%.3f, %.3f], asymptotic CI [%.3f, %.3f], P*=%d)",
        estimate.sigma_mS_cm,
        estimate.bootstrap_ci_low_mS_cm, estimate.bootstrap_ci_high_mS_cm,
        estimate.asymptotic_ci_low_mS_cm, estimate.asymptotic_ci_high_mS_cm,
        estimate.P_star,
    )
    return estimate


# =============================================================================
# Sigma from many short rollout bursts (autoregressive-rollout-stable path)
# =============================================================================


def sigma_from_burst_rollouts(
    burst_com_positions: list[np.ndarray],
    dt_fs: float,
    charges: np.ndarray,
    temperature_K: float,
    volume_ang3: float,
) -> CepstralConductivityEstimate:
    """Estimate sigma from many short molecular-COM rollout bursts.

    Each burst is an independent short rollout (well below the long-rollout
    divergence onset, plan §1q). The bursts are the cepstral estimator's native
    multi-burst input, so the correlation factor [1 + 2Σρ(τ)] — which converges
    by τ ≈ 5 — is resolved within each burst and pooled across bursts.

    Inputs:
      burst_com_positions: list of (burst_frames, n_molecules, 3) UNWRAPPED COM
                           arrays, all the same length.
      dt_fs, charges, temperature_K, volume_ang3: as in `sigma_from_trajectory`.
    """
    if len(burst_com_positions) < 2:
        raise ValueError(f"Need >= 2 bursts for the cepstral estimator, got {len(burst_com_positions)}")

    block_list = []
    labels_canonical: list[str] | None = None
    for com in burst_com_positions:
        velocities = central_difference_velocities(com, dt_fs)
        block_b, labels = cation_anion_block_currents_from_velocities(
            velocities, charges.astype(np.float64),
        )
        block_list.append(block_b)
        if labels_canonical is None:
            labels_canonical = labels
        elif labels != labels_canonical:
            raise RuntimeError(f"Block labels drift across bursts: {labels_canonical} vs {labels}")
    if labels_canonical is None:
        raise RuntimeError("No bursts processed")

    burst_lengths = {b.shape[0] for b in block_list}
    if len(burst_lengths) != 1:
        raise ValueError(f"All bursts must have equal length; got {sorted(burst_lengths)}")
    block_currents = np.stack(block_list, axis=0)
    logger.info(
        "Burst-rollout block currents shape %s (bursts, steps, [cation, anion], xyz)",
        block_currents.shape,
    )

    burst_set = CurrentBurstSet(
        block_currents=block_currents,
        dt_fs=dt_fs,
        volume_ang3=volume_ang3,
        temperature_K=temperature_K,
        block_labels=labels_canonical,
    )
    segmentation = SegmentationConfig(
        segment_length=block_currents.shape[1],
        overlap=0,
        n_tapers=DPSS_N_TAPERS,
        time_bandwidth=DPSS_TIME_BANDWIDTH,
    )
    estimate = estimate_conductivity_cepstral(
        burst_set=burst_set,
        truncation=TruncationAIC(),
        segmentation=segmentation,
        n_bootstrap=N_BOOTSTRAP,
        random_seed=RANDOM_SEED,
    )
    logger.info(
        "σ = %.3f mS/cm  (bootstrap CI [%.3f, %.3f], asymptotic CI [%.3f, %.3f], P*=%d)",
        estimate.sigma_mS_cm,
        estimate.bootstrap_ci_low_mS_cm, estimate.bootstrap_ci_high_mS_cm,
        estimate.asymptotic_ci_low_mS_cm, estimate.asymptotic_ci_high_mS_cm,
        estimate.P_star,
    )
    return estimate


# =============================================================================
# Direct Green-Kubo rho-sum (ultra-short-segment, divergence-proof path)
# =============================================================================


S_PER_M_TO_MS_PER_CM = 10.0   # unit conversion: 1 S/m = 10 mS/cm


@dataclass
class RhoSumEstimate:
    """Direct Green-Kubo rho-sum result. sigma in mS/cm."""
    sigma_mS_cm: float
    rho: np.ndarray                 # normalized autocorrelation rho(0..max_tau)
    correlation_factor: float       # 1 + 2*sum_{tau>=1} rho(tau)
    C_J0: float                     # instantaneous charge-current variance, per component
    n_segments: int
    diverged_fraction: float        # fraction of segments flagged as blown up


def sigma_direct_rho_sum(
    com_segments: list[np.ndarray],
    dt_fs: float,
    charges: np.ndarray,
    box: np.ndarray,
    temperature_K: float,
    volume_ang3: float,
    max_tau: int,
    divergence_displacement_ang: float,
) -> RhoSumEstimate:
    """Direct discrete Green-Kubo sum from ultra-short COM segments.

    sigma ∝ S_J(0) = dt · [C_J(0) + 2 Σ_{τ=1}^{max_tau} C_J(τ)], where C_J(τ) is
    the charge-current autocovariance per Cartesian component and
    J(t) = Σ_mol q_mol v_mol(t). Each segment is short enough that the
    autoregressive rollout cannot diverge to overflow (plan §1q); segments whose
    per-step COM displacement exceeds `divergence_displacement_ang` are counted
    and excluded — the excluded fraction is itself a stability diagnostic.

    Inputs:
      com_segments: list of (seg_frames, n_molecules, 3) COM arrays, seg_frames
                    >= max_tau + 2.
      dt_fs, charges, box, temperature_K, volume_ang3: physical parameters.
      max_tau: highest lag τ in the rho-sum.
      divergence_displacement_ang: per-step |Δr| above which a segment is judged
                    to have diverged and is excluded from the average.
    """
    q = np.asarray(charges, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)

    J_segments: list[np.ndarray] = []
    n_diverged = 0
    for com in com_segments:
        d = com[1:] - com[:-1]
        d = d - box * np.round(d / box)               # minimum image
        if np.max(np.linalg.norm(d, axis=-1)) > divergence_displacement_ang:
            n_diverged += 1
            continue
        velocities = d / dt_fs
        J_segments.append(np.einsum("m,nmx->nx", q, velocities))   # (seg_frames-1, 3)

    if len(J_segments) < 2:
        raise RuntimeError(
            f"Only {len(J_segments)} non-diverged segments out of {len(com_segments)}; "
            f"the propagator's rollout is too unstable to estimate sigma."
        )

    # pooled-mean demean (remove finite-sample DC bias)
    J_mean = np.concatenate(J_segments, axis=0).mean(axis=0)

    C = np.zeros(max_tau + 1, dtype=np.float64)
    counts = np.zeros(max_tau + 1, dtype=np.float64)
    for J in J_segments:
        Jc = J - J_mean
        n = Jc.shape[0]
        for tau in range(max_tau + 1):
            if n - tau <= 0:
                continue
            C[tau] += float(np.sum(Jc[: n - tau] * Jc[tau:]))
            counts[tau] += 3.0 * (n - tau)            # 3 Cartesian components
    C = C / counts                                    # per-component autocovariance

    s0_continuous = dt_fs * (C[0] + 2.0 * float(C[1:].sum()))
    kT_eV = KB_EV_PER_K * temperature_K
    sigma_S_m = (CONV_FACTOR / 2.0) * s0_continuous / (volume_ang3 * kT_eV)
    sigma_mS_cm = sigma_S_m * S_PER_M_TO_MS_PER_CM

    rho = C / C[0]
    correlation_factor = 1.0 + 2.0 * float(rho[1:].sum())
    logger.info(
        "Direct ρ-sum: σ = %.3f mS/cm | C_J(0) = %.5e | [1+2Σρ] = %.4f | "
        "ρ(1..%d) = %s | diverged %d/%d segments",
        sigma_mS_cm, C[0], correlation_factor, max_tau,
        np.array2string(rho[1:], precision=3), n_diverged, len(com_segments),
    )
    return RhoSumEstimate(
        sigma_mS_cm=sigma_mS_cm,
        rho=rho,
        correlation_factor=correlation_factor,
        C_J0=float(C[0]),
        n_segments=len(J_segments),
        diverged_fraction=n_diverged / len(com_segments),
    )


# =============================================================================
# CLI: end-to-end check on the raw FSI trajectory
# =============================================================================


@dataclass
class GateResult:
    sigma_mS_cm: float
    asymptotic_ci_low_mS_cm: float
    asymptotic_ci_high_mS_cm: float
    literature_low: float
    literature_high: float
    within_factor_2_of_literature: bool


def phase6_gate(estimate: CepstralConductivityEstimate) -> GateResult:
    """Phase 6 validation gate: σ within factor of 2 of LiFSI literature range."""
    sigma = estimate.sigma_mS_cm
    factor2_low = LIT_SIGMA_LOW_MSCM / 2.0
    factor2_high = LIT_SIGMA_HIGH_MSCM * 2.0
    within = (factor2_low <= sigma <= factor2_high)
    return GateResult(
        sigma_mS_cm=sigma,
        asymptotic_ci_low_mS_cm=estimate.asymptotic_ci_low_mS_cm,
        asymptotic_ci_high_mS_cm=estimate.asymptotic_ci_high_mS_cm,
        literature_low=LIT_SIGMA_LOW_MSCM,
        literature_high=LIT_SIGMA_HIGH_MSCM,
        within_factor_2_of_literature=within,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 cepstral σ on raw FSI trajectory.")
    parser.add_argument("--max-frames", type=int, default=20000,
                        help="Max raw frames to consume from the trajectory.")
    parser.add_argument("--frame-stride", type=int, default=20,
                        help="Raw-frame stride; effective dt = stride × descriptor.dt_fs.")
    args = parser.parse_args()

    logger.info("=== Phase 6 cepstral σ on raw FSI trajectory ===")
    trajectory = load_molecular_trajectory(
        descriptor=TRAJ_FSI_DESCRIPTOR,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    box_volume_ang3 = float(np.prod(trajectory.box))
    estimate = sigma_from_trajectory(
        trajectory=trajectory,
        charges=trajectory.formal_charges.astype(np.float64),
        temperature_K=trajectory.temperature_K,
        volume_ang3=box_volume_ang3,
    )

    gate = phase6_gate(estimate)
    logger.info("=== Phase 6 gate ===")
    logger.info("σ:                       %.3f mS/cm", gate.sigma_mS_cm)
    logger.info("asymptotic CI:           [%.3f, %.3f] mS/cm",
                gate.asymptotic_ci_low_mS_cm, gate.asymptotic_ci_high_mS_cm)
    logger.info("literature LiFSI range:  [%.1f, %.1f] mS/cm",
                gate.literature_low, gate.literature_high)
    logger.info("within factor of 2:      %s", gate.within_factor_2_of_literature)
    return 0 if gate.within_factor_2_of_literature else 1


if __name__ == "__main__":
    sys.exit(main())
