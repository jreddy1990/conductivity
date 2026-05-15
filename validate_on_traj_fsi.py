"""Validate the trained FM model against MD-derived sigma from the FSI trajectory.

Computes proper FSI- molecular center-of-mass via bond detection in frame 0,
then applies Einstein-Helfand on the unwrapped charge displacement trajectory.
Compares to the trained FM model prediction for the same composition and to
literature experimental sigma for 1 m LiFSI in EC/EMC 3:7 at 60 C.

Entry: python -m conductivity.validate_on_traj_fsi
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")
import control_framework.jax_m4_tuning  # noqa

import logging
import pickle
import subprocess
import time
from enum import IntEnum
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap

from constants import K_B, E_CHARGE, S_M_TO_MS_CM

from conductivity.flow_matching_conductivity import (
    K_SPECTRUM, ODE_STEPS, LabeledRow, SigmaMethod,
)
from conductivity.fm_train_fast import (
    DATA_DIR, build_smiles_cache, row_to_arrays, stack_batch,
    composition_encoder_arrays, target_spectrum_arrays, green_kubo_arrays,
    integrate_fm_ode_arrays, RowArrays,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Element enum (atomic numbers); replaces stringly-typed element dispatch
# =============================================================================


class Element(IntEnum):
    H = 1
    Li = 3
    C = 6
    N = 7
    O = 8
    F = 9
    S = 16


_SYMBOL_TO_ELEMENT = {e.name: e for e in Element}
_ATOMIC_MASS = {
    Element.H: 1.008,   Element.Li: 6.94,   Element.C: 12.011,
    Element.N: 14.007,  Element.O: 15.999,  Element.F: 18.998,
    Element.S: 32.06,
}


# =============================================================================
# Constants
# =============================================================================
TRAJ_PATH = DATA_DIR / "trajectories" / "traj_FSI.tar.gz"
N_FRAMES_TO_READ = 50000           # Explicit constant: ~75 ps at 1.5 fs/frame
N_ATOMS_PER_FRAME = 8010           # Explicit constant: 52 LiFSI + 149 EC + 400 EMC
DT_FS = 1.5                        # Explicit constant: trajectory frame stride
TEMPERATURE_K = 333.0              # Explicit constant: simulation T per Zenodo

N_LIFSI_MOLECULES = 52
N_EC_MOLECULES = 149
N_EMC_MOLECULES = 400

CHARGE_LI = +1.0
CHARGE_FSI = -1.0

# Explicit constant: bond cutoff for intra-FSI bond detection (S-N ~1.55, S-O ~1.45, S-F ~1.6 A; 1.9 A captures all with safety)
INTRA_FSI_BOND_CUTOFF_ANG = 1.9

# Einstein-Helfand fit window (ps)
FIT_WINDOW_PS_START = 5.0
FIT_WINDOW_PS_END = 50.0

STREAM_BUFFER_BYTES = 1 << 20
LOG_INTERVAL_FRAMES = 5000

N_FM_SAMPLES_AT_INFERENCE = 32

# Literature reference: 1 m LiFSI in EC:EMC 3:7 at 60 C ~ 13-18 mS/cm
LIT_SIGMA_MIN_MSCM = 13.0
LIT_SIGMA_MAX_MSCM = 18.0


# =============================================================================
# Frame 0 inspection: identify ALL atom positions, bond FSI molecules
# =============================================================================


def read_frame0(tar_path):
    """Return positions (N, 3), element_ids (N int array of Element values), box (Å), time (fs)."""
    proc = subprocess.Popen(
        ["tar", "-xzOf", str(tar_path), "traj_FSI.xyz"],
        stdout=subprocess.PIPE, bufsize=STREAM_BUFFER_BYTES, text=True,
    )
    header = proc.stdout.readline().split()
    n_atoms = int(header[0])
    box = float(header[1])
    t_fs = float(header[2])
    proc.stdout.readline()  # blank
    element_ids = np.zeros(n_atoms, dtype=np.int32)
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        rec = proc.stdout.readline().split()
        element_ids[i] = int(_SYMBOL_TO_ELEMENT[rec[0]])
        positions[i] = [float(rec[1]), float(rec[2]), float(rec[3])]
    proc.terminate()
    proc.wait()
    return positions, element_ids, box, t_fs


def min_image_distance(r_i, r_j, box):
    """Minimum-image distance under cubic PBC."""
    d = r_j - r_i
    d = d - box * np.round(d / box)
    return float(np.linalg.norm(d))


def identify_fsi_molecules(positions, element_ids, box):
    """Return (li_indices, fsi_groups). Each fsi_groups[m] = [N, S, S, O, O, O, O, F, F] (N first for PBC unfolding)."""
    li_indices = np.where(element_ids == int(Element.Li))[0]
    n_indices = np.where(element_ids == int(Element.N))[0]
    s_indices = np.where(element_ids == int(Element.S))[0]
    f_indices = np.where(element_ids == int(Element.F))[0]
    o_indices = np.where(element_ids == int(Element.O))[0]
    if len(li_indices) != N_LIFSI_MOLECULES:
        raise ValueError(f"Expected {N_LIFSI_MOLECULES} Li, got {len(li_indices)}")
    if len(n_indices) != N_LIFSI_MOLECULES:
        raise ValueError(f"Expected {N_LIFSI_MOLECULES} N, got {len(n_indices)}")

    used = set()
    expected_composition = {
        Element.N: 1, Element.S: 2, Element.O: 4, Element.F: 2,
    }

    def neighbors_within(seed_pos, cand_pos, cand_global, cutoff):
        out = []
        for k, p in enumerate(cand_pos):
            g = int(cand_global[k])
            if g in used:
                continue
            if min_image_distance(seed_pos, p, box) < cutoff:
                out.append(g)
        return out

    fsi_groups = []
    for n_global in n_indices:
        n_global = int(n_global)
        if n_global in used:
            continue
        group = {n_global}
        used.add(n_global)
        s_neighbors = neighbors_within(
            positions[n_global], positions[s_indices],
            s_indices, INTRA_FSI_BOND_CUTOFF_ANG,
        )
        for g in s_neighbors:
            group.add(g); used.add(g)
        s_in_group = [g for g in group if Element(int(element_ids[g])) == Element.S]
        for sg in s_in_group:
            of_nbrs = neighbors_within(
                positions[sg],
                np.concatenate([positions[o_indices], positions[f_indices]]),
                np.concatenate([o_indices, f_indices]),
                INTRA_FSI_BOND_CUTOFF_ANG,
            )
            for g in of_nbrs:
                group.add(g); used.add(g)
        if len(group) != 9:
            raise RuntimeError(
                f"FSI around N atom {n_global} has {len(group)} members, expected 9. "
                f"Members: {sorted(group)}"
            )
        comp = {}
        for g in group:
            e = Element(int(element_ids[g]))
            if e in comp:
                comp[e] += 1
            else:
                comp[e] = 1
        if comp != expected_composition:
            raise RuntimeError(
                f"FSI around N atom {n_global} composition {comp} != {expected_composition}"
            )
        # Order [N, S, S, O, O, O, O, F, F] so N is at index 0 for PBC unfolding.
        ordered = [n_global]
        ordered += sorted(g for g in group if Element(int(element_ids[g])) == Element.S)
        ordered += sorted(g for g in group if Element(int(element_ids[g])) == Element.O)
        ordered += sorted(g for g in group if Element(int(element_ids[g])) == Element.F)
        fsi_groups.append(ordered)

    return list(li_indices), fsi_groups


# =============================================================================
# Stream all frames, compute Li and FSI-COM positions per frame
# =============================================================================


def stream_li_and_fsi_com(tar_path, n_frames, n_atoms, li_indices, fsi_groups, element_ids):
    """Stream every frame; emit Li and FSI-COM position arrays.

    Returns:
      li_pos:        (n_frames, 52, 3)
      fsi_com_pos:   (n_frames, 52, 3)
      box_lengths:   (n_frames,)
      times_fs:      (n_frames,)
    """
    fsi_atom_indices = np.array([g for grp in fsi_groups for g in grp], dtype=np.int64)
    fsi_atom_masses = np.array([
        _ATOMIC_MASS[Element(int(element_ids[g]))] for g in fsi_atom_indices
    ])
    fsi_group_size = 9
    fsi_total_masses = np.array([
        sum(_ATOMIC_MASS[Element(int(element_ids[g]))] for g in grp)
        for grp in fsi_groups
    ])
    fsi_atom_weights = (
        fsi_atom_masses.reshape(N_LIFSI_MOLECULES, fsi_group_size)
        / fsi_total_masses[:, None]
    )

    li_slots = {int(g): k for k, g in enumerate(li_indices)}
    fsi_slots = {}
    for mol_k, grp in enumerate(fsi_groups):
        for atom_k, g in enumerate(grp):
            fsi_slots[int(g)] = (mol_k, atom_k)
    needed_set = set(li_slots) | set(fsi_slots)

    proc = subprocess.Popen(
        ["tar", "-xzOf", str(tar_path), "traj_FSI.xyz"],
        stdout=subprocess.PIPE, bufsize=STREAM_BUFFER_BYTES, text=True,
    )

    li_pos = np.zeros((n_frames, N_LIFSI_MOLECULES, 3))
    fsi_atom_pos = np.zeros((n_frames, N_LIFSI_MOLECULES, fsi_group_size, 3))
    box_lengths = np.zeros(n_frames)
    times_fs = np.zeros(n_frames)

    t_start = time.time()
    for f in range(n_frames):
        line = proc.stdout.readline()
        if not line:
            n_frames = f
            li_pos = li_pos[:f]; fsi_atom_pos = fsi_atom_pos[:f]
            box_lengths = box_lengths[:f]; times_fs = times_fs[:f]
            break
        header = line.split()
        box_lengths[f] = float(header[1])
        times_fs[f] = float(header[2])
        proc.stdout.readline()
        for i in range(n_atoms):
            rec = proc.stdout.readline().split()
            if i in needed_set:
                xyz = [float(rec[1]), float(rec[2]), float(rec[3])]
                if i in li_slots:
                    li_pos[f, li_slots[i]] = xyz
                else:
                    mol_k, atom_k = fsi_slots[i]
                    fsi_atom_pos[f, mol_k, atom_k] = xyz
        if (f + 1) % LOG_INTERVAL_FRAMES == 0:
            elapsed = time.time() - t_start
            rate = (f + 1) / elapsed if elapsed > 0 else 0
            logger.info("frame %d / %d (%.0f frames/s, %.1fs)",
                        f + 1, n_frames, rate, elapsed)
    proc.terminate(); proc.wait()

    # PBC unfold within each FSI molecule before COM. N atom (index 0) is the
    # reference; other atoms are shifted by minimum-image displacement from N
    # so the molecule is contiguous in unbounded space even when it straddles
    # a periodic boundary. Without this, the naive mass-weighted average puts
    # the COM near the box center and the COM "moves" wildly when the molecule
    # rotates across the boundary.
    n_ref = fsi_atom_pos[:, :, 0:1, :]                  # (T, 52, 1, 3): N atom
    delta = fsi_atom_pos - n_ref                        # (T, 52, 9, 3)
    box_T = box_lengths[:, None, None, None]            # (T, 1, 1, 1)
    delta_min_image = delta - box_T * np.round(delta / box_T)
    fsi_atom_unfolded = n_ref + delta_min_image
    fsi_com_pos = np.einsum("tmag,ma->tmg", fsi_atom_unfolded, fsi_atom_weights)
    return li_pos, fsi_com_pos, box_lengths, times_fs


# =============================================================================
# Unwrap and Einstein-Helfand
# =============================================================================


def unwrap_positions(positions, box_lengths):
    out = positions.copy()
    for t in range(1, len(positions)):
        L = box_lengths[t]
        delta = out[t] - out[t - 1]
        jumps = np.round(delta / L)
        out[t] = out[t] - jumps * L
    return out


def compute_sigma_einstein_helfand(li_pos_unwrap, fsi_com_unwrap, box_lengths, times_fs):
    n_frames = len(times_fs)
    li_sum = li_pos_unwrap.sum(axis=1)
    fsi_sum = fsi_com_unwrap.sum(axis=1)
    R_charge = CHARGE_LI * li_sum + CHARGE_FSI * fsi_sum  # (T, 3)

    msd_lags_fs = np.linspace(500, 50000, 100)
    msd = np.zeros(len(msd_lags_fs))
    for i, lag_fs in enumerate(msd_lags_fs):
        lag_frames = int(round(lag_fs / DT_FS))
        if lag_frames >= n_frames:
            msd[i] = np.nan
            continue
        origin_stride = max(1, (n_frames - lag_frames) // 200)
        origins = np.arange(0, n_frames - lag_frames, origin_stride)
        disps = R_charge[origins + lag_frames] - R_charge[origins]
        msd[i] = float(np.mean(np.sum(disps ** 2, axis=-1)))

    valid = np.isfinite(msd)
    t_ps = msd_lags_fs[valid] / 1000.0
    msd_v = msd[valid]
    fit_mask = (t_ps >= FIT_WINDOW_PS_START) & (t_ps <= FIT_WINDOW_PS_END)
    if fit_mask.sum() < 5:
        raise RuntimeError(f"Not enough MSD points in fit window: {fit_mask.sum()}")
    slope_ang2_per_ps, _ = np.polyfit(t_ps[fit_mask], msd_v[fit_mask], 1)

    # Unit conversions: Angstrom -> meter, picosecond -> second
    ANG_TO_M = 1e-10                   # Definitional: 1 Å = 1e-10 m
    ANG2_PER_PS_TO_M2_PER_S = 1e-8     # Derived: (Å→m)² × (1/ps → 1/s) = 1e-20 × 1e12 = 1e-8
    EINSTEIN_HELFAND_DIM_FACTOR = 6    # Explicit constant: factor "6" in σ = e²/(6VkT)·d⟨|ΔR|²⟩/dt: 3 spatial dimensions × factor of 2 (one-sided slope vs two-sided ⟨ΔR²⟩)
    avg_box = float(np.mean(box_lengths))
    volume_m3 = (avg_box * ANG_TO_M) ** 3
    kT_J = K_B * TEMPERATURE_K
    slope_m2_per_s = slope_ang2_per_ps * ANG2_PER_PS_TO_M2_PER_S
    sigma_S_per_m = E_CHARGE ** 2 * slope_m2_per_s / (
        EINSTEIN_HELFAND_DIM_FACTOR * volume_m3 * kT_J
    )
    return sigma_S_per_m * S_M_TO_MS_CM, slope_ang2_per_ps, msd_lags_fs, msd


# =============================================================================
# Model prediction
# =============================================================================


def predict_from_model(temperature_K):
    with open(DATA_DIR / "fm_conductivity_model.pkl", "rb") as f:
        state = pickle.load(f)
    params, norm = state["params"], state["norm_stats"]

    species_counts = np.array(
        [N_LIFSI_MOLECULES, N_LIFSI_MOLECULES, N_EC_MOLECULES, N_EMC_MOLECULES],
        dtype=np.float64,
    )
    smiles_list = [
        "[Li+]",
        "O=S(=O)([N-]S(=O)(=O)F)F",
        "O=C1OCCO1",
        "CCOC(=O)OC",
    ]
    mole_fractions = species_counts / species_counts.sum()
    row = LabeledRow(
        composition_key="validate-FSI",
        smiles_list=smiles_list,
        mole_fractions=mole_fractions,
        temperature_K=temperature_K,
        sigma_mScm=1.0,
        sigma_source="validation",
        sigma_method=SigmaMethod.GREEN_KUBO,
        sigma_uncertainty_log=0.0,
    )
    cache = build_smiles_cache([row])
    data = [row_to_arrays(row, cache)]
    batch = stack_batch(data, norm)
    rows_view = RowArrays(
        atom_features=batch.atom_features, bond_features=batch.bond_features,
        bond_src=batch.bond_src, bond_dst=batch.bond_dst,
        atom_masks=batch.atom_masks, bond_masks=batch.bond_masks,
        mole_fractions=batch.mole_fractions, species_mask=batch.species_masks,
        temperature=batch.temperatures,
    )

    def det_pred(row_in):
        z = composition_encoder_arrays(params["mol_gnn"], params["attn"], row_in, norm)
        xi = target_spectrum_arrays(params["fm"], z)
        return green_kubo_arrays(xi, params["fm"]["log_sigma_offset"])
    log_sigma_det = float(vmap(det_pred)(rows_view)[0])

    single_row = jax.tree_util.tree_map(lambda x: x[0], rows_view)
    def fm_pred(row_in, key):
        z = composition_encoder_arrays(params["mol_gnn"], params["attn"], row_in, norm)
        xi_0 = random.normal(key, (2 * K_SPECTRUM,))
        xi_1 = integrate_fm_ode_arrays(params["fm"], xi_0, z, ODE_STEPS)
        return green_kubo_arrays(xi_1, params["fm"]["log_sigma_offset"])
    sample_keys = random.split(random.PRNGKey(0), N_FM_SAMPLES_AT_INFERENCE)
    samples = vmap(fm_pred, in_axes=(None, 0))(single_row, sample_keys)
    return log_sigma_det, float(jnp.mean(samples)), float(jnp.std(samples))


# =============================================================================
# Main
# =============================================================================


def main():
    if not TRAJ_PATH.exists():
        raise FileNotFoundError(f"trajectory archive not found: {TRAJ_PATH}")

    logger.info("=" * 70)
    logger.info("STEP 1: Inspect frame 0, identify FSI molecules")
    logger.info("=" * 70)
    positions0, element_ids0, box0, t0 = read_frame0(TRAJ_PATH)
    logger.info("Frame 0: %d atoms, box=%.2f A, t=%.1f fs",
                len(element_ids0), box0, t0)
    li_idx, fsi_groups = identify_fsi_molecules(positions0, element_ids0, box0)
    elem_names_first_fsi = [Element(int(element_ids0[g])).name for g in fsi_groups[0]]
    logger.info("Identified %d Li atoms and %d FSI- molecules; first FSI elements: %s",
                len(li_idx), len(fsi_groups), elem_names_first_fsi)

    logger.info("=" * 70)
    logger.info("STEP 2: Stream %d frames, compute Li and FSI-COM positions",
                N_FRAMES_TO_READ)
    logger.info("=" * 70)
    li_pos, fsi_com_pos, box_lengths, times_fs = stream_li_and_fsi_com(
        TRAJ_PATH, N_FRAMES_TO_READ, N_ATOMS_PER_FRAME,
        li_idx, fsi_groups, element_ids0,
    )
    logger.info("Loaded %d frames; span %.1f ps",
                len(times_fs), (times_fs[-1] - times_fs[0]) / 1000)

    logger.info("=" * 70)
    logger.info("STEP 3: Diagnose wrapping then compute Einstein-Helfand sigma")
    logger.info("=" * 70)
    # Diagnostic: are positions wrapped (in [0, L]) or already unwrapped?
    li_min = float(li_pos.min())
    li_max = float(li_pos.max())
    box_mean = float(np.mean(box_lengths))
    logger.info("Li position range across frames: [%.2f, %.2f] A;  mean box %.2f A",
                li_min, li_max, box_mean)
    # Wrapped positions live in approximately [-L/2, +3L/2]; unwrapped ones drift far past these bounds.
    WRAP_LO_FRAC = -0.5
    WRAP_HI_FRAC = 1.5
    is_wrapped = (li_min >= box_mean * WRAP_LO_FRAC) and (li_max <= box_mean * WRAP_HI_FRAC)
    logger.info("Positions appear %s",
                "WRAPPED (will unwrap)" if is_wrapped else "ALREADY UNWRAPPED (skip unwrap)")
    if is_wrapped:
        li_proc = unwrap_positions(li_pos, box_lengths)
        fsi_proc = unwrap_positions(fsi_com_pos, box_lengths)
    else:
        li_proc = li_pos
        fsi_proc = fsi_com_pos
    sigma_traj, slope, lags_fs, msd = compute_sigma_einstein_helfand(
        li_proc, fsi_proc, box_lengths, times_fs,
    )
    logger.info("MSD slope (charge displacement) = %.4f A^2/ps in [%g, %g] ps",
                slope, FIT_WINDOW_PS_START, FIT_WINDOW_PS_END)
    logger.info("Trajectory-derived sigma (Einstein-Helfand): %.3f mS/cm", sigma_traj)

    logger.info("=" * 70)
    logger.info("STEP 4: Model prediction for matching composition")
    logger.info("=" * 70)
    log_det, log_fm_mean, log_fm_std = predict_from_model(TEMPERATURE_K)
    sigma_det = float(np.exp(log_det))
    sigma_fm = float(np.exp(log_fm_mean))
    logger.info("Model deterministic:  %.3f mS/cm", sigma_det)
    logger.info("Model FM-sampled:     %.3f mS/cm (factor x%.2f over %d samples)",
                sigma_fm, float(np.exp(log_fm_std)), N_FM_SAMPLES_AT_INFERENCE)

    logger.info("=" * 70)
    logger.info("STEP 5: Comparison vs literature")
    logger.info("=" * 70)
    logger.info("Composition: 1 m LiFSI in EC:EMC 3:7 at %g K", TEMPERATURE_K)
    logger.info("Literature range: %g - %g mS/cm",
                LIT_SIGMA_MIN_MSCM, LIT_SIGMA_MAX_MSCM)
    logger.info("Trajectory (Einstein-Helfand): %.3f mS/cm", sigma_traj)
    logger.info("Model deterministic:           %.3f mS/cm", sigma_det)
    logger.info("Model FM-sampled mean:         %.3f mS/cm", sigma_fm)
    mid_lit = 0.5 * (LIT_SIGMA_MIN_MSCM + LIT_SIGMA_MAX_MSCM)
    logger.info("Ratios to literature mid-point %.1f mS/cm:", mid_lit)
    logger.info("  trajectory / lit = %.2f", sigma_traj / mid_lit)
    logger.info("  model det / lit  = %.2f", sigma_det / mid_lit)
    logger.info("  model FM / lit   = %.2f", sigma_fm / mid_lit)


if __name__ == "__main__":
    main()
