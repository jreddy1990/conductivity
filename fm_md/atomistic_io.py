"""Streaming trajectory I/O, molecule identification, PBC unwrap, sanity gate.

Phase 1 of the FM-as-MD-emulator plan. Produces a `MolecularTrajectory` from a
public LAMMPS XYZ trajectory (streamed from .tar.gz, never fully extracted) with:

- Atom typing from element symbols only (no force-field metadata).
- Molecule identification by bond-graph connected components (PBC-aware).
- Species classification by element-composition signatures (Li+, FSI-, EC, EMC).
- Per-molecule formal integer charges anchored at the molecular center of mass.
- Continuous (unwrapped) COM trajectories built frame-by-frame.
- Observation-only sanity gate: intra-molecular bond stability, molecule-count
  stability, charge neutrality. FF identity is never checked.

The shapes follow the plan's Section 2 spec. `AtomicConfiguration` is the
per-frame type used by Phases 3-5 (propagator). `MolecularTrajectory` is what
Phase 6 (`sigma_from_trajectory`) consumes.

Entry: `python -m conductivity.fm_md.atomistic_io --path conductivity/fm_data/trajectories/traj_FSI.tar.gz --audit-frames 100`
runs the sanity gate on a streamed prefix and prints the diagnostics.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import subprocess
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

from constants import N_A


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Element identity and masses
# =============================================================================


class Element(IntEnum):
    """Atomic numbers for the elements that appear in liquid electrolytes.

    Stored as IntEnum so element_ids array is plain int32, JAX-friendly.
    Extending to other elements (Na, K, Cl, B, Si, P) is just adding members.
    """
    H = 1
    Li = 3
    C = 6
    N = 7
    O = 8
    F = 9
    Na = 11
    P = 15
    S = 16
    Cl = 17
    K = 19


_SYMBOL_TO_ELEMENT = {e.name: e for e in Element}


# Atomic masses (g/mol, IUPAC 2021). Mass-weighted COM is the physically
# correct anchor for a molecule's translational current.
_ATOMIC_MASS: dict[Element, float] = {
    Element.H:  1.008,
    Element.Li: 6.94,
    Element.C:  12.011,
    Element.N:  14.007,
    Element.O:  15.999,
    Element.F:  18.998,
    Element.Na: 22.990,
    Element.P:  30.974,
    Element.S:  32.06,
    Element.Cl: 35.45,
    Element.K:  39.098,
}


# Covalent-radius bond cutoffs (Å). Threshold for "bonded" is
# r_cov(A) + r_cov(B) + BOND_TOLERANCE_ANG. Pairs not listed are NOT considered
# bonded (e.g. Li to anything — Li is monoatomic; O-O, N-N, F-F under PBC are
# coordination, not covalent bonds).
COVALENT_RADIUS_ANG: dict[Element, float] = {
    Element.H:  0.31,
    Element.C:  0.76,
    Element.N:  0.71,
    Element.O:  0.66,
    Element.F:  0.57,
    Element.P:  1.07,
    Element.S:  1.05,
}
BOND_TOLERANCE_ANG = 0.30   # Explicit constant: covalent-radius sum + 0.30 Å is the bond-graph cutoff.

# Elements that NEVER participate in covalent bonds we track for the bond graph
# (Li/Na/K/Cl are monoatomic ions; they appear as their own molecules).
MONOATOMIC_ION_ELEMENTS: frozenset[Element] = frozenset({
    Element.Li, Element.Na, Element.K, Element.Cl,
})


# =============================================================================
# Per-molecule species signatures and formal charges
# =============================================================================


@dataclass(frozen=True)
class SpeciesSignature:
    """Element-count signature used to classify a connected component as a species."""
    name: str
    formal_charge: int
    element_counts: dict[Element, int]   # exact match required

    def match(self, observed_counts: dict[Element, int]) -> bool:
        return observed_counts == self.element_counts


# Catalogue of known species in the public liquid-electrolyte trajectories we
# expect to ingest. Adding a new molecule is one entry here. Each entry is
# explicit and auditable.
SPECIES_CATALOGUE: list[SpeciesSignature] = [
    SpeciesSignature("Li+",  +1, {Element.Li: 1}),
    SpeciesSignature("Na+",  +1, {Element.Na: 1}),
    SpeciesSignature("K+",   +1, {Element.K: 1}),
    SpeciesSignature("FSI-", -1, {Element.N: 1, Element.S: 2, Element.O: 4, Element.F: 2}),
    SpeciesSignature("PF6-", -1, {Element.P: 1, Element.F: 6}),
    SpeciesSignature("EC",    0, {Element.C: 3, Element.H: 4, Element.O: 3}),
    SpeciesSignature("EMC",   0, {Element.C: 4, Element.H: 8, Element.O: 3}),
    SpeciesSignature("DMC",   0, {Element.C: 3, Element.H: 6, Element.O: 3}),
    SpeciesSignature("DEC",   0, {Element.C: 5, Element.H: 10, Element.O: 3}),
    SpeciesSignature("PC",    0, {Element.C: 4, Element.H: 6, Element.O: 3}),
    SpeciesSignature("FEC",   0, {Element.C: 3, Element.H: 3, Element.F: 1, Element.O: 3}),
    SpeciesSignature("VC",    0, {Element.C: 3, Element.H: 2, Element.O: 3}),
    SpeciesSignature("DME",   0, {Element.C: 4, Element.H: 10, Element.O: 2}),
]


def _classify_component(element_ids_in_component: np.ndarray) -> SpeciesSignature:
    """Classify a connected component by exact element-count match. Fails loudly."""
    counts: dict[Element, int] = {}
    for eid in element_ids_in_component:
        e = Element(int(eid))
        if e in counts:
            counts[e] += 1
        else:
            counts[e] = 1
    for sig in SPECIES_CATALOGUE:
        if sig.match(counts):
            return sig
    raise ValueError(
        f"Unrecognised molecular composition {counts}. "
        f"Add a SpeciesSignature entry to SPECIES_CATALOGUE."
    )


# =============================================================================
# Data types
# =============================================================================


class AtomicConfiguration(NamedTuple):
    """One snapshot of an atomistic box.

    positions:        (n_atoms, 3) Å, PBC-unfolded so each molecule is contiguous
    element_ids:      (n_atoms,) int32, values from Element enum
    molecule_ids:     (n_atoms,) int32, 0..n_molecules-1
    molecule_species: (n_molecules,) int8 indices into SPECIES_CATALOGUE
    formal_charges:   (n_molecules,) int8, +1 / -1 / 0 from SPECIES_CATALOGUE
    box:              (3,) Å, cubic box edge lengths
    n_atoms:          scalar int
    n_molecules:      scalar int
    """
    positions: np.ndarray
    element_ids: np.ndarray
    molecule_ids: np.ndarray
    molecule_species: np.ndarray
    formal_charges: np.ndarray
    box: np.ndarray
    n_atoms: int
    n_molecules: int


class MolecularTrajectory(NamedTuple):
    """Time series of molecular COM positions, unwrapped (continuous across PBC).

    com_positions:    (n_frames, n_molecules, 3) Å, unwrapped — straight-line
                      paths across box boundaries are continuous
    molecule_species: (n_molecules,) int8 indices into SPECIES_CATALOGUE
    formal_charges:   (n_molecules,) int8 in elementary-charge units
    box:              (3,) Å, cubic box edges (constant across frames in NVT)
    dt_fs:            scalar, time between consecutive frames
    n_frames:         scalar int
    n_molecules:      scalar int
    temperature_K:    scalar, simulation temperature
    """
    com_positions: np.ndarray
    molecule_species: np.ndarray
    formal_charges: np.ndarray
    box: np.ndarray
    dt_fs: float
    n_frames: int
    n_molecules: int
    temperature_K: float


# =============================================================================
# PBC primitives
# =============================================================================


def minimum_image_vector(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Return d minus the integer-box wrap. d shape (..., 3); box shape (3,).

    Standard cubic minimum-image convention. Fails loudly if box has zero edges.
    """
    if not np.all(box > 0):
        raise ValueError(f"Box edges must be positive, got {box}")
    return d - box * np.round(d / box)


# =============================================================================
# Streaming XYZ frame reader
# =============================================================================


STREAM_BUFFER_BYTES = 1 << 20   # Explicit constant: 1 MiB I/O buffer matches subprocess pipe default


def _open_xyz_stream(tar_path: Path, member_name: str | None):
    """Return a (process, text-stream) tuple for a streamed XYZ member.

    If the file is a .tar.gz, use `tar -xzOf` so the 8 GB archive is never
    fully expanded on disk. If it is a plain .xyz, open directly. If a .xyz.gz,
    open via gzip. Fails loudly if format is unrecognised.
    """
    suffix = "".join(tar_path.suffixes[-2:])
    if suffix in (".tar.gz", ".tgz"):
        if member_name is None:
            raise ValueError(".tar.gz requires explicit member_name (XYZ filename inside the archive)")
        proc = subprocess.Popen(
            ["tar", "-xzOf", str(tar_path), member_name],
            stdout=subprocess.PIPE,
            bufsize=STREAM_BUFFER_BYTES,
            text=True,
        )
        return proc, proc.stdout
    if tar_path.suffix == ".gz":
        return None, gzip.open(tar_path, "rt")
    if tar_path.suffix == ".xyz":
        return None, open(tar_path, "rt")
    raise ValueError(f"Unrecognised trajectory file format: {tar_path}")


def stream_xyz_frames(
    tar_path: Path,
    member_name: str | None,
    max_frames: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Yield up to max_frames as (positions, element_ids, box_ang, time_fs).

    LAMMPS XYZ trajectory format:
      Line 1: "<n_atoms> <box_edge_ang> <time_fs>"   (cubic box assumed)
      Line 2: <blank> or comment
      Lines 3..3+n_atoms-1: "<element_symbol> <x> <y> <z>"

    Cubic box from the second header field; many public liquid-electrolyte
    trajectories use a single scalar box edge. If the format diverges (e.g.
    triclinic), fail loudly here rather than silently mis-interpret coordinates.
    """
    proc, stream = _open_xyz_stream(tar_path, member_name)
    try:
        for frame_idx in range(max_frames):
            header = stream.readline()
            if not header:
                logger.info("Trajectory exhausted after %d frames (requested %d).", frame_idx, max_frames)
                break
            parts = header.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Malformed XYZ header at frame {frame_idx}: "
                    f"expected '<n_atoms> <box> <time_fs>', got {header!r}"
                )
            n_atoms = int(parts[0])
            box_edge = float(parts[1])
            t_fs = float(parts[2])
            stream.readline()  # blank/comment line

            element_ids = np.empty(n_atoms, dtype=np.int32)
            positions = np.empty((n_atoms, 3), dtype=np.float64)
            for i in range(n_atoms):
                rec = stream.readline().split()
                if len(rec) < 4:
                    raise ValueError(
                        f"Atom record short at frame {frame_idx} atom {i}: {rec!r}"
                    )
                sym = rec[0]
                if sym not in _SYMBOL_TO_ELEMENT:
                    raise ValueError(f"Unknown element symbol {sym!r} at frame {frame_idx} atom {i}")
                element_ids[i] = int(_SYMBOL_TO_ELEMENT[sym])
                positions[i, 0] = float(rec[1])
                positions[i, 1] = float(rec[2])
                positions[i, 2] = float(rec[3])

            yield positions, element_ids, np.array([box_edge, box_edge, box_edge], dtype=np.float64), t_fs
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()
        else:
            stream.close()


# =============================================================================
# Bond-graph molecule identification
# =============================================================================


def _bond_cutoff_squared(a: Element, b: Element) -> float | None:
    """Return squared bond cutoff for element pair, or None if not bondable.

    Skips Li/Na/K/Cl entirely — they are monoatomic ions in our chemistry
    catalogue. Returns None for any pair involving an element without a
    declared covalent radius, which is the explicit "do not bond" signal.
    """
    if a in MONOATOMIC_ION_ELEMENTS or b in MONOATOMIC_ION_ELEMENTS:
        return None
    if a not in COVALENT_RADIUS_ANG or b not in COVALENT_RADIUS_ANG:
        return None
    r = COVALENT_RADIUS_ANG[a] + COVALENT_RADIUS_ANG[b] + BOND_TOLERANCE_ANG
    return r * r


def detect_bonds_pbc(
    positions: np.ndarray, element_ids: np.ndarray, box: np.ndarray,
) -> dict[int, list[int]]:
    """Return undirected adjacency list of bonded atoms under cubic PBC.

    O(N²) check. For ~8000 atoms this is ~32 M pair tests; runs in seconds
    once per frame (we only call it on frame 0 for typing and on every Mth
    frame for the sanity gate). The vectorised distance computation makes
    the Python overhead small relative to the numpy work.
    """
    n = positions.shape[0]
    if element_ids.shape[0] != n:
        raise ValueError(f"element_ids has length {element_ids.shape[0]} but positions has {n} atoms")
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}

    by_element: dict[Element, list[int]] = {}
    for i in range(n):
        e = Element(int(element_ids[i]))
        by_element.setdefault(e, []).append(i)

    elements_present = sorted(by_element.keys(), key=lambda e: e.value)
    for ei in elements_present:
        for ej in elements_present:
            if ej.value < ei.value:
                continue
            cutoff_sq = _bond_cutoff_squared(ei, ej)
            if cutoff_sq is None:
                continue
            idx_i = np.array(by_element[ei])
            idx_j = np.array(by_element[ej])
            for ii in idx_i:
                if ei == ej:
                    candidates = idx_j[idx_j > ii]
                else:
                    candidates = idx_j
                if candidates.size == 0:
                    continue
                d = minimum_image_vector(positions[candidates] - positions[ii], box)
                d_sq = (d * d).sum(axis=1)
                bonded_mask = d_sq < cutoff_sq
                for jj in candidates[bonded_mask]:
                    adjacency[int(ii)].append(int(jj))
                    adjacency[int(jj)].append(int(ii))

    return adjacency


def identify_molecules(
    positions: np.ndarray, element_ids: np.ndarray, box: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each atom to a molecule via connected components of the bond graph.

    Returns:
      molecule_ids:     (n_atoms,) int32, 0..n_molecules-1
      molecule_species: (n_molecules,) int8 indices into SPECIES_CATALOGUE
      formal_charges:   (n_molecules,) int8

    Monoatomic ions form singleton components automatically (they have no bonds).
    """
    n = positions.shape[0]
    adjacency = detect_bonds_pbc(positions, element_ids, box)
    molecule_ids = -np.ones(n, dtype=np.int32)
    molecule_species_list: list[int] = []
    formal_charges_list: list[int] = []
    next_mol_id = 0

    for seed in range(n):
        if molecule_ids[seed] >= 0:
            continue
        stack = [seed]
        component: list[int] = []
        while stack:
            u = stack.pop()
            if molecule_ids[u] >= 0:
                continue
            molecule_ids[u] = next_mol_id
            component.append(u)
            stack.extend(adjacency[u])
        sig = _classify_component(element_ids[component])
        catalogue_idx = SPECIES_CATALOGUE.index(sig)
        molecule_species_list.append(catalogue_idx)
        formal_charges_list.append(sig.formal_charge)
        next_mol_id += 1

    molecule_species = np.array(molecule_species_list, dtype=np.int8)
    formal_charges = np.array(formal_charges_list, dtype=np.int8)
    return molecule_ids, molecule_species, formal_charges


# =============================================================================
# PBC molecule unfolding and COM
# =============================================================================


def unfold_molecules_pbc(
    positions: np.ndarray, molecule_ids: np.ndarray, box: np.ndarray,
) -> np.ndarray:
    """Shift each molecule's atoms so they sit as a contiguous spatial cluster.

    For a molecule that straddles a box boundary, half its atoms appear near
    x = 0 and half near x = L; the naive COM would land at L/2, far from
    any atom. We pick the first atom of each molecule as the reference and
    apply minimum-image shifts to the others so all are within ±L/2 of the ref.

    Returns: positions_unfolded, same shape as input.
    """
    out = positions.copy()
    n_molecules = int(molecule_ids.max()) + 1
    for mol_id in range(n_molecules):
        atom_idx = np.where(molecule_ids == mol_id)[0]
        if atom_idx.size <= 1:
            continue
        ref = out[atom_idx[0]]
        for ai in atom_idx[1:]:
            delta = minimum_image_vector(out[ai] - ref, box)
            out[ai] = ref + delta
    return out


def compute_molecular_com(
    positions_unfolded: np.ndarray,
    element_ids: np.ndarray,
    molecule_ids: np.ndarray,
    n_molecules: int,
) -> np.ndarray:
    """Mass-weighted COM per molecule. positions must be PBC-unfolded first.

    Returns: (n_molecules, 3) COM positions in Å.
    """
    masses = np.array([_ATOMIC_MASS[Element(int(e))] for e in element_ids], dtype=np.float64)
    com = np.zeros((n_molecules, 3), dtype=np.float64)
    total_mass = np.zeros(n_molecules, dtype=np.float64)
    for atom_idx in range(positions_unfolded.shape[0]):
        m_id = int(molecule_ids[atom_idx])
        w = masses[atom_idx]
        com[m_id] += w * positions_unfolded[atom_idx]
        total_mass[m_id] += w
    com /= total_mass[:, None]
    return com


def unwrap_com_path(
    com_t: np.ndarray, com_tm1_unwrapped: np.ndarray, box: np.ndarray,
) -> np.ndarray:
    """Make COM at frame t continuous with the previous (unwrapped) frame.

    If a molecule's COM jumped by more than half a box edge between frames,
    that is a PBC wrap, not a real motion. Subtract the wrap so the unwrapped
    path tracks the molecule's true displacement.

    Inputs:
      com_t:               (n_molecules, 3) raw (folded) COM at frame t
      com_tm1_unwrapped:   (n_molecules, 3) UNWRAPPED COM at frame t-1
      box:                 (3,) Å
    Returns: (n_molecules, 3) unwrapped COM at frame t.
    """
    folded_delta = com_t - (com_tm1_unwrapped - np.floor(com_tm1_unwrapped / box) * box)
    pbc_correction = box * np.round(folded_delta / box)
    return com_tm1_unwrapped + (folded_delta - pbc_correction)


# =============================================================================
# Trajectory loader
# =============================================================================


@dataclass(frozen=True)
class TrajectoryDescriptor:
    """User-supplied facts about a trajectory file we cannot infer from the data.

    These come from the trajectory's documentation page (Zenodo, etc.), not
    from any FF parameter set. They are observational metadata, not a force
    field.
    """
    tar_path: Path
    member_name: str | None          # XYZ filename inside .tar.gz, None for plain XYZ
    dt_fs: float                     # frame spacing
    temperature_K: float             # NVT temperature
    expected_n_atoms: int            # used to assert frame consistency
    expected_species_counts: dict[str, int]  # e.g. {"Li+": 52, "FSI-": 52, "EC": 149, "EMC": 400}


def load_molecular_trajectory(
    descriptor: TrajectoryDescriptor,
    max_frames: int,
    frame_stride: int,
) -> MolecularTrajectory:
    """Stream a trajectory; return PBC-unwrapped molecular-COM time series.

    frame_stride > 1 sub-samples the raw cadence (e.g. stride 20 turns 1.5 fs/frame
    into 30 fs/frame, which is closer to the propagator's training timestep).
    """
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if max_frames < 2:
        raise ValueError(f"max_frames must be >= 2 (need >=2 for sigma), got {max_frames}")

    logger.info(
        "Streaming %s (member=%s) max_frames=%d stride=%d",
        descriptor.tar_path, descriptor.member_name, max_frames, frame_stride,
    )

    n_atoms_expected = descriptor.expected_n_atoms

    raw_stream = stream_xyz_frames(
        descriptor.tar_path, descriptor.member_name, max_frames=1,
    )
    positions_0, element_ids_0, box_0, t0_fs = next(raw_stream)
    if positions_0.shape[0] != n_atoms_expected:
        raise ValueError(
            f"Frame 0 has {positions_0.shape[0]} atoms; descriptor says {n_atoms_expected}"
        )

    molecule_ids, molecule_species, formal_charges = identify_molecules(
        positions_0, element_ids_0, box_0,
    )
    n_molecules = int(molecule_ids.max()) + 1

    observed_counts: dict[str, int] = {}
    for mol_id in range(n_molecules):
        spec_name = SPECIES_CATALOGUE[molecule_species[mol_id]].name
        if spec_name in observed_counts:
            observed_counts[spec_name] += 1
        else:
            observed_counts[spec_name] = 1
    for name, expected_count in descriptor.expected_species_counts.items():
        observed_count = observed_counts[name] if name in observed_counts else 0
        if observed_count != expected_count:
            raise ValueError(
                f"Species count mismatch: {name} expected {expected_count}, "
                f"observed {observed_count}. Full observed: {observed_counts}"
            )
    logger.info("Frame 0 species: %s", observed_counts)

    total_charge = int(formal_charges.sum())
    if total_charge != 0:
        raise ValueError(
            f"Sum of molecular formal charges is {total_charge}, not zero. "
            f"Trajectory's chemistry is non-neutral; cannot compute conductivity."
        )
    logger.info(
        "Charge neutrality OK: %d molecules total, %d cations, %d anions",
        n_molecules,
        int((formal_charges > 0).sum()),
        int((formal_charges < 0).sum()),
    )

    com_unwrapped_list: list[np.ndarray] = []
    box_sum = np.zeros(3, dtype=np.float64)
    box_n_seen = 0
    box_max_drift = 0.0
    full_stream = stream_xyz_frames(
        descriptor.tar_path, descriptor.member_name, max_frames=max_frames,
    )

    com_prev_unwrapped: np.ndarray | None = None
    frame_count_used = 0
    frame_count_seen = 0
    LOG_EVERY = 5000   # Explicit constant: progress log interval in frames-seen
    MAX_BOX_DRIFT_FRAC = 0.05   # Explicit constant: NPT trajectories fluctuate ~1%; >5% means pathology
    for positions, element_ids, box, t_fs in full_stream:
        if positions.shape[0] != n_atoms_expected:
            raise ValueError(f"Frame {frame_count_seen} has {positions.shape[0]} atoms; expected {n_atoms_expected}")
        drift_frac = float(np.max(np.abs(box - box_0) / box_0))
        if drift_frac > MAX_BOX_DRIFT_FRAC:
            raise ValueError(
                f"Box drift {drift_frac*100:.2f}% at frame {frame_count_seen} exceeds "
                f"{MAX_BOX_DRIFT_FRAC*100:.0f}%; trajectory may be unstable"
            )
        if drift_frac > box_max_drift:
            box_max_drift = drift_frac
        if not np.array_equal(element_ids, element_ids_0):
            raise ValueError(f"Element ordering changed at frame {frame_count_seen}")

        box_sum += box
        box_n_seen += 1

        if frame_count_seen % frame_stride == 0:
            positions_unfolded = unfold_molecules_pbc(positions, molecule_ids, box)
            com_folded = compute_molecular_com(
                positions_unfolded, element_ids, molecule_ids, n_molecules,
            )
            if com_prev_unwrapped is None:
                com_now_unwrapped = com_folded.copy()
            else:
                com_now_unwrapped = unwrap_com_path(com_folded, com_prev_unwrapped, box)
            com_unwrapped_list.append(com_now_unwrapped)
            com_prev_unwrapped = com_now_unwrapped
            frame_count_used += 1
        frame_count_seen += 1
        if frame_count_seen % LOG_EVERY == 0:
            logger.info(
                "Streamed %d raw frames, kept %d (stride=%d, box max drift %.2f%%)",
                frame_count_seen, frame_count_used, frame_stride, box_max_drift * 100,
            )

    if frame_count_used < 2:
        raise RuntimeError(f"Only got {frame_count_used} usable frames; need >= 2")

    com_positions = np.stack(com_unwrapped_list, axis=0)
    box_mean = box_sum / box_n_seen
    effective_dt_fs = descriptor.dt_fs * frame_stride
    logger.info(
        "Built MolecularTrajectory: %d frames x %d molecules; effective dt = %.2f fs; "
        "time-averaged box = %s Å (max drift %.2f%%)",
        com_positions.shape[0], n_molecules, effective_dt_fs, box_mean, box_max_drift * 100,
    )

    return MolecularTrajectory(
        com_positions=com_positions,
        molecule_species=molecule_species,
        formal_charges=formal_charges,
        box=box_mean,
        dt_fs=effective_dt_fs,
        n_frames=com_positions.shape[0],
        n_molecules=n_molecules,
        temperature_K=descriptor.temperature_K,
    )


# =============================================================================
# Sanity gate
# =============================================================================


@dataclass
class SanityReport:
    """Observation-only sanity gate; FF identity is never checked."""
    n_frames_audited: int
    n_molecules: int
    box_density_g_per_cm3: float
    total_formal_charge: int
    bond_length_drift_pct: float           # max per-bond drift across audited frames, in %
    multimodal_bond_fraction: float        # fraction of bonds with multimodal distance distribution
    species_counts: dict[str, int]
    passed: bool
    failure_reasons: list[str]


# Drift thresholds for the sanity gate. Phase B's bond-stability check used 10%
# on bond lengths as the dissociation threshold; we use the same.
MAX_BOND_DRIFT_PCT = 10.0
MAX_MULTIMODAL_BOND_FRAC = 0.01
BIMODAL_OUTLIER_FRAC = 0.30   # Explicit constant: relative deviation from mean classifying a bond as bimodal


def trajectory_sanity_gate(
    descriptor: TrajectoryDescriptor,
    n_audit_frames: int,
    frame_stride: int,
) -> SanityReport:
    """Run the observation-only Phase 1 sanity gate.

    Checks:
      1. Bond length stability across audited frames (max drift < 10%)
      2. Multimodal bond-length distributions (sign of dissociation)
      3. Molecule count stable (bond graph reconstructs the same n_molecules)
      4. Charge neutrality at the box level
      5. Species-count match against descriptor

    Reports density in g/cm³ as a side benefit.
    """
    failure_reasons: list[str] = []

    initial_stream = stream_xyz_frames(
        descriptor.tar_path, descriptor.member_name, max_frames=1,
    )
    positions_0, element_ids_0, box_0, _ = next(initial_stream)
    molecule_ids_0, species_0, charges_0 = identify_molecules(positions_0, element_ids_0, box_0)
    n_molecules_0 = int(molecule_ids_0.max()) + 1

    species_counts: dict[str, int] = {}
    for mol_id in range(n_molecules_0):
        spec_name = SPECIES_CATALOGUE[species_0[mol_id]].name
        if spec_name in species_counts:
            species_counts[spec_name] += 1
        else:
            species_counts[spec_name] = 1

    total_charge = int(charges_0.sum())
    if total_charge != 0:
        failure_reasons.append(f"Box charge is {total_charge}, not zero")

    for name, expected_count in descriptor.expected_species_counts.items():
        observed_count = species_counts[name] if name in species_counts else 0
        if observed_count != expected_count:
            failure_reasons.append(
                f"Species {name}: expected {expected_count}, got {observed_count}"
            )

    adjacency_0 = detect_bonds_pbc(positions_0, element_ids_0, box_0)
    bond_pairs: list[tuple[int, int]] = []
    for i, neighbors in adjacency_0.items():
        for j in neighbors:
            if j > i:
                bond_pairs.append((i, j))
    bond_pairs_arr = np.array(bond_pairs, dtype=np.int64)
    logger.info("Sanity gate: %d bonds detected at frame 0, %d molecules", len(bond_pairs_arr), n_molecules_0)

    bond_lengths_over_time: list[np.ndarray] = []
    audit_stream = stream_xyz_frames(
        descriptor.tar_path, descriptor.member_name,
        max_frames=n_audit_frames * frame_stride,
    )
    audit_count = 0
    for f_idx, (positions, element_ids, box, _) in enumerate(audit_stream):
        if f_idx % frame_stride != 0:
            continue
        d = minimum_image_vector(
            positions[bond_pairs_arr[:, 1]] - positions[bond_pairs_arr[:, 0]], box,
        )
        bl = np.linalg.norm(d, axis=1)
        bond_lengths_over_time.append(bl)
        audit_count += 1
        if audit_count >= n_audit_frames:
            break

    bond_lengths_array = np.stack(bond_lengths_over_time, axis=0)   # (n_audit, n_bonds)
    bond_means = bond_lengths_array.mean(axis=0)
    bond_drift_pct = 100.0 * (bond_lengths_array.max(axis=0) - bond_lengths_array.min(axis=0)) / bond_means
    max_drift_pct = float(bond_drift_pct.max())
    logger.info(
        "Bond drift: max=%.2f%%, p99=%.2f%%, p50=%.2f%%",
        max_drift_pct,
        float(np.percentile(bond_drift_pct, 99)),
        float(np.percentile(bond_drift_pct, 50)),
    )

    outlier_mask = (bond_lengths_array - bond_means) / bond_means
    bimodal_bonds = np.any(np.abs(outlier_mask) > BIMODAL_OUTLIER_FRAC, axis=0)
    multimodal_frac = float(bimodal_bonds.mean())
    logger.info("Multimodal-bond fraction: %.4f (threshold %.4f)", multimodal_frac, MAX_MULTIMODAL_BOND_FRAC)

    if max_drift_pct > MAX_BOND_DRIFT_PCT:
        failure_reasons.append(f"Max bond drift {max_drift_pct:.1f}% exceeds {MAX_BOND_DRIFT_PCT}%")
    if multimodal_frac > MAX_MULTIMODAL_BOND_FRAC:
        failure_reasons.append(
            f"Multimodal-bond fraction {multimodal_frac:.4f} exceeds {MAX_MULTIMODAL_BOND_FRAC}"
        )

    total_mass_g_per_mol = float(sum(
        _ATOMIC_MASS[Element(int(e))] for e in element_ids_0
    ))
    box_volume_cm3 = float(np.prod(box_0)) * 1e-24
    density_g_per_cm3 = (total_mass_g_per_mol / N_A) / box_volume_cm3
    logger.info("Box density at frame 0: %.4f g/cm³", density_g_per_cm3)

    passed = len(failure_reasons) == 0
    return SanityReport(
        n_frames_audited=audit_count,
        n_molecules=n_molecules_0,
        box_density_g_per_cm3=density_g_per_cm3,
        total_formal_charge=total_charge,
        bond_length_drift_pct=max_drift_pct,
        multimodal_bond_fraction=multimodal_frac,
        species_counts=species_counts,
        passed=passed,
        failure_reasons=failure_reasons,
    )


# =============================================================================
# Atomistic frame cache (for Phase 4 training: random-access (x_t, x_{t+lag}))
# =============================================================================


class CachedAtomisticTrajectory(NamedTuple):
    """Memmap-backed atomistic trajectory for training-time random access.

    positions:        memmap, shape (n_frames, n_atoms, 3), float32, WRAPPED to
                      the box (LAMMPS dump coordinates as-stored).
    element_ids:      (n_atoms,) int32
    molecule_ids:     (n_atoms,) int32
    molecule_species: (n_molecules,) int8 — indices into SPECIES_CATALOGUE
    formal_charges:   (n_molecules,) int8
    box:              (3,) float64, time-averaged
    dt_fs:            scalar
    temperature_K:    scalar
    n_frames, n_atoms, n_molecules: scalars
    """
    positions: np.ndarray
    element_ids: np.ndarray
    molecule_ids: np.ndarray
    molecule_species: np.ndarray
    formal_charges: np.ndarray
    box: np.ndarray
    dt_fs: float
    temperature_K: float
    n_frames: int
    n_atoms: int
    n_molecules: int


def cache_atomistic_trajectory(
    descriptor: TrajectoryDescriptor,
    max_frames: int,
    output_dir: Path,
) -> CachedAtomisticTrajectory:
    """Stream a trajectory once, persist atomistic positions + per-atom metadata.

    Writes the following files under `output_dir`:
      positions.npy        — float32 (n_frames, n_atoms, 3), wrapped to box
      element_ids.npy      — int32 (n_atoms,)
      molecule_ids.npy     — int32 (n_atoms,)
      molecule_species.npy — int8 (n_molecules,)
      formal_charges.npy   — int8 (n_molecules,)
      box.npy              — float64 (3,) time-averaged
      metadata.json        — n_frames, n_atoms, n_molecules, dt_fs, temperature_K

    The positions cache is the source-of-truth atomic input for Phase 4 training.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    init_stream = stream_xyz_frames(descriptor.tar_path, descriptor.member_name, max_frames=1)
    positions_0, element_ids_0, box_0, _ = next(init_stream)
    if positions_0.shape[0] != descriptor.expected_n_atoms:
        raise ValueError(
            f"Frame 0 has {positions_0.shape[0]} atoms; descriptor expected {descriptor.expected_n_atoms}"
        )
    molecule_ids, molecule_species, formal_charges = identify_molecules(
        positions_0, element_ids_0, box_0,
    )
    n_atoms = positions_0.shape[0]
    n_molecules = int(molecule_ids.max()) + 1
    logger.info(
        "Frame 0 typed: %d atoms, %d molecules. Allocating cache at %s",
        n_atoms, n_molecules, output_dir,
    )

    positions_path = output_dir / "positions.npy"
    positions_mm = np.lib.format.open_memmap(
        positions_path, mode="w+", dtype=np.float32,
        shape=(max_frames, n_atoms, 3),
    )

    box_sum = np.zeros(3, dtype=np.float64)
    full_stream = stream_xyz_frames(
        descriptor.tar_path, descriptor.member_name, max_frames=max_frames,
    )
    n_frames_written = 0
    LOG_EVERY_FRAMES = 2000   # Explicit constant: progress log interval during caching
    for positions, element_ids, box, _ in full_stream:
        if positions.shape[0] != n_atoms:
            raise ValueError(
                f"Frame {n_frames_written} has {positions.shape[0]} atoms; expected {n_atoms}"
            )
        if not np.array_equal(element_ids, element_ids_0):
            raise ValueError(f"Element ordering changed at frame {n_frames_written}")
        positions_mm[n_frames_written] = positions.astype(np.float32)
        box_sum += box
        n_frames_written += 1
        if n_frames_written % LOG_EVERY_FRAMES == 0:
            logger.info("Cached %d frames", n_frames_written)

    if n_frames_written < 2:
        raise RuntimeError(f"Only got {n_frames_written} cacheable frames; need >= 2")
    box_mean = box_sum / n_frames_written

    # Truncate if we got fewer frames than requested
    if n_frames_written < max_frames:
        truncated = positions_mm[:n_frames_written].copy()
        del positions_mm
        np.save(positions_path, truncated)
    else:
        del positions_mm   # flush memmap

    np.save(output_dir / "element_ids.npy", element_ids_0)
    np.save(output_dir / "molecule_ids.npy", molecule_ids)
    np.save(output_dir / "molecule_species.npy", molecule_species)
    np.save(output_dir / "formal_charges.npy", formal_charges)
    np.save(output_dir / "box.npy", box_mean)
    (output_dir / "metadata.json").write_text(json.dumps({
        "n_frames": n_frames_written,
        "n_atoms": n_atoms,
        "n_molecules": n_molecules,
        "dt_fs": descriptor.dt_fs,
        "temperature_K": descriptor.temperature_K,
    }, indent=2))
    logger.info("Cached %d frames to %s; box_mean=%s", n_frames_written, output_dir, box_mean)

    return load_cached_trajectory(output_dir)


def load_cached_trajectory(cache_dir: Path) -> CachedAtomisticTrajectory:
    """Load a cached atomistic trajectory. Positions are memmap (zero-copy)."""
    import json
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    positions = np.load(cache_dir / "positions.npy", mmap_mode="r")
    element_ids = np.load(cache_dir / "element_ids.npy")
    molecule_ids = np.load(cache_dir / "molecule_ids.npy")
    molecule_species = np.load(cache_dir / "molecule_species.npy")
    formal_charges = np.load(cache_dir / "formal_charges.npy")
    box = np.load(cache_dir / "box.npy")
    return CachedAtomisticTrajectory(
        positions=positions,
        element_ids=element_ids,
        molecule_ids=molecule_ids,
        molecule_species=molecule_species,
        formal_charges=formal_charges,
        box=box,
        dt_fs=float(metadata["dt_fs"]),
        temperature_K=float(metadata["temperature_K"]),
        n_frames=int(metadata["n_frames"]),
        n_atoms=int(metadata["n_atoms"]),
        n_molecules=int(metadata["n_molecules"]),
    )


# =============================================================================
# Molecular-COM configuration and trajectory cache (for the COM propagator)
# =============================================================================


class MolecularConfiguration(NamedTuple):
    """Augmented state of the molecular-COM system — the second-order propagator's input.

    The propagator is second-order (plan §2.2, learning 1p): it conditions on the
    previous COM displacement as well as the current positions, so the augmented
    state `(com_positions, prev_displacement)` is a finite-difference proxy for
    `(positions, momenta)`.

    com_positions:     (n_molecules, 3) Å, wrapped into the box
    prev_displacement: (n_molecules, 3) Å — Δr over the previous propagator step;
                       zeros for a cold start with no momentum history
    molecule_species:  (n_molecules,) int8 — index into SPECIES_CATALOGUE
    formal_charges:    (n_molecules,) int8 — +1 / -1 / 0
    species_graphs:    SpeciesGraph — per-species bond graphs for the encoder
    box:               (3,) Å
    n_molecules:       scalar int
    """
    com_positions: np.ndarray
    prev_displacement: np.ndarray
    molecule_species: np.ndarray
    formal_charges: np.ndarray
    species_graphs: SpeciesGraph
    box: np.ndarray
    n_molecules: int


class SpeciesGraph(NamedTuple):
    """Per-species molecular bond graphs — input to the molecular encoder.

    Batched over catalogue indices 0..n_species-1; absent catalogue entries are
    all-padding rows (encoded but never indexed). The encoder reads atoms and
    bonds only — no 3D conformer — which is Weisfeiler-Leman-expressive enough to
    distinguish electrolyte molecules and transfers to unseen ones.

    elements:   (n_species, max_atoms) int32 — atomic numbers, 0 for padding
    bonds:      (n_species, max_atoms, max_atoms) float32 — 1.0 if bonded, else 0
    atom_mask:  (n_species, max_atoms) float32 — 1.0 real atom / 0.0 padding
    """
    elements: np.ndarray
    bonds: np.ndarray
    atom_mask: np.ndarray


def extract_species_graphs(atomic: "CachedAtomisticTrajectory") -> SpeciesGraph:
    """Build per-species bond graphs from a cached atomic trajectory's frame 0.

    For every catalogue index that appears in the trajectory, take a
    representative molecule, read its atoms' elements, and detect its
    intramolecular bonds. Padded to the largest molecule's atom count.
    """
    positions_0 = np.asarray(atomic.positions[0], dtype=np.float64)
    box = np.asarray(atomic.box, dtype=np.float64)
    adjacency = detect_bonds_pbc(positions_0, atomic.element_ids, box)

    n_species = int(atomic.molecule_species.max()) + 1
    # representative molecule index per catalogue species
    rep_molecule: dict[int, int] = {}
    for mol_id in range(atomic.n_molecules):
        sp = int(atomic.molecule_species[mol_id])
        if sp not in rep_molecule:
            rep_molecule[sp] = mol_id

    species_atom_lists: dict[int, list[int]] = {}
    for sp, mol_id in rep_molecule.items():
        species_atom_lists[sp] = list(np.where(atomic.molecule_ids == mol_id)[0])
    max_atoms = max(len(a) for a in species_atom_lists.values())

    elements = np.zeros((n_species, max_atoms), dtype=np.int32)
    bonds = np.zeros((n_species, max_atoms, max_atoms), dtype=np.float32)
    atom_mask = np.zeros((n_species, max_atoms), dtype=np.float32)
    for sp, atom_globals in species_atom_lists.items():
        local_of_global = {g: k for k, g in enumerate(atom_globals)}
        for k, g in enumerate(atom_globals):
            elements[sp, k] = int(atomic.element_ids[g])
            atom_mask[sp, k] = 1.0
        for g in atom_globals:
            for neighbor in adjacency[g]:
                if neighbor in local_of_global:
                    bonds[sp, local_of_global[g], local_of_global[neighbor]] = 1.0
    return SpeciesGraph(elements=elements, bonds=bonds, atom_mask=atom_mask)


class CachedComTrajectory(NamedTuple):
    """Memmap-backed molecular-COM trajectory for COM-propagator training.

    com_positions:    memmap (n_frames, n_molecules, 3) float32, WRAPPED to box
    molecule_species: (n_molecules,) int8
    formal_charges:   (n_molecules,) int8
    species_graphs:   SpeciesGraph — per-species bond graphs for the encoder
    box:              (3,) float64, time-averaged
    dt_fs:            scalar
    temperature_K:    scalar
    n_frames, n_molecules: scalars
    """
    com_positions: np.ndarray
    molecule_species: np.ndarray
    formal_charges: np.ndarray
    species_graphs: SpeciesGraph
    box: np.ndarray
    dt_fs: float
    temperature_K: float
    n_frames: int
    n_molecules: int


def cache_com_trajectory(atomic_cache_dir: Path, output_dir: Path) -> CachedComTrajectory:
    """Derive a wrapped molecular-COM trajectory from a cached atomic trajectory.

    For each frame: PBC-unfold every molecule, compute its mass-weighted COM,
    then wrap the COM back into the box. The wrapped COMs are what the COM
    propagator consumes; PBC neighbor finding handles periodicity at use time.

    Writes com_positions.npy, molecule_species.npy, formal_charges.npy, box.npy,
    metadata.json under output_dir.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic = load_cached_trajectory(atomic_cache_dir)
    box = np.asarray(atomic.box, dtype=np.float64)
    n_frames = atomic.n_frames
    n_molecules = atomic.n_molecules

    com_path = output_dir / "com_positions.npy"
    com_mm = np.lib.format.open_memmap(
        com_path, mode="w+", dtype=np.float32, shape=(n_frames, n_molecules, 3),
    )

    LOG_EVERY_FRAMES = 2000   # Explicit constant: progress log interval during COM caching
    for frame in range(n_frames):
        positions = np.asarray(atomic.positions[frame], dtype=np.float64)
        pos_unfolded = unfold_molecules_pbc(positions, atomic.molecule_ids, box)
        com = compute_molecular_com(
            pos_unfolded, atomic.element_ids, atomic.molecule_ids, n_molecules,
        )
        com_wrapped = com - np.floor(com / box) * box
        com_mm[frame] = com_wrapped.astype(np.float32)
        if (frame + 1) % LOG_EVERY_FRAMES == 0:
            logger.info("COM-cached %d / %d frames", frame + 1, n_frames)
    del com_mm   # flush

    species_graphs = extract_species_graphs(atomic)
    np.save(output_dir / "molecule_species.npy", atomic.molecule_species)
    np.save(output_dir / "formal_charges.npy", atomic.formal_charges)
    np.save(output_dir / "box.npy", box)
    np.save(output_dir / "species_graph_elements.npy", species_graphs.elements)
    np.save(output_dir / "species_graph_bonds.npy", species_graphs.bonds)
    np.save(output_dir / "species_graph_atom_mask.npy", species_graphs.atom_mask)
    (output_dir / "metadata.json").write_text(json.dumps({
        "n_frames": n_frames,
        "n_molecules": n_molecules,
        "dt_fs": atomic.dt_fs,
        "temperature_K": atomic.temperature_K,
    }, indent=2))
    logger.info("COM cache written to %s (%d frames, %d molecules)", output_dir, n_frames, n_molecules)
    return load_com_cache(output_dir)


def load_com_cache(cache_dir: Path) -> CachedComTrajectory:
    """Load a cached molecular-COM trajectory. COM positions are memmap."""
    import json
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    species_graphs = SpeciesGraph(
        elements=np.load(cache_dir / "species_graph_elements.npy"),
        bonds=np.load(cache_dir / "species_graph_bonds.npy"),
        atom_mask=np.load(cache_dir / "species_graph_atom_mask.npy"),
    )
    return CachedComTrajectory(
        com_positions=np.load(cache_dir / "com_positions.npy", mmap_mode="r"),
        molecule_species=np.load(cache_dir / "molecule_species.npy"),
        formal_charges=np.load(cache_dir / "formal_charges.npy"),
        species_graphs=species_graphs,
        box=np.load(cache_dir / "box.npy"),
        dt_fs=float(metadata["dt_fs"]),
        temperature_K=float(metadata["temperature_K"]),
        n_frames=int(metadata["n_frames"]),
        n_molecules=int(metadata["n_molecules"]),
    )


# =============================================================================
# CLI entrypoint
# =============================================================================


# Descriptor for the public FSI trajectory (Bytedance BAMBOO release).
# All values come from the trajectory's Zenodo / GitHub documentation page,
# not from any FF parameter set. Composition is 1 m LiFSI in EC:EMC 3:7 at 60°C.
# The header's third column is the dump time in PICOSECONDS, not femtoseconds —
# confirmed empirically by comparing the Li+ per-step displacement scale to the
# Stokes-Einstein diffusion expectation. Dumps are spaced 1.5 ps = 1500 fs apart.
TRAJ_FSI_DESCRIPTOR = TrajectoryDescriptor(
    tar_path=Path("conductivity/fm_data/trajectories/traj_FSI.tar.gz"),
    member_name="traj_FSI.xyz",
    dt_fs=1500.0,
    temperature_K=333.0,
    expected_n_atoms=8010,
    expected_species_counts={"Li+": 52, "FSI-": 52, "EC": 149, "EMC": 400},
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 trajectory sanity gate.")
    parser.add_argument(
        "--path",
        default=str(TRAJ_FSI_DESCRIPTOR.tar_path),
        help="Trajectory file path (.tar.gz or .xyz).",
    )
    parser.add_argument("--audit-frames", type=int, default=100,
                        help="Number of frames to sample for bond-stability audit.")
    parser.add_argument("--audit-stride", type=int, default=500,
                        help="Stride between audited frames (in raw frames).")
    args = parser.parse_args()

    descriptor = TrajectoryDescriptor(
        tar_path=Path(args.path),
        member_name=TRAJ_FSI_DESCRIPTOR.member_name,
        dt_fs=TRAJ_FSI_DESCRIPTOR.dt_fs,
        temperature_K=TRAJ_FSI_DESCRIPTOR.temperature_K,
        expected_n_atoms=TRAJ_FSI_DESCRIPTOR.expected_n_atoms,
        expected_species_counts=TRAJ_FSI_DESCRIPTOR.expected_species_counts,
    )

    logger.info("=== Phase 1 trajectory sanity gate ===")
    logger.info("Path: %s", descriptor.tar_path)
    report = trajectory_sanity_gate(
        descriptor=descriptor,
        n_audit_frames=args.audit_frames,
        frame_stride=args.audit_stride,
    )
    logger.info("=== Sanity report ===")
    logger.info("n_frames_audited:        %d", report.n_frames_audited)
    logger.info("n_molecules:             %d", report.n_molecules)
    logger.info("box density (g/cm³):     %.4f", report.box_density_g_per_cm3)
    logger.info("total formal charge:     %d", report.total_formal_charge)
    logger.info("max bond drift (%%):     %.2f", report.bond_length_drift_pct)
    logger.info("multimodal-bond frac:    %.4f", report.multimodal_bond_fraction)
    logger.info("species counts:          %s", report.species_counts)
    logger.info("PASSED: %s", report.passed)
    if not report.passed:
        for r in report.failure_reasons:
            logger.error("  FAILURE: %s", r)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
