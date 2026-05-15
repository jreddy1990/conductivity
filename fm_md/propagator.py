"""Phase 3: SE(3)-equivariant flow-matching propagator (molecular-COM resolution).

The propagator operates on molecular centers of mass, not individual atoms —
see plan accumulated learning 1n. Each node is one molecule, represented by its
COM. The flow-matching velocity field is `u_theta(dr_s, s; x_t)`:

  dr_s : (n_molecules, 3) per-molecule COM displacement at flow time s
  s    : scalar flow time in [0, 1]
  x_t  : MolecularConfiguration — the current molecular-COM point cloud

At inference the FM ODE integrates `dr` from s=0 (unit Gaussian noise) to s=1,
where the endpoint is the COM displacement applied to advance the box by one
propagator timestep. The network is SE(3)-equivariant by construction: scalar
features update only through rotation-invariant quantities (distances, vector
norms, vector inner products); vector features update only through
rotation-equivariant quantities (scalar * direction, scalar * vector). A global
rotation R of COM positions and dr_s rotates the output by R; a translation
leaves it unchanged because only relative displacements r_j - r_i enter.

Conditioning:
  - The current configuration x_t enters through COM positions (the graph),
    molecular species identity, and molecular formal charge.
  - The noisy displacement dr_s enters as vector-feature channel 0.
  - The flow time s enters as a sinusoidal embedding added to scalar features.

Entry: `python -m conductivity.fm_md.propagator` runs the equivariance and
non-collapse self-tests on a random molecular-COM box.
"""

from __future__ import annotations

import logging
import sys
from typing import NamedTuple

import control_framework.jax_m4_tuning  # noqa: F401  -- MUST precede any jax import (M4 env)

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.spatial import cKDTree

from conductivity.fm_md.atomistic_io import (
    SPECIES_CATALOGUE,
    MolecularConfiguration,
    SpeciesGraph,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Architecture constants (explicit, auditable — see plan §2.2)
# =============================================================================

N_SCALAR = 64               # scalar features per molecule
N_VECTOR = 8                # equivariant vector channels per molecule
N_RADIAL_BASIS = 8          # Bessel radial basis functions per edge
N_MP_LAYERS = 3             # message-passing layers
CUTOFF_ANG = 8.0            # COM-COM neighbor cutoff — captures the first molecular coordination shell
MAX_NEIGHBORS = 48          # padded neighbor count; COM neighbor count within 8 Å is well below this
FLOW_EMBED_DIM = 16         # sinusoidal flow-time embedding dimension
RADIAL_HIDDEN = 32          # hidden width of the radial filter MLP
N_ENCODER_LAYERS = 3        # bond-graph message-passing layers in the molecular encoder
N_ELEMENT_SLOTS = 20        # element embedding table size; indexed by atomic number (max Z = 19)

N_SPECIES = len(SPECIES_CATALOGUE)


# =============================================================================
# Radial basis and cutoff
# =============================================================================


def bessel_radial_basis(distances: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    """Sine Bessel radial basis (Klicpera 2020 / PaiNN).

    B_n(d) = sqrt(2/r_cut) * sin(n*pi*d/r_cut) / d  for n = 1..N_RADIAL_BASIS.
    Input shape (...,); output (..., N_RADIAL_BASIS). The 1/d factor is
    regularised by a small epsilon; real edges always have d > 0.
    """
    n = jnp.arange(1, N_RADIAL_BASIS + 1, dtype=jnp.float32)
    d_safe = distances[..., None] + 1e-6
    norm = jnp.sqrt(2.0 / r_cut)
    return norm * jnp.sin(n * jnp.pi * d_safe / r_cut) / d_safe


def cosine_cutoff(distances: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    """Smooth cosine cutoff: 1 at d=0, 0 at d>=r_cut, C1-continuous."""
    inside = 0.5 * (jnp.cos(jnp.pi * distances / r_cut) + 1.0)
    return jnp.where(distances < r_cut, inside, 0.0)


def flow_time_embedding(s: jnp.ndarray) -> jnp.ndarray:
    """Sinusoidal embedding of the scalar flow time s in [0, 1], shape (FLOW_EMBED_DIM,)."""
    half = FLOW_EMBED_DIM // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), half))
    angles = s * freqs
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


# =============================================================================
# Parameter initialisation
# =============================================================================


def _glorot(rng: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
    fan_in = shape[0]
    fan_out = shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(rng, shape, minval=-limit, maxval=limit)


def _silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


# =============================================================================
# Molecular encoder — bond-graph GNN producing transferable per-species features
# =============================================================================


def _init_encoder(rng: jax.Array) -> dict:
    """Initialise the molecular encoder (bond-graph GNN)."""
    keys = random.split(rng, N_ENCODER_LAYERS + 2)
    return {
        # element embedding indexed by atomic number
        "element_embed": _glorot(keys[0], (N_ELEMENT_SLOTS, N_SCALAR)) * 0.5,
        # one graph-convolution weight per message-passing layer
        "msg_w": [_glorot(keys[1 + i], (N_SCALAR, N_SCALAR)) for i in range(N_ENCODER_LAYERS)],
        # readout after sum-pooling atoms to a per-molecule feature
        "readout_w": _glorot(keys[-1], (N_SCALAR, N_SCALAR)),
    }


def molecular_encoder(enc_params: dict, species_graphs: SpeciesGraph) -> jnp.ndarray:
    """SE(3)-invariant bond-graph GNN: per-species atom graphs -> (n_species, N_SCALAR).

    Reads atoms (elements) and bonds only — Weisfeiler-Leman-expressive enough to
    distinguish electrolyte molecules. Because it operates on atoms and bonds,
    which transfer across molecules, a species absent from training is still
    encoded sensibly; this is what gives the propagator generalisation.
    """
    elements = jnp.asarray(species_graphs.elements)               # (S, A) int
    bonds = jnp.asarray(species_graphs.bonds, dtype=jnp.float32)   # (S, A, A)
    atom_mask = jnp.asarray(species_graphs.atom_mask, dtype=jnp.float32)  # (S, A)

    h = enc_params["element_embed"][elements]                     # (S, A, N_SCALAR)
    h = h * atom_mask[..., None]
    for msg_w in enc_params["msg_w"]:
        # aggregate features from bonded neighbours, residual update, re-mask padding
        msg = jnp.einsum("sab,sbd->sad", bonds, _silu(h @ msg_w))
        h = (h + msg) * atom_mask[..., None]
    mol_feat = jnp.sum(h, axis=1)                                 # masked sum-pool -> (S, N_SCALAR)
    return mol_feat @ enc_params["readout_w"]


def _init_layer(rng: jax.Array) -> dict:
    """Initialise one PaiNN message + update block."""
    keys = random.split(rng, 9)
    return {
        "radial_w1": _glorot(keys[0], (N_RADIAL_BASIS, RADIAL_HIDDEN)),
        "radial_b1": jnp.zeros((RADIAL_HIDDEN,)),
        "radial_w2": _glorot(keys[1], (RADIAL_HIDDEN, 3 * N_SCALAR)),
        "radial_b2": jnp.zeros((3 * N_SCALAR,)),
        "phi_w1": _glorot(keys[2], (N_SCALAR, N_SCALAR)),
        "phi_b1": jnp.zeros((N_SCALAR,)),
        "phi_w2": _glorot(keys[3], (N_SCALAR, 3 * N_SCALAR)),
        "phi_b2": jnp.zeros((3 * N_SCALAR,)),
        "msg_vec_w": _glorot(keys[4], (N_SCALAR, N_VECTOR)),
        "upd_U": _glorot(keys[5], (N_VECTOR, N_VECTOR)),
        "upd_V": _glorot(keys[6], (N_VECTOR, N_VECTOR)),
        "upd_w1": _glorot(keys[7], (N_SCALAR + N_VECTOR, N_SCALAR)),
        "upd_b1": jnp.zeros((N_SCALAR,)),
        "upd_w2": _glorot(keys[8], (N_SCALAR, 2 * N_SCALAR + N_VECTOR)),
        "upd_b2": jnp.zeros((2 * N_SCALAR + N_VECTOR,)),
    }


def init_propagator_params(rng: jax.Array) -> dict:
    """Initialise the full propagator parameter tree."""
    keys = random.split(rng, 6)
    return {
        # molecular encoder: bond-graph GNN -> transferable per-species features
        "encoder": _init_encoder(keys[0]),
        # molecular formal charge (-1 / 0 / +1) -> scalar features
        "charge_w": _glorot(keys[1], (1, N_SCALAR)),
        # flow-time embedding -> scalar features
        "flowtime_w": _glorot(keys[2], (FLOW_EMBED_DIM, N_SCALAR)),
        # input mixing after combining species + charge + flowtime
        "input_w": _glorot(keys[3], (N_SCALAR, N_SCALAR)),
        "input_b": jnp.zeros((N_SCALAR,)),
        "layers": [_init_layer(k) for k in random.split(keys[4], N_MP_LAYERS)],
        # readout: vector channels -> single output 3-vector (per molecule)
        "readout_w": _glorot(keys[5], (N_VECTOR, 1)) * 0.1,
    }


def count_parameters(params: dict) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# =============================================================================
# Neighbor list (scipy cKDTree, PBC-aware, O(N log N))
# =============================================================================


class NeighborArrays(NamedTuple):
    """Fixed-shape neighbor list for the jitted propagator core.

    neighbors:      (n_molecules, MAX_NEIGHBORS) int32 — neighbor node indices
    displacements:  (n_molecules, MAX_NEIGHBORS, 3) float32 — r_j - r_i (min image)
    distances:      (n_molecules, MAX_NEIGHBORS) float32
    neighbor_mask:  (n_molecules, MAX_NEIGHBORS) float32 — 1.0 real / 0.0 padding
    """
    neighbors: np.ndarray
    displacements: np.ndarray
    distances: np.ndarray
    neighbor_mask: np.ndarray


def build_neighbor_arrays(positions: np.ndarray, box: np.ndarray) -> NeighborArrays:
    """PBC neighbor list via scipy cKDTree. O(N log N), runs outside JAX.

    The k = MAX_NEIGHBORS nearest neighbors are found per node (cKDTree handles
    periodic images via `boxsize`); neighbors beyond CUTOFF_ANG are masked out.
    Raises if any node has MAX_NEIGHBORS or more real neighbors within the
    cutoff, since that would silently truncate the graph.
    """
    pos = np.asarray(positions, dtype=np.float64)
    L = np.asarray(box, dtype=np.float64)
    n_nodes = pos.shape[0]
    pos_wrapped = pos - np.floor(pos / L) * L
    # The floor-wrap lands every coordinate in [0, L) up to floating-point error.
    # A gross violation means the wrap math itself is broken — fail loudly.
    WRAP_TOL_ANG = 1e-6
    if np.any(pos_wrapped < -WRAP_TOL_ANG) or np.any(pos_wrapped > L + WRAP_TOL_ANG):
        raise RuntimeError(
            f"PBC wrap produced out-of-box coordinates: range "
            f"[{pos_wrapped.min():.6f}, {pos_wrapped.max():.6f}], box {L}"
        )
    # cKDTree's `boxsize` requires strictly 0 <= x < L. A coordinate that
    # rounding placed at exactly L is, under PBC, identical to 0 — fold it
    # there; a tiny negative from rounding folds up by one box. This is the
    # exact periodic wrap completed at the boundary, not a clamp.
    pos_wrapped = np.where(pos_wrapped >= L, pos_wrapped - L, pos_wrapped)
    pos_wrapped = np.where(pos_wrapped < 0.0, pos_wrapped + L, pos_wrapped)

    tree = cKDTree(pos_wrapped, boxsize=L)
    dist, idx = tree.query(pos_wrapped, k=MAX_NEIGHBORS + 1)
    dist = np.asarray(dist)[:, 1:]
    idx = np.asarray(idx)[:, 1:]

    valid = (idx < n_nodes) & (dist < CUTOFF_ANG)
    idx_clean = np.where(valid, idx, 0).astype(np.int32)

    disp = pos_wrapped[idx_clean] - pos_wrapped[:, None, :]
    disp = disp - L * np.round(disp / L)
    dist_clean = np.linalg.norm(disp, axis=-1)

    mask = valid.astype(np.float32)
    disp = (disp * mask[..., None]).astype(np.float32)
    dist_clean = (dist_clean * mask).astype(np.float32)

    overflow = int(np.max(np.sum(valid, axis=1)))
    if overflow >= MAX_NEIGHBORS:
        raise RuntimeError(
            f"Neighbor overflow: a node has >= {MAX_NEIGHBORS} neighbors within "
            f"{CUTOFF_ANG} Å. Increase MAX_NEIGHBORS."
        )
    return NeighborArrays(
        neighbors=idx_clean,
        displacements=disp,
        distances=dist_clean,
        neighbor_mask=mask,
    )


# =============================================================================
# Message passing
# =============================================================================


def _message_and_update(
    layer: dict,
    scalar: jnp.ndarray,        # (n_nodes, N_SCALAR)
    vector: jnp.ndarray,        # (n_nodes, N_VECTOR, 3)
    neighbors: jnp.ndarray,     # (n_nodes, MAX_NEIGHBORS) int
    displacements: jnp.ndarray, # (n_nodes, MAX_NEIGHBORS, 3)  r_j - r_i
    distances: jnp.ndarray,     # (n_nodes, MAX_NEIGHBORS)
    neighbor_mask: jnp.ndarray, # (n_nodes, MAX_NEIGHBORS) 1.0 real / 0.0 pad
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One PaiNN message + update block. Returns (scalar, vector) residual-updated."""
    # ---- message block ----
    radial = bessel_radial_basis(distances, CUTOFF_ANG)
    fcut = cosine_cutoff(distances, CUTOFF_ANG)
    rad_h = _silu(radial @ layer["radial_w1"] + layer["radial_b1"])
    rad_gates = (rad_h @ layer["radial_w2"] + layer["radial_b2"]) * fcut[..., None]

    phi_h = _silu(scalar @ layer["phi_w1"] + layer["phi_b1"])
    phi = phi_h @ layer["phi_w2"] + layer["phi_b2"]
    phi_j = phi[neighbors]

    combined = phi_j * rad_gates
    split_ss, split_sv, split_vv = jnp.split(combined, 3, axis=-1)

    delta_scalar = jnp.sum(split_ss * neighbor_mask[..., None], axis=1)

    d_safe = distances[..., None] + 1e-6
    directions = displacements / d_safe
    sv_chan = split_sv @ layer["msg_vec_w"]
    vv_chan = split_vv @ layer["msg_vec_w"]
    vec_from_dir = sv_chan[..., None] * directions[:, :, None, :]
    vector_j = vector[neighbors]
    vec_from_vec = vv_chan[..., None] * vector_j
    delta_vector = jnp.sum(
        (vec_from_dir + vec_from_vec) * neighbor_mask[:, :, None, None], axis=1,
    )

    scalar = scalar + delta_scalar
    vector = vector + delta_vector

    # ---- update block ----
    Uv = jnp.einsum("avx,vw->awx", vector, layer["upd_U"])
    Vv = jnp.einsum("avx,vw->awx", vector, layer["upd_V"])
    Vv_norm = jnp.sqrt(jnp.sum(Vv * Vv, axis=-1) + 1e-8)

    upd_in = jnp.concatenate([scalar, Vv_norm], axis=-1)
    upd_h = _silu(upd_in @ layer["upd_w1"] + layer["upd_b1"])
    upd = upd_h @ layer["upd_w2"] + layer["upd_b2"]
    a_ss, a_sv, a_vv = jnp.split(upd, [N_SCALAR, 2 * N_SCALAR], axis=-1)

    inner_UV = jnp.sum(Uv * Vv, axis=-1)
    delta_scalar_2 = a_ss + a_sv * (inner_UV @ layer["msg_vec_w"].T / N_VECTOR)
    delta_vector_2 = a_vv[..., None] * Uv

    scalar = scalar + delta_scalar_2
    vector = vector + delta_vector_2
    return scalar, vector


# =============================================================================
# Forward pass
# =============================================================================


def propagator_core(
    params: dict,
    dr_s: jnp.ndarray,
    s: jnp.ndarray,
    dr_prev: jnp.ndarray,
    molecule_species: jnp.ndarray,
    formal_charges: jnp.ndarray,
    species_graphs: SpeciesGraph,
    nl: NeighborArrays,
) -> jnp.ndarray:
    """Jittable message-passing core. Takes a precomputed neighbor list.

    `dr_prev` is the previous-step COM displacement — the momentum state that
    makes the propagator second-order (plan §2.2, learning 1p). `species_graphs`
    feeds the molecular encoder, whose per-species features replace the old
    species-identity lookup (learning 1r). Training jits this directly and
    reuses one neighbor list per frame; the contracted `propagator_velocity`
    wraps it with on-the-fly neighbor finding.
    """
    n_nodes = dr_s.shape[0]

    # ---- initial scalar features ----
    # molecular encoder: bond graphs -> per-species features, indexed to molecules
    species_features = molecular_encoder(params["encoder"], species_graphs)  # (S, N_SCALAR)
    species_feat = species_features[molecule_species]              # (M, N_SCALAR)
    charge_feat = formal_charges[:, None] @ params["charge_w"]     # (M, N_SCALAR)
    s_embed = flow_time_embedding(jnp.asarray(s, dtype=jnp.float32))
    flow_feat = (s_embed @ params["flowtime_w"])[None, :]
    scalar = species_feat + charge_feat + flow_feat
    scalar = _silu(scalar @ params["input_w"] + params["input_b"])

    # ---- initial vector features: channel 0 = dr_s (flow state),
    #      channel 1 = dr_prev (momentum state), others zero ----
    vector = jnp.zeros((n_nodes, N_VECTOR, 3), dtype=jnp.float32)
    vector = vector.at[:, 0, :].set(jnp.asarray(dr_s, dtype=jnp.float32))
    vector = vector.at[:, 1, :].set(jnp.asarray(dr_prev, dtype=jnp.float32))

    # ---- message passing ----
    for layer in params["layers"]:
        scalar, vector = _message_and_update(
            layer, scalar, vector,
            nl.neighbors, nl.displacements, nl.distances, nl.neighbor_mask,
        )

    # ---- readout: vector channels -> single output 3-vector ----
    return jnp.einsum("avx,vw->awx", vector, params["readout_w"])[:, 0, :]


def propagator_velocity(
    params: dict,
    dr_s: jnp.ndarray,
    s: jnp.ndarray,
    x_t: MolecularConfiguration,
) -> jnp.ndarray:
    """SE(3)-equivariant flow-matching velocity field (contracted entry point).

    Inputs:
      params : propagator parameter tree from init_propagator_params
      dr_s   : (n_molecules, 3) per-molecule COM displacement at flow time s
      s      : scalar flow time in [0, 1]
      x_t    : MolecularConfiguration — current molecular-COM box, including
               the previous-step displacement that makes the propagator second-order
    Returns:
      (n_molecules, 3) velocity field — d(dr)/ds at flow time s.
    """
    nl = build_neighbor_arrays(np.asarray(x_t.com_positions), np.asarray(x_t.box))
    return propagator_core(
        params, dr_s, s,
        jnp.asarray(x_t.prev_displacement, dtype=jnp.float32),
        jnp.asarray(x_t.molecule_species),
        jnp.asarray(x_t.formal_charges, dtype=jnp.float32),
        x_t.species_graphs,
        NeighborArrays(
            neighbors=jnp.asarray(nl.neighbors),
            displacements=jnp.asarray(nl.displacements),
            distances=jnp.asarray(nl.distances),
            neighbor_mask=jnp.asarray(nl.neighbor_mask),
        ),
    )


# =============================================================================
# Self-tests: equivariance and non-collapse
# =============================================================================


def _random_com_box(rng: jax.Array, n_molecules: int, box_edge: float) -> MolecularConfiguration:
    """Build a random MolecularConfiguration for self-testing (not physical)."""
    k1, k2 = random.split(rng)
    com_positions = random.uniform(k1, (n_molecules, 3), minval=0.0, maxval=box_edge)
    prev_displacement = random.normal(k2, (n_molecules, 3)) * 0.5
    # Alternate cation / anion species and charges across nodes.
    molecule_species = np.where(
        np.arange(n_molecules) % 2 == 0, 0, 3,
    ).astype(np.int8)
    formal_charges = np.where(np.arange(n_molecules) % 2 == 0, 1, -1).astype(np.int8)
    # Minimal species graphs: 4 catalogue slots, species 0 a 1-atom ion,
    # species 3 a 3-atom molecule with two bonds. Not physical — exercises the
    # encoder's padding/masking and message passing.
    max_atoms = 3
    elements = np.zeros((4, max_atoms), dtype=np.int32)
    bonds = np.zeros((4, max_atoms, max_atoms), dtype=np.float32)
    atom_mask = np.zeros((4, max_atoms), dtype=np.float32)
    elements[0, 0] = 3                                  # species 0: single Li atom
    atom_mask[0, 0] = 1.0
    elements[3, :] = [7, 16, 16]                        # species 3: N-S-S chain
    atom_mask[3, :] = 1.0
    bonds[3, 0, 1] = bonds[3, 1, 0] = 1.0
    bonds[3, 1, 2] = bonds[3, 2, 1] = 1.0
    species_graphs = SpeciesGraph(elements=elements, bonds=bonds, atom_mask=atom_mask)
    return MolecularConfiguration(
        com_positions=np.asarray(com_positions),
        prev_displacement=np.asarray(prev_displacement),
        molecule_species=molecule_species,
        formal_charges=formal_charges,
        species_graphs=species_graphs,
        box=np.array([box_edge, box_edge, box_edge], dtype=np.float64),
        n_molecules=n_molecules,
    )


def run_self_tests() -> int:
    """Equivariance and non-collapse self-tests. Returns 0 on pass, 1 on fail."""
    rng = random.PRNGKey(0)
    k_params, k_box, k_dr, k_rot = random.split(rng, 4)
    n_molecules = 200
    box_edge = 40.0

    params = init_propagator_params(k_params)
    logger.info("Propagator parameters: %d", count_parameters(params))

    x_t = _random_com_box(k_box, n_molecules, box_edge)
    dr_s = random.normal(k_dr, (n_molecules, 3)) * 0.3
    s = 0.4

    vel = propagator_velocity(params, dr_s, s, x_t)
    logger.info("velocity output shape %s, std %.5f", vel.shape, float(jnp.std(vel)))

    failures: list[str] = []

    if float(jnp.std(vel)) < 1e-4:
        failures.append(f"output std {float(jnp.std(vel)):.2e} — propagator collapsed to a constant")

    # --- rotation equivariance ---
    A = random.normal(k_rot, (3, 3))
    Q, R = jnp.linalg.qr(A)
    Q = Q * jnp.sign(jnp.diag(R))
    Q = Q * jnp.sign(jnp.linalg.det(Q))

    big_box = box_edge * 3.0
    base_pos = np.asarray(x_t.com_positions) - np.asarray(x_t.com_positions).min(axis=0) + 1.0
    x_t_base = x_t._replace(
        com_positions=base_pos,
        box=np.array([big_box, big_box, big_box], dtype=np.float64),
    )
    rotated_pos = np.asarray(x_t.com_positions) @ np.asarray(Q).T
    rotated_pos = rotated_pos - rotated_pos.min(axis=0) + 1.0
    # prev_displacement is a vector feature — it rotates with the system.
    x_t_rot = x_t._replace(
        com_positions=rotated_pos,
        prev_displacement=np.asarray(x_t.prev_displacement) @ np.asarray(Q).T,
        box=np.array([big_box, big_box, big_box], dtype=np.float64),
    )
    dr_rot = np.asarray(dr_s) @ np.asarray(Q).T

    vel_base = propagator_velocity(params, jnp.asarray(np.asarray(dr_s)), s, x_t_base)
    vel_rot = propagator_velocity(params, jnp.asarray(dr_rot), s, x_t_rot)
    vel_base_then_rot = np.asarray(vel_base) @ np.asarray(Q).T
    rot_err = float(jnp.max(jnp.abs(vel_rot - vel_base_then_rot)))
    logger.info("rotation equivariance max error: %.2e", rot_err)
    if rot_err > 1e-3:
        failures.append(f"rotation equivariance error {rot_err:.2e} exceeds 1e-3")

    # --- translation invariance ---
    shift = np.array([3.1, -2.4, 5.7])
    x_t_shift = x_t_base._replace(com_positions=np.asarray(x_t_base.com_positions) + shift)
    vel_shift = propagator_velocity(params, jnp.asarray(np.asarray(dr_s)), s, x_t_shift)
    trans_err = float(jnp.max(jnp.abs(vel_shift - vel_base)))
    logger.info("translation invariance max error: %.2e", trans_err)
    if trans_err > 1e-3:
        failures.append(f"translation invariance error {trans_err:.2e} exceeds 1e-3")

    if failures:
        for f in failures:
            logger.error("SELF-TEST FAIL: %s", f)
        return 1
    logger.info("All propagator self-tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(run_self_tests())
