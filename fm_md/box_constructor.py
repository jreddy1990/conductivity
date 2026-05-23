"""Composition-to-initial-box constructor (plan §2.1).

Takes a composition specification `(species_list, mole_fractions, temperature_K,
target_n_atoms, seed)` and produces a `MolecularConfiguration` the propagator
can roll forward. This is the ENTRY POINT for inference on arbitrary
compositions — the piece that turns "1m LiFSI in EC:EMC 3:7" into a
ready-to-run propagator state.

Design (plan §2.1):
  - Per-species atom graphs come from RDKit on SMILES strings looked up in
    `SPECIES_SMILES`. No FF-based atom typing, no atomistic equilibration —
    the propagator handles relaxation.
  - Per-species count from mole fractions × target_n_atoms / atoms_per_species,
    rounded with charge-conservation.
  - Box edge from total mass / DEFAULT_DENSITY_G_CM3. Density is a single
    explicit constant documented at module level; varying it by composition is
    a future improvement once we want it composition-aware.
  - Cold-start `prev_displacement = 0` for the momentum channel; the chain
    relaxes onto pi_theta within ACF_ROLL_BURN steps in any case.

For known species (Li+, FSI-, EC, EMC) this reproduces what the FSI cache
holds; for unseen species (DMC, DEC, FEC, VC, DME, ...) RDKit builds the
graph from SMILES so the molecular encoder can produce a transferable
embedding (plan §1r).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from rdkit import Chem

from conductivity.fm_md.atomistic_io import (
    _ATOMIC_MASS,
    Element,
    MolecularConfiguration,
    SPECIES_CATALOGUE,
    SpeciesGraph,
)

ATOMIC_MASSES = _ATOMIC_MASS  # alias for readability in this module

logger = logging.getLogger(__name__)


# SMILES for every entry in SPECIES_CATALOGUE. Names match SPECIES_CATALOGUE[i].name
# exactly. Atom counts MUST match SPECIES_CATALOGUE[i].element_counts (asserted
# at startup) -- a mismatch means the SMILES is wrong, not the catalogue.
SPECIES_SMILES: dict[str, str] = {
    "Li+":  "[Li+]",
    "Na+":  "[Na+]",
    "K+":   "[K+]",
    # FSI- = bis(fluorosulfonyl)imide = [N-](SO2F)2
    "FSI-": "O=S(=O)(F)[N-]S(=O)(=O)F",
    # PF6- = hexafluorophosphate
    "PF6-": "F[P-](F)(F)(F)(F)F",
    # EC = ethylene carbonate (5-ring with -OC(=O)O- and -CH2CH2-)
    "EC":   "O=C1OCCO1",
    # EMC = ethyl methyl carbonate
    "EMC":  "COC(=O)OCC",
    # DMC = dimethyl carbonate
    "DMC":  "COC(=O)OC",
    # DEC = diethyl carbonate
    "DEC":  "CCOC(=O)OCC",
    # PC = propylene carbonate (5-ring, methyl on one ring C)
    "PC":   "CC1OC(=O)OC1",
    # FEC = fluoroethylene carbonate (one F on ring C)
    "FEC":  "O=C1OCC(F)O1",
    # VC = vinylene carbonate (5-ring with C=C)
    "VC":   "O=C1OC=CO1",
    # DME = 1,2-dimethoxyethane
    "DME":  "COCCOC",
    # TFSI- = bis(trifluoromethanesulfonyl)imide
    "TFSI-": "O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F",
    # ClO4- = perchlorate
    "ClO4-": "[O-]Cl(=O)(=O)=O",
    # CF3SO3- = triflate (lithium triflate anion)
    "CF3SO3-": "[O-]S(=O)(=O)C(F)(F)F",
}


# Explicit constant: representative liquid-electrolyte density. The Bytedance
# 1 m LiFSI in EC:EMC 3:7 FSI trajectory at 333 K has 8010 atoms in a 45.356 Å
# cubic box. Its mass density computed from atomistic_io.ATOMIC_MASSES and the
# species counts is 1.249 g/cm^3. We use this single nominal value across all
# compositions; composition-specific densities are a future improvement.
DEFAULT_DENSITY_G_CM3 = 1.249

# Avogadro * (cm/Å)^3 conversion: amu/Å^3 -> g/cm^3.
# 1 amu/Å^3 = (1/N_A) g / (1e-24 cm^3) = 1.66054 g/cm^3.
AMU_PER_ANG3_TO_G_PER_CM3 = 1.66053906892

# RMS per-coordinate COM displacement at one propagator step (dt=1.5 ps) on the
# Bytedance FSI training trajectory, measured by train.py measure_displacement_scale
# at run start. This is the CFM prior std the model was trained with. We
# initialise prev_displacement to N(0, this^2 I) so the propagator's momentum
# channel starts in-distribution; the actual model's sigma_prior is loaded from
# the checkpoint at inference time, but the box construction layer does not see
# the checkpoint, so we use the training value as the initialisation scale and
# rely on burn-in to renormalise (plan F3, 2026-05-21 OOD audit).
FSI_TRAINING_SIGMA_PRIOR_ANG = 0.8048


def _smiles_to_atom_graph(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse SMILES → (elements, bonds). Includes explicit hydrogens.

    Returns:
      elements: (n_atoms,) int32 — atomic numbers
      bonds:    (n_atoms, n_atoms) float32 — symmetric 0/1 bond matrix
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()
    elements = np.zeros(n_atoms, dtype=np.int32)
    for i, atom in enumerate(mol.GetAtoms()):
        elements[i] = int(atom.GetAtomicNum())
    bonds = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds[i, j] = bonds[j, i] = 1.0
    return elements, bonds


def _validate_species_signature(species_name: str, elements: np.ndarray) -> None:
    """Cross-check the RDKit atom list against SPECIES_CATALOGUE element_counts."""
    sig = next((s for s in SPECIES_CATALOGUE if s.name == species_name), None)
    if sig is None:
        raise ValueError(
            f"Species {species_name!r} is not in SPECIES_CATALOGUE. "
            f"Add a SpeciesSignature entry first."
        )
    actual: dict[Element, int] = {}
    for z in elements:
        e = Element(int(z))
        if e in actual:
            actual[e] += 1
        else:
            actual[e] = 1
    if actual != sig.element_counts:
        raise ValueError(
            f"SMILES for {species_name} gives elements {actual} but catalogue says "
            f"{sig.element_counts}. The SMILES or the catalogue is wrong."
        )


def _build_species_graphs(species_subset: list[str]) -> tuple[SpeciesGraph, dict[str, int]]:
    """Build a single SpeciesGraph (batched over species) and an
    {species_name: catalogue_index} mapping. The catalogue index is the position
    of that species inside the returned batch (NOT the SPECIES_CATALOGUE index)
    because the propagator's molecule_species field indexes into the batched
    SpeciesGraph, not into SPECIES_CATALOGUE."""
    per_species_graphs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in species_subset:
        if name not in SPECIES_SMILES:
            raise ValueError(f"No SMILES recorded for species {name!r}. Add one to SPECIES_SMILES.")
        elements, bonds = _smiles_to_atom_graph(SPECIES_SMILES[name])
        _validate_species_signature(name, elements)
        per_species_graphs[name] = (elements, bonds)
        logger.info("species %s: %d atoms via RDKit + SMILES %r",
                    name, elements.size, SPECIES_SMILES[name])

    max_atoms = max(e.size for e, _ in per_species_graphs.values())
    n_species = len(species_subset)
    elements_batch = np.zeros((n_species, max_atoms), dtype=np.int32)
    bonds_batch = np.zeros((n_species, max_atoms, max_atoms), dtype=np.float32)
    atom_mask_batch = np.zeros((n_species, max_atoms), dtype=np.float32)
    name_to_idx: dict[str, int] = {}
    for i, name in enumerate(species_subset):
        elements, bonds = per_species_graphs[name]
        n_a = elements.size
        elements_batch[i, :n_a] = elements
        bonds_batch[i, :n_a, :n_a] = bonds
        atom_mask_batch[i, :n_a] = 1.0
        name_to_idx[name] = i
    return SpeciesGraph(elements=elements_batch, bonds=bonds_batch, atom_mask=atom_mask_batch), name_to_idx


def _species_mass_amu(species_name: str) -> float:
    sig = next(s for s in SPECIES_CATALOGUE if s.name == species_name)
    return float(sum(ATOMIC_MASSES[e] * c for e, c in sig.element_counts.items()))


def _species_atom_count(species_name: str) -> int:
    sig = next(s for s in SPECIES_CATALOGUE if s.name == species_name)
    return int(sum(sig.element_counts.values()))


def _species_formal_charge(species_name: str) -> int:
    sig = next(s for s in SPECIES_CATALOGUE if s.name == species_name)
    return int(sig.formal_charge)


def build_initial_configuration(
    species_list: Sequence[str],
    mole_fractions: Sequence[float],
    temperature_K: float,
    target_n_atoms: int,
    seed: int,
) -> MolecularConfiguration:
    """Produce a MolecularConfiguration from a composition specification.

    Inputs:
      species_list   : species names from SPECIES_CATALOGUE (e.g. ["Li+", "FSI-", "EC", "EMC"])
      mole_fractions : same length as species_list; will be normalised
      temperature_K  : simulation temperature (passed through for metadata; not
                       used here for kinetic initialisation -- the propagator's
                       CFM prior is dimensionally-correct)
      target_n_atoms : approximate total atom count for the box
      seed           : RNG seed for the random COM-position layout

    Returns a MolecularConfiguration with cold-start prev_displacement=0. The
    propagator's ACF_ROLL_BURN burn-in handles momentum relaxation onto pi_theta.

    Box edge is derived from total mass / DEFAULT_DENSITY_G_CM3.

    Net charge MUST be 0; this is enforced by adjusting the highest-charge
    species count when float-rounding leaves a residual. Failure to enforce
    raises ValueError.
    """
    species_list = list(species_list)
    mole_fractions = np.asarray(mole_fractions, dtype=np.float64)
    if mole_fractions.shape != (len(species_list),):
        raise ValueError(f"mole_fractions has shape {mole_fractions.shape}; expected {(len(species_list),)}")
    if not np.all(mole_fractions >= 0):
        raise ValueError(f"mole_fractions must be non-negative; got {mole_fractions}")
    mole_fractions = mole_fractions / mole_fractions.sum()

    # Per-species atoms per molecule, masses, charges
    atoms_per_mol = np.array([_species_atom_count(s) for s in species_list], dtype=np.int64)
    mass_per_mol = np.array([_species_mass_amu(s) for s in species_list], dtype=np.float64)
    charge_per_mol = np.array([_species_formal_charge(s) for s in species_list], dtype=np.int64)

    # Choose total number of molecules so total_atoms ≈ target_n_atoms with the
    # right mole-fraction split. Solve: n_mol_total * sum_i (x_i * atoms_per_i) ≈ target_n_atoms.
    avg_atoms_per_mol = float((mole_fractions * atoms_per_mol).sum())
    n_mol_total = max(int(round(target_n_atoms / avg_atoms_per_mol)), len(species_list))
    n_mol_per_species = np.maximum(np.round(mole_fractions * n_mol_total).astype(np.int64), 0)
    # Enforce charge neutrality by adjusting counts of charged species pairwise.
    net_charge = int((n_mol_per_species * charge_per_mol).sum())
    if net_charge != 0:
        pos_idx = [i for i, q in enumerate(charge_per_mol) if q > 0]
        neg_idx = [i for i, q in enumerate(charge_per_mol) if q < 0]
        if net_charge > 0 and pos_idx:
            n_mol_per_species[pos_idx[0]] -= net_charge
        elif net_charge < 0 and neg_idx:
            n_mol_per_species[neg_idx[0]] -= (-net_charge)
    net_charge = int((n_mol_per_species * charge_per_mol).sum())
    if net_charge != 0:
        raise ValueError(
            f"could not enforce charge neutrality; net charge = {net_charge} "
            f"with counts {dict(zip(species_list, n_mol_per_species.tolist()))}"
        )

    n_molecules = int(n_mol_per_species.sum())
    if n_molecules <= 0:
        raise ValueError(f"target_n_atoms={target_n_atoms} too small to make any molecules with these species")

    # Box edge from mass / density. mass [amu] -> volume [Å^3] -> edge.
    total_mass_amu = float((n_mol_per_species * mass_per_mol).sum())
    volume_ang3 = total_mass_amu * AMU_PER_ANG3_TO_G_PER_CM3 / DEFAULT_DENSITY_G_CM3
    box_edge = float(volume_ang3 ** (1.0 / 3.0))
    box = np.array([box_edge, box_edge, box_edge], dtype=np.float64)

    # Build per-species graphs (RDKit) once for the species we actually use.
    species_graph, name_to_batch_idx = _build_species_graphs(species_list)

    # Assign species and charges to molecules in a contiguous block layout.
    molecule_species = np.zeros(n_molecules, dtype=np.int8)
    formal_charges = np.zeros(n_molecules, dtype=np.int8)
    offset = 0
    for name, n_mol in zip(species_list, n_mol_per_species.tolist()):
        if n_mol == 0:
            continue
        molecule_species[offset : offset + n_mol] = name_to_batch_idx[name]
        formal_charges[offset : offset + n_mol] = _species_formal_charge(name)
        offset += n_mol

    # Random COM positions in the box (uniform; the propagator handles relaxation).
    rng = np.random.default_rng(seed)
    com_positions = rng.uniform(0.0, box_edge, size=(n_molecules, 3)).astype(np.float64)

    # Scale-matched cold-start momentum: dr_prev ~ N(0, sigma_prior^2 I) per
    # molecule. The propagator's channel-1 dr_prev input was trained with
    # realistic MD displacements of magnitude ~sigma_prior, so zero momentum
    # is out-of-distribution for the network. Drawing from the CFM prior
    # itself keeps the cold-start state on-manifold and shortens the burn-in
    # transient (plan F3, root-cause attribution for 22-125x BAMBOO OOD
    # 2026-05-21). DEFAULT_DENSITY_G_CM3 is in the same module; sigma_prior
    # for the box-construction-time momentum is the same training-data
    # measurement the model carries in its checkpoint, but at this layer we
    # do not have the model yet, so use the FSI training value as the
    # initialisation scale and rely on the burn-in to renormalise.
    rng2 = np.random.default_rng(seed + 1)
    prev_displacement = (FSI_TRAINING_SIGMA_PRIOR_ANG * rng2.standard_normal(
        size=(n_molecules, 3))).astype(np.float64)

    logger.info(
        "box: %d atoms, %d molecules, box_edge=%.3f Å, density=%.3f g/cm^3, T=%.1f K",
        int((n_mol_per_species * atoms_per_mol).sum()), n_molecules, box_edge,
        DEFAULT_DENSITY_G_CM3, temperature_K,
    )
    logger.info("composition: %s",
                ", ".join(f"{s}={int(n)}" for s, n in zip(species_list, n_mol_per_species.tolist())))

    return MolecularConfiguration(
        com_positions=com_positions,
        prev_displacement=prev_displacement,
        molecule_species=molecule_species,
        formal_charges=formal_charges,
        species_graphs=species_graph,
        box=box,
        n_molecules=n_molecules,
        temperature_K=float(temperature_K),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    # Smoke test: 1m LiFSI in EC:EMC 3:7 at 333 K, ~2000 atoms.
    # From the BAMBOO FSI box: Li+:FSI-:EC:EMC = 52:52:149:400 (mol counts).
    cfg = build_initial_configuration(
        species_list=["Li+", "FSI-", "EC", "EMC"],
        mole_fractions=[52/653, 52/653, 149/653, 400/653],
        temperature_K=333.0,
        target_n_atoms=2000,
        seed=0,
    )
    print("smoke OK — box:", cfg.box, "n_molecules:", cfg.n_molecules)
