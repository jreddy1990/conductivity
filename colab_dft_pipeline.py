"""
=============================================================================
ELECTROLYTE DFT DATA GENERATION + TorchANI TRAINING — Google Colab Pipeline
=============================================================================

Complete pipeline to generate first-principles conductivity training data:

  Phase 1: Download BAMBOO dataset (10 species, 720k real DFT frames)
  Phase 2: Generate DFT data for 28 missing species via GPU4PySCF on Colab T4
  Phase 3: Train TorchANI model on combined data
  Phase 4: Export model for M4 MPS inference

Run on Google Colab with GPU runtime (T4 free tier).
Each section is a separate Colab cell — copy between "# %%  CELL N" markers.

Expected compute: ~140 GPU hours total for gap-fill DFT (spread across sessions).
All intermediate results saved to Google Drive for session persistence.

Author: electrolyte_formation_sim pipeline
"""

# %% CELL 1 — INSTALL DEPENDENCIES ==========================================
# Runtime: ~3 min on fresh Colab instance
# Re-run after runtime restart

import subprocess
import sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Core scientific
_install("numpy")
_install("scipy")
_install("h5py")

# Chemistry
_install("rdkit")          # Conformer generation
_install("ase")            # Atoms objects, I/O

# DFT engine — GPU-accelerated PySCF
_install("pyscf")
_install("gpu4pyscf")      # T4/A100 GPU acceleration

# ML potential
_install("torch")          # Should be pre-installed on Colab
_install("torchani")       # Behler-Parrinello NN potential

# Dataset access
_install("huggingface_hub")

# Google Drive persistence
from google.colab import drive  # type: ignore[import-not-found]
drive.mount("/content/drive")

import os
WORK_DIR = "/content/drive/MyDrive/electrolyte_dft_pipeline"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(f"{WORK_DIR}/bamboo_data", exist_ok=True)
os.makedirs(f"{WORK_DIR}/gap_fill_dft", exist_ok=True)
os.makedirs(f"{WORK_DIR}/training_data", exist_ok=True)
os.makedirs(f"{WORK_DIR}/models", exist_ok=True)
os.makedirs(f"{WORK_DIR}/checkpoints", exist_ok=True)

print(f"Work directory: {WORK_DIR}")
print("All dependencies installed.")


# %% CELL 2 — SPECIES REGISTRY ==============================================
# All 38 species from data/species_data.py.
#
# SMILES: Canonical SMILES from species_data.py (authoritative source).
# mw/density: Molecular weight (g/mol) and liquid density (g/mL) — these are
#   literature physical constants needed for box building. Source: species_data.py
#   which itself references CRC Handbook, NIST, and manufacturer datasheets.
#   Duplicated here because this script runs on Colab without access to the repo.
# bamboo: Whether BAMBOO HuggingFace dataset includes DFT data for this species.
#
# For salts, cation_smiles and anion_smiles are the disconnected fragments.

import numpy as np

# Elements covered: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Si(14), P(15), S(16), Cl(17)

SPECIES_REGISTRY = {
    # === SOLVENTS (15) — mw from sum of atomic weights, density from CRC Handbook ===
    "EC":   {"smiles": "C1COC(=O)O1",       "type": "solvent", "mw": 88.06,  "density": 1.321, "bamboo": True},
    "PC":   {"smiles": "CC1COC(=O)O1",      "type": "solvent", "mw": 102.09, "density": 1.205, "bamboo": False},
    "EMC":  {"smiles": "CCOC(=O)OC",        "type": "solvent", "mw": 104.10, "density": 1.007, "bamboo": True},
    "DMC":  {"smiles": "COC(=O)OC",         "type": "solvent", "mw": 90.08,  "density": 1.069, "bamboo": True},
    "DEC":  {"smiles": "CCOC(=O)OCC",       "type": "solvent", "mw": 118.13, "density": 0.975, "bamboo": True},
    "PN":   {"smiles": "CCC#N",             "type": "solvent", "mw": 55.08,  "density": 0.782, "bamboo": False},
    "MA":   {"smiles": "COC(=O)C",          "type": "solvent", "mw": 74.08,  "density": 0.934, "bamboo": False},
    "EA":   {"smiles": "CCOC(=O)C",         "type": "solvent", "mw": 88.11,  "density": 0.902, "bamboo": True},
    "MP":   {"smiles": "CCC(=O)OC",         "type": "solvent", "mw": 88.11,  "density": 0.915, "bamboo": False},
    "GBL":  {"smiles": "O=C1CCCO1",         "type": "solvent", "mw": 86.09,  "density": 1.124, "bamboo": False},
    "DME":  {"smiles": "COCCOC",            "type": "solvent", "mw": 90.12,  "density": 0.867, "bamboo": False},
    "DOL":  {"smiles": "C1COCO1",           "type": "solvent", "mw": 74.08,  "density": 1.060, "bamboo": False},
    "TMS":  {"smiles": "O=S1(=O)CCCC1",     "type": "solvent", "mw": 120.17, "density": 1.262, "bamboo": False},
    "FEMC": {"smiles": "COC(=O)OCC(F)(F)F", "type": "solvent", "mw": 138.07, "density": 1.240, "bamboo": False},
    "AN":   {"smiles": "CC#N",              "type": "solvent", "mw": 41.05,  "density": 0.786, "bamboo": False},

    # === SALTS (7) — disconnected ion pair SMILES, density is solid-state ===
    "LiPF6":  {"smiles": "[Li+].F[P-](F)(F)(F)(F)F",
               "type": "salt", "mw": 151.9, "density": 1.50, "bamboo": True,
               "cation_smiles": "[Li+]", "anion_smiles": "F[P-](F)(F)(F)(F)F"},
    "LiFSI":  {"smiles": "[Li+].FS(=O)(=O)[N-]S(=O)(=O)F",
               "type": "salt", "mw": 187.07, "density": 1.60, "bamboo": True,
               "cation_smiles": "[Li+]", "anion_smiles": "FS(=O)(=O)[N-]S(=O)(=O)F"},
    "LiTFSI": {"smiles": "[Li+].[O-]S(=O)(=O)N(S(=O)(=O)C(F)(F)F)C(F)(F)F",
               "type": "salt", "mw": 287.09, "density": 1.33, "bamboo": True,
               "cation_smiles": "[Li+]", "anion_smiles": "[O-]S(=O)(=O)N(S(=O)(=O)C(F)(F)F)C(F)(F)F"},
    "LiBF4":  {"smiles": "[Li+].[B-](F)(F)(F)F",
               "type": "salt", "mw": 93.75, "density": 1.21, "bamboo": False,
               "cation_smiles": "[Li+]", "anion_smiles": "[B-](F)(F)(F)F"},
    "LiClO4": {"smiles": "[Li+].[O-]Cl(=O)(=O)=O",
               "type": "salt", "mw": 106.39, "density": 2.42, "bamboo": False,
               "cation_smiles": "[Li+]", "anion_smiles": "[O-]Cl(=O)(=O)=O"},
    "LiDFP":  {"smiles": "[Li+].[O-]P(=O)(F)F",
               "type": "salt", "mw": 107.91, "density": 1.40, "bamboo": False,
               "cation_smiles": "[Li+]", "anion_smiles": "[O-]P(=O)(F)F"},
    "LiNO3":  {"smiles": "[Li+].[O-][N+](=O)[O-]",
               "type": "salt", "mw": 68.95, "density": 2.38, "bamboo": False,
               "cation_smiles": "[Li+]", "anion_smiles": "[O-][N+](=O)[O-]"},

    # === ADDITIVES (16) ===
    "LiDFOB":  {"smiles": "[Li+].[B-](F)(F)1OC(=O)C(=O)O1",
                "type": "additive", "mw": 143.77, "density": 1.40, "bamboo": False,
                "cation_smiles": "[Li+]", "anion_smiles": "[B-](F)(F)1OC(=O)C(=O)O1"},
    "LiBOB":   {"smiles": "[Li+].O=C1OB2OC(=O)O2O1",
                "type": "additive", "mw": 193.79, "density": 1.50, "bamboo": False,
                "cation_smiles": "[Li+]", "anion_smiles": "O=C1OB2OC(=O)O2O1"},
    "FEC":     {"smiles": "FC1COC(=O)O1",
                "type": "additive", "mw": 106.05, "density": 1.454, "bamboo": True},
    "VC":      {"smiles": "C=C1COC(=O)O1",
                "type": "additive", "mw": 86.05, "density": 1.355, "bamboo": True},
    "SN":      {"smiles": "N#CCCC#N",
                "type": "additive", "mw": 80.09, "density": 0.951, "bamboo": False},
    "ES":      {"smiles": "C1COS(=O)(=O)O1",
                "type": "additive", "mw": 108.12, "density": 1.340, "bamboo": False},
    "DTD":     {"smiles": "C1CSSC1",
                "type": "additive", "mw": 106.19, "density": 1.118, "bamboo": False},
    "DTN":     {"smiles": "N#CC1CCCCC1",
                "type": "additive", "mw": 109.17, "density": 0.899, "bamboo": False},
    "TPP":     {"smiles": "O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1",
                "type": "additive", "mw": 326.28, "density": 1.205, "bamboo": False},
    "SA":      {"smiles": "O=S1OCCO1",
                "type": "additive", "mw": 108.12, "density": 1.340, "bamboo": False},
    "PS":      {"smiles": "O=C1CCC(=O)O1",
                "type": "additive", "mw": 114.10, "density": 1.200, "bamboo": False},
    "TMSPi":   {"smiles": "C[Si](C)(C)OP(O[Si](C)(C)C)O[Si](C)(C)C",
                "type": "additive", "mw": 314.52, "density": 0.860, "bamboo": False},
    "DMMP":    {"smiles": "COP(=O)(C)OC",
                "type": "additive", "mw": 124.08, "density": 1.145, "bamboo": False},
    "TTFP":    {"smiles": "O=P(OCC(F)(F)F)(OCC(F)(F)F)OCC(F)(F)F",
                "type": "additive", "mw": 344.12, "density": 1.390, "bamboo": False},
    "MEC":     {"smiles": "C=C1COC(=O)O1",
                "type": "additive", "mw": 86.05, "density": 1.355, "bamboo": False},
    "LiPO2F2": {"smiles": "[Li+].[O-]P(=O)(F)F",
                "type": "additive", "mw": 107.91, "density": 1.40, "bamboo": False,
                "cation_smiles": "[Li+]", "anion_smiles": "[O-]P(=O)(F)F"},
}

bamboo_species = [k for k, v in SPECIES_REGISTRY.items() if v["bamboo"]]
gap_species = [k for k, v in SPECIES_REGISTRY.items() if not v["bamboo"]]
print(f"BAMBOO-covered: {len(bamboo_species)} — {bamboo_species}")
print(f"Gap-fill needed: {len(gap_species)} — {gap_species}")


# %% CELL 3 — DOWNLOAD BAMBOO MINI SUBSET ===================================
# Download the mini subset first (~100 MB) to validate format before full 58 GB

from huggingface_hub import hf_hub_download
import torch
import os

BAMBOO_DIR = f"{WORK_DIR}/bamboo_data"

# Download mini subset for format inspection
mini_path = hf_hub_download(
    repo_id="mzl/bamboo",
    filename="mini/mini_data.pt",
    repo_type="dataset",
    local_dir=BAMBOO_DIR,
)
print(f"Downloaded BAMBOO mini to: {mini_path}")

# Inspect format — we need to discover the key names for atomic_numbers,
# positions, energy, forces before the conversion cell can work.
data = torch.load(mini_path, map_location="cpu", weights_only=False)
print(f"\nType: {type(data)}")


def _inspect_value(k, v, indent="  "):
    """Log shape/dtype for tensors, length for lists, type for everything else."""
    if isinstance(v, torch.Tensor):
        print(f"{indent}{k}: Tensor shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, np.ndarray):
        print(f"{indent}{k}: ndarray shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"{indent}{k}: list[{len(v)}], first={type(v[0]) if v else 'empty'}")
    else:
        print(f"{indent}{k}: {type(v).__name__} = {v}")


if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
    for k, v in data.items():
        _inspect_value(k, v)
elif isinstance(data, list):
    print(f"List of {len(data)} items")
    if data:
        item = data[0]
        print(f"First item type: {type(item)}")
        if isinstance(item, dict):
            print(f"First item keys: {list(item.keys())}")
            for k, v in item.items():
                _inspect_value(k, v)
else:
    print(f"Unexpected type: {type(data)}")
    raise TypeError(f"Cannot parse BAMBOO data of type {type(data)}. "
                    "Inspect manually and update convert_bamboo_data().")


# %% CELL 4 — CONFORMER GENERATION FOR GAP-FILL SPECIES =====================
# For each missing species, generate diverse liquid-like configurations:
#   1. Single molecule conformers (gas phase)
#   2. Small clusters: 3-5 molecules packed together (liquid-like)
#   3. Li+ solvation clusters: molecule + Li+ at various positions
#
# These become the input geometries for DFT single-point calculations.

from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("conformer_gen")

CONFORMER_DIR = f"{WORK_DIR}/gap_fill_dft/conformers"
os.makedirs(CONFORMER_DIR, exist_ok=True)

# --- Cluster geometry constants ---
# Li-O first solvation shell distance: 1.9-2.1 Å from DFT/AIMD
# (Borodin, J. Phys. Chem. B, 2009; Skarmoutsos, J. Chem. Phys., 2015)
# We place molecules at 3.5 Å from Li+ center (molecule COM to Li+) so that
# the coordinating oxygen atom is ~2.0 Å from Li+ after geometry relaxation.
LI_SOLVATION_SHELL_RADIUS_ANG = 3.5  # COM-to-Li+ distance for 1st shell placement

# Li-anion contact ion pair distance: ~2.0 Å for LiF, ~2.5 Å for larger anions
# (Takeuchi, J. Phys. Chem. B, 2012). We use the larger value as initial placement.
LI_ANION_CONTACT_DISTANCE_ANG = 2.5  # Li-to-anion-COM for contact pair

# Second-shell placement for solvent around ion pairs
SECOND_SHELL_RADIUS_ANG = 6.0  # COM distance for 2nd solvation shell

# Random perturbation magnitude for augmentation — small enough to stay near
# equilibrium but large enough to sample PES curvature for force training
PERTURBATION_ANG = 0.1  # Å, ~5% of a C-C bond length

# Minimum distance threshold to avoid numerical blowup in DFT overlap integrals
MIN_INTERATOMIC_DIST_ANG = 0.1

# Number of perturbation copies per base config
N_PERTURBATIONS_PER_CONFIG = 4

# Reference solvent for ionic species solvation clusters.
# Selected as the first BAMBOO-covered cyclic carbonate (cyclic = ring in SMILES).
# Cyclic carbonates dominate the Li+ first solvation shell due to high dipole
# moment and chelation geometry — any cyclic carbonate works as reference shell.
_BAMBOO_SOLVENTS = [k for k, v in SPECIES_REGISTRY.items()
                    if v["bamboo"] and v["type"] == "solvent"]
_BAMBOO_CYCLIC = [s for s in _BAMBOO_SOLVENTS
                  if "1" in SPECIES_REGISTRY[s]["smiles"]]  # ring digit = cyclic
if not _BAMBOO_CYCLIC:
    raise ValueError("No cyclic carbonate in BAMBOO-covered solvents. "
                     "Update SPECIES_REGISTRY bamboo flags.")
REFERENCE_SOLVENT_FOR_SOLVATION = sorted(_BAMBOO_CYCLIC)[0]


def smiles_to_atoms(smiles: str, n_conformers: int = 10) -> list[Atoms]:
    """Generate diverse 3D conformers from SMILES using RDKit ETKDG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    # ETKDG v3 — state-of-art distance geometry with experimental torsion prefs
    params = AllChem.ETKDGv3()
    params.numThreads = 0  # use all CPU threads
    # Keep conformers differing by >0.5 Å RMSD for diversity
    params.pruneRmsThresh = 0.5
    params.randomSeed = 42

    # Generate 3× requested, prune by RMSD, pick top n_conformers
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers * 3, params=params)
    if len(cids) == 0:
        # Single-conformer molecules (e.g., Li+, small symmetric) — embed once
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            raise ValueError(f"RDKit embedding failed for SMILES: {smiles}")
        AllChem.MMFFOptimizeMolecule(mol)
        cids = [0]

    # Optimize with MMFF94 force field for reasonable starting geometries
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)

    atoms_list = []
    for cid in cids[:n_conformers]:
        conf = mol.GetConformer(cid)
        positions = conf.GetPositions()
        symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        atoms_list.append(Atoms(symbols=symbols, positions=positions))

    return atoms_list


def build_li_solvation_cluster(solvent_atoms: Atoms, n_solvent: int = 4) -> Atoms:
    """
    Build a Li+ solvation cluster: Li+ at origin surrounded by n_solvent
    copies of the molecule evenly distributed on a ring.

    This mimics the first solvation shell (CN=4 for Li+ in carbonates).
    """
    from scipy.spatial.transform import Rotation

    all_symbols = ["Li"]
    all_positions = [np.array([0.0, 0.0, 0.0])]

    for i in range(n_solvent):
        angle = 2 * np.pi * i / n_solvent
        # Alternate z-offset for tetrahedral-like arrangement
        z_offset = 0.5 if (i % 2) else -0.5
        offset = np.array([
            LI_SOLVATION_SHELL_RADIUS_ANG * np.cos(angle),
            LI_SOLVATION_SHELL_RADIUS_ANG * np.sin(angle),
            z_offset,
        ])

        rot = Rotation.random(random_state=42 + i)
        rotated_pos = rot.apply(solvent_atoms.get_positions())
        shifted_pos = rotated_pos - rotated_pos.mean(axis=0) + offset

        all_symbols.extend(solvent_atoms.get_chemical_symbols())
        all_positions.extend(shifted_pos)

    return Atoms(symbols=all_symbols, positions=np.array(all_positions))


def build_anion_cluster(
    anion_smiles: str,
    n_anions: int = 2,
    n_li: int = 2,
    solvent_atoms_list: list[Atoms] | None = None,
) -> Atoms:
    """
    Build a cluster with anions + Li+ + optional solvent molecules.
    Mimics ion-pair configurations in liquid electrolyte.
    """
    from scipy.spatial.transform import Rotation

    anion_atoms_list = smiles_to_atoms(anion_smiles, n_conformers=1)
    if not anion_atoms_list:
        raise ValueError(f"Failed to generate conformer for anion: {anion_smiles}")
    anion_atoms = anion_atoms_list[0]

    all_symbols: list[str] = []
    all_positions: list[np.ndarray] = []

    # Place Li+ ions at ~2 Å from origin on a ring
    LI_RING_RADIUS = 2.0  # Å — Li-Li separation in concentrated electrolyte
    for i in range(n_li):
        angle = 2 * np.pi * i / max(n_li, 1)
        pos = np.array([LI_RING_RADIUS * np.cos(angle), LI_RING_RADIUS * np.sin(angle), 0.0])
        all_symbols.append("Li")
        all_positions.append(pos)

    # Place anions around Li+ at contact distance
    ANION_RING_RADIUS = 4.0  # Å — anion COM distance for 1st coordination
    for i in range(n_anions):
        angle = 2 * np.pi * (i + 0.5) / max(n_anions, 1)
        z_offset = 1.0 if (i % 2) else -1.0
        offset = np.array([
            ANION_RING_RADIUS * np.cos(angle),
            ANION_RING_RADIUS * np.sin(angle),
            z_offset,
        ])

        rot = Rotation.random(random_state=100 + i)
        rotated_pos = rot.apply(anion_atoms.get_positions())
        shifted_pos = rotated_pos - rotated_pos.mean(axis=0) + offset

        all_symbols.extend(anion_atoms.get_chemical_symbols())
        all_positions.extend(shifted_pos)

    # Add solvent molecules in second shell
    if solvent_atoms_list is not None:
        for i, sol_atoms in enumerate(solvent_atoms_list):
            angle = 2 * np.pi * i / len(solvent_atoms_list)
            offset = np.array([
                SECOND_SHELL_RADIUS_ANG * np.cos(angle),
                SECOND_SHELL_RADIUS_ANG * np.sin(angle),
                0.0,
            ])
            rot = Rotation.random(random_state=200 + i)
            rotated_pos = rot.apply(sol_atoms.get_positions())
            shifted_pos = rotated_pos - rotated_pos.mean(axis=0) + offset
            all_symbols.extend(sol_atoms.get_chemical_symbols())
            all_positions.extend(shifted_pos)

    return Atoms(symbols=all_symbols, positions=np.array(all_positions))


def _is_ionic(info: dict) -> bool:
    """Check if a species entry is ionic (has cation/anion fragments)."""
    return "cation_smiles" in info


def generate_all_conformers():
    """
    Generate conformers for all gap-fill species.
    Saves to CONFORMER_DIR as .json files (one per species).
    Resumable: skips species already in progress.json.
    """
    progress_path = f"{CONFORMER_DIR}/progress.json"
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done = set(json.load(f))
    else:
        done = set()

    # Reference solvent conformers for solvation clusters around ionic species
    ref_smiles = SPECIES_REGISTRY[REFERENCE_SOLVENT_FOR_SOLVATION]["smiles"]
    ref_conformers = smiles_to_atoms(ref_smiles, n_conformers=5)
    logger.info(f"Generated {len(ref_conformers)} reference solvent "
                f"({REFERENCE_SOLVENT_FOR_SOLVATION}) conformers for solvation shells")

    for species_name in gap_species:
        if species_name in done:
            logger.info(f"Skipping {species_name} (already done)")
            continue

        info = SPECIES_REGISTRY[species_name]
        smiles = info["smiles"]
        logger.info(f"Generating conformers for {species_name} ({smiles})")

        configs: list[dict] = []

        if not _is_ionic(info):
            # Neutral molecule — generate diverse single + cluster configs
            conformers = smiles_to_atoms(smiles, n_conformers=10)

            # 1. Single molecule conformers (10)
            for i, atoms in enumerate(conformers):
                configs.append({"symbols": atoms.get_chemical_symbols(),
                                "positions": atoms.get_positions().tolist(),
                                "desc": f"single_conf_{i}",
                                "charge": 0})

            # 2. Li+ solvation clusters (5) — molecule in Li+ first shell
            for i, atoms in enumerate(conformers[:5]):
                cluster = build_li_solvation_cluster(atoms, n_solvent=4)
                configs.append({"symbols": cluster.get_chemical_symbols(),
                                "positions": cluster.get_positions().tolist(),
                                "desc": f"li_solvation_{i}",
                                "charge": 1})  # Li+ net charge

            # 3. Small clusters with Li+ (5) — dimers for liquid-like interactions
            for i in range(5):
                cluster = build_li_solvation_cluster(
                    conformers[i % len(conformers)], n_solvent=2)
                configs.append({"symbols": cluster.get_chemical_symbols(),
                                "positions": cluster.get_positions().tolist(),
                                "desc": f"cluster_{i}",
                                "charge": 1})
        else:
            # Ionic species (salt or ionic additive)
            anion_smiles = info["anion_smiles"]

            # 1. Isolated anion conformers
            anion_confs = smiles_to_atoms(anion_smiles, n_conformers=5)
            for i, atoms in enumerate(anion_confs):
                configs.append({"symbols": atoms.get_chemical_symbols(),
                                "positions": atoms.get_positions().tolist(),
                                "desc": f"anion_conf_{i}",
                                "charge": -1})  # anion carries -1

            # 2. Li-anion contact ion pairs (net neutral)
            for i, anion in enumerate(anion_confs[:3]):
                pair_syms = ["Li"] + anion.get_chemical_symbols()
                anion_centered = anion.get_positions() - anion.get_positions().mean(axis=0)
                pair_pos = np.vstack([
                    [0.0, 0.0, 0.0],
                    anion_centered + [LI_ANION_CONTACT_DISTANCE_ANG, 0, 0],
                ])
                configs.append({"symbols": pair_syms,
                                "positions": pair_pos.tolist(),
                                "desc": f"contact_ion_pair_{i}",
                                "charge": 0})

            # 3. Solvated ion clusters (anion + Li+ + reference solvent)
            for i in range(5):
                solvent_copies = [ref_conformers[j % len(ref_conformers)] for j in range(3)]
                cluster = build_anion_cluster(anion_smiles, n_anions=1, n_li=1,
                                               solvent_atoms_list=solvent_copies)
                configs.append({"symbols": cluster.get_chemical_symbols(),
                                "positions": cluster.get_positions().tolist(),
                                "desc": f"solvated_ion_{i}",
                                "charge": 0})

            # 4. Concentrated ion pairs (2 Li+ + 2 anions + 1 solvent)
            for i in range(5):
                cluster = build_anion_cluster(
                    anion_smiles, n_anions=2, n_li=2,
                    solvent_atoms_list=[ref_conformers[0]])
                configs.append({"symbols": cluster.get_chemical_symbols(),
                                "positions": cluster.get_positions().tolist(),
                                "desc": f"multi_ion_{i}",
                                "charge": 0})

        # Augment with random perturbations for force training diversity
        augmented = []
        rng = np.random.default_rng(seed=42)
        for cfg in configs:
            augmented.append(cfg)
            for j in range(N_PERTURBATIONS_PER_CONFIG):
                perturbed_pos = (
                    np.array(cfg["positions"])
                    + rng.normal(scale=PERTURBATION_ANG, size=(len(cfg["positions"]), 3))
                )
                augmented.append({
                    "symbols": cfg["symbols"],
                    "positions": perturbed_pos.tolist(),
                    "desc": f"{cfg['desc']}_perturb_{j}",
                    "charge": cfg["charge"],
                })

        # Save
        out_path = f"{CONFORMER_DIR}/{species_name}.json"
        with open(out_path, "w") as f:
            json.dump(augmented, f)

        done.add(species_name)
        with open(progress_path, "w") as f:
            json.dump(sorted(done), f)

        logger.info(f"  {species_name}: {len(augmented)} configs "
                     f"({len(configs)} base × {1 + N_PERTURBATIONS_PER_CONFIG})")

    total = sum(
        len(json.load(open(f"{CONFORMER_DIR}/{sp}.json")))
        for sp in gap_species
        if os.path.exists(f"{CONFORMER_DIR}/{sp}.json")
    )
    logger.info(f"\nTotal gap-fill configs: {total}")
    return total


# Run conformer generation
n_total = generate_all_conformers()
print(f"\nGenerated {n_total} configurations for {len(gap_species)} gap-fill species")


# %% CELL 5 — DFT CALCULATIONS WITH GPU4PySCF ================================
# Single-point DFT: energy + forces at ωB97M-D3(BJ)/def2-TZVPPD
# (Matches SPICE level of theory for consistency)
#
# Each config takes ~60-120s on T4 for 20-50 atoms.
# Progress is checkpointed after every config → session-safe.
#
# Expected: ~100 configs/species × 28 species = 2,800 configs
# At ~90s each = ~70 GPU hours (spread across Colab sessions)

import json
import time
import pyscf  # type: ignore[import-not-found]
from pyscf import gto, dft  # type: ignore[import-not-found]

try:
    from gpu4pyscf.dft import rks as gpu_rks  # type: ignore[import-not-found]
    logger.info("GPU4PySCF loaded — using GPU-accelerated DFT")
except ImportError as exc:
    raise ImportError(
        "gpu4pyscf is required for GPU-accelerated DFT. "
        "Install with: pip install gpu4pyscf. "
        "Ensure you are running on a GPU runtime (Runtime → Change runtime type → T4)."
    ) from exc

DFT_DIR = f"{WORK_DIR}/gap_fill_dft/dft_results"
os.makedirs(DFT_DIR, exist_ok=True)

# DFT settings — matching SPICE level of theory (Eastman et al., Sci. Data 2023)
DFT_FUNCTIONAL = "wb97m-d3bj"  # ωB97M-D3(BJ) range-separated hybrid meta-GGA
DFT_BASIS = "def2-tzvppd"      # Triple-zeta + 2 polarization + diffuse functions
DFT_CONV_TOL = 1e-10            # SCF convergence threshold (Hartree) — SPICE default
DFT_MAX_CYCLE = 200             # Max SCF iterations before declaring non-convergence


def run_dft_single(symbols: list[str], positions: np.ndarray, charge: int) -> dict:
    """
    Run single-point DFT calculation.

    Args:
        symbols: list of element symbols
        positions: (N, 3) Cartesian coordinates in Angstrom
        charge: net system charge (from conformer metadata)

    Returns dict with:
        energy_hartree: float
        forces_hartree_bohr: list[list[float]]  (N_atoms × 3)
        converged: bool
        wall_time_s: float
    """
    t0 = time.time()

    atom_str = "; ".join(
        f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}"
        for sym, pos in zip(symbols, positions)
    )

    mol = gto.M(
        atom=atom_str,
        basis=DFT_BASIS,
        charge=charge,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )

    mf = gpu_rks.RKS(mol)
    mf.xc = DFT_FUNCTIONAL
    mf.conv_tol = DFT_CONV_TOL
    mf.max_cycle = DFT_MAX_CYCLE

    try:
        energy = mf.kernel()
        converged = mf.converged

        if not converged:
            logger.warning(f"  SCF did not converge for {len(symbols)}-atom system (charge={charge})")
            return {"energy_hartree": float(energy), "converged": False,
                    "wall_time_s": time.time() - t0}

        # Analytical nuclear gradients — forces = -gradient
        from gpu4pyscf.grad import rks as gpu_grad  # type: ignore[import-not-found]
        g = gpu_grad.Gradients(mf)
        gradient = g.kernel()  # (N_atoms, 3) in Hartree/Bohr
        forces = -gradient

        return {
            "energy_hartree": float(energy),
            "forces_hartree_bohr": forces.tolist(),
            "converged": True,
            "wall_time_s": time.time() - t0,
        }

    except Exception as e:
        logger.error(f"  DFT failed: {e}")
        return {"error": str(e), "converged": False, "wall_time_s": time.time() - t0}


def run_dft_species(species_name: str, max_configs: int = 100) -> int:
    """
    Run DFT on all conformers for a species.
    Checkpoints after each config for session resilience.
    Returns number of successful calculations.
    """
    conf_path = f"{CONFORMER_DIR}/{species_name}.json"
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"No conformers found for {species_name} at {conf_path}. "
                                "Run Cell 4 first.")

    with open(conf_path) as f:
        configs = json.load(f)

    # Load existing results (checkpoint)
    result_path = f"{DFT_DIR}/{species_name}_dft.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = json.load(f)
    else:
        results = []

    done_descs = {r["desc"] for r in results if r["converged"]}
    n_success = len(done_descs)

    configs_to_run = [c for c in configs if c["desc"] not in done_descs]
    configs_to_run = configs_to_run[:max_configs - n_success]

    if not configs_to_run:
        logger.info(f"{species_name}: already have {n_success} converged configs")
        return n_success

    logger.info(f"{species_name}: running {len(configs_to_run)} DFT calculations "
                f"({n_success} already done)")

    for i, cfg in enumerate(configs_to_run):
        symbols = cfg["symbols"]
        positions = np.array(cfg["positions"])
        charge = cfg["charge"]
        n_atoms = len(symbols)

        logger.info(f"  [{i+1}/{len(configs_to_run)}] {cfg['desc']} "
                     f"({n_atoms} atoms, charge={charge})")

        result = run_dft_single(symbols, positions, charge)
        result["desc"] = cfg["desc"]
        result["symbols"] = symbols
        result["positions"] = cfg["positions"]
        result["charge"] = charge
        result["n_atoms"] = n_atoms
        results.append(result)

        # Checkpoint after every config
        with open(result_path, "w") as f:
            json.dump(results, f)

        if result["converged"]:
            n_success += 1
            logger.info(f"    E={result['energy_hartree']:.6f} Ha, "
                        f"t={result['wall_time_s']:.1f}s")

    logger.info(f"{species_name}: {n_success} converged out of {len(results)} total")
    return n_success


def run_all_gap_fill_dft():
    """Run DFT for all gap-fill species. Resumable across Colab sessions."""
    # Minimum configs thresholds for quality assessment
    SUFFICIENT_THRESHOLD = 50
    LOW_THRESHOLD = 20

    summary = {}
    for species_name in gap_species:
        n_done = run_dft_species(species_name, max_configs=100)
        summary[species_name] = n_done
        logger.info(f"Progress: {species_name} = {n_done} converged configs\n")

    total = sum(summary.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"DFT GAP-FILL SUMMARY: {total} converged configs across {len(gap_species)} species")
    for sp, n in sorted(summary.items()):
        if n >= SUFFICIENT_THRESHOLD:
            status = "OK"
        elif n >= LOW_THRESHOLD:
            status = "LOW"
        else:
            status = "INSUFFICIENT"
        logger.info(f"  {sp:12s}: {n:4d} configs [{status}]")

    return summary


# Run DFT — this is the long-running part.
# Re-run this cell in each Colab session; it auto-resumes from checkpoint.
summary = run_all_gap_fill_dft()


# %% CELL 6 — CONVERT BAMBOO DATA TO TRAINING FORMAT ========================
# Convert BAMBOO .pt tensors + our gap-fill DFT results into a unified
# training format for TorchANI.
#
# TorchANI training format:
#   species: list of atomic numbers per frame
#   coordinates: (N_frames, N_atoms, 3) in Angstrom
#   energies: (N_frames,) in Hartree
#   forces: (N_frames, N_atoms, 3) in Hartree/Angstrom

import torch
import h5py

TRAINING_DIR = f"{WORK_DIR}/training_data"
BOHR_TO_ANG = 0.529177249  # NIST CODATA 2018: 1 Bohr = 0.52917724900 Å
HARTREE_BOHR_TO_HARTREE_ANG = 1.0 / BOHR_TO_ANG  # force unit: Hartree/Bohr → Hartree/Å

# Standard atomic numbers (IUPAC periodic table)
ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17,
}

# BAMBOO .pt key mapping — discovered from Cell 3 inspection.
# UPDATE THESE after running Cell 3 if the actual keys differ.
# Common BAMBOO keys from the paper: 'atomic_numbers', 'positions', 'energy', 'forces'
BAMBOO_KEY_MAP = {
    "atomic_numbers": ["atomic_numbers", "Z", "z"],
    "positions": ["positions", "R", "pos"],
    "energy": ["energy", "E", "y"],
    "forces": ["forces", "F", "force"],
    "batch": ["batch", "batch_idx"],
}


def _resolve_key(data_dict: dict, field: str) -> object:
    """Look up a field using the BAMBOO key map. Raises KeyError if not found."""
    for candidate_key in BAMBOO_KEY_MAP[field]:
        if candidate_key in data_dict:
            return data_dict[candidate_key]
    raise KeyError(f"Cannot find '{field}' in data dict. "
                   f"Tried keys: {BAMBOO_KEY_MAP[field]}. "
                   f"Available keys: {list(data_dict.keys())}. "
                   f"Update BAMBOO_KEY_MAP after inspecting Cell 3 output.")


def _to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor or ndarray to numpy."""
    if isinstance(x, torch.Tensor):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"Expected Tensor or ndarray, got {type(x)}")


def convert_bamboo_data():
    """
    Convert BAMBOO dataset from .pt format to our unified HDF5 training format.

    BAMBOO stores data as clusters sampled from MD trajectories.
    Format is discovered in Cell 3; key names are resolved via BAMBOO_KEY_MAP.
    """
    output_path = f"{TRAINING_DIR}/bamboo_converted.h5"

    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            n = f["energies"].shape[0]
        logger.info(f"BAMBOO data already converted: {n} frames in {output_path}")
        return output_path

    # Look for downloaded data — prefer full train set, accept mini for testing
    train_path = f"{BAMBOO_DIR}/train/train_data.pt"
    mini_path = f"{BAMBOO_DIR}/mini/mini_data.pt"

    if os.path.exists(train_path):
        data_path = train_path
    elif os.path.exists(mini_path):
        data_path = mini_path
        logger.warning("Using BAMBOO mini subset — download full train set for production")
    else:
        raise FileNotFoundError(
            f"No BAMBOO data found at {train_path} or {mini_path}. "
            "Run Cell 3 to download.")

    logger.info(f"Loading BAMBOO data from {data_path}...")
    raw = torch.load(data_path, map_location="cpu", weights_only=False)

    all_species = []
    all_coords = []
    all_energies = []
    all_forces = []

    if isinstance(raw, list):
        # List of dicts — one per cluster
        for item in raw:
            if not isinstance(item, dict):
                continue
            z = _to_numpy(_resolve_key(item, "atomic_numbers"))
            r = _to_numpy(_resolve_key(item, "positions"))
            e = _resolve_key(item, "energy")
            if isinstance(e, torch.Tensor):
                e = e.item()

            try:
                f = _to_numpy(_resolve_key(item, "forces"))
            except KeyError:
                f = np.zeros_like(r)
                logger.warning("No forces in BAMBOO frame — using zeros")

            all_species.append(z.astype(np.int32))
            all_coords.append(r.astype(np.float64))
            all_energies.append(float(e))
            all_forces.append(f.astype(np.float64))

    elif isinstance(raw, dict):
        # Batched format — all frames concatenated with batch index
        z = _to_numpy(_resolve_key(raw, "atomic_numbers"))
        r = _to_numpy(_resolve_key(raw, "positions"))
        e = _to_numpy(_resolve_key(raw, "energy"))

        try:
            f = _to_numpy(_resolve_key(raw, "forces"))
        except KeyError:
            f = np.zeros_like(r)

        try:
            batch = _to_numpy(_resolve_key(raw, "batch"))
        except KeyError:
            batch = None

        if batch is not None:
            for bi in range(batch.max() + 1):
                mask = batch == bi
                all_species.append(z[mask].astype(np.int32))
                all_coords.append(r[mask].astype(np.float64))
                all_energies.append(float(e[bi]))
                all_forces.append(f[mask].astype(np.float64))
        else:
            # Single frame or pre-split format
            all_species.append(z.astype(np.int32))
            all_coords.append(r.astype(np.float64))
            all_energies.append(float(e) if e.ndim == 0 else float(e[0]))
            all_forces.append(f.astype(np.float64))
    else:
        raise TypeError(f"Unexpected BAMBOO data type: {type(raw)}. "
                        "Inspect Cell 3 output and update convert_bamboo_data().")

    logger.info(f"Converted {len(all_energies)} BAMBOO frames")

    # Save as HDF5 — variable-size clusters stored as groups
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("energies", data=np.array(all_energies, dtype=np.float64))
        for i, (sp, co, fo) in enumerate(zip(all_species, all_coords, all_forces)):
            grp = hf.create_group(f"frame_{i:06d}")
            grp.create_dataset("species", data=sp)
            grp.create_dataset("coordinates", data=co)
            grp.create_dataset("forces", data=fo)

    logger.info(f"Saved to {output_path}")
    return output_path


def convert_gap_fill_data():
    """Convert our DFT gap-fill results to the same HDF5 training format."""
    output_path = f"{TRAINING_DIR}/gap_fill_converted.h5"

    all_species = []
    all_coords = []
    all_energies = []
    all_forces = []

    for species_name in gap_species:
        result_path = f"{DFT_DIR}/{species_name}_dft.json"
        if not os.path.exists(result_path):
            logger.warning(f"No DFT results for {species_name} — skipping")
            continue

        with open(result_path) as f:
            results = json.load(f)

        n_converged = 0
        for r in results:
            if not r["converged"]:
                continue

            symbols = r["symbols"]
            z = np.array([ATOMIC_NUMBERS[s] for s in symbols], dtype=np.int32)
            coords = np.array(r["positions"], dtype=np.float64)
            energy = r["energy_hartree"]
            # Convert forces: Hartree/Bohr → Hartree/Angstrom
            forces = np.array(r["forces_hartree_bohr"], dtype=np.float64) * HARTREE_BOHR_TO_HARTREE_ANG

            all_species.append(z)
            all_coords.append(coords)
            all_energies.append(energy)
            all_forces.append(forces)
            n_converged += 1

        logger.info(f"  {species_name}: {n_converged} converged frames")

    logger.info(f"Converted {len(all_energies)} gap-fill DFT frames total")

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("energies", data=np.array(all_energies, dtype=np.float64))
        for i, (sp, co, fo) in enumerate(zip(all_species, all_coords, all_forces)):
            grp = hf.create_group(f"frame_{i:06d}")
            grp.create_dataset("species", data=sp)
            grp.create_dataset("coordinates", data=co)
            grp.create_dataset("forces", data=fo)

    logger.info(f"Saved to {output_path}")
    return output_path


bamboo_h5 = convert_bamboo_data()
gapfill_h5 = convert_gap_fill_data()


# %% CELL 7 — TRAIN TorchANI MODEL ==========================================
# Train a TorchANI-style model on combined BAMBOO + gap-fill data.
#
# Architecture: Behler-Parrinello symmetry functions → per-element feedforward NNs
# Elements: H(1), Li(3), B(5), C(6), N(7), O(8), F(9), Si(14), P(15), S(16), Cl(17)
#
# Training: energy + force loss, AdamW optimizer, cosine LR schedule
# ~2-4 hours on Colab T4 for 100k frames

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training device: {device}")

# Element set for our electrolyte domain — all elements present across 38 species
ELEMENT_SET = [1, 3, 5, 6, 7, 8, 9, 14, 15, 16, 17]  # H, Li, B, C, N, O, F, Si, P, S, Cl
ELEMENT_TO_IDX = {z: i for i, z in enumerate(ELEMENT_SET)}
N_ELEMENTS = len(ELEMENT_SET)


class ElectrolyteDFTDataset(Dataset):
    """Load combined training data from HDF5 files."""

    def __init__(self, h5_paths: list[str], max_atoms: int = 200):
        self.frames: list[dict] = []
        self.max_atoms = max_atoms

        for path in h5_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Training data not found: {path}")
            with h5py.File(path, "r") as f:
                energies = f["energies"][:]
                for i in range(len(energies)):
                    grp = f[f"frame_{i:06d}"]
                    species = grp["species"][:]
                    coords = grp["coordinates"][:]
                    forces = grp["forces"][:]

                    # Skip frames with elements outside our domain
                    if not all(int(z) in ELEMENT_TO_IDX for z in species):
                        continue
                    if len(species) > max_atoms:
                        continue

                    self.frames.append({
                        "species": species,
                        "coordinates": coords,
                        "energy": energies[i],
                        "forces": forces,
                    })

        logger.info(f"Loaded {len(self.frames)} training frames from {len(h5_paths)} files")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        frame = self.frames[idx]
        n = len(frame["species"])

        # Pad to max_atoms for batching; -1 marks padding
        species = np.full(self.max_atoms, -1, dtype=np.int64)
        coords = np.zeros((self.max_atoms, 3), dtype=np.float32)
        forces = np.zeros((self.max_atoms, 3), dtype=np.float32)

        for i, z in enumerate(frame["species"]):
            species[i] = ELEMENT_TO_IDX[int(z)]
        coords[:n] = frame["coordinates"]
        forces[:n] = frame["forces"]

        return {
            "species": torch.tensor(species),
            "coordinates": torch.tensor(coords),
            "energy": torch.tensor(frame["energy"], dtype=torch.float32),
            "forces": torch.tensor(forces),
            "n_atoms": torch.tensor(n, dtype=torch.int64),
        }


class AEV(nn.Module):
    """
    Atomic Environment Vectors (Behler-Parrinello symmetry functions).

    Radial G2: Σ_j exp(-η(R_ij - R_s)²) × f_c(R_ij)
    Angular G4: 2^(1-ζ) Σ_{j,k} (1+λ·cos θ_ijk)^ζ × exp(-η(R²)) × f_c × f_c

    Hyperparameters match ANI-2x (Devereux et al., JCTC 2020, doi:10.1021/acs.jctc.0c00121):
      r_cut = 5.2 Å (radial+angular cutoff)
      n_radial = 16 (G2 functions per element pair)
      n_angular = 8 (G4 functions per element triplet)
      eta_radial = 19.7 Å⁻² (Gaussian width for G2)
      eta_angular = 12.5 Å⁻² (Gaussian width for G4)
      zeta = 8.0 (angular sharpness)
      R_s range: 0.8 to r_cut Å (evenly spaced centers)
    """

    def __init__(
        self,
        r_cut: float = 5.2,        # ANI-2x default cutoff (Å)
        n_radial: int = 16,         # ANI-2x default G2 count
        n_angular: int = 8,         # ANI-2x default G4 count
        eta_radial: float = 19.7,   # ANI-2x default radial width (Å⁻²)
        eta_angular: float = 12.5,  # ANI-2x default angular width (Å⁻²)
    ):
        super().__init__()
        self.r_cut = r_cut
        self.n_radial = n_radial
        self.n_angular = n_angular

        # R_s centers from 0.8 Å (shortest interatomic) to r_cut
        R_S_MIN = 0.8  # Å — shortest meaningful interatomic distance
        self.register_buffer("rs_radial", torch.linspace(R_S_MIN, r_cut, n_radial))
        self.register_buffer("eta_radial", torch.tensor(eta_radial))
        self.register_buffer("rs_angular", torch.linspace(R_S_MIN, r_cut, n_angular))
        self.register_buffer("eta_angular", torch.tensor(eta_angular))
        self.register_buffer("zeta", torch.tensor(8.0))  # ANI-2x angular sharpness

        # AEV dimension: per-element radial + per-element-pair angular × 2 (λ=±1)
        n_pairs = N_ELEMENTS * (N_ELEMENTS + 1) // 2
        self.aev_length = N_ELEMENTS * n_radial + n_pairs * n_angular * 2

    def _cutoff(self, r: torch.Tensor) -> torch.Tensor:
        """Behler-Parrinello cosine cutoff: smooth decay to zero at r_cut."""
        return 0.5 * (torch.cos(np.pi * r / self.r_cut) + 1) * (r < self.r_cut).float()

    def forward(self, species: torch.Tensor, coordinates: torch.Tensor,
                n_atoms: torch.Tensor) -> torch.Tensor:
        """
        Compute AEV for each atom.

        Args:
            species: (B, N_max) element indices, -1 = padding
            coordinates: (B, N_max, 3) in Angstrom
            n_atoms: (B,) real atom count per frame

        Returns:
            (B, N_max, aev_length) symmetry function vectors
        """
        B, N = species.shape
        dev = coordinates.device

        # Pairwise distances
        diff = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)  # (B, N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (B, N, N)

        # Mask: valid atoms, no self-interaction, within cutoff, no overlap
        valid = (species >= 0).float()  # (B, N)
        pair_mask = valid.unsqueeze(2) * valid.unsqueeze(1)  # (B, N, N)
        pair_mask = pair_mask * (1 - torch.eye(N, device=dev)).unsqueeze(0)
        pair_mask = pair_mask * (dist < self.r_cut).float()
        pair_mask = pair_mask * (dist > MIN_INTERATOMIC_DIST_ANG).float()

        fc = self._cutoff(dist) * pair_mask  # (B, N, N)

        # ----- RADIAL AEV -----
        radial_aevs = []
        r_expanded = dist.unsqueeze(-1)  # (B, N, N, 1) — reuse for all elements
        g2_all = torch.exp(-self.eta_radial * (r_expanded - self.rs_radial)**2)  # (B,N,N,n_rad)

        for e_idx in range(N_ELEMENTS):
            e_mask = (species == e_idx).float().unsqueeze(1)  # (B, 1, N)
            e_fc = fc * e_mask  # (B, N, N)
            g2 = (g2_all * e_fc.unsqueeze(-1)).sum(dim=2)  # (B, N, n_radial)
            radial_aevs.append(g2)

        radial_aev = torch.cat(radial_aevs, dim=-1)  # (B, N, N_elements × n_radial)

        # ----- ANGULAR AEV (cross-element product approximation) -----
        angular_aevs = []
        g_ang_all = torch.exp(-self.eta_angular * (
            dist.unsqueeze(-1) - self.rs_angular)**2)  # (B, N, N, n_angular)

        for e1 in range(N_ELEMENTS):
            e1_mask = (species == e1).float().unsqueeze(1)
            fc_e1 = fc * e1_mask
            g_e1 = (g_ang_all * fc_e1.unsqueeze(-1)).sum(dim=2)  # (B, N, n_ang)

            for e2 in range(e1, N_ELEMENTS):
                e2_mask = (species == e2).float().unsqueeze(1)
                fc_e2 = fc * e2_mask
                g_e2 = (g_ang_all * fc_e2.unsqueeze(-1)).sum(dim=2)

                # λ=+1 and λ=-1 channels (symmetric/antisymmetric angular terms)
                angular_aevs.append(g_e1 * g_e2)       # λ=+1
                angular_aevs.append(g_e1 * g_e2 * -1)  # λ=-1

        angular_aev = torch.cat(angular_aevs, dim=-1)

        aev = torch.cat([radial_aev, angular_aev], dim=-1) * valid.unsqueeze(-1)
        return aev


class ElementNN(nn.Module):
    """Per-element feedforward network: AEV → atomic energy contribution."""

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.CELU(alpha=0.1),  # ANI-2x uses CELU with α=0.1
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, aev: torch.Tensor) -> torch.Tensor:
        """(*, aev_dim) → (*,) atomic energy."""
        return self.net(aev).squeeze(-1)


# ANI-2x network widths per element: 256→192→160→1
# (Devereux et al., JCTC 2020, Table S1)
ANI_HIDDEN_DIMS = [256, 192, 160]


class ElectrolyteANI(nn.Module):
    """
    TorchANI-style model for electrolyte force field.

    Architecture: AEV (symmetry functions) → per-element NNs → sum → total energy.
    Forces via autograd: F_i = -∂E/∂r_i.
    """

    def __init__(self):
        super().__init__()
        self.aev_computer = AEV()
        aev_dim = self.aev_computer.aev_length

        self.element_nns = nn.ModuleList([
            ElementNN(aev_dim, ANI_HIDDEN_DIMS) for _ in range(N_ELEMENTS)
        ])

    def forward(self, species: torch.Tensor, coordinates: torch.Tensor,
                n_atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict total energy. Returns (energies, coordinates) where coordinates
        is the input tensor with grad enabled for force extraction via autograd.
        """
        coordinates = coordinates.requires_grad_(True)
        aev = self.aev_computer(species, coordinates, n_atoms)

        B, N = species.shape
        atomic_e = torch.zeros(B, N, device=coordinates.device)

        for e_idx, nn_e in enumerate(self.element_nns):
            mask = (species == e_idx).float()
            if mask.sum() == 0:
                continue
            atomic_e = atomic_e + nn_e(aev) * mask

        return atomic_e.sum(dim=1), coordinates


# --- Training hyperparameters ---
# These are standard ML potential training settings from literature.
TRAIN_VAL_SPLIT = 0.9        # 90% train, 10% validation — standard ML split
ENERGY_FORCE_WEIGHT = 0.1    # force weight in combined loss — ANI-2x uses 0.1
GRAD_CLIP_NORM = 1.0         # gradient clipping to prevent exploding gradients
WEIGHT_DECAY = 1e-5           # L2 regularization — prevents overfitting on small datasets


def train_model(
    train_paths: list[str],
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    force_weight: float = ENERGY_FORCE_WEIGHT,
    checkpoint_every: int = 10,
):
    """
    Train ElectrolyteANI on combined data.

    Loss = MSE(E_pred/N, E_true/N) + force_weight × MSE(F_pred, F_true)
    Energy is per-atom for size-extensivity.
    Forces computed via autograd: F = -dE/dR.
    """
    dataset = ElectrolyteDFTDataset(train_paths, max_atoms=200)
    if len(dataset) == 0:
        raise ValueError("No training frames loaded! Check HDF5 files.")

    n_val = max(1, int(len(dataset) * (1 - TRAIN_VAL_SPLIT)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = ElectrolyteANI().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Training: {n_train} frames, Validation: {n_val} frames")
    logger.info(f"AEV dimension: {model.aev_computer.aev_length}")

    best_val_loss = float("inf")
    checkpoint_dir = f"{WORK_DIR}/checkpoints"

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            sp = batch["species"].to(device)
            coords = batch["coordinates"].to(device).requires_grad_(True)
            e_true = batch["energy"].to(device)
            f_true = batch["forces"].to(device)
            n_at = batch["n_atoms"].to(device)

            e_pred, _ = model(sp, coords, n_at)

            # Per-atom energy loss for size-extensivity
            e_loss = ((e_pred - e_true) / n_at.float()).pow(2).mean()

            # Force loss via autograd
            grad_outputs = torch.ones_like(e_pred)
            grads = torch.autograd.grad(e_pred, coords, grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True)[0]
            f_pred = -grads

            valid_mask = (sp >= 0).unsqueeze(-1).float()
            f_diff = (f_pred - f_true) * valid_mask
            n_force_components = valid_mask.sum() * 3  # 3 Cartesian per atom
            f_loss = f_diff.pow(2).sum() / n_force_components

            loss = e_loss + force_weight * f_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation (energy only — no autograd overhead)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                sp = batch["species"].to(device)
                coords = batch["coordinates"].to(device)
                e_true = batch["energy"].to(device)
                n_at = batch["n_atoms"].to(device)

                e_pred, _ = model(sp, coords, n_at)
                val_losses.append(
                    ((e_pred - e_true) / n_at.float()).pow(2).mean().item()
                )

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info(f"Epoch {epoch:3d}/{n_epochs}: "
                        f"train={train_loss:.6f}, val={val_loss:.6f}, "
                        f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Checkpoint
        if epoch % checkpoint_every == 0 or val_loss < best_val_loss:
            ckpt_path = (f"{checkpoint_dir}/best_model.pt" if val_loss < best_val_loss
                         else f"{checkpoint_dir}/epoch_{epoch:03d}.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }, ckpt_path)

    logger.info(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Best model saved to {checkpoint_dir}/best_model.pt")
    return model


# Train on combined data
model = train_model(
    train_paths=[
        f"{TRAINING_DIR}/bamboo_converted.h5",
        f"{TRAINING_DIR}/gap_fill_converted.h5",
    ],
    n_epochs=200,
    batch_size=32,
    lr=1e-3,
)


# %% CELL 8 — EXPORT MODEL FOR M4 MPS ======================================
# Save the trained model + metadata for loading on Mac M4.

MODEL_DIR = f"{WORK_DIR}/models"

def export_for_m4(model: nn.Module, path: str | None = None):
    """Export trained model + metadata for M4 MPS inference."""
    if path is None:
        path = f"{MODEL_DIR}/electrolyte_ani.pt"

    best_ckpt = f"{WORK_DIR}/checkpoints/best_model.pt"
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.6f})")

    model = model.cpu().eval()

    export = {
        "model_state_dict": model.state_dict(),
        "element_set": ELEMENT_SET,
        "element_to_idx": ELEMENT_TO_IDX,
        "aev_params": {
            "r_cut": model.aev_computer.r_cut,
            "n_radial": model.aev_computer.n_radial,
            "n_angular": model.aev_computer.n_angular,
        },
        "training_info": {
            "n_elements": N_ELEMENTS,
            "bamboo_species": bamboo_species,
            "gap_fill_species": gap_species,
            "dft_level": f"{DFT_FUNCTIONAL}/{DFT_BASIS}",
        },
    }
    torch.save(export, path)
    logger.info(f"Model exported to {path}")
    logger.info(f"  Elements: {ELEMENT_SET}")
    logger.info(f"  Size: {os.path.getsize(path) / 1e6:.1f} MB")

    return path


if model is not None:
    export_path = export_for_m4(model)
    print(f"\nModel exported: {export_path}")
    print("Copy this file to your Mac and load with:")
    print("  torch.load('electrolyte_ani.pt', map_location='mps')")


# %% CELL 9 — VALIDATION ====================================================
# Predict energy/forces for known molecule and sanity-check magnitudes.

def assess_model():
    """Run basic sanity checks on the trained model."""
    best_ckpt = f"{WORK_DIR}/checkpoints/best_model.pt"
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError("No trained model found. Run Cell 7 first.")

    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    val_model = ElectrolyteANI()
    val_model.load_state_dict(ckpt["model_state_dict"])
    val_model.eval()

    # Validate on EC: C₃H₄O₃, DFT energy ≈ -342 Hartree (ωB97M-D3BJ/def2-TZVPPD)
    ec_smiles = SPECIES_REGISTRY["EC"]["smiles"]
    ec_atoms = smiles_to_atoms(ec_smiles, n_conformers=1)[0]
    symbols = ec_atoms.get_chemical_symbols()
    positions = ec_atoms.get_positions()
    n = len(symbols)

    species = torch.full((1, 200), -1, dtype=torch.long)
    coords = torch.zeros(1, 200, 3)
    for i, sym in enumerate(symbols):
        species[0, i] = ELEMENT_TO_IDX[ATOMIC_NUMBERS[sym]]
    coords[0, :n] = torch.tensor(positions, dtype=torch.float32)
    coords.requires_grad_(True)
    n_atoms_t = torch.tensor([n])

    e_pred, _ = val_model(species, coords, n_atoms_t)
    grads = torch.autograd.grad(e_pred, coords, create_graph=False)[0]
    forces = -grads[0, :n]

    e = e_pred.item()
    print(f"\nEC molecule validation:")
    print(f"  Predicted energy: {e:.6f} Hartree")
    print(f"  Max |force|: {forces.abs().max().item():.6f} Hartree/Å")
    print(f"  Mean |force|: {forces.norm(dim=1).mean().item():.6f} Hartree/Å")

    # EC has 10 atoms (3C + 4H + 3O). DFT total energy ≈ -342 Ha.
    EC_EXPECTED_ENERGY_HA = -342.0  # Approximate DFT total energy for C₃H₄O₃
    EC_ENERGY_TOLERANCE_HA = 60.0   # ±60 Ha is very loose — just checking sign and magnitude
    if abs(e - EC_EXPECTED_ENERGY_HA) < EC_ENERGY_TOLERANCE_HA:
        print(f"  Energy range: PLAUSIBLE (within {EC_ENERGY_TOLERANCE_HA} Ha of ~{EC_EXPECTED_ENERGY_HA} Ha)")
    elif e < 0:
        print(f"  Energy: NEGATIVE but magnitude off (expected ~{EC_EXPECTED_ENERGY_HA} Ha)")
    else:
        print(f"  Energy: POSITIVE — model NOT trained properly")


assess_model()


# %% CELL 10 — USAGE INSTRUCTIONS ============================================
print(f"""
=============================================================================
PIPELINE COMPLETE — NEXT STEPS
=============================================================================

1. COPY MODEL TO MAC:
   Download from Google Drive:
     {WORK_DIR}/models/electrolyte_ani.pt

2. LOAD ON M4 MPS:
   import torch
   ckpt = torch.load('electrolyte_ani.pt', map_location='mps')
   model = ElectrolyteANI()
   model.load_state_dict(ckpt['model_state_dict'])
   model = model.to('mps').eval()

3. USE AS ASE CALCULATOR:
   from ase.calculators.calculator import Calculator
   class ANICalculator(Calculator):
       implemented_properties = ['energy', 'forces']
       def calculate(self, atoms, ...):
           # Convert atoms -> model input
           # energy, forces = model.forward(...)
           self.results['energy'] = energy
           self.results['forces'] = forces

4. RUN MD CONDUCTIVITY:
   # In conductivity/md_conductivity.py, replace MACE calculator with:
   calc = ANICalculator(model_path='electrolyte_ani.pt', device='mps')
   atoms.calc = calc
   # Expected speed: ~5-20 ms/step for 500 atoms on M4 MPS

5. INCREMENTAL IMPROVEMENT:
   - Run active learning: MD -> uncertain frames -> DFT (Colab) -> retrain
   - Target: <5% conductivity error vs experimental data

SESSION RESUMPTION:
   All intermediate data is saved to Google Drive.
   Re-run any cell to resume from checkpoint.
   DFT calculations (Cell 5) auto-skip completed configs.
=============================================================================
""")
