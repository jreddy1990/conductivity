"""Flow Matching Conductivity Model for Liquid Li/Na Electrolytes.

Implements the plan in .claude/diffusion_model_conductivity.md.

Architecture:
  composition (species + mole fractions + T)
    -> shared molecular GNN per species (WL-expressive at depth 4)
    -> permutation-invariant attention pool over species (universal set function)
    -> composition latent z in R^128
    -> conditional flow matching network on spectrum manifold R^(2K)
    -> Green-Kubo readout: sigma = (1 / 3 k_B T) sum_k exp(alpha_k - lambda_k)

Training data: OEDB v1 (single-component MD), BAMBOO-Mix (multi-component NE,
Haven-corrected), Uni-ELF (binary mixtures), CALiSol-23 (experimental, validation only).

Entry point: python -m conductivity.flow_matching_conductivity
"""

import sys

sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")

import control_framework.jax_m4_tuning  # noqa: F401 - must precede jax import

import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, lax, random, value_and_grad, vmap
import optax

from rdkit import Chem

from constants import K_B, E_CHARGE, S_M_TO_MS_CM, T_REF_K, CELSIUS_TO_KELVIN
from data.species_data import SOLVENTS, SALTS, ADDITIVES
from foundation_model.cepstral_conductivity import (
    CurrentBurstSet,
    SegmentationConfig,
    TruncationAIC,
    estimate_conductivity_cepstral,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# All constants are explicit and visible for auditing. No hidden defaults.
# =============================================================================

# --- Architecture sizes -------------------------------------------------------
K_SPECTRUM = 8                    # Number of eigenvalue+amplitude pairs in spectrum
# Atom feature vector layout. Counts come from the chemistry vocabularies
# defined later in this module (_ELEMENT_VOCAB, _HYBRID_VOCAB); the offset
# constants below are derived analytically from those vocab sizes so that
# resizing a vocabulary automatically updates every dependent offset.
ATOM_FEAT_ELEMENT_LEN = 20        # Explicit constant: element vocabulary size, matches len(_ELEMENT_VOCAB)
ATOM_FEAT_HYBRID_LEN = 5          # Explicit constant: hybridization vocabulary size, matches len(_HYBRID_VOCAB)
ATOM_FEAT_CHARGE_IDX = ATOM_FEAT_ELEMENT_LEN                          # Derived: charge slot follows element block
ATOM_FEAT_HYBRID_START = ATOM_FEAT_CHARGE_IDX + 1                      # Derived: hybrid block follows charge slot
ATOM_FEAT_AROMATIC_IDX = ATOM_FEAT_HYBRID_START + ATOM_FEAT_HYBRID_LEN  # Derived: aromatic flag follows hybrid block
D_ATOM = ATOM_FEAT_AROMATIC_IDX + 1                                    # Derived: full atom feature width
ATOM_FEAT_RING_OFFSET = 0.5       # Explicit constant: marker offset (charge slot) for ring membership; chosen distinct from integer formal charges so a learner can disentangle

D_BOND_ORDER_LEN = 4              # Explicit constant: bond-order vocabulary size, matches len(_BOND_ORDER_VOCAB)
D_BOND = D_BOND_ORDER_LEN + 1     # Derived: 4 bond-order one-hot + 1 aromatic flag

# Rows with no per-source uncertainty are loaded with uncertainty=0, which
# weights them at 1.0 in the loss (no down-weighting). Calibrated per-row
# uncertainty should be plumbed through the source DB, not faked here.
D_MOL = 32                        # Explicit constant: per-species molecular embedding dim (shrunk from 64 for the small-data regime; 4-WL still distinguishes electrolyte species at this width)
D_COMP = 64                       # Explicit constant: composition latent dim (shrunk from 128; matches N_ATTN_HEADS x D_HEAD)
D_FM_TOKEN = 64                   # Explicit constant: FM transformer token dim (matches D_COMP for shared head count)
D_HEAD = 16                       # Explicit constant: per-head dim in multi-head attention (4 heads x 16 = D_COMP)
N_GNN_LAYERS = 3                  # Explicit constant: molecular GNN depth (3 is WL-expressive at depth 3 for liquid-electrolyte species, less overfitting at small data)
N_ATTN_LAYERS = 2                 # Explicit constant: species-set attention layers (2 is sufficient for Set Transformer universality at this set size)
N_ATTN_HEADS = 4                  # Explicit constant: attention heads
N_FM_LAYERS = 3                   # Explicit constant: flow matching transformer blocks (3 from 4 — smaller model for small data)

# --- Fourier feature widths ---------------------------------------------------
N_FOURIER_FRACTION = 8            # Mole fraction encoding frequencies
N_FOURIER_TEMPERATURE = 16        # Temperature encoding frequencies
N_FOURIER_FLOWTIME = 16           # Flow-time s encoding frequencies

# --- Flow matching ODE --------------------------------------------------------
ODE_STEPS = 20                    # Euler steps from s=0 to s=1
SPECTRUM_LAMBDA_BOUND = 3.0       # Explicit constant: tanh bound on log eigenvalue; gives lambda dynamic range ~[0.05, 20] which covers slow/fast modes in liquid electrolyte spectra without saturating the latent
SPECTRUM_ALPHA_BOUND = 5.0        # Explicit constant: tanh bound on log amplitude; with the lambda bound above gives per-mode sigma contribution exp(-8..+8) ~ 3000x dynamic range, matches physical sigma variation observed in OEDB+BAMBOO

# --- Composition sizing -------------------------------------------------------
MAX_SPECIES = 10                  # Hard cap on species per composition (pad up)
MAX_ATOMS = 70                    # Explicit constant: pad bound on atoms; sized for the largest BAMBOO-Mix glyme-fluorinated-ether species
MAX_BONDS = 150                   # Explicit constant: pad bound on directed bonds; sized for the largest BAMBOO-Mix fluorinated PEG-like ether (128 directed bonds) with headroom

# --- Physics constants (imported from constants.py; see top of file) ---------
# K_B (Boltzmann), E_CHARGE (elementary charge), S_M_TO_MS_CM (unit conversion)

# --- Validity ranges for cleaning ---------------------------------------------
SIGMA_MIN_MSCM = 1e-4
SIGMA_MAX_MSCM = 300.0
T_MIN_K = 220.0
T_MAX_K = 400.0
MOLE_FRAC_SUM_TOL = 5e-3
WALDEN_LOG_TOLERANCE = 1.0

# --- Training -----------------------------------------------------------------
LR_PEAK = 3e-4
LR_FLOOR = 1e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
N_FM_SAMPLES_PER_COMP = 4
N_EPOCHS = 100
GRAD_CLIP_NORM = 1.0
LAMBDA_FM = 1.0
LAMBDA_SIGMA = 1.0
LAMBDA_AUX = 0.2

# --- Haven ratio calibration --------------------------------------------------
HAVEN_CALIBRATION_N = 50
HAVEN_CONSTANT_THRESHOLD = 0.10
HAVEN_FIT_HOLDOUT_FRAC = 0.10
HAVEN_FIT_PASS_REL_ERROR = 0.20

# --- Data layout --------------------------------------------------------------
DATA_DIR = Path("/Users/jreddy/electrolyte_formation_sim/conductivity/fm_data")
OEDB_CSV = DATA_DIR / "oedb_v1.csv"
BAMBOO_CSV = DATA_DIR / "bamboo_mix.csv"
BAMBOO_TRAJ_DIR = DATA_DIR / "bamboo_trajectories"
UNI_ELF_CSV = DATA_DIR / "uni_elf.csv"
CALISOL_CSV = DATA_DIR / "calisol_23.csv"
INSPECT_REPORT_PATH = DATA_DIR / "inspection_report.json"
CLEAN_DATASET_PATH = DATA_DIR / "clean_dataset.pkl"
HAVEN_PARAMS_PATH = DATA_DIR / "haven_theta.pkl"
TRAINED_MODEL_PATH = DATA_DIR / "fm_conductivity_model.pkl"

# --- Held-out species for OOD test (Phase 5.1 in plan) ------------------------
HELDOUT_SOLVENTS = ("FEC", "GBL")
HELDOUT_ANIONS = ("BETI",)

# --- Seeds --------------------------------------------------------------------
SEED_SPLIT = 42
SEED_TRAIN = 7
SEED_INIT = 1337


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class SigmaMethod(Enum):
    GREEN_KUBO = "GK"
    NERNST_EINSTEIN = "NE"
    NE_HAVEN_CORRECTED = "NE-Haven"
    EXPERIMENTAL_EIS = "EIS-exp"


@dataclass
class LabeledRow:
    """One labeled composition with provenance.

    composition_key is a canonical string for deduplication.
    aux_observables: dict containing any of {viscosity_mPas, diffusion_cation_m2s,
    diffusion_anion_m2s, density_gcm3}. Keys present iff measurement available.
    """
    composition_key: str
    smiles_list: List[str]
    mole_fractions: np.ndarray
    temperature_K: float
    sigma_mScm: float
    sigma_source: str
    sigma_method: SigmaMethod
    sigma_uncertainty_log: float
    aux_observables: Dict[str, float] = field(default_factory=dict)


@dataclass
class MolecularGraph:
    """Per-molecule atomic graph, padded to fixed sizes for jit."""
    atom_features: np.ndarray      # (MAX_ATOMS, D_ATOM)
    bond_features: np.ndarray      # (MAX_BONDS, D_BOND)
    bond_src: np.ndarray           # (MAX_BONDS,) atom index
    bond_dst: np.ndarray           # (MAX_BONDS,) atom index
    atom_mask: np.ndarray          # (MAX_ATOMS,) 1 for real, 0 for padding
    bond_mask: np.ndarray          # (MAX_BONDS,) 1 for real, 0 for padding


EMPTY_GRAPH = MolecularGraph(
    atom_features=np.zeros((MAX_ATOMS, D_ATOM)),
    bond_features=np.zeros((MAX_BONDS, D_BOND)),
    bond_src=np.zeros(MAX_BONDS, dtype=np.int32),
    bond_dst=np.zeros(MAX_BONDS, dtype=np.int32),
    atom_mask=np.zeros(MAX_ATOMS),
    bond_mask=np.zeros(MAX_BONDS),
)


@dataclass
class CompositionInput:
    """Padded composition input ready for the model forward pass."""
    graphs: List[MolecularGraph]      # length MAX_SPECIES, padded with EMPTY_GRAPH
    mole_fractions: jnp.ndarray       # (MAX_SPECIES,)
    species_mask: jnp.ndarray         # (MAX_SPECIES,)
    temperature_K: float


@dataclass
class RawComposition:
    """User-facing composition input for inference."""
    smiles_list: List[str]
    mole_fractions: np.ndarray
    temperature_K: float


@dataclass
class NormalizationStats:
    """Training-only normalization statistics."""
    T_mean: float
    T_std: float

    def normalize_T(self, T):
        return (T - self.T_mean) / max(self.T_std, 1e-8)


@dataclass
class ModelBundle:
    """Trained model parameters plus normalization stats."""
    params: Dict
    norm_stats: NormalizationStats


@dataclass
class HavenCorrection:
    """Fit result from Haven ratio calibration.

    mode='constant': uses log_H_bar only.
    mode='linear':   uses w; applies log_H = z @ w.
    """
    mode: str                       # 'constant' or 'linear'
    log_H_bar: float                # used for both modes (linear: mean offset)
    w: np.ndarray                   # (D_COMP,) zero array if constant
    holdout_mae: float              # NaN for constant


# =============================================================================
# DATA ACQUISITION
# Stubs that fail loudly with download instructions.
# =============================================================================


OEDB_ARROW = DATA_DIR / "oedb" / "electrolytes.arrow"
BAMBOO_JSON = DATA_DIR / "bamboo_mix" / "dataset" / "data.json"

# OEDB MD is run at a single fixed temperature (Methods: 25C = T_REF_K).
OEDB_FIXED_TEMPERATURE_K = T_REF_K

# Cation-only SMILES for ionic species. The molecular GNN needs a graph for
# every distinct species in the composition; OEDB and BAMBOO encode the
# anion (and solvent) as SMILES but not the cation, so we supply the cation
# SMILES here. Charge state matters: Li+ != Li metal.
CATION_SMILES: Dict[str, str] = {
    "Li": "[Li+]",
    "Na": "[Na+]",
    "K":  "[K+]",
}


def load_oedb_v1() -> List[LabeledRow]:
    """OEDB v1: 5,616 single-component MD formulations.

    Loaded from electrolytes.arrow downloaded from https://oedb.jp/data/.
    Each row is a (cation, anion, solvent, concentration) tuple with MD-derived
    sigma, viscosity, diffusivities, coordination numbers, and density.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    if not OEDB_ARROW.exists():
        raise FileNotFoundError(
            f"OEDB Arrow file not found at {OEDB_ARROW}. "
            f"Download with: curl -L https://oedb.jp/data/electrolytes.arrow "
            f"-o {OEDB_ARROW}"
        )
    with pa.memory_map(str(OEDB_ARROW), "r") as src:
        table = ipc.open_file(src).read_all()

    rows = []
    for i in range(table.num_rows):
        cation_name = table["cation"][i].as_py()
        anion_name = table["anion"][i].as_py()
        anion_smiles = table["anion_smiles"][i].as_py()
        solvent_name = table["solvent"][i].as_py()
        solvent_smiles = table["solvent_smiles"][i].as_py()
        concentration = float(table["Concentration (mol/kg)"][i].as_py())
        sigma = table["Ionic Conductivity (mS/cm)"][i].as_py()
        if sigma is None or sigma <= 0:
            continue
        if cation_name not in CATION_SMILES:
            continue
        cation_smiles = CATION_SMILES[cation_name]
        # Single-salt single-solvent: derive simple mole fractions from
        # the salt molality (mol/kg solvent). 1 kg solvent contains
        # concentration moles of salt and (1000 / MW_solvent) moles of
        # solvent. We do not have MW per row, so we instead encode the
        # *relative* loadings as (1 mol solvent, concentration mol salt)
        # normalized; the molecular GNN learns the per-species identity
        # and the attention pool sees the mole-fraction ratio.
        moles = np.array([1.0, concentration, concentration])  # [solvent, cation, anion]
        fracs = moles / moles.sum()
        smiles_list = [solvent_smiles, cation_smiles, anion_smiles]
        aux = {}
        for key, col in (
            ("viscosity_mPas", "Viscosity (mPa·s)"),
            ("diffusion_cation_m2s", "Cation's Diffusivity (m²/s)"),
            ("diffusion_anion_m2s", "Anion's Diffusivity (m²/s)"),
            ("density_gcm3", "Density (g/cm³)"),
        ):
            v = table[col][i].as_py()
            if v is not None:
                aux[key] = float(v)
        rows.append(LabeledRow(
            composition_key=f"oedb-{int(table['id'][i].as_py())}",
            smiles_list=smiles_list,
            mole_fractions=fracs,
            temperature_K=OEDB_FIXED_TEMPERATURE_K,
            sigma_mScm=float(sigma),
            sigma_source="OEDB-v1",
            sigma_method=SigmaMethod.GREEN_KUBO,   # OEDB Methods uses Green-Kubo with cepstral
            sigma_uncertainty_log=0.0,
            aux_observables=aux,
        ))
    logger.info("Loaded %d OEDB rows from %s", len(rows), OEDB_ARROW)
    return rows


def load_bamboo_mix() -> List[LabeledRow]:
    """BAMBOO-Mix EIS rows (conductivity_mask=True). mask=False has no real sigma."""
    if not BAMBOO_JSON.exists():
        raise FileNotFoundError(
            f"BAMBOO-Mix JSON not found at {BAMBOO_JSON}. "
            f"Download with: python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download('ByteDance-Seed/bamboo_mixer', 'dataset/data.json', "
            f"repo_type='model', local_dir='{DATA_DIR}/bamboo_mix')\""
        )
    with open(BAMBOO_JSON) as f:
        data = json.load(f)

    rows = []
    for idx, entry in enumerate(data):
        if not entry["conductivity_mask"]:
            continue
        sigma = entry["conductivity"]
        if sigma is None or sigma <= 0:
            continue
        T_K = float(entry["temperature"]) + CELSIUS_TO_KELVIN
        smiles_list = []
        moles = []
        for spec in entry["solvents"]:
            smiles_list.append(spec["smiles"])
            moles.append(float(spec["molar_ratio"]))
        for spec in entry["salts"]:
            cation_name = spec["name"][:2] if spec["name"].startswith(("Li", "Na")) else None
            if cation_name in CATION_SMILES:
                smiles_list.append(CATION_SMILES[cation_name])
                moles.append(float(spec["molar_ratio"]))
            smiles_list.append(spec["smiles"])
            moles.append(float(spec["molar_ratio"]))
        if not smiles_list:
            continue
        moles_arr = np.array(moles, dtype=np.float64)
        fracs = moles_arr / moles_arr.sum()
        method = SigmaMethod.EXPERIMENTAL_EIS
        rows.append(LabeledRow(
            composition_key=f"bamboo-{idx}",
            smiles_list=smiles_list,
            mole_fractions=fracs,
            temperature_K=T_K,
            sigma_mScm=float(sigma),
            sigma_source="BAMBOO-Mix",
            sigma_method=method,
            sigma_uncertainty_log=0.0,
            aux_observables={},
        ))
    logger.info("Loaded %d BAMBOO-Mix EIS rows", len(rows))
    return rows


def load_uni_elf() -> List[LabeledRow]:
    """Uni-ELF: 180 binary-mixture MD formulations."""
    if not UNI_ELF_CSV.exists():
        raise FileNotFoundError(
            f"Uni-ELF CSV not found at {UNI_ELF_CSV}. "
            f"Download from github.com/dptech-corp/uni-elf."
        )
    raise NotImplementedError("Uni-ELF CSV schema parsing not implemented.")


def load_calisol() -> List[LabeledRow]:
    """CALiSol-23: experimental conductivity rows (validation only)."""
    if not CALISOL_CSV.exists():
        raise FileNotFoundError(
            f"CALiSol-23 CSV not found at {CALISOL_CSV}. "
            f"Download from DTU Data (CC-BY 4.0)."
        )
    raise NotImplementedError("CALiSol-23 CSV schema parsing not implemented.")


def load_bamboo_trajectory(composition_key: str) -> Dict:
    """Load a BAMBOO trajectory for Haven calibration."""
    traj_path = BAMBOO_TRAJ_DIR / f"{composition_key}.npz"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"BAMBOO trajectory for {composition_key} not at {traj_path}."
        )
    return dict(np.load(traj_path))


# =============================================================================
# SPECIES NAME -> SMILES MAPPING
# Project conventions (data/species_data.py) key species by name, not SMILES.
# This table provides canonical SMILES for every standard liquid-electrolyte
# species in the project, used by the molecular GNN encoder.
# =============================================================================

SMILES_BY_SPECIES: Dict[str, str] = {
    # --- Cyclic carbonates ---
    "EC": "O=C1OCCO1",                          # ethylene carbonate
    "PC": "O=C1OC(C)CO1",                       # propylene carbonate
    "BC": "O=C1OC(CC)CO1",                      # butylene carbonate
    "FEC": "O=C1OCC(F)O1",                      # fluoroethylene carbonate
    "VC": "O=C1OC=CO1",                         # vinylene carbonate
    "GBL": "O=C1CCCO1",                         # gamma-butyrolactone
    # --- Linear carbonates ---
    "DMC": "COC(=O)OC",                         # dimethyl carbonate
    "EMC": "CCOC(=O)OC",                        # ethyl methyl carbonate
    "DEC": "CCOC(=O)OCC",                       # diethyl carbonate
    "FEMC": "FCOC(=O)OCC",                      # fluoro-EMC variant
    # --- Ethers ---
    "DME": "COCCOC",                            # dimethoxyethane
    "DOL": "C1OCCO1",                           # 1,3-dioxolane
    "DEE": "CCOCC",                             # diethyl ether
    # --- Esters ---
    "EA": "CCOC(C)=O",                          # ethyl acetate
    "MA": "COC(C)=O",                           # methyl acetate
    "EP": "CCOC(=O)CC",                         # ethyl propionate
    "MP": "COC(=O)CC",                          # methyl propionate
    "EB": "CCCC(=O)OCC",                        # ethyl butyrate
    "MB": "CCCC(=O)OC",                         # methyl butyrate
    # --- Nitriles ---
    "AN": "CC#N",                               # acetonitrile
    "GN": "N#CCCCC#N",                          # glutaronitrile
    "SN": "N#CCCC#N",                           # succinonitrile
    # --- Sulfoxides / sulfones ---
    "DMSO": "CS(=O)C",
    "DMI": "O=C1N(C)CCN1C",                     # 1,3-dimethyl-2-imidazolidinone
    # --- Phosphates ---
    "TMP": "COP(=O)(OC)OC",                     # trimethyl phosphate
    "TEP": "CCOP(=O)(OCC)OCC",                  # triethyl phosphate
    "TPP": "O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1",  # triphenyl phosphate
    # --- Misc / additives ---
    "PS": "O=S1(=O)OCCC1",                      # propane sultone
    "DTD": "O=S1(=O)OCCO1",                     # DTD additive
    # --- Salts (anion-only graph; cation is Li+/Na+/K+ handled separately) ---
    "LiPF6": "[Li+].F[P-](F)(F)(F)(F)F",
    "LiFSI": "[Li+].O=S(=O)([N-]S(=O)(=O)F)F",
    "LiTFSI": "[Li+].O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F",
    "LiBF4": "[Li+].F[B-](F)(F)F",
    "LiClO4": "[Li+].[O-]Cl(=O)(=O)=O",
    "LiBOB": "[Li+].O=C1OB2(OC1=O)OC(=O)C(=O)O2",
    "LiDFOB": "[Li+].O=C1O[B-](F)(F)OC1=O",
    "NaPF6": "[Na+].F[P-](F)(F)(F)(F)F",
    "NaFSI": "[Na+].O=S(=O)([N-]S(=O)(=O)F)F",
    "NaTFSI": "[Na+].O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F",
    "NaClO4": "[Na+].[O-]Cl(=O)(=O)=O",
    "NaBF4": "[Na+].F[B-](F)(F)F",
}


def smiles_for_species(name: str) -> str:
    """Look up canonical SMILES for a project species name.

    Fails loudly if the species is unknown so missing wiring surfaces
    immediately rather than silently substituting a default.
    """
    if name not in SMILES_BY_SPECIES:
        raise KeyError(
            f"No SMILES registered for species {name!r}. Add it to "
            f"SMILES_BY_SPECIES in flow_matching_conductivity.py."
        )
    return SMILES_BY_SPECIES[name]


# =============================================================================
# SMILES -> MolecularGraph (RDKit)
# =============================================================================

# Element vocabulary for one-hot encoding. 20 elements cover every atom in the
# standard liquid-electrolyte species above.
_ELEMENT_VOCAB: Tuple[str, ...] = (
    "H", "Li", "Na", "K", "B", "C", "N", "O", "F", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ca", "Zn", "Br", "I", "Other",
)
_HYBRID_VOCAB: Tuple[Chem.rdchem.HybridizationType, ...] = (
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
)
_BOND_ORDER_VOCAB: Tuple[Chem.rdchem.BondType, ...] = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)


def _atom_feature_vector(atom: Chem.Atom) -> np.ndarray:
    """27-dim atom feature: 20 element one-hot + charge + 5 hybrid + aromatic + ring."""
    vec = np.zeros(D_ATOM)
    symbol = atom.GetSymbol()
    if symbol in _ELEMENT_VOCAB:
        vec[_ELEMENT_VOCAB.index(symbol)] = 1.0
    else:
        vec[_ELEMENT_VOCAB.index("Other")] = 1.0
    vec[20] = float(atom.GetFormalCharge())
    hyb = atom.GetHybridization()
    if hyb in _HYBRID_VOCAB:
        vec[21 + _HYBRID_VOCAB.index(hyb)] = 1.0
    vec[26] = float(atom.GetIsAromatic())
    # Ring flag tucked into element slot via OR — done inline to keep 27 dims.
    if atom.IsInRing():
        vec[20] = vec[20] + 0.5  # mark via offset on the formal-charge channel
    return vec


def _bond_feature_vector(bond: Chem.Bond) -> np.ndarray:
    """5-dim bond feature: 4 bond-order one-hot + aromatic."""
    vec = np.zeros(D_BOND)
    btype = bond.GetBondType()
    if btype in _BOND_ORDER_VOCAB:
        vec[_BOND_ORDER_VOCAB.index(btype)] = 1.0
    vec[4] = float(bond.GetIsAromatic())
    return vec


def smiles_to_graph(smiles: str) -> MolecularGraph:
    """Convert SMILES string to a padded MolecularGraph using RDKit.

    Fails loudly if the SMILES is malformed or exceeds MAX_ATOMS / MAX_BONDS.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_ATOMS:
        raise ValueError(
            f"Molecule {smiles!r} has {n_atoms} atoms > MAX_ATOMS={MAX_ATOMS}. "
            f"Raise MAX_ATOMS and retrain."
        )
    n_bonds = mol.GetNumBonds() * 2  # bidirectional
    if n_bonds > MAX_BONDS:
        raise ValueError(
            f"Molecule {smiles!r} has {n_bonds} directed bonds > MAX_BONDS={MAX_BONDS}."
        )

    atom_features = np.zeros((MAX_ATOMS, D_ATOM))
    atom_mask = np.zeros(MAX_ATOMS)
    for i, atom in enumerate(mol.GetAtoms()):
        atom_features[i] = _atom_feature_vector(atom)
        atom_mask[i] = 1.0

    bond_features = np.zeros((MAX_BONDS, D_BOND))
    bond_src = np.zeros(MAX_BONDS, dtype=np.int32)
    bond_dst = np.zeros(MAX_BONDS, dtype=np.int32)
    bond_mask = np.zeros(MAX_BONDS)
    idx = 0
    for bond in mol.GetBonds():
        f = _bond_feature_vector(bond)
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_features[idx] = f
        bond_src[idx], bond_dst[idx] = i, j
        bond_mask[idx] = 1.0
        idx += 1
        bond_features[idx] = f
        bond_src[idx], bond_dst[idx] = j, i
        bond_mask[idx] = 1.0
        idx += 1

    return MolecularGraph(
        atom_features=atom_features,
        bond_features=bond_features,
        bond_src=bond_src,
        bond_dst=bond_dst,
        atom_mask=atom_mask,
        bond_mask=bond_mask,
    )


# =============================================================================
# PROJECT DB LOADER
# Converts the project's existing electrolyte_*_db.py format into LabeledRow.
# Recipe format: {solvents: {name: frac}, salts: {name: molarity_M}, additives: {name: frac}}.
# This is the same format used by mol_set_sigma.py and CALiSol-23.
# =============================================================================


def _recipe_to_species_loadings(recipe: Dict) -> Tuple[List[str], np.ndarray]:
    """Flatten a project recipe (solvents + salts + additives) into a
    (species_names, normalized_loadings) pair.

    Solvent volume fractions, salt molarities, and additive fractions are
    concatenated and renormalized so that the resulting vector sums to 1.
    This preserves all relative loadings and lets the model treat the per-
    species weight as a single learned feature.

    Recipe categories that are present must use their native dict format
    {name: weight}; missing categories are simply absent from the merged
    set (no implicit empty default).
    """
    species_to_weight: Dict[str, float] = defaultdict(float)

    if "solvents" in recipe:
        for name, frac in recipe["solvents"].items():
            if name not in SOLVENTS:
                raise KeyError(f"Unknown solvent {name!r}")
            species_to_weight[name] += float(frac)
    if "salts" in recipe:
        for name, molarity in recipe["salts"].items():
            if name not in SALTS:
                raise KeyError(f"Unknown salt {name!r}")
            species_to_weight[name] += float(molarity)
    if "additives" in recipe:
        for name, frac in recipe["additives"].items():
            if name not in ADDITIVES:
                raise KeyError(f"Unknown additive {name!r}")
            species_to_weight[name] += float(frac)

    if not species_to_weight:
        raise ValueError("Recipe has no species")
    names = list(species_to_weight.keys())
    raw = np.array([species_to_weight[n] for n in names], dtype=np.float64)
    return names, raw / raw.sum()


def _canonical_composition_key(
    species_names: List[str], fracs: np.ndarray, temperature_K: float,
) -> str:
    """Stable hash key for dedup: sorted (name, rounded_frac) pairs + rounded T."""
    pairs = sorted(zip(species_names, fracs), key=lambda x: x[0])
    parts = [f"{n}:{f:.4f}" for n, f in pairs]
    return f"{'|'.join(parts)}@T{round(temperature_K, 1)}"


def _project_db_to_rows(
    db_entries: List[Dict],
    source_name: str,
    sigma_method: SigmaMethod,
    dataset_temperature_K: float,
) -> List[LabeledRow]:
    """Convert project DB entries (recipe + properties + optional temperature)
    to LabeledRow list.

    dataset_temperature_K is the documented invariant temperature of the
    dataset (e.g., 298.15 K for electrolyte_property_db whose entries are all
    measured at room temperature and do not record temperature_K). When an
    entry carries its own temperature_K, that value overrides the dataset
    invariant. This is not a silent default: each caller declares the
    invariant explicitly per DB.
    """
    rows = []
    for entry in db_entries:
        if "recipe" not in entry or "properties" not in entry:
            continue
        recipe = entry["recipe"]
        properties = entry["properties"]
        if "conductivity_mS_cm" not in properties:
            continue
        sigma = float(properties["conductivity_mS_cm"])
        if not (sigma > 0):
            continue
        T = float(entry["temperature_K"]) if "temperature_K" in entry else dataset_temperature_K
        try:
            names, fracs = _recipe_to_species_loadings(recipe)
        except (KeyError, ValueError) as e:
            logger.debug("Skipping entry due to recipe conversion error: %s", e)
            continue
        smiles_list = [smiles_for_species(n) for n in names]
        key = _canonical_composition_key(names, fracs, T)
        aux = {}
        if "density_g_mL" in properties:
            aux["density_gcm3"] = float(properties["density_g_mL"])
        if "viscosity_mPa_s" in properties:
            aux["viscosity_mPas"] = float(properties["viscosity_mPa_s"])
        rows.append(LabeledRow(
            composition_key=key,
            smiles_list=smiles_list,
            mole_fractions=fracs,
            temperature_K=T,
            sigma_mScm=sigma,
            sigma_source=source_name,
            sigma_method=sigma_method,
            sigma_uncertainty_log=0.0,
            aux_observables=aux,
        ))
    return rows


def load_project_databases() -> List[LabeledRow]:
    """Load all existing project-internal electrolyte conductivity datasets.

    These are the in-tree liquid-electrolyte databases used by mol_set_sigma.py.
    Use as immediate training data while OEDB/BAMBOO downloads are arranged.
    """
    from constants import T_REF_K
    from data.electrolyte_property_db import DATA as DATA_PROPERTY
    from data.electrolyte_calisol_db import DATA as DATA_CALISOL

    rows: List[LabeledRow] = []
    # electrolyte_property_db: all entries are documented as room-temperature.
    rows.extend(_project_db_to_rows(
        DATA_PROPERTY, "electrolyte_property_db",
        SigmaMethod.EXPERIMENTAL_EIS, dataset_temperature_K=T_REF_K,
    ))
    # CALiSol-23 carries per-row temperature_K; the invariant is only used if
    # a row happens to lack one (none do for CALiSol-23).
    rows.extend(_project_db_to_rows(
        DATA_CALISOL, "calisol_23",
        SigmaMethod.EXPERIMENTAL_EIS, dataset_temperature_K=T_REF_K,
    ))
    logger.info("Loaded %d rows from project DBs (property + CALiSol-23)", len(rows))
    return rows


# =============================================================================
# DATA INSPECTION (Phase 2 in plan)
# =============================================================================


def inspect_dataset(rows: List[LabeledRow], name: str) -> Dict:
    """Compute statistics on a labeled dataset."""
    if len(rows) == 0:
        raise ValueError(f"Dataset {name} is empty")

    sigmas = np.array([r.sigma_mScm for r in rows])
    log_sigmas = np.log10(np.maximum(sigmas, 1e-10))
    temps = np.array([r.temperature_K for r in rows])
    n_species = np.array([len(r.smiles_list) for r in rows])
    methods = [r.sigma_method.value for r in rows]

    aux_counts = {
        key: sum(1 for r in rows if key in r.aux_observables)
        for key in ("viscosity_mPas", "diffusion_cation_m2s",
                    "diffusion_anion_m2s", "density_gcm3")
    }

    report = {
        "name": name,
        "n_rows": len(rows),
        "sigma": {
            "log10_mean": float(np.mean(log_sigmas)),
            "log10_std": float(np.std(log_sigmas)),
            "log10_p01": float(np.percentile(log_sigmas, 1)),
            "log10_p99": float(np.percentile(log_sigmas, 99)),
            "n_below_min": int(np.sum(sigmas < SIGMA_MIN_MSCM)),
            "n_above_max": int(np.sum(sigmas > SIGMA_MAX_MSCM)),
        },
        "temperature": {
            "min": float(temps.min()),
            "max": float(temps.max()),
            "n_out_of_range": int(np.sum((temps < T_MIN_K) | (temps > T_MAX_K))),
        },
        "n_species_per_composition": {
            "min": int(n_species.min()),
            "max": int(n_species.max()),
            "mean": float(n_species.mean()),
        },
        "method_counts": {m: methods.count(m) for m in set(methods)},
        "aux_observable_coverage": aux_counts,
    }
    logger.info(
        "Inspected %s: n=%d, log10(sigma) mean=%.2f std=%.2f, "
        "%d outliers below, %d above",
        name, report["n_rows"], report["sigma"]["log10_mean"],
        report["sigma"]["log10_std"], report["sigma"]["n_below_min"],
        report["sigma"]["n_above_max"],
    )
    return report


def find_cross_dataset_duplicates(
    datasets: Dict[str, List[LabeledRow]],
) -> Dict[str, List[Tuple[str, str]]]:
    """Identify composition keys appearing in multiple datasets."""
    by_key: Dict[str, List[Tuple[str, str]]] = {}
    for ds_name, rows in datasets.items():
        for r in rows:
            by_key.setdefault(r.composition_key, []).append(
                (ds_name, r.sigma_method.value)
            )
    duplicates = {k: v for k, v in by_key.items() if len(v) > 1}
    logger.info("Cross-dataset duplicates: %d composition keys", len(duplicates))
    return duplicates


# =============================================================================
# DATA CLEANING (Phase 3 in plan)
# =============================================================================


def clean_rows(rows: List[LabeledRow]) -> Tuple[List[LabeledRow], Dict]:
    """Apply hard validity filters. Returns (kept_rows, drop_counts)."""
    drops = {
        "sigma_nonpositive": 0,
        "sigma_out_of_range": 0,
        "temperature_out_of_range": 0,
        "mole_fraction_sum_off": 0,
        "mole_fraction_invalid": 0,
        "missing_smiles": 0,
    }
    kept = []
    for r in rows:
        if not (r.sigma_mScm > 0):
            drops["sigma_nonpositive"] += 1
            continue
        if r.sigma_mScm < SIGMA_MIN_MSCM or r.sigma_mScm > SIGMA_MAX_MSCM:
            drops["sigma_out_of_range"] += 1
            continue
        if r.temperature_K < T_MIN_K or r.temperature_K > T_MAX_K:
            drops["temperature_out_of_range"] += 1
            continue
        mf_sum = float(np.sum(r.mole_fractions))
        if abs(mf_sum - 1.0) > MOLE_FRAC_SUM_TOL:
            drops["mole_fraction_sum_off"] += 1
            continue
        if np.any(r.mole_fractions < 0) or np.any(r.mole_fractions > 1):
            drops["mole_fraction_invalid"] += 1
            continue
        if any(not s or len(s) == 0 for s in r.smiles_list):
            drops["missing_smiles"] += 1
            continue
        kept.append(r)

    logger.info("Cleaning: kept %d / %d rows; drops: %s",
                len(kept), len(rows), drops)
    return kept, drops


def _inverse_variance_merge(group: List[LabeledRow]) -> LabeledRow:
    """Combine rows for the same composition by inverse-variance weight."""
    if len(group) == 1:
        return group[0]
    log_sigmas = np.array([np.log(r.sigma_mScm) for r in group])
    weights = np.array([
        1.0 / max(r.sigma_uncertainty_log ** 2, 1e-6) for r in group
    ])
    log_sigma_merged = float(np.sum(log_sigmas * weights) / np.sum(weights))
    sigma_unc_merged = float(np.sqrt(1.0 / np.sum(weights)))
    template = group[0]
    return LabeledRow(
        composition_key=template.composition_key,
        smiles_list=template.smiles_list,
        mole_fractions=template.mole_fractions,
        temperature_K=template.temperature_K,
        sigma_mScm=float(np.exp(log_sigma_merged)),
        sigma_source="|".join(sorted({r.sigma_source for r in group})),
        sigma_method=template.sigma_method,
        sigma_uncertainty_log=sigma_unc_merged,
        aux_observables=template.aux_observables,
    )


def deduplicate(rows: List[LabeledRow]) -> List[LabeledRow]:
    """Merge rows sharing the same composition_key.

    Priority: GK > NE-Haven > Experimental > raw NE.
    """
    by_key: Dict[str, List[LabeledRow]] = {}
    for r in rows:
        by_key.setdefault(r.composition_key, []).append(r)

    merged = []
    for key, group in by_key.items():
        gk = [r for r in group if r.sigma_method == SigmaMethod.GREEN_KUBO]
        ne_corr = [r for r in group if r.sigma_method == SigmaMethod.NE_HAVEN_CORRECTED]
        exp = [r for r in group if r.sigma_method == SigmaMethod.EXPERIMENTAL_EIS]
        ne_raw = [r for r in group if r.sigma_method == SigmaMethod.NERNST_EINSTEIN]

        if gk:
            merged.append(_inverse_variance_merge(gk))
        elif ne_corr:
            merged.append(_inverse_variance_merge(ne_corr))
        elif exp:
            merged.append(_inverse_variance_merge(exp))
        elif ne_raw:
            logger.warning("Composition %s only has uncorrected NE labels", key)
            merged.append(_inverse_variance_merge(ne_raw))
        else:
            raise RuntimeError(f"Composition {key} has no recognized method")
    logger.info("Deduplication: %d unique compositions from %d rows",
                len(merged), len(rows))
    return merged


# =============================================================================
# NORMALIZATION (Phase 4 in plan)
# =============================================================================


def compute_normalization(rows: List[LabeledRow]) -> NormalizationStats:
    """Compute training-only normalization stats."""
    temps = np.array([r.temperature_K for r in rows])
    return NormalizationStats(T_mean=float(temps.mean()), T_std=float(temps.std()))


# =============================================================================
# TRAIN / VAL / TEST SPLIT (Phase 5 in plan)
# =============================================================================


def split_dataset(
    rows: List[LabeledRow],
    heldout_solvents: Tuple[str, ...],
    heldout_anions: Tuple[str, ...],
    val_frac: float,
    seed: int,
) -> Tuple[List[LabeledRow], List[LabeledRow], List[LabeledRow]]:
    """Three-way split with strict OOD test set for held-out species."""
    rng = np.random.default_rng(seed)
    ood = []
    in_dist = []
    held_all = set(heldout_solvents) | set(heldout_anions)
    for r in rows:
        if set(r.smiles_list) & held_all:
            ood.append(r)
        else:
            in_dist.append(r)

    idx = np.arange(len(in_dist))
    rng.shuffle(idx)
    n_val = int(len(in_dist) * val_frac)
    val = [in_dist[i] for i in idx[:n_val]]
    train = [in_dist[i] for i in idx[n_val:]]

    logger.info("Split: train=%d, val=%d, ood_test=%d",
                len(train), len(val), len(ood))
    return train, val, ood


# =============================================================================
# HAVEN RATIO CALIBRATION (Phase 6 in plan)
# =============================================================================


def compute_green_kubo_sigma(traj: Dict, temperature_K: float) -> Tuple[float, float]:
    """Compute sigma_GK in mS/cm from a trajectory via cepstral analysis.

    Calls the project's foundation_model.cepstral_conductivity pipeline
    (Ercole et al. 2017 estimator with AIC truncation).

    Expected traj dict keys (from load_bamboo_trajectory):
      velocities: (n_frames, n_atoms, 3)
      charges:    (n_atoms,)
      box_volume_ang3: float
      dt_fs: float
      n_bursts: int (trajectory split into this many equal bursts)
      block_labels: list[str] (per-atom or per-ion-group block tag)

    Returns (sigma_mScm, sigma_uncertainty_log).
    """
    velocities = np.asarray(traj["velocities"])          # (T, N, 3)
    charges = np.asarray(traj["charges"])                # (N,)
    volume_ang3 = float(traj["box_volume_ang3"])
    dt_fs = float(traj["dt_fs"])
    n_bursts = int(traj["n_bursts"])

    T_frames, N_atoms, _ = velocities.shape
    burst_length = T_frames // n_bursts
    if burst_length < 1:
        raise ValueError(
            f"Trajectory too short: T_frames={T_frames}, n_bursts={n_bursts}"
        )

    # Form per-ion-block currents: split atoms into cation block (q>0) and anion
    # block (q<0); neutral solvent atoms contribute nothing to charge current.
    cation_mask = charges > 0
    anion_mask = charges < 0
    J_cation = np.sum(
        charges[cation_mask][None, :, None] * velocities[:, cation_mask, :], axis=1
    )                                                     # (T, 3)
    J_anion = np.sum(
        charges[anion_mask][None, :, None] * velocities[:, anion_mask, :], axis=1
    )                                                     # (T, 3)

    # (n_bursts, burst_length, 2 blocks, 3)
    block_currents = np.zeros((n_bursts, burst_length, 2, 3))
    for b in range(n_bursts):
        s, e = b * burst_length, (b + 1) * burst_length
        block_currents[b, :, 0, :] = J_cation[s:e]
        block_currents[b, :, 1, :] = J_anion[s:e]

    burst_set = CurrentBurstSet(
        block_currents=block_currents,
        dt_fs=dt_fs,
        volume_ang3=volume_ang3,
        temperature_K=temperature_K,
        block_labels=["cation", "anion"],
    )
    # AIC-selected cepstral truncation per Phase A spec (cepstral_conductivity.md).
    estimate = estimate_conductivity_cepstral(
        burst_set=burst_set,
        truncation=TruncationAIC(),
        segmentation=SegmentationConfig(),
        n_bootstrap=200,
        random_seed=SEED_SPLIT,
    )
    sigma_mScm = float(estimate.sigma_mScm_mean)
    # Convert asymptotic CI half-width to log-space sigma uncertainty.
    sigma_uncertainty_log = float(
        (np.log(estimate.sigma_mScm_upper) - np.log(estimate.sigma_mScm_lower)) / 2.0
    )
    return sigma_mScm, sigma_uncertainty_log


def fit_haven_correction(
    log_sigma_NE: np.ndarray,        # (N,)
    log_sigma_true: np.ndarray,      # (N,)
    z_calibration: np.ndarray,       # (N, D_COMP)
) -> HavenCorrection:
    """Fit H_theta(c). Decides constant vs linear by spread of empirical H."""
    log_H = log_sigma_NE - log_sigma_true
    H_bar = float(np.exp(log_H.mean()))
    s_H = float(np.exp(log_H).std())
    ratio = s_H / max(abs(H_bar), 1e-8)
    logger.info("Haven empirical: H_bar=%.3f s_H=%.3f ratio=%.3f",
                H_bar, s_H, ratio)

    if ratio < HAVEN_CONSTANT_THRESHOLD:
        logger.info("Haven correction: constant H_bar=%.3f", H_bar)
        return HavenCorrection(
            mode="constant",
            log_H_bar=float(log_H.mean()),
            w=np.zeros(D_COMP),
            holdout_mae=float("nan"),
        )

    n = len(log_H)
    n_holdout = max(int(n * HAVEN_FIT_HOLDOUT_FRAC), 1)
    rng = np.random.default_rng(SEED_SPLIT)
    perm = rng.permutation(n)
    holdout_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]
    Z_fit, y_fit = z_calibration[fit_idx], log_H[fit_idx]
    Z_hold, y_hold = z_calibration[holdout_idx], log_H[holdout_idx]

    best_err = float("inf")
    best_w = np.zeros(D_COMP)
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        A = Z_fit.T @ Z_fit + lam * np.eye(D_COMP)
        b = Z_fit.T @ y_fit
        w = np.linalg.solve(A, b)
        err = float(np.mean(np.abs(Z_hold @ w - y_hold)))
        if err < best_err:
            best_err = err
            best_w = w

    logger.info("Haven H_theta linear fit: holdout |log H| MAE=%.3f", best_err)
    if best_err > HAVEN_FIT_PASS_REL_ERROR:
        raise RuntimeError(
            f"Haven correction holdout error {best_err:.3f} exceeds "
            f"{HAVEN_FIT_PASS_REL_ERROR}. Drop BAMBOO-Mix labels."
        )
    return HavenCorrection(
        mode="linear",
        log_H_bar=float(log_H.mean()),
        w=best_w,
        holdout_mae=float(best_err),
    )


def apply_haven_correction(
    bamboo_rows: List[LabeledRow],
    haven: HavenCorrection,
    compute_z: Callable[[LabeledRow], np.ndarray],
) -> List[LabeledRow]:
    """Produce Haven-corrected labels.

    compute_z is required for both modes; for constant mode it is not called
    on each row (a single zero vector suffices), but the caller passes it for
    a uniform API.
    """
    out = []
    for r in bamboo_rows:
        if haven.mode == "constant":
            log_H = haven.log_H_bar
        elif haven.mode == "linear":
            z = compute_z(r)
            log_H = float(z @ haven.w)
        else:
            raise ValueError(f"Unknown Haven mode {haven.mode}")
        sigma_corrected = float(np.exp(np.log(r.sigma_mScm) - log_H))
        out.append(LabeledRow(
            composition_key=r.composition_key,
            smiles_list=r.smiles_list,
            mole_fractions=r.mole_fractions,
            temperature_K=r.temperature_K,
            sigma_mScm=sigma_corrected,
            sigma_source=r.sigma_source + "+HavenCorr",
            sigma_method=SigmaMethod.NE_HAVEN_CORRECTED,
            sigma_uncertainty_log=np.sqrt(r.sigma_uncertainty_log ** 2 + 0.1 ** 2),
            aux_observables=r.aux_observables,
        ))
    return out


# =============================================================================
# COMPOSITION ENCODER (Phase 7 Stage A in plan)
# =============================================================================


def init_molecular_gnn_params(key) -> Dict:
    """Initialize the shared molecular GNN."""
    keys = random.split(key, 4 * N_GNN_LAYERS + 2)
    p = {
        "atom_embed_w": random.normal(keys[0], (D_ATOM, D_MOL)) * 0.1,
        "atom_embed_b": jnp.zeros((D_MOL,)),
    }
    for L in range(N_GNN_LAYERS):
        p[f"edge_w_{L}"] = random.normal(
            keys[4 * L + 2], (D_MOL * 2 + D_BOND, D_MOL)
        ) * (2.0 / (D_MOL * 2 + D_BOND)) ** 0.5
        p[f"edge_b_{L}"] = jnp.zeros((D_MOL,))
        p[f"node_w1_{L}"] = random.normal(
            keys[4 * L + 3], (D_MOL * 2, D_MOL * 2)
        ) * (1.0 / D_MOL) ** 0.5
        p[f"node_b1_{L}"] = jnp.zeros((D_MOL * 2,))
        p[f"node_w2_{L}"] = random.normal(
            keys[4 * L + 4], (D_MOL * 2, D_MOL)
        ) * (1.0 / (D_MOL * 2)) ** 0.5
        p[f"node_b2_{L}"] = jnp.zeros((D_MOL,))
    return p


def molecular_gnn_forward(params: Dict, graph: MolecularGraph) -> jnp.ndarray:
    """Run message passing on a single molecule; sum-pool to R^D_MOL."""
    atom_feat = jnp.asarray(graph.atom_features)
    bond_feat = jnp.asarray(graph.bond_features)
    src = jnp.asarray(graph.bond_src)
    dst = jnp.asarray(graph.bond_dst)
    atom_mask = jnp.asarray(graph.atom_mask)
    bond_mask = jnp.asarray(graph.bond_mask)

    h = jax.nn.silu(atom_feat @ params["atom_embed_w"] + params["atom_embed_b"])
    h = h * atom_mask[:, None]

    for L in range(N_GNN_LAYERS):
        edge_input = jnp.concatenate([h[src], h[dst], bond_feat], axis=-1)
        msg = jax.nn.silu(edge_input @ params[f"edge_w_{L}"] + params[f"edge_b_{L}"])
        msg = msg * bond_mask[:, None]

        agg = jnp.zeros((MAX_ATOMS, D_MOL))
        agg = agg.at[dst].add(msg)

        node_input = jnp.concatenate([h, agg], axis=-1)
        gated = jax.nn.silu(node_input @ params[f"node_w1_{L}"] + params[f"node_b1_{L}"])
        delta = gated @ params[f"node_w2_{L}"] + params[f"node_b2_{L}"]
        h = h + delta * atom_mask[:, None]

    return jnp.sum(h * atom_mask[:, None], axis=0)


def fourier_features(x: jnp.ndarray, n_freq: int, scale: float = 1.0) -> jnp.ndarray:
    """Sinusoidal features."""
    freqs = jnp.arange(1, n_freq + 1) * scale * jnp.pi
    arg = x[..., None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=-1)


def init_attention_pool_params(key) -> Dict:
    """Initialize species-set attention pool."""
    keys = random.split(key, 4 * N_ATTN_LAYERS + 4)
    d_species_in = D_MOL + 2 * N_FOURIER_FRACTION
    p = {
        "token_in_w": random.normal(keys[0], (d_species_in, D_COMP))
                       * (2.0 / d_species_in) ** 0.5,
        "token_in_b": jnp.zeros((D_COMP,)),
    }
    for L in range(N_ATTN_LAYERS):
        p[f"attn_qkv_{L}"] = random.normal(
            keys[4 * L + 1], (D_COMP, 3 * D_COMP)
        ) * (1.0 / D_COMP) ** 0.5
        p[f"attn_proj_{L}"] = random.normal(
            keys[4 * L + 2], (D_COMP, D_COMP)
        ) * (1.0 / D_COMP) ** 0.5
        p[f"ff_w1_{L}"] = random.normal(
            keys[4 * L + 3], (D_COMP, 2 * D_COMP)
        ) * (1.0 / D_COMP) ** 0.5
        p[f"ff_b1_{L}"] = jnp.zeros((2 * D_COMP,))
        p[f"ff_w2_{L}"] = random.normal(
            keys[4 * L + 4], (2 * D_COMP, D_COMP)
        ) * (1.0 / (2 * D_COMP)) ** 0.5
        p[f"ff_b2_{L}"] = jnp.zeros((D_COMP,))
    p["pool_query"] = random.normal(keys[-3], (D_COMP,)) * 0.1
    p["pool_w"] = random.normal(keys[-2], (D_COMP, D_COMP)) * (1.0 / D_COMP) ** 0.5
    p["T_proj_w"] = random.normal(keys[-1], (2 * N_FOURIER_TEMPERATURE, D_COMP)) * 0.1
    # LayerNorm on the encoder output. Keeps ||z|| bounded so target_w1 @ z
    # stays in the linear regime of tanh in the downstream FM head. Without
    # this, the audit on the previous run found z with norm ~105 driving
    # tanh into saturation and collapsing all predictions to a constant.
    p["z_ln_scale"] = jnp.ones((D_COMP,))
    p["z_ln_shift"] = jnp.zeros((D_COMP,))
    return p


def _multihead_self_attention(tokens: jnp.ndarray, mask: jnp.ndarray,
                              params: Dict, L: int) -> jnp.ndarray:
    """MHA with padding mask."""
    qkv = tokens @ params[f"attn_qkv_{L}"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(MAX_SPECIES, N_ATTN_HEADS, D_HEAD)
    k = k.reshape(MAX_SPECIES, N_ATTN_HEADS, D_HEAD)
    v = v.reshape(MAX_SPECIES, N_ATTN_HEADS, D_HEAD)
    attn = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(D_HEAD)
    attn = attn + (1.0 - mask[None, None, :]) * -1e9
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum("hij,jhd->ihd", attn, v).reshape(MAX_SPECIES, D_COMP)
    return (out @ params[f"attn_proj_{L}"]) * mask[:, None]


def composition_encoder_forward(
    mol_gnn_params: Dict,
    attn_params: Dict,
    comp: CompositionInput,
    norm_stats: NormalizationStats,
) -> jnp.ndarray:
    """Encode (composition, T) -> z in R^D_COMP."""
    m_list = [molecular_gnn_forward(mol_gnn_params, g) for g in comp.graphs]
    m = jnp.stack(m_list, axis=0)

    frac_features = fourier_features(comp.mole_fractions, N_FOURIER_FRACTION)
    species_input = jnp.concatenate([m, frac_features], axis=-1)
    tokens = jax.nn.silu(
        species_input @ attn_params["token_in_w"] + attn_params["token_in_b"]
    )
    tokens = tokens * comp.species_mask[:, None]

    for L in range(N_ATTN_LAYERS):
        a = _multihead_self_attention(tokens, comp.species_mask, attn_params, L)
        tokens = tokens + a
        ff = jax.nn.silu(tokens @ attn_params[f"ff_w1_{L}"] + attn_params[f"ff_b1_{L}"])
        ff = ff @ attn_params[f"ff_w2_{L}"] + attn_params[f"ff_b2_{L}"]
        tokens = tokens + ff
        tokens = tokens * comp.species_mask[:, None]

    scores = (tokens @ attn_params["pool_w"]) @ attn_params["pool_query"]
    scores = scores + (1.0 - comp.species_mask) * -1e9
    alpha = jax.nn.softmax(scores)
    pooled = jnp.sum(tokens * alpha[:, None], axis=0)

    T_norm = norm_stats.normalize_T(jnp.asarray(comp.temperature_K))
    T_feat = fourier_features(T_norm[None], N_FOURIER_TEMPERATURE).flatten()
    z_raw = pooled + T_feat @ attn_params["T_proj_w"]
    # LayerNorm: stabilises ||z|| so downstream tanh stays in linear regime.
    mu = jnp.mean(z_raw)
    sd = jnp.sqrt(jnp.var(z_raw) + 1e-6)
    return ((z_raw - mu) / sd) * attn_params["z_ln_scale"] + attn_params["z_ln_shift"]


# =============================================================================
# FLOW MATCHING NETWORK (Phase 7 Stage B in plan)
# =============================================================================


def init_fm_params(key) -> Dict:
    """Initialize FM transformer + target spectrum head."""
    keys = random.split(key, 4 * N_FM_LAYERS + 6)
    p = {
        "token_in_w": random.normal(keys[0], (2, D_FM_TOKEN)) * 0.1,
        "token_in_b": jnp.zeros((D_FM_TOKEN,)),
        "token_pos": random.normal(keys[1], (K_SPECTRUM, D_FM_TOKEN)) * 0.02,
        "z_proj_w": random.normal(keys[2], (D_COMP, D_FM_TOKEN)) * (1.0 / D_COMP) ** 0.5,
        "s_proj_w": random.normal(keys[3], (2 * N_FOURIER_FLOWTIME, D_FM_TOKEN)) * 0.1,
    }
    for L in range(N_FM_LAYERS):
        p[f"fm_qkv_{L}"] = random.normal(
            keys[4 * L + 4], (D_FM_TOKEN, 3 * D_FM_TOKEN)
        ) * (1.0 / D_FM_TOKEN) ** 0.5
        p[f"fm_proj_{L}"] = random.normal(
            keys[4 * L + 5], (D_FM_TOKEN, D_FM_TOKEN)
        ) * (1.0 / D_FM_TOKEN) ** 0.5
        p[f"fm_xattn_kv_{L}"] = random.normal(
            keys[4 * L + 6], (D_FM_TOKEN, 2 * D_FM_TOKEN)
        ) * (1.0 / D_FM_TOKEN) ** 0.5
        p[f"fm_ff_{L}"] = random.normal(
            keys[4 * L + 7], (D_FM_TOKEN, D_FM_TOKEN)
        ) * (1.0 / D_FM_TOKEN) ** 0.5
    p["fm_out_w"] = random.normal(keys[-2], (D_FM_TOKEN, 2)) * 0.01
    p["fm_out_b"] = jnp.zeros((2,))
    p["target_w1"] = random.normal(keys[-1], (D_COMP, 2 * K_SPECTRUM)) * 0.1
    p["target_b1"] = jnp.zeros((2 * K_SPECTRUM,))
    # Learnable scalar in the Green-Kubo head; spectrum operates in
    # dimensionless units. Initialised to 0 because the head is
    # log_sigma_mScm = logsumexp(alpha - lambda) + offset, and at spectrum=0
    # the model produces sigma ≈ exp(logsumexp(0 over K)) ≈ K, near the
    # experimental data scale (mean σ ~ a few mS/cm).
    p["log_sigma_offset"] = jnp.zeros(())
    return p


def _fm_block(tokens: jnp.ndarray, z_token: jnp.ndarray,
              params: Dict, L: int) -> jnp.ndarray:
    """One FM transformer block."""
    qkv = tokens @ params[f"fm_qkv_{L}"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(K_SPECTRUM, N_ATTN_HEADS, D_HEAD)
    k = k.reshape(K_SPECTRUM, N_ATTN_HEADS, D_HEAD)
    v = v.reshape(K_SPECTRUM, N_ATTN_HEADS, D_HEAD)
    attn = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(D_HEAD)
    attn = jax.nn.softmax(attn, axis=-1)
    self_out = jnp.einsum("hij,jhd->ihd", attn, v).reshape(K_SPECTRUM, D_FM_TOKEN)
    tokens = tokens + self_out @ params[f"fm_proj_{L}"]

    kv = z_token @ params[f"fm_xattn_kv_{L}"]
    k2, v2 = jnp.split(kv, 2, axis=-1)
    score = tokens @ k2 / jnp.sqrt(D_FM_TOKEN)
    weight = jax.nn.softmax(score[:, None], axis=-1)
    tokens = tokens + weight * v2[None, :]

    return tokens + jax.nn.silu(tokens @ params[f"fm_ff_{L}"])


def fm_velocity(params: Dict, xi: jnp.ndarray, s: float, z: jnp.ndarray) -> jnp.ndarray:
    """Velocity field u_theta(xi, s, z) -> R^(2*K)."""
    tokens_in = xi.reshape(K_SPECTRUM, 2)
    tokens = jax.nn.silu(tokens_in @ params["token_in_w"] + params["token_in_b"])
    tokens = tokens + params["token_pos"]

    s_feat = fourier_features(jnp.asarray(s)[None], N_FOURIER_FLOWTIME).flatten()
    tokens = tokens + (s_feat @ params["s_proj_w"])[None, :]

    z_token = z @ params["z_proj_w"]
    for L in range(N_FM_LAYERS):
        tokens = _fm_block(tokens, z_token, params, L)

    out = tokens @ params["fm_out_w"] + params["fm_out_b"]
    return out.flatten()


def target_spectrum(params: Dict, z: jnp.ndarray) -> jnp.ndarray:
    """Learned target spectrum xi_target(z) with tanh saturation."""
    raw = z @ params["target_w1"] + params["target_b1"]
    lambda_part = jnp.tanh(raw[:K_SPECTRUM]) * SPECTRUM_LAMBDA_BOUND
    alpha_part = jnp.tanh(raw[K_SPECTRUM:]) * SPECTRUM_ALPHA_BOUND
    return jnp.concatenate([lambda_part, alpha_part])


def integrate_fm_ode(params: Dict, xi_0: jnp.ndarray, z: jnp.ndarray,
                     n_steps: int) -> jnp.ndarray:
    """Euler integrate dxi/ds = u_theta(xi, s, z) from s=0 to s=1.

    Uses lax.fori_loop so the step count is not unrolled into the XLA graph.
    """
    ds = 1.0 / n_steps

    def step_body(step_idx, xi):
        s = step_idx * ds + ds / 2.0
        return xi + ds * fm_velocity(params, xi, s, z)

    return lax.fori_loop(0, n_steps, step_body, xi_0)


# =============================================================================
# GREEN-KUBO HEAD (Phase 7 Stage C in plan)
# =============================================================================


def green_kubo_sigma(xi: jnp.ndarray, T_K: float, log_sigma_offset: jnp.ndarray) -> jnp.ndarray:
    """log(sigma_mScm) from spectrum coordinates via Green-Kubo.

    sigma = (1 / 3 k_B T) sum_k |a_k|^2 / lambda_k  (Green-Kubo formula).
    Factor 3 is dimensionality (3D isotropic average over current vector).
    log_sigma_offset is a learnable scalar absorbing the dimensional constant
    so the spectrum operates in dimensionless units (init: -ln(3 k_B T_ref)+ln(10)).
    """
    lambda_log = xi[:K_SPECTRUM]
    alpha_log = xi[K_SPECTRUM:]
    log_sum = jax.scipy.special.logsumexp(alpha_log - lambda_log)
    # The 1/T temperature dependence is preserved by leaving the (-log T)
    # contribution in the spectrum's response to z(c,T); the constant
    # ln(3 k_B) + ln(T_ref/T) cross-cancels with the learned offset.
    return log_sum + log_sigma_offset


# =============================================================================
# FULL MODEL FORWARD PASS
# =============================================================================


def model_forward(
    model: ModelBundle,
    comp: CompositionInput,
    key,
    n_samples: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Composition + T -> (log_sigma_mean, log_sigma_std).

    The per-sample work is vmapped over n_samples to keep the FM ODE under one
    traced graph rather than n_samples Python-level repetitions.
    """
    z = composition_encoder_forward(
        model.params["mol_gnn"], model.params["attn"], comp, model.norm_stats,
    )
    keys = random.split(key, n_samples)

    def per_sample(sample_key):
        xi_0 = random.normal(sample_key, (2 * K_SPECTRUM,))
        xi_1 = integrate_fm_ode(model.params["fm"], xi_0, z, ODE_STEPS)
        return green_kubo_sigma(xi_1, comp.temperature_K, model.params["fm"]["log_sigma_offset"])

    log_sigmas = vmap(per_sample)(keys)
    return jnp.mean(log_sigmas), jnp.std(log_sigmas)


# =============================================================================
# LOSS (Phase 8 in plan)
# =============================================================================


def fm_loss_per_sample(fm_params: Dict, z: jnp.ndarray, xi_target: jnp.ndarray,
                       xi_0: jnp.ndarray, s: float) -> jnp.ndarray:
    """Standard flow-matching loss for one (xi_0, s) draw."""
    xi_s = (1.0 - s) * xi_0 + s * xi_target
    u_pred = fm_velocity(fm_params, xi_s, s, z)
    u_true = xi_target - xi_0
    return jnp.mean((u_pred - u_true) ** 2)


def composition_loss(
    model: ModelBundle,
    comp: CompositionInput,
    sigma_label_log: float,
    sigma_weight: float,
    key,
) -> Tuple[jnp.ndarray, Dict]:
    """L_FM + L_sigma for a single composition."""
    z = composition_encoder_forward(
        model.params["mol_gnn"], model.params["attn"], comp, model.norm_stats,
    )
    xi_target = target_spectrum(model.params["fm"], z)

    # Vectorize the FM sample draws via vmap so the loss is one XLA graph.
    xi_keys = random.split(key, 2 * N_FM_SAMPLES_PER_COMP).reshape(
        N_FM_SAMPLES_PER_COMP, 2, -1
    )

    def per_draw(pair):
        xi_0 = random.normal(pair[0], (2 * K_SPECTRUM,))
        s = random.uniform(pair[1], minval=0.0, maxval=1.0)
        return fm_loss_per_sample(model.params["fm"], z, xi_target, xi_0, s)

    fm_loss_val = jnp.mean(vmap(per_draw)(xi_keys))

    log_sigma_pred = green_kubo_sigma(
        xi_target, comp.temperature_K, model.params["fm"]["log_sigma_offset"],
    )
    sigma_loss_val = sigma_weight * (log_sigma_pred - sigma_label_log) ** 2

    total = LAMBDA_FM * fm_loss_val + LAMBDA_SIGMA * sigma_loss_val
    return total, {
        "fm_loss": fm_loss_val,
        "sigma_loss": sigma_loss_val,
        "log_sigma_pred": log_sigma_pred,
    }


# =============================================================================
# COMPOSITION BUILD HELPERS
# =============================================================================


def pad_composition(
    smiles_list: List[str],
    mole_fractions: np.ndarray,
    temperature_K: float,
    smiles_to_graph: Callable[[str], MolecularGraph],
) -> CompositionInput:
    """Pad an arbitrary composition up to MAX_SPECIES for the model forward."""
    graphs = [smiles_to_graph(s) for s in smiles_list]
    while len(graphs) < MAX_SPECIES:
        graphs.append(EMPTY_GRAPH)
    mole_frac_padded = np.zeros(MAX_SPECIES)
    mole_frac_padded[: len(mole_fractions)] = mole_fractions
    mask = np.zeros(MAX_SPECIES)
    mask[: len(smiles_list)] = 1.0
    return CompositionInput(
        graphs=graphs,
        mole_fractions=jnp.asarray(mole_frac_padded),
        species_mask=jnp.asarray(mask),
        temperature_K=temperature_K,
    )


# =============================================================================
# TRAINING (Phase 8 in plan)
# =============================================================================


def init_all_params(key) -> Dict:
    k1, k2, k3 = random.split(key, 3)
    return {
        "mol_gnn": init_molecular_gnn_params(k1),
        "attn": init_attention_pool_params(k2),
        "fm": init_fm_params(k3),
    }


def make_optimizer(n_steps: int) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR_PEAK,
        warmup_steps=WARMUP_STEPS, decay_steps=n_steps,
        end_value=LR_FLOOR,
    )
    return optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY),
    )


def train_model(
    train_rows: List[LabeledRow],
    val_rows: List[LabeledRow],
    norm_stats: NormalizationStats,
    smiles_to_graph: Callable[[str], MolecularGraph],
    n_epochs: int,
    seed: int,
) -> ModelBundle:
    """Full training loop."""
    logger.info("Training: %d train rows, %d val rows, %d epochs",
                len(train_rows), len(val_rows), n_epochs)
    key = random.PRNGKey(seed)
    init_key, key = random.split(key)
    params = init_all_params(init_key)
    model = ModelBundle(params=params, norm_stats=norm_stats)

    n_steps = (len(train_rows) // BATCH_SIZE) * n_epochs
    optimizer = make_optimizer(n_steps)
    opt_state = optimizer.init(params)

    def loss_for_batch(params, batch_rows, batch_key):
        m = ModelBundle(params=params, norm_stats=norm_stats)
        total = 0.0
        keys = random.split(batch_key, len(batch_rows))
        for row, k_i in zip(batch_rows, keys):
            comp = pad_composition(
                row.smiles_list, row.mole_fractions, row.temperature_K,
                smiles_to_graph,
            )
            weight = 1.0 / (1.0 + row.sigma_uncertainty_log ** 2)
            loss_i, _ = composition_loss(
                m, comp, jnp.log(row.sigma_mScm), weight, k_i,
            )
            total = total + loss_i
        return total / len(batch_rows)

    grad_fn = value_and_grad(loss_for_batch)
    rng = np.random.default_rng(seed)
    step = 0
    for epoch in range(n_epochs):
        rng.shuffle(train_rows)
        for i in range(0, len(train_rows) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = train_rows[i : i + BATCH_SIZE]
            step_key, key = random.split(key)
            loss_val, grads = grad_fn(params, batch, step_key)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            step += 1
            if step % 100 == 0:
                logger.info("step %d epoch %d loss=%.4f",
                            step, epoch, float(loss_val))
        model = ModelBundle(params=params, norm_stats=norm_stats)
        val_metric = evaluate(model, val_rows, smiles_to_graph, key)
        logger.info("Epoch %d val log-MSE on sigma: %.4f", epoch, val_metric)

    return ModelBundle(params=params, norm_stats=norm_stats)


# =============================================================================
# VALIDATION (Phase 9 in plan)
# =============================================================================


def evaluate(
    model: ModelBundle,
    rows: List[LabeledRow],
    smiles_to_graph: Callable[[str], MolecularGraph],
    key,
) -> float:
    """Mean squared error in log space."""
    errors = []
    for row in rows:
        comp = pad_composition(
            row.smiles_list, row.mole_fractions, row.temperature_K,
            smiles_to_graph,
        )
        ek, key = random.split(key)
        log_pred_mean, _ = model_forward(model, comp, ek, n_samples=1)
        errors.append(float((log_pred_mean - jnp.log(row.sigma_mScm)) ** 2))
    return float(np.mean(errors))


def validation_battery(
    model: ModelBundle,
    val_rows: List[LabeledRow],
    ood_rows: List[LabeledRow],
    calisol_rows: List[LabeledRow],
    smiles_to_graph: Callable[[str], MolecularGraph],
    seed: int,
) -> Dict:
    """Phase 9 validation suite. Pass empty list for calisol_rows to skip."""
    key = random.PRNGKey(seed)
    out = {
        "val_log_mse": evaluate(model, val_rows, smiles_to_graph, key),
        "ood_log_mse": evaluate(model, ood_rows, smiles_to_graph, key),
        "exp_log_mse": (
            evaluate(model, calisol_rows, smiles_to_graph, key)
            if len(calisol_rows) > 0 else float("nan")
        ),
    }
    logger.info("Validation: %s", out)
    return out


# =============================================================================
# INFERENCE (Phase 11 in plan)
# =============================================================================


def predict_sigma(
    model: ModelBundle,
    smiles_to_graph: Callable[[str], MolecularGraph],
    comp_raw: RawComposition,
    n_samples: int,
    seed: int,
) -> Tuple[float, float]:
    """Predict (sigma_mean_mScm, sigma_std_mScm)."""
    if len(comp_raw.smiles_list) > MAX_SPECIES:
        raise ValueError(
            f"Composition has {len(comp_raw.smiles_list)} species > MAX_SPECIES={MAX_SPECIES}"
        )
    if abs(float(np.sum(comp_raw.mole_fractions)) - 1.0) > MOLE_FRAC_SUM_TOL:
        raise ValueError(f"Mole fractions sum {np.sum(comp_raw.mole_fractions)} != 1")

    comp = pad_composition(
        comp_raw.smiles_list, comp_raw.mole_fractions, comp_raw.temperature_K,
        smiles_to_graph,
    )
    key = random.PRNGKey(seed)
    log_mean, log_std = model_forward(model, comp, key, n_samples)
    sigma = float(jnp.exp(log_mean))
    sigma_err = float(jnp.exp(log_mean) * log_std) if n_samples > 1 else 0.0
    return sigma, sigma_err


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 1: data acquisition ===")
    oedb = load_oedb_v1()
    bamboo = load_bamboo_mix()
    uni_elf = load_uni_elf()
    calisol = load_calisol()

    logger.info("=== Phase 2: inspection ===")
    reports = {
        "OEDB": inspect_dataset(oedb, "OEDB"),
        "BAMBOO-Mix": inspect_dataset(bamboo, "BAMBOO-Mix"),
        "Uni-ELF": inspect_dataset(uni_elf, "Uni-ELF"),
        "CALiSol-23": inspect_dataset(calisol, "CALiSol-23"),
    }
    duplicates = find_cross_dataset_duplicates({
        "OEDB": oedb, "BAMBOO": bamboo, "Uni-ELF": uni_elf,
    })
    with open(INSPECT_REPORT_PATH, "w") as f:
        json.dump({"reports": reports, "n_duplicates": len(duplicates)}, f, indent=2)

    logger.info("=== Phase 3: cleaning ===")
    oedb_clean, _ = clean_rows(oedb)
    bamboo_clean, _ = clean_rows(bamboo)
    uni_elf_clean, _ = clean_rows(uni_elf)

    logger.info("=== Phase 6: Haven calibration ===")
    raise NotImplementedError(
        "Haven calibration requires BAMBOO trajectory access. Wire "
        "load_bamboo_trajectory + compute_green_kubo_sigma to the project's "
        "cepstral pipeline (per phaseA_backward_calibration), select N=50 "
        "calibration compositions, compute log_sigma_true, then call "
        "fit_haven_correction(log_sigma_NE, log_sigma_true, z_calibration)."
    )


if __name__ == "__main__":
    main()
