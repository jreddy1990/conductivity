"""
Molecular dynamics conductivity prediction using MACE-MP-0 and ASE.

Workflow:
    1. Parse recipe → SMILES + counts for each species
    2. Build simulation box with packmol (cubic, periodic)
    3. Assign MACE-MP-0 calculator (universal pretrained MLIP)
    4. Equilibrate: NVT (100 ps) → NPT (200 ps) to get correct density
    5. Production: NVT (1-2 ns) collecting positions every 50 fs
    6. Compute conductivity via Einstein relation on ion displacements

Physics:
    σ = (e² / 6 V k_B T) × lim_{t→∞} d/dt ⟨|Σ_i z_i [r_i(t) - r_i(0)]|²⟩
    This is the collective (Nernst-Einstein-corrected) Einstein relation.
    Unlike single-particle MSD which overestimates by ~20-40% (ignores
    ion-ion correlations), the collective dipole displacement captures
    the Haven ratio automatically.

Hardware:
    - Uses PyTorch MPS backend on Apple Silicon (M4)
    - Falls back to CPU if MPS unavailable
    - Typical runtime: 2-6 hours per composition on M4

Dependencies:
    - mace-torch (MACE-MP-0 pretrained model)
    - ase (Atoms, MD integrators, I/O)
    - packmol (binary, via brew install packmol)
    - torch (with MPS backend)
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import ase

import numpy as np

from constants import S_M_TO_MS_CM, T_REF_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018, exact or defined)
# ---------------------------------------------------------------------------
E_CHARGE_C = 1.602176634e-19        # elementary charge [C]
K_BOLTZMANN_J_K = 1.380649e-23      # Boltzmann constant [J/K]
AVOGADRO = 6.02214076e23            # Avogadro's number [1/mol]
AMU_KG = 1.66053906660e-27          # atomic mass unit [kg]
ANGSTROM_M = 1e-10                  # Å → m
FS_S = 1e-15                        # fs → s
PS_S = 1e-12                        # ps → s
WATER_COMPRESSIBILITY_INV_BAR = 4.57e-5  # isothermal compressibility of water at 25°C [bar⁻¹] (CRC Handbook 97th ed.)

# MACE-MP-0 model identifier
# "small" (L=0): 0.64s/step on M4 MPS, ~18h for 100k steps at 388 atoms
# "medium" (L=1): 1.70s/step, better accuracy but 2.7x slower
# "large" (L=2): most accurate but too slow for M4
MACE_MP0_MODEL = "small"


@dataclass
class MDConfig:
    """Configuration for a single MD conductivity run."""
    # Box construction
    n_molecules: int = 256          # total molecules (solvents). More = better stats, slower.
    box_padding_angstrom: float = 2.0  # extra space around packed molecules

    # Equilibration
    equil_nvt_steps: int = 10_000    # NVT equilibration steps (× timestep = total time)
    equil_npt_steps: int = 20_000    # NPT equilibration to get correct density
    equil_timestep_fs: float = 1.0   # timestep [fs] — 1 fs safe for MACE

    # Production
    prod_steps: int = 500_000        # production NVT steps (500k × 1fs = 500 ps)
    prod_timestep_fs: float = 1.0    # timestep [fs]
    save_interval: int = 50          # save positions every N steps (50 fs cadence)

    # Thermostat / barostat
    temperature_k: float = T_REF_K   # target temperature [K]
    pressure_gpa: float = 1.01325e-4 # 1 atm in GPa (ASE units)
    nvt_friction: float = 0.01       # Langevin friction [1/fs]
    npt_ttime: float = 25.0          # Berendsen thermostat time constant [fs]
    npt_ptime: float = 100.0         # Berendsen barostat time constant [fs]

    # Device
    device: str = "mps"              # "mps", "cpu", "cuda"
    mace_model: str = MACE_MP0_MODEL

    # Output
    output_dir: str = "conductivity/md_runs"
    save_trajectory: bool = True


@dataclass
class MDResult:
    """Results from a single MD conductivity run."""
    recipe: dict
    conductivity_mS_cm: float
    conductivity_std_mS_cm: float    # from block averaging
    density_g_ml: float              # equilibrated density
    temperature_k: float             # average production temperature
    n_frames: int                    # number of saved frames
    wall_time_s: float               # total wall-clock time
    msd_slope_ang2_ps: float            # collective MSD slope (for debugging)
    haven_ratio: float               # σ_collective / σ_NE (< 1 means correlations matter)
    metadata: dict = field(default_factory=dict)


def _smiles_to_xyz(smiles: str, name: str, work_dir: str) -> str:
    """Convert SMILES to 3D XYZ file using RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    xyz_path = os.path.join(work_dir, f"{name}.xyz")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    # Generate 3D coords with ETKDG (best conformer generator in RDKit)
    EMBED_SEED = 42  # fixed seed for reproducible conformer generation
    MMFF_MAX_ITERS = 500  # MMFF94 geometry optimization iterations (sufficient for small organics)
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # ETKDGv3 failed (can happen for strained rings) — retry with ETKDGv2
        status2 = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if status2 != 0:
            raise RuntimeError(
                f"RDKit embedding failed for {name} (SMILES={smiles}). "
                f"ETKDGv3 rc={status}, ETKDG rc={status2}"
            )
    AllChem.MMFFOptimizeMolecule(mol, maxIters=MMFF_MAX_ITERS)

    # Write XYZ
    conf = mol.GetConformer()
    atoms_list = mol.GetAtoms()
    lines = [str(mol.GetNumAtoms()), name]
    for atom in atoms_list:
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    with open(xyz_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"  Generated 3D coords for {name} ({mol.GetNumAtoms()} atoms) via RDKit")
    return xyz_path


def _recipe_to_molecule_counts(recipe: dict, n_total_molecules: int) -> tuple[list[dict], float]:
    """
    Convert recipe dict to list of {name, smiles, count, charge, molecular_weight} for box building.

    Recipe format (same as XGBoost training data):
        {"solvents": {"EC": 0.3, "DMC": 0.7},     # volume fractions
         "salts": {"LiPF6": 1.0},                   # mol/L
         "additives": {"FEC": 0.02}}                 # weight fractions

    Returns:
        molecules: list of dicts with molecule counts scaled to n_total_molecules.
            Salt ions are separated into cation (Li+) and anion (PF6-, FSI-, etc.).
        avg_solvent_density_g_ml: volume-weighted average solvent density [g/mL]
    """
    from data.species_data import SOLVENTS, SALTS, ADDITIVES

    molecules = []

    # --- Solvents: volume fractions → molecule counts ---
    if "solvents" not in recipe or not recipe["solvents"]:
        raise ValueError("Recipe must have non-empty 'solvents' dict")
    if "salts" not in recipe or not recipe["salts"]:
        raise ValueError("Recipe must have non-empty 'salts' dict")

    solvent_total = sum(recipe["solvents"].values())
    for name, vfrac in recipe["solvents"].items():
        if name not in SOLVENTS:
            raise ValueError(f"Unknown solvent: {name}")
        spec = SOLVENTS[name]
        smiles = spec["SMILES"]
        count = max(1, round(n_total_molecules * (vfrac / solvent_total)))
        molecules.append({
            "name": name, "smiles": smiles, "count": count,
            "charge": 0, "role": "solvent",
            "molecular_weight": spec["molecular_weight"],
        })

    # --- Salts: mol/L → molecule counts ---
    # Estimate box volume from solvent densities to convert mol/L to counts
    # Precompute average solvent MW and density for salt/additive count estimation
    avg_mw = sum(
        SOLVENTS[s]["molecular_weight"] * (v / solvent_total)
        for s, v in recipe["solvents"].items()
    )
    avg_density = sum(
        SOLVENTS[s]["density_g_ml"] * (v / solvent_total)
        for s, v in recipe["solvents"].items()
    )

    for salt_name, molarity in recipe["salts"].items():
        if salt_name not in SALTS:
            raise ValueError(f"Unknown salt: {salt_name}")
        spec = SALTS[salt_name]
        # molarity × V_box gives moles of salt → multiply by N_A for count
        # mass_g = n_molecules × MW [g/mol] / N_A
        # V_mL = mass_g / density [g/mL]
        # V_L = V_mL / 1000
        ML_PER_L = 1000.0  # mL → L unit conversion
        mass_g = n_total_molecules * avg_mw / AVOGADRO
        v_box_mL = mass_g / avg_density
        v_box_liters = v_box_mL / ML_PER_L
        n_salt = max(1, round(molarity * v_box_liters * AVOGADRO))

        LI_ATOMIC_MASS = 6.941  # Li atomic mass [g/mol] (IUPAC 2021)
        anion_mw = spec["molecular_weight"] - LI_ATOMIC_MASS

        # Li cation
        molecules.append({
            "name": f"Li_from_{salt_name}",
            "smiles": "[Li+]",
            "count": n_salt,
            "charge": +1,
            "role": "cation",
            "molecular_weight": LI_ATOMIC_MASS,
        })

        # Anion — extract from salt SMILES (everything after the Li+ part)
        anion_smiles = _extract_anion_smiles(salt_name, spec["SMILES"])
        molecules.append({
            "name": f"{salt_name}_anion",
            "smiles": anion_smiles,
            "count": n_salt,
            "charge": int(spec["anion_charge"]),
            "role": "anion",
            "molecular_weight": anion_mw,
        })

    # --- Additives: weight fractions → molecule counts (additives are optional) ---
    additives_dict = recipe["additives"] if "additives" in recipe else {}
    for add_name, wfrac in additives_dict.items():
        if add_name not in ADDITIVES:
            raise ValueError(f"Unknown additive: {add_name}")
        spec = ADDITIVES[add_name]
        smiles = spec["SMILES"]
        # wfrac of total → approximate molecule count, scaled by MW ratio to solvents
        mw_ratio = spec["molecular_weight"] / avg_mw
        count = max(1, round(n_total_molecules * wfrac / mw_ratio))
        # Ionic additives (LiDFOB, LiBOB) are Li salts — their SMILES contain [Li+].
        # They dissociate into Li+ and anion just like primary salts.
        # Non-ionic additives (FEC, VC, PS) are neutral molecules — no dissociation.
        is_li_salt = "[Li+]" in smiles
        if is_li_salt:
            LI_ATOMIC_MASS = 6.941  # Li atomic mass [g/mol] (IUPAC 2021)
            add_anion_mw = spec["molecular_weight"] - LI_ATOMIC_MASS
            molecules.append({
                "name": f"Li_from_{add_name}",
                "smiles": "[Li+]",
                "count": count,
                "charge": +1,
                "role": "cation",
                "molecular_weight": LI_ATOMIC_MASS,
            })
            anion_smiles = _extract_anion_smiles(add_name, smiles)
            anion_charge = _detect_anion_charge(anion_smiles, add_name)
            molecules.append({
                "name": f"{add_name}_anion",
                "smiles": anion_smiles,
                "count": count,
                "charge": anion_charge,
                "role": "anion",
                "molecular_weight": add_anion_mw,
            })
        else:
            molecules.append({
                "name": add_name, "smiles": smiles, "count": count,
                "charge": 0, "role": "additive",
                "molecular_weight": spec["molecular_weight"],
            })

    total_atoms = sum(m["count"] for m in molecules)
    logger.info(f"Recipe → {len(molecules)} species, {total_atoms} total molecules")
    for m in molecules:
        logger.info(f"  {m['name']:20s}: {m['count']:4d} molecules (charge={m['charge']:+d})")

    return molecules, avg_density


def _extract_anion_smiles(salt_name: str, full_smiles: str) -> str:
    """Extract anion SMILES from a salt SMILES like '[Li+].F[P-](F)(F)(F)(F)F'."""
    parts = full_smiles.split(".")
    anion_parts = [p for p in parts if "[Li+" not in p]
    if not anion_parts:
        raise ValueError(f"Cannot extract anion from {salt_name} SMILES: {full_smiles}")
    return ".".join(anion_parts)


def _detect_anion_charge(anion_smiles: str, species_name: str) -> int:
    """
    Detect anion formal charge from SMILES bracket notation.

    Examples: '[B-]...' → -1, '[Al-2]...' → -2, 'F[P-](F)...' → -1
    """
    import re
    # Match patterns like [X-], [X-2], [X--] in SMILES bracket atoms
    charges = re.findall(r'\[[A-Za-z@]+(-{1,})(\d*)\]', anion_smiles)
    if not charges:
        raise ValueError(
            f"Cannot detect anion charge from SMILES '{anion_smiles}' for {species_name}. "
            f"No negatively-charged bracket atom found."
        )
    # Sum all negative charges (for multi-center anions)
    total_charge = 0
    for dashes, digit in charges:
        if digit:
            total_charge -= int(digit)
        else:
            total_charge -= len(dashes)
    logger.info(f"  Detected anion charge for {species_name}: {total_charge} (from SMILES: {anion_smiles})")
    return total_charge


def _compute_density_g_ml(atoms: "ase.Atoms") -> float:
    """Compute density [g/mL] from ASE Atoms (mass in amu, volume in ų)."""
    mass_amu = atoms.get_masses().sum()
    vol_ang3 = atoms.get_volume()
    # g/mL = (mass_amu × AMU_KG) / (vol_ang3 × ANGSTROM_M³) / 1000
    return (mass_amu * AMU_KG) / (vol_ang3 * ANGSTROM_M**3) / 1000


# Mapping from salt anion name to the element symbol used as its positional proxy
# in the collective Einstein relation. The center atom of each molecular anion
# represents the anion's position for MSD tracking.
_ANION_CENTER_ELEMENT: dict[str, str] = {
    "LiPF6": "P",      # P is the center of octahedral PF6-
    "LiFSI": "N",      # N is the center of bis(fluorosulfonyl)imide
    "LiTFSI": "N",     # N is the center of bis(trifluoromethylsulfonyl)imide
    "LiBF4": "B",      # B is the center of tetrahedral BF4-
    "LiClO4": "Cl",    # Cl is the center of tetrahedral ClO4-
    "LiDFP": "P",      # P is the center of difluorophosphate
    "LiNO3": "N",      # N is the center of nitrate
}


def _ion_charge_map_from_molecules(molecules: list[dict]) -> dict[str, int]:
    """Build element→charge mapping for ion tracking from parsed recipe molecules.

    Returns a dict like {"Li": +1, "P": -1} for LiPF6 systems.
    Each molecular anion is represented by its center atom element.
    """
    charge_map: dict[str, int] = {}

    for mol in molecules:
        if mol["role"] == "cation":
            charge_map["Li"] = +1
        elif mol["role"] == "anion":
            # Find which salt this anion belongs to
            # mol["name"] is like "LiPF6_anion" or "LiFSI_anion"
            salt_name = mol["name"].replace("_anion", "")
            if salt_name not in _ANION_CENTER_ELEMENT:
                raise ValueError(
                    f"Unknown anion center element for salt '{salt_name}'. "
                    f"Add to _ANION_CENTER_ELEMENT in md_conductivity.py."
                )
            center_elem = _ANION_CENTER_ELEMENT[salt_name]
            charge_map[center_elem] = mol["charge"]

    logger.info(f"Ion charge map: {charge_map}")
    return charge_map


def _build_box_packmol(
    molecules: list[dict],
    work_dir: str,
    box_side_angstrom: float,
) -> str:
    """
    Use packmol to pack molecules into a cubic box.

    Returns path to the combined XYZ file.
    """
    # Generate individual XYZ files for each species
    xyz_files = {}
    for mol in molecules:
        key = mol["smiles"]
        if key not in xyz_files:
            xyz_files[key] = _smiles_to_xyz(mol["smiles"], mol["name"], work_dir)

    output_xyz = os.path.join(work_dir, "box.xyz")
    tolerance = 2.0  # Å minimum distance between molecules

    # Build packmol input
    inp_lines = [
        f"tolerance {tolerance}",
        "filetype xyz",
        f"output {output_xyz}",
        "",
    ]
    for mol in molecules:
        xyz_path = xyz_files[mol["smiles"]]
        margin = 1.0
        inp_lines.extend([
            f"structure {xyz_path}",
            f"  number {mol['count']}",
            f"  inside box {margin} {margin} {margin} "
            f"{box_side_angstrom - margin} {box_side_angstrom - margin} {box_side_angstrom - margin}",
            "end structure",
            "",
        ])

    inp_path = os.path.join(work_dir, "packmol.inp")
    inp_text = "\n".join(inp_lines)
    with open(inp_path, "w") as f:
        f.write(inp_text)

    logger.info(f"Running packmol with box side = {box_side_angstrom:.1f} Å ...")
    PACKMOL_TIMEOUT_S = 300  # 5 min timeout for packing ~1000 molecules
    with open(inp_path, "r") as inp_file:
        result = subprocess.run(
            ["packmol"],
            stdin=inp_file,
            capture_output=True, text=True, timeout=PACKMOL_TIMEOUT_S,
        )
    if result.returncode != 0 or not os.path.exists(output_xyz):
        raise RuntimeError(
            f"Packmol failed (rc={result.returncode}):\n"
            f"STDOUT: {result.stdout[-500:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )

    logger.info(f"Packmol box built: {output_xyz}")
    return output_xyz


def _estimate_box_side(molecules: list[dict], target_density_g_ml: float) -> float:
    """Estimate cubic box side length [Å] from species counts and target density."""
    total_mass_amu = 0.0

    for mol in molecules:
        mw = mol["molecular_weight"]
        total_mass_amu += mol["count"] * mw

    total_mass_g = total_mass_amu / AVOGADRO
    volume_ml = total_mass_g / target_density_g_ml
    volume_angstrom3 = volume_ml * 1e24  # 1 mL = 1 cm³ = 1e24 ų
    side = volume_angstrom3 ** (1.0 / 3.0)

    logger.info(f"Estimated box: mass={total_mass_amu:.0f} amu, "
                f"density={target_density_g_ml:.3f} g/mL, side={side:.1f} Å")
    return side


def _xyz_to_ase_atoms(xyz_path: str, box_side_angstrom: float) -> "ase.Atoms":
    """Read packmol XYZ output into ASE Atoms with periodic boundary conditions."""
    from ase import Atoms
    from ase.io import read as ase_read

    result = ase_read(xyz_path, format="xyz")
    # ase.io.read returns Atoms for single frame, List[Atoms] for multi-frame
    if isinstance(result, list):
        raise RuntimeError(f"Expected single frame from {xyz_path}, got {len(result)} frames")
    atoms: Atoms = result
    # Set cubic cell and PBC
    atoms.set_cell([box_side_angstrom] * 3)
    atoms.set_pbc([True, True, True])
    # Center atoms in box
    atoms.center()

    logger.info(f"ASE Atoms: {len(atoms)} atoms, cell={box_side_angstrom:.1f} Å, PBC=True")
    return atoms


_MACE_MPS_PATCHED = False  # guard against double-patching


def _patch_mace_for_mps():
    """
    Monkey-patch MACE models to avoid .double() calls that MPS cannot handle.

    MACE v0.3.x models.py line 580 does:
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()
    MPS does not support float64. We patch to use float32 instead.
    The precision loss is negligible for MD (energy differences matter, not absolutes).
    """
    global _MACE_MPS_PATCHED
    if _MACE_MPS_PATCHED:
        logger.info("MACE MPS patch already applied, skipping")
        return

    import torch
    import mace.modules.models as mace_models

    original_forward = mace_models.ScaleShiftMACE.forward

    def patched_forward(self, data, **kwargs):
        original_double = torch.Tensor.double

        def safe_double(tensor):
            if tensor.device.type == "mps":
                return tensor.float()
            return original_double(tensor)

        torch.Tensor.double = safe_double
        try:
            return original_forward(self, data, **kwargs)
        finally:
            torch.Tensor.double = original_double

    mace_models.ScaleShiftMACE.forward = patched_forward
    _MACE_MPS_PATCHED = True
    logger.info("Patched MACE ScaleShiftMACE.forward for MPS float32 compatibility")


def _load_mace_mp0(device: str, model_size: str):
    """Load pretrained MACE-MP-0 calculator (supports MPS via float32 patch)."""
    import torch
    from mace.calculators import mace_mp

    t0 = time.time()
    if device == "mps":
        _patch_mace_for_mps()
        logger.info(f"Loading MACE-MP-0 ({model_size}) on CPU then moving to MPS ...")
        calc = mace_mp(model=model_size, device="cpu", default_dtype="float32")
        for model in calc.models:
            model.to(torch.device("mps"))
        calc.device = torch.device("mps")
    else:
        logger.info(f"Loading MACE-MP-0 ({model_size}) on {device} ...")
        calc = mace_mp(model=model_size, device=device, default_dtype="float32")
    logger.info(f"MACE-MP-0 loaded in {time.time() - t0:.1f}s on {device}")
    return calc


def _load_custom_mace_model(model_path: str, device: str):
    """Load a custom MACE .model file as an ASE calculator."""
    import torch
    from mace.calculators import MACECalculator

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    t0 = time.time()
    torch.serialization.add_safe_globals([slice])
    logger.info(f"Loading custom MACE model: {model_path} on {device} ...")
    if device == "mps":
        _patch_mace_for_mps()
        calc = MACECalculator(model_paths=model_path, device="cpu", default_dtype="float32")
        for model in calc.models:
            model.to(torch.device("mps"))
        calc.device = torch.device("mps")
    else:
        calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float32")
    logger.info(f"Custom MACE loaded in {time.time() - t0:.1f}s on {device}")
    return calc


def _run_md_equilibration(
    atoms: "ase.Atoms",
    config: MDConfig,
) -> "ase.Atoms":
    """
    Run NVT + NPT equilibration.

    1. NVT with Langevin thermostat (100 ps) — thermalize velocities
    2. NPT with Berendsen (200 ps) — relax density
    """
    from ase.md.langevin import Langevin
    from ase.md.nptberendsen import NPTBerendsen
    from ase import units

    logger.info("="*60)
    logger.info("EQUILIBRATION: NVT phase")
    logger.info(f"  Steps: {config.equil_nvt_steps}, dt={config.equil_timestep_fs} fs")

    # NVT with Langevin
    dyn = Langevin(
        atoms,
        timestep=config.equil_timestep_fs * units.fs,
        temperature_K=config.temperature_k,
        friction=config.nvt_friction / units.fs,
    )

    NVT_LOG_INTERVAL = 1000  # log every 1000 steps during equilibration

    def _log_nvt(step=[0]):
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        t_inst = e_kin / (1.5 * len(atoms) * units.kB)
        logger.info(f"  NVT step {step[0] * NVT_LOG_INTERVAL:6d}: E_pot={e_pot:.1f} eV, "
                   f"E_kin={e_kin:.1f} eV, T={t_inst:.0f} K")
        step[0] += 1

    dyn.attach(_log_nvt, interval=NVT_LOG_INTERVAL)
    dyn.run(config.equil_nvt_steps)

    logger.info("EQUILIBRATION: NPT phase")
    logger.info(f"  Steps: {config.equil_npt_steps}, dt={config.equil_timestep_fs} fs")

    # NPT with Berendsen (good for equilibration, not for production)
    dyn_npt = NPTBerendsen(
        atoms,
        timestep=config.equil_timestep_fs * units.fs,
        temperature_K=config.temperature_k,
        pressure_au=config.pressure_gpa * units.GPa,
        taut=config.npt_ttime * units.fs,
        taup=config.npt_ptime * units.fs,
        # Isothermal compressibility of water at 25°C (CRC Handbook, 97th ed.)
        # Used as initial estimate; NPT relaxation corrects volume regardless
        compressibility_au=WATER_COMPRESSIBILITY_INV_BAR / units.bar,
    )

    NPT_LOG_INTERVAL = 1000  # log every 1000 steps during equilibration

    def _log_npt(step=[0]):
        e_pot = atoms.get_potential_energy()
        vol = atoms.get_volume()
        density = _compute_density_g_ml(atoms)
        logger.info(f"  NPT step {step[0] * NPT_LOG_INTERVAL:6d}: E_pot={e_pot:.1f} eV, "
                   f"V={vol:.0f} ų, ρ={density:.4f} g/mL")
        step[0] += 1

    dyn_npt.attach(_log_npt, interval=NPT_LOG_INTERVAL)
    dyn_npt.run(config.equil_npt_steps)

    # Log final density
    density = _compute_density_g_ml(atoms)
    logger.info(f"Equilibration complete. Final density: {density:.4f} g/mL")
    logger.info(f"Final cell: {atoms.get_cell().lengths()}")

    return atoms


def _run_md_production(
    atoms: "ase.Atoms",
    config: MDConfig,
    ion_charge_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run production NVT MD, collecting all ion positions for MSD calculation.

    For correct collective Einstein conductivity, tracks ALL charge carriers:
    Li+ (z=+1) and anion center atoms (e.g. P for PF6- with z=-1).

    Args:
        atoms: equilibrated ASE Atoms with calculator attached
        config: MD configuration
        ion_charge_map: element_symbol → formal_charge for each ion to track.
            Example: {"Li": +1, "P": -1} for LiPF6 systems.
            P atom serves as proxy for PF6- center of mass.

    Returns:
        ion_positions: shape (n_frames, n_ions, 3) — unwrapped ion positions [Å]
        ion_charges: shape (n_ions,) — formal charge of each tracked ion
        times: shape (n_frames,) — time stamps [ps]
        avg_temperature: average temperature during production [K]
    """
    from ase.md.langevin import Langevin
    from ase import units

    logger.info("="*60)
    logger.info("PRODUCTION: NVT")
    logger.info(f"  Steps: {config.prod_steps}, dt={config.prod_timestep_fs} fs")
    logger.info(f"  Save interval: {config.save_interval} steps ({config.save_interval * config.prod_timestep_fs} fs)")
    logger.info(f"  Total time: {config.prod_steps * config.prod_timestep_fs / 1000:.1f} ps")

    # Identify all ion atoms
    symbols = np.array(atoms.get_chemical_symbols())
    ion_indices = []
    ion_charges_list = []
    for elem, charge in ion_charge_map.items():
        elem_idx = np.where(symbols == elem)[0]
        ion_indices.extend(elem_idx.tolist())
        ion_charges_list.extend([charge] * len(elem_idx))
        logger.info(f"  Tracking {len(elem_idx)} {elem} atoms (charge={charge:+d})")

    ion_indices = np.array(ion_indices, dtype=int)
    ion_charges = np.array(ion_charges_list, dtype=float)
    n_ions = len(ion_indices)
    logger.info(f"  Total tracked ions: {n_ions}")

    n_frames = config.prod_steps // config.save_interval
    ion_positions = np.zeros((n_frames, n_ions, 3))
    times = np.zeros(n_frames)
    temperatures = np.zeros(n_frames)

    # For unwrapping PBC: track cumulative displacement
    prev_pos = atoms.positions[ion_indices].copy()
    unwrapped_pos = prev_pos.copy()

    # Cell is fixed during NVT production — compute once
    cell_diag = np.diag(atoms.get_cell())

    frame_idx = [0]
    LOG_EVERY_N_PERCENT = 10  # log progress every 10% of frames
    log_interval = max(1, n_frames // LOG_EVERY_N_PERCENT)

    dyn = Langevin(
        atoms,
        timestep=config.prod_timestep_fs * units.fs,
        temperature_K=config.temperature_k,
        friction=config.nvt_friction / units.fs,
    )

    def _collect():
        nonlocal prev_pos, unwrapped_pos

        current_pos = atoms.positions[ion_indices].copy()

        # Unwrap PBC: minimum image correction handles jumps > half cell.
        # At save_interval=50 (50 fs), ion displacement ~ 0.001 Å << cell/2 (~15 Å),
        # so minimum image is safe at this cadence.
        delta = current_pos - prev_pos
        delta -= cell_diag * np.round(delta / cell_diag)
        unwrapped_pos += delta
        prev_pos = current_pos

        idx = frame_idx[0]
        if idx < n_frames:
            ion_positions[idx] = unwrapped_pos.copy()
            times[idx] = idx * config.save_interval * config.prod_timestep_fs / 1000  # ps

            # Only compute temperature at log intervals (avoids per-frame ASE property call)
            if idx % log_interval == 0:
                e_kin = atoms.get_kinetic_energy()
                t_inst = e_kin / (1.5 * len(atoms) * units.kB)
                temperatures[idx] = t_inst
                logger.info(f"  Production: frame {idx}/{n_frames}, "
                           f"t={times[idx]:.1f} ps, T={t_inst:.0f} K")

            frame_idx[0] = idx + 1

    dyn.attach(_collect, interval=config.save_interval)
    dyn.run(config.prod_steps)

    # Fill temperature for non-logged frames by sampling at end
    e_kin_final = atoms.get_kinetic_energy()
    t_final = e_kin_final / (1.5 * len(atoms) * units.kB)
    # Use mean of sampled temperatures (logged frames only)
    sampled = temperatures[temperatures > 0]
    avg_temp = float(np.mean(sampled)) if len(sampled) > 0 else float(t_final)
    logger.info(f"Production complete. {frame_idx[0]} frames collected, <T>={avg_temp:.1f} K")

    return ion_positions[:frame_idx[0]], ion_charges, times[:frame_idx[0]], avg_temp


def compute_conductivity_einstein(
    li_positions: np.ndarray,
    times: np.ndarray,
    volume_angstrom3: float,
    temperature_k: float,
    charges: Optional[np.ndarray] = None,
) -> tuple[float, float, float, float]:
    """
    Compute ionic conductivity from collective ion displacement (Einstein relation).

    σ = (e² / 6 V k_B T) × d/dt ⟨|M(t)|²⟩

    where M(t) = Σ_i z_i × [r_i(t) - r_i(0)] is the collective dipole displacement.

    For Li+ only (z=+1 for all):
        M(t) = Σ_i [r_i(t) - r_i(0)]

    Uses multiple time origins for better statistics.

    Args:
        li_positions: (n_frames, n_ions, 3) unwrapped positions [Å]
        times: (n_frames,) time stamps [ps]
        volume_angstrom3: box volume [ų]
        temperature_k: temperature [K]
        charges: (n_ions,) formal charges. Default: all +1 (Li+)

    Returns:
        sigma_mS_cm: conductivity [mS/cm]
        sigma_std: standard deviation from block averaging [mS/cm]
        msd_slope: slope of ⟨|M(t)|²⟩ vs t [Å²/ps]
        haven_ratio: ratio of collective to single-particle conductivity
    """
    n_frames, n_ions, _ = li_positions.shape
    resolved_charges: np.ndarray = charges if charges is not None else np.ones(n_ions)

    # --- Collective dipole displacement M(t) ---
    # M(t, t0) = Σ_i z_i × [r_i(t) - r_i(t0)]
    # ⟨|M|²⟩ averaged over time origins
    #
    # Vectorized: for each time origin t0, compute all lags at once via
    # broadcasting. Inner loop over lags eliminated.

    origin_stride = max(1, n_frames // 100)
    max_lag = n_frames // 2  # use first half as max lag time

    msd_collective = np.zeros(max_lag)
    msd_single = np.zeros(max_lag)  # single-particle MSD for Haven ratio
    counts = np.zeros(max_lag)

    # Charge-weighted positions: (n_frames, n_ions, 3) → sum over ions → (n_frames, 3)
    # M(t) = Σ_i z_i r_i(t), so M(t)-M(t0) = Σ_i z_i [r_i(t)-r_i(t0)]
    weighted_pos = resolved_charges[:, None] * li_positions  # (n_frames, n_ions, 3)
    M_cumulative = np.sum(weighted_pos, axis=1)  # (n_frames, 3)

    for t0_idx in range(0, n_frames - max_lag, origin_stride):
        # Vectorize over all lags at once: shape (max_lag-1, 3) and (max_lag-1, n_ions, 3)
        lags = np.arange(1, max_lag)
        t1_indices = t0_idx + lags  # (max_lag-1,)

        # Collective: |M(t1) - M(t0)|² for all lags
        dM = M_cumulative[t1_indices] - M_cumulative[t0_idx]  # (max_lag-1, 3)
        msd_collective[1:] += np.sum(dM**2, axis=1)  # (max_lag-1,)

        # Single-particle: mean over ions of |r_i(t1)-r_i(t0)|²
        dr = li_positions[t1_indices] - li_positions[t0_idx]  # (max_lag-1, n_ions, 3)
        msd_single[1:] += np.mean(np.sum(dr**2, axis=2), axis=1)  # (max_lag-1,)

        counts[1:] += 1

    # Average over time origins
    valid = counts > 0
    msd_collective[valid] /= counts[valid]
    msd_single[valid] /= counts[valid]

    lag_times = np.arange(max_lag) * (times[1] - times[0])  # [ps]

    # --- Linear fit to get slope ---
    # Use 20%-80% of lag range: skip short-time ballistic regime and
    # long-time regime where statistics degrade (standard MSD practice)
    MSD_FIT_START_FRAC = 5   # 1/5 = skip first 20% (ballistic)
    MSD_FIT_END_FRAC = 5     # 4/5 = skip last 20% (noisy)
    MIN_FIT_POINTS = 10      # minimum data points for a meaningful linear fit
    fit_start = max_lag // MSD_FIT_START_FRAC
    fit_end = 4 * max_lag // MSD_FIT_END_FRAC
    fit_range = slice(fit_start, fit_end)

    if fit_end <= fit_start + MIN_FIT_POINTS:
        raise RuntimeError(
            f"Too few frames for reliable MSD fit: fit_end={fit_end}, fit_start={fit_start}, "
            f"need at least {MIN_FIT_POINTS} points. Got {n_frames} total frames, max_lag={max_lag}."
        )

    # Collective MSD slope
    slope_coll, _ = np.polyfit(
        lag_times[fit_range], msd_collective[fit_range], 1
    )
    # Single-particle MSD slope
    slope_single, _ = np.polyfit(
        lag_times[fit_range], msd_single[fit_range], 1
    )

    # --- Convert to conductivity ---
    # σ = (e² / (6 V k_B T)) × d⟨|M|²⟩/dt
    # The 6 = 2 × 3 dimensions (from 3D diffusion: MSD = 6Dt in 3D)
    EINSTEIN_DIM_FACTOR = 6  # 2d for d=3 spatial dimensions
    # S_M_TO_MS_CM imported from constants

    slope_m2_s = slope_coll * (ANGSTROM_M**2) / PS_S
    volume_m3 = volume_angstrom3 * (ANGSTROM_M**3)

    sigma_SI = (E_CHARGE_C**2 / (EINSTEIN_DIM_FACTOR * volume_m3 * K_BOLTZMANN_J_K * temperature_k)) * slope_m2_s
    sigma_mS_cm = sigma_SI * S_M_TO_MS_CM

    # --- Single-particle (Nernst-Einstein) conductivity for Haven ratio ---
    slope_single_m2_s = slope_single * (ANGSTROM_M**2) / PS_S
    D_self = slope_single_m2_s / EINSTEIN_DIM_FACTOR
    sigma_NE_SI = (n_ions * E_CHARGE_C**2 * D_self) / (volume_m3 * K_BOLTZMANN_J_K * temperature_k)
    sigma_NE_mS_cm = sigma_NE_SI * S_M_TO_MS_CM

    haven = sigma_mS_cm / sigma_NE_mS_cm if sigma_NE_mS_cm > 0 else 0.0

    logger.info(f"Conductivity (collective Einstein):")
    logger.info(f"  MSD slope (collective): {slope_coll:.2f} Å²/ps")
    logger.info(f"  MSD slope (single):     {slope_single:.4f} Å²/ps")
    logger.info(f"  σ_collective = {sigma_mS_cm:.3f} mS/cm")
    logger.info(f"  σ_NE         = {sigma_NE_mS_cm:.3f} mS/cm")
    logger.info(f"  Haven ratio  = {haven:.3f}")
    logger.info(f"  D_self(Li+)  = {D_self:.3e} m²/s")

    # --- Block averaging for error estimate ---
    N_BLOCKS = 5  # standard block averaging: 5 blocks balances bias vs variance
    block_size = (fit_end - fit_start) // N_BLOCKS
    if block_size < 2:
        raise RuntimeError(
            f"Block averaging failed: block_size={block_size} too small. "
            f"Need more production frames (got {n_frames})."
        )
    # Vectorized: reshape fit region into (N_BLOCKS, block_size) and regress each block
    n_usable = N_BLOCKS * block_size
    block_times = lag_times[fit_start:fit_start + n_usable].reshape(N_BLOCKS, block_size)
    block_msds = msd_collective[fit_start:fit_start + n_usable].reshape(N_BLOCKS, block_size)
    # Least-squares slope for each block: slope = cov(t,y)/var(t)
    t_mean = block_times.mean(axis=1, keepdims=True)
    y_mean = block_msds.mean(axis=1, keepdims=True)
    block_slopes = (((block_times - t_mean) * (block_msds - y_mean)).sum(axis=1)
                    / ((block_times - t_mean)**2).sum(axis=1))  # (N_BLOCKS,)
    prefactor = (E_CHARGE_C**2 / (EINSTEIN_DIM_FACTOR * volume_m3 * K_BOLTZMANN_J_K * temperature_k)
                 * (ANGSTROM_M**2) / PS_S * S_M_TO_MS_CM)
    block_sigmas = block_slopes * prefactor
    sigma_std = float(np.std(block_sigmas))

    return float(sigma_mS_cm), sigma_std, float(slope_coll), float(haven)


def run_md_conductivity(
    recipe: dict,
    config: Optional[MDConfig] = None,
) -> MDResult:
    """
    Full MD conductivity pipeline for a single recipe.

    Steps:
        1. Parse recipe → molecule counts
        2. Build box with packmol
        3. Setup MACE calculator
        4. Equilibrate (NVT + NPT)
        5. Production NVT
        6. Compute conductivity from Einstein relation

    Args:
        recipe: {"solvents": {...}, "salts": {...}, "additives": {...}}
        config: MDConfig (uses defaults if None)

    Returns:
        MDResult with conductivity and metadata
    """
    if config is None:
        config = MDConfig()

    t_wall_start = time.time()

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Step 1: Recipe → molecule counts
    logger.info("="*60)
    logger.info("STEP 1: Recipe → molecule counts")
    molecules, avg_solvent_density = _recipe_to_molecule_counts(recipe, config.n_molecules)

    # Step 2: Estimate box size and build with packmol
    logger.info("="*60)
    logger.info("STEP 2: Build simulation box")

    # Salt increases solution density: ~8% per mol/L from partial molar volume
    # of Li salts in carbonate solvents (Nishida et al. 2003, J. Electrochem. Soc.)
    SALT_DENSITY_CORRECTION_PER_M = 0.08  # fractional density increase per mol/L
    total_salt_M = sum(recipe["salts"].values())
    target_density = avg_solvent_density * (1.0 + SALT_DENSITY_CORRECTION_PER_M * total_salt_M)

    box_side = _estimate_box_side(molecules, target_density)

    with tempfile.TemporaryDirectory(prefix="mace_md_") as work_dir:
        box_xyz = _build_box_packmol(molecules, work_dir, box_side)

        # Step 3: Load into ASE + attach MACE calculator
        logger.info("="*60)
        logger.info("STEP 3: Setup MACE calculator")
        atoms = _xyz_to_ase_atoms(box_xyz, box_side)
        if os.path.isfile(config.mace_model):
            calc = _load_custom_mace_model(config.mace_model, config.device)
        else:
            calc = _load_mace_mp0(config.device, config.mace_model)
        atoms.calc = calc

        # Step 4: Equilibrate
        logger.info("="*60)
        logger.info("STEP 4: Equilibration")
        atoms = _run_md_equilibration(atoms, config)

        # Record equilibrated density
        vol = atoms.get_volume()
        density = _compute_density_g_ml(atoms)

        # Step 5: Production MD
        logger.info("="*60)
        logger.info("STEP 5: Production MD")
        # Build ion charge map from recipe molecules:
        # Li+ → +1, anion center atom → anion charge
        # For PF6-: track P (center of octahedral PF6); for FSI-/TFSI-: track N
        ion_charge_map = _ion_charge_map_from_molecules(molecules)
        ion_positions, ion_charges, times, avg_temp = _run_md_production(
            atoms, config, ion_charge_map,
        )

        # Step 6: Compute conductivity
        logger.info("="*60)
        logger.info("STEP 6: Compute conductivity")
        sigma, sigma_std, msd_slope, haven = compute_conductivity_einstein(
            ion_positions, times, vol, avg_temp, charges=ion_charges,
        )

    wall_time = time.time() - t_wall_start

    result = MDResult(
        recipe=recipe,
        conductivity_mS_cm=sigma,
        conductivity_std_mS_cm=sigma_std,
        density_g_ml=density,
        temperature_k=avg_temp,
        n_frames=len(times),
        wall_time_s=wall_time,
        msd_slope_ang2_ps=msd_slope,
        haven_ratio=haven,
        metadata={
            "config": {
                "n_molecules": config.n_molecules,
                "prod_steps": config.prod_steps,
                "device": config.device,
                "mace_model": config.mace_model,
            },
            "molecules": [
                {"name": m["name"], "count": m["count"], "charge": m["charge"]}
                for m in molecules
            ],
        },
    )

    logger.info("="*60)
    logger.info("MD CONDUCTIVITY RESULT:")
    logger.info(f"  σ = {sigma:.3f} ± {sigma_std:.3f} mS/cm")
    logger.info(f"  ρ = {density:.4f} g/mL")
    logger.info(f"  <T> = {avg_temp:.1f} K")
    logger.info(f"  Haven ratio = {haven:.3f}")
    logger.info(f"  Wall time: {wall_time/3600:.2f} hours")
    logger.info("="*60)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MD conductivity via MACE")
    parser.add_argument("--model", default=MACE_MP0_MODEL,
                        help="MACE model: 'small'/'medium'/'large' for MP-0, or path to .model file")
    parser.add_argument("--device", default="cpu", help="Device: cpu, mps, cuda")
    parser.add_argument("--prod-steps", type=int, default=10_000,
                        help="Production steps (10k=quick test, 1M=production)")
    parser.add_argument("--n-molecules", type=int, default=MDConfig.n_molecules,
                        help="Total solvent molecules")
    parser.add_argument("--salt", default="LiPF6", help="Salt name")
    parser.add_argument("--molarity", type=float, default=1.0, help="Salt molarity")
    parser.add_argument("--solvents", default="EC:0.3,DMC:0.7",
                        help="Comma-separated name:fraction pairs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse solvents from CLI
    solvents = {}
    for pair in args.solvents.split(","):
        name, frac = pair.split(":")
        solvents[name.strip()] = float(frac)

    recipe = {
        "solvents": solvents,
        "salts": {args.salt: args.molarity},
    }

    config = MDConfig(
        n_molecules=args.n_molecules,
        equil_nvt_steps=1_000,
        equil_npt_steps=2_000,
        prod_steps=args.prod_steps,
        save_interval=MDConfig.save_interval,
        device=args.device,
        mace_model=args.model,
    )

    logger.info(f"Recipe: {recipe}")
    logger.info(f"Model: {args.model}, device: {args.device}")
    result = run_md_conductivity(recipe, config)

    logger.info(f"\nFinal result: σ = {result.conductivity_mS_cm:.3f} ± {result.conductivity_std_mS_cm:.3f} mS/cm")
    logger.info(f"Haven ratio: {result.haven_ratio:.3f}")
    logger.info(f"Density: {result.density_g_ml:.4f} g/mL")
    logger.info(f"Wall time: {result.wall_time_s/3600:.2f} hours")

    # Save result
    output_path = os.path.join(config.output_dir, "md_result.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "recipe": result.recipe,
            "conductivity_mS_cm": result.conductivity_mS_cm,
            "conductivity_std_mS_cm": result.conductivity_std_mS_cm,
            "density_g_ml": result.density_g_ml,
            "temperature_k": result.temperature_k,
            "wall_time_s": result.wall_time_s,
            "haven_ratio": result.haven_ratio,
        }, f, indent=2)
    logger.info(f"Result saved to {output_path}")
