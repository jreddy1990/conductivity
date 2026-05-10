"""
=============================================================================
BAMBOO MD CONDUCTIVITY ON GOOGLE COLAB
=============================================================================

Runs BAMBOO (GNN force field for Li-ion electrolytes) on Colab's free T4 GPU
to compute ionic conductivity from first principles for arbitrary compositions.

BAMBOO covers: EC, PC, DMC, DEC, EMC, EA, MA, FEC, VC, LiPF6, LiFSI, LiTFSI

Workflow:
  1. Build LAMMPS + BAMBOO pair style on Colab (cached to Drive)
  2. Build simulation box with Packmol
  3. Run NPT equilibration (1 ns) → correct density
  4. Run NVT production (5 ns) → trajectory for MSD
  5. Extract conductivity via collective Einstein relation

Expected: ~6-12 hours per composition on T4 (depending on box size).
Results saved to Google Drive for persistence across sessions.

Reference: Gong et al., Nature Machine Intelligence 7, 543-552 (2025)
"""

# %% CELL 1 — INSTALL + BUILD ================================================
# Builds LAMMPS with BAMBOO pair style. Takes ~25 min first time.
# Cached to Google Drive so subsequent sessions skip the build.

import subprocess
import sys
import os

def _run(cmd, **kwargs):
    """Run shell command, print output, raise on failure."""
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.stdout:
        print(result.stdout[-2000:])  # last 2000 chars
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Mount Google Drive for persistence
from google.colab import drive  # type: ignore[import-not-found]
drive.mount("/content/drive")

WORK_DIR = "/content/drive/MyDrive/bamboo_conductivity"
LAMMPS_BIN = f"{WORK_DIR}/lammps_bamboo/lmp"
os.makedirs(WORK_DIR, exist_ok=True)

# Install Python dependencies
_install("numpy")
_install("scipy")
_install("rdkit")
_install("ase")
_install("openbabel-wheel")  # for Gasteiger charges

# Check if LAMMPS binary is already built (cached on Drive)
if os.path.exists(LAMMPS_BIN):
    print(f"LAMMPS binary found at {LAMMPS_BIN} — skipping build")
else:
    print("Building LAMMPS + BAMBOO from source (~25 min)...")

    # System deps
    _run("apt-get update -y -qq")
    _run("apt-get install -y -qq gfortran libfftw3-dev libopenblas-dev cmake git")

    # Clone BAMBOO
    _run("cd /tmp && git clone --depth 1 https://github.com/bytedance/bamboo.git")

    # Run init_compile.sh (downloads libtorch, clones LAMMPS, patches source)
    _run("cd /tmp/bamboo/pair && bash ./init_compile.sh")

    # Modify build.sh for Colab T4 (sm_75, Turing architecture)
    build_sh = "/tmp/bamboo/pair/lammps/build.sh"
    with open("/tmp/bamboo/pair/build.sh") as f:
        build_content = f.read()

    # Replace RTX 4090 (Ada, sm_89) with T4 (Turing, sm_75)
    build_content = build_content.replace("CMAKE_CUDA_ARCHITECTURES=89", "CMAKE_CUDA_ARCHITECTURES=75")
    build_content = build_content.replace("DKokkos_ARCH_ADA89=ON", "DKokkos_ARCH_TURING75=ON")
    build_content = build_content.replace("GPU_ARCH=sm_89", "GPU_ARCH=sm_75")

    with open(build_sh, "w") as f:
        f.write(build_content)

    # Build LAMMPS
    _run("cd /tmp/bamboo/pair/lammps && bash ./build.sh")

    # Cache binary + BAMBOO model to Drive
    os.makedirs(f"{WORK_DIR}/lammps_bamboo", exist_ok=True)
    _run(f"cp /tmp/bamboo/pair/lammps/output/lmp {LAMMPS_BIN}")
    _run(f"cp /tmp/bamboo/benchmark/benchmark.pt {WORK_DIR}/lammps_bamboo/")

    # Also copy the new dispersion checkpoint (better accuracy per README)
    if os.path.exists("/tmp/bamboo/benchmark/paper_new_disp.pt"):
        _run(f"cp /tmp/bamboo/benchmark/paper_new_disp.pt {WORK_DIR}/lammps_bamboo/")

    print(f"LAMMPS + BAMBOO built and cached to {WORK_DIR}/lammps_bamboo/")

# Verify LAMMPS exists
assert os.path.exists(LAMMPS_BIN), f"LAMMPS binary not found at {LAMMPS_BIN}"
print(f"LAMMPS binary: {LAMMPS_BIN}")

# Install Packmol
PACKMOL_BIN = f"{WORK_DIR}/packmol"
if not os.path.exists(PACKMOL_BIN):
    _run("cd /tmp && git clone --depth 1 https://github.com/m3g/packmol.git")
    _run("cd /tmp/packmol && make")
    _run(f"cp /tmp/packmol/packmol {PACKMOL_BIN}")
print(f"Packmol: {PACKMOL_BIN}")


# %% CELL 2 — DEFINE COMPOSITION =============================================
# Specify the electrolyte recipe to simulate.
# All species must be in BAMBOO's training set.

import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bamboo_cond")

# ============ EDIT THIS RECIPE ============
RECIPE = {
    "solvents": {"EC": 0.3191, "DMC": 0.6809},  # volume fractions
    "salts": {"LiPF6": 0.198, "LiFSI": 0.78},   # mol/L
    "additives": {"FEC": 0.0518},                 # weight fraction
}
TARGET_CONDUCTIVITY = 11.91  # mS/cm — experimental reference for validation
# ==========================================

# Species SMILES (from data/species_data.py)
SPECIES_SMILES = {
    "EC":     "C1COC(=O)O1",
    "PC":     "CC1COC(=O)O1",
    "DMC":    "COC(=O)OC",
    "DEC":    "CCOC(=O)OCC",
    "EMC":    "CCOC(=O)OC",
    "EA":     "CCOC(=O)C",
    "MA":     "COC(=O)C",
    "FEC":    "FC1COC(=O)O1",
    "VC":     "C=C1COC(=O)O1",
}

# Species molecular data — mw (g/mol), density (g/mL) from CRC Handbook
SPECIES_MW = {
    "EC": 88.06, "PC": 102.09, "DMC": 90.08, "DEC": 118.13, "EMC": 104.10,
    "EA": 88.11, "MA": 74.08, "FEC": 106.05, "VC": 86.05,
    "LiPF6": 151.9, "LiFSI": 187.07, "LiTFSI": 287.09,
}
SPECIES_DENSITY = {
    "EC": 1.321, "PC": 1.205, "DMC": 1.069, "DEC": 0.975, "EMC": 1.007,
    "EA": 0.902, "MA": 0.934, "FEC": 1.454, "VC": 1.355,
}

# Salt ion components
SALT_IONS = {
    "LiPF6":  {"cation": "Li", "anion_smiles": "F[P-](F)(F)(F)(F)F",     "anion_name": "PF6"},
    "LiFSI":  {"cation": "Li", "anion_smiles": "FS(=O)(=O)[N-]S(=O)(=O)F", "anion_name": "FSI"},
    "LiTFSI": {"cation": "Li", "anion_smiles": "[O-]S(=O)(=O)N(S(=O)(=O)C(F)(F)F)C(F)(F)F", "anion_name": "TFSI"},
}

# LAMMPS element ordering — determines atom type numbering
# Must include ALL elements present in the composition
ELEMENT_ORDER = ["H", "Li", "C", "N", "O", "F", "P", "S"]
# Masses from IUPAC 2021 (g/mol)
ELEMENT_MASS = {"H": 1.008, "Li": 6.941, "C": 12.011, "N": 14.007,
                "O": 15.999, "F": 18.998, "P": 30.974, "S": 32.06}

# Simulation parameters
N_MOLECULES = 256            # total solvent+additive molecules — balance of stats vs speed
NPT_STEPS = 1_000_000       # 1 ns NPT equilibration at 1 fs timestep
NVT_STEPS = 5_000_000       # 5 ns NVT production
TEMPERATURE_K = 300.0        # standard temperature
TIMESTEP_FS = 1.0            # 1 fs — standard for BAMBOO
DUMP_INTERVAL = 1000         # save every 1 ps (1000 × 1 fs)
# BAMBOO pair_style cutoffs from benchmark/in.lammps (Å):
#   arg1 = NN cutoff (5.0), arg2 = Coulomb cutoff (5.0),
#   arg3 = dispersion cutoff (10.0), arg4 = flag (1)
BAMBOO_NN_CUTOFF = 5.0       # Å — GNN message-passing radius
BAMBOO_COUL_CUTOFF = 5.0     # Å — short-range Coulomb cutoff
BAMBOO_DISP_CUTOFF = 10.0    # Å — D3 dispersion cutoff


# %% CELL 3 — BUILD SIMULATION BOX ==========================================
# Convert recipe → molecule counts → 3D box via Packmol

from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
import tempfile

SIM_DIR = f"{WORK_DIR}/simulations"
os.makedirs(SIM_DIR, exist_ok=True)

# Create a unique directory for this recipe
recipe_hash = hash(json.dumps(RECIPE, sort_keys=True)) % 10**8
RUN_DIR = f"{SIM_DIR}/run_{recipe_hash:08d}"
os.makedirs(RUN_DIR, exist_ok=True)
logger.info(f"Simulation directory: {RUN_DIR}")


def recipe_to_molecule_counts(recipe: dict, n_molecules: int) -> dict:
    """
    Convert recipe (vol fracs, mol/L, wt fracs) to integer molecule counts.

    Returns dict mapping species_name → count, including split ions for salts.
    """
    counts = {}

    # 1. Solvent counts from volume fractions
    total_solvent = 0
    for solvent, vfrac in recipe["solvents"].items():
        n = max(1, round(n_molecules * vfrac))
        counts[solvent] = n
        total_solvent += n

    # 2. Additive counts from weight fractions (relative to total mass)
    total_mass = sum(counts[s] * SPECIES_MW[s] for s in counts)
    for additive, wfrac in (recipe["additives"].items() if "additives" in recipe else []):
        # wfrac = mass_additive / (mass_total + mass_additive)
        # n_add = wfrac * total_mass / ((1 - wfrac) * MW_additive)
        n_add = max(1, round(wfrac * total_mass / ((1 - wfrac) * SPECIES_MW[additive])))
        counts[additive] = n_add

    # 3. Salt counts from molarity (mol/L)
    # Estimate box volume from solvent counts + densities
    vol_ml = sum(counts[s] * SPECIES_MW[s] / SPECIES_DENSITY[s]
                 for s in counts if s in SPECIES_DENSITY)
    # Convert: molecules → moles (÷ Avogadro), volume in mL → L (÷ 1000)
    AVOGADRO = 6.022e23
    vol_L = vol_ml / AVOGADRO / 1000  # volume in liters per molecule

    for salt, molarity in recipe["salts"].items():
        n_salt = max(1, round(molarity * vol_L * AVOGADRO))
        # Each salt dissociates into cation + anion
        counts[f"Li_from_{salt}"] = n_salt
        counts[f"{SALT_IONS[salt]['anion_name']}"] = n_salt

    logger.info(f"Molecule counts (total {sum(counts.values())}):")
    for name, n in sorted(counts.items()):
        logger.info(f"  {name:20s}: {n}")

    return counts


def smiles_to_xyz_string(smiles: str, name: str) -> tuple[str, int]:
    """Generate a single 3D conformer as XYZ string. Returns (xyz_str, n_atoms)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    lines = [f"{n_atoms}", name]
    for i in range(n_atoms):
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        pos = conf.GetAtomPosition(i)
        lines.append(f"{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines), n_atoms


def get_species_smiles(species_name: str) -> str:
    """Get SMILES for a species (solvent/additive or ion fragment)."""
    if species_name.startswith("Li_from_"):
        return "[Li+]"
    for salt, info in SALT_IONS.items():
        if species_name == info["anion_name"]:
            return info["anion_smiles"]
    if species_name in SPECIES_SMILES:
        return SPECIES_SMILES[species_name]
    raise KeyError(f"Unknown species: {species_name}")


def build_box(counts: dict) -> str:
    """
    Build simulation box using Packmol.
    Returns path to output XYZ file.
    """
    # Estimate box size from mass and target density (~1.2 g/mL for typical electrolyte)
    AMU_TO_G = 1.66054e-24  # 1 amu in grams
    total_mass_g = 0
    species_xyz = {}

    for name, n in counts.items():
        smiles = get_species_smiles(name)
        xyz_str, n_atoms = smiles_to_xyz_string(smiles, name)
        species_xyz[name] = (xyz_str, n_atoms)

        # Rough mass estimate — count atoms by element weight
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse {smiles}")
        mol = Chem.AddHs(mol)
        mol_mass = sum(ELEMENT_MASS[mol.GetAtomWithIdx(i).GetSymbol()]
                       for i in range(mol.GetNumAtoms()))
        total_mass_g += n * mol_mass * AMU_TO_G

    # Target density ~1.2 g/mL for carbonate electrolytes (initial guess, NPT corrects)
    INITIAL_DENSITY_G_ML = 1.2  # g/mL — conservative; NPT will equilibrate to true density
    vol_cm3 = total_mass_g / INITIAL_DENSITY_G_ML
    vol_ang3 = vol_cm3 * 1e24  # cm³ → ų
    box_side = vol_ang3 ** (1/3)

    logger.info(f"Estimated box: mass={total_mass_g/AMU_TO_G:.0f} amu, "
                f"side={box_side:.1f} Å, density={INITIAL_DENSITY_G_ML} g/mL (initial)")

    # Write XYZ files and Packmol input
    xyz_dir = f"{RUN_DIR}/xyz"
    os.makedirs(xyz_dir, exist_ok=True)

    packmol_lines = [
        "tolerance 2.0",
        "filetype xyz",
        f"output {RUN_DIR}/box.xyz",
    ]

    MARGIN = 1.5  # Å — keep atoms away from box edges for PBC
    for name, n in counts.items():
        xyz_str, _ = species_xyz[name]
        xyz_path = f"{xyz_dir}/{name}.xyz"
        with open(xyz_path, "w") as f:
            f.write(xyz_str)

        packmol_lines.extend([
            f"structure {xyz_path}",
            f"  number {n}",
            f"  inside box {MARGIN} {MARGIN} {MARGIN} "
            f"{box_side - MARGIN:.2f} {box_side - MARGIN:.2f} {box_side - MARGIN:.2f}",
            "end structure",
        ])

    packmol_input = f"{RUN_DIR}/packmol.inp"
    with open(packmol_input, "w") as f:
        f.write("\n".join(packmol_lines))

    # Run Packmol
    result = subprocess.run(
        f"{PACKMOL_BIN} < {packmol_input}",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(f"{RUN_DIR}/box.xyz"):
        logger.error(f"Packmol failed: {result.stderr[-500:]}")
        raise RuntimeError("Packmol box building failed")

    logger.info(f"Box built: {RUN_DIR}/box.xyz")
    return f"{RUN_DIR}/box.xyz", box_side


def assign_charges(counts: dict) -> dict:
    """
    Assign partial charges per atom in each molecular species.
    Uses Open Babel Gasteiger charges as a fast approximation.
    Returns dict: species_name → list of per-atom charges.
    """
    from openbabel import openbabel as ob

    charges = {}
    for name in counts:
        smiles = get_species_smiles(name)

        obconv = ob.OBConversion()
        obconv.SetInFormat("smi")
        mol = ob.OBMol()
        obconv.ReadString(mol, smiles)
        mol.AddHydrogens()

        # Generate 3D coordinates for charge computation
        builder = ob.OBBuilder()
        builder.Build(mol)

        # Gasteiger charges
        charge_model = ob.OBChargeModel.FindType("gasteiger")
        if charge_model is None:
            raise RuntimeError("Gasteiger charge model not available in Open Babel")
        charge_model.ComputeCharges(mol)

        atom_charges = []
        for i in range(mol.NumAtoms()):
            atom = mol.GetAtom(i + 1)  # 1-indexed
            atom_charges.append(atom.GetPartialCharge())

        charges[name] = atom_charges
        logger.info(f"  {name}: {len(atom_charges)} atoms, "
                     f"net charge={sum(atom_charges):.3f}")

    return charges


def xyz_to_lammps_data(
    xyz_path: str,
    box_side: float,
    counts: dict,
    charges: dict,
) -> str:
    """
    Convert Packmol XYZ output to LAMMPS data file format.

    Returns path to written in.data file.
    """
    # Read XYZ
    with open(xyz_path) as f:
        lines = f.readlines()
    n_total = int(lines[0].strip())
    atom_data = []
    for line in lines[2:2+n_total]:
        parts = line.split()
        sym = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atom_data.append((sym, x, y, z))

    # Map elements to LAMMPS atom types
    elements_present = sorted(set(a[0] for a in atom_data),
                               key=lambda e: ELEMENT_ORDER.index(e))
    elem_to_type = {e: i+1 for i, e in enumerate(elements_present)}
    logger.info(f"Elements: {elements_present}")
    logger.info(f"Type mapping: {elem_to_type}")

    # Build atom entries with molecular IDs and charges
    # We need to assign molecular IDs and per-atom charges by matching
    # the Packmol output order to our species order.

    # Packmol writes molecules in the order they appear in the input.
    # Count atoms per species to build the mapping.
    species_order = []
    for name, n in counts.items():
        smiles = get_species_smiles(name)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse {smiles}")
        mol = Chem.AddHs(mol)
        n_atoms_per_mol = mol.GetNumAtoms()
        for copy_idx in range(n):
            species_order.append((name, n_atoms_per_mol, copy_idx))

    # Assign atom entries
    data_path = f"{RUN_DIR}/in.data"
    atom_idx = 0
    mol_id = 0
    atom_lines = []

    for name, n_atoms_per_mol, copy_idx in species_order:
        mol_id += 1
        mol_charges = charges[name]
        for j in range(n_atoms_per_mol):
            if atom_idx >= len(atom_data):
                raise ValueError(f"Atom count mismatch: expected more atoms for {name}")
            sym, x, y, z = atom_data[atom_idx]
            atype = elem_to_type[sym]
            q = mol_charges[j] if j < len(mol_charges) else 0.0
            atom_lines.append(
                f"{atom_idx+1} {mol_id} {atype} {q:.6f} {x:.4f} {y:.4f} {z:.4f} # {sym}"
            )
            atom_idx += 1

    if atom_idx != n_total:
        raise ValueError(f"Atom count mismatch: processed {atom_idx}, expected {n_total}")

    n_types = len(elements_present)

    with open(data_path, "w") as f:
        f.write(f"BAMBOO electrolyte: {json.dumps(RECIPE, sort_keys=True)}\n\n")
        f.write(f"{n_total} atoms\n\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"0 {box_side:.4f} xlo xhi\n")
        f.write(f"0 {box_side:.4f} ylo yhi\n")
        f.write(f"0 {box_side:.4f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for elem in elements_present:
            f.write(f"{elem_to_type[elem]} {ELEMENT_MASS[elem]:.3f} # {elem}\n")
        f.write("\nAtoms\n\n")
        for line in atom_lines:
            f.write(line + "\n")

    logger.info(f"LAMMPS data file: {data_path} ({n_total} atoms, {mol_id} molecules)")
    return data_path, elements_present


# Execute box building
counts = recipe_to_molecule_counts(RECIPE, N_MOLECULES)
box_xyz, box_side = build_box(counts)
mol_charges = assign_charges(counts)
data_path, elements_present = xyz_to_lammps_data(box_xyz, box_side, counts, mol_charges)


# %% CELL 4 — WRITE LAMMPS INPUT + RUN NPT ==================================
# NPT equilibration: 1 ns at 300 K, 1 atm
# Gets the density right before production NVT.

MODEL_PT = f"{WORK_DIR}/lammps_bamboo/paper_new_disp.pt"
if not os.path.exists(MODEL_PT):
    MODEL_PT = f"{WORK_DIR}/lammps_bamboo/benchmark.pt"
logger.info(f"Using model: {MODEL_PT}")

# Element string for pair_coeff (must match LAMMPS atom types in order)
element_str = " ".join(e if e != "Li" else "LI" for e in elements_present)

# Check for existing NPT output (resume support)
npt_data = f"{RUN_DIR}/npt.data"
npt_traj = f"{RUN_DIR}/dump_npt.lammpstrj"

if os.path.exists(npt_data):
    logger.info("NPT output found — skipping equilibration")
else:
    # Write in.lammps for NPT
    lammps_input = f"""# BAMBOO NPT equilibration
units             real
atom_style        full
atom_modify       map yes
newton off
read_data {data_path}

pair_style      bamboo {BAMBOO_NN_CUTOFF} {BAMBOO_COUL_CUTOFF} {BAMBOO_DISP_CUTOFF} 1
pair_coeff      {MODEL_PT} {element_str}

kspace_style    pppm 1.0e-6
kspace_modify   mesh 64 64 64

neighbor        3 bin
neigh_modify    delay 0 every 1 check yes
timestep        {TIMESTEP_FS}

thermo          {DUMP_INTERVAL}
thermo_style    custom step temp press vol density pe ke etotal

dump 1 all custom {DUMP_INTERVAL} {npt_traj} id type xu yu zu x y z ix iy iz vx vy vz fx fy fz q

velocity all create {TEMPERATURE_K} 4928459
velocity all zero linear

# Quick minimization
minimize 0.0 0.0 1000 100000

# NPT: {NPT_STEPS} steps = {NPT_STEPS * TIMESTEP_FS / 1e6:.1f} ns
fix 1 all npt temp {TEMPERATURE_K} {TEMPERATURE_K} 100 iso 0 0 1000

run {NPT_STEPS}
write_data {npt_data}
"""
    lammps_npt_path = f"{RUN_DIR}/in_npt.lammps"
    with open(lammps_npt_path, "w") as f:
        f.write(lammps_input)

    logger.info(f"Running NPT equilibration ({NPT_STEPS} steps = "
                f"{NPT_STEPS * TIMESTEP_FS / 1e6:.1f} ns)...")

    _run(f"cd {RUN_DIR} && {LAMMPS_BIN} -k on g 1 -sf kk "
         f"-in {lammps_npt_path} -log {RUN_DIR}/log_npt.lammps "
         f"> {RUN_DIR}/out_npt.log 2>&1")

    logger.info("NPT equilibration complete")


# %% CELL 5 — RUN NVT PRODUCTION =============================================
# NVT production: 5 ns at 300 K
# Trajectory saved for conductivity extraction.

nvt_traj = f"{RUN_DIR}/dump_nvt.lammpstrj"
nvt_data = f"{RUN_DIR}/nvt.data"

if os.path.exists(nvt_data):
    logger.info("NVT output found — skipping production")
else:
    # Write NVT input — reads from NPT equilibrated state
    lammps_nvt = f"""# BAMBOO NVT production
units             real
atom_style        full
atom_modify       map yes
newton off
read_data {npt_data}

pair_style      bamboo {BAMBOO_NN_CUTOFF} {BAMBOO_COUL_CUTOFF} {BAMBOO_DISP_CUTOFF} 1
pair_coeff      {MODEL_PT} {element_str}

kspace_style    pppm 1.0e-6
kspace_modify   mesh 64 64 64

neighbor        3 bin
neigh_modify    delay 0 every 1 check yes
timestep        {TIMESTEP_FS}

thermo          {DUMP_INTERVAL}
thermo_style    custom step temp press vol density pe ke etotal

dump 1 all custom {DUMP_INTERVAL} {nvt_traj} id type xu yu zu x y z ix iy iz vx vy vz fx fy fz q

# NVT: {NVT_STEPS} steps = {NVT_STEPS * TIMESTEP_FS / 1e6:.1f} ns
fix 1 all nvt temp {TEMPERATURE_K} {TEMPERATURE_K} 10

run {NVT_STEPS}
write_data {nvt_data}
"""
    lammps_nvt_path = f"{RUN_DIR}/in_nvt.lammps"
    with open(lammps_nvt_path, "w") as f:
        f.write(lammps_nvt)

    logger.info(f"Running NVT production ({NVT_STEPS} steps = "
                f"{NVT_STEPS * TIMESTEP_FS / 1e6:.1f} ns)...")

    _run(f"cd {RUN_DIR} && {LAMMPS_BIN} -k on g 1 -sf kk "
         f"-in {lammps_nvt_path} -log {RUN_DIR}/log_nvt.lammps "
         f"> {RUN_DIR}/out_nvt.log 2>&1")

    logger.info("NVT production complete")


# %% CELL 6 — EXTRACT CONDUCTIVITY ==========================================
# Parse NVT trajectory and compute ionic conductivity via the collective
# Einstein relation: σ = (e²/6VkBT) × d/dt⟨|M(t)|²⟩
# where M(t) = Σᵢ zᵢ[rᵢ(t) - rᵢ(0)] summed over all ions.

from scipy import constants

# Physical constants
E_CHARGE = constants.e                    # 1.602e-19 C
KB = constants.Boltzmann                  # 1.381e-23 J/K
ANGSTROM_M = 1e-10                        # Å → m
FS_TO_S = 1e-15                           # fs → s


def parse_lammps_trajectory(traj_path: str, elements_present: list[str]):
    """
    Parse LAMMPS custom dump file.

    Returns:
        positions: (n_frames, n_atoms, 3) unwrapped positions in Å
        types: (n_atoms,) atom type indices (1-based)
        box_dims: (3,) box side lengths in Å
        timesteps: (n_frames,) step numbers
    """
    logger.info(f"Parsing trajectory: {traj_path}")

    frames_pos = []
    types = None
    box_dims = None
    timesteps = []

    with open(traj_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            step = int(lines[i+1].strip())
            timesteps.append(step)
            i += 2
        elif "ITEM: NUMBER OF ATOMS" in lines[i]:
            n_atoms = int(lines[i+1].strip())
            i += 2
        elif "ITEM: BOX BOUNDS" in lines[i]:
            xlo, xhi = map(float, lines[i+1].split())
            ylo, yhi = map(float, lines[i+2].split())
            zlo, zhi = map(float, lines[i+3].split())
            box_dims = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
            i += 4
        elif "ITEM: ATOMS" in lines[i]:
            header = lines[i].split()[2:]  # column names
            xu_idx = header.index("xu")
            yu_idx = header.index("yu")
            zu_idx = header.index("zu")
            type_idx = header.index("type")
            q_idx = header.index("q")

            frame_data = np.zeros((n_atoms, 3))
            frame_types = np.zeros(n_atoms, dtype=int)
            frame_charges = np.zeros(n_atoms)

            for j in range(n_atoms):
                vals = lines[i + 1 + j].split()
                atom_id = int(vals[0]) - 1  # 0-indexed
                frame_types[atom_id] = int(vals[type_idx])
                frame_data[atom_id, 0] = float(vals[xu_idx])
                frame_data[atom_id, 1] = float(vals[yu_idx])
                frame_data[atom_id, 2] = float(vals[zu_idx])
                frame_charges[atom_id] = float(vals[q_idx])

            frames_pos.append(frame_data)
            if types is None:
                types = frame_types
                charges = frame_charges

            i += 1 + n_atoms
        else:
            i += 1

    positions = np.array(frames_pos)
    timesteps = np.array(timesteps)

    logger.info(f"Parsed {len(timesteps)} frames, {positions.shape[1]} atoms, "
                f"box={box_dims}")

    return positions, types, charges, box_dims, timesteps


def compute_conductivity(
    positions: np.ndarray,
    types: np.ndarray,
    charges: np.ndarray,
    box_dims: np.ndarray,
    timesteps: np.ndarray,
    elements_present: list[str],
) -> dict:
    """
    Compute ionic conductivity via collective Einstein relation.

    σ = (e²/6VkBT) × lim_{t→∞} d/dt ⟨|M(t)|²⟩

    M(t) = Σᵢ zᵢ [rᵢ(t) - rᵢ(0)]  (collective dipole displacement)
    """
    n_frames, n_atoms, _ = positions.shape
    dt_fs = (timesteps[1] - timesteps[0]) * TIMESTEP_FS  # time between saved frames

    # Identify ions: Li type and any charged species
    # Li is always type = elements_present.index("Li") + 1
    li_type = elements_present.index("Li") + 1 if "Li" in elements_present else None

    # Use charges from trajectory — ions have |q| > 0.3
    ion_mask = np.abs(charges) > 0.3
    ion_indices = np.where(ion_mask)[0]
    ion_charges = np.round(charges[ion_mask]).astype(int)  # nearest integer formal charge

    n_ions = len(ion_indices)
    logger.info(f"Found {n_ions} ions (|q| > 0.3)")
    for idx in ion_indices[:10]:
        elem = elements_present[types[idx] - 1]
        logger.info(f"  atom {idx}: type={types[idx]} ({elem}), q={charges[idx]:.3f}")

    if n_ions < 4:
        logger.warning(f"Only {n_ions} ions — results will be noisy")

    # Compute collective dipole displacement M(t)
    # M(t) = Σᵢ zᵢ × [rᵢ(t) - rᵢ(0)]
    ion_positions = positions[:, ion_indices, :]  # (n_frames, n_ions, 3)

    # Compute displacement from t=0
    displacements = ion_positions - ion_positions[0:1]  # (n_frames, n_ions, 3)

    # Charge-weighted sum → collective dipole displacement
    M = np.sum(ion_charges[None, :, None] * displacements, axis=1)  # (n_frames, 3)

    # MSD of collective dipole: ⟨|M(t)|²⟩ via multiple time origins
    max_lag = n_frames // 2
    msd_collective = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    ORIGIN_STRIDE = max(1, n_frames // 200)  # ~200 time origins for statistics

    for t0 in range(0, n_frames - max_lag, ORIGIN_STRIDE):
        for lag in range(1, max_lag):
            dM = M[t0 + lag] - M[t0]
            msd_collective[lag] += np.dot(dM, dM)
            counts[lag] += 1

    # Avoid division by zero
    valid = counts > 0
    msd_collective[valid] /= counts[valid]

    # Time array for lags
    lag_times_fs = np.arange(max_lag) * dt_fs
    lag_times_s = lag_times_fs * FS_TO_S

    # Fit slope of MSD vs time in the diffusive regime (10-80% of trajectory)
    fit_start = max_lag // 10
    fit_end = int(max_lag * 0.8)

    if fit_end - fit_start < 10:
        logger.warning("Not enough data points for reliable fit")
        fit_start = 1
        fit_end = max_lag - 1

    slope_ang2_fs, intercept = np.polyfit(
        lag_times_fs[fit_start:fit_end],
        msd_collective[fit_start:fit_end],
        1,
    )
    slope_m2_s = slope_ang2_fs * (ANGSTROM_M**2) / FS_TO_S

    # σ = (e²/6VkBT) × d|M|²/dt
    V_m3 = box_dims[0] * box_dims[1] * box_dims[2] * (ANGSTROM_M**3)
    sigma_S_m = (E_CHARGE**2 / (6 * V_m3 * KB * TEMPERATURE_K)) * slope_m2_s
    sigma_mS_cm = sigma_S_m * 0.1  # S/m → mS/cm: ×1000(S→mS) ÷ 100(m→cm) = ×0.1... wait

    # S/m → mS/cm: 1 S/m = 10 mS/cm
    sigma_mS_cm = sigma_S_m * 10.0

    # Also compute single-particle (Nernst-Einstein) for comparison
    ion_msd_per_ion = np.zeros(max_lag)
    counts_ne = np.zeros(max_lag)
    for t0 in range(0, n_frames - max_lag, ORIGIN_STRIDE):
        for lag in range(1, max_lag):
            dr = ion_positions[t0 + lag] - ion_positions[t0]  # (n_ions, 3)
            ion_msd_per_ion[lag] += np.mean(np.sum(dr**2, axis=1))
            counts_ne[lag] += 1
    valid_ne = counts_ne > 0
    ion_msd_per_ion[valid_ne] /= counts_ne[valid_ne]

    slope_ne, _ = np.polyfit(
        lag_times_fs[fit_start:fit_end],
        ion_msd_per_ion[fit_start:fit_end],
        1,
    )
    D_self_m2_s = slope_ne * (ANGSTROM_M**2) / FS_TO_S / 6  # MSD = 6Dt

    # Nernst-Einstein: σ_NE = n_ions × e² × D / (V × kB × T)
    n_density = n_ions / V_m3  # number density (ions/m³)
    sigma_ne_S_m = n_density * E_CHARGE**2 * D_self_m2_s / (KB * TEMPERATURE_K)
    sigma_ne_mS_cm = sigma_ne_S_m * 10.0

    # Haven ratio: σ_collective / σ_NE (should be < 1 for anti-correlated ion motion)
    haven_ratio = sigma_mS_cm / sigma_ne_mS_cm if sigma_ne_mS_cm > 0 else float("inf")

    # Density from box
    total_mass_amu = sum(
        ELEMENT_MASS[elements_present[types[i] - 1]]
        for i in range(n_atoms)
    )
    density_g_ml = (total_mass_amu * 1.66054e-24) / (V_m3 * 1e6)

    results = {
        "conductivity_mS_cm": float(sigma_mS_cm),
        "conductivity_ne_mS_cm": float(sigma_ne_mS_cm),
        "haven_ratio": float(haven_ratio),
        "D_self_m2_s": float(D_self_m2_s),
        "msd_slope_ang2_fs": float(slope_ang2_fs),
        "density_g_ml": float(density_g_ml),
        "n_ions": int(n_ions),
        "n_frames": int(n_frames),
        "box_side_ang": float(box_dims[0]),
        "temperature_k": float(TEMPERATURE_K),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"CONDUCTIVITY RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  σ (collective Einstein): {sigma_mS_cm:.3f} mS/cm")
    logger.info(f"  σ (Nernst-Einstein):     {sigma_ne_mS_cm:.3f} mS/cm")
    logger.info(f"  Haven ratio:             {haven_ratio:.3f}")
    logger.info(f"  D_self:                  {D_self_m2_s:.3e} m²/s")
    logger.info(f"  Density:                 {density_g_ml:.4f} g/mL")
    logger.info(f"  Box side:                {box_dims[0]:.2f} Å")
    logger.info(f"  N ions:                  {n_ions}")
    logger.info(f"  N frames:                {n_frames}")
    if TARGET_CONDUCTIVITY > 0:
        error_pct = abs(sigma_mS_cm - TARGET_CONDUCTIVITY) / TARGET_CONDUCTIVITY * 100
        logger.info(f"  Target:                  {TARGET_CONDUCTIVITY:.2f} mS/cm")
        logger.info(f"  Error:                   {error_pct:.1f}%")
    logger.info(f"{'='*60}")

    return results


# Run analysis
positions, types, charges, box_dims, timesteps = parse_lammps_trajectory(
    nvt_traj, elements_present)
results = compute_conductivity(
    positions, types, charges, box_dims, timesteps, elements_present)

# Save results
result_path = f"{RUN_DIR}/conductivity_result.json"
output = {
    "recipe": RECIPE,
    "results": results,
    "target_conductivity_mS_cm": TARGET_CONDUCTIVITY,
}
with open(result_path, "w") as f:
    json.dump(output, f, indent=2)
logger.info(f"Results saved to {result_path}")

print(f"\n\nσ = {results['conductivity_mS_cm']:.2f} mS/cm "
      f"(target: {TARGET_CONDUCTIVITY:.2f} mS/cm)")
