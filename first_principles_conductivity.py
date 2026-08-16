"""Bulk dc conductivity from the Hamiltonian Green-Kubo resolvent.

The executable refines composition-preserving periodic cells, static canonical
integrals, a nested phase-space basis, and the zero-frequency continuation before it
emits the scalar conductivity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import cache
import hashlib
from itertools import combinations_with_replacement
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Callable, Protocol
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import erfcinv
import torch
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from constants import (
    E_CHARGE,
    EPS_0,
    K_B,
    KG_M3_PER_G_ML,
    LITER_PER_M3,
    N_A,
    S_M_TO_MS_CM,
)
from conductivity.physical_library.library_io import load_physical_library
from conductivity.physical_library.physical_objects import (
    LJ_ATTRACTIVE_EXPONENT,
    LJ_REPULSIVE_EXPONENT_MULTIPLIER,
)
from electrolyte_model import ElectrolyteRecipeModel
from species_data import ADDITIVES, SALTS
from utils.strict_validation import read_json_object, write_json_object

Array = np.ndarray
CARTESIAN_DIMENSION = 3
TORCH_DTYPE = torch.float64
functorch_config.donated_buffer = False
if platform.system() == "Darwin" and platform.machine() == "arm64":
    ARM_CROSS_COMPILER = Path("/usr/local/opt/llvm/bin/clang++")
    APPLE_SDK_ROOT = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    if not ARM_CROSS_COMPILER.is_file() or not APPLE_SDK_ROOT.is_dir():
        raise RuntimeError(
            "native Torch Inductor requires LLVM clang++ and the Apple SDK"
        )
    os.environ["CCC_OVERRIDE_OPTIONS"] = (
        f"^{APPLE_SDK_ROOT} ^-isysroot ^--target=arm64-apple-darwin"
    )
    inductor_config.cpp.cxx = (str(ARM_CROSS_COMPILER),)
    inductor_config.cpp.threads = os.cpu_count()
    # Torch's shared Apple PCH can retain an obsolete header timestamp across runs.
    inductor_config.cpp_cache_precompile_headers = False
MILP_FEASIBILITY_TOLERANCE = 100.0 * math.sqrt(np.finfo(float).eps)
# Quarter scaling separates the lambda and square-root-lambda energy coefficients.
COMPONENT_DECOMPOSITION_LAMBDA = 0.25
# The standard C2 quintic switch is one at x=0 with zero first and second
# derivatives at both endpoints and zero at x=1.
QUINTIC_SWITCH_CUBIC_COEFFICIENT = 10.0  # C2 endpoint polynomial coefficient.
QUINTIC_SWITCH_QUARTIC_COEFFICIENT = 15.0  # C2 endpoint polynomial coefficient.
QUINTIC_SWITCH_QUINTIC_COEFFICIENT = 6.0  # C2 endpoint polynomial coefficient.
QUINTIC_SWITCH_CUBIC_POWER = 3  # Cubic term of the C2 endpoint polynomial.
QUINTIC_SWITCH_QUARTIC_POWER = 4  # Quartic term of the C2 endpoint polynomial.
QUINTIC_SWITCH_QUINTIC_POWER = 5  # Quintic term of the C2 endpoint polynomial.
# Integrating 4 epsilon (sigma/r)^n over 4 pi r^2 dr gives 16 pi epsilon.
LJ_RADIAL_ENERGY_INTEGRAL_PREFACTOR = 16.0 * math.pi  # 4 epsilon times 4 pi.
REST_INTERACTION_CLASS_COUNT = 3


@cache
def _physical_library_records():
    return load_physical_library(Path(__file__).parent / "physical_library")


def molecule_with_explicit_hydrogens(smiles: str):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    return Chem.AddHs(molecule)


def explicit_hydrogen_atom_count(smiles: str) -> int:
    return int(molecule_with_explicit_hydrogens(smiles).GetNumAtoms())


def molar_mass_g_mol(smiles: str) -> float:
    from rdkit.Chem.Descriptors import MolWt

    return float(MolWt(molecule_with_explicit_hydrogens(smiles)))


@dataclass(frozen=True)
class DynamicsSettings:
    initial_relaxation_maximum_force_evaluations: int
    initial_relaxation_timestep_s: float
    initial_relaxation_maximum_timestep_s: float
    initial_relaxation_initial_damping: float
    initial_relaxation_timestep_increase: float
    initial_relaxation_timestep_decrease: float
    initial_relaxation_damping_decrease: float
    initial_relaxation_positive_power_steps: int
    initial_relaxation_maximum_elapsed_s: float
    initial_relaxation_maximum_stagnant_iterations: int
    initial_relaxation_minimum_force_improvement_fraction: float
    initial_relaxation_progress_stride: int
    force_batch_size: int
    resolvent_operator_batch_size: int
    initial_force_tolerance_N: float
    equilibrium_sample_count: int
    equilibrium_chain_count: int
    equilibrium_maximum_refinement_batches: int
    hamiltonian_timestep_s: float
    ionic_hrex_lambdas: tuple[float, ...]
    hmc_steps_min: int
    hmc_steps_max: int
    hmc_momentum_persistence: float
    hmc_full_refresh_stride: int
    exchange_stride: int
    hrex_warmup_cycle_count: int
    hrex_measurement_stride: int
    hrex_block_cycle_count: int
    solvent_volume_fraction_tolerance: float
    salt_molarity_tolerance_mol_L: float
    additive_weight_fraction_tolerance: float
    maximum_explicit_molecule_count: int
    maximum_atom_count: int


@dataclass(frozen=True)
class NumericalSettings:
    initial_placement_attempts_per_molecule: int
    ewald_splitting_per_m: float
    ewald_reciprocal_relative_tolerance: float
    ewald_maximum_box_length_m: float
    lennard_jones_switch_start_m: float
    lennard_jones_cutoff_m: float
    dispersion_tail_quadrature_order: int
    polarization_residual_tolerance_V_m: float
    force_difference_step_m: float
    force_consistency_relative_tolerance: float
    resolvent_eta_values_s_inv: tuple[float, ...]
    equilibrium_standard_error_multiplier: float
    conductivity_tolerance_S_m: float
    minimum_interatomic_contact_ratio: float


@dataclass(frozen=True)
class IntegerRecipeRealization:
    formula_unit_counts: tuple[tuple[str, int], ...]
    explicit_species_counts: tuple[tuple[str, int], ...]
    explicit_molecule_count: int
    atom_count: int
    cell_mass_kg: float
    density_conditioned_volume_m3: float
    realized_solvent_volume_fractions: tuple[tuple[str, float], ...]
    realized_salt_molarities_mol_L: tuple[tuple[str, float], ...]
    realized_additive_weight_fractions: tuple[tuple[str, float], ...]
    native_unit_deviations: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ConductivityResult:
    conductivity_S_m: float
    conductivity_lower_bound_S_m: float
    conductivity_upper_bound_S_m: float
    conditioned_volume_m3: float
    conditioned_density_g_cm3: float
    thermodynamic_state: str
    density_source: str
    generator_name: str
    current_definition: str
    interval_definition: str
    interval_is_deterministic: bool
    basis_size: int
    basis_labels: tuple[str, ...]
    resolvent_intervals_S_m: tuple[tuple[float, float, float], ...]
    cell_conductivities_S_m: tuple[tuple[float, float, float, float], ...]
    basis_error_S_m: float
    eta_continuation_error_S_m: float
    finite_cell_error_S_m: float
    equilibrium_error_S_m: float
    linear_solve_error_S_m: float
    linear_solve_relative_residual: float
    equilibrium_sample_count: int
    equilibrium_chain_count: int
    conductivity_mcse_S_m: float
    finite_eta_resolvent_precision_reached: bool
    conductivity_precision_reached: bool
    realized_formula_unit_counts: tuple[tuple[str, int], ...]
    realized_molecule_counts: tuple[tuple[str, int], ...]
    realized_atom_count: int
    realized_solvent_volume_fractions: tuple[tuple[str, float], ...]
    realized_salt_molarities_mol_L: tuple[tuple[str, float], ...]
    realized_additive_weight_fractions: tuple[tuple[str, float], ...]
    realized_native_unit_deviations: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class HamiltonianPhaseSpaceSamples:
    configurations_m: Array
    box_vectors_m: Array
    momenta_kg_m_s: Array
    forces_N: Array
    chain_indices: Array


@dataclass(frozen=True)
class HamiltonianBasisEvaluation:
    basis_level: int
    basis_labels: tuple[str, ...]
    basis_values: Array
    generator_values: Array
    negative_generator_squared_values: Array
    current_values: Array
    chain_indices: Array


@dataclass(frozen=True)
class HamiltonianResolventEstimate:
    resolvent_iterate_S_m: float
    lower_bound_S_m: float
    upper_bound_S_m: float
    intervals_S_m: tuple[tuple[float, float, float], ...]
    basis_error_S_m: float
    eta_continuation_error_S_m: float
    equilibrium_error_S_m: float
    linear_solve_error_S_m: float
    linear_solve_relative_residual: float
    conductivity_mcse_S_m: float
    basis_size: int
    basis_labels: tuple[str, ...]
    finite_eta_precision_reached: bool


@dataclass(frozen=True)
class MolecularSystem:
    positions_m: Array
    box_vectors_m: Array
    masses_kg: Array
    charges_C: Array
    lj_sigma_m: Array
    lj_epsilon_J: Array
    polarizabilities_SI: Array
    molecule_index: Array
    molecule_atom_indices: tuple[Array, ...]
    molecule_species_names: tuple[str, ...]
    bonds: Array
    bond_force_constants_J_m2: Array
    bond_lengths_m: Array
    angles: Array
    angle_force_constants_J_rad2: Array
    angle_values_rad: Array
    torsions: tuple[
        tuple[int, int, int, int, tuple[tuple[float, int, float], ...]], ...
    ]
    nonbonded_mask: Array


def _molecular_system_from_checkpoint_record(record: dict) -> MolecularSystem:
    return MolecularSystem(
        positions_m=record["positions_m"],
        box_vectors_m=record["box_vectors_m"],
        masses_kg=record["masses_kg"],
        charges_C=record["charges_C"],
        lj_sigma_m=record["lj_sigma_m"],
        lj_epsilon_J=record["lj_epsilon_J"],
        polarizabilities_SI=record["polarizabilities_SI"],
        molecule_index=record["molecule_index"],
        molecule_atom_indices=tuple(record["molecule_atom_indices"]),
        molecule_species_names=tuple(record["molecule_species_names"]),
        bonds=record["bonds"],
        bond_force_constants_J_m2=record["bond_force_constants_J_m2"],
        bond_lengths_m=record["bond_lengths_m"],
        angles=record["angles"],
        angle_force_constants_J_rad2=record["angle_force_constants_J_rad2"],
        angle_values_rad=record["angle_values_rad"],
        torsions=tuple(record["torsions"]),
        nonbonded_mask=record["nonbonded_mask"],
    )


def _integer_recipe_realization_from_checkpoint_record(
    record: dict,
) -> IntegerRecipeRealization:
    return IntegerRecipeRealization(
        formula_unit_counts=tuple(record["formula_unit_counts"]),
        explicit_species_counts=tuple(record["explicit_species_counts"]),
        explicit_molecule_count=int(record["explicit_molecule_count"]),
        atom_count=int(record["atom_count"]),
        cell_mass_kg=float(record["cell_mass_kg"]),
        density_conditioned_volume_m3=float(record["density_conditioned_volume_m3"]),
        realized_solvent_volume_fractions=tuple(
            record["realized_solvent_volume_fractions"]
        ),
        realized_salt_molarities_mol_L=tuple(
            record["realized_salt_molarities_mol_L"]
        ),
        realized_additive_weight_fractions=tuple(
            record["realized_additive_weight_fractions"]
        ),
        native_unit_deviations=tuple(record["native_unit_deviations"]),
    )


@dataclass(frozen=True)
class TemperedEnergyComponents:
    fixed_J: float
    ion_ion_J: float
    ion_neutral_J: float


@dataclass(frozen=True)
class BatchedHamiltonianResult:
    energy_J: Array
    forces_N: Array
    fixed_energy_J: Array
    ion_ion_energy_J: Array
    ion_neutral_energy_J: Array
    polarization_residual_V_m: Array


@dataclass(frozen=True)
class BatchedEnergyComponents:
    fixed_energy_J: Array
    ion_ion_energy_J: Array
    ion_neutral_energy_J: Array
    polarization_residual_V_m: Array


@dataclass(frozen=True)
class BatchedPhysicalEnergyTerms:
    bonded_energy_J: Array
    lennard_jones_repulsion_energy_J: Array
    lennard_jones_attraction_energy_J: Array
    real_electrostatic_energy_J: Array
    reciprocal_electrostatic_energy_J: Array
    ewald_self_energy_J: Array
    electrostatic_exclusion_energy_J: Array
    polarization_energy_J: Array
    polarization_self_energy_J: Array
    polarization_exclusion_energy_J: Array


def minimum_image_displacement(displacement_m: Array, box_vectors_m: Array) -> Array:
    fractional = np.asarray(displacement_m) @ np.linalg.inv(box_vectors_m)
    fractional -= np.rint(fractional)
    return fractional @ box_vectors_m


def _torch_minimum_image(
    displacement_m: torch.Tensor, box_m: torch.Tensor
) -> torch.Tensor:
    inverse_box_m = torch.linalg.inv(box_m)
    broadcast_box_m = box_m
    while inverse_box_m.ndim < displacement_m.ndim + 1:
        inverse_box_m = inverse_box_m.unsqueeze(-3)
        broadcast_box_m = broadcast_box_m.unsqueeze(-3)
    fractional = torch.matmul(displacement_m.unsqueeze(-2), inverse_box_m).squeeze(-2)
    wrapped_fractional = fractional - torch.round(fractional)
    return torch.matmul(wrapped_fractional.unsqueeze(-2), broadcast_box_m).squeeze(-2)


def _random_rotation(random_generator: np.random.Generator) -> Array:
    quaternion = random_generator.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    scalar, x_value, y_value, z_value = quaternion
    return np.asarray(
        (
            (
                1 - 2 * (y_value**2 + z_value**2),
                2 * (x_value * y_value - scalar * z_value),
                2 * (x_value * z_value + scalar * y_value),
            ),
            (
                2 * (x_value * y_value + scalar * z_value),
                1 - 2 * (x_value**2 + z_value**2),
                2 * (y_value * z_value - scalar * x_value),
            ),
            (
                2 * (x_value * z_value - scalar * y_value),
                2 * (y_value * z_value + scalar * x_value),
                1 - 2 * (x_value**2 + y_value**2),
            ),
        )
    )


def _salt_ion_names(salt_name: str) -> tuple[str, str]:
    salt_record = SALTS[salt_name]
    cation_name = f"{salt_record['cation']}+"
    return cation_name, str(salt_record["anion"])


def select_integer_recipe_realization(
    recipe: ElectrolyteRecipeModel,
    liquid_density_kg_m3: float,
    solvent_volume_fraction_tolerance: float,
    salt_molarity_tolerance_mol_L: float,
    additive_weight_fraction_tolerance: float,
    minimum_explicit_molecule_count: int,
    maximum_explicit_molecule_count: int,
    maximum_atom_count: int,
) -> IntegerRecipeRealization:
    if liquid_density_kg_m3 <= 0.0:
        raise ValueError("liquid density must be positive")
    if (
        solvent_volume_fraction_tolerance <= 0.0
        or salt_molarity_tolerance_mol_L <= 0.0
        or additive_weight_fraction_tolerance <= 0.0
    ):
        raise ValueError("native-unit composition tolerances must be positive")
    if (
        minimum_explicit_molecule_count <= 0
        or maximum_explicit_molecule_count < minimum_explicit_molecule_count
        or maximum_atom_count <= 0
    ):
        raise ValueError("integer-realization size bounds are inconsistent")
    library = _physical_library_records()
    active_solvent_names = tuple(
        sorted(name for name, value in recipe.solvents.items() if value > 0.0)
    )
    active_salt_names = tuple(
        sorted(name for name, value in recipe.salts.items() if value > 0.0)
    )
    active_additive_names = tuple(
        sorted(name for name, value in recipe.additives.items() if value > 0.0)
    )
    formula_names = active_solvent_names + active_salt_names + active_additive_names
    formula_constituents: dict[str, tuple[str, ...]] = {}
    for formula_name in formula_names:
        if formula_name in recipe.solvents:
            formula_constituents[formula_name] = (formula_name,)
        if formula_name in recipe.salts:
            formula_constituents[formula_name] = _salt_ion_names(formula_name)
        if formula_name in recipe.additives:
            additive_record = ADDITIVES[formula_name]
            formula_constituents[formula_name] = (formula_name,)
            if "cation" in additive_record and "anion" in additive_record:
                formula_constituents[formula_name] = (
                    f"{additive_record['cation']}+",
                    str(additive_record["anion"]),
                )
    explicit_species_names = tuple(
        sorted(
            {
                species_name
                for constituents in formula_constituents.values()
                for species_name in constituents
            }
        )
    )
    formula_index = {name: index for index, name in enumerate(formula_names)}
    species_index = {name: index for index, name in enumerate(explicit_species_names)}
    stoichiometric_matrix = np.zeros((len(explicit_species_names), len(formula_names)))
    for formula_name, constituents in formula_constituents.items():
        for species_name in constituents:
            stoichiometric_matrix[
                species_index[species_name], formula_index[formula_name]
            ] += 1.0
    species_molar_masses_kg_mol = np.asarray(
        tuple(
            sum(
                float(site["mass_kg"])
                for site in library.species_records[species_name]["sites"]
            )
            * N_A
            for species_name in explicit_species_names
        )
    )
    species_atom_counts = np.asarray(
        tuple(
            len(library.species_records[species_name]["sites"])
            for species_name in explicit_species_names
        )
    )
    species_charges_e = np.asarray(
        tuple(
            float(library.species_records[species_name]["formal_charge_e"])
            for species_name in explicit_species_names
        )
    )
    formula_molar_masses_kg_mol = species_molar_masses_kg_mol @ stoichiometric_matrix
    formula_explicit_molecule_counts = np.sum(stoichiometric_matrix, axis=0)
    formula_atom_counts = species_atom_counts @ stoichiometric_matrix
    formula_charges_e = species_charges_e @ stoichiometric_matrix
    if np.any(np.abs(formula_charges_e) > MILP_FEASIBILITY_TOLERANCE):
        raise ValueError("recipe contains a formula unit that is not charge neutral")
    solvent_molar_volumes_m3_mol = np.zeros(len(formula_names))
    for solvent_name in active_solvent_names:
        solvent_molar_volumes_m3_mol[formula_index[solvent_name]] = float(
            library.species_records[solvent_name]["partial_molar_volume_m3_mol"]
        )
    native_constraint_rows: list[Array] = []
    native_constraint_lower_bounds: list[float] = []
    native_constraint_upper_bounds: list[float] = []
    native_constraint_labels: list[str] = []
    native_target_rows: list[Array] = []
    for solvent_name in active_solvent_names:
        target_fraction = recipe.solvents[solvent_name]
        solvent_index = formula_index[solvent_name]
        lower_fraction = max(
            0.0, float(target_fraction) - solvent_volume_fraction_tolerance
        )
        upper_fraction = float(target_fraction) + solvent_volume_fraction_tolerance
        lower_row = -lower_fraction * solvent_molar_volumes_m3_mol
        lower_row[solvent_index] += solvent_molar_volumes_m3_mol[solvent_index]
        upper_row = -upper_fraction * solvent_molar_volumes_m3_mol
        upper_row[solvent_index] += solvent_molar_volumes_m3_mol[solvent_index]
        target_row = -float(target_fraction) * solvent_molar_volumes_m3_mol
        target_row[solvent_index] += solvent_molar_volumes_m3_mol[solvent_index]
        native_constraint_rows.extend((lower_row, upper_row))
        native_constraint_lower_bounds.extend((0.0, -np.inf))
        native_constraint_upper_bounds.extend((np.inf, 0.0))
        native_constraint_labels.extend(
            (f"solvent:{solvent_name}:lower", f"solvent:{solvent_name}:upper")
        )
        native_target_rows.append(target_row)
    for salt_name in active_salt_names:
        target_molarity_mol_L = recipe.salts[salt_name]
        salt_index = formula_index[salt_name]
        lower_molarity_mol_L = max(
            0.0, float(target_molarity_mol_L) - salt_molarity_tolerance_mol_L
        )
        upper_molarity_mol_L = (
            float(target_molarity_mol_L) + salt_molarity_tolerance_mol_L
        )
        lower_row = -lower_molarity_mol_L * LITER_PER_M3 * formula_molar_masses_kg_mol
        lower_row[salt_index] += liquid_density_kg_m3
        upper_row = -upper_molarity_mol_L * LITER_PER_M3 * formula_molar_masses_kg_mol
        upper_row[salt_index] += liquid_density_kg_m3
        target_row = (
            -float(target_molarity_mol_L) * LITER_PER_M3 * formula_molar_masses_kg_mol
        )
        target_row[salt_index] += liquid_density_kg_m3
        native_constraint_rows.extend((lower_row, upper_row))
        native_constraint_lower_bounds.extend((0.0, -np.inf))
        native_constraint_upper_bounds.extend((np.inf, 0.0))
        native_constraint_labels.extend(
            (f"salt:{salt_name}:lower", f"salt:{salt_name}:upper")
        )
        native_target_rows.append(target_row)
    for additive_name in active_additive_names:
        target_weight_fraction = recipe.additives[additive_name]
        additive_index = formula_index[additive_name]
        lower_weight_fraction = max(
            0.0,
            float(target_weight_fraction) - additive_weight_fraction_tolerance,
        )
        upper_weight_fraction = (
            float(target_weight_fraction) + additive_weight_fraction_tolerance
        )
        lower_row = -lower_weight_fraction * formula_molar_masses_kg_mol
        lower_row[additive_index] += formula_molar_masses_kg_mol[additive_index]
        upper_row = -upper_weight_fraction * formula_molar_masses_kg_mol
        upper_row[additive_index] += formula_molar_masses_kg_mol[additive_index]
        target_row = -float(target_weight_fraction) * formula_molar_masses_kg_mol
        target_row[additive_index] += formula_molar_masses_kg_mol[additive_index]
        native_constraint_rows.extend((lower_row, upper_row))
        native_constraint_lower_bounds.extend((0.0, -np.inf))
        native_constraint_upper_bounds.extend((np.inf, 0.0))
        native_constraint_labels.extend(
            (
                f"additive:{additive_name}:lower",
                f"additive:{additive_name}:upper",
            )
        )
        native_target_rows.append(target_row)
    base_constraint_matrix = np.stack(
        (
            formula_explicit_molecule_counts,
            formula_atom_counts,
            formula_charges_e,
        )
    )
    base_lower_bounds = np.asarray((minimum_explicit_molecule_count, -np.inf, 0.0))
    base_upper_bounds = np.asarray(
        (maximum_explicit_molecule_count, maximum_atom_count, 0.0)
    )
    native_matrix = np.stack(native_constraint_rows)
    full_constraint_matrix = np.concatenate(
        (base_constraint_matrix, native_matrix), axis=0
    )
    full_lower_bounds = np.concatenate(
        (base_lower_bounds, np.asarray(native_constraint_lower_bounds))
    )
    full_upper_bounds = np.concatenate(
        (base_upper_bounds, np.asarray(native_constraint_upper_bounds))
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected.*",
            category=RuntimeWarning,
        )
        minimum_size_solution = milp(
            c=formula_atom_counts,
            integrality=np.ones(len(formula_names)),
            bounds=Bounds(
                np.ones(len(formula_names)),
                np.full(len(formula_names), np.inf),
            ),
            constraints=LinearConstraint(
                full_constraint_matrix,
                full_lower_bounds,
                full_upper_bounds,
            ),
            options={"mip_feasibility_tolerance": MILP_FEASIBILITY_TOLERANCE},
        )
    if not minimum_size_solution.success or minimum_size_solution.x is None:
        native_row_scales = np.maximum(
            np.max(np.abs(native_matrix), axis=1),
            np.finfo(float).tiny,
        )
        normalized_native_matrix = native_matrix / native_row_scales[:, None]
        native_constraint_count = normalized_native_matrix.shape[0]
        relaxed_constraint_matrix = np.pad(
            base_constraint_matrix,
            ((0, 0), (0, native_constraint_count)),
        )
        relaxed_lower_bounds = base_lower_bounds.copy()
        relaxed_upper_bounds = base_upper_bounds.copy()
        relaxed_native_rows: list[Array] = []
        relaxed_native_lower_bounds: list[float] = []
        relaxed_native_upper_bounds: list[float] = []
        for constraint_index, constraint_row in enumerate(normalized_native_matrix):
            relaxed_row = np.pad(
                constraint_row,
                (0, native_constraint_count),
            )
            if math.isfinite(native_constraint_lower_bounds[constraint_index]):
                relaxed_row[len(formula_names) + constraint_index] = 1.0
                relaxed_native_lower_bounds.append(
                    native_constraint_lower_bounds[constraint_index]
                    / native_row_scales[constraint_index]
                )
                relaxed_native_upper_bounds.append(np.inf)
            else:
                relaxed_row[len(formula_names) + constraint_index] = -1.0
                relaxed_native_lower_bounds.append(-np.inf)
                relaxed_native_upper_bounds.append(
                    native_constraint_upper_bounds[constraint_index]
                    / native_row_scales[constraint_index]
                )
            relaxed_native_rows.append(relaxed_row)
        relaxed_constraint_matrix = np.concatenate(
            (relaxed_constraint_matrix, np.stack(relaxed_native_rows)),
            axis=0,
        )
        relaxed_lower_bounds = np.concatenate(
            (relaxed_lower_bounds, np.asarray(relaxed_native_lower_bounds))
        )
        relaxed_upper_bounds = np.concatenate(
            (relaxed_upper_bounds, np.asarray(relaxed_native_upper_bounds))
        )
        relaxed_objective = np.concatenate(
            (
                formula_atom_counts / max(float(np.max(formula_atom_counts)), 1.0),
                np.ones(native_constraint_count),
            )
        )
        relaxed_integrality = np.concatenate(
            (np.ones(len(formula_names)), np.zeros(native_constraint_count))
        )
        relaxed_variable_lower_bounds = np.concatenate(
            (np.ones(len(formula_names)), np.zeros(native_constraint_count))
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected.*",
                category=RuntimeWarning,
            )
            closest_solution = milp(
                c=relaxed_objective,
                integrality=relaxed_integrality,
                bounds=Bounds(
                    relaxed_variable_lower_bounds,
                    np.full(len(relaxed_objective), np.inf),
                ),
                constraints=LinearConstraint(
                    relaxed_constraint_matrix,
                    relaxed_lower_bounds,
                    relaxed_upper_bounds,
                ),
                options={"mip_feasibility_tolerance": MILP_FEASIBILITY_TOLERANCE},
            )
        closest_detail = "no integer candidate exists within the size caps"
        if closest_solution.success and closest_solution.x is not None:
            closest_formula_counts = np.rint(
                closest_solution.x[: len(formula_names)]
            ).astype(int)
            closest_formula_mass_kg_mol = float(
                formula_molar_masses_kg_mol @ closest_formula_counts
            )
            closest_solvent_volume_m3_mol = float(
                solvent_molar_volumes_m3_mol @ closest_formula_counts
            )
            closest_solvents = tuple(
                (
                    solvent_name,
                    float(
                        closest_formula_counts[formula_index[solvent_name]]
                        * solvent_molar_volumes_m3_mol[formula_index[solvent_name]]
                        / closest_solvent_volume_m3_mol
                    ),
                )
                for solvent_name in active_solvent_names
            )
            closest_salts = tuple(
                (
                    salt_name,
                    float(
                        liquid_density_kg_m3
                        * closest_formula_counts[formula_index[salt_name]]
                        / (LITER_PER_M3 * closest_formula_mass_kg_mol)
                    ),
                )
                for salt_name in active_salt_names
            )
            closest_additives = tuple(
                (
                    additive_name,
                    float(
                        closest_formula_counts[formula_index[additive_name]]
                        * formula_molar_masses_kg_mol[formula_index[additive_name]]
                        / closest_formula_mass_kg_mol
                    ),
                )
                for additive_name in active_additive_names
            )
            closest_detail = (
                f"closest_formula_counts={tuple(zip(formula_names, closest_formula_counts, strict=True))}, "
                f"closest_solvent_vv={closest_solvents}, "
                f"closest_salt_mol_L={closest_salts}, "
                f"closest_additive_weight_fraction={closest_additives}"
            )
        raise ValueError(
            "no integer formula-unit realization satisfies the configured native-unit "
            f"bounds below molecule_cap={maximum_explicit_molecule_count} and "
            f"atom_cap={maximum_atom_count}; constraints={tuple(native_constraint_labels)}; "
            f"{closest_detail}"
        )
    minimum_atom_count = int(
        round(float(formula_atom_counts @ minimum_size_solution.x))
    )
    target_matrix = np.stack(native_target_rows)
    target_scales = np.maximum(
        np.max(np.abs(target_matrix), axis=1),
        np.finfo(float).tiny,
    )
    normalized_target_matrix = target_matrix / target_scales[:, None]
    formula_count = len(formula_names)
    deviation_count = normalized_target_matrix.shape[0]
    tie_constraint_matrix = np.pad(
        full_constraint_matrix,
        ((0, 0), (0, deviation_count)),
    )
    atom_equality_row = np.pad(formula_atom_counts, (0, deviation_count))
    positive_deviation_rows = np.concatenate(
        (normalized_target_matrix, -np.eye(deviation_count)), axis=1
    )
    negative_deviation_rows = np.concatenate(
        (-normalized_target_matrix, -np.eye(deviation_count)), axis=1
    )
    tie_constraint_matrix = np.concatenate(
        (
            tie_constraint_matrix,
            atom_equality_row[None, :],
            positive_deviation_rows,
            negative_deviation_rows,
        ),
        axis=0,
    )
    tie_lower_bounds = np.concatenate(
        (
            full_lower_bounds,
            np.asarray((minimum_atom_count,)),
            np.full(2 * deviation_count, -np.inf),
        )
    )
    tie_upper_bounds = np.concatenate(
        (
            full_upper_bounds,
            np.asarray((minimum_atom_count,)),
            np.zeros(2 * deviation_count),
        )
    )
    deterministic_tie_coefficients = (
        np.arange(1, formula_count + 1) * np.finfo(float).eps
    )
    tie_objective = np.concatenate(
        (deterministic_tie_coefficients, np.ones(deviation_count))
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected.*",
            category=RuntimeWarning,
        )
        tie_solution = milp(
            c=tie_objective,
            integrality=np.concatenate(
                (np.ones(formula_count), np.zeros(deviation_count))
            ),
            bounds=Bounds(
                np.concatenate((np.ones(formula_count), np.zeros(deviation_count))),
                np.full(formula_count + deviation_count, np.inf),
            ),
            constraints=LinearConstraint(
                tie_constraint_matrix,
                tie_lower_bounds,
                tie_upper_bounds,
            ),
            options={"mip_feasibility_tolerance": MILP_FEASIBILITY_TOLERANCE},
        )
    if not tie_solution.success or tie_solution.x is None:
        raise ValueError("native-unit realization tie-break failed")
    formula_counts = np.rint(tie_solution.x[:formula_count]).astype(int)
    explicit_species_counts = np.rint(stoichiometric_matrix @ formula_counts).astype(
        int
    )
    total_formula_mass_kg_mol = float(formula_molar_masses_kg_mol @ formula_counts)
    density_conditioned_volume_m3 = total_formula_mass_kg_mol / (
        N_A * liquid_density_kg_m3
    )
    total_solvent_volume_m3_mol = float(solvent_molar_volumes_m3_mol @ formula_counts)
    realized_solvent_volume_fractions = tuple(
        (
            solvent_name,
            float(
                formula_counts[formula_index[solvent_name]]
                * solvent_molar_volumes_m3_mol[formula_index[solvent_name]]
                / total_solvent_volume_m3_mol
            ),
        )
        for solvent_name in active_solvent_names
    )
    realized_salt_molarities_mol_L = tuple(
        (
            salt_name,
            float(
                liquid_density_kg_m3
                * formula_counts[formula_index[salt_name]]
                / (LITER_PER_M3 * total_formula_mass_kg_mol)
            ),
        )
        for salt_name in active_salt_names
    )
    realized_additive_weight_fractions = tuple(
        (
            additive_name,
            float(
                formula_counts[formula_index[additive_name]]
                * formula_molar_masses_kg_mol[formula_index[additive_name]]
                / total_formula_mass_kg_mol
            ),
        )
        for additive_name in active_additive_names
    )
    native_unit_deviations = (
        tuple(
            (f"solvent:{name}", value - float(recipe.solvents[name]))
            for name, value in realized_solvent_volume_fractions
        )
        + tuple(
            (f"salt:{name}", value - float(recipe.salts[name]))
            for name, value in realized_salt_molarities_mol_L
        )
        + tuple(
            (f"additive:{name}", value - float(recipe.additives[name]))
            for name, value in realized_additive_weight_fractions
        )
    )
    if any(
        abs(value - float(recipe.solvents[name]))
        > solvent_volume_fraction_tolerance + MILP_FEASIBILITY_TOLERANCE
        for name, value in realized_solvent_volume_fractions
    ):
        raise RuntimeError("integer solver returned an invalid solvent realization")
    if any(
        abs(value - float(recipe.salts[name]))
        > salt_molarity_tolerance_mol_L + MILP_FEASIBILITY_TOLERANCE
        for name, value in realized_salt_molarities_mol_L
    ):
        raise RuntimeError("integer solver returned an invalid salt realization")
    if any(
        abs(value - float(recipe.additives[name]))
        > additive_weight_fraction_tolerance + MILP_FEASIBILITY_TOLERANCE
        for name, value in realized_additive_weight_fractions
    ):
        raise RuntimeError("integer solver returned an invalid additive realization")
    return IntegerRecipeRealization(
        formula_unit_counts=tuple(
            (formula_name, int(formula_counts[formula_index[formula_name]]))
            for formula_name in formula_names
        ),
        explicit_species_counts=tuple(
            (
                species_name,
                int(explicit_species_counts[species_index[species_name]]),
            )
            for species_name in explicit_species_names
        ),
        explicit_molecule_count=int(np.sum(explicit_species_counts)),
        atom_count=int(species_atom_counts @ explicit_species_counts),
        cell_mass_kg=total_formula_mass_kg_mol / N_A,
        density_conditioned_volume_m3=density_conditioned_volume_m3,
        realized_solvent_volume_fractions=realized_solvent_volume_fractions,
        realized_salt_molarities_mol_L=realized_salt_molarities_mol_L,
        realized_additive_weight_fractions=realized_additive_weight_fractions,
        native_unit_deviations=native_unit_deviations,
    )


def build_periodic_molecular_system(
    explicit_species_counts: tuple[tuple[str, int], ...],
    box_volume_m3: float,
    minimum_interatomic_contact_ratio: float,
    initial_placement_attempts_per_molecule: int,
    random_seed: int,
) -> MolecularSystem:
    library = _physical_library_records()
    count_by_species = dict(explicit_species_counts)
    species_extent_names: list[tuple[float, str]] = []
    for species_name in count_by_species:
        species_record = library.species_records[species_name]
        reference_coordinates_m = np.asarray(
            species_record["reference_conformer_coordinates_m"], dtype=float
        )
        reference_center_m = np.mean(reference_coordinates_m, axis=0)
        maximum_site_extent_m = max(
            float(np.linalg.norm(site_position_m - reference_center_m))
            + 0.5 * float(site_record["lj_sigma_m"])
            for site_position_m, site_record in zip(
                reference_coordinates_m, species_record["sites"], strict=True
            )
        )
        species_extent_names.append((-maximum_site_extent_m, species_name))
    species_names = tuple(
        species_name
        for _negative_extent_m, species_name in sorted(species_extent_names)
    )
    species_records = tuple(library.species_records[name] for name in species_names)
    counts = np.asarray(tuple(count_by_species[name] for name in species_names))
    if box_volume_m3 <= 0.0:
        raise ValueError("packing box volume must be positive")
    box_length_m = box_volume_m3 ** (1.0 / CARTESIAN_DIMENSION)
    box_vectors_m = np.eye(3) * box_length_m
    random_generator = np.random.default_rng(random_seed)
    positions: list[Array] = []
    masses: list[float] = []
    charges: list[float] = []
    lj_sigma: list[float] = []
    lj_epsilon: list[float] = []
    polarizabilities: list[float] = []
    molecule_indices: list[int] = []
    molecule_atom_indices: list[Array] = []
    molecule_species_names: list[str] = []
    bonds: list[tuple[int, int]] = []
    bond_constants: list[float] = []
    bond_lengths: list[float] = []
    angles: list[tuple[int, int, int]] = []
    angle_constants: list[float] = []
    angle_values: list[float] = []
    torsions: list[tuple[int, int, int, int, tuple[tuple[float, int, float], ...]]] = []
    atom_offset = 0
    molecule_index = 0
    for count, species_name, record in zip(
        counts, species_names, species_records, strict=True
    ):
        reference = np.asarray(record["reference_conformer_coordinates_m"], dtype=float)
        reference -= np.mean(reference, axis=0)
        for _ in range(int(count)):
            site_sigma_m = np.asarray(
                tuple(float(site["lj_sigma_m"]) for site in record["sites"])
            )
            molecule_positions = np.empty_like(reference)
            placement_succeeded = False
            for _placement_attempt in range(initial_placement_attempts_per_molecule):
                rotation = _random_rotation(random_generator)
                center = random_generator.random(CARTESIAN_DIMENSION) @ box_vectors_m
                candidate_positions = reference @ rotation.T + center
                if positions:
                    existing_positions = np.asarray(positions)
                    displacements = minimum_image_displacement(
                        candidate_positions[:, None, :]
                        - existing_positions[None, :, :],
                        box_vectors_m,
                    )
                    distances = np.linalg.norm(displacements, axis=2)
                    contact_distances = 0.5 * (
                        site_sigma_m[:, None] + np.asarray(lj_sigma)[None, :]
                    )
                    if np.any(
                        distances
                        < minimum_interatomic_contact_ratio * contact_distances
                    ):
                        continue
                molecule_positions = candidate_positions
                placement_succeeded = True
                break
            if not placement_succeeded:
                raise ValueError(
                    "could not place all molecules at the requested density and "
                    "minimum interatomic contact ratio"
                )
            positions.extend(molecule_positions)
            atom_indices = np.arange(atom_offset, atom_offset + len(record["sites"]))
            molecule_atom_indices.append(atom_indices)
            molecule_species_names.append(species_name)
            for site in record["sites"]:
                masses.append(float(site["mass_kg"]))
                charges.append(float(site["charge_number"]) * E_CHARGE)
                lj_sigma.append(float(site["lj_sigma_m"]))
                lj_epsilon.append(float(site["lj_epsilon_J"]))
                polarizabilities.append(float(site["polarizability_SI"]))
                molecule_indices.append(molecule_index)
            for bond in record["bonds"]:
                bonds.append(
                    (
                        atom_offset + int(bond["site_i"]),
                        atom_offset + int(bond["site_j"]),
                    )
                )
                bond_constants.append(float(bond["k_J_m2_mol"]) / N_A)
                bond_lengths.append(float(bond["r0_m"]))
            for angle in record["angles"]:
                angles.append(
                    (
                        atom_offset + int(angle["site_i"]),
                        atom_offset + int(angle["site_j"]),
                        atom_offset + int(angle["site_k"]),
                    )
                )
                angle_constants.append(float(angle["k_J_rad2_mol"]) / N_A)
                angle_values.append(float(angle["theta0_rad"]))
            for torsion in record["torsions"]:
                terms = tuple(
                    (
                        float(term["Vn_J_mol"]) / (N_A * float(term["idivf"])),
                        int(term["periodicity"]),
                        float(term["phase_rad"]),
                    )
                    for term in torsion["terms"]
                )
                torsions.append(
                    (
                        atom_offset + int(torsion["site_i"]),
                        atom_offset + int(torsion["site_j"]),
                        atom_offset + int(torsion["site_k"]),
                        atom_offset + int(torsion["site_l"]),
                        terms,
                    )
                )
            atom_offset += len(record["sites"])
            molecule_index += 1
    atom_count = atom_offset
    nonbonded_mask = np.ones((atom_count, atom_count), dtype=bool)
    np.fill_diagonal(nonbonded_mask, False)
    for atom_indices in molecule_atom_indices:
        nonbonded_mask[np.ix_(atom_indices, atom_indices)] = False
    if abs(sum(charges)) > E_CHARGE * 1.0e-8:
        raise ValueError("constructed periodic system is not charge neutral")
    return MolecularSystem(
        positions_m=np.asarray(positions),
        box_vectors_m=box_vectors_m,
        masses_kg=np.asarray(masses),
        charges_C=np.asarray(charges),
        lj_sigma_m=np.asarray(lj_sigma),
        lj_epsilon_J=np.asarray(lj_epsilon),
        polarizabilities_SI=np.asarray(polarizabilities),
        molecule_index=np.asarray(molecule_indices),
        molecule_atom_indices=tuple(molecule_atom_indices),
        molecule_species_names=tuple(molecule_species_names),
        bonds=np.asarray(bonds, dtype=int).reshape((-1, 2)),
        bond_force_constants_J_m2=np.asarray(bond_constants),
        bond_lengths_m=np.asarray(bond_lengths),
        angles=np.asarray(angles, dtype=int).reshape((-1, 3)),
        angle_force_constants_J_rad2=np.asarray(angle_constants),
        angle_values_rad=np.asarray(angle_values),
        torsions=tuple(torsions),
        nonbonded_mask=nonbonded_mask,
    )


class AnalyticalPeriodicInteratomicModel:
    """Explicit bonded, LJ, Ewald, and induced-dipole Hamiltonian."""

    def __init__(self, system: MolecularSystem, numerics: NumericalSettings) -> None:
        if not 0.0 < numerics.ewald_reciprocal_relative_tolerance < 1.0:
            raise ValueError("Ewald reciprocal relative tolerance must lie in (0, 1)")
        if numerics.ewald_maximum_box_length_m <= 0.0:
            raise ValueError("Ewald maximum box length must be positive")
        if not (
            0.0
            < numerics.lennard_jones_switch_start_m
            < numerics.lennard_jones_cutoff_m
        ):
            raise ValueError(
                "Lennard-Jones switch start must be positive and below the cutoff"
            )
        if numerics.dispersion_tail_quadrature_order <= 0:
            raise ValueError("dispersion-tail quadrature order must be positive")
        self.system = system
        self.numerics = numerics
        self._has_bonds = bool(system.bonds.size)
        self._has_angles = bool(system.angles.size)
        self._lj_pair_scales, self._coulomb_pair_scales = (
            self._special_bond_pair_scales()
        )
        self._lj_pair_i, self._lj_pair_j = np.where(
            np.triu(self._lj_pair_scales > 0.0, 1)
        )
        self._lj_pair_i_tensor = torch.as_tensor(self._lj_pair_i)
        self._lj_pair_j_tensor = torch.as_tensor(self._lj_pair_j)
        self._lj_pair_scale_tensor = torch.as_tensor(
            self._lj_pair_scales[self._lj_pair_i, self._lj_pair_j]
        )
        molecule_charges_C = np.asarray(
            [
                np.sum(system.charges_C[molecule_atom_indices])
                for molecule_atom_indices in system.molecule_atom_indices
            ]
        )
        self._ionic_atom_mask = (
            np.abs(molecule_charges_C[system.molecule_index])
            > E_CHARGE * MILP_FEASIBILITY_TOLERANCE
        )
        self._electrostatic_pair_i, self._electrostatic_pair_j = np.triu_indices(
            system.positions_m.shape[0], 1
        )
        self._electrostatic_pair_i_tensor = torch.as_tensor(self._electrostatic_pair_i)
        self._electrostatic_pair_j_tensor = torch.as_tensor(self._electrostatic_pair_j)
        reciprocal_wavevector_cutoff_per_m = (
            2.0
            * numerics.ewald_splitting_per_m
            * math.sqrt(-math.log(numerics.ewald_reciprocal_relative_tolerance))
        )
        reciprocal_shell = math.ceil(
            reciprocal_wavevector_cutoff_per_m
            * numerics.ewald_maximum_box_length_m
            / (2.0 * math.pi)
        )
        self.reciprocal_wavevector_cutoff_per_m = reciprocal_wavevector_cutoff_per_m
        self.reciprocal_shell = reciprocal_shell
        electrostatics_are_active = bool(
            np.any(system.charges_C != 0.0) or np.any(system.polarizabilities_SI != 0.0)
        )
        if electrostatics_are_active:
            self._reciprocal_indices = np.asarray(
                [
                    (first_index, second_index, third_index)
                    for first_index in range(-reciprocal_shell, reciprocal_shell + 1)
                    for second_index in range(-reciprocal_shell, reciprocal_shell + 1)
                    for third_index in range(-reciprocal_shell, reciprocal_shell + 1)
                    if (first_index, second_index, third_index) != (0, 0, 0)
                ],
                dtype=float,
            )
        else:
            self._reciprocal_indices = np.empty((0, CARTESIAN_DIMENSION), dtype=float)
        self._bond_indices_tensor = torch.as_tensor(system.bonds)
        self._bond_force_constants_tensor = torch.as_tensor(
            system.bond_force_constants_J_m2
        )
        self._bond_lengths_tensor = torch.as_tensor(system.bond_lengths_m)
        self._angle_indices_tensor = torch.as_tensor(system.angles)
        self._angle_force_constants_tensor = torch.as_tensor(
            system.angle_force_constants_J_rad2
        )
        self._angle_values_tensor = torch.as_tensor(system.angle_values_rad)
        self._charges_tensor = torch.as_tensor(system.charges_C)
        self._ionic_atom_mask_tensor = torch.as_tensor(self._ionic_atom_mask)
        self._neutral_atom_mask_tensor = ~self._ionic_atom_mask_tensor
        self._electrostatic_pair_ionic_count_tensor = self._ionic_atom_mask_tensor[
            self._electrostatic_pair_i_tensor
        ].to(torch.int64) + self._ionic_atom_mask_tensor[
            self._electrostatic_pair_j_tensor
        ].to(torch.int64)
        self._lj_sigma_pair_tensor = 0.5 * (
            torch.as_tensor(system.lj_sigma_m[self._lj_pair_i])
            + torch.as_tensor(system.lj_sigma_m[self._lj_pair_j])
        )
        self._lj_epsilon_pair_tensor = torch.sqrt(
            torch.as_tensor(system.lj_epsilon_J[self._lj_pair_i])
            * torch.as_tensor(system.lj_epsilon_J[self._lj_pair_j])
        )
        self._lj_pair_ionic_count_tensor = self._ionic_atom_mask_tensor[
            self._lj_pair_i_tensor
        ].to(torch.int64) + self._ionic_atom_mask_tensor[self._lj_pair_j_tensor].to(
            torch.int64
        )
        quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
            numerics.dispersion_tail_quadrature_order
        )
        switch_width_m = (
            numerics.lennard_jones_cutoff_m - numerics.lennard_jones_switch_start_m
        )
        quadrature_distances_m = (
            0.5 * switch_width_m * (quadrature_nodes + 1.0)
            + numerics.lennard_jones_switch_start_m
        )
        normalized_switch_distance = (
            quadrature_distances_m - numerics.lennard_jones_switch_start_m
        ) / switch_width_m
        retained_switch_weight = (
            1.0
            - QUINTIC_SWITCH_CUBIC_COEFFICIENT
            * normalized_switch_distance**QUINTIC_SWITCH_CUBIC_POWER
            + QUINTIC_SWITCH_QUARTIC_COEFFICIENT
            * normalized_switch_distance**QUINTIC_SWITCH_QUARTIC_POWER
            - QUINTIC_SWITCH_QUINTIC_COEFFICIENT
            * normalized_switch_distance**QUINTIC_SWITCH_QUINTIC_POWER
        )
        omitted_switch_weight = 1.0 - retained_switch_weight
        pair_sigma_m = 0.5 * (
            system.lj_sigma_m[self._lj_pair_i] + system.lj_sigma_m[self._lj_pair_j]
        )
        pair_epsilon_J = np.sqrt(
            system.lj_epsilon_J[self._lj_pair_i] * system.lj_epsilon_J[self._lj_pair_j]
        )
        pair_scale = self._lj_pair_scales[self._lj_pair_i, self._lj_pair_j]
        radial_quadrature_weights_m = 0.5 * switch_width_m * quadrature_weights
        repulsive_switch_integrals_J_m3 = (
            4.0
            * math.pi
            * np.sum(
                radial_quadrature_weights_m[None, :]
                * quadrature_distances_m[None, :] ** 2
                * (
                    4.0
                    * pair_epsilon_J[:, None]
                    * (pair_sigma_m[:, None] / quadrature_distances_m[None, :])
                    ** (LJ_ATTRACTIVE_EXPONENT * LJ_REPULSIVE_EXPONENT_MULTIPLIER)
                )
                * omitted_switch_weight[None, :],
                axis=1,
            )
        )
        attractive_switch_integrals_J_m3 = (
            4.0
            * math.pi
            * np.sum(
                radial_quadrature_weights_m[None, :]
                * quadrature_distances_m[None, :] ** 2
                * (
                    -4.0
                    * pair_epsilon_J[:, None]
                    * (pair_sigma_m[:, None] / quadrature_distances_m[None, :])
                    ** LJ_ATTRACTIVE_EXPONENT
                )
                * omitted_switch_weight[None, :],
                axis=1,
            )
        )
        cutoff_m = numerics.lennard_jones_cutoff_m
        repulsive_beyond_cutoff_integrals_J_m3 = (
            LJ_RADIAL_ENERGY_INTEGRAL_PREFACTOR
            * pair_epsilon_J
            * pair_sigma_m
            ** (LJ_ATTRACTIVE_EXPONENT * LJ_REPULSIVE_EXPONENT_MULTIPLIER)
            / (
                (
                    LJ_ATTRACTIVE_EXPONENT * LJ_REPULSIVE_EXPONENT_MULTIPLIER
                    - CARTESIAN_DIMENSION
                )
                * cutoff_m
                ** (
                    LJ_ATTRACTIVE_EXPONENT * LJ_REPULSIVE_EXPONENT_MULTIPLIER
                    - CARTESIAN_DIMENSION
                )
            )
        )
        attractive_beyond_cutoff_integrals_J_m3 = (
            -LJ_RADIAL_ENERGY_INTEGRAL_PREFACTOR
            * pair_epsilon_J
            * pair_sigma_m**LJ_ATTRACTIVE_EXPONENT
            / (
                (LJ_ATTRACTIVE_EXPONENT - CARTESIAN_DIMENSION)
                * cutoff_m ** (LJ_ATTRACTIVE_EXPONENT - CARTESIAN_DIMENSION)
            )
        )
        repulsive_tail_coefficients_J_m3 = pair_scale * (
            repulsive_switch_integrals_J_m3 + repulsive_beyond_cutoff_integrals_J_m3
        )
        attractive_tail_coefficients_J_m3 = pair_scale * (
            attractive_switch_integrals_J_m3 + attractive_beyond_cutoff_integrals_J_m3
        )
        self._lj_repulsive_tail_coefficient_tensor = torch.as_tensor(
            np.sum(repulsive_tail_coefficients_J_m3)
        )
        self._lj_attractive_tail_coefficients_by_ionic_count_tensor = torch.as_tensor(
            np.asarray(
                [
                    np.sum(
                        attractive_tail_coefficients_J_m3[
                            (
                                self._ionic_atom_mask[self._lj_pair_i].astype(int)
                                + self._ionic_atom_mask[self._lj_pair_j].astype(int)
                            )
                            == ionic_count
                        ]
                    )
                    for ionic_count in range(REST_INTERACTION_CLASS_COUNT)
                ]
            )
        )
        self._reciprocal_indices_tensor = torch.as_tensor(self._reciprocal_indices)
        self._coulomb_pair_scales_tensor = torch.as_tensor(self._coulomb_pair_scales)
        electrostatic_pair_scales = self._coulomb_pair_scales[
            self._electrostatic_pair_i, self._electrostatic_pair_j
        ]
        excluded_pair_indices = np.flatnonzero(electrostatic_pair_scales < 1.0)
        self._excluded_pair_indices_tensor = torch.as_tensor(excluded_pair_indices)
        self._excluded_pair_i_tensor = torch.as_tensor(
            self._electrostatic_pair_i[excluded_pair_indices]
        )
        self._excluded_pair_j_tensor = torch.as_tensor(
            self._electrostatic_pair_j[excluded_pair_indices]
        )
        self._excluded_pair_weights_tensor = torch.as_tensor(
            1.0 - electrostatic_pair_scales[excluded_pair_indices]
        )
        self._excluded_pair_count = int(excluded_pair_indices.size)
        reciprocal_exclusion_weights = 1.0 - self._coulomb_pair_scales
        np.fill_diagonal(reciprocal_exclusion_weights, 0.0)
        self._reciprocal_exclusion_weights_tensor = torch.as_tensor(
            reciprocal_exclusion_weights
        )
        self._has_reciprocal_exclusions = bool(
            np.any(reciprocal_exclusion_weights > 0.0)
        )
        self._polarizabilities_tensor = torch.as_tensor(system.polarizabilities_SI)
        active_polarizable_indices = np.flatnonzero(system.polarizabilities_SI > 0.0)
        self._active_polarizable_indices = active_polarizable_indices
        self._active_polarizable_indices_tensor = torch.as_tensor(
            active_polarizable_indices
        )
        self._polarizable_atom_count = int(active_polarizable_indices.size)
        active_exclusion_weights = reciprocal_exclusion_weights[
            np.ix_(active_polarizable_indices, active_polarizable_indices)
        ]
        self._active_exclusion_weights_tensor = torch.as_tensor(
            active_exclusion_weights
        )
        self._has_active_exclusions = bool(np.any(active_exclusion_weights > 0.0))
        active_exclusion_pair_i, active_exclusion_pair_j = np.where(
            np.triu(active_exclusion_weights > 0.0, 1)
        )
        self._active_exclusion_pair_i_tensor = torch.as_tensor(active_exclusion_pair_i)
        self._active_exclusion_pair_j_tensor = torch.as_tensor(active_exclusion_pair_j)
        self._active_exclusion_pair_weights_tensor = torch.as_tensor(
            active_exclusion_weights[active_exclusion_pair_i, active_exclusion_pair_j]
        )
        self._active_pair_scales_tensor = torch.as_tensor(
            self._coulomb_pair_scales[
                np.ix_(active_polarizable_indices, active_polarizable_indices)
            ]
        )
        self._cartesian_identity_tensor = torch.eye(
            CARTESIAN_DIMENSION, dtype=TORCH_DTYPE
        )
        torsion_indices: list[tuple[int, int, int, int]] = []
        torsion_amplitudes_J: list[float] = []
        torsion_periodicities: list[int] = []
        torsion_phases_rad: list[float] = []
        for (
            first_index,
            second_index,
            third_index,
            fourth_index,
            terms,
        ) in system.torsions:
            for amplitude_J, periodicity, phase_rad in terms:
                torsion_indices.append(
                    (first_index, second_index, third_index, fourth_index)
                )
                torsion_amplitudes_J.append(amplitude_J)
                torsion_periodicities.append(periodicity)
                torsion_phases_rad.append(phase_rad)
        self._torsion_indices_tensor = torch.as_tensor(
            np.asarray(torsion_indices, dtype=int).reshape((-1, 4))
        )
        self._torsion_amplitudes_tensor = torch.as_tensor(torsion_amplitudes_J)
        self._torsion_periodicities_tensor = torch.as_tensor(torsion_periodicities)
        self._torsion_phases_tensor = torch.as_tensor(torsion_phases_rad)
        self._compiled_energy_components_batch_tensor = torch.compile(
            self._energy_components_batch_tensor,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        self._compiled_physical_energy_terms_batch_tensor = torch.compile(
            self._physical_energy_terms_batch_tensor,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        self._compiled_energy_force_components_batch_tensor = torch.compile(
            self._energy_force_components_batch_tensor,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        self._compiled_analytical_nonpolar_forces_batch_tensor = torch.compile(
            self._analytical_nonpolar_forces_batch_tensor,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        self._compiled_physical_energy_force_batch_tensor = torch.compile(
            self._physical_energy_force_batch_tensor,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )

    def _special_bond_pair_scales(self) -> tuple[Array, Array]:
        atom_count = self.system.positions_m.shape[0]
        graph_distances = np.full((atom_count, atom_count), atom_count, dtype=int)
        np.fill_diagonal(graph_distances, 0)
        for first_atom, second_atom in self.system.bonds:
            graph_distances[first_atom, second_atom] = 1
            graph_distances[second_atom, first_atom] = 1
        for intermediate_atom in range(atom_count):
            graph_distances = np.minimum(
                graph_distances,
                graph_distances[:, intermediate_atom, None]
                + graph_distances[None, intermediate_atom, :],
            )
        same_molecule = (
            self.system.molecule_index[:, None] == self.system.molecule_index[None, :]
        )
        pair_scales = np.ones((atom_count, atom_count), dtype=float)
        pair_scales[same_molecule & (graph_distances <= 2)] = 0.0
        pair_scales[same_molecule & (graph_distances == 3)] = 0.5
        np.fill_diagonal(pair_scales, 0.0)
        return pair_scales, pair_scales.copy()

    def _energy_tensor(
        self,
        positions_m: torch.Tensor,
        box_vectors_m: torch.Tensor,
        lambda_value: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if isinstance(lambda_value, float) and not 0.0 <= lambda_value <= 1.0:
            raise ValueError("ionic interaction lambda must be in [0, 1]")
        lambda_tensor = torch.as_tensor(lambda_value, dtype=TORCH_DTYPE)
        (
            fixed_energy,
            ion_ion_energy,
            ion_neutral_energy,
            _residual,
        ) = self._energy_components_tensor(positions_m, box_vectors_m)
        return (
            fixed_energy
            + lambda_tensor * ion_ion_energy
            + torch.sqrt(lambda_tensor) * ion_neutral_energy
        )

    def _nonpolar_energy_components_tensor(
        self,
        positions_m: torch.Tensor,
        box_vectors_m: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fixed_energy = torch.zeros((), dtype=TORCH_DTYPE)
        ion_ion_energy = torch.zeros((), dtype=TORCH_DTYPE)
        ion_neutral_energy = torch.zeros((), dtype=TORCH_DTYPE)
        bonded_energy = torch.zeros((), dtype=TORCH_DTYPE)
        if self._has_bonds:
            bond_indices = self._bond_indices_tensor
            displacement = _torch_minimum_image(
                positions_m[bond_indices[:, 0]] - positions_m[bond_indices[:, 1]],
                box_vectors_m,
            )
            lengths = torch.linalg.norm(displacement, dim=1)
            bonded_energy = bonded_energy + torch.sum(
                0.5
                * self._bond_force_constants_tensor
                * (lengths - self._bond_lengths_tensor) ** 2
            )
        if self._has_angles:
            angle_indices = self._angle_indices_tensor
            first = _torch_minimum_image(
                positions_m[angle_indices[:, 0]] - positions_m[angle_indices[:, 1]],
                box_vectors_m,
            )
            second = _torch_minimum_image(
                positions_m[angle_indices[:, 2]] - positions_m[angle_indices[:, 1]],
                box_vectors_m,
            )
            cross_norm = torch.sqrt(
                torch.sum(torch.linalg.cross(first, second, dim=1) ** 2, dim=1)
                + torch.finfo(TORCH_DTYPE).tiny
            )
            values = torch.atan2(cross_norm, torch.sum(first * second, dim=1))
            bonded_energy = bonded_energy + torch.sum(
                0.5
                * self._angle_force_constants_tensor
                * (values - self._angle_values_tensor) ** 2
            )
        if self._torsion_indices_tensor.shape[0] > 0:
            torsion_indices = self._torsion_indices_tensor
            first = _torch_minimum_image(
                positions_m[torsion_indices[:, 1]] - positions_m[torsion_indices[:, 0]],
                box_vectors_m,
            )
            second = _torch_minimum_image(
                positions_m[torsion_indices[:, 2]] - positions_m[torsion_indices[:, 1]],
                box_vectors_m,
            )
            third = _torch_minimum_image(
                positions_m[torsion_indices[:, 3]] - positions_m[torsion_indices[:, 2]],
                box_vectors_m,
            )
            first_normal = torch.linalg.cross(first, second, dim=1)
            second_normal = torch.linalg.cross(second, third, dim=1)
            dihedral = torch.atan2(
                torch.linalg.norm(second, dim=1)
                * torch.sum(first * second_normal, dim=1),
                torch.sum(first_normal * second_normal, dim=1)
                + torch.finfo(TORCH_DTYPE).tiny,
            )
            bonded_energy = bonded_energy + torch.sum(
                self._torsion_amplitudes_tensor
                * (
                    1.0
                    + torch.cos(
                        self._torsion_periodicities_tensor * dihedral
                        - self._torsion_phases_tensor
                    )
                )
            )
        fixed_energy = fixed_energy + bonded_energy
        displacement = _torch_minimum_image(
            positions_m[:, None, :] - positions_m[None, :, :], box_vectors_m
        )
        distance = torch.linalg.norm(displacement, dim=2)
        lj_pair_distance = distance[self._lj_pair_i_tensor, self._lj_pair_j_tensor]
        attractive_term = (
            self._lj_sigma_pair_tensor / lj_pair_distance
        ) ** LJ_ATTRACTIVE_EXPONENT
        normalized_switch_distance = (
            lj_pair_distance - self.numerics.lennard_jones_switch_start_m
        ) / (
            self.numerics.lennard_jones_cutoff_m
            - self.numerics.lennard_jones_switch_start_m
        )
        transition_switch_weight = (
            1.0
            - QUINTIC_SWITCH_CUBIC_COEFFICIENT
            * normalized_switch_distance**QUINTIC_SWITCH_CUBIC_POWER
            + QUINTIC_SWITCH_QUARTIC_COEFFICIENT
            * normalized_switch_distance**QUINTIC_SWITCH_QUARTIC_POWER
            - QUINTIC_SWITCH_QUINTIC_COEFFICIENT
            * normalized_switch_distance**QUINTIC_SWITCH_QUINTIC_POWER
        )
        switch_weight = torch.where(
            lj_pair_distance <= self.numerics.lennard_jones_switch_start_m,
            torch.ones_like(lj_pair_distance),
            torch.where(
                lj_pair_distance < self.numerics.lennard_jones_cutoff_m,
                transition_switch_weight,
                torch.zeros_like(lj_pair_distance),
            ),
        )
        repulsive_pair_energy = (
            self._lj_pair_scale_tensor
            * 4.0
            * self._lj_epsilon_pair_tensor
            * attractive_term**LJ_REPULSIVE_EXPONENT_MULTIPLIER
            * switch_weight
        )
        attractive_pair_energy = (
            self._lj_pair_scale_tensor
            * 4.0
            * self._lj_epsilon_pair_tensor
            * attractive_term
            * switch_weight
        )
        volume_m3 = torch.abs(torch.linalg.det(box_vectors_m))
        repulsive_tail_energy_J = self._lj_repulsive_tail_coefficient_tensor / volume_m3
        attractive_tail_energies_J = (
            self._lj_attractive_tail_coefficients_by_ionic_count_tensor / volume_m3
        )
        lennard_jones_repulsion_energy = (
            torch.sum(repulsive_pair_energy) + repulsive_tail_energy_J
        )
        lennard_jones_attraction_energy = -torch.sum(
            attractive_pair_energy
        ) + torch.sum(attractive_tail_energies_J)
        fixed_energy = fixed_energy + lennard_jones_repulsion_energy
        fixed_energy = (
            fixed_energy
            - torch.sum(
                torch.where(
                    self._lj_pair_ionic_count_tensor == 0,
                    attractive_pair_energy,
                    torch.zeros_like(attractive_pair_energy),
                )
            )
            + attractive_tail_energies_J[0]
        )
        ion_neutral_energy = (
            ion_neutral_energy
            - torch.sum(
                torch.where(
                    self._lj_pair_ionic_count_tensor == 1,
                    attractive_pair_energy,
                    torch.zeros_like(attractive_pair_energy),
                )
            )
            + attractive_tail_energies_J[1]
        )
        ion_ion_energy = (
            ion_ion_energy
            - torch.sum(
                torch.where(
                    self._lj_pair_ionic_count_tensor == 2,
                    attractive_pair_energy,
                    torch.zeros_like(attractive_pair_energy),
                )
            )
            + attractive_tail_energies_J[2]
        )
        charges = self._charges_tensor
        ewald_alpha = self.numerics.ewald_splitting_per_m
        electrostatic_pair_distances_m = distance[
            self._electrostatic_pair_i_tensor,
            self._electrostatic_pair_j_tensor,
        ]
        real_pair_energies_J = (
            charges[self._electrostatic_pair_i_tensor]
            * charges[self._electrostatic_pair_j_tensor]
            * torch.special.erfc(ewald_alpha * electrostatic_pair_distances_m)
            / (4.0 * math.pi * EPS_0 * electrostatic_pair_distances_m)
        )
        real_electrostatic_energy = torch.sum(real_pair_energies_J)
        fixed_energy = fixed_energy + torch.sum(
            torch.where(
                self._electrostatic_pair_ionic_count_tensor == 0,
                real_pair_energies_J,
                torch.zeros_like(real_pair_energies_J),
            )
        )
        ion_neutral_energy = ion_neutral_energy + torch.sum(
            torch.where(
                self._electrostatic_pair_ionic_count_tensor == 1,
                real_pair_energies_J,
                torch.zeros_like(real_pair_energies_J),
            )
        )
        ion_ion_energy = ion_ion_energy + torch.sum(
            torch.where(
                self._electrostatic_pair_ionic_count_tensor == 2,
                real_pair_energies_J,
                torch.zeros_like(real_pair_energies_J),
            )
        )
        reciprocal = (
            2.0
            * math.pi
            * self._reciprocal_indices_tensor
            @ torch.linalg.inv(box_vectors_m)
        )
        reciprocal_squared = torch.sum(reciprocal**2, dim=1)
        phases = positions_m @ reciprocal.T
        neutral_charges = torch.where(
            self._neutral_atom_mask_tensor, charges, torch.zeros_like(charges)
        )
        ionic_charges = torch.where(
            self._ionic_atom_mask_tensor, charges, torch.zeros_like(charges)
        )
        phase_cosines = torch.cos(phases)
        phase_sines = torch.sin(phases)
        neutral_structure_real = torch.sum(
            neutral_charges[:, None] * phase_cosines, dim=0
        )
        neutral_structure_imaginary = torch.sum(
            neutral_charges[:, None] * phase_sines, dim=0
        )
        ionic_structure_real = torch.sum(ionic_charges[:, None] * phase_cosines, dim=0)
        ionic_structure_imaginary = torch.sum(
            ionic_charges[:, None] * phase_sines, dim=0
        )
        green_weights = torch.exp(-reciprocal_squared / (4.0 * ewald_alpha**2)) / (
            EPS_0 * volume_m3 * reciprocal_squared
        )
        fixed_reciprocal_energy = 0.5 * torch.sum(
            green_weights * (neutral_structure_real**2 + neutral_structure_imaginary**2)
        )
        ion_neutral_reciprocal_energy = torch.sum(
            green_weights
            * (
                neutral_structure_real * ionic_structure_real
                + neutral_structure_imaginary * ionic_structure_imaginary
            )
        )
        ion_ion_reciprocal_energy = 0.5 * torch.sum(
            green_weights * (ionic_structure_real**2 + ionic_structure_imaginary**2)
        )
        fixed_energy = fixed_energy + fixed_reciprocal_energy
        ion_neutral_energy = ion_neutral_energy + ion_neutral_reciprocal_energy
        ion_ion_energy = ion_ion_energy + ion_ion_reciprocal_energy
        reciprocal_electrostatic_energy = (
            fixed_reciprocal_energy
            + ion_neutral_reciprocal_energy
            + ion_ion_reciprocal_energy
        )
        fixed_self_energy = -(
            ewald_alpha
            * torch.sum(neutral_charges**2)
            / (4.0 * math.pi * math.sqrt(math.pi) * EPS_0)
        )
        ion_ion_self_energy = -(
            ewald_alpha
            * torch.sum(ionic_charges**2)
            / (4.0 * math.pi * math.sqrt(math.pi) * EPS_0)
        )
        fixed_energy = fixed_energy + fixed_self_energy
        ion_ion_energy = ion_ion_energy + ion_ion_self_energy
        ewald_self_energy = fixed_self_energy + ion_ion_self_energy
        electrostatic_exclusion_energy = torch.zeros((), dtype=TORCH_DTYPE)
        if self._excluded_pair_count > 0:
            excluded_displacements_m = displacement[
                self._excluded_pair_i_tensor, self._excluded_pair_j_tensor
            ]
            excluded_phases = excluded_displacements_m @ reciprocal.T
            excluded_reciprocal_energies_J = (
                charges[self._excluded_pair_i_tensor]
                * charges[self._excluded_pair_j_tensor]
                * torch.sum(
                    green_weights[None, :] * torch.cos(excluded_phases),
                    dim=1,
                )
            )
            excluded_pair_energies_J = self._excluded_pair_weights_tensor * (
                real_pair_energies_J[self._excluded_pair_indices_tensor]
                + excluded_reciprocal_energies_J
            )
            excluded_ionic_count = self._electrostatic_pair_ionic_count_tensor[
                self._excluded_pair_indices_tensor
            ]
            fixed_energy = fixed_energy - torch.sum(
                torch.where(
                    excluded_ionic_count == 0,
                    excluded_pair_energies_J,
                    torch.zeros_like(excluded_pair_energies_J),
                )
            )
            ion_neutral_energy = ion_neutral_energy - torch.sum(
                torch.where(
                    excluded_ionic_count == 1,
                    excluded_pair_energies_J,
                    torch.zeros_like(excluded_pair_energies_J),
                )
            )
            ion_ion_energy = ion_ion_energy - torch.sum(
                torch.where(
                    excluded_ionic_count == 2,
                    excluded_pair_energies_J,
                    torch.zeros_like(excluded_pair_energies_J),
                )
            )
            electrostatic_exclusion_energy = -torch.sum(excluded_pair_energies_J)
        physical_energy_terms = torch.stack(
            (
                bonded_energy,
                lennard_jones_repulsion_energy,
                lennard_jones_attraction_energy,
                real_electrostatic_energy,
                reciprocal_electrostatic_energy,
                ewald_self_energy,
                electrostatic_exclusion_energy,
            )
        )
        return (
            fixed_energy,
            ion_ion_energy,
            ion_neutral_energy,
            neutral_charges,
            ionic_charges,
            phases,
            displacement,
            distance,
            reciprocal,
            green_weights,
            physical_energy_terms,
        )

    def _energy_components_tensor(
        self,
        positions_m: torch.Tensor,
        box_vectors_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        component_batch = self._energy_components_batch_tensor(
            positions_m[None, :, :], box_vectors_m[None, :, :]
        )
        return tuple(component[0] for component in component_batch)

    def _energy_components_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            polarization_residuals,
            _physical_energy_terms,
        ) = self._energy_components_and_physical_terms_batch_tensor(
            positions_batch_m,
            box_vectors_batch_m,
        )
        return (
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            polarization_residuals,
        )

    def _energy_components_and_physical_terms_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            neutral_charges,
            ionic_charges,
            phases,
            displacements,
            distances,
            reciprocal_vectors,
            green_weights,
            nonpolar_physical_energy_terms,
        ) = torch.vmap(self._nonpolar_energy_components_tensor)(
            positions_batch_m, box_vectors_batch_m
        )
        if self._polarizable_atom_count == 0:
            return (
                fixed_energies,
                ion_ion_energies,
                ion_neutral_energies,
                torch.zeros_like(fixed_energies),
                torch.cat(
                    (
                        nonpolar_physical_energy_terms,
                        torch.zeros(
                            (positions_batch_m.shape[0], REST_INTERACTION_CLASS_COUNT),
                            dtype=TORCH_DTYPE,
                        ),
                    ),
                    dim=1,
                ),
            )
        (
            polarization_fixed,
            polarization_ion_ion,
            polarization_ion_neutral,
            polarization_residuals,
            polarization_physical_terms,
        ) = self._polarization_energy_components_batch(
            neutral_charges_C=neutral_charges,
            ionic_charges_C=ionic_charges,
            phases=phases,
            displacement_m=displacements,
            distance_m=distances,
            reciprocal_m_inv=reciprocal_vectors,
            green_weights_J_m_C2=green_weights,
            ewald_splitting_per_m=self.numerics.ewald_splitting_per_m,
        )
        return (
            fixed_energies + polarization_fixed,
            ion_ion_energies + polarization_ion_ion,
            ion_neutral_energies + polarization_ion_neutral,
            polarization_residuals,
            torch.cat(
                (nonpolar_physical_energy_terms, polarization_physical_terms),
                dim=1,
            ),
        )

    def _physical_energy_terms_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
    ) -> torch.Tensor:
        return self._energy_components_and_physical_terms_batch_tensor(
            positions_batch_m,
            box_vectors_batch_m,
        )[4]

    def _summed_energy_with_components_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
        lambda_values: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        (
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            polarization_residuals,
        ) = self._energy_components_batch_tensor(positions_batch_m, box_vectors_batch_m)
        energies = (
            fixed_energies
            + lambda_values * ion_ion_energies
            + torch.sqrt(lambda_values) * ion_neutral_energies
        )
        return torch.sum(energies), (
            energies,
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            polarization_residuals,
        )

    def _energy_force_components_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
        lambda_values: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        energy_gradient, components = torch.func.grad(
            self._summed_energy_with_components_tensor,
            argnums=0,
            has_aux=True,
        )(positions_batch_m, box_vectors_batch_m, lambda_values)
        (
            energies,
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            polarization_residuals,
        ) = components
        return (
            energies,
            -energy_gradient,
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            polarization_residuals,
        )

    def _physical_energy_force_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physical_lambdas = torch.ones(
            positions_batch_m.shape[0],
            dtype=TORCH_DTYPE,
        )
        if self._polarizable_atom_count == 0:
            (
                fixed_energies,
                ion_ion_energies,
                ion_neutral_energies,
                polarization_residuals,
            ) = self._energy_components_batch_tensor(
                positions_batch_m,
                box_vectors_batch_m,
            )
            energies = fixed_energies + ion_ion_energies + ion_neutral_energies
            forces = self._analytical_nonpolar_forces_batch_tensor(
                positions_batch_m,
                box_vectors_batch_m,
                physical_lambdas,
            )
        else:
            (
                energies,
                forces,
                _fixed_energies,
                _ion_ion_energies,
                _ion_neutral_energies,
                polarization_residuals,
            ) = self._energy_force_components_batch_tensor(
                positions_batch_m,
                box_vectors_batch_m,
                physical_lambdas,
            )
        return energies, forces, polarization_residuals

    def _analytical_nonpolar_forces_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
        lambda_values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, atom_count, _cartesian_count = positions_batch_m.shape
        forces_N = torch.zeros_like(positions_batch_m)
        if self._has_bonds:
            first_indices = self._bond_indices_tensor[:, 0]
            second_indices = self._bond_indices_tensor[:, 1]
            bond_displacements_m = _torch_minimum_image(
                positions_batch_m[:, first_indices]
                - positions_batch_m[:, second_indices],
                box_vectors_batch_m,
            )
            bond_lengths_m = torch.linalg.norm(bond_displacements_m, dim=2)
            bond_force_vectors_N = (
                -(
                    self._bond_force_constants_tensor[None, :]
                    * (bond_lengths_m - self._bond_lengths_tensor[None, :])
                    / bond_lengths_m
                )[:, :, None]
                * bond_displacements_m
            )
            forces_N = forces_N.index_add(1, first_indices, bond_force_vectors_N)
            forces_N = forces_N.index_add(1, second_indices, -bond_force_vectors_N)
        if self._has_angles:
            first_indices = self._angle_indices_tensor[:, 0]
            center_indices = self._angle_indices_tensor[:, 1]
            third_indices = self._angle_indices_tensor[:, 2]
            first_vectors_m = _torch_minimum_image(
                positions_batch_m[:, first_indices]
                - positions_batch_m[:, center_indices],
                box_vectors_batch_m,
            )
            second_vectors_m = _torch_minimum_image(
                positions_batch_m[:, third_indices]
                - positions_batch_m[:, center_indices],
                box_vectors_batch_m,
            )
            angle_cross_vectors_m2 = torch.linalg.cross(
                first_vectors_m, second_vectors_m, dim=2
            )
            regularized_cross_norms_m2 = torch.sqrt(
                torch.sum(angle_cross_vectors_m2**2, dim=2)
                + torch.finfo(TORCH_DTYPE).tiny
            )
            angle_dot_products_m2 = torch.sum(
                first_vectors_m * second_vectors_m,
                dim=2,
            )
            angle_values_rad = torch.atan2(
                regularized_cross_norms_m2, angle_dot_products_m2
            )
            energy_angle_derivatives_J_rad = self._angle_force_constants_tensor[
                None, :
            ] * (angle_values_rad - self._angle_values_tensor[None, :])
            angle_denominators_m4 = (
                regularized_cross_norms_m2**2 + angle_dot_products_m2**2
            )
            first_angle_gradients_per_m = (
                angle_dot_products_m2[:, :, None]
                * torch.linalg.cross(second_vectors_m, angle_cross_vectors_m2, dim=2)
                / regularized_cross_norms_m2[:, :, None]
                - regularized_cross_norms_m2[:, :, None] * second_vectors_m
            ) / angle_denominators_m4[:, :, None]
            second_angle_gradients_per_m = (
                angle_dot_products_m2[:, :, None]
                * torch.linalg.cross(angle_cross_vectors_m2, first_vectors_m, dim=2)
                / regularized_cross_norms_m2[:, :, None]
                - regularized_cross_norms_m2[:, :, None] * first_vectors_m
            ) / angle_denominators_m4[:, :, None]
            first_angle_forces_N = (
                -energy_angle_derivatives_J_rad[:, :, None]
                * first_angle_gradients_per_m
            )
            third_angle_forces_N = (
                -energy_angle_derivatives_J_rad[:, :, None]
                * second_angle_gradients_per_m
            )
            forces_N = forces_N.index_add(1, first_indices, first_angle_forces_N)
            forces_N = forces_N.index_add(1, third_indices, third_angle_forces_N)
            forces_N = forces_N.index_add(
                1,
                center_indices,
                -(first_angle_forces_N + third_angle_forces_N),
            )
        if self._torsion_indices_tensor.shape[0] > 0:
            first_indices = self._torsion_indices_tensor[:, 0]
            second_indices = self._torsion_indices_tensor[:, 1]
            third_indices = self._torsion_indices_tensor[:, 2]
            fourth_indices = self._torsion_indices_tensor[:, 3]
            first_bonds_m = _torch_minimum_image(
                positions_batch_m[:, second_indices]
                - positions_batch_m[:, first_indices],
                box_vectors_batch_m,
            )
            second_bonds_m = _torch_minimum_image(
                positions_batch_m[:, third_indices]
                - positions_batch_m[:, second_indices],
                box_vectors_batch_m,
            )
            third_bonds_m = _torch_minimum_image(
                positions_batch_m[:, fourth_indices]
                - positions_batch_m[:, third_indices],
                box_vectors_batch_m,
            )
            first_normals = torch.linalg.cross(first_bonds_m, second_bonds_m, dim=2)
            second_normals = torch.linalg.cross(second_bonds_m, third_bonds_m, dim=2)
            second_bond_squared_m2 = torch.sum(second_bonds_m**2, dim=2)
            second_bond_lengths_m = torch.sqrt(second_bond_squared_m2)
            dihedral_values_rad = torch.atan2(
                second_bond_lengths_m
                * torch.sum(first_bonds_m * second_normals, dim=2),
                torch.sum(first_normals * second_normals, dim=2),
            )
            energy_dihedral_derivatives_J_rad = (
                -self._torsion_amplitudes_tensor[None, :]
                * self._torsion_periodicities_tensor[None, :]
                * torch.sin(
                    self._torsion_periodicities_tensor[None, :] * dihedral_values_rad
                    - self._torsion_phases_tensor[None, :]
                )
            )
            first_dihedral_gradients_per_m = (
                -(
                    second_bond_lengths_m
                    / (
                        torch.sum(first_normals**2, dim=2)
                        + torch.finfo(TORCH_DTYPE).tiny
                    )
                )[:, :, None]
                * first_normals
            )
            fourth_dihedral_gradients_per_m = (
                second_bond_lengths_m
                / (torch.sum(second_normals**2, dim=2) + torch.finfo(TORCH_DTYPE).tiny)
            )[:, :, None] * second_normals
            first_second_projection = (
                torch.sum(first_bonds_m * second_bonds_m, dim=2)
                / second_bond_squared_m2
            )
            third_second_projection = (
                torch.sum(third_bonds_m * second_bonds_m, dim=2)
                / second_bond_squared_m2
            )
            second_dihedral_gradients_per_m = (
                -(first_second_projection + 1.0)[:, :, None]
                * first_dihedral_gradients_per_m
                + third_second_projection[:, :, None] * fourth_dihedral_gradients_per_m
            )
            third_dihedral_gradients_per_m = -(
                first_dihedral_gradients_per_m
                + second_dihedral_gradients_per_m
                + fourth_dihedral_gradients_per_m
            )
            dihedral_force_scale = -energy_dihedral_derivatives_J_rad[:, :, None]
            forces_N = forces_N.index_add(
                1,
                first_indices,
                dihedral_force_scale * first_dihedral_gradients_per_m,
            )
            forces_N = forces_N.index_add(
                1,
                second_indices,
                dihedral_force_scale * second_dihedral_gradients_per_m,
            )
            forces_N = forces_N.index_add(
                1,
                third_indices,
                dihedral_force_scale * third_dihedral_gradients_per_m,
            )
            forces_N = forces_N.index_add(
                1,
                fourth_indices,
                dihedral_force_scale * fourth_dihedral_gradients_per_m,
            )
        displacements_m = _torch_minimum_image(
            positions_batch_m[:, :, None, :] - positions_batch_m[:, None, :, :],
            box_vectors_batch_m,
        )
        distances_m = torch.linalg.norm(displacements_m, dim=3)
        lj_displacements_m = displacements_m[
            :, self._lj_pair_i_tensor, self._lj_pair_j_tensor
        ]
        lj_distances_m = distances_m[:, self._lj_pair_i_tensor, self._lj_pair_j_tensor]
        attractive_terms = (
            self._lj_sigma_pair_tensor[None, :] / lj_distances_m
        ) ** LJ_ATTRACTIVE_EXPONENT
        attractive_scales = torch.where(
            self._lj_pair_ionic_count_tensor[None, :] == 2,
            lambda_values[:, None],
            torch.where(
                self._lj_pair_ionic_count_tensor[None, :] == 1,
                torch.sqrt(lambda_values)[:, None],
                torch.ones_like(attractive_terms),
            ),
        )
        lj_force_coefficients_N_m = (
            self._lj_pair_scale_tensor[None, :]
            * 4.0
            * self._lj_epsilon_pair_tensor[None, :]
            * LJ_ATTRACTIVE_EXPONENT
            * (
                LJ_REPULSIVE_EXPONENT_MULTIPLIER
                * attractive_terms**LJ_REPULSIVE_EXPONENT_MULTIPLIER
                - attractive_scales * attractive_terms
            )
            / lj_distances_m**2
        )
        lj_forces_N = lj_force_coefficients_N_m[:, :, None] * lj_displacements_m
        forces_N = forces_N.index_add(1, self._lj_pair_i_tensor, lj_forces_N)
        forces_N = forces_N.index_add(1, self._lj_pair_j_tensor, -lj_forces_N)
        charge_scales = torch.where(
            self._ionic_atom_mask_tensor[None, :],
            torch.sqrt(lambda_values)[:, None],
            torch.ones((batch_size, atom_count), dtype=TORCH_DTYPE),
        )
        charges_C = self._charges_tensor[None, :] * charge_scales
        pair_i = self._electrostatic_pair_i_tensor
        pair_j = self._electrostatic_pair_j_tensor
        electrostatic_displacements_m = displacements_m[:, pair_i, pair_j]
        electrostatic_distances_m = distances_m[:, pair_i, pair_j]
        electrostatic_charge_products_C2 = charges_C[:, pair_i] * charges_C[:, pair_j]
        ewald_alpha = self.numerics.ewald_splitting_per_m
        scaled_distances = ewald_alpha * electrostatic_distances_m
        real_force_coefficients_N_m = (
            electrostatic_charge_products_C2
            * self._coulomb_pair_scales_tensor[pair_i, pair_j][None, :]
            / (4.0 * math.pi * EPS_0)
            * (
                torch.special.erfc(scaled_distances) / electrostatic_distances_m**3
                + 2.0
                * ewald_alpha
                * torch.exp(-(scaled_distances**2))
                / (math.sqrt(math.pi) * electrostatic_distances_m**2)
            )
        )
        real_forces_N = (
            real_force_coefficients_N_m[:, :, None] * electrostatic_displacements_m
        )
        forces_N = forces_N.index_add(1, pair_i, real_forces_N)
        forces_N = forces_N.index_add(1, pair_j, -real_forces_N)
        reciprocal_vectors_m_inv = (
            2.0
            * math.pi
            * torch.einsum(
                "kd,bdc->bkc",
                self._reciprocal_indices_tensor,
                torch.linalg.inv(box_vectors_batch_m),
            )
        )
        reciprocal_squared_m_inv2 = torch.sum(reciprocal_vectors_m_inv**2, dim=2)
        phases = torch.einsum(
            "bnd,bkd->bnk", positions_batch_m, reciprocal_vectors_m_inv
        )
        structure_real = torch.sum(charges_C[:, :, None] * torch.cos(phases), dim=1)
        structure_imaginary = torch.sum(
            charges_C[:, :, None] * torch.sin(phases), dim=1
        )
        volumes_m3 = torch.abs(torch.linalg.det(box_vectors_batch_m))
        green_weights_J_m_C2 = torch.exp(
            -reciprocal_squared_m_inv2 / (4.0 * ewald_alpha**2)
        ) / (EPS_0 * volumes_m3[:, None] * reciprocal_squared_m_inv2)
        reciprocal_amplitudes = charges_C[:, :, None] * (
            structure_real[:, None, :] * torch.sin(phases)
            - structure_imaginary[:, None, :] * torch.cos(phases)
        )
        reciprocal_forces_N = torch.einsum(
            "bnk,bk,bkd->bnd",
            reciprocal_amplitudes,
            green_weights_J_m_C2,
            reciprocal_vectors_m_inv,
        )
        forces_N += reciprocal_forces_N
        if self._excluded_pair_count > 0:
            excluded_pair_phases = (
                phases[:, self._excluded_pair_i_tensor]
                - phases[:, self._excluded_pair_j_tensor]
            )
            excluded_pair_force_vectors_N = -(
                charges_C[:, self._excluded_pair_i_tensor]
                * charges_C[:, self._excluded_pair_j_tensor]
                * self._excluded_pair_weights_tensor[None, :]
            )[:, :, None] * torch.einsum(
                "bpk,bk,bkd->bpd",
                torch.sin(excluded_pair_phases),
                green_weights_J_m_C2,
                reciprocal_vectors_m_inv,
            )
            forces_N = forces_N.index_add(
                1,
                self._excluded_pair_i_tensor,
                excluded_pair_force_vectors_N,
            )
            forces_N = forces_N.index_add(
                1,
                self._excluded_pair_j_tensor,
                -excluded_pair_force_vectors_N,
            )
        return forces_N

    def _polarization_energy_matrix_batch(
        self,
        charge_columns: torch.Tensor,
        phases: torch.Tensor,
        displacement_m: torch.Tensor,
        distance_m: torch.Tensor,
        reciprocal_m_inv: torch.Tensor,
        green_weights_J_m_C2: torch.Tensor,
        ewald_splitting_per_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_cosines = torch.cos(phases)
        phase_sines = torch.sin(phases)
        charge_structure_cosine = torch.sum(
            charge_columns[:, :, :, None] * phase_cosines[:, :, None, :], dim=1
        )
        charge_structure_sine = torch.sum(
            charge_columns[:, :, :, None] * phase_sines[:, :, None, :], dim=1
        )
        reciprocal_field_amplitudes = (
            phase_sines[:, :, None, :] * charge_structure_cosine[:, None, :, :]
            - phase_cosines[:, :, None, :] * charge_structure_sine[:, None, :, :]
        )
        reciprocal_electric_fields = torch.einsum(
            "birk,bk,bkd->bird",
            reciprocal_field_amplitudes,
            green_weights_J_m_C2,
            reciprocal_m_inv,
        )
        nonzero_distance_m = torch.where(
            torch.eye(distance_m.shape[1], dtype=torch.bool)[None, :, :],
            torch.ones_like(distance_m),
            distance_m,
        )
        real_field_coefficient = (
            torch.special.erfc(ewald_splitting_per_m * nonzero_distance_m)
            / nonzero_distance_m**3
            + 2.0
            * ewald_splitting_per_m
            * torch.exp(-((ewald_splitting_per_m * nonzero_distance_m) ** 2))
            / (math.sqrt(math.pi) * nonzero_distance_m**2)
        ) / (4.0 * math.pi * EPS_0)
        electrostatic_pair_scales = self._coulomb_pair_scales_tensor
        real_field_coefficient *= electrostatic_pair_scales
        real_electric_fields = torch.einsum(
            "bij,bjr,bijd->bird",
            real_field_coefficient,
            charge_columns,
            displacement_m,
        )
        excluded_reciprocal_fields = torch.zeros_like(reciprocal_electric_fields)
        if self._has_reciprocal_exclusions:
            excluded_pair_phases = (
                phases[:, self._excluded_pair_i_tensor]
                - phases[:, self._excluded_pair_j_tensor]
            )
            excluded_pair_vectors = (
                torch.einsum(
                    "bpk,bk,bkd->bpd",
                    torch.sin(excluded_pair_phases),
                    green_weights_J_m_C2,
                    reciprocal_m_inv,
                )
                * self._excluded_pair_weights_tensor[None, :, None]
            )
            excluded_reciprocal_fields = excluded_reciprocal_fields.index_add(
                1,
                self._excluded_pair_i_tensor,
                excluded_pair_vectors[:, :, None, :]
                * charge_columns[:, self._excluded_pair_j_tensor, :, None],
            )
            excluded_reciprocal_fields = excluded_reciprocal_fields.index_add(
                1,
                self._excluded_pair_j_tensor,
                -excluded_pair_vectors[:, :, None, :]
                * charge_columns[:, self._excluded_pair_i_tensor, :, None],
            )
        electric_fields = (
            reciprocal_electric_fields
            + real_electric_fields
            - excluded_reciprocal_fields
        )
        active_tensor = self._active_polarizable_indices_tensor
        active_displacements_m = displacement_m[:, active_tensor][:, :, active_tensor]
        active_distances_m = nonzero_distance_m[:, active_tensor][:, :, active_tensor]
        radial_first_derivative = -(
            torch.special.erfc(ewald_splitting_per_m * active_distances_m)
            / active_distances_m**2
            + 2.0
            * ewald_splitting_per_m
            * torch.exp(-((ewald_splitting_per_m * active_distances_m) ** 2))
            / (math.sqrt(math.pi) * active_distances_m)
        ) / (4.0 * math.pi * EPS_0)
        radial_second_derivative = (
            2.0
            * torch.special.erfc(ewald_splitting_per_m * active_distances_m)
            / active_distances_m**3
            + 4.0
            * ewald_splitting_per_m
            * torch.exp(-((ewald_splitting_per_m * active_distances_m) ** 2))
            / (math.sqrt(math.pi) * active_distances_m**2)
            + 4.0
            * ewald_splitting_per_m**3
            * torch.exp(-((ewald_splitting_per_m * active_distances_m) ** 2))
            / math.sqrt(math.pi)
        ) / (4.0 * math.pi * EPS_0)
        unit_displacements = active_displacements_m / active_distances_m[:, :, :, None]
        radial_outer = (
            unit_displacements[:, :, :, :, None] * unit_displacements[:, :, :, None, :]
        )
        identity = self._cartesian_identity_tensor
        real_hessian = radial_second_derivative[:, :, :, None, None] * radial_outer + (
            radial_first_derivative / active_distances_m
        )[:, :, :, None, None] * (identity[None, None, None, :, :] - radial_outer)
        real_hessian *= self._active_pair_scales_tensor[None, :, :, None, None]
        self_hessian = (
            -4.0
            * ewald_splitting_per_m**3
            / (3.0 * math.sqrt(math.pi) * 4.0 * math.pi * EPS_0)
        )
        active_phases = phases[:, active_tensor]
        inverse_alpha = 1.0 / self._polarizabilities_tensor[active_tensor]

        def interaction_field_components(
            dipoles: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            dipole_projections = torch.einsum(
                "bprd,bkd->bprk", dipoles, reciprocal_m_inv
            )
            dipole_structure_cosine = torch.sum(
                dipole_projections * torch.cos(active_phases)[:, :, None, :],
                dim=1,
            )
            dipole_structure_sine = torch.sum(
                dipole_projections * torch.sin(active_phases)[:, :, None, :],
                dim=1,
            )
            reciprocal_amplitudes = (
                torch.cos(active_phases)[:, :, None, :]
                * dipole_structure_cosine[:, None, :, :]
                + torch.sin(active_phases)[:, :, None, :]
                * dipole_structure_sine[:, None, :, :]
            )
            reciprocal_interaction_field = -torch.einsum(
                "bprk,bk,bkd->bprd",
                reciprocal_amplitudes,
                green_weights_J_m_C2,
                reciprocal_m_inv,
            )
            real_interaction_field = torch.einsum(
                "bijde,bjre->bird", real_hessian, dipoles
            )
            excluded_interaction_field = torch.zeros_like(reciprocal_interaction_field)
            if self._has_active_exclusions:
                excluded_pair_phases = (
                    active_phases[:, self._active_exclusion_pair_i_tensor]
                    - active_phases[:, self._active_exclusion_pair_j_tensor]
                )
                excluded_pair_hessian = (
                    -torch.einsum(
                        "bpk,bk,bkd,bke->bpde",
                        torch.cos(excluded_pair_phases),
                        green_weights_J_m_C2,
                        reciprocal_m_inv,
                        reciprocal_m_inv,
                    )
                    * self._active_exclusion_pair_weights_tensor[None, :, None, None]
                )
                excluded_interaction_field = excluded_interaction_field.index_add(
                    1,
                    self._active_exclusion_pair_i_tensor,
                    torch.einsum(
                        "bpde,bpre->bprd",
                        excluded_pair_hessian,
                        dipoles[:, self._active_exclusion_pair_j_tensor],
                    ),
                )
                excluded_interaction_field = excluded_interaction_field.index_add(
                    1,
                    self._active_exclusion_pair_j_tensor,
                    torch.einsum(
                        "bpde,bpre->bprd",
                        excluded_pair_hessian,
                        dipoles[:, self._active_exclusion_pair_i_tensor],
                    ),
                )
            self_interaction_field = self_hessian * dipoles
            return (
                reciprocal_interaction_field,
                real_interaction_field,
                excluded_interaction_field,
                self_interaction_field,
            )

        def apply_operator(dipoles: torch.Tensor) -> torch.Tensor:
            (
                reciprocal_interaction_field,
                real_interaction_field,
                excluded_interaction_field,
                self_interaction_field,
            ) = interaction_field_components(dipoles)
            return (
                inverse_alpha[None, :, None, None] * dipoles
                - reciprocal_interaction_field
                - real_interaction_field
                + excluded_interaction_field
                - self_interaction_field
            )

        active_electric_field = electric_fields[:, active_tensor]
        with torch.no_grad():
            dipoles = torch.zeros_like(active_electric_field)
            residual_vector = active_electric_field.detach().clone()
            preconditioned_residual = (
                self._polarizabilities_tensor[None, active_tensor, None, None]
                * residual_vector
            )
            search_direction = preconditioned_residual.clone()
            residual_inner_product = torch.sum(
                residual_vector * preconditioned_residual, dim=(1, 3)
            )
            maximum_iteration_count = CARTESIAN_DIMENSION * self._polarizable_atom_count

            def pcg_not_converged(
                iteration_index: torch.Tensor,
                _dipoles: torch.Tensor,
                current_residual: torch.Tensor,
                _preconditioned_residual: torch.Tensor,
                _search_direction: torch.Tensor,
                _residual_inner_product: torch.Tensor,
            ) -> torch.Tensor:
                return torch.logical_and(
                    iteration_index < maximum_iteration_count,
                    torch.max(torch.linalg.norm(current_residual, dim=(1, 3)))
                    > self.numerics.polarization_residual_tolerance_V_m,
                )

            def pcg_iteration(
                iteration_index: torch.Tensor,
                current_dipoles: torch.Tensor,
                current_residual: torch.Tensor,
                current_preconditioned_residual: torch.Tensor,
                current_search_direction: torch.Tensor,
                current_residual_inner_product: torch.Tensor,
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                residual_norms = torch.linalg.norm(current_residual, dim=(1, 3))
                unconverged = residual_norms > (
                    self.numerics.polarization_residual_tolerance_V_m
                )
                operator_search_direction = apply_operator(current_search_direction)
                step_denominator = torch.sum(
                    current_search_direction * operator_search_direction,
                    dim=(1, 3),
                )
                safe_step_denominator = torch.where(
                    unconverged,
                    step_denominator,
                    torch.ones_like(step_denominator),
                )
                step_size = torch.where(
                    unconverged,
                    current_residual_inner_product / safe_step_denominator,
                    torch.zeros_like(step_denominator),
                )
                next_dipoles = (
                    current_dipoles
                    + step_size[:, None, :, None] * current_search_direction
                )
                next_residual = (
                    current_residual
                    - step_size[:, None, :, None] * operator_search_direction
                )
                next_preconditioned_residual = (
                    self._polarizabilities_tensor[None, active_tensor, None, None]
                    * next_residual
                )
                next_residual_inner_product = torch.sum(
                    next_residual * next_preconditioned_residual, dim=(1, 3)
                )
                safe_residual_inner_product = torch.where(
                    unconverged,
                    current_residual_inner_product,
                    torch.ones_like(current_residual_inner_product),
                )
                conjugate_coefficient = torch.where(
                    unconverged,
                    next_residual_inner_product / safe_residual_inner_product,
                    torch.zeros_like(current_residual_inner_product),
                )
                next_search_direction = next_preconditioned_residual + (
                    conjugate_coefficient[:, None, :, None] * current_search_direction
                )
                return (
                    iteration_index + 1,
                    next_dipoles,
                    next_residual,
                    next_preconditioned_residual,
                    next_search_direction,
                    next_residual_inner_product,
                )

            (
                _iteration_count,
                dipoles,
                _residual_vector,
                _preconditioned_residual,
                _search_direction,
                _residual_inner_product,
            ) = torch.while_loop(
                pcg_not_converged,
                pcg_iteration,
                (
                    torch.zeros((), dtype=torch.int64),
                    dipoles,
                    residual_vector,
                    preconditioned_residual,
                    search_direction,
                    residual_inner_product,
                ),
            )
        stationary_dipoles = dipoles.detach()
        operator_dipoles = apply_operator(stationary_dipoles)
        residual = torch.max(
            torch.linalg.norm(operator_dipoles - active_electric_field, dim=(1, 3)),
            dim=1,
        ).values
        polarization_energy_matrix = (1.0 / (2.0 * 2.0)) * (
            torch.einsum("bird,bisd->brs", stationary_dipoles, operator_dipoles)
            + torch.einsum("bisd,bird->brs", stationary_dipoles, operator_dipoles)
        ) - 0.5 * (
            torch.einsum("bird,bisd->brs", stationary_dipoles, active_electric_field)
            + torch.einsum("bisd,bird->brs", stationary_dipoles, active_electric_field)
        )
        (
            reciprocal_interaction_field,
            real_interaction_field,
            excluded_interaction_field,
            self_interaction_field,
        ) = interaction_field_components(stationary_dipoles)
        physical_dipoles = torch.sum(stationary_dipoles, dim=2)
        physical_reciprocal_interaction_field = torch.sum(
            reciprocal_interaction_field, dim=2
        )
        physical_real_interaction_field = torch.sum(real_interaction_field, dim=2)
        physical_excluded_interaction_field = torch.sum(
            excluded_interaction_field, dim=2
        )
        physical_self_interaction_field = torch.sum(self_interaction_field, dim=2)
        physical_reciprocal_electric_field = torch.sum(
            reciprocal_electric_fields[:, active_tensor], dim=2
        )
        physical_real_electric_field = torch.sum(
            real_electric_fields[:, active_tensor], dim=2
        )
        physical_excluded_electric_field = torch.sum(
            excluded_reciprocal_fields[:, active_tensor], dim=2
        )
        physical_polarization_energy = 0.5 * torch.sum(
            physical_dipoles
            * (
                inverse_alpha[None, :, None] * physical_dipoles
                - physical_reciprocal_interaction_field
                - physical_real_interaction_field
            ),
            dim=(1, 2),
        ) - torch.sum(
            physical_dipoles
            * (physical_reciprocal_electric_field + physical_real_electric_field),
            dim=(1, 2),
        )
        physical_polarization_self_energy = -0.5 * torch.sum(
            physical_dipoles * physical_self_interaction_field,
            dim=(1, 2),
        )
        physical_polarization_exclusion_energy = 0.5 * torch.sum(
            physical_dipoles * physical_excluded_interaction_field,
            dim=(1, 2),
        ) + torch.sum(
            physical_dipoles * physical_excluded_electric_field,
            dim=(1, 2),
        )
        physical_polarization_terms = torch.stack(
            (
                physical_polarization_energy,
                physical_polarization_self_energy,
                physical_polarization_exclusion_energy,
            ),
            dim=1,
        )
        return (
            polarization_energy_matrix,
            residual,
            physical_polarization_terms,
            physical_dipoles,
        )

    def _polarization_energy_components_batch(
        self,
        neutral_charges_C: torch.Tensor,
        ionic_charges_C: torch.Tensor,
        phases: torch.Tensor,
        displacement_m: torch.Tensor,
        distance_m: torch.Tensor,
        reciprocal_m_inv: torch.Tensor,
        green_weights_J_m_C2: torch.Tensor,
        ewald_splitting_per_m: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        energy_matrix, residual, physical_polarization_terms, _physical_dipoles = (
            self._polarization_energy_matrix_batch(
                charge_columns=torch.stack((neutral_charges_C, ionic_charges_C), dim=2),
                phases=phases,
                displacement_m=displacement_m,
                distance_m=distance_m,
                reciprocal_m_inv=reciprocal_m_inv,
                green_weights_J_m_C2=green_weights_J_m_C2,
                ewald_splitting_per_m=ewald_splitting_per_m,
            )
        )
        return (
            energy_matrix[:, 0, 0],
            energy_matrix[:, 1, 1],
            2.0 * energy_matrix[:, 0, 1],
            residual,
            physical_polarization_terms,
        )

    def _induced_polarization_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
    ) -> torch.Tensor:
        if self._polarizable_atom_count == 0:
            return torch.zeros(
                (positions_batch_m.shape[0], CARTESIAN_DIMENSION),
                dtype=TORCH_DTYPE,
            )
        (
            _fixed_energies,
            _ion_ion_energies,
            _ion_neutral_energies,
            neutral_charges,
            ionic_charges,
            phases,
            displacements,
            distances,
            reciprocal_vectors,
            green_weights,
            _nonpolar_physical_energy_terms,
        ) = torch.vmap(self._nonpolar_energy_components_tensor)(
            positions_batch_m,
            box_vectors_batch_m,
        )
        (
            _energy_matrix,
            residuals_V_m,
            _physical_polarization_terms,
            physical_dipoles_C_m,
        ) = self._polarization_energy_matrix_batch(
            charge_columns=torch.stack(
                (neutral_charges, ionic_charges),
                dim=2,
            ),
            phases=phases,
            displacement_m=displacements,
            distance_m=distances,
            reciprocal_m_inv=reciprocal_vectors,
            green_weights_J_m_C2=green_weights,
            ewald_splitting_per_m=self.numerics.ewald_splitting_per_m,
        )
        if float(torch.max(residuals_V_m)) > (
            self.numerics.polarization_residual_tolerance_V_m
        ):
            raise RuntimeError(
                "induced polarization current uses an unconverged dipole solve"
            )
        return torch.sum(physical_dipoles_C_m, dim=1)

    def energy_J(self, positions_m: Array, box_vectors_m: Array) -> float:
        self._validate_periodic_accuracy(box_vectors_m[None, :, :])
        return float(
            self._energy_tensor(
                torch.as_tensor(positions_m), torch.as_tensor(box_vectors_m)
            ).detach()
        )

    def tempered_energy_for_configuration_J(
        self,
        positions_m: Array,
        box_vectors_m: Array,
        lambda_value: float,
    ) -> float:
        self._validate_periodic_accuracy(box_vectors_m[None, :, :])
        return float(
            self._energy_tensor(
                torch.as_tensor(positions_m),
                torch.as_tensor(box_vectors_m),
                lambda_value,
            ).detach()
        )

    def energy_components_J(
        self, positions_m: Array, box_vectors_m: Array
    ) -> TemperedEnergyComponents:
        fixed_energy_J = self.tempered_energy_for_configuration_J(
            positions_m, box_vectors_m, 0.0
        )
        quarter_energy_J = self.tempered_energy_for_configuration_J(
            positions_m, box_vectors_m, COMPONENT_DECOMPOSITION_LAMBDA
        )
        physical_energy_J = self.tempered_energy_for_configuration_J(
            positions_m, box_vectors_m, 1.0
        )
        quarter_increment_J = quarter_energy_J - fixed_energy_J
        physical_increment_J = physical_energy_J - fixed_energy_J
        ion_ion_energy_J = 2.0 * physical_increment_J - 4.0 * quarter_increment_J
        ion_neutral_energy_J = physical_increment_J - ion_ion_energy_J
        return TemperedEnergyComponents(
            fixed_J=fixed_energy_J,
            ion_ion_J=ion_ion_energy_J,
            ion_neutral_J=ion_neutral_energy_J,
        )

    @staticmethod
    def tempered_energy_J(
        components: TemperedEnergyComponents, lambda_value: float
    ) -> float:
        if not 0.0 <= lambda_value <= 1.0:
            raise ValueError("ionic interaction lambda must be in [0, 1]")
        return float(
            components.fixed_J
            + lambda_value * components.ion_ion_J
            + math.sqrt(lambda_value) * components.ion_neutral_J
        )

    def forces_N(self, positions_m: Array, box_vectors_m: Array) -> Array:
        self._validate_periodic_accuracy(box_vectors_m[None, :, :])
        positions = torch.tensor(positions_m, dtype=TORCH_DTYPE, requires_grad=True)
        energy = self._energy_tensor(positions, torch.as_tensor(box_vectors_m))
        return -torch.autograd.grad(energy, positions)[0].detach().numpy()

    def tempered_forces_N(
        self,
        positions_m: Array,
        box_vectors_m: Array,
        lambda_value: float,
    ) -> Array:
        self._validate_periodic_accuracy(box_vectors_m[None, :, :])
        positions = torch.tensor(positions_m, dtype=TORCH_DTYPE, requires_grad=True)
        energy = self._energy_tensor(
            positions, torch.as_tensor(box_vectors_m), lambda_value
        )
        return -torch.autograd.grad(energy, positions)[0].detach().numpy()

    def energy_force_components_batch(
        self,
        positions_batch_m: Array,
        box_vectors_batch_m: Array,
        lambda_values: Array,
    ) -> BatchedHamiltonianResult:
        positions = torch.as_tensor(positions_batch_m, dtype=TORCH_DTYPE)
        box_vectors = torch.as_tensor(box_vectors_batch_m, dtype=TORCH_DTYPE)
        lambdas = torch.as_tensor(lambda_values, dtype=TORCH_DTYPE)
        self._validate_batched_hamiltonian_inputs(positions, box_vectors, lambdas)
        self._validate_periodic_accuracy(box_vectors_batch_m)
        if self._polarizable_atom_count == 0:
            (
                fixed_energies,
                ion_ion_energies,
                ion_neutral_energies,
                polarization_residuals,
            ) = self._compiled_energy_components_batch_tensor(positions, box_vectors)
            energies = (
                fixed_energies
                + lambdas * ion_ion_energies
                + torch.sqrt(lambdas) * ion_neutral_energies
            )
            if bool(torch.all(lambdas == 1.0)):
                energies, forces, physical_residuals = (
                    self._compiled_physical_energy_force_batch_tensor(
                        positions,
                        box_vectors,
                    )
                )
                polarization_residuals = torch.maximum(
                    polarization_residuals,
                    physical_residuals,
                )
            else:
                forces = self._compiled_analytical_nonpolar_forces_batch_tensor(
                    positions, box_vectors, lambdas
                )
        else:
            (
                energies,
                forces,
                fixed_energies,
                ion_ion_energies,
                ion_neutral_energies,
                polarization_residuals,
            ) = self._compiled_energy_force_components_batch_tensor(
                positions, box_vectors, lambdas
            )
        maximum_residual = float(torch.max(polarization_residuals))
        if maximum_residual > self.numerics.polarization_residual_tolerance_V_m:
            raise RuntimeError(
                "induced-dipole solve did not meet polarization tolerance: "
                f"{maximum_residual:.6e} V/m"
            )
        return BatchedHamiltonianResult(
            energy_J=energies.detach().numpy(),
            forces_N=forces.detach().numpy(),
            fixed_energy_J=fixed_energies.detach().numpy(),
            ion_ion_energy_J=ion_ion_energies.detach().numpy(),
            ion_neutral_energy_J=ion_neutral_energies.detach().numpy(),
            polarization_residual_V_m=polarization_residuals.detach().numpy(),
        )

    def physical_energy_force_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if positions_batch_m.dtype != TORCH_DTYPE:
            raise ValueError("physical relaxation positions must use torch.float64")
        if box_vectors_batch_m.dtype != TORCH_DTYPE:
            raise ValueError("physical relaxation boxes must use torch.float64")
        if (
            positions_batch_m.ndim != 3
            or positions_batch_m.shape[2] != CARTESIAN_DIMENSION
        ):
            raise ValueError(
                "physical relaxation positions must have shape (batch,N,3)"
            )
        if box_vectors_batch_m.shape != (
            positions_batch_m.shape[0],
            CARTESIAN_DIMENSION,
            CARTESIAN_DIMENSION,
        ):
            raise ValueError("physical relaxation boxes must have shape (batch,3,3)")
        energies, forces, polarization_residuals = (
            self._compiled_physical_energy_force_batch_tensor(
                positions_batch_m,
                box_vectors_batch_m,
            )
        )
        if not bool(torch.all(torch.isfinite(energies))):
            raise RuntimeError("physical relaxation produced nonfinite energy")
        if not bool(torch.all(torch.isfinite(forces))):
            raise RuntimeError("physical relaxation produced nonfinite force")
        maximum_residual = float(torch.max(polarization_residuals))
        if maximum_residual > self.numerics.polarization_residual_tolerance_V_m:
            raise RuntimeError(
                "physical relaxation polarization residual exceeds tolerance: "
                f"residual_V_m={maximum_residual:.12g}, "
                "tolerance_V_m="
                f"{self.numerics.polarization_residual_tolerance_V_m:.12g}"
            )
        return energies, forces, polarization_residuals

    @staticmethod
    def _validate_batched_hamiltonian_inputs(
        positions: torch.Tensor,
        box_vectors: torch.Tensor,
        lambdas: torch.Tensor,
    ) -> None:
        if positions.ndim != 3 or positions.shape[2] != CARTESIAN_DIMENSION:
            raise ValueError("batched positions must have shape (batch, atom, 3)")
        if box_vectors.shape != (
            positions.shape[0],
            CARTESIAN_DIMENSION,
            CARTESIAN_DIMENSION,
        ):
            raise ValueError("batched boxes must have shape (batch, 3, 3)")
        if lambdas.shape != (positions.shape[0],):
            raise ValueError("batched lambdas must identify every configuration")
        if torch.any(lambdas <= 0.0) or torch.any(lambdas > 1.0):
            raise ValueError("batched lambdas must lie in (0, 1]")

    def _validate_periodic_accuracy(self, box_vectors_batch_m: Array) -> None:
        box_lengths_m = np.linalg.norm(box_vectors_batch_m, axis=2)
        maximum_box_length_m = float(np.max(box_lengths_m))
        if maximum_box_length_m > self.numerics.ewald_maximum_box_length_m:
            raise ValueError(
                "box exceeds the reciprocal grid's configured physical coverage: "
                f"box_length_m={maximum_box_length_m:.12g}, "
                "ewald_maximum_box_length_m="
                f"{self.numerics.ewald_maximum_box_length_m:.12g}"
            )
        minimum_box_length_m = float(np.min(box_lengths_m))
        required_real_space_length_m = (
            2.0
            * float(erfcinv(self.numerics.ewald_reciprocal_relative_tolerance))
            / self.numerics.ewald_splitting_per_m
        )
        if minimum_box_length_m < required_real_space_length_m:
            raise ValueError(
                "box is too short for the configured real-space Ewald accuracy: "
                f"box_length_m={minimum_box_length_m:.12g}, "
                f"required_length_m={required_real_space_length_m:.12g}"
            )
        minimum_lennard_jones_box_length_m = 2.0 * self.numerics.lennard_jones_cutoff_m
        if minimum_box_length_m <= minimum_lennard_jones_box_length_m:
            raise ValueError(
                "box must exceed twice the physical Lennard-Jones cutoff: "
                f"box_length_m={minimum_box_length_m:.12g}, "
                f"required_length_m={minimum_lennard_jones_box_length_m:.12g}"
            )

    def energy_components_batch(
        self,
        positions_batch_m: Array,
        box_vectors_batch_m: Array,
    ) -> BatchedEnergyComponents:
        positions = torch.as_tensor(positions_batch_m, dtype=TORCH_DTYPE)
        box_vectors = torch.as_tensor(box_vectors_batch_m, dtype=TORCH_DTYPE)
        if positions.ndim != 3 or positions.shape[2] != CARTESIAN_DIMENSION:
            raise ValueError("batched positions must have shape (batch, atom, 3)")
        if box_vectors.shape != (
            positions.shape[0],
            CARTESIAN_DIMENSION,
            CARTESIAN_DIMENSION,
        ):
            raise ValueError("batched boxes must have shape (batch, 3, 3)")
        self._validate_periodic_accuracy(box_vectors_batch_m)
        (
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            polarization_residuals,
        ) = self._compiled_energy_components_batch_tensor(positions, box_vectors)
        maximum_residual = float(torch.max(polarization_residuals))
        if maximum_residual > self.numerics.polarization_residual_tolerance_V_m:
            raise RuntimeError(
                "induced-dipole solve did not meet polarization tolerance: "
                f"{maximum_residual:.6e} V/m"
            )
        return BatchedEnergyComponents(
            fixed_energy_J=fixed_energies.detach().numpy(),
            ion_ion_energy_J=ion_ion_energies.detach().numpy(),
            ion_neutral_energy_J=ion_neutral_energies.detach().numpy(),
            polarization_residual_V_m=polarization_residuals.detach().numpy(),
        )

    def physical_energy_terms_batch(
        self,
        positions_batch_m: Array,
        box_vectors_batch_m: Array,
    ) -> BatchedPhysicalEnergyTerms:
        positions = torch.as_tensor(positions_batch_m, dtype=TORCH_DTYPE)
        box_vectors = torch.as_tensor(box_vectors_batch_m, dtype=TORCH_DTYPE)
        if positions.ndim != 3 or positions.shape[2] != CARTESIAN_DIMENSION:
            raise ValueError("batched positions must have shape (batch, atom, 3)")
        if box_vectors.shape != (
            positions.shape[0],
            CARTESIAN_DIMENSION,
            CARTESIAN_DIMENSION,
        ):
            raise ValueError("batched boxes must have shape (batch, 3, 3)")
        self._validate_periodic_accuracy(box_vectors_batch_m)
        energy_terms = (
            self._compiled_physical_energy_terms_batch_tensor(
                positions,
                box_vectors,
            )
            .detach()
            .numpy()
        )
        (
            bonded_energy_J,
            lennard_jones_repulsion_energy_J,
            lennard_jones_attraction_energy_J,
            real_electrostatic_energy_J,
            reciprocal_electrostatic_energy_J,
            ewald_self_energy_J,
            electrostatic_exclusion_energy_J,
            polarization_energy_J,
            polarization_self_energy_J,
            polarization_exclusion_energy_J,
        ) = energy_terms.T
        return BatchedPhysicalEnergyTerms(
            bonded_energy_J=bonded_energy_J,
            lennard_jones_repulsion_energy_J=lennard_jones_repulsion_energy_J,
            lennard_jones_attraction_energy_J=lennard_jones_attraction_energy_J,
            real_electrostatic_energy_J=real_electrostatic_energy_J,
            reciprocal_electrostatic_energy_J=reciprocal_electrostatic_energy_J,
            ewald_self_energy_J=ewald_self_energy_J,
            electrostatic_exclusion_energy_J=electrostatic_exclusion_energy_J,
            polarization_energy_J=polarization_energy_J,
            polarization_self_energy_J=polarization_self_energy_J,
            polarization_exclusion_energy_J=polarization_exclusion_energy_J,
        )


def _report_hrex_block(
    stage: str,
    block_index: int,
    block_elapsed_s: float,
    total_elapsed_s: float,
    state: IonicHrexState,
    settings: IonicHrexSettings,
    block: IonicHrexBlock,
) -> None:
    hmc_attempts_by_lambda = np.sum(state.hmc_attempts, axis=0)
    hmc_expected_acceptance_by_lambda = np.divide(
        np.sum(state.hmc_expected_acceptance_sums, axis=0),
        hmc_attempts_by_lambda,
        out=np.zeros(len(settings.lambdas), dtype=float),
        where=hmc_attempts_by_lambda > 0,
    )
    hmc_realized_acceptance_by_lambda = np.divide(
        np.sum(state.hmc_acceptances, axis=0),
        hmc_attempts_by_lambda,
        out=np.zeros(len(settings.lambdas), dtype=float),
        where=hmc_attempts_by_lambda > 0,
    )
    hmc_absolute_energy_error_by_lambda = np.divide(
        np.sum(state.hmc_absolute_energy_error_over_kbt_sums, axis=0),
        hmc_attempts_by_lambda,
        out=np.zeros(len(settings.lambdas), dtype=float),
        where=hmc_attempts_by_lambda > 0,
    )
    hmc_rms_com_displacement_by_lambda_m = np.sqrt(
        np.divide(
            np.sum(
                state.hmc_molecular_com_squared_displacement_sums_m2,
                axis=0,
            ),
            hmc_attempts_by_lambda,
            out=np.zeros(len(settings.lambdas), dtype=float),
            where=hmc_attempts_by_lambda > 0,
        )
    )
    exchange_acceptance_by_ladder = np.divide(
        state.exchange_acceptances,
        state.exchange_attempts,
        out=np.zeros_like(state.exchange_acceptances, dtype=float),
        where=state.exchange_attempts > 0,
    )
    exchange_expected_acceptance_by_ladder = np.divide(
        state.exchange_expected_acceptance_sums,
        state.exchange_attempts,
        out=np.zeros_like(state.exchange_expected_acceptance_sums, dtype=float),
        where=state.exchange_attempts > 0,
    )
    pooled_exchange_expected_acceptance = float(
        np.sum(state.exchange_expected_acceptance_sums)
        / max(int(np.sum(state.exchange_attempts)), 1)
    )
    seconds_per_cycle = block_elapsed_s / block.cycle_count
    seconds_per_force_evaluation = block_elapsed_s / block.force_evaluation_count
    print(
        f"[HREX] stage={stage} block={block_index} "
        f"elapsed_s={total_elapsed_s:.3f} block_s={block_elapsed_s:.3f} "
        f"states={state.positions_m.shape[0]} lambdas={settings.lambdas} "
        f"retained={block.physical_configurations_m.shape[0]} "
        f"seconds_per_cycle={seconds_per_cycle:.6f} "
        f"seconds_per_force={seconds_per_force_evaluation:.6f} "
        f"hmc_timestep_s={tuple(float(value) for value in state.hmc_step_sizes_s.reshape(settings.independent_ladder_count, len(settings.lambdas))[0])} "
        f"hmc_expected={tuple(float(value) for value in hmc_expected_acceptance_by_lambda)} "
        f"hmc_realized={tuple(float(value) for value in hmc_realized_acceptance_by_lambda)} "
        f"hmc_rms_com_m={tuple(float(value) for value in hmc_rms_com_displacement_by_lambda_m)} "
        f"hmc_abs_delta_h_over_kbt={tuple(float(value) for value in hmc_absolute_energy_error_by_lambda)} "
        f"round_trips={tuple(map(tuple, state.completed_round_trips))} "
        f"edge_expected={tuple(map(tuple, exchange_expected_acceptance_by_ladder))} "
        f"edge_realized={tuple(map(tuple, exchange_acceptance_by_ladder))} "
        f"pooled_edge_expected={pooled_exchange_expected_acceptance:.6f}",
        flush=True,
    )


def relax_initial_configurations(
    model: AnalyticalPeriodicInteratomicModel,
    initial_systems: tuple[MolecularSystem, ...],
    dynamics: DynamicsSettings,
    checkpoint_path: Path,
    checkpoint_fingerprint: str,
    checkpoint_metadata,
) -> tuple[Array, Array, int]:
    if not initial_systems:
        raise ValueError("initial relaxation requires at least one molecular system")
    initial_positions_by_chain_m = np.stack(
        tuple(system.positions_m for system in initial_systems)
    )
    box_vectors_by_chain_m = np.stack(
        tuple(system.box_vectors_m for system in initial_systems)
    )
    if initial_positions_by_chain_m.shape[1:] != model.system.positions_m.shape:
        raise ValueError("initial relaxation coordinates do not match the topology")
    if any(
        not np.array_equal(system.masses_kg, model.system.masses_kg)
        for system in initial_systems
    ):
        raise ValueError("initial relaxation systems do not share atomic masses")
    chain_count = len(initial_systems)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_temporary_path = checkpoint_path.with_suffix(
        f"{checkpoint_path.suffix}.tmp"
    )

    def write_relaxation_checkpoint(
        stage: str,
        accepted_positions_m: torch.Tensor,
        timesteps_s: Array,
        damping: Array,
        positive_power_steps: Array,
        best_maximum_forces_N: Array,
        accepted_energies_J: Array,
        has_accepted_energy: Array,
        stagnant_iterations: Array,
        converged: Array,
        force_evaluation_count: int,
        iteration: int,
        elapsed_s: float,
    ) -> None:
        checkpoint_payload = {
            "schema": "hamiltonian_initialization_v2",
            "fingerprint": checkpoint_fingerprint,
            "metadata": checkpoint_metadata,
            "stage": stage,
            "systems": tuple(
                asdict(
                    replace(
                        system,
                        positions_m=accepted_positions_m[index].detach().numpy(),
                    )
                )
                for index, system in enumerate(initial_systems)
            ),
            "accepted_positions_m": accepted_positions_m.detach().numpy(),
            "velocities_m_s": np.zeros_like(accepted_positions_m.detach().numpy()),
            "timesteps_s": timesteps_s,
            "damping": damping,
            "positive_power_steps": positive_power_steps,
            "best_maximum_forces_N": best_maximum_forces_N,
            "accepted_energies_J": accepted_energies_J,
            "has_accepted_energy": has_accepted_energy,
            "stagnant_iterations": stagnant_iterations,
            "converged": converged,
            "force_evaluation_count": force_evaluation_count,
            "iteration": iteration,
            "elapsed_s": elapsed_s,
        }
        with checkpoint_temporary_path.open("wb") as checkpoint_file:
            pickle.dump(checkpoint_payload, checkpoint_file)
        os.replace(checkpoint_temporary_path, checkpoint_path)

    positions = torch.as_tensor(initial_positions_by_chain_m, dtype=TORCH_DTYPE).clone()
    accepted_positions = positions.clone()
    velocities = torch.zeros_like(positions)
    timesteps_s = np.full(chain_count, dynamics.initial_relaxation_timestep_s)
    damping = np.full(chain_count, dynamics.initial_relaxation_initial_damping)
    positive_power_steps = np.zeros(chain_count, dtype=int)
    best_maximum_forces_N = np.full(chain_count, np.inf)
    accepted_energies_J = np.zeros(chain_count)
    has_accepted_energy = np.zeros(chain_count, dtype=bool)
    stagnant_iterations = np.zeros(chain_count, dtype=int)
    converged = np.zeros(chain_count, dtype=bool)
    force_evaluation_count = 0
    invocation_force_evaluation_count = 0
    starting_iteration = 0
    previous_elapsed_s = 0.0
    if checkpoint_path.is_file():
        with checkpoint_path.open("rb") as checkpoint_file:
            checkpoint_payload = pickle.load(checkpoint_file)
        if (
            "schema" not in checkpoint_payload
            or checkpoint_payload["schema"] != "hamiltonian_initialization_v2"
        ):
            raise ValueError(
                "initialization checkpoint uses a retired serialization schema"
            )
        if checkpoint_payload["fingerprint"] != checkpoint_fingerprint:
            raise ValueError(
                "initialization checkpoint does not match the requested recipe, "
                "topology, or numerical settings"
            )
        checkpoint_systems = tuple(
            _molecular_system_from_checkpoint_record(system_record)
            for system_record in checkpoint_payload["systems"]
        )
        positions = torch.as_tensor(
            np.stack(tuple(system.positions_m for system in checkpoint_systems)),
            dtype=TORCH_DTYPE,
        )
        if checkpoint_payload["stage"] == "packed":
            accepted_positions = positions.clone()
        else:
            required_checkpoint_fields = (
                "accepted_positions_m",
                "accepted_energies_J",
                "has_accepted_energy",
            )
            missing_checkpoint_fields = tuple(
                field_name
                for field_name in required_checkpoint_fields
                if field_name not in checkpoint_payload
            )
            if missing_checkpoint_fields:
                raise ValueError(
                    "initialization checkpoint predates rollback-safe FIRE state: "
                    f"missing_fields={missing_checkpoint_fields}"
                )
            accepted_positions = torch.as_tensor(
                checkpoint_payload["accepted_positions_m"],
                dtype=TORCH_DTYPE,
            )
        velocities = torch.as_tensor(
            checkpoint_payload["velocities_m_s"],
            dtype=TORCH_DTYPE,
        )
        timesteps_s = np.asarray(checkpoint_payload["timesteps_s"], dtype=float)
        damping = np.asarray(checkpoint_payload["damping"], dtype=float)
        positive_power_steps = np.asarray(
            checkpoint_payload["positive_power_steps"], dtype=int
        )
        best_maximum_forces_N = np.asarray(
            checkpoint_payload["best_maximum_forces_N"], dtype=float
        )
        if checkpoint_payload["stage"] != "packed":
            accepted_energies_J = np.asarray(
                checkpoint_payload["accepted_energies_J"], dtype=float
            )
            has_accepted_energy = np.asarray(
                checkpoint_payload["has_accepted_energy"], dtype=bool
            )
        stagnant_iterations = np.asarray(
            checkpoint_payload["stagnant_iterations"], dtype=int
        )
        converged = np.asarray(checkpoint_payload["converged"], dtype=bool)
        force_evaluation_count = int(checkpoint_payload["force_evaluation_count"])
        starting_iteration = int(checkpoint_payload["iteration"])
        previous_elapsed_s = float(checkpoint_payload["elapsed_s"])
        if checkpoint_payload["stage"] == "relaxed" and np.all(converged):
            return positions.numpy(), best_maximum_forces_N, force_evaluation_count
        print(
            "[relaxation restart] "
            f"iteration={starting_iteration} "
            f"force_evaluations={force_evaluation_count} "
            f"converged={tuple(bool(value) for value in converged)}",
            flush=True,
        )
    box_vectors = torch.as_tensor(box_vectors_by_chain_m, dtype=TORCH_DTYPE)
    atomic_masses_kg = torch.as_tensor(
        model.system.masses_kg[None, :, None],
        dtype=TORCH_DTYPE,
    )
    pair_i, pair_j = np.where(np.triu(model.system.nonbonded_mask, 1))
    contact_distances_m = 0.5 * (
        model.system.lj_sigma_m[pair_i] + model.system.lj_sigma_m[pair_j]
    )
    relaxation_start_time = time.perf_counter()
    iteration = starting_iteration
    last_maximum_forces_N = best_maximum_forces_N.copy()
    while not np.all(converged):
        active_chain_indices = np.flatnonzero(~converged)
        required_force_calls = math.ceil(
            active_chain_indices.size / dynamics.force_batch_size
        )
        if (
            invocation_force_evaluation_count + required_force_calls
            > dynamics.initial_relaxation_maximum_force_evaluations
        ):
            break
        energies_J = torch.zeros(chain_count, dtype=TORCH_DTYPE)
        forces_N = torch.zeros_like(positions)
        for chunk_start in range(
            0,
            active_chain_indices.size,
            dynamics.force_batch_size,
        ):
            chunk_indices = active_chain_indices[
                chunk_start : chunk_start + dynamics.force_batch_size
            ]
            chunk_tensor = torch.as_tensor(chunk_indices, dtype=torch.long)
            chunk_energies_J, chunk_forces_N, _polarization_residuals = (
                model.physical_energy_force_batch_tensor(
                    positions[chunk_tensor],
                    box_vectors[chunk_tensor],
                )
            )
            energies_J[chunk_tensor] = chunk_energies_J
            forces_N[chunk_tensor] = chunk_forces_N
            force_evaluation_count += 1
            invocation_force_evaluation_count += 1
        invocation_elapsed_s = time.perf_counter() - relaxation_start_time
        elapsed_s = previous_elapsed_s + invocation_elapsed_s
        if iteration == starting_iteration:
            initial_forces_N = forces_N.detach().numpy()
            initial_maximum_forces_N = np.max(
                np.linalg.norm(initial_forces_N, axis=2), axis=1
            )
            initial_positions_m = positions.detach().numpy()
            initial_displacements_m = minimum_image_displacement(
                initial_positions_m[:, pair_i] - initial_positions_m[:, pair_j],
                model.system.box_vectors_m,
            )
            initial_contact_ratios = np.min(
                np.linalg.norm(initial_displacements_m, axis=2)
                / contact_distances_m[None, :],
                axis=1,
            )
            print(
                "[relaxation preflight] "
                f"atoms={positions.shape[1]} chains={chain_count} "
                f"force_batch_size={dynamics.force_batch_size} "
                f"maximum_forces_N={tuple(initial_maximum_forces_N)} "
                f"minimum_contact_ratios={tuple(initial_contact_ratios)}",
                flush=True,
            )
        evaluated_positions_m = positions.detach().numpy()
        evaluated_pair_displacements_m = minimum_image_displacement(
            evaluated_positions_m[:, pair_i] - evaluated_positions_m[:, pair_j],
            model.system.box_vectors_m,
        )
        evaluated_minimum_contact_ratios = np.min(
            np.linalg.norm(evaluated_pair_displacements_m, axis=2)
            / contact_distances_m[None, :],
            axis=1,
        )
        evaluated_energies_J = energies_J.detach().numpy()
        energy_changes_J = np.zeros(chain_count)
        energy_changes_J[has_accepted_energy] = (
            evaluated_energies_J[has_accepted_energy]
            - accepted_energies_J[has_accepted_energy]
        )
        rejected = np.zeros(chain_count, dtype=bool)
        rejected[active_chain_indices] = (
            (
                has_accepted_energy[active_chain_indices]
                & (energy_changes_J[active_chain_indices] > 0.0)
            )
            | (
                evaluated_minimum_contact_ratios[active_chain_indices]
                < model.numerics.minimum_interatomic_contact_ratio
            )
        )
        for chain_index in np.flatnonzero(rejected):
            positions[chain_index] = accepted_positions[chain_index]
            velocities[chain_index] = 0.0
            positive_power_steps[chain_index] = 0
            timesteps_s[chain_index] *= dynamics.initial_relaxation_timestep_decrease
            damping[chain_index] = dynamics.initial_relaxation_initial_damping
            stagnant_iterations[chain_index] = 0

        accepted_chain_indices = active_chain_indices[~rejected[active_chain_indices]]
        accepted_chain_tensor = torch.as_tensor(
            accepted_chain_indices,
            dtype=torch.long,
        )
        accepted_positions[accepted_chain_tensor] = positions[accepted_chain_tensor]
        accepted_energies_J[accepted_chain_indices] = evaluated_energies_J[
            accepted_chain_indices
        ]
        has_accepted_energy[accepted_chain_indices] = True

        force_norms_N = torch.linalg.norm(forces_N, dim=2)
        maximum_forces_N = torch.max(force_norms_N, dim=1).values.detach().numpy()
        last_maximum_forces_N[accepted_chain_indices] = maximum_forces_N[
            accepted_chain_indices
        ]
        converged[accepted_chain_indices] |= (
            maximum_forces_N[accepted_chain_indices]
            <= dynamics.initial_force_tolerance_N
        )
        velocities[torch.as_tensor(converged)] = 0.0
        for chain_index in accepted_chain_indices:
            improvement_limit_N = best_maximum_forces_N[chain_index] * (
                1.0 - dynamics.initial_relaxation_minimum_force_improvement_fraction
            )
            if maximum_forces_N[chain_index] < improvement_limit_N:
                best_maximum_forces_N[chain_index] = maximum_forces_N[chain_index]
                stagnant_iterations[chain_index] = 0
            else:
                stagnant_iterations[chain_index] += 1
            if (
                stagnant_iterations[chain_index]
                >= dynamics.initial_relaxation_maximum_stagnant_iterations
            ):
                velocities[chain_index] = 0.0
                positive_power_steps[chain_index] = 0
                timesteps_s[chain_index] *= (
                    dynamics.initial_relaxation_timestep_decrease
                )
                damping[chain_index] = dynamics.initial_relaxation_initial_damping
                stagnant_iterations[chain_index] = 0
        movable_chain_indices = accepted_chain_indices[
            ~converged[accepted_chain_indices]
        ]
        for chain_index in movable_chain_indices:
            timestep_s = timesteps_s[chain_index]
            velocities[chain_index] += (
                timestep_s * forces_N[chain_index] / atomic_masses_kg[0]
            )
            power_W = float(torch.sum(velocities[chain_index] * forces_N[chain_index]))
            if power_W > 0.0:
                velocity_norm_m_s = torch.linalg.norm(velocities[chain_index])
                force_norm_N = torch.linalg.norm(forces_N[chain_index])
                velocities[chain_index] = (1.0 - damping[chain_index]) * velocities[
                    chain_index
                ] + damping[chain_index] * velocity_norm_m_s * forces_N[
                    chain_index
                ] / force_norm_N
                positive_power_steps[chain_index] += 1
                if (
                    positive_power_steps[chain_index]
                    > dynamics.initial_relaxation_positive_power_steps
                ):
                    timesteps_s[chain_index] = min(
                        timestep_s * dynamics.initial_relaxation_timestep_increase,
                        dynamics.initial_relaxation_maximum_timestep_s,
                    )
                    damping[chain_index] *= dynamics.initial_relaxation_damping_decrease
            else:
                velocities[chain_index] = 0.0
                positive_power_steps[chain_index] = 0
                timesteps_s[chain_index] *= (
                    dynamics.initial_relaxation_timestep_decrease
                )
                damping[chain_index] = dynamics.initial_relaxation_initial_damping
            positions[chain_index] += timesteps_s[chain_index] * velocities[chain_index]
            positions[chain_index] %= torch.diag(box_vectors[chain_index])
        iteration += 1
        if iteration % dynamics.initial_relaxation_progress_stride == 0:
            average_seconds_per_force_call = (
                invocation_elapsed_s / invocation_force_evaluation_count
            )
            print(
                "[relaxation] "
                f"elapsed_s={elapsed_s:.6f} iteration={iteration} "
                f"force_evaluations={force_evaluation_count} "
                f"invocation_force_evaluations={invocation_force_evaluation_count} "
                f"seconds_per_force_call={average_seconds_per_force_call:.6f} "
                f"active_chains={tuple(int(value) for value in np.flatnonzero(~converged))} "
                f"rejected_chains={tuple(int(value) for value in np.flatnonzero(rejected))} "
                f"maximum_forces_N={tuple(maximum_forces_N)} "
                "minimum_contact_ratios="
                f"{tuple(evaluated_minimum_contact_ratios)} "
                f"energy_changes_J={tuple(energy_changes_J)} "
                f"timesteps_s={tuple(timesteps_s)}",
                flush=True,
            )
            write_relaxation_checkpoint(
                "relaxing",
                accepted_positions,
                timesteps_s,
                damping,
                positive_power_steps,
                best_maximum_forces_N,
                accepted_energies_J,
                has_accepted_energy,
                stagnant_iterations,
                converged,
                force_evaluation_count,
                iteration,
                elapsed_s,
            )
        if invocation_elapsed_s >= dynamics.initial_relaxation_maximum_elapsed_s:
            break
    final_elapsed_s = previous_elapsed_s + time.perf_counter() - relaxation_start_time
    final_stage = "relaxed" if np.all(converged) else "relaxing"
    positions = accepted_positions.clone()
    write_relaxation_checkpoint(
        final_stage,
        accepted_positions,
        timesteps_s,
        damping,
        positive_power_steps,
        best_maximum_forces_N,
        accepted_energies_J,
        has_accepted_energy,
        stagnant_iterations,
        converged,
        force_evaluation_count,
        iteration,
        final_elapsed_s,
    )
    if not np.all(converged):
        raise RuntimeError(
            "initial FIRE relaxation exhausted its bounded budget: "
            f"force_evaluations={force_evaluation_count}, "
            f"invocation_force_evaluations={invocation_force_evaluation_count}, "
            f"total_elapsed_s={final_elapsed_s:.12g}, "
            "invocation_elapsed_s="
            f"{final_elapsed_s - previous_elapsed_s:.12g}, "
            f"maximum_forces_N={tuple(last_maximum_forces_N)}, "
            f"best_maximum_forces_N={tuple(best_maximum_forces_N)}, "
            f"stagnant_iterations={tuple(stagnant_iterations)}, "
            f"checkpoint={checkpoint_path}"
        )
    final_positions_m = positions.detach().numpy()
    final_pair_displacements_m = minimum_image_displacement(
        final_positions_m[:, pair_i] - final_positions_m[:, pair_j],
        model.system.box_vectors_m,
    )
    final_minimum_contact_ratios = np.min(
        np.linalg.norm(final_pair_displacements_m, axis=2)
        / contact_distances_m[None, :],
        axis=1,
    )
    if np.any(
        final_minimum_contact_ratios < model.numerics.minimum_interatomic_contact_ratio
    ):
        raise RuntimeError(
            "relaxed configuration violates the minimum LJ contact ratio: "
            f"ratios={tuple(final_minimum_contact_ratios)}"
        )
    return final_positions_m, best_maximum_forces_N, force_evaluation_count


PhaseSpaceObservable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _hamiltonian_physical_forces(
    positions_m: torch.Tensor,
    box_vectors_m: torch.Tensor,
    model: AnalyticalPeriodicInteratomicModel,
) -> torch.Tensor:
    if positions_m.ndim == 2:
        def potential_energy_J(evaluated_positions_m: torch.Tensor) -> torch.Tensor:
            return model._energy_tensor(evaluated_positions_m, box_vectors_m)

    elif positions_m.ndim == 3:
        def potential_energy_J(evaluated_positions_m: torch.Tensor) -> torch.Tensor:
            (
                fixed_energies_J,
                ion_ion_energies_J,
                ion_neutral_energies_J,
                _polarization_residuals_V_m,
            ) = model._energy_components_batch_tensor(
                evaluated_positions_m,
                box_vectors_m,
            )
            return torch.sum(
                fixed_energies_J
                + ion_ion_energies_J
                + ion_neutral_energies_J
            )

    else:
        raise ValueError("Hamiltonian phase-space positions must be rank two or three")
    return -torch.func.grad(potential_energy_J)(positions_m)


def _hamiltonian_liouville_action(
    observable: PhaseSpaceObservable,
    positions_m: torch.Tensor,
    momenta_kg_m_s: torch.Tensor,
    physical_forces_N: torch.Tensor,
    model: AnalyticalPeriodicInteratomicModel,
) -> torch.Tensor:
    if positions_m.ndim == 2:
        atomic_masses_kg = torch.as_tensor(
            model.system.masses_kg[:, None],
            dtype=TORCH_DTYPE,
        )
    elif positions_m.ndim == 3:
        atomic_masses_kg = torch.as_tensor(
            model.system.masses_kg[None, :, None],
            dtype=TORCH_DTYPE,
        )
    else:
        raise ValueError("Hamiltonian phase-space positions must be rank two or three")
    if physical_forces_N.shape != positions_m.shape:
        raise ValueError("Hamiltonian physical forces must match the position shape")

    velocities_m_s = momenta_kg_m_s / atomic_masses_kg
    _observable_value, action_value = torch.func.jvp(
        observable,
        (positions_m, momenta_kg_m_s),
        (velocities_m_s, physical_forces_N),
    )
    return action_value


def _negative_hamiltonian_liouville_squared_action(
    observable: PhaseSpaceObservable,
    positions_m: torch.Tensor,
    box_vectors_m: torch.Tensor,
    momenta_kg_m_s: torch.Tensor,
    physical_forces_N: torch.Tensor,
    model: AnalyticalPeriodicInteratomicModel,
) -> torch.Tensor:
    if positions_m.ndim != 3:
        raise ValueError("squared Liouville action requires a sample batch")
    atomic_masses_kg = torch.as_tensor(
        model.system.masses_kg[None, :, None],
        dtype=TORCH_DTYPE,
    )
    physical_velocities_m_s = momenta_kg_m_s / atomic_masses_kg
    maximum_speed_m_s = torch.amax(
        torch.linalg.vector_norm(physical_velocities_m_s, dim=2),
        dim=1,
    )
    if bool(torch.any(maximum_speed_m_s <= 0.0)):
        raise ValueError("squared Liouville action requires nonzero momentum")
    difference_times_s = model.numerics.force_difference_step_m / maximum_speed_m_s

    def centered_second_action(
        evaluated_difference_times_s: torch.Tensor,
    ) -> torch.Tensor:
        position_displacements_m = (
            evaluated_difference_times_s[:, None, None]
            * physical_velocities_m_s
        )
        momentum_displacements_kg_m_s = (
            evaluated_difference_times_s[:, None, None] * physical_forces_N
        )
        positive_positions_m = positions_m + position_displacements_m
        negative_positions_m = positions_m - position_displacements_m
        positive_momenta_kg_m_s = (
            momenta_kg_m_s + momentum_displacements_kg_m_s
        )
        negative_momenta_kg_m_s = (
            momenta_kg_m_s - momentum_displacements_kg_m_s
        )
        positive_forces_N = _hamiltonian_physical_forces(
            positions_m=positive_positions_m,
            box_vectors_m=box_vectors_m,
            model=model,
        )
        negative_forces_N = _hamiltonian_physical_forces(
            positions_m=negative_positions_m,
            box_vectors_m=box_vectors_m,
            model=model,
        )
        positive_first_action = _hamiltonian_liouville_action(
            observable=observable,
            positions_m=positive_positions_m,
            momenta_kg_m_s=positive_momenta_kg_m_s,
            physical_forces_N=positive_forces_N,
            model=model,
        )
        negative_first_action = _hamiltonian_liouville_action(
            observable=observable,
            positions_m=negative_positions_m,
            momenta_kg_m_s=negative_momenta_kg_m_s,
            physical_forces_N=negative_forces_N,
            model=model,
        )
        action_denominator_s = evaluated_difference_times_s
        while action_denominator_s.ndim < positive_first_action.ndim:
            action_denominator_s = action_denominator_s[:, None]
        return (positive_first_action - negative_first_action) / (
            2.0 * action_denominator_s
        )

    coarse_second_action = centered_second_action(difference_times_s)
    fine_second_action = centered_second_action(0.5 * difference_times_s)
    richardson_second_action = (
        4.0 * fine_second_action - coarse_second_action
    ) / 3.0
    second_action_scale = max(
        float(torch.max(torch.abs(richardson_second_action))),
        np.finfo(float).tiny,
    )
    relative_difference_error = float(
        torch.max(torch.abs(fine_second_action - coarse_second_action))
        / second_action_scale
    )
    difference_tolerance = math.sqrt(
        model.numerics.force_consistency_relative_tolerance
    )
    if relative_difference_error > difference_tolerance:
        raise RuntimeError(
            "squared Liouville directional derivative did not converge: "
            f"relative_error={relative_difference_error:.12g}, "
            f"tolerance={difference_tolerance:.12g}"
        )
    return -richardson_second_action


def _phase_space_channel_metadata(
    system: MolecularSystem,
) -> tuple[tuple[str, bool], ...]:
    species_names = tuple(sorted(set(system.molecule_species_names)))
    reference_species_name = species_names[-1]
    metadata: list[tuple[str, bool]] = [("charge_collective", False)]
    molecule_indices_by_species = {
        species_name: tuple(
            molecule_index
            for molecule_index, observed_species_name in enumerate(
                system.molecule_species_names
            )
            if observed_species_name == species_name
        )
        for species_name in species_names
    }
    for species_name in species_names:
        metadata.append(
            (
                f"molecular_species={species_name}",
                species_name != reference_species_name,
            )
        )
        molecule_indices = molecule_indices_by_species[species_name]
        site_counts = {
            len(system.molecule_atom_indices[molecule_index])
            for molecule_index in molecule_indices
        }
        if len(site_counts) != 1:
            raise ValueError(
                f"species {species_name} has inconsistent molecular topology"
            )
        site_count = next(iter(site_counts))
        if site_count == 1:
            continue
        for site_index in range(site_count):
            metadata.append(
                (
                    f"atomic_site[species={species_name},site={site_index}]",
                    site_index + 1 < site_count,
                )
            )
    return tuple(metadata)


def _momentum_primitive_descriptor(
    system: MolecularSystem,
    primitive_index: int,
) -> tuple[int, int, int, int, int, bool]:
    if primitive_index < 0:
        raise ValueError("momentum primitive index must be nonnegative")
    channel_metadata = _phase_space_channel_metadata(system)
    primitive_families: list[tuple[int, int]] = [(0, 0)]
    primitive_families.extend(
        (channel_index, -1)
        for channel_index, (_channel_label, permits_uniform_mode) in enumerate(
            channel_metadata
        )
        if permits_uniform_mode
    )
    primitive_families.extend(
        (momentum_channel_index, density_channel_index)
        for momentum_channel_index in range(len(channel_metadata))
        for density_channel_index in range(len(channel_metadata))
        if (momentum_channel_index, density_channel_index) != (0, 0)
    )
    observed_primitive_count = 0
    shell_index = 0
    while True:
        maximum_family_index = min(shell_index, len(primitive_families) - 1)
        for family_index in range(maximum_family_index + 1):
            momentum_channel_index, density_channel_index = primitive_families[
                family_index
            ]
            if density_channel_index < 0 and shell_index != family_index:
                continue
            if density_channel_index < 0:
                for momentum_axis in range(CARTESIAN_DIMENSION):
                    if observed_primitive_count == primitive_index:
                        return (
                            momentum_channel_index,
                            -1,
                            0,
                            -1,
                            momentum_axis,
                            False,
                        )
                    observed_primitive_count += 1
                continue
            harmonic_index = shell_index - family_index + 1
            for transverse_mode in (False, True):
                for wave_axis in range(CARTESIAN_DIMENSION):
                    momentum_axes = (wave_axis,)
                    if transverse_mode:
                        momentum_axes = tuple(
                            axis
                            for axis in range(CARTESIAN_DIMENSION)
                            if axis != wave_axis
                        )
                    for momentum_axis in momentum_axes:
                        for sine_phase in (False, True):
                            if observed_primitive_count == primitive_index:
                                return (
                                    momentum_channel_index,
                                    density_channel_index,
                                    harmonic_index,
                                    wave_axis,
                                    momentum_axis,
                                    sine_phase,
                                )
                            observed_primitive_count += 1
        shell_index += 1


def _density_primitive_descriptor(
    system: MolecularSystem,
    primitive_index: int,
) -> tuple[int, int, int, int, bool]:
    if primitive_index < 0:
        raise ValueError("density primitive index must be nonnegative")
    channel_count = len(_phase_space_channel_metadata(system))
    primitive_families = tuple(
        combinations_with_replacement(range(channel_count), 2)
    )
    observed_primitive_count = 0
    shell_index = 0
    while True:
        maximum_family_index = min(shell_index, len(primitive_families) - 1)
        for family_index in range(maximum_family_index + 1):
            first_channel_index, second_channel_index = primitive_families[
                family_index
            ]
            harmonic_index = shell_index - family_index + 1
            for wave_axis in range(CARTESIAN_DIMENSION):
                for sine_phase in (False, True):
                    if sine_phase and first_channel_index == second_channel_index:
                        continue
                    if observed_primitive_count == primitive_index:
                        return (
                            first_channel_index,
                            second_channel_index,
                            harmonic_index,
                            wave_axis,
                            sine_phase,
                        )
                    observed_primitive_count += 1
        shell_index += 1


@cache
def _phase_space_basis_monomials(
    monomial_count: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if monomial_count < 0:
        raise ValueError("phase-space monomial count must be nonnegative")
    monomials: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    additional_factor_grade = 2 * CARTESIAN_DIMENSION * CARTESIAN_DIMENSION
    total_grade = 1
    while len(monomials) < monomial_count:
        maximum_factor_count = 1 + (total_grade - 1) // (
            additional_factor_grade + 1
        )
        for total_factor_count in range(1, maximum_factor_count + 1):
            primitive_grade = total_grade - additional_factor_grade * (
                total_factor_count - 1
            )
            if primitive_grade < total_factor_count:
                continue
            for momentum_factor_count in range(1, total_factor_count + 1, 2):
                density_factor_count = total_factor_count - momentum_factor_count
                maximum_momentum_grade = primitive_grade - density_factor_count
                for momentum_grade in range(
                    momentum_factor_count,
                    maximum_momentum_grade + 1,
                ):
                    density_grade = primitive_grade - momentum_grade
                    momentum_multisets = combinations_with_replacement(
                        range(momentum_grade), momentum_factor_count
                    )
                    for momentum_indices in momentum_multisets:
                        if (
                            sum(index + 1 for index in momentum_indices)
                            != momentum_grade
                        ):
                            continue
                        density_multisets = combinations_with_replacement(
                            range(density_grade), density_factor_count
                        )
                        for density_indices in density_multisets:
                            if (
                                sum(index + 1 for index in density_indices)
                                != density_grade
                            ):
                                continue
                            monomials.append(
                                (tuple(momentum_indices), tuple(density_indices))
                            )
                            if len(monomials) == monomial_count:
                                return tuple(monomials)
        total_grade += 1
    return tuple(monomials)


def _normalized_probabilists_hermite(
    standardized_value: torch.Tensor,
    polynomial_degree: int,
) -> torch.Tensor:
    if polynomial_degree < 0:
        raise ValueError("Hermite polynomial degree must be nonnegative")
    if polynomial_degree == 0:
        return torch.ones_like(standardized_value)
    preceding_value = torch.ones_like(standardized_value)
    current_value = standardized_value
    for recurrence_degree in range(1, polynomial_degree):
        next_value = (
            standardized_value * current_value
            - recurrence_degree * preceding_value
        )
        preceding_value = current_value
        current_value = next_value
    return current_value / math.sqrt(math.factorial(polynomial_degree))


def _momentum_primitive_label(
    system: MolecularSystem,
    primitive_index: int,
) -> str:
    (
        momentum_channel_index,
        density_channel_index,
        harmonic_index,
        wave_axis,
        momentum_axis,
        sine_phase,
    ) = _momentum_primitive_descriptor(system, primitive_index)
    channel_metadata = _phase_space_channel_metadata(system)
    momentum_channel_label = channel_metadata[momentum_channel_index][0]
    if harmonic_index == 0:
        return (
            f"P[{momentum_channel_label},uniform,momentum_axis={momentum_axis}]"
        )
    density_channel_label = channel_metadata[density_channel_index][0]
    phase_label = "cos"
    if sine_phase:
        phase_label = "sin"
    return (
        f"P_k_rho_-k[momentum={momentum_channel_label},"
        f"density={density_channel_label},harmonic={harmonic_index},"
        f"wave_axis={wave_axis},"
        f"momentum_axis={momentum_axis},phase={phase_label}]"
    )


def _density_primitive_label(
    system: MolecularSystem,
    primitive_index: int,
) -> str:
    (
        first_channel_index,
        second_channel_index,
        harmonic_index,
        wave_axis,
        sine_phase,
    ) = (
        _density_primitive_descriptor(system, primitive_index)
    )
    channel_metadata = _phase_space_channel_metadata(system)
    first_channel_label = channel_metadata[first_channel_index][0]
    second_channel_label = channel_metadata[second_channel_index][0]
    phase_label = "cos"
    if sine_phase:
        phase_label = "sin"
    return (
        f"rho_k_rho_-k[first={first_channel_label},second={second_channel_label},"
        f"harmonic={harmonic_index},wave_axis={wave_axis},"
        f"phase={phase_label}]"
    )


def _phase_space_basis_labels(
    system: MolecularSystem,
    basis_level: int,
) -> tuple[str, ...]:
    if basis_level <= 0:
        raise ValueError("Hamiltonian phase-space basis level must be positive")
    labels = [
        f"permanent_atomic_charge_current[{axis}]"
        for axis in range(CARTESIAN_DIMENSION)
    ]
    for momentum_indices, density_indices in _phase_space_basis_monomials(
        basis_level - 1
    ):
        primitive_labels = [
            "H_"
            f"{momentum_indices.count(primitive_index)}["
            f"{_momentum_primitive_label(system, primitive_index)}]"
            for primitive_index in sorted(set(momentum_indices))
        ]
        primitive_labels.extend(
            _density_primitive_label(system, primitive_index)
            for primitive_index in density_indices
        )
        labels.append("phase_space_monomial[" + " * ".join(primitive_labels) + "]")
    return tuple(labels)


def _phase_space_trial_basis(
    system: MolecularSystem,
    basis_level: int,
    temperature_K: float,
    box_vectors_m: torch.Tensor,
) -> tuple[PhaseSpaceObservable, tuple[str, ...]]:
    if temperature_K <= 0.0:
        raise ValueError("Hamiltonian phase-space basis temperature must be positive")
    basis_labels = _phase_space_basis_labels(system, basis_level)
    basis_monomials = _phase_space_basis_monomials(basis_level - 1)
    molecule_index_tensors = tuple(
        torch.as_tensor(atom_indices, dtype=torch.int64)
        for atom_indices in system.molecule_atom_indices
    )
    molecule_mass_tensors_kg = tuple(
        torch.as_tensor(system.masses_kg[atom_indices], dtype=TORCH_DTYPE)
        for atom_indices in system.molecule_atom_indices
    )
    molecule_masses_kg = torch.as_tensor(
        [
            float(np.sum(system.masses_kg[atom_indices]))
            for atom_indices in system.molecule_atom_indices
        ],
        dtype=TORCH_DTYPE,
    )
    molecule_charges_C = torch.as_tensor(
        [
            float(np.sum(system.charges_C[atom_indices]))
            for atom_indices in system.molecule_atom_indices
        ],
        dtype=TORCH_DTYPE,
    )
    absolute_charge_sum_C = float(torch.sum(torch.abs(molecule_charges_C)))
    charge_neutrality_tolerance_C = (
        math.sqrt(np.finfo(float).eps) * absolute_charge_sum_C
    )
    total_molecular_charge_C = float(torch.sum(molecule_charges_C))
    if abs(total_molecular_charge_C) > charge_neutrality_tolerance_C:
        raise ValueError(
            "molecular Helfand current requires a charge-neutral periodic cell: "
            f"net_charge_C={total_molecular_charge_C:.12g}, "
            f"tolerance_C={charge_neutrality_tolerance_C:.12g}"
        )
    charged_molecule_indices = torch.nonzero(
        molecule_charges_C != 0.0,
        as_tuple=False,
    ).flatten()
    if charged_molecule_indices.numel() == 0:
        raise ValueError("molecular Helfand current is zero for every molecule")
    species_names = tuple(sorted(set(system.molecule_species_names)))
    molecule_indices_by_species = {
        species_name: tuple(
            molecule_index
            for molecule_index, observed_species_name in enumerate(
                system.molecule_species_names
            )
            if observed_species_name == species_name
        )
        for species_name in species_names
    }
    channel_sources: list[tuple[bool, torch.Tensor, torch.Tensor]] = [
        (
            True,
            charged_molecule_indices,
            molecule_charges_C[charged_molecule_indices] / E_CHARGE,
        )
    ]
    for species_name in species_names:
        molecule_indices = molecule_indices_by_species[species_name]
        channel_sources.append(
            (
                True,
                torch.as_tensor(molecule_indices, dtype=torch.int64),
                torch.ones(len(molecule_indices), dtype=TORCH_DTYPE),
            )
        )
        site_count = len(system.molecule_atom_indices[molecule_indices[0]])
        if site_count == 1:
            continue
        for site_index in range(site_count):
            atomic_indices = torch.as_tensor(
                [
                    int(system.molecule_atom_indices[molecule_index][site_index])
                    for molecule_index in molecule_indices
                ],
                dtype=torch.int64,
            )
            channel_sources.append(
                (
                    False,
                    atomic_indices,
                    torch.ones(len(molecule_indices), dtype=TORCH_DTYPE),
                )
            )
    if len(channel_sources) != len(_phase_space_channel_metadata(system)):
        raise RuntimeError("phase-space channel construction is inconsistent")
    atomic_masses_kg = torch.as_tensor(system.masses_kg, dtype=TORCH_DTYPE)
    total_mass_kg = torch.sum(atomic_masses_kg)
    atom_molecule_indices = torch.as_tensor(system.molecule_index, dtype=torch.int64)
    atom_molecule_masses_kg = molecule_masses_kg[atom_molecule_indices]
    atom_mass_fractions = atomic_masses_kg / atom_molecule_masses_kg
    thermal_energy_J = K_B * temperature_K
    maximum_momentum_primitive_index = max(
        (
            primitive_index
            for momentum_indices, _density_indices in basis_monomials
            for primitive_index in momentum_indices
        ),
        default=-1,
    )
    maximum_density_primitive_index = max(
        (
            primitive_index
            for _momentum_indices, density_indices in basis_monomials
            for primitive_index in density_indices
        ),
        default=-1,
    )
    momentum_primitive_descriptors = tuple(
        _momentum_primitive_descriptor(system, primitive_index)
        for primitive_index in range(maximum_momentum_primitive_index + 1)
    )
    density_primitive_descriptors = tuple(
        _density_primitive_descriptor(system, primitive_index)
        for primitive_index in range(maximum_density_primitive_index + 1)
    )

    def phase_space_basis(
        evaluated_positions_m: torch.Tensor,
        evaluated_momenta_kg_m_s: torch.Tensor,
    ) -> torch.Tensor:
        total_momentum_kg_m_s = torch.sum(evaluated_momenta_kg_m_s, dim=-2)
        relative_momenta_kg_m_s = evaluated_momenta_kg_m_s - (
            atomic_masses_kg / total_mass_kg
        )[..., None] * total_momentum_kg_m_s[..., None, :]
        molecular_centers_m: list[torch.Tensor] = []
        molecular_momenta_kg_m_s: list[torch.Tensor] = []
        for atom_indices, atom_masses_kg, molecule_mass_kg in zip(
            molecule_index_tensors,
            molecule_mass_tensors_kg,
            molecule_masses_kg,
            strict=True,
        ):
            anchor_position_m = evaluated_positions_m[..., atom_indices[0], :]
            unwrapped_positions_m = anchor_position_m[..., None, :] + (
                _torch_minimum_image(
                    evaluated_positions_m[..., atom_indices, :]
                    - anchor_position_m[..., None, :],
                    box_vectors_m,
                )
            )
            molecular_centers_m.append(
                torch.sum(
                    atom_masses_kg[..., None] * unwrapped_positions_m,
                    dim=-2,
                )
                / molecule_mass_kg
            )
            molecular_momenta_kg_m_s.append(
                torch.sum(relative_momenta_kg_m_s[..., atom_indices, :], dim=-2)
            )
        centers_m = torch.stack(molecular_centers_m, dim=-2)
        molecular_momenta = torch.stack(molecular_momenta_kg_m_s, dim=-2)
        standardized_molecular_momenta = molecular_momenta / torch.sqrt(
            molecule_masses_kg[..., None] * thermal_energy_J
        )
        parent_molecular_momenta = molecular_momenta[..., atom_molecule_indices, :]
        internal_atomic_momenta = relative_momenta_kg_m_s - (
            atom_mass_fractions[..., None] * parent_molecular_momenta
        )
        standardized_internal_momenta = internal_atomic_momenta / torch.sqrt(
            atomic_masses_kg[..., None] * thermal_energy_J
        )
        inverse_box_m = torch.linalg.inv(box_vectors_m)
        fractional_centers = torch.matmul(
            centers_m.unsqueeze(-2), inverse_box_m.unsqueeze(-3)
        ).squeeze(-2)
        fractional_atomic_positions = torch.matmul(
            evaluated_positions_m.unsqueeze(-2), inverse_box_m.unsqueeze(-3)
        ).squeeze(-2)
        permanent_atomic_current_C_m_s = torch.einsum(
            "n,...na->...a",
            torch.as_tensor(system.charges_C, dtype=TORCH_DTYPE),
            relative_momenta_kg_m_s / atomic_masses_kg[..., None],
        )
        channel_fractional_positions: list[torch.Tensor] = []
        channel_standardized_momenta: list[torch.Tensor] = []
        channel_weights: list[torch.Tensor] = []
        channel_normalizations: list[torch.Tensor] = []
        for molecular_channel, particle_indices, particle_weights in channel_sources:
            if molecular_channel:
                channel_fractional_positions.append(
                    fractional_centers[..., particle_indices, :]
                )
                channel_standardized_momenta.append(
                    standardized_molecular_momenta[..., particle_indices, :]
                )
                channel_weights.append(particle_weights)
                channel_normalizations.append(
                    torch.linalg.vector_norm(particle_weights)
                )
                continue
            channel_fractional_positions.append(
                fractional_atomic_positions[..., particle_indices, :]
            )
            channel_standardized_momenta.append(
                standardized_internal_momenta[..., particle_indices, :]
            )
            channel_weights.append(particle_weights)
            channel_normalizations.append(torch.linalg.vector_norm(particle_weights))
        momentum_primitive_values: list[torch.Tensor] = []
        for momentum_descriptor in momentum_primitive_descriptors:
            (
                momentum_channel_index,
                density_channel_index,
                harmonic_index,
                wave_axis,
                momentum_axis,
                sine_phase,
            ) = momentum_descriptor
            standardized_momentum_component = channel_standardized_momenta[
                momentum_channel_index
            ][..., momentum_axis]
            phase_factor = torch.ones_like(standardized_momentum_component)
            if harmonic_index != 0:
                phase = (
                    2.0
                    * math.pi
                    * harmonic_index
                    * (
                        channel_fractional_positions[momentum_channel_index][
                            ..., :, None, wave_axis
                        ]
                        - channel_fractional_positions[density_channel_index][
                            ..., None, :, wave_axis
                        ]
                    )
                )
                phase_factor = torch.cos(phase)
                if sine_phase:
                    phase_factor = torch.sin(phase)
                standardized_momentum_component = (
                    standardized_momentum_component[..., :, None]
                )
                pair_weights = (
                    channel_weights[momentum_channel_index][:, None]
                    * channel_weights[density_channel_index][None, :]
                )
                momentum_primitive_values.append(
                    torch.sum(
                        pair_weights
                        * phase_factor
                        * standardized_momentum_component,
                        dim=(-2, -1),
                    )
                    / (
                        channel_normalizations[momentum_channel_index]
                        * channel_normalizations[density_channel_index]
                    )
                )
                continue
            momentum_primitive_values.append(
                torch.sum(
                    channel_weights[momentum_channel_index]
                    * phase_factor
                    * standardized_momentum_component,
                    dim=-1,
                )
                / channel_normalizations[momentum_channel_index]
            )
        density_primitive_values: list[torch.Tensor] = []
        for density_descriptor in density_primitive_descriptors:
            (
                first_channel_index,
                second_channel_index,
                harmonic_index,
                wave_axis,
                sine_phase,
            ) = density_descriptor
            phase = (
                2.0
                * math.pi
                * harmonic_index
                * (
                    channel_fractional_positions[first_channel_index][
                        ..., :, None, wave_axis
                    ]
                    - channel_fractional_positions[second_channel_index][
                        ..., None, :, wave_axis
                    ]
                )
            )
            phase_factor = torch.cos(phase)
            if sine_phase:
                phase_factor = torch.sin(phase)
            pair_weights = (
                channel_weights[first_channel_index][:, None]
                * channel_weights[second_channel_index][None, :]
            )
            density_primitive_values.append(
                torch.sum(
                    pair_weights * phase_factor,
                    dim=(-2, -1),
                )
                / (
                    channel_normalizations[first_channel_index]
                    * channel_normalizations[second_channel_index]
                )
            )
        basis_values = [
            permanent_atomic_current_C_m_s[..., axis]
            for axis in range(CARTESIAN_DIMENSION)
        ]
        for momentum_indices, density_indices in basis_monomials:
            monomial_value = torch.ones_like(
                permanent_atomic_current_C_m_s[..., 0]
            )
            for primitive_index in sorted(set(momentum_indices)):
                monomial_value = monomial_value * _normalized_probabilists_hermite(
                    standardized_value=momentum_primitive_values[primitive_index],
                    polynomial_degree=momentum_indices.count(primitive_index),
                )
            for primitive_index in density_indices:
                monomial_value = (
                    monomial_value * density_primitive_values[primitive_index]
                )
            basis_values.append(monomial_value)
        return torch.stack(basis_values, dim=-1)

    return phase_space_basis, basis_labels


def _evaluate_phase_space_basis_samples(
    phase_space_samples: HamiltonianPhaseSpaceSamples,
    model: AnalyticalPeriodicInteratomicModel,
    basis_level: int,
    temperature_K: float,
    operator_batch_size: int,
) -> HamiltonianBasisEvaluation:
    if basis_level <= 0:
        raise ValueError("Hamiltonian phase-space basis level must be positive")
    if operator_batch_size <= 0:
        raise ValueError("Hamiltonian operator batch size must be positive")
    sample_count = phase_space_samples.configurations_m.shape[0]
    expected_particle_shape = (
        sample_count,
        model.system.positions_m.shape[0],
        CARTESIAN_DIMENSION,
    )
    if phase_space_samples.configurations_m.shape != expected_particle_shape:
        raise ValueError("phase-space configurations have incompatible shape")
    if phase_space_samples.momenta_kg_m_s.shape != expected_particle_shape:
        raise ValueError("phase-space momenta have incompatible shape")
    if phase_space_samples.forces_N.shape != expected_particle_shape:
        raise ValueError("phase-space forces have incompatible shape")
    if phase_space_samples.box_vectors_m.shape != (
        sample_count,
        CARTESIAN_DIMENSION,
        CARTESIAN_DIMENSION,
    ):
        raise ValueError("phase-space boxes have incompatible shape")
    if phase_space_samples.chain_indices.shape != (sample_count,):
        raise ValueError("phase-space chain indices have incompatible shape")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            phase_space_samples.configurations_m,
            phase_space_samples.momenta_kg_m_s,
            phase_space_samples.forces_N,
            phase_space_samples.box_vectors_m,
        )
    ):
        raise FloatingPointError("phase-space samples contain nonfinite values")

    basis_value_rows: list[Array] = []
    generator_value_rows: list[Array] = []
    negative_generator_squared_value_rows: list[Array] = []
    current_value_rows: list[Array] = []
    operator_start_time = time.perf_counter()
    for batch_start in range(0, sample_count, operator_batch_size):
        batch_stop = min(batch_start + operator_batch_size, sample_count)
        positions_m = torch.as_tensor(
            phase_space_samples.configurations_m[batch_start:batch_stop],
            dtype=TORCH_DTYPE,
        )
        momenta_kg_m_s = torch.as_tensor(
            phase_space_samples.momenta_kg_m_s[batch_start:batch_stop],
            dtype=TORCH_DTYPE,
        )
        physical_forces_N = torch.as_tensor(
            phase_space_samples.forces_N[batch_start:batch_stop],
            dtype=TORCH_DTYPE,
        )
        box_vectors_m = torch.as_tensor(
            phase_space_samples.box_vectors_m[batch_start:batch_stop],
            dtype=TORCH_DTYPE,
        )
        phase_space_basis, basis_labels = _phase_space_trial_basis(
            system=model.system,
            basis_level=basis_level,
            temperature_K=temperature_K,
            box_vectors_m=box_vectors_m,
        )

        atomic_charges_C = torch.as_tensor(
            model.system.charges_C,
            dtype=TORCH_DTYPE,
        )
        atomic_masses_kg = torch.as_tensor(
            model.system.masses_kg[None, :, None],
            dtype=TORCH_DTYPE,
        )

        def complete_charge_polarization_C_m(
            evaluated_positions_m: torch.Tensor,
            _evaluated_momenta_kg_m_s: torch.Tensor,
        ) -> torch.Tensor:
            permanent_polarization_C_m = torch.einsum(
                "n,bna->ba",
                atomic_charges_C,
                evaluated_positions_m,
            )
            induced_polarization_C_m = model._induced_polarization_batch_tensor(
                positions_batch_m=evaluated_positions_m,
                box_vectors_batch_m=box_vectors_m,
            )
            return permanent_polarization_C_m + induced_polarization_C_m

        sample_basis_values = phase_space_basis(positions_m, momenta_kg_m_s)
        physical_velocities_m_s = momenta_kg_m_s / atomic_masses_kg
        maximum_speed_m_s = torch.amax(
            torch.linalg.vector_norm(physical_velocities_m_s, dim=2),
            dim=1,
        )
        if bool(torch.any(maximum_speed_m_s <= 0.0)):
            raise ValueError("complete polarization current requires nonzero momentum")
        polarization_difference_times_s = (
            model.numerics.force_difference_step_m / maximum_speed_m_s
        )

        def centered_polarization_derivative(
            difference_times_s: torch.Tensor,
        ) -> torch.Tensor:
            displacements_m = (
                difference_times_s[:, None, None] * physical_velocities_m_s
            )
            positive_polarization_C_m = complete_charge_polarization_C_m(
                positions_m + displacements_m,
                momenta_kg_m_s,
            )
            negative_polarization_C_m = complete_charge_polarization_C_m(
                positions_m - displacements_m,
                momenta_kg_m_s,
            )
            return (
                positive_polarization_C_m - negative_polarization_C_m
            ) / (2.0 * difference_times_s[:, None])

        coarse_polarization_current_C_m_s = centered_polarization_derivative(
            polarization_difference_times_s
        )
        fine_polarization_current_C_m_s = centered_polarization_derivative(
            0.5 * polarization_difference_times_s
        )
        sample_current_values = (
            4.0 * fine_polarization_current_C_m_s
            - coarse_polarization_current_C_m_s
        ) / 3.0
        polarization_current_error_scale_C_m_s = max(
            float(torch.max(torch.abs(sample_current_values))),
            np.finfo(float).tiny,
        )
        polarization_current_relative_error = float(
            torch.max(
                torch.abs(
                    fine_polarization_current_C_m_s
                    - coarse_polarization_current_C_m_s
                )
            )
            / polarization_current_error_scale_C_m_s
        )
        polarization_current_tolerance = math.sqrt(
            model.numerics.force_consistency_relative_tolerance
        )
        if polarization_current_relative_error > polarization_current_tolerance:
            raise RuntimeError(
                "complete polarization-current derivative did not converge: "
                f"relative_error={polarization_current_relative_error:.12g}, "
                f"tolerance={polarization_current_tolerance:.12g}"
            )
        sample_generator_values = _hamiltonian_liouville_action(
            observable=phase_space_basis,
            positions_m=positions_m,
            momenta_kg_m_s=momenta_kg_m_s,
            physical_forces_N=physical_forces_N,
            model=model,
        )
        sample_negative_generator_squared_values = (
            _negative_hamiltonian_liouville_squared_action(
                observable=phase_space_basis,
                positions_m=positions_m,
                box_vectors_m=box_vectors_m,
                momenta_kg_m_s=momenta_kg_m_s,
                physical_forces_N=physical_forces_N,
                model=model,
            )
        )
        basis_value_rows.append(
            sample_basis_values.detach().numpy()
        )
        generator_value_rows.append(
            sample_generator_values.detach().numpy()
        )
        negative_generator_squared_value_rows.append(
            sample_negative_generator_squared_values.detach().numpy()
        )
        current_value_rows.append(
            sample_current_values.detach().numpy()
        )
        if sample_count > operator_batch_size:
            print(
                "[Hamiltonian Liouville operator] "
                f"basis_level={basis_level} "
                f"completed_samples={batch_stop}/{sample_count} "
                f"elapsed_s={time.perf_counter() - operator_start_time:.3f}",
                flush=True,
            )
    return HamiltonianBasisEvaluation(
        basis_level=basis_level,
        basis_labels=basis_labels,
        basis_values=np.concatenate(basis_value_rows, axis=0),
        generator_values=np.concatenate(generator_value_rows, axis=0),
        negative_generator_squared_values=np.concatenate(
            negative_generator_squared_value_rows,
            axis=0,
        ),
        current_values=np.concatenate(current_value_rows, axis=0),
        chain_indices=phase_space_samples.chain_indices.copy(),
    )


def _zero_coordinate_limit_interval(
    positive_coordinates: Array,
    lower_values: Array,
    upper_values: Array,
) -> tuple[float, float, float, float]:
    coordinates = np.asarray(positive_coordinates, dtype=float)
    lower_bounds = np.asarray(lower_values, dtype=float)
    upper_bounds = np.asarray(upper_values, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 3:
        raise ValueError("limit continuation requires at least three coordinates")
    if lower_bounds.shape != coordinates.shape or upper_bounds.shape != coordinates.shape:
        raise ValueError("limit continuation bounds have incompatible shape")
    if np.any(coordinates <= 0.0) or np.any(upper_bounds < lower_bounds):
        raise ValueError("limit continuation inputs are outside their physical domain")
    normalized_coordinates = coordinates / float(np.max(coordinates))
    linear_design = np.column_stack(
        (np.ones(coordinates.size), normalized_coordinates)
    )
    intercept_weights = np.linalg.pinv(linear_design)[0]
    interval_midpoints = 0.5 * (lower_bounds + upper_bounds)
    interval_half_widths = 0.5 * (upper_bounds - lower_bounds)
    linear_limit = float(intercept_weights @ interval_midpoints)
    propagated_half_width = float(
        np.sum(np.abs(intercept_weights) * interval_half_widths)
    )
    quadratic_design = np.column_stack(
        (
            np.ones(coordinates.size),
            normalized_coordinates,
            normalized_coordinates**2,
        )
    )
    quadratic_limit = float(
        np.linalg.pinv(quadratic_design)[0] @ interval_midpoints
    )
    continuation_error = abs(quadratic_limit - linear_limit)
    total_half_width = propagated_half_width + continuation_error
    return (
        linear_limit,
        linear_limit - total_half_width,
        linear_limit + total_half_width,
        continuation_error,
    )


def _solve_hamiltonian_resolvent(
    basis_evaluation: HamiltonianBasisEvaluation,
    basis_size: int,
    temperature_K: float,
    volume_m3: float,
    independent_chain_count: int,
    eta_values_s_inv: tuple[float, ...],
    numerics: NumericalSettings,
) -> HamiltonianResolventEstimate:
    if len(eta_values_s_inv) < 3:
        raise ValueError(
            "Hamiltonian regularization sequence requires at least three eta values"
        )
    if independent_chain_count < 4 or independent_chain_count % 2 != 0:
        raise ValueError(
            "Hamiltonian cross-fitting requires an even number of at least four "
            "independent chains"
        )
    if np.any(basis_evaluation.chain_indices < 0) or np.any(
        basis_evaluation.chain_indices >= independent_chain_count
    ):
        raise ValueError("phase-space chain index is out of range")
    if basis_size < CARTESIAN_DIMENSION or basis_size > len(
        basis_evaluation.basis_labels
    ):
        raise ValueError("Hamiltonian basis size is outside the nested dictionary")
    basis_values = basis_evaluation.basis_values[:, :basis_size]
    generator_values = basis_evaluation.generator_values[:, :basis_size]
    negative_generator_squared_values = (
        basis_evaluation.negative_generator_squared_values[:, :basis_size]
    )
    current_values = basis_evaluation.current_values
    parent_basis_labels = basis_evaluation.basis_labels[:basis_size]
    if generator_values.shape != basis_values.shape:
        raise ValueError("Hamiltonian generator values do not match the basis")
    if negative_generator_squared_values.shape != basis_values.shape:
        raise ValueError("Hamiltonian squared-generator values do not match the basis")
    if current_values.shape != (
        basis_values.shape[0],
        CARTESIAN_DIMENSION,
    ):
        raise ValueError("Hamiltonian current values have incompatible shape")
    observed_chain_indices = set(
        int(chain_index) for chain_index in basis_evaluation.chain_indices
    )
    required_chain_indices = set(range(independent_chain_count))
    if observed_chain_indices != required_chain_indices:
        raise ValueError("Hamiltonian phase-space samples omit an independent chain")
    even_chain_indices = tuple(range(0, independent_chain_count, 2))
    odd_chain_indices = tuple(range(1, independent_chain_count, 2))
    cross_fit_splits = (
        (even_chain_indices, odd_chain_indices),
        (odd_chain_indices, even_chain_indices),
    )
    inverse_temperature_per_J = 1.0 / (K_B * temperature_K)
    intervals_S_m: list[tuple[float, float, float]] = []
    basis_errors_S_m: list[float] = []
    equilibrium_errors_S_m: list[float] = []
    linear_solve_errors_S_m: list[float] = []
    linear_solve_residuals: list[float] = []
    conductivity_mcse_values_S_m: list[float] = []

    def solve_empirical_resolvent(
        selected_basis_values: Array,
        selected_generator_values: Array,
        selected_current_values: Array,
        eta_s_inv: float,
        conductivity_prefactor: float,
    ) -> tuple[Array, float, float, float]:
        selected_sample_count = selected_basis_values.shape[0]
        overlap_matrix = (
            selected_basis_values.T
            @ selected_basis_values
            / selected_sample_count
        )
        generator_energy_matrix = (
            selected_generator_values.T
            @ selected_generator_values
            / selected_sample_count
        )
        current_coupling = (
            selected_basis_values.T
            @ selected_current_values
            / selected_sample_count
        )
        basis_scales = np.sqrt(np.diag(overlap_matrix))
        if np.any(~np.isfinite(basis_scales)) or np.any(basis_scales <= 0.0):
            raise ValueError(
                "Hamiltonian phase-space basis contains a zero-norm mode"
            )
        normalized_overlap_matrix = overlap_matrix / (
            basis_scales[:, None] * basis_scales[None, :]
        )
        normalized_generator_energy_matrix = generator_energy_matrix / (
            basis_scales[:, None] * basis_scales[None, :]
        )
        normalized_current_coupling = current_coupling / basis_scales[:, None]
        overlap_cholesky = np.linalg.cholesky(normalized_overlap_matrix)
        transformed_generator = np.linalg.solve(
            overlap_cholesky,
            np.linalg.solve(
                overlap_cholesky,
                normalized_generator_energy_matrix,
            ).T,
        ).T
        transformed_generator = 0.5 * (
            transformed_generator + transformed_generator.T
        )
        transformed_current_coupling = np.linalg.solve(
            overlap_cholesky,
            normalized_current_coupling,
        )
        transformed_resolvent_matrix = (
            eta_s_inv**2 * np.eye(basis_size, dtype=float)
            + transformed_generator
        )
        resolvent_cholesky = np.linalg.cholesky(transformed_resolvent_matrix)
        transformed_coefficients = np.linalg.solve(
            resolvent_cholesky.T,
            np.linalg.solve(
                resolvent_cholesky,
                transformed_current_coupling,
            ),
        )
        normalized_coefficients = np.linalg.solve(
            overlap_cholesky.T,
            transformed_coefficients,
        )
        coefficient_matrix = normalized_coefficients / basis_scales[:, None]
        transformed_linear_residual = (
            transformed_resolvent_matrix @ transformed_coefficients
            - transformed_current_coupling
        )
        transformed_coupling_norm = max(
            float(np.linalg.norm(transformed_current_coupling)),
            np.finfo(float).tiny,
        )
        relative_linear_residual = float(
            np.linalg.norm(transformed_linear_residual)
            / transformed_coupling_norm
        )
        transformed_linear_error_solution = np.linalg.solve(
            resolvent_cholesky.T,
            np.linalg.solve(
                resolvent_cholesky,
                transformed_linear_residual,
            ),
        )
        linear_solve_error_S_m = conductivity_prefactor * abs(
            float(
                np.sum(
                    transformed_linear_residual
                    * transformed_linear_error_solution
                )
            )
        )
        conductivity_S_m = conductivity_prefactor * float(
            np.sum(current_coupling * coefficient_matrix)
        )
        return (
            coefficient_matrix,
            conductivity_S_m,
            linear_solve_error_S_m,
            relative_linear_residual,
        )

    for eta_s_inv in eta_values_s_inv:
        conductivity_prefactor = inverse_temperature_per_J * eta_s_inv / (
            CARTESIAN_DIMENSION * volume_m3
        )
        residual_prefactor = inverse_temperature_per_J / (
            CARTESIAN_DIMENSION * volume_m3 * eta_s_inv
        )
        held_out_chain_lower_values_S_m: list[float] = []
        held_out_chain_upper_values_S_m: list[float] = []
        fold_linear_solve_errors_S_m: list[float] = []
        fold_relative_linear_residuals: list[float] = []
        for training_chain_indices, validation_chain_indices in cross_fit_splits:
            training_mask = np.isin(
                basis_evaluation.chain_indices,
                training_chain_indices,
            )
            (
                coefficient_matrix,
                _training_resolvent_iterate_S_m,
                fold_linear_solve_error_S_m,
                fold_relative_linear_residual,
            ) = solve_empirical_resolvent(
                selected_basis_values=basis_values[training_mask],
                selected_generator_values=generator_values[training_mask],
                selected_current_values=current_values[training_mask],
                eta_s_inv=eta_s_inv,
                conductivity_prefactor=conductivity_prefactor,
            )
            fold_linear_solve_errors_S_m.append(fold_linear_solve_error_S_m)
            fold_relative_linear_residuals.append(fold_relative_linear_residual)
            for validation_chain_index in validation_chain_indices:
                validation_mask = (
                    basis_evaluation.chain_indices == validation_chain_index
                )
                validation_basis_values = basis_values[validation_mask]
                validation_generator_values = generator_values[validation_mask]
                validation_negative_generator_squared_values = (
                    negative_generator_squared_values[validation_mask]
                )
                validation_current_values = current_values[validation_mask]
                validation_trial_values = (
                    validation_basis_values @ coefficient_matrix
                )
                validation_trial_generator_values = (
                    validation_generator_values @ coefficient_matrix
                )
                validation_trial_negative_generator_squared_values = (
                    validation_negative_generator_squared_values
                    @ coefficient_matrix
                )
                validation_variational_samples_S_m = conductivity_prefactor * (
                    2.0
                    * np.sum(
                        validation_current_values * validation_trial_values,
                        axis=1,
                    )
                    - eta_s_inv**2
                    * np.sum(validation_trial_values**2, axis=1)
                    - np.sum(validation_trial_generator_values**2, axis=1)
                )
                validation_full_residual = (
                    validation_current_values
                    - eta_s_inv**2 * validation_trial_values
                    - validation_trial_negative_generator_squared_values
                )
                validation_residual_errors_S_m = residual_prefactor * np.sum(
                    validation_full_residual**2,
                    axis=1,
                )
                held_out_chain_lower_values_S_m.append(
                    float(np.mean(validation_variational_samples_S_m))
                )
                held_out_chain_upper_values_S_m.append(
                    float(
                        np.mean(
                            validation_variational_samples_S_m
                            + validation_residual_errors_S_m
                        )
                    )
                )
        held_out_lower_mean_S_m = float(
            np.mean(held_out_chain_lower_values_S_m)
        )
        held_out_upper_mean_S_m = float(
            np.mean(held_out_chain_upper_values_S_m)
        )
        lower_mcse_S_m = float(
            np.std(held_out_chain_lower_values_S_m, ddof=1)
            / math.sqrt(independent_chain_count)
        )
        upper_mcse_S_m = float(
            np.std(held_out_chain_upper_values_S_m, ddof=1)
            / math.sqrt(independent_chain_count)
        )
        lower_equilibrium_error_S_m = (
            numerics.equilibrium_standard_error_multiplier * lower_mcse_S_m
        )
        upper_equilibrium_error_S_m = (
            numerics.equilibrium_standard_error_multiplier * upper_mcse_S_m
        )
        linear_solve_error_S_m = max(fold_linear_solve_errors_S_m)
        lower_bound_S_m = (
            held_out_lower_mean_S_m
            - lower_equilibrium_error_S_m
            - linear_solve_error_S_m
        )
        upper_bound_S_m = (
            held_out_upper_mean_S_m
            + upper_equilibrium_error_S_m
            + linear_solve_error_S_m
        )
        if not math.isfinite(lower_bound_S_m) or not math.isfinite(upper_bound_S_m):
            raise FloatingPointError("Hamiltonian resolvent interval is nonfinite")
        if upper_bound_S_m < lower_bound_S_m:
            raise RuntimeError("Hamiltonian resolvent interval is inverted")
        held_out_chain_midpoints_S_m = 0.5 * (
            np.asarray(held_out_chain_lower_values_S_m)
            + np.asarray(held_out_chain_upper_values_S_m)
        )
        conductivity_mcse_S_m = float(
            np.std(held_out_chain_midpoints_S_m, ddof=1)
            / math.sqrt(independent_chain_count)
        )
        intervals_S_m.append(
            (
                float(eta_s_inv),
                float(lower_bound_S_m),
                float(upper_bound_S_m),
            )
        )
        basis_errors_S_m.append(float(upper_bound_S_m - lower_bound_S_m))
        equilibrium_errors_S_m.append(
            float(
                max(
                    lower_equilibrium_error_S_m,
                    upper_equilibrium_error_S_m,
                )
            )
        )
        linear_solve_residuals.append(max(fold_relative_linear_residuals))
        linear_solve_errors_S_m.append(linear_solve_error_S_m)
        conductivity_mcse_values_S_m.append(conductivity_mcse_S_m)
    (
        zero_frequency_iterate_S_m,
        zero_frequency_lower_bound_S_m,
        zero_frequency_upper_bound_S_m,
        eta_continuation_error_S_m,
    ) = _zero_coordinate_limit_interval(
        positive_coordinates=np.asarray(eta_values_s_inv),
        lower_values=np.asarray(
            [interval[1] for interval in intervals_S_m],
        ),
        upper_values=np.asarray(
            [interval[2] for interval in intervals_S_m],
        ),
    )
    if len(parent_basis_labels) != basis_values.shape[1]:
        raise RuntimeError("Hamiltonian phase-space basis labels are inconsistent")
    return HamiltonianResolventEstimate(
        resolvent_iterate_S_m=zero_frequency_iterate_S_m,
        lower_bound_S_m=zero_frequency_lower_bound_S_m,
        upper_bound_S_m=zero_frequency_upper_bound_S_m,
        intervals_S_m=tuple(intervals_S_m),
        basis_error_S_m=basis_errors_S_m[-1],
        eta_continuation_error_S_m=eta_continuation_error_S_m,
        equilibrium_error_S_m=equilibrium_errors_S_m[-1],
        linear_solve_error_S_m=linear_solve_errors_S_m[-1],
        linear_solve_relative_residual=max(linear_solve_residuals),
        conductivity_mcse_S_m=conductivity_mcse_values_S_m[-1],
        basis_size=len(parent_basis_labels),
        basis_labels=parent_basis_labels,
        finite_eta_precision_reached=(
            max(basis_errors_S_m) <= numerics.conductivity_tolerance_S_m
        ),
    )


def _concatenate_hamiltonian_phase_space_samples(
    sample_blocks: list[HamiltonianPhaseSpaceSamples],
) -> HamiltonianPhaseSpaceSamples:
    if not sample_blocks:
        raise ValueError("Hamiltonian resolvent requires phase-space samples")
    return HamiltonianPhaseSpaceSamples(
        configurations_m=np.concatenate(
            [block.configurations_m for block in sample_blocks],
            axis=0,
        ),
        box_vectors_m=np.concatenate(
            [block.box_vectors_m for block in sample_blocks],
            axis=0,
        ),
        momenta_kg_m_s=np.concatenate(
            [block.momenta_kg_m_s for block in sample_blocks],
            axis=0,
        ),
        forces_N=np.concatenate(
            [block.forces_N for block in sample_blocks],
            axis=0,
        ),
        chain_indices=np.concatenate(
            [block.chain_indices for block in sample_blocks],
            axis=0,
        ),
    )


def _concatenate_hamiltonian_basis_evaluations(
    first_evaluation: HamiltonianBasisEvaluation,
    second_evaluation: HamiltonianBasisEvaluation,
) -> HamiltonianBasisEvaluation:
    if first_evaluation.basis_level != second_evaluation.basis_level:
        raise ValueError("Hamiltonian basis levels cannot be concatenated")
    if first_evaluation.basis_labels != second_evaluation.basis_labels:
        raise ValueError("Hamiltonian basis labels cannot be concatenated")
    return HamiltonianBasisEvaluation(
        basis_level=first_evaluation.basis_level,
        basis_labels=first_evaluation.basis_labels,
        basis_values=np.concatenate(
            (first_evaluation.basis_values, second_evaluation.basis_values),
            axis=0,
        ),
        generator_values=np.concatenate(
            (first_evaluation.generator_values, second_evaluation.generator_values),
            axis=0,
        ),
        negative_generator_squared_values=np.concatenate(
            (
                first_evaluation.negative_generator_squared_values,
                second_evaluation.negative_generator_squared_values,
            ),
            axis=0,
        ),
        current_values=np.concatenate(
            (first_evaluation.current_values, second_evaluation.current_values),
            axis=0,
        ),
        chain_indices=np.concatenate(
            (first_evaluation.chain_indices, second_evaluation.chain_indices),
            axis=0,
        ),
    )


def _write_hamiltonian_resolvent_checkpoint(
    checkpoint_path: Path,
    checkpoint_schema: str,
    checkpoint_fingerprint: str,
    state: IonicHrexState,
    phase_space_blocks: list[HamiltonianPhaseSpaceSamples],
    basis_evaluations: tuple[HamiltonianBasisEvaluation, ...],
    completed_refinement_blocks: int,
    basis_level: int,
) -> None:
    if len(basis_evaluations) > 1:
        raise ValueError("checkpoint accepts one accumulated basis evaluation")
    if basis_evaluations and basis_evaluations[0].basis_level != basis_level:
        raise ValueError("checkpoint basis level and evaluation disagree")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint_path = checkpoint_path.with_suffix(
        f"{checkpoint_path.suffix}.tmp"
    )
    checkpoint_payload = {
        "schema": checkpoint_schema,
        "fingerprint": checkpoint_fingerprint,
        "state": asdict(state),
        "phase_space_blocks": tuple(asdict(block) for block in phase_space_blocks),
        "basis_evaluations": tuple(
            asdict(basis_evaluation) for basis_evaluation in basis_evaluations
        ),
        "completed_refinement_blocks": completed_refinement_blocks,
        "basis_level": basis_level,
    }
    with temporary_checkpoint_path.open("wb") as checkpoint_file:
        pickle.dump(checkpoint_payload, checkpoint_file)
    os.replace(temporary_checkpoint_path, checkpoint_path)


def hamiltonian_green_kubo_sequence(
    model: AnalyticalPeriodicInteratomicModel,
    state: IonicHrexState,
    settings: IonicHrexSettings,
    system: MolecularSystem,
    temperature_K: float,
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
    checkpoint_path: Path,
    checkpoint_fingerprint: str,
) -> tuple[HamiltonianResolventEstimate, int]:
    phase_space_blocks: list[HamiltonianPhaseSpaceSamples] = []
    basis_evaluations: list[HamiltonianBasisEvaluation] = []
    completed_refinement_blocks = 0
    basis_level = 1
    checkpoint_schema = "hamiltonian_phase_space_resolvent_v7"
    if checkpoint_path.is_file():
        with checkpoint_path.open("rb") as checkpoint_file:
            checkpoint_payload = pickle.load(checkpoint_file)
        if (
            "schema" not in checkpoint_payload
            or checkpoint_payload["schema"] != checkpoint_schema
        ):
            raise ValueError(
                "Hamiltonian resolvent checkpoint uses the retired operator schema; "
                "remove it and restart from the retained initialization checkpoint"
            )
        if checkpoint_payload["fingerprint"] != checkpoint_fingerprint:
            raise ValueError(
                "Hamiltonian resolvent checkpoint does not match the calculation"
            )
        state_record = checkpoint_payload["state"]
        state = IonicHrexState(
            positions_m=state_record["positions_m"],
            boxes_m=state_record["boxes_m"],
            component_energies_J=state_record["component_energies_J"],
            momenta_kg_m_s=state_record["momenta_kg_m_s"],
            momentum_refresh_required=state_record["momentum_refresh_required"],
            auxiliary_masses_kg=state_record["auxiliary_masses_kg"],
            walker_identifiers=state_record["walker_identifiers"],
            visited_lowest_lambda=state_record["visited_lowest_lambda"],
            completed_round_trips=state_record["completed_round_trips"],
            round_trip_phase=state_record["round_trip_phase"],
            hmc_step_sizes_s=state_record["hmc_step_sizes_s"],
            hmc_attempts=state_record["hmc_attempts"],
            hmc_acceptances=state_record["hmc_acceptances"],
            hmc_expected_acceptance_sums=state_record[
                "hmc_expected_acceptance_sums"
            ],
            hmc_absolute_energy_error_over_kbt_sums=state_record[
                "hmc_absolute_energy_error_over_kbt_sums"
            ],
            hmc_molecular_com_squared_displacement_sums_m2=state_record[
                "hmc_molecular_com_squared_displacement_sums_m2"
            ],
            exchange_attempts=state_record["exchange_attempts"],
            exchange_acceptances=state_record["exchange_acceptances"],
            exchange_expected_acceptance_sums=state_record[
                "exchange_expected_acceptance_sums"
            ],
            cycle_index=int(state_record["cycle_index"]),
            random_generator_state=state_record["random_generator_state"],
        )
        phase_space_blocks.extend(
            HamiltonianPhaseSpaceSamples(
                configurations_m=block_record["configurations_m"],
                box_vectors_m=block_record["box_vectors_m"],
                momenta_kg_m_s=block_record["momenta_kg_m_s"],
                forces_N=block_record["forces_N"],
                chain_indices=block_record["chain_indices"],
            )
            for block_record in checkpoint_payload["phase_space_blocks"]
        )
        basis_evaluations.extend(
            HamiltonianBasisEvaluation(
                basis_level=int(evaluation_record["basis_level"]),
                basis_labels=tuple(evaluation_record["basis_labels"]),
                basis_values=evaluation_record["basis_values"],
                generator_values=evaluation_record["generator_values"],
                negative_generator_squared_values=evaluation_record[
                    "negative_generator_squared_values"
                ],
                current_values=evaluation_record["current_values"],
                chain_indices=evaluation_record["chain_indices"],
            )
            for evaluation_record in checkpoint_payload["basis_evaluations"]
        )
        completed_refinement_blocks = int(
            checkpoint_payload["completed_refinement_blocks"]
        )
        basis_level = int(checkpoint_payload["basis_level"])
        expected_basis_labels = _phase_space_basis_labels(system, basis_level)
        if any(
            evaluation.basis_labels != expected_basis_labels
            for evaluation in basis_evaluations
        ):
            raise ValueError(
                "Hamiltonian checkpoint basis definition does not match the source"
            )
        if len(basis_evaluations) > 1:
            raise ValueError(
                "Hamiltonian checkpoint contains multiple accumulated evaluations"
            )
        if len(phase_space_blocks) != completed_refinement_blocks:
            raise ValueError(
                "Hamiltonian checkpoint block count and sample blocks disagree"
            )
        if basis_evaluations and completed_refinement_blocks < (
            dynamics.equilibrium_maximum_refinement_batches
        ):
            raise ValueError(
                "Hamiltonian checkpoint evaluated operators before equilibrium "
                "sampling completed"
            )
        print(
            "[Hamiltonian resolvent restart] "
            f"blocks={completed_refinement_blocks} "
            f"basis_level={basis_level} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )
    if not checkpoint_path.is_file():
        warmup_start_s = time.perf_counter()
        state, warmup_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            cycle_count=settings.warmup_cycle_count,
            attempt_exchange=len(settings.lambdas) > 1,
            retain_samples=False,
        )
        warmup_elapsed_s = time.perf_counter() - warmup_start_s
        _report_hrex_block(
            stage="Hamiltonian-resolvent warmup",
            block_index=0,
            block_elapsed_s=warmup_elapsed_s,
            total_elapsed_s=warmup_elapsed_s,
            state=state,
            settings=settings,
            block=warmup_block,
        )
    sequence_start_s = time.perf_counter()
    for refinement_block_index in range(
        completed_refinement_blocks,
        dynamics.equilibrium_maximum_refinement_batches,
    ):
        block_start_s = time.perf_counter()
        state, sampling_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            cycle_count=settings.block_cycle_count,
            attempt_exchange=len(settings.lambdas) > 1,
            retain_samples=True,
        )
        sampling_elapsed_s = time.perf_counter() - block_start_s
        _report_hrex_block(
            stage="Hamiltonian-resolvent production",
            block_index=refinement_block_index + 1,
            block_elapsed_s=sampling_elapsed_s,
            total_elapsed_s=time.perf_counter() - sequence_start_s,
            state=state,
            settings=settings,
            block=sampling_block,
        )
        expected_configuration_count = (
            settings.independent_ladder_count * dynamics.equilibrium_sample_count
        )
        if sampling_block.physical_configurations_m.shape[0] != (
            expected_configuration_count
        ):
            raise ValueError(
                "Hamiltonian operator block did not retain the configured number "
                f"of configurations: observed="
                f"{sampling_block.physical_configurations_m.shape[0]}, "
                f"expected={expected_configuration_count}"
            )
        if (
            sampling_block.physical_momenta_kg_m_s.shape
            != sampling_block.physical_configurations_m.shape
        ):
            raise ValueError(
                "Hamiltonian operator block did not retain one canonical momentum "
                "sample per configuration"
            )
        if (
            sampling_block.physical_forces_N.shape
            != sampling_block.physical_configurations_m.shape
        ):
            raise ValueError(
                "Hamiltonian operator block did not retain one physical force "
                "sample per configuration"
            )
        phase_space_blocks.append(
            HamiltonianPhaseSpaceSamples(
                configurations_m=sampling_block.physical_configurations_m,
                box_vectors_m=sampling_block.physical_box_vectors_by_sample_m,
                momenta_kg_m_s=sampling_block.physical_momenta_kg_m_s,
                forces_N=sampling_block.physical_forces_N,
                chain_indices=sampling_block.physical_ladder_indices,
            )
        )
        completed_refinement_blocks = refinement_block_index + 1
        _write_hamiltonian_resolvent_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_schema=checkpoint_schema,
            checkpoint_fingerprint=checkpoint_fingerprint,
            state=state,
            phase_space_blocks=phase_space_blocks,
            basis_evaluations=tuple(basis_evaluations),
            completed_refinement_blocks=completed_refinement_blocks,
            basis_level=basis_level,
        )
    if not phase_space_blocks:
        raise RuntimeError("Hamiltonian equilibrium sequence retained no samples")
    accumulated_phase_space_samples = _concatenate_hamiltonian_phase_space_samples(
        phase_space_blocks
    )
    sample_count = accumulated_phase_space_samples.configurations_m.shape[0]
    samples_per_chain = int(sample_count / settings.independent_ladder_count)
    minimum_training_sample_count = min(
        int(
            np.sum(
                np.isin(
                    accumulated_phase_space_samples.chain_indices,
                    tuple(range(parity, settings.independent_ladder_count, 2)),
                )
            )
        )
        for parity in (0, 1)
    )
    configured_basis_level = dynamics.equilibrium_maximum_refinement_batches
    if not basis_evaluations:
        basis_level = configured_basis_level
    if basis_level != configured_basis_level:
        raise ValueError(
            "checkpoint basis level does not match the configured nested sequence"
        )
    configured_basis_size = len(_phase_space_basis_labels(system, basis_level))
    if minimum_training_sample_count < configured_basis_size:
        raise RuntimeError(
            "Hamiltonian equilibrium budget is smaller than the predetermined "
            "nested basis: "
            f"training_samples={minimum_training_sample_count}, "
            f"basis_size={configured_basis_size}"
        )
    evaluated_sample_count = 0
    if basis_evaluations:
        evaluated_sample_count = basis_evaluations[0].basis_values.shape[0]
    block_sample_start = 0
    for phase_space_block in phase_space_blocks:
        block_sample_count = phase_space_block.configurations_m.shape[0]
        block_sample_stop = block_sample_start + block_sample_count
        if block_sample_stop <= evaluated_sample_count:
            block_sample_start = block_sample_stop
            continue
        local_sample_start = max(
            evaluated_sample_count - block_sample_start,
            0,
        )
        for operator_batch_start in range(
            local_sample_start,
            block_sample_count,
            dynamics.resolvent_operator_batch_size,
        ):
            operator_batch_stop = min(
                operator_batch_start + dynamics.resolvent_operator_batch_size,
                block_sample_count,
            )
            operator_phase_space_samples = HamiltonianPhaseSpaceSamples(
                configurations_m=phase_space_block.configurations_m[
                    operator_batch_start:operator_batch_stop
                ],
                box_vectors_m=phase_space_block.box_vectors_m[
                    operator_batch_start:operator_batch_stop
                ],
                momenta_kg_m_s=phase_space_block.momenta_kg_m_s[
                    operator_batch_start:operator_batch_stop
                ],
                forces_N=phase_space_block.forces_N[
                    operator_batch_start:operator_batch_stop
                ],
                chain_indices=phase_space_block.chain_indices[
                    operator_batch_start:operator_batch_stop
                ],
            )
            operator_evaluation = _evaluate_phase_space_basis_samples(
                phase_space_samples=operator_phase_space_samples,
                model=model,
                basis_level=basis_level,
                temperature_K=temperature_K,
                operator_batch_size=dynamics.resolvent_operator_batch_size,
            )
            evaluation_already_started = bool(basis_evaluations)
            if evaluation_already_started:
                basis_evaluations[0] = _concatenate_hamiltonian_basis_evaluations(
                    first_evaluation=basis_evaluations[0],
                    second_evaluation=operator_evaluation,
                )
            if not evaluation_already_started:
                basis_evaluations.append(operator_evaluation)
            evaluated_sample_count = (
                block_sample_start + operator_batch_stop
            )
            _write_hamiltonian_resolvent_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_schema=checkpoint_schema,
                checkpoint_fingerprint=checkpoint_fingerprint,
                state=state,
                phase_space_blocks=phase_space_blocks,
                basis_evaluations=tuple(basis_evaluations),
                completed_refinement_blocks=completed_refinement_blocks,
                basis_level=basis_level,
            )
        block_sample_start = block_sample_stop
    basis_evaluation = basis_evaluations[0]
    if basis_evaluation.basis_level != basis_level:
        raise ValueError("checkpoint basis level and evaluation disagree")
    if basis_evaluation.basis_values.shape[0] != sample_count:
        raise ValueError("checkpoint basis evaluation omits equilibrium samples")
    nested_basis_estimates: list[HamiltonianResolventEstimate] = []
    for nested_basis_size in range(
        CARTESIAN_DIMENSION,
        len(basis_evaluation.basis_labels) + 1,
    ):
        current_estimate = _solve_hamiltonian_resolvent(
            basis_evaluation=basis_evaluation,
            basis_size=nested_basis_size,
            temperature_K=temperature_K,
            volume_m3=float(abs(np.linalg.det(system.box_vectors_m))),
            independent_chain_count=settings.independent_ladder_count,
            eta_values_s_inv=numerics.resolvent_eta_values_s_inv,
            numerics=numerics,
        )
        nested_basis_estimates.append(current_estimate)
        print(
            "[Hamiltonian nested basis] "
            f"basis_size={nested_basis_size} "
            f"lower_S_m={current_estimate.lower_bound_S_m:.12g} "
            f"upper_S_m={current_estimate.upper_bound_S_m:.12g} "
            f"basis_error_S_m={current_estimate.basis_error_S_m:.12g} "
            "finite_eta_precision_reached="
            f"{current_estimate.finite_eta_precision_reached}",
            flush=True,
        )
        if current_estimate.finite_eta_precision_reached:
            break
    if not nested_basis_estimates:
        raise RuntimeError("Hamiltonian nested basis sequence did not execute")
    current_estimate = nested_basis_estimates[-1]
    basis_evaluations[0] = basis_evaluation
    _write_hamiltonian_resolvent_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_schema=checkpoint_schema,
        checkpoint_fingerprint=checkpoint_fingerprint,
        state=state,
        phase_space_blocks=phase_space_blocks,
        basis_evaluations=tuple(basis_evaluations),
        completed_refinement_blocks=completed_refinement_blocks,
        basis_level=basis_level,
    )
    print(
        "[Hamiltonian Green-Kubo] "
        f"samples_per_chain={samples_per_chain} "
        f"basis_level={basis_level} "
        f"basis_size={current_estimate.basis_size} "
        f"resolvent_iterate_S_m={current_estimate.resolvent_iterate_S_m:.12g} "
        f"lower_S_m={current_estimate.lower_bound_S_m:.12g} "
        f"upper_S_m={current_estimate.upper_bound_S_m:.12g} "
        f"basis_error_S_m={current_estimate.basis_error_S_m:.12g} "
        f"eta_continuation_error_S_m="
        f"{current_estimate.eta_continuation_error_S_m:.12g} "
        f"equilibrium_error_S_m={current_estimate.equilibrium_error_S_m:.12g} "
        f"linear_solve_error_S_m={current_estimate.linear_solve_error_S_m:.12g} "
        f"mcse_S_m={current_estimate.conductivity_mcse_S_m:.12g} "
        f"eta_intervals={current_estimate.intervals_S_m} "
        "finite_eta_precision_reached="
        f"{current_estimate.finite_eta_precision_reached}",
        flush=True,
    )
    return current_estimate, samples_per_chain


def validate_force_consistency(
    model: AnalyticalPeriodicInteratomicModel,
    positions_m: Array,
    numerics: NumericalSettings,
    random_seed: int,
) -> None:
    random_generator = np.random.default_rng(random_seed)
    direction = random_generator.normal(size=positions_m.shape)
    direction /= np.linalg.norm(direction)
    displacement_m = numerics.force_difference_step_m * direction
    positive_energy_J = model.energy_J(
        positions_m + displacement_m,
        model.system.box_vectors_m,
    )
    negative_energy_J = model.energy_J(
        positions_m - displacement_m,
        model.system.box_vectors_m,
    )
    finite_difference_directional_derivative_N = (
        positive_energy_J - negative_energy_J
    ) / (2.0 * numerics.force_difference_step_m)
    analytical_directional_derivative_N = -float(
        np.sum(
            model.forces_N(positions_m, model.system.box_vectors_m)
            * direction
        )
    )
    comparison_scale_N = max(
        abs(finite_difference_directional_derivative_N),
        abs(analytical_directional_derivative_N),
        np.finfo(float).tiny,
    )
    relative_error = abs(
        finite_difference_directional_derivative_N
        - analytical_directional_derivative_N
    ) / comparison_scale_N
    if relative_error > numerics.force_consistency_relative_tolerance:
        raise ValueError(
            f"force directional derivative relative error {relative_error:.6e} "
            f"exceeds {numerics.force_consistency_relative_tolerance:.6e}"
        )


def _validate_settings(
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
) -> None:
    dynamics_record = asdict(dynamics)
    hrex_lambdas = tuple(dynamics_record.pop("ionic_hrex_lambdas"))
    numerics_record = asdict(numerics)
    resolvent_eta_values_s_inv = tuple(
        numerics_record.pop("resolvent_eta_values_s_inv")
    )
    values = tuple(dynamics_record.values()) + tuple(numerics_record.values())
    if any(float(value) <= 0.0 for value in values):
        raise ValueError(
            "all dynamics settings and numerical settings must be positive"
        )
    if not hrex_lambdas or hrex_lambdas[0] != 1.0:
        raise ValueError("ionic sampling lambdas must start at one")
    fractions = (
        dynamics.initial_relaxation_initial_damping,
        dynamics.initial_relaxation_timestep_decrease,
        dynamics.initial_relaxation_damping_decrease,
        dynamics.initial_relaxation_minimum_force_improvement_fraction,
        numerics.ewald_reciprocal_relative_tolerance,
        numerics.minimum_interatomic_contact_ratio,
    )
    if any(value >= 1.0 for value in fractions):
        raise ValueError("fractional numerical settings must be below one")
    if (
        dynamics.initial_relaxation_timestep_increase <= 1.0
        or dynamics.initial_relaxation_maximum_timestep_s
        < dynamics.initial_relaxation_timestep_s
    ):
        raise ValueError("FIRE timestep controls are inconsistent")
    if dynamics.force_batch_size > dynamics.equilibrium_chain_count:
        raise ValueError("relaxation force batch exceeds the independent chain count")
    if dynamics.initial_relaxation_maximum_force_evaluations < math.ceil(
        dynamics.equilibrium_chain_count / dynamics.force_batch_size
    ):
        raise ValueError("relaxation budget cannot complete one exact-shape preflight")
    if not (numerics.lennard_jones_switch_start_m < numerics.lennard_jones_cutoff_m):
        raise ValueError("Lennard-Jones switch start must be below the cutoff")
    if len(resolvent_eta_values_s_inv) < 3 or any(
        eta_s_inv <= 0.0 for eta_s_inv in resolvent_eta_values_s_inv
    ):
        raise ValueError("at least three positive resolvent eta values are required")
    if any(
        next_eta_s_inv >= eta_s_inv
        for eta_s_inv, next_eta_s_inv in zip(
            resolvent_eta_values_s_inv[:-1],
            resolvent_eta_values_s_inv[1:],
            strict=True,
        )
    ):
        raise ValueError("resolvent eta values must be strictly decreasing")
    if (
        dynamics.equilibrium_chain_count < 4
        or dynamics.equilibrium_chain_count % 2 != 0
    ):
        raise ValueError(
            "Hamiltonian cross-fitted equilibrium integration requires an even "
            "number of at least four chains"
        )
    if dynamics.hrex_block_cycle_count != (
        dynamics.equilibrium_sample_count * dynamics.hrex_measurement_stride
    ):
        raise ValueError(
            "HREX block cycles must retain exactly equilibrium_sample_count "
            "configurations per independent chain"
        )


def _compute_finite_cell_resolvent(
    recipe: ElectrolyteRecipeModel,
    recipe_realization: IntegerRecipeRealization,
    temperature_K: float,
    liquid_density_kg_m3: float,
    density_source: str,
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
    random_seed: int,
    initialization_checkpoint_path: Path,
) -> tuple[
    HamiltonianResolventEstimate,
    IntegerRecipeRealization,
    float,
    float,
]:
    _validate_settings(dynamics, numerics)
    if (
        temperature_K <= 0.0
        or liquid_density_kg_m3 <= 0.0
        or recipe_realization.explicit_molecule_count <= 0
    ):
        raise ValueError(
            "temperature, liquid density, and realized molecule count must be positive"
        )
    if not density_source.strip():
        raise ValueError("density source must identify the imposed NVT state")
    seed_sequences = np.random.SeedSequence(random_seed).spawn(
        dynamics.equilibrium_chain_count + 1
    )
    packing_random_seeds = tuple(
        int(seed_sequence.generate_state(1)[0]) for seed_sequence in seed_sequences[:-1]
    )
    sampling_random_seed = int(seed_sequences[-1].generate_state(1)[0])
    checkpoint_dynamics_record = asdict(dynamics)
    checkpoint_numerics_record = asdict(numerics)
    initialization_dynamics_keys = {
        field_name
        for field_name in checkpoint_dynamics_record
        if field_name.startswith("initial_relaxation_")
    }
    initialization_dynamics_keys.update(
        {
            "force_batch_size",
            "initial_force_tolerance_N",
            "equilibrium_chain_count",
            "solvent_volume_fraction_tolerance",
            "salt_molarity_tolerance_mol_L",
            "additive_weight_fraction_tolerance",
            "maximum_explicit_molecule_count",
            "maximum_atom_count",
        }
    )
    initialization_fingerprint_record = {
        "recipe": {
            "solvents": dict(recipe.solvents),
            "salts": dict(recipe.salts),
            "additives": dict(recipe.additives),
        },
        "temperature_K": temperature_K,
        "liquid_density_kg_m3": liquid_density_kg_m3,
        "recipe_realization": asdict(recipe_realization),
        "dynamics": {
            field_name: checkpoint_dynamics_record[field_name]
            for field_name in sorted(initialization_dynamics_keys)
        },
        "numerics": {
            field_name: field_value
            for field_name, field_value in checkpoint_numerics_record.items()
            if field_name
            not in {
                "resolvent_eta_values_s_inv",
                "equilibrium_standard_error_multiplier",
                "conductivity_tolerance_S_m",
            }
        },
        "packing_random_seeds": packing_random_seeds,
        "initialization_schema": "hamiltonian_initialization_v2",
    }
    initialization_checkpoint_fingerprint = hashlib.sha256(
        json.dumps(
            initialization_fingerprint_record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    resolvent_fingerprint_record = {
        "initialization_fingerprint": initialization_checkpoint_fingerprint,
        "dynamics": checkpoint_dynamics_record,
        "numerics": checkpoint_numerics_record,
        "sampling_random_seed": sampling_random_seed,
        "resolvent_schema": "hamiltonian_phase_space_resolvent_v7",
    }
    resolvent_checkpoint_fingerprint = hashlib.sha256(
        json.dumps(
            resolvent_fingerprint_record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if initialization_checkpoint_path.is_file():
        with initialization_checkpoint_path.open("rb") as checkpoint_file:
            checkpoint_payload = pickle.load(checkpoint_file)
        if (
            "schema" not in checkpoint_payload
            or checkpoint_payload["schema"] != "hamiltonian_initialization_v2"
        ):
            raise ValueError(
                "initialization checkpoint uses a retired serialization schema"
            )
        if (
            checkpoint_payload["fingerprint"]
            != initialization_checkpoint_fingerprint
        ):
            raise ValueError(
                "initialization checkpoint does not match the requested calculation"
            )
        checkpoint_recipe_realization = _integer_recipe_realization_from_checkpoint_record(
            checkpoint_payload["metadata"]["recipe_realization"]
        )
        if checkpoint_recipe_realization != recipe_realization:
            raise ValueError(
                "initialization checkpoint realization differs from the requested cell"
            )
    density_conditioned_length_m = recipe_realization.density_conditioned_volume_m3 ** (
        1.0 / CARTESIAN_DIMENSION
    )
    if density_conditioned_length_m > numerics.ewald_maximum_box_length_m:
        raise ValueError(
            "realized density-conditioned box exceeds the configured Ewald box "
            f"limit: box_length_m={density_conditioned_length_m:.12g}, "
            "ewald_maximum_box_length_m="
            f"{numerics.ewald_maximum_box_length_m:.12g}"
        )
    print(
        "[integer recipe realization] "
        f"formula_units={recipe_realization.formula_unit_counts} "
        f"explicit_species={recipe_realization.explicit_species_counts} "
        f"explicit_molecules={recipe_realization.explicit_molecule_count} "
        f"atoms={recipe_realization.atom_count} "
        f"volume_m3={recipe_realization.density_conditioned_volume_m3:.12g} "
        f"solvent_volume_fractions="
        f"{recipe_realization.realized_solvent_volume_fractions} "
        f"salt_molarities_mol_L={recipe_realization.realized_salt_molarities_mol_L} "
        f"additive_weight_fractions="
        f"{recipe_realization.realized_additive_weight_fractions} "
        f"deviations={recipe_realization.native_unit_deviations}",
        flush=True,
    )
    density_conditioned_systems: list[MolecularSystem] = []
    if initialization_checkpoint_path.is_file():
        with initialization_checkpoint_path.open("rb") as checkpoint_file:
            checkpoint_payload = pickle.load(checkpoint_file)
        if (
            checkpoint_payload["fingerprint"]
            != initialization_checkpoint_fingerprint
        ):
            raise ValueError(
                "initialization checkpoint does not match the requested calculation"
            )
        density_conditioned_systems.extend(
            _molecular_system_from_checkpoint_record(system_record)
            for system_record in checkpoint_payload["systems"]
        )
        initialization_checkpoint_metadata = checkpoint_payload["metadata"]
        print(
            "[initialization restart] "
            f"stage={checkpoint_payload['stage']} "
            f"checkpoint={initialization_checkpoint_path}",
            flush=True,
        )
    else:
        for packing_random_seed in packing_random_seeds:
            ladder_system = build_periodic_molecular_system(
                explicit_species_counts=recipe_realization.explicit_species_counts,
                box_volume_m3=recipe_realization.density_conditioned_volume_m3,
                minimum_interatomic_contact_ratio=(
                    numerics.minimum_interatomic_contact_ratio
                ),
                initial_placement_attempts_per_molecule=(
                    numerics.initial_placement_attempts_per_molecule
                ),
                random_seed=packing_random_seed,
            )
            density_conditioned_systems.append(ladder_system)
        reference_system_for_fingerprint = density_conditioned_systems[0]
        topology_hasher = hashlib.sha256()
        for topology_array in (
            reference_system_for_fingerprint.masses_kg,
            reference_system_for_fingerprint.charges_C,
            reference_system_for_fingerprint.lj_sigma_m,
            reference_system_for_fingerprint.lj_epsilon_J,
            reference_system_for_fingerprint.polarizabilities_SI,
            reference_system_for_fingerprint.molecule_index,
            reference_system_for_fingerprint.bonds,
            reference_system_for_fingerprint.angles,
        ):
            topology_hasher.update(np.ascontiguousarray(topology_array).tobytes())
        checkpoint_metadata = {
            "recipe_realization": asdict(recipe_realization),
            "formula_unit_counts": recipe_realization.formula_unit_counts,
            "explicit_species_counts": recipe_realization.explicit_species_counts,
            "explicit_molecule_count": recipe_realization.explicit_molecule_count,
            "atom_count": recipe_realization.atom_count,
            "box_vectors_m": reference_system_for_fingerprint.box_vectors_m.tolist(),
            "packing_random_seeds": packing_random_seeds,
            "sampling_random_seed": sampling_random_seed,
            "topology_fingerprint": topology_hasher.hexdigest(),
            "numerics_fingerprint": hashlib.sha256(
                json.dumps(
                    checkpoint_numerics_record,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        initialization_checkpoint_metadata = checkpoint_metadata
        initialization_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_temporary_path = initialization_checkpoint_path.with_suffix(
            f"{initialization_checkpoint_path.suffix}.tmp"
        )
        packed_checkpoint_payload = {
            "schema": "hamiltonian_initialization_v2",
            "fingerprint": initialization_checkpoint_fingerprint,
            "metadata": checkpoint_metadata,
            "stage": "packed",
            "systems": tuple(
                asdict(system) for system in density_conditioned_systems
            ),
            "velocities_m_s": np.zeros_like(
                np.stack(
                    tuple(system.positions_m for system in density_conditioned_systems)
                )
            ),
            "accepted_positions_m": np.stack(
                tuple(system.positions_m for system in density_conditioned_systems)
            ),
            "timesteps_s": np.full(
                dynamics.equilibrium_chain_count,
                dynamics.initial_relaxation_timestep_s,
            ),
            "damping": np.full(
                dynamics.equilibrium_chain_count,
                dynamics.initial_relaxation_initial_damping,
            ),
            "positive_power_steps": np.zeros(
                dynamics.equilibrium_chain_count,
                dtype=int,
            ),
            "best_maximum_forces_N": np.full(
                dynamics.equilibrium_chain_count,
                np.inf,
            ),
            "accepted_energies_J": np.zeros(dynamics.equilibrium_chain_count),
            "has_accepted_energy": np.zeros(
                dynamics.equilibrium_chain_count,
                dtype=bool,
            ),
            "stagnant_iterations": np.zeros(
                dynamics.equilibrium_chain_count,
                dtype=int,
            ),
            "converged": np.zeros(
                dynamics.equilibrium_chain_count,
                dtype=bool,
            ),
            "force_evaluation_count": 0,
            "iteration": 0,
            "elapsed_s": 0.0,
        }
        with checkpoint_temporary_path.open("wb") as checkpoint_file:
            pickle.dump(packed_checkpoint_payload, checkpoint_file)
        os.replace(checkpoint_temporary_path, initialization_checkpoint_path)
        print(
            "[initialization checkpoint] "
            f"stage=packed checkpoint={initialization_checkpoint_path}",
            flush=True,
        )
    reference_system = density_conditioned_systems[0]
    interaction_model = AnalyticalPeriodicInteratomicModel(reference_system, numerics)
    for ladder_system in density_conditioned_systems:
        if (
            ladder_system.molecule_species_names
            != reference_system.molecule_species_names
            or not np.array_equal(ladder_system.masses_kg, reference_system.masses_kg)
            or not np.array_equal(ladder_system.charges_C, reference_system.charges_C)
        ):
            raise ValueError("independent packings do not share one molecular topology")
    (
        relaxed_positions_by_ladder_m,
        maximum_forces_by_ladder_N,
        relaxation_force_evaluation_count,
    ) = relax_initial_configurations(
        model=interaction_model,
        initial_systems=tuple(density_conditioned_systems),
        dynamics=dynamics,
        checkpoint_path=initialization_checkpoint_path,
        checkpoint_fingerprint=initialization_checkpoint_fingerprint,
        checkpoint_metadata=initialization_checkpoint_metadata,
    )
    relaxed_systems: list[MolecularSystem] = []
    for ladder_index, ladder_system in enumerate(density_conditioned_systems):
        print(
            "[relaxation] "
            f"chain={ladder_index} "
            f"force_evaluations={relaxation_force_evaluation_count} "
            f"maximum_force_N={maximum_forces_by_ladder_N[ladder_index]:.12g}",
            flush=True,
        )
        relaxed_systems.append(
            replace(
                ladder_system,
                positions_m=relaxed_positions_by_ladder_m[ladder_index],
            )
        )
    relaxed_system = relaxed_systems[0]
    hrex_settings = IonicHrexSettings(
        lambdas=dynamics.ionic_hrex_lambdas,
        hmc_step_size_s=dynamics.hamiltonian_timestep_s,
        hmc_steps_min=dynamics.hmc_steps_min,
        hmc_steps_max=dynamics.hmc_steps_max,
        hmc_momentum_persistence=dynamics.hmc_momentum_persistence,
        hmc_full_refresh_stride=dynamics.hmc_full_refresh_stride,
        exchange_stride=dynamics.exchange_stride,
        independent_ladder_count=dynamics.equilibrium_chain_count,
        warmup_cycle_count=dynamics.hrex_warmup_cycle_count,
        measurement_stride=dynamics.hrex_measurement_stride,
        block_cycle_count=dynamics.hrex_block_cycle_count,
        force_batch_size=dynamics.force_batch_size,
    )
    initial_boxes_by_ladder_m = np.repeat(
        relaxed_system.box_vectors_m[None, :, :],
        dynamics.equilibrium_chain_count,
        axis=0,
    )
    sampling_state = initialize_ionic_hrex_state(
        model=interaction_model,
        settings=hrex_settings,
        random_seed=sampling_random_seed,
        initial_positions_by_ladder_m=np.stack(
            [ladder_system.positions_m for ladder_system in relaxed_systems]
        ),
        initial_box_vectors_by_ladder_m=initial_boxes_by_ladder_m,
    )
    resolvent_estimate, equilibrium_samples_per_chain = (
        hamiltonian_green_kubo_sequence(
            model=interaction_model,
            state=sampling_state,
            settings=hrex_settings,
            system=relaxed_system,
            temperature_K=temperature_K,
            dynamics=dynamics,
            numerics=numerics,
            checkpoint_path=initialization_checkpoint_path.with_suffix(
                ".hamiltonian_resolvent.pkl"
            ),
            checkpoint_fingerprint=resolvent_checkpoint_fingerprint,
        )
    )
    conditioned_volume_m3 = float(abs(np.linalg.det(relaxed_system.box_vectors_m)))
    conditioned_density_g_cm3 = float(
        np.sum(relaxed_system.masses_kg)
        / conditioned_volume_m3
        / KG_M3_PER_G_ML
    )
    if equilibrium_samples_per_chain <= 0:
        raise RuntimeError("finite-cell resolvent retained no equilibrium samples")
    return (
        resolvent_estimate,
        recipe_realization,
        conditioned_volume_m3,
        conditioned_density_g_cm3,
    )


def compute_first_principles_conductivity(
    recipe: ElectrolyteRecipeModel,
    temperature_K: float,
    liquid_density_kg_m3: float,
    density_source: str,
    minimum_explicit_molecule_count: int,
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
    random_seed: int,
    initialization_checkpoint_path: Path,
) -> ConductivityResult:
    _validate_settings(dynamics, numerics)
    base_realization = select_integer_recipe_realization(
        recipe=recipe,
        liquid_density_kg_m3=liquid_density_kg_m3,
        solvent_volume_fraction_tolerance=(
            dynamics.solvent_volume_fraction_tolerance
        ),
        salt_molarity_tolerance_mol_L=dynamics.salt_molarity_tolerance_mol_L,
        additive_weight_fraction_tolerance=(
            dynamics.additive_weight_fraction_tolerance
        ),
        minimum_explicit_molecule_count=minimum_explicit_molecule_count,
        maximum_explicit_molecule_count=dynamics.maximum_explicit_molecule_count,
        maximum_atom_count=dynamics.maximum_atom_count,
    )
    maximum_molecule_multiplier = (
        dynamics.maximum_explicit_molecule_count
        // base_realization.explicit_molecule_count
    )
    maximum_atom_multiplier = dynamics.maximum_atom_count // base_realization.atom_count
    maximum_length_multiplier = int(
        math.floor(
            (
                numerics.ewald_maximum_box_length_m**3
                / base_realization.density_conditioned_volume_m3
            )
        )
    )
    cell_multiplier_count = min(
        maximum_molecule_multiplier,
        maximum_atom_multiplier,
        maximum_length_multiplier,
    )
    if cell_multiplier_count < 3:
        raise ValueError(
            "bulk Green-Kubo continuation requires three composition-preserving "
            "cells within the configured molecule, atom, and Ewald limits"
        )
    cell_results: list[
        tuple[
            HamiltonianResolventEstimate,
            IntegerRecipeRealization,
            float,
            float,
        ]
    ] = []
    for cell_multiplier in range(1, cell_multiplier_count + 1):
        scaled_realization = IntegerRecipeRealization(
            formula_unit_counts=tuple(
                (species_name, count * cell_multiplier)
                for species_name, count in base_realization.formula_unit_counts
            ),
            explicit_species_counts=tuple(
                (species_name, count * cell_multiplier)
                for species_name, count in base_realization.explicit_species_counts
            ),
            explicit_molecule_count=(
                base_realization.explicit_molecule_count * cell_multiplier
            ),
            atom_count=base_realization.atom_count * cell_multiplier,
            cell_mass_kg=base_realization.cell_mass_kg * cell_multiplier,
            density_conditioned_volume_m3=(
                base_realization.density_conditioned_volume_m3 * cell_multiplier
            ),
            realized_solvent_volume_fractions=(
                base_realization.realized_solvent_volume_fractions
            ),
            realized_salt_molarities_mol_L=(
                base_realization.realized_salt_molarities_mol_L
            ),
            realized_additive_weight_fractions=(
                base_realization.realized_additive_weight_fractions
            ),
            native_unit_deviations=base_realization.native_unit_deviations,
        )
        cell_seed = int(
            np.random.SeedSequence((random_seed, cell_multiplier)).generate_state(1)[0]
        )
        cell_checkpoint_path = initialization_checkpoint_path.with_name(
            f"{initialization_checkpoint_path.stem}.cell_{cell_multiplier}"
            f"{initialization_checkpoint_path.suffix}"
        )
        cell_result = _compute_finite_cell_resolvent(
            recipe=recipe,
            recipe_realization=scaled_realization,
            temperature_K=temperature_K,
            liquid_density_kg_m3=liquid_density_kg_m3,
            density_source=density_source,
            dynamics=dynamics,
            numerics=numerics,
            random_seed=cell_seed,
            initialization_checkpoint_path=cell_checkpoint_path,
        )
        cell_results.append(cell_result)
        if not cell_result[0].finite_eta_precision_reached:
            raise RuntimeError(
                "finite-cell Green-Kubo basis and equilibrium budget ended before "
                "the full-residual interval reached the configured tolerance: "
                f"cell_multiplier={cell_multiplier}, "
                f"interval_width_S_m="
                f"{cell_result[0].upper_bound_S_m - cell_result[0].lower_bound_S_m:.12g}, "
                f"tolerance_S_m={numerics.conductivity_tolerance_S_m:.12g}"
            )
    inverse_box_lengths_per_m = np.asarray(
        [cell_result[2] ** (-1.0 / CARTESIAN_DIMENSION) for cell_result in cell_results]
    )
    bulk_eta_intervals_S_m: list[tuple[float, float, float]] = []
    finite_cell_errors_S_m: list[float] = []
    for eta_index, eta_s_inv in enumerate(numerics.resolvent_eta_values_s_inv):
        (
            bulk_eta_iterate_S_m,
            bulk_eta_lower_bound_S_m,
            bulk_eta_upper_bound_S_m,
            finite_cell_error_S_m,
        ) = _zero_coordinate_limit_interval(
            positive_coordinates=inverse_box_lengths_per_m,
            lower_values=np.asarray(
                [
                    cell_result[0].intervals_S_m[eta_index][1]
                    for cell_result in cell_results
                ]
            ),
            upper_values=np.asarray(
                [
                    cell_result[0].intervals_S_m[eta_index][2]
                    for cell_result in cell_results
                ]
            ),
        )
        if not (
            bulk_eta_lower_bound_S_m
            <= bulk_eta_iterate_S_m
            <= bulk_eta_upper_bound_S_m
        ):
            raise RuntimeError("bulk eta continuation produced an invalid interval")
        bulk_eta_intervals_S_m.append(
            (
                eta_s_inv,
                bulk_eta_lower_bound_S_m,
                bulk_eta_upper_bound_S_m,
            )
        )
        finite_cell_errors_S_m.append(finite_cell_error_S_m)
    (
        conductivity_S_m,
        conductivity_lower_bound_S_m,
        conductivity_upper_bound_S_m,
        eta_continuation_error_S_m,
    ) = _zero_coordinate_limit_interval(
        positive_coordinates=np.asarray(numerics.resolvent_eta_values_s_inv),
        lower_values=np.asarray(
            [interval[1] for interval in bulk_eta_intervals_S_m]
        ),
        upper_values=np.asarray(
            [interval[2] for interval in bulk_eta_intervals_S_m]
        ),
    )
    final_interval_width_S_m = (
        conductivity_upper_bound_S_m - conductivity_lower_bound_S_m
    )
    if final_interval_width_S_m > numerics.conductivity_tolerance_S_m:
        raise RuntimeError(
            "bulk dc Green-Kubo sequence ended before the configured tolerance: "
            f"iterate_S_m={conductivity_S_m:.12g}, "
            f"lower_S_m={conductivity_lower_bound_S_m:.12g}, "
            f"upper_S_m={conductivity_upper_bound_S_m:.12g}, "
            f"tolerance_S_m={numerics.conductivity_tolerance_S_m:.12g}"
        )
    largest_cell_estimate, largest_realization, largest_volume_m3, largest_density = (
        cell_results[-1]
    )
    return ConductivityResult(
        conductivity_S_m=conductivity_S_m,
        conductivity_lower_bound_S_m=conductivity_lower_bound_S_m,
        conductivity_upper_bound_S_m=conductivity_upper_bound_S_m,
        conditioned_volume_m3=largest_volume_m3,
        conditioned_density_g_cm3=largest_density,
        thermodynamic_state="bulk dc Green-Kubo limit at imposed NVT density",
        density_source=density_source,
        generator_name="Hamiltonian Liouville",
        current_definition=(
            "Liouville derivative of permanent atomic plus induced-dipole "
            "polarization"
        ),
        interval_definition=(
            "cross-fitted full-residual nested Galerkin intervals with inverse-box-"
            "length and zero-frequency continuation"
        ),
        interval_is_deterministic=False,
        basis_size=largest_cell_estimate.basis_size,
        basis_labels=largest_cell_estimate.basis_labels,
        resolvent_intervals_S_m=tuple(bulk_eta_intervals_S_m),
        cell_conductivities_S_m=tuple(
            (
                cell_result[2],
                cell_result[0].resolvent_iterate_S_m,
                cell_result[0].lower_bound_S_m,
                cell_result[0].upper_bound_S_m,
            )
            for cell_result in cell_results
        ),
        basis_error_S_m=max(
            cell_result[0].basis_error_S_m for cell_result in cell_results
        ),
        eta_continuation_error_S_m=eta_continuation_error_S_m,
        finite_cell_error_S_m=max(finite_cell_errors_S_m),
        equilibrium_error_S_m=max(
            cell_result[0].equilibrium_error_S_m for cell_result in cell_results
        ),
        linear_solve_error_S_m=max(
            cell_result[0].linear_solve_error_S_m for cell_result in cell_results
        ),
        linear_solve_relative_residual=max(
            cell_result[0].linear_solve_relative_residual
            for cell_result in cell_results
        ),
        equilibrium_sample_count=(
            dynamics.equilibrium_chain_count
            * dynamics.equilibrium_sample_count
            * dynamics.equilibrium_maximum_refinement_batches
            * len(cell_results)
        ),
        equilibrium_chain_count=(
            dynamics.equilibrium_chain_count * len(cell_results)
        ),
        conductivity_mcse_S_m=max(
            cell_result[0].conductivity_mcse_S_m for cell_result in cell_results
        ),
        finite_eta_resolvent_precision_reached=True,
        conductivity_precision_reached=True,
        realized_formula_unit_counts=largest_realization.formula_unit_counts,
        realized_molecule_counts=largest_realization.explicit_species_counts,
        realized_atom_count=largest_realization.atom_count,
        realized_solvent_volume_fractions=(
            largest_realization.realized_solvent_volume_fractions
        ),
        realized_salt_molarities_mol_L=(
            largest_realization.realized_salt_molarities_mol_L
        ),
        realized_additive_weight_fractions=(
            largest_realization.realized_additive_weight_fractions
        ),
        realized_native_unit_deviations=largest_realization.native_unit_deviations,
    )


@dataclass(frozen=True)
class BatchedHmcTransitionResult:
    positions_m: Array
    forces_N: Array
    momenta_kg_m_s: Array
    accepted: Array
    log_acceptance_probabilities: Array
    energy_errors_over_kbt: Array
    component_energies_J: Array
    force_evaluation_count: int


@dataclass(frozen=True)
class IonicHrexSettings:
    lambdas: tuple[float, ...]
    hmc_step_size_s: float
    hmc_steps_min: int
    hmc_steps_max: int
    hmc_momentum_persistence: float
    hmc_full_refresh_stride: int
    exchange_stride: int
    independent_ladder_count: int
    warmup_cycle_count: int
    measurement_stride: int
    block_cycle_count: int
    force_batch_size: int


@dataclass
class IonicHrexState:
    positions_m: Array
    boxes_m: Array
    component_energies_J: Array
    momenta_kg_m_s: Array
    momentum_refresh_required: Array
    auxiliary_masses_kg: Array
    walker_identifiers: Array
    visited_lowest_lambda: Array
    completed_round_trips: Array
    round_trip_phase: Array
    hmc_step_sizes_s: Array
    hmc_attempts: Array
    hmc_acceptances: Array
    hmc_expected_acceptance_sums: Array
    hmc_absolute_energy_error_over_kbt_sums: Array
    hmc_molecular_com_squared_displacement_sums_m2: Array
    exchange_attempts: Array
    exchange_acceptances: Array
    exchange_expected_acceptance_sums: Array
    cycle_index: int
    random_generator_state: dict


@dataclass(frozen=True)
class IonicHrexBlock:
    physical_configurations_m: Array
    physical_momenta_kg_m_s: Array
    physical_forces_N: Array
    physical_box_vectors_by_sample_m: Array
    physical_ladder_indices: Array
    cycle_count: int
    force_evaluation_count: int
    hmc_expected_acceptance_by_cycle_and_state: Array
    hmc_realized_acceptance_by_cycle_and_state: Array
    hmc_absolute_energy_error_over_kbt_by_cycle_and_state: Array
    hmc_molecular_com_squared_displacement_m2_by_cycle_and_state: Array


@dataclass(frozen=True)
class IonicMolecularSystem(Protocol):
    positions_m: Array
    box_vectors_m: Array
    masses_kg: Array
    charges_C: Array
    molecule_atom_indices: tuple[Array, ...]
    molecule_species_names: tuple[str, ...]


class IonicTemperedModel(Protocol):
    system: IonicMolecularSystem

    def energy_force_components_batch(
        self,
        positions_batch_m: Array,
        box_vectors_batch_m: Array,
        lambda_values: Array,
    ): ...

    def energy_components_batch(
        self,
        positions_batch_m: Array,
        box_vectors_batch_m: Array,
    ): ...


def batched_hmc_transition(
    model: IonicTemperedModel,
    positions_batch_m: Array,
    box_vectors_batch_m: Array,
    momenta_kg_m_s: Array,
    auxiliary_masses_kg: Array,
    momentum_refresh_required: Array,
    momentum_persistence: float,
    temperature_K: float,
    lambda_values: Array,
    timestep_values_s: Array,
    integration_step_counts: Array,
    force_batch_size: int,
    random_generator: np.random.Generator,
) -> BatchedHmcTransitionResult:
    batch_size = positions_batch_m.shape[0]
    if force_batch_size <= 0:
        raise ValueError("HMC force batch size must be positive")

    force_evaluation_count = 0

    def evaluate_energy_force_components(
        evaluated_positions_batch_m: Array,
    ) -> BatchedHamiltonianResult:
        nonlocal force_evaluation_count
        chunk_results: list[BatchedHamiltonianResult] = []
        for chunk_start in range(0, batch_size, force_batch_size):
            chunk_stop = min(chunk_start + force_batch_size, batch_size)
            chunk_results.append(
                model.energy_force_components_batch(
                    positions_batch_m=evaluated_positions_batch_m[
                        chunk_start:chunk_stop
                    ],
                    box_vectors_batch_m=box_vectors_batch_m[chunk_start:chunk_stop],
                    lambda_values=lambda_values[chunk_start:chunk_stop],
                )
            )
            force_evaluation_count += 1
        return BatchedHamiltonianResult(
            energy_J=np.concatenate(tuple(result.energy_J for result in chunk_results)),
            forces_N=np.concatenate(tuple(result.forces_N for result in chunk_results)),
            fixed_energy_J=np.concatenate(
                tuple(result.fixed_energy_J for result in chunk_results)
            ),
            ion_ion_energy_J=np.concatenate(
                tuple(result.ion_ion_energy_J for result in chunk_results)
            ),
            ion_neutral_energy_J=np.concatenate(
                tuple(result.ion_neutral_energy_J for result in chunk_results)
            ),
            polarization_residual_V_m=np.concatenate(
                tuple(result.polarization_residual_V_m for result in chunk_results)
            ),
        )

    thermal_momentum_scale_kg_m_s = np.sqrt(K_B * temperature_K * auxiliary_masses_kg)
    gaussian_momenta_kg_m_s = (
        random_generator.normal(size=positions_batch_m.shape)
        * thermal_momentum_scale_kg_m_s[None, :, None]
    )
    persistence = np.where(
        momentum_refresh_required,
        0.0,
        momentum_persistence,
    )
    initial_momenta_kg_m_s = (
        persistence[:, None, None] * momenta_kg_m_s
        + np.sqrt(1.0 - persistence**2)[:, None, None] * gaussian_momenta_kg_m_s
    )
    initial_velocities_m_s = initial_momenta_kg_m_s / auxiliary_masses_kg[None, :, None]
    total_mass_kg = float(np.sum(auxiliary_masses_kg))
    center_of_mass_velocities_m_s = (
        np.sum(auxiliary_masses_kg[None, :, None] * initial_velocities_m_s, axis=1)
        / total_mass_kg
    )
    initial_velocities_m_s -= center_of_mass_velocities_m_s[:, None, :]
    proposed_positions_m = positions_batch_m.copy()
    proposed_velocities_m_s = initial_velocities_m_s.copy()
    initial_state = evaluate_energy_force_components(proposed_positions_m)
    initial_kinetic_energies_J = 0.5 * np.sum(
        auxiliary_masses_kg[None, :, None] * initial_velocities_m_s**2,
        axis=(1, 2),
    )
    current_forces_N = initial_state.forces_N
    final_state = initial_state
    maximum_step_count = int(np.max(integration_step_counts))
    for integration_step_index in range(maximum_step_count):
        active = integration_step_index < integration_step_counts
        active_vector = active[:, None, None]
        proposed_velocities_m_s += active_vector * (
            0.5
            * timestep_values_s[:, None, None]
            * current_forces_N
            / auxiliary_masses_kg[None, :, None]
        )
        proposed_positions_m += active_vector * (
            timestep_values_s[:, None, None] * proposed_velocities_m_s
        )
        box_lengths_m = np.diagonal(box_vectors_batch_m, axis1=1, axis2=2)
        proposed_positions_m %= box_lengths_m[:, None, :]
        next_state = evaluate_energy_force_components(proposed_positions_m)
        proposed_velocities_m_s += active_vector * (
            0.5
            * timestep_values_s[:, None, None]
            * next_state.forces_N
            / auxiliary_masses_kg[None, :, None]
        )
        current_forces_N = np.where(
            active_vector, next_state.forces_N, current_forces_N
        )
        final_state = next_state
    final_kinetic_energies_J = 0.5 * np.sum(
        auxiliary_masses_kg[None, :, None] * proposed_velocities_m_s**2,
        axis=(1, 2),
    )
    energy_errors_J = (
        final_state.energy_J
        + final_kinetic_energies_J
        - initial_state.energy_J
        - initial_kinetic_energies_J
    )
    log_acceptance_probabilities = np.minimum(
        0.0, -energy_errors_J / (K_B * temperature_K)
    )
    accepted = (
        np.log(random_generator.random(batch_size)) < log_acceptance_probabilities
    )
    accepted_positions_m = np.where(
        accepted[:, None, None], proposed_positions_m, positions_batch_m
    )
    accepted_forces_N = np.where(
        accepted[:, None, None], final_state.forces_N, initial_state.forces_N
    )
    proposed_momenta_kg_m_s = (
        auxiliary_masses_kg[None, :, None] * proposed_velocities_m_s
    )
    retained_momenta_kg_m_s = np.where(
        accepted[:, None, None],
        proposed_momenta_kg_m_s,
        -initial_momenta_kg_m_s,
    )
    initial_components_J = np.stack(
        (
            initial_state.fixed_energy_J,
            initial_state.ion_ion_energy_J,
            initial_state.ion_neutral_energy_J,
        ),
        axis=1,
    )
    final_components_J = np.stack(
        (
            final_state.fixed_energy_J,
            final_state.ion_ion_energy_J,
            final_state.ion_neutral_energy_J,
        ),
        axis=1,
    )
    retained_components_J = np.where(
        accepted[:, None], final_components_J, initial_components_J
    )
    return BatchedHmcTransitionResult(
        positions_m=accepted_positions_m,
        forces_N=accepted_forces_N,
        momenta_kg_m_s=retained_momenta_kg_m_s,
        accepted=accepted,
        log_acceptance_probabilities=log_acceptance_probabilities,
        energy_errors_over_kbt=energy_errors_J / (K_B * temperature_K),
        component_energies_J=retained_components_J,
        force_evaluation_count=force_evaluation_count,
    )


def _tempered_energy_from_components_J(
    components_J: Array, lambda_value: float
) -> float:
    fixed_energy_J, ion_ion_energy_J, ion_neutral_energy_J = components_J
    return float(
        fixed_energy_J
        + lambda_value * ion_ion_energy_J
        + math.sqrt(lambda_value) * ion_neutral_energy_J
    )


def replica_exchange_log_acceptance(
    first_components_J: Array,
    second_components_J: Array,
    first_lambda: float,
    second_lambda: float,
    beta_per_J: float,
) -> float:
    if not 0.0 < first_lambda <= 1.0 or not 0.0 < second_lambda <= 1.0:
        raise ValueError("replica lambdas must be in (0, 1]")
    current_energy_J = _tempered_energy_from_components_J(
        first_components_J, first_lambda
    ) + _tempered_energy_from_components_J(second_components_J, second_lambda)
    exchanged_energy_J = _tempered_energy_from_components_J(
        second_components_J, first_lambda
    ) + _tempered_energy_from_components_J(first_components_J, second_lambda)
    return float(-beta_per_J * (exchanged_energy_J - current_energy_J))


def _validate_ionic_hrex_settings(settings: IonicHrexSettings) -> None:
    if not settings.lambdas or settings.lambdas[0] != 1.0:
        raise ValueError("ionic sampling lambdas must start at lambda=1")
    if any(
        lambda_value <= 0.0 or lambda_value > 1.0 for lambda_value in settings.lambdas
    ):
        raise ValueError("HREX lambdas must be in (0, 1]")
    if any(
        settings.lambdas[index] <= settings.lambdas[index + 1]
        for index in range(len(settings.lambdas) - 1)
    ):
        raise ValueError("HREX lambdas must be strictly decreasing")
    positive_integer_settings = (
        settings.hmc_steps_min,
        settings.hmc_steps_max,
        settings.hmc_full_refresh_stride,
        settings.exchange_stride,
        settings.independent_ladder_count,
        settings.warmup_cycle_count,
        settings.measurement_stride,
        settings.block_cycle_count,
        settings.force_batch_size,
    )
    if any(value <= 0 for value in positive_integer_settings):
        raise ValueError("HREX counts and strides must be positive")
    if settings.hmc_steps_min > settings.hmc_steps_max:
        raise ValueError("HREX minimum HMC steps exceed maximum HMC steps")
    if settings.hmc_step_size_s <= 0.0:
        raise ValueError("HREX HMC timestep must be positive")
    if not 0.0 <= settings.hmc_momentum_persistence < 1.0:
        raise ValueError("HREX momentum persistence must lie in [0, 1)")


def _run_exchange_round(
    replica_positions_m: Array,
    replica_box_vectors_m: Array,
    component_arrays_J: Array,
    replica_momenta_kg_m_s: Array,
    replica_forces_N: Array,
    lambdas: tuple[float, ...],
    beta_per_J: float,
    exchange_offset: int,
    random_generator: np.random.Generator,
    exchange_attempts: Array,
    exchange_acceptances: Array,
    exchange_expected_acceptance_sums: Array,
    walker_identifiers: Array,
) -> None:
    for first_replica in range(exchange_offset, len(lambdas) - 1, 2):
        second_replica = first_replica + 1
        exchange_attempts[first_replica] += 1
        log_acceptance = replica_exchange_log_acceptance(
            first_components_J=component_arrays_J[first_replica],
            second_components_J=component_arrays_J[second_replica],
            first_lambda=lambdas[first_replica],
            second_lambda=lambdas[second_replica],
            beta_per_J=beta_per_J,
        )
        if not math.isfinite(log_acceptance):
            raise FloatingPointError("replica exchange produced nonfinite acceptance")
        exchange_expected_acceptance_sums[first_replica] += math.exp(
            min(0.0, log_acceptance)
        )
        if math.log(random_generator.random()) < min(0.0, log_acceptance):
            replica_positions_m[[first_replica, second_replica]] = replica_positions_m[
                [second_replica, first_replica]
            ]
            replica_box_vectors_m[[first_replica, second_replica]] = (
                replica_box_vectors_m[[second_replica, first_replica]]
            )
            component_arrays_J[[first_replica, second_replica]] = component_arrays_J[
                [second_replica, first_replica]
            ]
            replica_momenta_kg_m_s[[first_replica, second_replica]] = (
                replica_momenta_kg_m_s[[second_replica, first_replica]]
            )
            replica_forces_N[[first_replica, second_replica]] = replica_forces_N[
                [second_replica, first_replica]
            ]
            exchanged_walker_identifier = walker_identifiers[first_replica]
            walker_identifiers[first_replica] = walker_identifiers[second_replica]
            walker_identifiers[second_replica] = exchanged_walker_identifier
            exchange_acceptances[first_replica] += 1


def _component_array_from_result(result) -> Array:
    return np.stack(
        (
            result.fixed_energy_J,
            result.ion_ion_energy_J,
            result.ion_neutral_energy_J,
        ),
        axis=1,
    )


def initialize_ionic_hrex_state(
    model: IonicTemperedModel,
    settings: IonicHrexSettings,
    random_seed: int,
    initial_positions_by_ladder_m: Array,
    initial_box_vectors_by_ladder_m: Array,
) -> IonicHrexState:
    _validate_ionic_hrex_settings(settings)
    ladder_count = settings.independent_ladder_count
    replica_count = len(settings.lambdas)
    batch_size = ladder_count * replica_count
    initial_positions = np.asarray(initial_positions_by_ladder_m, dtype=float)
    initial_boxes = np.asarray(initial_box_vectors_by_ladder_m, dtype=float)
    expected_position_shape = (
        ladder_count,
        model.system.positions_m.shape[0],
        CARTESIAN_DIMENSION,
    )
    expected_box_shape = (
        ladder_count,
        CARTESIAN_DIMENSION,
        CARTESIAN_DIMENSION,
    )
    if initial_positions.shape != expected_position_shape:
        raise ValueError("initial ladder positions have incompatible shape")
    if initial_boxes.shape != expected_box_shape:
        raise ValueError("initial ladder boxes have incompatible shape")
    positions_batch_m = np.repeat(initial_positions, replica_count, axis=0)
    box_vectors_batch_m = np.repeat(initial_boxes, replica_count, axis=0)
    initial_components = model.energy_components_batch(
        positions_batch_m=positions_batch_m,
        box_vectors_batch_m=box_vectors_batch_m,
    )
    random_generator = np.random.default_rng(random_seed)
    return IonicHrexState(
        positions_m=positions_batch_m,
        boxes_m=box_vectors_batch_m,
        component_energies_J=_component_array_from_result(initial_components),
        momenta_kg_m_s=np.zeros_like(positions_batch_m),
        momentum_refresh_required=np.ones(batch_size, dtype=bool),
        auxiliary_masses_kg=np.asarray(model.system.masses_kg, dtype=float).copy(),
        walker_identifiers=np.tile(np.arange(replica_count), (ladder_count, 1)),
        visited_lowest_lambda=np.zeros((ladder_count, replica_count), dtype=bool),
        completed_round_trips=np.zeros((ladder_count, replica_count), dtype=int),
        round_trip_phase=np.pad(
            np.ones((ladder_count, 1), dtype=np.int8),
            ((0, 0), (0, replica_count - 1)),
        ),
        hmc_step_sizes_s=np.full(batch_size, settings.hmc_step_size_s),
        hmc_attempts=np.zeros((ladder_count, replica_count), dtype=int),
        hmc_acceptances=np.zeros((ladder_count, replica_count), dtype=int),
        hmc_expected_acceptance_sums=np.zeros((ladder_count, replica_count)),
        hmc_absolute_energy_error_over_kbt_sums=np.zeros((ladder_count, replica_count)),
        hmc_molecular_com_squared_displacement_sums_m2=np.zeros(
            (ladder_count, replica_count)
        ),
        exchange_attempts=np.zeros((ladder_count, replica_count - 1), dtype=int),
        exchange_acceptances=np.zeros((ladder_count, replica_count - 1), dtype=int),
        exchange_expected_acceptance_sums=np.zeros((ladder_count, replica_count - 1)),
        cycle_index=0,
        random_generator_state=random_generator.bit_generator.state,
    )


def _update_hrex_round_trip_tracking(state: IonicHrexState) -> None:
    ladder_count, replica_count = state.walker_identifiers.shape
    for ladder_index in range(ladder_count):
        lowest_walker = state.walker_identifiers[ladder_index, replica_count - 1]
        if state.round_trip_phase[ladder_index, lowest_walker] == 1:
            state.round_trip_phase[ladder_index, lowest_walker] = 2
            state.visited_lowest_lambda[ladder_index, lowest_walker] = True
        physical_walker = state.walker_identifiers[ladder_index, 0]
        phase = state.round_trip_phase[ladder_index, physical_walker]
        if phase == 0:
            state.round_trip_phase[ladder_index, physical_walker] = 1
            continue
        if phase == 2:
            state.completed_round_trips[ladder_index, physical_walker] += 1
            state.round_trip_phase[ladder_index, physical_walker] = 1
            state.visited_lowest_lambda[ladder_index, physical_walker] = False


def advance_ionic_hrex(
    model: IonicTemperedModel,
    state: IonicHrexState,
    settings: IonicHrexSettings,
    temperature_K: float,
    cycle_count: int,
    attempt_exchange: bool,
    retain_samples: bool,
) -> tuple[IonicHrexState, IonicHrexBlock]:
    if cycle_count <= 0:
        raise ValueError("HREX advance cycle count must be positive")
    random_generator = np.random.default_rng()
    random_generator.bit_generator.state = state.random_generator_state
    ladder_count = settings.independent_ladder_count
    replica_count = len(settings.lambdas)
    batch_size = ladder_count * replica_count
    lambda_values = np.tile(np.asarray(settings.lambdas), ladder_count)
    beta_per_J = 1.0 / (K_B * temperature_K)
    physical_indices = np.arange(ladder_count) * replica_count
    configurations: list[Array] = []
    retained_momenta_kg_m_s: list[Array] = []
    retained_forces_N: list[Array] = []
    boxes: list[Array] = []
    ladder_indices: list[int] = []
    force_evaluation_count = 0
    expected_hmc_acceptance_rows: list[Array] = []
    realized_hmc_acceptance_rows: list[Array] = []
    absolute_energy_error_rows: list[Array] = []
    molecular_com_squared_displacement_rows_m2: list[Array] = []
    for block_cycle_index in range(cycle_count):
        absolute_cycle_index = state.cycle_index + block_cycle_index + 1
        integration_step_count = int(
            random_generator.integers(
                settings.hmc_steps_min, settings.hmc_steps_max + 1
            )
        )
        integration_steps = np.full(batch_size, integration_step_count, dtype=int)
        previous_positions_m = state.positions_m.copy()
        transition = batched_hmc_transition(
            model=model,
            positions_batch_m=state.positions_m,
            box_vectors_batch_m=state.boxes_m,
            momenta_kg_m_s=state.momenta_kg_m_s,
            auxiliary_masses_kg=state.auxiliary_masses_kg,
            momentum_refresh_required=(
                state.momentum_refresh_required
                | (absolute_cycle_index % settings.hmc_full_refresh_stride == 0)
            ),
            momentum_persistence=settings.hmc_momentum_persistence,
            temperature_K=temperature_K,
            lambda_values=lambda_values,
            timestep_values_s=state.hmc_step_sizes_s,
            integration_step_counts=integration_steps,
            force_batch_size=settings.force_batch_size,
            random_generator=random_generator,
        )
        force_evaluation_count += transition.force_evaluation_count
        state.positions_m = transition.positions_m
        state.momenta_kg_m_s = transition.momenta_kg_m_s
        state.momentum_refresh_required.fill(False)
        state.component_energies_J = transition.component_energies_J
        current_forces_N = transition.forces_N.copy()
        state.hmc_attempts += 1
        state.hmc_acceptances += transition.accepted.reshape(
            ladder_count, replica_count
        )
        expected_hmc_acceptance = np.exp(transition.log_acceptance_probabilities)
        expected_hmc_acceptance_rows.append(expected_hmc_acceptance.copy())
        realized_hmc_acceptance_rows.append(transition.accepted.copy())
        absolute_energy_error_rows.append(
            np.abs(transition.energy_errors_over_kbt).copy()
        )
        state.hmc_expected_acceptance_sums += expected_hmc_acceptance.reshape(
            ladder_count, replica_count
        )
        state.hmc_absolute_energy_error_over_kbt_sums += np.abs(
            transition.energy_errors_over_kbt
        ).reshape(ladder_count, replica_count)
        atomic_displacements_m = transition.positions_m - previous_positions_m
        for batch_index in range(batch_size):
            atomic_displacements_m[batch_index] = minimum_image_displacement(
                atomic_displacements_m[batch_index],
                state.boxes_m[batch_index],
            )
        molecular_com_displacements_m = np.stack(
            tuple(
                np.average(
                    atomic_displacements_m[:, molecule_atom_indices],
                    axis=1,
                    weights=model.system.masses_kg[molecule_atom_indices],
                )
                for molecule_atom_indices in model.system.molecule_atom_indices
            ),
            axis=1,
        )
        mean_molecular_com_squared_displacements_m2 = np.mean(
            np.sum(molecular_com_displacements_m**2, axis=2),
            axis=1,
        )
        state.hmc_molecular_com_squared_displacement_sums_m2 += (
            mean_molecular_com_squared_displacements_m2.reshape(
                ladder_count, replica_count
            )
        )
        molecular_com_squared_displacement_rows_m2.append(
            mean_molecular_com_squared_displacements_m2.copy()
        )
        if attempt_exchange and absolute_cycle_index % settings.exchange_stride == 0:
            exchange_offsets = (0,) if replica_count == 2 else (0, 1)
            for ladder_index in range(ladder_count):
                ladder_slice = slice(
                    ladder_index * replica_count,
                    (ladder_index + 1) * replica_count,
                )
                for exchange_offset in exchange_offsets:
                    _run_exchange_round(
                        state.positions_m[ladder_slice],
                        state.boxes_m[ladder_slice],
                        state.component_energies_J[ladder_slice],
                        state.momenta_kg_m_s[ladder_slice],
                        current_forces_N[ladder_slice],
                        settings.lambdas,
                        beta_per_J,
                        exchange_offset,
                        random_generator,
                        state.exchange_attempts[ladder_index],
                        state.exchange_acceptances[ladder_index],
                        state.exchange_expected_acceptance_sums[ladder_index],
                        state.walker_identifiers[ladder_index],
                    )
        if replica_count > 1:
            _update_hrex_round_trip_tracking(state)
        if retain_samples and absolute_cycle_index % settings.measurement_stride == 0:
            physical_forces_by_ladder_N = current_forces_N[physical_indices]
            if replica_count > 1:
                physical_force_result = model.energy_force_components_batch(
                    positions_batch_m=state.positions_m[physical_indices],
                    box_vectors_batch_m=state.boxes_m[physical_indices],
                    lambda_values=np.ones(ladder_count),
                )
                physical_forces_by_ladder_N = physical_force_result.forces_N
                force_evaluation_count += 1
            for ladder_index, physical_index in enumerate(physical_indices):
                configurations.append(state.positions_m[physical_index].copy())
                retained_momenta_kg_m_s.append(
                    state.momenta_kg_m_s[physical_index].copy()
                )
                retained_forces_N.append(
                    physical_forces_by_ladder_N[ladder_index].copy()
                )
                boxes.append(state.boxes_m[physical_index].copy())
                ladder_indices.append(ladder_index)
    state.cycle_index += cycle_count
    state.random_generator_state = random_generator.bit_generator.state
    return state, IonicHrexBlock(
        physical_configurations_m=np.asarray(configurations),
        physical_momenta_kg_m_s=np.asarray(retained_momenta_kg_m_s),
        physical_forces_N=np.asarray(retained_forces_N),
        physical_box_vectors_by_sample_m=np.asarray(boxes),
        physical_ladder_indices=np.asarray(ladder_indices, dtype=int),
        cycle_count=cycle_count,
        force_evaluation_count=force_evaluation_count,
        hmc_expected_acceptance_by_cycle_and_state=np.asarray(
            expected_hmc_acceptance_rows
        ),
        hmc_realized_acceptance_by_cycle_and_state=np.asarray(
            realized_hmc_acceptance_rows
        ),
        hmc_absolute_energy_error_over_kbt_by_cycle_and_state=np.asarray(
            absolute_energy_error_rows
        ),
        hmc_molecular_com_squared_displacement_m2_by_cycle_and_state=np.asarray(
            molecular_com_squared_displacement_rows_m2
        ),
    )


def _settings_from_record(record: dict) -> tuple[DynamicsSettings, NumericalSettings]:
    dynamics_record = record["dynamics"]
    numerics_record = record["numerics"]
    return DynamicsSettings(
        initial_relaxation_maximum_force_evaluations=int(
            dynamics_record["initial_relaxation_maximum_force_evaluations"]
        ),
        initial_relaxation_timestep_s=float(
            dynamics_record["initial_relaxation_timestep_s"]
        ),
        initial_relaxation_maximum_timestep_s=float(
            dynamics_record["initial_relaxation_maximum_timestep_s"]
        ),
        initial_relaxation_initial_damping=float(
            dynamics_record["initial_relaxation_initial_damping"]
        ),
        initial_relaxation_timestep_increase=float(
            dynamics_record["initial_relaxation_timestep_increase"]
        ),
        initial_relaxation_timestep_decrease=float(
            dynamics_record["initial_relaxation_timestep_decrease"]
        ),
        initial_relaxation_damping_decrease=float(
            dynamics_record["initial_relaxation_damping_decrease"]
        ),
        initial_relaxation_positive_power_steps=int(
            dynamics_record["initial_relaxation_positive_power_steps"]
        ),
        initial_relaxation_maximum_elapsed_s=float(
            dynamics_record["initial_relaxation_maximum_elapsed_s"]
        ),
        initial_relaxation_maximum_stagnant_iterations=int(
            dynamics_record["initial_relaxation_maximum_stagnant_iterations"]
        ),
        initial_relaxation_minimum_force_improvement_fraction=float(
            dynamics_record["initial_relaxation_minimum_force_improvement_fraction"]
        ),
        initial_relaxation_progress_stride=int(
            dynamics_record["initial_relaxation_progress_stride"]
        ),
        force_batch_size=int(dynamics_record["force_batch_size"]),
        resolvent_operator_batch_size=int(
            dynamics_record["resolvent_operator_batch_size"]
        ),
        initial_force_tolerance_N=float(
            dynamics_record["initial_force_tolerance_N"]
        ),
        equilibrium_sample_count=int(dynamics_record["equilibrium_sample_count"]),
        equilibrium_chain_count=int(dynamics_record["equilibrium_chain_count"]),
        equilibrium_maximum_refinement_batches=int(
            dynamics_record["equilibrium_maximum_refinement_batches"]
        ),
        hamiltonian_timestep_s=float(dynamics_record["hamiltonian_timestep_s"]),
        ionic_hrex_lambdas=tuple(
            float(value) for value in dynamics_record["ionic_hrex_lambdas"]
        ),
        hmc_steps_min=int(dynamics_record["hmc_steps_min"]),
        hmc_steps_max=int(dynamics_record["hmc_steps_max"]),
        hmc_momentum_persistence=float(
            dynamics_record["hmc_momentum_persistence"]
        ),
        hmc_full_refresh_stride=int(
            dynamics_record["hmc_full_refresh_stride"]
        ),
        exchange_stride=int(dynamics_record["exchange_stride"]),
        hrex_warmup_cycle_count=int(dynamics_record["hrex_warmup_cycle_count"]),
        hrex_measurement_stride=int(dynamics_record["hrex_measurement_stride"]),
        hrex_block_cycle_count=int(dynamics_record["hrex_block_cycle_count"]),
        solvent_volume_fraction_tolerance=float(
            dynamics_record["solvent_volume_fraction_tolerance"]
        ),
        salt_molarity_tolerance_mol_L=float(
            dynamics_record["salt_molarity_tolerance_mol_L"]
        ),
        additive_weight_fraction_tolerance=float(
            dynamics_record["additive_weight_fraction_tolerance"]
        ),
        maximum_explicit_molecule_count=int(
            dynamics_record["maximum_explicit_molecule_count"]
        ),
        maximum_atom_count=int(dynamics_record["maximum_atom_count"]),
    ), NumericalSettings(
        initial_placement_attempts_per_molecule=int(
            numerics_record["initial_placement_attempts_per_molecule"]
        ),
        ewald_splitting_per_m=float(numerics_record["ewald_splitting_per_m"]),
        ewald_reciprocal_relative_tolerance=float(
            numerics_record["ewald_reciprocal_relative_tolerance"]
        ),
        ewald_maximum_box_length_m=float(
            numerics_record["ewald_maximum_box_length_m"]
        ),
        lennard_jones_switch_start_m=float(
            numerics_record["lennard_jones_switch_start_m"]
        ),
        lennard_jones_cutoff_m=float(numerics_record["lennard_jones_cutoff_m"]),
        dispersion_tail_quadrature_order=int(
            numerics_record["dispersion_tail_quadrature_order"]
        ),
        polarization_residual_tolerance_V_m=float(
            numerics_record["polarization_residual_tolerance_V_m"]
        ),
        force_difference_step_m=float(numerics_record["force_difference_step_m"]),
        force_consistency_relative_tolerance=float(
            numerics_record["force_consistency_relative_tolerance"]
        ),
        resolvent_eta_values_s_inv=tuple(
            float(value) for value in numerics_record["resolvent_eta_values_s_inv"]
        ),
        equilibrium_standard_error_multiplier=float(
            numerics_record["equilibrium_standard_error_multiplier"]
        ),
        conductivity_tolerance_S_m=float(
            numerics_record["conductivity_tolerance_S_m"]
        ),
        minimum_interatomic_contact_ratio=float(
            numerics_record["minimum_interatomic_contact_ratio"]
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-json", required=True, type=Path)
    parser.add_argument("--numerics-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    arguments = parser.parse_args()
    recipe = ElectrolyteRecipeModel.model_validate(
        read_json_object(arguments.recipe_json, "electrolyte recipe")
    )
    settings_record = read_json_object(arguments.numerics_json, "conductivity numerics")
    dynamics, numerics = _settings_from_record(settings_record)
    temperature_K = float(settings_record["temperature_K"])
    liquid_density_kg_m3 = float(settings_record["liquid_density_kg_m3"])
    density_source = str(settings_record["density_source"])
    minimum_explicit_molecule_count = int(
        settings_record["minimum_explicit_molecule_count"]
    )
    random_seed = int(settings_record["random_seed"])
    result = compute_first_principles_conductivity(
        recipe=recipe,
        temperature_K=temperature_K,
        liquid_density_kg_m3=liquid_density_kg_m3,
        density_source=density_source,
        minimum_explicit_molecule_count=minimum_explicit_molecule_count,
        dynamics=dynamics,
        numerics=numerics,
        random_seed=random_seed,
        initialization_checkpoint_path=arguments.output_json.with_suffix(
            ".initialization.pkl"
        ),
    )
    write_json_object(arguments.output_json, asdict(result), "conductivity result")
    print(
        f"conductivity = {result.conductivity_S_m:.8g} S/m ({result.conductivity_S_m * S_M_TO_MS_CM:.8g} mS/cm)"
    )
    print(
        "conductivity interval = "
        f"[{result.conductivity_lower_bound_S_m:.8g}, "
        f"{result.conductivity_upper_bound_S_m:.8g}] S/m"
    )
    print(f"thermodynamic state = {result.thermodynamic_state}")
    print(f"density source = {result.density_source}")
    print(f"generator = {result.generator_name}")
    print(f"current = {result.current_definition}")
    print(f"interval definition = {result.interval_definition}")
    print(f"interval is deterministic = {result.interval_is_deterministic}")
    print(f"conditioned volume = {result.conditioned_volume_m3:.8g} m3")
    print(f"conditioned density = {result.conditioned_density_g_cm3:.8g} g/cm3")
    print(f"resolvent intervals = {result.resolvent_intervals_S_m}")
    print(f"cell conductivities = {result.cell_conductivities_S_m}")
    print(
        "realized recipe = "
        f"counts {result.realized_molecule_counts}; "
        f"solvents v/v {result.realized_solvent_volume_fractions}; "
        f"salts mol/L {result.realized_salt_molarities_mol_L}; "
        f"additives wt {result.realized_additive_weight_fractions}"
    )
    print(
        f"basis size = {result.basis_size}; equilibrium samples = "
        f"{result.equilibrium_sample_count}"
        f"; basis error = {result.basis_error_S_m:.6g} S/m"
        f"; conductivity MCSE = {result.conductivity_mcse_S_m:.6g} S/m"
        f"; equilibrium interval expansion = "
        f"{result.equilibrium_error_S_m:.6g} S/m"
        f"; eta continuation error = "
        f"{result.eta_continuation_error_S_m:.6g} S/m"
        f"; finite cell error = {result.finite_cell_error_S_m:.6g} S/m"
        f"; linear solve error = {result.linear_solve_error_S_m:.6g} S/m"
        f"; linear residual = {result.linear_solve_relative_residual:.6g}"
        f"; finite eta resolvent precision reached = "
        f"{result.finite_eta_resolvent_precision_reached}"
        f"; conductivity precision reached = "
        f"{result.conductivity_precision_reached}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
