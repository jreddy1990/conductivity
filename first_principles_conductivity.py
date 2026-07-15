"""Full-configuration reversible conductivity from an analytical molecular model.

The executable constructs one periodic molecular liquid, samples its Boltzmann
measure, and solves the reversible Smoluchowski current-corrector problem in a
nested basis of smooth full-configuration observables.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
from typing import Protocol, runtime_checkable
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from constants import E_CHARGE, EPS_0, K_B, N_A, S_M_TO_MS_CM
from conductivity.physical_library.library_io import load_physical_library
from conductivity.physical_library.physical_objects import (
    LJ_ATTRACTIVE_EXPONENT,
    LJ_REPULSIVE_EXPONENT_MULTIPLIER,
)
from electrolyte_model import ElectrolyteRecipeModel
from species_data import ADDITIVES, SALTS
from utils.strict_validation import read_json_object, write_json_object
from utils.time_series_statistics import linear_fit, select_stationary_suffix

Array = np.ndarray
CARTESIAN_DIMENSION = 3
STOKES_SPHERE_DRAG_FACTOR = 6.0  # Exact no-slip spherical Stokes drag factor.
GAUSSIAN_BLOB_VARIANCE_DENOMINATOR = (
    2.0 * CARTESIAN_DIMENSION
)  # Isotropic three-dimensional sphere form-factor scale.
INITIAL_RELAXATION_FORCE_MARGIN = 0.5  # Resolve below the final force criterion.
TORCH_DTYPE = torch.float64
MILP_FEASIBILITY_TOLERANCE = 100.0 * math.sqrt(np.finfo(float).eps)


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
    initial_relaxation_steps: int
    initial_relaxation_step_m: float
    initial_force_tolerance_N: float
    equilibrium_burn_in_sweeps: int
    equilibrium_sample_count: int
    equilibrium_sweeps_per_sample: int
    translation_proposal_m: float
    rotation_proposal_rad: float
    internal_proposal_m: float
    overdamped_timestep_s: float
    overdamped_steps: int
    overdamped_sample_stride: int
    solvent_viscosity_Pa_s: float
    minimum_overdamped_acceptance_fraction: float


@dataclass(frozen=True)
class NumericalSettings:
    initial_placement_attempts_per_molecule: int
    ewald_splitting_per_m: float
    ewald_reciprocal_shell: int
    polarization_residual_tolerance_V_m: float
    force_difference_step_m: float
    force_consistency_relative_tolerance: float
    basis_radial_count: int
    basis_fourier_shell: int
    basis_radial_cutoff_m: float
    maximum_basis_size: int
    eigenvalue_relative_tolerance: float
    residual_tolerance: float
    conductivity_tolerance_S_m: float
    projected_gk_tolerance_S_m: float
    minimum_effective_sample_size: float
    helfand_fit_start_fraction: float
    helfand_maximum_lag_fraction: float
    gk_noise_window: int
    gk_noise_standard_error_multiplier: float
    minimum_interatomic_contact_ratio: float
    density_relative_tolerance: float
    stationarity_standard_error_limit: float


@dataclass(frozen=True)
class ConductivityResult:
    conductivity_S_m: float
    direct_current_term_S_m: float
    projected_correction_S_m: float
    green_kubo_conductivity_S_m: float
    einstein_helfand_conductivity_S_m: float
    basis_size: int
    basis_conductivities_S_m: tuple[float, ...]
    residual_history: tuple[float, ...]
    maximum_residual_score: float
    sample_count: int
    effective_sample_size: float
    overdamped_acceptance_fraction: float


@dataclass(frozen=True)
class MolecularSystem:
    positions_m: Array
    box_vectors_m: Array
    masses_kg: Array
    charges_C: Array
    lj_sigma_m: Array
    lj_epsilon_J: Array
    hydrodynamic_radii_m: Array
    polarizabilities_SI: Array
    molecule_index: Array
    molecule_atom_indices: tuple[Array, ...]
    bonds: Array
    bond_force_constants_J_m2: Array
    bond_lengths_m: Array
    angles: Array
    angle_force_constants_J_rad2: Array
    angle_values_rad: Array
    torsions: tuple[tuple[int, int, int, int, tuple[tuple[float, int, float], ...]], ...]
    nonbonded_mask: Array


@runtime_checkable
class InteratomicModel(Protocol):
    def energy_J(self, positions_m: Array, box_vectors_m: Array) -> float: ...

    def forces_N(self, positions_m: Array, box_vectors_m: Array) -> Array: ...


@dataclass(frozen=True)
class PeriodicDiffusionOperator:
    """Block-diagonal local diffusion plus periodic transverse Fourier modes."""

    local_blocks_m2_s: Array
    spectral_factor_m_sqrt_s: Array

    def matrix(self) -> Array:
        atom_count = self.local_blocks_m2_s.shape[0]
        diffusion = np.zeros((3 * atom_count, 3 * atom_count))
        for atom_index, local_block in enumerate(self.local_blocks_m2_s):
            atom_slice = slice(3 * atom_index, 3 * atom_index + 3)
            diffusion[atom_slice, atom_slice] = local_block
        return diffusion + self.spectral_factor_m_sqrt_s @ self.spectral_factor_m_sqrt_s.T

    def apply(self, vectors: Array) -> Array:
        vector_array = np.asarray(vectors)
        original_shape = vector_array.shape
        flattened = vector_array.reshape((-1, original_shape[-1]))
        atom_count = self.local_blocks_m2_s.shape[0]
        local = np.einsum(
            "aij,naj->nai",
            self.local_blocks_m2_s,
            flattened.reshape((-1, atom_count, CARTESIAN_DIMENSION)),
        ).reshape(flattened.shape)
        spectral = (
            flattened @ self.spectral_factor_m_sqrt_s
        ) @ self.spectral_factor_m_sqrt_s.T
        return (local + spectral).reshape(original_shape)

    def sample_increment(
        self, random_generator: np.random.Generator
    ) -> Array:
        local_noise = np.concatenate(
            tuple(
                np.linalg.cholesky(local_block)
                @ random_generator.normal(size=CARTESIAN_DIMENSION)
                for local_block in self.local_blocks_m2_s
            )
        )
        spectral_noise = self.spectral_factor_m_sqrt_s @ random_generator.normal(
            size=self.spectral_factor_m_sqrt_s.shape[1]
        )
        return local_noise + spectral_noise

    def solve(self, vector: Array) -> Array:
        reshaped = np.asarray(vector).reshape((-1, CARTESIAN_DIMENSION))
        local_solution_blocks: list[Array] = []
        local_inverse_factor_blocks: list[Array] = []
        factor_blocks = self.spectral_factor_m_sqrt_s.reshape(
            (self.local_blocks_m2_s.shape[0], CARTESIAN_DIMENSION, -1)
        )
        for local_block, vector_block, factor_block in zip(
            self.local_blocks_m2_s,
            reshaped,
            factor_blocks,
            strict=True,
        ):
            local_cholesky = np.linalg.cholesky(local_block)
            local_solution_blocks.append(
                np.linalg.solve(
                    local_cholesky.T,
                    np.linalg.solve(local_cholesky, vector_block),
                )
            )
            local_inverse_factor_blocks.append(
                np.linalg.solve(
                    local_cholesky.T,
                    np.linalg.solve(local_cholesky, factor_block),
                )
            )
        local_solution = np.concatenate(local_solution_blocks)
        local_inverse_factor = np.asarray(
            local_inverse_factor_blocks
        ).reshape(self.spectral_factor_m_sqrt_s.shape)
        reduced_operator = (
            np.eye(self.spectral_factor_m_sqrt_s.shape[1])
            + self.spectral_factor_m_sqrt_s.T @ local_inverse_factor
        )
        reduced_rhs = self.spectral_factor_m_sqrt_s.T @ local_solution
        return local_solution - local_inverse_factor @ np.linalg.solve(
            reduced_operator, reduced_rhs
        )

    def log_determinant(self) -> float:
        local_log_determinant = 0.0
        whitened_factor_blocks: list[Array] = []
        factor_blocks = self.spectral_factor_m_sqrt_s.reshape(
            (self.local_blocks_m2_s.shape[0], CARTESIAN_DIMENSION, -1)
        )
        for local_block, factor_block in zip(
            self.local_blocks_m2_s, factor_blocks, strict=True
        ):
            local_cholesky = np.linalg.cholesky(local_block)
            local_log_determinant += 2.0 * float(
                np.sum(np.log(np.diag(local_cholesky)))
            )
            whitened_factor_blocks.append(
                np.linalg.solve(local_cholesky, factor_block)
            )
        whitened_factor = np.asarray(whitened_factor_blocks).reshape(
            self.spectral_factor_m_sqrt_s.shape
        )
        reduced_operator = (
            np.eye(self.spectral_factor_m_sqrt_s.shape[1])
            + whitened_factor.T @ whitened_factor
        )
        reduced_cholesky = np.linalg.cholesky(reduced_operator)
        return local_log_determinant + 2.0 * float(
            np.sum(np.log(np.diag(reduced_cholesky)))
        )


def minimum_image_displacement(displacement_m: Array, box_vectors_m: Array) -> Array:
    fractional = np.asarray(displacement_m) @ np.linalg.inv(box_vectors_m)
    fractional -= np.rint(fractional)
    return fractional @ box_vectors_m


def total_charge_current_density_A_m2(
    velocities_m_s: Array, charges_C: Array, volume_m3: float
) -> Array:
    return np.sum(charges_C[:, None] * velocities_m_s, axis=0) / volume_m3


def molecular_com_current_density_A_m2(
    velocities_m_s: Array, system: MolecularSystem
) -> Array:
    volume_m3 = abs(np.linalg.det(system.box_vectors_m))
    current_density_A_m2 = np.zeros(CARTESIAN_DIMENSION)
    for molecule_atom_indices in system.molecule_atom_indices:
        molecule_masses_kg = system.masses_kg[molecule_atom_indices]
        center_of_mass_velocity_m_s = np.average(
            velocities_m_s[molecule_atom_indices],
            axis=0,
            weights=molecule_masses_kg,
        )
        molecular_charge_C = float(np.sum(system.charges_C[molecule_atom_indices]))
        current_density_A_m2 += molecular_charge_C * center_of_mass_velocity_m_s
    return current_density_A_m2 / volume_m3


def internal_polarization_current_density_A_m2(
    velocities_m_s: Array, system: MolecularSystem
) -> Array:
    volume_m3 = abs(np.linalg.det(system.box_vectors_m))
    return total_charge_current_density_A_m2(
        velocities_m_s, system.charges_C, volume_m3
    ) - molecular_com_current_density_A_m2(velocities_m_s, system)


def _torch_minimum_image(displacement_m: torch.Tensor, box_m: torch.Tensor) -> torch.Tensor:
    fractional = displacement_m @ torch.linalg.inv(box_m)
    return (fractional - torch.round(fractional)) @ box_m


def _random_rotation(random_generator: np.random.Generator) -> Array:
    quaternion = random_generator.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    scalar, x_value, y_value, z_value = quaternion
    return np.asarray(
        (
            (1 - 2 * (y_value**2 + z_value**2), 2 * (x_value * y_value - scalar * z_value), 2 * (x_value * z_value + scalar * y_value)),
            (2 * (x_value * y_value + scalar * z_value), 1 - 2 * (x_value**2 + z_value**2), 2 * (y_value * z_value - scalar * x_value)),
            (2 * (x_value * z_value - scalar * y_value), 2 * (y_value * z_value + scalar * x_value), 1 - 2 * (x_value**2 + y_value**2)),
        )
    )


def _salt_ion_names(salt_name: str) -> tuple[str, str]:
    salt_record = SALTS[salt_name]
    cation_name = f"{salt_record['cation']}+"
    return cation_name, str(salt_record["anion"])


def _recipe_species_mole_weights(recipe: ElectrolyteRecipeModel) -> dict[str, float]:
    physical_library = load_physical_library(Path(__file__).parent / "physical_library")
    species_weights: dict[str, float] = {}
    solvent_molar_volume_m3_mol = 0.0
    solvent_molar_mass_kg_mol = 0.0
    for solvent_name, volume_fraction in recipe.solvents.items():
        species_record = physical_library.species_records[solvent_name]
        solvent_molar_volume_m3_mol += volume_fraction * float(species_record["partial_molar_volume_m3_mol"])
        solvent_molar_mass_kg_mol += volume_fraction * float(species_record["molecular_weight_kg_mol"])
        species_weights[solvent_name] = volume_fraction
    for salt_name, molarity_mol_l in recipe.salts.items():
        cation_name, anion_name = _salt_ion_names(salt_name)
        ion_weight = molarity_mol_l * 1000.0 * solvent_molar_volume_m3_mol
        species_weights[cation_name] = species_weights.get(cation_name, 0.0) + ion_weight
        species_weights[anion_name] = species_weights.get(anion_name, 0.0) + ion_weight
    for additive_name, weight_fraction in recipe.additives.items():
        additive_record = ADDITIVES[additive_name]
        additive_molar_mass_kg_mol = float(additive_record["molecular_weight"]) / 1000.0
        additive_weight = weight_fraction * solvent_molar_mass_kg_mol / additive_molar_mass_kg_mol
        if "cation" in additive_record and "anion" in additive_record:
            cation_name = f"{additive_record['cation']}+"
            anion_name = str(additive_record["anion"])
            species_weights[cation_name] = species_weights.get(cation_name, 0.0) + additive_weight
            species_weights[anion_name] = species_weights.get(anion_name, 0.0) + additive_weight
        else:
            species_weights[additive_name] = additive_weight
    total_weight = sum(species_weights.values())
    return {name: weight / total_weight for name, weight in species_weights.items()}


def charge_neutral_integer_counts(
    species_mole_fractions: Array,
    molecular_charges_e: Array,
    molecule_count: int,
) -> Array:
    species_count = species_mole_fractions.size
    target_counts = molecule_count * species_mole_fractions
    objective = np.concatenate((np.zeros(species_count), np.ones(species_count)))
    equality = np.zeros((2, 2 * species_count))
    equality[0, :species_count] = 1.0
    equality[1, :species_count] = molecular_charges_e
    inequality = np.zeros((2 * species_count, 2 * species_count))
    upper = np.empty(2 * species_count)
    for species_index in range(species_count):
        inequality[2 * species_index, species_index] = 1.0
        inequality[2 * species_index, species_count + species_index] = -1.0
        upper[2 * species_index] = target_counts[species_index]
        inequality[2 * species_index + 1, species_index] = -1.0
        inequality[2 * species_index + 1, species_count + species_index] = -1.0
        upper[2 * species_index + 1] = -target_counts[species_index]
    lower = np.zeros(2 * species_count)
    lower[:species_count] = np.where(species_mole_fractions > 0.0, 1.0, 0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected.*",
            category=RuntimeWarning,
        )
        solution = milp(
            c=objective,
            integrality=np.concatenate(
                (np.ones(species_count), np.zeros(species_count))
            ),
            bounds=Bounds(lower, np.full(2 * species_count, np.inf)),
            constraints=(
                LinearConstraint(
                    equality,
                    np.asarray((molecule_count, 0.0)),
                    np.asarray((molecule_count, 0.0)),
                ),
                LinearConstraint(
                    inequality, np.full(2 * species_count, -np.inf), upper
                ),
            ),
            options={"mip_feasibility_tolerance": MILP_FEASIBILITY_TOLERANCE},
        )
    if not solution.success or solution.x is None:
        raise ValueError("recipe has no charge-neutral integer molecular realization")
    return np.rint(solution.x[:species_count]).astype(int)


def build_periodic_molecular_system(
    recipe: ElectrolyteRecipeModel,
    density_kg_m3: float,
    molecule_count: int,
    minimum_interatomic_contact_ratio: float,
    initial_placement_attempts_per_molecule: int,
    random_seed: int,
) -> MolecularSystem:
    library = load_physical_library(Path(__file__).parent / "physical_library")
    mole_weights = _recipe_species_mole_weights(recipe)
    species_extent_names: list[tuple[float, str]] = []
    for species_name in mole_weights:
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
        species_name for _negative_extent_m, species_name in sorted(species_extent_names)
    )
    species_records = tuple(library.species_records[name] for name in species_names)
    molecular_charges_e = np.asarray([float(record["formal_charge_e"]) for record in species_records])
    fractions = np.asarray([mole_weights[name] for name in species_names])
    counts = charge_neutral_integer_counts(fractions, molecular_charges_e, molecule_count)
    total_mass_kg = sum(
        int(count)
        * sum(float(site["mass_kg"]) for site in record["sites"])
        for count, record in zip(counts, species_records, strict=True)
    )
    box_length_m = (total_mass_kg / density_kg_m3) ** (1.0 / 3.0)
    box_vectors_m = np.eye(3) * box_length_m
    random_generator = np.random.default_rng(random_seed)
    positions: list[Array] = []
    masses: list[float] = []
    charges: list[float] = []
    lj_sigma: list[float] = []
    lj_epsilon: list[float] = []
    radii: list[float] = []
    polarizabilities: list[float] = []
    molecule_indices: list[int] = []
    molecule_atom_indices: list[Array] = []
    bonds: list[tuple[int, int]] = []
    bond_constants: list[float] = []
    bond_lengths: list[float] = []
    angles: list[tuple[int, int, int]] = []
    angle_constants: list[float] = []
    angle_values: list[float] = []
    torsions: list[tuple[int, int, int, int, tuple[tuple[float, int, float], ...]]] = []
    atom_offset = 0
    molecule_index = 0
    for count, record in zip(counts, species_records, strict=True):
        reference = np.asarray(record["reference_conformer_coordinates_m"], dtype=float)
        reference -= np.mean(reference, axis=0)
        for _ in range(int(count)):
            site_sigma_m = np.asarray(
                tuple(float(site["lj_sigma_m"]) for site in record["sites"])
            )
            molecule_positions = np.empty_like(reference)
            placement_succeeded = False
            for _placement_attempt in range(
                initial_placement_attempts_per_molecule
            ):
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
            for site in record["sites"]:
                masses.append(float(site["mass_kg"]))
                charges.append(float(site["charge_number"]) * E_CHARGE)
                lj_sigma.append(float(site["lj_sigma_m"]))
                lj_epsilon.append(float(site["lj_epsilon_J"]))
                radii.append(float(site["hydrodynamic_radius_m"]))
                polarizabilities.append(float(site["polarizability_SI"]))
                molecule_indices.append(molecule_index)
            for bond in record["bonds"]:
                bonds.append((atom_offset + int(bond["site_i"]), atom_offset + int(bond["site_j"])))
                bond_constants.append(float(bond["k_J_m2_mol"]) / N_A)
                bond_lengths.append(float(bond["r0_m"]))
            for angle in record["angles"]:
                angles.append((atom_offset + int(angle["site_i"]), atom_offset + int(angle["site_j"]), atom_offset + int(angle["site_k"])))
                angle_constants.append(float(angle["k_J_rad2_mol"]) / N_A)
                angle_values.append(float(angle["theta0_rad"]))
            for torsion in record["torsions"]:
                terms = tuple(
                    (float(term["Vn_J_mol"]) / (N_A * float(term["idivf"])), int(term["periodicity"]), float(term["phase_rad"]))
                    for term in torsion["terms"]
                )
                torsions.append((atom_offset + int(torsion["site_i"]), atom_offset + int(torsion["site_j"]), atom_offset + int(torsion["site_k"]), atom_offset + int(torsion["site_l"]), terms))
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
        positions_m=np.asarray(positions), box_vectors_m=box_vectors_m, masses_kg=np.asarray(masses), charges_C=np.asarray(charges),
        lj_sigma_m=np.asarray(lj_sigma), lj_epsilon_J=np.asarray(lj_epsilon), hydrodynamic_radii_m=np.asarray(radii),
        polarizabilities_SI=np.asarray(polarizabilities), molecule_index=np.asarray(molecule_indices),
        molecule_atom_indices=tuple(molecule_atom_indices), bonds=np.asarray(bonds, dtype=int).reshape((-1, 2)),
        bond_force_constants_J_m2=np.asarray(bond_constants), bond_lengths_m=np.asarray(bond_lengths),
        angles=np.asarray(angles, dtype=int).reshape((-1, 3)), angle_force_constants_J_rad2=np.asarray(angle_constants),
        angle_values_rad=np.asarray(angle_values), torsions=tuple(torsions), nonbonded_mask=nonbonded_mask,
    )


class AnalyticalPeriodicInteratomicModel:
    """Explicit bonded, LJ, Ewald, and induced-dipole Hamiltonian."""

    def __init__(self, system: MolecularSystem, numerics: NumericalSettings) -> None:
        self.system = system
        self.numerics = numerics
        self._reciprocal_indices = np.asarray(
            [
                (i_value, j_value, k_value)
                for i_value in range(-numerics.ewald_reciprocal_shell, numerics.ewald_reciprocal_shell + 1)
                for j_value in range(-numerics.ewald_reciprocal_shell, numerics.ewald_reciprocal_shell + 1)
                for k_value in range(-numerics.ewald_reciprocal_shell, numerics.ewald_reciprocal_shell + 1)
                if (i_value, j_value, k_value) != (0, 0, 0)
            ], dtype=float
        )

    def _energy_tensor(self, positions_m: torch.Tensor, box_vectors_m: torch.Tensor) -> torch.Tensor:
        system = self.system
        energy = torch.zeros((), dtype=TORCH_DTYPE)
        if system.bonds.size:
            bond_indices = torch.as_tensor(system.bonds)
            displacement = _torch_minimum_image(positions_m[bond_indices[:, 0]] - positions_m[bond_indices[:, 1]], box_vectors_m)
            lengths = torch.linalg.norm(displacement, dim=1)
            energy += torch.sum(0.5 * torch.as_tensor(system.bond_force_constants_J_m2) * (lengths - torch.as_tensor(system.bond_lengths_m)) ** 2)
        if system.angles.size:
            angle_indices = torch.as_tensor(system.angles)
            first = _torch_minimum_image(positions_m[angle_indices[:, 0]] - positions_m[angle_indices[:, 1]], box_vectors_m)
            second = _torch_minimum_image(positions_m[angle_indices[:, 2]] - positions_m[angle_indices[:, 1]], box_vectors_m)
            cross_norm = torch.sqrt(
                torch.sum(torch.linalg.cross(first, second, dim=1) ** 2, dim=1)
                + torch.finfo(TORCH_DTYPE).tiny
            )
            values = torch.atan2(cross_norm, torch.sum(first * second, dim=1))
            energy += torch.sum(0.5 * torch.as_tensor(system.angle_force_constants_J_rad2) * (values - torch.as_tensor(system.angle_values_rad)) ** 2)
        for first_index, second_index, third_index, fourth_index, terms in system.torsions:
            first = _torch_minimum_image(positions_m[second_index] - positions_m[first_index], box_vectors_m)
            second = _torch_minimum_image(positions_m[third_index] - positions_m[second_index], box_vectors_m)
            third = _torch_minimum_image(positions_m[fourth_index] - positions_m[third_index], box_vectors_m)
            first_normal = torch.linalg.cross(first, second)
            second_normal = torch.linalg.cross(second, third)
            dihedral = torch.atan2(
                torch.linalg.norm(second) * torch.dot(first, second_normal),
                torch.dot(first_normal, second_normal)
                + torch.finfo(TORCH_DTYPE).tiny,
            )
            for amplitude_J, periodicity, phase_rad in terms:
                energy += amplitude_J * (1.0 + torch.cos(periodicity * dihedral - phase_rad))
        displacement = _torch_minimum_image(positions_m[:, None, :] - positions_m[None, :, :], box_vectors_m)
        distance = torch.linalg.norm(displacement, dim=2)
        lj_pair_i, lj_pair_j = np.where(np.triu(system.nonbonded_mask, 1))
        lj_pair_i_tensor = torch.as_tensor(lj_pair_i)
        lj_pair_j_tensor = torch.as_tensor(lj_pair_j)
        lj_pair_distance = distance[lj_pair_i_tensor, lj_pair_j_tensor]
        sigma = 0.5 * (
            torch.as_tensor(system.lj_sigma_m[lj_pair_i])
            + torch.as_tensor(system.lj_sigma_m[lj_pair_j])
        )
        epsilon = torch.sqrt(
            torch.as_tensor(system.lj_epsilon_J[lj_pair_i])
            * torch.as_tensor(system.lj_epsilon_J[lj_pair_j])
        )
        attractive_term = (sigma / lj_pair_distance) ** LJ_ATTRACTIVE_EXPONENT
        energy += torch.sum(
            4.0
            * epsilon
            * (
                attractive_term**LJ_REPULSIVE_EXPONENT_MULTIPLIER
                - attractive_term
            )
        )
        charges = torch.as_tensor(system.charges_C)
        ewald_alpha = self.numerics.ewald_splitting_per_m
        reciprocal = (
            2.0
            * math.pi
            * torch.as_tensor(self._reciprocal_indices)
            @ torch.linalg.inv(box_vectors_m)
        )
        reciprocal_squared = torch.sum(reciprocal**2, dim=1)
        phases = positions_m @ reciprocal.T
        structure_real = torch.sum(charges[:, None] * torch.cos(phases), dim=0)
        structure_imaginary = torch.sum(charges[:, None] * torch.sin(phases), dim=0)
        volume_m3 = torch.abs(torch.linalg.det(box_vectors_m))
        green_weights = (
            torch.exp(-reciprocal_squared / (4.0 * ewald_alpha**2))
            / (EPS_0 * volume_m3 * reciprocal_squared)
        )
        energy += 0.5 * torch.sum(
            green_weights * (structure_real**2 + structure_imaginary**2)
        )
        if np.any(system.polarizabilities_SI > 0.0):
            energy += self._polarization_energy(
                positions_m, reciprocal, green_weights
            )
        return energy

    def _polarization_energy(
        self,
        positions_m: torch.Tensor,
        reciprocal_m_inv: torch.Tensor,
        green_weights_J_m_C2: torch.Tensor,
    ) -> torch.Tensor:
        active = np.flatnonzero(self.system.polarizabilities_SI > 0.0)
        charges = torch.as_tensor(self.system.charges_C)
        phases = positions_m @ reciprocal_m_inv.T
        phase_differences = phases[:, None, :] - phases[None, :, :]
        sine_weighted_charge = torch.sum(
            charges[None, :, None] * torch.sin(phase_differences), dim=1
        )
        electric_fields = torch.einsum(
            "ik,k,kd->id",
            sine_weighted_charge,
            green_weights_J_m_C2,
            reciprocal_m_inv,
        )
        active_tensor = torch.as_tensor(active)
        active_phase_differences = phase_differences[active_tensor][:, active_tensor]
        interaction_blocks = -torch.einsum(
            "ijk,k,kd,ke->ijde",
            torch.cos(active_phase_differences),
            green_weights_J_m_C2,
            reciprocal_m_inv,
            reciprocal_m_inv,
        )
        interaction = interaction_blocks.permute(0, 2, 1, 3).reshape(
            3 * active.size, 3 * active.size
        )
        inverse_alpha = torch.repeat_interleave(
            1.0 / torch.as_tensor(self.system.polarizabilities_SI[active]), 3
        )
        operator = torch.diag(inverse_alpha) - interaction
        active_electric_field = electric_fields[active_tensor].reshape(-1)
        dipoles = torch.linalg.solve(operator, active_electric_field)
        residual = torch.linalg.norm(operator @ dipoles - active_electric_field)
        if (
            float(residual.detach())
            > self.numerics.polarization_residual_tolerance_V_m
        ):
            raise ValueError("induced-dipole solve did not meet polarization tolerance")
        return -0.5 * torch.dot(dipoles, active_electric_field)

    def energy_J(self, positions_m: Array, box_vectors_m: Array) -> float:
        return float(self._energy_tensor(torch.as_tensor(positions_m), torch.as_tensor(box_vectors_m)).detach())

    def forces_N(self, positions_m: Array, box_vectors_m: Array) -> Array:
        positions = torch.tensor(positions_m, dtype=TORCH_DTYPE, requires_grad=True)
        energy = self._energy_tensor(positions, torch.as_tensor(box_vectors_m))
        return -torch.autograd.grad(energy, positions)[0].detach().numpy()


def periodic_diffusion_operator(
    system: MolecularSystem,
    positions_m: Array,
    temperature_K: float,
    viscosity_Pa_s: float,
    reciprocal_shell: int,
) -> PeriodicDiffusionOperator:
    """Return the periodic Gaussian-regularized transverse diffusion operator.

    Every Fourier block is proportional to ``I - kk.T / |k|^2``. Its
    configuration divergence therefore vanishes analytically, so the reversible
    Itô drift is ``D F / (k_B T)`` without an additional thermal-drift term.
    """
    atom_count = positions_m.shape[0]
    identity = np.eye(CARTESIAN_DIMENSION)
    volume_m3 = abs(np.linalg.det(system.box_vectors_m))
    reciprocal_basis = 2.0 * math.pi * np.linalg.inv(system.box_vectors_m)
    spectral_columns: list[Array] = []
    spectral_self_blocks = np.zeros(
        (atom_count, CARTESIAN_DIMENSION, CARTESIAN_DIMENSION)
    )
    reciprocal_indices = (
        (first, second, third)
        for first in range(-reciprocal_shell, reciprocal_shell + 1)
        for second in range(-reciprocal_shell, reciprocal_shell + 1)
        for third in range(-reciprocal_shell, reciprocal_shell + 1)
        if (first, second, third) != (0, 0, 0)
    )
    for reciprocal_index in reciprocal_indices:
        wavevector = np.asarray(reciprocal_index, dtype=float) @ reciprocal_basis
        wavevector_squared = float(wavevector @ wavevector)
        transverse_projector = identity - np.outer(wavevector, wavevector) / wavevector_squared
        projector_eigenvalues, projector_eigenvectors = eigh(
            transverse_projector
        )
        transverse_directions = projector_eigenvectors[
            :, projector_eigenvalues > 0.5
        ]
        phases = positions_m @ wavevector
        attenuation = np.exp(
            -wavevector_squared
            * system.hydrodynamic_radii_m**2
            / GAUSSIAN_BLOB_VARIANCE_DENOMINATOR
        )
        cosine_amplitudes = attenuation * np.cos(phases)
        sine_amplitudes = attenuation * np.sin(phases)
        mode_scale = math.sqrt(
            K_B
            * temperature_K
            / (viscosity_Pa_s * volume_m3 * wavevector_squared)
        )
        for amplitudes in (cosine_amplitudes, sine_amplitudes):
            for transverse_direction in transverse_directions.T:
                spectral_columns.append(
                    (
                        mode_scale
                        * amplitudes[:, None]
                        * transverse_direction[None, :]
                    ).reshape(-1)
                )
        spectral_self_blocks += (
            K_B
            * temperature_K
            * attenuation[:, None, None] ** 2
            * transverse_projector[None, :, :]
            / (viscosity_Pa_s * volume_m3 * wavevector_squared)
        )
    spectral_factor = np.column_stack(spectral_columns)
    local_blocks = np.empty_like(spectral_self_blocks)
    for atom_index, radius_m in enumerate(system.hydrodynamic_radii_m):
        target_self_diffusion = K_B * temperature_K * identity / (
            STOKES_SPHERE_DRAG_FACTOR * math.pi * viscosity_Pa_s * radius_m
        )
        local_blocks[atom_index] = (
            target_self_diffusion - spectral_self_blocks[atom_index]
        )
        remainder_eigenvalues = np.linalg.eigvalsh(local_blocks[atom_index])
        if remainder_eigenvalues[0] <= 0.0:
            raise ValueError(
                "spectral RPY shell exceeds Stokes self mobility; reduce the "
                "reciprocal shell or increase the box"
            )
    return PeriodicDiffusionOperator(
        local_blocks_m2_s=local_blocks,
        spectral_factor_m_sqrt_s=spectral_factor,
    )


def sample_equilibrium_configurations(
    model: AnalyticalPeriodicInteratomicModel,
    temperature_K: float,
    dynamics: DynamicsSettings,
    random_seed: int,
) -> Array:
    random_generator = np.random.default_rng(random_seed)
    positions = relax_initial_configuration(model, temperature_K, dynamics)
    inverse_thermal_energy = 1.0 / (K_B * temperature_K)
    energy = model.energy_J(positions, model.system.box_vectors_m)
    samples: list[Array] = []
    total_sweeps = dynamics.equilibrium_burn_in_sweeps + dynamics.equilibrium_sample_count * dynamics.equilibrium_sweeps_per_sample
    for sweep_index in range(total_sweeps):
        for molecule_indices in model.system.molecule_atom_indices:
            proposal = positions.copy()
            if random_generator.random() < 0.5:
                proposal[molecule_indices] += random_generator.normal(scale=dynamics.translation_proposal_m, size=3)
            else:
                center = np.mean(proposal[molecule_indices], axis=0)
                axis = random_generator.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = random_generator.normal(scale=dynamics.rotation_proposal_rad)
                cross_matrix = np.asarray(((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)))
                rotation = np.eye(3) + math.sin(angle) * cross_matrix + (1.0 - math.cos(angle)) * (cross_matrix @ cross_matrix)
                proposal[molecule_indices] = (proposal[molecule_indices] - center) @ rotation.T + center
            proposal %= np.diag(model.system.box_vectors_m)
            proposal_energy = model.energy_J(proposal, model.system.box_vectors_m)
            if math.log(random_generator.random()) < -inverse_thermal_energy * (proposal_energy - energy):
                positions, energy = proposal, proposal_energy
        atom_index = int(random_generator.integers(positions.shape[0]))
        proposal = positions.copy()
        proposal[atom_index] += random_generator.normal(scale=dynamics.internal_proposal_m, size=3)
        proposal %= np.diag(model.system.box_vectors_m)
        proposal_energy = model.energy_J(proposal, model.system.box_vectors_m)
        if math.log(random_generator.random()) < -inverse_thermal_energy * (proposal_energy - energy):
            positions, energy = proposal, proposal_energy
        if sweep_index >= dynamics.equilibrium_burn_in_sweeps and (sweep_index - dynamics.equilibrium_burn_in_sweeps) % dynamics.equilibrium_sweeps_per_sample == 0:
            samples.append(positions.copy())
    return np.asarray(samples)


def relax_initial_configuration(
    model: AnalyticalPeriodicInteratomicModel,
    temperature_K: float,
    dynamics: DynamicsSettings,
) -> Array:
    coordinate_scale_m = dynamics.initial_relaxation_step_m
    energy_scale_J = K_B * temperature_K
    initial_coordinates = (model.system.positions_m / coordinate_scale_m).reshape(-1)

    def energy_and_gradient(scaled_coordinates: Array) -> tuple[float, Array]:
        positions_m = scaled_coordinates.reshape((-1, 3)) * coordinate_scale_m
        energy_J = model.energy_J(positions_m, model.system.box_vectors_m)
        gradient_J = (
            -model.forces_N(positions_m, model.system.box_vectors_m)
            * coordinate_scale_m
        )
        return energy_J / energy_scale_J, gradient_J.reshape(-1) / energy_scale_J

    optimization = minimize(
        energy_and_gradient,
        initial_coordinates,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": dynamics.initial_relaxation_steps,
            "gtol": (
                INITIAL_RELAXATION_FORCE_MARGIN
                *
                dynamics.initial_force_tolerance_N
                * coordinate_scale_m
                / energy_scale_J
            ),
            "ftol": np.finfo(float).eps,
        },
    )
    positions = (
        optimization.x.reshape((-1, 3)) * coordinate_scale_m
    ) % np.diag(model.system.box_vectors_m)
    if not np.isfinite(optimization.fun):
        raise ValueError("initial molecular relaxation produced nonfinite energy")
    pair_i, pair_j = np.where(np.triu(model.system.nonbonded_mask, 1))
    pair_displacements = minimum_image_displacement(
        positions[pair_i] - positions[pair_j], model.system.box_vectors_m
    )
    pair_distances = np.linalg.norm(pair_displacements, axis=1)
    contact_distances = 0.5 * (
        model.system.lj_sigma_m[pair_i] + model.system.lj_sigma_m[pair_j]
    )
    minimum_contact_ratio = float(np.min(pair_distances / contact_distances))
    if minimum_contact_ratio < model.numerics.minimum_interatomic_contact_ratio:
        raise ValueError(
            "relaxed molecular configuration minimum LJ contact ratio "
            f"{minimum_contact_ratio:.6f} is below "
            f"{model.numerics.minimum_interatomic_contact_ratio:.6f}"
        )
    return positions


def _basis_values_tensor(positions_m: torch.Tensor, system: MolecularSystem, numerics: NumericalSettings) -> torch.Tensor:
    box = torch.as_tensor(system.box_vectors_m)
    displacement = _torch_minimum_image(positions_m[:, None, :] - positions_m[None, :, :], box)
    distance = torch.linalg.norm(displacement + torch.eye(positions_m.shape[0])[:, :, None], dim=2)
    pair_i, pair_j = np.triu_indices(positions_m.shape[0], 1)
    pair_distance = distance[torch.as_tensor(pair_i), torch.as_tensor(pair_j)]
    pair_charge = torch.as_tensor(system.charges_C[pair_i] * system.charges_C[pair_j])
    features: list[torch.Tensor] = []
    charges = torch.as_tensor(system.charges_C)
    masses = torch.as_tensor(system.masses_kg)
    total_internal_polarization = torch.zeros(
        CARTESIAN_DIMENSION, dtype=TORCH_DTYPE
    )
    for molecule_atom_indices in system.molecule_atom_indices:
        molecule_indices = torch.as_tensor(molecule_atom_indices)
        anchor = positions_m[molecule_indices[0]]
        local_positions = anchor + _torch_minimum_image(
            positions_m[molecule_indices] - anchor, box
        )
        molecule_masses = masses[molecule_indices]
        center_of_mass = torch.sum(
            molecule_masses[:, None] * local_positions, dim=0
        ) / torch.sum(molecule_masses)
        total_internal_polarization += torch.sum(
            charges[molecule_indices, None]
            * (local_positions - center_of_mass),
            dim=0,
        )
    features.extend(torch.unbind(total_internal_polarization))
    cutoff = 0.5 * (torch.cos(math.pi * torch.clamp(pair_distance / numerics.basis_radial_cutoff_m, 0.0, 1.0)) + 1.0)
    cutoff = cutoff * (pair_distance < numerics.basis_radial_cutoff_m)
    for radial_mode_index in range(1, numerics.basis_radial_count + 1):
        radial = (
            torch.cos(
                radial_mode_index
                * math.pi
                * pair_distance
                / numerics.basis_radial_cutoff_m
            )
            * cutoff
        )
        features.extend((torch.sum(radial), torch.sum(pair_charge * radial)))
    reciprocal_base = 2.0 * math.pi * torch.linalg.inv(box)
    for shell in range(1, numerics.basis_fourier_shell + 1):
        for axis in range(3):
            reciprocal = shell * reciprocal_base[axis]
            phases = positions_m @ reciprocal
            features.extend((torch.sum(torch.cos(phases)), torch.sum(torch.sin(phases)), torch.sum(charges * torch.cos(phases)), torch.sum(charges * torch.sin(phases))))
    return torch.stack(features)


def atomic_charge_polarization_gradients(system: MolecularSystem) -> Array:
    gradients = np.zeros((CARTESIAN_DIMENSION, 3 * system.charges_C.size))
    for axis in range(CARTESIAN_DIMENSION):
        gradients[axis, axis::CARTESIAN_DIMENSION] = system.charges_C
    return gradients


def molecular_com_charge_polarization_gradients(system: MolecularSystem) -> Array:
    gradients = np.zeros((CARTESIAN_DIMENSION, 3 * system.charges_C.size))
    for molecule_atom_indices in system.molecule_atom_indices:
        molecular_charge_C = float(np.sum(system.charges_C[molecule_atom_indices]))
        molecule_masses_kg = system.masses_kg[molecule_atom_indices]
        molecular_mass_kg = float(np.sum(molecule_masses_kg))
        for atom_index, atom_mass_kg in zip(
            molecule_atom_indices, molecule_masses_kg, strict=True
        ):
            charge_weight_C = molecular_charge_C * atom_mass_kg / molecular_mass_kg
            for axis in range(CARTESIAN_DIMENSION):
                gradients[axis, CARTESIAN_DIMENSION * atom_index + axis] = (
                    charge_weight_C
                )
    return gradients


def molecular_charge_com_polarization(
    unwrapped_positions_m: Array, system: MolecularSystem
) -> Array:
    polarization_C_m = np.zeros(CARTESIAN_DIMENSION)
    for molecule_atom_indices in system.molecule_atom_indices:
        molecule_masses_kg = system.masses_kg[molecule_atom_indices]
        center_of_mass_m = np.average(
            unwrapped_positions_m[molecule_atom_indices],
            axis=0,
            weights=molecule_masses_kg,
        )
        molecular_charge_C = float(np.sum(system.charges_C[molecule_atom_indices]))
        polarization_C_m += molecular_charge_C * center_of_mass_m
    return polarization_C_m


def evaluate_basis_and_gradients(configurations_m: Array, system: MolecularSystem, numerics: NumericalSettings) -> tuple[Array, Array]:
    values: list[Array] = []
    gradients: list[Array] = []
    for configuration in configurations_m:
        positions = torch.tensor(configuration, dtype=TORCH_DTYPE, requires_grad=True)
        basis = _basis_values_tensor(positions, system, numerics)
        jacobian = torch.autograd.functional.jacobian(lambda coordinates: _basis_values_tensor(coordinates, system, numerics), positions, vectorize=True)
        values.append(basis.detach().numpy())
        gradients.append(jacobian.detach().numpy().reshape((basis.shape[0], -1)))
    value_array = np.asarray(values)
    value_array -= np.mean(value_array, axis=0)
    return value_array, np.asarray(gradients)


def symmetric_psd_pseudoinverse(matrix: Array, relative_tolerance: float) -> Array:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = eigh(symmetric)
    tolerance = relative_tolerance * float(eigenvalues[-1])
    if eigenvalues[0] < -tolerance:
        raise ValueError("Dirichlet matrix is not positive semidefinite")
    active = eigenvalues > tolerance
    return (eigenvectors[:, active] / eigenvalues[active]) @ eigenvectors[:, active].T


def projected_conductivity_sequence(
    configurations_m: Array,
    system: MolecularSystem,
    temperature_K: float,
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
) -> tuple[float, float, tuple[float, ...], tuple[float, ...], int]:
    minimum_partition_count = 2
    if configurations_m.shape[0] < 2 * minimum_partition_count:
        raise ValueError("projection requires at least four equilibrium samples")
    _basis_values, gradients = evaluate_basis_and_gradients(
        configurations_m, system, numerics
    )
    basis_count = min(gradients.shape[1], numerics.maximum_basis_size)
    polarization_gradients = atomic_charge_polarization_gradients(system)
    split_index = configurations_m.shape[0] // 2
    statistics: list[tuple[Array, Array, Array]] = []
    for sample_indices in (
        range(split_index),
        range(split_index, configurations_m.shape[0]),
    ):
        sample_indices_tuple = tuple(sample_indices)
        dirichlet = np.zeros((basis_count, basis_count))
        coupling = np.zeros((basis_count, CARTESIAN_DIMENSION))
        direct_axes = np.zeros(CARTESIAN_DIMENSION)
        for sample_index in sample_indices_tuple:
            diffusion_operator = periodic_diffusion_operator(
                system,
                configurations_m[sample_index],
                temperature_K,
                dynamics.solvent_viscosity_Pa_s,
                numerics.ewald_reciprocal_shell,
            )
            sample_gradients = gradients[sample_index, :basis_count]
            diffused_gradients = diffusion_operator.apply(sample_gradients)
            dirichlet += diffused_gradients @ sample_gradients.T
            coupling += diffused_gradients @ polarization_gradients.T
            diffused_polarization = diffusion_operator.apply(
                polarization_gradients
            )
            for axis in range(CARTESIAN_DIMENSION):
                direct_axes[axis] += (
                    polarization_gradients[axis]
                    @ diffused_polarization[axis]
                )
        sample_count = len(sample_indices_tuple)
        statistics.append(
            (
                dirichlet / sample_count,
                coupling / sample_count,
                direct_axes / sample_count,
            )
        )
    fit_dirichlet, fit_coupling, fit_direct_axes = statistics[0]
    heldout_dirichlet, heldout_coupling, _heldout_direct_axes = statistics[1]
    fit_diagonal = np.diag(fit_dirichlet)
    active_basis = fit_diagonal > 0.0
    if not np.any(active_basis):
        raise ValueError("basis has zero Dirichlet energy on the fit samples")
    basis_scales = np.sqrt(fit_diagonal[active_basis])
    fit_dirichlet = fit_dirichlet[np.ix_(active_basis, active_basis)] / (
        basis_scales[:, None] * basis_scales[None, :]
    )
    heldout_dirichlet = heldout_dirichlet[
        np.ix_(active_basis, active_basis)
    ] / (basis_scales[:, None] * basis_scales[None, :])
    fit_coupling = fit_coupling[active_basis] / basis_scales[:, None]
    heldout_coupling = heldout_coupling[active_basis] / basis_scales[:, None]
    basis_count = fit_dirichlet.shape[0]
    volume_m3 = abs(np.linalg.det(system.box_vectors_m))
    prefactor = 1.0 / (3.0 * K_B * temperature_K * volume_m3)
    direct = prefactor * float(np.sum(fit_direct_axes))
    history: list[float] = []
    residuals: list[float] = []
    previous = direct
    remaining_indices = list(range(basis_count))
    selected_indices: list[int] = []
    null_tolerance = numerics.eigenvalue_relative_tolerance
    while remaining_indices:
        if selected_indices:
            selected_array = np.asarray(selected_indices)
            fit_inverse = symmetric_psd_pseudoinverse(
                fit_dirichlet[np.ix_(selected_array, selected_array)],
                numerics.eigenvalue_relative_tolerance,
            )
        candidate_records: list[tuple[float, int]] = []
        for candidate_index in remaining_indices:
            fit_coupling_vector = fit_coupling[candidate_index].copy()
            fit_energy = float(fit_dirichlet[candidate_index, candidate_index])
            if selected_indices:
                fit_cross = fit_dirichlet[candidate_index, selected_array]
                fit_energy -= float(fit_cross @ fit_inverse @ fit_cross)
                fit_coupling_vector -= (
                    fit_cross @ fit_inverse @ fit_coupling[selected_array]
                )
            if fit_energy <= null_tolerance:
                continue
            score = (
                prefactor
                * float(fit_coupling_vector @ fit_coupling_vector)
                / fit_energy
            )
            candidate_records.append((score, candidate_index))
        if not candidate_records:
            break
        maximum_score, next_index = max(candidate_records)
        remaining_indices.remove(next_index)
        selected_indices.append(next_index)
        selected_array = np.asarray(selected_indices)
        selected_inverse = symmetric_psd_pseudoinverse(
            fit_dirichlet[np.ix_(selected_array, selected_array)],
            numerics.eigenvalue_relative_tolerance,
        )
        correction = prefactor * float(
            np.sum(
                fit_coupling[selected_array].T
                @ selected_inverse
                @ fit_coupling[selected_array]
            )
        )
        fit_coefficients = (
            selected_inverse @ fit_coupling[selected_array]
        )
        heldout_selected_inverse = symmetric_psd_pseudoinverse(
            heldout_dirichlet[np.ix_(selected_array, selected_array)],
            numerics.eigenvalue_relative_tolerance,
        )
        validation_scores: list[float] = []
        for validation_index in remaining_indices:
            heldout_cross = heldout_dirichlet[
                validation_index, selected_array
            ]
            validation_residual_coupling = (
                heldout_coupling[validation_index]
                - heldout_cross @ fit_coefficients
            )
            validation_energy = float(
                heldout_dirichlet[validation_index, validation_index]
                - heldout_cross
                @ heldout_selected_inverse
                @ heldout_cross
            )
            if validation_energy > null_tolerance:
                validation_scores.append(
                    prefactor
                    * float(
                        validation_residual_coupling
                        @ validation_residual_coupling
                    )
                    / validation_energy
                )
        if not validation_scores:
            raise ValueError(
                "basis hierarchy has no untouched validation candidates after "
                "the fitted level"
            )
        maximum_validation_score = max(validation_scores)
        conductivity = direct - correction
        if conductivity > previous + numerics.conductivity_tolerance_S_m:
            raise ValueError("projected conductivity sequence is not monotone")
        history.append(conductivity)
        residuals.append(maximum_validation_score)
        if (
            maximum_validation_score <= numerics.residual_tolerance
            and abs(conductivity - previous)
            <= numerics.conductivity_tolerance_S_m
        ):
            break
        previous = conductivity
    if not history:
        raise ValueError("basis contains no resolvable nonconstant mode")
    if (
        residuals[-1] > numerics.residual_tolerance
        or (
            len(history) > 1
            and abs(history[-1] - history[-2])
            > numerics.conductivity_tolerance_S_m
        )
    ):
        raise ValueError(
            "basis hierarchy exhausted before held-out generator residual and "
            "conductivity change converged"
        )
    return (
        history[-1],
        direct,
        tuple(history),
        tuple(residuals),
        len(selected_indices),
    )


def simulate_overdamped_polarization(
    model: AnalyticalPeriodicInteratomicModel,
    initial_positions_m: Array,
    temperature_K: float,
    dynamics: DynamicsSettings,
    random_seed: int,
) -> tuple[Array, float, float]:
    random_generator = np.random.default_rng(random_seed)
    unwrapped = initial_positions_m.copy()
    polarization: list[Array] = []
    timestep_s = dynamics.overdamped_timestep_s
    inverse_thermal_energy_J = 1.0 / (K_B * temperature_K)
    energy_J = model.energy_J(unwrapped, model.system.box_vectors_m)
    accepted_step_count = 0
    for step_index in range(dynamics.overdamped_steps):
        diffusion_operator = periodic_diffusion_operator(
            model.system,
            unwrapped,
            temperature_K,
            dynamics.solvent_viscosity_Pa_s,
            model.numerics.ewald_reciprocal_shell,
        )
        forces = model.forces_N(
            unwrapped, model.system.box_vectors_m
        ).reshape(-1)
        forward_mean = (
            timestep_s
            * diffusion_operator.apply(forces)
            * inverse_thermal_energy_J
        )
        noise = (
            math.sqrt(2.0 * timestep_s)
            * diffusion_operator.sample_increment(random_generator)
        )
        proposal_displacement = forward_mean + noise
        proposal_unwrapped = unwrapped + proposal_displacement.reshape(
            (-1, CARTESIAN_DIMENSION)
        )
        proposal_energy_J = model.energy_J(
            proposal_unwrapped, model.system.box_vectors_m
        )
        proposal_diffusion_operator = periodic_diffusion_operator(
            model.system,
            proposal_unwrapped,
            temperature_K,
            dynamics.solvent_viscosity_Pa_s,
            model.numerics.ewald_reciprocal_shell,
        )
        proposal_forces = model.forces_N(
            proposal_unwrapped, model.system.box_vectors_m
        ).reshape(-1)
        reverse_mean = (
            timestep_s
            * proposal_diffusion_operator.apply(proposal_forces)
            * inverse_thermal_energy_J
        )
        forward_residual = proposal_displacement - forward_mean
        reverse_residual = -proposal_displacement - reverse_mean
        forward_log_determinant = diffusion_operator.log_determinant()
        reverse_log_determinant = (
            proposal_diffusion_operator.log_determinant()
        )
        forward_quadratic = float(
            forward_residual @ diffusion_operator.solve(forward_residual)
        )
        reverse_quadratic = float(
            reverse_residual
            @ proposal_diffusion_operator.solve(reverse_residual)
        )
        log_acceptance = (
            -inverse_thermal_energy_J * (proposal_energy_J - energy_J)
            - 0.5 * reverse_log_determinant
            - reverse_quadratic / (4.0 * timestep_s)
            + 0.5 * forward_log_determinant
            + forward_quadratic / (4.0 * timestep_s)
        )
        if math.log(random_generator.random()) < min(0.0, log_acceptance):
            unwrapped = proposal_unwrapped
            energy_J = proposal_energy_J
            accepted_step_count += 1
        if step_index % dynamics.overdamped_sample_stride == 0:
            polarization.append(
                np.sum(model.system.charges_C[:, None] * unwrapped, axis=0)
            )
    acceptance_fraction = accepted_step_count / dynamics.overdamped_steps
    if acceptance_fraction < dynamics.minimum_overdamped_acceptance_fraction:
        raise ValueError(
            "overdamped acceptance fraction "
            f"{acceptance_fraction:.6g} is below "
            f"{dynamics.minimum_overdamped_acceptance_fraction:.6g}"
        )
    return (
        np.asarray(polarization),
        timestep_s * dynamics.overdamped_sample_stride,
        acceptance_fraction,
    )


def einstein_helfand_conductivity(
    polarization_C_m: Array,
    sample_interval_s: float,
    volume_m3: float,
    temperature_K: float,
    fit_start_fraction: float,
    maximum_lag_fraction: float,
) -> float:
    sample_count = polarization_C_m.shape[0]
    maximum_lag = int(maximum_lag_fraction * sample_count)
    if maximum_lag < 2:
        raise ValueError("Helfand trajectory is too short for the lag window")
    lags = np.unique(np.linspace(1, maximum_lag, min(maximum_lag, 64), dtype=int))
    mean_square = np.asarray([np.mean(np.sum((polarization_C_m[lag:] - polarization_C_m[:-lag]) ** 2, axis=1)) for lag in lags])
    times = lags * sample_interval_s
    start = max(1, int(fit_start_fraction * times.size))
    slope = linear_fit(times[start:], mean_square[start:]).slope
    if slope <= 0.0:
        raise ValueError("Helfand charge displacement has no positive diffusive slope")
    return slope / (6.0 * K_B * temperature_K * volume_m3)


def current_autocorrelation(current_C_m_s: Array) -> Array:
    centered = current_C_m_s - np.mean(current_C_m_s, axis=0)
    sample_count = centered.shape[0]
    padded_count = 2 * sample_count
    transforms = np.fft.rfft(centered, n=padded_count, axis=0)
    correlations = np.fft.irfft(
        transforms * np.conjugate(transforms), n=padded_count, axis=0
    )[:sample_count]
    normalization = np.arange(sample_count, 0, -1)[:, None]
    return np.sum(correlations / normalization, axis=1)


def integrated_green_kubo_conductivity(
    polarization_C_m: Array,
    sample_interval_s: float,
    volume_m3: float,
    temperature_K: float,
    noise_window: int,
    noise_standard_error_multiplier: float,
) -> float:
    current_C_m_s = np.diff(polarization_C_m, axis=0) / sample_interval_s
    autocorrelation = current_autocorrelation(current_C_m_s)
    sample_count = current_C_m_s.shape[0]
    zero_lag_standard_error = abs(float(autocorrelation[0])) / math.sqrt(
        sample_count
    )
    cutoff_lag = autocorrelation.size
    for lag in range(1, autocorrelation.size - noise_window + 1):
        lag_window = autocorrelation[lag : lag + noise_window]
        lag_standard_error = zero_lag_standard_error * math.sqrt(
            sample_count / (sample_count - lag)
        )
        if np.all(
            np.abs(lag_window)
            <= noise_standard_error_multiplier * lag_standard_error
        ):
            cutoff_lag = lag
            break
    integral_C2_m2_s = sample_interval_s * (
        0.5 * float(autocorrelation[0])
        + float(np.sum(autocorrelation[1:cutoff_lag]))
    )
    if integral_C2_m2_s <= 0.0:
        raise ValueError("Green-Kubo current integral is not positive")
    return integral_C2_m2_s / (3.0 * K_B * temperature_K * volume_m3)


def validate_force_consistency(
    model: AnalyticalPeriodicInteratomicModel,
    positions_m: Array,
    numerics: NumericalSettings,
    random_seed: int,
) -> None:
    random_generator = np.random.default_rng(random_seed)
    direction = random_generator.normal(size=positions_m.shape)
    direction /= np.linalg.norm(direction)
    displacement = numerics.force_difference_step_m * direction
    positive_energy = model.energy_J(
        positions_m + displacement, model.system.box_vectors_m
    )
    negative_energy = model.energy_J(
        positions_m - displacement, model.system.box_vectors_m
    )
    finite_difference = (positive_energy - negative_energy) / (
        2.0 * numerics.force_difference_step_m
    )
    analytical = -float(
        np.sum(
            model.forces_N(positions_m, model.system.box_vectors_m)
            * direction
        )
    )
    scale = max(abs(finite_difference), abs(analytical), np.finfo(float).tiny)
    relative_error = abs(finite_difference - analytical) / scale
    if relative_error > numerics.force_consistency_relative_tolerance:
        raise ValueError(
            f"force directional derivative relative error {relative_error:.6e} "
            f"exceeds {numerics.force_consistency_relative_tolerance:.6e}"
        )


def _validate_settings(
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
) -> None:
    values = tuple(asdict(dynamics).values()) + tuple(asdict(numerics).values())
    if any(float(value) <= 0.0 for value in values):
        raise ValueError("all dynamics settings and numerical settings must be positive")
    fractions = (
        numerics.helfand_fit_start_fraction,
        numerics.helfand_maximum_lag_fraction,
        numerics.minimum_interatomic_contact_ratio,
        numerics.density_relative_tolerance,
    )
    if any(value >= 1.0 for value in fractions):
        raise ValueError("fractional numerical settings must be below one")
    if dynamics.overdamped_sample_stride > dynamics.overdamped_steps:
        raise ValueError("overdamped sample stride exceeds trajectory length")


def compute_first_principles_conductivity(
    recipe: ElectrolyteRecipeModel,
    interatomic_model: InteratomicModel,
    temperature_K: float,
    density_kg_m3: float,
    molecule_count: int,
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
    random_seed: int,
) -> ConductivityResult:
    _validate_settings(dynamics, numerics)
    if not isinstance(interatomic_model, AnalyticalPeriodicInteratomicModel):
        raise TypeError("interatomic_model must be AnalyticalPeriodicInteratomicModel")
    system = interatomic_model.system
    actual_molecule_count = len(system.molecule_atom_indices)
    if actual_molecule_count != molecule_count:
        raise ValueError(
            f"model has {actual_molecule_count} molecules; requested {molecule_count}"
        )
    actual_density_kg_m3 = float(
        np.sum(system.masses_kg) / abs(np.linalg.det(system.box_vectors_m))
    )
    density_relative_error = abs(actual_density_kg_m3 - density_kg_m3) / density_kg_m3
    if density_relative_error > numerics.density_relative_tolerance:
        raise ValueError(
            f"model density {actual_density_kg_m3:.9g} kg/m3 differs from "
            f"requested {density_kg_m3:.9g} kg/m3"
        )
    configurations = sample_equilibrium_configurations(interatomic_model, temperature_K, dynamics, random_seed)
    validate_force_consistency(
        interatomic_model, configurations[-1], numerics, random_seed
    )
    energy_series = np.asarray([interatomic_model.energy_J(configuration, interatomic_model.system.box_vectors_m) for configuration in configurations])
    stationary_suffix = select_stationary_suffix(
        values=energy_series,
        maximum_split_mean_difference_standard_errors=(
            numerics.stationarity_standard_error_limit
        ),
        maximum_linear_drift_standard_errors=(
            numerics.stationarity_standard_error_limit
        ),
        minimum_effective_sample_size=numerics.minimum_effective_sample_size,
    )
    configurations = configurations[stationary_suffix.start_index :]
    effective_sample_size = (
        stationary_suffix.autocorrelation.effective_sample_size
    )
    conductivity, direct, history, residuals, basis_size = projected_conductivity_sequence(
        configurations, interatomic_model.system, temperature_K, dynamics, numerics
    )
    polarization, interval, acceptance_fraction = simulate_overdamped_polarization(
        interatomic_model,
        configurations[-1],
        temperature_K,
        dynamics,
        random_seed + 1,
    )
    volume_m3 = abs(np.linalg.det(interatomic_model.system.box_vectors_m))
    green_kubo = integrated_green_kubo_conductivity(
        polarization,
        interval,
        volume_m3,
        temperature_K,
        numerics.gk_noise_window,
        numerics.gk_noise_standard_error_multiplier,
    )
    einstein_helfand = einstein_helfand_conductivity(
        polarization,
        interval,
        volume_m3,
        temperature_K,
        numerics.helfand_fit_start_fraction,
        numerics.helfand_maximum_lag_fraction,
    )
    projected_gk_difference_S_m = abs(conductivity - green_kubo)
    if projected_gk_difference_S_m > numerics.projected_gk_tolerance_S_m:
        raise ValueError(
            "projected/Green-Kubo conductivity difference "
            f"{projected_gk_difference_S_m:.6e} S/m exceeds "
            f"{numerics.projected_gk_tolerance_S_m:.6e} S/m"
        )
    gk_helfand_difference_S_m = abs(green_kubo - einstein_helfand)
    if gk_helfand_difference_S_m > numerics.projected_gk_tolerance_S_m:
        raise ValueError(
            "Green-Kubo/Einstein-Helfand conductivity difference "
            f"{gk_helfand_difference_S_m:.6e} S/m exceeds "
            f"{numerics.projected_gk_tolerance_S_m:.6e} S/m"
        )
    return ConductivityResult(
        conductivity_S_m=conductivity,
        direct_current_term_S_m=direct,
        projected_correction_S_m=direct - conductivity,
        green_kubo_conductivity_S_m=green_kubo,
        einstein_helfand_conductivity_S_m=einstein_helfand,
        basis_size=basis_size,
        basis_conductivities_S_m=history,
        residual_history=residuals, maximum_residual_score=residuals[-1], sample_count=configurations.shape[0],
        effective_sample_size=effective_sample_size,
        overdamped_acceptance_fraction=acceptance_fraction,
    )


def _settings_from_record(record: dict) -> tuple[DynamicsSettings, NumericalSettings]:
    return DynamicsSettings(**record["dynamics"]), NumericalSettings(**record["numerics"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-json", required=True, type=Path)
    parser.add_argument("--interatomic-model-json", required=True, type=Path)
    parser.add_argument("--numerics-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    arguments = parser.parse_args()
    recipe = ElectrolyteRecipeModel.model_validate(read_json_object(arguments.recipe_json, "electrolyte recipe"))
    model_record = read_json_object(arguments.interatomic_model_json, "analytical molecular model")
    settings_record = read_json_object(arguments.numerics_json, "conductivity numerics")
    if model_record["model_type"] != "analytical_periodic_smoluchowski":
        raise ValueError("model_type must be analytical_periodic_smoluchowski")
    dynamics, numerics = _settings_from_record(settings_record)
    temperature_K = float(settings_record["temperature_K"])
    density_kg_m3 = float(settings_record["density_kg_m3"])
    molecule_count = int(settings_record["molecule_count"])
    random_seed = int(settings_record["random_seed"])
    system = build_periodic_molecular_system(
        recipe,
        density_kg_m3,
        molecule_count,
        numerics.minimum_interatomic_contact_ratio,
        numerics.initial_placement_attempts_per_molecule,
        random_seed,
    )
    model = AnalyticalPeriodicInteratomicModel(system, numerics)
    result = compute_first_principles_conductivity(recipe, model, temperature_K, density_kg_m3, molecule_count, dynamics, numerics, random_seed)
    write_json_object(arguments.output_json, asdict(result), "conductivity result")
    print(f"conductivity = {result.conductivity_S_m:.8g} S/m ({result.conductivity_S_m * S_M_TO_MS_CM:.8g} mS/cm)")
    print(f"direct = {result.direct_current_term_S_m:.8g} S/m")
    print(f"projected correction = {result.projected_correction_S_m:.8g} S/m")
    print(f"same-generator Green-Kubo = {result.green_kubo_conductivity_S_m:.8g} S/m")
    print(
        "same-generator Einstein-Helfand = "
        f"{result.einstein_helfand_conductivity_S_m:.8g} S/m"
    )
    print(f"basis sequence = {result.basis_conductivities_S_m}")
    print(f"residual sequence = {result.residual_history}")
    print(f"basis size = {result.basis_size}; samples = {result.sample_count}; ESS = {result.effective_sample_size:.6g}")
    print(
        "overdamped acceptance fraction = "
        f"{result.overdamped_acceptance_fraction:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
