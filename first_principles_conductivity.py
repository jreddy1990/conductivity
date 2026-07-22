"""Full-configuration reversible conductivity from an analytical molecular model.

The executable constructs one periodic molecular liquid, samples its Boltzmann
measure, and solves the reversible Smoluchowski current-corrector problem in a
nested basis of smooth full-configuration observables.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from itertools import combinations_with_replacement
import math
from pathlib import Path
import sys
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
from utils.time_series_statistics import select_stationary_suffix

Array = np.ndarray
CARTESIAN_DIMENSION = 3
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
    logarithmic_volume_proposal: float
    hamiltonian_timestep_s: float
    memory_equilibration_steps: int
    memory_production_steps: int
    memory_sample_stride: int
    memory_laplace_rate_per_s: float
    maximum_relative_energy_drift: float


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
    basis_angular_order: int
    basis_cluster_depth: int
    basis_correlation_order: int
    basis_radial_cutoff_m: float
    maximum_basis_size: int
    eigenvalue_relative_tolerance: float
    memory_psd_relative_tolerance: float
    memory_plateau_relative_tolerance: float
    memory_diffusive_exponent_tolerance: float
    memory_lag_window_count: int
    residual_tolerance: float
    conductivity_tolerance_S_m: float
    minimum_effective_sample_size: float
    minimum_interatomic_contact_ratio: float
    stationarity_standard_error_limit: float


@dataclass(frozen=True)
class ConductivityResult:
    conductivity_S_m: float
    direct_current_term_S_m: float
    projected_correction_S_m: float
    equilibrium_volume_m3: float
    equilibrium_density_kg_m3: float
    integrated_memory_eigenvalues_kg_s: tuple[float, ...]
    diffusion_eigenvalues_m2_s: tuple[float, ...]
    basis_size: int
    basis_conductivities_S_m: tuple[float, ...]
    residual_history: tuple[float, ...]
    maximum_residual_score: float
    equilibrium_sample_count: int
    memory_sample_count: int
    effective_sample_size: float


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
    torsions: tuple[tuple[int, int, int, int, tuple[tuple[float, int, float], ...]], ...]
    nonbonded_mask: Array


@dataclass(frozen=True)
class MolecularMemoryOperator:
    integrated_friction_kg_s: Array
    diffusion_m2_s: Array
    physical_range_projector: Array
    memory_scale_radial_edges_m: Array
    memory_scale_coefficients: Array
    lag_times_s: tuple[float, ...]
    diffusion_plateau_relative_change: float
    displacement_growth_exponent: float
    sample_count: int


def minimum_image_displacement(displacement_m: Array, box_vectors_m: Array) -> Array:
    fractional = np.asarray(displacement_m) @ np.linalg.inv(box_vectors_m)
    fractional -= np.rint(fractional)
    return fractional @ box_vectors_m


def molecular_translation_projector(system: MolecularSystem) -> Array:
    molecule_count = len(system.molecule_atom_indices)
    translation_modes = np.zeros(
        (CARTESIAN_DIMENSION * molecule_count, CARTESIAN_DIMENSION)
    )
    for molecule_index in range(molecule_count):
        molecule_slice = slice(
            CARTESIAN_DIMENSION * molecule_index,
            CARTESIAN_DIMENSION * (molecule_index + 1),
        )
        translation_modes[molecule_slice] = np.eye(CARTESIAN_DIMENSION)
    mode_gram_inverse = np.linalg.inv(translation_modes.T @ translation_modes)
    return (
        np.eye(CARTESIAN_DIMENSION * molecule_count)
        - translation_modes @ mode_gram_inverse @ translation_modes.T
    )


def estimate_molecular_memory_operator(
    molecular_velocities_m_s: Array,
    system: MolecularSystem,
    temperature_K: float,
    sample_interval_s: float,
    laplace_rate_per_s: float,
    eigenvalue_relative_tolerance: float,
    psd_relative_tolerance: float,
    plateau_relative_tolerance: float,
    diffusive_exponent_tolerance: float,
    lag_window_count: int,
) -> MolecularMemoryOperator:
    velocities = np.asarray(molecular_velocities_m_s, dtype=float)
    expected_shape = (
        velocities.shape[0],
        len(system.molecule_atom_indices),
        CARTESIAN_DIMENSION,
    )
    if velocities.shape != expected_shape or velocities.shape[0] < 2:
        raise ValueError(
            "molecular_velocities_m_s must contain at least two frames with "
            "shape (frames, molecules, 3)"
        )
    if not np.all(np.isfinite(velocities)):
        raise ValueError("molecular velocities must be finite")
    if temperature_K <= 0.0 or sample_interval_s <= 0.0:
        raise ValueError("temperature and memory sample interval must be positive")
    if laplace_rate_per_s <= 0.0:
        raise ValueError("memory Laplace rate must be positive")
    if lag_window_count < 2:
        raise ValueError("memory lag window count must be at least two")
    physical_range_projector = molecular_translation_projector(system)
    flattened_velocities = velocities.reshape((velocities.shape[0], -1))
    flattened_velocities -= np.mean(flattened_velocities, axis=0, keepdims=True)
    projected_velocities = flattened_velocities @ physical_range_projector
    maximum_lag_frames = min(
        velocities.shape[0] // 2,
        max(1, int(round(1.0 / (laplace_rate_per_s * sample_interval_s)))),
    )
    if maximum_lag_frames < lag_window_count:
        raise ValueError("trajectory is too short for the requested memory lag ladder")
    integrated_displacements_m = np.vstack(
        (
            np.zeros((1, projected_velocities.shape[1])),
            np.cumsum(projected_velocities * sample_interval_s, axis=0),
        )
    )
    lag_frames = np.unique(
        np.rint(
            np.linspace(
                maximum_lag_frames / lag_window_count,
                maximum_lag_frames,
                lag_window_count,
            )
        ).astype(int)
    )
    diffusion_sequence: list[Array] = []
    mean_square_displacements_m2: list[float] = []
    for lag_frame_count in lag_frames:
        displacement_samples_m = (
            integrated_displacements_m[lag_frame_count:]
            - integrated_displacements_m[:-lag_frame_count]
        )
        displacement_samples_m -= np.mean(
            displacement_samples_m, axis=0, keepdims=True
        )
        lag_time_s = lag_frame_count * sample_interval_s
        displacement_covariance_m2 = (
            displacement_samples_m.T
            @ displacement_samples_m
            / displacement_samples_m.shape[0]
        )
        diffusion_at_lag = displacement_covariance_m2 / (2.0 * lag_time_s)
        diffusion_sequence.append(0.5 * (diffusion_at_lag + diffusion_at_lag.T))
        mean_square_displacements_m2.append(float(np.trace(displacement_covariance_m2)))
    diffusion = diffusion_sequence[-1]
    diffusion_scale = max(float(np.linalg.norm(diffusion)), np.finfo(float).tiny)
    diffusion_plateau_relative_change = float(
        np.linalg.norm(diffusion_sequence[-1] - diffusion_sequence[-2])
        / diffusion_scale
    )
    lag_times_s = sample_interval_s * lag_frames
    displacement_growth_exponent = float(
        np.log(
            mean_square_displacements_m2[-1]
            / mean_square_displacements_m2[-2]
        )
        / np.log(lag_times_s[-1] / lag_times_s[-2])
    )
    if (
        diffusion_plateau_relative_change > plateau_relative_tolerance
        or abs(displacement_growth_exponent - 1.0) > diffusive_exponent_tolerance
    ):
        raise ValueError(
            "molecular displacement has no diffusive plateau: relative diffusion "
            f"change={diffusion_plateau_relative_change:.6g}, growth "
            f"exponent={displacement_growth_exponent:.6g}"
        )
    diffusion_eigenvalues, diffusion_eigenvectors = np.linalg.eigh(diffusion)
    diffusion_scale = max(
        float(np.max(np.abs(diffusion_eigenvalues))), np.finfo(float).tiny
    )
    if float(np.min(diffusion_eigenvalues)) < (
        -psd_relative_tolerance * diffusion_scale
    ):
        raise ValueError("zero-frequency molecular diffusion is not positive semidefinite")
    retained_modes = diffusion_eigenvalues > (
        eigenvalue_relative_tolerance * diffusion_scale
    )
    diffusion_inverse = (
        diffusion_eigenvectors[:, retained_modes]
        / diffusion_eigenvalues[retained_modes]
    ) @ diffusion_eigenvectors[:, retained_modes].T
    integrated_friction = (
        K_B
        * temperature_K
        * physical_range_projector
        @ diffusion_inverse
        @ physical_range_projector
    )
    integrated_friction = 0.5 * (integrated_friction + integrated_friction.T)
    return MolecularMemoryOperator(
        integrated_friction_kg_s=integrated_friction,
        diffusion_m2_s=diffusion,
        physical_range_projector=physical_range_projector,
        memory_scale_radial_edges_m=np.empty(0),
        memory_scale_coefficients=np.asarray((0.0,)),
        lag_times_s=tuple(float(value) for value in lag_times_s),
        diffusion_plateau_relative_change=diffusion_plateau_relative_change,
        displacement_growth_exponent=displacement_growth_exponent,
        sample_count=velocities.shape[0],
    )


def _matching_lammps_operator(
    system: MolecularSystem,
    temperature_K: float,
    operator_data_root: Path,
) -> tuple[Path, dict]:
    if temperature_K <= 0.0:
        raise ValueError("temperature must be positive")
    operator_paths = tuple(
        sorted(operator_data_root.glob("*/replica_averaged_operator.npz"))
    )
    if not operator_paths:
        raise ValueError("LAMMPS operator corpus contains no averaged operators")
    system_labels = tuple(dict.fromkeys(system.molecule_species_names))
    system_family_indices = np.asarray(
        tuple(system_labels.index(label) for label in system.molecule_species_names),
        dtype=int,
    )
    system_family_counts = np.bincount(
        system_family_indices, minlength=len(system_labels)
    )
    matching_operators: list[tuple[Path, dict]] = []
    for operator_path in operator_paths:
        report_path = operator_path.with_suffix(".json")
        report = read_json_object(report_path, "LAMMPS averaged molecular operator")
        if report["admitted"] is not True:
            continue
        if report["diffusion_plateau_gate_passed"] is not True:
            raise ValueError(f"admitted operator lacks diffusion plateau: {operator_path}")
        if report["diffusion_psd_gate_passed"] is not True:
            raise ValueError(f"admitted operator is not PSD: {operator_path}")
        with np.load(operator_path) as operator:
            family_labels = tuple(str(value) for value in operator["family_labels"])
            operator_family_indices = np.asarray(
                operator["molecule_family_indices"], dtype=int
            )
        if set(family_labels) != set(system_labels):
            continue
        operator_family_counts = np.bincount(
            operator_family_indices, minlength=len(family_labels)
        )
        reordered_counts = np.asarray(
            tuple(
                operator_family_counts[family_labels.index(label)]
                for label in system_labels
            )
        )
        if not np.allclose(
            reordered_counts / np.sum(reordered_counts),
            system_family_counts / np.sum(system_family_counts),
            rtol=MILP_FEASIBILITY_TOLERANCE,
            atol=MILP_FEASIBILITY_TOLERANCE,
        ):
            continue
        if not math.isclose(
            float(report["temperature_K"]),
            temperature_K,
            rel_tol=MILP_FEASIBILITY_TOLERANCE,
        ):
            continue
        matching_operators.append((operator_path, report))
    if len(matching_operators) != 1:
        raise ValueError(
            "LAMMPS operator corpus must contain exactly one admitted operator "
            "matching molecular composition and temperature"
        )
    return matching_operators[0]


def molecular_memory_from_lammps_operator_data(
    system: MolecularSystem,
    temperature_K: float,
    operator_data_root: Path,
    eigenvalue_relative_tolerance: float,
) -> MolecularMemoryOperator:
    operator_path, report = _matching_lammps_operator(
        system=system,
        temperature_K=temperature_K,
        operator_data_root=operator_data_root,
    )
    system_labels = tuple(dict.fromkeys(system.molecule_species_names))
    system_family_indices = np.asarray(
        tuple(system_labels.index(label) for label in system.molecule_species_names),
        dtype=int,
    )
    system_family_counts = np.bincount(
        system_family_indices, minlength=len(system_labels)
    )
    with np.load(operator_path) as operator:
        family_labels = tuple(str(value) for value in operator["family_labels"])
        collective_diffusion = np.asarray(operator["diffusion_tensor_m2_s"])
        self_diffusion = np.asarray(
            operator["molecular_self_memory_diffusion_m2_s"]
        )
        lag_times_s = np.asarray(operator["lag_times_s"])
        lag_sample_counts = np.asarray(operator["lag_sample_counts"])
        memory_scale_radial_edges_m = (
            np.asarray(operator["geometry_radial_bin_edges_A"], dtype=float) * 1.0e-10
        )
        memory_scale_coefficients = np.asarray(
            operator["conditional_memory_scale_coefficients"], dtype=float
        )
    family_order = tuple(family_labels.index(label) for label in system_labels)
    coordinate_order = np.concatenate(
        tuple(
            np.arange(3 * family_index, 3 * (family_index + 1))
            for family_index in family_order
        )
    )
    collective_diffusion = collective_diffusion[
        np.ix_(coordinate_order, coordinate_order)
    ]
    self_diffusion = self_diffusion[np.asarray(family_order)]
    molecule_count = len(system.molecule_atom_indices)
    unprojected_diffusion = np.zeros((3 * molecule_count, 3 * molecule_count))
    for first_molecule_index, first_family_index in enumerate(system_family_indices):
        first_slice = slice(3 * first_molecule_index, 3 * (first_molecule_index + 1))
        for second_molecule_index, second_family_index in enumerate(
            system_family_indices
        ):
            second_slice = slice(
                3 * second_molecule_index, 3 * (second_molecule_index + 1)
            )
            if first_molecule_index == second_molecule_index:
                block = self_diffusion[first_family_index]
            elif first_family_index == second_family_index:
                family_count = system_family_counts[first_family_index]
                family_slice = slice(
                    3 * first_family_index, 3 * (first_family_index + 1)
                )
                block = (
                    family_count
                    * collective_diffusion[family_slice, family_slice]
                    - self_diffusion[first_family_index]
                ) / (family_count - 1)
            else:
                block = collective_diffusion[
                    3 * first_family_index : 3 * (first_family_index + 1),
                    3 * second_family_index : 3 * (second_family_index + 1),
                ]
            unprojected_diffusion[first_slice, second_slice] = block
    physical_range_projector = molecular_translation_projector(system)
    diffusion = (
        physical_range_projector
        @ unprojected_diffusion
        @ physical_range_projector
    )
    diffusion = 0.5 * (diffusion + diffusion.T)
    diffusion_inverse = symmetric_psd_pseudoinverse(
        diffusion, eigenvalue_relative_tolerance
    )
    integrated_friction = K_B * temperature_K * diffusion_inverse
    return MolecularMemoryOperator(
        integrated_friction_kg_s=integrated_friction,
        diffusion_m2_s=diffusion,
        physical_range_projector=physical_range_projector,
        memory_scale_radial_edges_m=memory_scale_radial_edges_m,
        memory_scale_coefficients=memory_scale_coefficients,
        lag_times_s=tuple(float(value) for value in lag_times_s),
        diffusion_plateau_relative_change=float(
            report["maximum_lag_ladder_relative_change"]
        ),
        displacement_growth_exponent=1.0,
        sample_count=int(lag_sample_counts[0]),
    )


def configuration_conditioned_molecular_diffusion(
    positions_m: Array,
    system: MolecularSystem,
    molecular_memory: MolecularMemoryOperator,
) -> Array:
    radial_edges_m = np.asarray(
        molecular_memory.memory_scale_radial_edges_m, dtype=float
    )
    coefficients = np.asarray(molecular_memory.memory_scale_coefficients, dtype=float)
    if radial_edges_m.size == 0:
        return molecular_memory.diffusion_m2_s
    if coefficients.shape != (radial_edges_m.size,):
        raise ValueError("memory scale coefficients do not match radial bins")
    molecule_centers_m = np.asarray(
        tuple(
            np.average(
                positions_m[molecule_atom_indices],
                axis=0,
                weights=system.masses_kg[molecule_atom_indices],
            )
            for molecule_atom_indices in system.molecule_atom_indices
        )
    )
    first_indices, second_indices = np.triu_indices(molecule_centers_m.shape[0], k=1)
    displacements_m = minimum_image_displacement(
        molecule_centers_m[second_indices] - molecule_centers_m[first_indices],
        system.box_vectors_m,
    )
    distances_m = np.linalg.norm(displacements_m, axis=1)
    radial_bins = np.searchsorted(radial_edges_m, distances_m, side="right") - 1
    admitted = (radial_bins >= 0) & (radial_bins < radial_edges_m.size - 1)
    local_environment = np.zeros(
        (molecule_centers_m.shape[0], radial_edges_m.size - 1), dtype=float
    )
    np.add.at(local_environment, (first_indices[admitted], radial_bins[admitted]), 1.0)
    np.add.at(local_environment, (second_indices[admitted], radial_bins[admitted]), 1.0)
    logarithmic_friction_scale = coefficients[0] + local_environment @ coefficients[1:]
    logarithmic_friction_scale -= np.mean(logarithmic_friction_scale)
    inverse_sqrt_scale = np.repeat(
        np.exp(-0.5 * logarithmic_friction_scale), CARTESIAN_DIMENSION
    )
    conditioned_diffusion = (
        inverse_sqrt_scale[:, None]
        * molecular_memory.diffusion_m2_s
        * inverse_sqrt_scale[None, :]
    )
    conditioned_diffusion = (
        molecular_memory.physical_range_projector
        @ conditioned_diffusion
        @ molecular_memory.physical_range_projector
    )
    conditioned_diffusion = 0.5 * (
        conditioned_diffusion + conditioned_diffusion.T
    )
    if not np.all(np.isfinite(conditioned_diffusion)):
        raise ValueError("configuration-conditioned diffusion is non-finite")
    return conditioned_diffusion


def lammps_equilibrium_projection_data(
    composition_system: MolecularSystem,
    temperature_K: float,
    operator_data_root: Path,
    eigenvalue_relative_tolerance: float,
) -> tuple[MolecularSystem, Array, MolecularMemoryOperator]:
    operator_path, _report = _matching_lammps_operator(
        system=composition_system,
        temperature_K=temperature_K,
        operator_data_root=operator_data_root,
    )
    with np.load(operator_path) as operator:
        family_labels = tuple(str(value) for value in operator["family_labels"])
        family_indices = np.asarray(operator["molecule_family_indices"], dtype=int)
        configurations_m = (
            np.asarray(operator["geometry_sample_molecular_com_A"], dtype=float)
            * 1.0e-10
        )
        box_vectors_m = (
            np.asarray(operator["geometry_sample_box_vectors_A"], dtype=float)
            * 1.0e-10
        )
        molecular_masses_kg = np.asarray(operator["molecular_masses_kg"], dtype=float)
        molecular_charges_C = np.asarray(operator["molecular_charges_C"], dtype=float)
    if configurations_m.ndim != 3 or configurations_m.shape[2] != 3:
        raise ValueError("LAMMPS equilibrium configurations have invalid shape")
    if box_vectors_m.shape != (configurations_m.shape[0], 3, 3):
        raise ValueError("LAMMPS equilibrium boxes do not match configurations")
    molecule_count = configurations_m.shape[1]
    if molecular_masses_kg.shape != (molecule_count,):
        raise ValueError("LAMMPS molecular masses do not match configurations")
    if molecular_charges_C.shape != (molecule_count,):
        raise ValueError("LAMMPS molecular charges do not match configurations")
    equilibrium_box_vectors_m = np.mean(box_vectors_m, axis=0)
    scaled_configurations_m = configurations_m @ np.linalg.inv(box_vectors_m)
    scaled_configurations_m = scaled_configurations_m @ equilibrium_box_vectors_m
    projection_system = MolecularSystem(
        positions_m=scaled_configurations_m[0],
        box_vectors_m=equilibrium_box_vectors_m,
        masses_kg=molecular_masses_kg,
        charges_C=molecular_charges_C,
        lj_sigma_m=np.zeros(molecule_count),
        lj_epsilon_J=np.zeros(molecule_count),
        polarizabilities_SI=np.zeros(molecule_count),
        molecule_index=np.arange(molecule_count),
        molecule_atom_indices=tuple(
            np.asarray((molecule_index,), dtype=int)
            for molecule_index in range(molecule_count)
        ),
        molecule_species_names=tuple(family_labels[index] for index in family_indices),
        bonds=np.empty((0, 2), dtype=int),
        bond_force_constants_J_m2=np.empty(0),
        bond_lengths_m=np.empty(0),
        angles=np.empty((0, 3), dtype=int),
        angle_force_constants_J_rad2=np.empty(0),
        angle_values_rad=np.empty(0),
        torsions=(),
        nonbonded_mask=~np.eye(molecule_count, dtype=bool),
    )
    memory = molecular_memory_from_lammps_operator_data(
        system=projection_system,
        temperature_K=temperature_K,
        operator_data_root=operator_data_root,
        eigenvalue_relative_tolerance=eigenvalue_relative_tolerance,
    )
    return projection_system, scaled_configurations_m, memory


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
    initial_volume_m3 = sum(
        int(count) * float(record["partial_molar_volume_m3_mol"]) / N_A
        for count, record in zip(counts, species_records, strict=True)
    )
    if initial_volume_m3 <= 0.0:
        raise ValueError("species partial molar volumes do not define a positive cell")
    box_length_m = initial_volume_m3 ** (1.0 / CARTESIAN_DIMENSION)
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
            molecule_species_names.append(species_name)
            for site in record["sites"]:
                masses.append(float(site["mass_kg"]))
                charges.append(float(site["charge_number"]) * E_CHARGE)
                lj_sigma.append(float(site["lj_sigma_m"]))
                lj_epsilon.append(float(site["lj_epsilon_J"]))
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
        lj_sigma_m=np.asarray(lj_sigma), lj_epsilon_J=np.asarray(lj_epsilon),
        polarizabilities_SI=np.asarray(polarizabilities), molecule_index=np.asarray(molecule_indices),
        molecule_atom_indices=tuple(molecule_atom_indices),
        molecule_species_names=tuple(molecule_species_names),
        bonds=np.asarray(bonds, dtype=int).reshape((-1, 2)),
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
        electrostatic_pair_i, electrostatic_pair_j = np.triu_indices(
            positions_m.shape[0], 1
        )
        electrostatic_pair_i_tensor = torch.as_tensor(electrostatic_pair_i)
        electrostatic_pair_j_tensor = torch.as_tensor(electrostatic_pair_j)
        electrostatic_pair_distances_m = distance[
            electrostatic_pair_i_tensor, electrostatic_pair_j_tensor
        ]
        real_pair_energies_J = (
            charges[electrostatic_pair_i_tensor]
            * charges[electrostatic_pair_j_tensor]
            * torch.special.erfc(
                ewald_alpha * electrostatic_pair_distances_m
            )
            / (
                4.0
                * math.pi
                * EPS_0
                * electrostatic_pair_distances_m
            )
        )
        energy += torch.sum(real_pair_energies_J)
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
        energy -= (
            ewald_alpha
            * torch.sum(charges**2)
            / (4.0 * math.pi * math.sqrt(math.pi) * EPS_0)
        )
        excluded_pair_mask = ~system.nonbonded_mask[
            electrostatic_pair_i, electrostatic_pair_j
        ]
        if np.any(excluded_pair_mask):
            excluded_pair_indices = np.flatnonzero(excluded_pair_mask)
            excluded_i = electrostatic_pair_i[excluded_pair_indices]
            excluded_j = electrostatic_pair_j[excluded_pair_indices]
            excluded_displacements_m = displacement[excluded_i, excluded_j]
            excluded_phases = excluded_displacements_m @ reciprocal.T
            excluded_reciprocal_energies_J = (
                torch.as_tensor(system.charges_C[excluded_i])
                * torch.as_tensor(system.charges_C[excluded_j])
                * torch.sum(
                    green_weights[None, :] * torch.cos(excluded_phases),
                    dim=1,
                )
            )
            energy -= torch.sum(
                real_pair_energies_J[torch.as_tensor(excluded_pair_indices)]
                + excluded_reciprocal_energies_J
            )
        if np.any(system.polarizabilities_SI > 0.0):
            energy += self._polarization_energy(
                positions_m,
                reciprocal,
                green_weights,
                ewald_alpha,
                box_vectors_m,
            )
        return energy

    def _polarization_energy(
        self,
        positions_m: torch.Tensor,
        reciprocal_m_inv: torch.Tensor,
        green_weights_J_m_C2: torch.Tensor,
        ewald_splitting_per_m: float,
        box_vectors_m: torch.Tensor,
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
        displacement_m = _torch_minimum_image(
            positions_m[:, None, :] - positions_m[None, :, :],
            box_vectors_m,
        )
        distance_m = torch.linalg.norm(displacement_m, dim=2)
        nonzero_distance_m = torch.where(
            torch.eye(distance_m.shape[0], dtype=torch.bool),
            torch.ones_like(distance_m),
            distance_m,
        )
        real_field_coefficient = (
            torch.special.erfc(ewald_splitting_per_m * nonzero_distance_m)
            / nonzero_distance_m**3
            + 2.0
            * ewald_splitting_per_m
            * torch.exp(
                -(ewald_splitting_per_m * nonzero_distance_m) ** 2
            )
            / (math.sqrt(math.pi) * nonzero_distance_m**2)
        ) / (4.0 * math.pi * EPS_0)
        real_field_coefficient = torch.where(
            torch.as_tensor(self.system.nonbonded_mask),
            real_field_coefficient,
            torch.zeros_like(real_field_coefficient),
        )
        electric_fields += torch.einsum(
            "ij,j,ijd->id",
            real_field_coefficient,
            charges,
            displacement_m,
        )
        excluded_mask = ~self.system.nonbonded_mask
        np.fill_diagonal(excluded_mask, False)
        if np.any(excluded_mask):
            excluded_mask_tensor = torch.as_tensor(excluded_mask)
            excluded_reciprocal_fields = torch.einsum(
                "ijk,k,kd,j->id",
                torch.sin(phase_differences)
                * excluded_mask_tensor[:, :, None],
                green_weights_J_m_C2,
                reciprocal_m_inv,
                charges,
            )
            electric_fields -= excluded_reciprocal_fields
        active_tensor = torch.as_tensor(active)
        active_phase_differences = phase_differences[active_tensor][:, active_tensor]
        interaction_blocks = -torch.einsum(
            "ijk,k,kd,ke->ijde",
            torch.cos(active_phase_differences),
            green_weights_J_m_C2,
            reciprocal_m_inv,
            reciprocal_m_inv,
        )
        active_displacements_m = displacement_m[active_tensor][:, active_tensor]
        active_distances_m = nonzero_distance_m[active_tensor][:, active_tensor]
        active_allowed = torch.as_tensor(
            self.system.nonbonded_mask[np.ix_(active, active)]
        )
        radial_first_derivative = -(
            torch.special.erfc(
                ewald_splitting_per_m * active_distances_m
            )
            / active_distances_m**2
            + 2.0
            * ewald_splitting_per_m
            * torch.exp(
                -(ewald_splitting_per_m * active_distances_m) ** 2
            )
            / (math.sqrt(math.pi) * active_distances_m)
        ) / (4.0 * math.pi * EPS_0)
        radial_second_derivative = (
            2.0
            * torch.special.erfc(
                ewald_splitting_per_m * active_distances_m
            )
            / active_distances_m**3
            + 4.0
            * ewald_splitting_per_m
            * torch.exp(
                -(ewald_splitting_per_m * active_distances_m) ** 2
            )
            / (math.sqrt(math.pi) * active_distances_m**2)
            + 4.0
            * ewald_splitting_per_m**3
            * torch.exp(
                -(ewald_splitting_per_m * active_distances_m) ** 2
            )
            / math.sqrt(math.pi)
        ) / (4.0 * math.pi * EPS_0)
        unit_displacements = (
            active_displacements_m / active_distances_m[:, :, None]
        )
        radial_outer = (
            unit_displacements[:, :, :, None]
            * unit_displacements[:, :, None, :]
        )
        identity = torch.eye(CARTESIAN_DIMENSION)
        real_hessian = (
            radial_second_derivative[:, :, None, None] * radial_outer
            + (radial_first_derivative / active_distances_m)[:, :, None, None]
            * (identity[None, None, :, :] - radial_outer)
        )
        real_hessian = torch.where(
            active_allowed[:, :, None, None],
            real_hessian,
            torch.zeros_like(real_hessian),
        )
        interaction_blocks += real_hessian
        active_excluded = ~self.system.nonbonded_mask[np.ix_(active, active)]
        np.fill_diagonal(active_excluded, False)
        if np.any(active_excluded):
            excluded_active_tensor = torch.as_tensor(active_excluded)
            reciprocal_excluded_hessian = -torch.einsum(
                "ijk,k,kd,ke->ijde",
                torch.cos(active_phase_differences)
                * excluded_active_tensor[:, :, None],
                green_weights_J_m_C2,
                reciprocal_m_inv,
                reciprocal_m_inv,
            )
            interaction_blocks -= reciprocal_excluded_hessian
        self_hessian = (
            -4.0
            * ewald_splitting_per_m**3
            / (3.0 * math.sqrt(math.pi) * 4.0 * math.pi * EPS_0)
        )
        diagonal_indices = torch.arange(active.size)
        interaction_blocks[diagonal_indices, diagonal_indices] += (
            self_hessian * identity
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


def sample_isothermal_isobaric_equilibrium(
    model: AnalyticalPeriodicInteratomicModel,
    temperature_K: float,
    pressure_Pa: float,
    dynamics: DynamicsSettings,
    random_seed: int,
) -> tuple[AnalyticalPeriodicInteratomicModel, Array, Array]:
    if pressure_Pa <= 0.0:
        raise ValueError("pressure_Pa must be positive")
    random_generator = np.random.default_rng(random_seed)
    positions_m = relax_initial_configuration(model, temperature_K, dynamics)
    box_vectors_m = model.system.box_vectors_m.copy()
    inverse_thermal_energy_J = 1.0 / (K_B * temperature_K)
    energy_J = model.energy_J(positions_m, box_vectors_m)
    molecule_count = len(model.system.molecule_atom_indices)
    sampled_volumes_m3: list[float] = []
    sampled_positions_m: list[Array] = []
    total_sweeps = (
        dynamics.equilibrium_burn_in_sweeps
        + dynamics.equilibrium_sample_count
        * dynamics.equilibrium_sweeps_per_sample
    )
    for sweep_index in range(total_sweeps):
        for molecule_atom_indices in model.system.molecule_atom_indices:
            proposal_positions_m = positions_m.copy()
            proposal_positions_m[molecule_atom_indices] += random_generator.normal(
                scale=dynamics.translation_proposal_m,
                size=CARTESIAN_DIMENSION,
            )
            proposal_positions_m %= np.diag(box_vectors_m)
            proposal_energy_J = model.energy_J(
                proposal_positions_m, box_vectors_m
            )
            log_acceptance = -inverse_thermal_energy_J * (
                proposal_energy_J - energy_J
            )
            if math.log(random_generator.random()) < log_acceptance:
                positions_m = proposal_positions_m
                energy_J = proposal_energy_J

        current_volume_m3 = abs(np.linalg.det(box_vectors_m))
        logarithmic_volume_change = random_generator.normal(
            scale=dynamics.logarithmic_volume_proposal
        )
        proposal_volume_m3 = current_volume_m3 * math.exp(
            logarithmic_volume_change
        )
        length_scale = (proposal_volume_m3 / current_volume_m3) ** (
            1.0 / CARTESIAN_DIMENSION
        )
        proposal_box_vectors_m = box_vectors_m * length_scale
        proposal_positions_m = positions_m.copy()
        for molecule_atom_indices in model.system.molecule_atom_indices:
            molecule_positions_m = positions_m[molecule_atom_indices]
            anchor_m = molecule_positions_m[0]
            unwrapped_molecule_positions_m = anchor_m + minimum_image_displacement(
                molecule_positions_m - anchor_m,
                box_vectors_m,
            )
            molecule_masses_kg = model.system.masses_kg[molecule_atom_indices]
            center_of_mass_m = np.average(
                unwrapped_molecule_positions_m,
                axis=0,
                weights=molecule_masses_kg,
            )
            proposal_positions_m[molecule_atom_indices] = (
                unwrapped_molecule_positions_m
                - center_of_mass_m
                + length_scale * center_of_mass_m
            )
        proposal_positions_m %= np.diag(proposal_box_vectors_m)
        proposal_energy_J = model.energy_J(
            proposal_positions_m, proposal_box_vectors_m
        )
        log_volume_acceptance = (
            -inverse_thermal_energy_J
            * (
                proposal_energy_J
                - energy_J
                + pressure_Pa * (proposal_volume_m3 - current_volume_m3)
            )
            + molecule_count * logarithmic_volume_change
        )
        if math.log(random_generator.random()) < log_volume_acceptance:
            positions_m = proposal_positions_m
            box_vectors_m = proposal_box_vectors_m
            energy_J = proposal_energy_J

        sample_offset = sweep_index - dynamics.equilibrium_burn_in_sweeps
        if (
            sample_offset >= 0
            and sample_offset % dynamics.equilibrium_sweeps_per_sample == 0
        ):
            sampled_volumes_m3.append(abs(np.linalg.det(box_vectors_m)))
            sampled_positions_m.append(positions_m.copy())

    volume_series_m3 = np.asarray(sampled_volumes_m3)
    stationary_volume = select_stationary_suffix(
        values=volume_series_m3,
        maximum_split_mean_difference_standard_errors=(
            model.numerics.stationarity_standard_error_limit
        ),
        maximum_linear_drift_standard_errors=(
            model.numerics.stationarity_standard_error_limit
        ),
        minimum_effective_sample_size=model.numerics.minimum_effective_sample_size,
    )
    stationary_volumes_m3 = volume_series_m3[stationary_volume.start_index :]
    equilibrium_volume_m3 = float(np.mean(stationary_volumes_m3))
    equilibrium_length_m = equilibrium_volume_m3 ** (
        1.0 / CARTESIAN_DIMENSION
    )
    equilibrium_box_vectors_m = np.eye(CARTESIAN_DIMENSION) * equilibrium_length_m
    source_positions_m = sampled_positions_m[-1]
    source_box_vectors_m = box_vectors_m
    volume_scale = (
        equilibrium_volume_m3 / abs(np.linalg.det(source_box_vectors_m))
    ) ** (1.0 / CARTESIAN_DIMENSION)
    equilibrium_positions_m = source_positions_m.copy()
    for molecule_atom_indices in model.system.molecule_atom_indices:
        molecule_positions_m = source_positions_m[molecule_atom_indices]
        anchor_m = molecule_positions_m[0]
        unwrapped_molecule_positions_m = anchor_m + minimum_image_displacement(
            molecule_positions_m - anchor_m,
            source_box_vectors_m,
        )
        molecule_masses_kg = model.system.masses_kg[molecule_atom_indices]
        center_of_mass_m = np.average(
            unwrapped_molecule_positions_m,
            axis=0,
            weights=molecule_masses_kg,
        )
        equilibrium_positions_m[molecule_atom_indices] = (
            unwrapped_molecule_positions_m
            - center_of_mass_m
            + volume_scale * center_of_mass_m
        )
    equilibrium_positions_m %= equilibrium_length_m
    equilibrium_system = replace(
        model.system,
        positions_m=equilibrium_positions_m,
        box_vectors_m=equilibrium_box_vectors_m,
    )
    equilibrium_model = AnalyticalPeriodicInteratomicModel(
        equilibrium_system, model.numerics
    )
    configurations_m = sample_equilibrium_configurations(
        equilibrium_model,
        temperature_K,
        dynamics,
        random_seed + 1,
    )
    return equilibrium_model, configurations_m, stationary_volumes_m3


def remove_center_of_mass_momentum(
    velocities_m_s: Array, masses_kg: Array
) -> Array:
    center_of_mass_velocity_m_s = np.sum(
        masses_kg[:, None] * velocities_m_s, axis=0
    ) / float(np.sum(masses_kg))
    return velocities_m_s - center_of_mass_velocity_m_s


def velocity_verlet_step(
    model: AnalyticalPeriodicInteratomicModel,
    positions_m: Array,
    velocities_m_s: Array,
    timestep_s: float,
) -> tuple[Array, Array]:
    if timestep_s <= 0.0:
        raise ValueError("Hamiltonian timestep must be positive")
    forces_N = model.forces_N(positions_m, model.system.box_vectors_m)
    half_step_velocities_m_s = velocities_m_s + (
        0.5 * timestep_s * forces_N / model.system.masses_kg[:, None]
    )
    next_positions_m = (
        positions_m + timestep_s * half_step_velocities_m_s
    ) % np.diag(model.system.box_vectors_m)
    next_forces_N = model.forces_N(
        next_positions_m, model.system.box_vectors_m
    )
    next_velocities_m_s = half_step_velocities_m_s + (
        0.5
        * timestep_s
        * next_forces_N
        / model.system.masses_kg[:, None]
    )
    return next_positions_m, next_velocities_m_s


def sample_hamiltonian_molecular_velocities(
    model: AnalyticalPeriodicInteratomicModel,
    initial_positions_m: Array,
    temperature_K: float,
    dynamics: DynamicsSettings,
    random_seed: int,
) -> tuple[Array, float]:
    random_generator = np.random.default_rng(random_seed)
    thermal_velocity_scales_m_s = np.sqrt(
        K_B * temperature_K / model.system.masses_kg
    )
    velocities_m_s = random_generator.normal(
        scale=thermal_velocity_scales_m_s[:, None],
        size=initial_positions_m.shape,
    )
    velocities_m_s = remove_center_of_mass_momentum(
        velocities_m_s, model.system.masses_kg
    )
    positions_m = np.asarray(initial_positions_m, dtype=float).copy()
    initial_energy_J = model.energy_J(
        positions_m, model.system.box_vectors_m
    ) + 0.5 * float(
        np.sum(model.system.masses_kg[:, None] * velocities_m_s**2)
    )
    molecular_velocity_samples_m_s: list[Array] = []
    total_steps = (
        dynamics.memory_equilibration_steps + dynamics.memory_production_steps
    )
    for step_index in range(total_steps):
        positions_m, velocities_m_s = velocity_verlet_step(
            model,
            positions_m,
            velocities_m_s,
            dynamics.hamiltonian_timestep_s,
        )
        production_step_index = step_index - dynamics.memory_equilibration_steps
        if (
            production_step_index >= 0
            and production_step_index % dynamics.memory_sample_stride == 0
        ):
            molecular_velocities_m_s = np.asarray(
                tuple(
                    np.average(
                        velocities_m_s[molecule_atom_indices],
                        axis=0,
                        weights=model.system.masses_kg[molecule_atom_indices],
                    )
                    for molecule_atom_indices in model.system.molecule_atom_indices
                )
            )
            molecular_velocity_samples_m_s.append(molecular_velocities_m_s)
    final_energy_J = model.energy_J(
        positions_m, model.system.box_vectors_m
    ) + 0.5 * float(
        np.sum(model.system.masses_kg[:, None] * velocities_m_s**2)
    )
    relative_energy_drift = abs(final_energy_J - initial_energy_J) / max(
        abs(initial_energy_J), K_B * temperature_K
    )
    if relative_energy_drift > dynamics.maximum_relative_energy_drift:
        raise ValueError(
            "Hamiltonian trajectory relative energy drift "
            f"{relative_energy_drift:.6e} exceeds "
            f"{dynamics.maximum_relative_energy_drift:.6e}"
        )
    return np.asarray(molecular_velocity_samples_m_s), relative_energy_drift


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


def _basis_values_tensor(
    positions_m: torch.Tensor,
    system: MolecularSystem,
    numerics: NumericalSettings,
) -> torch.Tensor:
    box = torch.as_tensor(system.box_vectors_m)
    displacement = _torch_minimum_image(positions_m[:, None, :] - positions_m[None, :, :], box)
    distance = torch.linalg.norm(displacement + torch.eye(positions_m.shape[0])[:, :, None], dim=2)
    pair_i, pair_j = np.triu_indices(positions_m.shape[0], 1)
    pair_distance = distance[torch.as_tensor(pair_i), torch.as_tensor(pair_j)]
    pair_charge = torch.as_tensor(system.charges_C[pair_i] * system.charges_C[pair_j])
    primitive_features: list[torch.Tensor] = []
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
    primitive_features.extend(torch.unbind(total_internal_polarization))
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
        primitive_features.extend(
            (torch.sum(radial), torch.sum(pair_charge * radial))
        )
    molecule_centers: list[torch.Tensor] = []
    molecule_axes: list[torch.Tensor] = []
    molecule_charges: list[torch.Tensor] = []
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
        molecule_centers.append(center_of_mass)
        molecule_charges.append(torch.sum(charges[molecule_indices]))
        if len(molecule_atom_indices) > 1:
            molecular_axis = local_positions[-1] - local_positions[0]
            molecule_axes.append(
                molecular_axis
                / torch.sqrt(
                    torch.sum(molecular_axis**2) + torch.finfo(TORCH_DTYPE).tiny
                )
            )
        else:
            molecule_axes.append(torch.zeros(3, dtype=TORCH_DTYPE))
    center_tensor = torch.stack(molecule_centers)
    axis_tensor = torch.stack(molecule_axes)
    molecular_charge_tensor = torch.stack(molecule_charges)
    center_displacements = _torch_minimum_image(
        center_tensor[:, None, :] - center_tensor[None, :, :], box
    )
    center_distances = torch.linalg.norm(
        center_displacements
        + torch.eye(len(molecule_centers), dtype=TORCH_DTYPE)[:, :, None],
        dim=2,
    )
    center_pair_i, center_pair_j = np.triu_indices(len(molecule_centers), 1)
    center_pair_i_tensor = torch.as_tensor(center_pair_i)
    center_pair_j_tensor = torch.as_tensor(center_pair_j)
    pair_directions = center_displacements[
        center_pair_i_tensor, center_pair_j_tensor
    ] / center_distances[center_pair_i_tensor, center_pair_j_tensor, None]
    pair_axis_projection = torch.sum(
        axis_tensor[center_pair_i_tensor] * pair_directions, dim=1
    )
    pair_axis_alignment = torch.sum(
        axis_tensor[center_pair_i_tensor] * axis_tensor[center_pair_j_tensor],
        dim=1,
    )
    for angular_order in range(1, numerics.basis_angular_order + 1):
        primitive_features.extend(
            (
                torch.sum(pair_axis_projection**angular_order),
                torch.sum(pair_axis_alignment**angular_order),
            )
        )
    cluster_adjacency = torch.exp(
        -(
            center_distances
            / numerics.basis_radial_cutoff_m
        ) ** 2
    ) * (1.0 - torch.eye(len(molecule_centers), dtype=TORCH_DTYPE))
    cluster_power = cluster_adjacency
    for _cluster_depth in range(1, numerics.basis_cluster_depth + 1):
        primitive_features.extend(
            (
                torch.trace(cluster_power),
                molecular_charge_tensor
                @ cluster_power
                @ molecular_charge_tensor,
            )
        )
        cluster_power = cluster_power @ cluster_adjacency
    reciprocal_base = 2.0 * math.pi * torch.linalg.inv(box)
    for shell in range(1, numerics.basis_fourier_shell + 1):
        for axis in range(3):
            reciprocal = shell * reciprocal_base[axis]
            phases = positions_m @ reciprocal
            primitive_features.extend(
                (
                    torch.sum(torch.cos(phases)),
                    torch.sum(torch.sin(phases)),
                    torch.sum(charges * torch.cos(phases)),
                    torch.sum(charges * torch.sin(phases)),
                )
            )
    normalized_primitives = [
        feature / (torch.abs(feature.detach()) + 1.0)
        for feature in primitive_features
    ]
    features = list(normalized_primitives)
    for correlation_order in range(2, numerics.basis_correlation_order + 1):
        for feature_indices in combinations_with_replacement(
            range(len(normalized_primitives)), correlation_order
        ):
            product = torch.ones((), dtype=TORCH_DTYPE)
            for feature_index in feature_indices:
                product = product * normalized_primitives[feature_index]
            features.append(product)
            if len(features) >= numerics.maximum_basis_size:
                return torch.stack(features)
    return torch.stack(features[: numerics.maximum_basis_size])


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


def molecular_diffusion_in_atomic_coordinates(
    system: MolecularSystem, molecular_diffusion_m2_s: Array
) -> Array:
    molecule_count = len(system.molecule_atom_indices)
    expected_shape = (
        CARTESIAN_DIMENSION * molecule_count,
        CARTESIAN_DIMENSION * molecule_count,
    )
    diffusion = np.asarray(molecular_diffusion_m2_s, dtype=float)
    if diffusion.shape != expected_shape:
        raise ValueError(
            f"molecular diffusion must have shape {expected_shape}, got "
            f"{diffusion.shape}"
        )
    translation_lift = np.zeros(
        (CARTESIAN_DIMENSION * system.charges_C.size, expected_shape[0])
    )
    for molecule_index, molecule_atom_indices in enumerate(
        system.molecule_atom_indices
    ):
        molecule_slice = slice(
            CARTESIAN_DIMENSION * molecule_index,
            CARTESIAN_DIMENSION * (molecule_index + 1),
        )
        for atom_index in molecule_atom_indices:
            atom_slice = slice(
                CARTESIAN_DIMENSION * int(atom_index),
                CARTESIAN_DIMENSION * (int(atom_index) + 1),
            )
            translation_lift[atom_slice, molecule_slice] = np.eye(
                CARTESIAN_DIMENSION
            )
    atomic_diffusion = translation_lift @ diffusion @ translation_lift.T
    return 0.5 * (atomic_diffusion + atomic_diffusion.T)


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
    molecular_memory: MolecularMemoryOperator,
    numerics: NumericalSettings,
) -> tuple[float, float, tuple[float, ...], tuple[float, ...], int]:
    minimum_partition_count = 2
    if configurations_m.shape[0] < 2 * minimum_partition_count:
        raise ValueError("projection requires at least four equilibrium samples")
    _basis_values, gradients = evaluate_basis_and_gradients(
        configurations_m, system, numerics
    )
    basis_count = min(gradients.shape[1], numerics.maximum_basis_size)
    polarization_gradients = molecular_com_charge_polarization_gradients(system)
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
            molecular_diffusion_m2_s = configuration_conditioned_molecular_diffusion(
                positions_m=configurations_m[sample_index],
                system=system,
                molecular_memory=molecular_memory,
            )
            atomic_diffusion_m2_s = molecular_diffusion_in_atomic_coordinates(
                system, molecular_diffusion_m2_s
            )
            sample_gradients = gradients[sample_index, :basis_count]
            diffused_gradients = sample_gradients @ atomic_diffusion_m2_s
            dirichlet += diffused_gradients @ sample_gradients.T
            coupling += diffused_gradients @ polarization_gradients.T
            diffused_polarization = (
                polarization_gradients @ atomic_diffusion_m2_s
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
        dynamics.logarithmic_volume_proposal,
        dynamics.maximum_relative_energy_drift,
        numerics.memory_psd_relative_tolerance,
        numerics.minimum_interatomic_contact_ratio,
    )
    if any(value >= 1.0 for value in fractions):
        raise ValueError("fractional numerical settings must be below one")
    if dynamics.memory_sample_stride > dynamics.memory_production_steps:
        raise ValueError("memory sample stride exceeds production trajectory")


def compute_first_principles_conductivity(
    recipe: ElectrolyteRecipeModel,
    temperature_K: float,
    pressure_Pa: float,
    molecule_count: int,
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
    random_seed: int,
) -> ConductivityResult:
    _validate_settings(dynamics, numerics)
    if temperature_K <= 0.0 or pressure_Pa <= 0.0 or molecule_count <= 0:
        raise ValueError("temperature, pressure, and molecule count must be positive")
    system = build_periodic_molecular_system(
        recipe=recipe,
        molecule_count=molecule_count,
        minimum_interatomic_contact_ratio=(
            numerics.minimum_interatomic_contact_ratio
        ),
        initial_placement_attempts_per_molecule=(
            numerics.initial_placement_attempts_per_molecule
        ),
        random_seed=random_seed,
    )
    equilibrium_system, configurations, molecular_memory = (
        lammps_equilibrium_projection_data(
            composition_system=system,
            temperature_K=temperature_K,
            operator_data_root=(
                Path(__file__).parent / "physical_library" / "lammps_operator_data"
            ),
            eigenvalue_relative_tolerance=numerics.eigenvalue_relative_tolerance,
        )
    )
    conductivity, direct, history, residuals, basis_size = projected_conductivity_sequence(
        configurations_m=configurations,
        system=equilibrium_system,
        temperature_K=temperature_K,
        molecular_memory=molecular_memory,
        numerics=numerics,
    )
    equilibrium_volume_m3 = float(abs(np.linalg.det(equilibrium_system.box_vectors_m)))
    equilibrium_density_kg_m3 = float(
        np.sum(equilibrium_system.masses_kg) / equilibrium_volume_m3
    )
    return ConductivityResult(
        conductivity_S_m=conductivity,
        direct_current_term_S_m=direct,
        projected_correction_S_m=direct - conductivity,
        equilibrium_volume_m3=equilibrium_volume_m3,
        equilibrium_density_kg_m3=equilibrium_density_kg_m3,
        integrated_memory_eigenvalues_kg_s=tuple(
            float(value)
            for value in np.linalg.eigvalsh(
                molecular_memory.integrated_friction_kg_s
            )
        ),
        diffusion_eigenvalues_m2_s=tuple(
            float(value)
            for value in np.linalg.eigvalsh(molecular_memory.diffusion_m2_s)
        ),
        basis_size=basis_size,
        basis_conductivities_S_m=history,
        residual_history=residuals,
        maximum_residual_score=residuals[-1],
        equilibrium_sample_count=configurations.shape[0],
        memory_sample_count=molecular_memory.sample_count,
        effective_sample_size=float(configurations.shape[0]),
    )


def _settings_from_record(record: dict) -> tuple[DynamicsSettings, NumericalSettings]:
    return DynamicsSettings(**record["dynamics"]), NumericalSettings(**record["numerics"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-json", required=True, type=Path)
    parser.add_argument("--numerics-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    arguments = parser.parse_args()
    recipe = ElectrolyteRecipeModel.model_validate(read_json_object(arguments.recipe_json, "electrolyte recipe"))
    settings_record = read_json_object(arguments.numerics_json, "conductivity numerics")
    dynamics, numerics = _settings_from_record(settings_record)
    temperature_K = float(settings_record["temperature_K"])
    pressure_Pa = float(settings_record["pressure_Pa"])
    molecule_count = int(settings_record["molecule_count"])
    random_seed = int(settings_record["random_seed"])
    result = compute_first_principles_conductivity(
        recipe=recipe,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        molecule_count=molecule_count,
        dynamics=dynamics,
        numerics=numerics,
        random_seed=random_seed,
    )
    write_json_object(arguments.output_json, asdict(result), "conductivity result")
    print(f"conductivity = {result.conductivity_S_m:.8g} S/m ({result.conductivity_S_m * S_M_TO_MS_CM:.8g} mS/cm)")
    print(f"direct = {result.direct_current_term_S_m:.8g} S/m")
    print(f"projected correction = {result.projected_correction_S_m:.8g} S/m")
    print(f"equilibrium volume = {result.equilibrium_volume_m3:.8g} m3")
    print(f"equilibrium density = {result.equilibrium_density_kg_m3:.8g} kg/m3")
    print(f"basis sequence = {result.basis_conductivities_S_m}")
    print(f"residual sequence = {result.residual_history}")
    print(
        f"basis size = {result.basis_size}; equilibrium samples = "
        f"{result.equilibrium_sample_count}; memory samples = "
        f"{result.memory_sample_count}; ESS = {result.effective_sample_size:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
