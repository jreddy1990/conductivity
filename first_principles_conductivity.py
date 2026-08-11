"""Full-configuration reversible conductivity from an analytical molecular model.

The executable constructs one periodic molecular liquid, samples its Boltzmann
measure, and solves the reversible Smoluchowski current-corrector problem in a
nested basis of smooth full-configuration observables.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import cache
import hashlib
from itertools import combinations, combinations_with_replacement
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time
from typing import Protocol
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import Bounds, LinearConstraint, milp, nnls
from scipy.special import erfcinv
from scipy.stats import norm, rankdata
import torch
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from constants import (
    ANGSTROM_TO_M,
    E_CHARGE,
    EPS_0,
    K_B,
    KCAL_TO_J,
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
from utils.time_series_statistics import autocorrelation_and_effective_sample_size

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
MILP_FEASIBILITY_TOLERANCE = 100.0 * math.sqrt(np.finfo(float).eps)
# Quarter scaling separates the lambda and square-root-lambda energy coefficients.
COMPONENT_DECOMPOSITION_LAMBDA = 0.25
MINIMUM_OPERATOR_DIAGNOSTIC_PILOT_SAMPLES = 2 * CARTESIAN_DIMENSION
MINIMUM_OPERATOR_DIAGNOSTIC_EVALUATION_SAMPLES = CARTESIAN_DIMENSION + 1
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


def multivariate_batch_means_effective_sample_size(
    chain_operator_series: Array,
) -> float:
    chains = np.asarray(chain_operator_series, dtype=float)
    if chains.ndim != 3 or chains.shape[0] < 2 or chains.shape[1] < 4:
        raise ValueError("operator chains must have shape (chain, sample, observable)")
    chain_count, sample_count, observable_count = chains.shape
    batch_size = max(2, int(math.floor(math.sqrt(sample_count))))
    batch_count = sample_count // batch_size
    if batch_count < 2:
        raise ValueError("operator chains have too few complete batches")
    retained_sample_count = batch_count * batch_size
    retained = chains[:, :retained_sample_count]
    flattened = retained.reshape(-1, observable_count)
    centered = flattened - np.mean(flattened, axis=0)
    pooled_marginal_covariance = centered.T @ centered / (flattened.shape[0] - 1)
    pooled_mean_covariance = np.zeros((observable_count, observable_count), dtype=float)
    chain_weight = 1.0 / chain_count
    for chain_samples in retained:
        batch_means = chain_samples.reshape(
            batch_count, batch_size, observable_count
        ).mean(axis=1)
        batch_centered = batch_means - np.mean(batch_means, axis=0)
        long_run_covariance = (
            batch_size * (batch_centered.T @ batch_centered) / (batch_count - 1)
        )
        pooled_mean_covariance += (
            chain_weight**2 * long_run_covariance / retained_sample_count
        )
    marginal_eigenvalues = np.linalg.eigvalsh(pooled_marginal_covariance)
    mean_covariance_eigenvalues = np.linalg.eigvalsh(pooled_mean_covariance)
    active_count = min(
        observable_count,
        flattened.shape[0] - 1,
        batch_count - 1,
        int(np.count_nonzero(marginal_eigenvalues > np.finfo(float).eps)),
    )
    if active_count == 0:
        return 0.0
    marginal_active = marginal_eigenvalues[-active_count:]
    mean_covariance_active = mean_covariance_eigenvalues[-active_count:]
    if np.any(mean_covariance_active <= 0.0):
        return 0.0
    log_determinant_ratio = float(
        np.sum(np.log(marginal_active)) - np.sum(np.log(mean_covariance_active))
    )
    effective_sample_size = math.exp(log_determinant_ratio / active_count)
    total_sample_count = chain_count * retained_sample_count
    return float(min(total_sample_count, max(1.0, effective_sample_size)))


def rank_normalized_split_rhat(chain_operator_series: Array) -> float:
    chains = np.asarray(chain_operator_series, dtype=float)
    if chains.ndim != 3 or chains.shape[0] < 2 or chains.shape[1] < 4:
        raise ValueError("operator chains must have shape (chain, sample, observable)")
    half_sample_count = chains.shape[1] // 2
    split_chains = np.concatenate(
        (
            chains[:, :half_sample_count, :],
            chains[:, -half_sample_count:, :],
        ),
        axis=0,
    )
    split_chain_count, split_sample_count, observable_count = split_chains.shape
    maximum_rhat = 1.0
    for observable_index in range(observable_count):
        flattened = split_chains[:, :, observable_index].reshape(-1)
        ranks = rankdata(flattened, method="average")
        probabilities = (ranks - 0.5) / ranks.size
        normalized = norm.ppf(probabilities).reshape(
            split_chain_count, split_sample_count
        )
        chain_means = np.mean(normalized, axis=1)
        between_variance = split_sample_count * np.var(chain_means, ddof=1)
        within_variance = float(np.mean(np.var(normalized, axis=1, ddof=1)))
        if within_variance <= 0.0:
            if float(np.var(normalized)) <= 0.0:
                continue
            return math.inf
        variance_estimate = (
            (split_sample_count - 1) * within_variance + between_variance
        ) / split_sample_count
        maximum_rhat = max(maximum_rhat, math.sqrt(variance_estimate / within_variance))
    return float(maximum_rhat)


def fit_operator_diagnostic_basis(
    chain_operator_series: Array,
    eigenvalue_relative_tolerance: float,
    maximum_mode_count: int,
) -> OperatorDiagnosticBasis:
    chains = np.asarray(chain_operator_series, dtype=float)
    if (
        chains.ndim != 3
        or chains.shape[0] < 2
        or chains.shape[1] < MINIMUM_OPERATOR_DIAGNOSTIC_PILOT_SAMPLES
    ):
        raise ValueError(
            "operator diagnostic chains must have shape (chain, sample, observable)"
        )
    if eigenvalue_relative_tolerance <= 0.0 or maximum_mode_count <= 0:
        raise ValueError("operator diagnostic subspace controls must be positive")
    selection_samples = chains.reshape(-1, chains.shape[2])
    selection_mean = np.mean(selection_samples, axis=0)
    centered_selection = selection_samples - selection_mean
    covariance = (
        centered_selection.T @ centered_selection / (centered_selection.shape[0] - 1)
    )
    covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(covariance)
    covariance_scale = max(float(np.max(covariance_eigenvalues)), np.finfo(float).tiny)
    active_indices = np.flatnonzero(
        covariance_eigenvalues > eigenvalue_relative_tolerance * covariance_scale
    )
    if active_indices.size == 0:
        raise ValueError("operator diagnostic selection found no active modes")
    retained_indices = active_indices[-maximum_mode_count:]
    retained_vectors = covariance_eigenvectors[:, retained_indices]
    return OperatorDiagnosticBasis(mean=selection_mean, loadings=retained_vectors)


def _rank_normalized_split_statistics(
    scalar_chains: Array,
) -> tuple[float, tuple[float, ...], tuple[float, ...], float, float]:
    chains = np.asarray(scalar_chains, dtype=float)
    if chains.ndim != 2 or chains.shape[0] < 2 or chains.shape[1] < 4:
        raise ValueError("scalar chains require at least two chains and four samples")
    half_sample_count = chains.shape[1] // 2
    split_chains = np.concatenate(
        (chains[:, :half_sample_count], chains[:, -half_sample_count:]), axis=0
    )
    flattened = split_chains.reshape(-1)
    ranks = rankdata(flattened, method="average")
    normalized = norm.ppf((ranks - 0.5) / ranks.size).reshape(split_chains.shape)
    split_means = np.mean(normalized, axis=1)
    split_variances = np.var(normalized, axis=1, ddof=1)
    split_sample_count = split_chains.shape[1]
    between_variance = float(split_sample_count * np.var(split_means, ddof=1))
    within_variance = float(np.mean(split_variances))
    rhat = math.inf
    if within_variance > 0.0:
        variance_estimate = (
            (split_sample_count - 1) * within_variance + between_variance
        ) / split_sample_count
        rhat = math.sqrt(variance_estimate / within_variance)
    if within_variance <= 0.0 and float(np.var(normalized)) <= 0.0:
        rhat = 1.0
    return (
        float(rhat),
        tuple(float(value) for value in split_means),
        tuple(float(value) for value in split_variances),
        within_variance,
        between_variance,
    )


def fixed_operator_mode_diagnostics(
    chain_operator_series: Array,
    diagnostic_basis: OperatorDiagnosticBasis,
    dirichlet_diagonal_count: int,
    coupling_count: int,
    direct_count: int,
) -> tuple[OperatorModeDiagnostic, ...]:
    chains = np.asarray(chain_operator_series, dtype=float)
    expected_observable_count = dirichlet_diagonal_count + coupling_count + direct_count
    if chains.ndim != 3 or chains.shape[2] != expected_observable_count:
        raise ValueError("operator chains do not match diagnostic loading partitions")
    projected = (chains - diagnostic_basis.mean) @ diagnostic_basis.loadings
    diagnostics: list[OperatorModeDiagnostic] = []
    coupling_start = dirichlet_diagonal_count
    direct_start = coupling_start + coupling_count
    for mode_index in range(projected.shape[2]):
        scalar_chains = projected[:, :, mode_index]
        (
            bulk_rhat,
            split_means,
            split_variances,
            within_variance,
            between_variance,
        ) = _rank_normalized_split_statistics(scalar_chains)
        folded_chains = np.abs(scalar_chains - np.median(scalar_chains))
        folded_rhat = _rank_normalized_split_statistics(folded_chains)[0]
        effective_sample_size = float(
            sum(
                autocorrelation_and_effective_sample_size(chain).effective_sample_size
                for chain in scalar_chains
            )
        )
        loadings = diagnostic_basis.loadings[:, mode_index]
        diagnostics.append(
            OperatorModeDiagnostic(
                mode_index=mode_index,
                bulk_rhat=bulk_rhat,
                folded_rhat=folded_rhat,
                effective_sample_size=effective_sample_size,
                split_chain_means=split_means,
                split_chain_variances=split_variances,
                within_variance=within_variance,
                between_variance=between_variance,
                loadings_on_A_diagonal=loadings[:coupling_start].copy(),
                loadings_on_h=loadings[coupling_start:direct_start].copy(),
                loadings_on_direct=loadings[direct_start:].copy(),
            )
        )
    return tuple(diagnostics)


def conductivity_influence_diagnostic(
    chain_complete_operator_series: Array,
    basis_count: int,
    temperature_K: float,
    volume_m3: float,
    eigenvalue_relative_tolerance: float,
) -> ConductivityInfluenceDiagnostic:
    chains = np.asarray(chain_complete_operator_series, dtype=float)
    dirichlet_size = basis_count * basis_count
    coupling_size = basis_count * CARTESIAN_DIMENSION
    expected_size = dirichlet_size + coupling_size + CARTESIAN_DIMENSION
    if chains.ndim != 3 or chains.shape[2] != expected_size:
        raise ValueError("complete operator chains have incompatible dimensions")
    pooled_mean = np.mean(chains.reshape(-1, expected_size), axis=0)
    mean_dirichlet = pooled_mean[:dirichlet_size].reshape(basis_count, basis_count)
    mean_coupling = pooled_mean[
        dirichlet_size : dirichlet_size + coupling_size
    ].reshape(basis_count, CARTESIAN_DIMENSION)
    mean_direct = pooled_mean[-CARTESIAN_DIMENSION:]
    dirichlet_inverse = symmetric_psd_pseudoinverse(
        mean_dirichlet, eigenvalue_relative_tolerance
    )
    coefficient_matrix = dirichlet_inverse @ mean_coupling
    dirichlet_sensitivity = coefficient_matrix @ coefficient_matrix.T
    prefactor = 1.0 / (CARTESIAN_DIMENSION * K_B * temperature_K * volume_m3)
    influence_chains = np.empty(chains.shape[:2], dtype=float)
    for chain_index in range(chains.shape[0]):
        chain_values = chains[chain_index]
        centered_dirichlet = (
            chain_values[:, :dirichlet_size].reshape(-1, basis_count, basis_count)
            - mean_dirichlet
        )
        centered_coupling = (
            chain_values[:, dirichlet_size : dirichlet_size + coupling_size].reshape(
                -1, basis_count, CARTESIAN_DIMENSION
            )
            - mean_coupling
        )
        centered_direct = chain_values[:, -CARTESIAN_DIMENSION:] - mean_direct
        influence_chains[chain_index] = prefactor * (
            np.sum(centered_direct, axis=1)
            - 2.0 * np.einsum("ba,sba->s", coefficient_matrix, centered_coupling)
            + np.einsum("bc,sbc->s", dirichlet_sensitivity, centered_dirichlet)
        )
    (
        bulk_rhat,
        split_means,
        split_variances,
        _within_variance,
        _between_variance,
    ) = _rank_normalized_split_statistics(influence_chains)
    folded_chains = np.abs(influence_chains - np.median(influence_chains))
    folded_rhat = _rank_normalized_split_statistics(folded_chains)[0]
    effective_sample_size = float(
        sum(
            autocorrelation_and_effective_sample_size(chain).effective_sample_size
            for chain in influence_chains
        )
    )
    return ConductivityInfluenceDiagnostic(
        bulk_rhat=bulk_rhat,
        folded_rhat=folded_rhat,
        effective_sample_size=effective_sample_size,
        split_chain_means=split_means,
        split_chain_variances=split_variances,
    )


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
    equilibrium_observable_relative_tolerance: float
    maximum_split_rhat: float
    pressure_log_volume_derivative_step: float
    pressure_log_volume_derivative_check_step: float
    pressure_derivative_relative_tolerance: float
    pressure_volume_relative_tolerance: float


@dataclass(frozen=True)
class InternalPressureResult:
    internal_pressure_Pa: float
    ideal_pressure_Pa: float
    configurational_pressure_Pa: float
    mean_energy_derivative_J: float
    relative_derivative_mismatch: float
    bonded_pressure_Pa: float
    lennard_jones_repulsion_pressure_Pa: float
    lennard_jones_attraction_pressure_Pa: float
    real_electrostatic_pressure_Pa: float
    reciprocal_electrostatic_pressure_Pa: float
    ewald_self_pressure_Pa: float
    electrostatic_exclusion_pressure_Pa: float
    polarization_pressure_Pa: float
    polarization_self_pressure_Pa: float
    polarization_exclusion_pressure_Pa: float


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
    direct_current_term_S_m: float
    projected_correction_S_m: float
    conditioned_volume_m3: float
    conditioned_density_g_cm3: float
    thermodynamic_state: str
    density_source: str
    integrated_memory_eigenvalues_kg_s: tuple[float, ...]
    diffusion_eigenvalues_m2_s: tuple[float, ...]
    basis_size: int
    basis_conductivities_S_m: tuple[float, ...]
    residual_history: tuple[float, ...]
    maximum_residual_score: float
    equilibrium_sample_count: int
    equilibrium_chain_count: int
    memory_sample_count: int
    effective_sample_size: float
    maximum_split_rhat: float
    conductivity_mcse_S_m: float
    conductivity_precision_reached: bool
    basis_diagnostics_certified: bool
    operator_diagnostics_certified: bool
    species_direct_contributions_S_m: tuple[tuple[str, float], ...]
    molecular_self_frictions_by_species_kg_s: tuple[tuple[str, float], ...]
    molecular_pair_frictions_by_species_kg_s: tuple[tuple[str, float], ...]
    memory_descriptor_leverage_by_species: tuple[tuple[str, float], ...]
    memory_pair_descriptor_leverage_by_species: tuple[tuple[str, float], ...]
    memory_conditional_fit_relative_error: float
    memory_conditional_heldout_relative_error: float
    realized_formula_unit_counts: tuple[tuple[str, int], ...]
    realized_molecule_counts: tuple[tuple[str, int], ...]
    realized_atom_count: int
    realized_solvent_volume_fractions: tuple[tuple[str, float], ...]
    realized_salt_molarities_mol_L: tuple[tuple[str, float], ...]
    realized_additive_weight_fractions: tuple[tuple[str, float], ...]
    realized_native_unit_deviations: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class OperatorDiagnosticBasis:
    mean: Array
    loadings: Array


@dataclass(frozen=True)
class OperatorModeDiagnostic:
    mode_index: int
    bulk_rhat: float
    folded_rhat: float
    effective_sample_size: float
    split_chain_means: tuple[float, ...]
    split_chain_variances: tuple[float, ...]
    within_variance: float
    between_variance: float
    loadings_on_A_diagonal: Array
    loadings_on_h: Array
    loadings_on_direct: Array


@dataclass(frozen=True)
class ConductivityInfluenceDiagnostic:
    bulk_rhat: float
    folded_rhat: float
    effective_sample_size: float
    split_chain_means: tuple[float, ...]
    split_chain_variances: tuple[float, ...]


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


@dataclass(frozen=True)
class MolecularMemoryOperator:
    integrated_friction_kg_s: Array
    diffusion_m2_s: Array
    physical_range_projector: Array
    molecular_self_frictions_kg_s: Array
    molecular_pair_frictions_kg_s: Array
    temperature_K: float
    decay_times_s: tuple[float, ...]
    decay_weights: tuple[float, ...]
    self_memory_spectral_amplitudes_kg_s2: Array
    geometry_radial_edges_m: Array
    self_descriptor_schema: tuple[str, ...]
    pair_descriptor_schema: tuple[str, ...]
    self_descriptor_friction_scales: Array
    pair_descriptor_friction_scales: Array
    lag_times_s: tuple[float, ...]
    diffusion_plateau_relative_change: float
    displacement_growth_exponent: float
    sample_count: int
    training_family_labels: tuple[str, ...]
    training_dataset_count: int
    descriptor_rank: int
    conditional_kernel_fit_relative_error: float
    conditional_kernel_heldout_relative_error: float
    molecular_descriptor_leverages: Array
    molecular_pair_descriptor_leverages: Array


@dataclass(frozen=True)
class LammpsFamilyMetadata:
    molecular_charge_e: float
    atom_count: int
    bond_count: int
    mole_fraction: float
    mean_lj_sigma_m: float
    mean_lj_epsilon_J: float
    molecule_count: int


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


def fit_transferable_molecular_memory_operator(
    system: MolecularSystem,
    temperature_K: float,
    operator_data_root: Path,
    eigenvalue_relative_tolerance: float,
) -> MolecularMemoryOperator:
    if temperature_K <= 0.0:
        raise ValueError("temperature must be positive")
    feature_rows: list[Array] = []
    logarithmic_friction_targets: list[float] = []
    pair_feature_rows: list[Array] = []
    logarithmic_pair_friction_targets: list[float] = []
    radial_edge_sets_A: list[Array] = []
    self_fit_friction_rows: list[Array] = []
    self_fit_weight_rows: list[Array] = []
    self_heldout_friction_rows: list[Array] = []
    self_heldout_weight_rows: list[Array] = []
    pair_fit_friction_rows: list[Array] = []
    pair_fit_weight_rows: list[Array] = []
    pair_heldout_friction_rows: list[Array] = []
    pair_heldout_weight_rows: list[Array] = []
    self_descriptor_schemas: list[tuple[str, ...]] = []
    pair_descriptor_schemas: list[tuple[str, ...]] = []
    retained_lag_times_s: set[float] = set()
    sample_count = 0
    maximum_plateau_change = 0.0
    decay_weight_rows: list[Array] = []
    training_family_labels: set[str] = set()
    training_dataset_count = 0
    for operator_path in sorted(
        operator_data_root.glob("*/replica_averaged_operator.npz")
    ):
        report = read_json_object(
            operator_path.with_suffix(".json"),
            "LAMMPS averaged molecular operator",
        )
        if report["plateau_gate_passed"] is not True:
            continue
        if report["tail_gate_passed"] is not True:
            continue
        if report["psd_gate_passed"] is not True:
            continue
        training_temperature_K = float(report["temperature_K"])
        family_metadata = _load_lammps_family_metadata(operator_path.parent)
        with np.load(operator_path) as operator:
            required_conditional_fields = (
                "self_descriptor_schema",
                "pair_descriptor_schema",
                "conditional_self_fit_weight_sums",
                "conditional_self_fit_force_products_N2",
                "conditional_self_heldout_weight_sums",
                "conditional_self_heldout_force_products_N2",
                "conditional_pair_fit_weight_sums",
                "conditional_pair_fit_force_products_N2",
                "conditional_pair_heldout_weight_sums",
                "conditional_pair_heldout_force_products_N2",
            )
            missing_conditional_fields = tuple(
                field_name
                for field_name in required_conditional_fields
                if field_name not in operator.files
            )
            if missing_conditional_fields:
                raise ValueError(
                    "LAMMPS operator lacks conditional lagged force statistics: "
                    f"{missing_conditional_fields}"
                )
            family_labels = tuple(str(value) for value in operator["family_labels"])
            family_masses_kg = np.asarray(operator["family_masses_kg"], dtype=float)
            memory_kernel_kg_s2 = np.asarray(
                operator["memory_kernel_kg_s2"], dtype=float
            )
            conditional_lag_times_s = np.asarray(operator["lag_times_s"], dtype=float)
            radial_edge_sets_A.append(
                np.asarray(operator["geometry_radial_bin_edges_A"], dtype=float)
            )
            self_descriptor_schemas.append(
                tuple(str(value) for value in operator["self_descriptor_schema"])
            )
            pair_descriptor_schemas.append(
                tuple(str(value) for value in operator["pair_descriptor_schema"])
            )
            positive_conditional_lag_times_s = conditional_lag_times_s[1:]
            conditional_relaxation_time_count = min(
                positive_conditional_lag_times_s.size,
                max(
                    len(self_descriptor_schemas[-1]),
                    len(pair_descriptor_schemas[-1]),
                ),
            )
            conditional_relaxation_times_s = np.geomspace(
                positive_conditional_lag_times_s[0],
                positive_conditional_lag_times_s[-1],
                conditional_relaxation_time_count,
            )
            conditional_exponential_basis = np.exp(
                -conditional_lag_times_s[:, None]
                / conditional_relaxation_times_s[None, :]
            )
            for (
                weight_field,
                product_field,
                friction_sign,
                target_rows,
                weight_rows,
            ) in (
                (
                    "conditional_self_fit_weight_sums",
                    "conditional_self_fit_force_products_N2",
                    1.0,
                    self_fit_friction_rows,
                    self_fit_weight_rows,
                ),
                (
                    "conditional_self_heldout_weight_sums",
                    "conditional_self_heldout_force_products_N2",
                    1.0,
                    self_heldout_friction_rows,
                    self_heldout_weight_rows,
                ),
                (
                    "conditional_pair_fit_weight_sums",
                    "conditional_pair_fit_force_products_N2",
                    -1.0,
                    pair_fit_friction_rows,
                    pair_fit_weight_rows,
                ),
                (
                    "conditional_pair_heldout_weight_sums",
                    "conditional_pair_heldout_force_products_N2",
                    -1.0,
                    pair_heldout_friction_rows,
                    pair_heldout_weight_rows,
                ),
            ):
                partition_weights = np.asarray(operator[weight_field], dtype=float)
                partition_products = np.asarray(operator[product_field], dtype=float)
                zero_lag_weights = partition_weights[0]
                conditional_covariances_N2 = partition_products / np.maximum(
                    partition_weights[:, :, np.newaxis, np.newaxis],
                    np.finfo(float).tiny,
                )
                conditional_covariance_traces_N2 = np.trace(
                    conditional_covariances_N2, axis1=2, axis2=3
                )
                conditional_friction_kernel_kg_s2 = (
                    friction_sign
                    * conditional_covariance_traces_N2
                    / (CARTESIAN_DIMENSION * K_B * training_temperature_K)
                )
                integrated_partition_frictions_kg_s = np.asarray(
                    tuple(
                        float(
                            nnls(
                                conditional_exponential_basis,
                                conditional_friction_kernel_kg_s2[:, descriptor_index],
                            )[0]
                            @ conditional_relaxation_times_s
                        )
                        for descriptor_index in range(
                            conditional_friction_kernel_kg_s2.shape[1]
                        )
                    )
                )
                resolved_descriptors = zero_lag_weights > 0.0
                if friction_sign > 0.0 and np.any(
                    integrated_partition_frictions_kg_s[resolved_descriptors] <= 0.0
                ):
                    raise ValueError("conditional self Mori friction must be positive")
                target_rows.append(integrated_partition_frictions_kg_s)
                weight_rows.append(zero_lag_weights)
            retained_lag_times_s.update(
                float(value) for value in operator["lag_times_s"]
            )
            sample_count += int(operator["lag_sample_counts"][0])
        maximum_plateau_change = max(
            maximum_plateau_change,
            float(report["maximum_lag_ladder_relative_change"]),
        )
        operator_lag_times_s = np.asarray(sorted(retained_lag_times_s))
        operator_lag_times_s = operator_lag_times_s[: memory_kernel_kg_s2.shape[0]]
        positive_lag_times_s = operator_lag_times_s[1:]
        relaxation_time_count = min(
            positive_lag_times_s.size,
            len(self_descriptor_schemas[-1]),
        )
        relaxation_times_s = np.geomspace(
            positive_lag_times_s[0],
            positive_lag_times_s[-1],
            relaxation_time_count,
        )
        exponential_basis = np.exp(
            -operator_lag_times_s[:, None] / relaxation_times_s[None, :]
        )
        for family_index, _family_mass_kg in enumerate(family_masses_kg):
            family_label = family_labels[family_index]
            if family_label not in family_metadata:
                raise ValueError(
                    f"LAMMPS metadata omits operator family {family_label}"
                )
            family_slice = slice(
                CARTESIAN_DIMENSION * family_index,
                CARTESIAN_DIMENSION * (family_index + 1),
            )
            isotropic_memory_kernel_kg_s2 = (
                np.trace(
                    memory_kernel_kg_s2[:, family_slice, family_slice],
                    axis1=1,
                    axis2=2,
                )
                / CARTESIAN_DIMENSION
            )
            spectral_amplitudes_kg_s2, _residual_norm = nnls(
                exponential_basis,
                isotropic_memory_kernel_kg_s2,
            )
            integrated_friction_kg_s = float(
                spectral_amplitudes_kg_s2 @ relaxation_times_s
            )
            if integrated_friction_kg_s <= 0.0:
                raise ValueError("fitted molecular friction must be positive")
            feature_rows.append(
                _molecular_friction_features(
                    molecular_mass_kg=float(family_masses_kg[family_index]),
                    metadata=family_metadata[family_label],
                    temperature_K=training_temperature_K,
                )
            )
            training_family_labels.add(family_label)
            logarithmic_friction_targets.append(math.log(integrated_friction_kg_s))
            decay_weight_rows.append(
                spectral_amplitudes_kg_s2
                * relaxation_times_s
                / integrated_friction_kg_s
            )
        family_features = tuple(
            _molecular_friction_features(
                molecular_mass_kg=float(family_masses_kg[family_index]),
                metadata=family_metadata[family_label],
                temperature_K=training_temperature_K,
            )
            for family_index, family_label in enumerate(family_labels)
        )
        for first_family, second_family in combinations(range(len(family_labels)), 2):
            first_slice = slice(
                CARTESIAN_DIMENSION * first_family,
                CARTESIAN_DIMENSION * (first_family + 1),
            )
            second_slice = slice(
                CARTESIAN_DIMENSION * second_family,
                CARTESIAN_DIMENSION * (second_family + 1),
            )
            cross_kernel = (
                -np.trace(
                    0.5
                    * (
                        memory_kernel_kg_s2[:, first_slice, second_slice]
                        + memory_kernel_kg_s2[:, second_slice, first_slice]
                    ),
                    axis1=1,
                    axis2=2,
                )
                / CARTESIAN_DIMENSION
            )
            cross_amplitudes_kg_s2, _cross_residual = nnls(
                exponential_basis, cross_kernel
            )
            family_pair_friction_kg_s = float(
                cross_amplitudes_kg_s2 @ relaxation_times_s
            )
            if family_pair_friction_kg_s <= 0.0:
                continue
            pair_count = (
                family_metadata[family_labels[first_family]].molecule_count
                * family_metadata[family_labels[second_family]].molecule_count
            )
            pair_feature_rows.append(
                _pair_friction_features(
                    family_features[first_family],
                    family_features[second_family],
                )
            )
            logarithmic_pair_friction_targets.append(
                math.log(family_pair_friction_kg_s / pair_count)
            )
        training_dataset_count += 1
    if not feature_rows:
        raise ValueError("LAMMPS corpus contains no shared admitted operators")
    reference_radial_edges_A = radial_edge_sets_A[0]
    if any(
        not np.array_equal(radial_edges_A, reference_radial_edges_A)
        for radial_edges_A in radial_edge_sets_A[1:]
    ):
        raise ValueError("shared operator corpus uses inconsistent radial bins")
    reference_self_schema = self_descriptor_schemas[0]
    if any(
        descriptor_schema != reference_self_schema
        for descriptor_schema in self_descriptor_schemas[1:]
    ):
        raise ValueError("shared operator corpus uses inconsistent self descriptors")
    reference_pair_schema = pair_descriptor_schemas[0]
    if any(
        descriptor_schema != reference_pair_schema
        for descriptor_schema in pair_descriptor_schemas[1:]
    ):
        raise ValueError("shared operator corpus uses inconsistent pair descriptors")
    feature_matrix = np.stack(feature_rows)
    descriptor_rank = int(np.linalg.matrix_rank(feature_matrix))
    self_fit_friction_array = np.stack(self_fit_friction_rows)
    self_fit_weight_array = np.stack(self_fit_weight_rows)
    pair_fit_friction_array = np.stack(pair_fit_friction_rows)
    pair_fit_weight_array = np.stack(pair_fit_weight_rows)
    self_weight_totals = np.sum(self_fit_weight_array, axis=0)
    pair_weight_totals = np.sum(pair_fit_weight_array, axis=0)
    self_resolved_descriptors = self_weight_totals > 0.0
    pair_resolved_descriptors = pair_weight_totals > 0.0
    self_friction_coefficients = np.zeros_like(self_weight_totals)
    pair_friction_coefficients = np.zeros_like(pair_weight_totals)
    self_friction_coefficients[self_resolved_descriptors] = (
        np.sum(self_fit_weight_array * self_fit_friction_array, axis=0)[
            self_resolved_descriptors
        ]
        / self_weight_totals[self_resolved_descriptors]
    )
    pair_friction_coefficients[pair_resolved_descriptors] = (
        np.sum(pair_fit_weight_array * pair_fit_friction_array, axis=0)[
            pair_resolved_descriptors
        ]
        / pair_weight_totals[pair_resolved_descriptors]
    )
    uniform_descriptor_index = reference_self_schema.index("uniform")
    self_descriptor_friction_scales = (
        self_friction_coefficients
        / self_friction_coefficients[uniform_descriptor_index]
    )
    pair_descriptor_friction_scales = pair_friction_coefficients / np.average(
        pair_friction_coefficients[pair_resolved_descriptors],
        weights=pair_weight_totals[pair_resolved_descriptors],
    )
    conditional_kernel_fit_relative_error = float(
        np.sqrt(
            np.linalg.norm(
                np.sqrt(self_fit_weight_array)
                * (self_friction_coefficients[None, :] - self_fit_friction_array)
            )
            ** 2
            + np.linalg.norm(
                np.sqrt(pair_fit_weight_array)
                * (pair_friction_coefficients[None, :] - pair_fit_friction_array)
            )
            ** 2
        )
        / max(
            float(
                np.sqrt(
                    np.linalg.norm(
                        np.sqrt(self_fit_weight_array) * self_fit_friction_array
                    )
                    ** 2
                    + np.linalg.norm(
                        np.sqrt(pair_fit_weight_array) * pair_fit_friction_array
                    )
                    ** 2
                )
            ),
            np.finfo(float).tiny,
        )
    )
    self_heldout_friction_array = np.stack(self_heldout_friction_rows)
    self_heldout_weight_array = np.stack(self_heldout_weight_rows)
    pair_heldout_friction_array = np.stack(pair_heldout_friction_rows)
    pair_heldout_weight_array = np.stack(pair_heldout_weight_rows)
    conditional_kernel_heldout_relative_error = float(
        np.sqrt(
            np.linalg.norm(
                np.sqrt(self_heldout_weight_array)
                * (self_friction_coefficients[None, :] - self_heldout_friction_array)
            )
            ** 2
            + np.linalg.norm(
                np.sqrt(pair_heldout_weight_array)
                * (pair_friction_coefficients[None, :] - pair_heldout_friction_array)
            )
            ** 2
        )
        / max(
            float(
                np.sqrt(
                    np.linalg.norm(
                        np.sqrt(self_heldout_weight_array) * self_heldout_friction_array
                    )
                    ** 2
                    + np.linalg.norm(
                        np.sqrt(pair_heldout_weight_array) * pair_heldout_friction_array
                    )
                    ** 2
                )
            ),
            np.finfo(float).tiny,
        )
    )
    species_counts = {
        species_name: system.molecule_species_names.count(species_name)
        for species_name in set(system.molecule_species_names)
    }
    molecule_count = len(system.molecule_atom_indices)
    prediction_features = np.stack(
        tuple(
            _molecular_friction_features(
                molecular_mass_kg=float(
                    np.sum(system.masses_kg[molecule_atom_indices])
                ),
                metadata=_system_family_metadata(
                    system=system,
                    molecule_atom_indices=molecule_atom_indices,
                    mole_fraction=species_counts[species_name] / molecule_count,
                ),
                temperature_K=temperature_K,
            )
            for species_name, molecule_atom_indices in zip(
                system.molecule_species_names,
                system.molecule_atom_indices,
                strict=True,
            )
        )
    )
    predicted_log_molecular_frictions, molecular_descriptor_leverages = (
        _regularized_log_friction_prediction(
            training_features=feature_matrix,
            logarithmic_training_targets=np.asarray(logarithmic_friction_targets),
            prediction_features=prediction_features,
        )
    )
    molecular_frictions_kg_s = np.exp(predicted_log_molecular_frictions)
    molecule_pair_indices = tuple(combinations(range(molecule_count), 2))
    pair_prediction_features = np.stack(
        tuple(
            _pair_friction_features(
                prediction_features[first_molecule],
                prediction_features[second_molecule],
            )
            for first_molecule, second_molecule in molecule_pair_indices
        )
    )
    predicted_log_pair_frictions, molecular_pair_descriptor_leverages = (
        _regularized_log_friction_prediction(
            training_features=np.stack(pair_feature_rows),
            logarithmic_training_targets=np.asarray(logarithmic_pair_friction_targets),
            prediction_features=pair_prediction_features,
        )
    )
    molecular_pair_frictions_kg_s = np.exp(predicted_log_pair_frictions)
    positive_retained_lag_times_s = np.asarray(
        tuple(sorted(value for value in retained_lag_times_s if value > 0.0))
    )
    decay_times_s = np.geomspace(
        positive_retained_lag_times_s[0],
        positive_retained_lag_times_s[-1],
        len(decay_weight_rows[0]),
    )
    decay_weights = np.mean(np.stack(decay_weight_rows), axis=0)
    decay_weights /= np.sum(decay_weights)
    self_memory_spectral_amplitudes_kg_s2 = (
        molecular_frictions_kg_s[:, None]
        * decay_weights[None, :]
        / decay_times_s[None, :]
    )
    unprojected_friction = np.diag(
        np.repeat(molecular_frictions_kg_s, CARTESIAN_DIMENSION)
    )
    molecular_pair_friction_matrix_kg_s = np.zeros(
        (molecule_count, molecule_count), dtype=float
    )
    for (
        first_molecule,
        second_molecule,
    ), pair_friction_kg_s in zip(
        molecule_pair_indices,
        molecular_pair_frictions_kg_s,
        strict=True,
    ):
        molecular_pair_friction_matrix_kg_s[first_molecule, second_molecule] = (
            pair_friction_kg_s
        )
        molecular_pair_friction_matrix_kg_s[second_molecule, first_molecule] = (
            pair_friction_kg_s
        )
        for axis in range(CARTESIAN_DIMENSION):
            first_coordinate = CARTESIAN_DIMENSION * first_molecule + axis
            second_coordinate = CARTESIAN_DIMENSION * second_molecule + axis
            unprojected_friction[first_coordinate, first_coordinate] += (
                pair_friction_kg_s
            )
            unprojected_friction[second_coordinate, second_coordinate] += (
                pair_friction_kg_s
            )
            unprojected_friction[first_coordinate, second_coordinate] -= (
                pair_friction_kg_s
            )
            unprojected_friction[second_coordinate, first_coordinate] -= (
                pair_friction_kg_s
            )
    physical_range_projector = molecular_translation_projector(system)
    integrated_friction = (
        physical_range_projector @ unprojected_friction @ physical_range_projector
    )
    integrated_friction = 0.5 * (integrated_friction + integrated_friction.T)
    diffusion = (
        K_B
        * temperature_K
        * symmetric_psd_pseudoinverse(
            integrated_friction, eigenvalue_relative_tolerance
        )
    )
    return MolecularMemoryOperator(
        integrated_friction_kg_s=integrated_friction,
        diffusion_m2_s=diffusion,
        physical_range_projector=physical_range_projector,
        molecular_self_frictions_kg_s=molecular_frictions_kg_s,
        molecular_pair_frictions_kg_s=molecular_pair_friction_matrix_kg_s,
        temperature_K=temperature_K,
        decay_times_s=tuple(float(value) for value in decay_times_s),
        decay_weights=tuple(float(value) for value in decay_weights),
        self_memory_spectral_amplitudes_kg_s2=(self_memory_spectral_amplitudes_kg_s2),
        geometry_radial_edges_m=reference_radial_edges_A * ANGSTROM_TO_M,
        self_descriptor_schema=reference_self_schema,
        pair_descriptor_schema=reference_pair_schema,
        self_descriptor_friction_scales=self_descriptor_friction_scales,
        pair_descriptor_friction_scales=pair_descriptor_friction_scales,
        lag_times_s=tuple(sorted(retained_lag_times_s)),
        diffusion_plateau_relative_change=maximum_plateau_change,
        displacement_growth_exponent=1.0,
        sample_count=sample_count,
        training_family_labels=tuple(sorted(training_family_labels)),
        training_dataset_count=training_dataset_count,
        descriptor_rank=descriptor_rank,
        conditional_kernel_fit_relative_error=conditional_kernel_fit_relative_error,
        conditional_kernel_heldout_relative_error=(
            conditional_kernel_heldout_relative_error
        ),
        molecular_descriptor_leverages=molecular_descriptor_leverages,
        molecular_pair_descriptor_leverages=molecular_pair_descriptor_leverages,
    )


def _regularized_log_friction_prediction(
    training_features: Array,
    logarithmic_training_targets: Array,
    prediction_features: Array,
) -> tuple[Array, Array]:
    descriptor_features = np.asarray(training_features[:, 1:], dtype=float)
    prediction_descriptors = np.asarray(prediction_features[:, 1:], dtype=float)
    descriptor_mean = np.mean(descriptor_features, axis=0)
    descriptor_scale = np.std(descriptor_features, axis=0)
    active_descriptors = descriptor_scale > math.sqrt(np.finfo(float).eps)
    if not np.any(active_descriptors):
        return (
            np.full(
                prediction_features.shape[0],
                np.mean(logarithmic_training_targets),
            ),
            np.zeros(prediction_features.shape[0]),
        )
    standardized_training = (
        descriptor_features[:, active_descriptors] - descriptor_mean[active_descriptors]
    ) / descriptor_scale[active_descriptors]
    standardized_prediction = (
        prediction_descriptors[:, active_descriptors]
        - descriptor_mean[active_descriptors]
    ) / descriptor_scale[active_descriptors]
    centered_targets = logarithmic_training_targets - np.mean(
        logarithmic_training_targets
    )
    descriptor_gram = standardized_training.T @ standardized_training
    spectral_penalty = float(np.trace(descriptor_gram) / descriptor_gram.shape[0])
    regularized_descriptor_gram = descriptor_gram + spectral_penalty * np.eye(
        descriptor_gram.shape[0]
    )
    coefficients = np.linalg.solve(
        regularized_descriptor_gram,
        standardized_training.T @ centered_targets,
    )
    descriptor_leverages = np.einsum(
        "bi,ij,bj->b",
        standardized_prediction,
        np.linalg.inv(regularized_descriptor_gram),
        standardized_prediction,
    )
    return (
        np.mean(logarithmic_training_targets) + standardized_prediction @ coefficients,
        descriptor_leverages,
    )


def _pair_friction_features(
    first_features: Array,
    second_features: Array,
) -> Array:
    return np.concatenate(
        (
            np.asarray((1.0,)),
            first_features[1:] + second_features[1:],
            np.abs(first_features[1:] - second_features[1:]),
        )
    )


def _load_lammps_family_metadata(
    operator_directory: Path,
) -> dict[str, LammpsFamilyMetadata]:
    replica_directory = operator_directory / "replica_1"
    copies_path = next(replica_directory.glob("*.copies.json"))
    composition_paths = tuple(
        path
        for path in replica_directory.glob("*.composition.json")
        if not path.name.endswith(".md_composition.json")
    )
    if len(composition_paths) != 1:
        raise ValueError(
            f"{replica_directory} must contain one recipe composition record"
        )
    copies = read_json_object(copies_path, "LAMMPS molecule copies")
    composition = read_json_object(composition_paths[0], "LAMMPS recipe composition")
    coverage = read_json_object(
        replica_directory / "species_forcefield_coverage.json",
        "LAMMPS force-field coverage",
    )
    topology_path = replica_directory / f"{operator_directory.name}.lmp"
    pair_coefficients, molecule_atom_types = _read_lammps_lj_topology(topology_path)
    species_list = tuple(str(value) for value in composition["species_list"])
    mole_fractions = tuple(float(value) for value in composition["mole_fractions"])
    if len(species_list) != len(mole_fractions):
        raise ValueError("LAMMPS species and mole-fraction lengths differ")
    species_ranges = {
        str(record["name"]): int(record["first_mol_id"])
        for record in copies["species_ranges"]
    }
    metadata: dict[str, LammpsFamilyMetadata] = {}
    for species_name, mole_fraction in zip(species_list, mole_fractions, strict=True):
        if species_name not in coverage or species_name not in species_ranges:
            raise ValueError(f"incomplete LAMMPS metadata for {species_name}")
        atom_types = molecule_atom_types[species_ranges[species_name]]
        lj_coefficients = np.asarray(
            tuple(pair_coefficients[atom_type] for atom_type in atom_types)
        )
        metadata[species_name] = LammpsFamilyMetadata(
            molecular_charge_e=float(coverage[species_name]["net_charge_e"]),
            atom_count=int(coverage[species_name]["atom_count"]),
            bond_count=int(coverage[species_name]["bond_count"]),
            mole_fraction=mole_fraction,
            mean_lj_sigma_m=(float(np.mean(lj_coefficients[:, 1])) * ANGSTROM_TO_M),
            mean_lj_epsilon_J=(float(np.mean(lj_coefficients[:, 0])) * KCAL_TO_J / N_A),
            molecule_count=int(composition["molecule_counts"][species_name]),
        )
    return metadata


def _read_lammps_lj_topology(
    topology_path: Path,
) -> tuple[dict[int, tuple[float, float]], dict[int, list[int]]]:
    pair_coefficients: dict[int, tuple[float, float]] = {}
    molecule_atom_types: dict[int, list[int]] = {}
    section = ""
    for line in topology_path.read_text().splitlines():
        stripped = line.strip()
        if stripped in {"Pair Coeffs", "Atoms"}:
            section = stripped
            continue
        if stripped in {"Bond Coeffs", "Bonds"}:
            section = ""
            continue
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if section == "Pair Coeffs" and len(fields) >= 3:
            pair_coefficients[int(fields[0])] = (
                float(fields[1]),
                float(fields[2]),
            )
        elif section == "Atoms" and len(fields) >= 4:
            molecule_atom_types.setdefault(int(fields[1]), []).append(int(fields[2]))
    if not pair_coefficients or not molecule_atom_types:
        raise ValueError(f"could not read LAMMPS LJ topology from {topology_path}")
    return pair_coefficients, molecule_atom_types


def _system_family_metadata(
    system: MolecularSystem,
    molecule_atom_indices: Array,
    mole_fraction: float,
) -> LammpsFamilyMetadata:
    atom_membership = np.zeros(system.positions_m.shape[0], dtype=bool)
    atom_membership[molecule_atom_indices] = True
    bond_count = int(
        np.sum(
            atom_membership[system.bonds[:, 0]] & atom_membership[system.bonds[:, 1]]
        )
    )
    return LammpsFamilyMetadata(
        molecular_charge_e=float(
            np.sum(system.charges_C[molecule_atom_indices]) / E_CHARGE
        ),
        atom_count=int(molecule_atom_indices.size),
        bond_count=bond_count,
        mole_fraction=mole_fraction,
        mean_lj_sigma_m=float(np.mean(system.lj_sigma_m[molecule_atom_indices])),
        mean_lj_epsilon_J=float(np.mean(system.lj_epsilon_J[molecule_atom_indices])),
        molecule_count=1,
    )


def _molecular_friction_features(
    molecular_mass_kg: float,
    metadata: LammpsFamilyMetadata,
    temperature_K: float,
) -> Array:
    positive_values = (
        molecular_mass_kg,
        float(metadata.atom_count),
        metadata.mole_fraction,
        metadata.mean_lj_sigma_m,
        temperature_K,
    )
    if any(value <= 0.0 for value in positive_values):
        raise ValueError("molecular friction descriptors must be positive")
    if metadata.bond_count < 0:
        raise ValueError("molecular bond count cannot be negative")
    if metadata.mean_lj_epsilon_J < 0.0:
        raise ValueError("Lennard-Jones well depth cannot be negative")
    return np.asarray(
        (
            1.0,
            math.log(molecular_mass_kg),
            metadata.molecular_charge_e,
            abs(metadata.molecular_charge_e),
            math.log(float(metadata.atom_count)),
            metadata.bond_count / metadata.atom_count,
            math.log(metadata.mole_fraction),
            math.log(metadata.mean_lj_sigma_m),
            metadata.mean_lj_epsilon_J / (K_B * temperature_K),
            math.log(temperature_K),
        )
    )


def configuration_conditioned_integrated_friction(
    positions_m: Array,
    system: MolecularSystem,
    molecular_memory: MolecularMemoryOperator,
) -> Array:
    radial_edges_m = np.asarray(molecular_memory.geometry_radial_edges_m, dtype=float)
    if radial_edges_m.size == 0:
        return molecular_memory.integrated_friction_kg_s
    self_scales = np.asarray(
        molecular_memory.self_descriptor_friction_scales, dtype=float
    )
    pair_scales = np.asarray(
        molecular_memory.pair_descriptor_friction_scales, dtype=float
    )
    if self_scales.shape != (len(molecular_memory.self_descriptor_schema),):
        raise ValueError("self friction scales do not match descriptor schema")
    if pair_scales.shape != (len(molecular_memory.pair_descriptor_schema),):
        raise ValueError("pair friction scales do not match descriptor schema")
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
    molecule_count = molecule_centers_m.shape[0]
    molecular_charges_C = np.asarray(
        tuple(
            np.sum(system.charges_C[molecule_atom_indices])
            for molecule_atom_indices in system.molecule_atom_indices
        )
    )
    molecular_radius_gyration_m2 = np.zeros(molecule_count)
    molecular_orientation_dyads = np.zeros(
        (molecule_count, CARTESIAN_DIMENSION, CARTESIAN_DIMENSION)
    )
    for molecule_index, molecule_atom_indices in enumerate(
        system.molecule_atom_indices
    ):
        local_positions_m = positions_m[molecule_atom_indices]
        anchor_m = local_positions_m[0]
        local_positions_m = anchor_m + minimum_image_displacement(
            local_positions_m - anchor_m, system.box_vectors_m
        )
        centered_positions_m = local_positions_m - molecule_centers_m[molecule_index]
        molecule_masses_kg = system.masses_kg[molecule_atom_indices]
        molecular_radius_gyration_m2[molecule_index] = np.average(
            np.sum(centered_positions_m**2, axis=1),
            weights=molecule_masses_kg,
        )
        if molecule_atom_indices.size > 1:
            gyration_tensor_m2 = np.einsum(
                "i,ia,ib->ab",
                molecule_masses_kg,
                centered_positions_m,
                centered_positions_m,
            ) / np.sum(molecule_masses_kg)
            eigenvalues_m2, eigenvectors = np.linalg.eigh(gyration_tensor_m2)
            if eigenvalues_m2[-1] > np.finfo(float).eps:
                principal_axis = eigenvectors[:, -1]
                molecular_orientation_dyads[molecule_index] = np.outer(
                    principal_axis, principal_axis
                )
    first_indices, second_indices = np.where(~np.eye(molecule_count, dtype=bool))
    displacements_m = minimum_image_displacement(
        molecule_centers_m[second_indices] - molecule_centers_m[first_indices],
        system.box_vectors_m,
    )
    distances_m = np.linalg.norm(displacements_m, axis=1)
    radial_bins = np.searchsorted(radial_edges_m, distances_m, side="right") - 1
    admitted = (radial_bins >= 0) & (radial_bins < radial_edges_m.size - 1)
    source_indices = first_indices[admitted]
    target_indices = second_indices[admitted]
    pair_distances_m = distances_m[admitted]
    pair_displacements_m = displacements_m[admitted]
    radial_bins = radial_bins[admitted]
    unit_vectors = pair_displacements_m / pair_distances_m[:, None]
    adjacency = np.zeros((molecule_count, molecule_count), dtype=float)
    adjacency[source_indices, target_indices] = 0.5 * (
        1.0 + np.cos(np.pi * pair_distances_m / radial_edges_m[-1])
    )
    cluster_degree = np.sum(adjacency, axis=1)
    cluster_depth_two = adjacency @ cluster_degree
    charge_scale_C = max(
        float(np.max(np.abs(molecular_charges_C))), np.finfo(float).tiny
    )
    positive_charge = np.maximum(molecular_charges_C, 0.0) / charge_scale_C
    negative_charge = np.maximum(-molecular_charges_C, 0.0) / charge_scale_C
    radius_scale_m2 = max(
        float(np.max(molecular_radius_gyration_m2)), np.finfo(float).tiny
    )
    orientation_alignment = np.maximum(
        np.einsum(
            "pa,pab,pb->p",
            unit_vectors,
            molecular_orientation_dyads[source_indices],
            unit_vectors,
        ),
        0.0,
    )
    common_neighbor = np.sum(
        adjacency[source_indices] * adjacency[target_indices], axis=1
    )
    self_descriptor_values = np.zeros(
        (molecule_count, len(molecular_memory.self_descriptor_schema))
    )
    self_descriptor_index = {
        name: index
        for index, name in enumerate(molecular_memory.self_descriptor_schema)
    }
    self_descriptor_values[:, self_descriptor_index["uniform"]] = 1.0
    self_descriptor_values[:, self_descriptor_index["molecular_radius_gyration_A2"]] = (
        molecular_radius_gyration_m2 / radius_scale_m2
    )
    self_descriptor_values[:, self_descriptor_index["smooth_cluster_degree"]] = (
        cluster_degree
    )
    self_descriptor_values[:, self_descriptor_index["smooth_cluster_depth_2"]] = (
        cluster_depth_two
    )
    pair_descriptor_values = np.zeros(
        (source_indices.size, len(molecular_memory.pair_descriptor_schema))
    )
    pair_descriptor_index = {
        name: index
        for index, name in enumerate(molecular_memory.pair_descriptor_schema)
    }
    for pair_index, (source_index, target_index, radial_bin_index) in enumerate(
        zip(source_indices, target_indices, radial_bins, strict=True)
    ):
        for descriptor_name, value in (
            (f"number_density_bin_{radial_bin_index}", 1.0),
            (
                f"positive_charge_density_bin_{radial_bin_index}",
                positive_charge[target_index],
            ),
            (
                f"negative_charge_density_bin_{radial_bin_index}",
                negative_charge[target_index],
            ),
            (
                f"orientation_axis_alignment_bin_{radial_bin_index}",
                orientation_alignment[pair_index],
            ),
        ):
            self_descriptor_values[
                source_index, self_descriptor_index[descriptor_name]
            ] += value
        for descriptor_name, value in (
            (f"pair_radial_bin_{radial_bin_index}", 1.0),
            (
                f"pair_orientation_alignment_bin_{radial_bin_index}",
                orientation_alignment[pair_index],
            ),
            (
                f"pair_common_neighbor_bin_{radial_bin_index}",
                common_neighbor[pair_index],
            ),
        ):
            pair_descriptor_values[
                pair_index, pair_descriptor_index[descriptor_name]
            ] = value
        charge_product_C2 = (
            molecular_charges_C[source_index] * molecular_charges_C[target_index]
        )
        charge_descriptor = (
            f"pair_unlike_charge_bin_{radial_bin_index}"
            if charge_product_C2 < 0.0
            else f"pair_like_charge_bin_{radial_bin_index}"
        )
        if charge_product_C2 != 0.0:
            pair_descriptor_values[
                pair_index, pair_descriptor_index[charge_descriptor]
            ] = 1.0
    self_normalization = np.sum(self_descriptor_values, axis=1)
    pair_normalization = np.sum(pair_descriptor_values, axis=1)
    if np.any(self_normalization <= 0.0) or np.any(pair_normalization <= 0.0):
        raise ValueError("configuration has unresolved memory descriptors")
    self_friction_scales = self_descriptor_values @ self_scales / self_normalization
    pair_friction_scales = pair_descriptor_values @ pair_scales / pair_normalization
    unprojected_friction = np.diag(
        np.repeat(
            molecular_memory.molecular_self_frictions_kg_s * self_friction_scales,
            CARTESIAN_DIMENSION,
        )
    )
    for pair_index, (source_index, target_index) in enumerate(
        zip(source_indices, target_indices, strict=True)
    ):
        if source_index >= target_index:
            continue
        pair_friction_kg_s = (
            molecular_memory.molecular_pair_frictions_kg_s[source_index, target_index]
            * 0.5
            * (
                pair_friction_scales[pair_index]
                + pair_friction_scales[
                    np.flatnonzero(
                        (source_indices == target_index)
                        & (target_indices == source_index)
                    )[0]
                ]
            )
        )
        for axis in range(CARTESIAN_DIMENSION):
            source_coordinate = CARTESIAN_DIMENSION * source_index + axis
            target_coordinate = CARTESIAN_DIMENSION * target_index + axis
            unprojected_friction[source_coordinate, source_coordinate] += (
                pair_friction_kg_s
            )
            unprojected_friction[target_coordinate, target_coordinate] += (
                pair_friction_kg_s
            )
            unprojected_friction[source_coordinate, target_coordinate] -= (
                pair_friction_kg_s
            )
            unprojected_friction[target_coordinate, source_coordinate] -= (
                pair_friction_kg_s
            )
    conditioned_friction = (
        molecular_memory.physical_range_projector
        @ unprojected_friction
        @ molecular_memory.physical_range_projector
    )
    conditioned_friction = 0.5 * (conditioned_friction + conditioned_friction.T)
    return conditioned_friction


def configuration_conditioned_molecular_memory_kernel(
    positions_m: Array,
    system: MolecularSystem,
    molecular_memory: MolecularMemoryOperator,
    time_s: float,
) -> Array:
    if time_s < 0.0:
        raise ValueError("memory-kernel time must be nonnegative")
    decay_times_s = np.asarray(molecular_memory.decay_times_s)
    decay_weights = np.asarray(molecular_memory.decay_weights)
    if decay_times_s.shape != decay_weights.shape or np.any(decay_times_s <= 0.0):
        raise ValueError("memory decay times and weights are inconsistent")
    if np.any(decay_weights < 0.0) or not np.isclose(np.sum(decay_weights), 1.0):
        raise ValueError("memory decay weights must form a probability vector")
    integrated_friction = configuration_conditioned_integrated_friction(
        positions_m, system, molecular_memory
    )
    kernel_scale_per_s = float(
        np.sum(decay_weights * np.exp(-time_s / decay_times_s) / decay_times_s)
    )
    return integrated_friction * kernel_scale_per_s


def configuration_conditioned_molecular_diffusion(
    positions_m: Array,
    system: MolecularSystem,
    molecular_memory: MolecularMemoryOperator,
) -> Array:
    return configuration_conditioned_molecular_diffusion_batch(
        positions_batch_m=positions_m[None, :, :],
        system=system,
        molecular_memory=molecular_memory,
    )[0]


def configuration_conditioned_molecular_diffusion_batch(
    positions_batch_m: Array,
    system: MolecularSystem,
    molecular_memory: MolecularMemoryOperator,
) -> Array:
    positions = np.asarray(positions_batch_m, dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != system.positions_m.shape:
        raise ValueError("conditioned diffusion batch has invalid position shape")
    radial_edges_m = np.asarray(molecular_memory.geometry_radial_edges_m, dtype=float)
    if radial_edges_m.size == 0:
        return np.repeat(
            molecular_memory.diffusion_m2_s[None, :, :],
            positions.shape[0],
            axis=0,
        )
    molecule_count = len(system.molecule_atom_indices)
    batch_size = positions.shape[0]
    molecule_centers_m = np.stack(
        tuple(
            np.average(
                positions[:, molecule_atom_indices, :],
                axis=1,
                weights=system.masses_kg[molecule_atom_indices],
            )
            for molecule_atom_indices in system.molecule_atom_indices
        ),
        axis=1,
    )
    molecular_radius_gyration_m2 = np.zeros((batch_size, molecule_count))
    molecular_orientation_dyads = np.zeros(
        (batch_size, molecule_count, CARTESIAN_DIMENSION, CARTESIAN_DIMENSION)
    )
    for molecule_index, molecule_atom_indices in enumerate(
        system.molecule_atom_indices
    ):
        local_positions_m = positions[:, molecule_atom_indices, :]
        anchor_m = local_positions_m[:, :1, :]
        local_positions_m = anchor_m + minimum_image_displacement(
            local_positions_m - anchor_m,
            system.box_vectors_m,
        )
        centered_positions_m = (
            local_positions_m - molecule_centers_m[:, molecule_index, None, :]
        )
        molecule_masses_kg = system.masses_kg[molecule_atom_indices]
        molecular_radius_gyration_m2[:, molecule_index] = np.average(
            np.sum(centered_positions_m**2, axis=2),
            axis=1,
            weights=molecule_masses_kg,
        )
        if molecule_atom_indices.size > 1:
            gyration_tensors_m2 = np.einsum(
                "i,bia,bic->bac",
                molecule_masses_kg,
                centered_positions_m,
                centered_positions_m,
            ) / np.sum(molecule_masses_kg)
            eigenvalues_m2, eigenvectors = np.linalg.eigh(gyration_tensors_m2)
            resolved = eigenvalues_m2[:, -1] > np.finfo(float).eps
            principal_axes = eigenvectors[:, :, -1]
            molecular_orientation_dyads[:, molecule_index] = np.where(
                resolved[:, None, None],
                np.einsum("ba,bc->bac", principal_axes, principal_axes),
                0.0,
            )
    first_indices, second_indices = np.where(~np.eye(molecule_count, dtype=bool))
    pair_displacements_m = minimum_image_displacement(
        molecule_centers_m[:, second_indices, :]
        - molecule_centers_m[:, first_indices, :],
        system.box_vectors_m,
    )
    pair_distances_m = np.linalg.norm(pair_displacements_m, axis=2)
    radial_bins = np.searchsorted(radial_edges_m, pair_distances_m, side="right") - 1
    admitted = (
        (radial_bins >= 0)
        & (radial_bins < radial_edges_m.size - 1)
        & (pair_distances_m > 0.0)
    )
    safe_pair_distances_m = np.where(admitted, pair_distances_m, 1.0)
    pair_unit_vectors = pair_displacements_m / safe_pair_distances_m[:, :, None]
    adjacency = np.zeros((batch_size, molecule_count, molecule_count))
    adjacency[:, first_indices, second_indices] = np.where(
        admitted,
        0.5 * (1.0 + np.cos(np.pi * pair_distances_m / radial_edges_m[-1])),
        0.0,
    )
    cluster_degree = np.sum(adjacency, axis=2)
    cluster_depth_two = np.einsum("bij,bj->bi", adjacency, cluster_degree)
    molecular_charges_C = np.asarray(
        tuple(
            np.sum(system.charges_C[molecule_atom_indices])
            for molecule_atom_indices in system.molecule_atom_indices
        )
    )
    charge_scale_C = max(
        float(np.max(np.abs(molecular_charges_C))), np.finfo(float).tiny
    )
    positive_charge = np.maximum(molecular_charges_C, 0.0) / charge_scale_C
    negative_charge = np.maximum(-molecular_charges_C, 0.0) / charge_scale_C
    radius_scale_m2 = np.maximum(
        np.max(molecular_radius_gyration_m2, axis=1), np.finfo(float).tiny
    )
    orientation_alignment = np.maximum(
        np.einsum(
            "bpa,bpac,bpc->bp",
            pair_unit_vectors,
            molecular_orientation_dyads[:, first_indices],
            pair_unit_vectors,
        ),
        0.0,
    )
    common_neighbor = np.sum(
        adjacency[:, first_indices, :] * adjacency[:, second_indices, :], axis=2
    )
    self_descriptor_values = np.zeros(
        (
            batch_size,
            molecule_count,
            len(molecular_memory.self_descriptor_schema),
        )
    )
    self_descriptor_index = {
        name: index
        for index, name in enumerate(molecular_memory.self_descriptor_schema)
    }
    self_descriptor_values[:, :, self_descriptor_index["uniform"]] = 1.0
    self_descriptor_values[
        :, :, self_descriptor_index["molecular_radius_gyration_A2"]
    ] = molecular_radius_gyration_m2 / radius_scale_m2[:, None]
    self_descriptor_values[:, :, self_descriptor_index["smooth_cluster_degree"]] = (
        cluster_degree
    )
    self_descriptor_values[:, :, self_descriptor_index["smooth_cluster_depth_2"]] = (
        cluster_depth_two
    )
    pair_descriptor_values = np.zeros(
        (
            batch_size,
            first_indices.size,
            len(molecular_memory.pair_descriptor_schema),
        )
    )
    pair_descriptor_index = {
        name: index
        for index, name in enumerate(molecular_memory.pair_descriptor_schema)
    }
    source_membership = np.eye(molecule_count)[first_indices]
    radial_bin_count = radial_edges_m.size - 1
    radial_membership = (
        radial_bins[:, :, None] == np.arange(radial_bin_count)[None, None, :]
    ) & admitted[:, :, None]
    number_density = np.einsum(
        "pm,bpr->bmr", source_membership, radial_membership.astype(float)
    )
    positive_density = np.einsum(
        "pm,bpr,p->bmr",
        source_membership,
        radial_membership.astype(float),
        positive_charge[second_indices],
    )
    negative_density = np.einsum(
        "pm,bpr,p->bmr",
        source_membership,
        radial_membership.astype(float),
        negative_charge[second_indices],
    )
    orientation_density = np.einsum(
        "pm,bpr,bp->bmr",
        source_membership,
        radial_membership.astype(float),
        orientation_alignment,
    )
    charge_products_C2 = (
        molecular_charges_C[first_indices] * molecular_charges_C[second_indices]
    )
    for radial_bin_index in range(radial_bin_count):
        radial_mask = radial_membership[:, :, radial_bin_index].astype(float)
        for descriptor_name, values in (
            (f"number_density_bin_{radial_bin_index}", number_density),
            (f"positive_charge_density_bin_{radial_bin_index}", positive_density),
            (f"negative_charge_density_bin_{radial_bin_index}", negative_density),
            (
                f"orientation_axis_alignment_bin_{radial_bin_index}",
                orientation_density,
            ),
        ):
            self_descriptor_values[:, :, self_descriptor_index[descriptor_name]] = (
                values[:, :, radial_bin_index]
            )
        for descriptor_name, values in (
            (f"pair_radial_bin_{radial_bin_index}", radial_mask),
            (
                f"pair_orientation_alignment_bin_{radial_bin_index}",
                radial_mask * orientation_alignment,
            ),
            (
                f"pair_common_neighbor_bin_{radial_bin_index}",
                radial_mask * common_neighbor,
            ),
            (
                f"pair_unlike_charge_bin_{radial_bin_index}",
                radial_mask * (charge_products_C2 < 0.0)[None, :],
            ),
            (
                f"pair_like_charge_bin_{radial_bin_index}",
                radial_mask * (charge_products_C2 > 0.0)[None, :],
            ),
        ):
            pair_descriptor_values[:, :, pair_descriptor_index[descriptor_name]] = (
                values
            )
    self_normalization = np.sum(self_descriptor_values, axis=2)
    pair_normalization = np.sum(pair_descriptor_values, axis=2)
    if np.any(self_normalization <= 0.0) or np.any(
        admitted & (pair_normalization <= 0.0)
    ):
        raise ValueError("configuration batch has unresolved memory descriptors")
    self_friction_scales = (
        self_descriptor_values
        @ np.asarray(molecular_memory.self_descriptor_friction_scales)
        / self_normalization
    )
    pair_friction_scales = np.divide(
        pair_descriptor_values
        @ np.asarray(molecular_memory.pair_descriptor_friction_scales),
        pair_normalization,
        out=np.zeros_like(pair_normalization),
        where=pair_normalization > 0.0,
    )
    directed_pair_scales = np.zeros((batch_size, molecule_count, molecule_count))
    directed_pair_scales[:, first_indices, second_indices] = pair_friction_scales
    pair_frictions_kg_s = (
        molecular_memory.molecular_pair_frictions_kg_s[None, :, :]
        * 0.5
        * (directed_pair_scales + np.swapaxes(directed_pair_scales, 1, 2))
    )
    molecular_friction_matrices_kg_s = -pair_frictions_kg_s
    diagonal_indices = np.arange(molecule_count)
    molecular_friction_matrices_kg_s[:, diagonal_indices, diagonal_indices] = (
        molecular_memory.molecular_self_frictions_kg_s[None, :] * self_friction_scales
        + np.sum(pair_frictions_kg_s, axis=2)
    )
    molecular_row_means_kg_s = np.mean(
        molecular_friction_matrices_kg_s,
        axis=2,
        keepdims=True,
    )
    molecular_column_means_kg_s = np.mean(
        molecular_friction_matrices_kg_s,
        axis=1,
        keepdims=True,
    )
    molecular_grand_means_kg_s = np.mean(
        molecular_friction_matrices_kg_s,
        axis=(1, 2),
        keepdims=True,
    )
    conditioned_molecular_frictions_kg_s = (
        molecular_friction_matrices_kg_s
        - molecular_row_means_kg_s
        - molecular_column_means_kg_s
        + molecular_grand_means_kg_s
    )
    conditioned_molecular_frictions_kg_s = 0.5 * (
        conditioned_molecular_frictions_kg_s
        + np.swapaxes(conditioned_molecular_frictions_kg_s, 1, 2)
    )
    friction_eigenvalues_kg_s, friction_eigenvectors = np.linalg.eigh(
        conditioned_molecular_frictions_kg_s
    )
    friction_scales_kg_s = np.maximum(
        np.max(friction_eigenvalues_kg_s, axis=1), np.finfo(float).tiny
    )
    retained_friction_modes = friction_eigenvalues_kg_s > (
        math.sqrt(np.finfo(float).eps) * friction_scales_kg_s[:, None]
    )
    inverse_friction_eigenvalues_s_kg = np.divide(
        1.0,
        friction_eigenvalues_kg_s,
        out=np.zeros_like(friction_eigenvalues_kg_s),
        where=retained_friction_modes,
    )
    conditioned_molecular_diffusions_m2_s = (
        K_B
        * molecular_memory.temperature_K
        * np.einsum(
            "bik,bk,bjk->bij",
            friction_eigenvectors,
            inverse_friction_eigenvalues_s_kg,
            friction_eigenvectors,
        )
    )
    conditioned_diffusions_m2_s = np.einsum(
        "bij,ac->biajc",
        conditioned_molecular_diffusions_m2_s,
        np.eye(CARTESIAN_DIMENSION),
    ).reshape(
        batch_size,
        CARTESIAN_DIMENSION * molecule_count,
        CARTESIAN_DIMENSION * molecule_count,
    )
    if not np.all(np.isfinite(conditioned_diffusions_m2_s)):
        raise ValueError("configuration-conditioned diffusion batch is non-finite")
    return conditioned_diffusions_m2_s


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

    def _energy_batch_tensor(
        self,
        positions_batch_m: torch.Tensor,
        box_vectors_batch_m: torch.Tensor,
        lambda_values: torch.Tensor,
    ) -> torch.Tensor:
        (
            fixed_energies,
            ion_ion_energies,
            ion_neutral_energies,
            _polarization_residuals,
        ) = self._energy_components_batch_tensor(positions_batch_m, box_vectors_batch_m)
        return (
            fixed_energies
            + lambda_values * ion_ion_energies
            + torch.sqrt(lambda_values) * ion_neutral_energies
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return polarization_energy_matrix, residual, physical_polarization_terms

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
        energy_matrix, residual, physical_polarization_terms = (
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


def _scale_molecular_centers_to_box(
    positions_m: Array,
    source_box_vectors_m: Array,
    target_box_vectors_m: Array,
    system: MolecularSystem,
) -> Array:
    length_scale = (
        abs(np.linalg.det(target_box_vectors_m))
        / abs(np.linalg.det(source_box_vectors_m))
    ) ** (1.0 / CARTESIAN_DIMENSION)
    scaled_positions_m = positions_m.copy()
    for molecule_atom_indices in system.molecule_atom_indices:
        anchor_position_m = positions_m[molecule_atom_indices[0]]
        unwrapped_positions_m = anchor_position_m + minimum_image_displacement(
            positions_m[molecule_atom_indices] - anchor_position_m,
            source_box_vectors_m,
        )
        molecule_masses_kg = system.masses_kg[molecule_atom_indices]
        center_of_mass_m = np.average(
            unwrapped_positions_m,
            axis=0,
            weights=molecule_masses_kg,
        )
        scaled_positions_m[molecule_atom_indices] = (
            unwrapped_positions_m - center_of_mass_m + length_scale * center_of_mass_m
        )
    return scaled_positions_m % np.diag(target_box_vectors_m)


def molecular_center_internal_pressure(
    model: AnalyticalPeriodicInteratomicModel,
    positions_by_ladder_m: Array,
    box_vectors_by_ladder_m: Array,
    temperature_K: float,
) -> InternalPressureResult:
    if temperature_K <= 0.0:
        raise ValueError("pressure temperature must be positive")
    volumes_m3 = np.abs(np.linalg.det(box_vectors_by_ladder_m))
    if not np.allclose(volumes_m3, volumes_m3[0], rtol=1.0e-12, atol=0.0):
        raise ValueError("internal-pressure chains must share one candidate volume")
    scaled_positions: list[Array] = []
    scaled_boxes: list[Array] = []
    derivative_steps = (
        model.numerics.pressure_log_volume_derivative_step,
        model.numerics.pressure_log_volume_derivative_check_step,
    )
    for logarithmic_volume_step in derivative_steps:
        for direction in (1.0, -1.0):
            length_scale = math.exp(
                direction * logarithmic_volume_step / CARTESIAN_DIMENSION
            )
            for positions_m, box_vectors_m in zip(
                positions_by_ladder_m, box_vectors_by_ladder_m, strict=True
            ):
                target_box_vectors_m = box_vectors_m * length_scale
                scaled_positions.append(
                    _scale_molecular_centers_to_box(
                        positions_m=positions_m,
                        source_box_vectors_m=box_vectors_m,
                        target_box_vectors_m=target_box_vectors_m,
                        system=model.system,
                    )
                )
                scaled_boxes.append(target_box_vectors_m)
    energy_terms = model.physical_energy_terms_batch(
        positions_batch_m=np.asarray(scaled_positions),
        box_vectors_batch_m=np.asarray(scaled_boxes),
    )
    energy_term_matrix_J = np.column_stack(
        (
            energy_terms.bonded_energy_J,
            energy_terms.lennard_jones_repulsion_energy_J,
            energy_terms.lennard_jones_attraction_energy_J,
            energy_terms.real_electrostatic_energy_J,
            energy_terms.reciprocal_electrostatic_energy_J,
            energy_terms.ewald_self_energy_J,
            energy_terms.electrostatic_exclusion_energy_J,
            energy_terms.polarization_energy_J,
            energy_terms.polarization_self_energy_J,
            energy_terms.polarization_exclusion_energy_J,
        )
    )
    ladder_count = positions_by_ladder_m.shape[0]
    derivative_rows: list[Array] = []
    batch_offset = 0
    for logarithmic_volume_step in derivative_steps:
        expanded_term_energies_J = energy_term_matrix_J[
            batch_offset : batch_offset + ladder_count
        ]
        contracted_term_energies_J = energy_term_matrix_J[
            batch_offset + ladder_count : batch_offset + 2 * ladder_count
        ]
        derivative_rows.append(
            np.mean(
                expanded_term_energies_J - contracted_term_energies_J,
                axis=0,
            )
            / (2.0 * logarithmic_volume_step)
        )
        batch_offset += 2 * ladder_count
    primary_term_derivatives_J, check_term_derivatives_J = derivative_rows
    mean_derivative_J = float(np.sum(primary_term_derivatives_J))
    check_derivative_J = float(np.sum(check_term_derivatives_J))
    derivative_mismatch = abs(mean_derivative_J - check_derivative_J) / max(
        abs(mean_derivative_J),
        abs(check_derivative_J),
        K_B * temperature_K,
    )
    if derivative_mismatch > model.numerics.pressure_derivative_relative_tolerance:
        raise ValueError(
            "internal-pressure finite differences disagree: "
            f"relative_mismatch={derivative_mismatch:.12g}, "
            "tolerance="
            f"{model.numerics.pressure_derivative_relative_tolerance:.12g}"
        )
    mean_volume_m3 = float(np.mean(volumes_m3))
    configurational_pressures_Pa = -primary_term_derivatives_J / mean_volume_m3
    ideal_pressure_Pa = (
        len(model.system.molecule_atom_indices) * K_B * temperature_K / mean_volume_m3
    )
    configurational_pressure_Pa = float(np.sum(configurational_pressures_Pa))
    internal_pressure_Pa = ideal_pressure_Pa + configurational_pressure_Pa
    if not math.isfinite(internal_pressure_Pa):
        raise FloatingPointError(
            "internal-pressure evaluation produced a nonfinite value"
        )
    (
        bonded_pressure_Pa,
        lennard_jones_repulsion_pressure_Pa,
        lennard_jones_attraction_pressure_Pa,
        real_electrostatic_pressure_Pa,
        reciprocal_electrostatic_pressure_Pa,
        ewald_self_pressure_Pa,
        electrostatic_exclusion_pressure_Pa,
        polarization_pressure_Pa,
        polarization_self_pressure_Pa,
        polarization_exclusion_pressure_Pa,
    ) = configurational_pressures_Pa
    return InternalPressureResult(
        internal_pressure_Pa=float(internal_pressure_Pa),
        ideal_pressure_Pa=float(ideal_pressure_Pa),
        configurational_pressure_Pa=configurational_pressure_Pa,
        mean_energy_derivative_J=mean_derivative_J,
        relative_derivative_mismatch=float(derivative_mismatch),
        bonded_pressure_Pa=float(bonded_pressure_Pa),
        lennard_jones_repulsion_pressure_Pa=float(lennard_jones_repulsion_pressure_Pa),
        lennard_jones_attraction_pressure_Pa=float(
            lennard_jones_attraction_pressure_Pa
        ),
        real_electrostatic_pressure_Pa=float(real_electrostatic_pressure_Pa),
        reciprocal_electrostatic_pressure_Pa=float(
            reciprocal_electrostatic_pressure_Pa
        ),
        ewald_self_pressure_Pa=float(ewald_self_pressure_Pa),
        electrostatic_exclusion_pressure_Pa=float(electrostatic_exclusion_pressure_Pa),
        polarization_pressure_Pa=float(polarization_pressure_Pa),
        polarization_self_pressure_Pa=float(polarization_self_pressure_Pa),
        polarization_exclusion_pressure_Pa=float(polarization_exclusion_pressure_Pa),
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
        positions_m: torch.Tensor,
        velocities_m_s: torch.Tensor,
        timesteps_s: Array,
        damping: Array,
        positive_power_steps: Array,
        best_maximum_forces_N: Array,
        stagnant_iterations: Array,
        converged: Array,
        force_evaluation_count: int,
        iteration: int,
        elapsed_s: float,
    ) -> None:
        checkpoint_payload = {
            "fingerprint": checkpoint_fingerprint,
            "metadata": checkpoint_metadata,
            "stage": stage,
            "systems": tuple(
                replace(system, positions_m=positions_m[index].detach().numpy())
                for index, system in enumerate(initial_systems)
            ),
            "velocities_m_s": velocities_m_s.detach().numpy(),
            "timesteps_s": timesteps_s,
            "damping": damping,
            "positive_power_steps": positive_power_steps,
            "best_maximum_forces_N": best_maximum_forces_N,
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
    velocities = torch.zeros_like(positions)
    timesteps_s = np.full(chain_count, dynamics.initial_relaxation_timestep_s)
    damping = np.full(chain_count, dynamics.initial_relaxation_initial_damping)
    positive_power_steps = np.zeros(chain_count, dtype=int)
    best_maximum_forces_N = np.full(chain_count, np.inf)
    stagnant_iterations = np.zeros(chain_count, dtype=int)
    converged = np.zeros(chain_count, dtype=bool)
    force_evaluation_count = 0
    invocation_force_evaluation_count = 0
    starting_iteration = 0
    previous_energies_J = np.zeros(chain_count)
    has_previous_energies = False
    previous_elapsed_s = 0.0
    if checkpoint_path.is_file():
        with checkpoint_path.open("rb") as checkpoint_file:
            checkpoint_payload = pickle.load(checkpoint_file)
        if checkpoint_payload["fingerprint"] != checkpoint_fingerprint:
            raise ValueError(
                "initialization checkpoint does not match the requested recipe, "
                "topology, or numerical settings"
            )
        checkpoint_systems = tuple(checkpoint_payload["systems"])
        positions = torch.as_tensor(
            np.stack(tuple(system.positions_m for system in checkpoint_systems)),
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
        force_call_durations_s: list[float] = []
        for chunk_start in range(
            0,
            active_chain_indices.size,
            dynamics.force_batch_size,
        ):
            chunk_indices = active_chain_indices[
                chunk_start : chunk_start + dynamics.force_batch_size
            ]
            chunk_tensor = torch.as_tensor(chunk_indices, dtype=torch.long)
            force_call_start_time = time.perf_counter()
            chunk_energies_J, chunk_forces_N, _polarization_residuals = (
                model.physical_energy_force_batch_tensor(
                    positions[chunk_tensor],
                    box_vectors[chunk_tensor],
                )
            )
            force_call_durations_s.append(time.perf_counter() - force_call_start_time)
            energies_J[chunk_tensor] = chunk_energies_J
            forces_N[chunk_tensor] = chunk_forces_N
            force_evaluation_count += 1
            invocation_force_evaluation_count += 1
        invocation_elapsed_s = time.perf_counter() - relaxation_start_time
        elapsed_s = previous_elapsed_s + invocation_elapsed_s
        seconds_per_force_call = float(np.mean(force_call_durations_s))
        if iteration == starting_iteration:
            initial_forces_N = forces_N.detach().numpy()
            initial_maximum_forces_N = np.max(
                np.linalg.norm(initial_forces_N, axis=2), axis=1
            )
            remaining_force_calls = 0
            if np.any(initial_maximum_forces_N > dynamics.initial_force_tolerance_N):
                remaining_force_calls = (
                    dynamics.initial_relaxation_maximum_force_evaluations
                    - invocation_force_evaluation_count
                )
            projected_maximum_elapsed_s = (
                invocation_elapsed_s + remaining_force_calls * seconds_per_force_call
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
                f"seconds_per_force_call={seconds_per_force_call:.6f} "
                f"projected_maximum_elapsed_s={projected_maximum_elapsed_s:.6f} "
                f"maximum_forces_N={tuple(initial_maximum_forces_N)} "
                f"minimum_contact_ratios={tuple(initial_contact_ratios)}",
                flush=True,
            )
            if (
                projected_maximum_elapsed_s
                > dynamics.initial_relaxation_maximum_elapsed_s
            ):
                write_relaxation_checkpoint(
                    "relaxing",
                    positions,
                    velocities,
                    timesteps_s,
                    damping,
                    positive_power_steps,
                    best_maximum_forces_N,
                    stagnant_iterations,
                    converged,
                    force_evaluation_count,
                    iteration,
                    elapsed_s,
                )
                raise RuntimeError(
                    "exact-shape relaxation preflight exceeds the configured runtime: "
                    f"projected_s={projected_maximum_elapsed_s:.12g}, "
                    f"limit_s={dynamics.initial_relaxation_maximum_elapsed_s:.12g}, "
                    f"checkpoint={checkpoint_path}"
                )
        force_norms_N = torch.linalg.norm(forces_N, dim=2)
        maximum_forces_N = torch.max(force_norms_N, dim=1).values.detach().numpy()
        last_maximum_forces_N = maximum_forces_N.copy()
        converged |= maximum_forces_N <= dynamics.initial_force_tolerance_N
        velocities[torch.as_tensor(converged)] = 0.0
        for chain_index in active_chain_indices:
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
        active_chain_indices = np.flatnonzero(~converged)
        for chain_index in active_chain_indices:
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
            positions_m = positions.detach().numpy()
            pair_displacements_m = minimum_image_displacement(
                positions_m[:, pair_i] - positions_m[:, pair_j],
                model.system.box_vectors_m,
            )
            minimum_contact_ratios = np.min(
                np.linalg.norm(pair_displacements_m, axis=2)
                / contact_distances_m[None, :],
                axis=1,
            )
            energy_changes_J = np.zeros(chain_count)
            if has_previous_energies:
                energy_changes_J = energies_J.detach().numpy() - previous_energies_J
            print(
                "[relaxation] "
                f"elapsed_s={elapsed_s:.6f} iteration={iteration} "
                f"force_evaluations={force_evaluation_count} "
                f"invocation_force_evaluations={invocation_force_evaluation_count} "
                f"seconds_per_force_call={seconds_per_force_call:.6f} "
                f"active_chains={tuple(int(value) for value in np.flatnonzero(~converged))} "
                f"maximum_forces_N={tuple(maximum_forces_N)} "
                f"minimum_contact_ratios={tuple(minimum_contact_ratios)} "
                f"energy_changes_J={tuple(energy_changes_J)} "
                f"timesteps_s={tuple(timesteps_s)}",
                flush=True,
            )
            write_relaxation_checkpoint(
                "relaxing",
                positions,
                velocities,
                timesteps_s,
                damping,
                positive_power_steps,
                best_maximum_forces_N,
                stagnant_iterations,
                converged,
                force_evaluation_count,
                iteration,
                elapsed_s,
            )
        previous_energies_J = energies_J.detach().numpy()
        has_previous_energies = True
        if invocation_elapsed_s >= dynamics.initial_relaxation_maximum_elapsed_s:
            break
    final_elapsed_s = previous_elapsed_s + time.perf_counter() - relaxation_start_time
    final_stage = "relaxed" if np.all(converged) else "relaxing"
    write_relaxation_checkpoint(
        final_stage,
        positions,
        velocities,
        timesteps_s,
        damping,
        positive_power_steps,
        best_maximum_forces_N,
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


def species_coordination_observable_series(
    configurations_m: Array,
    system: MolecularSystem,
) -> dict[str, Array]:
    library = _physical_library_records()
    coordination_switches = library.basis_record["coordination_switches"]
    lithium_atomic_number = library.species_records["Li+"]["sites"][0]["atomic_number"]
    lithium_atom_indices: list[int] = []
    acceptor_indices_by_species: dict[str, list[int]] = {}
    switch_record_by_species: dict[str, dict] = {}
    for species_name, molecule_atom_indices in zip(
        system.molecule_species_names, system.molecule_atom_indices, strict=True
    ):
        species_record = library.species_records[species_name]
        formal_charge_e = float(species_record["formal_charge_e"])
        coordinating_indices = acceptor_indices_by_species.setdefault(species_name, [])
        for atom_index, site_record in zip(
            molecule_atom_indices, species_record["sites"], strict=True
        ):
            if int(site_record["atomic_number"]) == lithium_atomic_number:
                lithium_atom_indices.append(int(atom_index))
            if int(site_record["hba_count_contribution"]) > 0:
                coordinating_indices.append(int(atom_index))
        if formal_charge_e < 0.0:
            switch_record_by_species[species_name] = coordination_switches["Li_anion"]
        if formal_charge_e == 0.0:
            switch_record_by_species[species_name] = coordination_switches["Li_ligand"]
        if formal_charge_e > 0.0:
            switch_record_by_species[species_name] = coordination_switches["Li_solvent"]
    if not lithium_atom_indices:
        raise ValueError("coordination diagnostics require at least one lithium site")
    observable_series: dict[str, Array] = {}
    lithium_positions = configurations_m[:, np.asarray(lithium_atom_indices)]
    for species_name, coordinating_indices in acceptor_indices_by_species.items():
        if not coordinating_indices:
            continue
        switch_record = switch_record_by_species[species_name]
        coordinating_positions = configurations_m[:, np.asarray(coordinating_indices)]
        displacements_m = minimum_image_displacement(
            lithium_positions[:, :, None, :] - coordinating_positions[:, None, :, :],
            system.box_vectors_m,
        )
        distances_m = np.linalg.norm(displacements_m, axis=3)
        switch_values = 1.0 / (
            1.0
            + (distances_m / float(switch_record["r0_m"]))
            ** int(switch_record["exponent"])
        )
        observable_series[species_name] = np.sum(switch_values, axis=(1, 2))
    return observable_series


def symmetric_psd_pseudoinverse(matrix: Array, relative_tolerance: float) -> Array:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = eigh(symmetric)
    tolerance = relative_tolerance * float(eigenvalues[-1])
    if eigenvalues[0] < -tolerance:
        raise ValueError("Dirichlet matrix is not positive semidefinite")
    active = eigenvalues > tolerance
    return (eigenvectors[:, active] / eigenvalues[active]) @ eigenvectors[:, active].T


class AnalyticalMolecularGalerkinAssembler:
    """Exact molecular-translation basis gradients and batched operators."""

    def __init__(
        self,
        system: MolecularSystem,
        numerics: NumericalSettings,
        basis_level: int,
    ) -> None:
        if basis_level < 1:
            raise ValueError("basis_level must be at least one")
        self.system = system
        self.numerics = numerics
        self.basis_level = basis_level
        atom_count = system.positions_m.shape[0]
        molecule_count = len(system.molecule_atom_indices)
        atom_pair_i, atom_pair_j = np.triu_indices(atom_count, 1)
        molecule_pair_i, molecule_pair_j = np.triu_indices(molecule_count, 1)
        self._atom_pair_i = torch.as_tensor(atom_pair_i)
        self._atom_pair_j = torch.as_tensor(atom_pair_j)
        self._molecule_pair_i = torch.as_tensor(molecule_pair_i)
        self._molecule_pair_j = torch.as_tensor(molecule_pair_j)
        self._charges = torch.as_tensor(system.charges_C)
        self._pair_charges = (
            self._charges[self._atom_pair_i] * self._charges[self._atom_pair_j]
        )
        self._box = torch.as_tensor(system.box_vectors_m)
        atom_pair_incidence = np.zeros((atom_pair_i.size, molecule_count))
        atom_pair_incidence[
            np.arange(atom_pair_i.size), system.molecule_index[atom_pair_i]
        ] += 1.0
        atom_pair_incidence[
            np.arange(atom_pair_j.size), system.molecule_index[atom_pair_j]
        ] -= 1.0
        self._atom_pair_incidence = torch.as_tensor(atom_pair_incidence)
        molecule_pair_incidence = np.zeros((molecule_pair_i.size, molecule_count))
        molecule_pair_incidence[np.arange(molecule_pair_i.size), molecule_pair_i] = 1.0
        molecule_pair_incidence[np.arange(molecule_pair_j.size), molecule_pair_j] = -1.0
        self._molecule_pair_incidence = torch.as_tensor(molecule_pair_incidence)
        center_weights = np.zeros((molecule_count, atom_count))
        first_atoms: list[int] = []
        last_atoms: list[int] = []
        molecule_charges: list[float] = []
        for molecule_index, atom_indices in enumerate(system.molecule_atom_indices):
            masses_kg = system.masses_kg[atom_indices]
            center_weights[molecule_index, atom_indices] = masses_kg / np.sum(masses_kg)
            first_atoms.append(int(atom_indices[0]))
            last_atoms.append(int(atom_indices[-1]))
            molecule_charges.append(float(np.sum(system.charges_C[atom_indices])))
        self._center_weights = torch.as_tensor(center_weights)
        self._first_atoms = torch.as_tensor(first_atoms)
        self._last_atoms = torch.as_tensor(last_atoms)
        self._multi_atom_mask = torch.as_tensor(
            [indices.size > 1 for indices in system.molecule_atom_indices]
        )
        self._molecule_charges = torch.as_tensor(molecule_charges, dtype=TORCH_DTYPE)
        self._molecule_atom_index_tuples = tuple(
            tuple(int(atom_index) for atom_index in atom_indices)
            for atom_indices in system.molecule_atom_indices
        )
        self._molecule_mass_tensors = tuple(
            torch.as_tensor(system.masses_kg[atom_indices])
            for atom_indices in system.molecule_atom_indices
        )
        self._molecule_index_tensors = tuple(
            torch.as_tensor(atom_indices)
            for atom_indices in system.molecule_atom_indices
        )
        self._atom_to_molecule = torch.nn.functional.one_hot(
            torch.as_tensor(system.molecule_index), molecule_count
        ).to(TORCH_DTYPE)
        self._primitive_levels = self._make_primitive_levels()
        self._basis_exponents, self._basis_levels = self._make_basis_schema()
        derivative_exponents = self._basis_exponents[:, None, :].repeat(
            1, self._basis_exponents.shape[1], 1
        )
        derivative_exponents = derivative_exponents - torch.diag_embed(
            (self._basis_exponents > 0).to(torch.int64)
        )
        self._basis_derivative_exponents = derivative_exponents
        self.basis_count = int(self._basis_exponents.shape[0])
        self._compiled_basis = torch.compile(
            self._basis_tensor, backend="inductor", fullgraph=True, dynamic=False
        )
        self._compiled_assemble = torch.compile(
            self._assemble_tensor, backend="inductor", fullgraph=True, dynamic=False
        )

    def _make_primitive_levels(self) -> tuple[int, ...]:
        levels: list[int] = [1] * CARTESIAN_DIMENSION
        for level_limit, multiplicity in (
            (self.numerics.basis_radial_count, 2),
            (self.numerics.basis_angular_order, 2),
            (self.numerics.basis_cluster_depth, 2),
            (self.numerics.basis_fourier_shell, 4 * CARTESIAN_DIMENSION),
        ):
            for level in range(1, min(level_limit, self.basis_level) + 1):
                levels.extend([level] * multiplicity)
        return tuple(levels)

    def _make_basis_schema(self) -> tuple[torch.Tensor, torch.Tensor]:
        primitive_count = len(self._primitive_levels)
        candidate_limit = 2 * self.numerics.maximum_basis_size + self.basis_level
        correlation_limit = min(self.numerics.basis_correlation_order, self.basis_level)
        exponents: list[Array] = []
        levels: list[int] = []
        for level in range(1, self.basis_level + 1):
            for primitive_index, primitive_level in enumerate(self._primitive_levels):
                if primitive_level == level and len(exponents) < candidate_limit:
                    exponent = np.zeros(primitive_count, dtype=np.int64)
                    exponent[primitive_index] = 1
                    exponents.append(exponent)
                    levels.append(level)
            if len(exponents) >= candidate_limit:
                break
            if 2 <= level <= correlation_limit:
                available = tuple(
                    index
                    for index, primitive_level in enumerate(self._primitive_levels)
                    if primitive_level <= level
                )
                for combination in combinations_with_replacement(available, level):
                    exponent = np.zeros(primitive_count, dtype=np.int64)
                    np.add.at(exponent, list(combination), 1)
                    exponents.append(exponent)
                    levels.append(level)
                    if len(exponents) >= candidate_limit:
                        break
        return torch.as_tensor(np.stack(exponents)), torch.as_tensor(levels)

    def _invariant_internal_polarization(
        self, positions_m: torch.Tensor
    ) -> torch.Tensor:
        contributions: list[torch.Tensor] = []
        for indices, index_tensor, masses_kg in zip(
            self._molecule_atom_index_tuples,
            self._molecule_index_tensors,
            self._molecule_mass_tensors,
            strict=True,
        ):
            anchor = positions_m[:, indices[0], :]
            local_positions = anchor[:, None, :] + _torch_minimum_image(
                positions_m[:, index_tensor, :] - anchor[:, None, :], self._box
            )
            center_m = torch.sum(
                masses_kg[None, :, None] * local_positions, dim=1
            ) / torch.sum(masses_kg)
            contributions.append(
                torch.sum(
                    self._charges[index_tensor][None, :, None]
                    * (local_positions - center_m[:, None, :]),
                    dim=1,
                )
            )
        return torch.stack(contributions).sum(dim=0)

    def _primitive_tensor(
        self, positions_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_count = positions_m.shape[0]
        molecule_count = len(self.system.molecule_atom_indices)
        coordinate_count = CARTESIAN_DIMENSION * molecule_count
        values: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        zero_gradient = torch.zeros((sample_count, coordinate_count), dtype=TORCH_DTYPE)
        internal_polarization = self._invariant_internal_polarization(positions_m)
        for axis in range(CARTESIAN_DIMENSION):
            values.append(internal_polarization[:, axis])
            gradients.append(zero_gradient)

        atom_displacements = _torch_minimum_image(
            positions_m[:, self._atom_pair_i] - positions_m[:, self._atom_pair_j],
            self._box,
        )
        atom_distances = torch.linalg.norm(atom_displacements, dim=2)
        atom_directions = atom_displacements / atom_distances[:, :, None]
        normalized_distance = atom_distances / self.numerics.basis_radial_cutoff_m
        inside_cutoff = atom_distances < self.numerics.basis_radial_cutoff_m
        cutoff = 0.5 * (torch.cos(math.pi * normalized_distance) + 1.0) * inside_cutoff
        cutoff_derivative = (
            -0.5
            * math.pi
            / self.numerics.basis_radial_cutoff_m
            * torch.sin(math.pi * normalized_distance)
            * inside_cutoff
        )
        for radial_mode in range(
            1, min(self.numerics.basis_radial_count, self.basis_level) + 1
        ):
            phase = radial_mode * math.pi * normalized_distance
            radial = torch.cos(phase) * cutoff
            radial_derivative = (
                -radial_mode
                * math.pi
                / self.numerics.basis_radial_cutoff_m
                * torch.sin(phase)
                * cutoff
                + torch.cos(phase) * cutoff_derivative
            )
            derivative_vectors = radial_derivative[:, :, None] * atom_directions
            radial_gradient = torch.einsum(
                "spa,pm->sma", derivative_vectors, self._atom_pair_incidence
            ).reshape(sample_count, coordinate_count)
            charge_derivative_vectors = (
                derivative_vectors * self._pair_charges[None, :, None]
            )
            values.extend((radial.sum(dim=1), (radial * self._pair_charges).sum(dim=1)))
            gradients.extend(
                (
                    radial_gradient,
                    torch.einsum(
                        "spa,pm->sma",
                        charge_derivative_vectors,
                        self._atom_pair_incidence,
                    ).reshape(sample_count, coordinate_count),
                )
            )

        centers_m = torch.einsum("ma,sad->smd", self._center_weights, positions_m)
        raw_axes = _torch_minimum_image(
            positions_m[:, self._last_atoms] - positions_m[:, self._first_atoms],
            self._box,
        )
        axes = raw_axes / torch.sqrt(
            torch.sum(raw_axes**2, dim=2, keepdim=True) + torch.finfo(TORCH_DTYPE).tiny
        )
        axes = axes * self._multi_atom_mask[None, :, None]
        molecule_displacements = _torch_minimum_image(
            centers_m[:, self._molecule_pair_i] - centers_m[:, self._molecule_pair_j],
            self._box,
        )
        molecule_distances = torch.linalg.norm(molecule_displacements, dim=2)
        molecule_directions = molecule_displacements / molecule_distances[:, :, None]
        first_axes = axes[:, self._molecule_pair_i]
        second_axes = axes[:, self._molecule_pair_j]
        projection = torch.sum(first_axes * molecule_directions, dim=2)
        alignment = torch.sum(first_axes * second_axes, dim=2)
        projection_derivative = (
            first_axes - projection[:, :, None] * molecule_directions
        ) / molecule_distances[:, :, None]
        for angular_order in range(
            1, min(self.numerics.basis_angular_order, self.basis_level) + 1
        ):
            angular_derivative = (
                angular_order
                * projection[:, :, None] ** (angular_order - 1)
                * projection_derivative
            )
            values.extend(
                (
                    torch.sum(projection**angular_order, dim=1),
                    torch.sum(alignment**angular_order, dim=1),
                )
            )
            gradients.extend(
                (
                    torch.einsum(
                        "spa,pm->sma",
                        angular_derivative,
                        self._molecule_pair_incidence,
                    ).reshape(sample_count, coordinate_count),
                    zero_gradient,
                )
            )

        full_displacements = _torch_minimum_image(
            centers_m[:, :, None] - centers_m[:, None, :], self._box
        )
        full_distances = torch.linalg.norm(
            full_displacements
            + torch.eye(molecule_count, dtype=TORCH_DTYPE)[None, :, :, None],
            dim=3,
        )
        identity = torch.eye(molecule_count, dtype=TORCH_DTYPE)
        adjacency = torch.exp(
            -((full_distances / self.numerics.basis_radial_cutoff_m) ** 2)
        ) * (1.0 - identity[None])
        pair_adjacency = adjacency[:, self._molecule_pair_i, self._molecule_pair_j]
        pair_adjacency_derivative = (
            -2.0
            * pair_adjacency[:, :, None]
            * molecule_displacements
            / self.numerics.basis_radial_cutoff_m**2
        )
        cluster_power = adjacency
        for cluster_depth in range(
            1, min(self.numerics.basis_cluster_depth, self.basis_level) + 1
        ):
            trace_sensitivity = cluster_depth * torch.transpose(
                torch.linalg.matrix_power(adjacency, cluster_depth - 1), 1, 2
            )
            charge_sensitivity = torch.zeros_like(adjacency)
            for left_depth in range(cluster_depth):
                left_vector = (
                    torch.linalg.matrix_power(adjacency, left_depth)
                    @ self._molecule_charges
                )
                right_vector = (
                    torch.linalg.matrix_power(adjacency, cluster_depth - 1 - left_depth)
                    @ self._molecule_charges
                )
                charge_sensitivity = charge_sensitivity + (
                    left_vector[:, :, None] * right_vector[:, None, :]
                )
            trace_pair_sensitivity = (
                trace_sensitivity[:, self._molecule_pair_i, self._molecule_pair_j]
                + trace_sensitivity[:, self._molecule_pair_j, self._molecule_pair_i]
            )
            charge_pair_sensitivity = (
                charge_sensitivity[:, self._molecule_pair_i, self._molecule_pair_j]
                + charge_sensitivity[:, self._molecule_pair_j, self._molecule_pair_i]
            )
            values.extend(
                (
                    torch.diagonal(cluster_power, dim1=1, dim2=2).sum(dim=1),
                    torch.einsum(
                        "m,smn,n->s",
                        self._molecule_charges,
                        cluster_power,
                        self._molecule_charges,
                    ),
                )
            )
            gradients.extend(
                (
                    torch.einsum(
                        "sp,spa,pm->sma",
                        trace_pair_sensitivity,
                        pair_adjacency_derivative,
                        self._molecule_pair_incidence,
                    ).reshape(sample_count, coordinate_count),
                    torch.einsum(
                        "sp,spa,pm->sma",
                        charge_pair_sensitivity,
                        pair_adjacency_derivative,
                        self._molecule_pair_incidence,
                    ).reshape(sample_count, coordinate_count),
                )
            )
            cluster_power = cluster_power @ adjacency

        reciprocal_base = 2.0 * math.pi * torch.linalg.inv(self._box)
        for shell in range(
            1, min(self.numerics.basis_fourier_shell, self.basis_level) + 1
        ):
            for axis in range(CARTESIAN_DIMENSION):
                reciprocal = shell * reciprocal_base[axis]
                phases = positions_m @ reciprocal
                cosine = torch.cos(phases)
                sine = torch.sin(phases)
                for weights, feature, derivative in (
                    (torch.ones_like(self._charges), cosine, -sine),
                    (torch.ones_like(self._charges), sine, cosine),
                    (self._charges, cosine, -sine),
                    (self._charges, sine, cosine),
                ):
                    atom_gradient = (
                        weights[None, :, None]
                        * derivative[:, :, None]
                        * reciprocal[None, None, :]
                    )
                    values.append(torch.sum(weights[None] * feature, dim=1))
                    gradients.append(
                        torch.einsum(
                            "sad,am->smd", atom_gradient, self._atom_to_molecule
                        ).reshape(sample_count, coordinate_count)
                    )
        return torch.stack(values, dim=1), torch.stack(gradients, dim=1)

    def _basis_tensor(
        self, positions_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primitive_values, primitive_gradients = self._primitive_tensor(positions_m)
        basis_values = torch.prod(
            primitive_values[:, None, :] ** self._basis_exponents[None, :, :],
            dim=2,
        )
        product_coefficients = self._basis_exponents[None, :, :] * torch.prod(
            primitive_values[:, None, None, :]
            ** self._basis_derivative_exponents[None, :, :, :],
            dim=3,
        )
        basis_gradients = torch.einsum(
            "sbp,spd->sbd", product_coefficients, primitive_gradients
        )
        return basis_values, basis_gradients

    def basis_values_and_gradients(
        self, configurations_m: Array
    ) -> tuple[Array, Array]:
        values, gradients = self._compiled_basis(
            torch.as_tensor(configurations_m, dtype=TORCH_DTYPE)
        )
        values = values - torch.mean(values, dim=0)
        return values.detach().numpy(), gradients.detach().numpy()

    def assemble_batch(
        self,
        configurations_m: Array,
        diffusion_square_roots: Array,
        polarization_gradients: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        assembled = self._compiled_assemble(
            torch.as_tensor(configurations_m, dtype=TORCH_DTYPE),
            torch.as_tensor(diffusion_square_roots, dtype=TORCH_DTYPE),
            torch.as_tensor(polarization_gradients, dtype=TORCH_DTYPE),
        )
        return tuple(value.detach().numpy() for value in assembled)

    def _assemble_tensor(
        self,
        configurations_m: torch.Tensor,
        diffusion_square_roots: torch.Tensor,
        polarization_gradients: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _, gradients = self._basis_tensor(configurations_m)
        factored_gradients = torch.einsum(
            "sbd,sdr->sbr", gradients, diffusion_square_roots
        )
        factored_polarization = torch.einsum(
            "ad,sdr->sar", polarization_gradients, diffusion_square_roots
        )
        sample_dirichlet = torch.einsum(
            "sbr,scr->sbc", factored_gradients, factored_gradients
        )
        sample_coupling = torch.einsum(
            "sbr,sar->sba", factored_gradients, factored_polarization
        )
        sample_direct = torch.einsum(
            "sar,sar->sa", factored_polarization, factored_polarization
        )
        diagnostics = torch.cat(
            (
                torch.sum(factored_gradients**2, dim=2),
                torch.einsum(
                    "sbr,sar->sba", factored_gradients, factored_polarization
                ).reshape(factored_gradients.shape[0], -1),
                sample_direct,
            ),
            dim=1,
        )
        complete_operator_samples = torch.cat(
            (
                sample_dirichlet.reshape(sample_dirichlet.shape[0], -1),
                sample_coupling.reshape(sample_coupling.shape[0], -1),
                sample_direct,
            ),
            dim=1,
        )
        return (
            sample_dirichlet.sum(dim=0),
            sample_coupling.sum(dim=0),
            sample_direct.sum(dim=0),
            diagnostics,
            complete_operator_samples,
        )


def maximum_chain_operator_relative_disagreement(
    chain_statistics: tuple[tuple[Array, Array, Array], ...],
    eigenvalue_relative_tolerance: float,
    maximum_basis_size: int,
) -> float:
    pooled_dirichlet = np.mean(
        np.stack(tuple(statistics[0] for statistics in chain_statistics)), axis=0
    )
    pooled_coupling = np.mean(
        np.stack(tuple(statistics[1] for statistics in chain_statistics)), axis=0
    )
    pooled_direct = np.mean(
        np.stack(tuple(statistics[2] for statistics in chain_statistics)), axis=0
    )
    pooled_diagonal = np.diag(pooled_dirichlet)
    diagonal_scale = max(float(np.max(pooled_diagonal)), np.finfo(float).tiny)
    active_basis = pooled_diagonal > (eigenvalue_relative_tolerance * diagonal_scale)
    if not np.any(active_basis):
        raise ValueError("pooled chain Dirichlet operator has no active basis modes")
    basis_scales = np.sqrt(pooled_diagonal[active_basis])
    normalized_pooled_coupling = pooled_coupling[active_basis] / basis_scales[:, None]
    candidate_scores = np.sum(normalized_pooled_coupling**2, axis=1)
    selected_count = min(maximum_basis_size, candidate_scores.size)
    selected_indices = np.argsort(candidate_scores)[-selected_count:]
    normalized_pooled_dirichlet = pooled_dirichlet[
        np.ix_(active_basis, active_basis)
    ] / (basis_scales[:, None] * basis_scales[None, :])
    selected_pooled_dirichlet = normalized_pooled_dirichlet[
        np.ix_(selected_indices, selected_indices)
    ]
    selected_pooled_coupling = normalized_pooled_coupling[selected_indices]
    pooled_inverse = symmetric_psd_pseudoinverse(
        selected_pooled_dirichlet,
        eigenvalue_relative_tolerance,
    )
    pooled_correction = (
        selected_pooled_coupling.T @ pooled_inverse @ selected_pooled_coupling
    )
    direct_scale = max(float(np.linalg.norm(pooled_direct)), np.finfo(float).tiny)
    maximum_disagreement = 0.0
    for chain_dirichlet, chain_coupling, chain_direct in chain_statistics:
        normalized_chain_coupling = chain_coupling[active_basis] / basis_scales[:, None]
        normalized_chain_dirichlet = chain_dirichlet[
            np.ix_(active_basis, active_basis)
        ] / (basis_scales[:, None] * basis_scales[None, :])
        selected_chain_dirichlet = normalized_chain_dirichlet[
            np.ix_(selected_indices, selected_indices)
        ]
        selected_chain_coupling = normalized_chain_coupling[selected_indices]
        chain_inverse = symmetric_psd_pseudoinverse(
            selected_chain_dirichlet,
            eigenvalue_relative_tolerance,
        )
        chain_correction = (
            selected_chain_coupling.T @ chain_inverse @ selected_chain_coupling
        )
        direct_disagreement = float(
            np.linalg.norm(chain_direct - pooled_direct) / direct_scale
        )
        correction_disagreement = float(
            np.linalg.norm(chain_correction - pooled_correction) / direct_scale
        )
        maximum_disagreement = max(
            maximum_disagreement,
            direct_disagreement,
            correction_disagreement,
        )
    return maximum_disagreement


def projected_conductivity_sequence(
    model: AnalyticalPeriodicInteratomicModel,
    state: IonicHrexState,
    settings: IonicHrexSettings,
    equilibrium_samples_per_batch: int,
    equilibrium_maximum_refinement_batches: int,
    system: MolecularSystem,
    temperature_K: float,
    molecular_memory: MolecularMemoryOperator,
    numerics: NumericalSettings,
    operator_checkpoint_directory: Path,
) -> tuple[
    float,
    float,
    tuple[float, ...],
    tuple[float, ...],
    int,
    float,
    float,
    float,
    int,
    bool,
    tuple[tuple[str, float], ...],
    bool,
    bool,
]:
    if settings.independent_ladder_count < 2:
        raise ValueError("projection requires at least two independent chains")
    if equilibrium_samples_per_batch <= 0:
        raise ValueError("operator samples per refinement batch must be positive")
    operator_checkpoint_directory.mkdir(parents=True, exist_ok=True)
    sampling_start_s = time.perf_counter()
    warmup_block_count = math.ceil(
        settings.warmup_cycle_count / settings.block_cycle_count
    )
    for warmup_block_index in range(warmup_block_count):
        block_cycle_count = min(
            settings.block_cycle_count,
            settings.warmup_cycle_count
            - warmup_block_index * settings.block_cycle_count,
        )
        block_start_s = time.perf_counter()
        state, warmup_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            cycle_count=block_cycle_count,
            attempt_exchange=len(settings.lambdas) > 1,
            retain_samples=False,
        )
        _report_hrex_block(
            stage="nvt-burn-in",
            block_index=warmup_block_index + 1,
            block_elapsed_s=time.perf_counter() - block_start_s,
            total_elapsed_s=time.perf_counter() - sampling_start_s,
            state=state,
            settings=settings,
            block=warmup_block,
        )
    _reset_hrex_round_trip_tracking(
        state=state,
        replica_count=len(settings.lambdas),
        reset_walker_identifiers=False,
    )
    state.exchange_attempts.fill(0)
    state.exchange_acceptances.fill(0)
    state.exchange_expected_acceptance_sums.fill(0.0)
    state.hmc_attempts.fill(0)
    state.hmc_acceptances.fill(0)
    state.hmc_expected_acceptance_sums.fill(0.0)
    state.hmc_absolute_energy_error_over_kbt_sums.fill(0.0)
    state.hmc_molecular_com_squared_displacement_sums_m2.fill(0.0)
    unique_chain_indices = np.arange(settings.independent_ladder_count)
    maximum_basis_level = max(
        numerics.basis_radial_count,
        numerics.basis_fourier_shell,
        numerics.basis_angular_order,
        numerics.basis_cluster_depth,
        numerics.basis_correlation_order,
    )
    galerkin_assembler = AnalyticalMolecularGalerkinAssembler(
        system=system,
        numerics=numerics,
        basis_level=maximum_basis_level,
    )
    polarization_gradients = np.zeros(
        (CARTESIAN_DIMENSION, CARTESIAN_DIMENSION * len(system.molecule_atom_indices))
    )
    for molecule_index, molecule_atom_indices in enumerate(
        system.molecule_atom_indices
    ):
        molecule_slice = slice(
            CARTESIAN_DIMENSION * molecule_index,
            CARTESIAN_DIMENSION * (molecule_index + 1),
        )
        molecular_charge_C = float(np.sum(system.charges_C[molecule_atom_indices]))
        polarization_gradients[:, molecule_slice] = molecular_charge_C * np.eye(
            CARTESIAN_DIMENSION
        )
    cumulative_chain_dirichlet: list[Array] = []
    cumulative_chain_coupling: list[Array] = []
    cumulative_chain_direct: list[Array] = []
    chain_operator_series: list[list[Array]] = [
        [] for _chain_index in unique_chain_indices
    ]
    chain_complete_operator_series: list[list[Array]] = [
        [] for _chain_index in unique_chain_indices
    ]
    chain_operator_blocks: list[list[tuple[Array, Array, Array, int]]] = [
        [] for _chain_index in unique_chain_indices
    ]
    admitted_chain_statistics: tuple[tuple[Array, Array, Array], ...] = ()
    admitted_samples_per_chain = 0
    operator_effective_sample_size = 0.0
    maximum_split_rhat = math.inf
    projection_throughput_reported = False
    diagnostic_pilot_sample_count = MINIMUM_OPERATOR_DIAGNOSTIC_PILOT_SAMPLES
    diagnostic_bases: list[OperatorDiagnosticBasis] = []
    online_block_conductivities_by_chain_S_m: list[list[float]] = [
        [] for _chain_index in unique_chain_indices
    ]
    online_conductivity_estimates_S_m: list[float] = []
    online_operator_disagreements: list[float] = []
    previous_provisional_conductivity_S_m = math.inf
    operator_diagnostics_certified = False
    molecular_charges_C = np.asarray(
        tuple(
            np.sum(system.charges_C[molecule_atom_indices])
            for molecule_atom_indices in system.molecule_atom_indices
        )
    )
    ionic_species = tuple(
        sorted(
            {
                species_name
                for species_name, molecular_charge_C in zip(
                    system.molecule_species_names,
                    molecular_charges_C,
                    strict=True,
                )
                if molecular_charge_C != 0.0
            }
        )
    )
    species_direct_sums_S_m = {
        f"{first_species}|{second_species}": 0.0
        for first_species, second_species in combinations_with_replacement(
            ionic_species, 2
        )
    }
    molecular_diffusion_sample_count = 0
    operator_conductivity_prefactor = 1.0 / (
        3.0 * K_B * temperature_K * abs(np.linalg.det(system.box_vectors_m))
    )
    for refinement_batch in range(1, equilibrium_maximum_refinement_batches + 1):
        samples_per_chain = equilibrium_samples_per_batch * refinement_batch
        production_cycle_count = (
            equilibrium_samples_per_batch * settings.measurement_stride
        )
        block_start_s = time.perf_counter()
        state, production_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            cycle_count=production_cycle_count,
            attempt_exchange=len(settings.lambdas) > 1,
            retain_samples=True,
        )
        _report_hrex_block(
            stage="nvt-operator-production",
            block_index=refinement_batch,
            block_elapsed_s=time.perf_counter() - block_start_s,
            total_elapsed_s=time.perf_counter() - sampling_start_s,
            state=state,
            settings=settings,
            block=production_block,
        )
        for chain_position, chain_index in enumerate(unique_chain_indices):
            chain_mask = production_block.physical_ladder_indices == chain_index
            batch_configurations_m = production_block.physical_configurations_m[
                chain_mask
            ]
            if batch_configurations_m.shape[0] != equilibrium_samples_per_batch:
                raise ValueError(
                    "NVT operator block did not retain the configured samples "
                    f"for chain {chain_index}: "
                    f"retained={batch_configurations_m.shape[0]}, "
                    f"required={equilibrium_samples_per_batch}"
                )
            molecular_diffusions_m2_s = (
                configuration_conditioned_molecular_diffusion_batch(
                    positions_batch_m=batch_configurations_m,
                    system=system,
                    molecular_memory=molecular_memory,
                )
            )
            molecular_diffusion_blocks_m2_s = molecular_diffusions_m2_s.reshape(
                (
                    molecular_diffusions_m2_s.shape[0],
                    len(system.molecule_atom_indices),
                    CARTESIAN_DIMENSION,
                    len(system.molecule_atom_indices),
                    CARTESIAN_DIMENSION,
                )
            )
            charge_weighted_diffusion_traces_C2_m2_s = np.trace(
                molecular_diffusion_blocks_m2_s,
                axis1=2,
                axis2=4,
            ) * (
                molecular_charges_C[None, :, None] * molecular_charges_C[None, None, :]
            )
            for first_species, second_species in combinations_with_replacement(
                ionic_species, 2
            ):
                first_indices = np.flatnonzero(
                    np.asarray(system.molecule_species_names) == first_species
                )
                second_indices = np.flatnonzero(
                    np.asarray(system.molecule_species_names) == second_species
                )
                contribution = np.sum(
                    charge_weighted_diffusion_traces_C2_m2_s[
                        :, first_indices[:, None], second_indices[None, :]
                    ]
                )
                if first_species != second_species:
                    contribution += np.sum(
                        charge_weighted_diffusion_traces_C2_m2_s[
                            :, second_indices[:, None], first_indices[None, :]
                        ]
                    )
                species_direct_sums_S_m[f"{first_species}|{second_species}"] += (
                    operator_conductivity_prefactor * float(contribution)
                )
            molecular_diffusion_sample_count += molecular_diffusions_m2_s.shape[0]
            molecular_axis_diffusions_m2_s = molecular_diffusions_m2_s[
                :, ::CARTESIAN_DIMENSION, ::CARTESIAN_DIMENSION
            ]
            diffusion_eigenvalues, diffusion_eigenvectors = np.linalg.eigh(
                molecular_axis_diffusions_m2_s
            )
            diffusion_scales = np.maximum(
                np.max(diffusion_eigenvalues, axis=1), np.finfo(float).tiny
            )
            retained_diffusion_modes = diffusion_eigenvalues > (
                numerics.eigenvalue_relative_tolerance * diffusion_scales[:, None]
            )
            molecular_diffusion_square_roots = (
                diffusion_eigenvectors
                * np.sqrt(
                    np.where(retained_diffusion_modes, diffusion_eigenvalues, 0.0)
                )[:, None, :]
            )
            diffusion_square_roots = np.einsum(
                "bij,ac->biajc",
                molecular_diffusion_square_roots,
                np.eye(CARTESIAN_DIMENSION),
            ).reshape(
                molecular_diffusions_m2_s.shape[0],
                CARTESIAN_DIMENSION * len(system.molecule_atom_indices),
                CARTESIAN_DIMENSION * len(system.molecule_atom_indices),
            )
            assembly_start_s = time.perf_counter()
            (
                batch_dirichlet,
                batch_coupling,
                batch_direct,
                batch_diagnostics,
                batch_complete_operators,
            ) = galerkin_assembler.assemble_batch(
                configurations_m=batch_configurations_m,
                diffusion_square_roots=diffusion_square_roots,
                polarization_gradients=polarization_gradients,
            )
            if not projection_throughput_reported:
                throughput_elapsed_s = time.perf_counter() - assembly_start_s
                configurations_per_s = (
                    batch_configurations_m.shape[0] / throughput_elapsed_s
                )
                print(
                    "[operator assembly] "
                    f"configurations_per_s={configurations_per_s:.12g} "
                    f"batch_configurations={batch_configurations_m.shape[0]}",
                    flush=True,
                )
                projection_throughput_reported = True
            basis_count = galerkin_assembler.basis_count
            if not cumulative_chain_dirichlet:
                cumulative_chain_dirichlet.extend(
                    np.zeros((basis_count, basis_count))
                    for _chain_index in unique_chain_indices
                )
                cumulative_chain_coupling.extend(
                    np.zeros((basis_count, CARTESIAN_DIMENSION))
                    for _chain_index in unique_chain_indices
                )
                cumulative_chain_direct.extend(
                    np.zeros(CARTESIAN_DIMENSION)
                    for _chain_index in unique_chain_indices
                )
            cumulative_chain_dirichlet[chain_position] += batch_dirichlet
            cumulative_chain_coupling[chain_position] += batch_coupling
            cumulative_chain_direct[chain_position] += batch_direct
            chain_operator_series[chain_position].extend(batch_diagnostics)
            chain_complete_operator_series[chain_position].extend(
                batch_complete_operators
            )
            np.savez_compressed(
                operator_checkpoint_directory
                / (
                    f"refinement_{refinement_batch:04d}_"
                    f"chain_{chain_index:04d}.npz"
                ),
                physical_configurations_m=batch_configurations_m,
                physical_box_vectors_by_sample_m=(
                    production_block.physical_box_vectors_by_sample_m[chain_mask]
                ),
                molecular_diffusions_m2_s=molecular_diffusions_m2_s,
                complete_operator_samples=batch_complete_operators,
                chain_index=np.asarray(chain_index),
                refinement_batch=np.asarray(refinement_batch),
            )
            chain_operator_blocks[chain_position].append(
                (
                    batch_dirichlet,
                    batch_coupling,
                    batch_direct,
                    int(batch_configurations_m.shape[0]),
                )
            )
        write_json_object(
            operator_checkpoint_directory / "manifest.json",
            {
                "completed_refinement_batch": refinement_batch,
                "chain_count": int(unique_chain_indices.size),
                "samples_per_chain": samples_per_chain,
                "basis_count": galerkin_assembler.basis_count,
            },
            "conductivity operator checkpoint manifest",
        )
        chain_statistics = tuple(
            (
                cumulative_chain_dirichlet[chain_position] / samples_per_chain,
                cumulative_chain_coupling[chain_position] / samples_per_chain,
                cumulative_chain_direct[chain_position] / samples_per_chain,
            )
            for chain_position in range(unique_chain_indices.size)
        )
        maximum_relative_disagreement = maximum_chain_operator_relative_disagreement(
            chain_statistics=chain_statistics,
            eigenvalue_relative_tolerance=(numerics.eigenvalue_relative_tolerance),
            maximum_basis_size=numerics.maximum_basis_size,
        )
        pooled_dirichlet = np.mean(
            np.stack(tuple(values[0] for values in chain_statistics)), axis=0
        )
        pooled_coupling = np.mean(
            np.stack(tuple(values[1] for values in chain_statistics)), axis=0
        )
        pooled_direct_axes = np.mean(
            np.stack(tuple(values[2] for values in chain_statistics)), axis=0
        )
        volume_m3 = abs(np.linalg.det(system.box_vectors_m))
        conductivity_prefactor = 1.0 / (3.0 * K_B * temperature_K * volume_m3)
        pooled_dirichlet_diagonal = np.diag(pooled_dirichlet)
        online_active_basis = pooled_dirichlet_diagonal > 0.0
        if not np.any(online_active_basis):
            raise ValueError("pooled operator has zero Dirichlet energy")
        online_basis_scales = np.sqrt(pooled_dirichlet_diagonal[online_active_basis])
        normalized_pooled_dirichlet = pooled_dirichlet[
            np.ix_(online_active_basis, online_active_basis)
        ] / (online_basis_scales[:, None] * online_basis_scales[None, :])
        normalized_pooled_coupling = (
            pooled_coupling[online_active_basis] / online_basis_scales[:, None]
        )
        provisional_direct_S_m = conductivity_prefactor * float(
            np.sum(pooled_direct_axes)
        )
        provisional_correction_S_m = conductivity_prefactor * float(
            np.trace(
                normalized_pooled_coupling.T
                @ symmetric_psd_pseudoinverse(
                    normalized_pooled_dirichlet,
                    numerics.eigenvalue_relative_tolerance,
                )
                @ normalized_pooled_coupling
            )
        )
        provisional_conductivity_S_m = (
            provisional_direct_S_m - provisional_correction_S_m
        )
        online_conductivity_estimates_S_m.append(provisional_conductivity_S_m)
        online_operator_disagreements.append(maximum_relative_disagreement)
        for chain_position, chain_blocks in enumerate(chain_operator_blocks):
            (
                block_dirichlet_sum,
                block_coupling_sum,
                block_direct_sum,
                block_sample_count,
            ) = chain_blocks[-1]
            block_dirichlet = block_dirichlet_sum / block_sample_count
            block_coupling = block_coupling_sum / block_sample_count
            block_direct = block_direct_sum / block_sample_count
            normalized_block_dirichlet = block_dirichlet[
                np.ix_(online_active_basis, online_active_basis)
            ] / (online_basis_scales[:, None] * online_basis_scales[None, :])
            normalized_block_coupling = (
                block_coupling[online_active_basis] / online_basis_scales[:, None]
            )
            block_correction_S_m = conductivity_prefactor * float(
                np.trace(
                    normalized_block_coupling.T
                    @ symmetric_psd_pseudoinverse(
                        normalized_block_dirichlet,
                        numerics.eigenvalue_relative_tolerance,
                    )
                    @ normalized_block_coupling
                )
            )
            online_block_conductivities_by_chain_S_m[chain_position].append(
                conductivity_prefactor * float(np.sum(block_direct))
                - block_correction_S_m
            )
        provisional_conductivity_mcse_S_m = math.inf
        if unique_chain_indices.size > 1:
            chain_mean_conductivities_S_m = np.asarray(
                tuple(
                    np.mean(chain_block_conductivities_S_m)
                    for chain_block_conductivities_S_m in (
                        online_block_conductivities_by_chain_S_m
                    )
                )
            )
            provisional_conductivity_mcse_S_m = float(
                np.std(chain_mean_conductivities_S_m, ddof=1)
                / math.sqrt(unique_chain_indices.size)
            )
        provisional_conductivity_change_S_m = abs(
            provisional_conductivity_S_m - previous_provisional_conductivity_S_m
        )
        conductivity_precision_reached = (
            refinement_batch > 1
            and provisional_conductivity_mcse_S_m <= numerics.conductivity_tolerance_S_m
            and provisional_conductivity_change_S_m
            <= numerics.conductivity_tolerance_S_m
        )
        print(
            "[conductivity estimate] "
            f"refinement_batch={refinement_batch} "
            f"samples_per_chain={samples_per_chain} "
            f"conductivity_S_m={provisional_conductivity_S_m:.12g} "
            f"direct_S_m={provisional_direct_S_m:.12g} "
            f"correction_S_m={provisional_correction_S_m:.12g} "
            f"mcse_S_m={provisional_conductivity_mcse_S_m:.12g} "
            f"change_S_m={provisional_conductivity_change_S_m:.12g} "
            f"precision_reached={conductivity_precision_reached}",
            flush=True,
        )
        previous_provisional_conductivity_S_m = provisional_conductivity_S_m
        current_operator_chains = np.asarray(
            tuple(
                np.asarray(series[:samples_per_chain])
                for series in chain_operator_series
            )
        )
        if not diagnostic_bases and samples_per_chain >= diagnostic_pilot_sample_count:
            diagnostic_bases.append(
                fit_operator_diagnostic_basis(
                    chain_operator_series=current_operator_chains[
                        :, :diagnostic_pilot_sample_count
                    ],
                    eigenvalue_relative_tolerance=(
                        numerics.eigenvalue_relative_tolerance
                    ),
                    maximum_mode_count=numerics.maximum_basis_size,
                )
            )
        operator_certified = False
        influence_rhat = math.inf
        influence_effective_sample_size = 0.0
        if diagnostic_bases:
            diagnostic_basis = diagnostic_bases[0]
            diagnostic_operator_chains = (
                current_operator_chains[:, diagnostic_pilot_sample_count:]
                - diagnostic_basis.mean
            ) @ diagnostic_basis.loadings
            if diagnostic_operator_chains.shape[1] >= 4:
                operator_effective_sample_size = (
                    multivariate_batch_means_effective_sample_size(
                        diagnostic_operator_chains
                    )
                )
                mode_diagnostics = fixed_operator_mode_diagnostics(
                    chain_operator_series=current_operator_chains[
                        :, diagnostic_pilot_sample_count:
                    ],
                    diagnostic_basis=diagnostic_basis,
                    dirichlet_diagonal_count=basis_count,
                    coupling_count=basis_count * CARTESIAN_DIMENSION,
                    direct_count=CARTESIAN_DIMENSION,
                )
                maximum_split_rhat = max(
                    max(mode.bulk_rhat, mode.folded_rhat) for mode in mode_diagnostics
                )
                current_complete_operator_chains = np.asarray(
                    tuple(
                        np.asarray(series[:samples_per_chain])
                        for series in chain_complete_operator_series
                    )
                )
                influence_diagnostic = conductivity_influence_diagnostic(
                    chain_complete_operator_series=(
                        current_complete_operator_chains[
                            :, diagnostic_pilot_sample_count:
                        ]
                    ),
                    basis_count=basis_count,
                    temperature_K=temperature_K,
                    volume_m3=abs(np.linalg.det(system.box_vectors_m)),
                    eigenvalue_relative_tolerance=(
                        numerics.eigenvalue_relative_tolerance
                    ),
                )
                influence_rhat = max(
                    influence_diagnostic.bulk_rhat,
                    influence_diagnostic.folded_rhat,
                )
                influence_effective_sample_size = (
                    influence_diagnostic.effective_sample_size
                )
                operator_certified = (
                    maximum_relative_disagreement
                    <= numerics.equilibrium_observable_relative_tolerance
                    and operator_effective_sample_size
                    >= numerics.minimum_effective_sample_size
                    and maximum_split_rhat <= numerics.maximum_split_rhat
                    and influence_rhat <= numerics.maximum_split_rhat
                    and influence_effective_sample_size
                    >= numerics.minimum_effective_sample_size
                )
        operator_diagnostics_certified = operator_certified
        print(
            "[operator diagnostics] "
            f"refinement_batch={refinement_batch} "
            f"relative_disagreement={maximum_relative_disagreement:.12g} "
            f"effective_sample_size={operator_effective_sample_size:.12g} "
            f"maximum_split_rhat={maximum_split_rhat:.12g} "
            f"influence_effective_sample_size={influence_effective_sample_size:.12g} "
            f"influence_rhat={influence_rhat:.12g} "
            f"certified={operator_certified}",
            flush=True,
        )
        if conductivity_precision_reached:
            admitted_chain_statistics = chain_statistics
            admitted_samples_per_chain = samples_per_chain
            break
    if not admitted_chain_statistics:
        raise ValueError(
            "conductivity estimate did not reach the configured block precision "
            "within the refinement limit: "
            f"conductivity_mcse_S_m={provisional_conductivity_mcse_S_m:.12g}, "
            f"conductivity_change_S_m={provisional_conductivity_change_S_m:.12g}, "
            f"tolerance_S_m={numerics.conductivity_tolerance_S_m:.12g}, "
            f"refinement_batches={equilibrium_maximum_refinement_batches}"
        )
    conductivity = provisional_conductivity_S_m
    direct = provisional_direct_S_m
    history = tuple(online_conductivity_estimates_S_m)
    residuals = tuple(online_operator_disagreements)
    basis_size = int(np.count_nonzero(online_active_basis))
    conductivity_mcse_S_m = provisional_conductivity_mcse_S_m
    basis_diagnostics_certified = conductivity_precision_reached
    species_direct_contributions_S_m = tuple(
        (
            species_pair,
            contribution_sum_S_m / molecular_diffusion_sample_count,
        )
        for species_pair, contribution_sum_S_m in sorted(
            species_direct_sums_S_m.items()
        )
    )
    decomposed_direct_S_m = sum(
        contribution for _species_pair, contribution in species_direct_contributions_S_m
    )
    direct_decomposition_scale_S_m = max(abs(direct), np.finfo(float).tiny)
    if abs(decomposed_direct_S_m - direct) / direct_decomposition_scale_S_m > math.sqrt(
        np.finfo(float).eps
    ):
        raise ValueError(
            "species direct-conductivity decomposition does not reproduce the "
            "pooled direct term"
        )
    return (
        conductivity,
        direct,
        history,
        residuals,
        basis_size,
        operator_effective_sample_size,
        maximum_split_rhat,
        conductivity_mcse_S_m,
        admitted_samples_per_chain,
        operator_diagnostics_certified,
        species_direct_contributions_S_m,
        conductivity_precision_reached,
        basis_diagnostics_certified,
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
        np.sum(model.forces_N(positions_m, model.system.box_vectors_m) * direction)
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
    dynamics_record = asdict(dynamics)
    hrex_lambdas = tuple(dynamics_record.pop("ionic_hrex_lambdas"))
    values = tuple(dynamics_record.values()) + tuple(asdict(numerics).values())
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
        numerics.memory_psd_relative_tolerance,
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
    if not (
        numerics.pressure_log_volume_derivative_check_step
        < numerics.pressure_log_volume_derivative_step
    ):
        raise ValueError(
            "pressure derivative check step must be smaller than the primary step"
        )
    if not (numerics.lennard_jones_switch_start_m < numerics.lennard_jones_cutoff_m):
        raise ValueError("Lennard-Jones switch start must be below the cutoff")


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
    if (
        temperature_K <= 0.0
        or liquid_density_kg_m3 <= 0.0
        or minimum_explicit_molecule_count <= 0
    ):
        raise ValueError(
            "temperature, liquid density, and minimum molecule count must be positive"
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
    for runtime_control_name in (
        "initial_relaxation_maximum_force_evaluations",
        "initial_relaxation_maximum_elapsed_s",
        "initial_relaxation_maximum_stagnant_iterations",
        "initial_relaxation_progress_stride",
        "force_batch_size",
        "initial_force_tolerance_N",
        "equilibrium_sample_count",
        "equilibrium_maximum_refinement_batches",
        "hamiltonian_timestep_s",
        "ionic_hrex_lambdas",
        "hmc_steps_min",
        "hmc_steps_max",
        "hmc_momentum_persistence",
        "hmc_full_refresh_stride",
        "exchange_stride",
        "hrex_warmup_cycle_count",
        "hrex_measurement_stride",
        "hrex_block_cycle_count",
    ):
        checkpoint_dynamics_record.pop(runtime_control_name)
    checkpoint_numerics_record = asdict(numerics)
    request_fingerprint_record = {
        "recipe": {
            "solvents": dict(recipe.solvents),
            "salts": dict(recipe.salts),
            "additives": dict(recipe.additives),
        },
        "temperature_K": temperature_K,
        "liquid_density_kg_m3": liquid_density_kg_m3,
        "minimum_explicit_molecule_count": minimum_explicit_molecule_count,
        "dynamics": checkpoint_dynamics_record,
        "numerics": checkpoint_numerics_record,
        "packing_random_seeds": packing_random_seeds,
        "sampling_random_seed": sampling_random_seed,
    }
    checkpoint_fingerprint = hashlib.sha256(
        json.dumps(
            request_fingerprint_record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if initialization_checkpoint_path.is_file():
        with initialization_checkpoint_path.open("rb") as checkpoint_file:
            checkpoint_payload = pickle.load(checkpoint_file)
        if checkpoint_payload["fingerprint"] != checkpoint_fingerprint:
            raise ValueError(
                "initialization checkpoint does not match the requested calculation"
            )
        recipe_realization = checkpoint_payload["metadata"]["recipe_realization"]
    else:
        recipe_realization = select_integer_recipe_realization(
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
            maximum_explicit_molecule_count=(dynamics.maximum_explicit_molecule_count),
            maximum_atom_count=dynamics.maximum_atom_count,
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
        if checkpoint_payload["fingerprint"] != checkpoint_fingerprint:
            raise ValueError(
                "initialization checkpoint does not match the requested calculation"
            )
        density_conditioned_systems.extend(checkpoint_payload["systems"])
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
            "recipe_realization": recipe_realization,
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
            "fingerprint": checkpoint_fingerprint,
            "metadata": checkpoint_metadata,
            "stage": "packed",
            "systems": tuple(density_conditioned_systems),
            "velocities_m_s": np.zeros_like(
                np.stack(
                    tuple(system.positions_m for system in density_conditioned_systems)
                )
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
        checkpoint_fingerprint=checkpoint_fingerprint,
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
    molecular_memory = fit_transferable_molecular_memory_operator(
        system=relaxed_system,
        temperature_K=temperature_K,
        operator_data_root=(
            Path(__file__).parent / "physical_library" / "lammps_operator_data"
        ),
        eigenvalue_relative_tolerance=numerics.eigenvalue_relative_tolerance,
    )
    molecular_self_frictions_by_species_kg_s = tuple(
        (
            species_name,
            float(
                np.mean(
                    molecular_memory.molecular_self_frictions_kg_s[
                        np.asarray(relaxed_system.molecule_species_names)
                        == species_name
                    ]
                )
            ),
        )
        for species_name in sorted(set(relaxed_system.molecule_species_names))
    )
    memory_descriptor_leverage_by_species = tuple(
        (
            species_name,
            float(
                np.mean(
                    molecular_memory.molecular_descriptor_leverages[
                        np.asarray(relaxed_system.molecule_species_names)
                        == species_name
                    ]
                )
            ),
        )
        for species_name in sorted(set(relaxed_system.molecule_species_names))
    )
    pair_friction_values_by_species: dict[str, list[float]] = {}
    pair_leverage_values_by_species: dict[str, list[float]] = {}
    for pair_index, (first_molecule, second_molecule) in enumerate(
        combinations(range(len(relaxed_system.molecule_atom_indices)), 2)
    ):
        first_species = relaxed_system.molecule_species_names[first_molecule]
        second_species = relaxed_system.molecule_species_names[second_molecule]
        species_pair = "|".join(sorted((first_species, second_species)))
        if species_pair not in pair_friction_values_by_species:
            pair_friction_values_by_species[species_pair] = []
            pair_leverage_values_by_species[species_pair] = []
        pair_friction_values_by_species[species_pair].append(
            float(
                molecular_memory.molecular_pair_frictions_kg_s[
                    first_molecule, second_molecule
                ]
            )
        )
        pair_leverage_values_by_species[species_pair].append(
            float(molecular_memory.molecular_pair_descriptor_leverages[pair_index])
        )
    molecular_pair_frictions_by_species_kg_s = tuple(
        (species_pair, float(np.mean(pair_friction_values)))
        for species_pair, pair_friction_values in sorted(
            pair_friction_values_by_species.items()
        )
    )
    memory_pair_descriptor_leverage_by_species = tuple(
        (species_pair, float(np.mean(pair_leverages)))
        for species_pair, pair_leverages in sorted(
            pair_leverage_values_by_species.items()
        )
    )
    print(
        "[memory model] "
        f"conditional_fit_relative_error={molecular_memory.conditional_kernel_fit_relative_error:.12g} "
        f"conditional_heldout_relative_error={molecular_memory.conditional_kernel_heldout_relative_error:.12g} "
        f"self_frictions_kg_s={molecular_self_frictions_by_species_kg_s} "
        f"pair_frictions_kg_s={molecular_pair_frictions_by_species_kg_s} "
        f"self_descriptor_leverage={memory_descriptor_leverage_by_species} "
        f"pair_descriptor_leverage={memory_pair_descriptor_leverage_by_species}",
        flush=True,
    )
    (
        conductivity,
        direct,
        history,
        residuals,
        basis_size,
        operator_effective_sample_size,
        maximum_split_rhat,
        conductivity_mcse_S_m,
        equilibrium_samples_per_chain,
        operator_diagnostics_certified,
        species_direct_contributions_S_m,
        conductivity_precision_reached,
        basis_diagnostics_certified,
    ) = projected_conductivity_sequence(
        model=interaction_model,
        state=sampling_state,
        settings=hrex_settings,
        equilibrium_samples_per_batch=dynamics.equilibrium_sample_count,
        equilibrium_maximum_refinement_batches=(
            dynamics.equilibrium_maximum_refinement_batches
        ),
        system=relaxed_system,
        temperature_K=temperature_K,
        molecular_memory=molecular_memory,
        numerics=numerics,
        operator_checkpoint_directory=initialization_checkpoint_path.with_suffix(
            ".operator_corpus"
        ),
    )
    conditioned_volume_m3 = float(abs(np.linalg.det(relaxed_system.box_vectors_m)))
    conditioned_density_g_cm3 = float(
        np.sum(relaxed_system.masses_kg) / conditioned_volume_m3 / KG_M3_PER_G_ML
    )
    return ConductivityResult(
        conductivity_S_m=conductivity,
        direct_current_term_S_m=direct,
        projected_correction_S_m=direct - conductivity,
        conditioned_volume_m3=conditioned_volume_m3,
        conditioned_density_g_cm3=conditioned_density_g_cm3,
        thermodynamic_state="NVT density-conditioned",
        density_source=density_source,
        integrated_memory_eigenvalues_kg_s=tuple(
            float(value)
            for value in np.linalg.eigvalsh(molecular_memory.integrated_friction_kg_s)
        ),
        diffusion_eigenvalues_m2_s=tuple(
            float(value)
            for value in np.linalg.eigvalsh(molecular_memory.diffusion_m2_s)
        ),
        basis_size=basis_size,
        basis_conductivities_S_m=history,
        residual_history=residuals,
        maximum_residual_score=residuals[-1],
        equilibrium_sample_count=(
            dynamics.equilibrium_chain_count * equilibrium_samples_per_chain
        ),
        equilibrium_chain_count=dynamics.equilibrium_chain_count,
        memory_sample_count=molecular_memory.sample_count,
        effective_sample_size=operator_effective_sample_size,
        maximum_split_rhat=maximum_split_rhat,
        conductivity_mcse_S_m=conductivity_mcse_S_m,
        conductivity_precision_reached=conductivity_precision_reached,
        basis_diagnostics_certified=basis_diagnostics_certified,
        operator_diagnostics_certified=operator_diagnostics_certified,
        species_direct_contributions_S_m=species_direct_contributions_S_m,
        molecular_self_frictions_by_species_kg_s=(
            molecular_self_frictions_by_species_kg_s
        ),
        molecular_pair_frictions_by_species_kg_s=(
            molecular_pair_frictions_by_species_kg_s
        ),
        memory_descriptor_leverage_by_species=(memory_descriptor_leverage_by_species),
        memory_pair_descriptor_leverage_by_species=(
            memory_pair_descriptor_leverage_by_species
        ),
        memory_conditional_fit_relative_error=(
            molecular_memory.conditional_kernel_fit_relative_error
        ),
        memory_conditional_heldout_relative_error=(
            molecular_memory.conditional_kernel_heldout_relative_error
        ),
        realized_formula_unit_counts=recipe_realization.formula_unit_counts,
        realized_molecule_counts=recipe_realization.explicit_species_counts,
        realized_atom_count=recipe_realization.atom_count,
        realized_solvent_volume_fractions=(
            recipe_realization.realized_solvent_volume_fractions
        ),
        realized_salt_molarities_mol_L=(
            recipe_realization.realized_salt_molarities_mol_L
        ),
        realized_additive_weight_fractions=(
            recipe_realization.realized_additive_weight_fractions
        ),
        realized_native_unit_deviations=recipe_realization.native_unit_deviations,
    )


@dataclass(frozen=True)
class BatchedHmcTransitionResult:
    positions_m: Array
    momenta_kg_m_s: Array
    accepted: Array
    log_acceptance_probabilities: Array
    energy_errors_over_kbt: Array
    relative_energy_errors: Array
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
    physical_box_vectors_by_sample_m: Array
    physical_ladder_indices: Array
    sampled_volumes_m3: Array
    sampled_energies_J: Array
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


class TemperedComponents(Protocol):
    fixed_J: float
    ion_ion_J: float
    ion_neutral_J: float


class IonicTemperedModel(Protocol):
    system: IonicMolecularSystem

    def tempered_energy_for_configuration_J(
        self, positions_m: Array, box_vectors_m: Array, lambda_value: float
    ) -> float: ...

    def tempered_forces_N(
        self, positions_m: Array, box_vectors_m: Array, lambda_value: float
    ) -> Array: ...

    def energy_components_J(
        self, positions_m: Array, box_vectors_m: Array
    ) -> TemperedComponents: ...

    def energy_force_batch(
        self,
        positions_batch_m: Array,
        box_vectors_batch_m: Array,
        lambda_values: Array,
    ): ...

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
    energy_scales_J = np.maximum(
        np.abs(initial_state.energy_J + initial_kinetic_energies_J),
        np.finfo(float).tiny,
    )
    return BatchedHmcTransitionResult(
        positions_m=accepted_positions_m,
        momenta_kg_m_s=retained_momenta_kg_m_s,
        accepted=accepted,
        log_acceptance_probabilities=log_acceptance_probabilities,
        energy_errors_over_kbt=energy_errors_J / (K_B * temperature_K),
        relative_energy_errors=np.abs(energy_errors_J) / energy_scales_J,
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


def _reset_hrex_round_trip_tracking(
    state: IonicHrexState,
    replica_count: int,
    reset_walker_identifiers: bool,
) -> None:
    ladder_count = state.walker_identifiers.shape[0]
    if reset_walker_identifiers:
        state.walker_identifiers = np.tile(np.arange(replica_count), (ladder_count, 1))
    state.visited_lowest_lambda = np.zeros((ladder_count, replica_count), dtype=bool)
    state.completed_round_trips = np.zeros((ladder_count, replica_count), dtype=int)
    state.round_trip_phase = np.zeros((ladder_count, replica_count), dtype=np.int8)
    for ladder_index in range(ladder_count):
        physical_walker = state.walker_identifiers[ladder_index, 0]
        state.round_trip_phase[ladder_index, physical_walker] = 1


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
    boxes: list[Array] = []
    ladder_indices: list[int] = []
    volumes: list[float] = []
    energies: list[float] = []
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
            for ladder_index, physical_index in enumerate(physical_indices):
                configurations.append(state.positions_m[physical_index].copy())
                boxes.append(state.boxes_m[physical_index].copy())
                ladder_indices.append(ladder_index)
                volumes.append(abs(np.linalg.det(state.boxes_m[physical_index])))
                energies.append(
                    _tempered_energy_from_components_J(
                        state.component_energies_J[physical_index], 1.0
                    )
                )
    state.cycle_index += cycle_count
    state.random_generator_state = random_generator.bit_generator.state
    return state, IonicHrexBlock(
        physical_configurations_m=np.asarray(configurations),
        physical_box_vectors_by_sample_m=np.asarray(boxes),
        physical_ladder_indices=np.asarray(ladder_indices, dtype=int),
        sampled_volumes_m3=np.asarray(volumes),
        sampled_energies_J=np.asarray(energies),
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
    return DynamicsSettings(**record["dynamics"]), NumericalSettings(
        **record["numerics"]
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
    print(f"direct = {result.direct_current_term_S_m:.8g} S/m")
    print(f"projected correction = {result.projected_correction_S_m:.8g} S/m")
    print(f"thermodynamic state = {result.thermodynamic_state}")
    print(f"density source = {result.density_source}")
    print(f"conditioned volume = {result.conditioned_volume_m3:.8g} m3")
    print(f"conditioned density = {result.conditioned_density_g_cm3:.8g} g/cm3")
    print(f"basis sequence = {result.basis_conductivities_S_m}")
    print(f"residual sequence = {result.residual_history}")
    print(f"species direct contributions = {result.species_direct_contributions_S_m}")
    print(
        "realized recipe = "
        f"counts {result.realized_molecule_counts}; "
        f"solvents v/v {result.realized_solvent_volume_fractions}; "
        f"salts mol/L {result.realized_salt_molarities_mol_L}; "
        f"additives wt {result.realized_additive_weight_fractions}"
    )
    print(
        "memory regression = "
        f"fit error {result.memory_conditional_fit_relative_error:.6g}; "
        f"heldout error {result.memory_conditional_heldout_relative_error:.6g}; "
        f"self frictions {result.molecular_self_frictions_by_species_kg_s}; "
        f"pair frictions {result.molecular_pair_frictions_by_species_kg_s}; "
        f"self leverage {result.memory_descriptor_leverage_by_species}; "
        f"pair leverage {result.memory_pair_descriptor_leverage_by_species}"
    )
    print(
        f"basis size = {result.basis_size}; equilibrium samples = "
        f"{result.equilibrium_sample_count}; memory samples = "
        f"{result.memory_sample_count}; ESS = {result.effective_sample_size:.6g}"
        f"; split-Rhat = {result.maximum_split_rhat:.6g}"
        f"; conductivity MCSE = {result.conductivity_mcse_S_m:.6g} S/m"
        f"; conductivity precision reached = "
        f"{result.conductivity_precision_reached}"
        f"; basis diagnostics certified = {result.basis_diagnostics_certified}"
        f"; operator diagnostics certified = "
        f"{result.operator_diagnostics_certified}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
