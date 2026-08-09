"""Full-configuration reversible conductivity from an analytical molecular model.

The executable constructs one periodic molecular liquid, samples its Boltzmann
measure, and solves the reversible Smoluchowski current-corrector problem in a
nested basis of smooth full-configuration observables.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from functools import cache
from itertools import combinations, combinations_with_replacement
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Protocol
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import Bounds, LinearConstraint, milp, minimize, nnls
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
    N_A,
    SECONDS_PER_MINUTE,
    S_M_TO_MS_CM,
)
from conductivity.physical_library.library_io import load_physical_library
from conductivity.physical_library.physical_objects import (
    LJ_ATTRACTIVE_EXPONENT,
    LJ_REPULSIVE_EXPONENT_MULTIPLIER,
)
from electrolyte_model import ElectrolyteFormulation, ElectrolyteRecipeModel
from species_data import ADDITIVES, SALTS
from utils.strict_validation import read_json_object, write_json_object
from utils.time_series_statistics import (
    autocorrelation_and_effective_sample_size,
    stationary_suffix_candidates,
)

Array = np.ndarray
CARTESIAN_DIMENSION = 3
INITIAL_RELAXATION_FORCE_MARGIN = 0.5  # Resolve below the final force criterion.
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
# User operational limit: a projection exceeding one minute indicates a broken hot path.
PROJECTION_SANITY_RUNTIME_S = SECONDS_PER_MINUTE
MINIMUM_OPERATOR_DIAGNOSTIC_PILOT_SAMPLES = 2 * CARTESIAN_DIMENSION
MINIMUM_OPERATOR_DIAGNOSTIC_EVALUATION_SAMPLES = CARTESIAN_DIMENSION + 1
HMC_CALIBRATION_WINDOW_CYCLE_COUNT = 4
HMC_SELECTION_WINDOW_COUNT = 2
HMC_MAXIMUM_CANDIDATE_COUNT = 6
HMC_HOLDOUT_WINDOW_COUNT = 2
HMC_COMBINED_PILOT_CYCLE_COUNT = 16
HMC_MAXIMUM_COMBINED_PILOT_RETRY_COUNT = 1


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
    initial_relaxation_steps: int
    initial_relaxation_step_m: float
    initial_force_tolerance_N: float
    equilibrium_burn_in_sweeps: int
    equilibrium_sample_count: int
    equilibrium_sweeps_per_sample: int
    equilibrium_chain_count: int
    equilibrium_maximum_refinement_batches: int
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
    ionic_hrex_lambdas: tuple[float, ...]
    hmc_steps_min: int
    hmc_steps_max: int
    hmc_momentum_persistence: float
    hmc_full_refresh_stride: int
    exchange_stride: int
    volume_move_stride: int
    minimum_round_trips: int
    hrex_warmup_cycle_count: int
    hrex_production_cycle_count: int
    hrex_measurement_stride: int
    hrex_block_cycle_count: int
    hrex_block_runtime_limit_s: float
    burn_in_minimum_effective_sample_size: float
    hmc_target_acceptance_minimum: float
    hmc_target_acceptance_maximum: float
    hmc_step_size_adaptation_factor: float
    hmc_log_bracket_width_tolerance: float
    volume_target_acceptance: float
    volume_adaptation_gain: float
    minimum_log_volume_proposal: float
    maximum_log_volume_proposal: float


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
    equilibrium_observable_relative_tolerance: float
    maximum_split_rhat: float
    pressure_log_volume_derivative_step: float
    pressure_log_volume_derivative_check_step: float
    pressure_derivative_relative_tolerance: float
    pressure_volume_relative_tolerance: float


@dataclass(frozen=True)
class RelaxationResult:
    positions_m: Array
    maximum_force_N: float
    iteration_count: int


@dataclass(frozen=True)
class InternalPressureResult:
    internal_pressure_Pa: float
    mean_energy_derivative_J: float
    relative_derivative_mismatch: float


@dataclass(frozen=True)
class PressurePreconditioningResult:
    positions_by_ladder_m: Array
    box_vectors_by_ladder_m: Array
    equilibrium_volume_guess_m3: float
    internal_pressure_Pa: float
    relative_bracket_width: float


@dataclass(frozen=True)
class ConductivityResult:
    conductivity_S_m: float
    direct_current_term_S_m: float
    projected_correction_S_m: float
    equilibrium_volume_m3: float
    equilibrium_density_g_cm3: float
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
        displacement_samples_m -= np.mean(displacement_samples_m, axis=0, keepdims=True)
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
        np.log(mean_square_displacements_m2[-1] / mean_square_displacements_m2[-2])
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
        raise ValueError(
            "zero-frequency molecular diffusion is not positive semidefinite"
        )
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
        molecular_self_frictions_kg_s=(
            np.diag(integrated_friction).reshape((-1, CARTESIAN_DIMENSION)).mean(axis=1)
        ),
        molecular_pair_frictions_kg_s=np.zeros(
            (len(system.molecule_atom_indices), len(system.molecule_atom_indices))
        ),
        temperature_K=temperature_K,
        decay_times_s=(float(lag_times_s[-1]),),
        decay_weights=(1.0,),
        self_memory_spectral_amplitudes_kg_s2=(
            np.diag(integrated_friction)
            .reshape((-1, CARTESIAN_DIMENSION))
            .mean(axis=1)[:, None]
            / float(lag_times_s[-1])
        ),
        geometry_radial_edges_m=np.empty(0),
        self_descriptor_schema=("uniform",),
        pair_descriptor_schema=("pair_radial",),
        self_descriptor_friction_scales=np.asarray((1.0,)),
        pair_descriptor_friction_scales=np.asarray((1.0,)),
        lag_times_s=tuple(float(value) for value in lag_times_s),
        diffusion_plateau_relative_change=diffusion_plateau_relative_change,
        displacement_growth_exponent=displacement_growth_exponent,
        sample_count=velocities.shape[0],
        training_family_labels=(),
        training_dataset_count=0,
        descriptor_rank=0,
        conditional_kernel_fit_relative_error=0.0,
        conditional_kernel_heldout_relative_error=0.0,
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
    molecular_frictions_kg_s = np.exp(
        _regularized_log_friction_prediction(
            training_features=feature_matrix,
            logarithmic_training_targets=np.asarray(logarithmic_friction_targets),
            prediction_features=prediction_features,
        )
    )
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
    molecular_pair_frictions_kg_s = np.exp(
        _regularized_log_friction_prediction(
            training_features=np.stack(pair_feature_rows),
            logarithmic_training_targets=np.asarray(logarithmic_pair_friction_targets),
            prediction_features=pair_prediction_features,
        )
    )
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
    )


def _regularized_log_friction_prediction(
    training_features: Array,
    logarithmic_training_targets: Array,
    prediction_features: Array,
) -> Array:
    descriptor_features = np.asarray(training_features[:, 1:], dtype=float)
    prediction_descriptors = np.asarray(prediction_features[:, 1:], dtype=float)
    descriptor_mean = np.mean(descriptor_features, axis=0)
    descriptor_scale = np.std(descriptor_features, axis=0)
    active_descriptors = descriptor_scale > math.sqrt(np.finfo(float).eps)
    if not np.any(active_descriptors):
        return np.full(
            prediction_features.shape[0],
            np.mean(logarithmic_training_targets),
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
    coefficients = np.linalg.solve(
        descriptor_gram + spectral_penalty * np.eye(descriptor_gram.shape[0]),
        standardized_training.T @ centered_targets,
    )
    return (
        np.mean(logarithmic_training_targets) + standardized_prediction @ coefficients
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
    conditioned_friction = configuration_conditioned_integrated_friction(
        positions_m, system, molecular_memory
    )
    conditioned_diffusion = (
        K_B
        * molecular_memory.temperature_K
        * symmetric_psd_pseudoinverse(
            conditioned_friction,
            relative_tolerance=math.sqrt(np.finfo(float).eps),
        )
    )
    if not np.all(np.isfinite(conditioned_diffusion)):
        raise ValueError("configuration-conditioned diffusion is non-finite")
    return conditioned_diffusion


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


def _recipe_species_mole_weights(recipe: ElectrolyteRecipeModel) -> dict[str, float]:
    formula_species_names = tuple(
        dict.fromkeys(
            tuple(recipe.solvents) + tuple(recipe.salts) + tuple(recipe.additives)
        )
    )
    formulation = ElectrolyteFormulation.__new__(ElectrolyteFormulation)
    formulation.recipe = recipe.to_internal_recipe()
    formula_mole_fractions = formulation.to_mole_fractions(list(formula_species_names))
    species_amounts_mol: dict[str, float] = {}
    for formula_species_name, formula_mole_fraction in zip(
        formula_species_names, formula_mole_fractions, strict=True
    ):
        if formula_species_name in recipe.solvents:
            species_amounts_mol[formula_species_name] = float(formula_mole_fraction)
            continue
        if formula_species_name in recipe.additives:
            additive_record = ADDITIVES[formula_species_name]
            if "cation" not in additive_record or "anion" not in additive_record:
                species_amounts_mol[formula_species_name] = float(formula_mole_fraction)
                continue
            cation_name = f"{additive_record['cation']}+"
            anion_name = str(additive_record["anion"])
        else:
            cation_name, anion_name = _salt_ion_names(formula_species_name)
        for ion_name in (cation_name, anion_name):
            previous_amount_mol = (
                species_amounts_mol[ion_name]
                if ion_name in species_amounts_mol
                else 0.0
            )
            species_amounts_mol[ion_name] = previous_amount_mol + float(
                formula_mole_fraction
            )
    total_amount_mol = sum(species_amounts_mol.values())
    return {
        species_name: amount_mol / total_amount_mol
        for species_name, amount_mol in species_amounts_mol.items()
    }


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
    library = _physical_library_records()
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
        species_name
        for _negative_extent_m, species_name in sorted(species_extent_names)
    )
    species_records = tuple(library.species_records[name] for name in species_names)
    molecular_charges_e = np.asarray(
        [float(record["formal_charge_e"]) for record in species_records]
    )
    fractions = np.asarray([mole_weights[name] for name in species_names])
    counts = charge_neutral_integer_counts(
        fractions, molecular_charges_e, molecule_count
    )
    packing_volume_guess_m3 = sum(
        int(count) * float(record["partial_molar_volume_m3_mol"]) / N_A
        for count, record in zip(counts, species_records, strict=True)
    )
    if packing_volume_guess_m3 <= 0.0:
        raise ValueError("species partial molar volumes do not define a positive cell")
    box_length_m = packing_volume_guess_m3 ** (1.0 / CARTESIAN_DIMENSION)
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
        self._reciprocal_indices = np.asarray(
            [
                (i_value, j_value, k_value)
                for i_value in range(
                    -numerics.ewald_reciprocal_shell,
                    numerics.ewald_reciprocal_shell + 1,
                )
                for j_value in range(
                    -numerics.ewald_reciprocal_shell,
                    numerics.ewald_reciprocal_shell + 1,
                )
                for k_value in range(
                    -numerics.ewald_reciprocal_shell,
                    numerics.ewald_reciprocal_shell + 1,
                )
                if (i_value, j_value, k_value) != (0, 0, 0)
            ],
            dtype=float,
        )
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
        if self._has_bonds:
            bond_indices = self._bond_indices_tensor
            displacement = _torch_minimum_image(
                positions_m[bond_indices[:, 0]] - positions_m[bond_indices[:, 1]],
                box_vectors_m,
            )
            lengths = torch.linalg.norm(displacement, dim=1)
            fixed_energy = fixed_energy + torch.sum(
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
            fixed_energy = fixed_energy + torch.sum(
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
            fixed_energy = fixed_energy + torch.sum(
                self._torsion_amplitudes_tensor
                * (
                    1.0
                    + torch.cos(
                        self._torsion_periodicities_tensor * dihedral
                        - self._torsion_phases_tensor
                    )
                )
            )
        displacement = _torch_minimum_image(
            positions_m[:, None, :] - positions_m[None, :, :], box_vectors_m
        )
        distance = torch.linalg.norm(displacement, dim=2)
        lj_pair_distance = distance[self._lj_pair_i_tensor, self._lj_pair_j_tensor]
        attractive_term = (
            self._lj_sigma_pair_tensor / lj_pair_distance
        ) ** LJ_ATTRACTIVE_EXPONENT
        repulsive_pair_energy = (
            self._lj_pair_scale_tensor
            * 4.0
            * self._lj_epsilon_pair_tensor
            * attractive_term**LJ_REPULSIVE_EXPONENT_MULTIPLIER
        )
        attractive_pair_energy = (
            self._lj_pair_scale_tensor
            * 4.0
            * self._lj_epsilon_pair_tensor
            * attractive_term
        )
        fixed_energy = fixed_energy + torch.sum(repulsive_pair_energy)
        fixed_energy = fixed_energy - torch.sum(
            torch.where(
                self._lj_pair_ionic_count_tensor == 0,
                attractive_pair_energy,
                torch.zeros_like(attractive_pair_energy),
            )
        )
        ion_neutral_energy = ion_neutral_energy - torch.sum(
            torch.where(
                self._lj_pair_ionic_count_tensor == 1,
                attractive_pair_energy,
                torch.zeros_like(attractive_pair_energy),
            )
        )
        ion_ion_energy = ion_ion_energy - torch.sum(
            torch.where(
                self._lj_pair_ionic_count_tensor == 2,
                attractive_pair_energy,
                torch.zeros_like(attractive_pair_energy),
            )
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
        volume_m3 = torch.abs(torch.linalg.det(box_vectors_m))
        green_weights = torch.exp(-reciprocal_squared / (4.0 * ewald_alpha**2)) / (
            EPS_0 * volume_m3 * reciprocal_squared
        )
        fixed_energy = fixed_energy + 0.5 * torch.sum(
            green_weights * (neutral_structure_real**2 + neutral_structure_imaginary**2)
        )
        ion_neutral_energy = ion_neutral_energy + torch.sum(
            green_weights
            * (
                neutral_structure_real * ionic_structure_real
                + neutral_structure_imaginary * ionic_structure_imaginary
            )
        )
        ion_ion_energy = ion_ion_energy + 0.5 * torch.sum(
            green_weights * (ionic_structure_real**2 + ionic_structure_imaginary**2)
        )
        fixed_energy = fixed_energy - (
            ewald_alpha
            * torch.sum(neutral_charges**2)
            / (4.0 * math.pi * math.sqrt(math.pi) * EPS_0)
        )
        ion_ion_energy = ion_ion_energy - (
            ewald_alpha
            * torch.sum(ionic_charges**2)
            / (4.0 * math.pi * math.sqrt(math.pi) * EPS_0)
        )
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
            neutral_charges,
            ionic_charges,
            phases,
            displacements,
            distances,
            reciprocal_vectors,
            green_weights,
        ) = torch.vmap(self._nonpolar_energy_components_tensor)(
            positions_batch_m, box_vectors_batch_m
        )
        if self._polarizable_atom_count == 0:
            return (
                fixed_energies,
                ion_ion_energies,
                ion_neutral_energies,
                torch.zeros_like(fixed_energies),
            )
        (
            polarization_fixed,
            polarization_ion_ion,
            polarization_ion_neutral,
            polarization_residuals,
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
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        electric_fields = torch.einsum(
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
        electric_fields += torch.einsum(
            "bij,bjr,bijd->bird",
            real_field_coefficient,
            charge_columns,
            displacement_m,
        )
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
            excluded_reciprocal_fields = torch.zeros_like(electric_fields)
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
            electric_fields -= excluded_reciprocal_fields
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

        def apply_operator(dipoles: torch.Tensor) -> torch.Tensor:
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
            interaction_field = -torch.einsum(
                "bprk,bk,bkd->bprd",
                reciprocal_amplitudes,
                green_weights_J_m_C2,
                reciprocal_m_inv,
            )
            interaction_field += torch.einsum("bijde,bjre->bird", real_hessian, dipoles)
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
                excluded_field = torch.zeros_like(interaction_field)
                excluded_field = excluded_field.index_add(
                    1,
                    self._active_exclusion_pair_i_tensor,
                    torch.einsum(
                        "bpde,bpre->bprd",
                        excluded_pair_hessian,
                        dipoles[:, self._active_exclusion_pair_j_tensor],
                    ),
                )
                excluded_field = excluded_field.index_add(
                    1,
                    self._active_exclusion_pair_j_tensor,
                    torch.einsum(
                        "bpde,bpre->bprd",
                        excluded_pair_hessian,
                        dipoles[:, self._active_exclusion_pair_i_tensor],
                    ),
                )
                interaction_field -= excluded_field
            interaction_field += self_hessian * dipoles
            return inverse_alpha[None, :, None, None] * dipoles - interaction_field

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
        return polarization_energy_matrix, residual

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        energy_matrix, residual = self._polarization_energy_matrix_batch(
            charge_columns=torch.stack((neutral_charges_C, ionic_charges_C), dim=2),
            phases=phases,
            displacement_m=displacement_m,
            distance_m=distance_m,
            reciprocal_m_inv=reciprocal_m_inv,
            green_weights_J_m_C2=green_weights_J_m_C2,
            ewald_splitting_per_m=ewald_splitting_per_m,
        )
        return (
            energy_matrix[:, 0, 0],
            energy_matrix[:, 1, 1],
            2.0 * energy_matrix[:, 0, 1],
            residual,
        )

    def energy_J(self, positions_m: Array, box_vectors_m: Array) -> float:
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
        positions = torch.tensor(positions_m, dtype=TORCH_DTYPE, requires_grad=True)
        energy = self._energy_tensor(positions, torch.as_tensor(box_vectors_m))
        return -torch.autograd.grad(energy, positions)[0].detach().numpy()

    def tempered_forces_N(
        self,
        positions_m: Array,
        box_vectors_m: Array,
        lambda_value: float,
    ) -> Array:
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
    logarithmic_volume_step: float,
) -> tuple[float, float]:
    if temperature_K <= 0.0 or logarithmic_volume_step <= 0.0:
        raise ValueError("pressure temperature and log-volume step must be positive")
    volumes_m3 = np.abs(np.linalg.det(box_vectors_by_ladder_m))
    if not np.allclose(volumes_m3, volumes_m3[0], rtol=1.0e-12, atol=0.0):
        raise ValueError("internal-pressure chains must share one candidate volume")
    scaled_positions: list[Array] = []
    scaled_boxes: list[Array] = []
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
    components = model.energy_components_batch(
        positions_batch_m=np.asarray(scaled_positions),
        box_vectors_batch_m=np.asarray(scaled_boxes),
    )
    energies_J = (
        components.fixed_energy_J
        + components.ion_ion_energy_J
        + components.ion_neutral_energy_J
    )
    ladder_count = positions_by_ladder_m.shape[0]
    derivatives_J = (
        energies_J[:ladder_count] - energies_J[ladder_count:]
    ) / (2.0 * logarithmic_volume_step)
    mean_derivative_J = float(np.mean(derivatives_J))
    internal_pressure_Pa = (
        len(model.system.molecule_atom_indices) * K_B * temperature_K
        - mean_derivative_J
    ) / float(np.mean(volumes_m3))
    if not math.isfinite(internal_pressure_Pa):
        raise FloatingPointError("internal-pressure evaluation produced a nonfinite value")
    return float(internal_pressure_Pa), mean_derivative_J


def precondition_equilibrium_volume(
    model: AnalyticalPeriodicInteratomicModel,
    positions_by_ladder_m: Array,
    box_vectors_by_ladder_m: Array,
    temperature_K: float,
    pressure_Pa: float,
    dynamics: DynamicsSettings,
    random_seed: int,
) -> PressurePreconditioningResult:
    chain_count = positions_by_ladder_m.shape[0]
    if chain_count != dynamics.equilibrium_chain_count:
        raise ValueError("pressure preconditioner requires every independent chain")
    initial_volume_m3 = float(
        np.mean(np.abs(np.linalg.det(box_vectors_by_ladder_m)))
    )
    initial_box_m = np.eye(CARTESIAN_DIMENSION) * (
        initial_volume_m3 ** (1.0 / CARTESIAN_DIMENSION)
    )
    current_positions_m = np.asarray(
        [
            _scale_molecular_centers_to_box(
                positions_m=positions_m,
                source_box_vectors_m=box_vectors_m,
                target_box_vectors_m=initial_box_m,
                system=model.system,
            )
            for positions_m, box_vectors_m in zip(
                positions_by_ladder_m, box_vectors_by_ladder_m, strict=True
            )
        ]
    )
    current_boxes_m = np.repeat(initial_box_m[None, :, :], chain_count, axis=0)
    momenta_kg_m_s = np.zeros_like(current_positions_m)
    refresh_required = np.ones(chain_count, dtype=bool)
    random_generator = np.random.default_rng(random_seed)
    lower_is_set = False
    upper_is_set = False
    lower_log_volume = math.log(initial_volume_m3)
    upper_log_volume = lower_log_volume
    relative_bracket_width = math.inf
    internal_pressure_Pa = 0.0
    equilibrium_log_volume = lower_log_volume
    pressure_root_bracketed = False
    for candidate_index in range(dynamics.equilibrium_maximum_refinement_batches):
        for response_cycle_index in range(dynamics.hrex_block_cycle_count):
            transition = batched_hmc_transition(
                model=model,
                positions_batch_m=current_positions_m,
                box_vectors_batch_m=current_boxes_m,
                momenta_kg_m_s=momenta_kg_m_s,
                auxiliary_masses_kg=model.system.masses_kg,
                momentum_refresh_required=refresh_required,
                momentum_persistence=dynamics.hmc_momentum_persistence,
                temperature_K=temperature_K,
                lambda_values=np.ones(chain_count),
                timestep_values_s=np.full(chain_count, dynamics.hamiltonian_timestep_s),
                integration_step_counts=np.full(
                    chain_count, 1 + response_cycle_index % 2, dtype=int
                ),
                random_generator=random_generator,
            )
            current_positions_m = transition.positions_m
            momenta_kg_m_s = transition.momenta_kg_m_s
            refresh_required = ~transition.accepted
        internal_pressure_Pa, primary_derivative_J = molecular_center_internal_pressure(
            model=model,
            positions_by_ladder_m=current_positions_m,
            box_vectors_by_ladder_m=current_boxes_m,
            temperature_K=temperature_K,
            logarithmic_volume_step=model.numerics.pressure_log_volume_derivative_step,
        )
        _, check_derivative_J = molecular_center_internal_pressure(
            model=model,
            positions_by_ladder_m=current_positions_m,
            box_vectors_by_ladder_m=current_boxes_m,
            temperature_K=temperature_K,
            logarithmic_volume_step=(
                model.numerics.pressure_log_volume_derivative_check_step
            ),
        )
        derivative_mismatch = abs(primary_derivative_J - check_derivative_J) / max(
            abs(primary_derivative_J), abs(check_derivative_J), K_B * temperature_K
        )
        if derivative_mismatch > model.numerics.pressure_derivative_relative_tolerance:
            raise ValueError(
                "internal-pressure finite differences disagree: "
                f"relative_mismatch={derivative_mismatch:.12g}, "
                f"tolerance={model.numerics.pressure_derivative_relative_tolerance:.12g}"
            )
        current_log_volume = math.log(float(abs(np.linalg.det(current_boxes_m[0]))))
        pressure_error_Pa = internal_pressure_Pa - pressure_Pa
        print(
            f"[pressure preconditioner] candidate={candidate_index + 1} "
            f"volume_m3={math.exp(current_log_volume):.12g} "
            f"internal_pressure_Pa={internal_pressure_Pa:.12g} "
            f"pressure_error_Pa={pressure_error_Pa:.12g} "
            f"derivative_mismatch={derivative_mismatch:.12g}",
            flush=True,
        )
        if pressure_error_Pa > 0.0:
            lower_log_volume = current_log_volume
            lower_is_set = True
        if pressure_error_Pa <= 0.0:
            upper_log_volume = current_log_volume
            upper_is_set = True
        bracket_is_closed = lower_is_set and upper_is_set
        if bracket_is_closed:
            relative_bracket_width = math.exp(
                upper_log_volume - lower_log_volume
            ) - 1.0
            equilibrium_log_volume = 0.5 * (
                lower_log_volume + upper_log_volume
            )
            if relative_bracket_width <= model.numerics.pressure_volume_relative_tolerance:
                pressure_root_bracketed = True
                break
        if not bracket_is_closed and pressure_error_Pa > 0.0:
            equilibrium_log_volume = (
                current_log_volume + dynamics.maximum_log_volume_proposal
            )
        if not bracket_is_closed and pressure_error_Pa <= 0.0:
            equilibrium_log_volume = (
                current_log_volume - dynamics.maximum_log_volume_proposal
            )
        target_volume_m3 = math.exp(equilibrium_log_volume)
        target_box_m = np.eye(CARTESIAN_DIMENSION) * (
            target_volume_m3 ** (1.0 / CARTESIAN_DIMENSION)
        )
        current_positions_m = np.asarray(
            [
                _scale_molecular_centers_to_box(
                    positions_m=positions_m,
                    source_box_vectors_m=box_vectors_m,
                    target_box_vectors_m=target_box_m,
                    system=model.system,
                )
                for positions_m, box_vectors_m in zip(
                    current_positions_m, current_boxes_m, strict=True
                )
            ]
        )
        current_boxes_m = np.repeat(target_box_m[None, :, :], chain_count, axis=0)
        momenta_kg_m_s.fill(0.0)
        refresh_required.fill(True)
    if not pressure_root_bracketed:
        raise ValueError("pressure preconditioner failed to bracket the target pressure")
    final_volume_m3 = math.exp(equilibrium_log_volume)
    final_box_m = np.eye(CARTESIAN_DIMENSION) * (
        final_volume_m3 ** (1.0 / CARTESIAN_DIMENSION)
    )
    final_positions_m = np.asarray(
        [
            _scale_molecular_centers_to_box(
                positions_m=positions_m,
                source_box_vectors_m=box_vectors_m,
                target_box_vectors_m=final_box_m,
                system=model.system,
            )
            for positions_m, box_vectors_m in zip(
                current_positions_m, current_boxes_m, strict=True
            )
        ]
    )
    return PressurePreconditioningResult(
        positions_by_ladder_m=final_positions_m,
        box_vectors_by_ladder_m=np.repeat(
            final_box_m[None, :, :], chain_count, axis=0
        ),
        equilibrium_volume_guess_m3=final_volume_m3,
        internal_pressure_Pa=internal_pressure_Pa,
        relative_bracket_width=relative_bracket_width,
    )


def _concatenate_ionic_hrex_blocks(
    blocks: list[IonicHrexBlock],
) -> IonicHrexBlock:
    return IonicHrexBlock(
        physical_configurations_m=np.concatenate(
            [block.physical_configurations_m for block in blocks], axis=0
        ),
        physical_box_vectors_by_sample_m=np.concatenate(
            [block.physical_box_vectors_by_sample_m for block in blocks], axis=0
        ),
        physical_ladder_indices=np.concatenate(
            [block.physical_ladder_indices for block in blocks]
        ),
        sampled_volumes_m3=np.concatenate(
            [block.sampled_volumes_m3 for block in blocks]
        ),
        sampled_energies_J=np.concatenate(
            [block.sampled_energies_J for block in blocks]
        ),
        physical_volume_attempts=sum(
            block.physical_volume_attempts for block in blocks
        ),
        physical_volume_acceptances=sum(
            block.physical_volume_acceptances for block in blocks
        ),
        physical_delta_energy_over_kbt=np.concatenate(
            [block.physical_delta_energy_over_kbt for block in blocks]
        ),
        physical_pressure_work_over_kbt=np.concatenate(
            [block.physical_pressure_work_over_kbt for block in blocks]
        ),
        physical_jacobian_log_weight=np.concatenate(
            [block.physical_jacobian_log_weight for block in blocks]
        ),
        physical_log_acceptance_probabilities=np.concatenate(
            [block.physical_log_acceptance_probabilities for block in blocks]
        ),
        cycle_count=sum(block.cycle_count for block in blocks),
        force_evaluation_count=sum(block.force_evaluation_count for block in blocks),
        hmc_expected_acceptance_by_cycle_and_state=np.concatenate(
            [block.hmc_expected_acceptance_by_cycle_and_state for block in blocks]
        ),
        hmc_realized_acceptance_by_cycle_and_state=np.concatenate(
            [block.hmc_realized_acceptance_by_cycle_and_state for block in blocks]
        ),
        hmc_absolute_energy_error_over_kbt_by_cycle_and_state=np.concatenate(
            [
                block.hmc_absolute_energy_error_over_kbt_by_cycle_and_state
                for block in blocks
            ]
        ),
        hmc_molecular_com_squared_displacement_m2_by_cycle_and_state=np.concatenate(
            [
                block.hmc_molecular_com_squared_displacement_m2_by_cycle_and_state
                for block in blocks
            ]
        ),
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
        f"volume_acceptance={block.physical_volume_acceptances}/"
        f"{block.physical_volume_attempts} "
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
    normalized_block_runtime_s = seconds_per_cycle * settings.block_cycle_count
    if normalized_block_runtime_s > settings.block_runtime_limit_s:
        raise RuntimeError(
            f"{stage} fixed-shape block failed the runtime preflight: "
            f"normalized_block_s={normalized_block_runtime_s:.12g}, "
            f"seconds_per_force={seconds_per_force_evaluation:.12g}, "
            f"allowed_block_s={settings.block_runtime_limit_s:.12g}"
        )


def _adapt_volume_and_run_npt_pilot(
    model: AnalyticalPeriodicInteratomicModel,
    conditioning_state: IonicHrexState,
    settings: IonicHrexSettings,
    temperature_K: float,
    pressure_Pa: float,
    selected_step_sizes_s: Array,
    hrex_start_s: float,
) -> tuple[IonicHrexState, IonicHrexBlock, int]:
    state = copy.deepcopy(conditioning_state)
    replica_count = len(settings.lambdas)
    state.hmc_step_sizes_s = np.tile(
        selected_step_sizes_s,
        settings.independent_ladder_count,
    )
    remaining_warmup_cycle_count = (
        settings.warmup_cycle_count - HMC_CALIBRATION_WINDOW_CYCLE_COUNT
    )
    warmup_block_count = math.ceil(
        remaining_warmup_cycle_count / settings.block_cycle_count
    )
    adaptive_blocks: list[IonicHrexBlock] = []
    for warmup_block_index in range(warmup_block_count):
        block_start_s = time.perf_counter()
        state, warmup_subblock = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            cycle_count=min(
                settings.block_cycle_count,
                remaining_warmup_cycle_count
                - warmup_block_index * settings.block_cycle_count,
            ),
            sample_volume=True,
            attempt_exchange=False,
            adapt_volume=True,
            retain_samples=False,
        )
        _report_hrex_block(
            stage="npt-volume-adaptation",
            block_index=warmup_block_index + 1,
            block_elapsed_s=time.perf_counter() - block_start_s,
            total_elapsed_s=time.perf_counter() - hrex_start_s,
            state=state,
            settings=settings,
            block=warmup_subblock,
        )
        adaptive_blocks.append(warmup_subblock)
    warmup_block = _concatenate_ionic_hrex_blocks(adaptive_blocks)
    physical_scale_indices = (
        np.arange(settings.independent_ladder_count) * replica_count
    )
    _validate_physical_barostat_warmup(
        block=warmup_block,
        physical_proposal_scales=state.log_volume_proposal_scales[
            physical_scale_indices
        ],
    )
    _reset_hrex_round_trip_tracking(
        state=state,
        replica_count=replica_count,
        reset_walker_identifiers=True,
    )
    state.hmc_attempts.fill(0)
    state.hmc_acceptances.fill(0)
    state.hmc_expected_acceptance_sums.fill(0.0)
    state.hmc_absolute_energy_error_over_kbt_sums.fill(0.0)
    state.hmc_molecular_com_squared_displacement_sums_m2.fill(0.0)
    state.exchange_attempts.fill(0)
    state.exchange_acceptances.fill(0)
    state.exchange_expected_acceptance_sums.fill(0.0)
    state.volume_attempts.fill(0)
    state.volume_acceptances.fill(0)
    pilot_start_s = time.perf_counter()
    state, fixed_kernel_pilot = advance_ionic_hrex(
        model=model,
        state=state,
        settings=settings,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        cycle_count=HMC_COMBINED_PILOT_CYCLE_COUNT,
        sample_volume=True,
        attempt_exchange=True,
        adapt_volume=False,
        retain_samples=True,
    )
    _report_hrex_block(
        stage="npt-fixed-pilot",
        block_index=1,
        block_elapsed_s=time.perf_counter() - pilot_start_s,
        total_elapsed_s=time.perf_counter() - hrex_start_s,
        state=state,
        settings=settings,
        block=fixed_kernel_pilot,
    )
    pilot_volumes_m3 = fixed_kernel_pilot.sampled_volumes_m3
    pilot_midpoint = pilot_volumes_m3.size // 2
    first_half_volume_m3 = float(np.mean(pilot_volumes_m3[:pilot_midpoint]))
    second_half_volume_m3 = float(np.mean(pilot_volumes_m3[pilot_midpoint:]))
    relative_directed_volume_change = abs(
        second_half_volume_m3 - first_half_volume_m3
    ) / first_half_volume_m3
    if (
        relative_directed_volume_change
        > model.numerics.pressure_volume_relative_tolerance
    ):
        raise ValueError(
            "pressure-preconditioned NPT pilot retained a directed volume change: "
            f"relative_change={relative_directed_volume_change:.12g}, "
            f"tolerance={model.numerics.pressure_volume_relative_tolerance:.12g}"
        )
    return state, fixed_kernel_pilot, warmup_block_count


def _sample_fixed_ionic_hrex_npt(
    model: AnalyticalPeriodicInteratomicModel,
    temperature_K: float,
    pressure_Pa: float,
    settings: IonicHrexSettings,
    maximum_production_block_count: int,
    random_seed: int,
    initial_positions_by_ladder_m: Array,
    initial_box_vectors_by_ladder_m: Array,
) -> NptEquilibriumResult:
    state = initialize_ionic_hrex_state(
        model=model,
        settings=settings,
        random_seed=random_seed,
        initial_positions_by_ladder_m=initial_positions_by_ladder_m,
        initial_box_vectors_by_ladder_m=initial_box_vectors_by_ladder_m,
    )
    replica_lambda_values = np.tile(
        np.asarray(settings.lambdas), settings.independent_ladder_count
    )
    model.energy_force_components_batch(
        positions_batch_m=state.positions_m,
        box_vectors_batch_m=state.boxes_m,
        lambda_values=replica_lambda_values,
    )
    hrex_start_s = time.perf_counter()
    conditioning_start_s = time.perf_counter()
    state, conditioning_block = advance_ionic_hrex(
        model=model,
        state=state,
        settings=settings,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        cycle_count=HMC_CALIBRATION_WINDOW_CYCLE_COUNT,
        sample_volume=False,
        attempt_exchange=False,
        adapt_volume=False,
        retain_samples=False,
    )
    _report_hrex_block(
        stage="npt-hmc-conditioning",
        block_index=1,
        block_elapsed_s=time.perf_counter() - conditioning_start_s,
        total_elapsed_s=time.perf_counter() - hrex_start_s,
        state=state,
        settings=settings,
        block=conditioning_block,
    )
    conditioning_state = copy.deepcopy(state)
    validation_state = copy.deepcopy(conditioning_state)
    validation_random_generator = np.random.default_rng(
        np.random.SeedSequence(random_seed).spawn(1)[0]
    )
    validation_state.random_generator_state = copy.deepcopy(
        validation_random_generator.bit_generator.state
    )
    validation_state, _validation_conditioning_block = advance_ionic_hrex(
        model=model,
        state=validation_state,
        settings=settings,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        cycle_count=HMC_CALIBRATION_WINDOW_CYCLE_COUNT,
        sample_volume=False,
        attempt_exchange=False,
        adapt_volume=False,
        retain_samples=False,
    )
    replica_count = len(settings.lambdas)
    high_acceptance_step_sizes_s = np.zeros(replica_count)
    low_acceptance_step_sizes_s = np.full(replica_count, np.inf)
    physical_scale_indices = (
        np.arange(settings.independent_ladder_count) * replica_count
    )
    starting_step_sizes_s = np.full(replica_count, settings.hmc_step_size_s)
    for pilot_calibration_round in range(
        HMC_MAXIMUM_COMBINED_PILOT_RETRY_COUNT + 1
    ):
        (
            selected_step_sizes_s,
            high_acceptance_step_sizes_s,
            low_acceptance_step_sizes_s,
        ) = _calibrate_hmc_step_sizes_s(
            model=model,
            snapshot_state=conditioning_state,
            validation_state=validation_state,
            settings=settings,
            temperature_K=temperature_K,
            starting_step_sizes_s=starting_step_sizes_s,
            high_acceptance_step_sizes_s=high_acceptance_step_sizes_s,
            low_acceptance_step_sizes_s=low_acceptance_step_sizes_s,
        )
        state, fixed_kernel_pilot, warmup_block_count = (
            _adapt_volume_and_run_npt_pilot(
                model=model,
                conditioning_state=conditioning_state,
                settings=settings,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                selected_step_sizes_s=selected_step_sizes_s,
                hrex_start_s=hrex_start_s,
            )
        )
        expected_by_cycle_ladder_lambda = (
            fixed_kernel_pilot.hmc_expected_acceptance_by_cycle_and_state.reshape(
                fixed_kernel_pilot.cycle_count,
                settings.independent_ladder_count,
                replica_count,
            )
        )
        realized_by_cycle_ladder_lambda = (
            fixed_kernel_pilot.hmc_realized_acceptance_by_cycle_and_state.reshape(
                fixed_kernel_pilot.cycle_count,
                settings.independent_ladder_count,
                replica_count,
            )
        )
        pilot_passed = (
            fixed_kernel_pilot.physical_volume_attempts > 0
            and fixed_kernel_pilot.physical_volume_acceptances > 0
        )
        pilot_expected_acceptance_by_lambda: list[float] = []
        pilot_numerically_valid_by_lambda: list[bool] = []
        for replica_index, lambda_value in enumerate(settings.lambdas):
            lambda_expected = expected_by_cycle_ladder_lambda[:, :, replica_index]
            ladder_means = np.mean(lambda_expected, axis=0)
            expected_mean = float(np.mean(ladder_means))
            expected_standard_error = float(
                np.std(ladder_means, ddof=1) / math.sqrt(ladder_means.size)
            )
            pilot_expected_acceptance_by_lambda.append(expected_mean)
            lambda_energy_errors = (
                fixed_kernel_pilot.hmc_absolute_energy_error_over_kbt_by_cycle_and_state.reshape(
                    fixed_kernel_pilot.cycle_count,
                    settings.independent_ladder_count,
                    replica_count,
                )[:, :, replica_index]
            )
            lambda_displacements_m2 = (
                fixed_kernel_pilot.hmc_molecular_com_squared_displacement_m2_by_cycle_and_state.reshape(
                    fixed_kernel_pilot.cycle_count,
                    settings.independent_ladder_count,
                    replica_count,
                )[:, :, replica_index]
            )
            numerically_valid = bool(
                np.all(np.isfinite(lambda_expected))
                and np.all(np.isfinite(lambda_energy_errors))
                and float(np.max(lambda_energy_errors))
                <= -math.log(np.finfo(float).tiny)
            )
            pilot_numerically_valid_by_lambda.append(numerically_valid)
            if (
                not numerically_valid
                or expected_mean < settings.hmc_target_acceptance_minimum
            ):
                pilot_passed = False
            print(
                f"[HMC pilot] round={pilot_calibration_round + 1} "
                f"lambda={lambda_value:.12g} "
                f"timestep_s={selected_step_sizes_s[replica_index]:.12g} "
                f"raw_proposals={lambda_expected.size} "
                f"ladder_means={tuple(float(value) for value in ladder_means)} "
                f"expected={expected_mean:.6f} se={expected_standard_error:.6f} "
                f"realized={float(np.mean(realized_by_cycle_ladder_lambda[:, :, replica_index])):.6f} "
                f"rms_com_m={float(np.sqrt(np.mean(lambda_displacements_m2))):.12g} "
                f"median_abs_delta_h_over_kbt={float(np.median(lambda_energy_errors)):.6f} "
                f"max_abs_delta_h_over_kbt={float(np.max(lambda_energy_errors)):.6f} "
                f"force_evaluations={fixed_kernel_pilot.force_evaluation_count}",
                flush=True,
            )
        if pilot_passed:
            break
        for replica_index, expected_mean in enumerate(
            pilot_expected_acceptance_by_lambda
        ):
            if (
                not pilot_numerically_valid_by_lambda[replica_index]
                or expected_mean < settings.hmc_target_acceptance_minimum
            ):
                low_acceptance_step_sizes_s[replica_index] = min(
                    low_acceptance_step_sizes_s[replica_index],
                    selected_step_sizes_s[replica_index],
                )
        starting_step_sizes_s = np.asarray(
            [
                math.sqrt(high_value * low_value)
                if high_value > 0.0 and math.isfinite(low_value)
                else selected_value * settings.hmc_step_size_adaptation_factor
                for selected_value, high_value, low_value in zip(
                    selected_step_sizes_s,
                    high_acceptance_step_sizes_s,
                    low_acceptance_step_sizes_s,
                    strict=True,
                )
            ]
        )
    else:
        raise RuntimeError("fixed HMC pilot exhausted the calibration budget")
    frozen_mixing_blocks = 0
    frozen_burn_in_blocks: list[IonicHrexBlock] = []
    maximum_frozen_burn_in_blocks = maximum_production_block_count
    while frozen_mixing_blocks < maximum_frozen_burn_in_blocks:
        block_start_s = time.perf_counter()
        state, frozen_mixing_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            cycle_count=settings.block_cycle_count,
            sample_volume=True,
            attempt_exchange=True,
            adapt_volume=False,
            retain_samples=True,
        )
        block_elapsed_s = time.perf_counter() - block_start_s
        frozen_mixing_blocks += 1
        _report_hrex_block(
            stage="npt-frozen-burn-in",
            block_index=frozen_mixing_blocks,
            block_elapsed_s=block_elapsed_s,
            total_elapsed_s=time.perf_counter() - hrex_start_s,
            state=state,
            settings=settings,
            block=frozen_mixing_block,
        )
        frozen_burn_in_blocks.append(frozen_mixing_block)
        edge_acceptance_by_ladder = np.divide(
            state.exchange_acceptances,
            state.exchange_attempts,
            out=np.zeros_like(state.exchange_acceptances, dtype=float),
            where=state.exchange_attempts > 0,
        )
        accumulated_burn_in = _concatenate_ionic_hrex_blocks(frozen_burn_in_blocks)
        frozen_observables_stationary = True
        for ladder_index in range(settings.independent_ladder_count):
            ladder_mask = accumulated_burn_in.physical_ladder_indices == ladder_index
            ladder_volumes_m3 = accumulated_burn_in.sampled_volumes_m3[ladder_mask]
            ladder_enthalpies_J = (
                accumulated_burn_in.sampled_energies_J[ladder_mask]
                + pressure_Pa * ladder_volumes_m3
            )
            if ladder_volumes_m3.size < settings.warmup_cycle_count:
                frozen_observables_stationary = False
                continue
            for observable_values in (ladder_volumes_m3, ladder_enthalpies_J):
                if not stationary_suffix_candidates(
                    values=observable_values,
                    maximum_split_mean_difference_standard_errors=(
                        model.numerics.stationarity_standard_error_limit
                    ),
                    maximum_linear_drift_standard_errors=(
                        model.numerics.stationarity_standard_error_limit
                    ),
                    minimum_effective_sample_size=(
                        settings.burn_in_minimum_effective_sample_size
                    ),
                ):
                    frozen_observables_stationary = False
        if frozen_observables_stationary:
            break
    else:
        raise ValueError(
            "frozen HREX-NPT burn-in failed physical-observable stationarity; "
            f"round_trips_by_walker={tuple(map(tuple, state.completed_round_trips))}, "
            f"edge_acceptance_by_ladder={tuple(map(tuple, edge_acceptance_by_ladder))}"
        )
    production_blocks: list[IonicHrexBlock] = []
    cycles_per_block = settings.block_cycle_count
    for _production_block_index in range(maximum_production_block_count):
        block_start_s = time.perf_counter()
        state, production_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            cycle_count=cycles_per_block,
            sample_volume=True,
            attempt_exchange=True,
            adapt_volume=False,
            retain_samples=True,
        )
        block_elapsed_s = time.perf_counter() - block_start_s
        _report_hrex_block(
            stage="npt-production",
            block_index=_production_block_index + 1,
            block_elapsed_s=block_elapsed_s,
            total_elapsed_s=time.perf_counter() - hrex_start_s,
            state=state,
            settings=settings,
            block=production_block,
        )
        production_blocks.append(production_block)
        accumulated_block = _concatenate_ionic_hrex_blocks(production_blocks)
        _validate_physical_barostat_production_block(
            block=accumulated_block,
            physical_proposal_scales=state.log_volume_proposal_scales[
                physical_scale_indices
            ],
        )
        ladder_volume_candidates = []
        ladder_enthalpy_candidates = []
        ladder_volume_series: list[Array] = []
        ladder_enthalpy_series: list[Array] = []
        for ladder_index in range(settings.independent_ladder_count):
            ladder_mask = accumulated_block.physical_ladder_indices == ladder_index
            ladder_volumes_m3 = accumulated_block.sampled_volumes_m3[ladder_mask]
            ladder_energies_J = accumulated_block.sampled_energies_J[ladder_mask]
            ladder_enthalpies_J = ladder_energies_J + pressure_Pa * ladder_volumes_m3
            ladder_volume_series.append(ladder_volumes_m3)
            ladder_enthalpy_series.append(ladder_enthalpies_J)
            ladder_volume_candidates.append(
                stationary_suffix_candidates(
                    values=ladder_volumes_m3,
                    maximum_split_mean_difference_standard_errors=(
                        model.numerics.stationarity_standard_error_limit
                    ),
                    maximum_linear_drift_standard_errors=(
                        model.numerics.stationarity_standard_error_limit
                    ),
                    minimum_effective_sample_size=(
                        settings.burn_in_minimum_effective_sample_size
                    ),
                )
            )
            ladder_enthalpy_candidates.append(
                stationary_suffix_candidates(
                    values=ladder_enthalpies_J,
                    maximum_split_mean_difference_standard_errors=(
                        model.numerics.stationarity_standard_error_limit
                    ),
                    maximum_linear_drift_standard_errors=(
                        model.numerics.stationarity_standard_error_limit
                    ),
                    minimum_effective_sample_size=(
                        settings.burn_in_minimum_effective_sample_size
                    ),
                )
            )
        stationary_candidate_groups = (
            tuple(ladder_volume_candidates)
            + tuple(ladder_enthalpy_candidates)
        )
        if not all(stationary_candidate_groups):
            continue
        common_start_index = max(
            candidate.start_index
            for candidates in stationary_candidate_groups
            for candidate in candidates
        )
        common_volume_chains = np.stack(
            [values[common_start_index:] for values in ladder_volume_series]
        )
        common_enthalpy_chains = np.stack(
            [values[common_start_index:] for values in ladder_enthalpy_series]
        )
        volume_bulk_rhat = _rank_normalized_split_statistics(common_volume_chains)[0]
        volume_folded_rhat = _rank_normalized_split_statistics(
            np.abs(common_volume_chains - np.median(common_volume_chains))
        )[0]
        enthalpy_bulk_rhat = _rank_normalized_split_statistics(common_enthalpy_chains)[
            0
        ]
        enthalpy_folded_rhat = _rank_normalized_split_statistics(
            np.abs(common_enthalpy_chains - np.median(common_enthalpy_chains))
        )[0]
        volume_standard_errors_m3 = np.asarray(
            [
                np.std(values, ddof=1)
                / math.sqrt(
                    autocorrelation_and_effective_sample_size(
                        values
                    ).effective_sample_size
                )
                for values in common_volume_chains
            ]
        )
        pooled_volume_mcse_m3 = float(
            np.sqrt(np.sum(volume_standard_errors_m3**2))
            / settings.independent_ladder_count
        )
        mean_volume_m3 = float(np.mean(common_volume_chains))
        if (
            max(volume_bulk_rhat, volume_folded_rhat)
            <= model.numerics.maximum_split_rhat
            and max(enthalpy_bulk_rhat, enthalpy_folded_rhat)
            <= model.numerics.maximum_split_rhat
            and pooled_volume_mcse_m3
            <= model.numerics.equilibrium_observable_relative_tolerance * mean_volume_m3
        ):
            return NptEquilibriumResult(
                hrex_result=_ionic_hrex_result_from_state_and_block(
                    state=state,
                    block=accumulated_block,
                    settings=settings,
                ),
                terminal_state=state,
                settings=settings,
                stationary_volume_samples_m3=common_volume_chains.reshape(-1),
                stationary_common_start_index=common_start_index,
                equilibrium_volume_m3=mean_volume_m3,
                equilibrium_volume_mcse_m3=pooled_volume_mcse_m3,
            )
    final_block = _concatenate_ionic_hrex_blocks(production_blocks)
    ladder_volume_endpoints_m3 = []
    ladder_energy_endpoints_J = []
    for ladder_index in range(settings.independent_ladder_count):
        ladder_mask = final_block.physical_ladder_indices == ladder_index
        ladder_volumes_m3 = final_block.sampled_volumes_m3[ladder_mask]
        ladder_energies_J = final_block.sampled_energies_J[ladder_mask]
        ladder_volume_endpoints_m3.append(
            (float(ladder_volumes_m3[0]), float(ladder_volumes_m3[-1]))
        )
        ladder_energy_endpoints_J.append(
            (float(ladder_energies_J[0]), float(ladder_energies_J[-1]))
        )
    raise ValueError(
        "adaptive HREX-NPT exhausted production blocks without stationary "
        f"volume, energy, and cross-ladder agreement; volume_stationary="
        f"{tuple(bool(candidates) for candidates in ladder_volume_candidates)}, "
        f"enthalpy_stationary="
        f"{tuple(bool(candidates) for candidates in ladder_enthalpy_candidates)}, "
        f"volume_endpoints_m3={tuple(ladder_volume_endpoints_m3)}, "
        f"energy_endpoints_J={tuple(ladder_energy_endpoints_J)}; "
        f"{_physical_barostat_diagnostic_text(final_block, state.log_volume_proposal_scales[physical_scale_indices])}"
    )


def _continue_ionic_hrex_nvt(
    model: AnalyticalPeriodicInteratomicModel,
    state: IonicHrexState,
    temperature_K: float,
    settings: IonicHrexSettings,
    equilibrium_box_vectors_m: Array,
) -> IonicHrexResult:
    nvt_start_s = time.perf_counter()
    for batch_index in range(state.positions_m.shape[0]):
        state.positions_m[batch_index] = _scale_molecular_centers_to_box(
            positions_m=state.positions_m[batch_index],
            source_box_vectors_m=state.boxes_m[batch_index],
            target_box_vectors_m=equilibrium_box_vectors_m,
            system=model.system,
        )
    state.boxes_m = np.repeat(
        equilibrium_box_vectors_m[None, :, :],
        state.positions_m.shape[0],
        axis=0,
    )
    state.component_energies_J = _component_array_from_result(
        model.energy_components_batch(
            positions_batch_m=state.positions_m,
            box_vectors_batch_m=state.boxes_m,
        )
    )
    state.momentum_refresh_required.fill(True)
    adaptive_nvt_block_count = math.ceil(
        settings.warmup_cycle_count / settings.block_cycle_count
    )
    for adaptive_nvt_block_index in range(adaptive_nvt_block_count):
        block_start_s = time.perf_counter()
        state, adaptive_nvt_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            pressure_Pa=0.0,
            cycle_count=min(
                settings.block_cycle_count,
                settings.warmup_cycle_count
                - adaptive_nvt_block_index * settings.block_cycle_count,
            ),
            sample_volume=False,
            attempt_exchange=True,
            adapt_volume=False,
            retain_samples=False,
        )
        block_elapsed_s = time.perf_counter() - block_start_s
        _report_hrex_block(
            stage="nvt-adaptation",
            block_index=adaptive_nvt_block_index + 1,
            block_elapsed_s=block_elapsed_s,
            total_elapsed_s=time.perf_counter() - nvt_start_s,
            state=state,
            settings=settings,
            block=adaptive_nvt_block,
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
    maximum_nvt_mixing_blocks = math.ceil(
        settings.production_cycle_count / settings.warmup_cycle_count
    )
    nvt_burn_in_blocks: list[IonicHrexBlock] = []
    for _nvt_mixing_block_index in range(maximum_nvt_mixing_blocks):
        block_start_s = time.perf_counter()
        state, equilibration_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            pressure_Pa=0.0,
            cycle_count=settings.block_cycle_count,
            sample_volume=False,
            attempt_exchange=True,
            adapt_volume=False,
            retain_samples=True,
        )
        block_elapsed_s = time.perf_counter() - block_start_s
        _report_hrex_block(
            stage="nvt-frozen-burn-in",
            block_index=_nvt_mixing_block_index + 1,
            block_elapsed_s=block_elapsed_s,
            total_elapsed_s=time.perf_counter() - nvt_start_s,
            state=state,
            settings=settings,
            block=equilibration_block,
        )
        nvt_burn_in_blocks.append(equilibration_block)
        accumulated_nvt_burn_in = _concatenate_ionic_hrex_blocks(nvt_burn_in_blocks)
        edge_acceptance_by_ladder = np.divide(
            state.exchange_acceptances,
            state.exchange_attempts,
            out=np.zeros_like(state.exchange_acceptances, dtype=float),
            where=state.exchange_attempts > 0,
        )
        nvt_coordination_macrostates = ionic_coordination_macrostates(
            positions_batch_m=(accumulated_nvt_burn_in.physical_configurations_m),
            box_vectors_batch_m=(
                accumulated_nvt_burn_in.physical_box_vectors_by_sample_m
            ),
            system=model.system,
        )
        nvt_observables_stationary = True
        for ladder_index in range(settings.independent_ladder_count):
            ladder_mask = (
                accumulated_nvt_burn_in.physical_ladder_indices == ladder_index
            )
            ladder_energies_J = accumulated_nvt_burn_in.sampled_energies_J[ladder_mask]
            if ladder_energies_J.size < settings.warmup_cycle_count:
                nvt_observables_stationary = False
                continue
            diagnostic_series = [ladder_energies_J]
            ladder_coordination = nvt_coordination_macrostates[ladder_mask]
            diagnostic_series.extend(
                ladder_coordination[:, feature_index]
                for feature_index in range(ladder_coordination.shape[1])
                if np.var(ladder_coordination[:, feature_index]) > 0.0
            )
            for observable_values in diagnostic_series:
                if not stationary_suffix_candidates(
                    values=observable_values,
                    maximum_split_mean_difference_standard_errors=(
                        model.numerics.stationarity_standard_error_limit
                    ),
                    maximum_linear_drift_standard_errors=(
                        model.numerics.stationarity_standard_error_limit
                    ),
                    minimum_effective_sample_size=(
                        settings.burn_in_minimum_effective_sample_size
                    ),
                ):
                    nvt_observables_stationary = False
        if nvt_observables_stationary:
            break
    else:
        raise ValueError(
            "frozen HREX-NVT burn-in failed physical-observable stationarity; "
            f"round_trips_by_walker={tuple(map(tuple, state.completed_round_trips))}, "
            f"edge_acceptance_by_ladder={tuple(map(tuple, edge_acceptance_by_ladder))}"
        )
    nvt_production_blocks: list[IonicHrexBlock] = []
    retained_cycle_count = 0
    while retained_cycle_count < settings.production_cycle_count:
        cycle_count = min(
            settings.block_cycle_count,
            settings.production_cycle_count - retained_cycle_count,
        )
        block_start_s = time.perf_counter()
        state, production_block = advance_ionic_hrex(
            model=model,
            state=state,
            settings=settings,
            temperature_K=temperature_K,
            pressure_Pa=0.0,
            cycle_count=cycle_count,
            sample_volume=False,
            attempt_exchange=True,
            adapt_volume=False,
            retain_samples=True,
        )
        block_elapsed_s = time.perf_counter() - block_start_s
        nvt_production_blocks.append(production_block)
        retained_cycle_count += cycle_count
        _report_hrex_block(
            stage="nvt-production",
            block_index=len(nvt_production_blocks),
            block_elapsed_s=block_elapsed_s,
            total_elapsed_s=time.perf_counter() - nvt_start_s,
            state=state,
            settings=settings,
            block=production_block,
        )
    production_block = _concatenate_ionic_hrex_blocks(nvt_production_blocks)
    return _ionic_hrex_result_from_state_and_block(
        state=state,
        block=production_block,
        settings=settings,
    )


def sample_isothermal_isobaric_equilibrium(
    model: AnalyticalPeriodicInteratomicModel,
    initial_positions_by_ladder_m: Array,
    initial_box_vectors_by_ladder_m: Array,
    temperature_K: float,
    pressure_Pa: float,
    dynamics: DynamicsSettings,
    random_seed: int,
) -> tuple[AnalyticalPeriodicInteratomicModel, Array, Array, float, Array]:
    pressure_preconditioning = precondition_equilibrium_volume(
        model=model,
        positions_by_ladder_m=initial_positions_by_ladder_m,
        box_vectors_by_ladder_m=initial_box_vectors_by_ladder_m,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        dynamics=dynamics,
        random_seed=random_seed,
    )
    hrex_settings = IonicHrexSettings(
        lambdas=dynamics.ionic_hrex_lambdas,
        hmc_step_size_s=dynamics.hamiltonian_timestep_s,
        hmc_steps_min=dynamics.hmc_steps_min,
        hmc_steps_max=dynamics.hmc_steps_max,
        hmc_momentum_persistence=dynamics.hmc_momentum_persistence,
        hmc_full_refresh_stride=dynamics.hmc_full_refresh_stride,
        exchange_stride=dynamics.exchange_stride,
        volume_move_stride=dynamics.volume_move_stride,
        independent_ladder_count=dynamics.equilibrium_chain_count,
        minimum_round_trips=dynamics.minimum_round_trips,
        warmup_cycle_count=dynamics.hrex_warmup_cycle_count,
        production_cycle_count=(
            dynamics.equilibrium_sample_count * dynamics.hrex_measurement_stride
        ),
        measurement_stride=dynamics.hrex_measurement_stride,
        block_cycle_count=dynamics.hrex_block_cycle_count,
        block_runtime_limit_s=dynamics.hrex_block_runtime_limit_s,
        burn_in_minimum_effective_sample_size=(
            dynamics.burn_in_minimum_effective_sample_size
        ),
        logarithmic_volume_proposal=dynamics.logarithmic_volume_proposal,
        hmc_target_acceptance_minimum=dynamics.hmc_target_acceptance_minimum,
        hmc_target_acceptance_maximum=dynamics.hmc_target_acceptance_maximum,
        hmc_step_size_adaptation_factor=(dynamics.hmc_step_size_adaptation_factor),
        hmc_log_bracket_width_tolerance=(
            dynamics.hmc_log_bracket_width_tolerance
        ),
        volume_target_acceptance=dynamics.volume_target_acceptance,
        volume_adaptation_gain=dynamics.volume_adaptation_gain,
        minimum_log_volume_proposal=dynamics.minimum_log_volume_proposal,
        maximum_log_volume_proposal=dynamics.maximum_log_volume_proposal,
    )
    npt_equilibrium = _sample_fixed_ionic_hrex_npt(
        model=model,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        settings=hrex_settings,
        maximum_production_block_count=(
            dynamics.equilibrium_maximum_refinement_batches
        ),
        random_seed=random_seed,
        initial_positions_by_ladder_m=(
            pressure_preconditioning.positions_by_ladder_m
        ),
        initial_box_vectors_by_ladder_m=(
            pressure_preconditioning.box_vectors_by_ladder_m
        ),
    )
    npt_result = npt_equilibrium.hrex_result
    npt_state = npt_equilibrium.terminal_state
    hrex_settings = npt_equilibrium.settings
    equilibrium_volume_m3 = npt_equilibrium.equilibrium_volume_m3
    equilibrium_length_m = equilibrium_volume_m3 ** (1.0 / CARTESIAN_DIMENSION)
    equilibrium_box_vectors_m = np.eye(CARTESIAN_DIMENSION) * equilibrium_length_m
    nvt_settings = replace(
        hrex_settings,
        production_cycle_count=max(
            dynamics.hrex_production_cycle_count,
            dynamics.equilibrium_sample_count
            * dynamics.equilibrium_maximum_refinement_batches
            * dynamics.hrex_measurement_stride,
        ),
    )
    nvt_result = _continue_ionic_hrex_nvt(
        model=model,
        state=npt_state,
        temperature_K=temperature_K,
        settings=nvt_settings,
        equilibrium_box_vectors_m=equilibrium_box_vectors_m,
    )
    equilibrium_system = replace(
        model.system,
        positions_m=nvt_result.physical_configurations_m[-1],
        box_vectors_m=equilibrium_box_vectors_m,
    )
    energy_effective_sample_size = float(
        sum(
            autocorrelation_and_effective_sample_size(
                npt_result.sampled_energies_J[
                    npt_result.physical_ladder_indices == ladder_index
                ]
            ).effective_sample_size
            for ladder_index in range(dynamics.equilibrium_chain_count)
        )
    )
    return (
        AnalyticalPeriodicInteratomicModel(equilibrium_system, model.numerics),
        nvt_result.physical_configurations_m,
        npt_equilibrium.stationary_volume_samples_m3,
        energy_effective_sample_size,
        nvt_result.physical_ladder_indices,
    )


def relax_initial_configuration(
    model: AnalyticalPeriodicInteratomicModel,
    temperature_K: float,
    dynamics: DynamicsSettings,
) -> RelaxationResult:
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
                * dynamics.initial_force_tolerance_N
                * coordinate_scale_m
                / energy_scale_J
            ),
            "ftol": np.finfo(float).eps,
        },
    )
    positions = (optimization.x.reshape((-1, 3)) * coordinate_scale_m) % np.diag(
        model.system.box_vectors_m
    )
    if not np.isfinite(optimization.fun):
        raise ValueError("initial molecular relaxation produced nonfinite energy")
    final_forces_N = model.forces_N(positions, model.system.box_vectors_m)
    if not np.all(np.isfinite(final_forces_N)):
        raise ValueError("initial molecular relaxation produced nonfinite final forces")
    maximum_force_N = float(np.max(np.linalg.norm(final_forces_N, axis=1)))
    if maximum_force_N > dynamics.initial_force_tolerance_N:
        raise ValueError(
            "initial molecular relaxation maximum force exceeds tolerance: "
            f"maximum_force_N={maximum_force_N:.12g}, "
            f"tolerance_N={dynamics.initial_force_tolerance_N:.12g}, "
            f"iterations={optimization.nit}, optimizer_status={optimization.status}, "
            f"optimizer_message={optimization.message}"
        )
    if not optimization.success:
        print(
            "[relaxation] optimizer stopped after satisfying the force tolerance: "
            f"status={optimization.status}, message={optimization.message}",
            flush=True,
        )
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
    return RelaxationResult(
        positions_m=positions,
        maximum_force_N=maximum_force_N,
        iteration_count=int(optimization.nit),
    )


def _basis_values_tensor(
    positions_m: torch.Tensor,
    system: MolecularSystem,
    numerics: NumericalSettings,
    basis_level: int,
) -> torch.Tensor:
    if basis_level < 1:
        raise ValueError("basis_level must be at least one")
    radial_count = min(numerics.basis_radial_count, basis_level)
    angular_order_limit = min(numerics.basis_angular_order, basis_level)
    cluster_depth_limit = min(numerics.basis_cluster_depth, basis_level)
    fourier_shell_limit = min(numerics.basis_fourier_shell, basis_level)
    correlation_order_limit = min(numerics.basis_correlation_order, basis_level)
    box = torch.as_tensor(system.box_vectors_m)
    displacement = _torch_minimum_image(
        positions_m[:, None, :] - positions_m[None, :, :], box
    )
    distance = torch.linalg.norm(
        displacement + torch.eye(positions_m.shape[0])[:, :, None], dim=2
    )
    pair_i, pair_j = np.triu_indices(positions_m.shape[0], 1)
    pair_distance = distance[torch.as_tensor(pair_i), torch.as_tensor(pair_j)]
    pair_charge = torch.as_tensor(system.charges_C[pair_i] * system.charges_C[pair_j])
    leveled_primitive_features: list[tuple[int, torch.Tensor]] = []
    charges = torch.as_tensor(system.charges_C)
    masses = torch.as_tensor(system.masses_kg)
    total_internal_polarization = torch.zeros(CARTESIAN_DIMENSION, dtype=TORCH_DTYPE)
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
        total_internal_polarization = total_internal_polarization + torch.sum(
            charges[molecule_indices, None] * (local_positions - center_of_mass),
            dim=0,
        )
    leveled_primitive_features.extend(
        (1, component) for component in torch.unbind(total_internal_polarization)
    )
    cutoff = 0.5 * (
        torch.cos(
            math.pi
            * torch.clamp(pair_distance / numerics.basis_radial_cutoff_m, 0.0, 1.0)
        )
        + 1.0
    )
    cutoff = cutoff * (pair_distance < numerics.basis_radial_cutoff_m)
    for radial_mode_index in range(1, radial_count + 1):
        radial = (
            torch.cos(
                radial_mode_index
                * math.pi
                * pair_distance
                / numerics.basis_radial_cutoff_m
            )
            * cutoff
        )
        leveled_primitive_features.extend(
            (
                (radial_mode_index, torch.sum(radial)),
                (radial_mode_index, torch.sum(pair_charge * radial)),
            )
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
    pair_directions = (
        center_displacements[center_pair_i_tensor, center_pair_j_tensor]
        / center_distances[center_pair_i_tensor, center_pair_j_tensor, None]
    )
    pair_axis_projection = torch.sum(
        axis_tensor[center_pair_i_tensor] * pair_directions, dim=1
    )
    pair_axis_alignment = torch.sum(
        axis_tensor[center_pair_i_tensor] * axis_tensor[center_pair_j_tensor],
        dim=1,
    )
    for angular_order in range(1, angular_order_limit + 1):
        leveled_primitive_features.extend(
            (
                (
                    angular_order,
                    torch.sum(pair_axis_projection**angular_order),
                ),
                (
                    angular_order,
                    torch.sum(pair_axis_alignment**angular_order),
                ),
            )
        )
    cluster_adjacency = torch.exp(
        -((center_distances / numerics.basis_radial_cutoff_m) ** 2)
    ) * (1.0 - torch.eye(len(molecule_centers), dtype=TORCH_DTYPE))
    cluster_power = cluster_adjacency
    for cluster_depth in range(1, cluster_depth_limit + 1):
        leveled_primitive_features.extend(
            (
                (cluster_depth, torch.trace(cluster_power)),
                (
                    cluster_depth,
                    molecular_charge_tensor @ cluster_power @ molecular_charge_tensor,
                ),
            )
        )
        cluster_power = cluster_power @ cluster_adjacency
    reciprocal_base = 2.0 * math.pi * torch.linalg.inv(box)
    for shell in range(1, fourier_shell_limit + 1):
        for axis in range(3):
            reciprocal = shell * reciprocal_base[axis]
            phases = positions_m @ reciprocal
            leveled_primitive_features.extend(
                (
                    (shell, torch.sum(torch.cos(phases))),
                    (shell, torch.sum(torch.sin(phases))),
                    (shell, torch.sum(charges * torch.cos(phases))),
                    (shell, torch.sum(charges * torch.sin(phases))),
                )
            )
    features: list[torch.Tensor] = []
    candidate_limit = 2 * numerics.maximum_basis_size + basis_level
    for level in range(1, basis_level + 1):
        for feature_level, feature in leveled_primitive_features:
            if feature_level == level and len(features) < candidate_limit:
                features.append(feature)
        if len(features) >= candidate_limit:
            break
        if 2 <= level <= correlation_order_limit:
            available_primitives = [
                feature
                for feature_level, feature in leveled_primitive_features
                if feature_level <= level
            ]
            for feature_indices in combinations_with_replacement(
                range(len(available_primitives)), level
            ):
                product = torch.ones((), dtype=TORCH_DTYPE)
                for feature_index in feature_indices:
                    product = product * available_primitives[feature_index]
                features.append(product)
                if len(features) >= candidate_limit:
                    break
    return torch.stack(features)


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
    configurations_m: Array,
    equilibrium_chain_indices: Array,
    equilibrium_samples_per_batch: int,
    equilibrium_maximum_refinement_batches: int,
    system: MolecularSystem,
    temperature_K: float,
    molecular_memory: MolecularMemoryOperator,
    numerics: NumericalSettings,
) -> tuple[
    float,
    float,
    tuple[float, ...],
    tuple[float, ...],
    int,
    float,
    float,
    float,
]:
    chain_indices = np.asarray(equilibrium_chain_indices, dtype=int)
    if chain_indices.shape != (configurations_m.shape[0],):
        raise ValueError("equilibrium chain indices must identify every configuration")
    unique_chain_indices = np.unique(chain_indices)
    if unique_chain_indices.size < 2:
        raise ValueError("projection requires at least two independent chains")
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
    chain_sample_indices = tuple(
        np.flatnonzero(chain_indices == chain_index)
        for chain_index in unique_chain_indices
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
    admitted_refinement_batch = 0
    final_relative_disagreement = math.inf
    operator_effective_sample_size = 0.0
    maximum_split_rhat = math.inf
    projection_throughput_checked = False
    available_samples_per_chain = tuple(
        int(indices.size) for indices in chain_sample_indices
    )
    diagnostic_pilot_sample_count = max(
        MINIMUM_OPERATOR_DIAGNOSTIC_PILOT_SAMPLES,
        min(available_samples_per_chain) // 4,
    )
    if (
        diagnostic_pilot_sample_count + MINIMUM_OPERATOR_DIAGNOSTIC_EVALUATION_SAMPLES
        > min(available_samples_per_chain)
    ):
        raise ValueError(
            "operator diagnostics require eight pilot and four evaluation samples"
        )
    diagnostic_bases: list[OperatorDiagnosticBasis] = []
    mode_diagnostics: tuple[OperatorModeDiagnostic, ...] = ()
    for refinement_batch in range(1, equilibrium_maximum_refinement_batches + 1):
        samples_per_chain = equilibrium_samples_per_batch * refinement_batch
        if any(indices.size < samples_per_chain for indices in chain_sample_indices):
            break
        batch_start = equilibrium_samples_per_batch * (refinement_batch - 1)
        for chain_position, indices in enumerate(chain_sample_indices):
            batch_indices = indices[batch_start:samples_per_chain]
            molecular_diffusions_m2_s = np.stack(
                tuple(
                    configuration_conditioned_molecular_diffusion(
                        positions_m=configurations_m[sample_index],
                        system=system,
                        molecular_memory=molecular_memory,
                    )
                    for sample_index in batch_indices
                )
            )
            diffusion_eigenvalues, diffusion_eigenvectors = np.linalg.eigh(
                molecular_diffusions_m2_s
            )
            diffusion_scales = np.maximum(
                np.max(diffusion_eigenvalues, axis=1), np.finfo(float).tiny
            )
            retained_diffusion_modes = diffusion_eigenvalues > (
                numerics.eigenvalue_relative_tolerance * diffusion_scales[:, None]
            )
            diffusion_square_roots = (
                diffusion_eigenvectors
                * np.sqrt(
                    np.where(retained_diffusion_modes, diffusion_eigenvalues, 0.0)
                )[:, None, :]
            )
            (
                batch_dirichlet,
                batch_coupling,
                batch_direct,
                batch_diagnostics,
                batch_complete_operators,
            ) = galerkin_assembler.assemble_batch(
                configurations_m=configurations_m[batch_indices],
                diffusion_square_roots=diffusion_square_roots,
                polarization_gradients=polarization_gradients,
            )
            if not projection_throughput_checked:
                throughput_start_s = time.perf_counter()
                galerkin_assembler.assemble_batch(
                    configurations_m=configurations_m[batch_indices],
                    diffusion_square_roots=diffusion_square_roots,
                    polarization_gradients=polarization_gradients,
                )
                throughput_elapsed_s = time.perf_counter() - throughput_start_s
                configurations_per_s = batch_indices.size / throughput_elapsed_s
                maximum_projection_configuration_count = sum(
                    available_samples_per_chain
                )
                projected_runtime_s = (
                    maximum_projection_configuration_count / configurations_per_s
                )
                if projected_runtime_s > PROJECTION_SANITY_RUNTIME_S:
                    raise RuntimeError(
                        "compiled Galerkin projection failed the one-minute runtime "
                        "sanity gate: "
                        f"configurations_per_s={configurations_per_s:.12g}, "
                        f"projected_runtime_s={projected_runtime_s:.12g}, "
                        "allowed_runtime_s="
                        f"{PROJECTION_SANITY_RUNTIME_S:.12g}"
                    )
                projection_throughput_checked = True
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
            chain_operator_blocks[chain_position].append(
                (
                    batch_dirichlet,
                    batch_coupling,
                    batch_direct,
                    int(batch_indices.size),
                )
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
        final_relative_disagreement = maximum_relative_disagreement
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
        if not diagnostic_bases:
            continue
        diagnostic_basis = diagnostic_bases[0]
        diagnostic_operator_chains = (
            current_operator_chains[:, diagnostic_pilot_sample_count:]
            - diagnostic_basis.mean
        ) @ diagnostic_basis.loadings
        if diagnostic_operator_chains.shape[1] < 4:
            continue
        operator_effective_sample_size = multivariate_batch_means_effective_sample_size(
            diagnostic_operator_chains
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
            chain_complete_operator_series=current_complete_operator_chains[
                :, diagnostic_pilot_sample_count:
            ],
            basis_count=basis_count,
            temperature_K=temperature_K,
            volume_m3=abs(np.linalg.det(system.box_vectors_m)),
            eigenvalue_relative_tolerance=numerics.eigenvalue_relative_tolerance,
        )
        if (
            maximum_relative_disagreement
            <= numerics.equilibrium_observable_relative_tolerance
            and operator_effective_sample_size >= numerics.minimum_effective_sample_size
            and maximum_split_rhat <= numerics.maximum_split_rhat
            and max(
                influence_diagnostic.bulk_rhat,
                influence_diagnostic.folded_rhat,
            )
            <= numerics.maximum_split_rhat
            and influence_diagnostic.effective_sample_size
            >= numerics.minimum_effective_sample_size
        ):
            admitted_chain_statistics = chain_statistics
            admitted_samples_per_chain = samples_per_chain
            admitted_refinement_batch = refinement_batch
    if not admitted_chain_statistics:
        worst_mode_text = "unavailable"
        if mode_diagnostics:
            worst_mode = mode_diagnostics[0]
            worst_mode_rhat = max(worst_mode.bulk_rhat, worst_mode.folded_rhat)
            for candidate_mode in mode_diagnostics[1:]:
                candidate_rhat = max(
                    candidate_mode.bulk_rhat, candidate_mode.folded_rhat
                )
                if candidate_rhat > worst_mode_rhat:
                    worst_mode = candidate_mode
                    worst_mode_rhat = candidate_rhat
            diagnostic_configuration_indices = np.concatenate(
                tuple(
                    indices[
                        diagnostic_pilot_sample_count : current_operator_chains.shape[1]
                    ]
                    for indices in chain_sample_indices
                )
            )
            coordination_series = species_coordination_observable_series(
                configurations_m=configurations_m[diagnostic_configuration_indices],
                system=system,
            )
            worst_mode_values = diagnostic_operator_chains[
                :, :, worst_mode.mode_index
            ].reshape(-1)
            coordination_correlations: dict[str, float] = {}
            for species_name, observable_values in coordination_series.items():
                if float(np.std(observable_values)) <= np.finfo(float).tiny:
                    coordination_correlations[species_name] = 0.0
                    continue
                coordination_correlations[species_name] = float(
                    np.corrcoef(worst_mode_values, observable_values)[0, 1]
                )
            worst_mode_text = (
                f"index={worst_mode.mode_index},bulk_rhat={worst_mode.bulk_rhat:.12g},"
                f"folded_rhat={worst_mode.folded_rhat:.12g},"
                f"ess={worst_mode.effective_sample_size:.12g},"
                f"split_means={worst_mode.split_chain_means},"
                f"split_variances={worst_mode.split_chain_variances},"
                f"A_loading_norm={np.linalg.norm(worst_mode.loadings_on_A_diagonal):.12g},"
                f"h_loading_norm={np.linalg.norm(worst_mode.loadings_on_h):.12g},"
                f"direct_loading_norm={np.linalg.norm(worst_mode.loadings_on_direct):.12g}"
                f",influence_bulk_rhat={influence_diagnostic.bulk_rhat:.12g}"
                f",influence_folded_rhat={influence_diagnostic.folded_rhat:.12g}"
                f",influence_ess={influence_diagnostic.effective_sample_size:.12g}"
                f",coordination_correlations={coordination_correlations}"
            )
        raise ValueError(
            "independent equilibrium chains did not stabilize A, h, and direct "
            "conductivity within the configured refinement batches: "
            f"final_relative_disagreement={final_relative_disagreement:.12g}, "
            "allowed_relative_disagreement="
            f"{numerics.equilibrium_observable_relative_tolerance:.12g}, "
            f"available_samples_per_chain={available_samples_per_chain}, "
            f"operator_effective_sample_size={operator_effective_sample_size:.12g}, "
            f"maximum_split_rhat={maximum_split_rhat:.12g}, "
            f"worst_fixed_mode=({worst_mode_text})"
        )
    samples_per_chain = admitted_samples_per_chain

    statistics: list[tuple[Array, Array, Array]] = []
    for chain_statistics in (
        admitted_chain_statistics[::2],
        admitted_chain_statistics[1::2],
    ):
        statistics.append(
            (
                np.mean(
                    np.stack(tuple(values[0] for values in chain_statistics)), axis=0
                ),
                np.mean(
                    np.stack(tuple(values[1] for values in chain_statistics)), axis=0
                ),
                np.mean(
                    np.stack(tuple(values[2] for values in chain_statistics)), axis=0
                ),
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
    heldout_dirichlet = heldout_dirichlet[np.ix_(active_basis, active_basis)] / (
        basis_scales[:, None] * basis_scales[None, :]
    )
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
    while (
        len(remaining_indices) > 1
        and len(selected_indices) < numerics.maximum_basis_size
    ):
        if selected_indices:
            selected_array = np.asarray(selected_indices)
            fit_inverse = symmetric_psd_pseudoinverse(
                fit_dirichlet[np.ix_(selected_array, selected_array)],
                numerics.eigenvalue_relative_tolerance,
            )
            heldout_inverse = symmetric_psd_pseudoinverse(
                heldout_dirichlet[np.ix_(selected_array, selected_array)],
                numerics.eigenvalue_relative_tolerance,
            )
            fit_coefficients = fit_inverse @ fit_coupling[selected_array]
        candidate_records: list[tuple[float, int]] = []
        for candidate_index in remaining_indices:
            fit_energy = float(fit_dirichlet[candidate_index, candidate_index])
            heldout_energy = float(heldout_dirichlet[candidate_index, candidate_index])
            heldout_residual_coupling = heldout_coupling[candidate_index].copy()
            if selected_indices:
                fit_cross = fit_dirichlet[candidate_index, selected_array]
                fit_energy -= float(fit_cross @ fit_inverse @ fit_cross)
                heldout_cross = heldout_dirichlet[candidate_index, selected_array]
                heldout_energy -= float(heldout_cross @ heldout_inverse @ heldout_cross)
                heldout_residual_coupling -= heldout_cross @ fit_coefficients
            if fit_energy <= null_tolerance or heldout_energy <= null_tolerance:
                continue
            score = (
                prefactor
                * float(heldout_residual_coupling @ heldout_residual_coupling)
                / heldout_energy
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
        fit_coefficients = selected_inverse @ fit_coupling[selected_array]
        heldout_selected_inverse = symmetric_psd_pseudoinverse(
            heldout_dirichlet[np.ix_(selected_array, selected_array)],
            numerics.eigenvalue_relative_tolerance,
        )
        validation_scores: list[float] = []
        for validation_index in remaining_indices:
            heldout_cross = heldout_dirichlet[validation_index, selected_array]
            validation_residual_coupling = (
                heldout_coupling[validation_index] - heldout_cross @ fit_coefficients
            )
            validation_energy = float(
                heldout_dirichlet[validation_index, validation_index]
                - heldout_cross @ heldout_selected_inverse @ heldout_cross
            )
            if validation_energy > null_tolerance:
                validation_scores.append(
                    prefactor
                    * float(validation_residual_coupling @ validation_residual_coupling)
                    / validation_energy
                )
        if not validation_scores:
            raise ValueError("untouched basis candidates have zero validation energy")
        maximum_validation_score = max(validation_scores)
        conductivity = direct - correction
        if conductivity > previous + numerics.conductivity_tolerance_S_m:
            raise ValueError("projected conductivity sequence is not monotone")
        history.append(conductivity)
        residuals.append(maximum_validation_score)
        if (
            maximum_validation_score <= numerics.residual_tolerance
            and abs(conductivity - previous) <= numerics.conductivity_tolerance_S_m
        ):
            break
        previous = conductivity
    if not history:
        raise ValueError("basis contains no resolvable nonconstant mode")
    if residuals[-1] > numerics.residual_tolerance or (
        len(history) > 1
        and abs(history[-1] - history[-2]) > numerics.conductivity_tolerance_S_m
    ):
        final_conductivity_change_S_m = abs(history[-1] - direct)
        if len(history) > 1:
            final_conductivity_change_S_m = abs(history[-1] - history[-2])
        raise ValueError(
            "basis hierarchy exhausted before held-out generator residual and "
            "conductivity change converged: "
            f"selected_basis_size={len(selected_indices)}, "
            f"maximum_residual_score={residuals[-1]:.12g}, "
            f"conductivity_change_S_m={final_conductivity_change_S_m:.12g}"
        )
    block_conductivities_S_m: list[float] = []
    admitted_operator_blocks = tuple(
        operator_block
        for chain_blocks in chain_operator_blocks
        for operator_block in chain_blocks[:admitted_refinement_batch]
    )
    for (
        block_dirichlet_sum,
        block_coupling_sum,
        block_direct_sum,
        block_sample_count,
    ) in admitted_operator_blocks:
        block_dirichlet = block_dirichlet_sum / block_sample_count
        block_coupling = block_coupling_sum / block_sample_count
        block_direct = block_direct_sum / block_sample_count
        block_active_dirichlet = block_dirichlet[np.ix_(active_basis, active_basis)]
        block_active_dirichlet /= basis_scales[:, None] * basis_scales[None, :]
        block_active_coupling = block_coupling[active_basis] / basis_scales[:, None]
        block_selected_dirichlet = block_active_dirichlet[
            np.ix_(selected_array, selected_array)
        ]
        block_selected_coupling = block_active_coupling[selected_array]
        block_inverse = symmetric_psd_pseudoinverse(
            block_selected_dirichlet, numerics.eigenvalue_relative_tolerance
        )
        block_correction = prefactor * float(
            np.sum(block_selected_coupling.T @ block_inverse @ block_selected_coupling)
        )
        block_conductivities_S_m.append(
            prefactor * float(np.sum(block_direct)) - block_correction
        )
    if len(block_conductivities_S_m) < 2:
        raise ValueError("conductivity MCSE requires at least two complete blocks")
    conductivity_mcse_S_m = float(
        np.std(block_conductivities_S_m, ddof=1)
        / math.sqrt(len(block_conductivities_S_m))
    )
    if conductivity_mcse_S_m > numerics.conductivity_tolerance_S_m:
        raise ValueError(
            "blocked conductivity MCSE exceeds conductivity tolerance: "
            f"{conductivity_mcse_S_m:.12g} S/m"
        )
    return (
        history[-1],
        direct,
        tuple(history),
        tuple(residuals),
        len(selected_indices),
        operator_effective_sample_size,
        maximum_split_rhat,
        conductivity_mcse_S_m,
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
    minimum_round_trips = int(dynamics_record.pop("minimum_round_trips"))
    values = tuple(dynamics_record.values()) + tuple(asdict(numerics).values())
    if any(float(value) <= 0.0 for value in values):
        raise ValueError(
            "all dynamics settings and numerical settings must be positive"
        )
    if len(hrex_lambdas) < 2 or hrex_lambdas[0] != 1.0:
        raise ValueError("ionic HREX ladder must start at one and contain replicas")
    if minimum_round_trips < 0:
        raise ValueError("minimum HREX round trips cannot be negative")
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
    if not (
        numerics.pressure_log_volume_derivative_check_step
        < numerics.pressure_log_volume_derivative_step
    ):
        raise ValueError(
            "pressure derivative check step must be smaller than the primary step"
        )


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
    seed_sequences = np.random.SeedSequence(random_seed).spawn(
        dynamics.equilibrium_chain_count + 1
    )
    relaxed_systems: list[MolecularSystem] = []
    for ladder_seed_sequence in seed_sequences[:-1]:
        ladder_seed = int(ladder_seed_sequence.generate_state(1)[0])
        ladder_system = build_periodic_molecular_system(
            recipe=recipe,
            molecule_count=molecule_count,
            minimum_interatomic_contact_ratio=(
                numerics.minimum_interatomic_contact_ratio
            ),
            initial_placement_attempts_per_molecule=(
                numerics.initial_placement_attempts_per_molecule
            ),
            random_seed=ladder_seed,
        )
        ladder_model = AnalyticalPeriodicInteratomicModel(ladder_system, numerics)
        relaxation_result = relax_initial_configuration(
            model=ladder_model,
            temperature_K=temperature_K,
            dynamics=dynamics,
        )
        print(
            "[relaxation] "
            f"iterations={relaxation_result.iteration_count} "
            f"maximum_force_N={relaxation_result.maximum_force_N:.12g}",
            flush=True,
        )
        relaxed_systems.append(
            replace(ladder_system, positions_m=relaxation_result.positions_m)
        )
    relaxed_system = relaxed_systems[0]
    interaction_model = AnalyticalPeriodicInteratomicModel(relaxed_system, numerics)
    (
        equilibrium_model,
        configurations,
        _stationary_volumes_m3,
        _equilibrium_energy_effective_sample_size,
        equilibrium_chain_indices,
    ) = sample_isothermal_isobaric_equilibrium(
        model=interaction_model,
        initial_positions_by_ladder_m=np.stack(
            [ladder_system.positions_m for ladder_system in relaxed_systems]
        ),
        initial_box_vectors_by_ladder_m=np.stack(
            [ladder_system.box_vectors_m for ladder_system in relaxed_systems]
        ),
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        dynamics=dynamics,
        random_seed=int(seed_sequences[-1].generate_state(1)[0]),
    )
    equilibrium_system = equilibrium_model.system
    molecular_memory = fit_transferable_molecular_memory_operator(
        system=equilibrium_system,
        temperature_K=temperature_K,
        operator_data_root=(
            Path(__file__).parent / "physical_library" / "lammps_operator_data"
        ),
        eigenvalue_relative_tolerance=numerics.eigenvalue_relative_tolerance,
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
    ) = projected_conductivity_sequence(
        configurations_m=configurations,
        equilibrium_chain_indices=equilibrium_chain_indices,
        equilibrium_samples_per_batch=dynamics.equilibrium_sample_count,
        equilibrium_maximum_refinement_batches=(
            dynamics.equilibrium_maximum_refinement_batches
        ),
        system=equilibrium_system,
        temperature_K=temperature_K,
        molecular_memory=molecular_memory,
        numerics=numerics,
    )
    equilibrium_volume_m3 = float(abs(np.linalg.det(equilibrium_system.box_vectors_m)))
    equilibrium_density_g_cm3 = float(
        np.sum(equilibrium_system.masses_kg) / equilibrium_volume_m3 / KG_M3_PER_G_ML
    )
    return ConductivityResult(
        conductivity_S_m=conductivity,
        direct_current_term_S_m=direct,
        projected_correction_S_m=direct - conductivity,
        equilibrium_volume_m3=equilibrium_volume_m3,
        equilibrium_density_g_cm3=equilibrium_density_g_cm3,
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
        equilibrium_sample_count=configurations.shape[0],
        equilibrium_chain_count=dynamics.equilibrium_chain_count,
        memory_sample_count=molecular_memory.sample_count,
        effective_sample_size=operator_effective_sample_size,
        maximum_split_rhat=maximum_split_rhat,
        conductivity_mcse_S_m=conductivity_mcse_S_m,
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


@dataclass(frozen=True)
class IonicHrexSettings:
    lambdas: tuple[float, ...]
    hmc_step_size_s: float
    hmc_steps_min: int
    hmc_steps_max: int
    hmc_momentum_persistence: float
    hmc_full_refresh_stride: int
    exchange_stride: int
    volume_move_stride: int
    independent_ladder_count: int
    minimum_round_trips: int
    warmup_cycle_count: int
    production_cycle_count: int
    measurement_stride: int
    block_cycle_count: int
    block_runtime_limit_s: float
    burn_in_minimum_effective_sample_size: float
    logarithmic_volume_proposal: float
    hmc_target_acceptance_minimum: float
    hmc_target_acceptance_maximum: float
    hmc_step_size_adaptation_factor: float
    hmc_log_bracket_width_tolerance: float
    volume_target_acceptance: float
    volume_adaptation_gain: float
    minimum_log_volume_proposal: float
    maximum_log_volume_proposal: float


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
    log_volume_proposal_scales: Array
    hmc_attempts: Array
    hmc_acceptances: Array
    hmc_expected_acceptance_sums: Array
    hmc_absolute_energy_error_over_kbt_sums: Array
    hmc_molecular_com_squared_displacement_sums_m2: Array
    exchange_attempts: Array
    exchange_acceptances: Array
    exchange_expected_acceptance_sums: Array
    volume_attempts: Array
    volume_acceptances: Array
    cycle_index: int
    random_generator_state: dict


@dataclass(frozen=True)
class BatchedVolumeTransitionResult:
    positions_m: Array
    boxes_m: Array
    component_energies_J: Array
    accepted: Array
    expected_acceptance_probabilities: Array
    log_volume_changes: Array
    current_volumes_m3: Array
    proposed_volumes_m3: Array
    delta_energy_J: Array
    pressure_work_J: Array
    jacobian_log_weight: Array
    log_acceptance_probabilities: Array


@dataclass(frozen=True)
class IonicHrexBlock:
    physical_configurations_m: Array
    physical_box_vectors_by_sample_m: Array
    physical_ladder_indices: Array
    sampled_volumes_m3: Array
    sampled_energies_J: Array
    physical_volume_attempts: int
    physical_volume_acceptances: int
    physical_delta_energy_over_kbt: Array
    physical_pressure_work_over_kbt: Array
    physical_jacobian_log_weight: Array
    physical_log_acceptance_probabilities: Array
    cycle_count: int
    force_evaluation_count: int
    hmc_expected_acceptance_by_cycle_and_state: Array
    hmc_realized_acceptance_by_cycle_and_state: Array
    hmc_absolute_energy_error_over_kbt_by_cycle_and_state: Array
    hmc_molecular_com_squared_displacement_m2_by_cycle_and_state: Array


class VolumeMoveConfigurationError(RuntimeError):
    """Raised when an NPT block schedules no physical volume proposals."""


class BarostatNotMixingError(RuntimeError):
    """Raised when physical volume proposals have zero acceptance."""


class BarostatProposalCollapsedError(RuntimeError):
    """Raised when accepted NPT samples have no resolvable volume range."""


@dataclass(frozen=True)
class IonicHrexResult:
    physical_configurations_m: Array
    physical_box_vectors_m: Array
    physical_lambda_values: tuple[float, ...]
    hmc_acceptance_fractions: tuple[float, ...]
    exchange_acceptance_fractions: tuple[float, ...]
    round_trip_count: int
    sampled_volumes_m3: Array
    sampled_energies_J: Array
    physical_ladder_indices: Array
    physical_box_vectors_by_sample_m: Array
    volume_attempt_count: int
    volume_acceptance_count: int
    log_volume_proposal_scales: Array
    physical_delta_energy_over_kbt: Array
    physical_pressure_work_over_kbt: Array
    physical_jacobian_log_weight: Array
    physical_log_acceptance_probabilities: Array


@dataclass(frozen=True)
class NptEquilibriumResult:
    hrex_result: IonicHrexResult
    terminal_state: IonicHrexState
    settings: IonicHrexSettings
    stationary_volume_samples_m3: Array
    stationary_common_start_index: int
    equilibrium_volume_m3: float
    equilibrium_volume_mcse_m3: float


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


def _remove_center_of_mass_velocity(velocities_m_s: Array, masses_kg: Array) -> Array:
    center_of_mass_velocity_m_s = np.sum(
        masses_kg[:, None] * velocities_m_s, axis=0
    ) / float(np.sum(masses_kg))
    return velocities_m_s - center_of_mass_velocity_m_s


def ionic_coordination_macrostates(
    positions_batch_m: Array,
    box_vectors_batch_m: Array,
    system: IonicMolecularSystem,
) -> Array:
    library = _physical_library_records()
    lithium_atomic_number = library.species_records["Li+"]["sites"][0]["atomic_number"]
    lithium_atom_indices: list[int] = []
    acceptors_by_species: dict[str, list[int]] = {}
    charged_molecule_indices: list[int] = []
    molecule_charges_C = []
    for molecule_index, (species_name, atom_indices) in enumerate(
        zip(
            system.molecule_species_names,
            system.molecule_atom_indices,
            strict=True,
        )
    ):
        species_record = library.species_records[species_name]
        molecule_charge_C = float(np.sum(system.charges_C[atom_indices]))
        molecule_charges_C.append(molecule_charge_C)
        if not math.isclose(
            molecule_charge_C, 0.0, abs_tol=E_CHARGE * MILP_FEASIBILITY_TOLERANCE
        ):
            charged_molecule_indices.append(molecule_index)
        for atom_index, site_record in zip(
            atom_indices, species_record["sites"], strict=True
        ):
            if int(site_record["atomic_number"]) == lithium_atomic_number:
                lithium_atom_indices.append(int(atom_index))
            if int(site_record["hba_count_contribution"]) > 0:
                acceptors_by_species.setdefault(species_name, []).append(
                    int(atom_index)
                )
    if not lithium_atom_indices:
        raise ValueError("ionic macrostate requires lithium sites")
    ordered_species = tuple(sorted(acceptors_by_species))
    feature_rows: list[Array] = []
    for positions_m, box_vectors_m in zip(
        positions_batch_m, box_vectors_batch_m, strict=True
    ):
        lithium_positions_m = positions_m[np.asarray(lithium_atom_indices)]
        shell_features: list[int] = []
        contact_counts_by_molecule = np.zeros(
            len(system.molecule_atom_indices), dtype=int
        )
        for species_name in ordered_species:
            acceptor_indices = np.asarray(acceptors_by_species[species_name])
            species_record = library.species_records[species_name]
            formal_charge_e = float(species_record["formal_charge_e"])
            switch_name = "Li_anion" if formal_charge_e < 0.0 else "Li_ligand"
            switch_record = library.basis_record["coordination_switches"][switch_name]
            displacements_m = minimum_image_displacement(
                lithium_positions_m[:, None, :]
                - positions_m[acceptor_indices][None, :, :],
                box_vectors_m,
            )
            distances_m = np.linalg.norm(displacements_m, axis=2)
            smooth_contacts = 1.0 / (
                1.0
                + (distances_m / float(switch_record["r0_m"]))
                ** int(switch_record["exponent"])
            )
            shell_features.extend(
                np.rint(np.sum(smooth_contacts, axis=1)).astype(int).tolist()
            )
            if formal_charge_e < 0.0:
                contacted_acceptors = smooth_contacts >= 0.5
                for acceptor_position, atom_index in enumerate(acceptor_indices):
                    molecule_index = next(
                        index
                        for index, molecule_atoms in enumerate(
                            system.molecule_atom_indices
                        )
                        if atom_index in molecule_atoms
                    )
                    contact_counts_by_molecule[molecule_index] += int(
                        np.count_nonzero(contacted_acceptors[:, acceptor_position])
                    )
        charged_count = len(charged_molecule_indices)
        adjacency = np.zeros((charged_count, charged_count), dtype=bool)
        contact_cutoff_m = float(
            library.basis_record["coordination_switches"]["Li_anion"]["r0_m"]
        )
        for first_position, first_molecule in enumerate(charged_molecule_indices):
            for second_position in range(first_position + 1, charged_count):
                second_molecule = charged_molecule_indices[second_position]
                pair_displacements_m = minimum_image_displacement(
                    positions_m[system.molecule_atom_indices[first_molecule]][
                        :, None, :
                    ]
                    - positions_m[system.molecule_atom_indices[second_molecule]][
                        None, :, :
                    ],
                    box_vectors_m,
                )
                connected = bool(
                    np.min(np.linalg.norm(pair_displacements_m, axis=2))
                    < contact_cutoff_m
                )
                adjacency[first_position, second_position] = connected
                adjacency[second_position, first_position] = connected
        unseen = set(range(charged_count))
        cluster_sizes: list[int] = []
        while unseen:
            frontier = {unseen.pop()}
            cluster = set(frontier)
            while frontier:
                current = frontier.pop()
                neighbors = set(np.flatnonzero(adjacency[current])) & unseen
                unseen -= neighbors
                frontier |= neighbors
                cluster |= neighbors
            cluster_sizes.append(len(cluster))
        cluster_histogram = np.bincount(cluster_sizes, minlength=charged_count + 1)[1:]
        negative_bridge_count = sum(
            contact_counts_by_molecule[molecule_index] >= 2
            for molecule_index, charge_C in enumerate(molecule_charges_C)
            if charge_C < 0.0
        )
        feature_rows.append(
            np.asarray(
                shell_features
                + contact_counts_by_molecule.tolist()
                + [negative_bridge_count, max(cluster_sizes), cluster_sizes.count(1)]
                + cluster_histogram.tolist(),
                dtype=int,
            )
        )
    return np.stack(feature_rows)


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
    random_generator: np.random.Generator,
) -> BatchedHmcTransitionResult:
    batch_size = positions_batch_m.shape[0]
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
    initial_state = model.energy_force_components_batch(
        positions_batch_m=proposed_positions_m,
        box_vectors_batch_m=box_vectors_batch_m,
        lambda_values=lambda_values,
    )
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
        next_state = model.energy_force_components_batch(
            positions_batch_m=proposed_positions_m,
            box_vectors_batch_m=box_vectors_batch_m,
            lambda_values=lambda_values,
        )
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
    )


def _run_hmc_calibration_window(
    model: IonicTemperedModel,
    snapshot_state: IonicHrexState,
    settings: IonicHrexSettings,
    temperature_K: float,
    replica_index: int,
    timestep_s: float,
    cycle_count: int,
    random_generator_state: dict,
) -> tuple[Array, Array, Array, Array, Array, Array, dict, int]:
    ladder_count = settings.independent_ladder_count
    replica_count = len(settings.lambdas)
    state_indices = np.arange(ladder_count) * replica_count + replica_index
    positions_batch_m = snapshot_state.positions_m[state_indices].copy()
    box_vectors_batch_m = snapshot_state.boxes_m[state_indices].copy()
    momenta_batch_kg_m_s = snapshot_state.momenta_kg_m_s[state_indices].copy()
    refresh_required = snapshot_state.momentum_refresh_required[state_indices].copy()
    random_generator = np.random.default_rng()
    random_generator.bit_generator.state = copy.deepcopy(random_generator_state)
    expected_acceptance_rows: list[Array] = []
    realized_acceptance_rows: list[Array] = []
    absolute_energy_error_rows: list[Array] = []
    molecular_com_squared_displacement_rows_m2: list[Array] = []
    force_evaluation_count = 0
    for cycle_index in range(cycle_count):
        cycle_in_window = cycle_index % HMC_CALIBRATION_WINDOW_CYCLE_COUNT
        integration_step_count = settings.hmc_steps_min
        if cycle_in_window >= HMC_CALIBRATION_WINDOW_CYCLE_COUNT // 2:
            integration_step_count = settings.hmc_steps_max
        previous_positions_batch_m = positions_batch_m.copy()
        transition = batched_hmc_transition(
            model=model,
            positions_batch_m=positions_batch_m,
            box_vectors_batch_m=box_vectors_batch_m,
            momenta_kg_m_s=momenta_batch_kg_m_s,
            auxiliary_masses_kg=snapshot_state.auxiliary_masses_kg,
            momentum_refresh_required=refresh_required,
            momentum_persistence=settings.hmc_momentum_persistence,
            temperature_K=temperature_K,
            lambda_values=np.full(ladder_count, settings.lambdas[replica_index]),
            timestep_values_s=np.full(ladder_count, timestep_s),
            integration_step_counts=np.full(ladder_count, integration_step_count),
            random_generator=random_generator,
        )
        positions_batch_m = transition.positions_m
        momenta_batch_kg_m_s = transition.momenta_kg_m_s
        refresh_required.fill(False)
        expected_acceptance_rows.append(
            np.exp(transition.log_acceptance_probabilities)
        )
        realized_acceptance_rows.append(transition.accepted.copy())
        absolute_energy_error_rows.append(
            np.abs(transition.energy_errors_over_kbt)
        )
        atomic_displacements_m = positions_batch_m - previous_positions_batch_m
        for ladder_index in range(ladder_count):
            atomic_displacements_m[ladder_index] = minimum_image_displacement(
                atomic_displacements_m[ladder_index],
                box_vectors_batch_m[ladder_index],
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
        molecular_com_squared_displacement_rows_m2.append(
            np.mean(
                np.sum(molecular_com_displacements_m**2, axis=2),
                axis=1,
            )
        )
        force_evaluation_count += integration_step_count + 1
    return (
        np.asarray(expected_acceptance_rows),
        np.asarray(realized_acceptance_rows),
        np.asarray(absolute_energy_error_rows),
        np.asarray(molecular_com_squared_displacement_rows_m2),
        positions_batch_m,
        momenta_batch_kg_m_s,
        copy.deepcopy(random_generator.bit_generator.state),
        force_evaluation_count,
    )


def _calibrate_hmc_step_sizes_s(
    model: IonicTemperedModel,
    snapshot_state: IonicHrexState,
    validation_state: IonicHrexState,
    settings: IonicHrexSettings,
    temperature_K: float,
    starting_step_sizes_s: Array,
    high_acceptance_step_sizes_s: Array,
    low_acceptance_step_sizes_s: Array,
) -> tuple[Array, Array, Array]:
    selected_step_sizes_s = np.zeros(len(settings.lambdas))
    base_random_generator_state = copy.deepcopy(snapshot_state.random_generator_state)
    maximum_window_cycles = (
        HMC_SELECTION_WINDOW_COUNT * HMC_CALIBRATION_WINDOW_CYCLE_COUNT
    )
    for replica_index, lambda_value in enumerate(settings.lambdas):
        candidate_timestep_s = float(starting_step_sizes_s[replica_index])
        candidate_evaluation_count = 0
        safe_candidate_step_sizes_s: list[float] = []
        safe_candidate_movement_per_force_evaluation_m2: list[float] = []
        while candidate_evaluation_count < HMC_MAXIMUM_CANDIDATE_COUNT:
            candidate_evaluation_count += 1
            bracket_was_updated = False
            selection_state = copy.deepcopy(snapshot_state)
            selection_random_generator_state = copy.deepcopy(
                base_random_generator_state
            )
            selection_expected_rows: list[Array] = []
            selection_realized_rows: list[Array] = []
            selection_energy_error_rows: list[Array] = []
            selection_displacement_rows_m2: list[Array] = []
            selection_force_evaluation_count = 0
            selection_window_count = (
                maximum_window_cycles // HMC_CALIBRATION_WINDOW_CYCLE_COUNT
            )
            for _selection_window_index in range(selection_window_count):
                window_start_s = time.perf_counter()
                (
                    window_expected_acceptance,
                    window_realized_acceptance,
                    window_absolute_energy_errors,
                    window_molecular_com_squared_displacements_m2,
                    terminal_positions_m,
                    terminal_momenta_kg_m_s,
                    terminal_random_generator_state,
                    window_force_evaluation_count,
                ) = _run_hmc_calibration_window(
                    model=model,
                    snapshot_state=selection_state,
                    settings=settings,
                    temperature_K=temperature_K,
                    replica_index=replica_index,
                    timestep_s=candidate_timestep_s,
                    cycle_count=HMC_CALIBRATION_WINDOW_CYCLE_COUNT,
                    random_generator_state=selection_random_generator_state,
                )
                replica_count = len(settings.lambdas)
                state_indices = (
                    np.arange(settings.independent_ladder_count) * replica_count
                    + replica_index
                )
                selection_state.positions_m[state_indices] = terminal_positions_m
                selection_state.momenta_kg_m_s[state_indices] = (
                    terminal_momenta_kg_m_s
                )
                selection_state.momentum_refresh_required[state_indices] = False
                selection_random_generator_state = (
                    terminal_random_generator_state
                )
                selection_expected_rows.append(window_expected_acceptance)
                selection_realized_rows.append(window_realized_acceptance)
                selection_energy_error_rows.append(window_absolute_energy_errors)
                selection_displacement_rows_m2.append(
                    window_molecular_com_squared_displacements_m2
                )
                selection_force_evaluation_count += window_force_evaluation_count
                expected_acceptance = np.concatenate(selection_expected_rows)
                realized_acceptance = np.concatenate(selection_realized_rows)
                absolute_energy_errors = np.concatenate(
                    selection_energy_error_rows
                )
                molecular_com_squared_displacements_m2 = np.concatenate(
                    selection_displacement_rows_m2
                )
                ladder_means = np.mean(expected_acceptance, axis=0)
                mean_expected_acceptance = float(np.mean(ladder_means))
                standard_error = float(
                    np.std(ladder_means, ddof=1) / math.sqrt(ladder_means.size)
                )
                interval_minimum = mean_expected_acceptance - standard_error
                interval_maximum = mean_expected_acceptance + standard_error
                maximum_energy_error = float(np.max(absolute_energy_errors))
                median_energy_error = float(np.median(absolute_energy_errors))
                finite_window = bool(
                    np.all(np.isfinite(expected_acceptance))
                    and np.all(np.isfinite(absolute_energy_errors))
                )
                catastrophic_window = (
                    not finite_window
                    or maximum_energy_error > -math.log(np.finfo(float).tiny)
                )
                movement_per_force_evaluation_m2 = float(
                    np.sum(molecular_com_squared_displacements_m2)
                    / selection_force_evaluation_count
                )
                print(
                    f"[HMC calibration] lambda={lambda_value:.12g} "
                    f"timestep_s={candidate_timestep_s:.12g} "
                    f"raw_proposals={expected_acceptance.size} "
                    f"ladder_means={tuple(float(value) for value in ladder_means)} "
                    f"expected={mean_expected_acceptance:.6f} se={standard_error:.6f} "
                    f"interval=({interval_minimum:.6f}, {interval_maximum:.6f}) "
                    f"realized={float(np.mean(realized_acceptance)):.6f} "
                    f"rms_com_m={float(np.sqrt(np.mean(molecular_com_squared_displacements_m2))):.12g} "
                    f"median_abs_delta_h_over_kbt={median_energy_error:.6f} "
                    f"max_abs_delta_h_over_kbt={maximum_energy_error:.6f} "
                    f"force_evaluations={selection_force_evaluation_count} "
                    f"elapsed_s={time.perf_counter() - window_start_s:.3f}",
                    flush=True,
                )
                if len(selection_expected_rows) < HMC_SELECTION_WINDOW_COUNT:
                    continue
                if (
                    not catastrophic_window
                    and mean_expected_acceptance
                    >= settings.hmc_target_acceptance_minimum
                ):
                    safe_candidate_step_sizes_s.append(candidate_timestep_s)
                    safe_candidate_movement_per_force_evaluation_m2.append(
                        movement_per_force_evaluation_m2
                    )
                if catastrophic_window or (
                    mean_expected_acceptance
                    < settings.hmc_target_acceptance_minimum
                ):
                    low_acceptance_step_sizes_s[replica_index] = min(
                        low_acceptance_step_sizes_s[replica_index],
                        candidate_timestep_s,
                    )
                    bracket_was_updated = True
                    break
                if (
                    mean_expected_acceptance
                    > settings.hmc_target_acceptance_maximum
                ):
                    high_acceptance_step_sizes_s[replica_index] = max(
                        high_acceptance_step_sizes_s[replica_index],
                        candidate_timestep_s,
                    )
                    bracket_was_updated = True
                    break
                holdout_state = copy.deepcopy(validation_state)
                holdout_random_generator_state = copy.deepcopy(
                    validation_state.random_generator_state
                )
                holdout_expected_rows: list[Array] = []
                holdout_energy_error_rows: list[Array] = []
                print(
                    f"[HMC calibration holdout] lambda={lambda_value:.12g} "
                    f"timestep_s={candidate_timestep_s:.12g}",
                    flush=True,
                )
                for holdout_window_index in range(selection_window_count):
                    (
                        holdout_window_expected,
                        _holdout_window_realized,
                        holdout_window_energy_errors,
                        _holdout_window_displacements_m2,
                        holdout_terminal_positions_m,
                        holdout_terminal_momenta_kg_m_s,
                        holdout_terminal_random_generator_state,
                        _holdout_window_force_evaluation_count,
                    ) = _run_hmc_calibration_window(
                        model=model,
                        snapshot_state=holdout_state,
                        settings=settings,
                        temperature_K=temperature_K,
                        replica_index=replica_index,
                        timestep_s=candidate_timestep_s,
                        cycle_count=HMC_CALIBRATION_WINDOW_CYCLE_COUNT,
                        random_generator_state=holdout_random_generator_state,
                    )
                    holdout_state.positions_m[state_indices] = (
                        holdout_terminal_positions_m
                    )
                    holdout_state.momenta_kg_m_s[state_indices] = (
                        holdout_terminal_momenta_kg_m_s
                    )
                    holdout_state.momentum_refresh_required[state_indices] = False
                    holdout_random_generator_state = (
                        holdout_terminal_random_generator_state
                    )
                    holdout_expected_rows.append(holdout_window_expected)
                    holdout_energy_error_rows.append(
                        holdout_window_energy_errors
                    )
                    holdout_expected_acceptance = np.concatenate(
                        holdout_expected_rows
                    )
                    holdout_energy_errors = np.concatenate(
                        holdout_energy_error_rows
                    )
                    print(
                        f"[HMC calibration holdout] completed_cycles="
                        f"{(holdout_window_index + 1) * HMC_CALIBRATION_WINDOW_CYCLE_COUNT}",
                        flush=True,
                    )
                    holdout_ladder_means = np.mean(
                        holdout_expected_acceptance,
                        axis=0,
                    )
                    holdout_mean = float(np.mean(holdout_ladder_means))
                    holdout_standard_error = float(
                        np.std(holdout_ladder_means, ddof=1)
                        / math.sqrt(holdout_ladder_means.size)
                    )
                    print(
                        f"[HMC calibration holdout] "
                        f"ladder_means={tuple(float(value) for value in holdout_ladder_means)} "
                        f"expected={holdout_mean:.6f} "
                        f"se={holdout_standard_error:.6f}",
                        flush=True,
                    )
                    holdout_is_finite = bool(
                        np.all(np.isfinite(holdout_expected_acceptance))
                        and np.all(np.isfinite(holdout_energy_errors))
                    )
                    holdout_is_catastrophic = (
                        not holdout_is_finite
                        or float(np.max(holdout_energy_errors))
                        > -math.log(np.finfo(float).tiny)
                    )
                    if (
                        len(holdout_expected_rows)
                        < HMC_HOLDOUT_WINDOW_COUNT
                    ):
                        continue
                    if holdout_is_catastrophic or (
                        holdout_mean < settings.hmc_target_acceptance_minimum
                    ):
                        low_acceptance_step_sizes_s[replica_index] = min(
                            low_acceptance_step_sizes_s[replica_index],
                            candidate_timestep_s,
                        )
                        bracket_was_updated = True
                        break
                    if holdout_mean >= settings.hmc_target_acceptance_minimum:
                        selected_step_sizes_s[replica_index] = candidate_timestep_s
                        bracket_was_updated = False
                        break
                if selected_step_sizes_s[replica_index] == candidate_timestep_s:
                    break
                if bracket_was_updated:
                    break
            if selected_step_sizes_s[replica_index] == candidate_timestep_s:
                break
            if not bracket_was_updated:
                raise RuntimeError(
                    "HMC timestep uncertainty remained ambiguous after the "
                    f"calibration budget at lambda={lambda_value}"
                )
            high_acceptance_timestep_s = high_acceptance_step_sizes_s[replica_index]
            low_acceptance_timestep_s = low_acceptance_step_sizes_s[replica_index]
            if high_acceptance_timestep_s > 0.0 and math.isfinite(
                low_acceptance_timestep_s
            ):
                if (
                    math.log(
                        low_acceptance_timestep_s
                        / high_acceptance_timestep_s
                    )
                    <= settings.hmc_log_bracket_width_tolerance
                ):
                    break
                candidate_timestep_s = math.sqrt(
                    high_acceptance_timestep_s * low_acceptance_timestep_s
                )
                continue
            if high_acceptance_timestep_s > 0.0:
                candidate_timestep_s = high_acceptance_timestep_s / (
                    settings.hmc_step_size_adaptation_factor
                )
                continue
            if math.isfinite(low_acceptance_timestep_s):
                candidate_timestep_s = low_acceptance_timestep_s * (
                    settings.hmc_step_size_adaptation_factor
                )
                continue
            raise RuntimeError("HMC timestep calibration did not update its bracket")
        if (
            selected_step_sizes_s[replica_index] <= 0.0
            and safe_candidate_step_sizes_s
        ):
            best_candidate_index = int(
                np.argmax(safe_candidate_movement_per_force_evaluation_m2)
            )
            safe_candidate_timestep_s = safe_candidate_step_sizes_s[
                best_candidate_index
            ]
            fallback_holdout = _run_hmc_calibration_window(
                model=model,
                snapshot_state=validation_state,
                settings=settings,
                temperature_K=temperature_K,
                replica_index=replica_index,
                timestep_s=safe_candidate_timestep_s,
                cycle_count=(
                    HMC_HOLDOUT_WINDOW_COUNT
                    * HMC_CALIBRATION_WINDOW_CYCLE_COUNT
                ),
                random_generator_state=validation_state.random_generator_state,
            )
            fallback_expected_acceptance = float(np.mean(fallback_holdout[0]))
            fallback_energy_errors = fallback_holdout[2]
            fallback_is_valid = bool(
                np.all(np.isfinite(fallback_holdout[0]))
                and np.all(np.isfinite(fallback_energy_errors))
                and float(np.max(fallback_energy_errors))
                <= -math.log(np.finfo(float).tiny)
                and fallback_expected_acceptance
                >= settings.hmc_target_acceptance_minimum
            )
            if fallback_is_valid:
                selected_step_sizes_s[replica_index] = safe_candidate_timestep_s
        if selected_step_sizes_s[replica_index] <= 0.0:
            raise RuntimeError(
                f"HMC timestep calibration exhausted its budget at lambda={lambda_value}"
            )
    return (
        selected_step_sizes_s,
        high_acceptance_step_sizes_s,
        low_acceptance_step_sizes_s,
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
    if len(settings.lambdas) < 2 or settings.lambdas[0] != 1.0:
        raise ValueError("HREX ladder must start at lambda=1 and contain two replicas")
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
        settings.volume_move_stride,
        settings.independent_ladder_count,
        settings.warmup_cycle_count,
        settings.production_cycle_count,
        settings.measurement_stride,
        settings.block_cycle_count,
    )
    if any(value <= 0 for value in positive_integer_settings):
        raise ValueError("HREX counts and strides must be positive")
    if settings.hmc_steps_min > settings.hmc_steps_max:
        raise ValueError("HREX minimum HMC steps exceed maximum HMC steps")
    if settings.minimum_round_trips < 0:
        raise ValueError("HREX minimum round trips cannot be negative")
    if settings.hmc_step_size_s <= 0.0:
        raise ValueError("HREX HMC timestep must be positive")
    if not 0.0 <= settings.hmc_momentum_persistence < 1.0:
        raise ValueError("HREX momentum persistence must lie in [0, 1)")
    if not (
        0.0
        < settings.hmc_target_acceptance_minimum
        < settings.hmc_target_acceptance_maximum
        < 1.0
    ):
        raise ValueError("HREX target acceptance interval must lie in (0, 1)")
    if not 0.0 < settings.hmc_step_size_adaptation_factor < 1.0:
        raise ValueError("HREX step-size adaptation factor must lie in (0, 1)")
    if settings.hmc_log_bracket_width_tolerance <= 0.0:
        raise ValueError("HREX log-bracket width tolerance must be positive")
    if not 0.0 < settings.volume_target_acceptance < 1.0:
        raise ValueError("HREX volume target acceptance must lie in (0, 1)")
    if settings.volume_adaptation_gain <= 0.0:
        raise ValueError("HREX volume adaptation gain must be positive")
    if not (
        0.0
        < settings.minimum_log_volume_proposal
        < settings.maximum_log_volume_proposal
    ):
        raise ValueError("HREX volume proposal bounds must be positive and ordered")


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


def _scale_molecular_centers_for_volume(
    positions_m: Array,
    current_box_vectors_m: Array,
    proposed_box_vectors_m: Array,
    molecule_atom_indices: tuple[Array, ...],
    masses_kg: Array,
) -> Array:
    length_scale = (
        abs(np.linalg.det(proposed_box_vectors_m))
        / abs(np.linalg.det(current_box_vectors_m))
    ) ** (1.0 / 3.0)
    proposed_positions_m = positions_m.copy()
    current_box_lengths_m = np.diag(current_box_vectors_m)
    for atom_indices in molecule_atom_indices:
        molecule_positions_m = positions_m[atom_indices]
        anchor_m = molecule_positions_m[0]
        displacements_m = molecule_positions_m - anchor_m
        displacements_m -= current_box_lengths_m * np.round(
            displacements_m / current_box_lengths_m
        )
        unwrapped_positions_m = anchor_m + displacements_m
        center_of_mass_m = np.average(
            unwrapped_positions_m, axis=0, weights=masses_kg[atom_indices]
        )
        proposed_positions_m[atom_indices] = (
            unwrapped_positions_m - center_of_mass_m + length_scale * center_of_mass_m
        )
    return proposed_positions_m % np.diag(proposed_box_vectors_m)


def _component_array_from_result(result) -> Array:
    return np.stack(
        (
            result.fixed_energy_J,
            result.ion_ion_energy_J,
            result.ion_neutral_energy_J,
        ),
        axis=1,
    )


def _tempered_energies_from_component_array_J(
    component_energies_J: Array,
    lambda_values: Array,
) -> Array:
    return (
        component_energies_J[:, 0]
        + lambda_values * component_energies_J[:, 1]
        + np.sqrt(lambda_values) * component_energies_J[:, 2]
    )


def _batched_logarithmic_volume_transition(
    model: IonicTemperedModel,
    positions_batch_m: Array,
    box_vectors_batch_m: Array,
    component_energies_J: Array,
    lambda_values: Array,
    temperature_K: float,
    pressure_Pa: float,
    logarithmic_volume_proposal_scales: Array,
    random_generator: np.random.Generator,
) -> BatchedVolumeTransitionResult:
    batch_size = positions_batch_m.shape[0]
    current_volumes_m3 = np.abs(np.linalg.det(box_vectors_batch_m))
    logarithmic_volume_changes = random_generator.normal(
        scale=logarithmic_volume_proposal_scales, size=batch_size
    )
    proposed_volumes_m3 = current_volumes_m3 * np.exp(logarithmic_volume_changes)
    length_scales = (proposed_volumes_m3 / current_volumes_m3) ** (1.0 / 3.0)
    proposed_boxes_m = box_vectors_batch_m * length_scales[:, None, None]
    proposed_positions_m = np.asarray(
        [
            _scale_molecular_centers_for_volume(
                positions_m=positions_batch_m[batch_index],
                current_box_vectors_m=box_vectors_batch_m[batch_index],
                proposed_box_vectors_m=proposed_boxes_m[batch_index],
                molecule_atom_indices=model.system.molecule_atom_indices,
                masses_kg=model.system.masses_kg,
            )
            for batch_index in range(batch_size)
        ]
    )
    proposed_result = model.energy_components_batch(
        positions_batch_m=proposed_positions_m,
        box_vectors_batch_m=proposed_boxes_m,
    )
    proposed_components_J = _component_array_from_result(proposed_result)
    current_energies_J = _tempered_energies_from_component_array_J(
        component_energies_J, lambda_values
    )
    proposed_energies_J = _tempered_energies_from_component_array_J(
        proposed_components_J, lambda_values
    )
    beta_per_J = 1.0 / (K_B * temperature_K)
    delta_energy_J = proposed_energies_J - current_energies_J
    pressure_work_J = pressure_Pa * (proposed_volumes_m3 - current_volumes_m3)
    jacobian_log_weight = len(model.system.molecule_atom_indices) * np.log(
        proposed_volumes_m3 / current_volumes_m3
    )
    logarithmic_acceptance = (
        -beta_per_J * (delta_energy_J + pressure_work_J) + jacobian_log_weight
    )
    if not np.all(np.isfinite(logarithmic_acceptance)):
        raise FloatingPointError("volume proposal produced nonfinite acceptance terms")
    expected_acceptance_probabilities = np.exp(np.minimum(0.0, logarithmic_acceptance))
    accepted = np.log(random_generator.random(batch_size)) < np.minimum(
        0.0, logarithmic_acceptance
    )
    return BatchedVolumeTransitionResult(
        positions_m=np.where(
            accepted[:, None, None], proposed_positions_m, positions_batch_m
        ),
        boxes_m=np.where(
            accepted[:, None, None], proposed_boxes_m, box_vectors_batch_m
        ),
        component_energies_J=np.where(
            accepted[:, None], proposed_components_J, component_energies_J
        ),
        accepted=accepted,
        expected_acceptance_probabilities=expected_acceptance_probabilities,
        log_volume_changes=logarithmic_volume_changes,
        current_volumes_m3=current_volumes_m3,
        proposed_volumes_m3=proposed_volumes_m3,
        delta_energy_J=delta_energy_J,
        pressure_work_J=pressure_work_J,
        jacobian_log_weight=jacobian_log_weight,
        log_acceptance_probabilities=logarithmic_acceptance,
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
        log_volume_proposal_scales=np.tile(
            np.asarray(settings.lambdas) * 0.0 + settings.logarithmic_volume_proposal,
            ladder_count,
        ),
        hmc_attempts=np.zeros((ladder_count, replica_count), dtype=int),
        hmc_acceptances=np.zeros((ladder_count, replica_count), dtype=int),
        hmc_expected_acceptance_sums=np.zeros((ladder_count, replica_count)),
        hmc_absolute_energy_error_over_kbt_sums=np.zeros(
            (ladder_count, replica_count)
        ),
        hmc_molecular_com_squared_displacement_sums_m2=np.zeros(
            (ladder_count, replica_count)
        ),
        exchange_attempts=np.zeros((ladder_count, replica_count - 1), dtype=int),
        exchange_acceptances=np.zeros((ladder_count, replica_count - 1), dtype=int),
        exchange_expected_acceptance_sums=np.zeros(
            (ladder_count, replica_count - 1)
        ),
        volume_attempts=np.zeros((ladder_count, replica_count), dtype=int),
        volume_acceptances=np.zeros((ladder_count, replica_count), dtype=int),
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
    pressure_Pa: float,
    cycle_count: int,
    sample_volume: bool,
    attempt_exchange: bool,
    adapt_volume: bool,
    retain_samples: bool,
) -> tuple[IonicHrexState, IonicHrexBlock]:
    if cycle_count <= 0:
        raise ValueError("HREX advance cycle count must be positive")
    if adapt_volume and not sample_volume:
        raise ValueError("volume adaptation requires NPT sampling")
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
    delta_energy_over_kbt: list[float] = []
    pressure_work_over_kbt: list[float] = []
    jacobian_weights: list[float] = []
    log_acceptances: list[float] = []
    physical_volume_attempts = 0
    physical_volume_acceptances = 0
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
            random_generator=random_generator,
        )
        force_evaluation_count += integration_step_count + 1
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
        if sample_volume and absolute_cycle_index % settings.volume_move_stride == 0:
            volume_transition = _batched_logarithmic_volume_transition(
                model=model,
                positions_batch_m=state.positions_m,
                box_vectors_batch_m=state.boxes_m,
                component_energies_J=state.component_energies_J,
                lambda_values=lambda_values,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                logarithmic_volume_proposal_scales=state.log_volume_proposal_scales,
                random_generator=random_generator,
            )
            state.positions_m = volume_transition.positions_m
            state.boxes_m = volume_transition.boxes_m
            state.component_energies_J = volume_transition.component_energies_J
            accepted_matrix = volume_transition.accepted.reshape(
                ladder_count, replica_count
            )
            state.momentum_refresh_required |= volume_transition.accepted
            state.volume_attempts += 1
            state.volume_acceptances += accepted_matrix
            physical_volume_attempts += ladder_count
            physical_volume_acceptances += int(
                np.count_nonzero(volume_transition.accepted[physical_indices])
            )
            delta_energy_over_kbt.extend(
                (
                    beta_per_J * volume_transition.delta_energy_J[physical_indices]
                ).tolist()
            )
            pressure_work_over_kbt.extend(
                (
                    beta_per_J * volume_transition.pressure_work_J[physical_indices]
                ).tolist()
            )
            jacobian_weights.extend(
                volume_transition.jacobian_log_weight[physical_indices].tolist()
            )
            log_acceptances.extend(
                volume_transition.log_acceptance_probabilities[
                    physical_indices
                ].tolist()
            )
            if adapt_volume:
                expected_matrix = (
                    volume_transition.expected_acceptance_probabilities.reshape(
                        ladder_count, replica_count
                    )
                )
                expected_by_replica = np.mean(expected_matrix, axis=0)
                current_by_replica = state.log_volume_proposal_scales.reshape(
                    ladder_count, replica_count
                )[0]
                adaptation_rate = settings.volume_adaptation_gain / math.sqrt(
                    absolute_cycle_index
                )
                proposed_by_replica = np.exp(
                    np.log(current_by_replica)
                    + adaptation_rate
                    * (expected_by_replica - settings.volume_target_acceptance)
                )
                bounded_by_replica = np.clip(  # CLIP-OK: configured adaptation bounds prevent barostat proposal collapse or divergence.
                    proposed_by_replica,
                    settings.minimum_log_volume_proposal,
                    settings.maximum_log_volume_proposal,
                )
                state.log_volume_proposal_scales = np.tile(
                    bounded_by_replica, ladder_count
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
        physical_volume_attempts=physical_volume_attempts,
        physical_volume_acceptances=physical_volume_acceptances,
        physical_delta_energy_over_kbt=np.asarray(delta_energy_over_kbt),
        physical_pressure_work_over_kbt=np.asarray(pressure_work_over_kbt),
        physical_jacobian_log_weight=np.asarray(jacobian_weights),
        physical_log_acceptance_probabilities=np.asarray(log_acceptances),
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


def _ionic_hrex_result_from_state_and_block(
    state: IonicHrexState,
    block: IonicHrexBlock,
    settings: IonicHrexSettings,
) -> IonicHrexResult:
    hmc_fraction = np.divide(
        state.hmc_acceptances,
        state.hmc_attempts,
        out=np.zeros_like(state.hmc_acceptances, dtype=float),
        where=state.hmc_attempts > 0,
    )
    exchange_fraction = np.divide(
        state.exchange_acceptances,
        state.exchange_attempts,
        out=np.zeros_like(state.exchange_acceptances, dtype=float),
        where=state.exchange_attempts > 0,
    )
    round_trip_count = int(np.sum(state.completed_round_trips))
    replica_count = len(settings.lambdas)
    physical_indices = np.arange(settings.independent_ladder_count) * replica_count
    final_physical_boxes_m = state.boxes_m[physical_indices]
    physical_box_vectors_m = final_physical_boxes_m[0]
    if settings.independent_ladder_count > 1:
        physical_box_vectors_m = final_physical_boxes_m
    return IonicHrexResult(
        physical_configurations_m=block.physical_configurations_m,
        physical_box_vectors_m=physical_box_vectors_m,
        physical_lambda_values=tuple(
            1.0 for _configuration in block.physical_configurations_m
        ),
        hmc_acceptance_fractions=tuple(float(value) for value in hmc_fraction.ravel()),
        exchange_acceptance_fractions=tuple(
            float(value) for value in exchange_fraction.ravel()
        ),
        round_trip_count=round_trip_count,
        sampled_volumes_m3=block.sampled_volumes_m3,
        sampled_energies_J=block.sampled_energies_J,
        physical_ladder_indices=block.physical_ladder_indices,
        physical_box_vectors_by_sample_m=block.physical_box_vectors_by_sample_m,
        volume_attempt_count=block.physical_volume_attempts,
        volume_acceptance_count=block.physical_volume_acceptances,
        log_volume_proposal_scales=state.log_volume_proposal_scales.copy(),
        physical_delta_energy_over_kbt=block.physical_delta_energy_over_kbt,
        physical_pressure_work_over_kbt=block.physical_pressure_work_over_kbt,
        physical_jacobian_log_weight=block.physical_jacobian_log_weight,
        physical_log_acceptance_probabilities=(
            block.physical_log_acceptance_probabilities
        ),
    )


def _physical_barostat_diagnostic_text(
    block: IonicHrexBlock,
    physical_proposal_scales: Array,
) -> str:
    return (
        f"physical volume moves: attempts={block.physical_volume_attempts}, "
        f"accepted={block.physical_volume_acceptances}, "
        f"proposal_scales={tuple(float(value) for value in physical_proposal_scales)}, "
        f"median_abs_delta_energy_over_kbt="
        f"{float(np.median(np.abs(block.physical_delta_energy_over_kbt))):.8g}, "
        f"median_abs_pressure_work_over_kbt="
        f"{float(np.median(np.abs(block.physical_pressure_work_over_kbt))):.8g}, "
        f"median_abs_jacobian_log_weight="
        f"{float(np.median(np.abs(block.physical_jacobian_log_weight))):.8g}, "
        f"median_log_acceptance="
        f"{float(np.median(block.physical_log_acceptance_probabilities)):.8g}"
    )


def _validate_physical_barostat_warmup(
    block: IonicHrexBlock,
    physical_proposal_scales: Array,
) -> None:
    if block.physical_volume_attempts == 0:
        raise VolumeMoveConfigurationError(
            "NPT warm-up scheduled no physical volume moves"
        )
    if block.physical_volume_acceptances == 0:
        raise BarostatNotMixingError(
            _physical_barostat_diagnostic_text(block, physical_proposal_scales)
        )


def _validate_physical_barostat_production_block(
    block: IonicHrexBlock,
    physical_proposal_scales: Array,
) -> None:
    if block.physical_volume_attempts == 0:
        raise VolumeMoveConfigurationError(
            "NPT production block scheduled no physical volume moves"
        )
    diagnostic_text = _physical_barostat_diagnostic_text(
        block=block,
        physical_proposal_scales=physical_proposal_scales,
    )
    if block.physical_volume_acceptances == 0:
        raise BarostatNotMixingError(diagnostic_text)
    mean_volume_m3 = float(np.mean(block.sampled_volumes_m3))
    relative_volume_range = float(np.ptp(block.sampled_volumes_m3) / mean_volume_m3)
    if relative_volume_range <= np.finfo(float).eps:
        raise BarostatProposalCollapsedError(diagnostic_text)


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
    print(
        f"conductivity = {result.conductivity_S_m:.8g} S/m ({result.conductivity_S_m * S_M_TO_MS_CM:.8g} mS/cm)"
    )
    print(f"direct = {result.direct_current_term_S_m:.8g} S/m")
    print(f"projected correction = {result.projected_correction_S_m:.8g} S/m")
    print(f"equilibrium volume = {result.equilibrium_volume_m3:.8g} m3")
    print(f"equilibrium density = {result.equilibrium_density_g_cm3:.8g} g/cm3")
    print(f"basis sequence = {result.basis_conductivities_S_m}")
    print(f"residual sequence = {result.residual_history}")
    print(
        f"basis size = {result.basis_size}; equilibrium samples = "
        f"{result.equilibrium_sample_count}; memory samples = "
        f"{result.memory_sample_count}; ESS = {result.effective_sample_size:.6g}"
        f"; split-Rhat = {result.maximum_split_rhat:.6g}"
        f"; conductivity MCSE = {result.conductivity_mcse_S_m:.6g} S/m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
