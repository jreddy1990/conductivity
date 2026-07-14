"""Finite Markov-additive conductivity generator.

This module implements the closed-form conductivity readout for a reversible
finite motif generator. The recipe-level builder derives the finite objects from
the same composition, matrix, solvation, and speciation kernel used by the
auditable Onsager prototype, then evaluates the Green-Kubo/Einstein Poisson
formula on the finite Markov additive process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from constants import EPS_0, F, K_B, N_A, R
from conductivity.onsager_physics_sigma import (
    TransportKernelState,
    build_transport_kernel_state,
    crowding_factor_from_ionic_volume_fraction,
    transport_microviscosity_cP,
)
from conductivity.ion_atmosphere import (
    BulkIonAtmosphereInput,
    BulkIonAtmosphereState,
    build_bulk_ion_atmosphere_state,
)
from conductivity.finite_mori_conductivity import (
    MORI_NUMERICAL_TOLERANCE,
    ProjectedMoriConductivityInput,
    ProjectedMoriConductivityResult,
    compute_projected_mori_conductivity,
)
from data.species_data import ADDITIVES, SOLVENTS
from utils.config_load_cache import load_physics_config
from utils.strict_validation import require_float, require_mapping


AXIS_COUNT = 3
ANGSTROM_TO_M = 1.0e-10
NM_TO_M = 1.0e-9
CP_TO_PA_S = 1.0e-3
MOLARITY_TO_MOL_M3 = 1000.0
ML_TO_M3 = 1.0e-6
ANGSTROM3_TO_M3 = 1.0e-30
S_CM2_PER_MOL_TO_S_M2_PER_MOL = 1.0e-4
KJ_TO_J = 1000.0  # Explicit constant: unit conversion, 1 kJ = 1000 J.
THREE_DIMENSION_MSD_FACTOR = (
    6.0  # Explicit constant: Einstein relation in 3D, <r^2> = 6 D t.
)
STOKES_SPHERE_DRAG_FACTOR = (
    6.0  # Explicit constant: Stokes-Einstein sphere drag D = kBT/(6*pi*eta*R).
)
REVERSE_DIFFUSION_TOLERANCE = 1.0e-12
CATION_RADIUS_MATCH_TOLERANCE_A = (
    1.0e-9  # Explicit tolerance for metadata identity matching.
)
MASS_BALANCE_MAX_ITERATIONS = (
    80  # Numerical solver cap for log-activity Newton iterations.
)
MASS_BALANCE_DAMPING_ATTEMPTS = (
    16  # Numerical line-search cap for residual-reducing Newton steps.
)
MASS_BALANCE_ABSOLUTE_TOLERANCE_M = (
    1.0e-10  # Numerical mol/L residual tolerance for exact mass balance.
)
FINITE_MARKOV_ION_ATMOSPHERE_SOLVER = "finite_size_bulk_pnp_stokes_l1_cell"
ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL = "total_formal"
ATMOSPHERE_BATH_BASIS_EXTERNAL_FREE_BATH = "external_free_bath"
SUPPORTED_ATMOSPHERE_BATH_BASES = (
    ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
    ATMOSPHERE_BATH_BASIS_EXTERNAL_FREE_BATH,
)
RELAXATION_DYNAMIC_RESPONSE_OFF = "off"
RELAXATION_DYNAMIC_RESPONSE_STATE_LIFETIME = "state_lifetime"
SUPPORTED_RELAXATION_DYNAMIC_RESPONSES = (
    RELAXATION_DYNAMIC_RESPONSE_OFF,
    RELAXATION_DYNAMIC_RESPONSE_STATE_LIFETIME,
)
ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF = "off"
ANION_DIAGONAL_RELAXATION_FORM_FACTOR_RESOLVED_STATE_FINITE_SIZE = (
    "resolved_state_finite_size"
)
SUPPORTED_ANION_DIAGONAL_RELAXATION_FORM_FACTORS = (
    ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF,
    ANION_DIAGONAL_RELAXATION_FORM_FACTOR_RESOLVED_STATE_FINITE_SIZE,
)
GAUSSIAN_SELF_FORM_FACTOR_SQUARED_DENOMINATOR = (
    3.0  # Squared Gaussian charge-cloud form, exp[-(ka)^2/3].
)
CHARGE_CLOUD_SOURCE_NOT_APPLICABLE = "not_applicable_non_anion_center"
CHARGE_CLOUD_SOURCE_WEIGHTED_MISSING = "weighted_anion_missing_partial_charge_geometry"
PAIR_BASIN_QUADRATURE_POINTS = MASS_BALANCE_MAX_ITERATIONS
COULOMB_DENOMINATOR_FACTOR = 4.0  # Analytical Coulomb denominator: 4*pi*epsilon.
EXTERNAL_BATH_CONCENTRATION_TOLERANCE_MOL_M3 = (
    MASS_BALANCE_ABSOLUTE_TOLERANCE_M * MOLARITY_TO_MOL_M3
)


class ChemicalMotifKind(Enum):
    SOLVENT_CAGE = "solvent_cage"
    FREE_ANION = "free_anion"
    ADDITIVE_COORDINATED = "additive_coordinated"
    SSIP = "ssip"
    CIP = "cip"
    ADDITIVE_SSIP = "additive_ssip"
    AGGREGATE = "aggregate"
    BRIDGE_NETWORK = "bridge_network"
    LI2A_PLUS = "li2a_plus"
    LIA2_MINUS = "lia2_minus"
    LI2A2_NEUTRAL = "li2a2_neutral"


@dataclass(frozen=True)
class ChemicalMotif:
    label: str
    kind: ChemicalMotifKind
    feature_id: str | None


@dataclass(frozen=True)
class StateConcentrationKernel:
    state_labels: tuple[str, ...]
    species_labels: tuple[str, ...]
    stoichiometry: np.ndarray
    standard_free_energies_J_mol: np.ndarray
    free_activities_M: dict[str, float]
    state_concentrations_M: np.ndarray
    mass_balance_residuals_M: dict[str, float]


@dataclass(frozen=True)
class StateMassActionTemplate:
    chemical_motif: ChemicalMotif
    stoichiometry: tuple[float, ...]
    equilibrium_constant: float


@dataclass(frozen=True)
class FiniteMotifState:
    motif: str
    orientation: str
    chemical_motif: ChemicalMotif


@dataclass(frozen=True)
class ChargedCenter:
    label: str
    charge: float
    hydrodynamic_radius_m: float
    shape_factor: float
    local_diffusion_m2_s: float
    relative_position_m: tuple[float, float, float]
    charge_cloud_radius_available: bool
    charge_cloud_radius_A: float
    charge_cloud_source: str
    charge_cloud_site_count: int


@dataclass(frozen=True)
class ConstraintMode:
    labels: tuple[str, ...]
    vector: tuple[float, ...]
    lifetime_s: float
    atmosphere_lifetime_s: float
    length_m: float


@dataclass(frozen=True)
class TransportState:
    label: str
    probability: float
    concentration_mol_m3: float
    charged_centers: tuple[ChargedCenter, ...]
    constraints: tuple[ConstraintMode, ...]
    atmosphere_resistance_kg_s: tuple[tuple[float, ...], ...]
    atmosphere_resistance_before_lifetime_gate_kg_s: tuple[tuple[float, ...], ...]
    atmosphere_state_lifetime_s: float
    atmosphere_relaxation_time_s: float
    atmosphere_lifetime_gate: float
    atmosphere_diagnostic_lifetime_gate: float
    relaxation_dynamic_response: str
    anion_diagonal_relaxation_form_factor: str
    relaxation_lifetime_gate: float
    relaxation_resistance_before_gate_kg_s: tuple[tuple[float, ...], ...]
    relaxation_resistance_after_gate_kg_s: tuple[tuple[float, ...], ...]
    atmosphere_bath_basis: str
    ionic_strength_total_mol_m3: float
    ionic_strength_external_mol_m3: float
    external_over_total_ionic_strength: float
    free_energy_J_mol: float


@dataclass(frozen=True)
class StateAtmosphereResistance:
    gated_resistance_kg_s: tuple[tuple[float, ...], ...]
    ungated_resistance_kg_s: tuple[tuple[float, ...], ...]
    state_lifetime_s: float
    relaxation_time_s: float
    applied_lifetime_gate: float
    diagnostic_lifetime_gate: float
    relaxation_dynamic_response: str
    anion_diagonal_relaxation_form_factor: str
    relaxation_lifetime_gate: float
    relaxation_resistance_before_gate_kg_s: tuple[tuple[float, ...], ...]
    relaxation_resistance_after_gate_kg_s: tuple[tuple[float, ...], ...]
    atmosphere_bath_basis: str
    ionic_strength_total_mol_m3: float
    ionic_strength_external_mol_m3: float
    external_over_total_ionic_strength: float


@dataclass(frozen=True)
class MotifBindingKinetics:
    motif_label: str
    K_M_inv: float
    k_on_M_inv_s: float
    k_off_s_inv: float
    tau_s: float
    constraint_tau_s: float
    basin_length_m: float
    source: str


class MarkovAdditiveEdgeKind(Enum):
    MOTIF_EXCHANGE = "motif_exchange"
    VEHICULAR_JUMP = "vehicular_jump"
    STRUCTURAL_HOP = "structural_hop"


@dataclass(frozen=True)
class MarkovAdditiveEdge:
    source_index: int
    target_index: int
    rate_s_inv: float
    displacement_m: tuple[float, float, float]
    label: str
    kind: MarkovAdditiveEdgeKind


@dataclass(frozen=True)
class TransitionAuditRow:
    source_state: str
    target_state: str
    capacity_s_inv: float
    rate_s_inv: float
    charge_displacement_m: tuple[float, float, float]
    effective_charge: float
    hop_length_m: float


@dataclass(frozen=True)
class MixtureKernelAudit:
    dielectric_bruggeman: float
    dielectric_effective: float
    viscosity_cP: float
    ionic_occupied_volume_fraction: float
    crowding_factor: float
    anion_shape_factor_by_feature: dict[str, float]
    cation_microviscosity_coupling_exponent: float
    anion_microviscosity_coupling_exponent_by_feature: dict[str, float]
    carrier_strength_Li_mS_cm: float
    carrier_strength_anion_by_feature_mS_cm: dict[str, float]
    debye_kappa_inv_m: float
    cation_concentration_mol_m3: float
    shell_fractions: dict[str, float]
    free_fraction_by_feature: dict[str, float]
    paired_fraction_by_feature: dict[str, float]
    aggregate_fraction_by_feature: dict[str, float]


@dataclass(frozen=True)
class KernelDerivedMarkovModel:
    chemical_motifs: tuple[ChemicalMotif, ...]
    states: tuple[FiniteMotifState, ...]
    state_labels: tuple[str, ...]
    stationary_probabilities: np.ndarray
    generator_s_inv: np.ndarray
    capacity_matrix_s_inv: np.ndarray
    markov_additive_edges: tuple[MarkovAdditiveEdge, ...]
    transport_states: tuple[TransportState, ...]
    binding_kinetics: tuple[MotifBindingKinetics, ...]
    state_net_charges: np.ndarray
    transition_displacements_m: np.ndarray
    chemical_motif_populations: dict[str, float]
    state_concentration_kernel: StateConcentrationKernel
    state_concentrations_mol_m3: np.ndarray
    bulk_ion_atmosphere_state: BulkIonAtmosphereState
    atmosphere_bath_basis: str
    relaxation_dynamic_response: str
    anion_diagonal_relaxation_form_factor: str
    mixture_audit: MixtureKernelAudit
    transition_audit: tuple[TransitionAuditRow, ...]
    capacity_evaluation: str


@dataclass(frozen=True)
class FiniteMarkovInput:
    stationary_probabilities: Sequence[float] | np.ndarray
    state_concentrations_mol_m3: Sequence[float] | np.ndarray
    generator_s_inv: Sequence[Sequence[float]] | np.ndarray
    markov_additive_edges: Sequence[MarkovAdditiveEdge]
    transport_states: Sequence[TransportState]
    cation_concentration_mol_m3: float
    temperature_K: float
    state_labels: Sequence[str]
    capacity_evaluation: str
    generated_model: KernelDerivedMarkovModel | None


@dataclass(frozen=True)
class ParsedFiniteMarkovInput:
    stationary_probabilities: np.ndarray
    state_concentrations_mol_m3: np.ndarray
    generator_s_inv: np.ndarray
    markov_additive_edges: tuple[MarkovAdditiveEdge, ...]
    transport_states: tuple[TransportState, ...]


@dataclass(frozen=True)
class ReversibleGenerator:
    generator_s_inv: np.ndarray
    capacity_matrix_s_inv: np.ndarray
    markov_additive_edges: tuple[MarkovAdditiveEdge, ...]
    transition_audit: tuple[TransitionAuditRow, ...]


@dataclass(frozen=True)
class TransitionContext:
    states: tuple[FiniteMotifState, ...]
    state_labels: tuple[str, ...]
    stationary_probabilities: np.ndarray
    hydrodynamic_radius_m: dict[str, float]
    vehicular_diffusion_scalar_m2_s: dict[str, float]
    temperature_K: float


@dataclass(frozen=True)
class FiniteMarkovConductivityResult:
    sigma_S_m: float
    sigma_mS_cm: float
    D_Q_m2_s: float
    vehicular_D_Q_m2_s: float
    jump_D_Q_m2_s: float
    axis_D_Q_m2_s: tuple[float, float, float]
    axis_vehicular_D_Q_m2_s: tuple[float, float, float]
    axis_jump_D_Q_m2_s: tuple[float, float, float]
    poisson_correctors_m: np.ndarray
    drift_vectors_m_s: np.ndarray
    row_sum_residual_s_inv: float
    stationary_residual_s_inv: float
    detailed_balance_residual_s_inv: float
    capacity_evaluation: str
    generated_model: KernelDerivedMarkovModel | None
    projected_mori_conductivity: ProjectedMoriConductivityResult


def evaluate_finite_markov_conductivity(
    recipe: Mapping[str, object],
    temperature_K: float,
    atmosphere_bath_basis: str = ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
    relaxation_dynamic_response: str = RELAXATION_DYNAMIC_RESPONSE_OFF,
    anion_diagonal_relaxation_form_factor: str = ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF,
) -> FiniteMarkovConductivityResult:
    _validate_atmosphere_bath_basis(atmosphere_bath_basis)
    _validate_relaxation_dynamic_response(relaxation_dynamic_response)
    _validate_anion_diagonal_relaxation_form_factor(
        anion_diagonal_relaxation_form_factor
    )
    physics_config = load_physics_config()
    kernel_state = build_transport_kernel_state(recipe, temperature_K, physics_config)
    markov_model = build_kernel_derived_markov_model(
        kernel_state,
        temperature_K,
        physics_config,
        atmosphere_bath_basis,
        relaxation_dynamic_response,
        anion_diagonal_relaxation_form_factor,
    )
    return compute_finite_markov_conductivity(
        FiniteMarkovInput(
            stationary_probabilities=markov_model.stationary_probabilities,
            state_concentrations_mol_m3=markov_model.state_concentrations_mol_m3,
            generator_s_inv=markov_model.generator_s_inv,
            markov_additive_edges=markov_model.markov_additive_edges,
            transport_states=markov_model.transport_states,
            cation_concentration_mol_m3=markov_model.mixture_audit.cation_concentration_mol_m3,
            temperature_K=temperature_K,
            state_labels=markov_model.state_labels,
            capacity_evaluation=markov_model.capacity_evaluation,
            generated_model=markov_model,
        )
    )


def compute_finite_markov_conductivity(
    finite_input: FiniteMarkovInput,
) -> FiniteMarkovConductivityResult:
    parsed_input = ParsedFiniteMarkovInput(
        stationary_probabilities=_strict_vector(
            finite_input.stationary_probabilities,
            "stationary_probabilities",
        ),
        state_concentrations_mol_m3=_strict_vector(
            finite_input.state_concentrations_mol_m3,
            "state_concentrations_mol_m3",
        ),
        generator_s_inv=_strict_matrix(finite_input.generator_s_inv, "generator_s_inv"),
        markov_additive_edges=tuple(finite_input.markov_additive_edges),
        transport_states=tuple(finite_input.transport_states),
    )
    _validate_finite_input(finite_input, parsed_input)

    stationary_probabilities = parsed_input.stationary_probabilities
    state_concentrations_mol_m3 = parsed_input.state_concentrations_mol_m3
    generator_s_inv = parsed_input.generator_s_inv
    markov_additive_edges = parsed_input.markov_additive_edges
    transport_states = parsed_input.transport_states

    poisson_correctors = np.zeros(
        (len(stationary_probabilities), AXIS_COUNT), dtype=float
    )
    drift_vectors = np.zeros((len(stationary_probabilities), AXIS_COUNT), dtype=float)
    axis_diffusivities: list[float] = []
    axis_vehicular_diffusivities: list[float] = []
    axis_jump_diffusivities: list[float] = []

    for axis_index in range(AXIS_COUNT):
        drift_vector = _edge_drift_vector_m_s(
            markov_additive_edges,
            len(stationary_probabilities),
            axis_index,
        )
        poisson_corrector = solve_poisson_corrector(
            generator_s_inv,
            stationary_probabilities,
            drift_vector,
        )
        vehicular_axis = _transport_state_axis_transport_density_mol_m_s(
            state_concentrations_mol_m3,
            transport_states,
            finite_input.temperature_K,
            axis_index,
        )
        jump_axis = float(
            0.5
            * _edge_corrected_second_moment_density_mol_m_s(
                markov_additive_edges,
                state_concentrations_mol_m3,
                poisson_corrector,
                axis_index,
            )
        )
        poisson_correctors[:, axis_index] = poisson_corrector
        drift_vectors[:, axis_index] = drift_vector
        axis_vehicular_diffusivities.append(vehicular_axis)
        axis_jump_diffusivities.append(jump_axis)
        axis_diffusivities.append(vehicular_axis + jump_axis)

    projected_mori_input = _finite_markov_projected_mori_input(
        state_concentrations_mol_m3,
        transport_states,
        markov_additive_edges,
        poisson_correctors,
        finite_input.temperature_K,
    )
    projected_mori_result = compute_projected_mori_conductivity(projected_mori_input)
    _validate_projected_mori_axis_densities(
        projected_mori_result.quadratic_form_by_axis,
        tuple(axis_diffusivities),
    )

    cation_concentration_mol_m3 = finite_input.cation_concentration_mol_m3
    mori_axis_transport_densities = np.asarray(
        projected_mori_result.quadratic_form_by_axis,
        dtype=float,
    )
    D_Q_m2_s = float(
        np.mean(mori_axis_transport_densities) / cation_concentration_mol_m3
    )
    vehicular_D_Q_m2_s = float(
        np.mean(np.asarray(axis_vehicular_diffusivities, dtype=float))
        / cation_concentration_mol_m3
    )
    jump_D_Q_m2_s = float(
        np.mean(np.asarray(axis_jump_diffusivities, dtype=float))
        / cation_concentration_mol_m3
    )
    return FiniteMarkovConductivityResult(
        sigma_S_m=projected_mori_result.sigma_S_m,
        sigma_mS_cm=projected_mori_result.sigma_mS_cm,
        D_Q_m2_s=D_Q_m2_s,
        vehicular_D_Q_m2_s=vehicular_D_Q_m2_s,
        jump_D_Q_m2_s=jump_D_Q_m2_s,
        axis_D_Q_m2_s=tuple(
            float(value / cation_concentration_mol_m3)
            for value in mori_axis_transport_densities
        ),
        axis_vehicular_D_Q_m2_s=tuple(
            float(value / cation_concentration_mol_m3)
            for value in axis_vehicular_diffusivities
        ),
        axis_jump_D_Q_m2_s=tuple(
            float(value / cation_concentration_mol_m3)
            for value in axis_jump_diffusivities
        ),
        poisson_correctors_m=poisson_correctors,
        drift_vectors_m_s=drift_vectors,
        row_sum_residual_s_inv=_generator_row_sum_residual(generator_s_inv),
        stationary_residual_s_inv=_stationary_distribution_residual(
            generator_s_inv,
            stationary_probabilities,
        ),
        detailed_balance_residual_s_inv=_detailed_balance_residual(
            generator_s_inv,
            stationary_probabilities,
        ),
        capacity_evaluation=finite_input.capacity_evaluation,
        generated_model=finite_input.generated_model,
        projected_mori_conductivity=projected_mori_result,
    )


def _finite_markov_projected_mori_input(
    state_concentrations_mol_m3: np.ndarray,
    transport_states: tuple[TransportState, ...],
    markov_additive_edges: tuple[MarkovAdditiveEdge, ...],
    poisson_correctors_m: np.ndarray,
    temperature_K: float,
) -> ProjectedMoriConductivityInput:
    direct_energy_blocks: list[np.ndarray] = []
    memory_self_energy_blocks: list[np.ndarray] = []
    current_coupling_blocks: list[np.ndarray] = []

    for state_index, transport_state in enumerate(transport_states):
        center_count = len(transport_state.charged_centers)
        if center_count == 0:
            continue
        state_concentration_mol_m3 = float(state_concentrations_mol_m3[state_index])
        _assert_nonnegative_finite(
            state_concentration_mol_m3,
            f"{transport_state.label}.concentration_mol_m3",
        )
        direct_resistance_matrix = _transport_state_local_resistance_matrix_kg_s(
            transport_state,
            temperature_K,
        )
        memory_resistance_matrix = _transport_state_memory_resistance_matrix_kg_s(
            transport_state,
            temperature_K,
        )
        direct_energy_blocks.append(direct_resistance_matrix / (K_B * temperature_K))
        memory_self_energy_blocks.append(
            memory_resistance_matrix / (K_B * temperature_K)
        )
        charge_vector = np.asarray(
            [
                charged_center.charge
                for charged_center in transport_state.charged_centers
            ],
            dtype=float,
        )
        state_current_coupling = math.sqrt(state_concentration_mol_m3) * charge_vector
        current_coupling_blocks.append(np.tile(state_current_coupling, (AXIS_COUNT, 1)))

    for edge in markov_additive_edges:
        edge_rate_s_inv = float(edge.rate_s_inv)
        source_concentration_mol_m3 = float(
            state_concentrations_mol_m3[edge.source_index]
        )
        _assert_nonnegative_finite(
            source_concentration_mol_m3, f"{edge.label}.source_concentration_mol_m3"
        )
        _assert_positive_finite(edge_rate_s_inv, f"{edge.label}.rate_s_inv")
        source_rate_density_mol_m3_s = source_concentration_mol_m3 * edge_rate_s_inv
        if source_rate_density_mol_m3_s == 0.0:
            continue
        direct_energy_blocks.append(
            np.asarray([[2.0 / source_rate_density_mol_m3_s]], dtype=float)
        )
        memory_self_energy_blocks.append(np.zeros((1, 1), dtype=float))
        corrected_displacement_by_axis_m = np.asarray(
            [
                edge.displacement_m[axis_index]
                + poisson_correctors_m[edge.target_index, axis_index]
                - poisson_correctors_m[edge.source_index, axis_index]
                for axis_index in range(AXIS_COUNT)
            ],
            dtype=float,
        )
        current_coupling_blocks.append(
            corrected_displacement_by_axis_m.reshape(AXIS_COUNT, 1)
        )

    direct_energy_matrix = _block_diagonal_matrix(direct_energy_blocks)
    memory_self_energy_matrix = _block_diagonal_matrix(memory_self_energy_blocks)
    current_coupling_matrix = _horizontally_concatenated_current_coupling_matrix(
        current_coupling_blocks,
    )
    return ProjectedMoriConductivityInput(
        direct_energy_matrix=direct_energy_matrix,
        memory_self_energy_matrix=memory_self_energy_matrix,
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=F * F / (R * temperature_K),
    )


def _block_diagonal_matrix(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.zeros((1, 1), dtype=float)
    total_dimension = sum(block.shape[0] for block in blocks)
    block_diagonal_matrix = np.zeros((total_dimension, total_dimension), dtype=float)
    offset = 0
    for matrix_block in blocks:
        if matrix_block.ndim != 2:
            raise ValueError("projected Mori matrix block must be two-dimensional")
        if matrix_block.shape[0] != matrix_block.shape[1]:
            raise ValueError(
                f"projected Mori matrix block must be square, got {matrix_block.shape}"
            )
        next_offset = offset + matrix_block.shape[0]
        block_diagonal_matrix[offset:next_offset, offset:next_offset] = matrix_block
        offset = next_offset
    return block_diagonal_matrix


def _horizontally_concatenated_current_coupling_matrix(
    blocks: list[np.ndarray],
) -> np.ndarray:
    if not blocks:
        return np.zeros((AXIS_COUNT, 1), dtype=float)
    for current_coupling_block in blocks:
        if current_coupling_block.ndim != 2:
            raise ValueError(
                "projected Mori current coupling block must be two-dimensional"
            )
        if current_coupling_block.shape[0] != AXIS_COUNT:
            raise ValueError(
                "projected Mori current coupling block axis count mismatch: "
                f"{current_coupling_block.shape[0]} != {AXIS_COUNT}"
            )
    return np.concatenate(blocks, axis=1)


def _validate_projected_mori_axis_densities(
    projected_axis_transport_densities: tuple[float, float, float],
    direct_axis_transport_densities: tuple[float, float, float],
) -> None:
    for axis_index, projected_axis_density in enumerate(
        projected_axis_transport_densities
    ):
        direct_axis_density = direct_axis_transport_densities[axis_index]
        density_scale = max(
            abs(projected_axis_density),
            abs(direct_axis_density),
            np.finfo(float).tiny,
        )
        allowed_difference = MORI_NUMERICAL_TOLERANCE * density_scale
        density_difference = abs(projected_axis_density - direct_axis_density)
        if density_difference > allowed_difference:
            raise ValueError(
                "projected Mori readout does not match finite Markov density "
                f"for axis {axis_index}: projected={projected_axis_density}, "
                f"direct={direct_axis_density}, difference={density_difference}, "
                f"allowed={allowed_difference}"
            )


def solve_poisson_corrector(
    generator_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
    drift_vector_m_s: np.ndarray,
) -> np.ndarray:
    generator_arr = _strict_matrix(generator_s_inv, "generator_s_inv")
    stationary_arr = _strict_vector(
        stationary_probabilities, "stationary_probabilities"
    )
    drift_arr = _strict_vector(drift_vector_m_s, "drift_vector_m_s")
    if generator_arr.shape[0] != stationary_arr.shape[0]:
        raise ValueError("generator_s_inv and stationary_probabilities size mismatch")
    if drift_arr.shape[0] != stationary_arr.shape[0]:
        raise ValueError("drift_vector_m_s and stationary_probabilities size mismatch")
    gauge_matrix = generator_arr + np.outer(
        np.ones(stationary_arr.shape[0], dtype=float),
        stationary_arr,
    )
    corrector = -np.linalg.solve(gauge_matrix, drift_arr)
    gauge_residual = float(np.dot(stationary_arr, corrector))
    tolerance = _linear_solve_tolerance(generator_arr)
    if abs(gauge_residual) > tolerance:
        raise ValueError(
            f"Poisson corrector violates stationary gauge by {gauge_residual}"
        )
    return corrector


def _edge_drift_vector_m_s(
    markov_additive_edges: tuple[MarkovAdditiveEdge, ...],
    state_count: int,
    axis_index: int,
) -> np.ndarray:
    drift_vector = np.zeros(state_count, dtype=float)
    for edge in markov_additive_edges:
        drift_vector[edge.source_index] += (
            edge.rate_s_inv * edge.displacement_m[axis_index]
        )
    return drift_vector


def _edge_corrected_second_moment_density_mol_m_s(
    markov_additive_edges: tuple[MarkovAdditiveEdge, ...],
    state_concentrations_mol_m3: np.ndarray,
    poisson_corrector_m: np.ndarray,
    axis_index: int,
) -> float:
    second_moment_density_mol_m_s = 0.0
    for edge in markov_additive_edges:
        corrected_displacement_m = (
            edge.displacement_m[axis_index]
            + poisson_corrector_m[edge.target_index]
            - poisson_corrector_m[edge.source_index]
        )
        second_moment_density_mol_m_s += (
            state_concentrations_mol_m3[edge.source_index]
            * edge.rate_s_inv
            * corrected_displacement_m
            * corrected_displacement_m
        )
    return float(second_moment_density_mol_m_s)


def _transport_state_axis_transport_density_mol_m_s(
    state_concentrations_mol_m3: np.ndarray,
    transport_states: tuple[TransportState, ...],
    temperature_K: float,
    axis_index: int,
) -> float:
    axis_transport_density_mol_m_s = 0.0
    for state_index, transport_state in enumerate(transport_states):
        axis_transport_density_mol_m_s += state_concentrations_mol_m3[
            state_index
        ] * _transport_state_charge_diffusivity_axis_m2_s(
            transport_state,
            temperature_K,
            axis_index,
        )
    return float(axis_transport_density_mol_m_s)


def _transport_state_charge_diffusivity_axis_m2_s(
    transport_state: TransportState,
    temperature_K: float,
    axis_index: int,
) -> float:
    if axis_index < 0 or axis_index >= AXIS_COUNT:
        raise ValueError(f"axis_index must be in [0, {AXIS_COUNT}), got {axis_index}")
    charge_vector = np.asarray(
        [center.charge for center in transport_state.charged_centers], dtype=float
    )
    if charge_vector.size == 0:
        return 0.0
    resistance_matrix = _transport_state_resistance_matrix_kg_s(
        transport_state, temperature_K
    )
    diffusion_matrix = K_B * temperature_K * np.linalg.inv(resistance_matrix)
    return float(charge_vector @ diffusion_matrix @ charge_vector)


def _transport_state_resistance_matrix_kg_s(
    transport_state: TransportState,
    temperature_K: float,
) -> np.ndarray:
    return _transport_state_local_resistance_matrix_kg_s(
        transport_state, temperature_K
    ) + _transport_state_memory_resistance_matrix_kg_s(transport_state, temperature_K)


def _transport_state_local_resistance_matrix_kg_s(
    transport_state: TransportState,
    temperature_K: float,
) -> np.ndarray:
    center_count = len(transport_state.charged_centers)
    resistance_matrix = np.zeros((center_count, center_count), dtype=float)
    for center_index, charged_center in enumerate(transport_state.charged_centers):
        _assert_positive_finite(
            charged_center.local_diffusion_m2_s,
            f"{charged_center.label}.local_diffusion_m2_s",
        )
        resistance_matrix[center_index, center_index] = (
            K_B * temperature_K / charged_center.local_diffusion_m2_s
        )
    return resistance_matrix


def _transport_state_memory_resistance_matrix_kg_s(
    transport_state: TransportState,
    temperature_K: float,
) -> np.ndarray:
    center_count = len(transport_state.charged_centers)
    resistance_matrix = np.zeros((center_count, center_count), dtype=float)
    for constraint in transport_state.constraints:
        constraint_vector = np.asarray(constraint.vector, dtype=float)
        if constraint_vector.shape != (center_count,):
            raise ValueError(
                f"{transport_state.label}.{constraint.labels} constraint vector length mismatch"
            )
        _assert_nonnegative_finite(
            constraint.lifetime_s,
            f"{transport_state.label}.{constraint.labels}.lifetime_s",
        )
        _assert_positive_finite(
            constraint.length_m, f"{transport_state.label}.{constraint.labels}.length_m"
        )
        constraint_strength_kg_s = (
            K_B
            * temperature_K
            * constraint.lifetime_s
            / (constraint.length_m * constraint.length_m)
        )
        resistance_matrix += constraint_strength_kg_s * np.outer(
            constraint_vector, constraint_vector
        )
    resistance_matrix += _transport_state_atmosphere_resistance_matrix_kg_s(
        transport_state,
        center_count,
    )
    return resistance_matrix


def _transport_state_atmosphere_resistance_matrix_kg_s(
    transport_state: TransportState,
    center_count: int,
) -> np.ndarray:
    atmosphere_resistance_matrix = np.asarray(
        transport_state.atmosphere_resistance_kg_s,
        dtype=float,
    )
    if center_count == 0:
        if atmosphere_resistance_matrix.shape != (0,):
            raise ValueError(
                f"{transport_state.label}.atmosphere_resistance_kg_s must be empty"
            )
        return np.zeros((0, 0), dtype=float)
    if atmosphere_resistance_matrix.shape != (center_count, center_count):
        raise ValueError(
            f"{transport_state.label}.atmosphere_resistance_kg_s shape "
            f"{atmosphere_resistance_matrix.shape} does not match charged-center count {center_count}"
        )
    if not np.all(np.isfinite(atmosphere_resistance_matrix)):
        raise ValueError(
            f"{transport_state.label}.atmosphere_resistance_kg_s contains non-finite values"
        )
    if not np.allclose(atmosphere_resistance_matrix, atmosphere_resistance_matrix.T):
        raise ValueError(
            f"{transport_state.label}.atmosphere_resistance_kg_s must be symmetric"
        )
    eigenvalues = np.linalg.eigvalsh(atmosphere_resistance_matrix)
    if float(np.min(eigenvalues)) < -REVERSE_DIFFUSION_TOLERANCE:
        raise ValueError(
            f"{transport_state.label}.atmosphere_resistance_kg_s must be positive semidefinite"
        )
    return atmosphere_resistance_matrix


def build_kernel_derived_markov_model(
    kernel_state: TransportKernelState,
    temperature_K: float,
    physics_config,
    atmosphere_bath_basis: str = ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL,
    relaxation_dynamic_response: str = RELAXATION_DYNAMIC_RESPONSE_OFF,
    anion_diagonal_relaxation_form_factor: str = ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF,
) -> KernelDerivedMarkovModel:
    _assert_positive_finite(temperature_K, "temperature_K")
    _validate_atmosphere_bath_basis(atmosphere_bath_basis)
    _validate_relaxation_dynamic_response(relaxation_dynamic_response)
    _validate_anion_diagonal_relaxation_form_factor(
        anion_diagonal_relaxation_form_factor
    )

    total_cation_molarity_M = _total_cation_molarity(kernel_state)
    cation_concentration_mol_m3 = total_cation_molarity_M * MOLARITY_TO_MOL_M3
    state_concentration_kernel = _build_state_concentration_kernel(
        kernel_state, temperature_K, physics_config
    )
    chemical_motifs = _chemical_motifs_from_concentration_kernel(
        state_concentration_kernel
    )
    chemical_motif_populations = _normalized_state_population_by_label(
        state_concentration_kernel
    )
    state_concentrations_mol_m3 = (
        state_concentration_kernel.state_concentrations_M * MOLARITY_TO_MOL_M3
    )
    states = _build_motif_states(chemical_motifs)
    state_labels = tuple(_state_label(state) for state in states)
    stationary_probabilities = _state_stationary_probabilities(
        states,
        chemical_motif_populations,
    )

    hydrodynamic_radius_m = _motif_hydrodynamic_radii_m(
        kernel_state,
        chemical_motifs,
    )
    occupied_volume_fraction = kernel_state.mobility.network_occupied_volume_fraction
    crowding_factor = crowding_factor_from_ionic_volume_fraction(
        occupied_volume_fraction
    )
    motif_transport_viscosity_cP = _motif_transport_viscosity_cP(
        kernel_state,
        chemical_motifs,
    )
    vehicular_diffusion_scalar_no_crowding_m2_s = _motif_vehicular_diffusion_m2_s(
        hydrodynamic_radius_m,
        motif_transport_viscosity_cP,
        temperature_K,
    )
    vehicular_diffusion_scalar_m2_s = {
        motif.label: (
            vehicular_diffusion_scalar_no_crowding_m2_s[motif.label]
            * crowding_factor
            / _motif_shape_friction_factor(kernel_state, motif)
        )
        for motif in chemical_motifs
    }
    transition_context = TransitionContext(
        states=states,
        state_labels=state_labels,
        stationary_probabilities=stationary_probabilities,
        hydrodynamic_radius_m=hydrodynamic_radius_m,
        vehicular_diffusion_scalar_m2_s=vehicular_diffusion_scalar_m2_s,
        temperature_K=temperature_K,
    )
    reversible_generator = _build_reversible_generator(transition_context)
    transition_displacements_m = _build_transition_displacements_m(
        len(states),
        reversible_generator.markov_additive_edges,
    )
    bulk_ion_atmosphere_state = _bulk_ion_atmosphere_state(kernel_state, temperature_K)
    transport_state_build = _transport_states(
        kernel_state,
        states,
        hydrodynamic_radius_m,
        vehicular_diffusion_scalar_m2_s,
        temperature_K,
        stationary_probabilities,
        state_concentrations_mol_m3,
        bulk_ion_atmosphere_state,
        atmosphere_bath_basis,
        relaxation_dynamic_response,
        anion_diagonal_relaxation_form_factor,
        physics_config,
    )
    transport_states = transport_state_build[0]
    binding_kinetics = transport_state_build[1]
    state_net_charges = _state_net_charges(transport_states)
    mixture_audit = MixtureKernelAudit(
        dielectric_bruggeman=_bruggeman_dielectric(kernel_state),
        dielectric_effective=kernel_state.matrix.epsilon_effective,
        viscosity_cP=kernel_state.matrix.eta_solution_cP,
        ionic_occupied_volume_fraction=occupied_volume_fraction,
        crowding_factor=crowding_factor,
        anion_shape_factor_by_feature=dict(
            kernel_state.mobility.anion_shape_factor_by_feature
        ),
        cation_microviscosity_coupling_exponent=kernel_state.mobility.cation_microviscosity_coupling_exponent,
        anion_microviscosity_coupling_exponent_by_feature=dict(
            kernel_state.mobility.anion_microviscosity_coupling_exponent_by_feature
        ),
        carrier_strength_Li_mS_cm=_carrier_strength_li_mS_cm(kernel_state),
        carrier_strength_anion_by_feature_mS_cm=_carrier_strength_anion_by_feature_mS_cm(
            kernel_state
        ),
        debye_kappa_inv_m=_debye_kappa_inv_m(
            kernel_state, kernel_state.matrix.epsilon_effective, temperature_K
        ),
        cation_concentration_mol_m3=cation_concentration_mol_m3,
        shell_fractions=dict(kernel_state.solvation.shell_fractions),
        free_fraction_by_feature=dict(kernel_state.speciation.free_fraction_by_feature),
        paired_fraction_by_feature=dict(
            kernel_state.speciation.paired_fraction_by_feature
        ),
        aggregate_fraction_by_feature=dict(
            kernel_state.speciation.aggregate_fraction_by_feature
        ),
    )
    return KernelDerivedMarkovModel(
        chemical_motifs=chemical_motifs,
        states=states,
        state_labels=state_labels,
        stationary_probabilities=stationary_probabilities,
        generator_s_inv=reversible_generator.generator_s_inv,
        capacity_matrix_s_inv=reversible_generator.capacity_matrix_s_inv,
        markov_additive_edges=reversible_generator.markov_additive_edges,
        transport_states=transport_states,
        binding_kinetics=binding_kinetics,
        state_net_charges=state_net_charges,
        transition_displacements_m=transition_displacements_m,
        chemical_motif_populations=chemical_motif_populations,
        state_concentration_kernel=state_concentration_kernel,
        state_concentrations_mol_m3=state_concentrations_mol_m3,
        bulk_ion_atmosphere_state=bulk_ion_atmosphere_state,
        atmosphere_bath_basis=atmosphere_bath_basis,
        relaxation_dynamic_response=relaxation_dynamic_response,
        anion_diagonal_relaxation_form_factor=anion_diagonal_relaxation_form_factor,
        mixture_audit=mixture_audit,
        transition_audit=reversible_generator.transition_audit,
        capacity_evaluation="kramers_asymptotic",
    )


def _build_state_concentration_kernel(
    kernel_state: TransportKernelState,
    temperature_K: float,
    physics_config,
) -> StateConcentrationKernel:
    total_source_molarity_M = sum(
        site.molarity_M for site in kernel_state.site_measure.anion_sites
    )
    _assert_positive_finite(total_source_molarity_M, "total_feature_molarity_M")
    _assert_positive_finite(temperature_K, "temperature_K")
    pairing_config = require_mapping(
        physics_config, "ion_pairing_model", "physics_config"
    )
    aggregate_onset_M = require_float(
        pairing_config, "aggregate_onset_mol_l", "ion_pairing_model"
    )
    aggregate_scale_M = require_float(
        pairing_config, "aggregate_scale_mol_l", "ion_pairing_model"
    )
    aggregate_max_fraction = require_float(
        pairing_config,
        "aggregate_max_fraction_of_paired",
        "ion_pairing_model",
    )
    _assert_positive_finite(
        aggregate_scale_M, "ion_pairing_model.aggregate_scale_mol_l"
    )
    _assert_nonnegative_finite(
        aggregate_max_fraction,
        "ion_pairing_model.aggregate_max_fraction_of_paired",
    )

    species_labels = _state_concentration_species_labels(kernel_state)
    total_concentrations_M = _state_concentration_totals_M(kernel_state, species_labels)
    templates: list[StateMassActionTemplate] = []
    templates.append(
        StateMassActionTemplate(
            chemical_motif=ChemicalMotif("S", ChemicalMotifKind.SOLVENT_CAGE, None),
            stoichiometry=_stoichiometry_vector(
                species_labels,
                {kernel_state.site_measure.cation.canonical_feature_id: 1.0},
            ),
            equilibrium_constant=1.0,
        )
    )
    for anion_site in kernel_state.site_measure.anion_sites:
        anion_feature_id = anion_site.canonical_feature_id
        templates.append(
            StateMassActionTemplate(
                chemical_motif=ChemicalMotif(
                    f"{anion_feature_id}_FREE_ANION",
                    ChemicalMotifKind.FREE_ANION,
                    anion_feature_id,
                ),
                stoichiometry=_stoichiometry_vector(
                    species_labels, {anion_feature_id: 1.0}
                ),
                equilibrium_constant=1.0,
            )
        )
        ssip_association_M_inv, cip_association_M_inv = (
            _state_pair_association_constants_M_inv(
                kernel_state,
                anion_site,
                temperature_K,
            )
        )
        aggregate_association_M_inv = _state_aggregate_association_constant_M_inv(
            total_source_molarity_M,
            aggregate_onset_M,
            aggregate_scale_M,
            aggregate_max_fraction,
        )
        li2a_association_M_inv2, lia2_association_M_inv2 = (
            _state_charged_cluster_association_constants_M_inv2(
                anion_site,
                cip_association_M_inv,
            )
        )
        li2a2_association_M_inv3 = (
            aggregate_association_M_inv * cip_association_M_inv * cip_association_M_inv
        )
        templates.append(
            StateMassActionTemplate(
                chemical_motif=ChemicalMotif(
                    f"{anion_feature_id}_SSIP", ChemicalMotifKind.SSIP, anion_feature_id
                ),
                stoichiometry=_stoichiometry_vector(
                    species_labels,
                    {
                        kernel_state.site_measure.cation.canonical_feature_id: 1.0,
                        anion_feature_id: 1.0,
                    },
                ),
                equilibrium_constant=ssip_association_M_inv,
            )
        )
        templates.append(
            StateMassActionTemplate(
                chemical_motif=ChemicalMotif(
                    f"{anion_feature_id}_CIP", ChemicalMotifKind.CIP, anion_feature_id
                ),
                stoichiometry=_stoichiometry_vector(
                    species_labels,
                    {
                        kernel_state.site_measure.cation.canonical_feature_id: 1.0,
                        anion_feature_id: 1.0,
                    },
                ),
                equilibrium_constant=cip_association_M_inv,
            )
        )
        templates.append(
            StateMassActionTemplate(
                chemical_motif=ChemicalMotif(
                    f"{anion_feature_id}_Li2A_plus",
                    ChemicalMotifKind.LI2A_PLUS,
                    anion_feature_id,
                ),
                stoichiometry=_stoichiometry_vector(
                    species_labels,
                    {
                        kernel_state.site_measure.cation.canonical_feature_id: 2.0,
                        anion_feature_id: 1.0,
                    },
                ),
                equilibrium_constant=li2a_association_M_inv2,
            )
        )
        templates.append(
            StateMassActionTemplate(
                chemical_motif=ChemicalMotif(
                    f"{anion_feature_id}_LiA2_minus",
                    ChemicalMotifKind.LIA2_MINUS,
                    anion_feature_id,
                ),
                stoichiometry=_stoichiometry_vector(
                    species_labels,
                    {
                        kernel_state.site_measure.cation.canonical_feature_id: 1.0,
                        anion_feature_id: 2.0,
                    },
                ),
                equilibrium_constant=lia2_association_M_inv2,
            )
        )
        templates.append(
            StateMassActionTemplate(
                chemical_motif=ChemicalMotif(
                    f"{anion_feature_id}_Li2A2_neutral",
                    ChemicalMotifKind.LI2A2_NEUTRAL,
                    anion_feature_id,
                ),
                stoichiometry=_stoichiometry_vector(
                    species_labels,
                    {
                        kernel_state.site_measure.cation.canonical_feature_id: 2.0,
                        anion_feature_id: 2.0,
                    },
                ),
                equilibrium_constant=li2a2_association_M_inv3,
            )
        )
    for ligand_site in kernel_state.site_measure.neutral_ligand_sites:
        ligand_feature_id = ligand_site.canonical_feature_id
        templates.append(
            StateMassActionTemplate(
                chemical_motif=ChemicalMotif(
                    ligand_feature_id, ChemicalMotifKind.ADDITIVE_COORDINATED, None
                ),
                stoichiometry=_stoichiometry_vector(
                    species_labels,
                    {
                        kernel_state.site_measure.cation.canonical_feature_id: 1.0,
                        ligand_feature_id: 1.0,
                    },
                ),
                equilibrium_constant=ligand_site.coordination_affinity_M_inv,
            )
        )
        for anion_site in kernel_state.site_measure.anion_sites:
            ssip_association_M_inv, _cip_association_M_inv = (
                _state_pair_association_constants_M_inv(
                    kernel_state,
                    anion_site,
                    temperature_K,
                )
            )
            templates.append(
                StateMassActionTemplate(
                    chemical_motif=ChemicalMotif(
                        f"neutral_ligand_shell_{anion_site.canonical_feature_id}_{ligand_feature_id}_SSIP",
                        ChemicalMotifKind.ADDITIVE_SSIP,
                        anion_site.canonical_feature_id,
                    ),
                    stoichiometry=_stoichiometry_vector(
                        species_labels,
                        {
                            kernel_state.site_measure.cation.canonical_feature_id: 1.0,
                            anion_site.canonical_feature_id: 1.0,
                            ligand_feature_id: 1.0,
                        },
                    ),
                    equilibrium_constant=ssip_association_M_inv
                    * ligand_site.coordination_affinity_M_inv,
                )
            )

    stoichiometry = np.asarray(
        [template.stoichiometry for template in templates], dtype=float
    )
    equilibrium_constants = np.asarray(
        [template.equilibrium_constant for template in templates], dtype=float
    )
    for template in templates:
        _assert_positive_finite(
            template.equilibrium_constant,
            f"{template.chemical_motif.label}.equilibrium_constant",
        )
    free_activities_M, state_concentrations_M = _solve_state_concentrations_M(
        species_labels,
        total_concentrations_M,
        stoichiometry,
        equilibrium_constants,
    )
    standard_free_energies_J_mol = np.asarray(
        [
            -R * temperature_K * math.log(template.equilibrium_constant)
            for template in templates
        ],
        dtype=float,
    )
    mass_balance_residuals_M = _state_concentration_mass_balance_residuals_M(
        species_labels,
        total_concentrations_M,
        stoichiometry,
        equilibrium_constants,
        free_activities_M,
        state_concentrations_M,
    )
    return StateConcentrationKernel(
        state_labels=tuple(template.chemical_motif.label for template in templates),
        species_labels=species_labels,
        stoichiometry=stoichiometry,
        standard_free_energies_J_mol=standard_free_energies_J_mol,
        free_activities_M=free_activities_M,
        state_concentrations_M=state_concentrations_M,
        mass_balance_residuals_M=mass_balance_residuals_M,
    )


def _chemical_motifs_from_concentration_kernel(
    state_concentration_kernel: StateConcentrationKernel,
) -> tuple[ChemicalMotif, ...]:
    return tuple(
        _chemical_motif_from_state_label(label)
        for label in state_concentration_kernel.state_labels
    )


def _chemical_motif_from_state_label(state_label: str) -> ChemicalMotif:
    if state_label == "S":
        return ChemicalMotif(state_label, ChemicalMotifKind.SOLVENT_CAGE, None)
    if state_label.endswith("_FREE_ANION"):
        return ChemicalMotif(
            state_label,
            ChemicalMotifKind.FREE_ANION,
            state_label.removesuffix("_FREE_ANION"),
        )
    if state_label.endswith("_SSIP") and state_label.startswith(
        "neutral_ligand_shell_"
    ):
        label_body = state_label.removeprefix("neutral_ligand_shell_").removesuffix(
            "_SSIP"
        )
        feature_id = label_body.split("_neutral_ligand_site_")[0]
        return ChemicalMotif(state_label, ChemicalMotifKind.ADDITIVE_SSIP, feature_id)
    if state_label.endswith("_SSIP"):
        return ChemicalMotif(
            state_label, ChemicalMotifKind.SSIP, state_label.removesuffix("_SSIP")
        )
    if state_label.endswith("_CIP"):
        return ChemicalMotif(
            state_label, ChemicalMotifKind.CIP, state_label.removesuffix("_CIP")
        )
    if state_label.endswith("_Li2A_plus"):
        return ChemicalMotif(
            state_label,
            ChemicalMotifKind.LI2A_PLUS,
            state_label.removesuffix("_Li2A_plus"),
        )
    if state_label.endswith("_LiA2_minus"):
        return ChemicalMotif(
            state_label,
            ChemicalMotifKind.LIA2_MINUS,
            state_label.removesuffix("_LiA2_minus"),
        )
    if state_label.endswith("_Li2A2_neutral"):
        return ChemicalMotif(
            state_label,
            ChemicalMotifKind.LI2A2_NEUTRAL,
            state_label.removesuffix("_Li2A2_neutral"),
        )
    if state_label.startswith("neutral_ligand_site_"):
        return ChemicalMotif(state_label, ChemicalMotifKind.ADDITIVE_COORDINATED, None)
    raise ValueError(f"Cannot infer chemical motif kind from state label {state_label}")


def _normalized_state_population_by_label(
    state_concentration_kernel: StateConcentrationKernel,
) -> dict[str, float]:
    total_concentration_M = float(
        np.sum(state_concentration_kernel.state_concentrations_M)
    )
    _assert_positive_finite(total_concentration_M, "state_concentration_total_M")
    return {
        state_concentration_kernel.state_labels[state_index]: (
            float(state_concentration_kernel.state_concentrations_M[state_index])
            / total_concentration_M
        )
        for state_index in range(len(state_concentration_kernel.state_labels))
    }


def _state_concentration_species_labels(
    kernel_state: TransportKernelState,
) -> tuple[str, ...]:
    return tuple(
        [kernel_state.site_measure.cation.canonical_feature_id]
        + [
            anion_site.canonical_feature_id
            for anion_site in kernel_state.site_measure.anion_sites
        ]
        + [
            ligand_site.canonical_feature_id
            for ligand_site in kernel_state.site_measure.neutral_ligand_sites
        ]
    )


def _state_concentration_totals_M(
    kernel_state: TransportKernelState,
    species_labels: tuple[str, ...],
) -> dict[str, float]:
    totals_M: dict[str, float] = {}
    cation_feature_id = kernel_state.site_measure.cation.canonical_feature_id
    totals_M[cation_feature_id] = _total_cation_molarity(kernel_state)
    for anion_site in kernel_state.site_measure.anion_sites:
        totals_M[anion_site.canonical_feature_id] = anion_site.molarity_M
    for ligand_site in kernel_state.site_measure.neutral_ligand_sites:
        totals_M[ligand_site.canonical_feature_id] = ligand_site.molarity_M
    for species_label in species_labels:
        _assert_positive_finite(
            totals_M[species_label], f"total_concentration_M.{species_label}"
        )
    return totals_M


def _stoichiometry_vector(
    species_labels: tuple[str, ...],
    count_by_species_label: Mapping[str, float],
) -> tuple[float, ...]:
    counts: list[float] = []
    for species_label in count_by_species_label:
        if species_label not in species_labels:
            raise ValueError(
                f"Stoichiometry references unknown species {species_label}"
            )
    for species_label in species_labels:
        if species_label in count_by_species_label:
            count = float(count_by_species_label[species_label])
            _assert_positive_finite(count, f"stoichiometry.{species_label}")
            counts.append(count)
        else:
            counts.append(0.0)
    return tuple(counts)


def _solve_state_concentrations_M(
    species_labels: tuple[str, ...],
    total_concentrations_M: Mapping[str, float],
    stoichiometry: np.ndarray,
    equilibrium_constants: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    totals_vector = np.asarray(
        [
            _require_mapping_float(
                total_concentrations_M, species_label, "total_concentrations_M"
            )
            for species_label in species_labels
        ],
        dtype=float,
    )
    log_free_activities = np.log(totals_vector)
    residual_norm = math.inf
    state_concentrations_M = _state_concentrations_from_log_activities_M(
        log_free_activities,
        stoichiometry,
        equilibrium_constants,
    )
    implicit_free_species_mask = _implicit_free_species_mask(
        stoichiometry, equilibrium_constants
    )
    for _iteration_index in range(MASS_BALANCE_MAX_ITERATIONS):
        free_concentrations_M = np.exp(log_free_activities)
        residual = (
            stoichiometry.T @ state_concentrations_M
            + implicit_free_species_mask * free_concentrations_M
            - totals_vector
        )
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= MASS_BALANCE_ABSOLUTE_TOLERANCE_M:
            free_activities_M = {
                species_labels[species_index]: float(
                    math.exp(log_free_activities[species_index])
                )
                for species_index in range(len(species_labels))
            }
            return free_activities_M, state_concentrations_M
        jacobian = stoichiometry.T @ (state_concentrations_M[:, None] * stoichiometry)
        jacobian += np.diag(implicit_free_species_mask * free_concentrations_M)
        newton_step = np.linalg.solve(jacobian, -residual)
        accepted_step = False
        for damping_index in range(MASS_BALANCE_DAMPING_ATTEMPTS):
            damping_factor = 0.5**damping_index
            candidate_log_free_activities = (
                log_free_activities + damping_factor * newton_step
            )
            candidate_state_concentrations_M = (
                _state_concentrations_from_log_activities_M(
                    candidate_log_free_activities,
                    stoichiometry,
                    equilibrium_constants,
                )
            )
            candidate_free_concentrations_M = np.exp(candidate_log_free_activities)
            candidate_residual = (
                stoichiometry.T @ candidate_state_concentrations_M
                + implicit_free_species_mask * candidate_free_concentrations_M
                - totals_vector
            )
            candidate_residual_norm = float(np.max(np.abs(candidate_residual)))
            if candidate_residual_norm < residual_norm:
                log_free_activities = candidate_log_free_activities
                state_concentrations_M = candidate_state_concentrations_M
                accepted_step = True
                break
        if not accepted_step:
            raise ValueError(
                f"State concentration mass-balance solve stalled at residual {residual_norm} mol/L"
            )
    raise ValueError(
        f"State concentration mass-balance solve failed after {MASS_BALANCE_MAX_ITERATIONS} iterations; "
        f"residual {residual_norm} mol/L"
    )


def _state_concentrations_from_log_activities_M(
    log_free_activities: np.ndarray,
    stoichiometry: np.ndarray,
    equilibrium_constants: np.ndarray,
) -> np.ndarray:
    log_state_concentrations = (
        np.log(equilibrium_constants) + stoichiometry @ log_free_activities
    )
    state_concentrations_M = np.exp(log_state_concentrations)
    if not np.all(np.isfinite(state_concentrations_M)):
        raise ValueError("state concentration solve produced non-finite concentrations")
    return state_concentrations_M


def _implicit_free_species_mask(
    stoichiometry: np.ndarray,
    equilibrium_constants: np.ndarray,
) -> np.ndarray:
    species_count = int(stoichiometry.shape[1])
    has_explicit_free_state = np.zeros(species_count, dtype=bool)
    for state_index in range(int(stoichiometry.shape[0])):
        state_stoichiometry = stoichiometry[state_index]
        nonzero_species = np.flatnonzero(state_stoichiometry)
        if len(nonzero_species) != 1:
            continue
        species_index = int(nonzero_species[0])
        if (
            state_stoichiometry[species_index] == 1.0
            and equilibrium_constants[state_index] == 1.0
        ):
            has_explicit_free_state[species_index] = True
    return np.asarray(
        [
            0.0 if has_explicit_free_state[species_index] else 1.0
            for species_index in range(species_count)
        ],
        dtype=float,
    )


def _state_concentration_mass_balance_residuals_M(
    species_labels: tuple[str, ...],
    total_concentrations_M: Mapping[str, float],
    stoichiometry: np.ndarray,
    equilibrium_constants: np.ndarray,
    free_activities_M: Mapping[str, float],
    state_concentrations_M: np.ndarray,
) -> dict[str, float]:
    implicit_free_species_mask = _implicit_free_species_mask(
        stoichiometry, equilibrium_constants
    )
    free_concentrations_M = np.asarray(
        [
            _require_mapping_float(
                free_activities_M, species_label, "free_activities_M"
            )
            for species_label in species_labels
        ],
        dtype=float,
    )
    occupied_concentrations_M = (
        stoichiometry.T @ state_concentrations_M
        + implicit_free_species_mask * free_concentrations_M
    )
    residuals: dict[str, float] = {}
    for species_index, species_label in enumerate(species_labels):
        residuals[species_label] = _require_mapping_float(
            total_concentrations_M, species_label, "total_concentrations_M"
        ) - float(occupied_concentrations_M[species_index])
    return residuals


def _state_pair_association_constants_M_inv(
    kernel_state: TransportKernelState,
    anion_site,
    temperature_K: float,
) -> tuple[float, float]:
    contact_radius_m, cip_outer_radius_m, ssip_outer_radius_m = _pair_basin_radii_m(
        kernel_state, anion_site
    )
    cip_association_M_inv = _radial_pair_basin_integral_M_inv(
        kernel_state=kernel_state,
        anion_site=anion_site,
        temperature_K=temperature_K,
        dielectric=kernel_state.matrix.epsilon_effective,
        lower_radius_m=contact_radius_m,
        upper_radius_m=cip_outer_radius_m,
    )
    ssip_association_M_inv = _radial_pair_basin_integral_M_inv(
        kernel_state=kernel_state,
        anion_site=anion_site,
        temperature_K=temperature_K,
        dielectric=kernel_state.matrix.epsilon_effective,
        lower_radius_m=cip_outer_radius_m,
        upper_radius_m=ssip_outer_radius_m,
    )
    _assert_positive_finite(
        ssip_association_M_inv,
        f"{anion_site.canonical_feature_id}.ssip_association_M_inv",
    )
    _assert_positive_finite(
        cip_association_M_inv,
        f"{anion_site.canonical_feature_id}.cip_association_M_inv",
    )
    return ssip_association_M_inv, cip_association_M_inv


def _pair_basin_radii_m(
    kernel_state: TransportKernelState,
    anion_site,
) -> tuple[float, float, float]:
    contact_radius_m = (
        anion_site.cation_radius_A + anion_site.anion_radius_A
    ) * ANGSTROM_TO_M
    cation_bare_radius_m = (
        kernel_state.site_measure.cation.ionic_radius_A * ANGSTROM_TO_M
    )
    cation_shell_thickness_m = (
        kernel_state.site_measure.cation.solvated_radius_A
        - kernel_state.site_measure.cation.ionic_radius_A
    ) * ANGSTROM_TO_M
    cip_outer_radius_m = contact_radius_m + cation_bare_radius_m
    ssip_outer_radius_m = contact_radius_m + cation_shell_thickness_m
    _assert_positive_finite(
        contact_radius_m, f"{anion_site.canonical_feature_id}.contact_radius_m"
    )
    _assert_positive_finite(cation_bare_radius_m, "cation_bare_radius_m")
    _assert_positive_finite(cation_shell_thickness_m, "cation_shell_thickness_m")
    if cip_outer_radius_m <= contact_radius_m:
        raise ValueError(
            f"{anion_site.canonical_feature_id}.cip_outer_radius_m must exceed contact_radius_m"
        )
    if ssip_outer_radius_m <= cip_outer_radius_m:
        raise ValueError(
            f"{anion_site.canonical_feature_id}.ssip_outer_radius_m must exceed cip_outer_radius_m"
        )
    return contact_radius_m, cip_outer_radius_m, ssip_outer_radius_m


def _radial_pair_basin_integral_M_inv(
    kernel_state: TransportKernelState,
    anion_site,
    temperature_K: float,
    dielectric: float,
    lower_radius_m: float,
    upper_radius_m: float,
) -> float:
    _assert_positive_finite(temperature_K, "temperature_K")
    _assert_positive_finite(dielectric, "dielectric")
    _assert_positive_finite(lower_radius_m, "lower_radius_m")
    _assert_positive_finite(upper_radius_m, "upper_radius_m")
    if upper_radius_m <= lower_radius_m:
        raise ValueError("upper_radius_m must exceed lower_radius_m")
    debye_kappa_inv_m = _debye_kappa_inv_m(kernel_state, dielectric, temperature_K)
    radius_nodes_m, quadrature_weights = np.polynomial.legendre.leggauss(
        PAIR_BASIN_QUADRATURE_POINTS
    )
    radius_midpoint_m = (lower_radius_m + upper_radius_m) / 2.0
    radius_half_width_m = (upper_radius_m - lower_radius_m) / 2.0
    shifted_radius_nodes_m = radius_midpoint_m + radius_half_width_m * radius_nodes_m
    radial_integral_m3 = 0.0
    for radius_m, quadrature_weight in zip(shifted_radius_nodes_m, quadrature_weights):
        pair_pmf_J = _screened_pair_pmf_J(
            kernel_state=kernel_state,
            anion_site=anion_site,
            dielectric=dielectric,
            debye_kappa_inv_m=debye_kappa_inv_m,
            radius_m=float(radius_m),
        )
        boltzmann_weight = math.exp(-pair_pmf_J / (K_B * temperature_K))
        radial_integral_m3 += (
            float(quadrature_weight) * float(radius_m * radius_m) * boltzmann_weight
        )
    radial_integral_m3 *= radius_half_width_m
    association_M_inv = 4.0 * math.pi * N_A * radial_integral_m3 * MOLARITY_TO_MOL_M3
    _assert_positive_finite(
        association_M_inv,
        f"{anion_site.canonical_feature_id}.radial_basin_integral_M_inv",
    )
    return association_M_inv


def _screened_pair_pmf_J(
    kernel_state: TransportKernelState,
    anion_site,
    dielectric: float,
    debye_kappa_inv_m: float,
    radius_m: float,
) -> float:
    _assert_positive_finite(dielectric, "dielectric")
    _assert_positive_finite(debye_kappa_inv_m, "debye_kappa_inv_m")
    _assert_positive_finite(radius_m, "radius_m")
    elementary_charge_C = F / N_A
    cation_charge = float(kernel_state.site_measure.cation.charge)
    anion_charge = float(anion_site.charge)
    screened_coulomb_J = (
        cation_charge
        * anion_charge
        * elementary_charge_C
        * elementary_charge_C
        / (COULOMB_DENOMINATOR_FACTOR * math.pi * EPS_0 * dielectric * radius_m)
        * math.exp(-radius_m / debye_kappa_inv_m)
    )
    if not math.isfinite(screened_coulomb_J):
        raise ValueError(
            f"{anion_site.canonical_feature_id}.screened_coulomb_J must be finite"
        )
    return screened_coulomb_J


def _state_charged_cluster_association_constants_M_inv2(
    anion_site,
    contact_pair_association_M_inv: float,
) -> tuple[float, float]:
    _assert_positive_finite(
        contact_pair_association_M_inv,
        f"{anion_site.canonical_feature_id}.contact_pair_association_M_inv",
    )
    bridge_eligibility = _anion_capture_accessibility(
        donor_site_count=anion_site.donor_site_count,
        coordination_multiplicity=anion_site.coordination_multiplicity,
        preferred_coordination_number=anion_site.preferred_coordination_number,
        context=anion_site.canonical_feature_id,
    )
    cluster_association_M_inv2 = (
        bridge_eligibility
        * contact_pair_association_M_inv
        * contact_pair_association_M_inv
    )
    _assert_nonnegative_finite(
        cluster_association_M_inv2,
        f"{anion_site.canonical_feature_id}.cluster_association_M_inv2",
    )
    return cluster_association_M_inv2, cluster_association_M_inv2


def _state_aggregate_association_constant_M_inv(
    total_source_molarity_M: float,
    aggregate_onset_M: float,
    aggregate_scale_M: float,
    aggregate_max_fraction: float,
) -> float:
    aggregate_gate = aggregate_max_fraction / (
        1.0
        + math.exp(-(total_source_molarity_M - aggregate_onset_M) / aggregate_scale_M)
    )
    aggregate_association_M_inv = aggregate_gate / aggregate_scale_M
    _assert_nonnegative_finite(
        aggregate_association_M_inv, "aggregate_association_M_inv"
    )
    return aggregate_association_M_inv


def _logsumexp_pair(first_value: float, second_value: float) -> float:
    max_value = max(first_value, second_value)
    return max_value + math.log(
        math.exp(first_value - max_value) + math.exp(second_value - max_value)
    )


def _build_motif_states(
    chemical_motifs: tuple[ChemicalMotif, ...],
) -> tuple[FiniteMotifState, ...]:
    return tuple(
        FiniteMotifState(
            motif=chemical_motif.label,
            orientation="0",
            chemical_motif=chemical_motif,
        )
        for chemical_motif in chemical_motifs
    )


def _state_label(state: FiniteMotifState) -> str:
    if state.orientation == "0":
        return state.motif
    return f"{state.motif}|{state.orientation}"


def _state_stationary_probabilities(
    states: tuple[FiniteMotifState, ...],
    chemical_motif_populations: Mapping[str, float],
) -> np.ndarray:
    stationary_probabilities = np.asarray(
        [
            _require_mapping_float(
                chemical_motif_populations, state.motif, "chemical_motif_populations"
            )
            for state in states
        ],
        dtype=float,
    )
    stationary_probabilities /= float(np.sum(stationary_probabilities))
    return stationary_probabilities


def _build_reversible_generator(
    transition_context: TransitionContext,
) -> ReversibleGenerator:
    state_count = len(transition_context.states)
    capacity_matrix = np.zeros((state_count, state_count), dtype=float)
    generator_matrix = np.zeros((state_count, state_count), dtype=float)
    transition_audit_rows: list[TransitionAuditRow] = []
    markov_additive_edges: list[MarkovAdditiveEdge] = []

    for source_index in range(state_count):
        for target_index in range(source_index + 1, state_count):
            source_state = transition_context.states[source_index]
            target_state = transition_context.states[target_index]
            if not _motif_transition_allowed(
                source_state.chemical_motif, target_state.chemical_motif
            ):
                continue
            pair_capacity = _pair_capacity_s_inv(
                transition_context,
                source_index,
                target_index,
            )
            if pair_capacity <= 0.0:
                continue
            capacity_matrix[source_index, target_index] = pair_capacity
            capacity_matrix[target_index, source_index] = pair_capacity
            source_rate = (
                pair_capacity
                / transition_context.stationary_probabilities[source_index]
            )
            target_rate = (
                pair_capacity
                / transition_context.stationary_probabilities[target_index]
            )
            generator_matrix[source_index, target_index] = source_rate
            generator_matrix[target_index, source_index] = target_rate

    for source_index in range(state_count):
        generator_matrix[source_index, source_index] = -math.fsum(
            float(generator_matrix[source_index, target_index])
            for target_index in range(state_count)
            if target_index != source_index
        )

    for source_index in range(state_count):
        for target_index in range(state_count):
            if source_index == target_index:
                continue
            if generator_matrix[source_index, target_index] <= 0.0:
                continue
            markov_additive_edges.append(
                _motif_exchange_edge(
                    transition_context,
                    source_index,
                    target_index,
                    float(generator_matrix[source_index, target_index]),
                )
            )

    for edge in markov_additive_edges:
        source_state = transition_context.states[edge.source_index]
        target_state = transition_context.states[edge.target_index]
        displacement_norm_m = math.sqrt(
            math.fsum(component * component for component in edge.displacement_m)
        )
        hop_length_m = _hop_length_m(
            transition_context,
            source_state.chemical_motif,
            target_state.chemical_motif,
        )
        if hop_length_m <= 0.0:
            raise ValueError("edge hop length must be positive")
        effective_charge = displacement_norm_m / hop_length_m
        transition_audit_rows.append(
            TransitionAuditRow(
                source_state=transition_context.state_labels[edge.source_index],
                target_state=transition_context.state_labels[edge.target_index],
                capacity_s_inv=float(
                    transition_context.stationary_probabilities[edge.source_index]
                    * edge.rate_s_inv
                ),
                rate_s_inv=float(edge.rate_s_inv),
                charge_displacement_m=edge.displacement_m,
                effective_charge=float(effective_charge),
                hop_length_m=hop_length_m,
            )
        )

    return ReversibleGenerator(
        generator_s_inv=generator_matrix,
        capacity_matrix_s_inv=capacity_matrix,
        markov_additive_edges=tuple(markov_additive_edges),
        transition_audit=tuple(transition_audit_rows),
    )


def _motif_exchange_edge(
    transition_context: TransitionContext,
    source_index: int,
    target_index: int,
    total_rate_s_inv: float,
) -> MarkovAdditiveEdge:
    return MarkovAdditiveEdge(
        source_index=source_index,
        target_index=target_index,
        rate_s_inv=total_rate_s_inv,
        displacement_m=(0.0, 0.0, 0.0),
        label=(
            f"{transition_context.state_labels[source_index]}"
            f"->{transition_context.state_labels[target_index]}:motif_exchange"
        ),
        kind=MarkovAdditiveEdgeKind.MOTIF_EXCHANGE,
    )


def _pair_capacity_s_inv(
    transition_context: TransitionContext,
    source_index: int,
    target_index: int,
) -> float:
    source_state = transition_context.states[source_index]
    target_state = transition_context.states[target_index]
    source_diffusion = transition_context.vehicular_diffusion_scalar_m2_s[
        source_state.motif
    ]
    target_diffusion = transition_context.vehicular_diffusion_scalar_m2_s[
        target_state.motif
    ]
    pair_diffusion = math.sqrt(source_diffusion * target_diffusion)
    hop_length_m = _hop_length_m(
        transition_context,
        source_state.chemical_motif,
        target_state.chemical_motif,
    )
    attempt_rate_s_inv = (
        THREE_DIMENSION_MSD_FACTOR * pair_diffusion / (hop_length_m * hop_length_m)
    )
    population_pair = math.sqrt(
        transition_context.stationary_probabilities[source_index]
        * transition_context.stationary_probabilities[target_index]
    )
    return population_pair * attempt_rate_s_inv


def _motif_transition_allowed(
    source_motif: ChemicalMotif,
    target_motif: ChemicalMotif,
) -> bool:
    if source_motif.label == target_motif.label:
        return True
    if (
        source_motif.kind is ChemicalMotifKind.SOLVENT_CAGE
        and target_motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED
    ):
        return True
    if (
        target_motif.kind is ChemicalMotifKind.SOLVENT_CAGE
        and source_motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED
    ):
        return True
    if _same_feature_pair(
        source_motif, target_motif, ChemicalMotifKind.SSIP, ChemicalMotifKind.CIP
    ):
        return True
    if _same_feature_pair(
        source_motif,
        target_motif,
        ChemicalMotifKind.SSIP,
        ChemicalMotifKind.ADDITIVE_SSIP,
    ):
        return True
    if _feature_to_free_pair(source_motif, target_motif, ChemicalMotifKind.SSIP):
        return True
    if _feature_to_free_pair(
        source_motif, target_motif, ChemicalMotifKind.ADDITIVE_SSIP
    ):
        return True
    if _feature_to_free_pair(source_motif, target_motif, ChemicalMotifKind.CIP):
        return True
    if _feature_to_cluster_pair(source_motif, target_motif):
        return True
    if _feature_to_aggregate_pair(source_motif, target_motif):
        return True
    return False


def _same_feature_pair(
    source_motif: ChemicalMotif,
    target_motif: ChemicalMotif,
    source_kind: ChemicalMotifKind,
    target_kind: ChemicalMotifKind,
) -> bool:
    if source_motif.feature_id != target_motif.feature_id:
        return False
    source_to_target = (
        source_motif.kind is source_kind and target_motif.kind is target_kind
    )
    target_to_source = (
        source_motif.kind is target_kind and target_motif.kind is source_kind
    )
    return source_to_target or target_to_source


def _feature_to_free_pair(
    source_motif: ChemicalMotif,
    target_motif: ChemicalMotif,
    source_kind: ChemicalMotifKind,
) -> bool:
    if (
        source_motif.kind is source_kind
        and target_motif.kind is ChemicalMotifKind.SOLVENT_CAGE
    ):
        return True
    if (
        target_motif.kind is source_kind
        and source_motif.kind is ChemicalMotifKind.SOLVENT_CAGE
    ):
        return True
    if (
        source_motif.kind is source_kind
        and target_motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED
    ):
        return True
    if (
        target_motif.kind is source_kind
        and source_motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED
    ):
        return True
    if (
        source_motif.kind is source_kind
        and target_motif.kind is ChemicalMotifKind.FREE_ANION
    ):
        return source_motif.feature_id == target_motif.feature_id
    if (
        target_motif.kind is source_kind
        and source_motif.kind is ChemicalMotifKind.FREE_ANION
    ):
        return source_motif.feature_id == target_motif.feature_id
    return False


def _feature_to_cluster_pair(
    source_motif: ChemicalMotif,
    target_motif: ChemicalMotif,
) -> bool:
    source_is_cluster = source_motif.kind in (
        ChemicalMotifKind.LI2A_PLUS,
        ChemicalMotifKind.LIA2_MINUS,
        ChemicalMotifKind.LI2A2_NEUTRAL,
    )
    target_is_cluster = target_motif.kind in (
        ChemicalMotifKind.LI2A_PLUS,
        ChemicalMotifKind.LIA2_MINUS,
        ChemicalMotifKind.LI2A2_NEUTRAL,
    )
    if source_is_cluster and target_motif.feature_id == source_motif.feature_id:
        return target_motif.kind in (
            ChemicalMotifKind.SSIP,
            ChemicalMotifKind.CIP,
            ChemicalMotifKind.ADDITIVE_SSIP,
            ChemicalMotifKind.FREE_ANION,
        )
    if target_is_cluster and source_motif.feature_id == target_motif.feature_id:
        return source_motif.kind in (
            ChemicalMotifKind.SSIP,
            ChemicalMotifKind.CIP,
            ChemicalMotifKind.ADDITIVE_SSIP,
            ChemicalMotifKind.FREE_ANION,
        )
    return False


def _feature_to_aggregate_pair(
    source_motif: ChemicalMotif,
    target_motif: ChemicalMotif,
) -> bool:
    source_is_aggregate = source_motif.kind is ChemicalMotifKind.AGGREGATE
    target_is_aggregate = target_motif.kind is ChemicalMotifKind.AGGREGATE
    if source_is_aggregate and target_motif.feature_id is not None:
        return True
    if target_is_aggregate and source_motif.feature_id is not None:
        return True
    return False


def _build_transition_displacements_m(
    state_count: int,
    markov_additive_edges: tuple[MarkovAdditiveEdge, ...],
) -> np.ndarray:
    displacement_tensor = np.zeros((state_count, state_count, AXIS_COUNT), dtype=float)
    rate_by_pair = np.zeros((state_count, state_count), dtype=float)
    for edge in markov_additive_edges:
        if edge.source_index == edge.target_index:
            continue
        rate_by_pair[edge.source_index, edge.target_index] += edge.rate_s_inv
        for axis_index in range(AXIS_COUNT):
            displacement_tensor[edge.source_index, edge.target_index, axis_index] += (
                edge.rate_s_inv * edge.displacement_m[axis_index]
            )
    for source_index in range(state_count):
        for target_index in range(state_count):
            if rate_by_pair[source_index, target_index] <= 0.0:
                continue
            displacement_tensor[source_index, target_index, :] /= rate_by_pair[
                source_index, target_index
            ]
    return displacement_tensor


def _hop_length_m(
    transition_context: TransitionContext,
    source_motif: ChemicalMotif,
    target_motif: ChemicalMotif,
) -> float:
    source_radius = transition_context.hydrodynamic_radius_m[source_motif.label]
    target_radius = transition_context.hydrodynamic_radius_m[target_motif.label]
    return source_radius + target_radius


def _motif_hydrodynamic_radii_m(
    kernel_state: TransportKernelState,
    chemical_motifs: tuple[ChemicalMotif, ...],
) -> dict[str, float]:
    cation_volume_m3 = _sphere_volume_m3(
        kernel_state.site_measure.cation.solvated_radius_A * ANGSTROM_TO_M
    )
    neutral_shell_volume_m3 = (
        kernel_state.solvation.preferred_coordination_number
        * _weighted_neutral_molecular_volume_m3(kernel_state)
    )
    additive_shell_volume_m3 = (
        kernel_state.solvation.preferred_coordination_number
        * _weighted_coordinating_additive_molecular_volume_m3(kernel_state)
    )
    weighted_anion_volume_m3 = _weighted_anion_volume_m3(kernel_state)
    anion_site_by_id = kernel_state.site_measure.anion_by_canonical_id()
    radii: dict[str, float] = {}
    for motif in chemical_motifs:
        cation_count, anion_count = _motif_cation_anion_counts(motif)
        motif_volume_m3 = cation_count * (cation_volume_m3 + neutral_shell_volume_m3)
        if motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED:
            motif_volume_m3 += additive_shell_volume_m3
        if motif.kind is ChemicalMotifKind.ADDITIVE_SSIP:
            motif_volume_m3 += additive_shell_volume_m3
        if motif.feature_id is not None:
            anion_site = anion_site_by_id[motif.feature_id]
            motif_volume_m3 += (
                anion_count
                * require_float(
                    {"anion_volume": anion_site.anion_volume_A3},
                    "anion_volume",
                    f"anion feature {motif.feature_id}",
                )
                * ANGSTROM3_TO_M3
            )
        if motif.kind is ChemicalMotifKind.AGGREGATE:
            motif_volume_m3 += 2.0 * weighted_anion_volume_m3
        _assert_positive_finite(motif_volume_m3, f"{motif.label}.motif_volume_m3")
        radii[motif.label] = _sphere_radius_from_volume_m(motif_volume_m3)
    return radii


def _motif_transport_viscosity_cP(
    kernel_state: TransportKernelState,
    chemical_motifs: tuple[ChemicalMotif, ...],
) -> dict[str, float]:
    reference_viscosity_cP = kernel_state.mobility.reference_viscosity_cP
    _assert_positive_finite(reference_viscosity_cP, "reference_viscosity_cP")
    return {
        motif.label: transport_microviscosity_cP(
            reference_viscosity_cP=reference_viscosity_cP,
            matrix=kernel_state.matrix,
            viscosity_exponent=_motif_transport_viscosity_exponent(kernel_state, motif),
        )
        for motif in chemical_motifs
    }


def _motif_transport_viscosity_exponent(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
) -> float:
    cation_exponent = (
        kernel_state.site_measure.cation.stokes_einstein_alpha
        * kernel_state.mobility.cation_microviscosity_coupling_exponent
    )
    cation_shell_volume_m3 = _cation_shell_volume_m3(kernel_state)
    cation_count, anion_count = _motif_cation_anion_counts(motif)
    weighted_exponent_volume_m3 = (
        cation_count * cation_shell_volume_m3 * cation_exponent
    )
    motif_volume_m3 = cation_count * cation_shell_volume_m3

    if motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED:
        additive_shell_volume_m3 = _additive_shell_volume_m3(kernel_state)
        weighted_exponent_volume_m3 += additive_shell_volume_m3 * cation_exponent
        motif_volume_m3 += additive_shell_volume_m3
    if motif.kind is ChemicalMotifKind.ADDITIVE_SSIP:
        additive_shell_volume_m3 = _additive_shell_volume_m3(kernel_state)
        weighted_exponent_volume_m3 += additive_shell_volume_m3 * cation_exponent
        motif_volume_m3 += additive_shell_volume_m3
    if motif.feature_id is not None:
        anion_volume_m3 = _anion_volume_m3(kernel_state, motif.feature_id)
        anion_exponent = _anion_transport_viscosity_exponent(
            kernel_state, motif.feature_id
        )
        weighted_exponent_volume_m3 += anion_count * anion_volume_m3 * anion_exponent
        motif_volume_m3 += anion_count * anion_volume_m3
    if motif.kind is ChemicalMotifKind.AGGREGATE:
        weighted_anion_volume_m3 = _weighted_anion_volume_m3(kernel_state)
        weighted_anion_exponent = _weighted_anion_transport_viscosity_exponent(
            kernel_state
        )
        weighted_exponent_volume_m3 += (
            2.0 * weighted_anion_volume_m3 * weighted_anion_exponent
        )
        motif_volume_m3 += 2.0 * weighted_anion_volume_m3

    _assert_positive_finite(
        motif_volume_m3, f"{motif.label}.transport_viscosity_volume_m3"
    )
    exponent = weighted_exponent_volume_m3 / motif_volume_m3
    if exponent <= 0.0 or exponent > 1.0 or not math.isfinite(exponent):
        raise ValueError(
            f"{motif.label}.transport_viscosity_exponent must satisfy 0 < exponent <= 1, got {exponent}"
        )
    return exponent


def _cation_shell_volume_m3(kernel_state: TransportKernelState) -> float:
    cation_volume_m3 = _sphere_volume_m3(
        kernel_state.site_measure.cation.solvated_radius_A * ANGSTROM_TO_M
    )
    neutral_shell_volume_m3 = (
        kernel_state.solvation.preferred_coordination_number
        * _weighted_neutral_molecular_volume_m3(kernel_state)
    )
    return cation_volume_m3 + neutral_shell_volume_m3


def _additive_shell_volume_m3(kernel_state: TransportKernelState) -> float:
    return (
        kernel_state.solvation.preferred_coordination_number
        * _weighted_coordinating_additive_molecular_volume_m3(kernel_state)
    )


def _anion_volume_m3(
    kernel_state: TransportKernelState,
    feature_id: str,
) -> float:
    anion_site = kernel_state.site_measure.anion_by_canonical_id()[feature_id]
    return (
        require_float(
            {"anion_volume": anion_site.anion_volume_A3},
            "anion_volume",
            f"anion feature {feature_id}",
        )
        * ANGSTROM3_TO_M3
    )


def _anion_transport_viscosity_exponent(
    kernel_state: TransportKernelState,
    feature_id: str,
) -> float:
    anion_site = kernel_state.site_measure.anion_by_canonical_id()[feature_id]
    exponent = (
        anion_site.stokes_einstein_alpha_anion
        * kernel_state.mobility.anion_microviscosity_coupling_exponent_by_feature[
            feature_id
        ]
    )
    if exponent <= 0.0 or exponent > 1.0 or not math.isfinite(exponent):
        raise ValueError(
            f"{feature_id}.anion_transport_viscosity_exponent must satisfy 0 < exponent <= 1, got {exponent}"
        )
    return exponent


def _weighted_anion_transport_viscosity_exponent(
    kernel_state: TransportKernelState,
) -> float:
    total_molarity = _total_cation_molarity(kernel_state)
    weighted_exponent = 0.0
    for anion_site in kernel_state.site_measure.anion_sites:
        weighted_exponent += (
            anion_site.molarity_M
            / total_molarity
            * _anion_transport_viscosity_exponent(
                kernel_state, anion_site.canonical_feature_id
            )
        )
    if (
        weighted_exponent <= 0.0
        or weighted_exponent > 1.0
        or not math.isfinite(weighted_exponent)
    ):
        raise ValueError(
            f"weighted anion transport viscosity exponent must satisfy 0 < exponent <= 1, got {weighted_exponent}"
        )
    return weighted_exponent


def _motif_vehicular_diffusion_m2_s(
    hydrodynamic_radius_m: Mapping[str, float],
    viscosity_cP_by_motif: Mapping[str, float],
    temperature_K: float,
) -> dict[str, float]:
    diffusion_by_label: dict[str, float] = {}
    for label, radius_m in hydrodynamic_radius_m.items():
        _assert_positive_finite(radius_m, f"hydrodynamic_radius_m.{label}")
        viscosity_cP = _require_mapping_float(
            viscosity_cP_by_motif, label, "viscosity_cP_by_motif"
        )
        _assert_positive_finite(viscosity_cP, f"viscosity_cP_by_motif.{label}")
        eta_Pa_s = viscosity_cP * CP_TO_PA_S
        diffusion_by_label[label] = (
            K_B
            * temperature_K
            / (STOKES_SPHERE_DRAG_FACTOR * math.pi * eta_Pa_s * radius_m)
        )
    return diffusion_by_label


def _motif_shape_friction_factor(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
) -> float:
    if motif.feature_id is not None:
        anion_site = kernel_state.site_measure.anion_by_canonical_id()[motif.feature_id]
        return anion_site.shape_friction_factor
    if motif.kind is ChemicalMotifKind.AGGREGATE:
        return _weighted_anion_shape_friction_factor(kernel_state)
    return 1.0


def _motif_cation_anion_counts(motif: ChemicalMotif) -> tuple[float, float]:
    if motif.kind is ChemicalMotifKind.SOLVENT_CAGE:
        return 1.0, 0.0
    if motif.kind is ChemicalMotifKind.FREE_ANION:
        return 0.0, 1.0
    if motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED:
        return 1.0, 0.0
    if motif.kind in (
        ChemicalMotifKind.SSIP,
        ChemicalMotifKind.CIP,
        ChemicalMotifKind.ADDITIVE_SSIP,
    ):
        return 1.0, 1.0
    if motif.kind is ChemicalMotifKind.LI2A_PLUS:
        return 2.0, 1.0
    if motif.kind is ChemicalMotifKind.LIA2_MINUS:
        return 1.0, 2.0
    if motif.kind is ChemicalMotifKind.LI2A2_NEUTRAL:
        return 2.0, 2.0
    if motif.kind is ChemicalMotifKind.BRIDGE_NETWORK:
        return 2.0, 1.0
    if motif.kind is ChemicalMotifKind.AGGREGATE:
        return 1.0, 2.0
    raise ValueError(f"Unhandled motif kind {motif.kind}")


def _weighted_anion_shape_friction_factor(kernel_state: TransportKernelState) -> float:
    total_molarity = _total_cation_molarity(kernel_state)
    weighted_factor = 0.0
    for anion_site in kernel_state.site_measure.anion_sites:
        weighted_factor += (
            anion_site.molarity_M / total_molarity * anion_site.shape_friction_factor
        )
    _assert_positive_finite(weighted_factor, "weighted_anion_shape_friction_factor")
    return weighted_factor


def _carrier_strength_li_mS_cm(
    kernel_state: TransportKernelState,
) -> float:
    cation_symbol = kernel_state.site_measure.cation.ion_symbol
    return _require_mapping_float(
        kernel_state.mobility.carrier_strength_mS_cm,
        cation_symbol,
        "kernel_state.mobility.carrier_strength_mS_cm",
    )


def _carrier_strength_anion_by_feature_mS_cm(
    kernel_state: TransportKernelState,
) -> dict[str, float]:
    strengths: dict[str, float] = {}
    for anion_site in kernel_state.site_measure.anion_sites:
        strengths[anion_site.canonical_feature_id] = _require_mapping_float(
            kernel_state.mobility.carrier_strength_mS_cm,
            anion_site.carrier_label,
            "kernel_state.mobility.carrier_strength_mS_cm",
        )
    return strengths


def _cation_center_diffusivity_m2_s(
    kernel_state: TransportKernelState,
    temperature_K: float,
) -> float:
    cation_symbol = kernel_state.site_measure.cation.ion_symbol
    lambda_S_cm2_mol = _require_mapping_float(
        kernel_state.mobility.carrier_lambda_S_cm2_mol,
        cation_symbol,
        "kernel_state.mobility.carrier_lambda_S_cm2_mol",
    )
    return _diffusivity_from_molar_conductivity_m2_s(
        lambda_S_cm2_mol,
        float(kernel_state.site_measure.cation.charge),
        temperature_K,
    )


def _anion_charged_center_by_feature(
    kernel_state: TransportKernelState,
    temperature_K: float,
) -> dict[str, ChargedCenter]:
    diffusivity_by_feature = _anion_center_diffusivity_by_feature_m2_s(
        kernel_state,
        temperature_K,
    )
    charged_center_by_feature: dict[str, ChargedCenter] = {}
    for anion_site in kernel_state.site_measure.anion_sites:
        feature_id = anion_site.canonical_feature_id
        charged_center_by_feature[feature_id] = ChargedCenter(
            label=f"{feature_id}:anion",
            charge=float(anion_site.charge),
            hydrodynamic_radius_m=anion_site.anion_radius_A * ANGSTROM_TO_M,
            shape_factor=anion_site.shape_friction_factor,
            local_diffusion_m2_s=_require_mapping_float(
                diffusivity_by_feature,
                feature_id,
                "anion_center_diffusivity_by_feature_m2_s",
            ),
            relative_position_m=(0.0, 0.0, 0.0),
            charge_cloud_radius_available=anion_site.charge_cloud_radius_available,
            charge_cloud_radius_A=anion_site.charge_cloud_radius_A,
            charge_cloud_source=anion_site.charge_cloud_source,
            charge_cloud_site_count=anion_site.charge_cloud_site_count,
        )
    return charged_center_by_feature


def _transport_states(
    kernel_state: TransportKernelState,
    states: tuple[FiniteMotifState, ...],
    hydrodynamic_radius_m: Mapping[str, float],
    state_cation_diffusion_m2_s: Mapping[str, float],
    temperature_K: float,
    stationary_probabilities: np.ndarray,
    state_concentrations_mol_m3: np.ndarray,
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    atmosphere_bath_basis: str,
    relaxation_dynamic_response: str,
    anion_diagonal_relaxation_form_factor: str,
    physics_config,
) -> tuple[tuple[TransportState, ...], tuple[MotifBindingKinetics, ...]]:
    _validate_atmosphere_bath_basis(atmosphere_bath_basis)
    _validate_relaxation_dynamic_response(relaxation_dynamic_response)
    _validate_anion_diagonal_relaxation_form_factor(
        anion_diagonal_relaxation_form_factor
    )
    cation_center_label = kernel_state.site_measure.cation.canonical_feature_id
    weighted_anion_center = ChargedCenter(
        label="weighted_anion",
        charge=_weighted_anion_charge(kernel_state),
        hydrodynamic_radius_m=_weighted_anion_radius_m(kernel_state),
        shape_factor=_weighted_anion_shape_friction_factor(kernel_state),
        local_diffusion_m2_s=_weighted_anion_center_diffusivity_m2_s(
            kernel_state,
            temperature_K,
        ),
        relative_position_m=(0.0, 0.0, 0.0),
        charge_cloud_radius_available=False,
        charge_cloud_radius_A=0.0,
        charge_cloud_source=CHARGE_CLOUD_SOURCE_WEIGHTED_MISSING,
        charge_cloud_site_count=0,
    )
    anion_center_by_feature = _anion_charged_center_by_feature(
        kernel_state,
        temperature_K,
    )
    transport_states: list[TransportState] = []
    binding_kinetics_rows: list[MotifBindingKinetics] = []
    for state_index, state in enumerate(states):
        motif = state.chemical_motif
        cation_center = ChargedCenter(
            label=cation_center_label,
            charge=float(kernel_state.site_measure.cation.charge),
            hydrodynamic_radius_m=_require_mapping_float(
                hydrodynamic_radius_m,
                motif.label,
                "hydrodynamic_radius_m",
            ),
            shape_factor=_motif_shape_friction_factor(kernel_state, motif),
            local_diffusion_m2_s=_require_mapping_float(
                state_cation_diffusion_m2_s,
                motif.label,
                "state_cation_diffusion_m2_s",
            ),
            relative_position_m=(0.0, 0.0, 0.0),
            charge_cloud_radius_available=False,
            charge_cloud_radius_A=0.0,
            charge_cloud_source=CHARGE_CLOUD_SOURCE_NOT_APPLICABLE,
            charge_cloud_site_count=0,
        )
        if motif.kind is ChemicalMotifKind.SOLVENT_CAGE:
            charged_centers = (cation_center,)
            constraints = ()
        elif motif.kind is ChemicalMotifKind.FREE_ANION:
            feature_id = _require_feature_id(motif)
            charged_centers = (anion_center_by_feature[feature_id],)
            constraints = ()
        elif motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED:
            charged_centers = (cation_center,)
            constraints = ()
        elif motif.kind in (
            ChemicalMotifKind.SSIP,
            ChemicalMotifKind.CIP,
            ChemicalMotifKind.ADDITIVE_SSIP,
        ):
            feature_id = _require_feature_id(motif)
            charged_centers = (cation_center, anion_center_by_feature[feature_id])
            constraints = _constraint_modes_for_state(
                kernel_state,
                motif,
                charged_centers,
                binding_kinetics_rows,
                temperature_K,
                physics_config,
            )
        elif motif.kind is ChemicalMotifKind.LI2A_PLUS:
            feature_id = _require_feature_id(motif)
            charged_centers = (
                _renamed_charged_center(cation_center, f"{cation_center_label}_site_0"),
                _renamed_charged_center(cation_center, f"{cation_center_label}_site_1"),
                anion_center_by_feature[feature_id],
            )
            constraints = _constraint_modes_for_state(
                kernel_state,
                motif,
                charged_centers,
                binding_kinetics_rows,
                temperature_K,
                physics_config,
            )
        elif motif.kind is ChemicalMotifKind.LIA2_MINUS:
            feature_id = _require_feature_id(motif)
            anion_center = anion_center_by_feature[feature_id]
            charged_centers = (
                cation_center,
                _renamed_charged_center(anion_center, f"{feature_id}:anion_site_0"),
                _renamed_charged_center(anion_center, f"{feature_id}:anion_site_1"),
            )
            constraints = _constraint_modes_for_state(
                kernel_state,
                motif,
                charged_centers,
                binding_kinetics_rows,
                temperature_K,
                physics_config,
            )
        elif motif.kind is ChemicalMotifKind.LI2A2_NEUTRAL:
            feature_id = _require_feature_id(motif)
            anion_center = anion_center_by_feature[feature_id]
            charged_centers = (
                _renamed_charged_center(cation_center, f"{cation_center_label}_site_0"),
                _renamed_charged_center(cation_center, f"{cation_center_label}_site_1"),
                _renamed_charged_center(anion_center, f"{feature_id}:anion_site_0"),
                _renamed_charged_center(anion_center, f"{feature_id}:anion_site_1"),
            )
            constraints = _constraint_modes_for_state(
                kernel_state,
                motif,
                charged_centers,
                binding_kinetics_rows,
                temperature_K,
                physics_config,
            )
        elif motif.feature_id is None and motif.kind is ChemicalMotifKind.AGGREGATE:
            charged_centers = (
                cation_center,
                _renamed_charged_center(weighted_anion_center, "weighted_anion_site_0"),
                _renamed_charged_center(weighted_anion_center, "weighted_anion_site_1"),
            )
            constraints = _constraint_modes_for_state(
                kernel_state,
                motif,
                charged_centers,
                binding_kinetics_rows,
                temperature_K,
                physics_config,
            )
        else:
            raise ValueError(f"Unhandled motif kind for transport state {motif.kind}")
        charged_centers = _charged_centers_with_state_geometry(
            kernel_state,
            motif,
            charged_centers,
        )
        state_atmosphere_resistance = _state_atmosphere_resistance(
            kernel_state=kernel_state,
            charged_centers=charged_centers,
            constraints=constraints,
            bulk_ion_atmosphere_state=bulk_ion_atmosphere_state,
            state_concentration_mol_m3=float(state_concentrations_mol_m3[state_index]),
            atmosphere_bath_basis=atmosphere_bath_basis,
            relaxation_dynamic_response=relaxation_dynamic_response,
            anion_diagonal_relaxation_form_factor=anion_diagonal_relaxation_form_factor,
            motif_kind=motif.kind,
            temperature_K=temperature_K,
        )
        transport_states.append(
            TransportState(
                label=state.motif,
                probability=float(stationary_probabilities[state_index]),
                concentration_mol_m3=float(state_concentrations_mol_m3[state_index]),
                charged_centers=charged_centers,
                constraints=constraints,
                atmosphere_resistance_kg_s=state_atmosphere_resistance.gated_resistance_kg_s,
                atmosphere_resistance_before_lifetime_gate_kg_s=(
                    state_atmosphere_resistance.ungated_resistance_kg_s
                ),
                atmosphere_state_lifetime_s=state_atmosphere_resistance.state_lifetime_s,
                atmosphere_relaxation_time_s=state_atmosphere_resistance.relaxation_time_s,
                atmosphere_lifetime_gate=state_atmosphere_resistance.applied_lifetime_gate,
                atmosphere_diagnostic_lifetime_gate=(
                    state_atmosphere_resistance.diagnostic_lifetime_gate
                ),
                relaxation_dynamic_response=(
                    state_atmosphere_resistance.relaxation_dynamic_response
                ),
                anion_diagonal_relaxation_form_factor=(
                    state_atmosphere_resistance.anion_diagonal_relaxation_form_factor
                ),
                relaxation_lifetime_gate=state_atmosphere_resistance.relaxation_lifetime_gate,
                relaxation_resistance_before_gate_kg_s=(
                    state_atmosphere_resistance.relaxation_resistance_before_gate_kg_s
                ),
                relaxation_resistance_after_gate_kg_s=(
                    state_atmosphere_resistance.relaxation_resistance_after_gate_kg_s
                ),
                atmosphere_bath_basis=state_atmosphere_resistance.atmosphere_bath_basis,
                ionic_strength_total_mol_m3=state_atmosphere_resistance.ionic_strength_total_mol_m3,
                ionic_strength_external_mol_m3=(
                    state_atmosphere_resistance.ionic_strength_external_mol_m3
                ),
                external_over_total_ionic_strength=(
                    state_atmosphere_resistance.external_over_total_ionic_strength
                ),
                free_energy_J_mol=_transport_state_free_energy_J_mol(
                    stationary_probabilities[state_index],
                    temperature_K,
                ),
            )
        )
    return tuple(transport_states), tuple(binding_kinetics_rows)


def _state_atmosphere_resistance(
    kernel_state: TransportKernelState,
    charged_centers: tuple[ChargedCenter, ...],
    constraints: tuple[ConstraintMode, ...],
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    state_concentration_mol_m3: float,
    atmosphere_bath_basis: str,
    relaxation_dynamic_response: str,
    anion_diagonal_relaxation_form_factor: str,
    motif_kind: ChemicalMotifKind,
    temperature_K: float,
) -> StateAtmosphereResistance:
    _validate_atmosphere_bath_basis(atmosphere_bath_basis)
    _validate_relaxation_dynamic_response(relaxation_dynamic_response)
    _validate_anion_diagonal_relaxation_form_factor(
        anion_diagonal_relaxation_form_factor
    )
    state_bulk_ion_atmosphere_state = _state_bulk_ion_atmosphere_state(
        kernel_state=kernel_state,
        charged_centers=charged_centers,
        state_concentration_mol_m3=state_concentration_mol_m3,
        recipe_bulk_ion_atmosphere_state=bulk_ion_atmosphere_state,
        atmosphere_bath_basis=atmosphere_bath_basis,
        temperature_K=temperature_K,
    )
    total_ionic_strength_mol_m3 = _formal_ionic_strength_mol_m3(kernel_state)
    external_ionic_strength_mol_m3 = (
        state_bulk_ion_atmosphere_state.ionic_strength_mol_m3
    )
    external_over_total_ionic_strength = _nonnegative_ratio(
        external_ionic_strength_mol_m3,
        total_ionic_strength_mol_m3,
        "external_over_total_ionic_strength",
    )
    if not charged_centers:
        state_lifetime_s = _state_atmosphere_lifetime_s(constraints)
        relaxation_time_s = _debye_falkenhagen_relaxation_time_s(
            state_bulk_ion_atmosphere_state.kappa_inv_m,
            state_bulk_ion_atmosphere_state.ambipolar_diffusivity_m2_s,
        )
        diagnostic_lifetime_gate = debye_falkenhagen_lifetime_gate(
            state_lifetime_s,
            relaxation_time_s,
        )
        return StateAtmosphereResistance(
            gated_resistance_kg_s=(),
            ungated_resistance_kg_s=(),
            state_lifetime_s=state_lifetime_s,
            relaxation_time_s=relaxation_time_s,
            applied_lifetime_gate=1.0,
            diagnostic_lifetime_gate=diagnostic_lifetime_gate,
            relaxation_dynamic_response=relaxation_dynamic_response,
            anion_diagonal_relaxation_form_factor=anion_diagonal_relaxation_form_factor,
            relaxation_lifetime_gate=1.0,
            relaxation_resistance_before_gate_kg_s=(),
            relaxation_resistance_after_gate_kg_s=(),
            atmosphere_bath_basis=atmosphere_bath_basis,
            ionic_strength_total_mol_m3=total_ionic_strength_mol_m3,
            ionic_strength_external_mol_m3=external_ionic_strength_mol_m3,
            external_over_total_ionic_strength=external_over_total_ionic_strength,
        )
    carrier_index_by_label = {
        carrier_label: carrier_index
        for carrier_index, carrier_label in enumerate(
            state_bulk_ion_atmosphere_state.carrier_labels
        )
    }
    electrophoretic_single_center_resistance_values_kg_s = []
    relaxation_single_center_resistance_values_kg_s = []
    for center_index, charged_center in enumerate(charged_centers):
        projection_vector = _charged_center_bulk_projection_vector(
            kernel_state,
            charged_center,
            carrier_index_by_label,
        )
        electrophoretic_single_center_resistance_values_kg_s.append(
            float(
                projection_vector
                @ state_bulk_ion_atmosphere_state.resistance_ep_kg_s
                @ projection_vector
            )
        )
        relaxation_single_center_resistance_values_kg_s.append(
            float(
                projection_vector
                @ state_bulk_ion_atmosphere_state.resistance_rel_kg_s
                @ projection_vector
            )
        )
    electrophoretic_atmosphere_matrix = state_form_factor_atmosphere_resistance_kg_s(
        charged_centers=charged_centers,
        single_center_atmosphere_resistance_kg_s=tuple(
            electrophoretic_single_center_resistance_values_kg_s
        ),
        kappa_inv_m=state_bulk_ion_atmosphere_state.kappa_inv_m,
    )
    point_relaxation_atmosphere_matrix = state_form_factor_atmosphere_resistance_kg_s(
        charged_centers=charged_centers,
        single_center_atmosphere_resistance_kg_s=tuple(
            relaxation_single_center_resistance_values_kg_s
        ),
        kappa_inv_m=state_bulk_ion_atmosphere_state.kappa_inv_m,
    )
    relaxation_atmosphere_matrix = _finite_size_anion_diagonal_relaxation_matrix_kg_s(
        charged_centers,
        point_relaxation_atmosphere_matrix,
        motif_kind,
        state_bulk_ion_atmosphere_state.kappa_inv_m,
        anion_diagonal_relaxation_form_factor,
    )
    state_lifetime_s = _state_atmosphere_lifetime_s(constraints)
    relaxation_time_s = _debye_falkenhagen_relaxation_time_s(
        state_bulk_ion_atmosphere_state.kappa_inv_m,
        state_bulk_ion_atmosphere_state.ambipolar_diffusivity_m2_s,
    )
    diagnostic_lifetime_gate = debye_falkenhagen_lifetime_gate(
        state_lifetime_s,
        relaxation_time_s,
    )
    relaxation_lifetime_gate = _relaxation_component_lifetime_gate(
        relaxation_dynamic_response,
        diagnostic_lifetime_gate,
    )
    applied_lifetime_gate = 1.0
    relaxation_after_gate_matrix = (
        relaxation_lifetime_gate * relaxation_atmosphere_matrix
    )
    atmosphere_matrix = electrophoretic_atmosphere_matrix + relaxation_atmosphere_matrix
    gated_atmosphere_matrix = (
        electrophoretic_atmosphere_matrix + relaxation_after_gate_matrix
    )
    _validate_form_factor_atmosphere_matrix(gated_atmosphere_matrix)
    return StateAtmosphereResistance(
        gated_resistance_kg_s=_matrix_to_tuple(gated_atmosphere_matrix),
        ungated_resistance_kg_s=_matrix_to_tuple(atmosphere_matrix),
        state_lifetime_s=state_lifetime_s,
        relaxation_time_s=relaxation_time_s,
        applied_lifetime_gate=applied_lifetime_gate,
        diagnostic_lifetime_gate=diagnostic_lifetime_gate,
        relaxation_dynamic_response=relaxation_dynamic_response,
        anion_diagonal_relaxation_form_factor=anion_diagonal_relaxation_form_factor,
        relaxation_lifetime_gate=relaxation_lifetime_gate,
        relaxation_resistance_before_gate_kg_s=_matrix_to_tuple(
            relaxation_atmosphere_matrix
        ),
        relaxation_resistance_after_gate_kg_s=_matrix_to_tuple(
            relaxation_after_gate_matrix
        ),
        atmosphere_bath_basis=atmosphere_bath_basis,
        ionic_strength_total_mol_m3=total_ionic_strength_mol_m3,
        ionic_strength_external_mol_m3=external_ionic_strength_mol_m3,
        external_over_total_ionic_strength=external_over_total_ionic_strength,
    )


def _matrix_to_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _finite_size_anion_diagonal_relaxation_matrix_kg_s(
    charged_centers: tuple[ChargedCenter, ...],
    point_relaxation_atmosphere_matrix_kg_s: np.ndarray,
    motif_kind: ChemicalMotifKind,
    kappa_inv_m: float,
    anion_diagonal_relaxation_form_factor: str,
) -> np.ndarray:
    _validate_anion_diagonal_relaxation_form_factor(
        anion_diagonal_relaxation_form_factor
    )
    center_count = len(charged_centers)
    relaxation_atmosphere_matrix = np.asarray(
        point_relaxation_atmosphere_matrix_kg_s,
        dtype=float,
    )
    if relaxation_atmosphere_matrix.shape != (center_count, center_count):
        raise ValueError(
            "point relaxation atmosphere matrix shape must match charged centers"
        )
    finite_size_matrix = np.array(relaxation_atmosphere_matrix, dtype=float, copy=True)
    if (
        anion_diagonal_relaxation_form_factor
        == ANION_DIAGONAL_RELAXATION_FORM_FACTOR_OFF
    ):
        return finite_size_matrix
    if not _motif_kind_has_resolved_anion_diagonal_form_factor(motif_kind):
        return finite_size_matrix

    for center_index, charged_center in enumerate(charged_centers):
        point_resistance_kg_s = float(finite_size_matrix[center_index, center_index])
        _assert_nonnegative_finite(
            point_resistance_kg_s,
            f"{charged_center.label}.point_relaxation_resistance_kg_s",
        )
        if charged_center.charge >= 0.0:
            continue
        self_form_factor = _anion_diagonal_relaxation_self_form_factor(
            charged_center,
            kappa_inv_m,
        )
        finite_size_resistance_kg_s = point_resistance_kg_s * self_form_factor
        if finite_size_resistance_kg_s > point_resistance_kg_s:
            raise ValueError(
                "finite-size anion diagonal relaxation increased point relaxation resistance"
            )
        finite_size_matrix[center_index, center_index] = finite_size_resistance_kg_s
    _validate_form_factor_atmosphere_matrix(finite_size_matrix)
    return finite_size_matrix


def _motif_kind_has_resolved_anion_diagonal_form_factor(
    motif_kind: ChemicalMotifKind,
) -> bool:
    return motif_kind in (
        ChemicalMotifKind.SSIP,
        ChemicalMotifKind.ADDITIVE_SSIP,
        ChemicalMotifKind.CIP,
        ChemicalMotifKind.AGGREGATE,
        ChemicalMotifKind.LI2A_PLUS,
        ChemicalMotifKind.LIA2_MINUS,
        ChemicalMotifKind.LI2A2_NEUTRAL,
    )


def _anion_diagonal_relaxation_self_form_factor(
    charged_center: ChargedCenter,
    kappa_inv_m: float,
) -> float:
    if charged_center.charge >= 0.0:
        raise ValueError(
            f"{charged_center.label} is not an anion center for diagonal relaxation form factor"
        )
    if math.isinf(kappa_inv_m):
        return 1.0
    _assert_positive_finite(kappa_inv_m, "kappa_inv_m")
    _assert_positive_finite(
        charged_center.hydrodynamic_radius_m,
        f"{charged_center.label}.hydrodynamic_radius_m",
    )
    _assert_positive_finite(
        charged_center.shape_factor, f"{charged_center.label}.shape_factor"
    )
    effective_hydrodynamic_proxy_radius_m = (
        charged_center.hydrodynamic_radius_m * charged_center.shape_factor
    )
    form_factor_argument = effective_hydrodynamic_proxy_radius_m / kappa_inv_m
    self_form_factor = math.exp(
        -(
            form_factor_argument
            * form_factor_argument
            / GAUSSIAN_SELF_FORM_FACTOR_SQUARED_DENOMINATOR
        )
    )
    if (
        self_form_factor < 0.0
        or self_form_factor > 1.0
        or not math.isfinite(self_form_factor)
    ):
        raise ValueError(
            f"anion diagonal relaxation self form factor must be in [0, 1], got {self_form_factor}"
        )
    return self_form_factor


def electrostatic_charge_cloud_self_form_factor_squared(
    charge_cloud_radius_A: float,
    kappa_inv_m: float,
) -> float:
    _assert_nonnegative_finite(charge_cloud_radius_A, "charge_cloud_radius_A")
    if charge_cloud_radius_A == 0.0 or math.isinf(kappa_inv_m):
        return 1.0
    _assert_positive_finite(kappa_inv_m, "kappa_inv_m")
    charge_cloud_radius_m = charge_cloud_radius_A * ANGSTROM_TO_M
    form_factor_argument = charge_cloud_radius_m / kappa_inv_m
    form_factor_squared = math.exp(
        -(
            form_factor_argument
            * form_factor_argument
            / GAUSSIAN_SELF_FORM_FACTOR_SQUARED_DENOMINATOR
        )
    )
    if (
        form_factor_squared < 0.0
        or form_factor_squared > 1.0
        or not math.isfinite(form_factor_squared)
    ):
        raise ValueError(
            f"charge-cloud self form factor squared must be in [0, 1], got {form_factor_squared}"
        )
    return form_factor_squared


def _state_atmosphere_lifetime_s(
    constraints: tuple[ConstraintMode, ...],
) -> float:
    if not constraints:
        return math.inf
    finite_lifetimes_s = []
    for constraint in constraints:
        _assert_nonnegative_finite(
            constraint.atmosphere_lifetime_s,
            f"{constraint.labels}.atmosphere_lifetime_s",
        )
        finite_lifetimes_s.append(constraint.atmosphere_lifetime_s)
    return min(finite_lifetimes_s)


def _debye_falkenhagen_relaxation_time_s(
    kappa_inv_m: float,
    ambipolar_diffusivity_m2_s: float,
) -> float:
    if kappa_inv_m <= 0.0 or (
        not math.isfinite(kappa_inv_m) and not math.isinf(kappa_inv_m)
    ):
        raise ValueError(f"kappa_inv_m must be positive or infinite, got {kappa_inv_m}")
    _assert_nonnegative_finite(
        ambipolar_diffusivity_m2_s,
        "ambipolar_diffusivity_m2_s",
    )
    if math.isinf(kappa_inv_m) or ambipolar_diffusivity_m2_s == 0.0:
        return math.inf
    return (kappa_inv_m * kappa_inv_m) / ambipolar_diffusivity_m2_s


def debye_falkenhagen_lifetime_gate(
    state_lifetime_s: float,
    atmosphere_relaxation_time_s: float,
) -> float:
    if state_lifetime_s < 0.0 or (
        not math.isfinite(state_lifetime_s) and not math.isinf(state_lifetime_s)
    ):
        raise ValueError(
            f"state_lifetime_s must be nonnegative or infinite, got {state_lifetime_s}"
        )
    if atmosphere_relaxation_time_s < 0.0 or (
        not math.isfinite(atmosphere_relaxation_time_s)
        and not math.isinf(atmosphere_relaxation_time_s)
    ):
        raise ValueError(
            "atmosphere_relaxation_time_s must be nonnegative or infinite, "
            f"got {atmosphere_relaxation_time_s}"
        )
    if math.isinf(state_lifetime_s):
        return 1.0
    if state_lifetime_s == 0.0:
        return 0.0
    if atmosphere_relaxation_time_s == 0.0:
        return 1.0
    if math.isinf(atmosphere_relaxation_time_s):
        return 0.0
    lifetime_gate = state_lifetime_s / (state_lifetime_s + atmosphere_relaxation_time_s)
    if lifetime_gate < 0.0 or lifetime_gate > 1.0 or not math.isfinite(lifetime_gate):
        raise ValueError(
            f"Debye-Falkenhagen lifetime gate must be in [0, 1], got {lifetime_gate}"
        )
    return lifetime_gate


def _relaxation_component_lifetime_gate(
    relaxation_dynamic_response: str,
    diagnostic_lifetime_gate: float,
) -> float:
    _validate_relaxation_dynamic_response(relaxation_dynamic_response)
    if relaxation_dynamic_response == RELAXATION_DYNAMIC_RESPONSE_OFF:
        return 1.0
    if relaxation_dynamic_response == RELAXATION_DYNAMIC_RESPONSE_STATE_LIFETIME:
        if diagnostic_lifetime_gate < 0.0 or diagnostic_lifetime_gate > 1.0:
            raise ValueError(
                f"diagnostic_lifetime_gate must be in [0, 1], got {diagnostic_lifetime_gate}"
            )
        return diagnostic_lifetime_gate
    raise ValueError(
        f"Unsupported relaxation_dynamic_response {relaxation_dynamic_response!r}"
    )


def state_form_factor_atmosphere_resistance_kg_s(
    charged_centers: Sequence[ChargedCenter],
    single_center_atmosphere_resistance_kg_s: Sequence[float],
    kappa_inv_m: float,
) -> np.ndarray:
    center_tuple = tuple(charged_centers)
    single_center_resistance = np.asarray(
        single_center_atmosphere_resistance_kg_s, dtype=float
    )
    center_count = len(center_tuple)
    if single_center_resistance.shape != (center_count,):
        raise ValueError(
            "single_center_atmosphere_resistance_kg_s length must match charged_centers"
        )
    if center_count == 0:
        return np.zeros((0, 0), dtype=float)
    for center_index, charged_center in enumerate(center_tuple):
        _validate_relative_position_m(
            charged_center.label, charged_center.relative_position_m
        )
        if charged_center.charge == 0.0 or not math.isfinite(charged_center.charge):
            raise ValueError(
                f"{charged_center.label}.charge must be finite and nonzero"
            )
        _assert_nonnegative_finite(
            float(single_center_resistance[center_index]),
            f"{charged_center.label}.single_center_atmosphere_resistance_kg_s",
        )
    atmosphere_matrix = np.zeros((center_count, center_count), dtype=float)
    for first_center_index, first_center in enumerate(center_tuple):
        atmosphere_matrix[first_center_index, first_center_index] = (
            single_center_resistance[first_center_index]
        )
        for second_center_index in range(first_center_index + 1, center_count):
            second_center = center_tuple[second_center_index]
            center_distance_m = _center_distance_m(first_center, second_center)
            form_factor = _debye_charge_form_factor(center_distance_m, kappa_inv_m)
            sign_product = math.copysign(
                1.0, first_center.charge * second_center.charge
            )
            coupling_resistance_kg_s = (
                sign_product
                * math.sqrt(
                    float(single_center_resistance[first_center_index])
                    * float(single_center_resistance[second_center_index])
                )
                * form_factor
            )
            atmosphere_matrix[first_center_index, second_center_index] = (
                coupling_resistance_kg_s
            )
            atmosphere_matrix[second_center_index, first_center_index] = (
                coupling_resistance_kg_s
            )
    _validate_form_factor_atmosphere_matrix(atmosphere_matrix)
    return atmosphere_matrix


def _center_distance_m(
    first_center: ChargedCenter,
    second_center: ChargedCenter,
) -> float:
    first_position = np.asarray(first_center.relative_position_m, dtype=float)
    second_position = np.asarray(second_center.relative_position_m, dtype=float)
    center_distance_m = float(np.linalg.norm(first_position - second_position))
    _assert_nonnegative_finite(center_distance_m, "center_distance_m")
    return center_distance_m


def _debye_charge_form_factor(
    center_distance_m: float,
    kappa_inv_m: float,
) -> float:
    _assert_nonnegative_finite(center_distance_m, "center_distance_m")
    if kappa_inv_m <= 0.0 or (
        not math.isfinite(kappa_inv_m) and not math.isinf(kappa_inv_m)
    ):
        raise ValueError(f"kappa_inv_m must be positive or infinite, got {kappa_inv_m}")
    if center_distance_m == 0.0:
        return 1.0
    if math.isinf(kappa_inv_m):
        return 1.0
    form_factor = math.exp(-center_distance_m / kappa_inv_m)
    if form_factor < 0.0 or form_factor > 1.0 or not math.isfinite(form_factor):
        raise ValueError(f"charge form factor must be in [0, 1], got {form_factor}")
    return form_factor


def _validate_form_factor_atmosphere_matrix(atmosphere_matrix: np.ndarray) -> None:
    if not np.all(np.isfinite(atmosphere_matrix)):
        raise ValueError("form-factor atmosphere matrix contains non-finite values")
    if not np.allclose(atmosphere_matrix, atmosphere_matrix.T):
        raise ValueError("form-factor atmosphere matrix must be symmetric")
    if atmosphere_matrix.size == 0:
        return
    eigenvalues = np.linalg.eigvalsh(atmosphere_matrix)
    if float(np.min(eigenvalues)) < -REVERSE_DIFFUSION_TOLERANCE:
        raise ValueError("form-factor atmosphere matrix must be positive semidefinite")


def _bulk_ion_atmosphere_state(
    kernel_state: TransportKernelState,
    temperature_K: float,
) -> BulkIonAtmosphereState:
    return _build_bulk_ion_atmosphere_state_from_concentrations(
        kernel_state=kernel_state,
        temperature_K=temperature_K,
        carrier_concentrations_mol_m3=_current_recipe_bath_concentrations_mol_m3(
            kernel_state
        ),
    )


def _state_bulk_ion_atmosphere_state(
    kernel_state: TransportKernelState,
    charged_centers: tuple[ChargedCenter, ...],
    state_concentration_mol_m3: float,
    recipe_bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    atmosphere_bath_basis: str,
    temperature_K: float,
) -> BulkIonAtmosphereState:
    _assert_nonnegative_finite(state_concentration_mol_m3, "state_concentration_mol_m3")
    if atmosphere_bath_basis == ATMOSPHERE_BATH_BASIS_TOTAL_FORMAL:
        return recipe_bulk_ion_atmosphere_state
    if atmosphere_bath_basis != ATMOSPHERE_BATH_BASIS_EXTERNAL_FREE_BATH:
        raise ValueError(f"Unsupported atmosphere_bath_basis {atmosphere_bath_basis!r}")
    external_concentrations_mol_m3 = _external_bath_concentrations_mol_m3(
        kernel_state=kernel_state,
        charged_centers=charged_centers,
        state_concentration_mol_m3=state_concentration_mol_m3,
        carrier_labels=recipe_bulk_ion_atmosphere_state.carrier_labels,
    )
    return _build_bulk_ion_atmosphere_state_from_concentrations(
        kernel_state=kernel_state,
        temperature_K=temperature_K,
        carrier_concentrations_mol_m3=external_concentrations_mol_m3,
    )


def _build_bulk_ion_atmosphere_state_from_concentrations(
    kernel_state: TransportKernelState,
    temperature_K: float,
    carrier_concentrations_mol_m3: Mapping[str, float],
) -> BulkIonAtmosphereState:
    cation_symbol = kernel_state.site_measure.cation.ion_symbol
    anion_diffusivity_by_feature_m2_s = _anion_center_diffusivity_by_feature_m2_s(
        kernel_state,
        temperature_K,
    )
    carrier_labels = tuple(
        [cation_symbol]
        + [
            anion_site.carrier_label
            for anion_site in kernel_state.site_measure.anion_sites
        ]
    )
    carrier_charges: dict[str, int] = {}
    local_diffusivity_m2_s_by_carrier: dict[str, float] = {}
    hydrodynamic_radius_m_by_carrier: dict[str, float] = {}
    normalized_concentrations_mol_m3 = {
        carrier_label: _require_mapping_float(
            carrier_concentrations_mol_m3,
            carrier_label,
            "carrier_concentrations_mol_m3",
        )
        for carrier_label in carrier_labels
    }
    carrier_charges[cation_symbol] = int(kernel_state.site_measure.cation.charge)
    local_diffusivity_m2_s_by_carrier[cation_symbol] = _cation_center_diffusivity_m2_s(
        kernel_state,
        temperature_K,
    )
    hydrodynamic_radius_m_by_carrier[cation_symbol] = (
        kernel_state.site_measure.cation.solvated_radius_A * ANGSTROM_TO_M
    )
    for anion_site in kernel_state.site_measure.anion_sites:
        feature_id = anion_site.canonical_feature_id
        carrier_label = anion_site.carrier_label
        carrier_charges[carrier_label] = int(anion_site.charge)
        local_diffusivity_m2_s_by_carrier[carrier_label] = _require_mapping_float(
            anion_diffusivity_by_feature_m2_s,
            feature_id,
            "anion_center_diffusivity_by_feature_m2_s",
        )
        hydrodynamic_radius_m_by_carrier[carrier_label] = (
            anion_site.anion_radius_A * ANGSTROM_TO_M
        )
    return build_bulk_ion_atmosphere_state(
        BulkIonAtmosphereInput(
            carrier_labels=carrier_labels,
            carrier_concentrations_mol_m3=normalized_concentrations_mol_m3,
            carrier_charges=carrier_charges,
            local_diffusivity_m2_s_by_carrier=local_diffusivity_m2_s_by_carrier,
            hydrodynamic_radius_m_by_carrier=hydrodynamic_radius_m_by_carrier,
            viscosity_Pa_s=kernel_state.matrix.eta_solution_cP * CP_TO_PA_S,
            relative_dielectric=kernel_state.matrix.epsilon_effective,
            temperature_K=temperature_K,
            solver=FINITE_MARKOV_ION_ATMOSPHERE_SOLVER,
        )
    )


def _current_recipe_bath_concentrations_mol_m3(
    kernel_state: TransportKernelState,
) -> dict[str, float]:
    cation_symbol = kernel_state.site_measure.cation.ion_symbol
    carrier_concentrations_mol_m3: dict[str, float] = {
        cation_symbol: (
            _require_mapping_float(
                kernel_state.speciation.carrier_concentrations_M,
                cation_symbol,
                "kernel_state.speciation.carrier_concentrations_M",
            )
            * MOLARITY_TO_MOL_M3
        )
    }
    for anion_site in kernel_state.site_measure.anion_sites:
        carrier_concentrations_mol_m3[anion_site.carrier_label] = (
            _require_mapping_float(
                kernel_state.speciation.carrier_concentrations_M,
                anion_site.carrier_label,
                "kernel_state.speciation.carrier_concentrations_M",
            )
            * MOLARITY_TO_MOL_M3
        )
    return carrier_concentrations_mol_m3


def _formal_bath_concentrations_mol_m3(
    kernel_state: TransportKernelState,
) -> dict[str, float]:
    cation_symbol = kernel_state.site_measure.cation.ion_symbol
    carrier_concentrations_mol_m3: dict[str, float] = {
        cation_symbol: _total_cation_molarity(kernel_state) * MOLARITY_TO_MOL_M3
    }
    for anion_site in kernel_state.site_measure.anion_sites:
        carrier_concentrations_mol_m3[anion_site.carrier_label] = (
            anion_site.molarity_M * MOLARITY_TO_MOL_M3
        )
    return carrier_concentrations_mol_m3


def _external_bath_concentrations_mol_m3(
    kernel_state: TransportKernelState,
    charged_centers: tuple[ChargedCenter, ...],
    state_concentration_mol_m3: float,
    carrier_labels: tuple[str, ...],
) -> dict[str, float]:
    carrier_index_by_label = {
        carrier_label: carrier_index
        for carrier_index, carrier_label in enumerate(carrier_labels)
    }
    external_concentrations_mol_m3 = _formal_bath_concentrations_mol_m3(kernel_state)
    resolved_counts_by_carrier = np.zeros(len(carrier_labels), dtype=float)
    for charged_center in charged_centers:
        projection_vector = _charged_center_bulk_projection_vector(
            kernel_state,
            charged_center,
            carrier_index_by_label,
        )
        resolved_counts_by_carrier += projection_vector
    for carrier_index, carrier_label in enumerate(carrier_labels):
        resolved_concentration_mol_m3 = (
            resolved_counts_by_carrier[carrier_index] * state_concentration_mol_m3
        )
        candidate_concentration_mol_m3 = (
            _require_mapping_float(
                external_concentrations_mol_m3,
                carrier_label,
                "external_concentrations_mol_m3",
            )
            - resolved_concentration_mol_m3
        )
        if (
            candidate_concentration_mol_m3
            < -EXTERNAL_BATH_CONCENTRATION_TOLERANCE_MOL_M3
        ):
            raise ValueError(
                f"external bath concentration for {carrier_label} became negative: "
                f"{candidate_concentration_mol_m3} mol/m3"
            )
        if candidate_concentration_mol_m3 < 0.0:
            candidate_concentration_mol_m3 = 0.0
        external_concentrations_mol_m3[carrier_label] = candidate_concentration_mol_m3
    return external_concentrations_mol_m3


def _formal_ionic_strength_mol_m3(kernel_state: TransportKernelState) -> float:
    ionic_strength_mol_m3 = 0.0
    cation_charge = float(kernel_state.site_measure.cation.charge)
    ionic_strength_mol_m3 += (
        cation_charge
        * cation_charge
        * _total_cation_molarity(kernel_state)
        * MOLARITY_TO_MOL_M3
    )
    for anion_site in kernel_state.site_measure.anion_sites:
        anion_charge = float(anion_site.charge)
        ionic_strength_mol_m3 += (
            anion_charge * anion_charge * anion_site.molarity_M * MOLARITY_TO_MOL_M3
        )
    _assert_positive_finite(ionic_strength_mol_m3, "formal_ionic_strength_mol_m3")
    return ionic_strength_mol_m3


def _charged_center_bulk_projection_vector(
    kernel_state: TransportKernelState,
    charged_center: ChargedCenter,
    carrier_index_by_label: Mapping[str, int],
) -> np.ndarray:
    projection_vector = np.zeros(len(carrier_index_by_label), dtype=float)
    if charged_center.charge > 0.0:
        cation_symbol = kernel_state.site_measure.cation.ion_symbol
        cation_index = _require_mapping_int(
            carrier_index_by_label,
            cation_symbol,
            "carrier_index_by_label",
        )
        projection_vector[cation_index] = 1.0
        return projection_vector
    if _is_weighted_anion_center_label(charged_center.label):
        return _weighted_anion_projection_vector(kernel_state, carrier_index_by_label)
    feature_id = _feature_id_from_charged_center_label(
        kernel_state, charged_center.label
    )
    anion_site_by_feature = kernel_state.site_measure.anion_by_canonical_id()
    if feature_id not in anion_site_by_feature:
        raise KeyError(
            f"charged center {charged_center.label} references unknown anion feature {feature_id}"
        )
    carrier_label = anion_site_by_feature[feature_id].carrier_label
    carrier_index = _require_mapping_int(
        carrier_index_by_label,
        carrier_label,
        "carrier_index_by_label",
    )
    projection_vector[carrier_index] = 1.0
    return projection_vector


def _weighted_anion_projection_vector(
    kernel_state: TransportKernelState,
    carrier_index_by_label: Mapping[str, int],
) -> np.ndarray:
    projection_vector = np.zeros(len(carrier_index_by_label), dtype=float)
    concentration_sum_M = 0.0
    for anion_site in kernel_state.site_measure.anion_sites:
        concentration_sum_M += _require_mapping_float(
            kernel_state.speciation.carrier_concentrations_M,
            anion_site.carrier_label,
            "kernel_state.speciation.carrier_concentrations_M",
        )
    _assert_positive_finite(
        concentration_sum_M, "weighted_anion_projection_concentration_sum_M"
    )
    for anion_site in kernel_state.site_measure.anion_sites:
        carrier_label = anion_site.carrier_label
        carrier_index = _require_mapping_int(
            carrier_index_by_label,
            carrier_label,
            "carrier_index_by_label",
        )
        projection_vector[carrier_index] = (
            _require_mapping_float(
                kernel_state.speciation.carrier_concentrations_M,
                carrier_label,
                "kernel_state.speciation.carrier_concentrations_M",
            )
            / concentration_sum_M
        )
    return projection_vector


def _require_mapping_int(
    values: Mapping[str, int],
    key: str,
    context: str,
) -> int:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    return int(values[key])


def _is_weighted_anion_center_label(charged_center_label: str) -> bool:
    return charged_center_label in {
        "weighted_anion",
        "weighted_free_anion",
        "weighted_anion_site_0",
        "weighted_anion_site_1",
    }


def _feature_id_from_charged_center_label(
    kernel_state: TransportKernelState,
    charged_center_label: str,
) -> str:
    for anion_site in kernel_state.site_measure.anion_sites:
        feature_id = anion_site.canonical_feature_id
        if (
            charged_center_label == f"{feature_id}:anion"
            or charged_center_label.startswith(f"{feature_id}:anion_site_")
        ):
            return feature_id
    raise KeyError(
        f"charged center {charged_center_label} is not a known anion feature label"
    )


def _charged_centers_with_state_geometry(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
    charged_centers: tuple[ChargedCenter, ...],
) -> tuple[ChargedCenter, ...]:
    if not charged_centers:
        return ()
    origin_m = (0.0, 0.0, 0.0)
    if len(charged_centers) == 1:
        return (
            _positioned_charged_center(
                charged_centers[0], charged_centers[0].label, origin_m
            ),
        )
    separation_m = _charged_center_separation_m(kernel_state, motif)
    _assert_positive_finite(separation_m, f"{motif.label}.charge_center_separation_m")
    if motif.kind in (
        ChemicalMotifKind.SSIP,
        ChemicalMotifKind.CIP,
        ChemicalMotifKind.ADDITIVE_SSIP,
    ):
        return (
            _positioned_charged_center(
                charged_centers[0], charged_centers[0].label, origin_m
            ),
            _positioned_charged_center(
                charged_centers[1], charged_centers[1].label, (separation_m, 0.0, 0.0)
            ),
        )
    if motif.kind is ChemicalMotifKind.LI2A_PLUS:
        return (
            _positioned_charged_center(
                charged_centers[0], charged_centers[0].label, (-separation_m, 0.0, 0.0)
            ),
            _positioned_charged_center(
                charged_centers[1], charged_centers[1].label, (separation_m, 0.0, 0.0)
            ),
            _positioned_charged_center(
                charged_centers[2], charged_centers[2].label, origin_m
            ),
        )
    if motif.kind is ChemicalMotifKind.LIA2_MINUS:
        return (
            _positioned_charged_center(
                charged_centers[0], charged_centers[0].label, origin_m
            ),
            _positioned_charged_center(
                charged_centers[1], charged_centers[1].label, (separation_m, 0.0, 0.0)
            ),
            _positioned_charged_center(
                charged_centers[2], charged_centers[2].label, (-separation_m, 0.0, 0.0)
            ),
        )
    if motif.kind is ChemicalMotifKind.LI2A2_NEUTRAL:
        return (
            _positioned_charged_center(
                charged_centers[0], charged_centers[0].label, (0.0, -separation_m, 0.0)
            ),
            _positioned_charged_center(
                charged_centers[1], charged_centers[1].label, (0.0, separation_m, 0.0)
            ),
            _positioned_charged_center(
                charged_centers[2],
                charged_centers[2].label,
                (separation_m, -separation_m, 0.0),
            ),
            _positioned_charged_center(
                charged_centers[3],
                charged_centers[3].label,
                (separation_m, separation_m, 0.0),
            ),
        )
    if motif.kind in (ChemicalMotifKind.AGGREGATE, ChemicalMotifKind.BRIDGE_NETWORK):
        positioned_centers = [
            _positioned_charged_center(
                charged_centers[0], charged_centers[0].label, origin_m
            ),
        ]
        for center_index, charged_center in enumerate(charged_centers[1:], start=1):
            direction = -1.0 if center_index % 2 == 0 else 1.0
            positioned_centers.append(
                _positioned_charged_center(
                    charged_center,
                    charged_center.label,
                    (direction * separation_m, 0.0, 0.0),
                )
            )
        return tuple(positioned_centers)
    return tuple(
        _positioned_charged_center(charged_center, charged_center.label, origin_m)
        for charged_center in charged_centers
    )


def _positioned_charged_center(
    charged_center: ChargedCenter,
    label: str,
    relative_position_m: tuple[float, float, float],
) -> ChargedCenter:
    _validate_relative_position_m(label, relative_position_m)
    return ChargedCenter(
        label=label,
        charge=charged_center.charge,
        hydrodynamic_radius_m=charged_center.hydrodynamic_radius_m,
        shape_factor=charged_center.shape_factor,
        local_diffusion_m2_s=charged_center.local_diffusion_m2_s,
        relative_position_m=relative_position_m,
        charge_cloud_radius_available=charged_center.charge_cloud_radius_available,
        charge_cloud_radius_A=charged_center.charge_cloud_radius_A,
        charge_cloud_source=charged_center.charge_cloud_source,
        charge_cloud_site_count=charged_center.charge_cloud_site_count,
    )


def _validate_relative_position_m(
    label: str,
    relative_position_m: tuple[float, float, float],
) -> None:
    if len(relative_position_m) != AXIS_COUNT:
        raise ValueError(
            f"{label}.relative_position_m must have {AXIS_COUNT} components"
        )
    for component in relative_position_m:
        if not math.isfinite(component):
            raise ValueError(
                f"{label}.relative_position_m contains non-finite component {component}"
            )


def _renamed_charged_center(
    charged_center: ChargedCenter,
    label: str,
) -> ChargedCenter:
    return _positioned_charged_center(
        charged_center, label, charged_center.relative_position_m
    )


def _transport_state_free_energy_J_mol(
    stationary_probability: float,
    temperature_K: float,
) -> float:
    _assert_positive_finite(stationary_probability, "stationary_probability")
    return -R * temperature_K * math.log(stationary_probability)


def _constraint_modes_for_state(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
    charged_centers: tuple[ChargedCenter, ...],
    binding_kinetics_rows: list[MotifBindingKinetics],
    temperature_K: float,
    physics_config,
) -> tuple[ConstraintMode, ...]:
    if motif.kind is ChemicalMotifKind.SOLVENT_CAGE:
        return ()
    if motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED:
        return ()
    binding_kinetics = _motif_binding_kinetics(
        kernel_state,
        motif,
        charged_centers,
        temperature_K,
        physics_config,
    )
    binding_kinetics_rows.append(binding_kinetics)
    constraint_lifetime_s = binding_kinetics.constraint_tau_s
    atmosphere_lifetime_s = binding_kinetics.tau_s
    separation_m = binding_kinetics.basin_length_m
    if motif.kind is ChemicalMotifKind.CIP:
        return (
            _pair_constraint_mode(
                charged_centers,
                0,
                1,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
        )
    if (
        motif.kind is ChemicalMotifKind.SSIP
        or motif.kind is ChemicalMotifKind.ADDITIVE_SSIP
    ):
        return (
            _pair_constraint_mode(
                charged_centers,
                0,
                1,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
        )
    if motif.kind is ChemicalMotifKind.LI2A_PLUS:
        return (
            _pair_constraint_mode(
                charged_centers,
                0,
                2,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
            _pair_constraint_mode(
                charged_centers,
                1,
                2,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
        )
    if motif.kind is ChemicalMotifKind.LIA2_MINUS:
        return (
            _pair_constraint_mode(
                charged_centers,
                0,
                1,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
            _pair_constraint_mode(
                charged_centers,
                0,
                2,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
        )
    if motif.kind is ChemicalMotifKind.LI2A2_NEUTRAL:
        return (
            _pair_constraint_mode(
                charged_centers,
                0,
                2,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
            _pair_constraint_mode(
                charged_centers,
                1,
                3,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
        )
    if motif.kind is ChemicalMotifKind.BRIDGE_NETWORK:
        return (
            _pair_constraint_mode(
                charged_centers,
                0,
                1,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
        )
    if motif.kind is ChemicalMotifKind.AGGREGATE:
        return (
            _pair_constraint_mode(
                charged_centers,
                0,
                1,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
            _pair_constraint_mode(
                charged_centers,
                0,
                2,
                constraint_lifetime_s,
                atmosphere_lifetime_s,
                separation_m,
            ),
        )
    raise ValueError(f"Unhandled motif kind for constraint modes {motif.kind}")


def _pair_constraint_mode(
    charged_centers: tuple[ChargedCenter, ...],
    first_center_index: int,
    second_center_index: int,
    lifetime_s: float,
    atmosphere_lifetime_s: float,
    length_m: float,
) -> ConstraintMode:
    vector = [0.0 for _charged_center in charged_centers]
    vector[first_center_index] = 1.0
    vector[second_center_index] = -1.0
    return ConstraintMode(
        labels=(
            charged_centers[first_center_index].label,
            charged_centers[second_center_index].label,
        ),
        vector=tuple(vector),
        lifetime_s=lifetime_s,
        atmosphere_lifetime_s=atmosphere_lifetime_s,
        length_m=length_m,
    )


def _motif_binding_kinetics(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
    charged_centers: tuple[ChargedCenter, ...],
    temperature_K: float,
    physics_config,
) -> MotifBindingKinetics:
    if len(charged_centers) < 2:
        raise ValueError(
            f"{motif.label} binding kinetics requires at least two charged centers"
        )
    basin_equilibrium_constant_M_inv = _motif_basin_equilibrium_constant_M_inv(
        kernel_state,
        motif,
        temperature_K,
        physics_config,
    )
    constraint_equilibrium_constant_M_inv = (
        _motif_constraint_equilibrium_constant_M_inv(
            kernel_state,
            motif,
            basin_equilibrium_constant_M_inv,
            temperature_K,
        )
    )
    basin_length_m = _charged_center_separation_m(kernel_state, motif)
    capture_accessibility = _motif_capture_accessibility(kernel_state, motif)
    relative_diffusivity_m2_s = (
        charged_centers[0].local_diffusion_m2_s
        + charged_centers[1].local_diffusion_m2_s
    )
    _assert_positive_finite(
        relative_diffusivity_m2_s, f"{motif.label}.relative_diffusivity_m2_s"
    )
    k_on_M_inv_s = (
        4.0
        * math.pi
        * N_A
        * MOLARITY_TO_MOL_M3
        * relative_diffusivity_m2_s
        * basin_length_m
        * capture_accessibility
    )
    _assert_positive_finite(k_on_M_inv_s, f"{motif.label}.k_on_M_inv_s")
    k_off_s_inv = k_on_M_inv_s / basin_equilibrium_constant_M_inv
    _assert_positive_finite(k_off_s_inv, f"{motif.label}.k_off_s_inv")
    tau_s = 1.0 / k_off_s_inv
    _assert_positive_finite(tau_s, f"{motif.label}.tau_s")
    constraint_k_off_s_inv = k_on_M_inv_s / constraint_equilibrium_constant_M_inv
    _assert_positive_finite(
        constraint_k_off_s_inv, f"{motif.label}.constraint_k_off_s_inv"
    )
    constraint_tau_s = 1.0 / constraint_k_off_s_inv
    _assert_positive_finite(constraint_tau_s, f"{motif.label}.constraint_tau_s")
    return MotifBindingKinetics(
        motif_label=motif.label,
        K_M_inv=basin_equilibrium_constant_M_inv,
        k_on_M_inv_s=k_on_M_inv_s,
        k_off_s_inv=k_off_s_inv,
        tau_s=tau_s,
        constraint_tau_s=constraint_tau_s,
        basin_length_m=basin_length_m,
        source="bjerrum_site_smoluchowski",
    )


def _motif_basin_equilibrium_constant_M_inv(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
    temperature_K: float,
    physics_config,
) -> float:
    feature_id = _require_feature_id(motif)
    anion_site = kernel_state.site_measure.anion_by_canonical_id()[feature_id]
    ssip_association_M_inv, cip_association_M_inv = (
        _state_pair_association_constants_M_inv(
            kernel_state,
            anion_site,
            temperature_K,
        )
    )
    if (
        motif.kind is ChemicalMotifKind.SSIP
        or motif.kind is ChemicalMotifKind.ADDITIVE_SSIP
    ):
        return ssip_association_M_inv
    if motif.kind is ChemicalMotifKind.CIP:
        return cip_association_M_inv
    if motif.kind in (
        ChemicalMotifKind.LI2A_PLUS,
        ChemicalMotifKind.LIA2_MINUS,
        ChemicalMotifKind.LI2A2_NEUTRAL,
        ChemicalMotifKind.BRIDGE_NETWORK,
    ):
        return cip_association_M_inv
    raise ValueError(
        f"Motif kind {motif.kind} does not define a Li-anion basin constant"
    )


def _motif_constraint_equilibrium_constant_M_inv(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
    basin_equilibrium_constant_M_inv: float,
    temperature_K: float,
) -> float:
    if (
        motif.kind is ChemicalMotifKind.SSIP
        or motif.kind is ChemicalMotifKind.ADDITIVE_SSIP
    ):
        separation_m = _charged_center_separation_m(kernel_state, motif)
        debye_kappa_inv_m = _debye_kappa_inv_m(
            kernel_state, kernel_state.matrix.epsilon_effective, temperature_K
        )
        return _screened_ssip_constraint_constant_M_inv(
            basin_equilibrium_constant_M_inv,
            separation_m,
            debye_kappa_inv_m,
            motif.label,
        )
    return basin_equilibrium_constant_M_inv


def _screened_ssip_constraint_constant_M_inv(
    ssip_association_M_inv: float,
    separation_m: float,
    debye_kappa_inv_m: float,
    context: str,
) -> float:
    _assert_positive_finite(ssip_association_M_inv, f"{context}.ssip_association_M_inv")
    _assert_positive_finite(separation_m, f"{context}.separation_m")
    _assert_positive_finite(debye_kappa_inv_m, f"{context}.debye_kappa_inv_m")
    screened_constant_M_inv = ssip_association_M_inv * math.exp(
        -separation_m / debye_kappa_inv_m
    )
    _assert_positive_finite(
        screened_constant_M_inv, f"{context}.screened_ssip_constraint_constant_M_inv"
    )
    return screened_constant_M_inv


def _motif_capture_accessibility(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
) -> float:
    anion_site = kernel_state.site_measure.anion_by_canonical_id()[
        _require_feature_id(motif)
    ]
    return _anion_capture_accessibility(
        donor_site_count=anion_site.donor_site_count,
        coordination_multiplicity=anion_site.coordination_multiplicity,
        preferred_coordination_number=anion_site.preferred_coordination_number,
        context=anion_site.canonical_feature_id,
    )


def _anion_capture_accessibility(
    donor_site_count: float,
    coordination_multiplicity: float,
    preferred_coordination_number: float,
    context: str,
) -> float:
    _assert_nonnegative_finite(donor_site_count, f"{context}.donor_site_count")
    _assert_positive_finite(
        coordination_multiplicity, f"{context}.coordination_multiplicity"
    )
    _assert_positive_finite(
        preferred_coordination_number, f"{context}.preferred_coordination_number"
    )
    if donor_site_count == 0.0:
        return 1.0
    accessibility = (
        donor_site_count
        * coordination_multiplicity
        / (
            (donor_site_count + preferred_coordination_number)
            * preferred_coordination_number
        )
    )
    if accessibility <= 0.0 or accessibility > 1.0 or not math.isfinite(accessibility):
        raise ValueError(
            f"{context}.capture_accessibility must satisfy 0 < g <= 1, got {accessibility}"
        )
    return accessibility


def _charged_center_separation_m(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
) -> float:
    if motif.feature_id is None:
        return (
            kernel_state.site_measure.cation.ionic_radius_A * ANGSTROM_TO_M
            + _weighted_anion_radius_m(kernel_state)
        )
    anion_site = kernel_state.site_measure.anion_by_canonical_id()[
        _require_feature_id(motif)
    ]
    contact_distance_m = _li_anion_contact_distance_m(kernel_state, anion_site)
    if (
        motif.kind is ChemicalMotifKind.SSIP
        or motif.kind is ChemicalMotifKind.ADDITIVE_SSIP
    ):
        return contact_distance_m + _separator_thickness_m(kernel_state, motif)
    return contact_distance_m


def _li_anion_contact_distance_m(
    kernel_state: TransportKernelState,
    anion_site,
) -> float:
    contact_distance_m = (
        kernel_state.site_measure.cation.ionic_radius_A + anion_site.anion_radius_A
    ) * ANGSTROM_TO_M
    _assert_positive_finite(
        contact_distance_m, f"{anion_site.canonical_feature_id}.contact_distance_m"
    )
    return contact_distance_m


def _separator_thickness_m(
    kernel_state: TransportKernelState,
    motif: ChemicalMotif,
) -> float:
    if motif.kind is ChemicalMotifKind.ADDITIVE_SSIP:
        additive_separator_m = _coordinating_ligand_separator_thickness_m(kernel_state)
        if additive_separator_m > 0.0:
            return additive_separator_m
    return _solvent_shell_separator_thickness_m(kernel_state)


def _solvent_shell_separator_thickness_m(
    kernel_state: TransportKernelState,
) -> float:
    cation_shell_thickness_m = (
        kernel_state.site_measure.cation.solvated_radius_A
        - kernel_state.site_measure.cation.ionic_radius_A
    ) * ANGSTROM_TO_M
    _assert_positive_finite(cation_shell_thickness_m, "cation_shell_thickness_m")
    preferred_coordination_number = kernel_state.solvation.preferred_coordination_number
    _assert_positive_finite(
        preferred_coordination_number, "preferred_coordination_number"
    )
    separator_thickness_m = cation_shell_thickness_m / preferred_coordination_number
    _assert_positive_finite(
        separator_thickness_m, "solvent_shell_separator_thickness_m"
    )
    return separator_thickness_m


def _coordinating_ligand_separator_thickness_m(
    kernel_state: TransportKernelState,
) -> float:
    additive_molecular_volume_m3 = _weighted_coordinating_additive_molecular_volume_m3(
        kernel_state
    )
    if additive_molecular_volume_m3 == 0.0:
        return 0.0
    preferred_coordination_number = kernel_state.solvation.preferred_coordination_number
    _assert_positive_finite(
        preferred_coordination_number, "preferred_coordination_number"
    )
    ligand_site_volume_m3 = additive_molecular_volume_m3 / preferred_coordination_number
    separator_thickness_m = _sphere_radius_from_volume_m(ligand_site_volume_m3)
    _assert_positive_finite(
        separator_thickness_m, "coordinating_ligand_separator_thickness_m"
    )
    return separator_thickness_m


def _weighted_anion_radius_m(kernel_state: TransportKernelState) -> float:
    total_molarity = _total_cation_molarity(kernel_state)
    weighted_radius_m = 0.0
    for anion_site in kernel_state.site_measure.anion_sites:
        weighted_radius_m += (
            anion_site.molarity_M
            / total_molarity
            * anion_site.anion_radius_A
            * ANGSTROM_TO_M
        )
    _assert_positive_finite(weighted_radius_m, "weighted_anion_radius_m")
    return weighted_radius_m


def _anion_center_diffusivity_by_feature_m2_s(
    kernel_state: TransportKernelState,
    temperature_K: float,
) -> dict[str, float]:
    diffusivity_by_feature: dict[str, float] = {}
    for anion_site in kernel_state.site_measure.anion_sites:
        feature_id = anion_site.canonical_feature_id
        lambda_S_cm2_mol = _require_mapping_float(
            kernel_state.mobility.feature_lambda_split_S_cm2_mol[feature_id],
            anion_site.carrier_label,
            f"feature_lambda_split_S_cm2_mol.{feature_id}",
        )
        diffusivity_by_feature[feature_id] = _diffusivity_from_molar_conductivity_m2_s(
            lambda_S_cm2_mol,
            float(anion_site.charge),
            temperature_K,
        )
    return diffusivity_by_feature


def _weighted_anion_center_diffusivity_m2_s(
    kernel_state: TransportKernelState,
    temperature_K: float,
) -> float:
    total_molarity = _total_cation_molarity(kernel_state)
    diffusivity_by_feature = _anion_center_diffusivity_by_feature_m2_s(
        kernel_state, temperature_K
    )
    weighted_diffusivity_m2_s = 0.0
    for anion_site in kernel_state.site_measure.anion_sites:
        weighted_diffusivity_m2_s += (
            anion_site.molarity_M
            / total_molarity
            * diffusivity_by_feature[anion_site.canonical_feature_id]
        )
    _assert_positive_finite(
        weighted_diffusivity_m2_s, "weighted_anion_center_diffusivity_m2_s"
    )
    return weighted_diffusivity_m2_s


def _diffusivity_from_molar_conductivity_m2_s(
    lambda_S_cm2_mol: float,
    charge_number: float,
    temperature_K: float,
) -> float:
    _assert_positive_finite(lambda_S_cm2_mol, "lambda_S_cm2_mol")
    if charge_number == 0.0 or not math.isfinite(charge_number):
        raise ValueError(
            f"charge_number must be finite and nonzero, got {charge_number}"
        )
    return (
        lambda_S_cm2_mol
        * S_CM2_PER_MOL_TO_S_M2_PER_MOL
        * R
        * temperature_K
        / (F * F * charge_number * charge_number)
    )


def _state_net_charges(
    transport_states: tuple[TransportState, ...],
) -> np.ndarray:
    return np.asarray(
        [
            math.fsum(center.charge for center in transport_state.charged_centers)
            for transport_state in transport_states
        ],
        dtype=float,
    )


def _anion_charge_by_feature(kernel_state: TransportKernelState) -> dict[str, float]:
    charges: dict[str, float] = {}
    for anion_site in kernel_state.site_measure.anion_sites:
        charges[anion_site.canonical_feature_id] = float(anion_site.charge)
    return charges


def _single_cation_name(kernel_state: TransportKernelState) -> str:
    return kernel_state.site_measure.cation.canonical_feature_id


def _total_cation_molarity(kernel_state: TransportKernelState) -> float:
    total_molarity = 0.0
    for anion_site in kernel_state.site_measure.anion_sites:
        total_molarity += anion_site.molarity_M
    _assert_positive_finite(total_molarity, "total_cation_molarity_M")
    return total_molarity


def _coordinating_additive_shell_fraction(kernel_state: TransportKernelState) -> float:
    additive_shell_fraction = math.fsum(
        kernel_state.speciation.li_ligand_fraction_by_feature.values()
    )
    if additive_shell_fraction < 0.0 or additive_shell_fraction > 1.0:
        raise ValueError(
            f"Invalid coordinating additive shell fraction {additive_shell_fraction}"
        )
    return additive_shell_fraction


def _bruggeman_dielectric(kernel_state: TransportKernelState) -> float:
    volume_fractions = kernel_state.composition.neutral_liquid_volume_fractions
    eps_values = []
    for species_name in volume_fractions:
        props = _neutral_species_props(species_name)
        eps_values.append(
            require_float(props, "epsilon_r", f"neutral species {species_name}")
        )
    lower_bound = min(eps_values)
    upper_bound = max(eps_values)
    if lower_bound <= 0.0:
        raise ValueError("Bruggeman dielectric lower bound must be positive")
    if abs(upper_bound - lower_bound) <= REVERSE_DIFFUSION_TOLERANCE:
        return lower_bound
    left = lower_bound
    right = upper_bound
    for _iteration_index in range(100):
        midpoint = 0.5 * (left + right)
        residual = _bruggeman_residual(volume_fractions, midpoint)
        if residual > 0.0:
            left = midpoint
        else:
            right = midpoint
    return 0.5 * (left + right)


def _bruggeman_residual(
    volume_fractions: Mapping[str, float],
    dielectric: float,
) -> float:
    residual = 0.0
    for species_name, volume_fraction in volume_fractions.items():
        props = _neutral_species_props(species_name)
        species_dielectric = require_float(
            props, "epsilon_r", f"neutral species {species_name}"
        )
        residual += (
            volume_fraction
            * (species_dielectric - dielectric)
            / (species_dielectric + 2.0 * dielectric)
        )
    return residual


def _debye_kappa_inv_m(
    kernel_state: TransportKernelState,
    dielectric: float,
    temperature_K: float,
) -> float:
    ionic_strength_mol_m3 = 0.0
    cation_charge = float(kernel_state.site_measure.cation.charge)
    for anion_site in kernel_state.site_measure.anion_sites:
        anion_charge = float(anion_site.charge)
        ionic_strength_mol_m3 += (
            0.5
            * anion_site.molarity_M
            * MOLARITY_TO_MOL_M3
            * (cation_charge * cation_charge + anion_charge * anion_charge)
        )
    if ionic_strength_mol_m3 <= 0.0:
        raise ValueError("Cannot compute Debye kappa without ionic strength")
    debye_kappa_m_inv = math.sqrt(
        2.0 * F * F * ionic_strength_mol_m3 / (EPS_0 * dielectric * R * temperature_K)
    )
    _assert_positive_finite(debye_kappa_m_inv, "debye_kappa_m_inv")
    return 1.0 / debye_kappa_m_inv


def _weighted_neutral_molecular_volume_m3(kernel_state: TransportKernelState) -> float:
    weighted_volume = 0.0
    for species_name, shell_fraction in kernel_state.solvation.shell_fractions.items():
        props = _neutral_species_props(species_name)
        weighted_volume += shell_fraction * _molecular_volume_m3(
            props, f"neutral species {species_name}"
        )
    _assert_positive_finite(weighted_volume, "weighted_neutral_molecular_volume_m3")
    return weighted_volume


def _weighted_coordinating_additive_molecular_volume_m3(
    kernel_state: TransportKernelState,
) -> float:
    weighted_volume = 0.0
    additive_shell_fraction = _coordinating_additive_shell_fraction(kernel_state)
    if additive_shell_fraction <= REVERSE_DIFFUSION_TOLERANCE:
        return 0.0
    ligand_molarity = math.fsum(
        site.molarity_M for site in kernel_state.site_measure.neutral_ligand_sites
    )
    _assert_positive_finite(ligand_molarity, "neutral ligand molarity")
    for ligand_site in kernel_state.site_measure.neutral_ligand_sites:
        weighted_volume += (
            ligand_site.molarity_M
            / ligand_molarity
            * ligand_site.molecular_volume_cm3_mol
            * ML_TO_M3
            / N_A
        )
    _assert_positive_finite(
        weighted_volume, "weighted_coordinating_additive_molecular_volume_m3"
    )
    return weighted_volume


def _weighted_anion_volume_m3(kernel_state: TransportKernelState) -> float:
    total_molarity = _total_cation_molarity(kernel_state)
    weighted_volume = 0.0
    for anion_site in kernel_state.site_measure.anion_sites:
        weighted_volume += (
            anion_site.molarity_M
            / total_molarity
            * anion_site.anion_volume_A3
            * ANGSTROM3_TO_M3
        )
    _assert_positive_finite(weighted_volume, "weighted_anion_volume_m3")
    return weighted_volume


def _weighted_anion_charge(kernel_state: TransportKernelState) -> float:
    total_molarity = _total_cation_molarity(kernel_state)
    weighted_charge = 0.0
    for anion_site in kernel_state.site_measure.anion_sites:
        weighted_charge += (
            anion_site.molarity_M / total_molarity * float(anion_site.charge)
        )
    return weighted_charge


def _molecular_volume_m3(props: Mapping[str, object], context: str) -> float:
    molecular_weight_g_mol = require_float(props, "molecular_weight", context)
    density_g_ml = require_float(props, "density_g_ml", context)
    _assert_positive_finite(molecular_weight_g_mol, f"{context}.molecular_weight")
    _assert_positive_finite(density_g_ml, f"{context}.density_g_ml")
    molar_volume_m3_mol = molecular_weight_g_mol / density_g_ml * ML_TO_M3
    return molar_volume_m3_mol / N_A


def _sphere_volume_m3(radius_m: float) -> float:
    return 4.0 / 3.0 * math.pi * radius_m * radius_m * radius_m


def _sphere_radius_from_volume_m(volume_m3: float) -> float:
    _assert_positive_finite(volume_m3, "sphere volume")
    return (3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def _validate_finite_input(
    finite_input: FiniteMarkovInput,
    parsed_input: ParsedFiniteMarkovInput,
) -> None:
    stationary_probabilities = parsed_input.stationary_probabilities
    state_concentrations_mol_m3 = parsed_input.state_concentrations_mol_m3
    generator_s_inv = parsed_input.generator_s_inv
    markov_additive_edges = parsed_input.markov_additive_edges
    transport_states = parsed_input.transport_states
    state_count = stationary_probabilities.shape[0]
    if state_count == 0:
        raise ValueError("finite Markov input has no states")
    if generator_s_inv.shape != (state_count, state_count):
        raise ValueError("generator_s_inv must be square with one row per state")
    if state_concentrations_mol_m3.shape != (state_count,):
        raise ValueError(
            "state_concentrations_mol_m3 must have one concentration per state"
        )
    _validate_transport_states(
        transport_states,
        stationary_probabilities,
        state_concentrations_mol_m3,
        state_count,
        finite_input.temperature_K,
    )
    _validate_markov_additive_edges(
        markov_additive_edges,
        state_count,
        transport_states,
        finite_input.temperature_K,
    )
    if len(finite_input.state_labels) != state_count:
        raise ValueError("state_labels must have one label per state")
    _assert_positive_finite(
        finite_input.cation_concentration_mol_m3, "cation_concentration_mol_m3"
    )
    _assert_positive_finite(finite_input.temperature_K, "temperature_K")
    if np.any(stationary_probabilities <= 0.0):
        raise ValueError("stationary_probabilities must be strictly positive")
    if np.any(state_concentrations_mol_m3 <= 0.0):
        raise ValueError("state_concentrations_mol_m3 must be strictly positive")
    concentration_sum_mol_m3 = float(np.sum(state_concentrations_mol_m3))
    _assert_positive_finite(concentration_sum_mol_m3, "state_concentration_sum_mol_m3")
    concentration_probabilities = state_concentrations_mol_m3 / concentration_sum_mol_m3
    if not np.allclose(concentration_probabilities, stationary_probabilities):
        raise ValueError(
            "stationary_probabilities must be normalized state concentrations"
        )
    probability_sum = float(np.sum(stationary_probabilities))
    if abs(probability_sum - 1.0) > REVERSE_DIFFUSION_TOLERANCE:
        raise ValueError(f"stationary_probabilities sum to {probability_sum}, not 1")
    diagonal = np.diag(generator_s_inv)
    if np.any(diagonal > REVERSE_DIFFUSION_TOLERANCE):
        raise ValueError("generator_s_inv diagonal entries must be non-positive")
    offdiag = np.array(generator_s_inv, dtype=float)
    np.fill_diagonal(offdiag, 0.0)
    if np.any(offdiag < -REVERSE_DIFFUSION_TOLERANCE):
        raise ValueError("generator_s_inv off-diagonal entries must be nonnegative")
    tolerance = _linear_solve_tolerance(generator_s_inv)
    row_sum_residual = _generator_row_sum_residual(generator_s_inv)
    if row_sum_residual > tolerance:
        raise ValueError(
            f"generator_s_inv rows do not sum to zero; residual {row_sum_residual}"
        )
    stationary_residual = _stationary_distribution_residual(
        generator_s_inv, stationary_probabilities
    )
    if stationary_residual > tolerance:
        raise ValueError(
            f"stationary distribution is not invariant; residual {stationary_residual}"
        )
    detailed_balance_residual = _detailed_balance_residual(
        generator_s_inv, stationary_probabilities
    )
    if detailed_balance_residual > tolerance:
        raise ValueError(
            f"generator_s_inv violates detailed balance; residual {detailed_balance_residual}"
        )


def _validate_markov_additive_edges(
    markov_additive_edges: tuple[MarkovAdditiveEdge, ...],
    state_count: int,
    transport_states: tuple[TransportState, ...],
    temperature_K: float,
) -> None:
    for edge in markov_additive_edges:
        if not isinstance(edge.kind, MarkovAdditiveEdgeKind):
            raise ValueError(f"{edge.label}.kind must be a MarkovAdditiveEdgeKind")
        if edge.source_index < 0 or edge.source_index >= state_count:
            raise ValueError(
                f"edge source_index {edge.source_index} is outside state range"
            )
        if edge.target_index < 0 or edge.target_index >= state_count:
            raise ValueError(
                f"edge target_index {edge.target_index} is outside state range"
            )
        _assert_positive_finite(edge.rate_s_inv, f"{edge.label}.rate_s_inv")
        if len(edge.displacement_m) != AXIS_COUNT:
            raise ValueError(f"{edge.label}.displacement_m must have three axes")
        for axis_index, displacement_m in enumerate(edge.displacement_m):
            if not math.isfinite(float(displacement_m)):
                raise ValueError(
                    f"{edge.label}.displacement_m[{axis_index}] is non-finite"
                )
        if edge.kind is MarkovAdditiveEdgeKind.MOTIF_EXCHANGE:
            if _edge_displacement_norm_m(edge) > REVERSE_DIFFUSION_TOLERANCE:
                raise ValueError(
                    f"{edge.label} motif exchange edge must have zero displacement"
                )
        elif edge.kind is MarkovAdditiveEdgeKind.VEHICULAR_JUMP:
            if edge.source_index != edge.target_index:
                raise ValueError(
                    f"{edge.label} vehicular jump edge must be a self-loop"
                )
            if (
                _transport_state_charge_diffusivity_trace_average_m2_s(
                    transport_states[edge.source_index],
                    temperature_K,
                )
                > REVERSE_DIFFUSION_TOLERANCE
            ):
                raise ValueError(
                    f"{edge.label} represents vehicular diffusion both continuously and as jumps"
                )
        elif edge.kind is MarkovAdditiveEdgeKind.STRUCTURAL_HOP:
            continue
        else:
            raise ValueError(f"{edge.label}.kind is not supported")


def _validate_transport_states(
    transport_states: tuple[TransportState, ...],
    stationary_probabilities: np.ndarray,
    state_concentrations_mol_m3: np.ndarray,
    state_count: int,
    temperature_K: float,
) -> None:
    if len(transport_states) != state_count:
        raise ValueError("transport_states must have one record per state")
    for state_index, transport_state in enumerate(transport_states):
        if transport_state.label == "":
            raise ValueError(f"transport_states[{state_index}].label is empty")
        probability_delta = abs(
            transport_state.probability - float(stationary_probabilities[state_index])
        )
        if probability_delta > REVERSE_DIFFUSION_TOLERANCE:
            raise ValueError(
                f"transport_states[{state_index}].probability does not match stationary probability"
            )
        concentration_delta = abs(
            transport_state.concentration_mol_m3
            - float(state_concentrations_mol_m3[state_index])
        )
        if concentration_delta > REVERSE_DIFFUSION_TOLERANCE:
            raise ValueError(
                f"transport_states[{state_index}].concentration_mol_m3 does not match state concentration"
            )
        _assert_positive_finite(
            transport_state.concentration_mol_m3,
            f"transport_states[{state_index}].concentration_mol_m3",
        )
        if not math.isfinite(transport_state.free_energy_J_mol):
            raise ValueError(
                f"transport_states[{state_index}].free_energy_J_mol must be finite"
            )
        if transport_state.atmosphere_state_lifetime_s < 0.0 or (
            not math.isfinite(transport_state.atmosphere_state_lifetime_s)
            and not math.isinf(transport_state.atmosphere_state_lifetime_s)
        ):
            raise ValueError(
                f"transport_states[{state_index}].atmosphere_state_lifetime_s "
                "must be nonnegative or infinite"
            )
        if transport_state.atmosphere_relaxation_time_s < 0.0 or (
            not math.isfinite(transport_state.atmosphere_relaxation_time_s)
            and not math.isinf(transport_state.atmosphere_relaxation_time_s)
        ):
            raise ValueError(
                f"transport_states[{state_index}].atmosphere_relaxation_time_s "
                "must be nonnegative or infinite"
            )
        if (
            transport_state.atmosphere_lifetime_gate < 0.0
            or transport_state.atmosphere_lifetime_gate > 1.0
            or not math.isfinite(transport_state.atmosphere_lifetime_gate)
        ):
            raise ValueError(
                f"transport_states[{state_index}].atmosphere_lifetime_gate "
                "must be finite and in [0, 1]"
            )
        if (
            transport_state.atmosphere_diagnostic_lifetime_gate < 0.0
            or transport_state.atmosphere_diagnostic_lifetime_gate > 1.0
            or not math.isfinite(transport_state.atmosphere_diagnostic_lifetime_gate)
        ):
            raise ValueError(
                f"transport_states[{state_index}].atmosphere_diagnostic_lifetime_gate "
                "must be finite and in [0, 1]"
            )
        _validate_relaxation_dynamic_response(
            transport_state.relaxation_dynamic_response
        )
        if (
            transport_state.relaxation_lifetime_gate < 0.0
            or transport_state.relaxation_lifetime_gate > 1.0
            or not math.isfinite(transport_state.relaxation_lifetime_gate)
        ):
            raise ValueError(
                f"transport_states[{state_index}].relaxation_lifetime_gate "
                "must be finite and in [0, 1]"
            )
        center_labels = tuple(
            center.label for center in transport_state.charged_centers
        )
        if len(set(center_labels)) != len(center_labels):
            raise ValueError(
                f"transport_states[{state_index}] has duplicate charged-center labels"
            )
        for center in transport_state.charged_centers:
            if center.label == "":
                raise ValueError(
                    f"transport_states[{state_index}] has empty charged-center label"
                )
            if not math.isfinite(center.charge):
                raise ValueError(
                    f"{transport_state.label}.{center.label}.charge must be finite"
                )
            _assert_positive_finite(
                center.hydrodynamic_radius_m,
                f"{transport_state.label}.{center.label}.hydrodynamic_radius_m",
            )
            _assert_positive_finite(
                center.shape_factor,
                f"{transport_state.label}.{center.label}.shape_factor",
            )
            _assert_positive_finite(
                center.local_diffusion_m2_s,
                f"{transport_state.label}.{center.label}.local_diffusion_m2_s",
            )
            _validate_charged_center_charge_cloud_descriptor(
                transport_state.label, center
            )
        for constraint in transport_state.constraints:
            if len(constraint.labels) == 0:
                raise ValueError(f"{transport_state.label} constraint labels are empty")
            if len(constraint.vector) != len(transport_state.charged_centers):
                raise ValueError(
                    f"{transport_state.label}.{constraint.labels} vector length mismatch"
                )
            for label in constraint.labels:
                if label not in center_labels:
                    raise ValueError(
                        f"{transport_state.label}.{constraint.labels} references unknown center {label}"
                    )
            constraint_vector = np.asarray(constraint.vector, dtype=float)
            if not np.all(np.isfinite(constraint_vector)):
                raise ValueError(
                    f"{transport_state.label}.{constraint.labels} vector contains non-finite values"
                )
            _assert_nonnegative_finite(
                constraint.lifetime_s,
                f"{transport_state.label}.{constraint.labels}.lifetime_s",
            )
            _assert_nonnegative_finite(
                constraint.atmosphere_lifetime_s,
                f"{transport_state.label}.{constraint.labels}.atmosphere_lifetime_s",
            )
            _assert_positive_finite(
                constraint.length_m,
                f"{transport_state.label}.{constraint.labels}.length_m",
            )
        _validate_transport_state_atmosphere_field(
            transport_state,
            len(transport_state.charged_centers),
            "atmosphere_resistance_kg_s",
            transport_state.atmosphere_resistance_kg_s,
        )
        _validate_transport_state_atmosphere_field(
            transport_state,
            len(transport_state.charged_centers),
            "atmosphere_resistance_before_lifetime_gate_kg_s",
            transport_state.atmosphere_resistance_before_lifetime_gate_kg_s,
        )
        _validate_transport_state_atmosphere_field(
            transport_state,
            len(transport_state.charged_centers),
            "relaxation_resistance_before_gate_kg_s",
            transport_state.relaxation_resistance_before_gate_kg_s,
        )
        _validate_transport_state_atmosphere_field(
            transport_state,
            len(transport_state.charged_centers),
            "relaxation_resistance_after_gate_kg_s",
            transport_state.relaxation_resistance_after_gate_kg_s,
        )
        for axis_index in range(AXIS_COUNT):
            axis_charge_diffusivity = _transport_state_charge_diffusivity_axis_m2_s(
                transport_state,
                temperature_K,
                axis_index,
            )
            if axis_charge_diffusivity < -REVERSE_DIFFUSION_TOLERANCE:
                raise ValueError(
                    f"transport_states[{state_index}] charge diffusivity is negative on axis {axis_index}"
                )


def _validate_transport_state_atmosphere_field(
    transport_state: TransportState,
    center_count: int,
    field_name: str,
    field_value: tuple[tuple[float, ...], ...],
) -> None:
    atmosphere_matrix = np.asarray(field_value, dtype=float)
    if center_count == 0:
        if atmosphere_matrix.shape != (0,):
            raise ValueError(f"{transport_state.label}.{field_name} must be empty")
        return
    if atmosphere_matrix.shape != (center_count, center_count):
        raise ValueError(
            f"{transport_state.label}.{field_name} shape "
            f"{atmosphere_matrix.shape} does not match charged-center count {center_count}"
        )
    if not np.all(np.isfinite(atmosphere_matrix)):
        raise ValueError(
            f"{transport_state.label}.{field_name} contains non-finite values"
        )
    if not np.allclose(atmosphere_matrix, atmosphere_matrix.T):
        raise ValueError(f"{transport_state.label}.{field_name} must be symmetric")
    atmosphere_eigenvalues = np.linalg.eigvalsh(atmosphere_matrix)
    if float(np.min(atmosphere_eigenvalues)) < -REVERSE_DIFFUSION_TOLERANCE:
        raise ValueError(
            f"{transport_state.label}.{field_name} must be positive semidefinite"
        )


def _validate_charged_center_charge_cloud_descriptor(
    transport_state_label: str,
    charged_center: ChargedCenter,
) -> None:
    descriptor_context = f"{transport_state_label}.{charged_center.label}.charge_cloud"
    if charged_center.charge_cloud_source == "":
        raise ValueError(f"{descriptor_context}_source must not be empty")
    if charged_center.charge_cloud_site_count < 0:
        raise ValueError(
            f"{descriptor_context}_site_count must be nonnegative, got "
            f"{charged_center.charge_cloud_site_count}"
        )
    _assert_nonnegative_finite(
        charged_center.charge_cloud_radius_A,
        f"{descriptor_context}_radius_A",
    )
    if charged_center.charge_cloud_radius_available:
        if charged_center.charge_cloud_site_count <= 0:
            raise ValueError(
                f"{descriptor_context}_site_count must be positive when radius is available"
            )
    else:
        if charged_center.charge_cloud_radius_A != 0.0:
            raise ValueError(
                f"{descriptor_context}_radius_A must be zero when radius is unavailable"
            )
        if charged_center.charge_cloud_site_count != 0:
            raise ValueError(
                f"{descriptor_context}_site_count must be zero when radius is unavailable"
            )


def _transport_state_charge_diffusivity_trace_average_m2_s(
    transport_state: TransportState,
    temperature_K: float,
) -> float:
    return float(
        math.fsum(
            _transport_state_charge_diffusivity_axis_m2_s(
                transport_state,
                temperature_K,
                axis_index,
            )
            for axis_index in range(AXIS_COUNT)
        )
        / AXIS_COUNT
    )


def _edge_displacement_norm_m(edge: MarkovAdditiveEdge) -> float:
    return math.sqrt(
        math.fsum(
            float(component) * float(component) for component in edge.displacement_m
        )
    )


def _strict_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _strict_matrix(
    values: Sequence[Sequence[float]] | np.ndarray, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _detailed_balance_residual(
    generator_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
) -> float:
    flux = stationary_probabilities[:, None] * generator_s_inv
    raw_residual = float(np.max(np.abs(flux - flux.T)))
    flux_scale = max(1.0, float(np.max(np.abs(flux))))
    roundoff_tolerance = float(np.spacing(flux_scale)) * float(generator_s_inv.shape[0])
    if raw_residual <= roundoff_tolerance:
        return 0.0
    return raw_residual


def _generator_row_sum_residual(generator_s_inv: np.ndarray) -> float:
    residuals = []
    for row_index in range(generator_s_inv.shape[0]):
        residuals.append(
            abs(
                math.fsum(
                    float(generator_s_inv[row_index, column_index])
                    for column_index in range(generator_s_inv.shape[1])
                )
            )
        )
    raw_residual = max(residuals)
    rate_scale = max(1.0, float(np.max(np.abs(generator_s_inv))))
    roundoff_tolerance = float(np.spacing(rate_scale)) * float(generator_s_inv.shape[0])
    if raw_residual <= roundoff_tolerance:
        return 0.0
    return raw_residual


def _stationary_distribution_residual(
    generator_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
) -> float:
    residuals = []
    for column_index in range(generator_s_inv.shape[1]):
        residuals.append(
            abs(
                math.fsum(
                    float(stationary_probabilities[row_index])
                    * float(generator_s_inv[row_index, column_index])
                    for row_index in range(generator_s_inv.shape[0])
                )
            )
        )
    raw_residual = max(residuals)
    flux_scale = max(
        1.0,
        float(np.max(np.abs(stationary_probabilities[:, None] * generator_s_inv))),
    )
    roundoff_tolerance = float(np.spacing(flux_scale)) * float(generator_s_inv.shape[0])
    if raw_residual <= roundoff_tolerance:
        return 0.0
    return raw_residual


def _linear_solve_tolerance(generator_s_inv: np.ndarray) -> float:
    rate_scale = max(1.0, float(np.max(np.abs(generator_s_inv))))
    return 1.0e-10 * rate_scale


def _neutral_species_props(species_name: str) -> Mapping[str, object]:
    if species_name in SOLVENTS:
        return SOLVENTS[species_name]
    if species_name in ADDITIVES:
        props = ADDITIVES[species_name]
        if not _is_ionic_source_props(props):
            return props
    raise ValueError(f"Species {species_name} is not a neutral liquid species")


def _is_ionic_source_props(props: Mapping[str, object]) -> bool:
    has_cation_identity = "cation" in props or "cation_radius" in props
    return has_cation_identity and "anion" in props and "Lambda_0" in props


def _require_species(
    species_map: Mapping[str, Mapping[str, object]],
    species_name: str,
    species_kind: str,
) -> Mapping[str, object]:
    if species_name not in species_map:
        raise ValueError(f"Unknown {species_kind} species {species_name}")
    return species_map[species_name]


def _require_mapping_float(
    mapping: Mapping[str, float],
    key: str,
    context: str,
) -> float:
    if key not in mapping:
        raise ValueError(f"Missing {context}.{key}")
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"{context}.{key} must be finite")
    return value


def _require_feature_id(motif: ChemicalMotif) -> str:
    if motif.feature_id is None:
        raise ValueError(f"Motif {motif.label} has no anion feature")
    return motif.feature_id


def _validate_atmosphere_bath_basis(atmosphere_bath_basis: str) -> None:
    if atmosphere_bath_basis not in SUPPORTED_ATMOSPHERE_BATH_BASES:
        raise ValueError(
            f"Unsupported atmosphere_bath_basis {atmosphere_bath_basis!r}; "
            f"expected one of {SUPPORTED_ATMOSPHERE_BATH_BASES}"
        )


def _validate_relaxation_dynamic_response(relaxation_dynamic_response: str) -> None:
    if relaxation_dynamic_response not in SUPPORTED_RELAXATION_DYNAMIC_RESPONSES:
        raise ValueError(
            f"Unsupported relaxation_dynamic_response {relaxation_dynamic_response!r}; "
            f"expected one of {SUPPORTED_RELAXATION_DYNAMIC_RESPONSES}"
        )


def _validate_anion_diagonal_relaxation_form_factor(
    anion_diagonal_relaxation_form_factor: str,
) -> None:
    if (
        anion_diagonal_relaxation_form_factor
        not in SUPPORTED_ANION_DIAGONAL_RELAXATION_FORM_FACTORS
    ):
        raise ValueError(
            "Unsupported anion_diagonal_relaxation_form_factor "
            f"{anion_diagonal_relaxation_form_factor!r}; expected one of "
            f"{SUPPORTED_ANION_DIAGONAL_RELAXATION_FORM_FACTORS}"
        )


def _nonnegative_ratio(numerator: float, denominator: float, context: str) -> float:
    _assert_nonnegative_finite(numerator, f"{context}.numerator")
    _assert_positive_finite(denominator, f"{context}.denominator")
    ratio = numerator / denominator
    _assert_nonnegative_finite(ratio, context)
    return ratio


def _assert_positive_finite(value: float, context: str) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{context} must be positive and finite, got {value}")


def _assert_nonnegative_finite(value: float, context: str) -> None:
    if value < 0.0 or not math.isfinite(value):
        raise ValueError(f"{context} must be nonnegative and finite, got {value}")


def _validate_unit_interval(value: float, context: str) -> None:
    if value < 0.0 or value > 1.0 or not math.isfinite(value):
        raise ValueError(f"{context} must be in [0, 1], got {value}")
