"""Analytic descriptor-to-Mori primitive generator for liquid electrolytes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, TypeAlias

import numpy as np

from constants import E_CHARGE, EPS_0, F, K_B, N_A, R, S_M_TO_MS_CM
from conductivity.finite_markov_additive_green_kubo import (
    MarkovAdditiveConductivityResult,
    MarkovAdditiveEvent,
    MarkovAdditiveEventFamilyAttribution,
    compute_markov_additive_green_kubo_conductivity,
    compute_markov_additive_event_family_attribution,
    MarkovAdditiveConductivityInput,
)
from conductivity.finite_markov_conductivity import (
    ChargedCenter,
    state_form_factor_atmosphere_resistance_kg_s,
)
from conductivity.finite_mori_conductivity import (
    ProjectedMoriConductivityInput,
    ProjectedMoriConductivityResult,
    compute_projected_mori_conductivity,
)
from conductivity.ion_atmosphere import (
    BulkIonAtmosphereInput,
    BulkIonAtmosphereState,
    build_bulk_ion_atmosphere_state,
)
from utils.config_cache import load_physics_config, require_config
SpeciesRecord: TypeAlias = Mapping[str, float]


ANGSTROM_TO_M = 1.0e-10
CP_TO_PA_S = 1.0e-3
MOLARITY_TO_MOL_M3 = 1000.0
KJ_TO_J = 1000.0
AXIS_COUNT = 3
STOKES_SPHERE_FACTOR = 6.0  # Explicit analytical constant: Stokes sphere drag coefficient in D = kBT/(6*pi*eta*R).
COULOMB_DENOMINATOR_FACTOR = 4.0  # Explicit analytical constant: Coulomb denominator is 4*pi*epsilon.
BJERRUM_REFERENCE_DIELECTRIC = 20.0  # Explicit registry reference: bjerrum_K_A_ref entries are documented at epsilon_ref = 20.
ATMOSPHERE_DIELECTRIC_FLOOR = 3.0  # Explicit user-declared lower bound for concentrated-electrolyte atmosphere screening.
ASSOCIATION_DIELECTRIC_SENSITIVITY = 0.25  # Explicit user-declared sensitivity for bounded association dielectric response.
ASSOCIATION_EXPONENT_CLIP = 2.0  # Explicit user-declared cap on the dimensionless association dielectric exponent.
ASSOCIATION_PAIR_CAP_MULTIPLIER = 5.0  # Explicit user-declared fallback cap: K_pair <= 5 * bjerrum_K_A_ref.
K_LI2A_STEP_DEFAULT_M_INV = 0.25  # Explicit user-declared fallback Li2A+ step association constant.
K_LIA2_STEP_DEFAULT_M_INV = 0.25  # Explicit user-declared fallback LiA2- step association constant.
K_LI2A2_STEP_DEFAULT_M_INV = 0.05  # Explicit user-declared fallback Li2A2 neutral step association constant.
AGGREGATE_PACKING_CONCENTRATION_M = 2.5  # Explicit user-declared aggregate steric saturation concentration.
AGGREGATE_STERIC_POWER = 2.0  # Explicit user-declared aggregate steric saturation exponent.
BACKJUMP_CAGE_STERIC_ONSET = 0.24  # Explicit user-declared oriented-cage steric onset.
BACKJUMP_CAGE_STERIC_FULL = 0.42  # Explicit user-declared oriented-cage steric full-response value.
BACKJUMP_COMPACT_ANION_RADIUS_A = 3.2  # Explicit user-declared compact-anion radius scale.
BACKJUMP_COMPACT_ANION_SHARPNESS = 4.0  # Explicit user-declared compact-anion sharpness.
BACKJUMP_HIGH_SALT_ONSET_M = 0.8  # Explicit user-declared high-salt onset.
BACKJUMP_HIGH_SALT_FULL_M = 1.4  # Explicit user-declared high-salt full-response value.
BACKJUMP_LOW_DONOR_FULL = 10.0  # Explicit user-declared low-donor full-response value.
BACKJUMP_LOW_DONOR_OFF = 18.0  # Explicit user-declared low-donor off-response value.
BACKJUMP_F_CAGE_MAX = 0.20  # Explicit user-declared maximum oriented-cage occupancy.
BACKJUMP_ATTEMPT_FRACTION_MAX = 0.50  # Explicit user-declared maximum Li attempt budget routed through cage memory.
BACKJUMP_PROBABILITY_MAX = 0.85  # Explicit user-declared maximum backjump probability.
BACKJUMP_MIN_DRIVER_FOR_POINT = 0.15  # Explicit user-declared minimum driver for point ablation activation.
FREE_LI_OBSTRUCTION_MAX = 1.50  # Explicit user-declared free-Li local obstruction maximum.
COMPACT_ANION_OBSTRUCTION_MAX = 1.25  # Explicit user-declared compact-anion local obstruction maximum.
OBSTRUCTION_STERIC_ONSET = 0.22  # Explicit user-declared local-obstruction steric onset.
OBSTRUCTION_STERIC_FULL = 0.42  # Explicit user-declared local-obstruction steric full-response value.
OBSTRUCTION_COMPACT_ANION_RADIUS_A = 3.2  # Explicit user-declared compact-anion radius scale for local obstruction.
OBSTRUCTION_COMPACT_ANION_SHARPNESS = 4.0  # Explicit user-declared compact-anion obstruction sharpness.
OBSTRUCTION_HIGH_SALT_ONSET_M = 0.8  # Explicit user-declared high-salt onset for local obstruction.
OBSTRUCTION_HIGH_SALT_FULL_M = 1.4  # Explicit user-declared high-salt full-response value for local obstruction.
OBSTRUCTION_LOW_DONOR_ONSET = 18.0  # Explicit user-declared low-donor off-response edge for local obstruction.
OBSTRUCTION_LOW_DONOR_FULL = 10.0  # Explicit user-declared low-donor full-response edge for local obstruction.
NORMALIZED_PROBABILITY_SUM = 1.0
HALF_FACTOR = 0.5
PAIR_STATE_DISTANCE_FACTOR_SSIP = 2.0
PAIR_STATE_DISTANCE_FACTOR_CIP = 1.0
MASS_BALANCE_MAX_ITERATIONS = 80  # Explicit numerical cap: Newton iterations for the finite mass-action solve.
MASS_BALANCE_FINITE_DIFFERENCE_STEP = 1.0e-6
MASS_BALANCE_ABSOLUTE_TOLERANCE_M = 1.0e-11
MASS_BALANCE_DAMPING_ATTEMPTS = 16  # Explicit numerical cap: residual-reducing Newton line-search attempts.
SOLVENT_FRACTION_SUM_TOLERANCE = 1.0e-8
PSD_EIGENVALUE_ABSOLUTE_TOLERANCE = 1.0e-20  # Explicit floating-point tolerance for PSD validation after analytic matrix similarity transforms.
NOT_APPLICABLE_FEATURE = "not_applicable"
SHARED_LITHIUM_POOL = "shared_lithium_pool"
LITHIUM_CARRIER_LABEL = "Li"
FINITE_SIZE_BULK_ATMOSPHERE_SOLVER = "finite_size_bulk_pnp_stokes_l1_cell"
STATE_KIND_FREE_LI = "FREE_LI"
STATE_KIND_FREE_ANION = "FREE_ANION"
STATE_KIND_SSIP = "SSIP"
STATE_KIND_CIP = "CIP"
STATE_KIND_LI2A_PLUS = "LI2A_PLUS"
STATE_KIND_LIA2_MINUS = "LIA2_MINUS"
STATE_KIND_LI2A2_NEUTRAL = "LI2A2_NEUTRAL"
EVENT_FAMILY_ORDINARY_FREE_LI_TRANSLATION = "ordinary_free_Li_translation"
EVENT_FAMILY_ORDINARY_FREE_ANION_TRANSLATION = "ordinary_free_anion_translation"
EVENT_FAMILY_BOUND_STATE_TRANSLATION = "bound_state_translation"
EVENT_FAMILY_ORIENTATION_RELAXATION = "orientation_relaxation"
EVENT_FAMILY_CHEMICAL_INTERCONVERSION = "chemical_interconversion"
EVENT_FAMILY_STATIC_CARRIER_CAGE_EXCHANGE = "static_carrier_cage_exchange"
EVENT_FAMILY_LI_BACKJUMP_CAGE = "Li_backjump_cage"
EVENT_FAMILY_TIMESCALE_ATMOSPHERE_CAPTURE_BACKTRACKING = (
    "timescale_atmosphere_capture_backtracking"
)
EVENT_FAMILY_TIMESCALE_STRUCTURAL_CAGE_EXCHANGE = (
    "timescale_structural_cage_zero_displacement_exchange"
)
DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION = "analytic_mori_dense_free_volume_obstruction"
SELECTIVE_CARRIER_CAGE_CONFIG_SECTION = "analytic_mori_selective_carrier_cage"
DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION = (
    "analytic_mori_descriptor_atmosphere_release"
)
TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION = (
    "analytic_mori_timescale_structural_memory"
)
TRANSLATION_EVENT_AXES = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
BOUND_ORIENTATION_LABELS = ("plus_x", "minus_x", "plus_y", "minus_y", "plus_z", "minus_z")
ANALYTIC_MORI_ABLATION_BASELINE = "baseline"
ANALYTIC_MORI_ABLATION_ASSOCIATION_OFF = "association_off"
ANALYTIC_MORI_ABLATION_HIGHER_AGGREGATES_OFF = "higher_aggregates_off"
ANALYTIC_MORI_ABLATION_BINDING_RESISTANCE_OFF = "binding_resistance_off"
ANALYTIC_MORI_ABLATION_ATMOSPHERE_RESISTANCE_OFF = "atmosphere_resistance_off"
ANALYTIC_MORI_ABLATION_JONES_DOLE_VISCOSITY_OFF = "jones_dole_viscosity_off"
ANALYTIC_MORI_ABLATION_DIELECTRIC_DECREMENT_OFF = "dielectric_decrement_off"
ANALYTIC_MORI_ABLATION_FREE_ION_NE = "free_ion_ne"
ANALYTIC_MORI_ABLATION_CARRIER_CAGE_POINT_SUBSTATES = "carrier_cage_point_substates"
ANALYTIC_MORI_ABLATION_BACKJUMP_CAGE_MEMORY = "backjump_cage_memory"
ANALYTIC_MORI_ABLATION_FREE_LI_LOCAL_OBSTRUCTION = "free_li_local_obstruction"
ANALYTIC_MORI_ABLATION_COMPACT_ANION_LOCAL_OBSTRUCTION = "compact_anion_local_obstruction"
ANALYTIC_MORI_ABLATION_FREE_LI_PLUS_COMPACT_ANION_LOCAL_OBSTRUCTION = (
    "free_li_plus_compact_anion_local_obstruction"
)
ANALYTIC_MORI_ABLATION_SELECTIVE_CARRIER_CAGE_ONLY = "selective_carrier_cage_only"
ANALYTIC_MORI_ABLATION_DESCRIPTOR_ATMOSPHERE_RELAXATION_RELEASE_ONLY = (
    "descriptor_atmosphere_relaxation_release_only"
)
ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_RELEASE = (
    "selective_cage_plus_descriptor_release"
)
ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_REL_AND_EP_RELEASE = (
    "selective_cage_plus_descriptor_rel_and_ep_release"
)
ANALYTIC_MORI_ABLATION_TIMESCALE_STRUCTURAL_CAGE_MEMORY = (
    "timescale_structural_cage_memory"
)
SUPPORTED_ANALYTIC_MORI_ABLATIONS = (
    ANALYTIC_MORI_ABLATION_BASELINE,
    ANALYTIC_MORI_ABLATION_ASSOCIATION_OFF,
    ANALYTIC_MORI_ABLATION_HIGHER_AGGREGATES_OFF,
    ANALYTIC_MORI_ABLATION_BINDING_RESISTANCE_OFF,
    ANALYTIC_MORI_ABLATION_ATMOSPHERE_RESISTANCE_OFF,
    ANALYTIC_MORI_ABLATION_JONES_DOLE_VISCOSITY_OFF,
    ANALYTIC_MORI_ABLATION_DIELECTRIC_DECREMENT_OFF,
    ANALYTIC_MORI_ABLATION_FREE_ION_NE,
    ANALYTIC_MORI_ABLATION_CARRIER_CAGE_POINT_SUBSTATES,
    ANALYTIC_MORI_ABLATION_BACKJUMP_CAGE_MEMORY,
    ANALYTIC_MORI_ABLATION_FREE_LI_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_COMPACT_ANION_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_FREE_LI_PLUS_COMPACT_ANION_LOCAL_OBSTRUCTION,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CARRIER_CAGE_ONLY,
    ANALYTIC_MORI_ABLATION_DESCRIPTOR_ATMOSPHERE_RELAXATION_RELEASE_ONLY,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_RELEASE,
    ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_REL_AND_EP_RELEASE,
    ANALYTIC_MORI_ABLATION_TIMESCALE_STRUCTURAL_CAGE_MEMORY,
)


@dataclass(frozen=True)
class AnalyticMoriRecipe:
    solvent_volume_fractions: Mapping[str, float]
    salt_molarities_M: Mapping[str, float]
    additive_weight_fractions: Mapping[str, float]
    temperature_K: float
    active_volume_m3: float


@dataclass(frozen=True)
class StructuralPrimitiveUncertaintyBudget:
    association_logK_interval: tuple[float, float]
    dielectric_decrement_scale_interval: tuple[float, float]
    jones_dole_scale_interval: tuple[float, float]
    atmosphere_ep_scale_interval: tuple[float, float]
    atmosphere_rel_scale_interval: tuple[float, float]
    lithium_charge_cloud_radius_interval_A: tuple[float, float]
    anion_charge_cloud_radius_interval_A: tuple[float, float]
    cage_trapping_fraction_interval: tuple[float, float]
    jump_length_scale_interval: tuple[float, float]
    conversion_rate_scale_interval: tuple[float, float]
    forced_free_li_translation_scale_interval: tuple[float, float]
    forced_compact_anion_translation_scale_interval: tuple[float, float]
    certificate_threshold_mS_cm: float


@dataclass(frozen=True)
class AnalyticPrimitiveProductionPolicy:
    enable_carrier_relaxation_self_form_factor: bool
    enable_carrier_cage_point_substates: bool
    enable_backjump_cage_memory: bool
    include_cage_in_structural_uncertainty: bool


@dataclass(frozen=True)
class AnalyticMoriSpeciesCatalog:
    solvents: Mapping[str, SpeciesRecord]
    salts: Mapping[str, SpeciesRecord]
    additives: Mapping[str, SpeciesRecord]
    cations: Mapping[str, SpeciesRecord]


@dataclass(frozen=True)
class AnalyticAssociationConstants:
    anion_feature_id: str
    source_salt_name: str
    K_ssip_M_inv: float
    K_cip_M_inv: float
    K_li2a_M2_inv: float
    K_lia2_M2_inv: float
    K_li2a2_M3_inv: float


@dataclass(frozen=True)
class AnalyticDielectricHeads:
    epsilon_mixture: float
    epsilon_association: float
    epsilon_atmosphere: float


@dataclass(frozen=True)
class AnalyticAtmosphereCarrierRelaxationFormFactor:
    carrier_label: str
    carrier_kind: str
    source_species_name: str
    charge_cloud_radius_A: float
    form_factor_squared: float


@dataclass(frozen=True)
class AnalyticAtmosphereBlockDiagnostics:
    electrophoretic_trace_kg_s: float
    relaxation_trace_kg_s: float
    relaxation_lithium_self_trace_kg_s: float
    relaxation_anion_self_trace_kg_s: float
    relaxation_lithium_anion_cross_frobenius_kg_s: float
    relaxation_anion_anion_cross_frobenius_kg_s: float
    lithium_form_factor_squared: float
    minimum_anion_form_factor_squared: float
    minimum_lithium_anion_cross_form_factor: float


@dataclass(frozen=True)
class AnalyticCarrierCagePrimitive:
    carrier_label: str
    carrier_kind: str
    mobile_fraction: float
    caged_fraction: float
    caged_diffusion_scale: float
    exchange_rate_s_inv: float
    steric_driver: float
    compact_driver: float


@dataclass(frozen=True)
class AnalyticSelectiveCarrierCagePrimitive:
    carrier_label: str
    carrier_kind: str
    dense_driver: float
    descriptor_release_driver: float
    selective_cage_driver: float
    caged_fraction: float
    caged_diffusion_scale: float


@dataclass(frozen=True)
class AnalyticDescriptorAtmosphereReleasePrimitive:
    mixture_release_descriptor: float
    high_viscosity_driver: float
    low_dielectric_driver: float
    low_donor_driver: float
    weak_cage_driver: float
    release_driver: float
    relaxation_scale: float
    electrophoretic_scale: float


@dataclass(frozen=True)
class AnalyticBackjumpCagePrimitive:
    carrier_label: str
    carrier_kind: str
    cage_driver: float
    steric_driver: float
    compact_anion_driver: float
    carbonate_driver: float
    high_salt_driver: float
    low_donor_driver: float
    cage_occupancy_fraction: float
    attempt_fraction: float
    backjump_probability: float
    exit_rate_s_inv: float
    jump_length_m: float
    ordinary_translation_fraction: float
    direct_axis_density_m2_s_mol_m3: float
    direct_sigma_mS_cm: float
    point_active: bool


@dataclass(frozen=True)
class AnalyticFreeCarrierObstructionPrimitive:
    carrier_label: str
    carrier_kind: str
    source_species_name: str
    obstruction_factor: float
    diffusion_scale: float
    dense_free_volume_driver: float
    dense_free_volume_obstruction_factor: float
    steric_driver: float
    compact_anion_driver: float
    carbonate_driver: float
    high_salt_driver: float
    low_donor_driver: float


@dataclass(frozen=True)
class AnalyticTimescaleStructuralMemoryPrimitive:
    carrier_label: str
    carrier_kind: str
    local_diffusivity_m2_s: float
    atmosphere_relaxation_diffusivity_m2_s: float
    jump_length_m: float
    tau_hop_s: float
    tau_atmosphere_s: float
    tau_structural_s: float
    de_hop_structural: float
    atmosphere_structural_ratio: float
    size_void_ratio: float
    atmosphere_capture_fraction: float
    structural_cage_fraction: float
    k_capture_s_inv: float
    k_atmosphere_exit_s_inv: float
    k_structural_capture_s_inv: float
    k_structural_release_s_inv: float
    mobile_concentration_mol_m3: float
    atmosphere_concentration_per_orientation_mol_m3: float
    structural_cage_concentration_per_orientation_mol_m3: float


@dataclass(frozen=True)
class AnalyticTransportStatePrimitive:
    label: str
    state_kind: str
    anion_feature_id: str
    source_salt_name: str
    concentration_mol_m3: float
    charge_vector: tuple[float, ...]
    local_resistance_matrix_kg_s: np.ndarray
    binding_resistance_matrix_kg_s: np.ndarray
    atmosphere_resistance_matrix_kg_s: np.ndarray
    resistance_matrix_kg_s: np.ndarray
    charge_diffusivity_m2_s: float
    current_relaxation_length_m: float
    standard_free_energy_J_mol: float


@dataclass(frozen=True)
class AnalyticMoriMassBalance:
    species_labels: tuple[str, ...]
    total_concentrations_M: tuple[float, ...]
    free_activities_M: tuple[float, ...]
    residuals_M: tuple[float, ...]
    max_abs_residual_M: float


@dataclass(frozen=True)
class AnalyticPrimitiveUncertaintyCertificate:
    sigma_min_mS_cm: float
    sigma_max_mS_cm: float
    half_width_mS_cm: float
    dominant_uncertainty_head: str
    threshold_mS_cm: float
    certified_0p25_mS_cm: bool


@dataclass(frozen=True)
class AnalyticMoriPrimitiveResult:
    sigma_mS_cm: float
    sigma_S_m: float
    direct_mori_result: ProjectedMoriConductivityResult
    markov_additive_result: MarkovAdditiveConductivityResult
    mori_input: ProjectedMoriConductivityInput
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: tuple[float, ...]
    generator_s_inv: np.ndarray
    markov_additive_events: tuple[MarkovAdditiveEvent, ...]
    markov_event_family_attributions: tuple[
        MarkovAdditiveEventFamilyAttribution,
        ...,
    ]
    transport_states: tuple[AnalyticTransportStatePrimitive, ...]
    bulk_ion_atmosphere_state: BulkIonAtmosphereState
    atmosphere_carrier_relaxation_form_factors: tuple[
        AnalyticAtmosphereCarrierRelaxationFormFactor,
        ...,
    ]
    atmosphere_block_diagnostics: AnalyticAtmosphereBlockDiagnostics
    carrier_cage_primitives: tuple[AnalyticCarrierCagePrimitive, ...]
    selective_carrier_cage_primitives: tuple[
        AnalyticSelectiveCarrierCagePrimitive,
        ...,
    ]
    descriptor_atmosphere_release_primitive: (
        AnalyticDescriptorAtmosphereReleasePrimitive
    )
    timescale_structural_memory_primitives: tuple[
        AnalyticTimescaleStructuralMemoryPrimitive,
        ...,
    ]
    free_carrier_obstruction_primitives: tuple[
        AnalyticFreeCarrierObstructionPrimitive,
        ...,
    ]
    backjump_cage_primitives: tuple[AnalyticBackjumpCagePrimitive, ...]
    association_constants: tuple[AnalyticAssociationConstants, ...]
    mass_balance: AnalyticMoriMassBalance
    dielectric_heads: AnalyticDielectricHeads
    effective_dielectric: float
    effective_viscosity_cP: float
    active_volume_m3: float
    conductivity_uncertainty: AnalyticPrimitiveUncertaintyCertificate


@dataclass(frozen=True)
class _MassActionTemplate:
    label: str
    state_kind: str
    anion_index: int
    li_count: int
    anion_count: int
    equilibrium_constant: float


@dataclass(frozen=True)
class _ResolvedAnion:
    feature_id: str
    source_salt_name: str
    salt_record: SpeciesRecord
    total_concentration_M: float


@dataclass(frozen=True)
class _MassBalanceSolution:
    free_lithium_activity_M: float
    free_anion_activities_M: tuple[float, ...]
    state_concentrations_M_by_label: Mapping[str, float]
    mass_balance: AnalyticMoriMassBalance


@dataclass(frozen=True)
class _TransportPrimitiveBundle:
    transport_states: tuple[AnalyticTransportStatePrimitive, ...]
    bulk_ion_atmosphere_state: BulkIonAtmosphereState
    atmosphere_carrier_relaxation_form_factors: tuple[
        AnalyticAtmosphereCarrierRelaxationFormFactor,
        ...,
    ]
    atmosphere_block_diagnostics: AnalyticAtmosphereBlockDiagnostics
    carrier_cage_primitives: tuple[AnalyticCarrierCagePrimitive, ...]
    selective_carrier_cage_primitives: tuple[
        AnalyticSelectiveCarrierCagePrimitive,
        ...,
    ]
    descriptor_atmosphere_release_primitive: (
        AnalyticDescriptorAtmosphereReleasePrimitive
    )
    timescale_structural_memory_primitives: tuple[
        AnalyticTimescaleStructuralMemoryPrimitive,
        ...,
    ]
    free_carrier_obstruction_primitives: tuple[
        AnalyticFreeCarrierObstructionPrimitive,
        ...,
    ]


@dataclass(frozen=True)
class _MarkovAdditiveModelState:
    label: str
    parent_transport_label: str
    transport_state: AnalyticTransportStatePrimitive
    concentration_mol_m3: float
    orientation_label: str
    orientation_vector: tuple[float, float, float]
    polarization_m: tuple[float, float, float]
    translation_diffusion_scale: float
    cage_group_label: str
    cage_state_kind: str
    cage_exchange_rate_s_inv: float
    backjump_group_label: str
    backjump_state_kind: str
    backjump_exit_rate_s_inv: float
    backjump_probability: float
    backjump_length_m: float
    timescale_memory_group_label: str
    timescale_memory_state_kind: str
    timescale_capture_rate_s_inv: float
    timescale_atmosphere_exit_rate_s_inv: float
    timescale_structural_capture_rate_s_inv: float
    timescale_structural_release_rate_s_inv: float
    timescale_jump_length_m: float
    chemical_conversion_enabled: bool


@dataclass(frozen=True)
class _DenseFreeVolumeObstructionParameters:
    free_li_obstruction_strength: float
    compact_anion_obstruction_strength: float
    steric_reference_fraction: float
    free_volume_power: float


@dataclass(frozen=True)
class _SelectiveCarrierCageParameters:
    caged_fraction_max: float
    caged_diffusion_scale_min: float
    dense_driver_power: float
    descriptor_release_suppression_strength: float


@dataclass(frozen=True)
class _DescriptorAtmosphereReleaseParameters:
    relaxation_release_strength: float
    electrophoretic_release_strength: float
    mixture_descriptor_on_value: float
    mixture_descriptor_full_value: float
    viscosity_on_cP: float
    viscosity_full_cP: float
    dielectric_on_value: float
    dielectric_full_value: float
    donor_on_value: float
    donor_full_value: float


@dataclass(frozen=True)
class _TimescaleStructuralMemoryParameters:
    cage_fraction_max: float
    jump_length_radius_scale: float
    structural_free_volume_barrier_strength: float
    packing_fraction_limit: float
    deborah_on_value: float
    deborah_full_value: float
    size_void_ratio_on_value: float
    size_void_ratio_full_value: float
    steric_on_fraction: float
    steric_full_fraction: float
    atmosphere_capture_fraction_max: float


@dataclass(frozen=True)
class _AnalyticPrimitiveOptions:
    association_enabled: bool
    higher_aggregates_enabled: bool
    binding_resistance_enabled: bool
    atmosphere_resistance_enabled: bool
    jones_dole_viscosity_enabled: bool
    dielectric_decrement_enabled: bool
    carrier_cage_point_substates_enabled: bool
    backjump_cage_memory_enabled: bool
    free_li_local_obstruction_enabled: bool
    compact_anion_local_obstruction_enabled: bool
    forced_free_li_translation_scale_enabled: bool
    forced_free_li_translation_scale: float
    forced_compact_anion_translation_scale_enabled: bool
    forced_compact_anion_translation_scale: float
    selective_carrier_cage_enabled: bool = False
    descriptor_atmosphere_relaxation_release_enabled: bool = False
    descriptor_atmosphere_ep_release_enabled: bool = False
    timescale_structural_cage_memory_enabled: bool = False


PRODUCTION_PRIMITIVE_POLICY = AnalyticPrimitiveProductionPolicy(
    enable_carrier_relaxation_self_form_factor=True,
    enable_carrier_cage_point_substates=False,
    enable_backjump_cage_memory=False,
    include_cage_in_structural_uncertainty=True,
)


def evaluate_analytic_mori_conductivity(
    recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> AnalyticMoriPrimitiveResult:
    """Generate analytic finite Mori primitives from registry descriptors."""

    return _evaluate_analytic_mori_conductivity_with_options(
        recipe,
        uncertainty_budget,
        species_catalog,
        _BASELINE_ANALYTIC_PRIMITIVE_OPTIONS,
    )


def evaluate_analytic_mori_ablation_conductivity(
    recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    ablation_mode: str,
) -> AnalyticMoriPrimitiveResult:
    """Generate analytic Mori primitives for a named audit ablation."""

    return _evaluate_analytic_mori_conductivity_with_options(
        recipe,
        uncertainty_budget,
        species_catalog,
        _analytic_primitive_options_for_ablation(ablation_mode),
    )


def evaluate_analytic_mori_forced_free_carrier_obstruction_conductivity(
    recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    free_li_translation_scale: float,
    compact_anion_translation_scale: float,
) -> AnalyticMoriPrimitiveResult:
    """Generate analytic Mori primitives with fixed free-carrier translation scales."""

    validated_free_li_translation_scale = _translation_scale(
        free_li_translation_scale,
        "free_li_translation_scale",
    )
    validated_compact_anion_translation_scale = _translation_scale(
        compact_anion_translation_scale,
        "compact_anion_translation_scale",
    )
    return _evaluate_analytic_mori_conductivity_with_options(
        recipe,
        uncertainty_budget,
        species_catalog,
        _AnalyticPrimitiveOptions(
            association_enabled=True,
            higher_aggregates_enabled=True,
            binding_resistance_enabled=True,
            atmosphere_resistance_enabled=True,
            jones_dole_viscosity_enabled=True,
            dielectric_decrement_enabled=True,
            carrier_cage_point_substates_enabled=False,
            backjump_cage_memory_enabled=False,
            free_li_local_obstruction_enabled=False,
            compact_anion_local_obstruction_enabled=False,
            forced_free_li_translation_scale_enabled=True,
            forced_free_li_translation_scale=validated_free_li_translation_scale,
            forced_compact_anion_translation_scale_enabled=True,
            forced_compact_anion_translation_scale=validated_compact_anion_translation_scale,
        ),
    )


_BASELINE_ANALYTIC_PRIMITIVE_OPTIONS = _AnalyticPrimitiveOptions(
    association_enabled=True,
    higher_aggregates_enabled=True,
    binding_resistance_enabled=True,
    atmosphere_resistance_enabled=True,
    jones_dole_viscosity_enabled=True,
    dielectric_decrement_enabled=True,
    carrier_cage_point_substates_enabled=False,
    backjump_cage_memory_enabled=False,
    free_li_local_obstruction_enabled=False,
    compact_anion_local_obstruction_enabled=False,
    forced_free_li_translation_scale_enabled=False,
    forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
    forced_compact_anion_translation_scale_enabled=False,
    forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
)

_ANALYTIC_PRIMITIVE_OPTIONS_BY_ABLATION = {
    ANALYTIC_MORI_ABLATION_BASELINE: _BASELINE_ANALYTIC_PRIMITIVE_OPTIONS,
    ANALYTIC_MORI_ABLATION_ASSOCIATION_OFF: _AnalyticPrimitiveOptions(
        association_enabled=False,
        higher_aggregates_enabled=False,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_HIGHER_AGGREGATES_OFF: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=False,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_BINDING_RESISTANCE_OFF: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=False,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_ATMOSPHERE_RESISTANCE_OFF: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=False,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_JONES_DOLE_VISCOSITY_OFF: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=False,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_DIELECTRIC_DECREMENT_OFF: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=False,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_FREE_ION_NE: _AnalyticPrimitiveOptions(
        association_enabled=False,
        higher_aggregates_enabled=False,
        binding_resistance_enabled=False,
        atmosphere_resistance_enabled=False,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_CARRIER_CAGE_POINT_SUBSTATES: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=True,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_BACKJUMP_CAGE_MEMORY: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=True,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_FREE_LI_LOCAL_OBSTRUCTION: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=True,
        compact_anion_local_obstruction_enabled=False,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_COMPACT_ANION_LOCAL_OBSTRUCTION: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=False,
        compact_anion_local_obstruction_enabled=True,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_FREE_LI_PLUS_COMPACT_ANION_LOCAL_OBSTRUCTION: _AnalyticPrimitiveOptions(
        association_enabled=True,
        higher_aggregates_enabled=True,
        binding_resistance_enabled=True,
        atmosphere_resistance_enabled=True,
        jones_dole_viscosity_enabled=True,
        dielectric_decrement_enabled=True,
        carrier_cage_point_substates_enabled=False,
        backjump_cage_memory_enabled=False,
        free_li_local_obstruction_enabled=True,
        compact_anion_local_obstruction_enabled=True,
        forced_free_li_translation_scale_enabled=False,
        forced_free_li_translation_scale=NORMALIZED_PROBABILITY_SUM,
        forced_compact_anion_translation_scale_enabled=False,
        forced_compact_anion_translation_scale=NORMALIZED_PROBABILITY_SUM,
    ),
    ANALYTIC_MORI_ABLATION_SELECTIVE_CARRIER_CAGE_ONLY: replace(
        _BASELINE_ANALYTIC_PRIMITIVE_OPTIONS,
        selective_carrier_cage_enabled=True,
    ),
    ANALYTIC_MORI_ABLATION_DESCRIPTOR_ATMOSPHERE_RELAXATION_RELEASE_ONLY: replace(
        _BASELINE_ANALYTIC_PRIMITIVE_OPTIONS,
        descriptor_atmosphere_relaxation_release_enabled=True,
    ),
    ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_RELEASE: replace(
        _BASELINE_ANALYTIC_PRIMITIVE_OPTIONS,
        selective_carrier_cage_enabled=True,
        descriptor_atmosphere_relaxation_release_enabled=True,
    ),
    ANALYTIC_MORI_ABLATION_SELECTIVE_CAGE_PLUS_DESCRIPTOR_REL_AND_EP_RELEASE: replace(
        _BASELINE_ANALYTIC_PRIMITIVE_OPTIONS,
        selective_carrier_cage_enabled=True,
        descriptor_atmosphere_relaxation_release_enabled=True,
        descriptor_atmosphere_ep_release_enabled=True,
    ),
    ANALYTIC_MORI_ABLATION_TIMESCALE_STRUCTURAL_CAGE_MEMORY: replace(
        _BASELINE_ANALYTIC_PRIMITIVE_OPTIONS,
        timescale_structural_cage_memory_enabled=True,
    ),
}


def _evaluate_analytic_mori_conductivity_with_options(
    recipe: AnalyticMoriRecipe,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
    species_catalog: AnalyticMoriSpeciesCatalog,
    primitive_options: _AnalyticPrimitiveOptions,
) -> AnalyticMoriPrimitiveResult:
    _validate_structural_uncertainty_budget(uncertainty_budget)
    temperature_K = _positive_float(recipe.temperature_K, "recipe.temperature_K")
    active_volume_m3 = _positive_float(recipe.active_volume_m3, "recipe.active_volume_m3")
    normalized_solvent_fractions = _validated_solvent_fractions(recipe.solvent_volume_fractions)
    validated_salt_molarities = _validated_positive_mapping(
        recipe.salt_molarities_M,
        "recipe.salt_molarities_M",
    )
    validated_additive_weight_fractions = _validated_nonnegative_mapping(
        recipe.additive_weight_fractions,
        "recipe.additive_weight_fractions",
    )
    normalized_analytic_recipe = AnalyticMoriRecipe(
        solvent_volume_fractions=normalized_solvent_fractions,
        salt_molarities_M=validated_salt_molarities,
        additive_weight_fractions=validated_additive_weight_fractions,
        temperature_K=temperature_K,
        active_volume_m3=active_volume_m3,
    )
    dielectric_heads = _effective_dielectric_heads(
        normalized_solvent_fractions,
        validated_additive_weight_fractions,
        validated_salt_molarities,
        species_catalog,
        primitive_options.dielectric_decrement_enabled,
    )
    effective_viscosity_cP = _effective_viscosity_cP(
        normalized_solvent_fractions,
        validated_additive_weight_fractions,
        validated_salt_molarities,
        species_catalog,
        primitive_options.jones_dole_viscosity_enabled,
    )
    resolved_anions = _resolved_anions(validated_salt_molarities, species_catalog)
    total_salt_concentration_M = math.fsum(validated_salt_molarities.values())
    association_constants = tuple(
        _association_constants_for_anion(
            resolved_anion,
            dielectric_heads.epsilon_association,
            temperature_K,
            total_salt_concentration_M,
            primitive_options.association_enabled,
            primitive_options.higher_aggregates_enabled,
        )
        for resolved_anion in resolved_anions
    )
    mass_templates = _mass_action_templates(association_constants)
    mass_solution = _solve_mass_balance(
        total_lithium_concentration_M=math.fsum(validated_salt_molarities.values()),
        resolved_anions=resolved_anions,
        templates=mass_templates,
    )
    transport_bundle = _transport_state_primitives(
        mass_solution,
        mass_templates,
        resolved_anions,
        effective_viscosity_cP,
        dielectric_heads.epsilon_atmosphere,
        temperature_K,
        species_catalog,
        primitive_options.binding_resistance_enabled,
        primitive_options.atmosphere_resistance_enabled,
        normalized_analytic_recipe,
        primitive_options,
    )
    transport_states = transport_bundle.transport_states
    backjump_cage_primitives = _backjump_cage_primitives(
        transport_bundle,
        mass_solution,
        mass_templates,
        resolved_anions,
        normalized_analytic_recipe,
        species_catalog,
    )
    direct_mori_input = _direct_resistance_mori_input(
        transport_states,
        temperature_K,
    )
    direct_mori_result = compute_projected_mori_conductivity(direct_mori_input)
    markov_additive_model_states = _markov_additive_model_states(
        transport_states,
        transport_bundle.carrier_cage_primitives,
        primitive_options.carrier_cage_point_substates_enabled,
        transport_bundle.selective_carrier_cage_primitives,
        primitive_options.selective_carrier_cage_enabled,
        backjump_cage_primitives,
        primitive_options.backjump_cage_memory_enabled,
        transport_bundle.timescale_structural_memory_primitives,
        primitive_options.timescale_structural_cage_memory_enabled,
    )
    markov_additive_events = _markov_additive_events_from_model_states(
        markov_additive_model_states,
        temperature_K,
    )
    markov_state_concentrations_mol_m3 = np.asarray(
        [state.concentration_mol_m3 for state in markov_additive_model_states],
        dtype=float,
    )
    markov_additive_result = compute_markov_additive_green_kubo_conductivity(
        MarkovAdditiveConductivityInput(
            state_labels=tuple(state.label for state in markov_additive_model_states),
            state_concentrations_mol_m3=markov_state_concentrations_mol_m3,
            events=markov_additive_events,
            temperature_K=temperature_K,
        )
    )
    markov_event_family_attributions = (
        compute_markov_additive_event_family_attribution(
            markov_additive_result,
            markov_additive_events,
            markov_state_concentrations_mol_m3,
            {
                event.label: event.family_label
                for event in markov_additive_events
            },
            temperature_K,
            tuple(),
        )
    )
    uncertainty_certificate = _uncertainty_certificate(
        markov_additive_result.sigma_mS_cm,
        uncertainty_budget,
    )
    return AnalyticMoriPrimitiveResult(
        sigma_mS_cm=markov_additive_result.sigma_mS_cm,
        sigma_S_m=markov_additive_result.sigma_S_m,
        direct_mori_result=direct_mori_result,
        markov_additive_result=markov_additive_result,
        mori_input=markov_additive_result.corrector_mori_input,
        state_labels=tuple(state.label for state in markov_additive_model_states),
        state_concentrations_mol_m3=tuple(
            state.concentration_mol_m3 for state in markov_additive_model_states
        ),
        generator_s_inv=markov_additive_result.generator_s_inv.copy(),
        markov_additive_events=markov_additive_events,
        markov_event_family_attributions=markov_event_family_attributions,
        transport_states=transport_states,
        bulk_ion_atmosphere_state=transport_bundle.bulk_ion_atmosphere_state,
        atmosphere_carrier_relaxation_form_factors=(
            transport_bundle.atmosphere_carrier_relaxation_form_factors
        ),
        atmosphere_block_diagnostics=transport_bundle.atmosphere_block_diagnostics,
        carrier_cage_primitives=transport_bundle.carrier_cage_primitives,
        selective_carrier_cage_primitives=(
            transport_bundle.selective_carrier_cage_primitives
        ),
        descriptor_atmosphere_release_primitive=(
            transport_bundle.descriptor_atmosphere_release_primitive
        ),
        timescale_structural_memory_primitives=(
            transport_bundle.timescale_structural_memory_primitives
        ),
        free_carrier_obstruction_primitives=(
            transport_bundle.free_carrier_obstruction_primitives
        ),
        backjump_cage_primitives=backjump_cage_primitives,
        association_constants=association_constants,
        mass_balance=mass_solution.mass_balance,
        dielectric_heads=dielectric_heads,
        effective_dielectric=dielectric_heads.epsilon_atmosphere,
        effective_viscosity_cP=effective_viscosity_cP,
        active_volume_m3=active_volume_m3,
        conductivity_uncertainty=uncertainty_certificate,
    )


def _analytic_primitive_options_for_ablation(
    ablation_mode: str,
) -> _AnalyticPrimitiveOptions:
    if ablation_mode not in _ANALYTIC_PRIMITIVE_OPTIONS_BY_ABLATION:
        raise ValueError(
            f"Unsupported analytic Mori ablation {ablation_mode!r}; expected one of "
            f"{SUPPORTED_ANALYTIC_MORI_ABLATIONS}"
        )
    return _ANALYTIC_PRIMITIVE_OPTIONS_BY_ABLATION[ablation_mode]


def _required_record(
    collection: Mapping[str, SpeciesRecord],
    species_name: str,
    context: str,
) -> SpeciesRecord:
    if species_name not in collection:
        raise ValueError(f"{context} species {species_name} is not in the species catalog")
    return collection[species_name]


def _required_float(record: SpeciesRecord, key: str, context: str) -> float:
    if key not in record:
        raise ValueError(f"{context} missing required descriptor {key}")
    value = record[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context}.{key} must be numeric")
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{context}.{key} must be finite")
    return parsed_value


def _optional_positive_float(
    record: SpeciesRecord,
    key: str,
    context: str,
    default_value: float,
) -> float:
    if key not in record:
        return _positive_float(default_value, f"{context}.{key}.default")
    return _positive_float(
        _required_float(record, key, context),
        f"{context}.{key}",
    )


def _required_str(record: SpeciesRecord, key: str, context: str) -> str:
    if key not in record:
        raise ValueError(f"{context} missing required descriptor {key}")
    value = record[key]
    if not isinstance(value, str):
        raise TypeError(f"{context}.{key} must be a string")
    if value == "":
        raise ValueError(f"{context}.{key} must be nonempty")
    return value


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _nonnegative_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        raise ValueError(f"{context} must be nonnegative and finite")
    return parsed_value


def _unit_interval_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if (
        not math.isfinite(parsed_value)
        or parsed_value < 0.0
        or parsed_value > NORMALIZED_PROBABILITY_SUM
    ):
        raise ValueError(f"{context} must be finite and in [0, 1]")
    return parsed_value


def _translation_scale(value: float, context: str) -> float:
    parsed_value = float(value)
    if (
        not math.isfinite(parsed_value)
        or parsed_value <= 0.0
        or parsed_value > NORMALIZED_PROBABILITY_SUM
    ):
        raise ValueError(f"{context} must be finite and in (0, 1]")
    return parsed_value


def _dense_free_volume_obstruction_parameters() -> _DenseFreeVolumeObstructionParameters:
    physics_config = load_physics_config()
    config_section = require_config(
        physics_config,
        DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION,
        context="_dense_free_volume_obstruction_parameters",
    )
    if not isinstance(config_section, dict):
        raise TypeError(
            f"{DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION} must be a config object"
        )
    steric_reference_fraction = _unit_interval_float(
        require_config(
            config_section,
            "steric_reference_fraction",
            context="_dense_free_volume_obstruction_parameters",
        ),
        f"{DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION}.steric_reference_fraction",
    )
    if steric_reference_fraction >= NORMALIZED_PROBABILITY_SUM:
        raise ValueError(
            f"{DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION}.steric_reference_fraction "
            "must be less than one"
        )
    return _DenseFreeVolumeObstructionParameters(
        free_li_obstruction_strength=_positive_float(
            require_config(
                config_section,
                "free_li_obstruction_strength",
                context="_dense_free_volume_obstruction_parameters",
            ),
            f"{DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION}.free_li_obstruction_strength",
        ),
        compact_anion_obstruction_strength=_positive_float(
            require_config(
                config_section,
                "compact_anion_obstruction_strength",
                context="_dense_free_volume_obstruction_parameters",
            ),
            f"{DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION}.compact_anion_obstruction_strength",
        ),
        steric_reference_fraction=steric_reference_fraction,
        free_volume_power=_positive_float(
            require_config(
                config_section,
                "free_volume_power",
                context="_dense_free_volume_obstruction_parameters",
            ),
            f"{DENSE_FREE_VOLUME_OBSTRUCTION_CONFIG_SECTION}.free_volume_power",
        ),
    )


def _selective_carrier_cage_parameters() -> _SelectiveCarrierCageParameters:
    physics_config = load_physics_config()
    config_section = require_config(
        physics_config,
        SELECTIVE_CARRIER_CAGE_CONFIG_SECTION,
        context="_selective_carrier_cage_parameters",
    )
    if not isinstance(config_section, dict):
        raise TypeError(f"{SELECTIVE_CARRIER_CAGE_CONFIG_SECTION} must be a config object")
    caged_fraction_max = _unit_interval_float(
        require_config(
            config_section,
            "caged_fraction_max",
            context="_selective_carrier_cage_parameters",
        ),
        f"{SELECTIVE_CARRIER_CAGE_CONFIG_SECTION}.caged_fraction_max",
    )
    caged_diffusion_scale_min = _translation_scale(
        require_config(
            config_section,
            "caged_diffusion_scale_min",
            context="_selective_carrier_cage_parameters",
        ),
        f"{SELECTIVE_CARRIER_CAGE_CONFIG_SECTION}.caged_diffusion_scale_min",
    )
    return _SelectiveCarrierCageParameters(
        caged_fraction_max=caged_fraction_max,
        caged_diffusion_scale_min=caged_diffusion_scale_min,
        dense_driver_power=_positive_float(
            require_config(
                config_section,
                "dense_driver_power",
                context="_selective_carrier_cage_parameters",
            ),
            f"{SELECTIVE_CARRIER_CAGE_CONFIG_SECTION}.dense_driver_power",
        ),
        descriptor_release_suppression_strength=_unit_interval_float(
            require_config(
                config_section,
                "descriptor_release_suppression_strength",
                context="_selective_carrier_cage_parameters",
            ),
            (
                f"{SELECTIVE_CARRIER_CAGE_CONFIG_SECTION}."
                "descriptor_release_suppression_strength"
            ),
        ),
    )


def _descriptor_atmosphere_release_parameters() -> _DescriptorAtmosphereReleaseParameters:
    physics_config = load_physics_config()
    config_section = require_config(
        physics_config,
        DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION,
        context="_descriptor_atmosphere_release_parameters",
    )
    if not isinstance(config_section, dict):
        raise TypeError(
            f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION} must be a config object"
        )
    mixture_descriptor_on_value = _unit_interval_float(
        require_config(
            config_section,
            "mixture_descriptor_on_value",
            context="_descriptor_atmosphere_release_parameters",
        ),
        (
            f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}."
            "mixture_descriptor_on_value"
        ),
    )
    mixture_descriptor_full_value = _unit_interval_float(
        require_config(
            config_section,
            "mixture_descriptor_full_value",
            context="_descriptor_atmosphere_release_parameters",
        ),
        (
            f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}."
            "mixture_descriptor_full_value"
        ),
    )
    if mixture_descriptor_full_value <= mixture_descriptor_on_value:
        raise ValueError(
            f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}."
            "mixture_descriptor_full_value must exceed mixture_descriptor_on_value"
        )
    viscosity_on_cP = _positive_float(
        require_config(
            config_section,
            "viscosity_on_cP",
            context="_descriptor_atmosphere_release_parameters",
        ),
        f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.viscosity_on_cP",
    )
    viscosity_full_cP = _positive_float(
        require_config(
            config_section,
            "viscosity_full_cP",
            context="_descriptor_atmosphere_release_parameters",
        ),
        f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.viscosity_full_cP",
    )
    if viscosity_full_cP <= viscosity_on_cP:
        raise ValueError(
            f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.viscosity_full_cP "
            "must exceed viscosity_on_cP"
        )
    dielectric_on_value = _positive_float(
        require_config(
            config_section,
            "dielectric_on_value",
            context="_descriptor_atmosphere_release_parameters",
        ),
        f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.dielectric_on_value",
    )
    dielectric_full_value = _positive_float(
        require_config(
            config_section,
            "dielectric_full_value",
            context="_descriptor_atmosphere_release_parameters",
        ),
        f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.dielectric_full_value",
    )
    if dielectric_full_value >= dielectric_on_value:
        raise ValueError(
            f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.dielectric_full_value "
            "must be below dielectric_on_value"
        )
    donor_on_value = _positive_float(
        require_config(
            config_section,
            "donor_on_value",
            context="_descriptor_atmosphere_release_parameters",
        ),
        f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.donor_on_value",
    )
    donor_full_value = _positive_float(
        require_config(
            config_section,
            "donor_full_value",
            context="_descriptor_atmosphere_release_parameters",
        ),
        f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.donor_full_value",
    )
    if donor_full_value >= donor_on_value:
        raise ValueError(
            f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}.donor_full_value "
            "must be below donor_on_value"
        )
    return _DescriptorAtmosphereReleaseParameters(
        relaxation_release_strength=_unit_interval_float(
            require_config(
                config_section,
                "relaxation_release_strength",
                context="_descriptor_atmosphere_release_parameters",
            ),
            (
                f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}."
                "relaxation_release_strength"
            ),
        ),
        electrophoretic_release_strength=_unit_interval_float(
            require_config(
                config_section,
                "electrophoretic_release_strength",
                context="_descriptor_atmosphere_release_parameters",
            ),
            (
                f"{DESCRIPTOR_ATMOSPHERE_RELEASE_CONFIG_SECTION}."
                "electrophoretic_release_strength"
            ),
        ),
        mixture_descriptor_on_value=mixture_descriptor_on_value,
        mixture_descriptor_full_value=mixture_descriptor_full_value,
        viscosity_on_cP=viscosity_on_cP,
        viscosity_full_cP=viscosity_full_cP,
        dielectric_on_value=dielectric_on_value,
        dielectric_full_value=dielectric_full_value,
        donor_on_value=donor_on_value,
        donor_full_value=donor_full_value,
    )


def _timescale_structural_memory_parameters() -> _TimescaleStructuralMemoryParameters:
    physics_config = load_physics_config()
    config_section = require_config(
        physics_config,
        TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION,
        context="_timescale_structural_memory_parameters",
    )
    if not isinstance(config_section, dict):
        raise TypeError(
            f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION} must be a config object"
        )
    packing_fraction_limit = _unit_interval_float(
        require_config(
            config_section,
            "packing_fraction_limit",
            context="_timescale_structural_memory_parameters",
        ),
        f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.packing_fraction_limit",
    )
    if packing_fraction_limit <= 0.0:
        raise ValueError(
            f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.packing_fraction_limit "
            "must be positive"
        )
    deborah_on_value = _positive_float(
        require_config(
            config_section,
            "deborah_on_value",
            context="_timescale_structural_memory_parameters",
        ),
        f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.deborah_on_value",
    )
    deborah_full_value = _positive_float(
        require_config(
            config_section,
            "deborah_full_value",
            context="_timescale_structural_memory_parameters",
        ),
        f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.deborah_full_value",
    )
    if deborah_full_value <= deborah_on_value:
        raise ValueError(
            f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.deborah_full_value "
            "must exceed deborah_on_value"
        )
    size_void_ratio_on_value = _positive_float(
        require_config(
            config_section,
            "size_void_ratio_on_value",
            context="_timescale_structural_memory_parameters",
        ),
        f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.size_void_ratio_on_value",
    )
    size_void_ratio_full_value = _positive_float(
        require_config(
            config_section,
            "size_void_ratio_full_value",
            context="_timescale_structural_memory_parameters",
        ),
        f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.size_void_ratio_full_value",
    )
    if size_void_ratio_full_value <= size_void_ratio_on_value:
        raise ValueError(
            f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.size_void_ratio_full_value "
            "must exceed size_void_ratio_on_value"
        )
    steric_on_fraction = _unit_interval_float(
        require_config(
            config_section,
            "steric_on_fraction",
            context="_timescale_structural_memory_parameters",
        ),
        f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.steric_on_fraction",
    )
    steric_full_fraction = _unit_interval_float(
        require_config(
            config_section,
            "steric_full_fraction",
            context="_timescale_structural_memory_parameters",
        ),
        f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.steric_full_fraction",
    )
    if steric_full_fraction <= steric_on_fraction:
        raise ValueError(
            f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.steric_full_fraction "
            "must exceed steric_on_fraction"
        )
    return _TimescaleStructuralMemoryParameters(
        cage_fraction_max=_unit_interval_float(
            require_config(
                config_section,
                "cage_fraction_max",
                context="_timescale_structural_memory_parameters",
            ),
            f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.cage_fraction_max",
        ),
        jump_length_radius_scale=_positive_float(
            require_config(
                config_section,
                "jump_length_radius_scale",
                context="_timescale_structural_memory_parameters",
            ),
            f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}.jump_length_radius_scale",
        ),
        structural_free_volume_barrier_strength=_nonnegative_float(
            require_config(
                config_section,
                "structural_free_volume_barrier_strength",
                context="_timescale_structural_memory_parameters",
            ),
            (
                f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}."
                "structural_free_volume_barrier_strength"
            ),
        ),
        packing_fraction_limit=packing_fraction_limit,
        deborah_on_value=deborah_on_value,
        deborah_full_value=deborah_full_value,
        size_void_ratio_on_value=size_void_ratio_on_value,
        size_void_ratio_full_value=size_void_ratio_full_value,
        steric_on_fraction=steric_on_fraction,
        steric_full_fraction=steric_full_fraction,
        atmosphere_capture_fraction_max=_unit_interval_float(
            require_config(
                config_section,
                "atmosphere_capture_fraction_max",
                context="_timescale_structural_memory_parameters",
            ),
            (
                f"{TIMESCALE_STRUCTURAL_MEMORY_CONFIG_SECTION}."
                "atmosphere_capture_fraction_max"
            ),
        ),
    )


def _dense_free_volume_obstruction_driver(
    steric_volume_fraction: float,
    compact_anion_driver: float,
    carbonate_driver: float,
    high_salt_driver: float,
    low_donor_driver: float,
    dense_parameters: _DenseFreeVolumeObstructionParameters,
    context: str,
) -> float:
    parsed_steric_volume_fraction = _unit_interval_float(
        steric_volume_fraction,
        f"{context}.steric_volume_fraction",
    )
    if parsed_steric_volume_fraction <= dense_parameters.steric_reference_fraction:
        return 0.0
    free_volume_denominator = NORMALIZED_PROBABILITY_SUM - parsed_steric_volume_fraction
    if free_volume_denominator <= 0.0:
        raise ValueError(f"{context}.free_volume_denominator must be positive")
    dense_free_volume_coordinate = (
        parsed_steric_volume_fraction - dense_parameters.steric_reference_fraction
    ) / free_volume_denominator
    return float(
        (dense_free_volume_coordinate ** dense_parameters.free_volume_power)
        * _unit_interval_float(compact_anion_driver, f"{context}.compact_anion_driver")
        * _unit_interval_float(carbonate_driver, f"{context}.carbonate_driver")
        * _unit_interval_float(high_salt_driver, f"{context}.high_salt_driver")
        * _unit_interval_float(low_donor_driver, f"{context}.low_donor_driver")
    )


def _validated_positive_mapping(
    values: Mapping[str, float],
    context: str,
) -> dict[str, float]:
    if len(values) == 0:
        raise ValueError(f"{context} must contain at least one entry")
    parsed_values: dict[str, float] = {}
    for name, value in values.items():
        parsed_values[name] = _positive_float(value, f"{context}.{name}")
    return parsed_values


def _validated_nonnegative_mapping(
    values: Mapping[str, float],
    context: str,
) -> dict[str, float]:
    parsed_values: dict[str, float] = {}
    for name, value in values.items():
        parsed_values[name] = _nonnegative_float(value, f"{context}.{name}")
    return parsed_values


def _validated_solvent_fractions(values: Mapping[str, float]) -> dict[str, float]:
    parsed_values = _validated_positive_mapping(values, "recipe.solvent_volume_fractions")
    fraction_sum = math.fsum(parsed_values.values())
    if abs(fraction_sum - NORMALIZED_PROBABILITY_SUM) > SOLVENT_FRACTION_SUM_TOLERANCE:
        raise ValueError("recipe.solvent_volume_fractions must sum to one")
    return parsed_values


def _validate_structural_uncertainty_budget(
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
) -> None:
    interval_widths = (
        _validated_interval_width(
            uncertainty_budget.association_logK_interval,
            "association_logK_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.dielectric_decrement_scale_interval,
            "dielectric_decrement_scale_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.jones_dole_scale_interval,
            "jones_dole_scale_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.atmosphere_ep_scale_interval,
            "atmosphere_ep_scale_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.atmosphere_rel_scale_interval,
            "atmosphere_rel_scale_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.lithium_charge_cloud_radius_interval_A,
            "lithium_charge_cloud_radius_interval_A",
        ),
        _validated_interval_width(
            uncertainty_budget.anion_charge_cloud_radius_interval_A,
            "anion_charge_cloud_radius_interval_A",
        ),
        _validated_interval_width(
            uncertainty_budget.cage_trapping_fraction_interval,
            "cage_trapping_fraction_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.jump_length_scale_interval,
            "jump_length_scale_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.conversion_rate_scale_interval,
            "conversion_rate_scale_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.forced_free_li_translation_scale_interval,
            "forced_free_li_translation_scale_interval",
        ),
        _validated_interval_width(
            uncertainty_budget.forced_compact_anion_translation_scale_interval,
            "forced_compact_anion_translation_scale_interval",
        ),
    )
    _validate_translation_scale_interval(
        uncertainty_budget.forced_free_li_translation_scale_interval,
        "forced_free_li_translation_scale_interval",
    )
    _validate_translation_scale_interval(
        uncertainty_budget.forced_compact_anion_translation_scale_interval,
        "forced_compact_anion_translation_scale_interval",
    )
    _positive_float(
        uncertainty_budget.certificate_threshold_mS_cm,
        "certificate_threshold_mS_cm",
    )
    if math.fsum(interval_widths) <= 0.0:
        raise ValueError("structural primitive uncertainty budget must contain a positive interval width")


def _validated_interval_width(interval: tuple[float, float], context: str) -> float:
    if len(interval) != 2:
        raise ValueError(f"{context} must contain exactly two bounds")
    lower_bound = float(interval[0])
    upper_bound = float(interval[1])
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
        raise ValueError(f"{context} bounds must be finite")
    if upper_bound < lower_bound:
        raise ValueError(f"{context} upper bound must be greater than or equal to lower bound")
    return float(upper_bound - lower_bound)


def _validate_translation_scale_interval(
    interval: tuple[float, float],
    context: str,
) -> None:
    _validated_interval_width(interval, context)
    lower_bound = float(interval[0])
    upper_bound = float(interval[1])
    if lower_bound <= 0.0 or upper_bound > NORMALIZED_PROBABILITY_SUM:
        raise ValueError(f"{context} bounds must be in (0, 1]")


def _effective_dielectric_heads(
    solvent_volume_fractions: Mapping[str, float],
    additive_weight_fractions: Mapping[str, float],
    salt_molarities_M: Mapping[str, float],
    species_catalog: AnalyticMoriSpeciesCatalog,
    dielectric_decrement_enabled: bool,
) -> AnalyticDielectricHeads:
    weighted_dielectric_sum = 0.0
    total_volume_weight = 0.0
    for solvent_name, volume_fraction in solvent_volume_fractions.items():
        solvent_record = _required_record(species_catalog.solvents, solvent_name, "solvent")
        epsilon_r = _required_float(solvent_record, "epsilon_r", f"solvent.{solvent_name}")
        weighted_dielectric_sum += volume_fraction * epsilon_r
        total_volume_weight += volume_fraction
    for additive_name, weight_fraction in additive_weight_fractions.items():
        if weight_fraction == 0.0:
            continue
        additive_record = _required_record(species_catalog.additives, additive_name, "additive")
        epsilon_r = _required_float(additive_record, "epsilon_r", f"additive.{additive_name}")
        density_g_ml = _required_float(additive_record, "density_g_ml", f"additive.{additive_name}")
        volume_weight = weight_fraction / density_g_ml
        weighted_dielectric_sum += volume_weight * epsilon_r
        total_volume_weight += volume_weight
    epsilon_mixture = weighted_dielectric_sum / total_volume_weight
    if epsilon_mixture <= 0.0:
        raise ValueError("mixture dielectric must be positive")
    epsilon_association = float(epsilon_mixture)
    if not dielectric_decrement_enabled:
        return AnalyticDielectricHeads(
            epsilon_mixture=float(epsilon_mixture),
            epsilon_association=epsilon_association,
            epsilon_atmosphere=float(epsilon_mixture),
        )
    dielectric_decrement_product = NORMALIZED_PROBABILITY_SUM
    for salt_name, molarity_M in salt_molarities_M.items():
        salt_record = _required_record(species_catalog.salts, salt_name, "salt")
        dielectric_decrement = _required_float(
            salt_record,
            "dielectric_decrement_frac_per_M",
            f"salt.{salt_name}",
        )
        decrement_factor = NORMALIZED_PROBABILITY_SUM - dielectric_decrement * molarity_M
        dielectric_decrement_product *= decrement_factor
    epsilon_atmosphere = max(
        ATMOSPHERE_DIELECTRIC_FLOOR,
        epsilon_mixture * dielectric_decrement_product,
    )
    return AnalyticDielectricHeads(
        epsilon_mixture=float(epsilon_mixture),
        epsilon_association=epsilon_association,
        epsilon_atmosphere=float(epsilon_atmosphere),
    )


def _effective_viscosity_cP(
    solvent_volume_fractions: Mapping[str, float],
    additive_weight_fractions: Mapping[str, float],
    salt_molarities_M: Mapping[str, float],
    species_catalog: AnalyticMoriSpeciesCatalog,
    jones_dole_viscosity_enabled: bool,
) -> float:
    log_viscosity_sum = 0.0
    total_volume_weight = 0.0
    for solvent_name, volume_fraction in solvent_volume_fractions.items():
        solvent_record = _required_record(species_catalog.solvents, solvent_name, "solvent")
        viscosity_cP = _required_float(solvent_record, "viscosity_cP", f"solvent.{solvent_name}")
        log_viscosity_sum += volume_fraction * math.log(
            _positive_float(viscosity_cP, f"solvent.{solvent_name}.viscosity_cP")
        )
        total_volume_weight += volume_fraction
    for additive_name, weight_fraction in additive_weight_fractions.items():
        if weight_fraction == 0.0:
            continue
        additive_record = _required_record(species_catalog.additives, additive_name, "additive")
        viscosity_cP = _required_float(additive_record, "viscosity_cP", f"additive.{additive_name}")
        density_g_ml = _required_float(additive_record, "density_g_ml", f"additive.{additive_name}")
        volume_weight = weight_fraction / density_g_ml
        log_viscosity_sum += volume_weight * math.log(
            _positive_float(viscosity_cP, f"additive.{additive_name}.viscosity_cP")
        )
        total_volume_weight += volume_weight
    base_viscosity_cP = math.exp(log_viscosity_sum / total_volume_weight)
    if not jones_dole_viscosity_enabled:
        return float(base_viscosity_cP)
    jones_dole_factor = NORMALIZED_PROBABILITY_SUM
    for salt_name, molarity_M in salt_molarities_M.items():
        salt_record = _required_record(species_catalog.salts, salt_name, "salt")
        jones_dole_A = _required_float(salt_record, "jones_dole_A", f"salt.{salt_name}")
        jones_dole_B = _required_float(salt_record, "jones_dole_B", f"salt.{salt_name}")
        jones_dole_factor += jones_dole_A * math.sqrt(molarity_M) + jones_dole_B * molarity_M
    if jones_dole_factor <= 0.0:
        raise ValueError("Jones-Dole viscosity factor must be positive")
    return float(base_viscosity_cP * jones_dole_factor)


def _resolved_anions(
    salt_molarities_M: Mapping[str, float],
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> tuple[_ResolvedAnion, ...]:
    resolved_anions: list[_ResolvedAnion] = []
    for anion_index, (salt_name, molarity_M) in enumerate(sorted(salt_molarities_M.items())):
        salt_record = _required_record(species_catalog.salts, salt_name, "salt")
        _required_str(salt_record, "cation", f"salt.{salt_name}")
        _required_str(salt_record, "anion", f"salt.{salt_name}")
        _required_float(salt_record, "anion_charge", f"salt.{salt_name}")
        resolved_anions.append(
            _ResolvedAnion(
                feature_id=f"anion_site_{anion_index}",
                source_salt_name=salt_name,
                salt_record=salt_record,
                total_concentration_M=molarity_M,
            )
        )
    return tuple(resolved_anions)


def _association_constants_for_anion(
    resolved_anion: _ResolvedAnion,
    association_dielectric: float,
    temperature_K: float,
    total_salt_concentration_M: float,
    association_enabled: bool,
    higher_aggregates_enabled: bool,
) -> AnalyticAssociationConstants:
    context = f"salt.{resolved_anion.source_salt_name}"
    salt_record = resolved_anion.salt_record
    if not association_enabled:
        return AnalyticAssociationConstants(
            anion_feature_id=resolved_anion.feature_id,
            source_salt_name=resolved_anion.source_salt_name,
            K_ssip_M_inv=0.0,
            K_cip_M_inv=0.0,
            K_li2a_M2_inv=0.0,
            K_lia2_M2_inv=0.0,
            K_li2a2_M3_inv=0.0,
        )
    cation_radius_A = _required_float(salt_record, "cation_radius", context)
    anion_radius_A = _required_float(salt_record, "anion_radius", context)
    anion_charge = _required_float(salt_record, "anion_charge", context)
    bjerrum_K_A_ref = _positive_float(
        _required_float(salt_record, "bjerrum_K_A_ref", context),
        f"{context}.bjerrum_K_A_ref",
    )
    preferred_coordination_number = _positive_float(
        _required_float(salt_record, "preferred_coordination_number", context),
        f"{context}.preferred_coordination_number",
    )
    ligand_field_asymmetry = _nonnegative_float(
        _required_float(salt_record, "ligand_field_asymmetry", context),
        f"{context}.ligand_field_asymmetry",
    )
    contact_distance_m = (cation_radius_A + anion_radius_A) * ANGSTROM_TO_M
    current_pair_energy_J_mol = (
        abs(anion_charge)
        * E_CHARGE
        * E_CHARGE
        * N_A
        / (COULOMB_DENOMINATOR_FACTOR * math.pi * EPS_0 * association_dielectric * contact_distance_m)
    )
    reference_pair_energy_J_mol = (
        abs(anion_charge)
        * E_CHARGE
        * E_CHARGE
        * N_A
        / (
            COULOMB_DENOMINATOR_FACTOR
            * math.pi
            * EPS_0
            * BJERRUM_REFERENCE_DIELECTRIC
            * contact_distance_m
        )
    )
    dimensionless_pair_energy_shift = (
        current_pair_energy_J_mol - reference_pair_energy_J_mol
    ) / (R * temperature_K)
    bounded_pair_energy_shift = max(
        -ASSOCIATION_EXPONENT_CLIP,
        min(ASSOCIATION_EXPONENT_CLIP, dimensionless_pair_energy_shift),
    )
    pair_association_raw_M_inv = bjerrum_K_A_ref * math.exp(
        ASSOCIATION_DIELECTRIC_SENSITIVITY * bounded_pair_energy_shift
    )
    pair_association_cap_M_inv = _optional_positive_float(
        salt_record,
        "association_cap_M_inv",
        context,
        ASSOCIATION_PAIR_CAP_MULTIPLIER * bjerrum_K_A_ref,
    )
    pair_association_M_inv = min(
        pair_association_raw_M_inv,
        pair_association_cap_M_inv,
    )
    cip_fraction = (
        (NORMALIZED_PROBABILITY_SUM + ligand_field_asymmetry)
        / (
            preferred_coordination_number
            + 2.0
            + ligand_field_asymmetry
        )
    )
    if cip_fraction <= 0.0 or cip_fraction >= 1.0:
        raise ValueError(f"{context} produced invalid CIP fraction {cip_fraction}")
    K_cip_M_inv = pair_association_M_inv * cip_fraction
    K_ssip_M_inv = pair_association_M_inv * (NORMALIZED_PROBABILITY_SUM - cip_fraction)
    if higher_aggregates_enabled:
        steric_factor = _aggregate_steric_factor(total_salt_concentration_M)
        li2a_step_M_inv = _optional_positive_float(
            salt_record,
            "li2a_step_ref_M_inv",
            context,
            K_LI2A_STEP_DEFAULT_M_INV,
        )
        lia2_step_M_inv = _optional_positive_float(
            salt_record,
            "lia2_step_ref_M_inv",
            context,
            K_LIA2_STEP_DEFAULT_M_INV,
        )
        li2a2_step_M_inv = _optional_positive_float(
            salt_record,
            "li2a2_step_ref_M_inv",
            context,
            K_LI2A2_STEP_DEFAULT_M_INV,
        )
        K_li2a_M2_inv = K_cip_M_inv * li2a_step_M_inv * steric_factor
        K_lia2_M2_inv = K_ssip_M_inv * lia2_step_M_inv * steric_factor
        K_li2a2_M3_inv = (
            K_cip_M_inv
            * K_ssip_M_inv
            * li2a2_step_M_inv
            * steric_factor
            * steric_factor
        )
    else:
        K_li2a_M2_inv = 0.0
        K_lia2_M2_inv = 0.0
        K_li2a2_M3_inv = 0.0
    return AnalyticAssociationConstants(
        anion_feature_id=resolved_anion.feature_id,
        source_salt_name=resolved_anion.source_salt_name,
        K_ssip_M_inv=float(K_ssip_M_inv),
        K_cip_M_inv=float(K_cip_M_inv),
        K_li2a_M2_inv=float(K_li2a_M2_inv),
        K_lia2_M2_inv=float(K_lia2_M2_inv),
        K_li2a2_M3_inv=float(K_li2a2_M3_inv),
    )


def _aggregate_steric_factor(total_salt_concentration_M: float) -> float:
    parsed_concentration_M = _nonnegative_float(
        total_salt_concentration_M,
        "total_salt_concentration_M",
    )
    packing_ratio = parsed_concentration_M / AGGREGATE_PACKING_CONCENTRATION_M
    return float(
        NORMALIZED_PROBABILITY_SUM
        / (NORMALIZED_PROBABILITY_SUM + packing_ratio ** AGGREGATE_STERIC_POWER)
    )


def _mass_action_templates(
    association_constants: tuple[AnalyticAssociationConstants, ...],
) -> tuple[_MassActionTemplate, ...]:
    templates: list[_MassActionTemplate] = []
    for anion_index, constants in enumerate(association_constants):
        templates.append(
            _MassActionTemplate(
                label=f"{constants.anion_feature_id}_ssip",
                state_kind="SSIP",
                anion_index=anion_index,
                li_count=1,
                anion_count=1,
                equilibrium_constant=constants.K_ssip_M_inv,
            )
        )
        templates.append(
            _MassActionTemplate(
                label=f"{constants.anion_feature_id}_cip",
                state_kind="CIP",
                anion_index=anion_index,
                li_count=1,
                anion_count=1,
                equilibrium_constant=constants.K_cip_M_inv,
            )
        )
        if constants.K_li2a_M2_inv > 0.0:
            templates.append(
                _MassActionTemplate(
                    label=f"{constants.anion_feature_id}_li2a_plus",
                    state_kind="LI2A_PLUS",
                    anion_index=anion_index,
                    li_count=2,
                    anion_count=1,
                    equilibrium_constant=constants.K_li2a_M2_inv,
                )
            )
        if constants.K_lia2_M2_inv > 0.0:
            templates.append(
                _MassActionTemplate(
                    label=f"{constants.anion_feature_id}_lia2_minus",
                    state_kind="LIA2_MINUS",
                    anion_index=anion_index,
                    li_count=1,
                    anion_count=2,
                    equilibrium_constant=constants.K_lia2_M2_inv,
                )
            )
        if constants.K_li2a2_M3_inv > 0.0:
            templates.append(
                _MassActionTemplate(
                    label=f"{constants.anion_feature_id}_li2a2_neutral",
                    state_kind="LI2A2_NEUTRAL",
                    anion_index=anion_index,
                    li_count=2,
                    anion_count=2,
                    equilibrium_constant=constants.K_li2a2_M3_inv,
                )
            )
    return tuple(templates)


def _solve_mass_balance(
    total_lithium_concentration_M: float,
    resolved_anions: tuple[_ResolvedAnion, ...],
    templates: tuple[_MassActionTemplate, ...],
) -> _MassBalanceSolution:
    species_count = len(resolved_anions) + 1
    initial_free_values = [HALF_FACTOR * total_lithium_concentration_M]
    initial_free_values.extend(
        HALF_FACTOR * resolved_anion.total_concentration_M
        for resolved_anion in resolved_anions
    )
    log_activities = np.log(np.asarray(initial_free_values, dtype=float))
    target_totals = np.asarray(
        [total_lithium_concentration_M]
        + [resolved_anion.total_concentration_M for resolved_anion in resolved_anions],
        dtype=float,
    )
    for iteration_index in range(MASS_BALANCE_MAX_ITERATIONS):
        residual_vector, state_concentrations = _mass_balance_residual(
            log_activities,
            target_totals,
            templates,
        )
        if float(np.max(np.abs(residual_vector))) <= MASS_BALANCE_ABSOLUTE_TOLERANCE_M:
            return _mass_balance_solution(
                log_activities,
                target_totals,
                residual_vector,
                state_concentrations,
                resolved_anions,
            )
        jacobian_matrix = np.zeros((species_count, species_count), dtype=float)
        for column_index in range(species_count):
            shifted_log_activities = log_activities.copy()
            shifted_log_activities[column_index] += MASS_BALANCE_FINITE_DIFFERENCE_STEP
            shifted_residual, _ = _mass_balance_residual(
                shifted_log_activities,
                target_totals,
                templates,
            )
            jacobian_matrix[:, column_index] = (
                shifted_residual - residual_vector
            ) / MASS_BALANCE_FINITE_DIFFERENCE_STEP
        step = np.linalg.solve(jacobian_matrix, -residual_vector)
        current_norm = float(np.linalg.norm(residual_vector))
        accepted = False
        for damping_index in range(MASS_BALANCE_DAMPING_ATTEMPTS):
            damping_factor = HALF_FACTOR ** damping_index
            candidate_log_activities = log_activities + damping_factor * step
            candidate_residual, _ = _mass_balance_residual(
                candidate_log_activities,
                target_totals,
                templates,
            )
            candidate_norm = float(np.linalg.norm(candidate_residual))
            if candidate_norm < current_norm:
                log_activities = candidate_log_activities
                accepted = True
                break
        if not accepted:
            raise ValueError(f"mass-action solver could not reduce residual at iteration {iteration_index}")
    raise ValueError("mass-action solver did not converge")


def _mass_balance_residual(
    log_activities: np.ndarray,
    target_totals: np.ndarray,
    templates: tuple[_MassActionTemplate, ...],
) -> tuple[np.ndarray, dict[str, float]]:
    free_activities = np.exp(log_activities)
    consumed_totals = free_activities.copy()
    state_concentrations: dict[str, float] = {}
    free_lithium_activity = float(free_activities[0])
    for anion_index in range(target_totals.shape[0] - 1):
        state_concentrations[f"anion_site_{anion_index}_free"] = float(
            free_activities[anion_index + 1]
        )
    state_concentrations["free_li"] = free_lithium_activity
    for template in templates:
        free_anion_activity = float(free_activities[template.anion_index + 1])
        concentration_M = (
            template.equilibrium_constant
            * (free_lithium_activity ** template.li_count)
            * (free_anion_activity ** template.anion_count)
        )
        state_concentrations[template.label] = float(concentration_M)
        consumed_totals[0] += template.li_count * concentration_M
        consumed_totals[template.anion_index + 1] += template.anion_count * concentration_M
    return consumed_totals - target_totals, state_concentrations


def _mass_balance_solution(
    log_activities: np.ndarray,
    target_totals: np.ndarray,
    residual_vector: np.ndarray,
    state_concentrations: Mapping[str, float],
    resolved_anions: tuple[_ResolvedAnion, ...],
) -> _MassBalanceSolution:
    free_activities = np.exp(log_activities)
    species_labels = ("Li",) + tuple(
        resolved_anion.feature_id for resolved_anion in resolved_anions
    )
    return _MassBalanceSolution(
        free_lithium_activity_M=float(free_activities[0]),
        free_anion_activities_M=tuple(float(value) for value in free_activities[1:]),
        state_concentrations_M_by_label=dict(state_concentrations),
        mass_balance=AnalyticMoriMassBalance(
            species_labels=species_labels,
            total_concentrations_M=tuple(float(value) for value in target_totals),
            free_activities_M=tuple(float(value) for value in free_activities),
            residuals_M=tuple(float(value) for value in residual_vector),
            max_abs_residual_M=float(np.max(np.abs(residual_vector))),
        ),
    )


def _transport_state_primitives(
    mass_solution: _MassBalanceSolution,
    templates: tuple[_MassActionTemplate, ...],
    resolved_anions: tuple[_ResolvedAnion, ...],
    effective_viscosity_cP: float,
    effective_dielectric: float,
    temperature_K: float,
    species_catalog: AnalyticMoriSpeciesCatalog,
    binding_resistance_enabled: bool,
    atmosphere_resistance_enabled: bool,
    analytic_recipe: AnalyticMoriRecipe,
    primitive_options: _AnalyticPrimitiveOptions,
) -> _TransportPrimitiveBundle:
    li_radius_A = _li_solvated_radius_A(species_catalog)
    li_diffusion_m2_s = _li_diffusion_m2_s(
        effective_viscosity_cP,
        temperature_K,
        species_catalog,
    )
    anion_diffusion_m2_s_by_feature: dict[str, float] = {}
    anion_radius_A_by_feature: dict[str, float] = {}
    for resolved_anion in resolved_anions:
        anion_diffusion_m2_s_by_feature[resolved_anion.feature_id] = _anion_diffusion_m2_s(
            resolved_anion,
            effective_viscosity_cP,
            temperature_K,
        )
        anion_radius_A_by_feature[resolved_anion.feature_id] = _required_float(
            resolved_anion.salt_record,
            "anion_radius",
            f"salt.{resolved_anion.source_salt_name}",
        )
    (
        base_bulk_ion_atmosphere_state,
        atmosphere_carrier_relaxation_form_factors,
    ) = _bulk_ion_atmosphere_state(
        mass_solution,
        templates,
        resolved_anions,
        li_diffusion_m2_s,
        li_radius_A,
        anion_diffusion_m2_s_by_feature,
        anion_radius_A_by_feature,
        effective_viscosity_cP,
        effective_dielectric,
        temperature_K,
        atmosphere_resistance_enabled,
    )
    free_carrier_obstruction_primitives = _free_carrier_obstruction_primitives(
        mass_solution,
        templates,
        resolved_anions,
        base_bulk_ion_atmosphere_state,
        analytic_recipe,
        species_catalog,
        primitive_options,
    )
    descriptor_atmosphere_release_primitive = _descriptor_atmosphere_release_primitive(
        free_carrier_obstruction_primitives,
        analytic_recipe,
        species_catalog,
        effective_viscosity_cP,
        effective_dielectric,
        primitive_options,
    )
    bulk_ion_atmosphere_state = _apply_descriptor_atmosphere_release(
        base_bulk_ion_atmosphere_state,
        descriptor_atmosphere_release_primitive,
    )
    free_li_diffusion_m2_s = (
        li_diffusion_m2_s
        * _free_carrier_obstruction_scale(
            free_carrier_obstruction_primitives,
            "free_li",
        )
    )
    states: list[AnalyticTransportStatePrimitive] = [
        _free_lithium_state(
            mass_solution,
            free_li_diffusion_m2_s,
            li_radius_A,
            bulk_ion_atmosphere_state,
            atmosphere_resistance_enabled,
            temperature_K=temperature_K,
        )
    ]
    template_by_label = {template.label: template for template in templates}
    for resolved_anion in resolved_anions:
        anion_diffusion_m2_s = anion_diffusion_m2_s_by_feature[resolved_anion.feature_id]
        free_anion_diffusion_m2_s = (
            anion_diffusion_m2_s
            * _free_carrier_obstruction_scale(
                free_carrier_obstruction_primitives,
                f"{resolved_anion.feature_id}_free",
            )
        )
        states.append(
            _free_anion_state(
                mass_solution,
                resolved_anion,
                free_anion_diffusion_m2_s,
                bulk_ion_atmosphere_state,
                atmosphere_resistance_enabled,
                temperature_K=temperature_K,
            )
        )
        for label, concentration_M in mass_solution.state_concentrations_M_by_label.items():
            if not label.startswith(f"{resolved_anion.feature_id}_"):
                continue
            if label.endswith("_free"):
                continue
            template = template_by_label[label]
            states.append(
                _paired_center_state(
                    template,
                    concentration_M=concentration_M,
                    li_diffusion_m2_s=li_diffusion_m2_s,
                    anion_diffusion_m2_s=anion_diffusion_m2_s,
                    resolved_anion=resolved_anion,
                    bulk_ion_atmosphere_state=bulk_ion_atmosphere_state,
                    binding_resistance_enabled=binding_resistance_enabled,
                    atmosphere_resistance_enabled=atmosphere_resistance_enabled,
                    temperature_K=temperature_K,
                )
            )
    active_transport_states = tuple(
        state for state in states if state.concentration_mol_m3 > 0.0
    )
    carrier_cage_primitives = _carrier_cage_primitives(
        active_transport_states,
        bulk_ion_atmosphere_state,
        mass_solution,
        templates,
        resolved_anions,
        li_radius_A,
        temperature_K,
    )
    selective_carrier_cage_primitives = _selective_carrier_cage_primitives(
        active_transport_states,
        carrier_cage_primitives,
        free_carrier_obstruction_primitives,
        descriptor_atmosphere_release_primitive,
    )
    if primitive_options.timescale_structural_cage_memory_enabled:
        timescale_structural_memory_primitives = _timescale_structural_memory_primitives(
            active_transport_states,
            bulk_ion_atmosphere_state,
            analytic_recipe,
            species_catalog,
            effective_viscosity_cP,
            temperature_K,
        )
        _validate_timescale_structural_memory_coverage(
            active_transport_states,
            timescale_structural_memory_primitives,
        )
    else:
        timescale_structural_memory_primitives = ()
    return _TransportPrimitiveBundle(
        transport_states=active_transport_states,
        bulk_ion_atmosphere_state=bulk_ion_atmosphere_state,
        atmosphere_carrier_relaxation_form_factors=atmosphere_carrier_relaxation_form_factors,
        atmosphere_block_diagnostics=_atmosphere_block_diagnostics(
            bulk_ion_atmosphere_state,
            atmosphere_carrier_relaxation_form_factors,
        ),
        carrier_cage_primitives=carrier_cage_primitives,
        selective_carrier_cage_primitives=selective_carrier_cage_primitives,
        descriptor_atmosphere_release_primitive=(
            descriptor_atmosphere_release_primitive
        ),
        timescale_structural_memory_primitives=timescale_structural_memory_primitives,
        free_carrier_obstruction_primitives=free_carrier_obstruction_primitives,
    )


def _bulk_ion_atmosphere_state(
    mass_solution: _MassBalanceSolution,
    templates: tuple[_MassActionTemplate, ...],
    resolved_anions: tuple[_ResolvedAnion, ...],
    li_diffusion_m2_s: float,
    li_radius_A: float,
    anion_diffusion_m2_s_by_feature: Mapping[str, float],
    anion_radius_A_by_feature: Mapping[str, float],
    effective_viscosity_cP: float,
    effective_dielectric: float,
    temperature_K: float,
    atmosphere_resistance_enabled: bool,
) -> tuple[
    BulkIonAtmosphereState,
    tuple[AnalyticAtmosphereCarrierRelaxationFormFactor, ...],
]:
    atmosphere_solver = _bulk_atmosphere_solver_name(atmosphere_resistance_enabled)
    carrier_labels = tuple([LITHIUM_CARRIER_LABEL] + [anion.feature_id for anion in resolved_anions])
    mobile_carrier_concentrations_mol_m3 = _mobile_atmosphere_carrier_concentrations_mol_m3(
        mass_solution,
        templates,
        resolved_anions,
    )
    carrier_concentrations_mol_m3: dict[str, float] = {
        LITHIUM_CARRIER_LABEL: mobile_carrier_concentrations_mol_m3[LITHIUM_CARRIER_LABEL]
    }
    carrier_charges: dict[str, int] = {LITHIUM_CARRIER_LABEL: 1}
    local_diffusivity_m2_s_by_carrier: dict[str, float] = {
        LITHIUM_CARRIER_LABEL: li_diffusion_m2_s
    }
    hydrodynamic_radius_m_by_carrier: dict[str, float] = {
        LITHIUM_CARRIER_LABEL: li_radius_A * ANGSTROM_TO_M
    }
    for resolved_anion in resolved_anions:
        context = f"salt.{resolved_anion.source_salt_name}"
        carrier_concentrations_mol_m3[resolved_anion.feature_id] = (
            mobile_carrier_concentrations_mol_m3[resolved_anion.feature_id]
        )
        carrier_charges[resolved_anion.feature_id] = int(
            _required_float(resolved_anion.salt_record, "anion_charge", context)
        )
        local_diffusivity_m2_s_by_carrier[resolved_anion.feature_id] = (
            anion_diffusion_m2_s_by_feature[resolved_anion.feature_id]
        )
        hydrodynamic_radius_m_by_carrier[resolved_anion.feature_id] = (
            anion_radius_A_by_feature[resolved_anion.feature_id] * ANGSTROM_TO_M
        )
    raw_bulk_ion_atmosphere_state = build_bulk_ion_atmosphere_state(
        BulkIonAtmosphereInput(
            carrier_labels=carrier_labels,
            carrier_concentrations_mol_m3=carrier_concentrations_mol_m3,
            carrier_charges=carrier_charges,
            local_diffusivity_m2_s_by_carrier=local_diffusivity_m2_s_by_carrier,
            hydrodynamic_radius_m_by_carrier=hydrodynamic_radius_m_by_carrier,
            viscosity_Pa_s=effective_viscosity_cP * CP_TO_PA_S,
            relative_dielectric=effective_dielectric,
            temperature_K=temperature_K,
            solver=atmosphere_solver,
        )
    )
    if not atmosphere_resistance_enabled:
        return raw_bulk_ion_atmosphere_state, tuple()
    return _apply_carrier_relaxation_self_form_factor(
        raw_bulk_ion_atmosphere_state,
        li_radius_A,
        resolved_anions,
    )


def _apply_carrier_relaxation_self_form_factor(
    raw_bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    li_solvated_radius_A: float,
    resolved_anions: tuple[_ResolvedAnion, ...],
) -> tuple[
    BulkIonAtmosphereState,
    tuple[AnalyticAtmosphereCarrierRelaxationFormFactor, ...],
]:
    if math.isinf(raw_bulk_ion_atmosphere_state.kappa_inv_m):
        kappa_m_inv = 0.0
    else:
        kappa_m_inv = 1.0 / _positive_float(
            raw_bulk_ion_atmosphere_state.kappa_inv_m,
            "raw_bulk_ion_atmosphere_state.kappa_inv_m",
        )
    scale_by_carrier_label = {
        carrier_label: NORMALIZED_PROBABILITY_SUM
        for carrier_label in raw_bulk_ion_atmosphere_state.carrier_labels
    }
    form_factors: list[AnalyticAtmosphereCarrierRelaxationFormFactor] = []
    lithium_form_factor_squared = _gaussian_charge_cloud_self_form_factor_squared(
        kappa_m_inv,
        _positive_float(li_solvated_radius_A, "li_solvated_radius_A"),
    )
    scale_by_carrier_label[LITHIUM_CARRIER_LABEL] = math.sqrt(lithium_form_factor_squared)
    form_factors.append(
        AnalyticAtmosphereCarrierRelaxationFormFactor(
            carrier_label=LITHIUM_CARRIER_LABEL,
            carrier_kind="cation",
            source_species_name=LITHIUM_CARRIER_LABEL,
            charge_cloud_radius_A=float(li_solvated_radius_A),
            form_factor_squared=float(lithium_form_factor_squared),
        )
    )
    for resolved_anion in resolved_anions:
        context = f"salt.{resolved_anion.source_salt_name}"
        anion_radius_A = _required_float(
            resolved_anion.salt_record,
            "anion_radius",
            context,
        )
        ligand_field_asymmetry = _required_float(
            resolved_anion.salt_record,
            "ligand_field_asymmetry",
            context,
        )
        charge_cloud_radius_A = anion_radius_A * ligand_field_asymmetry
        _nonnegative_float(charge_cloud_radius_A, f"{context}.analytic_charge_cloud_radius_A")
        form_factor_squared = _gaussian_charge_cloud_self_form_factor_squared(
            kappa_m_inv,
            charge_cloud_radius_A,
        )
        scale_by_carrier_label[resolved_anion.feature_id] = math.sqrt(form_factor_squared)
        form_factors.append(
            AnalyticAtmosphereCarrierRelaxationFormFactor(
                carrier_label=resolved_anion.feature_id,
                carrier_kind="anion",
                source_species_name=resolved_anion.source_salt_name,
                charge_cloud_radius_A=float(charge_cloud_radius_A),
                form_factor_squared=float(form_factor_squared),
            )
        )
    diagonal_scale = np.asarray(
        [
            scale_by_carrier_label[carrier_label]
            for carrier_label in raw_bulk_ion_atmosphere_state.carrier_labels
        ],
        dtype=float,
    )
    scaled_relaxation_resistance_matrix_kg_s = (
        diagonal_scale[:, None]
        * raw_bulk_ion_atmosphere_state.resistance_rel_kg_s
        * diagonal_scale[None, :]
    )
    scaled_total_resistance_matrix_kg_s = (
        raw_bulk_ion_atmosphere_state.resistance_ep_kg_s
        + scaled_relaxation_resistance_matrix_kg_s
    )
    _validate_symmetric_psd_matrix(
        scaled_relaxation_resistance_matrix_kg_s,
        "scaled_relaxation_resistance_matrix_kg_s",
    )
    _validate_symmetric_psd_matrix(
        scaled_total_resistance_matrix_kg_s,
        "scaled_total_resistance_matrix_kg_s",
    )
    return (
        replace(
            raw_bulk_ion_atmosphere_state,
            resistance_matrix_kg_s=scaled_total_resistance_matrix_kg_s,
            resistance_rel_kg_s=scaled_relaxation_resistance_matrix_kg_s,
        ),
        tuple(form_factors),
    )


def _gaussian_charge_cloud_self_form_factor_squared(
    kappa_m_inv: float,
    charge_cloud_radius_A: float,
) -> float:
    _nonnegative_float(kappa_m_inv, "kappa_m_inv")
    _nonnegative_float(charge_cloud_radius_A, "charge_cloud_radius_A")
    charge_cloud_radius_m = charge_cloud_radius_A * ANGSTROM_TO_M
    form_factor_argument = (
        (kappa_m_inv * charge_cloud_radius_m)
        * (kappa_m_inv * charge_cloud_radius_m)
        / float(AXIS_COUNT)
    )
    form_factor_squared = math.exp(-form_factor_argument)
    if form_factor_squared <= 0.0 or form_factor_squared > NORMALIZED_PROBABILITY_SUM:
        raise ValueError(
            "relaxation_form_factor_squared must be in (0, 1], "
            f"got {form_factor_squared}"
        )
    return float(form_factor_squared)


def _atmosphere_block_diagnostics(
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    form_factors: tuple[AnalyticAtmosphereCarrierRelaxationFormFactor, ...],
) -> AnalyticAtmosphereBlockDiagnostics:
    carrier_index_by_label = {
        carrier_label: carrier_index
        for carrier_index, carrier_label in enumerate(bulk_ion_atmosphere_state.carrier_labels)
    }
    if LITHIUM_CARRIER_LABEL not in carrier_index_by_label:
        raise ValueError("bulk atmosphere state must contain lithium carrier")
    lithium_index = carrier_index_by_label[LITHIUM_CARRIER_LABEL]
    anion_indices = tuple(
        carrier_index
        for carrier_label, carrier_index in carrier_index_by_label.items()
        if carrier_label != LITHIUM_CARRIER_LABEL
    )
    relaxation_matrix = np.asarray(
        bulk_ion_atmosphere_state.resistance_rel_kg_s,
        dtype=float,
    )
    electrophoretic_matrix = np.asarray(
        bulk_ion_atmosphere_state.resistance_ep_kg_s,
        dtype=float,
    )
    lithium_anion_block = relaxation_matrix[
        np.ix_((lithium_index,), anion_indices)
    ]
    anion_anion_block = relaxation_matrix[np.ix_(anion_indices, anion_indices)]
    anion_anion_cross_block = anion_anion_block.copy()
    if anion_anion_cross_block.size:
        np.fill_diagonal(anion_anion_cross_block, 0.0)
    form_factor_by_label = {
        form_factor.carrier_label: form_factor.form_factor_squared
        for form_factor in form_factors
    }
    lithium_form_factor_squared = form_factor_by_label.get(
        LITHIUM_CARRIER_LABEL,
        NORMALIZED_PROBABILITY_SUM,
    )
    anion_form_factor_values = tuple(
        form_factor_by_label.get(carrier_label, NORMALIZED_PROBABILITY_SUM)
        for carrier_label in bulk_ion_atmosphere_state.carrier_labels
        if carrier_label != LITHIUM_CARRIER_LABEL
    )
    minimum_anion_form_factor_squared = (
        min(anion_form_factor_values)
        if anion_form_factor_values
        else NORMALIZED_PROBABILITY_SUM
    )
    minimum_lithium_anion_cross_form_factor = (
        min(
            math.sqrt(lithium_form_factor_squared)
            * math.sqrt(anion_form_factor_squared)
            for anion_form_factor_squared in anion_form_factor_values
        )
        if anion_form_factor_values
        else NORMALIZED_PROBABILITY_SUM
    )
    return AnalyticAtmosphereBlockDiagnostics(
        electrophoretic_trace_kg_s=float(np.trace(electrophoretic_matrix)),
        relaxation_trace_kg_s=float(np.trace(relaxation_matrix)),
        relaxation_lithium_self_trace_kg_s=float(
            relaxation_matrix[lithium_index, lithium_index]
        ),
        relaxation_anion_self_trace_kg_s=float(
            math.fsum(relaxation_matrix[index, index] for index in anion_indices)
        ),
        relaxation_lithium_anion_cross_frobenius_kg_s=float(
            math.sqrt(2.0) * np.linalg.norm(lithium_anion_block)
        ),
        relaxation_anion_anion_cross_frobenius_kg_s=float(
            np.linalg.norm(anion_anion_cross_block)
        ),
        lithium_form_factor_squared=float(lithium_form_factor_squared),
        minimum_anion_form_factor_squared=float(minimum_anion_form_factor_squared),
        minimum_lithium_anion_cross_form_factor=float(
            minimum_lithium_anion_cross_form_factor
        ),
    )


def _carrier_cage_primitives(
    transport_states: tuple[AnalyticTransportStatePrimitive, ...],
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    mass_solution: _MassBalanceSolution,
    templates: tuple[_MassActionTemplate, ...],
    resolved_anions: tuple[_ResolvedAnion, ...],
    li_radius_A: float,
    temperature_K: float,
) -> tuple[AnalyticCarrierCagePrimitive, ...]:
    mobile_carrier_concentrations_mol_m3 = _mobile_atmosphere_carrier_concentrations_mol_m3(
        mass_solution,
        templates,
        resolved_anions,
    )
    compact_driver_by_carrier = _compact_driver_by_carrier(
        mobile_carrier_concentrations_mol_m3,
        resolved_anions,
        li_radius_A,
    )
    steric_driver = _nonnegative_float(
        bulk_ion_atmosphere_state.steric_volume_fraction,
        "bulk_ion_atmosphere_state.steric_volume_fraction",
    )
    if steric_driver >= NORMALIZED_PROBABILITY_SUM:
        raise ValueError("steric_driver must be below one")
    cage_primitives: list[AnalyticCarrierCagePrimitive] = []
    for transport_state in transport_states:
        if transport_state.state_kind not in (STATE_KIND_FREE_LI, STATE_KIND_FREE_ANION):
            continue
        carrier_label = _carrier_label_for_free_transport_state(transport_state)
        compact_driver = compact_driver_by_carrier[carrier_label]
        caged_fraction = steric_driver * compact_driver
        mobile_fraction = NORMALIZED_PROBABILITY_SUM - caged_fraction
        if mobile_fraction <= 0.0 or caged_fraction < 0.0:
            raise ValueError(f"{transport_state.label} cage fractions are invalid")
        caged_diffusion_scale = (
            (NORMALIZED_PROBABILITY_SUM - steric_driver)
            * (NORMALIZED_PROBABILITY_SUM - compact_driver)
        )
        if caged_diffusion_scale <= 0.0 or caged_diffusion_scale > NORMALIZED_PROBABILITY_SUM:
            raise ValueError(f"{transport_state.label} caged diffusion scale is invalid")
        exchange_rate_s_inv = _carrier_cage_exchange_rate_s_inv(
            transport_state,
            temperature_K,
        )
        cage_primitives.append(
            AnalyticCarrierCagePrimitive(
                carrier_label=transport_state.label,
                carrier_kind=transport_state.state_kind,
                mobile_fraction=float(mobile_fraction),
                caged_fraction=float(caged_fraction),
                caged_diffusion_scale=float(caged_diffusion_scale),
                exchange_rate_s_inv=float(exchange_rate_s_inv),
                steric_driver=float(steric_driver),
                compact_driver=float(compact_driver),
            )
        )
    return tuple(cage_primitives)


def _free_carrier_obstruction_primitives(
    mass_solution: _MassBalanceSolution,
    templates: tuple[_MassActionTemplate, ...],
    resolved_anions: tuple[_ResolvedAnion, ...],
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    analytic_recipe: AnalyticMoriRecipe,
    species_catalog: AnalyticMoriSpeciesCatalog,
    primitive_options: _AnalyticPrimitiveOptions,
) -> tuple[AnalyticFreeCarrierObstructionPrimitive, ...]:
    mobile_carrier_concentrations_mol_m3 = _mobile_atmosphere_carrier_concentrations_mol_m3(
        mass_solution,
        templates,
        resolved_anions,
    )
    (
        lithium_compact_anion_driver,
        compact_anion_driver_by_feature,
    ) = _compact_anion_obstruction_drivers(
        mobile_carrier_concentrations_mol_m3,
        resolved_anions,
    )
    dense_parameters = _dense_free_volume_obstruction_parameters()
    steric_volume_fraction = _unit_interval_float(
        bulk_ion_atmosphere_state.steric_volume_fraction,
        "bulk_ion_atmosphere_state.steric_volume_fraction",
    )
    steric_driver = _smoothstep(
        OBSTRUCTION_STERIC_ONSET,
        OBSTRUCTION_STERIC_FULL,
        steric_volume_fraction,
    )
    carbonate_driver = _solvent_carbonate_driver(
        analytic_recipe.solvent_volume_fractions,
        species_catalog,
    )
    high_salt_driver = _smoothstep(
        OBSTRUCTION_HIGH_SALT_ONSET_M,
        OBSTRUCTION_HIGH_SALT_FULL_M,
        math.fsum(analytic_recipe.salt_molarities_M.values()),
    )
    low_donor_driver = _reverse_smoothstep(
        OBSTRUCTION_LOW_DONOR_ONSET,
        OBSTRUCTION_LOW_DONOR_FULL,
        _weighted_solvent_donor_number(
            analytic_recipe.solvent_volume_fractions,
            species_catalog,
        ),
    )
    primitives: list[AnalyticFreeCarrierObstructionPrimitive] = []
    lithium_local_obstruction_driver = (
        steric_driver
        * lithium_compact_anion_driver
        * carbonate_driver
        * high_salt_driver
        * low_donor_driver
    )
    lithium_dense_free_volume_driver = _dense_free_volume_obstruction_driver(
        steric_volume_fraction,
        lithium_compact_anion_driver,
        carbonate_driver,
        high_salt_driver,
        low_donor_driver,
        dense_parameters,
        "free_li.dense_free_volume_obstruction",
    )
    lithium_dense_obstruction_factor = (
        dense_parameters.free_li_obstruction_strength
        * lithium_dense_free_volume_driver
    )
    lithium_obstruction_factor, lithium_diffusion_scale = _obstruction_factor_and_scale(
        FREE_LI_OBSTRUCTION_MAX,
        lithium_local_obstruction_driver,
        lithium_dense_obstruction_factor,
        primitive_options.free_li_local_obstruction_enabled,
        primitive_options.forced_free_li_translation_scale_enabled,
        primitive_options.forced_free_li_translation_scale,
        "free_li.local_obstruction",
    )
    primitives.append(
        AnalyticFreeCarrierObstructionPrimitive(
            carrier_label="free_li",
            carrier_kind="lithium",
            source_species_name=LITHIUM_CARRIER_LABEL,
            obstruction_factor=float(lithium_obstruction_factor),
            diffusion_scale=float(lithium_diffusion_scale),
            dense_free_volume_driver=float(lithium_dense_free_volume_driver),
            dense_free_volume_obstruction_factor=float(lithium_dense_obstruction_factor),
            steric_driver=float(steric_driver),
            compact_anion_driver=float(lithium_compact_anion_driver),
            carbonate_driver=float(carbonate_driver),
            high_salt_driver=float(high_salt_driver),
            low_donor_driver=float(low_donor_driver),
        )
    )
    for resolved_anion in resolved_anions:
        compact_anion_driver = compact_anion_driver_by_feature[resolved_anion.feature_id]
        anion_local_obstruction_driver = (
            steric_driver
            * compact_anion_driver
            * carbonate_driver
            * high_salt_driver
            * low_donor_driver
        )
        anion_dense_free_volume_driver = _dense_free_volume_obstruction_driver(
            steric_volume_fraction,
            compact_anion_driver,
            carbonate_driver,
            high_salt_driver,
            low_donor_driver,
            dense_parameters,
            f"{resolved_anion.feature_id}.dense_free_volume_obstruction",
        )
        anion_dense_obstruction_factor = (
            dense_parameters.compact_anion_obstruction_strength
            * anion_dense_free_volume_driver
        )
        anion_obstruction_factor, anion_diffusion_scale = _obstruction_factor_and_scale(
            COMPACT_ANION_OBSTRUCTION_MAX,
            anion_local_obstruction_driver,
            anion_dense_obstruction_factor,
            primitive_options.compact_anion_local_obstruction_enabled,
            primitive_options.forced_compact_anion_translation_scale_enabled,
            primitive_options.forced_compact_anion_translation_scale,
            f"{resolved_anion.feature_id}.local_obstruction",
        )
        primitives.append(
            AnalyticFreeCarrierObstructionPrimitive(
                carrier_label=f"{resolved_anion.feature_id}_free",
                carrier_kind="anion",
                source_species_name=resolved_anion.source_salt_name,
                obstruction_factor=float(anion_obstruction_factor),
                diffusion_scale=float(anion_diffusion_scale),
                dense_free_volume_driver=float(anion_dense_free_volume_driver),
                dense_free_volume_obstruction_factor=float(anion_dense_obstruction_factor),
                steric_driver=float(steric_driver),
                compact_anion_driver=float(compact_anion_driver),
                carbonate_driver=float(carbonate_driver),
                high_salt_driver=float(high_salt_driver),
                low_donor_driver=float(low_donor_driver),
            )
        )
    return tuple(primitives)


def _compact_anion_obstruction_drivers(
    mobile_carrier_concentrations_mol_m3: Mapping[str, float],
    resolved_anions: tuple[_ResolvedAnion, ...],
) -> tuple[float, dict[str, float]]:
    weighted_compactness_sum = 0.0
    concentration_sum = 0.0
    compactness_by_feature: dict[str, float] = {}
    for resolved_anion in resolved_anions:
        context = f"salt.{resolved_anion.source_salt_name}"
        anion_radius_A = _positive_float(
            _required_float(resolved_anion.salt_record, "anion_radius", context),
            f"{context}.anion_radius",
        )
        compactness = NORMALIZED_PROBABILITY_SUM / (
            NORMALIZED_PROBABILITY_SUM
            + (anion_radius_A / OBSTRUCTION_COMPACT_ANION_RADIUS_A)
            ** OBSTRUCTION_COMPACT_ANION_SHARPNESS
        )
        compactness_by_feature[resolved_anion.feature_id] = float(compactness)
        concentration_mol_m3 = mobile_carrier_concentrations_mol_m3[
            resolved_anion.feature_id
        ]
        weighted_compactness_sum += concentration_mol_m3 * compactness
        concentration_sum += concentration_mol_m3
    if concentration_sum <= 0.0:
        raise ValueError("mobile anion concentration must be positive for obstruction driver")
    return float(weighted_compactness_sum / concentration_sum), compactness_by_feature


def _obstruction_factor_and_scale(
    obstruction_factor_max: float,
    driver: float,
    dense_obstruction_factor: float,
    obstruction_enabled: bool,
    forced_translation_scale_enabled: bool,
    forced_translation_scale: float,
    context: str,
) -> tuple[float, float]:
    if forced_translation_scale_enabled:
        diffusion_scale = _translation_scale(forced_translation_scale, f"{context}.forced_scale")
        return float((NORMALIZED_PROBABILITY_SUM / diffusion_scale) - NORMALIZED_PROBABILITY_SUM), diffusion_scale
    _unit_interval_float(driver, f"{context}.driver")
    obstruction_factor = _nonnegative_float(
        dense_obstruction_factor,
        f"{context}.dense_obstruction_factor",
    )
    if obstruction_enabled:
        obstruction_factor += _nonnegative_float(
            obstruction_factor_max,
            f"{context}.obstruction_factor_max",
        ) * driver
    if obstruction_factor == 0.0:
        return 0.0, NORMALIZED_PROBABILITY_SUM
    diffusion_scale = NORMALIZED_PROBABILITY_SUM / (
        NORMALIZED_PROBABILITY_SUM + obstruction_factor
    )
    return float(obstruction_factor), float(diffusion_scale)


def _free_carrier_obstruction_scale(
    obstruction_primitives: tuple[AnalyticFreeCarrierObstructionPrimitive, ...],
    carrier_label: str,
) -> float:
    for obstruction_primitive in obstruction_primitives:
        if obstruction_primitive.carrier_label == carrier_label:
            return _translation_scale(
                obstruction_primitive.diffusion_scale,
                f"{carrier_label}.free_carrier_obstruction.diffusion_scale",
            )
    raise ValueError(f"missing free-carrier obstruction primitive for {carrier_label}")


def _raw_selective_cage_driver(
    dense_free_volume_driver: float,
    cage_parameters: _SelectiveCarrierCageParameters,
    context: str,
) -> float:
    parsed_dense_driver = _unit_interval_float(
        dense_free_volume_driver,
        f"{context}.dense_free_volume_driver",
    )
    return float(parsed_dense_driver ** cage_parameters.dense_driver_power)


def _maximum_raw_selective_cage_driver(
    obstruction_primitives: tuple[AnalyticFreeCarrierObstructionPrimitive, ...],
    cage_parameters: _SelectiveCarrierCageParameters,
) -> float:
    if not obstruction_primitives:
        raise ValueError("selective cage driver requires obstruction primitives")
    return float(
        max(
            _raw_selective_cage_driver(
                obstruction_primitive.dense_free_volume_driver,
                cage_parameters,
                obstruction_primitive.carrier_label,
            )
            for obstruction_primitive in obstruction_primitives
        )
    )


def _descriptor_atmosphere_release_primitive(
    obstruction_primitives: tuple[AnalyticFreeCarrierObstructionPrimitive, ...],
    analytic_recipe: AnalyticMoriRecipe,
    species_catalog: AnalyticMoriSpeciesCatalog,
    effective_viscosity_cP: float,
    effective_dielectric: float,
    primitive_options: _AnalyticPrimitiveOptions,
) -> AnalyticDescriptorAtmosphereReleasePrimitive:
    cage_parameters = _selective_carrier_cage_parameters()
    release_parameters = _descriptor_atmosphere_release_parameters()
    raw_cage_driver = _maximum_raw_selective_cage_driver(
        obstruction_primitives,
        cage_parameters,
    )
    weak_cage_driver = NORMALIZED_PROBABILITY_SUM - raw_cage_driver
    carbonate_descriptor = _solvent_carbonate_driver(
        analytic_recipe.solvent_volume_fractions,
        species_catalog,
    )
    mixture_descriptor_driver = _smoothstep(
        release_parameters.mixture_descriptor_on_value,
        release_parameters.mixture_descriptor_full_value,
        carbonate_descriptor,
    )
    high_viscosity_driver = _smoothstep(
        release_parameters.viscosity_on_cP,
        release_parameters.viscosity_full_cP,
        effective_viscosity_cP,
    )
    low_dielectric_driver = _reverse_smoothstep(
        release_parameters.dielectric_on_value,
        release_parameters.dielectric_full_value,
        effective_dielectric,
    )
    low_donor_driver = _reverse_smoothstep(
        release_parameters.donor_on_value,
        release_parameters.donor_full_value,
        _weighted_solvent_donor_number(
            analytic_recipe.solvent_volume_fractions,
            species_catalog,
        ),
    )
    release_driver = (
        mixture_descriptor_driver
        * high_viscosity_driver
        * low_dielectric_driver
        * low_donor_driver
        * _unit_interval_float(
            weak_cage_driver,
            "descriptor_atmosphere_release.weak_cage_driver",
        )
    )
    if primitive_options.descriptor_atmosphere_relaxation_release_enabled:
        relaxation_scale = (
            NORMALIZED_PROBABILITY_SUM
            - release_parameters.relaxation_release_strength * release_driver
        )
    else:
        relaxation_scale = NORMALIZED_PROBABILITY_SUM
    if primitive_options.descriptor_atmosphere_ep_release_enabled:
        electrophoretic_scale = (
            NORMALIZED_PROBABILITY_SUM
            - release_parameters.electrophoretic_release_strength * release_driver
        )
    else:
        electrophoretic_scale = NORMALIZED_PROBABILITY_SUM
    return AnalyticDescriptorAtmosphereReleasePrimitive(
        mixture_release_descriptor=float(carbonate_descriptor),
        high_viscosity_driver=float(high_viscosity_driver),
        low_dielectric_driver=float(low_dielectric_driver),
        low_donor_driver=float(low_donor_driver),
        weak_cage_driver=float(weak_cage_driver),
        release_driver=float(release_driver),
        relaxation_scale=_unit_interval_float(
            relaxation_scale,
            "descriptor_atmosphere_release.relaxation_scale",
        ),
        electrophoretic_scale=_unit_interval_float(
            electrophoretic_scale,
            "descriptor_atmosphere_release.electrophoretic_scale",
        ),
    )


def _apply_descriptor_atmosphere_release(
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    descriptor_release_primitive: AnalyticDescriptorAtmosphereReleasePrimitive,
) -> BulkIonAtmosphereState:
    scaled_relaxation_resistance_matrix_kg_s = (
        descriptor_release_primitive.relaxation_scale
        * bulk_ion_atmosphere_state.resistance_rel_kg_s
    )
    scaled_electrophoretic_resistance_matrix_kg_s = (
        descriptor_release_primitive.electrophoretic_scale
        * bulk_ion_atmosphere_state.resistance_ep_kg_s
    )
    scaled_total_resistance_matrix_kg_s = (
        scaled_electrophoretic_resistance_matrix_kg_s
        + scaled_relaxation_resistance_matrix_kg_s
    )
    _validate_symmetric_psd_matrix(
        scaled_relaxation_resistance_matrix_kg_s,
        "descriptor_scaled_relaxation_resistance_matrix_kg_s",
    )
    _validate_symmetric_psd_matrix(
        scaled_electrophoretic_resistance_matrix_kg_s,
        "descriptor_scaled_electrophoretic_resistance_matrix_kg_s",
    )
    _validate_symmetric_psd_matrix(
        scaled_total_resistance_matrix_kg_s,
        "descriptor_scaled_total_resistance_matrix_kg_s",
    )
    return replace(
        bulk_ion_atmosphere_state,
        resistance_matrix_kg_s=scaled_total_resistance_matrix_kg_s,
        resistance_ep_kg_s=scaled_electrophoretic_resistance_matrix_kg_s,
        resistance_rel_kg_s=scaled_relaxation_resistance_matrix_kg_s,
    )


def _selective_carrier_cage_primitives(
    transport_states: tuple[AnalyticTransportStatePrimitive, ...],
    carrier_cage_primitives: tuple[AnalyticCarrierCagePrimitive, ...],
    obstruction_primitives: tuple[AnalyticFreeCarrierObstructionPrimitive, ...],
    descriptor_release_primitive: AnalyticDescriptorAtmosphereReleasePrimitive,
) -> tuple[AnalyticSelectiveCarrierCagePrimitive, ...]:
    cage_parameters = _selective_carrier_cage_parameters()
    obstruction_primitive_by_label = {
        obstruction_primitive.carrier_label: obstruction_primitive
        for obstruction_primitive in obstruction_primitives
    }
    carrier_cage_primitive_by_label = {
        carrier_cage_primitive.carrier_label: carrier_cage_primitive
        for carrier_cage_primitive in carrier_cage_primitives
    }
    selective_primitives: list[AnalyticSelectiveCarrierCagePrimitive] = []
    for transport_state in transport_states:
        if transport_state.state_kind not in (STATE_KIND_FREE_LI, STATE_KIND_FREE_ANION):
            continue
        if transport_state.label not in obstruction_primitive_by_label:
            raise ValueError(f"missing obstruction primitive for {transport_state.label}")
        if transport_state.label not in carrier_cage_primitive_by_label:
            raise ValueError(f"missing static cage primitive for {transport_state.label}")
        obstruction_primitive = obstruction_primitive_by_label[transport_state.label]
        carrier_cage_primitive = carrier_cage_primitive_by_label[transport_state.label]
        raw_cage_driver = _raw_selective_cage_driver(
            obstruction_primitive.dense_free_volume_driver,
            cage_parameters,
            transport_state.label,
        )
        static_cage_ratio = (
            carrier_cage_primitive.caged_fraction
            / cage_parameters.caged_fraction_max
        )
        _nonnegative_float(
            static_cage_ratio,
            f"{transport_state.label}.static_cage_ratio",
        )
        static_cage_driver = NORMALIZED_PROBABILITY_SUM - math.exp(
            -static_cage_ratio
        )
        cage_basis_driver = max(raw_cage_driver, static_cage_driver)
        selective_cage_driver = raw_cage_driver * (
            NORMALIZED_PROBABILITY_SUM
            - (
                cage_parameters.descriptor_release_suppression_strength
                * descriptor_release_primitive.release_driver
            )
        )
        selective_cage_driver = max(
            selective_cage_driver,
            static_cage_driver
            * (
                NORMALIZED_PROBABILITY_SUM
                - (
                    cage_parameters.descriptor_release_suppression_strength
                    * descriptor_release_primitive.release_driver
                )
            ),
        )
        selective_cage_driver = _unit_interval_float(
            selective_cage_driver,
            f"{transport_state.label}.selective_cage_driver",
        )
        caged_fraction = (
            cage_parameters.caged_fraction_max * selective_cage_driver
        )
        caged_diffusion_scale = (
            NORMALIZED_PROBABILITY_SUM
            - (
                (NORMALIZED_PROBABILITY_SUM - cage_parameters.caged_diffusion_scale_min)
                * selective_cage_driver
            )
        )
        selective_primitives.append(
            AnalyticSelectiveCarrierCagePrimitive(
                carrier_label=transport_state.label,
                carrier_kind=transport_state.state_kind,
                dense_driver=float(cage_basis_driver),
                descriptor_release_driver=float(
                    descriptor_release_primitive.release_driver
                ),
                selective_cage_driver=float(selective_cage_driver),
                caged_fraction=_unit_interval_float(
                    caged_fraction,
                    f"{transport_state.label}.selective_caged_fraction",
                ),
                caged_diffusion_scale=_translation_scale(
                    caged_diffusion_scale,
                    f"{transport_state.label}.selective_caged_diffusion_scale",
                ),
            )
        )
    return tuple(selective_primitives)


def _timescale_structural_memory_primitives(
    transport_states: tuple[AnalyticTransportStatePrimitive, ...],
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    analytic_recipe: AnalyticMoriRecipe,
    species_catalog: AnalyticMoriSpeciesCatalog,
    effective_viscosity_cP: float,
    temperature_K: float,
) -> tuple[AnalyticTimescaleStructuralMemoryPrimitive, ...]:
    parameters = _timescale_structural_memory_parameters()
    if math.isinf(bulk_ion_atmosphere_state.kappa_inv_m):
        return tuple()
    kappa_m_inv = 1.0 / _positive_float(
        bulk_ion_atmosphere_state.kappa_inv_m,
        "bulk_ion_atmosphere_state.kappa_inv_m",
    )
    structural_relaxation_time_s = _structural_relaxation_time_s(
        analytic_recipe,
        species_catalog,
        effective_viscosity_cP,
        temperature_K,
        bulk_ion_atmosphere_state.steric_volume_fraction,
        parameters,
    )
    void_radius_m = _free_volume_void_radius_m(
        analytic_recipe,
        species_catalog,
        bulk_ion_atmosphere_state.steric_volume_fraction,
        parameters,
    )
    orientation_count = len(TRANSLATION_EVENT_AXES)
    primitives: list[AnalyticTimescaleStructuralMemoryPrimitive] = []
    for transport_state in transport_states:
        net_charge_number = float(math.fsum(transport_state.charge_vector))
        if net_charge_number == 0.0:
            continue
        local_diffusivity_m2_s = _local_center_of_mass_diffusion_m2_s(
            transport_state,
            temperature_K,
        )
        atmosphere_relaxation_diffusivity_m2_s = (
            _countercharge_relaxation_diffusivity_m2_s(
                transport_state,
                transport_states,
                temperature_K,
            )
        )
        hydrodynamic_radius_m = _positive_float(
            transport_state.current_relaxation_length_m,
            f"{transport_state.label}.current_relaxation_length_m",
        )
        jump_length_m = parameters.jump_length_radius_scale * hydrodynamic_radius_m
        tau_hop_s = (jump_length_m * jump_length_m) / local_diffusivity_m2_s
        k_atmosphere_exit_s_inv = (
            atmosphere_relaxation_diffusivity_m2_s * kappa_m_inv * kappa_m_inv
        )
        tau_atmosphere_s = 1.0 / _positive_float(
            k_atmosphere_exit_s_inv,
            f"{transport_state.label}.k_atmosphere_exit_s_inv",
        )
        de_hop_structural = structural_relaxation_time_s / tau_hop_s
        atmosphere_structural_ratio = structural_relaxation_time_s / tau_atmosphere_s
        size_void_ratio = (
            0.0
            if math.isinf(void_radius_m)
            else hydrodynamic_radius_m
            / _positive_float(void_radius_m, "free_volume_void_radius_m")
        )
        structural_cage_fraction = _structural_cage_fraction(
            de_hop_structural,
            size_void_ratio,
            bulk_ion_atmosphere_state.steric_volume_fraction,
            parameters,
            transport_state.label,
        )
        atmosphere_capture_fraction = _atmosphere_capture_fraction(
            transport_state,
            parameters,
        )
        if atmosphere_capture_fraction == 0.0:
            continue
        k_capture_s_inv = (
            atmosphere_capture_fraction
            * local_diffusivity_m2_s
            / (jump_length_m * jump_length_m)
        )
        if k_capture_s_inv <= 0.0:
            raise ValueError(f"{transport_state.label} capture rate must be positive")
        residence_ratio_atmosphere = k_capture_s_inv / k_atmosphere_exit_s_inv
        if structural_cage_fraction >= NORMALIZED_PROBABILITY_SUM:
            raise ValueError(
                f"{transport_state.label} structural cage fraction must be below one"
            )
        if structural_cage_fraction > 0.0:
            structural_residence_ratio = structural_cage_fraction / (
                NORMALIZED_PROBABILITY_SUM - structural_cage_fraction
            )
            k_structural_capture_s_inv = (
                structural_cage_fraction / structural_relaxation_time_s
            )
            k_structural_release_s_inv = (
                (NORMALIZED_PROBABILITY_SUM - structural_cage_fraction)
                / structural_relaxation_time_s
            )
        else:
            structural_residence_ratio = 0.0
            k_structural_capture_s_inv = 0.0
            k_structural_release_s_inv = 1.0 / structural_relaxation_time_s
        concentration_denominator = (
            NORMALIZED_PROBABILITY_SUM
            + orientation_count
            * residence_ratio_atmosphere
            * (NORMALIZED_PROBABILITY_SUM + structural_residence_ratio)
        )
        mobile_concentration_mol_m3 = (
            transport_state.concentration_mol_m3 / concentration_denominator
        )
        atmosphere_concentration_per_orientation_mol_m3 = (
            residence_ratio_atmosphere * mobile_concentration_mol_m3
        )
        structural_cage_concentration_per_orientation_mol_m3 = (
            structural_residence_ratio
            * atmosphere_concentration_per_orientation_mol_m3
        )
        primitives.append(
            AnalyticTimescaleStructuralMemoryPrimitive(
                carrier_label=transport_state.label,
                carrier_kind=transport_state.state_kind,
                local_diffusivity_m2_s=float(local_diffusivity_m2_s),
                atmosphere_relaxation_diffusivity_m2_s=float(
                    atmosphere_relaxation_diffusivity_m2_s
                ),
                jump_length_m=float(jump_length_m),
                tau_hop_s=float(tau_hop_s),
                tau_atmosphere_s=float(tau_atmosphere_s),
                tau_structural_s=float(structural_relaxation_time_s),
                de_hop_structural=float(de_hop_structural),
                atmosphere_structural_ratio=float(atmosphere_structural_ratio),
                size_void_ratio=float(size_void_ratio),
                atmosphere_capture_fraction=float(atmosphere_capture_fraction),
                structural_cage_fraction=float(structural_cage_fraction),
                k_capture_s_inv=float(k_capture_s_inv),
                k_atmosphere_exit_s_inv=float(k_atmosphere_exit_s_inv),
                k_structural_capture_s_inv=float(k_structural_capture_s_inv),
                k_structural_release_s_inv=float(k_structural_release_s_inv),
                mobile_concentration_mol_m3=float(mobile_concentration_mol_m3),
                atmosphere_concentration_per_orientation_mol_m3=float(
                    atmosphere_concentration_per_orientation_mol_m3
                ),
                structural_cage_concentration_per_orientation_mol_m3=float(
                    structural_cage_concentration_per_orientation_mol_m3
                ),
            )
        )
    return tuple(primitives)


def _validate_timescale_structural_memory_coverage(
    transport_states: tuple[AnalyticTransportStatePrimitive, ...],
    timescale_structural_memory_primitives: tuple[
        AnalyticTimescaleStructuralMemoryPrimitive, ...
    ],
) -> None:
    charged_transport_labels = tuple(
        transport_state.label
        for transport_state in transport_states
        if float(math.fsum(transport_state.charge_vector)) != 0.0
    )
    primitive_labels = frozenset(
        timescale_primitive.carrier_label
        for timescale_primitive in timescale_structural_memory_primitives
    )
    missing_charged_labels = tuple(
        transport_label
        for transport_label in charged_transport_labels
        if transport_label not in primitive_labels
    )
    if missing_charged_labels:
        raise ValueError(
            "timescale structural-memory primitive missing charged carriers: "
            + ", ".join(missing_charged_labels)
        )


def _structural_relaxation_time_s(
    analytic_recipe: AnalyticMoriRecipe,
    species_catalog: AnalyticMoriSpeciesCatalog,
    effective_viscosity_cP: float,
    temperature_K: float,
    steric_volume_fraction: float,
    parameters: _TimescaleStructuralMemoryParameters,
) -> float:
    parsed_steric_volume_fraction = _unit_interval_float(
        steric_volume_fraction,
        "steric_volume_fraction",
    )
    if parsed_steric_volume_fraction >= parameters.packing_fraction_limit:
        raise ValueError(
            "steric_volume_fraction must be below timescale structural-memory "
            "packing_fraction_limit"
        )
    cage_volume_m3 = _mixture_molecular_cage_volume_m3(
        analytic_recipe.solvent_volume_fractions,
        species_catalog,
    )
    viscosity_pa_s = _positive_float(effective_viscosity_cP, "effective_viscosity_cP") * CP_TO_PA_S
    free_volume_coordinate = parsed_steric_volume_fraction / (
        parameters.packing_fraction_limit - parsed_steric_volume_fraction
    )
    return float(
        viscosity_pa_s
        * cage_volume_m3
        / (K_B * temperature_K)
        * math.exp(
            parameters.structural_free_volume_barrier_strength
            * free_volume_coordinate
        )
    )


def _free_volume_void_radius_m(
    analytic_recipe: AnalyticMoriRecipe,
    species_catalog: AnalyticMoriSpeciesCatalog,
    steric_volume_fraction: float,
    parameters: _TimescaleStructuralMemoryParameters,
) -> float:
    parsed_steric_volume_fraction = _unit_interval_float(
        steric_volume_fraction,
        "steric_volume_fraction",
    )
    if parsed_steric_volume_fraction == 0.0:
        return math.inf
    if parsed_steric_volume_fraction >= parameters.packing_fraction_limit:
        raise ValueError(
            "steric_volume_fraction must be below timescale structural-memory "
            "packing_fraction_limit"
        )
    cage_volume_m3 = _mixture_molecular_cage_volume_m3(
        analytic_recipe.solvent_volume_fractions,
        species_catalog,
    )
    void_volume_m3 = cage_volume_m3 * (
        parameters.packing_fraction_limit - parsed_steric_volume_fraction
    ) / parsed_steric_volume_fraction
    return float((3.0 * void_volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0))


def _mixture_molecular_cage_volume_m3(
    solvent_volume_fractions: Mapping[str, float],
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> float:
    weighted_volume_m3 = 0.0
    volume_fraction_sum = 0.0
    for solvent_name, volume_fraction in solvent_volume_fractions.items():
        solvent_record = _required_record(species_catalog.solvents, solvent_name, "solvent")
        molecular_volume_m3 = _species_molecular_volume_m3(
            solvent_record,
            f"solvent.{solvent_name}",
        )
        parsed_volume_fraction = _nonnegative_float(
            volume_fraction,
            f"solvent.{solvent_name}.volume_fraction",
        )
        weighted_volume_m3 += parsed_volume_fraction * molecular_volume_m3
        volume_fraction_sum += parsed_volume_fraction
    if volume_fraction_sum <= 0.0:
        raise ValueError("solvent volume fraction sum must be positive")
    return float(weighted_volume_m3 / volume_fraction_sum)


def _species_molecular_volume_m3(record: SpeciesRecord, context: str) -> float:
    if "molecular_volume_A3" in record:
        return _positive_float(
            _required_float(record, "molecular_volume_A3", context),
            f"{context}.molecular_volume_A3",
        ) * 1.0e-30
    molecular_weight_g_mol = _positive_float(
        _required_float(record, "molecular_weight", context),
        f"{context}.molecular_weight",
    )
    density_g_ml = _positive_float(
        _required_float(record, "density_g_ml", context),
        f"{context}.density_g_ml",
    )
    molar_volume_m3_mol = (molecular_weight_g_mol / density_g_ml) * 1.0e-6
    return float(molar_volume_m3_mol / N_A)


def _countercharge_relaxation_diffusivity_m2_s(
    transport_state: AnalyticTransportStatePrimitive,
    transport_states: tuple[AnalyticTransportStatePrimitive, ...],
    temperature_K: float,
) -> float:
    source_charge_number = float(math.fsum(transport_state.charge_vector))
    if source_charge_number == 0.0:
        raise ValueError(f"{transport_state.label} must be charged")
    weighted_countercharge_diffusivity = 0.0
    countercharge_weight = 0.0
    for candidate_state in transport_states:
        candidate_charge_number = float(math.fsum(candidate_state.charge_vector))
        if source_charge_number * candidate_charge_number >= 0.0:
            continue
        candidate_weight = (
            candidate_state.concentration_mol_m3 * abs(candidate_charge_number)
        )
        weighted_countercharge_diffusivity += candidate_weight * (
            _local_center_of_mass_diffusion_m2_s(candidate_state, temperature_K)
        )
        countercharge_weight += candidate_weight
    if countercharge_weight <= 0.0:
        raise ValueError(
            f"{transport_state.label} requires at least one opposite-charge carrier"
        )
    return float(
        _local_center_of_mass_diffusion_m2_s(transport_state, temperature_K)
        + weighted_countercharge_diffusivity / countercharge_weight
    )


def _local_center_of_mass_diffusion_m2_s(
    transport_state: AnalyticTransportStatePrimitive,
    temperature_K: float,
) -> float:
    local_drag_kg_s = float(np.trace(transport_state.local_resistance_matrix_kg_s))
    if local_drag_kg_s <= 0.0:
        raise ValueError(f"{transport_state.label} local drag must be positive")
    return float(K_B * temperature_K / local_drag_kg_s)


def _atmosphere_capture_fraction(
    transport_state: AnalyticTransportStatePrimitive,
    parameters: _TimescaleStructuralMemoryParameters,
) -> float:
    local_drag_kg_s = float(np.trace(transport_state.local_resistance_matrix_kg_s))
    atmosphere_drag_kg_s = float(
        np.trace(transport_state.atmosphere_resistance_matrix_kg_s)
    )
    if local_drag_kg_s <= 0.0:
        raise ValueError(f"{transport_state.label} local drag must be positive")
    if atmosphere_drag_kg_s < 0.0:
        raise ValueError(f"{transport_state.label} atmosphere drag must be nonnegative")
    if atmosphere_drag_kg_s == 0.0:
        return 0.0
    return float(
        parameters.atmosphere_capture_fraction_max
        * atmosphere_drag_kg_s
        / (local_drag_kg_s + atmosphere_drag_kg_s)
    )


def _structural_cage_fraction(
    de_hop_structural: float,
    size_void_ratio: float,
    steric_volume_fraction: float,
    parameters: _TimescaleStructuralMemoryParameters,
    carrier_label: str,
) -> float:
    deborah_driver = _smoothstep(
        parameters.deborah_on_value,
        parameters.deborah_full_value,
        _nonnegative_float(de_hop_structural, f"{carrier_label}.de_hop_structural"),
    )
    size_void_driver = _smoothstep(
        parameters.size_void_ratio_on_value,
        parameters.size_void_ratio_full_value,
        _nonnegative_float(size_void_ratio, f"{carrier_label}.size_void_ratio"),
    )
    steric_driver = _smoothstep(
        parameters.steric_on_fraction,
        parameters.steric_full_fraction,
        _unit_interval_float(
            steric_volume_fraction,
            f"{carrier_label}.steric_volume_fraction",
        ),
    )
    return _unit_interval_float(
        parameters.cage_fraction_max
        * deborah_driver
        * size_void_driver
        * steric_driver,
        f"{carrier_label}.structural_cage_fraction",
    )


def _compact_driver_by_carrier(
    mobile_carrier_concentrations_mol_m3: Mapping[str, float],
    resolved_anions: tuple[_ResolvedAnion, ...],
    li_radius_A: float,
) -> dict[str, float]:
    li_radius = _positive_float(li_radius_A, "li_radius_A")
    compactness_by_anion: dict[str, float] = {}
    weighted_compactness_sum = 0.0
    concentration_sum = 0.0
    for resolved_anion in resolved_anions:
        context = f"salt.{resolved_anion.source_salt_name}"
        anion_radius_A = _required_float(
            resolved_anion.salt_record,
            "anion_radius",
            context,
        )
        compactness = li_radius / (li_radius + _positive_float(anion_radius_A, f"{context}.anion_radius"))
        compactness_by_anion[resolved_anion.feature_id] = float(compactness)
        concentration_mol_m3 = mobile_carrier_concentrations_mol_m3[
            resolved_anion.feature_id
        ]
        weighted_compactness_sum += concentration_mol_m3 * compactness
        concentration_sum += concentration_mol_m3
    if concentration_sum <= 0.0:
        raise ValueError("mobile anion concentration must be positive for cage driver")
    compact_driver_by_carrier = {
        LITHIUM_CARRIER_LABEL: float(weighted_compactness_sum / concentration_sum)
    }
    for resolved_anion in resolved_anions:
        compact_driver_by_carrier[resolved_anion.feature_id] = compactness_by_anion[
            resolved_anion.feature_id
        ]
    return compact_driver_by_carrier


def _carrier_label_for_free_transport_state(
    transport_state: AnalyticTransportStatePrimitive,
) -> str:
    if transport_state.state_kind == STATE_KIND_FREE_LI:
        return LITHIUM_CARRIER_LABEL
    if transport_state.state_kind == STATE_KIND_FREE_ANION:
        return transport_state.anion_feature_id
    raise ValueError(f"{transport_state.label} is not a free-carrier state")


def _carrier_cage_exchange_rate_s_inv(
    transport_state: AnalyticTransportStatePrimitive,
    temperature_K: float,
) -> float:
    hop_length_m = _positive_float(
        transport_state.current_relaxation_length_m,
        f"{transport_state.label}.current_relaxation_length_m",
    )
    center_of_mass_diffusion_m2_s = _center_of_mass_diffusion_m2_s(
        transport_state,
        temperature_K,
    )
    return float(center_of_mass_diffusion_m2_s / (hop_length_m * hop_length_m))


def _backjump_cage_primitives(
    transport_bundle: _TransportPrimitiveBundle,
    mass_solution: _MassBalanceSolution,
    templates: tuple[_MassActionTemplate, ...],
    resolved_anions: tuple[_ResolvedAnion, ...],
    analytic_recipe: AnalyticMoriRecipe,
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> tuple[AnalyticBackjumpCagePrimitive, ...]:
    mobile_carrier_concentrations_mol_m3 = _mobile_atmosphere_carrier_concentrations_mol_m3(
        mass_solution,
        templates,
        resolved_anions,
    )
    compact_anion_driver = _compact_anion_driver_for_backjump_cage(
        mobile_carrier_concentrations_mol_m3,
        resolved_anions,
    )
    steric_driver = _smoothstep(
        BACKJUMP_CAGE_STERIC_ONSET,
        BACKJUMP_CAGE_STERIC_FULL,
        _nonnegative_float(
            transport_bundle.bulk_ion_atmosphere_state.steric_volume_fraction,
            "bulk_ion_atmosphere_state.steric_volume_fraction",
        ),
    )
    carbonate_driver = _solvent_carbonate_driver(
        analytic_recipe.solvent_volume_fractions,
        species_catalog,
    )
    high_salt_driver = _smoothstep(
        BACKJUMP_HIGH_SALT_ONSET_M,
        BACKJUMP_HIGH_SALT_FULL_M,
        math.fsum(analytic_recipe.salt_molarities_M.values()),
    )
    low_donor_driver = NORMALIZED_PROBABILITY_SUM - _smoothstep(
        BACKJUMP_LOW_DONOR_FULL,
        BACKJUMP_LOW_DONOR_OFF,
        _weighted_solvent_donor_number(
            analytic_recipe.solvent_volume_fractions,
            species_catalog,
        ),
    )
    cage_driver = (
        steric_driver
        * compact_anion_driver
        * carbonate_driver
        * high_salt_driver
        * low_donor_driver
    )
    point_active = bool(cage_driver >= BACKJUMP_MIN_DRIVER_FOR_POINT)
    primitives: list[AnalyticBackjumpCagePrimitive] = []
    for transport_state in transport_bundle.transport_states:
        if transport_state.state_kind != STATE_KIND_FREE_LI:
            continue
        jump_length_m = _positive_float(
            transport_state.current_relaxation_length_m,
            f"{transport_state.label}.current_relaxation_length_m",
        )
        center_of_mass_diffusion_m2_s = _center_of_mass_diffusion_m2_s(
            transport_state,
            analytic_recipe.temperature_K,
        )
        cage_occupancy_fraction = BACKJUMP_F_CAGE_MAX * cage_driver
        attempt_fraction = BACKJUMP_ATTEMPT_FRACTION_MAX * cage_driver
        backjump_probability = BACKJUMP_PROBABILITY_MAX * cage_driver
        if point_active:
            _positive_float(cage_occupancy_fraction, "backjump_cage.cage_occupancy_fraction")
            _positive_float(attempt_fraction, "backjump_cage.attempt_fraction")
            _positive_float(backjump_probability, "backjump_cage.backjump_probability")
            if cage_occupancy_fraction >= NORMALIZED_PROBABILITY_SUM:
                raise ValueError("backjump cage occupancy fraction must be below one")
            if attempt_fraction >= NORMALIZED_PROBABILITY_SUM:
                raise ValueError("backjump attempt fraction must be below one")
            if backjump_probability >= NORMALIZED_PROBABILITY_SUM:
                raise ValueError("backjump probability must be below one")
            exit_rate_s_inv = (
                len(TRANSLATION_EVENT_AXES)
                * attempt_fraction
                * center_of_mass_diffusion_m2_s
                / (
                    cage_occupancy_fraction
                    * (NORMALIZED_PROBABILITY_SUM + backjump_probability)
                    * jump_length_m
                    * jump_length_m
                )
            )
            ordinary_translation_fraction = (
                NORMALIZED_PROBABILITY_SUM - attempt_fraction
            )
            direct_axis_density_m2_s_mol_m3 = (
                transport_state.concentration_mol_m3
                * cage_occupancy_fraction
                * (NORMALIZED_PROBABILITY_SUM + backjump_probability)
                * exit_rate_s_inv
                * jump_length_m
                * jump_length_m
                / len(TRANSLATION_EVENT_AXES)
            )
        else:
            exit_rate_s_inv = 0.0
            ordinary_translation_fraction = NORMALIZED_PROBABILITY_SUM
            direct_axis_density_m2_s_mol_m3 = 0.0
        direct_sigma_S_m = (
            F
            * F
            / (R * analytic_recipe.temperature_K)
            * direct_axis_density_m2_s_mol_m3
        )
        primitives.append(
            AnalyticBackjumpCagePrimitive(
                carrier_label=transport_state.label,
                carrier_kind=transport_state.state_kind,
                cage_driver=float(cage_driver),
                steric_driver=float(steric_driver),
                compact_anion_driver=float(compact_anion_driver),
                carbonate_driver=float(carbonate_driver),
                high_salt_driver=float(high_salt_driver),
                low_donor_driver=float(low_donor_driver),
                cage_occupancy_fraction=float(cage_occupancy_fraction),
                attempt_fraction=float(attempt_fraction),
                backjump_probability=float(backjump_probability),
                exit_rate_s_inv=float(exit_rate_s_inv),
                jump_length_m=float(jump_length_m),
                ordinary_translation_fraction=float(ordinary_translation_fraction),
                direct_axis_density_m2_s_mol_m3=float(direct_axis_density_m2_s_mol_m3),
                direct_sigma_mS_cm=float(direct_sigma_S_m * S_M_TO_MS_CM),
                point_active=point_active,
            )
        )
    return tuple(primitives)


def _compact_anion_driver_for_backjump_cage(
    mobile_carrier_concentrations_mol_m3: Mapping[str, float],
    resolved_anions: tuple[_ResolvedAnion, ...],
) -> float:
    compactness_sum = 0.0
    concentration_sum = 0.0
    for resolved_anion in resolved_anions:
        context = f"salt.{resolved_anion.source_salt_name}"
        anion_radius_A = _positive_float(
            _required_float(resolved_anion.salt_record, "anion_radius", context),
            f"{context}.anion_radius",
        )
        compactness = NORMALIZED_PROBABILITY_SUM / (
            NORMALIZED_PROBABILITY_SUM
            + (anion_radius_A / BACKJUMP_COMPACT_ANION_RADIUS_A)
            ** BACKJUMP_COMPACT_ANION_SHARPNESS
        )
        concentration_mol_m3 = mobile_carrier_concentrations_mol_m3[
            resolved_anion.feature_id
        ]
        compactness_sum += concentration_mol_m3 * compactness
        concentration_sum += concentration_mol_m3
    if concentration_sum <= 0.0:
        raise ValueError("mobile anion concentration must be positive for backjump cage driver")
    return float(compactness_sum / concentration_sum)


def _solvent_carbonate_driver(
    solvent_volume_fractions: Mapping[str, float],
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> float:
    weighted_carbonate_score = 0.0
    volume_fraction_sum = 0.0
    for solvent_name, volume_fraction in solvent_volume_fractions.items():
        solvent_record = _required_record(species_catalog.solvents, solvent_name, "solvent")
        functional_groups = _required_functional_groups(
            solvent_record,
            f"solvent.{solvent_name}",
        )
        carbonyl_count = _functional_group_count(functional_groups, "C=O")
        ether_oxygen_count = _functional_group_count(functional_groups, "C-O")
        carbonate_score = min(
            NORMALIZED_PROBABILITY_SUM,
            carbonyl_count,
        ) * min(
            NORMALIZED_PROBABILITY_SUM,
            ether_oxygen_count / 2.0,
        )
        weighted_carbonate_score += volume_fraction * carbonate_score
        volume_fraction_sum += volume_fraction
    if volume_fraction_sum <= 0.0:
        raise ValueError("solvent volume fraction sum must be positive")
    return float(weighted_carbonate_score / volume_fraction_sum)


def _weighted_solvent_donor_number(
    solvent_volume_fractions: Mapping[str, float],
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> float:
    weighted_donor_number = 0.0
    volume_fraction_sum = 0.0
    for solvent_name, volume_fraction in solvent_volume_fractions.items():
        solvent_record = _required_record(species_catalog.solvents, solvent_name, "solvent")
        donor_number = _required_float(
            solvent_record,
            "donor_number",
            f"solvent.{solvent_name}",
        )
        weighted_donor_number += volume_fraction * donor_number
        volume_fraction_sum += volume_fraction
    if volume_fraction_sum <= 0.0:
        raise ValueError("solvent volume fraction sum must be positive")
    return float(weighted_donor_number / volume_fraction_sum)


def _required_functional_groups(
    record: SpeciesRecord,
    context: str,
) -> Mapping[str, float]:
    if "functional_groups" not in record:
        raise ValueError(f"{context} missing required descriptor functional_groups")
    functional_groups = record["functional_groups"]
    if not isinstance(functional_groups, Mapping):
        raise TypeError(f"{context}.functional_groups must be a mapping")
    return functional_groups


def _functional_group_count(
    functional_groups: Mapping[str, float],
    group_label: str,
) -> float:
    if group_label not in functional_groups:
        return 0.0
    return _nonnegative_float(
        functional_groups[group_label],
        f"functional_groups.{group_label}",
    )


def _smoothstep(
    lower_edge: float,
    upper_edge: float,
    value: float,
) -> float:
    if upper_edge <= lower_edge:
        raise ValueError("smoothstep upper_edge must exceed lower_edge")
    scaled_value = (value - lower_edge) / (upper_edge - lower_edge)
    bounded_value = min(
        NORMALIZED_PROBABILITY_SUM,
        max(0.0, scaled_value),
    )
    return float(bounded_value * bounded_value * (3.0 - 2.0 * bounded_value))


def _reverse_smoothstep(
    upper_edge: float,
    lower_edge: float,
    value: float,
) -> float:
    if upper_edge <= lower_edge:
        raise ValueError("reverse_smoothstep upper_edge must exceed lower_edge")
    return float(
        NORMALIZED_PROBABILITY_SUM
        - _smoothstep(lower_edge, upper_edge, value)
    )


def _validate_symmetric_psd_matrix(matrix: np.ndarray, context: str) -> None:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError(f"{context} must be a square matrix")
    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(f"{context} entries must be finite")
    if not np.allclose(matrix_array, matrix_array.T):
        raise ValueError(f"{context} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix_array)
    if float(np.min(eigenvalues)) < -PSD_EIGENVALUE_ABSOLUTE_TOLERANCE:
        raise ValueError(f"{context} must be positive semidefinite")


def _bulk_atmosphere_solver_name(atmosphere_resistance_enabled: bool) -> str:
    if atmosphere_resistance_enabled:
        return FINITE_SIZE_BULK_ATMOSPHERE_SOLVER
    return "off"


def _mobile_atmosphere_carrier_concentrations_mol_m3(
    mass_solution: _MassBalanceSolution,
    templates: tuple[_MassActionTemplate, ...],
    resolved_anions: tuple[_ResolvedAnion, ...],
) -> dict[str, float]:
    carrier_concentrations_mol_m3: dict[str, float] = {
        LITHIUM_CARRIER_LABEL: (
            mass_solution.state_concentrations_M_by_label["free_li"]
            * MOLARITY_TO_MOL_M3
        )
    }
    for resolved_anion in resolved_anions:
        carrier_concentrations_mol_m3[resolved_anion.feature_id] = (
            mass_solution.state_concentrations_M_by_label[
                f"{resolved_anion.feature_id}_free"
            ]
            * MOLARITY_TO_MOL_M3
        )
    for template in templates:
        concentration_mol_m3 = (
            mass_solution.state_concentrations_M_by_label[template.label]
            * MOLARITY_TO_MOL_M3
        )
        if template.state_kind == STATE_KIND_LI2A_PLUS:
            carrier_concentrations_mol_m3[LITHIUM_CARRIER_LABEL] += concentration_mol_m3
        elif template.state_kind == STATE_KIND_LIA2_MINUS:
            anion_feature_id = resolved_anions[template.anion_index].feature_id
            carrier_concentrations_mol_m3[anion_feature_id] += concentration_mol_m3
    return carrier_concentrations_mol_m3


def _free_lithium_state(
    mass_solution: _MassBalanceSolution,
    diffusion_m2_s: float,
    li_radius_A: float,
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    atmosphere_resistance_enabled: bool,
    temperature_K: float,
) -> AnalyticTransportStatePrimitive:
    concentration_M = mass_solution.state_concentrations_M_by_label["free_li"]
    concentration_mol_m3 = concentration_M * MOLARITY_TO_MOL_M3
    local_resistance_matrix = np.asarray([[K_B * temperature_K / diffusion_m2_s]], dtype=float)
    binding_resistance_matrix = np.zeros_like(local_resistance_matrix)
    charged_center = ChargedCenter(
        label="free_li_center",
        charge=NORMALIZED_PROBABILITY_SUM,
        hydrodynamic_radius_m=li_radius_A * ANGSTROM_TO_M,
        shape_factor=NORMALIZED_PROBABILITY_SUM,
        local_diffusion_m2_s=diffusion_m2_s,
        relative_position_m=(0.0, 0.0, 0.0),
        charge_cloud_radius_available=False,
        charge_cloud_radius_A=0.0,
        charge_cloud_source=NOT_APPLICABLE_FEATURE,
        charge_cloud_site_count=0,
    )
    atmosphere_resistance_matrix = _state_atmosphere_matrix_for_mode(
        atmosphere_resistance_enabled,
        local_resistance_matrix,
        bulk_ion_atmosphere_state,
        (charged_center,),
        (LITHIUM_CARRIER_LABEL,),
    )
    resistance_matrix = (
        local_resistance_matrix
        + binding_resistance_matrix
        + atmosphere_resistance_matrix
    )
    charge_vector = np.asarray([NORMALIZED_PROBABILITY_SUM], dtype=float)
    charge_diffusivity_m2_s = float(
        charge_vector @ (K_B * temperature_K * np.linalg.inv(resistance_matrix)) @ charge_vector
    )
    return AnalyticTransportStatePrimitive(
        label="free_li",
        state_kind="FREE_LI",
        anion_feature_id=NOT_APPLICABLE_FEATURE,
        source_salt_name=SHARED_LITHIUM_POOL,
        concentration_mol_m3=float(concentration_mol_m3),
        charge_vector=(NORMALIZED_PROBABILITY_SUM,),
        local_resistance_matrix_kg_s=local_resistance_matrix,
        binding_resistance_matrix_kg_s=binding_resistance_matrix,
        atmosphere_resistance_matrix_kg_s=atmosphere_resistance_matrix,
        resistance_matrix_kg_s=resistance_matrix,
        charge_diffusivity_m2_s=charge_diffusivity_m2_s,
        current_relaxation_length_m=li_radius_A * ANGSTROM_TO_M,
        standard_free_energy_J_mol=0.0,
    )


def _free_anion_state(
    mass_solution: _MassBalanceSolution,
    resolved_anion: _ResolvedAnion,
    diffusion_m2_s: float,
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    atmosphere_resistance_enabled: bool,
    temperature_K: float,
) -> AnalyticTransportStatePrimitive:
    context = f"salt.{resolved_anion.source_salt_name}"
    label = f"{resolved_anion.feature_id}_free"
    concentration_M = mass_solution.state_concentrations_M_by_label[label]
    anion_radius_A = _required_float(resolved_anion.salt_record, "anion_radius", context)
    anion_charge = _required_float(resolved_anion.salt_record, "anion_charge", context)
    ligand_field_asymmetry = _required_float(
        resolved_anion.salt_record,
        "ligand_field_asymmetry",
        context,
    )
    concentration_mol_m3 = concentration_M * MOLARITY_TO_MOL_M3
    local_resistance_matrix = np.asarray([[K_B * temperature_K / diffusion_m2_s]], dtype=float)
    binding_resistance_matrix = np.zeros_like(local_resistance_matrix)
    charged_center = ChargedCenter(
        label=f"{resolved_anion.feature_id}_free_center",
        charge=anion_charge,
        hydrodynamic_radius_m=anion_radius_A * ANGSTROM_TO_M,
        shape_factor=NORMALIZED_PROBABILITY_SUM + ligand_field_asymmetry,
        local_diffusion_m2_s=diffusion_m2_s,
        relative_position_m=(0.0, 0.0, 0.0),
        charge_cloud_radius_available=False,
        charge_cloud_radius_A=0.0,
        charge_cloud_source=NOT_APPLICABLE_FEATURE,
        charge_cloud_site_count=0,
    )
    atmosphere_resistance_matrix = _state_atmosphere_matrix_for_mode(
        atmosphere_resistance_enabled,
        local_resistance_matrix,
        bulk_ion_atmosphere_state,
        (charged_center,),
        (resolved_anion.feature_id,),
    )
    resistance_matrix = (
        local_resistance_matrix
        + binding_resistance_matrix
        + atmosphere_resistance_matrix
    )
    charge_vector = np.asarray([anion_charge], dtype=float)
    charge_diffusivity_m2_s = float(
        charge_vector @ (K_B * temperature_K * np.linalg.inv(resistance_matrix)) @ charge_vector
    )
    return AnalyticTransportStatePrimitive(
        label=label,
        state_kind="FREE_ANION",
        anion_feature_id=resolved_anion.feature_id,
        source_salt_name=resolved_anion.source_salt_name,
        concentration_mol_m3=float(concentration_mol_m3),
        charge_vector=(float(anion_charge),),
        local_resistance_matrix_kg_s=local_resistance_matrix,
        binding_resistance_matrix_kg_s=binding_resistance_matrix,
        atmosphere_resistance_matrix_kg_s=atmosphere_resistance_matrix,
        resistance_matrix_kg_s=resistance_matrix,
        charge_diffusivity_m2_s=charge_diffusivity_m2_s,
        current_relaxation_length_m=anion_radius_A * ANGSTROM_TO_M,
        standard_free_energy_J_mol=0.0,
    )


def _paired_center_state(
    template: _MassActionTemplate,
    concentration_M: float,
    li_diffusion_m2_s: float,
    anion_diffusion_m2_s: float,
    resolved_anion: _ResolvedAnion,
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    binding_resistance_enabled: bool,
    atmosphere_resistance_enabled: bool,
    temperature_K: float,
) -> AnalyticTransportStatePrimitive:
    context = f"salt.{resolved_anion.source_salt_name}"
    anion_charge = _required_float(resolved_anion.salt_record, "anion_charge", context)
    ion_pair_binding_kj_mol = _required_float(
        resolved_anion.salt_record,
        "ion_pair_binding_kj_mol",
        context,
    )
    ligand_field_asymmetry = _required_float(
        resolved_anion.salt_record,
        "ligand_field_asymmetry",
        context,
    )
    cation_radius_A = _required_float(resolved_anion.salt_record, "cation_radius", context)
    anion_radius_A = _required_float(resolved_anion.salt_record, "anion_radius", context)
    li_collective_diffusion_m2_s = li_diffusion_m2_s / float(template.li_count)
    anion_collective_diffusion_m2_s = anion_diffusion_m2_s / float(template.anion_count)
    local_resistance_matrix = np.diag(
        [
            K_B * temperature_K / li_collective_diffusion_m2_s,
            K_B * temperature_K / anion_collective_diffusion_m2_s,
        ]
    )
    pair_distance_factor = (
        PAIR_STATE_DISTANCE_FACTOR_CIP
        if template.state_kind == "CIP"
        else PAIR_STATE_DISTANCE_FACTOR_SSIP
    )
    basin_length_m = pair_distance_factor * (cation_radius_A + anion_radius_A) * ANGSTROM_TO_M
    relative_diffusion_m2_s = li_collective_diffusion_m2_s + anion_collective_diffusion_m2_s
    diffusion_time_s = basin_length_m * basin_length_m / relative_diffusion_m2_s
    barrier_factor = math.exp(
        (ion_pair_binding_kj_mol * KJ_TO_J) / (R * temperature_K)
    )
    constraint_lifetime_s = diffusion_time_s * barrier_factor
    constraint_vector = np.asarray([NORMALIZED_PROBABILITY_SUM, -NORMALIZED_PROBABILITY_SUM])
    paired_binding_resistance_matrix = (
        K_B
        * temperature_K
        * constraint_lifetime_s
        / (basin_length_m * basin_length_m)
        * np.outer(constraint_vector, constraint_vector)
    )
    if binding_resistance_enabled:
        binding_resistance_matrix = paired_binding_resistance_matrix
    else:
        binding_resistance_matrix = np.zeros_like(local_resistance_matrix)
    charged_centers = (
        ChargedCenter(
            label=f"{template.label}_li_center",
            charge=float(template.li_count),
            hydrodynamic_radius_m=cation_radius_A * ANGSTROM_TO_M,
            shape_factor=NORMALIZED_PROBABILITY_SUM,
            local_diffusion_m2_s=li_collective_diffusion_m2_s,
            relative_position_m=(0.0, 0.0, 0.0),
            charge_cloud_radius_available=False,
            charge_cloud_radius_A=0.0,
            charge_cloud_source=NOT_APPLICABLE_FEATURE,
            charge_cloud_site_count=0,
        ),
        ChargedCenter(
            label=f"{template.label}_anion_center",
            charge=float(template.anion_count) * anion_charge,
            hydrodynamic_radius_m=anion_radius_A * ANGSTROM_TO_M,
            shape_factor=NORMALIZED_PROBABILITY_SUM + ligand_field_asymmetry,
            local_diffusion_m2_s=anion_collective_diffusion_m2_s,
            relative_position_m=(basin_length_m, 0.0, 0.0),
            charge_cloud_radius_available=False,
            charge_cloud_radius_A=0.0,
            charge_cloud_source=NOT_APPLICABLE_FEATURE,
            charge_cloud_site_count=0,
        ),
    )
    atmosphere_resistance_matrix = _state_atmosphere_matrix_for_mode(
        atmosphere_resistance_enabled,
        local_resistance_matrix,
        bulk_ion_atmosphere_state,
        charged_centers,
        (LITHIUM_CARRIER_LABEL, resolved_anion.feature_id),
    )
    resistance_matrix = (
        local_resistance_matrix
        + binding_resistance_matrix
        + atmosphere_resistance_matrix
    )
    charge_vector_array = np.asarray(
        [float(template.li_count), float(template.anion_count) * anion_charge],
        dtype=float,
    )
    charge_diffusivity_m2_s = float(
        charge_vector_array @ (K_B * temperature_K * np.linalg.inv(resistance_matrix)) @ charge_vector_array
    )
    return AnalyticTransportStatePrimitive(
        label=template.label,
        state_kind=template.state_kind,
        anion_feature_id=resolved_anion.feature_id,
        source_salt_name=resolved_anion.source_salt_name,
        concentration_mol_m3=float(concentration_M * MOLARITY_TO_MOL_M3),
        charge_vector=tuple(float(value) for value in charge_vector_array),
        local_resistance_matrix_kg_s=local_resistance_matrix,
        binding_resistance_matrix_kg_s=binding_resistance_matrix,
        atmosphere_resistance_matrix_kg_s=atmosphere_resistance_matrix,
        resistance_matrix_kg_s=resistance_matrix,
        charge_diffusivity_m2_s=charge_diffusivity_m2_s,
        current_relaxation_length_m=basin_length_m,
        standard_free_energy_J_mol=-(ion_pair_binding_kj_mol * KJ_TO_J),
    )


def _state_atmosphere_matrix(
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    charged_centers: tuple[ChargedCenter, ...],
    carrier_labels_by_center: tuple[str, ...],
) -> np.ndarray:
    if len(charged_centers) != len(carrier_labels_by_center):
        raise ValueError("charged center and carrier label counts must match")
    carrier_index_by_label = {
        carrier_label: carrier_index
        for carrier_index, carrier_label in enumerate(bulk_ion_atmosphere_state.carrier_labels)
    }
    single_center_resistances_kg_s: list[float] = []
    for carrier_label in carrier_labels_by_center:
        if carrier_label not in carrier_index_by_label:
            raise ValueError(f"carrier label {carrier_label} missing from bulk ion atmosphere state")
        carrier_index = carrier_index_by_label[carrier_label]
        single_center_resistances_kg_s.append(
            float(bulk_ion_atmosphere_state.resistance_matrix_kg_s[carrier_index, carrier_index])
        )
    return state_form_factor_atmosphere_resistance_kg_s(
        charged_centers=charged_centers,
        single_center_atmosphere_resistance_kg_s=tuple(single_center_resistances_kg_s),
        kappa_inv_m=bulk_ion_atmosphere_state.kappa_inv_m,
    )


def _state_atmosphere_matrix_for_mode(
    atmosphere_resistance_enabled: bool,
    local_resistance_matrix: np.ndarray,
    bulk_ion_atmosphere_state: BulkIonAtmosphereState,
    charged_centers: tuple[ChargedCenter, ...],
    carrier_labels_by_center: tuple[str, ...],
) -> np.ndarray:
    if atmosphere_resistance_enabled:
        return _state_atmosphere_matrix(
            bulk_ion_atmosphere_state,
            charged_centers,
            carrier_labels_by_center,
        )
    return np.zeros_like(local_resistance_matrix)


def _li_solvated_radius_A(species_catalog: AnalyticMoriSpeciesCatalog) -> float:
    cation_record = _required_record(species_catalog.cations, "Li", "cation")
    return _required_float(cation_record, "solvated_radius_A", "cation.Li")


def _li_diffusion_m2_s(
    effective_viscosity_cP: float,
    temperature_K: float,
    species_catalog: AnalyticMoriSpeciesCatalog,
) -> float:
    cation_record = _required_record(species_catalog.cations, "Li", "cation")
    solvated_radius_A = _required_float(cation_record, "solvated_radius_A", "cation.Li")
    stokes_alpha = _required_float(cation_record, "stokes_einstein_alpha", "cation.Li")
    return _stokes_einstein_diffusion_m2_s(
        radius_A=solvated_radius_A,
        viscosity_cP=effective_viscosity_cP,
        stokes_alpha=stokes_alpha,
        shape_factor=1.0,
        temperature_K=temperature_K,
    )


def _anion_diffusion_m2_s(
    resolved_anion: _ResolvedAnion,
    effective_viscosity_cP: float,
    temperature_K: float,
) -> float:
    context = f"salt.{resolved_anion.source_salt_name}"
    anion_radius_A = _required_float(resolved_anion.salt_record, "anion_radius", context)
    stokes_alpha = _required_float(
        resolved_anion.salt_record,
        "stokes_einstein_alpha_anion",
        context,
    )
    ligand_field_asymmetry = _required_float(
        resolved_anion.salt_record,
        "ligand_field_asymmetry",
        context,
    )
    return _stokes_einstein_diffusion_m2_s(
        radius_A=anion_radius_A,
        viscosity_cP=effective_viscosity_cP,
        stokes_alpha=stokes_alpha,
        shape_factor=NORMALIZED_PROBABILITY_SUM + ligand_field_asymmetry,
        temperature_K=temperature_K,
    )


def _stokes_einstein_diffusion_m2_s(
    radius_A: float,
    viscosity_cP: float,
    stokes_alpha: float,
    shape_factor: float,
    temperature_K: float,
) -> float:
    _positive_float(radius_A, "radius_A")
    _positive_float(viscosity_cP, "viscosity_cP")
    _positive_float(shape_factor, "shape_factor")
    if stokes_alpha <= 0.0 or stokes_alpha > 1.0:
        raise ValueError("stokes_alpha must be in (0, 1]")
    viscosity_pa_s = viscosity_cP * CP_TO_PA_S
    reference_viscosity_pa_s = CP_TO_PA_S
    microviscosity_pa_s = (
        (viscosity_pa_s ** stokes_alpha)
        * (reference_viscosity_pa_s ** (NORMALIZED_PROBABILITY_SUM - stokes_alpha))
    )
    hydrodynamic_radius_m = radius_A * ANGSTROM_TO_M * shape_factor
    return float(
        K_B
        * temperature_K
        / (STOKES_SPHERE_FACTOR * math.pi * microviscosity_pa_s * hydrodynamic_radius_m)
    )


def _direct_resistance_mori_input(
    transport_states: tuple[AnalyticTransportStatePrimitive, ...],
    temperature_K: float,
) -> ProjectedMoriConductivityInput:
    direct_blocks: list[np.ndarray] = []
    current_blocks: list[np.ndarray] = []
    for transport_state in transport_states:
        direct_blocks.append(transport_state.resistance_matrix_kg_s / (K_B * temperature_K))
        charge_vector = np.asarray(transport_state.charge_vector, dtype=float)
        current_coupling = math.sqrt(transport_state.concentration_mol_m3) * charge_vector
        current_blocks.append(np.tile(current_coupling, (AXIS_COUNT, 1)))
    direct_energy_matrix = _block_diagonal_matrix(direct_blocks)
    current_coupling_matrix = np.concatenate(current_blocks, axis=1)
    return ProjectedMoriConductivityInput(
        direct_energy_matrix=direct_energy_matrix,
        memory_self_energy_matrix=np.zeros_like(direct_energy_matrix),
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=F * F / (R * temperature_K),
    )


def _markov_additive_model_states(
    transport_states: tuple[AnalyticTransportStatePrimitive, ...],
    carrier_cage_primitives: tuple[AnalyticCarrierCagePrimitive, ...],
    carrier_cage_point_substates_enabled: bool,
    selective_carrier_cage_primitives: tuple[
        AnalyticSelectiveCarrierCagePrimitive,
        ...,
    ],
    selective_carrier_cage_enabled: bool,
    backjump_cage_primitives: tuple[AnalyticBackjumpCagePrimitive, ...],
    backjump_cage_memory_enabled: bool,
    timescale_structural_memory_primitives: tuple[
        AnalyticTimescaleStructuralMemoryPrimitive,
        ...,
    ],
    timescale_structural_cage_memory_enabled: bool,
) -> tuple[_MarkovAdditiveModelState, ...]:
    active_hidden_transport_branches = sum(
        int(branch_enabled)
        for branch_enabled in (
            carrier_cage_point_substates_enabled,
            selective_carrier_cage_enabled,
            backjump_cage_memory_enabled,
            timescale_structural_cage_memory_enabled,
        )
    )
    if active_hidden_transport_branches > 1:
        raise ValueError(
            "only one finite hidden transport branch can be active per analytic run"
        )
    model_states: list[_MarkovAdditiveModelState] = []
    cage_primitive_by_carrier_label = {
        cage_primitive.carrier_label: cage_primitive
        for cage_primitive in carrier_cage_primitives
    }
    selective_cage_primitive_by_carrier_label = {
        selective_cage_primitive.carrier_label: selective_cage_primitive
        for selective_cage_primitive in selective_carrier_cage_primitives
    }
    backjump_primitive_by_carrier_label = {
        backjump_primitive.carrier_label: backjump_primitive
        for backjump_primitive in backjump_cage_primitives
    }
    timescale_primitive_by_carrier_label = {
        timescale_primitive.carrier_label: timescale_primitive
        for timescale_primitive in timescale_structural_memory_primitives
    }
    for transport_state in transport_states:
        if (
            timescale_structural_cage_memory_enabled
            and transport_state.label in timescale_primitive_by_carrier_label
        ):
            timescale_primitive = timescale_primitive_by_carrier_label[
                transport_state.label
            ]
            model_states.append(
                _MarkovAdditiveModelState(
                    label=f"{transport_state.label}:timescale_mobile",
                    parent_transport_label=transport_state.label,
                    transport_state=transport_state,
                    concentration_mol_m3=(
                        timescale_primitive.mobile_concentration_mol_m3
                    ),
                    orientation_label=NOT_APPLICABLE_FEATURE,
                    orientation_vector=(0.0, 0.0, 0.0),
                    polarization_m=(0.0, 0.0, 0.0),
                    translation_diffusion_scale=(
                        NORMALIZED_PROBABILITY_SUM
                        - timescale_primitive.atmosphere_capture_fraction
                    ),
                    cage_group_label=NOT_APPLICABLE_FEATURE,
                    cage_state_kind=NOT_APPLICABLE_FEATURE,
                    cage_exchange_rate_s_inv=0.0,
                    backjump_group_label=NOT_APPLICABLE_FEATURE,
                    backjump_state_kind=NOT_APPLICABLE_FEATURE,
                    backjump_exit_rate_s_inv=0.0,
                    backjump_probability=0.0,
                    backjump_length_m=0.0,
                    timescale_memory_group_label=transport_state.label,
                    timescale_memory_state_kind="mobile",
                    timescale_capture_rate_s_inv=(
                        timescale_primitive.k_capture_s_inv
                    ),
                    timescale_atmosphere_exit_rate_s_inv=(
                        timescale_primitive.k_atmosphere_exit_s_inv
                    ),
                    timescale_structural_capture_rate_s_inv=(
                        timescale_primitive.k_structural_capture_s_inv
                    ),
                    timescale_structural_release_rate_s_inv=(
                        timescale_primitive.k_structural_release_s_inv
                    ),
                    timescale_jump_length_m=timescale_primitive.jump_length_m,
                    chemical_conversion_enabled=True,
                )
            )
            for orientation_label, orientation_vector in zip(
                BOUND_ORIENTATION_LABELS,
                TRANSLATION_EVENT_AXES,
            ):
                model_states.append(
                    _MarkovAdditiveModelState(
                        label=f"{transport_state.label}:timescale_atmosphere:{orientation_label}",
                        parent_transport_label=transport_state.label,
                        transport_state=transport_state,
                        concentration_mol_m3=(
                            timescale_primitive.atmosphere_concentration_per_orientation_mol_m3
                        ),
                        orientation_label=orientation_label,
                        orientation_vector=orientation_vector,
                        polarization_m=(0.0, 0.0, 0.0),
                        translation_diffusion_scale=0.0,
                        cage_group_label=NOT_APPLICABLE_FEATURE,
                        cage_state_kind=NOT_APPLICABLE_FEATURE,
                        cage_exchange_rate_s_inv=0.0,
                        backjump_group_label=NOT_APPLICABLE_FEATURE,
                        backjump_state_kind=NOT_APPLICABLE_FEATURE,
                        backjump_exit_rate_s_inv=0.0,
                        backjump_probability=0.0,
                        backjump_length_m=0.0,
                        timescale_memory_group_label=transport_state.label,
                        timescale_memory_state_kind="atmosphere",
                        timescale_capture_rate_s_inv=(
                            timescale_primitive.k_capture_s_inv
                        ),
                        timescale_atmosphere_exit_rate_s_inv=(
                            timescale_primitive.k_atmosphere_exit_s_inv
                        ),
                        timescale_structural_capture_rate_s_inv=(
                            timescale_primitive.k_structural_capture_s_inv
                        ),
                        timescale_structural_release_rate_s_inv=(
                            timescale_primitive.k_structural_release_s_inv
                        ),
                        timescale_jump_length_m=timescale_primitive.jump_length_m,
                        chemical_conversion_enabled=False,
                    )
                )
                if (
                    timescale_primitive.structural_cage_concentration_per_orientation_mol_m3
                    > 0.0
                ):
                    model_states.append(
                        _MarkovAdditiveModelState(
                            label=f"{transport_state.label}:timescale_structural:{orientation_label}",
                            parent_transport_label=transport_state.label,
                            transport_state=transport_state,
                            concentration_mol_m3=(
                                timescale_primitive.structural_cage_concentration_per_orientation_mol_m3
                            ),
                            orientation_label=orientation_label,
                            orientation_vector=orientation_vector,
                            polarization_m=(0.0, 0.0, 0.0),
                            translation_diffusion_scale=0.0,
                            cage_group_label=NOT_APPLICABLE_FEATURE,
                            cage_state_kind=NOT_APPLICABLE_FEATURE,
                            cage_exchange_rate_s_inv=0.0,
                            backjump_group_label=NOT_APPLICABLE_FEATURE,
                            backjump_state_kind=NOT_APPLICABLE_FEATURE,
                            backjump_exit_rate_s_inv=0.0,
                            backjump_probability=0.0,
                            backjump_length_m=0.0,
                            timescale_memory_group_label=transport_state.label,
                            timescale_memory_state_kind="structural",
                            timescale_capture_rate_s_inv=(
                                timescale_primitive.k_capture_s_inv
                            ),
                            timescale_atmosphere_exit_rate_s_inv=(
                                timescale_primitive.k_atmosphere_exit_s_inv
                            ),
                            timescale_structural_capture_rate_s_inv=(
                                timescale_primitive.k_structural_capture_s_inv
                            ),
                            timescale_structural_release_rate_s_inv=(
                                timescale_primitive.k_structural_release_s_inv
                            ),
                            timescale_jump_length_m=timescale_primitive.jump_length_m,
                            chemical_conversion_enabled=False,
                        )
                    )
        elif _state_is_oriented(transport_state):
            orientation_count = len(TRANSLATION_EVENT_AXES)
            for orientation_label, orientation_vector in zip(
                BOUND_ORIENTATION_LABELS,
                TRANSLATION_EVENT_AXES,
            ):
                model_states.append(
                    _MarkovAdditiveModelState(
                        label=f"{transport_state.label}:{orientation_label}",
                        parent_transport_label=transport_state.label,
                        transport_state=transport_state,
                        concentration_mol_m3=(
                            transport_state.concentration_mol_m3 / orientation_count
                        ),
                        orientation_label=orientation_label,
                        orientation_vector=orientation_vector,
                        polarization_m=_state_polarization_m(
                            transport_state,
                            orientation_vector,
                        ),
                        translation_diffusion_scale=NORMALIZED_PROBABILITY_SUM,
                        cage_group_label=NOT_APPLICABLE_FEATURE,
                        cage_state_kind=NOT_APPLICABLE_FEATURE,
                        cage_exchange_rate_s_inv=0.0,
                        backjump_group_label=NOT_APPLICABLE_FEATURE,
                        backjump_state_kind=NOT_APPLICABLE_FEATURE,
                        backjump_exit_rate_s_inv=0.0,
                        backjump_probability=0.0,
                        backjump_length_m=0.0,
                        timescale_memory_group_label=NOT_APPLICABLE_FEATURE,
                        timescale_memory_state_kind=NOT_APPLICABLE_FEATURE,
                        timescale_capture_rate_s_inv=0.0,
                        timescale_atmosphere_exit_rate_s_inv=0.0,
                        timescale_structural_capture_rate_s_inv=0.0,
                        timescale_structural_release_rate_s_inv=0.0,
                        timescale_jump_length_m=0.0,
                        chemical_conversion_enabled=True,
                    )
                )
        elif (
            backjump_cage_memory_enabled
            and transport_state.label in backjump_primitive_by_carrier_label
            and backjump_primitive_by_carrier_label[transport_state.label].point_active
        ):
            backjump_primitive = backjump_primitive_by_carrier_label[transport_state.label]
            mobile_concentration_mol_m3 = (
                transport_state.concentration_mol_m3
                * (
                    NORMALIZED_PROBABILITY_SUM
                    - backjump_primitive.cage_occupancy_fraction
                )
            )
            mobile_translation_diffusion_scale = (
                backjump_primitive.ordinary_translation_fraction
                / (
                    NORMALIZED_PROBABILITY_SUM
                    - backjump_primitive.cage_occupancy_fraction
                )
            )
            if mobile_translation_diffusion_scale <= 0.0:
                raise ValueError(f"{transport_state.label} backjump mobile scale is invalid")
            model_states.append(
                _MarkovAdditiveModelState(
                    label=f"{transport_state.label}:backjump_mobile",
                    parent_transport_label=transport_state.label,
                    transport_state=transport_state,
                    concentration_mol_m3=mobile_concentration_mol_m3,
                    orientation_label=NOT_APPLICABLE_FEATURE,
                    orientation_vector=(0.0, 0.0, 0.0),
                    polarization_m=(0.0, 0.0, 0.0),
                    translation_diffusion_scale=float(mobile_translation_diffusion_scale),
                    cage_group_label=NOT_APPLICABLE_FEATURE,
                    cage_state_kind=NOT_APPLICABLE_FEATURE,
                    cage_exchange_rate_s_inv=0.0,
                    backjump_group_label=transport_state.label,
                    backjump_state_kind="mobile",
                    backjump_exit_rate_s_inv=backjump_primitive.exit_rate_s_inv,
                    backjump_probability=backjump_primitive.backjump_probability,
                    backjump_length_m=backjump_primitive.jump_length_m,
                    timescale_memory_group_label=NOT_APPLICABLE_FEATURE,
                    timescale_memory_state_kind=NOT_APPLICABLE_FEATURE,
                    timescale_capture_rate_s_inv=0.0,
                    timescale_atmosphere_exit_rate_s_inv=0.0,
                    timescale_structural_capture_rate_s_inv=0.0,
                    timescale_structural_release_rate_s_inv=0.0,
                    timescale_jump_length_m=0.0,
                    chemical_conversion_enabled=True,
                )
            )
            orientation_count = len(TRANSLATION_EVENT_AXES)
            for orientation_label, orientation_vector in zip(
                BOUND_ORIENTATION_LABELS,
                TRANSLATION_EVENT_AXES,
            ):
                model_states.append(
                    _MarkovAdditiveModelState(
                        label=f"{transport_state.label}:backjump_cage:{orientation_label}",
                        parent_transport_label=f"{transport_state.label}:backjump_cage:{orientation_label}",
                        transport_state=transport_state,
                        concentration_mol_m3=(
                            transport_state.concentration_mol_m3
                            * backjump_primitive.cage_occupancy_fraction
                            / orientation_count
                        ),
                        orientation_label=orientation_label,
                        orientation_vector=orientation_vector,
                        polarization_m=(0.0, 0.0, 0.0),
                        translation_diffusion_scale=0.0,
                        cage_group_label=NOT_APPLICABLE_FEATURE,
                        cage_state_kind=NOT_APPLICABLE_FEATURE,
                        cage_exchange_rate_s_inv=0.0,
                        backjump_group_label=transport_state.label,
                        backjump_state_kind="cage",
                        backjump_exit_rate_s_inv=backjump_primitive.exit_rate_s_inv,
                        backjump_probability=backjump_primitive.backjump_probability,
                        backjump_length_m=backjump_primitive.jump_length_m,
                        timescale_memory_group_label=NOT_APPLICABLE_FEATURE,
                        timescale_memory_state_kind=NOT_APPLICABLE_FEATURE,
                        timescale_capture_rate_s_inv=0.0,
                        timescale_atmosphere_exit_rate_s_inv=0.0,
                        timescale_structural_capture_rate_s_inv=0.0,
                        timescale_structural_release_rate_s_inv=0.0,
                        timescale_jump_length_m=0.0,
                        chemical_conversion_enabled=False,
                    )
                )
        elif (
            carrier_cage_point_substates_enabled
            and transport_state.label in cage_primitive_by_carrier_label
        ):
            cage_primitive = cage_primitive_by_carrier_label[transport_state.label]
            model_states.append(
                _MarkovAdditiveModelState(
                    label=f"{transport_state.label}:mobile",
                    parent_transport_label=f"{transport_state.label}:mobile",
                    transport_state=transport_state,
                    concentration_mol_m3=(
                        transport_state.concentration_mol_m3
                        * cage_primitive.mobile_fraction
                    ),
                    orientation_label=NOT_APPLICABLE_FEATURE,
                    orientation_vector=(0.0, 0.0, 0.0),
                    polarization_m=(0.0, 0.0, 0.0),
                    translation_diffusion_scale=NORMALIZED_PROBABILITY_SUM,
                    cage_group_label=transport_state.label,
                    cage_state_kind="mobile",
                    cage_exchange_rate_s_inv=cage_primitive.exchange_rate_s_inv,
                    backjump_group_label=NOT_APPLICABLE_FEATURE,
                    backjump_state_kind=NOT_APPLICABLE_FEATURE,
                    backjump_exit_rate_s_inv=0.0,
                    backjump_probability=0.0,
                    backjump_length_m=0.0,
                    timescale_memory_group_label=NOT_APPLICABLE_FEATURE,
                    timescale_memory_state_kind=NOT_APPLICABLE_FEATURE,
                    timescale_capture_rate_s_inv=0.0,
                    timescale_atmosphere_exit_rate_s_inv=0.0,
                    timescale_structural_capture_rate_s_inv=0.0,
                    timescale_structural_release_rate_s_inv=0.0,
                    timescale_jump_length_m=0.0,
                    chemical_conversion_enabled=True,
                )
            )
            model_states.append(
                _MarkovAdditiveModelState(
                    label=f"{transport_state.label}:caged",
                    parent_transport_label=f"{transport_state.label}:caged",
                    transport_state=transport_state,
                    concentration_mol_m3=(
                        transport_state.concentration_mol_m3
                        * cage_primitive.caged_fraction
                    ),
                    orientation_label=NOT_APPLICABLE_FEATURE,
                    orientation_vector=(0.0, 0.0, 0.0),
                    polarization_m=(0.0, 0.0, 0.0),
                    translation_diffusion_scale=cage_primitive.caged_diffusion_scale,
                    cage_group_label=transport_state.label,
                    cage_state_kind="caged",
                    cage_exchange_rate_s_inv=cage_primitive.exchange_rate_s_inv,
                    backjump_group_label=NOT_APPLICABLE_FEATURE,
                    backjump_state_kind=NOT_APPLICABLE_FEATURE,
                    backjump_exit_rate_s_inv=0.0,
                    backjump_probability=0.0,
                    backjump_length_m=0.0,
                    timescale_memory_group_label=NOT_APPLICABLE_FEATURE,
                    timescale_memory_state_kind=NOT_APPLICABLE_FEATURE,
                    timescale_capture_rate_s_inv=0.0,
                    timescale_atmosphere_exit_rate_s_inv=0.0,
                    timescale_structural_capture_rate_s_inv=0.0,
                    timescale_structural_release_rate_s_inv=0.0,
                    timescale_jump_length_m=0.0,
                    chemical_conversion_enabled=True,
                )
            )
        elif (
            selective_carrier_cage_enabled
            and transport_state.label in selective_cage_primitive_by_carrier_label
            and selective_cage_primitive_by_carrier_label[
                transport_state.label
            ].caged_fraction > 0.0
        ):
            selective_cage_primitive = selective_cage_primitive_by_carrier_label[
                transport_state.label
            ]
            if transport_state.label not in cage_primitive_by_carrier_label:
                raise ValueError(
                    f"missing cage exchange primitive for {transport_state.label}"
                )
            cage_exchange_rate_s_inv = cage_primitive_by_carrier_label[
                transport_state.label
            ].exchange_rate_s_inv
            mobile_fraction = (
                NORMALIZED_PROBABILITY_SUM - selective_cage_primitive.caged_fraction
            )
            if mobile_fraction <= 0.0:
                raise ValueError(
                    f"{transport_state.label} selective mobile fraction is invalid"
                )
            model_states.append(
                _MarkovAdditiveModelState(
                    label=f"{transport_state.label}:selective_mobile",
                    parent_transport_label=transport_state.label,
                    transport_state=transport_state,
                    concentration_mol_m3=(
                        transport_state.concentration_mol_m3
                        * mobile_fraction
                    ),
                    orientation_label=NOT_APPLICABLE_FEATURE,
                    orientation_vector=(0.0, 0.0, 0.0),
                    polarization_m=(0.0, 0.0, 0.0),
                    translation_diffusion_scale=NORMALIZED_PROBABILITY_SUM,
                    cage_group_label=transport_state.label,
                    cage_state_kind="mobile",
                    cage_exchange_rate_s_inv=cage_exchange_rate_s_inv,
                    backjump_group_label=NOT_APPLICABLE_FEATURE,
                    backjump_state_kind=NOT_APPLICABLE_FEATURE,
                    backjump_exit_rate_s_inv=0.0,
                    backjump_probability=0.0,
                    backjump_length_m=0.0,
                    timescale_memory_group_label=NOT_APPLICABLE_FEATURE,
                    timescale_memory_state_kind=NOT_APPLICABLE_FEATURE,
                    timescale_capture_rate_s_inv=0.0,
                    timescale_atmosphere_exit_rate_s_inv=0.0,
                    timescale_structural_capture_rate_s_inv=0.0,
                    timescale_structural_release_rate_s_inv=0.0,
                    timescale_jump_length_m=0.0,
                    chemical_conversion_enabled=True,
                )
            )
            model_states.append(
                _MarkovAdditiveModelState(
                    label=f"{transport_state.label}:selective_caged",
                    parent_transport_label=transport_state.label,
                    transport_state=transport_state,
                    concentration_mol_m3=(
                        transport_state.concentration_mol_m3
                        * selective_cage_primitive.caged_fraction
                    ),
                    orientation_label=NOT_APPLICABLE_FEATURE,
                    orientation_vector=(0.0, 0.0, 0.0),
                    polarization_m=(0.0, 0.0, 0.0),
                    translation_diffusion_scale=(
                        selective_cage_primitive.caged_diffusion_scale
                    ),
                    cage_group_label=transport_state.label,
                    cage_state_kind="caged",
                    cage_exchange_rate_s_inv=cage_exchange_rate_s_inv,
                    backjump_group_label=NOT_APPLICABLE_FEATURE,
                    backjump_state_kind=NOT_APPLICABLE_FEATURE,
                    backjump_exit_rate_s_inv=0.0,
                    backjump_probability=0.0,
                    backjump_length_m=0.0,
                    timescale_memory_group_label=NOT_APPLICABLE_FEATURE,
                    timescale_memory_state_kind=NOT_APPLICABLE_FEATURE,
                    timescale_capture_rate_s_inv=0.0,
                    timescale_atmosphere_exit_rate_s_inv=0.0,
                    timescale_structural_capture_rate_s_inv=0.0,
                    timescale_structural_release_rate_s_inv=0.0,
                    timescale_jump_length_m=0.0,
                    chemical_conversion_enabled=True,
                )
            )
        else:
            model_states.append(
                _MarkovAdditiveModelState(
                    label=transport_state.label,
                    parent_transport_label=transport_state.label,
                    transport_state=transport_state,
                    concentration_mol_m3=transport_state.concentration_mol_m3,
                    orientation_label=NOT_APPLICABLE_FEATURE,
                    orientation_vector=(0.0, 0.0, 0.0),
                    polarization_m=(0.0, 0.0, 0.0),
                    translation_diffusion_scale=NORMALIZED_PROBABILITY_SUM,
                    cage_group_label=NOT_APPLICABLE_FEATURE,
                    cage_state_kind=NOT_APPLICABLE_FEATURE,
                    cage_exchange_rate_s_inv=0.0,
                    backjump_group_label=NOT_APPLICABLE_FEATURE,
                    backjump_state_kind=NOT_APPLICABLE_FEATURE,
                    backjump_exit_rate_s_inv=0.0,
                    backjump_probability=0.0,
                    backjump_length_m=0.0,
                    timescale_memory_group_label=NOT_APPLICABLE_FEATURE,
                    timescale_memory_state_kind=NOT_APPLICABLE_FEATURE,
                    timescale_capture_rate_s_inv=0.0,
                    timescale_atmosphere_exit_rate_s_inv=0.0,
                    timescale_structural_capture_rate_s_inv=0.0,
                    timescale_structural_release_rate_s_inv=0.0,
                    timescale_jump_length_m=0.0,
                    chemical_conversion_enabled=True,
                )
            )
    return tuple(model_states)


def _markov_additive_events_from_model_states(
    model_states: tuple[_MarkovAdditiveModelState, ...],
    temperature_K: float,
) -> tuple[MarkovAdditiveEvent, ...]:
    events: list[MarkovAdditiveEvent] = []
    state_index_by_label = {
        model_state.label: state_index
        for state_index, model_state in enumerate(model_states)
    }
    for state_index, model_state in enumerate(model_states):
        events.extend(
            _translation_events_for_state(
                state_index,
                model_state,
                temperature_K,
            )
        )
    events.extend(
        _orientation_relaxation_events(model_states, state_index_by_label, temperature_K)
    )
    events.extend(_carrier_cage_exchange_events(model_states, state_index_by_label))
    events.extend(_backjump_cage_events(model_states, state_index_by_label))
    events.extend(
        _timescale_structural_memory_events(model_states, state_index_by_label)
    )
    events.extend(
        _chemical_conversion_events(
            model_states,
            state_index_by_label,
            temperature_K,
        )
    )
    if len(events) == 0:
        raise ValueError("analytic Markov-additive model produced no events")
    return tuple(events)


def _translation_events_for_state(
    state_index: int,
    model_state: _MarkovAdditiveModelState,
    temperature_K: float,
) -> tuple[MarkovAdditiveEvent, ...]:
    transport_state = model_state.transport_state
    if model_state.timescale_memory_state_kind == "mobile":
        hop_length_m = _positive_float(
            model_state.timescale_jump_length_m,
            f"{model_state.label}.timescale_jump_length_m",
        )
        center_of_mass_diffusion_m2_s = _local_center_of_mass_diffusion_m2_s(
            transport_state,
            temperature_K,
        )
    else:
        hop_length_m = _positive_float(
            transport_state.current_relaxation_length_m,
            f"{transport_state.label}.current_relaxation_length_m",
        )
        center_of_mass_diffusion_m2_s = _center_of_mass_diffusion_m2_s(
            transport_state,
            temperature_K,
        )
    hop_rate_s_inv = (
        model_state.translation_diffusion_scale
        * center_of_mass_diffusion_m2_s
        / (hop_length_m * hop_length_m)
    )
    if hop_rate_s_inv < 0.0:
        raise ValueError(f"{model_state.label} translation hop rate is negative")
    if hop_rate_s_inv == 0.0:
        return ()
    net_charge_number = float(math.fsum(transport_state.charge_vector))
    events: list[MarkovAdditiveEvent] = []
    for axis_index, axis_vector in enumerate(TRANSLATION_EVENT_AXES):
        charge_displacement_m = tuple(
            float(net_charge_number * hop_length_m * axis_component)
            for axis_component in axis_vector
        )
        events.append(
            MarkovAdditiveEvent(
                from_state_index=state_index,
                to_state_index=state_index,
                rate_s_inv=hop_rate_s_inv,
                charge_displacement_m=charge_displacement_m,
                label=f"{transport_state.label}:translation:{axis_index}",
                family_label=_translation_event_family_label(transport_state),
            )
        )
    return tuple(events)


def _translation_event_family_label(
    transport_state: AnalyticTransportStatePrimitive,
) -> str:
    if transport_state.state_kind == STATE_KIND_FREE_LI:
        return EVENT_FAMILY_ORDINARY_FREE_LI_TRANSLATION
    if transport_state.state_kind == STATE_KIND_FREE_ANION:
        return EVENT_FAMILY_ORDINARY_FREE_ANION_TRANSLATION
    return EVENT_FAMILY_BOUND_STATE_TRANSLATION


def _carrier_cage_exchange_events(
    model_states: tuple[_MarkovAdditiveModelState, ...],
    state_index_by_label: Mapping[str, int],
) -> tuple[MarkovAdditiveEvent, ...]:
    states_by_cage_group: dict[str, list[_MarkovAdditiveModelState]] = {}
    for model_state in model_states:
        if model_state.cage_group_label == NOT_APPLICABLE_FEATURE:
            continue
        if model_state.cage_group_label not in states_by_cage_group:
            states_by_cage_group[model_state.cage_group_label] = []
        states_by_cage_group[model_state.cage_group_label].append(model_state)
    events: list[MarkovAdditiveEvent] = []
    for cage_group_label, cage_states in states_by_cage_group.items():
        mobile_states = tuple(
            cage_state
            for cage_state in cage_states
            if cage_state.cage_state_kind == "mobile"
        )
        caged_states = tuple(
            cage_state
            for cage_state in cage_states
            if cage_state.cage_state_kind == "caged"
        )
        if len(mobile_states) != 1 or len(caged_states) != 1:
            raise ValueError(f"{cage_group_label} must have one mobile and one caged state")
        mobile_state = mobile_states[0]
        caged_state = caged_states[0]
        conductance_mol_m3_s = (
            mobile_state.cage_exchange_rate_s_inv
            * math.sqrt(mobile_state.concentration_mol_m3 * caged_state.concentration_mol_m3)
        )
        mobile_to_caged_rate_s_inv = (
            conductance_mol_m3_s / mobile_state.concentration_mol_m3
        )
        caged_to_mobile_rate_s_inv = (
            conductance_mol_m3_s / caged_state.concentration_mol_m3
        )
        zero_displacement_m = (0.0, 0.0, 0.0)
        events.append(
            MarkovAdditiveEvent(
                from_state_index=state_index_by_label[mobile_state.label],
                to_state_index=state_index_by_label[caged_state.label],
                rate_s_inv=mobile_to_caged_rate_s_inv,
                charge_displacement_m=zero_displacement_m,
                label=f"{mobile_state.label}->{caged_state.label}:cage_exchange",
                family_label=EVENT_FAMILY_STATIC_CARRIER_CAGE_EXCHANGE,
            )
        )
        events.append(
            MarkovAdditiveEvent(
                from_state_index=state_index_by_label[caged_state.label],
                to_state_index=state_index_by_label[mobile_state.label],
                rate_s_inv=caged_to_mobile_rate_s_inv,
                charge_displacement_m=zero_displacement_m,
                label=f"{caged_state.label}->{mobile_state.label}:cage_exchange",
                family_label=EVENT_FAMILY_STATIC_CARRIER_CAGE_EXCHANGE,
            )
        )
    return tuple(events)


def _backjump_cage_events(
    model_states: tuple[_MarkovAdditiveModelState, ...],
    state_index_by_label: Mapping[str, int],
) -> tuple[MarkovAdditiveEvent, ...]:
    states_by_backjump_group: dict[str, list[_MarkovAdditiveModelState]] = {}
    for model_state in model_states:
        if model_state.backjump_group_label == NOT_APPLICABLE_FEATURE:
            continue
        if model_state.backjump_group_label not in states_by_backjump_group:
            states_by_backjump_group[model_state.backjump_group_label] = []
        states_by_backjump_group[model_state.backjump_group_label].append(model_state)
    events: list[MarkovAdditiveEvent] = []
    for backjump_group_label, backjump_states in states_by_backjump_group.items():
        mobile_states = tuple(
            backjump_state
            for backjump_state in backjump_states
            if backjump_state.backjump_state_kind == "mobile"
        )
        cage_states = tuple(
            backjump_state
            for backjump_state in backjump_states
            if backjump_state.backjump_state_kind == "cage"
        )
        if len(mobile_states) != 1:
            raise ValueError(f"{backjump_group_label} must have one backjump mobile state")
        if len(cage_states) != len(TRANSLATION_EVENT_AXES):
            raise ValueError(
                f"{backjump_group_label} must have one backjump cage state per axis direction"
            )
        mobile_state = mobile_states[0]
        total_cage_concentration_mol_m3 = math.fsum(
            cage_state.concentration_mol_m3 for cage_state in cage_states
        )
        if total_cage_concentration_mol_m3 <= 0.0:
            raise ValueError(f"{backjump_group_label} cage concentration must be positive")
        capture_rate_s_inv = (
            total_cage_concentration_mol_m3
            * mobile_state.backjump_exit_rate_s_inv
            / (
                len(TRANSLATION_EVENT_AXES)
                * mobile_state.concentration_mol_m3
            )
        )
        if capture_rate_s_inv <= 0.0:
            raise ValueError(f"{backjump_group_label} capture rate must be positive")
        for cage_state in cage_states:
            capture_displacement_m = _backjump_charge_displacement_m(
                mobile_state,
                cage_state.orientation_vector,
                mobile_state.backjump_length_m,
            )
            backjump_displacement_m = tuple(
                -axis_displacement_m for axis_displacement_m in capture_displacement_m
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index_by_label[mobile_state.label],
                    to_state_index=state_index_by_label[cage_state.label],
                    rate_s_inv=capture_rate_s_inv,
                    charge_displacement_m=capture_displacement_m,
                    label=f"{mobile_state.label}->{cage_state.label}:backjump_capture",
                    family_label=EVENT_FAMILY_LI_BACKJUMP_CAGE,
                )
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index_by_label[cage_state.label],
                    to_state_index=state_index_by_label[mobile_state.label],
                    rate_s_inv=cage_state.backjump_exit_rate_s_inv,
                    charge_displacement_m=backjump_displacement_m,
                    label=f"{cage_state.label}->{mobile_state.label}:backjump_release",
                    family_label=EVENT_FAMILY_LI_BACKJUMP_CAGE,
                )
            )
    return tuple(events)


def _backjump_charge_displacement_m(
    mobile_state: _MarkovAdditiveModelState,
    orientation_vector: tuple[float, float, float],
    jump_length_m: float,
) -> tuple[float, float, float]:
    net_charge_number = float(math.fsum(mobile_state.transport_state.charge_vector))
    return tuple(
        float(net_charge_number * jump_length_m * axis_component)
        for axis_component in orientation_vector
    )


def _timescale_structural_memory_events(
    model_states: tuple[_MarkovAdditiveModelState, ...],
    state_index_by_label: Mapping[str, int],
) -> tuple[MarkovAdditiveEvent, ...]:
    states_by_timescale_group: dict[str, list[_MarkovAdditiveModelState]] = {}
    for model_state in model_states:
        if model_state.timescale_memory_group_label == NOT_APPLICABLE_FEATURE:
            continue
        if model_state.timescale_memory_group_label not in states_by_timescale_group:
            states_by_timescale_group[model_state.timescale_memory_group_label] = []
        states_by_timescale_group[
            model_state.timescale_memory_group_label
        ].append(model_state)
    events: list[MarkovAdditiveEvent] = []
    for timescale_group_label, timescale_states in states_by_timescale_group.items():
        mobile_states = tuple(
            timescale_state
            for timescale_state in timescale_states
            if timescale_state.timescale_memory_state_kind == "mobile"
        )
        atmosphere_states = tuple(
            timescale_state
            for timescale_state in timescale_states
            if timescale_state.timescale_memory_state_kind == "atmosphere"
        )
        structural_states = tuple(
            timescale_state
            for timescale_state in timescale_states
            if timescale_state.timescale_memory_state_kind == "structural"
        )
        if len(mobile_states) != 1:
            raise ValueError(
                f"{timescale_group_label} must have one timescale mobile state"
            )
        if len(atmosphere_states) != len(TRANSLATION_EVENT_AXES):
            raise ValueError(
                f"{timescale_group_label} must have one timescale atmosphere "
                "state per axis direction"
            )
        mobile_state = mobile_states[0]
        if (
            structural_states
            and len(structural_states) != len(TRANSLATION_EVENT_AXES)
        ):
            raise ValueError(
                f"{timescale_group_label} structural states must cover every "
                "axis direction"
            )
        structural_state_by_orientation = {
            structural_state.orientation_label: structural_state
            for structural_state in structural_states
        }
        for atmosphere_state in atmosphere_states:
            capture_displacement_m = _timescale_charge_displacement_m(
                mobile_state,
                atmosphere_state.orientation_vector,
                mobile_state.timescale_jump_length_m,
            )
            backtracking_displacement_m = tuple(
                -axis_displacement_m for axis_displacement_m in capture_displacement_m
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index_by_label[mobile_state.label],
                    to_state_index=state_index_by_label[atmosphere_state.label],
                    rate_s_inv=mobile_state.timescale_capture_rate_s_inv,
                    charge_displacement_m=capture_displacement_m,
                    label=f"{mobile_state.label}->{atmosphere_state.label}:timescale_capture",
                    family_label=EVENT_FAMILY_TIMESCALE_ATMOSPHERE_CAPTURE_BACKTRACKING,
                )
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index_by_label[atmosphere_state.label],
                    to_state_index=state_index_by_label[mobile_state.label],
                    rate_s_inv=atmosphere_state.timescale_atmosphere_exit_rate_s_inv,
                    charge_displacement_m=backtracking_displacement_m,
                    label=f"{atmosphere_state.label}->{mobile_state.label}:timescale_backtrack",
                    family_label=EVENT_FAMILY_TIMESCALE_ATMOSPHERE_CAPTURE_BACKTRACKING,
                )
            )
            if not structural_states:
                continue
            if atmosphere_state.orientation_label not in structural_state_by_orientation:
                raise ValueError(
                    f"{timescale_group_label} is missing structural state for "
                    f"{atmosphere_state.orientation_label}"
                )
            structural_state = structural_state_by_orientation[
                atmosphere_state.orientation_label
            ]
            zero_displacement_m = (0.0, 0.0, 0.0)
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index_by_label[atmosphere_state.label],
                    to_state_index=state_index_by_label[structural_state.label],
                    rate_s_inv=atmosphere_state.timescale_structural_capture_rate_s_inv,
                    charge_displacement_m=zero_displacement_m,
                    label=f"{atmosphere_state.label}->{structural_state.label}:timescale_structural_capture",
                    family_label=EVENT_FAMILY_TIMESCALE_STRUCTURAL_CAGE_EXCHANGE,
                )
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=state_index_by_label[structural_state.label],
                    to_state_index=state_index_by_label[atmosphere_state.label],
                    rate_s_inv=structural_state.timescale_structural_release_rate_s_inv,
                    charge_displacement_m=zero_displacement_m,
                    label=f"{structural_state.label}->{atmosphere_state.label}:timescale_structural_release",
                    family_label=EVENT_FAMILY_TIMESCALE_STRUCTURAL_CAGE_EXCHANGE,
                )
            )
    return tuple(events)


def _timescale_charge_displacement_m(
    mobile_state: _MarkovAdditiveModelState,
    orientation_vector: tuple[float, float, float],
    jump_length_m: float,
) -> tuple[float, float, float]:
    net_charge_number = float(math.fsum(mobile_state.transport_state.charge_vector))
    return tuple(
        float(net_charge_number * jump_length_m * axis_component)
        for axis_component in orientation_vector
    )


def _orientation_relaxation_events(
    model_states: tuple[_MarkovAdditiveModelState, ...],
    state_index_by_label: Mapping[str, int],
    temperature_K: float,
) -> tuple[MarkovAdditiveEvent, ...]:
    events: list[MarkovAdditiveEvent] = []
    conversion_model_states = tuple(
        model_state for model_state in model_states if model_state.chemical_conversion_enabled
    )
    model_states_by_parent = _model_states_by_parent_label(conversion_model_states)
    for parent_label, oriented_states in model_states_by_parent.items():
        if len(oriented_states) <= 1:
            continue
        parent_transport_state = oriented_states[0].transport_state
        if not _state_is_oriented(parent_transport_state):
            continue
        relaxation_rate_s_inv = _orientation_relaxation_rate_s_inv(
            parent_transport_state,
            temperature_K,
        )
        transition_rate_s_inv = relaxation_rate_s_inv / float(len(oriented_states) - 1)
        for source_model_state in oriented_states:
            for target_model_state in oriented_states:
                if source_model_state.label == target_model_state.label:
                    continue
                events.append(
                    MarkovAdditiveEvent(
                        from_state_index=state_index_by_label[source_model_state.label],
                        to_state_index=state_index_by_label[target_model_state.label],
                        rate_s_inv=transition_rate_s_inv,
                        charge_displacement_m=_polarization_displacement_m(
                            source_model_state,
                            target_model_state,
                        ),
                        label=f"{parent_label}:orientation:{source_model_state.orientation_label}->{target_model_state.orientation_label}",
                        family_label=EVENT_FAMILY_ORIENTATION_RELAXATION,
                    )
                )
    return tuple(events)


def _chemical_conversion_events(
    model_states: tuple[_MarkovAdditiveModelState, ...],
    state_index_by_label: Mapping[str, int],
    temperature_K: float,
) -> tuple[MarkovAdditiveEvent, ...]:
    events: list[MarkovAdditiveEvent] = []
    conversion_model_states = tuple(
        model_state for model_state in model_states if model_state.chemical_conversion_enabled
    )
    model_states_by_parent = _model_states_by_parent_label(conversion_model_states)
    parent_items = tuple(
        (parent_label, model_states_for_parent[0].transport_state)
        for parent_label, model_states_for_parent in model_states_by_parent.items()
    )
    for source_parent_index, source_parent_item in enumerate(parent_items):
        source_parent_label, source_state = source_parent_item
        for target_parent_item in parent_items[source_parent_index + 1:]:
            target_parent_label, target_state = target_parent_item
            if source_state.label == target_state.label:
                continue
            if not _states_share_conversion_family(source_state, target_state):
                continue
            source_model_states = model_states_by_parent[source_parent_label]
            target_model_states = model_states_by_parent[target_parent_label]
            events.extend(
                _reversible_conversion_events_for_parent_pair(
                    source_state,
                    target_state,
                    source_model_states,
                    target_model_states,
                    state_index_by_label,
                    temperature_K,
                )
            )
    return tuple(events)


def _reversible_conversion_events_for_parent_pair(
    source_state: AnalyticTransportStatePrimitive,
    target_state: AnalyticTransportStatePrimitive,
    source_model_states: tuple[_MarkovAdditiveModelState, ...],
    target_model_states: tuple[_MarkovAdditiveModelState, ...],
    state_index_by_label: Mapping[str, int],
    temperature_K: float,
) -> tuple[MarkovAdditiveEvent, ...]:
    conversion_pairs = _conversion_model_state_pairs(source_model_states, target_model_states)
    pair_weight_sum = math.fsum(
        math.sqrt(
            source_model_state.concentration_mol_m3
            * target_model_state.concentration_mol_m3
        )
        for source_model_state, target_model_state in conversion_pairs
    )
    if pair_weight_sum <= 0.0:
        raise ValueError("conversion pair concentration weight must be positive")
    conductance_mol_m3_s = _symmetric_conversion_conductance_mol_m3_s(
        source_state,
        target_state,
        _model_state_total_concentration_mol_m3(source_model_states),
        _model_state_total_concentration_mol_m3(target_model_states),
        temperature_K,
    )
    events: list[MarkovAdditiveEvent] = []
    for source_model_state, target_model_state in conversion_pairs:
        pair_weight = math.sqrt(
            source_model_state.concentration_mol_m3
            * target_model_state.concentration_mol_m3
        )
        pair_conductance_mol_m3_s = (
            conductance_mol_m3_s * pair_weight / pair_weight_sum
        )
        source_rate_s_inv = (
            pair_conductance_mol_m3_s / source_model_state.concentration_mol_m3
        )
        target_rate_s_inv = (
            pair_conductance_mol_m3_s / target_model_state.concentration_mol_m3
        )
        displacement_m = _polarization_displacement_m(source_model_state, target_model_state)
        reverse_displacement_m = tuple(-axis_value for axis_value in displacement_m)
        events.append(
            MarkovAdditiveEvent(
                from_state_index=state_index_by_label[source_model_state.label],
                to_state_index=state_index_by_label[target_model_state.label],
                rate_s_inv=source_rate_s_inv,
                charge_displacement_m=displacement_m,
                label=f"{source_model_state.label}->{target_model_state.label}",
                family_label=EVENT_FAMILY_CHEMICAL_INTERCONVERSION,
            )
        )
        events.append(
            MarkovAdditiveEvent(
                from_state_index=state_index_by_label[target_model_state.label],
                to_state_index=state_index_by_label[source_model_state.label],
                rate_s_inv=target_rate_s_inv,
                charge_displacement_m=reverse_displacement_m,
                label=f"{target_model_state.label}->{source_model_state.label}",
                family_label=EVENT_FAMILY_CHEMICAL_INTERCONVERSION,
            )
        )
    return tuple(events)


def _conversion_model_state_pairs(
    source_model_states: tuple[_MarkovAdditiveModelState, ...],
    target_model_states: tuple[_MarkovAdditiveModelState, ...],
) -> tuple[tuple[_MarkovAdditiveModelState, _MarkovAdditiveModelState], ...]:
    if len(source_model_states) == 1 and len(target_model_states) == 1:
        return ((source_model_states[0], target_model_states[0]),)
    if len(source_model_states) == 1:
        return tuple((source_model_states[0], target_state) for target_state in target_model_states)
    if len(target_model_states) == 1:
        return tuple((source_state, target_model_states[0]) for source_state in source_model_states)
    if not (
        _model_state_set_has_internal_orientations(source_model_states)
        and _model_state_set_has_internal_orientations(target_model_states)
    ):
        return tuple(
            (source_state, target_state)
            for source_state in source_model_states
            for target_state in target_model_states
        )
    target_by_orientation = {
        target_state.orientation_label: target_state
        for target_state in target_model_states
    }
    pairs: list[tuple[_MarkovAdditiveModelState, _MarkovAdditiveModelState]] = []
    for source_state in source_model_states:
        if source_state.orientation_label not in target_by_orientation:
            raise ValueError("oriented conversion states must share orientation labels")
        pairs.append((source_state, target_by_orientation[source_state.orientation_label]))
    return tuple(pairs)


def _model_state_set_has_internal_orientations(
    model_states: tuple[_MarkovAdditiveModelState, ...],
) -> bool:
    return any(
        model_state.orientation_label != NOT_APPLICABLE_FEATURE
        for model_state in model_states
    )


def _symmetric_conversion_conductance_mol_m3_s(
    source_state: AnalyticTransportStatePrimitive,
    target_state: AnalyticTransportStatePrimitive,
    source_concentration_mol_m3: float,
    target_concentration_mol_m3: float,
    temperature_K: float,
) -> float:
    source_rate_scale_s_inv = _center_of_mass_diffusion_m2_s(
        source_state,
        temperature_K,
    ) / (
        source_state.current_relaxation_length_m
        * source_state.current_relaxation_length_m
    )
    target_rate_scale_s_inv = _center_of_mass_diffusion_m2_s(
        target_state,
        temperature_K,
    ) / (
        target_state.current_relaxation_length_m
        * target_state.current_relaxation_length_m
    )
    barrier_J_mol = HALF_FACTOR * abs(
        target_state.standard_free_energy_J_mol - source_state.standard_free_energy_J_mol
    )
    barrier_factor = math.exp(-barrier_J_mol / (R * temperature_K))
    rate_scale_s_inv = math.sqrt(source_rate_scale_s_inv * target_rate_scale_s_inv)
    return float(
        rate_scale_s_inv
        * math.sqrt(
            _positive_float(source_concentration_mol_m3, "source_concentration_mol_m3")
            * _positive_float(target_concentration_mol_m3, "target_concentration_mol_m3")
        )
        * barrier_factor
    )


def _model_state_total_concentration_mol_m3(
    model_states: tuple[_MarkovAdditiveModelState, ...],
) -> float:
    total_concentration_mol_m3 = math.fsum(
        model_state.concentration_mol_m3 for model_state in model_states
    )
    return _positive_float(
        total_concentration_mol_m3,
        "model_state_total_concentration_mol_m3",
    )


def _states_share_conversion_family(
    source_state: AnalyticTransportStatePrimitive,
    target_state: AnalyticTransportStatePrimitive,
) -> bool:
    if source_state.label == target_state.label:
        return False
    if source_state.state_kind == STATE_KIND_FREE_LI:
        return target_state.state_kind in (
            STATE_KIND_SSIP,
            STATE_KIND_CIP,
            STATE_KIND_LI2A_PLUS,
            STATE_KIND_LIA2_MINUS,
        )
    if target_state.state_kind == STATE_KIND_FREE_LI:
        return source_state.state_kind in (
            STATE_KIND_SSIP,
            STATE_KIND_CIP,
            STATE_KIND_LI2A_PLUS,
            STATE_KIND_LIA2_MINUS,
        )
    if source_state.anion_feature_id != target_state.anion_feature_id:
        return False
    source_kind = source_state.state_kind
    target_kind = target_state.state_kind
    return (
        (source_kind == STATE_KIND_FREE_ANION and target_kind in (STATE_KIND_SSIP, STATE_KIND_CIP, STATE_KIND_LIA2_MINUS))
        or (target_kind == STATE_KIND_FREE_ANION and source_kind in (STATE_KIND_SSIP, STATE_KIND_CIP, STATE_KIND_LIA2_MINUS))
        or (source_kind == STATE_KIND_SSIP and target_kind in (STATE_KIND_CIP, STATE_KIND_LIA2_MINUS))
        or (target_kind == STATE_KIND_SSIP and source_kind in (STATE_KIND_CIP, STATE_KIND_LIA2_MINUS))
        or (source_kind == STATE_KIND_CIP and target_kind in (STATE_KIND_LI2A_PLUS, STATE_KIND_LI2A2_NEUTRAL))
        or (target_kind == STATE_KIND_CIP and source_kind in (STATE_KIND_LI2A_PLUS, STATE_KIND_LI2A2_NEUTRAL))
        or (source_kind == STATE_KIND_LI2A_PLUS and target_kind == STATE_KIND_LI2A2_NEUTRAL)
        or (target_kind == STATE_KIND_LI2A_PLUS and source_kind == STATE_KIND_LI2A2_NEUTRAL)
        or (source_kind == STATE_KIND_LIA2_MINUS and target_kind == STATE_KIND_LI2A2_NEUTRAL)
        or (target_kind == STATE_KIND_LIA2_MINUS and source_kind == STATE_KIND_LI2A2_NEUTRAL)
    )


def _model_states_by_parent_label(
    model_states: tuple[_MarkovAdditiveModelState, ...],
) -> dict[str, tuple[_MarkovAdditiveModelState, ...]]:
    grouped_model_states: dict[str, list[_MarkovAdditiveModelState]] = {}
    for model_state in model_states:
        if model_state.parent_transport_label not in grouped_model_states:
            grouped_model_states[model_state.parent_transport_label] = []
        grouped_model_states[model_state.parent_transport_label].append(model_state)
    return {
        parent_label: tuple(parent_model_states)
        for parent_label, parent_model_states in grouped_model_states.items()
    }


def _state_is_oriented(transport_state: AnalyticTransportStatePrimitive) -> bool:
    return transport_state.state_kind not in (STATE_KIND_FREE_LI, STATE_KIND_FREE_ANION)


def _orientation_relaxation_rate_s_inv(
    transport_state: AnalyticTransportStatePrimitive,
    temperature_K: float,
) -> float:
    if len(transport_state.charge_vector) <= 1:
        raise ValueError(f"{transport_state.label} has no internal orientation")
    local_center_diffusivity_sum_m2_s = math.fsum(
        K_B * temperature_K / float(diagonal_resistance_kg_s)
        for diagonal_resistance_kg_s in np.diag(transport_state.local_resistance_matrix_kg_s)
    )
    polarization_length_m = _positive_float(
        transport_state.current_relaxation_length_m,
        f"{transport_state.label}.current_relaxation_length_m",
    )
    return float(
        local_center_diffusivity_sum_m2_s / (polarization_length_m * polarization_length_m)
    )


def _center_of_mass_diffusion_m2_s(
    transport_state: AnalyticTransportStatePrimitive,
    temperature_K: float,
) -> float:
    mobile_resistance_matrix = (
        transport_state.local_resistance_matrix_kg_s
        + transport_state.atmosphere_resistance_matrix_kg_s
    )
    center_of_mass_drag_kg_s = float(np.trace(mobile_resistance_matrix))
    if center_of_mass_drag_kg_s <= 0.0:
        raise ValueError(f"{transport_state.label} center-of-mass drag must be positive")
    return float(K_B * temperature_K / center_of_mass_drag_kg_s)


def _state_polarization_m(
    transport_state: AnalyticTransportStatePrimitive,
    orientation_vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(transport_state.charge_vector) == 1:
        return (0.0, 0.0, 0.0)
    anion_charge_number = float(transport_state.charge_vector[1])
    return tuple(
        anion_charge_number
        * transport_state.current_relaxation_length_m
        * orientation_axis_value
        for orientation_axis_value in orientation_vector
    )


def _polarization_displacement_m(
    source_model_state: _MarkovAdditiveModelState,
    target_model_state: _MarkovAdditiveModelState,
) -> tuple[float, float, float]:
    return tuple(
        target_axis_value - source_axis_value
        for source_axis_value, target_axis_value in zip(
            source_model_state.polarization_m,
            target_model_state.polarization_m,
        )
    )


def _block_diagonal_matrix(blocks: list[np.ndarray]) -> np.ndarray:
    total_dimension = sum(block.shape[0] for block in blocks)
    matrix = np.zeros((total_dimension, total_dimension), dtype=float)
    offset = 0
    for block in blocks:
        next_offset = offset + block.shape[0]
        matrix[offset:next_offset, offset:next_offset] = block
        offset = next_offset
    return matrix


def _uncertainty_certificate(
    sigma_mS_cm: float,
    uncertainty_budget: StructuralPrimitiveUncertaintyBudget,
) -> AnalyticPrimitiveUncertaintyCertificate:
    sigma_magnitude = abs(float(sigma_mS_cm))
    head_widths = {
        "association_logK": _validated_interval_width(
            uncertainty_budget.association_logK_interval,
            "association_logK_interval",
        ),
        "dielectric_decrement": _dimensionless_interval_fraction_width(
            uncertainty_budget.dielectric_decrement_scale_interval,
            "dielectric_decrement_scale_interval",
        ),
        "jones_dole": _dimensionless_interval_fraction_width(
            uncertainty_budget.jones_dole_scale_interval,
            "jones_dole_scale_interval",
        ),
        "atmosphere_ep": _dimensionless_interval_fraction_width(
            uncertainty_budget.atmosphere_ep_scale_interval,
            "atmosphere_ep_scale_interval",
        ),
        "atmosphere_rel": _dimensionless_interval_fraction_width(
            uncertainty_budget.atmosphere_rel_scale_interval,
            "atmosphere_rel_scale_interval",
        ),
        "lithium_charge_cloud": _positive_interval_fraction_width(
            uncertainty_budget.lithium_charge_cloud_radius_interval_A,
            "lithium_charge_cloud_radius_interval_A",
        ),
        "anion_charge_cloud": _positive_interval_fraction_width(
            uncertainty_budget.anion_charge_cloud_radius_interval_A,
            "anion_charge_cloud_radius_interval_A",
        ),
        "cage_trapping": _validated_interval_width(
            uncertainty_budget.cage_trapping_fraction_interval,
            "cage_trapping_fraction_interval",
        ),
        "jump_length": _dimensionless_interval_fraction_width(
            uncertainty_budget.jump_length_scale_interval,
            "jump_length_scale_interval",
        ),
        "conversion_rate": _dimensionless_interval_fraction_width(
            uncertainty_budget.conversion_rate_scale_interval,
            "conversion_rate_scale_interval",
        ),
        "forced_free_li_translation": _dimensionless_interval_fraction_width(
            uncertainty_budget.forced_free_li_translation_scale_interval,
            "forced_free_li_translation_scale_interval",
        ),
        "forced_compact_anion_translation": _dimensionless_interval_fraction_width(
            uncertainty_budget.forced_compact_anion_translation_scale_interval,
            "forced_compact_anion_translation_scale_interval",
        ),
    }
    dominant_uncertainty_head = max(head_widths, key=head_widths.__getitem__)
    relative_half_width = HALF_FACTOR * math.fsum(head_widths.values())
    half_width = sigma_magnitude * relative_half_width
    sigma_min = max(0.0, float(sigma_mS_cm) - half_width)
    sigma_max = float(sigma_mS_cm) + half_width
    return AnalyticPrimitiveUncertaintyCertificate(
        sigma_min_mS_cm=float(sigma_min),
        sigma_max_mS_cm=float(sigma_max),
        half_width_mS_cm=float(half_width),
        dominant_uncertainty_head=dominant_uncertainty_head,
        threshold_mS_cm=uncertainty_budget.certificate_threshold_mS_cm,
        certified_0p25_mS_cm=bool(
            half_width <= uncertainty_budget.certificate_threshold_mS_cm
        ),
    )


def _dimensionless_interval_fraction_width(
    interval: tuple[float, float],
    context: str,
) -> float:
    _validated_interval_width(interval, context)
    midpoint = HALF_FACTOR * (float(interval[0]) + float(interval[1]))
    if midpoint <= 0.0:
        raise ValueError(f"{context} midpoint must be positive")
    return float((float(interval[1]) - float(interval[0])) / midpoint)


def _positive_interval_fraction_width(
    interval: tuple[float, float],
    context: str,
) -> float:
    _validated_interval_width(interval, context)
    lower_bound = _positive_float(interval[0], f"{context}.lower")
    upper_bound = _positive_float(interval[1], f"{context}.upper")
    midpoint = HALF_FACTOR * (lower_bound + upper_bound)
    return float((upper_bound - lower_bound) / midpoint)
