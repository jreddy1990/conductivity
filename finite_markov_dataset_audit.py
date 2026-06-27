"""Dataset-level audit utilities for the finite Markov conductivity generator."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from constants import F, K_B, MS_CM_TO_S_M, R
from conductivity.finite_markov_conductivity import (
    ChargedCenter,
    ChemicalMotif,
    ChemicalMotifKind,
    FiniteMarkovConductivityResult,
    KernelDerivedMarkovModel,
    TransportState,
    evaluate_finite_markov_conductivity,
)
from data.species_data import ADDITIVES, SALTS, SOLVENTS
from utils.strict_validation import require_float


RecipeDict = dict[str, dict[str, float]]

BASIS_SUM_TOLERANCE = 1.0e-12
MILLILITER_PER_LITER = 1000.0
PERCENT = 100.0
TOP_CONTRIBUTION_COUNT = 10  # Explicit constant: user-requested top_10 ledger entries.
VECTOR_GREEN_KUBO_DIVISOR = 6.0  # Explicit constant: 3D vector jump readout, 1/(2*3).
ISOTROPIC_TRACE_DIVISOR = 3.0  # Explicit constant: isotropic trace average over x,y,z axes.
ANGSTROM_TO_M = 1.0e-10  # Explicit unit conversion for charge-center separation diagnostics.
FAMILY_OWNER_LOG_STRONG_THRESHOLD = 0.15  # User-declared reporting threshold, about 16% multiplicative error.
FAMILY_OWNER_LOG_WEAK_THRESHOLD = 0.10  # User-declared reporting threshold for kappa-tracked atmosphere error.
FAMILY_OWNER_CORRELATION_THRESHOLD = 0.5  # User-declared reporting threshold for family trend ownership.
LOWER_QUARTILE_FRACTION = 0.25  # Analytical lower quartile fraction for quantile bins, not a physics parameter.
MEDIAN_QUARTILE_FRACTION = 0.50  # Analytical median fraction for quantile bins, not a physics parameter.
UPPER_QUARTILE_FRACTION = 0.75  # Analytical upper quartile fraction for quantile bins, not a physics parameter.


@dataclass(frozen=True)
class EmpiricalRecipeCanonicalization:
    recipe: RecipeDict
    adjustments: tuple[str, ...]


@dataclass(frozen=True)
class StateContribution:
    state: str
    motif: str
    motif_kind: str
    orientation: str
    state_concentration_M: float
    stoichiometry: dict[str, float]
    contribution_m2_s: float
    D_NE_alpha_m2_s: float
    D_after_binding_alpha_m2_s: float
    D_none_alpha_m2_s: float
    D_ep_alpha_m2_s: float
    D_rel_alpha_m2_s: float
    D_rel_Li_alpha_m2_s: float
    D_rel_anion_alpha_m2_s: float
    D_rel_diag_alpha_m2_s: float
    D_rel_full_alpha_m2_s: float
    D_full_alpha_m2_s: float
    d_alpha_m2_s: float
    H_binding_alpha: float
    H_atmosphere_alpha: float
    H_ep_alpha: float
    H_rel_alpha: float
    H_rel_Li_alpha: float
    H_rel_anion_alpha: float
    H_rel_diag_alpha: float
    H_rel_cross_alpha: float
    H_rel_before_gate_alpha: float
    H_rel_after_gate_alpha: float
    H_full_alpha: float
    drag_ep_alpha: float
    drag_rel_alpha: float
    drag_rel_Li_alpha: float
    drag_rel_anion_alpha: float
    drag_rel_diag_alpha: float
    drag_rel_cross_alpha: float
    drag_rel_before_gate_alpha: float
    drag_rel_after_gate_alpha: float
    drag_full_alpha: float
    H_alpha: float
    constraint_tau_s: float
    constraint_length_m: float
    constraint_mu: float
    local_resistance_trace_kg_s: float
    binding_resistance_trace_kg_s: float
    atmosphere_resistance_trace_kg_s: float
    electrophoretic_resistance_trace_kg_s: float
    relaxation_resistance_trace_kg_s: float
    relaxation_Li_resistance_trace_kg_s: float
    relaxation_anion_resistance_trace_kg_s: float
    relaxation_diag_resistance_trace_kg_s: float
    relaxation_cross_resistance_offdiag_norm_kg_s: float
    relaxation_resistance_before_gate_trace_kg_s: float
    relaxation_resistance_after_gate_trace_kg_s: float
    single_ion_atmosphere_trace_kg_s: float
    form_factor_atmosphere_trace_kg_s: float
    atmosphere_state_lifetime_s: float
    atmosphere_relaxation_time_s: float
    atmosphere_lifetime_gate: float
    atmosphere_diagnostic_lifetime_gate: float
    relaxation_dynamic_response: str
    relaxation_lifetime_gate: float
    raw_atmosphere_form_factor: float
    effective_atmosphere_form_factor: float
    atmosphere_resistance_before_lifetime_gate_trace_kg_s: float
    atmosphere_resistance_after_lifetime_gate_trace_kg_s: float
    atmosphere_offdiag_norm_kg_s: float
    electrophoretic_offdiag_norm_kg_s: float
    relaxation_offdiag_norm_kg_s: float
    atmosphere_min_eig_kg_s: float
    atmosphere_max_eig_kg_s: float
    atmosphere_bath_basis: str
    ionic_strength_total_mol_m3: float
    ionic_strength_external_mol_m3: float
    external_over_total_ionic_strength: float
    resolved_charge_center_count: int
    anion_feature_id: str
    local_D_Li_m2_s: float
    local_D_anion_m2_s: float
    kappa_radius_Li: float
    kappa_radius_anion: float
    debye_kappa_inv_A: float
    separation_over_debye: float
    mean_charge_center_separation_A: float
    atmosphere_form_factor_cancellation: float
    thermodynamic_factor_trace: float
    thermodynamic_factor_eigenvalues: tuple[float, ...]
    structure_factor_charge_mode: float
    stationary_probability: float
    charge: float


@dataclass(frozen=True)
class EdgeContribution:
    source_state: str
    target_state: str
    contribution_m2_s: float
    rate_s_inv: float
    raw_delta2_m2: float
    corrected_delta2_m2: float


@dataclass(frozen=True)
class BiasLedgerRow:
    row_id: int
    sigma_exp_mS_cm: float
    sigma_pred_mS_cm: float
    log_error: float
    absolute_error_mS_cm: float
    signed_error_mS_cm: float
    C_Li_mol_m3: float
    ionic_strength_mol_m3: float
    salt_family: str
    solvent_family: str
    additive_basis: str
    eta_rel: float
    ionic_occupied_volume_fraction: float
    crowding_factor: float
    D_uncorr_no_crowding_m2_s: float
    D_uncorr_with_crowding_m2_s: float
    anion_shape_factor_by_feature: dict[str, float]
    cation_microviscosity_coupling_exponent: float
    anion_microviscosity_coupling_exponent_by_feature: dict[str, float]
    carrier_strength_Li_mS_cm: float
    carrier_strength_anion_by_feature_mS_cm: dict[str, float]
    D_NE_state_m2_s: float
    D_after_binding_m2_s: float
    D_after_atmosphere_m2_s: float
    D_state_m2_s: float
    H_binding: float
    H_atmosphere: float
    H_atmosphere_target: float
    H_ep: float
    H_rel: float
    H_rel_Li: float
    H_rel_anion: float
    H_rel_diag: float
    H_rel_cross: float
    H_rel_before_gate: float
    H_rel_after_gate: float
    H_full: float
    drag_ep: float
    drag_rel: float
    drag_rel_Li: float
    drag_rel_anion: float
    drag_rel_diag: float
    drag_rel_cross: float
    drag_rel_before_gate: float
    drag_rel_after_gate: float
    drag_full: float
    drag_ep_current_over_target: float
    drag_rel_current_over_target: float
    drag_rel_Li_current_over_target: float
    drag_rel_anion_current_over_target: float
    drag_rel_diag_current_over_target: float
    drag_rel_cross_current_over_target: float
    D_none_state_m2_s: float
    D_ep_state_m2_s: float
    D_rel_state_m2_s: float
    D_rel_Li_state_m2_s: float
    D_rel_anion_state_m2_s: float
    D_rel_diag_state_m2_s: float
    D_rel_full_state_m2_s: float
    D_full_state_m2_s: float
    r_atmosphere_current: float
    r_atmosphere_target: float
    r_atmosphere_current_over_target: float
    atmosphere_bath_basis: str
    relaxation_dynamic_response: str
    mean_relaxation_lifetime_gate: float
    ionic_strength_total_mol_m3: float
    ionic_strength_external_mol_m3: float
    external_over_total_ionic_strength: float
    H_atmosphere_total_bath: float
    H_atmosphere_total_bath_evaluated: bool
    H_atmosphere_external_bath: float
    H_atmosphere_external_bath_evaluated: bool
    top_state_resolved_charge_count: int
    association_required_multiplier: float
    association_required_deltaG_kJ_mol: float
    base_mobility_required_multiplier: float
    base_mobility_required_deltaG_kJ_mol: float
    H_state: float
    D_uncorr_m2_s: float
    D_Q_m2_s: float
    D_Q_target_m2_s: float
    H_gen: float
    H_jump: float
    D_veh_m2_s: float
    D_jump_m2_s: float
    motif_populations: dict[str, float]
    motif_lifetimes_s: dict[str, float]
    state_concentration_by_state_M: dict[str, float]
    state_stoichiometry_by_state: dict[str, dict[str, float]]
    ssip_association_constant_by_feature_M_inv: dict[str, float]
    cip_association_constant_by_feature_M_inv: dict[str, float]
    tau_cage_s: float
    tau_pair_by_feature_s: dict[str, float]
    mean_delta2_raw_m2: float
    mean_delta2_corrected_m2: float
    poisson_correction_ratio: float
    sigma_no_poisson_mS_cm: float
    sigma_veh_only_mS_cm: float
    row_sum_residual_s_inv: float
    stationary_residual_s_inv: float
    detailed_balance_residual_s_inv: float
    basis_adjustments: tuple[str, ...]
    top_state_contributions: tuple[StateContribution, ...]
    top_edge_contributions: tuple[EdgeContribution, ...]


@dataclass(frozen=True)
class FailedDatasetRow:
    row_id: int
    error: str


@dataclass(frozen=True)
class SaltFamilyMetrics:
    count: int
    bias_mS_cm: float
    mae_mS_cm: float
    rmse_mS_cm: float


@dataclass(frozen=True)
class FamilyAtmosphereMetrics:
    group_name: str
    group_value: str
    count: int
    bias_mS_cm: float
    mae_mS_cm: float
    rmse_mS_cm: float
    mean_log_sigma_error: float
    mean_log_base_error: float
    mean_log_atmosphere_error: float
    mean_log_ep_error: float
    mean_log_rel_error: float
    mean_log_rel_Li_error: float
    mean_log_rel_anion_error: float
    mean_log_rel_diag_error: float
    mean_log_rel_cross_error: float
    mean_log_rel_before_gate_error: float
    mean_log_rel_after_gate_error: float
    mean_H_atmosphere: float
    mean_H_atmosphere_target: float
    mean_H_ratio: float
    mean_H_ep: float
    mean_H_rel: float
    mean_H_rel_Li: float
    mean_H_rel_anion: float
    mean_H_rel_diag: float
    mean_H_rel_cross: float
    mean_H_rel_before_gate: float
    mean_H_rel_after_gate: float
    mean_H_full: float
    mean_r_atm_current: float
    mean_r_atm_target: float
    mean_r_atm_current_over_target: float
    mean_drag_ep_current_over_target: float
    mean_drag_rel_current_over_target: float
    mean_drag_rel_Li_current_over_target: float
    mean_drag_rel_anion_current_over_target: float
    mean_drag_rel_diag_current_over_target: float
    mean_drag_rel_cross_current_over_target: float
    mean_drag_rel_before_gate_current_over_target: float
    mean_drag_rel_after_gate_current_over_target: float
    mean_relaxation_lifetime_gate: float
    mean_eta_rel: float
    mean_kappa_inv_A: float
    mean_ionic_strength_mol_m3: float
    dominant_top_state: str
    owner: str


@dataclass(frozen=True)
class DatasetAuditResult:
    labeled_rows: int
    evaluated_rows: int
    failed_rows: int
    mae_mS_cm: float
    rmse_mS_cm: float
    bias_mS_cm: float
    mape_percent: float
    r2: float
    pearson_r: float
    max_row_sum_residual_s_inv: float
    max_stationary_residual_s_inv: float
    max_detailed_balance_residual_s_inv: float
    ledger_rows: tuple[BiasLedgerRow, ...]
    failures: tuple[FailedDatasetRow, ...]
    salt_family_metrics: dict[str, SaltFamilyMetrics]
    family_atmosphere_metrics: tuple[FamilyAtmosphereMetrics, ...]


def canonicalize_empirical_recipe(empirical_recipe: Mapping[str, object]) -> EmpiricalRecipeCanonicalization:
    """Convert empirical-database recipes to the hard recipe model basis."""

    raw_sections = _require_recipe_sections(empirical_recipe)
    raw_solvents = _strict_float_section(raw_sections["solvents"], "recipe.solvents")
    raw_salts = _strict_float_section(raw_sections["salts"], "recipe.salts")
    raw_additives = _strict_float_section(raw_sections["additives"], "recipe.additives")
    adjustments: list[str] = []

    solvent_role_additives: dict[str, float] = {}
    canonical_solvents: dict[str, float] = {}
    for species_name, fraction in raw_solvents.items():
        _assert_nonnegative_finite(fraction, f"recipe.solvents.{species_name}")
        if species_name in SOLVENTS:
            canonical_solvents[species_name] = fraction
        elif species_name in ADDITIVES:
            solvent_role_additives[species_name] = fraction
            adjustments.append(f"solvent_role_to_additive_weight:{species_name}")
        else:
            raise KeyError(f"recipe solvent-role species {species_name} is neither solvent nor additive")

    solvent_fraction_sum = math.fsum(canonical_solvents.values())
    _assert_positive_finite(solvent_fraction_sum, "canonical solvent fraction sum")
    if abs(solvent_fraction_sum - 1.0) > BASIS_SUM_TOLERANCE:
        adjustments.append(f"solvent_basis_normalized:{solvent_fraction_sum:.12g}")
    canonical_solvents = {
        species_name: fraction / solvent_fraction_sum
        for species_name, fraction in canonical_solvents.items()
    }

    canonical_salts: dict[str, float] = {}
    ionic_additive_molarities: dict[str, float] = {}
    for species_name, molarity_M in raw_salts.items():
        _assert_nonnegative_finite(molarity_M, f"recipe.salts.{species_name}")
        if species_name in SALTS:
            canonical_salts[species_name] = molarity_M
        elif _is_ionic_source(species_name):
            ionic_additive_molarities[species_name] = molarity_M
            adjustments.append(f"ionic_additive_salt_to_weight:{species_name}")
        else:
            raise KeyError(f"recipe salt-role species {species_name} is not an ionic source")

    canonical_additives = dict(raw_additives)
    for species_name, weight_fraction in canonical_additives.items():
        _assert_nonnegative_finite(weight_fraction, f"recipe.additives.{species_name}")
        if species_name not in ADDITIVES:
            raise KeyError(f"recipe additive species {species_name} is not an additive")

    converted_volume_role_additives = _volume_role_additives_to_weight_fractions(
        solvent_role_additives,
        canonical_solvents,
    )
    for species_name, weight_fraction in converted_volume_role_additives.items():
        _accumulate_value(canonical_additives, species_name, weight_fraction)

    converted_ionic_additives = _ionic_additive_salts_to_weight_fractions(
        ionic_additive_molarities,
        canonical_solvents,
        canonical_salts,
        canonical_additives,
    )
    for species_name, weight_fraction in converted_ionic_additives.items():
        _accumulate_value(canonical_additives, species_name, weight_fraction)

    return EmpiricalRecipeCanonicalization(
        recipe={
            "solvents": canonical_solvents,
            "salts": canonical_salts,
            "additives": canonical_additives,
        },
        adjustments=tuple(adjustments),
    )


def build_bias_ledger_row(
    row_id: int,
    empirical_conductivity_mS_cm: float,
    canonicalization: EmpiricalRecipeCanonicalization,
    result: FiniteMarkovConductivityResult,
    temperature_K: float,
) -> BiasLedgerRow:
    _assert_positive_finite(empirical_conductivity_mS_cm, "empirical_conductivity_mS_cm")
    _assert_positive_finite(temperature_K, "temperature_K")
    model = _require_generated_model(result)
    cation_concentration_mol_m3 = model.mixture_audit.cation_concentration_mol_m3
    _assert_positive_finite(cation_concentration_mol_m3, "C_Li_mol_m3")
    target_conductivity_S_m = empirical_conductivity_mS_cm * MS_CM_TO_S_M
    target_D_Q_m2_s = (
        target_conductivity_S_m
        * R
        * temperature_K
        / (F * F * cation_concentration_mol_m3)
    )
    _assert_positive_finite(target_D_Q_m2_s, "D_Q_target_m2_s")

    state_NE_D_Q_m2_s = _state_nernst_einstein_diffusivity_m2_s(model)
    state_binding_D_Q_m2_s = _state_binding_diffusivity_m2_s(model, temperature_K)
    state_ep_D_Q_m2_s = _state_component_atmosphere_diffusivity_m2_s(
        model,
        temperature_K,
        "electrophoretic",
    )
    state_rel_D_Q_m2_s = _state_component_atmosphere_diffusivity_m2_s(
        model,
        temperature_K,
        "relaxation",
    )
    state_rel_Li_D_Q_m2_s = _state_component_atmosphere_diffusivity_m2_s(
        model,
        temperature_K,
        "relaxation_Li_after_ep",
    )
    state_rel_anion_D_Q_m2_s = _state_component_atmosphere_diffusivity_m2_s(
        model,
        temperature_K,
        "relaxation_anion_after_ep",
    )
    state_rel_before_gate_D_Q_m2_s = _state_component_atmosphere_diffusivity_m2_s(
        model,
        temperature_K,
        "relaxation_before_gate",
    )
    state_rel_diag_D_Q_m2_s = _state_component_atmosphere_diffusivity_m2_s(
        model,
        temperature_K,
        "relaxation_diag_after_ep",
    )
    state_resistance_D_Q_m2_s = _state_resistance_diffusivity_m2_s(model, temperature_K)
    _assert_positive_finite(state_NE_D_Q_m2_s, "D_NE_state_m2_s")
    _assert_positive_finite(state_binding_D_Q_m2_s, "D_after_binding_m2_s")
    _assert_positive_finite(state_ep_D_Q_m2_s, "D_ep_state_m2_s")
    _assert_positive_finite(state_rel_D_Q_m2_s, "D_rel_state_m2_s")
    _assert_positive_finite(state_rel_Li_D_Q_m2_s, "D_rel_Li_state_m2_s")
    _assert_positive_finite(state_rel_anion_D_Q_m2_s, "D_rel_anion_state_m2_s")
    _assert_positive_finite(state_rel_before_gate_D_Q_m2_s, "D_rel_before_gate_state_m2_s")
    _assert_positive_finite(state_rel_diag_D_Q_m2_s, "D_rel_diag_state_m2_s")
    _assert_positive_finite(state_resistance_D_Q_m2_s, "D_state_m2_s")
    binding_factor = state_binding_D_Q_m2_s / state_NE_D_Q_m2_s
    atmosphere_factor = state_resistance_D_Q_m2_s / state_binding_D_Q_m2_s
    atmosphere_target_factor = target_D_Q_m2_s / state_binding_D_Q_m2_s
    electrophoretic_factor = state_ep_D_Q_m2_s / state_binding_D_Q_m2_s
    relaxation_factor = state_rel_D_Q_m2_s / state_binding_D_Q_m2_s
    relaxation_Li_factor = state_rel_Li_D_Q_m2_s / state_ep_D_Q_m2_s
    relaxation_anion_factor = state_rel_anion_D_Q_m2_s / state_ep_D_Q_m2_s
    relaxation_before_gate_factor = state_rel_before_gate_D_Q_m2_s / state_binding_D_Q_m2_s
    relaxation_diag_factor = state_rel_diag_D_Q_m2_s / state_ep_D_Q_m2_s
    relaxation_cross_factor = state_resistance_D_Q_m2_s / state_rel_diag_D_Q_m2_s
    electrophoretic_drag_ratio = (1.0 / electrophoretic_factor) - 1.0
    relaxation_drag_ratio = (1.0 / relaxation_factor) - 1.0
    relaxation_Li_drag_ratio = (1.0 / relaxation_Li_factor) - 1.0
    relaxation_anion_drag_ratio = (1.0 / relaxation_anion_factor) - 1.0
    relaxation_before_gate_drag_ratio = (1.0 / relaxation_before_gate_factor) - 1.0
    relaxation_diag_drag_ratio = (1.0 / relaxation_diag_factor) - 1.0
    relaxation_cross_drag_ratio = (1.0 / relaxation_cross_factor) - 1.0
    atmosphere_current_drag_ratio = (1.0 / atmosphere_factor) - 1.0
    atmosphere_target_drag_ratio = (1.0 / atmosphere_target_factor) - 1.0
    atmosphere_current_over_target_drag_ratio = _ratio_with_zero_denominator_convention(
        atmosphere_current_drag_ratio,
        atmosphere_target_drag_ratio,
    )
    state_resistance_factor = state_resistance_D_Q_m2_s / state_NE_D_Q_m2_s
    uncorrelated_D_Q_m2_s = state_resistance_D_Q_m2_s + _uncorrelated_jump_diffusivity_m2_s(model)
    _assert_positive_finite(model.mixture_audit.crowding_factor, "mixture_audit.crowding_factor")
    uncorrelated_D_Q_no_crowding_m2_s = uncorrelated_D_Q_m2_s / model.mixture_audit.crowding_factor
    uncorrelated_jump_D_Q_m2_s = _uncorrelated_jump_diffusivity_m2_s(model)
    _assert_nonnegative_finite(uncorrelated_jump_D_Q_m2_s, "uncorrelated_jump_D_Q_m2_s")
    memory_factor = result.D_Q_m2_s / uncorrelated_D_Q_m2_s
    jump_factor = result.D_Q_m2_s / state_resistance_D_Q_m2_s
    if uncorrelated_jump_D_Q_m2_s == 0.0:
        if result.jump_D_Q_m2_s != 0.0:
            raise ValueError("jump_D_Q_m2_s is nonzero with zero uncorrelated jump diffusivity")
        poisson_correction_ratio = 1.0
    else:
        poisson_correction_ratio = result.jump_D_Q_m2_s / uncorrelated_jump_D_Q_m2_s
    sigma_no_poisson_mS_cm = (
        F
        * F
        * cation_concentration_mol_m3
        * uncorrelated_D_Q_m2_s
        / (R * temperature_K * MS_CM_TO_S_M)
    )
    sigma_veh_only_mS_cm = (
        F
        * F
        * cation_concentration_mol_m3
        * result.vehicular_D_Q_m2_s
        / (R * temperature_K * MS_CM_TO_S_M)
    )
    solvent_viscosity_cP = _solvent_blend_viscosity_cP(canonicalization.recipe["solvents"])
    eta_relative = model.mixture_audit.viscosity_cP / solvent_viscosity_cP
    predicted_conductivity = result.sigma_mS_cm
    signed_error = predicted_conductivity - empirical_conductivity_mS_cm

    motif_lifetimes = _motif_lifetimes_s(model)
    motif_populations = _motif_population_rollup(model)
    top_state_contributions = _top_state_contributions(model, temperature_K)
    top_state_resolved_charge_count = _top_state_resolved_charge_count(top_state_contributions)
    total_ionic_strength_mol_m3 = _concentration_weighted_total_ionic_strength_mol_m3(model)
    ionic_strength_external_mol_m3 = _concentration_weighted_external_ionic_strength_mol_m3(model)
    external_over_total_ionic_strength = _ratio_with_zero_denominator_convention(
        ionic_strength_external_mol_m3,
        total_ionic_strength_mol_m3,
    )
    base_mobility_required_multiplier = _positive_ratio(
        target_D_Q_m2_s / atmosphere_factor,
        state_binding_D_Q_m2_s,
        f"row {row_id} base_mobility_required_multiplier",
    )
    association_required_multiplier = 1.0 / base_mobility_required_multiplier
    total_bath_is_evaluated = model.atmosphere_bath_basis == "total_formal"
    external_bath_is_evaluated = model.atmosphere_bath_basis == "external_free_bath"
    return BiasLedgerRow(
        row_id=row_id,
        sigma_exp_mS_cm=empirical_conductivity_mS_cm,
        sigma_pred_mS_cm=predicted_conductivity,
        log_error=math.log(predicted_conductivity / empirical_conductivity_mS_cm),
        absolute_error_mS_cm=abs(signed_error),
        signed_error_mS_cm=signed_error,
        C_Li_mol_m3=cation_concentration_mol_m3,
        ionic_strength_mol_m3=_ionic_strength_mol_m3(model),
        salt_family=_salt_family(canonicalization.recipe),
        solvent_family=_solvent_family(canonicalization.recipe),
        additive_basis=_additive_basis(canonicalization.recipe),
        eta_rel=eta_relative,
        ionic_occupied_volume_fraction=model.mixture_audit.ionic_occupied_volume_fraction,
        crowding_factor=model.mixture_audit.crowding_factor,
        D_uncorr_no_crowding_m2_s=uncorrelated_D_Q_no_crowding_m2_s,
        D_uncorr_with_crowding_m2_s=uncorrelated_D_Q_m2_s,
        anion_shape_factor_by_feature=dict(model.mixture_audit.anion_shape_factor_by_feature),
        cation_microviscosity_coupling_exponent=model.mixture_audit.cation_microviscosity_coupling_exponent,
        anion_microviscosity_coupling_exponent_by_feature=dict(
            model.mixture_audit.anion_microviscosity_coupling_exponent_by_feature
        ),
        carrier_strength_Li_mS_cm=model.mixture_audit.carrier_strength_Li_mS_cm,
        carrier_strength_anion_by_feature_mS_cm=dict(model.mixture_audit.carrier_strength_anion_by_feature_mS_cm),
        D_NE_state_m2_s=state_NE_D_Q_m2_s,
        D_after_binding_m2_s=state_binding_D_Q_m2_s,
        D_after_atmosphere_m2_s=state_resistance_D_Q_m2_s,
        D_state_m2_s=state_resistance_D_Q_m2_s,
        H_binding=binding_factor,
        H_atmosphere=atmosphere_factor,
        H_atmosphere_target=atmosphere_target_factor,
        H_ep=electrophoretic_factor,
        H_rel=relaxation_factor,
        H_rel_Li=relaxation_Li_factor,
        H_rel_anion=relaxation_anion_factor,
        H_rel_diag=relaxation_diag_factor,
        H_rel_cross=relaxation_cross_factor,
        H_rel_before_gate=relaxation_before_gate_factor,
        H_rel_after_gate=relaxation_factor,
        H_full=atmosphere_factor,
        drag_ep=electrophoretic_drag_ratio,
        drag_rel=relaxation_drag_ratio,
        drag_rel_Li=relaxation_Li_drag_ratio,
        drag_rel_anion=relaxation_anion_drag_ratio,
        drag_rel_diag=relaxation_diag_drag_ratio,
        drag_rel_cross=relaxation_cross_drag_ratio,
        drag_rel_before_gate=relaxation_before_gate_drag_ratio,
        drag_rel_after_gate=relaxation_drag_ratio,
        drag_full=atmosphere_current_drag_ratio,
        drag_ep_current_over_target=_ratio_with_zero_denominator_convention(
            electrophoretic_drag_ratio,
            atmosphere_target_drag_ratio,
        ),
        drag_rel_current_over_target=_ratio_with_zero_denominator_convention(
            relaxation_drag_ratio,
            atmosphere_target_drag_ratio,
        ),
        drag_rel_Li_current_over_target=_ratio_with_zero_denominator_convention(
            relaxation_Li_drag_ratio,
            atmosphere_target_drag_ratio,
        ),
        drag_rel_anion_current_over_target=_ratio_with_zero_denominator_convention(
            relaxation_anion_drag_ratio,
            atmosphere_target_drag_ratio,
        ),
        drag_rel_diag_current_over_target=_ratio_with_zero_denominator_convention(
            relaxation_diag_drag_ratio,
            atmosphere_target_drag_ratio,
        ),
        drag_rel_cross_current_over_target=_ratio_with_zero_denominator_convention(
            relaxation_cross_drag_ratio,
            atmosphere_target_drag_ratio,
        ),
        D_none_state_m2_s=state_binding_D_Q_m2_s,
        D_ep_state_m2_s=state_ep_D_Q_m2_s,
        D_rel_state_m2_s=state_rel_D_Q_m2_s,
        D_rel_Li_state_m2_s=state_rel_Li_D_Q_m2_s,
        D_rel_anion_state_m2_s=state_rel_anion_D_Q_m2_s,
        D_rel_diag_state_m2_s=state_rel_diag_D_Q_m2_s,
        D_rel_full_state_m2_s=state_resistance_D_Q_m2_s,
        D_full_state_m2_s=state_resistance_D_Q_m2_s,
        r_atmosphere_current=atmosphere_current_drag_ratio,
        r_atmosphere_target=atmosphere_target_drag_ratio,
        r_atmosphere_current_over_target=atmosphere_current_over_target_drag_ratio,
        atmosphere_bath_basis=model.atmosphere_bath_basis,
        relaxation_dynamic_response=model.relaxation_dynamic_response,
        mean_relaxation_lifetime_gate=_state_concentration_weighted_relaxation_gate(model),
        ionic_strength_total_mol_m3=total_ionic_strength_mol_m3,
        ionic_strength_external_mol_m3=ionic_strength_external_mol_m3,
        external_over_total_ionic_strength=external_over_total_ionic_strength,
        H_atmosphere_total_bath=atmosphere_factor if total_bath_is_evaluated else 0.0,
        H_atmosphere_total_bath_evaluated=total_bath_is_evaluated,
        H_atmosphere_external_bath=atmosphere_factor if external_bath_is_evaluated else 0.0,
        H_atmosphere_external_bath_evaluated=external_bath_is_evaluated,
        top_state_resolved_charge_count=top_state_resolved_charge_count,
        association_required_multiplier=association_required_multiplier,
        association_required_deltaG_kJ_mol=(
            -R * temperature_K * math.log(association_required_multiplier) / 1000.0
        ),
        base_mobility_required_multiplier=base_mobility_required_multiplier,
        base_mobility_required_deltaG_kJ_mol=(
            -R * temperature_K * math.log(base_mobility_required_multiplier) / 1000.0
        ),
        H_state=state_resistance_factor,
        D_uncorr_m2_s=uncorrelated_D_Q_m2_s,
        D_Q_m2_s=result.D_Q_m2_s,
        D_Q_target_m2_s=target_D_Q_m2_s,
        H_gen=memory_factor,
        H_jump=jump_factor,
        D_veh_m2_s=result.vehicular_D_Q_m2_s,
        D_jump_m2_s=result.jump_D_Q_m2_s,
        motif_populations=motif_populations,
        motif_lifetimes_s=motif_lifetimes,
        state_concentration_by_state_M=_state_concentration_by_state_M(model),
        state_stoichiometry_by_state=_state_stoichiometry_by_state(model),
        ssip_association_constant_by_feature_M_inv=_association_constants_by_feature_M_inv(
            model,
            ChemicalMotifKind.SSIP,
        ),
        cip_association_constant_by_feature_M_inv=_association_constants_by_feature_M_inv(
            model,
            ChemicalMotifKind.CIP,
        ),
        tau_cage_s=_mean_lifetime_for_motif_kinds(
            model,
            motif_lifetimes,
            (ChemicalMotifKind.SOLVENT_CAGE, ChemicalMotifKind.ADDITIVE_COORDINATED),
        ),
        tau_pair_by_feature_s=_pair_lifetimes_by_feature_s(model, motif_lifetimes),
        mean_delta2_raw_m2=_weighted_edge_delta2_m2(model, None),
        mean_delta2_corrected_m2=_weighted_edge_delta2_m2(model, result.poisson_correctors_m),
        poisson_correction_ratio=poisson_correction_ratio,
        sigma_no_poisson_mS_cm=sigma_no_poisson_mS_cm,
        sigma_veh_only_mS_cm=sigma_veh_only_mS_cm,
        row_sum_residual_s_inv=result.row_sum_residual_s_inv,
        stationary_residual_s_inv=result.stationary_residual_s_inv,
        detailed_balance_residual_s_inv=result.detailed_balance_residual_s_inv,
        basis_adjustments=canonicalization.adjustments,
        top_state_contributions=top_state_contributions,
        top_edge_contributions=_top_edge_contributions(model, result),
    )


def audit_empirical_conductivity_dataset(
    entries: Sequence[Mapping[str, object]],
    temperature_K: float,
    atmosphere_bath_basis: str = "total_formal",
    relaxation_dynamic_response: str = "off",
) -> DatasetAuditResult:
    _assert_positive_finite(temperature_K, "temperature_K")
    ledger_rows: list[BiasLedgerRow] = []
    failures: list[FailedDatasetRow] = []
    labeled_rows = 0

    for row_id, entry in enumerate(entries):
        entry_sections = _require_entry(entry, row_id)
        properties = entry_sections["properties"]
        if "conductivity_mS_cm" not in properties:
            continue
        labeled_rows += 1
        try:
            empirical_conductivity = require_float(
                properties,
                "conductivity_mS_cm",
                f"DATA[{row_id}].properties",
            )
            canonicalization = canonicalize_empirical_recipe(entry_sections["recipe"])
            result = evaluate_finite_markov_conductivity(
                canonicalization.recipe,
                temperature_K,
                atmosphere_bath_basis,
                relaxation_dynamic_response,
            )
            ledger_rows.append(
                build_bias_ledger_row(
                    row_id=row_id,
                    empirical_conductivity_mS_cm=empirical_conductivity,
                    canonicalization=canonicalization,
                    result=result,
                    temperature_K=temperature_K,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(FailedDatasetRow(row_id=row_id, error=str(exc)))

    metrics = _dataset_metrics(ledger_rows)
    return DatasetAuditResult(
        labeled_rows=labeled_rows,
        evaluated_rows=len(ledger_rows),
        failed_rows=len(failures),
        mae_mS_cm=metrics["mae_mS_cm"],
        rmse_mS_cm=metrics["rmse_mS_cm"],
        bias_mS_cm=metrics["bias_mS_cm"],
        mape_percent=metrics["mape_percent"],
        r2=metrics["r2"],
        pearson_r=metrics["pearson_r"],
        max_row_sum_residual_s_inv=_max_row_sum_residual_s_inv(ledger_rows),
        max_stationary_residual_s_inv=_max_stationary_residual_s_inv(ledger_rows),
        max_detailed_balance_residual_s_inv=_max_detailed_balance_residual_s_inv(ledger_rows),
        ledger_rows=tuple(ledger_rows),
        failures=tuple(failures),
        salt_family_metrics=_salt_family_metrics(ledger_rows),
        family_atmosphere_metrics=_family_atmosphere_metrics(ledger_rows),
    )


def _require_generated_model(result: FiniteMarkovConductivityResult) -> KernelDerivedMarkovModel:
    if result.generated_model is None:
        raise ValueError("finite Markov result does not include generated_model")
    return result.generated_model


def _state_concentration_by_state_M(model: KernelDerivedMarkovModel) -> dict[str, float]:
    return {
        model.state_concentration_kernel.state_labels[state_index]: float(
            model.state_concentration_kernel.state_concentrations_M[state_index]
        )
        for state_index in range(len(model.state_concentration_kernel.state_labels))
    }


def _state_stoichiometry_by_state(model: KernelDerivedMarkovModel) -> dict[str, dict[str, float]]:
    return {
        state_label: _state_stoichiometry_for_index(model, state_index)
        for state_index, state_label in enumerate(model.state_concentration_kernel.state_labels)
    }


def _state_stoichiometry_for_index(
    model: KernelDerivedMarkovModel,
    state_index: int,
) -> dict[str, float]:
    stoichiometry_row = model.state_concentration_kernel.stoichiometry[state_index]
    stoichiometry: dict[str, float] = {}
    for species_index, species_label in enumerate(model.state_concentration_kernel.species_labels):
        species_count = float(stoichiometry_row[species_index])
        if species_count != 0.0:
            stoichiometry[species_label] = species_count
    return stoichiometry


def _association_constants_by_feature_M_inv(
    model: KernelDerivedMarkovModel,
    motif_kind: ChemicalMotifKind,
) -> dict[str, float]:
    constants_by_feature: dict[str, float] = {}
    for state_index, state in enumerate(model.states):
        motif = state.chemical_motif
        if motif.kind is motif_kind and motif.feature_id is not None:
            state_concentration_M = float(model.state_concentration_kernel.state_concentrations_M[state_index])
            activity_product_M = _state_activity_product_M(model, state_index)
            constants_by_feature[motif.feature_id] = state_concentration_M / activity_product_M
    return constants_by_feature


def _state_activity_product_M(
    model: KernelDerivedMarkovModel,
    state_index: int,
) -> float:
    activity_product_M = 1.0
    stoichiometry_row = model.state_concentration_kernel.stoichiometry[state_index]
    for species_index, species_label in enumerate(model.state_concentration_kernel.species_labels):
        species_count = float(stoichiometry_row[species_index])
        if species_count != 0.0:
            free_activity_M = float(model.state_concentration_kernel.free_activities_M[species_label])
            _assert_positive_finite(free_activity_M, f"free_activity_M.{species_label}")
            activity_product_M *= free_activity_M ** species_count
    _assert_positive_finite(activity_product_M, "state_activity_product_M")
    return activity_product_M


def _model_cation_concentration_mol_m3(model: KernelDerivedMarkovModel) -> float:
    cation_concentration_mol_m3 = model.mixture_audit.cation_concentration_mol_m3
    _assert_positive_finite(cation_concentration_mol_m3, "mixture_audit.cation_concentration_mol_m3")
    return cation_concentration_mol_m3


def _state_resistance_diffusivity_m2_s(
    model: KernelDerivedMarkovModel,
    temperature_K: float,
) -> float:
    cation_concentration_mol_m3 = _model_cation_concentration_mol_m3(model)
    return float(
        math.fsum(
            float(model.state_concentrations_mol_m3[state_index])
            * _transport_state_trace_average_m2_s(transport_state, temperature_K)
            for state_index, transport_state in enumerate(model.transport_states)
        )
        / cation_concentration_mol_m3
    )


def _state_binding_diffusivity_m2_s(
    model: KernelDerivedMarkovModel,
    temperature_K: float,
) -> float:
    cation_concentration_mol_m3 = _model_cation_concentration_mol_m3(model)
    return float(
        math.fsum(
            float(model.state_concentrations_mol_m3[state_index])
            * _transport_state_trace_average_for_resistance_stage_m2_s(
                transport_state,
                temperature_K,
                "binding",
            )
            for state_index, transport_state in enumerate(model.transport_states)
        )
        / cation_concentration_mol_m3
    )


def _state_component_atmosphere_diffusivity_m2_s(
    model: KernelDerivedMarkovModel,
    temperature_K: float,
    atmosphere_component: str,
) -> float:
    cation_concentration_mol_m3 = _model_cation_concentration_mol_m3(model)
    return float(
        math.fsum(
            float(model.state_concentrations_mol_m3[state_index])
            * _transport_state_trace_average_for_atmosphere_component_m2_s(
                model,
                transport_state,
                temperature_K,
                atmosphere_component,
            )
            for state_index, transport_state in enumerate(model.transport_states)
        )
        / cation_concentration_mol_m3
    )


def _state_nernst_einstein_diffusivity_m2_s(model: KernelDerivedMarkovModel) -> float:
    cation_concentration_mol_m3 = _model_cation_concentration_mol_m3(model)
    return float(
        math.fsum(
            float(model.state_concentrations_mol_m3[state_index])
            * _transport_state_nernst_einstein_m2_s(transport_state)
            for state_index, transport_state in enumerate(model.transport_states)
        )
        / cation_concentration_mol_m3
    )


def _uncorrelated_jump_diffusivity_m2_s(model: KernelDerivedMarkovModel) -> float:
    jump_second_moment_m2_s = 0.0
    for edge in model.markov_additive_edges:
        raw_delta2_m2 = math.fsum(component * component for component in edge.displacement_m)
        jump_second_moment_m2_s += (
            model.state_concentrations_mol_m3[edge.source_index]
            * edge.rate_s_inv
            * raw_delta2_m2
        )
    return float(jump_second_moment_m2_s / (VECTOR_GREEN_KUBO_DIVISOR * _model_cation_concentration_mol_m3(model)))


def _top_state_contributions(
    model: KernelDerivedMarkovModel,
    temperature_K: float,
) -> tuple[StateContribution, ...]:
    contributions: list[StateContribution] = []
    for state_index, state in enumerate(model.states):
        transport_state = model.transport_states[state_index]
        state_resistance_diffusivity_m2_s = _transport_state_trace_average_m2_s(
            transport_state,
            temperature_K,
        )
        state_binding_diffusivity_m2_s = _transport_state_trace_average_for_resistance_stage_m2_s(
            transport_state,
            temperature_K,
            "binding",
        )
        state_ep_diffusivity_m2_s = _transport_state_trace_average_for_atmosphere_component_m2_s(
            model,
            transport_state,
            temperature_K,
            "electrophoretic",
        )
        state_rel_diffusivity_m2_s = _transport_state_trace_average_for_atmosphere_component_m2_s(
            model,
            transport_state,
            temperature_K,
            "relaxation",
        )
        state_rel_before_gate_diffusivity_m2_s = (
            _transport_state_trace_average_for_atmosphere_component_m2_s(
                model,
                transport_state,
                temperature_K,
                "relaxation_before_gate",
            )
        )
        state_rel_diag_diffusivity_m2_s = (
            _transport_state_trace_average_for_atmosphere_component_m2_s(
                model,
                transport_state,
                temperature_K,
                "relaxation_diag_after_ep",
            )
        )
        state_NE_diffusivity_m2_s = _transport_state_nernst_einstein_m2_s(transport_state)
        contribution = float(
            model.state_concentrations_mol_m3[state_index]
            * state_resistance_diffusivity_m2_s
            / _model_cation_concentration_mol_m3(model)
        )
        binding_factor = _state_factor(state_binding_diffusivity_m2_s, state_NE_diffusivity_m2_s)
        atmosphere_factor = _state_factor(state_resistance_diffusivity_m2_s, state_binding_diffusivity_m2_s)
        electrophoretic_factor = _state_factor(state_ep_diffusivity_m2_s, state_binding_diffusivity_m2_s)
        relaxation_factor = _state_factor(state_rel_diffusivity_m2_s, state_binding_diffusivity_m2_s)
        relaxation_before_gate_factor = _state_factor(
            state_rel_before_gate_diffusivity_m2_s,
            state_binding_diffusivity_m2_s,
        )
        relaxation_diag_factor = _state_factor(
            state_rel_diag_diffusivity_m2_s,
            state_ep_diffusivity_m2_s,
        )
        relaxation_cross_factor = _state_factor(
            state_resistance_diffusivity_m2_s,
            state_rel_diag_diffusivity_m2_s,
        )
        state_factor = _state_factor(state_resistance_diffusivity_m2_s, state_NE_diffusivity_m2_s)
        constraint_tau_s = _transport_state_constraint_tau_s(transport_state)
        constraint_length_m = _transport_state_constraint_length_m(transport_state)
        constraint_mu = _transport_state_constraint_mu(transport_state)
        atmosphere_eigenvalues = _transport_state_atmosphere_eigenvalues_kg_s(transport_state)
        contributions.append(
            StateContribution(
                state=model.state_labels[state_index],
                motif=state.motif,
                motif_kind=state.chemical_motif.kind.value,
                orientation=state.orientation,
                state_concentration_M=float(model.state_concentration_kernel.state_concentrations_M[state_index]),
                stoichiometry=_state_stoichiometry_for_index(model, state_index),
                contribution_m2_s=contribution,
                D_NE_alpha_m2_s=state_NE_diffusivity_m2_s,
                D_after_binding_alpha_m2_s=state_binding_diffusivity_m2_s,
                D_none_alpha_m2_s=state_binding_diffusivity_m2_s,
                D_ep_alpha_m2_s=state_ep_diffusivity_m2_s,
                D_rel_alpha_m2_s=state_rel_diffusivity_m2_s,
                D_rel_diag_alpha_m2_s=state_rel_diag_diffusivity_m2_s,
                D_rel_full_alpha_m2_s=state_resistance_diffusivity_m2_s,
                D_full_alpha_m2_s=state_resistance_diffusivity_m2_s,
                d_alpha_m2_s=state_resistance_diffusivity_m2_s,
                H_binding_alpha=binding_factor,
                H_atmosphere_alpha=atmosphere_factor,
                H_ep_alpha=electrophoretic_factor,
                H_rel_alpha=relaxation_factor,
                H_rel_diag_alpha=relaxation_diag_factor,
                H_rel_cross_alpha=relaxation_cross_factor,
                H_rel_before_gate_alpha=relaxation_before_gate_factor,
                H_rel_after_gate_alpha=relaxation_factor,
                H_full_alpha=atmosphere_factor,
                drag_ep_alpha=_drag_from_atmosphere_factor(electrophoretic_factor),
                drag_rel_alpha=_drag_from_atmosphere_factor(relaxation_factor),
                drag_rel_diag_alpha=_drag_from_atmosphere_factor(relaxation_diag_factor),
                drag_rel_cross_alpha=_drag_from_atmosphere_factor(relaxation_cross_factor),
                drag_rel_before_gate_alpha=_drag_from_atmosphere_factor(
                    relaxation_before_gate_factor
                ),
                drag_rel_after_gate_alpha=_drag_from_atmosphere_factor(relaxation_factor),
                drag_full_alpha=_drag_from_atmosphere_factor(atmosphere_factor),
                H_alpha=state_factor,
                constraint_tau_s=constraint_tau_s,
                constraint_length_m=constraint_length_m,
                constraint_mu=constraint_mu,
                local_resistance_trace_kg_s=math.fsum(
                    K_B * temperature_K / charged_center.local_diffusion_m2_s
                    for charged_center in transport_state.charged_centers
                ),
                binding_resistance_trace_kg_s=_transport_state_binding_resistance_trace_kg_s(
                    transport_state,
                    temperature_K,
                ),
                atmosphere_resistance_trace_kg_s=_transport_state_atmosphere_resistance_trace_kg_s(
                    transport_state,
                ),
                electrophoretic_resistance_trace_kg_s=_transport_state_component_atmosphere_trace_kg_s(
                    model,
                    transport_state,
                    "electrophoretic",
                ),
                relaxation_resistance_trace_kg_s=_transport_state_component_atmosphere_trace_kg_s(
                    model,
                    transport_state,
                    "relaxation",
                ),
                relaxation_diag_resistance_trace_kg_s=float(
                    np.trace(_transport_state_relaxation_diag_matrix_kg_s(transport_state))
                ),
                relaxation_cross_resistance_offdiag_norm_kg_s=float(
                    np.linalg.norm(_transport_state_relaxation_cross_matrix_kg_s(transport_state))
                ),
                relaxation_resistance_before_gate_trace_kg_s=(
                    _transport_state_component_atmosphere_trace_kg_s(
                        model,
                        transport_state,
                        "relaxation_before_gate",
                    )
                ),
                relaxation_resistance_after_gate_trace_kg_s=(
                    _transport_state_component_atmosphere_trace_kg_s(
                        model,
                        transport_state,
                        "relaxation",
                    )
                ),
                single_ion_atmosphere_trace_kg_s=_transport_state_single_ion_atmosphere_trace_kg_s(
                    transport_state,
                ),
                form_factor_atmosphere_trace_kg_s=_transport_state_atmosphere_before_lifetime_gate_trace_kg_s(
                    transport_state,
                ),
                atmosphere_state_lifetime_s=transport_state.atmosphere_state_lifetime_s,
                atmosphere_relaxation_time_s=transport_state.atmosphere_relaxation_time_s,
                atmosphere_lifetime_gate=transport_state.atmosphere_lifetime_gate,
                atmosphere_diagnostic_lifetime_gate=(
                    transport_state.atmosphere_diagnostic_lifetime_gate
                ),
                relaxation_dynamic_response=transport_state.relaxation_dynamic_response,
                relaxation_lifetime_gate=transport_state.relaxation_lifetime_gate,
                raw_atmosphere_form_factor=_transport_state_form_factor_cancellation(
                    transport_state,
                    model.bulk_ion_atmosphere_state.kappa_inv_m,
                ),
                effective_atmosphere_form_factor=(
                    transport_state.atmosphere_lifetime_gate
                    * _transport_state_form_factor_cancellation(
                        transport_state,
                        model.bulk_ion_atmosphere_state.kappa_inv_m,
                    )
                ),
                atmosphere_resistance_before_lifetime_gate_trace_kg_s=(
                    _transport_state_atmosphere_before_lifetime_gate_trace_kg_s(
                        transport_state,
                    )
                ),
                atmosphere_resistance_after_lifetime_gate_trace_kg_s=(
                    _transport_state_atmosphere_resistance_trace_kg_s(
                        transport_state,
                    )
                ),
                atmosphere_offdiag_norm_kg_s=_transport_state_atmosphere_offdiag_norm_kg_s(
                    transport_state,
                ),
                electrophoretic_offdiag_norm_kg_s=_transport_state_component_atmosphere_offdiag_norm_kg_s(
                    model,
                    transport_state,
                    "electrophoretic",
                ),
                relaxation_offdiag_norm_kg_s=_transport_state_component_atmosphere_offdiag_norm_kg_s(
                    model,
                    transport_state,
                    "relaxation",
                ),
                atmosphere_min_eig_kg_s=atmosphere_eigenvalues[0],
                atmosphere_max_eig_kg_s=atmosphere_eigenvalues[1],
                atmosphere_bath_basis=transport_state.atmosphere_bath_basis,
                ionic_strength_total_mol_m3=transport_state.ionic_strength_total_mol_m3,
                ionic_strength_external_mol_m3=transport_state.ionic_strength_external_mol_m3,
                external_over_total_ionic_strength=(
                    transport_state.external_over_total_ionic_strength
                ),
                resolved_charge_center_count=len(transport_state.charged_centers),
                debye_kappa_inv_A=_debye_kappa_inv_A(
                    model.bulk_ion_atmosphere_state.kappa_inv_m,
                ),
                separation_over_debye=_transport_state_separation_over_debye(
                    transport_state,
                    model.bulk_ion_atmosphere_state.kappa_inv_m,
                ),
                mean_charge_center_separation_A=_transport_state_mean_charge_center_separation_A(
                    transport_state,
                ),
                atmosphere_form_factor_cancellation=_transport_state_form_factor_cancellation(
                    transport_state,
                    model.bulk_ion_atmosphere_state.kappa_inv_m,
                ),
                thermodynamic_factor_trace=model.bulk_ion_atmosphere_state.thermodynamic_factor_trace,
                thermodynamic_factor_eigenvalues=(
                    model.bulk_ion_atmosphere_state.thermodynamic_factor_eigenvalues
                ),
                structure_factor_charge_mode=(
                    model.bulk_ion_atmosphere_state.structure_factor_charge_mode
                ),
                stationary_probability=float(model.stationary_probabilities[state_index]),
                charge=float(model.state_net_charges[state_index]),
            )
        )
    contributions.sort(key=lambda item: abs(item.contribution_m2_s), reverse=True)
    return tuple(contributions[:TOP_CONTRIBUTION_COUNT])


def _transport_state_nernst_einstein_m2_s(transport_state: TransportState) -> float:
    return float(
        math.fsum(
            charged_center.charge * charged_center.charge * charged_center.local_diffusion_m2_s
            for charged_center in transport_state.charged_centers
        )
    )


def _state_factor(
    state_resistance_diffusivity_m2_s: float,
    state_NE_diffusivity_m2_s: float,
) -> float:
    if state_NE_diffusivity_m2_s == 0.0:
        if state_resistance_diffusivity_m2_s != 0.0:
            raise ValueError("state resistance diffusivity is nonzero with zero Nernst-Einstein state diffusivity")
        return 1.0
    return state_resistance_diffusivity_m2_s / state_NE_diffusivity_m2_s


def _drag_from_atmosphere_factor(atmosphere_factor: float) -> float:
    _assert_positive_finite(atmosphere_factor, "atmosphere_factor")
    return (1.0 / atmosphere_factor) - 1.0


def _ratio_with_zero_denominator_convention(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        return math.copysign(math.inf, numerator)
    return numerator / denominator


def _transport_state_constraint_tau_s(transport_state: TransportState) -> float:
    if not transport_state.constraints:
        return 0.0
    return math.fsum(constraint.lifetime_s for constraint in transport_state.constraints) / len(transport_state.constraints)


def _transport_state_constraint_length_m(transport_state: TransportState) -> float:
    if not transport_state.constraints:
        return 0.0
    return math.fsum(constraint.length_m for constraint in transport_state.constraints) / len(transport_state.constraints)


def _transport_state_constraint_mu(transport_state: TransportState) -> float:
    if not transport_state.constraints:
        return 0.0
    center_by_label = {center.label: center for center in transport_state.charged_centers}
    mu_values = []
    for constraint in transport_state.constraints:
        relative_diffusivity_m2_s = math.fsum(
            center_by_label[label].local_diffusion_m2_s
            for label in constraint.labels
        )
        mu_values.append(relative_diffusivity_m2_s * constraint.lifetime_s / (constraint.length_m * constraint.length_m))
    return max(mu_values)


def _transport_state_binding_resistance_trace_kg_s(
    transport_state: TransportState,
    temperature_K: float,
) -> float:
    trace_kg_s = 0.0
    for constraint in transport_state.constraints:
        constraint_vector = np.asarray(constraint.vector, dtype=float)
        constraint_strength_kg_s = (
            K_B * temperature_K * constraint.lifetime_s / (constraint.length_m * constraint.length_m)
        )
        trace_kg_s += constraint_strength_kg_s * float(np.dot(constraint_vector, constraint_vector))
    return float(trace_kg_s)


def _transport_state_atmosphere_resistance_trace_kg_s(transport_state: TransportState) -> float:
    atmosphere_matrix = _transport_state_atmosphere_matrix_kg_s(transport_state)
    if atmosphere_matrix.size == 0:
        return 0.0
    return float(np.trace(atmosphere_matrix))


def _transport_state_atmosphere_before_lifetime_gate_trace_kg_s(
    transport_state: TransportState,
) -> float:
    atmosphere_matrix = _transport_state_atmosphere_before_lifetime_gate_matrix_kg_s(
        transport_state,
    )
    if atmosphere_matrix.size == 0:
        return 0.0
    return float(np.trace(atmosphere_matrix))


def _transport_state_single_ion_atmosphere_trace_kg_s(transport_state: TransportState) -> float:
    atmosphere_matrix = _transport_state_atmosphere_before_lifetime_gate_matrix_kg_s(
        transport_state,
    )
    if atmosphere_matrix.size == 0:
        return 0.0
    return float(math.fsum(float(atmosphere_matrix[center_index, center_index]) for center_index in range(atmosphere_matrix.shape[0])))


def _transport_state_atmosphere_offdiag_norm_kg_s(transport_state: TransportState) -> float:
    atmosphere_matrix = _transport_state_atmosphere_matrix_kg_s(transport_state)
    if atmosphere_matrix.size == 0:
        return 0.0
    offdiag_matrix = atmosphere_matrix - np.diag(np.diag(atmosphere_matrix))
    return float(np.linalg.norm(offdiag_matrix))


def _transport_state_atmosphere_eigenvalues_kg_s(
    transport_state: TransportState,
) -> tuple[float, float]:
    atmosphere_matrix = _transport_state_atmosphere_matrix_kg_s(transport_state)
    if atmosphere_matrix.size == 0:
        return (0.0, 0.0)
    eigenvalues = np.linalg.eigvalsh(atmosphere_matrix)
    return (float(np.min(eigenvalues)), float(np.max(eigenvalues)))


def _transport_state_mean_charge_center_separation_A(transport_state: TransportState) -> float:
    if len(transport_state.charged_centers) < 2:
        return 0.0
    separation_sum_m = 0.0
    separation_count = 0
    for first_center_index, first_center in enumerate(transport_state.charged_centers):
        first_position = np.asarray(first_center.relative_position_m, dtype=float)
        for second_center in transport_state.charged_centers[first_center_index + 1:]:
            second_position = np.asarray(second_center.relative_position_m, dtype=float)
            separation_sum_m += float(np.linalg.norm(first_position - second_position))
            separation_count += 1
    if separation_count == 0:
        return 0.0
    return separation_sum_m / separation_count / ANGSTROM_TO_M


def _transport_state_separation_over_debye(
    transport_state: TransportState,
    kappa_inv_m: float,
) -> float:
    mean_separation_A = _transport_state_mean_charge_center_separation_A(transport_state)
    if mean_separation_A == 0.0:
        return 0.0
    debye_kappa_inv_A = _debye_kappa_inv_A(kappa_inv_m)
    if math.isinf(debye_kappa_inv_A):
        return 0.0
    return mean_separation_A / debye_kappa_inv_A


def _debye_kappa_inv_A(kappa_inv_m: float) -> float:
    if math.isinf(kappa_inv_m):
        return math.inf
    return kappa_inv_m / ANGSTROM_TO_M


def _transport_state_form_factor_cancellation(
    transport_state: TransportState,
    kappa_inv_m: float,
) -> float:
    opposite_charge_factor_sum = 0.0
    opposite_charge_pair_count = 0
    for first_center_index, first_center in enumerate(transport_state.charged_centers):
        first_position = np.asarray(first_center.relative_position_m, dtype=float)
        for second_center in transport_state.charged_centers[first_center_index + 1:]:
            if first_center.charge * second_center.charge >= 0.0:
                continue
            second_position = np.asarray(second_center.relative_position_m, dtype=float)
            separation_m = float(np.linalg.norm(first_position - second_position))
            opposite_charge_factor_sum += _debye_form_factor(separation_m, kappa_inv_m)
            opposite_charge_pair_count += 1
    if opposite_charge_pair_count == 0:
        return 0.0
    return opposite_charge_factor_sum / opposite_charge_pair_count


def _debye_form_factor(
    separation_m: float,
    kappa_inv_m: float,
) -> float:
    if separation_m == 0.0:
        return 1.0
    if math.isinf(kappa_inv_m):
        return 1.0
    return math.exp(-separation_m / kappa_inv_m)


def _transport_state_trace_average_m2_s(
    transport_state: TransportState,
    temperature_K: float,
) -> float:
    return _transport_state_trace_average_for_resistance_stage_m2_s(
        transport_state,
        temperature_K,
        "atmosphere",
    )


def _transport_state_trace_average_for_resistance_stage_m2_s(
    transport_state: TransportState,
    temperature_K: float,
    resistance_stage: str,
) -> float:
    return math.fsum(
        _transport_state_axis_value_m2_s(
            transport_state,
            temperature_K,
            axis_index,
            resistance_stage,
        )
        for axis_index in range(3)
    ) / ISOTROPIC_TRACE_DIVISOR


def _transport_state_axis_value_m2_s(
    transport_state: TransportState,
    temperature_K: float,
    axis_index: int,
    resistance_stage: str,
) -> float:
    if axis_index < 0 or axis_index >= 3:
        raise ValueError(f"axis_index must be 0, 1, or 2; got {axis_index}")
    if resistance_stage not in {"local", "binding", "atmosphere"}:
        raise ValueError(f"Unsupported resistance_stage {resistance_stage}")
    charge_vector = np.asarray(
        [center.charge for center in transport_state.charged_centers],
        dtype=float,
    )
    if charge_vector.size == 0:
        return 0.0
    resistance_matrix = np.zeros((charge_vector.size, charge_vector.size), dtype=float)
    for center_index, charged_center in enumerate(transport_state.charged_centers):
        resistance_matrix[center_index, center_index] = (
            K_B * temperature_K / charged_center.local_diffusion_m2_s
        )
    if resistance_stage in {"binding", "atmosphere"}:
        for constraint in transport_state.constraints:
            constraint_vector = np.asarray(constraint.vector, dtype=float)
            constraint_strength_kg_s = (
                K_B * temperature_K * constraint.lifetime_s / (constraint.length_m * constraint.length_m)
            )
            resistance_matrix += constraint_strength_kg_s * np.outer(
                constraint_vector,
                constraint_vector,
            )
    if resistance_stage == "atmosphere":
        resistance_matrix += _transport_state_atmosphere_matrix_kg_s(transport_state)
    diffusion_matrix = K_B * temperature_K * np.linalg.inv(resistance_matrix)
    return float(charge_vector @ diffusion_matrix @ charge_vector)


def _transport_state_trace_average_for_atmosphere_component_m2_s(
    model: KernelDerivedMarkovModel,
    transport_state: TransportState,
    temperature_K: float,
    atmosphere_component: str,
) -> float:
    component_atmosphere_matrix_kg_s = _transport_state_component_atmosphere_matrix_kg_s(
        model,
        transport_state,
        atmosphere_component,
    )
    return math.fsum(
        _transport_state_axis_value_with_component_atmosphere_m2_s(
            transport_state,
            temperature_K,
            axis_index,
            component_atmosphere_matrix_kg_s,
        )
        for axis_index in range(3)
    ) / ISOTROPIC_TRACE_DIVISOR


def _transport_state_axis_value_with_component_atmosphere_m2_s(
    transport_state: TransportState,
    temperature_K: float,
    axis_index: int,
    component_atmosphere_matrix_kg_s: np.ndarray,
) -> float:
    if axis_index < 0 or axis_index >= 3:
        raise ValueError(f"axis_index must be 0, 1, or 2; got {axis_index}")
    charge_vector = np.asarray(
        [center.charge for center in transport_state.charged_centers],
        dtype=float,
    )
    if charge_vector.size == 0:
        return 0.0
    if component_atmosphere_matrix_kg_s.shape != (charge_vector.size, charge_vector.size):
        raise ValueError(
            f"{transport_state.label}.component_atmosphere_matrix_kg_s shape mismatch"
        )
    resistance_matrix = np.zeros((charge_vector.size, charge_vector.size), dtype=float)
    for center_index, charged_center in enumerate(transport_state.charged_centers):
        resistance_matrix[center_index, center_index] = (
            K_B * temperature_K / charged_center.local_diffusion_m2_s
        )
    for constraint in transport_state.constraints:
        constraint_vector = np.asarray(constraint.vector, dtype=float)
        constraint_strength_kg_s = (
            K_B * temperature_K * constraint.lifetime_s / (constraint.length_m * constraint.length_m)
        )
        resistance_matrix += constraint_strength_kg_s * np.outer(
            constraint_vector,
            constraint_vector,
        )
    resistance_matrix += component_atmosphere_matrix_kg_s
    diffusion_matrix = K_B * temperature_K * np.linalg.inv(resistance_matrix)
    return float(charge_vector @ diffusion_matrix @ charge_vector)


def _transport_state_atmosphere_matrix_kg_s(transport_state: TransportState) -> np.ndarray:
    center_count = len(transport_state.charged_centers)
    atmosphere_matrix = np.asarray(transport_state.atmosphere_resistance_kg_s, dtype=float)
    if center_count == 0:
        return np.zeros((0, 0), dtype=float)
    if atmosphere_matrix.shape != (center_count, center_count):
        raise ValueError(f"{transport_state.label}.atmosphere_resistance_kg_s shape mismatch")
    return atmosphere_matrix


def _transport_state_component_atmosphere_matrix_kg_s(
    model: KernelDerivedMarkovModel,
    transport_state: TransportState,
    atmosphere_component: str,
) -> np.ndarray:
    center_count = len(transport_state.charged_centers)
    if center_count == 0:
        return np.zeros((0, 0), dtype=float)
    if atmosphere_component == "relaxation_diag_after_ep":
        return _transport_state_relaxation_diag_after_ep_matrix_kg_s(
            model,
            transport_state,
        )
    if atmosphere_component == "relaxation_full_after_ep":
        return _transport_state_atmosphere_matrix_kg_s(transport_state)
    if atmosphere_component == "relaxation":
        return _transport_state_relaxation_after_gate_matrix_kg_s(transport_state)
    if atmosphere_component == "relaxation_before_gate":
        return _transport_state_relaxation_before_gate_matrix_kg_s(transport_state)
    component_bulk_matrix_kg_s = _bulk_component_resistance_matrix_kg_s(
        model,
        atmosphere_component,
    )
    carrier_index_by_label = {
        carrier_label: carrier_index
        for carrier_index, carrier_label in enumerate(model.bulk_ion_atmosphere_state.carrier_labels)
    }
    single_center_resistance_values_kg_s: list[float] = []
    for charged_center in transport_state.charged_centers:
        projection_vector = _audit_charged_center_bulk_projection_vector(
            model,
            charged_center,
            carrier_index_by_label,
        )
        projected_resistance_kg_s = float(
            projection_vector
            @ component_bulk_matrix_kg_s
            @ projection_vector
        )
        _assert_nonnegative_finite(
            projected_resistance_kg_s,
            f"{transport_state.label}.{charged_center.label}.{atmosphere_component}_resistance_kg_s",
        )
        single_center_resistance_values_kg_s.append(projected_resistance_kg_s)
    return _audit_form_factor_atmosphere_matrix_kg_s(
        transport_state,
        tuple(single_center_resistance_values_kg_s),
        model.bulk_ion_atmosphere_state.kappa_inv_m,
    )


def _bulk_component_resistance_matrix_kg_s(
    model: KernelDerivedMarkovModel,
    atmosphere_component: str,
) -> np.ndarray:
    if atmosphere_component == "electrophoretic":
        return np.asarray(model.bulk_ion_atmosphere_state.resistance_ep_kg_s, dtype=float)
    raise ValueError(f"Unsupported atmosphere_component {atmosphere_component}")


def _audit_charged_center_bulk_projection_vector(
    model: KernelDerivedMarkovModel,
    charged_center: ChargedCenter,
    carrier_index_by_label: Mapping[str, int],
) -> np.ndarray:
    projection_vector = np.zeros(len(carrier_index_by_label), dtype=float)
    if charged_center.charge > 0.0:
        cation_carrier_label = _single_cation_carrier_label(model)
        carrier_index = _require_mapping_int(
            carrier_index_by_label,
            cation_carrier_label,
            "carrier_index_by_label",
        )
        projection_vector[carrier_index] = 1.0
        return projection_vector
    if charged_center.label.startswith("weighted_anion"):
        return _audit_weighted_anion_projection_vector(model, carrier_index_by_label)
    anion_carrier_label = _anion_carrier_label_for_center(model, charged_center.label)
    carrier_index = _require_mapping_int(
        carrier_index_by_label,
        anion_carrier_label,
        "carrier_index_by_label",
    )
    projection_vector[carrier_index] = 1.0
    return projection_vector


def _single_cation_carrier_label(model: KernelDerivedMarkovModel) -> str:
    candidate_labels = tuple(
        carrier_label
        for carrier_label in model.bulk_ion_atmosphere_state.carrier_labels
        if not carrier_label.startswith("anion_site_")
    )
    if len(candidate_labels) != 1:
        raise ValueError(f"expected exactly one cation carrier label, got {candidate_labels}")
    return candidate_labels[0]


def _anion_carrier_label_for_center(
    model: KernelDerivedMarkovModel,
    center_label: str,
) -> str:
    candidate_labels = tuple(
        carrier_label
        for carrier_label in model.bulk_ion_atmosphere_state.carrier_labels
        if carrier_label.startswith("anion_site_") and center_label.startswith(carrier_label)
    )
    if len(candidate_labels) != 1:
        raise ValueError(f"could not map charged center {center_label} to one anion carrier")
    return candidate_labels[0]


def _audit_weighted_anion_projection_vector(
    model: KernelDerivedMarkovModel,
    carrier_index_by_label: Mapping[str, int],
) -> np.ndarray:
    projection_vector = np.zeros(len(carrier_index_by_label), dtype=float)
    anion_species_labels = tuple(
        species_label
        for species_label in model.state_concentration_kernel.species_labels
        if species_label.startswith("anion_site_")
    )
    if not anion_species_labels:
        raise ValueError("weighted anion projection requires at least one anion species label")
    free_activity_sum_M = math.fsum(
        _require_mapping_float(
            model.state_concentration_kernel.free_activities_M,
            anion_species_label,
            "state_concentration_kernel.free_activities_M",
        )
        for anion_species_label in anion_species_labels
    )
    _assert_positive_finite(free_activity_sum_M, "weighted_anion_projection_free_activity_sum_M")
    for anion_species_label in anion_species_labels:
        carrier_index = _require_mapping_int(
            carrier_index_by_label,
            anion_species_label,
            "carrier_index_by_label",
        )
        projection_vector[carrier_index] = (
            _require_mapping_float(
                model.state_concentration_kernel.free_activities_M,
                anion_species_label,
                "state_concentration_kernel.free_activities_M",
            )
            / free_activity_sum_M
        )
    return projection_vector


def _audit_form_factor_atmosphere_matrix_kg_s(
    transport_state: TransportState,
    single_center_resistance_values_kg_s: tuple[float, ...],
    kappa_inv_m: float,
) -> np.ndarray:
    center_count = len(transport_state.charged_centers)
    single_center_resistance = np.asarray(single_center_resistance_values_kg_s, dtype=float)
    if single_center_resistance.shape != (center_count,):
        raise ValueError("single_center_resistance_values_kg_s length must match charged centers")
    atmosphere_matrix = np.zeros((center_count, center_count), dtype=float)
    for first_center_index, first_center in enumerate(transport_state.charged_centers):
        atmosphere_matrix[first_center_index, first_center_index] = single_center_resistance[
            first_center_index
        ]
        for second_center_index in range(first_center_index + 1, center_count):
            second_center = transport_state.charged_centers[second_center_index]
            first_position = np.asarray(first_center.relative_position_m, dtype=float)
            second_position = np.asarray(second_center.relative_position_m, dtype=float)
            center_distance_m = float(np.linalg.norm(first_position - second_position))
            form_factor = _debye_form_factor(center_distance_m, kappa_inv_m)
            sign_product = math.copysign(1.0, first_center.charge * second_center.charge)
            coupling_resistance_kg_s = (
                sign_product
                * math.sqrt(
                    float(single_center_resistance[first_center_index])
                    * float(single_center_resistance[second_center_index])
                )
                * form_factor
            )
            atmosphere_matrix[first_center_index, second_center_index] = coupling_resistance_kg_s
            atmosphere_matrix[second_center_index, first_center_index] = coupling_resistance_kg_s
    return atmosphere_matrix


def _transport_state_component_atmosphere_trace_kg_s(
    model: KernelDerivedMarkovModel,
    transport_state: TransportState,
    atmosphere_component: str,
) -> float:
    component_atmosphere_matrix = _transport_state_component_atmosphere_matrix_kg_s(
        model,
        transport_state,
        atmosphere_component,
    )
    if component_atmosphere_matrix.size == 0:
        return 0.0
    return float(np.trace(component_atmosphere_matrix))


def _transport_state_component_atmosphere_offdiag_norm_kg_s(
    model: KernelDerivedMarkovModel,
    transport_state: TransportState,
    atmosphere_component: str,
) -> float:
    component_atmosphere_matrix = _transport_state_component_atmosphere_matrix_kg_s(
        model,
        transport_state,
        atmosphere_component,
    )
    if component_atmosphere_matrix.size == 0:
        return 0.0
    offdiag_matrix = component_atmosphere_matrix - np.diag(np.diag(component_atmosphere_matrix))
    return float(np.linalg.norm(offdiag_matrix))


def _transport_state_relaxation_before_gate_matrix_kg_s(
    transport_state: TransportState,
) -> np.ndarray:
    center_count = len(transport_state.charged_centers)
    relaxation_matrix = np.asarray(
        transport_state.relaxation_resistance_before_gate_kg_s,
        dtype=float,
    )
    if center_count == 0:
        return np.zeros((0, 0), dtype=float)
    if relaxation_matrix.shape != (center_count, center_count):
        raise ValueError(
            f"{transport_state.label}.relaxation_resistance_before_gate_kg_s shape mismatch"
        )
    return relaxation_matrix


def _transport_state_relaxation_after_gate_matrix_kg_s(
    transport_state: TransportState,
) -> np.ndarray:
    center_count = len(transport_state.charged_centers)
    relaxation_matrix = np.asarray(
        transport_state.relaxation_resistance_after_gate_kg_s,
        dtype=float,
    )
    if center_count == 0:
        return np.zeros((0, 0), dtype=float)
    if relaxation_matrix.shape != (center_count, center_count):
        raise ValueError(
            f"{transport_state.label}.relaxation_resistance_after_gate_kg_s shape mismatch"
        )
    return relaxation_matrix


def _transport_state_relaxation_diag_matrix_kg_s(
    transport_state: TransportState,
) -> np.ndarray:
    relaxation_matrix = _transport_state_relaxation_after_gate_matrix_kg_s(transport_state)
    if relaxation_matrix.size == 0:
        return np.zeros((0, 0), dtype=float)
    return np.diag(np.diag(relaxation_matrix))


def _transport_state_relaxation_cross_matrix_kg_s(
    transport_state: TransportState,
) -> np.ndarray:
    relaxation_matrix = _transport_state_relaxation_after_gate_matrix_kg_s(transport_state)
    if relaxation_matrix.size == 0:
        return np.zeros((0, 0), dtype=float)
    return relaxation_matrix - np.diag(np.diag(relaxation_matrix))


def _transport_state_relaxation_diag_after_ep_matrix_kg_s(
    model: KernelDerivedMarkovModel,
    transport_state: TransportState,
) -> np.ndarray:
    electrophoretic_matrix = _transport_state_component_atmosphere_matrix_kg_s(
        model,
        transport_state,
        "electrophoretic",
    )
    return electrophoretic_matrix + _transport_state_relaxation_diag_matrix_kg_s(transport_state)


def _transport_state_atmosphere_before_lifetime_gate_matrix_kg_s(
    transport_state: TransportState,
) -> np.ndarray:
    center_count = len(transport_state.charged_centers)
    atmosphere_matrix = np.asarray(
        transport_state.atmosphere_resistance_before_lifetime_gate_kg_s,
        dtype=float,
    )
    if center_count == 0:
        return np.zeros((0, 0), dtype=float)
    if atmosphere_matrix.shape != (center_count, center_count):
        raise ValueError(
            f"{transport_state.label}.atmosphere_resistance_before_lifetime_gate_kg_s "
            "shape mismatch"
        )
    return atmosphere_matrix


def _top_edge_contributions(
    model: KernelDerivedMarkovModel,
    result: FiniteMarkovConductivityResult,
) -> tuple[EdgeContribution, ...]:
    contributions: list[EdgeContribution] = []
    for edge in model.markov_additive_edges:
        corrected_displacement = tuple(
            edge.displacement_m[axis_index]
            + result.poisson_correctors_m[edge.target_index, axis_index]
            - result.poisson_correctors_m[edge.source_index, axis_index]
            for axis_index in range(3)
        )
        raw_delta2_m2 = math.fsum(component * component for component in edge.displacement_m)
        corrected_delta2_m2 = math.fsum(component * component for component in corrected_displacement)
        contribution = float(
            model.state_concentrations_mol_m3[edge.source_index]
            * edge.rate_s_inv
            * corrected_delta2_m2
            / (VECTOR_GREEN_KUBO_DIVISOR * _model_cation_concentration_mol_m3(model))
        )
        contributions.append(
            EdgeContribution(
                source_state=model.state_labels[edge.source_index],
                target_state=model.state_labels[edge.target_index],
                contribution_m2_s=contribution,
                rate_s_inv=float(edge.rate_s_inv),
                raw_delta2_m2=float(raw_delta2_m2),
                corrected_delta2_m2=float(corrected_delta2_m2),
            )
        )
    contributions.sort(key=lambda item: abs(item.contribution_m2_s), reverse=True)
    return tuple(contributions[:TOP_CONTRIBUTION_COUNT])


def _motif_population_rollup(model: KernelDerivedMarkovModel) -> dict[str, float]:
    populations: defaultdict[str, float] = defaultdict(float)
    for motif in model.chemical_motifs:
        population = _require_mapping_float(
            model.chemical_motif_populations,
            motif.label,
            "chemical_motif_populations",
        )
        if motif.kind is ChemicalMotifKind.SOLVENT_CAGE:
            populations["free_or_cage"] += population
        elif motif.kind is ChemicalMotifKind.FREE_ANION:
            populations["free_or_cage"] += population
            populations[f"free_anion:{_feature_id_for_motif(motif)}"] += population
        elif motif.kind is ChemicalMotifKind.ADDITIVE_COORDINATED:
            populations["additive_coordinated"] += population
        elif motif.kind is ChemicalMotifKind.SSIP:
            populations["SSIP_total"] += population
            populations[f"SSIP:{_feature_id_for_motif(motif)}"] += population
        elif motif.kind is ChemicalMotifKind.ADDITIVE_SSIP:
            populations["SSIP_total"] += population
            populations["additive_SSIP_total"] += population
            populations[f"SSIP:{_feature_id_for_motif(motif)}"] += population
        elif motif.kind is ChemicalMotifKind.CIP:
            populations["CIP_total"] += population
            populations[f"CIP:{_feature_id_for_motif(motif)}"] += population
        elif motif.kind is ChemicalMotifKind.AGGREGATE:
            populations["aggregate"] += population
        elif motif.kind is ChemicalMotifKind.BRIDGE_NETWORK:
            populations["bridge_network"] += population
            populations[f"bridge_network:{_feature_id_for_motif(motif)}"] += population
        elif motif.kind in (
            ChemicalMotifKind.LI2A_PLUS,
            ChemicalMotifKind.LIA2_MINUS,
            ChemicalMotifKind.LI2A2_NEUTRAL,
        ):
            populations["bridge_network"] += population
            populations[f"bridge_network:{_feature_id_for_motif(motif)}"] += population
            if motif.kind is ChemicalMotifKind.LI2A2_NEUTRAL:
                populations["aggregate"] += population
        else:
            raise ValueError(f"Unhandled motif kind {motif.kind.value}")
    return dict(populations)


def _motif_lifetimes_s(model: KernelDerivedMarkovModel) -> dict[str, float]:
    lifetimes: dict[str, float] = {}
    offdiag_generator = np.array(model.generator_s_inv, dtype=float)
    np.fill_diagonal(offdiag_generator, 0.0)
    for motif in model.chemical_motifs:
        state_indices = [index for index, state in enumerate(model.states) if state.motif == motif.label]
        exit_flux_s_inv = 0.0
        motif_probability = 0.0
        for source_index in state_indices:
            motif_probability += float(model.stationary_probabilities[source_index])
            for target_index, target_state in enumerate(model.states):
                if target_state.motif == motif.label:
                    continue
                exit_flux_s_inv += float(
                    model.stationary_probabilities[source_index]
                    * offdiag_generator[source_index, target_index]
                )
        if exit_flux_s_inv > 0.0:
            lifetimes[motif.label] = motif_probability / exit_flux_s_inv
    return lifetimes


def _mean_lifetime_for_motif_kinds(
    model: KernelDerivedMarkovModel,
    motif_lifetimes_s: Mapping[str, float],
    motif_kinds: tuple[ChemicalMotifKind, ...],
) -> float:
    weighted_sum = 0.0
    population_sum = 0.0
    for motif in model.chemical_motifs:
        if motif.kind not in motif_kinds:
            continue
        if motif.label not in motif_lifetimes_s:
            continue
        population = _require_mapping_float(
            model.chemical_motif_populations,
            motif.label,
            "chemical_motif_populations",
        )
        weighted_sum += population * motif_lifetimes_s[motif.label]
        population_sum += population
    if population_sum <= 0.0:
        return 0.0
    return weighted_sum / population_sum


def _pair_lifetimes_by_feature_s(
    model: KernelDerivedMarkovModel,
    motif_lifetimes_s: Mapping[str, float],
) -> dict[str, float]:
    weighted_sum_by_feature: defaultdict[str, float] = defaultdict(float)
    population_by_feature: defaultdict[str, float] = defaultdict(float)
    for motif in model.chemical_motifs:
        if motif.kind is not ChemicalMotifKind.CIP:
            continue
        if motif.label not in motif_lifetimes_s:
            continue
        feature_id = _feature_id_for_motif(motif)
        population = _require_mapping_float(
            model.chemical_motif_populations,
            motif.label,
            "chemical_motif_populations",
        )
        weighted_sum_by_feature[feature_id] += population * motif_lifetimes_s[motif.label]
        population_by_feature[feature_id] += population
    lifetimes: dict[str, float] = {}
    for feature_id, population in population_by_feature.items():
        if population <= 0.0:
            continue
        lifetimes[feature_id] = weighted_sum_by_feature[feature_id] / population
    return lifetimes


def _weighted_edge_delta2_m2(
    model: KernelDerivedMarkovModel,
    poisson_correctors_m: np.ndarray | None,
) -> float:
    weight_sum = 0.0
    weighted_delta2_sum = 0.0
    for edge in model.markov_additive_edges:
        edge_weight = model.stationary_probabilities[edge.source_index] * edge.rate_s_inv
        if poisson_correctors_m is None:
            displacement = edge.displacement_m
        else:
            displacement = tuple(
                edge.displacement_m[axis_index]
                + poisson_correctors_m[edge.target_index, axis_index]
                - poisson_correctors_m[edge.source_index, axis_index]
                for axis_index in range(3)
            )
        weighted_delta2_sum += edge_weight * math.fsum(component * component for component in displacement)
        weight_sum += edge_weight
    _assert_positive_finite(weight_sum, "transition weight sum")
    return float(weighted_delta2_sum / weight_sum)


def _ionic_strength_mol_m3(model: KernelDerivedMarkovModel) -> float:
    return model.mixture_audit.cation_concentration_mol_m3


def _concentration_weighted_external_ionic_strength_mol_m3(
    model: KernelDerivedMarkovModel,
) -> float:
    concentration_sum_mol_m3 = float(np.sum(model.state_concentrations_mol_m3))
    _assert_positive_finite(concentration_sum_mol_m3, "state concentration sum")
    weighted_external_ionic_strength = math.fsum(
        float(model.state_concentrations_mol_m3[state_index])
        * model.transport_states[state_index].ionic_strength_external_mol_m3
        for state_index in range(len(model.transport_states))
    )
    return float(weighted_external_ionic_strength / concentration_sum_mol_m3)


def _concentration_weighted_total_ionic_strength_mol_m3(
    model: KernelDerivedMarkovModel,
) -> float:
    concentration_sum_mol_m3 = float(np.sum(model.state_concentrations_mol_m3))
    _assert_positive_finite(concentration_sum_mol_m3, "state concentration sum")
    weighted_total_ionic_strength = math.fsum(
        float(model.state_concentrations_mol_m3[state_index])
        * model.transport_states[state_index].ionic_strength_total_mol_m3
        for state_index in range(len(model.transport_states))
    )
    return float(weighted_total_ionic_strength / concentration_sum_mol_m3)


def _state_concentration_weighted_relaxation_gate(model: KernelDerivedMarkovModel) -> float:
    concentration_sum_mol_m3 = float(np.sum(model.state_concentrations_mol_m3))
    _assert_positive_finite(concentration_sum_mol_m3, "state concentration sum")
    weighted_relaxation_gate = math.fsum(
        float(model.state_concentrations_mol_m3[state_index])
        * model.transport_states[state_index].relaxation_lifetime_gate
        for state_index in range(len(model.transport_states))
    )
    gate_average = weighted_relaxation_gate / concentration_sum_mol_m3
    if abs(gate_average - 1.0) <= BASIS_SUM_TOLERANCE:
        return 1.0
    if abs(gate_average) <= BASIS_SUM_TOLERANCE:
        return 0.0
    if gate_average < 0.0 or gate_average > 1.0 or not math.isfinite(gate_average):
        raise ValueError(f"mean relaxation lifetime gate must be in [0, 1], got {gate_average}")
    return float(gate_average)


def _top_state_resolved_charge_count(
    top_state_contributions: Sequence[StateContribution],
) -> int:
    if not top_state_contributions:
        return 0
    return top_state_contributions[0].resolved_charge_center_count


def _salt_family(
    canonical_recipe: RecipeDict,
) -> str:
    ionic_names = set()
    for species_name, molarity in canonical_recipe["salts"].items():
        if molarity > 0.0:
            ionic_names.add(species_name)
    if not ionic_names:
        raise ValueError("salt_family cannot be formed without ionic sources")
    return "+".join(sorted(ionic_names))


def _solvent_family(canonical_recipe: RecipeDict) -> str:
    names = [
        species_name
        for species_name, fraction in canonical_recipe["solvents"].items()
        if fraction > 0.0
    ]
    if not names:
        raise ValueError("solvent_family cannot be formed without solvents")
    return "+".join(sorted(names))


def _additive_basis(canonical_recipe: RecipeDict) -> str:
    names = [
        species_name
        for species_name, fraction in canonical_recipe["additives"].items()
        if fraction > 0.0
    ]
    if not names:
        return "none"
    return "+".join(sorted(names))


def _solvent_blend_viscosity_cP(solvents: Mapping[str, float]) -> float:
    log_viscosity = 0.0
    fraction_sum = 0.0
    for species_name, fraction in solvents.items():
        props = _require_species(SOLVENTS, species_name, "solvent")
        viscosity_cP = require_float(props, "viscosity_cP", f"solvent {species_name}")
        _assert_positive_finite(viscosity_cP, f"solvent {species_name} viscosity_cP")
        log_viscosity += fraction * math.log(viscosity_cP)
        fraction_sum += fraction
    _assert_positive_finite(fraction_sum, "solvent fraction sum")
    return math.exp(log_viscosity / fraction_sum)


def _volume_role_additives_to_weight_fractions(
    solvent_role_additives: Mapping[str, float],
    canonical_solvents: Mapping[str, float],
) -> dict[str, float]:
    if not solvent_role_additives:
        return {}
    mass_parts: dict[str, float] = {}
    for species_name, fraction in canonical_solvents.items():
        props = _require_species(SOLVENTS, species_name, "solvent")
        density_g_ml = require_float(props, "density_g_ml", f"solvent {species_name}")
        mass_parts[species_name] = fraction * density_g_ml
    for species_name, fraction in solvent_role_additives.items():
        props = _require_species(ADDITIVES, species_name, "additive")
        density_g_ml = require_float(props, "density_g_ml", f"additive {species_name}")
        mass_parts[species_name] = fraction * density_g_ml
    total_mass_part = math.fsum(mass_parts.values())
    _assert_positive_finite(total_mass_part, "volume-role neutral mass parts")
    converted: dict[str, float] = {}
    for species_name in solvent_role_additives:
        converted[species_name] = mass_parts[species_name] / total_mass_part
    return converted


def _ionic_additive_salts_to_weight_fractions(
    ionic_additive_molarities: Mapping[str, float],
    canonical_solvents: Mapping[str, float],
    canonical_salts: Mapping[str, float],
    canonical_additives: Mapping[str, float],
) -> dict[str, float]:
    if not ionic_additive_molarities:
        return {}
    solvent_mass_g = 0.0
    for species_name, volume_fraction in canonical_solvents.items():
        props = _require_species(SOLVENTS, species_name, "solvent")
        density_g_ml = require_float(props, "density_g_ml", f"solvent {species_name}")
        solvent_mass_g += volume_fraction * MILLILITER_PER_LITER * density_g_ml

    salt_mass_g = 0.0
    for species_name, molarity_M in canonical_salts.items():
        props = _require_species(SALTS, species_name, "salt")
        molecular_weight_g_mol = require_float(props, "molecular_weight", f"salt {species_name}")
        salt_mass_g += molarity_M * molecular_weight_g_mol

    ionic_additive_mass_g_by_species: dict[str, float] = {}
    for species_name, molarity_M in ionic_additive_molarities.items():
        props = _require_species(ADDITIVES, species_name, "additive")
        molecular_weight_g_mol = require_float(props, "molecular_weight", f"additive {species_name}")
        ionic_additive_mass_g_by_species[species_name] = molarity_M * molecular_weight_g_mol

    existing_additive_weight_fraction = math.fsum(canonical_additives.values())
    if existing_additive_weight_fraction >= 1.0:
        raise ValueError(
            f"existing additive weight fraction {existing_additive_weight_fraction} leaves no mass basis"
        )
    total_mass_before_existing_additives_g = (
        solvent_mass_g
        + salt_mass_g
        + math.fsum(ionic_additive_mass_g_by_species.values())
    )
    total_mass_g = total_mass_before_existing_additives_g / (1.0 - existing_additive_weight_fraction)
    _assert_positive_finite(total_mass_g, "canonical total mass for ionic additive conversion")
    converted: dict[str, float] = {}
    for species_name, mass_g in ionic_additive_mass_g_by_species.items():
        converted[species_name] = mass_g / total_mass_g
    return converted


def _dataset_metrics(ledger_rows: Sequence[BiasLedgerRow]) -> dict[str, float]:
    if not ledger_rows:
        raise ValueError("cannot compute dataset metrics with zero evaluated rows")
    y_true = np.asarray([row.sigma_exp_mS_cm for row in ledger_rows], dtype=float)
    y_pred = np.asarray([row.sigma_pred_mS_cm for row in ledger_rows], dtype=float)
    residuals = y_pred - y_true
    total_sum_squares = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if total_sum_squares <= 0.0:
        raise ValueError("cannot compute R2 with zero empirical variance")
    if len(ledger_rows) <= 1:
        raise ValueError("cannot compute Pearson correlation with fewer than two evaluated rows")
    residual_sum_squares = float(np.sum(residuals * residuals))
    return {
        "mae_mS_cm": float(np.mean(np.abs(residuals))),
        "rmse_mS_cm": float(math.sqrt(float(np.mean(residuals * residuals)))),
        "bias_mS_cm": float(np.mean(residuals)),
        "mape_percent": float(np.mean(np.abs(residuals / y_true)) * PERCENT),
        "r2": 1.0 - residual_sum_squares / total_sum_squares,
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def _salt_family_metrics(ledger_rows: Sequence[BiasLedgerRow]) -> dict[str, SaltFamilyMetrics]:
    rows_by_family: defaultdict[str, list[BiasLedgerRow]] = defaultdict(list)
    for ledger_row in ledger_rows:
        rows_by_family[ledger_row.salt_family].append(ledger_row)
    metrics_by_family: dict[str, SaltFamilyMetrics] = {}
    for family, family_rows in rows_by_family.items():
        residuals = np.asarray(
            [row.sigma_pred_mS_cm - row.sigma_exp_mS_cm for row in family_rows],
            dtype=float,
        )
        metrics_by_family[family] = SaltFamilyMetrics(
            count=len(family_rows),
            bias_mS_cm=float(np.mean(residuals)),
            mae_mS_cm=float(np.mean(np.abs(residuals))),
            rmse_mS_cm=float(math.sqrt(float(np.mean(residuals * residuals)))),
        )
    return metrics_by_family


def _family_atmosphere_metrics(
    ledger_rows: Sequence[BiasLedgerRow],
) -> tuple[FamilyAtmosphereMetrics, ...]:
    eta_relative_quantile_cuts = _quantile_cuts(
        tuple(ledger_row.eta_rel for ledger_row in ledger_rows)
    )
    debye_length_quantile_cuts_A = _quantile_cuts(
        tuple(_row_kappa_inv_A(ledger_row) for ledger_row in ledger_rows)
    )
    rows_by_group: defaultdict[tuple[str, str], list[BiasLedgerRow]] = defaultdict(list)
    for ledger_row in ledger_rows:
        top_state_kind = _top_state_kind(ledger_row)
        eta_relative_bin = _quantile_bin(ledger_row.eta_rel, eta_relative_quantile_cuts)
        debye_length_bin = _quantile_bin(_row_kappa_inv_A(ledger_row), debye_length_quantile_cuts_A)
        _append_family_group(rows_by_group, "salt_family", ledger_row.salt_family, ledger_row)
        _append_family_group(rows_by_group, "solvent_family", ledger_row.solvent_family, ledger_row)
        _append_family_group(rows_by_group, "additive_basis", ledger_row.additive_basis, ledger_row)
        _append_family_group(rows_by_group, "top_state_kind", top_state_kind, ledger_row)
        _append_family_group(rows_by_group, "eta_rel_bin", eta_relative_bin, ledger_row)
        _append_family_group(rows_by_group, "kappa_inv_bin", debye_length_bin, ledger_row)
        _append_family_group(
            rows_by_group,
            "salt_family_x_top_state_kind",
            f"{ledger_row.salt_family} x {top_state_kind}",
            ledger_row,
        )
        _append_family_group(
            rows_by_group,
            "solvent_family_x_top_state_kind",
            f"{ledger_row.solvent_family} x {top_state_kind}",
            ledger_row,
        )
        _append_family_group(
            rows_by_group,
            "additive_basis_x_top_state_kind",
            f"{ledger_row.additive_basis} x {top_state_kind}",
            ledger_row,
        )
        _append_family_group(
            rows_by_group,
            "solvent_family_x_additive_basis",
            f"{ledger_row.solvent_family} x {ledger_row.additive_basis}",
            ledger_row,
        )
    metrics = [
        _build_family_atmosphere_metrics(group_name, group_value, group_rows)
        for (group_name, group_value), group_rows in rows_by_group.items()
    ]
    metrics.sort(key=lambda metric: (metric.group_name, metric.group_value))
    return tuple(metrics)


def _append_family_group(
    rows_by_group: defaultdict[tuple[str, str], list[BiasLedgerRow]],
    group_name: str,
    group_value: str,
    ledger_row: BiasLedgerRow,
) -> None:
    rows_by_group[(group_name, group_value)].append(ledger_row)


def _build_family_atmosphere_metrics(
    group_name: str,
    group_value: str,
    group_rows: Sequence[BiasLedgerRow],
) -> FamilyAtmosphereMetrics:
    if not group_rows:
        raise ValueError(f"{group_name}={group_value} has no rows")
    residuals = np.asarray(
        [ledger_row.sigma_pred_mS_cm - ledger_row.sigma_exp_mS_cm for ledger_row in group_rows],
        dtype=float,
    )
    log_base_errors = tuple(_row_log_base_error(ledger_row) for ledger_row in group_rows)
    log_atmosphere_errors = tuple(_row_log_atmosphere_error(ledger_row) for ledger_row in group_rows)
    log_ep_errors = tuple(
        math.log(
            _positive_ratio(
                ledger_row.H_ep,
                ledger_row.H_atmosphere_target,
                f"row {ledger_row.row_id} H_ep/H_atmosphere_target",
            )
        )
        for ledger_row in group_rows
    )
    log_rel_errors = tuple(
        math.log(
            _positive_ratio(
                ledger_row.H_rel,
                ledger_row.H_atmosphere_target,
                f"row {ledger_row.row_id} H_rel/H_atmosphere_target",
            )
        )
        for ledger_row in group_rows
    )
    log_rel_diag_errors = tuple(
        math.log(
            _positive_ratio(
                ledger_row.H_rel_diag,
                1.0,
                f"row {ledger_row.row_id} H_rel_diag",
            )
        )
        for ledger_row in group_rows
    )
    log_rel_cross_errors = tuple(
        math.log(
            _positive_ratio(
                ledger_row.H_rel_cross,
                1.0,
                f"row {ledger_row.row_id} H_rel_cross",
            )
        )
        for ledger_row in group_rows
    )
    log_rel_before_gate_errors = tuple(
        math.log(
            _positive_ratio(
                ledger_row.H_rel_before_gate,
                ledger_row.H_atmosphere_target,
                f"row {ledger_row.row_id} H_rel_before_gate/H_atmosphere_target",
            )
        )
        for ledger_row in group_rows
    )
    log_rel_after_gate_errors = tuple(
        math.log(
            _positive_ratio(
                ledger_row.H_rel_after_gate,
                ledger_row.H_atmosphere_target,
                f"row {ledger_row.row_id} H_rel_after_gate/H_atmosphere_target",
            )
        )
        for ledger_row in group_rows
    )
    eta_relative_values = tuple(ledger_row.eta_rel for ledger_row in group_rows)
    debye_lengths_A = tuple(_row_kappa_inv_A(ledger_row) for ledger_row in group_rows)
    eta_base_error_correlation = _pearson_or_zero(eta_relative_values, log_base_errors)
    debye_atmosphere_error_correlation = _pearson_or_zero(debye_lengths_A, log_atmosphere_errors)
    mean_log_base_error = float(np.mean(np.asarray(log_base_errors, dtype=float)))
    mean_log_atmosphere_error = float(np.mean(np.asarray(log_atmosphere_errors, dtype=float)))
    return FamilyAtmosphereMetrics(
        group_name=group_name,
        group_value=group_value,
        count=len(group_rows),
        bias_mS_cm=float(np.mean(residuals)),
        mae_mS_cm=float(np.mean(np.abs(residuals))),
        rmse_mS_cm=float(math.sqrt(float(np.mean(residuals * residuals)))),
        mean_log_sigma_error=float(np.mean(np.asarray([ledger_row.log_error for ledger_row in group_rows]))),
        mean_log_base_error=mean_log_base_error,
        mean_log_atmosphere_error=mean_log_atmosphere_error,
        mean_log_ep_error=float(np.mean(np.asarray(log_ep_errors, dtype=float))),
        mean_log_rel_error=float(np.mean(np.asarray(log_rel_errors, dtype=float))),
        mean_log_rel_diag_error=float(np.mean(np.asarray(log_rel_diag_errors, dtype=float))),
        mean_log_rel_cross_error=float(np.mean(np.asarray(log_rel_cross_errors, dtype=float))),
        mean_log_rel_before_gate_error=float(
            np.mean(np.asarray(log_rel_before_gate_errors, dtype=float))
        ),
        mean_log_rel_after_gate_error=float(
            np.mean(np.asarray(log_rel_after_gate_errors, dtype=float))
        ),
        mean_H_atmosphere=float(np.mean(np.asarray([ledger_row.H_atmosphere for ledger_row in group_rows]))),
        mean_H_atmosphere_target=float(
            np.mean(np.asarray([ledger_row.H_atmosphere_target for ledger_row in group_rows]))
        ),
        mean_H_ratio=float(
            np.mean(
                np.asarray(
                    [
                        _positive_ratio(
                            ledger_row.H_atmosphere,
                            ledger_row.H_atmosphere_target,
                            f"row {ledger_row.row_id} H_atmosphere/H_atmosphere_target",
                        )
                        for ledger_row in group_rows
                    ],
                    dtype=float,
                )
            )
        ),
        mean_H_ep=float(np.mean(np.asarray([ledger_row.H_ep for ledger_row in group_rows]))),
        mean_H_rel=float(np.mean(np.asarray([ledger_row.H_rel for ledger_row in group_rows]))),
        mean_H_rel_diag=float(np.mean(np.asarray([ledger_row.H_rel_diag for ledger_row in group_rows]))),
        mean_H_rel_cross=float(np.mean(np.asarray([ledger_row.H_rel_cross for ledger_row in group_rows]))),
        mean_H_rel_before_gate=float(
            np.mean(np.asarray([ledger_row.H_rel_before_gate for ledger_row in group_rows]))
        ),
        mean_H_rel_after_gate=float(
            np.mean(np.asarray([ledger_row.H_rel_after_gate for ledger_row in group_rows]))
        ),
        mean_H_full=float(np.mean(np.asarray([ledger_row.H_full for ledger_row in group_rows]))),
        mean_r_atm_current=float(np.mean(np.asarray([ledger_row.r_atmosphere_current for ledger_row in group_rows]))),
        mean_r_atm_target=float(np.mean(np.asarray([ledger_row.r_atmosphere_target for ledger_row in group_rows]))),
        mean_r_atm_current_over_target=float(
            np.mean(np.asarray([ledger_row.r_atmosphere_current_over_target for ledger_row in group_rows]))
        ),
        mean_drag_ep_current_over_target=float(
            np.mean(np.asarray([ledger_row.drag_ep_current_over_target for ledger_row in group_rows]))
        ),
        mean_drag_rel_current_over_target=float(
            np.mean(np.asarray([ledger_row.drag_rel_current_over_target for ledger_row in group_rows]))
        ),
        mean_drag_rel_diag_current_over_target=float(
            np.mean(
                np.asarray(
                    [
                        _ratio_with_zero_denominator_convention(
                            ledger_row.drag_rel_diag,
                            ledger_row.r_atmosphere_target,
                        )
                        for ledger_row in group_rows
                    ]
                )
            )
        ),
        mean_drag_rel_cross_current_over_target=float(
            np.mean(
                np.asarray(
                    [
                        _ratio_with_zero_denominator_convention(
                            ledger_row.drag_rel_cross,
                            ledger_row.r_atmosphere_target,
                        )
                        for ledger_row in group_rows
                    ]
                )
            )
        ),
        mean_drag_rel_before_gate_current_over_target=float(
            np.mean(
                np.asarray(
                    [
                        _ratio_with_zero_denominator_convention(
                            ledger_row.drag_rel_before_gate,
                            ledger_row.r_atmosphere_target,
                        )
                        for ledger_row in group_rows
                    ]
                )
            )
        ),
        mean_drag_rel_after_gate_current_over_target=float(
            np.mean(
                np.asarray(
                    [
                        _ratio_with_zero_denominator_convention(
                            ledger_row.drag_rel_after_gate,
                            ledger_row.r_atmosphere_target,
                        )
                        for ledger_row in group_rows
                    ]
                )
            )
        ),
        mean_relaxation_lifetime_gate=float(
            np.mean(np.asarray([ledger_row.mean_relaxation_lifetime_gate for ledger_row in group_rows]))
        ),
        mean_eta_rel=float(np.mean(np.asarray(eta_relative_values, dtype=float))),
        mean_kappa_inv_A=float(np.mean(np.asarray(debye_lengths_A, dtype=float))),
        mean_ionic_strength_mol_m3=float(
            np.mean(np.asarray([ledger_row.ionic_strength_mol_m3 for ledger_row in group_rows]))
        ),
        dominant_top_state=_dominant_top_state(group_rows),
        owner=_classify_family_owner(
            mean_log_base_error,
            mean_log_atmosphere_error,
            eta_base_error_correlation,
            debye_atmosphere_error_correlation,
        ),
    )


def _row_log_atmosphere_error(ledger_row: BiasLedgerRow) -> float:
    return math.log(
        _positive_ratio(
            ledger_row.H_atmosphere,
            ledger_row.H_atmosphere_target,
            f"row {ledger_row.row_id} H_atmosphere/H_atmosphere_target",
        )
    )


def _row_log_base_error(ledger_row: BiasLedgerRow) -> float:
    _assert_positive_finite(
        ledger_row.H_atmosphere_target,
        f"row {ledger_row.row_id}.H_atmosphere_target",
    )
    return -math.log(ledger_row.H_atmosphere_target)


def _positive_ratio(
    numerator: float,
    denominator: float,
    context: str,
) -> float:
    _assert_positive_finite(numerator, f"{context}.numerator")
    _assert_positive_finite(denominator, f"{context}.denominator")
    ratio = numerator / denominator
    _assert_positive_finite(ratio, f"{context}.ratio")
    return ratio


def _top_state_kind(ledger_row: BiasLedgerRow) -> str:
    if not ledger_row.top_state_contributions:
        return "none"
    return ledger_row.top_state_contributions[0].motif_kind


def _dominant_top_state(group_rows: Sequence[BiasLedgerRow]) -> str:
    top_state_counts = Counter(_top_state_kind(ledger_row) for ledger_row in group_rows)
    if not top_state_counts:
        return "none"
    max_count = max(top_state_counts.values())
    dominant_states = sorted(
        top_state for top_state, top_state_count in top_state_counts.items()
        if top_state_count == max_count
    )
    return dominant_states[0]


def _row_kappa_inv_A(ledger_row: BiasLedgerRow) -> float:
    if not ledger_row.top_state_contributions:
        raise ValueError(f"row {ledger_row.row_id} has no top state contribution for kappa audit")
    debye_kappa_inv_A = ledger_row.top_state_contributions[0].debye_kappa_inv_A
    _assert_positive_finite(debye_kappa_inv_A, f"row {ledger_row.row_id}.debye_kappa_inv_A")
    return debye_kappa_inv_A


def _quantile_cuts(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    value_array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(value_array)):
        raise ValueError("quantile input contains non-finite value")
    quantiles = np.quantile(
        value_array,
        (LOWER_QUARTILE_FRACTION, MEDIAN_QUARTILE_FRACTION, UPPER_QUARTILE_FRACTION),
    )
    return (float(quantiles[0]), float(quantiles[1]), float(quantiles[2]))


def _quantile_bin(
    value: float,
    quantile_cut_values: tuple[float, float, float],
) -> str:
    if not math.isfinite(value):
        raise ValueError(f"quantile bin value must be finite, got {value}")
    if value <= quantile_cut_values[0]:
        return "q1"
    if value <= quantile_cut_values[1]:
        return "q2"
    if value <= quantile_cut_values[2]:
        return "q3"
    return "q4"


def _pearson_or_zero(
    first_values: Sequence[float],
    second_values: Sequence[float],
) -> float:
    if len(first_values) != len(second_values):
        raise ValueError("correlation inputs must have equal length")
    if len(first_values) < 2:
        return 0.0
    first_array = np.asarray(first_values, dtype=float)
    second_array = np.asarray(second_values, dtype=float)
    if not np.all(np.isfinite(first_array)) or not np.all(np.isfinite(second_array)):
        raise ValueError("correlation input contains non-finite value")
    if float(np.std(first_array)) == 0.0 or float(np.std(second_array)) == 0.0:
        return 0.0
    return float(np.corrcoef(first_array, second_array)[0, 1])


def _classify_family_owner(
    mean_log_base_error: float,
    mean_log_atmosphere_error: float,
    eta_base_error_correlation: float,
    debye_atmosphere_error_correlation: float,
) -> str:
    if mean_log_atmosphere_error > FAMILY_OWNER_LOG_STRONG_THRESHOLD:
        return "atmosphere_too_weak"
    if mean_log_atmosphere_error < -FAMILY_OWNER_LOG_STRONG_THRESHOLD:
        return "atmosphere_too_strong"
    if (
        abs(mean_log_base_error) > FAMILY_OWNER_LOG_STRONG_THRESHOLD
        and abs(eta_base_error_correlation) >= FAMILY_OWNER_CORRELATION_THRESHOLD
    ):
        return "microviscosity_base_mobility"
    if (
        abs(mean_log_atmosphere_error) > FAMILY_OWNER_LOG_WEAK_THRESHOLD
        and abs(debye_atmosphere_error_correlation) >= FAMILY_OWNER_CORRELATION_THRESHOLD
    ):
        return "bulk_atmosphere_kernel"
    if abs(mean_log_base_error) > FAMILY_OWNER_LOG_STRONG_THRESHOLD:
        return "base_mobility_or_speciation"
    return "mixed_or_noise"


def _max_row_sum_residual_s_inv(ledger_rows: Sequence[BiasLedgerRow]) -> float:
    if not ledger_rows:
        return 0.0
    return max(row.row_sum_residual_s_inv for row in ledger_rows)


def _max_stationary_residual_s_inv(ledger_rows: Sequence[BiasLedgerRow]) -> float:
    if not ledger_rows:
        return 0.0
    return max(row.stationary_residual_s_inv for row in ledger_rows)


def _max_detailed_balance_residual_s_inv(ledger_rows: Sequence[BiasLedgerRow]) -> float:
    if not ledger_rows:
        return 0.0
    return max(row.detailed_balance_residual_s_inv for row in ledger_rows)


def _require_recipe_sections(empirical_recipe: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    required_sections = ("solvents", "salts", "additives")
    sections: dict[str, Mapping[str, object]] = {}
    for section_name in required_sections:
        if section_name not in empirical_recipe:
            raise KeyError(f"empirical recipe missing {section_name}")
        section = empirical_recipe[section_name]
        if not isinstance(section, Mapping):
            raise TypeError(f"empirical recipe {section_name} must be a mapping")
        sections[section_name] = section
    return sections


def _require_entry(
    entry: Mapping[str, object],
    row_id: int,
) -> dict[str, Mapping[str, object]]:
    required_sections = ("recipe", "properties")
    sections: dict[str, Mapping[str, object]] = {}
    for section_name in required_sections:
        if section_name not in entry:
            raise KeyError(f"DATA[{row_id}] missing {section_name}")
        section = entry[section_name]
        if not isinstance(section, Mapping):
            raise TypeError(f"DATA[{row_id}].{section_name} must be a mapping")
        sections[section_name] = section
    return sections


def _strict_float_section(section: Mapping[str, object], context: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for species_name, raw_value in section.items():
        if not isinstance(species_name, str):
            raise TypeError(f"{context} species key must be string, got {species_name!r}")
        if not isinstance(raw_value, (int, float)):
            raise TypeError(f"{context}.{species_name} must be numeric, got {raw_value!r}")
        parsed[species_name] = float(raw_value)
    return parsed


def _accumulate_value(values: dict[str, float], species_name: str, increment: float) -> None:
    _assert_nonnegative_finite(increment, f"{species_name} increment")
    if species_name in values:
        values[species_name] += increment
    else:
        values[species_name] = increment


def _is_ionic_source(species_name: str) -> bool:
    if species_name in SALTS:
        props = SALTS[species_name]
    elif species_name in ADDITIVES:
        props = ADDITIVES[species_name]
    else:
        return False
    return _is_ionic_source_props(props)


def _is_ionic_source_props(props: Mapping[str, object]) -> bool:
    has_cation_identity = "cation" in props or "cation_radius" in props
    return has_cation_identity and "anion" in props and "Lambda_0" in props


def _require_species(
    species_table: Mapping[str, Mapping[str, object]],
    species_name: str,
    species_kind: str,
) -> Mapping[str, object]:
    if species_name not in species_table:
        raise KeyError(f"{species_kind} {species_name} missing from species table")
    return species_table[species_name]


def _require_mapping_float(
    values: Mapping[str, float],
    key: str,
    context: str,
) -> float:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    value = float(values[key])
    _assert_nonnegative_finite(value, f"{context}.{key}")
    return value


def _require_mapping_int(
    values: Mapping[str, int],
    key: str,
    context: str,
) -> int:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    return int(values[key])


def _feature_id_for_motif(motif: ChemicalMotif) -> str:
    if motif.feature_id is None:
        raise ValueError(f"motif {motif.label} is missing feature_id")
    return motif.feature_id


def _assert_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _assert_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value}")
