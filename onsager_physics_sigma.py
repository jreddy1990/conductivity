"""Auditable correlated-carrier conductivity model.

This module is a standalone conductivity prototype. It keeps the final readout
as a fixed charge-correlation quadratic form and exposes the intermediate
composition, matrix, solvation, speciation, mobility, and correlation heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from constants import BJERRUM_LENGTH_NM, R, T_REF_K
from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
from electrolyte_model import ElectrolyteRecipeModel
from utils.config_cache import load_physics_config


LITER_TO_ML = 1000.0


@dataclass(frozen=True)
class OnsagerConductivityParams:
    """Calibrated scalar parameters for the correlated-carrier prototype."""

    mobility_scale: float
    bjerrum_dielectric_scale: float = 1.0
    viscosity_exponent_scale: float = 1.0
    liquid_excess_viscosity_scale: float = 1.0
    dimer_viscosity_scale: float = 1.0
    salt_viscosity_scale: float = 1.0
    pair_correlation_gain: float = 1.0
    aggregate_correlation_gain: float = 1.0
    steric_anticorrelation_gain: float = 1.0
    salt_mobility_scales: Mapping[str, float] | None = None


@dataclass(frozen=True)
class CompositionState:
    recipe: dict[str, Any]
    temperature_K: float
    solvent_volumes_ml: dict[str, float]
    solvent_moles: dict[str, float]
    additive_masses_g: dict[str, float]
    additive_volumes_ml: dict[str, float]
    additive_moles: dict[str, float]
    additive_molarities_M: dict[str, float]
    salt_moles: dict[str, float]
    salt_molarities_M: dict[str, float]
    salt_masses_g: dict[str, float]
    salt_volumes_ml: dict[str, float]
    ionic_source_molarities_M: dict[str, float]
    neutral_liquid_volume_fractions: dict[str, float]
    species_mole_fractions: dict[str, float]
    total_mass_g: float
    density_g_ml: float
    molality_mol_kg_solvent: float


@dataclass(frozen=True)
class MatrixState:
    eta_liquid_cP: float
    eta_solution_cP: float
    epsilon_liquid: float
    epsilon_effective: float
    dimer_viscosity_factor: float
    salt_viscosity_factor: float
    dielectric_decrement_fraction: float


@dataclass(frozen=True)
class SolvationState:
    shell_fractions: dict[str, float]
    shell_coordination_strengths: dict[str, float]
    shell_steric_disruption: float
    shell_donor_number: float
    shell_binding_energy_kJ_mol: float
    preferred_coordination_number: float


@dataclass(frozen=True)
class SpeciationState:
    free_fraction_by_source: dict[str, float]
    paired_fraction_by_source: dict[str, float]
    aggregate_fraction_by_source: dict[str, float]
    association_constant_M_inv: dict[str, float]
    carrier_concentrations_M: dict[str, float]
    carrier_charges: dict[str, int]


@dataclass(frozen=True)
class MobilityState:
    carrier_lambda_S_cm2_mol: dict[str, float]
    carrier_strength_mS_cm: dict[str, float]
    source_lambda_split_S_cm2_mol: dict[str, dict[str, float]]


@dataclass(frozen=True)
class CorrelationState:
    carrier_order: list[str]
    matrix: np.ndarray
    li_anion_rho: dict[str, float]
    raw_li_anion_coupling: dict[str, float]


@dataclass(frozen=True)
class OnsagerConductivityResult:
    sigma_mS_cm: float
    sigma_uncorrelated_mS_cm: float
    composition: CompositionState
    matrix: MatrixState
    solvation: SolvationState
    speciation: SpeciationState
    mobility: MobilityState
    correlation: CorrelationState


@dataclass(frozen=True)
class CalibrationResult:
    params: OnsagerConductivityParams
    n_rows: int
    mae_mS_cm: float
    rmse_mS_cm: float


def evaluate_onsager_conductivity(
    recipe: Mapping[str, Any],
    temperature_K: float = T_REF_K,
    params: OnsagerConductivityParams | None = None,
    physics_config: Mapping[str, Any] | None = None,
) -> OnsagerConductivityResult:
    """Evaluate conductivity and expose all intermediate transport heads."""

    if params is None:
        params = OnsagerConductivityParams(mobility_scale=1.0)
    if physics_config is None:
        physics_config = load_physics_config()

    _assert_positive_float(temperature_K, "temperature_K")
    _validate_params(params)

    recipe_model = ElectrolyteRecipeModel.model_validate(dict(recipe))
    composition = _build_composition_state(recipe_model, temperature_K)
    matrix = _build_matrix_state(composition, physics_config, params)
    solvation = _build_solvation_state(composition)
    speciation = _build_speciation_state(composition, matrix, physics_config, params)
    mobility = _build_mobility_state(composition, matrix, speciation, temperature_K, params, physics_config)
    correlation = _build_correlation_state(composition, matrix, solvation, speciation, mobility, physics_config, params)
    sigma_uncorrelated, sigma = _onsager_readout(speciation, mobility, correlation)

    return OnsagerConductivityResult(
        sigma_mS_cm=sigma,
        sigma_uncorrelated_mS_cm=sigma_uncorrelated,
        composition=composition,
        matrix=matrix,
        solvation=solvation,
        speciation=speciation,
        mobility=mobility,
        correlation=correlation,
    )


def predict_onsager_conductivity_mS_cm(
    recipe: Mapping[str, Any],
    temperature_K: float = T_REF_K,
    params: OnsagerConductivityParams | None = None,
    physics_config: Mapping[str, Any] | None = None,
) -> float:
    """Return only the fixed-readout conductivity in mS/cm."""

    return evaluate_onsager_conductivity(recipe, temperature_K, params, physics_config).sigma_mS_cm


def fit_global_mobility_scale(
    entries: Sequence[Mapping[str, Any]],
    temperature_K: float = T_REF_K,
    physics_config: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    """Fit one global mobility scale to measured conductivity rows.

    Since every carrier mobility is multiplied by the same scalar, the final
    fixed-readout conductivity is linear in ``mobility_scale``. The fit is
    therefore a one-parameter least-squares regression through the origin.
    """

    if physics_config is None:
        physics_config = load_physics_config()
    _assert_positive_float(temperature_K, "temperature_K")

    predictions: list[float] = []
    measurements: list[float] = []
    unit_params = OnsagerConductivityParams(mobility_scale=1.0)

    for idx, entry in enumerate(entries):
        recipe = _require_mapping(entry, "recipe", f"entry[{idx}]")
        properties = _require_mapping(entry, "properties", f"entry[{idx}]")
        measured = _require_float(properties, "conductivity_mS_cm", f"entry[{idx}].properties")
        if measured <= 0.0:
            raise ValueError(f"entry[{idx}].properties.conductivity_mS_cm must be positive")
        prediction = evaluate_onsager_conductivity(
            recipe,
            temperature_K=temperature_K,
            params=unit_params,
            physics_config=physics_config,
        ).sigma_mS_cm
        if prediction <= 0.0 or not math.isfinite(prediction):
            raise ValueError(f"entry[{idx}] produced invalid unit-scale prediction {prediction}")
        predictions.append(prediction)
        measurements.append(measured)

    if not predictions:
        raise ValueError("Cannot fit mobility scale without measured conductivity rows")

    pred_arr = np.asarray(predictions, dtype=float)
    meas_arr = np.asarray(measurements, dtype=float)
    denom = float(np.dot(pred_arr, pred_arr))
    if denom <= 0.0:
        raise ValueError("Global mobility-scale fit has zero prediction norm")
    mobility_scale = float(np.dot(pred_arr, meas_arr) / denom)
    if mobility_scale <= 0.0 or not math.isfinite(mobility_scale):
        raise ValueError(f"Fitted invalid global mobility scale {mobility_scale}")

    residual = mobility_scale * pred_arr - meas_arr
    mae = float(np.mean(np.abs(residual)))
    rmse = float(math.sqrt(np.mean(residual * residual)))
    return CalibrationResult(
        params=OnsagerConductivityParams(mobility_scale=mobility_scale),
        n_rows=len(predictions),
        mae_mS_cm=mae,
        rmse_mS_cm=rmse,
    )


def fit_mechanism_params(
    entries: Sequence[Mapping[str, Any]],
    temperature_K: float = T_REF_K,
    physics_config: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    """Fit constrained mechanism-head parameters to measured rows.

    This keeps the final conductivity readout fixed. The calibrated quantities
    are mechanism coefficients: dielectric-pairing sensitivity, viscosity
    friction strength, salt viscosity strength, correlation gains, and
    salt-level mobility scales for salts that appear in the measured rows.
    """

    from scipy.optimize import least_squares

    if physics_config is None:
        physics_config = load_physics_config()
    _assert_positive_float(temperature_K, "temperature_K")

    rows: list[tuple[Mapping[str, Any], float]] = []
    active_sources: set[str] = set()
    for idx, entry in enumerate(entries):
        recipe = _require_mapping(entry, "recipe", f"entry[{idx}]")
        properties = _require_mapping(entry, "properties", f"entry[{idx}]")
        measured = _require_float(properties, "conductivity_mS_cm", f"entry[{idx}].properties")
        if measured <= 0.0:
            raise ValueError(f"entry[{idx}].properties.conductivity_mS_cm must be positive")
        evaluated = evaluate_onsager_conductivity(
            recipe,
            temperature_K=temperature_K,
            params=OnsagerConductivityParams(mobility_scale=1.0),
            physics_config=physics_config,
        )
        active_sources.update(evaluated.composition.ionic_source_molarities_M)
        rows.append((recipe, measured))

    if not rows:
        raise ValueError("Cannot fit mechanism parameters without measured conductivity rows")
    if not active_sources:
        raise ValueError("Cannot fit mechanism parameters without active ionic sources")

    reference_source = "LiPF6" if "LiPF6" in active_sources else sorted(active_sources)[0]
    fitted_sources = [source for source in sorted(active_sources) if source != reference_source]
    global_fit = fit_global_mobility_scale(entries, temperature_K, physics_config)

    x0 = np.zeros(9 + len(fitted_sources), dtype=float)
    x0[0] = math.log(global_fit.params.mobility_scale)

    def unpack(x: np.ndarray) -> OnsagerConductivityParams:
        salt_scales = {reference_source: 1.0}
        for offset, source in enumerate(fitted_sources, start=9):
            salt_scales[source] = float(math.exp(x[offset]))
        return OnsagerConductivityParams(
            mobility_scale=float(math.exp(x[0])),
            bjerrum_dielectric_scale=float(math.exp(x[1])),
            viscosity_exponent_scale=float(math.exp(x[2])),
            liquid_excess_viscosity_scale=float(math.exp(x[3])),
            dimer_viscosity_scale=float(math.exp(x[4])),
            salt_viscosity_scale=float(math.exp(x[5])),
            pair_correlation_gain=float(math.exp(x[6])),
            aggregate_correlation_gain=float(math.exp(x[7])),
            steric_anticorrelation_gain=float(math.exp(x[8])),
            salt_mobility_scales=salt_scales,
        )

    def residuals(x: np.ndarray) -> np.ndarray:
        params = unpack(x)
        errs = []
        for recipe, measured in rows:
            predicted = evaluate_onsager_conductivity(
                recipe,
                temperature_K=temperature_K,
                params=params,
                physics_config=physics_config,
            ).sigma_mS_cm
            errs.append(predicted - measured)
        return np.asarray(errs, dtype=float)

    fit = least_squares(residuals, x0, method="trf")
    params = unpack(fit.x)
    residual = residuals(fit.x)
    mae = float(np.mean(np.abs(residual)))
    rmse = float(math.sqrt(np.mean(residual * residual)))
    return CalibrationResult(
        params=params,
        n_rows=len(rows),
        mae_mS_cm=mae,
        rmse_mS_cm=rmse,
    )


def _build_composition_state(recipe_model: ElectrolyteRecipeModel, temperature_K: float) -> CompositionState:
    solvents = {name: float(value) for name, value in recipe_model.solvents.items()}
    salts = {name: float(value) for name, value in recipe_model.salts.items()}
    additives = {name: float(value) for name, value in recipe_model.additives.items()}

    total_salt_mass_g = 0.0
    salt_masses_g: dict[str, float] = {}
    salt_volumes_ml: dict[str, float] = {}
    salt_moles: dict[str, float] = {}
    for name, molarity in salts.items():
        props = _require_species(SALTS, name, "salt")
        _assert_nonnegative_float(molarity, f"salts.{name}")
        mw = _require_float(props, "molecular_weight", f"salt {name}")
        density = _require_float(props, "density_g_ml", f"salt {name}")
        moles = molarity
        mass = moles * mw
        volume = mass / density
        salt_moles[name] = moles
        salt_masses_g[name] = mass
        salt_volumes_ml[name] = volume
        total_salt_mass_g += mass

    non_additive_mass_fraction = 1.0 - sum(additives.values())
    if non_additive_mass_fraction <= 0.0:
        raise ValueError("Total additive weight fraction must be below 1.0")

    additive_volume_per_total_mass = 0.0
    for name, wt_fraction in additives.items():
        props = _require_species(ADDITIVES, name, "additive")
        _assert_nonnegative_float(wt_fraction, f"additives.{name}")
        density = _require_float(props, "density_g_ml", f"additive {name}")
        additive_volume_per_total_mass += wt_fraction / density

    solvent_blend_density = sum(
        frac * _require_float(_require_species(SOLVENTS, name, "solvent"), "density_g_ml", f"solvent {name}")
        for name, frac in solvents.items()
    )
    solvent_volume_numerator = (
        LITER_TO_ML
        - sum(salt_volumes_ml.values())
        - total_salt_mass_g * additive_volume_per_total_mass / non_additive_mass_fraction
    )
    solvent_volume_denominator = (
        1.0 + solvent_blend_density * additive_volume_per_total_mass / non_additive_mass_fraction
    )
    solvent_total_volume_ml = solvent_volume_numerator / solvent_volume_denominator
    if solvent_total_volume_ml <= 0.0:
        raise ValueError(
            f"Computed non-positive solvent volume {solvent_total_volume_ml} mL for recipe {recipe_model}"
        )

    solvent_volumes_ml: dict[str, float] = {}
    solvent_moles: dict[str, float] = {}
    total_solvent_mass_g = 0.0
    for name, frac in solvents.items():
        props = _require_species(SOLVENTS, name, "solvent")
        density = _require_float(props, "density_g_ml", f"solvent {name}")
        mw = _require_float(props, "molecular_weight", f"solvent {name}")
        volume = frac * solvent_total_volume_ml
        mass = volume * density
        solvent_volumes_ml[name] = volume
        solvent_moles[name] = mass / mw
        total_solvent_mass_g += mass

    total_mass_g = (total_solvent_mass_g + total_salt_mass_g) / non_additive_mass_fraction
    additive_masses_g: dict[str, float] = {}
    additive_volumes_ml: dict[str, float] = {}
    additive_moles: dict[str, float] = {}
    additive_molarities_M: dict[str, float] = {}
    for name, wt_fraction in additives.items():
        props = _require_species(ADDITIVES, name, "additive")
        density = _require_float(props, "density_g_ml", f"additive {name}")
        mw = _require_float(props, "molecular_weight", f"additive {name}")
        mass = wt_fraction * total_mass_g
        volume = mass / density
        moles = mass / mw
        additive_masses_g[name] = mass
        additive_volumes_ml[name] = volume
        additive_moles[name] = moles
        additive_molarities_M[name] = moles

    ionic_source_molarities = dict(salts)
    for name, molarity in additive_molarities_M.items():
        props = _require_species(ADDITIVES, name, "additive")
        if _is_ionic_source(props):
            ionic_source_molarities[name] = molarity

    neutral_volumes = dict(solvent_volumes_ml)
    for name, volume in additive_volumes_ml.items():
        props = _require_species(ADDITIVES, name, "additive")
        if not _is_ionic_source(props):
            neutral_volumes[name] = volume
    neutral_volume_total = sum(neutral_volumes.values())
    if neutral_volume_total <= 0.0:
        raise ValueError("Neutral liquid volume is non-positive")
    neutral_liquid_volume_fractions = {
        name: volume / neutral_volume_total for name, volume in neutral_volumes.items()
    }

    all_moles = dict(solvent_moles)
    all_moles.update(salt_moles)
    all_moles.update(additive_moles)
    total_moles = sum(all_moles.values())
    if total_moles <= 0.0:
        raise ValueError("Total recipe moles are non-positive")
    species_mole_fractions = {name: moles / total_moles for name, moles in all_moles.items()}

    return CompositionState(
        recipe={
            "solvents": solvents,
            "salts": salts,
            "additives": additives,
        },
        temperature_K=temperature_K,
        solvent_volumes_ml=solvent_volumes_ml,
        solvent_moles=solvent_moles,
        additive_masses_g=additive_masses_g,
        additive_volumes_ml=additive_volumes_ml,
        additive_moles=additive_moles,
        additive_molarities_M=additive_molarities_M,
        salt_moles=salt_moles,
        salt_molarities_M=salts,
        salt_masses_g=salt_masses_g,
        salt_volumes_ml=salt_volumes_ml,
        ionic_source_molarities_M=ionic_source_molarities,
        neutral_liquid_volume_fractions=neutral_liquid_volume_fractions,
        species_mole_fractions=species_mole_fractions,
        total_mass_g=total_mass_g,
        density_g_ml=total_mass_g / LITER_TO_ML,
        molality_mol_kg_solvent=sum(ionic_source_molarities.values()) / (total_solvent_mass_g / 1000.0),
    )


def _build_matrix_state(
    composition: CompositionState,
    physics_config: Mapping[str, Any],
    params: OnsagerConductivityParams,
) -> MatrixState:
    viscosity_cfg = _require_mapping(physics_config, "viscosity_model", "physics_config")
    dielectric_cfg = _require_mapping(physics_config, "dielectric_mixing", "physics_config")
    pair_excess = _require_mapping(dielectric_cfg, "excess_interaction_parameters", "dielectric_mixing")

    ln_eta = 0.0
    epsilon_liquid = 0.0
    neutral_names = list(composition.neutral_liquid_volume_fractions)
    for name, phi in composition.neutral_liquid_volume_fractions.items():
        props = _neutral_species_props(name)
        eta = _require_float(props, "viscosity_cP", f"neutral species {name}")
        eps = _require_float(props, "epsilon_r", f"neutral species {name}")
        if eta <= 0.0:
            raise ValueError(f"neutral species {name}.viscosity_cP must be positive")
        ln_eta += phi * math.log(eta)
        epsilon_liquid += phi * eps

    for i, name_i in enumerate(neutral_names):
        for name_j in neutral_names[i + 1 :]:
            excess_value = _optional_pair_float(
                pair_excess,
                name_i,
                name_j,
                "dielectric_mixing.excess_interaction_parameters",
            )
            if excess_value is not None:
                epsilon_liquid += (
                    excess_value
                    * composition.neutral_liquid_volume_fractions[name_i]
                    * composition.neutral_liquid_volume_fractions[name_j]
                )

    ln_eta += (
        params.liquid_excess_viscosity_scale
        * _compute_eyring_nrtl_excess_log_viscosity(composition, physics_config)
    )
    eta_liquid = math.exp(ln_eta)
    dimer_factor = _compute_dimer_viscosity_factor(composition, physics_config, params.dimer_viscosity_scale)
    salt_factor = _compute_salt_viscosity_factor(composition, viscosity_cfg, params.salt_viscosity_scale)
    eta_solution = eta_liquid * dimer_factor * salt_factor
    if eta_solution <= 0.0 or not math.isfinite(eta_solution):
        raise ValueError(f"Invalid solution viscosity {eta_solution}")

    decrement_fraction = 0.0
    for source, molarity in composition.ionic_source_molarities_M.items():
        props = _ionic_source_props(source)
        decrement_fraction += molarity * _require_float(
            props,
            "dielectric_decrement_frac_per_M",
            f"ionic source {source}",
        )
    epsilon_effective = epsilon_liquid * (1.0 - decrement_fraction)
    if epsilon_effective <= 0.0:
        raise ValueError(
            f"Effective dielectric became non-positive: epsilon_liquid={epsilon_liquid}, "
            f"dielectric_decrement_fraction={decrement_fraction}"
        )

    return MatrixState(
        eta_liquid_cP=eta_liquid,
        eta_solution_cP=eta_solution,
        epsilon_liquid=epsilon_liquid,
        epsilon_effective=epsilon_effective,
        dimer_viscosity_factor=dimer_factor,
        salt_viscosity_factor=salt_factor,
        dielectric_decrement_fraction=decrement_fraction,
    )


def _build_solvation_state(composition: CompositionState) -> SolvationState:
    strengths: dict[str, float] = {}
    for name in composition.neutral_liquid_volume_fractions:
        props = _neutral_species_props(name)
        affinity = _require_float(props, "coordination_affinity_M_inv", f"neutral species {name}")
        concentration_M = _neutral_species_moles(composition, name)
        strengths[name] = affinity * concentration_M

    total_strength = sum(strengths.values())
    if total_strength <= 0.0:
        raise ValueError("Li solvation shell has no positive coordination strength")
    shell_fractions = {name: value / total_strength for name, value in strengths.items()}

    shell_steric = 0.0
    shell_donor = 0.0
    shell_binding = 0.0
    for name, fraction in shell_fractions.items():
        props = _neutral_species_props(name)
        shell_steric += fraction * _optional_float(props, "steric_disruption_beta", f"neutral species {name}")
        shell_donor += fraction * _require_float(props, "donor_number", f"neutral species {name}")
        shell_binding += fraction * _require_float(props, "li_binding_energy_kJ_mol", f"neutral species {name}")

    weighted_cn = 0.0
    total_source_M = sum(composition.ionic_source_molarities_M.values())
    if total_source_M <= 0.0:
        raise ValueError("At least one ionic source is required for conductivity")
    for source, molarity in composition.ionic_source_molarities_M.items():
        props = _ionic_source_props(source)
        weighted_cn += (
            molarity
            / total_source_M
            * _require_float(props, "preferred_coordination_number", f"ionic source {source}")
        )

    return SolvationState(
        shell_fractions=shell_fractions,
        shell_coordination_strengths=strengths,
        shell_steric_disruption=shell_steric,
        shell_donor_number=shell_donor,
        shell_binding_energy_kJ_mol=shell_binding,
        preferred_coordination_number=weighted_cn,
    )


def _build_speciation_state(
    composition: CompositionState,
    matrix: MatrixState,
    physics_config: Mapping[str, Any],
    params: OnsagerConductivityParams,
) -> SpeciationState:
    pairing_cfg = _require_mapping(physics_config, "ion_pairing_model", "physics_config")
    eps_ref = _require_float(pairing_cfg, "bjerrum_eps_ref", "ion_pairing_model")
    aggregate_onset = _require_float(pairing_cfg, "aggregate_onset_mol_l", "ion_pairing_model")
    aggregate_scale = _require_float(pairing_cfg, "aggregate_scale_mol_l", "ion_pairing_model")
    aggregate_max = _require_float(pairing_cfg, "aggregate_max_fraction_of_paired", "ion_pairing_model")
    if aggregate_scale <= 0.0:
        raise ValueError("ion_pairing_model.aggregate_scale_mol_l must be positive")

    free_by_source: dict[str, float] = {}
    paired_by_source: dict[str, float] = {}
    aggregate_by_source: dict[str, float] = {}
    association_by_source: dict[str, float] = {}
    carrier_concentrations: dict[str, float] = {}
    carrier_charges: dict[str, int] = {}

    for source, molarity in composition.ionic_source_molarities_M.items():
        props = _ionic_source_props(source)
        cation = _require_string(props, "cation", f"ionic source {source}")
        anion = _require_string(props, "anion", f"ionic source {source}")
        anion_charge = int(_require_float(props, "anion_charge", f"ionic source {source}"))
        cation_props = _require_species(CATION_PROPERTIES, cation, "cation")
        cation_charge = int(_require_float(cation_props, "charge", f"cation {cation}"))

        cation_radius = _require_float(props, "cation_radius", f"ionic source {source}")
        anion_radius = _require_float(props, "anion_radius", f"ionic source {source}")
        contact_distance_nm = (cation_radius + anion_radius) * 0.1
        if contact_distance_nm <= 0.0:
            raise ValueError(f"ionic source {source} produced non-positive contact distance")

        ka_ref = _require_float(props, "bjerrum_K_A_ref", f"ionic source {source}")
        coulomb_term = BJERRUM_LENGTH_NM * (T_REF_K / composition.temperature_K) / contact_distance_nm
        ka = ka_ref * math.exp(
            params.bjerrum_dielectric_scale
            * coulomb_term
            * (1.0 / matrix.epsilon_effective - 1.0 / eps_ref)
        )
        if ka < 0.0 or not math.isfinite(ka):
            raise ValueError(f"ionic source {source} produced invalid association constant {ka}")

        ka_c = ka * molarity
        if ka_c == 0.0:
            free_fraction = 1.0
        else:
            free_fraction = (math.sqrt(1.0 + 4.0 * ka_c) - 1.0) / (2.0 * ka_c)
        paired_fraction = 1.0 - free_fraction
        aggregate_fraction = paired_fraction * aggregate_max / (
            1.0 + math.exp(-(molarity - aggregate_onset) / aggregate_scale)
        )

        free_by_source[source] = free_fraction
        paired_by_source[source] = paired_fraction
        aggregate_by_source[source] = aggregate_fraction
        association_by_source[source] = ka

        cation_symbol = _require_string(cation_props, "ion_symbol", f"cation {cation}")
        carrier_concentrations[cation_symbol] = carrier_concentrations.get(cation_symbol, 0.0) + molarity * free_fraction
        carrier_concentrations[anion] = carrier_concentrations.get(anion, 0.0) + molarity * free_fraction
        carrier_charges[cation_symbol] = cation_charge
        carrier_charges[anion] = anion_charge

    return SpeciationState(
        free_fraction_by_source=free_by_source,
        paired_fraction_by_source=paired_by_source,
        aggregate_fraction_by_source=aggregate_by_source,
        association_constant_M_inv=association_by_source,
        carrier_concentrations_M=carrier_concentrations,
        carrier_charges=carrier_charges,
    )


def _build_mobility_state(
    composition: CompositionState,
    matrix: MatrixState,
    speciation: SpeciationState,
    temperature_K: float,
    params: OnsagerConductivityParams,
    physics_config: Mapping[str, Any],
) -> MobilityState:
    arrhenius_cfg = _require_mapping(physics_config, "transport_arrhenius", "physics_config")
    eta_ref = _require_float(arrhenius_cfg, "reference_viscosity_cP", "transport_arrhenius")
    ea = _require_float(arrhenius_cfg, "diffusion_activation_energy_J_mol", "transport_arrhenius")
    if eta_ref <= 0.0:
        raise ValueError("transport_arrhenius.reference_viscosity_cP must be positive")

    carrier_lambda: dict[str, float] = {}
    carrier_strength: dict[str, float] = {}
    source_splits: dict[str, dict[str, float]] = {}
    temp_factor = math.exp(-ea / R * (1.0 / temperature_K - 1.0 / T_REF_K))

    for source, molarity in composition.ionic_source_molarities_M.items():
        props = _ionic_source_props(source)
        cation = _require_string(props, "cation", f"ionic source {source}")
        anion = _require_string(props, "anion", f"ionic source {source}")
        cation_props = _require_species(CATION_PROPERTIES, cation, "cation")
        cation_symbol = _require_string(cation_props, "ion_symbol", f"cation {cation}")
        lambda0_total = _require_float(props, "Lambda_0", f"ionic source {source}")
        if lambda0_total <= 0.0:
            raise ValueError(f"ionic source {source}.Lambda_0 must be positive")

        cation_radius = _require_float(cation_props, "solvated_radius_A", f"cation {cation}")
        anion_radius = _require_float(props, "anion_radius", f"ionic source {source}")
        cation_weight = 1.0 / cation_radius
        anion_weight = 1.0 / anion_radius
        lambda_cation0 = lambda0_total * cation_weight / (cation_weight + anion_weight)
        lambda_anion0 = lambda0_total - lambda_cation0

        alpha_cation = _require_float(cation_props, "stokes_einstein_alpha", f"cation {cation}")
        alpha_anion = _require_float(props, "stokes_einstein_alpha_anion", f"ionic source {source}")
        lambda_cation = (
            lambda_cation0
            * params.mobility_scale
            * _salt_mobility_scale(params, source)
            * temp_factor
            * (eta_ref / matrix.eta_solution_cP) ** (params.viscosity_exponent_scale * alpha_cation)
        )
        lambda_anion = (
            lambda_anion0
            * params.mobility_scale
            * _salt_mobility_scale(params, source)
            * temp_factor
            * (eta_ref / matrix.eta_solution_cP) ** (params.viscosity_exponent_scale * alpha_anion)
        )

        source_splits[source] = {
            cation_symbol: lambda_cation,
            anion: lambda_anion,
        }
        free_concentration = molarity * speciation.free_fraction_by_source[source]
        carrier_lambda[cation_symbol] = carrier_lambda.get(cation_symbol, 0.0) + lambda_cation
        carrier_lambda[anion] = carrier_lambda.get(anion, 0.0) + lambda_anion
        carrier_strength[cation_symbol] = carrier_strength.get(cation_symbol, 0.0) + free_concentration * lambda_cation
        carrier_strength[anion] = carrier_strength.get(anion, 0.0) + free_concentration * lambda_anion

    return MobilityState(
        carrier_lambda_S_cm2_mol=carrier_lambda,
        carrier_strength_mS_cm=carrier_strength,
        source_lambda_split_S_cm2_mol=source_splits,
    )


def _build_correlation_state(
    composition: CompositionState,
    matrix: MatrixState,
    solvation: SolvationState,
    speciation: SpeciationState,
    mobility: MobilityState,
    physics_config: Mapping[str, Any],
    params: OnsagerConductivityParams,
) -> CorrelationState:
    pairing_cfg = _require_mapping(physics_config, "ion_pairing_model", "physics_config")
    eps_ref = _require_float(pairing_cfg, "bjerrum_eps_ref", "ion_pairing_model")

    cation_symbol = _single_cation_symbol(composition)
    carrier_order = [cation_symbol] + [
        carrier for carrier in mobility.carrier_strength_mS_cm if carrier != cation_symbol
    ]
    if len(carrier_order) != len(mobility.carrier_strength_mS_cm):
        raise ValueError("Carrier order construction lost at least one carrier")

    basis_vectors: list[np.ndarray] = []
    raw_by_anion: dict[str, float] = {}
    rho_by_anion: dict[str, float] = {}
    dimension = len(carrier_order)
    basis_vectors.append(_unit_vector(dimension, 0))

    for idx, carrier in enumerate(carrier_order[1:], start=1):
        source = _source_for_anion(composition, carrier)
        props = _ionic_source_props(source)
        cation = _require_string(props, "cation", f"ionic source {source}")
        cation_props = _require_species(CATION_PROPERTIES, cation, "cation")
        anion_radius = _require_float(props, "anion_radius", f"ionic source {source}")
        cation_radius = _require_float(cation_props, "solvated_radius_A", f"cation {cation}")
        dielectric_support = matrix.epsilon_effective / (matrix.epsilon_effective + eps_ref)
        anion_size_share = anion_radius / (anion_radius + cation_radius)
        raw = (
            params.pair_correlation_gain * speciation.paired_fraction_by_source[source]
            + params.aggregate_correlation_gain * speciation.aggregate_fraction_by_source[source]
            - params.steric_anticorrelation_gain
            * solvation.shell_steric_disruption
            * dielectric_support
            * anion_size_share
        )
        rho = math.tanh(raw)
        raw_by_anion[carrier] = raw
        rho_by_anion[carrier] = rho
        vector = np.zeros(dimension, dtype=float)
        vector[0] = rho
        vector[idx] = 1.0 / math.cosh(raw)
        basis_vectors.append(vector)

    basis = np.vstack(basis_vectors)
    corr = basis @ basis.T
    return CorrelationState(
        carrier_order=carrier_order,
        matrix=corr,
        li_anion_rho=rho_by_anion,
        raw_li_anion_coupling=raw_by_anion,
    )


def _onsager_readout(
    speciation: SpeciationState,
    mobility: MobilityState,
    correlation: CorrelationState,
) -> tuple[float, float]:
    strengths = np.asarray(
        [mobility.carrier_strength_mS_cm[carrier] for carrier in correlation.carrier_order],
        dtype=float,
    )
    charges = np.asarray(
        [speciation.carrier_charges[carrier] for carrier in correlation.carrier_order],
        dtype=float,
    )
    if np.any(strengths < 0.0):
        raise ValueError(f"Negative carrier strength encountered: {strengths}")

    sqrt_strength = np.sqrt(strengths)
    onsager_matrix = np.diag(sqrt_strength) @ correlation.matrix @ np.diag(sqrt_strength)
    sigma = float(charges @ onsager_matrix @ charges)
    sigma_uncorrelated = float(np.dot(charges * charges, strengths))
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError(
            f"Charge-correlation readout produced invalid conductivity {sigma}; "
            f"uncorrelated={sigma_uncorrelated}, charges={charges}, strengths={strengths}"
        )
    return sigma_uncorrelated, sigma


def _compute_dimer_viscosity_factor(
    composition: CompositionState,
    physics_config: Mapping[str, Any],
    dimer_viscosity_scale: float,
) -> float:
    corrections = _require_mapping(physics_config, "interaction_corrections", "physics_config")
    dimer_cfg = _require_mapping(corrections, "dimerization_viscosity", "interaction_corrections")
    viscosity_dimer_factor = _require_float(
        dimer_cfg,
        "viscosity_dimer_factor",
        "interaction_corrections.dimerization_viscosity",
    )

    factor = 1.0
    for name, phi in composition.neutral_liquid_volume_fractions.items():
        props = _neutral_species_props(name)
        k_dimer = _optional_float(props, "dimerization_constant_M_inv", f"neutral species {name}")
        if k_dimer == 0.0:
            continue
        concentration = _neutral_species_moles(composition, name)
        dimer_fraction = k_dimer * concentration / (1.0 + k_dimer * concentration)
        factor *= 1.0 + dimer_viscosity_scale * viscosity_dimer_factor * phi * dimer_fraction
    return factor


def _compute_salt_viscosity_factor(
    composition: CompositionState,
    viscosity_cfg: Mapping[str, Any],
    salt_viscosity_scale: float,
) -> float:
    jones_dole_by_species = _require_mapping(
        viscosity_cfg,
        "jones_dole_B_by_species",
        "viscosity_model",
    )
    b_vis = _require_float(viscosity_cfg, "jones_dole_B_vis", "viscosity_model")
    b_linear_ref = _require_float(viscosity_cfg, "jones_dole_B_linear_ref", "viscosity_model")
    d_vis = _require_float(viscosity_cfg, "jones_dole_D_vis", "viscosity_model")
    if b_linear_ref <= 0.0:
        raise ValueError("viscosity_model.jones_dole_B_linear_ref must be positive")

    linear = 0.0
    total_molarity = 0.0
    for source, molarity in composition.ionic_source_molarities_M.items():
        b_linear = _require_float(
            jones_dole_by_species,
            source,
            "viscosity_model.jones_dole_B_by_species",
        )
        linear += b_linear * b_vis / b_linear_ref * molarity
        total_molarity += molarity
    return math.exp(salt_viscosity_scale * (linear + d_vis * total_molarity * total_molarity))


def _compute_eyring_nrtl_excess_log_viscosity(
    composition: CompositionState,
    physics_config: Mapping[str, Any],
) -> float:
    viscosity_cfg = _require_mapping(physics_config, "viscosity_model", "physics_config")
    eyring_cfg = _require_mapping(viscosity_cfg, "eyring_nrtl", "viscosity_model")
    tau_params = _require_mapping(eyring_cfg, "tau_parameters", "viscosity_model.eyring_nrtl")
    alpha = _require_float(eyring_cfg, "alpha_nonrandomness", "viscosity_model.eyring_nrtl")

    neutral_names = list(composition.neutral_liquid_volume_fractions)
    neutral_moles = {name: _neutral_species_moles(composition, name) for name in neutral_names}
    total_moles = sum(neutral_moles.values())
    if total_moles <= 0.0:
        raise ValueError("Cannot compute Eyring-NRTL viscosity with zero neutral-liquid moles")

    x = np.asarray([neutral_moles[name] / total_moles for name in neutral_names], dtype=float)
    n = len(neutral_names)
    tau = np.zeros((n, n), dtype=float)
    g = np.ones((n, n), dtype=float)
    for i, name_i in enumerate(neutral_names):
        for j, name_j in enumerate(neutral_names):
            if i == j:
                continue
            key = f"{name_i}>{name_j}"
            if key in tau_params:
                tau[i, j] = _require_float(tau_params, key, "viscosity_model.eyring_nrtl.tau_parameters")
                g[i, j] = math.exp(-alpha * tau[i, j])

    excess = 0.0
    for i in range(n):
        denom = float(np.dot(x, g[:, i]))
        if denom <= 0.0:
            raise ValueError("Eyring-NRTL denominator became non-positive")
        numer = float(np.dot(x, tau[:, i] * g[:, i]))
        excess += float(x[i]) * numer / denom

    rk_cfg = _require_mapping(eyring_cfg, "redlich_kister_asymmetry", "viscosity_model.eyring_nrtl")
    for i, name_i in enumerate(neutral_names):
        for j, name_j in enumerate(neutral_names[i + 1 :], start=i + 1):
            g1 = _optional_pair_float(
                rk_cfg,
                name_i,
                name_j,
                "viscosity_model.eyring_nrtl.redlich_kister_asymmetry",
            )
            if g1 is not None:
                if name_i <= name_j:
                    excess += x[i] * x[j] * g1 * (x[i] - x[j])
                else:
                    excess += x[i] * x[j] * g1 * (x[j] - x[i])
    return float(excess)


def _salt_mobility_scale(params: OnsagerConductivityParams, source: str) -> float:
    if params.salt_mobility_scales is None:
        return 1.0
    if source not in params.salt_mobility_scales:
        raise ValueError(f"Missing fitted salt mobility scale for ionic source {source}")
    scale = float(params.salt_mobility_scales[source])
    _assert_positive_float(scale, f"params.salt_mobility_scales.{source}")
    return scale


def _is_ionic_source(props: Mapping[str, Any]) -> bool:
    return "cation" in props and "anion" in props and "Lambda_0" in props


def _ionic_source_props(source: str) -> Mapping[str, Any]:
    if source in SALTS:
        return SALTS[source]
    if source in ADDITIVES:
        props = ADDITIVES[source]
        if _is_ionic_source(props):
            return props
    raise ValueError(f"Species {source} is not an ionic source")


def _neutral_species_props(name: str) -> Mapping[str, Any]:
    if name in SOLVENTS:
        return SOLVENTS[name]
    if name in ADDITIVES:
        props = ADDITIVES[name]
        if not _is_ionic_source(props):
            return props
    raise ValueError(f"Species {name} is not a neutral liquid species")


def _neutral_species_moles(composition: CompositionState, name: str) -> float:
    if name in composition.solvent_moles:
        return composition.solvent_moles[name]
    if name in composition.additive_moles:
        return composition.additive_moles[name]
    raise ValueError(f"Neutral species {name} has no mole count in composition")


def _source_for_anion(composition: CompositionState, anion: str) -> str:
    matches = []
    for source in composition.ionic_source_molarities_M:
        props = _ionic_source_props(source)
        if _require_string(props, "anion", f"ionic source {source}") == anion:
            matches.append(source)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source for anion {anion}, found {matches}")
    return matches[0]


def _single_cation_symbol(composition: CompositionState) -> str:
    symbols: set[str] = set()
    for source in composition.ionic_source_molarities_M:
        props = _ionic_source_props(source)
        cation = _require_string(props, "cation", f"ionic source {source}")
        cation_props = _require_species(CATION_PROPERTIES, cation, "cation")
        symbols.add(_require_string(cation_props, "ion_symbol", f"cation {cation}"))
    if len(symbols) != 1:
        raise ValueError(f"Onsager prototype currently requires one cation family, found {sorted(symbols)}")
    return next(iter(symbols))


def _pair_key(name_i: str, name_j: str) -> str:
    return ":".join(sorted((name_i, name_j)))


def _optional_pair_float(
    mapping: Mapping[str, Any],
    name_i: str,
    name_j: str,
    context: str,
) -> float | None:
    keys = (f"{name_i}:{name_j}", f"{name_j}:{name_i}", _pair_key(name_i, name_j))
    for key in keys:
        if key in mapping:
            return _require_float(mapping, key, context)
    return None


def _unit_vector(dimension: int, index: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=float)
    vector[index] = 1.0
    return vector


def _require_mapping(mapping: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    if key not in mapping:
        raise ValueError(f"Missing required key {context}.{key}")
    value = mapping[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{key} must be a mapping")
    return value


def _require_species(registry: Mapping[str, Mapping[str, Any]], name: str, role: str) -> Mapping[str, Any]:
    if name not in registry:
        raise ValueError(f"Unknown {role} species {name}")
    return registry[name]


def _require_float(mapping: Mapping[str, Any], key: str, context: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing required key {context}.{key}")
    value = mapping[key]
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{key} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context}.{key} must be finite, got {parsed}")
    return parsed


def _optional_float(mapping: Mapping[str, Any], key: str, context: str) -> float:
    if key not in mapping:
        return 0.0
    value = mapping[key]
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{key} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context}.{key} must be finite, got {parsed}")
    return parsed


def _require_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    if key not in mapping:
        raise ValueError(f"Missing required key {context}.{key}")
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _validate_params(params: OnsagerConductivityParams) -> None:
    _assert_positive_float(params.mobility_scale, "params.mobility_scale")
    _assert_positive_float(params.bjerrum_dielectric_scale, "params.bjerrum_dielectric_scale")
    _assert_positive_float(params.viscosity_exponent_scale, "params.viscosity_exponent_scale")
    _assert_positive_float(params.liquid_excess_viscosity_scale, "params.liquid_excess_viscosity_scale")
    _assert_positive_float(params.dimer_viscosity_scale, "params.dimer_viscosity_scale")
    _assert_positive_float(params.salt_viscosity_scale, "params.salt_viscosity_scale")
    _assert_positive_float(params.pair_correlation_gain, "params.pair_correlation_gain")
    _assert_positive_float(params.aggregate_correlation_gain, "params.aggregate_correlation_gain")
    _assert_positive_float(params.steric_anticorrelation_gain, "params.steric_anticorrelation_gain")
    if params.salt_mobility_scales is not None:
        for source, scale in params.salt_mobility_scales.items():
            _assert_positive_float(float(scale), f"params.salt_mobility_scales.{source}")


def _assert_positive_float(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{context} must be a positive finite number, got {value}")


def _assert_nonnegative_float(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{context} must be a non-negative finite number, got {value}")
