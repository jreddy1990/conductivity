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

from constants import BJERRUM_LENGTH_NM, CM2_PER_M2, F, K_B, N_A, PA_S_TO_CP, R, T_REF_K
from conductivity.site_measure_transport import (
    AnionSiteFeature,
    CationSiteFeature,
    NeutralLigandSiteFeature,
    TransportSiteMeasure,
    build_transport_site_measure,
)
from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
from electrolyte_model import ElectrolyteRecipeModel
from utils.config_cache import load_physics_config
from utils.strict_validation import (
    require_float,
    require_float as _require_float,
    require_mapping,
    require_mapping as _require_mapping,
    require_string,
    require_string as _require_string,
)


LITER_TO_ML = 1000.0
CATION_RADIUS_MATCH_TOLERANCE_A = 1.0e-9  # Explicit tolerance for metadata identity matching.
COORDINATION_NUMBER_MATCH_TOLERANCE = 1.0e-9  # Explicit tolerance for cation-family CN uniqueness.
KJ_TO_J = 1000.0  # Explicit constant: unit conversion, 1 kJ = 1000 J.
ANGSTROM_TO_NM = float("0.1")  # Explicit constant: unit conversion, 1 Angstrom = 0.1 nm.
ANGSTROM_TO_M = 1.0e-10  # Explicit constant: unit conversion, 1 Angstrom = 1e-10 m.
CP_TO_PA_S = 1.0 / PA_S_TO_CP
CM3_TO_M3 = 1.0e-6  # Explicit constant: unit conversion, 1 cm^3 = 1e-6 m^3.
NANOSECOND_TO_SECOND = 1.0e-9  # Explicit constant: unit conversion, 1 ns = 1e-9 s.
STOKES_SPHERE_DRAG_FACTOR = 6.0  # Explicit constant: Stokes-Einstein sphere drag denominator.
SHARED_LI_NEWTON_MAX_ITERATIONS = 80  # Explicit constant: numerical sentinel, 16 Newton attempts per variable for expected <=5-variable systems.
SHARED_LI_BACKTRACKING_STEPS = 24  # Explicit constant: numerical sentinel, 24 halvings give a 2^-24 trial step before loud failure.
SHARED_LI_LINE_SEARCH_REDUCTION = 0.5  # Explicit constant: numerical sentinel, bisection-style Newton backtracking.
SHARED_LI_RELATIVE_TOLERANCE = 1.0e-10  # Explicit constant: numerical sentinel, mass-balance residual tolerance below data precision.
FINITE_DIFFERENCE_STEP = math.sqrt(float(np.finfo(float).eps))


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
    salt_additive_viscosity_factor: float
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
    free_fraction_by_feature: dict[str, float]
    paired_fraction_by_feature: dict[str, float]
    ssip_fraction_by_feature: dict[str, float]
    cip_fraction_by_feature: dict[str, float]
    aggregate_fraction_by_feature: dict[str, float]
    bridge_network_fraction_by_feature: dict[str, float]
    li_ligand_fraction_by_feature: dict[str, float]
    association_constant_by_feature_M_inv: dict[str, float]
    carrier_concentrations_M: dict[str, float]
    carrier_charges: dict[str, int]
    motif_concentrations_M: dict[str, float]


@dataclass(frozen=True)
class SharedLiAssociationKernel:
    anion_feature_ids: tuple[str, ...]
    neutral_ligand_feature_ids: tuple[str, ...]
    total_lithium_molarity_M: float
    anion_molarities_M: dict[str, float]
    neutral_ligand_molarities_M: dict[str, float]
    ssip_association_M_inv: dict[str, float]
    cip_association_M_inv: dict[str, float]
    aggregate_association_M_inv: dict[str, float]
    bridge_network_association_M_inv3: dict[str, float]
    neutral_ligand_association_M_inv: dict[str, float]


@dataclass(frozen=True)
class IonPairingKernelConfig:
    bjerrum_eps_ref: float
    aggregate_onset_M: float
    aggregate_scale_M: float
    aggregate_max_fraction: float


@dataclass(frozen=True)
class SharedLiSpeciationSolution:
    free_lithium_molarity_M: float
    free_anion_molarity_M: dict[str, float]
    ssip_molarity_M: dict[str, float]
    cip_molarity_M: dict[str, float]
    aggregate_molarity_M: dict[str, float]
    bridge_network_molarity_M: dict[str, float]
    li_ligand_molarity_M: dict[str, float]


@dataclass(frozen=True)
class MobilityState:
    carrier_lambda_S_cm2_mol: dict[str, float]
    carrier_strength_mS_cm: dict[str, float]
    carrier_strength_no_crowding_mS_cm: dict[str, float]
    feature_lambda_split_S_cm2_mol: dict[str, dict[str, float]]
    feature_lambda_no_shape_S_cm2_mol: dict[str, dict[str, float]]
    anion_shape_factor_by_feature: dict[str, float]
    cation_microviscosity_coupling_exponent: float
    anion_microviscosity_coupling_exponent_by_feature: dict[str, float]
    reference_viscosity_cP: float
    raw_ionic_occupied_volume_fraction: float
    network_occupied_volume_fraction: float
    crowding_factor: float


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


@dataclass(frozen=True)
class TransportKernelState:
    composition: CompositionState
    matrix: MatrixState
    solvation: SolvationState
    speciation: SpeciationState
    mobility: MobilityState
    site_measure: TransportSiteMeasure

    def with_site_measure(self, site_measure: TransportSiteMeasure) -> "TransportKernelState":
        return TransportKernelState(
            composition=self.composition,
            matrix=self.matrix,
            solvation=self.solvation,
            speciation=self.speciation,
            mobility=self.mobility,
            site_measure=site_measure,
        )


def build_transport_kernel_state(
    recipe: Mapping[str, Any],
    temperature_K: float,
    physics_config: Mapping[str, Any],
) -> TransportKernelState:
    """Build the shared composition and matrix kernel state for transport models."""

    _assert_positive_float(temperature_K, "temperature_K")

    recipe_model = ElectrolyteRecipeModel.model_validate(dict(recipe))
    composition = _build_composition_state(recipe_model, temperature_K)
    unit_params = OnsagerConductivityParams(mobility_scale=1.0)
    matrix = _build_matrix_state(composition, physics_config, unit_params)
    solvation = _build_solvation_state(composition)
    site_measure = build_transport_site_measure(composition)
    speciation = _build_speciation_state(composition, matrix, solvation, physics_config, unit_params, site_measure)
    mobility = _build_mobility_state(
        composition,
        matrix,
        solvation,
        speciation,
        temperature_K,
        unit_params,
        physics_config,
        site_measure,
    )
    return TransportKernelState(
        composition=composition,
        matrix=matrix,
        solvation=solvation,
        speciation=speciation,
        mobility=mobility,
        site_measure=site_measure,
    )


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
    site_measure = build_transport_site_measure(composition)
    speciation = _build_speciation_state(composition, matrix, solvation, physics_config, params, site_measure)
    mobility = _build_mobility_state(
        composition,
        matrix,
        solvation,
        speciation,
        temperature_K,
        params,
        physics_config,
        site_measure,
    )
    correlation = _build_correlation_state(
        composition,
        matrix,
        solvation,
        speciation,
        mobility,
        physics_config,
        params,
        site_measure,
    )
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
        recipe = require_mapping(entry, "recipe", f"entry[{idx}]")
        properties = require_mapping(entry, "properties", f"entry[{idx}]")
        measured = require_float(properties, "conductivity_mS_cm", f"entry[{idx}].properties")
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
    friction strength, salt viscosity strength, and correlation gains. It does
    not fit species-specific mobility scales.
    """

    from scipy.optimize import least_squares

    if physics_config is None:
        physics_config = load_physics_config()
    _assert_positive_float(temperature_K, "temperature_K")

    rows: list[tuple[Mapping[str, Any], float]] = []
    for idx, entry in enumerate(entries):
        recipe = require_mapping(entry, "recipe", f"entry[{idx}]")
        properties = require_mapping(entry, "properties", f"entry[{idx}]")
        measured = require_float(properties, "conductivity_mS_cm", f"entry[{idx}].properties")
        if measured <= 0.0:
            raise ValueError(f"entry[{idx}].properties.conductivity_mS_cm must be positive")
        evaluate_onsager_conductivity(
            recipe,
            temperature_K=temperature_K,
            params=OnsagerConductivityParams(mobility_scale=1.0),
            physics_config=physics_config,
        )
        rows.append((recipe, measured))

    if not rows:
        raise ValueError("Cannot fit mechanism parameters without measured conductivity rows")

    global_fit = fit_global_mobility_scale(entries, temperature_K, physics_config)

    x0 = np.zeros(9, dtype=float)
    x0[0] = math.log(global_fit.params.mobility_scale)

    def unpack(x: np.ndarray) -> OnsagerConductivityParams:
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
    salt_additive_factor = _compute_salt_additive_viscosity_factor(
        composition,
        viscosity_cfg,
        eta_liquid,
        params.salt_viscosity_scale,
    )
    eta_solution = eta_liquid * dimer_factor * salt_factor * salt_additive_factor
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
        salt_additive_viscosity_factor=salt_additive_factor,
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
            * _ionic_source_preferred_coordination_number(source, props)
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
    solvation: SolvationState,
    physics_config: Mapping[str, Any],
    params: OnsagerConductivityParams,
    site_measure: TransportSiteMeasure,
) -> SpeciationState:
    pairing_cfg = _require_mapping(physics_config, "ion_pairing_model", "physics_config")
    eps_ref = _require_float(pairing_cfg, "bjerrum_eps_ref", "ion_pairing_model")
    aggregate_onset = _require_float(pairing_cfg, "aggregate_onset_mol_l", "ion_pairing_model")
    aggregate_scale = _require_float(pairing_cfg, "aggregate_scale_mol_l", "ion_pairing_model")
    aggregate_max = _require_float(pairing_cfg, "aggregate_max_fraction_of_paired", "ion_pairing_model")
    if aggregate_scale <= 0.0:
        raise ValueError("ion_pairing_model.aggregate_scale_mol_l must be positive")

    pairing_kernel_config = IonPairingKernelConfig(
        bjerrum_eps_ref=eps_ref,
        aggregate_onset_M=aggregate_onset,
        aggregate_scale_M=aggregate_scale,
        aggregate_max_fraction=aggregate_max,
    )
    association_kernel = _build_shared_li_association_kernel(
        composition,
        matrix,
        solvation,
        params,
        pairing_kernel_config,
        site_measure,
    )
    speciation_solution = solve_shared_li_speciation(association_kernel)
    free_by_feature: dict[str, float] = {}
    paired_by_feature: dict[str, float] = {}
    ssip_by_feature: dict[str, float] = {}
    cip_by_feature: dict[str, float] = {}
    aggregate_by_feature: dict[str, float] = {}
    bridge_network_by_feature: dict[str, float] = {}
    li_ligand_by_feature: dict[str, float] = {}
    carrier_concentrations: dict[str, float] = {}
    carrier_charges: dict[str, int] = {}
    motif_concentrations: dict[str, float] = {}

    for anion_site in site_measure.anion_sites:
        anion_feature_id = anion_site.canonical_feature_id
        molarity = association_kernel.anion_molarities_M[anion_feature_id]
        cation_charge = site_measure.cation.charge
        cation_symbol = site_measure.cation.ion_symbol
        free_cation_concentration = (
            speciation_solution.free_lithium_molarity_M
            * molarity
            / association_kernel.total_lithium_molarity_M
        )
        free_anion_concentration = speciation_solution.free_anion_molarity_M[anion_feature_id]
        ssip_concentration = speciation_solution.ssip_molarity_M[anion_feature_id]
        cip_concentration = speciation_solution.cip_molarity_M[anion_feature_id]
        aggregate_concentration = speciation_solution.aggregate_molarity_M[anion_feature_id]
        bridge_network_concentration = speciation_solution.bridge_network_molarity_M[anion_feature_id]

        free_by_feature[anion_feature_id] = free_cation_concentration / molarity
        ssip_by_feature[anion_feature_id] = ssip_concentration / molarity
        cip_by_feature[anion_feature_id] = cip_concentration / molarity
        aggregate_by_feature[anion_feature_id] = aggregate_concentration / molarity
        bridge_network_by_feature[anion_feature_id] = 2.0 * bridge_network_concentration / molarity
        paired_by_feature[anion_feature_id] = (
            ssip_concentration
            + cip_concentration
            + aggregate_concentration
            + 2.0 * bridge_network_concentration
        ) / molarity

        motif_concentrations[f"free_cation:{anion_feature_id}"] = free_cation_concentration
        motif_concentrations[f"free_anion:{anion_feature_id}"] = free_anion_concentration
        motif_concentrations[f"SSIP:{anion_feature_id}"] = ssip_concentration
        motif_concentrations[f"CIP:{anion_feature_id}"] = cip_concentration
        motif_concentrations[f"AGG:{anion_feature_id}"] = aggregate_concentration
        motif_concentrations[f"BRIDGE_NETWORK:{anion_feature_id}"] = bridge_network_concentration

        _accumulate_float(carrier_concentrations, cation_symbol, free_cation_concentration)
        _accumulate_float(carrier_concentrations, anion_site.carrier_label, free_anion_concentration)
        carrier_charges[cation_symbol] = cation_charge
        carrier_charges[anion_site.carrier_label] = anion_site.charge

    for ligand_site in site_measure.neutral_ligand_sites:
        ligand_feature_id = ligand_site.canonical_feature_id
        li_ligand_concentration = speciation_solution.li_ligand_molarity_M[ligand_feature_id]
        li_ligand_by_feature[ligand_feature_id] = (
            li_ligand_concentration / association_kernel.total_lithium_molarity_M
        )
        motif_concentrations[f"Li_ligand:{ligand_feature_id}"] = li_ligand_concentration

    return SpeciationState(
        free_fraction_by_feature=free_by_feature,
        paired_fraction_by_feature=paired_by_feature,
        ssip_fraction_by_feature=ssip_by_feature,
        cip_fraction_by_feature=cip_by_feature,
        aggregate_fraction_by_feature=aggregate_by_feature,
        bridge_network_fraction_by_feature=bridge_network_by_feature,
        li_ligand_fraction_by_feature=li_ligand_by_feature,
        association_constant_by_feature_M_inv={
            anion_feature_id: (
                association_kernel.ssip_association_M_inv[anion_feature_id]
                + association_kernel.cip_association_M_inv[anion_feature_id]
            )
            for anion_feature_id in association_kernel.anion_feature_ids
        },
        carrier_concentrations_M=carrier_concentrations,
        carrier_charges=carrier_charges,
        motif_concentrations_M=motif_concentrations,
    )


def _build_shared_li_association_kernel(
    composition: CompositionState,
    matrix: MatrixState,
    solvation: SolvationState,
    params: OnsagerConductivityParams,
    pairing_kernel_config: IonPairingKernelConfig,
    site_measure: TransportSiteMeasure,
) -> SharedLiAssociationKernel:
    _assert_positive_float(pairing_kernel_config.bjerrum_eps_ref, "ion_pairing_model.bjerrum_eps_ref")
    _assert_positive_float(pairing_kernel_config.aggregate_scale_M, "ion_pairing_model.aggregate_scale_mol_l")
    _assert_nonnegative_float(
        pairing_kernel_config.aggregate_max_fraction,
        "ion_pairing_model.aggregate_max_fraction_of_paired",
    )

    anion_molarities: dict[str, float] = {}
    ssip_association: dict[str, float] = {}
    cip_association: dict[str, float] = {}
    aggregate_association: dict[str, float] = {}
    bridge_network_association: dict[str, float] = {}
    for anion_site in site_measure.anion_sites:
        anion_feature_id = anion_site.canonical_feature_id
        molarity_M = anion_site.molarity_M
        _assert_nonnegative_float(molarity_M, f"anion feature {anion_feature_id} molarity")
        if molarity_M == 0.0:
            continue
        total_association = _feature_association_constant_M_inv(
            anion_site,
            composition,
            matrix,
            params,
            pairing_kernel_config.bjerrum_eps_ref,
        )
        ssip_fraction, cip_fraction = _ssip_cip_split_from_feature(anion_site, composition, solvation)
        anion_molarities[anion_feature_id] = molarity_M
        ssip_association[anion_feature_id] = total_association * ssip_fraction
        cip_association[anion_feature_id] = total_association * cip_fraction
        aggregate_association[anion_feature_id] = _aggregate_association_constant_M_inv(
            anion_feature_id,
            total_source_molarity_M=sum(site.molarity_M for site in site_measure.anion_sites),
            aggregate_onset_M=pairing_kernel_config.aggregate_onset_M,
            aggregate_scale_M=pairing_kernel_config.aggregate_scale_M,
            aggregate_max_fraction=pairing_kernel_config.aggregate_max_fraction,
        )
        bridge_network_association[anion_feature_id] = _bridge_network_association_constant_from_feature_M_inv3(
            anion_site,
            total_association,
            pairing_kernel_config.aggregate_scale_M,
        )

    if not anion_molarities:
        raise ValueError("shared-Li speciation requires at least one positive ionic source")

    ligand_molarities: dict[str, float] = {}
    ligand_association: dict[str, float] = {}
    for ligand_site in site_measure.neutral_ligand_sites:
        ligand_feature_id = ligand_site.canonical_feature_id
        ligand_molarities[ligand_feature_id] = ligand_site.molarity_M
        ligand_association[ligand_feature_id] = ligand_site.coordination_affinity_M_inv
        _assert_positive_float(
            ligand_association[ligand_feature_id],
            f"neutral ligand feature {ligand_feature_id} coordination affinity",
        )

    total_lithium_molarity = math.fsum(anion_molarities.values())
    _assert_positive_float(total_lithium_molarity, "total lithium molarity")
    return SharedLiAssociationKernel(
        anion_feature_ids=tuple(anion_molarities.keys()),
        neutral_ligand_feature_ids=tuple(ligand_molarities.keys()),
        total_lithium_molarity_M=total_lithium_molarity,
        anion_molarities_M=anion_molarities,
        neutral_ligand_molarities_M=ligand_molarities,
        ssip_association_M_inv=ssip_association,
        cip_association_M_inv=cip_association,
        aggregate_association_M_inv=aggregate_association,
        bridge_network_association_M_inv3=bridge_network_association,
        neutral_ligand_association_M_inv=ligand_association,
    )


def solve_shared_li_speciation(
    association_kernel: SharedLiAssociationKernel,
) -> SharedLiSpeciationSolution:
    log_concentrations = _initial_shared_li_log_concentrations(association_kernel)
    residual = _shared_li_residual_vector(log_concentrations, association_kernel)
    residual_scale = _shared_li_residual_scale(association_kernel)
    tolerance = SHARED_LI_RELATIVE_TOLERANCE * residual_scale

    for iteration_index in range(SHARED_LI_NEWTON_MAX_ITERATIONS):
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm <= tolerance:
            return _shared_li_solution_from_log_concentrations(log_concentrations, association_kernel)
        jacobian = _shared_li_finite_difference_jacobian(
            log_concentrations,
            residual,
            association_kernel,
        )
        try:
            newton_step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError("shared-Li speciation Jacobian is singular") from exc
        log_concentrations, residual = _shared_li_backtracking_update(
            log_concentrations,
            residual,
            newton_step,
            association_kernel,
        )

    final_norm = float(np.linalg.norm(residual, ord=np.inf))
    raise ValueError(
        f"shared-Li speciation did not converge after {SHARED_LI_NEWTON_MAX_ITERATIONS} "
        f"iterations; residual={final_norm}, tolerance={tolerance}"
    )


def _initial_shared_li_log_concentrations(
    association_kernel: SharedLiAssociationKernel,
) -> np.ndarray:
    association_load = 0.0
    for anion_feature_id in association_kernel.anion_feature_ids:
        total_association = (
            association_kernel.ssip_association_M_inv[anion_feature_id]
            + association_kernel.cip_association_M_inv[anion_feature_id]
        )
        association_load += total_association * association_kernel.anion_molarities_M[anion_feature_id]
    for ligand_feature_id in association_kernel.neutral_ligand_feature_ids:
        association_load += (
            association_kernel.neutral_ligand_association_M_inv[ligand_feature_id]
            * association_kernel.neutral_ligand_molarities_M[ligand_feature_id]
        )
    free_lithium_guess = association_kernel.total_lithium_molarity_M / (1.0 + association_load)
    _assert_positive_float(free_lithium_guess, "shared-Li initial free lithium")
    guesses = [math.log(free_lithium_guess)]
    for anion_feature_id in association_kernel.anion_feature_ids:
        total_association = (
            association_kernel.ssip_association_M_inv[anion_feature_id]
            + association_kernel.cip_association_M_inv[anion_feature_id]
        )
        free_anion_guess = (
            association_kernel.anion_molarities_M[anion_feature_id]
            / (1.0 + total_association * free_lithium_guess)
        )
        _assert_positive_float(free_anion_guess, f"shared-Li initial free anion {anion_feature_id}")
        guesses.append(math.log(free_anion_guess))
    return np.asarray(guesses, dtype=float)


def _shared_li_residual_vector(
    log_concentrations: np.ndarray,
    association_kernel: SharedLiAssociationKernel,
) -> np.ndarray:
    solution = _shared_li_solution_from_log_concentrations(log_concentrations, association_kernel)
    lithium_balance = (
        solution.free_lithium_molarity_M
        + math.fsum(solution.ssip_molarity_M.values())
        + math.fsum(solution.cip_molarity_M.values())
        + math.fsum(solution.aggregate_molarity_M.values())
        + 2.0 * math.fsum(solution.bridge_network_molarity_M.values())
        + math.fsum(solution.li_ligand_molarity_M.values())
    )
    residuals = [lithium_balance - association_kernel.total_lithium_molarity_M]
    for anion_feature_id in association_kernel.anion_feature_ids:
        source_balance = (
            solution.free_anion_molarity_M[anion_feature_id]
            + solution.ssip_molarity_M[anion_feature_id]
            + solution.cip_molarity_M[anion_feature_id]
            + solution.aggregate_molarity_M[anion_feature_id]
            + 2.0 * solution.bridge_network_molarity_M[anion_feature_id]
        )
        residuals.append(source_balance - association_kernel.anion_molarities_M[anion_feature_id])
    return np.asarray(residuals, dtype=float)


def _shared_li_solution_from_log_concentrations(
    log_concentrations: np.ndarray,
    association_kernel: SharedLiAssociationKernel,
) -> SharedLiSpeciationSolution:
    if log_concentrations.shape[0] != len(association_kernel.anion_feature_ids) + 1:
        raise ValueError("shared-Li log concentration vector has wrong size")
    if not np.all(np.isfinite(log_concentrations)):
        raise ValueError("shared-Li log concentrations must be finite")
    free_lithium_molarity = float(math.exp(float(log_concentrations[0])))
    free_anion_molarity: dict[str, float] = {}
    for source_index, anion_feature_id in enumerate(association_kernel.anion_feature_ids, start=1):
        free_anion_molarity[anion_feature_id] = float(math.exp(float(log_concentrations[source_index])))
    return _shared_li_solution_from_free_concentrations(
        association_kernel,
        free_lithium_molarity,
        free_anion_molarity,
    )


def _shared_li_solution_from_free_concentrations(
    association_kernel: SharedLiAssociationKernel,
    free_lithium_molarity_M: float,
    free_anion_molarity_M: Mapping[str, float],
) -> SharedLiSpeciationSolution:
    _assert_positive_float(free_lithium_molarity_M, "free_lithium_molarity_M")
    ssip_molarity: dict[str, float] = {}
    cip_molarity: dict[str, float] = {}
    aggregate_molarity: dict[str, float] = {}
    bridge_network_molarity: dict[str, float] = {}
    for anion_feature_id in association_kernel.anion_feature_ids:
        free_anion_molarity = float(free_anion_molarity_M[anion_feature_id])
        _assert_positive_float(free_anion_molarity, f"free_anion_molarity_M.{anion_feature_id}")
        ssip_molarity[anion_feature_id] = (
            association_kernel.ssip_association_M_inv[anion_feature_id]
            * free_lithium_molarity_M
            * free_anion_molarity
        )
        cip_molarity[anion_feature_id] = (
            association_kernel.cip_association_M_inv[anion_feature_id]
            * free_lithium_molarity_M
            * free_anion_molarity
        )
        aggregate_molarity[anion_feature_id] = (
            association_kernel.aggregate_association_M_inv[anion_feature_id]
            * cip_molarity[anion_feature_id]
            * (free_lithium_molarity_M + free_anion_molarity)
        )
        bridge_network_molarity[anion_feature_id] = (
            association_kernel.bridge_network_association_M_inv3[anion_feature_id]
            * free_lithium_molarity_M
            * free_lithium_molarity_M
            * free_anion_molarity
            * free_anion_molarity
        )
    li_ligand_molarity: dict[str, float] = {}
    for ligand_feature_id in association_kernel.neutral_ligand_feature_ids:
        li_ligand_molarity[ligand_feature_id] = (
            association_kernel.neutral_ligand_association_M_inv[ligand_feature_id]
            * free_lithium_molarity_M
            * association_kernel.neutral_ligand_molarities_M[ligand_feature_id]
        )
    return SharedLiSpeciationSolution(
        free_lithium_molarity_M=free_lithium_molarity_M,
        free_anion_molarity_M=dict(free_anion_molarity_M),
        ssip_molarity_M=ssip_molarity,
        cip_molarity_M=cip_molarity,
        aggregate_molarity_M=aggregate_molarity,
        bridge_network_molarity_M=bridge_network_molarity,
        li_ligand_molarity_M=li_ligand_molarity,
    )


def _shared_li_finite_difference_jacobian(
    log_concentrations: np.ndarray,
    base_residual: np.ndarray,
    association_kernel: SharedLiAssociationKernel,
) -> np.ndarray:
    jacobian = np.zeros((base_residual.shape[0], log_concentrations.shape[0]), dtype=float)
    for column_index in range(log_concentrations.shape[0]):
        perturbed = np.array(log_concentrations, dtype=float)
        perturbed[column_index] += FINITE_DIFFERENCE_STEP
        perturbed_residual = _shared_li_residual_vector(perturbed, association_kernel)
        jacobian[:, column_index] = (perturbed_residual - base_residual) / FINITE_DIFFERENCE_STEP
    return jacobian


def _shared_li_backtracking_update(
    log_concentrations: np.ndarray,
    base_residual: np.ndarray,
    newton_step: np.ndarray,
    association_kernel: SharedLiAssociationKernel,
) -> tuple[np.ndarray, np.ndarray]:
    base_norm = float(np.linalg.norm(base_residual, ord=np.inf))
    step_multiplier = 1.0
    for backtracking_index in range(SHARED_LI_BACKTRACKING_STEPS):
        trial_log_concentrations = log_concentrations + step_multiplier * newton_step
        trial_residual = _shared_li_residual_vector(trial_log_concentrations, association_kernel)
        trial_norm = float(np.linalg.norm(trial_residual, ord=np.inf))
        if math.isfinite(trial_norm) and trial_norm < base_norm:
            return trial_log_concentrations, trial_residual
        step_multiplier *= SHARED_LI_LINE_SEARCH_REDUCTION
    raise ValueError(
        f"shared-Li speciation line search failed after {SHARED_LI_BACKTRACKING_STEPS} "
        f"backtracking steps; residual={base_norm}"
    )


def _shared_li_residual_scale(association_kernel: SharedLiAssociationKernel) -> float:
    scale_candidates = [association_kernel.total_lithium_molarity_M]
    for anion_feature_id in association_kernel.anion_feature_ids:
        scale_candidates.append(association_kernel.anion_molarities_M[anion_feature_id])
    return max(scale_candidates)


def _feature_association_constant_M_inv(
    anion_site: AnionSiteFeature,
    composition: CompositionState,
    matrix: MatrixState,
    params: OnsagerConductivityParams,
    eps_ref: float,
) -> float:
    contact_distance_nm = (anion_site.cation_radius_A + anion_site.anion_radius_A) * ANGSTROM_TO_NM
    _assert_positive_float(contact_distance_nm, f"anion feature {anion_site.canonical_feature_id} contact distance")
    _assert_positive_float(
        anion_site.bjerrum_association_reference_M_inv,
        f"anion feature {anion_site.canonical_feature_id}.bjerrum_association_reference_M_inv",
    )
    coulomb_term = BJERRUM_LENGTH_NM * (T_REF_K / composition.temperature_K) / contact_distance_nm
    association_constant = anion_site.bjerrum_association_reference_M_inv * math.exp(
        params.bjerrum_dielectric_scale
        * coulomb_term
        * (1.0 / matrix.epsilon_effective - 1.0 / eps_ref)
    )
    _assert_positive_float(association_constant, f"anion feature {anion_site.canonical_feature_id} association constant")
    return association_constant


def _ssip_cip_split_from_feature(
    anion_site: AnionSiteFeature,
    composition: CompositionState,
    solvation: SolvationState,
) -> tuple[float, float]:
    thermal_kJ_mol = R * composition.temperature_K / KJ_TO_J
    _assert_positive_float(thermal_kJ_mol, "thermal energy kJ/mol")
    ssip_log_weight = solvation.shell_binding_energy_kJ_mol / thermal_kJ_mol
    cip_log_weight = anion_site.ion_pair_binding_kJ_mol / thermal_kJ_mol
    log_norm = _logsumexp_pair(ssip_log_weight, cip_log_weight)
    ssip_fraction = math.exp(ssip_log_weight - log_norm)
    cip_fraction = math.exp(cip_log_weight - log_norm)
    return ssip_fraction, cip_fraction


def _aggregate_association_constant_M_inv(
    source_name: str,
    total_source_molarity_M: float,
    aggregate_onset_M: float,
    aggregate_scale_M: float,
    aggregate_max_fraction: float,
) -> float:
    _assert_positive_float(total_source_molarity_M, "total source molarity")
    _assert_positive_float(aggregate_scale_M, "aggregate scale")
    _assert_nonnegative_float(aggregate_max_fraction, "aggregate max fraction")
    aggregate_gate = aggregate_max_fraction / (
        1.0 + math.exp(-(total_source_molarity_M - aggregate_onset_M) / aggregate_scale_M)
    )
    aggregate_association = aggregate_gate / aggregate_scale_M
    _assert_nonnegative_float(aggregate_association, f"ionic source {source_name} aggregate association")
    return aggregate_association


def _bridge_network_association_constant_from_feature_M_inv3(
    anion_site: AnionSiteFeature,
    association_constant_M_inv: float,
    aggregate_scale_M: float,
) -> float:
    _assert_positive_float(
        association_constant_M_inv,
        f"anion feature {anion_site.canonical_feature_id} association constant",
    )
    _assert_positive_float(aggregate_scale_M, "aggregate scale")
    if anion_site.donor_site_count <= 0.0:
        return 0.0
    _assert_positive_float(
        anion_site.preferred_coordination_number,
        f"anion feature {anion_site.canonical_feature_id} preferred coordination number",
    )
    bridge_eligibility = (
        anion_site.donor_site_count
        * anion_site.coordination_multiplicity
        / (
            (anion_site.donor_site_count + anion_site.preferred_coordination_number)
            * anion_site.preferred_coordination_number
        )
    )
    _assert_nonnegative_float(bridge_eligibility, f"anion feature {anion_site.canonical_feature_id} bridge eligibility")
    return bridge_eligibility * association_constant_M_inv / (aggregate_scale_M * aggregate_scale_M)


def _logsumexp_pair(first_value: float, second_value: float) -> float:
    max_value = max(first_value, second_value)
    return max_value + math.log(math.exp(first_value - max_value) + math.exp(second_value - max_value))


def _accumulate_float(values: dict[str, float], key: str, increment: float) -> None:
    _assert_nonnegative_float(increment, f"{key} increment")
    if key in values:
        values[key] += increment
    else:
        values[key] = increment


def raw_ionic_occupied_volume_fraction(
    composition: CompositionState,
    speciation: SpeciationState,
    site_measure: TransportSiteMeasure,
) -> float:
    occupied_volume_fraction = 0.0
    for motif_name, concentration_M in speciation.motif_concentrations_M.items():
        _assert_nonnegative_float(concentration_M, f"speciation.motif_concentrations_M.{motif_name}")
        partial_molar_volume_cm3_mol = _motif_partial_molar_volume_cm3_mol(
            motif_name,
            composition,
            site_measure,
        )
        occupied_volume_fraction += concentration_M * partial_molar_volume_cm3_mol / LITER_TO_ML
    if occupied_volume_fraction < 0.0 or occupied_volume_fraction >= 1.0:
        raise ValueError(
            "ionic occupied volume fraction must satisfy 0 <= phi < 1, "
            f"got {occupied_volume_fraction}"
        )
    return occupied_volume_fraction


def persistent_network_occupied_volume_fraction(
    composition: CompositionState,
    matrix: MatrixState,
    speciation: SpeciationState,
    site_measure: TransportSiteMeasure,
    temperature_K: float,
) -> float:
    _assert_positive_float(temperature_K, "temperature_K")
    occupied_volume_fraction = 0.0
    for motif_name, concentration_M in speciation.motif_concentrations_M.items():
        _assert_nonnegative_float(concentration_M, f"speciation.motif_concentrations_M.{motif_name}")
        partial_molar_volume_cm3_mol = _motif_partial_molar_volume_cm3_mol(
            motif_name,
            composition,
            site_measure,
        )
        persistence_factor = _motif_network_persistence_factor(
            motif_name,
            matrix,
            site_measure,
            temperature_K,
        )
        occupied_volume_fraction += (
            concentration_M
            * partial_molar_volume_cm3_mol
            * persistence_factor
            / LITER_TO_ML
        )
    if occupied_volume_fraction < 0.0 or occupied_volume_fraction >= 1.0:
        raise ValueError(
            "persistent network occupied volume fraction must satisfy 0 <= phi < 1, "
            f"got {occupied_volume_fraction}"
        )
    return occupied_volume_fraction


def crowding_factor_from_ionic_volume_fraction(occupied_volume_fraction: float) -> float:
    if occupied_volume_fraction < 0.0 or occupied_volume_fraction >= 1.0:
        raise ValueError(
            "crowding factor requires 0 <= ionic occupied volume fraction < 1, "
            f"got {occupied_volume_fraction}"
        )
    numerator = 1.0 - occupied_volume_fraction
    denominator = 1.0 + occupied_volume_fraction
    factor = (numerator / denominator) ** 2
    _assert_positive_float(factor, "crowding_factor")
    return factor


def _motif_partial_molar_volume_cm3_mol(
    motif_name: str,
    composition: CompositionState,
    site_measure: TransportSiteMeasure,
) -> float:
    anion_site_by_id = site_measure.anion_by_canonical_id()
    ligand_site_by_id = site_measure.neutral_ligand_by_canonical_id()
    if motif_name.startswith("free_cation:"):
        _motif_feature_id(motif_name, "free_cation:")
        return site_measure.cation.molar_volume_cm3_mol
    if motif_name.startswith("free_anion:"):
        anion_feature_id = _motif_feature_id(motif_name, "free_anion:")
        return anion_site_by_id[anion_feature_id].anion_molar_volume_cm3_mol
    if motif_name.startswith("SSIP:"):
        anion_feature_id = _motif_feature_id(motif_name, "SSIP:")
        return site_measure.cation.molar_volume_cm3_mol + anion_site_by_id[anion_feature_id].anion_molar_volume_cm3_mol
    if motif_name.startswith("CIP:"):
        anion_feature_id = _motif_feature_id(motif_name, "CIP:")
        return site_measure.cation.molar_volume_cm3_mol + anion_site_by_id[anion_feature_id].anion_molar_volume_cm3_mol
    if motif_name.startswith("AGG:"):
        anion_feature_id = _motif_feature_id(motif_name, "AGG:")
        return site_measure.cation.molar_volume_cm3_mol + anion_site_by_id[anion_feature_id].anion_molar_volume_cm3_mol
    if motif_name.startswith("BRIDGE_NETWORK:"):
        anion_feature_id = _motif_feature_id(motif_name, "BRIDGE_NETWORK:")
        return 2.0 * (
            site_measure.cation.molar_volume_cm3_mol
            + anion_site_by_id[anion_feature_id].anion_molar_volume_cm3_mol
        )
    if motif_name.startswith("Li_ligand:"):
        ligand_feature_id = _motif_feature_id(motif_name, "Li_ligand:")
        return site_measure.cation.molar_volume_cm3_mol + ligand_site_by_id[ligand_feature_id].molecular_volume_cm3_mol
    raise ValueError(f"Unhandled motif concentration key {motif_name}")


def _motif_network_persistence_factor(
    motif_name: str,
    matrix: MatrixState,
    site_measure: TransportSiteMeasure,
    temperature_K: float,
) -> float:
    if motif_name.startswith("Li_ligand:"):
        ligand_feature_id = _motif_feature_id(motif_name, "Li_ligand:")
        ligand_site = site_measure.neutral_ligand_by_canonical_id()[ligand_feature_id]
        return _neutral_ligand_network_persistence_factor(
            ligand_site,
            site_measure.cation,
            matrix,
            temperature_K,
        )
    return 1.0


def _neutral_ligand_network_persistence_factor(
    ligand_site: NeutralLigandSiteFeature,
    cation_site: CationSiteFeature,
    matrix: MatrixState,
    temperature_K: float,
) -> float:
    bound_lifetime_s = _neutral_ligand_bound_lifetime_s(
        ligand_site,
        cation_site,
        matrix,
        temperature_K,
    )
    hop_lifetime_s = _neutral_ligand_cage_hop_lifetime_s(
        ligand_site,
        cation_site,
        matrix,
        temperature_K,
    )
    persistence_factor = bound_lifetime_s / (bound_lifetime_s + hop_lifetime_s)
    if persistence_factor < 0.0 or persistence_factor > 1.0 or not math.isfinite(persistence_factor):
        raise ValueError(
            f"neutral ligand persistence factor must satisfy 0 <= p <= 1, got {persistence_factor}"
        )
    return persistence_factor


def _neutral_ligand_bound_lifetime_s(
    ligand_site: NeutralLigandSiteFeature,
    cation_site: CationSiteFeature,
    matrix: MatrixState,
    temperature_K: float,
) -> float:
    association_rate_M_inv_s = _smoluchowski_association_rate_M_inv_s(
        ligand_site,
        cation_site,
        matrix,
        temperature_K,
    )
    bound_lifetime_s = ligand_site.coordination_affinity_M_inv / association_rate_M_inv_s
    _assert_positive_float(bound_lifetime_s, f"{ligand_site.canonical_feature_id}.bound_lifetime_s")
    return bound_lifetime_s


def _neutral_ligand_cage_hop_lifetime_s(
    ligand_site: NeutralLigandSiteFeature,
    cation_site: CationSiteFeature,
    matrix: MatrixState,
    temperature_K: float,
) -> float:
    capture_radius_m = _neutral_ligand_capture_radius_m(ligand_site, cation_site)
    cation_diffusivity_m2_s = _stokes_diffusivity_m2_s(
        cation_site.solvated_radius_A * ANGSTROM_TO_M,
        matrix,
        temperature_K,
    )
    hop_lifetime_s = (
        capture_radius_m
        * capture_radius_m
        / (STOKES_SPHERE_DRAG_FACTOR * cation_diffusivity_m2_s)
    )
    _assert_positive_float(hop_lifetime_s, f"{ligand_site.canonical_feature_id}.hop_lifetime_s")
    return hop_lifetime_s


def _smoluchowski_association_rate_M_inv_s(
    ligand_site: NeutralLigandSiteFeature,
    cation_site: CationSiteFeature,
    matrix: MatrixState,
    temperature_K: float,
) -> float:
    ligand_radius_m = _neutral_ligand_hydrodynamic_radius_m(ligand_site)
    cation_radius_m = cation_site.solvated_radius_A * ANGSTROM_TO_M
    capture_radius_m = ligand_radius_m + cation_radius_m
    ligand_diffusivity_m2_s = _stokes_diffusivity_m2_s(ligand_radius_m, matrix, temperature_K)
    cation_diffusivity_m2_s = _stokes_diffusivity_m2_s(cation_radius_m, matrix, temperature_K)
    association_rate_M_inv_s = (
        4.0
        * math.pi
        * (ligand_diffusivity_m2_s + cation_diffusivity_m2_s)
        * capture_radius_m
        * N_A
        * LITER_TO_ML
    )
    _assert_positive_float(association_rate_M_inv_s, f"{ligand_site.canonical_feature_id}.association_rate_M_inv_s")
    return association_rate_M_inv_s


def _neutral_ligand_capture_radius_m(
    ligand_site: NeutralLigandSiteFeature,
    cation_site: CationSiteFeature,
) -> float:
    return _neutral_ligand_hydrodynamic_radius_m(ligand_site) + cation_site.solvated_radius_A * ANGSTROM_TO_M


def _neutral_ligand_hydrodynamic_radius_m(ligand_site: NeutralLigandSiteFeature) -> float:
    molecular_volume_m3 = ligand_site.molecular_volume_cm3_mol * 1.0e-6 / N_A
    _assert_positive_float(molecular_volume_m3, f"{ligand_site.canonical_feature_id}.molecular_volume_m3")
    radius_m = (3.0 * molecular_volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    _assert_positive_float(radius_m, f"{ligand_site.canonical_feature_id}.hydrodynamic_radius_m")
    return radius_m


def _stokes_diffusivity_m2_s(
    radius_m: float,
    matrix: MatrixState,
    temperature_K: float,
) -> float:
    _assert_positive_float(radius_m, "stokes_radius_m")
    diffusivity_m2_s = (
        K_B
        * temperature_K
        / (
            STOKES_SPHERE_DRAG_FACTOR
            * math.pi
            * matrix.eta_solution_cP
            * CP_TO_PA_S
            * radius_m
        )
    )
    _assert_positive_float(diffusivity_m2_s, "stokes_diffusivity_m2_s")
    return diffusivity_m2_s


def _microviscosity_coupling_exponent(
    fractional_stokes_alpha: float,
    shell_persistence_factor: float,
    context: str,
) -> float:
    if fractional_stokes_alpha <= 0.0 or fractional_stokes_alpha > 1.0 or not math.isfinite(fractional_stokes_alpha):
        raise ValueError(f"{context}.fractional_stokes_alpha must satisfy 0 < alpha <= 1, got {fractional_stokes_alpha}")
    if shell_persistence_factor < 0.0 or shell_persistence_factor > 1.0 or not math.isfinite(shell_persistence_factor):
        raise ValueError(f"{context}.shell_persistence_factor must satisfy 0 <= p <= 1, got {shell_persistence_factor}")
    coupling_exponent = fractional_stokes_alpha + (1.0 - fractional_stokes_alpha) * shell_persistence_factor
    if coupling_exponent <= 0.0 or coupling_exponent > 1.0 or not math.isfinite(coupling_exponent):
        raise ValueError(f"{context}.microviscosity_coupling_exponent must satisfy 0 < xi <= 1, got {coupling_exponent}")
    return coupling_exponent


def transport_microviscosity_cP(
    reference_viscosity_cP: float,
    matrix: MatrixState,
    viscosity_exponent: float,
) -> float:
    _assert_positive_float(reference_viscosity_cP, "reference_viscosity_cP")
    if viscosity_exponent <= 0.0 or viscosity_exponent > 1.0 or not math.isfinite(viscosity_exponent):
        raise ValueError(f"transport viscosity exponent must satisfy 0 < exponent <= 1, got {viscosity_exponent}")
    liquid_microviscosity_cP = (
        reference_viscosity_cP
        * (matrix.eta_liquid_cP / reference_viscosity_cP) ** viscosity_exponent
    )
    transport_viscosity_cP = (
        liquid_microviscosity_cP
        * matrix.dimer_viscosity_factor
        * matrix.salt_viscosity_factor
        * matrix.salt_additive_viscosity_factor
    )
    _assert_positive_float(transport_viscosity_cP, "transport_microviscosity_cP")
    return transport_viscosity_cP


def _solvation_shell_persistence_factor(
    matrix: MatrixState,
    solvation: SolvationState,
    cation_site: CationSiteFeature,
    mobile_radius_m: float,
    temperature_K: float,
) -> float:
    shell_lifetime_s = _weighted_solvation_shell_lifetime_s(
        matrix,
        solvation,
        cation_site,
        temperature_K,
    )
    cage_crossing_lifetime_s = _solvation_cage_crossing_lifetime_s(
        matrix,
        solvation,
        mobile_radius_m,
        temperature_K,
    )
    persistence_factor = shell_lifetime_s / (shell_lifetime_s + cage_crossing_lifetime_s)
    if persistence_factor < 0.0 or persistence_factor > 1.0 or not math.isfinite(persistence_factor):
        raise ValueError(f"solvation shell persistence factor must satisfy 0 <= p <= 1, got {persistence_factor}")
    return persistence_factor


def _weighted_solvation_shell_lifetime_s(
    matrix: MatrixState,
    solvation: SolvationState,
    cation_site: CationSiteFeature,
    temperature_K: float,
) -> float:
    shell_lifetime_s = 0.0
    for neutral_species_name, shell_fraction in solvation.shell_fractions.items():
        _assert_nonnegative_float(shell_fraction, f"solvation.shell_fractions.{neutral_species_name}")
        neutral_props = _neutral_species_props(neutral_species_name)
        neutral_lifetime_s = _neutral_species_shell_lifetime_s(
            neutral_species_name,
            neutral_props,
            cation_site,
            matrix,
            temperature_K,
        )
        shell_lifetime_s += shell_fraction * neutral_lifetime_s
    _assert_positive_float(shell_lifetime_s, "weighted_solvation_shell_lifetime_s")
    return shell_lifetime_s


def _neutral_species_shell_lifetime_s(
    neutral_species_name: str,
    neutral_props: Mapping[str, Any],
    cation_site: CationSiteFeature,
    matrix: MatrixState,
    temperature_K: float,
) -> float:
    context = f"neutral species {neutral_species_name}"
    if "residence_time_ns" in neutral_props:
        residence_time_ns = _require_float(neutral_props, "residence_time_ns", context)
        _assert_positive_float(residence_time_ns, f"{context}.residence_time_ns")
        return residence_time_ns * NANOSECOND_TO_SECOND
    coordination_affinity_M_inv = _require_float(neutral_props, "coordination_affinity_M_inv", context)
    _assert_positive_float(coordination_affinity_M_inv, f"{context}.coordination_affinity_M_inv")
    association_rate_M_inv_s = _neutral_species_association_rate_M_inv_s(
        neutral_props,
        cation_site,
        matrix,
        temperature_K,
        context,
    )
    bound_lifetime_s = coordination_affinity_M_inv / association_rate_M_inv_s
    _assert_positive_float(bound_lifetime_s, f"{context}.derived_shell_lifetime_s")
    return bound_lifetime_s


def _solvation_cage_crossing_lifetime_s(
    matrix: MatrixState,
    solvation: SolvationState,
    mobile_radius_m: float,
    temperature_K: float,
) -> float:
    _assert_positive_float(mobile_radius_m, "mobile_radius_m")
    weighted_neutral_radius_m = _weighted_neutral_shell_radius_m(solvation)
    shell_path_length_m = (
        mobile_radius_m
        + solvation.preferred_coordination_number * weighted_neutral_radius_m
    )
    _assert_positive_float(shell_path_length_m, "solvation_shell_path_length_m")
    mobile_diffusivity_m2_s = _stokes_diffusivity_m2_s(mobile_radius_m, matrix, temperature_K)
    cage_crossing_lifetime_s = (
        shell_path_length_m
        * shell_path_length_m
        / (STOKES_SPHERE_DRAG_FACTOR * mobile_diffusivity_m2_s)
    )
    _assert_positive_float(cage_crossing_lifetime_s, "solvation_cage_crossing_lifetime_s")
    return cage_crossing_lifetime_s


def _weighted_neutral_shell_radius_m(
    solvation: SolvationState,
) -> float:
    weighted_radius_m = 0.0
    for neutral_species_name, shell_fraction in solvation.shell_fractions.items():
        _assert_nonnegative_float(shell_fraction, f"solvation.shell_fractions.{neutral_species_name}")
        neutral_props = _neutral_species_props(neutral_species_name)
        weighted_radius_m += shell_fraction * _neutral_species_hydrodynamic_radius_m(
            neutral_props,
            f"neutral species {neutral_species_name}",
        )
    _assert_positive_float(weighted_radius_m, "weighted_neutral_shell_radius_m")
    return weighted_radius_m


def _neutral_species_association_rate_M_inv_s(
    neutral_props: Mapping[str, Any],
    cation_site: CationSiteFeature,
    matrix: MatrixState,
    temperature_K: float,
    context: str,
) -> float:
    neutral_radius_m = _neutral_species_hydrodynamic_radius_m(neutral_props, context)
    cation_radius_m = cation_site.solvated_radius_A * ANGSTROM_TO_M
    capture_radius_m = neutral_radius_m + cation_radius_m
    neutral_diffusivity_m2_s = _stokes_diffusivity_m2_s(neutral_radius_m, matrix, temperature_K)
    cation_diffusivity_m2_s = _stokes_diffusivity_m2_s(cation_radius_m, matrix, temperature_K)
    association_rate_M_inv_s = (
        4.0
        * math.pi
        * (neutral_diffusivity_m2_s + cation_diffusivity_m2_s)
        * capture_radius_m
        * N_A
        * LITER_TO_ML
    )
    _assert_positive_float(association_rate_M_inv_s, f"{context}.association_rate_M_inv_s")
    return association_rate_M_inv_s


def _neutral_species_hydrodynamic_radius_m(
    neutral_props: Mapping[str, Any],
    context: str,
) -> float:
    molecular_weight_g_mol = _require_float(neutral_props, "molecular_weight", context)
    density_g_ml = _require_float(neutral_props, "density_g_ml", context)
    _assert_positive_float(molecular_weight_g_mol, f"{context}.molecular_weight")
    _assert_positive_float(density_g_ml, f"{context}.density_g_ml")
    molar_volume_m3_mol = molecular_weight_g_mol / density_g_ml * CM3_TO_M3
    molecular_volume_m3 = molar_volume_m3_mol / N_A
    _assert_positive_float(molecular_volume_m3, f"{context}.molecular_volume_m3")
    radius_m = (3.0 * molecular_volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    _assert_positive_float(radius_m, f"{context}.hydrodynamic_radius_m")
    return radius_m


def _motif_feature_id(motif_name: str, prefix: str) -> str:
    if not motif_name.startswith(prefix):
        raise ValueError(f"Motif key {motif_name} does not start with {prefix}")
    feature_id = motif_name[len(prefix) :]
    if not feature_id:
        raise ValueError(f"Motif key {motif_name} has an empty feature id")
    return feature_id


def _build_mobility_state(
    composition: CompositionState,
    matrix: MatrixState,
    solvation: SolvationState,
    speciation: SpeciationState,
    temperature_K: float,
    params: OnsagerConductivityParams,
    physics_config: Mapping[str, Any],
    site_measure: TransportSiteMeasure,
) -> MobilityState:
    arrhenius_cfg = _require_mapping(physics_config, "transport_arrhenius", "physics_config")
    eta_ref = _require_float(arrhenius_cfg, "reference_viscosity_cP", "transport_arrhenius")
    ea = _require_float(arrhenius_cfg, "diffusion_activation_energy_J_mol", "transport_arrhenius")
    if eta_ref <= 0.0:
        raise ValueError("transport_arrhenius.reference_viscosity_cP must be positive")

    carrier_lambda_weighted_sum: dict[str, float] = {}
    carrier_concentration_sum: dict[str, float] = {}
    carrier_strength: dict[str, float] = {}
    carrier_strength_no_crowding: dict[str, float] = {}
    feature_splits: dict[str, dict[str, float]] = {}
    feature_no_shape_splits: dict[str, dict[str, float]] = {}
    anion_shape_factor_by_feature: dict[str, float] = {}
    anion_microviscosity_coupling_by_feature: dict[str, float] = {}
    temp_factor = math.exp(-ea / R * (1.0 / temperature_K - 1.0 / T_REF_K))
    raw_volume_fraction = raw_ionic_occupied_volume_fraction(composition, speciation, site_measure)
    network_volume_fraction = persistent_network_occupied_volume_fraction(
        composition,
        matrix,
        speciation,
        site_measure,
        temperature_K,
    )
    crowding_factor = crowding_factor_from_ionic_volume_fraction(network_volume_fraction)
    shared_cation_symbol = site_measure.cation.ion_symbol
    lambda_cation0 = _shared_cation_mobility_baseline_S_cm2_mol(
        site_measure.cation,
        eta_ref,
        T_REF_K,
    )
    alpha_cation = site_measure.cation.stokes_einstein_alpha
    cation_microviscosity_coupling = _microviscosity_coupling_exponent(
        fractional_stokes_alpha=alpha_cation,
        shell_persistence_factor=_solvation_shell_persistence_factor(
            matrix,
            solvation,
            site_measure.cation,
            site_measure.cation.solvated_radius_A * ANGSTROM_TO_M,
            temperature_K,
        ),
        context=f"cation feature {site_measure.cation.canonical_feature_id}",
    )
    lambda_cation_no_crowding = (
        lambda_cation0
        * params.mobility_scale
        * temp_factor
        * (
            eta_ref
            / transport_microviscosity_cP(
                reference_viscosity_cP=eta_ref,
                matrix=matrix,
                viscosity_exponent=params.viscosity_exponent_scale
                * alpha_cation
                * cation_microviscosity_coupling,
            )
        )
    )
    lambda_cation = lambda_cation_no_crowding * crowding_factor

    for anion_site in site_measure.anion_sites:
        anion_feature_id = anion_site.canonical_feature_id
        anion_carrier = anion_site.carrier_label
        lambda0_total = anion_site.limiting_molar_conductivity_S_cm2_mol
        if lambda0_total <= 0.0:
            raise ValueError(f"anion feature {anion_feature_id}.limiting_molar_conductivity must be positive")

        lambda_anion0 = lambda0_total - lambda_cation0
        if lambda_anion0 <= 0.0:
            raise ValueError(
                f"anion feature {anion_feature_id} limiting mobility is not larger than shared cation mobility baseline "
                f"{lambda_cation0}"
            )

        alpha_anion = anion_site.stokes_einstein_alpha_anion
        anion_microviscosity_coupling = _microviscosity_coupling_exponent(
            fractional_stokes_alpha=alpha_anion,
            shell_persistence_factor=_solvation_shell_persistence_factor(
                matrix,
                solvation,
                site_measure.cation,
                anion_site.anion_radius_A * ANGSTROM_TO_M,
                temperature_K,
            ),
            context=f"anion feature {anion_feature_id}",
        )
        lambda_anion_no_crowding_no_shape = (
            lambda_anion0
            * params.mobility_scale
            * temp_factor
            * (
                eta_ref
                / transport_microviscosity_cP(
                    reference_viscosity_cP=eta_ref,
                    matrix=matrix,
                    viscosity_exponent=params.viscosity_exponent_scale
                    * alpha_anion
                    * anion_microviscosity_coupling,
                )
            )
        )
        anion_shape_factor = anion_site.shape_friction_factor
        lambda_anion_no_crowding = lambda_anion_no_crowding_no_shape / anion_shape_factor
        lambda_anion_no_shape = lambda_anion_no_crowding_no_shape * crowding_factor
        lambda_anion = lambda_anion_no_shape / anion_shape_factor
        anion_shape_factor_by_feature[anion_feature_id] = anion_shape_factor
        anion_microviscosity_coupling_by_feature[anion_feature_id] = anion_microviscosity_coupling

        feature_splits[anion_feature_id] = {
            shared_cation_symbol: lambda_cation,
            anion_carrier: lambda_anion,
        }
        feature_no_shape_splits[anion_feature_id] = {
            shared_cation_symbol: lambda_cation,
            anion_carrier: lambda_anion_no_shape,
        }
        free_cation_concentration = _require_float(
            speciation.motif_concentrations_M,
            f"free_cation:{anion_feature_id}",
            "speciation.motif_concentrations_M",
        )
        free_anion_concentration = _require_float(
            speciation.motif_concentrations_M,
            f"free_anion:{anion_feature_id}",
            "speciation.motif_concentrations_M",
        )
        _accumulate_float(carrier_lambda_weighted_sum, shared_cation_symbol, free_cation_concentration * lambda_cation)
        _accumulate_float(carrier_lambda_weighted_sum, anion_carrier, free_anion_concentration * lambda_anion)
        _accumulate_float(carrier_concentration_sum, shared_cation_symbol, free_cation_concentration)
        _accumulate_float(carrier_concentration_sum, anion_carrier, free_anion_concentration)
        _accumulate_float(
            carrier_strength_no_crowding,
            shared_cation_symbol,
            free_cation_concentration * lambda_cation_no_crowding,
        )
        _accumulate_float(
            carrier_strength_no_crowding,
            anion_carrier,
            free_anion_concentration * lambda_anion_no_crowding,
        )
        _accumulate_float(carrier_strength, shared_cation_symbol, free_cation_concentration * lambda_cation)
        _accumulate_float(carrier_strength, anion_carrier, free_anion_concentration * lambda_anion)

    carrier_lambda: dict[str, float] = {}
    for carrier_name, weighted_lambda_sum in carrier_lambda_weighted_sum.items():
        concentration_sum = carrier_concentration_sum[carrier_name]
        _assert_positive_float(concentration_sum, f"carrier concentration sum {carrier_name}")
        carrier_lambda[carrier_name] = weighted_lambda_sum / concentration_sum

    return MobilityState(
        carrier_lambda_S_cm2_mol=carrier_lambda,
        carrier_strength_mS_cm=carrier_strength,
        carrier_strength_no_crowding_mS_cm=carrier_strength_no_crowding,
        feature_lambda_split_S_cm2_mol=feature_splits,
        feature_lambda_no_shape_S_cm2_mol=feature_no_shape_splits,
        anion_shape_factor_by_feature=anion_shape_factor_by_feature,
        cation_microviscosity_coupling_exponent=cation_microviscosity_coupling,
        anion_microviscosity_coupling_exponent_by_feature=anion_microviscosity_coupling_by_feature,
        reference_viscosity_cP=eta_ref,
        raw_ionic_occupied_volume_fraction=raw_volume_fraction,
        network_occupied_volume_fraction=network_volume_fraction,
        crowding_factor=crowding_factor,
    )


def _build_correlation_state(
    composition: CompositionState,
    matrix: MatrixState,
    solvation: SolvationState,
    speciation: SpeciationState,
    mobility: MobilityState,
    physics_config: Mapping[str, Any],
    params: OnsagerConductivityParams,
    site_measure: TransportSiteMeasure,
) -> CorrelationState:
    pairing_cfg = _require_mapping(physics_config, "ion_pairing_model", "physics_config")
    eps_ref = _require_float(pairing_cfg, "bjerrum_eps_ref", "ion_pairing_model")

    cation_symbol = site_measure.cation.ion_symbol
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

    anion_site_by_carrier = {site.carrier_label: site for site in site_measure.anion_sites}
    for idx, carrier in enumerate(carrier_order[1:], start=1):
        if carrier not in anion_site_by_carrier:
            raise ValueError(f"carrier {carrier} has no anion site feature")
        anion_site = anion_site_by_carrier[carrier]
        anion_feature_id = anion_site.canonical_feature_id
        anion_radius = anion_site.anion_radius_A
        cation_radius = site_measure.cation.solvated_radius_A
        dielectric_support = matrix.epsilon_effective / (matrix.epsilon_effective + eps_ref)
        anion_size_share = anion_radius / (anion_radius + cation_radius)
        memory_fraction = _motif_memory_fraction(speciation, anion_feature_id, dielectric_support)
        decoupling_fraction = _motif_decoupling_fraction(speciation, anion_feature_id)
        raw = (
            params.pair_correlation_gain * memory_fraction
            - params.steric_anticorrelation_gain
            * (
                decoupling_fraction
                + solvation.shell_steric_disruption
                * dielectric_support
                * anion_size_share
            )
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


def _motif_memory_fraction(
    speciation: SpeciationState,
    anion_feature_id: str,
    dielectric_support: float,
) -> float:
    cip_fraction = speciation.cip_fraction_by_feature[anion_feature_id]
    ssip_fraction = speciation.ssip_fraction_by_feature[anion_feature_id]
    aggregate_fraction = speciation.aggregate_fraction_by_feature[anion_feature_id]
    bridge_fraction = speciation.bridge_network_fraction_by_feature[anion_feature_id]
    ssip_memory_weight = _ssip_memory_weight(dielectric_support)
    memory_fraction = (
        cip_fraction
        + ssip_memory_weight * ssip_fraction
        + aggregate_fraction
        + bridge_fraction
    )
    _assert_nonnegative_float(memory_fraction, f"{anion_feature_id}.motif_memory_fraction")
    return memory_fraction


def _ssip_memory_weight(dielectric_support: float) -> float:
    if dielectric_support < 0.0 or dielectric_support > 1.0 or not math.isfinite(dielectric_support):
        raise ValueError(f"dielectric_support must satisfy 0 <= x <= 1, got {dielectric_support}")
    return dielectric_support * (1.0 - dielectric_support)


def _motif_decoupling_fraction(
    speciation: SpeciationState,
    anion_feature_id: str,
) -> float:
    ligand_fraction = math.fsum(speciation.li_ligand_fraction_by_feature.values())
    decoupling_fraction = (
        speciation.ssip_fraction_by_feature[anion_feature_id]
        + ligand_fraction
    )
    _assert_nonnegative_float(decoupling_fraction, f"{anion_feature_id}.motif_decoupling_fraction")
    return decoupling_fraction


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


def _compute_salt_additive_viscosity_factor(
    composition: CompositionState,
    viscosity_cfg: Mapping[str, Any],
    eta_liquid_cP: float,
    salt_viscosity_scale: float,
) -> float:
    _assert_positive_float(eta_liquid_cP, "eta_liquid_cP")
    jones_dole_by_species = _require_mapping(
        viscosity_cfg,
        "jones_dole_B_by_species",
        "viscosity_model",
    )
    b_vis = _require_float(viscosity_cfg, "jones_dole_B_vis", "viscosity_model")
    b_linear_ref = _require_float(viscosity_cfg, "jones_dole_B_linear_ref", "viscosity_model")
    d_vis = _require_float(viscosity_cfg, "jones_dole_D_vis", "viscosity_model")
    _assert_positive_float(b_linear_ref, "viscosity_model.jones_dole_B_linear_ref")
    _assert_positive_float(composition.total_mass_g, "composition.total_mass_g")

    log_factor = 0.0
    for source_name, molarity_M in composition.ionic_source_molarities_M.items():
        _assert_nonnegative_float(molarity_M, f"ionic source {source_name} molarity")
        if molarity_M == 0.0:
            continue
        b_linear = _require_float(
            jones_dole_by_species,
            source_name,
            "viscosity_model.jones_dole_B_by_species",
        )
        source_viscosity_strength = (
            b_linear
            * b_vis
            / b_linear_ref
            * molarity_M
            + d_vis
            * molarity_M
            * molarity_M
        )
        for additive_name, additive_mass_g in composition.additive_masses_g.items():
            _assert_nonnegative_float(additive_mass_g, f"additive {additive_name} mass")
            if additive_mass_g == 0.0:
                continue
            additive_props = _require_species(ADDITIVES, additive_name, "additive")
            if _is_ionic_source(additive_props):
                continue
            additive_viscosity_cP = _require_float(
                additive_props,
                "viscosity_cP",
                f"additive {additive_name}",
            )
            _assert_positive_float(additive_viscosity_cP, f"additive {additive_name} viscosity_cP")
            additive_weight_fraction = additive_mass_g / composition.total_mass_g
            viscosity_contrast = additive_viscosity_cP / eta_liquid_cP - 1.0
            log_factor += source_viscosity_strength * additive_weight_fraction * viscosity_contrast
    return math.exp(salt_viscosity_scale * log_factor)


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


def _is_ionic_source(props: Mapping[str, Any]) -> bool:
    has_cation_identity = "cation" in props or "cation_radius" in props
    return has_cation_identity and "anion" in props and "Lambda_0" in props


def _ionic_source_props(source: str) -> Mapping[str, Any]:
    if source in SALTS:
        return SALTS[source]
    if source in ADDITIVES:
        props = ADDITIVES[source]
        if _is_ionic_source(props):
            return props
    raise ValueError(f"Species {source} is not an ionic source")


def _ionic_source_cation_name(source: str, props: Mapping[str, Any]) -> str:
    if "cation" in props:
        return _require_string(props, "cation", f"ionic source {source}")
    cation_radius_A = _require_float(props, "cation_radius", f"ionic source {source}")
    matches: list[str] = []
    for cation_name, cation_props in CATION_PROPERTIES.items():
        reference_radius_A = _require_float(cation_props, "ionic_radius_A", f"cation {cation_name}")
        if abs(reference_radius_A - cation_radius_A) <= CATION_RADIUS_MATCH_TOLERANCE_A:
            matches.append(cation_name)
    if len(matches) != 1:
        raise ValueError(
            f"ionic source {source} cation_radius {cation_radius_A} A matched cations {matches}"
        )
    return matches[0]


def _ionic_source_preferred_coordination_number(source: str, props: Mapping[str, Any]) -> float:
    if "preferred_coordination_number" in props:
        return _require_float(props, "preferred_coordination_number", f"ionic source {source}")
    cation_name = _ionic_source_cation_name(source, props)
    reference_values: list[float] = []
    for salt_name, salt_props in SALTS.items():
        salt_cation_name = _ionic_source_cation_name(salt_name, salt_props)
        if salt_cation_name != cation_name:
            continue
        reference_values.append(
            _require_float(
                salt_props,
                "preferred_coordination_number",
                f"salt reference {salt_name}",
            )
        )
    if not reference_values:
        raise ValueError(
            f"ionic source {source} has no cation-family preferred coordination reference"
        )
    reference_min = min(reference_values)
    reference_max = max(reference_values)
    if reference_max - reference_min > COORDINATION_NUMBER_MATCH_TOLERANCE:
        raise ValueError(
            f"ionic source {source} cation {cation_name} has non-unique coordination references "
            f"{reference_values}"
        )
    return reference_values[0]


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


def _single_cation_name(composition: CompositionState) -> str:
    cation_names: set[str] = set()
    for source in composition.ionic_source_molarities_M:
        props = _ionic_source_props(source)
        cation_names.add(_ionic_source_cation_name(source, props))
    if len(cation_names) != 1:
        raise ValueError(f"Onsager prototype currently requires one cation family, found {sorted(cation_names)}")
    return next(iter(cation_names))


def _single_cation_symbol(composition: CompositionState) -> str:
    symbols: set[str] = set()
    for source in composition.ionic_source_molarities_M:
        props = _ionic_source_props(source)
        cation = _ionic_source_cation_name(source, props)
        cation_props = _require_species(CATION_PROPERTIES, cation, "cation")
        symbols.add(_require_string(cation_props, "ion_symbol", f"cation {cation}"))
    if len(symbols) != 1:
        raise ValueError(f"Onsager prototype currently requires one cation family, found {sorted(symbols)}")
    return next(iter(symbols))


def _shared_cation_mobility_baseline_S_cm2_mol(
    cation_feature: CationSiteFeature,
    reference_viscosity_cP: float,
    reference_temperature_K: float,
) -> float:
    solvated_radius_A = cation_feature.solvated_radius_A
    charge = float(cation_feature.charge)
    _assert_positive_float(solvated_radius_A, f"cation feature {cation_feature.canonical_feature_id} solvated_radius_A")
    _assert_positive_float(reference_viscosity_cP, "reference_viscosity_cP")
    _assert_positive_float(reference_temperature_K, "reference_temperature_K")
    radius_m = solvated_radius_A * ANGSTROM_TO_NM * 1.0e-9
    diffusion_m2_s = (
        K_B
        * reference_temperature_K
        / (STOKES_SPHERE_DRAG_FACTOR * math.pi * reference_viscosity_cP * CP_TO_PA_S * radius_m)
    )
    mobility_S_m2_mol = charge * charge * F * F * diffusion_m2_s / (R * reference_temperature_K)
    mobility_S_cm2_mol = mobility_S_m2_mol * CM2_PER_M2
    _assert_positive_float(
        mobility_S_cm2_mol,
        f"shared cation mobility baseline {cation_feature.canonical_feature_id}",
    )
    return mobility_S_cm2_mol


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


def species_record(registry: Mapping[str, Mapping[str, Any]], name: str, role: str) -> Mapping[str, Any]:
    if name not in registry:
        raise ValueError(f"Unknown {role} species {name}")
    return registry[name]


_require_species = species_record


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


def _assert_positive_float(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{context} must be a positive finite number, got {value}")


def _assert_nonnegative_float(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{context} must be a non-negative finite number, got {value}")
