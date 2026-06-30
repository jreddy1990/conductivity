"""Molecular property-DB adapter for descriptor-neutral conductivity primitives."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from constants import F, N_A, PA_PER_ATM, R, S_M_TO_MS_CM, T_REF_K
from conductivity.fit_conductivity_primitive_parameters import (
    PrimitiveParameterTransform,
    PrimitiveFitOptions,
    PrimitiveFitDatasetEvaluation,
)
from conductivity.generic_speciation import (
    STANDARD_STATE_CONCENTRATION_MOL_M3,
    cluster_activity_correction_J_mol,
)
from conductivity.molecular_descriptors import (
    MolecularSpeciesInput,
    ProvidedPropertyDescriptorBackend,
    ROLE_ADDITIVE,
    ROLE_ANION,
    ROLE_CATION,
    ROLE_SOLVENT,
)
from conductivity.molecular_electrolyte_mori_generator import (
    MolecularElectrolyteRecipe,
    MolecularMixtureProperties,
    MolecularMoriConductivityResult,
    MolecularMoriOptions,
    TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
    TRANSPORT_ROLE_CLUSTER_COM_CENTER,
    TRANSPORT_ROLE_CONTACT_PAIR_CENTER,
    TRANSPORT_ROLE_FREE_ION_CENTER,
    TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
    TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    compute_molecular_electrolyte_conductivity_with_diagnostic_cluster_shifts,
)
from conductivity.molecular_primitive_parameters import (
    CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES,
    CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME,
    ConductivityPrimitiveParameterSet,
    PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED,
    PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE,
    conductivity_primitive_parameters_from_mapping,
    validate_conductivity_primitive_parameters,
)


PHYSICS_CONFIG_PATH = Path("config/physics.json")
OPTIMIZATION_CONFIG_PATH = Path("config/optimization.json")
PROPERTY_DB_AUDIT_CONFIG_KEY = "molecular_property_db_audit"
PRIMITIVE_PARAMETER_CONFIG_KEY = "molecular_conductivity_primitive_parameters"
PRIMITIVE_PARAMETER_FIT_CONFIG_KEY = "molecular_primitive_parameter_fit"
PRIMITIVE_PARAMETER_CONFIG_METADATA_KEYS = ("description",)
CUBIC_CENTIMETER_TO_CUBIC_ANGSTROM = 1.0e24  # Unit conversion: cm^3 to A^3.
CUBIC_ANGSTROM_PER_MOLE_TO_CM3 = 1.0e-24  # Unit conversion: A^3 molecule^-1 to cm^3 molecule^-1.
GRAMS_PER_LITER_PER_G_ML = 1000.0  # Unit conversion: g/mL to g/L.
SPHERE_VOLUME_FACTOR = 4.0 / 3.0  # Sphere volume coefficient: 4*pi*r^3/3.
SPHERE_AREA_FACTOR = 4.0  # Sphere area coefficient: 4*pi*r^2.
POLARIZABILITY_CLAUSIUS_MOSSOTTI_FACTOR = 3.0 / (4.0 * math.pi)
TRANSPORT_ROLE_DIRECT_CORRECTOR_ATTRIBUTION_LABELS = (
    TRANSPORT_ROLE_FREE_ION_CENTER,
    TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
    TRANSPORT_ROLE_CLUSTER_COM_CENTER,
    TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
    TRANSPORT_ROLE_CONTACT_PAIR_CENTER,
)


@dataclass(frozen=True)
class MolecularPropertyDbAuditOptions:
    liquid_occupied_volume_fraction: float
    reference_ion_viscosity_cP: float
    ion_reference_density_g_ml: float
    ion_reference_dielectric_base: float
    max_cluster_ion_count: int
    max_packing_fraction: float
    free_volume_exponent: float
    translation_jump_length_multiplier: float
    viscosity_monotonicity_scale: float
    packing_monotonicity_volume_scale: float
    parameter_consumption_perturbation_scales: tuple[float, ...]
    cluster_sensitivity_step_over_RT: float
    audit_worst_row_count: int
    validation_excluded_source_row_ids: tuple[int, ...]
    neutral_hbond_donor_count: int
    ion_hbond_donor_count: int
    ion_hbond_acceptor_count: int


@dataclass(frozen=True)
class MolecularPropertyDbRegistrySource:
    solvent_registry: Mapping[str, dict]
    salt_registry: Mapping[str, dict]
    additive_registry: Mapping[str, dict]
    cation_registry: Mapping[str, dict]


@dataclass(frozen=True)
class _MolecularPropertyDbRowState:
    solvent_loadings: Mapping[str, float]
    salt_loadings_M: Mapping[str, float]
    additive_loadings: Mapping[str, float]
    mixture_density_g_ml: float
    mixture_viscosity_cP: float
    mixture_dielectric_constant: float


@dataclass(frozen=True)
class MolecularPropertyDbCase:
    row_id: int
    source_row_ids: tuple[int, ...]
    source_formulation_key: tuple
    source_solvent_loadings: Mapping[str, float]
    source_salt_loadings_M: Mapping[str, float]
    source_additive_loadings: Mapping[str, float]
    recipe: MolecularElectrolyteRecipe
    species_inputs: Mapping[str, MolecularSpeciesInput]
    empirical_sigma_mS_cm: float
    empirical_sigma_spread_mS_cm: float


@dataclass(frozen=True)
class MolecularPropertyDbFormulationGroup:
    representative_row_id: int
    source_row_ids: tuple[int, ...]
    empirical_sigmas_mS_cm: tuple[float, ...]
    target_sigma_mS_cm: float
    empirical_sigma_spread_mS_cm: float
    solvent_loadings: Mapping[str, float]
    salt_loadings_M: Mapping[str, float]
    additive_loadings: Mapping[str, float]


@dataclass(frozen=True)
class MolecularPropertyDbExcludedRow:
    row_id: int
    empirical_sigma_mS_cm: float
    solvent_loadings: Mapping[str, float]
    salt_loadings_M: Mapping[str, float]
    additive_loadings: Mapping[str, float]


@dataclass(frozen=True)
class MolecularPropertyDbCaseSelection:
    source_labeled_rows: int
    cases: tuple[MolecularPropertyDbCase, ...]
    excluded_rows: tuple[MolecularPropertyDbExcludedRow, ...]
    formulation_groups: tuple[MolecularPropertyDbFormulationGroup, ...]


@dataclass(frozen=True)
class MolecularPropertyDbRowResult:
    row_id: int
    source_row_ids: tuple[int, ...]
    empirical_sigma_mS_cm: float
    empirical_sigma_spread_mS_cm: float
    predicted_sigma_mS_cm: float
    residual_mS_cm: float
    failed: bool
    failure_reason: str
    direct_sigma_mS_cm: float
    corrector_sigma_mS_cm: float
    direct_capacity_gap_mS_cm: float
    corrector_target_mS_cm: float
    corrector_residual_mS_cm: float
    direct_capacity_failure: bool
    corrector_too_strong_failure: bool
    corrector_too_weak_failure: bool
    direct_sigma_by_transport_role_mS_cm: Mapping[str, float]
    corrector_sigma_by_transport_role_mS_cm: Mapping[str, float]
    net_sigma_by_transport_role_mS_cm: Mapping[str, float]
    charge_weighted_transport_concentration_mol_m3: float
    mass_balance_residual_mol_m3: float
    row_sum_residual: float
    stationary_residual_mol_m3_s: float
    detailed_balance_residual_mol_m3_s: float
    event_reversal_residual_mol_m3_s: float
    free_ion_fraction: float
    charged_cluster_fraction: float
    neutral_cluster_fraction: float
    cluster_transport_mobility_density_mol_m_s: float
    charged_cluster_transport_mobility_density_mol_m_s: float
    neutral_cluster_transport_mobility_density_mol_m_s: float
    charged_cluster_direct_sigma_mS_cm: float
    charged_cluster_corrector_sigma_mS_cm: float
    charged_cluster_net_sigma_mS_cm: float
    cluster_thermodynamic_diagnostics: tuple[
        MolecularClusterThermodynamicDiagnostic,
        ...
    ]


@dataclass(frozen=True)
class MolecularClusterThermodynamicDiagnostic:
    row_id: int
    cluster_label: str
    cluster_kind: str
    stoichiometry: Mapping[str, int]
    net_charge_number: int
    concentration_mol_m3: float
    concentration_fraction_of_total_ion: float
    standard_free_energy_J_mol: float
    standard_free_energy_over_RT: float
    log_equilibrium_constant: float
    coulomb_J_mol: float
    desolvation_J_mol: float
    coordination_J_mol: float
    steric_J_mol: float
    entropy_J_mol: float
    activity_reference_J_mol: float
    activity_correction_J_mol: float
    hydrodynamic_radius_A: float
    molecular_volume_A3: float


@dataclass(frozen=True)
class MolecularClusterSensitivityDiagnostic:
    row_id: int
    cluster_label: str
    cluster_kind: str
    net_charge_number: int
    baseline_concentration_mol_m3: float
    baseline_deltaG_over_RT: float
    sigma_lower_deltaG_mS_cm: float
    sigma_higher_deltaG_mS_cm: float
    sensitivity_mS_cm_per_logK: float
    direction_needed: str


@dataclass(frozen=True)
class ConductivityDecompositionDiagnostic:
    direct_capacity_gap_mS_cm: float
    corrector_target_mS_cm: float
    corrector_residual_mS_cm: float
    direct_capacity_failure: bool
    corrector_too_strong_failure: bool
    corrector_too_weak_failure: bool


@dataclass(frozen=True)
class MolecularPropertyDbAuditResult:
    rows: tuple[MolecularPropertyDbRowResult, ...]
    labeled_rows: int
    evaluated_rows: int
    failed_rows: int
    mae_mS_cm: float
    rmse_mS_cm: float
    bias_mS_cm: float
    pearson_r: float
    maximum_abs_residual_mS_cm: float
    maximum_mass_balance_residual: float
    maximum_row_sum_residual: float
    maximum_stationary_residual: float
    maximum_detailed_balance_residual: float
    maximum_event_reversal_residual: float
    zero_charge_sigma_mS_cm: float
    higher_viscosity_lowers_dilute_conductivity: bool
    higher_packing_lowers_local_mobility: bool


class MolecularPropertyDbPrimitiveEvaluator:
    def __init__(
        self,
        cases: tuple[MolecularPropertyDbCase, ...],
        options: MolecularPropertyDbAuditOptions,
        fit_options: PrimitiveFitOptions,
    ) -> None:
        if not cases:
            raise ValueError("molecular property-DB evaluator requires cases")
        self._cases = cases
        self._options = options
        self._fit_options = fit_options
        self._has_measured_consumed_parameter_fields = False
        self._consumed_parameter_fields: tuple[str, ...] = tuple()

    def evaluate(
        self,
        primitive_parameters: ConductivityPrimitiveParameterSet,
    ) -> PrimitiveFitDatasetEvaluation:
        audit_result = audit_molecular_property_db_cases(
            self._cases,
            primitive_parameters,
            self._options,
        )
        if not self._has_measured_consumed_parameter_fields:
            self._consumed_parameter_fields = _measured_consumed_parameter_fields(
                self._cases,
                primitive_parameters,
                self._options,
                audit_result,
            )
            self._has_measured_consumed_parameter_fields = True
        return PrimitiveFitDatasetEvaluation(
            empirical_sigmas_mS_cm=tuple(
                row.empirical_sigma_mS_cm for row in audit_result.rows
            ),
            predicted_sigmas_mS_cm=tuple(
                row.predicted_sigma_mS_cm for row in audit_result.rows
            ),
            direct_sigmas_mS_cm=tuple(
                row.direct_sigma_mS_cm for row in audit_result.rows
            ),
            corrector_sigmas_mS_cm=tuple(
                row.corrector_sigma_mS_cm for row in audit_result.rows
            ),
            direct_capacity_gaps_mS_cm=tuple(
                row.direct_capacity_gap_mS_cm for row in audit_result.rows
            ),
            corrector_targets_mS_cm=tuple(
                row.corrector_target_mS_cm for row in audit_result.rows
            ),
            corrector_residuals_mS_cm=tuple(
                row.corrector_residual_mS_cm for row in audit_result.rows
            ),
            direct_capacity_failure_count=sum(
                1 for row in audit_result.rows if row.direct_capacity_failure
            ),
            corrector_too_strong_failure_count=sum(
                1 for row in audit_result.rows
                if row.corrector_too_strong_failure
            ),
            corrector_too_weak_failure_count=sum(
                1 for row in audit_result.rows
                if row.corrector_too_weak_failure
            ),
            empirical_sigma_spreads_mS_cm=tuple(
                row.empirical_sigma_spread_mS_cm for row in audit_result.rows
            ),
            cluster_activation_penalty=_cluster_activation_penalty(
                self._cases,
                primitive_parameters,
                self._options,
                self._fit_options,
                audit_result,
            ),
            failed_rows=audit_result.failed_rows,
            maximum_mass_balance_residual=(
                audit_result.maximum_mass_balance_residual
            ),
            maximum_row_sum_residual=audit_result.maximum_row_sum_residual,
            maximum_stationary_residual=(
                audit_result.maximum_stationary_residual
            ),
            maximum_detailed_balance_residual=(
                audit_result.maximum_detailed_balance_residual
            ),
            maximum_event_reversal_residual=(
                audit_result.maximum_event_reversal_residual
            ),
            zero_charge_sigma_mS_cm=audit_result.zero_charge_sigma_mS_cm,
            higher_viscosity_lowers_dilute_conductivity=(
                audit_result.higher_viscosity_lowers_dilute_conductivity
            ),
            higher_packing_lowers_local_mobility=(
                audit_result.higher_packing_lowers_local_mobility
            ),
            consumed_parameter_fields=self._consumed_parameter_fields,
        )


def _cluster_activation_penalty(
    cases: tuple[MolecularPropertyDbCase, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
    fit_options: PrimitiveFitOptions,
    audit_result: MolecularPropertyDbAuditResult,
) -> float:
    if fit_options.cluster_activation_loss_weight == 0.0:
        return 0.0
    case_by_row_id = {molecular_case.row_id: molecular_case for molecular_case in cases}
    selected_rows = tuple(
        sorted(
            audit_result.rows,
            key=lambda row_result: abs(row_result.residual_mS_cm),
            reverse=True,
        )[:fit_options.residual_tail_count]
    )
    penalty_terms: list[float] = []
    residual_threshold_mS_cm = fit_options.cluster_activation_residual_threshold_mS_cm
    minimum_charged_cluster_fraction = (
        fit_options.cluster_activation_min_charged_cluster_fraction
    )
    minimum_charged_cluster_net_sigma_mS_cm = (
        fit_options.cluster_activation_min_charged_cluster_net_sigma_mS_cm
    )
    for row_result in selected_rows:
        if abs(row_result.residual_mS_cm) <= residual_threshold_mS_cm:
            continue
        if row_result.row_id not in case_by_row_id:
            raise ValueError(f"missing molecular case for row {row_result.row_id}")
        if (
            row_result.charged_cluster_fraction
            >= minimum_charged_cluster_fraction
            and abs(row_result.charged_cluster_net_sigma_mS_cm)
            >= minimum_charged_cluster_net_sigma_mS_cm
        ):
            continue
        sensitivity_diagnostics = cluster_sensitivity_diagnostics_for_row(
            case_by_row_id[row_result.row_id],
            primitive_parameters,
            options,
            row_result,
        )
        charged_cluster_sensitivity_weight = math.fsum(
            abs(sensitivity_diagnostic.sensitivity_mS_cm_per_logK)
            for sensitivity_diagnostic in sensitivity_diagnostics
            if (
                sensitivity_diagnostic.net_charge_number != 0
                and sensitivity_diagnostic.direction_needed == "increase_logK"
            )
        )
        if charged_cluster_sensitivity_weight <= 0.0:
            continue
        fraction_deficit = max(
            0.0,
            (
                minimum_charged_cluster_fraction
                - row_result.charged_cluster_fraction
            )
            / minimum_charged_cluster_fraction,
        )
        sigma_deficit = max(
            0.0,
            (
                minimum_charged_cluster_net_sigma_mS_cm
                - abs(row_result.charged_cluster_net_sigma_mS_cm)
            )
            / minimum_charged_cluster_net_sigma_mS_cm,
        )
        penalty_terms.append(
            charged_cluster_sensitivity_weight
            * (
                fraction_deficit * fraction_deficit
                + sigma_deficit * sigma_deficit
            )
        )
    return float(math.fsum(penalty_terms))


REFERENCE_LOGK_OFFSET_PARAMETER_VALUE = 0.0
LOGK_OFFSET_PARAMETER_NAMES = frozenset(
    (
        "pair_logK_offset",
        "solvent_separated_pair_logK_offset",
        "contact_pair_logK_offset",
        "positive_charged_triplet_logK_offset",
        "negative_charged_triplet_logK_offset",
        "neutral_cluster_logK_offset",
        "higher_charged_cluster_logK_offset",
        "cluster_order_logK_slope",
        "cluster_charge_magnitude_logK_slope",
    )
)
REFERENCE_NEAR_NEUTRAL_MOBILITY_EXPONENT_PARAMETER_VALUE = 1.0e-3
NEAR_NEUTRAL_MOBILITY_EXPONENT_PARAMETER_NAMES = frozenset(
    (
        "negative_ion_intrinsic_dielectric_drag_mobility_exponent",
        "negative_ion_shape_delocalization_mobility_exponent",
    )
)


REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS = (
    conductivity_primitive_parameters_from_mapping(
        {
            field_name: (
                REFERENCE_LOGK_OFFSET_PARAMETER_VALUE
                if field_name in LOGK_OFFSET_PARAMETER_NAMES
                else REFERENCE_NEAR_NEUTRAL_MOBILITY_EXPONENT_PARAMETER_VALUE
                if field_name in NEAR_NEUTRAL_MOBILITY_EXPONENT_PARAMETER_NAMES
                else 1.0
            )
            for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        }
    )
)

validate_conductivity_primitive_parameters(REFERENCE_CONDUCTIVITY_PRIMITIVE_PARAMETERS)


def default_molecular_property_db_audit_options() -> MolecularPropertyDbAuditOptions:
    config_mapping = _load_physics_config_section(PROPERTY_DB_AUDIT_CONFIG_KEY)
    return MolecularPropertyDbAuditOptions(
        liquid_occupied_volume_fraction=_required_positive_config_float(
            config_mapping,
            "liquid_occupied_volume_fraction",
        ),
        reference_ion_viscosity_cP=_required_positive_config_float(
            config_mapping,
            "reference_ion_viscosity_cP",
        ),
        ion_reference_density_g_ml=_required_positive_config_float(
            config_mapping,
            "ion_reference_density_g_ml",
        ),
        ion_reference_dielectric_base=_required_positive_config_float(
            config_mapping,
            "ion_reference_dielectric_base",
        ),
        max_cluster_ion_count=_required_positive_config_int(
            config_mapping,
            "max_cluster_ion_count",
        ),
        max_packing_fraction=_required_positive_config_float(
            config_mapping,
            "max_packing_fraction",
        ),
        free_volume_exponent=_required_positive_config_float(
            config_mapping,
            "free_volume_exponent",
        ),
        translation_jump_length_multiplier=_required_positive_config_float(
            config_mapping,
            "translation_jump_length_multiplier",
        ),
        viscosity_monotonicity_scale=_required_positive_config_float(
            config_mapping,
            "viscosity_monotonicity_scale",
        ),
        packing_monotonicity_volume_scale=_required_positive_config_float(
            config_mapping,
            "packing_monotonicity_volume_scale",
        ),
        parameter_consumption_perturbation_scales=_required_positive_float_tuple(
            config_mapping,
            "parameter_consumption_perturbation_scales",
            "molecular_property_db_audit.parameter_consumption_perturbation_scales",
        ),
        cluster_sensitivity_step_over_RT=_required_positive_config_float(
            config_mapping,
            "cluster_sensitivity_step_over_RT",
        ),
        audit_worst_row_count=_required_positive_config_int(
            config_mapping,
            "audit_worst_row_count",
        ),
        validation_excluded_source_row_ids=_required_nonnegative_int_tuple(
            config_mapping,
            "validation_excluded_source_row_ids",
            "molecular_property_db_audit.validation_excluded_source_row_ids",
        ),
        neutral_hbond_donor_count=_required_nonnegative_config_int(
            config_mapping,
            "neutral_hbond_donor_count",
        ),
        ion_hbond_donor_count=_required_nonnegative_config_int(
            config_mapping,
            "ion_hbond_donor_count",
        ),
        ion_hbond_acceptor_count=_required_nonnegative_config_int(
            config_mapping,
            "ion_hbond_acceptor_count",
        ),
    )


def configured_conductivity_primitive_parameters() -> ConductivityPrimitiveParameterSet:
    config_mapping = _load_physics_config_section(PRIMITIVE_PARAMETER_CONFIG_KEY)
    _validate_primitive_parameter_config_keys(config_mapping)
    primitive_parameters = conductivity_primitive_parameters_from_mapping(
        {
            field_name: _required_primitive_parameter_config_float(
                config_mapping,
                field_name,
            )
            for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        }
    )
    validate_conductivity_primitive_parameters(primitive_parameters)
    return primitive_parameters


def _validate_primitive_parameter_config_keys(config_mapping: dict) -> None:
    permitted_config_keys = set(CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES)
    permitted_config_keys.update(PRIMITIVE_PARAMETER_CONFIG_METADATA_KEYS)
    unknown_config_keys = tuple(
        sorted(
            config_key for config_key in config_mapping
            if config_key not in permitted_config_keys
        )
    )
    if unknown_config_keys:
        raise ValueError(
            "unknown molecular conductivity primitive parameter config keys: "
            f"{unknown_config_keys}"
        )


def default_molecular_primitive_fit_configuration() -> tuple[
    PrimitiveFitOptions,
    tuple[PrimitiveParameterTransform, ...],
]:
    fit_config_mapping = _load_optimization_config_section(
        PRIMITIVE_PARAMETER_FIT_CONFIG_KEY
    )
    coordinate_bounds_mapping = _required_mapping(
        fit_config_mapping,
        "coordinate_bounds",
        "molecular_primitive_parameter_fit.coordinate_bounds",
    )
    unknown_coordinate_bound_names = tuple(
        sorted(
            parameter_name for parameter_name in coordinate_bounds_mapping
            if parameter_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        )
    )
    if unknown_coordinate_bound_names:
        raise ValueError(
            "unknown molecular primitive fit coordinate bounds: "
            f"{unknown_coordinate_bound_names}"
        )
    missing_coordinate_bound_names = tuple(
        parameter_name
        for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        if parameter_name not in coordinate_bounds_mapping
    )
    if missing_coordinate_bound_names:
        raise ValueError(
            "molecular primitive fit coordinate_bounds missing parameters "
            f"{missing_coordinate_bound_names}"
        )
    fit_options = PrimitiveFitOptions(
        huber_delta_mS_cm=_required_positive_config_float(
            fit_config_mapping,
            "huber_delta_mS_cm",
        ),
        empirical_sigma_floor_mS_cm=_required_positive_config_float(
            fit_config_mapping,
            "empirical_sigma_floor_mS_cm",
        ),
        coordinate_regularization_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "coordinate_regularization_weight",
        ),
        residual_tail_loss_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "residual_tail_loss_weight",
        ),
        residual_tail_count=_required_positive_config_int(
            fit_config_mapping,
            "residual_tail_count",
        ),
        cluster_activation_loss_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "cluster_activation_loss_weight",
        ),
        cluster_activation_residual_threshold_mS_cm=(
            _required_positive_config_float(
                fit_config_mapping,
                "cluster_activation_residual_threshold_mS_cm",
            )
        ),
        cluster_activation_min_charged_cluster_fraction=(
            _required_positive_config_float(
                fit_config_mapping,
                "cluster_activation_min_charged_cluster_fraction",
            )
        ),
        cluster_activation_min_charged_cluster_net_sigma_mS_cm=(
            _required_positive_config_float(
                fit_config_mapping,
                "cluster_activation_min_charged_cluster_net_sigma_mS_cm",
            )
        ),
        direct_capacity_loss_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "direct_capacity_loss_weight",
        ),
        corrector_loss_weight=_required_nonnegative_config_float(
            fit_config_mapping,
            "corrector_loss_weight",
        ),
        role_direct_scaling_regularization_weight=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "role_direct_scaling_regularization_weight",
            )
        ),
        role_direct_scaling_lower_bound=_required_positive_config_float(
            fit_config_mapping,
            "role_direct_scaling_lower_bound",
        ),
        role_direct_scaling_upper_bound=_required_positive_config_float(
            fit_config_mapping,
            "role_direct_scaling_upper_bound",
        ),
        latin_hypercube_samples_per_parameter=_required_positive_config_float(
            fit_config_mapping,
            "latin_hypercube_samples_per_parameter",
        ),
        coordinate_search_rounds=_required_nonnegative_config_int(
            fit_config_mapping,
            "coordinate_search_rounds",
        ),
        initial_coordinate_step=_required_positive_config_float(
            fit_config_mapping,
            "initial_coordinate_step",
        ),
        coordinate_step_shrinkage=_required_positive_config_float(
            fit_config_mapping,
            "coordinate_step_shrinkage",
        ),
        minimum_coordinate_step=_required_positive_config_float(
            fit_config_mapping,
            "minimum_coordinate_step",
        ),
        powell_max_iterations_per_parameter=_required_nonnegative_config_float(
            fit_config_mapping,
            "powell_max_iterations_per_parameter",
        ),
        powell_max_function_evaluations_per_parameter=_required_nonnegative_config_float(
            fit_config_mapping,
            "powell_max_function_evaluations_per_parameter",
        ),
        decomposed_block_powell_max_iterations_per_parameter=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "decomposed_block_powell_max_iterations_per_parameter",
            )
        ),
        decomposed_block_powell_max_function_evaluations_per_parameter=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "decomposed_block_powell_max_function_evaluations_per_parameter",
            )
        ),
        decomposed_block_cluster_activation_loss_weight=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "decomposed_block_cluster_activation_loss_weight",
            )
        ),
        powell_xtol_coordinate=_required_positive_config_float(
            fit_config_mapping,
            "powell_xtol_coordinate",
        ),
        powell_ftol_objective=_required_positive_config_float(
            fit_config_mapping,
            "powell_ftol_objective",
        ),
        random_seed=_required_nonnegative_config_int(
            fit_config_mapping,
            "random_seed",
        ),
        maximum_failed_rows=_required_nonnegative_config_int(
            fit_config_mapping,
            "maximum_failed_rows",
        ),
        maximum_mass_balance_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_mass_balance_residual",
        ),
        maximum_row_sum_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_row_sum_residual",
        ),
        maximum_stationary_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_stationary_residual",
        ),
        maximum_detailed_balance_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_detailed_balance_residual",
        ),
        maximum_event_reversal_residual=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_event_reversal_residual",
        ),
        maximum_zero_charge_sigma_mS_cm=_required_nonnegative_config_float(
            fit_config_mapping,
            "maximum_zero_charge_sigma_mS_cm",
        ),
        descriptor_matrix_high_correlation_threshold=(
            _required_positive_config_float(
                fit_config_mapping,
                "descriptor_matrix_high_correlation_threshold",
            )
        ),
        descriptor_matrix_condition_number_warn_threshold=(
            _required_positive_config_float(
                fit_config_mapping,
                "descriptor_matrix_condition_number_warn_threshold",
            )
        ),
        descriptor_matrix_reported_correlation_pair_count=(
            _required_nonnegative_config_int(
                fit_config_mapping,
                "descriptor_matrix_reported_correlation_pair_count",
            )
        ),
        prediction_sensitivity_coordinate_step=_required_positive_config_float(
            fit_config_mapping,
            "prediction_sensitivity_coordinate_step",
        ),
        prediction_sensitivity_min_column_norm_mS_cm_per_coordinate=(
            _required_positive_config_float(
                fit_config_mapping,
                "prediction_sensitivity_min_column_norm_mS_cm_per_coordinate",
            )
        ),
        prediction_sensitivity_relative_singular_value_threshold=(
            _required_positive_config_float(
                fit_config_mapping,
                "prediction_sensitivity_relative_singular_value_threshold",
            )
        ),
        prediction_sensitivity_high_correlation_threshold=(
            _required_positive_config_float(
                fit_config_mapping,
                "prediction_sensitivity_high_correlation_threshold",
            )
        ),
        prediction_sensitivity_reported_correlation_pair_count=(
            _required_nonnegative_config_int(
                fit_config_mapping,
                "prediction_sensitivity_reported_correlation_pair_count",
            )
        ),
        candidate_output_path=_required_config_string(
            fit_config_mapping,
            "candidate_output_path",
        ),
        promotion_maximum_mae_mS_cm=_required_positive_config_float(
            fit_config_mapping,
            "promotion_maximum_mae_mS_cm",
        ),
        promotion_maximum_abs_bias_mS_cm=_required_nonnegative_config_float(
            fit_config_mapping,
            "promotion_maximum_abs_bias_mS_cm",
        ),
        promotion_maximum_worst_abs_residual_mS_cm=(
            _required_nonnegative_config_float(
                fit_config_mapping,
                "promotion_maximum_worst_abs_residual_mS_cm",
            )
        ),
        promotion_require_mae_improvement=_required_config_bool(
            fit_config_mapping,
            "promotion_require_mae_improvement",
        ),
    )
    coordinate_bounds: list[PrimitiveParameterTransform] = []
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        coordinate_bounds.append(
            _primitive_fit_coordinate_bound_from_config(
                coordinate_bounds_mapping,
                parameter_name,
            )
        )
    return fit_options, tuple(coordinate_bounds)


def build_molecular_property_db_case_selection(
    property_db_rows: tuple[dict, ...],
    registry_source: MolecularPropertyDbRegistrySource,
    options: MolecularPropertyDbAuditOptions,
) -> MolecularPropertyDbCaseSelection:
    cases: list[MolecularPropertyDbCase] = []
    excluded_rows: list[MolecularPropertyDbExcludedRow] = []
    validation_excluded_source_row_id_set = set(
        options.validation_excluded_source_row_ids
    )
    for row_index, row_mapping in enumerate(property_db_rows):
        empirical_sigma_mS_cm = _property_db_conductivity_mS_cm(row_mapping)
        recipe_mapping = _required_mapping(row_mapping, "recipe", "row.recipe")
        solvent_loadings = _positive_float_mapping(
            _required_mapping(recipe_mapping, "solvents", "recipe.solvents"),
            "recipe.solvents",
        )
        salt_loadings_M = _positive_float_mapping(
            _required_mapping(recipe_mapping, "salts", "recipe.salts"),
            "recipe.salts",
        )
        additive_loadings_with_zeros = _nonnegative_float_mapping(
            _required_mapping(recipe_mapping, "additives", "recipe.additives"),
            "recipe.additives",
        )
        additive_loadings = {
            species_name: loading_value
            for species_name, loading_value in additive_loadings_with_zeros.items()
            if loading_value > 0.0
        }
        if row_index in validation_excluded_source_row_id_set:
            excluded_rows.append(
                MolecularPropertyDbExcludedRow(
                    row_id=row_index,
                    empirical_sigma_mS_cm=empirical_sigma_mS_cm,
                    solvent_loadings=solvent_loadings,
                    salt_loadings_M=salt_loadings_M,
                    additive_loadings=additive_loadings,
                )
            )
            continue
        mixture_density_g_ml = _mixture_density_g_ml(
            row_mapping,
            solvent_loadings,
            salt_loadings_M,
            additive_loadings,
            registry_source,
        )
        mixture_viscosity_cP = _mixture_viscosity_cP(
            solvent_loadings,
            salt_loadings_M,
            additive_loadings,
            mixture_density_g_ml,
            registry_source,
        )
        mixture_dielectric_constant = _mixture_dielectric_constant(
            solvent_loadings,
            salt_loadings_M,
            additive_loadings,
            mixture_density_g_ml,
            registry_source,
        )
        row_state = _MolecularPropertyDbRowState(
            solvent_loadings=solvent_loadings,
            salt_loadings_M=salt_loadings_M,
            additive_loadings=additive_loadings,
            mixture_density_g_ml=mixture_density_g_ml,
            mixture_viscosity_cP=mixture_viscosity_cP,
            mixture_dielectric_constant=mixture_dielectric_constant,
        )
        molecular_recipe, species_inputs = _molecular_recipe_and_species_inputs(
            row_state,
            registry_source,
            options,
        )
        cases.append(
            MolecularPropertyDbCase(
                row_id=row_index,
                source_row_ids=(row_index,),
                source_formulation_key=_source_formulation_key(
                    solvent_loadings,
                    salt_loadings_M,
                    additive_loadings,
                    T_REF_K,
                ),
                source_solvent_loadings=solvent_loadings,
                source_salt_loadings_M=salt_loadings_M,
                source_additive_loadings=additive_loadings,
                recipe=molecular_recipe,
                species_inputs=species_inputs,
                empirical_sigma_mS_cm=empirical_sigma_mS_cm,
                empirical_sigma_spread_mS_cm=0.0,
            )
        )
    observed_excluded_row_ids = tuple(
        sorted(excluded_row.row_id for excluded_row in excluded_rows)
    )
    if observed_excluded_row_ids != options.validation_excluded_source_row_ids:
        raise ValueError(
            "validation_excluded_source_row_ids did not match property DB rows: "
            f"configured={options.validation_excluded_source_row_ids}, "
            f"observed={observed_excluded_row_ids}"
        )
    grouped_cases, formulation_groups = _group_duplicate_formulation_cases(
        tuple(cases),
        registry_source,
        options,
    )
    return MolecularPropertyDbCaseSelection(
        source_labeled_rows=len(property_db_rows),
        cases=grouped_cases,
        excluded_rows=tuple(excluded_rows),
        formulation_groups=formulation_groups,
    )


def _group_duplicate_formulation_cases(
    cases: tuple[MolecularPropertyDbCase, ...],
    registry_source: MolecularPropertyDbRegistrySource,
    options: MolecularPropertyDbAuditOptions,
) -> tuple[
    tuple[MolecularPropertyDbCase, ...],
    tuple[MolecularPropertyDbFormulationGroup, ...],
]:
    cases_by_formulation_key: dict[tuple, list[MolecularPropertyDbCase]] = {}
    for molecular_case in cases:
        formulation_key = _molecular_case_formulation_key(molecular_case)
        if formulation_key not in cases_by_formulation_key:
            cases_by_formulation_key[formulation_key] = []
        cases_by_formulation_key[formulation_key].append(molecular_case)
    grouped_cases: list[MolecularPropertyDbCase] = []
    formulation_groups: list[MolecularPropertyDbFormulationGroup] = []
    for formulation_cases in cases_by_formulation_key.values():
        grouped_case, formulation_group = _grouped_formulation_case(
            tuple(formulation_cases),
            registry_source,
            options,
        )
        grouped_cases.append(grouped_case)
        formulation_groups.append(formulation_group)
    return (
        tuple(sorted(grouped_cases, key=lambda molecular_case: molecular_case.row_id)),
        tuple(
            sorted(
                formulation_groups,
                key=lambda formulation_group: formulation_group.representative_row_id,
            )
        ),
    )


def _grouped_formulation_case(
    formulation_cases: tuple[MolecularPropertyDbCase, ...],
    registry_source: MolecularPropertyDbRegistrySource,
    options: MolecularPropertyDbAuditOptions,
) -> tuple[MolecularPropertyDbCase, MolecularPropertyDbFormulationGroup]:
    if not formulation_cases:
        raise ValueError("formulation group must contain at least one case")
    sorted_cases = tuple(
        sorted(formulation_cases, key=lambda molecular_case: molecular_case.row_id)
    )
    representative_case = sorted_cases[0]
    empirical_sigmas = tuple(
        molecular_case.empirical_sigma_mS_cm for molecular_case in sorted_cases
    )
    target_sigma_mS_cm = _median_float(empirical_sigmas, "empirical_sigmas_mS_cm")
    empirical_sigma_spread_mS_cm = max(empirical_sigmas) - min(empirical_sigmas)
    source_row_ids = tuple(molecular_case.row_id for molecular_case in sorted_cases)
    grouped_recipe, grouped_species_inputs = (
        _median_formulation_recipe_and_species_inputs(
            sorted_cases,
            registry_source,
            options,
        )
    )
    grouped_case = MolecularPropertyDbCase(
        row_id=representative_case.row_id,
        source_row_ids=source_row_ids,
        source_formulation_key=representative_case.source_formulation_key,
        source_solvent_loadings=representative_case.source_solvent_loadings,
        source_salt_loadings_M=representative_case.source_salt_loadings_M,
        source_additive_loadings=representative_case.source_additive_loadings,
        recipe=grouped_recipe,
        species_inputs=grouped_species_inputs,
        empirical_sigma_mS_cm=target_sigma_mS_cm,
        empirical_sigma_spread_mS_cm=empirical_sigma_spread_mS_cm,
    )
    formulation_group = MolecularPropertyDbFormulationGroup(
        representative_row_id=representative_case.row_id,
        source_row_ids=source_row_ids,
        empirical_sigmas_mS_cm=empirical_sigmas,
        target_sigma_mS_cm=target_sigma_mS_cm,
        empirical_sigma_spread_mS_cm=empirical_sigma_spread_mS_cm,
        solvent_loadings=representative_case.source_solvent_loadings,
        salt_loadings_M=representative_case.source_salt_loadings_M,
        additive_loadings=representative_case.source_additive_loadings,
    )
    return grouped_case, formulation_group


def _median_formulation_recipe_and_species_inputs(
    formulation_cases: tuple[MolecularPropertyDbCase, ...],
    registry_source: MolecularPropertyDbRegistrySource,
    options: MolecularPropertyDbAuditOptions,
) -> tuple[MolecularElectrolyteRecipe, Mapping[str, MolecularSpeciesInput]]:
    representative_case = formulation_cases[0]
    grouped_row_state = _MolecularPropertyDbRowState(
        solvent_loadings=representative_case.source_solvent_loadings,
        salt_loadings_M=representative_case.source_salt_loadings_M,
        additive_loadings=representative_case.source_additive_loadings,
        mixture_density_g_ml=_median_float(
            tuple(
                molecular_case.recipe.mixture_properties.density_g_ml
                for molecular_case in formulation_cases
            ),
            "formulation_group.density_g_ml",
        ),
        mixture_viscosity_cP=_median_float(
            tuple(
                molecular_case.recipe.mixture_properties.viscosity_cP
                for molecular_case in formulation_cases
            ),
            "formulation_group.viscosity_cP",
        ),
        mixture_dielectric_constant=_median_float(
            tuple(
                molecular_case.recipe.mixture_properties.dielectric_constant
                for molecular_case in formulation_cases
            ),
            "formulation_group.dielectric_constant",
        ),
    )
    return _molecular_recipe_and_species_inputs(
        grouped_row_state,
        registry_source,
        options,
    )


def _median_float(
    values: tuple[float, ...],
    context: str,
) -> float:
    if not values:
        raise ValueError(f"{context} must contain at least one value")
    sorted_values = tuple(sorted(_finite_float(value, context) for value in values))
    middle_index = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[middle_index])
    return float(
        0.5 * (sorted_values[middle_index - 1] + sorted_values[middle_index])
    )


def _molecular_case_formulation_key(
    molecular_case: MolecularPropertyDbCase,
) -> tuple:
    return molecular_case.source_formulation_key


def _source_formulation_key(
    solvent_loadings: Mapping[str, float],
    salt_loadings_M: Mapping[str, float],
    additive_loadings: Mapping[str, float],
    temperature_K: float,
) -> tuple:
    return (
        _sorted_loading_items(solvent_loadings),
        _sorted_loading_items(salt_loadings_M),
        _sorted_loading_items(additive_loadings),
        float(temperature_K),
    )


def _sorted_loading_items(
    loading_mapping: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (species_name, float(loading_value))
            for species_name, loading_value in loading_mapping.items()
        )
    )


def audit_molecular_property_db_cases(
    cases: tuple[MolecularPropertyDbCase, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
) -> MolecularPropertyDbAuditResult:
    if not cases:
        raise ValueError("molecular property-DB audit requires cases")
    validate_conductivity_primitive_parameters(primitive_parameters)
    row_results: list[MolecularPropertyDbRowResult] = []
    for molecular_case in cases:
        row_results.append(
            _evaluate_molecular_property_db_case(
                molecular_case,
                primitive_parameters,
                options,
            )
        )
    successful_rows = tuple(row for row in row_results if not row.failed)
    empirical_sigmas = tuple(row.empirical_sigma_mS_cm for row in successful_rows)
    predicted_sigmas = tuple(row.predicted_sigma_mS_cm for row in successful_rows)
    residuals = tuple(row.residual_mS_cm for row in successful_rows)
    failed_rows = len(row_results) - len(successful_rows)
    zero_charge_sigma_mS_cm = _zero_charge_sigma_mS_cm(
        cases[0],
        primitive_parameters,
        options,
    )
    viscosity_monotonicity = _higher_viscosity_lowers_dilute_conductivity(
        cases[0],
        primitive_parameters,
        options,
    )
    packing_monotonicity = _higher_packing_lowers_local_mobility(
        cases[0],
        primitive_parameters,
        options,
    )
    return MolecularPropertyDbAuditResult(
        rows=tuple(row_results),
        labeled_rows=len(row_results),
        evaluated_rows=len(successful_rows),
        failed_rows=failed_rows,
        mae_mS_cm=_mean_absolute_residual_or_zero(residuals),
        rmse_mS_cm=_root_mean_square_residual_or_zero(residuals),
        bias_mS_cm=_mean_residual_or_zero(residuals),
        pearson_r=_pearson_or_zero(empirical_sigmas, predicted_sigmas),
        maximum_abs_residual_mS_cm=_maximum_abs_residual_or_zero(residuals),
        maximum_mass_balance_residual=_maximum_successful_value(
            successful_rows,
            "mass_balance_residual_mol_m3",
        ),
        maximum_row_sum_residual=_maximum_successful_value(
            successful_rows,
            "row_sum_residual",
        ),
        maximum_stationary_residual=_maximum_successful_value(
            successful_rows,
            "stationary_residual_mol_m3_s",
        ),
        maximum_detailed_balance_residual=_maximum_successful_value(
            successful_rows,
            "detailed_balance_residual_mol_m3_s",
        ),
        maximum_event_reversal_residual=_maximum_successful_value(
            successful_rows,
            "event_reversal_residual_mol_m3_s",
        ),
        zero_charge_sigma_mS_cm=zero_charge_sigma_mS_cm,
        higher_viscosity_lowers_dilute_conductivity=viscosity_monotonicity,
        higher_packing_lowers_local_mobility=packing_monotonicity,
    )


def validate_molecular_property_db_audit_result(
    audit_result: MolecularPropertyDbAuditResult,
    fit_options: PrimitiveFitOptions,
) -> None:
    if audit_result.evaluated_rows != audit_result.labeled_rows:
        raise ValueError(
            "molecular property-DB audit did not evaluate every labeled row: "
            f"{audit_result.evaluated_rows}/{audit_result.labeled_rows}"
        )
    if audit_result.failed_rows > fit_options.maximum_failed_rows:
        raise ValueError(
            "molecular property-DB audit failed row count "
            f"{audit_result.failed_rows} exceeds {fit_options.maximum_failed_rows}"
        )
    _raise_if_above_limit(
        audit_result.maximum_mass_balance_residual,
        fit_options.maximum_mass_balance_residual,
        "maximum_mass_balance_residual",
    )
    _raise_if_above_limit(
        audit_result.maximum_row_sum_residual,
        fit_options.maximum_row_sum_residual,
        "maximum_row_sum_residual",
    )
    _raise_if_above_limit(
        audit_result.maximum_stationary_residual,
        fit_options.maximum_stationary_residual,
        "maximum_stationary_residual",
    )
    _raise_if_above_limit(
        audit_result.maximum_detailed_balance_residual,
        fit_options.maximum_detailed_balance_residual,
        "maximum_detailed_balance_residual",
    )
    _raise_if_above_limit(
        audit_result.maximum_event_reversal_residual,
        fit_options.maximum_event_reversal_residual,
        "maximum_event_reversal_residual",
    )
    _raise_if_above_limit(
        abs(audit_result.zero_charge_sigma_mS_cm),
        fit_options.maximum_zero_charge_sigma_mS_cm,
        "zero_charge_sigma_mS_cm",
    )
    if not audit_result.higher_viscosity_lowers_dilute_conductivity:
        raise ValueError("higher-viscosity molecular invariant failed")
    if not audit_result.higher_packing_lowers_local_mobility:
        raise ValueError("higher-packing molecular invariant failed")


def _evaluate_molecular_property_db_case(
    molecular_case: MolecularPropertyDbCase,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
) -> MolecularPropertyDbRowResult:
    molecular_result = _compute_case_result(
        molecular_case,
        primitive_parameters,
        options,
    )
    validation = molecular_result.markov_additive_result.validation
    predicted_sigma_mS_cm = molecular_result.sigma_mS_cm
    direct_sigma_mS_cm = molecular_result.markov_additive_result.direct_sigma_mS_cm
    corrector_sigma_mS_cm = (
        molecular_result.markov_additive_result.corrector_sigma_mS_cm
    )
    decomposition_diagnostic = conductivity_decomposition_diagnostic(
        direct_sigma_mS_cm,
        corrector_sigma_mS_cm,
        predicted_sigma_mS_cm,
        molecular_case.empirical_sigma_mS_cm,
    )
    cluster_fraction_rollup = _cluster_fraction_rollup(molecular_result)
    cluster_mobility_rollup = _cluster_transport_mobility_rollup(
        molecular_result,
    )
    charged_cluster_sigma_rollup = _charged_cluster_event_sigma_rollup(
        molecular_result,
    )
    transport_role_sigma_rollup = _transport_role_event_sigma_rollup(
        molecular_result,
    )
    return MolecularPropertyDbRowResult(
        row_id=molecular_case.row_id,
        source_row_ids=molecular_case.source_row_ids,
        empirical_sigma_mS_cm=molecular_case.empirical_sigma_mS_cm,
        empirical_sigma_spread_mS_cm=(
            molecular_case.empirical_sigma_spread_mS_cm
        ),
        predicted_sigma_mS_cm=predicted_sigma_mS_cm,
        residual_mS_cm=predicted_sigma_mS_cm - molecular_case.empirical_sigma_mS_cm,
        failed=False,
        failure_reason="",
        direct_sigma_mS_cm=direct_sigma_mS_cm,
        corrector_sigma_mS_cm=corrector_sigma_mS_cm,
        direct_capacity_gap_mS_cm=(
            decomposition_diagnostic.direct_capacity_gap_mS_cm
        ),
        corrector_target_mS_cm=decomposition_diagnostic.corrector_target_mS_cm,
        corrector_residual_mS_cm=(
            decomposition_diagnostic.corrector_residual_mS_cm
        ),
        direct_capacity_failure=decomposition_diagnostic.direct_capacity_failure,
        corrector_too_strong_failure=(
            decomposition_diagnostic.corrector_too_strong_failure
        ),
        corrector_too_weak_failure=(
            decomposition_diagnostic.corrector_too_weak_failure
        ),
        direct_sigma_by_transport_role_mS_cm=(
            transport_role_sigma_rollup["direct_sigma_by_transport_role_mS_cm"]
        ),
        corrector_sigma_by_transport_role_mS_cm=(
            transport_role_sigma_rollup["corrector_sigma_by_transport_role_mS_cm"]
        ),
        net_sigma_by_transport_role_mS_cm=(
            transport_role_sigma_rollup["net_sigma_by_transport_role_mS_cm"]
        ),
        charge_weighted_transport_concentration_mol_m3=(
            _charge_weighted_transport_concentration_mol_m3(molecular_result)
        ),
        mass_balance_residual_mol_m3=molecular_result.mass_balance_residual_mol_m3,
        row_sum_residual=validation.row_sum_residual,
        stationary_residual_mol_m3_s=validation.stationary_residual_mol_m3_s,
        detailed_balance_residual_mol_m3_s=(
            validation.detailed_balance_residual_mol_m3_s
        ),
        event_reversal_residual_mol_m3_s=(
            molecular_result.markov_additive_result.event_reversal_residual_mol_m3_s
        ),
        free_ion_fraction=cluster_fraction_rollup["free_ion_fraction"],
        charged_cluster_fraction=cluster_fraction_rollup[
            "charged_cluster_fraction"
        ],
        neutral_cluster_fraction=cluster_fraction_rollup[
            "neutral_cluster_fraction"
        ],
        cluster_transport_mobility_density_mol_m_s=cluster_mobility_rollup[
            "cluster_transport_mobility_density_mol_m_s"
        ],
        charged_cluster_transport_mobility_density_mol_m_s=(
            cluster_mobility_rollup[
                "charged_cluster_transport_mobility_density_mol_m_s"
            ]
        ),
        neutral_cluster_transport_mobility_density_mol_m_s=(
            cluster_mobility_rollup[
                "neutral_cluster_transport_mobility_density_mol_m_s"
            ]
        ),
        charged_cluster_direct_sigma_mS_cm=charged_cluster_sigma_rollup[
            "charged_cluster_direct_sigma_mS_cm"
        ],
        charged_cluster_corrector_sigma_mS_cm=charged_cluster_sigma_rollup[
            "charged_cluster_corrector_sigma_mS_cm"
        ],
        charged_cluster_net_sigma_mS_cm=charged_cluster_sigma_rollup[
            "charged_cluster_net_sigma_mS_cm"
        ],
        cluster_thermodynamic_diagnostics=(
            _cluster_thermodynamic_diagnostics(
                molecular_case,
                molecular_result,
                primitive_parameters,
            )
        ),
    )


def _compute_case_result(
    molecular_case: MolecularPropertyDbCase,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
) -> MolecularMoriConductivityResult:
    return _compute_case_result_with_cluster_shifts(
        molecular_case,
        primitive_parameters,
        options,
        {},
    )


def _compute_case_result_with_cluster_shifts(
    molecular_case: MolecularPropertyDbCase,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
) -> MolecularMoriConductivityResult:
    molecular_options = MolecularMoriOptions(
        max_cluster_ion_count=options.max_cluster_ion_count,
        max_packing_fraction=options.max_packing_fraction,
        free_volume_exponent=options.free_volume_exponent,
        translation_jump_length_multiplier=(
            options.translation_jump_length_multiplier
        ),
        primitive_parameters=primitive_parameters,
    )
    return compute_molecular_electrolyte_conductivity_with_diagnostic_cluster_shifts(
        molecular_case.recipe,
        molecular_case.species_inputs,
        ProvidedPropertyDescriptorBackend(),
        molecular_options,
        diagnostic_cluster_standard_free_energy_shift_over_RT_by_label,
    )


def conductivity_decomposition_diagnostic(
    direct_sigma_mS_cm: float,
    corrector_sigma_mS_cm: float,
    predicted_sigma_mS_cm: float,
    empirical_sigma_mS_cm: float,
) -> ConductivityDecompositionDiagnostic:
    direct_sigma_value = _nonnegative_float(
        direct_sigma_mS_cm,
        "direct_sigma_mS_cm",
    )
    corrector_sigma_value = _nonnegative_float(
        corrector_sigma_mS_cm,
        "corrector_sigma_mS_cm",
    )
    predicted_sigma_value = _finite_float(
        predicted_sigma_mS_cm,
        "predicted_sigma_mS_cm",
    )
    empirical_sigma_value = _nonnegative_float(
        empirical_sigma_mS_cm,
        "empirical_sigma_mS_cm",
    )
    if direct_sigma_value == 0.0:
        if empirical_sigma_value == 0.0 and predicted_sigma_value == 0.0:
            return ConductivityDecompositionDiagnostic(
                direct_capacity_gap_mS_cm=0.0,
                corrector_target_mS_cm=0.0,
                corrector_residual_mS_cm=0.0,
                direct_capacity_failure=False,
                corrector_too_strong_failure=False,
                corrector_too_weak_failure=False,
            )
        raise ValueError(
            "positive conductivity row has zero direct Markov-additive capacity"
        )
    direct_capacity_gap_mS_cm = empirical_sigma_value - direct_sigma_value
    corrector_target_mS_cm = max(0.0, direct_sigma_value - empirical_sigma_value)
    corrector_residual_mS_cm = corrector_sigma_value - corrector_target_mS_cm
    direct_capacity_failure = direct_capacity_gap_mS_cm > 0.0
    corrector_too_strong_failure = (
        not direct_capacity_failure and corrector_residual_mS_cm > 0.0
    )
    corrector_too_weak_failure = (
        not direct_capacity_failure and corrector_residual_mS_cm < 0.0
    )
    return ConductivityDecompositionDiagnostic(
        direct_capacity_gap_mS_cm=_finite_float(
            direct_capacity_gap_mS_cm,
            "direct_capacity_gap_mS_cm",
        ),
        corrector_target_mS_cm=_nonnegative_float(
            corrector_target_mS_cm,
            "corrector_target_mS_cm",
        ),
        corrector_residual_mS_cm=_finite_float(
            corrector_residual_mS_cm,
            "corrector_residual_mS_cm",
        ),
        direct_capacity_failure=direct_capacity_failure,
        corrector_too_strong_failure=corrector_too_strong_failure,
        corrector_too_weak_failure=corrector_too_weak_failure,
    )


def measured_molecular_property_db_consumed_parameter_fields(
    cases: tuple[MolecularPropertyDbCase, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
    baseline_audit_result: MolecularPropertyDbAuditResult,
) -> tuple[str, ...]:
    return _measured_consumed_parameter_fields(
        cases,
        primitive_parameters,
        options,
        baseline_audit_result,
    )


def cluster_sensitivity_diagnostics_for_row(
    molecular_case: MolecularPropertyDbCase,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
    row_result: MolecularPropertyDbRowResult,
) -> tuple[MolecularClusterSensitivityDiagnostic, ...]:
    sensitivity_step_over_RT = _positive_float(
        options.cluster_sensitivity_step_over_RT,
        "cluster_sensitivity_step_over_RT",
    )
    diagnostics: list[MolecularClusterSensitivityDiagnostic] = []
    for cluster_diagnostic in row_result.cluster_thermodynamic_diagnostics:
        lower_deltaG_result = _compute_case_result_with_cluster_shifts(
            molecular_case,
            primitive_parameters,
            options,
            {cluster_diagnostic.cluster_label: -sensitivity_step_over_RT},
        )
        higher_deltaG_result = _compute_case_result_with_cluster_shifts(
            molecular_case,
            primitive_parameters,
            options,
            {cluster_diagnostic.cluster_label: sensitivity_step_over_RT},
        )
        sensitivity_mS_cm_per_logK = (
            lower_deltaG_result.sigma_mS_cm
            - higher_deltaG_result.sigma_mS_cm
        ) / (2.0 * sensitivity_step_over_RT)
        diagnostics.append(
            MolecularClusterSensitivityDiagnostic(
                row_id=row_result.row_id,
                cluster_label=cluster_diagnostic.cluster_label,
                cluster_kind=cluster_diagnostic.cluster_kind,
                net_charge_number=cluster_diagnostic.net_charge_number,
                baseline_concentration_mol_m3=(
                    cluster_diagnostic.concentration_mol_m3
                ),
                baseline_deltaG_over_RT=(
                    cluster_diagnostic.standard_free_energy_over_RT
                ),
                sigma_lower_deltaG_mS_cm=lower_deltaG_result.sigma_mS_cm,
                sigma_higher_deltaG_mS_cm=higher_deltaG_result.sigma_mS_cm,
                sensitivity_mS_cm_per_logK=sensitivity_mS_cm_per_logK,
                direction_needed=_cluster_logK_direction_needed(
                    row_result.residual_mS_cm,
                    sensitivity_mS_cm_per_logK,
                ),
            )
        )
    return tuple(
        sorted(
            diagnostics,
            key=lambda diagnostic: abs(diagnostic.sensitivity_mS_cm_per_logK),
            reverse=True,
        )
    )


def _cluster_logK_direction_needed(
    residual_mS_cm: float,
    sensitivity_mS_cm_per_logK: float,
) -> str:
    residual_value = _finite_float(residual_mS_cm, "residual_mS_cm")
    sensitivity_value = _finite_float(
        sensitivity_mS_cm_per_logK,
        "sensitivity_mS_cm_per_logK",
    )
    if residual_value == 0.0 or sensitivity_value == 0.0:
        return "none"
    if residual_value > 0.0:
        if sensitivity_value < 0.0:
            return "increase_logK"
        return "decrease_logK"
    if sensitivity_value > 0.0:
        return "increase_logK"
    return "decrease_logK"


def _cluster_fraction_rollup(
    molecular_result: MolecularMoriConductivityResult,
) -> Mapping[str, float]:
    total_ion_concentration_mol_m3 = _total_analytical_ion_concentration_mol_m3(
        molecular_result,
    )
    free_ion_concentration_mol_m3 = math.fsum(
        molecular_result.speciation.free_component_concentrations_mol_m3[
            component.species_name
        ]
        for component in molecular_result.speciation.components
    )
    charged_cluster_ion_concentration_mol_m3 = 0.0
    neutral_cluster_ion_concentration_mol_m3 = 0.0
    for cluster_template in molecular_result.speciation.cluster_templates:
        cluster_concentration_mol_m3 = (
            molecular_result.speciation.cluster_concentrations_mol_m3[
                cluster_template.label
            ]
        )
        cluster_ion_concentration_mol_m3 = (
            cluster_concentration_mol_m3
            * math.fsum(cluster_template.stoichiometry.values())
        )
        if cluster_template.net_charge_number == 0:
            neutral_cluster_ion_concentration_mol_m3 += (
                cluster_ion_concentration_mol_m3
            )
        else:
            charged_cluster_ion_concentration_mol_m3 += (
                cluster_ion_concentration_mol_m3
            )
    return {
        "free_ion_fraction": (
            free_ion_concentration_mol_m3 / total_ion_concentration_mol_m3
        ),
        "charged_cluster_fraction": (
            charged_cluster_ion_concentration_mol_m3
            / total_ion_concentration_mol_m3
        ),
        "neutral_cluster_fraction": (
            neutral_cluster_ion_concentration_mol_m3
            / total_ion_concentration_mol_m3
        ),
    }


def _cluster_transport_mobility_rollup(
    molecular_result: MolecularMoriConductivityResult,
) -> Mapping[str, float]:
    cluster_transport_mobility_density_mol_m_s = 0.0
    charged_cluster_transport_mobility_density_mol_m_s = 0.0
    neutral_cluster_transport_mobility_density_mol_m_s = 0.0
    for transport_state in molecular_result.transport_states:
        if not _transport_state_is_cluster_center(transport_state.transport_role):
            continue
        mobility_density_mol_m_s = (
            transport_state.concentration_mol_m3
            * transport_state.diffusion_m2_s
        )
        cluster_transport_mobility_density_mol_m_s += (
            mobility_density_mol_m_s
        )
        if transport_state.center_charge_number == 0:
            neutral_cluster_transport_mobility_density_mol_m_s += (
                mobility_density_mol_m_s
            )
        else:
            charged_cluster_transport_mobility_density_mol_m_s += (
                mobility_density_mol_m_s
            )
    return {
        "cluster_transport_mobility_density_mol_m_s": (
            cluster_transport_mobility_density_mol_m_s
        ),
        "charged_cluster_transport_mobility_density_mol_m_s": (
            charged_cluster_transport_mobility_density_mol_m_s
        ),
        "neutral_cluster_transport_mobility_density_mol_m_s": (
            neutral_cluster_transport_mobility_density_mol_m_s
        ),
    }


def _charged_cluster_event_sigma_rollup(
    molecular_result: MolecularMoriConductivityResult,
) -> Mapping[str, float]:
    charged_cluster_labels = tuple(
        transport_state.label
        for transport_state in molecular_result.transport_states
        if _transport_state_is_cluster_center(transport_state.transport_role)
        and transport_state.center_charge_number != 0
    )
    sigma_rollup = _event_sigma_rollup_for_transport_labels(
        molecular_result,
        charged_cluster_labels,
    )
    return {
        "charged_cluster_direct_sigma_mS_cm": (
            sigma_rollup["direct_sigma_mS_cm"]
        ),
        "charged_cluster_corrector_sigma_mS_cm": (
            sigma_rollup["corrector_sigma_mS_cm"]
        ),
        "charged_cluster_net_sigma_mS_cm": sigma_rollup["net_sigma_mS_cm"],
    }


def _transport_role_event_sigma_rollup(
    molecular_result: MolecularMoriConductivityResult,
) -> Mapping[str, Mapping[str, float]]:
    direct_sigma_by_transport_role_mS_cm: dict[str, float] = {}
    corrector_sigma_by_transport_role_mS_cm: dict[str, float] = {}
    net_sigma_by_transport_role_mS_cm: dict[str, float] = {}
    for transport_role in TRANSPORT_ROLE_DIRECT_CORRECTOR_ATTRIBUTION_LABELS:
        transport_state_labels = tuple(
            transport_state.label
            for transport_state in molecular_result.transport_states
            if transport_state.transport_role == transport_role
        )
        sigma_rollup = _event_sigma_rollup_for_transport_labels(
            molecular_result,
            transport_state_labels,
        )
        direct_sigma_by_transport_role_mS_cm[transport_role] = sigma_rollup[
            "direct_sigma_mS_cm"
        ]
        corrector_sigma_by_transport_role_mS_cm[transport_role] = sigma_rollup[
            "corrector_sigma_mS_cm"
        ]
        net_sigma_by_transport_role_mS_cm[transport_role] = sigma_rollup[
            "net_sigma_mS_cm"
        ]
    return {
        "direct_sigma_by_transport_role_mS_cm": (
            direct_sigma_by_transport_role_mS_cm
        ),
        "corrector_sigma_by_transport_role_mS_cm": (
            corrector_sigma_by_transport_role_mS_cm
        ),
        "net_sigma_by_transport_role_mS_cm": net_sigma_by_transport_role_mS_cm,
    }


def _event_sigma_rollup_for_transport_labels(
    molecular_result: MolecularMoriConductivityResult,
    transport_state_labels: tuple[str, ...],
) -> Mapping[str, float]:
    if not transport_state_labels:
        return {
            "direct_sigma_mS_cm": 0.0,
            "corrector_sigma_mS_cm": 0.0,
            "net_sigma_mS_cm": 0.0,
        }
    state_concentrations = np.asarray(
        molecular_result.markov_state_concentrations_mol_m3,
        dtype=float,
    )
    state_count = len(molecular_result.markov_state_labels)
    if state_concentrations.shape != (state_count,):
        raise ValueError("markov state concentration shape mismatch")
    if molecular_result.markov_additive_result.drift_by_state_m_s.shape != (
        state_count,
        3,
    ):
        raise ValueError("markov drift shape mismatch")
    direct_axis_density = np.zeros(3, dtype=float)
    drift_by_state = np.zeros((state_count, 3), dtype=float)
    for event in molecular_result.events:
        if not _event_label_matches_transport_labels(
            event.label,
            transport_state_labels,
        ):
            continue
        if event.from_state_index < 0 or event.from_state_index >= state_count:
            raise ValueError(f"{event.label}.from_state_index out of range")
        displacement = np.asarray(event.charge_displacement_m, dtype=float)
        if displacement.shape != (3,):
            raise ValueError(f"{event.label}.charge_displacement_m shape mismatch")
        event_rate_s_inv = _positive_float(
            event.rate_s_inv,
            f"{event.label}.rate_s_inv",
        )
        source_concentration_mol_m3 = state_concentrations[event.from_state_index]
        direct_axis_density += (
            0.5
            * source_concentration_mol_m3
            * event_rate_s_inv
            * displacement
            * displacement
        )
        drift_by_state[event.from_state_index, :] += event_rate_s_inv * displacement
    direct_sigma_mS_cm = _axis_density_sigma_mS_cm(
        direct_axis_density,
        molecular_result.solvent_environment.temperature_K,
    )
    corrector_sigma_mS_cm = _marginal_corrector_sigma_mS_cm(
        subset_drift_by_state=drift_by_state,
        total_drift_by_state=molecular_result.markov_additive_result.drift_by_state_m_s,
        memory_matrix=molecular_result.markov_additive_result.corrector_mori_input.memory_self_energy_matrix,
        state_concentrations=state_concentrations,
        temperature_K=molecular_result.solvent_environment.temperature_K,
    )
    return {
        "direct_sigma_mS_cm": direct_sigma_mS_cm,
        "corrector_sigma_mS_cm": corrector_sigma_mS_cm,
        "net_sigma_mS_cm": direct_sigma_mS_cm - corrector_sigma_mS_cm,
    }


def _event_label_matches_transport_labels(
    event_label: str,
    transport_state_labels: tuple[str, ...],
) -> bool:
    if not event_label:
        raise ValueError("event_label must be nonempty")
    for transport_state_label in transport_state_labels:
        ordinary_prefix = f"ordinary_mobile_translation:{transport_state_label}:"
        capture_prefix = f"atmosphere_memory_capture:{transport_state_label}:"
        back_prefix = (
            f"atmosphere_memory_back_relaxation:{transport_state_label}:"
        )
        ssip_relative_prefix = (
            "solvent_separated_pair_relative_translation:"
            f"{transport_state_label}:"
        )
        ssip_com_prefix = (
            "solvent_separated_pair_com_translation:"
            f"{transport_state_label}:"
        )
        ssip_residual_prefix = (
            "solvent_separated_pair_residual_center_translation:"
            f"{transport_state_label}:"
        )
        if (
            event_label.startswith(ordinary_prefix)
            or event_label.startswith(capture_prefix)
            or event_label.startswith(back_prefix)
            or event_label.startswith(ssip_relative_prefix)
            or event_label.startswith(ssip_com_prefix)
            or event_label.startswith(ssip_residual_prefix)
        ):
            return True
    return False


def _transport_state_is_cluster_center(
    transport_role: str,
) -> bool:
    return transport_role in (
        TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
        TRANSPORT_ROLE_CLUSTER_COM_CENTER,
        TRANSPORT_ROLE_CONTACT_PAIR_CENTER,
        TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    )


def _axis_density_sigma_mS_cm(
    axis_density_m2_s_mol_m3: np.ndarray,
    temperature_K: float,
) -> float:
    if axis_density_m2_s_mol_m3.shape != (3,):
        raise ValueError("axis density must have three Cartesian components")
    return float(
        F
        * F
        / (3.0 * R * _positive_float(temperature_K, "temperature_K"))
        * float(np.sum(axis_density_m2_s_mol_m3))
        * S_M_TO_MS_CM
    )


def _marginal_corrector_sigma_mS_cm(
    subset_drift_by_state: np.ndarray,
    total_drift_by_state: np.ndarray,
    memory_matrix: np.ndarray,
    state_concentrations: np.ndarray,
    temperature_K: float,
) -> float:
    self_corrector_density = _cross_corrector_axis_density(
        subset_drift_by_state,
        subset_drift_by_state,
        memory_matrix,
        state_concentrations,
    )
    total_cross_density = _cross_corrector_axis_density(
        subset_drift_by_state,
        total_drift_by_state,
        memory_matrix,
        state_concentrations,
    )
    marginal_corrector_density = 2.0 * total_cross_density - self_corrector_density
    return _axis_density_sigma_mS_cm(
        marginal_corrector_density,
        temperature_K,
    )


def _cross_corrector_axis_density(
    first_drift_by_state: np.ndarray,
    second_drift_by_state: np.ndarray,
    memory_matrix: np.ndarray,
    state_concentrations: np.ndarray,
) -> np.ndarray:
    if first_drift_by_state.shape != second_drift_by_state.shape:
        raise ValueError("drift matrix shapes must match")
    state_count, axis_count = first_drift_by_state.shape
    if axis_count != 3:
        raise ValueError("drift matrix must have three Cartesian axes")
    if memory_matrix.shape != (state_count, state_count):
        raise ValueError("memory matrix shape must match drift state count")
    if state_concentrations.shape != (state_count,):
        raise ValueError("state concentration shape must match drift state count")
    concentration_sqrt = np.sqrt(state_concentrations)
    first_coupling = (concentration_sqrt[:, None] * first_drift_by_state).T
    second_coupling = (concentration_sqrt[:, None] * second_drift_by_state).T
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (memory_matrix + memory_matrix.T)
    )
    tolerance = math.sqrt(np.finfo(float).eps) * max(
        1.0,
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 1.0,
    )
    positive_mode_mask = eigenvalues > tolerance
    axis_density = np.zeros(3, dtype=float)
    if not np.any(positive_mode_mask):
        return axis_density
    positive_eigenvalues = eigenvalues[positive_mode_mask]
    positive_eigenvectors = eigenvectors[:, positive_mode_mask]
    for axis_index in range(3):
        first_projection = positive_eigenvectors.T @ first_coupling[axis_index]
        second_projection = positive_eigenvectors.T @ second_coupling[axis_index]
        axis_density[axis_index] = float(
            np.sum(first_projection * second_projection / positive_eigenvalues)
        )
    return axis_density


def _cluster_thermodynamic_diagnostics(
    molecular_case: MolecularPropertyDbCase,
    molecular_result: MolecularMoriConductivityResult,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> tuple[MolecularClusterThermodynamicDiagnostic, ...]:
    total_ion_concentration_mol_m3 = _total_analytical_ion_concentration_mol_m3(
        molecular_result,
    )
    diagnostics: list[MolecularClusterThermodynamicDiagnostic] = []
    for cluster_template in molecular_result.speciation.cluster_templates:
        cluster_concentration_mol_m3 = (
            molecular_result.speciation.cluster_concentrations_mol_m3[
                cluster_template.label
            ]
        )
        cluster_ion_count = math.fsum(cluster_template.stoichiometry.values())
        activity_correction_J_mol = cluster_activity_correction_J_mol(
            molecular_result.speciation.components,
            cluster_template,
            molecular_result.speciation.free_component_concentrations_mol_m3,
            molecular_result.solvent_environment,
            primitive_parameters,
        )
        standard_free_energy_over_RT = (
            cluster_template.standard_free_energy_J_mol
            / (R * molecular_case.recipe.temperature_K)
        )
        diagnostics.append(
            MolecularClusterThermodynamicDiagnostic(
                row_id=molecular_case.row_id,
                cluster_label=cluster_template.label,
                cluster_kind=cluster_template.cluster_kind,
                stoichiometry=dict(cluster_template.stoichiometry),
                net_charge_number=cluster_template.net_charge_number,
                concentration_mol_m3=cluster_concentration_mol_m3,
                concentration_fraction_of_total_ion=(
                    cluster_concentration_mol_m3
                    * cluster_ion_count
                    / total_ion_concentration_mol_m3
                ),
                standard_free_energy_J_mol=(
                    cluster_template.standard_free_energy_J_mol
                ),
                standard_free_energy_over_RT=standard_free_energy_over_RT,
                log_equilibrium_constant=-standard_free_energy_over_RT,
                coulomb_J_mol=cluster_template.coulomb_J_mol,
                desolvation_J_mol=cluster_template.desolvation_J_mol,
                coordination_J_mol=cluster_template.coordination_J_mol,
                steric_J_mol=cluster_template.steric_J_mol,
                entropy_J_mol=cluster_template.entropy_J_mol,
                activity_reference_J_mol=(
                    cluster_template.activity_reference_J_mol
                ),
                activity_correction_J_mol=activity_correction_J_mol,
                hydrodynamic_radius_A=cluster_template.hydrodynamic_radius_A,
                molecular_volume_A3=cluster_template.molecular_volume_A3,
            )
        )
    return tuple(diagnostics)


def _total_analytical_ion_concentration_mol_m3(
    molecular_result: MolecularMoriConductivityResult,
) -> float:
    return _positive_float(
        math.fsum(
            component.analytical_concentration_M
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            for component in molecular_result.speciation.components
        ),
        "total_analytical_ion_concentration_mol_m3",
    )


def _charge_weighted_transport_concentration_mol_m3(
    molecular_result: MolecularMoriConductivityResult,
) -> float:
    charge_weighted_concentration_mol_m3 = math.fsum(
        transport_state.concentration_mol_m3
        * transport_state.center_charge_number
        * transport_state.center_charge_number
        for transport_state in molecular_result.transport_states
    )
    return _nonnegative_float(
        charge_weighted_concentration_mol_m3,
        "charge_weighted_transport_concentration_mol_m3",
    )


def _measured_consumed_parameter_fields(
    cases: tuple[MolecularPropertyDbCase, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
    baseline_audit_result: MolecularPropertyDbAuditResult,
) -> tuple[str, ...]:
    perturbation_scales = _positive_float_tuple(
        options.parameter_consumption_perturbation_scales,
        "parameter_consumption_perturbation_scales",
    )
    if all(perturbation_scale == 1.0 for perturbation_scale in perturbation_scales):
        raise ValueError(
            "parameter_consumption_perturbation_scales must include a value "
            "that differs from one"
        )
    consumed_parameter_fields: list[str] = []
    for parameter_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        if _parameter_is_consumed_by_any_perturbation(
            cases,
            primitive_parameters,
            options,
            baseline_audit_result,
            parameter_name,
            perturbation_scales,
        ):
            consumed_parameter_fields.append(parameter_name)
    return tuple(consumed_parameter_fields)


def _parameter_is_consumed_by_any_perturbation(
    cases: tuple[MolecularPropertyDbCase, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
    baseline_audit_result: MolecularPropertyDbAuditResult,
    parameter_name: str,
    perturbation_scales: tuple[float, ...],
) -> bool:
    baseline_parameter_value = getattr(primitive_parameters, parameter_name)
    for perturbation_scale in perturbation_scales:
        if perturbation_scale == 1.0:
            continue
        perturbed_parameter_value = _primitive_parameter_perturbed_value(
            parameter_name,
            baseline_parameter_value,
            perturbation_scale,
        )
        perturbed_parameters = replace(
            primitive_parameters,
            **{parameter_name: perturbed_parameter_value},
        )
        try:
            perturbed_audit_result = audit_molecular_property_db_cases(
                cases,
                perturbed_parameters,
                options,
            )
        except (
            FloatingPointError,
            OverflowError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            return True
        if _audit_results_differ(
            baseline_audit_result,
            perturbed_audit_result,
        ):
            return True
    return False


def _primitive_parameter_perturbed_value(
    parameter_name: str,
    baseline_parameter_value: float,
    perturbation_scale: float,
) -> float:
    transform_name = CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[
        parameter_name
    ]
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return baseline_parameter_value * perturbation_scale
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return baseline_parameter_value + math.log(perturbation_scale)
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


def _audit_results_differ(
    baseline_audit_result: MolecularPropertyDbAuditResult,
    perturbed_audit_result: MolecularPropertyDbAuditResult,
) -> bool:
    if len(baseline_audit_result.rows) != len(perturbed_audit_result.rows):
        return True
    baseline_values = _audit_comparison_values(baseline_audit_result)
    perturbed_values = _audit_comparison_values(perturbed_audit_result)
    tolerance_factor = math.sqrt(np.finfo(float).eps)
    for baseline_value, perturbed_value in zip(
        baseline_values,
        perturbed_values,
    ):
        difference_scale = max(
            1.0,
            abs(baseline_value),
            abs(perturbed_value),
        )
        tolerance = tolerance_factor * difference_scale
        if abs(baseline_value - perturbed_value) > tolerance:
            return True
    return False


def _audit_comparison_values(
    audit_result: MolecularPropertyDbAuditResult,
) -> tuple[float, ...]:
    values: list[float] = [
        audit_result.mae_mS_cm,
        audit_result.rmse_mS_cm,
        audit_result.bias_mS_cm,
        audit_result.pearson_r,
        audit_result.maximum_abs_residual_mS_cm,
        audit_result.maximum_mass_balance_residual,
        audit_result.maximum_row_sum_residual,
        audit_result.maximum_stationary_residual,
        audit_result.maximum_detailed_balance_residual,
        audit_result.maximum_event_reversal_residual,
        audit_result.zero_charge_sigma_mS_cm,
    ]
    for row_result in audit_result.rows:
        values.extend(
            (
                row_result.predicted_sigma_mS_cm,
                row_result.direct_sigma_mS_cm,
                row_result.corrector_sigma_mS_cm,
                row_result.direct_capacity_gap_mS_cm,
                row_result.corrector_target_mS_cm,
                row_result.corrector_residual_mS_cm,
                float(row_result.direct_capacity_failure),
                float(row_result.corrector_too_strong_failure),
                float(row_result.corrector_too_weak_failure),
                row_result.charge_weighted_transport_concentration_mol_m3,
                row_result.charged_cluster_direct_sigma_mS_cm,
                row_result.charged_cluster_corrector_sigma_mS_cm,
                row_result.charged_cluster_net_sigma_mS_cm,
                row_result.mass_balance_residual_mol_m3,
                row_result.row_sum_residual,
                row_result.stationary_residual_mol_m3_s,
                row_result.detailed_balance_residual_mol_m3_s,
                row_result.event_reversal_residual_mol_m3_s,
                row_result.free_ion_fraction,
                row_result.charged_cluster_fraction,
                row_result.neutral_cluster_fraction,
                row_result.cluster_transport_mobility_density_mol_m_s,
                row_result.charged_cluster_transport_mobility_density_mol_m_s,
                row_result.neutral_cluster_transport_mobility_density_mol_m_s,
            )
        )
        for transport_role in TRANSPORT_ROLE_DIRECT_CORRECTOR_ATTRIBUTION_LABELS:
            values.extend(
                (
                    row_result.direct_sigma_by_transport_role_mS_cm[
                        transport_role
                    ],
                    row_result.corrector_sigma_by_transport_role_mS_cm[
                        transport_role
                    ],
                    row_result.net_sigma_by_transport_role_mS_cm[transport_role],
                )
            )
        for cluster_diagnostic in row_result.cluster_thermodynamic_diagnostics:
            values.extend(
                (
                    cluster_diagnostic.concentration_mol_m3,
                    cluster_diagnostic.concentration_fraction_of_total_ion,
                    cluster_diagnostic.standard_free_energy_J_mol,
                    cluster_diagnostic.standard_free_energy_over_RT,
                    cluster_diagnostic.log_equilibrium_constant,
                    cluster_diagnostic.coulomb_J_mol,
                    cluster_diagnostic.desolvation_J_mol,
                    cluster_diagnostic.coordination_J_mol,
                    cluster_diagnostic.steric_J_mol,
                    cluster_diagnostic.entropy_J_mol,
                    cluster_diagnostic.activity_correction_J_mol,
                    cluster_diagnostic.hydrodynamic_radius_A,
                    cluster_diagnostic.molecular_volume_A3,
                )
            )
    return tuple(_finite_float(value, "audit_comparison_value") for value in values)


def _molecular_recipe_and_species_inputs(
    row_state: _MolecularPropertyDbRowState,
    registry_source: MolecularPropertyDbRegistrySource,
    options: MolecularPropertyDbAuditOptions,
) -> tuple[MolecularElectrolyteRecipe, Mapping[str, MolecularSpeciesInput]]:
    species_inputs: dict[str, MolecularSpeciesInput] = {}
    cation_molarities: dict[str, float] = {}
    anion_molarities: dict[str, float] = {}
    neutral_additives: dict[str, float] = {}
    for solvent_name in row_state.solvent_loadings:
        solvent_record = _required_neutral_compound_record(
            registry_source,
            solvent_name,
        )
        species_inputs[solvent_name] = _neutral_species_input(
            solvent_name,
            ROLE_SOLVENT,
            solvent_record,
            options,
        )
    for salt_name, salt_molarity_M in row_state.salt_loadings_M.items():
        salt_record = _required_ionic_compound_record(registry_source, salt_name)
        _add_ionic_compound_to_molecular_recipe(
            salt_record,
            salt_molarity_M,
            cation_molarities,
            anion_molarities,
            species_inputs,
            registry_source.cation_registry,
            options,
        )
    for additive_name, additive_weight_fraction in row_state.additive_loadings.items():
        additive_record = _required_registry_record(
            registry_source.additive_registry,
            additive_name,
            "additive",
        )
        if _record_is_ionic_compound(additive_record):
            additive_molarity_M = _additive_molarity_M(
                additive_weight_fraction,
                additive_record,
                row_state.mixture_density_g_ml,
            )
            _add_ionic_compound_to_molecular_recipe(
                additive_record,
                additive_molarity_M,
                cation_molarities,
                anion_molarities,
                species_inputs,
                registry_source.cation_registry,
                options,
            )
        else:
            neutral_additives[additive_name] = additive_weight_fraction
            species_inputs[additive_name] = _neutral_species_input(
                additive_name,
                ROLE_ADDITIVE,
                additive_record,
                options,
            )
    recipe = MolecularElectrolyteRecipe(
        cations=dict(cation_molarities),
        anions=dict(anion_molarities),
        solvents=dict(row_state.solvent_loadings),
        additives=neutral_additives,
        temperature_K=T_REF_K,
        pressure_Pa=PA_PER_ATM,
        mixture_properties=MolecularMixtureProperties(
            density_g_ml=row_state.mixture_density_g_ml,
            viscosity_cP=row_state.mixture_viscosity_cP,
            dielectric_constant=row_state.mixture_dielectric_constant,
        ),
    )
    return recipe, species_inputs


def _add_ionic_compound_to_molecular_recipe(
    compound_record: dict,
    compound_molarity_M: float,
    cation_molarities: dict[str, float],
    anion_molarities: dict[str, float],
    species_inputs: dict[str, MolecularSpeciesInput],
    cation_registry: Mapping[str, dict],
    options: MolecularPropertyDbAuditOptions,
) -> None:
    cation_name = _required_string(compound_record, "cation", "ionic.cation")
    anion_label = _required_string(compound_record, "anion", "ionic.anion")
    cation_record = _required_registry_record(cation_registry, cation_name, "cation")
    cation_charge_number = _required_int(
        cation_record,
        "charge",
        f"{cation_name}.charge",
    )
    anion_charge_number = _required_int(
        compound_record,
        "anion_charge",
        f"{anion_label}.anion_charge",
    )
    if cation_charge_number <= 0:
        raise ValueError(f"{cation_name}.charge must be positive")
    if anion_charge_number >= 0:
        raise ValueError(f"{anion_label}.anion_charge must be negative")
    common_charge_factor = math.gcd(
        cation_charge_number,
        abs(anion_charge_number),
    )
    cation_stoichiometric_count = abs(anion_charge_number) // common_charge_factor
    anion_stoichiometric_count = cation_charge_number // common_charge_factor
    cation_label = _cation_species_label(cation_name, cation_record)
    _add_molarity(
        cation_molarities,
        cation_label,
        cation_stoichiometric_count * compound_molarity_M,
    )
    _add_molarity(
        anion_molarities,
        anion_label,
        anion_stoichiometric_count * compound_molarity_M,
    )
    if cation_label not in species_inputs:
        species_inputs[cation_label] = _cation_species_input(
            cation_label,
            cation_record,
            options,
        )
    if anion_label not in species_inputs:
        species_inputs[anion_label] = _anion_species_input(
            anion_label,
            compound_record,
            cation_record,
            options,
        )


def _add_molarity(
    concentration_by_species_name: dict[str, float],
    species_name: str,
    concentration_M: float,
) -> None:
    parsed_concentration_M = _positive_float(
        concentration_M,
        f"{species_name}.concentration_M",
    )
    if species_name in concentration_by_species_name:
        concentration_by_species_name[species_name] += parsed_concentration_M
    else:
        concentration_by_species_name[species_name] = parsed_concentration_M


def _cation_species_input(
    cation_label: str,
    cation_record: dict,
    options: MolecularPropertyDbAuditOptions,
) -> MolecularSpeciesInput:
    charge_number = _required_int(cation_record, "charge", f"{cation_label}.charge")
    ionic_radius_A = _required_positive_number(
        cation_record,
        "ionic_radius_A",
        f"{cation_label}.ionic_radius_A",
    )
    solvated_radius_A = _required_positive_number(
        cation_record,
        "solvated_radius_A",
        f"{cation_label}.solvated_radius_A",
    )
    molecular_volume_A3 = _sphere_volume_A3(ionic_radius_A)
    molecular_weight_g_mol = _molecular_weight_from_volume_density(
        molecular_volume_A3,
        options.ion_reference_density_g_ml,
    )
    return MolecularSpeciesInput(
        name=cation_label,
        role=ROLE_CATION,
        charge_number=charge_number,
        smiles="",
        xyz_coordinates=tuple(),
        property_overrides={
            "molecular_weight_g_mol": molecular_weight_g_mol,
            "hard_sphere_radius_A": ionic_radius_A,
            "hydrodynamic_radius_A": solvated_radius_A,
            "cavity_radius_A": ionic_radius_A,
            "charge_cloud_radius_A": ionic_radius_A,
            "molecular_volume_A3": molecular_volume_A3,
            "solvent_accessible_area_A2": _sphere_area_A2(ionic_radius_A),
            "dipole_D": 0.0,
            "quadrupole_D_A": 0.0,
            "polarizability_A3": ionic_radius_A * ionic_radius_A * ionic_radius_A,
            "donor_number": 0.0,
            "acceptor_number": 0.0,
            "hbond_donor_count": float(options.ion_hbond_donor_count),
            "hbond_acceptor_count": float(options.ion_hbond_acceptor_count),
            "epsilon_r_pure": options.ion_reference_dielectric_base,
            "viscosity_cP_pure": options.reference_ion_viscosity_cP,
            "density_g_ml": options.ion_reference_density_g_ml,
            "born_solvation_radius_A": ionic_radius_A,
            "coordination_affinity_J_mol": 0.0,
            "ligand_field_asymmetry": 1.0,
        },
        coordination_sites=tuple(),
    )


def _anion_species_input(
    anion_label: str,
    compound_record: dict,
    cation_record: dict,
    options: MolecularPropertyDbAuditOptions,
) -> MolecularSpeciesInput:
    anion_charge_number = _required_int(
        compound_record,
        "anion_charge",
        f"{anion_label}.anion_charge",
    )
    anion_radius_A = _required_positive_number(
        compound_record,
        "anion_radius",
        f"{anion_label}.anion_radius",
    )
    anion_volume_A3 = _required_positive_number(
        compound_record,
        "anion_volume",
        f"{anion_label}.anion_volume",
    )
    cation_radius_A = _required_positive_number(
        cation_record,
        "ionic_radius_A",
        "cation.ionic_radius_A",
    )
    compound_molecular_weight_g_mol = _required_positive_number(
        compound_record,
        "molecular_weight",
        f"{anion_label}.compound_molecular_weight",
    )
    cation_proxy_weight_g_mol = _molecular_weight_from_volume_density(
        _sphere_volume_A3(cation_radius_A),
        options.ion_reference_density_g_ml,
    )
    anion_molecular_weight_g_mol = compound_molecular_weight_g_mol - cation_proxy_weight_g_mol
    if anion_molecular_weight_g_mol <= 0.0:
        raise ValueError(f"{anion_label}.molecular_weight_g_mol must be positive")
    return MolecularSpeciesInput(
        name=anion_label,
        role=ROLE_ANION,
        charge_number=anion_charge_number,
        smiles=_required_string(compound_record, "SMILES", f"{anion_label}.SMILES"),
        xyz_coordinates=tuple(),
        property_overrides={
            "molecular_weight_g_mol": anion_molecular_weight_g_mol,
            "hard_sphere_radius_A": anion_radius_A,
            "hydrodynamic_radius_A": anion_radius_A,
            "cavity_radius_A": anion_radius_A,
            "charge_cloud_radius_A": anion_radius_A,
            "molecular_volume_A3": anion_volume_A3,
            "solvent_accessible_area_A2": _sphere_area_A2(anion_radius_A),
            "dipole_D": _optional_nonnegative_number(
                compound_record,
                "dipole_moment_D",
            ),
            "quadrupole_D_A": 0.0,
            "polarizability_A3": _polarizability_A3_from_record(
                compound_record,
                anion_volume_A3,
            ),
            "donor_number": _optional_nonnegative_number(
                compound_record,
                "donor_number",
            ),
            "acceptor_number": float(_functional_group_acceptor_count(compound_record)),
            "hbond_donor_count": float(options.ion_hbond_donor_count),
            "hbond_acceptor_count": float(options.ion_hbond_acceptor_count),
            "epsilon_r_pure": _required_positive_number(
                compound_record,
                "epsilon_r",
                f"{anion_label}.epsilon_r",
            ),
            "viscosity_cP_pure": _optional_positive_number_or_reference(
                compound_record,
                "viscosity_cP",
                options.reference_ion_viscosity_cP,
            ),
            "density_g_ml": _required_positive_number(
                compound_record,
                "density_g_ml",
                f"{anion_label}.density_g_ml",
            ),
            "born_solvation_radius_A": anion_radius_A,
            "coordination_affinity_J_mol": _coordination_affinity_J_mol(
                compound_record,
            ),
            "ligand_field_asymmetry": _positive_shape_factor(compound_record),
        },
        coordination_sites=tuple(),
    )


def _neutral_species_input(
    species_name: str,
    role: str,
    species_record: dict,
    options: MolecularPropertyDbAuditOptions,
) -> MolecularSpeciesInput:
    molecular_weight_g_mol = _required_positive_number(
        species_record,
        "molecular_weight",
        f"{species_name}.molecular_weight",
    )
    density_g_ml = _required_positive_number(
        species_record,
        "density_g_ml",
        f"{species_name}.density_g_ml",
    )
    molecular_volume_A3 = _occupied_molecular_volume_A3(
        molecular_weight_g_mol,
        density_g_ml,
        options.liquid_occupied_volume_fraction,
    )
    molecular_radius_A = _sphere_radius_A(molecular_volume_A3)
    return MolecularSpeciesInput(
        name=species_name,
        role=role,
        charge_number=0,
        smiles=_required_string(species_record, "SMILES", f"{species_name}.SMILES"),
        xyz_coordinates=tuple(),
        property_overrides={
            "molecular_weight_g_mol": molecular_weight_g_mol,
            "hard_sphere_radius_A": molecular_radius_A,
            "hydrodynamic_radius_A": molecular_radius_A,
            "cavity_radius_A": molecular_radius_A,
            "charge_cloud_radius_A": molecular_radius_A,
            "molecular_volume_A3": molecular_volume_A3,
            "solvent_accessible_area_A2": _sphere_area_A2(molecular_radius_A),
            "dipole_D": _required_nonnegative_number(
                species_record,
                "dipole_moment_D",
                f"{species_name}.dipole_moment_D",
            ),
            "quadrupole_D_A": 0.0,
            "polarizability_A3": _polarizability_A3_from_record(
                species_record,
                molecular_volume_A3,
            ),
            "donor_number": _required_nonnegative_number(
                species_record,
                "donor_number",
                f"{species_name}.donor_number",
            ),
            "acceptor_number": _required_nonnegative_number(
                species_record,
                "acceptor_number",
                f"{species_name}.acceptor_number",
            ),
            "hbond_donor_count": float(options.neutral_hbond_donor_count),
            "hbond_acceptor_count": float(
                _functional_group_acceptor_count(species_record)
            ),
            "epsilon_r_pure": _required_positive_number(
                species_record,
                "epsilon_r",
                f"{species_name}.epsilon_r",
            ),
            "viscosity_cP_pure": _required_positive_number(
                species_record,
                "viscosity_cP",
                f"{species_name}.viscosity_cP",
            ),
            "density_g_ml": density_g_ml,
            "born_solvation_radius_A": molecular_radius_A,
            "coordination_affinity_J_mol": _coordination_affinity_J_mol(
                species_record,
            ),
            "ligand_field_asymmetry": _positive_shape_factor(species_record),
        },
        coordination_sites=tuple(),
    )


def _mixture_density_g_ml(
    row_mapping: dict,
    solvent_loadings: Mapping[str, float],
    salt_loadings_M: Mapping[str, float],
    additive_loadings: Mapping[str, float],
    registry_source: MolecularPropertyDbRegistrySource,
) -> float:
    property_mapping = _required_mapping(row_mapping, "properties", "row.properties")
    if "density" in property_mapping:
        return _required_positive_number(property_mapping, "density", "properties.density")
    solvent_mass_g_per_liter = math.fsum(
        solvent_volume_fraction
        * GRAMS_PER_LITER_PER_G_ML
        * _required_positive_number(
            _required_neutral_compound_record(registry_source, solvent_name),
            "density_g_ml",
            f"{solvent_name}.density_g_ml",
        )
        for solvent_name, solvent_volume_fraction in solvent_loadings.items()
    )
    salt_mass_g_per_liter = math.fsum(
        salt_molarity_M
        * _required_positive_number(
            _required_ionic_compound_record(registry_source, salt_name),
            "molecular_weight",
            f"{salt_name}.molecular_weight",
        )
        for salt_name, salt_molarity_M in salt_loadings_M.items()
    )
    total_additive_weight_fraction = math.fsum(additive_loadings.values())
    if total_additive_weight_fraction >= 1.0:
        raise ValueError("total additive weight fraction must be below one")
    base_mass_g_per_liter = solvent_mass_g_per_liter + salt_mass_g_per_liter
    total_mass_g_per_liter = base_mass_g_per_liter / (
        1.0 - total_additive_weight_fraction
    )
    for additive_name in additive_loadings:
        _required_registry_record(
            registry_source.additive_registry,
            additive_name,
            "additive",
        )
    return _positive_float(
        total_mass_g_per_liter / GRAMS_PER_LITER_PER_G_ML,
        "mixture.density_g_ml",
    )


def _mixture_viscosity_cP(
    solvent_loadings: Mapping[str, float],
    salt_loadings_M: Mapping[str, float],
    additive_loadings: Mapping[str, float],
    mixture_density_g_ml: float,
    registry_source: MolecularPropertyDbRegistrySource,
) -> float:
    solvent_volume_fraction_sum = _positive_float(
        math.fsum(solvent_loadings.values()),
        "solvent_volume_fraction_sum",
    )
    solvent_log_viscosity = math.fsum(
        solvent_volume_fraction
        * math.log(
            _required_positive_number(
                _required_neutral_compound_record(registry_source, solvent_name),
                "viscosity_cP",
                f"{solvent_name}.viscosity_cP",
            )
        )
        for solvent_name, solvent_volume_fraction in solvent_loadings.items()
    ) / solvent_volume_fraction_sum
    jones_dole_factor = 1.0
    for salt_name, salt_molarity_M in salt_loadings_M.items():
        salt_record = _required_ionic_compound_record(registry_source, salt_name)
        jones_dole_factor += _jones_dole_viscosity_increment(
            salt_molarity_M,
            salt_record,
        )
    neutral_additive_log_viscosity_increment = 0.0
    for additive_name, additive_weight_fraction in additive_loadings.items():
        additive_record = _required_registry_record(
            registry_source.additive_registry,
            additive_name,
            "additive",
        )
        if _record_is_ionic_compound(additive_record):
            additive_molarity_M = _additive_molarity_M(
                additive_weight_fraction,
                additive_record,
                mixture_density_g_ml,
            )
            jones_dole_factor += _jones_dole_viscosity_increment(
                additive_molarity_M,
                additive_record,
            )
        else:
            neutral_additive_log_viscosity_increment += (
                additive_weight_fraction
                * math.log(
                    _required_positive_number(
                        additive_record,
                        "viscosity_cP",
                        f"{additive_name}.viscosity_cP",
                    )
                )
            )
    return _positive_float(
        math.exp(solvent_log_viscosity + neutral_additive_log_viscosity_increment)
        * jones_dole_factor,
        "mixture.viscosity_cP",
    )


def _mixture_dielectric_constant(
    solvent_loadings: Mapping[str, float],
    salt_loadings_M: Mapping[str, float],
    additive_loadings: Mapping[str, float],
    mixture_density_g_ml: float,
    registry_source: MolecularPropertyDbRegistrySource,
) -> float:
    solvent_volume_fraction_sum = _positive_float(
        math.fsum(solvent_loadings.values()),
        "solvent_volume_fraction_sum",
    )
    dielectric_numerator = math.fsum(
        solvent_volume_fraction
        * _required_positive_number(
            _required_neutral_compound_record(registry_source, solvent_name),
            "epsilon_r",
            f"{solvent_name}.epsilon_r",
        )
        for solvent_name, solvent_volume_fraction in solvent_loadings.items()
    )
    dielectric_denominator = solvent_volume_fraction_sum
    decrement_fraction = 0.0
    for salt_name, salt_molarity_M in salt_loadings_M.items():
        salt_record = _required_ionic_compound_record(registry_source, salt_name)
        decrement_fraction += (
            salt_molarity_M
            * _required_nonnegative_number(
                salt_record,
                "dielectric_decrement_frac_per_M",
                f"{salt_name}.dielectric_decrement_frac_per_M",
            )
        )
    for additive_name, additive_weight_fraction in additive_loadings.items():
        additive_record = _required_registry_record(
            registry_source.additive_registry,
            additive_name,
            "additive",
        )
        if _record_is_ionic_compound(additive_record):
            additive_molarity_M = _additive_molarity_M(
                additive_weight_fraction,
                additive_record,
                mixture_density_g_ml,
            )
            decrement_fraction += (
                additive_molarity_M
                * _required_nonnegative_number(
                    additive_record,
                    "dielectric_decrement_frac_per_M",
                    f"{additive_name}.dielectric_decrement_frac_per_M",
                )
            )
        else:
            additive_volume_fraction = _additive_volume_fraction(
                additive_weight_fraction,
                additive_record,
                mixture_density_g_ml,
            )
            dielectric_numerator += additive_volume_fraction * _required_positive_number(
                additive_record,
                "epsilon_r",
                f"{additive_name}.epsilon_r",
            )
            dielectric_denominator += additive_volume_fraction
    if decrement_fraction >= 1.0:
        raise ValueError("dielectric decrement fraction must remain below one")
    solvent_additive_dielectric = dielectric_numerator / dielectric_denominator
    return _positive_float(
        solvent_additive_dielectric * (1.0 - decrement_fraction),
        "mixture.dielectric_constant",
    )


def _jones_dole_viscosity_increment(
    molarity_M: float,
    compound_record: dict,
) -> float:
    concentration_M = _positive_float(molarity_M, "jones_dole.concentration_M")
    jones_dole_A = _required_nonnegative_number(
        compound_record,
        "jones_dole_A",
        "jones_dole_A",
    )
    jones_dole_B = _required_nonnegative_number(
        compound_record,
        "jones_dole_B",
        "jones_dole_B",
    )
    return jones_dole_A * math.sqrt(concentration_M) + jones_dole_B * concentration_M


def _additive_molarity_M(
    additive_weight_fraction: float,
    additive_record: dict,
    mixture_density_g_ml: float,
) -> float:
    additive_mass_g_per_liter = (
        _nonnegative_float(additive_weight_fraction, "additive_weight_fraction")
        * _positive_float(mixture_density_g_ml, "mixture_density_g_ml")
        * GRAMS_PER_LITER_PER_G_ML
    )
    molecular_weight_g_mol = _required_positive_number(
        additive_record,
        "molecular_weight",
        "additive.molecular_weight",
    )
    return _positive_float(
        additive_mass_g_per_liter / molecular_weight_g_mol,
        "additive_molarity_M",
    )


def _additive_volume_fraction(
    additive_weight_fraction: float,
    additive_record: dict,
    mixture_density_g_ml: float,
) -> float:
    additive_mass_g_per_liter = (
        _nonnegative_float(additive_weight_fraction, "additive_weight_fraction")
        * _positive_float(mixture_density_g_ml, "mixture_density_g_ml")
        * GRAMS_PER_LITER_PER_G_ML
    )
    additive_density_g_ml = _required_positive_number(
        additive_record,
        "density_g_ml",
        "additive.density_g_ml",
    )
    return _nonnegative_float(
        additive_mass_g_per_liter
        / additive_density_g_ml
        / GRAMS_PER_LITER_PER_G_ML,
        "additive_volume_fraction",
    )


def _zero_charge_sigma_mS_cm(
    molecular_case: MolecularPropertyDbCase,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
) -> float:
    neutral_species_inputs: dict[str, MolecularSpeciesInput] = {}
    neutral_solvents: dict[str, float] = {}
    for species_name, volume_fraction in molecular_case.recipe.solvents.items():
        original_species_input = molecular_case.species_inputs[species_name]
        neutral_species_inputs[species_name] = MolecularSpeciesInput(
            name=original_species_input.name,
            role=ROLE_SOLVENT,
            charge_number=0,
            smiles=original_species_input.smiles,
            xyz_coordinates=original_species_input.xyz_coordinates,
            property_overrides=original_species_input.property_overrides,
            coordination_sites=original_species_input.coordination_sites,
        )
        neutral_solvents[species_name] = volume_fraction
    neutral_recipe = MolecularElectrolyteRecipe(
        cations={},
        anions={},
        solvents=neutral_solvents,
        additives={},
        temperature_K=molecular_case.recipe.temperature_K,
        pressure_Pa=molecular_case.recipe.pressure_Pa,
        mixture_properties=molecular_case.recipe.mixture_properties,
    )
    neutral_case = MolecularPropertyDbCase(
        row_id=molecular_case.row_id,
        source_row_ids=molecular_case.source_row_ids,
        source_formulation_key=molecular_case.source_formulation_key,
        source_solvent_loadings=molecular_case.source_solvent_loadings,
        source_salt_loadings_M=molecular_case.source_salt_loadings_M,
        source_additive_loadings=molecular_case.source_additive_loadings,
        recipe=neutral_recipe,
        species_inputs=neutral_species_inputs,
        empirical_sigma_mS_cm=0.0,
        empirical_sigma_spread_mS_cm=0.0,
    )
    return _compute_case_result(
        neutral_case,
        primitive_parameters,
        options,
    ).sigma_mS_cm


def _higher_viscosity_lowers_dilute_conductivity(
    molecular_case: MolecularPropertyDbCase,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
) -> bool:
    baseline_sigma_mS_cm = _compute_case_result(
        molecular_case,
        primitive_parameters,
        options,
    ).sigma_mS_cm
    higher_viscosity_recipe = MolecularElectrolyteRecipe(
        cations=molecular_case.recipe.cations,
        anions=molecular_case.recipe.anions,
        solvents=molecular_case.recipe.solvents,
        additives=molecular_case.recipe.additives,
        temperature_K=molecular_case.recipe.temperature_K,
        pressure_Pa=molecular_case.recipe.pressure_Pa,
        mixture_properties=MolecularMixtureProperties(
            density_g_ml=molecular_case.recipe.mixture_properties.density_g_ml,
            viscosity_cP=(
                molecular_case.recipe.mixture_properties.viscosity_cP
                * options.viscosity_monotonicity_scale
            ),
            dielectric_constant=(
                molecular_case.recipe.mixture_properties.dielectric_constant
            ),
        ),
    )
    higher_viscosity_case = MolecularPropertyDbCase(
        row_id=molecular_case.row_id,
        source_row_ids=molecular_case.source_row_ids,
        source_formulation_key=molecular_case.source_formulation_key,
        source_solvent_loadings=molecular_case.source_solvent_loadings,
        source_salt_loadings_M=molecular_case.source_salt_loadings_M,
        source_additive_loadings=molecular_case.source_additive_loadings,
        recipe=higher_viscosity_recipe,
        species_inputs=molecular_case.species_inputs,
        empirical_sigma_mS_cm=molecular_case.empirical_sigma_mS_cm,
        empirical_sigma_spread_mS_cm=(
            molecular_case.empirical_sigma_spread_mS_cm
        ),
    )
    higher_viscosity_sigma_mS_cm = _compute_case_result(
        higher_viscosity_case,
        primitive_parameters,
        options,
    ).sigma_mS_cm
    return higher_viscosity_sigma_mS_cm < baseline_sigma_mS_cm


def _higher_packing_lowers_local_mobility(
    molecular_case: MolecularPropertyDbCase,
    primitive_parameters: ConductivityPrimitiveParameterSet,
    options: MolecularPropertyDbAuditOptions,
) -> bool:
    baseline_sigma_mS_cm = _compute_case_result(
        molecular_case,
        primitive_parameters,
        options,
    ).sigma_mS_cm
    expanded_species_inputs = {
        species_name: _volume_scaled_species_input(
            species_input,
            options.packing_monotonicity_volume_scale,
        )
        for species_name, species_input in molecular_case.species_inputs.items()
    }
    expanded_case = MolecularPropertyDbCase(
        row_id=molecular_case.row_id,
        source_row_ids=molecular_case.source_row_ids,
        source_formulation_key=molecular_case.source_formulation_key,
        source_solvent_loadings=molecular_case.source_solvent_loadings,
        source_salt_loadings_M=molecular_case.source_salt_loadings_M,
        source_additive_loadings=molecular_case.source_additive_loadings,
        recipe=molecular_case.recipe,
        species_inputs=expanded_species_inputs,
        empirical_sigma_mS_cm=molecular_case.empirical_sigma_mS_cm,
        empirical_sigma_spread_mS_cm=(
            molecular_case.empirical_sigma_spread_mS_cm
        ),
    )
    expanded_sigma_mS_cm = _compute_case_result(
        expanded_case,
        primitive_parameters,
        options,
    ).sigma_mS_cm
    return expanded_sigma_mS_cm < baseline_sigma_mS_cm


def _volume_scaled_species_input(
    species_input: MolecularSpeciesInput,
    volume_scale: float,
) -> MolecularSpeciesInput:
    scale = _positive_float(volume_scale, "packing_monotonicity_volume_scale")
    radius_scale = scale ** (1.0 / 3.0)
    property_overrides = dict(species_input.property_overrides)
    for radius_key in (
        "hard_sphere_radius_A",
        "hydrodynamic_radius_A",
        "cavity_radius_A",
        "charge_cloud_radius_A",
    ):
        property_overrides[radius_key] = (
            _positive_float(property_overrides[radius_key], radius_key)
            * radius_scale
        )
    property_overrides["molecular_volume_A3"] = (
        _positive_float(
            property_overrides["molecular_volume_A3"],
            "molecular_volume_A3",
        )
        * scale
    )
    property_overrides["solvent_accessible_area_A2"] = (
        _nonnegative_float(
            property_overrides["solvent_accessible_area_A2"],
            "solvent_accessible_area_A2",
        )
        * (radius_scale * radius_scale)
    )
    return MolecularSpeciesInput(
        name=species_input.name,
        role=species_input.role,
        charge_number=species_input.charge_number,
        smiles=species_input.smiles,
        xyz_coordinates=species_input.xyz_coordinates,
        property_overrides=property_overrides,
        coordination_sites=species_input.coordination_sites,
    )


def _record_is_ionic_compound(record: dict) -> bool:
    return "cation" in record and "anion" in record and "anion_charge" in record


def _property_db_conductivity_mS_cm(row_mapping: dict) -> float:
    property_mapping = _required_mapping(row_mapping, "properties", "row.properties")
    return _required_positive_number(
        property_mapping,
        "conductivity_mS_cm",
        "properties.conductivity_mS_cm",
    )


def _cation_species_label(
    cation_name: str,
    cation_record: dict,
) -> str:
    if "ion_symbol" in cation_record:
        return _required_string(cation_record, "ion_symbol", f"{cation_name}.ion_symbol")
    return f"{cation_name}+"


def _occupied_molecular_volume_A3(
    molecular_weight_g_mol: float,
    density_g_ml: float,
    liquid_occupied_volume_fraction: float,
) -> float:
    unoccupied_molecular_volume_A3 = (
        _positive_float(molecular_weight_g_mol, "molecular_weight_g_mol")
        / _positive_float(density_g_ml, "density_g_ml")
        / N_A
        * CUBIC_CENTIMETER_TO_CUBIC_ANGSTROM
    )
    return _positive_float(
        unoccupied_molecular_volume_A3
        * _positive_float(
            liquid_occupied_volume_fraction,
            "liquid_occupied_volume_fraction",
        ),
        "occupied_molecular_volume_A3",
    )


def _molecular_weight_from_volume_density(
    molecular_volume_A3: float,
    density_g_ml: float,
) -> float:
    return _positive_float(
        _positive_float(molecular_volume_A3, "molecular_volume_A3")
        * CUBIC_ANGSTROM_PER_MOLE_TO_CM3
        * N_A
        * _positive_float(density_g_ml, "density_g_ml"),
        "molecular_weight_from_volume_density_g_mol",
    )


def _sphere_radius_A(molecular_volume_A3: float) -> float:
    return _positive_float(
        (
            _positive_float(molecular_volume_A3, "molecular_volume_A3")
            / (SPHERE_VOLUME_FACTOR * math.pi)
        )
        ** (1.0 / 3.0),
        "sphere_radius_A",
    )


def _sphere_volume_A3(radius_A: float) -> float:
    radius = _positive_float(radius_A, "radius_A")
    return SPHERE_VOLUME_FACTOR * math.pi * radius * radius * radius


def _sphere_area_A2(radius_A: float) -> float:
    radius = _positive_float(radius_A, "radius_A")
    return SPHERE_AREA_FACTOR * math.pi * radius * radius


def _polarizability_A3_from_record(
    species_record: dict,
    molecular_volume_A3: float,
) -> float:
    molecular_volume = _positive_float(molecular_volume_A3, "molecular_volume_A3")
    if "refractive_index_n_d" in species_record:
        refractive_index = _required_positive_number(
            species_record,
            "refractive_index_n_d",
            "refractive_index_n_d",
        )
        optical_dielectric = refractive_index * refractive_index
        return _clausius_mossotti_polarizability_A3(
            molecular_volume,
            optical_dielectric,
        )
    dielectric_constant = _required_positive_number(
        species_record,
        "epsilon_r",
        "epsilon_r",
    )
    return _clausius_mossotti_polarizability_A3(
        molecular_volume,
        dielectric_constant,
    )


def _clausius_mossotti_polarizability_A3(
    molecular_volume_A3: float,
    relative_dielectric: float,
) -> float:
    dielectric = _positive_float(relative_dielectric, "relative_dielectric")
    return _nonnegative_float(
        POLARIZABILITY_CLAUSIUS_MOSSOTTI_FACTOR
        * _positive_float(molecular_volume_A3, "molecular_volume_A3")
        * (dielectric - 1.0)
        / (dielectric + 2.0),
        "polarizability_A3",
    )


def _coordination_affinity_J_mol(species_record: dict) -> float:
    if "li_binding_energy_kJ_mol" in species_record:
        return (
            _required_nonnegative_number(
                species_record,
                "li_binding_energy_kJ_mol",
                "li_binding_energy_kJ_mol",
            )
            * 1000.0
        )
    if "ion_pair_binding_kj_mol" in species_record:
        return (
            _required_nonnegative_number(
                species_record,
                "ion_pair_binding_kj_mol",
                "ion_pair_binding_kj_mol",
            )
            * 1000.0
        )
    if "coordination_affinity_M_inv" in species_record:
        affinity_M_inv = _required_nonnegative_number(
            species_record,
            "coordination_affinity_M_inv",
            "coordination_affinity_M_inv",
        )
        return R * T_REF_K * math.log1p(affinity_M_inv)
    donor_number = _optional_nonnegative_number(species_record, "donor_number")
    acceptor_number = _optional_nonnegative_number(species_record, "acceptor_number")
    return R * T_REF_K * math.log1p(donor_number + acceptor_number)


def _positive_shape_factor(species_record: dict) -> float:
    if "ligand_field_asymmetry" in species_record:
        return 1.0 + _required_nonnegative_number(
            species_record,
            "ligand_field_asymmetry",
            "ligand_field_asymmetry",
        )
    if "n_rotatable_bonds" in species_record:
        return 1.0 + _required_nonnegative_number(
            species_record,
            "n_rotatable_bonds",
            "n_rotatable_bonds",
        ) / (
            1.0
            + _required_positive_number(
                species_record,
                "molecular_weight",
                "molecular_weight",
            )
        )
    return 1.0


def _functional_group_acceptor_count(species_record: dict) -> int:
    if "functional_groups" not in species_record:
        return 0
    functional_groups = _required_mapping(
        species_record,
        "functional_groups",
        "functional_groups",
    )
    acceptor_count = 0
    for functional_group_name in functional_groups:
        if "O" in str(functional_group_name) or "N" in str(functional_group_name):
            acceptor_count += int(
                _required_nonnegative_number(
                    functional_groups,
                    functional_group_name,
                    f"functional_groups.{functional_group_name}",
                )
            )
    return acceptor_count


def _mean_absolute_residual_or_zero(residuals_mS_cm: tuple[float, ...]) -> float:
    if not residuals_mS_cm:
        return 0.0
    residual_sum = math.fsum(
        abs(_finite_float(residual_mS_cm, "residual_mS_cm"))
        for residual_mS_cm in residuals_mS_cm
    )
    return float(residual_sum / len(residuals_mS_cm))


def _root_mean_square_residual_or_zero(residuals_mS_cm: tuple[float, ...]) -> float:
    if not residuals_mS_cm:
        return 0.0
    squared_residual_sum = math.fsum(
        _finite_float(residual_mS_cm, "residual_mS_cm")
        * _finite_float(residual_mS_cm, "residual_mS_cm")
        for residual_mS_cm in residuals_mS_cm
    )
    return float(math.sqrt(squared_residual_sum / len(residuals_mS_cm)))


def _mean_residual_or_zero(residuals_mS_cm: tuple[float, ...]) -> float:
    if not residuals_mS_cm:
        return 0.0
    residual_sum = math.fsum(
        _finite_float(residual_mS_cm, "residual_mS_cm")
        for residual_mS_cm in residuals_mS_cm
    )
    return float(residual_sum / len(residuals_mS_cm))


def _maximum_abs_residual_or_zero(residuals_mS_cm: tuple[float, ...]) -> float:
    if not residuals_mS_cm:
        return 0.0
    return float(
        max(
            abs(_finite_float(residual_mS_cm, "residual_mS_cm"))
            for residual_mS_cm in residuals_mS_cm
        )
    )


def _pearson_or_zero(
    empirical_sigmas_mS_cm: tuple[float, ...],
    predicted_sigmas_mS_cm: tuple[float, ...],
) -> float:
    if len(empirical_sigmas_mS_cm) != len(predicted_sigmas_mS_cm):
        raise ValueError("Pearson inputs must have equal length")
    if len(empirical_sigmas_mS_cm) < 2:
        return 0.0
    empirical_array = np.asarray(empirical_sigmas_mS_cm, dtype=float)
    predicted_array = np.asarray(predicted_sigmas_mS_cm, dtype=float)
    if not np.all(np.isfinite(empirical_array)):
        raise ValueError("empirical sigmas must be finite")
    if not np.all(np.isfinite(predicted_array)):
        raise ValueError("predicted sigmas must be finite")
    if float(np.std(empirical_array)) <= 0.0:
        return 0.0
    if float(np.std(predicted_array)) <= 0.0:
        return 0.0
    return float(np.corrcoef(empirical_array, predicted_array)[0, 1])


def _maximum_successful_value(
    successful_rows: tuple[MolecularPropertyDbRowResult, ...],
    field_name: str,
) -> float:
    if not successful_rows:
        return 0.0
    return float(
        max(
            _nonnegative_float(
                float(getattr(row_result, field_name)),
                field_name,
            )
            for row_result in successful_rows
        )
    )


def _raise_if_above_limit(
    observed_value: float,
    maximum_allowed_value: float,
    context: str,
) -> None:
    observed = _nonnegative_float(observed_value, context)
    maximum_allowed = _nonnegative_float(maximum_allowed_value, f"{context}.maximum")
    if observed > maximum_allowed:
        raise ValueError(f"{context} {observed} exceeds {maximum_allowed}")


def _load_physics_config_section(config_key: str) -> dict:
    with PHYSICS_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        physics_config = json.load(config_file)
    if config_key not in physics_config:
        raise ValueError(f"missing physics config section {config_key}")
    config_section = physics_config[config_key]
    if not isinstance(config_section, dict):
        raise TypeError(f"physics config section {config_key} must be a mapping")
    return config_section


def _load_optimization_config_section(config_key: str) -> dict:
    with OPTIMIZATION_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        optimization_config = json.load(config_file)
    if config_key not in optimization_config:
        raise ValueError(f"missing optimization config section {config_key}")
    config_section = optimization_config[config_key]
    if not isinstance(config_section, dict):
        raise TypeError(f"optimization config section {config_key} must be a mapping")
    return config_section


def _required_mapping(
    mapping: dict,
    key: str,
    context: str,
) -> dict:
    if key not in mapping:
        raise ValueError(f"{context} missing key {key}")
    value = mapping[key]
    if not isinstance(value, dict):
        raise TypeError(f"{context}.{key} must be a mapping")
    return value


def _required_registry_record(
    registry: Mapping[str, dict],
    species_name: str,
    registry_name: str,
) -> dict:
    if species_name not in registry:
        raise ValueError(f"{registry_name} registry missing {species_name}")
    record = registry[species_name]
    if not isinstance(record, dict):
        raise TypeError(
            f"{registry_name} registry entry {species_name} must be a mapping"
        )
    return record


def _required_ionic_compound_record(
    registry_source: MolecularPropertyDbRegistrySource,
    compound_name: str,
) -> dict:
    if compound_name in registry_source.salt_registry:
        return _required_registry_record(
            registry_source.salt_registry,
            compound_name,
            "salt",
        )
    if compound_name in registry_source.additive_registry:
        additive_record = _required_registry_record(
            registry_source.additive_registry,
            compound_name,
            "additive",
        )
        if _record_is_ionic_compound(additive_record):
            return additive_record
    raise ValueError(f"ionic compound registry missing {compound_name}")


def _required_neutral_compound_record(
    registry_source: MolecularPropertyDbRegistrySource,
    compound_name: str,
) -> dict:
    if compound_name in registry_source.solvent_registry:
        return _required_registry_record(
            registry_source.solvent_registry,
            compound_name,
            "solvent",
        )
    if compound_name in registry_source.additive_registry:
        additive_record = _required_registry_record(
            registry_source.additive_registry,
            compound_name,
            "additive",
        )
        if not _record_is_ionic_compound(additive_record):
            return additive_record
    raise ValueError(f"neutral compound registry missing {compound_name}")


def _positive_float_mapping(
    mapping: dict,
    context: str,
) -> Mapping[str, float]:
    parsed_mapping: dict[str, float] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} keys must be strings")
        parsed_mapping[key] = _positive_float(value, f"{context}.{key}")
    return parsed_mapping


def _nonnegative_float_mapping(
    mapping: dict,
    context: str,
) -> Mapping[str, float]:
    parsed_mapping: dict[str, float] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} keys must be strings")
        parsed_mapping[key] = _nonnegative_float(value, f"{context}.{key}")
    return parsed_mapping


def _required_list(
    mapping: dict,
    key: str,
    context: str,
) -> list:
    if key not in mapping:
        raise ValueError(f"{context} missing")
    value = mapping[key]
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return value


def _mapping_value(
    value,
    context: str,
) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    return value


def _required_nonempty_string_tuple(
    mapping: dict,
    key: str,
    context: str,
) -> tuple[str, ...]:
    values = _required_list(mapping, key, context)
    if not values:
        raise ValueError(f"{context} must be nonempty")
    parsed_values: list[str] = []
    for value_index, value in enumerate(values):
        value_context = f"{context}[{value_index}]"
        if not isinstance(value, str) or value == "":
            raise ValueError(f"{value_context} must be a nonempty string")
        parsed_values.append(value)
    duplicate_values = tuple(
        sorted(
            value
            for value in set(parsed_values)
            if parsed_values.count(value) > 1
        )
    )
    if duplicate_values:
        raise ValueError(f"{context} contains duplicates {duplicate_values}")
    return tuple(parsed_values)


def _required_positive_float_tuple(
    mapping: dict,
    key: str,
    context: str,
) -> tuple[float, ...]:
    values = _required_list(mapping, key, context)
    return _positive_float_tuple(tuple(values), context)


def _positive_float_tuple(
    values: tuple[float, ...],
    context: str,
) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{context} must be nonempty")
    parsed_values: list[float] = []
    for value_index, value in enumerate(values):
        parsed_values.append(
            _positive_float(value, f"{context}[{value_index}]")
        )
    return tuple(parsed_values)


def _required_nonnegative_int_tuple(
    mapping: dict,
    key: str,
    context: str,
) -> tuple[int, ...]:
    values = _required_list(mapping, key, context)
    parsed_values: list[int] = []
    for value_index, value in enumerate(values):
        value_context = f"{context}[{value_index}]"
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{value_context} must be a nonnegative integer")
        parsed_values.append(value)
    duplicate_values = tuple(
        sorted(
            value
            for value in set(parsed_values)
            if parsed_values.count(value) > 1
        )
    )
    if duplicate_values:
        raise ValueError(f"{context} contains duplicates {duplicate_values}")
    return tuple(parsed_values)


def _required_string(
    mapping: dict,
    key: str,
    context: str,
) -> str:
    if key not in mapping:
        raise ValueError(f"{context} missing")
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _required_int(
    mapping: dict,
    key: str,
    context: str,
) -> int:
    if key not in mapping:
        raise ValueError(f"{context} missing")
    value = mapping[key]
    if not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _required_positive_number(
    mapping: dict,
    key: str,
    context: str,
) -> float:
    if key not in mapping:
        raise ValueError(f"{context} missing")
    return _positive_float(mapping[key], context)


def _required_nonnegative_number(
    mapping: dict,
    key: str,
    context: str,
) -> float:
    if key not in mapping:
        raise ValueError(f"{context} missing")
    return _nonnegative_float(mapping[key], context)


def _optional_nonnegative_number(
    mapping: dict,
    key: str,
) -> float:
    if key not in mapping:
        return 0.0
    return _nonnegative_float(mapping[key], key)


def _optional_positive_number_or_reference(
    mapping: dict,
    key: str,
    reference_value: float,
) -> float:
    if key not in mapping:
        return _positive_float(reference_value, f"{key}.reference")
    return _positive_float(mapping[key], key)


def _required_positive_config_float(
    mapping: dict,
    key: str,
) -> float:
    if key not in mapping:
        raise ValueError(f"missing physics config {key}")
    return _positive_float(mapping[key], key)


def _required_nonnegative_config_float(
    mapping: dict,
    key: str,
) -> float:
    if key not in mapping:
        raise ValueError(f"missing physics config {key}")
    return _nonnegative_float(mapping[key], key)


def _required_finite_config_float(
    mapping: dict,
    key: str,
) -> float:
    if key not in mapping:
        raise ValueError(f"missing physics config {key}")
    return _finite_float(mapping[key], key)


def _required_positive_config_int(
    mapping: dict,
    key: str,
) -> int:
    if key not in mapping:
        raise ValueError(f"missing physics config {key}")
    value = mapping[key]
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"physics config {key} must be a positive integer")
    return value


def _required_nonnegative_config_int(
    mapping: dict,
    key: str,
) -> int:
    if key not in mapping:
        raise ValueError(f"missing physics config {key}")
    value = mapping[key]
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"physics config {key} must be a nonnegative integer")
    return value


def _required_config_bool(
    mapping: dict,
    key: str,
) -> bool:
    if key not in mapping:
        raise ValueError(f"missing physics config {key}")
    value = mapping[key]
    if not isinstance(value, bool):
        raise TypeError(f"physics config {key} must be a boolean")
    return value


def _required_config_string(
    mapping: dict,
    key: str,
) -> str:
    if key not in mapping:
        raise ValueError(f"missing physics config {key}")
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"physics config {key} must be a nonempty string")
    return value


def _primitive_fit_coordinate_bound_from_config(
    coordinate_bounds_mapping: dict,
    parameter_name: str,
) -> PrimitiveParameterTransform:
    if parameter_name not in coordinate_bounds_mapping:
        raise ValueError(f"missing coordinate bound for {parameter_name}")
    bound_values = coordinate_bounds_mapping[parameter_name]
    if not isinstance(bound_values, list):
        raise TypeError(f"coordinate bound for {parameter_name} must be a list")
    expected_bound_count = 2
    if len(bound_values) != expected_bound_count:
        raise ValueError(
            f"coordinate bound for {parameter_name} must contain two values"
        )
    lower_value = _primitive_bound_endpoint_from_config(
        parameter_name,
        bound_values[0],
        "lower_bound",
    )
    upper_value = _primitive_bound_endpoint_from_config(
        parameter_name,
        bound_values[1],
        "upper_bound",
    )
    if lower_value >= upper_value:
        raise ValueError(
            f"coordinate bound for {parameter_name} lower must be below upper"
        )
    return PrimitiveParameterTransform(
        name=parameter_name,
        transform=CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[
            parameter_name
        ],
        lower=lower_value,
        upper=upper_value,
    )


def _primitive_bound_endpoint_from_config(
    parameter_name: str,
    endpoint_value: float,
    endpoint_name: str,
) -> float:
    context = f"{parameter_name}.{endpoint_name}"
    transform_name = CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[
        parameter_name
    ]
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return math.log(_positive_float(endpoint_value, context))
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return _finite_float(endpoint_value, context)
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


def _required_primitive_parameter_config_float(
    mapping: dict,
    key: str,
) -> float:
    transform_name = CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[key]
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return _required_positive_config_float(mapping, key)
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return _required_finite_config_float(mapping, key)
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


def _finite_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{context} must be finite")
    return parsed_value


def _positive_float(value: float, context: str) -> float:
    parsed_value = _finite_float(value, context)
    if parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive")
    return parsed_value


def _nonnegative_float(value: float, context: str) -> float:
    parsed_value = _finite_float(value, context)
    if parsed_value < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    return parsed_value
