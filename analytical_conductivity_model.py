"""Single-file descriptor analytical conductivity model.

This file contains the complete descriptor analytical conductivity path:

    molecular species descriptors + primitive parameters
    -> topology-aware generic speciation
    -> center-resolved molecular transport centers
    -> reversible finite Markov-additive events
    -> Green-Kubo direct-minus-corrector conductivity.

The trajectory-derived projection model is intentionally separate because it
estimates (c, Q, d) from an observed trajectory rather than from descriptors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Mapping, Protocol, Sequence

import numpy as np

from constants import E_CHARGE, EPS_0, F, K_B, N_A, R, S_M_TO_MS_CM



# ---- molecular_descriptors.py ----

ROLE_CATION = "cation"
ROLE_ANION = "anion"
ROLE_SOLVENT = "solvent"
ROLE_ADDITIVE = "additive"
ROLE_NEUTRAL = "neutral"
SUPPORTED_MOLECULAR_SPECIES_ROLES = (
    ROLE_CATION,
    ROLE_ANION,
    ROLE_SOLVENT,
    ROLE_ADDITIVE,
    ROLE_NEUTRAL,
)


@dataclass(frozen=True)
class MolecularSpeciesInput:
    name: str
    role: str
    charge_number: int
    smiles: str
    xyz_coordinates: tuple[tuple[str, float, float, float], ...]
    property_overrides: Mapping[str, float]
    coordination_sites: tuple[str, ...]


@dataclass(frozen=True)
class MolecularSpeciesDescriptor:
    name: str
    role: str
    charge_number: int
    molecular_weight_g_mol: float
    hard_sphere_radius_A: float
    hydrodynamic_radius_A: float
    cavity_radius_A: float
    charge_cloud_radius_A: float
    molecular_volume_A3: float
    solvent_accessible_area_A2: float
    dipole_D: float
    quadrupole_D_A: float
    polarizability_A3: float
    donor_number: float
    acceptor_number: float
    hbond_donor_count: int
    hbond_acceptor_count: int
    epsilon_r_pure: float
    viscosity_cP_pure: float
    density_g_ml: float
    born_solvation_radius_A: float
    coordination_sites: tuple[str, ...]
    coordination_affinity_J_mol: float
    ligand_field_asymmetry: float


class MolecularDescriptorBackend(Protocol):
    def describe_species(
        self,
        species: MolecularSpeciesInput,
        temperature_K: float,
    ) -> MolecularSpeciesDescriptor:
        ...


class ProvidedPropertyDescriptorBackend:
    """Build descriptors only from user-supplied molecular property values."""

    def describe_species(
        self,
        species: MolecularSpeciesInput,
        temperature_K: float,
    ) -> MolecularSpeciesDescriptor:
        _validate_species_identity(species)
        _positive_float(temperature_K, "temperature_K")
        properties = species.property_overrides
        return MolecularSpeciesDescriptor(
            name=species.name,
            role=species.role,
            charge_number=species.charge_number,
            molecular_weight_g_mol=_required_positive_property(
                properties,
                "molecular_weight_g_mol",
                species.name,
            ),
            hard_sphere_radius_A=_required_positive_property(
                properties,
                "hard_sphere_radius_A",
                species.name,
            ),
            hydrodynamic_radius_A=_required_positive_property(
                properties,
                "hydrodynamic_radius_A",
                species.name,
            ),
            cavity_radius_A=_required_positive_property(
                properties,
                "cavity_radius_A",
                species.name,
            ),
            charge_cloud_radius_A=_required_positive_property(
                properties,
                "charge_cloud_radius_A",
                species.name,
            ),
            molecular_volume_A3=_required_positive_property(
                properties,
                "molecular_volume_A3",
                species.name,
            ),
            solvent_accessible_area_A2=_required_nonnegative_property(
                properties,
                "solvent_accessible_area_A2",
                species.name,
            ),
            dipole_D=_required_nonnegative_property(
                properties,
                "dipole_D",
                species.name,
            ),
            quadrupole_D_A=_required_nonnegative_property(
                properties,
                "quadrupole_D_A",
                species.name,
            ),
            polarizability_A3=_required_nonnegative_property(
                properties,
                "polarizability_A3",
                species.name,
            ),
            donor_number=_required_nonnegative_property(
                properties,
                "donor_number",
                species.name,
            ),
            acceptor_number=_required_nonnegative_property(
                properties,
                "acceptor_number",
                species.name,
            ),
            hbond_donor_count=_required_nonnegative_integer_property(
                properties,
                "hbond_donor_count",
                species.name,
            ),
            hbond_acceptor_count=_required_nonnegative_integer_property(
                properties,
                "hbond_acceptor_count",
                species.name,
            ),
            epsilon_r_pure=_required_positive_property(
                properties,
                "epsilon_r_pure",
                species.name,
            ),
            viscosity_cP_pure=_required_positive_property(
                properties,
                "viscosity_cP_pure",
                species.name,
            ),
            density_g_ml=_required_positive_property(
                properties,
                "density_g_ml",
                species.name,
            ),
            born_solvation_radius_A=_required_positive_property(
                properties,
                "born_solvation_radius_A",
                species.name,
            ),
            coordination_sites=tuple(species.coordination_sites),
            coordination_affinity_J_mol=_required_nonnegative_property(
                properties,
                "coordination_affinity_J_mol",
                species.name,
            ),
            ligand_field_asymmetry=_required_positive_property(
                properties,
                "ligand_field_asymmetry",
                species.name,
            ),
        )


def _validate_species_identity(species: MolecularSpeciesInput) -> None:
    if species.name == "":
        raise ValueError("species name must be nonempty")
    if species.role not in SUPPORTED_MOLECULAR_SPECIES_ROLES:
        raise ValueError(
            f"species {species.name} has unsupported role {species.role!r}"
        )
    if not isinstance(species.charge_number, int):
        raise TypeError(f"species {species.name} charge_number must be an integer")
    if species.role == ROLE_CATION and species.charge_number <= 0:
        raise ValueError(f"cation {species.name} must have positive charge_number")
    if species.role == ROLE_ANION and species.charge_number >= 0:
        raise ValueError(f"anion {species.name} must have negative charge_number")


def _required_positive_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    return _positive_float(
        _required_numeric_property(properties, key, species_name),
        f"{species_name}.{key}",
    )


def _required_nonnegative_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    return _nonnegative_float(
        _required_numeric_property(properties, key, species_name),
        f"{species_name}.{key}",
    )


def _required_nonnegative_integer_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> int:
    value = _required_nonnegative_property(properties, key, species_name)
    integer_value = int(value)
    if float(integer_value) != value:
        raise ValueError(f"{species_name}.{key} must be an integer-valued number")
    return integer_value


def _required_numeric_property(
    properties: Mapping[str, float],
    key: str,
    species_name: str,
) -> float:
    if key not in properties:
        raise ValueError(f"species {species_name} missing molecular descriptor {key}")
    value = properties[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"species {species_name} descriptor {key} must be numeric")
    return float(value)


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


# ---- molecular_primitive_parameters.py ----

PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE = "log_positive"
PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED = "identity_signed"


@dataclass(frozen=True)
class ConductivityPrimitiveParameterSet:
    coulomb_scale: float
    desolvation_scale: float
    coordination_scale: float
    steric_free_energy_scale: float
    cluster_entropy_penalty_scale: float
    association_crowding_stabilization_scale: float
    association_crowding_ionic_strength_exponent: float
    association_crowding_charge_density_exponent: float
    activity_debye_scale: float
    activity_size_scale: float
    activity_hard_sphere_scale: float
    cluster_activity_scale: float
    pair_logK_offset: float
    solvent_separated_pair_logK_offset: float
    contact_pair_logK_offset: float
    positive_charged_triplet_logK_offset: float
    negative_charged_triplet_logK_offset: float
    neutral_cluster_logK_offset: float
    higher_charged_cluster_logK_offset: float
    cluster_order_logK_slope: float
    cluster_charge_magnitude_logK_slope: float
    cluster_hydrodynamic_radius_scale: float
    hydrodynamic_radius_scale_positive_ion: float
    hydrodynamic_radius_scale_negative_ion: float
    hydrodynamic_radius_scale_cluster: float
    shape_friction_exponent: float
    free_volume_exponent: float
    dielectric_mobility_exponent: float
    solvation_mobility_exponent: float
    additive_shape_solvation_mobility_exponent: float
    positive_ion_charge_density_mobility_exponent: float
    negative_ion_charge_density_mobility_exponent: float
    positive_ion_counteranion_charge_cloud_mobility_exponent: float
    negative_ion_charge_cloud_mobility_exponent: float
    negative_ion_intrinsic_dielectric_drag_mobility_exponent: float
    negative_ion_shape_delocalization_mobility_exponent: float
    positive_ion_anion_disorder_mobility_exponent: float
    negative_ion_anion_disorder_mobility_exponent: float
    local_obstruction_strength: float
    local_obstruction_free_volume_exponent: float
    local_obstruction_ionic_strength_exponent: float
    local_obstruction_additive_solvation_exponent: float
    local_obstruction_size_exponent: float
    local_obstruction_charge_density_exponent: float
    local_obstruction_solvation_exponent: float
    atmosphere_ep_scale: float
    atmosphere_rel_scale: float
    charge_cloud_radius_scale: float
    cross_relaxation_scale: float
    jump_length_scale: float
    atmosphere_capture_scale: float
    atmosphere_exit_scale: float
    association_conversion_rate_scale: float
    orientation_relaxation_rate_scale: float


CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES = tuple(
    field.name for field in fields(ConductivityPrimitiveParameterSet)
)


@dataclass(frozen=True)
class ConductivityPrimitivePositiveScales:
    coulomb_scale: float
    desolvation_scale: float
    coordination_scale: float
    steric_free_energy_scale: float
    cluster_entropy_penalty_scale: float
    association_crowding_stabilization_scale: float
    association_crowding_ionic_strength_exponent: float
    association_crowding_charge_density_exponent: float
    activity_debye_scale: float
    activity_size_scale: float
    activity_hard_sphere_scale: float
    cluster_activity_scale: float
    cluster_hydrodynamic_radius_scale: float
    hydrodynamic_radius_scale_positive_ion: float
    hydrodynamic_radius_scale_negative_ion: float
    hydrodynamic_radius_scale_cluster: float
    shape_friction_exponent: float
    free_volume_exponent: float
    dielectric_mobility_exponent: float
    solvation_mobility_exponent: float
    additive_shape_solvation_mobility_exponent: float
    positive_ion_charge_density_mobility_exponent: float
    negative_ion_charge_density_mobility_exponent: float
    positive_ion_counteranion_charge_cloud_mobility_exponent: float
    negative_ion_charge_cloud_mobility_exponent: float
    negative_ion_intrinsic_dielectric_drag_mobility_exponent: float
    negative_ion_shape_delocalization_mobility_exponent: float
    positive_ion_anion_disorder_mobility_exponent: float
    negative_ion_anion_disorder_mobility_exponent: float
    local_obstruction_strength: float
    local_obstruction_free_volume_exponent: float
    local_obstruction_ionic_strength_exponent: float
    local_obstruction_additive_solvation_exponent: float
    local_obstruction_size_exponent: float
    local_obstruction_charge_density_exponent: float
    local_obstruction_solvation_exponent: float
    atmosphere_ep_scale: float
    atmosphere_rel_scale: float
    charge_cloud_radius_scale: float
    cross_relaxation_scale: float
    jump_length_scale: float
    atmosphere_capture_scale: float
    atmosphere_exit_scale: float
    association_conversion_rate_scale: float
    orientation_relaxation_rate_scale: float


@dataclass(frozen=True)
class ConductivityPrimitiveSignedOffsets:
    pair_logK_offset: float
    solvent_separated_pair_logK_offset: float
    contact_pair_logK_offset: float
    positive_charged_triplet_logK_offset: float
    negative_charged_triplet_logK_offset: float
    neutral_cluster_logK_offset: float
    higher_charged_cluster_logK_offset: float
    cluster_order_logK_slope: float
    cluster_charge_magnitude_logK_slope: float


CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES = tuple(
    field.name for field in fields(ConductivityPrimitiveSignedOffsets)
)


CONDUCTIVITY_PRIMITIVE_POSITIVE_PARAMETER_FIELD_NAMES = tuple(
    field_name for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    if field_name not in CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES
)


CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME = {
    **{
        field_name: PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE
        for field_name in CONDUCTIVITY_PRIMITIVE_POSITIVE_PARAMETER_FIELD_NAMES
    },
    **{
        field_name: PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED
        for field_name in CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES
    },
}


def conductivity_primitive_parameters_to_coordinate_values(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> tuple[float, ...]:
    validate_conductivity_primitive_parameters(primitive_parameters)
    return tuple(
        _parameter_coordinate_value(primitive_parameters, field_name)
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    )


def conductivity_primitive_parameter_coordinate_values_for_names(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    parameter_names: tuple[str, ...],
) -> tuple[float, ...]:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_parameter_name_tuple(parameter_names)
    return tuple(
        _parameter_coordinate_value(primitive_parameters, parameter_name)
        for parameter_name in parameter_names
    )


def conductivity_primitive_parameters_from_coordinate_values(
    coordinate_values: tuple[float, ...],
) -> ConductivityPrimitiveParameterSet:
    field_names = CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    if len(coordinate_values) != len(field_names):
        raise ValueError(
            "coordinate_values length must match ConductivityPrimitiveParameterSet field count"
        )
    parameter_values: dict[str, float] = {}
    for field_name, coordinate_value in zip(field_names, coordinate_values):
        parameter_values[field_name] = _parameter_value_from_coordinate(
            field_name,
            coordinate_value,
        )
    return _conductivity_primitive_parameters_from_validated_values(
        parameter_values
    )


def conductivity_primitive_parameters_with_coordinate_updates(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    parameter_names: tuple[str, ...],
    coordinate_values: tuple[float, ...],
) -> ConductivityPrimitiveParameterSet:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_parameter_name_tuple(parameter_names)
    if len(parameter_names) != len(coordinate_values):
        raise ValueError("parameter_names and coordinate_values must have equal length")
    parameter_values = {
        field_name: _parameter_value(primitive_parameters, field_name)
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    }
    for parameter_name, coordinate_value in zip(parameter_names, coordinate_values):
        parameter_values[parameter_name] = _parameter_value_from_coordinate(
            parameter_name,
            coordinate_value,
        )
    return _conductivity_primitive_parameters_from_validated_values(
        parameter_values
    )


def conductivity_primitive_parameters_to_mapping(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> dict[str, float]:
    validate_conductivity_primitive_parameters(primitive_parameters)
    return {
        field_name: _parameter_value(primitive_parameters, field_name)
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    }


def conductivity_primitive_parameters_from_mapping(
    parameter_mapping: Mapping[str, float],
) -> ConductivityPrimitiveParameterSet:
    missing_parameter_names = tuple(
        field_name for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        if field_name not in parameter_mapping
    )
    if missing_parameter_names:
        raise ValueError(
            "missing conductivity primitive parameters "
            f"{missing_parameter_names}"
        )
    unknown_parameter_names = tuple(
        sorted(
            field_name for field_name in parameter_mapping
            if field_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        )
    )
    if unknown_parameter_names:
        raise ValueError(
            "unknown conductivity primitive parameters "
            f"{unknown_parameter_names}"
        )
    parameter_values = {
        field_name: _validated_parameter_value(
            field_name,
            parameter_mapping[field_name],
        )
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    }
    return _conductivity_primitive_parameters_from_validated_values(
        parameter_values
    )


def _conductivity_primitive_parameters_from_validated_values(
    parameter_values: Mapping[str, float],
) -> ConductivityPrimitiveParameterSet:
    primitive_parameters = ConductivityPrimitiveParameterSet(
        **{
            field_name: _validated_parameter_value(
                field_name,
                parameter_values[field_name],
            )
            for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
        }
    )
    validate_conductivity_primitive_parameters(primitive_parameters)
    return primitive_parameters


def validate_conductivity_primitive_parameters(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> None:
    for primitive_parameter_field in fields(ConductivityPrimitiveParameterSet):
        field_name = primitive_parameter_field.name
        _validated_parameter_value(
            field_name,
            getattr(primitive_parameters, field_name),
        )


def _parameter_value(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    field_name: str,
) -> float:
    if field_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        raise ValueError(f"unknown conductivity primitive parameter {field_name}")
    return _validated_parameter_value(field_name, getattr(primitive_parameters, field_name))


def _parameter_coordinate_value(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    field_name: str,
) -> float:
    parameter_value = _parameter_value(primitive_parameters, field_name)
    transform_name = _parameter_transform_name(field_name)
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return math.log(parameter_value)
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return parameter_value
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


def _parameter_value_from_coordinate(
    field_name: str,
    coordinate_value: float,
) -> float:
    parsed_coordinate_value = _finite_float(
        coordinate_value,
        f"primitive_parameters.{field_name}.coordinate",
    )
    transform_name = _parameter_transform_name(field_name)
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE:
        return _positive_float(
            math.exp(parsed_coordinate_value),
            f"primitive_parameters.{field_name}",
        )
    if transform_name == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED:
        return parsed_coordinate_value
    raise ValueError(f"unknown primitive parameter transform {transform_name}")


def _parameter_transform_name(field_name: str) -> str:
    if field_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME:
        raise ValueError(f"unknown conductivity primitive parameter {field_name}")
    return CONDUCTIVITY_PRIMITIVE_PARAMETER_TRANSFORM_BY_NAME[field_name]


def _validated_parameter_value(field_name: str, value: float) -> float:
    context = f"primitive_parameters.{field_name}"
    if field_name in CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES:
        return _finite_float(value, context)
    return _positive_float(value, context)


def _validate_parameter_name_tuple(parameter_names: tuple[str, ...]) -> None:
    if not parameter_names:
        raise ValueError("parameter_names must be nonempty")
    seen_parameter_names: set[str] = set()
    for parameter_name in parameter_names:
        if parameter_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
            raise ValueError(f"unknown conductivity primitive parameter {parameter_name}")
        if parameter_name in seen_parameter_names:
            raise ValueError(f"duplicate conductivity primitive parameter {parameter_name}")
        seen_parameter_names.add(parameter_name)


def _positive_float(value: float, context: str) -> float:
    parsed_value = _finite_float(value, context)
    if parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _finite_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{context} must be finite")
    return parsed_value


# ---- finite_mori_conductivity.py ----

CARTESIAN_AXIS_COUNT = 3.0  # Analytical isotropic average over x, y, z current axes.
MORI_NUMERICAL_TOLERANCE = math.sqrt(np.finfo(float).eps)
TRAPEZOID_ENDPOINT_WEIGHT = 0.5  # Analytical trapezoid-rule endpoint weight.


@dataclass(frozen=True)
class ProjectedMoriConductivityInput:
    direct_energy_matrix: np.ndarray
    memory_self_energy_matrix: np.ndarray
    current_coupling_matrix: np.ndarray
    beta_over_volume: float


@dataclass(frozen=True)
class ProjectedMoriConductivityResult:
    sigma_S_m: float
    sigma_mS_cm: float
    axis_conductivity_S_m: tuple[float, float, float]
    quadratic_form_by_axis: tuple[float, float, float]
    energy_eigenvalues: tuple[float, ...]
    effective_energy_matrix: np.ndarray


@dataclass(frozen=True)
class MoriOracleClosureComparison:
    sigma_oracle_S_m: float
    sigma_oracle_mS_cm: float
    sigma_trajectory_mS_cm: float
    closure_gap_mS_cm: float
    tolerance_mS_cm: float
    passes_tolerance: bool


@dataclass(frozen=True)
class TrajectoryProjectedMoriConstruction:
    mori_input: ProjectedMoriConductivityInput
    centered_current_time_series: np.ndarray
    whitened_basis_time_series: np.ndarray
    retained_basis_time_series: np.ndarray
    projected_current_time_series: np.ndarray
    raw_green_kubo_sigma_mS_cm: float
    projected_green_kubo_sigma_mS_cm: float
    raw_green_kubo_axis_integrals: tuple[float, float, float]
    projected_green_kubo_axis_integrals: tuple[float, float, float]
    maximum_lag_steps_used: int
    retained_gram_eigenvalues: tuple[float, ...]
    retained_zero_frequency_covariance_eigenvalues: tuple[float, ...]
    all_zero_frequency_covariance_eigenvalues: tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryMoriClosureAuditInput:
    sample_id: str
    basis_feature_time_series: np.ndarray
    current_time_series: np.ndarray
    time_step_s: float
    maximum_lag_steps: int
    beta_over_volume: float
    sigma_property_db_mS_cm: float
    sigma_recipe_mori_mS_cm: float
    gram_relative_tolerance: float = 1.0e-10
    zero_frequency_relative_tolerance: float = 1.0e-8


@dataclass(frozen=True)
class TrajectoryMoriClosureAuditRow:
    sample_id: str
    sigma_property_db_mS_cm: float
    sigma_raw_gk_mS_cm: float
    sigma_projected_gk_mS_cm: float
    sigma_mori_oracle_mS_cm: float
    sigma_recipe_mori_mS_cm: float
    projection_gap_mS_cm: float
    recipe_gap_mS_cm: float
    label_gap_mS_cm: float
    gram_rank: int
    kz_rank: int
    min_kz_eigenvalue: float
    dropped_gram_modes: int
    dropped_kz_modes: int


def compute_projected_mori_conductivity(
    mori_input: ProjectedMoriConductivityInput,
) -> ProjectedMoriConductivityResult:
    """Evaluate beta/(3V) sum_axis h_axis^T A_M^# h_axis."""

    direct_energy_matrix = _validated_square_matrix(
        mori_input.direct_energy_matrix,
        "direct_energy_matrix",
    )
    memory_self_energy_matrix = _validated_square_matrix(
        mori_input.memory_self_energy_matrix,
        "memory_self_energy_matrix",
    )
    if direct_energy_matrix.shape != memory_self_energy_matrix.shape:
        raise ValueError(
            "direct_energy_matrix and memory_self_energy_matrix must have the same shape"
        )
    _validate_symmetric_matrix(direct_energy_matrix, "direct_energy_matrix")
    _validate_symmetric_matrix(memory_self_energy_matrix, "memory_self_energy_matrix")
    _validate_positive_semidefinite_matrix(direct_energy_matrix, "direct_energy_matrix")
    _validate_positive_semidefinite_matrix(memory_self_energy_matrix, "memory_self_energy_matrix")

    current_coupling_matrix = _validated_current_coupling_matrix(
        mori_input.current_coupling_matrix,
        direct_energy_matrix.shape[0],
    )
    _assert_positive_finite(mori_input.beta_over_volume, "beta_over_volume")

    effective_energy_matrix = direct_energy_matrix + memory_self_energy_matrix
    _validate_symmetric_matrix(effective_energy_matrix, "effective_energy_matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(effective_energy_matrix)
    _validate_eigenvalues_are_psd(eigenvalues, "effective_energy_matrix")

    quadratic_forms: list[float] = []
    axis_conductivities: list[float] = []
    for axis_index in range(current_coupling_matrix.shape[0]):
        axis_current_coupling = current_coupling_matrix[axis_index, :]
        quadratic_form = _projected_quadratic_form(
            eigenvalues,
            eigenvectors,
            axis_current_coupling,
            f"axis_{axis_index}",
        )
        quadratic_forms.append(quadratic_form)
        axis_conductivities.append(
            mori_input.beta_over_volume * quadratic_form / CARTESIAN_AXIS_COUNT
        )

    sigma_S_m = math.fsum(axis_conductivities)
    _assert_nonnegative_finite(sigma_S_m, "sigma_S_m")
    return ProjectedMoriConductivityResult(
        sigma_S_m=sigma_S_m,
        sigma_mS_cm=sigma_S_m * S_M_TO_MS_CM,
        axis_conductivity_S_m=(
            axis_conductivities[0],
            axis_conductivities[1],
            axis_conductivities[2],
        ),
        quadratic_form_by_axis=(
            quadratic_forms[0],
            quadratic_forms[1],
            quadratic_forms[2],
        ),
        energy_eigenvalues=tuple(float(eigenvalue) for eigenvalue in eigenvalues),
        effective_energy_matrix=effective_energy_matrix.copy(),
    )


def compare_mori_oracle_to_trajectory(
    mori_result: ProjectedMoriConductivityResult,
    trajectory_sigma_mS_cm: float,
    tolerance_mS_cm: float,
) -> MoriOracleClosureComparison:
    """Compare a projected Mori oracle result to trajectory Green-Kubo conductivity."""

    _assert_nonnegative_finite(mori_result.sigma_mS_cm, "mori_result.sigma_mS_cm")
    _assert_nonnegative_finite(trajectory_sigma_mS_cm, "trajectory_sigma_mS_cm")
    _assert_positive_finite(tolerance_mS_cm, "tolerance_mS_cm")
    closure_gap_mS_cm = float(abs(mori_result.sigma_mS_cm - trajectory_sigma_mS_cm))
    return MoriOracleClosureComparison(
        sigma_oracle_S_m=mori_result.sigma_S_m,
        sigma_oracle_mS_cm=mori_result.sigma_mS_cm,
        sigma_trajectory_mS_cm=trajectory_sigma_mS_cm,
        closure_gap_mS_cm=closure_gap_mS_cm,
        tolerance_mS_cm=tolerance_mS_cm,
        passes_tolerance=bool(closure_gap_mS_cm < tolerance_mS_cm),
    )


def build_trajectory_projected_mori_input(
    basis_feature_time_series: np.ndarray,
    current_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
    beta_over_volume: float,
    gram_relative_tolerance: float = 1.0e-10,
    zero_frequency_relative_tolerance: float = 1.0e-8,
) -> TrajectoryProjectedMoriConstruction:
    """Build projected Mori matrices from one centered trajectory current process."""

    feature_array = _validated_feature_time_series(basis_feature_time_series)
    current_array = _validated_current_time_series(current_time_series, "current_time_series")
    if feature_array.shape[0] != current_array.shape[0]:
        raise ValueError(
            "basis_feature_time_series and current_time_series must have the same frame count"
        )
    _assert_positive_finite(time_step_s, "time_step_s")
    _assert_positive_finite(beta_over_volume, "beta_over_volume")
    _assert_positive_finite(gram_relative_tolerance, "gram_relative_tolerance")
    _assert_positive_finite(
        zero_frequency_relative_tolerance,
        "zero_frequency_relative_tolerance",
    )
    retained_maximum_lag_steps = _validated_maximum_lag_steps(
        maximum_lag_steps,
        current_array.shape[0],
    )

    centered_features = feature_array - np.mean(feature_array, axis=0, keepdims=True)
    centered_current = current_array - np.mean(current_array, axis=0, keepdims=True)

    frame_count = centered_features.shape[0]
    gram_matrix = _symmetrized_matrix(centered_features.T @ centered_features / frame_count)
    retained_gram_eigenvalues, retained_gram_eigenvectors, _ = _eigh_retained_psd(
        gram_matrix,
        gram_relative_tolerance,
        "trajectory feature Gram matrix",
    )
    if retained_gram_eigenvalues.size == 0:
        raise ValueError("all trajectory basis features are null under the Gram matrix")

    inverse_sqrt_gram_eigenvalues = np.diag(1.0 / np.sqrt(retained_gram_eigenvalues))
    whitening_matrix = retained_gram_eigenvectors @ inverse_sqrt_gram_eigenvalues
    whitened_basis_time_series = centered_features @ whitening_matrix

    current_coupling_matrix = centered_current.T @ whitened_basis_time_series / frame_count
    integrated_basis_covariance = _integrated_symmetrized_basis_covariance(
        whitened_basis_time_series,
        time_step_s,
        retained_maximum_lag_steps,
    )
    (
        retained_zero_frequency_covariance_eigenvalues,
        retained_zero_frequency_covariance_eigenvectors,
        all_zero_frequency_covariance_eigenvalues,
    ) = _eigh_retained_psd(
        integrated_basis_covariance,
        zero_frequency_relative_tolerance,
        "integrated projected covariance",
    )
    if retained_zero_frequency_covariance_eigenvalues.size == 0:
        raise ValueError(
            "projected basis has no positive zero-frequency covariance modes"
        )

    memory_self_energy_matrix = np.diag(
        1.0 / retained_zero_frequency_covariance_eigenvalues
    )
    retained_current_coupling_matrix = (
        current_coupling_matrix @ retained_zero_frequency_covariance_eigenvectors
    )
    direct_energy_matrix = np.zeros_like(memory_self_energy_matrix)

    retained_basis_time_series = (
        whitened_basis_time_series @ retained_zero_frequency_covariance_eigenvectors
    )
    projected_current_time_series = retained_basis_time_series @ retained_current_coupling_matrix.T
    raw_axis_integrals = green_kubo_axis_integrals_from_current_time_series(
        centered_current,
        time_step_s,
        retained_maximum_lag_steps,
        center_current=False,
    )
    projected_axis_integrals = green_kubo_axis_integrals_from_current_time_series(
        projected_current_time_series,
        time_step_s,
        retained_maximum_lag_steps,
        center_current=True,
    )
    raw_green_kubo_sigma_mS_cm = _green_kubo_sigma_mS_cm_from_axis_integrals(
        raw_axis_integrals,
        beta_over_volume,
    )
    projected_green_kubo_sigma_mS_cm = _green_kubo_sigma_mS_cm_from_axis_integrals(
        projected_axis_integrals,
        beta_over_volume,
    )

    return TrajectoryProjectedMoriConstruction(
        mori_input=ProjectedMoriConductivityInput(
            direct_energy_matrix=direct_energy_matrix,
            memory_self_energy_matrix=memory_self_energy_matrix,
            current_coupling_matrix=retained_current_coupling_matrix,
            beta_over_volume=beta_over_volume,
        ),
        centered_current_time_series=centered_current.copy(),
        whitened_basis_time_series=whitened_basis_time_series.copy(),
        retained_basis_time_series=retained_basis_time_series.copy(),
        projected_current_time_series=projected_current_time_series.copy(),
        raw_green_kubo_sigma_mS_cm=raw_green_kubo_sigma_mS_cm,
        projected_green_kubo_sigma_mS_cm=projected_green_kubo_sigma_mS_cm,
        raw_green_kubo_axis_integrals=raw_axis_integrals,
        projected_green_kubo_axis_integrals=projected_axis_integrals,
        maximum_lag_steps_used=retained_maximum_lag_steps,
        retained_gram_eigenvalues=tuple(
            float(eigenvalue) for eigenvalue in retained_gram_eigenvalues
        ),
        retained_zero_frequency_covariance_eigenvalues=tuple(
            float(eigenvalue)
            for eigenvalue in retained_zero_frequency_covariance_eigenvalues
        ),
        all_zero_frequency_covariance_eigenvalues=tuple(
            float(eigenvalue) for eigenvalue in all_zero_frequency_covariance_eigenvalues
        ),
    )


def build_trajectory_mori_closure_audit_row(
    closure_input: TrajectoryMoriClosureAuditInput,
) -> TrajectoryMoriClosureAuditRow:
    """Build the trajectory/property/recipe Mori closure decomposition row."""

    validated_sample_id = _validated_sample_id(closure_input.sample_id)
    feature_array = _validated_feature_time_series(
        closure_input.basis_feature_time_series
    )
    _assert_nonnegative_finite(
        closure_input.sigma_property_db_mS_cm,
        "sigma_property_db_mS_cm",
    )
    _assert_nonnegative_finite(
        closure_input.sigma_recipe_mori_mS_cm,
        "sigma_recipe_mori_mS_cm",
    )

    construction = build_trajectory_projected_mori_input(
        basis_feature_time_series=feature_array,
        current_time_series=closure_input.current_time_series,
        time_step_s=closure_input.time_step_s,
        maximum_lag_steps=closure_input.maximum_lag_steps,
        beta_over_volume=closure_input.beta_over_volume,
        gram_relative_tolerance=closure_input.gram_relative_tolerance,
        zero_frequency_relative_tolerance=(
            closure_input.zero_frequency_relative_tolerance
        ),
    )
    mori_oracle_result = compute_projected_mori_conductivity(
        construction.mori_input
    )

    raw_feature_count = int(feature_array.shape[1])
    gram_rank = len(construction.retained_gram_eigenvalues)
    kz_rank = len(construction.retained_zero_frequency_covariance_eigenvalues)
    min_kz_eigenvalue = float(
        min(construction.all_zero_frequency_covariance_eigenvalues)
    )
    dropped_gram_modes = raw_feature_count - gram_rank
    dropped_kz_modes = gram_rank - kz_rank
    if dropped_gram_modes < 0:
        raise ValueError("dropped_gram_modes cannot be negative")
    if dropped_kz_modes < 0:
        raise ValueError("dropped_kz_modes cannot be negative")

    return TrajectoryMoriClosureAuditRow(
        sample_id=validated_sample_id,
        sigma_property_db_mS_cm=float(closure_input.sigma_property_db_mS_cm),
        sigma_raw_gk_mS_cm=construction.raw_green_kubo_sigma_mS_cm,
        sigma_projected_gk_mS_cm=construction.projected_green_kubo_sigma_mS_cm,
        sigma_mori_oracle_mS_cm=mori_oracle_result.sigma_mS_cm,
        sigma_recipe_mori_mS_cm=float(closure_input.sigma_recipe_mori_mS_cm),
        projection_gap_mS_cm=(
            construction.projected_green_kubo_sigma_mS_cm
            - construction.raw_green_kubo_sigma_mS_cm
        ),
        recipe_gap_mS_cm=(
            float(closure_input.sigma_recipe_mori_mS_cm)
            - mori_oracle_result.sigma_mS_cm
        ),
        label_gap_mS_cm=(
            float(closure_input.sigma_property_db_mS_cm)
            - construction.raw_green_kubo_sigma_mS_cm
        ),
        gram_rank=gram_rank,
        kz_rank=kz_rank,
        min_kz_eigenvalue=min_kz_eigenvalue,
        dropped_gram_modes=dropped_gram_modes,
        dropped_kz_modes=dropped_kz_modes,
    )


def green_kubo_axis_integrals_from_current_time_series(
    current_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
    center_current: bool,
) -> tuple[float, float, float]:
    """Compute one-sided trapezoidal current autocorrelation integrals by axis."""

    current_array = _validated_current_time_series(current_time_series, "current_time_series")
    _assert_positive_finite(time_step_s, "time_step_s")
    retained_maximum_lag_steps = _validated_maximum_lag_steps(
        maximum_lag_steps,
        current_array.shape[0],
    )
    if center_current:
        centered_current = current_array - np.mean(current_array, axis=0, keepdims=True)
    else:
        centered_current = current_array

    axis_integrals = np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float)
    frame_count = centered_current.shape[0]
    for lag_steps in range(retained_maximum_lag_steps + 1):
        lagged_covariance_by_axis = np.sum(
            centered_current[lag_steps:, :] * centered_current[: frame_count - lag_steps, :],
            axis=0,
        ) / (frame_count - lag_steps)
        lag_weight = _trapezoid_lag_weight(lag_steps, retained_maximum_lag_steps)
        axis_integrals += lag_weight * lagged_covariance_by_axis

    axis_integrals = axis_integrals * time_step_s
    if not np.all(np.isfinite(axis_integrals)):
        raise ValueError("Green-Kubo axis integrals contain non-finite values")
    return (
        float(axis_integrals[0]),
        float(axis_integrals[1]),
        float(axis_integrals[2]),
    )


def _projected_quadratic_form(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    current_coupling: np.ndarray,
    context: str,
) -> float:
    projected_current = eigenvectors.T @ current_coupling
    positive_mode_mask = _positive_energy_mode_mask(eigenvalues)
    null_mode_projection = projected_current[~positive_mode_mask]
    null_projection_norm = float(np.linalg.norm(null_mode_projection))
    current_norm = float(np.linalg.norm(current_coupling))
    allowed_null_projection = MORI_NUMERICAL_TOLERANCE * max(
        current_norm,
        MORI_NUMERICAL_TOLERANCE,
    )
    if null_projection_norm > allowed_null_projection:
        raise ValueError(
            f"{context} current_coupling projects onto a null energy mode; "
            "zero-frequency conductivity is not finite in this projected basis"
        )
    if not np.any(positive_mode_mask):
        return 0.0
    positive_projected_current = projected_current[positive_mode_mask]
    positive_eigenvalues = eigenvalues[positive_mode_mask]
    quadratic_form = float(
        math.fsum(
            float(projected_value * projected_value / eigenvalue)
            for projected_value, eigenvalue in zip(
                positive_projected_current,
                positive_eigenvalues,
            )
        )
    )
    _assert_nonnegative_finite(quadratic_form, f"{context}.quadratic_form")
    return quadratic_form


def _validated_feature_time_series(feature_time_series: np.ndarray) -> np.ndarray:
    feature_array = np.asarray(feature_time_series, dtype=float)
    if feature_array.ndim != 2:
        raise ValueError("basis_feature_time_series must have shape (n_frames, n_basis_raw)")
    if feature_array.shape[0] < 2:
        raise ValueError("basis_feature_time_series must contain at least two frames")
    if feature_array.shape[1] == 0:
        raise ValueError("basis_feature_time_series must contain at least one feature")
    if not np.all(np.isfinite(feature_array)):
        raise ValueError("basis_feature_time_series contains non-finite values")
    return feature_array


def _validated_current_time_series(
    current_time_series: np.ndarray,
    name: str,
) -> np.ndarray:
    current_array = np.asarray(current_time_series, dtype=float)
    if current_array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_frames, 3)")
    expected_axis_count = int(CARTESIAN_AXIS_COUNT)
    if current_array.shape[1] != expected_axis_count:
        raise ValueError(f"{name} must have shape (n_frames, 3)")
    if current_array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two frames")
    if not np.all(np.isfinite(current_array)):
        raise ValueError(f"{name} contains non-finite values")
    return current_array


def _validated_maximum_lag_steps(
    maximum_lag_steps: int,
    frame_count: int,
) -> int:
    if not isinstance(maximum_lag_steps, int):
        raise TypeError("maximum_lag_steps must be an integer")
    if maximum_lag_steps < 0:
        raise ValueError("maximum_lag_steps must be nonnegative")
    return min(maximum_lag_steps, frame_count - 1)


def _symmetrized_matrix(matrix: np.ndarray) -> np.ndarray:
    return TRAPEZOID_ENDPOINT_WEIGHT * (matrix + matrix.T)


def _eigh_retained_psd(
    matrix: np.ndarray,
    relative_tolerance: float,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symmetrized_matrix = _symmetrized_matrix(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized_matrix)
    eigenvalue_scale = _eigenvalue_scale(eigenvalues)
    allowed_negative_eigenvalue = relative_tolerance * eigenvalue_scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(
            f"{context} has significant negative eigenvalues; "
            f"minimum={minimum_eigenvalue}, allowed={allowed_negative_eigenvalue}"
        )
    retained_mode_mask = eigenvalues > relative_tolerance * eigenvalue_scale
    return (
        eigenvalues[retained_mode_mask],
        eigenvectors[:, retained_mode_mask],
        eigenvalues,
    )


def _integrated_symmetrized_basis_covariance(
    whitened_basis_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
) -> np.ndarray:
    basis_dimension = whitened_basis_time_series.shape[1]
    frame_count = whitened_basis_time_series.shape[0]
    integrated_covariance = np.zeros((basis_dimension, basis_dimension), dtype=float)
    for lag_steps in range(maximum_lag_steps + 1):
        lagged_covariance = (
            whitened_basis_time_series[lag_steps:, :].T
            @ whitened_basis_time_series[: frame_count - lag_steps, :]
            / (frame_count - lag_steps)
        )
        lag_weight = _trapezoid_lag_weight(lag_steps, maximum_lag_steps)
        integrated_covariance += lag_weight * _symmetrized_matrix(lagged_covariance)
    return _symmetrized_matrix(integrated_covariance * time_step_s)


def _trapezoid_lag_weight(
    lag_steps: int,
    maximum_lag_steps: int,
) -> float:
    if lag_steps == 0 or lag_steps == maximum_lag_steps:
        return TRAPEZOID_ENDPOINT_WEIGHT
    return 1.0


def _green_kubo_sigma_mS_cm_from_axis_integrals(
    axis_integrals: tuple[float, float, float],
    beta_over_volume: float,
) -> float:
    sigma_S_m = beta_over_volume * math.fsum(axis_integrals) / CARTESIAN_AXIS_COUNT
    _assert_nonnegative_finite(sigma_S_m, "green_kubo_sigma_S_m")
    return sigma_S_m * S_M_TO_MS_CM


def _validated_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix_array.shape}")
    if matrix_array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one basis function")
    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix_array


def _validated_current_coupling_matrix(
    current_coupling_matrix: np.ndarray,
    basis_size: int,
) -> np.ndarray:
    current_coupling_array = np.asarray(current_coupling_matrix, dtype=float)
    if current_coupling_array.ndim != 2:
        raise ValueError("current_coupling_matrix must be a two-dimensional matrix")
    expected_shape = (int(CARTESIAN_AXIS_COUNT), basis_size)
    if current_coupling_array.shape != expected_shape:
        raise ValueError(
            "current_coupling_matrix must have shape "
            f"{expected_shape}, got {current_coupling_array.shape}"
        )
    if not np.all(np.isfinite(current_coupling_array)):
        raise ValueError("current_coupling_matrix contains non-finite values")
    return current_coupling_array


def _validate_symmetric_matrix(matrix: np.ndarray, name: str) -> None:
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=MORI_NUMERICAL_TOLERANCE,
        atol=MORI_NUMERICAL_TOLERANCE,
    ):
        raise ValueError(f"{name} must be symmetric")


def _validate_positive_semidefinite_matrix(matrix: np.ndarray, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    _validate_eigenvalues_are_psd(eigenvalues, name)


def _validate_eigenvalues_are_psd(eigenvalues: np.ndarray, name: str) -> None:
    minimum_eigenvalue = float(np.min(eigenvalues))
    allowed_negative_eigenvalue = MORI_NUMERICAL_TOLERANCE * _eigenvalue_scale(eigenvalues)
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(
            f"{name} must be positive semidefinite; minimum eigenvalue is "
            f"{minimum_eigenvalue}"
        )


def _positive_energy_mode_mask(eigenvalues: np.ndarray) -> np.ndarray:
    positive_threshold = MORI_NUMERICAL_TOLERANCE * _eigenvalue_scale(eigenvalues)
    return eigenvalues > positive_threshold


def _validated_sample_id(sample_id: str) -> str:
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a string")
    stripped_sample_id = sample_id.strip()
    if stripped_sample_id == "":
        raise ValueError("sample_id must not be empty")
    return stripped_sample_id


def _eigenvalue_scale(eigenvalues: np.ndarray) -> float:
    return max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)


def _assert_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _assert_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value}")


# ---- finite_markov_additive_green_kubo.py ----

HALF_JUMP_VARIANCE_FACTOR = 0.5
ZERO_VALUE = 0.0
FINITE_MARKOV_ADDITIVE_TOLERANCE = math.sqrt(np.finfo(float).eps)


@dataclass(frozen=True)
class MarkovAdditiveEvent:
    from_state_index: int
    to_state_index: int
    rate_s_inv: float
    charge_displacement_m: tuple[float, float, float]
    label: str
    family_label: str


@dataclass(frozen=True)
class MarkovAdditiveEventFamilyAttribution:
    family_label: str
    direct_sigma_mS_cm: float
    self_corrector_sigma_mS_cm: float
    marginal_corrector_sigma_mS_cm: float
    marginal_net_sigma_mS_cm: float
    direct_fraction: float
    marginal_net_fraction: float


@dataclass(frozen=True)
class MarkovAdditiveConductivityInput:
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: np.ndarray
    events: tuple[MarkovAdditiveEvent, ...]
    temperature_K: float


@dataclass(frozen=True)
class ConcentrationGeneratorValidation:
    row_sum_residual: float
    stationary_residual_mol_m3_s: float
    detailed_balance_residual_mol_m3_s: float
    minimum_offdiagonal_rate_s_inv: float
    concentration_sum_mol_m3: float


@dataclass(frozen=True)
class MarkovAdditiveConductivityResult:
    generator_s_inv: np.ndarray
    validation: ConcentrationGeneratorValidation
    event_reversal_residual_mol_m3_s: float
    drift_by_state_m_s: np.ndarray
    corrector_mori_input: ProjectedMoriConductivityInput
    corrector_mori_result: ProjectedMoriConductivityResult
    direct_axis_density_m2_s_mol_m3: tuple[float, float, float]
    corrector_axis_density_m2_s_mol_m3: tuple[float, float, float]
    effective_axis_density_m2_s_mol_m3: tuple[float, float, float]
    direct_sigma_S_m: float
    corrector_sigma_S_m: float
    sigma_S_m: float
    direct_sigma_mS_cm: float
    corrector_sigma_mS_cm: float
    sigma_mS_cm: float
    minimum_effective_axis_density_m2_s_mol_m3: float


def build_generator_from_events(
    state_count: int,
    events: Sequence[MarkovAdditiveEvent],
) -> np.ndarray:
    """Build a CTMC generator from off-diagonal Markov-additive events."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    if len(events) == 0:
        raise ValueError("events must contain at least one event")
    generator_matrix = np.zeros((state_count, state_count), dtype=float)
    for event in events:
        _validate_event_indices(event, state_count)
        _positive_float(event.rate_s_inv, f"{event.label}.rate_s_inv")
        _validated_displacement(event.charge_displacement_m, event.label)
        if event.from_state_index != event.to_state_index:
            generator_matrix[event.from_state_index, event.to_state_index] += (
                event.rate_s_inv
            )
    row_exit_rates = np.sum(generator_matrix, axis=1)
    np.fill_diagonal(generator_matrix, -row_exit_rates)
    return generator_matrix


def validate_concentration_reversible_generator(
    generator_matrix_s_inv: np.ndarray,
    state_concentrations_mol_m3: np.ndarray,
) -> ConcentrationGeneratorValidation:
    """Validate row conservation, stationarity, and concentration detailed balance."""

    generator_matrix = _validated_generator_matrix(generator_matrix_s_inv)
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        generator_matrix.shape[0],
    )
    tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    row_sum_residual = float(np.max(np.abs(np.sum(generator_matrix, axis=1))))
    offdiagonal_rates = generator_matrix[~np.eye(generator_matrix.shape[0], dtype=bool)]
    minimum_offdiagonal_rate = (
        float(np.min(offdiagonal_rates)) if offdiagonal_rates.size else ZERO_VALUE
    )
    maximum_diagonal_entry = float(np.max(np.diag(generator_matrix)))
    stationary_residual = float(
        np.max(np.abs(state_concentrations @ generator_matrix))
    )
    detailed_balance_matrix = (
        state_concentrations[:, None] * generator_matrix
        - state_concentrations[None, :] * generator_matrix.T
    )
    detailed_balance_residual = float(np.max(np.abs(detailed_balance_matrix)))
    if row_sum_residual > tolerance:
        raise ValueError(f"generator row-sum residual {row_sum_residual} exceeds {tolerance}")
    if minimum_offdiagonal_rate < -tolerance:
        raise ValueError("generator off-diagonal entries must be nonnegative")
    if maximum_diagonal_entry > tolerance:
        raise ValueError("generator diagonal entries must be nonpositive")
    if stationary_residual > tolerance:
        raise ValueError(
            f"stationary concentration residual {stationary_residual} exceeds {tolerance}"
        )
    if detailed_balance_residual > tolerance:
        raise ValueError(
            f"detailed-balance residual {detailed_balance_residual} exceeds {tolerance}"
        )
    return ConcentrationGeneratorValidation(
        row_sum_residual=row_sum_residual,
        stationary_residual_mol_m3_s=stationary_residual,
        detailed_balance_residual_mol_m3_s=detailed_balance_residual,
        minimum_offdiagonal_rate_s_inv=minimum_offdiagonal_rate,
        concentration_sum_mol_m3=float(np.sum(state_concentrations)),
    )


def validate_event_displacement_reversibility(
    events: Sequence[MarkovAdditiveEvent],
    state_concentrations_mol_m3: np.ndarray,
    state_count: int,
) -> float:
    """Validate reverse flux symmetry for every nonzero displacement event."""

    if len(events) == 0:
        raise ValueError("events must contain at least one event")
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        state_count,
    )
    weighted_flux_by_event_key: dict[tuple[int, int, tuple[float, float, float]], float] = {}
    for event in events:
        _validate_event_indices(event, state_count)
        event_rate_s_inv = _positive_float(event.rate_s_inv, f"{event.label}.rate_s_inv")
        displacement_array = _validated_displacement(
            event.charge_displacement_m,
            event.label,
        )
        if _is_zero_displacement(displacement_array):
            continue
        displacement_key = _displacement_key(displacement_array)
        event_key = (
            event.from_state_index,
            event.to_state_index,
            displacement_key,
        )
        weighted_flux = (
            state_concentrations[event.from_state_index]
            * event_rate_s_inv
        )
        weighted_flux_by_event_key[event_key] = (
            weighted_flux_by_event_key.get(event_key, ZERO_VALUE)
            + weighted_flux
        )
    if not weighted_flux_by_event_key:
        return ZERO_VALUE
    maximum_weighted_flux = max(
        abs(weighted_flux)
        for weighted_flux in weighted_flux_by_event_key.values()
    )
    tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        maximum_weighted_flux,
    )
    maximum_reversal_residual = ZERO_VALUE
    for event_key, weighted_flux in weighted_flux_by_event_key.items():
        from_state_index, to_state_index, displacement_key = event_key
        reverse_displacement_key = tuple(
            _canonical_float_for_key(-component)
            for component in displacement_key
        )
        reverse_key = (
            to_state_index,
            from_state_index,
            reverse_displacement_key,
        )
        reverse_weighted_flux = weighted_flux_by_event_key.get(
            reverse_key,
            ZERO_VALUE,
        )
        reversal_residual = abs(weighted_flux - reverse_weighted_flux)
        maximum_reversal_residual = max(
            maximum_reversal_residual,
            reversal_residual,
        )
    if maximum_reversal_residual > tolerance:
        raise ValueError(
            "event displacement reverse residual "
            f"{maximum_reversal_residual} exceeds {tolerance}"
        )
    return float(maximum_reversal_residual)


def compute_markov_additive_green_kubo_conductivity(
    markov_additive_input: MarkovAdditiveConductivityInput,
) -> MarkovAdditiveConductivityResult:
    """Evaluate sigma = direct jump variance minus Mori corrector."""

    state_labels = tuple(markov_additive_input.state_labels)
    if len(state_labels) == 0:
        raise ValueError("state_labels must contain at least one state")
    if len(set(state_labels)) != len(state_labels):
        raise ValueError("state_labels must be unique")
    state_concentrations = _validated_state_concentrations(
        markov_additive_input.state_concentrations_mol_m3,
        len(state_labels),
    )
    temperature_K = _positive_float(markov_additive_input.temperature_K, "temperature_K")
    generator_matrix = build_generator_from_events(
        len(state_labels),
        markov_additive_input.events,
    )
    validation = validate_concentration_reversible_generator(
        generator_matrix,
        state_concentrations,
    )
    event_reversal_residual = validate_event_displacement_reversibility(
        markov_additive_input.events,
        state_concentrations,
        len(state_labels),
    )
    axis_count = int(CARTESIAN_AXIS_COUNT)
    direct_axis_density = np.zeros(axis_count, dtype=float)
    drift_by_state = np.zeros((len(state_labels), axis_count), dtype=float)
    for event in markov_additive_input.events:
        displacement_array = np.asarray(event.charge_displacement_m, dtype=float)
        direct_axis_density += (
            HALF_JUMP_VARIANCE_FACTOR
            * state_concentrations[event.from_state_index]
            * event.rate_s_inv
            * displacement_array
            * displacement_array
        )
        drift_by_state[event.from_state_index, :] += (
            event.rate_s_inv * displacement_array
        )
    stationary_drift = state_concentrations @ drift_by_state
    drift_tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    if float(np.max(np.abs(stationary_drift))) > drift_tolerance:
        raise ValueError("Markov-additive process has nonzero stationary drift")

    symmetrized_energy_matrix = _symmetrized_energy_matrix(
        generator_matrix,
        state_concentrations,
    )
    current_coupling_matrix = (
        np.sqrt(state_concentrations)[:, None] * drift_by_state
    ).T
    beta_factor = F * F / (R * temperature_K)
    corrector_mori_input = ProjectedMoriConductivityInput(
        direct_energy_matrix=np.zeros_like(symmetrized_energy_matrix),
        memory_self_energy_matrix=symmetrized_energy_matrix,
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=beta_factor,
    )
    corrector_mori_result = compute_projected_mori_conductivity(corrector_mori_input)
    corrector_axis_density = np.asarray(
        corrector_mori_result.quadratic_form_by_axis,
        dtype=float,
    )
    effective_axis_density = direct_axis_density - corrector_axis_density
    density_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(direct_axis_density))),
        float(np.max(np.abs(corrector_axis_density))),
    )
    if float(np.min(effective_axis_density)) < -density_tolerance:
        raise ValueError("Markov-additive effective axis density became negative")
    effective_axis_density = np.asarray(
        [
            ZERO_VALUE if abs(value) <= density_tolerance else float(value)
            for value in effective_axis_density
        ],
        dtype=float,
    )
    direct_sigma_S_m = beta_factor * float(np.sum(direct_axis_density)) / CARTESIAN_AXIS_COUNT
    corrector_sigma_S_m = corrector_mori_result.sigma_S_m
    sigma_S_m = direct_sigma_S_m - corrector_sigma_S_m
    if sigma_S_m < -density_tolerance:
        raise ValueError("Markov-additive conductivity became negative")
    if abs(sigma_S_m) <= density_tolerance:
        sigma_S_m = ZERO_VALUE
    return MarkovAdditiveConductivityResult(
        generator_s_inv=generator_matrix,
        validation=validation,
        event_reversal_residual_mol_m3_s=event_reversal_residual,
        drift_by_state_m_s=drift_by_state,
        corrector_mori_input=corrector_mori_input,
        corrector_mori_result=corrector_mori_result,
        direct_axis_density_m2_s_mol_m3=tuple(float(value) for value in direct_axis_density),
        corrector_axis_density_m2_s_mol_m3=tuple(float(value) for value in corrector_axis_density),
        effective_axis_density_m2_s_mol_m3=tuple(float(value) for value in effective_axis_density),
        direct_sigma_S_m=float(direct_sigma_S_m),
        corrector_sigma_S_m=float(corrector_sigma_S_m),
        sigma_S_m=float(sigma_S_m),
        direct_sigma_mS_cm=float(direct_sigma_S_m * S_M_TO_MS_CM),
        corrector_sigma_mS_cm=float(corrector_sigma_S_m * S_M_TO_MS_CM),
        sigma_mS_cm=float(sigma_S_m * S_M_TO_MS_CM),
        minimum_effective_axis_density_m2_s_mol_m3=float(np.min(effective_axis_density)),
    )


def compute_markov_additive_event_family_attribution(
    markov_result: MarkovAdditiveConductivityResult,
    events: tuple[MarkovAdditiveEvent, ...],
    state_concentrations_mol_m3: np.ndarray,
    event_family_by_label: Mapping[str, str],
    temperature_K: float,
) -> tuple[MarkovAdditiveEventFamilyAttribution, ...]:
    """Attribute direct and corrector terms to event families with fixed Q."""

    if len(events) == 0:
        raise ValueError("events must contain at least one event")
    temperature_K = _positive_float(temperature_K, "temperature_K")
    generator_matrix = _validated_generator_matrix(markov_result.generator_s_inv)
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        generator_matrix.shape[0],
    )
    validate_concentration_reversible_generator(generator_matrix, state_concentrations)
    validate_event_displacement_reversibility(
        events,
        state_concentrations,
        generator_matrix.shape[0],
    )
    event_generator_matrix = build_generator_from_events(
        generator_matrix.shape[0],
        events,
    )
    generator_tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    generator_difference = float(
        np.max(np.abs(event_generator_matrix - generator_matrix))
    )
    if generator_difference > generator_tolerance:
        raise ValueError(
            f"event-family attribution generator mismatch {generator_difference} "
            f"exceeds {generator_tolerance}"
        )
    family_labels = _event_family_labels(events, event_family_by_label)
    axis_count = int(CARTESIAN_AXIS_COUNT)
    direct_density_by_family: dict[str, np.ndarray] = {
        family_label: np.zeros(axis_count, dtype=float)
        for family_label in family_labels
    }
    drift_by_family: dict[str, np.ndarray] = {
        family_label: np.zeros((state_concentrations.shape[0], axis_count), dtype=float)
        for family_label in family_labels
    }
    for event in events:
        family_label = event_family_by_label[event.label]
        displacement_array = np.asarray(event.charge_displacement_m, dtype=float)
        direct_density_by_family[family_label] += (
            HALF_JUMP_VARIANCE_FACTOR
            * state_concentrations[event.from_state_index]
            * event.rate_s_inv
            * displacement_array
            * displacement_array
        )
        drift_by_family[family_label][event.from_state_index, :] += (
            event.rate_s_inv * displacement_array
        )

    symmetrized_energy_matrix = _symmetrized_energy_matrix(
        generator_matrix,
        state_concentrations,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized_energy_matrix)
    _validate_energy_eigenvalues(eigenvalues, "event_family_attribution.energy_matrix")
    total_drift_matrix = np.zeros((state_concentrations.shape[0], axis_count), dtype=float)
    for family_label in family_labels:
        total_drift_matrix += drift_by_family[family_label]
    beta_factor = F * F / (R * temperature_K)
    total_direct_sigma_mS_cm = markov_result.direct_sigma_mS_cm
    total_sigma_mS_cm = markov_result.sigma_mS_cm
    attributions: list[MarkovAdditiveEventFamilyAttribution] = []
    for family_label in family_labels:
        family_drift_matrix = drift_by_family[family_label]
        direct_sigma_mS_cm = (
            beta_factor
            * float(np.sum(direct_density_by_family[family_label]))
            / CARTESIAN_AXIS_COUNT
            * S_M_TO_MS_CM
        )
        self_corrector_sigma_mS_cm = _family_cross_corrector_sigma_mS_cm(
            family_drift_matrix,
            family_drift_matrix,
            state_concentrations,
            eigenvalues,
            eigenvectors,
            beta_factor,
        )
        family_total_cross_sigma_mS_cm = _family_cross_corrector_sigma_mS_cm(
            family_drift_matrix,
            total_drift_matrix,
            state_concentrations,
            eigenvalues,
            eigenvectors,
            beta_factor,
        )
        marginal_corrector_sigma_mS_cm = (
            2.0 * family_total_cross_sigma_mS_cm
            - self_corrector_sigma_mS_cm
        )
        marginal_net_sigma_mS_cm = (
            direct_sigma_mS_cm - marginal_corrector_sigma_mS_cm
        )
        direct_fraction = (
            direct_sigma_mS_cm / total_direct_sigma_mS_cm
            if total_direct_sigma_mS_cm > 0.0
            else 0.0
        )
        marginal_net_fraction = (
            marginal_net_sigma_mS_cm / total_sigma_mS_cm
            if total_sigma_mS_cm > 0.0
            else 0.0
        )
        attributions.append(
            MarkovAdditiveEventFamilyAttribution(
                family_label=family_label,
                direct_sigma_mS_cm=float(direct_sigma_mS_cm),
                self_corrector_sigma_mS_cm=float(self_corrector_sigma_mS_cm),
                marginal_corrector_sigma_mS_cm=float(marginal_corrector_sigma_mS_cm),
                marginal_net_sigma_mS_cm=float(marginal_net_sigma_mS_cm),
                direct_fraction=float(direct_fraction),
                marginal_net_fraction=float(marginal_net_fraction),
            )
        )
    return tuple(
        sorted(
            attributions,
            key=lambda attribution: abs(attribution.marginal_net_sigma_mS_cm),
            reverse=True,
        )
    )


def _event_family_labels(
    events: tuple[MarkovAdditiveEvent, ...],
    event_family_by_label: Mapping[str, str],
) -> tuple[str, ...]:
    family_labels: list[str] = []
    for event in events:
        if event.label not in event_family_by_label:
            raise ValueError(f"missing event family for event {event.label}")
        mapped_family_label = event_family_by_label[event.label]
        if mapped_family_label != event.family_label:
            raise ValueError(f"event family mapping disagrees with event {event.label}")
        if mapped_family_label == "":
            raise ValueError(f"event {event.label} has an empty family label")
        if mapped_family_label not in family_labels:
            family_labels.append(mapped_family_label)
    return tuple(family_labels)


def _family_cross_corrector_sigma_mS_cm(
    left_drift_by_state: np.ndarray,
    right_drift_by_state: np.ndarray,
    state_concentrations: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    beta_factor: float,
) -> float:
    sqrt_concentrations = np.sqrt(state_concentrations)
    left_current_coupling_matrix = (sqrt_concentrations[:, None] * left_drift_by_state).T
    right_current_coupling_matrix = (sqrt_concentrations[:, None] * right_drift_by_state).T
    cross_density_sum = 0.0
    for axis_index in range(int(CARTESIAN_AXIS_COUNT)):
        cross_density_sum += _projected_cross_form(
            eigenvalues,
            eigenvectors,
            left_current_coupling_matrix[axis_index, :],
            right_current_coupling_matrix[axis_index, :],
        )
    return float(
        beta_factor
        * cross_density_sum
        / CARTESIAN_AXIS_COUNT
        * S_M_TO_MS_CM
    )


def _projected_cross_form(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    left_current_coupling: np.ndarray,
    right_current_coupling: np.ndarray,
) -> float:
    positive_mode_mask = _positive_energy_mode_mask(eigenvalues)
    if not np.any(positive_mode_mask):
        return 0.0
    projected_left = eigenvectors[:, positive_mode_mask].T @ left_current_coupling
    projected_right = eigenvectors[:, positive_mode_mask].T @ right_current_coupling
    return float(
        math.fsum(
            float(left_value * right_value / eigenvalue)
            for left_value, right_value, eigenvalue in zip(
                projected_left,
                projected_right,
                eigenvalues[positive_mode_mask],
            )
        )
    )


def _validate_energy_eigenvalues(eigenvalues: np.ndarray, context: str) -> None:
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), FINITE_MARKOV_ADDITIVE_TOLERANCE)
    allowed_negative_eigenvalue = FINITE_MARKOV_ADDITIVE_TOLERANCE * eigenvalue_scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(f"{context} must be positive semidefinite")


def _positive_energy_mode_mask(eigenvalues: np.ndarray) -> np.ndarray:
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), FINITE_MARKOV_ADDITIVE_TOLERANCE)
    return eigenvalues > FINITE_MARKOV_ADDITIVE_TOLERANCE * eigenvalue_scale


def _symmetrized_energy_matrix(
    generator_matrix: np.ndarray,
    state_concentrations: np.ndarray,
) -> np.ndarray:
    sqrt_concentrations = np.sqrt(state_concentrations)
    inverse_sqrt_concentrations = 1.0 / sqrt_concentrations
    energy_matrix = (
        sqrt_concentrations[:, None]
        * (-generator_matrix)
        * inverse_sqrt_concentrations[None, :]
    )
    return _symmetrized_matrix(energy_matrix)


def _validated_generator_matrix(generator_matrix_s_inv: np.ndarray) -> np.ndarray:
    generator_matrix = np.asarray(generator_matrix_s_inv, dtype=float)
    if generator_matrix.ndim != 2:
        raise ValueError("generator_matrix_s_inv must be two-dimensional")
    if generator_matrix.shape[0] != generator_matrix.shape[1]:
        raise ValueError("generator_matrix_s_inv must be square")
    if generator_matrix.shape[0] == 0:
        raise ValueError("generator_matrix_s_inv must contain at least one state")
    if not np.all(np.isfinite(generator_matrix)):
        raise ValueError("generator_matrix_s_inv contains non-finite values")
    return generator_matrix


def _validated_state_concentrations(
    state_concentrations_mol_m3: np.ndarray,
    state_count: int,
) -> np.ndarray:
    state_concentrations = np.asarray(state_concentrations_mol_m3, dtype=float)
    if state_concentrations.ndim != 1:
        raise ValueError("state_concentrations_mol_m3 must be one-dimensional")
    if state_concentrations.shape[0] != state_count:
        raise ValueError("state_concentrations_mol_m3 length must equal state count")
    if not np.all(np.isfinite(state_concentrations)):
        raise ValueError("state_concentrations_mol_m3 contains non-finite values")
    if np.any(state_concentrations <= ZERO_VALUE):
        raise ValueError("state_concentrations_mol_m3 must be strictly positive")
    return state_concentrations


def _validate_event_indices(
    event: MarkovAdditiveEvent,
    state_count: int,
) -> None:
    if not isinstance(event.from_state_index, int):
        raise TypeError(f"{event.label}.from_state_index must be an integer")
    if not isinstance(event.to_state_index, int):
        raise TypeError(f"{event.label}.to_state_index must be an integer")
    if event.from_state_index < 0 or event.from_state_index >= state_count:
        raise ValueError(f"{event.label}.from_state_index is outside the state range")
    if event.to_state_index < 0 or event.to_state_index >= state_count:
        raise ValueError(f"{event.label}.to_state_index is outside the state range")


def _validated_displacement(
    charge_displacement_m: tuple[float, float, float],
    event_label: str,
) -> np.ndarray:
    displacement_array = np.asarray(charge_displacement_m, dtype=float)
    if displacement_array.shape != (int(CARTESIAN_AXIS_COUNT),):
        raise ValueError(f"{event_label}.charge_displacement_m must have length three")
    if not np.all(np.isfinite(displacement_array)):
        raise ValueError(f"{event_label}.charge_displacement_m contains non-finite values")
    return displacement_array


def _is_zero_displacement(displacement_array: np.ndarray) -> bool:
    return bool(np.all(displacement_array == ZERO_VALUE))


def _displacement_key(displacement_array: np.ndarray) -> tuple[float, float, float]:
    return tuple(
        _canonical_float_for_key(float(component))
        for component in displacement_array
    )


def _canonical_float_for_key(value: float) -> float:
    if value == ZERO_VALUE:
        return ZERO_VALUE
    return float(value)


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= ZERO_VALUE:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _symmetrized_matrix(matrix: np.ndarray) -> np.ndarray:
    return HALF_JUMP_VARIANCE_FACTOR * (matrix + matrix.T)


def _matrix_tolerance(
    generator_matrix: np.ndarray,
    state_concentrations: np.ndarray,
) -> float:
    return FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(generator_matrix))),
        float(np.max(np.abs(state_concentrations))),
        float(np.sum(state_concentrations)),
    )


# ---- ion_atmosphere.py ----

SUPPORTED_ION_ATMOSPHERE_SOLVERS = ("off", "diagonal_pnp_stokes_l1_cell_experimental")
SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS = ("off", "finite_size_bulk_pnp_stokes_l1_cell")
STOKES_TRANSLATION_AXIS_COUNT = 3  # Cartesian translation axes in the spherical Stokes drag solution.
STOKES_NO_SLIP_BOUNDARY_FACTOR = 2  # No-slip sphere doubles the axis count in zeta = 6*pi*eta*a.
STOKES_SPHERE_DRAG_FACTOR = STOKES_NO_SLIP_BOUNDARY_FACTOR * STOKES_TRANSLATION_AXIS_COUNT
SPHERE_VOLUME_NUMERATOR = STOKES_SPHERE_DRAG_FACTOR - STOKES_NO_SLIP_BOUNDARY_FACTOR
SPHERE_VOLUME_DENOMINATOR = STOKES_TRANSLATION_AXIS_COUNT


@dataclass(frozen=True)
class IonAtmosphereInput:
    carrier_concentrations_mol_m3: Mapping[str, float]
    carrier_charges: Mapping[str, int]
    local_diffusivity_m2_s_by_carrier: Mapping[str, float]
    hydrodynamic_radius_m_by_carrier: Mapping[str, float]
    viscosity_Pa_s: float
    relative_dielectric: float
    temperature_K: float
    solver: str


@dataclass(frozen=True)
class BulkIonAtmosphereInput:
    carrier_labels: tuple[str, ...]
    carrier_concentrations_mol_m3: Mapping[str, float]
    carrier_charges: Mapping[str, int]
    local_diffusivity_m2_s_by_carrier: Mapping[str, float]
    hydrodynamic_radius_m_by_carrier: Mapping[str, float]
    viscosity_Pa_s: float
    relative_dielectric: float
    temperature_K: float
    solver: str


@dataclass(frozen=True)
class IonAtmosphereState:
    kappa_inv_m: float
    ionic_strength_mol_m3: float
    friction_ratio_by_carrier: dict[str, float]
    zeta0_by_carrier: dict[str, float]
    zeta_ep_by_carrier: dict[str, float]
    zeta_rel_by_carrier: dict[str, float]
    zeta_atm_by_carrier: dict[str, float]
    D_micro_by_carrier: dict[str, float]
    D_eff_by_carrier: dict[str, float]
    solver: str


@dataclass(frozen=True)
class BulkIonAtmosphereState:
    carrier_labels: tuple[str, ...]
    kappa_inv_m: float
    ionic_strength_mol_m3: float
    ambipolar_diffusivity_m2_s: float
    resistance_matrix_kg_s: np.ndarray
    resistance_ep_kg_s: np.ndarray
    resistance_rel_kg_s: np.ndarray
    steric_volume_fraction: float
    thermodynamic_factor_trace: float
    thermodynamic_factor_matrix: np.ndarray
    thermodynamic_factor_eigenvalues: tuple[float, ...]
    structure_response_matrix: np.ndarray
    structure_factor_charge_mode: float
    kappa_radius_by_carrier: dict[str, float]
    solver: str


def build_ion_atmosphere_state(ion_atmosphere_input: IonAtmosphereInput) -> IonAtmosphereState:
    """Build ion-atmosphere friction diagnostics for charged mobile carriers."""

    _assert_positive_finite(ion_atmosphere_input.viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(ion_atmosphere_input.relative_dielectric, "relative_dielectric")
    _assert_positive_finite(ion_atmosphere_input.temperature_K, "temperature_K")
    _validate_solver(ion_atmosphere_input.solver)

    carrier_names = tuple(ion_atmosphere_input.carrier_concentrations_mol_m3)
    if not carrier_names:
        raise ValueError("ion atmosphere requires at least one charged carrier")

    charge_weighted_concentration_mol_m3 = 0.0
    friction_ratio_by_carrier: dict[str, float] = {}
    zeta0_by_carrier: dict[str, float] = {}
    zeta_ep_by_carrier: dict[str, float] = {}
    zeta_rel_by_carrier: dict[str, float] = {}
    zeta_atm_by_carrier: dict[str, float] = {}
    D_micro_by_carrier: dict[str, float] = {}
    D_eff_by_carrier: dict[str, float] = {}
    local_diffusivity_by_carrier: dict[str, float] = {}
    hydrodynamic_radius_by_carrier: dict[str, float] = {}
    charge_by_carrier: dict[str, int] = {}

    for carrier_name in carrier_names:
        concentration_mol_m3 = _require_nonnegative_finite(
            ion_atmosphere_input.carrier_concentrations_mol_m3,
            carrier_name,
            "carrier_concentrations_mol_m3",
        )
        charge_number = _require_charge(ion_atmosphere_input.carrier_charges, carrier_name)
        local_diffusivity_m2_s = _require_positive_finite(
            ion_atmosphere_input.local_diffusivity_m2_s_by_carrier,
            carrier_name,
            "local_diffusivity_m2_s_by_carrier",
        )
        hydrodynamic_radius_m = _require_positive_finite(
            ion_atmosphere_input.hydrodynamic_radius_m_by_carrier,
            carrier_name,
            "hydrodynamic_radius_m_by_carrier",
        )
        charge_weighted_concentration_mol_m3 += charge_number * charge_number * concentration_mol_m3
        local_diffusivity_by_carrier[carrier_name] = local_diffusivity_m2_s
        hydrodynamic_radius_by_carrier[carrier_name] = hydrodynamic_radius_m
        charge_by_carrier[carrier_name] = charge_number

    kappa_inv_m = _debye_kappa_inv_m(
        charge_weighted_concentration_mol_m3,
        ion_atmosphere_input.relative_dielectric,
        ion_atmosphere_input.temperature_K,
    )
    if math.isinf(kappa_inv_m):
        kappa_m_inv = 0.0
    else:
        kappa_m_inv = 1.0 / kappa_inv_m

    for carrier_name in carrier_names:
        local_diffusivity_m2_s = local_diffusivity_by_carrier[carrier_name]
        hydrodynamic_radius_m = hydrodynamic_radius_by_carrier[carrier_name]
        charge_number = charge_by_carrier[carrier_name]
        zeta0_kg_s = K_B * ion_atmosphere_input.temperature_K / local_diffusivity_m2_s
        if ion_atmosphere_input.solver == "off":
            zeta_ep_kg_s = 0.0
            zeta_rel_kg_s = 0.0
        elif ion_atmosphere_input.solver == "diagonal_pnp_stokes_l1_cell_experimental":
            zeta_ep_kg_s = _electrophoretic_drag_kg_s(
                viscosity_Pa_s=ion_atmosphere_input.viscosity_Pa_s,
                hydrodynamic_radius_m=hydrodynamic_radius_m,
                kappa_m_inv=kappa_m_inv,
            )
            zeta_rel_kg_s = _relaxation_drag_kg_s(
                charge_number=charge_number,
                local_diffusivity_m2_s=local_diffusivity_m2_s,
                hydrodynamic_radius_m=hydrodynamic_radius_m,
                relative_dielectric=ion_atmosphere_input.relative_dielectric,
                kappa_m_inv=kappa_m_inv,
            )
        else:
            raise ValueError(f"Unsupported ion-atmosphere solver {ion_atmosphere_input.solver!r}")
        zeta_atm_kg_s = zeta_ep_kg_s + zeta_rel_kg_s
        _assert_nonnegative_finite(zeta_ep_kg_s, f"{carrier_name}.zeta_ep_kg_s")
        _assert_nonnegative_finite(zeta_rel_kg_s, f"{carrier_name}.zeta_rel_kg_s")
        _assert_nonnegative_finite(zeta_atm_kg_s, f"{carrier_name}.zeta_atm_kg_s")
        friction_ratio = zeta0_kg_s / (zeta0_kg_s + zeta_atm_kg_s)
        if friction_ratio <= 0.0 or friction_ratio > 1.0:
            raise ValueError(f"{carrier_name}.friction_ratio must be in (0, 1], got {friction_ratio}")

        zeta0_by_carrier[carrier_name] = zeta0_kg_s
        zeta_ep_by_carrier[carrier_name] = zeta_ep_kg_s
        zeta_rel_by_carrier[carrier_name] = zeta_rel_kg_s
        zeta_atm_by_carrier[carrier_name] = zeta_atm_kg_s
        D_micro_by_carrier[carrier_name] = local_diffusivity_m2_s
        D_eff_by_carrier[carrier_name] = local_diffusivity_m2_s * friction_ratio
        friction_ratio_by_carrier[carrier_name] = friction_ratio

    return IonAtmosphereState(
        kappa_inv_m=kappa_inv_m,
        ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
        friction_ratio_by_carrier=friction_ratio_by_carrier,
        zeta0_by_carrier=zeta0_by_carrier,
        zeta_ep_by_carrier=zeta_ep_by_carrier,
        zeta_rel_by_carrier=zeta_rel_by_carrier,
        zeta_atm_by_carrier=zeta_atm_by_carrier,
        D_micro_by_carrier=D_micro_by_carrier,
        D_eff_by_carrier=D_eff_by_carrier,
        solver=ion_atmosphere_input.solver,
    )


def build_bulk_ion_atmosphere_state(
    bulk_ion_atmosphere_input: BulkIonAtmosphereInput,
) -> BulkIonAtmosphereState:
    """Build a finite-size bulk carrier atmosphere resistance matrix."""

    _assert_positive_finite(bulk_ion_atmosphere_input.viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(bulk_ion_atmosphere_input.relative_dielectric, "relative_dielectric")
    _assert_positive_finite(bulk_ion_atmosphere_input.temperature_K, "temperature_K")
    _validate_bulk_solver(bulk_ion_atmosphere_input.solver)
    carrier_labels = tuple(bulk_ion_atmosphere_input.carrier_labels)
    if not carrier_labels:
        raise ValueError("bulk ion atmosphere requires at least one charged carrier")
    if len(set(carrier_labels)) != len(carrier_labels):
        raise ValueError("bulk ion atmosphere carrier_labels must be unique")

    charge_weighted_concentration_mol_m3 = 0.0
    steric_volume_fraction = 0.0
    concentration_by_carrier: dict[str, float] = {}
    charge_by_carrier: dict[str, int] = {}
    diffusivity_by_carrier: dict[str, float] = {}
    radius_by_carrier: dict[str, float] = {}
    for carrier_label in carrier_labels:
        concentration_mol_m3 = _require_nonnegative_finite(
            bulk_ion_atmosphere_input.carrier_concentrations_mol_m3,
            carrier_label,
            "carrier_concentrations_mol_m3",
        )
        charge_number = _require_charge(bulk_ion_atmosphere_input.carrier_charges, carrier_label)
        local_diffusivity_m2_s = _require_positive_finite(
            bulk_ion_atmosphere_input.local_diffusivity_m2_s_by_carrier,
            carrier_label,
            "local_diffusivity_m2_s_by_carrier",
        )
        hydrodynamic_radius_m = _require_positive_finite(
            bulk_ion_atmosphere_input.hydrodynamic_radius_m_by_carrier,
            carrier_label,
            "hydrodynamic_radius_m_by_carrier",
        )
        charge_weighted_concentration_mol_m3 += charge_number * charge_number * concentration_mol_m3
        steric_volume_fraction += (
            concentration_mol_m3
            * N_A
            * _sphere_volume_m3(hydrodynamic_radius_m)
        )
        concentration_by_carrier[carrier_label] = concentration_mol_m3
        charge_by_carrier[carrier_label] = charge_number
        diffusivity_by_carrier[carrier_label] = local_diffusivity_m2_s
        radius_by_carrier[carrier_label] = hydrodynamic_radius_m
    _assert_nonnegative_finite(steric_volume_fraction, "steric_volume_fraction")
    if steric_volume_fraction >= 1.0:
        raise ValueError(f"steric_volume_fraction must be below one, got {steric_volume_fraction}")
    ambipolar_diffusivity_m2_s = _ambipolar_diffusivity_m2_s(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        diffusivity_by_carrier=diffusivity_by_carrier,
    )

    kappa_inv_m = _debye_kappa_inv_m(
        charge_weighted_concentration_mol_m3,
        bulk_ion_atmosphere_input.relative_dielectric,
        bulk_ion_atmosphere_input.temperature_K,
    )
    if (
        bulk_ion_atmosphere_input.solver == "off"
        or charge_weighted_concentration_mol_m3 == 0.0
    ):
        matrix_shape = (len(carrier_labels), len(carrier_labels))
        zero_matrix = np.zeros(matrix_shape, dtype=float)
        thermodynamic_factor_matrix = _finite_size_thermodynamic_factor_matrix(
            carrier_labels=carrier_labels,
            concentration_by_carrier=concentration_by_carrier,
            radius_by_carrier=radius_by_carrier,
            steric_volume_fraction=steric_volume_fraction,
        )
        thermodynamic_factor_trace = float(np.trace(thermodynamic_factor_matrix))
        thermodynamic_factor_eigenvalues = _matrix_eigenvalue_tuple(
            thermodynamic_factor_matrix,
            "thermodynamic_factor_matrix",
        )
        structure_response_matrix = thermodynamic_factor_matrix.copy()
        return BulkIonAtmosphereState(
            carrier_labels=carrier_labels,
            kappa_inv_m=kappa_inv_m,
            ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
            ambipolar_diffusivity_m2_s=ambipolar_diffusivity_m2_s,
            resistance_matrix_kg_s=zero_matrix,
            resistance_ep_kg_s=zero_matrix.copy(),
            resistance_rel_kg_s=zero_matrix.copy(),
            steric_volume_fraction=steric_volume_fraction,
            thermodynamic_factor_trace=thermodynamic_factor_trace,
            thermodynamic_factor_matrix=thermodynamic_factor_matrix,
            thermodynamic_factor_eigenvalues=thermodynamic_factor_eigenvalues,
            structure_response_matrix=structure_response_matrix,
            structure_factor_charge_mode=0.0,
            kappa_radius_by_carrier={carrier_label: 0.0 for carrier_label in carrier_labels},
            solver=bulk_ion_atmosphere_input.solver,
        )
    if bulk_ion_atmosphere_input.solver != "finite_size_bulk_pnp_stokes_l1_cell":
        raise ValueError(f"Unsupported bulk ion-atmosphere solver {bulk_ion_atmosphere_input.solver!r}")
    if math.isinf(kappa_inv_m):
        kappa_m_inv = 0.0
    else:
        kappa_m_inv = 1.0 / kappa_inv_m
    finite_size_result = _finite_size_bulk_resistance_matrices_kg_s(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        diffusivity_by_carrier=diffusivity_by_carrier,
        radius_by_carrier=radius_by_carrier,
        viscosity_Pa_s=bulk_ion_atmosphere_input.viscosity_Pa_s,
        relative_dielectric=bulk_ion_atmosphere_input.relative_dielectric,
        kappa_m_inv=kappa_m_inv,
        steric_volume_fraction=steric_volume_fraction,
    )
    zeta_ep_values_kg_s = finite_size_result[0]
    zeta_rel_values_kg_s = finite_size_result[1]
    kappa_radius_by_carrier = finite_size_result[2]
    overlap_values = finite_size_result[3]
    relaxation_sign_values = finite_size_result[4]
    thermodynamic_factor_matrix = _finite_size_thermodynamic_factor_matrix(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        radius_by_carrier=radius_by_carrier,
        steric_volume_fraction=steric_volume_fraction,
    )
    thermodynamic_factor_trace = float(np.trace(thermodynamic_factor_matrix))
    thermodynamic_factor_eigenvalues = _matrix_eigenvalue_tuple(
        thermodynamic_factor_matrix,
        "thermodynamic_factor_matrix",
    )
    structure_response = _finite_size_structure_response_matrix(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        radius_by_carrier=radius_by_carrier,
        kappa_m_inv=kappa_m_inv,
        thermodynamic_factor_matrix=thermodynamic_factor_matrix,
    )
    structure_response_matrix = structure_response[0]
    structure_factor_charge_mode = structure_response[1]
    electrophoretic_sign_values = np.ones(len(carrier_labels), dtype=float)
    resistance_ep_kg_s = _matrix_coupled_psd_component_kg_s(
        zeta_values_kg_s=zeta_ep_values_kg_s,
        overlap_values=overlap_values,
        coupling_sign_values=electrophoretic_sign_values,
        structure_response_matrix=structure_response_matrix,
    )
    resistance_rel_kg_s = _matrix_coupled_psd_component_kg_s(
        zeta_values_kg_s=zeta_rel_values_kg_s,
        overlap_values=overlap_values,
        coupling_sign_values=relaxation_sign_values,
        structure_response_matrix=structure_response_matrix,
    )
    resistance_matrix_kg_s = resistance_ep_kg_s + resistance_rel_kg_s
    _validate_bulk_resistance_matrix(resistance_matrix_kg_s, "resistance_matrix_kg_s")
    return BulkIonAtmosphereState(
        carrier_labels=carrier_labels,
        kappa_inv_m=kappa_inv_m,
        ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
        ambipolar_diffusivity_m2_s=ambipolar_diffusivity_m2_s,
        resistance_matrix_kg_s=resistance_matrix_kg_s,
        resistance_ep_kg_s=resistance_ep_kg_s,
        resistance_rel_kg_s=resistance_rel_kg_s,
        steric_volume_fraction=steric_volume_fraction,
        thermodynamic_factor_trace=thermodynamic_factor_trace,
        thermodynamic_factor_matrix=thermodynamic_factor_matrix,
        thermodynamic_factor_eigenvalues=thermodynamic_factor_eigenvalues,
        structure_response_matrix=structure_response_matrix,
        structure_factor_charge_mode=structure_factor_charge_mode,
        kappa_radius_by_carrier=kappa_radius_by_carrier,
        solver=bulk_ion_atmosphere_input.solver,
    )


def _ambipolar_diffusivity_m2_s(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    diffusivity_by_carrier: Mapping[str, float],
) -> float:
    charge_weighted_diffusivity_sum = 0.0
    charge_weighted_concentration_sum = 0.0
    for carrier_label in carrier_labels:
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        charge_number = charge_by_carrier[carrier_label]
        diffusivity_m2_s = diffusivity_by_carrier[carrier_label]
        charge_weight = charge_number * charge_number * concentration_mol_m3
        charge_weighted_diffusivity_sum += charge_weight * diffusivity_m2_s
        charge_weighted_concentration_sum += charge_weight
    _assert_nonnegative_finite(
        charge_weighted_concentration_sum,
        "charge_weighted_concentration_sum",
    )
    if charge_weighted_concentration_sum == 0.0:
        return 0.0
    ambipolar_diffusivity_m2_s = (
        charge_weighted_diffusivity_sum / charge_weighted_concentration_sum
    )
    _assert_positive_finite(ambipolar_diffusivity_m2_s, "ambipolar_diffusivity_m2_s")
    return ambipolar_diffusivity_m2_s


def _finite_size_thermodynamic_factor_matrix(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    radius_by_carrier: Mapping[str, float],
    steric_volume_fraction: float,
) -> np.ndarray:
    free_volume_fraction = 1.0 - steric_volume_fraction
    _assert_positive_finite(free_volume_fraction, "free_volume_fraction")
    carrier_count = len(carrier_labels)
    finite_volume_vector = np.zeros(carrier_count, dtype=float)
    for carrier_index, carrier_label in enumerate(carrier_labels):
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        hydrodynamic_radius_m = radius_by_carrier[carrier_label]
        finite_volume_argument = (
            concentration_mol_m3
            * N_A
            * _sphere_volume_m3(hydrodynamic_radius_m)
        )
        _assert_nonnegative_finite(
            finite_volume_argument,
            f"{carrier_label}.finite_volume_argument",
        )
        finite_volume_vector[carrier_index] = math.sqrt(finite_volume_argument)
    thermodynamic_factor_matrix = (
        np.eye(carrier_count, dtype=float)
        + np.outer(finite_volume_vector, finite_volume_vector) / free_volume_fraction
    )
    _validate_bulk_resistance_matrix(
        thermodynamic_factor_matrix,
        "thermodynamic_factor_matrix",
    )
    return thermodynamic_factor_matrix


def _finite_size_structure_response_matrix(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    radius_by_carrier: Mapping[str, float],
    kappa_m_inv: float,
    thermodynamic_factor_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    charge_weighted_concentration = 0.0
    concentration_sum_mol_m3 = 0.0
    radius_weighted_concentration_m = 0.0
    charge_mode_vector = np.zeros(len(carrier_labels), dtype=float)
    for carrier_index, carrier_label in enumerate(carrier_labels):
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        charge_number = charge_by_carrier[carrier_label]
        radius_m = radius_by_carrier[carrier_label]
        charge_weighted_concentration += charge_number * charge_number * concentration_mol_m3
        concentration_sum_mol_m3 += concentration_mol_m3
        radius_weighted_concentration_m += concentration_mol_m3 * radius_m
        charge_mode_vector[carrier_index] = math.sqrt(concentration_mol_m3) * abs(charge_number)
    if charge_weighted_concentration == 0.0:
        return (thermodynamic_factor_matrix.copy(), 0.0)
    _assert_positive_finite(concentration_sum_mol_m3, "concentration_sum_mol_m3")
    average_hydrodynamic_radius_m = radius_weighted_concentration_m / concentration_sum_mol_m3
    _assert_positive_finite(average_hydrodynamic_radius_m, "average_hydrodynamic_radius_m")
    charge_mode_norm = float(np.linalg.norm(charge_mode_vector))
    _assert_positive_finite(charge_mode_norm, "charge_mode_norm")
    normalized_charge_mode = charge_mode_vector / charge_mode_norm
    structure_factor_charge_mode = kappa_m_inv * average_hydrodynamic_radius_m
    _assert_nonnegative_finite(structure_factor_charge_mode, "structure_factor_charge_mode")
    structure_response_matrix = (
        thermodynamic_factor_matrix
        + structure_factor_charge_mode * np.outer(normalized_charge_mode, normalized_charge_mode)
    )
    _validate_bulk_resistance_matrix(
        structure_response_matrix,
        "structure_response_matrix",
    )
    return (structure_response_matrix, structure_factor_charge_mode)


def _finite_size_bulk_resistance_matrices_kg_s(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    diffusivity_by_carrier: Mapping[str, float],
    radius_by_carrier: Mapping[str, float],
    viscosity_Pa_s: float,
    relative_dielectric: float,
    kappa_m_inv: float,
    steric_volume_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    free_volume_fraction = 1.0 - steric_volume_fraction
    _assert_positive_finite(free_volume_fraction, "free_volume_fraction")
    carrier_count = len(carrier_labels)
    zeta_ep_values = np.zeros(carrier_count, dtype=float)
    zeta_rel_values = np.zeros(carrier_count, dtype=float)
    overlap_values = np.zeros(carrier_count, dtype=float)
    relaxation_sign_values = np.zeros(carrier_count, dtype=float)
    kappa_radius_by_carrier: dict[str, float] = {}
    for carrier_index, carrier_label in enumerate(carrier_labels):
        hydrodynamic_radius_m = radius_by_carrier[carrier_label]
        stern_radius_m = hydrodynamic_radius_m + _opposite_charge_weighted_radius_m(
            carrier_label=carrier_label,
            carrier_labels=carrier_labels,
            concentration_by_carrier=concentration_by_carrier,
            charge_by_carrier=charge_by_carrier,
            radius_by_carrier=radius_by_carrier,
        )
        effective_kappa_m_inv = kappa_m_inv * free_volume_fraction / (1.0 + kappa_m_inv * stern_radius_m)
        kappa_radius_by_carrier[carrier_label] = effective_kappa_m_inv * hydrodynamic_radius_m
        zeta_ep_values[carrier_index] = _electrophoretic_drag_kg_s(
            viscosity_Pa_s=viscosity_Pa_s,
            hydrodynamic_radius_m=hydrodynamic_radius_m,
            kappa_m_inv=effective_kappa_m_inv,
        )
        zeta_rel_values[carrier_index] = _relaxation_drag_kg_s(
            charge_number=charge_by_carrier[carrier_label],
            local_diffusivity_m2_s=diffusivity_by_carrier[carrier_label],
            hydrodynamic_radius_m=hydrodynamic_radius_m,
            relative_dielectric=relative_dielectric,
            kappa_m_inv=effective_kappa_m_inv,
        )
        overlap_values[carrier_index] = math.exp(-effective_kappa_m_inv * stern_radius_m)
        relaxation_sign_values[carrier_index] = math.copysign(1.0, charge_by_carrier[carrier_label])
    return (
        zeta_ep_values,
        zeta_rel_values,
        kappa_radius_by_carrier,
        overlap_values,
        relaxation_sign_values,
    )


def _matrix_coupled_psd_component_kg_s(
    zeta_values_kg_s: np.ndarray,
    overlap_values: np.ndarray,
    coupling_sign_values: np.ndarray,
    structure_response_matrix: np.ndarray,
) -> np.ndarray:
    weighted_zeta_values = zeta_values_kg_s * overlap_values
    sign_matrix = np.diag(coupling_sign_values)
    weighted_matrix = np.diag(np.sqrt(weighted_zeta_values))
    residual_diagonal = zeta_values_kg_s * (1.0 - overlap_values)
    coupled_matrix = (
        sign_matrix
        @ weighted_matrix
        @ structure_response_matrix
        @ weighted_matrix
        @ sign_matrix
    )
    resistance_matrix = coupled_matrix + np.diag(residual_diagonal)
    _validate_bulk_resistance_matrix(resistance_matrix, "matrix_coupled_component_kg_s")
    return resistance_matrix


def _matrix_eigenvalue_tuple(
    matrix: np.ndarray,
    context: str,
) -> tuple[float, ...]:
    _validate_bulk_resistance_matrix(matrix, context)
    return tuple(float(value) for value in np.linalg.eigvalsh(matrix))


def _opposite_charge_weighted_radius_m(
    carrier_label: str,
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    radius_by_carrier: Mapping[str, float],
) -> float:
    source_charge = charge_by_carrier[carrier_label]
    weighted_radius_m = 0.0
    concentration_sum_mol_m3 = 0.0
    for other_label in carrier_labels:
        if source_charge * charge_by_carrier[other_label] >= 0:
            continue
        concentration_mol_m3 = concentration_by_carrier[other_label]
        weighted_radius_m += concentration_mol_m3 * radius_by_carrier[other_label]
        concentration_sum_mol_m3 += concentration_mol_m3
    _assert_positive_finite(concentration_sum_mol_m3, f"{carrier_label}.opposite_charge_concentration")
    return weighted_radius_m / concentration_sum_mol_m3


def _electrophoretic_drag_kg_s(
    viscosity_Pa_s: float,
    hydrodynamic_radius_m: float,
    kappa_m_inv: float,
) -> float:
    _assert_positive_finite(viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(hydrodynamic_radius_m, "hydrodynamic_radius_m")
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    kappa_radius = kappa_m_inv * hydrodynamic_radius_m
    if kappa_radius == 0.0:
        return 0.0
    return (
        STOKES_SPHERE_DRAG_FACTOR
        * math.pi
        * viscosity_Pa_s
        * hydrodynamic_radius_m
        * kappa_radius
        / (1.0 + kappa_radius)
    )


def _relaxation_drag_kg_s(
    charge_number: int,
    local_diffusivity_m2_s: float,
    hydrodynamic_radius_m: float,
    relative_dielectric: float,
    kappa_m_inv: float,
) -> float:
    _assert_positive_finite(local_diffusivity_m2_s, "local_diffusivity_m2_s")
    _assert_positive_finite(hydrodynamic_radius_m, "hydrodynamic_radius_m")
    _assert_positive_finite(relative_dielectric, "relative_dielectric")
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    kappa_radius = kappa_m_inv * hydrodynamic_radius_m
    if kappa_radius == 0.0:
        return 0.0
    elementary_charge_C = F / N_A
    return (
        charge_number
        * charge_number
        * elementary_charge_C
        * elementary_charge_C
        * kappa_m_inv
        / (
            STOKES_SPHERE_DRAG_FACTOR
            * math.pi
            * EPS_0
            * relative_dielectric
            * local_diffusivity_m2_s
            * (1.0 + kappa_radius)
        )
    )


def _debye_kappa_inv_m(
    charge_weighted_concentration_mol_m3: float,
    relative_dielectric: float,
    temperature_K: float,
) -> float:
    _assert_nonnegative_finite(
        charge_weighted_concentration_mol_m3,
        "charge_weighted_concentration_mol_m3",
    )
    if charge_weighted_concentration_mol_m3 == 0.0:
        return math.inf
    kappa_squared_m_inv2 = (
        F
        * F
        * charge_weighted_concentration_mol_m3
        / (EPS_0 * relative_dielectric * R * temperature_K)
    )
    _assert_positive_finite(kappa_squared_m_inv2, "kappa_squared_m_inv2")
    return 1.0 / math.sqrt(kappa_squared_m_inv2)


def _sphere_volume_m3(radius_m: float) -> float:
    _assert_positive_finite(radius_m, "sphere_radius_m")
    return (
        SPHERE_VOLUME_NUMERATOR
        / SPHERE_VOLUME_DENOMINATOR
        * math.pi
        * radius_m
        * radius_m
        * radius_m
    )


def _validate_solver(solver: str) -> None:
    if solver not in SUPPORTED_ION_ATMOSPHERE_SOLVERS:
        raise ValueError(
            "Unsupported ion-atmosphere solver "
            f"{solver!r}; supported solvers are {SUPPORTED_ION_ATMOSPHERE_SOLVERS}"
        )


def _validate_bulk_solver(solver: str) -> None:
    if solver not in SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS:
        raise ValueError(
            "Unsupported bulk ion-atmosphere solver "
            f"{solver!r}; supported solvers are {SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS}"
        )


def _validate_bulk_resistance_matrix(
    resistance_matrix_kg_s: np.ndarray,
    context: str,
) -> None:
    if resistance_matrix_kg_s.ndim != 2:
        raise ValueError(f"{context} must be a matrix")
    if resistance_matrix_kg_s.shape[0] != resistance_matrix_kg_s.shape[1]:
        raise ValueError(f"{context} must be square")
    if not np.all(np.isfinite(resistance_matrix_kg_s)):
        raise ValueError(f"{context} contains non-finite values")
    if not np.allclose(resistance_matrix_kg_s, resistance_matrix_kg_s.T):
        raise ValueError(f"{context} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(resistance_matrix_kg_s)
    if float(np.min(eigenvalues)) < 0.0:
        raise ValueError(f"{context} must be positive semidefinite")


def _require_charge(
    values: Mapping[str, int],
    key: str,
) -> int:
    if key not in values:
        raise KeyError(f"carrier_charges missing {key}")
    charge_number = int(values[key])
    if charge_number == 0:
        raise ValueError(f"carrier_charges.{key} must be nonzero")
    return charge_number


def _require_positive_finite(
    values: Mapping[str, float],
    key: str,
    context: str,
) -> float:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    value = float(values[key])
    _assert_positive_finite(value, f"{context}.{key}")
    return value


def _require_nonnegative_finite(
    values: Mapping[str, float],
    key: str,
    context: str,
) -> float:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    value = float(values[key])
    _assert_nonnegative_finite(value, f"{context}.{key}")
    return value


def _assert_positive_finite(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{context} must be a positive finite number, got {value}")


def _assert_nonnegative_finite(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{context} must be a non-negative finite number, got {value}")


# ---- generic_speciation.py ----

STANDARD_STATE_CONCENTRATION_MOL_M3 = 1000.0  # Unit conversion: 1 mol/L standard state in mol/m^3.
ANGSTROM_TO_M = 1.0e-10  # Unit conversion: angstrom to meter.
CUBIC_ANGSTROM_TO_CUBIC_M = 1.0e-30  # Unit conversion: A^3 to m^3.
COULOMB_DENOMINATOR_FACTOR = 4.0  # Electrostatic denominator: 4*pi*epsilon0*epsilon*r.
BORN_DENOMINATOR_FACTOR = 2.0 * COULOMB_DENOMINATOR_FACTOR  # Born denominator is twice 4*pi*epsilon0*r.
DESOLVATION_OCCLUSION_SURFACE_FACTOR = 4.0  # Spherical surface area factor for contact occlusion fraction.
PAIR_COORDINATION_AVERAGE_FACTOR = 0.5  # Mean of two component coordination affinities.
NEWTON_MAX_ITERATIONS = 80  # Numerical solver iteration cap for mass-balance equations.
NEWTON_LINE_SEARCH_BACKOFF = 0.5  # Numerical damping factor for positivity-preserving Newton steps.
NEWTON_MIN_STEP_FRACTION = 2.0 ** -40  # Numerical sentinel for failed line search.
MASS_BALANCE_TOLERANCE_FACTOR = math.sqrt(np.finfo(float).eps)  # Floating-point residual scale.
CONTACT_PAIR_CLUSTER_KIND = "contact_pair"
SOLVENT_SEPARATED_PAIR_CLUSTER_KIND = "solvent_separated_pair"
POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND = "positive_charged_triplet"
NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND = "negative_charged_triplet"
NEUTRAL_CLUSTER_KIND = "neutral_cluster"
HIGHER_CHARGED_CLUSTER_KIND = "higher_charged_cluster"
ACTIVITY_IONIC_STRENGTH_REFERENCE_MOL_M3 = STANDARD_STATE_CONCENTRATION_MOL_M3


@dataclass(frozen=True)
class MolecularSolventEnvironment:
    dielectric_constant: float
    viscosity_cP: float
    hard_sphere_volume_fraction: float
    temperature_K: float
    solvent_effective_radius_A: float
    mean_molecular_volume_A3: float


@dataclass(frozen=True)
class IonComponent:
    species_name: str
    charge_number: int
    analytical_concentration_M: float
    descriptor: MolecularSpeciesDescriptor


@dataclass(frozen=True)
class ClusterChargedCenter:
    species_name: str
    charge_number: int
    position_A: tuple[float, float, float]


@dataclass(frozen=True)
class ClusterStateTemplate:
    label: str
    cluster_kind: str
    stoichiometry: Mapping[str, int]
    net_charge_number: int
    standard_free_energy_J_mol: float
    coulomb_J_mol: float
    desolvation_J_mol: float
    coordination_J_mol: float
    steric_J_mol: float
    entropy_J_mol: float
    standard_state_correction_J_mol: float
    activity_reference_J_mol: float
    geometry: tuple[ClusterChargedCenter, ...]
    orientation_count: int
    hydrodynamic_radius_A: float
    molecular_volume_A3: float


@dataclass(frozen=True)
class _ClusterFreeEnergyTerms:
    standard_free_energy_J_mol: float
    coulomb_J_mol: float
    desolvation_J_mol: float
    coordination_J_mol: float
    steric_J_mol: float
    entropy_J_mol: float
    standard_state_correction_J_mol: float
    activity_reference_J_mol: float


@dataclass(frozen=True)
class _PairFreeEnergyTerms:
    coulomb_J_mol: float
    desolvation_J_mol: float
    coordination_J_mol: float


@dataclass(frozen=True)
class ClusterEnumerationOptions:
    max_cluster_ion_count: int
    primitive_parameters: ConductivityPrimitiveParameterSet


@dataclass(frozen=True)
class GenericSpeciationResult:
    components: tuple[IonComponent, ...]
    cluster_templates: tuple[ClusterStateTemplate, ...]
    free_component_concentrations_mol_m3: Mapping[str, float]
    cluster_concentrations_mol_m3: Mapping[str, float]
    mass_balance_residual_mol_m3: float


def build_cluster_state_templates(
    components: tuple[IonComponent, ...],
    solvent_environment: MolecularSolventEnvironment,
    options: ClusterEnumerationOptions,
) -> tuple[ClusterStateTemplate, ...]:
    _validate_solvent_environment(solvent_environment)
    validate_conductivity_primitive_parameters(options.primitive_parameters)
    if options.max_cluster_ion_count < 1:
        raise ValueError("max_cluster_ion_count must be positive")
    if options.max_cluster_ion_count < 2:
        return tuple()
    templates: list[ClusterStateTemplate] = []
    for stoichiometric_counts in _cluster_stoichiometric_counts(
        len(components),
        options.max_cluster_ion_count,
    ):
        if not _contains_opposite_charges(components, stoichiometric_counts):
            continue
        for cluster_kind in _cluster_kinds_for_stoichiometry(
            components,
            stoichiometric_counts,
        ):
            templates.append(
                _stoichiometric_cluster_template(
                    components,
                    stoichiometric_counts,
                    cluster_kind,
                    solvent_environment,
                    options.primitive_parameters,
                )
            )
    return tuple(templates)


def solve_generic_mass_balance(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> GenericSpeciationResult:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_solvent_environment(solvent_environment)
    if not components:
        return GenericSpeciationResult(
            components=tuple(),
            cluster_templates=tuple(cluster_templates),
            free_component_concentrations_mol_m3={},
            cluster_concentrations_mol_m3={},
            mass_balance_residual_mol_m3=0.0,
        )
    component_names = tuple(component.species_name for component in components)
    if len(set(component_names)) != len(component_names):
        raise ValueError("component species names must be unique")
    total_concentrations = np.asarray(
        [
            _positive_float(
                component.analytical_concentration_M,
                f"{component.species_name}.analytical_concentration_M",
            )
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            for component in components
        ],
        dtype=float,
    )
    if not cluster_templates:
        return GenericSpeciationResult(
            components=tuple(components),
            cluster_templates=tuple(),
            free_component_concentrations_mol_m3={
                component.species_name: float(total_concentrations[index])
                for index, component in enumerate(components)
            },
            cluster_concentrations_mol_m3={},
            mass_balance_residual_mol_m3=0.0,
        )
    free_concentrations = _solve_free_concentrations(
        components,
        cluster_templates,
        total_concentrations,
        solvent_environment,
        primitive_parameters,
    )
    cluster_concentrations = _cluster_concentrations(
        components,
        cluster_templates,
        free_concentrations,
        solvent_environment,
        primitive_parameters,
    )
    residual = _mass_balance_residual(
        components,
        cluster_templates,
        free_concentrations,
        cluster_concentrations,
        total_concentrations,
    )
    return GenericSpeciationResult(
        components=tuple(components),
        cluster_templates=tuple(cluster_templates),
        free_component_concentrations_mol_m3={
            component.species_name: float(free_concentrations[index])
            for index, component in enumerate(components)
        },
        cluster_concentrations_mol_m3=cluster_concentrations,
        mass_balance_residual_mol_m3=float(np.max(np.abs(residual))),
    )


def cluster_activity_correction_J_mol(
    components: tuple[IonComponent, ...],
    cluster_template: ClusterStateTemplate,
    free_component_concentrations_mol_m3: Mapping[str, float],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_solvent_environment(solvent_environment)
    component_index_by_name = {
        component.species_name: index for index, component in enumerate(components)
    }
    free_concentration_array = np.asarray(
        [
            _positive_float(
                free_component_concentrations_mol_m3[component.species_name],
                f"{component.species_name}.free_concentration_mol_m3",
            )
            for component in components
        ],
        dtype=float,
    )
    activity_log_factor = _cluster_activity_log_factor(
        components,
        cluster_template,
        free_concentration_array,
        component_index_by_name,
        solvent_environment,
        primitive_parameters,
    )
    return float(
        -R * solvent_environment.temperature_K * activity_log_factor
    )


def _cluster_stoichiometric_counts(
    component_count: int,
    max_cluster_ion_count: int,
) -> tuple[tuple[int, ...], ...]:
    if component_count <= 0:
        return tuple()
    counts: list[tuple[int, ...]] = []
    current_counts = [0 for _component_index in range(component_count)]
    _append_cluster_stoichiometric_counts(
        component_count,
        max_cluster_ion_count,
        0,
        current_counts,
        counts,
    )
    return tuple(counts)


def _append_cluster_stoichiometric_counts(
    component_count: int,
    max_cluster_ion_count: int,
    component_index: int,
    current_counts: list[int],
    counts: list[tuple[int, ...]],
) -> None:
    if component_index == component_count:
        total_ion_count = sum(current_counts)
        if total_ion_count >= 2 and total_ion_count <= max_cluster_ion_count:
            counts.append(tuple(current_counts))
        return
    current_total_count = sum(current_counts)
    maximum_count_for_component = max_cluster_ion_count - current_total_count
    for component_count_value in range(maximum_count_for_component + 1):
        current_counts[component_index] = component_count_value
        _append_cluster_stoichiometric_counts(
            component_count,
            max_cluster_ion_count,
            component_index + 1,
            current_counts,
            counts,
        )
    current_counts[component_index] = 0


def _contains_opposite_charges(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> bool:
    has_positive_charge = False
    has_negative_charge = False
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        if stoichiometric_count == 0:
            continue
        if component.charge_number > 0:
            has_positive_charge = True
        if component.charge_number < 0:
            has_negative_charge = True
    return has_positive_charge and has_negative_charge


def _cluster_kinds_for_stoichiometry(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> tuple[str, ...]:
    total_ion_count = sum(stoichiometric_counts)
    if total_ion_count == 2:
        return (
            CONTACT_PAIR_CLUSTER_KIND,
            SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
        )
    net_charge_number = _cluster_net_charge_number(
        components,
        stoichiometric_counts,
    )
    if net_charge_number == 0:
        return (NEUTRAL_CLUSTER_KIND,)
    if total_ion_count == 3 and net_charge_number > 0:
        return (POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,)
    if total_ion_count == 3 and net_charge_number < 0:
        return (NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,)
    return (HIGHER_CHARGED_CLUSTER_KIND,)


def _stoichiometric_cluster_template(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> ClusterStateTemplate:
    if len(components) != len(stoichiometric_counts):
        raise ValueError("stoichiometric count length must match components")
    stoichiometry = {
        component.species_name: int(stoichiometric_count)
        for component, stoichiometric_count in zip(components, stoichiometric_counts)
        if stoichiometric_count > 0
    }
    if len(stoichiometry) < 2:
        raise ValueError("cluster template requires at least two species")
    geometry = _cluster_geometry(
        components,
        stoichiometric_counts,
        cluster_kind,
        solvent_environment,
    )
    free_energy_terms = _cluster_standard_free_energy_terms(
        components,
        stoichiometric_counts,
        cluster_kind,
        geometry,
        solvent_environment,
        primitive_parameters,
    )
    return ClusterStateTemplate(
        label=_cluster_label(components, stoichiometric_counts, cluster_kind),
        cluster_kind=cluster_kind,
        stoichiometry=stoichiometry,
        net_charge_number=_cluster_net_charge_number(
            components,
            stoichiometric_counts,
        ),
        standard_free_energy_J_mol=free_energy_terms.standard_free_energy_J_mol,
        coulomb_J_mol=free_energy_terms.coulomb_J_mol,
        desolvation_J_mol=free_energy_terms.desolvation_J_mol,
        coordination_J_mol=free_energy_terms.coordination_J_mol,
        steric_J_mol=free_energy_terms.steric_J_mol,
        entropy_J_mol=free_energy_terms.entropy_J_mol,
        standard_state_correction_J_mol=(
            free_energy_terms.standard_state_correction_J_mol
        ),
        activity_reference_J_mol=free_energy_terms.activity_reference_J_mol,
        geometry=geometry,
        orientation_count=1,
        hydrodynamic_radius_A=_cluster_hydrodynamic_radius_A(
            components,
            stoichiometric_counts,
            primitive_parameters,
        ),
        molecular_volume_A3=_cluster_molecular_volume_A3(
            components,
            stoichiometric_counts,
        ),
    )


def _cluster_geometry(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
    solvent_environment: MolecularSolventEnvironment,
) -> tuple[ClusterChargedCenter, ...]:
    expanded_components: list[IonComponent] = []
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        for _species_copy_index in range(stoichiometric_count):
            expanded_components.append(component)
    if len(expanded_components) < 2:
        raise ValueError("cluster geometry requires at least two charged centers")
    center_positions_A: list[float] = [0.0]
    for previous_component, next_component in zip(
        expanded_components[:-1],
        expanded_components[1:],
    ):
        separation_A = (
            previous_component.descriptor.cavity_radius_A
            + next_component.descriptor.cavity_radius_A
            + _cluster_separation_extra_A(
                previous_component,
                next_component,
                cluster_kind,
                solvent_environment,
            )
        )
        center_positions_A.append(center_positions_A[-1] + separation_A)
    center_mean_A = math.fsum(center_positions_A) / len(center_positions_A)
    geometry: list[ClusterChargedCenter] = []
    for component, center_position_A in zip(expanded_components, center_positions_A):
        geometry.append(
            ClusterChargedCenter(
                species_name=component.species_name,
                charge_number=component.charge_number,
                position_A=(center_position_A - center_mean_A, 0.0, 0.0),
            )
        )
    return tuple(geometry)


def _cluster_separation_extra_A(
    previous_component: IonComponent,
    next_component: IonComponent,
    cluster_kind: str,
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    if cluster_kind != SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
        return 0.0
    if previous_component.charge_number * next_component.charge_number >= 0:
        return 0.0
    return 2.0 * _positive_float(
        solvent_environment.solvent_effective_radius_A,
        "solvent_effective_radius_A",
    )


def _cluster_standard_free_energy_terms(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
    geometry: tuple[ClusterChargedCenter, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> _ClusterFreeEnergyTerms:
    component_by_name = {component.species_name: component for component in components}
    coulomb_J_mol = 0.0
    desolvation_J_mol = 0.0
    coordination_J_mol = 0.0
    for first_index, first_center in enumerate(geometry):
        for second_center in geometry[first_index + 1:]:
            first_component = component_by_name[first_center.species_name]
            second_component = component_by_name[second_center.species_name]
            center_distance_A = _center_distance_A(
                first_center.position_A,
                second_center.position_A,
            )
            pair_terms = _pair_interaction_free_energy_terms(
                first_component,
                second_component,
                center_distance_A,
                solvent_environment,
                primitive_parameters,
            )
            coulomb_J_mol += pair_terms.coulomb_J_mol
            desolvation_J_mol += pair_terms.desolvation_J_mol
            coordination_J_mol += pair_terms.coordination_J_mol
    total_ion_count = sum(stoichiometric_counts)
    steric_J_mol = (
        primitive_parameters.steric_free_energy_scale
        * R
        * solvent_environment.temperature_K
        * solvent_environment.hard_sphere_volume_fraction
        * _cluster_molecular_volume_A3(components, stoichiometric_counts)
        * CUBIC_ANGSTROM_TO_CUBIC_M
        * N_A
        * STANDARD_STATE_CONCENTRATION_MOL_M3
        * total_ion_count
    )
    entropy_J_mol = (
        primitive_parameters.cluster_entropy_penalty_scale
        * R
        * solvent_environment.temperature_K
        * (total_ion_count - 1)
    )
    standard_state_correction_J_mol = (
        _cluster_crowding_stabilization_J_mol(
            components,
            stoichiometric_counts,
            solvent_environment,
            primitive_parameters,
        )
        + _cluster_topology_standard_state_correction_J_mol(
            components,
            stoichiometric_counts,
            cluster_kind,
            solvent_environment,
            primitive_parameters,
        )
    )
    activity_reference_J_mol = _cluster_activity_reference_J_mol(
        components,
        stoichiometric_counts,
        solvent_environment,
        primitive_parameters,
    )
    standard_free_energy_J_mol = (
        coulomb_J_mol
        + desolvation_J_mol
        + coordination_J_mol
        + steric_J_mol
        + entropy_J_mol
        + standard_state_correction_J_mol
    )
    return _ClusterFreeEnergyTerms(
        standard_free_energy_J_mol=float(standard_free_energy_J_mol),
        coulomb_J_mol=float(coulomb_J_mol),
        desolvation_J_mol=float(desolvation_J_mol),
        coordination_J_mol=float(coordination_J_mol),
        steric_J_mol=float(steric_J_mol),
        entropy_J_mol=float(entropy_J_mol),
        standard_state_correction_J_mol=float(standard_state_correction_J_mol),
        activity_reference_J_mol=float(activity_reference_J_mol),
    )


def _cluster_topology_standard_state_correction_J_mol(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    total_ion_count = sum(stoichiometric_counts)
    net_charge_number = abs(
        _cluster_net_charge_number(components, stoichiometric_counts)
    )
    log_equilibrium_offset = (
        _cluster_kind_logK_offset(cluster_kind, primitive_parameters)
        + (total_ion_count - 2)
        * primitive_parameters.cluster_order_logK_slope
        + net_charge_number
        * primitive_parameters.cluster_charge_magnitude_logK_slope
    )
    return float(
        -R * solvent_environment.temperature_K * log_equilibrium_offset
    )


def _cluster_activity_reference_J_mol(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    reference_ionic_strength_ratio = (
        ACTIVITY_IONIC_STRENGTH_REFERENCE_MOL_M3
        / STANDARD_STATE_CONCENTRATION_MOL_M3
    )
    component_activity_log_sum = 0.0
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        if stoichiometric_count <= 0:
            continue
        component_activity_log_sum += (
            stoichiometric_count
            * _species_activity_ln_gamma(
                charge_number=component.charge_number,
                activity_size_radius_A=component.descriptor.cavity_radius_A,
                molecular_volume_A3=component.descriptor.molecular_volume_A3,
                ionic_strength_ratio=reference_ionic_strength_ratio,
                solvent_environment=solvent_environment,
                primitive_parameters=primitive_parameters,
            )
        )
    cluster_activity_log_gamma = _species_activity_ln_gamma(
        charge_number=_cluster_net_charge_number(components, stoichiometric_counts),
        activity_size_radius_A=_cluster_hydrodynamic_radius_A(
            components,
            stoichiometric_counts,
            primitive_parameters,
        ),
        molecular_volume_A3=_cluster_molecular_volume_A3(
            components,
            stoichiometric_counts,
        ),
        ionic_strength_ratio=reference_ionic_strength_ratio,
        solvent_environment=solvent_environment,
        primitive_parameters=primitive_parameters,
    )
    activity_log_factor = (
        component_activity_log_sum
        - primitive_parameters.cluster_activity_scale * cluster_activity_log_gamma
    )
    return float(-R * solvent_environment.temperature_K * activity_log_factor)


def _cluster_kind_logK_offset(
    cluster_kind: str,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    log_offset_by_cluster_kind = {
        CONTACT_PAIR_CLUSTER_KIND: (
            primitive_parameters.pair_logK_offset
            + primitive_parameters.contact_pair_logK_offset
        ),
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND: (
            primitive_parameters.pair_logK_offset
            + primitive_parameters.solvent_separated_pair_logK_offset
        ),
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND: (
            primitive_parameters.positive_charged_triplet_logK_offset
        ),
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND: (
            primitive_parameters.negative_charged_triplet_logK_offset
        ),
        NEUTRAL_CLUSTER_KIND: primitive_parameters.neutral_cluster_logK_offset,
        HIGHER_CHARGED_CLUSTER_KIND: (
            primitive_parameters.higher_charged_cluster_logK_offset
        ),
    }
    if cluster_kind not in log_offset_by_cluster_kind:
        raise ValueError(f"unknown cluster kind {cluster_kind}")
    return log_offset_by_cluster_kind[cluster_kind]


def _pair_interaction_free_energy_terms(
    first_component: IonComponent,
    second_component: IonComponent,
    center_distance_A: float,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> _PairFreeEnergyTerms:
    first_descriptor = first_component.descriptor
    second_descriptor = second_component.descriptor
    contact_distance_m = _positive_float(
        center_distance_A,
        "center_distance_A",
    ) * ANGSTROM_TO_M
    charge_cloud_distance_m = math.sqrt(
        contact_distance_m ** 2
        + (first_descriptor.charge_cloud_radius_A * ANGSTROM_TO_M) ** 2
        + (second_descriptor.charge_cloud_radius_A * ANGSTROM_TO_M) ** 2
    )
    coulomb_energy_J_mol = (
        N_A
        * first_component.charge_number
        * second_component.charge_number
        * E_CHARGE ** 2
        / (
            COULOMB_DENOMINATOR_FACTOR
            * math.pi
            * EPS_0
            * solvent_environment.dielectric_constant
            * charge_cloud_distance_m
        )
    )
    coordination_energy_J_mol = 0.0
    if first_component.charge_number * second_component.charge_number < 0:
        coordination_energy_J_mol = (
            -PAIR_COORDINATION_AVERAGE_FACTOR
            * (
                first_descriptor.coordination_affinity_J_mol
                + second_descriptor.coordination_affinity_J_mol
            )
        )
    desolvation_energy_J_mol = _pair_desolvation_penalty_J_mol(
        first_component,
        second_component,
        center_distance_A,
        solvent_environment,
    )
    return _PairFreeEnergyTerms(
        coulomb_J_mol=float(
            primitive_parameters.coulomb_scale * coulomb_energy_J_mol
        ),
        desolvation_J_mol=float(
            primitive_parameters.desolvation_scale * desolvation_energy_J_mol
        ),
        coordination_J_mol=float(
            primitive_parameters.coordination_scale * coordination_energy_J_mol
        ),
    )


def _cluster_label(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    cluster_kind: str,
) -> str:
    label_parts = []
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        if stoichiometric_count == 0:
            continue
        label_parts.append(f"{component.species_name}^{stoichiometric_count}")
    return "cluster:" + cluster_kind + ":" + ":".join(label_parts)


def _cluster_net_charge_number(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> int:
    net_charge_number = 0
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        net_charge_number += component.charge_number * stoichiometric_count
    return int(net_charge_number)


def _cluster_molecular_volume_A3(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> float:
    cluster_volume_A3 = 0.0
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        cluster_volume_A3 += (
            stoichiometric_count
            * component.descriptor.molecular_volume_A3
        )
    return _positive_float(cluster_volume_A3, "cluster_volume_A3")


def _cluster_hydrodynamic_radius_A(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    radius_cubed_sum = 0.0
    for component, stoichiometric_count in zip(components, stoichiometric_counts):
        hydrodynamic_radius_A = component.descriptor.hydrodynamic_radius_A
        radius_cubed_sum += (
            stoichiometric_count * hydrodynamic_radius_A ** 3
        )
    return (
        primitive_parameters.cluster_hydrodynamic_radius_scale
        * _positive_float(radius_cubed_sum, "cluster_radius_cubed_sum") ** (
            1.0 / 3.0
        )
    )


def _cluster_crowding_stabilization_J_mol(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    total_ion_count = sum(stoichiometric_counts)
    if total_ion_count <= 1:
        return 0.0
    ionic_strength_ratio = _analytical_ionic_strength_ratio(components)
    charge_cloud_compactness_ratio = _cluster_charge_cloud_compactness_ratio(
        components,
        stoichiometric_counts,
    )
    stabilization_magnitude_J_mol = (
        primitive_parameters.association_crowding_stabilization_scale
        * R
        * solvent_environment.temperature_K
        * (total_ion_count - 1)
        * (
            ionic_strength_ratio
            ** primitive_parameters.association_crowding_ionic_strength_exponent
        )
        * (
            charge_cloud_compactness_ratio
            ** primitive_parameters.association_crowding_charge_density_exponent
        )
    )
    return -_positive_float(
        stabilization_magnitude_J_mol,
        "association_crowding_stabilization_magnitude_J_mol",
    )


def _analytical_ionic_strength_ratio(
    components: tuple[IonComponent, ...],
) -> float:
    ionic_strength_mol_m3 = 0.0
    for component in components:
        analytical_concentration_mol_m3 = (
            _positive_float(
                component.analytical_concentration_M,
                f"{component.species_name}.analytical_concentration_M",
            )
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        ionic_strength_mol_m3 += (
            analytical_concentration_mol_m3
            * component.charge_number
            * component.charge_number
        )
    return _positive_float(
        ionic_strength_mol_m3 / STANDARD_STATE_CONCENTRATION_MOL_M3,
        "analytical_ionic_strength_ratio",
    )


def _cluster_charge_cloud_compactness_ratio(
    components: tuple[IonComponent, ...],
    stoichiometric_counts: tuple[int, ...],
) -> float:
    cluster_charge_cloud_compactness = _charge_cloud_compactness_for_counts(
        components,
        stoichiometric_counts,
        "cluster_charge_cloud_compactness",
    )
    analytical_weight_counts = tuple(
        _positive_float(
            component.analytical_concentration_M,
            f"{component.species_name}.analytical_concentration_M",
        )
        for component in components
    )
    mixture_charge_cloud_compactness = _charge_cloud_compactness_for_counts(
        components,
        analytical_weight_counts,
        "mixture_charge_cloud_compactness",
    )
    return _positive_float(
        cluster_charge_cloud_compactness / mixture_charge_cloud_compactness,
        "cluster_charge_cloud_compactness_ratio",
    )


def _charge_cloud_compactness_for_counts(
    components: tuple[IonComponent, ...],
    component_weights: tuple[float, ...],
    context: str,
) -> float:
    if len(components) != len(component_weights):
        raise ValueError(f"{context} weights must match components")
    weighted_charge_number_sum = 0.0
    weighted_charge_cloud_volume_A3 = 0.0
    for component, component_weight in zip(components, component_weights):
        nonnegative_component_weight = _nonnegative_float(
            component_weight,
            f"{context}.component_weight",
        )
        if nonnegative_component_weight == 0.0:
            continue
        charge_cloud_radius_A = _positive_float(
            component.descriptor.charge_cloud_radius_A,
            f"{context}.{component.species_name}.charge_cloud_radius_A",
        )
        weighted_charge_number_sum += (
            nonnegative_component_weight * abs(component.charge_number)
        )
        weighted_charge_cloud_volume_A3 += (
            nonnegative_component_weight
            * charge_cloud_radius_A
            * charge_cloud_radius_A
            * charge_cloud_radius_A
        )
    return _positive_float(
        weighted_charge_number_sum / weighted_charge_cloud_volume_A3,
        context,
    )


def _center_distance_A(
    first_position_A: tuple[float, float, float],
    second_position_A: tuple[float, float, float],
) -> float:
    squared_distance = 0.0
    for first_coordinate, second_coordinate in zip(first_position_A, second_position_A):
        difference = first_coordinate - second_coordinate
        squared_distance += difference ** 2
    return _positive_float(math.sqrt(squared_distance), "center_distance_A")


def _pair_desolvation_penalty_J_mol(
    first_component: IonComponent,
    second_component: IonComponent,
    center_distance_A: float,
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    first_descriptor = first_component.descriptor
    second_descriptor = second_component.descriptor
    first_occlusion_fraction = _contact_occlusion_fraction(
        occluding_radius_A=second_descriptor.cavity_radius_A,
        center_separation_A=center_distance_A,
    )
    second_occlusion_fraction = _contact_occlusion_fraction(
        occluding_radius_A=first_descriptor.cavity_radius_A,
        center_separation_A=center_distance_A,
    )
    first_born_magnitude_J_mol = abs(
        _born_solvation_energy_J_mol(
            charge_number=first_component.charge_number,
            born_solvation_radius_A=first_descriptor.born_solvation_radius_A,
            solvent_environment=solvent_environment,
        )
    )
    second_born_magnitude_J_mol = abs(
        _born_solvation_energy_J_mol(
            charge_number=second_component.charge_number,
            born_solvation_radius_A=second_descriptor.born_solvation_radius_A,
            solvent_environment=solvent_environment,
        )
    )
    return float(
        first_occlusion_fraction * first_born_magnitude_J_mol
        + second_occlusion_fraction * second_born_magnitude_J_mol
    )


def _contact_occlusion_fraction(
    occluding_radius_A: float,
    center_separation_A: float,
) -> float:
    radius_A = _positive_float(occluding_radius_A, "occluding_radius_A")
    separation_A = _positive_float(center_separation_A, "center_separation_A")
    occlusion_fraction = (
        radius_A ** 2
        / (
            DESOLVATION_OCCLUSION_SURFACE_FACTOR
            * separation_A ** 2
        )
    )
    if occlusion_fraction >= 1.0:
        raise ValueError(
            "contact occlusion fraction must remain below one for pair geometry"
        )
    return float(occlusion_fraction)


def _born_solvation_energy_J_mol(
    charge_number: int,
    born_solvation_radius_A: float,
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    radius_m = _positive_float(
        born_solvation_radius_A,
        "born_solvation_radius_A",
    ) * ANGSTROM_TO_M
    dielectric = _positive_float(
        solvent_environment.dielectric_constant,
        "dielectric_constant",
    )
    charge_squared = charge_number * charge_number
    return float(
        -N_A
        * charge_squared
        * E_CHARGE ** 2
        * (1.0 - 1.0 / dielectric)
        / (
            BORN_DENOMINATOR_FACTOR
            * math.pi
            * EPS_0
            * radius_m
        )
    )


def _solve_free_concentrations(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    total_concentrations: np.ndarray,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> np.ndarray:
    free_concentrations = np.asarray(total_concentrations, dtype=float)
    tolerance = MASS_BALANCE_TOLERANCE_FACTOR * max(
        1.0,
        float(np.max(total_concentrations)),
    )
    for _iteration_index in range(NEWTON_MAX_ITERATIONS):
        cluster_concentrations = _cluster_concentrations_array(
            components,
            cluster_templates,
            free_concentrations,
            solvent_environment,
            primitive_parameters,
        )
        residual = _mass_balance_residual_array(
            components,
            cluster_templates,
            free_concentrations,
            cluster_concentrations,
            total_concentrations,
        )
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= tolerance:
            return free_concentrations
        jacobian = _mass_balance_jacobian(
            components,
            cluster_templates,
            free_concentrations,
            total_concentrations,
            solvent_environment,
            primitive_parameters,
        )
        newton_step = np.linalg.solve(jacobian, residual)
        step_fraction = 1.0
        accepted_step = False
        while step_fraction >= NEWTON_MIN_STEP_FRACTION:
            trial_free_concentrations = free_concentrations - step_fraction * newton_step
            if np.all(trial_free_concentrations > 0.0):
                trial_cluster_concentrations = _cluster_concentrations_array(
                    components,
                    cluster_templates,
                    trial_free_concentrations,
                    solvent_environment,
                    primitive_parameters,
                )
                trial_residual = _mass_balance_residual_array(
                    components,
                    cluster_templates,
                    trial_free_concentrations,
                    trial_cluster_concentrations,
                    total_concentrations,
                )
                trial_norm = float(np.max(np.abs(trial_residual)))
                if trial_norm < residual_norm:
                    free_concentrations = trial_free_concentrations
                    accepted_step = True
                    break
            step_fraction *= NEWTON_LINE_SEARCH_BACKOFF
        if not accepted_step:
            raise ValueError("generic mass-balance Newton solve failed to reduce residual")
    raise ValueError("generic mass-balance Newton solve exceeded iteration limit")


def _cluster_concentrations(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> Mapping[str, float]:
    cluster_concentration_array = _cluster_concentrations_array(
        components,
        cluster_templates,
        free_concentrations,
        solvent_environment,
        primitive_parameters,
    )
    return {
        template.label: float(cluster_concentration_array[index])
        for index, template in enumerate(cluster_templates)
    }


def _cluster_concentrations_array(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> np.ndarray:
    component_index_by_name = {
        component.species_name: index for index, component in enumerate(components)
    }
    concentrations: list[float] = []
    for template in cluster_templates:
        activity_log_factor = _cluster_activity_log_factor(
            components,
            template,
            free_concentrations,
            component_index_by_name,
            solvent_environment,
            primitive_parameters,
        )
        exponent = (
            -template.standard_free_energy_J_mol
            / (R * solvent_environment.temperature_K)
            + activity_log_factor
        )
        equilibrium_constant = math.exp(exponent)
        concentration = STANDARD_STATE_CONCENTRATION_MOL_M3 * equilibrium_constant
        for species_name, stoichiometric_count in template.stoichiometry.items():
            component_index = component_index_by_name[species_name]
            concentration *= (
                free_concentrations[component_index]
                / STANDARD_STATE_CONCENTRATION_MOL_M3
            ) ** stoichiometric_count
        concentrations.append(float(concentration))
    return np.asarray(concentrations, dtype=float)


def _cluster_activity_log_factor(
    components: tuple[IonComponent, ...],
    cluster_template: ClusterStateTemplate,
    free_concentrations: np.ndarray,
    component_index_by_name: Mapping[str, int],
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    ionic_strength_ratio = _free_ionic_strength_ratio(
        components,
        free_concentrations,
    )
    component_activity_log_sum = 0.0
    for species_name, stoichiometric_count in cluster_template.stoichiometry.items():
        component = components[component_index_by_name[species_name]]
        component_activity_log_sum += (
            stoichiometric_count
            * _species_activity_ln_gamma(
                charge_number=component.charge_number,
                activity_size_radius_A=component.descriptor.cavity_radius_A,
                molecular_volume_A3=component.descriptor.molecular_volume_A3,
                ionic_strength_ratio=ionic_strength_ratio,
                solvent_environment=solvent_environment,
                primitive_parameters=primitive_parameters,
            )
        )
    cluster_activity_log_gamma = _species_activity_ln_gamma(
        charge_number=cluster_template.net_charge_number,
        activity_size_radius_A=cluster_template.hydrodynamic_radius_A,
        molecular_volume_A3=cluster_template.molecular_volume_A3,
        ionic_strength_ratio=ionic_strength_ratio,
        solvent_environment=solvent_environment,
        primitive_parameters=primitive_parameters,
    )
    return float(
        component_activity_log_sum
        - primitive_parameters.cluster_activity_scale * cluster_activity_log_gamma
    )


def _free_ionic_strength_ratio(
    components: tuple[IonComponent, ...],
    free_concentrations: np.ndarray,
) -> float:
    ionic_strength_mol_m3 = 0.0
    for component, free_concentration_mol_m3 in zip(
        components,
        free_concentrations,
    ):
        ionic_strength_mol_m3 += (
            _positive_float(
                free_concentration_mol_m3,
                f"{component.species_name}.free_concentration_mol_m3",
            )
            * component.charge_number
            * component.charge_number
        )
    return _positive_float(
        ionic_strength_mol_m3 / ACTIVITY_IONIC_STRENGTH_REFERENCE_MOL_M3,
        "free_ionic_strength_ratio",
    )


def _species_activity_ln_gamma(
    charge_number: int,
    activity_size_radius_A: float,
    molecular_volume_A3: float,
    ionic_strength_ratio: float,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    square_root_ionic_strength = math.sqrt(
        _positive_float(ionic_strength_ratio, "ionic_strength_ratio")
    )
    size_radius_A = _positive_float(activity_size_radius_A, "activity_size_radius_A")
    debye_denominator = (
        1.0
        + primitive_parameters.activity_size_scale
        * size_radius_A
        * square_root_ionic_strength
    )
    debye_term = (
        -primitive_parameters.activity_debye_scale
        * charge_number
        * charge_number
        * square_root_ionic_strength
        / debye_denominator
    )
    hard_sphere_term = (
        primitive_parameters.activity_hard_sphere_scale
        * _positive_float(molecular_volume_A3, "activity_molecular_volume_A3")
        / _positive_float(
            solvent_environment.mean_molecular_volume_A3,
            "mean_molecular_volume_A3",
        )
        * _activity_packing_ratio(solvent_environment)
    )
    return float(debye_term + hard_sphere_term)


def _activity_packing_ratio(
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    hard_sphere_volume_fraction = _nonnegative_float(
        solvent_environment.hard_sphere_volume_fraction,
        "hard_sphere_volume_fraction",
    )
    if hard_sphere_volume_fraction >= 1.0:
        raise ValueError(
            "hard_sphere_volume_fraction must be below one for activity model"
        )
    return float(
        hard_sphere_volume_fraction / (1.0 - hard_sphere_volume_fraction)
    )


def _mass_balance_residual(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    cluster_concentrations: Mapping[str, float],
    total_concentrations: np.ndarray,
) -> np.ndarray:
    cluster_array = np.asarray(
        [
            cluster_concentrations[template.label]
            for template in cluster_templates
        ],
        dtype=float,
    )
    return _mass_balance_residual_array(
        components,
        cluster_templates,
        free_concentrations,
        cluster_array,
        total_concentrations,
    )


def _mass_balance_residual_array(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    cluster_concentrations: np.ndarray,
    total_concentrations: np.ndarray,
) -> np.ndarray:
    component_index_by_name = {
        component.species_name: index for index, component in enumerate(components)
    }
    reconstructed = np.asarray(free_concentrations, dtype=float).copy()
    for template_index, template in enumerate(cluster_templates):
        for species_name, stoichiometric_count in template.stoichiometry.items():
            component_index = component_index_by_name[species_name]
            reconstructed[component_index] += (
                stoichiometric_count * cluster_concentrations[template_index]
            )
    return reconstructed - total_concentrations


def _mass_balance_jacobian(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_concentrations: np.ndarray,
    total_concentrations: np.ndarray,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> np.ndarray:
    component_count = len(components)
    base_cluster_concentrations = _cluster_concentrations_array(
        components,
        cluster_templates,
        free_concentrations,
        solvent_environment,
        primitive_parameters,
    )
    base_residual = _mass_balance_residual_array(
        components,
        cluster_templates,
        free_concentrations,
        base_cluster_concentrations,
        total_concentrations,
    )
    jacobian = np.zeros((component_count, component_count), dtype=float)
    finite_difference_scale = math.sqrt(np.finfo(float).eps)
    for column_index in range(component_count):
        perturbation_mol_m3 = finite_difference_scale * max(
            1.0,
            abs(free_concentrations[column_index]),
        )
        trial_free_concentrations = np.asarray(free_concentrations, dtype=float).copy()
        trial_free_concentrations[column_index] += perturbation_mol_m3
        trial_cluster_concentrations = _cluster_concentrations_array(
            components,
            cluster_templates,
            trial_free_concentrations,
            solvent_environment,
            primitive_parameters,
        )
        trial_residual = _mass_balance_residual_array(
            components,
            cluster_templates,
            trial_free_concentrations,
            trial_cluster_concentrations,
            total_concentrations,
        )
        jacobian[:, column_index] = (
            trial_residual - base_residual
        ) / perturbation_mol_m3
    return jacobian


def _validate_solvent_environment(
    solvent_environment: MolecularSolventEnvironment,
) -> None:
    _positive_float(solvent_environment.dielectric_constant, "dielectric_constant")
    _positive_float(solvent_environment.viscosity_cP, "viscosity_cP")
    _nonnegative_float(
        solvent_environment.hard_sphere_volume_fraction,
        "hard_sphere_volume_fraction",
    )
    _positive_float(solvent_environment.temperature_K, "temperature_K")
    _positive_float(
        solvent_environment.solvent_effective_radius_A,
        "solvent_effective_radius_A",
    )
    _positive_float(
        solvent_environment.mean_molecular_volume_A3,
        "mean_molecular_volume_A3",
    )


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


# ---- molecular_electrolyte_mori_generator.py ----

CP_TO_PA_S = 1.0e-3  # Unit conversion: cP to Pa*s.
GRAMS_PER_LITER_PER_G_ML = 1000.0  # Unit conversion: g/mL to g/L.
GRAMS_PER_M3_PER_G_ML = 1.0e6  # Unit conversion: g/mL to g/m^3.
STOKES_DENOMINATOR_FACTOR = 6.0  # Stokes-Einstein sphere denominator: 6*pi*eta*r.
CARTESIAN_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)  # Cartesian unit vectors for isotropic translation events.
TRANSLATION_EVENT_SIGNS = (-1.0, 1.0)  # Symmetric plus/minus jump directions.
ION_ATMOSPHERE_SOLVER_DIAGONAL = "diagonal_pnp_stokes_l1_cell_experimental"
MINIMUM_CLUSTER_ION_COUNT = 2  # Molecular production must include cation-anion pair states.
GAUSSIAN_CHARGE_CLOUD_FORM_FACTOR_DENOMINATOR = 6.0  # Gaussian F_q(kappa,a_q)=exp(-(kappa*a_q)^2/6).
ISOTROPIC_SHAPE_FACTOR = 1.0  # Dimensionless reference: lambda_s=1 is isotropic.
TRANSPORT_STATE_CONCENTRATION_RESOLUTION_FACTOR = np.finfo(float).eps  # Markov-basis floor for zero-measure clusters.
TRANSPORT_ROLE_FREE_ION_CENTER = "free_ion_center"
TRANSPORT_ROLE_CONTACT_PAIR_CENTER = "contact_pair_center"
TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER = "solvent_separated_pair_center"
TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER = "charged_triplet_center"
TRANSPORT_ROLE_CLUSTER_COM_CENTER = "cluster_com_center"
TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER = "internal_polarization_center"
TRANSPORT_ROLE_NEUTRAL_CENTER = "neutral_center"


@dataclass(frozen=True)
class MolecularMixtureProperties:
    density_g_ml: float
    viscosity_cP: float
    dielectric_constant: float


@dataclass(frozen=True)
class MolecularElectrolyteRecipe:
    cations: Mapping[str, float]
    anions: Mapping[str, float]
    solvents: Mapping[str, float]
    additives: Mapping[str, float]
    temperature_K: float
    pressure_Pa: float
    mixture_properties: MolecularMixtureProperties


@dataclass(frozen=True)
class MolecularMoriOptions:
    max_cluster_ion_count: int
    max_packing_fraction: float
    free_volume_exponent: float
    translation_jump_length_multiplier: float
    primitive_parameters: ConductivityPrimitiveParameterSet


@dataclass(frozen=True)
class MolecularTransportCenter:
    label: str
    parent_cluster_label: str
    parent_cluster_kind: str
    concentration_mol_m3: float
    center_species_name: str
    center_charge_number: int
    center_index: int
    hydrodynamic_radius_A: float
    charge_cloud_radius_A: float
    molecular_volume_A3: float
    diffusion_m2_s: float
    local_obstruction_factor: float
    local_obstruction_diffusion_scale: float
    transport_role: str


@dataclass(frozen=True)
class MolecularIonAtmosphereDiagnostics:
    solver: str
    charged_carrier_count: int
    kappa_inv_m: float
    ionic_strength_mol_m3: float
    charge_cloud_form_factor_by_state: Mapping[str, float]
    friction_ratio_by_state: Mapping[str, float]
    zeta0_kg_s_by_state: Mapping[str, float]
    zeta_ep_kg_s_by_state: Mapping[str, float]
    zeta_rel_kg_s_by_state: Mapping[str, float]
    countercharge_relaxation_diffusivity_m2_s_by_state: Mapping[str, float]


@dataclass(frozen=True)
class _MolecularMixtureDescriptorState:
    hard_sphere_volume_fraction: float
    max_packing_fraction: float
    ionic_strength_mol_m3: float
    void_radius_A: float
    donor_number: float
    acceptor_number: float
    polarizability_volume_ratio: float
    solvation_obstruction_factor: float
    additive_solvation_obstruction_factor: float
    mean_anion_charge_cloud_radius_A: float
    anion_composition_entropy: float


@dataclass(frozen=True)
class _ChargeDensityReferenceEntry:
    concentration_mol_m3: float
    net_charge_number: int
    charge_cloud_radius_A: float


@dataclass(frozen=True)
class _TransportCenterConstructionContext:
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor]
    mixture_descriptor_state: _MolecularMixtureDescriptorState
    charge_density_reference_A_inv3: float
    solvent_environment: MolecularSolventEnvironment
    options: MolecularMoriOptions


@dataclass(frozen=True)
class MolecularAtmosphereMemoryPrimitive:
    state_label: str
    D_local_m2_s: float
    atmosphere_relaxation_diffusivity_m2_s: float
    jump_length_m: float
    k_capture_s_inv: float
    k_exit_s_inv: float
    atmosphere_coupling_fraction: float
    back_relaxation_probability: float
    mobile_concentration_mol_m3: float
    atmosphere_concentration_per_direction_mol_m3: float
    zeta0_kg_s: float
    zeta_ep_kg_s: float
    zeta_rel_kg_s: float


@dataclass(frozen=True)
class _AtmosphereTransportStateResult:
    transport_states: tuple[MolecularTransportCenter, ...]
    diagnostics: MolecularIonAtmosphereDiagnostics


@dataclass(frozen=True)
class _MarkovProcessConstruction:
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: np.ndarray
    events: tuple[MarkovAdditiveEvent, ...]
    memory_primitives: tuple[MolecularAtmosphereMemoryPrimitive, ...]


@dataclass(frozen=True)
class _MobileTransportStateIndex:
    transport_state: MolecularTransportCenter
    mobile_state_index: int
    mobile_concentration_mol_m3: float


@dataclass(frozen=True)
class _SolventSeparatedPairModeRateBudget:
    relative_rate_s_inv: float
    co_motion_rate_s_inv: float
    positive_residual_rate_s_inv: float
    negative_residual_rate_s_inv: float


@dataclass(frozen=True)
class MolecularMoriConductivityResult:
    sigma_mS_cm: float
    sigma_S_m: float
    markov_additive_result: MarkovAdditiveConductivityResult
    descriptors: Mapping[str, MolecularSpeciesDescriptor]
    solvent_environment: MolecularSolventEnvironment
    speciation: GenericSpeciationResult
    cluster_states: tuple[ClusterStateTemplate, ...]
    transport_states: tuple[MolecularTransportCenter, ...]
    markov_state_labels: tuple[str, ...]
    markov_state_concentrations_mol_m3: tuple[float, ...]
    events: tuple[MarkovAdditiveEvent, ...]
    ion_atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics
    atmosphere_memory_primitives: tuple[MolecularAtmosphereMemoryPrimitive, ...]
    mass_balance_residual_mol_m3: float
    detailed_balance_residual_mol_m3_s: float


def compute_molecular_electrolyte_conductivity(
    recipe: MolecularElectrolyteRecipe,
    species_inputs: Mapping[str, MolecularSpeciesInput],
    descriptor_backend: MolecularDescriptorBackend,
    options: MolecularMoriOptions,
) -> MolecularMoriConductivityResult:
    return _compute_molecular_electrolyte_conductivity(
        recipe,
        species_inputs,
        descriptor_backend,
        options,
        {},
    )


def compute_molecular_electrolyte_conductivity_with_diagnostic_cluster_shifts(
    recipe: MolecularElectrolyteRecipe,
    species_inputs: Mapping[str, MolecularSpeciesInput],
    descriptor_backend: MolecularDescriptorBackend,
    options: MolecularMoriOptions,
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
) -> MolecularMoriConductivityResult:
    return _compute_molecular_electrolyte_conductivity(
        recipe,
        species_inputs,
        descriptor_backend,
        options,
        diagnostic_cluster_standard_free_energy_shift_over_RT_by_label,
    )


def _compute_molecular_electrolyte_conductivity(
    recipe: MolecularElectrolyteRecipe,
    species_inputs: Mapping[str, MolecularSpeciesInput],
    descriptor_backend: MolecularDescriptorBackend,
    options: MolecularMoriOptions,
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
) -> MolecularMoriConductivityResult:
    _validate_recipe(recipe)
    _validate_options(options)
    descriptors = _describe_recipe_species(
        recipe,
        species_inputs,
        descriptor_backend,
    )
    solvent_environment = _molecular_solvent_environment(
        recipe,
        descriptors,
        options,
    )
    components = _ion_components(recipe, descriptors)
    _validate_ionic_charge_balance(components)
    cluster_templates = build_cluster_state_templates(
        components,
        solvent_environment,
        ClusterEnumerationOptions(
            max_cluster_ion_count=options.max_cluster_ion_count,
            primitive_parameters=options.primitive_parameters,
        ),
    )
    cluster_templates = _apply_cluster_standard_free_energy_shifts(
        cluster_templates,
        diagnostic_cluster_standard_free_energy_shift_over_RT_by_label,
        solvent_environment.temperature_K,
    )
    speciation = solve_generic_mass_balance(
        components,
        cluster_templates,
        solvent_environment,
        options.primitive_parameters,
    )
    local_transport_states = _molecular_transport_states(
        recipe,
        descriptors,
        speciation,
        solvent_environment,
        options,
    )
    atmosphere_transport_state_result = _apply_ion_atmosphere_to_transport_states(
        local_transport_states,
        solvent_environment,
        options,
    )
    transport_states = atmosphere_transport_state_result.transport_states
    markov_process = _markov_process_from_transport_states(
        transport_states,
        options,
        atmosphere_transport_state_result.diagnostics,
    )
    markov_result = compute_markov_additive_green_kubo_conductivity(
        MarkovAdditiveConductivityInput(
            state_labels=markov_process.state_labels,
            state_concentrations_mol_m3=markov_process.state_concentrations_mol_m3,
            events=markov_process.events,
            temperature_K=recipe.temperature_K,
        )
    )
    return MolecularMoriConductivityResult(
        sigma_mS_cm=markov_result.sigma_mS_cm,
        sigma_S_m=markov_result.sigma_S_m,
        markov_additive_result=markov_result,
        descriptors=descriptors,
        solvent_environment=solvent_environment,
        speciation=speciation,
        cluster_states=cluster_templates,
        transport_states=transport_states,
        markov_state_labels=markov_process.state_labels,
        markov_state_concentrations_mol_m3=tuple(
            float(concentration_mol_m3)
            for concentration_mol_m3 in markov_process.state_concentrations_mol_m3
        ),
        events=markov_process.events,
        ion_atmosphere_diagnostics=atmosphere_transport_state_result.diagnostics,
        atmosphere_memory_primitives=markov_process.memory_primitives,
        mass_balance_residual_mol_m3=speciation.mass_balance_residual_mol_m3,
        detailed_balance_residual_mol_m3_s=(
            markov_result.validation.detailed_balance_residual_mol_m3_s
        ),
    )


def _describe_recipe_species(
    recipe: MolecularElectrolyteRecipe,
    species_inputs: Mapping[str, MolecularSpeciesInput],
    descriptor_backend: MolecularDescriptorBackend,
) -> Mapping[str, MolecularSpeciesDescriptor]:
    recipe_species_names = _recipe_species_names(recipe)
    descriptors: dict[str, MolecularSpeciesDescriptor] = {}
    for species_name in recipe_species_names:
        if species_name not in species_inputs:
            raise ValueError(f"missing molecular species input for {species_name}")
        descriptors[species_name] = descriptor_backend.describe_species(
            species_inputs[species_name],
            recipe.temperature_K,
        )
    return descriptors


def _recipe_species_names(recipe: MolecularElectrolyteRecipe) -> tuple[str, ...]:
    species_names: list[str] = []
    for loading_map in (
        recipe.cations,
        recipe.anions,
        recipe.solvents,
        recipe.additives,
    ):
        for species_name in loading_map:
            if species_name not in species_names:
                species_names.append(species_name)
    if not species_names:
        raise ValueError("molecular electrolyte recipe must contain at least one species")
    return tuple(species_names)


def _molecular_solvent_environment(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    options: MolecularMoriOptions,
) -> MolecularSolventEnvironment:
    return MolecularSolventEnvironment(
        dielectric_constant=recipe.mixture_properties.dielectric_constant,
        viscosity_cP=recipe.mixture_properties.viscosity_cP,
        hard_sphere_volume_fraction=_hard_sphere_volume_fraction(
            recipe,
            descriptors,
        ),
        temperature_K=recipe.temperature_K,
        solvent_effective_radius_A=_mixture_effective_radius_A(
            recipe,
            descriptors,
        ),
        mean_molecular_volume_A3=_mixture_mean_molecular_volume_A3(
            recipe,
            descriptors,
        ),
    )


def _ion_components(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> tuple[IonComponent, ...]:
    components: list[IonComponent] = []
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        if descriptor.role != ROLE_CATION:
            raise ValueError(f"recipe cation {species_name} descriptor role mismatch")
        components.append(
            IonComponent(
                species_name=species_name,
                charge_number=descriptor.charge_number,
                analytical_concentration_M=_positive_float(
                    concentration_M,
                    f"{species_name}.concentration_M",
                ),
                descriptor=descriptor,
            )
        )
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        if descriptor.role != ROLE_ANION:
            raise ValueError(f"recipe anion {species_name} descriptor role mismatch")
        components.append(
            IonComponent(
                species_name=species_name,
                charge_number=descriptor.charge_number,
                analytical_concentration_M=_positive_float(
                    concentration_M,
                    f"{species_name}.concentration_M",
                ),
                descriptor=descriptor,
            )
        )
    return tuple(components)


def _validate_ionic_charge_balance(
    components: tuple[IonComponent, ...],
) -> None:
    net_charge_concentration_M = math.fsum(
        component.charge_number * component.analytical_concentration_M
        for component in components
    )
    charge_scale_M = math.fsum(
        abs(component.charge_number) * component.analytical_concentration_M
        for component in components
    )
    tolerance_M = MASS_BALANCE_TOLERANCE_FACTOR * max(1.0, charge_scale_M)
    if abs(net_charge_concentration_M) > tolerance_M:
        raise ValueError(
            "ionic recipe must be charge neutral; "
            f"net analytical charge concentration is {net_charge_concentration_M} M"
        )


def _apply_cluster_standard_free_energy_shifts(
    cluster_templates: tuple[ClusterStateTemplate, ...],
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
    temperature_K: float,
) -> tuple[ClusterStateTemplate, ...]:
    if not diagnostic_cluster_standard_free_energy_shift_over_RT_by_label:
        return cluster_templates
    known_cluster_labels = {
        cluster_template.label for cluster_template in cluster_templates
    }
    unknown_cluster_labels = tuple(
        sorted(
            cluster_label
            for cluster_label in diagnostic_cluster_standard_free_energy_shift_over_RT_by_label
            if cluster_label not in known_cluster_labels
        )
    )
    if unknown_cluster_labels:
        raise ValueError(
            "unknown cluster standard-free-energy shift labels "
            f"{unknown_cluster_labels}"
        )
    shifted_templates: list[ClusterStateTemplate] = []
    for cluster_template in cluster_templates:
        if cluster_template.label in diagnostic_cluster_standard_free_energy_shift_over_RT_by_label:
            raw_shift_over_RT = (
                diagnostic_cluster_standard_free_energy_shift_over_RT_by_label[
                    cluster_template.label
                ]
            )
        else:
            raw_shift_over_RT = 0.0
        shift_over_RT = _finite_float(
            raw_shift_over_RT,
            f"{cluster_template.label}.standard_free_energy_shift_over_RT",
        )
        if shift_over_RT == 0.0:
            shifted_templates.append(cluster_template)
            continue
        shift_J_mol = R * temperature_K * shift_over_RT
        shifted_templates.append(
            replace(
                cluster_template,
                standard_free_energy_J_mol=(
                    cluster_template.standard_free_energy_J_mol
                    + shift_J_mol
                ),
                standard_state_correction_J_mol=(
                    cluster_template.standard_state_correction_J_mol
                    + shift_J_mol
                ),
                activity_reference_J_mol=(
                    cluster_template.activity_reference_J_mol
                ),
            )
        )
    return tuple(shifted_templates)


def _molecular_transport_states(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    speciation: GenericSpeciationResult,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> tuple[MolecularTransportCenter, ...]:
    states: list[MolecularTransportCenter] = []
    component_descriptor_by_name = {
        component.species_name: component.descriptor
        for component in speciation.components
    }
    mixture_descriptor_state = _molecular_mixture_descriptor_state(
        recipe,
        descriptors,
        solvent_environment,
        options,
    )
    charge_density_reference_A_inv3 = _charge_density_reference_A_inv3(
        speciation,
        component_descriptor_by_name,
        options,
    )
    transport_context = _TransportCenterConstructionContext(
        component_descriptor_by_name=component_descriptor_by_name,
        mixture_descriptor_state=mixture_descriptor_state,
        charge_density_reference_A_inv3=charge_density_reference_A_inv3,
        solvent_environment=solvent_environment,
        options=options,
    )
    concentration_resolution_mol_m3 = (
        _transport_state_concentration_resolution_mol_m3(speciation)
    )
    for component in speciation.components:
        descriptor = component.descriptor
        concentration_mol_m3 = speciation.free_component_concentrations_mol_m3[
            component.species_name
        ]
        states.append(
            _transport_state_from_descriptor(
                label=f"free:{component.species_name}",
                parent_cluster_label=f"free:{component.species_name}",
                parent_cluster_kind=TRANSPORT_ROLE_FREE_ION_CENTER,
                concentration_mol_m3=concentration_mol_m3,
                center_species_name=component.species_name,
                center_charge_number=component.charge_number,
                center_index=0,
                descriptor=descriptor,
                hydrodynamic_radius_scale=_free_ion_hydrodynamic_radius_scale(
                    component,
                    options,
                ),
                transport_context=transport_context,
                transport_role=TRANSPORT_ROLE_FREE_ION_CENTER,
            )
        )
    for cluster_template in speciation.cluster_templates:
        concentration_mol_m3 = speciation.cluster_concentrations_mol_m3[
            cluster_template.label
        ]
        if concentration_mol_m3 <= concentration_resolution_mol_m3:
            continue
        states.extend(
            _cluster_transport_centers(
                cluster_template,
                concentration_mol_m3,
                transport_context,
            )
        )
    states.extend(
        _neutral_transport_states(
            recipe,
            descriptors,
            transport_context,
        )
    )
    if not states:
        raise ValueError("molecular transport state construction produced no states")
    return tuple(states)


def _cluster_transport_centers(
    cluster_template: ClusterStateTemplate,
    concentration_mol_m3: float,
    transport_context: _TransportCenterConstructionContext,
) -> tuple[MolecularTransportCenter, ...]:
    if cluster_template.cluster_kind == SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
        return _cluster_internal_transport_centers(
            cluster_template,
            concentration_mol_m3,
            transport_context,
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
        )
    if cluster_template.cluster_kind in (
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    ):
        return (
            _cluster_com_transport_center(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CLUSTER_COM_CENTER,
            ),
            *_cluster_internal_transport_centers(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
            ),
        )
    if cluster_template.cluster_kind == HIGHER_CHARGED_CLUSTER_KIND:
        return (
            _cluster_com_transport_center(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CLUSTER_COM_CENTER,
            ),
            *_cluster_internal_transport_centers(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
            ),
        )
    if cluster_template.cluster_kind in (
        CONTACT_PAIR_CLUSTER_KIND,
        NEUTRAL_CLUSTER_KIND,
    ):
        return (
            _cluster_com_transport_center(
                cluster_template,
                concentration_mol_m3,
                transport_context,
                TRANSPORT_ROLE_CONTACT_PAIR_CENTER
                if cluster_template.cluster_kind == CONTACT_PAIR_CLUSTER_KIND
                else TRANSPORT_ROLE_CLUSTER_COM_CENTER,
            ),
        )
    raise ValueError(f"unknown cluster kind {cluster_template.cluster_kind}")


def _cluster_internal_transport_centers(
    cluster_template: ClusterStateTemplate,
    concentration_mol_m3: float,
    transport_context: _TransportCenterConstructionContext,
    transport_role: str,
) -> tuple[MolecularTransportCenter, ...]:
    centers: list[MolecularTransportCenter] = []
    for center_index, charged_center in enumerate(cluster_template.geometry):
        descriptor = transport_context.component_descriptor_by_name[
            charged_center.species_name
        ]
        centers.append(
            _transport_state_from_descriptor(
                label=(
                    f"{cluster_template.label}:center{center_index}:"
                    f"{charged_center.species_name}"
                ),
                parent_cluster_label=cluster_template.label,
                parent_cluster_kind=cluster_template.cluster_kind,
                concentration_mol_m3=concentration_mol_m3,
                center_species_name=charged_center.species_name,
                center_charge_number=charged_center.charge_number,
                center_index=center_index,
                descriptor=descriptor,
                hydrodynamic_radius_scale=_hydrodynamic_radius_scale_for_charge(
                    charged_center.charge_number,
                    transport_context.options,
                )
                * transport_context.options.primitive_parameters.hydrodynamic_radius_scale_cluster,
                transport_context=transport_context,
                transport_role=transport_role,
            )
        )
    return tuple(centers)


def _cluster_com_transport_center(
    cluster_template: ClusterStateTemplate,
    concentration_mol_m3: float,
    transport_context: _TransportCenterConstructionContext,
    transport_role: str,
) -> MolecularTransportCenter:
    charge_cloud_radius_A = _cluster_charge_cloud_radius_A(
        cluster_template,
        transport_context.component_descriptor_by_name,
        transport_context.options,
    )
    hydrodynamic_radius_A = (
        transport_context.options.primitive_parameters.hydrodynamic_radius_scale_cluster
        * cluster_template.hydrodynamic_radius_A
    )
    base_diffusion_m2_s = _diffusion_m2_s(
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        shape_factor=_cluster_shape_factor(
            cluster_template,
            transport_context.component_descriptor_by_name,
        ),
        intrinsic_dielectric_constant=_cluster_intrinsic_dielectric_constant(
            cluster_template,
            transport_context.component_descriptor_by_name,
        ),
        net_charge_number=cluster_template.net_charge_number,
        charge_cloud_radius_A=charge_cloud_radius_A,
        charge_density_reference_A_inv3=transport_context.charge_density_reference_A_inv3,
        mixture_descriptor_state=transport_context.mixture_descriptor_state,
        solvent_environment=transport_context.solvent_environment,
        options=transport_context.options,
    )
    local_obstruction_factor = _local_obstruction_factor(
        label=cluster_template.label,
        net_charge_number=cluster_template.net_charge_number,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        mixture_descriptor_state=transport_context.mixture_descriptor_state,
        options=transport_context.options,
        charge_density_reference_A_inv3=transport_context.charge_density_reference_A_inv3,
    )
    local_obstruction_diffusion_scale = _local_obstruction_diffusion_scale(
        local_obstruction_factor,
        cluster_template.label,
    )
    return MolecularTransportCenter(
        label=f"{cluster_template.label}:com",
        parent_cluster_label=cluster_template.label,
        parent_cluster_kind=cluster_template.cluster_kind,
        concentration_mol_m3=_positive_float(
            concentration_mol_m3,
            f"{cluster_template.label}.concentration_mol_m3",
        ),
        center_species_name=cluster_template.label,
        center_charge_number=cluster_template.net_charge_number,
        center_index=0,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        molecular_volume_A3=cluster_template.molecular_volume_A3,
        diffusion_m2_s=base_diffusion_m2_s * local_obstruction_diffusion_scale,
        local_obstruction_factor=local_obstruction_factor,
        local_obstruction_diffusion_scale=local_obstruction_diffusion_scale,
        transport_role=transport_role,
    )


def _hydrodynamic_radius_scale_for_charge(
    center_charge_number: int,
    options: MolecularMoriOptions,
) -> float:
    if center_charge_number > 0:
        return options.primitive_parameters.hydrodynamic_radius_scale_positive_ion
    if center_charge_number < 0:
        return options.primitive_parameters.hydrodynamic_radius_scale_negative_ion
    raise ValueError("transport center charge must be nonzero")


def _transport_state_concentration_resolution_mol_m3(
    speciation: GenericSpeciationResult,
) -> float:
    analytical_ion_concentration_mol_m3 = math.fsum(
        component.analytical_concentration_M
        * STANDARD_STATE_CONCENTRATION_MOL_M3
        for component in speciation.components
    )
    return TRANSPORT_STATE_CONCENTRATION_RESOLUTION_FACTOR * max(
        1.0,
        _nonnegative_float(
            analytical_ion_concentration_mol_m3,
            "analytical_ion_concentration_mol_m3",
        ),
    )


def _apply_ion_atmosphere_to_transport_states(
    transport_states: tuple[MolecularTransportCenter, ...],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> _AtmosphereTransportStateResult:
    charged_states = tuple(
        state for state in transport_states
        if state.center_charge_number != 0
    )
    if not charged_states:
        return _AtmosphereTransportStateResult(
            transport_states=transport_states,
            diagnostics=MolecularIonAtmosphereDiagnostics(
                solver=ION_ATMOSPHERE_SOLVER_DIAGONAL,
                charged_carrier_count=0,
                kappa_inv_m=math.inf,
                ionic_strength_mol_m3=0.0,
                charge_cloud_form_factor_by_state={},
                friction_ratio_by_state={},
                zeta0_kg_s_by_state={},
                zeta_ep_kg_s_by_state={},
                zeta_rel_kg_s_by_state={},
                countercharge_relaxation_diffusivity_m2_s_by_state={},
            ),
        )
    carrier_concentrations_mol_m3 = {
        state.label: state.concentration_mol_m3 for state in charged_states
    }
    carrier_charges = {
        state.label: state.center_charge_number for state in charged_states
    }
    local_diffusivity_m2_s_by_carrier = {
        state.label: state.diffusion_m2_s for state in charged_states
    }
    hydrodynamic_radius_m_by_carrier = {
        state.label: state.hydrodynamic_radius_A * ANGSTROM_TO_M
        for state in charged_states
    }
    atmosphere_state = build_ion_atmosphere_state(
        IonAtmosphereInput(
            carrier_concentrations_mol_m3=carrier_concentrations_mol_m3,
            carrier_charges=carrier_charges,
            local_diffusivity_m2_s_by_carrier=local_diffusivity_m2_s_by_carrier,
            hydrodynamic_radius_m_by_carrier=hydrodynamic_radius_m_by_carrier,
            viscosity_Pa_s=solvent_environment.viscosity_cP * CP_TO_PA_S,
            relative_dielectric=solvent_environment.dielectric_constant,
            temperature_K=solvent_environment.temperature_K,
            solver=ION_ATMOSPHERE_SOLVER_DIAGONAL,
        )
    )
    countercharge_relaxation_diffusivity_by_state = (
        _countercharge_relaxation_diffusivity_by_state(
            charged_states,
            local_diffusivity_m2_s_by_carrier,
        )
    )
    zeta0_kg_s_by_state = {
        state_label: _positive_float(
            zeta0_kg_s,
            f"{state_label}.zeta0_kg_s",
        )
        for state_label, zeta0_kg_s in atmosphere_state.zeta0_by_carrier.items()
    }
    if math.isinf(atmosphere_state.kappa_inv_m):
        raise ValueError("charged molecular atmosphere kappa_inv_m must be finite")
    kappa_m_inv = 1.0 / _positive_float(
        atmosphere_state.kappa_inv_m,
        "molecular_atmosphere.kappa_inv_m",
    )
    charge_cloud_form_factor_by_state = {
        state.label: _charge_cloud_form_factor(
            state,
            kappa_m_inv,
        )
        for state in charged_states
    }
    zeta_ep_kg_s_by_state = {
        state_label: (
            options.primitive_parameters.atmosphere_ep_scale
            * charge_cloud_form_factor_by_state[state_label]
            * _nonnegative_float(
                zeta_ep_kg_s,
                f"{state_label}.zeta_ep_kg_s",
            )
        )
        for state_label, zeta_ep_kg_s in atmosphere_state.zeta_ep_by_carrier.items()
    }
    zeta_rel_kg_s_by_state = {
        state_label: (
            options.primitive_parameters.atmosphere_rel_scale
            * options.primitive_parameters.cross_relaxation_scale
            * charge_cloud_form_factor_by_state[state_label]
            * _nonnegative_float(
                zeta_rel_kg_s,
                f"{state_label}.zeta_rel_kg_s",
            )
        )
        for state_label, zeta_rel_kg_s in atmosphere_state.zeta_rel_by_carrier.items()
    }
    friction_ratio_by_state = {
        state.label: _atmosphere_friction_ratio(
            state.label,
            zeta0_kg_s_by_state,
            zeta_ep_kg_s_by_state,
            zeta_rel_kg_s_by_state,
        )
        for state in charged_states
    }
    return _AtmosphereTransportStateResult(
        transport_states=transport_states,
        diagnostics=MolecularIonAtmosphereDiagnostics(
            solver=atmosphere_state.solver,
            charged_carrier_count=len(charged_states),
            kappa_inv_m=atmosphere_state.kappa_inv_m,
            ionic_strength_mol_m3=atmosphere_state.ionic_strength_mol_m3,
            charge_cloud_form_factor_by_state=charge_cloud_form_factor_by_state,
            friction_ratio_by_state=friction_ratio_by_state,
            zeta0_kg_s_by_state=zeta0_kg_s_by_state,
            zeta_ep_kg_s_by_state=zeta_ep_kg_s_by_state,
            zeta_rel_kg_s_by_state=zeta_rel_kg_s_by_state,
            countercharge_relaxation_diffusivity_m2_s_by_state=(
                countercharge_relaxation_diffusivity_by_state
            ),
        ),
    )


def _atmosphere_friction_ratio(
    state_label: str,
    zeta0_kg_s_by_state: Mapping[str, float],
    zeta_ep_kg_s_by_state: Mapping[str, float],
    zeta_rel_kg_s_by_state: Mapping[str, float],
) -> float:
    zeta0_kg_s = _positive_float(
        zeta0_kg_s_by_state[state_label],
        f"{state_label}.zeta0_kg_s",
    )
    zeta_ep_kg_s = _nonnegative_float(
        zeta_ep_kg_s_by_state[state_label],
        f"{state_label}.zeta_ep_kg_s",
    )
    zeta_rel_kg_s = _nonnegative_float(
        zeta_rel_kg_s_by_state[state_label],
        f"{state_label}.zeta_rel_kg_s",
    )
    return float(zeta0_kg_s / (zeta0_kg_s + zeta_ep_kg_s + zeta_rel_kg_s))


def _charge_cloud_form_factor(
    transport_state: MolecularTransportCenter,
    inverse_screening_length_m_inv: float,
) -> float:
    charge_cloud_radius_m = (
        _positive_float(
            transport_state.charge_cloud_radius_A,
            f"{transport_state.label}.charge_cloud_radius_A",
        )
        * ANGSTROM_TO_M
    )
    screening_radius_product = (
        _positive_float(inverse_screening_length_m_inv, "inverse_screening_length_m_inv")
        * charge_cloud_radius_m
    )
    gaussian_exponent = -(
        screening_radius_product * screening_radius_product
    ) / GAUSSIAN_CHARGE_CLOUD_FORM_FACTOR_DENOMINATOR
    return _positive_float(
        math.exp(gaussian_exponent),
        f"{transport_state.label}.charge_cloud_form_factor",
    )


def _countercharge_relaxation_diffusivity_by_state(
    charged_states: tuple[MolecularTransportCenter, ...],
    local_diffusivity_m2_s_by_carrier: Mapping[str, float],
) -> Mapping[str, float]:
    relaxation_diffusivity_by_state: dict[str, float] = {}
    for source_state in charged_states:
        countercharge_weighted_diffusivity = 0.0
        countercharge_weight = 0.0
        for target_state in charged_states:
            if (
                source_state.center_charge_number
                * target_state.center_charge_number
                >= 0
            ):
                continue
            target_weight = (
                target_state.concentration_mol_m3
                * abs(target_state.center_charge_number)
            )
            countercharge_weight += target_weight
            countercharge_weighted_diffusivity += (
                target_weight
                * local_diffusivity_m2_s_by_carrier[target_state.label]
            )
        if countercharge_weight <= 0.0:
            raise ValueError(
                f"{source_state.label} has no opposite-charge carrier for atmosphere relaxation"
            )
        source_diffusivity = _positive_float(
            local_diffusivity_m2_s_by_carrier[source_state.label],
            f"{source_state.label}.local_diffusivity_m2_s",
        )
        countercharge_diffusivity = countercharge_weighted_diffusivity / countercharge_weight
        relaxation_diffusivity_by_state[source_state.label] = (
            source_diffusivity
            + _positive_float(
                countercharge_diffusivity,
                f"{source_state.label}.countercharge_diffusivity_m2_s",
            )
        )
    return relaxation_diffusivity_by_state


def _transport_state_from_descriptor(
    label: str,
    parent_cluster_label: str,
    parent_cluster_kind: str,
    concentration_mol_m3: float,
    center_species_name: str,
    center_charge_number: int,
    center_index: int,
    descriptor: MolecularSpeciesDescriptor,
    hydrodynamic_radius_scale: float,
    transport_context: _TransportCenterConstructionContext,
    transport_role: str,
) -> MolecularTransportCenter:
    hydrodynamic_radius_A = (
        _positive_float(hydrodynamic_radius_scale, f"{label}.hydrodynamic_radius_scale")
        * descriptor.hydrodynamic_radius_A
    )
    charge_cloud_radius_A = _scaled_charge_cloud_radius_A(
        descriptor,
        transport_context.options,
    )
    base_diffusion_m2_s = _diffusion_m2_s(
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        shape_factor=descriptor.ligand_field_asymmetry,
        intrinsic_dielectric_constant=descriptor.epsilon_r_pure,
        net_charge_number=center_charge_number,
        charge_cloud_radius_A=charge_cloud_radius_A,
        charge_density_reference_A_inv3=(
            transport_context.charge_density_reference_A_inv3
        ),
        mixture_descriptor_state=transport_context.mixture_descriptor_state,
        solvent_environment=transport_context.solvent_environment,
        options=transport_context.options,
    )
    local_obstruction_factor = _local_obstruction_factor(
        label=label,
        net_charge_number=center_charge_number,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        mixture_descriptor_state=transport_context.mixture_descriptor_state,
        options=transport_context.options,
        charge_density_reference_A_inv3=(
            transport_context.charge_density_reference_A_inv3
        ),
    )
    local_obstruction_diffusion_scale = _local_obstruction_diffusion_scale(
        local_obstruction_factor,
        label,
    )
    return MolecularTransportCenter(
        label=label,
        parent_cluster_label=parent_cluster_label,
        parent_cluster_kind=parent_cluster_kind,
        concentration_mol_m3=_positive_float(
            concentration_mol_m3,
            f"{label}.concentration_mol_m3",
        ),
        center_species_name=center_species_name,
        center_charge_number=center_charge_number,
        center_index=center_index,
        hydrodynamic_radius_A=hydrodynamic_radius_A,
        charge_cloud_radius_A=charge_cloud_radius_A,
        molecular_volume_A3=descriptor.molecular_volume_A3,
        diffusion_m2_s=base_diffusion_m2_s * local_obstruction_diffusion_scale,
        local_obstruction_factor=local_obstruction_factor,
        local_obstruction_diffusion_scale=local_obstruction_diffusion_scale,
        transport_role=transport_role,
    )


def _free_ion_hydrodynamic_radius_scale(
    component: IonComponent,
    options: MolecularMoriOptions,
) -> float:
    if component.charge_number > 0:
        return options.primitive_parameters.hydrodynamic_radius_scale_positive_ion
    if component.charge_number < 0:
        return options.primitive_parameters.hydrodynamic_radius_scale_negative_ion
    raise ValueError(f"{component.species_name} free ion must be charged")


def _neutral_transport_states(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    transport_context: _TransportCenterConstructionContext,
) -> tuple[MolecularTransportCenter, ...]:
    states: list[MolecularTransportCenter] = []
    for species_name, volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        if descriptor.role != ROLE_SOLVENT:
            raise ValueError(f"recipe solvent {species_name} descriptor role mismatch")
        concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            volume_fraction,
            descriptor,
        )
        states.append(
            _transport_state_from_descriptor(
                label=f"neutral:{species_name}",
                parent_cluster_label=f"neutral:{species_name}",
                parent_cluster_kind=TRANSPORT_ROLE_NEUTRAL_CENTER,
                concentration_mol_m3=concentration_mol_m3,
                center_species_name=species_name,
                center_charge_number=0,
                center_index=0,
                descriptor=descriptor,
                hydrodynamic_radius_scale=1.0,
                transport_context=transport_context,
                transport_role=TRANSPORT_ROLE_NEUTRAL_CENTER,
            )
        )
    for species_name, weight_fraction in recipe.additives.items():
        descriptor = descriptors[species_name]
        if descriptor.role != ROLE_ADDITIVE:
            raise ValueError(f"recipe additive {species_name} descriptor role mismatch")
        concentration_mol_m3 = (
            _positive_float(weight_fraction, f"{species_name}.weight_fraction")
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol
        )
        states.append(
            _transport_state_from_descriptor(
                label=f"neutral:{species_name}",
                parent_cluster_label=f"neutral:{species_name}",
                parent_cluster_kind=TRANSPORT_ROLE_NEUTRAL_CENTER,
                concentration_mol_m3=concentration_mol_m3,
                center_species_name=species_name,
                center_charge_number=0,
                center_index=0,
                descriptor=descriptor,
                hydrodynamic_radius_scale=1.0,
                transport_context=transport_context,
                transport_role=TRANSPORT_ROLE_NEUTRAL_CENTER,
            )
        )
    return tuple(states)


def _markov_process_from_transport_states(
    transport_states: tuple[MolecularTransportCenter, ...],
    options: MolecularMoriOptions,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
) -> _MarkovProcessConstruction:
    charged_states = tuple(
        transport_state for transport_state in transport_states
        if transport_state.center_charge_number != 0
    )
    if not charged_states:
        return _neutral_markov_process_from_transport_states(
            transport_states,
            options,
        )
    state_labels: list[str] = []
    state_concentrations: list[float] = []
    events: list[MarkovAdditiveEvent] = []
    memory_primitives: list[MolecularAtmosphereMemoryPrimitive] = []
    mobile_state_indices: list[_MobileTransportStateIndex] = []
    for transport_state in charged_states:
        if (
            transport_state.transport_role
            == TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER
        ):
            mobile_state_index = len(state_labels)
            state_labels.append(f"{transport_state.label}:mobile")
            state_concentrations.append(transport_state.concentration_mol_m3)
            mobile_state_indices.append(
                _MobileTransportStateIndex(
                    transport_state=transport_state,
                    mobile_state_index=mobile_state_index,
                    mobile_concentration_mol_m3=transport_state.concentration_mol_m3,
                )
            )
            continue
        if _state_has_zero_atmosphere_coupling(
            transport_state,
            atmosphere_diagnostics,
        ):
            mobile_state_index = len(state_labels)
            state_labels.append(f"{transport_state.label}:mobile")
            state_concentrations.append(transport_state.concentration_mol_m3)
            mobile_state_indices.append(
                _MobileTransportStateIndex(
                    transport_state=transport_state,
                    mobile_state_index=mobile_state_index,
                    mobile_concentration_mol_m3=transport_state.concentration_mol_m3,
                )
            )
            jump_length_m = _jump_length_m(transport_state, options)
            ordinary_rate_s_inv = (
                transport_state.diffusion_m2_s
                / (jump_length_m * jump_length_m)
            )
            _append_ordinary_mobile_translation_events(
                events,
                mobile_state_index,
                transport_state,
                jump_length_m,
                ordinary_rate_s_inv,
            )
            continue
        primitive = _atmosphere_memory_primitive(
            transport_state,
            options,
            atmosphere_diagnostics,
        )
        memory_primitives.append(primitive)
        mobile_state_index = len(state_labels)
        state_labels.append(f"{transport_state.label}:mobile")
        state_concentrations.append(primitive.mobile_concentration_mol_m3)
        mobile_state_indices.append(
            _MobileTransportStateIndex(
                transport_state=transport_state,
                mobile_state_index=mobile_state_index,
                mobile_concentration_mol_m3=primitive.mobile_concentration_mol_m3,
            )
        )
        ordinary_rate_s_inv = (
            (1.0 - primitive.atmosphere_coupling_fraction)
            * primitive.D_local_m2_s
            / (primitive.jump_length_m * primitive.jump_length_m)
        )
        _append_ordinary_mobile_translation_events(
            events,
            mobile_state_index,
            transport_state,
            primitive.jump_length_m,
            ordinary_rate_s_inv,
        )
        for axis_index, axis_vector in enumerate(CARTESIAN_DIRECTIONS):
            for direction_sign in TRANSLATION_EVENT_SIGNS:
                atmosphere_state_index = len(state_labels)
                sign_label = "plus" if direction_sign > 0.0 else "minus"
                state_labels.append(
                    f"{transport_state.label}:axis{axis_index}:{sign_label}:atmosphere"
                )
                state_concentrations.append(
                    primitive.atmosphere_concentration_per_direction_mol_m3
                )
                memory_displacement_m = _charge_displacement_m(
                    transport_state,
                    primitive.jump_length_m,
                    axis_vector,
                    direction_sign,
                )
                events.append(
                    MarkovAdditiveEvent(
                        from_state_index=mobile_state_index,
                        to_state_index=atmosphere_state_index,
                        rate_s_inv=primitive.k_capture_s_inv,
                        charge_displacement_m=memory_displacement_m,
                        label=(
                            "atmosphere_memory_capture:"
                            f"{transport_state.label}:axis{axis_index}:{sign_label}"
                        ),
                        family_label="atmosphere_memory_translation",
                    )
                )
                back_relaxation_displacement_m = tuple(
                    float(-component) for component in memory_displacement_m
                )
                events.append(
                    MarkovAdditiveEvent(
                        from_state_index=atmosphere_state_index,
                        to_state_index=mobile_state_index,
                        rate_s_inv=primitive.k_exit_s_inv,
                        charge_displacement_m=back_relaxation_displacement_m,
                        label=(
                            "atmosphere_memory_back_relaxation:"
                            f"{transport_state.label}:axis{axis_index}:{sign_label}"
                        ),
                        family_label="atmosphere_memory_translation",
                    )
                )
    _append_association_conversion_events(
        events,
        tuple(mobile_state_indices),
        options,
    )
    _append_solvent_separated_pair_center_events(
        events,
        tuple(mobile_state_indices),
        options,
    )
    return _MarkovProcessConstruction(
        state_labels=tuple(state_labels),
        state_concentrations_mol_m3=np.asarray(state_concentrations, dtype=float),
        events=tuple(events),
        memory_primitives=tuple(memory_primitives),
    )


def _state_has_zero_atmosphere_coupling(
    transport_state: MolecularTransportCenter,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
) -> bool:
    state_label = transport_state.label
    zeta_ep_kg_s = _nonnegative_float(
        atmosphere_diagnostics.zeta_ep_kg_s_by_state[state_label],
        f"{state_label}.zeta_ep_kg_s",
    )
    zeta_rel_kg_s = _nonnegative_float(
        atmosphere_diagnostics.zeta_rel_kg_s_by_state[state_label],
        f"{state_label}.zeta_rel_kg_s",
    )
    return (zeta_ep_kg_s + zeta_rel_kg_s) == 0.0


def _append_ordinary_mobile_translation_events(
    events: list[MarkovAdditiveEvent],
    mobile_state_index: int,
    transport_state: MolecularTransportCenter,
    jump_length_m: float,
    rate_s_inv: float,
) -> None:
    _positive_float(rate_s_inv, f"{transport_state.label}.ordinary_rate_s_inv")
    for axis_index, axis_vector in enumerate(CARTESIAN_DIRECTIONS):
        for direction_sign in TRANSLATION_EVENT_SIGNS:
            sign_label = "plus" if direction_sign > 0.0 else "minus"
            displacement_m = _charge_displacement_m(
                transport_state,
                jump_length_m,
                axis_vector,
                direction_sign,
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=mobile_state_index,
                    to_state_index=mobile_state_index,
                    rate_s_inv=rate_s_inv,
                    charge_displacement_m=displacement_m,
                    label=(
                        "ordinary_mobile_translation:"
                        f"{transport_state.label}:axis{axis_index}:{sign_label}"
                    ),
                    family_label="ordinary_mobile_translation",
                )
            )


def _append_association_conversion_events(
    events: list[MarkovAdditiveEvent],
    mobile_state_indices: tuple[_MobileTransportStateIndex, ...],
    options: MolecularMoriOptions,
) -> None:
    for source_index, source_state_index in enumerate(mobile_state_indices):
        for target_state_index in mobile_state_indices[source_index + 1:]:
            if (
                source_state_index.transport_state.center_charge_number
                != target_state_index.transport_state.center_charge_number
            ):
                continue
            if source_state_index.transport_state.center_charge_number == 0:
                continue
            _append_reversible_association_conversion_pair(
                events,
                source_state_index,
                target_state_index,
                options,
            )


def _append_solvent_separated_pair_center_events(
    events: list[MarkovAdditiveEvent],
    mobile_state_indices: tuple[_MobileTransportStateIndex, ...],
    options: MolecularMoriOptions,
) -> None:
    ssip_mobile_indices_by_parent: dict[str, list[_MobileTransportStateIndex]] = {}
    for mobile_state_index in mobile_state_indices:
        transport_state = mobile_state_index.transport_state
        if (
            transport_state.transport_role
            != TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER
        ):
            continue
        if transport_state.parent_cluster_kind != SOLVENT_SEPARATED_PAIR_CLUSTER_KIND:
            raise ValueError(
                f"{transport_state.label} SSIP center has parent kind "
                f"{transport_state.parent_cluster_kind}"
            )
        ssip_mobile_indices_by_parent.setdefault(
            transport_state.parent_cluster_label,
            [],
        ).append(mobile_state_index)
    for parent_cluster_label, ssip_mobile_indices in ssip_mobile_indices_by_parent.items():
        positive_centers = tuple(
            mobile_state_index
            for mobile_state_index in ssip_mobile_indices
            if mobile_state_index.transport_state.center_charge_number > 0
        )
        negative_centers = tuple(
            mobile_state_index
            for mobile_state_index in ssip_mobile_indices
            if mobile_state_index.transport_state.center_charge_number < 0
        )
        if not positive_centers or not negative_centers:
            raise ValueError(
                f"{parent_cluster_label} solvent-separated pair must contain "
                "opposite charged centers"
            )
        for positive_center in positive_centers:
            for negative_center in negative_centers:
                mode_rate_budget = _solvent_separated_pair_mode_rate_budget(
                    positive_center.transport_state,
                    negative_center.transport_state,
                    options,
                )
                _append_solvent_separated_pair_relative_translation_events(
                    events,
                    positive_center,
                    negative_center,
                    mode_rate_budget.relative_rate_s_inv,
                    options,
                )
                _append_solvent_separated_pair_com_translation_events(
                    events,
                    positive_center,
                    negative_center,
                    mode_rate_budget.co_motion_rate_s_inv,
                    options,
                )
                _append_solvent_separated_pair_residual_center_events(
                    events,
                    positive_center,
                    mode_rate_budget.positive_residual_rate_s_inv,
                    options,
                )
                _append_solvent_separated_pair_residual_center_events(
                    events,
                    negative_center,
                    mode_rate_budget.negative_residual_rate_s_inv,
                    options,
                )


def _solvent_separated_pair_mode_rate_budget(
    positive_transport_center: MolecularTransportCenter,
    negative_transport_center: MolecularTransportCenter,
    options: MolecularMoriOptions,
) -> _SolventSeparatedPairModeRateBudget:
    positive_center_rate_budget_s_inv = _center_translation_rate_budget_s_inv(
        positive_transport_center,
        options,
    )
    negative_center_rate_budget_s_inv = _center_translation_rate_budget_s_inv(
        negative_transport_center,
        options,
    )
    paired_center_rate_budget_s_inv = min(
        positive_center_rate_budget_s_inv,
        negative_center_rate_budget_s_inv,
    )
    absolute_net_charge_number = abs(
        positive_transport_center.center_charge_number
        + negative_transport_center.center_charge_number
    )
    absolute_center_charge_sum = (
        abs(positive_transport_center.center_charge_number)
        + abs(negative_transport_center.center_charge_number)
    )
    charge_sum_scale = _positive_float(
        absolute_center_charge_sum,
        "solvent_separated_pair.absolute_center_charge_sum",
    )
    co_motion_fraction = absolute_net_charge_number / charge_sum_scale
    relative_fraction = 1.0 - co_motion_fraction
    if relative_fraction < 0.0:
        raise ValueError("solvent-separated-pair relative fraction is negative")
    return _SolventSeparatedPairModeRateBudget(
        relative_rate_s_inv=paired_center_rate_budget_s_inv * relative_fraction,
        co_motion_rate_s_inv=paired_center_rate_budget_s_inv * co_motion_fraction,
        positive_residual_rate_s_inv=(
            positive_center_rate_budget_s_inv - paired_center_rate_budget_s_inv
        ),
        negative_residual_rate_s_inv=(
            negative_center_rate_budget_s_inv - paired_center_rate_budget_s_inv
        ),
    )


def _center_translation_rate_budget_s_inv(
    transport_center: MolecularTransportCenter,
    options: MolecularMoriOptions,
) -> float:
    jump_length_m = _jump_length_m(transport_center, options)
    return _positive_float(
        transport_center.diffusion_m2_s,
        f"{transport_center.label}.diffusion_m2_s",
    ) / (jump_length_m * jump_length_m)


def _append_solvent_separated_pair_relative_translation_events(
    events: list[MarkovAdditiveEvent],
    positive_center_index: _MobileTransportStateIndex,
    negative_center_index: _MobileTransportStateIndex,
    relative_rate_s_inv: float,
    options: MolecularMoriOptions,
) -> None:
    positive_transport_center = positive_center_index.transport_state
    negative_transport_center = negative_center_index.transport_state
    positive_jump_length_m = _jump_length_m(positive_transport_center, options)
    negative_jump_length_m = _jump_length_m(negative_transport_center, options)
    if relative_rate_s_inv == 0.0:
        return
    relative_charge_step_m = (
        positive_transport_center.center_charge_number * positive_jump_length_m
        - negative_transport_center.center_charge_number * negative_jump_length_m
    )
    _append_solvent_separated_pair_axis_events(
        events,
        positive_center_index.mobile_state_index,
        positive_transport_center,
        negative_transport_center,
        relative_charge_step_m,
        relative_rate_s_inv,
        "solvent_separated_pair_relative_translation",
    )


def _append_solvent_separated_pair_com_translation_events(
    events: list[MarkovAdditiveEvent],
    positive_center_index: _MobileTransportStateIndex,
    negative_center_index: _MobileTransportStateIndex,
    co_motion_rate_s_inv: float,
    options: MolecularMoriOptions,
) -> None:
    positive_transport_center = positive_center_index.transport_state
    negative_transport_center = negative_center_index.transport_state
    positive_jump_length_m = _jump_length_m(positive_transport_center, options)
    negative_jump_length_m = _jump_length_m(negative_transport_center, options)
    if co_motion_rate_s_inv == 0.0:
        return
    co_motion_length_m = math.sqrt(positive_jump_length_m * negative_jump_length_m)
    co_motion_charge_step_m = (
        (
            positive_transport_center.center_charge_number
            + negative_transport_center.center_charge_number
        )
        * co_motion_length_m
    )
    _append_solvent_separated_pair_axis_events(
        events,
        positive_center_index.mobile_state_index,
        positive_transport_center,
        negative_transport_center,
        co_motion_charge_step_m,
        co_motion_rate_s_inv,
        "solvent_separated_pair_com_translation",
    )


def _append_solvent_separated_pair_residual_center_events(
    events: list[MarkovAdditiveEvent],
    center_index: _MobileTransportStateIndex,
    residual_rate_s_inv: float,
    options: MolecularMoriOptions,
) -> None:
    if residual_rate_s_inv < 0.0:
        raise ValueError(
            f"{center_index.transport_state.label}.residual_rate_s_inv is negative"
        )
    if residual_rate_s_inv == 0.0:
        return
    _append_solvent_separated_pair_center_axis_events(
        events,
        center_index.mobile_state_index,
        center_index.transport_state,
        _jump_length_m(center_index.transport_state, options),
        residual_rate_s_inv,
        "solvent_separated_pair_residual_center_translation",
    )


def _append_solvent_separated_pair_axis_events(
    events: list[MarkovAdditiveEvent],
    source_mobile_state_index: int,
    positive_transport_center: MolecularTransportCenter,
    negative_transport_center: MolecularTransportCenter,
    charge_step_m: float,
    rate_s_inv: float,
    family_label: str,
) -> None:
    _positive_float(rate_s_inv, f"{family_label}.rate_s_inv")
    for axis_index, axis_vector in enumerate(CARTESIAN_DIRECTIONS):
        for direction_sign in TRANSLATION_EVENT_SIGNS:
            sign_label = "plus" if direction_sign > 0.0 else "minus"
            displacement_m = tuple(
                float(direction_sign * charge_step_m * axis_component)
                for axis_component in axis_vector
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=source_mobile_state_index,
                    to_state_index=source_mobile_state_index,
                    rate_s_inv=rate_s_inv,
                    charge_displacement_m=displacement_m,
                    label=(
                        f"{family_label}:"
                        f"{positive_transport_center.label}:"
                        f"{negative_transport_center.label}:"
                        f"axis{axis_index}:{sign_label}"
                    ),
                    family_label=family_label,
                )
            )


def _append_solvent_separated_pair_center_axis_events(
    events: list[MarkovAdditiveEvent],
    source_mobile_state_index: int,
    transport_center: MolecularTransportCenter,
    jump_length_m: float,
    rate_s_inv: float,
    family_label: str,
) -> None:
    _positive_float(rate_s_inv, f"{family_label}.rate_s_inv")
    for axis_index, axis_vector in enumerate(CARTESIAN_DIRECTIONS):
        for direction_sign in TRANSLATION_EVENT_SIGNS:
            sign_label = "plus" if direction_sign > 0.0 else "minus"
            displacement_m = _charge_displacement_m(
                transport_center,
                jump_length_m,
                axis_vector,
                direction_sign,
            )
            events.append(
                MarkovAdditiveEvent(
                    from_state_index=source_mobile_state_index,
                    to_state_index=source_mobile_state_index,
                    rate_s_inv=rate_s_inv,
                    charge_displacement_m=displacement_m,
                    label=(
                        f"{family_label}:"
                        f"{transport_center.label}:axis{axis_index}:{sign_label}"
                    ),
                    family_label=family_label,
                )
            )


def _append_reversible_association_conversion_pair(
    events: list[MarkovAdditiveEvent],
    first_state_index: _MobileTransportStateIndex,
    second_state_index: _MobileTransportStateIndex,
    options: MolecularMoriOptions,
) -> None:
    conversion_length_m = (
        (
            first_state_index.transport_state.hydrodynamic_radius_A
            + second_state_index.transport_state.hydrodynamic_radius_A
        )
        * ANGSTROM_TO_M
    )
    encounter_rate_s_inv = (
        first_state_index.transport_state.diffusion_m2_s
        + second_state_index.transport_state.diffusion_m2_s
    ) / (
        _positive_float(conversion_length_m, "association_conversion_length_m")
        ** 2
    )
    symmetric_conductance_mol_m3_s = (
        options.primitive_parameters.association_conversion_rate_scale
        * _positive_float(encounter_rate_s_inv, "association_encounter_rate_s_inv")
        * math.sqrt(
            _positive_float(
                first_state_index.mobile_concentration_mol_m3,
                f"{first_state_index.transport_state.label}.mobile_concentration_mol_m3",
            )
            * _positive_float(
                second_state_index.mobile_concentration_mol_m3,
                f"{second_state_index.transport_state.label}.mobile_concentration_mol_m3",
            )
        )
    )
    first_to_second_rate_s_inv = (
        symmetric_conductance_mol_m3_s
        / first_state_index.mobile_concentration_mol_m3
    )
    second_to_first_rate_s_inv = (
        symmetric_conductance_mol_m3_s
        / second_state_index.mobile_concentration_mol_m3
    )
    events.append(
        MarkovAdditiveEvent(
            from_state_index=first_state_index.mobile_state_index,
            to_state_index=second_state_index.mobile_state_index,
            rate_s_inv=_positive_float(
                first_to_second_rate_s_inv,
                "association_conversion_first_to_second_rate_s_inv",
            ),
            charge_displacement_m=(0.0, 0.0, 0.0),
            label=(
                "association_conversion:"
                f"{first_state_index.transport_state.label}:"
                f"{second_state_index.transport_state.label}"
            ),
            family_label="association_conversion",
        )
    )
    events.append(
        MarkovAdditiveEvent(
            from_state_index=second_state_index.mobile_state_index,
            to_state_index=first_state_index.mobile_state_index,
            rate_s_inv=_positive_float(
                second_to_first_rate_s_inv,
                "association_conversion_second_to_first_rate_s_inv",
            ),
            charge_displacement_m=(0.0, 0.0, 0.0),
            label=(
                "association_conversion:"
                f"{second_state_index.transport_state.label}:"
                f"{first_state_index.transport_state.label}"
            ),
            family_label="association_conversion",
        )
    )


def _charge_displacement_m(
    transport_state: MolecularTransportCenter,
    jump_length_m: float,
    axis_vector: tuple[float, float, float],
    direction_sign: float,
) -> tuple[float, float, float]:
    return tuple(
        float(
            direction_sign
            * transport_state.center_charge_number
            * jump_length_m
            * axis_component
        )
        for axis_component in axis_vector
    )


def _atmosphere_memory_primitive(
    transport_state: MolecularTransportCenter,
    options: MolecularMoriOptions,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
) -> MolecularAtmosphereMemoryPrimitive:
    state_label = transport_state.label
    zeta0_kg_s = _positive_float(
        atmosphere_diagnostics.zeta0_kg_s_by_state[state_label],
        f"{state_label}.zeta0_kg_s",
    )
    zeta_ep_kg_s = _nonnegative_float(
        atmosphere_diagnostics.zeta_ep_kg_s_by_state[state_label],
        f"{state_label}.zeta_ep_kg_s",
    )
    zeta_rel_kg_s = _nonnegative_float(
        atmosphere_diagnostics.zeta_rel_kg_s_by_state[state_label],
        f"{state_label}.zeta_rel_kg_s",
    )
    zeta_atmosphere_kg_s = zeta_ep_kg_s + zeta_rel_kg_s
    _positive_float(zeta_atmosphere_kg_s, f"{state_label}.zeta_atmosphere_kg_s")
    atmosphere_coupling_fraction = zeta_atmosphere_kg_s / (
        zeta0_kg_s + zeta_atmosphere_kg_s
    )
    if atmosphere_coupling_fraction <= 0.0 or atmosphere_coupling_fraction >= 1.0:
        raise ValueError(
            f"{state_label}.atmosphere_coupling_fraction must be in (0, 1)"
        )
    back_relaxation_probability = zeta_rel_kg_s / zeta_atmosphere_kg_s
    if back_relaxation_probability < 0.0 or back_relaxation_probability > 1.0:
        raise ValueError(
            f"{state_label}.back_relaxation_probability must be in [0, 1]"
        )
    jump_length_m = _jump_length_m(transport_state, options)
    local_diffusivity_m2_s = _positive_float(
        transport_state.diffusion_m2_s,
        f"{state_label}.D_local_m2_s",
    )
    atmosphere_relaxation_diffusivity_m2_s = _positive_float(
        atmosphere_diagnostics.countercharge_relaxation_diffusivity_m2_s_by_state[
            state_label
        ],
        f"{state_label}.atmosphere_relaxation_diffusivity_m2_s",
    )
    k_capture_s_inv = (
        options.primitive_parameters.atmosphere_capture_scale
        * atmosphere_coupling_fraction
        * local_diffusivity_m2_s
        / (jump_length_m * jump_length_m)
    )
    k_exit_s_inv = _atmosphere_memory_exit_rate_s_inv(
        atmosphere_relaxation_diffusivity_m2_s,
        atmosphere_diagnostics,
        state_label,
        options,
    )
    residence_ratio = k_capture_s_inv / k_exit_s_inv
    _positive_float(residence_ratio, f"{state_label}.atmosphere_residence_ratio")
    orientation_count = len(CARTESIAN_DIRECTIONS) * len(TRANSLATION_EVENT_SIGNS)
    mobile_concentration_mol_m3 = transport_state.concentration_mol_m3 / (
        1.0 + orientation_count * residence_ratio
    )
    atmosphere_concentration_per_direction_mol_m3 = (
        residence_ratio * mobile_concentration_mol_m3
    )
    return MolecularAtmosphereMemoryPrimitive(
        state_label=state_label,
        D_local_m2_s=local_diffusivity_m2_s,
        atmosphere_relaxation_diffusivity_m2_s=(
            atmosphere_relaxation_diffusivity_m2_s
        ),
        jump_length_m=jump_length_m,
        k_capture_s_inv=k_capture_s_inv,
        k_exit_s_inv=k_exit_s_inv,
        atmosphere_coupling_fraction=float(atmosphere_coupling_fraction),
        back_relaxation_probability=float(back_relaxation_probability),
        mobile_concentration_mol_m3=float(mobile_concentration_mol_m3),
        atmosphere_concentration_per_direction_mol_m3=float(
            atmosphere_concentration_per_direction_mol_m3
        ),
        zeta0_kg_s=zeta0_kg_s,
        zeta_ep_kg_s=zeta_ep_kg_s,
        zeta_rel_kg_s=zeta_rel_kg_s,
    )


def _atmosphere_memory_exit_rate_s_inv(
    atmosphere_relaxation_diffusivity_m2_s: float,
    atmosphere_diagnostics: MolecularIonAtmosphereDiagnostics,
    state_label: str,
    options: MolecularMoriOptions,
) -> float:
    if math.isinf(atmosphere_diagnostics.kappa_inv_m):
        raise ValueError(f"{state_label}.kappa_inv_m must be finite")
    kappa_m_inv = 1.0 / _positive_float(
        atmosphere_diagnostics.kappa_inv_m,
        f"{state_label}.kappa_inv_m",
    )
    exit_rate_s_inv = (
        options.primitive_parameters.orientation_relaxation_rate_scale
        * options.primitive_parameters.atmosphere_exit_scale
        * _positive_float(
            atmosphere_relaxation_diffusivity_m2_s,
            f"{state_label}.atmosphere_relaxation_diffusivity_m2_s",
        )
        * kappa_m_inv
        * kappa_m_inv
    )
    return _positive_float(exit_rate_s_inv, f"{state_label}.k_exit_s_inv")


def _neutral_markov_process_from_transport_states(
    transport_states: tuple[MolecularTransportCenter, ...],
    options: MolecularMoriOptions,
) -> _MarkovProcessConstruction:
    state_labels = tuple(state.label for state in transport_states)
    state_concentrations = np.asarray(
        [state.concentration_mol_m3 for state in transport_states],
        dtype=float,
    )
    events: list[MarkovAdditiveEvent] = []
    for state_index, state in enumerate(transport_states):
        jump_length_m = _jump_length_m(state, options)
        rate_s_inv = state.diffusion_m2_s / (jump_length_m * jump_length_m)
        events.append(
            MarkovAdditiveEvent(
                from_state_index=state_index,
                to_state_index=state_index,
                rate_s_inv=rate_s_inv,
                charge_displacement_m=(0.0, 0.0, 0.0),
                label=f"neutral_translation:{state.label}",
                family_label="neutral_translation",
            )
        )
    return _MarkovProcessConstruction(
        state_labels=state_labels,
        state_concentrations_mol_m3=state_concentrations,
        events=tuple(events),
        memory_primitives=tuple(),
    )


def _jump_length_m(
    transport_state: MolecularTransportCenter,
    options: MolecularMoriOptions,
) -> float:
    jump_length_m = (
        options.translation_jump_length_multiplier
        * options.primitive_parameters.jump_length_scale
        * transport_state.hydrodynamic_radius_A
        * ANGSTROM_TO_M
    )
    return _positive_float(jump_length_m, f"{transport_state.label}.jump_length_m")


def _diffusion_m2_s(
    hydrodynamic_radius_A: float,
    shape_factor: float,
    intrinsic_dielectric_constant: float,
    net_charge_number: int,
    charge_cloud_radius_A: float,
    charge_density_reference_A_inv3: float,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> float:
    free_volume_factor = _free_volume_friction_factor(
        solvent_environment.hard_sphere_volume_fraction,
        options,
    )
    shape_friction_factor = _shape_friction_factor(shape_factor, options)
    dielectric_mobility_factor = _dielectric_mobility_friction_factor(
        solvent_environment,
        options,
    )
    solvation_mobility_factor = _solvation_mobility_friction_factor(
        shape_factor,
        mixture_descriptor_state,
        options,
    )
    charge_density_mobility_factor = _charge_density_mobility_friction_factor(
        net_charge_number,
        charge_cloud_radius_A,
        charge_density_reference_A_inv3,
        options,
    )
    charge_cloud_extent_mobility_factor = (
        _charge_cloud_extent_mobility_friction_factor(
            net_charge_number,
            hydrodynamic_radius_A,
            charge_cloud_radius_A,
            mixture_descriptor_state,
            options,
        )
    )
    intrinsic_dielectric_drag_factor = (
        _negative_ion_intrinsic_dielectric_drag_mobility_friction_factor(
            net_charge_number,
            intrinsic_dielectric_constant,
            options,
        )
    )
    shape_delocalization_factor = (
        _negative_ion_shape_delocalization_mobility_friction_factor(
            net_charge_number,
            shape_factor,
            options,
        )
    )
    anion_composition_disorder_factor = (
        _anion_composition_disorder_mobility_friction_factor(
            net_charge_number,
            mixture_descriptor_state,
            options,
        )
    )
    viscosity_Pa_s = solvent_environment.viscosity_cP * CP_TO_PA_S
    radius_m = (
        _positive_float(hydrodynamic_radius_A, "hydrodynamic_radius_A")
        * ANGSTROM_TO_M
    )
    denominator = (
        STOKES_DENOMINATOR_FACTOR
        * math.pi
        * viscosity_Pa_s
        * radius_m
        * shape_friction_factor
        * free_volume_factor
        * dielectric_mobility_factor
        * solvation_mobility_factor
        * charge_density_mobility_factor
        * charge_cloud_extent_mobility_factor
        * intrinsic_dielectric_drag_factor
        * shape_delocalization_factor
        * anion_composition_disorder_factor
    )
    return float(K_B * solvent_environment.temperature_K / denominator)


def _shape_friction_factor(
    shape_factor: float,
    options: MolecularMoriOptions,
) -> float:
    descriptor_shape_factor = _positive_float(shape_factor, "shape_factor")
    shape_friction_exponent = _positive_float(
        options.primitive_parameters.shape_friction_exponent,
        "shape_friction_exponent",
    )
    return _positive_float(
        descriptor_shape_factor ** shape_friction_exponent,
        "shape_friction_factor",
    )


def _dielectric_mobility_friction_factor(
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> float:
    dielectric_constant = _positive_float(
        solvent_environment.dielectric_constant,
        "solvent_environment.dielectric_constant",
    )
    dielectric_mobility_exponent = _positive_float(
        options.primitive_parameters.dielectric_mobility_exponent,
        "dielectric_mobility_exponent",
    )
    return _positive_float(
        dielectric_constant ** (-dielectric_mobility_exponent),
        "dielectric_mobility_friction_factor",
    )


def _solvation_mobility_friction_factor(
    shape_factor: float,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    options: MolecularMoriOptions,
) -> float:
    solvation_mobility_base = (
        _positive_float(
            mixture_descriptor_state.solvation_obstruction_factor,
            "mixture.solvation_obstruction_factor",
        )
        * _positive_float(
            mixture_descriptor_state.additive_solvation_obstruction_factor,
            "mixture.additive_solvation_obstruction_factor",
        )
    )
    shape_anisotropy = (
        _positive_float(shape_factor, "shape_factor")
        - ISOTROPIC_SHAPE_FACTOR
    )
    additive_shape_exponent = (
        options.primitive_parameters.additive_shape_solvation_mobility_exponent
        * shape_anisotropy
        * shape_anisotropy
    )
    return _positive_float(
        solvation_mobility_base
        ** options.primitive_parameters.solvation_mobility_exponent,
        "solvation_mobility_friction_factor",
    ) * _positive_float(
        mixture_descriptor_state.additive_solvation_obstruction_factor
        ** additive_shape_exponent,
        "additive_shape_solvation_mobility_friction_factor",
    )


def _charge_density_mobility_friction_factor(
    net_charge_number: int,
    charge_cloud_radius_A: float,
    charge_density_reference_A_inv3: float,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number == 0:
        return 1.0
    charge_density_reference = _positive_float(
        charge_density_reference_A_inv3,
        "charge_density_reference_A_inv3",
    )
    charge_cloud_radius = _positive_float(
        charge_cloud_radius_A,
        "charge_cloud_radius_A",
    )
    state_charge_density_A_inv3 = abs(net_charge_number) / (
        charge_cloud_radius
        * charge_cloud_radius
        * charge_cloud_radius
    )
    normalized_charge_density = state_charge_density_A_inv3 / charge_density_reference
    if net_charge_number > 0:
        charge_density_mobility_exponent = (
            options.primitive_parameters.positive_ion_charge_density_mobility_exponent
        )
    else:
        charge_density_mobility_exponent = (
            options.primitive_parameters.negative_ion_charge_density_mobility_exponent
        )
    return _positive_float(
        normalized_charge_density
        ** charge_density_mobility_exponent,
        "charge_density_mobility_friction_factor",
    )


def _charge_cloud_extent_mobility_friction_factor(
    net_charge_number: int,
    hydrodynamic_radius_A: float,
    charge_cloud_radius_A: float,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number == 0:
        return 1.0
    hydrodynamic_radius = _positive_float(
        hydrodynamic_radius_A,
        "hydrodynamic_radius_A",
    )
    if net_charge_number > 0:
        effective_charge_cloud_radius_A = _positive_float(
            mixture_descriptor_state.mean_anion_charge_cloud_radius_A,
            "mixture.mean_anion_charge_cloud_radius_A",
        )
        mobility_exponent = _positive_float(
            options.primitive_parameters.positive_ion_counteranion_charge_cloud_mobility_exponent,
            "positive_ion_counteranion_charge_cloud_mobility_exponent",
        )
    else:
        effective_charge_cloud_radius_A = _positive_float(
            charge_cloud_radius_A,
            "charge_cloud_radius_A",
        )
        mobility_exponent = _positive_float(
            options.primitive_parameters.negative_ion_charge_cloud_mobility_exponent,
            "negative_ion_charge_cloud_mobility_exponent",
        )
    charge_cloud_extent = 1.0 + effective_charge_cloud_radius_A / hydrodynamic_radius
    return _positive_float(
        charge_cloud_extent ** (-mobility_exponent),
        "charge_cloud_extent_mobility_friction_factor",
    )


def _negative_ion_intrinsic_dielectric_drag_mobility_friction_factor(
    net_charge_number: int,
    intrinsic_dielectric_constant: float,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number >= 0:
        return 1.0
    dielectric_constant = _positive_float(
        intrinsic_dielectric_constant,
        "intrinsic_dielectric_constant",
    )
    mobility_exponent = _positive_float(
        options.primitive_parameters.negative_ion_intrinsic_dielectric_drag_mobility_exponent,
        "negative_ion_intrinsic_dielectric_drag_mobility_exponent",
    )
    return _positive_float(
        (1.0 + dielectric_constant) ** mobility_exponent,
        "negative_ion_intrinsic_dielectric_drag_mobility_friction_factor",
    )


def _negative_ion_shape_delocalization_mobility_friction_factor(
    net_charge_number: int,
    shape_factor: float,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number >= 0:
        return 1.0
    descriptor_shape_factor = _positive_float(shape_factor, "shape_factor")
    mobility_exponent = _positive_float(
        options.primitive_parameters.negative_ion_shape_delocalization_mobility_exponent,
        "negative_ion_shape_delocalization_mobility_exponent",
    )
    return _positive_float(
        descriptor_shape_factor ** (-mobility_exponent),
        "negative_ion_shape_delocalization_mobility_friction_factor",
    )


def _anion_composition_disorder_mobility_friction_factor(
    net_charge_number: int,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    options: MolecularMoriOptions,
) -> float:
    if net_charge_number == 0:
        return 1.0
    anion_composition_entropy = _nonnegative_float(
        mixture_descriptor_state.anion_composition_entropy,
        "mixture.anion_composition_entropy",
    )
    if net_charge_number > 0:
        mobility_exponent = _positive_float(
            options.primitive_parameters.positive_ion_anion_disorder_mobility_exponent,
            "positive_ion_anion_disorder_mobility_exponent",
        )
    else:
        mobility_exponent = _positive_float(
            options.primitive_parameters.negative_ion_anion_disorder_mobility_exponent,
            "negative_ion_anion_disorder_mobility_exponent",
        )
    return _positive_float(
        math.exp(-mobility_exponent * anion_composition_entropy),
        "anion_composition_disorder_mobility_friction_factor",
    )


def _free_volume_friction_factor(
    hard_sphere_volume_fraction: float,
    options: MolecularMoriOptions,
) -> float:
    if hard_sphere_volume_fraction >= options.max_packing_fraction:
        raise ValueError(
            "hard_sphere_volume_fraction must be below max_packing_fraction"
        )
    remaining_free_volume = (
        options.max_packing_fraction - hard_sphere_volume_fraction
    ) / options.max_packing_fraction
    effective_free_volume_exponent = (
        options.free_volume_exponent
        * options.primitive_parameters.free_volume_exponent
    )
    return float(remaining_free_volume ** (-effective_free_volume_exponent))


def _molecular_mixture_descriptor_state(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
    solvent_environment: MolecularSolventEnvironment,
    options: MolecularMoriOptions,
) -> _MolecularMixtureDescriptorState:
    density_packing_scale = _density_packing_scale(recipe, descriptors)
    total_anion_concentration_M = 0.0
    anion_charge_cloud_weighted_sum_A = 0.0
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        if descriptor.charge_number >= 0:
            raise ValueError(f"{species_name} must be an anion")
        anion_concentration_M = _positive_float(
            concentration_M,
            f"{species_name}.concentration_M",
        )
        total_anion_concentration_M += anion_concentration_M
        anion_charge_cloud_weighted_sum_A += (
            anion_concentration_M
            * _positive_float(
                descriptor.charge_cloud_radius_A,
                f"{species_name}.charge_cloud_radius_A",
            )
        )
    if total_anion_concentration_M > 0.0:
        mean_anion_charge_cloud_radius_A = (
            anion_charge_cloud_weighted_sum_A / total_anion_concentration_M
        )
    else:
        mean_anion_charge_cloud_radius_A = _positive_float(
            solvent_environment.solvent_effective_radius_A,
            "solvent_environment.solvent_effective_radius_A",
        )
    anion_composition_entropy = 0.0
    for species_name, concentration_M in recipe.anions.items():
        anion_mole_fraction = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            / _positive_float(
                total_anion_concentration_M,
                "mixture.total_anion_concentration_M",
            )
        )
        anion_composition_entropy -= anion_mole_fraction * math.log(
            anion_mole_fraction
        )
    total_concentration_mol_m3 = 0.0
    donor_number_weighted_sum = 0.0
    acceptor_number_weighted_sum = 0.0
    polarizability_weighted_sum_A3 = 0.0
    molecular_volume_weighted_sum_A3 = 0.0
    additive_solvation_support = 0.0
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        (
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        ) = _accumulate_mixture_descriptor_weights(
            concentration_mol_m3,
            descriptor,
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        )
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        (
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        ) = _accumulate_mixture_descriptor_weights(
            concentration_mol_m3,
            descriptor,
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        )
    for species_name, volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            volume_fraction * density_packing_scale,
            descriptor,
        )
        (
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        ) = _accumulate_mixture_descriptor_weights(
            concentration_mol_m3,
            descriptor,
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        )
    for species_name, weight_fraction in recipe.additives.items():
        descriptor = descriptors[species_name]
        positive_weight_fraction = _positive_float(
            weight_fraction,
            f"{species_name}.weight_fraction",
        )
        additive_solvation_support += (
            positive_weight_fraction
            * _additive_solvation_support(descriptor)
        )
        concentration_mol_m3 = (
            positive_weight_fraction
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol
        )
        (
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        ) = _accumulate_mixture_descriptor_weights(
            concentration_mol_m3,
            descriptor,
            total_concentration_mol_m3,
            donor_number_weighted_sum,
            acceptor_number_weighted_sum,
            polarizability_weighted_sum_A3,
            molecular_volume_weighted_sum_A3,
        )
    _positive_float(total_concentration_mol_m3, "mixture.total_concentration_mol_m3")
    available_free_volume_fraction = (
        _positive_float(options.max_packing_fraction, "max_packing_fraction")
        - _nonnegative_float(
            solvent_environment.hard_sphere_volume_fraction,
            "hard_sphere_volume_fraction",
        )
    )
    if available_free_volume_fraction <= 0.0:
        raise ValueError("available_free_volume_fraction must be positive")
    total_number_density_m3 = N_A * total_concentration_mol_m3
    free_volume_per_particle_m3 = available_free_volume_fraction / total_number_density_m3
    void_radius_A = (
        (
            3.0
            * free_volume_per_particle_m3
            / (4.0 * math.pi)
        )
        ** (1.0 / 3.0)
        / ANGSTROM_TO_M
    )
    donor_number = donor_number_weighted_sum / total_concentration_mol_m3
    acceptor_number = acceptor_number_weighted_sum / total_concentration_mol_m3
    polarizability_volume_ratio = (
        polarizability_weighted_sum_A3
        / _positive_float(
            molecular_volume_weighted_sum_A3,
            "mixture.molecular_volume_weighted_sum_A3",
        )
    )
    solvation_support = (
        _nonnegative_float(donor_number, "mixture.donor_number")
        + _nonnegative_float(acceptor_number, "mixture.acceptor_number")
        + _nonnegative_float(
            polarizability_volume_ratio,
            "mixture.polarizability_volume_ratio",
        )
    )
    solvation_obstruction_factor = 1.0 / (1.0 + solvation_support)
    additive_solvation_obstruction_factor = 1.0 / (
        1.0
        + _nonnegative_float(
            additive_solvation_support,
            "mixture.additive_solvation_support",
        )
    )
    ionic_strength_mol_m3 = _analytical_ionic_strength_mol_m3(
        recipe,
        descriptors,
    )
    return _MolecularMixtureDescriptorState(
        hard_sphere_volume_fraction=float(
            solvent_environment.hard_sphere_volume_fraction
        ),
        max_packing_fraction=float(options.max_packing_fraction),
        ionic_strength_mol_m3=ionic_strength_mol_m3,
        void_radius_A=_positive_float(void_radius_A, "mixture.void_radius_A"),
        donor_number=float(donor_number),
        acceptor_number=float(acceptor_number),
        polarizability_volume_ratio=float(polarizability_volume_ratio),
        solvation_obstruction_factor=float(solvation_obstruction_factor),
        additive_solvation_obstruction_factor=float(
            additive_solvation_obstruction_factor
        ),
        mean_anion_charge_cloud_radius_A=_positive_float(
            mean_anion_charge_cloud_radius_A,
            "mixture.mean_anion_charge_cloud_radius_A",
        ),
        anion_composition_entropy=float(anion_composition_entropy),
    )


def _analytical_ionic_strength_mol_m3(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    ionic_strength_mol_m3 = 0.0
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        ionic_strength_mol_m3 += (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            * descriptor.charge_number
            * descriptor.charge_number
        )
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        ionic_strength_mol_m3 += (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
            * descriptor.charge_number
            * descriptor.charge_number
        )
    return _nonnegative_float(ionic_strength_mol_m3, "ionic_strength_mol_m3")


def _accumulate_mixture_descriptor_weights(
    concentration_mol_m3: float,
    descriptor: MolecularSpeciesDescriptor,
    total_concentration_mol_m3: float,
    donor_number_weighted_sum: float,
    acceptor_number_weighted_sum: float,
    polarizability_weighted_sum_A3: float,
    molecular_volume_weighted_sum_A3: float,
) -> tuple[float, float, float, float, float]:
    positive_concentration_mol_m3 = _positive_float(
        concentration_mol_m3,
        f"{descriptor.name}.concentration_mol_m3",
    )
    return (
        total_concentration_mol_m3 + positive_concentration_mol_m3,
        donor_number_weighted_sum
        + positive_concentration_mol_m3 * descriptor.donor_number,
        acceptor_number_weighted_sum
        + positive_concentration_mol_m3 * descriptor.acceptor_number,
        polarizability_weighted_sum_A3
        + positive_concentration_mol_m3 * descriptor.polarizability_A3,
        molecular_volume_weighted_sum_A3
        + positive_concentration_mol_m3 * descriptor.molecular_volume_A3,
    )


def _additive_solvation_support(
    descriptor: MolecularSpeciesDescriptor,
) -> float:
    polarizability_volume_ratio = (
        _nonnegative_float(
            descriptor.polarizability_A3,
            f"{descriptor.name}.polarizability_A3",
        )
        / _positive_float(
            descriptor.molecular_volume_A3,
            f"{descriptor.name}.molecular_volume_A3",
        )
    )
    return _nonnegative_float(
        _nonnegative_float(descriptor.donor_number, f"{descriptor.name}.donor_number")
        + _nonnegative_float(
            descriptor.acceptor_number,
            f"{descriptor.name}.acceptor_number",
        )
        + _nonnegative_float(
            float(descriptor.hbond_acceptor_count),
            f"{descriptor.name}.hbond_acceptor_count",
        )
        + polarizability_volume_ratio,
        f"{descriptor.name}.additive_solvation_support",
    )


def _charge_density_reference_A_inv3(
    speciation: GenericSpeciationResult,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
    options: MolecularMoriOptions,
) -> float:
    entries: list[_ChargeDensityReferenceEntry] = []
    for component in speciation.components:
        concentration_mol_m3 = speciation.free_component_concentrations_mol_m3[
            component.species_name
        ]
        entries.append(
            _ChargeDensityReferenceEntry(
                concentration_mol_m3=concentration_mol_m3,
                net_charge_number=component.charge_number,
                charge_cloud_radius_A=_scaled_charge_cloud_radius_A(
                    component.descriptor,
                    options,
                ),
            )
        )
    for cluster_template in speciation.cluster_templates:
        entries.append(
            _ChargeDensityReferenceEntry(
                concentration_mol_m3=speciation.cluster_concentrations_mol_m3[
                    cluster_template.label
                ],
                net_charge_number=cluster_template.net_charge_number,
                charge_cloud_radius_A=_cluster_charge_cloud_radius_A(
                    cluster_template,
                    component_descriptor_by_name,
                    options,
                ),
            )
        )
    concentration_weighted_charge_density = 0.0
    concentration_weight = 0.0
    for entry in entries:
        if entry.net_charge_number == 0:
            continue
        concentration_mol_m3 = _positive_float(
            entry.concentration_mol_m3,
            "charge_density_reference.concentration_mol_m3",
        )
        charge_cloud_radius_A = _positive_float(
            entry.charge_cloud_radius_A,
            "charge_density_reference.charge_cloud_radius_A",
        )
        concentration_weight += concentration_mol_m3
        concentration_weighted_charge_density += (
            concentration_mol_m3
            * abs(entry.net_charge_number)
            / (charge_cloud_radius_A ** 3)
        )
    if concentration_weight <= 0.0:
        return 0.0
    return _positive_float(
        concentration_weighted_charge_density / concentration_weight,
        "charge_density_reference_A_inv3",
    )


def _scaled_charge_cloud_radius_A(
    descriptor: MolecularSpeciesDescriptor,
    options: MolecularMoriOptions,
) -> float:
    return (
        options.primitive_parameters.charge_cloud_radius_scale
        * _positive_float(
            descriptor.charge_cloud_radius_A,
            f"{descriptor.name}.charge_cloud_radius_A",
        )
    )


def _cluster_charge_cloud_radius_A(
    cluster_template: ClusterStateTemplate,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
    options: MolecularMoriOptions,
) -> float:
    charge_weighted_radius_squared_sum = 0.0
    charge_weight_sum = 0.0
    for species_name, stoichiometric_count in cluster_template.stoichiometry.items():
        descriptor = component_descriptor_by_name[species_name]
        charge_weight = abs(descriptor.charge_number) * stoichiometric_count
        if charge_weight == 0:
            continue
        charge_cloud_radius_A = _scaled_charge_cloud_radius_A(descriptor, options)
        charge_weight_sum += charge_weight
        charge_weighted_radius_squared_sum += (
            charge_weight * charge_cloud_radius_A * charge_cloud_radius_A
        )
    if charge_weight_sum <= 0.0:
        raise ValueError(f"{cluster_template.label} charge weight must be positive")
    return _positive_float(
        math.sqrt(charge_weighted_radius_squared_sum / charge_weight_sum),
        f"{cluster_template.label}.charge_cloud_radius_A",
    )


def _cluster_intrinsic_dielectric_constant(
    cluster_template: ClusterStateTemplate,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    stoichiometric_count_sum = 0.0
    dielectric_weighted_sum = 0.0
    for species_name, stoichiometric_count in cluster_template.stoichiometry.items():
        descriptor = component_descriptor_by_name[species_name]
        positive_count = _positive_float(
            stoichiometric_count,
            f"{cluster_template.label}.{species_name}.stoichiometric_count",
        )
        stoichiometric_count_sum += positive_count
        dielectric_weighted_sum += (
            positive_count
            * _positive_float(
                descriptor.epsilon_r_pure,
                f"{species_name}.epsilon_r_pure",
            )
        )
    return _positive_float(
        dielectric_weighted_sum
        / _positive_float(
            stoichiometric_count_sum,
            f"{cluster_template.label}.stoichiometric_count_sum",
        ),
        f"{cluster_template.label}.intrinsic_dielectric_constant",
    )


def _local_obstruction_factor(
    label: str,
    net_charge_number: int,
    hydrodynamic_radius_A: float,
    charge_cloud_radius_A: float,
    mixture_descriptor_state: _MolecularMixtureDescriptorState,
    options: MolecularMoriOptions,
    charge_density_reference_A_inv3: float,
) -> float:
    if net_charge_number == 0:
        return 0.0
    charge_density_reference = _positive_float(
        charge_density_reference_A_inv3,
        "charge_density_reference_A_inv3",
    )
    packing_denominator = (
        _positive_float(
            mixture_descriptor_state.max_packing_fraction,
            "max_packing_fraction",
        )
        - _nonnegative_float(
            mixture_descriptor_state.hard_sphere_volume_fraction,
            "hard_sphere_volume_fraction",
        )
    )
    if packing_denominator <= 0.0:
        raise ValueError(f"{label}.packing_denominator must be positive")
    free_volume_ratio = (
        mixture_descriptor_state.hard_sphere_volume_fraction / packing_denominator
    )
    size_ratio = (
        _positive_float(hydrodynamic_radius_A, f"{label}.hydrodynamic_radius_A")
        / _positive_float(mixture_descriptor_state.void_radius_A, "void_radius_A")
    )
    state_charge_density_A_inv3 = (
        abs(net_charge_number)
        / (
            _positive_float(charge_cloud_radius_A, f"{label}.charge_cloud_radius_A")
            ** 3
        )
    )
    normalized_charge_density = state_charge_density_A_inv3 / charge_density_reference
    ionic_strength_ratio = (
        _nonnegative_float(
            mixture_descriptor_state.ionic_strength_mol_m3,
            "ionic_strength_mol_m3",
        )
        / STANDARD_STATE_CONCENTRATION_MOL_M3
    )
    ionic_strength_crowding_factor = (1.0 + ionic_strength_ratio) / 2.0
    obstruction_factor = (
        options.primitive_parameters.local_obstruction_strength
        * (
            free_volume_ratio
            ** options.primitive_parameters.local_obstruction_free_volume_exponent
        )
        * (
            ionic_strength_crowding_factor
            ** options.primitive_parameters.local_obstruction_ionic_strength_exponent
        )
        * (
            mixture_descriptor_state.additive_solvation_obstruction_factor
            ** options.primitive_parameters.local_obstruction_additive_solvation_exponent
        )
        * (size_ratio ** options.primitive_parameters.local_obstruction_size_exponent)
        * (
            normalized_charge_density
            ** options.primitive_parameters.local_obstruction_charge_density_exponent
        )
        * (
            mixture_descriptor_state.solvation_obstruction_factor
            ** options.primitive_parameters.local_obstruction_solvation_exponent
        )
    )
    return _nonnegative_float(obstruction_factor, f"{label}.local_obstruction_factor")


def _local_obstruction_diffusion_scale(
    local_obstruction_factor: float,
    label: str,
) -> float:
    obstruction_factor = _nonnegative_float(
        local_obstruction_factor,
        f"{label}.local_obstruction_factor",
    )
    return _positive_float(
        1.0 / (1.0 + obstruction_factor),
        f"{label}.local_obstruction_diffusion_scale",
    )


def _hard_sphere_volume_fraction(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    density_packing_scale = _density_packing_scale(recipe, descriptors)
    volume_fraction = 0.0
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        volume_fraction += _species_volume_fraction_from_molarity(
            concentration_M,
            descriptor,
        )
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        volume_fraction += _species_volume_fraction_from_molarity(
            concentration_M,
            descriptor,
        )
    for species_name, solvent_volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            solvent_volume_fraction * density_packing_scale,
            descriptor,
        )
        volume_fraction += _species_volume_fraction_from_concentration(
            concentration_mol_m3,
            descriptor,
        )
    for species_name, additive_weight_fraction in recipe.additives.items():
        descriptor = descriptors[species_name]
        concentration_mol_m3 = (
            _positive_float(
                additive_weight_fraction,
                f"{species_name}.weight_fraction",
            )
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol
        )
        volume_fraction += _species_volume_fraction_from_concentration(
            concentration_mol_m3,
            descriptor,
        )
    return float(volume_fraction)


def _mixture_effective_radius_A(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    mean_molecular_volume_A3 = _mixture_mean_molecular_volume_A3(
        recipe,
        descriptors,
    )
    return _positive_float(
        (
            3.0
            * mean_molecular_volume_A3
            / (4.0 * math.pi)
        ) ** (1.0 / 3.0),
        "mixture_effective_radius_A",
    )


def _mixture_mean_molecular_volume_A3(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    density_packing_scale = _density_packing_scale(recipe, descriptors)
    weighted_volume_A3 = 0.0
    concentration_weight = 0.0
    for species_name, concentration_M in recipe.cations.items():
        descriptor = descriptors[species_name]
        species_concentration_mol_m3 = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        weighted_volume_A3 += (
            species_concentration_mol_m3 * descriptor.molecular_volume_A3
        )
        concentration_weight += species_concentration_mol_m3
    for species_name, concentration_M in recipe.anions.items():
        descriptor = descriptors[species_name]
        species_concentration_mol_m3 = (
            _positive_float(concentration_M, f"{species_name}.concentration_M")
            * STANDARD_STATE_CONCENTRATION_MOL_M3
        )
        weighted_volume_A3 += (
            species_concentration_mol_m3 * descriptor.molecular_volume_A3
        )
        concentration_weight += species_concentration_mol_m3
    for species_name, solvent_volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        species_concentration_mol_m3 = _liquid_component_concentration_mol_m3(
            solvent_volume_fraction * density_packing_scale,
            descriptor,
        )
        weighted_volume_A3 += (
            species_concentration_mol_m3 * descriptor.molecular_volume_A3
        )
        concentration_weight += species_concentration_mol_m3
    for species_name, additive_weight_fraction in recipe.additives.items():
        descriptor = descriptors[species_name]
        species_concentration_mol_m3 = (
            _positive_float(
                additive_weight_fraction,
                f"{species_name}.weight_fraction",
            )
            * recipe.mixture_properties.density_g_ml
            * GRAMS_PER_M3_PER_G_ML
            / descriptor.molecular_weight_g_mol
        )
        weighted_volume_A3 += (
            species_concentration_mol_m3 * descriptor.molecular_volume_A3
        )
        concentration_weight += species_concentration_mol_m3
    return _positive_float(
        weighted_volume_A3 / concentration_weight,
        "mixture_mean_molecular_volume_A3",
    )


def _density_packing_scale(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    ideal_density_g_ml = _ideal_recipe_density_g_ml(recipe, descriptors)
    measured_density_g_ml = _positive_float(
        recipe.mixture_properties.density_g_ml,
        "recipe.mixture_properties.density_g_ml",
    )
    return _positive_float(
        measured_density_g_ml / ideal_density_g_ml,
        "density_packing_scale",
    )


def _ideal_recipe_density_g_ml(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    solvent_mass_g_per_liter = math.fsum(
        _positive_float(volume_fraction, f"{species_name}.volume_fraction")
        * _positive_float(
            descriptors[species_name].density_g_ml,
            f"{species_name}.density_g_ml",
        )
        * GRAMS_PER_LITER_PER_G_ML
        for species_name, volume_fraction in recipe.solvents.items()
    )
    cation_mass_g_per_liter = math.fsum(
        _positive_float(concentration_M, f"{species_name}.concentration_M")
        * _positive_float(
            descriptors[species_name].molecular_weight_g_mol,
            f"{species_name}.molecular_weight_g_mol",
        )
        for species_name, concentration_M in recipe.cations.items()
    )
    anion_mass_g_per_liter = math.fsum(
        _positive_float(concentration_M, f"{species_name}.concentration_M")
        * _positive_float(
            descriptors[species_name].molecular_weight_g_mol,
            f"{species_name}.molecular_weight_g_mol",
        )
        for species_name, concentration_M in recipe.anions.items()
    )
    total_neutral_additive_weight_fraction = math.fsum(
        _positive_float(weight_fraction, f"{species_name}.weight_fraction")
        for species_name, weight_fraction in recipe.additives.items()
    )
    if total_neutral_additive_weight_fraction >= 1.0:
        raise ValueError("total neutral additive weight fraction must be below one")
    base_mass_g_per_liter = (
        solvent_mass_g_per_liter
        + cation_mass_g_per_liter
        + anion_mass_g_per_liter
    )
    total_mass_g_per_liter = base_mass_g_per_liter / (
        1.0 - total_neutral_additive_weight_fraction
    )
    return _positive_float(
        total_mass_g_per_liter / GRAMS_PER_LITER_PER_G_ML,
        "ideal_recipe_density_g_ml",
    )


def _species_volume_fraction_from_molarity(
    concentration_M: float,
    descriptor: MolecularSpeciesDescriptor,
) -> float:
    concentration_mol_m3 = (
        _positive_float(concentration_M, f"{descriptor.name}.concentration_M")
        * STANDARD_STATE_CONCENTRATION_MOL_M3
    )
    return _species_volume_fraction_from_concentration(
        concentration_mol_m3,
        descriptor,
    )


def _species_volume_fraction_from_concentration(
    concentration_mol_m3: float,
    descriptor: MolecularSpeciesDescriptor,
) -> float:
    return float(
        concentration_mol_m3
        * N_A
        * descriptor.molecular_volume_A3
        * CUBIC_ANGSTROM_TO_CUBIC_M
    )


def _liquid_component_concentration_mol_m3(
    volume_fraction: float,
    descriptor: MolecularSpeciesDescriptor,
) -> float:
    return float(
        _positive_float(volume_fraction, f"{descriptor.name}.volume_fraction")
        * descriptor.density_g_ml
        * GRAMS_PER_M3_PER_G_ML
        / descriptor.molecular_weight_g_mol
    )


def _cluster_shape_factor(
    cluster_template: ClusterStateTemplate,
    component_descriptor_by_name: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    return float(
        max(
            component_descriptor_by_name[species_name].ligand_field_asymmetry
            for species_name in cluster_template.stoichiometry
        )
    )


def _validate_recipe(recipe: MolecularElectrolyteRecipe) -> None:
    _positive_float(recipe.temperature_K, "temperature_K")
    _positive_float(recipe.pressure_Pa, "pressure_Pa")
    _positive_float(recipe.mixture_properties.density_g_ml, "mixture.density_g_ml")
    _positive_float(recipe.mixture_properties.viscosity_cP, "mixture.viscosity_cP")
    _positive_float(
        recipe.mixture_properties.dielectric_constant,
        "mixture.dielectric_constant",
    )
    for species_name, volume_fraction in recipe.solvents.items():
        _positive_float(volume_fraction, f"{species_name}.volume_fraction")
    for species_name, weight_fraction in recipe.additives.items():
        _positive_float(weight_fraction, f"{species_name}.weight_fraction")


def _validate_options(options: MolecularMoriOptions) -> None:
    if options.max_cluster_ion_count < MINIMUM_CLUSTER_ION_COUNT:
        raise ValueError(
            "max_cluster_ion_count must include at least cation-anion pair states"
        )
    validate_conductivity_primitive_parameters(options.primitive_parameters)
    _positive_float(options.max_packing_fraction, "max_packing_fraction")
    _nonnegative_float(options.free_volume_exponent, "free_volume_exponent")
    _positive_float(
        options.translation_jump_length_multiplier,
        "translation_jump_length_multiplier",
    )


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _finite_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise ValueError(f"{context} must be finite")
    return parsed_value


def _nonnegative_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < 0.0:
        raise ValueError(f"{context} must be nonnegative and finite")
    return parsed_value


@dataclass(frozen=True)
class AnalyticalConductivityModelInput:
    recipe: MolecularElectrolyteRecipe
    species_inputs: Mapping[str, MolecularSpeciesInput]
    descriptor_backend: MolecularDescriptorBackend
    options: MolecularMoriOptions


AnalyticalConductivityModelResult = MolecularMoriConductivityResult


def compute_analytical_conductivity_model(
    model_input: AnalyticalConductivityModelInput,
) -> AnalyticalConductivityModelResult:
    validate_conductivity_primitive_parameters(
        model_input.options.primitive_parameters,
    )
    return compute_molecular_electrolyte_conductivity(
        model_input.recipe,
        model_input.species_inputs,
        model_input.descriptor_backend,
        model_input.options,
    )


def compute_analytical_conductivity_model_with_diagnostic_cluster_shifts(
    model_input: AnalyticalConductivityModelInput,
    diagnostic_cluster_standard_free_energy_shift_over_RT_by_label: Mapping[str, float],
) -> AnalyticalConductivityModelResult:
    validate_conductivity_primitive_parameters(
        model_input.options.primitive_parameters,
    )
    return compute_molecular_electrolyte_conductivity_with_diagnostic_cluster_shifts(
        model_input.recipe,
        model_input.species_inputs,
        model_input.descriptor_backend,
        model_input.options,
        diagnostic_cluster_standard_free_energy_shift_over_RT_by_label,
    )
