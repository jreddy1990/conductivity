"""Primitive parameter scales for descriptor-driven molecular conductivity."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Mapping


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
