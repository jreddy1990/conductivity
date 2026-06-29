"""Primitive parameter scales for descriptor-driven molecular conductivity."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Mapping


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


def conductivity_primitive_parameters_to_log_values(
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> tuple[float, ...]:
    validate_conductivity_primitive_parameters(primitive_parameters)
    return tuple(
        math.log(_parameter_value(primitive_parameters, field_name))
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    )


def conductivity_primitive_parameter_log_values_for_names(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    parameter_names: tuple[str, ...],
) -> tuple[float, ...]:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_parameter_name_tuple(parameter_names)
    return tuple(
        math.log(_parameter_value(primitive_parameters, parameter_name))
        for parameter_name in parameter_names
    )


def conductivity_primitive_parameters_from_log_values(
    log_values: tuple[float, ...],
) -> ConductivityPrimitiveParameterSet:
    field_names = CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    if len(log_values) != len(field_names):
        raise ValueError(
            "log_values length must match ConductivityPrimitiveParameterSet field count"
        )
    parameter_values: dict[str, float] = {}
    for field_name, log_value in zip(field_names, log_values):
        parsed_log_value = float(log_value)
        if not math.isfinite(parsed_log_value):
            raise ValueError(f"log parameter {field_name} must be finite")
        parameter_values[field_name] = math.exp(parsed_log_value)
    return _conductivity_primitive_parameters_from_validated_values(
        parameter_values
    )


def conductivity_primitive_parameters_with_log_updates(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    parameter_names: tuple[str, ...],
    log_values: tuple[float, ...],
) -> ConductivityPrimitiveParameterSet:
    validate_conductivity_primitive_parameters(primitive_parameters)
    _validate_parameter_name_tuple(parameter_names)
    if len(parameter_names) != len(log_values):
        raise ValueError("parameter_names and log_values must have equal length")
    parameter_values = {
        field_name: _parameter_value(primitive_parameters, field_name)
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    }
    for parameter_name, log_value in zip(parameter_names, log_values):
        parsed_log_value = float(log_value)
        if not math.isfinite(parsed_log_value):
            raise ValueError(f"log parameter {parameter_name} must be finite")
        parameter_values[parameter_name] = math.exp(parsed_log_value)
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
        field_name: _positive_float(
            parameter_mapping[field_name],
            f"primitive_parameters.{field_name}",
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
            field_name: _positive_float(
                parameter_values[field_name],
                f"primitive_parameters.{field_name}",
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
        _positive_float(
            getattr(primitive_parameters, field_name),
            f"primitive_parameters.{field_name}",
        )


def _parameter_value(
    primitive_parameters: ConductivityPrimitiveParameterSet,
    field_name: str,
) -> float:
    if field_name not in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES:
        raise ValueError(f"unknown conductivity primitive parameter {field_name}")
    return _positive_float(
        getattr(primitive_parameters, field_name),
        f"primitive_parameters.{field_name}",
    )


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
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value
