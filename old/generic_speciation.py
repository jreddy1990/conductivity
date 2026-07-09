"""Generic molecular speciation for descriptor-driven electrolyte conductivity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from constants import E_CHARGE, EPS_0, N_A, R
from conductivity.molecular_descriptors import MolecularSpeciesDescriptor
from conductivity.molecular_primitive_parameters import (
    ConductivityPrimitiveParameterSet,
    validate_conductivity_primitive_parameters,
)


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
    solvent_volume_fractions: Mapping[str, float]
    solvent_coordination_affinity_J_mol: float


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
    solvation_competition_J_mol: float
    steric_J_mol: float
    entropy_J_mol: float
    standard_state_correction_J_mol: float
    activity_reference_J_mol: float


@dataclass(frozen=True)
class _PairFreeEnergyTerms:
    coulomb_J_mol: float
    desolvation_J_mol: float
    coordination_J_mol: float
    solvation_competition_J_mol: float


@dataclass(frozen=True)
class PMFTerm:
    name: str
    free_energy_J_mol: float
    source: str


@dataclass(frozen=True)
class SolvationCompetitionPMFPartition:
    salt_label: str
    cation_label: str
    anion_label: str
    solvent_composition: Mapping[str, float]
    temperature_K: float
    basin_labels: tuple[str, ...]
    basin_boundaries: Mapping[str, tuple[float, float]]
    pmf_terms: tuple[PMFTerm, ...]
    restricted_partition_weights: Mapping[str, float]
    concentrations_mol_m3: Mapping[str, float]


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
    solvation_competition_pmf_partitions: tuple[
        SolvationCompetitionPMFPartition, ...
    ]
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
            solvation_competition_pmf_partitions=tuple(),
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
            solvation_competition_pmf_partitions=tuple(),
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
        solvation_competition_pmf_partitions=solvation_competition_pmf_partition(
            components,
            cluster_templates,
            {
                component.species_name: float(free_concentrations[index])
                for index, component in enumerate(components)
            },
            cluster_concentrations,
            solvent_environment,
        ),
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
    solvation_competition_J_mol = 0.0
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
            solvation_competition_J_mol += pair_terms.solvation_competition_J_mol
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
        + solvation_competition_J_mol
        + steric_J_mol
        + entropy_J_mol
        + standard_state_correction_J_mol
    )
    return _ClusterFreeEnergyTerms(
        standard_free_energy_J_mol=float(standard_free_energy_J_mol),
        coulomb_J_mol=float(coulomb_J_mol),
        desolvation_J_mol=float(desolvation_J_mol),
        coordination_J_mol=float(coordination_J_mol),
        solvation_competition_J_mol=float(solvation_competition_J_mol),
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
    solvation_competition_energy_J_mol = (
        _pair_solvation_competition_penalty_J_mol(
            first_component,
            second_component,
            solvent_environment,
            primitive_parameters,
        )
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
        solvation_competition_J_mol=float(solvation_competition_energy_J_mol),
    )


def solvation_competition_pmf_partition(
    components: tuple[IonComponent, ...],
    cluster_templates: tuple[ClusterStateTemplate, ...],
    free_component_concentrations_mol_m3: Mapping[str, float],
    cluster_concentrations_mol_m3: Mapping[str, float],
    solvent_environment: MolecularSolventEnvironment,
) -> tuple[SolvationCompetitionPMFPartition, ...]:
    _validate_solvent_environment(solvent_environment)
    component_by_name = {component.species_name: component for component in components}
    partitions: list[SolvationCompetitionPMFPartition] = []
    for cluster_template in cluster_templates:
        if len(cluster_template.stoichiometry) != 2:
            continue
        charged_species_names = tuple(cluster_template.stoichiometry)
        first_component = component_by_name[charged_species_names[0]]
        second_component = component_by_name[charged_species_names[1]]
        if first_component.charge_number * second_component.charge_number >= 0:
            continue
        cation_component = (
            first_component if first_component.charge_number > 0 else second_component
        )
        anion_component = (
            first_component if first_component.charge_number < 0 else second_component
        )
        cation_concentration = free_component_concentrations_mol_m3[
            cation_component.species_name
        ]
        anion_concentration = free_component_concentrations_mol_m3[
            anion_component.species_name
        ]
        cluster_concentration = cluster_concentrations_mol_m3[cluster_template.label]
        total_partition_concentration = _positive_float(
            cation_concentration + anion_concentration + cluster_concentration,
            f"{cluster_template.label}.partition_total_concentration_mol_m3",
        )
        partitions.append(
            SolvationCompetitionPMFPartition(
                salt_label=f"{cation_component.species_name}:{anion_component.species_name}",
                cation_label=cation_component.species_name,
                anion_label=anion_component.species_name,
                solvent_composition=dict(solvent_environment.solvent_volume_fractions),
                temperature_K=solvent_environment.temperature_K,
                basin_labels=(
                    "free_ion_center",
                    SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
                    CONTACT_PAIR_CLUSTER_KIND,
                    NEUTRAL_CLUSTER_KIND,
                    HIGHER_CHARGED_CLUSTER_KIND,
                ),
                basin_boundaries=_solvation_competition_basin_boundaries_A(
                    cation_component,
                    anion_component,
                    solvent_environment,
                ),
                pmf_terms=(
                    PMFTerm("coulomb", cluster_template.coulomb_J_mol, "pair_pmf"),
                    PMFTerm(
                        "desolvation",
                        cluster_template.desolvation_J_mol,
                        "pair_pmf",
                    ),
                    PMFTerm(
                        "coordination",
                        cluster_template.coordination_J_mol,
                        "pair_pmf",
                    ),
                    PMFTerm(
                        "solvation_competition",
                        cluster_template.standard_free_energy_J_mol
                        - cluster_template.coulomb_J_mol
                        - cluster_template.desolvation_J_mol
                        - cluster_template.coordination_J_mol
                        - cluster_template.steric_J_mol
                        - cluster_template.entropy_J_mol
                        - cluster_template.standard_state_correction_J_mol,
                        "solvent_shell_competition_pmf",
                    ),
                ),
                restricted_partition_weights={
                    "free_ion_center": float(
                        (cation_concentration + anion_concentration)
                        / total_partition_concentration
                    ),
                    cluster_template.cluster_kind: float(
                        cluster_concentration / total_partition_concentration
                    ),
                },
                concentrations_mol_m3={
                    cation_component.species_name: float(cation_concentration),
                    anion_component.species_name: float(anion_concentration),
                    cluster_template.label: float(cluster_concentration),
                },
            )
        )
    return tuple(partitions)


def _pair_solvation_competition_penalty_J_mol(
    first_component: IonComponent,
    second_component: IonComponent,
    solvent_environment: MolecularSolventEnvironment,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> float:
    if first_component.charge_number * second_component.charge_number >= 0:
        return 0.0
    solvent_competition_affinity_J_mol = _solvent_shell_competition_affinity_J_mol(
        solvent_environment
    )
    ion_pair_coordination_affinity_J_mol = PAIR_COORDINATION_AVERAGE_FACTOR * (
        first_component.descriptor.coordination_affinity_J_mol
        + second_component.descriptor.coordination_affinity_J_mol
    )
    competition_penalty_J_mol = (
        solvent_competition_affinity_J_mol - ion_pair_coordination_affinity_J_mol
    )
    if competition_penalty_J_mol <= 0.0:
        return 0.0
    ionic_strength_ratio = _analytical_ionic_strength_ratio(
        (first_component, second_component)
    )
    crowding_denominator = 1.0 + (
        primitive_parameters.association_crowding_stabilization_scale
        * ionic_strength_ratio
        ** primitive_parameters.association_crowding_ionic_strength_exponent
    )
    return float(
        competition_penalty_J_mol
        / _positive_float(crowding_denominator, "solvation_competition_crowding")
    )


def _solvent_shell_competition_affinity_J_mol(
    solvent_environment: MolecularSolventEnvironment,
) -> float:
    return _positive_float(
        solvent_environment.solvent_coordination_affinity_J_mol,
        "solvent_coordination_affinity_J_mol",
    )


def _solvation_competition_basin_boundaries_A(
    cation_component: IonComponent,
    anion_component: IonComponent,
    solvent_environment: MolecularSolventEnvironment,
) -> Mapping[str, tuple[float, float]]:
    contact_upper_A = (
        cation_component.descriptor.cavity_radius_A
        + anion_component.descriptor.cavity_radius_A
    )
    ssip_upper_A = contact_upper_A + (
        2.0 * solvent_environment.solvent_effective_radius_A
    )
    return {
        CONTACT_PAIR_CLUSTER_KIND: (0.0, float(contact_upper_A)),
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND: (
            float(contact_upper_A),
            float(ssip_upper_A),
        ),
        "free_ion_center": (float(ssip_upper_A), math.inf),
    }


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
    _positive_float(
        solvent_environment.solvent_effective_radius_A,
        "solvent_effective_radius_A",
    )
    _positive_float(
        solvent_environment.mean_molecular_volume_A3,
        "mean_molecular_volume_A3",
    )
    _positive_float(
        solvent_environment.solvent_coordination_affinity_J_mol,
        "solvent_coordination_affinity_J_mol",
    )
    solvent_fraction_sum = math.fsum(
        _nonnegative_float(volume_fraction, f"{solvent_name}.volume_fraction")
        for solvent_name, volume_fraction in solvent_environment.solvent_volume_fractions.items()
    )
    _positive_float(solvent_fraction_sum, "solvent_volume_fraction_sum")
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
