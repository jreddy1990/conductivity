"""Descriptor-driven molecular electrolyte conductivity through Markov-additive GK."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from constants import K_B, N_A, R
from conductivity.finite_markov_additive_green_kubo import (
    MarkovAdditiveConductivityInput,
    MarkovAdditiveConductivityResult,
    MarkovAdditiveEvent,
    compute_markov_additive_green_kubo_conductivity,
)
from conductivity.generic_speciation import (
    ANGSTROM_TO_M,
    CONTACT_PAIR_CLUSTER_KIND,
    CUBIC_ANGSTROM_TO_CUBIC_M,
    ClusterEnumerationOptions,
    ClusterStateTemplate,
    GenericSpeciationResult,
    HIGHER_CHARGED_CLUSTER_KIND,
    IonComponent,
    MASS_BALANCE_TOLERANCE_FACTOR,
    MolecularSolventEnvironment,
    NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    NEUTRAL_CLUSTER_KIND,
    POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    STANDARD_STATE_CONCENTRATION_MOL_M3,
    SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    build_cluster_state_templates,
    solve_generic_mass_balance,
)
from conductivity.ion_atmosphere import (
    IonAtmosphereInput,
    build_ion_atmosphere_state,
)
from conductivity.molecular_descriptors import (
    MolecularDescriptorBackend,
    MolecularSpeciesDescriptor,
    MolecularSpeciesInput,
    ROLE_ADDITIVE,
    ROLE_ANION,
    ROLE_CATION,
    ROLE_SOLVENT,
)
from conductivity.molecular_primitive_parameters import (
    ConductivityPrimitiveParameterSet,
    validate_conductivity_primitive_parameters,
)


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
        solvent_volume_fractions=dict(recipe.solvents),
        solvent_coordination_affinity_J_mol=(
            _mixture_solvent_coordination_affinity_J_mol(recipe, descriptors)
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


def _mixture_solvent_coordination_affinity_J_mol(
    recipe: MolecularElectrolyteRecipe,
    descriptors: Mapping[str, MolecularSpeciesDescriptor],
) -> float:
    solvent_fraction_sum = math.fsum(
        _nonnegative_float(volume_fraction, f"{species_name}.volume_fraction")
        for species_name, volume_fraction in recipe.solvents.items()
    )
    _positive_float(solvent_fraction_sum, "solvent_volume_fraction_sum")
    weighted_coordination_affinity_J_mol = 0.0
    for species_name, volume_fraction in recipe.solvents.items():
        descriptor = descriptors[species_name]
        weighted_coordination_affinity_J_mol += (
            _nonnegative_float(volume_fraction, f"{species_name}.volume_fraction")
            * _positive_float(
                descriptor.coordination_affinity_J_mol,
                f"{species_name}.coordination_affinity_J_mol",
            )
        )
    return _positive_float(
        weighted_coordination_affinity_J_mol / solvent_fraction_sum,
        "mixture_solvent_coordination_affinity_J_mol",
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
