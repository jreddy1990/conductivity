"""Recipe and physical-library builders for projected generator inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from pathlib import Path

import numpy as np

from constants import R
from conductivity.physical_library.basin_builder import (
    OrientationBasin,
    assign_orientation_basin,
    compute_role_coordination_number,
)
from conductivity.physical_library.mixture_closures import (
    MixtureClosureResult,
    MixtureComposition,
    compute_mixture_closures,
)
from conductivity.physical_library.physical_generator_builder import (
    LOCAL_FIELD_VECTOR_LENGTH,
    PhysicalGeneratorBuildInput,
    PhysicalLocalFields,
    PhysicalStateQuadrature,
    PhysicalTransitionQuadrature,
    build_reduced_generator_specification_from_physical_objects,
)
from conductivity.physical_library.physical_objects import (
    CARTESIAN_DIMENSION,
    PairBasin,
    SiteConfiguration,
    build_physical_objects,
    compute_atmosphere_resistance_diagnostics,
    compute_charge_polarization_m,
    compute_local_packing_fraction,
)
from conductivity.physical_library.projected_analytical_conductivity import (
    ProjectedConductivityResult,
    _compute_projected_analytical_conductivity_from_input,
)
from conductivity.physical_library.library_io import (
    PhysicalLibraryRecords,
    RecipeBuildResult,
    RecipeComponentLoading,
    build_recipe_library_context,
)
from conductivity.physical_library.reduced_generator import build_projected_generator_input
from conductivity.physical_library.reduced_generator import ReducedGeneratorSpecification
from conductivity.physical_library.transition_surface_builder import (
    MomentBoundaryValueInput,
    OneDimensionalCommittorInput,
    solve_endpoint_moment_bvp,
    solve_one_dimensional_committor,
)

Array = np.ndarray
NO_TRANSITION_FAMILY = ""
STATE_KEY_PAIR_FIELD = "pair"
STATE_KEY_SHELL_FIELD = "shell"
STATE_KEY_LIGAND_FIELD = "ligand"
STATE_KEY_ANION_FIELD = "anion"
STATE_KEY_ORIENTATION_FIELD = "orientation"
STATE_KEY_CLUSTER_FIELD = "cluster"
STATE_KEY_PARTNER_FIELD = "partner"
STATE_KEY_IDENTITY_FIELD = "identity"
STATE_KEY_HOP_FIELD = "hop"
STATE_KEY_CAGE_FIELD = "cage"
STATE_KEY_ENVIRONMENT_FIELD = "environment"
STATE_KEY_ATMOSPHERE_FIELD = "atmosphere"
STATE_KEY_FIELDS = (
    STATE_KEY_PAIR_FIELD,
    STATE_KEY_SHELL_FIELD,
    STATE_KEY_LIGAND_FIELD,
    STATE_KEY_ANION_FIELD,
    STATE_KEY_ORIENTATION_FIELD,
    STATE_KEY_CLUSTER_FIELD,
    STATE_KEY_PARTNER_FIELD,
    STATE_KEY_IDENTITY_FIELD,
    STATE_KEY_HOP_FIELD,
    STATE_KEY_CAGE_FIELD,
    STATE_KEY_ENVIRONMENT_FIELD,
    STATE_KEY_ATMOSPHERE_FIELD,
)
STATE_KEY_LENGTH = len(STATE_KEY_FIELDS)
STATE_KEY_PAIR_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_PAIR_FIELD)
STATE_KEY_SHELL_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_SHELL_FIELD)
STATE_KEY_LIGAND_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_LIGAND_FIELD)
STATE_KEY_ANION_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_ANION_FIELD)
STATE_KEY_ORIENTATION_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_ORIENTATION_FIELD)
STATE_KEY_CLUSTER_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_CLUSTER_FIELD)
STATE_KEY_PARTNER_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_PARTNER_FIELD)
STATE_KEY_IDENTITY_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_IDENTITY_FIELD)
STATE_KEY_HOP_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_HOP_FIELD)
STATE_KEY_CAGE_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_CAGE_FIELD)
STATE_KEY_ENVIRONMENT_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_ENVIRONMENT_FIELD)
STATE_KEY_ATMOSPHERE_INDEX = STATE_KEY_FIELDS.index(STATE_KEY_ATMOSPHERE_FIELD)
STATE_KEY_COMPONENT_SEPARATOR = ":"
NO_ACTIVE_ANION_COMPONENT = "__no_active_anion_component__"
NO_ACTIVE_ADDITIVE_COMPONENT = "__no_active_additive_component__"
LOCAL_FIELD_LAW_REQUIRED_KEYS = (
    "dielectric_decrement_per_mol_m3",
    "jones_dole_A_sqrt_m3_mol",
    "jones_dole_B_m3_mol",
    "additive_microviscosity_coefficients",
    "local_ionic_strength_packing_coupling",
    "local_ionic_strength_kernel_width_m",
    "packing_viscosity_exponent",
)
PAIR_STATE_FREE_ADDITIVE_RESERVOIR = "free_additive_reservoir"
LIGAND_STATE_FREE_ADDITIVE_RESERVOIR = "free_ligand_reservoir"
LIGAND_STATE_ADDITIVE_SEPARATOR = "additive_separator"
PAIR_TRANSITION_KEY_INDICES = (
    STATE_KEY_SHELL_INDEX,
    STATE_KEY_LIGAND_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
LIGAND_TRANSITION_KEY_INDICES = (
    STATE_KEY_PAIR_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
ADDITIVE_SEPARATED_PAIR_TRANSITION_KEY_INDICES = (
    STATE_KEY_SHELL_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
SHELL_TRANSITION_KEY_INDICES = (
    STATE_KEY_PAIR_INDEX,
    STATE_KEY_LIGAND_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
CLUSTER_TRANSITION_KEY_INDICES = (
    STATE_KEY_PAIR_INDEX,
    STATE_KEY_SHELL_INDEX,
    STATE_KEY_LIGAND_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
PARTNER_TRANSITION_KEY_INDICES = (
    STATE_KEY_PAIR_INDEX,
    STATE_KEY_SHELL_INDEX,
    STATE_KEY_LIGAND_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
IDENTITY_TRANSITION_KEY_INDICES = (
    STATE_KEY_PAIR_INDEX,
    STATE_KEY_SHELL_INDEX,
    STATE_KEY_LIGAND_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
HOP_TRANSITION_KEY_INDICES = (
    STATE_KEY_PAIR_INDEX,
    STATE_KEY_SHELL_INDEX,
    STATE_KEY_LIGAND_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
CAGE_TRANSITION_KEY_INDICES = (
    STATE_KEY_PAIR_INDEX,
    STATE_KEY_SHELL_INDEX,
    STATE_KEY_LIGAND_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
    STATE_KEY_ATMOSPHERE_INDEX,
)
ATMOSPHERE_TRANSITION_KEY_INDICES = (
    STATE_KEY_PAIR_INDEX,
    STATE_KEY_SHELL_INDEX,
    STATE_KEY_LIGAND_INDEX,
    STATE_KEY_ANION_INDEX,
    STATE_KEY_ORIENTATION_INDEX,
    STATE_KEY_CLUSTER_INDEX,
    STATE_KEY_PARTNER_INDEX,
    STATE_KEY_IDENTITY_INDEX,
    STATE_KEY_HOP_INDEX,
    STATE_KEY_CAGE_INDEX,
    STATE_KEY_ENVIRONMENT_INDEX,
)


class SpeciesRole(Enum):
    CATION = "cation"
    ANION = "anion"
    SOLVENT = "solvent"
    ADDITIVE = "additive"


class ReducedCoordinate(Enum):
    LI_ANION_DISTANCE = "Li_anion_distance"
    LI_SOLVENT_COORDINATION = "Li_solvent_coordination"
    LI_LIGAND_COORDINATION = "Li_ligand_coordination"
    LI_ANION_COORDINATION = "Li_anion_coordination"
    ANION_ORIENTATION = "anion_orientation"
    LOCAL_PACKING_FRACTION = "local_packing_fraction"
    LOCAL_IONIC_STRENGTH = "local_ionic_strength"
    LOCAL_DIELECTRIC = "local_dielectric"
    LOCAL_VISCOSITY = "local_viscosity"
    ATMOSPHERE_POLARIZATION = "atmosphere_polarization"
    CAGE_COORDINATE = "cage_coordinate"
    PARTNER_RESIDENCE_COORDINATE = "partner_residence_coordinate"
    CLUSTER_COORDINATE = "cluster_coordinate"
    IDENTITY_COORDINATE = "identity_coordinate"
    STRUCTURAL_HOP_COORDINATE = "structural_hop_coordinate"


class MemoryCoordinateFamily(Enum):
    ATMOSPHERE_POLARIZATION = "atmosphere_polarization"
    CAGE_BACKJUMP = "cage_backjump"
    PARTNER_RESIDENCE = "partner_residence"
    LIGAND_SHELL = "ligand_shell"
    ANION_ORIENTATION = "anion_orientation"
    FREE_VOLUME_STRESS = "free_volume_stress"
    BOUNDED_INTERNAL_POLARIZATION = "bounded_internal_polarization"


PERPENDICULAR_AXIS_ALIGNMENT_LIMIT = 0.99  # Numerical sentinel: switch helper axis before near-collinear cross construction.
FINITE_DIFFERENCE_STEP_M = 1.0e-12  # Site-coordinate gradient step for reduced scalar coordinates.
CENTRAL_DIFFERENCE_WIDTH = 2.0
ORIENTATION_MEMORY_VALUES = {
    OrientationBasin.RADIAL: 1.0,
    OrientationBasin.BRIDGING: -1.0,
    OrientationBasin.TANGENTIAL: 0.0,
    OrientationBasin.UNASSIGNED: 0.0,
}


@dataclass(frozen=True)
class NumericalOptions:
    reference_box_lengths_m: Array
    volume_m3: float
    state_quadrature_order: int
    transition_grid_count: int


@dataclass(frozen=True)
class MemoryCoordinate:
    family: MemoryCoordinateFamily
    records: PhysicalLibraryRecords
    value_function: Callable[[PhysicalLibraryRecords, SiteConfiguration], float]
    gradient_function: Callable[[PhysicalLibraryRecords, SiteConfiguration], Array]


@dataclass(frozen=True)
class TransitionEdge:
    from_state_index: int
    to_state_index: int
    family: str


@dataclass(frozen=True)
class StateChargeMobilityDiagnostics:
    charge_mobility_m2_s: float
    cation_mobility_m2_s: float
    anion_mobility_m2_s: float
    cation_anion_cross_mobility_m2_s: float
    cation_anion_center_mobility_m2_s: float
    charged_center_labels: tuple[str, ...]
    charged_center_charge_numbers: tuple[float, ...]
    charged_center_mobility_matrix_m2_s: tuple[tuple[float, ...], ...]
    potential_energy_J_mol: float
    dielectric_constant: float
    viscosity_Pa_s: float
    ionic_strength_mol_m3: float
    local_packing_fraction: float
    atmosphere_electrophoretic_trace_kg_s: float
    atmosphere_relaxation_trace_kg_s: float
    atmosphere_cation_diagonal_trace_kg_s: float
    atmosphere_anion_diagonal_trace_kg_s: float
    atmosphere_cation_anion_cross_trace_kg_s: float
    atmosphere_mean_charge_cloud_form_factor: float
    atmosphere_mean_state_geometry_form_factor: float
    atmosphere_minimum_separation_over_debye_length: float
    atmosphere_debye_falkenhagen_time_s: float


@dataclass
class StateQuadratureGroup:
    state_key: tuple[str, ...]
    configurations: list[SiteConfiguration]
    coordinate_values: list[dict[str, float]]
    local_fields: list[PhysicalLocalFields]
    weights: list[float]


def compute_conductivity_from_recipe(
    recipe: Path,
    library_root: Path,
    numerical_options: NumericalOptions,
) -> ProjectedConductivityResult:
    recipe_context = build_recipe_library_context(recipe, library_root)
    records = recipe_context.library_records
    _validate_active_local_field_laws(records)
    mixture = compute_mixture_closures(
        records=records,
        composition=mixture_composition_from_recipe_context(recipe_context),
        temperature_K=recipe_context.temperature_K,
    )
    template_configuration = build_template_site_configuration(
        records=records,
        recipe_context=recipe_context,
        mixture=mixture,
        numerical_options=numerical_options,
    )
    state_quadratures = build_all_state_quadratures(
        records=records,
        template_configuration=template_configuration,
        mixture=mixture,
        recipe_context=recipe_context,
        numerical_options=numerical_options,
    )
    transition_quadratures = build_all_transition_quadratures(
        records=records,
        state_quadratures=state_quadratures,
        template_configuration=template_configuration,
        mixture=mixture,
        temperature_K=recipe_context.temperature_K,
        numerical_options=numerical_options,
    )
    memory_gradient_functions = build_all_memory_coordinate_gradients(
        records=records,
        template_configuration=template_configuration,
        state_quadratures=state_quadratures,
        mixture=mixture,
        numerical_options=numerical_options,
    )
    projected_components = _projected_mass_balance_components(recipe_context)
    component_concentrations = np.asarray(
        [component.concentration_mol_m3 for component in projected_components],
        dtype=float,
    )
    reduced_specification = build_reduced_generator_specification_from_physical_objects(
        PhysicalGeneratorBuildInput(
            records=records,
            template_configuration=template_configuration,
            state_quadratures=state_quadratures,
            transition_quadratures=transition_quadratures,
            memory_coordinate_gradient_functions=memory_gradient_functions,
            total_component_concentrations_mol_m3=component_concentrations,
            temperature_K=recipe_context.temperature_K,
            volume_m3=numerical_options.volume_m3,
        )
    )
    generator_input = build_projected_generator_input(
        _normalize_potential_energy_reference(reduced_specification)
    )
    conductivity_result = _compute_projected_analytical_conductivity_from_input(
        generator_input
    )
    transition_edges = finite_generator_transition_edges(
        records,
        state_quadratures,
        recipe_context.temperature_K,
    )
    _validate_transition_rate_bounds(
        records,
        transition_edges,
        transition_quadratures,
        conductivity_result.reversible_generator_Q_ij_s_inv,
        recipe_context.temperature_K,
    )
    _annotate_transition_generator_diagnostics(
        conductivity_result,
        records,
        transition_edges,
        transition_quadratures,
        recipe_context.temperature_K,
    )
    _annotate_state_charge_mobility_diagnostics(
        conductivity_result,
        records,
        state_quadratures,
        recipe_context.temperature_K,
    )
    _annotate_component_mass_balance_diagnostics(
        conductivity_result,
        projected_components,
        state_quadratures,
    )
    return conductivity_result


def _validate_active_local_field_laws(records: PhysicalLibraryRecords) -> None:
    active_local_field_coordinates = tuple(
        coordinate.value
        for coordinate in (
            ReducedCoordinate.LOCAL_IONIC_STRENGTH,
            ReducedCoordinate.LOCAL_DIELECTRIC,
            ReducedCoordinate.LOCAL_VISCOSITY,
            ReducedCoordinate.LOCAL_PACKING_FRACTION,
            ReducedCoordinate.ATMOSPHERE_POLARIZATION,
        )
        if coordinate.value in records.basis_record["active_state_axis_coordinates"]
        or coordinate.value in records.basis_record["coordinate_domains"]
    )
    if not active_local_field_coordinates:
        return
    if "local_fields" not in records.mixture_record:
        raise KeyError(
            "mixture.yaml missing local_fields while local-field coordinates are active: "
            f"{active_local_field_coordinates}"
        )
    local_field_record = records.mixture_record["local_fields"]
    missing_keys = tuple(
        required_key
        for required_key in LOCAL_FIELD_LAW_REQUIRED_KEYS
        if required_key not in local_field_record
    )
    if missing_keys:
        raise KeyError(
            "mixture.local_fields missing required active local-field laws: "
            f"{missing_keys}"
        )


def _annotate_component_mass_balance_diagnostics(
    conductivity_result: ProjectedConductivityResult,
    projected_components: tuple[RecipeComponentLoading, ...],
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
) -> None:
    component_totals_mol_m3 = np.asarray(
        [component.concentration_mol_m3 for component in projected_components],
        dtype=float,
    )
    state_stoichiometry = np.asarray(
        [state_quadrature.stoichiometry for state_quadrature in state_quadratures],
        dtype=float,
    )
    if state_stoichiometry.ndim != 2:
        raise ValueError("state stoichiometry diagnostics require a 2D matrix")
    if state_stoichiometry.shape[1] != component_totals_mol_m3.size:
        raise ValueError("state stoichiometry component count mismatch")
    component_residuals_mol_m3 = (
        state_stoichiometry.T
        @ np.asarray(conductivity_result.state_concentrations_mol_m3, dtype=float)
        - component_totals_mol_m3
    )
    conductivity_result.effect_attribution.update(
        {
            "component_names": tuple(component.name for component in projected_components),
            "component_total_concentrations_mol_m3": component_totals_mol_m3,
            "state_stoichiometry": state_stoichiometry,
            "component_mass_balance_residuals_mol_m3": component_residuals_mol_m3,
            "state_additive_stoichiometry": np.asarray(
                [
                    _state_additive_stoichiometry(
                        _state_key_from_label(state_quadrature.label)
                    )
                    for state_quadrature in state_quadratures
                ],
                dtype=float,
            ),
        }
    )


def mixture_composition_from_recipe_context(
    recipe_context: RecipeBuildResult,
) -> MixtureComposition:
    ion_concentrations_mol_m3: dict[str, float] = {}
    for component in recipe_context.components:
        role = _species_role(recipe_context.library_records, component.name)
        if role in (SpeciesRole.CATION, SpeciesRole.ANION):
            ion_concentrations_mol_m3[component.name] = component.concentration_mol_m3
    return MixtureComposition(
        solvent_volume_fractions=recipe_context.solvent_volume_fractions,
        ion_concentrations_mol_m3=ion_concentrations_mol_m3,
        additive_weight_fractions=recipe_context.additive_weight_fractions,
    )


def build_template_site_configuration(
    records: PhysicalLibraryRecords,
    recipe_context: RecipeBuildResult,
    mixture: MixtureClosureResult,
    numerical_options: NumericalOptions,
) -> SiteConfiguration:
    _validate_numerical_options(numerical_options)
    _positive_float(mixture.viscosity_Pa_s, "mixture.viscosity_Pa_s")
    species_names: list[str] = []
    molecule_ids: list[int] = []
    site_ids: list[int] = []
    positions: list[Array] = []
    molecule_id = 0
    for component in recipe_context.components:
        if component.concentration_mol_m3 <= 0.0:
            continue
        species_record = records.species_records[component.name]
        conformer_coordinates = np.asarray(
            species_record["reference_conformer_coordinates_m"],
            dtype=float,
        )
        offset = _reference_offset_m(records, component.name, molecule_id)
        for site_index, site_record in enumerate(species_record["sites"]):
            species_names.append(component.name)
            molecule_ids.append(molecule_id)
            site_ids.append(int(site_record["site_id"]))
            positions.append(conformer_coordinates[site_index] + offset)
        molecule_id += 1
    if not species_names:
        raise ValueError("recipe generated no template sites")
    configuration = SiteConfiguration(
        species_names=tuple(species_names),
        molecule_ids=np.asarray(molecule_ids, dtype=int),
        site_ids=np.asarray(site_ids, dtype=int),
        positions_m=np.asarray(positions, dtype=float),
        unwrapped_positions_m=np.asarray(positions, dtype=float),
        box_lengths_m=np.asarray(numerical_options.reference_box_lengths_m, dtype=float),
    )
    return _configuration_with_pair_distance(
        records,
        configuration,
        float(records.basis_record["pair_basins"]["r_SSIP_m"]),
    )


def build_all_state_quadratures(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
    recipe_context: RecipeBuildResult,
    numerical_options: NumericalOptions,
) -> tuple[PhysicalStateQuadrature, ...]:
    declared_coordinates = _declared_reduced_coordinates(records)
    component_names = tuple(
        component.name for component in _projected_mass_balance_components(recipe_context)
    )
    anion_component_names = tuple(
        component.name
        for component in _projected_mass_balance_components(recipe_context)
        if _species_role(records, component.name) == SpeciesRole.ANION
    )
    additive_component_names = tuple(
        component.name
        for component in _projected_mass_balance_components(recipe_context)
        if _species_role(records, component.name) == SpeciesRole.ADDITIVE
    )
    if not anion_component_names:
        raise ValueError("projected conductivity state builder needs anion components")
    quadratures: list[PhysicalStateQuadrature] = []
    for label, lower_distance_m, upper_distance_m in _state_distance_bounds(
        records,
        recipe_context,
    ):
        coordinate_nodes = _state_coordinate_nodes(
            records,
            declared_coordinates,
            lower_distance_m,
            upper_distance_m,
            recipe_context,
            mixture,
            numerical_options,
        )
        grouped_quadrature = _group_state_quadrature_nodes(
            records,
            template_configuration,
            mixture,
            label,
            coordinate_nodes,
        )
        if not grouped_quadrature:
            raise ValueError("state quadrature generated no configurations")
        for state_label, state_group in grouped_quadrature.items():
            for anion_component_name in anion_component_names:
                state_key = _state_key_with_active_anion(
                    state_group.state_key,
                    anion_component_name,
                )
                if _state_requires_additive_component(state_key):
                    if not additive_component_names:
                        raise ValueError("additive state generated without additive component")
                    active_additive_component_names = additive_component_names
                else:
                    active_additive_component_names = (NO_ACTIVE_ADDITIVE_COMPONENT,)
                for active_additive_component_name in active_additive_component_names:
                    additive_state_key = _state_key_with_active_additive(
                        state_key,
                        active_additive_component_name,
                    )
                    state_label_with_components = "|".join(additive_state_key)
                    configurations = tuple(
                        _state_local_transport_configuration(
                            records,
                            configuration,
                            anion_component_name,
                            active_additive_component_name,
                            additive_state_key,
                        )
                        for configuration in state_group.configurations
                    )
                    local_fields = tuple(
                        _local_fields_for_coordinate_values(
                            records,
                            configuration,
                            coordinate_values,
                        )
                        for configuration, coordinate_values in zip(
                            configurations,
                            state_group.coordinate_values,
                            strict=True,
                        )
                    )
                    representative_configuration = configurations[
                        len(configurations) // 2
                    ]
                    quadratures.append(
                        PhysicalStateQuadrature(
                            label=state_label_with_components,
                            configurations=configurations,
                            local_fields=local_fields,
                            weights=np.asarray(state_group.weights, dtype=float),
                            stoichiometry=_transport_state_stoichiometry(
                                records,
                                component_names,
                                anion_component_name,
                                active_additive_component_name,
                                additive_state_key,
                            ),
                            self_current_projector=build_self_current_projector(
                                state_key=additive_state_key,
                                configuration=representative_configuration,
                                records=records,
                            ),
                        ),
                    )
    quadratures.extend(
        _additive_reservoir_state_quadratures(
            records,
            template_configuration,
            mixture,
            component_names,
            additive_component_names,
        )
    )
    return _filter_state_quadratures_by_partition_weight(
        records,
        tuple(quadratures),
        recipe_context.temperature_K,
    )


def _additive_reservoir_state_quadratures(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
    component_names: tuple[str, ...],
    additive_component_names: tuple[str, ...],
) -> list[PhysicalStateQuadrature]:
    quadratures: list[PhysicalStateQuadrature] = []
    for additive_component_name in additive_component_names:
        state_key = _additive_reservoir_state_key(additive_component_name)
        configuration = _single_molecule_configuration_for_species(
            template_configuration,
            additive_component_name,
        )
        local_fields = _local_fields_for_coordinate_values(
            records,
            configuration,
            _bulk_local_field_coordinate_values(records, mixture),
        )
        quadratures.append(
            PhysicalStateQuadrature(
                label="|".join(state_key),
                configurations=(configuration,),
                local_fields=(local_fields,),
                weights=np.asarray([1.0], dtype=float),
                stoichiometry=_transport_state_stoichiometry(
                    records,
                    component_names,
                    active_anion_component_name=NO_ACTIVE_ANION_COMPONENT,
                    active_additive_component_name=additive_component_name,
                    state_key=state_key,
                ),
                self_current_projector=build_self_current_projector(
                    state_key=state_key,
                    configuration=configuration,
                    records=records,
                ),
            )
        )
    return quadratures


def _filter_state_quadratures_by_partition_weight(
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    temperature_K: float,
) -> tuple[PhysicalStateQuadrature, ...]:
    if not state_quadratures:
        raise ValueError("state quadrature builder produced no states")
    log_partition_values = np.asarray(
        [
            _state_log_partition_value(records, state_quadrature, temperature_K)
            for state_quadrature in state_quadratures
        ],
        dtype=float,
    )
    reference_log_partition = _finite_float(
        float(np.max(log_partition_values)),
        "reference_log_partition",
    )
    population_cutoff = _nonnegative_float(
        records.basis_record["population_cutoff"],
        "basis.population_cutoff",
    )
    if population_cutoff == 0.0:
        retained = tuple(state_quadratures)
    else:
        retained = tuple(
            state_quadrature
            for state_quadrature, log_partition_value in zip(
                state_quadratures,
                log_partition_values,
                strict=True,
            )
            if _state_key_from_label(state_quadrature.label)[STATE_KEY_PAIR_INDEX]
            == PAIR_STATE_FREE_ADDITIVE_RESERVOIR
            or float(np.exp(log_partition_value - reference_log_partition))
            > population_cutoff
        )
    if not retained:
        raise ValueError("population cutoff removed every generated state")
    return retained


def _state_key_with_active_anion(
    state_key: tuple[str, ...],
    anion_component_name: str,
) -> tuple[str, ...]:
    if len(state_key) != STATE_KEY_LENGTH:
        raise ValueError("state key has wrong length")
    key_parts = list(state_key)
    key_parts[STATE_KEY_ANION_INDEX] = (
        f"{anion_component_name}{STATE_KEY_COMPONENT_SEPARATOR}{key_parts[STATE_KEY_ANION_INDEX]}"
    )
    return tuple(key_parts)


def _state_key_with_active_additive(
    state_key: tuple[str, ...],
    active_additive_component_name: str,
) -> tuple[str, ...]:
    if len(state_key) != STATE_KEY_LENGTH:
        raise ValueError("state key has wrong length")
    state_requires_additive = _state_requires_additive_component(state_key)
    if active_additive_component_name == NO_ACTIVE_ADDITIVE_COMPONENT:
        if state_requires_additive:
            raise ValueError("additive state needs an active additive component")
        return state_key
    if not state_requires_additive:
        raise ValueError("non-additive state received an active additive component")
    key_parts = list(state_key)
    pair_state = _state_key_base_value(key_parts[STATE_KEY_PAIR_INDEX])
    ligand_state = _state_key_base_value(key_parts[STATE_KEY_LIGAND_INDEX])
    if ligand_state == "none":
        if pair_state != "addSSIP":
            raise ValueError("additive-consuming state resolved to ligand none")
        ligand_state = LIGAND_STATE_ADDITIVE_SEPARATOR
    key_parts[STATE_KEY_LIGAND_INDEX] = (
        f"{active_additive_component_name}{STATE_KEY_COMPONENT_SEPARATOR}{ligand_state}"
    )
    return tuple(key_parts)


def _additive_reservoir_state_key(additive_component_name: str) -> tuple[str, ...]:
    return (
        PAIR_STATE_FREE_ADDITIVE_RESERVOIR,
        "solvent_only",
        f"{additive_component_name}{STATE_KEY_COMPONENT_SEPARATOR}{LIGAND_STATE_FREE_ADDITIVE_RESERVOIR}",
        "none",
        "free_rotating",
        "Li+",
        "partner_a",
        "carrier_identity_a",
        "hop_source",
        "cage_released",
        "bulk_environment",
        "atmosphere_relaxed",
    )


def _bulk_local_field_coordinate_values(
    records: PhysicalLibraryRecords,
    mixture: MixtureClosureResult,
) -> dict[str, float]:
    return {
        ReducedCoordinate.LOCAL_PACKING_FRACTION.value: 0.0,
        ReducedCoordinate.LOCAL_IONIC_STRENGTH.value: float(
            mixture.ionic_strength_mol_m3
        ),
        ReducedCoordinate.LOCAL_DIELECTRIC.value: float(mixture.dielectric_constant),
        ReducedCoordinate.LOCAL_VISCOSITY.value: float(mixture.viscosity_Pa_s),
        ReducedCoordinate.LI_LIGAND_COORDINATION.value: 1.0,
        ReducedCoordinate.LI_ANION_DISTANCE.value: float(
            records.basis_record["pair_basins"]["r_free_m"]
        ),
    }


def _single_molecule_configuration_for_species(
    configuration: SiteConfiguration,
    species_name: str,
) -> SiteConfiguration:
    site_indices = _first_molecule_indices_for_species(configuration, species_name)
    positions = np.asarray(configuration.positions_m, dtype=float)[
        np.asarray(site_indices, dtype=int)
    ]
    unwrapped_positions = np.asarray(configuration.unwrapped_positions_m, dtype=float)[
        np.asarray(site_indices, dtype=int)
    ]
    return SiteConfiguration(
        species_names=tuple(configuration.species_names[site_index] for site_index in site_indices),
        molecule_ids=np.zeros(len(site_indices), dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int)[
            np.asarray(site_indices, dtype=int)
        ],
        positions_m=positions,
        unwrapped_positions_m=unwrapped_positions,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _transport_state_stoichiometry(
    records: PhysicalLibraryRecords,
    component_names: tuple[str, ...],
    active_anion_component_name: str,
    active_additive_component_name: str,
    state_key: tuple[str, ...],
) -> Array:
    if len(state_key) != STATE_KEY_LENGTH:
        raise ValueError("state key has wrong length")
    stoichiometry = np.zeros(len(component_names), dtype=float)
    is_additive_reservoir_state = (
        state_key[STATE_KEY_PAIR_INDEX] == PAIR_STATE_FREE_ADDITIVE_RESERVOIR
    )
    for component_index, component_name in enumerate(component_names):
        component_role = _species_role(records, component_name)
        if component_role == SpeciesRole.CATION and not is_additive_reservoir_state:
            stoichiometry[component_index] = 1.0
            continue
        if component_role == SpeciesRole.CATION:
            continue
        if component_role == SpeciesRole.ANION and component_name == active_anion_component_name:
            stoichiometry[component_index] = 1.0
            continue
        if component_role == SpeciesRole.ANION:
            continue
        if component_role == SpeciesRole.ADDITIVE:
            if component_name == active_additive_component_name:
                stoichiometry[component_index] = _state_additive_stoichiometry(state_key)
            continue
        raise ValueError(
            f"projected mass-balance component is not transport-active: {component_name}"
        )
    if float(np.sum(stoichiometry)) <= 0.0:
        raise ValueError("transport state stoichiometry is empty")
    return stoichiometry


def _state_additive_stoichiometry(state_key: tuple[str, ...]) -> float:
    if _state_requires_additive_component(state_key):
        return 1.0
    return 0.0


def _state_requires_additive_component(state_key: tuple[str, ...]) -> bool:
    pair_state = _state_key_base_value(state_key[STATE_KEY_PAIR_INDEX])
    shell_state = _state_key_base_value(state_key[STATE_KEY_SHELL_INDEX])
    ligand_state = _state_key_base_value(state_key[STATE_KEY_LIGAND_INDEX])
    cluster_state = _state_key_base_value(state_key[STATE_KEY_CLUSTER_INDEX])
    additive_states = {
        "addSSIP",
        "neutral_ligand_bound",
        "mixed_ligand_anion",
        "monodentate",
        "multidentate",
        LIGAND_STATE_ADDITIVE_SEPARATOR,
        "bridge",
        "Li_ligand_anion",
        LIGAND_STATE_FREE_ADDITIVE_RESERVOIR,
    }
    if (
        pair_state in additive_states
        or shell_state in additive_states
        or ligand_state in additive_states
        or cluster_state in additive_states
    ):
        return True
    return False


def _state_key_base_value(state_key_value: str) -> str:
    if STATE_KEY_COMPONENT_SEPARATOR not in state_key_value:
        return state_key_value
    return state_key_value.split(STATE_KEY_COMPONENT_SEPARATOR, maxsplit=1)[1]


def _configuration_with_active_anion_species(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    active_anion_component_name: str,
) -> SiteConfiguration:
    if active_anion_component_name not in configuration.species_names:
        raise ValueError(f"configuration has no active anion {active_anion_component_name}")
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    positions = np.asarray(configuration.positions_m, dtype=float).copy()
    cation_position = positions[cation_index].copy()
    current_anion_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    current_anion_center = np.mean(
        positions[np.asarray(current_anion_indices, dtype=int)],
        axis=0,
    )
    active_pair_distance_m = _positive_float(
        float(np.linalg.norm(current_anion_center - cation_position)),
        "active_pair_distance_m",
    )
    active_anion_indices = _first_molecule_indices_for_species(
        configuration,
        active_anion_component_name,
    )
    active_anion_center = np.mean(
        positions[np.asarray(active_anion_indices, dtype=int)],
        axis=0,
    )
    target_active_center = cation_position + np.asarray(
        [active_pair_distance_m, 0.0, 0.0],
        dtype=float,
    )
    active_shift = target_active_center - active_anion_center
    for site_index in active_anion_indices:
        positions[site_index] += active_shift
    inactive_offset_index = 1
    inactive_distance_m = _positive_float(
        float(records.basis_record["pair_basins"]["r_free_m"]),
        "pair_basins.r_free_m",
    )
    for species_name in sorted(set(configuration.species_names)):
        if species_name == active_anion_component_name:
            continue
        if _species_role(records, species_name) != SpeciesRole.ANION:
            continue
        inactive_indices = _first_molecule_indices_for_species(configuration, species_name)
        inactive_center = np.mean(positions[np.asarray(inactive_indices, dtype=int)], axis=0)
        target_inactive_center = cation_position + np.asarray(
            [
                0.0,
                inactive_distance_m * float(inactive_offset_index),
                inactive_distance_m,
            ],
            dtype=float,
        )
        inactive_shift = target_inactive_center - inactive_center
        for site_index in inactive_indices:
            positions[site_index] += inactive_shift
        inactive_offset_index += 1
    return SiteConfiguration(
        species_names=configuration.species_names,
        molecule_ids=np.asarray(configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=positions,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _state_local_transport_configuration(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    active_anion_component_name: str,
    active_additive_component_name: str,
    state_key: tuple[str, ...],
) -> SiteConfiguration:
    anion_resolved_configuration = _configuration_with_active_anion_species(
        records,
        configuration,
        active_anion_component_name,
    )
    additive_resolved_configuration = _configuration_with_active_additive_species(
        records,
        anion_resolved_configuration,
        active_additive_component_name,
        state_key,
    )
    return _configuration_with_state_local_species(
        records,
        additive_resolved_configuration,
        active_anion_component_name,
        active_additive_component_name,
    )


def _configuration_with_active_additive_species(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    active_additive_component_name: str,
    state_key: tuple[str, ...],
) -> SiteConfiguration:
    state_requires_additive = _state_requires_additive_component(state_key)
    if active_additive_component_name == NO_ACTIVE_ADDITIVE_COMPONENT:
        if state_requires_additive:
            raise ValueError("additive-consuming state missing active additive species")
        return configuration
    if not state_requires_additive:
        raise ValueError("active additive species supplied for non-additive state")
    if active_additive_component_name not in configuration.species_names:
        raise ValueError(
            f"configuration has no active additive {active_additive_component_name}"
        )
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    positions = np.asarray(configuration.positions_m, dtype=float).copy()
    cation_position = positions[cation_index].copy()
    active_additive_indices = _first_molecule_indices_for_species(
        configuration,
        active_additive_component_name,
    )
    active_additive_center = np.mean(
        positions[np.asarray(active_additive_indices, dtype=int)],
        axis=0,
    )
    active_ligand_distance_m = _positive_float(
        float(np.linalg.norm(active_additive_center - cation_position)),
        "active_ligand_distance_m",
    )
    target_active_center = cation_position + np.asarray(
        [0.0, active_ligand_distance_m, active_ligand_distance_m],
        dtype=float,
    )
    active_shift = target_active_center - active_additive_center
    for site_index in active_additive_indices:
        positions[site_index] += active_shift
    inactive_offset_index = 1
    inactive_distance_m = _positive_float(
        float(records.basis_record["pair_basins"]["r_free_m"]),
        "pair_basins.r_free_m",
    )
    for species_name in sorted(set(configuration.species_names)):
        if species_name == active_additive_component_name:
            continue
        if _species_role(records, species_name) != SpeciesRole.ADDITIVE:
            continue
        inactive_indices = _first_molecule_indices_for_species(configuration, species_name)
        inactive_center = np.mean(positions[np.asarray(inactive_indices, dtype=int)], axis=0)
        target_inactive_center = cation_position + np.asarray(
            [
                inactive_distance_m * float(inactive_offset_index),
                inactive_distance_m,
                0.0,
            ],
            dtype=float,
        )
        inactive_shift = target_inactive_center - inactive_center
        for site_index in inactive_indices:
            positions[site_index] += inactive_shift
        inactive_offset_index += 1
    return SiteConfiguration(
        species_names=configuration.species_names,
        molecule_ids=np.asarray(configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=positions,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _configuration_with_state_local_species(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    active_anion_component_name: str,
    active_additive_component_name: str,
) -> SiteConfiguration:
    molecule_keys_to_keep: set[tuple[str, int]] = set()
    for site_index, species_name in enumerate(configuration.species_names):
        species_role = _species_role(records, species_name)
        molecule_key = (species_name, int(configuration.molecule_ids[site_index]))
        if species_role == SpeciesRole.ADDITIVE:
            if species_name == active_additive_component_name:
                molecule_keys_to_keep.add(molecule_key)
            continue
        if species_role == SpeciesRole.ANION:
            if species_name == active_anion_component_name:
                molecule_keys_to_keep.add(molecule_key)
            continue
        molecule_keys_to_keep.add(molecule_key)
    if not molecule_keys_to_keep:
        raise ValueError("state-local configuration retained no molecules")
    site_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if (species_name, int(configuration.molecule_ids[site_index]))
        in molecule_keys_to_keep
    )
    if not site_indices:
        raise ValueError("state-local configuration retained no sites")
    return _configuration_from_site_indices(configuration, site_indices)


def _configuration_from_site_indices(
    configuration: SiteConfiguration,
    site_indices: tuple[int, ...],
) -> SiteConfiguration:
    site_index_array = np.asarray(site_indices, dtype=int)
    molecule_id_by_original_key: dict[tuple[str, int], int] = {}
    molecule_ids: list[int] = []
    next_molecule_id = 0
    for site_index in site_indices:
        molecule_key = (
            configuration.species_names[site_index],
            int(configuration.molecule_ids[site_index]),
        )
        if molecule_key not in molecule_id_by_original_key:
            molecule_id_by_original_key[molecule_key] = next_molecule_id
            next_molecule_id += 1
        molecule_ids.append(molecule_id_by_original_key[molecule_key])
    return SiteConfiguration(
        species_names=tuple(configuration.species_names[site_index] for site_index in site_indices),
        molecule_ids=np.asarray(molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int)[site_index_array],
        positions_m=np.asarray(configuration.positions_m, dtype=float)[site_index_array],
        unwrapped_positions_m=np.asarray(
            configuration.unwrapped_positions_m,
            dtype=float,
        )[site_index_array],
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _state_log_partition_value(
    records: PhysicalLibraryRecords,
    state_quadrature: PhysicalStateQuadrature,
    temperature_K: float,
) -> float:
    log_terms = []
    for configuration, local_fields, weight in zip(
        state_quadrature.configurations,
        state_quadrature.local_fields,
        state_quadrature.weights,
        strict=True,
    ):
        physical_objects = build_physical_objects(
            records,
            configuration,
            temperature_K,
            local_fields.dielectric_constant,
            local_fields.viscosity_Pa_s,
            local_fields.ionic_strength_mol_m3,
            local_fields.local_packing_fraction,
        )
        if weight <= 0.0:
            raise ValueError(f"{state_quadrature.label} has nonpositive quadrature weight")
        log_terms.append(
            np.log(float(weight))
            - physical_objects.potential_energy_J_mol / (R * temperature_K)
        )
    term_array = np.asarray(log_terms, dtype=float)
    maximum_log_term = _finite_float(
        float(np.max(term_array)),
        f"{state_quadrature.label}.maximum_log_term",
    )
    return maximum_log_term + float(
        np.log(np.sum(np.exp(term_array - maximum_log_term)))
    )


def build_self_current_projector(
    state_key: tuple[str, ...],
    configuration: SiteConfiguration,
    records: PhysicalLibraryRecords,
) -> Array:
    _ = state_key
    _ = records
    coordinate_count = (
        len(configuration.species_names) * CARTESIAN_DIMENSION + LOCAL_FIELD_VECTOR_LENGTH
    )
    return np.eye(coordinate_count, dtype=float)


def _group_state_quadrature_nodes(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
    pair_label: str,
    coordinate_nodes: tuple[tuple[ReducedCoordinate, Array, Array], ...],
) -> dict[str, StateQuadratureGroup]:
    grouped_quadrature: dict[str, StateQuadratureGroup] = {}
    for node_indices in _state_node_index_tuples(records, coordinate_nodes):
        coordinate_values = {
            coordinate_name.value: float(values[node_index])
            for node_index, (coordinate_name, values, _weights) in zip(
                node_indices,
                coordinate_nodes,
                strict=True,
            )
        }
        coordinate_weight = _coordinate_product_weight(node_indices, coordinate_nodes)
        configuration = _configuration_with_reduced_coordinate_values(
            records,
            template_configuration,
            coordinate_values,
        )
        state_key = _state_key_from_reduced_coordinates(
            records,
            configuration,
            mixture,
            pair_label,
            coordinate_values,
        )
        state_label = "|".join(state_key)
        if state_label not in grouped_quadrature:
            grouped_quadrature[state_label] = StateQuadratureGroup(
                state_key=state_key,
                configurations=[],
                coordinate_values=[],
                local_fields=[],
                weights=[],
            )
        grouped_quadrature[state_label].configurations.append(configuration)
        grouped_quadrature[state_label].coordinate_values.append(coordinate_values)
        grouped_quadrature[state_label].local_fields.append(
            _local_fields_for_coordinate_values(records, configuration, coordinate_values)
        )
        grouped_quadrature[state_label].weights.append(coordinate_weight)
    return grouped_quadrature


def _state_node_index_tuples(
    records: PhysicalLibraryRecords,
    coordinate_nodes: tuple[tuple[ReducedCoordinate, Array, Array], ...],
) -> tuple[tuple[int, ...], ...]:
    state_axis_generation = str(records.basis_record["state_axis_generation"])
    if state_axis_generation == "full_tensor_product":
        return tuple(
            product(
                *(
                    range(values.size)
                    for _coordinate_name, values, _weights in coordinate_nodes
                )
            )
        )
    if state_axis_generation != "sparse_single_axis_plus_pair":
        raise ValueError(f"unsupported basis.state_axis_generation {state_axis_generation}")
    pair_coordinate_index = _coordinate_nodes_index(
        coordinate_nodes,
        ReducedCoordinate.LI_ANION_DISTANCE,
    )
    baseline_indices = tuple(
        _baseline_node_index_for_coordinate(coordinate_name, values)
        for coordinate_name, values, _weights in coordinate_nodes
    )
    index_tuples: list[tuple[int, ...]] = []
    for pair_node_index in range(coordinate_nodes[pair_coordinate_index][1].size):
        pair_baseline_indices = list(baseline_indices)
        pair_baseline_indices[pair_coordinate_index] = pair_node_index
        index_tuples.append(tuple(pair_baseline_indices))
        for coordinate_index, (_coordinate_name, values, _weights) in enumerate(
            coordinate_nodes
        ):
            if coordinate_index == pair_coordinate_index:
                continue
            baseline_node_index = baseline_indices[coordinate_index]
            for node_index in range(values.size):
                if node_index == baseline_node_index:
                    continue
                conditioning_indices = _transport_axis_conditioning_indices(
                    coordinate_nodes,
                    coordinate_index,
                    pair_baseline_indices,
                )
                for conditioned_indices in conditioning_indices:
                    varied_indices = list(conditioned_indices)
                    varied_indices[coordinate_index] = node_index
                    index_tuples.append(tuple(varied_indices))
    return tuple(dict.fromkeys(index_tuples))


def _transport_axis_conditioning_indices(
    coordinate_nodes: tuple[tuple[ReducedCoordinate, Array, Array], ...],
    coordinate_index: int,
    pair_baseline_indices: list[int],
) -> tuple[tuple[int, ...], ...]:
    coordinate = coordinate_nodes[coordinate_index][0]
    if coordinate not in _unit_interval_state_coordinates():
        return (tuple(pair_baseline_indices),)
    conditioned_indices = [tuple(pair_baseline_indices)]
    ligand_coordinate_index = _coordinate_nodes_index(
        coordinate_nodes,
        ReducedCoordinate.LI_LIGAND_COORDINATION,
    )
    ligand_values = coordinate_nodes[ligand_coordinate_index][1]
    if ligand_values.size > 1:
        ligand_bound_indices = list(pair_baseline_indices)
        ligand_bound_indices[ligand_coordinate_index] = ligand_values.size - 1
        conditioned_indices.append(tuple(ligand_bound_indices))
    return tuple(dict.fromkeys(conditioned_indices))


def _coordinate_nodes_index(
    coordinate_nodes: tuple[tuple[ReducedCoordinate, Array, Array], ...],
    coordinate: ReducedCoordinate,
) -> int:
    for coordinate_index, (current_coordinate, _values, _weights) in enumerate(
        coordinate_nodes
    ):
        if current_coordinate == coordinate:
            return coordinate_index
    raise ValueError(f"state coordinate nodes missing {coordinate.value}")


def _baseline_node_index_for_coordinate(
    coordinate: ReducedCoordinate,
    values: Array,
) -> int:
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size == 0:
        raise ValueError("state coordinate values must be a nonempty vector")
    if coordinate in _source_baseline_coordinates():
        return 0
    return int(value_array.size // 2)


def _source_baseline_coordinates() -> set[ReducedCoordinate]:
    return _unit_interval_state_coordinates() | {
        ReducedCoordinate.LI_SOLVENT_COORDINATION,
        ReducedCoordinate.LI_LIGAND_COORDINATION,
        ReducedCoordinate.LI_ANION_COORDINATION,
    }


def _unit_interval_state_coordinates() -> set[ReducedCoordinate]:
    return {
        ReducedCoordinate.ATMOSPHERE_POLARIZATION,
        ReducedCoordinate.CAGE_COORDINATE,
        ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE,
        ReducedCoordinate.CLUSTER_COORDINATE,
        ReducedCoordinate.IDENTITY_COORDINATE,
        ReducedCoordinate.STRUCTURAL_HOP_COORDINATE,
    }


def _coordinate_product_weight(
    node_indices: tuple[int, ...],
    coordinate_nodes: tuple[tuple[ReducedCoordinate, Array, Array], ...],
) -> float:
    weights = [
        weights_for_coordinate[node_index]
        for node_index, (
            _coordinate_name,
            _values,
            weights_for_coordinate,
        ) in zip(node_indices, coordinate_nodes, strict=True)
    ]
    return float(np.prod(np.asarray(weights, dtype=float)))




def build_all_transition_quadratures(
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    template_configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
    temperature_K: float,
    numerical_options: NumericalOptions,
) -> tuple[PhysicalTransitionQuadrature, ...]:
    transitions: list[PhysicalTransitionQuadrature] = []
    transition_edges = finite_generator_transition_edges(
        records,
        state_quadratures,
        temperature_K,
    )
    if not transition_edges and records.transition_record["families"]:
        state_labels = tuple(state_quadrature.label for state_quadrature in state_quadratures)
        raise ValueError(
            "physical-library recipe generated no transition edges for declared "
            f"transition families {tuple(records.transition_record['families'])}; "
            f"retained state labels were {state_labels}"
        )
    for edge in transition_edges:
        transition_record = _transition_family_record(records, edge.family)
        _validate_transition_family_reaction_coordinate(edge.family, transition_record)
        reaction_coordinate = _transition_reaction_coordinate(transition_record)
        lower_coordinate_value, upper_coordinate_value = _transition_coordinate_bounds(
            records,
            reaction_coordinate,
            transition_record,
            state_quadratures[edge.from_state_index],
            state_quadratures[edge.to_state_index],
        )
        reaction_coordinate_grid = np.linspace(
            lower_coordinate_value,
            upper_coordinate_value,
            numerical_options.transition_grid_count,
        )
        base_coordinate_values = _transition_base_coordinate_values(
            records,
            state_quadratures[edge.from_state_index],
            state_quadratures[edge.to_state_index],
            mixture,
        )
        configurations = tuple(
            _configuration_for_transition_coordinate(
                records,
                template_configuration,
                base_coordinate_values,
                transition_record,
                coordinate_value,
            )
            for coordinate_value in reaction_coordinate_grid
        )
        local_fields = tuple(
            _local_fields_for_coordinate_values(
                records,
                configuration,
                {
                    **base_coordinate_values,
                    reaction_coordinate.value: float(coordinate_grid_value),
                },
            )
            for configuration, coordinate_grid_value in zip(
                configurations,
                reaction_coordinate_grid,
                strict=True,
            )
        )
        flat_gradients = np.vstack(
            tuple(
                _reaction_coordinate_gradient(
                    records,
                    configuration,
                    transition_record,
                )
                for configuration in configurations
            )
        )
        physical_objects = tuple(
            build_physical_objects(
                records,
                configuration,
                temperature_K,
                local_fields_at_grid_point.dielectric_constant,
                local_fields_at_grid_point.viscosity_Pa_s,
                local_fields_at_grid_point.ionic_strength_mol_m3,
                local_fields_at_grid_point.local_packing_fraction,
            )
            for configuration, local_fields_at_grid_point in zip(
                configurations,
                local_fields,
                strict=True,
            )
        )
        free_energy = _transition_free_energy_profile_J_mol(
            records,
            reaction_coordinate,
            reaction_coordinate_grid,
            base_coordinate_values,
            state_quadratures[edge.from_state_index].label,
            state_quadratures[edge.to_state_index].label,
            transition_record,
        )
        diffusivity = np.asarray(
            [
                project_diffusivity_onto_reaction_coordinate(
                    physical_objects[grid_index].mobility_tensor_m2_s,
                    flat_gradients[grid_index],
                )
                for grid_index in range(len(configurations))
            ],
            dtype=float,
        )
        committor_result = solve_one_dimensional_committor(
            OneDimensionalCommittorInput(
                grid_points=reaction_coordinate_grid,
                free_energy_J_mol=free_energy,
                diffusivity_m2_s=diffusivity,
                temperature_K=temperature_K,
                left_state_index=0,
                right_state_index=reaction_coordinate_grid.size - 1,
            )
        )
        committor_gradients = committor_result.committor_gradient[:, None] * flat_gradients
        charge_polarization_by_grid = np.asarray(
            [
                compute_charge_polarization_m(records, configuration)
                for configuration in configurations
            ],
            dtype=float,
        )
        if (
            str(transition_record["displacement_policy"]) != "zero"
            and "endpoint_geometry" in transition_record
        ):
            endpoint_displacement_m = _endpoint_geometry_displacement_vector_m(
                records,
                configurations[0],
                transition_record,
                edge.family,
            )
            charge_polarization_by_grid[-1] = (
                charge_polarization_by_grid[0] + endpoint_displacement_m
            )
        reactive_exit_weights = np.zeros(reaction_coordinate_grid.size, dtype=float)
        reactive_exit_weights[0] = 1.0
        moment_input = MomentBoundaryValueInput(
            grid_points=np.asarray(reaction_coordinate_grid, dtype=float),
            free_energy_J_mol=np.asarray(free_energy, dtype=float),
            diffusivity_m2_s=np.asarray(diffusivity, dtype=float),
            committor=np.asarray(committor_result.committor, dtype=float),
            left_boundary_index=0,
            right_boundary_index=reaction_coordinate_grid.size - 1,
            charge_polarization_by_grid=np.asarray(
                charge_polarization_by_grid,
                dtype=float,
            ),
            reactive_exit_weights=np.asarray(reactive_exit_weights, dtype=float),
            temperature_K=temperature_K,
        )
        first_displacement_moment_m, second_displacement_moment_m2 = (
            _transition_displacement_moments(transition_record, moment_input)
        )
        _validate_transition_displacement_policy(
            edge.family,
            transition_record,
            first_displacement_moment_m,
            second_displacement_moment_m2,
        )
        uses_declared_rate_constant = _uses_declared_rate_constant(
            edge.family,
            transition_record,
        )
        transitions.append(
            PhysicalTransitionQuadrature(
                from_state_index=edge.from_state_index,
                to_state_index=edge.to_state_index,
                configurations=configurations,
                local_fields=local_fields,
                weights=committor_result.quadrature_weights,
                committor_gradients=committor_gradients,
                surface_state_indices=np.asarray(
                    [edge.from_state_index] * reaction_coordinate_grid.size,
                    dtype=int,
                ),
                path_start_configurations=(configurations[0],),
                path_end_configurations=(configurations[-1],),
                path_weights=np.asarray([1.0], dtype=float),
                first_displacement_moment_m=first_displacement_moment_m,
                second_displacement_moment_m2=second_displacement_moment_m2,
                log_capacity_integral=committor_result.log_capacity_integral,
                uses_residence_rate_constant=uses_declared_rate_constant,
                residence_rate_constant_s_inv=_transition_residence_rate_constant_s_inv(
                    edge.family,
                    transition_record,
                    temperature_K,
                )
                if uses_declared_rate_constant
                else 0.0,
            )
        )
    return tuple(transitions)


def _transition_reaction_coordinate(transition_record: dict) -> ReducedCoordinate:
    reaction_coordinate_name = str(transition_record["reaction_coordinate"])
    for reduced_coordinate in ReducedCoordinate:
        if reduced_coordinate.value == reaction_coordinate_name:
            return reduced_coordinate
    raise ValueError(f"unsupported transition reaction_coordinate {reaction_coordinate_name}")


def _transition_coordinate_bounds(
    records: PhysicalLibraryRecords,
    reaction_coordinate: ReducedCoordinate,
    transition_record: dict,
    from_state_quadrature: PhysicalStateQuadrature,
    to_state_quadrature: PhysicalStateQuadrature,
) -> tuple[float, float]:
    if reaction_coordinate == ReducedCoordinate.LI_ANION_DISTANCE:
        from_pair_label = from_state_quadrature.label.split("|")[0]
        to_pair_label = to_state_quadrature.label.split("|")[0]
        return _transition_distance_bounds(records, from_pair_label, to_pair_label)
    if reaction_coordinate.value in records.basis_record["coordinate_domains"]:
        domain = records.basis_record["coordinate_domains"][reaction_coordinate.value]
        return _domain_bounds(domain, f"basis.coordinate_domains.{reaction_coordinate.value}")
    if "coordinate_domain" not in transition_record:
        raise KeyError(
            f"transition record for {reaction_coordinate.value} missing coordinate_domain"
        )
    return _domain_bounds(
        transition_record["coordinate_domain"],
        f"transition.coordinate_domain.{reaction_coordinate.value}",
    )


def _domain_bounds(domain_record: dict, label: str) -> tuple[float, float]:
    lower = _finite_float(float(domain_record["lower"]), f"{label}.lower")
    upper = _finite_float(float(domain_record["upper"]), f"{label}.upper")
    if not lower < upper:
        raise ValueError(f"{label} bounds must be increasing")
    return lower, upper


def _transition_base_coordinate_values(
    records: PhysicalLibraryRecords,
    from_state_quadrature: PhysicalStateQuadrature,
    to_state_quadrature: PhysicalStateQuadrature,
    mixture: MixtureClosureResult,
) -> dict[str, float]:
    representative_configuration = from_state_quadrature.configurations[
        len(from_state_quadrature.configurations) // 2
    ]
    pair_label = from_state_quadrature.label.split("|")[0]
    to_pair_label = to_state_quadrature.label.split("|")[0]
    if pair_label != to_pair_label and {pair_label, to_pair_label} != {
        PairBasin.FREE.value,
        PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
    } and {pair_label, to_pair_label} != {
        PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
        PairBasin.CONTACT_ION_PAIR.value,
    }:
        pair_label = PairBasin.SOLVENT_SEPARATED_ION_PAIR.value
    values = {
        ReducedCoordinate.LI_ANION_DISTANCE.value: _representative_pair_distance_m(
            records,
            pair_label,
        ),
        ReducedCoordinate.LI_SOLVENT_COORDINATION.value: compute_role_coordination_number(
            records,
            representative_configuration,
            center_role=SpeciesRole.CATION.value,
            ligand_role=SpeciesRole.SOLVENT.value,
            switch_name="Li_solvent",
        ),
        ReducedCoordinate.LI_LIGAND_COORDINATION.value: compute_role_coordination_number(
            records,
            representative_configuration,
            center_role=SpeciesRole.CATION.value,
            ligand_role=SpeciesRole.ADDITIVE.value,
            switch_name="Li_ligand",
        ),
        ReducedCoordinate.LI_ANION_COORDINATION.value: compute_role_coordination_number(
            records,
            representative_configuration,
            center_role=SpeciesRole.CATION.value,
            ligand_role=SpeciesRole.ANION.value,
            switch_name="Li_anion",
        ),
        ReducedCoordinate.ANION_ORIENTATION.value: 0.0,
        ReducedCoordinate.LOCAL_PACKING_FRACTION.value: compute_local_packing_fraction(
            records,
            representative_configuration,
        ),
        ReducedCoordinate.LOCAL_IONIC_STRENGTH.value: mixture.ionic_strength_mol_m3,
        ReducedCoordinate.LOCAL_DIELECTRIC.value: mixture.dielectric_constant,
        ReducedCoordinate.LOCAL_VISCOSITY.value: mixture.viscosity_Pa_s,
        ReducedCoordinate.ATMOSPHERE_POLARIZATION.value: 0.0,
        ReducedCoordinate.CAGE_COORDINATE.value: 0.0,
        ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value: 0.5,
        ReducedCoordinate.CLUSTER_COORDINATE.value: 0.0,
        ReducedCoordinate.IDENTITY_COORDINATE.value: 0.0,
        ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value: 0.0,
    }
    return values


def _transition_free_energy_profile_J_mol(
    records: PhysicalLibraryRecords,
    reaction_coordinate: ReducedCoordinate,
    reaction_coordinate_grid: Array,
    base_coordinate_values: dict[str, float],
    from_state_label: str,
    to_state_label: str,
    transition_record: dict,
) -> Array:
    lower_coordinate_value = float(reaction_coordinate_grid[0])
    upper_coordinate_value = float(reaction_coordinate_grid[-1])
    coordinate_span = _positive_float(
        upper_coordinate_value - lower_coordinate_value,
        "transition_coordinate_span",
    )
    barrier_J_mol = _nonnegative_float(
        float(transition_record["barrier_J_mol"]),
        f"transition_records.{transition_record['reaction_coordinate']}.barrier_J_mol",
    )
    free_energy_values = []
    for coordinate_value in reaction_coordinate_grid:
        reduced_coordinate_value = (
            float(coordinate_value) - lower_coordinate_value
        ) / coordinate_span
        barrier_shape = 4.0 * reduced_coordinate_value * (1.0 - reduced_coordinate_value)
        free_energy_values.append(
            _reduced_coordinate_free_energy_J_mol(
                records,
                {
                    **base_coordinate_values,
                    reaction_coordinate.value: float(coordinate_value),
                },
                from_state_label,
                to_state_label,
            )
            + barrier_shape * barrier_J_mol
        )
    profile = np.asarray(free_energy_values, dtype=float)
    if not np.all(np.isfinite(profile)):
        raise ValueError("transition reduced free-energy profile must be finite")
    return profile


def _reduced_coordinate_free_energy_J_mol(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
    from_state_label: str,
    to_state_label: str,
) -> float:
    free_energy_terms = records.basis_record["free_energy_terms"]
    pair_energy = _reduced_pair_free_energy_J_mol(
        records,
        float(coordinate_values[ReducedCoordinate.LI_ANION_DISTANCE.value]),
        from_state_label,
        to_state_label,
    )
    coordination_energy = _reduced_coordination_free_energy_J_mol(
        free_energy_terms,
        coordinate_values,
    )
    competition_energy = _reduced_solvation_competition_energy_J_mol(
        free_energy_terms,
        coordinate_values,
    )
    auxiliary_coordinate_energy = _reduced_auxiliary_coordinate_energy_J_mol(
        free_energy_terms,
        coordinate_values,
    )
    return (
        pair_energy
        + coordination_energy
        + competition_energy
        + auxiliary_coordinate_energy
    )


def _reduced_pair_free_energy_J_mol(
    records: PhysicalLibraryRecords,
    li_anion_distance_m: float,
    from_state_label: str,
    to_state_label: str,
) -> float:
    pair_energy_record = records.basis_record["free_energy_terms"]["pair_basin_J_mol"]
    pair_basins = records.basis_record["pair_basins"]
    contact_cutoff_m = float(pair_basins["r_CIP_m"])
    solvent_separated_cutoff_m = float(pair_basins["r_SSIP_m"])
    free_cutoff_m = float(pair_basins["r_free_m"])
    if li_anion_distance_m < contact_cutoff_m:
        return float(pair_energy_record[PairBasin.CONTACT_ION_PAIR.value])
    if li_anion_distance_m < solvent_separated_cutoff_m:
        return float(pair_energy_record[PairBasin.SOLVENT_SEPARATED_ION_PAIR.value])
    if li_anion_distance_m >= free_cutoff_m:
        return float(pair_energy_record[PairBasin.FREE.value])
    left_pair = _state_key_from_label(from_state_label)[STATE_KEY_PAIR_INDEX]
    right_pair = _state_key_from_label(to_state_label)[STATE_KEY_PAIR_INDEX]
    endpoint_labels = (
        _pair_energy_label(left_pair),
        _pair_energy_label(right_pair),
    )
    if endpoint_labels[0] == endpoint_labels[1]:
        return float(pair_energy_record[endpoint_labels[0]])
    endpoint_energy_values = np.asarray(
        [float(pair_energy_record[label]) for label in endpoint_labels],
        dtype=float,
    )
    transition_energy = float(pair_energy_record[PairBasin.TRANSITION.value])
    span_m = free_cutoff_m - solvent_separated_cutoff_m
    if span_m <= 0.0:
        raise ValueError("free pair transition span must be positive")
    normalized_coordinate = (
        li_anion_distance_m - solvent_separated_cutoff_m
    ) / span_m
    linear_endpoint_energy = (
        (1.0 - normalized_coordinate) * endpoint_energy_values[0]
        + normalized_coordinate * endpoint_energy_values[1]
    )
    barrier_shape = 4.0 * normalized_coordinate * (1.0 - normalized_coordinate)
    return float(
        linear_endpoint_energy
        + barrier_shape * (transition_energy - float(np.mean(endpoint_energy_values)))
    )


def _pair_energy_label(pair_label: str) -> str:
    if pair_label == "addSSIP":
        return PairBasin.SOLVENT_SEPARATED_ION_PAIR.value
    if pair_label in ("aggregate", "bridge_network"):
        return PairBasin.CONTACT_ION_PAIR.value
    return pair_label


def _reduced_coordination_free_energy_J_mol(
    free_energy_terms: dict,
    coordinate_values: dict[str, float],
) -> float:
    coordination_record = free_energy_terms["coordination_J_mol"]
    coordinate_map = {
        "Li_solvent": ReducedCoordinate.LI_SOLVENT_COORDINATION.value,
        "Li_ligand": ReducedCoordinate.LI_LIGAND_COORDINATION.value,
        "Li_anion": ReducedCoordinate.LI_ANION_COORDINATION.value,
    }
    total_energy = 0.0
    for coordination_name, coordinate_name in coordinate_map.items():
        total_energy += float(coordination_record[coordination_name]) * float(
            coordinate_values[coordinate_name]
        )
    return float(total_energy)


def _reduced_solvation_competition_energy_J_mol(
    free_energy_terms: dict,
    coordinate_values: dict[str, float],
) -> float:
    competition_record = free_energy_terms["solvation_competition"]
    targets = competition_record["targets"]
    stiffnesses = competition_record["stiffness_J_mol"]
    coordinate_map = {
        "Li_solvent": ReducedCoordinate.LI_SOLVENT_COORDINATION.value,
        "Li_ligand": ReducedCoordinate.LI_LIGAND_COORDINATION.value,
        "Li_anion": ReducedCoordinate.LI_ANION_COORDINATION.value,
    }
    total_energy = 0.0
    for coordination_name, coordinate_name in coordinate_map.items():
        displacement = float(coordinate_values[coordinate_name]) - float(
            targets[coordination_name]
        )
        total_energy += (
            0.5
            * float(stiffnesses[coordination_name])
            * displacement
            * displacement
        )
    return float(total_energy)


def _reduced_auxiliary_coordinate_energy_J_mol(
    free_energy_terms: dict,
    coordinate_values: dict[str, float],
) -> float:
    stiffness_values = np.asarray(
        [
            float(value)
            for value in free_energy_terms["solvation_competition"][
                "stiffness_J_mol"
            ].values()
        ],
        dtype=float,
    )
    if stiffness_values.size == 0:
        raise ValueError("solvation_competition stiffness_J_mol must not be empty")
    auxiliary_stiffness_J_mol = float(np.mean(stiffness_values))
    auxiliary_coordinate_names = (
        ReducedCoordinate.CLUSTER_COORDINATE.value,
        ReducedCoordinate.IDENTITY_COORDINATE.value,
        ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value,
        ReducedCoordinate.CAGE_COORDINATE.value,
        ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value,
        ReducedCoordinate.ATMOSPHERE_POLARIZATION.value,
    )
    total_energy = 0.0
    for coordinate_name in auxiliary_coordinate_names:
        coordinate_value = float(coordinate_values[coordinate_name])
        total_energy += (
            0.5
            * auxiliary_stiffness_J_mol
            * coordinate_value
            * coordinate_value
        )
    orientation = float(coordinate_values[ReducedCoordinate.ANION_ORIENTATION.value])
    total_energy += 0.5 * auxiliary_stiffness_J_mol * orientation * orientation
    return float(total_energy)


def _representative_pair_distance_m(
    records: PhysicalLibraryRecords,
    pair_label: str,
) -> float:
    pair_basins = records.basis_record["pair_basins"]
    contact_cutoff_m = float(pair_basins["r_CIP_m"])
    solvent_separated_cutoff_m = float(pair_basins["r_SSIP_m"])
    free_cutoff_m = float(pair_basins["r_free_m"])
    if pair_label == PairBasin.CONTACT_ION_PAIR.value:
        return contact_cutoff_m / 2.0
    if pair_label == PairBasin.SOLVENT_SEPARATED_ION_PAIR.value:
        return (contact_cutoff_m + solvent_separated_cutoff_m) / 2.0
    if pair_label == PairBasin.FREE.value:
        return free_cutoff_m
    if pair_label == "addSSIP":
        return (contact_cutoff_m + solvent_separated_cutoff_m) / 2.0
    if pair_label in ("aggregate", "bridge_network"):
        return contact_cutoff_m / 2.0
    raise ValueError(f"unsupported representative pair label {pair_label}")


def _configuration_for_transition_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    base_coordinate_values: dict[str, float],
    transition_record: dict,
    coordinate_value: float,
) -> SiteConfiguration:
    reaction_coordinate = _transition_reaction_coordinate(transition_record)
    coordinate_values = {**base_coordinate_values, reaction_coordinate.value: coordinate_value}
    builders = {
        ReducedCoordinate.LI_ANION_DISTANCE: _configuration_with_standard_coordinate,
        ReducedCoordinate.LI_SOLVENT_COORDINATION: _configuration_with_standard_coordinate,
        ReducedCoordinate.LI_LIGAND_COORDINATION: _configuration_with_standard_coordinate,
        ReducedCoordinate.LI_ANION_COORDINATION: _configuration_with_standard_coordinate,
        ReducedCoordinate.ANION_ORIENTATION: _configuration_with_standard_coordinate,
        ReducedCoordinate.LOCAL_PACKING_FRACTION: _configuration_with_standard_coordinate,
        ReducedCoordinate.LOCAL_IONIC_STRENGTH: _configuration_with_standard_coordinate,
        ReducedCoordinate.LOCAL_DIELECTRIC: _configuration_with_standard_coordinate,
        ReducedCoordinate.LOCAL_VISCOSITY: _configuration_with_standard_coordinate,
        ReducedCoordinate.ATMOSPHERE_POLARIZATION: _configuration_with_standard_coordinate,
        ReducedCoordinate.CAGE_COORDINATE: _configuration_with_cage_coordinate,
        ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE: (
            _configuration_with_partner_residence_coordinate
        ),
        ReducedCoordinate.IDENTITY_COORDINATE: _configuration_with_identity_coordinate,
        ReducedCoordinate.STRUCTURAL_HOP_COORDINATE: (
            _configuration_with_structural_hop_coordinate
        ),
        ReducedCoordinate.CLUSTER_COORDINATE: _configuration_with_cluster_coordinate,
    }
    if reaction_coordinate not in builders:
        raise ValueError(f"no transition builder for {reaction_coordinate.value}")
    builder = builders[reaction_coordinate]
    return builder(records, template_configuration, coordinate_values, transition_record)


def _configuration_with_standard_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    transition_record: dict,
) -> SiteConfiguration:
    _ = transition_record
    return _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )


def _configuration_with_identity_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    transition_record: dict,
) -> SiteConfiguration:
    configuration = _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )
    displacement_m = _endpoint_geometry_displacement_vector_m(
        records,
        configuration,
        transition_record,
        "identity_diffusion",
    ) * float(
        coordinate_values[ReducedCoordinate.IDENTITY_COORDINATE.value]
    )
    return _configuration_with_cation_displacement(records, configuration, displacement_m)


def _configuration_with_structural_hop_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    transition_record: dict,
) -> SiteConfiguration:
    configuration = _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )
    displacement_m = _endpoint_geometry_displacement_vector_m(
        records,
        configuration,
        transition_record,
        "structural_hop",
    ) * float(
        coordinate_values[ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value]
    )
    return _configuration_with_cation_displacement(records, configuration, displacement_m)


def _configuration_with_partner_residence_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    transition_record: dict,
) -> SiteConfiguration:
    configuration = _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )
    displacement_m = _endpoint_geometry_displacement_vector_m(
        records,
        configuration,
        transition_record,
        "partner_switch",
    ) * float(
        coordinate_values[ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value]
    )
    return _configuration_with_cation_displacement(records, configuration, displacement_m)


def _configuration_with_cage_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    transition_record: dict,
) -> SiteConfiguration:
    configuration = _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )
    cage_value = float(coordinate_values[ReducedCoordinate.CAGE_COORDINATE.value])
    displacement_m = _endpoint_geometry_displacement_vector_m(
        records,
        configuration,
        transition_record,
        "cage_capture_release",
    ) * cage_value
    return _configuration_with_cation_displacement(records, configuration, displacement_m)


def _configuration_with_cation_displacement(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    displacement_m: Array,
) -> SiteConfiguration:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    positions = np.asarray(configuration.positions_m, dtype=float).copy()
    displacement = np.asarray(displacement_m, dtype=float)
    if displacement.shape != (CARTESIAN_DIMENSION,):
        raise ValueError("cation displacement must have shape (3,)")
    positions[cation_index] = positions[cation_index] + displacement
    return SiteConfiguration(
        species_names=configuration.species_names,
        molecule_ids=np.asarray(configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=positions,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _configuration_with_cluster_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    transition_record: dict,
) -> SiteConfiguration:
    cluster_value = float(coordinate_values[ReducedCoordinate.CLUSTER_COORDINATE.value])
    pair_basins = records.basis_record["pair_basins"]
    contact_distance_m = float(pair_basins["r_CIP_m"]) / 2.0
    separated_distance_m = float(pair_basins["r_SSIP_m"])
    coordinate_values = {
        **coordinate_values,
        ReducedCoordinate.LI_ANION_DISTANCE.value: (
            separated_distance_m
            + cluster_value * (contact_distance_m - separated_distance_m)
        ),
    }
    configuration = _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )
    if "endpoint_geometry" not in transition_record:
        return configuration
    displacement_m = _endpoint_geometry_displacement_vector_m(
        records,
        configuration,
        transition_record,
        "bridge_network_formation_breakup",
    ) * cluster_value
    return _configuration_with_cation_displacement(records, configuration, displacement_m)


def _local_fields_for_coordinate_values(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
) -> PhysicalLocalFields:
    coordinate_packing_fraction = _nonnegative_float(
        float(coordinate_values[ReducedCoordinate.LOCAL_PACKING_FRACTION.value]),
        ReducedCoordinate.LOCAL_PACKING_FRACTION.value,
    )
    configuration_packing_fraction = compute_local_packing_fraction(records, configuration)
    local_packing_fraction = max(
        coordinate_packing_fraction,
        configuration_packing_fraction,
    )
    local_ionic_strength_mol_m3 = _local_ionic_strength_mol_m3(
        records,
        coordinate_values,
        local_packing_fraction,
    )
    local_dielectric_constant = _local_dielectric_constant(
        records,
        coordinate_values,
        local_ionic_strength_mol_m3,
    )
    local_viscosity_Pa_s = _local_viscosity_Pa_s(
        records,
        configuration,
        coordinate_values,
        local_ionic_strength_mol_m3,
        local_packing_fraction,
    )
    return PhysicalLocalFields(
        dielectric_constant=local_dielectric_constant,
        viscosity_Pa_s=local_viscosity_Pa_s,
        ionic_strength_mol_m3=local_ionic_strength_mol_m3,
        local_packing_fraction=local_packing_fraction,
    )


def _local_ionic_strength_mol_m3(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
    local_packing_fraction: float,
) -> float:
    base_ionic_strength_mol_m3 = _nonnegative_float(
        float(coordinate_values[ReducedCoordinate.LOCAL_IONIC_STRENGTH.value]),
        ReducedCoordinate.LOCAL_IONIC_STRENGTH.value,
    )
    coupling = float(
        records.mixture_record["local_fields"]["local_ionic_strength_packing_coupling"]
    )
    if coupling < 0.0:
        raise ValueError("local_ionic_strength_packing_coupling must be nonnegative")
    return _nonnegative_float(
        base_ionic_strength_mol_m3 * (1.0 + coupling * local_packing_fraction),
        "local_ionic_strength_mol_m3",
    )


def _local_dielectric_constant(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
    local_ionic_strength_mol_m3: float,
) -> float:
    base_dielectric = _positive_float(
        float(coordinate_values[ReducedCoordinate.LOCAL_DIELECTRIC.value]),
        ReducedCoordinate.LOCAL_DIELECTRIC.value,
    )
    decrement_per_mol_m3 = float(
        records.mixture_record["local_fields"]["dielectric_decrement_per_mol_m3"]
    )
    if decrement_per_mol_m3 < 0.0:
        raise ValueError("dielectric_decrement_per_mol_m3 must be nonnegative")
    return _positive_float(
        base_dielectric - decrement_per_mol_m3 * local_ionic_strength_mol_m3,
        "local_dielectric_constant",
    )


def _local_viscosity_Pa_s(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    local_ionic_strength_mol_m3: float,
    local_packing_fraction: float,
) -> float:
    base_viscosity_Pa_s = _positive_float(
        float(coordinate_values[ReducedCoordinate.LOCAL_VISCOSITY.value]),
        ReducedCoordinate.LOCAL_VISCOSITY.value,
    )
    local_field_record = records.mixture_record["local_fields"]
    jones_dole_a = float(local_field_record["jones_dole_A_sqrt_m3_mol"])
    jones_dole_b = float(local_field_record["jones_dole_B_m3_mol"])
    if jones_dole_a < 0.0 or jones_dole_b < 0.0:
        raise ValueError("Jones-Dole local viscosity coefficients must be nonnegative")
    salt_factor = (
        1.0
        + jones_dole_a * float(np.sqrt(local_ionic_strength_mol_m3))
        + jones_dole_b * local_ionic_strength_mol_m3
    )
    packing_record = records.mixture_record["packing"]
    phi_max = _positive_float(float(packing_record["phi_max"]), "mixture.packing.phi_max")
    if local_packing_fraction >= phi_max:
        raise ValueError("local_packing_fraction exceeds mixture.packing.phi_max")
    packing_exponent = _positive_float(
        float(local_field_record["packing_viscosity_exponent"]),
        "packing_viscosity_exponent",
    )
    packing_factor = (1.0 - local_packing_fraction / phi_max) ** (-packing_exponent)
    additive_factor = 1.0 + _configuration_additive_fraction(
        records,
        configuration,
        local_field_record["additive_microviscosity_coefficients"],
    )
    return _positive_float(
        base_viscosity_Pa_s * salt_factor * packing_factor * additive_factor,
        "local_viscosity_Pa_s",
    )


def _configuration_additive_fraction(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    additive_coefficients: dict,
) -> float:
    site_count = len(configuration.species_names)
    if site_count == 0:
        raise ValueError("configuration has no sites")
    additive_weight = 0.0
    for species_name in configuration.species_names:
        if _species_role(records, species_name) != SpeciesRole.ADDITIVE:
            continue
        if species_name not in additive_coefficients:
            raise KeyError(
                f"local_fields.additive_microviscosity_coefficients missing {species_name}"
            )
        additive_weight += float(additive_coefficients[species_name])
    return additive_weight / float(site_count)


def _reaction_coordinate_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    gradient_policy = str(transition_record["gradient_policy"])
    gradient_functions = {
        "pair_distance_gradient": _pair_distance_gradient_from_record,
        "coordination_switch_gradient": _coordination_switch_gradient_from_record,
        "cluster_coordinate_gradient": _cluster_coordinate_gradient_from_record,
        "partner_residence_gradient": _partner_residence_gradient_from_record,
        "identity_coordinate_gradient": _identity_coordinate_gradient_from_record,
        "structural_hop_gradient": _structural_hop_gradient_from_record,
        "cage_coordinate_gradient": _cage_coordinate_gradient_from_record,
        "orientation_gradient": _orientation_gradient_from_record,
        "atmosphere_coordinate_gradient": _atmosphere_coordinate_gradient_from_record,
    }
    if gradient_policy not in gradient_functions:
        raise ValueError(f"unsupported transition gradient_policy {gradient_policy}")
    return gradient_functions[gradient_policy](records, configuration, transition_record)


def _pair_distance_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    _ = transition_record
    return _pair_distance_gradient(records, configuration)


def _coordination_switch_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    reaction_coordinate = _transition_reaction_coordinate(transition_record)
    switch_by_coordinate = {
        ReducedCoordinate.LI_SOLVENT_COORDINATION: "Li_solvent",
        ReducedCoordinate.LI_LIGAND_COORDINATION: "Li_ligand",
        ReducedCoordinate.LI_ANION_COORDINATION: "Li_anion",
    }
    if reaction_coordinate not in switch_by_coordinate:
        raise ValueError(
            "coordination_switch_gradient requires a Li coordination coordinate"
        )
    return _coordination_switch_gradient(
        records,
        configuration,
        switch_by_coordinate[reaction_coordinate],
    )


def _coordination_switch_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    switch_name: str,
) -> Array:
    switch_record = records.basis_record["coordination_switches"][switch_name]
    center_role = SpeciesRole(str(switch_record["center_role"]))
    ligand_roles = tuple(SpeciesRole(str(role)) for role in switch_record["ligand_roles"])
    switch_radius_m = _positive_float(float(switch_record["r0_m"]), f"{switch_name}.r0_m")
    exponent = _positive_float(float(switch_record["exponent"]), f"{switch_name}.exponent")
    center_index = _first_role_index(records, configuration, center_role)
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    gradient = np.zeros(len(configuration.species_names) * CARTESIAN_DIMENSION, dtype=float)
    center_gradient = np.zeros(CARTESIAN_DIMENSION, dtype=float)
    for ligand_index, species_name in enumerate(configuration.species_names):
        if ligand_index == center_index:
            continue
        if _species_role(records, species_name) not in ligand_roles:
            continue
        center_to_ligand_m = _minimum_image_vector_m(
            positions_m[center_index],
            positions_m[ligand_index],
            configuration.box_lengths_m,
        )
        distance_m = float(np.linalg.norm(center_to_ligand_m))
        if distance_m <= 0.0:
            raise ValueError("coordination switch distance must be positive")
        reduced_distance = distance_m / switch_radius_m
        denominator = 1.0 + reduced_distance**exponent
        derivative_wrt_distance = (
            -exponent
            * reduced_distance ** (exponent - 1.0)
            / switch_radius_m
            / (denominator * denominator)
        )
        ligand_gradient = derivative_wrt_distance * center_to_ligand_m / distance_m
        ligand_start = ligand_index * CARTESIAN_DIMENSION
        ligand_stop = ligand_start + CARTESIAN_DIMENSION
        gradient[ligand_start:ligand_stop] += ligand_gradient
        center_gradient -= ligand_gradient
    center_start = center_index * CARTESIAN_DIMENSION
    center_stop = center_start + CARTESIAN_DIMENSION
    gradient[center_start:center_stop] += center_gradient
    return gradient


def _cluster_coordinate_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    _ = transition_record
    pair_basins = records.basis_record["pair_basins"]
    contact_distance_m = float(pair_basins["r_CIP_m"]) / 2.0
    separated_distance_m = float(pair_basins["r_SSIP_m"])
    span_m = separated_distance_m - contact_distance_m
    if span_m <= 0.0:
        raise ValueError("cluster coordinate span must be positive")
    return -_pair_distance_gradient(records, configuration) / span_m


def _partner_residence_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    _ = transition_record
    return _finite_difference_named_scalar_gradient(
        records,
        configuration,
        "partner_residence",
    )


def _identity_coordinate_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    displacement_m = _endpoint_geometry_displacement_m(
        transition_record,
        "identity_diffusion",
    )
    return _cation_x_gradient(records, configuration) / displacement_m


def _structural_hop_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    displacement_m = _endpoint_geometry_displacement_m(
        transition_record,
        "structural_hop",
    )
    return _cation_x_gradient(records, configuration) / displacement_m


def _cage_coordinate_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    if "endpoint_geometry" in transition_record:
        displacement_norm_m = _endpoint_geometry_displacement_m(
            transition_record,
            "cage_capture_release",
        )
        return _cation_displacement_axis_gradient(
            records,
            configuration,
            transition_record,
        ) / displacement_norm_m
    return _pair_distance_gradient(records, configuration) / _positive_float(
        float(records.basis_record["pair_basins"]["r_SSIP_m"]),
        "pair_basins.r_SSIP_m",
    )


def _orientation_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    _ = transition_record
    return _finite_difference_named_scalar_gradient(
        records,
        configuration,
        "anion_orientation",
    )


def _atmosphere_coordinate_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    _ = transition_record
    return _finite_difference_named_scalar_gradient(
        records,
        configuration,
        "atmosphere_polarization",
    )


def _cation_x_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    gradient = np.zeros(len(configuration.species_names) * CARTESIAN_DIMENSION, dtype=float)
    gradient[cation_index * CARTESIAN_DIMENSION] = 1.0
    return gradient


def _cation_displacement_axis_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    axis = _endpoint_geometry_direction_unit_vector_m(
        records,
        configuration,
        transition_record,
    )
    gradient = np.zeros(len(configuration.species_names) * CARTESIAN_DIMENSION, dtype=float)
    start = cation_index * CARTESIAN_DIMENSION
    stop = start + CARTESIAN_DIMENSION
    gradient[start:stop] = axis
    return gradient


def _finite_difference_named_scalar_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    scalar_name: str,
) -> Array:
    coordinate_count = len(configuration.species_names) * CARTESIAN_DIMENSION
    flat_positions = np.asarray(configuration.positions_m, dtype=float).reshape(
        coordinate_count
    )
    gradient = np.zeros(coordinate_count, dtype=float)
    for coordinate_index in range(coordinate_count):
        plus_positions = flat_positions.copy()
        minus_positions = flat_positions.copy()
        plus_positions[coordinate_index] += FINITE_DIFFERENCE_STEP_M
        minus_positions[coordinate_index] -= FINITE_DIFFERENCE_STEP_M
        plus_configuration = _configuration_with_flat_positions(
            configuration,
            plus_positions,
        )
        minus_configuration = _configuration_with_flat_positions(
            configuration,
            minus_positions,
        )
        gradient[coordinate_index] = (
            _named_scalar_value(records, plus_configuration, scalar_name)
            - _named_scalar_value(records, minus_configuration, scalar_name)
        ) / (CENTRAL_DIFFERENCE_WIDTH * FINITE_DIFFERENCE_STEP_M)
    return gradient


def _named_scalar_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    scalar_name: str,
) -> float:
    coordination_specs = {
        "coordination:Li_solvent": (SpeciesRole.SOLVENT.value, "Li_solvent"),
        "coordination:Li_ligand": (SpeciesRole.ADDITIVE.value, "Li_ligand"),
        "coordination:Li_anion": (SpeciesRole.ANION.value, "Li_anion"),
        "partner_residence": (SpeciesRole.ANION.value, "Li_anion"),
    }
    if scalar_name in coordination_specs:
        ligand_role, switch_name = coordination_specs[scalar_name]
        return compute_role_coordination_number(
            records,
            configuration,
            center_role=SpeciesRole.CATION.value,
            ligand_role=ligand_role,
            switch_name=switch_name,
        )
    if scalar_name == "anion_orientation":
        return _anion_orientation_cosine(records, configuration)
    if scalar_name == "atmosphere_polarization":
        return float(np.linalg.norm(compute_charge_polarization_m(records, configuration)))
    raise ValueError(f"unsupported scalar gradient value {scalar_name}")


def _configuration_with_flat_positions(
    template_configuration: SiteConfiguration,
    flat_positions: Array,
) -> SiteConfiguration:
    positions = np.asarray(flat_positions, dtype=float).reshape(
        (len(template_configuration.species_names), CARTESIAN_DIMENSION)
    )
    return SiteConfiguration(
        species_names=template_configuration.species_names,
        molecule_ids=np.asarray(template_configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(template_configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=positions,
        box_lengths_m=np.asarray(template_configuration.box_lengths_m, dtype=float),
    )


def _anion_orientation_cosine(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if _species_role(records, species_name) == SpeciesRole.ANION
    )
    if len(anion_indices) < 2:
        return 0.0
    anchor_index = anion_indices[0]
    orientation_index = anion_indices[1]
    orientation_vector = (
        np.asarray(configuration.positions_m[orientation_index], dtype=float)
        - np.asarray(configuration.positions_m[anchor_index], dtype=float)
    )
    pair_vector = (
        np.asarray(configuration.positions_m[anchor_index], dtype=float)
        - np.asarray(configuration.positions_m[cation_index], dtype=float)
    )
    orientation_norm = float(np.linalg.norm(orientation_vector))
    pair_norm = float(np.linalg.norm(pair_vector))
    if orientation_norm <= 0.0 or pair_norm <= 0.0:
        return 0.0
    return float(np.dot(orientation_vector, pair_vector) / (orientation_norm * pair_norm))


def _transition_displacement_moments(
    transition_record: dict,
    moment_input: MomentBoundaryValueInput,
) -> tuple[Array, Array]:
    moment_policy = str(transition_record["moment_policy"])
    moment_builders = {
        "conditioned_endpoint_bvp": _conditioned_endpoint_moments,
        "zero_motif_exchange": _zero_transition_moments,
        "endpoint_geometry": _endpoint_geometry_moments,
        "identity_diffusion": _endpoint_geometry_moments,
    }
    if moment_policy not in moment_builders:
        raise ValueError(f"unsupported transition moment_policy {moment_policy}")
    return moment_builders[moment_policy](moment_input)


def _validate_transition_displacement_policy(
    family: str,
    transition_record: dict,
    first_displacement_moment_m: Array,
    second_displacement_moment_m2: Array,
) -> None:
    displacement_policy = str(transition_record["displacement_policy"])
    first_moment = np.asarray(first_displacement_moment_m, dtype=float)
    second_moment = np.asarray(second_displacement_moment_m2, dtype=float)
    if first_moment.shape != (CARTESIAN_DIMENSION,):
        raise ValueError(f"transition family {family} first moment must have shape (3,)")
    if second_moment.shape != (CARTESIAN_DIMENSION, CARTESIAN_DIMENSION):
        raise ValueError(
            f"transition family {family} second moment must have shape (3,3)"
        )
    if not np.all(np.isfinite(first_moment)) or not np.all(np.isfinite(second_moment)):
        raise ValueError(f"transition family {family} moments must be finite")
    first_norm_m = float(np.linalg.norm(first_moment))
    second_trace_m2 = float(np.trace(second_moment))
    if second_trace_m2 < 0.0:
        raise ValueError(
            f"transition family {family} produced negative displacement second moment trace"
        )
    if displacement_policy == "zero":
        if first_norm_m != 0.0 or second_trace_m2 != 0.0:
            raise ValueError(
                f"zero-displacement transition family {family} produced nonzero moments"
            )
        return
    if first_norm_m == 0.0 and second_trace_m2 == 0.0:
        raise ValueError(
            f"conductivity-carrying transition family {family} produced zero d and M"
        )


def _conditioned_endpoint_moments(
    moment_input: MomentBoundaryValueInput,
) -> tuple[Array, Array]:
    result = solve_endpoint_moment_bvp(moment_input)
    return _directed_transition_moments(
        result.first_displacement_moment_m,
        result.second_displacement_moment_m2,
    )


def _zero_transition_moments(
    moment_input: MomentBoundaryValueInput,
) -> tuple[Array, Array]:
    _ = moment_input
    return np.zeros(CARTESIAN_DIMENSION, dtype=float), np.zeros(
        (CARTESIAN_DIMENSION, CARTESIAN_DIMENSION),
        dtype=float,
    )


def _endpoint_geometry_moments(
    moment_input: MomentBoundaryValueInput,
) -> tuple[Array, Array]:
    displacement = (
        np.asarray(moment_input.charge_polarization_by_grid[-1], dtype=float)
        - np.asarray(moment_input.charge_polarization_by_grid[0], dtype=float)
    )
    return _directed_transition_moments(displacement, np.outer(displacement, displacement))


def _directed_transition_moments(
    displacement_m: Array,
    second_moment_m2: Array,
) -> tuple[Array, Array]:
    displacement = np.asarray(displacement_m, dtype=float)
    second_moment = np.asarray(second_moment_m2, dtype=float)
    if displacement.shape != (CARTESIAN_DIMENSION,):
        raise ValueError("transition displacement must have shape (3,)")
    if second_moment.shape != (CARTESIAN_DIMENSION, CARTESIAN_DIMENSION):
        raise ValueError("transition second moment must have shape (3,3)")
    directed_second_moment = np.asarray(second_moment, dtype=float)
    displacement_outer = np.outer(displacement, displacement)
    if float(np.trace(directed_second_moment)) < float(displacement @ displacement):
        directed_second_moment = displacement_outer
    return displacement, 0.5 * (directed_second_moment + directed_second_moment.T)


def _transition_family_record(records: PhysicalLibraryRecords, family: str) -> dict:
    transition_records = records.transition_record["transition_records"]
    if family not in transition_records:
        raise KeyError(f"transition_records missing {family}")
    return transition_records[family]


def _validate_transition_family_reaction_coordinate(
    family: str,
    transition_record: dict,
) -> None:
    _ = family
    _transition_reaction_coordinate(transition_record)
    gradient_policy = str(transition_record["gradient_policy"])
    moment_policy = str(transition_record["moment_policy"])
    displacement_policy = str(transition_record["displacement_policy"])
    _transition_rate_bounds_s_inv(transition_record, family)
    _nonnegative_float(
        float(transition_record["barrier_J_mol"]),
        f"transition_records.{family}.barrier_J_mol",
    )
    allowed_gradient_policies = {
        "pair_distance_gradient",
        "coordination_switch_gradient",
        "cluster_coordinate_gradient",
        "partner_residence_gradient",
        "identity_coordinate_gradient",
        "structural_hop_gradient",
        "cage_coordinate_gradient",
        "orientation_gradient",
        "atmosphere_coordinate_gradient",
    }
    allowed_moment_policies = {
        "conditioned_endpoint_bvp",
        "zero_motif_exchange",
        "endpoint_geometry",
        "identity_diffusion",
    }
    allowed_displacement_policies = {
        "charge_polarization_endpoint_moment",
        "zero",
        "unwrapped_identity_displacement",
        "unwrapped_structural_hop",
    }
    if gradient_policy not in allowed_gradient_policies:
        raise ValueError(f"transition family {family} has unsupported gradient policy")
    if moment_policy not in allowed_moment_policies:
        raise ValueError(f"transition family {family} has unsupported moment policy")
    if displacement_policy not in allowed_displacement_policies:
        raise ValueError(f"transition family {family} has unsupported displacement policy")
    _validate_transition_policy_geometry_consistency(
        family,
        transition_record,
        moment_policy,
        displacement_policy,
    )
    if displacement_policy in (
        "unwrapped_identity_displacement",
        "unwrapped_structural_hop",
    ) or (
        displacement_policy == "charge_polarization_endpoint_moment"
        and "endpoint_geometry" in transition_record
    ):
        _validate_transition_endpoint_geometry(
            transition_record,
            family,
            displacement_policy,
        )
    if _uses_declared_rate_constant(family, transition_record):
        _transition_attempt_frequency_s_inv(transition_record, family)


def _validate_transition_policy_geometry_consistency(
    family: str,
    transition_record: dict,
    moment_policy: str,
    displacement_policy: str,
) -> None:
    if displacement_policy == "zero":
        if moment_policy != "zero_motif_exchange":
            raise ValueError(
                f"zero-displacement transition family {family} must use "
                "zero_motif_exchange moment_policy"
            )
        if "endpoint_geometry" in transition_record:
            raise ValueError(
                f"zero-displacement transition family {family} must not declare "
                "endpoint_geometry"
            )
        return
    if moment_policy == "zero_motif_exchange":
        raise ValueError(
            f"charge-carrying transition family {family} cannot use "
            "zero_motif_exchange moment_policy"
        )
    if displacement_policy == "unwrapped_identity_displacement":
        if moment_policy != "identity_diffusion":
            raise ValueError(
                f"identity-displacement transition family {family} must use "
                "identity_diffusion moment_policy"
            )
        return
    if displacement_policy == "unwrapped_structural_hop":
        if moment_policy != "conditioned_endpoint_bvp":
            raise ValueError(
                f"structural-hop transition family {family} must use "
                "conditioned_endpoint_bvp moment_policy"
            )
        return
    if (
        displacement_policy == "charge_polarization_endpoint_moment"
        and moment_policy != "conditioned_endpoint_bvp"
    ):
        raise ValueError(
            f"charge-polarization transition family {family} must use "
            "conditioned_endpoint_bvp moment_policy"
        )


def _validate_transition_endpoint_geometry(
    transition_record: dict,
    family: str,
    displacement_policy: str,
) -> None:
    if "endpoint_geometry" not in transition_record:
        raise KeyError(f"transition family {family} missing endpoint_geometry")
    endpoint_geometry = transition_record["endpoint_geometry"]
    if not isinstance(endpoint_geometry, dict):
        raise TypeError(f"transition family {family} endpoint_geometry must be a mapping")
    start_record = endpoint_geometry["start"]
    end_record = endpoint_geometry["end"]
    displacement_record = endpoint_geometry["displacement"]
    if not isinstance(start_record, dict):
        raise TypeError(f"transition family {family} endpoint_geometry.start must be a mapping")
    if not isinstance(end_record, dict):
        raise TypeError(f"transition family {family} endpoint_geometry.end must be a mapping")
    if not isinstance(displacement_record, dict):
        raise TypeError(
            f"transition family {family} endpoint_geometry.displacement must be a mapping"
        )
    _validate_endpoint_geometry_family_keys(family, start_record, end_record)
    displacement_type = str(displacement_record["type"])
    direction_policy = str(displacement_record["direction_policy"])
    allowed_direction_policies = {
        "pair_axis",
        "identity_axis",
        "hop_axis",
        "cluster_axis",
        "cage_axis",
    }
    if direction_policy not in allowed_direction_policies:
        raise ValueError(
            f"transition family {family} has unsupported endpoint direction policy"
        )
    if displacement_policy == "unwrapped_identity_displacement":
        if displacement_type != "charge_identity":
            raise ValueError(
                f"transition family {family} identity displacement requires "
                "endpoint displacement type charge_identity"
            )
        if direction_policy not in ("pair_axis", "identity_axis"):
            raise ValueError(
                f"transition family {family} identity displacement has incompatible "
                "direction_policy"
            )
    if displacement_policy == "unwrapped_structural_hop":
        if displacement_type != "unwrapped_structural_hop":
            raise ValueError(
                f"transition family {family} structural hop requires endpoint "
                "displacement type unwrapped_structural_hop"
            )
        if direction_policy != "hop_axis":
            raise ValueError(
                f"transition family {family} structural hop requires hop_axis direction"
            )
    if displacement_policy == "charge_polarization_endpoint_moment":
        allowed_charge_polarization_types = {
            "charge_polarization",
            "aggregate_bridge_rearrangement",
            "cage_backjump_transport",
        }
        if displacement_type not in allowed_charge_polarization_types:
            raise ValueError(
                f"transition family {family} charge-polarization endpoint requires "
                "a charge-polarization displacement type"
            )
    _positive_float(
        float(displacement_record["length_m"]),
        f"transition_records.{family}.endpoint_geometry.displacement.length_m",
    )


def _validate_endpoint_geometry_family_keys(
    family: str,
    start_record: dict,
    end_record: dict,
) -> None:
    required_key_map = {
        "partner_switch": (("Li_partner", "Li_position"), ("Li_partner", "Li_position")),
        "identity_diffusion": (("carrier_identity",), ("carrier_identity",)),
        "structural_hop": (("hop_site",), ("hop_site",)),
        "bridge_network_formation_breakup": (
            ("aggregate_state",),
            ("bridge_state",),
        ),
        "cage_capture_release": (("cage_state",), ("cage_state",)),
    }
    if family not in required_key_map:
        return
    required_start_keys, required_end_keys = required_key_map[family]
    for required_key in required_start_keys:
        if required_key not in start_record:
            raise KeyError(
                f"transition family {family} missing endpoint_geometry.start.{required_key}"
            )
    for required_key in required_end_keys:
        if required_key not in end_record:
            raise KeyError(
                f"transition family {family} missing endpoint_geometry.end.{required_key}"
            )


def _endpoint_geometry_displacement_m(transition_record: dict, family: str) -> float:
    if "endpoint_geometry" not in transition_record:
        raise KeyError(f"transition family {family} missing endpoint_geometry")
    endpoint_geometry = transition_record["endpoint_geometry"]
    if "displacement" not in endpoint_geometry:
        raise KeyError(f"transition family {family} missing endpoint_geometry.displacement")
    displacement_record = endpoint_geometry["displacement"]
    displacement_type = str(displacement_record["type"])
    allowed_types = {
        "aggregate_bridge_rearrangement",
        "charge_identity",
        "charge_polarization",
        "cage_backjump_transport",
        "unwrapped_structural_hop",
    }
    if displacement_type not in allowed_types:
        raise ValueError(
            f"transition family {family} has unsupported endpoint displacement type"
        )
    return _positive_float(
        float(displacement_record["length_m"]),
        f"transition_records.{family}.endpoint_geometry.displacement.length_m",
    )


def _endpoint_geometry_displacement_vector_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
    family: str,
) -> Array:
    length_m = _endpoint_geometry_displacement_m(transition_record, family)
    return (
        length_m
        * _endpoint_geometry_direction_unit_vector_m(
            records,
            configuration,
            transition_record,
        )
    )


def _endpoint_geometry_direction_unit_vector_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    endpoint_geometry = transition_record["endpoint_geometry"]
    displacement_record = endpoint_geometry["displacement"]
    direction_policy = str(displacement_record["direction_policy"])
    if direction_policy == "identity_axis":
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    if direction_policy in ("pair_axis", "cluster_axis", "cage_axis"):
        return _cation_to_anion_unit_vector(records, configuration)
    if direction_policy == "hop_axis":
        return _perpendicular_hop_unit_vector(records, configuration)
    raise ValueError(f"unsupported endpoint direction policy {direction_policy}")


def _cation_to_anion_unit_vector(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_index = _first_role_index(records, configuration, SpeciesRole.ANION)
    axis = _minimum_image_vector_m(
        configuration.positions_m[cation_index],
        configuration.positions_m[anion_index],
        configuration.box_lengths_m,
    )
    norm = _positive_float(float(np.linalg.norm(axis)), "endpoint pair-axis norm")
    return np.asarray(axis, dtype=float) / norm


def _perpendicular_hop_unit_vector(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    pair_axis = _cation_to_anion_unit_vector(records, configuration)
    helper_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(pair_axis, helper_axis))) > PERPENDICULAR_AXIS_ALIGNMENT_LIMIT:
        helper_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
    hop_axis = np.cross(pair_axis, helper_axis)
    norm = _positive_float(float(np.linalg.norm(hop_axis)), "endpoint hop-axis norm")
    return hop_axis / norm


def _transition_rate_bounds_s_inv(
    transition_record: dict,
    family: str,
) -> tuple[float, float]:
    if "rate_bounds_s_inv" not in transition_record:
        raise KeyError(f"transition family {family} missing rate_bounds_s_inv")
    bounds = transition_record["rate_bounds_s_inv"]
    lower_bound_s_inv = _nonnegative_float(
        float(bounds["lower"]),
        f"transition_records.{family}.rate_bounds_s_inv.lower",
    )
    upper_bound_s_inv = _positive_float(
        float(bounds["upper"]),
        f"transition_records.{family}.rate_bounds_s_inv.upper",
    )
    if lower_bound_s_inv > upper_bound_s_inv:
        raise ValueError(f"transition family {family} rate bounds are inconsistent")
    return lower_bound_s_inv, upper_bound_s_inv


def _transition_attempt_frequency_s_inv(
    transition_record: dict,
    family: str,
) -> float:
    if "attempt_frequency_s_inv" not in transition_record:
        raise KeyError(f"transition family {family} missing attempt_frequency_s_inv")
    return _positive_float(
        float(transition_record["attempt_frequency_s_inv"]),
        f"transition_records.{family}.attempt_frequency_s_inv",
    )


def _transition_residence_rate_constant_s_inv(
    family: str,
    transition_record: dict,
    temperature_K: float,
) -> float:
    attempt_frequency_s_inv = _transition_attempt_frequency_s_inv(
        transition_record,
        family,
    )
    barrier_J_mol = _nonnegative_float(
        float(transition_record["barrier_J_mol"]),
        f"transition_records.{family}.barrier_J_mol",
    )
    thermal_energy_J_mol = R * _positive_float(temperature_K, "temperature_K")
    return _positive_float(
        attempt_frequency_s_inv * float(np.exp(-barrier_J_mol / thermal_energy_J_mol)),
        f"transition_records.{family}.residence_rate_constant_s_inv",
    )


def _validate_transition_rate_bounds(
    records: PhysicalLibraryRecords,
    transition_edges: tuple[TransitionEdge, ...],
    transition_quadratures: tuple[PhysicalTransitionQuadrature, ...],
    reversible_generator_Q_ij_s_inv: Array,
    temperature_K: float,
) -> None:
    generator = np.asarray(reversible_generator_Q_ij_s_inv, dtype=float)
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("reversible_generator_Q_ij_s_inv must be square")
    if len(transition_edges) != len(transition_quadratures):
        raise ValueError("transition edge/quadrature count mismatch")
    for edge, transition_quadrature in zip(
        transition_edges,
        transition_quadratures,
        strict=True,
    ):
        transition_record = _transition_family_record(records, edge.family)
        uses_declared_rate_constant = _uses_declared_rate_constant(
            edge.family,
            transition_record,
        )
        lower_bound_s_inv, upper_bound_s_inv = _derived_transition_rate_bounds_s_inv(
            records,
            edge.family,
            transition_record,
            transition_quadrature,
            temperature_K,
        )
        forward_rate_s_inv = float(generator[edge.from_state_index, edge.to_state_index])
        reverse_rate_s_inv = float(generator[edge.to_state_index, edge.from_state_index])
        for rate_label, rate_s_inv in (
            ("forward", forward_rate_s_inv),
            ("reverse", reverse_rate_s_inv),
        ):
            if not np.isfinite(rate_s_inv) or rate_s_inv < 0.0:
                raise ValueError(
                    f"transition family {edge.family} {rate_label} rate is invalid "
                    f"for edge {edge.from_state_index}->{edge.to_state_index}"
                )
            _ = lower_bound_s_inv
            if rate_s_inv > upper_bound_s_inv:
                bound_source = (
                    "configured residence upper bound"
                    if uses_declared_rate_constant
                    else "derived upper bound"
                )
                raise ValueError(
                    f"transition family {edge.family} {rate_label} rate {rate_s_inv} "
                    f"s^-1 above {bound_source} {upper_bound_s_inv} for "
                    f"edge {edge.from_state_index}->{edge.to_state_index}"
                )


def _derived_transition_rate_bounds_s_inv(
    records: PhysicalLibraryRecords,
    family: str,
    transition_record: dict,
    transition_quadrature: PhysicalTransitionQuadrature,
    temperature_K: float,
) -> tuple[float, float]:
    if _uses_declared_rate_constant(family, transition_record):
        return _transition_rate_bounds_s_inv(transition_record, family)
    projected_diffusivities = _transition_projected_diffusivity_profile(
        records,
        transition_record,
        transition_quadrature,
        temperature_K,
    )
    positive_projected_diffusivities = projected_diffusivities[
        projected_diffusivities > 0.0
    ]
    if positive_projected_diffusivities.size == 0:
        raise ValueError(f"transition family {family} has no positive D_xi values")
    coordinate_span = _transition_coordinate_span(
        records,
        transition_record,
        transition_quadrature,
    )
    barrier_J_mol = _nonnegative_float(
        float(transition_record["barrier_J_mol"]),
        f"transition_records.{family}.barrier_J_mol",
    )
    thermal_energy_J_mol = R * _positive_float(temperature_K, "temperature_K")
    upper_bound_s_inv = float(np.max(positive_projected_diffusivities)) / (
        coordinate_span * coordinate_span
    )
    lower_bound_s_inv = (
        float(np.min(positive_projected_diffusivities))
        / (coordinate_span * coordinate_span)
        * float(np.exp(-barrier_J_mol / thermal_energy_J_mol))
    )
    if lower_bound_s_inv > upper_bound_s_inv:
        raise ValueError(
            f"derived transition family {family} rate bounds are inconsistent"
        )
    return lower_bound_s_inv, upper_bound_s_inv


def _uses_declared_rate_constant(family: str, transition_record: dict) -> bool:
    _ = family
    return "attempt_frequency_s_inv" in transition_record


def _transition_projected_diffusivity_profile(
    records: PhysicalLibraryRecords,
    transition_record: dict,
    transition_quadrature: PhysicalTransitionQuadrature,
    temperature_K: float,
) -> Array:
    return np.asarray(
        [
            project_diffusivity_onto_reaction_coordinate(
                build_physical_objects(
                    records,
                    configuration,
                    temperature_K,
                    local_fields.dielectric_constant,
                    local_fields.viscosity_Pa_s,
                    local_fields.ionic_strength_mol_m3,
                    local_fields.local_packing_fraction,
                ).mobility_tensor_m2_s,
                _reaction_coordinate_gradient(records, configuration, transition_record),
            )
            for configuration, local_fields in zip(
                transition_quadrature.configurations,
                transition_quadrature.local_fields,
                strict=True,
            )
        ],
        dtype=float,
    )


def _transition_coordinate_span(
    records: PhysicalLibraryRecords,
    transition_record: dict,
    transition_quadrature: PhysicalTransitionQuadrature,
) -> float:
    reaction_coordinate = _transition_reaction_coordinate(transition_record)
    if reaction_coordinate == ReducedCoordinate.LI_ANION_DISTANCE:
        pair_distances_m = np.asarray(
            [
                _li_anion_distance_m(records, configuration)
                for configuration in transition_quadrature.configurations
            ],
            dtype=float,
        )
        return _positive_float(
            float(np.max(pair_distances_m) - np.min(pair_distances_m)),
            "Li_anion_distance transition span",
        )
    if reaction_coordinate.value in records.basis_record["coordinate_domains"]:
        coordinate_domain = records.basis_record["coordinate_domains"][
            reaction_coordinate.value
        ]
        lower_coordinate_value, upper_coordinate_value = _domain_bounds(
            coordinate_domain,
            f"basis.coordinate_domains.{reaction_coordinate.value}",
        )
        return _positive_float(
            upper_coordinate_value - lower_coordinate_value,
            f"{reaction_coordinate.value} transition span",
        )
    if "coordinate_domain" not in transition_record:
        raise KeyError(
            f"transition record for {reaction_coordinate.value} missing coordinate_domain"
        )
    lower_coordinate_value, upper_coordinate_value = _domain_bounds(
        transition_record["coordinate_domain"],
        f"transition_records.{reaction_coordinate.value}.coordinate_domain",
    )
    return _positive_float(
        upper_coordinate_value - lower_coordinate_value,
        f"{reaction_coordinate.value} transition span",
    )


def _annotate_transition_generator_diagnostics(
    conductivity_result: ProjectedConductivityResult,
    records: PhysicalLibraryRecords,
    transition_edges: tuple[TransitionEdge, ...],
    transition_quadratures: tuple[PhysicalTransitionQuadrature, ...],
    temperature_K: float,
) -> None:
    if len(transition_edges) != len(transition_quadratures):
        raise ValueError("transition edge/quadrature count mismatch")
    generator = np.asarray(conductivity_result.reversible_generator_Q_ij_s_inv, dtype=float)
    capacity_fluxes = np.asarray(
        conductivity_result.symmetric_capacity_fluxes_K_ij_mol_m3_s,
        dtype=float,
    )
    first_moments = np.asarray(
        conductivity_result.transition_first_moments_d_ij_m,
        dtype=float,
    )
    second_moments = np.asarray(
        conductivity_result.transition_second_moments_M_ij_m2,
        dtype=float,
    )
    edge_from_indices = np.asarray(
        [edge.from_state_index for edge in transition_edges],
        dtype=int,
    )
    edge_to_indices = np.asarray(
        [edge.to_state_index for edge in transition_edges],
        dtype=int,
    )
    state_exit_rates_s_inv = -np.diag(generator)
    state_lifetimes_s = np.full(state_exit_rates_s_inv.shape, np.inf, dtype=float)
    active_state_mask = state_exit_rates_s_inv > 0.0
    state_lifetimes_s[active_state_mask] = 1.0 / state_exit_rates_s_inv[
        active_state_mask
    ]
    edge_capacity_fluxes = np.asarray(
        [
            capacity_fluxes[edge.from_state_index, edge.to_state_index]
            for edge in transition_edges
        ],
        dtype=float,
    )
    edge_forward_rates_s_inv = np.asarray(
        [
            generator[edge.from_state_index, edge.to_state_index]
            for edge in transition_edges
        ],
        dtype=float,
    )
    edge_reverse_rates_s_inv = np.asarray(
        [
            generator[edge.to_state_index, edge.from_state_index]
            for edge in transition_edges
        ],
        dtype=float,
    )
    edge_first_moment_norms_m = np.asarray(
        [
            np.linalg.norm(first_moments[edge.from_state_index, edge.to_state_index])
            for edge in transition_edges
        ],
        dtype=float,
    )
    edge_second_moment_traces_m2 = np.asarray(
        [
            np.trace(second_moments[edge.from_state_index, edge.to_state_index])
            for edge in transition_edges
        ],
        dtype=float,
    )
    transition_records = tuple(
        _transition_family_record(records, edge.family) for edge in transition_edges
    )
    edge_coordinate_spans = np.asarray(
        [
            _transition_coordinate_span(records, transition_record, transition_quadrature)
            for transition_record, transition_quadrature in zip(
                transition_records,
                transition_quadratures,
                strict=True,
            )
        ],
        dtype=float,
    )
    projected_diffusivity_profiles = tuple(
        _transition_projected_diffusivity_profile(
            records,
            transition_record,
            transition_quadrature,
            temperature_K,
        )
        for transition_record, transition_quadrature in zip(
            transition_records,
            transition_quadratures,
            strict=True,
        )
    )
    edge_projected_diffusivity_min = np.asarray(
        [
            float(np.min(profile))
            for profile in projected_diffusivity_profiles
        ],
        dtype=float,
    )
    edge_projected_diffusivity_max = np.asarray(
        [
            float(np.max(profile))
            for profile in projected_diffusivity_profiles
        ],
        dtype=float,
    )
    thermal_energy_J_mol = R * _positive_float(temperature_K, "temperature_K")
    edge_barrier_over_RT = np.asarray(
        [
            _nonnegative_float(
                float(transition_record["barrier_J_mol"]),
                f"transition_records.{edge.family}.barrier_J_mol",
            )
            / thermal_energy_J_mol
            for edge, transition_record in zip(
                transition_edges,
                transition_records,
                strict=True,
            )
        ],
        dtype=float,
    )
    edge_log_capacity_integrals = np.asarray(
        [
            transition_quadrature.log_capacity_integral
            for transition_quadrature in transition_quadratures
        ],
        dtype=float,
    )
    edge_direct_trace_contribution = edge_capacity_fluxes * edge_second_moment_traces_m2
    edge_inactive_reasons = _transition_edge_inactive_reasons(
        transition_records,
        edge_capacity_fluxes,
        edge_forward_rates_s_inv,
        edge_reverse_rates_s_inv,
        edge_first_moment_norms_m,
        edge_second_moment_traces_m2,
        edge_direct_trace_contribution,
    )
    finite_state_drift_vectors_m_s = np.einsum("ij,ija->ia", generator, first_moments)
    finite_state_poisson_b_vector_norms_m_s = np.linalg.norm(
        finite_state_drift_vectors_m_s,
        axis=1,
    )
    edge_endpoint_poisson_b_norms_m_s = np.asarray(
        [
            max(
                finite_state_poisson_b_vector_norms_m_s[edge.from_state_index],
                finite_state_poisson_b_vector_norms_m_s[edge.to_state_index],
            )
            for edge in transition_edges
        ],
        dtype=float,
    )
    (
        component_state_indices,
        component_weighted_drift_vectors_mol_m2_s,
        component_weighted_drift_norms_mol_m2_s,
        component_weighted_absolute_drift_scales_mol_m2_s,
        component_max_state_drift_norms_m_s,
    ) = _finite_state_component_drift_diagnostics(
        generator,
        np.asarray(conductivity_result.state_concentrations_mol_m3, dtype=float),
        finite_state_drift_vectors_m_s,
    )
    edge_component_indices = _transition_edge_component_indices(
        transition_edges,
        component_state_indices,
        generator.shape[0],
    )
    edge_weighted_drift_contributions_mol_m2_s = (
        _transition_edge_weighted_drift_contributions(
            transition_edges,
            np.asarray(conductivity_result.state_concentrations_mol_m3, dtype=float),
            generator,
            first_moments,
        )
    )
    _validate_charge_carrying_transition_activity(
        transition_edges,
        transition_records,
        edge_capacity_fluxes,
        edge_first_moment_norms_m,
        edge_second_moment_traces_m2,
        edge_direct_trace_contribution,
    )
    conductivity_result.effect_attribution.update(
        {
            "transition_edge_families": tuple(edge.family for edge in transition_edges),
            "transition_edge_from_state_indices": edge_from_indices,
            "transition_edge_to_state_indices": edge_to_indices,
            "transition_edge_capacity_fluxes_K_ij_mol_m3_s": edge_capacity_fluxes,
            "transition_edge_forward_rates_Q_ij_s_inv": edge_forward_rates_s_inv,
            "transition_edge_reverse_rates_Q_ji_s_inv": edge_reverse_rates_s_inv,
            "transition_edge_first_moment_norms_m": edge_first_moment_norms_m,
            "transition_edge_second_moment_traces_m2": edge_second_moment_traces_m2,
            "transition_edge_K_trace_M_mol_m5_s": edge_direct_trace_contribution,
            "transition_edge_inactive_reasons": edge_inactive_reasons,
            "transition_edge_moment_policies": tuple(
                str(transition_record["moment_policy"])
                for transition_record in transition_records
            ),
            "transition_edge_displacement_policies": tuple(
                str(transition_record["displacement_policy"])
                for transition_record in transition_records
            ),
            "transition_edge_endpoint_displacement_types": tuple(
                _endpoint_geometry_diagnostic_field(transition_record, "type")
                for transition_record in transition_records
            ),
            "transition_edge_endpoint_direction_policies": tuple(
                _endpoint_geometry_diagnostic_field(
                    transition_record,
                    "direction_policy",
                )
                for transition_record in transition_records
            ),
            "transition_edge_endpoint_displacement_lengths_m": np.asarray(
                [
                    _endpoint_geometry_diagnostic_length_m(transition_record)
                    for transition_record in transition_records
                ],
                dtype=float,
            ),
            "transition_edge_reaction_coordinates": tuple(
                str(transition_record["reaction_coordinate"])
                for transition_record in transition_records
            ),
            "transition_edge_coordinate_spans": edge_coordinate_spans,
            "transition_edge_projected_diffusivity_min": edge_projected_diffusivity_min,
            "transition_edge_projected_diffusivity_max": edge_projected_diffusivity_max,
            "transition_edge_barrier_over_RT": edge_barrier_over_RT,
            "transition_edge_log_capacity_integrals": edge_log_capacity_integrals,
            "transition_edge_direct_trace_contribution": edge_direct_trace_contribution,
            "transition_edge_component_indices": edge_component_indices,
            "transition_edge_endpoint_poisson_b_norms_m_s": (
                edge_endpoint_poisson_b_norms_m_s
            ),
            "transition_edge_weighted_drift_contributions_mol_m2_s": (
                edge_weighted_drift_contributions_mol_m2_s
            ),
            "finite_state_drift_vectors_m_s": finite_state_drift_vectors_m_s,
            "finite_state_poisson_b_vectors_m_s": finite_state_drift_vectors_m_s,
            "finite_state_poisson_b_vector_norms_m_s": (
                finite_state_poisson_b_vector_norms_m_s
            ),
            "state_weighted_finite_state_drift_vectors_mol_m2_s": (
                np.asarray(conductivity_result.state_concentrations_mol_m3, dtype=float)[
                    :, np.newaxis
                ]
                * finite_state_drift_vectors_m_s
            ),
            "finite_state_component_state_indices": component_state_indices,
            "finite_state_component_weighted_drift_vectors_mol_m2_s": (
                component_weighted_drift_vectors_mol_m2_s
            ),
            "finite_state_poisson_component_c_dot_b_vectors_mol_m2_s": (
                component_weighted_drift_vectors_mol_m2_s
            ),
            "finite_state_component_weighted_drift_norms_mol_m2_s": (
                component_weighted_drift_norms_mol_m2_s
            ),
            "finite_state_poisson_component_c_dot_b_norms_mol_m2_s": (
                component_weighted_drift_norms_mol_m2_s
            ),
            "finite_state_component_weighted_absolute_drift_scales_mol_m2_s": (
                component_weighted_absolute_drift_scales_mol_m2_s
            ),
            "finite_state_poisson_component_absolute_c_b_scales_mol_m2_s": (
                component_weighted_absolute_drift_scales_mol_m2_s
            ),
            "finite_state_component_max_state_drift_norms_m_s": (
                component_max_state_drift_norms_m_s
            ),
            "state_exit_rates_s_inv": state_exit_rates_s_inv,
            "state_lifetimes_s": state_lifetimes_s,
            "active_state_lifetimes_s": state_lifetimes_s[active_state_mask],
        }
    )


def _endpoint_geometry_diagnostic_field(
    transition_record: dict,
    field_name: str,
) -> str:
    if "endpoint_geometry" not in transition_record:
        return ""
    endpoint_geometry = transition_record["endpoint_geometry"]
    if "displacement" not in endpoint_geometry:
        return ""
    displacement_record = endpoint_geometry["displacement"]
    if field_name not in displacement_record:
        return ""
    return str(displacement_record[field_name])


def _endpoint_geometry_diagnostic_length_m(transition_record: dict) -> float:
    if "endpoint_geometry" not in transition_record:
        return 0.0
    endpoint_geometry = transition_record["endpoint_geometry"]
    if "displacement" not in endpoint_geometry:
        return 0.0
    displacement_record = endpoint_geometry["displacement"]
    if "length_m" not in displacement_record:
        return 0.0
    return _positive_float(
        float(displacement_record["length_m"]),
        "transition endpoint displacement length_m",
    )


def _finite_state_component_drift_diagnostics(
    reversible_generator_Q_ij_s_inv: Array,
    state_concentrations_mol_m3: Array,
    finite_state_drift_vectors_m_s: Array,
) -> tuple[tuple[tuple[int, ...], ...], Array, Array, Array, Array]:
    generator = np.asarray(reversible_generator_Q_ij_s_inv, dtype=float)
    concentrations = np.asarray(state_concentrations_mol_m3, dtype=float)
    drift_vectors = np.asarray(finite_state_drift_vectors_m_s, dtype=float)
    component_arrays = _generator_connected_components_for_diagnostics(generator)
    component_state_indices = tuple(
        tuple(int(state_index) for state_index in component_indices)
        for component_indices in component_arrays
    )
    weighted_drift_vectors = []
    weighted_drift_norms = []
    weighted_absolute_drift_scales = []
    max_state_drift_norms = []
    for component_indices in component_arrays:
        component_concentrations = concentrations[component_indices]
        component_drift_vectors = drift_vectors[component_indices]
        weighted_drift_vector = np.einsum(
            "i,ia->a",
            component_concentrations,
            component_drift_vectors,
        )
        weighted_drift_vectors.append(weighted_drift_vector)
        weighted_drift_norms.append(float(np.linalg.norm(weighted_drift_vector)))
        weighted_absolute_drift_scales.append(
            float(
                np.sum(
                    np.abs(
                        component_concentrations[:, np.newaxis]
                        * component_drift_vectors
                    )
                )
            )
        )
        max_state_drift_norms.append(
            float(np.max(np.linalg.norm(component_drift_vectors, axis=1)))
        )
    return (
        component_state_indices,
        np.asarray(weighted_drift_vectors, dtype=float),
        np.asarray(weighted_drift_norms, dtype=float),
        np.asarray(weighted_absolute_drift_scales, dtype=float),
        np.asarray(max_state_drift_norms, dtype=float),
    )


def _transition_edge_component_indices(
    transition_edges: tuple[TransitionEdge, ...],
    component_state_indices: tuple[tuple[int, ...], ...],
    state_count: int,
) -> Array:
    state_component_indices = np.full(state_count, -1, dtype=int)
    for component_index, component_indices in enumerate(component_state_indices):
        for state_index in component_indices:
            state_component_indices[state_index] = component_index
    edge_component_indices = np.asarray(
        [
            state_component_indices[edge.from_state_index]
            for edge in transition_edges
        ],
        dtype=int,
    )
    for edge, component_index in zip(
        transition_edges,
        edge_component_indices,
        strict=True,
    ):
        if component_index < 0:
            raise ValueError("transition edge references a state without component")
        if state_component_indices[edge.to_state_index] != component_index:
            raise ValueError("transition edge crosses generator connected components")
    return edge_component_indices


def _transition_edge_weighted_drift_contributions(
    transition_edges: tuple[TransitionEdge, ...],
    state_concentrations_mol_m3: Array,
    reversible_generator_Q_ij_s_inv: Array,
    transition_first_moments_d_ij_m: Array,
) -> Array:
    concentrations = np.asarray(state_concentrations_mol_m3, dtype=float)
    generator = np.asarray(reversible_generator_Q_ij_s_inv, dtype=float)
    first_moments = np.asarray(transition_first_moments_d_ij_m, dtype=float)
    return np.asarray(
        [
            concentrations[edge.from_state_index]
            * generator[edge.from_state_index, edge.to_state_index]
            * first_moments[edge.from_state_index, edge.to_state_index]
            + concentrations[edge.to_state_index]
            * generator[edge.to_state_index, edge.from_state_index]
            * first_moments[edge.to_state_index, edge.from_state_index]
            for edge in transition_edges
        ],
        dtype=float,
    )


def _generator_connected_components_for_diagnostics(generator_Q_ij_s_inv: Array) -> tuple[Array, ...]:
    generator = np.asarray(generator_Q_ij_s_inv, dtype=float)
    state_count = generator.shape[0]
    adjacency = (np.abs(generator) > 0.0) | (np.abs(generator.T) > 0.0)
    visited = np.zeros(state_count, dtype=bool)
    components = []
    for seed_state_index in range(state_count):
        if visited[seed_state_index]:
            continue
        stack = [seed_state_index]
        visited[seed_state_index] = True
        component_indices = []
        while stack:
            state_index = stack.pop()
            component_indices.append(state_index)
            neighbor_indices = np.nonzero(adjacency[state_index])[0]
            for neighbor_index in neighbor_indices:
                if not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    stack.append(int(neighbor_index))
        components.append(np.asarray(sorted(component_indices), dtype=int))
    return tuple(components)


def _validate_charge_carrying_transition_activity(
    transition_edges: tuple[TransitionEdge, ...],
    transition_records: tuple[dict, ...],
    edge_capacity_fluxes: Array,
    edge_first_moment_norms_m: Array,
    edge_second_moment_traces_m2: Array,
    edge_direct_trace_contribution: Array,
) -> None:
    charge_carrying_indices = tuple(
        edge_index
        for edge_index, transition_record in enumerate(transition_records)
        if str(transition_record["displacement_policy"]) != "zero"
    )
    for edge_index in charge_carrying_indices:
        inactive_reason = _transition_edge_activity_reason(
            transition_records[edge_index],
            float(edge_capacity_fluxes[edge_index]),
            float(edge_first_moment_norms_m[edge_index]),
            float(edge_second_moment_traces_m2[edge_index]),
            float(edge_direct_trace_contribution[edge_index]),
        )
        if inactive_reason:
            edge = transition_edges[edge_index]
            raise ValueError(
                f"charge-carrying transition family {edge.family} produced inactive "
                f"primitive contribution on edge "
                f"{edge.from_state_index}->{edge.to_state_index}: {inactive_reason}"
            )


def _transition_edge_inactive_reasons(
    transition_records: tuple[dict, ...],
    edge_capacity_fluxes: Array,
    edge_forward_rates_s_inv: Array,
    edge_reverse_rates_s_inv: Array,
    edge_first_moment_norms_m: Array,
    edge_second_moment_traces_m2: Array,
    edge_direct_trace_contribution: Array,
) -> tuple[str, ...]:
    return tuple(
        _transition_edge_activity_reason(
            transition_record,
            float(edge_capacity_fluxes[edge_index]),
            float(edge_first_moment_norms_m[edge_index]),
            float(edge_second_moment_traces_m2[edge_index]),
            float(edge_direct_trace_contribution[edge_index]),
        )
        or _transition_edge_rate_activity_reason(
            float(edge_forward_rates_s_inv[edge_index]),
            float(edge_reverse_rates_s_inv[edge_index]),
        )
        for edge_index, transition_record in enumerate(transition_records)
    )


def _transition_edge_activity_reason(
    transition_record: dict,
    capacity_flux_K_ij_mol_m3_s: float,
    first_moment_norm_m: float,
    second_moment_trace_m2: float,
    direct_trace_contribution_mol_m5_s: float,
) -> str:
    if str(transition_record["displacement_policy"]) == "zero":
        if first_moment_norm_m == 0.0 and second_moment_trace_m2 == 0.0:
            return "zero_motif_exchange"
    if capacity_flux_K_ij_mol_m3_s <= 0.0:
        return "K_ij_mol_m3_s_nonpositive"
    if first_moment_norm_m <= 0.0:
        return "first_moment_norm_m_nonpositive"
    if second_moment_trace_m2 <= 0.0:
        return "second_moment_trace_m2_nonpositive"
    if direct_trace_contribution_mol_m5_s <= 0.0:
        return "K_trace_M_mol_m5_s_nonpositive"
    return ""


def _transition_edge_rate_activity_reason(
    forward_rate_Q_ij_s_inv: float,
    reverse_rate_Q_ji_s_inv: float,
) -> str:
    if forward_rate_Q_ij_s_inv <= 0.0 and reverse_rate_Q_ji_s_inv <= 0.0:
        return "Q_ij_and_Q_ji_s_inv_nonpositive"
    if forward_rate_Q_ij_s_inv <= 0.0:
        return "Q_ij_s_inv_nonpositive"
    if reverse_rate_Q_ji_s_inv <= 0.0:
        return "Q_ji_s_inv_nonpositive"
    return ""


def _annotate_state_charge_mobility_diagnostics(
    conductivity_result: ProjectedConductivityResult,
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    temperature_K: float,
) -> None:
    diagnostics = tuple(
        _state_charge_mobility_diagnostics(
            records,
            state_quadrature,
            temperature_K,
        )
        for state_quadrature in state_quadratures
    )
    conductivity_result.effect_attribution.update(
        {
            "state_labels": tuple(state.label for state in state_quadratures),
            "state_charge_mobility_zDz_m2_s": np.asarray(
                [diagnostic.charge_mobility_m2_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_charged_center_D_Q_zDz_m2_s": np.asarray(
                [diagnostic.charge_mobility_m2_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_cation_mobility_zDz_m2_s": np.asarray(
                [diagnostic.cation_mobility_m2_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_charged_center_D_Li_m2_s": np.asarray(
                [diagnostic.cation_mobility_m2_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_anion_mobility_zDz_m2_s": np.asarray(
                [diagnostic.anion_mobility_m2_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_charged_center_D_anion_m2_s": np.asarray(
                [diagnostic.anion_mobility_m2_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_cation_anion_cross_mobility_zDz_m2_s": np.asarray(
                [
                    diagnostic.cation_anion_cross_mobility_m2_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_charged_center_D_Li_anion_m2_s": np.asarray(
                [
                    diagnostic.cation_anion_cross_mobility_m2_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_cation_anion_center_mobility_m2_s": np.asarray(
                [
                    diagnostic.cation_anion_center_mobility_m2_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_charged_center_labels": tuple(
                diagnostic.charged_center_labels for diagnostic in diagnostics
            ),
            "state_charged_center_charge_numbers": tuple(
                diagnostic.charged_center_charge_numbers for diagnostic in diagnostics
            ),
            "state_charged_center_mobility_matrices_m2_s": tuple(
                diagnostic.charged_center_mobility_matrix_m2_s
                for diagnostic in diagnostics
            ),
            "state_potential_energies_J_mol": np.asarray(
                [diagnostic.potential_energy_J_mol for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_local_dielectric_constants": np.asarray(
                [diagnostic.dielectric_constant for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_local_viscosities_Pa_s": np.asarray(
                [diagnostic.viscosity_Pa_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_local_ionic_strengths_mol_m3": np.asarray(
                [diagnostic.ionic_strength_mol_m3 for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_local_packing_fractions": np.asarray(
                [diagnostic.local_packing_fraction for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_atmosphere_electrophoretic_traces_kg_s": np.asarray(
                [
                    diagnostic.atmosphere_electrophoretic_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_atmosphere_relaxation_traces_kg_s": np.asarray(
                [
                    diagnostic.atmosphere_relaxation_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_atmosphere_cation_diagonal_traces_kg_s": np.asarray(
                [
                    diagnostic.atmosphere_cation_diagonal_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_atmosphere_anion_diagonal_traces_kg_s": np.asarray(
                [
                    diagnostic.atmosphere_anion_diagonal_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_atmosphere_cation_anion_cross_traces_kg_s": np.asarray(
                [
                    diagnostic.atmosphere_cation_anion_cross_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_atmosphere_mean_charge_cloud_form_factors": np.asarray(
                [
                    diagnostic.atmosphere_mean_charge_cloud_form_factor
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_atmosphere_mean_state_geometry_form_factors": np.asarray(
                [
                    diagnostic.atmosphere_mean_state_geometry_form_factor
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_atmosphere_minimum_separation_over_debye_length": np.asarray(
                [
                    diagnostic.atmosphere_minimum_separation_over_debye_length
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_atmosphere_debye_falkenhagen_times_s": np.asarray(
                [
                    diagnostic.atmosphere_debye_falkenhagen_time_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
        }
    )


def _state_charge_mobility_diagnostics(
    records: PhysicalLibraryRecords,
    state_quadrature: PhysicalStateQuadrature,
    temperature_K: float,
) -> StateChargeMobilityDiagnostics:
    active_anion_name = _active_anion_name_from_state_label(state_quadrature.label)
    point_diagnostics = tuple(
        _state_charge_mobility_point_terms(
            records,
            configuration,
            local_fields,
            active_anion_name,
            temperature_K,
        )
        for configuration, local_fields in zip(
            state_quadrature.configurations,
            state_quadrature.local_fields,
            strict=True,
        )
    )
    normalized_weights = _normalized_boltzmann_quadrature_weights(
        state_quadrature.weights,
        tuple(diagnostic.potential_energy_J_mol for diagnostic in point_diagnostics),
        temperature_K,
    )
    return StateChargeMobilityDiagnostics(
        charge_mobility_m2_s=float(
            np.dot(
                normalized_weights,
                [diagnostic.charge_mobility_m2_s for diagnostic in point_diagnostics],
            )
        ),
        cation_mobility_m2_s=float(
            np.dot(
                normalized_weights,
                [diagnostic.cation_mobility_m2_s for diagnostic in point_diagnostics],
            )
        ),
        anion_mobility_m2_s=float(
            np.dot(
                normalized_weights,
                [diagnostic.anion_mobility_m2_s for diagnostic in point_diagnostics],
            )
        ),
        cation_anion_cross_mobility_m2_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.cation_anion_cross_mobility_m2_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        cation_anion_center_mobility_m2_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.cation_anion_center_mobility_m2_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        charged_center_labels=point_diagnostics[0].charged_center_labels,
        charged_center_charge_numbers=point_diagnostics[0].charged_center_charge_numbers,
        charged_center_mobility_matrix_m2_s=_matrix_to_nested_tuple(
            _weighted_center_mobility_matrix(
                normalized_weights,
                point_diagnostics,
            )
        ),
        potential_energy_J_mol=float(
            np.dot(
                normalized_weights,
                [diagnostic.potential_energy_J_mol for diagnostic in point_diagnostics],
            )
        ),
        dielectric_constant=float(
            np.dot(
                normalized_weights,
                [diagnostic.dielectric_constant for diagnostic in point_diagnostics],
            )
        ),
        viscosity_Pa_s=float(
            np.dot(
                normalized_weights,
                [diagnostic.viscosity_Pa_s for diagnostic in point_diagnostics],
            )
        ),
        ionic_strength_mol_m3=float(
            np.dot(
                normalized_weights,
                [diagnostic.ionic_strength_mol_m3 for diagnostic in point_diagnostics],
            )
        ),
        local_packing_fraction=float(
            np.dot(
                normalized_weights,
                [diagnostic.local_packing_fraction for diagnostic in point_diagnostics],
            )
        ),
        atmosphere_electrophoretic_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_electrophoretic_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        atmosphere_relaxation_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_relaxation_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        atmosphere_cation_diagonal_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_cation_diagonal_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        atmosphere_anion_diagonal_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_anion_diagonal_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        atmosphere_cation_anion_cross_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_cation_anion_cross_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        atmosphere_mean_charge_cloud_form_factor=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_mean_charge_cloud_form_factor
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        atmosphere_mean_state_geometry_form_factor=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_mean_state_geometry_form_factor
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        atmosphere_minimum_separation_over_debye_length=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_minimum_separation_over_debye_length
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        atmosphere_debye_falkenhagen_time_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.atmosphere_debye_falkenhagen_time_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
    )


def _state_charge_mobility_point_terms(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    local_fields: PhysicalLocalFields,
    active_anion_name: str,
    temperature_K: float,
) -> StateChargeMobilityDiagnostics:
    physical_objects = build_physical_objects(
        records,
        configuration,
        temperature_K,
        local_fields.dielectric_constant,
        local_fields.viscosity_Pa_s,
        local_fields.ionic_strength_mol_m3,
        local_fields.local_packing_fraction,
    )
    mobility = np.asarray(physical_objects.mobility_tensor_m2_s, dtype=float)
    charges = _configuration_charge_numbers(records, configuration)
    cation_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if _species_role(records, species_name) == SpeciesRole.CATION
        and abs(charges[site_index]) > 0.0
    )
    anion_indices = (
        ()
        if active_anion_name == NO_ACTIVE_ANION_COMPONENT
        else _charged_site_indices_for_species(
            records,
            configuration,
            active_anion_name,
            charges,
        )
    )
    charged_center_indices = cation_indices + anion_indices
    charged_center_mobility_matrix = _charged_center_mobility_matrix(
        mobility,
        charges.size,
        charged_center_indices,
    )
    charged_center_charge_numbers = tuple(
        float(charges[site_index]) for site_index in charged_center_indices
    )
    charge_mobility_m2_s = charge_covariance_mobility_from_center_matrix(
        charged_center_charge_numbers,
        charged_center_mobility_matrix,
    )
    atmosphere_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        configuration,
        local_fields.dielectric_constant,
        local_fields.ionic_strength_mol_m3,
        temperature_K,
    )
    return StateChargeMobilityDiagnostics(
        charge_mobility_m2_s=charge_mobility_m2_s,
        cation_mobility_m2_s=_charged_subset_pair_scalar_mobility(
            mobility,
            charges,
            cation_indices,
            cation_indices,
        ),
        anion_mobility_m2_s=_charged_subset_pair_scalar_mobility(
            mobility,
            charges,
            anion_indices,
            anion_indices,
        ),
        cation_anion_cross_mobility_m2_s=_charged_subset_pair_scalar_mobility(
            mobility,
            charges,
            cation_indices,
            anion_indices,
        ),
        cation_anion_center_mobility_m2_s=_center_subset_pair_scalar_mobility(
            mobility,
            charges.size,
            cation_indices,
            anion_indices,
        ),
        charged_center_labels=tuple(
            configuration.species_names[site_index] for site_index in charged_center_indices
        ),
        charged_center_charge_numbers=charged_center_charge_numbers,
        charged_center_mobility_matrix_m2_s=_matrix_to_nested_tuple(
            charged_center_mobility_matrix
        ),
        potential_energy_J_mol=float(physical_objects.potential_energy_J_mol),
        dielectric_constant=float(local_fields.dielectric_constant),
        viscosity_Pa_s=float(local_fields.viscosity_Pa_s),
        ionic_strength_mol_m3=float(local_fields.ionic_strength_mol_m3),
        local_packing_fraction=float(local_fields.local_packing_fraction),
        atmosphere_electrophoretic_trace_kg_s=float(
            np.trace(atmosphere_diagnostics.electrophoretic_resistance_tensor_kg_s)
        ),
        atmosphere_relaxation_trace_kg_s=float(
            np.trace(atmosphere_diagnostics.relaxation_resistance_tensor_kg_s)
        ),
        atmosphere_cation_diagonal_trace_kg_s=float(
            atmosphere_diagnostics.cation_diagonal_resistance_trace_kg_s
        ),
        atmosphere_anion_diagonal_trace_kg_s=float(
            atmosphere_diagnostics.anion_diagonal_resistance_trace_kg_s
        ),
        atmosphere_cation_anion_cross_trace_kg_s=float(
            atmosphere_diagnostics.cation_anion_cross_resistance_trace_kg_s
        ),
        atmosphere_mean_charge_cloud_form_factor=float(
            atmosphere_diagnostics.mean_charge_cloud_form_factor
        ),
        atmosphere_mean_state_geometry_form_factor=float(
            atmosphere_diagnostics.mean_state_geometry_form_factor
        ),
        atmosphere_minimum_separation_over_debye_length=float(
            atmosphere_diagnostics.minimum_separation_over_debye_length
        ),
        atmosphere_debye_falkenhagen_time_s=float(
            atmosphere_diagnostics.debye_falkenhagen_time_s
        ),
    )


def _normalized_boltzmann_quadrature_weights(
    weights: Array,
    potential_energies_J_mol: tuple[float, ...],
    temperature_K: float,
) -> Array:
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or np.any(weight_array <= 0.0):
        raise ValueError("state quadrature weights must be a positive vector")
    if weight_array.size != len(potential_energies_J_mol):
        raise ValueError("state quadrature weights and energies must have same length")
    beta_mol = 1.0 / (R * _positive_float(temperature_K, "temperature_K"))
    log_terms = np.asarray(
        [
            np.log(float(weight)) - beta_mol * float(potential_energy_J_mol)
            for weight, potential_energy_J_mol in zip(
                weight_array,
                potential_energies_J_mol,
                strict=True,
            )
        ],
        dtype=float,
    )
    log_reference = _finite_float(float(np.max(log_terms)), "state diagnostic log weight")
    shifted_weights = np.exp(log_terms - log_reference)
    normalized_weight_sum = _positive_float(
        float(np.sum(shifted_weights)),
        "state diagnostic normalized weight sum",
    )
    return shifted_weights / normalized_weight_sum


def _active_anion_name_from_state_label(state_label: str) -> str:
    state_key = _state_key_from_label(state_label)
    anion_field = state_key[STATE_KEY_ANION_INDEX]
    if anion_field == "none":
        return NO_ACTIVE_ANION_COMPONENT
    parts = anion_field.split(":")
    if len(parts) != 2:
        raise ValueError(f"state anion field must be '<anion>:<feature>': {anion_field}")
    return parts[0]


def _configuration_charge_numbers(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return np.asarray(
        [
            float(records.species_records[species_name]["sites"][int(site_id)]["charge_number"])
            for species_name, site_id in zip(
                configuration.species_names,
                configuration.site_ids,
                strict=True,
            )
        ],
        dtype=float,
    )


def _charged_subset_pair_scalar_mobility(
    mobility_tensor_m2_s: Array,
    charge_numbers: Array,
    first_site_indices: tuple[int, ...],
    second_site_indices: tuple[int, ...],
) -> float:
    mobility = np.asarray(mobility_tensor_m2_s, dtype=float)
    charges = np.asarray(charge_numbers, dtype=float)
    if mobility.shape != (
        charges.size * CARTESIAN_DIMENSION,
        charges.size * CARTESIAN_DIMENSION,
    ):
        raise ValueError("mobility tensor shape does not match charge site count")
    total = 0.0
    for first_site_index in first_site_indices:
        for second_site_index in second_site_indices:
            first_charge = float(charges[first_site_index])
            second_charge = float(charges[second_site_index])
            first_start = first_site_index * CARTESIAN_DIMENSION
            second_start = second_site_index * CARTESIAN_DIMENSION
            block = mobility[
                first_start : first_start + CARTESIAN_DIMENSION,
                second_start : second_start + CARTESIAN_DIMENSION,
            ]
            total += (
                first_charge
                * second_charge
                * float(np.trace(block))
                / CARTESIAN_DIMENSION
            )
    return float(total)


def _center_subset_pair_scalar_mobility(
    mobility_tensor_m2_s: Array,
    site_count: int,
    first_site_indices: tuple[int, ...],
    second_site_indices: tuple[int, ...],
) -> float:
    mobility = np.asarray(mobility_tensor_m2_s, dtype=float)
    if mobility.shape != (
        site_count * CARTESIAN_DIMENSION,
        site_count * CARTESIAN_DIMENSION,
    ):
        raise ValueError("mobility tensor shape does not match site count")
    total = 0.0
    for first_site_index in first_site_indices:
        for second_site_index in second_site_indices:
            first_start = first_site_index * CARTESIAN_DIMENSION
            second_start = second_site_index * CARTESIAN_DIMENSION
            block = mobility[
                first_start : first_start + CARTESIAN_DIMENSION,
                second_start : second_start + CARTESIAN_DIMENSION,
            ]
            total += float(np.trace(block)) / CARTESIAN_DIMENSION
    return float(total)


def _charged_center_mobility_matrix(
    mobility_tensor_m2_s: Array,
    site_count: int,
    charged_center_indices: tuple[int, ...],
) -> Array:
    mobility = np.asarray(mobility_tensor_m2_s, dtype=float)
    if mobility.shape != (
        site_count * CARTESIAN_DIMENSION,
        site_count * CARTESIAN_DIMENSION,
    ):
        raise ValueError("mobility tensor shape does not match site count")
    center_count = len(charged_center_indices)
    center_mobility = np.zeros((center_count, center_count), dtype=float)
    for first_center_index, first_site_index in enumerate(charged_center_indices):
        for second_center_index, second_site_index in enumerate(charged_center_indices):
            first_start = first_site_index * CARTESIAN_DIMENSION
            second_start = second_site_index * CARTESIAN_DIMENSION
            block = mobility[
                first_start : first_start + CARTESIAN_DIMENSION,
                second_start : second_start + CARTESIAN_DIMENSION,
            ]
            center_mobility[first_center_index, second_center_index] = (
                float(np.trace(block)) / CARTESIAN_DIMENSION
            )
    return 0.5 * (center_mobility + center_mobility.T)


def charge_covariance_mobility_from_center_matrix(
    charge_numbers: tuple[float, ...],
    charged_center_mobility_matrix_m2_s: Array,
) -> float:
    charge_vector = np.asarray(charge_numbers, dtype=float)
    center_mobility = np.asarray(charged_center_mobility_matrix_m2_s, dtype=float)
    if center_mobility.shape != (charge_vector.size, charge_vector.size):
        raise ValueError("charged center mobility matrix shape does not match charges")
    if not np.all(np.isfinite(center_mobility)) or not np.all(np.isfinite(charge_vector)):
        raise ValueError("charged center covariance inputs must be finite")
    return float(charge_vector @ center_mobility @ charge_vector)


def _weighted_center_mobility_matrix(
    normalized_weights: Array,
    point_diagnostics: tuple[StateChargeMobilityDiagnostics, ...],
) -> Array:
    weights = np.asarray(normalized_weights, dtype=float)
    matrices = tuple(
        _nested_center_mobility_matrix_to_array(
            diagnostic.charged_center_mobility_matrix_m2_s
        )
        for diagnostic in point_diagnostics
    )
    if not matrices:
        raise ValueError("point diagnostics must not be empty")
    reference_shape = matrices[0].shape
    if any(matrix.shape != reference_shape for matrix in matrices):
        raise ValueError("charged-center mobility matrix shape changed within state")
    total = np.zeros(reference_shape, dtype=float)
    for weight, matrix in zip(weights, matrices, strict=True):
        total += float(weight) * matrix
    return total


def _nested_center_mobility_matrix_to_array(
    matrix: tuple[tuple[float, ...], ...],
) -> Array:
    if len(matrix) == 0:
        return np.zeros((0, 0), dtype=float)
    return np.asarray(matrix, dtype=float)


def _matrix_to_nested_tuple(matrix: Array) -> tuple[tuple[float, ...], ...]:
    result = np.asarray(matrix, dtype=float)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError("matrix must be finite and two-dimensional")
    return tuple(
        tuple(
            float(result[row_index, column_index])
            for column_index in range(result.shape[1])
        )
        for row_index in range(result.shape[0])
    )


def _charged_site_indices_for_species(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    species_name: str,
    charges: Array,
) -> tuple[int, ...]:
    if species_name not in configuration.species_names:
        raise ValueError(f"configuration missing active anion species {species_name}")
    return tuple(
        site_index
        for site_index, current_species_name in enumerate(configuration.species_names)
        if current_species_name == species_name and abs(charges[site_index]) > 0.0
    )


def enumerate_transition_edges(
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    records: PhysicalLibraryRecords,
) -> tuple[TransitionEdge, ...]:
    declared_families = tuple(str(family) for family in records.transition_record["families"])
    supported_families = (
        "free_to_SSIP",
        "SSIP_to_CIP",
        "SSIP_to_additive_separated_SSIP",
        "pair_to_aggregate",
        "Li_to_Li_ligand",
        "LiA_to_Li_ligand_anion",
        "partner_switch",
        "identity_diffusion",
        "structural_hop",
        "bridge_network_formation_breakup",
        "cage_capture_release",
        "atmosphere_capture_release",
    )
    unsupported_families = tuple(
        family for family in declared_families if family not in supported_families
    )
    if unsupported_families:
        raise ValueError(
            "transition builder cannot instantiate declared transition families: "
            f"{unsupported_families}"
        )
    edges: list[TransitionEdge] = []
    generated_edge_counts = {family: 0 for family in declared_families}
    state_keys = tuple(_state_key_from_label(state.label) for state in state_quadratures)
    for from_state_index, from_key in enumerate(state_keys):
        for to_state_index, to_key in enumerate(state_keys):
            if from_state_index == to_state_index:
                continue
            family = _transition_family_for_state_keys(from_key, to_key)
            if family == NO_TRANSITION_FAMILY or family not in declared_families:
                continue
            edges.append(
                TransitionEdge(
                    from_state_index=from_state_index,
                    to_state_index=to_state_index,
                    family=family,
                )
            )
            generated_edge_counts[family] += 1
    missing_applicable_families = tuple(
        family
        for family in declared_families
        if _transition_family_is_applicable(family, state_keys)
        and generated_edge_counts[family] == 0
    )
    if missing_applicable_families:
        raise ValueError(
            "transition builder generated zero edges for applicable declared "
            f"families: {missing_applicable_families}"
        )
    return tuple(edges)


def finite_generator_transition_edges(
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    temperature_K: float,
) -> tuple[TransitionEdge, ...]:
    transition_edges = enumerate_transition_edges(state_quadratures, records)
    if not transition_edges:
        return transition_edges
    log_partition_values = np.asarray(
        [
            _state_log_partition_value(records, state_quadrature, temperature_K)
            for state_quadrature in state_quadratures
        ],
        dtype=float,
    )
    reference_log_partition = _finite_float(
        float(
            np.max(
                [
                    min(
                        float(log_partition_values[edge.from_state_index]),
                        float(log_partition_values[edge.to_state_index]),
                    )
                    for edge in transition_edges
                ]
            )
        ),
        "finite_generator_reference_edge_log_partition",
    )
    log_partition_span = _positive_float(
        float(records.basis_record["finite_generator_log_partition_span"]),
        "basis.finite_generator_log_partition_span",
    )
    retained_edges = tuple(
        edge
        for edge in transition_edges
        if log_partition_values[edge.from_state_index]
        >= reference_log_partition - log_partition_span
        and log_partition_values[edge.to_state_index]
        >= reference_log_partition - log_partition_span
    )
    if not retained_edges:
        raise ValueError(
            "finite-generator partition filter removed every transition edge; "
            "increase basis.finite_generator_log_partition_span or fix generated state energies"
        )
    return retained_edges


def _state_key_from_label(label: str) -> tuple[str, ...]:
    state_key = tuple(label.split("|"))
    if len(state_key) != STATE_KEY_LENGTH:
        raise ValueError(f"state label has wrong key length: {label}")
    return state_key


def _transition_family_for_state_keys(
    from_key: tuple[str, ...],
    to_key: tuple[str, ...],
) -> str:
    from_pair = from_key[STATE_KEY_PAIR_INDEX]
    to_pair = to_key[STATE_KEY_PAIR_INDEX]
    from_shell = from_key[STATE_KEY_SHELL_INDEX]
    to_shell = to_key[STATE_KEY_SHELL_INDEX]
    from_ligand = _state_key_base_value(from_key[STATE_KEY_LIGAND_INDEX])
    to_ligand = _state_key_base_value(to_key[STATE_KEY_LIGAND_INDEX])
    from_cluster = from_key[STATE_KEY_CLUSTER_INDEX]
    to_cluster = to_key[STATE_KEY_CLUSTER_INDEX]
    from_partner = from_key[STATE_KEY_PARTNER_INDEX]
    to_partner = to_key[STATE_KEY_PARTNER_INDEX]
    from_identity = from_key[STATE_KEY_IDENTITY_INDEX]
    to_identity = to_key[STATE_KEY_IDENTITY_INDEX]
    from_hop = from_key[STATE_KEY_HOP_INDEX]
    to_hop = to_key[STATE_KEY_HOP_INDEX]
    from_cage = from_key[STATE_KEY_CAGE_INDEX]
    to_cage = to_key[STATE_KEY_CAGE_INDEX]
    from_atmosphere = from_key[STATE_KEY_ATMOSPHERE_INDEX]
    to_atmosphere = to_key[STATE_KEY_ATMOSPHERE_INDEX]
    if (
        _state_keys_match_on_indices(from_key, to_key, PAIR_TRANSITION_KEY_INDICES)
        and from_pair == PairBasin.FREE.value
        and to_pair == PairBasin.SOLVENT_SEPARATED_ION_PAIR.value
    ):
        return "free_to_SSIP"
    if (
        _state_keys_match_on_indices(from_key, to_key, PAIR_TRANSITION_KEY_INDICES)
        and from_pair == PairBasin.SOLVENT_SEPARATED_ION_PAIR.value
        and to_pair == PairBasin.CONTACT_ION_PAIR.value
    ):
        return "SSIP_to_CIP"
    if (
        _state_keys_match_on_indices(
            from_key,
            to_key,
            ADDITIVE_SEPARATED_PAIR_TRANSITION_KEY_INDICES,
        )
        and from_pair == PairBasin.SOLVENT_SEPARATED_ION_PAIR.value
        and to_pair == "addSSIP"
        and from_ligand == "none"
        and to_ligand == LIGAND_STATE_ADDITIVE_SEPARATOR
    ):
        return "SSIP_to_additive_separated_SSIP"
    if (
        _state_keys_match_on_indices(from_key, to_key, LIGAND_TRANSITION_KEY_INDICES)
        and from_ligand == "none"
        and to_ligand != "none"
    ):
        return "Li_to_Li_ligand"
    if (
        _state_keys_match_on_indices(from_key, to_key, SHELL_TRANSITION_KEY_INDICES)
        and from_shell == "anion_coordinated"
        and to_shell == "mixed_ligand_anion"
    ):
        return "LiA_to_Li_ligand_anion"
    if (
        _state_keys_match_on_indices(from_key, to_key, CLUSTER_TRANSITION_KEY_INDICES)
        and from_cluster not in ("aggregate", "bridge_network")
        and to_cluster == "aggregate"
    ):
        return "pair_to_aggregate"
    if (
        _state_keys_match_on_indices(from_key, to_key, CLUSTER_TRANSITION_KEY_INDICES)
        and from_cluster == "aggregate"
        and to_cluster == "bridge_network"
    ):
        return "bridge_network_formation_breakup"
    if (
        _state_keys_match_on_indices(from_key, to_key, PARTNER_TRANSITION_KEY_INDICES)
        and _ordered_label_transition(
            from_partner,
            to_partner,
            ("partner_a", "partner_switching", "partner_b"),
        )
    ):
        return "partner_switch"
    if (
        _state_keys_match_on_indices(from_key, to_key, IDENTITY_TRANSITION_KEY_INDICES)
        and _ordered_label_transition(
            from_identity,
            to_identity,
            (
                "carrier_identity_a",
                "carrier_identity_transition",
                "carrier_identity_b",
            ),
        )
    ):
        return "identity_diffusion"
    if (
        _state_keys_match_on_indices(from_key, to_key, HOP_TRANSITION_KEY_INDICES)
        and _ordered_label_transition(
            from_hop,
            to_hop,
            ("hop_source", "hop_transition", "hop_target"),
        )
    ):
        return "structural_hop"
    if (
        _state_keys_match_on_indices(from_key, to_key, CAGE_TRANSITION_KEY_INDICES)
        and _ordered_label_transition(
            from_cage,
            to_cage,
            ("cage_released", "cage_boundary", "cage_captured"),
        )
    ):
        return "cage_capture_release"
    if (
        _state_keys_match_on_indices(from_key, to_key, ATMOSPHERE_TRANSITION_KEY_INDICES)
        and from_atmosphere != to_atmosphere
    ):
        return "atmosphere_capture_release"
    return NO_TRANSITION_FAMILY


def _ordered_label_transition(
    from_label: str,
    to_label: str,
    ordered_labels: tuple[str, ...],
) -> bool:
    label_rank = {label: label_index for label_index, label in enumerate(ordered_labels)}
    if from_label not in label_rank or to_label not in label_rank:
        return False
    return label_rank[to_label] == label_rank[from_label] + 1


def _state_keys_match_on_indices(
    first_key: tuple[str, ...],
    second_key: tuple[str, ...],
    key_indices: tuple[int, ...],
) -> bool:
    for key_index in key_indices:
        if first_key[key_index] != second_key[key_index]:
            return False
    return True


def _transition_family_is_applicable(
    family: str,
    state_keys: tuple[tuple[str, ...], ...],
) -> bool:
    for from_key in state_keys:
        for to_key in state_keys:
            if from_key == to_key:
                continue
            if _transition_family_for_state_keys(from_key, to_key) == family:
                return True
    return False


def project_diffusivity_onto_reaction_coordinate(
    mobility_tensor_m2_s: Array,
    reaction_coordinate_gradient: Array,
) -> float:
    mobility = np.asarray(mobility_tensor_m2_s, dtype=float)
    gradient = np.asarray(reaction_coordinate_gradient, dtype=float)
    if mobility.ndim != 2 or mobility.shape[0] != mobility.shape[1]:
        raise ValueError("mobility_tensor_m2_s must be square")
    if gradient.shape != (mobility.shape[0],):
        raise ValueError("reaction_coordinate_gradient length must match mobility")
    projected = float(gradient @ mobility @ gradient.T)
    return _positive_float(projected, "projected reaction-coordinate diffusivity")


def build_all_memory_coordinate_gradients(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    mixture: MixtureClosureResult,
    numerical_options: NumericalOptions,
):
    _ = state_quadratures
    _ = mixture
    _validate_numerical_options(numerical_options)
    memory_coordinates = build_default_memory_coordinates(records, template_configuration)
    implemented_family_map = {
        "atmosphere_polarization": {"atmosphere_polarization"},
        "charge_density_relaxation": {
            "charge_density_cosine",
            "charge_density_sine",
        },
        "cage_backjump": {"cage_backjump"},
        "partner_residence": {"partner_residence"},
        "ligand_shell_residence": {"ligand_shell"},
        "anion_orientation": {"anion_orientation"},
        "free_volume_stress": {"free_volume_stress"},
        "bounded_internal_polarization": {"bounded_internal_polarization"},
    }
    implemented_coordinate_families = {
        memory_coordinate.family.value for memory_coordinate in memory_coordinates
    }
    missing_families = []
    for declared_family in records.memory_record["families"]:
        declared_family_name = str(declared_family)
        if declared_family_name not in implemented_family_map:
            missing_families.append(declared_family_name)
            continue
        implementation_names = implemented_family_map[declared_family_name]
        if implemented_coordinate_families.isdisjoint(implementation_names):
            missing_families.append(declared_family_name)
    if missing_families:
        raise ValueError(
            "memory builder cannot instantiate declared memory families: "
            f"{tuple(missing_families)}"
        )
    allowed_implementation_families: set[str] = set()
    for declared_family in records.memory_record["families"]:
        allowed_implementation_families.update(
            implemented_family_map[str(declared_family)]
        )
    selected_memory_coordinates = tuple(
        memory_coordinate
        for memory_coordinate in memory_coordinates
        if memory_coordinate.family.value in allowed_implementation_families
    )
    return tuple(
        _bind_memory_gradient(memory_coordinate)
        for memory_coordinate in selected_memory_coordinates
    )


def build_default_memory_coordinates(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
) -> tuple[MemoryCoordinate, ...]:
    """Build current-coupled memory coordinates from physical records."""

    _ = template_configuration
    return (
        MemoryCoordinate(
            family=MemoryCoordinateFamily.CAGE_BACKJUMP,
            records=records,
            value_function=_cage_backjump_memory_value,
            gradient_function=_memory_pair_distance_gradient,
        ),
        MemoryCoordinate(
            family=MemoryCoordinateFamily.PARTNER_RESIDENCE,
            records=records,
            value_function=_partner_residence_memory_value,
            gradient_function=_memory_li_anion_coordination_gradient,
        ),
        MemoryCoordinate(
            family=MemoryCoordinateFamily.LIGAND_SHELL,
            records=records,
            value_function=_ligand_shell_memory_value,
            gradient_function=_memory_li_ligand_coordination_gradient,
        ),
        MemoryCoordinate(
            family=MemoryCoordinateFamily.ANION_ORIENTATION,
            records=records,
            value_function=_anion_orientation_memory_value,
            gradient_function=_zero_memory_gradient,
        ),
        MemoryCoordinate(
            family=MemoryCoordinateFamily.FREE_VOLUME_STRESS,
            records=records,
            value_function=_free_volume_stress_memory_value,
            gradient_function=_zero_memory_gradient,
        ),
        MemoryCoordinate(
            family=MemoryCoordinateFamily.BOUNDED_INTERNAL_POLARIZATION,
            records=records,
            value_function=_bounded_internal_polarization_memory_value,
            gradient_function=_bounded_internal_polarization_memory_gradient,
        ),
    )


def combine_memory_values(
    coordinates: tuple[MemoryCoordinate, ...],
    configuration: SiteConfiguration,
) -> Array:
    return np.asarray(
        [
            coordinate.value_function(coordinate.records, configuration)
            for coordinate in coordinates
        ],
        dtype=float,
    )


def combine_memory_gradients(
    coordinates: tuple[MemoryCoordinate, ...],
    configuration: SiteConfiguration,
) -> Array:
    if not coordinates:
        coordinate_count = len(configuration.species_names) * CARTESIAN_DIMENSION
        return np.zeros((0, coordinate_count), dtype=float)
    return np.vstack(
        tuple(
            _memory_gradient_row(
                coordinate.gradient_function(coordinate.records, configuration)
            )
            for coordinate in coordinates
        )
    )


def _bind_memory_gradient(
    memory_coordinate: MemoryCoordinate,
) -> Callable[[SiteConfiguration], Array]:
    def gradient(configuration: SiteConfiguration) -> Array:
        return _memory_gradient_row(
            memory_coordinate.gradient_function(
                memory_coordinate.records,
                configuration,
            )
        )

    return gradient


def _memory_gradient_row(gradient: Array) -> Array:
    row = np.asarray(gradient, dtype=float)
    if row.ndim == 1:
        return row.reshape((1, row.size))
    if row.ndim == 2 and row.shape[0] == 1:
        return row
    raise ValueError("memory gradient must be a single row")


def _ligand_shell_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    return compute_role_coordination_number(
        records,
        configuration,
        center_role=SpeciesRole.CATION.value,
        ligand_role=SpeciesRole.ADDITIVE.value,
        switch_name="Li_ligand",
    )


def _memory_li_ligand_coordination_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _coordination_switch_gradient(records, configuration, "Li_ligand")


def _cage_backjump_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    return _li_anion_distance_m(records, configuration) - float(
        records.basis_record["pair_basins"]["r_SSIP_m"]
    )


def _partner_residence_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    return compute_role_coordination_number(
        records,
        configuration,
        center_role=SpeciesRole.CATION.value,
        ligand_role=SpeciesRole.ANION.value,
        switch_name="Li_anion",
    )


def _memory_li_anion_coordination_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _coordination_switch_gradient(records, configuration, "Li_anion")


def _bounded_internal_polarization_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_index = _first_role_index(records, configuration, SpeciesRole.ANION)
    pair_vector = (
        np.asarray(configuration.positions_m[anion_index], dtype=float)
        - np.asarray(configuration.positions_m[cation_index], dtype=float)
    )
    charge_numbers = _configuration_charge_numbers(records, configuration)
    return float(
        (charge_numbers[cation_index] - charge_numbers[anion_index]) * pair_vector[0]
    )


def _bounded_internal_polarization_memory_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_index = _first_role_index(records, configuration, SpeciesRole.ANION)
    charge_numbers = _configuration_charge_numbers(records, configuration)
    charge_difference = charge_numbers[cation_index] - charge_numbers[anion_index]
    gradient = np.zeros(
        (1, len(configuration.species_names) * CARTESIAN_DIMENSION),
        dtype=float,
    )
    gradient[0, cation_index * CARTESIAN_DIMENSION] = -charge_difference
    gradient[0, anion_index * CARTESIAN_DIMENSION] = charge_difference
    return gradient


def _anion_orientation_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_index = _first_role_index(records, configuration, SpeciesRole.ANION)
    orientation = assign_orientation_basin(
        records,
        configuration,
        cation_index,
        anion_index,
    )
    if orientation not in ORIENTATION_MEMORY_VALUES:
        raise ValueError(f"unsupported orientation basin {orientation}")
    return ORIENTATION_MEMORY_VALUES[orientation]


def _free_volume_stress_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    packing_fraction = compute_local_packing_fraction(records, configuration)
    phi_max = float(records.mixture_record["packing"]["phi_max"])
    if packing_fraction >= phi_max:
        raise ValueError("packing_fraction must be below phi_max")
    return packing_fraction / (phi_max - packing_fraction)


def _zero_memory_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    _ = records
    return np.zeros(
        (1, len(configuration.species_names) * CARTESIAN_DIMENSION),
        dtype=float,
    )


def _memory_pair_distance_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _memory_gradient_row(_pair_distance_gradient(records, configuration))


def _declared_reduced_coordinates(
    records: PhysicalLibraryRecords,
) -> tuple[ReducedCoordinate, ...]:
    coordinates = []
    for coordinate_name in records.basis_record["coordinates"]:
        matched_coordinate = None
        for reduced_coordinate in ReducedCoordinate:
            if reduced_coordinate.value == str(coordinate_name):
                matched_coordinate = reduced_coordinate
        if matched_coordinate is None:
            raise ValueError(f"unsupported reduced coordinate {coordinate_name}")
        coordinates.append(matched_coordinate)
    return tuple(coordinates)


def _state_coordinate_nodes(
    records: PhysicalLibraryRecords,
    declared_coordinates: tuple[ReducedCoordinate, ...],
    lower_distance_m: float,
    upper_distance_m: float,
    recipe_context: RecipeBuildResult,
    mixture: MixtureClosureResult,
    numerical_options: NumericalOptions,
) -> tuple[tuple[ReducedCoordinate, Array, Array], ...]:
    nodes: list[tuple[ReducedCoordinate, Array, Array]] = []
    for coordinate in declared_coordinates:
        if coordinate == ReducedCoordinate.LI_ANION_DISTANCE:
            coordinate_values, coordinate_weights = _gauss_legendre_interval(
                lower_distance_m,
                upper_distance_m,
                numerical_options.state_quadrature_order,
            )
            nodes.append((coordinate, coordinate_values, coordinate_weights))
            continue
        coordinate_values, coordinate_weights = _non_distance_coordinate_nodes(
            records,
            coordinate,
            recipe_context,
            mixture,
        )
        nodes.append((coordinate, coordinate_values, coordinate_weights))
    return tuple(nodes)


def _non_distance_coordinate_nodes(
    records: PhysicalLibraryRecords,
    coordinate: ReducedCoordinate,
    recipe_context: RecipeBuildResult,
    mixture: MixtureClosureResult,
) -> tuple[Array, Array]:
    if coordinate == ReducedCoordinate.LI_LIGAND_COORDINATION:
        return _ligand_coordination_nodes(records, coordinate, recipe_context)
    if coordinate == ReducedCoordinate.LI_SOLVENT_COORDINATION:
        return _solvent_coordination_nodes(records, coordinate, recipe_context)
    if coordinate == ReducedCoordinate.LOCAL_IONIC_STRENGTH:
        return (
            np.asarray([mixture.ionic_strength_mol_m3], dtype=float),
            np.asarray([1.0], dtype=float),
        )
    if coordinate == ReducedCoordinate.LOCAL_DIELECTRIC:
        return (
            np.asarray([mixture.dielectric_constant], dtype=float),
            np.asarray([1.0], dtype=float),
        )
    if coordinate == ReducedCoordinate.LOCAL_VISCOSITY:
        return (
            np.asarray([mixture.viscosity_Pa_s], dtype=float),
            np.asarray([1.0], dtype=float),
        )
    return _configured_coordinate_nodes(records, coordinate)


def _ligand_coordination_nodes(
    records: PhysicalLibraryRecords,
    coordinate: ReducedCoordinate,
    recipe_context: RecipeBuildResult,
) -> tuple[Array, Array]:
    if not _recipe_has_role(recipe_context, SpeciesRole.ADDITIVE):
        return np.asarray([0.0], dtype=float), np.asarray([1.0], dtype=float)
    return _configured_coordinate_nodes(records, coordinate)


def _solvent_coordination_nodes(
    records: PhysicalLibraryRecords,
    coordinate: ReducedCoordinate,
    recipe_context: RecipeBuildResult,
) -> tuple[Array, Array]:
    if not _recipe_has_role(recipe_context, SpeciesRole.SOLVENT):
        return np.asarray([0.0], dtype=float), np.asarray([1.0], dtype=float)
    return _configured_coordinate_nodes(records, coordinate)


def _configured_coordinate_nodes(
    records: PhysicalLibraryRecords,
    coordinate: ReducedCoordinate,
) -> tuple[Array, Array]:
    if _unit_interval_coordinate_is_inactive(records, coordinate):
        return np.asarray([0.0], dtype=float), np.asarray([1.0], dtype=float)
    domains = records.basis_record["coordinate_domains"]
    if coordinate.value not in domains:
        raise KeyError(f"basis coordinate_domains missing {coordinate.value}")
    domain = domains[coordinate.value]
    lower = float(domain["lower"])
    upper = float(domain["upper"])
    order = int(domain["quadrature_order"])
    if order == 1:
        bin_nodes = _configured_bin_nodes(records, coordinate, lower, upper)
        if bin_nodes[0].size > 0:
            return bin_nodes
        return (
            np.asarray([(lower + upper) / 2.0], dtype=float),
            np.asarray([upper - lower], dtype=float),
        )
    return _gauss_legendre_interval(lower, upper, order)


def _unit_interval_coordinate_is_inactive(
    records: PhysicalLibraryRecords,
    coordinate: ReducedCoordinate,
) -> bool:
    if coordinate not in _unit_interval_state_coordinates():
        return False
    active_coordinates = {
        str(coordinate_name)
        for coordinate_name in records.basis_record["active_state_axis_coordinates"]
    }
    return coordinate.value not in active_coordinates


def _configured_bin_nodes(
    records: PhysicalLibraryRecords,
    coordinate: ReducedCoordinate,
    lower: float,
    upper: float,
) -> tuple[Array, Array]:
    orientation_bins = records.basis_record["orientation_bins"]
    coordination_switch_by_coordinate = {
        ReducedCoordinate.LI_SOLVENT_COORDINATION: "Li_solvent",
        ReducedCoordinate.LI_LIGAND_COORDINATION: "Li_ligand",
        ReducedCoordinate.LI_ANION_COORDINATION: "Li_anion",
    }
    if coordinate in coordination_switch_by_coordinate:
        return _coordination_basin_representative_nodes(
            records,
            coordination_switch_by_coordinate[coordinate],
            lower,
            upper,
        )
    threshold_by_coordinate = {
        ReducedCoordinate.ANION_ORIENTATION: [
            orientation_bins["bridging_max"],
            -float(orientation_bins["tangential_abs_max"]),
            orientation_bins["tangential_abs_max"],
            orientation_bins["radial_min"],
        ],
        ReducedCoordinate.LOCAL_PACKING_FRACTION: records.basis_record[
            "environment_bins"
        ]["packing_fraction"],
        ReducedCoordinate.LOCAL_IONIC_STRENGTH: records.basis_record[
            "environment_bins"
        ]["ionic_strength_mol_m3"],
        ReducedCoordinate.LOCAL_DIELECTRIC: records.basis_record["environment_bins"][
            "dielectric"
        ],
        ReducedCoordinate.LOCAL_VISCOSITY: records.basis_record["environment_bins"][
            "viscosity_Pa_s"
        ],
        ReducedCoordinate.ATMOSPHERE_POLARIZATION: records.basis_record[
            "unit_interval_state_bins"
        ],
        ReducedCoordinate.CAGE_COORDINATE: records.basis_record[
            "unit_interval_state_bins"
        ],
        ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE: records.basis_record[
            "unit_interval_state_bins"
        ],
        ReducedCoordinate.CLUSTER_COORDINATE: records.basis_record[
            "unit_interval_state_bins"
        ],
        ReducedCoordinate.IDENTITY_COORDINATE: records.basis_record[
            "unit_interval_state_bins"
        ],
        ReducedCoordinate.STRUCTURAL_HOP_COORDINATE: records.basis_record[
            "unit_interval_state_bins"
        ],
    }
    if coordinate not in threshold_by_coordinate:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return _nodes_from_thresholds(lower, upper, threshold_by_coordinate[coordinate])


def _coordination_basin_representative_nodes(
    records: PhysicalLibraryRecords,
    switch_name: str,
    lower: float,
    upper: float,
) -> tuple[Array, Array]:
    cutoff = _coordination_cutoff(records, switch_name)
    if not lower < cutoff < upper:
        raise ValueError(f"{switch_name} cutoff must lie inside coordinate domain")
    below_value = (lower + cutoff) / 2.0
    interior_fraction = _positive_float(
        float(records.basis_record["coordination_high_bin_interior_fraction"]),
        "basis.coordination_high_bin_interior_fraction",
    )
    if interior_fraction >= 1.0:
        raise ValueError("basis.coordination_high_bin_interior_fraction must be below one")
    above_value = cutoff + min(cutoff - lower, upper - cutoff) * interior_fraction
    if not cutoff < above_value < upper:
        raise ValueError(f"{switch_name} high-bin representative is outside domain")
    return (
        np.asarray([below_value, above_value], dtype=float),
        np.asarray([cutoff - lower, upper - cutoff], dtype=float),
    )


def _nodes_from_thresholds(
    lower: float,
    upper: float,
    thresholds: list[float],
) -> tuple[Array, Array]:
    sorted_thresholds = [
        float(threshold)
        for threshold in sorted(thresholds)
        if lower < float(threshold) < upper
    ]
    bounds = [lower, *sorted_thresholds, upper]
    values = []
    weights = []
    for lower_bound, upper_bound in zip(bounds[:-1], bounds[1:], strict=True):
        if not lower_bound < upper_bound:
            raise ValueError("coordinate bin bounds must be increasing")
        values.append((lower_bound + upper_bound) / 2.0)
        weights.append(upper_bound - lower_bound)
    return np.asarray(values, dtype=float), np.asarray(weights, dtype=float)


def _recipe_has_role(
    recipe_context: RecipeBuildResult,
    role: SpeciesRole,
) -> bool:
    for component in recipe_context.components:
        if _species_role(recipe_context.library_records, component.name) == role:
            return True
    return False


def _configuration_with_reduced_coordinate_values(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
) -> SiteConfiguration:
    configuration = _configuration_with_pair_distance(
        records,
        template_configuration,
        _positive_float(
            coordinate_values[ReducedCoordinate.LI_ANION_DISTANCE.value],
            ReducedCoordinate.LI_ANION_DISTANCE.value,
        ),
    )
    configuration = _configuration_with_role_coordination(
        records,
        configuration,
        SpeciesRole.SOLVENT,
        "Li_solvent",
        float(coordinate_values[ReducedCoordinate.LI_SOLVENT_COORDINATION.value]),
    )
    configuration = _configuration_with_role_coordination(
        records,
        configuration,
        SpeciesRole.ADDITIVE,
        "Li_ligand",
        float(coordinate_values[ReducedCoordinate.LI_LIGAND_COORDINATION.value]),
    )
    configuration = _configuration_with_anion_orientation(
        records,
        configuration,
        float(coordinate_values[ReducedCoordinate.ANION_ORIENTATION.value]),
    )
    return configuration


def _configuration_with_role_coordination(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    ligand_role: SpeciesRole,
    switch_name: str,
    coordination_value: float,
) -> SiteConfiguration:
    ligand_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if _species_role(records, species_name) == ligand_role
    )
    if not ligand_indices:
        if abs(coordination_value) > 0.0:
            raise ValueError(
                f"{ligand_role.value} coordination is nonzero but no ligand is present"
            )
        return configuration
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    switch_record = records.basis_record["coordination_switches"][switch_name]
    ligand_site_count = len(ligand_indices)
    single_ligand_coordination = coordination_value / ligand_site_count
    distance_m = _coordination_switch_inverse_distance_m(
        single_ligand_coordination,
        float(switch_record["r0_m"]),
        float(switch_record["exponent"]),
        float(switch_record["minimum_value"]),
    )
    positions = np.asarray(configuration.positions_m, dtype=float).copy()
    cation_position = positions[cation_index].copy()
    molecule_ids = tuple(
        sorted({int(configuration.molecule_ids[index]) for index in ligand_indices})
    )
    for molecule_offset, molecule_id in enumerate(molecule_ids):
        molecule_site_indices = tuple(
            site_index
            for site_index in ligand_indices
            if int(configuration.molecule_ids[site_index]) == molecule_id
        )
        anchor_index = molecule_site_indices[0]
        target_position = cation_position + np.asarray(
            [0.0, distance_m, (molecule_offset + 1) * distance_m],
            dtype=float,
        )
        shift = target_position - positions[anchor_index]
        for site_index in molecule_site_indices:
            positions[site_index] += shift
    return SiteConfiguration(
        species_names=configuration.species_names,
        molecule_ids=np.asarray(configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=positions,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _coordination_switch_inverse_distance_m(
    coordination_value: float,
    switch_radius_m: float,
    exponent: float,
    minimum_value: float,
) -> float:
    if minimum_value <= 0.0 or minimum_value >= 1.0:
        raise ValueError("coordination switch minimum_value must be between zero and one")
    if exponent <= 0.0:
        raise ValueError("coordination switch exponent must be positive")
    if coordination_value <= 0.0:
        coordination_value = minimum_value
    if coordination_value >= 1.0:
        raise ValueError("coordination_value must be below one per ligand site")
    return switch_radius_m * float(
        np.power((1.0 / coordination_value) - 1.0, 1.0 / exponent)
    )


def _configuration_with_anion_orientation(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    orientation_cosine: float,
) -> SiteConfiguration:
    if orientation_cosine < -1.0 or orientation_cosine > 1.0:
        raise ValueError("anion_orientation coordinate must be within [-1, 1]")
    anion_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if _species_role(records, species_name) == SpeciesRole.ANION
    )
    if len(anion_indices) < 2:
        return configuration
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_anchor = anion_indices[0]
    anion_orient_site = anion_indices[1]
    positions = np.asarray(configuration.positions_m, dtype=float).copy()
    pair_axis = positions[anion_anchor] - positions[cation_index]
    pair_axis_norm = _positive_float(float(np.linalg.norm(pair_axis)), "pair_axis_norm")
    unit_pair_axis = pair_axis / pair_axis_norm
    perpendicular_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    if (
        abs(float(np.dot(unit_pair_axis, perpendicular_axis)))
        > PERPENDICULAR_AXIS_ALIGNMENT_LIMIT
    ):
        perpendicular_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
    perpendicular_axis = perpendicular_axis - (
        float(np.dot(unit_pair_axis, perpendicular_axis)) * unit_pair_axis
    )
    perpendicular_axis = perpendicular_axis / _positive_float(
        float(np.linalg.norm(perpendicular_axis)),
        "perpendicular_axis_norm",
    )
    original_vector = positions[anion_orient_site] - positions[anion_anchor]
    original_length = _positive_float(
        float(np.linalg.norm(original_vector)),
        "anion_orientation_vector_length",
    )
    sine_component = float(np.sqrt(max(0.0, 1.0 - orientation_cosine * orientation_cosine)))
    target_vector = original_length * (
        orientation_cosine * unit_pair_axis + sine_component * perpendicular_axis
    )
    positions[anion_orient_site] = positions[anion_anchor] + target_vector
    return SiteConfiguration(
        species_names=configuration.species_names,
        molecule_ids=np.asarray(configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=positions,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _state_distance_bounds(
    records: PhysicalLibraryRecords,
    recipe_context: RecipeBuildResult,
) -> tuple[tuple[str, float, float], ...]:
    pair_basins = records.basis_record["pair_basins"]
    contact_cutoff_m = float(pair_basins["r_CIP_m"])
    solvent_separated_cutoff_m = float(pair_basins["r_SSIP_m"])
    free_cutoff_m = float(pair_basins["r_free_m"])
    free_outer_factor = float(records.basis_record["quadrature"]["free_outer_distance_factor"])
    specs = [
        (PairBasin.CONTACT_ION_PAIR.value, contact_cutoff_m / 2.0, contact_cutoff_m),
        (
            PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
            contact_cutoff_m,
            solvent_separated_cutoff_m,
        ),
        (PairBasin.FREE.value, free_cutoff_m, free_outer_factor * free_cutoff_m),
    ]
    if any(
        _species_role(records, component.name) == SpeciesRole.ADDITIVE
        for component in recipe_context.components
    ):
        specs.append(("addSSIP", contact_cutoff_m, solvent_separated_cutoff_m))
    return tuple(specs)


def _normalize_potential_energy_reference(
    reduced_specification: ReducedGeneratorSpecification,
) -> ReducedGeneratorSpecification:
    sampled_energies = []
    transition_state_indices = {
        int(transition_quadrature.from_state_index)
        for transition_quadrature in reduced_specification.transition_quadratures
    } | {
        int(transition_quadrature.to_state_index)
        for transition_quadrature in reduced_specification.transition_quadratures
    }
    if transition_state_indices:
        reference_state_indices = transition_state_indices
    else:
        reference_state_indices = set(range(len(reduced_specification.state_quadratures)))
    non_reference_point_keys = {
        _reduced_point_cache_key(point)
        for state_index, state_quadrature in enumerate(
            reduced_specification.state_quadratures
        )
        if state_index not in reference_state_indices
        for point in state_quadrature.points
    }
    for state_index, state_quadrature in enumerate(reduced_specification.state_quadratures):
        if state_index not in reference_state_indices:
            continue
        for point in state_quadrature.points:
            sampled_energies.append(
                float(reduced_specification.potential_energy_J_mol(point))
            )
    energy_reference_J_mol = _finite_float(
        float(np.min(np.asarray(sampled_energies, dtype=float))),
        "energy_reference_J_mol",
    )

    def shifted_potential_energy_J_mol(point: Array) -> float:
        if _reduced_point_cache_key(point) in non_reference_point_keys:
            return 0.0
        return (
            float(reduced_specification.potential_energy_J_mol(point))
            - energy_reference_J_mol
        )

    return replace(
        reduced_specification,
        potential_energy_J_mol=shifted_potential_energy_J_mol,
    )


def _reduced_point_cache_key(point: Array) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(point, dtype=float), np.finfo(float).precision))


def _projected_mass_balance_components(recipe_context: RecipeBuildResult):
    components = tuple(
        _component_with_active_recipe_loading(recipe_context, component)
        for component in recipe_context.components
        if _species_role(recipe_context.library_records, component.name)
        in (SpeciesRole.CATION, SpeciesRole.ANION, SpeciesRole.ADDITIVE)
    )
    if not components:
        raise ValueError("projected conductivity mass balance needs charged components")
    return components


def _component_with_active_recipe_loading(
    recipe_context: RecipeBuildResult,
    component: RecipeComponentLoading,
) -> RecipeComponentLoading:
    if _species_role(recipe_context.library_records, component.name) != SpeciesRole.ADDITIVE:
        return component
    additive_weight_fraction = recipe_context.additive_weight_fractions[component.name]
    species_record = recipe_context.library_records.species_records[component.name]
    molecular_weight_kg_mol = _positive_float(
        float(species_record["molecular_weight_kg_mol"]),
        f"{component.name}.molecular_weight_kg_mol",
    )
    reference_density_kg_m3 = _positive_float(
        float(recipe_context.library_records.mixture_record["reference_density_kg_m3"]),
        "mixture.reference_density_kg_m3",
    )
    return replace(
        component,
        concentration_mol_m3=(
            additive_weight_fraction * reference_density_kg_m3 / molecular_weight_kg_mol
        ),
    )


def _transition_distance_bounds(
    records: PhysicalLibraryRecords,
    from_pair_label: str,
    to_pair_label: str,
) -> tuple[float, float]:
    pair_basins = records.basis_record["pair_basins"]
    contact_cutoff_m = float(pair_basins["r_CIP_m"])
    solvent_separated_cutoff_m = float(pair_basins["r_SSIP_m"])
    free_cutoff_m = float(pair_basins["r_free_m"])
    labels = {from_pair_label, to_pair_label}
    if labels == {PairBasin.FREE.value, PairBasin.SOLVENT_SEPARATED_ION_PAIR.value}:
        return solvent_separated_cutoff_m, free_cutoff_m
    if labels == {
        PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
        PairBasin.CONTACT_ION_PAIR.value,
    }:
        return contact_cutoff_m, solvent_separated_cutoff_m
    if labels == {PairBasin.SOLVENT_SEPARATED_ION_PAIR.value, "addSSIP"}:
        return contact_cutoff_m, solvent_separated_cutoff_m
    raise ValueError(f"unsupported transition labels {from_pair_label}, {to_pair_label}")


def _gauss_legendre_interval(
    lower_bound: float,
    upper_bound: float,
    order: int,
) -> tuple[Array, Array]:
    if order < 1:
        raise ValueError("quadrature order must be positive")
    if not lower_bound < upper_bound:
        raise ValueError("quadrature bounds must be increasing")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    midpoint = (upper_bound + lower_bound) / 2.0
    half_width = (upper_bound - lower_bound) / 2.0
    return midpoint + half_width * nodes, half_width * weights


def _configuration_with_pair_distance(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    pair_distance_m: float,
) -> SiteConfiguration:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    positions = np.asarray(configuration.positions_m, dtype=float).copy()
    cation_position = positions[cation_index].copy()
    target_anion_position = cation_position + np.asarray(
        [pair_distance_m, 0.0, 0.0],
        dtype=float,
    )
    anion_center_position = np.mean(positions[np.asarray(anion_indices, dtype=int)], axis=0)
    shift = target_anion_position - anion_center_position
    for site_index in anion_indices:
        positions[site_index] += shift
    return SiteConfiguration(
        species_names=configuration.species_names,
        molecule_ids=np.asarray(configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=positions,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _pair_distance_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    positions = np.asarray(configuration.positions_m, dtype=float)
    anion_center_position = np.mean(positions[np.asarray(anion_indices, dtype=int)], axis=0)
    displacement = (
        anion_center_position
        - np.asarray(configuration.positions_m[cation_index], dtype=float)
    )
    distance_m = _positive_float(float(np.linalg.norm(displacement)), "pair distance")
    unit_vector = displacement / distance_m
    gradient = np.zeros(len(configuration.species_names) * CARTESIAN_DIMENSION, dtype=float)
    cation_start = cation_index * CARTESIAN_DIMENSION
    gradient[cation_start : cation_start + CARTESIAN_DIMENSION] = -unit_vector
    anion_weight = 1.0 / float(len(anion_indices))
    for anion_index in anion_indices:
        anion_start = anion_index * CARTESIAN_DIMENSION
        gradient[anion_start : anion_start + CARTESIAN_DIMENSION] = (
            anion_weight * unit_vector
        )
    return gradient


def _li_anion_distance_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    positions = np.asarray(configuration.positions_m, dtype=float)
    anion_center_position = np.mean(
        positions[np.asarray(anion_indices, dtype=int)],
        axis=0,
    )
    displacement = anion_center_position - positions[cation_index]
    return _positive_float(float(np.linalg.norm(displacement)), "Li_anion_distance_m")


def _minimum_image_vector_m(
    first_position_m: Array,
    second_position_m: Array,
    box_lengths_m: Array,
) -> Array:
    displacement_m = np.asarray(second_position_m, dtype=float) - np.asarray(
        first_position_m,
        dtype=float,
    )
    box_lengths = np.asarray(box_lengths_m, dtype=float)
    if np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths_m must be positive")
    return displacement_m - box_lengths * np.round(displacement_m / box_lengths)


def _state_key_for_configuration(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
    pair_label: str,
) -> tuple[str, ...]:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_index = _first_role_index(records, configuration, SpeciesRole.ANION)
    orientation = assign_orientation_basin(records, configuration, cation_index, anion_index)
    environment_label = _environment_label(records, configuration, mixture)
    return (
        pair_label,
        _lithium_shell_label(records, configuration),
        _ligand_state_label(records, configuration),
        _anion_feature_label(records, configuration),
        orientation.value,
        _cluster_label(pair_label),
        "partner_a",
        "carrier_identity_a",
        "hop_source",
        "cage_released",
        environment_label,
        "atmosphere_released",
    )


def _state_key_from_reduced_coordinates(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
    pair_label: str,
    coordinate_values: dict[str, float],
) -> tuple[str, ...]:
    _ = mixture
    return (
        pair_label,
        _lithium_shell_label_from_coordinates(records, coordinate_values),
        _ligand_state_label_from_coordinates(records, coordinate_values),
        _anion_feature_label(records, configuration),
        _orientation_label_from_coordinates(records, coordinate_values),
        _cluster_label_from_coordinates(records, pair_label, coordinate_values),
        _partner_label_from_coordinates(records, coordinate_values),
        _identity_label_from_coordinates(records, coordinate_values),
        _structural_hop_label_from_coordinates(records, coordinate_values),
        _cage_label_from_coordinates(records, coordinate_values),
        _environment_label_from_coordinates(records, coordinate_values),
        _atmosphere_label_from_coordinates(records, coordinate_values),
    )


def sparse_state_key_from_reduced_observation(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
    pair_label: str,
    active_anion_component_name: str,
    coordinate_values: dict[str, float],
) -> tuple[str, ...]:
    """Assign an observed reduced configuration to the active sparse state key."""

    _validate_reduced_observation_coordinates(records, coordinate_values)
    state_key = _state_key_from_reduced_coordinates(
        records,
        configuration,
        mixture,
        pair_label,
        coordinate_values,
    )
    return _state_key_with_active_anion(state_key, active_anion_component_name)


def _validate_reduced_observation_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> None:
    required_coordinates = _declared_reduced_coordinates(records)
    missing_coordinates = tuple(
        coordinate.value
        for coordinate in required_coordinates
        if coordinate.value not in coordinate_values
    )
    if missing_coordinates:
        raise KeyError(
            "reduced observation missing active state coordinates "
            f"{missing_coordinates}"
        )
    for coordinate in required_coordinates:
        value = float(coordinate_values[coordinate.value])
        if not np.isfinite(value):
            raise ValueError(f"reduced coordinate {coordinate.value} must be finite")


def _lithium_shell_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    ligand_coordination = float(
        coordinate_values[ReducedCoordinate.LI_LIGAND_COORDINATION.value]
    )
    anion_coordination = float(
        coordinate_values[ReducedCoordinate.LI_ANION_COORDINATION.value]
    )
    solvent_coordination = float(
        coordinate_values[ReducedCoordinate.LI_SOLVENT_COORDINATION.value]
    )
    ligand_cutoff = _coordination_cutoff(records, "Li_ligand")
    anion_cutoff = _coordination_cutoff(records, "Li_anion")
    if ligand_coordination >= ligand_cutoff and anion_coordination >= anion_cutoff:
        return "mixed_ligand_anion"
    if ligand_coordination >= ligand_cutoff:
        return "neutral_ligand_bound"
    if anion_coordination >= anion_cutoff:
        return "anion_coordinated"
    if solvent_coordination > 0.0:
        return "solvent_only"
    return "solvent_poor"


def _ligand_state_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    ligand_coordination = float(
        coordinate_values[ReducedCoordinate.LI_LIGAND_COORDINATION.value]
    )
    if ligand_coordination < _coordination_cutoff(records, "Li_ligand"):
        return "none"
    return "monodentate"


def _orientation_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    orientation_value = float(coordinate_values[ReducedCoordinate.ANION_ORIENTATION.value])
    bins = records.basis_record["orientation_bins"]
    if orientation_value > float(bins["radial_min"]):
        return "radial"
    if abs(orientation_value) <= float(bins["tangential_abs_max"]):
        return "tangential"
    if orientation_value < float(bins["bridging_max"]):
        return "bridging"
    return "free_rotating"


def _environment_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    environment_bins = records.basis_record["environment_bins"]
    packing_bin = _threshold_bin_index(
        float(coordinate_values[ReducedCoordinate.LOCAL_PACKING_FRACTION.value]),
        environment_bins["packing_fraction"],
    )
    ionic_strength_bin = _threshold_bin_index(
        float(coordinate_values[ReducedCoordinate.LOCAL_IONIC_STRENGTH.value]),
        environment_bins["ionic_strength_mol_m3"],
    )
    dielectric_bin = _threshold_bin_index(
        float(coordinate_values[ReducedCoordinate.LOCAL_DIELECTRIC.value]),
        environment_bins["dielectric"],
    )
    viscosity_bin = _threshold_bin_index(
        float(coordinate_values[ReducedCoordinate.LOCAL_VISCOSITY.value]),
        environment_bins["viscosity_Pa_s"],
    )
    return (
        f"packing_{packing_bin}:ionic_{ionic_strength_bin}:"
        f"dielectric_{dielectric_bin}:viscosity_{viscosity_bin}"
    )


def _cluster_label_from_coordinates(
    records: PhysicalLibraryRecords,
    pair_label: str,
    coordinate_values: dict[str, float],
) -> str:
    cluster_value = float(coordinate_values[ReducedCoordinate.CLUSTER_COORDINATE.value])
    return _label_from_unit_interval_bins(
        records,
        cluster_value,
        (_cluster_label(pair_label), "aggregate", "bridge_network"),
    )


def _partner_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    partner_value = float(
        coordinate_values[ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value]
    )
    return _label_from_unit_interval_bins(
        records,
        partner_value,
        ("partner_a", "partner_switching", "partner_b"),
    )


def _identity_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    identity_value = float(coordinate_values[ReducedCoordinate.IDENTITY_COORDINATE.value])
    return _label_from_unit_interval_bins(
        records,
        identity_value,
        (
            "carrier_identity_a",
            "carrier_identity_transition",
            "carrier_identity_b",
        ),
    )


def _structural_hop_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    hop_value = float(coordinate_values[ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value])
    return _label_from_unit_interval_bins(
        records,
        hop_value,
        ("hop_source", "hop_transition", "hop_target"),
    )


def _cage_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    cage_value = float(coordinate_values[ReducedCoordinate.CAGE_COORDINATE.value])
    return _label_from_unit_interval_bins(
        records,
        cage_value,
        ("cage_released", "cage_boundary", "cage_captured"),
    )


def _atmosphere_label_from_coordinates(
    records: PhysicalLibraryRecords,
    coordinate_values: dict[str, float],
) -> str:
    atmosphere_value = float(
        coordinate_values[ReducedCoordinate.ATMOSPHERE_POLARIZATION.value]
    )
    return _label_from_unit_interval_bins(
        records,
        atmosphere_value,
        ("atmosphere_released", "atmosphere_boundary", "atmosphere_captured"),
    )


def _unit_interval_thresholds(records: PhysicalLibraryRecords) -> list[float]:
    return list(records.basis_record["unit_interval_state_bins"])


def _label_from_unit_interval_bins(
    records: PhysicalLibraryRecords,
    coordinate_value: float,
    labels: tuple[str, ...],
) -> str:
    bin_index = _threshold_bin_index(coordinate_value, _unit_interval_thresholds(records))
    if bin_index >= len(labels):
        raise ValueError("unit interval coordinate produced unsupported state bin")
    return labels[bin_index]


def _coordination_cutoff(records: PhysicalLibraryRecords, switch_name: str) -> float:
    return float(records.basis_record["coordination_switches"][switch_name]["cutoff"])


def _threshold_bin_index(value: float, thresholds: list[float]) -> int:
    bin_index = 0
    for threshold in thresholds:
        if value >= float(threshold):
            bin_index += 1
    return bin_index


def _lithium_shell_label(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> str:
    roles = tuple(_species_role(records, name) for name in configuration.species_names)
    if SpeciesRole.ADDITIVE in roles:
        return "neutral_ligand_bound"
    if SpeciesRole.ANION in roles:
        return "anion_coordinated"
    return "solvent_only"


def _ligand_state_label(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> str:
    for species_name in configuration.species_names:
        if _species_role(records, species_name) == SpeciesRole.ADDITIVE:
            return "monodentate"
    return "none"


def _anion_feature_label(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> str:
    anion_index = _first_role_index(records, configuration, SpeciesRole.ANION)
    anion_record = records.species_records[configuration.species_names[anion_index]]
    anion_site = anion_record["sites"][int(configuration.site_ids[anion_index])]
    cloud_radius_m = float(anion_site["charge_cloud_radius_m"])
    threshold_m = float(
        records.basis_record["anion_feature_bins"]["charge_cloud_localized_threshold_m"]
    )
    if cloud_radius_m < threshold_m:
        return "anion_localized"
    if cloud_radius_m >= threshold_m:
        return "anion_delocalized"
    raise ValueError("anion charge cloud radius is not classifiable")


def _cluster_label(pair_label: str) -> str:
    if pair_label == PairBasin.FREE.value:
        return "free_ions"
    if pair_label == "addSSIP":
        return "Li_ligand_anion"
    if pair_label == PairBasin.CONTACT_ION_PAIR.value:
        return "LiA"
    if pair_label == PairBasin.SOLVENT_SEPARATED_ION_PAIR.value:
        return "LiA"
    raise ValueError(f"unsupported pair label {pair_label}")


def _environment_label(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
) -> str:
    _positive_float(mixture.dielectric_constant, "mixture.dielectric_constant")
    packing = compute_local_packing_fraction(records, configuration)
    return f"packing_{packing:.6e}"


def _reference_offset_m(
    records: PhysicalLibraryRecords,
    species_name: str,
    molecule_id: int,
) -> Array:
    role = _species_role(records, species_name)
    geometry = records.basis_record["template_geometry"]
    if role == SpeciesRole.CATION:
        return np.zeros(CARTESIAN_DIMENSION, dtype=float)
    if role == SpeciesRole.ANION:
        return np.asarray([float(geometry["anion_reference_offset_m"]), 0.0, 0.0])
    if role in (SpeciesRole.SOLVENT, SpeciesRole.ADDITIVE):
        neutral_spacing_m = float(geometry["neutral_shell_spacing_m"])
        return np.asarray([0.0, (molecule_id + 1) * neutral_spacing_m, 0.0])
    raise ValueError(f"unsupported species role {role.value}")


def _first_role_index(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
) -> int:
    for site_index, species_name in enumerate(configuration.species_names):
        if _species_role(records, species_name) == role:
            return site_index
    raise ValueError(f"configuration has no species with role {role.value}")


def _first_molecule_indices_with_role(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
) -> tuple[int, ...]:
    first_site_index = _first_role_index(records, configuration, role)
    molecule_id = int(configuration.molecule_ids[first_site_index])
    species_name = configuration.species_names[first_site_index]
    indices = tuple(
        site_index
        for site_index, current_species_name in enumerate(configuration.species_names)
        if current_species_name == species_name
        and int(configuration.molecule_ids[site_index]) == molecule_id
    )
    if not indices:
        raise ValueError(f"configuration has no molecule indices for role {role.value}")
    return indices


def _first_molecule_indices_for_species(
    configuration: SiteConfiguration,
    species_name: str,
) -> tuple[int, ...]:
    molecule_id = None
    for site_index, current_species_name in enumerate(configuration.species_names):
        if current_species_name == species_name:
            molecule_id = int(configuration.molecule_ids[site_index])
            break
    if molecule_id is None:
        raise ValueError(f"configuration has no species {species_name}")
    indices = tuple(
        site_index
        for site_index, current_species_name in enumerate(configuration.species_names)
        if current_species_name == species_name
        and int(configuration.molecule_ids[site_index]) == molecule_id
    )
    if not indices:
        raise ValueError(f"configuration has no molecule indices for species {species_name}")
    return indices


def _species_role(records: PhysicalLibraryRecords, species_name: str) -> SpeciesRole:
    role_name = str(records.species_records[species_name]["role"])
    for role in SpeciesRole:
        if role.value == role_name:
            return role
    raise ValueError(f"unsupported species role {role_name}")


def _validate_numerical_options(numerical_options: NumericalOptions) -> None:
    box_lengths = np.asarray(numerical_options.reference_box_lengths_m, dtype=float)
    if box_lengths.shape != (CARTESIAN_DIMENSION,) or np.any(box_lengths <= 0.0):
        raise ValueError("reference_box_lengths_m must have shape (3,) and be positive")
    _positive_float(numerical_options.volume_m3, "volume_m3")
    if numerical_options.state_quadrature_order < 1:
        raise ValueError("state_quadrature_order must be positive")
    if numerical_options.transition_grid_count < 3:
        raise ValueError("transition_grid_count must be at least 3")


def _positive_float(value: float, label: str) -> float:
    numeric_value = float(value)
    if numeric_value <= 0.0 or not np.isfinite(numeric_value):
        raise ValueError(f"{label} must be positive and finite")
    return numeric_value


def _nonnegative_float(value: float, label: str) -> float:
    numeric_value = float(value)
    if numeric_value < 0.0 or not np.isfinite(numeric_value):
        raise ValueError(f"{label} must be nonnegative and finite")
    return numeric_value


def _finite_float(value: float, label: str) -> float:
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{label} must be finite")
    return numeric_value
