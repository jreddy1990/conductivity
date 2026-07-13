"""Recipe and physical-library builders for projected generator inputs."""

from __future__ import annotations

import copy
import os
import pickle
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from pathlib import Path
from threading import RLock, get_ident

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
    molecule_center_of_mass_m,
    molecule_site_indices_and_mass_fractions,
    anion_internal_charge_separation_factor,
    build_physical_objects,
    compute_atmosphere_resistance_diagnostics,
    compute_charge_polarization_m,
    compute_born_energy_J_mol,
    compute_local_packing_fraction,
    compute_resistance_component_diagnostics,
)
from conductivity.physical_library.projected_analytical_conductivity import (
    ProjectedConductivityResult,
    ProjectedGeneratorInput,
    StateTransportOwnershipBasis,
    TransportOwnership,
    _compute_projected_analytical_conductivity_from_input,
    compute_restricted_log_partition_values,
    primitive_prediction_readiness_as_effect_attribution,
    symmetric_psd_pseudoinverse,
)
from conductivity.physical_library.library_io import (
    PhysicalLibraryRecords,
    RecipeBuildResult,
    RecipeComponentLoading,
    build_recipe_library_context,
)
from utils.config_loader import content_hash_files
from utils.typed_sqlite import sha256_text
from conductivity.physical_library.reduced_generator import (
    build_projected_generator_input,
)
from conductivity.physical_library.reduced_generator import (
    ReducedGeneratorSpecification,
)
from conductivity.physical_library.transition_surface_builder import (
    MomentBoundaryValueInput,
    OneDimensionalCommittorInput,
    solve_one_dimensional_committor,
)
from conductivity.physical_library.transition_moment_bvp import (
    EndpointTransportMomentInput,
    build_endpoint_transport_moments,
)

Array = np.ndarray
SUMMATION_ROUNDOFF_EPSILON_FACTOR = 64.0  # Numerical epsilon factor for accumulated sparse-sum roundoff.
CHAIN_OUTER_CENTER_OFFSET_MULTIPLIER = 3.0 / 2.0  # Four-center chain spacing places outer centers at three half-bonds.
CHAIN_INNER_CENTER_OFFSET_MULTIPLIER = 1.0 / 2.0  # Four-center chain spacing places inner centers at one half-bond.
__all__ = ("NumericalOptions", "compute_conductivity_from_recipe")
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
    "local_ionic_strength_packing_coupling",
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


class InfeasibleStateGeometryError(ValueError):
    """A candidate state has no physically valid local geometry."""


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
    BOUNDED_INTERNAL_POLARIZATION = "bounded_internal_polarization"


PERPENDICULAR_AXIS_ALIGNMENT_LIMIT = 0.99  # Numerical sentinel: switch helper axis before near-collinear cross construction.
FINITE_DIFFERENCE_STEP_M = (
    1.0e-12  # Site-coordinate gradient step for reduced scalar coordinates.
)
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
    label: str
    family: MemoryCoordinateFamily
    records: PhysicalLibraryRecords
    required_roles: tuple[SpeciesRole, ...]
    required_species_names: tuple[str, ...]
    value_function: Callable[[PhysicalLibraryRecords, SiteConfiguration], float]
    gradient_function: Callable[[PhysicalLibraryRecords, SiteConfiguration], Array]


@dataclass(frozen=True)
class TransitionEdge:
    from_state_index: int
    to_state_index: int
    family: str


@dataclass(frozen=True)
class ChargedCenterPairCovarianceEntry:
    state_label: str
    first_center_label: str
    second_center_label: str
    first_charge_number: float
    second_charge_number: float
    center_mobility_m2_s: float
    charge_weighted_covariance_m2_s: float


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
    charged_center_pair_covariance_entries: tuple[ChargedCenterPairCovarianceEntry, ...]
    potential_energy_J_mol: float
    dielectric_constant: float
    viscosity_Pa_s: float
    ionic_strength_mol_m3: float
    local_packing_fraction: float
    resistance_stokes_trace_kg_s: float
    resistance_free_volume_trace_kg_s: float
    resistance_charge_cloud_trace_kg_s: float
    resistance_atmosphere_trace_kg_s: float
    resistance_cage_constraint_trace_kg_s: float
    resistance_ligand_shell_obstruction_trace_kg_s: float
    resistance_aggregate_constraint_trace_kg_s: float
    resistance_bridge_constraint_trace_kg_s: float
    resistance_orientation_denticity_trace_kg_s: float
    resistance_total_trace_kg_s: float
    atmosphere_electrophoretic_trace_kg_s: float
    atmosphere_relaxation_trace_kg_s: float
    atmosphere_cation_diagonal_trace_kg_s: float
    atmosphere_anion_diagonal_trace_kg_s: float
    atmosphere_cation_anion_cross_trace_kg_s: float
    atmosphere_mean_charge_cloud_form_factor: float
    atmosphere_mean_state_geometry_form_factor: float
    atmosphere_minimum_separation_over_debye_length: float
    atmosphere_debye_falkenhagen_time_s: float


@dataclass(frozen=True)
class MolecularChargeCenter:
    label: str
    formal_charge_number: float
    site_indices: tuple[int, ...]
    center_of_mass_weights: tuple[float, ...]


@dataclass
class StateQuadratureGroup:
    state_key: tuple[str, ...]
    configurations: list[SiteConfiguration]
    coordinate_values: list[dict[str, float]]
    local_fields: list[PhysicalLocalFields]
    weights: list[float]


@dataclass(frozen=True)
class StateQuadratureNode:
    node_indices: tuple[int, ...]
    active_coordinates: frozenset[ReducedCoordinate]


_CONDUCTIVITY_RESULT_CACHE: dict[tuple, ProjectedConductivityResult] = {}
_CONDUCTIVITY_RESULT_CACHE_LOCK = RLock()


def compute_conductivity_from_recipe(
    recipe: Path,
    library_root: Path,
    numerical_options: NumericalOptions,
) -> ProjectedConductivityResult:
    cache_key = _conductivity_result_cache_key(
        recipe,
        library_root,
        numerical_options,
    )
    with _CONDUCTIVITY_RESULT_CACHE_LOCK:
        cached_result = _CONDUCTIVITY_RESULT_CACHE.get(cache_key)
    if cached_result is not None:
        _validate_cached_conductivity_result(cached_result)
        return copy.deepcopy(cached_result)
    persistent_cache_path = _persistent_conductivity_cache_path(cache_key)
    if persistent_cache_path.exists():
        with persistent_cache_path.open("rb") as cache_file:
            persistent_result = pickle.load(cache_file)
        if not isinstance(persistent_result, ProjectedConductivityResult):
            raise TypeError("persistent conductivity cache has wrong result type")
        _validate_cached_conductivity_result(persistent_result)
        with _CONDUCTIVITY_RESULT_CACHE_LOCK:
            _CONDUCTIVITY_RESULT_CACHE[cache_key] = copy.deepcopy(persistent_result)
        return persistent_result
    conductivity_result = _compute_conductivity_from_recipe_uncached(
        recipe,
        library_root,
        numerical_options,
    )
    with _CONDUCTIVITY_RESULT_CACHE_LOCK:
        _CONDUCTIVITY_RESULT_CACHE[cache_key] = copy.deepcopy(conductivity_result)
    persistent_cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache_path = persistent_cache_path.with_suffix(
        f".{os.getpid()}.{get_ident()}.tmp"
    )
    with temporary_cache_path.open("wb") as cache_file:
        pickle.dump(conductivity_result, cache_file)
    temporary_cache_path.replace(persistent_cache_path)
    return conductivity_result


def _validate_cached_conductivity_result(
    conductivity_result: ProjectedConductivityResult,
) -> None:
    attribution = conductivity_result.effect_attribution
    if attribution["primitive_prediction_readiness_status"] != "complete":
        raise ValueError("cached conductivity result is not primitive-ready")
    if attribution["basis_refinement_convergence_status"] != "converged":
        raise ValueError("cached conductivity result has unconverged basis")
    owner_table = tuple(attribution["state_primitive_owner_table"])
    state_count = len(conductivity_result.state_concentrations_mol_m3)
    if len(owner_table) != state_count:
        raise ValueError("cached conductivity owner table length mismatch")
    ownership_records = tuple(attribution["transport_ownership_state_tensors"])
    if len(ownership_records) != state_count:
        raise ValueError("cached conductivity ownership tensor length mismatch")
    _validate_state_charge_mobility_invariants(conductivity_result)


def _conductivity_result_cache_key(
    recipe: Path,
    library_root: Path,
    numerical_options: NumericalOptions,
) -> tuple:
    library_paths = tuple(
        sorted(
            path
            for path in library_root.rglob("*")
            if path.suffix in {".py", ".yaml"}
        )
    )
    if not library_paths:
        raise ValueError("physical library contains no YAML records")
    return (
        content_hash_files(recipe, *library_paths),
        tuple(
            float(length_m)
            for length_m in np.asarray(
                numerical_options.reference_box_lengths_m,
                dtype=float,
            )
        ),
        float(numerical_options.volume_m3),
        int(numerical_options.state_quadrature_order),
        int(numerical_options.transition_grid_count),
    )


def _persistent_conductivity_cache_path(cache_key: tuple) -> Path:
    cache_digest = sha256_text(repr(cache_key))
    return Path(".conductivity_cache") / f"projected_conductivity_{cache_digest}.pkl"


def _compute_conductivity_from_recipe_uncached(
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
    transition_edges = finite_generator_transition_edges(
        records,
        state_quadratures,
        recipe_context.temperature_K,
    )
    _validate_transport_graph_closure(
        records=records,
        state_quadratures=state_quadratures,
        declared_edges=enumerate_transition_edges(state_quadratures, records),
        retained_edges=transition_edges,
    )
    selected_memory_coordinates = _selected_memory_coordinates(
        records,
        template_configuration,
        transition_edges,
    )
    memory_gradient_functions = tuple(
        _bind_memory_gradient(memory_coordinate)
        for memory_coordinate in selected_memory_coordinates
    )
    state_quadratures = _state_quadratures_with_transport_ownership_bases(
        records=records,
        state_quadratures=state_quadratures,
        transition_edges=transition_edges,
        selected_memory_coordinates=selected_memory_coordinates,
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
            state_memory_value_matrix=_state_family_memory_value_matrix(
                state_quadratures,
                transition_edges,
            ),
            total_component_concentrations_mol_m3=component_concentrations,
            temperature_K=recipe_context.temperature_K,
            volume_m3=numerical_options.volume_m3,
        )
    )
    generator_input = build_projected_generator_input(
        _normalize_potential_energy_reference(reduced_specification),
    )
    conductivity_result = _compute_projected_analytical_conductivity_from_input(
        generator_input
    )
    _annotate_partition_measure_diagnostics(
        conductivity_result,
        records,
        state_quadratures,
        generator_input,
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
    _validate_state_charge_mobility_invariants(conductivity_result)
    _annotate_component_mass_balance_diagnostics(
        conductivity_result,
        projected_components,
        state_quadratures,
    )
    _annotate_mori_mode_diagnostics(
        conductivity_result=conductivity_result,
        records=records,
        template_configuration=template_configuration,
        state_quadratures=state_quadratures,
        transition_edges=transition_edges,
    )
    _annotate_state_primitive_owner_table(
        conductivity_result=conductivity_result,
        state_quadratures=state_quadratures,
        transition_edges=transition_edges,
    )
    _validate_state_transport_owner_closure(
        conductivity_result=conductivity_result,
        state_quadratures=state_quadratures,
    )
    return conductivity_result


def _annotate_partition_measure_diagnostics(
    conductivity_result: ProjectedConductivityResult,
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    generator_input: ProjectedGeneratorInput,
) -> None:
    log_partitions = compute_restricted_log_partition_values(
        generator_input.potential_energy_J_mol,
        generator_input.basin_quadrature_points,
        generator_input.basin_quadrature_weights,
        generator_input.basin_energy_references_J_mol,
        generator_input.temperature_K,
    )
    relative_log_partitions = log_partitions - float(np.max(log_partitions))
    maximum_span = float(
        records.association_record["association_residual"][
            "maximum_relative_log_partition_span"
        ]
    )
    observed_span = float(np.max(log_partitions) - np.min(log_partitions))
    if observed_span > maximum_span:
        raise ValueError(
            "STATE_RELATIVE_LOG_PARTITION_SPAN_EXCEEDED: "
            f"observed={observed_span:.9g}; configured={maximum_span:.9g}"
        )
    thermal_energy_J_mol = R * float(generator_input.temperature_K)
    state_free_energies_over_RT = np.asarray(
        [
            _state_feature_sum(
                records.association_record["association_residual"],
                _state_key_from_label(state.label),
                "population_operator_missing",
            )
            / thermal_energy_J_mol
            for state in state_quadratures
        ],
        dtype=float,
    )
    relative_energy_ranges_over_RT = np.asarray(
        [
            (
                max(
                    float(generator_input.potential_energy_J_mol(point))
                    for point in points
                )
                - float(reference_J_mol)
            )
            / thermal_energy_J_mol
            for points, reference_J_mol in zip(
                generator_input.basin_quadrature_points,
                generator_input.basin_energy_references_J_mol,
                strict=True,
            )
        ],
        dtype=float,
    )
    conductivity_result.effect_attribution.update(
        {
            "state_relative_log_partition_values": relative_log_partitions,
            "state_free_energies_over_RT": state_free_energies_over_RT,
            "state_relative_internal_energy_ranges_over_RT": (
                relative_energy_ranges_over_RT
            ),
            "relative_log_partition_span": observed_span,
            "maximum_relative_log_partition_span": maximum_span,
        }
    )


def _validate_state_charge_mobility_invariants(
    conductivity_result: ProjectedConductivityResult,
) -> None:
    attribution = conductivity_result.effect_attribution
    charge_diffusivities = np.asarray(
        attribution["state_charged_center_D_Q_zDz_m2_s"], dtype=float
    )
    lithium_diffusivities = np.asarray(
        attribution["state_charged_center_D_Li_m2_s"], dtype=float
    )
    anion_diffusivities = np.asarray(
        attribution["state_charged_center_D_anion_m2_s"], dtype=float
    )
    lithium_anion_covariances = np.asarray(
        attribution["state_charged_center_D_Li_anion_m2_s"], dtype=float
    )
    state_labels = tuple(str(label) for label in attribution["state_labels"])
    expected_shape = (len(state_labels),)
    diagnostic_arrays = (
        charge_diffusivities,
        lithium_diffusivities,
        anion_diffusivities,
        lithium_anion_covariances,
    )
    if any(array.shape != expected_shape for array in diagnostic_arrays):
        raise ValueError("state charged-center diagnostic length mismatch")
    if any(np.any(~np.isfinite(array)) for array in diagnostic_arrays):
        raise ValueError("state charged-center diagnostics must be finite")
    if np.any(charge_diffusivities < 0.0):
        invalid_state_indices = tuple(
            int(index) for index in np.flatnonzero(charge_diffusivities < 0.0)
        )
        raise ValueError(
            "state charge diffusivity z^T D z must be nonnegative for states "
            f"{invalid_state_indices}"
        )
    reconstructed_charge_diffusivities = (
        lithium_diffusivities
        + anion_diffusivities
        - 2.0 * lithium_anion_covariances
    )
    comparison_scale = np.maximum(
        np.maximum(
            np.abs(charge_diffusivities),
            np.abs(reconstructed_charge_diffusivities),
        ),
        np.finfo(float).tiny,
    )
    invariant_residuals = np.abs(
        charge_diffusivities - reconstructed_charge_diffusivities
    )
    invariant_tolerance = (
        SUMMATION_ROUNDOFF_EPSILON_FACTOR
        * np.finfo(float).eps
        * comparison_scale
    )
    invalid_state_indices = tuple(
        int(index)
        for index in np.flatnonzero(invariant_residuals > invariant_tolerance)
    )
    if invalid_state_indices:
        invalid_state_labels = tuple(
            state_labels[index] for index in invalid_state_indices
        )
        raise ValueError(
            "state charged-center mobility violates "
            "D_Q = D_Li + D_anion - 2 D_Li_anion for states "
            f"{invalid_state_labels}"
        )


def _annotate_state_primitive_owner_table(
    conductivity_result: ProjectedConductivityResult,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    transition_edges: tuple[TransitionEdge, ...],
) -> None:
    attribution = conductivity_result.effect_attribution
    state_count = len(state_quadratures)
    concentrations = np.asarray(
        conductivity_result.state_concentrations_mol_m3,
        dtype=float,
    )
    self_current_tensors = np.asarray(
        conductivity_result.self_current_tensors_D_self_i_m2_s,
        dtype=float,
    )
    ownership_records = tuple(attribution["transport_ownership_state_tensors"])
    mori_mode_ledger = tuple(attribution["mori_mode_ledger"])
    if concentrations.size != state_count or self_current_tensors.shape[0] != state_count:
        raise ValueError("state primitive owner table input length mismatch")
    incident_transition_indices = tuple(
        tuple(
            edge_index
            for edge_index, edge in enumerate(transition_edges)
            if state_index in (edge.from_state_index, edge.to_state_index)
        )
        for state_index in range(state_count)
    )
    state_primitive_owner_table = tuple(
        {
            "state_index": state_index,
            "state_label": state_quadrature.label,
            "component_stoichiometry": np.asarray(
                state_quadrature.stoichiometry,
                dtype=float,
            ).copy(),
            "concentration_mol_m3": float(concentrations[state_index]),
            "owner_classes": _state_primitive_owner_classes(
                ownership_records[state_index]
            ),
            "D_Q_zDz_m2_s": float(
                attribution["state_charged_center_D_Q_zDz_m2_s"][state_index]
            ),
            "D_Li_m2_s": float(
                attribution["state_charged_center_D_Li_m2_s"][state_index]
            ),
            "D_anion_m2_s": float(
                attribution["state_charged_center_D_anion_m2_s"][state_index]
            ),
            "D_Li_anion_m2_s": float(
                attribution["state_charged_center_D_Li_anion_m2_s"][state_index]
            ),
            "c_i_trace_D_self_mol_m_s": float(
                concentrations[state_index]
                * np.trace(self_current_tensors[state_index])
            ),
            "R_hydro_RPY_trace_kg_s": float(
                attribution["state_resistance_stokes_traces_kg_s"][state_index]
            ),
            "R_shape_trace_kg_s": 0.0,
            "R_cloud_short_k_trace_kg_s": float(
                attribution["state_resistance_charge_cloud_traces_kg_s"][state_index]
            ),
            "R_atmosphere_electrophoretic_trace_kg_s": float(
                attribution["state_atmosphere_electrophoretic_traces_kg_s"][
                    state_index
                ]
            ),
            "R_atmosphere_relaxation_trace_kg_s": float(
                attribution["state_atmosphere_relaxation_traces_kg_s"][state_index]
            ),
            "R_atmosphere_cross_trace_kg_s": float(
                attribution["state_atmosphere_cation_anion_cross_traces_kg_s"][
                    state_index
                ]
            ),
            "R_free_volume_trace_kg_s": float(
                attribution["state_resistance_free_volume_traces_kg_s"][state_index]
            ),
            "R_cage_constraint_trace_kg_s": float(
                attribution["state_resistance_cage_constraint_traces_kg_s"][
                    state_index
                ]
            ),
            "R_ligand_shell_obstruction_trace_kg_s": float(
                attribution["state_resistance_ligand_shell_obstruction_traces_kg_s"][
                    state_index
                ]
            ),
            "R_aggregate_constraint_trace_kg_s": float(
                attribution["state_resistance_aggregate_constraint_traces_kg_s"][
                    state_index
                ]
            ),
            "R_bridge_constraint_trace_kg_s": float(
                attribution["state_resistance_bridge_constraint_traces_kg_s"][
                    state_index
                ]
            ),
            "R_orientation_denticity_trace_kg_s": float(
                attribution["state_resistance_orientation_denticity_traces_kg_s"][
                    state_index
                ]
            ),
            "mori_mode_labels": tuple(
                str(mode_record["mode_label"])
                for mode_record in mori_mode_ledger
                if state_quadrature.label in mode_record["state_support"]
            ),
            "incident_transition_edge_indices": incident_transition_indices[state_index],
        }
        for state_index, state_quadrature in enumerate(state_quadratures)
    )
    attribution["state_primitive_owner_table"] = state_primitive_owner_table


def _state_primitive_owner_classes(ownership_record: dict) -> tuple[str, ...]:
    owner_tensor_fields = (
        (TransportOwnership.DC_SELF.value, "D_Q_dc_self"),
        (TransportOwnership.TRANSITION_DISPLACEMENT.value, "D_Q_transition_owned"),
        (TransportOwnership.BOUNDED_MEMORY.value, "D_Q_bounded_memory"),
        (TransportOwnership.DIAGNOSTIC.value, "D_Q_diagnostic"),
    )
    return tuple(
        owner_name
        for owner_name, tensor_field in owner_tensor_fields
        if float(np.linalg.norm(np.asarray(ownership_record[tensor_field]), ord=2)) > 0.0
    )


def _state_quadratures_with_transport_ownership_bases(
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    transition_edges: tuple[TransitionEdge, ...],
    selected_memory_coordinates: tuple[MemoryCoordinate, ...],
) -> tuple[PhysicalStateQuadrature, ...]:
    incident_edges_by_state: list[list[tuple[int, TransitionEdge]]] = [
        [] for _state_quadrature in state_quadratures
    ]
    for transition_edge_index, transition_edge in enumerate(transition_edges):
        transition_record = _transition_family_record(records, transition_edge.family)
        if (
            _transition_transport_ownership(transition_record)
            is not TransportOwnership.TRANSITION_DISPLACEMENT
        ):
            continue
        edge_record = (transition_edge_index, transition_edge)
        incident_edges_by_state[transition_edge.from_state_index].append(edge_record)
        incident_edges_by_state[transition_edge.to_state_index].append(edge_record)
    return tuple(
        replace(
            state_quadrature,
            transport_ownership_bases=tuple(
                _transport_ownership_basis_for_state_point(
                    records=records,
                    state_key=_state_key_from_label(state_quadrature.label),
                    configuration=configuration,
                    incident_edges=incident_edges_by_state[state_index],
                    selected_memory_coordinates=selected_memory_coordinates,
                )
                for configuration in state_quadrature.configurations
            ),
        )
        for state_index, state_quadrature in enumerate(state_quadratures)
    )


def _transport_ownership_basis_for_state_point(
    records: PhysicalLibraryRecords,
    state_key: tuple[str, ...],
    configuration: SiteConfiguration,
    incident_edges: list[tuple[int, TransitionEdge]],
    selected_memory_coordinates: tuple[MemoryCoordinate, ...],
) -> StateTransportOwnershipBasis:
    coordinate_dimension = int(np.asarray(configuration.positions_m).size)
    transition_rows = tuple(
        np.asarray(
            _reaction_coordinate_gradient(
                records,
                configuration,
                _transition_family_record(records, transition_edge.family),
            ),
            dtype=float,
        )
        for _transition_edge_index, transition_edge in incident_edges
    )
    transition_edge_indices = np.asarray(
        [transition_edge_index for transition_edge_index, _edge in incident_edges],
        dtype=int,
    )
    bounded_memory_records = tuple(
        (memory_mode_index, memory_coordinate)
        for memory_mode_index, memory_coordinate in enumerate(
            selected_memory_coordinates
        )
        if _memory_transport_ownership(
            records,
            memory_coordinate.family.value,
        )[0]
        is TransportOwnership.BOUNDED_MEMORY
        and _memory_coordinate_is_supported(memory_coordinate, configuration)
        and _memory_coordinate_is_active_for_state(
            memory_coordinate,
            state_key,
            configuration,
        )
    )
    diagnostic_records = tuple(
        (memory_mode_index, memory_coordinate)
        for memory_mode_index, memory_coordinate in enumerate(
            selected_memory_coordinates
        )
        if _memory_transport_ownership(
            records,
            memory_coordinate.family.value,
        )[0]
        is TransportOwnership.DIAGNOSTIC
        and _memory_coordinate_is_supported(memory_coordinate, configuration)
    )
    bounded_memory_rows = tuple(
        np.asarray(memory_coordinate.gradient_function(records, configuration), dtype=float)
        for _memory_mode_index, memory_coordinate in bounded_memory_records
    )
    diagnostic_rows = tuple(
        np.asarray(memory_coordinate.gradient_function(records, configuration), dtype=float)
        for _memory_mode_index, memory_coordinate in diagnostic_records
    )
    return StateTransportOwnershipBasis(
        transition_displacement_gradients=_ownership_gradient_rows(
            transition_rows,
            coordinate_dimension,
        ),
        transition_edge_indices=transition_edge_indices,
        bounded_memory_gradients=_ownership_gradient_rows(
            bounded_memory_rows,
            coordinate_dimension,
        ),
        bounded_memory_mode_indices=np.asarray(
            [memory_mode_index for memory_mode_index, _coordinate in bounded_memory_records],
            dtype=int,
        ),
        diagnostic_gradients=_ownership_gradient_rows(
            diagnostic_rows,
            coordinate_dimension,
        ),
        diagnostic_source_ids=tuple(
            memory_coordinate.family.value
            for _memory_mode_index, memory_coordinate in diagnostic_records
        ),
    )


def _memory_coordinate_is_active_for_state(
    memory_coordinate: MemoryCoordinate,
    state_key: tuple[str, ...],
    configuration: SiteConfiguration,
) -> bool:
    if memory_coordinate.family is not MemoryCoordinateFamily.PARTNER_RESIDENCE:
        return True
    ligand_state = _state_key_base_value(state_key[STATE_KEY_LIGAND_INDEX])
    shell_state = _state_key_base_value(state_key[STATE_KEY_SHELL_INDEX])
    pair_state = _state_key_base_value(state_key[STATE_KEY_PAIR_INDEX])
    cluster_state = _state_key_base_value(state_key[STATE_KEY_CLUSTER_INDEX])
    coordinating_additive_is_present = any(
        _species_role(memory_coordinate.records, species_name)
        is SpeciesRole.ADDITIVE
        for species_name in configuration.species_names
    )
    pair_has_continuous_residence = (
        pair_state == PairBasin.CONTACT_ION_PAIR.value
        or (
            pair_state == PairBasin.SOLVENT_SEPARATED_ION_PAIR.value
            and not coordinating_additive_is_present
        )
    )
    return (
        ligand_state == "none"
        and shell_state not in {"neutral_ligand_bound", "mixed_ligand_anion"}
        and pair_has_continuous_residence
        and cluster_state == "LiA"
    )


def _ownership_gradient_rows(
    gradient_rows: tuple[Array, ...],
    coordinate_dimension: int,
) -> Array:
    if not gradient_rows:
        return np.empty((0, coordinate_dimension), dtype=float)
    rows = np.asarray(
        [
            _ownership_gradient_row(gradient_row, coordinate_dimension)
            for gradient_row in gradient_rows
        ],
        dtype=float,
    )
    if rows.shape != (len(gradient_rows), coordinate_dimension):
        raise ValueError("transport ownership gradient has wrong physical width")
    return rows


def _ownership_gradient_row(
    gradient_row: Array,
    coordinate_dimension: int,
) -> Array:
    row = np.asarray(gradient_row, dtype=float)
    if row.shape == (1, coordinate_dimension):
        return row[0]
    if row.shape == (coordinate_dimension,):
        return row
    raise ValueError("transport ownership source must provide one physical covector")


def _annotate_mori_mode_diagnostics(
    conductivity_result: ProjectedConductivityResult,
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    transition_edges: tuple[TransitionEdge, ...],
) -> None:
    selected_coordinates = _selected_memory_coordinates(
        records,
        template_configuration,
        transition_edges,
    )
    candidate_labels = tuple(coordinate.label for coordinate in selected_coordinates)
    selected_indices = np.asarray(
        conductivity_result.effect_attribution[
            "mori_filter_accepted_candidate_indices"
        ],
        dtype=int,
    )
    memory_matrix = np.asarray(
        conductivity_result.mori_memory_matrix_A,
        dtype=float,
    )
    current_coupling = np.asarray(
        conductivity_result.mori_current_coupling_matrix_h,
        dtype=float,
    )
    state_family_memory_values = _state_family_memory_value_matrix(
        state_quadratures,
        transition_edges,
    )
    retained_state_family_mode_indices = np.asarray(
        conductivity_result.effect_attribution[
            "state_family_memory_retained_indices"
        ],
        dtype=int,
    )
    state_family_memory_values = state_family_memory_values[
        :, retained_state_family_mode_indices
    ]
    expected_memory_count = selected_indices.size + state_family_memory_values.shape[1]
    if expected_memory_count != memory_matrix.shape[0]:
        raise ValueError("continuous and discrete Mori mode counts do not match final matrix")
    if current_coupling.shape != (expected_memory_count, CARTESIAN_DIMENSION):
        raise ValueError("selected Mori current coupling shape mismatch")
    if expected_memory_count == 0:
        conductivity_result.effect_attribution["mori_mode_ledger"] = ()
        return

    if selected_indices.size:
        maximum_candidate_index = int(np.max(selected_indices))
        if maximum_candidate_index >= len(candidate_labels):
            raise ValueError("selected Mori coordinate index exceeds candidate labels")
    memory_solution = symmetric_psd_pseudoinverse(memory_matrix) @ current_coupling
    mode_contributions = np.einsum("ia,ia->i", current_coupling, memory_solution)
    mode_ledger = []
    for selected_position, candidate_index in enumerate(selected_indices):
        supported_state_labels = []
        for state_index, state_quadrature in enumerate(
            state_quadratures
        ):
            coordinate_is_active = any(
                candidate_index
                in np.asarray(
                    ownership_basis.bounded_memory_mode_indices,
                    dtype=int,
                )
                for ownership_basis in state_quadrature.transport_ownership_bases
            )
            if coordinate_is_active:
                supported_state_labels.append(state_quadratures[state_index].label)
        coordinate_label = candidate_labels[candidate_index]
        transport_ownership, matching_transition_families = _memory_transport_ownership(
            records, coordinate_label
        )
        mode_ledger.append(
            {
                "mode_label": coordinate_label,
                "A_mu_mu": float(memory_matrix[selected_position, selected_position]),
                "h_mu_norm": float(np.linalg.norm(current_coupling[selected_position])),
                "h_mu_A_pseudoinverse_h_contribution": float(
                    mode_contributions[selected_position]
                ),
                "state_support": tuple(supported_state_labels),
                "physical_owner": _memory_coordinate_physical_owner(coordinate_label),
                "transport_ownership": transport_ownership.value,
                "matching_transition_families": matching_transition_families,
            }
        )
    for family_position in range(state_family_memory_values.shape[1]):
        matrix_index = selected_indices.size + family_position
        supported_state_labels = tuple(
            state_quadrature.label
            for state_index, state_quadrature in enumerate(state_quadratures)
            if abs(state_family_memory_values[state_index, family_position])
            > np.finfo(float).eps
        )
        mode_ledger.append(
            {
                "mode_label": f"state_family_residence_subspace[{family_position}]",
                "A_mu_mu": float(memory_matrix[matrix_index, matrix_index]),
                "h_mu_norm": float(np.linalg.norm(current_coupling[matrix_index])),
                "h_mu_A_pseudoinverse_h_contribution": float(
                    mode_contributions[matrix_index]
                ),
                "state_support": supported_state_labels,
                "physical_owner": "state_conditioned_residence_memory",
                "transport_ownership": TransportOwnership.BOUNDED_MEMORY.value,
                "matching_transition_families": (),
            }
        )
    if not np.isclose(
        float(np.sum(mode_contributions)),
        float(np.trace(conductivity_result.continuous_mori_correction_tensor)),
    ):
        raise ValueError("Mori mode ledger does not sum to the correction tensor")
    conductivity_result.effect_attribution["mori_mode_ledger"] = tuple(mode_ledger)


def _memory_coordinate_physical_owner(coordinate_label: str) -> str:
    owner_by_coordinate = {
        MemoryCoordinateFamily.CAGE_BACKJUMP.value: "cage_backjump_memory",
        MemoryCoordinateFamily.PARTNER_RESIDENCE.value: "partner_residence_memory",
        MemoryCoordinateFamily.LIGAND_SHELL.value: "ligand_shell_residence_memory",
        MemoryCoordinateFamily.ANION_ORIENTATION.value: "anion_orientation_memory",
        MemoryCoordinateFamily.BOUNDED_INTERNAL_POLARIZATION.value: (
            "bounded_internal_polarization_memory"
        ),
    }
    coordinate_family_label = coordinate_label.split("[", maxsplit=1)[0]
    if coordinate_family_label not in owner_by_coordinate:
        raise ValueError(
            f"missing physical owner for Mori coordinate {coordinate_label}"
        )
    return owner_by_coordinate[coordinate_family_label]


def _memory_transport_ownership(
    records: PhysicalLibraryRecords,
    coordinate_label: str,
) -> tuple[TransportOwnership, tuple[str, ...]]:
    record_key_by_coordinate = {
        MemoryCoordinateFamily.CAGE_BACKJUMP.value: "cage_backjump",
        MemoryCoordinateFamily.PARTNER_RESIDENCE.value: "partner_residence",
        MemoryCoordinateFamily.LIGAND_SHELL.value: "ligand_shell_residence",
        MemoryCoordinateFamily.ANION_ORIENTATION.value: "anion_orientation",
        MemoryCoordinateFamily.BOUNDED_INTERNAL_POLARIZATION.value: (
            "bounded_internal_polarization"
        ),
    }
    coordinate_family_label = coordinate_label.split("[", maxsplit=1)[0]
    if coordinate_family_label not in record_key_by_coordinate:
        return TransportOwnership.BOUNDED_MEMORY, ()
    memory_record_key = record_key_by_coordinate[coordinate_family_label]
    memory_family_record = records.memory_record["memory_records"][memory_record_key]
    ownership = TransportOwnership(str(memory_family_record["transport_ownership"]))
    if ownership not in (
        TransportOwnership.BOUNDED_MEMORY,
        TransportOwnership.DIAGNOSTIC,
    ):
        raise ValueError(
            f"memory coordinate {coordinate_label} has invalid transport ownership"
        )
    return ownership, tuple(memory_family_record["matching_transition_families"])


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
            "component_names": tuple(
                component.name for component in projected_components
            ),
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
    for component in recipe_context.resolved_species:
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
    for component in recipe_context.resolved_species:
        active_component = _component_with_active_recipe_loading(
            recipe_context, component
        )
        if active_component.concentration_mol_m3 <= 0.0:
            continue
        species_record = records.species_records[active_component.name]
        conformer_coordinates = np.asarray(
            species_record["reference_conformer_coordinates_m"],
            dtype=float,
        )
        offset = _reference_offset_m(records, active_component.name, molecule_id)
        for site_index, site_record in enumerate(species_record["sites"]):
            species_names.append(active_component.name)
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
        box_lengths_m=np.asarray(
            numerical_options.reference_box_lengths_m, dtype=float
        ),
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
        component.name
        for component in _projected_mass_balance_components(recipe_context)
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
            template_configuration,
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
                if not _state_key_has_valid_transport_topology(state_key):
                    continue
                if _state_requires_additive_component(state_key):
                    if not additive_component_names:
                        raise ValueError(
                            "additive state generated without additive component"
                        )
                    active_additive_component_names = additive_component_names
                else:
                    active_additive_component_names = (NO_ACTIVE_ADDITIVE_COMPONENT,)
                topology_records = _aggregate_topology_records_for_state(
                    records,
                    state_key,
                )
                for active_additive_component_name, topology_record in product(
                    active_additive_component_names,
                    topology_records,
                ):
                    additive_state_key = _state_key_with_active_additive(
                        state_key,
                        active_additive_component_name,
                    )
                    additive_state_key = _state_key_with_aggregate_topology(
                        additive_state_key,
                        topology_record,
                    )
                    if (
                        topology_record
                        and _state_requires_additive_component(additive_state_key)
                        and int(
                            topology_record["component_stoichiometry"]["ligand"]
                        )
                        == 0
                    ):
                        continue
                    state_label_with_components = "|".join(additive_state_key)
                    configurations = _feasible_state_local_transport_configurations(
                        records=records,
                        template_configuration=template_configuration,
                        coordinate_values_by_node=state_group.coordinate_values,
                        active_anion_component_name=anion_component_name,
                        active_additive_component_name=(
                            active_additive_component_name
                        ),
                        state_key=additive_state_key,
                        topology_record=topology_record,
                    )
                    if not configurations:
                        continue
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
                    relative_descriptor = _state_relative_displacement_descriptor(
                        records,
                        configurations,
                        local_fields,
                        np.asarray(state_group.weights, dtype=float),
                        recipe_context.temperature_K,
                        anion_component_name,
                        additive_state_key,
                    )
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
                            transport_ownership_bases=tuple(
                                _empty_physical_transport_ownership_basis(
                                    configuration
                                )
                                for configuration in configurations
                            ),
                            relative_displacement_fluctuations_m=relative_descriptor[0],
                            relative_displacement_mobility_m2_s=relative_descriptor[1],
                            relative_center_charge_numbers=relative_descriptor[2],
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
    state_quadratures = _normalize_transport_equivalent_partition_weights(
        tuple(quadratures)
    )
    state_quadratures = _apply_state_symmetry_and_degeneracy_factors(
        records,
        state_quadratures,
    )
    state_quadratures = _state_quadratures_with_equilibrium_attribution(
        records,
        state_quadratures,
        recipe_context.temperature_K,
    )
    return _filter_state_quadratures_by_partition_weight(
        records,
        state_quadratures,
        recipe_context.temperature_K,
    )


def _state_quadratures_with_equilibrium_attribution(
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    temperature_K: float,
) -> tuple[PhysicalStateQuadrature, ...]:
    thermal_energy_J_mol = R * _positive_float(temperature_K, "temperature_K")
    attributed_quadratures = []
    for state_quadrature in state_quadratures:
        attributed_weights = []
        for configuration, local_fields, weight in zip(
            state_quadrature.configurations,
            state_quadrature.local_fields,
            state_quadrature.weights,
            strict=True,
        ):
            base_potential = build_physical_objects(
                records,
                configuration,
                temperature_K,
                local_fields.dielectric_constant,
                local_fields.viscosity_Pa_s,
                local_fields.ionic_strength_mol_m3,
                local_fields.local_packing_fraction,
            ).potential_energy_J_mol
            attributed_potential = _state_equilibrium_potential_energy_J_mol(
                records,
                state_quadrature.label,
                configuration,
                local_fields.dielectric_constant,
                base_potential,
            )
            attribution_energy = attributed_potential - base_potential
            attributed_weights.append(
                float(weight) * float(np.exp(-attribution_energy / thermal_energy_J_mol))
            )
        attributed_quadratures.append(
            replace(
                state_quadrature,
                weights=np.asarray(attributed_weights, dtype=float),
            )
        )
    return tuple(attributed_quadratures)


def _normalize_transport_equivalent_partition_weights(
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
) -> tuple[PhysicalStateQuadrature, ...]:
    thermodynamic_group_by_state = tuple(
        _thermodynamic_state_group(_state_key_from_label(state.label))
        for state in state_quadratures
    )
    group_weight = Counter()
    for state, thermodynamic_group in zip(
        state_quadratures,
        thermodynamic_group_by_state,
        strict=True,
    ):
        group_weight[thermodynamic_group] += float(np.sum(state.weights))
    return tuple(
        replace(
            state,
            weights=np.asarray(state.weights, dtype=float)
            / float(group_weight[thermodynamic_group]),
        )
        for state, thermodynamic_group in zip(
            state_quadratures,
            thermodynamic_group_by_state,
            strict=True,
        )
    )


def _thermodynamic_state_group(state_key: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        state_key[state_index]
        for state_index in (
            STATE_KEY_PAIR_INDEX,
            STATE_KEY_SHELL_INDEX,
            STATE_KEY_LIGAND_INDEX,
            STATE_KEY_ANION_INDEX,
            STATE_KEY_CLUSTER_INDEX,
        )
    )


def _apply_state_symmetry_and_degeneracy_factors(
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
) -> tuple[PhysicalStateQuadrature, ...]:
    adjusted_states = []
    for state in state_quadratures:
        topology_record = _aggregate_topology_record_from_state_key(
            records,
            _state_key_from_label(state.label),
        )
        if not topology_record:
            adjusted_states.append(state)
            continue
        symmetry_number = int(topology_record["symmetry_number"])
        degeneracy = int(topology_record["degeneracy"])
        adjusted_states.append(
            replace(
                state,
                weights=np.asarray(state.weights, dtype=float)
                * float(degeneracy)
                / float(symmetry_number),
            )
        )
    return tuple(adjusted_states)


def _state_key_has_valid_transport_topology(state_key: tuple[str, ...]) -> bool:
    pair_state = _state_key_base_value(state_key[STATE_KEY_PAIR_INDEX])
    if pair_state != "addSSIP":
        return True
    ligand_state = _state_key_base_value(state_key[STATE_KEY_LIGAND_INDEX])
    shell_state = _state_key_base_value(state_key[STATE_KEY_SHELL_INDEX])
    return ligand_state in {"monodentate", "multidentate"} or shell_state in {
        "neutral_ligand_bound",
        "mixed_ligand_anion",
    }


def _aggregate_topology_records_for_state(
    records: PhysicalLibraryRecords,
    state_key: tuple[str, ...],
) -> tuple[dict, ...]:
    cluster_state = _state_key_base_value(state_key[STATE_KEY_CLUSTER_INDEX])
    if cluster_state not in {"aggregate", "bridge_network"}:
        return ({},)
    topology_records = tuple(
        topology_record
        for topology_record in records.association_record["aggregate_topologies"].values()
        if str(topology_record["cluster_family"]) == cluster_state
    )
    if not topology_records:
        raise ValueError(f"aggregate_topology_missing: {cluster_state}")
    return topology_records


def _state_key_with_aggregate_topology(
    state_key: tuple[str, ...],
    topology_record: dict,
) -> tuple[str, ...]:
    if not topology_record:
        return state_key
    key_parts = list(state_key)
    cluster_family = str(topology_record["cluster_family"])
    key_parts[STATE_KEY_CLUSTER_INDEX] = (
        f'{cluster_family}{STATE_KEY_COMPONENT_SEPARATOR}{topology_record["topology_id"]}'
    )
    return tuple(key_parts)


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
                transport_ownership_bases=(
                    _empty_physical_transport_ownership_basis(configuration),
                ),
                relative_displacement_fluctuations_m=np.empty(
                    (0, CARTESIAN_DIMENSION), dtype=float
                ),
                relative_displacement_mobility_m2_s=np.empty((0, 0), dtype=float),
                relative_center_charge_numbers=np.empty(0, dtype=float),
            )
        )
    return quadratures


def _empty_physical_transport_ownership_basis(
    configuration: SiteConfiguration,
) -> StateTransportOwnershipBasis:
    coordinate_dimension = int(np.asarray(configuration.positions_m).size)
    return StateTransportOwnershipBasis(
        transition_displacement_gradients=np.empty(
            (0, coordinate_dimension),
            dtype=float,
        ),
        transition_edge_indices=np.empty(0, dtype=int),
        bounded_memory_gradients=np.empty(
            (0, coordinate_dimension),
            dtype=float,
        ),
        bounded_memory_mode_indices=np.empty(0, dtype=int),
        diagnostic_gradients=np.empty((0, coordinate_dimension), dtype=float),
        diagnostic_source_ids=(),
    )


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
            if _state_key_base_value(
                _state_key_from_label(state_quadrature.label)[STATE_KEY_PAIR_INDEX]
            )
            in {PairBasin.FREE.value, PAIR_STATE_FREE_ADDITIVE_RESERVOIR}
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
        "partner_inactive",
        "identity_inactive",
        "hop_inactive",
        "cage_inactive",
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
        species_names=tuple(
            configuration.species_names[site_index] for site_index in site_indices
        ),
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
    topology_record = _aggregate_topology_record_from_state_key(records, state_key)
    topology_stoichiometry = (
        topology_record["component_stoichiometry"] if topology_record else {}
    )
    for component_index, component_name in enumerate(component_names):
        component_role = _species_role(records, component_name)
        if component_role == SpeciesRole.CATION and not is_additive_reservoir_state:
            stoichiometry[component_index] = float(
                topology_stoichiometry.get("Li", 1)
            )
            continue
        if component_role == SpeciesRole.CATION:
            continue
        if (
            component_role == SpeciesRole.ANION
            and component_name == active_anion_component_name
        ):
            stoichiometry[component_index] = float(
                topology_stoichiometry.get("A", 1)
            )
            continue
        if component_role == SpeciesRole.ANION:
            continue
        if component_role == SpeciesRole.ADDITIVE:
            if component_name == active_additive_component_name:
                stoichiometry[component_index] = _state_additive_stoichiometry(
                    state_key
                )
            continue
        raise ValueError(
            f"projected mass-balance component is not transport-active: {component_name}"
        )
    if float(np.sum(stoichiometry)) <= 0.0:
        raise ValueError("transport state stoichiometry is empty")
    return stoichiometry


def _aggregate_topology_record_from_state_key(
    records: PhysicalLibraryRecords,
    state_key: tuple[str, ...],
) -> dict:
    cluster_value = state_key[STATE_KEY_CLUSTER_INDEX]
    if STATE_KEY_COMPONENT_SEPARATOR not in cluster_value:
        return {}
    cluster_family, topology_id = cluster_value.split(
        STATE_KEY_COMPONENT_SEPARATOR,
        maxsplit=1,
    )
    if cluster_family not in {"aggregate", "bridge_network"}:
        return {}
    topology_records = records.association_record["aggregate_topologies"]
    if topology_id not in topology_records:
        raise ValueError(f"aggregate_topology_missing: {topology_id}")
    return topology_records[topology_id]


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


def _state_key_component_name(state_key_value: str) -> str:
    if STATE_KEY_COMPONENT_SEPARATOR not in state_key_value:
        return ""
    component_name, _feature_name = state_key_value.split(
        STATE_KEY_COMPONENT_SEPARATOR,
        maxsplit=1,
    )
    if not component_name:
        raise ValueError(f"state key component is empty: {state_key_value}")
    return component_name


def _active_additive_name_from_state_key(state_key: tuple[str, ...]) -> str:
    ligand_component_name = _state_key_component_name(state_key[STATE_KEY_LIGAND_INDEX])
    if ligand_component_name:
        return ligand_component_name
    return NO_ACTIVE_ADDITIVE_COMPONENT


def _active_anion_name_from_state_key(state_key: tuple[str, ...]) -> str:
    anion_field = state_key[STATE_KEY_ANION_INDEX]
    if anion_field == "none":
        return NO_ACTIVE_ANION_COMPONENT
    anion_component_name = _state_key_component_name(anion_field)
    if not anion_component_name:
        raise ValueError(
            f"state anion field must be '<anion>:<feature>': {anion_field}"
        )
    return anion_component_name


def _transition_active_additive_name(
    from_state_key: tuple[str, ...],
    to_state_key: tuple[str, ...],
) -> str:
    active_additive_names = tuple(
        additive_name
        for additive_name in (
            _active_additive_name_from_state_key(from_state_key),
            _active_additive_name_from_state_key(to_state_key),
        )
        if additive_name != NO_ACTIVE_ADDITIVE_COMPONENT
    )
    unique_active_additive_names = tuple(dict.fromkeys(active_additive_names))
    if len(unique_active_additive_names) > 1:
        raise ValueError(
            "transition edge has multiple active additive components: "
            f"{unique_active_additive_names}"
        )
    if unique_active_additive_names:
        return unique_active_additive_names[0]
    return NO_ACTIVE_ADDITIVE_COMPONENT


def _transition_local_template_configuration(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    from_state_label: str,
    to_state_label: str,
) -> SiteConfiguration:
    from_state_key = _state_key_from_label(from_state_label)
    to_state_key = _state_key_from_label(to_state_label)
    from_active_anion_name = _active_anion_name_from_state_key(from_state_key)
    to_active_anion_name = _active_anion_name_from_state_key(to_state_key)
    if from_active_anion_name != to_active_anion_name:
        raise ValueError(
            "transition edge changes active anion component: "
            f"{from_active_anion_name} -> {to_active_anion_name}"
        )
    active_additive_name = _transition_active_additive_name(
        from_state_key,
        to_state_key,
    )
    additive_reference_key = from_state_key
    if _state_requires_additive_component(to_state_key):
        additive_reference_key = to_state_key
    return _state_local_transport_configuration(
        records,
        template_configuration,
        from_active_anion_name,
        active_additive_name,
        additive_reference_key,
    )


def _state_local_transport_configuration(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    active_anion_component_name: str,
    active_additive_component_name: str,
    state_key: tuple[str, ...],
) -> SiteConfiguration:
    state_local_configuration = _configuration_with_state_local_species(
        records,
        configuration,
        active_anion_component_name,
        active_additive_component_name,
        state_key,
    )
    return _configuration_with_active_anion_first(
        records,
        state_local_configuration,
        active_anion_component_name,
    )


def _state_local_transport_configuration_from_coordinates(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    active_anion_component_name: str,
    active_additive_component_name: str,
    state_key: tuple[str, ...],
) -> SiteConfiguration:
    configured_state = _configured_state_from_reduced_coordinates(
        records=records,
        template_configuration=template_configuration,
        coordinate_values=coordinate_values,
        active_anion_component_name=active_anion_component_name,
        active_additive_component_name=active_additive_component_name,
        state_key=state_key,
    )
    if _state_key_base_value(state_key[STATE_KEY_PAIR_INDEX]) == "addSSIP":
        return _configuration_with_ligand_separator_geometry(
            records,
            configured_state,
        )
    return configured_state


def _configured_state_from_reduced_coordinates(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    active_anion_component_name: str,
    active_additive_component_name: str,
    state_key: tuple[str, ...],
) -> SiteConfiguration:
    state_coordinate_values = dict(coordinate_values)
    if active_additive_component_name == NO_ACTIVE_ADDITIVE_COMPONENT:
        state_coordinate_values[ReducedCoordinate.LI_LIGAND_COORDINATION.value] = 0.0
    state_local_template = _configuration_with_state_local_species(
        records,
        template_configuration,
        active_anion_component_name,
        active_additive_component_name,
        state_key,
    )
    active_anion_first_template = _configuration_with_active_anion_first(
        records,
        state_local_template,
        active_anion_component_name,
    )
    return _configuration_with_reduced_coordinate_values(
        records,
        active_anion_first_template,
        state_coordinate_values,
    )


def _feasible_state_local_transport_configurations(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values_by_node: tuple[dict[str, float], ...],
    active_anion_component_name: str,
    active_additive_component_name: str,
    state_key: tuple[str, ...],
    topology_record: dict,
) -> tuple[SiteConfiguration, ...]:
    if topology_record:
        return tuple(
            _configuration_for_aggregate_topology(
                records,
                _state_local_transport_configuration_from_coordinates(
                    records=records,
                    template_configuration=template_configuration,
                    coordinate_values=coordinate_values,
                    active_anion_component_name=active_anion_component_name,
                    active_additive_component_name=active_additive_component_name,
                    state_key=state_key,
                ),
                topology_record,
                float(coordinate_values[ReducedCoordinate.LI_ANION_DISTANCE.value]),
            )
            for coordinate_values in coordinate_values_by_node
        )
    if _state_key_base_value(state_key[STATE_KEY_PAIR_INDEX]) == "addSSIP":
        unseparated_configurations = tuple(
            _configured_state_from_reduced_coordinates(
                records=records,
                template_configuration=template_configuration,
                coordinate_values=coordinate_values,
                active_anion_component_name=active_anion_component_name,
                active_additive_component_name=active_additive_component_name,
                state_key=state_key,
            )
            for coordinate_values in coordinate_values_by_node
        )
        if not all(
            _ligand_separator_geometry_is_feasible(records, configuration)
            for configuration in unseparated_configurations
        ):
            return ()
        return tuple(
            _configuration_with_ligand_separator_geometry(records, configuration)
            for configuration in unseparated_configurations
        )
    return tuple(
        _state_local_transport_configuration_from_coordinates(
            records=records,
            template_configuration=template_configuration,
            coordinate_values=coordinate_values,
            active_anion_component_name=active_anion_component_name,
            active_additive_component_name=active_additive_component_name,
            state_key=state_key,
        )
        for coordinate_values in coordinate_values_by_node
    )


def _configuration_for_aggregate_topology(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    topology_record: dict,
    pair_distance_m: float,
) -> SiteConfiguration:
    pair_distance = _positive_float(pair_distance_m, "aggregate topology distance")
    component_stoichiometry = topology_record["component_stoichiometry"]
    cation_count = int(component_stoichiometry["Li"])
    anion_count = int(component_stoichiometry["A"])
    ligand_count = int(component_stoichiometry["ligand"])
    if cation_count < int(topology_record["minimum_cation_count"]):
        raise ValueError("aggregate_multiplicity_mismatch: cation count")
    if anion_count < int(topology_record["minimum_anion_count"]):
        raise ValueError("aggregate_multiplicity_mismatch: anion count")
    if ligand_count < int(topology_record["minimum_ligand_count"]):
        raise ValueError("aggregate_multiplicity_mismatch: ligand count")

    cation_indices = _first_molecule_indices_with_role(
        records, configuration, SpeciesRole.CATION
    )
    anion_indices = _first_molecule_indices_with_role(
        records, configuration, SpeciesRole.ANION
    )
    pair_axis = _aggregate_pair_axis(
        records,
        configuration,
        cation_indices,
        anion_indices,
    )
    center_positions = _aggregate_topology_center_positions(
        topology_record,
        pair_distance,
        pair_axis,
    )
    retained_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if _species_role(records, species_name)
        not in (SpeciesRole.CATION, SpeciesRole.ANION)
    )
    species_names = [configuration.species_names[index] for index in retained_indices]
    site_ids = [int(configuration.site_ids[index]) for index in retained_indices]
    positions_m = [
        np.asarray(configuration.unwrapped_positions_m[index], dtype=float)
        for index in retained_indices
    ]
    molecule_ids = [int(configuration.molecule_ids[index]) for index in retained_indices]
    next_molecule_id = max(molecule_ids, default=-1) + 1
    source_indices_by_role = {
        "Li": cation_indices,
        "A": anion_indices,
    }
    formal_charge_number = 0.0
    for center_id, target_center_m in center_positions:
        center_role = "Li" if center_id.startswith("Li") else "A"
        source_indices = source_indices_by_role[center_role]
        species_name = configuration.species_names[source_indices[0]]
        source_molecule_id = int(configuration.molecule_ids[source_indices[0]])
        source_center_m = molecule_center_of_mass_m(
            records,
            configuration,
            species_name,
            source_molecule_id,
        )
        for source_index in source_indices:
            species_names.append(species_name)
            site_ids.append(int(configuration.site_ids[source_index]))
            positions_m.append(
                np.asarray(configuration.unwrapped_positions_m[source_index], dtype=float)
                - source_center_m
                + target_center_m
            )
            molecule_ids.append(next_molecule_id)
        formal_charge_number += float(
            records.species_records[species_name]["formal_charge_e"]
        )
        next_molecule_id += 1
    if formal_charge_number != float(topology_record["net_formal_charge_e"]):
        raise ValueError("aggregate_charge_mismatch")
    unwrapped_positions_m = np.asarray(positions_m, dtype=float)
    box_lengths_m = np.asarray(configuration.box_lengths_m, dtype=float)
    wrapped_positions_m = unwrapped_positions_m - box_lengths_m * np.floor(
        unwrapped_positions_m / box_lengths_m
    )
    aggregate_configuration = SiteConfiguration(
        species_names=tuple(species_names),
        molecule_ids=np.asarray(molecule_ids, dtype=int),
        site_ids=np.asarray(site_ids, dtype=int),
        positions_m=wrapped_positions_m,
        unwrapped_positions_m=unwrapped_positions_m,
        box_lengths_m=box_lengths_m,
    )
    _validate_aggregate_graph_distances(
        records,
        aggregate_configuration,
        topology_record,
        pair_distance,
    )
    return aggregate_configuration


def _aggregate_pair_axis(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    cation_indices: tuple[int, ...],
    anion_indices: tuple[int, ...],
) -> Array:
    cation_species_name = configuration.species_names[cation_indices[0]]
    anion_species_name = configuration.species_names[anion_indices[0]]
    cation_center_m = molecule_center_of_mass_m(
        records,
        configuration,
        cation_species_name,
        int(configuration.molecule_ids[cation_indices[0]]),
    )
    anion_center_m = molecule_center_of_mass_m(
        records,
        configuration,
        anion_species_name,
        int(configuration.molecule_ids[anion_indices[0]]),
    )
    pair_axis = _minimum_image_vector_m(
        cation_center_m, anion_center_m, configuration.box_lengths_m
    )
    return pair_axis / _positive_float(float(np.linalg.norm(pair_axis)), "aggregate pair axis")


def _aggregate_topology_center_positions(
    topology_record: dict,
    pair_distance_m: float,
    pair_axis: Array,
) -> tuple[tuple[str, Array], ...]:
    topology_id = str(topology_record["topology_id"])
    if topology_id == "Li2A_positive":
        return (
            ("Li0", -pair_distance_m * pair_axis),
            ("A0", np.zeros(CARTESIAN_DIMENSION)),
            ("Li1", pair_distance_m * pair_axis),
        )
    if topology_id == "LiA2_negative":
        return (
            ("Li0", np.zeros(CARTESIAN_DIMENSION)),
            ("A0", -pair_distance_m * pair_axis),
            ("A1", pair_distance_m * pair_axis),
        )
    if topology_id == "Li2A2_neutral":
        return tuple(
            (center_id, multiplier * pair_distance_m * pair_axis)
            for center_id, multiplier in (
                ("Li0", -CHAIN_OUTER_CENTER_OFFSET_MULTIPLIER),
                ("A0", -CHAIN_INNER_CENTER_OFFSET_MULTIPLIER),
                ("Li1", CHAIN_INNER_CENTER_OFFSET_MULTIPLIER),
                ("A1", CHAIN_OUTER_CENTER_OFFSET_MULTIPLIER),
            )
        )
    if str(topology_record["cluster_family"]) == "bridge_network":
        basis_axes = np.eye(CARTESIAN_DIMENSION)
        basis_axis = basis_axes[int(np.argmin(np.abs(basis_axes @ pair_axis)))]
        second_axis = basis_axis - float(np.dot(basis_axis, pair_axis)) * pair_axis
        second_axis /= _positive_float(float(np.linalg.norm(second_axis)), "bridge second axis")
        radial_offset_m = pair_distance_m / np.sqrt(2.0)
        return (
            ("Li0", radial_offset_m * pair_axis),
            ("A0", radial_offset_m * second_axis),
            ("Li1", -radial_offset_m * pair_axis),
            ("A1", -radial_offset_m * second_axis),
        )
    raise ValueError(f"aggregate_topology_missing: {topology_id}")


def _validate_aggregate_graph_distances(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    topology_record: dict,
    pair_distance_m: float,
) -> None:
    center_positions = {}
    role_molecules = {
        "Li": _molecule_site_index_groups_with_role(records, configuration, SpeciesRole.CATION),
        "A": _molecule_site_index_groups_with_role(records, configuration, SpeciesRole.ANION),
    }
    for role_name, molecule_groups in role_molecules.items():
        for molecule_index, molecule_indices in enumerate(molecule_groups):
            species_name = configuration.species_names[molecule_indices[0]]
            center_positions[f"{role_name}{molecule_index}"] = molecule_center_of_mass_m(
                records, configuration, species_name, int(configuration.molecule_ids[molecule_indices[0]])
            )
    tolerance_m = (
        np.finfo(float).eps
        * pair_distance_m
        * SUMMATION_ROUNDOFF_EPSILON_FACTOR
    )
    for first_center_id, second_center_id in topology_record["graph_edges"]:
        distance_m = float(
            np.linalg.norm(
                center_positions[second_center_id]
                - center_positions[first_center_id]
            )
        )
        if abs(distance_m - pair_distance_m) > tolerance_m:
            raise ValueError("aggregate_multiplicity_mismatch: graph-edge distance")


def _ligand_separator_geometry_is_feasible(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> bool:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    additive_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ADDITIVE,
    )
    anion_site_index = _molecule_coordination_site_index(
        records,
        configuration,
        anion_indices,
    )
    additive_site_index = _molecule_coordination_site_index(
        records,
        configuration,
        additive_indices,
    )
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    cation_position_m = positions_m[cation_index]
    cation_anion_distance_m = float(
        np.linalg.norm(
            _minimum_image_vector_m(
                cation_position_m,
                positions_m[anion_site_index],
                configuration.box_lengths_m,
            )
        )
    )
    cation_ligand_distance_m = float(
        np.linalg.norm(
            _minimum_image_vector_m(
                cation_position_m,
                positions_m[additive_site_index],
                configuration.box_lengths_m,
            )
        )
    )
    return 0.0 < cation_ligand_distance_m < cation_anion_distance_m


def _configuration_with_ligand_separator_geometry(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> SiteConfiguration:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    additive_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ADDITIVE,
    )
    anion_coordination_site_index = _molecule_coordination_site_index(
        records,
        configuration,
        anion_indices,
    )
    additive_coordination_site_index = _molecule_coordination_site_index(
        records,
        configuration,
        additive_indices,
    )
    positions_m = np.asarray(configuration.positions_m, dtype=float).copy()
    cation_position_m = positions_m[cation_index]
    cation_to_anion_m = _minimum_image_vector_m(
        cation_position_m,
        positions_m[anion_coordination_site_index],
        configuration.box_lengths_m,
    )
    cation_anion_distance_m = _positive_float(
        float(np.linalg.norm(cation_to_anion_m)),
        "addSSIP cation-anion distance",
    )
    separator_direction = cation_to_anion_m / cation_anion_distance_m
    current_ligand_displacement_m = _minimum_image_vector_m(
        cation_position_m,
        positions_m[additive_coordination_site_index],
        configuration.box_lengths_m,
    )
    li_ligand_distance_m = _positive_float(
        float(np.linalg.norm(current_ligand_displacement_m)),
        "addSSIP Li-ligand distance",
    )
    if li_ligand_distance_m >= cation_anion_distance_m:
        raise ValueError("addSSIP ligand does not fit between cation and anion")
    target_coordination_site_m = (
        cation_position_m + li_ligand_distance_m * separator_direction
    )
    additive_shift_m = (
        target_coordination_site_m - positions_m[additive_coordination_site_index]
    )
    positions_m[np.asarray(additive_indices, dtype=int)] += additive_shift_m
    return SiteConfiguration(
        species_names=configuration.species_names,
        molecule_ids=np.asarray(configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int),
        positions_m=positions_m,
        unwrapped_positions_m=positions_m,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _configuration_with_active_anion_first(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    active_anion_component_name: str,
) -> SiteConfiguration:
    non_anion_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if _species_role(records, species_name) != SpeciesRole.ANION
    )
    active_anion_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if species_name == active_anion_component_name
    )
    other_anion_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if _species_role(records, species_name) == SpeciesRole.ANION
        and species_name != active_anion_component_name
    )
    if not active_anion_indices:
        raise ValueError(
            f"state-local configuration has no active anion {active_anion_component_name}"
        )
    return _configuration_from_site_indices(
        configuration,
        non_anion_indices + active_anion_indices + other_anion_indices,
    )


def _configuration_with_state_local_species(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    active_anion_component_name: str,
    active_additive_component_name: str,
    state_key: tuple[str, ...],
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
        species_names=tuple(
            configuration.species_names[site_index] for site_index in site_indices
        ),
        molecule_ids=np.asarray(molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int)[site_index_array],
        positions_m=np.asarray(configuration.positions_m, dtype=float)[
            site_index_array
        ],
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
    potential_energies_J_mol = []
    positive_weights = []
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
            raise ValueError(
                f"{state_quadrature.label} has nonpositive quadrature weight"
            )
        positive_weights.append(float(weight))
        potential_energies_J_mol.append(physical_objects.potential_energy_J_mol)
    energy_array = np.asarray(potential_energies_J_mol, dtype=float)
    state_energy_reference_J_mol = float(np.min(energy_array))
    log_terms = np.log(np.asarray(positive_weights, dtype=float)) - (
        energy_array - state_energy_reference_J_mol
    ) / (R * temperature_K)
    term_array = np.asarray(log_terms, dtype=float)
    maximum_log_term = _finite_float(
        float(np.max(term_array)),
        f"{state_quadrature.label}.maximum_log_term",
    )
    return maximum_log_term + float(
        np.log(np.sum(np.exp(term_array - maximum_log_term)))
    )


def _state_equilibrium_potential_energy_J_mol(
    records: PhysicalLibraryRecords,
    state_label: str,
    configuration: SiteConfiguration,
    dielectric_constant: float,
    base_potential_energy_J_mol: float,
) -> float:
    state_key = _state_key_from_label(state_label)
    association_energy_J_mol = _state_feature_sum(
        records.association_record["association_residual"],
        state_key,
        "population_operator_missing",
    )
    occluded_fraction = _state_feature_sum(
        records.association_record["state_resolved_born"],
        state_key,
        "state_resolved_born_missing",
    )
    if occluded_fraction < 0.0 or occluded_fraction >= 1.0:
        raise ValueError("state-resolved Born occlusion fraction must be in [0, 1)")
    born_energy_J_mol = compute_born_energy_J_mol(
        records,
        configuration,
        dielectric_constant,
    )
    desolvation_energy_J_mol = -born_energy_J_mol * occluded_fraction
    return _finite_float(
        base_potential_energy_J_mol
        + association_energy_J_mol
        + desolvation_energy_J_mol,
        f"{state_label}.state_equilibrium_potential_energy_J_mol",
    )


def _state_feature_sum(
    operator_record: dict,
    state_key: tuple[str, ...],
    failure_class: str,
) -> float:
    state_index_by_feature = {
        "pair": STATE_KEY_PAIR_INDEX,
        "shell": STATE_KEY_SHELL_INDEX,
        "ligand": STATE_KEY_LIGAND_INDEX,
        "cluster": STATE_KEY_CLUSTER_INDEX,
        "orientation": STATE_KEY_ORIENTATION_INDEX,
    }
    total = 0.0
    for feature_name, coefficients in operator_record["state_features"].items():
        if feature_name not in state_index_by_feature:
            raise ValueError(f"{failure_class}: unsupported state feature {feature_name}")
        state_value = _state_key_base_value(
            state_key[state_index_by_feature[feature_name]]
        )
        if state_value not in coefficients:
            raise KeyError(
                f"{failure_class}: missing {feature_name} record for {state_value}"
            )
        total += float(coefficients[state_value])
    return _finite_float(total, f"{failure_class}.state_feature_sum")


def build_self_current_projector(
    state_key: tuple[str, ...],
    configuration: SiteConfiguration,
    records: PhysicalLibraryRecords,
) -> Array:
    coordinate_count = (
        len(configuration.species_names) * CARTESIAN_DIMENSION
        + LOCAL_FIELD_VECTOR_LENGTH
    )
    projector = np.zeros((coordinate_count, coordinate_count), dtype=float)
    molecule_site_indices: dict[tuple[str, int], list[int]] = {}
    for site_index, (species_name, molecule_id) in enumerate(
        zip(
            configuration.species_names,
            np.asarray(configuration.molecule_ids, dtype=int),
            strict=True,
        )
    ):
        molecule_site_indices.setdefault(
            (species_name, int(molecule_id)), []
        ).append(site_index)
    topology_record = _aggregate_topology_record_from_state_key(records, state_key)
    if topology_record:
        charged_cluster_site_indices = tuple(
            site_index
            for (species_name, _molecule_id), site_indices in molecule_site_indices.items()
            if float(records.species_records[species_name]["formal_charge_e"]) != 0.0
            for site_index in site_indices
        )
        _assign_translation_projector_block(
            projector,
            charged_cluster_site_indices,
        )
    else:
        for (species_name, _molecule_id), site_indices in molecule_site_indices.items():
            formal_charge_number = float(
                records.species_records[species_name]["formal_charge_e"]
            )
            if formal_charge_number == 0.0:
                continue
            _assign_translation_projector_block(projector, tuple(site_indices))
    if not np.allclose(projector, projector.T):
        raise ValueError("self-current coordinate projector must be symmetric")
    if not np.allclose(projector @ projector, projector):
        raise ValueError("self-current coordinate projector must be idempotent")
    return projector


def _assign_translation_projector_block(
    projector: Array,
    site_indices: tuple[int, ...],
) -> None:
    if not site_indices:
        raise ValueError("self-current translation projector requires molecular sites")
    translation_weight = 1.0 / len(site_indices)
    for first_site_index in site_indices:
        for second_site_index in site_indices:
            for cartesian_index in range(CARTESIAN_DIMENSION):
                first_coordinate_index = (
                    first_site_index * CARTESIAN_DIMENSION + cartesian_index
                )
                second_coordinate_index = (
                    second_site_index * CARTESIAN_DIMENSION + cartesian_index
                )
                projector[first_coordinate_index, second_coordinate_index] = (
                    translation_weight
                )


def _group_state_quadrature_nodes(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    mixture: MixtureClosureResult,
    pair_label: str,
    coordinate_nodes: tuple[tuple[ReducedCoordinate, Array, Array], ...],
) -> dict[str, StateQuadratureGroup]:
    grouped_quadrature: dict[str, StateQuadratureGroup] = {}
    state_nodes = _state_node_index_tuples(records, coordinate_nodes)
    state_node_weights = _state_node_quadrature_weights(
        records,
        coordinate_nodes,
        state_nodes,
    )
    for state_node, coordinate_weight in zip(
        state_nodes,
        state_node_weights,
        strict=True,
    ):
        node_indices = state_node.node_indices
        coordinate_values = {
            coordinate_name.value: float(values[node_index])
            for node_index, (coordinate_name, values, _weights) in zip(
                node_indices,
                coordinate_nodes,
                strict=True,
            )
        }
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
            state_node.active_coordinates,
        )
        state_key = _population_basin_state_key(state_key)
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
            _local_fields_for_coordinate_values(
                records, configuration, coordinate_values
            )
        )
        grouped_quadrature[state_label].weights.append(coordinate_weight)
    return grouped_quadrature


def _population_basin_state_key(state_key: tuple[str, ...]) -> tuple[str, ...]:
    if len(state_key) != STATE_KEY_LENGTH:
        raise ValueError("state key has wrong length")
    population_key = list(state_key)
    population_key[STATE_KEY_ORIENTATION_INDEX] = "free_rotating"
    population_key[STATE_KEY_PARTNER_INDEX] = "partner_inactive"
    population_key[STATE_KEY_CAGE_INDEX] = "cage_inactive"
    population_key[STATE_KEY_ATMOSPHERE_INDEX] = "atmosphere_inactive"
    return tuple(population_key)


def _state_node_quadrature_weights(
    records: PhysicalLibraryRecords,
    coordinate_nodes: tuple[tuple[ReducedCoordinate, Array, Array], ...],
    state_nodes: tuple[StateQuadratureNode, ...],
) -> Array:
    raw_weights = np.asarray(
        [
            _coordinate_product_weight(state_node.node_indices, coordinate_nodes)
            for state_node in state_nodes
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(raw_weights)) or np.any(raw_weights <= 0.0):
        raise ValueError("state quadrature weights must be positive and finite")
    state_axis_generation = str(records.basis_record["state_axis_generation"])
    if state_axis_generation == "full_tensor_product":
        return raw_weights
    if state_axis_generation != "sparse_single_axis_plus_pair":
        raise ValueError(
            f"unsupported basis.state_axis_generation {state_axis_generation}"
        )
    pair_coordinate_index = _coordinate_nodes_index(
        coordinate_nodes,
        ReducedCoordinate.LI_ANION_DISTANCE,
    )
    pair_weights = np.asarray(
        coordinate_nodes[pair_coordinate_index][2],
        dtype=float,
    )
    normalized_weights = np.zeros_like(raw_weights)
    for pair_node_index, pair_weight in enumerate(pair_weights):
        matching_node_indices = np.asarray(
            [
                state_node_index
                for state_node_index, state_node in enumerate(state_nodes)
                if state_node.node_indices[pair_coordinate_index] == pair_node_index
            ],
            dtype=int,
        )
        if matching_node_indices.size == 0:
            raise ValueError(
                f"sparse state quadrature omitted pair node {pair_node_index}"
            )
        matching_raw_weight = float(np.sum(raw_weights[matching_node_indices]))
        normalized_weights[matching_node_indices] = (
            raw_weights[matching_node_indices]
            * float(pair_weight)
            / matching_raw_weight
        )
        if not np.isclose(
            float(np.sum(normalized_weights[matching_node_indices])),
            float(pair_weight),
            rtol=SUMMATION_ROUNDOFF_EPSILON_FACTOR * np.finfo(float).eps,
            atol=np.finfo(float).tiny,
        ):
            raise ValueError(
                f"sparse state quadrature does not conserve pair node {pair_node_index}"
            )
    return normalized_weights


def _state_node_index_tuples(
    records: PhysicalLibraryRecords,
    coordinate_nodes: tuple[tuple[ReducedCoordinate, Array, Array], ...],
) -> tuple[StateQuadratureNode, ...]:
    state_axis_generation = str(records.basis_record["state_axis_generation"])
    if state_axis_generation == "full_tensor_product":
        return tuple(
            StateQuadratureNode(
                node_indices=node_indices,
                active_coordinates=frozenset(
                    coordinate_name
                    for coordinate_name, _values, _weights in coordinate_nodes
                ),
            )
            for node_indices in product(
                *(
                    range(values.size)
                    for _coordinate_name, values, _weights in coordinate_nodes
                )
            )
        )
    if state_axis_generation != "sparse_single_axis_plus_pair":
        raise ValueError(
            f"unsupported basis.state_axis_generation {state_axis_generation}"
        )
    pair_coordinate_index = _coordinate_nodes_index(
        coordinate_nodes,
        ReducedCoordinate.LI_ANION_DISTANCE,
    )
    ligand_coordinate_index = _coordinate_nodes_index(
        coordinate_nodes,
        ReducedCoordinate.LI_LIGAND_COORDINATION,
    )
    baseline_indices = tuple(
        _baseline_node_index_for_coordinate(coordinate_name, values)
        for coordinate_name, values, _weights in coordinate_nodes
    )
    state_nodes: list[StateQuadratureNode] = []
    for pair_node_index in range(coordinate_nodes[pair_coordinate_index][1].size):
        pair_baseline_indices = list(baseline_indices)
        pair_baseline_indices[pair_coordinate_index] = pair_node_index
        state_nodes.append(
            StateQuadratureNode(
                node_indices=tuple(pair_baseline_indices),
                active_coordinates=frozenset(
                    {ReducedCoordinate.LI_ANION_DISTANCE}
                ),
            )
        )
        for coordinate_index, (_coordinate_name, values, _weights) in enumerate(
            coordinate_nodes
        ):
            if coordinate_index == pair_coordinate_index:
                continue
            baseline_node_index = baseline_indices[coordinate_index]
            for node_index in range(values.size):
                coordinate = coordinate_nodes[coordinate_index][0]
                if (
                    coordinate in _bounded_memory_only_coordinates()
                    and node_index == baseline_node_index
                ):
                    continue
                if (
                    node_index == baseline_node_index
                    and coordinate not in _unit_interval_state_coordinates()
                ):
                    continue
                conditioning_indices = _transport_axis_conditioning_indices(
                    coordinate_nodes,
                    coordinate_index,
                    pair_baseline_indices,
                )
                for conditioned_indices in conditioning_indices:
                    varied_indices = list(conditioned_indices)
                    varied_indices[coordinate_index] = node_index
                    active_coordinates = {
                        ReducedCoordinate.LI_ANION_DISTANCE,
                    }
                    if coordinate not in _bounded_memory_only_coordinates():
                        active_coordinates.add(coordinate)
                    if conditioned_indices[ligand_coordinate_index] != (
                        pair_baseline_indices[ligand_coordinate_index]
                    ):
                        active_coordinates.add(
                            ReducedCoordinate.LI_LIGAND_COORDINATION
                        )
                    state_nodes.append(
                        StateQuadratureNode(
                            node_indices=tuple(varied_indices),
                            active_coordinates=frozenset(active_coordinates),
                        )
                    )
    return tuple(dict.fromkeys(state_nodes))


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


def _bounded_memory_only_coordinates() -> set[ReducedCoordinate]:
    return {
        ReducedCoordinate.ATMOSPHERE_POLARIZATION,
        ReducedCoordinate.CAGE_COORDINATE,
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
        state_labels = tuple(
            state_quadrature.label for state_quadrature in state_quadratures
        )
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
        transition_template_configuration = _transition_local_template_configuration(
            records,
            template_configuration,
            state_quadratures[edge.from_state_index].label,
            state_quadratures[edge.to_state_index].label,
        )
        configurations = tuple(
            _configuration_for_transition_coordinate(
                records,
                transition_template_configuration,
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
        committor_gradients = (
            committor_result.committor_gradient[:, None] * flat_gradients
        )
        charge_polarization_by_grid = np.asarray(
            [
                compute_charge_polarization_m(records, configuration)
                for configuration in configurations
            ],
            dtype=float,
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
        uses_declared_rate_constant = _uses_declared_rate_constant(
            edge.family,
            transition_record,
        )
        residence_rate_constant_s_inv = (
            _transition_residence_rate_constant_s_inv(
                edge.family,
                transition_record,
                temperature_K,
            )
            if uses_declared_rate_constant
            else _derived_transition_residence_rate_constant_s_inv(
                edge.family,
                transition_record,
                reaction_coordinate_grid,
                diffusivity,
                temperature_K,
            )
        )
        first_displacement_moment_m, second_displacement_moment_m2 = (
            _transition_displacement_moments(
                transition_record,
                moment_input,
                _endpoint_geometry_displacement_vector_m(
                    records,
                    configurations[0],
                    transition_record,
                    edge.family,
                )
                if str(transition_record["moment_policy"])
                != "zero_motif_exchange"
                else np.zeros(CARTESIAN_DIMENSION, dtype=float),
            )
        )
        _validate_transition_displacement_policy(
            edge.family,
            transition_record,
            first_displacement_moment_m,
            second_displacement_moment_m2,
        )
        transitions.append(
            PhysicalTransitionQuadrature(
                from_state_index=edge.from_state_index,
                to_state_index=edge.to_state_index,
                transition_family=edge.family,
                transport_ownership=_transition_transport_ownership(
                    transition_record
                ),
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
                uses_residence_rate_constant=True,
                residence_rate_constant_s_inv=residence_rate_constant_s_inv,
            )
        )
    return tuple(transitions)


def _derived_transition_residence_rate_constant_s_inv(
    family: str,
    transition_record: dict,
    reaction_coordinate_grid: Array,
    diffusivity_profile_m2_s: Array,
    temperature_K: float,
) -> float:
    grid = np.asarray(reaction_coordinate_grid, dtype=float)
    diffusivity = np.asarray(diffusivity_profile_m2_s, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError(f"transition family {family} grid must be a finite 1D vector")
    if diffusivity.shape != grid.shape or not np.all(np.isfinite(diffusivity)):
        raise ValueError(
            f"transition family {family} diffusivity profile must match grid shape"
        )
    positive_diffusivity = diffusivity[diffusivity > 0.0]
    if positive_diffusivity.size == 0:
        raise ValueError(f"transition family {family} has no positive D_xi values")
    coordinate_span = _positive_float(
        float(np.max(grid) - np.min(grid)),
        f"transition family {family} coordinate span",
    )
    barrier_J_mol = _nonnegative_float(
        float(transition_record["barrier_J_mol"]),
        f"transition_records.{family}.barrier_J_mol",
    )
    thermal_energy_J_mol = R * _positive_float(temperature_K, "temperature_K")
    return _positive_float(
        float(np.min(positive_diffusivity))
        / (coordinate_span * coordinate_span)
        * float(np.exp(-barrier_J_mol / thermal_energy_J_mol)),
        f"transition_records.{family}.derived_residence_rate_constant_s_inv",
    )


def _transition_reaction_coordinate(transition_record: dict) -> ReducedCoordinate:
    reaction_coordinate_name = str(transition_record["reaction_coordinate"])
    for reduced_coordinate in ReducedCoordinate:
        if reduced_coordinate.value == reaction_coordinate_name:
            return reduced_coordinate
    raise ValueError(
        f"unsupported transition reaction_coordinate {reaction_coordinate_name}"
    )


def _transition_transport_ownership(transition_record: dict) -> TransportOwnership:
    return TransportOwnership(str(transition_record["transport_ownership"]))


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
        return _domain_bounds(
            domain, f"basis.coordinate_domains.{reaction_coordinate.value}"
        )
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
    if (
        pair_label != to_pair_label
        and {pair_label, to_pair_label}
        != {
            PairBasin.FREE.value,
            PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
        }
        and {pair_label, to_pair_label}
        != {
            PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
            PairBasin.CONTACT_ION_PAIR.value,
        }
    ):
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
        barrier_shape = (
            4.0 * reduced_coordinate_value * (1.0 - reduced_coordinate_value)
        )
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
        records,
        free_energy_terms,
        coordinate_values,
        from_state_label,
        to_state_label,
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
    normalized_coordinate = (li_anion_distance_m - solvent_separated_cutoff_m) / span_m
    linear_endpoint_energy = (1.0 - normalized_coordinate) * endpoint_energy_values[
        0
    ] + normalized_coordinate * endpoint_energy_values[1]
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


def _reduced_li_anion_feature_coordination_multiplier(
    records: PhysicalLibraryRecords,
    from_state_label: str,
    to_state_label: str,
) -> float:
    from_anion_name = _active_anion_name_from_state_key(
        _state_key_from_label(from_state_label)
    )
    to_anion_name = _active_anion_name_from_state_key(
        _state_key_from_label(to_state_label)
    )
    if from_anion_name == NO_ACTIVE_ANION_COMPONENT:
        if to_anion_name == NO_ACTIVE_ANION_COMPONENT:
            return 1.0
        return 1.0 + anion_internal_charge_separation_factor(records, to_anion_name)
    if to_anion_name == NO_ACTIVE_ANION_COMPONENT:
        return 1.0 + anion_internal_charge_separation_factor(records, from_anion_name)
    if from_anion_name != to_anion_name:
        raise ValueError(
            f"reduced Li-anion coordination changed active anion: {from_anion_name} -> {to_anion_name}"
        )
    return 1.0 + anion_internal_charge_separation_factor(records, from_anion_name)


def _reduced_coordination_free_energy_J_mol(
    records: PhysicalLibraryRecords,
    free_energy_terms: dict,
    coordinate_values: dict[str, float],
    from_state_label: str,
    to_state_label: str,
) -> float:
    coordination_record = free_energy_terms["coordination_J_mol"]
    coordinate_map = {
        "Li_solvent": ReducedCoordinate.LI_SOLVENT_COORDINATION.value,
        "Li_ligand": ReducedCoordinate.LI_LIGAND_COORDINATION.value,
        "Li_anion": ReducedCoordinate.LI_ANION_COORDINATION.value,
    }
    total_energy = 0.0
    for coordination_name, coordinate_name in coordinate_map.items():
        coordination_coefficient_J_mol = float(coordination_record[coordination_name])
        if coordination_name == "Li_anion":
            coordination_coefficient_J_mol *= (
                _reduced_li_anion_feature_coordination_multiplier(
                    records,
                    from_state_label,
                    to_state_label,
                )
            )
        total_energy += coordination_coefficient_J_mol * float(
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
            0.5 * float(stiffnesses[coordination_name]) * displacement * displacement
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
            0.5 * auxiliary_stiffness_J_mol * coordinate_value * coordinate_value
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
    coordinate_values = {
        **base_coordinate_values,
        reaction_coordinate.value: coordinate_value,
    }
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
    return builder(
        records, template_configuration, coordinate_values, transition_record
    )


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
    return _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )


def _configuration_with_structural_hop_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    transition_record: dict,
) -> SiteConfiguration:
    return _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )


def _configuration_with_partner_residence_coordinate(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
    transition_record: dict,
) -> SiteConfiguration:
    return _configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )


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
    if str(transition_record["displacement_policy"]) == "zero":
        return configuration
    cage_value = float(coordinate_values[ReducedCoordinate.CAGE_COORDINATE.value])
    displacement_m = (
        _endpoint_geometry_displacement_vector_m(
            records,
            configuration,
            transition_record,
            "cage_capture_release",
        )
        * cage_value
    )
    return _configuration_with_cation_displacement(
        records, configuration, displacement_m
    )


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
    displacement_m = (
        _endpoint_geometry_displacement_vector_m(
            records,
            configuration,
            transition_record,
            "bridge_network_formation_breakup",
        )
        * cluster_value
    )
    return _configuration_with_cation_displacement(
        records, configuration, displacement_m
    )


def _local_fields_for_coordinate_values(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
) -> PhysicalLocalFields:
    coordinate_packing_fraction = _nonnegative_float(
        float(coordinate_values[ReducedCoordinate.LOCAL_PACKING_FRACTION.value]),
        ReducedCoordinate.LOCAL_PACKING_FRACTION.value,
    )
    configuration_packing_fraction = compute_local_packing_fraction(
        records, configuration
    )
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
    phi_max = _positive_float(
        float(packing_record["phi_max"]), "mixture.packing.phi_max"
    )
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
    )
    return _positive_float(
        base_viscosity_Pa_s * salt_factor * packing_factor * additive_factor,
        "local_viscosity_Pa_s",
    )


def _configuration_additive_fraction(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    molecule_keys = tuple(
        dict.fromkeys(
            zip(
                configuration.species_names,
                np.asarray(configuration.molecule_ids, dtype=int),
                strict=True,
            )
        )
    )
    if not molecule_keys:
        raise ValueError("configuration has no sites")
    additive_weight = 0.0
    for species_name, _molecule_id in molecule_keys:
        if _species_role(records, species_name) != SpeciesRole.ADDITIVE:
            continue
        species_record = records.species_records[species_name]
        if "local_microviscosity_coefficient" not in species_record:
            raise KeyError(f"{species_name} missing local_microviscosity_coefficient")
        additive_weight += _nonnegative_float(
            float(species_record["local_microviscosity_coefficient"]),
            f"{species_name}.local_microviscosity_coefficient",
        )
    return additive_weight / float(len(molecule_keys))


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
    return gradient_functions[gradient_policy](
        records, configuration, transition_record
    )


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
    ligand_roles = tuple(
        SpeciesRole(str(role)) for role in switch_record["ligand_roles"]
    )
    switch_radius_m = _positive_float(
        float(switch_record["r0_m"]), f"{switch_name}.r0_m"
    )
    exponent = _positive_float(
        float(switch_record["exponent"]), f"{switch_name}.exponent"
    )
    center_index = _first_role_index(records, configuration, center_role)
    positions_m = np.asarray(configuration.positions_m, dtype=float)
    gradient = np.zeros(
        len(configuration.species_names) * CARTESIAN_DIMENSION, dtype=float
    )
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
    return (
        _cation_displacement_axis_gradient(
            records,
            configuration,
            transition_record,
            "identity_diffusion",
        )
        / displacement_m
    )


def _structural_hop_gradient_from_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
) -> Array:
    displacement_m = _endpoint_geometry_displacement_m(
        transition_record,
        "structural_hop",
    )
    return (
        _cation_displacement_axis_gradient(
            records,
            configuration,
            transition_record,
            "structural_hop",
        )
        / displacement_m
    )


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
        return (
            _cation_displacement_axis_gradient(
                records,
                configuration,
                transition_record,
                "cage_capture_release",
            )
            / displacement_norm_m
        )
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
    gradient = np.zeros(
        len(configuration.species_names) * CARTESIAN_DIMENSION, dtype=float
    )
    gradient[cation_index * CARTESIAN_DIMENSION] = 1.0
    return gradient


def _cation_displacement_axis_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
    family: str,
) -> Array:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    axis = _endpoint_geometry_direction_unit_vector_m(
        records,
        configuration,
        transition_record,
        family,
    )
    gradient = np.zeros(
        len(configuration.species_names) * CARTESIAN_DIMENSION, dtype=float
    )
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
        return float(
            np.linalg.norm(compute_charge_polarization_m(records, configuration))
        )
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
    orientation_vector = np.asarray(
        configuration.positions_m[orientation_index], dtype=float
    ) - np.asarray(configuration.positions_m[anchor_index], dtype=float)
    pair_vector = np.asarray(
        configuration.positions_m[anchor_index], dtype=float
    ) - np.asarray(configuration.positions_m[cation_index], dtype=float)
    orientation_norm = float(np.linalg.norm(orientation_vector))
    pair_norm = float(np.linalg.norm(pair_vector))
    if orientation_norm <= 0.0 or pair_norm <= 0.0:
        return 0.0
    return float(
        np.dot(orientation_vector, pair_vector) / (orientation_norm * pair_norm)
    )


def _transition_displacement_moments(
    transition_record: dict,
    moment_input: MomentBoundaryValueInput,
    endpoint_displacement_m: Array,
) -> tuple[Array, Array]:
    moment_policy = str(transition_record["moment_policy"])
    if moment_policy == "zero_motif_exchange":
        return _zero_transition_moments(moment_input)
    directed_policies = {
        "conditioned_endpoint_bvp": True,
        "isotropic_endpoint_bvp": False,
        "endpoint_geometry": True,
        "identity_diffusion": True,
    }
    if moment_policy not in directed_policies:
        raise ValueError(f"unsupported transition moment_policy {moment_policy}")
    endpoint_displacement = np.asarray(endpoint_displacement_m, dtype=float)
    return build_endpoint_transport_moments(
        EndpointTransportMomentInput(
            endpoint_displacement_m=endpoint_displacement,
            directed_endpoint=directed_policies[moment_policy],
        )
    )


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
        raise ValueError(
            f"transition family {family} first moment must have shape (3,)"
        )
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


def _zero_transition_moments(
    moment_input: MomentBoundaryValueInput,
) -> tuple[Array, Array]:
    _ = moment_input
    return np.zeros(CARTESIAN_DIMENSION, dtype=float), np.zeros(
        (CARTESIAN_DIMENSION, CARTESIAN_DIMENSION),
        dtype=float,
    )


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
    transport_ownership = _transition_transport_ownership(transition_record)
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
        "isotropic_endpoint_bvp",
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
        raise ValueError(
            f"transition family {family} has unsupported displacement policy"
        )
    _validate_transition_policy_geometry_consistency(
        family,
        transition_record,
        moment_policy,
        displacement_policy,
    )
    if transport_ownership is TransportOwnership.TRANSITION_DISPLACEMENT:
        if moment_policy == "zero_motif_exchange" or displacement_policy == "zero":
            raise ValueError(
                f"transition-normal family {family} must own a nonzero endpoint displacement"
            )
    elif moment_policy != "zero_motif_exchange" or displacement_policy != "zero":
        raise ValueError(
            f"{transport_ownership.value} family {family} cannot own transition moments"
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
        if moment_policy not in ("identity_diffusion", "isotropic_endpoint_bvp"):
            raise ValueError(
                f"identity-displacement transition family {family} must use "
                "identity_diffusion or isotropic_endpoint_bvp moment_policy"
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
        and moment_policy
        not in (
            "conditioned_endpoint_bvp",
            "isotropic_endpoint_bvp",
        )
    ):
        raise ValueError(
            f"charge-polarization transition family {family} must use "
            "conditioned_endpoint_bvp or isotropic_endpoint_bvp moment_policy"
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
        raise TypeError(
            f"transition family {family} endpoint_geometry must be a mapping"
        )
    start_record = endpoint_geometry["start"]
    end_record = endpoint_geometry["end"]
    displacement_record = endpoint_geometry["displacement"]
    if not isinstance(start_record, dict):
        raise TypeError(
            f"transition family {family} endpoint_geometry.start must be a mapping"
        )
    if not isinstance(end_record, dict):
        raise TypeError(
            f"transition family {family} endpoint_geometry.end must be a mapping"
        )
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
        "partner_switch": (
            ("Li_partner", "Li_position"),
            ("Li_partner", "Li_position"),
        ),
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
        raise KeyError(
            f"transition family {family} missing endpoint_geometry.displacement"
        )
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
    if family == "partner_switch":
        return _endpoint_geometry_displacement_m(
            transition_record,
            family,
        ) * _endpoint_geometry_direction_unit_vector_m(
            records,
            configuration,
            transition_record,
            family,
        )
    if family == "identity_diffusion":
        return _same_role_molecule_center_displacement_m(
            records,
            configuration,
            SpeciesRole.CATION,
        )
    if family == "bridge_network_formation_breakup":
        return _cation_to_anion_bridge_midpoint_displacement_m(
            records,
            configuration,
        )
    length_m = _endpoint_geometry_displacement_m(transition_record, family)
    return length_m * _endpoint_geometry_direction_unit_vector_m(
        records,
        configuration,
        transition_record,
        family,
    )


def _same_role_molecule_center_displacement_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
) -> Array:
    molecule_keys = _charged_role_molecule_keys(records, configuration, role)
    if len(molecule_keys) < 2:
        raise ValueError(
            f"{role.value} identity displacement requires two distinct molecules"
        )
    first_species_name, first_molecule_id = molecule_keys[0]
    second_species_name, second_molecule_id = molecule_keys[1]
    first_center_m = molecule_center_of_mass_m(
        records,
        configuration,
        first_species_name,
        first_molecule_id,
    )
    second_center_m = molecule_center_of_mass_m(
        records,
        configuration,
        second_species_name,
        second_molecule_id,
    )
    return _minimum_image_vector_m(
        first_center_m,
        second_center_m,
        configuration.box_lengths_m,
    )


def _site_indices_for_configuration_molecule(
    configuration: SiteConfiguration,
    species_name: str,
    molecule_id: int,
) -> tuple[int, ...]:
    site_indices = tuple(
        site_index
        for site_index, current_species_name in enumerate(configuration.species_names)
        if current_species_name == species_name
        and int(configuration.molecule_ids[site_index]) == molecule_id
    )
    if not site_indices:
        raise ValueError("configuration molecule has no sites")
    return site_indices


def _cation_to_anion_bridge_midpoint_displacement_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    anion_molecule_keys = _charged_role_molecule_keys(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    if len(anion_molecule_keys) < 2:
        raise ValueError("bridge displacement requires two distinct anion molecules")
    first_anion_species, first_anion_molecule_id = anion_molecule_keys[0]
    second_anion_species, second_anion_molecule_id = anion_molecule_keys[1]
    first_anion_center_m = molecule_center_of_mass_m(
        records,
        configuration,
        first_anion_species,
        first_anion_molecule_id,
    )
    second_anion_center_m = molecule_center_of_mass_m(
        records,
        configuration,
        second_anion_species,
        second_anion_molecule_id,
    )
    first_to_second_m = _minimum_image_vector_m(
        first_anion_center_m,
        second_anion_center_m,
        configuration.box_lengths_m,
    )
    bridge_midpoint_m = first_anion_center_m + 0.5 * first_to_second_m
    cation_species, cation_molecule_id = _charged_role_molecule_keys(
        records,
        configuration,
        SpeciesRole.CATION,
    )[0]
    cation_center_m = molecule_center_of_mass_m(
        records,
        configuration,
        cation_species,
        cation_molecule_id,
    )
    return _minimum_image_vector_m(
        cation_center_m,
        bridge_midpoint_m,
        configuration.box_lengths_m,
    )


def _charged_role_molecule_keys(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
) -> tuple[tuple[str, int], ...]:
    molecule_keys = []
    for site_index, species_name in enumerate(configuration.species_names):
        if _species_role(records, species_name) != role:
            continue
        molecule_key = (species_name, int(configuration.molecule_ids[site_index]))
        if molecule_key not in molecule_keys:
            molecule_keys.append(molecule_key)
    if not molecule_keys:
        raise ValueError(f"configuration has no charged {role.value} molecules")
    return tuple(molecule_keys)


def _endpoint_geometry_direction_unit_vector_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    transition_record: dict,
    family: str,
) -> Array:
    endpoint_geometry = transition_record["endpoint_geometry"]
    displacement_record = endpoint_geometry["displacement"]
    direction_policy = str(displacement_record["direction_policy"])
    if direction_policy in ("pair_axis", "cluster_axis", "cage_axis"):
        return _endpoint_geometry_charge_axis_unit_vector_m(
            records,
            configuration,
            family,
            direction_policy,
        )
    if direction_policy == "identity_axis":
        return _endpoint_geometry_charge_axis_unit_vector_m(
            records,
            configuration,
            family,
            direction_policy,
        )
    if direction_policy == "hop_axis":
        return _structural_hop_unit_vector(records, configuration)
    raise ValueError(f"unsupported endpoint direction policy {direction_policy}")


def _endpoint_geometry_charge_axis_unit_vector_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    family: str,
    direction_policy: str,
) -> Array:
    if family == "identity_diffusion":
        axis = _same_charge_transport_axis_m(records, configuration, SpeciesRole.CATION)
    elif family == "partner_switch":
        axis = _partner_switch_transport_axis_m(records, configuration)
    elif family == "bridge_network_formation_breakup":
        axis = _same_charge_transport_axis_m(records, configuration, SpeciesRole.ANION)
    elif family == "cage_capture_release":
        axis = -_cation_to_anion_axis_m(records, configuration)
    elif direction_policy in ("pair_axis", "cluster_axis", "cage_axis"):
        axis = _cation_to_anion_axis_m(records, configuration)
    else:
        raise ValueError(f"transition family {family} has no endpoint geometry axis")
    norm = _positive_float(float(np.linalg.norm(axis)), "endpoint geometry axis norm")
    return np.asarray(axis, dtype=float) / norm


def _cation_to_anion_axis_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    cation_index = _charged_role_site_index(records, configuration, SpeciesRole.CATION)
    anion_indices = _first_molecule_indices_with_role(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    anion_index = _molecule_coordination_site_index(
        records,
        configuration,
        anion_indices,
    )
    return _minimum_image_vector_m(
        configuration.positions_m[cation_index],
        configuration.positions_m[anion_index],
        configuration.box_lengths_m,
    )


def _partner_switch_transport_axis_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    cation_index = _charged_role_site_index(records, configuration, SpeciesRole.CATION)
    anion_indices = _charged_role_site_indices(
        records, configuration, SpeciesRole.ANION
    )
    if len(anion_indices) >= 2:
        return _nearest_distinct_role_site_axis_m(
            configuration,
            anion_indices[0],
            anion_indices[1:],
        )
    return _minimum_image_vector_m(
        configuration.positions_m[cation_index],
        configuration.positions_m[anion_indices[0]],
        configuration.box_lengths_m,
    )


def _same_charge_transport_axis_m(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
) -> Array:
    charged_indices = _charged_role_site_indices(records, configuration, role)
    if len(charged_indices) >= 2:
        return _nearest_distinct_role_site_axis_m(
            configuration,
            charged_indices[0],
            charged_indices[1:],
        )
    return _nearest_periodic_image_axis_m(configuration, charged_indices[0])


def _nearest_distinct_role_site_axis_m(
    configuration: SiteConfiguration,
    reference_site_index: int,
    candidate_site_indices: tuple[int, ...],
) -> Array:
    nearest_site_index = candidate_site_indices[0]
    nearest_distance_m = _site_distance_m(
        configuration,
        reference_site_index,
        nearest_site_index,
    )
    for candidate_site_index in candidate_site_indices[1:]:
        candidate_distance_m = _site_distance_m(
            configuration,
            reference_site_index,
            candidate_site_index,
        )
        if candidate_distance_m < nearest_distance_m:
            nearest_site_index = candidate_site_index
            nearest_distance_m = candidate_distance_m
    return _minimum_image_vector_m(
        configuration.positions_m[reference_site_index],
        configuration.positions_m[nearest_site_index],
        configuration.box_lengths_m,
    )


def _nearest_periodic_image_axis_m(
    configuration: SiteConfiguration,
    reference_site_index: int,
) -> Array:
    _ = reference_site_index
    box_lengths = np.asarray(configuration.box_lengths_m, dtype=float)
    if box_lengths.shape != (CARTESIAN_DIMENSION,):
        raise ValueError("box_lengths_m must have shape (3,)")
    axis_index = int(np.argmin(box_lengths))
    axis = np.zeros(CARTESIAN_DIMENSION, dtype=float)
    axis[axis_index] = float(box_lengths[axis_index])
    return axis


def _site_distance_m(
    configuration: SiteConfiguration,
    first_site_index: int,
    second_site_index: int,
) -> float:
    displacement = _minimum_image_vector_m(
        configuration.positions_m[first_site_index],
        configuration.positions_m[second_site_index],
        configuration.box_lengths_m,
    )
    return float(np.linalg.norm(displacement))


def _charged_role_site_index(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
) -> int:
    return _charged_role_site_indices(records, configuration, role)[0]


def _charged_role_site_indices(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
) -> tuple[int, ...]:
    charges = _configuration_charge_numbers(records, configuration)
    indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if _species_role(records, species_name) == role
        and abs(charges[site_index]) > 0.0
    )
    if not indices:
        raise ValueError(f"configuration has no charged {role.value} sites")
    return indices


def _structural_hop_unit_vector(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    pair_axis = _endpoint_geometry_charge_axis_unit_vector_m(
        records,
        configuration,
        "structural_hop",
        "pair_axis",
    )
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
    mobility_cache_m2_s: dict[tuple, Array] = {}
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
            mobility_cache_m2_s,
        )
        forward_rate_s_inv = float(
            generator[edge.from_state_index, edge.to_state_index]
        )
        reverse_rate_s_inv = float(
            generator[edge.to_state_index, edge.from_state_index]
        )
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
    mobility_cache_m2_s: dict[tuple, Array],
) -> tuple[float, float]:
    if _uses_declared_rate_constant(family, transition_record):
        return _transition_rate_bounds_s_inv(transition_record, family)
    projected_diffusivities = _transition_projected_diffusivity_profile(
        records,
        transition_record,
        transition_quadrature,
        temperature_K,
        mobility_cache_m2_s,
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
    mobility_cache_m2_s: dict[tuple, Array],
) -> Array:
    projected_diffusivities = []
    for configuration, local_fields in zip(
        transition_quadrature.configurations,
        transition_quadrature.local_fields,
        strict=True,
    ):
        cache_key = (
            configuration.species_names,
            tuple(np.asarray(configuration.molecule_ids, dtype=int)),
            tuple(np.asarray(configuration.site_ids, dtype=int)),
            tuple(np.asarray(configuration.positions_m, dtype=float).reshape(-1)),
            local_fields,
        )
        if cache_key not in mobility_cache_m2_s:
            mobility_cache_m2_s[cache_key] = build_physical_objects(
                    records,
                    configuration,
                    temperature_K,
                    local_fields.dielectric_constant,
                    local_fields.viscosity_Pa_s,
                    local_fields.ionic_strength_mol_m3,
                    local_fields.local_packing_fraction,
                ).mobility_tensor_m2_s
        projected_diffusivities.append(
            project_diffusivity_onto_reaction_coordinate(
                mobility_cache_m2_s[cache_key],
                _reaction_coordinate_gradient(records, configuration, transition_record),
            )
        )
    return np.asarray(projected_diffusivities, dtype=float)


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
    generator = np.asarray(
        conductivity_result.reversible_generator_Q_ij_s_inv, dtype=float
    )
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
    state_lifetimes_s[active_state_mask] = (
        1.0 / state_exit_rates_s_inv[active_state_mask]
    )
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
    edge_first_moment_vectors_m = np.asarray(
        [
            first_moments[edge.from_state_index, edge.to_state_index]
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
            _transition_coordinate_span(
                records, transition_record, transition_quadrature
            )
            for transition_record, transition_quadrature in zip(
                transition_records,
                transition_quadratures,
                strict=True,
            )
        ],
        dtype=float,
    )
    mobility_cache_m2_s: dict[tuple, Array] = {}
    projected_diffusivity_profiles = tuple(
        _transition_projected_diffusivity_profile(
            records,
            transition_record,
            transition_quadrature,
            temperature_K,
            mobility_cache_m2_s,
        )
        for transition_record, transition_quadrature in zip(
            transition_records,
            transition_quadratures,
            strict=True,
        )
    )
    edge_projected_diffusivity_min = np.asarray(
        [float(np.min(profile)) for profile in projected_diffusivity_profiles],
        dtype=float,
    )
    edge_projected_diffusivity_max = np.asarray(
        [float(np.max(profile)) for profile in projected_diffusivity_profiles],
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
    edge_is_conductivity_carrying = _transition_edge_conductivity_carrying_mask(
        transition_records
    )
    edge_is_active_conductivity_carrying = edge_is_conductivity_carrying & (
        edge_capacity_fluxes > 0.0
    )
    active_conductivity_carrying_second_moment_mask = (
        edge_is_active_conductivity_carrying & (edge_second_moment_traces_m2 > 0.0)
    )
    active_conductivity_carrying_zero_second_moment_edges = (
        _active_conductivity_carrying_zero_second_moment_edges(
            transition_edges,
            edge_is_active_conductivity_carrying,
            edge_second_moment_traces_m2,
        )
    )
    edge_inactive_reasons = _transition_edge_inactive_reasons(
        transition_records,
        edge_capacity_fluxes,
        edge_forward_rates_s_inv,
        edge_reverse_rates_s_inv,
        edge_first_moment_norms_m,
        edge_second_moment_traces_m2,
        edge_direct_trace_contribution,
    )
    _update_finite_process_transition_moment_completion(
        conductivity_result,
        int(np.count_nonzero(edge_is_active_conductivity_carrying)),
        int(np.count_nonzero(active_conductivity_carrying_second_moment_mask)),
        active_conductivity_carrying_zero_second_moment_edges,
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
        edge_capacity_fluxes,
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
    transition_activity_ledger = tuple(
        {
            "family": edge.family,
            "K": float(edge_capacity_fluxes[edge_index]),
            "Q": float(edge_forward_rates_s_inv[edge_index]),
            "endpoint_displacement_m": edge_first_moment_vectors_m[edge_index].copy(),
            "TrM": float(edge_second_moment_traces_m2[edge_index]),
            "KTrM": float(edge_direct_trace_contribution[edge_index]),
            "commitment_policy": {
                "moment_policy": str(transition_records[edge_index]["moment_policy"]),
                "displacement_policy": str(
                    transition_records[edge_index]["displacement_policy"]
                ),
            },
            "ownership": _transition_transport_ownership(
                transition_records[edge_index]
            ).value,
        }
        for edge_index, edge in enumerate(transition_edges)
    )
    _validate_transition_activity_ledger(transition_activity_ledger)
    conductivity_result.effect_attribution.update(
        {
            "transition_activity_ledger": transition_activity_ledger,
            "transition_edge_families": tuple(edge.family for edge in transition_edges),
            "transition_edge_from_state_indices": edge_from_indices,
            "transition_edge_to_state_indices": edge_to_indices,
            "transition_edge_capacity_fluxes_K_ij_mol_m3_s": edge_capacity_fluxes,
            "transition_edge_forward_rates_Q_ij_s_inv": edge_forward_rates_s_inv,
            "transition_edge_reverse_rates_Q_ji_s_inv": edge_reverse_rates_s_inv,
            "transition_edge_first_moment_norms_m": edge_first_moment_norms_m,
            "transition_edge_first_moment_vectors_m": edge_first_moment_vectors_m,
            "transition_edge_second_moment_traces_m2": edge_second_moment_traces_m2,
            "transition_edge_K_trace_M_mol_m5_s": edge_direct_trace_contribution,
            "transition_edge_inactive_reasons": edge_inactive_reasons,
            "transition_edge_is_conductivity_carrying": edge_is_conductivity_carrying,
            "transition_edge_is_active_conductivity_carrying": (
                edge_is_active_conductivity_carrying
            ),
            "active_conductivity_carrying_transition_count": int(
                np.count_nonzero(edge_is_active_conductivity_carrying)
            ),
            "active_conductivity_carrying_transition_second_moment_count": int(
                np.count_nonzero(active_conductivity_carrying_second_moment_mask)
            ),
            "active_conductivity_carrying_zero_second_moment_edges": (
                active_conductivity_carrying_zero_second_moment_edges
            ),
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
                np.asarray(
                    conductivity_result.state_concentrations_mol_m3, dtype=float
                )[:, np.newaxis]
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


def _validate_transition_activity_ledger(transition_activity_ledger: tuple[dict, ...]) -> None:
    for edge_record in transition_activity_ledger:
        ownership = TransportOwnership(str(edge_record["ownership"]))
        capacity_flux = float(edge_record["K"])
        forward_rate = float(edge_record["Q"])
        second_moment_trace = float(edge_record["TrM"])
        direct_contribution = float(edge_record["KTrM"])
        family = str(edge_record["family"])
        if ownership is TransportOwnership.TRANSITION_DISPLACEMENT and (
            capacity_flux > 0.0 or forward_rate > 0.0
        ):
            if capacity_flux <= 0.0 or forward_rate <= 0.0 or second_moment_trace <= 0.0:
                raise ValueError(
                    "active transition_displacement edge lacks positive committed "
                    f"transport: family={family}, K={capacity_flux}, Q={forward_rate}, "
                    f"TrM={second_moment_trace}"
                )
        if ownership is TransportOwnership.DIAGNOSTIC and (
            second_moment_trace != 0.0 or direct_contribution != 0.0
        ):
            raise ValueError(
                f"diagnostic-owned transition family {family} contributes K*M"
            )


def _validate_state_transport_owner_closure(
    conductivity_result: ProjectedConductivityResult,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
) -> None:
    concentrations = np.asarray(
        conductivity_result.state_concentrations_mol_m3, dtype=float
    )
    ownership_records = tuple(
        conductivity_result.effect_attribution["transport_ownership_state_tensors"]
    )
    consumed_bounded_modes = set(
        np.asarray(
            conductivity_result.effect_attribution[
                "mori_declared_bounded_memory_mode_indices"
            ],
            dtype=int,
        ).tolist()
    )
    if len(ownership_records) != len(state_quadratures):
        raise ValueError("state transport ownership ledger length mismatch")
    self_only_states = []
    for state_index, (state_quadrature, ownership_record) in enumerate(
        zip(state_quadratures, ownership_records, strict=True)
    ):
        if concentrations[state_index] <= 0.0:
            continue
        short_tensor = np.asarray(ownership_record["D_Q_short"], dtype=float)
        transition_tensor = np.asarray(
            ownership_record["D_Q_transition_owned"], dtype=float
        )
        bounded_tensor = np.asarray(
            ownership_record["D_Q_bounded_memory"], dtype=float
        )
        diagnostic_tensor = np.asarray(ownership_record["D_Q_diagnostic"], dtype=float)
        unowned_tensor = np.asarray(ownership_record["D_Q_unowned"], dtype=float)
        final_self_tensor = np.asarray(
            conductivity_result.self_current_tensors_D_self_i_m2_s, dtype=float
        )[state_index]
        tangent_self_tensor = np.asarray(
            ownership_record["D_Q_dc_self"], dtype=float
        )
        ownership_scale = max(
            float(np.linalg.norm(short_tensor, ord=2)),
            np.finfo(float).tiny,
        )
        ownership_subtraction_scale = ownership_scale + sum(
            float(np.linalg.norm(owner_tensor, ord=2))
            for owner_tensor in (
                tangent_self_tensor,
                transition_tensor,
                bounded_tensor,
                diagnostic_tensor,
            )
        )
        maximum_coordinate_support_rank = max(
            point_tensor.coordinate_support_rank
            for point_tensor in conductivity_result.state_transport_ownership_quadratures[
                state_index
            ].point_tensors
        )
        ownership_tolerance = max(
            len(state_quadrature.configurations)
            * maximum_coordinate_support_rank
            * np.sqrt(np.finfo(float).eps)
            * ownership_subtraction_scale,
            float(
                conductivity_result.state_transport_ownership_quadratures[
                    state_index
                ].maximum_closure_residual_m2_s
            ),
        )
        inclusive_ownership_tolerance = ownership_tolerance * (
            1.0 + np.sqrt(np.finfo(float).eps)
        )
        if float(
            np.linalg.norm(
                final_self_tensor - tangent_self_tensor - bounded_tensor,
                ord=2,
            )
        ) > inclusive_ownership_tolerance:
            raise ValueError(
                f"state {state_quadrature.label} D_self violates owner closure"
            )
        if (
            float(np.linalg.norm(diagnostic_tensor, ord=2))
            > inclusive_ownership_tolerance
        ):
            raise ValueError("diagnostic-owned state transport contributes D_self")
        if (
            float(np.linalg.norm(unowned_tensor, ord=2))
            > inclusive_ownership_tolerance
        ):
            raise ValueError(
                "populated state has unowned short-time transport: "
                f"{state_quadrature.label}; "
                f"unowned_norm={float(np.linalg.norm(unowned_tensor, ord=2)):.9g}; "
                f"short_norm={float(np.linalg.norm(short_tensor, ord=2)):.9g}; "
                f"tolerance={ownership_tolerance:.9g}"
            )
        required_bounded_modes = {
            int(mode_index)
            for basis in state_quadrature.transport_ownership_bases
            for mode_index in np.asarray(basis.bounded_memory_mode_indices, dtype=int)
        }
        missing_bounded_modes = required_bounded_modes - consumed_bounded_modes
        if missing_bounded_modes:
            raise ValueError(
                f"populated state {state_quadrature.label} has bounded-memory owners "
                f"absent from A/h: {tuple(sorted(missing_bounded_modes))}"
            )
        if (
            float(np.linalg.norm(short_tensor, ord=2)) > ownership_tolerance
            and float(np.linalg.norm(transition_tensor, ord=2)) <= ownership_tolerance
            and float(np.linalg.norm(bounded_tensor, ord=2)) <= ownership_tolerance
        ):
            self_only_states.append(state_quadrature.label)
    conductivity_result.effect_attribution[
        "populated_nonzero_short_DQ_self_only_state_labels"
    ] = tuple(self_only_states)


def _transition_edge_conductivity_carrying_mask(
    transition_records: tuple[dict, ...],
) -> Array:
    return np.asarray(
        [
            str(transition_record["displacement_policy"]) != "zero"
            for transition_record in transition_records
        ],
        dtype=bool,
    )


def _active_conductivity_carrying_zero_second_moment_edges(
    transition_edges: tuple[TransitionEdge, ...],
    edge_is_active_conductivity_carrying: Array,
    edge_second_moment_traces_m2: Array,
) -> tuple[str, ...]:
    active_mask = np.asarray(edge_is_active_conductivity_carrying, dtype=bool)
    second_moment_traces = np.asarray(edge_second_moment_traces_m2, dtype=float)
    return tuple(
        (
            f"{edge.family}:{edge.from_state_index}->{edge.to_state_index}:"
            "second_moment_trace_m2_nonpositive"
        )
        for edge_index, edge in enumerate(transition_edges)
        if active_mask[edge_index] and second_moment_traces[edge_index] <= 0.0
    )


def _update_finite_process_transition_moment_completion(
    conductivity_result: ProjectedConductivityResult,
    active_conductivity_carrying_transition_count: int,
    active_conductivity_carrying_transition_second_moment_count: int,
    active_conductivity_carrying_zero_second_moment_edges: tuple[str, ...],
) -> None:
    if active_conductivity_carrying_transition_count <= 0:
        return
    if active_conductivity_carrying_zero_second_moment_edges:
        return
    if (
        active_conductivity_carrying_transition_second_moment_count
        != active_conductivity_carrying_transition_count
    ):
        raise ValueError(
            "active conductivity-carrying transition moment count is inconsistent with "
            "zero-second-moment edge diagnostics"
        )
    if (
        "finite_process_not_complete_reasons"
        not in conductivity_result.effect_attribution
    ):
        raise KeyError("effect_attribution missing finite_process_not_complete_reasons")
    existing_reasons = tuple(
        str(reason)
        for reason in conductivity_result.effect_attribution[
            "finite_process_not_complete_reasons"
        ]
    )
    filtered_reasons = tuple(
        reason
        for reason in existing_reasons
        if reason != "active_transitions_have_zero_second_moments"
    )
    conductivity_result.effect_attribution["finite_process_not_complete_reasons"] = (
        filtered_reasons
    )
    if "finite_process_readout_status" in conductivity_result.effect_attribution:
        conductivity_result.effect_attribution.update(
            primitive_prediction_readiness_as_effect_attribution(
                conductivity_result.effect_attribution,
            )
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
    edge_capacity_fluxes_mol_m3_s: Array,
    component_state_indices: tuple[tuple[int, ...], ...],
    state_count: int,
) -> Array:
    capacity_fluxes = np.asarray(edge_capacity_fluxes_mol_m3_s, dtype=float)
    if capacity_fluxes.shape != (len(transition_edges),):
        raise ValueError("edge capacity flux count does not match transition edges")
    state_component_indices = np.full(state_count, -1, dtype=int)
    for component_index, component_indices in enumerate(component_state_indices):
        for state_index in component_indices:
            state_component_indices[state_index] = component_index
    edge_component_indices = np.full(len(transition_edges), -1, dtype=int)
    for edge_index, (edge, capacity_flux) in enumerate(
        zip(
            transition_edges,
            capacity_fluxes,
            strict=True,
        )
    ):
        if capacity_flux == 0.0:
            continue
        component_index = state_component_indices[edge.from_state_index]
        if component_index < 0:
            raise ValueError("transition edge references a state without component")
        if state_component_indices[edge.to_state_index] != component_index:
            raise ValueError("transition edge crosses generator connected components")
        edge_component_indices[edge_index] = component_index
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


def _generator_connected_components_for_diagnostics(
    generator_Q_ij_s_inv: Array,
) -> tuple[Array, ...]:
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
        if float(edge_capacity_fluxes[edge_index]) == 0.0:
            continue
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
    if first_moment_norm_m <= 0.0 and second_moment_trace_m2 <= 0.0:
        return "first_and_second_moments_nonpositive"
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
                    diagnostic.cation_anion_center_mobility_m2_s
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
            "state_charged_center_pair_covariance_entries": tuple(
                diagnostic.charged_center_pair_covariance_entries
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
            "state_resistance_stokes_traces_kg_s": np.asarray(
                [diagnostic.resistance_stokes_trace_kg_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_resistance_free_volume_traces_kg_s": np.asarray(
                [
                    diagnostic.resistance_free_volume_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_resistance_charge_cloud_traces_kg_s": np.asarray(
                [
                    diagnostic.resistance_charge_cloud_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_resistance_atmosphere_traces_kg_s": np.asarray(
                [
                    diagnostic.resistance_atmosphere_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_resistance_cage_constraint_traces_kg_s": np.asarray(
                [diagnostic.resistance_cage_constraint_trace_kg_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_resistance_ligand_shell_obstruction_traces_kg_s": np.asarray(
                [
                    diagnostic.resistance_ligand_shell_obstruction_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_resistance_aggregate_constraint_traces_kg_s": np.asarray(
                [
                    diagnostic.resistance_aggregate_constraint_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_resistance_bridge_constraint_traces_kg_s": np.asarray(
                [diagnostic.resistance_bridge_constraint_trace_kg_s for diagnostic in diagnostics],
                dtype=float,
            ),
            "state_resistance_orientation_denticity_traces_kg_s": np.asarray(
                [
                    diagnostic.resistance_orientation_denticity_trace_kg_s
                    for diagnostic in diagnostics
                ],
                dtype=float,
            ),
            "state_resistance_total_traces_kg_s": np.asarray(
                [diagnostic.resistance_total_trace_kg_s for diagnostic in diagnostics],
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
    weighted_center_mobility_matrix = _weighted_center_mobility_matrix(
        normalized_weights,
        point_diagnostics,
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
        charged_center_charge_numbers=point_diagnostics[
            0
        ].charged_center_charge_numbers,
        charged_center_mobility_matrix_m2_s=_matrix_to_nested_tuple(
            weighted_center_mobility_matrix
        ),
        charged_center_pair_covariance_entries=_charged_center_pair_covariance_entries(
            state_quadrature.label,
            point_diagnostics[0].charged_center_labels,
            point_diagnostics[0].charged_center_charge_numbers,
            weighted_center_mobility_matrix,
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
        resistance_stokes_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.resistance_stokes_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        resistance_free_volume_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.resistance_free_volume_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        resistance_charge_cloud_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.resistance_charge_cloud_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        resistance_atmosphere_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.resistance_atmosphere_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        resistance_cage_constraint_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [diagnostic.resistance_cage_constraint_trace_kg_s for diagnostic in point_diagnostics],
            )
        ),
        resistance_ligand_shell_obstruction_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.resistance_ligand_shell_obstruction_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        resistance_aggregate_constraint_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.resistance_aggregate_constraint_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        resistance_bridge_constraint_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [diagnostic.resistance_bridge_constraint_trace_kg_s for diagnostic in point_diagnostics],
            )
        ),
        resistance_orientation_denticity_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.resistance_orientation_denticity_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
            )
        ),
        resistance_total_trace_kg_s=float(
            np.dot(
                normalized_weights,
                [
                    diagnostic.resistance_total_trace_kg_s
                    for diagnostic in point_diagnostics
                ],
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


def _state_relative_displacement_descriptor(
    records: PhysicalLibraryRecords,
    configurations: tuple[SiteConfiguration, ...],
    local_fields: tuple[PhysicalLocalFields, ...],
    quadrature_weights: Array,
    temperature_K: float,
    active_anion_name: str,
    state_key: tuple[str, ...],
) -> tuple[Array, Array, Array]:
    if state_key[STATE_KEY_PAIR_INDEX] == PairBasin.FREE.value:
        return (
            np.empty((0, CARTESIAN_DIMENSION), dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty(0, dtype=float),
        )
    point_terms = tuple(
        _relative_displacement_point_terms(
            records,
            configuration,
            current_local_fields,
            temperature_K,
            active_anion_name,
        )
        for configuration, current_local_fields in zip(
            configurations, local_fields, strict=True
        )
    )
    if any(term[2].size == 0 for term in point_terms):
        return (
            np.empty((0, CARTESIAN_DIMENSION), dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty(0, dtype=float),
        )
    normalized_weights = _normalized_boltzmann_quadrature_weights(
        quadrature_weights,
        tuple(float(term[3]) for term in point_terms),
        temperature_K,
    )
    positions = np.asarray([term[0] for term in point_terms], dtype=float)
    fluctuations = positions - np.einsum("i,ia->a", normalized_weights, positions)
    relative_mobility = np.einsum(
        "i,iab->ab",
        normalized_weights,
        np.asarray([term[1] for term in point_terms], dtype=float),
    )
    weighted_fluctuations = np.sqrt(normalized_weights)[:, None] * fluctuations
    return (
        weighted_fluctuations,
        0.5 * (relative_mobility + relative_mobility.T),
        point_terms[0][2],
    )


def _relative_displacement_point_terms(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    local_fields: PhysicalLocalFields,
    temperature_K: float,
    active_anion_name: str,
) -> tuple[Array, Array, Array, float]:
    cation_centers = _molecular_charge_centers_for_role(
        records, configuration, SpeciesRole.CATION, ""
    )
    anion_centers = _molecular_charge_centers_for_role(
        records, configuration, SpeciesRole.ANION, active_anion_name
    )
    if len(cation_centers) != 1 or len(anion_centers) != 1:
        return (
            np.empty(0, dtype=float),
            np.empty((0, 0), dtype=float),
            np.empty(0, dtype=float),
            0.0,
        )
    physical_objects = build_physical_objects(
        records,
        configuration,
        temperature_K,
        local_fields.dielectric_constant,
        local_fields.viscosity_Pa_s,
        local_fields.ionic_strength_mol_m3,
        local_fields.local_packing_fraction,
    )
    cation_projection = _molecular_center_projection(
        cation_centers[0], len(configuration.species_names)
    )
    anion_projection = _molecular_center_projection(
        anion_centers[0], len(configuration.species_names)
    )
    relative_projection = cation_projection - anion_projection
    return (
        _molecular_center_position(configuration, cation_centers[0])
        - _molecular_center_position(configuration, anion_centers[0]),
        relative_projection
        @ np.asarray(physical_objects.mobility_tensor_m2_s, dtype=float)
        @ relative_projection.T,
        np.asarray(
            [
                cation_centers[0].formal_charge_number,
                anion_centers[0].formal_charge_number,
            ],
            dtype=float,
        ),
        float(physical_objects.potential_energy_J_mol),
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
    cation_centers = _molecular_charge_centers_for_role(
        records,
        configuration,
        SpeciesRole.CATION,
        "",
    )
    if active_anion_name == NO_ACTIVE_ANION_COMPONENT:
        anion_centers = ()
    else:
        anion_centers = _molecular_charge_centers_for_role(
            records,
            configuration,
            SpeciesRole.ANION,
            active_anion_name,
        )
    charged_centers = cation_centers + anion_centers
    charged_center_charge_numbers = tuple(
        center.formal_charge_number for center in charged_centers
    )
    charged_center_mobility_matrix = _charged_center_mobility_matrix(
        mobility,
        len(configuration.species_names),
        charged_centers,
    )
    charge_mobility_m2_s = charge_covariance_mobility_from_center_matrix(
        charged_center_charge_numbers,
        charged_center_mobility_matrix,
    )
    cation_center_indices = tuple(range(len(cation_centers)))
    anion_center_indices = tuple(
        range(len(cation_centers), len(cation_centers) + len(anion_centers))
    )
    atmosphere_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        configuration,
        local_fields.dielectric_constant,
        local_fields.ionic_strength_mol_m3,
        temperature_K,
        local_fields.viscosity_Pa_s,
    )
    resistance_diagnostics = compute_resistance_component_diagnostics(
        records,
        configuration,
        local_fields.viscosity_Pa_s,
        local_fields.dielectric_constant,
        local_fields.ionic_strength_mol_m3,
        temperature_K,
        local_fields.local_packing_fraction,
    )
    return StateChargeMobilityDiagnostics(
        charge_mobility_m2_s=charge_mobility_m2_s,
        cation_mobility_m2_s=_charged_center_subset_pair_scalar_mobility(
            charged_center_mobility_matrix,
            charged_center_charge_numbers,
            cation_center_indices,
            cation_center_indices,
        ),
        anion_mobility_m2_s=_charged_center_subset_pair_scalar_mobility(
            charged_center_mobility_matrix,
            charged_center_charge_numbers,
            anion_center_indices,
            anion_center_indices,
        ),
        cation_anion_cross_mobility_m2_s=_charged_center_subset_pair_scalar_mobility(
            charged_center_mobility_matrix,
            charged_center_charge_numbers,
            cation_center_indices,
            anion_center_indices,
        ),
        cation_anion_center_mobility_m2_s=_center_subset_pair_scalar_mobility(
            charged_center_mobility_matrix,
            cation_center_indices,
            anion_center_indices,
        ),
        charged_center_labels=tuple(center.label for center in charged_centers),
        charged_center_charge_numbers=charged_center_charge_numbers,
        charged_center_mobility_matrix_m2_s=_matrix_to_nested_tuple(
            charged_center_mobility_matrix
        ),
        charged_center_pair_covariance_entries=_charged_center_pair_covariance_entries(
            "",
            tuple(center.label for center in charged_centers),
            charged_center_charge_numbers,
            charged_center_mobility_matrix,
        ),
        potential_energy_J_mol=float(physical_objects.potential_energy_J_mol),
        dielectric_constant=float(local_fields.dielectric_constant),
        viscosity_Pa_s=float(local_fields.viscosity_Pa_s),
        ionic_strength_mol_m3=float(local_fields.ionic_strength_mol_m3),
        local_packing_fraction=float(local_fields.local_packing_fraction),
        resistance_stokes_trace_kg_s=float(resistance_diagnostics.stokes_trace_kg_s),
        resistance_free_volume_trace_kg_s=float(
            resistance_diagnostics.free_volume_trace_kg_s
        ),
        resistance_charge_cloud_trace_kg_s=float(
            resistance_diagnostics.charge_cloud_trace_kg_s
        ),
        resistance_atmosphere_trace_kg_s=float(
            resistance_diagnostics.atmosphere_trace_kg_s
        ),
        resistance_cage_constraint_trace_kg_s=float(
            resistance_diagnostics.cage_constraint_trace_kg_s
        ),
        resistance_ligand_shell_obstruction_trace_kg_s=float(
            resistance_diagnostics.ligand_shell_obstruction_trace_kg_s
        ),
        resistance_aggregate_constraint_trace_kg_s=float(
            resistance_diagnostics.aggregate_constraint_trace_kg_s
        ),
        resistance_bridge_constraint_trace_kg_s=float(
            resistance_diagnostics.bridge_constraint_trace_kg_s
        ),
        resistance_orientation_denticity_trace_kg_s=float(
            resistance_diagnostics.orientation_denticity_trace_kg_s
        ),
        resistance_total_trace_kg_s=float(resistance_diagnostics.total_trace_kg_s),
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
    log_reference = _finite_float(
        float(np.max(log_terms)), "state diagnostic log weight"
    )
    shifted_weights = np.exp(log_terms - log_reference)
    normalized_weight_sum = _positive_float(
        float(np.sum(shifted_weights)),
        "state diagnostic normalized weight sum",
    )
    return shifted_weights / normalized_weight_sum


def _active_anion_name_from_state_label(state_label: str) -> str:
    return _active_anion_name_from_state_key(_state_key_from_label(state_label))


def _configuration_charge_numbers(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return np.asarray(
        [
            float(
                records.species_records[species_name]["sites"][int(site_id)][
                    "charge_number"
                ]
            )
            for species_name, site_id in zip(
                configuration.species_names,
                configuration.site_ids,
                strict=True,
            )
        ],
        dtype=float,
    )


def _molecular_charge_centers_for_role(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
    active_species_name: str,
) -> tuple[MolecularChargeCenter, ...]:
    centers = []
    visited_molecules: set[tuple[str, int]] = set()
    for site_index, species_name in enumerate(configuration.species_names):
        molecule_id = int(configuration.molecule_ids[site_index])
        molecule_key = (species_name, molecule_id)
        if molecule_key in visited_molecules:
            continue
        visited_molecules.add(molecule_key)
        if active_species_name and species_name != active_species_name:
            continue
        if _species_role(records, species_name) != role:
            continue
        formal_charge_number = float(
            records.species_records[species_name]["formal_charge_e"]
        )
        if formal_charge_number == 0.0:
            continue
        molecule_site_indices, center_of_mass_weights = (
            molecule_site_indices_and_mass_fractions(
                records,
                configuration,
                species_name,
                molecule_id,
            )
        )
        centers.append(
            MolecularChargeCenter(
                label=f"{species_name}:{molecule_id}",
                formal_charge_number=formal_charge_number,
                site_indices=molecule_site_indices,
                center_of_mass_weights=tuple(float(value) for value in center_of_mass_weights),
            )
        )
    return tuple(centers)


def _molecular_center_projection(
    center: MolecularChargeCenter,
    site_count: int,
) -> Array:
    projection = np.zeros(
        (CARTESIAN_DIMENSION, site_count * CARTESIAN_DIMENSION), dtype=float
    )
    for site_index, center_weight in zip(
        center.site_indices, center.center_of_mass_weights, strict=True
    ):
        cartesian_coordinate_start = site_index * CARTESIAN_DIMENSION
        cartesian_coordinate_slice = slice(
            cartesian_coordinate_start,
            cartesian_coordinate_start + CARTESIAN_DIMENSION,
        )
        projection[:, cartesian_coordinate_slice] = (
            center_weight * np.eye(CARTESIAN_DIMENSION, dtype=float)
        )
    return projection


def _molecular_center_position(
    configuration: SiteConfiguration,
    center: MolecularChargeCenter,
) -> Array:
    return np.einsum(
        "i,ia->a",
        np.asarray(center.center_of_mass_weights, dtype=float),
        np.asarray(configuration.unwrapped_positions_m, dtype=float)[
            np.asarray(center.site_indices, dtype=int)
        ],
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


def _charged_center_subset_pair_scalar_mobility(
    charged_center_mobility_matrix_m2_s: Array,
    charge_numbers: tuple[float, ...],
    first_center_indices: tuple[int, ...],
    second_center_indices: tuple[int, ...],
) -> float:
    center_mobility = np.asarray(charged_center_mobility_matrix_m2_s, dtype=float)
    charge_vector = np.asarray(charge_numbers, dtype=float)
    if center_mobility.shape != (charge_vector.size, charge_vector.size):
        raise ValueError("charged center mobility matrix shape does not match charges")
    total = 0.0
    for first_center_index in first_center_indices:
        for second_center_index in second_center_indices:
            total += (
                float(charge_vector[first_center_index])
                * float(charge_vector[second_center_index])
                * float(center_mobility[first_center_index, second_center_index])
            )
    return float(total)


def _center_subset_pair_scalar_mobility(
    center_mobility_matrix_m2_s: Array,
    first_center_indices: tuple[int, ...],
    second_center_indices: tuple[int, ...],
) -> float:
    center_mobility = np.asarray(center_mobility_matrix_m2_s, dtype=float)
    total = 0.0
    for first_center_index in first_center_indices:
        for second_center_index in second_center_indices:
            total += float(center_mobility[first_center_index, second_center_index])
    return float(total)


def _charged_center_mobility_matrix(
    mobility_tensor_m2_s: Array,
    site_count: int,
    charged_centers: tuple[MolecularChargeCenter, ...],
) -> Array:
    mobility = np.asarray(mobility_tensor_m2_s, dtype=float)
    if mobility.shape != (
        site_count * CARTESIAN_DIMENSION,
        site_count * CARTESIAN_DIMENSION,
    ):
        raise ValueError("mobility tensor shape does not match site count")
    center_count = len(charged_centers)
    center_mobility = np.zeros((center_count, center_count), dtype=float)
    center_projections = tuple(
        _molecular_center_projection(center, site_count) for center in charged_centers
    )
    for first_center_index, first_projection in enumerate(center_projections):
        for second_center_index, second_projection in enumerate(center_projections):
            block = first_projection @ mobility @ second_projection.T
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
    if not np.all(np.isfinite(center_mobility)) or not np.all(
        np.isfinite(charge_vector)
    ):
        raise ValueError("charged center covariance inputs must be finite")
    return float(charge_vector @ center_mobility @ charge_vector)


def _charged_center_pair_covariance_entries(
    state_label: str,
    center_labels: tuple[str, ...],
    charge_numbers: tuple[float, ...],
    charged_center_mobility_matrix_m2_s: Array,
) -> tuple[ChargedCenterPairCovarianceEntry, ...]:
    charge_vector = np.asarray(charge_numbers, dtype=float)
    center_mobility = np.asarray(charged_center_mobility_matrix_m2_s, dtype=float)
    if center_mobility.shape != (charge_vector.size, charge_vector.size):
        raise ValueError("charged center mobility matrix shape does not match charges")
    if len(center_labels) != charge_vector.size:
        raise ValueError("charged center labels length does not match charges")
    entries: list[ChargedCenterPairCovarianceEntry] = []
    for first_center_index in range(charge_vector.size):
        for second_center_index in range(first_center_index + 1, charge_vector.size):
            center_mobility_m2_s = float(
                center_mobility[first_center_index, second_center_index]
            )
            charge_weighted_covariance_m2_s = (
                2.0
                * float(charge_vector[first_center_index])
                * float(charge_vector[second_center_index])
                * center_mobility_m2_s
            )
            entries.append(
                ChargedCenterPairCovarianceEntry(
                    state_label=state_label,
                    first_center_label=center_labels[first_center_index],
                    second_center_label=center_labels[second_center_index],
                    first_charge_number=float(charge_vector[first_center_index]),
                    second_charge_number=float(charge_vector[second_center_index]),
                    center_mobility_m2_s=center_mobility_m2_s,
                    charge_weighted_covariance_m2_s=charge_weighted_covariance_m2_s,
                )
            )
    return tuple(entries)


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
    declared_families = tuple(
        str(family) for family in records.transition_record["families"]
    )
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
    state_keys = tuple(
        _state_key_from_label(state.label) for state in state_quadratures
    )
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
    _positive_float(temperature_K, "temperature_K")
    transition_edges = enumerate_transition_edges(state_quadratures, records)
    transition_edges = tuple(
        edge
        for edge in transition_edges
        if _transition_family_has_required_molecular_multiplicity(
            records,
            state_quadratures[edge.from_state_index].configurations[0],
            edge.family,
        )
    )
    return transition_edges


def _transition_family_has_required_molecular_multiplicity(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    family: str,
) -> bool:
    required_counts = {
        "partner_switch": ("anion", 2),
        "identity_diffusion": ("cation", 2),
        "bridge_network_formation_breakup": ("anion", 2),
    }
    if family not in required_counts:
        return True
    molecule_keys_by_role: dict[str, set[tuple[str, int]]] = {
        "cation": set(),
        "anion": set(),
    }
    for species_name, molecule_id in zip(
        configuration.species_names,
        np.asarray(configuration.molecule_ids, dtype=int),
        strict=True,
    ):
        role = str(records.species_records[species_name]["role"])
        if role in molecule_keys_by_role:
            molecule_keys_by_role[role].add((species_name, int(molecule_id)))
    role, required_count = required_counts[family]
    return len(molecule_keys_by_role[role]) >= required_count


def _validate_transport_graph_closure(
    records: PhysicalLibraryRecords,
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    declared_edges: tuple[TransitionEdge, ...],
    retained_edges: tuple[TransitionEdge, ...],
) -> None:
    retained_edge_set = set(retained_edges)
    displacement_edges = tuple(
        edge
        for edge in declared_edges
        if _transition_transport_ownership(
            _transition_family_record(records, edge.family)
        )
        is TransportOwnership.TRANSITION_DISPLACEMENT
        and _transition_family_has_required_molecular_multiplicity(
            records,
            state_quadratures[edge.from_state_index].configurations[0],
            edge.family,
        )
    )
    missing_displacement_edges = tuple(
        edge for edge in displacement_edges if edge not in retained_edge_set
    )

    retained_degrees = np.zeros(len(state_quadratures), dtype=int)
    for edge in retained_edges:
        retained_degrees[edge.from_state_index] += 1
        retained_degrees[edge.to_state_index] += 1
    displacement_incident_state_indices = {
        state_index
        for edge in displacement_edges
        for state_index in (edge.from_state_index, edge.to_state_index)
    }
    isolated_displacement_state_labels = tuple(
        state_quadratures[state_index].label
        for state_index in sorted(displacement_incident_state_indices)
        if retained_degrees[state_index] == 0
    )
    if missing_displacement_edges or isolated_displacement_state_labels:
        missing_edge_labels = tuple(
            (
                state_quadratures[edge.from_state_index].label,
                state_quadratures[edge.to_state_index].label,
                edge.family,
            )
            for edge in missing_displacement_edges
        )
        raise ValueError(
            "finite-generator transport graph is not closed before readout: "
            f"missing transition_displacement edges={missing_edge_labels}; "
            "zero-degree transition_displacement incident states="
            f"{isolated_displacement_state_labels}"
        )


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
    if _state_keys_match_on_indices(
        from_key, to_key, PARTNER_TRANSITION_KEY_INDICES
    ) and _ordered_label_transition(
        from_partner,
        to_partner,
        ("partner_a", "partner_switching", "partner_b"),
    ):
        return "partner_switch"
    if _state_keys_match_on_indices(
        from_key, to_key, IDENTITY_TRANSITION_KEY_INDICES
    ) and _ordered_label_transition(
        from_identity,
        to_identity,
        (
            "carrier_identity_a",
            "carrier_identity_transition",
            "carrier_identity_b",
        ),
    ):
        return "identity_diffusion"
    if _state_keys_match_on_indices(
        from_key, to_key, HOP_TRANSITION_KEY_INDICES
    ) and _ordered_label_transition(
        from_hop,
        to_hop,
        ("hop_source", "hop_transition", "hop_target"),
    ):
        return "structural_hop"
    if _state_keys_match_on_indices(
        from_key, to_key, CAGE_TRANSITION_KEY_INDICES
    ) and _ordered_label_transition(
        from_cage,
        to_cage,
        ("cage_released", "cage_boundary", "cage_captured"),
    ):
        return "cage_capture_release"
    if (
        _state_keys_match_on_indices(
            from_key, to_key, ATMOSPHERE_TRANSITION_KEY_INDICES
        )
        and from_atmosphere != to_atmosphere
    ):
        return "atmosphere_capture_release"
    return NO_TRANSITION_FAMILY


def _ordered_label_transition(
    from_label: str,
    to_label: str,
    ordered_labels: tuple[str, ...],
) -> bool:
    label_rank = {
        label: label_index for label_index, label in enumerate(ordered_labels)
    }
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
    transition_edges: tuple[TransitionEdge, ...],
    mixture: MixtureClosureResult,
    numerical_options: NumericalOptions,
):
    _ = state_quadratures
    _ = mixture
    _validate_numerical_options(numerical_options)
    selected_memory_coordinates = _selected_memory_coordinates(
        records,
        template_configuration,
        transition_edges,
    )

    return tuple(
        _bind_memory_gradient(memory_coordinate)
        for memory_coordinate in selected_memory_coordinates
    )


def _selected_memory_coordinates(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    transition_edges: tuple[TransitionEdge, ...],
) -> tuple[MemoryCoordinate, ...]:
    memory_coordinates = build_default_memory_coordinates(
        records, template_configuration
    )
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
        "bounded_internal_polarization": {"bounded_internal_polarization"},
    }
    implemented_coordinate_families = {
        memory_coordinate.family.value for memory_coordinate in memory_coordinates
    }
    present_roles = {
        _species_role(records, species_name)
        for species_name in template_configuration.species_names
    }
    required_role_by_declared_family = {
        "atmosphere_polarization": SpeciesRole.ANION,
        "charge_density_relaxation": SpeciesRole.ANION,
        "cage_backjump": SpeciesRole.ANION,
        "partner_residence": SpeciesRole.ANION,
        "ligand_shell_residence": SpeciesRole.ADDITIVE,
        "anion_orientation": SpeciesRole.ANION,
        "bounded_internal_polarization": SpeciesRole.ANION,
    }
    missing_families = []
    for declared_family in records.memory_record["memory_records"]:
        declared_family_name = str(declared_family)
        if declared_family_name not in implemented_family_map:
            missing_families.append(declared_family_name)
            continue
        required_role = required_role_by_declared_family[declared_family_name]
        if required_role not in present_roles:
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
    for declared_family in records.memory_record["memory_records"]:
        allowed_implementation_families.update(
            implemented_family_map[str(declared_family)]
        )
    retained_transition_families = {edge.family for edge in transition_edges}
    selected_coordinates = []
    for memory_coordinate in memory_coordinates:
        if memory_coordinate.family.value not in allowed_implementation_families:
            continue
        ownership, matching_transition_families = _memory_transport_ownership(
            records,
            memory_coordinate.family.value,
        )
        if ownership is TransportOwnership.DIAGNOSTIC:
            continue
        active_matching_families = retained_transition_families.intersection(
            matching_transition_families
        )
        if (
            ownership is TransportOwnership.TRANSITION_DISPLACEMENT
            and matching_transition_families
        ):
            if not active_matching_families:
                continue
        selected_coordinates.append(memory_coordinate)
    return tuple(selected_coordinates)


def build_default_memory_coordinates(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
) -> tuple[MemoryCoordinate, ...]:
    """Build current-coupled memory coordinates from physical records."""
    active_species_names = tuple(dict.fromkeys(template_configuration.species_names))
    anion_species_names = tuple(
        species_name
        for species_name in active_species_names
        if _species_role(records, species_name) == SpeciesRole.ANION
    )
    additive_species_names = tuple(
        species_name
        for species_name in active_species_names
        if _species_role(records, species_name) == SpeciesRole.ADDITIVE
    )
    coordinates: list[MemoryCoordinate] = []
    for anion_species_name in anion_species_names:
        coordinates.extend(
            (
                MemoryCoordinate(
                    label=f"cage_backjump[{anion_species_name}]",
                    family=MemoryCoordinateFamily.CAGE_BACKJUMP,
                    records=records,
                    required_roles=(SpeciesRole.CATION, SpeciesRole.ANION),
                    required_species_names=(anion_species_name,),
                    value_function=_cage_backjump_memory_value,
                    gradient_function=_memory_pair_distance_gradient,
                ),
                MemoryCoordinate(
                    label=f"partner_residence[{anion_species_name}]",
                    family=MemoryCoordinateFamily.PARTNER_RESIDENCE,
                    records=records,
                    required_roles=(SpeciesRole.CATION, SpeciesRole.ANION),
                    required_species_names=(anion_species_name,),
                    value_function=_partner_residence_memory_value,
                    gradient_function=_memory_li_anion_coordination_gradient,
                ),
                MemoryCoordinate(
                    label=f"anion_orientation[{anion_species_name}]",
                    family=MemoryCoordinateFamily.ANION_ORIENTATION,
                    records=records,
                    required_roles=(SpeciesRole.ANION,),
                    required_species_names=(anion_species_name,),
                    value_function=_anion_orientation_memory_value,
                    gradient_function=_anion_orientation_memory_gradient,
                ),
            )
        )
        for axis_name, value_function, gradient_function in (
            (
                "x",
                _bounded_internal_polarization_x_memory_value,
                _bounded_internal_polarization_x_memory_gradient,
            ),
            (
                "y",
                _bounded_internal_polarization_y_memory_value,
                _bounded_internal_polarization_y_memory_gradient,
            ),
            (
                "z",
                _bounded_internal_polarization_z_memory_value,
                _bounded_internal_polarization_z_memory_gradient,
            ),
        ):
            coordinates.append(
                MemoryCoordinate(
                    label=(
                        "bounded_internal_polarization"
                        f"[{anion_species_name},{axis_name}]"
                    ),
                    family=MemoryCoordinateFamily.BOUNDED_INTERNAL_POLARIZATION,
                    records=records,
                    required_roles=(SpeciesRole.CATION, SpeciesRole.ANION),
                    required_species_names=(anion_species_name,),
                    value_function=value_function,
                    gradient_function=gradient_function,
                )
            )
    for additive_species_name in additive_species_names:
        coordinates.append(
            MemoryCoordinate(
                label=f"ligand_shell[{additive_species_name}]",
                family=MemoryCoordinateFamily.LIGAND_SHELL,
                records=records,
                required_roles=(SpeciesRole.CATION, SpeciesRole.ADDITIVE),
                required_species_names=(additive_species_name,),
                value_function=_ligand_shell_memory_value,
                gradient_function=_memory_li_ligand_coordination_gradient,
            )
        )
    return tuple(coordinates)


def _state_family_memory_keys(
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
) -> tuple[tuple[str, ...], ...]:
    family_indices = (
        STATE_KEY_ANION_INDEX,
        STATE_KEY_PAIR_INDEX,
        STATE_KEY_LIGAND_INDEX,
        STATE_KEY_CLUSTER_INDEX,
        STATE_KEY_CAGE_INDEX,
        STATE_KEY_ORIENTATION_INDEX,
        STATE_KEY_PARTNER_INDEX,
    )
    return tuple(
        tuple(_state_key_from_label(state_quadrature.label)[index] for index in family_indices)
        for state_quadrature in state_quadratures
    )


def _state_family_memory_value_matrix(
    state_quadratures: tuple[PhysicalStateQuadrature, ...],
    transition_edges: tuple[TransitionEdge, ...],
) -> Array:
    state_family_keys = _state_family_memory_keys(state_quadratures)
    unique_family_keys = _active_state_family_memory_keys(
        state_family_keys,
        transition_edges,
    )
    indicator_matrix = np.asarray(
        [
            [float(state_family_key == candidate_key) for candidate_key in unique_family_keys]
            for state_family_key in state_family_keys
        ],
        dtype=float,
    )
    if indicator_matrix.size == 0:
        return np.zeros((len(state_quadratures), 0), dtype=float)
    centered_matrix = indicator_matrix - np.mean(indicator_matrix, axis=0, keepdims=True)
    left_vectors, singular_values, _right_vectors = np.linalg.svd(
        centered_matrix,
        full_matrices=False,
    )
    if singular_values.size == 0:
        return np.zeros((len(state_quadratures), 0), dtype=float)
    rank_tolerance = (
        max(centered_matrix.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
    )
    retained_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    return left_vectors[:, :retained_rank]


def _active_state_family_memory_keys(
    state_family_keys: tuple[tuple[str, ...], ...],
    transition_edges: tuple[TransitionEdge, ...],
) -> tuple[tuple[str, ...], ...]:
    crossed_family_keys = {
        state_family_keys[state_index]
        for transition_edge in transition_edges
        for state_index, other_state_index in (
            (transition_edge.from_state_index, transition_edge.to_state_index),
            (transition_edge.to_state_index, transition_edge.from_state_index),
        )
        if state_family_keys[state_index] != state_family_keys[other_state_index]
    }
    return tuple(
        family_key
        for family_key in dict.fromkeys(state_family_keys)
        if family_key in crossed_family_keys
    )


def combine_memory_values(
    coordinates: tuple[MemoryCoordinate, ...],
    configuration: SiteConfiguration,
) -> Array:
    return np.asarray(
        [
            coordinate.value_function(coordinate.records, configuration)
            if _memory_coordinate_is_supported(coordinate, configuration)
            else 0.0
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
            if _memory_coordinate_is_supported(coordinate, configuration)
            else np.zeros(
                (1, len(configuration.species_names) * CARTESIAN_DIMENSION),
                dtype=float,
            )
            for coordinate in coordinates
        )
    )


def _bind_memory_gradient(
    memory_coordinate: MemoryCoordinate,
) -> Callable[[SiteConfiguration], Array]:
    def gradient(configuration: SiteConfiguration) -> Array:
        if not _memory_coordinate_is_supported(memory_coordinate, configuration):
            return np.zeros(
                (1, len(configuration.species_names) * CARTESIAN_DIMENSION),
                dtype=float,
            )
        return _memory_gradient_row(
            memory_coordinate.gradient_function(
                memory_coordinate.records,
                configuration,
            )
        )

    return gradient


def _memory_coordinate_is_supported(
    memory_coordinate: MemoryCoordinate,
    configuration: SiteConfiguration,
) -> bool:
    present_roles = {
        _species_role(memory_coordinate.records, species_name)
        for species_name in configuration.species_names
    }
    present_species_names = set(configuration.species_names)
    return all(
        role in present_roles for role in memory_coordinate.required_roles
    ) and all(
        species_name in present_species_names
        for species_name in memory_coordinate.required_species_names
    )


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
    anion_coordination = compute_role_coordination_number(
        records,
        configuration,
        center_role=SpeciesRole.CATION.value,
        ligand_role=SpeciesRole.ANION.value,
        switch_name="Li_anion",
    )
    ligand_coordination = compute_role_coordination_number(
        records,
        configuration,
        center_role=SpeciesRole.CATION.value,
        ligand_role=SpeciesRole.ADDITIVE.value,
        switch_name="Li_ligand",
    )
    return anion_coordination / (1.0 + ligand_coordination)


def _memory_li_anion_coordination_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    anion_coordination = compute_role_coordination_number(
        records,
        configuration,
        center_role=SpeciesRole.CATION.value,
        ligand_role=SpeciesRole.ANION.value,
        switch_name="Li_anion",
    )
    ligand_coordination = compute_role_coordination_number(
        records,
        configuration,
        center_role=SpeciesRole.CATION.value,
        ligand_role=SpeciesRole.ADDITIVE.value,
        switch_name="Li_ligand",
    )
    anion_gradient = _coordination_switch_gradient(
        records,
        configuration,
        "Li_anion",
    )
    ligand_gradient = _coordination_switch_gradient(
        records,
        configuration,
        "Li_ligand",
    )
    ligand_denominator = 1.0 + ligand_coordination
    return (
        anion_gradient / ligand_denominator
        - anion_coordination * ligand_gradient / ligand_denominator**2
    )


def _bounded_internal_polarization_x_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    return _bounded_internal_polarization_memory_value_for_axis(
        records,
        configuration,
        0,
    )


def _bounded_internal_polarization_y_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    return _bounded_internal_polarization_memory_value_for_axis(
        records,
        configuration,
        1,
    )


def _bounded_internal_polarization_z_memory_value(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> float:
    return _bounded_internal_polarization_memory_value_for_axis(
        records,
        configuration,
        2,
    )


def _bounded_internal_polarization_memory_value_for_axis(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    cartesian_axis: int,
) -> float:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_index = _first_role_index(records, configuration, SpeciesRole.ANION)
    pair_vector = np.asarray(
        configuration.positions_m[anion_index], dtype=float
    ) - np.asarray(configuration.positions_m[cation_index], dtype=float)
    charge_numbers = _configuration_charge_numbers(records, configuration)
    return float(
        (charge_numbers[cation_index] - charge_numbers[anion_index])
        * pair_vector[cartesian_axis]
    )


def _bounded_internal_polarization_x_memory_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _bounded_internal_polarization_memory_gradient_for_axis(
        records,
        configuration,
        0,
    )


def _bounded_internal_polarization_y_memory_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _bounded_internal_polarization_memory_gradient_for_axis(
        records,
        configuration,
        1,
    )


def _bounded_internal_polarization_z_memory_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _bounded_internal_polarization_memory_gradient_for_axis(
        records,
        configuration,
        2,
    )


def _bounded_internal_polarization_memory_gradient_for_axis(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    cartesian_axis: int,
) -> Array:
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    anion_index = _first_role_index(records, configuration, SpeciesRole.ANION)
    charge_numbers = _configuration_charge_numbers(records, configuration)
    charge_difference = charge_numbers[cation_index] - charge_numbers[anion_index]
    gradient = np.zeros(
        (1, len(configuration.species_names) * CARTESIAN_DIMENSION),
        dtype=float,
    )
    gradient[
        0,
        cation_index * CARTESIAN_DIMENSION + cartesian_axis,
    ] = -charge_difference
    gradient[
        0,
        anion_index * CARTESIAN_DIMENSION + cartesian_axis,
    ] = charge_difference
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


def _anion_orientation_memory_gradient(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
) -> Array:
    return _memory_gradient_row(
        _finite_difference_named_scalar_gradient(
            records,
            configuration,
            "anion_orientation",
        )
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
    template_configuration: SiteConfiguration,
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
            radial_reference_m = _positive_float(
                float(records.basis_record["pair_basins"]["r_free_m"]),
                "pair_basins.r_free_m",
            )
            coordinate_weights = coordinate_weights * (
                coordinate_values / radial_reference_m
            ) ** 2
            nodes.append((coordinate, coordinate_values, coordinate_weights))
            continue
        coordinate_values, coordinate_weights = _non_distance_coordinate_nodes(
            records,
            template_configuration,
            coordinate,
            recipe_context,
            mixture,
        )
        nodes.append((coordinate, coordinate_values, coordinate_weights))
    return tuple(nodes)


def _non_distance_coordinate_nodes(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate: ReducedCoordinate,
    recipe_context: RecipeBuildResult,
    mixture: MixtureClosureResult,
) -> tuple[Array, Array]:
    required_role_multiplicity = {
        ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE: (SpeciesRole.ANION, 2),
        ReducedCoordinate.IDENTITY_COORDINATE: (SpeciesRole.CATION, 2),
        ReducedCoordinate.CLUSTER_COORDINATE: (SpeciesRole.ANION, 2),
    }
    if coordinate in required_role_multiplicity:
        role, required_molecule_count = required_role_multiplicity[coordinate]
        molecule_count = len(
            _charged_role_molecule_keys(records, template_configuration, role)
        )
        if molecule_count < required_molecule_count:
            return np.asarray([0.0], dtype=float), np.asarray([1.0], dtype=float)
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
    if coordinate == ReducedCoordinate.LOCAL_PACKING_FRACTION:
        return (
            np.asarray([0.0], dtype=float),
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
        raise ValueError(
            "basis.coordination_high_bin_interior_fraction must be below one"
        )
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
    for component in recipe_context.resolved_species:
        if _species_role(recipe_context.library_records, component.name) == role:
            return True
    return False


def _configuration_with_reduced_coordinate_values(
    records: PhysicalLibraryRecords,
    template_configuration: SiteConfiguration,
    coordinate_values: dict[str, float],
) -> SiteConfiguration:
    configuration = _configuration_with_anion_orientation(
        records,
        template_configuration,
        float(coordinate_values[ReducedCoordinate.ANION_ORIENTATION.value]),
    )
    configuration = _configuration_with_pair_distance(
        records,
        configuration,
        _positive_float(
            coordinate_values[ReducedCoordinate.LI_ANION_DISTANCE.value],
            ReducedCoordinate.LI_ANION_DISTANCE.value,
        ),
    )
    configuration = _configuration_with_secondary_anion_geometry(
        records,
        configuration,
        _positive_float(
            coordinate_values[ReducedCoordinate.LI_ANION_DISTANCE.value],
            ReducedCoordinate.LI_ANION_DISTANCE.value,
        ),
    )
    displacement_coordinates = (
        ("identity_diffusion", ReducedCoordinate.IDENTITY_COORDINATE),
        ("structural_hop", ReducedCoordinate.STRUCTURAL_HOP_COORDINATE),
    )
    for transition_family, coordinate in displacement_coordinates:
        coordinate_value = float(coordinate_values[coordinate.value])
        if coordinate_value == 0.0:
            continue
        if not _transition_family_has_required_molecular_multiplicity(
            records,
            configuration,
            transition_family,
        ):
            continue
        transition_record = _transition_family_record(records, transition_family)
        displacement_m = (
            _endpoint_geometry_displacement_vector_m(
                records,
                configuration,
                transition_record,
                transition_family,
            )
            * coordinate_value
        )
        configuration = _configuration_with_cation_displacement(
            records,
            configuration,
            displacement_m,
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
    solvent_directions = (
        np.asarray([0.0, 1.0, 0.0], dtype=float),
        np.asarray([0.0, -1.0, 0.0], dtype=float),
        np.asarray([0.0, 0.0, 1.0], dtype=float),
        np.asarray([0.0, 0.0, -1.0], dtype=float),
    )
    additive_directions = (
        np.asarray([-1.0, 0.0, 0.0], dtype=float),
        np.asarray([-1.0, 1.0, 0.0], dtype=float) / np.sqrt(2.0),
        np.asarray([-1.0, -1.0, 0.0], dtype=float) / np.sqrt(2.0),
        np.asarray([-1.0, 0.0, 1.0], dtype=float) / np.sqrt(2.0),
    )
    radial_directions = (
        solvent_directions
        if ligand_role is SpeciesRole.SOLVENT
        else additive_directions
    )
    if len(molecule_ids) > len(radial_directions):
        raise ValueError(
            f"{ligand_role.value} coordination supports at most "
            f"{len(radial_directions)} molecules"
        )
    for molecule_offset, molecule_id in enumerate(molecule_ids):
        molecule_site_indices = tuple(
            site_index
            for site_index in ligand_indices
            if int(configuration.molecule_ids[site_index]) == molecule_id
        )
        coordination_site_index = _molecule_coordination_site_index(
            records,
            configuration,
            molecule_site_indices,
        )
        molecule_center_position = np.mean(
            positions[np.asarray(molecule_site_indices, dtype=int)],
            axis=0,
        )
        center_to_coordination_site = (
            positions[coordination_site_index] - molecule_center_position
        )
        center_to_coordination_site_norm = float(
            np.linalg.norm(center_to_coordination_site)
        )
        radial_direction = radial_directions[molecule_offset]
        if center_to_coordination_site_norm > 0.0:
            target_center_to_coordination_site = (
                -center_to_coordination_site_norm * radial_direction
            )
            radial_alignment = _rotation_matrix_between_unit_vectors(
                center_to_coordination_site / center_to_coordination_site_norm,
                target_center_to_coordination_site / center_to_coordination_site_norm,
            )
            for site_index in molecule_site_indices:
                positions[site_index] = molecule_center_position + radial_alignment @ (
                    positions[site_index] - molecule_center_position
                )
        target_coordination_site = cation_position + distance_m * radial_direction
        shift = target_coordination_site - positions[coordination_site_index]
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
        raise ValueError(
            "coordination switch minimum_value must be between zero and one"
        )
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
    unwrapped_positions = np.asarray(
        configuration.unwrapped_positions_m, dtype=float
    ).copy()
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
    sine_component = float(
        np.sqrt(max(0.0, 1.0 - orientation_cosine * orientation_cosine))
    )
    target_vector = original_length * (
        orientation_cosine * unit_pair_axis + sine_component * perpendicular_axis
    )
    rotation_matrix = _rotation_matrix_between_unit_vectors(
        original_vector / original_length,
        target_vector / original_length,
    )
    anchor_position = positions[anion_anchor].copy()
    unwrapped_anchor_position = unwrapped_positions[anion_anchor].copy()
    for site_index in anion_indices:
        positions[site_index] = anchor_position + rotation_matrix @ (
            positions[site_index] - anchor_position
        )
        unwrapped_positions[site_index] = (
            unwrapped_anchor_position
            + rotation_matrix
            @ (unwrapped_positions[site_index] - unwrapped_anchor_position)
        )
    return SiteConfiguration(
        species_names=configuration.species_names,
        molecule_ids=np.asarray(configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(configuration.site_ids, dtype=int),
        positions_m=positions,
        unwrapped_positions_m=unwrapped_positions,
        box_lengths_m=np.asarray(configuration.box_lengths_m, dtype=float),
    )


def _rotation_matrix_between_unit_vectors(
    source_unit_vector: Array,
    target_unit_vector: Array,
) -> Array:
    source_vector = np.asarray(source_unit_vector, dtype=float)
    target_vector = np.asarray(target_unit_vector, dtype=float)
    source_norm = _positive_float(
        float(np.linalg.norm(source_vector)), "source_unit_norm"
    )
    target_norm = _positive_float(
        float(np.linalg.norm(target_vector)), "target_unit_norm"
    )
    source_unit = source_vector / source_norm
    target_unit = target_vector / target_norm
    cosine_angle = float(np.dot(source_unit, target_unit))
    if cosine_angle > 1.0:
        if not np.isclose(cosine_angle, 1.0):
            raise ValueError("rotation cosine exceeds unit-vector upper bound")
        cosine_angle = 1.0
    if cosine_angle < -1.0:
        if not np.isclose(cosine_angle, -1.0):
            raise ValueError("rotation cosine exceeds unit-vector lower bound")
        cosine_angle = -1.0
    if np.isclose(cosine_angle, 1.0):
        return np.eye(CARTESIAN_DIMENSION, dtype=float)
    cross_vector = np.cross(source_unit, target_unit)
    cross_norm = float(np.linalg.norm(cross_vector))
    if np.isclose(cosine_angle, -1.0):
        perpendicular_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
        if (
            abs(float(np.dot(source_unit, perpendicular_axis)))
            > PERPENDICULAR_AXIS_ALIGNMENT_LIMIT
        ):
            perpendicular_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
        cross_vector = np.cross(source_unit, perpendicular_axis)
        cross_norm = _positive_float(
            float(np.linalg.norm(cross_vector)), "rotation_axis_norm"
        )
        unit_axis = cross_vector / cross_norm
        return -np.eye(CARTESIAN_DIMENSION, dtype=float) + 2.0 * np.outer(
            unit_axis,
            unit_axis,
        )
    unit_axis = cross_vector / _positive_float(cross_norm, "rotation_axis_norm")
    skew_matrix = np.asarray(
        [
            [0.0, -unit_axis[2], unit_axis[1]],
            [unit_axis[2], 0.0, -unit_axis[0]],
            [-unit_axis[1], unit_axis[0], 0.0],
        ],
        dtype=float,
    )
    return (
        np.eye(CARTESIAN_DIMENSION, dtype=float)
        + skew_matrix * cross_norm
        + (skew_matrix @ skew_matrix) * (1.0 - cosine_angle)
    )


def _state_distance_bounds(
    records: PhysicalLibraryRecords,
    recipe_context: RecipeBuildResult,
) -> tuple[tuple[str, float, float], ...]:
    pair_basins = records.basis_record["pair_basins"]
    contact_cutoff_m = float(pair_basins["r_CIP_m"])
    solvent_separated_cutoff_m = float(pair_basins["r_SSIP_m"])
    free_cutoff_m = float(pair_basins["r_free_m"])
    free_outer_factor = float(
        records.basis_record["quadrature"]["free_outer_distance_factor"]
    )
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
        for component in recipe_context.resolved_species
    ):
        specs.append(("addSSIP", contact_cutoff_m, solvent_separated_cutoff_m))
    return tuple(specs)


def _normalize_potential_energy_reference(
    reduced_specification: ReducedGeneratorSpecification,
) -> ReducedGeneratorSpecification:
    sampled_energies = []
    for state_quadrature in reduced_specification.state_quadratures:
        for point in state_quadrature.points:
            sampled_energies.append(
                float(reduced_specification.potential_energy_J_mol(point))
            )
    for transition_quadrature in reduced_specification.transition_quadratures:
        for point in transition_quadrature.points:
            sampled_energies.append(
                float(reduced_specification.potential_energy_J_mol(point))
            )
    energy_reference_J_mol = _finite_float(
        float(np.min(np.asarray(sampled_energies, dtype=float))),
        "energy_reference_J_mol",
    )

    def shifted_potential_energy_J_mol(point: Array) -> float:
        return (
            float(reduced_specification.potential_energy_J_mol(point))
            - energy_reference_J_mol
        )

    return replace(
        reduced_specification,
        potential_energy_J_mol=shifted_potential_energy_J_mol,
    )


def _projected_mass_balance_components(recipe_context: RecipeBuildResult):
    unsorted_components = tuple(
        active_component
        for component in recipe_context.resolved_species
        for active_component in (
            _component_with_active_recipe_loading(recipe_context, component),
        )
        if _species_role(recipe_context.library_records, component.name)
        in (SpeciesRole.CATION, SpeciesRole.ANION, SpeciesRole.ADDITIVE)
        and active_component.concentration_mol_m3 > 0.0
    )
    components = tuple(
        sorted(
            unsorted_components,
            key=_projected_component_sort_key(recipe_context.library_records),
        )
    )
    if not components:
        raise ValueError("projected conductivity mass balance needs charged components")
    return components


def _projected_component_sort_key(records: PhysicalLibraryRecords):
    role_rank = {
        SpeciesRole.CATION: 0,
        SpeciesRole.ANION: 1,
        SpeciesRole.ADDITIVE: 2,
    }

    def component_key(component: RecipeComponentLoading) -> tuple[int, str]:
        return role_rank[_species_role(records, component.name)], component.name

    return component_key


def _component_with_active_recipe_loading(
    recipe_context: RecipeBuildResult,
    component: RecipeComponentLoading,
) -> RecipeComponentLoading:
    if (
        _species_role(recipe_context.library_records, component.name)
        != SpeciesRole.ADDITIVE
    ):
        return component
    if component.name not in recipe_context.additive_weight_fractions:
        return replace(component, concentration_mol_m3=0.0)
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
    raise ValueError(
        f"unsupported transition labels {from_pair_label}, {to_pair_label}"
    )


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
    coordination_site_index = _molecule_coordination_site_index(
        records,
        configuration,
        anion_indices,
    )
    positions = np.asarray(configuration.positions_m, dtype=float).copy()
    cation_position = positions[cation_index].copy()
    anion_center_position = np.mean(
        positions[np.asarray(anion_indices, dtype=int)],
        axis=0,
    )
    center_to_coordination_site = (
        positions[coordination_site_index] - anion_center_position
    )
    center_to_coordination_site_norm = float(
        np.linalg.norm(center_to_coordination_site)
    )
    if center_to_coordination_site_norm > 0.0:
        target_center_to_coordination_site = np.asarray(
            [-center_to_coordination_site_norm, 0.0, 0.0],
            dtype=float,
        )
        radial_alignment = _rotation_matrix_between_unit_vectors(
            center_to_coordination_site / center_to_coordination_site_norm,
            target_center_to_coordination_site / center_to_coordination_site_norm,
        )
        for site_index in anion_indices:
            positions[site_index] = anion_center_position + radial_alignment @ (
                positions[site_index] - anion_center_position
            )
    target_anion_position = cation_position + np.asarray(
        [pair_distance_m, 0.0, 0.0],
        dtype=float,
    )
    shift = target_anion_position - positions[coordination_site_index]
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


def _configuration_with_secondary_anion_geometry(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    pair_distance_m: float,
) -> SiteConfiguration:
    anion_molecules = _molecule_site_index_groups_with_role(
        records,
        configuration,
        SpeciesRole.ANION,
    )
    if len(anion_molecules) < 2:
        return configuration
    cation_index = _first_role_index(records, configuration, SpeciesRole.CATION)
    positions = np.asarray(configuration.positions_m, dtype=float).copy()
    cation_position = positions[cation_index].copy()
    transport_directions = (
        np.asarray([0.0, 0.0, 1.0], dtype=float),
        np.asarray([0.0, 0.0, -1.0], dtype=float),
        np.asarray([0.0, 1.0, 0.0], dtype=float),
        np.asarray([0.0, -1.0, 0.0], dtype=float),
    )
    if len(anion_molecules) - 1 > len(transport_directions):
        raise ValueError("secondary-anion geometry supports at most four partners")
    for partner_offset, anion_indices in enumerate(anion_molecules[1:]):
        coordination_site_index = _molecule_coordination_site_index(
            records,
            configuration,
            anion_indices,
        )
        anion_center_position = np.mean(
            positions[np.asarray(anion_indices, dtype=int)],
            axis=0,
        )
        center_to_coordination_site = (
            positions[coordination_site_index] - anion_center_position
        )
        center_to_coordination_site_norm = _positive_float(
            float(np.linalg.norm(center_to_coordination_site)),
            "secondary anion center-to-coordination-site distance",
        )
        transport_direction = transport_directions[partner_offset]
        target_center_to_coordination_site = (
            -center_to_coordination_site_norm * transport_direction
        )
        radial_alignment = _rotation_matrix_between_unit_vectors(
            center_to_coordination_site / center_to_coordination_site_norm,
            target_center_to_coordination_site / center_to_coordination_site_norm,
        )
        for site_index in anion_indices:
            positions[site_index] = anion_center_position + radial_alignment @ (
                positions[site_index] - anion_center_position
            )
        target_coordination_site = (
            cation_position + pair_distance_m * transport_direction
        )
        shift = target_coordination_site - positions[coordination_site_index]
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


def _molecule_site_index_groups_with_role(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    role: SpeciesRole,
) -> tuple[tuple[int, ...], ...]:
    molecule_keys: list[tuple[str, int]] = []
    for site_index, species_name in enumerate(configuration.species_names):
        if _species_role(records, species_name) != role:
            continue
        molecule_key = (species_name, int(configuration.molecule_ids[site_index]))
        if molecule_key not in molecule_keys:
            molecule_keys.append(molecule_key)
    return tuple(
        tuple(
            site_index
            for site_index, species_name in enumerate(configuration.species_names)
            if (
                species_name,
                int(configuration.molecule_ids[site_index]),
            )
            == molecule_key
        )
        for molecule_key in molecule_keys
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
    coordination_site_index = _molecule_coordination_site_index(
        records,
        configuration,
        anion_indices,
    )
    positions = np.asarray(configuration.positions_m, dtype=float)
    displacement = positions[coordination_site_index] - np.asarray(
        configuration.positions_m[cation_index], dtype=float
    )
    distance_m = _positive_float(float(np.linalg.norm(displacement)), "pair distance")
    unit_vector = displacement / distance_m
    gradient = np.zeros(
        len(configuration.species_names) * CARTESIAN_DIMENSION, dtype=float
    )
    cation_start = cation_index * CARTESIAN_DIMENSION
    gradient[cation_start : cation_start + CARTESIAN_DIMENSION] = -unit_vector
    anion_start = coordination_site_index * CARTESIAN_DIMENSION
    gradient[anion_start : anion_start + CARTESIAN_DIMENSION] = unit_vector
    return gradient


def _molecule_coordination_site_index(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    anion_site_indices: tuple[int, ...],
) -> int:
    if not anion_site_indices:
        raise ValueError("coordination site selection requires a molecule")
    acceptor_indices = tuple(
        site_index
        for site_index in anion_site_indices
        if bool(
            _configuration_site_record(records, configuration, site_index)[
                "acceptor_flag"
            ]
        )
    )
    candidate_indices = acceptor_indices or anion_site_indices
    selected_site_index = candidate_indices[0]
    selected_key = _anion_coordination_site_sort_key(
        records,
        configuration,
        selected_site_index,
    )
    for candidate_site_index in candidate_indices[1:]:
        candidate_key = _anion_coordination_site_sort_key(
            records,
            configuration,
            candidate_site_index,
        )
        if candidate_key < selected_key:
            selected_site_index = candidate_site_index
            selected_key = candidate_key
    return selected_site_index


def _anion_coordination_site_sort_key(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
) -> tuple[float, int]:
    return (
        float(
            _configuration_site_record(records, configuration, site_index)[
                "charge_number"
            ]
        ),
        int(configuration.site_ids[site_index]),
    )


def _configuration_site_record(
    records: PhysicalLibraryRecords,
    configuration: SiteConfiguration,
    site_index: int,
) -> dict:
    species_name = configuration.species_names[site_index]
    site_id = int(configuration.site_ids[site_index])
    matching_records = tuple(
        site_record
        for site_record in records.species_records[species_name]["sites"]
        if int(site_record["site_id"]) == site_id
    )
    if len(matching_records) != 1:
        raise ValueError(
            f"species {species_name} must have one site record for site_id {site_id}"
        )
    return matching_records[0]


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
    coordination_site_index = _molecule_coordination_site_index(
        records,
        configuration,
        anion_indices,
    )
    positions = np.asarray(configuration.positions_m, dtype=float)
    displacement = positions[coordination_site_index] - positions[cation_index]
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
    orientation = assign_orientation_basin(
        records, configuration, cation_index, anion_index
    )
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
    active_coordinates: frozenset[ReducedCoordinate],
) -> tuple[str, ...]:
    _ = mixture
    return (
        pair_label,
        _lithium_shell_label_from_coordinates(records, coordinate_values),
        _ligand_state_label_from_coordinates(records, coordinate_values),
        _anion_feature_label(records, configuration),
        _orientation_label_from_coordinates(records, coordinate_values),
        _cluster_label_from_coordinates(records, pair_label, coordinate_values)
        if ReducedCoordinate.CLUSTER_COORDINATE in active_coordinates
        else _cluster_label(pair_label),
        _partner_label_from_coordinates(records, coordinate_values)
        if ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE in active_coordinates
        else "partner_inactive",
        _identity_label_from_coordinates(records, coordinate_values)
        if ReducedCoordinate.IDENTITY_COORDINATE in active_coordinates
        else "identity_inactive",
        _structural_hop_label_from_coordinates(records, coordinate_values)
        if ReducedCoordinate.STRUCTURAL_HOP_COORDINATE in active_coordinates
        else "hop_inactive",
        _cage_label_from_coordinates(records, coordinate_values)
        if ReducedCoordinate.CAGE_COORDINATE in active_coordinates
        else "cage_inactive",
        _environment_label_from_coordinates(records, coordinate_values),
        _atmosphere_label_from_coordinates(records, coordinate_values)
        if ReducedCoordinate.ATMOSPHERE_POLARIZATION in active_coordinates
        else "atmosphere_inactive",
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
        frozenset(_declared_reduced_coordinates(records)),
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
    orientation_value = float(
        coordinate_values[ReducedCoordinate.ANION_ORIENTATION.value]
    )
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
    identity_value = float(
        coordinate_values[ReducedCoordinate.IDENTITY_COORDINATE.value]
    )
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
    hop_value = float(
        coordinate_values[ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value]
    )
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
    bin_index = _threshold_bin_index(
        coordinate_value, _unit_interval_thresholds(records)
    )
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
        raise ValueError(
            f"configuration has no molecule indices for species {species_name}"
        )
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
