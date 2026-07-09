"""Strict physical-library loader for projected conductivity."""

from conductivity.physical_library.library_io import (
    PhysicalLibraryRecords,
    load_required_species_records,
    load_physical_library,
    validate_physical_library_records,
)
from conductivity.physical_library.basin_builder import (
    BasinFeatureVector,
    StateDefinition,
    build_state_definition,
    compute_basin_features,
)
from conductivity.physical_library.generator_construction import (
    MemoryCoordinate,
    build_default_memory_coordinates,
    combine_memory_gradients,
    combine_memory_values,
)
from conductivity.physical_library.mixture_closures import (
    MixtureClosureResult,
    MixtureComposition,
    compute_mixture_closures,
)
from conductivity.physical_library.physical_generator_builder import (
    PhysicalGeneratorBuildInput,
    PhysicalStateQuadrature,
    PhysicalTransitionQuadrature,
    build_reduced_generator_specification_from_physical_objects,
)
from conductivity.physical_library.physical_objects import (
    PairBasin,
    PhysicalObjectBundle,
    SiteConfiguration,
    assign_pair_basin,
    build_physical_objects,
    compute_charge_polarization_gradient,
    compute_charge_polarization_m,
)
from conductivity.physical_library.projected_primitives_io import (
    PrimitiveExternalScalarInput,
    PrimitiveOracleAuditReport,
    PrimitiveScalarEstimateNotProvided,
    PrimitiveScalarEstimateValue,
    PrimitiveScalarGapNotComputed,
    PrimitiveScalarGapValue,
    PrimitiveTensorGaps,
    PrimitiveTensorNorms,
    audit_primitive_oracle_closure,
    audit_primitive_oracle_closure_from_yaml,
    compute_conductivity_from_primitive_yaml,
    read_projected_primitive_yaml,
    write_projected_primitive_yaml,
)
from conductivity.physical_library.transition_surface_builder import (
    OneDimensionalTransitionBuildInput,
    TransitionSurfaceBuildResult,
    build_one_dimensional_transition_surface,
)

__all__ = [
    "BasinFeatureVector",
    "MemoryCoordinate",
    "MixtureClosureResult",
    "MixtureComposition",
    "OneDimensionalTransitionBuildInput",
    "PairBasin",
    "PhysicalGeneratorBuildInput",
    "PhysicalLibraryRecords",
    "PhysicalObjectBundle",
    "PhysicalStateQuadrature",
    "PhysicalTransitionQuadrature",
    "PrimitiveExternalScalarInput",
    "PrimitiveOracleAuditReport",
    "PrimitiveScalarEstimateNotProvided",
    "PrimitiveScalarEstimateValue",
    "PrimitiveScalarGapNotComputed",
    "PrimitiveScalarGapValue",
    "PrimitiveTensorGaps",
    "PrimitiveTensorNorms",
    "SiteConfiguration",
    "StateDefinition",
    "TransitionSurfaceBuildResult",
    "assign_pair_basin",
    "audit_primitive_oracle_closure",
    "audit_primitive_oracle_closure_from_yaml",
    "build_default_memory_coordinates",
    "build_one_dimensional_transition_surface",
    "build_physical_objects",
    "build_reduced_generator_specification_from_physical_objects",
    "build_state_definition",
    "combine_memory_gradients",
    "combine_memory_values",
    "compute_basin_features",
    "compute_charge_polarization_gradient",
    "compute_charge_polarization_m",
    "compute_conductivity_from_primitive_yaml",
    "compute_mixture_closures",
    "load_physical_library",
    "load_required_species_records",
    "read_projected_primitive_yaml",
    "validate_physical_library_records",
    "write_projected_primitive_yaml",
]
