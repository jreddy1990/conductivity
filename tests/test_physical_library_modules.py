from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from constants import T_REF_K
from conductivity.physical_library import generator_construction
from conductivity.physical_library import extract_projected_primitives
from conductivity.physical_library import physical_generator_builder
from conductivity.physical_library import property_db_validation
from conductivity.physical_library.transition_surface_builder import (
    MomentBoundaryValueInput,
    OneDimensionalCommittorInput,
    build_path_moment_arrays,
    solve_endpoint_moment_bvp,
    solve_one_dimensional_committor,
)
from conductivity.physical_library.basin_builder import build_state_definition
from conductivity.physical_library.generator_construction import (
    build_default_memory_coordinates,
    combine_memory_gradients,
    combine_memory_values,
)
from conductivity.physical_library.mixture_closures import (
    MixtureComposition,
    compute_bulk_dielectric_constant,
    compute_bulk_viscosity_Pa_s,
    compute_mixture_closures,
)
from conductivity.physical_library.physical_generator_builder import (
    LOCAL_FIELD_VECTOR_LENGTH,
    PhysicalGeneratorBuildInput,
    PhysicalLocalFields,
    PhysicalStateQuadrature,
    build_reduced_generator_specification_from_physical_objects,
    flatten_configuration_with_local_fields,
)
from conductivity.physical_library.physical_objects import (
    PairBasin,
    SiteConfiguration,
    assign_pair_basin,
    build_physical_objects,
    compute_atmosphere_resistance_diagnostics,
    compute_charge_polarization_m,
    compute_local_packing_fraction,
    compute_resistance_component_diagnostics,
    compute_resistance_tensor_kg_s,
    _rpy_cross_mobility_block_kg_inv_s,
    _rigid_body_kinematic_map,
    anion_internal_charge_separation_factor,
    li_anion_feature_coordination_energy_multiplier,
)
from conductivity.physical_library.projected_analytical_conductivity import (
    StateTransportOwnershipBasis,
    TransportOwnership,
)
from conductivity.physical_library.projected_primitives_io import (
    PRIMITIVE_SCHEMA,
    PrimitiveExternalScalarInput,
    PrimitiveScalarEstimateNotProvided,
    PrimitiveScalarEstimateValue,
    PrimitiveScalarGapNotComputed,
    PrimitiveScalarGapValue,
    _projected_sigma_mS_cm,
    audit_primitive_oracle_closure,
    audit_primitive_oracle_closure_from_yaml,
    interpolate_primitive_closure,
    load_closure_fit,
    primitive_owner_audit_table,
    read_projected_primitive_yaml,
    validate_projected_primitive_artifact_input,
    write_failed_projected_primitive_yaml,
    write_projected_primitive_yaml,
)
from conductivity.physical_library.reduced_generator import (
    ReducedGeneratorSpecification,
    ReducedStateQuadrature,
    build_projected_generator_input,
)
from conductivity.physical_library.library_io import (
    _require_species,
    build_recipe_library_context_from_record,
    load_physical_library,
)
from conductivity.physical_library.library_io import validate_physical_library_records
from conductivity.physical_library.transition_surface_builder import (
    OneDimensionalTransitionBuildInput,
    build_one_dimensional_transition_surface,
)
from conductivity.physical_library.trajectory_primitives import (
    TrajectoryMarkovAdditiveSampleInput,
    compute_finite_process_component_drift_residuals,
    diagnose_finite_process_legality,
    project_sampled_trajectory_to_generator_primitives,
)
from conductivity.physical_library.projected_analytical_conductivity import (
    ProjectedConductivityResult,
    ProjectedPrimitiveInput,
    compute_symmetric_capacity_fluxes,
    compute_projected_analytical_conductivity_from_primitives,
)

FORBIDDEN_RECIPE_PRODUCTION_PRIMITIVE_ANCHOR_NAMES = (
    "interpolate_primitive_closure",
    "load_closure_fit",
    "read_projected_primitive_yaml",
    "compute_conductivity_from_primitive_yaml",
    "primitive_yaml_paths",
)

PHYSICAL_LIBRARY_ROOT = Path(__file__).resolve().parents[1] / "physical_library"


def _empty_transport_ownership_basis(
    coordinate_dimension: int,
) -> StateTransportOwnershipBasis:
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


def test_one_dimensional_committor_is_monotone_and_capacity_matches_resistance() -> None:
    grid_points = np.linspace(0.0, 1.0, 6)
    free_energy = np.zeros(grid_points.size, dtype=float)
    diffusivity = np.ones(grid_points.size, dtype=float) * 2.0

    result = solve_one_dimensional_committor(
        OneDimensionalCommittorInput(
            grid_points=grid_points,
            free_energy_J_mol=free_energy,
            diffusivity_m2_s=diffusivity,
            temperature_K=300.0,
            left_state_index=0,
            right_state_index=5,
        )
    )

    assert result.committor[0] == pytest.approx(0.0)
    assert result.committor[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result.committor) >= 0.0)
    assert result.capacity_integral == pytest.approx(2.0)


def test_one_dimensional_capacity_integral_survives_to_K() -> None:
    committor_result = solve_one_dimensional_committor(
        OneDimensionalCommittorInput(
            grid_points=np.linspace(0.0, 1.0, 6),
            free_energy_J_mol=np.zeros(6, dtype=float),
            diffusivity_m2_s=np.ones(6, dtype=float),
            temperature_K=300.0,
            left_state_index=0,
            right_state_index=5,
        )
    )

    capacity_fluxes = compute_symmetric_capacity_fluxes(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        np.asarray([[0, 1]], dtype=int),
        (np.asarray([[0.5]], dtype=float),),
        (np.asarray([1.0], dtype=float),),
        (np.asarray([[1.0]], dtype=float),),
        (np.asarray([0], dtype=int),),
        np.asarray([[1.0], [1.0]], dtype=float),
        np.asarray([0.0], dtype=float),
        300.0,
        2,
        np.asarray([committor_result.log_capacity_integral], dtype=float),
        np.asarray([False], dtype=bool),
        np.asarray([0.0], dtype=float),
        np.asarray([1000.0, 1000.0], dtype=float),
    )

    assert committor_result.capacity_integral > 0.0
    assert capacity_fluxes[0, 1] > 0.0
    assert capacity_fluxes[1, 0] == pytest.approx(capacity_fluxes[0, 1])


def test_transition_capacity_uses_precomputed_log_capacity() -> None:
    def unavailable_potential_energy_J_mol(point: np.ndarray) -> float:
        raise AssertionError("capacity flux assembly recomputed raw quadrature")

    def unavailable_mobility_tensor_m2_s(point: np.ndarray) -> np.ndarray:
        raise AssertionError("capacity flux assembly recomputed raw quadrature")

    capacity_fluxes = compute_symmetric_capacity_fluxes(
        unavailable_potential_energy_J_mol,
        unavailable_mobility_tensor_m2_s,
        np.asarray([[0, 1]], dtype=int),
        (np.asarray([[0.5]], dtype=float),),
        (np.asarray([1.0], dtype=float),),
        (np.asarray([[1.0]], dtype=float),),
        (np.asarray([0], dtype=int),),
        np.asarray([[1.0], [1.0]], dtype=float),
        np.asarray([0.0], dtype=float),
        300.0,
        2,
        np.asarray([np.log(2.0)], dtype=float),
        np.asarray([False], dtype=bool),
        np.asarray([0.0], dtype=float),
        np.asarray([1000.0, 1000.0], dtype=float),
    )

    assert capacity_fluxes[0, 1] == pytest.approx(2000.0)


def test_zero_capacity_is_not_silent() -> None:
    capacity_fluxes = compute_symmetric_capacity_fluxes(
        _zero_potential_J_mol,
        _unit_mobility_tensor_m2_s,
        np.asarray([[0, 1]], dtype=int),
        (np.asarray([[0.5]], dtype=float),),
        (np.asarray([1.0], dtype=float),),
        (np.asarray([[1.0]], dtype=float),),
        (np.asarray([0], dtype=int),),
        np.asarray([[1.0], [1.0]], dtype=float),
        np.asarray([0.0], dtype=float),
        300.0,
        2,
        np.asarray([-1.0e100], dtype=float),
        np.asarray([False], dtype=bool),
        np.asarray([0.0], dtype=float),
        np.asarray([1000.0, 1000.0], dtype=float),
    )

    assert capacity_fluxes == pytest.approx(np.zeros((2, 2), dtype=float))


def test_moment_bvp_builds_reversible_first_and_second_moments() -> None:
    committor = np.linspace(0.0, 1.0, 5)
    polarization = np.column_stack(
        (
            np.linspace(0.0, 2.0e-10, 5),
            np.zeros(5),
            np.zeros(5),
        )
    )
    result = solve_endpoint_moment_bvp(
        MomentBoundaryValueInput(
            grid_points=np.linspace(0.0, 1.0, 5),
            free_energy_J_mol=np.zeros(5),
            diffusivity_m2_s=np.ones(5),
            committor=committor,
            left_boundary_index=0,
            right_boundary_index=4,
            charge_polarization_by_grid=polarization,
            reactive_exit_weights=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
            temperature_K=300.0,
        )
    )
    first, second = build_path_moment_arrays(
        state_count=2,
        transition_pairs=np.asarray([[0, 1]], dtype=int),
        moment_results=(result,),
    )

    assert first[0, 1, 0] == pytest.approx(2.0e-10)
    assert first[1, 0, 0] == pytest.approx(-2.0e-10)
    assert second[0, 1, 0, 0] == pytest.approx(4.0e-20)
    assert second[1, 0, 0, 0] == pytest.approx(4.0e-20)


def test_projected_primitive_yaml_round_trip_computes_same_sigma(tmp_path) -> None:
    primitive_input = _two_state_primitive_input(
        capacity_flux_mol_m3_s=2.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=1.0e-10,
    )
    conductivity_result = _compute_from_primitive_input(primitive_input)
    primitive_path = tmp_path / "primitive.yaml"

    write_projected_primitive_yaml(
        primitive_path,
        ("left", "right"),
        primitive_input,
        _complete_artifact_result(conductivity_result),
    )
    artifact = read_projected_primitive_yaml(primitive_path)

    assert artifact.state_labels == ("left", "right")
    assert artifact.primitive_input.temperature_K == pytest.approx(T_REF_K)
    assert artifact.primitive_input.self_current_tensors_D_self_i_m2_s[0, 0, 0] == (
        pytest.approx(1.0e-10)
    )


def test_failed_projected_primitive_yaml_reports_component_drift_payload(tmp_path) -> None:
    primitive_path = tmp_path / "failed_primitive.yaml"
    write_failed_projected_primitive_yaml(
        primitive_path,
        PRIMITIVE_SCHEMA,
        "finite-state drift is not solvable on a generator component",
        {
            "component_drift_residuals": [
                {
                    "component_id": 3,
                    "weighted_drift_norm_mol_m2_s": 1.0e-18,
                    "top_edge_contributions": [
                        {
                            "component_id": 3,
                            "from_state_label": "state_a",
                            "to_state_label": "state_b",
                            "missing_reverse_event_candidate": True,
                        }
                    ],
                }
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="component_id.*3.*missing_reverse_event_candidate",
    ):
        read_projected_primitive_yaml(primitive_path)


def test_succeeded_projected_primitive_record_requires_projected_sigma() -> None:
    with pytest.raises(KeyError, match="missing finite sigma_mS_cm field"):
        _projected_sigma_mS_cm(
            {"schema": PRIMITIVE_SCHEMA},
            Path("primitive_without_sigma.yaml"),
        )


def test_succeeded_projected_primitive_record_rejects_nonfinite_projected_sigma() -> None:
    with pytest.raises(ValueError, match="sigma_mS_cm must be finite"):
        _projected_sigma_mS_cm(
            {"schema": PRIMITIVE_SCHEMA, "sigma_mS_cm": float("inf")},
            Path("primitive_nonfinite_sigma.yaml"),
        )


def test_succeeded_primitive_artifact_input_rejects_nonreciprocal_first_moments() -> None:
    primitive_input = ProjectedPrimitiveInput(
        state_concentrations_mol_m3=np.asarray([500.0, 500.0], dtype=float),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
            [[0.0, 2.0e12], [2.0e12, 0.0]],
            dtype=float,
        ),
        transition_first_moments_d_ij_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [2.0e-10, 0.0, 0.0]],
                [[1.0e-10, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        transition_second_moments_M_ij_m2=np.zeros((2, 2, 3, 3), dtype=float),
        self_current_tensors_D_self_i_m2_s=np.zeros((2, 3, 3), dtype=float),
        mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        mori_current_coupling_matrix_h=np.zeros((0, 3), dtype=float),
        temperature_K=T_REF_K,
        volume_m3=1.0,
    )

    with pytest.raises(ValueError, match="d_ji must equal -d_ij"):
        validate_projected_primitive_artifact_input(primitive_input)


def test_primitive_closure_interpolates_anchor_tensors(tmp_path) -> None:
    first_input = _two_state_primitive_input(
        capacity_flux_mol_m3_s=2.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=1.0e-10,
    )
    second_input = _two_state_primitive_input(
        capacity_flux_mol_m3_s=2.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=3.0e-10,
    )
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    for primitive_path, primitive_input in (
        (first_path, first_input),
        (second_path, second_input),
    ):
        write_projected_primitive_yaml(
            primitive_path,
            ("left", "right"),
            primitive_input,
            _complete_artifact_result(_compute_from_primitive_input(primitive_input)),
        )
    closure_fit = load_closure_fit(
        primitive_yaml_paths=(first_path, second_path),
        feature_vectors=(np.asarray([0.0]), np.asarray([1.0])),
        length_scales=np.asarray([1.0e6]),
    )

    interpolated = interpolate_primitive_closure(
        closure_fit,
        np.asarray([0.5], dtype=float),
    )

    assert interpolated.self_current_tensors_D_self_i_m2_s[0, 0, 0] == pytest.approx(
        2.0e-10
    )


def test_failed_projected_primitive_yaml_cannot_be_closure_anchor(tmp_path) -> None:
    failed_anchor_path = tmp_path / "failed_anchor.yaml"
    write_failed_projected_primitive_yaml(
        failed_anchor_path,
        PRIMITIVE_SCHEMA,
        "finite-state drift is not solvable on a generator component",
        {
            "component_drift_residuals": [
                {
                    "component_id": 0,
                    "weighted_drift_norm_mol_m2_s": 2.0e-18,
                    "top_edge_contributions": [],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="invalid for readout"):
        load_closure_fit(
            primitive_yaml_paths=(failed_anchor_path,),
            feature_vectors=(np.asarray([0.0], dtype=float),),
            length_scales=np.asarray([1.0], dtype=float),
        )


def test_primitive_oracle_audit_separates_projection_recipe_and_scalar_gaps() -> None:
    trajectory_primitives = _two_state_primitive_input(
        capacity_flux_mol_m3_s=2.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=1.0e-10,
    )
    projected_primitives = _two_state_primitive_input(
        capacity_flux_mol_m3_s=3.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=1.0e-10,
    )
    recipe_primitives = _two_state_primitive_input(
        capacity_flux_mol_m3_s=3.0e12,
        first_moment_m=2.0e-10,
        self_diffusion_m2_s=4.0e-10,
    )

    report = audit_primitive_oracle_closure(
        trajectory_primitives,
        projected_primitives,
        recipe_primitives,
        PrimitiveExternalScalarInput(
            green_kubo=PrimitiveScalarEstimateValue(sigma_mS_cm=4.0),
            einstein_helfand=PrimitiveScalarEstimateNotProvided(
                scalar_name="einstein_helfand"
            ),
        ),
        PrimitiveExternalScalarInput(
            green_kubo=PrimitiveScalarEstimateValue(sigma_mS_cm=5.0),
            einstein_helfand=PrimitiveScalarEstimateNotProvided(
                scalar_name="einstein_helfand"
            ),
        ),
        PrimitiveExternalScalarInput(
            green_kubo=PrimitiveScalarEstimateValue(sigma_mS_cm=7.0),
            einstein_helfand=PrimitiveScalarEstimateNotProvided(
                scalar_name="einstein_helfand"
            ),
        ),
    )

    assert report.projection_gap.K_gap_mol_m3_s > 0.0
    assert report.projection_gap.Q_gap_s_inv > 0.0
    assert report.projection_gap.d_gap_m == pytest.approx(0.0)
    assert report.projection_gap.D_self_gap_m2_s == pytest.approx(0.0)
    assert report.recipe_primitive_gap.K_gap_mol_m3_s == pytest.approx(0.0)
    assert report.recipe_primitive_gap.d_gap_m > 0.0
    assert report.recipe_primitive_gap.D_self_gap_m2_s > 0.0
    assert report.trajectory_norms.c_norm_mol_m3 > 0.0
    assert report.trajectory_norms.Q_norm_s_inv > 0.0
    assert report.scalar_gap.finite_projected_sigma_gap_mS_cm > 0.0
    assert isinstance(report.scalar_gap.green_kubo, PrimitiveScalarGapValue)
    assert report.scalar_gap.green_kubo.gap_mS_cm == pytest.approx(3.0)
    assert isinstance(report.scalar_gap.einstein_helfand, PrimitiveScalarGapNotComputed)


def test_primitive_oracle_audit_rejects_failed_yaml_anchor(tmp_path) -> None:
    primitive_input = _two_state_primitive_input(
        capacity_flux_mol_m3_s=2.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=1.0e-10,
    )
    valid_path = tmp_path / "valid.yaml"
    failed_path = tmp_path / "failed.yaml"
    write_projected_primitive_yaml(
        valid_path,
        ("left", "right"),
        primitive_input,
        _complete_artifact_result(_compute_from_primitive_input(primitive_input)),
    )
    write_failed_projected_primitive_yaml(
        failed_path,
        PRIMITIVE_SCHEMA,
        "finite-state drift is not solvable on a generator component",
        {
            "component_drift_residuals": [
                {
                    "component_id": 0,
                    "weighted_drift_norm_mol_m2_s": 2.0e-18,
                    "top_edge_contributions": [],
                }
            ],
        },
    )
    scalar_input = PrimitiveExternalScalarInput(
        green_kubo=PrimitiveScalarEstimateNotProvided(scalar_name="green_kubo"),
        einstein_helfand=PrimitiveScalarEstimateNotProvided(
            scalar_name="einstein_helfand"
        ),
    )

    with pytest.raises(ValueError, match="invalid for readout"):
        audit_primitive_oracle_closure_from_yaml(
            valid_path,
            failed_path,
            valid_path,
            scalar_input,
            scalar_input,
            scalar_input,
        )


def test_reduced_generator_assembles_projected_input() -> None:
    specification = ReducedGeneratorSpecification(
        potential_energy_J_mol=_zero_potential_J_mol,
        mobility_tensor_m2_s=_unit_mobility_tensor_m2_s,
        charge_polarization_gradient=_single_axis_charge_gradient,
        memory_coordinate_gradient=_empty_memory_gradient,
        state_quadratures=(
            ReducedStateQuadrature(
                points=np.asarray([[0.0]]),
                weights=np.asarray([1.0]),
                stoichiometry=np.asarray([1.0]),
                self_current_projector=np.eye(1),
                transport_ownership_bases=(
                    _empty_transport_ownership_basis(1),
                ),
                relative_displacement_fluctuations_m=np.empty((0, 3)),
                relative_displacement_mobility_m2_s=np.empty((0, 0)),
                relative_center_charge_numbers=np.empty(0),
            ),
        ),
        transition_quadratures=(),
        total_component_concentrations_mol_m3=np.asarray([1.0]),
        temperature_K=300.0,
        volume_m3=1.0,
    )

    projected_input = build_projected_generator_input(specification)

    assert projected_input.basin_stoichiometry.shape == (1, 1)
    assert projected_input.transition_pair_indices.shape == (0, 2)


def test_physical_object_builder_computes_site_level_objects() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    configuration = _two_lithium_configuration()

    bundle = build_physical_objects(
        records,
        configuration,
        temperature_K=T_REF_K,
        dielectric_constant=20.0,
        viscosity_Pa_s=1.0e-3,
        ionic_strength_mol_m3=1000.0,
        local_packing_fraction=compute_local_packing_fraction(records, configuration),
    )

    assert np.isfinite(bundle.potential_energy_J_mol)
    assert bundle.mobility_tensor_m2_s.shape == (6, 6)
    assert np.all(np.diag(bundle.mobility_tensor_m2_s) > 0.0)
    assert bundle.charge_polarization_m[0] == pytest.approx(5.0e-10)
    assert bundle.charge_polarization_gradient[0, 0] == pytest.approx(1.0)
    assert bundle.charge_polarization_gradient[0, 3] == pytest.approx(1.0)
    assert bundle.local_packing_fraction > 0.0


def test_charge_polarization_uses_formal_ion_centers_not_neutral_partial_charges() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    neutral_configuration = _single_fec_configuration()
    neutral_bundle = build_physical_objects(
        records,
        neutral_configuration,
        temperature_K=T_REF_K,
        dielectric_constant=20.0,
        viscosity_Pa_s=1.0e-3,
        ionic_strength_mol_m3=1000.0,
        local_packing_fraction=compute_local_packing_fraction(
            records,
            neutral_configuration,
        ),
    )

    assert np.array_equal(
        neutral_bundle.charge_polarization_m,
        np.zeros(3, dtype=float),
    )
    assert np.array_equal(
        neutral_bundle.charge_polarization_gradient,
        np.zeros((3, neutral_configuration.positions_m.size), dtype=float),
    )

    ion_configuration = _lithium_full_pf6_configuration(pair_distance_m=5.0e-10)
    ion_bundle = build_physical_objects(
        records,
        ion_configuration,
        temperature_K=T_REF_K,
        dielectric_constant=20.0,
        viscosity_Pa_s=1.0e-3,
        ionic_strength_mol_m3=1000.0,
        local_packing_fraction=compute_local_packing_fraction(records, ion_configuration),
    )

    assert ion_bundle.charge_polarization_m[0] == pytest.approx(-5.0e-10)
    assert ion_bundle.charge_polarization_gradient[0, 0] == pytest.approx(1.0)
    phosphorus_site_index = 2
    pf6_site_indices = tuple(range(1, len(ion_configuration.species_names)))
    pf6_site_masses_kg = np.asarray(
        [
            records.species_records["PF6-"]["sites"][site_id]["mass_kg"]
            for site_id in ion_configuration.site_ids[1:]
        ],
        dtype=float,
    )
    assert ion_bundle.charge_polarization_gradient[
        0,
        3 * phosphorus_site_index,
    ] == pytest.approx(
        -float(pf6_site_masses_kg[phosphorus_site_index - 1])
        / float(np.sum(pf6_site_masses_kg))
    )
    assert sum(
        ion_bundle.charge_polarization_gradient[0, 3 * site_index]
        for site_index in pf6_site_indices
    ) == pytest.approx(-1.0)


def test_pair_basin_assignment_uses_configured_cutoffs() -> None:
    basis_record = {
        "pair_basins": {
            "r_CIP_m": 3.0e-10,
            "r_SSIP_m": 6.0e-10,
            "r_free_m": 1.0e-9,
        }
    }

    assert assign_pair_basin(2.5e-10, basis_record) == PairBasin.CONTACT_ION_PAIR
    assert assign_pair_basin(4.0e-10, basis_record) == (
        PairBasin.SOLVENT_SEPARATED_ION_PAIR
    )
    assert assign_pair_basin(8.0e-10, basis_record) == PairBasin.TRANSITION
    assert assign_pair_basin(1.2e-9, basis_record) == PairBasin.FREE


def test_physical_generator_builder_exposes_physical_functions() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    configuration = _two_lithium_configuration()
    coordinate_count = configuration.positions_m.size + LOCAL_FIELD_VECTOR_LENGTH
    generator_specification = build_reduced_generator_specification_from_physical_objects(
        PhysicalGeneratorBuildInput(
            records=records,
            template_configuration=configuration,
            state_quadratures=(
                PhysicalStateQuadrature(
                    label="charged_pair",
                    configurations=(configuration,),
                    local_fields=(
                        _local_fields(records, configuration, 20.0, 1.0e-3, 1000.0),
                    ),
                    weights=np.asarray([1.0], dtype=float),
                    stoichiometry=np.asarray([2.0], dtype=float),
                    self_current_projector=np.eye(coordinate_count, dtype=float),
                    transport_ownership_bases=(
                        _empty_transport_ownership_basis(coordinate_count),
                    ),
                    relative_displacement_fluctuations_m=np.empty((0, 3)),
                    relative_displacement_mobility_m2_s=np.empty((0, 0)),
                    relative_center_charge_numbers=np.empty(0),
                ),
            ),
            transition_quadratures=(),
            memory_coordinate_gradient_functions=(),
            total_component_concentrations_mol_m3=np.asarray([1.0], dtype=float),
            temperature_K=T_REF_K,
            volume_m3=1.0,
        )
    )
    point = generator_specification.state_quadratures[0].points[0]
    registered_coordinate_count = coordinate_count + 1

    assert np.isfinite(generator_specification.potential_energy_J_mol(point))
    assert generator_specification.mobility_tensor_m2_s(point).shape == (
        registered_coordinate_count,
        registered_coordinate_count,
    )
    assert generator_specification.charge_polarization_gradient(point).shape == (
        3,
        registered_coordinate_count,
    )
    assert generator_specification.memory_coordinate_gradient(point).shape == (
        4,
        registered_coordinate_count,
    )


def test_bulk_matrix_properties_exclude_additive_local_field_corrections() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    matrix_composition = MixtureComposition(
        solvent_volume_fractions={"EC": 0.3, "DMC": 0.7},
        ion_concentrations_mol_m3={"Li+": 1000.0, "PF6-": 1000.0},
        additive_weight_fractions={},
    )
    additive_composition = MixtureComposition(
        solvent_volume_fractions=matrix_composition.solvent_volume_fractions,
        ion_concentrations_mol_m3=matrix_composition.ion_concentrations_mol_m3,
        additive_weight_fractions={"FEC": 0.1},
    )

    assert compute_bulk_dielectric_constant(
        records, additive_composition
    ) == pytest.approx(compute_bulk_dielectric_constant(records, matrix_composition))
    assert compute_bulk_viscosity_Pa_s(
        records, additive_composition
    ) == pytest.approx(compute_bulk_viscosity_Pa_s(records, matrix_composition))
    assert records.mixture_record["property_ownership"] == {
        "dielectric": {
            "bulk": "solvent_matrix_reference",
            "local": "ionic_strength_decrement",
        },
        "viscosity": {
            "bulk": "solvent_matrix_reference",
            "local": "Jones_Dole_packing_and_state_additive_microviscosity",
        },
        "additive_properties": {
            "bulk": "excluded",
            "local": "state_configuration_occupancy",
        },
    }


def test_mixture_basin_memory_and_transition_builders_are_executable() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    first_configuration = _lithium_pf6_configuration(pair_distance_m=4.0e-10)
    second_configuration = _lithium_pf6_configuration(pair_distance_m=8.0e-10)
    mixture = compute_mixture_closures(
        records,
        MixtureComposition(
            solvent_volume_fractions={"EC": 0.3, "DMC": 0.7},
            ion_concentrations_mol_m3={"Li+": 1000.0, "PF6-": 1000.0},
            additive_weight_fractions={},
        ),
        temperature_K=T_REF_K,
    )
    state_definition = build_state_definition(
        records,
        first_configuration,
        mixture,
        component_names=("Li+", "PF6-"),
    )
    memory_coordinates = build_default_memory_coordinates(records, first_configuration)
    memory_values = combine_memory_values(memory_coordinates, first_configuration)
    memory_gradients = combine_memory_gradients(memory_coordinates, first_configuration)
    coordinate_count = first_configuration.positions_m.size
    generator_coordinate_count = coordinate_count + LOCAL_FIELD_VECTOR_LENGTH
    endpoint_displacement = (
        compute_charge_polarization_m(records, second_configuration)
        - compute_charge_polarization_m(records, first_configuration)
    )
    transition = build_one_dimensional_transition_surface(
        OneDimensionalTransitionBuildInput(
            from_state_index=0,
            to_state_index=1,
            transition_family="free_to_SSIP",
            transport_ownership=TransportOwnership.BOUNDED_MEMORY,
            grid_configurations=(first_configuration, second_configuration),
            local_fields=(
                _local_fields(
                    records,
                    first_configuration,
                    mixture.dielectric_constant,
                    mixture.viscosity_Pa_s,
                    mixture.ionic_strength_mol_m3,
                ),
                _local_fields(
                    records,
                    second_configuration,
                    mixture.dielectric_constant,
                    mixture.viscosity_Pa_s,
                    mixture.ionic_strength_mol_m3,
                ),
            ),
            reaction_coordinate_values=np.asarray([4.0e-10, 8.0e-10], dtype=float),
            reaction_coordinate_gradients=np.asarray(
                [
                    _pair_distance_gradient(first_configuration),
                    _pair_distance_gradient(second_configuration),
                ],
                dtype=float,
            ),
            free_energy_J_mol=np.zeros(2, dtype=float),
            diffusivity_m2_s=np.ones(2, dtype=float),
            temperature_K=T_REF_K,
            left_state_grid_index=0,
            right_state_grid_index=1,
            surface_state_indices=np.asarray([0, 1], dtype=int),
            path_start_configurations=(first_configuration,),
            path_end_configurations=(second_configuration,),
            path_weights=np.asarray([1.0], dtype=float),
            first_displacement_moment_m=endpoint_displacement,
            second_displacement_moment_m2=np.outer(
                endpoint_displacement,
                endpoint_displacement,
            ),
        )
    )
    generator_specification = build_reduced_generator_specification_from_physical_objects(
        PhysicalGeneratorBuildInput(
            records=records,
            template_configuration=first_configuration,
            state_quadratures=(
                PhysicalStateQuadrature(
                    label="first",
                    configurations=(first_configuration,),
                    local_fields=(
                        _local_fields(
                            records,
                            first_configuration,
                            mixture.dielectric_constant,
                            mixture.viscosity_Pa_s,
                            mixture.ionic_strength_mol_m3,
                        ),
                    ),
                    weights=np.asarray([1.0], dtype=float),
                    stoichiometry=np.asarray([1.0, 1.0], dtype=float),
                    self_current_projector=np.eye(
                        generator_coordinate_count,
                        dtype=float,
                    ),
                    transport_ownership_bases=(
                        _empty_transport_ownership_basis(coordinate_count),
                    ),
                    relative_displacement_fluctuations_m=np.empty((0, 3)),
                    relative_displacement_mobility_m2_s=np.empty((0, 0)),
                    relative_center_charge_numbers=np.empty(0),
                ),
                PhysicalStateQuadrature(
                    label="second",
                    configurations=(second_configuration,),
                    local_fields=(
                        _local_fields(
                            records,
                            second_configuration,
                            mixture.dielectric_constant,
                            mixture.viscosity_Pa_s,
                            mixture.ionic_strength_mol_m3,
                        ),
                    ),
                    weights=np.asarray([1.0], dtype=float),
                    stoichiometry=np.asarray([1.0, 1.0], dtype=float),
                    self_current_projector=np.eye(
                        generator_coordinate_count,
                        dtype=float,
                    ),
                    transport_ownership_bases=(
                        _empty_transport_ownership_basis(coordinate_count),
                    ),
                    relative_displacement_fluctuations_m=np.empty((0, 3)),
                    relative_displacement_mobility_m2_s=np.empty((0, 0)),
                    relative_center_charge_numbers=np.empty(0),
                ),
            ),
            transition_quadratures=(transition.transition_quadrature,),
            memory_coordinate_gradient_functions=(),
            total_component_concentrations_mol_m3=np.asarray([1.0, 1.0], dtype=float),
            temperature_K=T_REF_K,
            volume_m3=1.0,
        )
    )

    assert mixture.dielectric_constant > 0.0
    assert mixture.viscosity_Pa_s > 0.0
    assert mixture.ionic_strength_mol_m3 == pytest.approx(1000.0)
    assert state_definition.stoichiometry.tolist() == [1.0, 1.0]
    assert memory_values.shape[0] == len(memory_coordinates)
    assert memory_gradients.shape == (len(memory_coordinates), coordinate_count)
    assert transition.transition_quadrature.committor_gradients.shape == (
        2,
        coordinate_count,
    )
    assert generator_specification.transition_quadratures[0].path_displacements_m.shape == (
        1,
        3,
    )


def test_non_pair_transition_family_declares_executable_coordinate() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    pair_record = records.transition_record["transition_records"]["free_to_SSIP"]
    ligand_record = records.transition_record["transition_records"]["Li_to_Li_ligand"]
    generator_construction._validate_transition_family_reaction_coordinate(
        "free_to_SSIP",
        pair_record,
    )
    generator_construction._validate_transition_family_reaction_coordinate(
        "Li_to_Li_ligand",
        ligand_record,
    )
    configuration = generator_construction._configuration_with_reduced_coordinate_values(
        records,
        _lithium_pf6_fec_configuration(pair_distance_m=4.0e-10),
        {
            "Li_anion_distance": 4.0e-10,
            "Li_solvent_coordination": 0.0,
            "Li_ligand_coordination": 0.8,
            "Li_anion_coordination": 0.8,
            "anion_orientation": 0.0,
            "local_packing_fraction": 0.2,
            "local_ionic_strength": 1000.0,
            "local_dielectric": 30.0,
            "local_viscosity": 1.0e-3,
            "atmosphere_polarization": 0.0,
            "cage_coordinate": 0.0,
            "partner_residence_coordinate": 0.5,
            "cluster_coordinate": 0.0,
            "identity_coordinate": 0.0,
            "structural_hop_coordinate": 0.0,
        },
    )
    gradient = generator_construction._reaction_coordinate_gradient(
        records,
        configuration,
        ligand_record,
    )

    assert gradient.shape == (configuration.positions_m.size,)
    assert np.any(np.abs(gradient) > 0.0)


def test_physical_generator_registry_keeps_species_distinct_at_identical_coordinates() -> None:
    pf6_configuration = _lithium_pf6_configuration(pair_distance_m=4.0e-10)
    fsi_configuration = SiteConfiguration(
        species_names=("Li+", "FSI-"),
        molecule_ids=np.asarray(pf6_configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(pf6_configuration.site_ids, dtype=int),
        positions_m=np.asarray(pf6_configuration.positions_m, dtype=float),
        unwrapped_positions_m=np.asarray(
            pf6_configuration.unwrapped_positions_m,
            dtype=float,
        ),
        box_lengths_m=np.asarray(pf6_configuration.box_lengths_m, dtype=float),
    )
    local_fields = PhysicalLocalFields(
        dielectric_constant=30.0,
        viscosity_Pa_s=1.0e-3,
        ionic_strength_mol_m3=1000.0,
        local_packing_fraction=0.2,
    )
    signatures = (
        physical_generator_builder._configuration_identity_signature(
            pf6_configuration
        ),
        physical_generator_builder._configuration_identity_signature(
            fsi_configuration
        ),
    )
    configuration_identity_indices = {
        signature: identity_index
        for identity_index, signature in enumerate(sorted(signatures), start=1)
    }
    common_coordinate_count = (
        flatten_configuration_with_local_fields(pf6_configuration, local_fields).size
        + 1
    )
    registry = {}

    physical_generator_builder._register_configuration_points(
        registry,
        (pf6_configuration, fsi_configuration),
        (local_fields, local_fields),
        common_coordinate_count,
        configuration_identity_indices,
    )

    assert len(registry) == 2
    assert {
        registered_configuration.species_names
        for registered_configuration, _registered_fields in registry.values()
    } == {("Li+", "PF6-"), ("Li+", "FSI-")}


def test_mixed_salt_recipe_keeps_additive_stoichiometry_and_transition_channel() -> None:
    recipe_context = generator_construction.build_recipe_library_context(
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_lifsi_fec.yaml",
        PHYSICAL_LIBRARY_ROOT,
    )
    records = recipe_context.library_records
    mixture = generator_construction.compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    numerical_options = generator_construction.NumericalOptions(
        reference_box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
        volume_m3=1.0e-24,
        state_quadrature_order=2,
        transition_grid_count=7,
    )
    template_configuration = generator_construction.build_template_site_configuration(
        records,
        recipe_context,
        mixture,
        numerical_options,
    )
    state_quadratures = generator_construction.build_all_state_quadratures(
        records,
        template_configuration,
        mixture,
        recipe_context,
        numerical_options,
    )
    projected_components = generator_construction._projected_mass_balance_components(
        recipe_context
    )
    component_names = tuple(component.name for component in projected_components)
    state_stoichiometry = np.asarray(
        [state_quadrature.stoichiometry for state_quadrature in state_quadratures],
        dtype=float,
    )
    active_additive_name = _single_active_additive_name(recipe_context)
    active_additive_component_index = component_names.index(active_additive_name)
    assert component_names == ("Li+", "FSI-", "PF6-", active_additive_name)
    assert np.any(state_stoichiometry[:, active_additive_component_index] == 1.0)
    assert np.any(state_stoichiometry[:, active_additive_component_index] == 0.0)
    assert any(
        state_quadrature.label.startswith("free_additive_reservoir")
        for state_quadrature in state_quadratures
    )
    additive_state_quadratures = tuple(
        state_quadrature
        for state_quadrature in state_quadratures
        if state_quadrature.stoichiometry[active_additive_component_index] == 1.0
        and not state_quadrature.label.startswith("free_additive_reservoir")
    )
    assert additive_state_quadratures
    for state_quadrature in additive_state_quadratures:
        assert f"|{active_additive_name}:none|" not in state_quadrature.label
        for configuration in state_quadrature.configurations:
            assert active_additive_name in configuration.species_names
            assert _nearest_species_center_distance_m(
                records,
                configuration,
                active_additive_name,
            ) < float(records.basis_record["pair_basins"]["r_free_m"])
    non_additive_state_quadratures = tuple(
        state_quadrature
        for state_quadrature in state_quadratures
        if state_quadrature.stoichiometry[active_additive_component_index] == 0.0
    )
    assert non_additive_state_quadratures
    for state_quadrature in non_additive_state_quadratures:
        for configuration in state_quadrature.configurations:
            assert active_additive_name not in configuration.species_names

    representative_additive_state = additive_state_quadratures[0]
    representative_non_additive_state = non_additive_state_quadratures[0]
    additive_configuration = representative_additive_state.configurations[0]
    non_additive_configuration = representative_non_additive_state.configurations[0]
    additive_local_fields = representative_additive_state.local_fields[0]
    non_additive_local_fields = representative_non_additive_state.local_fields[0]
    additive_physical_objects = _physical_objects_for_local_fields(
        records,
        additive_configuration,
        recipe_context.temperature_K,
        additive_local_fields,
    )
    non_additive_physical_objects = _physical_objects_for_local_fields(
        records,
        non_additive_configuration,
        recipe_context.temperature_K,
        non_additive_local_fields,
    )
    additive_ligand_coordination = generator_construction.compute_role_coordination_number(
        records,
        additive_configuration,
        center_role=generator_construction.SpeciesRole.CATION.value,
        ligand_role=generator_construction.SpeciesRole.ADDITIVE.value,
        switch_name="Li_ligand",
    )

    assert additive_local_fields.viscosity_Pa_s > non_additive_local_fields.viscosity_Pa_s
    assert compute_local_packing_fraction(records, additive_configuration) > (
        compute_local_packing_fraction(records, non_additive_configuration)
    )
    assert additive_ligand_coordination > 0.0
    assert additive_physical_objects.potential_energy_J_mol != pytest.approx(
        non_additive_physical_objects.potential_energy_J_mol
    )
    assert additive_physical_objects.mobility_tensor_m2_s.shape != (
        non_additive_physical_objects.mobility_tensor_m2_s.shape
    )


def _compute_conductivity_from_recipe_context(
    recipe_context: generator_construction.RecipeBuildResult,
    numerical_options: generator_construction.NumericalOptions,
) -> generator_construction.ProjectedConductivityResult:
    records = recipe_context.library_records
    mixture = generator_construction.compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    template_configuration = generator_construction.build_template_site_configuration(
        records,
        recipe_context,
        mixture,
        numerical_options,
    )
    state_quadratures = generator_construction.build_all_state_quadratures(
        records,
        template_configuration,
        mixture,
        recipe_context,
        numerical_options,
    )
    transition_quadratures = generator_construction.build_all_transition_quadratures(
        records,
        state_quadratures,
        template_configuration,
        mixture,
        recipe_context.temperature_K,
        numerical_options,
    )
    memory_gradient_functions = (
        generator_construction.build_all_memory_coordinate_gradients(
            records,
            template_configuration,
            state_quadratures,
            generator_construction.finite_generator_transition_edges(
                records,
                state_quadratures,
                recipe_context.temperature_K,
            ),
            mixture,
            numerical_options,
        )
    )
    projected_components = generator_construction._projected_mass_balance_components(
        recipe_context
    )
    component_concentrations = np.asarray(
        [component.concentration_mol_m3 for component in projected_components],
        dtype=float,
    )
    reduced_specification = (
        generator_construction.build_reduced_generator_specification_from_physical_objects(
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
    )
    generator_input = generator_construction.build_projected_generator_input(
        generator_construction._normalize_potential_energy_reference(
            reduced_specification
        )
    )
    conductivity_result = (
        generator_construction._compute_projected_analytical_conductivity_from_input(
            generator_input
        )
    )
    return conductivity_result


def test_recipe_generator_production_path_excludes_primitive_anchor_interpolation():
    production_sources = (
        inspect.getsource(generator_construction),
        inspect.getsource(property_db_validation),
    )
    for production_source in production_sources:
        for forbidden_name in FORBIDDEN_RECIPE_PRODUCTION_PRIMITIVE_ANCHOR_NAMES:
            assert forbidden_name not in production_source


def test_recipe_context_is_invariant_to_mapping_order() -> None:
    forward_record = {
        "temperature_K": T_REF_K,
        "solvents_vv": {"EC": 0.34, "DMC": 0.66},
        "salts_mol_l": {"Li+": 1.1, "PF6-": 0.8, "FSI-": 0.3},
        "additives_weight_fraction": {
            "TPP": 0.02,
            "PS": 0.01,
            "VC": 0.005,
            "LiDFOB": 0.005,
        },
    }
    reverse_record = {
        "temperature_K": T_REF_K,
        "solvents_vv": dict(reversed(tuple(forward_record["solvents_vv"].items()))),
        "salts_mol_l": dict(reversed(tuple(forward_record["salts_mol_l"].items()))),
        "additives_weight_fraction": dict(
            reversed(tuple(forward_record["additives_weight_fraction"].items()))
        ),
    }
    forward_context = build_recipe_library_context_from_record(
        forward_record,
        PHYSICAL_LIBRARY_ROOT,
    )
    reverse_context = build_recipe_library_context_from_record(
        reverse_record,
        PHYSICAL_LIBRARY_ROOT,
    )

    assert forward_context.components == reverse_context.components
    assert forward_context.solvent_volume_fractions == (
        reverse_context.solvent_volume_fractions
    )
    assert forward_context.additive_weight_fractions == (
        reverse_context.additive_weight_fractions
    )


def test_reference_recipe_primitive_owner_audit_table() -> None:
    numerical_options = generator_construction.NumericalOptions(
        reference_box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
        volume_m3=1.0,
        state_quadrature_order=2,
        transition_grid_count=5,
    )
    recipe_paths = (
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_1m.yaml",
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_lifsi_fec.yaml",
    )
    expected_ranges_mS_cm = {
        "recipe_ec_dmc_lipf6_1m.yaml": (8.0, 10.0),
        "recipe_ec_dmc_lipf6_lifsi_fec.yaml": (11.5, 12.5),
    }
    expected_owners = (
        "projected_readout",
        "primitive_legality",
        "basis_convergence",
        "primitive_prediction_readiness",
    )
    expected_statuses = ("correct", "correct", "correct", "correct")

    for recipe_path in recipe_paths:
        conductivity_result = generator_construction.compute_conductivity_from_recipe(
            recipe_path,
            PHYSICAL_LIBRARY_ROOT,
            numerical_options,
        )
        owner_rows = primitive_owner_audit_table(
            recipe_path.name,
            conductivity_result,
        )
        expected_min_mS_cm, expected_max_mS_cm = expected_ranges_mS_cm[recipe_path.name]
        assert tuple(row.primitive_owner for row in owner_rows) == expected_owners
        assert tuple(row.correctness_status for row in owner_rows) == expected_statuses
        assert owner_rows[3].detail == "primitive_prediction"
        assert expected_min_mS_cm <= conductivity_result.sigma_mS_cm <= expected_max_mS_cm
        ownership_state_tensors = conductivity_result.effect_attribution[
            "transport_ownership_state_tensors"
        ]
        assert ownership_state_tensors
        for ownership_state_tensor in ownership_state_tensors:
            short_mobility_tensor = ownership_state_tensor["D_Q_short"]
            unowned_mobility_tensor = ownership_state_tensor["D_Q_unowned"]
            ownership_scale_m2_s = max(
                float(np.linalg.norm(short_mobility_tensor, ord=2)),
                np.finfo(float).tiny,
            )
            assert float(np.linalg.norm(unowned_mobility_tensor, ord=2)) <= (
                64.0 * np.finfo(float).eps * ownership_scale_m2_s
            )
            assert np.allclose(
                ownership_state_tensor["D_Q_short"],
                ownership_state_tensor["D_Q_dc_self"]
                + ownership_state_tensor["D_Q_transition_owned"]
                + ownership_state_tensor["D_Q_bounded_memory"]
                + ownership_state_tensor["D_Q_diagnostic"],
                rtol=1.0e-10,
                atol=1.0e-30,
            )
        assert conductivity_result.effect_attribution[
            "primitive_prediction_scalar_label"
        ] == "primitive_prediction"
        direct_ledger = conductivity_result.effect_attribution
        assert direct_ledger["B_total_trace_mol_m_s"] == pytest.approx(
            direct_ledger["B_self_tangent_trace_mol_m_s"]
            + direct_ledger["B_transition_trace_mol_m_s"]
        )
        assert direct_ledger["B_self_full_trace_mol_m_s"] == pytest.approx(
            direct_ledger["B_self_tangent_trace_mol_m_s"]
            + direct_ledger["B_overlap_removed_trace_mol_m_s"]
        )
        assert len(direct_ledger["state_drift_b_i_m_s"]) == len(
            conductivity_result.state_concentrations_mol_m3
        )
        assert direct_ledger["state_drift_components"]
        assert all(
            len(state_label.split("|")) == generator_construction.STATE_KEY_LENGTH
            for state_label in direct_ledger["state_labels"]
        )
        mori_mode_ledger = direct_ledger["mori_mode_ledger"]
        assert all(
            mode["transport_ownership"] == "bounded_memory"
            and mode["A_mu_mu"] > 0.0
            and mode["h_mu_A_pseudoinverse_h_contribution"] >= 0.0
            and mode["state_support"]
            for mode in mori_mode_ledger
        )
        for resistance_trace_key in (
            "state_resistance_stokes_traces_kg_s",
            "state_resistance_free_volume_traces_kg_s",
            "state_resistance_charge_cloud_traces_kg_s",
            "state_resistance_atmosphere_traces_kg_s",
            "state_resistance_total_traces_kg_s",
        ):
            resistance_trace_values = np.asarray(
                conductivity_result.effect_attribution[resistance_trace_key],
                dtype=float,
            )
            assert resistance_trace_values.shape == (
                conductivity_result.state_concentrations_mol_m3.size,
            )
            assert np.all(np.isfinite(resistance_trace_values))


def test_physical_library_rejects_scalar_sigma_fitted_parameter_provenance() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    fitted_mixture_record = {
        **records.mixture_record,
        "parameter_provenance": {
            **records.mixture_record["parameter_provenance"],
            "mobility.free_volume_exponent": "fitted_to_scalar_sigma",
        },
    }
    fitted_records = records.__class__(
        root=records.root,
        manifest=records.manifest,
        species_records=records.species_records,
        pair_records=records.pair_records,
        mixture_record=fitted_mixture_record,
        basis_record=records.basis_record,
        transition_record=records.transition_record,
        memory_record=records.memory_record,
    )

    with pytest.raises(ValueError, match="rejects scalar-sigma-fitted parameter"):
        validate_physical_library_records(fitted_records)


def _single_active_additive_name(
    recipe_context: generator_construction.RecipeBuildResult,
) -> str:
    additive_names = tuple(recipe_context.additive_weight_fractions)
    if len(additive_names) != 1:
        raise AssertionError("state-local additive test requires one active additive")
    return additive_names[0]


def _recipe_context_with_perturbed_loadings(
    recipe_context: generator_construction.RecipeBuildResult,
    salt_scale: float,
    target_additive_weight_fraction: float,
) -> generator_construction.RecipeBuildResult:
    active_additive_name = _single_active_additive_name(recipe_context)
    base_additive_weight_fraction = recipe_context.additive_weight_fractions[
        active_additive_name
    ]
    additive_concentration_scale = (
        target_additive_weight_fraction / base_additive_weight_fraction
    )
    components = tuple(
        generator_construction.RecipeComponentLoading(
            name=component.name,
            concentration_mol_m3=(
                component.concentration_mol_m3 * salt_scale
                if component.role == "salt_component"
                else component.concentration_mol_m3 * additive_concentration_scale
                if component.name == active_additive_name
                else component.concentration_mol_m3
            ),
            role=component.role,
        )
        for component in recipe_context.components
    )
    additive_weight_fractions = {
        additive_name: (
            target_additive_weight_fraction
            if additive_name == active_additive_name
            else current_additive_weight_fraction
        )
        for additive_name, current_additive_weight_fraction in (
            recipe_context.additive_weight_fractions.items()
        )
    }
    return generator_construction.RecipeBuildResult(
        temperature_K=recipe_context.temperature_K,
        components=components,
        solvent_volume_fractions=dict(recipe_context.solvent_volume_fractions),
        additive_weight_fractions=additive_weight_fractions,
        library_records=recipe_context.library_records,
    )


def test_salt_and_additive_perturbations_change_upstream_local_field_primitives() -> None:
    base_recipe_context = generator_construction.build_recipe_library_context(
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_lifsi_fec.yaml",
        PHYSICAL_LIBRARY_ROOT,
    )
    numerical_options = generator_construction.NumericalOptions(
        reference_box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
        volume_m3=1.0e-24,
        state_quadrature_order=2,
        transition_grid_count=7,
    )

    base_result = _compute_conductivity_from_recipe_context(
        base_recipe_context,
        numerical_options,
    )
    salt_perturbed_result = _compute_conductivity_from_recipe_context(
        _recipe_context_with_perturbed_loadings(base_recipe_context, 1.2, 0.05),
        numerical_options,
    )
    additive_perturbed_result = _compute_conductivity_from_recipe_context(
        _recipe_context_with_perturbed_loadings(base_recipe_context, 1.0, 0.10),
        numerical_options,
    )

    salt_perturbed_primitive_names = (
        "state_concentrations_mol_m3",
        "self_current_tensors_D_self_i_m2_s",
        "symmetric_capacity_fluxes_K_ij_mol_m3_s",
        "mori_memory_matrix_A",
        "mori_current_coupling_matrix_h",
    )
    additive_perturbed_primitive_names = (
        "self_current_tensors_D_self_i_m2_s",
        "symmetric_capacity_fluxes_K_ij_mol_m3_s",
        "mori_memory_matrix_A",
        "mori_current_coupling_matrix_h",
    )
    for primitive_name in salt_perturbed_primitive_names:
        base_primitive = np.asarray(getattr(base_result, primitive_name), dtype=float)
        salt_primitive = np.asarray(
            getattr(salt_perturbed_result, primitive_name),
            dtype=float,
        )
        if salt_primitive.shape == base_primitive.shape and base_primitive.size > 0:
            assert np.linalg.norm(salt_primitive - base_primitive) > 0.0

    for primitive_name in additive_perturbed_primitive_names:
        base_primitive = np.asarray(getattr(base_result, primitive_name), dtype=float)
        additive_primitive = np.asarray(
            getattr(additive_perturbed_result, primitive_name),
            dtype=float,
        )
        if additive_primitive.shape == base_primitive.shape and base_primitive.size > 0:
            assert np.linalg.norm(additive_primitive - base_primitive) > 0.0


def test_missing_additive_record_fails_loudly() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    missing_species_name = "missing-neutral-additive"

    with pytest.raises(
        KeyError,
        match=f"recipe species {missing_species_name} is absent",
    ):
        _require_species(missing_species_name, records)


def test_trajectory_state_key_alignment_uses_active_sparse_basis() -> None:
    recipe_context = generator_construction.build_recipe_library_context(
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_lifsi_fec.yaml",
        PHYSICAL_LIBRARY_ROOT,
    )
    records = recipe_context.library_records
    mixture = generator_construction.compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    center_catalog = extract_projected_primitives.ChargedCenterCatalog(
        molecule_ids=np.asarray([0, 1], dtype=int),
        species_labels=("Li+", "PF6-"),
        roles=("cation", "anion"),
        formal_charges_e=np.asarray([1.0, -1.0], dtype=float),
    )
    pair_distance_A = 8.0
    center_frame = extract_projected_primitives.ChargedCenterFrame(
        positions_A=np.asarray(
            [[0.0, 0.0, 0.0], [pair_distance_A, 0.0, 0.0]],
            dtype=float,
        ),
        wrapped_positions_A=np.asarray(
            [[0.0, 0.0, 0.0], [pair_distance_A, 0.0, 0.0]],
            dtype=float,
        ),
        box_bounds_A=np.asarray(
            [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]],
            dtype=float,
        ),
    )
    environment_catalog = extract_projected_primitives.MolecularEnvironmentCatalog(
        molecule_ids=np.asarray([0, 1], dtype=int),
        species_labels=("Li+", "PF6-"),
    )
    environment_frame = extract_projected_primitives.MolecularEnvironmentFrame(
        positions_A=center_frame.positions_A,
        wrapped_positions_A=center_frame.wrapped_positions_A,
        orientation_vectors=np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float
        ),
        box_bounds_A=center_frame.box_bounds_A,
    )
    temporal_coordinates = {
        generator_construction.ReducedCoordinate.CAGE_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.IDENTITY_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value: 0.0,
    }
    thresholds = extract_projected_primitives.AssociationThresholds(
        contact_pair_max_distance_A=4.0,
        solvent_separated_pair_max_distance_A=10.0,
    )
    trajectory_label = extract_projected_primitives._active_sparse_state_label_for_center(
        records=records,
        mixture=mixture,
        center_frame=center_frame,
        center_catalog=center_catalog,
        environment_frame=environment_frame,
        environment_catalog=environment_catalog,
        center_index=0,
        counterion_index=1,
        distances_A=np.asarray([pair_distance_A], dtype=float),
        thresholds=thresholds,
        temporal_coordinates=temporal_coordinates,
    )
    configuration = (
        extract_projected_primitives._two_center_site_configuration_from_frame(
            center_frame,
            center_catalog,
            0,
            1,
        )
    )
    coordinate_values = (
        extract_projected_primitives._reduced_coordinate_values_from_center_observation(
            records,
            mixture,
            configuration,
            configuration,
            environment_frame,
            environment_catalog,
            1,
            PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
            np.asarray([pair_distance_A], dtype=float),
            thresholds,
            temporal_coordinates,
        )
    )
    expected_label = "|".join(
        generator_construction.sparse_state_key_from_reduced_observation(
            records=records,
            configuration=configuration,
            mixture=mixture,
            pair_label=PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
            active_anion_component_name="PF6-",
            coordinate_values=coordinate_values,
        )
    )

    assert trajectory_label == expected_label
    assert "solvent_separated_pair_center" not in trajectory_label
    assert len(trajectory_label.split("|")) == generator_construction.STATE_KEY_LENGTH


def test_trajectory_transition_moments_are_reciprocal_edge_oriented() -> None:
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("state_a", "state_b"),
        occupancy_state_index_by_observation=np.asarray([0, 1, 0, 1], dtype=int),
        from_state_index_by_step=np.asarray([0, 1], dtype=int),
        to_state_index_by_step=np.asarray([1, 0], dtype=int),
        charge_displacement_by_step_m=np.asarray(
            [[2.0e-10, 0.0, 0.0], [-4.0e-10, 0.0, 0.0]],
            dtype=float,
        ),
        self_charge_polarization_by_frame_and_center_m=np.zeros((5, 2, 3)),
        state_index_by_frame_and_center=np.asarray(((0, 1),) * 5, dtype=int),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )

    primitive_set = project_sampled_trajectory_to_generator_primitives(sample_input)

    assert len(primitive_set.conditional_displacement_moments) == 1
    moment = primitive_set.conditional_displacement_moments[0]
    assert moment.from_state_label == "state_a"
    assert moment.to_state_label == "state_b"
    assert moment.sample_count == 2
    assert np.asarray(moment.mean_charge_displacement_m, dtype=float) == pytest.approx(
        np.asarray([3.0e-10, 0.0, 0.0], dtype=float)
    )
    component_residual = primitive_set.diagnostics.component_drift_residuals[0]
    assert component_residual.weighted_drift_norm_mol_m2_s == pytest.approx(0.0)
    assert (
        primitive_set.diagnostics.finite_process_legality.maximum_detailed_balance_residual_mol_m3_s
        == pytest.approx(0.0)
    )
    assert component_residual.top_edge_contributions
    primitive_arrays = extract_projected_primitives._primitive_arrays_from_projected_set(
        primitive_set
    )
    capacity_fluxes = primitive_arrays["symmetric_capacity_fluxes_K_ij_mol_m3_s"]
    first_moments = primitive_arrays["transition_first_moments_d_ij_m"]
    second_moments = primitive_arrays["transition_second_moments_M_ij_m2"]
    assert capacity_fluxes == pytest.approx(capacity_fluxes.T)
    assert first_moments + np.swapaxes(first_moments, 0, 1) == pytest.approx(0.0)
    assert second_moments == pytest.approx(np.swapaxes(second_moments, 0, 1))


def test_component_drift_residual_diagnoses_illegal_finite_process() -> None:
    first_moments = np.asarray(
        [
            [[0.0, 0.0, 0.0], [2.0e-10, 0.0, 0.0]],
            [[1.0e-10, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )

    component_residuals = compute_finite_process_component_drift_residuals(
        state_labels=("state_a", "state_b"),
        state_concentrations_mol_m3=np.asarray([500.0, 500.0], dtype=float),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
            [[0.0, 2.0e12], [2.0e12, 0.0]],
            dtype=float,
        ),
        transition_first_moments_d_ij_m=first_moments,
        directed_transition_sample_counts=np.asarray(
            [[0, 4], [0, 0]],
            dtype=int,
        ),
    )

    assert len(component_residuals) == 1
    component_residual = component_residuals[0]
    assert component_residual.weighted_drift_norm_mol_m2_s > 0.0
    assert component_residual.top_edge_contributions[0].component_id == 0
    assert component_residual.top_edge_contributions[0].from_state_label == "state_a"
    assert component_residual.top_edge_contributions[0].forward_sample_count == 4
    assert component_residual.top_edge_contributions[0].reverse_sample_count == 0
    assert component_residual.top_edge_contributions[0].missing_reverse_event_candidate


def test_finite_process_legality_reports_detailed_balance_and_top_edges() -> None:
    diagnostic = diagnose_finite_process_legality(
        state_labels=("state_a", "state_b"),
        state_concentrations_mol_m3=np.asarray([300.0, 700.0], dtype=float),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
            [[0.0, 2.0e12], [2.0e12, 0.0]],
            dtype=float,
        ),
        transition_first_moments_d_ij_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [4.0e-10, 0.0, 0.0]],
                [[-4.0e-10, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        transition_second_moments_M_ij_m2=np.asarray(
            [
                [np.zeros((3, 3), dtype=float), np.eye(3) * 1.0e-20],
                [np.eye(3) * 1.0e-20, np.zeros((3, 3), dtype=float)],
            ],
            dtype=float,
        ),
        directed_transition_sample_counts=np.asarray([[0, 5], [0, 0]], dtype=int),
    )

    assert diagnostic.maximum_detailed_balance_residual_mol_m3_s == pytest.approx(0.0)
    component_residual = diagnostic.component_drift_residuals[0]
    assert component_residual.weighted_drift_norm_mol_m2_s == pytest.approx(0.0)
    assert component_residual.top_edge_contributions[0].from_state_label == "state_a"
    assert component_residual.top_edge_contributions[0].missing_reverse_event_candidate


def test_finite_process_legality_rejects_nonreciprocal_second_moments() -> None:
    with pytest.raises(ValueError, match="M_ji_m2 must equal M_ij"):
        diagnose_finite_process_legality(
            state_labels=("state_a", "state_b"),
            state_concentrations_mol_m3=np.asarray([500.0, 500.0], dtype=float),
            symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
                [[0.0, 2.0e12], [2.0e12, 0.0]],
                dtype=float,
            ),
            transition_first_moments_d_ij_m=np.asarray(
                [
                    [[0.0, 0.0, 0.0], [2.0e-10, 0.0, 0.0]],
                    [[-2.0e-10, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                dtype=float,
            ),
            transition_second_moments_M_ij_m2=np.asarray(
                [
                    [np.zeros((3, 3), dtype=float), np.eye(3) * 1.0e-20],
                    [np.eye(3) * 2.0e-20, np.zeros((3, 3), dtype=float)],
                ],
                dtype=float,
            ),
            directed_transition_sample_counts=np.asarray([[0, 1], [1, 0]], dtype=int),
        )


def test_component_drift_diagnostic_identifies_offending_directed_edge() -> None:
    state_labels = ("state_a", "state_b")
    concentrations = np.asarray([2.0, 2.0], dtype=float)
    capacity_fluxes = np.asarray([[0.0, 3.0], [3.0, 0.0]], dtype=float)
    first_moments = np.asarray(
        [
            [[0.0, 0.0, 0.0], [2.0e-10, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    directed_counts = np.asarray([[0, 5], [0, 0]], dtype=int)

    raw_residual = compute_finite_process_component_drift_residuals(
        state_labels,
        concentrations,
        capacity_fluxes,
        first_moments,
        directed_counts,
    )[0]
    assert raw_residual.weighted_drift_norm_mol_m2_s > 0.0
    assert raw_residual.top_edge_contributions[0].from_state_label == "state_a"
    assert raw_residual.top_edge_contributions[0].to_state_label == "state_b"
    assert raw_residual.top_edge_contributions[0].forward_sample_count == 5
    assert raw_residual.top_edge_contributions[0].reverse_sample_count == 0


def test_extractor_rejects_component_drift_with_offending_edges() -> None:
    residuals = compute_finite_process_component_drift_residuals(
        ("state_a", "state_b"),
        np.asarray([2.0, 2.0], dtype=float),
        np.asarray([[0.0, 3.0], [3.0, 0.0]], dtype=float),
        np.asarray(
            [
                [[0.0, 0.0, 0.0], [2.0e-10, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        np.asarray([[0, 5], [0, 0]], dtype=int),
    )
    assert extract_projected_primitives._component_drift_violation(residuals)
    failure_reason = extract_projected_primitives._invalid_component_drift_failure_reason(
        residuals
    )
    assert "invalid finite-state drift; offending edges:" in failure_reason
    assert "state_a->state_b" in failure_reason
    assert "forward=5, reverse=0" in failure_reason


def test_local_field_perturbation_changes_upstream_primitives() -> None:
    recipe_context = generator_construction.build_recipe_library_context(
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_lifsi_fec.yaml",
        PHYSICAL_LIBRARY_ROOT,
    )
    records = recipe_context.library_records
    mixture = generator_construction.compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    numerical_options = generator_construction.NumericalOptions(
        reference_box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
        volume_m3=1.0e-24,
        state_quadrature_order=2,
        transition_grid_count=7,
    )
    template_configuration = generator_construction.build_template_site_configuration(
        records,
        recipe_context,
        mixture,
        numerical_options,
    )
    coordinate_values = _local_field_test_coordinate_values()
    crowded_coordinate_values = {
        **coordinate_values,
        generator_construction.ReducedCoordinate.LOCAL_PACKING_FRACTION.value: 0.4,
        generator_construction.ReducedCoordinate.LOCAL_IONIC_STRENGTH.value: 2000.0,
    }
    configuration = generator_construction._configuration_with_reduced_coordinate_values(
        records,
        template_configuration,
        coordinate_values,
    )
    dilute_fields = generator_construction._local_fields_for_coordinate_values(
        records,
        configuration,
        coordinate_values,
    )
    crowded_fields = generator_construction._local_fields_for_coordinate_values(
        records,
        configuration,
        crowded_coordinate_values,
    )
    dilute_charge_mobility = _charge_mobility_trace(
        records,
        configuration,
        recipe_context.temperature_K,
        dilute_fields,
    )
    crowded_charge_mobility = _charge_mobility_trace(
        records,
        configuration,
        recipe_context.temperature_K,
        crowded_fields,
    )
    dilute_objects = _physical_objects_for_local_fields(
        records,
        configuration,
        recipe_context.temperature_K,
        dilute_fields,
    )
    crowded_objects = _physical_objects_for_local_fields(
        records,
        configuration,
        recipe_context.temperature_K,
        crowded_fields,
    )
    pair_distance_gradient = generator_construction._pair_distance_gradient_from_record(
        records,
        configuration,
        {
            "reaction_coordinate": "Li_anion_distance",
            "gradient_policy": "pair_distance_gradient",
        },
    )
    dilute_projected_diffusivity_m2_s = (
        generator_construction.project_diffusivity_onto_reaction_coordinate(
            dilute_objects.mobility_tensor_m2_s,
            pair_distance_gradient,
        )
    )
    crowded_projected_diffusivity_m2_s = (
        generator_construction.project_diffusivity_onto_reaction_coordinate(
            crowded_objects.mobility_tensor_m2_s,
            pair_distance_gradient,
        )
    )

    assert crowded_fields.ionic_strength_mol_m3 > dilute_fields.ionic_strength_mol_m3
    assert crowded_fields.dielectric_constant < dilute_fields.dielectric_constant
    assert crowded_fields.viscosity_Pa_s > dilute_fields.viscosity_Pa_s
    assert crowded_charge_mobility < dilute_charge_mobility
    assert crowded_objects.potential_energy_J_mol > dilute_objects.potential_energy_J_mol
    assert crowded_projected_diffusivity_m2_s < dilute_projected_diffusivity_m2_s


def test_anion_internal_charge_separation_amplifies_li_coordination_response() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    pf6_configuration = _lithium_pf6_fec_configuration(pair_distance_m=4.0e-10)
    configuration = SiteConfiguration(
        species_names=tuple(
            "FSI-" if species_name == "PF6-" else species_name
            for species_name in pf6_configuration.species_names
        ),
        molecule_ids=np.asarray(pf6_configuration.molecule_ids, dtype=int),
        site_ids=np.asarray(pf6_configuration.site_ids, dtype=int),
        positions_m=np.asarray(pf6_configuration.positions_m, dtype=float),
        unwrapped_positions_m=np.asarray(
            pf6_configuration.unwrapped_positions_m,
            dtype=float,
        ),
        box_lengths_m=np.asarray(pf6_configuration.box_lengths_m, dtype=float),
    )
    separation_factor = anion_internal_charge_separation_factor(records, "FSI-")

    assert 0.0 < separation_factor < 1.0
    assert li_anion_feature_coordination_energy_multiplier(
        records,
        configuration,
    ) == pytest.approx(1.0 + separation_factor)


def test_additive_microviscosity_comes_from_species_feature_record() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    configuration = _lithium_pf6_fec_configuration(pair_distance_m=4.0e-10)
    molecule_keys = tuple(
        dict.fromkeys(
            zip(
                configuration.species_names,
                np.asarray(configuration.molecule_ids, dtype=int),
                strict=True,
            )
        )
    )
    additive_molecule_count = sum(
        species_name == "FEC" for species_name, _molecule_id in molecule_keys
    )

    expected_fraction = (
        additive_molecule_count
        * float(records.species_records["FEC"]["local_microviscosity_coefficient"])
        / len(molecule_keys)
    )
    assert generator_construction._configuration_additive_fraction(
        records,
        configuration,
    ) == pytest.approx(expected_fraction)


def test_state_quadrature_stores_physical_local_field_laws() -> None:
    recipe_context = generator_construction.build_recipe_library_context(
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_lifsi_fec.yaml",
        PHYSICAL_LIBRARY_ROOT,
    )
    records = recipe_context.library_records
    mixture = generator_construction.compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    numerical_options = generator_construction.NumericalOptions(
        reference_box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
        volume_m3=1.0e-24,
        state_quadrature_order=2,
        transition_grid_count=7,
    )
    template_configuration = generator_construction.build_template_site_configuration(
        records,
        recipe_context,
        mixture,
        numerical_options,
    )
    coordinate_nodes = tuple(
        (
            coordinate,
            np.asarray([values[-1]], dtype=float)
            if coordinate == generator_construction.ReducedCoordinate.LOCAL_PACKING_FRACTION
            else np.asarray([values[len(values) // 2]], dtype=float),
            np.asarray([1.0], dtype=float),
        )
        for coordinate, values, _weights in generator_construction._state_coordinate_nodes(
            records,
            template_configuration,
            generator_construction._declared_reduced_coordinates(records),
            0.0,
            float(records.basis_record["pair_basins"]["r_free_m"]),
            recipe_context,
            mixture,
            numerical_options,
        )
    )
    grouped_states = generator_construction._group_state_quadrature_nodes(
        records,
        template_configuration,
        mixture,
        PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
        coordinate_nodes,
    )
    state_group = next(iter(grouped_states.values()))
    local_fields = state_group.local_fields[0]
    coordinate_values = {
        coordinate.value: float(values[0])
        for coordinate, values, _weights in coordinate_nodes
    }

    assert local_fields.local_packing_fraction >= coordinate_values[
        generator_construction.ReducedCoordinate.LOCAL_PACKING_FRACTION.value
    ]
    assert local_fields.ionic_strength_mol_m3 > coordinate_values[
        generator_construction.ReducedCoordinate.LOCAL_IONIC_STRENGTH.value
    ]
    assert local_fields.dielectric_constant < coordinate_values[
        generator_construction.ReducedCoordinate.LOCAL_DIELECTRIC.value
    ]
    assert local_fields.viscosity_Pa_s > coordinate_values[
        generator_construction.ReducedCoordinate.LOCAL_VISCOSITY.value
    ]


def test_zero_motif_transition_policy_requires_zero_moments() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    zero_record = records.transition_record["transition_records"][
        "SSIP_to_additive_separated_SSIP"
    ]

    generator_construction._validate_transition_displacement_policy(
        "SSIP_to_additive_separated_SSIP",
        zero_record,
        np.zeros(3, dtype=float),
        np.zeros((3, 3), dtype=float),
    )
    with pytest.raises(ValueError, match="zero-displacement"):
        generator_construction._validate_transition_displacement_policy(
            "SSIP_to_additive_separated_SSIP",
            zero_record,
            np.asarray([1.0e-10, 0.0, 0.0], dtype=float),
            np.zeros((3, 3), dtype=float),
        )


def test_conductivity_carrying_transition_policy_requires_nonzero_moments() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    carrier_record = records.transition_record["transition_records"]["partner_switch"]

    with pytest.raises(ValueError, match="produced zero d and M"):
        generator_construction._validate_transition_displacement_policy(
            "partner_switch",
            carrier_record,
            np.zeros(3, dtype=float),
            np.zeros((3, 3), dtype=float),
        )


def test_transition_geometry_records_partition_charge_and_zero_families() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    transition_records = records.transition_record["transition_records"]
    charge_carrying_families = (
        "partner_switch",
        "identity_diffusion",
        "structural_hop",
        "bridge_network_formation_breakup",
    )
    zero_motif_families = (
        "pair_to_aggregate",
        "cage_capture_release",
    )
    reference_zero_record = transition_records[zero_motif_families[0]]

    assert "cage_backjump" in records.memory_record["memory_records"]
    for family in charge_carrying_families:
        transition_record = transition_records[family]
        generator_construction._validate_transition_family_reaction_coordinate(
            family,
            transition_record,
        )
        assert "endpoint_geometry" in transition_record
        assert (
            generator_construction._endpoint_geometry_diagnostic_length_m(
                transition_record
            )
            > 0.0
        )

    for family in zero_motif_families:
        transition_record = transition_records[family]
        generator_construction._validate_transition_family_reaction_coordinate(
            family,
            transition_record,
        )
        assert transition_record["moment_policy"] == reference_zero_record["moment_policy"]
        assert (
            transition_record["displacement_policy"]
            == reference_zero_record["displacement_policy"]
        )
        assert "endpoint_geometry" not in transition_record
        first_moment, second_moment = generator_construction._zero_transition_moments(
            None
        )
        assert first_moment == pytest.approx(np.zeros(3, dtype=float))
        assert second_moment == pytest.approx(np.zeros((3, 3), dtype=float))


def test_charge_carrying_transition_endpoint_geometry_schema_is_required() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    partner_record = {
        **records.transition_record["transition_records"]["partner_switch"],
        "endpoint_geometry": {
            **records.transition_record["transition_records"]["partner_switch"][
                "endpoint_geometry"
            ],
            "start": {"Li_partner": "old_anion_partner"},
        },
    }

    with pytest.raises(
        KeyError,
        match="endpoint_geometry.start.Li_position",
    ):
        generator_construction._validate_transition_family_reaction_coordinate(
            "partner_switch",
            partner_record,
        )


def test_structural_hop_endpoint_geometry_requires_structural_displacement() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    structural_hop_record = {
        **records.transition_record["transition_records"]["structural_hop"],
        "endpoint_geometry": {
            **records.transition_record["transition_records"]["structural_hop"][
                "endpoint_geometry"
            ],
            "displacement": {
                **records.transition_record["transition_records"]["structural_hop"][
                    "endpoint_geometry"
                ]["displacement"],
                "type": "charge_identity",
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="structural hop requires endpoint displacement type",
    ):
        generator_construction._validate_transition_family_reaction_coordinate(
            "structural_hop",
            structural_hop_record,
        )


def test_transition_rate_bounds_fail_loudly() -> None:
    recipe_context = generator_construction.build_recipe_library_context(
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_lifsi_fec.yaml",
        PHYSICAL_LIBRARY_ROOT,
    )
    records = recipe_context.library_records
    mixture = generator_construction.compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    numerical_options = generator_construction.NumericalOptions(
        reference_box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
        volume_m3=1.0e-24,
        state_quadrature_order=2,
        transition_grid_count=7,
    )
    template_configuration = generator_construction.build_template_site_configuration(
        records,
        recipe_context,
        mixture,
        numerical_options,
    )
    state_quadratures = generator_construction.build_all_state_quadratures(
        records,
        template_configuration,
        mixture,
        recipe_context,
        numerical_options,
    )
    transition_edges = generator_construction.finite_generator_transition_edges(
        records,
        state_quadratures,
        recipe_context.temperature_K,
    )
    transition_quadratures = generator_construction.build_all_transition_quadratures(
        records,
        state_quadratures,
        template_configuration,
        mixture,
        recipe_context.temperature_K,
        numerical_options,
    )
    state_count = len(state_quadratures)
    first_edge = transition_edges[0]
    generator = (
        np.eye(state_count, dtype=float) * 0.0
        + np.eye(state_count, dtype=float)[first_edge.from_state_index][:, np.newaxis]
        * np.eye(state_count, dtype=float)[first_edge.to_state_index][np.newaxis, :]
        * 1.0e20
        + np.eye(state_count, dtype=float)[first_edge.to_state_index][:, np.newaxis]
        * np.eye(state_count, dtype=float)[first_edge.from_state_index][np.newaxis, :]
        - np.eye(state_count, dtype=float)[first_edge.from_state_index][:, np.newaxis]
        * np.eye(state_count, dtype=float)[first_edge.from_state_index][np.newaxis, :]
        * 1.0e20
        - np.eye(state_count, dtype=float)[first_edge.to_state_index][:, np.newaxis]
        * np.eye(state_count, dtype=float)[first_edge.to_state_index][np.newaxis, :]
    )

    with pytest.raises(ValueError, match="above .* upper bound"):
        generator_construction._validate_transition_rate_bounds(
            records,
            (first_edge,),
            (transition_quadratures[0],),
            generator,
            recipe_context.temperature_K,
        )


def test_atmosphere_resistance_cancels_colocated_neutral_translation() -> None:
    records = _records_with_synthetic_neutral_pair(load_physical_library(PHYSICAL_LIBRARY_ROOT))
    colocated_configuration = _synthetic_neutral_pair_configuration(0.0)
    translation_vector = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    atmosphere_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        colocated_configuration,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=1000.0,
        temperature_K=T_REF_K,
        viscosity_Pa_s=1.0e-3,
    )
    assert (
        float(
            translation_vector
            @ atmosphere_diagnostics.atmosphere_resistance_tensor_kg_s
            @ translation_vector
        )
        == pytest.approx(0.0)
    )


def test_atmosphere_resistance_separation_recovers_independent_ion_drag() -> None:
    records = _records_with_synthetic_neutral_pair(load_physical_library(PHYSICAL_LIBRARY_ROOT))
    translation_vector = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    colocated_configuration = _synthetic_neutral_pair_configuration(0.0)
    separated_configuration = _synthetic_neutral_pair_configuration(1.0e-7)
    colocated_resistance = compute_resistance_tensor_kg_s(
        records,
        colocated_configuration,
        viscosity_Pa_s=1.0e-3,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=1000.0,
        temperature_K=T_REF_K,
        local_packing_fraction=compute_local_packing_fraction(
            records,
            colocated_configuration,
        ),
    )
    separated_resistance = compute_resistance_tensor_kg_s(
        records,
        separated_configuration,
        viscosity_Pa_s=1.0e-3,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=1000.0,
        temperature_K=T_REF_K,
        local_packing_fraction=compute_local_packing_fraction(
            records,
            separated_configuration,
        ),
    )

    assert float(translation_vector @ separated_resistance @ translation_vector) > float(
        translation_vector @ colocated_resistance @ translation_vector
    )


def test_atmosphere_resistance_is_zero_at_zero_ionic_strength() -> None:
    records = _records_with_synthetic_neutral_pair(load_physical_library(PHYSICAL_LIBRARY_ROOT))
    configuration = _synthetic_neutral_pair_configuration(1.0e-9)
    zero_ionic_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        configuration,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=0.0,
        temperature_K=T_REF_K,
        viscosity_Pa_s=1.0e-3,
    )
    assert np.max(np.abs(zero_ionic_diagnostics.atmosphere_resistance_tensor_kg_s)) == (
        pytest.approx(0.0)
    )


def test_atmosphere_resistance_diagnostics_split_components() -> None:
    records = _records_with_synthetic_neutral_pair(load_physical_library(PHYSICAL_LIBRARY_ROOT))
    colocated_configuration = _synthetic_neutral_pair_configuration(0.0)
    separated_configuration = _synthetic_neutral_pair_configuration(1.0e-7)
    translation_vector = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    colocated_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        colocated_configuration,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=1000.0,
        temperature_K=T_REF_K,
        viscosity_Pa_s=1.0e-3,
    )
    separated_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        separated_configuration,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=1000.0,
        temperature_K=T_REF_K,
        viscosity_Pa_s=1.0e-3,
    )
    zero_ionic_diagnostics = compute_atmosphere_resistance_diagnostics(
        records,
        separated_configuration,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=0.0,
        temperature_K=T_REF_K,
        viscosity_Pa_s=1.0e-3,
    )

    assert colocated_diagnostics.atmosphere_resistance_tensor_kg_s == pytest.approx(
        colocated_diagnostics.electrophoretic_resistance_tensor_kg_s
        + colocated_diagnostics.relaxation_resistance_tensor_kg_s
    )
    assert float(
        translation_vector
        @ colocated_diagnostics.atmosphere_resistance_tensor_kg_s
        @ translation_vector
    ) == pytest.approx(0.0)
    assert colocated_diagnostics.cation_diagonal_resistance_trace_kg_s > 0.0
    assert colocated_diagnostics.anion_diagonal_resistance_trace_kg_s > 0.0
    assert colocated_diagnostics.cation_anion_cross_resistance_trace_kg_s < 0.0
    assert (
        separated_diagnostics.minimum_separation_over_debye_length
        > colocated_diagnostics.minimum_separation_over_debye_length
    )
    assert (
        abs(separated_diagnostics.cation_anion_cross_resistance_trace_kg_s)
        < abs(colocated_diagnostics.cation_anion_cross_resistance_trace_kg_s)
    )
    assert separated_diagnostics.debye_falkenhagen_time_s > 0.0
    assert np.max(np.abs(zero_ionic_diagnostics.atmosphere_resistance_tensor_kg_s)) == (
        pytest.approx(0.0)
    )


def test_atmosphere_resistance_diagnostics_cloud_and_ionic_strength_response() -> None:
    base_records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    localized_cloud_records = _records_with_synthetic_neutral_pair_cloud_radius(
        base_records,
        1.0e-10,
    )
    diffuse_cloud_records = _records_with_synthetic_neutral_pair_cloud_radius(
        base_records,
        8.0e-10,
    )
    configuration = _synthetic_neutral_pair_configuration(8.0e-10)
    low_ionic_diagnostics = compute_atmosphere_resistance_diagnostics(
        localized_cloud_records,
        configuration,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=100.0,
        temperature_K=T_REF_K,
        viscosity_Pa_s=1.0e-3,
    )
    high_ionic_diagnostics = compute_atmosphere_resistance_diagnostics(
        localized_cloud_records,
        configuration,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=2000.0,
        temperature_K=T_REF_K,
        viscosity_Pa_s=1.0e-3,
    )
    diffuse_cloud_diagnostics = compute_atmosphere_resistance_diagnostics(
        diffuse_cloud_records,
        configuration,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=2000.0,
        temperature_K=T_REF_K,
        viscosity_Pa_s=1.0e-3,
    )

    assert (
        high_ionic_diagnostics.cation_diagonal_resistance_trace_kg_s
        > low_ionic_diagnostics.cation_diagonal_resistance_trace_kg_s
    )
    assert (
        high_ionic_diagnostics.mean_charge_cloud_form_factor
        < low_ionic_diagnostics.mean_charge_cloud_form_factor
    )
    assert (
        diffuse_cloud_diagnostics.mean_charge_cloud_form_factor
        < high_ionic_diagnostics.mean_charge_cloud_form_factor
    )
    assert (
        diffuse_cloud_diagnostics.cation_diagonal_resistance_trace_kg_s
        < high_ionic_diagnostics.cation_diagonal_resistance_trace_kg_s
    )


def test_resistance_component_diagnostics_expose_finite_thickness_shape_drag() -> None:
    recipe_context = generator_construction.build_recipe_library_context(
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_1m.yaml",
        PHYSICAL_LIBRARY_ROOT,
    )
    records = recipe_context.library_records
    numerical_options = generator_construction.NumericalOptions(
        reference_box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
        volume_m3=1.0,
        state_quadrature_order=2,
        transition_grid_count=5,
    )
    mixture = generator_construction.compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    template_configuration = generator_construction.build_template_site_configuration(
        records,
        recipe_context,
        mixture,
        numerical_options,
    )
    state_quadrature = generator_construction.build_all_state_quadratures(
        records,
        template_configuration,
        mixture,
        recipe_context,
        numerical_options,
    )[0]
    local_fields = state_quadrature.local_fields[0]

    diagnostics = compute_resistance_component_diagnostics(
        records,
        state_quadrature.configurations[0],
        local_fields.viscosity_Pa_s,
        local_fields.dielectric_constant,
        local_fields.ionic_strength_mol_m3,
        recipe_context.temperature_K,
        local_fields.local_packing_fraction,
    )

    assert 0.0 < diagnostics.free_volume_trace_kg_s < diagnostics.stokes_trace_kg_s
    assert diagnostics.charge_cloud_trace_kg_s > diagnostics.stokes_trace_kg_s
    assert diagnostics.total_trace_kg_s == pytest.approx(
        diagnostics.stokes_trace_kg_s
        + diagnostics.free_volume_trace_kg_s
        + diagnostics.charge_cloud_trace_kg_s
        + diagnostics.atmosphere_trace_kg_s
    )


def test_charged_center_covariance_controls_neutral_pair_charge_mobility() -> None:
    charge_numbers = (1.0, -1.0)
    uncorrelated_center_mobility = np.asarray(
        [[2.0e-10, 0.0], [0.0, 3.0e-10]],
        dtype=float,
    )
    positive_comotion_center_mobility = np.asarray(
        [[2.0e-10, 1.0e-10], [1.0e-10, 3.0e-10]],
        dtype=float,
    )
    anticorrelated_center_mobility = np.asarray(
        [[2.0e-10, -1.0e-10], [-1.0e-10, 3.0e-10]],
        dtype=float,
    )
    perfect_comotion_center_mobility = np.asarray(
        [[2.0e-10, 2.0e-10], [2.0e-10, 2.0e-10]],
        dtype=float,
    )

    uncorrelated_charge_mobility = (
        generator_construction.charge_covariance_mobility_from_center_matrix(
            charge_numbers,
            uncorrelated_center_mobility,
        )
    )
    positive_comotion_charge_mobility = (
        generator_construction.charge_covariance_mobility_from_center_matrix(
            charge_numbers,
            positive_comotion_center_mobility,
        )
    )
    anticorrelated_charge_mobility = (
        generator_construction.charge_covariance_mobility_from_center_matrix(
            charge_numbers,
            anticorrelated_center_mobility,
        )
    )
    perfect_comotion_charge_mobility = (
        generator_construction.charge_covariance_mobility_from_center_matrix(
            charge_numbers,
            perfect_comotion_center_mobility,
        )
    )
    positive_comotion_entries = (
        generator_construction._charged_center_pair_covariance_entries(
            "SSIP_multi_center_state",
            ("Li_center", "anion_center"),
            charge_numbers,
            positive_comotion_center_mobility,
        )
    )
    anticorrelated_entries = (
        generator_construction._charged_center_pair_covariance_entries(
            "SSIP_multi_center_state",
            ("Li_center", "anion_center"),
            charge_numbers,
            anticorrelated_center_mobility,
        )
    )

    assert uncorrelated_charge_mobility == pytest.approx(5.0e-10)
    assert positive_comotion_charge_mobility < uncorrelated_charge_mobility
    assert anticorrelated_charge_mobility > uncorrelated_charge_mobility
    assert perfect_comotion_charge_mobility == pytest.approx(0.0)
    assert positive_comotion_entries[0].state_label == "SSIP_multi_center_state"
    assert positive_comotion_entries[0].charge_weighted_covariance_m2_s < 0.0
    assert anticorrelated_entries[0].charge_weighted_covariance_m2_s > 0.0


def _compute_from_primitive_input(primitive_input: ProjectedPrimitiveInput):
    return compute_projected_analytical_conductivity_from_primitives(
        primitive_input.state_concentrations_mol_m3,
        primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s,
        primitive_input.transition_first_moments_d_ij_m,
        primitive_input.transition_second_moments_M_ij_m2,
        primitive_input.self_current_tensors_D_self_i_m2_s,
        primitive_input.mori_memory_matrix_A,
        primitive_input.mori_current_coupling_matrix_h,
        primitive_input.temperature_K,
        primitive_input.volume_m3,
    )


def _complete_artifact_result(
    conductivity_result: ProjectedConductivityResult,
) -> ProjectedConductivityResult:
    effect_attribution = {
        **dict(conductivity_result.effect_attribution),
        "basis_refinement_convergence_status": "converged",
        "basis_refinement_not_complete_reasons": (),
        "basis_refinement_hard_convergence_failure": False,
        "primitive_prediction_readiness_status": "complete",
        "primitive_prediction_scalar_label": "primitive_prediction",
        "primitive_prediction_not_complete_reasons": (),
    }
    return ProjectedConductivityResult(
        sigma_S_m=conductivity_result.sigma_S_m,
        sigma_mS_cm=conductivity_result.sigma_mS_cm,
        projected_diffusivity_tensor=conductivity_result.projected_diffusivity_tensor,
        direct_diffusivity_tensor=conductivity_result.direct_diffusivity_tensor,
        finite_state_memory_correction_tensor=(
            conductivity_result.finite_state_memory_correction_tensor
        ),
        continuous_mori_correction_tensor=(
            conductivity_result.continuous_mori_correction_tensor
        ),
        state_concentrations_mol_m3=(
            conductivity_result.state_concentrations_mol_m3
        ),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=(
            conductivity_result.symmetric_capacity_fluxes_K_ij_mol_m3_s
        ),
        reversible_generator_Q_ij_s_inv=(
            conductivity_result.reversible_generator_Q_ij_s_inv
        ),
        transition_first_moments_d_ij_m=(
            conductivity_result.transition_first_moments_d_ij_m
        ),
        transition_second_moments_M_ij_m2=(
            conductivity_result.transition_second_moments_M_ij_m2
        ),
        self_current_tensors_D_self_i_m2_s=(
            conductivity_result.self_current_tensors_D_self_i_m2_s
        ),
        mori_memory_matrix_A=conductivity_result.mori_memory_matrix_A,
        mori_current_coupling_matrix_h=(
            conductivity_result.mori_current_coupling_matrix_h
        ),
        state_transport_ownership_quadratures=(
            conductivity_result.state_transport_ownership_quadratures
        ),
        effect_attribution=effect_attribution,
    )


def _local_fields(
    records,
    configuration: SiteConfiguration,
    dielectric_constant: float,
    viscosity_Pa_s: float,
    ionic_strength_mol_m3: float,
) -> PhysicalLocalFields:
    return PhysicalLocalFields(
        dielectric_constant=dielectric_constant,
        viscosity_Pa_s=viscosity_Pa_s,
        ionic_strength_mol_m3=ionic_strength_mol_m3,
        local_packing_fraction=compute_local_packing_fraction(records, configuration),
    )


def _zero_potential_J_mol(coordinate: np.ndarray) -> float:
    return 0.0


def _unit_mobility_tensor_m2_s(coordinate: np.ndarray) -> np.ndarray:
    return np.eye(1, dtype=float)


def _single_axis_charge_gradient(coordinate: np.ndarray) -> np.ndarray:
    return np.asarray([[1.0], [0.0], [0.0]], dtype=float)


def _empty_memory_gradient(coordinate: np.ndarray) -> np.ndarray:
    return np.zeros((0, 1), dtype=float)


def _two_lithium_configuration() -> SiteConfiguration:
    return SiteConfiguration(
        species_names=("Li+", "Li+"),
        molecule_ids=np.asarray([0, 1], dtype=int),
        site_ids=np.asarray([0, 0], dtype=int),
        positions_m=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [5.0e-10, 0.0, 0.0],
            ],
            dtype=float,
        ),
        unwrapped_positions_m=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [5.0e-10, 0.0, 0.0],
            ],
            dtype=float,
        ),
        box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
    )


def _single_fec_configuration() -> SiteConfiguration:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    fec_record = records.species_records["FEC"]
    conformer_positions_m = np.asarray(
        fec_record["reference_conformer_coordinates_m"],
        dtype=float,
    )
    return SiteConfiguration(
        species_names=tuple("FEC" for _site_record in fec_record["sites"]),
        molecule_ids=np.zeros(len(fec_record["sites"]), dtype=int),
        site_ids=np.asarray(
            [int(site_record["site_id"]) for site_record in fec_record["sites"]],
            dtype=int,
        ),
        positions_m=conformer_positions_m,
        unwrapped_positions_m=conformer_positions_m,
        box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
    )


def _lithium_pf6_configuration(pair_distance_m: float) -> SiteConfiguration:
    return SiteConfiguration(
        species_names=("Li+", "PF6-"),
        molecule_ids=np.asarray([0, 1], dtype=int),
        site_ids=np.asarray([0, 1], dtype=int),
        positions_m=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [pair_distance_m, 0.0, 0.0],
            ],
            dtype=float,
        ),
        unwrapped_positions_m=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [pair_distance_m, 0.0, 0.0],
            ],
            dtype=float,
        ),
        box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
    )


def _lithium_full_pf6_configuration(pair_distance_m: float) -> SiteConfiguration:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    pf6_record = records.species_records["PF6-"]
    pf6_conformer_positions_m = np.asarray(
        pf6_record["reference_conformer_coordinates_m"],
        dtype=float,
    )
    phosphorus_site_position_m = pf6_conformer_positions_m[1]
    pf6_positions_m = (
        pf6_conformer_positions_m
        - phosphorus_site_position_m
        + np.asarray([pair_distance_m, 0.0, 0.0], dtype=float)
    )
    positions_m = np.vstack(
        (
            np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            pf6_positions_m,
        )
    )
    return SiteConfiguration(
        species_names=("Li+",) + tuple("PF6-" for _site in pf6_record["sites"]),
        molecule_ids=np.asarray(
            [0] + [1 for _site in pf6_record["sites"]],
            dtype=int,
        ),
        site_ids=np.asarray(
            [0] + [int(site_record["site_id"]) for site_record in pf6_record["sites"]],
            dtype=int,
        ),
        positions_m=positions_m,
        unwrapped_positions_m=positions_m,
        box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
    )


def _lithium_pf6_fec_configuration(pair_distance_m: float) -> SiteConfiguration:
    return SiteConfiguration(
        species_names=("Li+", "PF6-", "FEC"),
        molecule_ids=np.asarray([0, 1, 2], dtype=int),
        site_ids=np.asarray([0, 0, 0], dtype=int),
        positions_m=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [pair_distance_m, 0.0, 0.0],
                [0.0, 4.0e-10, 0.0],
            ],
            dtype=float,
        ),
        unwrapped_positions_m=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [pair_distance_m, 0.0, 0.0],
                [0.0, 4.0e-10, 0.0],
            ],
            dtype=float,
        ),
        box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
    )


def _synthetic_neutral_pair_configuration(pair_distance_m: float) -> SiteConfiguration:
    return SiteConfiguration(
        species_names=("X+", "X-"),
        molecule_ids=np.asarray([0, 1], dtype=int),
        site_ids=np.asarray([0, 0], dtype=int),
        positions_m=np.asarray(
            [[0.0, 0.0, 0.0], [pair_distance_m, 0.0, 0.0]],
            dtype=float,
        ),
        unwrapped_positions_m=np.asarray(
            [[0.0, 0.0, 0.0], [pair_distance_m, 0.0, 0.0]],
            dtype=float,
        ),
        box_lengths_m=np.asarray([1.0e-6, 1.0e-6, 1.0e-6], dtype=float),
    )


def _records_with_synthetic_neutral_pair(records):
    return _records_with_synthetic_neutral_pair_cloud_radius(records, 1.0e-10)


def _records_with_synthetic_neutral_pair_cloud_radius(
    records,
    charge_cloud_radius_m: float,
):
    return records.__class__(
        root=records.root,
        manifest=records.manifest,
        species_records={
            **records.species_records,
            "X+": _single_site_charged_species_record(
                "cation",
                1.0,
                charge_cloud_radius_m,
            ),
            "X-": _single_site_charged_species_record(
                "anion",
                -1.0,
                charge_cloud_radius_m,
            ),
        },
        pair_records=records.pair_records,
        mixture_record=records.mixture_record,
        basis_record=records.basis_record,
        transition_record=records.transition_record,
        memory_record=records.memory_record,
    )


def _single_site_charged_species_record(
    role: str,
    charge_number: float,
    charge_cloud_radius_m: float,
) -> dict:
    return {
        "role": role,
        "formal_charge_e": charge_number,
        "molecular_weight_kg_mol": 0.01,
        "density_kg_m3": 1000.0,
        "partial_molar_volume_m3_mol": 1.0e-5,
        "dielectric_constant": 1.0,
        "viscosity_Pa_s": 1.0e-3,
        "sites": (
            {
                "site_id": 0,
                "element": "X",
                "mass_kg": 1.0e-26,
                "steric_radius_m": 1.0e-10,
                "hydrodynamic_radius_m": 1.0e-10,
                "volume_m3": 4.0e-30,
                "lj_sigma_m": 2.0e-10,
                "lj_epsilon_J": 1.0e-21,
                "charge_number": charge_number,
                "charge_cloud_radius_m": charge_cloud_radius_m,
                "born_radius_m": 1.0e-10,
                "polarizability_SI": 0.0,
                "donor_flag": 0,
                "acceptor_flag": 0,
            },
        ),
        "bonds": (),
        "angles": (),
        "torsions": (),
        "constraints": (),
    }


def _pair_distance_gradient(configuration: SiteConfiguration) -> np.ndarray:
    vector_m = configuration.positions_m[1] - configuration.positions_m[0]
    distance_m = np.linalg.norm(vector_m)
    unit_vector = vector_m / distance_m
    return np.concatenate((-unit_vector, unit_vector))


def test_unequal_radius_regularized_rpy_is_continuous_at_regime_boundaries() -> None:
    viscosity_Pa_s = 1.0e-3
    first_radius_m = 1.2e-10
    second_radius_m = 2.0e-10
    relative_boundary_offset = 1.0e-9
    containment_boundary_m = abs(first_radius_m - second_radius_m)
    contact_boundary_m = first_radius_m + second_radius_m

    for boundary_m in (containment_boundary_m, contact_boundary_m):
        below_block = _rpy_cross_mobility_block_kg_inv_s(
            np.asarray(
                [boundary_m * (1.0 - relative_boundary_offset), 0.0, 0.0],
                dtype=float,
            ),
            first_radius_m,
            second_radius_m,
            viscosity_Pa_s,
        )
        above_block = _rpy_cross_mobility_block_kg_inv_s(
            np.asarray(
                [boundary_m * (1.0 + relative_boundary_offset), 0.0, 0.0],
                dtype=float,
            ),
            first_radius_m,
            second_radius_m,
            viscosity_Pa_s,
        )
        np.testing.assert_allclose(
            below_block,
            above_block,
            rtol=1.0e-8,
            atol=0.0,
        )
        np.testing.assert_allclose(below_block, below_block.T, rtol=0.0, atol=0.0)


def test_rigid_kinematic_map_preserves_all_intramolecular_distances() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    configuration = _lithium_full_pf6_configuration(pair_distance_m=4.5e-10)
    rigid_kinematic_map = _rigid_body_kinematic_map(records, configuration)
    generalized_velocity = np.arange(
        rigid_kinematic_map.shape[1],
        dtype=float,
    )
    site_velocities = (rigid_kinematic_map @ generalized_velocity).reshape((-1, 3))
    pf6_site_indices = tuple(
        site_index
        for site_index, species_name in enumerate(configuration.species_names)
        if species_name == "PF6-"
    )
    unwrapped_positions_m = np.asarray(configuration.unwrapped_positions_m, dtype=float)
    for first_offset, first_site_index in enumerate(pf6_site_indices):
        for second_site_index in pf6_site_indices[first_offset + 1 :]:
            relative_position_m = (
                unwrapped_positions_m[first_site_index]
                - unwrapped_positions_m[second_site_index]
            )
            relative_velocity_m_s = (
                site_velocities[first_site_index] - site_velocities[second_site_index]
            )
            assert float(relative_position_m @ relative_velocity_m_s) == pytest.approx(
                0.0,
                abs=1.0e-24,
            )


def _nearest_species_center_distance_m(
    records,
    configuration: SiteConfiguration,
    species_name: str,
) -> float:
    cation_index = generator_construction._first_role_index(
        records,
        configuration,
        generator_construction.SpeciesRole.CATION,
    )
    species_indices = generator_construction._first_molecule_indices_for_species(
        configuration,
        species_name,
    )
    species_center = np.mean(
        np.asarray(configuration.positions_m, dtype=float)[
            np.asarray(species_indices, dtype=int)
        ],
        axis=0,
    )
    return float(
        np.linalg.norm(
            species_center
            - np.asarray(configuration.positions_m[cation_index], dtype=float)
        )
    )


def _local_field_test_coordinate_values() -> dict[str, float]:
    return {
        generator_construction.ReducedCoordinate.LI_ANION_DISTANCE.value: 6.0e-10,
        generator_construction.ReducedCoordinate.LI_SOLVENT_COORDINATION.value: 0.0,
        generator_construction.ReducedCoordinate.LI_LIGAND_COORDINATION.value: 1.0,
        generator_construction.ReducedCoordinate.LI_ANION_COORDINATION.value: 0.0,
        generator_construction.ReducedCoordinate.ANION_ORIENTATION.value: 0.0,
        generator_construction.ReducedCoordinate.LOCAL_PACKING_FRACTION.value: 0.1,
        generator_construction.ReducedCoordinate.LOCAL_IONIC_STRENGTH.value: 500.0,
        generator_construction.ReducedCoordinate.LOCAL_DIELECTRIC.value: 50.0,
        generator_construction.ReducedCoordinate.LOCAL_VISCOSITY.value: 1.0e-3,
        generator_construction.ReducedCoordinate.ATMOSPHERE_POLARIZATION.value: 0.0,
        generator_construction.ReducedCoordinate.CAGE_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.CLUSTER_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.IDENTITY_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value: 0.0,
    }


def _charge_mobility_trace(
    records,
    configuration: SiteConfiguration,
    temperature_K: float,
    local_fields: PhysicalLocalFields,
) -> float:
    physical_objects = _physical_objects_for_local_fields(
        records,
        configuration,
        temperature_K,
        local_fields,
    )
    return float(
        np.trace(
            physical_objects.charge_polarization_gradient
            @ physical_objects.mobility_tensor_m2_s
            @ physical_objects.charge_polarization_gradient.T
        )
    )


def _physical_objects_for_local_fields(
    records,
    configuration: SiteConfiguration,
    temperature_K: float,
    local_fields: PhysicalLocalFields,
):
    return build_physical_objects(
        records,
        configuration,
        temperature_K,
        local_fields.dielectric_constant,
        local_fields.viscosity_Pa_s,
        local_fields.ionic_strength_mol_m3,
        local_fields.local_packing_fraction,
    )


def _primitive_input(
    temperature_K: float,
    self_diffusion_m2_s: float,
) -> ProjectedPrimitiveInput:
    return ProjectedPrimitiveInput(
        state_concentrations_mol_m3=np.asarray([1000.0], dtype=float),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=np.zeros((1, 1), dtype=float),
        transition_first_moments_d_ij_m=np.zeros((1, 1, 3), dtype=float),
        transition_second_moments_M_ij_m2=np.zeros((1, 1, 3, 3), dtype=float),
        self_current_tensors_D_self_i_m2_s=np.asarray(
            [
                [
                    [self_diffusion_m2_s, 0.0, 0.0],
                    [0.0, self_diffusion_m2_s, 0.0],
                    [0.0, 0.0, self_diffusion_m2_s],
                ]
            ],
            dtype=float,
        ),
        mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        mori_current_coupling_matrix_h=np.zeros((0, 3), dtype=float),
        temperature_K=temperature_K,
        volume_m3=1.0,
    )


def _two_state_primitive_input(
    capacity_flux_mol_m3_s: float,
    first_moment_m: float,
    self_diffusion_m2_s: float,
) -> ProjectedPrimitiveInput:
    first_moment_m2 = first_moment_m * first_moment_m
    return ProjectedPrimitiveInput(
        state_concentrations_mol_m3=np.asarray([1000.0, 1000.0], dtype=float),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
            [
                [0.0, capacity_flux_mol_m3_s],
                [capacity_flux_mol_m3_s, 0.0],
            ],
            dtype=float,
        ),
        transition_first_moments_d_ij_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [first_moment_m, 0.0, 0.0]],
                [[-first_moment_m, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        transition_second_moments_M_ij_m2=np.asarray(
            [
                [
                    np.zeros((3, 3), dtype=float),
                    [
                        [first_moment_m2, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ],
                [
                    [
                        [first_moment_m2, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    np.zeros((3, 3), dtype=float),
                ],
            ],
            dtype=float,
        ),
        self_current_tensors_D_self_i_m2_s=np.asarray(
            [
                [
                    [self_diffusion_m2_s, 0.0, 0.0],
                    [0.0, self_diffusion_m2_s, 0.0],
                    [0.0, 0.0, self_diffusion_m2_s],
                ],
                [
                    [self_diffusion_m2_s, 0.0, 0.0],
                    [0.0, self_diffusion_m2_s, 0.0],
                    [0.0, 0.0, self_diffusion_m2_s],
                ],
            ],
            dtype=float,
        ),
        mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        mori_current_coupling_matrix_h=np.zeros((0, 3), dtype=float),
        temperature_K=T_REF_K,
        volume_m3=1.0,
    )
