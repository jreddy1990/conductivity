from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from constants import T_REF_K
from conductivity.physical_library import generator_construction
from conductivity.physical_library import extract_projected_primitives
from conductivity.physical_library.committor_bvp import (
    OneDimensionalCommittorInput,
    solve_one_dimensional_committor,
)
from conductivity.physical_library.moment_bvp import (
    MomentBoundaryValueInput,
    build_path_moment_arrays,
    solve_endpoint_moment_bvp,
)
from conductivity.physical_library.basin_builder import build_state_definition
from conductivity.physical_library.memory_coordinates import (
    build_default_memory_coordinates,
    combine_memory_gradients,
    combine_memory_values,
)
from conductivity.physical_library.mixture_closures import (
    MixtureComposition,
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
    compute_charge_polarization_m,
    compute_local_packing_fraction,
    compute_resistance_tensor_kg_s,
)
from conductivity.physical_library.primitive_closure_fit import (
    interpolate_primitive_closure,
    load_closure_fit,
)
from conductivity.physical_library.projected_primitives_io import (
    read_projected_primitive_yaml,
    write_projected_primitive_yaml,
)
from conductivity.physical_library.reduced_generator import (
    ReducedGeneratorSpecification,
    ReducedStateQuadrature,
    build_projected_generator_input,
)
from conductivity.physical_library.library_io import load_physical_library
from conductivity.physical_library.transition_surface_builder import (
    OneDimensionalTransitionBuildInput,
    build_one_dimensional_transition_surface,
)
from conductivity.physical_library.trajectory_primitives import (
    TrajectoryMarkovAdditiveSampleInput,
    project_sampled_trajectory_to_generator_primitives,
)
from conductivity.physical_library.projected_analytical_conductivity import (
    CARTESIAN,
    ProjectedPrimitiveInput,
    compute_symmetric_capacity_fluxes,
    compute_projected_analytical_conductivity_from_primitives,
)

PHYSICAL_LIBRARY_ROOT = Path(__file__).resolve().parents[1] / "physical_library"


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
    )

    assert capacity_fluxes[0, 1] == pytest.approx(2000.0)


def test_zero_capacity_is_not_silent() -> None:
    with pytest.raises(ValueError, match="all generated transition capacity fluxes"):
        compute_symmetric_capacity_fluxes(
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
        )


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
    primitive_input = _primitive_input(temperature_K=300.0, self_diffusion_m2_s=1.0e-10)
    conductivity_result = _compute_from_primitive_input(primitive_input)
    primitive_path = tmp_path / "primitive.yaml"

    write_projected_primitive_yaml(
        primitive_path,
        ("free",),
        primitive_input,
        conductivity_result,
    )
    artifact = read_projected_primitive_yaml(primitive_path)

    assert artifact.state_labels == ("free",)
    assert artifact.primitive_input.temperature_K == pytest.approx(300.0)
    assert artifact.primitive_input.self_current_tensors_D_self_i_m2_s[0, 0, 0] == (
        pytest.approx(1.0e-10)
    )


def test_primitive_closure_interpolates_anchor_tensors(tmp_path) -> None:
    first_input = _primitive_input(temperature_K=300.0, self_diffusion_m2_s=1.0e-10)
    second_input = _primitive_input(temperature_K=300.0, self_diffusion_m2_s=3.0e-10)
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    for primitive_path, primitive_input in (
        (first_path, first_input),
        (second_path, second_input),
    ):
        write_projected_primitive_yaml(
            primitive_path,
            ("free",),
            primitive_input,
            _compute_from_primitive_input(primitive_input),
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
                ),
            ),
            transition_quadratures=(),
            memory_coordinate_gradient_functions=(),
            total_component_concentrations_mol_m3=np.asarray([1.0], dtype=float),
            temperature_K=T_REF_K,
            volume_m3=1.0,
        )
    )
    point = flatten_configuration_with_local_fields(
        configuration,
        _local_fields(records, configuration, 20.0, 1.0e-3, 1000.0),
    )

    assert np.isfinite(generator_specification.potential_energy_J_mol(point))
    assert generator_specification.mobility_tensor_m2_s(point).shape == (
        coordinate_count,
        coordinate_count,
    )
    assert generator_specification.charge_polarization_gradient(point).shape == (
        3,
        coordinate_count,
    )
    assert generator_specification.memory_coordinate_gradient(point).shape == (
        0,
        coordinate_count,
    )


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


def test_mixed_salt_recipe_generates_nonzero_transition_capacities() -> None:
    result = generator_construction.compute_conductivity_from_recipe(
        PHYSICAL_LIBRARY_ROOT / "recipe_ec_dmc_lipf6_lifsi_fec.yaml",
        PHYSICAL_LIBRARY_ROOT,
        generator_construction.NumericalOptions(
            reference_box_lengths_m=np.asarray([1.0e-8, 1.0e-8, 1.0e-8], dtype=float),
            volume_m3=1.0e-24,
            state_quadrature_order=2,
            transition_grid_count=7,
        ),
    )

    assert result.state_concentrations_mol_m3.size > 24
    assert np.count_nonzero(result.symmetric_capacity_fluxes_K_ij_mol_m3_s) > 0
    assert np.count_nonzero(result.reversible_generator_Q_ij_s_inv) > 0
    assert np.all(np.isfinite(result.symmetric_capacity_fluxes_K_ij_mol_m3_s))
    assert np.all(np.isfinite(result.reversible_generator_Q_ij_s_inv))
    assert np.all(
        np.asarray(
            result.effect_attribution["transition_edge_capacity_fluxes_K_ij_mol_m3_s"],
            dtype=float,
        )
        > 0.0
    )
    transition_families = tuple(result.effect_attribution["transition_edge_families"])
    assert "partner_switch" in transition_families
    assert "identity_diffusion" in transition_families
    assert "structural_hop" in transition_families
    assert "pair_to_aggregate" in transition_families
    assert "bridge_network_formation_breakup" in transition_families
    assert "cage_capture_release" in transition_families
    assert "atmosphere_capture_release" in transition_families
    state_labels = tuple(result.effect_attribution["state_labels"])
    assert any("|aggregate|" in state_label for state_label in state_labels)
    assert any("|bridge_network|" in state_label for state_label in state_labels)
    assert any("|cage_captured|" in state_label for state_label in state_labels)
    assert any("|atmosphere_captured" in state_label for state_label in state_labels)
    assert tuple(result.effect_attribution["component_names"]) == ("Li+", "PF6-", "FSI-")
    state_additive_stoichiometry = np.asarray(
        result.effect_attribution["state_additive_stoichiometry"],
        dtype=float,
    )
    assert np.any(state_additive_stoichiometry > 0.0)
    assert np.any(state_additive_stoichiometry == 0.0)
    component_residuals_mol_m3 = np.asarray(
        result.effect_attribution["component_mass_balance_residuals_mol_m3"],
        dtype=float,
    )
    component_totals_mol_m3 = np.asarray(
        result.effect_attribution["component_total_concentrations_mol_m3"],
        dtype=float,
    )
    assert np.max(np.abs(component_residuals_mol_m3)) <= (
        np.max(component_totals_mol_m3) * 1.0e-8
    )
    edge_second_moment_traces_m2 = np.asarray(
        result.effect_attribution["transition_edge_second_moment_traces_m2"],
        dtype=float,
    )
    edge_first_moment_norms_m = np.asarray(
        result.effect_attribution["transition_edge_first_moment_norms_m"],
        dtype=float,
    )
    assert np.any(edge_second_moment_traces_m2 > 0.0)
    assert np.any(edge_first_moment_norms_m > 0.0)
    assert float(
        np.sum(
            np.asarray(
                result.effect_attribution["trace_transition_direct_by_edge"],
                dtype=float,
            )
        )
    ) > 0.0
    assert result.effect_attribution["trace_finite_state_memory_correction"] > 0.0
    assert np.all(
        np.isfinite(
            np.asarray(result.effect_attribution["active_state_lifetimes_s"], dtype=float)
        )
    )
    state_charge_mobility = np.asarray(
        result.effect_attribution["state_charge_mobility_zDz_m2_s"],
        dtype=float,
    )
    state_self_current_trace_average = np.trace(
        result.self_current_tensors_D_self_i_m2_s,
        axis1=1,
        axis2=2,
    ) / float(CARTESIAN)
    assert state_charge_mobility == pytest.approx(state_self_current_trace_average)
    cation_anion_cross_mobility = np.asarray(
        result.effect_attribution["state_cation_anion_cross_mobility_zDz_m2_s"],
        dtype=float,
    )
    assert np.any(np.abs(cation_anion_cross_mobility) > 0.0)


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
    thresholds = extract_projected_primitives.AssociationThresholds(
        contact_pair_max_distance_A=4.0,
        solvent_separated_pair_max_distance_A=10.0,
    )
    trajectory_label = extract_projected_primitives._active_sparse_state_label_for_center(
        records=records,
        mixture=mixture,
        center_frame=center_frame,
        center_catalog=center_catalog,
        center_index=0,
        counterion_index=1,
        distances_A=np.asarray([pair_distance_A], dtype=float),
        thresholds=thresholds,
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
            PairBasin.SOLVENT_SEPARATED_ION_PAIR.value,
            np.asarray([pair_distance_A], dtype=float),
            thresholds,
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

    assert crowded_fields.ionic_strength_mol_m3 > dilute_fields.ionic_strength_mol_m3
    assert crowded_fields.dielectric_constant < dilute_fields.dielectric_constant
    assert crowded_fields.viscosity_Pa_s > dilute_fields.viscosity_Pa_s
    assert crowded_charge_mobility < dilute_charge_mobility


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

    with pytest.raises(ValueError, match="above derived upper bound"):
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
    resistance = compute_resistance_tensor_kg_s(
        records,
        colocated_configuration,
        viscosity_Pa_s=1.0e-3,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=1000.0,
        temperature_K=T_REF_K,
        local_packing_fraction=compute_local_packing_fraction(records, colocated_configuration),
    )
    no_atmosphere_resistance = compute_resistance_tensor_kg_s(
        _records_with_atmosphere_lambda(records, 0.0),
        colocated_configuration,
        viscosity_Pa_s=1.0e-3,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=1000.0,
        temperature_K=T_REF_K,
        local_packing_fraction=compute_local_packing_fraction(records, colocated_configuration),
    )

    assert float(translation_vector @ resistance @ translation_vector) == pytest.approx(
        float(translation_vector @ no_atmosphere_resistance @ translation_vector)
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
    translation_vector = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    zero_ionic_resistance = compute_resistance_tensor_kg_s(
        records,
        configuration,
        viscosity_Pa_s=1.0e-3,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=0.0,
        temperature_K=T_REF_K,
        local_packing_fraction=compute_local_packing_fraction(records, configuration),
    )
    no_atmosphere_resistance = compute_resistance_tensor_kg_s(
        _records_with_atmosphere_lambda(records, 0.0),
        configuration,
        viscosity_Pa_s=1.0e-3,
        dielectric_constant=20.0,
        ionic_strength_mol_m3=0.0,
        temperature_K=T_REF_K,
        local_packing_fraction=compute_local_packing_fraction(records, configuration),
    )

    assert float(translation_vector @ zero_ionic_resistance @ translation_vector) == (
        pytest.approx(float(translation_vector @ no_atmosphere_resistance @ translation_vector))
    )


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
    return records.__class__(
        root=records.root,
        manifest=records.manifest,
        species_records={
            **records.species_records,
            "X+": _single_site_charged_species_record("cation", 1.0),
            "X-": _single_site_charged_species_record("anion", -1.0),
        },
        pair_records=records.pair_records,
        mixture_record=records.mixture_record,
        basis_record=records.basis_record,
        transition_record=records.transition_record,
        memory_record=records.memory_record,
    )


def _records_with_atmosphere_lambda(records, atmosphere_lambda_kg_s: float):
    return records.__class__(
        root=records.root,
        manifest=records.manifest,
        species_records=records.species_records,
        pair_records=records.pair_records,
        mixture_record={
            **records.mixture_record,
            "mobility": {
                **records.mixture_record["mobility"],
                "atmosphere_lambda_kg_s": atmosphere_lambda_kg_s,
            },
        },
        basis_record=records.basis_record,
        transition_record=records.transition_record,
        memory_record=records.memory_record,
    )


def _single_site_charged_species_record(role: str, charge_number: float) -> dict:
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
                "charge_cloud_radius_m": 1.0e-10,
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
    physical_objects = build_physical_objects(
        records,
        configuration,
        temperature_K,
        local_fields.dielectric_constant,
        local_fields.viscosity_Pa_s,
        local_fields.ionic_strength_mol_m3,
        local_fields.local_packing_fraction,
    )
    return float(
        np.trace(
            physical_objects.charge_polarization_gradient
            @ physical_objects.mobility_tensor_m2_s
            @ physical_objects.charge_polarization_gradient.T
        )
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
