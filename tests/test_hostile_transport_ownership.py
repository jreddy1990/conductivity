from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from conductivity.physical_library import generator_construction
from conductivity.physical_library import physical_generator_builder
from conductivity.physical_library import projected_analytical_conductivity as model
from conductivity.physical_library.library_io import load_physical_library
from conductivity.physical_library.reduced_generator import (
    ReducedGeneratorSpecification,
    ReducedStateQuadrature,
    ReducedTransitionQuadrature,
    build_projected_generator_input,
)
from conductivity.physical_library.transition_moment_bvp import (
    EndpointTransportMomentInput,
    build_endpoint_transport_moments,
)


def _zero_potential_J_mol(point: np.ndarray) -> float:
    return 0.0


def _offset_linear_potential_J_mol(point: np.ndarray) -> float:
    return 100.0 + float(point[0])


def _unit_mobility_tensor_m2_s(point: np.ndarray) -> np.ndarray:
    return np.eye(point.size, dtype=float)


def _zero_charge_gradient(point: np.ndarray) -> np.ndarray:
    return np.zeros((3, point.size), dtype=float)


def _empty_memory_gradient(point: np.ndarray) -> np.ndarray:
    return np.zeros((0, point.size), dtype=float)


def _two_coordinate_charge_gradient(point: np.ndarray) -> np.ndarray:
    if point.size != 2:
        raise ValueError("two-coordinate charge gradient requires two coordinates")
    return np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)


def _ownership_basis(
    transition_gradients: np.ndarray,
    transition_edge_indices: np.ndarray,
    bounded_memory_gradients: np.ndarray,
    bounded_memory_mode_indices: np.ndarray,
    diagnostic_gradients: np.ndarray,
    diagnostic_source_ids: tuple[str, ...],
) -> model.StateTransportOwnershipBasis:
    return model.StateTransportOwnershipBasis(
        transition_displacement_gradients=transition_gradients,
        transition_edge_indices=transition_edge_indices,
        bounded_memory_gradients=bounded_memory_gradients,
        bounded_memory_mode_indices=bounded_memory_mode_indices,
        diagnostic_gradients=diagnostic_gradients,
        diagnostic_source_ids=diagnostic_source_ids,
    )


def _empty_ownership_basis(coordinate_dimension: int):
    return _ownership_basis(
        transition_gradients=np.empty((0, coordinate_dimension)),
        transition_edge_indices=np.empty(0, dtype=int),
        bounded_memory_gradients=np.empty((0, coordinate_dimension)),
        bounded_memory_mode_indices=np.empty(0, dtype=int),
        diagnostic_gradients=np.empty((0, coordinate_dimension)),
        diagnostic_source_ids=(),
    )


def test_transport_ownership_pointwise_anisotropic_four_owner_closure() -> None:
    mobility_tensor_m2_s = np.diag([2.0, 3.0, 5.0, 7.0])
    charge_polarization_gradient = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    tensor_set = model.compute_transport_ownership_tensor_set(
        state_index=3,
        quadrature_index=5,
        mobility_tensor_m2_s=mobility_tensor_m2_s,
        charge_polarization_gradient=charge_polarization_gradient,
        ownership_basis=_ownership_basis(
            transition_gradients=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            transition_edge_indices=np.asarray([11]),
            bounded_memory_gradients=np.asarray([[1.0, 1.0, 0.0, 0.0]]),
            bounded_memory_mode_indices=np.asarray([13]),
            diagnostic_gradients=np.asarray([[0.0, 1.0, 1.0, 0.0]]),
            diagnostic_source_ids=("shape_audit",),
        ),
    )

    owner_tensors = (
        tensor_set.dc_self_tensor_m2_s,
        tensor_set.transition_displacement_tensor_m2_s,
        tensor_set.bounded_memory_tensor_m2_s,
        tensor_set.diagnostic_tensor_m2_s,
    )
    assert tensor_set.state_index == 3
    assert tensor_set.quadrature_index == 5
    assert tensor_set.coordinate_support_rank == 4
    assert tensor_set.transition_rank == 1
    assert tensor_set.bounded_memory_rank == 1
    assert tensor_set.diagnostic_rank == 1
    for owner_tensor in owner_tensors:
        assert np.min(np.linalg.eigvalsh(owner_tensor)) >= -1.0e-14
    for first_index, first_tensor in enumerate(owner_tensors):
        for second_tensor in owner_tensors[first_index + 1 :]:
            assert np.trace(first_tensor @ second_tensor) == pytest.approx(0.0)
    assert sum(owner_tensors, start=np.zeros((3, 3))) == pytest.approx(
        tensor_set.full_short_time_tensor_m2_s
    )
    assert tensor_set.closure_residual_tensor_m2_s == pytest.approx(
        np.zeros((3, 3))
    )


def test_transport_ownership_residualizes_cross_owner_overlap() -> None:
    tensor_set = model.compute_transport_ownership_tensor_set(
        state_index=0,
        quadrature_index=0,
        mobility_tensor_m2_s=np.diag([2.0, 3.0, 5.0]),
        charge_polarization_gradient=np.eye(3),
        ownership_basis=_ownership_basis(
            transition_gradients=np.asarray([[1.0, 0.0, 0.0]]),
            transition_edge_indices=np.asarray([0]),
            bounded_memory_gradients=np.asarray([[1.0, 1.0, 0.0]]),
            bounded_memory_mode_indices=np.asarray([0]),
            diagnostic_gradients=np.empty((0, 3)),
            diagnostic_source_ids=(),
        ),
    )

    assert tensor_set.transition_displacement_tensor_m2_s == pytest.approx(
        np.diag([2.0, 0.0, 0.0])
    )
    assert tensor_set.bounded_memory_tensor_m2_s == pytest.approx(
        np.diag([0.0, 3.0, 0.0])
    )
    assert tensor_set.dc_self_tensor_m2_s == pytest.approx(
        np.diag([0.0, 0.0, 5.0])
    )


def test_transport_ownership_rejects_diagnostic_nonzero_current_with_code() -> None:
    with pytest.raises(ValueError, match="^DIAGNOSTIC_CURRENT_NONZERO:"):
        model.compute_transport_ownership_tensor_set(
            state_index=2,
            quadrature_index=7,
            mobility_tensor_m2_s=np.eye(2),
            charge_polarization_gradient=np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float
            ),
            ownership_basis=_ownership_basis(
                transition_gradients=np.empty((0, 2)),
                transition_edge_indices=np.empty(0, dtype=int),
                bounded_memory_gradients=np.empty((0, 2)),
                bounded_memory_mode_indices=np.empty(0, dtype=int),
                diagnostic_gradients=np.asarray([[1.0, 0.0]]),
                diagnostic_source_ids=("forbidden_current",),
            ),
        )


@pytest.mark.parametrize(
    ("corrupted_field", "corrupted_value", "failure_code"),
    (
        (
            "transition_edge_indices",
            np.empty(0, dtype=int),
            "TRANSITION_OWNER_SOURCE_CARDINALITY_FAILED",
        ),
        (
            "bounded_memory_mode_indices",
            np.empty(0, dtype=int),
            "MEMORY_OWNER_SOURCE_CARDINALITY_FAILED",
        ),
        (
            "diagnostic_source_ids",
            (),
            "DIAGNOSTIC_OWNER_SOURCE_CARDINALITY_FAILED",
        ),
    ),
)
def test_transport_ownership_rejects_source_cardinality_with_code(
    corrupted_field: str,
    corrupted_value: np.ndarray | tuple[str, ...],
    failure_code: str,
) -> None:
    ownership_basis = _ownership_basis(
        transition_gradients=np.asarray([[1.0, 0.0]]),
        transition_edge_indices=np.asarray([0]),
        bounded_memory_gradients=np.asarray([[0.0, 1.0]]),
        bounded_memory_mode_indices=np.asarray([0]),
        diagnostic_gradients=np.asarray([[1.0, 1.0]]),
        diagnostic_source_ids=("audit",),
    )
    object.__setattr__(ownership_basis, corrupted_field, corrupted_value)
    with pytest.raises(ValueError, match=f"^{failure_code}$"):
        model.compute_transport_ownership_tensor_set(
            state_index=0,
            quadrature_index=0,
            mobility_tensor_m2_s=np.eye(2),
            charge_polarization_gradient=np.zeros((3, 2)),
            ownership_basis=ownership_basis,
        )


def test_static_local_field_rows_are_not_mori_owners() -> None:
    padded = physical_generator_builder._pad_transport_ownership_basis(
        _empty_ownership_basis(2),
        generator_dimension=7,
        physical_position_coordinate_count=2,
    )

    assert padded.bounded_memory_mode_indices.size == 0
    assert padded.bounded_memory_gradients.shape == (0, 7)


def test_molecular_charge_center_excludes_rigid_internal_rotation() -> None:
    identity = np.eye(3, dtype=float)
    internal_rotation_mobility = np.block(
        [[identity, -identity], [-identity, identity]]
    )
    center = generator_construction.MolecularChargeCenter(
        label="anion:0",
        formal_charge_number=-1.0,
        site_indices=(0, 1),
        center_of_mass_weights=(0.5, 0.5),
    )

    center_mobility = generator_construction._charged_center_mobility_matrix(
        internal_rotation_mobility,
        site_count=2,
        charged_centers=(center,),
    )

    assert center_mobility == pytest.approx(np.zeros((1, 1)))


def test_orientation_bins_preserve_unsplit_cosine_measure() -> None:
    records = load_physical_library(
        Path(__file__).resolve().parents[1] / "physical_library"
    )
    orientation_bins = records.basis_record["orientation_bins"]
    values, weights = generator_construction._nodes_from_thresholds(
        -1.0,
        1.0,
        [
            orientation_bins["bridging_max"],
            -float(orientation_bins["tangential_abs_max"]),
            orientation_bins["tangential_abs_max"],
            orientation_bins["radial_min"],
        ],
    )

    assert values.size == 5
    assert float(np.sum(weights)) == pytest.approx(2.0)


def test_energy_reference_preserves_disconnected_basin_energy_difference() -> None:
    state_quadratures = tuple(
        ReducedStateQuadrature(
            points=np.asarray([[coordinate]], dtype=float),
            weights=np.asarray([1.0], dtype=float),
            stoichiometry=np.asarray([1.0], dtype=float),
            self_current_projector=np.eye(1, dtype=float),
            transport_ownership_bases=(_empty_ownership_basis(1),),
            relative_displacement_fluctuations_m=np.empty((0, 3), dtype=float),
            relative_displacement_mobility_m2_s=np.empty((0, 0), dtype=float),
            relative_center_charge_numbers=np.empty(0, dtype=float),
        )
        for coordinate in (2.0, 7.0)
    )
    specification = ReducedGeneratorSpecification(
        potential_energy_J_mol=_offset_linear_potential_J_mol,
        mobility_tensor_m2_s=_unit_mobility_tensor_m2_s,
        charge_polarization_gradient=_zero_charge_gradient,
        memory_coordinate_gradient=_empty_memory_gradient,
        state_quadratures=state_quadratures,
        transition_quadratures=(),
        total_component_concentrations_mol_m3=np.asarray([1.0], dtype=float),
        temperature_K=300.0,
        volume_m3=1.0,
    )

    normalized = generator_construction._normalize_potential_energy_reference(
        specification
    )
    normalized_energies = tuple(
        normalized.potential_energy_J_mol(state.points[0])
        for state in normalized.state_quadratures
    )

    assert normalized_energies == pytest.approx((0.0, 5.0))


def test_mori_owner_consumer_closure_rejects_duplicate_owner() -> None:
    with pytest.raises(
        ValueError, match="^MORI_OWNER_CONSUMER_CLOSURE_DUPLICATE_OWNER$"
    ):
        model._validate_bounded_memory_owner_consumer_closure(
            np.asarray([2, 2], dtype=int),
            np.eye(2),
            np.zeros((2, 3)),
        )


def test_mori_owner_consumer_closure_rejects_missing_A_h_row() -> None:
    with pytest.raises(ValueError, match="mori_memory_matrix_A must have shape"):
        model._validate_bounded_memory_owner_consumer_closure(
            np.asarray([2, 3], dtype=int),
            np.eye(1),
            np.zeros((1, 3)),
        )


def _primitive_readout(
    concentrations_mol_m3: np.ndarray,
    capacity_fluxes_mol_m3_s: np.ndarray,
    first_moments_m: np.ndarray,
    second_moments_m2: np.ndarray,
    self_current_tensors_m2_s: np.ndarray,
) -> model.ProjectedConductivityResult:
    return model.compute_projected_analytical_conductivity_from_primitives(
        concentrations_mol_m3,
        capacity_fluxes_mol_m3_s,
        first_moments_m,
        second_moments_m2,
        self_current_tensors_m2_s,
        np.zeros((0, 0), dtype=float),
        np.zeros((0, 3), dtype=float),
        300.0,
    )


def _transport_closure_records() -> SimpleNamespace:
    return SimpleNamespace(
        transition_record={
            "transition_records": {
                "displacement_family": {
                    "transport_ownership": "transition_displacement"
                },
                "memory_family": {"transport_ownership": "bounded_memory"},
            }
        }
    )


def test_transport_graph_closure_accepts_retained_displacement_edges() -> None:
    displacement_edges = (
        generator_construction.TransitionEdge(0, 1, "displacement_family"),
        generator_construction.TransitionEdge(1, 0, "displacement_family"),
    )
    generator_construction._validate_transport_graph_closure(
        records=_transport_closure_records(),
        state_quadratures=(
            SimpleNamespace(label="state-0", configurations=(None,)),
            SimpleNamespace(label="state-1", configurations=(None,)),
        ),
        declared_edges=displacement_edges,
        retained_edges=displacement_edges,
    )


def test_transport_graph_closure_rejects_pruned_displacement_edges() -> None:
    displacement_edges = (
        generator_construction.TransitionEdge(0, 1, "displacement_family"),
        generator_construction.TransitionEdge(1, 0, "displacement_family"),
    )
    retained_memory_edge = generator_construction.TransitionEdge(0, 2, "memory_family")
    with pytest.raises(
        ValueError,
        match="finite-generator transport graph is not closed before readout",
    ):
        generator_construction._validate_transport_graph_closure(
            records=_transport_closure_records(),
            state_quadratures=tuple(
                SimpleNamespace(label=f"state-{state_index}", configurations=(None,))
                for state_index in range(3)
            ),
            declared_edges=displacement_edges + (retained_memory_edge,),
            retained_edges=(retained_memory_edge,),
        )


def test_partner_residence_memory_declares_partner_switch_owner() -> None:
    records = generator_construction.build_recipe_library_context(
        generator_construction.Path(
            "conductivity/physical_library/recipe_ec_dmc_lipf6_1m.yaml"
        ),
        generator_construction.Path("conductivity/physical_library"),
    ).library_records
    partner_residence_record = records.memory_record["memory_records"][
        "partner_residence"
    ]

    assert partner_residence_record["transport_ownership"] == "bounded_memory"
    assert partner_residence_record["matching_transition_families"] == [
        "partner_switch"
    ]
    assert (
        records.transition_record["transition_records"]["partner_switch"][
            "transport_ownership"
        ]
        == "transition_displacement"
    )


def test_full_self_plus_transition_overcounts_while_tangent_self_is_correct() -> None:
    concentrations = np.asarray([2.0, 3.0], dtype=float)
    capacity_flux = 5.0
    capacity_fluxes = np.asarray(
        [[0.0, capacity_flux], [capacity_flux, 0.0]], dtype=float
    )
    crossing_second_moment = np.diag([7.0, 0.0, 0.0])
    second_moments = np.asarray(
        [
            [np.zeros((3, 3)), crossing_second_moment],
            [crossing_second_moment, np.zeros((3, 3))],
        ],
        dtype=float,
    )
    full_self_current_tensors = np.asarray(
        [np.diag([11.0, 0.0, 0.0]), np.diag([13.0, 0.0, 0.0])],
        dtype=float,
    )

    overcounted_result = _primitive_readout(
        concentrations,
        capacity_fluxes,
        np.zeros((2, 2, 3), dtype=float),
        second_moments,
        full_self_current_tensors,
    )
    tangent_result = _primitive_readout(
        concentrations,
        capacity_fluxes,
        np.zeros((2, 2, 3), dtype=float),
        second_moments,
        np.zeros((2, 3, 3), dtype=float),
    )

    full_self_normal_budget = 2.0 * 11.0 + 3.0 * 13.0
    analytic_transition_budget = capacity_flux * 7.0
    assert overcounted_result.direct_diffusivity_tensor[0, 0] == pytest.approx(
        analytic_transition_budget + full_self_normal_budget
    )
    assert tangent_result.direct_diffusivity_tensor[0, 0] == pytest.approx(
        analytic_transition_budget
    )
    assert (
        overcounted_result.direct_diffusivity_tensor[0, 0]
        - tangent_result.direct_diffusivity_tensor[0, 0]
    ) == pytest.approx(full_self_normal_budget)


def _two_state_family_specification() -> ReducedGeneratorSpecification:
    coordinate_projector = np.diag([1.0, 0.0])
    states = tuple(
        ReducedStateQuadrature(
            points=np.asarray([[coordinate, 0.0]], dtype=float),
            weights=np.asarray([1.0], dtype=float),
            stoichiometry=np.asarray([1.0], dtype=float),
            self_current_projector=coordinate_projector,
            transport_ownership_bases=(
                _ownership_basis(
                    transition_gradients=np.asarray([[1.0, 0.0]], dtype=float),
                    transition_edge_indices=np.asarray([0], dtype=int),
                    bounded_memory_gradients=np.empty((0, 2), dtype=float),
                    bounded_memory_mode_indices=np.empty(0, dtype=int),
                    diagnostic_gradients=np.empty((0, 2), dtype=float),
                    diagnostic_source_ids=(),
                ),
            ),
            relative_displacement_fluctuations_m=np.empty((0, 3), dtype=float),
            relative_displacement_mobility_m2_s=np.empty((0, 0), dtype=float),
            relative_center_charge_numbers=np.empty(0, dtype=float),
        )
        for coordinate in (-1.0, 1.0)
    )
    transition = ReducedTransitionQuadrature(
        from_state_index=0,
        to_state_index=1,
        transition_family="test_transition",
        transport_ownership=model.TransportOwnership.TRANSITION_DISPLACEMENT,
        points=np.asarray([[0.0, 0.0]], dtype=float),
        weights=np.asarray([1.0], dtype=float),
        committor_gradients=np.asarray([[1.0, 0.0]], dtype=float),
        surface_state_indices=np.asarray([0], dtype=int),
        path_displacements_m=np.asarray([[2.0, 0.0, 0.0]], dtype=float),
        path_weights=np.asarray([1.0], dtype=float),
        first_displacement_moment_m=np.asarray([2.0, 0.0, 0.0], dtype=float),
        second_displacement_moment_m2=np.diag([4.0, 0.0, 0.0]),
        log_capacity_integral=0.0,
        uses_residence_rate_constant=True,
        residence_rate_constant_s_inv=2.0,
    )
    return ReducedGeneratorSpecification(
        potential_energy_J_mol=_zero_potential_J_mol,
        mobility_tensor_m2_s=_unit_mobility_tensor_m2_s,
        charge_polarization_gradient=_two_coordinate_charge_gradient,
        memory_coordinate_gradient=_empty_memory_gradient,
        state_quadratures=states,
        transition_quadratures=(transition,),
        total_component_concentrations_mol_m3=np.asarray([2.0], dtype=float),
        temperature_K=300.0,
        volume_m3=1.0,
    )


@pytest.mark.parametrize(
    "transition_family",
    tuple(
        family
        for family, transition_record in generator_construction.build_recipe_library_context(
            generator_construction.Path(
                "conductivity/physical_library/recipe_ec_dmc_lipf6_1m.yaml"
            ),
            generator_construction.Path("conductivity/physical_library"),
        )
        .library_records.transition_record["transition_records"]
        .items()
        if transition_record["transport_ownership"] == "transition_displacement"
    ),
)
def test_production_transition_family_closes_transport_ownership_ledger(
    transition_family: str,
) -> None:
    records = generator_construction.build_recipe_library_context(
        generator_construction.Path(
            "conductivity/physical_library/recipe_ec_dmc_lipf6_1m.yaml"
        ),
        generator_construction.Path("conductivity/physical_library"),
    ).library_records
    assert set(records.transition_record["families"]) == set(
        records.transition_record["transition_records"]
    )
    projected_input = build_projected_generator_input(
        _two_state_family_specification()
    )

    assert len(projected_input.transition_committor_gradients) == 1
    assert tuple(
        int(surface_states[0])
        for surface_states in projected_input.transition_surface_state_indices
    ) == (0,)
    full_self_tensors, tangent_self_tensors, *_owner_tensors = (
        model.compute_state_transport_ownership_quadratures(
            mobility_tensor_m2_s=projected_input.mobility_tensor_m2_s,
            charge_polarization_gradient=projected_input.charge_polarization_gradient,
            basin_quadrature_points=projected_input.basin_quadrature_points,
            basin_density_weights_mol_m3=(np.asarray([1.0]), np.asarray([1.0])),
            basin_concentrations_mol_m3=np.asarray([1.0, 1.0]),
            state_transport_ownership_bases=(
                projected_input.state_transport_ownership_bases
            ),
        )
    )
    assert full_self_tensors == pytest.approx(
        np.asarray([np.diag([1.0, 1.0, 0.0])] * 2)
    )
    assert tangent_self_tensors == pytest.approx(
        np.asarray([np.diag([0.0, 1.0, 0.0])] * 2)
    )

    concentrations = np.asarray([1.0, 1.0])
    generator = np.asarray([[-1.0, 1.0], [1.0, -1.0]])
    capacity_fluxes = concentrations[:, None] * generator
    finite_state_correction = model.compute_finite_state_memory_correction(
        concentrations,
        generator,
        projected_input.transition_first_moments_d_ij_m,
    )
    ledger = model.compute_direct_primitive_audit_ledger(
        concentrations,
        capacity_fluxes,
        generator,
        projected_input.transition_first_moments_d_ij_m,
        projected_input.transition_second_moments_M_ij_m2,
        full_self_tensors,
        tangent_self_tensors,
        finite_state_correction,
    )
    assert ledger.B_self_full_tensor_mol_m_s == pytest.approx(
        ledger.B_self_tangent_tensor_mol_m_s + ledger.B_overlap_removed_tensor_mol_m_s
    )
    assert ledger.B_total_tensor_mol_m_s == pytest.approx(
        ledger.B_self_tangent_tensor_mol_m_s + ledger.B_transition_tensor_mol_m_s
    )


def test_constructor_preserves_ill_scaled_independent_normal_row_space() -> None:
    specification = _two_state_family_specification()
    transition = specification.transition_quadratures[0]
    ill_scaled_transition = ReducedTransitionQuadrature(
        from_state_index=transition.from_state_index,
        to_state_index=transition.to_state_index,
        transition_family=transition.transition_family,
        transport_ownership=transition.transport_ownership,
        points=np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        weights=np.asarray([0.5, 0.5]),
        committor_gradients=np.asarray([[1.0e20, 0.0], [0.0, 1.0e-20]]),
        surface_state_indices=np.asarray([0, 0]),
        path_displacements_m=transition.path_displacements_m,
        path_weights=transition.path_weights,
        first_displacement_moment_m=transition.first_displacement_moment_m,
        second_displacement_moment_m2=transition.second_displacement_moment_m2,
        log_capacity_integral=transition.log_capacity_integral,
        uses_residence_rate_constant=transition.uses_residence_rate_constant,
        residence_rate_constant_s_inv=transition.residence_rate_constant_s_inv,
    )
    ill_scaled_states = tuple(
        ReducedStateQuadrature(
            points=state.points,
            weights=state.weights,
            stoichiometry=state.stoichiometry,
            self_current_projector=state.self_current_projector,
            transport_ownership_bases=(
                _ownership_basis(
                    transition_gradients=np.asarray(
                        [[1.0e20, 0.0], [0.0, 1.0e-20]],
                        dtype=float,
                    ),
                    transition_edge_indices=np.asarray([0, 0], dtype=int),
                    bounded_memory_gradients=np.empty((0, 2), dtype=float),
                    bounded_memory_mode_indices=np.empty(0, dtype=int),
                    diagnostic_gradients=np.empty((0, 2), dtype=float),
                    diagnostic_source_ids=(),
                ),
            ),
            relative_displacement_fluctuations_m=(
                state.relative_displacement_fluctuations_m
            ),
            relative_displacement_mobility_m2_s=(
                state.relative_displacement_mobility_m2_s
            ),
            relative_center_charge_numbers=state.relative_center_charge_numbers,
        )
        for state in specification.state_quadratures
    )
    specification = ReducedGeneratorSpecification(
        potential_energy_J_mol=specification.potential_energy_J_mol,
        mobility_tensor_m2_s=specification.mobility_tensor_m2_s,
        charge_polarization_gradient=specification.charge_polarization_gradient,
        memory_coordinate_gradient=specification.memory_coordinate_gradient,
        state_quadratures=ill_scaled_states,
        transition_quadratures=(ill_scaled_transition,),
        total_component_concentrations_mol_m3=(
            specification.total_component_concentrations_mol_m3
        ),
        temperature_K=specification.temperature_K,
        volume_m3=specification.volume_m3,
    )
    projected_input = build_projected_generator_input(specification)

    for state_index in (0, 1):
        normals = projected_input.state_transport_ownership_bases[state_index][
            0
        ].transition_displacement_gradients
        np.testing.assert_array_equal(
            normals,
            np.asarray([[1.0e20, 0.0], [0.0, 1.0e-20]], dtype=float),
        )


def test_ssip_center_transport_budget_not_double_counted() -> None:
    ssip_concentration_mol_m3 = 4.0
    ssip_center_diffusivity_m2_s = 2.5
    result = _primitive_readout(
        np.asarray([ssip_concentration_mol_m3], dtype=float),
        np.zeros((1, 1), dtype=float),
        np.zeros((1, 1, 3), dtype=float),
        np.zeros((1, 1, 3, 3), dtype=float),
        np.asarray([np.diag([ssip_center_diffusivity_m2_s, 0.0, 0.0])]),
    )

    expected_ssip_budget = ssip_concentration_mol_m3 * ssip_center_diffusivity_m2_s
    assert result.direct_diffusivity_tensor == pytest.approx(
        np.diag([expected_ssip_budget, 0.0, 0.0])
    )
    assert result.effect_attribution[
        "trace_transition_direct_by_edge"
    ] == pytest.approx(np.zeros((1, 1), dtype=float))


def test_two_basin_1d_full_crossing_is_separate_from_tangent_motion() -> None:
    crossing_displacement_m = np.asarray([3.0e-10, 0.0, 0.0], dtype=float)
    first_moment, crossing_second_moment = build_endpoint_transport_moments(
        EndpointTransportMomentInput(
            endpoint_displacement_m=crossing_displacement_m,
            directed_endpoint=True,
        )
    )
    capacity_fluxes = np.asarray([[0.0, 2.0e20], [2.0e20, 0.0]], dtype=float)
    first_moments = np.asarray(
        [
            [np.zeros(3, dtype=float), first_moment],
            [-first_moment, np.zeros(3, dtype=float)],
        ]
    )
    second_moments = np.asarray(
        [
            [np.zeros((3, 3), dtype=float), crossing_second_moment],
            [crossing_second_moment, np.zeros((3, 3), dtype=float)],
        ]
    )
    tangent_self_current = np.asarray(
        [np.diag([0.0, 5.0, 0.0]), np.diag([0.0, 7.0, 0.0])], dtype=float
    )

    result = _primitive_readout(
        np.ones(2, dtype=float),
        capacity_fluxes,
        first_moments,
        second_moments,
        tangent_self_current,
    )

    assert result.direct_diffusivity_tensor == pytest.approx(np.diag([18.0, 12.0, 0.0]))
    assert result.finite_state_memory_correction_tensor == pytest.approx(
        np.diag([18.0, 0.0, 0.0])
    )
    assert result.projected_diffusivity_tensor == pytest.approx(
        np.diag([0.0, 12.0, 0.0])
    )


def test_transition_second_moment_minus_outer_first_moment_is_psd() -> None:
    first_moment = np.asarray([2.0, -1.0, 0.5], dtype=float)
    analytic_covariance = np.diag([3.0, 4.0, 5.0])
    second_moment = np.outer(first_moment, first_moment) + analytic_covariance
    covariance = second_moment - np.outer(first_moment, first_moment)

    assert covariance == pytest.approx(analytic_covariance)
    assert np.linalg.eigvalsh(covariance) == pytest.approx([3.0, 4.0, 5.0])


def test_reversible_generator_has_fixed_detailed_balance_flux() -> None:
    concentrations = np.asarray([2.0, 5.0], dtype=float)
    capacity_fluxes = np.asarray([[0.0, 10.0], [10.0, 0.0]], dtype=float)

    generator = model.compute_reversible_generator(capacity_fluxes, concentrations)

    assert generator == pytest.approx(
        np.asarray([[-5.0, 5.0], [2.0, -2.0]], dtype=float)
    )
    assert concentrations[:, np.newaxis] * generator == pytest.approx(
        np.asarray([[-10.0, 10.0], [10.0, -10.0]], dtype=float)
    )
    assert concentrations @ generator == pytest.approx(
        np.asarray([0.0, 0.0], dtype=float)
    )


def test_reduced_generator_projection_owns_explicit_transition_moments() -> None:
    supplied_first_moment = np.asarray([2.0e-10, 0.0, 0.0], dtype=float)
    supplied_second_moment = np.diag([5.0e-20, 0.0, 0.0])
    transition = ReducedTransitionQuadrature(
        from_state_index=0,
        to_state_index=1,
        transition_family="test_transition",
        transport_ownership=model.TransportOwnership.TRANSITION_DISPLACEMENT,
        points=np.asarray([[0.0]], dtype=float),
        weights=np.asarray([1.0], dtype=float),
        committor_gradients=np.asarray([[1.0]], dtype=float),
        surface_state_indices=np.asarray([0], dtype=int),
        path_displacements_m=np.asarray([[9.0e-10, 0.0, 0.0]], dtype=float),
        path_weights=np.asarray([1.0], dtype=float),
        first_displacement_moment_m=supplied_first_moment,
        second_displacement_moment_m2=supplied_second_moment,
        log_capacity_integral=0.0,
        uses_residence_rate_constant=False,
        residence_rate_constant_s_inv=0.0,
    )
    states = tuple(
        ReducedStateQuadrature(
            points=np.asarray([[coordinate]], dtype=float),
            weights=np.asarray([1.0], dtype=float),
            stoichiometry=np.asarray([1.0], dtype=float),
            self_current_projector=np.eye(1, dtype=float),
            transport_ownership_bases=(_empty_ownership_basis(1),),
            relative_displacement_fluctuations_m=np.empty((0, 3), dtype=float),
            relative_displacement_mobility_m2_s=np.empty((0, 0), dtype=float),
            relative_center_charge_numbers=np.empty(0, dtype=float),
        )
        for coordinate in (-1.0, 1.0)
    )
    specification = ReducedGeneratorSpecification(
        potential_energy_J_mol=_zero_potential_J_mol,
        mobility_tensor_m2_s=_unit_mobility_tensor_m2_s,
        charge_polarization_gradient=_zero_charge_gradient,
        memory_coordinate_gradient=_empty_memory_gradient,
        state_quadratures=states,
        transition_quadratures=(transition,),
        total_component_concentrations_mol_m3=np.asarray([2.0], dtype=float),
        temperature_K=300.0,
        volume_m3=1.0,
    )

    projected_input = build_projected_generator_input(specification)

    assert projected_input.transition_first_moments_d_ij_m[0, 1] == pytest.approx(
        supplied_first_moment
    )
    assert projected_input.transition_first_moments_d_ij_m[1, 0] == pytest.approx(
        -supplied_first_moment
    )
    assert projected_input.transition_second_moments_M_ij_m2[0, 1] == pytest.approx(
        supplied_second_moment
    )
    assert projected_input.transition_second_moments_M_ij_m2[1, 0] == pytest.approx(
        supplied_second_moment
    )
