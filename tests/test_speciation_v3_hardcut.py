from dataclasses import dataclass, fields
import inspect
import math
import operator

import numpy as np
import pytest

from constants import F, PA_PER_ATM, R, S_M_TO_MS_CM, T_REF_K
from data.electrolyte_property_db import DATA
from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS
import conductivity.analytical_conductivity_model as analytical_model
from conductivity.generic_speciation import (
    CONTACT_PAIR_CLUSTER_KIND,
    NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
)
from conductivity.fit_conductivity_primitive_parameters import (
    C_STABLE_DIRECT_CAPACITY_BLOCK_PARAMETER_NAMES,
    DescriptorCalibrationTarget,
    MolecularPropertyDbPrimitiveEvaluator,
    PrimitiveFitDatasetEvaluation,
    PrimitivePromotionMetrics,
    _calibration_row_bucket_report_mapping,
    default_molecular_primitive_fit_configuration,
    initialize_topology_logk_offsets_from_trajectory_concentrations,
    load_trajectory_primitive_calibration_targets,
    primitive_parameter_promotion_rejection_reasons,
    primitive_prediction_sensitivity_diagnostics,
    trajectory_concentration_target_coverage,
)
from conductivity.molecular_descriptors import (
    MolecularSpeciesInput,
    ProvidedPropertyDescriptorBackend,
    ROLE_ADDITIVE,
    ROLE_ANION,
    ROLE_CATION,
    ROLE_SOLVENT,
)
from conductivity.analytical_conductivity_model import (
    CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME,
    EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE,
    FULL_MICROSCOPIC_GENERATOR_DERIVED_PROOF_STATUS,
    MarkovAdditiveConductivityInput,
    MarkovAdditiveEvent,
    PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY,
    PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION,
    PROJECTED_GENERATOR_CLASS_DIAGNOSTIC_ONLY,
    PROJECTED_GENERATOR_CLASS_MORI_MEMORY_BASIS,
    PROJECTED_GENERATOR_CLASS_REACTIVE_FLUX,
    PROJECTED_GENERATOR_CLASS_RESTRICTED_POPULATION,
    PROJECTED_GENERATOR_CLASS_STATE_CHANGING_DISPLACEMENT,
    PROJECTED_GENERATOR_CLASS_WITHIN_STATE_SELF_CURRENT,
    PROJECTED_READOUT_PROVEN_DESCRIPTOR_CLOSURE_EMPIRICAL,
    SUPPORTED_MORI_BASIS_SOURCES,
    MolecularElectrolyteRecipe,
    MolecularMixtureProperties,
    MolecularMoriOptions,
    MolecularSolventEnvironment,
    MolecularTransportCenter,
    MolecularAtmosphereMemoryPrimitive,
    ADDITIVE_SEPARATED_PAIR_CLUSTER_KIND,
    ProjectedChargedCenter,
    ProjectedTransportState,
    TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
    TRANSPORT_ROLE_CLUSTER_COM_CENTER,
    TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER,
    TRANSPORT_ROLE_CONTACT_PAIR_CENTER,
    TRANSPORT_ROLE_FREE_ION_CENTER,
    TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
    TRANSPORT_ROLE_LIGAND_SHELL_CENTER,
    TRANSPORT_ROLE_NEUTRAL_CENTER,
    TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    compute_markov_additive_green_kubo_conductivity,
    compute_molecular_electrolyte_conductivity,
    compute_molecular_electrolyte_conductivity_with_diagnostic_cluster_shifts,
    compute_projected_electrolyte_transport_model,
    compute_projected_transport_state_charge_diffusivity_m2_s,
)
from conductivity.molecular_primitive_parameters import (
    CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES,
    CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES,
    PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED,
    PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE,
    ConductivityPrimitiveParameterSet,
    conductivity_primitive_parameter_coordinate_values_for_names,
    conductivity_primitive_parameters_from_mapping,
    conductivity_primitive_parameters_to_mapping,
    conductivity_primitive_parameters_with_coordinate_updates,
    validate_conductivity_primitive_parameters,
)
from conductivity.molecular_property_db_audit import (
    MolecularClusterThermodynamicDiagnostic,
    MolecularPrimitiveResidualOwner,
    MolecularPropertyDbRegistrySource,
    MolecularPropertyDbRowResult,
    PRIMITIVE_RESIDUAL_COMPONENT_SYNTHETIC_DIRECT_CAPACITY,
    PRIMITIVE_RESIDUAL_HEAD_SELF_CURRENT_MOBILITY,
    PRIMITIVE_RESIDUAL_PRODUCTION_LEVER_ANION_SHAPE_FRICTION,
    PRIMITIVE_RESIDUAL_PRODUCTION_LEVER_LOCAL_OBSTRUCTION,
    PRIMITIVE_RESIDUAL_PRODUCTION_LEVER_SOLVENT_SEPARATED_PAIR_SELF_CURRENT,
    PRIMITIVE_RESIDUAL_THEOREM_SELF_CURRENT_MOBILITY,
    EVENT_FAMILY_PROJECTED_SINGLE_CENTER_SELF_CURRENT,
    audit_molecular_property_db_cases,
    build_molecular_property_db_case_selection,
    configured_conductivity_primitive_parameters,
    default_molecular_property_db_audit_options,
    _production_lever_for_event_family,
)


BASE_CONCENTRATION_M = 1.0
BASE_VISCOSITY_CP = 2.0
BASE_DENSITY_G_ML = 1.2
BASE_DIELECTRIC = 30.0
MAX_PACKING_FRACTION = 0.95
PAIR_CLUSTER_ION_COUNT = 2
TRIPLET_CLUSTER_ION_COUNT = 3
JUMP_LENGTH_MULTIPLIER = 1.0
NEGATIVE_LOGK_OFFSET = -5.0
POSITIVE_SSIP_LOGK_OFFSET = 8.0
POSITIVE_TRIPLET_LOGK_OFFSET = 8.0
NEGATIVE_TRIPLET_LOGK_OFFSET = 8.0


def test_signed_logK_offsets_accept_negative_values():
    primitive_parameters = _primitive_parameters_with_replacements(
        solvent_separated_pair_logK_offset=NEGATIVE_LOGK_OFFSET,
        neutral_cluster_logK_offset=0.0,
    )

    validate_conductivity_primitive_parameters(primitive_parameters)
    coordinate_values = conductivity_primitive_parameter_coordinate_values_for_names(
        primitive_parameters,
        ("solvent_separated_pair_logK_offset",),
    )
    updated_parameters = conductivity_primitive_parameters_with_coordinate_updates(
        primitive_parameters,
        ("solvent_separated_pair_logK_offset",),
        (-7.0,),
    )

    assert coordinate_values == pytest.approx((NEGATIVE_LOGK_OFFSET,))
    assert updated_parameters.solvent_separated_pair_logK_offset == pytest.approx(-7.0)


def test_fit_parameter_metadata_marks_logK_offsets_as_identity_signed():
    _fit_options, parameter_transforms = default_molecular_primitive_fit_configuration()
    transform_by_name = {
        parameter_transform.name: parameter_transform
        for parameter_transform in parameter_transforms
    }

    assert set(transform_by_name) == set(CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES)
    assert (
        transform_by_name["solvent_separated_pair_logK_offset"].transform
        == PRIMITIVE_PARAMETER_TRANSFORM_IDENTITY_SIGNED
    )
    assert transform_by_name["solvent_separated_pair_logK_offset"].lower < 0.0
    assert transform_by_name["solvent_separated_pair_logK_offset"].upper > 0.0
    assert (
        transform_by_name["coulomb_scale"].transform
        == PRIMITIVE_PARAMETER_TRANSFORM_LOG_POSITIVE
    )


def test_c_stable_direct_capacity_block_excludes_speciation_parameters():
    forbidden_c_parameter_names = {
        "contact_pair_desolvation_offset_over_RT",
        "solvent_separated_pair_logK_offset",
        "contact_pair_logK_offset",
        "internal_polarization_projection_offset",
        "cluster_order_logK_slope",
        "cluster_charge_magnitude_logK_slope",
    }

    assert not (
        set(C_STABLE_DIRECT_CAPACITY_BLOCK_PARAMETER_NAMES)
        & forbidden_c_parameter_names
    )
    assert "hydrodynamic_radius_scale_positive_ion" in (
        C_STABLE_DIRECT_CAPACITY_BLOCK_PARAMETER_NAMES
    )
    assert "jump_length_scale" not in C_STABLE_DIRECT_CAPACITY_BLOCK_PARAMETER_NAMES


def test_promotion_rejects_empirical_descriptor_closure_status():
    fit_options, _parameter_transforms = default_molecular_primitive_fit_configuration()
    baseline_metrics = _synthetic_promotion_metrics(())
    candidate_metrics = _synthetic_promotion_metrics(
        (PROJECTED_READOUT_PROVEN_DESCRIPTOR_CLOSURE_EMPIRICAL,)
    )

    rejection_reasons = primitive_parameter_promotion_rejection_reasons(
        baseline_metrics,
        candidate_metrics,
        fit_options,
    )

    assert "descriptor_closure_empirical" in rejection_reasons


def test_calibration_error_decomposition_report_buckets_rows_by_owner():
    fit_options, _parameter_transforms = default_molecular_primitive_fit_configuration()
    direct_capacity_row = _synthetic_property_db_row_result(
        row_id=1,
        empirical_sigma_mS_cm=8.0,
        predicted_sigma_mS_cm=5.0,
        direct_sigma_mS_cm=6.0,
        corrector_sigma_mS_cm=1.0,
        direct_capacity_failure=True,
        corrector_too_strong_failure=False,
        corrector_too_weak_failure=False,
    )
    corrector_too_strong_row = _synthetic_property_db_row_result(
        row_id=2,
        empirical_sigma_mS_cm=4.0,
        predicted_sigma_mS_cm=2.0,
        direct_sigma_mS_cm=5.0,
        corrector_sigma_mS_cm=3.0,
        direct_capacity_failure=False,
        corrector_too_strong_failure=True,
        corrector_too_weak_failure=False,
    )
    corrector_too_weak_row = _synthetic_property_db_row_result(
        row_id=3,
        empirical_sigma_mS_cm=3.0,
        predicted_sigma_mS_cm=6.0,
        direct_sigma_mS_cm=8.0,
        corrector_sigma_mS_cm=2.0,
        direct_capacity_failure=False,
        corrector_too_strong_failure=False,
        corrector_too_weak_failure=True,
    )

    report_mapping = _calibration_row_bucket_report_mapping(
        (
            direct_capacity_row,
            corrector_too_strong_row,
            corrector_too_weak_row,
        ),
        fit_options,
    )

    assert report_mapping["direct_capacity_failures"]["row_count"] == 1
    assert (
        report_mapping["direct_capacity_failures"]["rows"][0]["row_id"]
        == direct_capacity_row.row_id
    )
    assert report_mapping["corrector_too_strong_failures"]["row_count"] == 1
    assert (
        report_mapping["corrector_too_strong_failures"]["rows"][0]["row_id"]
        == corrector_too_strong_row.row_id
    )
    assert report_mapping["corrector_too_weak_failures"]["row_count"] == 1
    assert (
        report_mapping["corrector_too_weak_failures"]["rows"][0]["row_id"]
        == corrector_too_weak_row.row_id
    )
    assert report_mapping["worst_residual_tail"]["row_count"] == 3


def test_property_db_row_reports_projected_primitive_residual_owner():
    row_result = _synthetic_property_db_row_result(
        row_id=4,
        empirical_sigma_mS_cm=8.0,
        predicted_sigma_mS_cm=5.0,
        direct_sigma_mS_cm=6.0,
        corrector_sigma_mS_cm=1.0,
        direct_capacity_failure=True,
        corrector_too_strong_failure=False,
        corrector_too_weak_failure=False,
    )
    owner = row_result.primitive_residual_owners[0]

    assert owner.primitive_head == PRIMITIVE_RESIDUAL_HEAD_SELF_CURRENT_MOBILITY
    assert owner.theorem_object == PRIMITIVE_RESIDUAL_THEOREM_SELF_CURRENT_MOBILITY
    assert (
        owner.production_lever
        == PRIMITIVE_RESIDUAL_PRODUCTION_LEVER_SOLVENT_SEPARATED_PAIR_SELF_CURRENT
    )
    assert (
        owner.residual_component
        == PRIMITIVE_RESIDUAL_COMPONENT_SYNTHETIC_DIRECT_CAPACITY
    )
    assert owner.residual_mS_cm == pytest.approx(2.0)


def test_property_db_audit_summarizes_residual_owners_by_production_lever():
    audit_options = default_molecular_property_db_audit_options()
    registry_source = MolecularPropertyDbRegistrySource(
        solvent_registry=SOLVENTS,
        salt_registry=SALTS,
        additive_registry=ADDITIVES,
        cation_registry=CATION_PROPERTIES,
    )
    case_selection = build_molecular_property_db_case_selection(
        tuple(DATA[:3]),
        registry_source,
        audit_options,
    )
    audit_result = audit_molecular_property_db_cases(
        case_selection.cases,
        configured_conductivity_primitive_parameters(),
        audit_options,
    )

    assert audit_result.primitive_residual_owner_summaries
    for summary in audit_result.primitive_residual_owner_summaries:
        assert summary.row_count > 0
        assert summary.sum_abs_residual_mS_cm >= 0.0
        assert summary.mean_abs_residual_mS_cm >= 0.0
        assert summary.production_lever


def test_single_center_self_current_owner_splits_mobility_factor_levers():
    direct_sigma_by_transport_role_mS_cm = {
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER: 0.0,
    }
    solvent_environment = MolecularSolventEnvironment(
        dielectric_constant=BASE_DIELECTRIC,
        viscosity_cP=1.0,
        hard_sphere_volume_fraction=0.1,
        temperature_K=T_REF_K,
        solvent_effective_radius_A=2.0,
        mean_molecular_volume_A3=100.0,
        solvent_volume_fractions={"EC": 1.0},
        solvent_coordination_affinity_J_mol=0.0,
        additive_ligand_site_occupancy=0.0,
        additive_coordination_affinity_J_mol=0.0,
        additive_solvation_support=0.0,
        additive_molecular_volume_A3=100.0,
    )
    obstructed_cation_center = MolecularTransportCenter(
        label="obstructed_cation",
        parent_cluster_label="free:Li",
        parent_cluster_kind="free",
        concentration_mol_m3=1000.0,
        center_species_name="Li+",
        center_charge_number=1,
        center_index=0,
        hydrodynamic_radius_A=1.0,
        charge_cloud_radius_A=1.0,
        molecular_volume_A3=10.0,
        ligand_field_asymmetry=1.0,
        diffusion_m2_s=1.0e-10,
        local_obstruction_factor=9.0,
        local_obstruction_diffusion_scale=0.1,
        transport_role=TRANSPORT_ROLE_FREE_ION_CENTER,
    )
    shaped_anion_center = MolecularTransportCenter(
        label="shaped_anion",
        parent_cluster_label="free:anion",
        parent_cluster_kind="free",
        concentration_mol_m3=1000.0,
        center_species_name="A-",
        center_charge_number=-1,
        center_index=0,
        hydrodynamic_radius_A=1.0,
        charge_cloud_radius_A=1.0,
        molecular_volume_A3=10.0,
        ligand_field_asymmetry=8.0,
        diffusion_m2_s=1.0e-10,
        local_obstruction_factor=0.0,
        local_obstruction_diffusion_scale=1.0,
        transport_role=TRANSPORT_ROLE_FREE_ION_CENTER,
    )
    obstruction_lever = _production_lever_for_event_family(
        EVENT_FAMILY_PROJECTED_SINGLE_CENTER_SELF_CURRENT,
        direct_sigma_by_transport_role_mS_cm,
        (
            "projected_single_center_self_current:"
            "feature_keyed_transport_center:role=free_ion_center:"
            "center=0:role=cation:z=1"
        ),
        obstructed_cation_center,
        solvent_environment,
    )
    shape_lever = _production_lever_for_event_family(
        EVENT_FAMILY_PROJECTED_SINGLE_CENTER_SELF_CURRENT,
        direct_sigma_by_transport_role_mS_cm,
        (
            "projected_single_center_self_current:"
            "feature_keyed_transport_center:role=free_ion_center:"
            "center=0:role=anion:z=-1"
        ),
        shaped_anion_center,
        solvent_environment,
    )

    assert obstruction_lever == PRIMITIVE_RESIDUAL_PRODUCTION_LEVER_LOCAL_OBSTRUCTION
    assert shape_lever == PRIMITIVE_RESIDUAL_PRODUCTION_LEVER_ANION_SHAPE_FRICTION


def test_primitive_fit_evaluation_uses_descriptor_calibration_targets():
    evaluation_field_names = tuple(
        evaluation_field.name
        for evaluation_field in fields(PrimitiveFitDatasetEvaluation)
    )

    assert "descriptor_calibration_targets" in evaluation_field_names
    assert "empirical_sigmas_mS_cm" not in evaluation_field_names
    assert "empirical_sigma_spreads_mS_cm" not in evaluation_field_names


def test_all_parameter_fields_consumed():
    audit_options = default_molecular_property_db_audit_options()
    fit_options, parameter_transforms = default_molecular_primitive_fit_configuration()
    registry_source = MolecularPropertyDbRegistrySource(
        solvent_registry=SOLVENTS,
        salt_registry=SALTS,
        additive_registry=ADDITIVES,
        cation_registry=CATION_PROPERTIES,
    )
    case_selection = build_molecular_property_db_case_selection(
        tuple(DATA),
        registry_source,
        audit_options,
    )
    evaluator = MolecularPropertyDbPrimitiveEvaluator(
        case_selection.cases,
        audit_options,
        fit_options,
        tuple(),
    )
    evaluation = evaluator.evaluate(configured_conductivity_primitive_parameters())

    assert set(evaluation.consumed_parameter_fields) == {
        parameter_transform.name
        for parameter_transform in parameter_transforms
    }


def test_prediction_sensitivity_freezes_collinear_and_zero_response_parameters():
    fit_options, parameter_transforms = default_molecular_primitive_fit_configuration()
    transform_by_name = {
        parameter_transform.name: parameter_transform
        for parameter_transform in parameter_transforms
    }
    fitted_parameter_names = (
        "coulomb_scale",
        "desolvation_scale",
        "contact_pair_logK_offset",
    )
    evaluator = _SyntheticCollinearPrimitiveEvaluator(fitted_parameter_names)

    diagnostics = primitive_prediction_sensitivity_diagnostics(
        _identity_primitive_parameters(),
        tuple(transform_by_name[name] for name in fitted_parameter_names),
        evaluator,
        fit_options,
    )

    assert diagnostics.rank == 1
    assert len(diagnostics.identifiable_parameter_names) == 1
    assert diagnostics.identifiable_parameter_names[0] in {
        "coulomb_scale",
        "desolvation_scale",
    }
    assert "contact_pair_logK_offset" in diagnostics.zero_sensitivity_parameter_names
    assert "contact_pair_logK_offset" in diagnostics.frozen_parameter_names
    assert {
        "coulomb_scale",
        "desolvation_scale",
    }.intersection(diagnostics.frozen_parameter_names)


def test_solvent_separated_pair_negative_offset_suppresses_neutral_ssip():
    baseline_result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _identity_primitive_parameters(),
    )
    suppressed_result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            solvent_separated_pair_logK_offset=NEGATIVE_LOGK_OFFSET,
        ),
    )

    assert _cluster_concentration_by_kind(
        suppressed_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    ) < _cluster_concentration_by_kind(
        baseline_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )


def test_contact_pair_and_ssip_are_distinct_templates():
    result = _compute_result(PAIR_CLUSTER_ION_COUNT, _identity_primitive_parameters())
    contact_pair = _cluster_state_by_kind(result, CONTACT_PAIR_CLUSTER_KIND)
    solvent_separated_pair = _cluster_state_by_kind(
        result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )

    assert contact_pair.label != solvent_separated_pair.label
    assert contact_pair.cluster_kind == CONTACT_PAIR_CLUSTER_KIND
    assert solvent_separated_pair.cluster_kind == SOLVENT_SEPARATED_PAIR_CLUSTER_KIND
    assert _cluster_extent_A(solvent_separated_pair) > _cluster_extent_A(contact_pair)
    assert solvent_separated_pair.desolvation_J_mol < contact_pair.desolvation_J_mol
    assert math.isfinite(contact_pair.activity_reference_J_mol)
    assert math.isfinite(solvent_separated_pair.activity_reference_J_mol)


def test_contact_pair_desolvation_offset_lowers_contact_pair_free_energy():
    baseline_result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _identity_primitive_parameters(),
    )
    shifted_result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            contact_pair_desolvation_offset_over_RT=-20.0,
        ),
    )
    baseline_contact_pair = _cluster_state_by_kind(
        baseline_result,
        CONTACT_PAIR_CLUSTER_KIND,
    )
    shifted_contact_pair = _cluster_state_by_kind(
        shifted_result,
        CONTACT_PAIR_CLUSTER_KIND,
    )

    assert _is_strictly_greater(
        baseline_contact_pair.desolvation_J_mol,
        shifted_contact_pair.desolvation_J_mol,
    )
    assert _is_strictly_greater(
        baseline_contact_pair.standard_free_energy_J_mol,
        shifted_contact_pair.standard_free_energy_J_mol,
    )
    assert _is_strictly_greater(
        _cluster_concentration_by_kind(
            shifted_result,
            CONTACT_PAIR_CLUSTER_KIND,
        ),
        _cluster_concentration_by_kind(
            baseline_result,
            CONTACT_PAIR_CLUSTER_KIND,
        ),
    )


def test_ssip_pair_generates_projected_multi_center_transport_state():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
        ),
    )
    solvent_separated_pair = _cluster_state_by_kind(
        result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )
    ssip_centers = _transport_centers_for_parent_and_role(
        result,
        solvent_separated_pair.label,
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    )
    ssip_projected_states = _projected_states_for_basin(
        result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )
    diagnostic_ssip_concentration_mol_m3 = result.speciation.cluster_concentrations_mol_m3[
        solvent_separated_pair.label
    ]

    assert diagnostic_ssip_concentration_mol_m3 > 0.0
    assert len(ssip_centers) == 1
    assert len(ssip_projected_states) == 1
    assert sorted(
        center.charge_number
        for center in ssip_projected_states[0].charged_centers
    ) == [-1, 1]
    assert ssip_centers[0].parent_cluster_kind == SOLVENT_SEPARATED_PAIR_CLUSTER_KIND
    assert ssip_centers[0].concentration_mol_m3 == pytest.approx(
        ssip_projected_states[0].concentration_mol_m3
    )
    assert ssip_projected_states[0].concentration_mol_m3 > 0.0


def test_internal_polarization_projection_splits_ssip_center_population():
    no_projection_result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
            internal_polarization_projection_offset=-20.0,
        ),
    )
    projected_result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
            internal_polarization_projection_offset=0.0,
            internal_polarization_projection_ionic_strength_slope=1.0,
            internal_polarization_projection_counterion_crowding_slope=1.0,
        ),
    )

    assert _is_strictly_greater(
        _transport_role_concentration_mol_m3(
            projected_result,
            TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
        ),
        _transport_role_concentration_mol_m3(
            no_projection_result,
            TRANSPORT_ROLE_INTERNAL_POLARIZATION_CENTER,
        ),
    )
    assert _cluster_concentration_by_kind(
        projected_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    ) == pytest.approx(
        _cluster_concentration_by_kind(
            no_projection_result,
            SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
        )
    )
    assert _is_strictly_greater(
        _cluster_concentration_by_kind(
            projected_result,
            SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
        ),
        _transport_role_concentration_mol_m3(
            projected_result,
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
        ) / 2.0,
    )


def test_trajectory_contact_pair_initializer_solves_reachability_response():
    audit_options = default_molecular_property_db_audit_options()
    fit_options, parameter_transforms = default_molecular_primitive_fit_configuration()
    trajectory_targets = load_trajectory_primitive_calibration_targets(
        fit_options.trajectory_primitive_target_paths,
    )
    baseline_parameters = configured_conductivity_primitive_parameters()
    initialized_parameters = (
        initialize_topology_logk_offsets_from_trajectory_concentrations(
            baseline_parameters,
            trajectory_targets,
            audit_options,
            parameter_transforms,
            fit_options,
        )
    )
    baseline_coverage = trajectory_concentration_target_coverage(
        trajectory_targets,
        baseline_parameters,
        audit_options,
    )[0]
    initialized_coverage = trajectory_concentration_target_coverage(
        trajectory_targets,
        initialized_parameters,
        audit_options,
    )[0]
    initialized_mapping = conductivity_primitive_parameters_to_mapping(
        initialized_parameters,
    )

    assert "contact_pair_center:Li+" in baseline_coverage.under_floor_target_labels
    assert "contact_pair_center:PF6-" in baseline_coverage.under_floor_target_labels
    assert not initialized_coverage.under_floor_target_labels
    assert _is_strictly_greater(
        0.0,
        initialized_mapping["contact_pair_desolvation_offset_over_RT"],
    )
    for contact_pair_label in (
        "contact_pair_center:Li+",
        "contact_pair_center:PF6-",
    ):
        target_concentration_mol_m3, predicted_concentration_mol_m3 = (
            _trajectory_coverage_target_and_prediction(
                initialized_coverage,
                contact_pair_label,
            )
        )
        assert _is_strictly_greater(
            predicted_concentration_mol_m3,
            0.5 * target_concentration_mol_m3,
        )
        assert _is_strictly_greater(
            2.0 * target_concentration_mol_m3,
            predicted_concentration_mol_m3,
        )


def test_neutral_contact_pair_has_zero_com_charge_transport():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            contact_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
        ),
    )
    contact_pair = _cluster_state_by_kind(result, CONTACT_PAIR_CLUSTER_KIND)
    contact_com_centers = _transport_centers_for_parent_and_role(
        result,
        contact_pair.label,
        TRANSPORT_ROLE_CONTACT_PAIR_CENTER,
    )
    contact_member_centers = _transport_centers_for_parent_and_role(
        result,
        contact_pair.label,
        TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER,
    )
    contact_com_center = contact_com_centers[0]
    nonzero_contact_com_events = tuple(
        event
        for event in result.events
        if contact_com_center.label in event.label
        and not _zero_displacement(event.charge_displacement_m)
    )
    member_event_labels = tuple(
        event.label
        for event in result.events
        for contact_member_center in contact_member_centers
        if contact_member_center.label in event.label
    )

    assert len(contact_com_centers) == 1
    assert contact_com_center.center_charge_number == 0
    assert nonzero_contact_com_events == tuple()
    assert contact_member_centers == tuple()
    projected_contact_states = _projected_states_for_basin(
        result,
        CONTACT_PAIR_CLUSTER_KIND,
    )
    assert len(projected_contact_states) == 1
    assert sorted(
        charged_center.charge_number
        for charged_center in projected_contact_states[0].charged_centers
    ) == [-1, 1]
    assert member_event_labels == tuple()
    assert _onsager_ne_sigma_for_transport_role(
        result,
        TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER,
    ) == pytest.approx(0.0)


def test_associated_exchange_flux_is_pure_motif_exchange_without_dc_transport():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            contact_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
        ),
    )
    exchange_events = _events_for_family(
        result,
        EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE,
    )

    assert exchange_events
    assert all(
        event.charge_displacement_m == pytest.approx((0.0, 0.0, 0.0))
        for event in exchange_events
    )
    assert all(
        np.allclose(
            np.asarray(
                event.charge_displacement_second_moment_m2,
                dtype=float,
            ),
            np.zeros((3, 3)),
        )
        for event in exchange_events
    )


def test_associated_exchange_flux_satisfies_detailed_balance():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            contact_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
        ),
    )
    exchange_events = _events_for_family(
        result,
        EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE,
    )
    event_by_transition = {
        (event.from_state_index, event.to_state_index): event
        for event in exchange_events
    }

    assert exchange_events
    for event in exchange_events:
        reverse_event = event_by_transition[(
            event.to_state_index,
            event.from_state_index,
        )]
        forward_flux_mol_m3_s = (
            result.markov_state_concentrations_mol_m3[event.from_state_index]
            * event.rate_s_inv
        )
        reverse_flux_mol_m3_s = (
            result.markov_state_concentrations_mol_m3[reverse_event.from_state_index]
            * reverse_event.rate_s_inv
        )
        assert forward_flux_mol_m3_s == pytest.approx(reverse_flux_mol_m3_s)


def test_associated_exchange_moments_are_valid_reversible_moments():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            contact_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
        ),
    )
    exchange_events = _events_for_family(
        result,
        EVENT_FAMILY_ASSOCIATED_STATE_EXCHANGE,
    )
    event_by_transition = {
        (event.from_state_index, event.to_state_index): event
        for event in exchange_events
    }

    assert exchange_events
    for event in exchange_events:
        reverse_event = event_by_transition[(
            event.to_state_index,
            event.from_state_index,
        )]
        displacement_vector = event.charge_displacement_m
        reverse_displacement_vector = reverse_event.charge_displacement_m
        second_moment_matrix = np.asarray(
            event.charge_displacement_second_moment_m2,
            dtype=float,
        )
        covariance_matrix = second_moment_matrix - np.outer(
            displacement_vector,
            displacement_vector,
        )

        assert reverse_displacement_vector == pytest.approx(
            tuple(-component for component in displacement_vector)
        )
        assert np.allclose(
            np.asarray(event.charge_displacement_second_moment_m2, dtype=float),
            np.asarray(
                reverse_event.charge_displacement_second_moment_m2,
                dtype=float,
            ),
        )
        assert np.linalg.eigvalsh(covariance_matrix).min() >= -1.0e-30


def test_ssip_center_translation_contributes_nonzero_direct_sigma():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
        ),
    )
    ssip_direct_sigma_mS_cm = _direct_sigma_for_transport_role(
        result,
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    )

    assert ssip_direct_sigma_mS_cm > 0.0


def test_ssip_center_transport_budget_not_double_counted():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
        ),
    )
    solvent_separated_pair = _cluster_state_by_kind(
        result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )
    ssip_centers = _transport_centers_for_parent_and_role(
        result,
        solvent_separated_pair.label,
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    )

    assert not _events_for_transport_role_and_family(
        result,
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
        "ordinary_mobile_translation",
    )
    obsolete_family_labels = (
        "solvent_separated_pair_relative_translation",
        "solvent_separated_pair_com_translation",
        "solvent_separated_pair_residual_center_translation",
    )
    assert all(
        event.family_label not in obsolete_family_labels
        for event in result.events
    )
    assert ssip_centers
    assert _events_for_family(
        result,
        "projected_solvent_separated_pair_self_current",
    )
    assert _direct_sigma_for_transport_role(
        result,
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    ) > 0.0


def test_projected_ssip_labels_are_feature_keyed_and_include_ligand_coordinate():
    result = compute_molecular_electrolyte_conductivity(
        _recipe_with_coordinating_additive(BASE_CONCENTRATION_M),
        _species_inputs_with_coordinating_additive(),
        ProvidedPropertyDescriptorBackend(),
        _options(
            PAIR_CLUSTER_ION_COUNT,
            _primitive_parameters_with_replacements(
                solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
            ),
        ),
    )
    projected_ssip_events = _events_for_family(
        result,
        "projected_solvent_separated_pair_self_current",
    )

    assert projected_ssip_events
    assert result.solvent_environment.additive_ligand_site_occupancy > 0.0
    for event in projected_ssip_events:
        assert "feature_keyed:" in event.label
        assert "ligand_bound=" in event.label
        assert "test_cation" not in event.label
        assert "test_anion" not in event.label
        assert "test_additive" not in event.label
    for state_label in result.markov_state_labels:
        assert "test_cation" not in state_label
        assert "test_anion" not in state_label
        assert "test_additive" not in state_label
    assert not any(
        transport_state.transport_role == TRANSPORT_ROLE_NEUTRAL_CENTER
        and transport_state.center_species_name == "test_additive"
        for transport_state in result.transport_states
    )
    analytical_model.assert_no_species_names_in_transport_labels(
        result.projected_transport_model.projected_transport_states,
        result.events,
        tuple(),
    )


def test_transport_label_validator_rejects_species_name_after_descriptor_lookup():
    synthetic_concentration_mol_m3 = BASE_CONCENTRATION_M * 1000.0
    synthetic_diffusion_m2_s = 1.0e-10
    projected_state = ProjectedTransportState(
        label="feature_keyed:single_center:mobile",
        concentration_mol_m3=synthetic_concentration_mol_m3,
        charged_centers=(
            ProjectedChargedCenter(
                label="positive_center:z=1",
                charge_number=1,
                diffusion_m2_s=synthetic_diffusion_m2_s,
            ),
        ),
        constraint_modes=tuple(),
        atmosphere_resistance_matrix_kg_s=tuple(),
        mobility_covariance_matrix_m2_s=((synthetic_diffusion_m2_s,),),
        ligand_shell_features={"coordination_site_count": 0.0},
        pair_basin=TRANSPORT_ROLE_FREE_ION_CENTER,
        residence_time_s=math.inf,
        partner_switch_time_s=math.inf,
        parent_cluster_label="diagnostic_species_cluster",
        parent_cluster_kind=TRANSPORT_ROLE_FREE_ION_CENTER,
        center_species_name="lookup_species_token",
        center_charge_number=1,
        center_index=0,
        hydrodynamic_radius_A=1.0,
        charge_cloud_radius_A=1.0,
        molecular_volume_A3=10.0,
        ligand_field_asymmetry=1.0,
        diffusion_m2_s=synthetic_diffusion_m2_s,
        local_obstruction_factor=1.0,
        local_obstruction_diffusion_scale=1.0,
        transport_role=TRANSPORT_ROLE_FREE_ION_CENTER,
    )
    species_name_leaking_event = MarkovAdditiveEvent(
        from_state_index=0,
        to_state_index=0,
        rate_s_inv=1.0,
        charge_displacement_m=(0.0, 0.0, 0.0),
        charge_displacement_second_moment_m2=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        label="feature_keyed:lookup_species_token:self_current",
        family_label="projected_single_center_self_current",
    )

    with pytest.raises(ValueError, match="species-name token"):
        analytical_model.assert_no_species_names_in_transport_labels(
            (projected_state,),
            (species_name_leaking_event,),
            tuple(),
        )


def test_coordinating_additive_creates_ligand_shell_population_primitive():
    result = compute_molecular_electrolyte_conductivity(
        _recipe_with_coordinating_additive(BASE_CONCENTRATION_M),
        _species_inputs_with_coordinating_additive(),
        ProvidedPropertyDescriptorBackend(),
        _options(
            PAIR_CLUSTER_ION_COUNT,
            _primitive_parameters_with_replacements(
                solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
            ),
        ),
    )
    ligand_shell_concentration_mol_m3 = math.fsum(
        result.speciation.cation_ligand_concentrations_mol_m3.values()
    )
    free_cation_transport_concentration_mol_m3 = math.fsum(
        transport_center.concentration_mol_m3
        for transport_center in result.transport_states
        if transport_center.transport_role == TRANSPORT_ROLE_FREE_ION_CENTER
        and transport_center.center_charge_number > 0
    )
    ligand_shell_transport_concentration_mol_m3 = (
        _transport_role_concentration_mol_m3(
            result,
            TRANSPORT_ROLE_LIGAND_SHELL_CENTER,
        )
    )
    projected_ligand_shell_concentration_mol_m3 = math.fsum(
        projected_state.concentration_mol_m3
        for projected_state in result.projected_transport_model.projected_transport_states
        if projected_state.transport_role == TRANSPORT_ROLE_LIGAND_SHELL_CENTER
    )
    projected_free_cation_concentration_mol_m3 = math.fsum(
        projected_state.concentration_mol_m3
        for projected_state in result.projected_transport_model.projected_transport_states
        if projected_state.transport_role == TRANSPORT_ROLE_FREE_ION_CENTER
        and projected_state.center_charge_number > 0
    )

    assert ligand_shell_concentration_mol_m3 > 0.0
    assert free_cation_transport_concentration_mol_m3 == pytest.approx(
        projected_free_cation_concentration_mol_m3
    )
    assert ligand_shell_transport_concentration_mol_m3 == pytest.approx(
        projected_ligand_shell_concentration_mol_m3
    )
    assert projected_ligand_shell_concentration_mol_m3 > 0.0
    assert result.speciation.neutral_ligand_site_concentrations_mol_m3
    assert all(
        "test_additive" not in motif_label
        and "test_cation" not in motif_label
        for motif_label in result.speciation.cation_ligand_concentrations_mol_m3
    )


def test_coordinating_additive_solves_ternary_ssip_population_primitive():
    result = compute_molecular_electrolyte_conductivity(
        _recipe_with_coordinating_additive(BASE_CONCENTRATION_M),
        _species_inputs_with_coordinating_additive_and_bulky_anion(),
        ProvidedPropertyDescriptorBackend(),
        _options(
            PAIR_CLUSTER_ION_COUNT,
            _primitive_parameters_with_replacements(
                solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
            ),
        ),
    )
    solvent_separated_pair = _cluster_state_by_kind(
        result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )
    ordinary_projected_pairs = tuple(
        projected_state
        for projected_state in result.projected_transport_model.projected_transport_states
        if projected_state.pair_basin == SOLVENT_SEPARATED_PAIR_CLUSTER_KIND
    )
    additive_projected_pairs = tuple(
        projected_state
        for projected_state in result.projected_transport_model.projected_transport_states
        if projected_state.pair_basin
        == ADDITIVE_SEPARATED_PAIR_CLUSTER_KIND
    )
    ligand_bound_ssip_concentration_mol_m3 = (
        result.speciation.cluster_ligand_bound_concentrations_mol_m3[
            solvent_separated_pair.label
        ]
    )
    diagnostic_ssip_cluster_concentration_mol_m3 = (
        result.speciation.cluster_concentrations_mol_m3[
            solvent_separated_pair.label
        ]
    )
    ssip_centers = _transport_centers_for_parent_and_role(
        result,
        solvent_separated_pair.label,
        TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER,
    )
    projected_pair_concentration_mol_m3 = math.fsum(
        projected_state.concentration_mol_m3
        for projected_state in ordinary_projected_pairs + additive_projected_pairs
    )
    ordinary_projected_pair_concentration_mol_m3 = math.fsum(
        projected_state.concentration_mol_m3
        for projected_state in ordinary_projected_pairs
    )
    additive_projected_pair_concentration_mol_m3 = math.fsum(
        projected_state.concentration_mol_m3
        for projected_state in additive_projected_pairs
    )

    assert result.speciation.cation_ligand_anion_concentrations_mol_m3
    assert result.speciation.cation_ligand_anion_parent_cluster_by_label
    assert ligand_bound_ssip_concentration_mol_m3 > 0.0
    assert diagnostic_ssip_cluster_concentration_mol_m3 > 0.0
    assert ordinary_projected_pairs
    assert additive_projected_pairs
    assert ssip_centers
    assert additive_projected_pair_concentration_mol_m3 > 0.0
    assert ordinary_projected_pair_concentration_mol_m3 > 0.0
    assert projected_pair_concentration_mol_m3 == pytest.approx(
        math.fsum(
            transport_center.concentration_mol_m3
            for transport_center in ssip_centers
        )
    )
    assert set(
        result.speciation.cation_ligand_anion_parent_cluster_by_label.values()
    ) == {solvent_separated_pair.label}
    assert all(
        "test_additive" not in motif_label
        and "test_cation" not in motif_label
        and "test_anion" not in motif_label
        for motif_label in result.speciation.cation_ligand_anion_concentrations_mol_m3
    )


def test_coordinating_additive_bulky_anion_ssip_derives_signed_covariance():
    primitive_parameters = _primitive_parameters_with_replacements(
        solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
    )
    bulky_anion_result = compute_molecular_electrolyte_conductivity(
        _recipe(BASE_CONCENTRATION_M),
        _species_inputs_with_bulky_anion(),
        ProvidedPropertyDescriptorBackend(),
        _options(PAIR_CLUSTER_ION_COUNT, primitive_parameters),
    )
    additive_result = compute_molecular_electrolyte_conductivity(
        _recipe_with_coordinating_additive(BASE_CONCENTRATION_M),
        _species_inputs_with_coordinating_additive_and_bulky_anion(),
        ProvidedPropertyDescriptorBackend(),
        _options(PAIR_CLUSTER_ION_COUNT, primitive_parameters),
    )
    bulky_anion_moment_m2 = _projected_ssip_second_moment_trace_m2(
        bulky_anion_result
    )
    additive_moment_m2 = _projected_ssip_second_moment_trace_m2(additive_result)
    additive_event_labels = tuple(
        event.label
        for event in _events_for_family(
            additive_result,
            "projected_solvent_separated_pair_self_current",
        )
    )

    assert additive_result.solvent_environment.additive_ligand_site_occupancy > 0.0
    assert any(
        "additive_separated_solvent_separated_pair" in event_label
        for event_label in additive_event_labels
    )
    assert _is_strictly_greater(additive_moment_m2, bulky_anion_moment_m2)


def test_onsager_cutover_removes_all_ordinary_translation_events():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(),
    )
    assert all(
        event.family_label != "ordinary_mobile_translation"
        for event in result.events
    )


def test_ssip_transport_does_not_require_suppressing_ssip_logK():
    suppressed_result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            solvent_separated_pair_logK_offset=NEGATIVE_LOGK_OFFSET,
        ),
    )
    favored_result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            solvent_separated_pair_logK_offset=POSITIVE_SSIP_LOGK_OFFSET,
        ),
    )

    assert _cluster_concentration_by_kind(
        favored_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    ) > _cluster_concentration_by_kind(
        suppressed_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )
    assert _projected_basin_concentration_mol_m3(
        favored_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    ) > _projected_basin_concentration_mol_m3(
        suppressed_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )
    assert favored_result.sigma_mS_cm > 0.0


def test_positive_and_negative_charged_triplet_offsets_are_separate():
    baseline_result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _identity_primitive_parameters(),
    )
    positive_shifted_result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            positive_charged_triplet_logK_offset=POSITIVE_TRIPLET_LOGK_OFFSET,
        ),
    )
    negative_shifted_result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            negative_charged_triplet_logK_offset=NEGATIVE_TRIPLET_LOGK_OFFSET,
        ),
    )

    assert _cluster_concentration_by_kind(
        positive_shifted_result,
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    ) > _cluster_concentration_by_kind(
        baseline_result,
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    )
    assert _cluster_concentration_by_kind(
        negative_shifted_result,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    ) > _cluster_concentration_by_kind(
        baseline_result,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    )
    assert _cluster_state_by_kind(
        positive_shifted_result,
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    ).standard_free_energy_J_mol < _cluster_state_by_kind(
        baseline_result,
        POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    ).standard_free_energy_J_mol
    assert _cluster_state_by_kind(
        negative_shifted_result,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    ).standard_free_energy_J_mol < _cluster_state_by_kind(
        baseline_result,
        NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
    ).standard_free_energy_J_mol


def test_activity_coefficients_affect_mass_balance_solution():
    weak_activity_result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            activity_debye_scale=0.01,
            activity_hard_sphere_scale=0.01,
        ),
    )
    strong_activity_result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            activity_debye_scale=4.0,
            activity_hard_sphere_scale=4.0,
        ),
    )

    assert not math.isclose(
        _total_cluster_concentration_mol_m3(weak_activity_result),
        _total_cluster_concentration_mol_m3(strong_activity_result),
        rel_tol=1.0e-8,
        abs_tol=1.0e-12,
    )


def test_production_options_do_not_accept_label_specific_cluster_shifts():
    option_field_names = {
        option_field.name for option_field in fields(MolecularMoriOptions)
    }

    assert "cluster_standard_free_energy_shift_over_RT_by_label" not in option_field_names
    with pytest.raises(TypeError):
        MolecularMoriOptions(
            max_cluster_ion_count=PAIR_CLUSTER_ION_COUNT,
            max_packing_fraction=MAX_PACKING_FRACTION,
            free_volume_exponent=0.0,
            translation_jump_length_multiplier=JUMP_LENGTH_MULTIPLIER,
            primitive_parameters=_identity_primitive_parameters(),
            cluster_standard_free_energy_shift_over_RT_by_label={},
        )


def test_diagnostic_cluster_shift_is_separate_from_production_options():
    primitive_parameters = _identity_primitive_parameters()
    production_result = _compute_result(PAIR_CLUSTER_ION_COUNT, primitive_parameters)
    solvent_separated_pair = _cluster_state_by_kind(
        production_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )
    diagnostic_result = compute_molecular_electrolyte_conductivity_with_diagnostic_cluster_shifts(
        _recipe(BASE_CONCENTRATION_M),
        _species_inputs(),
        ProvidedPropertyDescriptorBackend(),
        _options(PAIR_CLUSTER_ION_COUNT, primitive_parameters),
        {solvent_separated_pair.label: abs(NEGATIVE_LOGK_OFFSET)},
    )

    assert _cluster_concentration_by_kind(
        diagnostic_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    ) < _cluster_concentration_by_kind(
        production_result,
        SOLVENT_SEPARATED_PAIR_CLUSTER_KIND,
    )


def test_charged_triplet_activation_changes_charged_triplet_direct_sigma():
    baseline_result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _identity_primitive_parameters(),
    )
    charged_triplet_result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            positive_charged_triplet_logK_offset=POSITIVE_TRIPLET_LOGK_OFFSET,
            negative_charged_triplet_logK_offset=NEGATIVE_TRIPLET_LOGK_OFFSET,
        ),
    )

    baseline_direct_sigma_mS_cm = _onsager_ne_sigma_for_transport_role(
        baseline_result,
        TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
    )
    charged_triplet_direct_sigma_mS_cm = _onsager_ne_sigma_for_transport_role(
        charged_triplet_result,
        TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
    )

    assert _charged_cluster_concentration_mol_m3(charged_triplet_result) > (
        _charged_cluster_concentration_mol_m3(baseline_result)
    )
    assert charged_triplet_direct_sigma_mS_cm > baseline_direct_sigma_mS_cm


def test_charged_triplet_has_nonzero_transport_channel():
    result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            positive_charged_triplet_logK_offset=POSITIVE_TRIPLET_LOGK_OFFSET,
            negative_charged_triplet_logK_offset=NEGATIVE_TRIPLET_LOGK_OFFSET,
        ),
    )
    charged_triplet_centers = tuple(
        transport_center
        for transport_center in result.transport_states
        if transport_center.parent_cluster_kind
        in (
            POSITIVE_CHARGED_TRIPLET_CLUSTER_KIND,
            NEGATIVE_CHARGED_TRIPLET_CLUSTER_KIND,
        )
    )

    assert any(
        center.transport_role == TRANSPORT_ROLE_CLUSTER_COM_CENTER
        and center.center_charge_number != 0
        for center in charged_triplet_centers
    )
    assert any(
        center.transport_role == TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER
        and center.center_charge_number != 0
        for center in charged_triplet_centers
    )
    assert _onsager_ne_sigma_for_transport_role(
        result,
        TRANSPORT_ROLE_CHARGED_TRIPLET_CENTER,
    ) > 0.0


def test_molecular_result_exposes_projected_transport_model():
    result = _compute_result(
        TRIPLET_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(
            positive_charged_triplet_logK_offset=POSITIVE_TRIPLET_LOGK_OFFSET,
            negative_charged_triplet_logK_offset=NEGATIVE_TRIPLET_LOGK_OFFSET,
        ),
    )
    projected_model = result.projected_transport_model
    generator_model = projected_model.projected_generator_model

    assert result.proof_status == FULL_MICROSCOPIC_GENERATOR_DERIVED_PROOF_STATUS
    assert (
        projected_model.proof_status
        == FULL_MICROSCOPIC_GENERATOR_DERIVED_PROOF_STATUS
    )
    assert (
        generator_model
        .microscopic_generator_source
        .equilibrium_measure_source
        == "descriptor_derived_equilibrium_measure_mu_x_from_U_x"
    )
    assert (
        generator_model
        .microscopic_generator_source
        .generator_formula_id
        == "Q_ij=K_ij/c_i_with_detailed_balance"
    )
    assert generator_model.microscopic_generator_source.descriptor_complete
    assert (
        generator_model.charge_polarization_source.formula_id
        == "P=sum_a z_a e R_a"
    )
    assert tuple(
        partition_state.state_label
        for partition_state in projected_model.partition_states
    ) == result.markov_state_labels
    projected_state_by_label = {
        projected_transport_state.label: projected_transport_state
        for projected_transport_state in projected_model.projected_transport_states
    }
    expected_projected_state_labels = tuple(
        markov_state_label.removesuffix(":mobile")
        for markov_state_label in result.markov_state_labels
    )
    assert set(projected_state_by_label) == set(expected_projected_state_labels)
    assert len(projected_model.projected_transport_states) == len(
        result.markov_state_labels
    )
    for markov_state_label, markov_concentration_mol_m3 in zip(
        result.markov_state_labels,
        result.markov_state_concentrations_mol_m3,
    ):
        projected_state_label = markov_state_label.removesuffix(":mobile")
        projected_transport_state = projected_state_by_label[projected_state_label]
        assert math.isclose(
            projected_transport_state.concentration_mol_m3,
            markov_concentration_mol_m3,
        )
    multi_center_projected_states = tuple(
        projected_transport_state
        for projected_transport_state in projected_model.projected_transport_states
        if len(projected_transport_state.charged_centers) > 1
    )
    assert multi_center_projected_states
    assert all(
        len(projected_transport_state.mobility_covariance_matrix_m2_s)
        == len(projected_transport_state.charged_centers)
        for projected_transport_state in multi_center_projected_states
    )
    assert all(
        compute_projected_transport_state_charge_diffusivity_m2_s(
            projected_transport_state,
            result.solvent_environment.temperature_K,
        )
        >= 0.0
        for projected_transport_state in multi_center_projected_states
    )
    assert all(
        partition_state.population_source
        == "restricted_population_from_mu_x_over_A_i"
        for partition_state in projected_model.partition_states
    )
    assert all(
        partition_state.pmf_partition_model.state_label == partition_state.state_label
        for partition_state in projected_model.partition_states
    )
    assert (
        projected_model.closure_contract.readout_theorem
        == "finite_markov_additive_green_kubo_poisson_readout"
    )
    assert (
        projected_model
        .closure_contract
        .equilibrium_measure_source
        .source_name
        == "descriptor_derived_equilibrium_measure_mu_x_from_U_x"
    )
    assert (
        "coulomb_scale"
        in projected_model
        .closure_contract
        .equilibrium_measure_source
        .source_parameter_names
    )
    assert (
        projected_model.closure_contract.concentration_source.source_name
        == "restricted_population_c_i=C_mu_x_A_i"
    )
    assert (
        projected_model
        .closure_contract
        .reactive_flux_source
        .source_name
        == "descriptor_derived_symmetric_reactive_flux_K_ij_from_L_x_surface"
    )
    assert (
        projected_model
        .closure_contract
        .displacement_moment_source
        .source_name
        == "descriptor_derived_conditional_charge_displacement_moments_from_P_x"
    )
    rejected_source_names = {
        "descriptor_mass_action_restricted_partition_weights",
        "descriptor_closed_symmetric_reactive_fluxes",
        "descriptor_closed_charge_displacement_moments",
    }
    assert not rejected_source_names.intersection(
        {
            projected_model.closure_contract.concentration_source.source_name,
            projected_model.closure_contract.reactive_flux_source.source_name,
            projected_model.closure_contract.displacement_moment_source.source_name,
        }
    )
    assert (
        "cross_relaxation_scale"
        in projected_model
        .closure_contract
        .reactive_flux_source
        .source_parameter_names
    )
    assert (
        projected_model
        .closure_contract
        .descriptor_closure_derives_full_microscopic_generator
    )
    assert (
        projected_model
        .closure_contract
        .descriptor_closure_derives_finite_projected_generator
    )
    assert (
        set(
            projected_model
            .closure_contract
            .primitive_parameter_theorem_role_by_name
        )
        == set(CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES)
    )
    assert (
        projected_model
        .closure_contract
        .primitive_parameter_theorem_role_by_name[
            "internal_polarization_projection_offset"
        ]
        == PRIMITIVE_PARAMETER_ROLE_PARTITION_DEFINITION
    )
    assert (
        projected_model
        .closure_contract
        .primitive_parameter_theorem_role_by_name[
            "contact_pair_desolvation_offset_over_RT"
        ]
        == PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    )
    assert not (
        projected_model
        .closure_contract
        .equilibrium_measure_source
        .empirical_closure_parameter_names
    )
    assert not (
        projected_model
        .closure_contract
        .partition_source
        .empirical_closure_parameter_names
    )
    assert not (
        projected_model
        .closure_contract
        .reactive_flux_source
        .empirical_closure_parameter_names
    )
    assert not (
        projected_model
        .closure_contract
        .displacement_moment_source
        .empirical_closure_parameter_names
    )
    assert (
        CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME
        == projected_model
        .closure_contract
        .primitive_parameter_theorem_role_by_name
    )
    assert projected_model.free_energy_functional.terms
    assert (
        projected_model.free_energy_functional.partition_weight_formula_id
        == "restricted_weight_i=exp(-DeltaG_i/RT)/sum_k exp(-DeltaG_k/RT)"
    )
    assert {
        term.parameter_names[0]
        for term in projected_model.free_energy_functional.terms
    } == {
        parameter_name
        for parameter_name, theorem_role in (
            CONDUCTIVITY_PRIMITIVE_PARAMETER_THEOREM_ROLE_BY_NAME.items()
        )
        if theorem_role == PRIMITIVE_PARAMETER_ROLE_EQUILIBRIUM_FREE_ENERGY
    }
    assert projected_model.state_labels == result.markov_state_labels
    assert projected_model.stationary_concentrations_mol_m3 == (
        result.markov_state_concentrations_mol_m3
    )
    assert math.fsum(projected_model.stationary_probabilities) == pytest.approx(1.0)
    assert projected_model.restricted_partition_weights == (
        projected_model.stationary_probabilities
    )
    assert projected_model.conductivity_result.sigma_mS_cm == pytest.approx(
        result.sigma_mS_cm
    )
    assert (
        projected_model.reactive_flux_integrals
        or projected_model.self_displacement_moments
    )
    assert projected_model.mori_basis_functions
    assert generator_model.basis.basis_labels == tuple(
        basis_function.label
        for basis_function in projected_model.mori_basis_functions
    )
    assert generator_model.populations.state_labels == projected_model.state_labels
    assert generator_model.reactive_fluxes.reactive_fluxes == (
        projected_model.reactive_flux_integrals
    )
    assert generator_model.displacement_moments.displacement_moments == (
        projected_model.reactive_flux_integrals
    )
    assert generator_model.self_current_tensors.self_current_tensors == (
        projected_model.self_displacement_moments
    )
    assert generator_model.current_coupling.axis_count == 3
    assert generator_model.conductivity.sigma_mS_cm == pytest.approx(
        projected_model.conductivity_result.sigma_mS_cm
    )
    assert all(
        basis_function.source in SUPPORTED_MORI_BASIS_SOURCES
        for basis_function in projected_model.mori_basis_functions
    )
    contribution_classes = {
        classification.contribution_class
        for classification in projected_model.contribution_classifications
    }
    assert PROJECTED_GENERATOR_CLASS_RESTRICTED_POPULATION in contribution_classes
    assert PROJECTED_GENERATOR_CLASS_REACTIVE_FLUX in contribution_classes
    assert (
        PROJECTED_GENERATOR_CLASS_STATE_CHANGING_DISPLACEMENT
        in contribution_classes
    )
    assert PROJECTED_GENERATOR_CLASS_WITHIN_STATE_SELF_CURRENT in contribution_classes
    assert PROJECTED_GENERATOR_CLASS_MORI_MEMORY_BASIS in contribution_classes
    assert PROJECTED_GENERATOR_CLASS_DIAGNOSTIC_ONLY not in contribution_classes
    assert all(
        classification.conductivity_contribution_allowed
        for classification in projected_model.contribution_classifications
    )
    assert all(
        0.0 < reactive_flux.symmetric_flux_mol_m3_s
        for reactive_flux in projected_model.reactive_flux_integrals
    )
    assert all(
        reactive_flux.reactive_flux_derivation.symmetric_flux_mol_m3_s
        == pytest.approx(reactive_flux.symmetric_flux_mol_m3_s)
        for reactive_flux in projected_model.reactive_flux_integrals
    )
    assert all(
        reactive_flux.derived_reactive_flux_model.symmetric_flux_mol_m3_s
        == pytest.approx(reactive_flux.symmetric_flux_mol_m3_s)
        for reactive_flux in projected_model.reactive_flux_integrals
    )
    assert all(
        reactive_flux.derived_reactive_flux_model.from_partition
        == reactive_flux.from_state_label
        for reactive_flux in projected_model.reactive_flux_integrals
    )
    assert all(
        reactive_flux.derived_reactive_flux_model.to_partition
        == reactive_flux.to_state_label
        for reactive_flux in projected_model.reactive_flux_integrals
    )
    assert all(
        reactive_flux.displacement_moment_derivation.charge_displacement_m
        == reactive_flux.charge_displacement_m
        for reactive_flux in projected_model.reactive_flux_integrals
    )
    assert all(
        reactive_flux.conditional_displacement_model.mean_displacement_m
        == reactive_flux.charge_displacement_m
        for reactive_flux in projected_model.reactive_flux_integrals
    )
    assert all(
        not reactive_flux.detailed_balance_residual_mol_m3_s < 0.0
        for reactive_flux in projected_model.reactive_flux_integrals
    )
    assert all(
        0.0 <= axis_density
        for self_moment in projected_model.self_displacement_moments
        for axis_density in self_moment.direct_axis_density_m2_s_mol_m3
    )
    assert all(
        partition_state.partition_derivation.state_label
        == partition_state.state_label
        for partition_state in projected_model.partition_states
    )
    assert all(
        self_moment.displacement_moment_derivation.charge_displacement_m
        == self_moment.charge_displacement_m
        for self_moment in projected_model.self_displacement_moments
    )
    assert all(
        self_moment.self_current_projection_model.partition
        == self_moment.state_label
        for self_moment in projected_model.self_displacement_moments
    )
    assert all(
        self_moment.conditional_displacement_model.mean_displacement_m
        == self_moment.charge_displacement_m
        for self_moment in projected_model.self_displacement_moments
    )


def test_projected_transport_model_builder_uses_projected_state_boundary():
    projected_builder_parameters = tuple(
        inspect.signature(
            compute_projected_electrolyte_transport_model
        ).parameters
    )

    assert "projected_transport_states" in projected_builder_parameters
    assert "construction_transport_centers" not in projected_builder_parameters
    assert "cluster_templates" not in projected_builder_parameters
    assert "solvent_environment" not in projected_builder_parameters


def test_markov_builder_uses_projected_transport_state_boundary():
    markov_builder_parameters = tuple(
        inspect.signature(
            analytical_model._markov_process_from_projected_transport_states
        ).parameters
    )

    assert "projected_transport_states" in markov_builder_parameters
    assert "transport_states" not in markov_builder_parameters
    assert not hasattr(analytical_model, "_markov_process_from_transport_states")
    assert not hasattr(analytical_model, "_ProjectedMotifTransportPrimitive")


def test_property_db_core_uses_projected_primitive_generator_boundary():
    production_source = inspect.getsource(
        analytical_model._compute_molecular_electrolyte_conductivity
    )

    for required_call in (
        "build_analytic_recipe_microscopic_generator",
        "build_recipe_projection_basis",
        "project_analytic_generator_to_primitives",
        "compute_projected_gk_mori_conductivity",
    ):
        assert required_call in production_source
    for removed_call in (
        "_solve_projected_speciation_mass_balance",
        "_projected_kinetic_states_from_speciation",
        "_apply_ion_atmosphere_to_projected_kinetic_states",
        "_projected_transport_states_from_kinetic_states",
        "_projected_transport_states_from_speciation",
        "_projected_motif_transport_primitives_from_projected_transport_states",
        "_markov_process_from_projected_motif_primitives",
        "_atmosphere_mori_corrections",
        "_projected_primitive_set_from_transport_projection",
        "_molecular_transport_states",
        "_apply_ion_atmosphere_to_transport_states",
        "_projected_transport_states_from_transport_centers",
        "_projected_motif_transport_primitives_from_transport_centers",
        "_speciation_with_neutral_ligand_motifs(",
    ):
        assert removed_call not in production_source


def test_recipe_projection_payload_exports_projected_states_directly():
    generator_field_names = {
        generator_field.name
        for generator_field in fields(analytical_model.AnalyticRecipeGenerator)
    }
    projection_field_names = {
        projection_field.name
        for projection_field in fields(analytical_model.RecipePrimitiveProjection)
    }

    assert "projected_transport_states" in generator_field_names
    assert "equilibrium_measure_mu_x" not in generator_field_names
    assert "reversible_generator_Lx" not in generator_field_names
    assert "mobility_tensor_Dx" not in generator_field_names
    assert "charge_polarization_Px" not in generator_field_names
    assert "analytic_pmf_energy_J_mol_by_state_label" not in generator_field_names
    assert "capacity_flux_matrix_K_ij_mol_m3_s" not in generator_field_names
    assert "transition_first_moments_d_ij_m" not in generator_field_names
    assert "mori_memory_energy_matrix_A" not in generator_field_names
    assert "reduced_basin_coordinate_models" in generator_field_names
    assert "memory_coordinate_models" in generator_field_names
    assert "projected_kinetic_states" not in generator_field_names
    assert "projected_transport_states" in projection_field_names
    assert "motif_primitives" not in projection_field_names
    assert not hasattr(analytical_model, "_ProjectedMotifKineticState")
    assert not hasattr(analytical_model, "_ProjectedMotifConstructionState")
    for required_method in (
        "U",
        "grad_U",
        "mu_integral",
        "capacity_flux",
        "transition_path_moments",
        "self_current_tensor",
        "mori_A_h",
        "P",
    ):
        assert callable(getattr(analytical_model.AnalyticRecipeGenerator, required_method))
    assert not hasattr(
        analytical_model,
        "_projected_motif_transport_primitives_from_transport_centers",
    )
    assert not hasattr(
        analytical_model,
        "_projected_motif_transport_primitive_from_center",
    )
    assert not hasattr(
        analytical_model,
        "_projected_motif_transport_primitives_from_projected_transport_states",
    )
    assert not hasattr(
        analytical_model,
        "_projected_primitive_set_from_transport_projection",
    )


def test_recipe_generator_projection_boundary_owns_primitive_assembly():
    generator_source = inspect.getsource(
        analytical_model.build_analytic_recipe_microscopic_generator
    )
    projection_source = inspect.getsource(
        analytical_model.project_analytic_generator_to_primitives
    )
    markov_builder_source = inspect.getsource(
        analytical_model._markov_process_from_projected_transport_states
    )

    for required_call in (
        "_solve_projected_speciation_mass_balance",
        "_projected_transport_states_from_mass_balance",
        "_ion_atmosphere_diagnostics_for_projected_transport_states",
        "_projected_basis_transport_inventory_from_states",
        "_analytic_recipe_reduced_coordinate_models",
        "_analytic_restricted_populations_from_reduced_generator",
        "_memory_primitives_from_projected_transport_states",
        "_analytic_recipe_mori_memory_coordinate_models",
    ):
        assert required_call in generator_source
    for removed_call in (
        "_analytic_recipe_pmf_integral_parameters",
        "_analytic_restricted_populations_from_pmf",
        "_analytic_recipe_transition_path_primitives",
        "_analytic_recipe_mori_A_h_from_memory_primitives",
    ):
        assert removed_call not in generator_source
    assert "_reduced_pmf_integral_parameters_from_projected_states" not in (
        generator_source
    )
    assert "_markov_process_from_projected_transport_states" not in generator_source
    assert "recipe_generator.project_markov_process" not in projection_source
    assert "project_generator_to_primitives" in projection_source
    assert "_projected_flux_integrals_from_events" in projection_source
    assert projection_source.index(
        "project_generator_to_primitives"
    ) < projection_source.index("_projected_flux_integrals_from_events")
    assert not hasattr(
        analytical_model.AnalyticRecipeGenerator,
        "project_markov_process",
    )
    projector_source = inspect.getsource(
        analytical_model.project_generator_to_primitives
    ) + inspect.getsource(
        analytical_model._projected_primitive_set_from_executable_generator
    )
    for required_projector_call in (
        "generator.mu_integral",
        "generator.capacity_flux",
        "generator.transition_path_moments",
        "generator.self_current_tensor",
        "generator.mori_A_h",
    ):
        assert required_projector_call in projector_source
    assert tuple(
        inspect.signature(analytical_model.project_generator_to_primitives).parameters
    ) == ("generator", "projection_basis")
    assert tuple(
        inspect.signature(
            analytical_model.AnalyticRecipeGenerator.transition_path_moments
        ).parameters
    ) == ("self", "basin_i", "basin_j")
    assert tuple(
        inspect.signature(
            analytical_model.AnalyticRecipeGenerator.self_current_tensor
        ).parameters
    ) == ("self", "basin")
    assert tuple(
        inspect.signature(analytical_model.AnalyticRecipeGenerator.mori_A_h).parameters
    ) == ("self", "mori_basis_functions")
    assert "return _nonnegative_float(" not in inspect.getsource(
        analytical_model.AnalyticRecipeGenerator.mu_integral
    )
    generator_method_source = (
        inspect.getsource(analytical_model.AnalyticRecipeGenerator.U)
        + inspect.getsource(analytical_model.AnalyticRecipeGenerator.grad_U)
        + inspect.getsource(analytical_model.AnalyticRecipeGenerator.capacity_flux)
        + inspect.getsource(
            analytical_model.AnalyticRecipeGenerator.transition_path_moments
        )
        + inspect.getsource(analytical_model.AnalyticRecipeGenerator.mori_A_h)
    )
    for removed_lookup in (
        "analytic_pmf_energy_J_mol_by_state_label",
        "capacity_flux_matrix_K_ij_mol_m3_s",
        "transition_first_moments_d_ij_m",
        "mori_memory_energy_matrix_A",
        "mori_current_coupling_matrix_h",
    ):
        assert removed_lookup not in generator_method_source
    assert not hasattr(
        analytical_model,
        "_append_solvent_separated_pair_motif_states_and_events",
    )
    assert not hasattr(analytical_model, "_speciation_with_neutral_ligand_motifs")
    assert "_append_solvent_separated_pair_motif_states_and_events" not in (
        markov_builder_source
    )
    assert "_projected_transport_states_from_kinetic_states" not in generator_source
    assert "_projected_motif_transport_primitives" not in projection_source
    assert "_atmosphere_mori_corrections(" not in projection_source
    assert "_atmosphere_mori_corrections(" not in markov_builder_source
    assert (
        "_projected_current_memory_corrections(" in markov_builder_source
    )


def test_pair_covariance_is_state_persistence_not_occupancy_shape_gate():
    covariance_source = inspect.getsource(
        analytical_model._projected_pair_signed_current_covariance_fraction
    )

    assert "additive_ligand_site_occupancy" not in covariance_source
    assert "anion_structure_fraction" not in covariance_source
    assert "state_persistence_fraction" in covariance_source
    assert "center_geometry_fraction" in covariance_source


def test_atmosphere_memory_capacity_is_bounded_by_projected_state_diffusivity():
    center = MolecularTransportCenter(
        label="projected_low_mobility_center",
        parent_cluster_label="free:positive",
        parent_cluster_kind=TRANSPORT_ROLE_FREE_ION_CENTER,
        concentration_mol_m3=1000.0,
        center_species_name="generic_positive",
        center_charge_number=1,
        center_index=0,
        hydrodynamic_radius_A=1.0,
        charge_cloud_radius_A=1.0,
        molecular_volume_A3=10.0,
        ligand_field_asymmetry=1.0,
        diffusion_m2_s=1.0e-10,
        local_obstruction_factor=1.0,
        local_obstruction_diffusion_scale=1.0,
        transport_role=TRANSPORT_ROLE_FREE_ION_CENTER,
    )
    projected_state = ProjectedTransportState(
        label="projected_low_mobility_center",
        concentration_mol_m3=1000.0,
        charged_centers=(
            ProjectedChargedCenter(
                label="positive_center:z=1",
                charge_number=1,
                diffusion_m2_s=1.0e-10,
            ),
        ),
        constraint_modes=tuple(),
        atmosphere_resistance_matrix_kg_s=tuple(),
        mobility_covariance_matrix_m2_s=((1.0e-12,),),
        ligand_shell_features={"temperature_K": T_REF_K},
        pair_basin=TRANSPORT_ROLE_FREE_ION_CENTER,
        residence_time_s=math.inf,
        partner_switch_time_s=math.inf,
        parent_cluster_label=center.parent_cluster_label,
        parent_cluster_kind=center.parent_cluster_kind,
        center_species_name=center.center_species_name,
        center_charge_number=center.center_charge_number,
        center_index=center.center_index,
        hydrodynamic_radius_A=center.hydrodynamic_radius_A,
        charge_cloud_radius_A=center.charge_cloud_radius_A,
        molecular_volume_A3=center.molecular_volume_A3,
        ligand_field_asymmetry=center.ligand_field_asymmetry,
        diffusion_m2_s=center.diffusion_m2_s,
        local_obstruction_factor=center.local_obstruction_factor,
        local_obstruction_diffusion_scale=center.local_obstruction_diffusion_scale,
        transport_role=center.transport_role,
    )
    memory_primitive = MolecularAtmosphereMemoryPrimitive(
        state_label=projected_state.label,
        D_local_m2_s=1.0e-10,
        atmosphere_relaxation_diffusivity_m2_s=1.0e-10,
        jump_length_m=1.0e-10,
        k_capture_s_inv=1.0,
        k_exit_s_inv=1.0,
        atmosphere_coupling_fraction=0.5,
        back_relaxation_probability=0.5,
        mobile_concentration_mol_m3=projected_state.concentration_mol_m3,
        atmosphere_concentration_per_direction_mol_m3=0.0,
        zeta0_kg_s=1.0e-9,
        zeta_ep_kg_s=1.0e-9,
        zeta_rel_kg_s=1.0e-9,
    )

    with pytest.raises(ValueError, match="memory correction exceeds"):
        analytical_model._projected_current_memory_correction_from_atmosphere_coordinate(
            memory_primitive,
            projected_state,
            T_REF_K,
        )


def test_atmosphere_memory_is_not_encoded_as_markov_pseudo_states():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(),
    )
    assert not any(
        state_label.endswith(":atmosphere")
        for state_label in result.markov_state_labels
    )
    assert not any(
        reactive_flux.family_label == "atmosphere_memory_translation"
        for reactive_flux in result.projected_transport_model.reactive_flux_integrals
    )


def test_atmosphere_mori_keeps_full_transport_concentration_on_mobile_state():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(),
    )
    markov_concentration_by_label = dict(
        zip(
            result.markov_state_labels,
            result.markov_state_concentrations_mol_m3,
        )
    )

    for transport_state in result.transport_states:
        if transport_state.center_charge_number == 0:
            continue
        if transport_state.transport_role == TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER:
            continue
        mobile_state_label = f"{transport_state.label}:mobile"
        assert mobile_state_label in markov_concentration_by_label
        assert markov_concentration_by_label[mobile_state_label] == pytest.approx(
            transport_state.concentration_mol_m3
        )


def test_projected_memory_coordinates_are_reported_in_result():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(),
    )

    assert result.projected_current_memory_corrections
    assert not result.atmosphere_mori_corrections
    assert (
        result.projected_transport_model.projected_current_memory_corrections
        == result.projected_current_memory_corrections
    )
    assert not (
        result
        .projected_transport_model
        .onsager_maxwell_stefan_operator
        .sigma_onsager_mS_cm
        < 0.0
    )
    assert not (
        result
        .projected_transport_model
        .onsager_transport_operator
        .correlation_corrector_mS_cm
        < 0.0
    )
    assert (
        result
        .markov_additive_result
        .projected_current_memory_corrector_sigma_mS_cm
        > 0.0
    )
    assert (
        result.markov_additive_result.projected_current_memory_corrector_sigma_mS_cm
        >= result.markov_additive_result.atmosphere_corrector_sigma_mS_cm
    )
    assert result.markov_additive_result.corrector_sigma_mS_cm == pytest.approx(
        result.markov_additive_result.event_mori_corrector_sigma_mS_cm
        + (
            result
            .markov_additive_result
            .projected_current_memory_corrector_sigma_mS_cm
        )
    )
    assert result.sigma_mS_cm == pytest.approx(
        result.markov_additive_result.direct_sigma_mS_cm
        - result.markov_additive_result.corrector_sigma_mS_cm
    )


def test_projected_current_memory_A_h_carries_projected_relaxation_rates():
    result = _compute_result(
        PAIR_CLUSTER_ION_COUNT,
        _primitive_parameters_with_replacements(),
    )

    combined_energy_matrix = (
        result
        .projected_transport_model
        .projected_primitive_set
        .mori_memory_energy_matrix_A
    )
    current_memory_energy_diagonal = np.diag(combined_energy_matrix)
    expected_current_memory_rates = np.asarray(
        tuple(
            correction.memory_self_energy_s_inv
            for correction in result.projected_current_memory_corrections
        ),
        dtype=float,
    )

    assert current_memory_energy_diagonal.size == expected_current_memory_rates.size
    assert current_memory_energy_diagonal == pytest.approx(
        expected_current_memory_rates
    )
    assert not np.allclose(
        current_memory_energy_diagonal,
        np.ones_like(current_memory_energy_diagonal),
    )


def test_projected_current_memory_A_h_matches_known_mori_quadratic_form():
    state_concentration_mol_m3 = 1000.0
    charge_diffusivity_m2_s = 1.0e-10
    memory_fraction = 0.25
    direct_axis_density_m2_s_mol_m3 = (
        state_concentration_mol_m3 * charge_diffusivity_m2_s
    )
    correction_axis_density_m2_s_mol_m3 = (
        direct_axis_density_m2_s_mol_m3 * memory_fraction,
        direct_axis_density_m2_s_mol_m3 * memory_fraction,
        direct_axis_density_m2_s_mol_m3 * memory_fraction,
    )
    correction_sigma_S_m = (
        (F * F / (R * T_REF_K))
        * math.fsum(correction_axis_density_m2_s_mol_m3)
        / 3.0
    )
    markov_result = compute_markov_additive_green_kubo_conductivity(
        MarkovAdditiveConductivityInput(
            state_labels=("mobile_motif",),
            state_concentrations_mol_m3=(state_concentration_mol_m3,),
            temperature_K=T_REF_K,
            events=(
                MarkovAdditiveEvent(
                    from_state_index=0,
                    to_state_index=0,
                    rate_s_inv=1.0,
                    charge_displacement_m=(0.0, 0.0, 0.0),
                    charge_displacement_second_moment_m2=(
                        (2.0 * charge_diffusivity_m2_s, 0.0, 0.0),
                        (0.0, 2.0 * charge_diffusivity_m2_s, 0.0),
                        (0.0, 0.0, 2.0 * charge_diffusivity_m2_s),
                    ),
                    label="projected_single_center_self_current:mobile_motif",
                    family_label=EVENT_FAMILY_PROJECTED_SINGLE_CENTER_SELF_CURRENT,
                ),
            ),
        )
    )

    assert correction_sigma_S_m > 0.0
    assert markov_result.event_mori_corrector_sigma_mS_cm == pytest.approx(0.0)
    current_memory_correction = analytical_model.ProjectedCurrentMemoryCorrection(
        state_label="mobile_motif:mobile",
        transport_state_label="mobile_motif",
        memory_family_label="synthetic_projected_cage_current_memory",
        concentration_mol_m3=state_concentration_mol_m3,
        memory_self_energy_s_inv=5.0e9,
        correction_axis_density_m2_s_mol_m3=correction_axis_density_m2_s_mol_m3,
        correction_sigma_S_m=correction_sigma_S_m,
        correction_sigma_mS_cm=correction_sigma_S_m * S_M_TO_MS_CM,
        source="synthetic_known_mori_quadratic_form",
    )
    current_memory_mori_result = analytical_model.compute_projected_mori_conductivity(
        analytical_model._projected_mori_input_from_current_memory_corrections(
            (current_memory_correction,),
            T_REF_K,
        )
    )

    assert current_memory_mori_result.quadratic_form_by_axis == pytest.approx(
        correction_axis_density_m2_s_mol_m3
    )
    corrected_result = analytical_model._markov_result_with_projected_current_memory(
        markov_result,
        (current_memory_correction,),
        tuple(),
        T_REF_K,
    )

    assert (
        corrected_result.projected_current_memory_corrector_sigma_mS_cm
        == pytest.approx(current_memory_correction.correction_sigma_mS_cm)
    )
    assert corrected_result.sigma_mS_cm == pytest.approx(
        markov_result.direct_sigma_mS_cm
        - current_memory_correction.correction_sigma_mS_cm
    )


def test_event_second_moment_must_dominate_mean_outer_product():
    with pytest.raises(ValueError, match="conditional_covariance_m2"):
        compute_markov_additive_green_kubo_conductivity(
            MarkovAdditiveConductivityInput(
                state_labels=("state_a", "state_b"),
                state_concentrations_mol_m3=(1.0, 1.0),
                temperature_K=T_REF_K,
                events=(
                    MarkovAdditiveEvent(
                        from_state_index=0,
                        to_state_index=1,
                        rate_s_inv=1.0,
                        charge_displacement_m=(1.0, 0.0, 0.0),
                        charge_displacement_second_moment_m2=(
                            (0.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0),
                        ),
                        label="invalid_event",
                        family_label="ordinary_mobile_translation",
                    ),
                    MarkovAdditiveEvent(
                        from_state_index=1,
                        to_state_index=0,
                        rate_s_inv=1.0,
                        charge_displacement_m=(-1.0, 0.0, 0.0),
                        charge_displacement_second_moment_m2=(
                            (0.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0),
                        ),
                        label="invalid_event_reverse",
                        family_label="ordinary_mobile_translation",
                    ),
                ),
            )
        )


def _compute_result(
    max_cluster_ion_count: int,
    primitive_parameters: ConductivityPrimitiveParameterSet,
):
    return compute_molecular_electrolyte_conductivity(
        _recipe(BASE_CONCENTRATION_M),
        _species_inputs(),
        ProvidedPropertyDescriptorBackend(),
        _options(max_cluster_ion_count, primitive_parameters),
    )


def _recipe(ion_concentration_M: float) -> MolecularElectrolyteRecipe:
    return MolecularElectrolyteRecipe(
        cations={"test_cation": ion_concentration_M},
        anions={"test_anion": ion_concentration_M},
        solvents={"test_solvent": 1.0},
        additives={},
        temperature_K=T_REF_K,
        pressure_Pa=PA_PER_ATM,
        mixture_properties=MolecularMixtureProperties(
            density_g_ml=BASE_DENSITY_G_ML,
            viscosity_cP=BASE_VISCOSITY_CP,
            dielectric_constant=BASE_DIELECTRIC,
        ),
    )


def _recipe_with_coordinating_additive(
    ion_concentration_M: float,
) -> MolecularElectrolyteRecipe:
    return MolecularElectrolyteRecipe(
        cations={"test_cation": ion_concentration_M},
        anions={"test_anion": ion_concentration_M},
        solvents={"test_solvent": 1.0},
        additives={"test_additive": 0.02},
        temperature_K=T_REF_K,
        pressure_Pa=PA_PER_ATM,
        mixture_properties=MolecularMixtureProperties(
            density_g_ml=BASE_DENSITY_G_ML,
            viscosity_cP=BASE_VISCOSITY_CP,
            dielectric_constant=BASE_DIELECTRIC,
        ),
    )


def _options(
    max_cluster_ion_count: int,
    primitive_parameters: ConductivityPrimitiveParameterSet,
) -> MolecularMoriOptions:
    return MolecularMoriOptions(
        max_cluster_ion_count=max_cluster_ion_count,
        max_packing_fraction=MAX_PACKING_FRACTION,
        free_volume_exponent=0.0,
        translation_jump_length_multiplier=JUMP_LENGTH_MULTIPLIER,
        primitive_parameters=primitive_parameters,
    )


def _species_inputs() -> dict[str, MolecularSpeciesInput]:
    return {
        "test_cation": _species_input(
            "test_cation",
            ROLE_CATION,
            1,
            _ion_properties(),
        ),
        "test_anion": _species_input(
            "test_anion",
            ROLE_ANION,
            -1,
            _ion_properties(),
        ),
        "test_solvent": _species_input(
            "test_solvent",
            ROLE_SOLVENT,
            0,
            _solvent_properties(),
        ),
    }


def _species_inputs_with_coordinating_additive() -> dict[str, MolecularSpeciesInput]:
    return {
        **_species_inputs(),
        "test_additive": _species_input(
            "test_additive",
            ROLE_ADDITIVE,
            0,
            _coordinating_additive_properties(),
            ("carbonyl_oxygen",),
        ),
    }


def _species_inputs_with_bulky_anion() -> dict[str, MolecularSpeciesInput]:
    return {
        **_species_inputs(),
        "test_anion": _species_input(
            "test_anion",
            ROLE_ANION,
            -1,
            _bulky_asymmetric_anion_properties(),
        ),
    }


def _species_inputs_with_coordinating_additive_and_bulky_anion() -> dict[
    str,
    MolecularSpeciesInput,
]:
    return {
        **_species_inputs_with_bulky_anion(),
        "test_additive": _species_input(
            "test_additive",
            ROLE_ADDITIVE,
            0,
            _coordinating_additive_properties(),
            ("carbonyl_oxygen",),
        ),
    }


def _species_input(
    species_name: str,
    species_role: str,
    charge_number: int,
    properties: dict[str, float],
    coordination_sites: tuple[str, ...] = (),
) -> MolecularSpeciesInput:
    return MolecularSpeciesInput(
        name=species_name,
        role=species_role,
        charge_number=charge_number,
        smiles="",
        xyz_coordinates=(),
        property_overrides=properties,
        coordination_sites=coordination_sites,
    )


def _ion_properties() -> dict[str, float]:
    return {
        "molecular_weight_g_mol": 80.0,
        "hard_sphere_radius_A": 2.4,
        "hydrodynamic_radius_A": 2.4,
        "cavity_radius_A": 2.4,
        "charge_cloud_radius_A": 1.2,
        "molecular_volume_A3": 58.0,
        "solvent_accessible_area_A2": 72.0,
        "dipole_D": 0.0,
        "quadrupole_D_A": 0.0,
        "polarizability_A3": 5.0,
        "donor_number": 0.0,
        "acceptor_number": 0.0,
        "hbond_donor_count": 0.0,
        "hbond_acceptor_count": 0.0,
        "epsilon_r_pure": BASE_DIELECTRIC,
        "viscosity_cP_pure": BASE_VISCOSITY_CP,
        "density_g_ml": BASE_DENSITY_G_ML,
        "born_solvation_radius_A": 2.4,
        "coordination_affinity_J_mol": 0.0,
        "ligand_field_asymmetry": 1.0,
    }


def _bulky_asymmetric_anion_properties() -> dict[str, float]:
    return {
        **_ion_properties(),
        "hydrodynamic_radius_A": 4.8,
        "hard_sphere_radius_A": 4.2,
        "cavity_radius_A": 4.2,
        "charge_cloud_radius_A": 3.6,
        "molecular_volume_A3": 220.0,
        "solvent_accessible_area_A2": 210.0,
        "ligand_field_asymmetry": 1.8,
    }


def _solvent_properties() -> dict[str, float]:
    return {
        "molecular_weight_g_mol": 100.0,
        "hard_sphere_radius_A": 3.0,
        "hydrodynamic_radius_A": 3.0,
        "cavity_radius_A": 3.0,
        "charge_cloud_radius_A": 1.0,
        "molecular_volume_A3": 70.0,
        "solvent_accessible_area_A2": 100.0,
        "dipole_D": 4.0,
        "quadrupole_D_A": 1.0,
        "polarizability_A3": 8.0,
        "donor_number": 16.0,
        "acceptor_number": 18.0,
        "hbond_donor_count": 0.0,
        "hbond_acceptor_count": 2.0,
        "epsilon_r_pure": BASE_DIELECTRIC,
        "viscosity_cP_pure": BASE_VISCOSITY_CP,
        "density_g_ml": BASE_DENSITY_G_ML,
        "born_solvation_radius_A": 3.0,
        "coordination_affinity_J_mol": 0.0,
        "ligand_field_asymmetry": 1.0,
    }


def _coordinating_additive_properties() -> dict[str, float]:
    return {
        "molecular_weight_g_mol": 110.0,
        "hard_sphere_radius_A": 3.2,
        "hydrodynamic_radius_A": 3.2,
        "cavity_radius_A": 3.2,
        "charge_cloud_radius_A": 1.0,
        "molecular_volume_A3": 82.0,
        "solvent_accessible_area_A2": 118.0,
        "dipole_D": 5.0,
        "quadrupole_D_A": 1.5,
        "polarizability_A3": 9.0,
        "donor_number": 12.0,
        "acceptor_number": 26.0,
        "hbond_donor_count": 0.0,
        "hbond_acceptor_count": 2.0,
        "epsilon_r_pure": BASE_DIELECTRIC,
        "viscosity_cP_pure": BASE_VISCOSITY_CP,
        "density_g_ml": BASE_DENSITY_G_ML,
        "born_solvation_radius_A": 3.2,
        "coordination_affinity_J_mol": 12000.0,
        "ligand_field_asymmetry": 1.4,
    }


def _identity_primitive_parameters() -> ConductivityPrimitiveParameterSet:
    signed_parameter_names = set(CONDUCTIVITY_PRIMITIVE_SIGNED_PARAMETER_FIELD_NAMES)
    parameter_values = {
        field_name: 0.0 if field_name in signed_parameter_names else 1.0
        for field_name in CONDUCTIVITY_PRIMITIVE_PARAMETER_FIELD_NAMES
    }
    return conductivity_primitive_parameters_from_mapping(parameter_values)


def _synthetic_property_db_row_result(
    row_id: int,
    empirical_sigma_mS_cm: float,
    predicted_sigma_mS_cm: float,
    direct_sigma_mS_cm: float,
    corrector_sigma_mS_cm: float,
    direct_capacity_failure: bool,
    corrector_too_strong_failure: bool,
    corrector_too_weak_failure: bool,
) -> MolecularPropertyDbRowResult:
    direct_capacity_gap_mS_cm = empirical_sigma_mS_cm - direct_sigma_mS_cm
    corrector_target_mS_cm = max(0.0, direct_sigma_mS_cm - empirical_sigma_mS_cm)
    corrector_residual_mS_cm = corrector_sigma_mS_cm - corrector_target_mS_cm
    return MolecularPropertyDbRowResult(
        row_id=row_id,
        source_row_ids=(row_id,),
        empirical_sigma_mS_cm=empirical_sigma_mS_cm,
        empirical_sigma_spread_mS_cm=0.0,
        predicted_sigma_mS_cm=predicted_sigma_mS_cm,
        residual_mS_cm=predicted_sigma_mS_cm - empirical_sigma_mS_cm,
        failed=False,
        failure_reason="",
        proof_status=PROJECTED_READOUT_PROVEN_DESCRIPTOR_CLOSURE_EMPIRICAL,
        direct_sigma_mS_cm=direct_sigma_mS_cm,
        corrector_sigma_mS_cm=corrector_sigma_mS_cm,
        onsager_ne_sigma_mS_cm=0.0,
        onsager_sigma_mS_cm=0.0,
        onsager_correlation_corrector_mS_cm=0.0,
        markov_state_changing_direct_sigma_mS_cm=direct_sigma_mS_cm,
        markov_state_changing_corrector_sigma_mS_cm=corrector_sigma_mS_cm,
        projected_current_memory_corrector_sigma_mS_cm=0.0,
        atmosphere_current_memory_corrector_sigma_mS_cm=0.0,
        structural_current_memory_corrector_delta_mS_cm=0.0,
        sigma_without_structural_current_memory_mS_cm=predicted_sigma_mS_cm,
        onsager_edge_count=0,
        max_onsager_edge_friction=0.0,
        top_onsager_friction_edges=tuple(),
        min_friction_matrix_eigenvalue=0.0,
        min_projected_mobility_eigenvalue=0.0,
        direct_capacity_gap_mS_cm=direct_capacity_gap_mS_cm,
        corrector_target_mS_cm=corrector_target_mS_cm,
        corrector_residual_mS_cm=corrector_residual_mS_cm,
        direct_capacity_failure=direct_capacity_failure,
        corrector_too_strong_failure=corrector_too_strong_failure,
        corrector_too_weak_failure=corrector_too_weak_failure,
        direct_sigma_by_transport_role_mS_cm={
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER: direct_sigma_mS_cm,
            TRANSPORT_ROLE_CONTACT_PAIR_CENTER: 0.0,
            TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER: 0.0,
        },
        corrector_sigma_by_transport_role_mS_cm={
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER: corrector_sigma_mS_cm,
            TRANSPORT_ROLE_CONTACT_PAIR_CENTER: 0.0,
            TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER: 0.0,
        },
        net_sigma_by_transport_role_mS_cm={
            TRANSPORT_ROLE_SOLVENT_SEPARATED_PAIR_CENTER: (
                direct_sigma_mS_cm - corrector_sigma_mS_cm
            ),
            TRANSPORT_ROLE_CONTACT_PAIR_CENTER: 0.0,
            TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER: 0.0,
        },
        charge_weighted_transport_concentration_mol_m3=1000.0,
        mass_balance_residual_mol_m3=0.0,
        row_sum_residual=0.0,
        stationary_residual_mol_m3_s=0.0,
        detailed_balance_residual_mol_m3_s=0.0,
        event_reversal_residual_mol_m3_s=0.0,
        free_ion_fraction=0.5,
        charged_cluster_fraction=0.1,
        neutral_cluster_fraction=0.1,
        cluster_transport_mobility_density_mol_m_s=1.0,
        charged_cluster_transport_mobility_density_mol_m_s=0.1,
        neutral_cluster_transport_mobility_density_mol_m_s=0.1,
        charged_cluster_direct_sigma_mS_cm=0.2,
        charged_cluster_corrector_sigma_mS_cm=0.1,
        charged_cluster_net_sigma_mS_cm=0.1,
        primitive_residual_owners=(
            MolecularPrimitiveResidualOwner(
                primitive_head=PRIMITIVE_RESIDUAL_HEAD_SELF_CURRENT_MOBILITY,
                theorem_object=PRIMITIVE_RESIDUAL_THEOREM_SELF_CURRENT_MOBILITY,
                production_lever=(
                    PRIMITIVE_RESIDUAL_PRODUCTION_LEVER_SOLVENT_SEPARATED_PAIR_SELF_CURRENT
                ),
                residual_component=(
                    PRIMITIVE_RESIDUAL_COMPONENT_SYNTHETIC_DIRECT_CAPACITY
                ),
                residual_mS_cm=direct_capacity_gap_mS_cm,
                evidence_label="synthetic_event_family",
                source_label="synthetic_projected_generator_audit",
            ),
        ),
        markov_event_family_attributions=tuple(),
        cluster_thermodynamic_diagnostics=(
            MolecularClusterThermodynamicDiagnostic(
                row_id=row_id,
                cluster_label="synthetic_cluster",
                cluster_kind="synthetic_kind",
                stoichiometry={"Li+": 1, "PF6-": 1},
                net_charge_number=0,
                concentration_mol_m3=1.0,
                concentration_fraction_of_total_ion=0.001,
                standard_free_energy_J_mol=1.0,
                standard_free_energy_over_RT=1.0,
                log_equilibrium_constant=-1.0,
                coulomb_J_mol=0.1,
                desolvation_J_mol=0.2,
                coordination_J_mol=0.3,
                steric_J_mol=0.4,
                entropy_J_mol=0.5,
                activity_reference_J_mol=0.0,
                activity_correction_J_mol=0.0,
                hydrodynamic_radius_A=5.0,
                molecular_volume_A3=100.0,
            ),
        ),
    )


def _synthetic_promotion_metrics(
    proof_statuses: tuple[str, ...],
) -> PrimitivePromotionMetrics:
    return PrimitivePromotionMetrics(
        mae_mS_cm=0.0,
        bias_mS_cm=0.0,
        pearson_r=1.0,
        worst_abs_residual_mS_cm=0.0,
        proof_statuses=proof_statuses,
        failed_rows=0,
        trajectory_concentration_unreachable_target_count=0,
        trajectory_concentration_under_floor_target_count=0,
        maximum_mass_balance_residual=0.0,
        maximum_row_sum_residual=0.0,
        maximum_stationary_residual=0.0,
        maximum_detailed_balance_residual=0.0,
        maximum_event_reversal_residual=0.0,
        zero_charge_sigma_mS_cm=0.0,
        higher_viscosity_lowers_dilute_conductivity=True,
        higher_packing_lowers_local_mobility=True,
    )


@dataclass(frozen=True)
class _SyntheticCollinearPrimitiveEvaluator:
    fitted_parameter_names: tuple[str, ...]

    def evaluate(
        self,
        primitive_parameters: ConductivityPrimitiveParameterSet,
    ) -> PrimitiveFitDatasetEvaluation:
        coordinate_values = conductivity_primitive_parameter_coordinate_values_for_names(
            primitive_parameters,
            self.fitted_parameter_names,
        )
        collinear_signal_mS_cm = coordinate_values[0] + coordinate_values[1]
        predicted_sigmas_mS_cm = (
            5.0 + collinear_signal_mS_cm,
            6.0 + 2.0 * collinear_signal_mS_cm,
            7.0 + 3.0 * collinear_signal_mS_cm,
        )
        direct_sigmas_mS_cm = (10.0, 12.0, 14.0)
        corrector_sigmas_mS_cm = tuple(
            direct_sigma_mS_cm - predicted_sigma_mS_cm
            for direct_sigma_mS_cm, predicted_sigma_mS_cm in zip(
                direct_sigmas_mS_cm,
                predicted_sigmas_mS_cm,
            )
        )
        return PrimitiveFitDatasetEvaluation(
            descriptor_calibration_targets=(
                DescriptorCalibrationTarget(
                    target_id="synthetic:0",
                    source_row_ids=(0,),
                    descriptor_driver_values=(("synthetic_descriptor", 0.0),),
                    empirical_sigma_mS_cm=5.0,
                    empirical_sigma_spread_mS_cm=0.0,
                    residual_weight=1.0,
                ),
                DescriptorCalibrationTarget(
                    target_id="synthetic:1",
                    source_row_ids=(1,),
                    descriptor_driver_values=(("synthetic_descriptor", 1.0),),
                    empirical_sigma_mS_cm=6.0,
                    empirical_sigma_spread_mS_cm=0.0,
                    residual_weight=1.0,
                ),
                DescriptorCalibrationTarget(
                    target_id="synthetic:2",
                    source_row_ids=(2,),
                    descriptor_driver_values=(("synthetic_descriptor", 2.0),),
                    empirical_sigma_mS_cm=7.0,
                    empirical_sigma_spread_mS_cm=0.0,
                    residual_weight=1.0,
                ),
            ),
            trajectory_primitive_calibration_targets=tuple(),
            predicted_sigmas_mS_cm=predicted_sigmas_mS_cm,
            direct_sigmas_mS_cm=direct_sigmas_mS_cm,
            corrector_sigmas_mS_cm=corrector_sigmas_mS_cm,
            direct_capacity_gaps_mS_cm=tuple(
                empirical_sigma_mS_cm - direct_sigma_mS_cm
                for empirical_sigma_mS_cm, direct_sigma_mS_cm in zip(
                    (5.0, 6.0, 7.0),
                    direct_sigmas_mS_cm,
                )
            ),
            corrector_targets_mS_cm=tuple(
                max(0.0, direct_sigma_mS_cm - empirical_sigma_mS_cm)
                for empirical_sigma_mS_cm, direct_sigma_mS_cm in zip(
                    (5.0, 6.0, 7.0),
                    direct_sigmas_mS_cm,
                )
            ),
            corrector_residuals_mS_cm=tuple(
                corrector_sigma_mS_cm
                - max(0.0, direct_sigma_mS_cm - empirical_sigma_mS_cm)
                for empirical_sigma_mS_cm, direct_sigma_mS_cm, corrector_sigma_mS_cm in zip(
                    (5.0, 6.0, 7.0),
                    direct_sigmas_mS_cm,
                    corrector_sigmas_mS_cm,
                )
            ),
            direct_capacity_failure_count=0,
            corrector_too_strong_failure_count=0,
            corrector_too_weak_failure_count=0,
            trajectory_concentration_loss=0.0,
            trajectory_transition_rate_loss=0.0,
            trajectory_displacement_moment_loss=0.0,
            trajectory_sigma_loss_mS_cm=0.0,
            trajectory_concentration_unreachable_target_count=0,
            trajectory_concentration_under_floor_target_count=0,
            cluster_activation_penalty=0.0,
            failed_rows=0,
            maximum_mass_balance_residual=0.0,
            maximum_row_sum_residual=0.0,
            maximum_stationary_residual=0.0,
            maximum_detailed_balance_residual=0.0,
            maximum_event_reversal_residual=0.0,
            zero_charge_sigma_mS_cm=0.0,
            higher_viscosity_lowers_dilute_conductivity=True,
            higher_packing_lowers_local_mobility=True,
            consumed_parameter_fields=self.fitted_parameter_names,
        )


def _primitive_parameters_with_replacements(
    coulomb_scale: float = 1.0,
    desolvation_scale: float = 1.0,
    coordination_scale: float = 1.0,
    steric_free_energy_scale: float = 1.0,
    cluster_entropy_penalty_scale: float = 1.0,
    association_crowding_stabilization_scale: float = 1.0,
    association_crowding_ionic_strength_exponent: float = 1.0,
    association_crowding_charge_density_exponent: float = 1.0,
    activity_debye_scale: float = 1.0,
    activity_size_scale: float = 1.0,
    activity_hard_sphere_scale: float = 1.0,
    cluster_activity_scale: float = 1.0,
    pair_logK_offset: float = 0.0,
    solvent_separated_pair_logK_offset: float = 0.0,
    contact_pair_logK_offset: float = 0.0,
    positive_charged_triplet_logK_offset: float = 0.0,
    negative_charged_triplet_logK_offset: float = 0.0,
    neutral_cluster_logK_offset: float = 0.0,
    higher_charged_cluster_logK_offset: float = 0.0,
    contact_pair_desolvation_offset_over_RT: float = 0.0,
    solvent_separated_pair_desolvation_offset_over_RT: float = 0.0,
    higher_charged_cluster_desolvation_offset_over_RT: float = 0.0,
    internal_polarization_projection_offset: float = -20.0,
    internal_polarization_projection_ionic_strength_slope: float = 1.0,
    internal_polarization_projection_counterion_crowding_slope: float = 1.0,
    cluster_order_logK_slope: float = 0.0,
    cluster_charge_magnitude_logK_slope: float = 0.0,
) -> ConductivityPrimitiveParameterSet:
    parameter_values = {
        **conductivity_primitive_parameters_to_mapping(_identity_primitive_parameters()),
        "coulomb_scale": coulomb_scale,
        "desolvation_scale": desolvation_scale,
        "coordination_scale": coordination_scale,
        "steric_free_energy_scale": steric_free_energy_scale,
        "cluster_entropy_penalty_scale": cluster_entropy_penalty_scale,
        "association_crowding_stabilization_scale": (
            association_crowding_stabilization_scale
        ),
        "association_crowding_ionic_strength_exponent": (
            association_crowding_ionic_strength_exponent
        ),
        "association_crowding_charge_density_exponent": (
            association_crowding_charge_density_exponent
        ),
        "activity_debye_scale": activity_debye_scale,
        "activity_size_scale": activity_size_scale,
        "activity_hard_sphere_scale": activity_hard_sphere_scale,
        "cluster_activity_scale": cluster_activity_scale,
        "pair_logK_offset": pair_logK_offset,
        "solvent_separated_pair_logK_offset": solvent_separated_pair_logK_offset,
        "contact_pair_logK_offset": contact_pair_logK_offset,
        "positive_charged_triplet_logK_offset": (
            positive_charged_triplet_logK_offset
        ),
        "negative_charged_triplet_logK_offset": (
            negative_charged_triplet_logK_offset
        ),
        "neutral_cluster_logK_offset": neutral_cluster_logK_offset,
        "higher_charged_cluster_logK_offset": higher_charged_cluster_logK_offset,
        "contact_pair_desolvation_offset_over_RT": (
            contact_pair_desolvation_offset_over_RT
        ),
        "solvent_separated_pair_desolvation_offset_over_RT": (
            solvent_separated_pair_desolvation_offset_over_RT
        ),
        "higher_charged_cluster_desolvation_offset_over_RT": (
            higher_charged_cluster_desolvation_offset_over_RT
        ),
        "internal_polarization_projection_offset": (
            internal_polarization_projection_offset
        ),
        "internal_polarization_projection_ionic_strength_slope": (
            internal_polarization_projection_ionic_strength_slope
        ),
        "internal_polarization_projection_counterion_crowding_slope": (
            internal_polarization_projection_counterion_crowding_slope
        ),
        "cluster_order_logK_slope": cluster_order_logK_slope,
        "cluster_charge_magnitude_logK_slope": cluster_charge_magnitude_logK_slope,
    }
    return conductivity_primitive_parameters_from_mapping(parameter_values)


def _cluster_state_by_kind(result, cluster_kind: str):
    for cluster_state in result.cluster_states:
        if cluster_state.cluster_kind == cluster_kind:
            return cluster_state
    raise AssertionError(f"missing cluster state kind {cluster_kind}")


def _cluster_concentration_by_kind(result, cluster_kind: str) -> float:
    return math.fsum(
        result.speciation.cluster_concentrations_mol_m3[cluster_state.label]
        for cluster_state in result.cluster_states
        if cluster_state.cluster_kind == cluster_kind
    )


def _total_cluster_concentration_mol_m3(result) -> float:
    return math.fsum(result.speciation.cluster_concentrations_mol_m3.values())


def _charged_cluster_concentration_mol_m3(result) -> float:
    return math.fsum(
        result.speciation.cluster_concentrations_mol_m3[cluster_state.label]
        for cluster_state in result.cluster_states
        if cluster_state.net_charge_number != 0
    )


def _transport_role_concentration_mol_m3(result, transport_role: str) -> float:
    return math.fsum(
        transport_center.concentration_mol_m3
        for transport_center in result.transport_states
        if transport_center.transport_role == transport_role
    )


def _trajectory_coverage_target_and_prediction(
    trajectory_coverage,
    target_label: str,
) -> tuple[float, float]:
    for (
        row_target_label,
        target_concentration_mol_m3,
        predicted_concentration_mol_m3,
        _reachable,
    ) in trajectory_coverage.predicted_target_rows:
        if row_target_label == target_label:
            return target_concentration_mol_m3, predicted_concentration_mol_m3
    raise ValueError(f"missing trajectory coverage row for {target_label}")


def _is_strictly_greater(left_value: float, right_value: float) -> bool:
    return operator.gt(left_value, right_value)


def _transport_centers_for_parent_and_role(
    result,
    parent_cluster_label: str,
    transport_role: str,
):
    return tuple(
        transport_center
        for transport_center in result.transport_states
        if transport_center.parent_cluster_label == parent_cluster_label
        and transport_center.transport_role == transport_role
    )


def _projected_states_for_basin(
    result,
    pair_basin: str,
):
    return tuple(
        projected_state
        for projected_state in result.projected_transport_model.projected_transport_states
        if projected_state.pair_basin == pair_basin
    )


def _projected_basin_concentration_mol_m3(result, pair_basin: str) -> float:
    return math.fsum(
        projected_state.concentration_mol_m3
        for projected_state in _projected_states_for_basin(result, pair_basin)
    )


def _direct_sigma_for_transport_role(
    result,
    transport_role: str,
) -> float:
    transport_center_labels = tuple(
        transport_center.label
        for transport_center in result.transport_states
        if transport_center.transport_role == transport_role
    )
    state_concentrations_mol_m3 = result.markov_state_concentrations_mol_m3
    direct_axis_density_m2_s_mol_m3 = tuple(
        math.fsum(
            0.5
            * state_concentrations_mol_m3[event.from_state_index]
            * event.rate_s_inv
            * float(
                np.asarray(
                    event.charge_displacement_second_moment_m2,
                    dtype=float,
                )[axis_index, axis_index]
            )
            for event in result.events
            if _event_matches_any_center_label(event.label, transport_center_labels)
        )
        for axis_index in range(3)
    )
    return (
        F
        * F
        / (3.0 * R * result.solvent_environment.temperature_K)
        * math.fsum(direct_axis_density_m2_s_mol_m3)
        * S_M_TO_MS_CM
    )


def _onsager_ne_sigma_for_transport_role(
    result,
    transport_role: str,
) -> float:
    onsager_operator = result.projected_transport_model.onsager_transport_operator
    if not onsager_operator.state_labels:
        return _direct_sigma_for_transport_role(result, transport_role)
    direct_sigma_S_m = math.fsum(
        F
        * F
        / (R * result.solvent_environment.temperature_K)
        * concentration_mol_m3
        * charge_number
        * charge_number
        * diffusivity_m2_s
        for state_label, concentration_mol_m3, charge_number, diffusivity_m2_s in zip(
            onsager_operator.state_labels,
            onsager_operator.concentrations_mol_m3,
            onsager_operator.charge_numbers,
            onsager_operator.bare_diffusivities_m2_s,
        )
        if state_label.startswith(f"{transport_role}:")
    )
    return direct_sigma_S_m * S_M_TO_MS_CM


def _events_for_family(result, family_label: str):
    return tuple(
        event
        for event in result.events
        if event.family_label == family_label
    )


def _projected_ssip_second_moment_trace_m2(result) -> float:
    projected_ssip_events = _events_for_family(
        result,
        "projected_solvent_separated_pair_self_current",
    )
    return math.fsum(
        np.trace(np.asarray(event.charge_displacement_second_moment_m2, dtype=float))
        for event in projected_ssip_events
    )


def _events_for_transport_role_and_family(
    result,
    transport_role: str,
    family_label: str,
):
    transport_center_labels = tuple(
        transport_center.label
        for transport_center in result.transport_states
        if transport_center.transport_role == transport_role
    )
    return tuple(
        event
        for event in result.events
        if event.family_label == family_label
        and _event_matches_any_center_label(event.label, transport_center_labels)
    )


def _event_matches_any_center_label(
    event_label: str,
    transport_center_labels: tuple[str, ...],
) -> bool:
    return any(
        transport_center_label in event_label
        for transport_center_label in transport_center_labels
    )


def _zero_displacement(
    charge_displacement_m: tuple[float, float, float],
) -> bool:
    return all(displacement_m == 0.0 for displacement_m in charge_displacement_m)


def _cluster_extent_A(cluster_state) -> float:
    positions_A = tuple(center.position_A[0] for center in cluster_state.geometry)
    return max(positions_A) - min(positions_A)
