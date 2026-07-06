from dataclasses import replace

import numpy as np

from constants import T_REF_K
from conductivity.analytical_conductivity_model import (
    BulkIonAtmosphereInput,
    MolecularIonAtmosphereDiagnostics,
    MolecularSolventEnvironment,
    MolecularTransportCenter,
    TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER,
    _aggregate_onsager_carrier_states,
    _onsager_transport_operator_from_transport_states,
    build_bulk_ion_atmosphere_state,
)
from conductivity.molecular_property_db_audit import (
    MolecularPropertyDbRegistrySource,
    build_molecular_property_db_case_selection,
    configured_conductivity_primitive_parameters,
    default_molecular_property_db_audit_options,
    audit_molecular_property_db_cases,
)
from data.electrolyte_property_db import DATA
from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS, SOLVENTS


def _transport_state(
    label,
    charge_number,
    concentration_mol_m3,
    diffusion_m2_s,
):
    return MolecularTransportCenter(
        label=label,
        parent_cluster_label=label,
        parent_cluster_kind="test_cluster",
        concentration_mol_m3=concentration_mol_m3,
        center_species_name=label,
        center_charge_number=charge_number,
        center_index=0,
        hydrodynamic_radius_A=2.5,
        charge_cloud_radius_A=3.0,
        molecular_volume_A3=40.0,
        ligand_field_asymmetry=1.0,
        diffusion_m2_s=diffusion_m2_s,
        local_obstruction_factor=1.0,
        local_obstruction_diffusion_scale=1.0,
        transport_role="free_ion_center",
    )


def _solvent_environment():
    return MolecularSolventEnvironment(
        dielectric_constant=20.0,
        viscosity_cP=2.0,
        hard_sphere_volume_fraction=0.15,
        temperature_K=T_REF_K,
        solvent_effective_radius_A=2.0,
        mean_molecular_volume_A3=120.0,
        solvent_volume_fractions={"test_solvent": 1.0},
        solvent_coordination_affinity_J_mol=30000.0,
        additive_ligand_site_occupancy=0.0,
        additive_coordination_affinity_J_mol=0.0,
        additive_solvation_support=0.0,
        additive_molecular_volume_A3=0.0,
    )


def _atmosphere_diagnostics():
    return MolecularIonAtmosphereDiagnostics(
        solver="finite_size_bulk_pnp_stokes_l1_cell",
        charged_carrier_count=2,
        kappa_inv_m=1.0e-9,
        ionic_strength_mol_m3=2000.0,
        charge_cloud_form_factor_by_state={
            "Li_state": 1.0,
            "PF6_state": 1.0,
        },
        friction_ratio_by_state={
            "Li_state": 1.0,
            "PF6_state": 1.0,
        },
        zeta0_kg_s_by_state={
            "Li_state": 1.0,
            "PF6_state": 1.0,
        },
        zeta_ep_kg_s_by_state={
            "Li_state": 1.0,
            "PF6_state": 1.0,
        },
        zeta_rel_kg_s_by_state={
            "Li_state": 1.0,
            "PF6_state": 1.0,
        },
        countercharge_relaxation_diffusivity_m2_s_by_state={
            "Li_state": 1.0e-10,
            "PF6_state": 1.0e-10,
        },
    )


def _onsager_operator(cross_relaxation_scale, atmosphere_ep_scale, atmosphere_rel_scale):
    primitive_parameters = replace(
        configured_conductivity_primitive_parameters(),
        cross_relaxation_scale=cross_relaxation_scale,
        atmosphere_ep_scale=atmosphere_ep_scale,
        atmosphere_rel_scale=atmosphere_rel_scale,
    )
    transport_states = (
        _transport_state("Li_state", 1, 1000.0, 1.5e-10),
        _transport_state("PF6_state", -1, 1000.0, 1.1e-10),
    )
    return _onsager_transport_operator_from_transport_states(
        transport_states,
        _solvent_environment(),
        primitive_parameters,
        T_REF_K,
        _atmosphere_diagnostics(),
    )


def _neutral_cluster_member_operator(local_obstruction_factor):
    primitive_parameters = replace(
        configured_conductivity_primitive_parameters(),
        cross_relaxation_scale=0.0,
        atmosphere_ep_scale=1.0,
        atmosphere_rel_scale=1.0,
    )
    transport_states = (
        MolecularTransportCenter(
            label="neutral_cluster_li",
            parent_cluster_label="neutral_cluster",
            parent_cluster_kind="neutral_cluster",
            concentration_mol_m3=900.0,
            center_species_name="Li+",
            center_charge_number=1,
            center_index=0,
            hydrodynamic_radius_A=2.5,
            charge_cloud_radius_A=3.0,
            molecular_volume_A3=40.0,
            ligand_field_asymmetry=1.0,
            diffusion_m2_s=1.5e-10,
            local_obstruction_factor=local_obstruction_factor,
            local_obstruction_diffusion_scale=1.0,
            transport_role=TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER,
        ),
        MolecularTransportCenter(
            label="neutral_cluster_pf6",
            parent_cluster_label="neutral_cluster",
            parent_cluster_kind="neutral_cluster",
            concentration_mol_m3=900.0,
            center_species_name="PF6-",
            center_charge_number=-1,
            center_index=1,
            hydrodynamic_radius_A=2.9,
            charge_cloud_radius_A=3.4,
            molecular_volume_A3=45.0,
            ligand_field_asymmetry=1.0,
            diffusion_m2_s=1.1e-10,
            local_obstruction_factor=local_obstruction_factor,
            local_obstruction_diffusion_scale=1.0,
            transport_role=TRANSPORT_ROLE_CLUSTER_MEMBER_CENTER,
        ),
    )
    atmosphere_diagnostics = MolecularIonAtmosphereDiagnostics(
        solver="finite_size_bulk_pnp_stokes_l1_cell",
        charged_carrier_count=2,
        kappa_inv_m=1.0e-9,
        ionic_strength_mol_m3=1800.0,
        charge_cloud_form_factor_by_state={
            "neutral_cluster_li": 1.0,
            "neutral_cluster_pf6": 1.0,
        },
        friction_ratio_by_state={
            "neutral_cluster_li": 1.0,
            "neutral_cluster_pf6": 1.0,
        },
        zeta0_kg_s_by_state={
            "neutral_cluster_li": 1.0,
            "neutral_cluster_pf6": 1.0,
        },
        zeta_ep_kg_s_by_state={
            "neutral_cluster_li": 1.0,
            "neutral_cluster_pf6": 1.0,
        },
        zeta_rel_kg_s_by_state={
            "neutral_cluster_li": 1.0,
            "neutral_cluster_pf6": 1.0,
        },
        countercharge_relaxation_diffusivity_m2_s_by_state={
            "neutral_cluster_li": 1.0e-10,
            "neutral_cluster_pf6": 1.0e-10,
        },
    )
    return _onsager_transport_operator_from_transport_states(
        transport_states,
        _solvent_environment(),
        primitive_parameters,
        T_REF_K,
        atmosphere_diagnostics,
    )

def test_onsager_friction_matrix_is_laplacian_signed():
    operator = _onsager_operator(
        cross_relaxation_scale=1.0,
        atmosphere_ep_scale=1.0,
        atmosphere_rel_scale=1.0,
    )
    friction_matrix = np.asarray(operator.friction_matrix, dtype=float)
    assert friction_matrix.shape == (2, 2)
    assert friction_matrix[0, 1] <= 0.0
    assert friction_matrix[1, 0] <= 0.0
    assert not friction_matrix[0, 0] < 0.0
    assert not friction_matrix[1, 1] < 0.0
    assert np.allclose(friction_matrix, friction_matrix.T)


def test_onsager_operator_reduces_to_nernst_einstein_when_edges_zero():
    operator = _onsager_operator(
        cross_relaxation_scale=0.0,
        atmosphere_ep_scale=1.0,
        atmosphere_rel_scale=1.0,
    )
    assert operator.friction_edges == tuple()
    assert np.isclose(
        operator.onsager_sigma_mS_cm,
        operator.nernst_einstein_sigma_mS_cm,
    )
    assert np.isclose(operator.correlation_corrector_mS_cm, 0.0)


def test_positive_pair_friction_reduces_conductivity():
    uncoupled_operator = _onsager_operator(
        cross_relaxation_scale=0.0,
        atmosphere_ep_scale=1.0,
        atmosphere_rel_scale=1.0,
    )
    coupled_operator = _onsager_operator(
        cross_relaxation_scale=1.0,
        atmosphere_ep_scale=1.0,
        atmosphere_rel_scale=1.0,
    )
    assert coupled_operator.onsager_sigma_mS_cm <= (
        coupled_operator.nernst_einstein_sigma_mS_cm + 1.0e-12
    )
    assert coupled_operator.onsager_sigma_mS_cm <= (
        uncoupled_operator.onsager_sigma_mS_cm + 1.0e-12
    )


def test_neutral_bound_pair_member_centers_do_not_enter_onsager_self_current():
    operator = _neutral_cluster_member_operator(
        local_obstruction_factor=1.0,
    )

    assert operator.state_labels == tuple()
    assert operator.friction_edges == tuple()
    assert operator.nernst_einstein_sigma_mS_cm == 0.0
    assert operator.onsager_sigma_mS_cm == 0.0
    assert operator.correlation_corrector_mS_cm == 0.0


def test_onsager_corrector_never_exceeds_ne_direct():
    operator = _onsager_operator(
        cross_relaxation_scale=1.0,
        atmosphere_ep_scale=1.0,
        atmosphere_rel_scale=1.0,
    )
    assert not operator.correlation_corrector_mS_cm < 0.0
    assert operator.correlation_corrector_mS_cm <= (
        operator.nernst_einstein_sigma_mS_cm + 1.0e-12
    )


def test_maxwell_stefan_pair_drag_concentration_scaling():
    bulk_state = build_bulk_ion_atmosphere_state(
        BulkIonAtmosphereInput(
            carrier_labels=("Li_state", "PF6_state"),
            carrier_concentrations_mol_m3={
                "Li_state": 600.0,
                "PF6_state": 900.0,
            },
            carrier_charges={
                "Li_state": 1,
                "PF6_state": -1,
            },
            local_diffusivity_m2_s_by_carrier={
                "Li_state": 1.5e-10,
                "PF6_state": 1.1e-10,
            },
            hydrodynamic_radius_m_by_carrier={
                "Li_state": 2.5e-10,
                "PF6_state": 2.7e-10,
            },
            viscosity_Pa_s=2.0e-3,
            relative_dielectric=20.0,
            temperature_K=T_REF_K,
            solver="finite_size_bulk_pnp_stokes_l1_cell",
        )
    )
    pair_friction_matrix = np.asarray(
        bulk_state.maxwell_stefan_pair_friction_coefficient_matrix_J_s_mol_m2,
        dtype=float,
    )
    pair_drag_matrix = np.asarray(
        bulk_state.maxwell_stefan_pair_drag_matrix_J_s_mol_m2,
        dtype=float,
    )
    effective_pair_friction_J_s_mol_m2 = pair_friction_matrix[0, 1]
    assert effective_pair_friction_J_s_mol_m2 > 0.0
    assert np.isclose(
        pair_drag_matrix[0, 1],
        600.0 * 900.0 * effective_pair_friction_J_s_mol_m2,
    )


def test_ssip_internal_polarization_states_do_not_enter_onsager_self_current():
    atmosphere_diagnostics = MolecularIonAtmosphereDiagnostics(
        solver="finite_size_bulk_pnp_stokes_l1_cell",
        charged_carrier_count=4,
        kappa_inv_m=1.0e-9,
        ionic_strength_mol_m3=2000.0,
        charge_cloud_form_factor_by_state={
            "ssip_li_a": 1.0,
            "internal_li_b": 1.0,
            "ssip_pf6_a": 1.0,
            "internal_pf6_b": 1.0,
        },
        friction_ratio_by_state={
            "ssip_li_a": 1.0,
            "internal_li_b": 1.0,
            "ssip_pf6_a": 1.0,
            "internal_pf6_b": 1.0,
        },
        zeta0_kg_s_by_state={
            "ssip_li_a": 1.0,
            "internal_li_b": 1.0,
            "ssip_pf6_a": 1.0,
            "internal_pf6_b": 1.0,
        },
        zeta_ep_kg_s_by_state={
            "ssip_li_a": 1.0,
            "internal_li_b": 1.0,
            "ssip_pf6_a": 1.0,
            "internal_pf6_b": 1.0,
        },
        zeta_rel_kg_s_by_state={
            "ssip_li_a": 1.0,
            "internal_li_b": 1.0,
            "ssip_pf6_a": 1.0,
            "internal_pf6_b": 1.0,
        },
        countercharge_relaxation_diffusivity_m2_s_by_state={
            "ssip_li_a": 1.0e-10,
            "internal_li_b": 1.0e-10,
            "ssip_pf6_a": 1.0e-10,
            "internal_pf6_b": 1.0e-10,
        },
    )
    transport_states = (
        MolecularTransportCenter(
            label="ssip_li_a",
            parent_cluster_label="ssip_a",
            parent_cluster_kind="solvent_separated_pair",
            concentration_mol_m3=150.0,
            center_species_name="Li+",
            center_charge_number=1,
            center_index=0,
            hydrodynamic_radius_A=2.5,
            charge_cloud_radius_A=3.0,
            molecular_volume_A3=40.0,
            ligand_field_asymmetry=1.0,
            diffusion_m2_s=1.5e-10,
            local_obstruction_factor=1.0,
            local_obstruction_diffusion_scale=1.0,
            transport_role="solvent_separated_pair_center",
        ),
        MolecularTransportCenter(
            label="internal_li_b",
            parent_cluster_label="ssip_b",
            parent_cluster_kind="solvent_separated_pair",
            concentration_mol_m3=50.0,
            center_species_name="Li+",
            center_charge_number=1,
            center_index=0,
            hydrodynamic_radius_A=2.7,
            charge_cloud_radius_A=3.2,
            molecular_volume_A3=42.0,
            ligand_field_asymmetry=1.0,
            diffusion_m2_s=1.2e-10,
            local_obstruction_factor=1.0,
            local_obstruction_diffusion_scale=1.0,
            transport_role="internal_polarization_center",
        ),
        MolecularTransportCenter(
            label="ssip_pf6_a",
            parent_cluster_label="ssip_a",
            parent_cluster_kind="solvent_separated_pair",
            concentration_mol_m3=160.0,
            center_species_name="PF6-",
            center_charge_number=-1,
            center_index=1,
            hydrodynamic_radius_A=2.9,
            charge_cloud_radius_A=3.4,
            molecular_volume_A3=45.0,
            ligand_field_asymmetry=1.0,
            diffusion_m2_s=1.1e-10,
            local_obstruction_factor=1.0,
            local_obstruction_diffusion_scale=1.0,
            transport_role="solvent_separated_pair_center",
        ),
        MolecularTransportCenter(
            label="internal_pf6_b",
            parent_cluster_label="ssip_b",
            parent_cluster_kind="solvent_separated_pair",
            concentration_mol_m3=40.0,
            center_species_name="PF6-",
            center_charge_number=-1,
            center_index=1,
            hydrodynamic_radius_A=3.1,
            charge_cloud_radius_A=3.5,
            molecular_volume_A3=46.0,
            ligand_field_asymmetry=1.0,
            diffusion_m2_s=1.0e-10,
            local_obstruction_factor=1.0,
            local_obstruction_diffusion_scale=1.0,
            transport_role="internal_polarization_center",
        ),
    )
    aggregated_carriers = _aggregate_onsager_carrier_states(
        transport_states,
        atmosphere_diagnostics,
    )
    assert aggregated_carriers == tuple()


def test_audit_reports_onsager_decomposition_fields():
    audit_options = default_molecular_property_db_audit_options()
    registry_source = MolecularPropertyDbRegistrySource(
        solvent_registry=SOLVENTS,
        salt_registry=SALTS,
        additive_registry=ADDITIVES,
        cation_registry=CATION_PROPERTIES,
    )
    case_selection = build_molecular_property_db_case_selection(
        tuple(DATA[:1]),
        registry_source,
        audit_options,
    )
    primitive_parameters = replace(
        configured_conductivity_primitive_parameters(),
        cross_relaxation_scale=1.0e-30,
    )
    audit_result = audit_molecular_property_db_cases(
        case_selection.cases,
        primitive_parameters,
        audit_options,
    )
    row_result = audit_result.rows[0]
    assert hasattr(row_result, "onsager_ne_sigma_mS_cm")
    assert hasattr(row_result, "onsager_sigma_mS_cm")
    assert hasattr(row_result, "onsager_correlation_corrector_mS_cm")
    assert hasattr(row_result, "markov_state_changing_direct_sigma_mS_cm")
    assert hasattr(row_result, "markov_state_changing_corrector_sigma_mS_cm")
    assert hasattr(row_result, "onsager_edge_count")
    assert hasattr(row_result, "top_onsager_friction_edges")


def test_no_double_counting_between_onsager_and_markov_direct():
    audit_options = default_molecular_property_db_audit_options()
    registry_source = MolecularPropertyDbRegistrySource(
        solvent_registry=SOLVENTS,
        salt_registry=SALTS,
        additive_registry=ADDITIVES,
        cation_registry=CATION_PROPERTIES,
    )
    case_selection = build_molecular_property_db_case_selection(
        tuple(DATA[:1]),
        registry_source,
        audit_options,
    )
    primitive_parameters = replace(
        configured_conductivity_primitive_parameters(),
        cross_relaxation_scale=1.0e-30,
    )
    audit_result = audit_molecular_property_db_cases(
        case_selection.cases,
        primitive_parameters,
        audit_options,
    )
    row_result = audit_result.rows[0]
    assert np.isclose(
        row_result.direct_sigma_mS_cm,
        row_result.onsager_ne_sigma_mS_cm
        + row_result.markov_state_changing_direct_sigma_mS_cm,
    )
