from __future__ import annotations

from pathlib import Path

import pytest

from constants import R, T_REF_K
from conductivity.physical_library import generator_construction
from conductivity.physical_library.library_io import (
    PhysicalLibraryRecords,
    load_physical_library,
    validate_physical_library_records,
)

PHYSICAL_LIBRARY_ROOT = Path("conductivity/physical_library")


def _records_with_association(association_record: dict) -> PhysicalLibraryRecords:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    return PhysicalLibraryRecords(
        root=records.root,
        manifest=records.manifest,
        species_records=records.species_records,
        pair_records=records.pair_records,
        mixture_record=records.mixture_record,
        basis_record=records.basis_record,
        transition_record=records.transition_record,
        memory_record=records.memory_record,
        association_record=association_record,
        equilibria_record=records.equilibria_record,
    )


def test_production_association_coefficients_are_exact_rt_initializations() -> None:
    residual = load_physical_library(PHYSICAL_LIBRARY_ROOT).association_record["association_residual"]
    assert residual["parameter_provenance"] == "initialized_estimate"
    assert residual["initialization_basis"] == "user_authorized_pre_validation_physical_initialization"
    expected = {
        ("pair", "CIP"): -0.25,
        ("pair", "SSIP"): -0.125,
        ("pair", "addSSIP"): -0.1875,
        ("cluster", "Li2A_positive"): 3.0,
        ("cluster", "LiA2_negative"): 3.0,
        ("cluster", "Li2A2_neutral"): 4.0,
        ("cluster", "bridge_network"): 5.0,
        ("ligand", "monodentate"): 1.0,
        ("ligand", "multidentate"): 2.0,
        ("ligand", "additive_separator"): 1.5,
    }
    for (feature_name, state_value), multiplier in expected.items():
        assert residual["state_features"][feature_name][state_value] == multiplier * R * T_REF_K


def test_production_aggregate_topology_inventory_is_explicit() -> None:
    topologies = load_physical_library(PHYSICAL_LIBRARY_ROOT).association_record["aggregate_topologies"]
    assert topologies["Li2A_positive"]["component_stoichiometry"] == {"Li": 2, "A": 1, "ligand": 0}
    assert topologies["LiA2_negative"]["net_formal_charge_e"] == -1
    assert topologies["Li2A2_neutral"]["graph_edges"] == [["Li0", "A0"], ["A0", "Li1"], ["Li1", "A1"]]
    assert topologies["bridge_network"]["graph_edges"] == [["Li0", "A0"], ["A0", "Li1"], ["Li1", "A1"], ["A1", "Li0"]]


def test_missing_required_state_free_energy_operator_fails_validation() -> None:
    production = load_physical_library(PHYSICAL_LIBRARY_ROOT).association_record
    residual = production["association_residual"]
    cluster = {key: value for key, value in residual["state_features"]["cluster"].items() if key != "Li2A2_neutral"}
    association_record = {**production, "association_residual": {**residual, "state_features": {**residual["state_features"], "cluster": cluster}}}
    with pytest.raises(KeyError, match="missing_state_free_energy_operator.*Li2A2_neutral"):
        validate_physical_library_records(_records_with_association(association_record))


def test_initialized_estimate_requires_authorized_basis() -> None:
    production = load_physical_library(PHYSICAL_LIBRARY_ROOT).association_record
    residual = production["association_residual"]
    association_record = {**production, "association_residual": {**residual, "initialization_basis": "conductivity_fit"}}
    with pytest.raises(ValueError, match="user-authorized initialization_basis"):
        validate_physical_library_records(_records_with_association(association_record))


def test_production_born_schema_rejects_nonfractional_occlusion() -> None:
    production = load_physical_library(PHYSICAL_LIBRARY_ROOT).association_record
    born = production["state_resolved_born"]
    pair = {**born["state_features"]["pair"], "CIP": 1.01}
    association_record = {**production, "state_resolved_born": {**born, "state_features": {**born["state_features"], "pair": pair}}}
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        validate_physical_library_records(_records_with_association(association_record))


def test_production_operator_is_conditioned_on_pair_and_cluster() -> None:
    residual = load_physical_library(PHYSICAL_LIBRARY_ROOT).association_record["association_residual"]
    state_key = ("CIP", "solvent_only", "none", "PF6-:anion_localized", "free_rotating", "Li2A_positive", "partner_free", "identity_localized", "hop_localized", "cage_relaxed", "environment", "atmosphere_relaxed")
    energy_J_mol = generator_construction._state_feature_sum(residual, state_key, "missing_state_free_energy_operator")
    assert energy_J_mol == (-0.25 + 3.0) * R * T_REF_K
