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


def _association_record_with_cip_energy(shift_J_mol: float) -> dict:
    association_record = load_physical_library(
        PHYSICAL_LIBRARY_ROOT
    ).association_record
    residual_record = association_record["association_residual"]
    state_features = residual_record["state_features"]
    return {
        **association_record,
        "association_residual": {
            **residual_record,
            "state_features": {
                **state_features,
                "pair": {**state_features["pair"], "CIP": shift_J_mol},
            },
        },
    }


def test_missing_required_association_record_fails_validation() -> None:
    records = load_physical_library(PHYSICAL_LIBRARY_ROOT)
    incomplete_records = PhysicalLibraryRecords(
        root=records.root,
        manifest=records.manifest,
        species_records=records.species_records,
        pair_records=records.pair_records,
        mixture_record=records.mixture_record,
        basis_record=records.basis_record,
        transition_record=records.transition_record,
        memory_record=records.memory_record,
        association_record={
            "schema": records.association_record["schema"],
            "association_residual": records.association_record[
                "association_residual"
            ],
        },
        equilibria_record=records.equilibria_record,
    )

    with pytest.raises(KeyError, match="state_resolved_born"):
        validate_physical_library_records(incomplete_records)


def test_quarter_rt_cip_association_operator_has_required_energy_response() -> None:
    state_key = (
        "CIP",
        "solvent_only",
        "none",
        "PF6-:anion_localized",
        "free_rotating",
        "LiA",
        "partner_free",
        "identity_localized",
        "hop_localized",
        "cage_relaxed",
        "environment",
        "atmosphere_relaxed",
    )
    perturbation_J_mol = 0.25 * R * T_REF_K
    favorable_record = _association_record_with_cip_energy(-perturbation_J_mol)
    unfavorable_record = _association_record_with_cip_energy(perturbation_J_mol)

    favorable_energy = generator_construction._state_feature_sum(
        favorable_record["association_residual"],
        state_key,
        "population_operator_missing",
    )
    unfavorable_energy = generator_construction._state_feature_sum(
        unfavorable_record["association_residual"],
        state_key,
        "population_operator_missing",
    )

    assert favorable_energy == pytest.approx(-perturbation_J_mol)
    assert unfavorable_energy == pytest.approx(perturbation_J_mol)
