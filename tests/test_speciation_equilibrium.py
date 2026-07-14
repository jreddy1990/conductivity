from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from constants import T_REF_K
from conductivity.physical_library import generator_construction, library_io
from conductivity.physical_library.library_io import build_recipe_library_context
from conductivity.physical_library.speciation_equilibrium import solve_speciation_equilibrium

LIBRARY_ROOT = Path("conductivity/physical_library")


def test_lidfob_recipe_resolves_conserved_loading_into_equilibrium_species() -> None:
    context = build_recipe_library_context(
        LIBRARY_ROOT
        / "canonical_recipes"
        / "ec_dmc_34_66_lipf6_0p8_lifsi_0p3_tpp_2wt_ps_1wt_vc_0p5wt_lidfob_0p5wt_298K.yaml",
        LIBRARY_ROOT,
    )
    conserved = {component.name: component.concentration_mol_m3 for component in context.conserved_components}
    resolved = {species.name: species.concentration_mol_m3 for species in context.resolved_species}

    assert 0.0 < resolved["LiDFOB"] < conserved["LiDFOB"]
    assert resolved["DFOB-"] == pytest.approx(conserved["LiDFOB"] - resolved["LiDFOB"])
    assert resolved["Li+"] == pytest.approx(conserved["Li+"] + resolved["DFOB-"])
    assert np.max(np.abs(context.speciation_equilibrium.component_balance_residuals_mol_m3)) < 1.0e-6
    assert abs(context.speciation_equilibrium.electroneutrality_residual_mol_m3) < 1.0e-6
    assert np.max(np.abs(context.speciation_equilibrium.mass_action_residuals)) < 1.0e-9


def test_solver_is_reaction_driven_without_species_name_branches() -> None:
    record = {
        "standard_concentration_mol_m3": 1000.0,
        "relative_residual_tolerance": 1.0e-10,
        "solver_tolerance_fraction": 0.01,
        "maximum_function_evaluations": 1000,
        "recipe_component_formulas": {"feed": {"positive": 1.0, "negative": 1.0}},
        "equilibrium_species_formulas": {
            "undissociated": {"positive": 1.0, "negative": 1.0},
            "mobile_positive": {"positive": 1.0},
            "mobile_negative": {"negative": 1.0},
        },
        "reactions": [
            {
                "id": "generic_dissociation",
                "stoichiometry": {
                    "undissociated": -1.0,
                    "mobile_positive": 1.0,
                    "mobile_negative": 1.0,
                },
                "equilibrium_constant_at_reference": 1.0,
                "reference_temperature_K": 300.0,
                "reaction_enthalpy_J_mol": 0.0,
            }
        ],
    }
    result = solve_speciation_equilibrium(
        recipe_concentrations_mol_m3={"feed": 1000.0},
        species_charges_e={
            "undissociated": 0.0,
            "mobile_positive": 1.0,
            "mobile_negative": -1.0,
        },
        equilibrium_record=record,
        temperature_K=300.0,
    )
    concentrations = {species.name: species.concentration_mol_m3 for species in result.species}
    assert concentrations["mobile_positive"] == pytest.approx(concentrations["mobile_negative"])
    assert concentrations["mobile_positive"] ** 2 == pytest.approx(
        1000.0 * concentrations["undissociated"]
    )


def test_active_recipe_component_requires_equilibrium_record() -> None:
    record = yaml.safe_load((LIBRARY_ROOT / "equilibria.yaml").read_text())
    with pytest.raises(KeyError, match="missing active recipe records"):
        solve_speciation_equilibrium(
            recipe_concentrations_mol_m3={"undeclared_salt": 1000.0},
            species_charges_e={"Li+": 1.0},
            equilibrium_record=record,
            temperature_K=T_REF_K,
        )


def test_library_validation_requires_strict_solver_tolerance_fraction() -> None:
    context = build_recipe_library_context(
        LIBRARY_ROOT / "canonical_recipes" / "ec_dmc_3_7_lipf6_1m_298K.yaml",
        LIBRARY_ROOT,
    )
    equilibria_record = yaml.safe_load((LIBRARY_ROOT / "equilibria.yaml").read_text())
    missing_fraction_record = {
        key: value
        for key, value in equilibria_record.items()
        if key != "solver_tolerance_fraction"
    }
    with pytest.raises(KeyError, match="solver_tolerance_fraction"):
        library_io._validate_equilibria_record(
            missing_fraction_record,
            context.library_records.species_records,
        )

    invalid_fraction_record = {
        **equilibria_record,
        "solver_tolerance_fraction": 1.0,
    }
    with pytest.raises(ValueError, match="must be between zero and one"):
        library_io._validate_equilibria_record(
            invalid_fraction_record,
            context.library_records.species_records,
        )


def test_neutral_salt_reservoir_does_not_activate_ligand_coordinate() -> None:
    context = build_recipe_library_context(
        LIBRARY_ROOT / "canonical_recipes" / "ec_dmc_3_7_lipf6_1m_298K.yaml",
        LIBRARY_ROOT,
    )
    coordinate_values, coordinate_weights = (
        generator_construction._ligand_coordination_nodes(
            context.library_records,
            generator_construction.ReducedCoordinate.LI_LIGAND_COORDINATION,
            SimpleNamespace(additive_weight_fractions={}),
        )
    )

    assert coordinate_values == pytest.approx(np.asarray([0.0]))
    assert coordinate_weights == pytest.approx(np.asarray([1.0]))
