"""Reaction-driven equilibrium speciation for conductivity recipes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from constants import R

Array = np.ndarray


@dataclass(frozen=True)
class EquilibriumSpeciesConcentration:
    name: str
    concentration_mol_m3: float


@dataclass(frozen=True)
class SpeciationEquilibriumResult:
    species: tuple[EquilibriumSpeciesConcentration, ...]
    conserved_component_names: tuple[str, ...]
    conserved_component_totals_mol_m3: Array
    component_balance_residuals_mol_m3: Array
    electroneutrality_residual_mol_m3: float
    mass_action_residuals: Array


def solve_speciation_equilibrium(
    recipe_concentrations_mol_m3: dict[str, float],
    species_charges_e: dict[str, float],
    equilibrium_record: dict,
    temperature_K: float,
) -> SpeciationEquilibriumResult:
    """Resolve recipe loadings into equilibrium species in logarithmic concentration."""
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive and finite")
    standard_concentration_mol_m3 = _positive_float(
        equilibrium_record["standard_concentration_mol_m3"], "standard concentration"
    )
    relative_tolerance = _positive_float(
        equilibrium_record["relative_residual_tolerance"], "relative residual tolerance"
    )
    solver_tolerance_fraction = _positive_float(
        equilibrium_record["solver_tolerance_fraction"], "solver tolerance fraction"
    )
    if solver_tolerance_fraction >= 1.0:
        raise ValueError("solver_tolerance_fraction must be less than one")
    solver_tolerance = relative_tolerance * solver_tolerance_fraction
    recipe_formulas = equilibrium_record["recipe_component_formulas"]
    species_formulas = equilibrium_record["equilibrium_species_formulas"]
    recipe_names = tuple(sorted(recipe_concentrations_mol_m3))
    missing_records = tuple(name for name in recipe_names if name not in recipe_formulas)
    if missing_records:
        raise KeyError(f"equilibria missing active recipe records: {missing_records}")
    component_names = tuple(
        sorted({component for name in recipe_names for component in recipe_formulas[name]})
    )
    species_names = tuple(
        sorted(
            name
            for name, formula in species_formulas.items()
            if set(formula).issubset(component_names)
        )
    )
    component_matrix = _stoichiometry_matrix(
        species_names, component_names, species_formulas, "equilibrium species", True
    )
    recipe_matrix = _stoichiometry_matrix(
        recipe_names, component_names, recipe_formulas, "recipe components", False
    )
    recipe_values = np.asarray(
        [recipe_concentrations_mol_m3[name] for name in recipe_names], dtype=float
    )
    if np.any(~np.isfinite(recipe_values)) or np.any(recipe_values <= 0.0):
        raise ValueError("equilibrium recipe concentrations must be positive and finite")
    component_totals = recipe_matrix @ recipe_values
    reaction_matrix, log_constants = _reaction_system(
        equilibrium_record["reactions"], species_names, temperature_K
    )
    charges = np.asarray([species_charges_e[name] for name in species_names], dtype=float)
    concentration_scale = float(np.max(component_totals))

    def residual(log_concentrations: Array) -> Array:
        concentrations = standard_concentration_mol_m3 * np.exp(log_concentrations)
        balances = (component_matrix @ concentrations - component_totals) / concentration_scale
        mass_action = reaction_matrix.T @ log_concentrations - log_constants
        charge = np.asarray([float(charges @ concentrations) / concentration_scale])
        return np.concatenate((balances, mass_action, charge))

    initial_value = np.log(
        concentration_scale / len(species_names) / standard_concentration_mol_m3
    )
    solution = least_squares(
        residual,
        np.full(len(species_names), initial_value),
        xtol=solver_tolerance,
        ftol=solver_tolerance,
        gtol=solver_tolerance,
        max_nfev=int(equilibrium_record["maximum_function_evaluations"]),
    )
    concentrations = standard_concentration_mol_m3 * np.exp(solution.x)
    component_residuals = component_matrix @ concentrations - component_totals
    mass_action_residuals = reaction_matrix.T @ solution.x - log_constants
    charge_residual = float(charges @ concentrations)
    maximum_residual = float(np.max(np.abs(residual(solution.x))))
    if not solution.success or maximum_residual > relative_tolerance:
        raise RuntimeError(
            "speciation equilibrium failed: "
            f"status={solution.status}, maximum_scaled_residual={maximum_residual}, "
            f"component_residuals_mol_m3={component_residuals.tolist()}, "
            f"mass_action_residuals={mass_action_residuals.tolist()}, "
            f"electroneutrality_residual_mol_m3={charge_residual}"
        )
    return SpeciationEquilibriumResult(
        species=tuple(
            EquilibriumSpeciesConcentration(name=name, concentration_mol_m3=float(value))
            for name, value in zip(species_names, concentrations, strict=True)
        ),
        conserved_component_names=component_names,
        conserved_component_totals_mol_m3=component_totals,
        component_balance_residuals_mol_m3=component_residuals,
        electroneutrality_residual_mol_m3=charge_residual,
        mass_action_residuals=mass_action_residuals,
    )


def _stoichiometry_matrix(
    item_names: tuple[str, ...],
    component_names: tuple[str, ...],
    formulas: dict,
    label: str,
    require_full_component_rank: bool,
) -> Array:
    matrix = np.zeros((len(component_names), len(item_names)))
    for item_index, item_name in enumerate(item_names):
        formula = formulas[item_name]
        if not isinstance(formula, dict) or not formula:
            raise ValueError(f"{label} formula for {item_name} must be non-empty")
        for component_name, coefficient in formula.items():
            matrix[component_names.index(component_name), item_index] = _positive_float(
                coefficient, f"{label}.{item_name}.{component_name}"
            )
    if require_full_component_rank and np.linalg.matrix_rank(matrix) != len(component_names):
        raise ValueError(f"{label} have rank-deficient component coverage")
    return matrix


def _reaction_system(
    reaction_records: list[dict], species_names: tuple[str, ...], temperature_K: float
) -> tuple[Array, Array]:
    reaction_vectors = []
    log_constants = []
    for record in reaction_records:
        stoichiometry = record["stoichiometry"]
        if not set(stoichiometry).issubset(species_names):
            continue
        reaction_vector = np.zeros(len(species_names))
        for species_name, coefficient in stoichiometry.items():
            reaction_vector[species_names.index(species_name)] = float(coefficient)
        reference_temperature_K = _positive_float(
            record["reference_temperature_K"], f"{record['id']}.reference_temperature_K"
        )
        equilibrium_constant = _positive_float(
            record["equilibrium_constant_at_reference"],
            f"{record['id']}.equilibrium_constant_at_reference",
        )
        enthalpy_J_mol = float(record["reaction_enthalpy_J_mol"])
        if not np.isfinite(enthalpy_J_mol):
            raise ValueError(f"{record['id']}.reaction_enthalpy_J_mol must be finite")
        reaction_vectors.append(reaction_vector)
        log_constants.append(
            np.log(equilibrium_constant)
            - enthalpy_J_mol
            / R
            * (1.0 / temperature_K - 1.0 / reference_temperature_K)
        )
    if not reaction_vectors:
        return np.zeros((len(species_names), 0)), np.zeros(0)
    reaction_matrix = np.column_stack(reaction_vectors)
    if np.linalg.matrix_rank(reaction_matrix) != len(reaction_vectors):
        raise ValueError("active equilibrium reactions are linearly dependent")
    return reaction_matrix, np.asarray(log_constants)


def _positive_float(value, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return number
