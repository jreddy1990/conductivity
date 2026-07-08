"""Validate projected analytical conductivity on property DB projected inputs."""

from __future__ import annotations

import numpy as np

from conductivity import projected_analytical_conductivity
from data.electrolyte_property_db import DATA
from utils.strict_validation import require_key, strict_mapping, strict_positive_float

PROPERTY_DB_WORST_ROW_COUNT = 5  # Report the five largest residual rows.


def _evaluate_property_db_row(row_index: int, row):
    row_mapping = strict_mapping(row, f"DATA[{row_index}]")
    properties = strict_mapping(
        require_key(row_mapping, "properties", f"DATA[{row_index}]"),
        f"DATA[{row_index}].properties",
    )
    empirical_conductivity_mS_cm = strict_positive_float(
        require_key(properties, "conductivity_mS_cm", f"DATA[{row_index}].properties"),
        f"DATA[{row_index}].properties.conductivity_mS_cm",
    )
    try:
        projected_result = _evaluate_projected_property_payload(row_index, row_mapping)
    except Exception as exc:
        return {
            "row_id": row_index,
            "empirical_mS_cm": empirical_conductivity_mS_cm,
            "predicted_mS_cm": None,
            "residual_mS_cm": None,
            "failure": str(exc),
        }

    predicted_conductivity_mS_cm = float(projected_result.sigma_mS_cm)
    return {
        "row_id": row_index,
        "empirical_mS_cm": empirical_conductivity_mS_cm,
        "predicted_mS_cm": predicted_conductivity_mS_cm,
        "residual_mS_cm": predicted_conductivity_mS_cm
        - empirical_conductivity_mS_cm,
        "failure": None,
    }


def _evaluate_projected_property_payload(row_index: int, row):
    properties = strict_mapping(
        require_key(row, "properties", f"DATA[{row_index}]"),
        f"DATA[{row_index}].properties",
    )
    if "projected_primitives" in properties:
        primitive_inputs = strict_mapping(
            properties["projected_primitives"],
            f"DATA[{row_index}].properties.projected_primitives",
        )
        return projected_analytical_conductivity.compute_projected_analytical_conductivity_from_primitives(
            require_key(
                primitive_inputs,
                "state_concentrations_mol_m3",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
            require_key(
                primitive_inputs,
                "symmetric_capacity_fluxes_K_ij_mol_m3_s",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
            require_key(
                primitive_inputs,
                "transition_first_moments_d_ij_m",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
            require_key(
                primitive_inputs,
                "transition_second_moments_M_ij_m2",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
            require_key(
                primitive_inputs,
                "self_current_tensors_D_self_i_m2_s",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
            require_key(
                primitive_inputs,
                "mori_memory_matrix_A",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
            require_key(
                primitive_inputs,
                "mori_current_coupling_matrix_h",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
            require_key(
                primitive_inputs,
                "temperature_K",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
            require_key(
                primitive_inputs,
                "volume_m3",
                f"DATA[{row_index}].properties.projected_primitives",
            ),
        )
    if "projected_generator_inputs" in properties:
        generator_inputs = strict_mapping(
            properties["projected_generator_inputs"],
            f"DATA[{row_index}].properties.projected_generator_inputs",
        )
        context = f"DATA[{row_index}].properties.projected_generator_inputs"
        return projected_analytical_conductivity.compute_projected_analytical_conductivity(
            require_key(generator_inputs, "potential_energy_J_mol", context),
            require_key(generator_inputs, "mobility_tensor_m2_s", context),
            require_key(generator_inputs, "charge_polarization_gradient", context),
            require_key(generator_inputs, "memory_coordinate_gradient", context),
            require_key(generator_inputs, "basin_quadrature_points", context),
            require_key(generator_inputs, "basin_quadrature_weights", context),
            require_key(generator_inputs, "transition_pair_indices", context),
            require_key(generator_inputs, "transition_quadrature_points", context),
            require_key(generator_inputs, "transition_quadrature_weights", context),
            require_key(generator_inputs, "transition_committor_gradients", context),
            require_key(generator_inputs, "transition_surface_state_indices", context),
            require_key(generator_inputs, "transition_path_displacements_m", context),
            require_key(generator_inputs, "transition_path_weights", context),
            require_key(
                generator_inputs,
                "total_component_concentrations_mol_m3",
                context,
            ),
            require_key(generator_inputs, "basin_stoichiometry", context),
            require_key(generator_inputs, "temperature_K", context),
            require_key(generator_inputs, "volume_m3", context),
            require_key(generator_inputs, "self_current_coordinate_projectors", context),
        )
    raise ValueError(
        f"DATA[{row_index}] is missing projected_primitives or "
        "projected_generator_inputs; recipe-only conductivity validation requires a "
        "populated full ConductivityPhysicalLibrary"
    )


def _labeled_rows() -> list[tuple[int, dict]]:
    labeled_rows = []
    for row_index, row in enumerate(DATA):
        row_mapping = strict_mapping(row, f"DATA[{row_index}]")
        properties = strict_mapping(
            require_key(row_mapping, "properties", f"DATA[{row_index}]"),
            f"DATA[{row_index}].properties",
        )
        if "conductivity_mS_cm" in properties:
            labeled_rows.append((row_index, row_mapping))
    return labeled_rows


def _formulation_key(row) -> tuple:
    recipe = strict_mapping(require_key(row, "recipe", "row"), "row.recipe")
    return tuple(
        (
            phase_name,
            tuple(
                sorted(
                    strict_mapping(phase_mapping, f"row.recipe.{phase_name}").items()
                )
            ),
        )
        for phase_name, phase_mapping in sorted(recipe.items())
    )


def _print_successful_predictions(successful_results: list[dict]) -> None:
    print("projected_analytical_property_db_predictions")
    for result in successful_results:
        print(
            "row_id={row_id} empirical_mS_cm={empirical:.9g} "
            "predicted_mS_cm={predicted:.9g} residual_mS_cm={residual:.9g}".format(
                row_id=result["row_id"],
                empirical=result["empirical_mS_cm"],
                predicted=result["predicted_mS_cm"],
                residual=result["residual_mS_cm"],
            )
        )


def _print_metrics(row_results: list[dict], labeled_rows: list[tuple[int, dict]]) -> None:
    successful_results = [row for row in row_results if row["failure"] is None]
    failed_results = [row for row in row_results if row["failure"] is not None]
    formulation_keys = {_formulation_key(row) for _, row in labeled_rows}
    print(f"source_labeled_rows={len(labeled_rows)}")
    print(f"formulation_group_count={len(formulation_keys)}")
    print(f"evaluated_rows={len(successful_results)}")
    print(f"failed_rows={len(failed_results)}")
    if successful_results:
        _print_successful_predictions(successful_results)
        empirical = np.asarray(
            [row["empirical_mS_cm"] for row in successful_results],
            dtype=float,
        )
        predicted = np.asarray(
            [row["predicted_mS_cm"] for row in successful_results],
            dtype=float,
        )
        residual = predicted - empirical
        print(f"mae_mS_cm={float(np.mean(np.abs(residual))):.9g}")
        print(f"rmse_mS_cm={float(np.sqrt(np.mean(residual**2))):.9g}")
        print(f"bias_mS_cm={float(np.mean(residual)):.9g}")
        if len(successful_results) > 1:
            pearson = float(np.corrcoef(empirical, predicted)[0, 1])
            print(f"pearson_r={pearson:.9g}")
        print(f"maximum_abs_residual_mS_cm={float(np.max(np.abs(residual))):.9g}")
        print("worst_rows")
        worst_indices = np.argsort(np.abs(residual))[::-1][
            :PROPERTY_DB_WORST_ROW_COUNT
        ]
        for result_index in worst_indices:
            row = successful_results[int(result_index)]
            print(
                "row_id={row_id} empirical_mS_cm={empirical:.9g} "
                "predicted_mS_cm={predicted:.9g} residual_mS_cm={residual:.9g}".format(
                    row_id=row["row_id"],
                    empirical=row["empirical_mS_cm"],
                    predicted=row["predicted_mS_cm"],
                    residual=row["residual_mS_cm"],
                )
            )
    if failed_results:
        print("failed_row_details")
        for row in failed_results:
            print(f"row_id={row['row_id']} failure={row['failure']}")


def validate_projected_analytical_property_db() -> list[dict]:
    labeled_rows = _labeled_rows()
    row_results = [
        _evaluate_property_db_row(row_index, row) for row_index, row in labeled_rows
    ]
    _print_metrics(row_results, labeled_rows)
    return row_results


def main() -> None:
    validate_projected_analytical_property_db()


if __name__ == "__main__":
    main()
