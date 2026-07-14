"""Extract projected conductivity primitives from a LAMMPS trajectory dump."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

import numpy as np
import yaml

from constants import N_A
from data.species_data import SALTS
from electrolyte_model import ElectrolyteRecipeModel
from conductivity.physical_library.projected_analytical_conductivity import (
    CARTESIAN,
    POISSON_SOLVABILITY_ABS_TOL,
    POISSON_SOLVABILITY_EPSILON_FACTOR,
    PROJECTED_REFERENCE_VOLUME_M3,
    basis_refinement_as_effect_attribution,
    compute_projected_analytical_conductivity_from_primitives,
    primitive_prediction_readiness_as_effect_attribution,
)
from conductivity.physical_library import generator_construction
from conductivity.physical_library.microscopic_convergence import (
    microscopic_helfand_moment_C_m,
)
from conductivity.physical_library.library_io import (
    RecipeBuildResult,
    build_recipe_library_context_from_record,
)
from conductivity.physical_library.mixture_closures import compute_mixture_closures
from conductivity.physical_library.physical_objects import PairBasin, SiteConfiguration
from conductivity.physical_library.projected_primitives_io import (
    PRIMITIVE_SCHEMA,
    PROJECTED_READOUT_DIRECT_ONLY,
    PROJECTED_READOUT_SUCCEEDED,
    _persist_prediction_readiness_diagnostics,
    _projected_readout_status_from_result,
    write_failed_projected_primitive_yaml,
)
from conductivity.physical_library.trajectory_primitives import (
    ANGSTROM_TO_M,
    FiniteProcessComponentDriftResidual,
    FiniteProcessEdgeDriftContribution,
    ProjectedGeneratorPrimitiveSet,
    TrajectoryMarkovAdditiveSampleInput,
    diagnose_finite_process_legality,
    project_sampled_trajectory_to_generator_primitives,
    refine_trajectory_basis_from_state_current_samples,
)


LAMMPS_COLUMN_ID = 0
LAMMPS_COLUMN_MOLECULE_ID = 1
LAMMPS_COLUMN_CHARGE_E = 2
LAMMPS_POSITION_COLUMN_START = 3
LAMMPS_POSITION_COLUMN_STOP = LAMMPS_POSITION_COLUMN_START + CARTESIAN
LAMMPS_VELOCITY_COLUMN_START = LAMMPS_POSITION_COLUMN_STOP
LAMMPS_VELOCITY_COLUMN_STOP = LAMMPS_VELOCITY_COLUMN_START + CARTESIAN
LAMMPS_REQUIRED_COLUMNS = ("id", "mol", "q", "xu", "yu", "zu")
LAMMPS_MICROSCOPIC_COLUMNS = (
    "id",
    "mol",
    "q",
    "xu",
    "yu",
    "zu",
    "vx",
    "vy",
    "vz",
)
BOX_BOUND_LOW_COLUMN = 0
BOX_BOUND_HIGH_COLUMN = 1
MINIMUM_LOCAL_MINIMUM_COUNT = 2
MINIMUM_COUNTERION_COUNT_FOR_AGGREGATE = 2
ROLE_CATION = "cation"
ROLE_ANION = "anion"
ROLE_NEUTRAL = "neutral"
SCHEMA_NAME = "projected_primitives_from_lammps_trajectory_v1"


class LammpsDumpHeader(Enum):
    TIMESTEP = "ITEM: TIMESTEP"
    ATOM_COUNT = "ITEM: NUMBER OF ATOMS"
    BOX_BOUNDS_PREFIX = "ITEM: BOX BOUNDS"
    ATOMS_PREFIX = "ITEM: ATOMS"


@dataclass(frozen=True)
class LammpsDumpFrame:
    timestep: int
    box_bounds_A: np.ndarray
    atom_table: np.ndarray
    atom_columns: tuple[str, ...]


@dataclass(frozen=True)
class SpeciesRange:
    name: str
    first_molecule_id: int
    last_molecule_id: int
    role: str
    formal_charge_e: float


@dataclass(frozen=True)
class ChargedCenterCatalog:
    molecule_ids: np.ndarray
    species_labels: tuple[str, ...]
    roles: tuple[str, ...]
    formal_charges_e: np.ndarray


@dataclass(frozen=True)
class ChargedCenterFrame:
    positions_A: np.ndarray
    wrapped_positions_A: np.ndarray
    box_bounds_A: np.ndarray


@dataclass(frozen=True)
class MolecularEnvironmentCatalog:
    molecule_ids: np.ndarray
    species_labels: tuple[str, ...]


@dataclass(frozen=True)
class MolecularEnvironmentFrame:
    positions_A: np.ndarray
    wrapped_positions_A: np.ndarray
    orientation_vectors: np.ndarray
    box_bounds_A: np.ndarray


@dataclass(frozen=True)
class AssociationThresholds:
    contact_pair_max_distance_A: float
    solvent_separated_pair_max_distance_A: float


def charge_helfand_series_from_lammps_dump(
    trajectory_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the all-atom partial-charge Helfand moment used by current.dat."""
    frames = _read_lammps_custom_dump(trajectory_path)
    helfand_moments_C_m = np.asarray(
        [
            microscopic_helfand_moment_C_m(
                frame.atom_table[:, LAMMPS_COLUMN_CHARGE_E],
                frame.atom_table[
                    :,
                    LAMMPS_POSITION_COLUMN_START:LAMMPS_POSITION_COLUMN_STOP,
                ],
            )
            for frame in frames
        ],
        dtype=float,
    )
    box_volumes_m3 = np.asarray(
        [
            np.prod(frame.box_bounds_A[:, BOX_BOUND_HIGH_COLUMN] - frame.box_bounds_A[:, BOX_BOUND_LOW_COLUMN])
            * ANGSTROM_TO_M**CARTESIAN
            for frame in frames
        ],
        dtype=float,
    )
    return helfand_moments_C_m, box_volumes_m3


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project a LAMMPS trajectory.lammpstrj dump into conductivity "
            "primitive tensors and evaluate the full projected readout."
        )
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--composition-json", required=True, type=Path)
    parser.add_argument("--copies-json", required=True, type=Path)
    parser.add_argument("--physical-library-root", required=True, type=Path)
    parser.add_argument("--dt-fs", required=True, type=float)
    parser.add_argument("--trajectory-dump-stride-steps", required=True, type=int)
    parser.add_argument("--stationary-start-frame-index", required=True, type=int)
    parser.add_argument("--output-yaml", required=True, type=Path)
    args = parser.parse_args()

    extract_projected_primitives_from_lammps_dump(
        trajectory_path=args.trajectory,
        composition_json_path=args.composition_json,
        copies_json_path=args.copies_json,
        physical_library_root=args.physical_library_root,
        timestep_fs=float(args.dt_fs),
        trajectory_dump_stride_steps=int(args.trajectory_dump_stride_steps),
        stationary_start_frame_index=int(args.stationary_start_frame_index),
        output_yaml_path=args.output_yaml,
    )
    return 0


def extract_projected_primitives_from_lammps_dump(
    trajectory_path: Path,
    composition_json_path: Path,
    copies_json_path: Path,
    physical_library_root: Path,
    timestep_fs: float,
    trajectory_dump_stride_steps: int,
    stationary_start_frame_index: int,
    output_yaml_path: Path,
):
    composition_record = _load_json_mapping(composition_json_path)
    recipe_record = _projected_recipe_record_from_composition_record(composition_record)
    recipe_context = build_recipe_library_context_from_record(
        recipe_record,
        physical_library_root,
    )
    return _extract_projected_primitives_from_lammps_dump_with_context(
        trajectory_path=trajectory_path,
        composition_json_path=composition_json_path,
        copies_json_path=copies_json_path,
        composition_record=composition_record,
        recipe_context=recipe_context,
        timestep_fs=timestep_fs,
        trajectory_dump_stride_steps=trajectory_dump_stride_steps,
        stationary_start_frame_index=stationary_start_frame_index,
        output_yaml_path=output_yaml_path,
    )


def _extract_projected_primitives_from_lammps_dump_with_context(
    trajectory_path: Path,
    composition_json_path: Path,
    copies_json_path: Path,
    composition_record: dict,
    recipe_context: RecipeBuildResult,
    timestep_fs: float,
    trajectory_dump_stride_steps: int,
    stationary_start_frame_index: int,
    output_yaml_path: Path,
):
    if timestep_fs <= 0.0:
        raise ValueError("timestep_fs must be positive")
    if trajectory_dump_stride_steps <= 0:
        raise ValueError("trajectory_dump_stride_steps must be positive")
    if stationary_start_frame_index < 0:
        raise ValueError("stationary_start_frame_index must be nonnegative")
    copies_record = _load_json_mapping(copies_json_path)
    records = recipe_context.library_records
    trajectory_basis_config = records.basis_record["trajectory_basis_refinement"]
    basis_residual_tolerance_m2_s = float(
        trajectory_basis_config["residual_score_tolerance_m2_s"]
    )
    sigma_change_tolerance_S_m = float(
        trajectory_basis_config["conductivity_change_tolerance_S_m"]
    )
    if basis_residual_tolerance_m2_s <= 0.0:
        raise ValueError("basis residual tolerance in basis.yaml must be positive")
    if sigma_change_tolerance_S_m <= 0.0:
        raise ValueError("conductivity change tolerance in basis.yaml must be positive")
    mixture = compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    species_ranges = _species_ranges_from_copies_record(copies_record)
    center_catalog = _charged_center_catalog_from_species_ranges(species_ranges)
    environment_catalog = _molecular_environment_catalog_from_species_ranges(
        species_ranges
    )
    frames = tuple(_read_lammps_custom_dump(trajectory_path))[
        stationary_start_frame_index:
    ]
    if len(frames) < MINIMUM_LOCAL_MINIMUM_COUNT:
        raise ValueError("primitive extraction needs at least two trajectory frames")

    center_frames = tuple(
        _charged_center_frame_from_lammps_frame(frame, center_catalog)
        for frame in frames
    )
    environment_frames = _unwrap_environment_molecular_centers(
        tuple(
            _molecular_environment_frame_from_lammps_frame(
                frame, environment_catalog
            )
            for frame in frames
        )
    )
    thresholds = _association_thresholds_from_physical_library(records)
    (
        observed_state_labels,
        state_index_by_frame_and_center,
        counterion_index_by_frame_and_center,
    ) = _assign_center_states(
        center_frames,
        center_catalog,
        environment_frames,
        environment_catalog,
        thresholds,
        records,
        mixture,
    )
    state_labels, state_index_by_frame_and_center = (
        _merge_transport_equivalent_observed_states(
            observed_state_labels,
            state_index_by_frame_and_center,
        )
    )
    charge_displacements_m = _charge_displacements_by_step_m(
        center_frames,
        center_catalog,
        state_index_by_frame_and_center,
        counterion_index_by_frame_and_center,
        thresholds,
    )
    self_charge_polarizations_m = _self_charge_polarization_by_frame_and_center_m(
        center_frames,
        center_catalog,
        environment_frames,
        environment_catalog,
        counterion_index_by_frame_and_center,
        thresholds,
    )
    mean_volume_m3 = _mean_box_volume_m3(center_frames)
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=state_labels,
        occupancy_state_index_by_observation=state_index_by_frame_and_center.reshape(
            -1
        ),
        from_state_index_by_step=state_index_by_frame_and_center[:-1].reshape(-1),
        to_state_index_by_step=state_index_by_frame_and_center[1:].reshape(-1),
        charge_displacement_by_step_m=charge_displacements_m,
        self_charge_polarization_by_frame_and_center_m=self_charge_polarizations_m,
        state_index_by_frame_and_center=state_index_by_frame_and_center,
        self_current_valid_step_by_center=(
            _state_local_membership_stable_step_mask(
                center_frames,
                center_catalog,
                counterion_index_by_frame_and_center,
                thresholds,
            )
        ),
        transition_commitment_time_s=float(
            records.transition_record["trajectory_projection"]["commitment_time_s"]
        ),
        zero_frequency_integration_window_s=float(
            records.transition_record["trajectory_projection"][
                "zero_frequency_integration_window_s"
            ]
        ),
        zero_frequency_plateau_window_s=float(
            records.transition_record["trajectory_projection"][
                "zero_frequency_plateau_window_s"
            ]
        ),
        dt_s=(
            timestep_fs
            * float(trajectory_dump_stride_steps)
            * _seconds_per_femtosecond()
        ),
        total_transport_concentration_mol_m3=(
            float(center_catalog.molecule_ids.size) / N_A / mean_volume_m3
        ),
        temperature_K=float(composition_record["temperature_K"]),
    )
    primitive_set = project_sampled_trajectory_to_generator_primitives(sample_input)
    failed_diffusion_states = tuple(
        diagnostic.state_label
        for diagnostic in primitive_set.diagnostics.self_diffusion_convergence
        if diagnostic.convergence_status != "converged"
    )
    if failed_diffusion_states:
        failure_reason = (
            "no converged long-time diffusive window for occupied states: "
            + ", ".join(failed_diffusion_states)
        )
        failure_diagnostics = {
            "self_diffusion_readiness_status": "failed",
            "component_drift_residuals": [],
            "self_diffusion_convergence": _self_diffusion_convergence_records(
                primitive_set.diagnostics.self_diffusion_convergence
            ),
        }
        write_failed_projected_primitive_yaml(
            output_yaml_path, PRIMITIVE_SCHEMA, failure_reason, failure_diagnostics
        )
        raise ValueError(failure_reason)
    primitive_arrays = _primitive_arrays_from_projected_set(primitive_set)
    component_drift_violation = _component_drift_violation(
        primitive_set.diagnostics.component_drift_residuals
    )
    diagnostics = _projected_primitive_extraction_diagnostics(
        primitive_set,
        component_drift_violation,
    )
    diagnostics["observed_microstate_count"] = len(observed_state_labels)
    diagnostics["merged_transport_state_count"] = len(state_labels)
    if component_drift_violation:
        failure_reason = _invalid_component_drift_failure_reason(
            primitive_set.diagnostics.component_drift_residuals
        )
        write_failed_projected_primitive_yaml(
            output_yaml_path,
            PRIMITIVE_SCHEMA,
            failure_reason,
            diagnostics,
        )
        raise ValueError(failure_reason)
    direct_result = compute_projected_analytical_conductivity_from_primitives(
        state_concentrations_mol_m3=primitive_arrays["state_concentrations_mol_m3"],
        symmetric_capacity_fluxes_K_ij_mol_m3_s=primitive_arrays[
            "symmetric_capacity_fluxes_K_ij_mol_m3_s"
        ],
        transition_first_moments_d_ij_m=primitive_arrays[
            "transition_first_moments_d_ij_m"
        ],
        transition_second_moments_M_ij_m2=primitive_arrays[
            "transition_second_moments_M_ij_m2"
        ],
        self_current_tensors_D_self_i_m2_s=primitive_arrays[
            "self_current_tensors_D_self_i_m2_s"
        ],
        mori_memory_matrix_A=primitive_arrays["mori_memory_matrix_A"],
        mori_current_coupling_matrix_h=primitive_arrays[
            "mori_current_coupling_matrix_h"
        ],
        temperature_K=float(composition_record["temperature_K"]),
        volume_m3=PROJECTED_REFERENCE_VOLUME_M3,
    )
    trajectory_basis_refinement = refine_trajectory_basis_from_state_current_samples(
        sample_input=sample_input,
        state_labels=state_labels,
        state_index_by_step=state_index_by_frame_and_center[:-1].reshape(-1),
        samples_per_frame=int(center_catalog.molecule_ids.size),
        direct_diffusivity_tensor_m2_s=(
            direct_result.direct_diffusivity_tensor
            - direct_result.finite_state_memory_correction_tensor
        ),
        residual_score_tolerance_m2_s=basis_residual_tolerance_m2_s,
        conductivity_change_tolerance_S_m=sigma_change_tolerance_S_m,
    )
    primitive_arrays["mori_memory_matrix_A"] = (
        trajectory_basis_refinement.final_mori_memory_matrix_A
    )
    primitive_arrays["mori_current_coupling_matrix_h"] = (
        trajectory_basis_refinement.final_mori_current_coupling_matrix_h
    )
    diagnostics["trajectory_basis_refinement"] = {
        "candidate_labels": list(trajectory_basis_refinement.candidate_labels),
        "candidate_count": len(trajectory_basis_refinement.candidate_labels),
        "candidate_sample_count": trajectory_basis_refinement.candidate_sample_count,
        "selected_candidate_indices": list(
            trajectory_basis_refinement.selected_candidate_indices
        ),
        "conductivity_history_S_m": list(
            trajectory_basis_refinement.conductivity_history_S_m
        ),
        "selected_residual_score_history_m2_s": list(
            trajectory_basis_refinement.selected_residual_score_history_m2_s
        ),
        "candidate_set_exhausted": trajectory_basis_refinement.candidate_set_exhausted,
        "convergence_status": trajectory_basis_refinement.convergence_status,
        "not_complete_reasons": list(
            trajectory_basis_refinement.not_complete_reasons
        ),
        "final_maximum_residual_score_m2_s": (
            trajectory_basis_refinement.final_maximum_residual_score_m2_s
        ),
        "residual_score_tolerance_m2_s": (
            trajectory_basis_refinement.residual_score_tolerance_m2_s
        ),
        "final_conductivity_change_abs_S_m": (
            trajectory_basis_refinement.final_conductivity_change_abs_S_m
        ),
        "conductivity_change_tolerance_S_m": (
            trajectory_basis_refinement.conductivity_change_tolerance_S_m
        ),
    }
    artifact = {
        "schema": PRIMITIVE_SCHEMA,
        "source_schema": SCHEMA_NAME,
        "state_labels": tuple(primitive_set.state_labels),
        "temperature_K": float(composition_record["temperature_K"]),
        "volume_m3": float(mean_volume_m3),
        "primitives": {
            "state_concentrations_mol_m3": np.asarray(
                primitive_arrays["state_concentrations_mol_m3"],
                dtype=float,
            ).tolist(),
            "symmetric_capacity_fluxes_K_ij_mol_m3_s": np.asarray(
                primitive_arrays["symmetric_capacity_fluxes_K_ij_mol_m3_s"],
                dtype=float,
            ).tolist(),
            "transition_first_moments_d_ij_m": np.asarray(
                primitive_arrays["transition_first_moments_d_ij_m"],
                dtype=float,
            ).tolist(),
            "transition_second_moments_M_ij_m2": np.asarray(
                primitive_arrays["transition_second_moments_M_ij_m2"],
                dtype=float,
            ).tolist(),
            "self_current_tensors_D_self_i_m2_s": np.asarray(
                primitive_arrays["self_current_tensors_D_self_i_m2_s"],
                dtype=float,
            ).tolist(),
            "mori_memory_matrix_A": np.asarray(
                primitive_arrays["mori_memory_matrix_A"],
                dtype=float,
            ).tolist(),
            "mori_current_coupling_matrix_h": np.asarray(
                primitive_arrays["mori_current_coupling_matrix_h"],
                dtype=float,
            ).tolist(),
        },
        "source_metadata": {
            "tag": composition_record["tag"],
            "species_list": composition_record["species_list"],
            "mole_fractions": composition_record["mole_fractions"],
            "copies_per_species": copies_record["copies_per_species"],
            "timestep_fs": float(timestep_fs),
            "trajectory_dump_stride_steps": int(trajectory_dump_stride_steps),
            "frame_count": int(len(frames)),
            "mean_volume_m3": float(mean_volume_m3),
            "association_thresholds_A": {
                "contact_pair_max_distance_A": thresholds.contact_pair_max_distance_A,
                "solvent_separated_pair_max_distance_A": (
                    thresholds.solvent_separated_pair_max_distance_A
                ),
            },
        },
        "diagnostics": diagnostics,
    }
    projected_result = compute_projected_analytical_conductivity_from_primitives(
        state_concentrations_mol_m3=primitive_arrays["state_concentrations_mol_m3"],
        symmetric_capacity_fluxes_K_ij_mol_m3_s=primitive_arrays[
            "symmetric_capacity_fluxes_K_ij_mol_m3_s"
        ],
        transition_first_moments_d_ij_m=primitive_arrays[
            "transition_first_moments_d_ij_m"
        ],
        transition_second_moments_M_ij_m2=primitive_arrays[
            "transition_second_moments_M_ij_m2"
        ],
        self_current_tensors_D_self_i_m2_s=primitive_arrays[
            "self_current_tensors_D_self_i_m2_s"
        ],
        mori_memory_matrix_A=primitive_arrays["mori_memory_matrix_A"],
        mori_current_coupling_matrix_h=primitive_arrays[
            "mori_current_coupling_matrix_h"
        ],
        temperature_K=float(composition_record["temperature_K"]),
        volume_m3=PROJECTED_REFERENCE_VOLUME_M3,
    )
    projected_result.effect_attribution.update(
        basis_refinement_as_effect_attribution(
            {
                "convergence_status": trajectory_basis_refinement.convergence_status,
                "not_complete_reasons": trajectory_basis_refinement.not_complete_reasons,
                "hard_convergence_failure": (
                    trajectory_basis_refinement.convergence_status != "converged"
                ),
                "final_maximum_residual_score": (
                    trajectory_basis_refinement.final_maximum_residual_score_m2_s
                ),
                "final_conductivity_change_abs_S_m": (
                    trajectory_basis_refinement.final_conductivity_change_abs_S_m
                ),
                "selected_residual_score_history": np.asarray([], dtype=float),
            }
        )
    )
    projected_result.effect_attribution.update(
        primitive_prediction_readiness_as_effect_attribution(
            projected_result.effect_attribution
        )
    )
    artifact["projected_readout_status"] = _projected_readout_status_from_result(
        projected_result
    )
    _persist_prediction_readiness_diagnostics(
        diagnostics, projected_result.effect_attribution
    )
    if artifact["projected_readout_status"] == PROJECTED_READOUT_SUCCEEDED:
        artifact["sigma_mS_cm"] = float(projected_result.sigma_mS_cm)
        artifact["sigma_S_m"] = float(projected_result.sigma_S_m)
    if artifact["projected_readout_status"] == PROJECTED_READOUT_DIRECT_ONLY:
        diagnostics["direct_only_reasons"] = tuple(
            projected_result.effect_attribution[
                "primitive_prediction_not_complete_reasons"
            ]
        )
        diagnostics["primitive_prediction_readiness_status"] = str(
            projected_result.effect_attribution["primitive_prediction_readiness_status"]
        )
        diagnostics["primitive_prediction_scalar_label"] = str(
            projected_result.effect_attribution["primitive_prediction_scalar_label"]
        )
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    output_yaml_path.write_text(yaml.safe_dump(artifact, sort_keys=False))
    return artifact


def _projected_recipe_record_from_composition_record(composition_record: dict) -> dict:
    recipe_model = ElectrolyteRecipeModel.model_validate(
        composition_record["source_recipe"]
    )
    salt_component_molarities_mol_l = _salt_component_molarities_from_formula_loadings(
        recipe_model.salts,
    )
    return {
        "temperature_K": float(composition_record["temperature_K"]),
        "solvents_vv": dict(recipe_model.solvents),
        "salts_mol_l": salt_component_molarities_mol_l,
        "additives_weight_fraction": dict(recipe_model.additives),
    }


def _salt_component_molarities_from_formula_loadings(
    salt_formula_molarities_mol_l,
) -> dict[str, float]:
    component_molarities_mol_l: dict[str, float] = {}
    for salt_formula_name, salt_molarity_mol_l in salt_formula_molarities_mol_l.items():
        if salt_formula_name not in SALTS:
            raise KeyError(
                f"unknown lithium salt formula in source recipe: {salt_formula_name}"
            )
        salt_record = SALTS[salt_formula_name]
        cation_name = str(salt_record["cation"])
        anion_name = str(salt_record["anion"])
        anion_charge = int(salt_record["anion_charge"])
        if cation_name != "Li" or anion_charge != -1:
            raise ValueError(
                f"salt {salt_formula_name} requires unsupported projected "
                f"stoichiometry: cation={cation_name}, anion_charge={anion_charge}"
            )
        _accumulate_component_molarity(
            component_molarities_mol_l,
            "Li+",
            float(salt_molarity_mol_l),
        )
        _accumulate_component_molarity(
            component_molarities_mol_l,
            anion_name,
            float(salt_molarity_mol_l),
        )
    return component_molarities_mol_l


def _accumulate_component_molarity(
    component_molarities_mol_l: dict[str, float],
    component_name: str,
    component_molarity_mol_l: float,
) -> None:
    if component_molarity_mol_l < 0.0:
        raise ValueError(
            f"component {component_name} molarity must be non-negative, "
            f"got {component_molarity_mol_l}"
        )
    if component_name in component_molarities_mol_l:
        component_molarities_mol_l[component_name] += component_molarity_mol_l
        return
    component_molarities_mol_l[component_name] = component_molarity_mol_l


def _load_json_mapping(path: Path):
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return record


def _species_ranges_from_copies_record(copies_record) -> tuple[SpeciesRange, ...]:
    species_ranges = copies_record["species_ranges"]
    charges_per_species = copies_record["charges_per_species"]
    if not isinstance(species_ranges, list):
        raise TypeError("copies.json species_ranges must be a list")
    if not isinstance(charges_per_species, dict):
        raise TypeError("copies.json charges_per_species must be a mapping")
    ranges: list[SpeciesRange] = []
    for species_range in species_ranges:
        if not isinstance(species_range, dict):
            raise TypeError("species_ranges entries must be mappings")
        species_name = str(species_range["name"])
        ranges.append(
            SpeciesRange(
                name=species_name,
                first_molecule_id=int(species_range["first_mol_id"]),
                last_molecule_id=int(species_range["last_mol_id"]),
                role=str(species_range["role"]),
                formal_charge_e=float(charges_per_species[species_name]),
            )
        )
    return tuple(ranges)


def _charged_center_catalog_from_species_ranges(
    species_ranges: tuple[SpeciesRange, ...],
) -> ChargedCenterCatalog:
    molecule_ids: list[int] = []
    species_labels: list[str] = []
    roles: list[str] = []
    formal_charges_e: list[float] = []
    for species_range in species_ranges:
        if species_range.role == ROLE_NEUTRAL:
            continue
        if species_range.role not in (ROLE_CATION, ROLE_ANION):
            raise ValueError(f"unknown species role {species_range.role}")
        for molecule_id in range(
            species_range.first_molecule_id,
            species_range.last_molecule_id + 1,
        ):
            molecule_ids.append(molecule_id)
            species_labels.append(species_range.name)
            roles.append(species_range.role)
            formal_charges_e.append(species_range.formal_charge_e)
    if not molecule_ids:
        raise ValueError("no charged molecule centers found in copies.json")
    return ChargedCenterCatalog(
        molecule_ids=np.asarray(molecule_ids, dtype=int),
        species_labels=tuple(species_labels),
        roles=tuple(roles),
        formal_charges_e=np.asarray(formal_charges_e, dtype=float),
    )


def _molecular_environment_catalog_from_species_ranges(
    species_ranges: tuple[SpeciesRange, ...],
) -> MolecularEnvironmentCatalog:
    molecule_ids = []
    species_labels = []
    for species_range in species_ranges:
        for molecule_id in range(
            species_range.first_molecule_id,
            species_range.last_molecule_id + 1,
        ):
            molecule_ids.append(molecule_id)
            species_labels.append(species_range.name)
    if len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError("molecular environment identities must be unique")
    return MolecularEnvironmentCatalog(
        molecule_ids=np.asarray(molecule_ids, dtype=int),
        species_labels=tuple(species_labels),
    )


def _read_lammps_custom_dump(path: Path) -> tuple[LammpsDumpFrame, ...]:
    frames: list[LammpsDumpFrame] = []
    with path.open() as trajectory_file:
        header_line = trajectory_file.readline()
        while header_line:
            _validate_exact_header_line(header_line, LammpsDumpHeader.TIMESTEP)
            timestep = int(trajectory_file.readline().strip())
            _expect_exact_header(trajectory_file, LammpsDumpHeader.ATOM_COUNT)
            atom_count = int(trajectory_file.readline().strip())
            box_header = trajectory_file.readline().strip()
            if not box_header.startswith(LammpsDumpHeader.BOX_BOUNDS_PREFIX.value):
                raise ValueError(f"expected BOX BOUNDS header, got {box_header}")
            box_bounds_A = np.asarray(
                [
                    _read_box_bound_line(trajectory_file.readline()),
                    _read_box_bound_line(trajectory_file.readline()),
                    _read_box_bound_line(trajectory_file.readline()),
                ],
                dtype=float,
            )
            atoms_header = trajectory_file.readline().strip()
            if not atoms_header.startswith(LammpsDumpHeader.ATOMS_PREFIX.value):
                raise ValueError(f"expected ATOMS header, got {atoms_header}")
            atom_columns = _atom_columns_from_header(atoms_header)
            atom_rows = [
                _read_atom_line(trajectory_file.readline(), len(atom_columns))
                for _atom_index in range(atom_count)
            ]
            atom_table = np.asarray(atom_rows, dtype=float)
            order = np.argsort(atom_table[:, LAMMPS_COLUMN_ID])
            frames.append(
                LammpsDumpFrame(
                    timestep=timestep,
                    box_bounds_A=box_bounds_A,
                    atom_table=atom_table[order],
                    atom_columns=atom_columns,
                )
            )
            header_line = trajectory_file.readline()
    if not frames:
        raise ValueError(f"{path} contained no LAMMPS dump frames")
    return tuple(frames)


def _validate_exact_header_line(
    header_line: str,
    expected_header: LammpsDumpHeader,
) -> None:
    observed_header = header_line.strip()
    if observed_header != expected_header.value:
        raise ValueError(
            f"expected {expected_header.value} header, got {observed_header}"
        )


def _expect_exact_header(trajectory_file, expected_header: LammpsDumpHeader) -> None:
    _validate_exact_header_line(trajectory_file.readline(), expected_header)


def _read_box_bound_line(line: str) -> tuple[float, float]:
    pieces = line.split()
    if len(pieces) < MINIMUM_LOCAL_MINIMUM_COUNT:
        raise ValueError(f"invalid box bound line: {line}")
    return float(pieces[0]), float(pieces[1])


def _atom_columns_from_header(atoms_header: str) -> tuple[str, ...]:
    header_pieces = atoms_header.split()
    atom_columns = tuple(header_pieces[2:])
    if not atom_columns:
        raise ValueError("LAMMPS ATOMS header declares no columns")
    missing_columns = tuple(
        column_name
        for column_name in LAMMPS_REQUIRED_COLUMNS
        if column_name not in atom_columns
    )
    if missing_columns:
        raise ValueError(
            "LAMMPS trajectory is missing required columns: "
            + ", ".join(missing_columns)
        )
    expected_prefix = atom_columns[: len(LAMMPS_REQUIRED_COLUMNS)]
    if expected_prefix != LAMMPS_REQUIRED_COLUMNS:
        raise ValueError(
            "LAMMPS trajectory columns must begin with "
            + " ".join(LAMMPS_REQUIRED_COLUMNS)
        )
    return atom_columns


def _read_atom_line(line: str, column_count: int) -> tuple[float, ...]:
    pieces = line.split()
    if len(pieces) != column_count:
        raise ValueError(f"invalid atom dump row: {line}")
    return tuple(float(piece) for piece in pieces)


def _charged_center_frame_from_lammps_frame(
    frame: LammpsDumpFrame,
    center_catalog: ChargedCenterCatalog,
) -> ChargedCenterFrame:
    molecule_ids = frame.atom_table[:, LAMMPS_COLUMN_MOLECULE_ID].astype(int)
    atom_charges_e = frame.atom_table[:, LAMMPS_COLUMN_CHARGE_E]
    atom_positions_A = frame.atom_table[
        :,
        LAMMPS_POSITION_COLUMN_START:LAMMPS_POSITION_COLUMN_STOP,
    ]
    center_positions_A = np.zeros(
        (center_catalog.molecule_ids.size, CARTESIAN),
        dtype=float,
    )
    for center_index, molecule_id in enumerate(center_catalog.molecule_ids):
        molecule_mask = molecule_ids == molecule_id
        if not np.any(molecule_mask):
            raise ValueError(f"dump frame missing molecule id {molecule_id}")
        molecule_charges_e = atom_charges_e[molecule_mask]
        molecule_positions_A = atom_positions_A[molecule_mask]
        net_charge_e = float(np.sum(molecule_charges_e))
        if abs(net_charge_e) <= np.finfo(float).eps:
            raise ValueError(f"charged molecule {molecule_id} has zero net charge")
        center_positions_A[center_index] = (
            np.einsum("i,ia->a", molecule_charges_e, molecule_positions_A)
            / net_charge_e
        )
    return ChargedCenterFrame(
        positions_A=center_positions_A,
        wrapped_positions_A=_wrap_positions_into_box_A(
            center_positions_A,
            frame.box_bounds_A,
        ),
        box_bounds_A=frame.box_bounds_A,
    )


def _molecular_environment_frame_from_lammps_frame(
    frame: LammpsDumpFrame,
    environment_catalog: MolecularEnvironmentCatalog,
) -> MolecularEnvironmentFrame:
    molecule_ids = frame.atom_table[:, LAMMPS_COLUMN_MOLECULE_ID].astype(int)
    atom_positions_A = frame.atom_table[
        :, LAMMPS_POSITION_COLUMN_START:LAMMPS_POSITION_COLUMN_STOP
    ]
    box_low_A = frame.box_bounds_A[:, BOX_BOUND_LOW_COLUMN]
    box_high_A = frame.box_bounds_A[:, BOX_BOUND_HIGH_COLUMN]
    box_lengths_A = box_high_A - box_low_A
    wrapped_atom_positions_A = box_low_A + np.mod(
        atom_positions_A - box_low_A,
        box_lengths_A,
    )
    positions_A = np.zeros((environment_catalog.molecule_ids.size, CARTESIAN))
    orientations = np.zeros_like(positions_A)
    for molecule_index, molecule_id in enumerate(environment_catalog.molecule_ids):
        molecule_positions_A = wrapped_atom_positions_A[molecule_ids == molecule_id]
        if molecule_positions_A.size == 0:
            raise ValueError(f"dump frame missing molecule id {molecule_id}")
        anchor_position_A = molecule_positions_A[0]
        local_positions_A = molecule_positions_A - anchor_position_A
        local_positions_A -= box_lengths_A * np.round(
            local_positions_A / box_lengths_A
        )
        center_A = anchor_position_A + np.mean(local_positions_A, axis=0)
        positions_A[molecule_index] = box_low_A + np.mod(
            center_A - box_low_A,
            box_lengths_A,
        )
        centered_positions_A = local_positions_A - np.mean(
            local_positions_A,
            axis=0,
        )
        if molecule_positions_A.shape[0] == 1:
            orientations[molecule_index, 0] = 1.0
            continue
        _, _, right_singular_vectors = np.linalg.svd(
            centered_positions_A,
            full_matrices=False,
        )
        orientations[molecule_index] = right_singular_vectors[0]
    return MolecularEnvironmentFrame(
        positions_A=positions_A,
        wrapped_positions_A=_wrap_positions_into_box_A(positions_A, frame.box_bounds_A),
        orientation_vectors=orientations,
        box_bounds_A=frame.box_bounds_A,
    )


def _unwrap_environment_molecular_centers(
    environment_frames: tuple[MolecularEnvironmentFrame, ...],
) -> tuple[MolecularEnvironmentFrame, ...]:
    if not environment_frames:
        raise ValueError("environment frame sequence must not be empty")
    unwrapped_positions_A = np.asarray(
        environment_frames[0].wrapped_positions_A,
        dtype=float,
    ).copy()
    unwrapped_frames = [
        MolecularEnvironmentFrame(
            positions_A=unwrapped_positions_A.copy(),
            wrapped_positions_A=environment_frames[0].wrapped_positions_A,
            orientation_vectors=environment_frames[0].orientation_vectors,
            box_bounds_A=environment_frames[0].box_bounds_A,
        )
    ]
    previous_wrapped_positions_A = np.asarray(
        environment_frames[0].wrapped_positions_A,
        dtype=float,
    )
    for environment_frame in environment_frames[1:]:
        box_lengths_A = (
            environment_frame.box_bounds_A[:, BOX_BOUND_HIGH_COLUMN]
            - environment_frame.box_bounds_A[:, BOX_BOUND_LOW_COLUMN]
        )
        current_wrapped_positions_A = np.asarray(
            environment_frame.wrapped_positions_A,
            dtype=float,
        )
        displacement_A = current_wrapped_positions_A - previous_wrapped_positions_A
        displacement_A -= box_lengths_A * np.round(displacement_A / box_lengths_A)
        unwrapped_positions_A = unwrapped_positions_A + displacement_A
        unwrapped_frames.append(
            MolecularEnvironmentFrame(
                positions_A=unwrapped_positions_A.copy(),
                wrapped_positions_A=current_wrapped_positions_A,
                orientation_vectors=environment_frame.orientation_vectors,
                box_bounds_A=environment_frame.box_bounds_A,
            )
        )
        previous_wrapped_positions_A = current_wrapped_positions_A
    return tuple(unwrapped_frames)


def _wrap_positions_into_box_A(
    positions_A: np.ndarray,
    box_bounds_A: np.ndarray,
) -> np.ndarray:
    box_low_A = box_bounds_A[:, BOX_BOUND_LOW_COLUMN]
    box_high_A = box_bounds_A[:, BOX_BOUND_HIGH_COLUMN]
    box_lengths_A = box_high_A - box_low_A
    return box_low_A + np.mod(positions_A - box_low_A, box_lengths_A)


def _nearest_counterion_distances_A(
    center_frames: tuple[ChargedCenterFrame, ...],
    center_catalog: ChargedCenterCatalog,
) -> np.ndarray:
    cation_indices = _role_indices(center_catalog, ROLE_CATION)
    anion_indices = _role_indices(center_catalog, ROLE_ANION)
    distances_A: list[float] = []
    for center_frame in center_frames:
        pair_distances_A = _cation_anion_distance_matrix_A(
            center_frame,
            cation_indices,
            anion_indices,
        )
        distances_A.extend(float(value) for value in np.min(pair_distances_A, axis=1))
        distances_A.extend(float(value) for value in np.min(pair_distances_A, axis=0))
    return np.asarray(distances_A, dtype=float)


def _association_thresholds_from_distances_A(
    distances_A: np.ndarray,
) -> AssociationThresholds:
    finite_distances_A = np.asarray(distances_A, dtype=float)
    finite_distances_A = finite_distances_A[np.isfinite(finite_distances_A)]
    sorted_distances_A = np.sort(finite_distances_A)
    distance_gaps_A = np.diff(sorted_distances_A)
    positive_gap_indices = np.flatnonzero(distance_gaps_A > np.finfo(float).eps)
    if positive_gap_indices.size < MINIMUM_LOCAL_MINIMUM_COUNT:
        raise ValueError(
            "trajectory distance distribution has fewer than two resolved association "
            "gaps; run longer or use a larger box"
        )
    largest_gap_indices = positive_gap_indices[
        np.argsort(distance_gaps_A[positive_gap_indices])[-MINIMUM_LOCAL_MINIMUM_COUNT:]
    ]
    threshold_gap_indices = np.sort(largest_gap_indices)
    thresholds_A = (
        sorted_distances_A[threshold_gap_indices]
        + sorted_distances_A[threshold_gap_indices + 1]
    ) / MINIMUM_LOCAL_MINIMUM_COUNT
    return AssociationThresholds(
        contact_pair_max_distance_A=float(thresholds_A[0]),
        solvent_separated_pair_max_distance_A=float(thresholds_A[1]),
    )


def _association_thresholds_from_physical_library(
    records,
) -> AssociationThresholds:
    pair_basins = records.basis_record["pair_basins"]
    contact_pair_max_distance_A = float(pair_basins["r_CIP_m"]) / ANGSTROM_TO_M
    solvent_separated_pair_max_distance_A = (
        float(pair_basins["r_SSIP_m"]) / ANGSTROM_TO_M
    )
    if not (
        0.0
        < contact_pair_max_distance_A
        < solvent_separated_pair_max_distance_A
    ):
        raise ValueError("physical-library association thresholds are not ordered")
    return AssociationThresholds(
        contact_pair_max_distance_A=contact_pair_max_distance_A,
        solvent_separated_pair_max_distance_A=solvent_separated_pair_max_distance_A,
    )


def _assign_center_states(
    center_frames: tuple[ChargedCenterFrame, ...],
    center_catalog: ChargedCenterCatalog,
    environment_frames: tuple[MolecularEnvironmentFrame, ...],
    environment_catalog: MolecularEnvironmentCatalog,
    thresholds: AssociationThresholds,
    records,
    mixture,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if len(environment_frames) != len(center_frames):
        raise ValueError("molecular environment frames must align with center frames")
    if environment_catalog.molecule_ids.size < center_catalog.molecule_ids.size:
        raise ValueError("molecular environment catalog must include all charged centers")
    state_labels: list[str] = []
    state_index_by_label: dict[str, int] = {}
    state_indices = np.zeros(
        (len(center_frames), center_catalog.molecule_ids.size),
        dtype=int,
    )
    counterion_indices = np.zeros_like(state_indices)
    cation_indices = _role_indices(center_catalog, ROLE_CATION)
    anion_indices = _role_indices(center_catalog, ROLE_ANION)
    for frame_index, center_frame in enumerate(center_frames):
        environment_frame = environment_frames[frame_index]
        pair_distances_A = _cation_anion_distance_matrix_A(
            center_frame,
            cation_indices,
            anion_indices,
        )
        for local_cation_index, center_index in enumerate(cation_indices):
            nearest_anion_local_index = int(
                np.argmin(pair_distances_A[local_cation_index])
            )
            nearest_anion_index = int(anion_indices[nearest_anion_local_index])
            counterion_indices[frame_index, center_index] = nearest_anion_index
            temporal_coordinates = _observed_temporal_coordinates(
                frame_index,
                int(center_index),
                nearest_anion_index,
                counterion_indices,
            )
            label = _active_sparse_state_label_for_center(
                records=records,
                mixture=mixture,
                center_frame=center_frame,
                center_catalog=center_catalog,
                environment_frame=environment_frame,
                environment_catalog=environment_catalog,
                center_index=int(center_index),
                counterion_index=nearest_anion_index,
                distances_A=pair_distances_A[local_cation_index],
                thresholds=thresholds,
                temporal_coordinates=temporal_coordinates,
            )
            state_indices[frame_index, center_index] = _state_index_for_label(
                label,
                state_labels,
                state_index_by_label,
            )
        for local_anion_index, center_index in enumerate(anion_indices):
            nearest_cation_local_index = int(
                np.argmin(pair_distances_A[:, local_anion_index])
            )
            nearest_cation_index = int(cation_indices[nearest_cation_local_index])
            counterion_indices[frame_index, center_index] = nearest_cation_index
            temporal_coordinates = _observed_temporal_coordinates(
                frame_index,
                int(center_index),
                nearest_cation_index,
                counterion_indices,
            )
            label = _active_sparse_state_label_for_center(
                records=records,
                mixture=mixture,
                center_frame=center_frame,
                center_catalog=center_catalog,
                environment_frame=environment_frame,
                environment_catalog=environment_catalog,
                center_index=int(center_index),
                counterion_index=nearest_cation_index,
                distances_A=pair_distances_A[:, local_anion_index],
                thresholds=thresholds,
                temporal_coordinates=temporal_coordinates,
            )
            state_indices[frame_index, center_index] = _state_index_for_label(
                label,
                state_labels,
                state_index_by_label,
            )
    return tuple(state_labels), state_indices, counterion_indices


def _merge_transport_equivalent_observed_states(
    observed_state_labels: tuple[str, ...],
    observed_state_indices: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    retained_field_indices = (
        generator_construction.STATE_KEY_PAIR_INDEX,
        generator_construction.STATE_KEY_LIGAND_INDEX,
        generator_construction.STATE_KEY_CLUSTER_INDEX,
        generator_construction.STATE_KEY_ENVIRONMENT_INDEX,
    )
    merged_labels: list[str] = []
    merged_index_by_label: dict[str, int] = {}
    observed_to_merged_index = np.empty(len(observed_state_labels), dtype=int)
    for observed_state_index, observed_label in enumerate(observed_state_labels):
        state_key = tuple(observed_label.split("|"))
        if len(state_key) != generator_construction.STATE_KEY_LENGTH:
            raise ValueError(f"observed state label has wrong key length: {observed_label}")
        merged_label = "|".join(
            state_key[field_index] for field_index in retained_field_indices
        )
        if merged_label not in merged_index_by_label:
            merged_index_by_label[merged_label] = len(merged_labels)
            merged_labels.append(merged_label)
        observed_to_merged_index[observed_state_index] = merged_index_by_label[
            merged_label
        ]
    state_indices = np.asarray(observed_state_indices, dtype=int)
    if np.any(state_indices < 0) or np.any(state_indices >= len(observed_state_labels)):
        raise ValueError("observed state indices are out of range")
    return _merge_nonpersistent_transport_states(
        tuple(merged_labels),
        observed_to_merged_index[state_indices],
    )


def _merge_nonpersistent_transport_states(
    state_labels: tuple[str, ...],
    state_indices: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    merged_labels = tuple(state_labels)
    merged_indices = np.asarray(state_indices, dtype=int).copy()
    while True:
        self_step_counts = np.asarray(
            [
                np.count_nonzero(
                    (merged_indices[:-1] == state_index)
                    & (merged_indices[1:] == state_index)
                )
                for state_index in range(len(merged_labels))
            ],
            dtype=int,
        )
        nonpersistent_indices = np.flatnonzero(self_step_counts < 2)
        if nonpersistent_indices.size == 0:
            return merged_labels, merged_indices
        if len(merged_labels) == 1:
            raise ValueError(
                "trajectory has no persistent transport basin for self-current estimation"
            )
        state_index_to_merge = int(nonpersistent_indices[0])
        neighbor_counts = np.zeros(len(merged_labels), dtype=int)
        for first_frame_indices, second_frame_indices in zip(
            merged_indices[:-1],
            merged_indices[1:],
            strict=True,
        ):
            leaving_mask = first_frame_indices == state_index_to_merge
            entering_mask = second_frame_indices == state_index_to_merge
            if np.any(leaving_mask):
                neighbor_counts += np.bincount(
                    second_frame_indices[leaving_mask],
                    minlength=len(merged_labels),
                )
            if np.any(entering_mask):
                neighbor_counts += np.bincount(
                    first_frame_indices[entering_mask],
                    minlength=len(merged_labels),
                )
        neighbor_counts[state_index_to_merge] = 0
        if np.max(neighbor_counts) == 0:
            raise ValueError(
                f"nonpersistent state {merged_labels[state_index_to_merge]} has no neighboring basin"
            )
        target_state_index = int(np.argmax(neighbor_counts))
        merged_indices[merged_indices == state_index_to_merge] = target_state_index
        retained_old_indices = tuple(
            index for index in range(len(merged_labels)) if index != state_index_to_merge
        )
        compact_index_by_old_index = {
            old_index: new_index
            for new_index, old_index in enumerate(retained_old_indices)
        }
        merged_indices = np.asarray(
            [compact_index_by_old_index[int(index)] for index in merged_indices.flat],
            dtype=int,
        ).reshape(merged_indices.shape)
        merged_labels = tuple(merged_labels[index] for index in retained_old_indices)


def _state_index_for_label(
    state_label: str,
    state_labels: list[str],
    state_index_by_label: dict[str, int],
) -> int:
    if state_label not in state_index_by_label:
        state_index_by_label[state_label] = len(state_labels)
        state_labels.append(state_label)
    return state_index_by_label[state_label]


def _observed_temporal_coordinates(
    frame_index: int,
    center_index: int,
    counterion_index: int,
    counterion_indices: np.ndarray,
) -> dict[str, float]:
    partner_retained = False
    identity_changed = False
    cage_coordinate = -1.0
    if frame_index > 0:
        partner_retained = (
            int(counterion_indices[frame_index - 1, center_index])
            == counterion_index
        )
        identity_changed = not partner_retained
        if partner_retained:
            cage_coordinate = 1.0
    return {
        generator_construction.ReducedCoordinate.CAGE_COORDINATE.value: cage_coordinate,
        generator_construction.ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value: float(
            partner_retained
        ),
        generator_construction.ReducedCoordinate.IDENTITY_COORDINATE.value: float(
            identity_changed
        ),
        generator_construction.ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value: float(
            identity_changed
        ),
    }


def _active_sparse_state_label_for_center(
    records,
    mixture,
    center_frame: ChargedCenterFrame,
    center_catalog: ChargedCenterCatalog,
    environment_frame: MolecularEnvironmentFrame,
    environment_catalog: MolecularEnvironmentCatalog,
    center_index: int,
    counterion_index: int,
    distances_A: np.ndarray,
    thresholds: AssociationThresholds,
    temporal_coordinates: dict[str, float],
) -> str:
    pair_label = _pair_label_for_counterion_distances(distances_A, thresholds)
    if center_catalog.roles[center_index] == ROLE_ANION:
        active_anion_name = center_catalog.species_labels[center_index]
        cation_index = counterion_index
        anion_index = center_index
    elif center_catalog.roles[center_index] == ROLE_CATION:
        active_anion_name = center_catalog.species_labels[counterion_index]
        cation_index = center_index
        anion_index = counterion_index
    else:
        raise ValueError(f"unsupported charged-center role {center_catalog.roles[center_index]}")
    configuration = _two_center_site_configuration_from_frame(
        center_frame, center_catalog, cation_index, anion_index
    )
    environment_configuration = _environment_site_configuration_from_frame(
        environment_frame,
        environment_catalog,
        center_catalog.molecule_ids[cation_index],
    )
    coordinate_values = _reduced_coordinate_values_from_center_observation(
        records,
        mixture,
        configuration,
        environment_configuration,
        environment_frame,
        environment_catalog,
        center_catalog.molecule_ids[anion_index],
        pair_label,
        distances_A,
        thresholds,
        temporal_coordinates,
    )
    state_key = generator_construction.sparse_state_key_from_reduced_observation(
        records=records,
        configuration=configuration,
        mixture=mixture,
        pair_label=pair_label,
        active_anion_component_name=active_anion_name,
        coordinate_values=coordinate_values,
    )
    return "|".join(state_key)


def _two_center_site_configuration_from_frame(
    center_frame: ChargedCenterFrame,
    center_catalog: ChargedCenterCatalog,
    cation_index: int,
    anion_index: int,
) -> SiteConfiguration:
    box_lengths_m = (
        center_frame.box_bounds_A[:, BOX_BOUND_HIGH_COLUMN]
        - center_frame.box_bounds_A[:, BOX_BOUND_LOW_COLUMN]
    ) * ANGSTROM_TO_M
    positions_m = np.asarray(
        [
            center_frame.wrapped_positions_A[cation_index] * ANGSTROM_TO_M,
            center_frame.wrapped_positions_A[anion_index] * ANGSTROM_TO_M,
        ],
        dtype=float,
    )
    unwrapped_positions_m = np.asarray(
        [
            center_frame.positions_A[cation_index] * ANGSTROM_TO_M,
            center_frame.positions_A[anion_index] * ANGSTROM_TO_M,
        ],
        dtype=float,
    )
    return SiteConfiguration(
        species_names=(
            center_catalog.species_labels[cation_index],
            center_catalog.species_labels[anion_index],
        ),
        molecule_ids=np.asarray(
            [
                center_catalog.molecule_ids[cation_index],
                center_catalog.molecule_ids[anion_index],
            ],
            dtype=int,
        ),
        site_ids=np.asarray([0, 0], dtype=int),
        positions_m=positions_m,
        unwrapped_positions_m=unwrapped_positions_m,
        box_lengths_m=np.asarray(box_lengths_m, dtype=float),
    )


def _environment_site_configuration_from_frame(
    environment_frame: MolecularEnvironmentFrame,
    environment_catalog: MolecularEnvironmentCatalog,
    focal_cation_molecule_id: int,
) -> SiteConfiguration:
    box_lengths_m = (
        environment_frame.box_bounds_A[:, BOX_BOUND_HIGH_COLUMN]
        - environment_frame.box_bounds_A[:, BOX_BOUND_LOW_COLUMN]
    ) * ANGSTROM_TO_M
    molecule_count = environment_catalog.molecule_ids.size
    cation_matches = np.flatnonzero(
        environment_catalog.molecule_ids == focal_cation_molecule_id
    )
    if cation_matches.size != 1:
        raise ValueError(
            f"cation molecule {focal_cation_molecule_id} must occur once in environment catalog"
        )
    focal_position_A = environment_frame.wrapped_positions_A[int(cation_matches[0])]
    relative_positions_A = environment_frame.wrapped_positions_A - focal_position_A
    relative_positions_A -= (box_lengths_m / ANGSTROM_TO_M) * np.round(
        relative_positions_A / (box_lengths_m / ANGSTROM_TO_M)
    )
    focal_index = int(cation_matches[0])
    molecule_order = np.concatenate(
        (
            np.asarray((focal_index,), dtype=int),
            np.delete(np.arange(molecule_count, dtype=int), focal_index),
        )
    )
    return SiteConfiguration(
        species_names=tuple(
            environment_catalog.species_labels[index] for index in molecule_order
        ),
        molecule_ids=environment_catalog.molecule_ids[molecule_order],
        site_ids=np.zeros(molecule_count, dtype=int),
        positions_m=relative_positions_A[molecule_order] * ANGSTROM_TO_M,
        unwrapped_positions_m=(
            environment_frame.positions_A[molecule_order] * ANGSTROM_TO_M
        ),
        box_lengths_m=box_lengths_m,
    )


def _observed_anion_orientation(
    pair_configuration: SiteConfiguration,
    environment_frame: MolecularEnvironmentFrame,
    environment_catalog: MolecularEnvironmentCatalog,
    anion_molecule_id: int,
) -> float:
    matching_indices = np.flatnonzero(
        environment_catalog.molecule_ids == anion_molecule_id
    )
    if matching_indices.size != 1:
        raise ValueError(
            f"anion molecule {anion_molecule_id} must occur once in environment catalog"
        )
    pair_axis_m = pair_configuration.positions_m[1] - pair_configuration.positions_m[0]
    pair_axis_m -= pair_configuration.box_lengths_m * np.round(
        pair_axis_m / pair_configuration.box_lengths_m
    )
    pair_distance_m = float(np.linalg.norm(pair_axis_m))
    if pair_distance_m == 0.0:
        raise ValueError("Li-anion orientation axis has zero length")
    orientation = environment_frame.orientation_vectors[int(matching_indices[0])]
    return float(np.dot(orientation, pair_axis_m / pair_distance_m))


def _reduced_coordinate_values_from_center_observation(
    records,
    mixture,
    configuration: SiteConfiguration,
    environment_configuration: SiteConfiguration,
    environment_frame: MolecularEnvironmentFrame,
    environment_catalog: MolecularEnvironmentCatalog,
    anion_molecule_id: int,
    pair_label: str,
    distances_A: np.ndarray,
    thresholds: AssociationThresholds,
    temporal_coordinates: dict[str, float],
) -> dict[str, float]:
    cation_position_m = configuration.positions_m[0]
    anion_position_m = configuration.positions_m[1]
    pair_distance_m = float(np.linalg.norm(anion_position_m - cation_position_m))
    coordinate_values = {
        generator_construction.ReducedCoordinate.LI_ANION_DISTANCE.value: pair_distance_m,
        generator_construction.ReducedCoordinate.LI_SOLVENT_COORDINATION.value: generator_construction.compute_role_coordination_number(
            records, environment_configuration, "cation", "solvent", "Li_solvent"
        ),
        generator_construction.ReducedCoordinate.LI_LIGAND_COORDINATION.value: generator_construction.compute_role_coordination_number(
            records, environment_configuration, "cation", "additive", "Li_ligand"
        ),
        generator_construction.ReducedCoordinate.LI_ANION_COORDINATION.value: generator_construction.compute_role_coordination_number(
            records, environment_configuration, "cation", "anion", "Li_anion"
        ),
        generator_construction.ReducedCoordinate.ANION_ORIENTATION.value: _observed_anion_orientation(
            configuration,
            environment_frame,
            environment_catalog,
            anion_molecule_id,
        ),
        generator_construction.ReducedCoordinate.LOCAL_PACKING_FRACTION.value: generator_construction.compute_local_packing_fraction(
            records, environment_configuration
        ),
        generator_construction.ReducedCoordinate.LOCAL_IONIC_STRENGTH.value: (
            mixture.ionic_strength_mol_m3
        ),
        generator_construction.ReducedCoordinate.LOCAL_DIELECTRIC.value: (
            mixture.dielectric_constant
        ),
        generator_construction.ReducedCoordinate.LOCAL_VISCOSITY.value: (
            mixture.viscosity_Pa_s
        ),
        generator_construction.LOCAL_ADDITIVE_COLLISION_EXPOSURE: (
            mixture.additive_collision_exposure
        ),
        generator_construction.ReducedCoordinate.ATMOSPHERE_POLARIZATION.value: 0.0,
        generator_construction.ReducedCoordinate.CAGE_COORDINATE.value: temporal_coordinates[
            generator_construction.ReducedCoordinate.CAGE_COORDINATE.value
        ],
        generator_construction.ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value: temporal_coordinates[
            generator_construction.ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value
        ],
        generator_construction.ReducedCoordinate.CLUSTER_COORDINATE.value: (
            _cluster_coordinate_for_counterion_distances(distances_A, thresholds)
        ),
        generator_construction.ReducedCoordinate.IDENTITY_COORDINATE.value: temporal_coordinates[
            generator_construction.ReducedCoordinate.IDENTITY_COORDINATE.value
        ],
        generator_construction.ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value: temporal_coordinates[
            generator_construction.ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value
        ],
    }
    local_fields = generator_construction._local_fields_for_coordinate_values(
        records,
        environment_configuration,
        coordinate_values,
    )
    coordinate_values[generator_construction.ReducedCoordinate.LOCAL_IONIC_STRENGTH.value] = (
        local_fields.ionic_strength_mol_m3
    )
    coordinate_values[generator_construction.ReducedCoordinate.LOCAL_DIELECTRIC.value] = (
        local_fields.dielectric_constant
    )
    coordinate_values[generator_construction.ReducedCoordinate.LOCAL_VISCOSITY.value] = (
        local_fields.viscosity_Pa_s
    )
    return coordinate_values


def _pair_label_for_counterion_distances(
    distances_A: np.ndarray,
    thresholds: AssociationThresholds,
) -> str:
    contact_count = int(
        np.count_nonzero(distances_A < thresholds.contact_pair_max_distance_A)
    )
    nearest_distance_A = float(np.min(distances_A))
    if contact_count > 0:
        return PairBasin.CONTACT_ION_PAIR.value
    if nearest_distance_A < thresholds.solvent_separated_pair_max_distance_A:
        return PairBasin.SOLVENT_SEPARATED_ION_PAIR.value
    if nearest_distance_A >= thresholds.solvent_separated_pair_max_distance_A:
        return PairBasin.FREE.value
    raise RuntimeError("unreachable sparse-state pair classification")


def _cluster_coordinate_for_counterion_distances(
    distances_A: np.ndarray,
    thresholds: AssociationThresholds,
) -> float:
    associated_count = int(
        np.count_nonzero(distances_A < thresholds.solvent_separated_pair_max_distance_A)
    )
    if associated_count >= MINIMUM_COUNTERION_COUNT_FOR_AGGREGATE:
        return 1.0
    return 0.0


def _state_labels(center_catalog: ChargedCenterCatalog) -> tuple[str, ...]:
    labels: list[str] = []
    label_families = (
        "free_ion_center",
        "contact_pair_center",
        "solvent_separated_pair_center",
        "aggregate_center",
    )
    for species_label, role in zip(
        center_catalog.species_labels,
        center_catalog.roles,
        strict=True,
    ):
        for label_family in label_families:
            label = f"{label_family}:{species_label}:{role}"
            if label not in labels:
                labels.append(label)
    return tuple(labels)


def _state_label_for_counterion_distances(
    distances_A: np.ndarray,
    thresholds: AssociationThresholds,
    species_label: str,
    role: str,
) -> str:
    contact_count = int(
        np.count_nonzero(distances_A < thresholds.contact_pair_max_distance_A)
    )
    associated_count = int(
        np.count_nonzero(distances_A < thresholds.solvent_separated_pair_max_distance_A)
    )
    nearest_distance_A = float(np.min(distances_A))
    if associated_count >= MINIMUM_COUNTERION_COUNT_FOR_AGGREGATE:
        return f"aggregate_center:{species_label}:{role}"
    if contact_count > 0:
        return f"contact_pair_center:{species_label}:{role}"
    if nearest_distance_A < thresholds.solvent_separated_pair_max_distance_A:
        return f"solvent_separated_pair_center:{species_label}:{role}"
    if nearest_distance_A >= thresholds.solvent_separated_pair_max_distance_A:
        return f"free_ion_center:{species_label}:{role}"
    raise RuntimeError("unreachable association-state classification")


def _cation_anion_distance_matrix_A(
    center_frame: ChargedCenterFrame,
    cation_indices: np.ndarray,
    anion_indices: np.ndarray,
) -> np.ndarray:
    cation_positions_A = center_frame.wrapped_positions_A[cation_indices]
    anion_positions_A = center_frame.wrapped_positions_A[anion_indices]
    box_low_A = center_frame.box_bounds_A[:, BOX_BOUND_LOW_COLUMN]
    box_high_A = center_frame.box_bounds_A[:, BOX_BOUND_HIGH_COLUMN]
    box_lengths_A = box_high_A - box_low_A
    displacements_A = (
        cation_positions_A[:, np.newaxis, :] - anion_positions_A[np.newaxis, :, :]
    )
    displacements_A -= box_lengths_A * np.round(displacements_A / box_lengths_A)
    return np.linalg.norm(displacements_A, axis=2)


def _role_indices(
    center_catalog: ChargedCenterCatalog,
    role: str,
) -> np.ndarray:
    role_indices = np.asarray(
        [
            center_index
            for center_index, center_role in enumerate(center_catalog.roles)
            if center_role == role
        ],
        dtype=int,
    )
    if role_indices.size == 0:
        raise ValueError(f"no charged centers with role {role}")
    return role_indices


def _charge_displacements_by_step_m(
    center_frames: tuple[ChargedCenterFrame, ...],
    center_catalog: ChargedCenterCatalog,
    state_index_by_frame_and_center: np.ndarray,
    counterion_index_by_frame_and_center: np.ndarray,
    thresholds: AssociationThresholds,
) -> np.ndarray:
    center_count = int(center_catalog.molecule_ids.size)
    expected_state_shape = (len(center_frames), center_count)
    if state_index_by_frame_and_center.shape != expected_state_shape:
        raise ValueError(
            "state_index_by_frame_and_center must have shape "
            f"{expected_state_shape}"
        )
    if counterion_index_by_frame_and_center.shape != expected_state_shape:
        raise ValueError(
            "counterion_index_by_frame_and_center must have shape "
            f"{expected_state_shape}"
        )
    if np.unique(center_catalog.molecule_ids).size != center_count:
        raise ValueError("charged-center molecule identities must be unique")
    for center_frame in center_frames:
        if center_frame.positions_A.shape != (center_count, CARTESIAN):
            raise ValueError(
                "charged-center positions must preserve catalog identity order"
            )

    displacements: list[np.ndarray] = []
    for frame_index in range(len(center_frames) - 1):
        center_displacements_A = (
            center_frames[frame_index + 1].positions_A
            - center_frames[frame_index].positions_A
        )
        for center_index in range(center_count):
            counterion_index = int(
                counterion_index_by_frame_and_center[frame_index, center_index]
            )
            if counterion_index < 0 or counterion_index >= center_count:
                raise ValueError(
                    f"counterion index {counterion_index} is outside the center catalog"
                )
            if counterion_index == center_index:
                raise ValueError("focal center cannot be its own counterion")
            local_center_indices = _state_local_charged_center_indices(
                center_frames[frame_index],
                center_catalog,
                center_index,
                counterion_index,
                thresholds,
            )
            displacements.append(
                np.einsum(
                    "i,ia->a",
                    center_catalog.formal_charges_e[local_center_indices],
                    center_displacements_A[local_center_indices],
                )
                * ANGSTROM_TO_M
            )
    return np.asarray(displacements, dtype=float)


def _self_charge_polarization_by_frame_and_center_m(
    center_frames: tuple[ChargedCenterFrame, ...],
    center_catalog: ChargedCenterCatalog,
    environment_frames: tuple[MolecularEnvironmentFrame, ...],
    environment_catalog: MolecularEnvironmentCatalog,
    counterion_index_by_frame_and_center: np.ndarray,
    thresholds: AssociationThresholds,
) -> np.ndarray:
    center_count = int(center_catalog.molecule_ids.size)
    expected_shape = (len(center_frames), center_count)
    if counterion_index_by_frame_and_center.shape != expected_shape:
        raise ValueError(
            "counterion_index_by_frame_and_center must have shape "
            f"{expected_shape}"
        )
    if len(environment_frames) != len(center_frames):
        raise ValueError("environment frames must align with charged-center frames")
    environment_index_by_molecule_id = {
        int(molecule_id): environment_index
        for environment_index, molecule_id in enumerate(
            environment_catalog.molecule_ids
        )
    }
    polarizations_m = np.zeros(
        (len(center_frames), center_count, CARTESIAN), dtype=float
    )
    for frame_index, (center_frame, environment_frame) in enumerate(
        zip(center_frames, environment_frames, strict=True)
    ):
        for center_index in range(center_count):
            counterion_index = int(
                counterion_index_by_frame_and_center[frame_index, center_index]
            )
            local_center_indices = _state_local_charged_center_indices(
                center_frame, center_catalog, center_index, counterion_index, thresholds
            )
            local_molecule_ids = center_catalog.molecule_ids[local_center_indices]
            local_environment_indices = np.asarray(
                [
                    environment_index_by_molecule_id[int(molecule_id)]
                    for molecule_id in local_molecule_ids
                ],
                dtype=int,
            )
            cluster_center_A = np.mean(
                environment_frame.positions_A[local_environment_indices],
                axis=0,
            )
            net_formal_charge_e = float(
                np.sum(center_catalog.formal_charges_e[local_center_indices])
            )
            polarizations_m[frame_index, center_index] = (
                net_formal_charge_e * cluster_center_A * ANGSTROM_TO_M
            )
    return polarizations_m


def _state_local_charged_center_indices(
    center_frame: ChargedCenterFrame,
    center_catalog: ChargedCenterCatalog,
    center_index: int,
    counterion_index: int,
    thresholds: AssociationThresholds,
) -> np.ndarray:
    focal_position_A = center_frame.wrapped_positions_A[center_index]
    box_lengths_A = (
        center_frame.box_bounds_A[:, BOX_BOUND_HIGH_COLUMN]
        - center_frame.box_bounds_A[:, BOX_BOUND_LOW_COLUMN]
    )
    displacement_A = center_frame.wrapped_positions_A - focal_position_A
    displacement_A -= box_lengths_A * np.round(displacement_A / box_lengths_A)
    distances_A = np.linalg.norm(displacement_A, axis=1)
    opposite_role = (
        ROLE_ANION
        if center_catalog.roles[center_index] == ROLE_CATION
        else ROLE_CATION
    )
    associated_indices = np.asarray(
        [
            index
            for index, role in enumerate(center_catalog.roles)
            if role == opposite_role
            and distances_A[index] < thresholds.solvent_separated_pair_max_distance_A
        ],
        dtype=int,
    )
    if associated_indices.size == 0:
        return np.asarray((center_index,), dtype=int)
    if counterion_index not in associated_indices:
        raise ValueError(
            f"assigned counterion {counterion_index} is outside the observed local state"
        )
    return np.concatenate((np.asarray((center_index,), dtype=int), associated_indices))


def _state_local_membership_stable_step_mask(
    center_frames: tuple[ChargedCenterFrame, ...],
    center_catalog: ChargedCenterCatalog,
    counterion_index_by_frame_and_center: np.ndarray,
    thresholds: AssociationThresholds,
) -> np.ndarray:
    center_count = int(center_catalog.molecule_ids.size)
    memberships: list[list[tuple[int, ...]]] = []
    for frame_index, center_frame in enumerate(center_frames):
        frame_memberships = []
        for center_index in range(center_count):
            counterion_index = int(
                counterion_index_by_frame_and_center[frame_index, center_index]
            )
            local_indices = _state_local_charged_center_indices(
                center_frame,
                center_catalog,
                center_index,
                counterion_index,
                thresholds,
            )
            frame_memberships.append(tuple(sorted(int(index) for index in local_indices)))
        memberships.append(frame_memberships)
    return np.asarray(
        [
            [
                memberships[frame_index][center_index]
                == memberships[frame_index + 1][center_index]
                for center_index in range(center_count)
            ]
            for frame_index in range(len(center_frames) - 1)
        ],
        dtype=bool,
    )


def _mean_box_volume_m3(center_frames: tuple[ChargedCenterFrame, ...]) -> float:
    volumes_m3 = []
    for center_frame in center_frames:
        box_low_A = center_frame.box_bounds_A[:, BOX_BOUND_LOW_COLUMN]
        box_high_A = center_frame.box_bounds_A[:, BOX_BOUND_HIGH_COLUMN]
        box_lengths_A = box_high_A - box_low_A
        volumes_m3.append(float(np.prod(box_lengths_A)) * ANGSTROM_TO_M**CARTESIAN)
    mean_volume_m3 = float(np.mean(np.asarray(volumes_m3, dtype=float)))
    if mean_volume_m3 <= 0.0:
        raise ValueError("mean trajectory box volume must be positive")
    return mean_volume_m3


def _primitive_arrays_from_projected_set(
    primitive_set: ProjectedGeneratorPrimitiveSet,
) -> dict[str, np.ndarray]:
    state_count = len(primitive_set.state_labels)
    state_index_by_label = {
        label: state_index
        for state_index, label in enumerate(primitive_set.state_labels)
    }
    concentrations = np.asarray(
        [
            float(primitive_set.state_concentrations_mol_m3[state_label])
            for state_label in primitive_set.state_labels
        ],
        dtype=float,
    )
    capacity_fluxes = np.zeros((state_count, state_count), dtype=float)
    first_moments = np.zeros((state_count, state_count, CARTESIAN), dtype=float)
    second_moments = np.zeros(
        (state_count, state_count, CARTESIAN, CARTESIAN),
        dtype=float,
    )
    self_current_tensors = np.zeros(
        (state_count, CARTESIAN, CARTESIAN),
        dtype=float,
    )
    for flux_record in primitive_set.reactive_fluxes:
        from_index = state_index_by_label[flux_record.from_state_label]
        to_index = state_index_by_label[flux_record.to_state_label]
        capacity_fluxes[from_index, to_index] = float(
            flux_record.symmetric_flux_mol_m3_s
        )
        capacity_fluxes[to_index, from_index] = float(
            flux_record.symmetric_flux_mol_m3_s
        )
    for moment_record in primitive_set.conditional_displacement_moments:
        from_index = state_index_by_label[moment_record.from_state_label]
        to_index = state_index_by_label[moment_record.to_state_label]
        mean_displacement_m = np.asarray(
            moment_record.mean_charge_displacement_m,
            dtype=float,
        )
        second_moment_m2 = np.asarray(moment_record.second_moment_m2, dtype=float)
        first_moments[from_index, to_index] = mean_displacement_m
        first_moments[to_index, from_index] = -mean_displacement_m
        second_moments[from_index, to_index] = second_moment_m2
        second_moments[to_index, from_index] = second_moment_m2
    for tensor_record in primitive_set.self_current_tensors:
        state_index = state_index_by_label[tensor_record.state_label]
        self_current_tensors[state_index] = np.asarray(
            tensor_record.diffusion_tensor_m2_s,
            dtype=float,
        )
    diagnose_finite_process_legality(
        primitive_set.state_labels,
        concentrations,
        capacity_fluxes,
        first_moments,
        second_moments,
        _directed_transition_sample_counts_from_projected_set(primitive_set),
    )
    return {
        "state_concentrations_mol_m3": concentrations,
        "symmetric_capacity_fluxes_K_ij_mol_m3_s": capacity_fluxes,
        "transition_first_moments_d_ij_m": first_moments,
        "transition_second_moments_M_ij_m2": second_moments,
        "self_current_tensors_D_self_i_m2_s": self_current_tensors,
        "mori_memory_matrix_A": np.zeros((0, 0), dtype=float),
        "mori_current_coupling_matrix_h": np.zeros((0, CARTESIAN), dtype=float),
    }


def _directed_transition_sample_counts_from_projected_set(
    primitive_set: ProjectedGeneratorPrimitiveSet,
) -> np.ndarray:
    state_index_by_label = {
        state_label: state_index
        for state_index, state_label in enumerate(primitive_set.state_labels)
    }
    directed_counts = np.zeros(
        (len(primitive_set.state_labels), len(primitive_set.state_labels)),
        dtype=int,
    )
    for flux_record in primitive_set.reactive_fluxes:
        from_state_index = state_index_by_label[flux_record.from_state_label]
        to_state_index = state_index_by_label[flux_record.to_state_label]
        directed_counts[from_state_index, to_state_index] = int(
            flux_record.forward_sample_count
        )
        directed_counts[to_state_index, from_state_index] = int(
            flux_record.reverse_sample_count
        )
    return directed_counts


def _component_drift_violation(
    component_drift_residuals: tuple[FiniteProcessComponentDriftResidual, ...],
) -> bool:
    for component_drift_residual in component_drift_residuals:
        tolerance = max(
            POISSON_SOLVABILITY_ABS_TOL,
            POISSON_SOLVABILITY_EPSILON_FACTOR
            * np.finfo(float).eps
            * component_drift_residual.weighted_absolute_drift_scale_mol_m2_s,
        )
        if component_drift_residual.weighted_drift_norm_mol_m2_s > tolerance:
            return True
    return False


def _invalid_component_drift_failure_reason(
    component_drift_residuals: tuple[FiniteProcessComponentDriftResidual, ...],
) -> str:
    offending_components = []
    for residual in component_drift_residuals:
        tolerance = max(
            POISSON_SOLVABILITY_ABS_TOL,
            POISSON_SOLVABILITY_EPSILON_FACTOR
            * np.finfo(float).eps
            * residual.weighted_absolute_drift_scale_mol_m2_s,
        )
        if residual.weighted_drift_norm_mol_m2_s <= tolerance:
            continue
        edges = ", ".join(
            f"{edge.from_state_label}->{edge.to_state_label}"
            f"(norm={edge.contribution_norm_mol_m2_s:.6e},"
            f" forward={edge.forward_sample_count}, reverse={edge.reverse_sample_count})"
            for edge in residual.top_edge_contributions
        )
        offending_components.append(
            f"component={residual.component_id}, residual="
            f"{residual.weighted_drift_norm_mol_m2_s:.6e}, edges=[{edges}]"
        )
    return "invalid finite-state drift; offending edges: " + "; ".join(
        offending_components
    )


def _projected_primitive_extraction_diagnostics(
    primitive_set: ProjectedGeneratorPrimitiveSet,
    component_drift_violation: bool,
) -> dict:
    return {
        "visited_state_count": int(primitive_set.diagnostics.visited_state_count),
        "transition_sample_count": int(
            primitive_set.diagnostics.transition_sample_count
        ),
        "self_displacement_sample_count": int(
            primitive_set.diagnostics.self_displacement_sample_count
        ),
        "generated_event_count": int(primitive_set.diagnostics.generated_event_count),
        "trajectory_time_s": float(primitive_set.diagnostics.trajectory_time_s),
        "self_diffusion_readiness_status": "succeeded",
        "self_diffusion_convergence": _self_diffusion_convergence_records(
            primitive_set.diagnostics.self_diffusion_convergence
        ),
        "total_transport_concentration_mol_m3": float(
            primitive_set.diagnostics.total_transport_concentration_mol_m3
        ),
        "component_drift_residuals": _component_drift_residual_records(
            primitive_set.diagnostics.component_drift_residuals
        ),
        "component_drift_violation": bool(component_drift_violation),
        "finite_process_legality": {
            "maximum_detailed_balance_residual_mol_m3_s": float(
                primitive_set.diagnostics.finite_process_legality.maximum_detailed_balance_residual_mol_m3_s
            ),
            "component_drift_residuals": _component_drift_residual_records(
                primitive_set.diagnostics.finite_process_legality.component_drift_residuals
            ),
        },
    }


def _self_diffusion_convergence_records(convergence_diagnostics) -> list[dict]:
    return [
        {
            "state_label": diagnostic.state_label,
            "convergence_status": diagnostic.convergence_status,
            "not_complete_reason": diagnostic.not_complete_reason,
            "lag_start_frames": diagnostic.lag_start_frames,
            "lag_stop_frames": diagnostic.lag_stop_frames,
            "lag_count": diagnostic.lag_count,
            "minimum_samples_per_lag": diagnostic.minimum_samples_per_lag,
            "maximum_samples_per_lag": diagnostic.maximum_samples_per_lag,
            "trace_slope_m2_s": diagnostic.trace_slope_m2_s,
            "trace_slope_standard_error_m2_s": diagnostic.trace_slope_standard_error_m2_s,
            "log_log_exponent": diagnostic.log_log_exponent,
            "log_log_exponent_standard_error": diagnostic.log_log_exponent_standard_error,
        }
        for diagnostic in convergence_diagnostics
    ]


def _component_drift_residual_records(
    component_drift_residuals: tuple[FiniteProcessComponentDriftResidual, ...],
):
    records = []
    for component_drift_residual in component_drift_residuals:
        records.append(
            {
                "component_id": int(component_drift_residual.component_id),
                "state_labels": list(component_drift_residual.state_labels),
                "state_concentrations_mol_m3": list(
                    component_drift_residual.state_concentrations_mol_m3
                ),
                "exit_rates_s_inv": list(component_drift_residual.exit_rates_s_inv),
                "concentration_sum_mol_m3": float(
                    component_drift_residual.concentration_sum_mol_m3
                ),
                "weighted_drift_mol_m2_s": list(
                    component_drift_residual.weighted_drift_mol_m2_s
                ),
                "weighted_drift_norm_mol_m2_s": float(
                    component_drift_residual.weighted_drift_norm_mol_m2_s
                ),
                "weighted_absolute_drift_scale_mol_m2_s": float(
                    component_drift_residual.weighted_absolute_drift_scale_mol_m2_s
                ),
                "top_edge_contributions": _edge_drift_contribution_records(
                    component_drift_residual.top_edge_contributions
                ),
            }
        )
    return records


def _edge_drift_contribution_records(
    edge_drift_contributions: tuple[FiniteProcessEdgeDriftContribution, ...],
):
    records = []
    for edge_drift_contribution in edge_drift_contributions:
        records.append(
            {
                "component_id": int(edge_drift_contribution.component_id),
                "from_state_label": edge_drift_contribution.from_state_label,
                "to_state_label": edge_drift_contribution.to_state_label,
                "contribution_mol_m2_s": list(
                    edge_drift_contribution.contribution_mol_m2_s
                ),
                "contribution_norm_mol_m2_s": float(
                    edge_drift_contribution.contribution_norm_mol_m2_s
                ),
                "capacity_flux_mol_m3_s": float(
                    edge_drift_contribution.capacity_flux_mol_m3_s
                ),
                "first_moment_norm_m": float(
                    edge_drift_contribution.first_moment_norm_m
                ),
                "forward_sample_count": int(
                    edge_drift_contribution.forward_sample_count
                ),
                "reverse_sample_count": int(
                    edge_drift_contribution.reverse_sample_count
                ),
                "missing_reverse_event_candidate": bool(
                    edge_drift_contribution.missing_reverse_event_candidate
                ),
            }
        )
    return records


def _seconds_per_femtosecond() -> float:
    return 1.0e-15


if __name__ == "__main__":
    raise SystemExit(main())
