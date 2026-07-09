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
from conductivity.physical_library.projected_analytical_conductivity import (
    CARTESIAN,
    POISSON_SOLVABILITY_ABS_TOL,
    POISSON_SOLVABILITY_EPSILON_FACTOR,
    PROJECTED_REFERENCE_VOLUME_M3,
    compute_projected_analytical_conductivity_from_primitives,
)
from conductivity.physical_library import generator_construction
from conductivity.physical_library.mixture_closures import compute_mixture_closures
from conductivity.physical_library.physical_objects import PairBasin, SiteConfiguration
from conductivity.physical_library.projected_primitives_io import (
    PRIMITIVE_SCHEMA,
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
)


LAMMPS_COLUMN_ID = 0
LAMMPS_COLUMN_MOLECULE_ID = 1
LAMMPS_COLUMN_CHARGE_E = 2
LAMMPS_POSITION_COLUMN_START = 3
LAMMPS_POSITION_COLUMN_STOP = LAMMPS_POSITION_COLUMN_START + CARTESIAN
LAMMPS_DUMP_COLUMN_COUNT = LAMMPS_POSITION_COLUMN_STOP
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
class AssociationThresholds:
    contact_pair_max_distance_A: float
    solvent_separated_pair_max_distance_A: float


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
    parser.add_argument("--recipe-yaml", required=True, type=Path)
    parser.add_argument("--physical-library-root", required=True, type=Path)
    parser.add_argument("--dt-fs", required=True, type=float)
    parser.add_argument("--trajectory-dump-stride-steps", required=True, type=int)
    parser.add_argument("--output-yaml", required=True, type=Path)
    args = parser.parse_args()

    extract_projected_primitives_from_lammps_dump(
        trajectory_path=args.trajectory,
        composition_json_path=args.composition_json,
        copies_json_path=args.copies_json,
        recipe_yaml_path=args.recipe_yaml,
        physical_library_root=args.physical_library_root,
        timestep_fs=float(args.dt_fs),
        trajectory_dump_stride_steps=int(args.trajectory_dump_stride_steps),
        output_yaml_path=args.output_yaml,
    )
    return 0


def extract_projected_primitives_from_lammps_dump(
    trajectory_path: Path,
    composition_json_path: Path,
    copies_json_path: Path,
    recipe_yaml_path: Path,
    physical_library_root: Path,
    timestep_fs: float,
    trajectory_dump_stride_steps: int,
    output_yaml_path: Path,
):
    if timestep_fs <= 0.0:
        raise ValueError("timestep_fs must be positive")
    if trajectory_dump_stride_steps <= 0:
        raise ValueError("trajectory_dump_stride_steps must be positive")

    composition_record = _load_json_mapping(composition_json_path)
    copies_record = _load_json_mapping(copies_json_path)
    recipe_context = generator_construction.build_recipe_library_context(
        recipe_yaml_path,
        physical_library_root,
    )
    records = recipe_context.library_records
    mixture = compute_mixture_closures(
        records=records,
        composition=generator_construction.mixture_composition_from_recipe_context(
            recipe_context
        ),
        temperature_K=recipe_context.temperature_K,
    )
    species_ranges = _species_ranges_from_copies_record(copies_record)
    center_catalog = _charged_center_catalog_from_species_ranges(species_ranges)
    frames = tuple(_read_lammps_custom_dump(trajectory_path))
    if len(frames) < MINIMUM_LOCAL_MINIMUM_COUNT:
        raise ValueError("primitive extraction needs at least two trajectory frames")

    center_frames = tuple(
        _charged_center_frame_from_lammps_frame(frame, center_catalog)
        for frame in frames
    )
    association_distances_A = _nearest_counterion_distances_A(
        center_frames,
        center_catalog,
    )
    thresholds = _association_thresholds_from_distances_A(
        association_distances_A,
    )
    state_labels, state_index_by_frame_and_center = _assign_center_states(
        center_frames,
        center_catalog,
        thresholds,
        records,
        mixture,
    )
    charge_displacements_m = _charge_displacements_by_step_m(
        center_frames,
        center_catalog,
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
    primitive_arrays = _primitive_arrays_from_projected_set(primitive_set)
    component_drift_violation = _component_drift_violation(
        primitive_set.diagnostics.component_drift_residuals
    )
    diagnostics = _projected_primitive_extraction_diagnostics(
        primitive_set,
        component_drift_violation,
    )
    if component_drift_violation:
        primitive_arrays = _primitive_arrays_with_component_solvable_first_moments(
            primitive_arrays,
            primitive_set,
        )
        if _component_drift_violation(
            primitive_set.diagnostics.component_solvable_projection.projected_component_drift_residuals
        ):
            failure_reason = (
                "component-solvable first-moment projection did not remove "
                "finite-state drift"
            )
            write_failed_projected_primitive_yaml(
                output_yaml_path,
                PRIMITIVE_SCHEMA,
                failure_reason,
                diagnostics,
            )
            raise ValueError(failure_reason)
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
    artifact["projected_readout_status"] = _projected_readout_status_from_result(
        projected_result
    )
    artifact["sigma_mS_cm"] = float(projected_result.sigma_mS_cm)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    output_yaml_path.write_text(yaml.safe_dump(artifact, sort_keys=False))
    return artifact


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
            atom_rows = [
                _read_atom_line(trajectory_file.readline())
                for _atom_index in range(atom_count)
            ]
            atom_table = np.asarray(atom_rows, dtype=float)
            order = np.argsort(atom_table[:, LAMMPS_COLUMN_ID])
            frames.append(
                LammpsDumpFrame(
                    timestep=timestep,
                    box_bounds_A=box_bounds_A,
                    atom_table=atom_table[order],
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


def _read_atom_line(line: str) -> tuple[float, ...]:
    pieces = line.split()
    if len(pieces) != LAMMPS_DUMP_COLUMN_COUNT:
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


def _assign_center_states(
    center_frames: tuple[ChargedCenterFrame, ...],
    center_catalog: ChargedCenterCatalog,
    thresholds: AssociationThresholds,
    records,
    mixture,
) -> tuple[tuple[str, ...], np.ndarray]:
    state_labels: list[str] = []
    state_index_by_label: dict[str, int] = {}
    state_indices = np.zeros(
        (len(center_frames), center_catalog.molecule_ids.size),
        dtype=int,
    )
    cation_indices = _role_indices(center_catalog, ROLE_CATION)
    anion_indices = _role_indices(center_catalog, ROLE_ANION)
    for frame_index, center_frame in enumerate(center_frames):
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
            label = _active_sparse_state_label_for_center(
                records=records,
                mixture=mixture,
                center_frame=center_frame,
                center_catalog=center_catalog,
                center_index=int(center_index),
                counterion_index=nearest_anion_index,
                distances_A=pair_distances_A[local_cation_index],
                thresholds=thresholds,
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
            label = _active_sparse_state_label_for_center(
                records=records,
                mixture=mixture,
                center_frame=center_frame,
                center_catalog=center_catalog,
                center_index=int(center_index),
                counterion_index=nearest_cation_index,
                distances_A=pair_distances_A[:, local_anion_index],
                thresholds=thresholds,
            )
            state_indices[frame_index, center_index] = _state_index_for_label(
                label,
                state_labels,
                state_index_by_label,
            )
    return tuple(state_labels), state_indices


def _state_index_for_label(
    state_label: str,
    state_labels: list[str],
    state_index_by_label: dict[str, int],
) -> int:
    if state_label not in state_index_by_label:
        state_index_by_label[state_label] = len(state_labels)
        state_labels.append(state_label)
    return state_index_by_label[state_label]


def _active_sparse_state_label_for_center(
    records,
    mixture,
    center_frame: ChargedCenterFrame,
    center_catalog: ChargedCenterCatalog,
    center_index: int,
    counterion_index: int,
    distances_A: np.ndarray,
    thresholds: AssociationThresholds,
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
        center_frame,
        center_catalog,
        cation_index,
        anion_index,
    )
    coordinate_values = _reduced_coordinate_values_from_center_observation(
        records,
        mixture,
        configuration,
        pair_label,
        distances_A,
        thresholds,
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


def _reduced_coordinate_values_from_center_observation(
    records,
    mixture,
    configuration: SiteConfiguration,
    pair_label: str,
    distances_A: np.ndarray,
    thresholds: AssociationThresholds,
) -> dict[str, float]:
    cation_position_m = configuration.positions_m[0]
    anion_position_m = configuration.positions_m[1]
    pair_distance_m = float(np.linalg.norm(anion_position_m - cation_position_m))
    return {
        generator_construction.ReducedCoordinate.LI_ANION_DISTANCE.value: pair_distance_m,
        generator_construction.ReducedCoordinate.LI_SOLVENT_COORDINATION.value: 0.0,
        generator_construction.ReducedCoordinate.LI_LIGAND_COORDINATION.value: 0.0,
        generator_construction.ReducedCoordinate.LI_ANION_COORDINATION.value: (
            0.0
            if pair_label == PairBasin.FREE.value
            else generator_construction._coordination_cutoff(records, "Li_anion")
        ),
        generator_construction.ReducedCoordinate.ANION_ORIENTATION.value: 0.0,
        generator_construction.ReducedCoordinate.LOCAL_PACKING_FRACTION.value: 0.0,
        generator_construction.ReducedCoordinate.LOCAL_IONIC_STRENGTH.value: (
            mixture.ionic_strength_mol_m3
        ),
        generator_construction.ReducedCoordinate.LOCAL_DIELECTRIC.value: (
            mixture.dielectric_constant
        ),
        generator_construction.ReducedCoordinate.LOCAL_VISCOSITY.value: (
            mixture.viscosity_Pa_s
        ),
        generator_construction.ReducedCoordinate.ATMOSPHERE_POLARIZATION.value: 0.0,
        generator_construction.ReducedCoordinate.CAGE_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.PARTNER_RESIDENCE_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.CLUSTER_COORDINATE.value: (
            _cluster_coordinate_for_counterion_distances(distances_A, thresholds)
        ),
        generator_construction.ReducedCoordinate.IDENTITY_COORDINATE.value: 0.0,
        generator_construction.ReducedCoordinate.STRUCTURAL_HOP_COORDINATE.value: 0.0,
    }


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
) -> np.ndarray:
    displacements: list[np.ndarray] = []
    for frame_index in range(len(center_frames) - 1):
        center_displacements_A = (
            center_frames[frame_index + 1].positions_A
            - center_frames[frame_index].positions_A
        )
        charge_displacements_A = (
            center_catalog.formal_charges_e[:, np.newaxis] * center_displacements_A
        )
        displacements.extend(
            charge_displacements_A[center_index] * ANGSTROM_TO_M
            for center_index in range(center_catalog.molecule_ids.size)
        )
    return np.asarray(displacements, dtype=float)


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


def _primitive_arrays_with_component_solvable_first_moments(
    primitive_arrays: dict[str, np.ndarray],
    primitive_set: ProjectedGeneratorPrimitiveSet,
) -> dict[str, np.ndarray]:
    repaired_arrays = {
        primitive_name: np.asarray(primitive_value, dtype=float).copy()
        for primitive_name, primitive_value in primitive_arrays.items()
    }
    repaired_arrays["transition_first_moments_d_ij_m"] = np.asarray(
        primitive_set.diagnostics.component_solvable_projection.projected_first_moments_d_ij_m,
        dtype=float,
    )
    return repaired_arrays


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


def _projected_primitive_extraction_diagnostics(
    primitive_set: ProjectedGeneratorPrimitiveSet,
    component_solvable_projection_applied: bool,
) -> dict:
    component_solvable_projection = (
        primitive_set.diagnostics.component_solvable_projection
    )
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
        "total_transport_concentration_mol_m3": float(
            primitive_set.diagnostics.total_transport_concentration_mol_m3
        ),
        "component_drift_residuals": _component_drift_residual_records(
            primitive_set.diagnostics.component_drift_residuals
        ),
        "finite_process_legality": {
            "maximum_detailed_balance_residual_mol_m3_s": float(
                primitive_set.diagnostics.finite_process_legality.maximum_detailed_balance_residual_mol_m3_s
            ),
            "component_drift_residuals": _component_drift_residual_records(
                primitive_set.diagnostics.finite_process_legality.component_drift_residuals
            ),
        },
        "component_solvable_projection": {
            "applied_to_primitives": bool(component_solvable_projection_applied),
            "maximum_removed_first_moment_norm_m": float(
                component_solvable_projection.maximum_removed_first_moment_norm_m
            ),
            "removed_first_moments_d_ij_m": [
                [list(vector) for vector in row]
                for row in component_solvable_projection.removed_first_moments_d_ij_m
            ],
            "projected_component_drift_residuals": _component_drift_residual_records(
                component_solvable_projection.projected_component_drift_residuals
            ),
        },
    }


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
