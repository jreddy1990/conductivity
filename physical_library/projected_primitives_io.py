"""YAML I/O for projected conductivity primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from conductivity.physical_library.projected_analytical_conductivity import (
    ProjectedConductivityResult,
    ProjectedPrimitiveInput,
    compute_projected_analytical_conductivity_from_primitives,
)

Array = np.ndarray
PRIMITIVE_SCHEMA = "projected_primitives_v1"


@dataclass(frozen=True)
class ProjectedPrimitiveArtifact:
    schema: str
    state_labels: tuple[str, ...]
    primitive_input: ProjectedPrimitiveInput


def read_projected_primitive_yaml(path: Path) -> ProjectedPrimitiveArtifact:
    """Read projected primitive tensors from YAML."""

    record = yaml.safe_load(path.read_text())
    if not isinstance(record, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    schema = str(record["schema"])
    state_labels = tuple(str(label) for label in record["state_labels"])
    primitives = record["primitives"]
    memory_matrix_A = _array(
        primitives["mori_memory_matrix_A"],
        "mori_memory_matrix_A",
    )
    current_coupling_matrix_h = _array(
        primitives["mori_current_coupling_matrix_h"],
        "mori_current_coupling_matrix_h",
    )
    if memory_matrix_A.size == 0:
        memory_matrix_A = np.zeros((0, 0), dtype=float)
    if current_coupling_matrix_h.size == 0:
        current_coupling_matrix_h = np.zeros((0, 3), dtype=float)
    primitive_input = ProjectedPrimitiveInput(
        state_concentrations_mol_m3=_array(
            primitives["state_concentrations_mol_m3"],
            "state_concentrations_mol_m3",
        ),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=_array(
            primitives["symmetric_capacity_fluxes_K_ij_mol_m3_s"],
            "symmetric_capacity_fluxes_K_ij_mol_m3_s",
        ),
        transition_first_moments_d_ij_m=_array(
            primitives["transition_first_moments_d_ij_m"],
            "transition_first_moments_d_ij_m",
        ),
        transition_second_moments_M_ij_m2=_array(
            primitives["transition_second_moments_M_ij_m2"],
            "transition_second_moments_M_ij_m2",
        ),
        self_current_tensors_D_self_i_m2_s=_array(
            primitives["self_current_tensors_D_self_i_m2_s"],
            "self_current_tensors_D_self_i_m2_s",
        ),
        mori_memory_matrix_A=memory_matrix_A,
        mori_current_coupling_matrix_h=current_coupling_matrix_h,
        temperature_K=float(record["temperature_K"]),
        volume_m3=float(record["volume_m3"]),
    )
    return ProjectedPrimitiveArtifact(
        schema=schema,
        state_labels=state_labels,
        primitive_input=primitive_input,
    )


def write_projected_primitive_yaml(
    path: Path,
    state_labels: tuple[str, ...],
    primitive_input: ProjectedPrimitiveInput,
    conductivity_result: ProjectedConductivityResult,
) -> None:
    """Write projected primitive tensors and readout diagnostics to YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": PRIMITIVE_SCHEMA,
        "state_labels": list(state_labels),
        "temperature_K": float(primitive_input.temperature_K),
        "volume_m3": float(primitive_input.volume_m3),
        "primitives": {
            "state_concentrations_mol_m3": np.asarray(
                primitive_input.state_concentrations_mol_m3,
                dtype=float,
            ).tolist(),
            "symmetric_capacity_fluxes_K_ij_mol_m3_s": np.asarray(
                primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s,
                dtype=float,
            ).tolist(),
            "transition_first_moments_d_ij_m": np.asarray(
                primitive_input.transition_first_moments_d_ij_m,
                dtype=float,
            ).tolist(),
            "transition_second_moments_M_ij_m2": np.asarray(
                primitive_input.transition_second_moments_M_ij_m2,
                dtype=float,
            ).tolist(),
            "self_current_tensors_D_self_i_m2_s": np.asarray(
                primitive_input.self_current_tensors_D_self_i_m2_s,
                dtype=float,
            ).tolist(),
            "mori_memory_matrix_A": np.asarray(
                primitive_input.mori_memory_matrix_A,
                dtype=float,
            ).tolist(),
            "mori_current_coupling_matrix_h": np.asarray(
                primitive_input.mori_current_coupling_matrix_h,
                dtype=float,
            ).tolist(),
        },
        "sigma_mS_cm": float(conductivity_result.sigma_mS_cm),
        "sigma_S_m": float(conductivity_result.sigma_S_m),
    }
    path.write_text(yaml.safe_dump(record, sort_keys=False))


def compute_conductivity_from_primitive_yaml(path: Path) -> ProjectedConductivityResult:
    """Read primitive YAML and run the projected conductivity readout."""

    artifact = read_projected_primitive_yaml(path)
    primitive_input = artifact.primitive_input
    return compute_projected_analytical_conductivity_from_primitives(
        primitive_input.state_concentrations_mol_m3,
        primitive_input.symmetric_capacity_fluxes_K_ij_mol_m3_s,
        primitive_input.transition_first_moments_d_ij_m,
        primitive_input.transition_second_moments_M_ij_m2,
        primitive_input.self_current_tensors_D_self_i_m2_s,
        primitive_input.mori_memory_matrix_A,
        primitive_input.mori_current_coupling_matrix_h,
        primitive_input.temperature_K,
        primitive_input.volume_m3,
    )


def _array(value, label: str) -> Array:
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result
