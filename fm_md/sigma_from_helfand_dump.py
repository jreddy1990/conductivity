"""Estimate Einstein--Helfand conductivity from a LAMMPS molecular trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from constants import FEMTOSECOND_TO_S, S_M_TO_MS_CM
from conductivity.physical_library.extract_projected_primitives import (
    charge_helfand_series_from_lammps_dump,
)
from conductivity.physical_library.microscopic_convergence import (
    estimate_adaptive_einstein_helfand_conductivity,
)
from utils.strict_validation import read_json_object, write_json_object


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--composition-json", required=True, type=Path)
    parser.add_argument("--dt-fs", required=True, type=float)
    parser.add_argument("--trajectory-dump-stride-steps", required=True, type=int)
    parser.add_argument("--stationary-start-frame-index", required=True, type=int)
    parser.add_argument("--confidence-level", required=True, type=float)
    parser.add_argument("--output-json", required=True, type=Path)
    arguments = parser.parse_args()

    composition = read_json_object(arguments.composition_json, "MD composition")
    helfand_moments_C_m, box_volumes_m3 = charge_helfand_series_from_lammps_dump(
        trajectory_path=arguments.trajectory,
    )
    if arguments.stationary_start_frame_index < 0:
        raise ValueError("stationary_start_frame_index must be nonnegative")
    helfand_moments_C_m = helfand_moments_C_m[
        arguments.stationary_start_frame_index:
    ]
    box_volumes_m3 = box_volumes_m3[arguments.stationary_start_frame_index:]
    frame_interval_s = (
        arguments.dt_fs
        * arguments.trajectory_dump_stride_steps
        * FEMTOSECOND_TO_S
    )
    estimate = estimate_adaptive_einstein_helfand_conductivity(
        helfand_moment_C_m=helfand_moments_C_m,
        frame_interval_s=frame_interval_s,
        volume_m3=float(np.mean(box_volumes_m3)),
        temperature_K=float(composition["temperature_K"]),
        confidence_level=arguments.confidence_level,
    )
    write_json_object(
        arguments.output_json,
        {
            "sigma_S_m": estimate.conductivity_S_m,
            "sigma_mS_cm": estimate.conductivity_S_m * S_M_TO_MS_CM,
            "standard_error_S_m": estimate.standard_error_S_m,
            "standard_error_mS_cm": (
                estimate.standard_error_S_m * S_M_TO_MS_CM
            ),
            "accepted_windows": tuple(
                asdict(window) for window in estimate.accepted_windows
            ),
        },
        "Einstein-Helfand estimate",
    )
    print(
        f"sigma = {estimate.conductivity_S_m * S_M_TO_MS_CM:.6g} mS/cm; "
        f"standard error = "
        f"{estimate.standard_error_S_m * S_M_TO_MS_CM:.6g} mS/cm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
