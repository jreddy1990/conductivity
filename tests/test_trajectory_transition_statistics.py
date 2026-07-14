from __future__ import annotations

import numpy as np
import pytest

from conductivity.physical_library.trajectory_transition_statistics import (
    DESTINATION_BASIN_LABEL,
    INTERIOR_BASIN_LABEL,
    SOURCE_BASIN_LABEL,
    estimate_empirical_committor,
    estimate_transition_moments_by_replica,
)


def test_empirical_committor_reports_bins_capacity_and_residual() -> None:
    forward_coordinate = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    reverse_coordinate = forward_coordinate[::-1]
    reaction_coordinate = np.tile(
        np.concatenate((forward_coordinate, reverse_coordinate)), 20
    )
    forward_labels = np.asarray(
        [
            SOURCE_BASIN_LABEL,
            INTERIOR_BASIN_LABEL,
            INTERIOR_BASIN_LABEL,
            INTERIOR_BASIN_LABEL,
            INTERIOR_BASIN_LABEL,
            DESTINATION_BASIN_LABEL,
        ]
    )
    basin_labels = np.tile(
        np.concatenate((forward_labels, forward_labels[::-1])), 20
    )

    estimate = estimate_empirical_committor(
        reaction_coordinate=reaction_coordinate,
        basin_labels=basin_labels,
        frame_interval_s=1.0e-12,
        bin_edges=np.asarray([0.0, 0.5, 1.0]),
    )

    assert len(estimate.bins) == 2
    assert estimate.symmetric_capacity_events_per_s > 0.0
    assert np.isfinite(estimate.maximum_standardized_residual)


def test_transition_moments_use_replica_uncertainty_and_edge_orientation() -> None:
    displacement_m = 1.0e-10
    estimates = estimate_transition_moments_by_replica(
        from_state_labels=("A", "B", "A", "B"),
        to_state_labels=("B", "A", "B", "A"),
        replica_ids=("r1", "r1", "r2", "r2"),
        charge_displacements_m=np.asarray(
            [
                [displacement_m, 0.0, 0.0],
                [-displacement_m, 0.0, 0.0],
                [displacement_m, 0.0, 0.0],
                [-displacement_m, 0.0, 0.0],
            ]
        ),
        replica_durations_s=(("r1", 2.0e-9), ("r2", 2.0e-9)),
    )

    assert len(estimates) == 1
    estimate = estimates[0]
    assert estimate.first_moment_m[0] == pytest.approx(displacement_m)
    assert estimate.second_moment_m2[0][0] == pytest.approx(displacement_m**2)
    assert estimate.capacity_standard_error_events_per_s == 0.0
