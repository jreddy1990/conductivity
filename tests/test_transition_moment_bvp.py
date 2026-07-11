from __future__ import annotations

import numpy as np
import pytest

from conductivity.physical_library.transition_moment_bvp import (
    EndpointTransportMomentInput,
    build_endpoint_transport_moments,
)


def test_directed_endpoint_moment_contains_only_crossing_geometry() -> None:
    displacement_m = np.asarray([3.0e-10, 0.0, 0.0], dtype=float)

    first_moment, second_moment = build_endpoint_transport_moments(
        EndpointTransportMomentInput(
            endpoint_displacement_m=displacement_m,
            directed_endpoint=True,
        )
    )

    assert first_moment == pytest.approx(displacement_m)
    assert second_moment == pytest.approx(np.outer(displacement_m, displacement_m))


def test_isotropic_endpoint_moment_contains_only_orientation_averaged_geometry() -> None:
    first_moment, second_moment = build_endpoint_transport_moments(
        EndpointTransportMomentInput(
            endpoint_displacement_m=np.asarray([0.0, 2.0e-10, 0.0], dtype=float),
            directed_endpoint=False,
        )
    )

    assert np.array_equal(first_moment, np.zeros(3, dtype=float))
    geometric_trace_m2 = (2.0e-10) ** 2
    assert second_moment == pytest.approx(
        np.eye(3) * geometric_trace_m2 / 3.0
    )
