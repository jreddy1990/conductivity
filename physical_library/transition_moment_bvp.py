"""Endpoint-conditioned charge-transport moments for reduced transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conductivity.physical_library.projected_analytical_conductivity import CARTESIAN
from utils.strict_validation import finite_vector, positive_float

Array = np.ndarray


@dataclass(frozen=True)
class EndpointTransportMomentInput:
    endpoint_displacement_m: Array
    directed_endpoint: bool


def build_endpoint_transport_moments(
    moment_input: EndpointTransportMomentInput,
) -> tuple[Array, Array]:
    """Build moments owned by the discrete endpoint crossing geometry."""

    displacement = finite_vector(
        moment_input.endpoint_displacement_m,
        "endpoint_displacement_m",
    )
    if displacement.shape != (CARTESIAN,):
        raise ValueError("endpoint_displacement_m must have shape (3,)")
    positive_float(float(np.linalg.norm(displacement)), "endpoint_displacement_length_m")
    displacement_outer = np.outer(displacement, displacement)
    if moment_input.directed_endpoint:
        return displacement, displacement_outer
    return (
        np.zeros(CARTESIAN, dtype=float),
        np.eye(CARTESIAN, dtype=float)
        * (float(np.trace(displacement_outer)) / CARTESIAN),
    )
