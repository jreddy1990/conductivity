from __future__ import annotations

import pytest

from conductivity.physical_library.generator_construction import (
    charge_covariance_mobility_from_center_matrix,
)


def test_charge_covariance_uses_center_mobility_matrix() -> None:
    charge_numbers = (1.0, -1.0)

    uncorrelated_charge_mobility = charge_covariance_mobility_from_center_matrix(
        charge_numbers,
        ((1.0e-10, 0.0), (0.0, 2.0e-10)),
    )
    perfect_comotion_charge_mobility = charge_covariance_mobility_from_center_matrix(
        charge_numbers,
        ((2.0e-10, 2.0e-10), (2.0e-10, 2.0e-10)),
    )
    anticorrelated_charge_mobility = charge_covariance_mobility_from_center_matrix(
        charge_numbers,
        ((1.0e-10, -0.4e-10), (-0.4e-10, 1.0e-10)),
    )

    assert uncorrelated_charge_mobility == pytest.approx(3.0e-10)
    assert perfect_comotion_charge_mobility == pytest.approx(0.0)
    assert anticorrelated_charge_mobility == pytest.approx(2.8e-10)
