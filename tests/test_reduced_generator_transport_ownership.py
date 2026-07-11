from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect

import numpy as np
import pytest

from conductivity.physical_library.reduced_generator import (
    TransportOwnership,
    TransportOwnershipRecord,
)


def test_transport_ownership_record_has_explicit_required_contract() -> None:
    signature = inspect.signature(TransportOwnershipRecord)

    assert tuple(signature.parameters) == (
        "state",
        "label",
        "owner",
        "gradient",
        "physical_basis",
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert tuple(field.name for field in fields(TransportOwnershipRecord)) == tuple(
        signature.parameters
    )


def test_transport_ownership_record_carries_immutable_physical_evidence() -> None:
    source_gradient = np.asarray([1.0, -2.0, 3.0], dtype=float)
    record = TransportOwnershipRecord(
        state=2,
        label="solvent_exchange_normal",
        owner=TransportOwnership.TRANSITION_DISPLACEMENT,
        gradient=source_gradient,
        physical_basis="committor-normal reactive flux",
    )
    source_gradient[0] = 9.0

    assert record.state == 2
    assert record.label == "solvent_exchange_normal"
    assert record.owner is TransportOwnership.TRANSITION_DISPLACEMENT
    assert record.gradient == pytest.approx(np.asarray([1.0, -2.0, 3.0]))
    assert record.physical_basis == "committor-normal reactive flux"
    with pytest.raises(ValueError, match="read-only"):
        record.gradient[0] = 4.0
    with pytest.raises(FrozenInstanceError):
        record.label = "replacement"


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_type", "message"),
    (
        ("state", -1, ValueError, "state must be nonnegative"),
        ("state", True, TypeError, "state must be an integer"),
        ("label", " ", ValueError, "label must not be empty"),
        ("owner", "dc_self", TypeError, "owner must be a TransportOwnership"),
        ("gradient", np.asarray([]), ValueError, "gradient must not be empty"),
        ("gradient", np.asarray([np.nan]), ValueError, "gradient must be a finite 1D array"),
        ("physical_basis", "", ValueError, "physical_basis must not be empty"),
    ),
)
def test_transport_ownership_record_rejects_invalid_evidence(
    field_name: str,
    field_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    arguments = {
        "state": 0,
        "label": "self_tangent",
        "owner": TransportOwnership.DC_SELF,
        "gradient": np.asarray([1.0, 0.0]),
        "physical_basis": "basin-tangent charge motion",
    }
    arguments[field_name] = field_value

    with pytest.raises(error_type, match=message):
        TransportOwnershipRecord(
            state=arguments["state"],
            label=arguments["label"],
            owner=arguments["owner"],
            gradient=arguments["gradient"],
            physical_basis=arguments["physical_basis"],
        )
