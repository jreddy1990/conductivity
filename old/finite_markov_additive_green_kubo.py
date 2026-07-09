"""Concentration-weighted Markov-additive Green-Kubo conductivity readout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from constants import F, R, S_M_TO_MS_CM
from conductivity.old.finite_mori_conductivity import (
    CARTESIAN_AXIS_COUNT,
    ProjectedMoriConductivityInput,
    ProjectedMoriConductivityResult,
    compute_projected_mori_conductivity,
)


HALF_JUMP_VARIANCE_FACTOR = 0.5
ZERO_VALUE = 0.0
FINITE_MARKOV_ADDITIVE_TOLERANCE = math.sqrt(np.finfo(float).eps)


@dataclass(frozen=True)
class MarkovAdditiveEvent:
    from_state_index: int
    to_state_index: int
    rate_s_inv: float
    charge_displacement_m: tuple[float, float, float]
    label: str
    family_label: str


@dataclass(frozen=True)
class MarkovAdditiveEventFamilyAttribution:
    family_label: str
    direct_sigma_mS_cm: float
    self_corrector_sigma_mS_cm: float
    marginal_corrector_sigma_mS_cm: float
    marginal_net_sigma_mS_cm: float
    direct_fraction: float
    marginal_net_fraction: float


@dataclass(frozen=True)
class MarkovAdditiveConductivityInput:
    state_labels: tuple[str, ...]
    state_concentrations_mol_m3: np.ndarray
    events: tuple[MarkovAdditiveEvent, ...]
    temperature_K: float


@dataclass(frozen=True)
class ConcentrationGeneratorValidation:
    row_sum_residual: float
    stationary_residual_mol_m3_s: float
    detailed_balance_residual_mol_m3_s: float
    minimum_offdiagonal_rate_s_inv: float
    concentration_sum_mol_m3: float


@dataclass(frozen=True)
class MarkovAdditiveConductivityResult:
    generator_s_inv: np.ndarray
    validation: ConcentrationGeneratorValidation
    event_reversal_residual_mol_m3_s: float
    drift_by_state_m_s: np.ndarray
    corrector_mori_input: ProjectedMoriConductivityInput
    corrector_mori_result: ProjectedMoriConductivityResult
    direct_axis_density_m2_s_mol_m3: tuple[float, float, float]
    corrector_axis_density_m2_s_mol_m3: tuple[float, float, float]
    effective_axis_density_m2_s_mol_m3: tuple[float, float, float]
    direct_sigma_S_m: float
    corrector_sigma_S_m: float
    sigma_S_m: float
    direct_sigma_mS_cm: float
    corrector_sigma_mS_cm: float
    sigma_mS_cm: float
    minimum_effective_axis_density_m2_s_mol_m3: float


def build_generator_from_events(
    state_count: int,
    events: Sequence[MarkovAdditiveEvent],
) -> np.ndarray:
    """Build a CTMC generator from off-diagonal Markov-additive events."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    if len(events) == 0:
        raise ValueError("events must contain at least one event")
    generator_matrix = np.zeros((state_count, state_count), dtype=float)
    for event in events:
        _validate_event_indices(event, state_count)
        _positive_float(event.rate_s_inv, f"{event.label}.rate_s_inv")
        _validated_displacement(event.charge_displacement_m, event.label)
        if event.from_state_index != event.to_state_index:
            generator_matrix[event.from_state_index, event.to_state_index] += (
                event.rate_s_inv
            )
    row_exit_rates = np.sum(generator_matrix, axis=1)
    np.fill_diagonal(generator_matrix, -row_exit_rates)
    return generator_matrix


def validate_concentration_reversible_generator(
    generator_matrix_s_inv: np.ndarray,
    state_concentrations_mol_m3: np.ndarray,
) -> ConcentrationGeneratorValidation:
    """Validate row conservation, stationarity, and concentration detailed balance."""

    generator_matrix = _validated_generator_matrix(generator_matrix_s_inv)
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        generator_matrix.shape[0],
    )
    tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    row_sum_residual = float(np.max(np.abs(np.sum(generator_matrix, axis=1))))
    offdiagonal_rates = generator_matrix[~np.eye(generator_matrix.shape[0], dtype=bool)]
    minimum_offdiagonal_rate = (
        float(np.min(offdiagonal_rates)) if offdiagonal_rates.size else ZERO_VALUE
    )
    maximum_diagonal_entry = float(np.max(np.diag(generator_matrix)))
    stationary_residual = float(
        np.max(np.abs(state_concentrations @ generator_matrix))
    )
    detailed_balance_matrix = (
        state_concentrations[:, None] * generator_matrix
        - state_concentrations[None, :] * generator_matrix.T
    )
    detailed_balance_residual = float(np.max(np.abs(detailed_balance_matrix)))
    if row_sum_residual > tolerance:
        raise ValueError(f"generator row-sum residual {row_sum_residual} exceeds {tolerance}")
    if minimum_offdiagonal_rate < -tolerance:
        raise ValueError("generator off-diagonal entries must be nonnegative")
    if maximum_diagonal_entry > tolerance:
        raise ValueError("generator diagonal entries must be nonpositive")
    if stationary_residual > tolerance:
        raise ValueError(
            f"stationary concentration residual {stationary_residual} exceeds {tolerance}"
        )
    if detailed_balance_residual > tolerance:
        raise ValueError(
            f"detailed-balance residual {detailed_balance_residual} exceeds {tolerance}"
        )
    return ConcentrationGeneratorValidation(
        row_sum_residual=row_sum_residual,
        stationary_residual_mol_m3_s=stationary_residual,
        detailed_balance_residual_mol_m3_s=detailed_balance_residual,
        minimum_offdiagonal_rate_s_inv=minimum_offdiagonal_rate,
        concentration_sum_mol_m3=float(np.sum(state_concentrations)),
    )


def validate_event_displacement_reversibility(
    events: Sequence[MarkovAdditiveEvent],
    state_concentrations_mol_m3: np.ndarray,
    state_count: int,
) -> float:
    """Validate reverse flux symmetry for every nonzero displacement event."""

    if len(events) == 0:
        raise ValueError("events must contain at least one event")
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        state_count,
    )
    weighted_flux_by_event_key: dict[tuple[int, int, tuple[float, float, float]], float] = {}
    for event in events:
        _validate_event_indices(event, state_count)
        event_rate_s_inv = _positive_float(event.rate_s_inv, f"{event.label}.rate_s_inv")
        displacement_array = _validated_displacement(
            event.charge_displacement_m,
            event.label,
        )
        if _is_zero_displacement(displacement_array):
            continue
        displacement_key = _displacement_key(displacement_array)
        event_key = (
            event.from_state_index,
            event.to_state_index,
            displacement_key,
        )
        weighted_flux = (
            state_concentrations[event.from_state_index]
            * event_rate_s_inv
        )
        weighted_flux_by_event_key[event_key] = (
            weighted_flux_by_event_key.get(event_key, ZERO_VALUE)
            + weighted_flux
        )
    if not weighted_flux_by_event_key:
        return ZERO_VALUE
    maximum_weighted_flux = max(
        abs(weighted_flux)
        for weighted_flux in weighted_flux_by_event_key.values()
    )
    tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        maximum_weighted_flux,
    )
    maximum_reversal_residual = ZERO_VALUE
    for event_key, weighted_flux in weighted_flux_by_event_key.items():
        from_state_index, to_state_index, displacement_key = event_key
        reverse_displacement_key = tuple(
            _canonical_float_for_key(-component)
            for component in displacement_key
        )
        reverse_key = (
            to_state_index,
            from_state_index,
            reverse_displacement_key,
        )
        reverse_weighted_flux = weighted_flux_by_event_key.get(
            reverse_key,
            ZERO_VALUE,
        )
        reversal_residual = abs(weighted_flux - reverse_weighted_flux)
        maximum_reversal_residual = max(
            maximum_reversal_residual,
            reversal_residual,
        )
    if maximum_reversal_residual > tolerance:
        raise ValueError(
            "event displacement reverse residual "
            f"{maximum_reversal_residual} exceeds {tolerance}"
        )
    return float(maximum_reversal_residual)


def compute_markov_additive_green_kubo_conductivity(
    markov_additive_input: MarkovAdditiveConductivityInput,
) -> MarkovAdditiveConductivityResult:
    """Evaluate sigma = direct jump variance minus Mori corrector."""

    state_labels = tuple(markov_additive_input.state_labels)
    if len(state_labels) == 0:
        raise ValueError("state_labels must contain at least one state")
    if len(set(state_labels)) != len(state_labels):
        raise ValueError("state_labels must be unique")
    state_concentrations = _validated_state_concentrations(
        markov_additive_input.state_concentrations_mol_m3,
        len(state_labels),
    )
    temperature_K = _positive_float(markov_additive_input.temperature_K, "temperature_K")
    generator_matrix = build_generator_from_events(
        len(state_labels),
        markov_additive_input.events,
    )
    validation = validate_concentration_reversible_generator(
        generator_matrix,
        state_concentrations,
    )
    event_reversal_residual = validate_event_displacement_reversibility(
        markov_additive_input.events,
        state_concentrations,
        len(state_labels),
    )
    axis_count = int(CARTESIAN_AXIS_COUNT)
    direct_axis_density = np.zeros(axis_count, dtype=float)
    drift_by_state = np.zeros((len(state_labels), axis_count), dtype=float)
    for event in markov_additive_input.events:
        displacement_array = np.asarray(event.charge_displacement_m, dtype=float)
        direct_axis_density += (
            HALF_JUMP_VARIANCE_FACTOR
            * state_concentrations[event.from_state_index]
            * event.rate_s_inv
            * displacement_array
            * displacement_array
        )
        drift_by_state[event.from_state_index, :] += (
            event.rate_s_inv * displacement_array
        )
    stationary_drift = state_concentrations @ drift_by_state
    drift_tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    if float(np.max(np.abs(stationary_drift))) > drift_tolerance:
        raise ValueError("Markov-additive process has nonzero stationary drift")

    symmetrized_energy_matrix = _symmetrized_energy_matrix(
        generator_matrix,
        state_concentrations,
    )
    current_coupling_matrix = (
        np.sqrt(state_concentrations)[:, None] * drift_by_state
    ).T
    beta_factor = F * F / (R * temperature_K)
    corrector_mori_input = ProjectedMoriConductivityInput(
        direct_energy_matrix=np.zeros_like(symmetrized_energy_matrix),
        memory_self_energy_matrix=symmetrized_energy_matrix,
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=beta_factor,
    )
    corrector_mori_result = compute_projected_mori_conductivity(corrector_mori_input)
    corrector_axis_density = np.asarray(
        corrector_mori_result.quadratic_form_by_axis,
        dtype=float,
    )
    effective_axis_density = direct_axis_density - corrector_axis_density
    density_tolerance = FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(direct_axis_density))),
        float(np.max(np.abs(corrector_axis_density))),
    )
    if float(np.min(effective_axis_density)) < -density_tolerance:
        raise ValueError("Markov-additive effective axis density became negative")
    effective_axis_density = np.asarray(
        [
            ZERO_VALUE if abs(value) <= density_tolerance else float(value)
            for value in effective_axis_density
        ],
        dtype=float,
    )
    direct_sigma_S_m = beta_factor * float(np.sum(direct_axis_density)) / CARTESIAN_AXIS_COUNT
    corrector_sigma_S_m = corrector_mori_result.sigma_S_m
    sigma_S_m = direct_sigma_S_m - corrector_sigma_S_m
    if sigma_S_m < -density_tolerance:
        raise ValueError("Markov-additive conductivity became negative")
    if abs(sigma_S_m) <= density_tolerance:
        sigma_S_m = ZERO_VALUE
    return MarkovAdditiveConductivityResult(
        generator_s_inv=generator_matrix,
        validation=validation,
        event_reversal_residual_mol_m3_s=event_reversal_residual,
        drift_by_state_m_s=drift_by_state,
        corrector_mori_input=corrector_mori_input,
        corrector_mori_result=corrector_mori_result,
        direct_axis_density_m2_s_mol_m3=tuple(float(value) for value in direct_axis_density),
        corrector_axis_density_m2_s_mol_m3=tuple(float(value) for value in corrector_axis_density),
        effective_axis_density_m2_s_mol_m3=tuple(float(value) for value in effective_axis_density),
        direct_sigma_S_m=float(direct_sigma_S_m),
        corrector_sigma_S_m=float(corrector_sigma_S_m),
        sigma_S_m=float(sigma_S_m),
        direct_sigma_mS_cm=float(direct_sigma_S_m * S_M_TO_MS_CM),
        corrector_sigma_mS_cm=float(corrector_sigma_S_m * S_M_TO_MS_CM),
        sigma_mS_cm=float(sigma_S_m * S_M_TO_MS_CM),
        minimum_effective_axis_density_m2_s_mol_m3=float(np.min(effective_axis_density)),
    )


def compute_markov_additive_event_family_attribution(
    markov_result: MarkovAdditiveConductivityResult,
    events: tuple[MarkovAdditiveEvent, ...],
    state_concentrations_mol_m3: np.ndarray,
    event_family_by_label: Mapping[str, str],
    temperature_K: float,
) -> tuple[MarkovAdditiveEventFamilyAttribution, ...]:
    """Attribute direct and corrector terms to event families with fixed Q."""

    if len(events) == 0:
        raise ValueError("events must contain at least one event")
    temperature_K = _positive_float(temperature_K, "temperature_K")
    generator_matrix = _validated_generator_matrix(markov_result.generator_s_inv)
    state_concentrations = _validated_state_concentrations(
        state_concentrations_mol_m3,
        generator_matrix.shape[0],
    )
    validate_concentration_reversible_generator(generator_matrix, state_concentrations)
    validate_event_displacement_reversibility(
        events,
        state_concentrations,
        generator_matrix.shape[0],
    )
    event_generator_matrix = build_generator_from_events(
        generator_matrix.shape[0],
        events,
    )
    generator_tolerance = _matrix_tolerance(generator_matrix, state_concentrations)
    generator_difference = float(
        np.max(np.abs(event_generator_matrix - generator_matrix))
    )
    if generator_difference > generator_tolerance:
        raise ValueError(
            f"event-family attribution generator mismatch {generator_difference} "
            f"exceeds {generator_tolerance}"
        )
    family_labels = _event_family_labels(events, event_family_by_label)
    axis_count = int(CARTESIAN_AXIS_COUNT)
    direct_density_by_family: dict[str, np.ndarray] = {
        family_label: np.zeros(axis_count, dtype=float)
        for family_label in family_labels
    }
    drift_by_family: dict[str, np.ndarray] = {
        family_label: np.zeros((state_concentrations.shape[0], axis_count), dtype=float)
        for family_label in family_labels
    }
    for event in events:
        family_label = event_family_by_label[event.label]
        displacement_array = np.asarray(event.charge_displacement_m, dtype=float)
        direct_density_by_family[family_label] += (
            HALF_JUMP_VARIANCE_FACTOR
            * state_concentrations[event.from_state_index]
            * event.rate_s_inv
            * displacement_array
            * displacement_array
        )
        drift_by_family[family_label][event.from_state_index, :] += (
            event.rate_s_inv * displacement_array
        )

    symmetrized_energy_matrix = _symmetrized_energy_matrix(
        generator_matrix,
        state_concentrations,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized_energy_matrix)
    _validate_energy_eigenvalues(eigenvalues, "event_family_attribution.energy_matrix")
    total_drift_matrix = np.zeros((state_concentrations.shape[0], axis_count), dtype=float)
    for family_label in family_labels:
        total_drift_matrix += drift_by_family[family_label]
    beta_factor = F * F / (R * temperature_K)
    total_direct_sigma_mS_cm = markov_result.direct_sigma_mS_cm
    total_sigma_mS_cm = markov_result.sigma_mS_cm
    attributions: list[MarkovAdditiveEventFamilyAttribution] = []
    for family_label in family_labels:
        family_drift_matrix = drift_by_family[family_label]
        direct_sigma_mS_cm = (
            beta_factor
            * float(np.sum(direct_density_by_family[family_label]))
            / CARTESIAN_AXIS_COUNT
            * S_M_TO_MS_CM
        )
        self_corrector_sigma_mS_cm = _family_cross_corrector_sigma_mS_cm(
            family_drift_matrix,
            family_drift_matrix,
            state_concentrations,
            eigenvalues,
            eigenvectors,
            beta_factor,
        )
        family_total_cross_sigma_mS_cm = _family_cross_corrector_sigma_mS_cm(
            family_drift_matrix,
            total_drift_matrix,
            state_concentrations,
            eigenvalues,
            eigenvectors,
            beta_factor,
        )
        marginal_corrector_sigma_mS_cm = (
            2.0 * family_total_cross_sigma_mS_cm
            - self_corrector_sigma_mS_cm
        )
        marginal_net_sigma_mS_cm = (
            direct_sigma_mS_cm - marginal_corrector_sigma_mS_cm
        )
        direct_fraction = (
            direct_sigma_mS_cm / total_direct_sigma_mS_cm
            if total_direct_sigma_mS_cm > 0.0
            else 0.0
        )
        marginal_net_fraction = (
            marginal_net_sigma_mS_cm / total_sigma_mS_cm
            if total_sigma_mS_cm > 0.0
            else 0.0
        )
        attributions.append(
            MarkovAdditiveEventFamilyAttribution(
                family_label=family_label,
                direct_sigma_mS_cm=float(direct_sigma_mS_cm),
                self_corrector_sigma_mS_cm=float(self_corrector_sigma_mS_cm),
                marginal_corrector_sigma_mS_cm=float(marginal_corrector_sigma_mS_cm),
                marginal_net_sigma_mS_cm=float(marginal_net_sigma_mS_cm),
                direct_fraction=float(direct_fraction),
                marginal_net_fraction=float(marginal_net_fraction),
            )
        )
    return tuple(
        sorted(
            attributions,
            key=_absolute_marginal_net_sigma_mS_cm,
            reverse=True,
        )
    )


def _absolute_marginal_net_sigma_mS_cm(
    attribution: MarkovAdditiveEventFamilyAttribution,
) -> float:
    return abs(attribution.marginal_net_sigma_mS_cm)


def _event_family_labels(
    events: tuple[MarkovAdditiveEvent, ...],
    event_family_by_label: Mapping[str, str],
) -> tuple[str, ...]:
    family_labels: list[str] = []
    for event in events:
        if event.label not in event_family_by_label:
            raise ValueError(f"missing event family for event {event.label}")
        mapped_family_label = event_family_by_label[event.label]
        if mapped_family_label != event.family_label:
            raise ValueError(f"event family mapping disagrees with event {event.label}")
        if mapped_family_label == "":
            raise ValueError(f"event {event.label} has an empty family label")
        if mapped_family_label not in family_labels:
            family_labels.append(mapped_family_label)
    return tuple(family_labels)


def _family_cross_corrector_sigma_mS_cm(
    left_drift_by_state: np.ndarray,
    right_drift_by_state: np.ndarray,
    state_concentrations: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    beta_factor: float,
) -> float:
    sqrt_concentrations = np.sqrt(state_concentrations)
    left_current_coupling_matrix = (sqrt_concentrations[:, None] * left_drift_by_state).T
    right_current_coupling_matrix = (sqrt_concentrations[:, None] * right_drift_by_state).T
    cross_density_sum = 0.0
    for axis_index in range(int(CARTESIAN_AXIS_COUNT)):
        cross_density_sum += _projected_cross_form(
            eigenvalues,
            eigenvectors,
            left_current_coupling_matrix[axis_index, :],
            right_current_coupling_matrix[axis_index, :],
        )
    return float(
        beta_factor
        * cross_density_sum
        / CARTESIAN_AXIS_COUNT
        * S_M_TO_MS_CM
    )


def _projected_cross_form(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    left_current_coupling: np.ndarray,
    right_current_coupling: np.ndarray,
) -> float:
    positive_mode_mask = _positive_energy_mode_mask(eigenvalues)
    if not np.any(positive_mode_mask):
        return 0.0
    projected_left = eigenvectors[:, positive_mode_mask].T @ left_current_coupling
    projected_right = eigenvectors[:, positive_mode_mask].T @ right_current_coupling
    return float(
        math.fsum(
            float(left_value * right_value / eigenvalue)
            for left_value, right_value, eigenvalue in zip(
                projected_left,
                projected_right,
                eigenvalues[positive_mode_mask],
            )
        )
    )


def _validate_energy_eigenvalues(eigenvalues: np.ndarray, context: str) -> None:
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), FINITE_MARKOV_ADDITIVE_TOLERANCE)
    allowed_negative_eigenvalue = FINITE_MARKOV_ADDITIVE_TOLERANCE * eigenvalue_scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(f"{context} must be positive semidefinite")


def _positive_energy_mode_mask(eigenvalues: np.ndarray) -> np.ndarray:
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), FINITE_MARKOV_ADDITIVE_TOLERANCE)
    return eigenvalues > FINITE_MARKOV_ADDITIVE_TOLERANCE * eigenvalue_scale


def _symmetrized_energy_matrix(
    generator_matrix: np.ndarray,
    state_concentrations: np.ndarray,
) -> np.ndarray:
    sqrt_concentrations = np.sqrt(state_concentrations)
    inverse_sqrt_concentrations = 1.0 / sqrt_concentrations
    energy_matrix = (
        sqrt_concentrations[:, None]
        * (-generator_matrix)
        * inverse_sqrt_concentrations[None, :]
    )
    return _symmetrized_matrix(energy_matrix)


def _validated_generator_matrix(generator_matrix_s_inv: np.ndarray) -> np.ndarray:
    generator_matrix = np.asarray(generator_matrix_s_inv, dtype=float)
    if generator_matrix.ndim != 2:
        raise ValueError("generator_matrix_s_inv must be two-dimensional")
    if generator_matrix.shape[0] != generator_matrix.shape[1]:
        raise ValueError("generator_matrix_s_inv must be square")
    if generator_matrix.shape[0] == 0:
        raise ValueError("generator_matrix_s_inv must contain at least one state")
    if not np.all(np.isfinite(generator_matrix)):
        raise ValueError("generator_matrix_s_inv contains non-finite values")
    return generator_matrix


def _validated_state_concentrations(
    state_concentrations_mol_m3: np.ndarray,
    state_count: int,
) -> np.ndarray:
    state_concentrations = np.asarray(state_concentrations_mol_m3, dtype=float)
    if state_concentrations.ndim != 1:
        raise ValueError("state_concentrations_mol_m3 must be one-dimensional")
    if state_concentrations.shape[0] != state_count:
        raise ValueError("state_concentrations_mol_m3 length must equal state count")
    if not np.all(np.isfinite(state_concentrations)):
        raise ValueError("state_concentrations_mol_m3 contains non-finite values")
    if np.any(state_concentrations <= ZERO_VALUE):
        raise ValueError("state_concentrations_mol_m3 must be strictly positive")
    return state_concentrations


def _validate_event_indices(
    event: MarkovAdditiveEvent,
    state_count: int,
) -> None:
    if not isinstance(event.from_state_index, int):
        raise TypeError(f"{event.label}.from_state_index must be an integer")
    if not isinstance(event.to_state_index, int):
        raise TypeError(f"{event.label}.to_state_index must be an integer")
    if event.from_state_index < 0 or event.from_state_index >= state_count:
        raise ValueError(f"{event.label}.from_state_index is outside the state range")
    if event.to_state_index < 0 or event.to_state_index >= state_count:
        raise ValueError(f"{event.label}.to_state_index is outside the state range")


def _validated_displacement(
    charge_displacement_m: tuple[float, float, float],
    event_label: str,
) -> np.ndarray:
    displacement_array = np.asarray(charge_displacement_m, dtype=float)
    if displacement_array.shape != (int(CARTESIAN_AXIS_COUNT),):
        raise ValueError(f"{event_label}.charge_displacement_m must have length three")
    if not np.all(np.isfinite(displacement_array)):
        raise ValueError(f"{event_label}.charge_displacement_m contains non-finite values")
    return displacement_array


def _is_zero_displacement(displacement_array: np.ndarray) -> bool:
    return bool(np.all(displacement_array == ZERO_VALUE))


def _displacement_key(displacement_array: np.ndarray) -> tuple[float, float, float]:
    return tuple(
        _canonical_float_for_key(float(component))
        for component in displacement_array
    )


def _canonical_float_for_key(value: float) -> float:
    if value == ZERO_VALUE:
        return ZERO_VALUE
    return float(value)


def _positive_float(value: float, context: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= ZERO_VALUE:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value


def _symmetrized_matrix(matrix: np.ndarray) -> np.ndarray:
    return HALF_JUMP_VARIANCE_FACTOR * (matrix + matrix.T)


def _matrix_tolerance(
    generator_matrix: np.ndarray,
    state_concentrations: np.ndarray,
) -> float:
    return FINITE_MARKOV_ADDITIVE_TOLERANCE * max(
        1.0,
        float(np.max(np.abs(generator_matrix))),
        float(np.max(np.abs(state_concentrations))),
        float(np.sum(state_concentrations)),
    )
