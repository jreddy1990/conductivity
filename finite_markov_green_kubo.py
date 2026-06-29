"""Exact Green-Kubo and Galerkin readouts for reversible finite Markov models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from constants import S_M_TO_MS_CM
from conductivity.finite_mori_conductivity import (
    CARTESIAN_AXIS_COUNT,
    MORI_NUMERICAL_TOLERANCE,
    ProjectedMoriConductivityInput,
    ProjectedMoriConductivityResult,
    compute_projected_mori_conductivity,
)


FINITE_MARKOV_GREEN_KUBO_TOLERANCE = math.sqrt(np.finfo(float).eps)
HALF_JUMP_VARIANCE_FACTOR = 0.5  # Analytical one-half in CTMC jump quadratic variation.
NORMALIZED_PROBABILITY_SUM = 1.0  # Analytical normalization for stationary probabilities.
ZERO_VALUE = 0.0  # Named numerical zero used in validation and PSD roundoff checks.


@dataclass(frozen=True)
class ReversibleGeneratorValidation:
    row_sum_residual: float
    stationary_residual: float
    detailed_balance_residual: float
    minimum_offdiagonal_rate_s_inv: float
    stationary_probability_sum: float


@dataclass(frozen=True)
class StateCurrentGreenKuboResult:
    mori_input: ProjectedMoriConductivityInput
    mori_result: ProjectedMoriConductivityResult
    validation: ReversibleGeneratorValidation
    centered_current_by_state: np.ndarray
    symmetrized_energy_matrix: np.ndarray


@dataclass(frozen=True)
class GalerkinProjectedConductivityResult:
    mori_input: ProjectedMoriConductivityInput
    mori_result: ProjectedMoriConductivityResult
    validation: ReversibleGeneratorValidation
    retained_basis_matrix: np.ndarray
    retained_gram_eigenvalues: tuple[float, ...]
    all_gram_eigenvalues: tuple[float, ...]
    energy_matrix: np.ndarray


@dataclass(frozen=True)
class NestedGalerkinComparison:
    projected_results: tuple[GalerkinProjectedConductivityResult, ...]
    exact_result: StateCurrentGreenKuboResult
    sigma_mS_cm_by_basis: tuple[float, ...]
    monotone_non_decreasing: bool
    exact_closure_gap_mS_cm: float


@dataclass(frozen=True)
class MarkovAdditiveJumpEdge:
    source_index: int
    target_index: int
    rate_s_inv: float
    displacement_by_axis: tuple[float, float, float]


@dataclass(frozen=True)
class MarkovAdditiveJumpConductivityResult:
    validation: ReversibleGeneratorValidation
    drift_current_by_state: np.ndarray
    corrector_mori_input: ProjectedMoriConductivityInput
    corrector_mori_result: ProjectedMoriConductivityResult
    direct_axis_diffusivity: tuple[float, float, float]
    corrector_axis_diffusivity: tuple[float, float, float]
    effective_axis_diffusivity: tuple[float, float, float]
    sigma_S_m: float
    sigma_mS_cm: float


def validate_reversible_generator(
    generator_matrix_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
) -> ReversibleGeneratorValidation:
    """Validate a finite reversible CTMC generator."""

    generator_matrix = _validated_generator_matrix(generator_matrix_s_inv)
    stationary_distribution = _validated_stationary_probabilities(
        stationary_probabilities,
        generator_matrix.shape[0],
    )
    tolerance = _matrix_tolerance(generator_matrix)

    row_sum_residual = float(np.max(np.abs(np.sum(generator_matrix, axis=1))))
    offdiagonal_rates = generator_matrix[~np.eye(generator_matrix.shape[0], dtype=bool)]
    minimum_offdiagonal_rate = float(np.min(offdiagonal_rates)) if offdiagonal_rates.size else ZERO_VALUE
    maximum_diagonal_entry = float(np.max(np.diag(generator_matrix)))
    stationary_residual = float(
        np.max(np.abs(stationary_distribution @ generator_matrix))
    )
    detailed_balance_matrix = (
        stationary_distribution[:, None] * generator_matrix
        - stationary_distribution[None, :] * generator_matrix.T
    )
    detailed_balance_residual = float(np.max(np.abs(detailed_balance_matrix)))
    stationary_probability_sum = float(np.sum(stationary_distribution))

    if row_sum_residual > tolerance:
        raise ValueError(f"generator row-sum residual {row_sum_residual} exceeds {tolerance}")
    if minimum_offdiagonal_rate < -tolerance:
        raise ValueError("generator off-diagonal entries must be nonnegative")
    if maximum_diagonal_entry > tolerance:
        raise ValueError("generator diagonal entries must be nonpositive")
    if abs(stationary_probability_sum - NORMALIZED_PROBABILITY_SUM) > tolerance:
        raise ValueError("stationary probabilities must sum to one")
    if stationary_residual > tolerance:
        raise ValueError(
            f"stationary distribution residual {stationary_residual} exceeds {tolerance}"
        )
    if detailed_balance_residual > tolerance:
        raise ValueError(
            f"detailed-balance residual {detailed_balance_residual} exceeds {tolerance}"
        )

    return ReversibleGeneratorValidation(
        row_sum_residual=row_sum_residual,
        stationary_residual=stationary_residual,
        detailed_balance_residual=detailed_balance_residual,
        minimum_offdiagonal_rate_s_inv=minimum_offdiagonal_rate,
        stationary_probability_sum=stationary_probability_sum,
    )


def finite_markov_to_projected_mori_input(
    generator_matrix_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
    current_by_state: np.ndarray,
    beta_over_volume: float,
) -> ProjectedMoriConductivityInput:
    """Build the projected Mori input for exact finite-chain state-current GK."""

    construction = compute_finite_markov_green_kubo_conductivity(
        generator_matrix_s_inv,
        stationary_probabilities,
        current_by_state,
        beta_over_volume,
    )
    return construction.mori_input


def compute_finite_markov_green_kubo_conductivity(
    generator_matrix_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
    current_by_state: np.ndarray,
    beta_over_volume: float,
) -> StateCurrentGreenKuboResult:
    """Evaluate exact finite-chain GK conductivity for a state current."""

    generator_matrix = _validated_generator_matrix(generator_matrix_s_inv)
    stationary_distribution = _validated_stationary_probabilities(
        stationary_probabilities,
        generator_matrix.shape[0],
    )
    validation = validate_reversible_generator(generator_matrix, stationary_distribution)
    centered_current_by_state = _centered_current_by_state(
        current_by_state,
        stationary_distribution,
    )
    symmetrized_energy_matrix = _symmetrized_energy_matrix(
        generator_matrix,
        stationary_distribution,
    )
    current_coupling_matrix = (
        np.sqrt(stationary_distribution)[:, None] * centered_current_by_state
    ).T
    mori_input = ProjectedMoriConductivityInput(
        direct_energy_matrix=np.zeros_like(symmetrized_energy_matrix),
        memory_self_energy_matrix=symmetrized_energy_matrix,
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=_positive_float(beta_over_volume, "beta_over_volume"),
    )
    mori_result = compute_projected_mori_conductivity(mori_input)
    return StateCurrentGreenKuboResult(
        mori_input=mori_input,
        mori_result=mori_result,
        validation=validation,
        centered_current_by_state=centered_current_by_state,
        symmetrized_energy_matrix=symmetrized_energy_matrix,
    )


def compute_galerkin_projected_conductivity(
    generator_matrix_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
    current_by_state: np.ndarray,
    basis_matrix: np.ndarray,
    beta_over_volume: float,
    gram_relative_tolerance: float,
) -> GalerkinProjectedConductivityResult:
    """Evaluate the finite Galerkin projection of the state-current GK problem."""

    generator_matrix = _validated_generator_matrix(generator_matrix_s_inv)
    stationary_distribution = _validated_stationary_probabilities(
        stationary_probabilities,
        generator_matrix.shape[0],
    )
    validation = validate_reversible_generator(generator_matrix, stationary_distribution)
    centered_current_by_state = _centered_current_by_state(
        current_by_state,
        stationary_distribution,
    )
    validated_basis_matrix = _validated_basis_matrix(
        basis_matrix,
        generator_matrix.shape[0],
    )
    retained_basis_matrix, retained_gram_eigenvalues, all_gram_eigenvalues = (
        _pi_orthonormal_basis(
            validated_basis_matrix,
            stationary_distribution,
            gram_relative_tolerance,
        )
    )
    generator_energy_action = -generator_matrix @ retained_basis_matrix
    energy_matrix = _symmetrized_matrix(
        retained_basis_matrix.T
        @ (stationary_distribution[:, None] * generator_energy_action)
    )
    current_coupling_matrix = (
        retained_basis_matrix.T
        @ (stationary_distribution[:, None] * centered_current_by_state)
    ).T
    mori_input = ProjectedMoriConductivityInput(
        direct_energy_matrix=np.zeros_like(energy_matrix),
        memory_self_energy_matrix=energy_matrix,
        current_coupling_matrix=current_coupling_matrix,
        beta_over_volume=_positive_float(beta_over_volume, "beta_over_volume"),
    )
    mori_result = compute_projected_mori_conductivity(mori_input)
    return GalerkinProjectedConductivityResult(
        mori_input=mori_input,
        mori_result=mori_result,
        validation=validation,
        retained_basis_matrix=retained_basis_matrix,
        retained_gram_eigenvalues=tuple(float(value) for value in retained_gram_eigenvalues),
        all_gram_eigenvalues=tuple(float(value) for value in all_gram_eigenvalues),
        energy_matrix=energy_matrix,
    )


def compare_nested_galerkin_bounds(
    generator_matrix_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
    current_by_state: np.ndarray,
    basis_matrices: Sequence[np.ndarray],
    beta_over_volume: float,
    gram_relative_tolerance: float,
) -> NestedGalerkinComparison:
    """Compare a nested Galerkin sequence against exact finite-chain closure."""

    if len(basis_matrices) == 0:
        raise ValueError("basis_matrices must contain at least one basis")
    projected_results = tuple(
        compute_galerkin_projected_conductivity(
            generator_matrix_s_inv,
            stationary_probabilities,
            current_by_state,
            basis_matrix,
            beta_over_volume,
            gram_relative_tolerance,
        )
        for basis_matrix in basis_matrices
    )
    exact_result = compute_finite_markov_green_kubo_conductivity(
        generator_matrix_s_inv,
        stationary_probabilities,
        current_by_state,
        beta_over_volume,
    )
    sigma_mS_cm_by_basis = tuple(
        projected_result.mori_result.sigma_mS_cm
        for projected_result in projected_results
    )
    monotone_non_decreasing = _is_monotone_non_decreasing(
        sigma_mS_cm_by_basis,
        FINITE_MARKOV_GREEN_KUBO_TOLERANCE,
    )
    exact_closure_gap_mS_cm = abs(
        sigma_mS_cm_by_basis[-1] - exact_result.mori_result.sigma_mS_cm
    )
    return NestedGalerkinComparison(
        projected_results=projected_results,
        exact_result=exact_result,
        sigma_mS_cm_by_basis=sigma_mS_cm_by_basis,
        monotone_non_decreasing=monotone_non_decreasing,
        exact_closure_gap_mS_cm=exact_closure_gap_mS_cm,
    )


def compute_markov_additive_jump_conductivity(
    generator_matrix_s_inv: np.ndarray,
    stationary_probabilities: np.ndarray,
    jump_edges: Sequence[MarkovAdditiveJumpEdge],
    beta_over_volume: float,
) -> MarkovAdditiveJumpConductivityResult:
    """Evaluate reversible Markov-additive jump conductivity."""

    generator_matrix = _validated_generator_matrix(generator_matrix_s_inv)
    stationary_distribution = _validated_stationary_probabilities(
        stationary_probabilities,
        generator_matrix.shape[0],
    )
    validation = validate_reversible_generator(generator_matrix, stationary_distribution)
    _validate_jump_edges(jump_edges, generator_matrix)

    state_count = generator_matrix.shape[0]
    axis_count = int(CARTESIAN_AXIS_COUNT)
    direct_axis_diffusivity_array = np.zeros(axis_count, dtype=float)
    drift_current_by_state = np.zeros((state_count, axis_count), dtype=float)

    for jump_edge in jump_edges:
        displacement_array = np.asarray(jump_edge.displacement_by_axis, dtype=float)
        direct_axis_diffusivity_array += (
            HALF_JUMP_VARIANCE_FACTOR
            * stationary_distribution[jump_edge.source_index]
            * jump_edge.rate_s_inv
            * displacement_array
            * displacement_array
        )
        drift_current_by_state[jump_edge.source_index, :] += (
            jump_edge.rate_s_inv * displacement_array
        )

    stationary_drift_by_axis = stationary_distribution @ drift_current_by_state
    drift_tolerance = _matrix_tolerance(generator_matrix)
    if float(np.max(np.abs(stationary_drift_by_axis))) > drift_tolerance:
        raise ValueError("Markov-additive jump process has nonzero stationary drift")

    corrector_result = compute_finite_markov_green_kubo_conductivity(
        generator_matrix,
        stationary_distribution,
        drift_current_by_state,
        beta_over_volume,
    )
    corrector_axis_diffusivity_array = np.asarray(
        corrector_result.mori_result.quadratic_form_by_axis,
        dtype=float,
    )
    effective_axis_diffusivity_array = (
        direct_axis_diffusivity_array - corrector_axis_diffusivity_array
    )
    diffusivity_tolerance = FINITE_MARKOV_GREEN_KUBO_TOLERANCE * max(
        NORMALIZED_PROBABILITY_SUM,
        float(np.max(np.abs(direct_axis_diffusivity_array))),
    )
    minimum_effective_diffusivity = float(np.min(effective_axis_diffusivity_array))
    if minimum_effective_diffusivity < -diffusivity_tolerance:
        raise ValueError("Markov-additive effective diffusivity became negative")
    effective_axis_diffusivity_array = np.asarray(
        [
            ZERO_VALUE if abs(value) <= diffusivity_tolerance else float(value)
            for value in effective_axis_diffusivity_array
        ],
        dtype=float,
    )
    sigma_S_m = (
        _positive_float(beta_over_volume, "beta_over_volume")
        * float(np.sum(effective_axis_diffusivity_array))
        / CARTESIAN_AXIS_COUNT
    )
    if sigma_S_m < -diffusivity_tolerance:
        raise ValueError("Markov-additive conductivity became negative")
    if abs(sigma_S_m) <= diffusivity_tolerance:
        sigma_S_m = ZERO_VALUE
    return MarkovAdditiveJumpConductivityResult(
        validation=validation,
        drift_current_by_state=drift_current_by_state,
        corrector_mori_input=corrector_result.mori_input,
        corrector_mori_result=corrector_result.mori_result,
        direct_axis_diffusivity=tuple(float(value) for value in direct_axis_diffusivity_array),
        corrector_axis_diffusivity=tuple(float(value) for value in corrector_axis_diffusivity_array),
        effective_axis_diffusivity=tuple(float(value) for value in effective_axis_diffusivity_array),
        sigma_S_m=float(sigma_S_m),
        sigma_mS_cm=float(sigma_S_m * S_M_TO_MS_CM),
    )


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


def _validated_stationary_probabilities(
    stationary_probabilities: np.ndarray,
    state_count: int,
) -> np.ndarray:
    stationary_distribution = np.asarray(stationary_probabilities, dtype=float)
    if stationary_distribution.ndim != 1:
        raise ValueError("stationary_probabilities must be one-dimensional")
    if stationary_distribution.shape[0] != state_count:
        raise ValueError("stationary_probabilities length must equal state count")
    if not np.all(np.isfinite(stationary_distribution)):
        raise ValueError("stationary_probabilities contains non-finite values")
    if np.any(stationary_distribution <= ZERO_VALUE):
        raise ValueError("stationary_probabilities must be strictly positive")
    return stationary_distribution


def _centered_current_by_state(
    current_by_state: np.ndarray,
    stationary_distribution: np.ndarray,
) -> np.ndarray:
    current_array = np.asarray(current_by_state, dtype=float)
    expected_shape = (stationary_distribution.shape[0], int(CARTESIAN_AXIS_COUNT))
    if current_array.shape != expected_shape:
        raise ValueError(f"current_by_state must have shape {expected_shape}")
    if not np.all(np.isfinite(current_array)):
        raise ValueError("current_by_state contains non-finite values")
    stationary_current_mean = stationary_distribution @ current_array
    return current_array - stationary_current_mean[None, :]


def _symmetrized_energy_matrix(
    generator_matrix: np.ndarray,
    stationary_distribution: np.ndarray,
) -> np.ndarray:
    sqrt_stationary_distribution = np.sqrt(stationary_distribution)
    inverse_sqrt_stationary_distribution = NORMALIZED_PROBABILITY_SUM / sqrt_stationary_distribution
    energy_matrix = (
        sqrt_stationary_distribution[:, None]
        * (-generator_matrix)
        * inverse_sqrt_stationary_distribution[None, :]
    )
    return _symmetrized_matrix(energy_matrix)


def _validated_basis_matrix(
    basis_matrix: np.ndarray,
    state_count: int,
) -> np.ndarray:
    basis_array = np.asarray(basis_matrix, dtype=float)
    if basis_array.ndim != 2:
        raise ValueError("basis_matrix must be two-dimensional")
    if basis_array.shape[0] != state_count:
        raise ValueError("basis_matrix row count must equal state count")
    if basis_array.shape[1] == 0:
        raise ValueError("basis_matrix must contain at least one basis function")
    if not np.all(np.isfinite(basis_array)):
        raise ValueError("basis_matrix contains non-finite values")
    return basis_array


def _pi_orthonormal_basis(
    basis_matrix: np.ndarray,
    stationary_distribution: np.ndarray,
    gram_relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _positive_float(gram_relative_tolerance, "gram_relative_tolerance")
    gram_matrix = _symmetrized_matrix(
        basis_matrix.T @ (stationary_distribution[:, None] * basis_matrix)
    )
    gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(gram_matrix)
    eigenvalue_scale = max(
        float(np.max(np.abs(gram_eigenvalues))),
        np.finfo(float).tiny,
    )
    allowed_negative_eigenvalue = gram_relative_tolerance * eigenvalue_scale
    minimum_gram_eigenvalue = float(np.min(gram_eigenvalues))
    if minimum_gram_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError("basis Gram matrix has significant negative eigenvalues")
    retained_mask = gram_eigenvalues > gram_relative_tolerance * eigenvalue_scale
    if not np.any(retained_mask):
        raise ValueError("basis_matrix has no retained Gram modes")
    retained_eigenvalues = gram_eigenvalues[retained_mask]
    retained_eigenvectors = gram_eigenvectors[:, retained_mask]
    whitening_matrix = retained_eigenvectors @ np.diag(
        NORMALIZED_PROBABILITY_SUM / np.sqrt(retained_eigenvalues)
    )
    return basis_matrix @ whitening_matrix, retained_eigenvalues, gram_eigenvalues


def _validate_jump_edges(
    jump_edges: Sequence[MarkovAdditiveJumpEdge],
    generator_matrix: np.ndarray,
) -> None:
    if len(jump_edges) == 0:
        raise ValueError("jump_edges must contain at least one edge")
    state_count = generator_matrix.shape[0]
    offdiagonal_rate_by_pair = np.zeros((state_count, state_count), dtype=float)
    for jump_edge in jump_edges:
        if not isinstance(jump_edge.source_index, int):
            raise TypeError("jump edge source_index must be an integer")
        if not isinstance(jump_edge.target_index, int):
            raise TypeError("jump edge target_index must be an integer")
        if jump_edge.source_index < 0 or jump_edge.source_index >= state_count:
            raise ValueError("jump edge source_index is outside the state range")
        if jump_edge.target_index < 0 or jump_edge.target_index >= state_count:
            raise ValueError("jump edge target_index is outside the state range")
        _positive_float(jump_edge.rate_s_inv, "jump edge rate_s_inv")
        displacement_array = np.asarray(jump_edge.displacement_by_axis, dtype=float)
        if displacement_array.shape != (int(CARTESIAN_AXIS_COUNT),):
            raise ValueError("jump edge displacement_by_axis must have length three")
        if not np.all(np.isfinite(displacement_array)):
            raise ValueError("jump edge displacement_by_axis contains non-finite values")
        if jump_edge.source_index != jump_edge.target_index:
            offdiagonal_rate_by_pair[
                jump_edge.source_index,
                jump_edge.target_index,
            ] += jump_edge.rate_s_inv

    tolerance = _matrix_tolerance(generator_matrix)
    excess_rate_matrix = offdiagonal_rate_by_pair - np.maximum(generator_matrix, ZERO_VALUE)
    maximum_excess_rate = float(np.max(excess_rate_matrix))
    if maximum_excess_rate > tolerance:
        raise ValueError("off-diagonal jump edge rates exceed generator transition rates")


def _is_monotone_non_decreasing(
    values: Sequence[float],
    tolerance: float,
) -> bool:
    for previous_value, next_value in zip(values[:-1], values[1:]):
        if next_value + tolerance < previous_value:
            return False
    return True


def _symmetrized_matrix(matrix: np.ndarray) -> np.ndarray:
    return HALF_JUMP_VARIANCE_FACTOR * (matrix + matrix.T)


def _matrix_tolerance(matrix: np.ndarray) -> float:
    return FINITE_MARKOV_GREEN_KUBO_TOLERANCE * max(
        NORMALIZED_PROBABILITY_SUM,
        float(np.max(np.abs(matrix))),
    )


def _positive_float(
    value: float,
    context: str,
) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= ZERO_VALUE:
        raise ValueError(f"{context} must be positive and finite")
    return parsed_value
