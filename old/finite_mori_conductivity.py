"""Finite Mori/Galerkin conductivity calculator for supplied current-memory blocks.

This module only evaluates the zero-frequency quadratic form. It does not infer
true trajectory-projected Mori objects from a recipe. Production finite Markov
conductivity supplies latent, property-calibrated finite-process blocks; a
trajectory oracle audit may instead supply blocks estimated from J(t) and B(t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from constants import S_M_TO_MS_CM


CARTESIAN_AXIS_COUNT = 3.0  # Analytical isotropic average over x, y, z current axes.
MORI_NUMERICAL_TOLERANCE = math.sqrt(np.finfo(float).eps)
TRAPEZOID_ENDPOINT_WEIGHT = 0.5  # Analytical trapezoid-rule endpoint weight.


@dataclass(frozen=True)
class ProjectedMoriConductivityInput:
    direct_energy_matrix: np.ndarray
    memory_self_energy_matrix: np.ndarray
    current_coupling_matrix: np.ndarray
    beta_over_volume: float


@dataclass(frozen=True)
class ProjectedMoriConductivityResult:
    sigma_S_m: float
    sigma_mS_cm: float
    axis_conductivity_S_m: tuple[float, float, float]
    quadratic_form_by_axis: tuple[float, float, float]
    energy_eigenvalues: tuple[float, ...]
    effective_energy_matrix: np.ndarray


@dataclass(frozen=True)
class MoriOracleClosureComparison:
    sigma_oracle_S_m: float
    sigma_oracle_mS_cm: float
    sigma_trajectory_mS_cm: float
    closure_gap_mS_cm: float
    tolerance_mS_cm: float
    passes_tolerance: bool


@dataclass(frozen=True)
class TrajectoryProjectedMoriConstruction:
    mori_input: ProjectedMoriConductivityInput
    centered_current_time_series: np.ndarray
    whitened_basis_time_series: np.ndarray
    retained_basis_time_series: np.ndarray
    projected_current_time_series: np.ndarray
    raw_green_kubo_sigma_mS_cm: float
    projected_green_kubo_sigma_mS_cm: float
    raw_green_kubo_axis_integrals: tuple[float, float, float]
    projected_green_kubo_axis_integrals: tuple[float, float, float]
    maximum_lag_steps_used: int
    retained_gram_eigenvalues: tuple[float, ...]
    retained_zero_frequency_covariance_eigenvalues: tuple[float, ...]
    all_zero_frequency_covariance_eigenvalues: tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryMoriClosureAuditInput:
    sample_id: str
    basis_feature_time_series: np.ndarray
    current_time_series: np.ndarray
    time_step_s: float
    maximum_lag_steps: int
    beta_over_volume: float
    sigma_property_db_mS_cm: float
    sigma_recipe_mori_mS_cm: float
    gram_relative_tolerance: float = 1.0e-10
    zero_frequency_relative_tolerance: float = 1.0e-8


@dataclass(frozen=True)
class TrajectoryMoriClosureAuditRow:
    sample_id: str
    sigma_property_db_mS_cm: float
    sigma_raw_gk_mS_cm: float
    sigma_projected_gk_mS_cm: float
    sigma_mori_oracle_mS_cm: float
    sigma_recipe_mori_mS_cm: float
    projection_gap_mS_cm: float
    recipe_gap_mS_cm: float
    label_gap_mS_cm: float
    gram_rank: int
    kz_rank: int
    min_kz_eigenvalue: float
    dropped_gram_modes: int
    dropped_kz_modes: int


def compute_projected_mori_conductivity(
    mori_input: ProjectedMoriConductivityInput,
) -> ProjectedMoriConductivityResult:
    """Evaluate beta/(3V) sum_axis h_axis^T A_M^# h_axis."""

    direct_energy_matrix = _validated_square_matrix(
        mori_input.direct_energy_matrix,
        "direct_energy_matrix",
    )
    memory_self_energy_matrix = _validated_square_matrix(
        mori_input.memory_self_energy_matrix,
        "memory_self_energy_matrix",
    )
    if direct_energy_matrix.shape != memory_self_energy_matrix.shape:
        raise ValueError(
            "direct_energy_matrix and memory_self_energy_matrix must have the same shape"
        )
    _validate_symmetric_matrix(direct_energy_matrix, "direct_energy_matrix")
    _validate_symmetric_matrix(memory_self_energy_matrix, "memory_self_energy_matrix")
    _validate_positive_semidefinite_matrix(direct_energy_matrix, "direct_energy_matrix")
    _validate_positive_semidefinite_matrix(memory_self_energy_matrix, "memory_self_energy_matrix")

    current_coupling_matrix = _validated_current_coupling_matrix(
        mori_input.current_coupling_matrix,
        direct_energy_matrix.shape[0],
    )
    _assert_positive_finite(mori_input.beta_over_volume, "beta_over_volume")

    effective_energy_matrix = direct_energy_matrix + memory_self_energy_matrix
    _validate_symmetric_matrix(effective_energy_matrix, "effective_energy_matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(effective_energy_matrix)
    _validate_eigenvalues_are_psd(eigenvalues, "effective_energy_matrix")

    quadratic_forms: list[float] = []
    axis_conductivities: list[float] = []
    for axis_index in range(current_coupling_matrix.shape[0]):
        axis_current_coupling = current_coupling_matrix[axis_index, :]
        quadratic_form = _projected_quadratic_form(
            eigenvalues,
            eigenvectors,
            axis_current_coupling,
            f"axis_{axis_index}",
        )
        quadratic_forms.append(quadratic_form)
        axis_conductivities.append(
            mori_input.beta_over_volume * quadratic_form / CARTESIAN_AXIS_COUNT
        )

    sigma_S_m = math.fsum(axis_conductivities)
    _assert_nonnegative_finite(sigma_S_m, "sigma_S_m")
    return ProjectedMoriConductivityResult(
        sigma_S_m=sigma_S_m,
        sigma_mS_cm=sigma_S_m * S_M_TO_MS_CM,
        axis_conductivity_S_m=(
            axis_conductivities[0],
            axis_conductivities[1],
            axis_conductivities[2],
        ),
        quadratic_form_by_axis=(
            quadratic_forms[0],
            quadratic_forms[1],
            quadratic_forms[2],
        ),
        energy_eigenvalues=tuple(float(eigenvalue) for eigenvalue in eigenvalues),
        effective_energy_matrix=effective_energy_matrix.copy(),
    )


def compare_mori_oracle_to_trajectory(
    mori_result: ProjectedMoriConductivityResult,
    trajectory_sigma_mS_cm: float,
    tolerance_mS_cm: float,
) -> MoriOracleClosureComparison:
    """Compare a projected Mori oracle result to trajectory Green-Kubo conductivity."""

    _assert_nonnegative_finite(mori_result.sigma_mS_cm, "mori_result.sigma_mS_cm")
    _assert_nonnegative_finite(trajectory_sigma_mS_cm, "trajectory_sigma_mS_cm")
    _assert_positive_finite(tolerance_mS_cm, "tolerance_mS_cm")
    closure_gap_mS_cm = float(abs(mori_result.sigma_mS_cm - trajectory_sigma_mS_cm))
    return MoriOracleClosureComparison(
        sigma_oracle_S_m=mori_result.sigma_S_m,
        sigma_oracle_mS_cm=mori_result.sigma_mS_cm,
        sigma_trajectory_mS_cm=trajectory_sigma_mS_cm,
        closure_gap_mS_cm=closure_gap_mS_cm,
        tolerance_mS_cm=tolerance_mS_cm,
        passes_tolerance=bool(closure_gap_mS_cm < tolerance_mS_cm),
    )


def build_trajectory_projected_mori_input(
    basis_feature_time_series: np.ndarray,
    current_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
    beta_over_volume: float,
    gram_relative_tolerance: float = 1.0e-10,
    zero_frequency_relative_tolerance: float = 1.0e-8,
) -> TrajectoryProjectedMoriConstruction:
    """Build projected Mori matrices from one centered trajectory current process."""

    feature_array = _validated_feature_time_series(basis_feature_time_series)
    current_array = _validated_current_time_series(current_time_series, "current_time_series")
    if feature_array.shape[0] != current_array.shape[0]:
        raise ValueError(
            "basis_feature_time_series and current_time_series must have the same frame count"
        )
    _assert_positive_finite(time_step_s, "time_step_s")
    _assert_positive_finite(beta_over_volume, "beta_over_volume")
    _assert_positive_finite(gram_relative_tolerance, "gram_relative_tolerance")
    _assert_positive_finite(
        zero_frequency_relative_tolerance,
        "zero_frequency_relative_tolerance",
    )
    retained_maximum_lag_steps = _validated_maximum_lag_steps(
        maximum_lag_steps,
        current_array.shape[0],
    )

    centered_features = feature_array - np.mean(feature_array, axis=0, keepdims=True)
    centered_current = current_array - np.mean(current_array, axis=0, keepdims=True)

    frame_count = centered_features.shape[0]
    gram_matrix = _symmetrized_matrix(centered_features.T @ centered_features / frame_count)
    retained_gram_eigenvalues, retained_gram_eigenvectors, _ = _eigh_retained_psd(
        gram_matrix,
        gram_relative_tolerance,
        "trajectory feature Gram matrix",
    )
    if retained_gram_eigenvalues.size == 0:
        raise ValueError("all trajectory basis features are null under the Gram matrix")

    inverse_sqrt_gram_eigenvalues = np.diag(1.0 / np.sqrt(retained_gram_eigenvalues))
    whitening_matrix = retained_gram_eigenvectors @ inverse_sqrt_gram_eigenvalues
    whitened_basis_time_series = centered_features @ whitening_matrix

    current_coupling_matrix = centered_current.T @ whitened_basis_time_series / frame_count
    integrated_basis_covariance = _integrated_symmetrized_basis_covariance(
        whitened_basis_time_series,
        time_step_s,
        retained_maximum_lag_steps,
    )
    (
        retained_zero_frequency_covariance_eigenvalues,
        retained_zero_frequency_covariance_eigenvectors,
        all_zero_frequency_covariance_eigenvalues,
    ) = _eigh_retained_psd(
        integrated_basis_covariance,
        zero_frequency_relative_tolerance,
        "integrated projected covariance",
    )
    if retained_zero_frequency_covariance_eigenvalues.size == 0:
        raise ValueError(
            "projected basis has no positive zero-frequency covariance modes"
        )

    memory_self_energy_matrix = np.diag(
        1.0 / retained_zero_frequency_covariance_eigenvalues
    )
    retained_current_coupling_matrix = (
        current_coupling_matrix @ retained_zero_frequency_covariance_eigenvectors
    )
    direct_energy_matrix = np.zeros_like(memory_self_energy_matrix)

    retained_basis_time_series = (
        whitened_basis_time_series @ retained_zero_frequency_covariance_eigenvectors
    )
    projected_current_time_series = retained_basis_time_series @ retained_current_coupling_matrix.T
    raw_axis_integrals = green_kubo_axis_integrals_from_current_time_series(
        centered_current,
        time_step_s,
        retained_maximum_lag_steps,
        center_current=False,
    )
    projected_axis_integrals = green_kubo_axis_integrals_from_current_time_series(
        projected_current_time_series,
        time_step_s,
        retained_maximum_lag_steps,
        center_current=True,
    )
    raw_green_kubo_sigma_mS_cm = _green_kubo_sigma_mS_cm_from_axis_integrals(
        raw_axis_integrals,
        beta_over_volume,
    )
    projected_green_kubo_sigma_mS_cm = _green_kubo_sigma_mS_cm_from_axis_integrals(
        projected_axis_integrals,
        beta_over_volume,
    )

    return TrajectoryProjectedMoriConstruction(
        mori_input=ProjectedMoriConductivityInput(
            direct_energy_matrix=direct_energy_matrix,
            memory_self_energy_matrix=memory_self_energy_matrix,
            current_coupling_matrix=retained_current_coupling_matrix,
            beta_over_volume=beta_over_volume,
        ),
        centered_current_time_series=centered_current.copy(),
        whitened_basis_time_series=whitened_basis_time_series.copy(),
        retained_basis_time_series=retained_basis_time_series.copy(),
        projected_current_time_series=projected_current_time_series.copy(),
        raw_green_kubo_sigma_mS_cm=raw_green_kubo_sigma_mS_cm,
        projected_green_kubo_sigma_mS_cm=projected_green_kubo_sigma_mS_cm,
        raw_green_kubo_axis_integrals=raw_axis_integrals,
        projected_green_kubo_axis_integrals=projected_axis_integrals,
        maximum_lag_steps_used=retained_maximum_lag_steps,
        retained_gram_eigenvalues=tuple(
            float(eigenvalue) for eigenvalue in retained_gram_eigenvalues
        ),
        retained_zero_frequency_covariance_eigenvalues=tuple(
            float(eigenvalue)
            for eigenvalue in retained_zero_frequency_covariance_eigenvalues
        ),
        all_zero_frequency_covariance_eigenvalues=tuple(
            float(eigenvalue) for eigenvalue in all_zero_frequency_covariance_eigenvalues
        ),
    )


def build_trajectory_mori_closure_audit_row(
    closure_input: TrajectoryMoriClosureAuditInput,
) -> TrajectoryMoriClosureAuditRow:
    """Build the trajectory/property/recipe Mori closure decomposition row."""

    validated_sample_id = _validated_sample_id(closure_input.sample_id)
    feature_array = _validated_feature_time_series(
        closure_input.basis_feature_time_series
    )
    _assert_nonnegative_finite(
        closure_input.sigma_property_db_mS_cm,
        "sigma_property_db_mS_cm",
    )
    _assert_nonnegative_finite(
        closure_input.sigma_recipe_mori_mS_cm,
        "sigma_recipe_mori_mS_cm",
    )

    construction = build_trajectory_projected_mori_input(
        basis_feature_time_series=feature_array,
        current_time_series=closure_input.current_time_series,
        time_step_s=closure_input.time_step_s,
        maximum_lag_steps=closure_input.maximum_lag_steps,
        beta_over_volume=closure_input.beta_over_volume,
        gram_relative_tolerance=closure_input.gram_relative_tolerance,
        zero_frequency_relative_tolerance=(
            closure_input.zero_frequency_relative_tolerance
        ),
    )
    mori_oracle_result = compute_projected_mori_conductivity(
        construction.mori_input
    )

    raw_feature_count = int(feature_array.shape[1])
    gram_rank = len(construction.retained_gram_eigenvalues)
    kz_rank = len(construction.retained_zero_frequency_covariance_eigenvalues)
    min_kz_eigenvalue = float(
        min(construction.all_zero_frequency_covariance_eigenvalues)
    )
    dropped_gram_modes = raw_feature_count - gram_rank
    dropped_kz_modes = gram_rank - kz_rank
    if dropped_gram_modes < 0:
        raise ValueError("dropped_gram_modes cannot be negative")
    if dropped_kz_modes < 0:
        raise ValueError("dropped_kz_modes cannot be negative")

    return TrajectoryMoriClosureAuditRow(
        sample_id=validated_sample_id,
        sigma_property_db_mS_cm=float(closure_input.sigma_property_db_mS_cm),
        sigma_raw_gk_mS_cm=construction.raw_green_kubo_sigma_mS_cm,
        sigma_projected_gk_mS_cm=construction.projected_green_kubo_sigma_mS_cm,
        sigma_mori_oracle_mS_cm=mori_oracle_result.sigma_mS_cm,
        sigma_recipe_mori_mS_cm=float(closure_input.sigma_recipe_mori_mS_cm),
        projection_gap_mS_cm=(
            construction.projected_green_kubo_sigma_mS_cm
            - construction.raw_green_kubo_sigma_mS_cm
        ),
        recipe_gap_mS_cm=(
            float(closure_input.sigma_recipe_mori_mS_cm)
            - mori_oracle_result.sigma_mS_cm
        ),
        label_gap_mS_cm=(
            float(closure_input.sigma_property_db_mS_cm)
            - construction.raw_green_kubo_sigma_mS_cm
        ),
        gram_rank=gram_rank,
        kz_rank=kz_rank,
        min_kz_eigenvalue=min_kz_eigenvalue,
        dropped_gram_modes=dropped_gram_modes,
        dropped_kz_modes=dropped_kz_modes,
    )


def green_kubo_axis_integrals_from_current_time_series(
    current_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
    center_current: bool,
) -> tuple[float, float, float]:
    """Compute one-sided trapezoidal current autocorrelation integrals by axis."""

    current_array = _validated_current_time_series(current_time_series, "current_time_series")
    _assert_positive_finite(time_step_s, "time_step_s")
    retained_maximum_lag_steps = _validated_maximum_lag_steps(
        maximum_lag_steps,
        current_array.shape[0],
    )
    if center_current:
        centered_current = current_array - np.mean(current_array, axis=0, keepdims=True)
    else:
        centered_current = current_array

    axis_integrals = np.zeros(int(CARTESIAN_AXIS_COUNT), dtype=float)
    frame_count = centered_current.shape[0]
    for lag_steps in range(retained_maximum_lag_steps + 1):
        lagged_covariance_by_axis = np.sum(
            centered_current[lag_steps:, :] * centered_current[: frame_count - lag_steps, :],
            axis=0,
        ) / (frame_count - lag_steps)
        lag_weight = _trapezoid_lag_weight(lag_steps, retained_maximum_lag_steps)
        axis_integrals += lag_weight * lagged_covariance_by_axis

    axis_integrals = axis_integrals * time_step_s
    if not np.all(np.isfinite(axis_integrals)):
        raise ValueError("Green-Kubo axis integrals contain non-finite values")
    return (
        float(axis_integrals[0]),
        float(axis_integrals[1]),
        float(axis_integrals[2]),
    )


def _projected_quadratic_form(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    current_coupling: np.ndarray,
    context: str,
) -> float:
    projected_current = eigenvectors.T @ current_coupling
    positive_mode_mask = _positive_energy_mode_mask(eigenvalues)
    null_mode_projection = projected_current[~positive_mode_mask]
    null_projection_norm = float(np.linalg.norm(null_mode_projection))
    current_norm = float(np.linalg.norm(current_coupling))
    allowed_null_projection = MORI_NUMERICAL_TOLERANCE * max(
        current_norm,
        MORI_NUMERICAL_TOLERANCE,
    )
    if null_projection_norm > allowed_null_projection:
        raise ValueError(
            f"{context} current_coupling projects onto a null energy mode; "
            "zero-frequency conductivity is not finite in this projected basis"
        )
    if not np.any(positive_mode_mask):
        return 0.0
    positive_projected_current = projected_current[positive_mode_mask]
    positive_eigenvalues = eigenvalues[positive_mode_mask]
    quadratic_form = float(
        math.fsum(
            float(projected_value * projected_value / eigenvalue)
            for projected_value, eigenvalue in zip(
                positive_projected_current,
                positive_eigenvalues,
            )
        )
    )
    _assert_nonnegative_finite(quadratic_form, f"{context}.quadratic_form")
    return quadratic_form


def _validated_feature_time_series(feature_time_series: np.ndarray) -> np.ndarray:
    feature_array = np.asarray(feature_time_series, dtype=float)
    if feature_array.ndim != 2:
        raise ValueError("basis_feature_time_series must have shape (n_frames, n_basis_raw)")
    if feature_array.shape[0] < 2:
        raise ValueError("basis_feature_time_series must contain at least two frames")
    if feature_array.shape[1] == 0:
        raise ValueError("basis_feature_time_series must contain at least one feature")
    if not np.all(np.isfinite(feature_array)):
        raise ValueError("basis_feature_time_series contains non-finite values")
    return feature_array


def _validated_current_time_series(
    current_time_series: np.ndarray,
    name: str,
) -> np.ndarray:
    current_array = np.asarray(current_time_series, dtype=float)
    if current_array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_frames, 3)")
    expected_axis_count = int(CARTESIAN_AXIS_COUNT)
    if current_array.shape[1] != expected_axis_count:
        raise ValueError(f"{name} must have shape (n_frames, 3)")
    if current_array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two frames")
    if not np.all(np.isfinite(current_array)):
        raise ValueError(f"{name} contains non-finite values")
    return current_array


def _validated_maximum_lag_steps(
    maximum_lag_steps: int,
    frame_count: int,
) -> int:
    if not isinstance(maximum_lag_steps, int):
        raise TypeError("maximum_lag_steps must be an integer")
    if maximum_lag_steps < 0:
        raise ValueError("maximum_lag_steps must be nonnegative")
    return min(maximum_lag_steps, frame_count - 1)


def _symmetrized_matrix(matrix: np.ndarray) -> np.ndarray:
    return TRAPEZOID_ENDPOINT_WEIGHT * (matrix + matrix.T)


def _eigh_retained_psd(
    matrix: np.ndarray,
    relative_tolerance: float,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symmetrized_matrix = _symmetrized_matrix(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrized_matrix)
    eigenvalue_scale = _eigenvalue_scale(eigenvalues)
    allowed_negative_eigenvalue = relative_tolerance * eigenvalue_scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(
            f"{context} has significant negative eigenvalues; "
            f"minimum={minimum_eigenvalue}, allowed={allowed_negative_eigenvalue}"
        )
    retained_mode_mask = eigenvalues > relative_tolerance * eigenvalue_scale
    return (
        eigenvalues[retained_mode_mask],
        eigenvectors[:, retained_mode_mask],
        eigenvalues,
    )


def _integrated_symmetrized_basis_covariance(
    whitened_basis_time_series: np.ndarray,
    time_step_s: float,
    maximum_lag_steps: int,
) -> np.ndarray:
    basis_dimension = whitened_basis_time_series.shape[1]
    frame_count = whitened_basis_time_series.shape[0]
    integrated_covariance = np.zeros((basis_dimension, basis_dimension), dtype=float)
    for lag_steps in range(maximum_lag_steps + 1):
        lagged_covariance = (
            whitened_basis_time_series[lag_steps:, :].T
            @ whitened_basis_time_series[: frame_count - lag_steps, :]
            / (frame_count - lag_steps)
        )
        lag_weight = _trapezoid_lag_weight(lag_steps, maximum_lag_steps)
        integrated_covariance += lag_weight * _symmetrized_matrix(lagged_covariance)
    return _symmetrized_matrix(integrated_covariance * time_step_s)


def _trapezoid_lag_weight(
    lag_steps: int,
    maximum_lag_steps: int,
) -> float:
    if lag_steps == 0 or lag_steps == maximum_lag_steps:
        return TRAPEZOID_ENDPOINT_WEIGHT
    return 1.0


def _green_kubo_sigma_mS_cm_from_axis_integrals(
    axis_integrals: tuple[float, float, float],
    beta_over_volume: float,
) -> float:
    sigma_S_m = beta_over_volume * math.fsum(axis_integrals) / CARTESIAN_AXIS_COUNT
    _assert_nonnegative_finite(sigma_S_m, "green_kubo_sigma_S_m")
    return sigma_S_m * S_M_TO_MS_CM


def _validated_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix_array.shape}")
    if matrix_array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one basis function")
    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix_array


def _validated_current_coupling_matrix(
    current_coupling_matrix: np.ndarray,
    basis_size: int,
) -> np.ndarray:
    current_coupling_array = np.asarray(current_coupling_matrix, dtype=float)
    if current_coupling_array.ndim != 2:
        raise ValueError("current_coupling_matrix must be a two-dimensional matrix")
    expected_shape = (int(CARTESIAN_AXIS_COUNT), basis_size)
    if current_coupling_array.shape != expected_shape:
        raise ValueError(
            "current_coupling_matrix must have shape "
            f"{expected_shape}, got {current_coupling_array.shape}"
        )
    if not np.all(np.isfinite(current_coupling_array)):
        raise ValueError("current_coupling_matrix contains non-finite values")
    return current_coupling_array


def _validate_symmetric_matrix(matrix: np.ndarray, name: str) -> None:
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=MORI_NUMERICAL_TOLERANCE,
        atol=MORI_NUMERICAL_TOLERANCE,
    ):
        raise ValueError(f"{name} must be symmetric")


def _validate_positive_semidefinite_matrix(matrix: np.ndarray, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    _validate_eigenvalues_are_psd(eigenvalues, name)


def _validate_eigenvalues_are_psd(eigenvalues: np.ndarray, name: str) -> None:
    minimum_eigenvalue = float(np.min(eigenvalues))
    allowed_negative_eigenvalue = MORI_NUMERICAL_TOLERANCE * _eigenvalue_scale(eigenvalues)
    if minimum_eigenvalue < -allowed_negative_eigenvalue:
        raise ValueError(
            f"{name} must be positive semidefinite; minimum eigenvalue is "
            f"{minimum_eigenvalue}"
        )


def _positive_energy_mode_mask(eigenvalues: np.ndarray) -> np.ndarray:
    positive_threshold = MORI_NUMERICAL_TOLERANCE * _eigenvalue_scale(eigenvalues)
    return eigenvalues > positive_threshold


def _validated_sample_id(sample_id: str) -> str:
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a string")
    stripped_sample_id = sample_id.strip()
    if stripped_sample_id == "":
        raise ValueError("sample_id must not be empty")
    return stripped_sample_id


def _eigenvalue_scale(eigenvalues: np.ndarray) -> float:
    return max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)


def _assert_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _assert_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value}")
