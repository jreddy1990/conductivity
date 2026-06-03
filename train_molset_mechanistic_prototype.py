"""Train and audit the mechanism-head MolSet conductivity prototype."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import control_framework.jax_m4_tuning  # noqa: F401  must run before jax import
import jax
import jax.numpy as jnp
import numpy as np
import optax

from constants import T_REF_K
from conductivity.mol_set_sigma_mechanistic_prototype import (
    CURRENT_HEAD_NAMES,
    MECHANISM_FEATURE_NAMES,
    MODEL_FEATURE_NAMES,
    PHYSICAL_FEATURE_NAMES,
    batch_tuple_from_mechanistic_batch,
    compute_physical_feature_stats,
    compute_physical_features_for_batch,
    evaluate_mechanistic_recipe,
    forward_batch,
    init_mechanistic_params,
    loss_fn,
)
from conductivity.mol_set_sigma_unit_aware_prototype import build_unit_aware_recipe_inputs
from conductivity.molset_mechanistic_data import (
    MECHANISTIC_DATA_SOURCES,
    MechanisticBatch,
    MechanisticRow,
    audit_mechanistic_rows,
    build_mechanistic_batch,
    lifsi_dominant_fec_rows,
    load_mechanistic_rows,
    normalization_from_registered_species,
    source_counts,
)
from utils.strict_validation import require_mapping


DEFAULT_STEPS = 1500
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_BATCH_SIZE = 512
DEFAULT_DATA_SOURCES = (
    "property_db",
    "logan2018",
    "valoen2005",
    "transport_targets2019",
    "electrolytomics",
    "calisol23_vv",
    "oedb_li_aux",
    "bamboo_mix_eis",
    "clean_oedb_li_aux",
)
DERIVED_MECHANISM_TARGET_SOURCES = ("property_db",)
LOG_INTERVAL = 250
INTERACTION_DIAGNOSTIC_FEATURE_NAMES = (
    "mixed_anion_competition",
    "salt_pair_molarity_M2",
    "salt_pair_lambda_contrast",
    "salt_pair_anion_size_contrast",
    "salt_pair_binding_contrast",
    "salt_pair_kd_contrast",
    "salt_pair_dielectric_decrement_contrast",
    "salt_pair_stokes_contrast",
    "salt_pair_flex_contrast",
    "salt_additive_shell_coupling",
    "salt_additive_steric_coupling",
    "salt_additive_saturation",
    "salt_pair_transport_contrast",
    "salt_pair_anticorrelation_screening",
    "salt_pair_like_current_support",
    "salt_pair_cluster_transport_support",
    "ionic_network_transport_support",
    "salt_additive_dielectric_screening",
    "salt_additive_anticorrelation_screening",
    "mixed_anion_additive_current_support",
    "salt_additive_like_current_support",
    "salt_additive_cluster_transport_support",
    "additive_transport_drag",
)
INTERACTION_DIAGNOSTIC_INDICES = tuple(
    PHYSICAL_FEATURE_NAMES.index(name) for name in INTERACTION_DIAGNOSTIC_FEATURE_NAMES
)

@dataclass(frozen=True)
class FitMetrics:
    """Scalar fit metrics for a trained prototype batch."""

    loss: float
    mae_mS_cm: float
    rmse_mS_cm: float
    mape_percent: float
    max_abs_mS_cm: float
    density_mae_g_ml: float


@dataclass(frozen=True)
class ConductivityStratumMetrics:
    """Scalar conductivity metrics for an evaluation stratum."""

    stratum: str
    rows: int
    labels: int
    mae_mS_cm: float
    rmse_mS_cm: float
    mape_percent: float
    max_abs_mS_cm: float
    min_temperature_K: float
    max_temperature_K: float
    min_sigma_mS_cm: float
    max_sigma_mS_cm: float


@dataclass(frozen=True)
class HighConductivityAuditRow:
    """High-conductivity labeled row shown to prevent aggregate-metric ambiguity."""

    source: str
    row_index: int
    temperature_K: float
    conductivity_mS_cm: float
    stratum: str
    recipe: Mapping[str, object]


@dataclass(frozen=True)
class AuditDiagnostics:
    """Evaluation-only diagnostics that never feed back into sigma."""

    species_property_distance: float
    loading_distance: float
    interaction_distance: float
    nearest_loading_distance: float
    max_loading_z: float
    support_ratio: float
    unsupported_region: bool


@dataclass(frozen=True)
class AuditRow:
    """One reported prediction row with mechanism diagnostics."""

    label: str
    recipe: Mapping[str, object]
    observed_mS_cm: float | None
    predicted_mS_cm: float
    eta_cP: float
    self_current_scale_prior_mS_cm: float
    sigma_self_mS_cm: float
    cation_anion_distinct_mS_cm: float
    mixed_anion_anticorrelation_mS_cm: float
    cation_cation_distinct_mS_cm: float
    anion_anion_distinct_mS_cm: float
    cluster_drift_mS_cm: float
    ionic_network_current_mS_cm: float
    mixed_anion_additive_current_mS_cm: float
    relaxation_tail_mS_cm: float
    distinct_current_correction_mS_cm: float
    association_fraction: float
    crowding: float
    activity_M: float
    mobile_carrier_density_M: float
    finite_concentration_mobility_factor: float
    finite_concentration_correlation_drive: float
    species_property_distance: float
    loading_distance: float
    interaction_distance: float
    nearest_loading_distance: float
    max_loading_z: float
    support_ratio: float
    unsupported_region: bool


@dataclass(frozen=True)
class TrainingResult:
    """Trained model and audit context."""

    params: Mapping[str, jnp.ndarray]
    norm_mean: np.ndarray
    norm_std: np.ndarray
    physical_mean: np.ndarray
    physical_std: np.ndarray
    train_batch: MechanisticBatch
    holdout_batch: MechanisticBatch | None
    train_physical_z: np.ndarray
    species_support_radius: float
    loading_support_radius: float
    interaction_support_radius: float
    train_rows: tuple[MechanisticRow, ...]
    holdout_rows: tuple[MechanisticRow, ...]
    history: tuple[tuple[int, float], ...]
    batch_size: int
    batching_policy: str
    temperature_calibration: "TemperatureCalibrationAudit"
    current_gate_calibration: "CurrentGateCalibration"
    derived_target_audit: "DerivedMechanismTargetAudit"


@dataclass(frozen=True)
class CurrentGateCalibration:
    """Data-derived current-gate initialization from auxiliary distinct-current labels."""

    labels: int
    negative_label_fraction: float
    positive_signed_fraction: float
    cation_anion_gate_probability: float
    positive_gate_probability: float
    generic_positive_current_scale: float
    mixed_anion_anticorrelation_gate_probability: float


@dataclass(frozen=True)
class TemperatureCalibrationAudit:
    """Data-derived temperature/friction calibration from same-recipe series."""

    grouped_recipe_series: int
    candidate_pairs: int
    accepted_pairs: int
    rejected_nonmonotone_pairs: int
    activation_min_K: float
    activation_median_K: float
    activation_max_K: float


@dataclass(frozen=True)
class DerivedMechanismTargetAudit:
    """Audit for mechanism targets derived from measured sigma plus property priors."""

    candidate_rows: int
    viscosity_targets: int
    association_targets: int
    current_distinct_targets: int
    viscosity_min_cP: float
    viscosity_max_cP: float
    association_min: float
    association_max: float
    current_distinct_min_mS_cm: float
    current_distinct_max_mS_cm: float


@dataclass(frozen=True)
class LoadedMechanisticCheckpoint:
    """Self-contained checkpoint payload for prototype evaluation."""

    params: Mapping[str, jnp.ndarray]
    norm_mean: np.ndarray
    norm_std: np.ndarray
    physical_mean: np.ndarray
    physical_std: np.ndarray


def train_mechanistic_prototype(
    n_steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    seed: int,
    holdout_family: str,
    data_sources: Sequence[str],
    batch_size: int,
) -> TrainingResult:
    """Train the MolSet mechanism-head prototype on empirical conductivity rows."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = load_mechanistic_rows(data_sources)
    train_rows, holdout_rows = _split_rows(rows, holdout_family)
    norm_mean, norm_std = normalization_from_registered_species()
    train_batch = build_mechanistic_batch(train_rows, norm_mean, norm_std)
    holdout_batch = None
    if holdout_rows:
        holdout_batch = build_mechanistic_batch(holdout_rows, norm_mean, norm_std)
    train_batch, derived_target_audit = _apply_property_derived_mechanism_targets(train_batch)
    temperature_calibration = _calibrate_temperature_transport_activation(train_rows)
    physical_mean, physical_std = compute_physical_feature_stats(train_batch)
    params = init_mechanistic_params(jax.random.PRNGKey(seed), physical_mean, physical_std)
    params = _set_temperature_transport_activation(params, temperature_calibration)
    params, current_gate_calibration = _initialize_current_gates_from_auxiliary_labels(params, train_batch)
    batch_tuple = batch_tuple_from_mechanistic_batch(train_batch)
    training_index_groups = _source_index_groups(train_batch.sources)
    fixed_physical_mean = jnp.asarray(physical_mean)
    fixed_physical_std = jnp.asarray(physical_std)
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    def train_step(
        step_params: Mapping[str, jnp.ndarray],
        step_opt_state: optax.OptState,
        step_batch_tuple: tuple[jnp.ndarray, ...],
    ) -> tuple[Mapping[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
        loss_value, grads = jax.value_and_grad(loss_fn)(step_params, step_batch_tuple)
        frozen_grads = dict(grads)
        frozen_grads["physical_mean"] = jnp.zeros_like(grads["physical_mean"])
        frozen_grads["physical_std"] = jnp.zeros_like(grads["physical_std"])
        frozen_grads["generic_positive_current_scale"] = jnp.zeros_like(
            grads["generic_positive_current_scale"]
        )
        frozen_grads["temperature_transport_activation_K"] = jnp.zeros_like(
            grads["temperature_transport_activation_K"]
        )
        updates, next_opt_state = optimizer.update(frozen_grads, step_opt_state, step_params)
        next_params = optax.apply_updates(step_params, updates)
        restored_params = dict(next_params)
        restored_params["physical_mean"] = fixed_physical_mean
        restored_params["physical_std"] = fixed_physical_std
        return restored_params, next_opt_state, loss_value

    jit_train_step = jax.jit(train_step)
    history: list[tuple[int, float]] = []
    for step_idx in range(n_steps + 1):
        if step_idx % LOG_INTERVAL == 0 or step_idx == n_steps:
            history_batch_tuple = _training_step_batch_tuple(
                batch_tuple,
                len(train_rows),
                batch_size,
                step_idx + n_steps + 1,
                seed,
                training_index_groups,
            )
            current_loss = float(loss_fn(params, history_batch_tuple))
            history.append((step_idx, current_loss))
        if step_idx == n_steps:
            continue
        step_batch_tuple = _training_step_batch_tuple(
            batch_tuple,
            len(train_rows),
            batch_size,
            step_idx,
            seed,
            training_index_groups,
        )
        params, opt_state, _loss_value = jit_train_step(params, opt_state, step_batch_tuple)

    train_physical_z = _batch_physical_z(
        batch=train_batch,
        physical_mean=physical_mean,
        physical_std=physical_std,
    )
    train_interaction_z = train_physical_z[:, np.asarray(INTERACTION_DIAGNOSTIC_INDICES, dtype=int)]
    return TrainingResult(
        params=params,
        norm_mean=norm_mean,
        norm_std=norm_std,
        physical_mean=physical_mean,
        physical_std=physical_std,
        train_batch=train_batch,
        holdout_batch=holdout_batch,
        train_physical_z=train_physical_z,
        species_support_radius=_species_support_radius(train_batch),
        loading_support_radius=_nearest_distance_support_radius(train_physical_z),
        interaction_support_radius=_nearest_distance_support_radius(train_interaction_z),
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        history=tuple(history),
        batch_size=batch_size,
        batching_policy="source-balanced",
        temperature_calibration=temperature_calibration,
        current_gate_calibration=current_gate_calibration,
        derived_target_audit=derived_target_audit,
    )


def _apply_property_derived_mechanism_targets(
    batch: MechanisticBatch,
) -> tuple[MechanisticBatch, DerivedMechanismTargetAudit]:
    physical = compute_physical_features_for_batch(batch)
    candidate_mask = _derived_mechanism_candidate_mask(batch)
    candidate_count = int(np.sum(candidate_mask))
    if candidate_count == 0:
        return batch, _empty_derived_target_audit()

    eta_prior = _physical_column(physical, "eta_solution_prior_cP")
    contact_pair_prior = _physical_column(physical, "contact_pair_prior")
    activity_prior = _physical_column(physical, "activity_prior_M")
    mean_lambda0 = _physical_column(physical, "mean_lambda0_S_cm2_mol")
    crowding_prior = _physical_column(physical, "crowding_prior")
    additive_transport_drag = _physical_column(physical, "additive_transport_drag")
    screening_support = (
        _physical_column(physical, "salt_additive_dielectric_screening")
        + _physical_column(physical, "salt_additive_anticorrelation_screening")
    )

    _require_positive_finite_targets(eta_prior[candidate_mask], "derived viscosity target")
    _require_nonnegative_finite_targets(screening_support[candidate_mask], "derived screening support")
    association_target = contact_pair_prior / (1.0 + screening_support)
    _require_fraction_targets(association_target[candidate_mask], "derived association target")
    free_ion_target = 1.0 - association_target
    denominator = eta_prior * (1.0 + crowding_prior + additive_transport_drag)
    _require_positive_finite_targets(denominator[candidate_mask], "derived sigma-backbone denominator")
    self_current_scale_prior_target = (
        activity_prior
        * mean_lambda0
        * free_ion_target
        / denominator
    )
    _require_positive_finite_targets(
        self_current_scale_prior_target[candidate_mask],
        "derived self-current scale prior target",
    )
    current_distinct_target = batch.sigma_mS_cm - self_current_scale_prior_target
    _require_finite_targets(current_distinct_target[candidate_mask], "derived distinct-current target")

    viscosity_target_mask = candidate_mask & (batch.viscosity_mask <= 0.0)
    association_target_mask = candidate_mask & (batch.association_fraction_mask <= 0.0)
    current_distinct_target_mask = candidate_mask & (batch.current_distinct_mask <= 0.0)

    viscosity_cP = np.where(viscosity_target_mask, eta_prior, batch.viscosity_cP)
    viscosity_mask = np.where(viscosity_target_mask, 1.0, batch.viscosity_mask)
    association_fraction = np.where(
        association_target_mask,
        association_target,
        batch.association_fraction,
    )
    association_fraction_mask = np.where(
        association_target_mask,
        1.0,
        batch.association_fraction_mask,
    )
    current_distinct_mS_cm = np.where(
        current_distinct_target_mask,
        current_distinct_target,
        batch.current_distinct_mS_cm,
    )
    current_distinct_mask = np.where(
        current_distinct_target_mask,
        1.0,
        batch.current_distinct_mask,
    )

    updated_batch = replace(
        batch,
        viscosity_cP=viscosity_cP,
        viscosity_mask=viscosity_mask,
        association_fraction=association_fraction,
        association_fraction_mask=association_fraction_mask,
        current_distinct_mS_cm=current_distinct_mS_cm,
        current_distinct_mask=current_distinct_mask,
    )
    return (
        updated_batch,
        DerivedMechanismTargetAudit(
            candidate_rows=candidate_count,
            viscosity_targets=int(np.sum(viscosity_target_mask)),
            association_targets=int(np.sum(association_target_mask)),
            current_distinct_targets=int(np.sum(current_distinct_target_mask)),
            viscosity_min_cP=_masked_min(eta_prior, viscosity_target_mask),
            viscosity_max_cP=_masked_max(eta_prior, viscosity_target_mask),
            association_min=_masked_min(association_target, association_target_mask),
            association_max=_masked_max(association_target, association_target_mask),
            current_distinct_min_mS_cm=_masked_min(
                current_distinct_target,
                current_distinct_target_mask,
            ),
            current_distinct_max_mS_cm=_masked_max(
                current_distinct_target,
                current_distinct_target_mask,
            ),
        ),
    )


def _calibrate_temperature_transport_activation(
    rows: Sequence[MechanisticRow],
) -> TemperatureCalibrationAudit:
    grouped: dict[tuple[str, str], dict[float, list[float]]] = {}
    for row in rows:
        if row.has_conductivity <= 0.0:
            continue
        if row.conductivity_mS_cm <= 0.0:
            raise ValueError("Temperature calibration requires positive conductivity labels")
        if row.temperature_K <= 0.0:
            raise ValueError("Temperature calibration requires positive temperatures")
        key = (row.source, row.recipe_key)
        if key not in grouped:
            grouped[key] = {}
        if row.temperature_K not in grouped[key]:
            grouped[key][row.temperature_K] = []
        grouped[key][row.temperature_K].append(row.conductivity_mS_cm)

    grouped_recipe_series = 0
    candidate_pairs = 0
    rejected_nonmonotone_pairs = 0
    activations: list[float] = []
    for temperature_to_sigma in grouped.values():
        if len(temperature_to_sigma) < 2:
            continue
        grouped_recipe_series += 1
        sorted_temperatures = sorted(temperature_to_sigma)
        mean_sigmas = [
            float(np.mean(np.asarray(temperature_to_sigma[temperature], dtype=np.float64)))
            for temperature in sorted_temperatures
        ]
        for idx in range(len(sorted_temperatures) - 1):
            lower_T = float(sorted_temperatures[idx])
            upper_T = float(sorted_temperatures[idx + 1])
            lower_sigma = float(mean_sigmas[idx])
            upper_sigma = float(mean_sigmas[idx + 1])
            candidate_pairs += 1
            if upper_sigma <= lower_sigma:
                rejected_nonmonotone_pairs += 1
                continue
            denominator = (1.0 / lower_T) - (1.0 / upper_T)
            if denominator <= 0.0:
                raise ValueError("Temperature calibration encountered non-increasing temperature order")
            activation_K = float(np.log(upper_sigma / lower_sigma) / denominator)
            if not np.isfinite(activation_K) or activation_K <= 0.0:
                raise ValueError("Temperature calibration produced a non-positive activation value")
            activations.append(activation_K)

    if not activations:
        raise ValueError(
            "Temperature calibration found no monotone same-source same-recipe conductivity-temperature pairs"
        )
    activation_array = np.asarray(activations, dtype=np.float64)
    return TemperatureCalibrationAudit(
        grouped_recipe_series=grouped_recipe_series,
        candidate_pairs=candidate_pairs,
        accepted_pairs=len(activations),
        rejected_nonmonotone_pairs=rejected_nonmonotone_pairs,
        activation_min_K=float(np.min(activation_array)),
        activation_median_K=float(np.median(activation_array)),
        activation_max_K=float(np.max(activation_array)),
    )


def _set_temperature_transport_activation(
    params: Mapping[str, jnp.ndarray],
    audit: TemperatureCalibrationAudit,
) -> Mapping[str, jnp.ndarray]:
    next_params = dict(params)
    next_params["temperature_transport_activation_K"] = jnp.asarray(
        audit.activation_median_K
    )
    return next_params


def _derived_mechanism_candidate_mask(batch: MechanisticBatch) -> np.ndarray:
    additive_rows = np.sum(batch.additive_weight_fraction, axis=1) > 0.0
    multi_salt_rows = np.sum(batch.salt_molarity > 0.0, axis=1) > 1
    ionic_rows = np.sum(batch.salt_molarity, axis=1) > 0.0
    source_rows = np.isin(np.asarray(batch.sources), np.asarray(DERIVED_MECHANISM_TARGET_SOURCES))
    return (
        (batch.conductivity_mask > 0.0)
        & ionic_rows
        & source_rows
        & (additive_rows | multi_salt_rows)
    )


def _physical_column(physical: np.ndarray, name: str) -> np.ndarray:
    return physical[:, PHYSICAL_FEATURE_NAMES.index(name)]


def _empty_derived_target_audit() -> DerivedMechanismTargetAudit:
    return DerivedMechanismTargetAudit(
        candidate_rows=0,
        viscosity_targets=0,
        association_targets=0,
        current_distinct_targets=0,
        viscosity_min_cP=0.0,
        viscosity_max_cP=0.0,
        association_min=0.0,
        association_max=0.0,
        current_distinct_min_mS_cm=0.0,
        current_distinct_max_mS_cm=0.0,
    )


def _require_finite_targets(values: np.ndarray, context: str) -> None:
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{context} contains non-finite values")


def _require_positive_finite_targets(values: np.ndarray, context: str) -> None:
    _require_finite_targets(values, context)
    if np.any(values <= 0.0):
        raise ValueError(f"{context} must be positive")


def _require_nonnegative_finite_targets(values: np.ndarray, context: str) -> None:
    _require_finite_targets(values, context)
    if np.any(values < 0.0):
        raise ValueError(f"{context} must be nonnegative")


def _require_fraction_targets(values: np.ndarray, context: str) -> None:
    _require_finite_targets(values, context)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{context} must be in [0, 1]")


def _masked_min(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    if selected.size == 0:
        return 0.0
    _require_finite_targets(selected, "masked minimum")
    return float(np.min(selected))


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    if selected.size == 0:
        return 0.0
    _require_finite_targets(selected, "masked maximum")
    return float(np.max(selected))


def _training_step_batch_tuple(
    full_batch_tuple: tuple[jnp.ndarray, ...],
    n_rows: int,
    batch_size: int,
    step_idx: int,
    seed: int,
    source_index_groups: Sequence[np.ndarray],
) -> tuple[jnp.ndarray, ...]:
    if batch_size >= n_rows:
        return full_batch_tuple
    rng = np.random.default_rng(seed + step_idx)
    indices = _source_balanced_indices(source_index_groups, batch_size, rng)
    return tuple(array[indices] for array in full_batch_tuple)


def _source_index_groups(sources: Sequence[str]) -> tuple[np.ndarray, ...]:
    source_array = np.asarray(sources)
    groups = tuple(
        np.flatnonzero(source_array == source) for source in sorted(set(sources))
    )
    if not groups:
        raise ValueError("At least one source group is required for source-balanced batching")
    for group in groups:
        if group.size == 0:
            raise ValueError("Source-balanced batching received an empty source group")
    return groups


def _source_balanced_indices(
    source_index_groups: Sequence[np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_groups = len(source_index_groups)
    group_order = rng.permutation(n_groups)
    base_draw = batch_size // n_groups
    remainder = batch_size % n_groups
    selected: list[np.ndarray] = []
    for position, group_idx in enumerate(group_order):
        draw_count = base_draw
        if position < remainder:
            draw_count += 1
        if draw_count <= 0:
            continue
        group = source_index_groups[int(group_idx)]
        replace = draw_count > group.size
        selected.append(rng.choice(group, size=draw_count, replace=replace))
    if not selected:
        raise ValueError("source-balanced batching produced no row indices")
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices


def _initialize_current_gates_from_auxiliary_labels(
    params: Mapping[str, jnp.ndarray],
    batch: MechanisticBatch,
) -> tuple[Mapping[str, jnp.ndarray], CurrentGateCalibration]:
    current_label_mask = (
        (batch.current_distinct_mask > 0.0)
        & (batch.cation_self_current_mask > 0.0)
        & (batch.anion_self_current_mask > 0.0)
    )
    label_count = int(np.sum(current_label_mask))
    if label_count == 0:
        return (
            params,
            CurrentGateCalibration(
                labels=0,
                negative_label_fraction=0.0,
                positive_signed_fraction=0.0,
                cation_anion_gate_probability=0.5,
                positive_gate_probability=0.5,
                generic_positive_current_scale=1.0,
                mixed_anion_anticorrelation_gate_probability=0.5,
            ),
        )

    self_current = (
        batch.cation_self_current_mS_cm[current_label_mask]
        + batch.anion_self_current_mS_cm[current_label_mask]
    )
    if np.any(self_current <= 0.0) or np.any(~np.isfinite(self_current)):
        raise ValueError("Current-gate calibration requires positive finite self-current labels")
    distinct_ratio = batch.current_distinct_mS_cm[current_label_mask] / self_current
    if np.any(~np.isfinite(distinct_ratio)):
        raise ValueError("Current-gate calibration found non-finite distinct/self ratios")

    negative_label_fraction = float(np.mean(distinct_ratio < 0.0))
    positive_signed_fraction = float(np.mean(np.maximum(distinct_ratio, 0.0)))
    cation_anion_gate_probability = _strict_open_probability(
        negative_label_fraction,
        "negative distinct-current label fraction",
    )
    positive_gate_probability = _strict_open_probability(
        positive_signed_fraction,
        "mean positive distinct/self fraction",
    )
    next_params = dict(params)
    bias = np.asarray(params["mech_out_b"]).copy()
    bias[CURRENT_HEAD_NAMES.index("cation_anion_gate")] = _logit_probability(
        cation_anion_gate_probability
    )
    for gate_name in (
        "cation_cation_gate",
        "anion_anion_gate",
        "cluster_drift_gate",
        "relaxation_tail_gate",
        "ionic_network_gate",
    ):
        bias[CURRENT_HEAD_NAMES.index(gate_name)] = _logit_probability(positive_gate_probability)
    next_params["mech_out_b"] = jnp.asarray(bias)
    next_params["generic_positive_current_scale"] = jnp.asarray(positive_signed_fraction)
    next_params["mixed_anion_anticorrelation_logit"] = jnp.asarray(
        _logit_probability(cation_anion_gate_probability)
    )
    return (
        next_params,
        CurrentGateCalibration(
            labels=label_count,
            negative_label_fraction=negative_label_fraction,
            positive_signed_fraction=positive_signed_fraction,
            cation_anion_gate_probability=cation_anion_gate_probability,
            positive_gate_probability=positive_gate_probability,
            generic_positive_current_scale=positive_signed_fraction,
            mixed_anion_anticorrelation_gate_probability=cation_anion_gate_probability,
        ),
    )


def _strict_open_probability(value: float, context: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    if parsed <= 0.0 or parsed >= 1.0:
        raise ValueError(f"{context} must be in the open unit interval, got {parsed}")
    return parsed


def _logit_probability(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def _batch_physical_z(
    batch: MechanisticBatch,
    physical_mean: np.ndarray,
    physical_std: np.ndarray,
) -> np.ndarray:
    physical = compute_physical_features_for_batch(batch)
    return (physical - physical_mean[None, :]) / physical_std[None, :]


def _species_support_radius(batch: MechanisticBatch) -> float:
    active_props = batch.species_props_norm[batch.mask > 0.0]
    unique_props = np.unique(active_props, axis=0)
    return _nearest_distance_support_radius(unique_props)


def _nearest_distance_support_radius(points: np.ndarray) -> float:
    if points.shape[0] <= 1:
        return float(np.finfo(np.float64).eps)
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, _indices = tree.query(points, k=2)
    nearest = distances[:, 1] / np.sqrt(float(points.shape[1]))
    radius = float(np.max(nearest))
    if not np.isfinite(radius) or radius <= 0.0:
        return float(np.finfo(np.float64).eps)
    return radius


def compute_fit_metrics(
    params: Mapping[str, jnp.ndarray],
    batch: MechanisticBatch,
) -> FitMetrics:
    """Compute empirical sigma and available-density metrics."""

    log_pred, features = forward_batch(
        params=params,
        species_props_norm=jnp.asarray(batch.species_props_norm),
        species_props_raw=jnp.asarray(batch.species_props_raw),
        solvent_volume_fraction=jnp.asarray(batch.solvent_volume_fraction),
        salt_molarity=jnp.asarray(batch.salt_molarity),
        additive_weight_fraction=jnp.asarray(batch.additive_weight_fraction),
        mask=jnp.asarray(batch.mask),
        temperature_K=jnp.asarray(batch.temperature_K),
    )
    pred = np.asarray(jnp.exp(log_pred))
    sigma_mask = batch.conductivity_mask > 0.0
    if not np.any(sigma_mask):
        raise ValueError("compute_fit_metrics requires at least one conductivity-labeled row")
    target = batch.sigma_mS_cm[sigma_mask]
    err = pred[sigma_mask] - target
    density_pred = np.asarray(features[:, len(PHYSICAL_FEATURE_NAMES)])
    density_mask = batch.density_mask > 0.0
    density_mae = 0.0
    if np.any(density_mask):
        density_mae = float(np.mean(np.abs(density_pred[density_mask] - batch.density_g_ml[density_mask])))
    return FitMetrics(
        loss=float(loss_fn(params, batch_tuple_from_mechanistic_batch(batch))),
        mae_mS_cm=float(np.mean(np.abs(err))),
        rmse_mS_cm=float(np.sqrt(np.mean(err * err))),
        mape_percent=float(100.0 * np.mean(np.abs(err) / target)),
        max_abs_mS_cm=float(np.max(np.abs(err))),
        density_mae_g_ml=density_mae,
    )


def conductivity_stratum_metrics(training: TrainingResult) -> tuple[ConductivityStratumMetrics, ...]:
    """Compute scalar conductivity metrics by evaluation stratum."""

    if len(training.train_rows) != len(training.train_batch.sources):
        raise ValueError("Training rows and batch rows must have matching order for stratum metrics")
    log_pred, _features = forward_batch(
        params=training.params,
        species_props_norm=jnp.asarray(training.train_batch.species_props_norm),
        species_props_raw=jnp.asarray(training.train_batch.species_props_raw),
        solvent_volume_fraction=jnp.asarray(training.train_batch.solvent_volume_fraction),
        salt_molarity=jnp.asarray(training.train_batch.salt_molarity),
        additive_weight_fraction=jnp.asarray(training.train_batch.additive_weight_fraction),
        mask=jnp.asarray(training.train_batch.mask),
        temperature_K=jnp.asarray(training.train_batch.temperature_K),
    )
    pred = np.asarray(jnp.exp(log_pred))
    strata = np.asarray([_evaluation_stratum(row) for row in training.train_rows])
    metrics: list[ConductivityStratumMetrics] = []
    for stratum in sorted(set(strata)):
        stratum_mask = strata == stratum
        label_mask = stratum_mask & (training.train_batch.conductivity_mask > 0.0)
        row_count = int(np.sum(stratum_mask))
        label_count = int(np.sum(label_mask))
        temperatures = training.train_batch.temperature_K[stratum_mask]
        if label_count == 0:
            metrics.append(
                ConductivityStratumMetrics(
                    stratum=stratum,
                    rows=row_count,
                    labels=0,
                    mae_mS_cm=0.0,
                    rmse_mS_cm=0.0,
                    mape_percent=0.0,
                    max_abs_mS_cm=0.0,
                    min_temperature_K=float(np.min(temperatures)),
                    max_temperature_K=float(np.max(temperatures)),
                    min_sigma_mS_cm=0.0,
                    max_sigma_mS_cm=0.0,
                )
            )
            continue
        target = training.train_batch.sigma_mS_cm[label_mask]
        error = pred[label_mask] - target
        metrics.append(
            ConductivityStratumMetrics(
                stratum=stratum,
                rows=row_count,
                labels=label_count,
                mae_mS_cm=float(np.mean(np.abs(error))),
                rmse_mS_cm=float(np.sqrt(np.mean(error * error))),
                mape_percent=float(100.0 * np.mean(np.abs(error) / target)),
                max_abs_mS_cm=float(np.max(np.abs(error))),
                min_temperature_K=float(np.min(training.train_batch.temperature_K[label_mask])),
                max_temperature_K=float(np.max(training.train_batch.temperature_K[label_mask])),
                min_sigma_mS_cm=float(np.min(target)),
                max_sigma_mS_cm=float(np.max(target)),
            )
        )
    return tuple(metrics)


def high_conductivity_audit_rows(
    rows: Sequence[MechanisticRow],
    max_rows: int,
) -> tuple[HighConductivityAuditRow, ...]:
    """Highest-conductivity labeled rows shown to prevent aggregate-metric ambiguity."""

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    selected = [
        row
        for row in rows
        if row.has_conductivity > 0.0
    ]
    selected.sort(key=lambda row: row.conductivity_mS_cm, reverse=True)
    return tuple(
        HighConductivityAuditRow(
            source=row.source,
            row_index=row.row_index,
            temperature_K=row.temperature_K,
            conductivity_mS_cm=row.conductivity_mS_cm,
            stratum=_evaluation_stratum(row),
            recipe=row.recipe,
        )
        for row in selected[:max_rows]
    )


def _evaluation_stratum(row: MechanisticRow) -> str:
    if row.has_conductivity <= 0.0:
        return f"auxiliary_only|{_temperature_key(row.temperature_K)}"
    return f"conductivity|{_temperature_key(row.temperature_K)}"


def _temperature_key(temperature_K: float) -> str:
    return f"T={float(temperature_K):.2f}K"


def empirical_lifsi_fec_audit(
    training: TrainingResult,
) -> tuple[AuditRow, ...]:
    """Audit exact empirical LiFSI-dominant mixed-salt FEC rows."""

    rows = lifsi_dominant_fec_rows(training.train_rows + training.holdout_rows)
    audit_rows: list[AuditRow] = []
    for row in rows:
        label = _fec_label(row.recipe)
        audit_rows.append(
            _audit_recipe(
                label=label,
                recipe=row.recipe,
                observed_mS_cm=row.conductivity_mS_cm,
                training=training,
            )
        )
    return tuple(audit_rows)


def generated_sweep_audit(
    training: TrainingResult,
    label_prefix: str,
    recipes: Sequence[Mapping[str, object]],
) -> tuple[AuditRow, ...]:
    """Audit generated recipe sweeps without using observed labels."""

    rows: list[AuditRow] = []
    for recipe in recipes:
        rows.append(
            _audit_recipe(
                label=f"{label_prefix} {_recipe_loading_label(recipe)}",
                recipe=recipe,
                observed_mS_cm=None,
                training=training,
            )
        )
    return tuple(rows)


def fec_single_salt_sweep(salt_name: str) -> tuple[Mapping[str, object], ...]:
    """EC:DMC 30:70 v/v + one 1.0 M salt + FEC loading sweep."""

    return tuple(_fec_single_salt_recipe(salt_name, loading) for loading in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15))


def ttfp_fr_sweep() -> tuple[Mapping[str, object], ...]:
    """EC:DMC 30:70 v/v + LiPF6 1.0 M + TTFP loading sweep."""

    return tuple(_additive_recipe("LiPF6", "TTFP", loading) for loading in (0.0, 0.05, 0.10, 0.15))


def salt_concentration_sweep(salt_name: str) -> tuple[Mapping[str, object], ...]:
    """EC:DMC 30:70 v/v + salt concentration sweep."""

    return tuple(_salt_recipe(salt_name, concentration) for concentration in (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0))


def print_training_report(training: TrainingResult, include_fit_metrics: bool) -> None:
    """Print fit metrics and sweep audits."""

    print("MolSet mechanistic prototype training")
    print(f"  data sources: {','.join(source for source in source_counts(training.train_rows + training.holdout_rows))}")
    print("  dataset audit:")
    for audit in audit_mechanistic_rows(training.train_rows + training.holdout_rows):
        print(
            f"    {audit.source}: rows={audit.rows}, "
            f"sigma={audit.conductivity_labels}, viscosity={audit.viscosity_labels}, "
            f"cation_self={audit.cation_self_current_labels}, "
            f"anion_self={audit.anion_self_current_labels}, "
            f"distinct={audit.current_distinct_labels}, "
            f"assoc={audit.association_fraction_labels}, "
            f"multi_salt={audit.multi_salt_rows}, additives={audit.additive_rows}, "
            f"sigma_range=[{audit.min_sigma_mS_cm:.3f},{audit.max_sigma_mS_cm:.3f}], "
            f"T=[{audit.min_temperature_K:.2f},{audit.max_temperature_K:.2f}]"
        )
    print(f"  train rows: {len(training.train_rows)}")
    print(f"  holdout rows: {len(training.holdout_rows)}")
    print(f"  batch size: {training.batch_size}")
    print(f"  batching policy: {training.batching_policy}")
    print(
        "  derived mechanism-target audit: "
        f"candidates={training.derived_target_audit.candidate_rows}, "
        f"viscosity={training.derived_target_audit.viscosity_targets} "
        f"[{training.derived_target_audit.viscosity_min_cP:.3f},"
        f"{training.derived_target_audit.viscosity_max_cP:.3f}] cP, "
        f"association={training.derived_target_audit.association_targets} "
        f"[{training.derived_target_audit.association_min:.3f},"
        f"{training.derived_target_audit.association_max:.3f}], "
        f"distinct={training.derived_target_audit.current_distinct_targets} "
        f"[{training.derived_target_audit.current_distinct_min_mS_cm:.3f},"
        f"{training.derived_target_audit.current_distinct_max_mS_cm:.3f}] mS/cm"
    )
    print(
        "  temperature/friction calibration: "
        f"series={training.temperature_calibration.grouped_recipe_series}, "
        f"pairs={training.temperature_calibration.candidate_pairs}, "
        f"accepted={training.temperature_calibration.accepted_pairs}, "
        f"rejected_nonmonotone={training.temperature_calibration.rejected_nonmonotone_pairs}, "
        f"activation_K=[{training.temperature_calibration.activation_min_K:.1f},"
        f"{training.temperature_calibration.activation_median_K:.1f},"
        f"{training.temperature_calibration.activation_max_K:.1f}]"
    )
    print(
        "  current-gate calibration: "
        f"labels={training.current_gate_calibration.labels}, "
        f"negative_fraction={training.current_gate_calibration.negative_label_fraction:.3f}, "
        f"positive_signed_fraction={training.current_gate_calibration.positive_signed_fraction:.3f}, "
        f"ca_gate={training.current_gate_calibration.cation_anion_gate_probability:.3f}, "
        f"positive_gate={training.current_gate_calibration.positive_gate_probability:.3f}, "
        f"generic_positive_scale={training.current_gate_calibration.generic_positive_current_scale:.3f}, "
        "mixed_ca_gate="
        f"{training.current_gate_calibration.mixed_anion_anticorrelation_gate_probability:.3f}, "
        "mixed_additive=physical_support"
    )
    print(f"  density-labeled train rows: {int(np.sum(training.train_batch.density_mask))}")
    print(f"  viscosity-labeled train rows: {int(np.sum(training.train_batch.viscosity_mask))}")
    print(f"  dielectric-labeled train rows: {int(np.sum(training.train_batch.dielectric_mask))}")
    print(
        "  current-decomposition-labeled train rows: "
        f"{int(np.sum(training.train_batch.cation_self_current_mask))}/"
        f"{int(np.sum(training.train_batch.anion_self_current_mask))}/"
        f"{int(np.sum(training.train_batch.cation_anion_distinct_mask))}/"
        f"{int(np.sum(training.train_batch.current_distinct_mask))}"
    )
    print(
        "  association-labeled train rows: "
        f"{int(np.sum(training.train_batch.association_fraction_mask))}"
    )
    print("  loss history:")
    for step_idx, loss_value in training.history:
        print(f"    step {step_idx:5d}: loss={loss_value:.6f}")
    print("")
    if include_fit_metrics:
        _print_metrics("train", compute_fit_metrics(training.params, training.train_batch))
        if training.holdout_batch is not None:
            _print_metrics("holdout", compute_fit_metrics(training.params, training.holdout_batch))
        _print_conductivity_stratum_metrics(conductivity_stratum_metrics(training))
        _print_high_conductivity_rows(high_conductivity_audit_rows(training.train_rows, max_rows=12))
        print("")
    else:
        print("fit metrics: skipped by --skip-fit-metrics")
        print("")
    _print_audit_table("Empirical LiFSI-dominant mixed-salt + FEC", empirical_lifsi_fec_audit(training))
    _print_lifsi_calibration_ablation(training)
    _print_audit_table(
        "Generated EC:DMC 30:70 + LiFSI 1.0 M + FEC",
        generated_sweep_audit(training, "LiFSI", fec_single_salt_sweep("LiFSI")),
    )
    _print_audit_table(
        "Generated EC:DMC 30:70 + LiPF6 1.0 M + FEC",
        generated_sweep_audit(training, "LiPF6", fec_single_salt_sweep("LiPF6")),
    )
    _print_audit_table(
        "Generated EC:DMC 30:70 + LiPF6 1.0 M + TTFP",
        generated_sweep_audit(training, "TTFP", ttfp_fr_sweep()),
    )
    _print_audit_table(
        "Generated EC:DMC 30:70 + LiPF6 concentration",
        generated_sweep_audit(training, "LiPF6", salt_concentration_sweep("LiPF6")),
    )
    _print_acceptance_summary(training)


def _print_acceptance_summary(training: TrainingResult) -> None:
    lifsi_rows = empirical_lifsi_fec_audit(training)
    lifsi_observed = np.asarray([row.observed_mS_cm for row in lifsi_rows], dtype=np.float64)
    lifsi_predicted = np.asarray([row.predicted_mS_cm for row in lifsi_rows], dtype=np.float64)
    lifsi_eta = np.asarray([row.eta_cP for row in lifsi_rows], dtype=np.float64)
    lifsi_mix_add = np.asarray(
        [row.mixed_anion_additive_current_mS_cm for row in lifsi_rows],
        dtype=np.float64,
    )
    lifsi_max_abs = float(np.max(np.abs(lifsi_predicted - lifsi_observed)))
    lifsi_pass = (
        lifsi_max_abs <= 0.1
        and lifsi_predicted[1] > lifsi_predicted[0]
        and lifsi_predicted[2] > lifsi_predicted[0]
        and lifsi_predicted[3] > lifsi_predicted[0] - 0.25
        and bool(np.all(np.diff(lifsi_eta) > 0.0))
        and lifsi_mix_add[1] > lifsi_mix_add[0]
        and lifsi_mix_add[2] >= lifsi_mix_add[1]
    )

    lifsi_generated = generated_sweep_audit(training, "LiFSI", fec_single_salt_sweep("LiFSI"))
    lipf6_generated = generated_sweep_audit(training, "LiPF6", fec_single_salt_sweep("LiPF6"))
    lifsi_generated_pred = np.asarray([row.predicted_mS_cm for row in lifsi_generated], dtype=np.float64)
    lipf6_generated_pred = np.asarray([row.predicted_mS_cm for row in lipf6_generated], dtype=np.float64)
    lipf6_control_pass = (
        lifsi_generated_pred[1] > lifsi_generated_pred[0]
        and lifsi_generated_pred[2] >= lifsi_generated_pred[0]
        and lipf6_generated_pred[2] < lipf6_generated_pred[0]
        and lipf6_generated_pred[-1] < lipf6_generated_pred[0]
    )

    fr_rows = generated_sweep_audit(training, "TTFP", ttfp_fr_sweep())
    fr_pred = np.asarray([row.predicted_mS_cm for row in fr_rows], dtype=np.float64)
    fr_eta = np.asarray([row.eta_cP for row in fr_rows], dtype=np.float64)
    fr_pass = (
        bool(np.all(np.diff(fr_eta) > 0.0))
        and fr_pred[1] < fr_pred[0]
        and fr_pred[2] < fr_pred[1]
        and fr_pred[3] < fr_pred[2]
    )

    salt_rows = generated_sweep_audit(training, "LiPF6", salt_concentration_sweep("LiPF6"))
    salt_pred = np.asarray([row.predicted_mS_cm for row in salt_rows], dtype=np.float64)
    salt_eta = np.asarray([row.eta_cP for row in salt_rows], dtype=np.float64)
    salt_crowding = np.asarray([row.crowding for row in salt_rows], dtype=np.float64)
    peak_idx = int(np.argmax(salt_pred))
    salt_pass = (
        1 < peak_idx < len(salt_pred) - 2
        and salt_pred[peak_idx] > salt_pred[0]
        and salt_pred[-1] < salt_pred[peak_idx]
        and salt_pred[-1] < salt_pred[3]
        and bool(np.all(np.diff(salt_eta) > 0.0))
        and bool(np.all(np.diff(salt_crowding) > 0.0))
    )

    print("Acceptance summary")
    print(
        "  LiFSI empirical FEC: "
        f"max_abs={lifsi_max_abs:.3f} mS/cm, threshold=0.100, pass={lifsi_pass}"
    )
    print(f"  LiPF6/FEC control contrast: pass={lipf6_control_pass}")
    print(f"  TTFP FR sweep: pass={fr_pass}")
    print(f"  LiPF6 salt dome: peak_idx={peak_idx}, pass={salt_pass}")


def _print_lifsi_calibration_ablation(training: TrainingResult) -> None:
    variants = (
        ("add-support", False),
        ("mixed-ca+add", True),
    )
    print("LiFSI empirical two-effect ablation")
    for label, mixed_ca_active in variants:
        variant_params = _calibration_variant_params(
            params=training.params,
            mixed_ca_active=mixed_ca_active,
        )
        variant_training = replace(training, params=variant_params)
        rows = empirical_lifsi_fec_audit(variant_training)
        observed = np.asarray([row.observed_mS_cm for row in rows], dtype=np.float64)
        predicted = np.asarray([row.predicted_mS_cm for row in rows], dtype=np.float64)
        max_abs = float(np.max(np.abs(predicted - observed)))
        pred_text = ", ".join(f"{value:.3f}" for value in predicted)
        print(f"  {label:<9} max_abs={max_abs:.3f} mS/cm, pred=[{pred_text}]")
    print("")


def _calibration_variant_params(
    params: Mapping[str, jnp.ndarray],
    mixed_ca_active: bool,
) -> Mapping[str, jnp.ndarray]:
    variant = dict(params)
    if not mixed_ca_active:
        variant["mixed_anion_anticorrelation_logit"] = jnp.asarray(-jnp.inf)
    return variant


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_GRAD_CLIP_NORM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-family", choices=("none", "lifsi_fec"), default="none")
    parser.add_argument("--skip-fit-metrics", action="store_true")
    parser.add_argument(
        "--data-sources",
        default=",".join(DEFAULT_DATA_SOURCES),
        help=f"Comma-separated sources from: {','.join(MECHANISTIC_DATA_SOURCES)}",
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    data_sources = _parse_data_sources(args.data_sources)
    training = train_mechanistic_prototype(
        n_steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        holdout_family=args.holdout_family,
        data_sources=data_sources,
        batch_size=args.batch_size,
    )
    print_training_report(training, include_fit_metrics=not args.skip_fit_metrics)
    if args.checkpoint is not None:
        _save_checkpoint(training, args.checkpoint)
        print(f"saved checkpoint: {args.checkpoint}")


def _split_rows(
    rows: Sequence[MechanisticRow],
    holdout_family: str,
) -> tuple[tuple[MechanisticRow, ...], tuple[MechanisticRow, ...]]:
    if holdout_family == "none":
        return tuple(rows), tuple()
    if holdout_family != "lifsi_fec":
        raise ValueError(f"Unsupported holdout_family {holdout_family!r}")
    holdout = lifsi_dominant_fec_rows(rows)
    holdout_keys = {(row.source, row.row_index) for row in holdout}
    train = tuple(row for row in rows if (row.source, row.row_index) not in holdout_keys)
    if not holdout:
        raise ValueError("LiFSI-dominant FEC holdout requested but no rows were found")
    if not train:
        raise ValueError("LiFSI-dominant FEC holdout removed all training rows")
    return train, holdout


def _parse_data_sources(raw_sources: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in raw_sources.split(",") if part.strip())
    if not parsed:
        raise ValueError("--data-sources must contain at least one source")
    for source in parsed:
        if source not in MECHANISTIC_DATA_SOURCES:
            raise ValueError(f"Unsupported --data-sources entry {source!r}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"--data-sources contains duplicate entries: {raw_sources!r}")
    return parsed


def _audit_recipe(
    label: str,
    recipe: Mapping[str, object],
    observed_mS_cm: float | None,
    training: TrainingResult,
) -> AuditRow:
    result = evaluate_mechanistic_recipe(
        recipe=recipe,
        temperature_K=T_REF_K,
        params=training.params,
        norm_mean=training.norm_mean,
        norm_std=training.norm_std,
    )
    diagnostics = _ood_diagnostics(recipe, training)
    features = result.features
    return AuditRow(
        label=label,
        recipe=recipe,
        observed_mS_cm=observed_mS_cm,
        predicted_mS_cm=result.sigma_mS_cm,
        eta_cP=features["eta_solution_cP"],
        self_current_scale_prior_mS_cm=features["self_current_scale_prior_mS_cm"],
        sigma_self_mS_cm=features["sigma_self_mS_cm"],
        cation_anion_distinct_mS_cm=features["cation_anion_distinct_mS_cm"],
        mixed_anion_anticorrelation_mS_cm=features["mixed_anion_anticorrelation_mS_cm"],
        cation_cation_distinct_mS_cm=features["cation_cation_distinct_mS_cm"],
        anion_anion_distinct_mS_cm=features["anion_anion_distinct_mS_cm"],
        cluster_drift_mS_cm=features["cluster_drift_mS_cm"],
        ionic_network_current_mS_cm=features["ionic_network_current_mS_cm"],
        mixed_anion_additive_current_mS_cm=features["mixed_anion_additive_current_mS_cm"],
        relaxation_tail_mS_cm=features["relaxation_tail_mS_cm"],
        distinct_current_correction_mS_cm=features["distinct_current_correction_mS_cm"],
        association_fraction=features["transport_association_fraction"],
        crowding=features["crowding"],
        activity_M=features["effective_ion_concentration_M"],
        mobile_carrier_density_M=features["mobile_carrier_density_M"],
        finite_concentration_mobility_factor=features["finite_concentration_mobility_factor"],
        finite_concentration_correlation_drive=features["finite_concentration_correlation_drive"],
        species_property_distance=diagnostics.species_property_distance,
        loading_distance=diagnostics.loading_distance,
        interaction_distance=diagnostics.interaction_distance,
        nearest_loading_distance=diagnostics.nearest_loading_distance,
        max_loading_z=diagnostics.max_loading_z,
        support_ratio=diagnostics.support_ratio,
        unsupported_region=diagnostics.unsupported_region,
    )


def _ood_diagnostics(
    recipe: Mapping[str, object],
    training: TrainingResult,
) -> AuditDiagnostics:
    inputs = build_unit_aware_recipe_inputs(recipe, training.norm_mean, training.norm_std)
    active_props = inputs.species_props_norm[inputs.mask > 0.0]
    train_props = training.train_batch.species_props_norm[training.train_batch.mask > 0.0]
    diffs = active_props[:, None, :] - train_props[None, :, :]
    distances = np.sqrt(np.mean(diffs * diffs, axis=2))
    nearest_distances = np.min(distances, axis=1)
    result = evaluate_mechanistic_recipe(
        recipe=recipe,
        temperature_K=T_REF_K,
        params=training.params,
        norm_mean=training.norm_mean,
        norm_std=training.norm_std,
    )
    physical = result.feature_vector[: len(PHYSICAL_FEATURE_NAMES)]
    z = (physical - training.physical_mean) / training.physical_std
    interaction_z = z[np.asarray(INTERACTION_DIAGNOSTIC_INDICES, dtype=int)]
    train_diffs = training.train_physical_z - z[None, :]
    nearest_loading_distances = np.sqrt(np.mean(train_diffs * train_diffs, axis=1))
    train_interaction_z = training.train_physical_z[:, np.asarray(INTERACTION_DIAGNOSTIC_INDICES, dtype=int)]
    interaction_diffs = train_interaction_z - interaction_z[None, :]
    nearest_interaction_distances = np.sqrt(np.mean(interaction_diffs * interaction_diffs, axis=1))
    species_ratio = float(np.max(nearest_distances)) / training.species_support_radius
    loading_ratio = float(np.min(nearest_loading_distances)) / training.loading_support_radius
    interaction_ratio = float(np.min(nearest_interaction_distances)) / training.interaction_support_radius
    support_ratio = max(species_ratio, loading_ratio, interaction_ratio)
    return AuditDiagnostics(
        species_property_distance=float(np.max(nearest_distances)),
        loading_distance=float(np.sqrt(np.mean(z * z))),
        interaction_distance=float(np.sqrt(np.mean(interaction_z * interaction_z))),
        nearest_loading_distance=float(np.min(nearest_loading_distances)),
        max_loading_z=float(np.max(np.abs(z))),
        support_ratio=support_ratio,
        unsupported_region=bool(support_ratio > 1.0),
    )


def _print_metrics(name: str, metrics: FitMetrics) -> None:
    print(
        f"{name}: loss={metrics.loss:.6f}, "
        f"MAE={metrics.mae_mS_cm:.3f} mS/cm, "
        f"RMSE={metrics.rmse_mS_cm:.3f} mS/cm, "
        f"MAPE={metrics.mape_percent:.2f}%, "
        f"max_abs={metrics.max_abs_mS_cm:.3f} mS/cm, "
        f"density_MAE={metrics.density_mae_g_ml:.4f} g/mL"
    )


def _print_conductivity_stratum_metrics(
    metrics: Sequence[ConductivityStratumMetrics],
) -> None:
    print("conductivity strata:")
    print(
        "  stratum                            rows labels   MAE   RMSE   MAPE max_abs "
        "T_min T_max sigma_min sigma_max"
    )
    for item in metrics:
        if item.labels == 0:
            print(
                f"  {item.stratum:<34} {item.rows:5d} {item.labels:6d} "
                "    --     --     --      -- "
                f"{item.min_temperature_K:5.2f} {item.max_temperature_K:5.2f}       --       --"
            )
            continue
        print(
            f"  {item.stratum:<34} {item.rows:5d} {item.labels:6d} "
            f"{item.mae_mS_cm:6.3f} "
            f"{item.rmse_mS_cm:6.3f} "
            f"{item.mape_percent:6.2f} "
            f"{item.max_abs_mS_cm:7.3f} "
            f"{item.min_temperature_K:5.2f} "
            f"{item.max_temperature_K:5.2f} "
            f"{item.min_sigma_mS_cm:9.3f} "
            f"{item.max_sigma_mS_cm:9.3f}"
        )


def _print_high_conductivity_rows(
    rows: Sequence[HighConductivityAuditRow],
) -> None:
    print("highest-conductivity labeled rows:")
    if not rows:
        print("  none")
        return
    for row in rows:
        print(
            f"  sigma={row.conductivity_mS_cm:7.3f} "
            f"source={row.source:<18} "
            f"row={row.row_index:<6d} "
            f"T={row.temperature_K:6.2f} "
            f"stratum={row.stratum} "
            f"recipe={row.recipe}"
        )


def _print_audit_table(title: str, rows: Sequence[AuditRow]) -> None:
    print(title)
    print(
        "  label                  obs     pred    eta  prior    self  dCorr     ca  mixCA     cc     aa    "
        "cluster network mixAdd relax assoc crowd actM  mCar  fMob corrD  propOOD loadOOD intOOD nearOOD support unsupported"
    )
    for row in rows:
        observed = "   --"
        if row.observed_mS_cm is not None:
            observed = f"{row.observed_mS_cm:5.2f}"
        print(
            f"  {row.label:<20} {observed} "
            f"{row.predicted_mS_cm:8.3f} "
            f"{row.eta_cP:6.3f} "
            f"{row.self_current_scale_prior_mS_cm:7.3f} "
            f"{row.sigma_self_mS_cm:7.3f} "
            f"{row.distinct_current_correction_mS_cm:7.3f} "
            f"{row.cation_anion_distinct_mS_cm:7.3f} "
            f"{row.mixed_anion_anticorrelation_mS_cm:7.3f} "
            f"{row.cation_cation_distinct_mS_cm:7.3f} "
            f"{row.anion_anion_distinct_mS_cm:7.3f} "
            f"{row.cluster_drift_mS_cm:7.3f} "
            f"{row.ionic_network_current_mS_cm:7.3f} "
            f"{row.mixed_anion_additive_current_mS_cm:7.3f} "
            f"{row.relaxation_tail_mS_cm:7.3f} "
            f"{row.association_fraction:5.3f} "
            f"{row.crowding:5.3f} "
            f"{row.activity_M:5.3f} "
            f"{row.mobile_carrier_density_M:5.3f} "
            f"{row.finite_concentration_mobility_factor:5.3f} "
            f"{row.finite_concentration_correlation_drive:5.3f} "
            f"{row.species_property_distance:7.3f} "
            f"{row.loading_distance:7.3f} "
            f"{row.interaction_distance:7.3f} "
            f"{row.nearest_loading_distance:7.3f} "
            f"{row.support_ratio:7.3f} "
            f"{row.unsupported_region}"
        )
    print("")


def _fec_label(recipe: Mapping[str, object]) -> str:
    additives = require_mapping(recipe, "additives", "recipe")
    loading = 0.0
    if "FEC" in additives:
        loading = float(additives["FEC"])
    return f"FEC={100.0 * loading:.2f}wt%"


def _recipe_loading_label(recipe: Mapping[str, object]) -> str:
    additives = require_mapping(recipe, "additives", "recipe")
    if additives:
        parts = [f"{name}={100.0 * float(value):.2f}wt%" for name, value in sorted(additives.items())]
        return ",".join(parts)
    salts = require_mapping(recipe, "salts", "recipe")
    parts = [f"{name}={float(value):.2f}M" for name, value in sorted(salts.items())]
    return ",".join(parts)


def _fec_single_salt_recipe(salt_name: str, fec_wt_fraction: float) -> Mapping[str, object]:
    additives: dict[str, float] = {}
    if fec_wt_fraction > 0.0:
        additives["FEC"] = fec_wt_fraction
    return {
        "solvents": {"EC": 0.30, "DMC": 0.70},
        "salts": {salt_name: 1.0},
        "additives": additives,
    }


def _additive_recipe(
    salt_name: str,
    additive_name: str,
    additive_wt_fraction: float,
) -> Mapping[str, object]:
    additives: dict[str, float] = {}
    if additive_wt_fraction > 0.0:
        additives[additive_name] = additive_wt_fraction
    return {
        "solvents": {"EC": 0.30, "DMC": 0.70},
        "salts": {salt_name: 1.0},
        "additives": additives,
    }


def _salt_recipe(salt_name: str, concentration_M: float) -> Mapping[str, object]:
    return {
        "solvents": {"EC": 0.30, "DMC": 0.70},
        "salts": {salt_name: concentration_M},
        "additives": {},
    }


def _save_checkpoint(training: TrainingResult, path: Path) -> None:
    serializable = {f"param__{name}": np.asarray(value) for name, value in training.params.items()}
    serializable["norm_mean"] = np.asarray(training.norm_mean)
    serializable["norm_std"] = np.asarray(training.norm_std)
    serializable["physical_mean"] = np.asarray(training.physical_mean)
    serializable["physical_std"] = np.asarray(training.physical_std)
    np.savez(path, **serializable)


def load_mechanistic_checkpoint(path: Path) -> LoadedMechanisticCheckpoint:
    """Load a self-contained prototype checkpoint."""

    with np.load(path) as data:
        required = ("norm_mean", "norm_std", "physical_mean", "physical_std")
        for name in required:
            if name not in data:
                raise ValueError(f"Checkpoint {path} is missing required array {name!r}")
        param_names = sorted(name for name in data.files if name.startswith("param__"))
        if not param_names:
            raise ValueError(f"Checkpoint {path} does not contain any parameter arrays")
        params = {
            name.removeprefix("param__"): jnp.asarray(data[name])
            for name in param_names
        }
        return LoadedMechanisticCheckpoint(
            params=params,
            norm_mean=np.asarray(data["norm_mean"]),
            norm_std=np.asarray(data["norm_std"]),
            physical_mean=np.asarray(data["physical_mean"]),
            physical_std=np.asarray(data["physical_std"]),
        )


if __name__ == "__main__":
    main()
