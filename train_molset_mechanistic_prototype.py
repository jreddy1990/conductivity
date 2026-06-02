"""Train and audit the mechanism-head MolSet conductivity prototype."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import control_framework.jax_m4_tuning  # noqa: F401  must run before jax import
import jax
import jax.numpy as jnp
import numpy as np
import optax

from constants import T_REF_K
from conductivity.mol_set_sigma_mechanistic_prototype import (
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
    sigma_self_mS_cm: float
    cation_anion_distinct_mS_cm: float
    cation_cation_distinct_mS_cm: float
    anion_anion_distinct_mS_cm: float
    cluster_drift_mS_cm: float
    ionic_network_current_mS_cm: float
    mixed_anion_additive_current_mS_cm: float
    relaxation_tail_mS_cm: float
    association_fraction: float
    crowding: float
    activity_M: float
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
    physical_mean, physical_std = compute_physical_feature_stats(train_batch)
    params = init_mechanistic_params(jax.random.PRNGKey(seed), physical_mean, physical_std)
    batch_tuple = batch_tuple_from_mechanistic_batch(train_batch)
    source_index_groups = _source_index_groups(train_batch.sources)
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
                source_index_groups,
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
            source_index_groups,
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
    )


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
    groups = tuple(np.flatnonzero(source_array == source) for source in sorted(set(sources)))
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
        print("")
    else:
        print("fit metrics: skipped by --skip-fit-metrics")
        print("")
    _print_audit_table("Empirical LiFSI-dominant mixed-salt + FEC", empirical_lifsi_fec_audit(training))
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
    lifsi_ca = np.asarray([row.cation_anion_distinct_mS_cm for row in lifsi_rows], dtype=np.float64)
    lifsi_max_abs = float(np.max(np.abs(lifsi_predicted - lifsi_observed)))
    lifsi_pass = (
        lifsi_max_abs <= 0.1
        and lifsi_predicted[1] > lifsi_predicted[0]
        and lifsi_predicted[2] > lifsi_predicted[0]
        and lifsi_predicted[3] > lifsi_predicted[0] - 0.25
        and bool(np.all(np.diff(lifsi_eta) > 0.0))
        and lifsi_ca[-1] > lifsi_ca[0]
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
        sigma_self_mS_cm=features["sigma_self_mS_cm"],
        cation_anion_distinct_mS_cm=features["cation_anion_distinct_mS_cm"],
        cation_cation_distinct_mS_cm=features["cation_cation_distinct_mS_cm"],
        anion_anion_distinct_mS_cm=features["anion_anion_distinct_mS_cm"],
        cluster_drift_mS_cm=features["cluster_drift_mS_cm"],
        ionic_network_current_mS_cm=features["ionic_network_current_mS_cm"],
        mixed_anion_additive_current_mS_cm=features["mixed_anion_additive_current_mS_cm"],
        relaxation_tail_mS_cm=features["relaxation_tail_mS_cm"],
        association_fraction=features["association_fraction"],
        crowding=features["crowding"],
        activity_M=features["effective_ion_concentration_M"],
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


def _print_audit_table(title: str, rows: Sequence[AuditRow]) -> None:
    print(title)
    print(
        "  label                  obs     pred    eta    self     ca     cc     aa    "
        "cluster network mixed relax assoc crowd actM  propOOD loadOOD intOOD nearOOD support unsupported"
    )
    for row in rows:
        observed = "   --"
        if row.observed_mS_cm is not None:
            observed = f"{row.observed_mS_cm:5.2f}"
        print(
            f"  {row.label:<20} {observed} "
            f"{row.predicted_mS_cm:8.3f} "
            f"{row.eta_cP:6.3f} "
            f"{row.sigma_self_mS_cm:7.3f} "
            f"{row.cation_anion_distinct_mS_cm:7.3f} "
            f"{row.cation_cation_distinct_mS_cm:7.3f} "
            f"{row.anion_anion_distinct_mS_cm:7.3f} "
            f"{row.cluster_drift_mS_cm:7.3f} "
            f"{row.ionic_network_current_mS_cm:7.3f} "
            f"{row.mixed_anion_additive_current_mS_cm:7.3f} "
            f"{row.relaxation_tail_mS_cm:7.3f} "
            f"{row.association_fraction:5.3f} "
            f"{row.crowding:5.3f} "
            f"{row.activity_M:5.3f} "
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
