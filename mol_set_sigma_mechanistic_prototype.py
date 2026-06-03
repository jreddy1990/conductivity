"""Trained mechanism-head MolSet prototype for bulk electrolyte conductivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import control_framework.jax_m4_tuning  # noqa: F401  must run before jax import
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from constants import MS_CM_TO_S_M as _MS_CM_TO_S_M
from constants import T_REF_K
from conductivity import mol_set_sigma as _base
from conductivity.mol_set_sigma_unit_aware_prototype import (
    D_INPUT,
    IDX_ANION_R,
    IDX_BINDING,
    IDX_CATION_R,
    IDX_COORD_AFFINITY,
    IDX_DENSITY,
    IDX_DIELECTRIC_DECREMENT,
    IDX_DIMERIZATION,
    IDX_EPSILON,
    IDX_ION_PAIR_KD,
    IDX_JONES_DOLE,
    IDX_LAMBDA0,
    IDX_MW,
    IDX_RESIDENCE_TIME,
    IDX_STERIC,
    IDX_STOKES_ALPHA_ANION,
    IDX_VISCOSITY,
    N_LAYERS,
    N_MAX_SPECIES,
    build_unit_aware_recipe_inputs,
)
from conductivity.molset_mechanistic_data import MechanisticBatch


D_HIDDEN = _base.D_HIDDEN
D_FFN = _base.D_FFN
NUMERICAL_EPS = 1e-12
LITER_TO_ML = 1000.0
ENCODER_AUX_FEATURE_NAMES = (
    "solvent_volume_fraction",
    "salt_molarity",
    "additive_weight_fraction",
    "log1p_solvent_volume_fraction",
    "log1p_salt_molarity",
    "log1p_additive_weight_fraction",
    "is_solvent",
    "is_salt",
    "is_additive",
    "temperature_scaled",
)
ENCODER_AUX_DIM = len(ENCODER_AUX_FEATURE_NAMES)
MECH_HIDDEN = D_HIDDEN
PAIR_HIDDEN = D_HIDDEN
PAIR_INPUT_DIM = 2 * D_HIDDEN + D_INPUT
PAIR_POOL_NAMES = (
    "salt_salt_pair_pool",
    "salt_additive_pair_pool",
    "solvent_salt_pair_pool",
    "solvent_additive_pair_pool",
)
PAIR_POOL_DIM = len(PAIR_POOL_NAMES) * D_HIDDEN
CURRENT_HEAD_NAMES = (
    "density_log_ratio_head",
    "viscosity_log_ratio_head",
    "dielectric_log_ratio_head",
    "association_logit_delta_head",
    "additive_shell_logit_delta_head",
    "cluster_population_logit_delta_head",
    "cluster_persistence_log_ratio_head",
    "cation_self_mobility_gate",
    "anion_self_mobility_gate",
    "cation_anion_gate",
    "cation_cation_gate",
    "anion_anion_gate",
    "cluster_drift_gate",
    "relaxation_tail_gate",
    "ionic_network_gate",
)
NEUTRAL_ADDITIVE_INVARIANT_HEAD_NAMES = (
    "viscosity_log_ratio_head",
    "association_logit_delta_head",
    "cluster_population_logit_delta_head",
    "cluster_persistence_log_ratio_head",
)

PHYSICAL_FEATURE_NAMES = (
    "solvent_volume_fraction_sum",
    "salt_molarity_total_M",
    "additive_weight_fraction_total",
    "neutral_additive_volume_fraction",
    "ionic_additive_molarity_M",
    "solvent_total_volume_ml_per_L",
    "density_prior_g_ml",
    "salt_volume_fraction",
    "eta_liquid_prior_cP",
    "eta_solution_prior_cP",
    "eta_salt_solvent_prior_cP",
    "neutral_additive_viscosity_excess_factor",
    "epsilon_liquid",
    "epsilon_effective",
    "dielectric_decrement_fraction",
    "mean_lambda0_S_cm2_mol",
    "mean_cation_radius_A",
    "mean_anion_radius_A",
    "mean_anion_flex",
    "mean_jones_dole_B",
    "mean_ion_pair_Kd_M",
    "mean_ion_pair_binding_kJ_mol",
    "mean_stokes_alpha_anion",
    "cation_solvation_strength_M",
    "additive_shell_fraction",
    "neutral_shell_persistence_ns",
    "contact_pair_prior",
    "free_ion_prior",
    "ionic_strength_M",
    "ionic_source_diversity",
    "free_solvent_availability",
    "solvent_starvation",
    "cluster_population_prior",
    "cluster_persistence_prior",
    "ionic_network_transport_support",
    "crowding_prior",
    "activity_prior_M",
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
    "salt_additive_dielectric_screening",
    "salt_additive_anticorrelation_screening",
    "mixed_anion_additive_current_support",
    "salt_additive_like_current_support",
    "salt_additive_cluster_transport_support",
    "additive_transport_drag",
    "temperature_scaled",
)
SELF_MOBILITY_PHYSICAL_FEATURE_NAMES = (
    "salt_molarity_total_M",
    "mean_lambda0_S_cm2_mol",
    "mean_cation_radius_A",
    "mean_anion_radius_A",
    "mean_anion_flex",
    "mean_jones_dole_B",
    "mean_ion_pair_Kd_M",
    "mean_ion_pair_binding_kJ_mol",
    "mean_stokes_alpha_anion",
    "ionic_strength_M",
    "ionic_source_diversity",
    "temperature_scaled",
)
SELF_MOBILITY_PHYSICAL_FEATURE_MASK = np.asarray(
    [name in SELF_MOBILITY_PHYSICAL_FEATURE_NAMES for name in PHYSICAL_FEATURE_NAMES],
    dtype=np.float64,
)

MECHANISM_FEATURE_NAMES = (
    "density_pred_g_ml",
    "eta_solution_cP",
    "eta_supervised_fit_cP",
    "epsilon_effective_pred",
    "effective_ion_concentration_M",
    "association_fraction",
    "free_ion_fraction",
    "additive_shell_participation",
    "crowding",
    "cluster_population",
    "cluster_persistence",
    "additive_transport_drag",
    "temperature_viscosity_factor",
    "additive_anticorrelation_screening_support",
    "transport_association_fraction",
    "transport_free_ion_fraction",
    "mobile_carrier_density_M",
    "cation_viscosity_friction_factor",
    "anion_viscosity_friction_factor",
    "free_solvent_mobility_factor",
    "finite_concentration_mobility_factor",
    "finite_concentration_correlation_drive",
    "anticorrelation_screening",
    "like_current_support",
    "cluster_transport_support",
    "positive_current_support_fraction",
    "self_current_scale_prior_mS_cm",
    "cation_self_mobility_gate_factor",
    "anion_self_mobility_gate_factor",
    "cation_self_current_mS_cm",
    "anion_self_current_mS_cm",
    "sigma_self_mS_cm",
    "cation_anion_distinct_fraction",
    "cation_anion_distinct_mS_cm",
    "mixed_anion_anticorrelation_mS_cm",
    "cation_cation_distinct_fraction",
    "cation_cation_distinct_mS_cm",
    "anion_anion_distinct_fraction",
    "anion_anion_distinct_mS_cm",
    "cluster_drift_fraction",
    "cluster_drift_mS_cm",
    "ionic_network_current_mS_cm",
    "mixed_anion_additive_current_mS_cm",
    "relaxation_tail_fraction",
    "relaxation_tail_mS_cm",
    "distinct_current_correction_mS_cm",
    "current_correlation_integral_mS_cm",
    "sigma_mS_cm",
)

MODEL_FEATURE_NAMES = PHYSICAL_FEATURE_NAMES + MECHANISM_FEATURE_NAMES
READOUT_INPUT_DIM = 3 * D_HIDDEN + PAIR_POOL_DIM + len(PHYSICAL_FEATURE_NAMES)
N_HEAD_OUTPUTS = len(CURRENT_HEAD_NAMES)

FEATURE_IDX_DENSITY = len(PHYSICAL_FEATURE_NAMES)
FEATURE_IDX_SIGMA = len(MODEL_FEATURE_NAMES) - 1


@dataclass(frozen=True)
class MechanisticMolSetResult:
    """Prediction and auditable mechanism features for one recipe."""

    sigma_mS_cm: float
    sigma_s_m: float
    features: dict[str, float]
    feature_vector: np.ndarray


def model_feature_names() -> tuple[str, ...]:
    """Return feature order for mechanism audit vectors."""

    return MODEL_FEATURE_NAMES


def init_mechanistic_params(
    key: jax.Array,
    physical_mean: np.ndarray,
    physical_std: np.ndarray,
) -> dict[str, jnp.ndarray]:
    """Initialize a trainable mechanism-head MolSet."""

    _validate_physical_stats(physical_mean, physical_std)
    params: dict[str, jnp.ndarray] = {}

    def linear_init(rng: jax.Array, d_in: int, d_out: int, name: str) -> None:
        k1, _k2 = random.split(rng)
        scale = jnp.sqrt(2.0 / float(d_in))
        params[f"{name}_w"] = random.normal(k1, (d_in, d_out)) * scale
        params[f"{name}_b"] = jnp.zeros(d_out)

    n_keys = 1 + N_LAYERS * 6 + 5
    keys = random.split(key, n_keys)
    key_idx = 0

    linear_init(keys[key_idx], D_INPUT + ENCODER_AUX_DIM, D_HIDDEN, "enc")
    key_idx += 1
    for layer in range(N_LAYERS):
        linear_init(keys[key_idx], D_HIDDEN, D_HIDDEN, f"attn{layer}_q")
        key_idx += 1
        linear_init(keys[key_idx], D_HIDDEN, D_HIDDEN, f"attn{layer}_k")
        key_idx += 1
        linear_init(keys[key_idx], D_HIDDEN, D_HIDDEN, f"attn{layer}_v")
        key_idx += 1
        linear_init(keys[key_idx], D_HIDDEN, D_HIDDEN, f"attn{layer}_out")
        key_idx += 1
        params[f"ln{layer}_attn_scale"] = jnp.ones(D_HIDDEN)
        params[f"ln{layer}_attn_bias"] = jnp.zeros(D_HIDDEN)
        linear_init(keys[key_idx], D_HIDDEN, D_FFN, f"ffn{layer}_1")
        key_idx += 1
        linear_init(keys[key_idx], D_FFN, D_HIDDEN, f"ffn{layer}_2")
        key_idx += 1
        params[f"ln{layer}_ffn_scale"] = jnp.ones(D_HIDDEN)
        params[f"ln{layer}_ffn_bias"] = jnp.zeros(D_HIDDEN)

    linear_init(keys[key_idx], PAIR_INPUT_DIM, PAIR_HIDDEN, "pair_h")
    key_idx += 1
    linear_init(keys[key_idx], PAIR_HIDDEN, D_HIDDEN, "pair_out")
    key_idx += 1
    linear_init(keys[key_idx], READOUT_INPUT_DIM, MECH_HIDDEN, "mech_h1")
    key_idx += 1
    linear_init(keys[key_idx], MECH_HIDDEN, MECH_HIDDEN, "mech_h2")
    key_idx += 1
    linear_init(keys[key_idx], MECH_HIDDEN, N_HEAD_OUTPUTS, "mech_out")
    params["mech_out_w"] = jnp.zeros_like(params["mech_out_w"])
    params["mech_out_b"] = jnp.zeros_like(params["mech_out_b"])

    params["physical_mean"] = jnp.asarray(physical_mean)
    params["physical_std"] = jnp.asarray(physical_std)
    params["generic_positive_current_scale"] = jnp.asarray(1.0)
    params["mixed_anion_anticorrelation_logit"] = jnp.asarray(0.0)
    params["temperature_transport_activation_K"] = jnp.asarray(0.0)
    return params


def compute_physical_feature_stats(batch: MechanisticBatch) -> tuple[np.ndarray, np.ndarray]:
    """Compute training-set normalization for physical-effect features."""

    features_np = compute_physical_features_for_batch(batch)
    mean = features_np.mean(axis=0)
    std = features_np.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def compute_physical_features_for_batch(batch: MechanisticBatch) -> np.ndarray:
    """Compute deterministic physical-effect features without neural readout."""

    features = _physical_features_batch(
        raw_props=jnp.asarray(batch.species_props_raw),
        solvent_volume_fraction=jnp.asarray(batch.solvent_volume_fraction),
        salt_molarity=jnp.asarray(batch.salt_molarity),
        additive_weight_fraction=jnp.asarray(batch.additive_weight_fraction),
        mask=jnp.asarray(batch.mask),
        temperature_K=jnp.asarray(batch.temperature_K),
    )
    return np.asarray(features)


def evaluate_mechanistic_recipe(
    recipe: Mapping[str, object],
    temperature_K: float,
    params: Mapping[str, jnp.ndarray],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> MechanisticMolSetResult:
    """Evaluate one recipe with the trained mechanistic prototype."""

    inputs = build_unit_aware_recipe_inputs(recipe, norm_mean, norm_std)
    log_sigma, feature_vector = forward_single(
        params=params,
        species_props_norm=jnp.asarray(inputs.species_props_norm),
        species_props_raw=jnp.asarray(inputs.species_props_raw),
        solvent_volume_fraction=jnp.asarray(inputs.solvent_volume_fraction),
        salt_molarity=jnp.asarray(inputs.salt_molarity),
        additive_weight_fraction=jnp.asarray(inputs.additive_weight_fraction),
        mask=jnp.asarray(inputs.mask),
        temperature_K=jnp.asarray(temperature_K),
    )
    sigma_mS_cm = float(jnp.exp(log_sigma))
    features = {
        name: float(value)
        for name, value in zip(MODEL_FEATURE_NAMES, np.asarray(feature_vector), strict=True)
    }
    return MechanisticMolSetResult(
        sigma_mS_cm=sigma_mS_cm,
        sigma_s_m=sigma_mS_cm * _MS_CM_TO_S_M,
        features=features,
        feature_vector=np.asarray(feature_vector),
    )


def forward_single(
    params: Mapping[str, jnp.ndarray],
    species_props_norm: jnp.ndarray,
    species_props_raw: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass for one recipe."""

    z = _encode_species_set(
        params=params,
        species_props_norm=species_props_norm,
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=additive_weight_fraction,
        mask=mask,
        temperature_K=temperature_K,
    )
    physical = _compute_physical_features(
        raw_props=species_props_raw,
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=additive_weight_fraction,
        mask=mask,
        temperature_K=temperature_K,
    )
    physical_norm = (physical - params["physical_mean"]) / params["physical_std"]
    solvent_pool = _weighted_pool(z, solvent_volume_fraction * mask)
    salt_pool = _weighted_pool(z, salt_molarity * mask)
    additive_pool = _weighted_pool(z, additive_weight_fraction * mask)
    pair_pools = _pairwise_role_pair_pools(
        params=params,
        z=z,
        species_props_norm=species_props_norm,
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=additive_weight_fraction,
        mask=mask,
    )
    readout_input = jnp.concatenate(
        [solvent_pool, salt_pool, additive_pool, pair_pools, physical_norm],
        axis=0,
    )
    hidden1 = jax.nn.gelu(readout_input @ params["mech_h1_w"] + params["mech_h1_b"])
    hidden2 = jax.nn.gelu(hidden1 @ params["mech_h2_w"] + params["mech_h2_b"])
    full_head = hidden2 @ params["mech_out_w"] + params["mech_out_b"]
    self_additive_weight_fraction = _ionic_additive_weight_for_self(
        species_props_raw=species_props_raw,
        additive_weight_fraction=additive_weight_fraction,
        mask=mask,
    )
    self_mask = mask * jnp.where(
        (solvent_volume_fraction > 0.0)
        | (salt_molarity > 0.0)
        | (self_additive_weight_fraction > 0.0),
        1.0,
        0.0,
    )
    z_self = _encode_species_set(
        params=params,
        species_props_norm=species_props_norm,
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=self_additive_weight_fraction,
        mask=self_mask,
        temperature_K=temperature_K,
    )
    self_solvent_pool = _weighted_pool(z_self, solvent_volume_fraction * self_mask)
    self_salt_pool = _weighted_pool(z_self, salt_molarity * self_mask)
    self_pair_pools_full = _pairwise_role_pair_pools(
        params=params,
        z=z_self,
        species_props_norm=species_props_norm,
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=self_additive_weight_fraction,
        mask=self_mask,
    )
    self_physical_norm = physical_norm * jnp.asarray(SELF_MOBILITY_PHYSICAL_FEATURE_MASK)
    self_pair_pools = _self_mobility_pair_pools(self_pair_pools_full)
    self_readout_input = jnp.concatenate(
        [
            self_solvent_pool,
            self_salt_pool,
            jnp.zeros_like(additive_pool),
            self_pair_pools,
            self_physical_norm,
        ],
        axis=0,
    )
    self_hidden1 = jax.nn.gelu(self_readout_input @ params["mech_h1_w"] + params["mech_h1_b"])
    self_hidden2 = jax.nn.gelu(self_hidden1 @ params["mech_h2_w"] + params["mech_h2_b"])
    self_head = self_hidden2 @ params["mech_out_w"] + params["mech_out_b"]
    head = full_head.at[CURRENT_HEAD_NAMES.index("cation_self_mobility_gate")].set(
        _head_value(self_head, "cation_self_mobility_gate")
    )
    head = head.at[CURRENT_HEAD_NAMES.index("anion_self_mobility_gate")].set(
        _head_value(self_head, "anion_self_mobility_gate")
    )
    for head_name in NEUTRAL_ADDITIVE_INVARIANT_HEAD_NAMES:
        head = head.at[CURRENT_HEAD_NAMES.index(head_name)].set(
            _head_value(self_head, head_name)
        )
    mechanism = _mechanism_readout(
        head,
        physical,
        params["generic_positive_current_scale"],
        params["mixed_anion_anticorrelation_logit"],
        params["temperature_transport_activation_K"],
    )
    feature_vector = jnp.concatenate([physical, mechanism], axis=0)
    sigma_mS_cm = mechanism[FEATURE_IDX_SIGMA - len(PHYSICAL_FEATURE_NAMES)]
    return jnp.log(sigma_mS_cm), feature_vector


def forward_batch(
    params: Mapping[str, jnp.ndarray],
    species_props_norm: jnp.ndarray,
    species_props_raw: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized forward pass."""

    return jax.vmap(forward_single, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
        params,
        species_props_norm,
        species_props_raw,
        solvent_volume_fraction,
        salt_molarity,
        additive_weight_fraction,
        mask,
        temperature_K,
    )


def loss_fn(
    params: Mapping[str, jnp.ndarray],
    batch_tuple: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
) -> jnp.ndarray:
    """Conductivity-label training loss for the mechanism-current readout."""

    (
        species_props_norm,
        species_props_raw,
        solvent_volume_fraction,
        salt_molarity,
        additive_weight_fraction,
        mask,
        temperature_K,
        log_sigma_target,
        conductivity_mask,
        weights,
        density_target,
        density_mask,
        viscosity_target,
        viscosity_mask,
        dielectric_target,
        dielectric_mask,
        cation_self_current_target,
        cation_self_current_mask,
        anion_self_current_target,
        anion_self_current_mask,
        cation_anion_distinct_target,
        cation_anion_distinct_mask,
        current_distinct_target,
        current_distinct_mask,
        association_fraction_target,
        association_fraction_mask,
    ) = batch_tuple
    _pred_log_sigma, features = forward_batch(
        params=params,
        species_props_norm=species_props_norm,
        species_props_raw=species_props_raw,
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=additive_weight_fraction,
        mask=mask,
        temperature_K=temperature_K,
    )
    density_pred = features[:, FEATURE_IDX_DENSITY]
    eta_prior = features[:, PHYSICAL_FEATURE_NAMES.index("eta_solution_prior_cP")]
    activity_prior = features[:, PHYSICAL_FEATURE_NAMES.index("activity_prior_M")]
    ionic_row_mask = jnp.where(
        features[:, PHYSICAL_FEATURE_NAMES.index("ionic_strength_M")] > 0.0,
        1.0,
        0.0,
    )
    contact_pair_prior = features[:, PHYSICAL_FEATURE_NAMES.index("contact_pair_prior")]
    cluster_population_prior = features[:, PHYSICAL_FEATURE_NAMES.index("cluster_population_prior")]
    cluster_population_prior_fraction = cluster_population_prior / (1.0 + cluster_population_prior)
    cluster_persistence_prior = features[:, PHYSICAL_FEATURE_NAMES.index("cluster_persistence_prior")]
    viscosity_pred = features[:, _model_feature_idx("eta_supervised_fit_cP")]
    temperature_viscosity_factor_pred = features[
        :, _model_feature_idx("temperature_viscosity_factor")
    ]
    dielectric_pred = features[:, _model_feature_idx("epsilon_effective_pred")]
    cation_self_pred = features[:, _model_feature_idx("cation_self_current_mS_cm")]
    anion_self_pred = features[:, _model_feature_idx("anion_self_current_mS_cm")]
    sigma_self_pred = features[:, _model_feature_idx("sigma_self_mS_cm")]
    cation_self_gate_pred = features[:, _model_feature_idx("cation_self_mobility_gate_factor")]
    anion_self_gate_pred = features[:, _model_feature_idx("anion_self_mobility_gate_factor")]
    cation_anion_distinct_pred = features[:, _model_feature_idx("cation_anion_distinct_mS_cm")]
    current_integral_pred = features[:, _model_feature_idx("current_correlation_integral_mS_cm")]
    current_distinct_pred = features[:, _model_feature_idx("distinct_current_correction_mS_cm")]
    effective_ion_concentration_pred = features[:, _model_feature_idx("effective_ion_concentration_M")]
    association_fraction_pred = features[:, _model_feature_idx("association_fraction")]
    transport_association_fraction_pred = features[:, _model_feature_idx("transport_association_fraction")]
    cluster_population_pred = features[:, _model_feature_idx("cluster_population")]
    cluster_persistence_pred = features[:, _model_feature_idx("cluster_persistence")]
    sigma_target = jnp.exp(log_sigma_target)
    cation_self_gate_base = cation_self_pred / (cation_self_gate_pred + NUMERICAL_EPS)
    anion_self_gate_base = anion_self_pred / (anion_self_gate_pred + NUMERICAL_EPS)
    cation_self_gate_only_pred = (
        jax.lax.stop_gradient(cation_self_gate_base) * cation_self_gate_pred
    )
    anion_self_gate_only_pred = (
        jax.lax.stop_gradient(anion_self_gate_base) * anion_self_gate_pred
    )
    scalar_only_self_pred = (
        cation_self_gate_only_pred
        + anion_self_gate_only_pred
    )
    scalar_sigma_pred = jnp.where(
        current_distinct_mask > 0.0,
        current_integral_pred,
        scalar_only_self_pred + jax.lax.stop_gradient(current_distinct_pred),
    )
    routed_log_sigma = jnp.log(scalar_sigma_pred + NUMERICAL_EPS)
    sigma_target_log = jnp.log(sigma_target + NUMERICAL_EPS)
    routed_sigma_error = (routed_log_sigma - sigma_target_log) ** 2
    weighted_sigma_mask = conductivity_mask * weights
    weighted_log_sigma_loss = weighted_sigma_mask * routed_sigma_error
    log_sigma_loss = jnp.sum(weighted_log_sigma_loss) / (
        jnp.sum(weighted_sigma_mask) + NUMERICAL_EPS
    )
    sigma_current_scale_loss = _masked_current_scale_loss(
        scalar_sigma_pred,
        sigma_target,
        sigma_target,
        conductivity_mask,
        weights,
    )
    decomposed_sigma_mask = conductivity_mask * current_distinct_mask
    decomposed_sigma_identity_loss = _masked_squared_current_loss(
        current_integral_pred,
        sigma_target,
        decomposed_sigma_mask,
        weights,
    )
    sigma_loss = log_sigma_loss + sigma_current_scale_loss + decomposed_sigma_identity_loss
    density_loss = _masked_log_loss(density_pred, density_target, density_mask, weights)
    viscosity_loss = _masked_log_loss(viscosity_pred, viscosity_target, viscosity_mask, weights)
    viscosity_prior_loss = _masked_log_loss(
        viscosity_pred,
        eta_prior * temperature_viscosity_factor_pred,
        1.0 - viscosity_mask,
        weights,
    )
    activity_prior_loss = _masked_log_loss(
        effective_ion_concentration_pred,
        activity_prior,
        ionic_row_mask,
        weights,
    )
    association_prior_loss = _masked_fraction_loss(
        association_fraction_pred,
        contact_pair_prior,
        (1.0 - association_fraction_mask) * ionic_row_mask,
        weights,
    )
    cluster_population_prior_loss = _masked_fraction_loss(
        cluster_population_pred,
        cluster_population_prior_fraction,
        ionic_row_mask,
        weights,
    )
    cluster_persistence_prior_loss = _masked_log_loss(
        cluster_persistence_pred,
        cluster_persistence_prior,
        ionic_row_mask,
        weights,
    )
    dielectric_loss = _masked_log_loss(dielectric_pred, dielectric_target, dielectric_mask, weights)
    cation_self_loss_pred = jnp.where(
        association_fraction_mask > 0.0,
        cation_self_pred,
        cation_self_gate_only_pred,
    )
    anion_self_loss_pred = jnp.where(
        association_fraction_mask > 0.0,
        anion_self_pred,
        anion_self_gate_only_pred,
    )
    cation_self_loss = _masked_log_loss(
        cation_self_loss_pred,
        cation_self_current_target,
        cation_self_current_mask,
        weights,
    )
    anion_self_loss = _masked_log_loss(
        anion_self_loss_pred,
        anion_self_current_target,
        anion_self_current_mask,
        weights,
    )
    cation_anion_distinct_loss = _masked_relative_loss(
        cation_anion_distinct_pred,
        cation_anion_distinct_target,
        cation_anion_distinct_mask,
        weights,
    )
    self_current_scale = cation_self_current_target + anion_self_current_target
    current_distinct_scale = jnp.where(
        (cation_self_current_mask > 0.0) & (anion_self_current_mask > 0.0),
        self_current_scale,
        current_distinct_target,
    )
    current_distinct_loss = _masked_current_scale_loss(
        current_distinct_pred,
        current_distinct_target,
        current_distinct_scale,
        current_distinct_mask,
        weights,
    )
    association_fraction_loss = _masked_fraction_loss(
        transport_association_fraction_pred,
        association_fraction_target,
        association_fraction_mask,
        weights,
    )
    return (
        sigma_loss
        + density_loss
        + viscosity_loss
        + viscosity_prior_loss
        + activity_prior_loss
        + association_prior_loss
        + cluster_population_prior_loss
        + cluster_persistence_prior_loss
        + dielectric_loss
        + cation_self_loss
        + anion_self_loss
        + cation_anion_distinct_loss
        + current_distinct_loss
        + association_fraction_loss
        + _physical_consistency_loss(features)
    )


def batch_tuple_from_mechanistic_batch(
    batch: MechanisticBatch,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Convert a dataclass batch into JAX arrays for training."""

    return (
        jnp.asarray(batch.species_props_norm),
        jnp.asarray(batch.species_props_raw),
        jnp.asarray(batch.solvent_volume_fraction),
        jnp.asarray(batch.salt_molarity),
        jnp.asarray(batch.additive_weight_fraction),
        jnp.asarray(batch.mask),
        jnp.asarray(batch.temperature_K),
        jnp.asarray(batch.log_sigma),
        jnp.asarray(batch.conductivity_mask),
        jnp.asarray(batch.weights),
        jnp.asarray(batch.density_g_ml),
        jnp.asarray(batch.density_mask),
        jnp.asarray(batch.viscosity_cP),
        jnp.asarray(batch.viscosity_mask),
        jnp.asarray(batch.dielectric),
        jnp.asarray(batch.dielectric_mask),
        jnp.asarray(batch.cation_self_current_mS_cm),
        jnp.asarray(batch.cation_self_current_mask),
        jnp.asarray(batch.anion_self_current_mS_cm),
        jnp.asarray(batch.anion_self_current_mask),
        jnp.asarray(batch.cation_anion_distinct_mS_cm),
        jnp.asarray(batch.cation_anion_distinct_mask),
        jnp.asarray(batch.current_distinct_mS_cm),
        jnp.asarray(batch.current_distinct_mask),
        jnp.asarray(batch.association_fraction),
        jnp.asarray(batch.association_fraction_mask),
    )


def _encode_species_set(
    params: Mapping[str, jnp.ndarray],
    species_props_norm: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> jnp.ndarray:
    aux = _role_loading_aux(
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=additive_weight_fraction,
        mask=mask,
        temperature_K=temperature_K,
    )
    return _encode_species_set_from_aux(
        params=params,
        species_props_norm=species_props_norm,
        aux=aux,
        mask=mask,
    )


def _role_loading_aux(
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> jnp.ndarray:
    temperature_scaled = temperature_K / T_REF_K
    solvent_mask = jnp.where(solvent_volume_fraction > 0.0, 1.0, 0.0)
    salt_mask = jnp.where(salt_molarity > 0.0, 1.0, 0.0)
    additive_mask = jnp.where(additive_weight_fraction > 0.0, 1.0, 0.0)
    return jnp.stack(
        [
            solvent_volume_fraction,
            salt_molarity,
            additive_weight_fraction,
            jnp.log1p(solvent_volume_fraction),
            jnp.log1p(salt_molarity),
            jnp.log1p(additive_weight_fraction),
            solvent_mask,
            salt_mask,
            additive_mask,
            jnp.full_like(mask, temperature_scaled),
        ],
        axis=1,
    )


def _encode_species_set_from_aux(
    params: Mapping[str, jnp.ndarray],
    species_props_norm: jnp.ndarray,
    aux: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    z = jax.nn.gelu(
        jnp.concatenate([species_props_norm, aux], axis=1) @ params["enc_w"] + params["enc_b"]
    ) * mask[:, None]
    prop_bias = _property_similarity_bias(species_props_norm, mask)
    for layer in range(N_LAYERS):
        q = z @ params[f"attn{layer}_q_w"] + params[f"attn{layer}_q_b"]
        k = z @ params[f"attn{layer}_k_w"] + params[f"attn{layer}_k_b"]
        v = z @ params[f"attn{layer}_v_w"] + params[f"attn{layer}_v_b"]
        attn_out = _stable_multihead_attention(q, k, v, mask, prop_bias)
        attn_out = attn_out @ params[f"attn{layer}_out_w"] + params[f"attn{layer}_out_b"]
        z = _base._layer_norm(
            z + attn_out * mask[:, None],
            params[f"ln{layer}_attn_scale"],
            params[f"ln{layer}_attn_bias"],
        ) * mask[:, None]
        ffn = jax.nn.gelu(z @ params[f"ffn{layer}_1_w"] + params[f"ffn{layer}_1_b"])
        ffn = ffn @ params[f"ffn{layer}_2_w"] + params[f"ffn{layer}_2_b"]
        z = _base._layer_norm(
            z + ffn * mask[:, None],
            params[f"ln{layer}_ffn_scale"],
            params[f"ln{layer}_ffn_bias"],
        ) * mask[:, None]
    return z


def _physical_features_batch(
    raw_props: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(_compute_physical_features, in_axes=(0, 0, 0, 0, 0, 0))(
        raw_props,
        solvent_volume_fraction,
        salt_molarity,
        additive_weight_fraction,
        mask,
        temperature_K,
    )


def _compute_physical_features(
    raw_props: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> jnp.ndarray:
    active = mask > 0.0
    mw = jnp.where(active, raw_props[:, IDX_MW], 1.0)
    density = jnp.where(active, raw_props[:, IDX_DENSITY], 1.0)
    epsilon = jnp.where(active, raw_props[:, IDX_EPSILON], 1.0)
    viscosity = jnp.where(active, raw_props[:, IDX_VISCOSITY], 1.0)
    lambda0 = jnp.where(active, raw_props[:, IDX_LAMBDA0], 0.0)
    anion_radius = jnp.where(active, raw_props[:, IDX_ANION_R], 0.0)
    cation_radius = jnp.where(active, raw_props[:, IDX_CATION_R], 0.0)
    binding = jnp.where(active, raw_props[:, IDX_BINDING], 0.0)
    coord_affinity = _positive_part(jnp.where(active, raw_props[:, IDX_COORD_AFFINITY], 0.0))
    jones_dole = jnp.where(active, raw_props[:, IDX_JONES_DOLE], 0.0)
    ion_pair_kd = jnp.where(active, raw_props[:, IDX_ION_PAIR_KD], 0.0)
    steric = jnp.where(active, raw_props[:, IDX_STERIC], 0.0)
    dimerization = _positive_part(jnp.where(active, raw_props[:, IDX_DIMERIZATION], 0.0))
    dielectric_decrement = jnp.where(active, raw_props[:, IDX_DIELECTRIC_DECREMENT], 0.0)
    stokes_alpha = jnp.where(active, raw_props[:, IDX_STOKES_ALPHA_ANION], 0.0)
    residence_time = jnp.where(active, raw_props[:, IDX_RESIDENCE_TIME], 0.0)

    solvent_sum = jnp.sum(solvent_volume_fraction)
    solvent_phi = solvent_volume_fraction / (solvent_sum + NUMERICAL_EPS)
    additive_total_wt = jnp.sum(additive_weight_fraction)
    non_additive_mass_fraction = 1.0 - additive_total_wt
    salt_mass_g = salt_molarity * mw
    salt_volume_ml = salt_mass_g / density
    total_salt_mass_g = jnp.sum(salt_mass_g)
    solvent_volumes_ml = solvent_phi * LITER_TO_ML
    solvent_total_volume_ml = jnp.sum(solvent_volumes_ml)
    solvent_masses_g = solvent_volumes_ml * density
    total_solvent_mass_g = jnp.sum(solvent_masses_g)
    total_non_additive_mass_g = total_solvent_mass_g + total_salt_mass_g
    total_mass_g = total_non_additive_mass_g / (non_additive_mass_fraction + NUMERICAL_EPS)

    additive_masses_g = additive_weight_fraction * total_mass_g
    additive_volumes_ml = additive_masses_g / density
    apparent_component_volume_ml = LITER_TO_ML + jnp.sum(salt_volume_ml) + jnp.sum(additive_volumes_ml)
    density_prior = total_mass_g / (apparent_component_volume_ml + NUMERICAL_EPS)
    salt_volume_fraction = jnp.sum(salt_volume_ml) / (apparent_component_volume_ml + NUMERICAL_EPS)
    additive_moles = additive_masses_g / (mw + NUMERICAL_EPS)
    ionic_additive_mask = jnp.where((additive_weight_fraction > 0.0) & (lambda0 > 0.0), 1.0, 0.0)
    neutral_additive_mask = jnp.where((additive_weight_fraction > 0.0) & (lambda0 <= 0.0), 1.0, 0.0)
    ionic_additive_molarity = additive_moles * ionic_additive_mask
    ionic_source_molarity = salt_molarity + ionic_additive_molarity
    total_ionic_source_M = jnp.sum(ionic_source_molarity)

    neutral_volumes_ml = solvent_volumes_ml + additive_volumes_ml * neutral_additive_mask
    neutral_volume_total_ml = jnp.sum(neutral_volumes_ml) + NUMERICAL_EPS
    neutral_phi = neutral_volumes_ml / neutral_volume_total_ml
    neutral_additive_volume_fraction = (
        jnp.sum(additive_volumes_ml * neutral_additive_mask) / neutral_volume_total_ml
    )
    eta_liquid = jnp.exp(jnp.sum(neutral_phi * jnp.log(viscosity + NUMERICAL_EPS)))
    solvent_volume_total_ml = jnp.sum(solvent_volumes_ml) + NUMERICAL_EPS
    solvent_only_phi = solvent_volumes_ml / solvent_volume_total_ml
    eta_salt_solvent_liquid = jnp.exp(
        jnp.sum(solvent_only_phi * jnp.log(viscosity + NUMERICAL_EPS))
    )
    neutral_moles = solvent_volumes_ml * density / (mw + NUMERICAL_EPS) + additive_moles * neutral_additive_mask
    solvent_only_moles = solvent_volumes_ml * density / (mw + NUMERICAL_EPS)
    dimer_fraction = dimerization * neutral_moles / (1.0 + dimerization * neutral_moles)
    solvent_only_dimer_fraction = dimerization * solvent_only_moles / (
        1.0 + dimerization * solvent_only_moles
    )
    shell_strength = coord_affinity * neutral_moles
    shell_strength_sum = jnp.sum(shell_strength) + NUMERICAL_EPS
    shell_fraction = shell_strength / shell_strength_sum
    shell_steric = jnp.sum(shell_fraction * steric)
    shell_dimer_fraction = jnp.sum(shell_fraction * dimer_fraction)
    solvent_shell_strength = coord_affinity * solvent_only_moles
    solvent_shell_strength_sum = jnp.sum(solvent_shell_strength) + NUMERICAL_EPS
    solvent_shell_fraction = solvent_shell_strength / solvent_shell_strength_sum
    salt_solvent_shell_dimer_fraction = jnp.sum(
        solvent_shell_fraction * solvent_only_dimer_fraction
    )
    additive_shell_fraction = jnp.sum(shell_strength * neutral_additive_mask) / shell_strength_sum
    neutral_shell_persistence = jnp.sum(shell_fraction * residence_time)

    source_weight_sum = total_ionic_source_M + NUMERICAL_EPS
    source_weight = ionic_source_molarity / source_weight_sum
    ionic_source_diversity = 1.0 - jnp.sum(source_weight * source_weight)
    mean_lambda0 = jnp.sum(source_weight * lambda0)
    mean_cation_radius = jnp.sum(source_weight * cation_radius)
    mean_anion_radius = jnp.sum(source_weight * anion_radius)
    mean_anion_flex = _anion_flexibility(mean_anion_radius)
    mean_jones_dole = jnp.sum(source_weight * jones_dole)
    mean_ion_pair_kd = jnp.sum(source_weight * ion_pair_kd)
    mean_binding = jnp.sum(source_weight * binding)
    mean_stokes_alpha = jnp.sum(source_weight * stokes_alpha)
    mean_dielectric_decrement = jnp.sum(ionic_source_molarity * dielectric_decrement)
    salt_viscosity_factor = jnp.exp(mean_jones_dole * total_ionic_source_M)
    eta_solution_prior = eta_liquid * salt_viscosity_factor * (1.0 + shell_dimer_fraction)
    eta_salt_solvent_prior = (
        eta_salt_solvent_liquid
        * salt_viscosity_factor
        * (1.0 + salt_solvent_shell_dimer_fraction)
    )
    neutral_additive_viscosity_excess_factor = eta_solution_prior / (
        eta_salt_solvent_prior + NUMERICAL_EPS
    )
    epsilon_liquid = jnp.sum(neutral_phi * epsilon)
    epsilon_effective = epsilon_liquid / (1.0 + mean_dielectric_decrement)

    salt_total_M = jnp.sum(salt_molarity)
    salt_weight = salt_molarity / (salt_total_M + NUMERICAL_EPS)
    salt_pair_molarity = (salt_total_M * salt_total_M - jnp.sum(salt_molarity * salt_molarity)) / 2.0
    salt_pair_lambda_contrast = _weighted_std(lambda0, salt_weight)
    salt_pair_anion_size_contrast = _weighted_std(anion_radius, salt_weight)
    salt_pair_binding_contrast = _weighted_std(binding, salt_weight)
    salt_pair_kd_contrast = _weighted_std(ion_pair_kd, salt_weight)
    salt_pair_dielectric_decrement_contrast = _weighted_std(dielectric_decrement, salt_weight)
    salt_pair_stokes_contrast = _weighted_std(stokes_alpha, salt_weight)
    salt_pair_flex_contrast = _weighted_std(_anion_flexibility(anion_radius), salt_weight)

    association_drive = (
        total_ionic_source_M
        * (1.0 + mean_binding / (jnp.abs(mean_binding) + 1.0))
        * (epsilon_liquid / (epsilon_liquid + epsilon_effective + NUMERICAL_EPS))
        / (1.0 + mean_ion_pair_kd)
    )
    contact_pair_prior = association_drive / (1.0 + association_drive)
    free_ion_prior = 1.0 / (1.0 + association_drive)
    free_solvent_availability = (solvent_total_volume_ml / LITER_TO_ML) / (1.0 + total_ionic_source_M)
    solvent_starvation = total_ionic_source_M / (solvent_total_volume_ml / LITER_TO_ML + NUMERICAL_EPS)
    cluster_population_prior = contact_pair_prior * (1.0 + ionic_source_diversity)
    cluster_persistence_prior = cluster_population_prior * (1.0 + neutral_shell_persistence)
    crowding_prior = (
        total_ionic_source_M * mean_jones_dole
        + neutral_additive_volume_fraction
        + shell_dimer_fraction
    )
    activity_prior = total_ionic_source_M * free_ion_prior
    dielectric_transport_support = epsilon_effective / (
        epsilon_effective + jnp.abs(mean_binding) + 1.0
    )
    ionic_network_transport_support = (
        total_ionic_source_M
        * mean_anion_flex
        * (free_ion_prior + cluster_population_prior)
        * dielectric_transport_support
        / (1.0 + crowding_prior + cluster_persistence_prior)
    )
    mixed_anion_competition = ionic_source_diversity * mean_anion_flex / (1.0 + crowding_prior)
    normalized_lambda_contrast = salt_pair_lambda_contrast / (mean_lambda0 + NUMERICAL_EPS)
    normalized_binding_contrast = salt_pair_binding_contrast / (jnp.abs(mean_binding) + 1.0)
    normalized_kd_contrast = salt_pair_kd_contrast / (mean_ion_pair_kd + 1.0)
    normalized_stokes_contrast = salt_pair_stokes_contrast / (mean_stokes_alpha + 1.0)
    salt_additive_shell_coupling = salt_total_M * additive_shell_fraction
    salt_additive_steric_coupling = salt_additive_shell_coupling * shell_steric
    salt_additive_saturation = additive_shell_fraction * neutral_shell_persistence * neutral_additive_volume_fraction
    salt_pair_transport_contrast = (
        normalized_lambda_contrast
        + salt_pair_anion_size_contrast / (mean_anion_radius + NUMERICAL_EPS)
        + normalized_binding_contrast
        + normalized_kd_contrast
        + salt_pair_dielectric_decrement_contrast / (mean_dielectric_decrement + 1.0)
        + normalized_stokes_contrast
        + salt_pair_flex_contrast
    )
    neutral_additive_volume_ml = jnp.sum(additive_volumes_ml * neutral_additive_mask)
    neutral_additive_steric = (
        jnp.sum(additive_volumes_ml * neutral_additive_mask * steric)
        / (neutral_additive_volume_ml + NUMERICAL_EPS)
    )
    neutral_additive_epsilon = (
        jnp.sum(additive_volumes_ml * neutral_additive_mask * epsilon)
        / (neutral_additive_volume_ml + NUMERICAL_EPS)
    )
    additive_dielectric_enrichment = _positive_part(
        (neutral_additive_epsilon - epsilon_liquid) / (jnp.abs(epsilon_liquid) + NUMERICAL_EPS)
    )
    additive_transport_drag = (
        neutral_additive_volume_fraction
        * neutral_additive_steric
        / (1.0 + additive_dielectric_enrichment)
    )
    salt_pair_anticorrelation_screening = (
        salt_pair_molarity
        * mean_anion_flex
        * (1.0 + normalized_binding_contrast + normalized_kd_contrast)
        / (1.0 + crowding_prior)
    )
    salt_pair_like_current_support = (
        salt_pair_molarity
        * mean_anion_flex
        * (1.0 + normalized_lambda_contrast + normalized_stokes_contrast + salt_pair_flex_contrast)
        / (1.0 + crowding_prior)
    )
    salt_pair_cluster_transport_support = (
        salt_pair_molarity
        * (cluster_population_prior + ionic_source_diversity)
        * (1.0 + salt_pair_flex_contrast)
        / (1.0 + crowding_prior + cluster_persistence_prior)
    )
    salt_additive_anticorrelation_screening = (
        salt_additive_shell_coupling
        * mean_anion_flex
        / (1.0 + salt_additive_saturation + additive_transport_drag)
    )
    mixed_anion_additive_current_support = (
        salt_additive_anticorrelation_screening
        * (
            salt_pair_transport_contrast
            + additive_dielectric_enrichment
            + mixed_anion_competition
        )
        / (1.0 + salt_additive_saturation + additive_transport_drag)
    )
    salt_additive_dielectric_screening = (
        salt_total_M
        * neutral_additive_volume_fraction
        * additive_dielectric_enrichment
        * mean_anion_flex
        / (1.0 + additive_transport_drag)
    )
    salt_additive_like_current_support = (
        salt_additive_steric_coupling
        * mean_anion_flex
        / (1.0 + crowding_prior + salt_additive_saturation)
    )
    salt_additive_cluster_transport_support = (
        salt_additive_shell_coupling
        * cluster_population_prior
        / (1.0 + crowding_prior + salt_additive_saturation + additive_transport_drag)
    )
    temperature_scaled = temperature_K / T_REF_K

    return jnp.asarray(
        [
            solvent_sum,
            total_ionic_source_M,
            additive_total_wt,
            neutral_additive_volume_fraction,
            jnp.sum(ionic_additive_molarity),
            solvent_total_volume_ml,
            density_prior,
            salt_volume_fraction,
            eta_liquid,
            eta_solution_prior,
            eta_salt_solvent_prior,
            neutral_additive_viscosity_excess_factor,
            epsilon_liquid,
            epsilon_effective,
            mean_dielectric_decrement,
            mean_lambda0,
            mean_cation_radius,
            mean_anion_radius,
            mean_anion_flex,
            mean_jones_dole,
            mean_ion_pair_kd,
            mean_binding,
            mean_stokes_alpha,
            shell_strength_sum,
            additive_shell_fraction,
            neutral_shell_persistence,
            contact_pair_prior,
            free_ion_prior,
            total_ionic_source_M,
            ionic_source_diversity,
            free_solvent_availability,
            solvent_starvation,
            cluster_population_prior,
            cluster_persistence_prior,
            ionic_network_transport_support,
            crowding_prior,
            activity_prior,
            mixed_anion_competition,
            salt_pair_molarity,
            salt_pair_lambda_contrast,
            salt_pair_anion_size_contrast,
            salt_pair_binding_contrast,
            salt_pair_kd_contrast,
            salt_pair_dielectric_decrement_contrast,
            salt_pair_stokes_contrast,
            salt_pair_flex_contrast,
            salt_additive_shell_coupling,
            salt_additive_steric_coupling,
            salt_additive_saturation,
            salt_pair_transport_contrast,
            salt_pair_anticorrelation_screening,
            salt_pair_like_current_support,
            salt_pair_cluster_transport_support,
            salt_additive_dielectric_screening,
            salt_additive_anticorrelation_screening,
            mixed_anion_additive_current_support,
            salt_additive_like_current_support,
            salt_additive_cluster_transport_support,
            additive_transport_drag,
            temperature_scaled,
        ]
    )


def _mechanism_readout(
    head: jnp.ndarray,
    physical: jnp.ndarray,
    generic_positive_current_scale: jnp.ndarray,
    mixed_anion_anticorrelation_logit: jnp.ndarray,
    temperature_transport_activation_K: jnp.ndarray,
) -> jnp.ndarray:
    density_prior = physical[PHYSICAL_FEATURE_NAMES.index("density_prior_g_ml")]
    eta_base_prior = physical[PHYSICAL_FEATURE_NAMES.index("eta_solution_prior_cP")]
    eta_salt_solvent_base_prior = physical[
        PHYSICAL_FEATURE_NAMES.index("eta_salt_solvent_prior_cP")
    ]
    neutral_additive_viscosity_excess_factor = physical[
        PHYSICAL_FEATURE_NAMES.index("neutral_additive_viscosity_excess_factor")
    ]
    eta_liquid_base_prior = physical[PHYSICAL_FEATURE_NAMES.index("eta_liquid_prior_cP")]
    salt_volume_fraction = physical[PHYSICAL_FEATURE_NAMES.index("salt_volume_fraction")]
    temperature_scaled = physical[PHYSICAL_FEATURE_NAMES.index("temperature_scaled")]
    temperature_K = temperature_scaled * T_REF_K
    temperature_viscosity_factor = jnp.exp(
        temperature_transport_activation_K
        * ((1.0 / (temperature_K + NUMERICAL_EPS)) - (1.0 / T_REF_K))
    )
    eta_prior = eta_base_prior * temperature_viscosity_factor
    eta_salt_solvent_prior = eta_salt_solvent_base_prior * temperature_viscosity_factor
    eta_liquid_prior = eta_liquid_base_prior * temperature_viscosity_factor
    neutral_additive_volume_fraction = physical[
        PHYSICAL_FEATURE_NAMES.index("neutral_additive_volume_fraction")
    ]
    epsilon_effective_prior = physical[PHYSICAL_FEATURE_NAMES.index("epsilon_effective")]
    activity_prior = physical[PHYSICAL_FEATURE_NAMES.index("activity_prior_M")]
    mean_lambda0 = physical[PHYSICAL_FEATURE_NAMES.index("mean_lambda0_S_cm2_mol")]
    mean_cation_radius = physical[PHYSICAL_FEATURE_NAMES.index("mean_cation_radius_A")]
    mean_anion_radius = physical[PHYSICAL_FEATURE_NAMES.index("mean_anion_radius_A")]
    mean_stokes_alpha = physical[PHYSICAL_FEATURE_NAMES.index("mean_stokes_alpha_anion")]
    cation_solvation_strength = physical[PHYSICAL_FEATURE_NAMES.index("cation_solvation_strength_M")]
    crowding_prior = physical[PHYSICAL_FEATURE_NAMES.index("crowding_prior")]
    free_solvent_availability = physical[PHYSICAL_FEATURE_NAMES.index("free_solvent_availability")]
    solvent_starvation = physical[PHYSICAL_FEATURE_NAMES.index("solvent_starvation")]
    additive_shell_prior = physical[PHYSICAL_FEATURE_NAMES.index("additive_shell_fraction")]
    contact_pair_prior = physical[PHYSICAL_FEATURE_NAMES.index("contact_pair_prior")]
    free_ion_prior = physical[PHYSICAL_FEATURE_NAMES.index("free_ion_prior")]
    cluster_population_prior = physical[PHYSICAL_FEATURE_NAMES.index("cluster_population_prior")]
    cluster_persistence_prior = physical[PHYSICAL_FEATURE_NAMES.index("cluster_persistence_prior")]
    ionic_network_transport_support = physical[
        PHYSICAL_FEATURE_NAMES.index("ionic_network_transport_support")
    ]
    salt_additive_saturation = physical[PHYSICAL_FEATURE_NAMES.index("salt_additive_saturation")]
    salt_pair_anticorrelation_screening = physical[
        PHYSICAL_FEATURE_NAMES.index("salt_pair_anticorrelation_screening")
    ]
    salt_pair_like_current_support = physical[PHYSICAL_FEATURE_NAMES.index("salt_pair_like_current_support")]
    salt_pair_cluster_transport_support = physical[
        PHYSICAL_FEATURE_NAMES.index("salt_pair_cluster_transport_support")
    ]
    salt_additive_dielectric_screening = physical[
        PHYSICAL_FEATURE_NAMES.index("salt_additive_dielectric_screening")
    ]
    salt_additive_anticorrelation_screening = physical[
        PHYSICAL_FEATURE_NAMES.index("salt_additive_anticorrelation_screening")
    ]
    mixed_anion_competition = physical[PHYSICAL_FEATURE_NAMES.index("mixed_anion_competition")]
    salt_pair_transport_contrast = physical[PHYSICAL_FEATURE_NAMES.index("salt_pair_transport_contrast")]
    mixed_anion_additive_current_support = physical[
        PHYSICAL_FEATURE_NAMES.index("mixed_anion_additive_current_support")
    ]
    salt_additive_like_current_support = physical[
        PHYSICAL_FEATURE_NAMES.index("salt_additive_like_current_support")
    ]
    salt_additive_cluster_transport_support = physical[
        PHYSICAL_FEATURE_NAMES.index("salt_additive_cluster_transport_support")
    ]
    additive_transport_drag = physical[PHYSICAL_FEATURE_NAMES.index("additive_transport_drag")]

    density_pred = density_prior * _positive_unit_multiplier(_head_value(head, "density_log_ratio_head"))
    viscosity_multiplier = _positive_unit_multiplier(_head_value(head, "viscosity_log_ratio_head"))
    eta_baseline_fit = eta_salt_solvent_prior * viscosity_multiplier
    eta_solution = eta_baseline_fit * neutral_additive_viscosity_excess_factor
    eta_supervised_fit = eta_solution
    eta_transport = eta_solution
    epsilon_effective_pred = epsilon_effective_prior * _positive_unit_multiplier(
        _head_value(head, "dielectric_log_ratio_head")
    )
    effective_ion_concentration = activity_prior
    association_fraction = jax.nn.sigmoid(
        _safe_logit(contact_pair_prior) + _head_value(head, "association_logit_delta_head")
    )
    free_ion_fraction = 1.0 - association_fraction
    additive_shell_participation = jax.nn.sigmoid(
        _safe_logit(additive_shell_prior)
        + _head_value(head, "additive_shell_logit_delta_head")
    )
    crowding = crowding_prior
    cluster_population_prior_fraction = cluster_population_prior / (1.0 + cluster_population_prior)
    cluster_population = jax.nn.sigmoid(
        _safe_logit(cluster_population_prior_fraction)
        + _head_value(head, "cluster_population_logit_delta_head")
    )
    cluster_persistence = cluster_persistence_prior * _positive_unit_multiplier(
        _head_value(head, "cluster_persistence_log_ratio_head")
    )
    additive_anticorrelation_screening_support = (
        salt_additive_dielectric_screening
        + salt_additive_anticorrelation_screening
    )
    activity_transport = effective_ion_concentration
    association_transport = association_fraction / (
        1.0 + additive_anticorrelation_screening_support
    )
    free_ion_transport = 1.0 - association_transport
    crowding_transport = crowding
    cluster_population_transport = cluster_population
    cluster_persistence_transport = cluster_persistence

    shell_occupancy = cation_solvation_strength / (1.0 + cation_solvation_strength)
    viscosity_ratio = eta_liquid_base_prior / (eta_transport + NUMERICAL_EPS)
    viscosity_coupling = (
        crowding_transport
        + cluster_population_transport
    ) / (
        1.0
        + crowding_transport
        + cluster_population_transport
    )
    cation_viscosity_friction = jnp.exp(
        shell_occupancy * viscosity_coupling * jnp.log(viscosity_ratio + NUMERICAL_EPS)
    )
    anion_stokes_drag = _positive_part(mean_stokes_alpha)
    anion_viscosity_friction = jnp.exp(
        anion_stokes_drag * viscosity_coupling * jnp.log(viscosity_ratio + NUMERICAL_EPS)
    )
    free_solvent_mobility = free_solvent_availability / (
        free_solvent_availability
        + salt_volume_fraction
        + cluster_persistence_transport
        + NUMERICAL_EPS
    )
    finite_concentration_drag = (
        1.0
        + crowding_transport * (association_transport + cluster_population_transport)
        + cluster_persistence_transport
        + solvent_starvation * cluster_population_transport
        + additive_transport_drag
    )
    finite_concentration_mobility = free_solvent_mobility / finite_concentration_drag
    finite_concentration_correlation_drive = (
        association_transport
        * (
            crowding_transport
            + cluster_population_transport
            + cluster_persistence_transport
            + solvent_starvation
        )
    )
    cation_mobility_weight = 1.0 / (
        mean_cation_radius * (1.0 + shell_occupancy) + NUMERICAL_EPS
    )
    anion_mobility_weight = 1.0 / (
        mean_anion_radius * (1.0 + anion_stokes_drag * crowding_transport) + NUMERICAL_EPS
    )
    mobility_weight_sum = cation_mobility_weight + anion_mobility_weight + NUMERICAL_EPS
    cation_self_mobility_gate = jax.nn.sigmoid(
        _head_value(head, "cation_self_mobility_gate")
    )
    anion_self_mobility_gate = jax.nn.sigmoid(
        _head_value(head, "anion_self_mobility_gate")
    )
    cation_lambda = (
        mean_lambda0
        * cation_mobility_weight
        / mobility_weight_sum
        * cation_self_mobility_gate
    )
    anion_lambda = (
        mean_lambda0
        * anion_mobility_weight
        / mobility_weight_sum
        * anion_self_mobility_gate
    )
    current_common = (
        (activity_transport + NUMERICAL_EPS)
        * (free_ion_transport + NUMERICAL_EPS)
    )
    cation_self = (
        current_common
        * cation_lambda
        * cation_viscosity_friction
        / (1.0 + additive_transport_drag)
    )
    anion_self = (
        current_common
        * anion_lambda
        * anion_viscosity_friction
        / (1.0 + additive_transport_drag)
    )
    sigma_self = cation_self + anion_self
    self_current_scale_prior = (
        (activity_transport + NUMERICAL_EPS)
        * (mean_lambda0 + NUMERICAL_EPS)
        * (free_ion_transport + NUMERICAL_EPS)
        * (cation_viscosity_friction + anion_viscosity_friction)
        / (2.0 * (1.0 + additive_transport_drag))
    )

    anticorrelation_screening = (
        salt_pair_anticorrelation_screening
        + additive_anticorrelation_screening_support
    )
    like_current_support = salt_pair_like_current_support
    cluster_transport_support = (
        salt_pair_cluster_transport_support
        + ionic_network_transport_support
    )
    generic_anticorrelation_support = salt_pair_anticorrelation_screening
    collective_current_support = (
        like_current_support
        + cluster_transport_support
        + generic_anticorrelation_support
    )
    ca_drive = (
        association_transport
        * (1.0 + finite_concentration_correlation_drive + additive_transport_drag)
        / (1.0 + anticorrelation_screening)
    )
    ca_drive_without_additive_screening = (
        association_fraction
        * (1.0 + finite_concentration_correlation_drive + additive_transport_drag)
        / (1.0 + salt_pair_anticorrelation_screening)
    )
    distinct_current_scale = sigma_self
    ca_negative_drive = (
        ca_drive
        * jax.nn.sigmoid(_head_value(head, "cation_anion_gate"))
    )
    ca_negative_drive_without_additive_screening = (
        ca_drive_without_additive_screening
        * jax.nn.sigmoid(_head_value(head, "cation_anion_gate"))
    )
    mixed_anion_drive = (
        mixed_anion_competition
        * (1.0 + salt_pair_transport_contrast)
        * association_transport
        * (1.0 + finite_concentration_correlation_drive)
        / (1.0 + anticorrelation_screening + additive_transport_drag)
    )
    mixed_anion_drive_without_additive_screening = (
        mixed_anion_competition
        * (1.0 + salt_pair_transport_contrast)
        * association_fraction
        * (1.0 + finite_concentration_correlation_drive)
        / (1.0 + salt_pair_anticorrelation_screening + additive_transport_drag)
    )
    mixed_anion_negative_drive = (
        mixed_anion_drive
        * jax.nn.sigmoid(mixed_anion_anticorrelation_logit)
    )
    mixed_anion_negative_drive_without_additive_screening = (
        mixed_anion_drive_without_additive_screening
        * jax.nn.sigmoid(mixed_anion_anticorrelation_logit)
    )
    total_negative_drive = ca_negative_drive + mixed_anion_negative_drive + NUMERICAL_EPS
    total_negative_drive_without_additive_screening = (
        ca_negative_drive_without_additive_screening
        + mixed_anion_negative_drive_without_additive_screening
        + NUMERICAL_EPS
    )
    total_negative_fraction = total_negative_drive / (1.0 + total_negative_drive)
    total_negative_fraction_without_additive_screening = (
        total_negative_drive_without_additive_screening
        / (1.0 + total_negative_drive_without_additive_screening)
    )
    cation_anion_fraction = (
        -total_negative_fraction * ca_negative_drive / total_negative_drive
    )
    mixed_anion_anticorrelation_fraction = (
        -total_negative_fraction * mixed_anion_negative_drive / total_negative_drive
    )
    cation_anion_fraction_without_additive_screening = (
        -total_negative_fraction_without_additive_screening
        * ca_negative_drive_without_additive_screening
        / total_negative_drive_without_additive_screening
    )
    mixed_anion_anticorrelation_fraction_without_additive_screening = (
        -total_negative_fraction_without_additive_screening
        * mixed_anion_negative_drive_without_additive_screening
        / total_negative_drive_without_additive_screening
    )
    cation_cation_score = (
        free_ion_transport
        * (like_current_support + generic_anticorrelation_support)
        * jax.nn.sigmoid(_head_value(head, "cation_cation_gate"))
    )
    anion_anion_score = (
        free_ion_transport
        * (like_current_support + generic_anticorrelation_support)
        * jax.nn.sigmoid(_head_value(head, "anion_anion_gate"))
    )
    cluster_drift_score = (
        cluster_population_transport
        * (cluster_transport_support + generic_anticorrelation_support)
        * jax.nn.sigmoid(_head_value(head, "cluster_drift_gate"))
    )
    relaxation_tail_score = (
        cluster_population_transport
        * cluster_persistence_transport
        * (cluster_transport_support + generic_anticorrelation_support)
        * jax.nn.sigmoid(_head_value(head, "relaxation_tail_gate"))
    )
    ionic_network_score = (
        cluster_population_transport
        * ionic_network_transport_support
        * jax.nn.sigmoid(_head_value(head, "ionic_network_gate"))
    )
    positive_score_sum = (
        cation_cation_score
        + anion_anion_score
        + cluster_drift_score
        + relaxation_tail_score
        + ionic_network_score
        + NUMERICAL_EPS
    )
    positive_current_support_fraction = (
        collective_current_support
    ) / (
        1.0
        + collective_current_support
        + crowding_transport
        + salt_additive_saturation
        + additive_transport_drag
    )
    positive_current_capacity = distinct_current_scale * positive_current_support_fraction
    cation_cation_fraction = (
        generic_positive_current_scale * cation_cation_score / positive_score_sum
    )
    anion_anion_fraction = (
        generic_positive_current_scale * anion_anion_score / positive_score_sum
    )
    cluster_drift_fraction = (
        generic_positive_current_scale * cluster_drift_score / positive_score_sum
    )
    relaxation_tail_fraction = (
        generic_positive_current_scale * relaxation_tail_score / positive_score_sum
    )
    ionic_network_fraction = ionic_network_score / positive_score_sum
    additive_compensation_support = (
        mixed_anion_additive_current_support
        + salt_additive_like_current_support
        + salt_additive_cluster_transport_support
    )
    mixed_anion_additive_drive = additive_compensation_support / (
        1.0
        + association_transport
        + mixed_anion_competition
        + salt_additive_saturation
        + additive_transport_drag
    )
    mixed_anion_additive_fraction = mixed_anion_additive_drive / (
        1.0 + mixed_anion_additive_drive
    )

    cation_anion_distinct = sigma_self * cation_anion_fraction
    mixed_anion_anticorrelation_current = (
        sigma_self * mixed_anion_anticorrelation_fraction
    )
    cation_anion_distinct_without_additive_screening = (
        sigma_self * cation_anion_fraction_without_additive_screening
    )
    mixed_anion_anticorrelation_current_without_additive_screening = (
        sigma_self * mixed_anion_anticorrelation_fraction_without_additive_screening
    )
    cation_cation_distinct = positive_current_capacity * cation_cation_fraction
    anion_anion_distinct = positive_current_capacity * anion_anion_fraction
    cluster_drift = positive_current_capacity * cluster_drift_fraction
    ionic_network_current = positive_current_capacity * ionic_network_fraction
    anticorrelation_loss = -(cation_anion_distinct + mixed_anion_anticorrelation_current)
    anticorrelation_loss_without_additive_screening = -(
        cation_anion_distinct_without_additive_screening
        + mixed_anion_anticorrelation_current_without_additive_screening
    )
    avoided_anticorrelation_loss = _positive_part(
        anticorrelation_loss_without_additive_screening - anticorrelation_loss
    )
    mixed_anion_additive_current = (
        avoided_anticorrelation_loss
    ) * mixed_anion_additive_fraction

    relaxation_tail = positive_current_capacity * relaxation_tail_fraction
    raw_distinct_current_correction = (
        cation_anion_distinct
        + mixed_anion_anticorrelation_current
        + cation_cation_distinct
        + anion_anion_distinct
        + cluster_drift
        + ionic_network_current
        + mixed_anion_additive_current
        + relaxation_tail
    )
    distinct_ratio = raw_distinct_current_correction / (sigma_self + NUMERICAL_EPS)
    distinct_scale = 1.0 / (1.0 + jnp.abs(distinct_ratio))
    cation_anion_distinct = cation_anion_distinct * distinct_scale
    mixed_anion_anticorrelation_current = mixed_anion_anticorrelation_current * distinct_scale
    cation_cation_distinct = cation_cation_distinct * distinct_scale
    anion_anion_distinct = anion_anion_distinct * distinct_scale
    cluster_drift = cluster_drift * distinct_scale
    ionic_network_current = ionic_network_current * distinct_scale
    mixed_anion_additive_current = mixed_anion_additive_current * distinct_scale
    relaxation_tail = relaxation_tail * distinct_scale
    distinct_current_correction = raw_distinct_current_correction * distinct_scale
    current_integral = (
        sigma_self
        + distinct_current_correction
    )
    sigma_mS_cm = current_integral
    return jnp.asarray(
        [
            density_pred,
            eta_solution,
            eta_supervised_fit,
            epsilon_effective_pred,
            effective_ion_concentration,
            association_fraction,
            free_ion_fraction,
            additive_shell_participation,
            crowding,
            cluster_population,
            cluster_persistence,
            additive_transport_drag,
            temperature_viscosity_factor,
            additive_anticorrelation_screening_support,
            association_transport,
            free_ion_transport,
            current_common,
            cation_viscosity_friction,
            anion_viscosity_friction,
            free_solvent_mobility,
            finite_concentration_mobility,
            finite_concentration_correlation_drive,
            anticorrelation_screening,
            like_current_support,
            cluster_transport_support,
            positive_current_support_fraction,
            self_current_scale_prior,
            cation_self_mobility_gate,
            anion_self_mobility_gate,
            cation_self,
            anion_self,
            sigma_self,
            cation_anion_fraction,
            cation_anion_distinct,
            mixed_anion_anticorrelation_current,
            cation_cation_fraction,
            cation_cation_distinct,
            anion_anion_fraction,
            anion_anion_distinct,
            cluster_drift_fraction,
            cluster_drift,
            ionic_network_current,
            mixed_anion_additive_current,
            relaxation_tail_fraction,
            relaxation_tail,
            distinct_current_correction,
            current_integral,
            sigma_mS_cm,
        ]
    )


def _head_value(head: jnp.ndarray, name: str) -> jnp.ndarray:
    return head[CURRENT_HEAD_NAMES.index(name)]


def _positive_unit_multiplier(raw_value: jnp.ndarray) -> jnp.ndarray:
    unit_shift = jnp.log(jnp.expm1(1.0))
    return jax.nn.softplus(raw_value + unit_shift)


def _model_feature_idx(name: str) -> int:
    return MODEL_FEATURE_NAMES.index(name)


def _physical_consistency_loss(features: jnp.ndarray) -> jnp.ndarray:
    cation_self = features[:, _model_feature_idx("cation_self_current_mS_cm")]
    anion_self = features[:, _model_feature_idx("anion_self_current_mS_cm")]
    sigma = features[:, _model_feature_idx("sigma_mS_cm")]
    cation_anion_fraction = features[:, _model_feature_idx("cation_anion_distinct_fraction")]
    sign_violation = (
        jax.nn.relu(-cation_self)
        + jax.nn.relu(-anion_self)
        + jax.nn.relu(cation_anion_fraction)
        + jax.nn.relu(-sigma)
    )
    return jnp.mean(sign_violation * sign_violation)


def _masked_log_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    weighted_mask = mask * weights
    mask_sum = jnp.sum(weighted_mask)
    active = weighted_mask > 0.0
    log_error = jnp.where(
        active,
        jnp.log(prediction + NUMERICAL_EPS) - jnp.log(target + NUMERICAL_EPS),
        0.0,
    )
    return jnp.where(
        mask_sum > 0.0,
        jnp.sum(weighted_mask * log_error * log_error) / (mask_sum + NUMERICAL_EPS),
        0.0,
    )


def _masked_relative_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    weighted_mask = mask * weights
    mask_sum = jnp.sum(weighted_mask)
    active = weighted_mask > 0.0
    denominator = jnp.where(active, jnp.abs(target) + NUMERICAL_EPS, 1.0)
    scaled_error = jnp.where(active, (prediction - target) / denominator, 0.0)
    return jnp.where(
        mask_sum > 0.0,
        jnp.sum(weighted_mask * scaled_error * scaled_error) / (mask_sum + NUMERICAL_EPS),
        0.0,
    )


def _masked_current_scale_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    current_scale: jnp.ndarray,
    mask: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    weighted_mask = mask * weights
    mask_sum = jnp.sum(weighted_mask)
    active = weighted_mask > 0.0
    denominator = jnp.where(active, jnp.abs(current_scale) + NUMERICAL_EPS, 1.0)
    scaled_error = jnp.where(active, (prediction - target) / denominator, 0.0)
    return jnp.where(
        mask_sum > 0.0,
        jnp.sum(weighted_mask * scaled_error * scaled_error) / (mask_sum + NUMERICAL_EPS),
        0.0,
    )


def _masked_squared_current_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    weighted_mask = mask * weights
    mask_sum = jnp.sum(weighted_mask)
    active = weighted_mask > 0.0
    error = jnp.where(active, prediction - target, 0.0)
    return jnp.where(
        mask_sum > 0.0,
        jnp.sum(weighted_mask * error * error) / (mask_sum + NUMERICAL_EPS),
        0.0,
    )


def _masked_fraction_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    weighted_mask = mask * weights
    mask_sum = jnp.sum(weighted_mask)
    active = weighted_mask > 0.0
    error = jnp.where(active, prediction - target, 0.0)
    return jnp.where(
        mask_sum > 0.0,
        jnp.sum(weighted_mask * error * error) / (mask_sum + NUMERICAL_EPS),
        0.0,
    )


def _weighted_pool(z: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    weight_sum = jnp.sum(weights) + NUMERICAL_EPS
    return jnp.sum(z * weights[:, None], axis=0) / weight_sum


def _ionic_additive_weight_for_self(
    species_props_raw: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    ionic_additive_mask = jnp.where(
        (additive_weight_fraction > 0.0) & (species_props_raw[:, IDX_LAMBDA0] > 0.0),
        1.0,
        0.0,
    )
    return additive_weight_fraction * ionic_additive_mask * mask


def _pairwise_role_pair_pools(
    params: Mapping[str, jnp.ndarray],
    z: jnp.ndarray,
    species_props_norm: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    zi = jnp.broadcast_to(z[:, None, :], (N_MAX_SPECIES, N_MAX_SPECIES, D_HIDDEN))
    zj = jnp.broadcast_to(z[None, :, :], (N_MAX_SPECIES, N_MAX_SPECIES, D_HIDDEN))
    prop_delta = jnp.abs(species_props_norm[:, None, :] - species_props_norm[None, :, :])
    pair_input = jnp.concatenate([zi, zj, prop_delta], axis=2)
    pair_hidden = jax.nn.gelu(pair_input @ params["pair_h_w"] + params["pair_h_b"])
    pair_message = jnp.tanh(pair_hidden @ params["pair_out_w"] + params["pair_out_b"])

    solvent_loading = solvent_volume_fraction * mask
    salt_loading = salt_molarity * mask
    additive_loading = additive_weight_fraction * mask
    non_self_pair = 1.0 - jnp.eye(N_MAX_SPECIES, dtype=z.dtype)
    salt_salt_weight = salt_loading[:, None] * salt_loading[None, :] * non_self_pair
    salt_additive_weight = (
        salt_loading[:, None] * additive_loading[None, :]
        + additive_loading[:, None] * salt_loading[None, :]
    )
    solvent_salt_weight = (
        solvent_loading[:, None] * salt_loading[None, :]
        + salt_loading[:, None] * solvent_loading[None, :]
    )
    solvent_additive_weight = (
        solvent_loading[:, None] * additive_loading[None, :]
        + additive_loading[:, None] * solvent_loading[None, :]
    )
    return jnp.concatenate(
        [
            _weighted_pair_pool(pair_message, salt_salt_weight),
            _weighted_pair_pool(pair_message, salt_additive_weight),
            _weighted_pair_pool(pair_message, solvent_salt_weight),
            _weighted_pair_pool(pair_message, solvent_additive_weight),
        ],
        axis=0,
    )


def _self_mobility_pair_pools(pair_pools: jnp.ndarray) -> jnp.ndarray:
    salt_salt_pool, salt_additive_pool, solvent_salt_pool, solvent_additive_pool = jnp.split(
        pair_pools,
        len(PAIR_POOL_NAMES),
    )
    additive_pair_zero = jnp.zeros_like(salt_additive_pool)
    solvent_additive_pair_zero = jnp.zeros_like(solvent_additive_pool)
    return jnp.concatenate(
        [
            salt_salt_pool,
            additive_pair_zero,
            solvent_salt_pool,
            solvent_additive_pair_zero,
        ],
        axis=0,
    )


def _weighted_pair_pool(pair_message: jnp.ndarray, pair_weights: jnp.ndarray) -> jnp.ndarray:
    weight_sum = jnp.sum(pair_weights) + NUMERICAL_EPS
    return jnp.sum(pair_message * pair_weights[:, :, None], axis=(0, 1)) / weight_sum


def _property_similarity_bias(species_props_norm: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    props_norm = species_props_norm / (
        jnp.sqrt(jnp.sum(species_props_norm * species_props_norm, axis=1, keepdims=True)) + NUMERICAL_EPS
    )
    return (props_norm @ props_norm.T) * mask[:, None] * mask[None, :]


def _stable_multihead_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray,
    prop_bias: jnp.ndarray,
) -> jnp.ndarray:
    seq_len, d_model = q.shape
    d_head = d_model // _base.N_HEADS
    q_heads = q.reshape(seq_len, _base.N_HEADS, d_head).transpose(1, 0, 2)
    k_heads = k.reshape(seq_len, _base.N_HEADS, d_head).transpose(1, 0, 2)
    v_heads = v.reshape(seq_len, _base.N_HEADS, d_head).transpose(1, 0, 2)
    scale = jnp.sqrt(jnp.asarray(d_head, dtype=jnp.float64))
    logits = jnp.matmul(q_heads, k_heads.transpose(0, 2, 1)) / scale
    logits = logits + prop_bias[None, :, :]
    key_mask = mask[None, None, :]
    query_mask = mask[None, :, None]
    logits = jnp.where(key_mask > 0.0, logits, -1e9)
    weights = jax.nn.softmax(logits, axis=-1)
    weights = weights * key_mask * query_mask
    return jnp.matmul(weights, v_heads).transpose(1, 0, 2).reshape(seq_len, d_model)


def _weighted_std(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.sum(weights * values)
    centered = values - mean
    return jnp.sqrt(jnp.sum(weights * centered * centered) + NUMERICAL_EPS)


def _positive_part(value: jnp.ndarray) -> jnp.ndarray:
    return (value + jnp.sqrt(value * value + NUMERICAL_EPS)) / 2.0


def _safe_logit(value: jnp.ndarray) -> jnp.ndarray:
    bounded = (value + NUMERICAL_EPS) / (1.0 + 2.0 * NUMERICAL_EPS)
    return jnp.log(bounded / (1.0 - bounded))


def _anion_flexibility(mean_anion_radius: jnp.ndarray) -> jnp.ndarray:
    cutoff = _base.ANTICORR_R_CUTOFF_A
    reference = _base.ANTICORR_R_FLEX_REF_A
    scaled = (mean_anion_radius - cutoff) / (reference - cutoff + NUMERICAL_EPS)
    scaled_nonnegative = _positive_part(scaled)
    return jnp.where(
        mean_anion_radius <= cutoff,
        0.0,
        jnp.where(mean_anion_radius >= reference, 1.0, scaled_nonnegative ** _base.ANTICORR_ALPHA_FLEX),
    )


def _validate_physical_stats(physical_mean: np.ndarray, physical_std: np.ndarray) -> None:
    expected_shape = (len(PHYSICAL_FEATURE_NAMES),)
    if physical_mean.shape != expected_shape:
        raise ValueError(f"physical_mean must have shape {expected_shape}, got {physical_mean.shape}")
    if physical_std.shape != expected_shape:
        raise ValueError(f"physical_std must have shape {expected_shape}, got {physical_std.shape}")
    if np.any(~np.isfinite(physical_mean)):
        raise ValueError("physical_mean contains non-finite values")
    if np.any(~np.isfinite(physical_std)) or np.any(physical_std <= 0.0):
        raise ValueError("physical_std must contain positive finite values")
