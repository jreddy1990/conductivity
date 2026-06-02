"""MolSet-native unit-aware conductivity prototype.

The incumbent production path converts solvent v/v, salt molarity, and additive
wt fraction correctly, then collapses them into one ``fracs`` vector before the
MolSet model sees the recipe. This prototype keeps the MolSet species/property
architecture but carries separate role-aware loading channels through encoding,
pooling, physics features, and the residual head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from constants import MS_CM_TO_S_M as _MS_CM_TO_S_M
from constants import T_REF_K
from conductivity import mol_set_sigma as _base
from data.species_data import ADDITIVES, SALTS, SOLVENTS
from utils.strict_validation import finite_float, positive_finite_float, require_mapping


EXTRA_PROPERTY_KEYS = (
    "steric_disruption_beta",
    "dimerization_constant_M_inv",
    "dielectric_decrement_frac_per_M",
    "stokes_einstein_alpha_anion",
    "residence_time_ns",
    "fragility_index",
)
PROTOTYPE_PROPERTY_KEYS = tuple(_base.PROPERTY_KEYS) + EXTRA_PROPERTY_KEYS

D_BASE_INPUT = _base.D_INPUT
D_INPUT = len(PROTOTYPE_PROPERTY_KEYS)
D_HIDDEN = _base.D_HIDDEN
D_FFN = _base.D_FFN
N_HEADS = _base.N_HEADS
N_LAYERS = _base.N_LAYERS
N_MAX_SPECIES = _base.N_MAX_SPECIES

IDX_MW = _base.IDX_MW
IDX_DENSITY = _base.IDX_DENSITY
IDX_EPSILON = _base.IDX_EPSILON
IDX_VISCOSITY = _base.IDX_VISCOSITY
IDX_DONOR = _base.IDX_DONOR
IDX_ACCEPTOR = _base.IDX_ACCEPTOR
IDX_LAMBDA0 = _base.IDX_LAMBDA0
IDX_ANION_R = _base.IDX_ANION_R
IDX_CATION_R = _base.IDX_CATION_R
IDX_BINDING = _base.IDX_ION_PAIR_BINDING
IDX_ION_PAIR_KD = _base.IDX_ION_PAIR_KD
IDX_COORD_AFFINITY = _base.IDX_COORD_AFFINITY
IDX_JONES_DOLE = _base.IDX_JONES_DOLE
IDX_STERIC = D_BASE_INPUT + EXTRA_PROPERTY_KEYS.index("steric_disruption_beta")
IDX_DIMERIZATION = D_BASE_INPUT + EXTRA_PROPERTY_KEYS.index("dimerization_constant_M_inv")
IDX_DIELECTRIC_DECREMENT = D_BASE_INPUT + EXTRA_PROPERTY_KEYS.index("dielectric_decrement_frac_per_M")
IDX_STOKES_ALPHA_ANION = D_BASE_INPUT + EXTRA_PROPERTY_KEYS.index("stokes_einstein_alpha_anion")
IDX_RESIDENCE_TIME = D_BASE_INPUT + EXTRA_PROPERTY_KEYS.index("residence_time_ns")
IDX_FRAGILITY = D_BASE_INPUT + EXTRA_PROPERTY_KEYS.index("fragility_index")

LITER_TO_ML = 1000.0
NUMERICAL_EPS = 1e-12
ENCODER_AUX_DIM = 10
PROTOTYPE_FEATURE_NAMES = (
    "solvent_volume_fraction_sum",
    "salt_molarity_total_M",
    "total_additive_wt_fraction",
    "neutral_additive_volume_fraction",
    "ionic_additive_molarity_M",
    "solvent_total_volume_ml_per_L",
    "electrolyte_density_g_ml",
    "salt_volume_fraction",
    "eta_liquid_cP",
    "eta_solution_cP",
    "dimer_viscosity_factor",
    "salt_viscosity_factor",
    "viscosity_mobility_factor",
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
    "association_drive",
    "contact_pair_fraction",
    "free_ion_fraction",
    "crowding_state",
    "ion_network_state",
    "ionic_source_diversity",
    "mixed_anion_relief_drive",
    "shell_steric_disruption",
    "shell_coordination_strength_M",
    "shell_dimer_fraction",
    "anticorrelation_relief_drive",
    "cation_anion_anticorrelation_drive",
    "cation_self_conductivity_mS_cm",
    "anion_self_conductivity_mS_cm",
    "sigma_self_mS_cm",
    "cation_anion_distinct_fraction",
    "cation_anion_distinct_mS_cm",
    "like_ion_distinct_fraction",
    "like_ion_distinct_mS_cm",
    "cluster_drift_fraction",
    "cluster_drift_mS_cm",
    "current_correlation_integral_mS_cm",
    "sigma_physics_mS_cm",
)
READOUT_INPUT_DIM = 3 * D_HIDDEN + len(PROTOTYPE_FEATURE_NAMES)


@dataclass(frozen=True)
class UnitAwareMolSetInputs:
    """Padded MolSet inputs with role-separated loading channels."""

    species_names: tuple[str, ...]
    species_props_norm: np.ndarray
    species_props_raw: np.ndarray
    solvent_volume_fraction: np.ndarray
    salt_molarity: np.ndarray
    additive_weight_fraction: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class UnitAwareMolSetResult:
    """Prototype prediction plus auditable MolSet-native transport features."""

    sigma_mS_cm: float
    sigma_s_m: float
    physics_sigma_mS_cm: float
    residual_log_sigma: float
    features: dict[str, float]
    feature_vector: np.ndarray
    species_names: tuple[str, ...]
    solvent_volume_fraction: np.ndarray
    salt_molarity: np.ndarray
    additive_weight_fraction: np.ndarray


def prototype_feature_names() -> tuple[str, ...]:
    """Return the role-aware transport feature order."""

    return PROTOTYPE_FEATURE_NAMES


def get_unit_aware_property_vector(name: str) -> np.ndarray:
    """Return incumbent MolSet properties plus prototype transport properties."""

    species_data = _species_data(name)
    base_vec = _base.get_raw_property_vector(name)
    extra = np.zeros(len(EXTRA_PROPERTY_KEYS), dtype=np.float64)
    for idx, key in enumerate(EXTRA_PROPERTY_KEYS):
        extra[idx] = _optional_numeric_species_property(species_data, key)
    return np.concatenate([base_vec, extra])


def compute_unit_aware_normalization(species_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute z-score stats for the prototype property vector."""

    if not species_names:
        raise ValueError("species_names must not be empty")
    vectors = np.asarray([get_unit_aware_property_vector(name) for name in species_names], dtype=np.float64)
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    return mean, std


def build_unit_aware_recipe_inputs(
    recipe: Mapping[str, Any],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> UnitAwareMolSetInputs:
    """Build padded role-aware MolSet inputs from ElectrolyteRecipeModel units."""

    _validate_normalization(norm_mean, norm_std)
    solvents = require_mapping(recipe, "solvents", "recipe")
    salts = require_mapping(recipe, "salts", "recipe")
    additives = require_mapping(recipe, "additives", "recipe")
    species_rows = _role_ordered_species(solvents, salts, additives)
    if len(species_rows) > N_MAX_SPECIES:
        raise ValueError(f"Recipe has {len(species_rows)} species; N_MAX_SPECIES is {N_MAX_SPECIES}")

    props_norm = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    props_raw = np.zeros((N_MAX_SPECIES, D_INPUT), dtype=np.float64)
    solvent_vv = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    salt_m = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    additive_wt = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    mask = np.zeros(N_MAX_SPECIES, dtype=np.float64)
    names = []

    for idx, row in enumerate(species_rows):
        raw_vec = get_unit_aware_property_vector(row.name)
        props_raw[idx] = raw_vec
        props_norm[idx] = (raw_vec - norm_mean) / norm_std
        solvent_vv[idx] = row.solvent_volume_fraction
        salt_m[idx] = row.salt_molarity
        additive_wt[idx] = row.additive_weight_fraction
        mask[idx] = 1.0
        names.append(row.name)

    _validate_role_channels(solvent_vv, salt_m, additive_wt, mask)
    return UnitAwareMolSetInputs(
        species_names=tuple(names),
        species_props_norm=props_norm,
        species_props_raw=props_raw,
        solvent_volume_fraction=solvent_vv,
        salt_molarity=salt_m,
        additive_weight_fraction=additive_wt,
        mask=mask,
    )


def init_unit_aware_molset_params(key: jax.Array) -> dict[str, jnp.ndarray]:
    """Initialize a MolSet-style unit-aware prototype.

    The neural residual readout is zero-initialized, so the initial model is the
    role-aware transport backbone. Training can then learn bounded residuals on
    top of explicit mechanism features.
    """

    params: dict[str, jnp.ndarray] = {}

    def linear_init(rng: jax.Array, d_in: int, d_out: int, name: str) -> None:
        k1, _k2 = random.split(rng)
        scale = jnp.sqrt(2.0 / float(d_in))
        params[f"{name}_w"] = random.normal(k1, (d_in, d_out)) * scale
        params[f"{name}_b"] = jnp.zeros(d_out)

    n_keys = 1 + N_LAYERS * 6 + 2
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

    linear_init(keys[key_idx], READOUT_INPUT_DIM, D_HIDDEN, "residual_h")
    params["residual_out_w"] = jnp.zeros((D_HIDDEN, 1))
    params["residual_out_b"] = jnp.zeros(1)

    params["pairing_strength"] = jnp.array(1.0)
    params["salt_viscosity_scale"] = jnp.array(0.30)
    params["liquid_viscosity_exponent"] = jnp.array(1.05)
    params["dimer_viscosity_scale"] = jnp.array(1.0)
    params["association_binding_scale"] = jnp.array(0.04)
    params["association_dielectric_scale"] = jnp.array(35.0)
    params["crowding_salt_scale"] = jnp.array(0.16)
    params["crowding_additive_scale"] = jnp.array(4.0)
    params["crowding_dimer_scale"] = jnp.array(1.0)
    params["network_flex_scale"] = jnp.array(0.65)
    params["ca_anticorrelation_scale"] = jnp.array(0.65)
    params["network_anticorrelation_scale"] = jnp.array(3.0)
    params["crowding_anticorrelation_scale"] = jnp.array(0.55)
    params["mixed_anion_relief_scale"] = jnp.array(3.0)
    params["anticorrelation_relief_scale"] = jnp.array(110.0)
    params["relief_crowding_damping_scale"] = jnp.array(4.0)
    params["relief_additive_quadratic_damping_scale"] = jnp.array(250.0)
    params["relief_dimer_damping_scale"] = jnp.array(4.0)
    params["like_ion_correlation_scale"] = jnp.array(0.10)
    params["cluster_drift_scale"] = jnp.array(0.08)
    params["reference_liquid_viscosity_cP"] = jnp.array(1.0)
    params["conductivity_scale"] = jnp.array(0.22)
    return params


def evaluate_unit_aware_molset(
    recipe: Mapping[str, Any],
    temperature_K: float,
    params: Mapping[str, Any],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> UnitAwareMolSetResult:
    """Evaluate the MolSet-native unit-aware prototype for one recipe."""

    positive_finite_float(temperature_K, "temperature_K")
    inputs = build_unit_aware_recipe_inputs(recipe, norm_mean, norm_std)
    log_sigma, residual_log_sigma, feature_vector = forward_unit_aware_molset(
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
    residual_value = float(residual_log_sigma)
    features = {
        name: float(value)
        for name, value in zip(PROTOTYPE_FEATURE_NAMES, np.asarray(feature_vector), strict=True)
    }
    physics_sigma = features["sigma_physics_mS_cm"]
    return UnitAwareMolSetResult(
        sigma_mS_cm=sigma_mS_cm,
        sigma_s_m=sigma_mS_cm * _MS_CM_TO_S_M,
        physics_sigma_mS_cm=physics_sigma,
        residual_log_sigma=residual_value,
        features=features,
        feature_vector=np.asarray(feature_vector),
        species_names=inputs.species_names,
        solvent_volume_fraction=inputs.solvent_volume_fraction,
        salt_molarity=inputs.salt_molarity,
        additive_weight_fraction=inputs.additive_weight_fraction,
    )


def forward_unit_aware_molset(
    params: Mapping[str, Any],
    species_props_norm: jnp.ndarray,
    species_props_raw: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MolSet-style forward pass with separate role loading channels."""

    n_max = species_props_norm.shape[0]
    temperature_scaled = temperature_K / T_REF_K
    solvent_mask = jnp.where(solvent_volume_fraction > 0.0, 1.0, 0.0)
    salt_mask = jnp.where(salt_molarity > 0.0, 1.0, 0.0)
    additive_mask = jnp.where(additive_weight_fraction > 0.0, 1.0, 0.0)
    aux = jnp.stack(
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
    z = jax.nn.gelu(
        jnp.concatenate([species_props_norm, aux], axis=1) @ params["enc_w"] + params["enc_b"]
    ) * mask[:, None]

    prop_bias = _property_similarity_bias(species_props_norm, mask)
    for layer in range(N_LAYERS):
        q = z @ params[f"attn{layer}_q_w"] + params[f"attn{layer}_q_b"]
        k = z @ params[f"attn{layer}_k_w"] + params[f"attn{layer}_k_b"]
        v = z @ params[f"attn{layer}_v_w"] + params[f"attn{layer}_v_b"]
        attn_out = _base._multihead_attention(
            q,
            k,
            v,
            mask,
            prop_bias,
            random.PRNGKey(0),
            0.0,
        )
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

    transport_features, log_sigma_physics = _compute_role_aware_transport_features(
        raw_props=species_props_raw,
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=additive_weight_fraction,
        mask=mask,
        temperature_K=temperature_K,
        params=params,
    )
    solvent_pool = _weighted_pool(z, solvent_volume_fraction * mask)
    salt_pool = _weighted_pool(z, salt_molarity * mask)
    additive_pool = _weighted_pool(z, additive_weight_fraction * mask)
    residual_input = jnp.concatenate(
        [solvent_pool, salt_pool, additive_pool, transport_features],
        axis=0,
    )
    residual_hidden = jax.nn.gelu(
        residual_input @ params["residual_h_w"] + params["residual_h_b"]
    )
    residual_log_sigma = (residual_hidden @ params["residual_out_w"] + params["residual_out_b"])[0]
    return log_sigma_physics + residual_log_sigma, residual_log_sigma, transport_features


def molset_conductivity_s_m_unit_aware(
    params: Mapping[str, Any],
    species_props_norm: jnp.ndarray,
    species_props_raw: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
) -> jnp.ndarray:
    """Pure-JAX optimizer-facing unit-aware prototype conductivity in S/m."""

    log_sigma, _residual, _features = forward_unit_aware_molset(
        params=params,
        species_props_norm=species_props_norm,
        species_props_raw=species_props_raw,
        solvent_volume_fraction=solvent_volume_fraction,
        salt_molarity=salt_molarity,
        additive_weight_fraction=additive_weight_fraction,
        mask=mask,
        temperature_K=temperature_K,
    )
    return jnp.exp(log_sigma) * _MS_CM_TO_S_M


def _compute_role_aware_transport_features(
    raw_props: jnp.ndarray,
    solvent_volume_fraction: jnp.ndarray,
    salt_molarity: jnp.ndarray,
    additive_weight_fraction: jnp.ndarray,
    mask: jnp.ndarray,
    temperature_K: jnp.ndarray,
    params: Mapping[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mw = raw_props[:, IDX_MW]
    density = jnp.maximum(raw_props[:, IDX_DENSITY], 0.05)
    epsilon = raw_props[:, IDX_EPSILON]
    viscosity = jnp.maximum(raw_props[:, IDX_VISCOSITY], 0.05)
    lambda0 = raw_props[:, IDX_LAMBDA0]
    anion_radius = raw_props[:, IDX_ANION_R]
    binding = raw_props[:, IDX_BINDING]
    coord_affinity = raw_props[:, IDX_COORD_AFFINITY]
    jones_dole = raw_props[:, IDX_JONES_DOLE]
    steric = raw_props[:, IDX_STERIC]
    k_dimer = raw_props[:, IDX_DIMERIZATION]
    dielectric_decrement = raw_props[:, IDX_DIELECTRIC_DECREMENT]
    stokes_alpha = raw_props[:, IDX_STOKES_ALPHA_ANION]
    cation_radius = raw_props[:, IDX_CATION_R]
    ion_pair_kd = raw_props[:, IDX_ION_PAIR_KD]
    residence_time = raw_props[:, IDX_RESIDENCE_TIME]

    solvent_sum = jnp.sum(solvent_volume_fraction)
    solvent_phi = solvent_volume_fraction / jnp.maximum(solvent_sum, NUMERICAL_EPS)
    additive_total_wt = jnp.sum(additive_weight_fraction)
    non_additive_mass_fraction = jnp.maximum(1.0 - additive_total_wt, NUMERICAL_EPS)

    salt_mass_g = salt_molarity * mw
    salt_volume_ml = salt_mass_g / density
    total_salt_mass_g = jnp.sum(salt_mass_g)
    salt_volume_fraction = jnp.sum(salt_volume_ml) / LITER_TO_ML
    additive_volume_per_total_mass = jnp.sum(additive_weight_fraction / density)
    solvent_blend_density = jnp.sum(solvent_phi * density)
    solvent_total_volume_ml = (
        LITER_TO_ML
        - jnp.sum(salt_volume_ml)
        - total_salt_mass_g * additive_volume_per_total_mass / non_additive_mass_fraction
    ) / (
        1.0
        + solvent_blend_density * additive_volume_per_total_mass / non_additive_mass_fraction
    )
    solvent_volumes_ml = solvent_phi * solvent_total_volume_ml
    solvent_masses_g = solvent_volumes_ml * density
    total_solvent_mass_g = jnp.sum(solvent_masses_g)
    total_mass_g = (total_solvent_mass_g + total_salt_mass_g) / non_additive_mass_fraction

    additive_masses_g = additive_weight_fraction * total_mass_g
    additive_volumes_ml = additive_masses_g / density
    additive_moles = additive_masses_g / jnp.maximum(mw, NUMERICAL_EPS)
    ionic_additive_mask = jnp.where((additive_weight_fraction > 0.0) & (lambda0 > 0.0), 1.0, 0.0)
    neutral_additive_mask = jnp.where((additive_weight_fraction > 0.0) & (lambda0 <= 0.0), 1.0, 0.0)
    ionic_additive_molarity = additive_moles * ionic_additive_mask
    ionic_source_molarity = salt_molarity + ionic_additive_molarity
    total_ionic_source_M = jnp.sum(ionic_source_molarity)

    neutral_volumes_ml = solvent_volumes_ml + additive_volumes_ml * neutral_additive_mask
    neutral_volume_total_ml = jnp.maximum(jnp.sum(neutral_volumes_ml), NUMERICAL_EPS)
    neutral_phi = neutral_volumes_ml / neutral_volume_total_ml
    neutral_additive_volume_fraction = jnp.sum(additive_volumes_ml * neutral_additive_mask) / neutral_volume_total_ml

    ln_eta_liquid = jnp.sum(neutral_phi * jnp.log(viscosity))
    eta_liquid = jnp.exp(ln_eta_liquid)
    neutral_moles = solvent_volumes_ml * density / jnp.maximum(mw, NUMERICAL_EPS) + additive_moles * neutral_additive_mask
    neutral_concentration_M = neutral_moles
    dimer_fraction = k_dimer * neutral_concentration_M / (1.0 + k_dimer * neutral_concentration_M)
    shell_strength = coord_affinity * neutral_concentration_M
    shell_strength_sum = jnp.maximum(jnp.sum(shell_strength), NUMERICAL_EPS)
    shell_fraction = shell_strength / shell_strength_sum
    shell_steric = jnp.sum(shell_fraction * steric)
    shell_dimer_fraction = jnp.sum(shell_fraction * dimer_fraction)
    cation_solvation_strength = jnp.sum(shell_strength)
    additive_shell_strength = jnp.sum(shell_strength * neutral_additive_mask)
    additive_shell_fraction = additive_shell_strength / shell_strength_sum
    neutral_shell_persistence = jnp.sum(shell_fraction * residence_time)
    dimer_viscosity_factor = 1.0 + params["dimer_viscosity_scale"] * jnp.sum(neutral_phi * dimer_fraction)

    source_weight_sum = jnp.maximum(total_ionic_source_M, NUMERICAL_EPS)
    source_weight = ionic_source_molarity / source_weight_sum
    ionic_source_diversity = 1.0 - jnp.sum(source_weight * source_weight)
    mean_lambda0 = jnp.sum(source_weight * lambda0)
    mean_anion_radius = jnp.sum(source_weight * anion_radius)
    mean_cation_radius = jnp.sum(source_weight * cation_radius)
    mean_binding = jnp.sum(source_weight * binding)
    mean_jones_dole = jnp.sum(source_weight * jones_dole)
    mean_ion_pair_kd = jnp.sum(source_weight * ion_pair_kd)
    mean_dielectric_decrement = jnp.sum(ionic_source_molarity * dielectric_decrement)
    mean_stokes_alpha = jnp.sum(source_weight * stokes_alpha)
    salt_viscosity_factor = jnp.exp(params["salt_viscosity_scale"] * mean_jones_dole * total_ionic_source_M)

    eta_solution = eta_liquid * salt_viscosity_factor * dimer_viscosity_factor
    epsilon_liquid = jnp.sum(neutral_phi * epsilon)
    epsilon_effective = epsilon_liquid * (1.0 - mean_dielectric_decrement)
    epsilon_effective = jnp.maximum(epsilon_effective, 1.0)

    dielectric_pairing_factor = params["association_dielectric_scale"] / (
        epsilon_effective + params["association_dielectric_scale"]
    )
    association_drive = (
        params["pairing_strength"]
        * total_ionic_source_M
        * (1.0 + params["association_binding_scale"] * mean_binding)
        * dielectric_pairing_factor
        / (1.0 + jnp.maximum(mean_ion_pair_kd, 0.0))
    )
    contact_pair_fraction = association_drive / (1.0 + association_drive)
    free_ion_fraction = 1.0 / (1.0 + association_drive)

    anion_flex = _anion_flexibility(mean_anion_radius)
    dielectric_support = epsilon_effective / (epsilon_effective + jnp.maximum(mean_binding, 1.0))
    crowding_state = (
        params["crowding_salt_scale"] * total_ionic_source_M * mean_jones_dole
        + params["crowding_additive_scale"] * neutral_additive_volume_fraction
        + params["crowding_dimer_scale"] * shell_dimer_fraction
    )
    ion_network_state = (
        params["network_flex_scale"]
        * anion_flex
        * total_ionic_source_M
        * (1.0 + mean_dielectric_decrement)
        / (1.0 + mean_ion_pair_kd)
    )
    mixed_anion_relief_drive = (
        params["mixed_anion_relief_scale"]
        * ionic_source_diversity
        * dielectric_support
        / (1.0 + crowding_state)
    )
    relief_raw = (
        params["anticorrelation_relief_scale"]
        * shell_steric
        * anion_flex
        * dielectric_support
    )
    relief_damping = (
        1.0
        + params["relief_crowding_damping_scale"] * neutral_additive_volume_fraction
        + params["relief_additive_quadratic_damping_scale"]
        * neutral_additive_volume_fraction
        * neutral_additive_volume_fraction
        + crowding_state
        + params["relief_dimer_damping_scale"] * shell_dimer_fraction
    )
    anticorrelation_relief_drive = relief_raw / relief_damping

    shell_persistence_factor = neutral_shell_persistence / (1.0 + neutral_shell_persistence)
    anticorrelation_drive = (
        params["ca_anticorrelation_scale"]
        * (
            contact_pair_fraction
            + params["network_anticorrelation_scale"] * ion_network_state
        )
        * (1.0 + params["crowding_anticorrelation_scale"] * crowding_state)
        * (1.0 + shell_persistence_factor)
        / (1.0 + anticorrelation_relief_drive + mixed_anion_relief_drive)
    )
    cation_anion_distinct_fraction = -anticorrelation_drive / (1.0 + anticorrelation_drive)

    like_ion_drive = (
        params["like_ion_correlation_scale"]
        * ion_network_state
        * free_ion_fraction
        / (1.0 + crowding_state)
    )
    like_ion_distinct_fraction = like_ion_drive / (1.0 + like_ion_drive)

    cluster_drift_drive = (
        params["cluster_drift_scale"]
        * contact_pair_fraction
        * anion_flex
        * dielectric_support
        / (1.0 + crowding_state + shell_dimer_fraction)
    )
    cluster_drift_fraction = cluster_drift_drive / (1.0 + cluster_drift_drive)

    viscosity_mobility_factor = (
        params["reference_liquid_viscosity_cP"] / jnp.maximum(eta_solution, NUMERICAL_EPS)
    ) ** params["liquid_viscosity_exponent"]
    cation_mobility_weight = 1.0 / jnp.maximum(mean_cation_radius + 0.25 * cation_solvation_strength, NUMERICAL_EPS)
    anion_mobility_weight = 1.0 / jnp.maximum(
        mean_anion_radius * (1.0 + mean_stokes_alpha * crowding_state),
        NUMERICAL_EPS,
    )
    mobility_weight_sum = jnp.maximum(cation_mobility_weight + anion_mobility_weight, NUMERICAL_EPS)
    cation_lambda0 = mean_lambda0 * cation_mobility_weight / mobility_weight_sum
    anion_lambda0 = mean_lambda0 * anion_mobility_weight / mobility_weight_sum

    cation_self_conductivity = (
        params["conductivity_scale"]
        * total_ionic_source_M
        * cation_lambda0
        * free_ion_fraction
        * viscosity_mobility_factor
    )
    anion_self_conductivity = (
        params["conductivity_scale"]
        * total_ionic_source_M
        * anion_lambda0
        * free_ion_fraction
        * viscosity_mobility_factor
    )
    sigma_self = cation_self_conductivity + anion_self_conductivity
    cation_anion_distinct = sigma_self * cation_anion_distinct_fraction
    like_ion_distinct = sigma_self * like_ion_distinct_fraction
    cluster_drift = sigma_self * cluster_drift_fraction
    current_correlation_integral = (
        sigma_self
        + cation_anion_distinct
        + like_ion_distinct
        + cluster_drift
    )
    sigma_physics = jnp.maximum(current_correlation_integral, 1e-9)
    log_sigma_physics = jnp.log(sigma_physics)

    features = jnp.asarray(
        [
            solvent_sum,
            total_ionic_source_M,
            additive_total_wt,
            neutral_additive_volume_fraction,
            jnp.sum(ionic_additive_molarity),
            solvent_total_volume_ml,
            total_mass_g / LITER_TO_ML,
            salt_volume_fraction,
            eta_liquid,
            eta_solution,
            dimer_viscosity_factor,
            salt_viscosity_factor,
            viscosity_mobility_factor,
            epsilon_liquid,
            epsilon_effective,
            mean_dielectric_decrement,
            mean_lambda0,
            mean_cation_radius,
            mean_anion_radius,
            anion_flex,
            mean_jones_dole,
            mean_ion_pair_kd,
            mean_binding,
            mean_stokes_alpha,
            cation_solvation_strength,
            additive_shell_fraction,
            neutral_shell_persistence,
            association_drive,
            contact_pair_fraction,
            free_ion_fraction,
            crowding_state,
            ion_network_state,
            ionic_source_diversity,
            mixed_anion_relief_drive,
            shell_steric,
            shell_strength_sum,
            shell_dimer_fraction,
            anticorrelation_relief_drive,
            anticorrelation_drive,
            cation_self_conductivity,
            anion_self_conductivity,
            sigma_self,
            cation_anion_distinct_fraction,
            cation_anion_distinct,
            like_ion_distinct_fraction,
            like_ion_distinct,
            cluster_drift_fraction,
            cluster_drift,
            current_correlation_integral,
            sigma_physics,
        ]
    )
    return features, log_sigma_physics


def _weighted_pool(z: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    weight_sum = jnp.maximum(jnp.sum(weights), NUMERICAL_EPS)
    return jnp.sum(z * weights[:, None], axis=0) / weight_sum


def _property_similarity_bias(species_props_norm: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    props = species_props_norm[:, :D_BASE_INPUT]
    norm = jnp.maximum(jnp.sqrt(jnp.sum(props * props, axis=1, keepdims=True)), NUMERICAL_EPS)
    props_normed = props / norm
    return (props_normed @ props_normed.T) * mask[:, None] * mask[None, :]


def _anion_flexibility(mean_anion_radius: jnp.ndarray) -> jnp.ndarray:
    cutoff = _base.ANTICORR_R_CUTOFF_A
    reference = _base.ANTICORR_R_FLEX_REF_A
    scaled = (mean_anion_radius - cutoff) / jnp.maximum(reference - cutoff, NUMERICAL_EPS)
    return jnp.where(
        mean_anion_radius <= cutoff,
        0.0,
        jnp.where(mean_anion_radius >= reference, 1.0, scaled ** _base.ANTICORR_ALPHA_FLEX),
    )


@dataclass(frozen=True)
class _SpeciesRoleRow:
    name: str
    solvent_volume_fraction: float
    salt_molarity: float
    additive_weight_fraction: float


def _role_ordered_species(
    solvents: Mapping[str, Any],
    salts: Mapping[str, Any],
    additives: Mapping[str, Any],
) -> tuple[_SpeciesRoleRow, ...]:
    rows: list[_SpeciesRoleRow] = []
    for name, value in sorted(solvents.items()):
        rows.append(_SpeciesRoleRow(name, _nonnegative_float(value, f"solvents.{name}"), 0.0, 0.0))
    for name, value in sorted(salts.items()):
        rows.append(_SpeciesRoleRow(name, 0.0, _nonnegative_float(value, f"salts.{name}"), 0.0))
    for name, value in sorted(additives.items()):
        rows.append(_SpeciesRoleRow(name, 0.0, 0.0, _nonnegative_float(value, f"additives.{name}")))
    if not rows:
        raise ValueError("Recipe must contain at least one species")
    return tuple(rows)


def _validate_normalization(norm_mean: np.ndarray, norm_std: np.ndarray) -> None:
    if norm_mean.shape != (D_INPUT,):
        raise ValueError(f"norm_mean must have shape {(D_INPUT,)}, got {norm_mean.shape}")
    if norm_std.shape != (D_INPUT,):
        raise ValueError(f"norm_std must have shape {(D_INPUT,)}, got {norm_std.shape}")
    if np.any(~np.isfinite(norm_mean)):
        raise ValueError("norm_mean contains non-finite values")
    if np.any(~np.isfinite(norm_std)) or np.any(norm_std <= 0.0):
        raise ValueError("norm_std must contain positive finite values")


def _validate_role_channels(
    solvent_volume_fraction: np.ndarray,
    salt_molarity: np.ndarray,
    additive_weight_fraction: np.ndarray,
    mask: np.ndarray,
) -> None:
    if np.sum(mask) <= 0.0:
        raise ValueError("At least one active species is required")
    if np.sum(solvent_volume_fraction) <= 0.0:
        raise ValueError("At least one solvent volume fraction is required")
    additive_total = float(np.sum(additive_weight_fraction))
    if additive_total >= 1.0:
        raise ValueError(f"Total additive weight fraction must be below 1.0, got {additive_total}")


def _species_data(name: str) -> Mapping[str, Any]:
    if name in SOLVENTS:
        return SOLVENTS[name]
    if name in SALTS:
        return SALTS[name]
    if name in ADDITIVES:
        return ADDITIVES[name]
    raise ValueError(f"Unknown species {name}")


def _optional_numeric_species_property(species_data: Mapping[str, Any], key: str) -> float:
    if key not in species_data:
        return 0.0
    value = species_data[key]
    if value is None:
        return 0.0
    if not isinstance(value, (int, float)):
        raise ValueError(f"Species property {key} must be numeric when present, got {value!r}")
    return finite_float(float(value), f"species.{key}")


def _nonnegative_float(value: Any, context: str) -> float:
    try:
        parsed_raw = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric, got {value!r}") from exc
    parsed = finite_float(parsed_raw, context)
    if parsed < 0.0:
        raise ValueError(f"{context} must be non-negative, got {parsed}")
    return parsed
