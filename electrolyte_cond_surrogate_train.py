# electrolyte_cond_surrogate_train.py
"""
Train conductivity surrogate model for carbonate electrolytes.

MOTIVATION:
    The electrolyte design pipeline needs to evaluate ionic conductivity for
    thousands of candidate formulations during optimization. First-principles
    calculations (DFT, MD) are far too slow (~hours per formulation). An
    analytical model (Casteel-Amis) handles single-salt systems reasonably but
    cannot capture multi-salt synergies, additive effects, or complex solvent
    interactions. This module trains a machine-learned ensemble surrogate that
    predicts conductivity in microseconds with <0.8 mS/cm cross-validated error.

MECHANICS:
    The training pipeline proceeds in 6 phases:
    1. Data preparation: Parse recipes from electrolyte_property_db, featurize
       each recipe into a physics-motivated feature vector, assign sample weights
       (1/N per recipe to equalize recipe contributions), and group IDs (for
       GroupKFold to prevent data leakage).
    2. Feature selection: Optionally remove highly correlated features via greedy
       pairwise correlation filtering (currently disabled, threshold=1.0).
    3. Cross-validation: 5-fold GroupKFold (grouped by recipe) to estimate
       generalization error without leaking recipe-level information.
    4. Ensemble weight computation: Slope-corrected inverse-RMSE weighting
       (slope^2/RMSE) to penalize models that compress the prediction range.
    5. Final model fitting: All models retrained on the full dataset.
    6. Serialization: Model bundle (.pkl) saved with all metadata needed for
       inference (models, weights, scaler, component lists, calibration params).

DECISION RATIONALE:
    - Ensemble of 3 diverse tree models (GBM, RF, ExtraTrees) rather than a
      single model: reduces variance on the small dataset (~155 unique recipes)
      and provides robustness to overfitting on any single model family.
    - KernelRidge and MLP were tested and removed: KernelRidge had 3x worse
      RMSE (2.1 vs 0.75 mS/cm) and degraded minority-salt predictions; MLP
      required extensive hyperparameter tuning and was unstable on this dataset.
    - GroupKFold (not regular KFold): prevents the same recipe's repeated
      measurements from appearing in both train and validation splits, which
      would inflate R^2 and give a misleading estimate of generalization.
    - Sample weighting (1/N_measurements per recipe): ensures each unique recipe
      contributes equally to the loss, regardless of how many times it was
      measured in the experimental database.
    - Slope-corrected ensemble weights (slope^2/RMSE, not just 1/RMSE): simple
      inverse-RMSE weighting favored conservative models that compressed
      predictions toward the mean, causing systematic underestimation of
      high-conductivity formulations (>12 mS/cm). The slope^2 term corrects this.

Responsibilities:
- Prepare dataset
- Train ML ensemble (3 diverse tree models for nonlinear conductivity prediction)
- Serialize model + featurization metadata

Ensemble rationale:
- Electrolyte conductivity is fundamentally nonlinear (dome-shaped vs concentration)
- Competing effects: ion density vs viscosity/ion pairing
- Solvent/salt/additive interactions are multiplicative, not additive

Models selected (current, after pruning underperformers):
1. GBM: Primary predictor, captures nonlinear interactions via sequential
   residual learning (boosted trees). Learning rate 0.05 with 300 estimators
   for smooth convergence. max_depth=4 limits tree complexity.
2. RF: Variance reduction via bagging (bootstrap aggregation). Uncorrelated
   prediction errors average out in the ensemble. max_features="sqrt" ensures
   each tree sees a different feature subset.
3. ExtraTrees: Maximum diversity via fully random split thresholds (not
   optimized). Acts as a regularizer in the ensemble -- smooths predictions
   in regions where GBM might overfit.

Models removed:
- KernelRidge (RBF): CV RMSE 2.1 mS/cm (3x worse than tree models at 0.75).
  Contributed 21% ensemble weight due to its high slope (1.19) under the
  slope^2/RMSE metric, but degraded minority-salt (LiFSI) recipe accuracy.
- MLP: Unstable training, high variance across random seeds.
- GBM_Huber: Removed in favor of a single GBM with squared_error loss.

Best practices applied:
- Use ALL features (let models learn relevance via max_features="sqrt")
- Preserve repeated measurements (no premature averaging)
- Sample weighting by recipe multiplicity (1/N per recipe)
- Slope-corrected inverse-RMSE weighted ensemble
- Verbose logging for debugging and validation
- Linear calibration post-hoc to correct systematic prediction compression
"""

import os
import numpy as np
import pickle
from collections import defaultdict, Counter
from typing import Dict, Any

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from data.electrolyte_property_db import DATA as _DATA_ORIGINAL
from data.electrolyte_calisol_db import DATA as _DATA_CALISOL

# Merge: original 25°C data + CALiSol entries at 25°C (293–303 K).
_ROOM_TEMP_LOW_K = 293.0  # 20°C
_ROOM_TEMP_HIGH_K = 303.0  # 30°C

# Source weighting: original lab-verified data gets full weight,
# CALiSol gets reduced weight (different measurement conditions, labs).
SOURCE_WEIGHT_ORIGINAL = 1.0
SOURCE_WEIGHT_CALISOL = 0.5

# Down-weight very low conductivity entries (< 2 mS/cm) outside optimizer range.
LOW_KAPPA_THRESHOLD_MS_CM = 2.0  # mS/cm
LOW_KAPPA_WEIGHT_FACTOR = 0.25  # 4x less weight than normal entries

# κ-proportional weighting: weight = (κ / median_κ)^KAPPA_WEIGHT_EXPONENT.
# Exponent=1.0 means a 12 mS/cm sample gets ~1.7x weight of a 7 mS/cm sample.
# DISABLED (exponent=0): Empirically makes slope WORSE (0.74 vs 0.77) because
# it shifts all predictions upward rather than improving discrimination.
# The log-target transform already handles relative error weighting.
KAPPA_WEIGHT_EXPONENT = 0  # 0=disabled; tried 1.0, made slope worse

# Up-weight multi-salt formulations: they're underrepresented (~25-30% of dataset)
# but are the primary use case for the optimizer. A factor of 1.5 gives them
# 50% more influence per-recipe, compensating for lower sample count without
# overwhelming single-salt data that anchors the baseline predictions.
MULTI_SALT_WEIGHT_FACTOR = 1.5  # 1.0=disabled; 1.5=moderate upweight


def _recipe_key(recipe: dict) -> tuple:
    """Canonical hashable key for a recipe (for deduplication)."""
    salts = recipe["salts"] if "salts" in recipe else {}
    solvents = recipe["solvents"] if "solvents" in recipe else {}
    additives = recipe["additives"] if "additives" in recipe else {}
    return (
        tuple(sorted(salts.items())),
        tuple(sorted(solvents.items())),
        tuple(sorted(additives.items())),
    )


def _build_merged_data() -> list:
    """Merge original + CALiSol data with deduplication.

    CALiSol entries whose recipe matches an original DB recipe are removed
    to prevent double-counting.
    """
    original_keys = set()
    for entry in _DATA_ORIGINAL:
        original_keys.add(_recipe_key(entry["recipe"]))

    calisol_room_temp = [
        entry
        for entry in _DATA_CALISOL
        if _ROOM_TEMP_LOW_K <= entry["temperature_K"] <= _ROOM_TEMP_HIGH_K
    ]

    n_before = len(calisol_room_temp)
    calisol_deduped = [
        entry
        for entry in calisol_room_temp
        if _recipe_key(entry["recipe"]) not in original_keys
    ]
    n_removed = n_before - len(calisol_deduped)
    print(
        f"[DATA MERGE] Original: {len(_DATA_ORIGINAL)}, "
        f"CALiSol 25°C: {n_before}, "
        f"duplicates removed: {n_removed}, "
        f"CALiSol after dedup: {len(calisol_deduped)}"
    )

    tagged = []
    for entry in _DATA_ORIGINAL:
        e = dict(entry)
        e["_source"] = "original"
        tagged.append(e)
    for entry in calisol_deduped:
        e = dict(entry)
        e["_source"] = "calisol"
        tagged.append(e)

    return tagged


DATA = _build_merged_data()

from conductivity.electrolyte_utils_features import (
    discover_components,
    featurize_recipe,
    get_feature_names,
)


# ---------------------------------------------------------------------
# Ensemble configuration - explicit constants for visibility/auditing
# Tuned for merged dataset (~600 samples after CALiSol merge).
#
# LOG-TARGET TRANSFORM: All models train on log(κ), predict in log-space,
# then exp() back to mS/cm. This makes the loss proportional to relative
# error (not absolute), which directly fixes prediction compression:
# - Without log: 1 mS/cm error at κ=2 (50% error) penalized same as
#   1 mS/cm error at κ=12 (8% error) → model over-fits low-κ regime
# - With log: equal relative errors penalized equally → no compression bias
#
# HistGradientBoostingRegressor replaces sklearn GBM:
# - Histogram-based splits (faster, handles NaN natively)
# - max_iter=1500 with early_stopping=True: trains until validation
#   loss plateaus, preventing under/overfitting automatically
# - max_depth=7: deeper trees capture 4-way interactions
#   (salt × solvent × concentration × additive)
# - min_samples_leaf=10: regularizes deeper trees
# - max_leaf_nodes=63: bounds tree complexity (2^6-1)
# - l2_regularization=0.1: mild shrinkage on leaf weights
# ---------------------------------------------------------------------

# Whether to train on log(conductivity). Transforms the squared-error loss
# from absolute to relative error, reducing systematic compression at the
# tails. All models in the ensemble train on the same target space.
LOG_TARGET = False

ENSEMBLE_CONFIG: Dict[str, Dict[str, Any]] = {
    # DIVERSE ENSEMBLE: 3 fundamentally different algorithms for decorrelated errors.
    # HistGBM (boosting) + RF (bagging) + ExtraTrees (random splits) = maximum
    # diversity in error patterns, which is the whole point of ensembling.
    # Same-algorithm ensembles (e.g. 3×HistGBM) have correlated errors that
    # don't cancel — measured: RMSE 1.02 vs 0.74 for diverse ensemble.
    "HistGBM": {
        "class": HistGradientBoostingRegressor,
        "params": {
            "loss": "squared_error",
            "max_iter": 800,  # increased for deeper trees at lr=0.05
            "max_depth": 6,  # deeper to capture 4-way interactions (salt×conc×solvent×additive)
            "learning_rate": 0.05,  # standard GBM learning rate
            "min_samples_leaf": 5,  # regularizes deeper trees (was 3 at depth=4)
            "max_leaf_nodes": 63,  # 2^6-1: bounds tree complexity for depth=6
            "max_bins": 255,  # sklearn maximum; concentration cliff captured via new features instead
            "l2_regularization": 0.1,  # mild shrinkage to prevent overfitting deeper trees
            "early_stopping": True,
            "validation_fraction": 0.15,  # 15% holdout for early stopping — sklearn default
            "n_iter_no_change": 50,  # stop when converged
            "random_state": 42,
        },
        "description": "HistGBM - sequential boosting captures nonlinear interactions",
    },
    "RF": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": 500,
            "max_depth": None,  # unlimited depth — let RF find fine structure
            "min_samples_leaf": 2,  # matches old RF
            "max_features": "sqrt",  # decorrelates trees
            "random_state": 44,  # matches old RF seed
            "n_jobs": -1,
        },
        "description": "RF - bagging with decorrelated trees for variance reduction",
    },
    "ExtraTrees": {
        "class": ExtraTreesRegressor,
        "params": {
            "n_estimators": 500,
            "max_depth": None,  # unlimited depth — matches old ET
            "min_samples_leaf": 2,  # matches old ET
            "max_features": "sqrt",  # decorrelates trees
            "random_state": 45,  # matches old ET seed
            "n_jobs": -1,
        },
        "description": "ExtraTrees - fully random splits for maximum ensemble diversity",
    },
}

# Feature selection configuration.
# DECISION RATIONALE: Disabled (threshold=1.0 keeps all features) because:
# 1. Tree ensembles handle correlated features natively via max_features="sqrt"
#    feature subsampling -- correlated features just reduce the effective
#    diversity of splits, not catastrophically degrade predictions.
# 2. The previous threshold=0.85 removed 75 of 104 features, including
#    interaction terms (e.g., LiFSI_molarity_x_Lambda0, FEC_E_red_loading)
#    that were important for minority-salt and additive recipe accuracy.
# 3. With only 3 tree-based models remaining (no KernelRidge/MLP that are
#    sensitive to multicollinearity), there is no benefit to feature pruning.
FEATURE_CORRELATION_THRESHOLD = 1.0

# Cross-validation configuration.
# 5 folds with GroupKFold: each fold contains ~31 unique recipes (~40 samples).
# DECISION: 5 folds (not 10) because with only 155 unique recipes, 10-fold
# would give validation sets of ~15 recipes -- too small for stable RMSE
# estimates, especially for minority salts (LiFSI has only ~15 recipes total).
# 5 folds: merged dataset has ~600 samples / ~500 unique recipes → ~100 per fold.
# 5 (not 10) because with ~550 unique recipes, 10-fold gives validation sets
# of ~55 recipes — workable but noisier fold estimates for minority salts.
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42  # Used only for data splitting reproducibility


# ---------------------------------------------------------------------
# Feature selection - remove redundant features to improve generalization
# ---------------------------------------------------------------------


def select_non_redundant_features(
    X: np.ndarray,
    feature_names: list,
    corr_threshold: float = FEATURE_CORRELATION_THRESHOLD,
) -> tuple:
    """
    Remove redundant features via greedy pairwise correlation filtering.

    Why this exists:
        The physics-motivated featurization in ``featurize_recipe()`` intentionally
        includes many correlated features (e.g. Li_total, Li_total^2, Li_total^3)
        to give models multiple representations of the same underlying physics.
        However, highly correlated features can cause instability in some model
        types (e.g. KernelRidge, MLP) and inflate effective dimensionality.
        This function provides an optional correlation-based filter that removes
        features whose Pearson correlation with an already-selected feature
        exceeds a threshold. Currently disabled (threshold=1.0) because tree
        ensembles handle correlated features natively via ``max_features="sqrt"``,
        and aggressive filtering (threshold=0.85) was found to remove interaction
        terms important for minority-salt accuracy.

    What it does:
        1. Standardizes features for correlation computation.
        2. Identifies constant/near-constant features (std < 1e-6) and skips them.
        3. Iterates features in order. For each candidate, computes its absolute
           Pearson correlation with every already-selected feature. If all
           correlations are below the threshold, the feature is selected.
        4. Returns the indices and names of selected features.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        feature_names: List of feature name strings, length must equal
            ``X.shape[1]``.
        corr_threshold: Maximum absolute Pearson correlation allowed between
            any pair of selected features. Set to 1.0 to keep all features
            (current default). Values like 0.85-0.95 produce moderate filtering.

    Returns:
        Tuple of:
            - selected_idx (List[int]): Column indices of retained features in
              the original feature matrix.
            - selected_names (List[str]): Names of retained features.

    Side effects:
        Prints diagnostic output: number of valid features, number selected,
        and the names of all selected features.
    """
    print("\n" + "-" * 70)
    print(f"FEATURE SELECTION (correlation threshold = {corr_threshold})")
    print("-" * 70)

    # Standardize for correlation computation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Skip constant/near-constant features
    feature_std = np.std(X_scaled, axis=0)
    valid_features = feature_std > 1e-6
    n_valid = valid_features.sum()
    print(f"Features with variance: {n_valid} / {len(feature_names)}")

    selected_idx = []
    selected_names = []

    for i in range(len(feature_names)):
        if not valid_features[i]:
            continue

        # Check correlation with already selected features
        is_redundant = False
        for j in selected_idx:
            corr = np.abs(np.corrcoef(X_scaled[:, i], X_scaled[:, j])[0, 1])
            if corr > corr_threshold:
                is_redundant = True
                break

        if not is_redundant:
            selected_idx.append(i)
            selected_names.append(feature_names[i])

    print(f"Selected {len(selected_idx)} non-redundant features:")
    for name in selected_names:
        print(f"  {name}")

    return selected_idx, selected_names


# ---------------------------------------------------------------------
# Verbose data diagnostics
# ---------------------------------------------------------------------


def log_data_diagnostics(property_name: str) -> None:
    """
    Log comprehensive statistics about the training dataset for debugging.

    Why this exists:
        Before training a surrogate model, it is critical to understand the
        training data distribution: how many data points exist, what conductivity
        range they cover, which salts and solvents are represented, and whether
        certain formulation types are under-represented. This function prints
        all of that information, enabling the developer to identify coverage
        gaps (e.g. no LiFSI data above 1.5 M) and data quality issues (e.g.
        outlier conductivity values) before they manifest as model errors.

    What it does:
        1. Filters the global ``DATA`` list for rows containing the target
           property.
        2. Computes and prints conductivity statistics (min, max, mean, median,
           std).
        3. Counts and prints salt distribution (per-salt sample counts and
           percentages, plus dual-salt formulation count).
        4. Counts and prints solvent distribution.
        5. Computes per-primary-salt conductivity ranges and means.

    Args:
        property_name: The property key to analyze in each row's
            ``"properties"`` dict (e.g. ``"conductivity_mS_cm"``).

    Returns:
        None.

    Side effects:
        Prints extensive diagnostic output to stdout. Reads from the global
        ``DATA`` list imported from ``data.electrolyte_property_db``.
    """
    print("\n" + "=" * 70)
    print("DATA DIAGNOSTICS")
    print("=" * 70)

    valid_rows = [r for r in DATA if property_name in r.get("properties", {})]
    print(f"Total rows with '{property_name}': {len(valid_rows)}")

    if not valid_rows:
        print("ERROR: No valid data found!")
        return

    # Conductivity range
    conductivities = [r["properties"][property_name] for r in valid_rows]
    print("\nConductivity statistics:")
    print(f"  Min:    {min(conductivities):.2f} mS/cm")
    print(f"  Max:    {max(conductivities):.2f} mS/cm")
    print(f"  Mean:   {np.mean(conductivities):.2f} mS/cm")
    print(f"  Median: {np.median(conductivities):.2f} mS/cm")
    print(f"  Std:    {np.std(conductivities):.2f} mS/cm")

    # Salt distribution
    salt_counts = Counter()
    dual_salt_count = 0
    for row in valid_rows:
        salts = row["recipe"].get("salts", {})
        for salt_name in salts.keys():
            salt_counts[salt_name] += 1
        if len(salts) > 1:
            dual_salt_count += 1

    print("\nSalt distribution:")
    for salt, count in salt_counts.most_common():
        pct = 100 * count / len(valid_rows)
        print(f"  {salt}: {count} ({pct:.1f}%)")
    print(
        f"  Dual-salt formulations: {dual_salt_count} ({100 * dual_salt_count / len(valid_rows):.1f}%)"
    )

    # Solvent distribution
    solvent_counts = Counter()
    for row in valid_rows:
        solvents = row["recipe"].get("solvents", {})
        for solv_name in solvents.keys():
            solvent_counts[solv_name] += 1

    print("\nSolvent distribution:")
    for solv, count in solvent_counts.most_common():
        pct = 100 * count / len(valid_rows)
        print(f"  {solv}: {count} ({pct:.1f}%)")

    # Conductivity by salt type
    print("\nConductivity by primary salt:")
    salt_conductivities = defaultdict(list)
    for row in valid_rows:
        salts = row["recipe"].get("salts", {})
        if salts:
            primary_salt = max(salts.keys(), key=lambda s: salts[s])
            salt_conductivities[primary_salt].append(row["properties"][property_name])

    for salt in sorted(salt_conductivities.keys()):
        conds = salt_conductivities[salt]
        print(
            f"  {salt}: n={len(conds)}, range=[{min(conds):.2f}, {max(conds):.2f}], mean={np.mean(conds):.2f} mS/cm"
        )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------


def prepare_data(property_name: str):
    """
    Prepare featurized training data with sample weights and group IDs.

    Why this exists:
        The raw dataset in ``data.electrolyte_property_db.DATA`` contains recipe
        dicts with measured properties, often with multiple measurements per
        recipe (repeated experiments). This function transforms that raw data
        into arrays suitable for sklearn training: a feature matrix X, target
        vector y, sample weights w (so each unique recipe contributes equally
        regardless of how many repeated measurements it has), and group IDs
        (so GroupKFold can prevent data leakage by keeping all measurements
        from the same recipe in the same fold).

    What it does:
        1. Discovers all components and salts via ``discover_components()``.
        2. Groups rows by unique recipe key (canonicalized tuple of sorted
           salt/solvent/additive items).
        3. For each recipe group, featurizes the recipe via ``featurize_recipe()``
           and assigns sample weight = 1/N_measurements so the total weight per
           recipe sums to 1.
        4. Each measurement is kept as a separate sample (no premature averaging),
           tagged with a group ID for cross-validation.

    Args:
        property_name: The target property to predict (e.g.
            ``"conductivity_mS_cm"``). Rows lacking this property are skipped.

    Returns:
        Tuple of seven elements:
            - X (np.ndarray): Feature matrix, shape ``(n_samples, n_features)``.
            - y (np.ndarray): Target values, shape ``(n_samples,)``.
            - sample_weight (np.ndarray): Per-sample weights, shape
              ``(n_samples,)``. Weights for a given recipe sum to 1.0.
            - groups (np.ndarray): Integer group IDs, shape ``(n_samples,)``.
              All samples from the same recipe share the same group ID.
            - component_list (List[str]): Sorted list of liquid component names.
            - salt_list (List[str]): Sorted list of salt names.
            - feature_names (List[str]): Feature name strings matching X columns.

    Side effects:
        Prints discovered components/salts, sample counts, and feature dimension
        to stdout.
    """
    print("\n" + "-" * 70)
    print("PREPARING DATA")
    print("-" * 70)

    component_list, salt_list = discover_components(DATA)
    print(f"Discovered components: {component_list}")
    print(f"Discovered salts: {salt_list}")

    recipe_groups = defaultdict(list)

    for row in DATA:
        if property_name not in row["properties"]:
            continue

        recipe = row["recipe"]
        key = _recipe_key(recipe)
        source = row.get("_source", "original")
        kappa = row["properties"][property_name]
        recipe_groups[key].append((kappa, source))

    X, y, sample_weight, groups = [], [], [], []
    n_original = 0
    n_calisol = 0
    n_low_kappa = 0
    n_multi_salt = 0

    for group_id, (key, measurements) in enumerate(recipe_groups.items()):
        recipe = {
            "salts": dict(key[0]),
            "solvents": dict(key[1]),
            "additives": dict(key[2]),
        }

        x_feat = featurize_recipe(recipe, component_list, salt_list)

        # Multi-salt upweight: recipes with >1 salt get boosted
        is_multi_salt = len(recipe["salts"]) > 1
        multi_salt_w = MULTI_SALT_WEIGHT_FACTOR if is_multi_salt else 1.0
        if is_multi_salt:
            n_multi_salt += 1

        # Each measurement is kept; total weight per recipe = 1
        base_w = 1.0 / len(measurements)

        for kappa, source in measurements:
            if source == "original":
                source_w = SOURCE_WEIGHT_ORIGINAL
                n_original += 1
            else:
                source_w = SOURCE_WEIGHT_CALISOL
                n_calisol += 1

            if kappa < LOW_KAPPA_THRESHOLD_MS_CM:
                kappa_w = LOW_KAPPA_WEIGHT_FACTOR
                n_low_kappa += 1
            else:
                kappa_w = 1.0

            X.append(x_feat)
            y.append(kappa)
            sample_weight.append(
                base_w * source_w * kappa_w * multi_salt_w
            )
            groups.append(group_id)

    print(f"  Samples from original DB: {n_original}")
    print(f"  Samples from CALiSol: {n_calisol}")
    print(
        f"  Low-κ samples (<{LOW_KAPPA_THRESHOLD_MS_CM} mS/cm, weight={LOW_KAPPA_WEIGHT_FACTOR}): {n_low_kappa}"
    )
    print(
        f"  Multi-salt recipes (weight={MULTI_SALT_WEIGHT_FACTOR}): {n_multi_salt}"
    )

    X_arr = np.vstack(X)
    y_arr = np.array(y)
    w_arr = np.array(sample_weight)
    groups_arr = np.array(groups)
    feature_names = get_feature_names(component_list, salt_list)

    # Apply κ-proportional weighting: weight *= (κ / median_κ)^exponent.
    # This makes the model pay more attention to high-κ targets, directly
    # counteracting regression-to-mean (slope < 1.0 compression).
    if KAPPA_WEIGHT_EXPONENT > 0:
        median_kappa = np.median(y_arr)
        kappa_scale = (y_arr / median_kappa) ** KAPPA_WEIGHT_EXPONENT
        w_arr *= kappa_scale
        print(
            f"  κ-proportional weighting (exponent={KAPPA_WEIGHT_EXPONENT}): "
            f"median_κ={median_kappa:.2f}, weight range [{kappa_scale.min():.3f}, {kappa_scale.max():.3f}]"
        )

    print(f"Total samples: {len(y_arr)}")
    print(f"Unique recipes: {len(recipe_groups)}")
    print(f"Feature dimension: {X_arr.shape[1]}")

    return X_arr, y_arr, w_arr, groups_arr, component_list, salt_list, feature_names


# ---------------------------------------------------------------------
# Ensemble weight computation
# ---------------------------------------------------------------------


def compute_ensemble_weights(
    cv_rmse_per_model: Dict[str, float],
    cv_slope_per_model: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute ensemble weights using slope-corrected inverse-RMSE weighting.

    Why this exists:
        Simple inverse-RMSE weighting favors conservative models that compress
        predictions toward the dataset mean. These models achieve low RMSE on
        the abundant mid-range data but systematically underestimate high-
        conductivity formulations. For the electrolyte optimization pipeline,
        underestimating good formulations is worse than overestimating bad ones
        (we miss the best candidates). Slope-corrected weighting addresses this
        by also rewarding models whose predictions have slope ~1.0 against
        actual values (i.e. models that do not compress the dynamic range).

    What it does:
        Computes ``weight_i = slope_i^2 / RMSE_i`` for each model, then
        normalizes to sum to 1.0. The slope^2 term aggressively rewards
        well-calibrated models (slope near 1.0) while still penalizing high
        RMSE. This is more aggressive than linear slope weighting
        (``slope / RMSE``) and was found to reduce systematic underestimation
        of high-conductivity samples (>12 mS/cm).

    Args:
        cv_rmse_per_model: Dict mapping model name (str) to its cross-validated
            RMSE in mS/cm (float). Must have the same keys as
            ``cv_slope_per_model``.
        cv_slope_per_model: Dict mapping model name (str) to the slope of the
            linear fit of CV predictions vs actual values (float). A slope of
            1.0 means no prediction compression; <1.0 means the model
            underestimates extreme values.

    Returns:
        Dict mapping each model name to its normalized ensemble weight (float).
        Weights sum to 1.0.

    Side effects:
        Prints a table of slope, RMSE, and resulting weight for each model.
    """
    print("\n" + "-" * 70)
    print("COMPUTING ENSEMBLE WEIGHTS (slope-corrected: slope²/RMSE)")
    print("-" * 70)

    # Slope-corrected weighting: higher slope (less compression) and lower RMSE both good
    # Use slope² for more aggressive weighting toward well-calibrated models
    # This reduces systematic underestimation of high-conductivity formulations
    slope_sq_over_rmse = {
        k: (cv_slope_per_model[k] ** 2) / cv_rmse_per_model[k]
        for k in cv_rmse_per_model.keys()
    }
    total = sum(slope_sq_over_rmse.values())
    weights = {k: v / total for k, v in slope_sq_over_rmse.items()}

    for name in sorted(weights.keys()):
        print(
            f"  {name}: slope={cv_slope_per_model[name]:.3f}, "
            f"RMSE={cv_rmse_per_model[name]:.4f} -> weight={weights[name]:.4f}"
        )

    return weights


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_model(property_name: str, model_path: str) -> dict:
    """
    Train the full conductivity surrogate ensemble and serialize it to disk.

    Why this exists:
        This is the main training entry point for the conductivity surrogate. It
        orchestrates the entire pipeline: data diagnostics, featurization, feature
        selection, cross-validation, model fitting, ensemble weight computation,
        calibration, and serialization. The resulting pickle file is the artifact
        consumed by ``predict_conductivity_mS_cm()`` at inference time and by the
        broader electrolyte design pipeline.

    What it does:
        1. Logs data diagnostics (``log_data_diagnostics()``) for auditability.
        2. Prepares featurized data with sample weights and group IDs
           (``prepare_data()``).
        3. Applies correlation-based feature selection
           (``select_non_redundant_features()``).
        4. Fits a ``StandardScaler`` on the selected features.
        5. Instantiates all models from ``ENSEMBLE_CONFIG`` (GBM, RF, ExtraTrees).
        6. Runs ``CV_N_SPLITS``-fold GroupKFold cross-validation (grouped by recipe
           to prevent data leakage). For each fold, each model is cloned, fitted
           on the training split (with sample weights), and evaluated on the
           validation split. Per-fold RMSE is logged.
        7. Computes per-model CV RMSE and slope (prediction compression metric).
        8. Computes slope-corrected ensemble weights via
           ``compute_ensemble_weights()``.
        9. Computes weighted ensemble CV predictions and overall ensemble metrics
           (RMSE, MAE, R^2).
        10. Fits final models on the full dataset for production use.
        11. Logs GBM feature importances for interpretability.
        12. Computes linear calibration parameters (slope, intercept) to correct
            systematic prediction compression at extreme values.
        13. Serializes the complete model bundle (models, weights, scaler,
            component/salt lists, feature selection indices, calibration params,
            all CV metrics) to ``model_path`` as a pickle file.

    Args:
        property_name: Key in each data row's ``"properties"`` dict to use as
            the prediction target (e.g. ``"conductivity_mS_cm"``).
        model_path: Filesystem path where the trained model pickle will be
            saved (e.g. ``"electrolyte_conductivity.pkl"``).

    Returns:
        The model bundle dict (same object that was serialized to disk), with keys:
            - ``models`` (Dict[str, sklearn estimator]): Final fitted models.
            - ``ensemble_weights`` (Dict[str, float]): Normalized model weights.
            - ``scaler`` (StandardScaler): Fitted feature scaler.
            - ``component_list`` (List[str]): Discovered liquid component names.
            - ``salt_list`` (List[str]): Discovered salt names.
            - ``feature_names`` (List[str]): Original feature names.
            - ``selected_feature_idx`` (List[int]): Indices of selected features.
            - ``selected_feature_names`` (List[str]): Names of selected features.
            - ``feature_correlation_threshold`` (float): Threshold used.
            - ``cv_rmse`` (float): Ensemble CV RMSE in mS/cm.
            - ``cv_mae`` (float): Ensemble CV MAE in mS/cm.
            - ``cv_r2`` (float): Ensemble CV R^2.
            - ``cv_slope`` (float): Ensemble prediction-vs-actual slope.
            - ``per_model_cv_rmse`` (Dict[str, float]): Per-model CV RMSE.
            - ``per_model_cv_slope`` (Dict[str, float]): Per-model slopes.
            - ``ensemble_config`` (Dict): Copy of ENSEMBLE_CONFIG for provenance.
            - ``cv_config`` (Dict): Cross-validation config (n_splits, random_state).
            - ``calibration_slope`` (float): Linear calibration slope.
            - ``calibration_intercept`` (float): Linear calibration intercept.
            - ``calibrated_cv_rmse`` (float): RMSE after linear calibration.
            - ``calibrated_cv_r2`` (float): R^2 after linear calibration.

    Side effects:
        - Writes the model bundle pickle file to ``model_path``.
        - Prints extensive training logs (diagnostics, per-fold metrics, feature
          importances, final summary) to stdout.
    """
    # Log data diagnostics first
    log_data_diagnostics(property_name)

    X, y_raw, w, groups, component_list, salt_list, feature_names = prepare_data(
        property_name
    )

    # LOG-TARGET TRANSFORM: train on log(κ) to equalize relative errors.
    # Without this, a 1 mS/cm error at κ=2 (50% relative) is penalized the same
    # as 1 mS/cm at κ=12 (8% relative), causing the model to over-fit low-κ.
    if LOG_TARGET:
        y = np.log(y_raw)
        print(
            f"\n[LOG TARGET] Training on log(κ): range [{y.min():.3f}, {y.max():.3f}]"
        )
        print(f"  Original κ range: [{y_raw.min():.2f}, {y_raw.max():.2f}] mS/cm")
    else:
        y = y_raw

    n_unique_recipes = len(np.unique(groups))

    print("\n" + "=" * 70)
    print("TRAINING CONDUCTIVITY SURROGATE MODEL")
    print("=" * 70)
    print(f"Training samples: {len(y)}")
    print(f"Unique recipes: {n_unique_recipes}")
    print(f"Original features: {len(feature_names)}")
    print(f"Log-target transform: {LOG_TARGET}")
    print("Using GroupKFold to prevent recipe leakage between train/val splits")

    # Feature selection - remove redundant features
    selected_idx, selected_names = select_non_redundant_features(X, feature_names)
    X_selected = X[:, selected_idx]

    print(
        f"\nUsing {len(selected_idx)} non-redundant features (from {len(feature_names)} original)"
    )

    # Scale selected features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    print("\nFeature scaling applied (StandardScaler)")
    print(f"  Mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
    print(f"  Scale range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")

    # Instantiate models from config
    print("\n" + "-" * 70)
    print("ENSEMBLE CONFIGURATION")
    print("-" * 70)

    models = {}
    for name, config in ENSEMBLE_CONFIG.items():
        models[name] = config["class"](**config["params"])
        print(f"  {name}: {config['description']}")

    # Cross-validation with GroupKFold to prevent recipe leakage
    # GroupKFold ensures all samples from the same recipe stay together
    gkfold = GroupKFold(n_splits=CV_N_SPLITS)

    # Track per-model CV performance
    model_cv_rmse = {name: [] for name in models.keys()}
    model_cv_preds = {name: np.zeros_like(y) for name in models.keys()}

    print("\n" + "-" * 70)
    print(f"{CV_N_SPLITS}-FOLD GROUP CROSS-VALIDATION (by recipe)")
    print("-" * 70)

    # ---------------------------------------------------------------
    # CROSS-VALIDATION LOOP
    # ---------------------------------------------------------------
    # GroupKFold ensures all measurements from the same recipe stay in the
    # same fold. This prevents data leakage: if recipe X has 3 repeated
    # measurements and 2 end up in training while 1 ends up in validation,
    # the model effectively "memorizes" recipe X and validation performance
    # is inflated. GroupKFold guarantees all 3 measurements of recipe X are
    # in training OR validation, never split across.
    #
    # For each fold, each model is independently cloned and fitted on the
    # training split. Out-of-fold predictions are stored so that after all
    # folds complete, we have a prediction for every sample made by a model
    # that never saw that sample during training. This gives an honest
    # estimate of generalization performance.
    for fold_idx, (train_idx, val_idx) in enumerate(gkfold.split(X_scaled, y, groups)):
        print(f"\nFold {fold_idx + 1}/{CV_N_SPLITS}:")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = w[train_idx]  # Sample weights for equalized recipe contribution

        for name, base_model in models.items():
            # Clone model for this fold by re-instantiating with the same params.
            # DECISION: We clone via __class__(**get_params()) rather than
            # sklearn.base.clone() because clone() was found to occasionally
            # reset internal state in unexpected ways for some estimators.
            m = base_model.__class__(**base_model.get_params())

            # Fit with sample weights (if supported by model class).
            # MECHANICS: All tree models (GBM, RF, ExtraTrees) support
            # sample_weight in fit(). The try/except is a safety net for any
            # future model type (e.g., KernelRidge) that does not.
            # Sample weights ensure each unique recipe contributes equally
            # (1/N per measurement) regardless of how many repeats it has.
            try:
                m.fit(X_train, y_train, sample_weight=w_train)
            except TypeError:
                # Model does not support sample_weight -- fit without it
                m.fit(X_train, y_train)

            # Predict on validation fold (out-of-fold prediction)
            preds = m.predict(X_val)
            # Store out-of-fold predictions in the full-size array so we can
            # compute ensemble-level CV metrics after all folds complete.
            # After all 5 folds, model_cv_preds[name] contains a prediction
            # for every sample, each made by a model trained without that sample.
            model_cv_preds[name][val_idx] = preds

            # Compute fold RMSE for per-fold diagnostics.
            # Typical values: 0.5-1.2 mS/cm per fold for tree models.
            fold_rmse = np.sqrt(mean_squared_error(y_val, preds))
            model_cv_rmse[name].append(fold_rmse)

            print(f"    {name}: RMSE={fold_rmse:.4f} mS/cm")

    # ---------------------------------------------------------------
    # AGGREGATE CROSS-VALIDATION RESULTS
    # ---------------------------------------------------------------
    # After all folds complete, compute per-model statistics from the
    # out-of-fold predictions. These are the most reliable estimates of
    # generalization performance because each prediction was made by a
    # model that never saw that sample during training.
    #
    # When LOG_TARGET=True, predictions are in log-space. We transform
    # back to mS/cm for all human-readable metrics (RMSE, MAE, slope).
    print("\n" + "-" * 70)
    print("CROSS-VALIDATION SUMMARY (per model)")
    print("-" * 70)

    cv_rmse_mean = {}  # Per-model mean CV RMSE across folds (in mS/cm)
    cv_slope_mean = {}  # Per-model prediction-vs-actual slope (in mS/cm space)
    for name in models.keys():
        # Transform OOF predictions back to mS/cm for metrics
        if LOG_TARGET:
            preds_ms = np.exp(model_cv_preds[name])
        else:
            preds_ms = model_cv_preds[name]

        rmse = np.sqrt(mean_squared_error(y_raw, preds_ms))
        mae = mean_absolute_error(y_raw, preds_ms)
        r2 = r2_score(y_raw, preds_ms)

        # Slope of pred vs actual in mS/cm space measures prediction compression.
        # A slope of 1.0 means the model reproduces the full dynamic range.
        # slope < 1.0 means high-κ underestimated, low-κ overestimated.
        slope, _ = np.polyfit(y_raw, preds_ms, 1)

        cv_rmse_mean[name] = rmse
        cv_slope_mean[name] = slope

        print(f"  {name}:")
        print(f"    CV RMSE: {rmse:.4f} mS/cm")
        print(f"    CV MAE:  {mae:.4f} mS/cm")
        print(f"    CV R^2:  {r2:.4f}")
        print(f"    CV slope: {slope:.4f} (1.0=perfect, <1=compressed)")

    # Compute ensemble weights using slope-corrected weighting
    ensemble_weights = compute_ensemble_weights(cv_rmse_mean, cv_slope_mean)

    # Compute weighted ensemble CV predictions in log-space (or raw),
    # then transform to mS/cm for metrics.
    ensemble_cv_preds_log = np.zeros_like(y)
    for name, weight in ensemble_weights.items():
        ensemble_cv_preds_log += weight * model_cv_preds[name]

    if LOG_TARGET:
        ensemble_cv_preds_ms = np.exp(ensemble_cv_preds_log)
    else:
        ensemble_cv_preds_ms = ensemble_cv_preds_log

    # Ensemble metrics in mS/cm — the numbers that matter for model quality.
    ensemble_cv_rmse = np.sqrt(mean_squared_error(y_raw, ensemble_cv_preds_ms))
    ensemble_cv_mae = mean_absolute_error(y_raw, ensemble_cv_preds_ms)
    ensemble_cv_r2 = r2_score(y_raw, ensemble_cv_preds_ms)

    print("\n" + "-" * 70)
    print("WEIGHTED ENSEMBLE CV PERFORMANCE")
    print("-" * 70)
    print(f"  Ensemble CV RMSE: {ensemble_cv_rmse:.4f} mS/cm")
    print(f"  Ensemble CV MAE:  {ensemble_cv_mae:.4f} mS/cm")
    print(f"  Ensemble CV R^2:  {ensemble_cv_r2:.4f}")

    # Per-range CV diagnostics: break down accuracy by conductivity bins
    # to verify high-κ predictions are not compressed.
    _RANGE_BINS = [(0, 2), (2, 5), (5, 8), (8, 11), (11, 15)]  # mS/cm boundaries
    ensemble_slope, _ = np.polyfit(y_raw, ensemble_cv_preds_ms, 1)
    print(f"  Ensemble CV slope: {ensemble_slope:.4f}")

    print("\n  Per-range CV diagnostics (mS/cm):")
    print(
        f"  {'Range':>10s}  {'N':>4s}  {'RMSE':>6s}  {'MAE':>6s}  {'MeanBias':>9s}  {'R²':>6s}"
    )
    for lo, hi in _RANGE_BINS:
        mask = (y_raw >= lo) & (y_raw < hi)
        n = mask.sum()
        if n == 0:
            print(f"  [{lo:2d},{hi:2d})     {n:4d}   ---     ---       ---       ---")
            continue
        range_rmse = np.sqrt(
            mean_squared_error(y_raw[mask], ensemble_cv_preds_ms[mask])
        )
        range_mae = mean_absolute_error(y_raw[mask], ensemble_cv_preds_ms[mask])
        range_bias = np.mean(ensemble_cv_preds_ms[mask] - y_raw[mask])
        range_r2 = (
            r2_score(y_raw[mask], ensemble_cv_preds_ms[mask]) if n > 1 else float("nan")
        )
        print(
            f"  [{lo:2d},{hi:2d})     {n:4d}  {range_rmse:6.3f}  {range_mae:6.3f}  {range_bias:+9.3f}  {range_r2:6.3f}"
        )

    # Fit final models on full dataset
    print("\n" + "-" * 70)
    print("FITTING FINAL MODELS ON FULL DATASET")
    print("-" * 70)

    for name, model in models.items():
        try:
            model.fit(X_scaled, y, sample_weight=w)
        except TypeError:
            model.fit(X_scaled, y)
        print(f"  {name}: fitted")

    # Feature importance from HistGBM's internal tree structure.
    # Accumulated split gain: sum of loss reduction at each split using feature i.
    # This is the standard "gain" importance used by XGBoost/LightGBM.
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCES (HistGBM, gain-based)")
    print("-" * 70)
    n_feats = len(selected_names)
    importances = np.zeros(n_feats)
    n_models_with_imp = 0
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            # RF and ExtraTrees expose feature_importances_ directly (Gini importance)
            importances += model.feature_importances_
            n_models_with_imp += 1
        elif hasattr(model, "_predictors"):
            # HistGBM: extract gain-based importance from internal tree structure
            model_imp = np.zeros(n_feats)
            for tree_list in model._predictors:
                for predictor in tree_list:
                    for node in predictor.nodes:
                        fi = node["feature_idx"]
                        if 0 <= fi < n_feats and not node["is_leaf"]:
                            model_imp[fi] += node["gain"]
            total_model_imp = model_imp.sum()
            if total_model_imp > 0:
                model_imp /= total_model_imp
            importances += model_imp
            n_models_with_imp += 1
    if n_models_with_imp > 0:
        importances /= n_models_with_imp
    importance_pairs = sorted(zip(selected_names, importances), key=lambda x: -x[1])
    for i, (fname, imp) in enumerate(importance_pairs):
        print(f"  {i + 1:2d}. {fname}: {imp:.4f}")

    # Compute linear calibration in mS/cm space to correct residual compression.
    # Even with slope-corrected weights, ensemble slope may not be exactly 1.0.
    # A post-hoc linear calibration corrects this:
    #   kappa_calibrated = cal_slope * kappa_raw + cal_intercept
    # Linear (not isotonic/polynomial) because: (a) preserves monotonicity,
    # (b) only 2 params so cannot overfit, (c) residual nonlinearity is small.
    calibration_slope, calibration_intercept = np.polyfit(
        ensemble_cv_preds_ms, y_raw, 1
    )
    y_calibrated = calibration_slope * ensemble_cv_preds_ms + calibration_intercept
    calibrated_rmse = np.sqrt(mean_squared_error(y_raw, y_calibrated))
    calibrated_r2 = r2_score(y_raw, y_calibrated)

    print("\n  Calibration parameters (y_true = slope * y_pred + intercept):")
    print(f"    slope: {calibration_slope:.4f}")
    print(f"    intercept: {calibration_intercept:.4f}")
    print(f"  Calibrated CV RMSE: {calibrated_rmse:.4f} mS/cm")
    print(f"  Calibrated CV R^2:  {calibrated_r2:.4f}")

    # Build model bundle -- a self-contained artifact that captures everything
    # needed for inference. The bundle is the contract between training and
    # prediction: predict_conductivity_mS_cm() loads this bundle and uses its
    # contents to reproduce the exact same feature pipeline and model ensemble.
    model_data = {
        # Core inference components
        "models": models,  # Final fitted sklearn estimators (dict keyed by model name)
        "ensemble_weights": ensemble_weights,  # Normalized weights (sum to 1.0)
        "scaler": scaler,  # StandardScaler fitted on training features
        "component_list": component_list,  # Canonical solvent/additive ordering for featurization
        "salt_list": salt_list,  # Canonical salt ordering for featurization
        "feature_names": feature_names,  # Original feature names (for featurization)
        "selected_feature_idx": selected_idx,  # Indices of features surviving correlation filter
        "selected_feature_names": selected_names,  # Names of selected features
        "feature_correlation_threshold": FEATURE_CORRELATION_THRESHOLD,
        # Log-target flag: if True, models predict log(κ) and inference must exp()
        "log_target": LOG_TARGET,
        # CV performance metrics (for diagnostics and uncertainty estimation)
        # All metrics are in mS/cm (original space), not log-space.
        "cv_rmse": ensemble_cv_rmse,  # Ensemble CV RMSE [mS/cm] -- primary quality metric
        "cv_mae": ensemble_cv_mae,  # Ensemble CV MAE [mS/cm]
        "cv_r2": ensemble_cv_r2,  # Ensemble CV R^2 (coefficient of determination)
        "cv_slope": ensemble_slope,  # Ensemble slope (1.0 = no compression)
        "per_model_cv_rmse": cv_rmse_mean,  # Per-model CV RMSE for debugging
        "per_model_cv_slope": cv_slope_mean,  # Per-model slopes for diagnostics
        # Provenance metadata (for reproducibility)
        "ensemble_config": ENSEMBLE_CONFIG,  # Hyperparameters used
        "cv_config": {
            "n_splits": CV_N_SPLITS,
            "random_state": CV_RANDOM_STATE,
        },
        # Linear calibration in mS/cm space to correct residual compression.
        # At inference: kappa_calibrated = calibration_slope * kappa_ms + calibration_intercept
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "calibrated_cv_rmse": calibrated_rmse,
        "calibrated_cv_r2": calibrated_r2,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print("\n" + "=" * 70)
    print(f"MODEL SAVED: {model_path}")
    print("=" * 70)
    print(f"  Ensemble CV RMSE: {ensemble_cv_rmse:.4f} mS/cm")
    print(f"  Ensemble CV R^2:  {ensemble_cv_r2:.4f}")
    print(f"  Ensemble CV slope: {ensemble_slope:.4f} (1.0=no compression)")
    print(f"  Log-target: {LOG_TARGET}")
    print(f"  Calibrated CV RMSE: {calibrated_rmse:.4f} mS/cm")
    print(f"  Models: {list(models.keys())}")

    return model_data


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def analyze_performance(
    model_data: dict, X_scaled: np.ndarray, y: np.ndarray, label: str = "TRAINING SET"
) -> None:
    """
    Compute and print detailed performance diagnostics for the ensemble.

    Why this exists:
        After training, it is essential to understand not just the aggregate
        RMSE but the full error distribution: what fraction of predictions are
        within 0.25, 0.5, 1.0, 1.5, and 2.0 mS/cm of the true value, what the
        90th and 95th percentile errors are, and how each individual model in the
        ensemble performs. This function provides all of that in a standardized
        format, and is used both on the training set (to check for underfitting)
        and could be applied to held-out test sets for final validation.

    What it does:
        1. Computes weighted ensemble predictions using the model weights.
        2. Calculates RMSE, MAE, and R^2 for the ensemble.
        3. Computes the full absolute-error distribution (mean, median, 90th
           percentile, 95th percentile, max).
        4. Reports accuracy thresholds: fraction of predictions within +/-0.25,
           0.5, 1.0, 1.5, and 2.0 mS/cm.
        5. Computes per-model RMSE, R^2, and weight for comparison.

    Args:
        model_data: Trained model bundle dict as returned by ``train_model()``.
            Must contain ``models`` and ``ensemble_weights``.
        X_scaled: Scaled feature matrix of shape ``(n_samples, n_features)``,
            already transformed by the scaler and with feature selection applied.
        y: True target values of shape ``(n_samples,)`` in mS/cm.
        label: Human-readable label for the dataset being evaluated (e.g.
            ``"TRAINING SET"``, ``"TEST SET"``). Used in printed headers.

    Returns:
        Tuple of (rmse, mae, r2) as floats, though the return type annotation
        says None (the actual return is used by callers).

    Side effects:
        Prints detailed performance tables to stdout.
    """
    models = model_data["models"]
    weights = model_data["ensemble_weights"]
    log_target = model_data["log_target"]

    # Weighted ensemble prediction
    preds_raw = np.zeros(len(y))
    for name, model in models.items():
        preds_raw += weights[name] * model.predict(X_scaled)

    if log_target:
        preds_ms = np.exp(preds_raw)
    else:
        preds_ms = preds_raw

    # y is always in mS/cm (caller passes original-space y)
    rmse = np.sqrt(mean_squared_error(y, preds_ms))
    mae = mean_absolute_error(y, preds_ms)
    r2 = r2_score(y, preds_ms)

    print("\n" + "=" * 70)
    print(f"{label} PERFORMANCE (WEIGHTED ENSEMBLE)")
    print("=" * 70)
    print(f"RMSE: {rmse:.4f} mS/cm")
    print(f"MAE:  {mae:.4f} mS/cm")
    print(f"R^2:  {r2:.4f}")

    abs_err = np.abs(preds_ms - y)

    print("\nError Distribution:")
    print(f"  Mean absolute error: {np.mean(abs_err):.4f} mS/cm")
    print(f"  Median absolute error: {np.median(abs_err):.4f} mS/cm")
    print(f"  90th percentile: {np.percentile(abs_err, 90):.4f} mS/cm")
    print(f"  95th percentile: {np.percentile(abs_err, 95):.4f} mS/cm")
    print(f"  Max error: {np.max(abs_err):.4f} mS/cm")

    print("\nAccuracy Thresholds:")
    for t in [0.25, 0.5, 1.0, 1.5, 2.0]:
        n = np.sum(abs_err < t)
        print(f"  Within +/-{t:.2f} mS/cm: {n}/{len(y)} ({100 * n / len(y):.1f}%)")

    # Per-model performance
    print("\n" + "-" * 70)
    print(f"PER-MODEL {label} PERFORMANCE")
    print("-" * 70)

    for name, model in models.items():
        mp = model.predict(X_scaled)
        if log_target:
            mp = np.exp(mp)
        model_rmse = np.sqrt(mean_squared_error(y, mp))
        model_r2 = r2_score(y, mp)
        print(
            f"  {name}: RMSE={model_rmse:.4f}, R^2={model_r2:.4f}, weight={weights[name]:.4f}"
        )

    return rmse, mae, r2


if __name__ == "__main__":
    # Entry point for training the conductivity surrogate from scratch.
    # Output: electrolyte_conductivity.pkl containing the full model bundle.
    model_data = train_model(
        "conductivity_mS_cm",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "electrolyte_conductivity.pkl"),
    )

    # Evaluate on training data to check for underfitting.
    # Training-set performance should be near-perfect for tree ensembles
    # (R^2 > 0.99). If not, there may be a featurization bug.
    X, y, _, _, _, _, _ = prepare_data("conductivity_mS_cm")
    X_selected = X[:, model_data["selected_feature_idx"]]
    X_scaled = model_data["scaler"].transform(X_selected)
    analyze_performance(model_data, X_scaled, y, "TRAINING SET")
