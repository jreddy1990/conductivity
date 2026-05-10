"""Test impact of adding Electrolytomics data to XGBoost conductivity training.

Runs 5-fold GroupKFold CV twice:
  1. Baseline: electrolyte_property_db + CALiSol (current pipeline)
  2. +Electrolytomics: baseline + electrolyte_electrolytomics_db

Compares CV RMSE, MAE, R², slope, and per-range accuracy.
Does NOT overwrite the production model.
"""

import numpy as np
from collections import defaultdict, Counter

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
from data.electrolyte_electrolytomics_db import DATA as _DATA_ELYTOMICS

from conductivity.electrolyte_utils_features import (
    discover_components,
    featurize_recipe,
    get_feature_names,
)

from conductivity.electrolyte_cond_surrogate_train import (
    _recipe_key,
    _ROOM_TEMP_LOW_K,
    _ROOM_TEMP_HIGH_K,
    SOURCE_WEIGHT_ORIGINAL,
    SOURCE_WEIGHT_CALISOL,
    LOW_KAPPA_THRESHOLD_MS_CM,
    LOW_KAPPA_WEIGHT_FACTOR,
    MULTI_SALT_WEIGHT_FACTOR,
    ENSEMBLE_CONFIG,
    CV_N_SPLITS,
    LOG_TARGET,
    compute_ensemble_weights,
)

SOURCE_WEIGHT_ELECTROLYTOMICS = 0.5


def _tag_original() -> list:
    tagged = []
    for e in _DATA_ORIGINAL:
        d = dict(e)
        d["_source"] = "original"
        tagged.append(d)
    return tagged


def _tag_calisol_deduped(existing_keys: set) -> list:
    calisol_room = [
        e for e in _DATA_CALISOL
        if _ROOM_TEMP_LOW_K <= e["temperature_K"] <= _ROOM_TEMP_HIGH_K
    ]
    tagged = []
    for e in calisol_room:
        if _recipe_key(e["recipe"]) not in existing_keys:
            d = dict(e)
            d["_source"] = "calisol"
            tagged.append(d)
    return tagged


def _tag_electrolytomics_deduped(existing_keys: set) -> list:
    elyt_room = [
        e for e in _DATA_ELYTOMICS
        if _ROOM_TEMP_LOW_K <= e["temperature_K"] <= _ROOM_TEMP_HIGH_K
    ]
    tagged = []
    for e in elyt_room:
        if _recipe_key(e["recipe"]) not in existing_keys:
            d = dict(e)
            d["_source"] = "electrolytomics"
            tagged.append(d)
    print(f"  Electrolytomics 25°C: {len(elyt_room)}, "
          f"after dedup: {len(tagged)}")
    return tagged


def build_baseline_dataset() -> list:
    original = _tag_original()
    original_keys = {_recipe_key(e["recipe"]) for e in _DATA_ORIGINAL}
    calisol = _tag_calisol_deduped(original_keys)
    return original + calisol


def build_augmented_dataset() -> list:
    original = _tag_original()
    original_keys = {_recipe_key(e["recipe"]) for e in _DATA_ORIGINAL}
    calisol = _tag_calisol_deduped(original_keys)
    all_keys = original_keys | {_recipe_key(e["recipe"]) for e in calisol}
    elytomics = _tag_electrolytomics_deduped(all_keys)
    return original + calisol + elytomics


def prepare(data: list, property_name: str):
    """Featurize and prepare arrays."""
    component_list, salt_list = discover_components(data)

    recipe_groups = defaultdict(list)
    for row in data:
        if property_name not in row["properties"]:
            continue
        key = _recipe_key(row["recipe"])
        source = row.get("_source", "original")
        kappa = row["properties"][property_name]
        recipe_groups[key].append((kappa, source))

    X, y, w, groups = [], [], [], []
    source_counts = Counter()
    featurize_failures = 0

    for gid, (key, measurements) in enumerate(recipe_groups.items()):
        recipe = {
            "salts": dict(key[0]),
            "solvents": dict(key[1]),
            "additives": dict(key[2]),
        }

        try:
            x_feat = featurize_recipe(recipe, component_list, salt_list)
        except Exception:
            featurize_failures += 1
            continue

        is_multi_salt = len(recipe["salts"]) > 1
        multi_salt_w = MULTI_SALT_WEIGHT_FACTOR if is_multi_salt else 1.0
        base_w = 1.0 / len(measurements)

        for kappa, source in measurements:
            if source == "original":
                src_w = SOURCE_WEIGHT_ORIGINAL
            elif source == "calisol":
                src_w = SOURCE_WEIGHT_CALISOL
            else:
                src_w = SOURCE_WEIGHT_ELECTROLYTOMICS

            kappa_w = LOW_KAPPA_WEIGHT_FACTOR if kappa < LOW_KAPPA_THRESHOLD_MS_CM else 1.0

            X.append(x_feat)
            y.append(kappa)
            w.append(base_w * src_w * kappa_w * multi_salt_w)
            groups.append(gid)
            source_counts[source] += 1

    if featurize_failures > 0:
        print(f"  Featurize failures (skipped): {featurize_failures}")

    X_arr = np.vstack(X)
    y_arr = np.array(y)
    w_arr = np.array(w)
    g_arr = np.array(groups)

    return X_arr, y_arr, w_arr, g_arr, component_list, salt_list, source_counts


def run_cv(X, y, w, groups, label: str):
    """Run 5-fold GroupKFold CV and return metrics."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Samples: {len(y)}, Unique recipes: {len(set(groups))}")
    print(f"  σ range: {y.min():.2f} - {y.max():.2f} mS/cm, "
          f"mean: {y.mean():.2f}, median: {np.median(y):.2f}")

    gkf = GroupKFold(n_splits=CV_N_SPLITS)

    oof_pred = np.zeros_like(y)
    oof_counts = np.zeros_like(y, dtype=int)

    per_model_oof = {name: np.zeros_like(y) for name in ENSEMBLE_CONFIG}
    cv_rmse_per_model = {name: [] for name in ENSEMBLE_CONFIG}
    cv_slope_per_model = {name: [] for name in ENSEMBLE_CONFIG}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = w[train_idx]

        y_train_t = np.log(y_train) if LOG_TARGET else y_train

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        for name, cfg in ENSEMBLE_CONFIG.items():
            model = cfg["class"](**cfg["params"])
            model.fit(X_train_s, y_train_t, sample_weight=w_train)

            pred = model.predict(X_val_s)
            if LOG_TARGET:
                pred = np.exp(pred)

            fold_rmse = np.sqrt(mean_squared_error(y_val, pred))
            A = np.vstack([pred, np.ones_like(pred)]).T
            slope = np.linalg.lstsq(A, y_val, rcond=None)[0][0]

            cv_rmse_per_model[name].append(fold_rmse)
            cv_slope_per_model[name].append(slope)
            per_model_oof[name][val_idx] = pred

        oof_counts[val_idx] += 1

    model_rmse_mean = {n: np.mean(v) for n, v in cv_rmse_per_model.items()}
    model_slope_mean = {n: np.mean(v) for n, v in cv_slope_per_model.items()}

    ens_weights = compute_ensemble_weights(model_rmse_mean, model_slope_mean)

    for name in ENSEMBLE_CONFIG:
        oof_pred += ens_weights[name] * per_model_oof[name]

    valid = oof_counts > 0
    oof_pred = oof_pred[valid]
    y_valid = y[valid]

    rmse = np.sqrt(mean_squared_error(y_valid, oof_pred))
    mae = mean_absolute_error(y_valid, oof_pred)
    r2 = r2_score(y_valid, oof_pred)
    A = np.vstack([oof_pred, np.ones_like(oof_pred)]).T
    slope = np.linalg.lstsq(A, y_valid, rcond=None)[0][0]

    abs_err = np.abs(y_valid - oof_pred)

    print(f"\n  Ensemble CV Results:")
    print(f"    RMSE:   {rmse:.4f} mS/cm")
    print(f"    MAE:    {mae:.4f} mS/cm")
    print(f"    R²:     {r2:.4f}")
    print(f"    Slope:  {slope:.4f}")
    print(f"    Median |err|: {np.median(abs_err):.4f} mS/cm")
    print(f"    P90 |err|:    {np.percentile(abs_err, 90):.4f} mS/cm")
    print(f"    P95 |err|:    {np.percentile(abs_err, 95):.4f} mS/cm")
    print(f"    Max |err|:    {np.max(abs_err):.4f} mS/cm")

    print(f"\n  Accuracy thresholds:")
    for t in [0.25, 0.5, 1.0, 1.5, 2.0]:
        n = np.sum(abs_err < t)
        print(f"    Within ±{t:.2f}: {n}/{len(y_valid)} ({100*n/len(y_valid):.1f}%)")

    bins = [(0, 2), (2, 5), (5, 8), (8, 11), (11, 15), (15, 25)]
    print(f"\n  Per-range RMSE:")
    for lo, hi in bins:
        mask = (y_valid >= lo) & (y_valid < hi)
        if mask.sum() > 0:
            range_rmse = np.sqrt(mean_squared_error(y_valid[mask], oof_pred[mask]))
            print(f"    [{lo:2d}-{hi:2d}) mS/cm: RMSE={range_rmse:.3f} (n={mask.sum()})")

    print(f"\n  Per-model CV RMSE / Slope / Weight:")
    for name in ENSEMBLE_CONFIG:
        print(f"    {name:12s}: RMSE={model_rmse_mean[name]:.4f}, "
              f"slope={model_slope_mean[name]:.4f}, "
              f"weight={ens_weights[name]:.4f}")

    return {
        "rmse": rmse, "mae": mae, "r2": r2, "slope": slope,
        "per_model_rmse": model_rmse_mean,
        "n_samples": len(y_valid),
        "n_recipes": len(set(groups[valid])),
    }


def main():
    prop = "conductivity_mS_cm"

    # --- Baseline ---
    print("\n" + "#"*70)
    print("  BASELINE: electrolyte_property_db + CALiSol")
    print("#"*70)
    data_base = build_baseline_dataset()
    X_b, y_b, w_b, g_b, _, _, src_b = prepare(data_base, prop)
    print(f"  Source counts: {dict(src_b)}")
    res_base = run_cv(X_b, y_b, w_b, g_b, "BASELINE CV")

    # --- With Electrolytomics ---
    print("\n" + "#"*70)
    print("  +ELECTROLYTOMICS: baseline + Electrolytomics 25°C data")
    print("#"*70)
    data_plus = build_augmented_dataset()
    X_p, y_p, w_p, g_p, _, _, src_p = prepare(data_plus, prop)
    print(f"  Source counts: {dict(src_p)}")
    res_plus = run_cv(X_p, y_p, w_p, g_p, "+ELECTROLYTOMICS CV")

    # --- Comparison ---
    print("\n" + "="*70)
    print("  COMPARISON")
    print("="*70)
    print(f"  {'Metric':<20s} {'Baseline':>12s} {'+ Elytomics':>12s} {'Delta':>12s}")
    print(f"  {'-'*56}")
    for key, fmt in [("rmse", ".4f"), ("mae", ".4f"), ("r2", ".4f"), ("slope", ".4f"),
                      ("n_samples", "d"), ("n_recipes", "d")]:
        b = res_base[key]
        p = res_plus[key]
        if isinstance(b, float):
            delta = p - b
            sign = "+" if delta > 0 else ""
            print(f"  {key:<20s} {b:>12{fmt}} {p:>12{fmt}} {sign}{delta:>11{fmt}}")
        else:
            print(f"  {key:<20s} {b:>12{fmt}} {p:>12{fmt}} {'+' if p>b else ''}{p-b:>11{fmt}}")

    print()
    if res_plus["rmse"] < res_base["rmse"]:
        print(f"  --> Electrolytomics IMPROVED CV RMSE by "
              f"{res_base['rmse'] - res_plus['rmse']:.4f} mS/cm")
    else:
        print(f"  --> Electrolytomics INCREASED CV RMSE by "
              f"{res_plus['rmse'] - res_base['rmse']:.4f} mS/cm")


if __name__ == "__main__":
    main()
