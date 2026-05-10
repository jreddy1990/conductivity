"""Compare XGBoost, Onsager limiting-law, and MolSet Transformer conductivity predictions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "control_framework"))
import jax_m4_tuning  # noqa: F401 — must precede jax import

import numpy as np
import pickle
import jax.numpy as jnp

from conductivity.electrolyte_cond_surrogate_test import load_model_bundle, predict_conductivity_mS_cm
from conductivity.electrolyte_utils_features import featurize_recipe
from conductivity.mol_set_sigma import load_model as load_molset, predict_sigma

RECIPES = [
    {
        "label": "LiPF6 1.0M / EC:EMC 3:7",
        "recipe": {
            "salts": {"LiPF6": 1.0},
            "solvents": {"EC": 0.30, "EMC": 0.70},
            "additives": {},
        },
    },
    {
        "label": "LiPF6 1.0M / EC:EMC 3:7 + 2.5% FEC",
        "recipe": {
            "salts": {"LiPF6": 1.0},
            "solvents": {"EC": 0.2925, "EMC": 0.6825},
            "additives": {"FEC": 0.025},
        },
    },
    {
        "label": "LiPF6 1.0M / EC:EMC 3:7 + 5% FEC",
        "recipe": {
            "salts": {"LiPF6": 1.0},
            "solvents": {"EC": 0.285, "EMC": 0.665},
            "additives": {"FEC": 0.05},
        },
    },
    {
        "label": "LiFSI 1.2M / EC:EMC 3:7",
        "recipe": {
            "salts": {"LiFSI": 1.2},
            "solvents": {"EC": 0.30, "EMC": 0.70},
            "additives": {},
        },
    },
    {
        "label": "LiFSI 1.2M / EC:EMC 3:7 + 5% FEC",
        "recipe": {
            "salts": {"LiFSI": 1.2},
            "solvents": {"EC": 0.285, "EMC": 0.665},
            "additives": {"FEC": 0.05},
        },
    },
    {
        "label": "LiPF6 0.8M + LiFSI 0.4M / EC:EMC 3:7",
        "recipe": {
            "salts": {"LiPF6": 0.8, "LiFSI": 0.4},
            "solvents": {"EC": 0.30, "EMC": 0.70},
            "additives": {},
        },
    },
    {
        "label": "LiPF6 1.0M / EC:DMC 1:1",
        "recipe": {
            "salts": {"LiPF6": 1.0},
            "solvents": {"EC": 0.50, "DMC": 0.50},
            "additives": {},
        },
    },
    {
        "label": "LiPF6 1.0M / EC:DMC:EMC 1:1:1",
        "recipe": {
            "salts": {"LiPF6": 1.0},
            "solvents": {"EC": 0.333, "DMC": 0.333, "EMC": 0.334},
            "additives": {},
        },
    },
    {
        "label": "LiPF6 1.0M / EC:DEC 1:1",
        "recipe": {
            "salts": {"LiPF6": 1.0},
            "solvents": {"EC": 0.50, "DEC": 0.50},
            "additives": {},
        },
    },
    {
        "label": "LiTFSI 1.0M / EC:EMC 3:7",
        "recipe": {
            "salts": {"LiTFSI": 1.0},
            "solvents": {"EC": 0.30, "EMC": 0.70},
            "additives": {},
        },
    },
]


def get_onsager_kappa(recipe: dict, component_list: list, salt_list: list) -> float:
    """Extract kappa_onsager from the feature vector (feature index 38 = kappa_onsager)."""
    x = featurize_recipe(recipe, component_list, salt_list)
    # kappa_onsager is at a known position in the feature vector.
    # From the feature list: it's feature #38 (0-indexed) in the 52-feature vector.
    # Let's verify by checking the feature names from the model.
    return float(x[38])


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load XGBoost model
    xgb_path = os.path.join(base_dir, "electrolyte_conductivity.pkl")
    print(f"Loading XGBoost model from {xgb_path}")
    xgb_bundle = load_model_bundle(xgb_path)
    component_list = xgb_bundle["component_list"]
    salt_list = xgb_bundle["salt_list"]
    feature_names = xgb_bundle["feature_names"]
    print(f"  Components: {component_list}")
    print(f"  Salts: {salt_list}")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # Find kappa_onsager index
    onsager_idx = None
    for i, fn in enumerate(feature_names):
        if fn == "kappa_onsager":
            onsager_idx = i
            break
    if onsager_idx is None:
        print("WARNING: kappa_onsager not found in feature names, trying index 38")
        onsager_idx = 38
    else:
        print(f"  kappa_onsager at feature index {onsager_idx}")

    # Load MolSet Transformer
    molset_path = os.path.join(base_dir, "mol_set_sigma.pkl")
    print(f"\nLoading MolSet Transformer from {molset_path}")
    ms_params, ms_norm_mean, ms_norm_std = load_molset(molset_path)
    print(f"  Loaded ({len(ms_params)} param arrays)")

    # Run predictions
    print("\n" + "=" * 100)
    print(f"{'Recipe':<45} {'XGBoost':>10} {'Onsager':>10} {'MolSet':>10} {'XGB-Ons':>10} {'XGB-MS':>10}")
    print(f"{'':45} {'(mS/cm)':>10} {'(mS/cm)':>10} {'(mS/cm)':>10} {'Δ':>10} {'Δ':>10}")
    print("-" * 100)

    results = []
    for entry in RECIPES:
        label = entry["label"]
        recipe = entry["recipe"]

        # XGBoost
        kappa_xgb = predict_conductivity_mS_cm(xgb_bundle, recipe)

        # Onsager limiting-law (from featurize)
        x_feat = featurize_recipe(recipe, component_list, salt_list)
        kappa_ons = float(x_feat[onsager_idx])

        # MolSet Transformer
        kappa_ms = predict_sigma(ms_params, ms_norm_mean, ms_norm_std, recipe, 298.15)

        delta_xgb_ons = kappa_xgb - kappa_ons
        delta_xgb_ms = kappa_xgb - kappa_ms

        print(f"{label:<45} {kappa_xgb:>10.2f} {kappa_ons:>10.2f} {kappa_ms:>10.2f} {delta_xgb_ons:>+10.2f} {delta_xgb_ms:>+10.2f}")
        results.append({
            "label": label,
            "xgb": kappa_xgb,
            "onsager": kappa_ons,
            "molset": kappa_ms,
        })

    print("=" * 100)

    # FEC effect analysis
    print("\nFEC EFFECT (Δκ from adding FEC to base):")
    print("-" * 80)
    # LiPF6 + 2.5% FEC
    base_lipf6 = results[0]
    fec25_lipf6 = results[1]
    fec50_lipf6 = results[2]
    print(f"  LiPF6 + 2.5% FEC:   XGB {fec25_lipf6['xgb'] - base_lipf6['xgb']:+.2f}   "
          f"Onsager {fec25_lipf6['onsager'] - base_lipf6['onsager']:+.2f}   "
          f"MolSet {fec25_lipf6['molset'] - base_lipf6['molset']:+.2f}")
    print(f"  LiPF6 + 5.0% FEC:   XGB {fec50_lipf6['xgb'] - base_lipf6['xgb']:+.2f}   "
          f"Onsager {fec50_lipf6['onsager'] - base_lipf6['onsager']:+.2f}   "
          f"MolSet {fec50_lipf6['molset'] - base_lipf6['molset']:+.2f}")

    # LiFSI + 5% FEC
    base_lifsi = results[3]
    fec50_lifsi = results[4]
    print(f"  LiFSI + 5.0% FEC:   XGB {fec50_lifsi['xgb'] - base_lifsi['xgb']:+.2f}   "
          f"Onsager {fec50_lifsi['onsager'] - base_lifsi['onsager']:+.2f}   "
          f"MolSet {fec50_lifsi['molset'] - base_lifsi['molset']:+.2f}")

    # Model agreement
    print("\nMODEL AGREEMENT:")
    print("-" * 80)
    xgb_vals = np.array([r["xgb"] for r in results])
    ons_vals = np.array([r["onsager"] for r in results])
    ms_vals = np.array([r["molset"] for r in results])

    corr_xgb_ons = np.corrcoef(xgb_vals, ons_vals)[0, 1]
    corr_xgb_ms = np.corrcoef(xgb_vals, ms_vals)[0, 1]
    corr_ons_ms = np.corrcoef(ons_vals, ms_vals)[0, 1]

    mae_xgb_ms = np.mean(np.abs(xgb_vals - ms_vals))
    mae_xgb_ons = np.mean(np.abs(xgb_vals - ons_vals))

    print(f"  Pearson r(XGB, Onsager):  {corr_xgb_ons:.4f}")
    print(f"  Pearson r(XGB, MolSet):   {corr_xgb_ms:.4f}")
    print(f"  Pearson r(Onsager, MolSet): {corr_ons_ms:.4f}")
    print(f"  MAE(XGB vs MolSet):       {mae_xgb_ms:.2f} mS/cm")
    print(f"  MAE(XGB vs Onsager):      {mae_xgb_ons:.2f} mS/cm")


if __name__ == "__main__":
    main()
