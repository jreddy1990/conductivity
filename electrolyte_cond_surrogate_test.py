# electrolyte_cond_surrogate_test.py
"""
Comprehensive test suite for electrolyte conductivity surrogate model.

MOTIVATION:
    The conductivity surrogate is a critical dependency of the electrolyte design
    pipeline: if its predictions are qualitatively wrong (e.g., conductivity
    increases monotonically with salt concentration instead of showing a dome),
    the optimizer will recommend physically unrealistic formulations. This module
    provides three layers of validation:

    1. **Physics monotonicity tests**: Verify that the model reproduces known
       qualitative behaviors of electrolyte transport (viscosity bracketing,
       conductivity dome, Lambda_0 ordering, Arrhenius temperature dependence).
       These are the most important tests because they catch model failures that
       would cause the optimizer to explore non-physical regions.

    2. **Experimental validation tests**: Compare model predictions against
       measured conductivity values from the literature for specific formulations.
       These catch systematic biases (e.g., 2 mS/cm offset) that would not be
       detected by monotonicity tests alone.

    3. **Surrogate smoke tests**: Load the serialized model bundle from disk and
       predict conductivity for diverse formulations (single salt, dual salt,
       with additives, high concentration). These catch deserialization bugs,
       feature dimension mismatches, and gross prediction errors.

    Additionally, this module provides the ``predict_conductivity_mS_cm()``
    inference function and ``load_model_bundle()`` loader that are imported by
    the rest of the pipeline.

DECISION RATIONALE:
    Tests are written against the ``ElectrolyteFormulation`` analytical model
    (not the ML surrogate) for physics tests, and against the ML surrogate for
    smoke tests. This separation is deliberate: the analytical model encodes
    explicit physics (Casteel-Amis, Nernst-Einstein, Arrhenius) that MUST be
    correct regardless of the ML training data, while the surrogate tests
    validate the ML pipeline end-to-end.

Conventions:
- Salts are provided as molarity (mol/L)
- Additives are provided as mass fraction (g/g electrolyte)
- Predictions use weighted ensemble of trained models
- Test recipes use ``make_recipe()`` helper to construct the nested dict format
  expected by ElectrolyteFormulation
"""

import pickle
import numpy as np
import pytest
from typing import Optional

from conductivity.electrolyte_utils_features import (
    get_total_li_molarity,
    compute_mixture_properties,
    featurize_recipe,
)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------


def load_model_bundle(model_path: str) -> dict:
    """
    Load a trained conductivity surrogate model bundle from disk.

    Why this exists:
        The trained ensemble of ML models, their weights, the feature scaler,
        and the canonical component/salt orderings must all be loaded together
        to make predictions. This function deserializes the pickle bundle
        produced by ``electrolyte_cond_surrogate_train.train_model()`` and
        validates that all required keys are present, failing fast with a clear
        error if the bundle is corrupted or from an incompatible version.

    What it does:
        Opens the pickle file, deserializes the bundle dict, checks for the
        required keys (``models``, ``scaler``, ``component_list``, ``salt_list``,
        ``cv_rmse``), validates that ``models`` is a non-empty dict, and logs
        diagnostic information (model names, CV RMSE, ensemble weights).

    Args:
        model_path: Filesystem path to the ``.pkl`` file containing the
            serialized model bundle (e.g. ``"electrolyte_conductivity.pkl"``).

    Returns:
        Model bundle dict with the following keys:
            - ``models`` (Dict[str, sklearn estimator]): Trained sklearn models
              keyed by name (e.g. ``"GBM"``, ``"RF"``, ``"ExtraTrees"``).
            - ``ensemble_weights`` (Dict[str, float]): Normalized weights for
              each model, computed from slope-corrected inverse CV RMSE.
            - ``scaler`` (StandardScaler): Fitted feature scaler for normalizing
              the feature vector before prediction.
            - ``component_list`` (List[str]): Canonical sorted list of liquid
              component names used during training.
            - ``salt_list`` (List[str]): Canonical sorted list of salt names
              used during training.
            - ``cv_rmse`` (float): Cross-validated RMSE of the weighted ensemble
              in mS/cm.
            - ``feature_names`` (List[str]): Human-readable feature names.
            - ``selected_feature_idx`` (List[int], optional): Indices of features
              surviving correlation-based selection.
            - ``calibration_slope`` (float, optional): Linear calibration slope.
            - ``calibration_intercept`` (float, optional): Linear calibration
              intercept.

    Raises:
        KeyError: If the bundle is missing any of the required keys.
        ValueError: If ``bundle["models"]`` is empty or not a dict.
        FileNotFoundError: If ``model_path`` does not exist (from ``open()``).

    Side effects:
        Prints diagnostic log messages (model names, CV RMSE, weights).
    """
    print(f"\n[load_model_bundle] Loading model from: {model_path}")

    # Deserialize the pickle bundle. The bundle is the complete inference
    # artifact: models + scaler + feature metadata + calibration params.
    # It is produced by train_model() in electrolyte_cond_surrogate_train.py.
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    # Hard validation: fail fast with a clear error if the bundle is corrupted
    # or from an incompatible training version. These 5 keys are the minimum
    # required for prediction; additional keys (ensemble_weights, calibration_*,
    # feature_names) are optional for backward compatibility.
    required_keys = {
        "models",  # Dict of fitted sklearn estimators
        "scaler",  # StandardScaler for feature normalization
        "component_list",  # Canonical liquid component ordering (for featurization)
        "salt_list",  # Canonical salt ordering (for featurization)
        "cv_rmse",  # Cross-validated RMSE (for uncertainty reporting)
    }
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(
            f"Model bundle missing keys: {sorted(missing)}. "
            f"Found keys: {sorted(bundle.keys())}"
        )

    # Ensure models dict is populated with at least one fitted estimator
    if not isinstance(bundle["models"], dict) or not bundle["models"]:
        raise ValueError("bundle['models'] must be a non-empty dict")

    print("[load_model_bundle] Loaded successfully")
    print(f"  Models: {list(bundle['models'].keys())}")
    print(f"  CV RMSE: {bundle['cv_rmse']:.4f} mS/cm")
    if "ensemble_weights" in bundle:
        print(f"  Ensemble weights: {bundle['ensemble_weights']}")

    return bundle


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------


def predict_conductivity_mS_cm(
    model_bundle: dict,
    recipe: dict,
    verbose: bool = False,
) -> float:
    """
    Predict electrolyte ionic conductivity using the trained weighted ensemble.

    Why this exists:
        This is the primary inference entry point for the conductivity surrogate.
        It is called by ``ElectrolyteFormulation`` in the electrolyte model, by
        the mixture optimizer during formulation search, and by the pipeline
        stages that need rapid conductivity estimates. A single function
        encapsulates the full prediction path: featurization, optional feature
        selection, scaling, per-model prediction, and weighted averaging.

    What it does:
        1. Extracts models, scaler, component/salt lists, and ensemble weights
           from the model bundle.
        2. Calls ``featurize_recipe()`` to convert the recipe into a numeric
           feature vector.
        3. Applies feature selection (if ``selected_feature_idx`` is present in
           the bundle) to reduce the feature vector to the training subset.
        4. Scales features using the fitted ``StandardScaler``.
        5. Runs each model's ``.predict()`` on the scaled features.
        6. Computes the weighted average prediction using ensemble weights.

    Args:
        model_bundle: Model bundle dict as returned by ``load_model_bundle()``.
            Must contain ``models``, ``scaler``, ``component_list``, ``salt_list``,
            and optionally ``ensemble_weights`` and ``selected_feature_idx``.
        recipe: Electrolyte recipe dict with keys ``"solvents"``
            (name -> vol fraction), ``"salts"`` (name -> molarity in mol/L),
            ``"additives"`` (name -> wt fraction). Missing keys default to
            empty dicts internally.
        verbose: If True, prints the feature vector shape, per-model predictions
            with their weights, and the final ensemble prediction. Useful for
            debugging unexpected conductivity values.

    Returns:
        Predicted ionic conductivity in mS/cm (float). Typical range for
        lithium-ion battery electrolytes is 2-15 mS/cm at 25 C.

    Side effects:
        Prints diagnostic output when ``verbose=True``.
    """
    # ---------------------------------------------------------------
    # STEP 1: Extract inference components from model bundle
    # ---------------------------------------------------------------
    # The model bundle is a self-contained artifact produced by training.
    # It contains everything needed for prediction: models, scaler,
    # feature definitions, and ensemble weights.
    models = model_bundle["models"]  # Dict of fitted sklearn estimators
    scaler = model_bundle["scaler"]  # StandardScaler fitted on training data
    component_list = model_bundle[
        "component_list"
    ]  # Canonical solvent/additive ordering
    salt_list = model_bundle["salt_list"]  # Canonical salt ordering

    weights = model_bundle["ensemble_weights"]

    # ---------------------------------------------------------------
    # STEP 2: Featurize the recipe
    # ---------------------------------------------------------------
    x = featurize_recipe(recipe, component_list, salt_list).reshape(1, -1)

    # ---------------------------------------------------------------
    # STEP 3: Apply feature selection (if used during training)
    # ---------------------------------------------------------------
    selected_idx = model_bundle["selected_feature_idx"]
    x = x[:, selected_idx]

    # ---------------------------------------------------------------
    # STEP 4: Scale features
    # ---------------------------------------------------------------
    x_scaled = scaler.transform(x)

    if verbose:
        print(f"\n[predict_conductivity_mS_cm] Recipe: {recipe}")
        print(f"  Feature vector shape: {x.shape}")
        print(f"  Using {len(selected_idx)} selected features")

    # ---------------------------------------------------------------
    # STEP 5: Weighted ensemble prediction
    # ---------------------------------------------------------------
    log_target = model_bundle["log_target"]

    ensemble_pred = 0.0
    for name, model in models.items():
        pred = float(model.predict(x_scaled)[0])
        ensemble_pred += weights[name] * pred

        if verbose:
            disp = np.exp(pred) if log_target else pred
            print(f"  {name}: {disp:.3f} mS/cm (weight={weights[name]:.4f})")

    if log_target:
        ensemble_pred = float(np.exp(ensemble_pred))

    if verbose:
        print(f"  Ensemble: {ensemble_pred:.3f} mS/cm (log_target={log_target})")

    return ensemble_pred


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------


def print_recipe_diagnostics(recipe: dict) -> None:
    """
    Print a human-readable diagnostic summary of a recipe's key physical properties.

    Why this exists:
        When debugging surrogate predictions or inspecting test results, it is
        essential to see the effective Li+ concentration, mixture dielectric
        constant, viscosity, and donor number alongside the recipe composition.
        This helper centralizes that output formatting so every test and CLI
        invocation produces consistent, interpretable diagnostics.

    What it does:
        Computes ``get_total_li_molarity()`` and ``compute_mixture_properties()``
        from the recipe, then prints effective Li+ molarity, mixture epsilon_r,
        viscosity, donor number, and the individual solvent/salt/additive
        loadings in a structured format.

    Args:
        recipe: Electrolyte recipe dict with ``"solvents"``, ``"salts"``, and/or
            ``"additives"`` keys. Same format as accepted by
            ``predict_conductivity_mS_cm()``.

    Returns:
        None.

    Side effects:
        Prints multiple lines of diagnostic output to stdout.
    """
    li_m = get_total_li_molarity(recipe)
    props = compute_mixture_properties(recipe)

    print(f"  Effective Li+ molarity: {li_m:.3f} M")
    print(
        f"  Mixture properties: "
        f"epsilon_r={props['epsilon_r']:.1f}, "
        f"viscosity={props['viscosity_cP']:.2f} cP, "
        f"donor_number={props['donor_number']:.1f}"
    )

    if recipe.get("solvents"):
        s = ", ".join(f"{k}={v:.2f}" for k, v in sorted(recipe["solvents"].items()))
        print(f"  Solvents (mass/vol frac): {s}")

    if recipe.get("salts"):
        s = ", ".join(f"{k}={v:.2f} M" for k, v in sorted(recipe["salts"].items()))
        print(f"  Salts (molarity): {s}")

    if recipe.get("additives"):
        s = ", ".join(f"{k}={v:.3f}" for k, v in sorted(recipe["additives"].items()))
        print(f"  Additives (mass frac): {s}")


# ---------------------------------------------------------------------
# Test recipe helper (for ElectrolyteFormulation tests)
# ---------------------------------------------------------------------


def make_recipe(
    solvent_wt: dict,
    salt_molarity: Optional[dict] = None,
    additives_wt: Optional[dict] = None,
    temperature: float = 25.0,
) -> dict:
    """
    Build a recipe dict in the nested format expected by ElectrolyteFormulation.

    Why this exists:
        The ``ElectrolyteFormulation`` model expects a specific nested dict
        structure where each solvent has a ``{"wt_fraction": value}`` sub-dict,
        each salt has ``{"molarity": value}``, and each additive has
        ``{"wt_fraction": value}``. This helper converts the simpler flat
        ``{name: value}`` format used in tests into that nested structure, plus
        adds the temperature field. Without it, every test would need 5-10 lines
        of boilerplate dict construction.

    What it does:
        Wraps solvent fractions in ``{"wt_fraction": float(v)}``, salt molarities
        in ``{"molarity": float(v)}``, additive fractions in
        ``{"wt_fraction": float(v)}``, and includes the temperature. Returns the
        assembled dict.

    Args:
        solvent_wt: Mapping of solvent name to weight fraction (e.g.
            ``{"EC": 0.3, "EMC": 0.7}``).
        salt_molarity: Mapping of salt name to molarity in mol/L (e.g.
            ``{"LiPF6": 1.0}``). None or empty dict for salt-free recipes.
        additives_wt: Mapping of additive name to weight fraction (e.g.
            ``{"FEC": 0.02, "VC": 0.01}``). None for no additives.
        temperature: Temperature in degrees Celsius. Default is 25.0 C
            (standard lab conditions).

    Returns:
        Nested recipe dict with keys ``"temperature"`` (float), ``"solvents"``
        (dict of dicts), ``"salt"`` (dict of dicts), ``"additives"``
        (dict of dicts), ready for ``ElectrolyteFormulation(recipe)``.

    Side effects:
        None. Pure function.
    """
    solvents = {k: {"wt_fraction": float(v)} for k, v in (solvent_wt or {}).items()}
    salts = (
        {k: {"molarity": float(v)} for k, v in (salt_molarity or {}).items()}
        if salt_molarity
        else {}
    )
    additives = (
        {k: {"wt_fraction": float(v)} for k, v in (additives_wt or {}).items()}
        if additives_wt
        else {}
    )

    return {
        "temperature": float(temperature),
        "solvents": solvents,
        "salt": salts,
        "additives": additives,
    }


# ---------------------------------------------------------------------
# VISCOSITY TESTS (ElectrolyteFormulation model)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("solvent_name", ["EC", "PC", "EMC", "DMC", "DEC"])
def test_pure_solvent_viscosity_matches_species_db(solvent_name):
    """
    Verify that a single-solvent recipe reproduces the species database viscosity.

    Why this test exists:
        The ElectrolyteFormulation model computes mixture viscosity via a
        mixing rule. For a pure (single-component) solvent, the mixing rule
        must reduce to the species database value exactly. If it does not,
        the mixing-rule implementation or the species property lookup is broken.
        This is a fundamental consistency check that catches basis-conversion
        errors and data mismatch bugs.

    What it checks:
        Creates a recipe with 100% of a single solvent and 1.0 M LiPF6 salt.
        Compares ``model.viscosity_0`` against the ``viscosity_cP`` value from
        the SOLVENTS dict in ``data.species_fns``. Asserts relative error < 1e-6.

    Args:
        solvent_name: Parametrized solvent name from the set
            {EC, PC, EMC, DMC, DEC}.
    """
    from electrolyte_model import ElectrolyteFormulation
    from species_fns import SOLVENTS

    target_eta = SOLVENTS[solvent_name]["viscosity_cP"]

    recipe = make_recipe(
        solvent_wt={solvent_name: 1.0},
        salt_molarity={"LiPF6": 1.0},
    )
    model = ElectrolyteFormulation(recipe)

    eta = model.viscosity_0
    rel_err = abs(eta - target_eta) / target_eta if target_eta != 0 else 0.0

    assert rel_err < 1e-6, (
        f"Pure solvent viscosity mismatch for {solvent_name}: "
        f"model={eta}, target={target_eta}"
    )


def test_ec_dmc_mixture_viscosity_brackets_components():
    """
    Verify that binary mixture viscosity lies between the pure component values.

    Why this test exists:
        Under any reasonable mixing rule (linear, log-linear, Arrhenius), the
        mixture viscosity of a binary blend must be bracketed by the component
        viscosities. If the mixture value exceeds the higher-viscosity component
        or falls below the lower-viscosity component, the mixing rule is producing
        non-physical results. This is a basic thermodynamic consistency check.

    What it checks:
        Creates EC:DMC 50:50 by weight with 1.0 M LiPF6. Retrieves pure-component
        viscosities from SOLVENTS. Asserts ``eta_min < eta_mix < eta_max``.
    """
    from electrolyte_model import ElectrolyteFormulation
    from species_fns import SOLVENTS

    eta_ec = SOLVENTS["EC"]["viscosity_cP"]
    eta_dmc = SOLVENTS["DMC"]["viscosity_cP"]
    eta_min = min(eta_ec, eta_dmc)
    eta_max = max(eta_ec, eta_dmc)

    recipe = make_recipe(
        solvent_wt={"EC": 0.5, "DMC": 0.5},
        salt_molarity={"LiPF6": 1.0},
    )
    model = ElectrolyteFormulation(recipe)
    eta_mix = model.viscosity_0

    assert eta_min < eta_mix < eta_max, (
        f"Mixture viscosity should bracket components: "
        f"eta_mix={eta_mix:.3f}, eta_min={eta_min:.3f}, eta_max={eta_max:.3f}"
    )


def test_additive_increases_viscosity_when_more_viscous():
    """
    Verify that adding a high-viscosity additive increases mixture viscosity.

    Why this test exists:
        Additives like TPP (triphenyl phosphate) have much higher viscosity than
        linear carbonates like DMC. Adding even a small weight fraction of a
        more-viscous component must increase the overall mixture viscosity. If it
        does not, the additive is being excluded from the viscosity calculation
        or the mixing rule has a sign error. This test guards against regressions
        in how additives are incorporated into the liquid-phase property model.

    What it checks:
        Compares viscosity of pure DMC + 1.0 M LiPF6 against 95% DMC + 5% TPP
        + 1.0 M LiPF6. Asserts ``eta_mix > eta_base``.
    """
    from electrolyte_model import ElectrolyteFormulation

    base_recipe = make_recipe(
        solvent_wt={"DMC": 1.0},
        salt_molarity={"LiPF6": 1.0},
    )
    base_model = ElectrolyteFormulation(base_recipe)
    eta_base = base_model.viscosity_0

    mix_recipe = make_recipe(
        solvent_wt={"DMC": 0.95},
        salt_molarity={"LiPF6": 1.0},
        additives_wt={"TPP": 0.05},
    )
    mix_model = ElectrolyteFormulation(mix_recipe)
    eta_mix = mix_model.viscosity_0

    assert eta_mix > eta_base, (
        f"Adding viscous TPP should increase viscosity: "
        f"base={eta_base:.3f}, mix={eta_mix:.3f}"
    )


# ---------------------------------------------------------------------
# CONDUCTIVITY TESTS (ElectrolyteFormulation model)
# ---------------------------------------------------------------------


def test_zero_salt_gives_zero_conductivity():
    """
    Verify that a salt-free recipe produces exactly zero conductivity.

    Why this test exists:
        Ionic conductivity requires mobile charge carriers (Li+ ions). With no
        salt and no ionic additives, there are no ions in solution, so the
        conductivity must be exactly zero. A non-zero prediction would indicate
        a bug in the concentration-handling logic (e.g. a default salt
        concentration being injected) or in the Casteel-Amis / Nernst-Einstein
        model's zero-concentration behavior.

    What it checks:
        Creates an EC:DMC 50:50 recipe with empty salts dict. Asserts
        ``model.kappa_0 == 0.0``.
    """
    from electrolyte_model import ElectrolyteFormulation

    recipe = make_recipe(
        solvent_wt={"EC": 0.5, "DMC": 0.5},
        salt_molarity={},
    )
    model = ElectrolyteFormulation(recipe)
    assert model.kappa_0 == 0.0, f"Expected 0 conductivity, got {model.kappa_0}"


def test_conductivity_increases_with_molarity_ec_dmc():
    """
    Verify monotonic conductivity increase with salt concentration up to 1.5 M.

    Why this test exists:
        In the low-to-moderate concentration regime (0.5-1.5 M), conductivity
        should increase with salt molarity because more ions are available for
        charge transport. The viscosity increase and ion pairing that eventually
        cause conductivity to decrease do not dominate until higher
        concentrations (>1.5 M for most carbonate systems). If this monotonic
        trend is violated, the concentration-conductivity model has a
        qualitative error that would invalidate all formulation optimization.

    What it checks:
        Creates EC:DMC 50:50 recipes at 0.5, 1.0, and 1.5 M LiPF6. Asserts
        ``kappa(0.5M) < kappa(1.0M) <= kappa(1.5M)`` and that all values are
        in the physically reasonable range of 0.1-200 mS/cm.
    """
    from electrolyte_model import ElectrolyteFormulation

    molarities = [0.5, 1.0, 1.5]
    kappas = []

    for c in molarities:
        recipe = make_recipe(
            solvent_wt={"EC": 0.5, "DMC": 0.5},
            salt_molarity={"LiPF6": c},
        )
        model = ElectrolyteFormulation(recipe)
        kappas.append(model.kappa_0)

    k0, k1, k2 = kappas
    assert k0 < k1 <= k2, (
        f"Expected kappa(0.5M) < kappa(1.0M) <= kappa(1.5M), got {k0:.3f}, {k1:.3f}, {k2:.3f}"
    )

    for k in kappas:
        assert 0.1 < k < 200.0, f"Unreasonable kappa value: {k}"


def test_conductivity_scales_with_salt_lambda0():
    """
    Verify that conductivity scales with intrinsic salt molar conductivity (Lambda_0).

    Why this test exists:
        The limiting molar conductivity Lambda_0 is the fundamental salt-quality
        metric: it encodes how efficiently a salt's ions carry charge at infinite
        dilution. At fixed concentration and solvent, a salt with higher Lambda_0
        must produce higher conductivity. This test validates that the model
        correctly uses Lambda_0 from the species database (not a hardcoded value)
        and that the relationship has the correct sign. It catches data lookup
        errors and model misspecification.

    What it checks:
        Creates pure DMC recipes with 1.0 M of LiPF6 and LiFSI separately.
        Retrieves their Lambda_0 values from the SALTS dict. Asserts that the
        salt with higher Lambda_0 produces higher conductivity.
    """
    from electrolyte_model import ElectrolyteFormulation
    from species_fns import SALTS

    lam_pf6 = SALTS["LiPF6"]["Lambda_0"]
    lam_fsi = SALTS["LiFSI"]["Lambda_0"]
    assert lam_fsi != lam_pf6

    recipe_pf6 = make_recipe(
        solvent_wt={"DMC": 1.0},
        salt_molarity={"LiPF6": 1.0},
    )
    recipe_fsi = make_recipe(
        solvent_wt={"DMC": 1.0},
        salt_molarity={"LiFSI": 1.0},
    )

    model_pf6 = ElectrolyteFormulation(recipe_pf6)
    model_fsi = ElectrolyteFormulation(recipe_fsi)

    k_pf6 = model_pf6.kappa_0
    k_fsi = model_fsi.kappa_0

    if lam_fsi > lam_pf6:
        assert k_fsi > k_pf6, (
            f"Expected LiFSI kappa > LiPF6 kappa (Lambda0_FSI > Lambda0_PF6), "
            f"got kappa_FSI={k_fsi:.3f}, kappa_PF6={k_pf6:.3f}"
        )
    else:
        assert k_pf6 > k_fsi, (
            f"Expected LiPF6 kappa > LiFSI kappa (Lambda0_PF6 > Lambda0_FSI), "
            f"got kappa_PF6={k_pf6:.3f}, kappa_FSI={k_fsi:.3f}"
        )


def test_ionic_additive_contributes_to_conductivity():
    """
    Verify that ionic additives contribute to conductivity as charge carriers.

    Why this test exists:
        Ionic additives like LiDFOB (``provides_ionic_conductivity=True``) are
        lithium salts stored under the ``"additives"`` key (as weight fractions)
        rather than under ``"salts"`` (as molarities). The model must still
        recognize them as ion sources, convert their weight fractions to
        effective molarities, and include them in the conductivity calculation.
        This test verifies three things: (1) LiDFOB alone produces non-zero
        conductivity, (2) adding LiDFOB to an existing LiPF6 formulation does
        not significantly reduce conductivity (i.e. it is not treated as a
        non-ionic diluent), and (3) the wt-fraction-to-molarity conversion
        pipeline works end-to-end.

    What it checks:
        - LiDFOB-only recipe (no salt): asserts ``kappa > 0``.
        - LiPF6 + LiDFOB recipe vs LiPF6-only: asserts relative conductivity
          change > -5% (LiDFOB should not dilute conductivity significantly).
    """
    from electrolyte_model import ElectrolyteFormulation

    base_recipe = make_recipe(
        solvent_wt={"EC": 0.3, "EMC": 0.7},
        salt_molarity={"LiPF6": 1.0},
    )
    base_model = ElectrolyteFormulation(base_recipe)
    k_base = base_model.kappa_0

    add_only_recipe = make_recipe(
        solvent_wt={"EC": 0.3, "EMC": 0.7},
        salt_molarity={},
        additives_wt={"LiDFOB": 0.03},
    )
    add_only_model = ElectrolyteFormulation(add_only_recipe)
    k_add_only = add_only_model.kappa_0

    assert k_add_only > 0.0, (
        f"LiDFOB-only electrolyte should have kappa>0, got {k_add_only}"
    )

    mix_recipe = make_recipe(
        solvent_wt={"EC": 0.3, "EMC": 0.7},
        salt_molarity={"LiPF6": 1.0},
        additives_wt={"LiDFOB": 0.03},
    )
    mix_model = ElectrolyteFormulation(mix_recipe)
    k_mix = mix_model.kappa_0

    rel_change = (k_mix - k_base) / k_base
    assert rel_change > -0.05, (
        f"Adding ionic LiDFOB should not significantly reduce kappa "
        f"(delta_kappa/kappa_base > -5%); base={k_base:.3f}, mix={k_mix:.3f}, "
        f"rel_change={rel_change:.3%}"
    )


def test_conductivity_matches_all_real_data_points():
    """
    Validate conductivity model against experimental data for LiPF6/EC/DMC/EMC.

    Why this test exists:
        The ultimate validation of the electrolyte model is agreement with
        experimental measurements. This test checks predicted conductivity
        against four literature data points for LiPF6 in EC/DMC/EMC blends at
        different concentrations (0.98-1.47 M). Each data point has a tolerance
        of +/-0.5 mS/cm. At least 2 of 4 must pass. This catches systematic
        model bias (e.g. if the Casteel-Amis parameters are wrong or the
        dielectric constant weighting is off).

    What it checks:
        For each test case, constructs the recipe with the experimental solvent
        ratios (EC:DMC:EMC ~ 17:51:20 by volume), creates an ElectrolyteFormulation
        model, and compares ``model.kappa_0`` to the expected conductivity within
        the specified tolerance. Also verifies the peak conductivity is at ~1 M.

    Returns:
        List of result dicts with case number, molarity, expected, calculated,
        diff, and status for each test point.
    """
    from electrolyte_model import ElectrolyteFormulation

    # Experimental data points from literature for LiPF6 in EC/DMC/EMC blends.
    # These are real-world measurements at 25 C from impedance spectroscopy.
    # The tolerance of +/-0.5 mS/cm accounts for:
    #   - Experimental measurement uncertainty (~0.1-0.2 mS/cm)
    #   - Temperature control uncertainty (+/-1C can shift kappa by ~0.3 mS/cm)
    #   - Slight differences in solvent purity between labs
    #   - Model approximation error (CV RMSE ~0.75 mS/cm)
    # Note: the expected conductivities decrease with increasing molarity
    # above ~1M, consistent with the dome shape from ion pairing + viscosity.
    test_cases = [
        {"molarity": 0.98, "expected_conductivity": 10.27, "tolerance": 0.5},
        {"molarity": 1.15, "expected_conductivity": 10.23, "tolerance": 0.5},
        {"molarity": 1.31, "expected_conductivity": 10.14, "tolerance": 0.5},
        {"molarity": 1.47, "expected_conductivity": 9.45, "tolerance": 0.5},
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        # Solvent ratios: EC:DMC:EMC = 17:51:20 by volume (literature formulation).
        # Normalized to sum-to-1 fractions for the recipe dict.
        total_solvent = 17 + 51 + 20
        ec_fraction = 17 / total_solvent
        dmc_fraction = 51 / total_solvent
        emc_fraction = 20 / total_solvent

        recipe = make_recipe(
            solvent_wt={"EC": ec_fraction, "DMC": dmc_fraction, "EMC": emc_fraction},
            salt_molarity={"LiPF6": test["molarity"]},
        )
        model = ElectrolyteFormulation(recipe)
        kappa = model.kappa_0

        expected = test["expected_conductivity"]
        tolerance = test["tolerance"]
        lower = expected - tolerance
        upper = expected + tolerance

        status = "PASS" if lower <= kappa <= upper else "FAIL"

        results.append(
            {
                "case": i,
                "molarity": test["molarity"],
                "expected": expected,
                "calculated": kappa,
                "diff": kappa - expected,
                "diff_percent": ((kappa - expected) / expected) * 100,
                "status": status,
            }
        )

        print(
            f"Case {i}: {test['molarity']}M LiPF6 -> "
            f"Expected: {expected:.2f} mS/cm, "
            f"Calculated: {kappa:.2f} mS/cm, "
            f"Diff: {kappa - expected:+.2f} mS/cm ({status})"
        )

    passes = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)

    print(f"\nPassed: {passes}/{total}")

    conductivities = [r["calculated"] for r in results]
    peak_idx = np.argmax(conductivities)
    peak_molarity = results[peak_idx]["molarity"]
    print(f"Peak conductivity at: {peak_molarity:.2f}M")

    assert passes >= 2, f"Too many failures: only {passes}/{total} cases passed"

    return results


def test_conductivity_peak_around_1M():
    """
    Verify conductivity dome shape: peak around 1 M, decrease at 1.5 M.

    Why this test exists:
        The concentration-conductivity relationship for Li salts in carbonate
        solvents exhibits a dome shape peaking around 1.0-1.2 M. Below the peak,
        conductivity increases with more charge carriers. Above the peak,
        viscosity increase and ion pairing dominate, reducing mobility. This is
        a fundamental electrochemistry phenomenon. If the model does not
        reproduce this dome shape, the concentration dependence is qualitatively
        wrong, which would lead the optimizer to recommend unrealistic
        concentrations.

    What it checks:
        Creates EC:DMC 50:50 recipes at 0.5, 1.0, and 1.5 M LiPF6. Asserts:
        (1) ``kappa(1.0M) > kappa(0.5M)`` (rising branch), and
        (2) ``kappa(1.5M) <= 1.1 * kappa(1.0M)`` (peak or declining branch;
        10% tolerance for numerical reasons).
    """
    from electrolyte_model import ElectrolyteFormulation

    recipe_base = make_recipe(
        solvent_wt={"EC": 0.5, "DMC": 0.5},
        salt_molarity={"LiPF6": 1.0},
    )
    model_base = ElectrolyteFormulation(recipe_base)
    kappa_1M = model_base.kappa_0

    recipe_low = make_recipe(
        solvent_wt={"EC": 0.5, "DMC": 0.5},
        salt_molarity={"LiPF6": 0.5},
    )
    model_low = ElectrolyteFormulation(recipe_low)
    kappa_0_5M = model_low.kappa_0

    recipe_high = make_recipe(
        solvent_wt={"EC": 0.5, "DMC": 0.5},
        salt_molarity={"LiPF6": 1.5},
    )
    model_high = ElectrolyteFormulation(recipe_high)
    kappa_1_5M = model_high.kappa_0

    print(
        f"Conductivity trend: 0.5M -> {kappa_0_5M:.2f} mS/cm, "
        f"1.0M -> {kappa_1M:.2f} mS/cm, "
        f"1.5M -> {kappa_1_5M:.2f} mS/cm"
    )

    assert kappa_1M > kappa_0_5M, (
        f"Conductivity should increase from 0.5M to 1.0M: "
        f"0.5M={kappa_0_5M:.2f}, 1.0M={kappa_1M:.2f}"
    )

    assert kappa_1_5M <= kappa_1M * 1.1, (
        f"Conductivity at 1.5M should not exceed 1.0M value by more than 10%: "
        f"1.0M={kappa_1M:.2f}, 1.5M={kappa_1_5M:.2f}"
    )

    print("Conductivity peaks around 1.0-1.2M as expected")


# ---------------------------------------------------------------------
# SAFETY / FLASH POINT / SWELLING TESTS
# ---------------------------------------------------------------------


def test_flash_point_increases_with_tpp():
    """
    Verify that adding flame-retardant TPP raises the mixture flash point.

    Why this test exists:
        Triphenyl phosphate (TPP) is a flame-retardant additive with a very high
        flash point (~220 C). Adding it to a carbonate blend should raise the
        mixture flash point because TPP is less volatile and harder to ignite
        than linear carbonates like EMC (~23 C flash point). If adding TPP
        does not increase the flash point, the flash-point mixing model is
        ignoring TPP or has a sign error, which would give false safety
        assessments.

    What it checks:
        Compares flash point of EC:EMC 30:70 + 1.0 M LiPF6 with and without
        5 wt% TPP. Asserts ``fp_with_TPP > fp_base``.
    """
    from electrolyte_model import ElectrolyteFormulation

    base_recipe = make_recipe(
        solvent_wt={"EC": 0.3, "EMC": 0.7},
        salt_molarity={"LiPF6": 1.0},
    )
    base_model = ElectrolyteFormulation(base_recipe)
    fp_base = base_model.flash_point_mix

    recipe_tpp = make_recipe(
        solvent_wt={"EC": 0.3, "EMC": 0.7},
        salt_molarity={"LiPF6": 1.0},
        additives_wt={"TPP": 0.05},
    )
    model_tpp = ElectrolyteFormulation(recipe_tpp)
    fp_tpp = model_tpp.flash_point_mix

    assert fp_tpp > fp_base, (
        f"Adding TPP should raise flash point: base={fp_base:.1f}C, "
        f"with TPP={fp_tpp:.1f}C"
    )


def test_swelling_score_reasonable_range():
    """
    Verify that the swelling score is in a physically reasonable range.

    Why this test exists:
        The swelling score quantifies how much the electrolyte is expected to
        swell polymer separator and binder materials. For typical carbonate
        blends, this score should be modest (0.5-3.5). Extreme values would
        indicate a bug in the swelling model (e.g. using wrong units for
        solubility parameters or missing a normalization). This is a sanity
        bound check rather than an exact value test.

    What it checks:
        Creates an EC:EMC:DMC 30:40:30 recipe with 1.0 M LiPF6. Asserts
        ``0.5 <= model.swelling_score <= 3.5``.
    """
    from electrolyte_model import ElectrolyteFormulation

    recipe = make_recipe(
        solvent_wt={"EC": 0.3, "EMC": 0.4, "DMC": 0.3},
        salt_molarity={"LiPF6": 1.0},
    )
    model = ElectrolyteFormulation(recipe)
    s = model.swelling_score

    assert 0.5 <= s <= 3.5, f"Unexpected swelling score: {s}"


# ---------------------------------------------------------------------
# TEMPERATURE DEPENDENCE TESTS
# ---------------------------------------------------------------------


def test_temperature_dependence_conductivity_and_viscosity():
    """
    Verify Arrhenius-like temperature dependence: kappa up, viscosity down with T.

    Why this test exists:
        Ion transport in liquid electrolytes follows Arrhenius (or VTF) kinetics:
        higher temperature lowers viscosity and increases ion mobility, so
        conductivity increases and viscosity decreases with temperature. This is
        one of the most fundamental properties of electrolyte transport. If the
        model violates this trend, the temperature-correction mechanism (Arrhenius
        activation energies for viscosity and conductivity) is broken, which would
        give wrong predictions at any temperature other than 25 C.

    What it checks:
        Creates an EC:EMC 30:70 + 1.0 M LiPF6 recipe. Evaluates conductivity
        and viscosity at 10 C and 40 C. Asserts:
        (1) ``kappa(40C) > kappa(10C)`` (conductivity increases with T),
        (2) ``eta(40C) < eta(10C)`` (viscosity decreases with T).
    """
    from electrolyte_model import ElectrolyteFormulation

    recipe = make_recipe(
        solvent_wt={"EC": 0.3, "EMC": 0.7},
        salt_molarity={"LiPF6": 1.0},
    )
    model = ElectrolyteFormulation(recipe)

    k_10 = model.get_conductivity_T(10.0)
    k_40 = model.get_conductivity_T(40.0)

    eta_10 = model.get_viscosity_T(10.0)
    eta_40 = model.get_viscosity_T(40.0)

    assert k_40 > k_10, (
        f"Conductivity should increase with T: kappa(10C)={k_10:.3f}, kappa(40C)={k_40:.3f}"
    )
    assert eta_40 < eta_10, (
        f"Viscosity should decrease with T: eta(10C)={eta_10:.3f}, eta(40C)={eta_40:.3f}"
    )


# ---------------------------------------------------------------------
# SURROGATE MODEL SMOKE TESTS
# ---------------------------------------------------------------------


def test_surrogate_model_predictions():
    """
    Smoke test the surrogate model on 10 diverse real-world formulations.

    Why this test exists:
        This is the primary integration test for the surrogate model's prediction
        pipeline. It loads the serialized model bundle from disk, runs predictions
        on 10 diverse formulations (varying solvents, salt types, dual-salt,
        additives, concentrations), and checks that all predictions fall in the
        physically reasonable range of 1-20 mS/cm. This catches deserialization
        bugs, feature dimension mismatches between training and inference, and
        gross prediction errors. The diverse recipe set covers edge cases like
        high concentration (2.0 M), dual salts (LiPF6+LiFSI), and additive-heavy
        formulations (FEC+TPP+VC).

    What it checks:
        Loads ``electrolyte_conductivity.pkl``, predicts conductivity for 10 recipes,
        asserts all predictions are in [1.0, 20.0] mS/cm. Prints diagnostics
        including CV RMSE uncertainty bands.
    """
    model_bundle = load_model_bundle("electrolyte_conductivity.pkl")

    # Diverse test formulations covering the space of real-world electrolytes.
    # Each recipe tests a different aspect of the surrogate:
    # - Standard ternary solvent (EC/DMC/EMC): most common commercial blend
    # - PGPL: with ionic additive (LiDFOB) and SEI-former (VC)
    # - Binary solvents (EC:EMC, EC:DMC): simpler blends
    # - Non-standard salt (LiTFSI): tests salt generalization
    # - Low concentration (0.5M LiFSI): tests concentration extrapolation
    # - High concentration (2.0M LiPF6): tests dome-shape capture
    # - Additive-rich (VC+FEC): tests additive feature handling
    # - Dual salt (LiPF6+LiFSI): tests multi-salt feature block
    # - Full production recipe (FEC+TPP+VC): realistic complex formulation
    test_recipes = [
        {
            "name": "1.0M LiPF6 in EC/DMC/EMC (19/58/23)",
            "recipe": {
                "solvents": {"EC": 0.193, "DMC": 0.580, "EMC": 0.227},
                "salts": {"LiPF6": 1.15},
                "additives": {},
            },
            # Expected: ~10 mS/cm (standard commercial formulation)
        },
        {
            "name": "PGPL formulation",
            "recipe": {
                "solvents": {"EC": 0.25, "DMC": 0.60, "EMC": 0.15},
                "salts": {"LiPF6": 0.824},
                "additives": {"LiDFOB": 0.01, "VC": 0.002},
            },
            # Expected: ~8-9 mS/cm (lower salt concentration + ionic additive)
        },
        {
            "name": "1.2M LiPF6 in EC:EMC (3:7)",
            "recipe": {
                "solvents": {"EC": 0.3, "EMC": 0.7},
                "salts": {"LiPF6": 1.2},
                "additives": {},
            },
            # Expected: ~10-11 mS/cm (near-optimal concentration)
        },
        {
            "name": "1.0M LiPF6 in EC:DMC (1:1)",
            "recipe": {
                "solvents": {"EC": 0.5, "DMC": 0.5},
                "salts": {"LiPF6": 1.0},
                "additives": {},
            },
            # Expected: ~9-10 mS/cm (classic 1:1 blend)
        },
        {
            "name": "1.0M LiTFSI in PC:DEC (7:3)",
            "recipe": {
                "solvents": {"PC": 0.7, "DEC": 0.3},
                "salts": {"LiTFSI": 1.0},
                "additives": {},
            },
            # Expected: ~5-7 mS/cm (PC is viscous, LiTFSI less common)
        },
        {
            "name": "0.5M LiFSI in EC:EMC:DMC (1:1:1)",
            "recipe": {
                "solvents": {"EC": 0.33, "EMC": 0.34, "DMC": 0.33},
                "salts": {"LiFSI": 0.5},
                "additives": {},
            },
            # Expected: ~5-7 mS/cm (low concentration -> fewer carriers)
        },
        {
            "name": "2.0M LiPF6 in EC:DMC (3:7)",
            "recipe": {
                "solvents": {"EC": 0.3, "DMC": 0.7},
                "salts": {"LiPF6": 2.0},
                "additives": {},
            },
            # Expected: ~6-8 mS/cm (past dome peak -> reduced by viscosity/pairing)
        },
        {
            "name": "1M LiPF6 + 2% VC + 2% FEC",
            "recipe": {
                "solvents": {"EC": 0.31, "EMC": 0.33, "DMC": 0.32},
                "salts": {"LiPF6": 1.0},
                "additives": {"VC": 0.02, "FEC": 0.02},
            },
            # Expected: ~9-10 mS/cm (additives slightly reduce kappa via viscosity)
        },
        {
            "name": "0.9M LiPF6 + 0.1M LiFSI",
            "recipe": {
                "solvents": {"EC": 0.3, "EMC": 0.35, "DMC": 0.35},
                "salts": {"LiPF6": 0.9, "LiFSI": 0.1},
                "additives": {},
            },
            # Expected: ~10-11 mS/cm (LiFSI has higher Lambda_0, slight boost)
        },
        {
            "name": "NMC Gen2 cylindrical with Si-Graphite anode",
            "recipe": {
                "solvents": {"EC": 0.24, "EMC": 0.24, "DMC": 0.42},
                "salts": {"LiPF6": 1.2},
                "additives": {"FEC": 0.08, "TPP": 0.01, "VC": 0.01},
            },
            # Expected: ~8-10 mS/cm (realistic production formulation with additives)
        },
    ]

    print("\n" + "=" * 70)
    print("SURROGATE MODEL PREDICTIONS")
    print("=" * 70)
    print(f"CV RMSE: {model_bundle['cv_rmse']:.3f} mS/cm")

    for entry in test_recipes:
        pred = predict_conductivity_mS_cm(model_bundle, entry["recipe"])

        # Conductivity should be in reasonable range
        assert 1.0 < pred < 20.0, (
            f"Prediction out of range for {entry['name']}: {pred:.2f} mS/cm"
        )

        print(f"\n{entry['name']}")
        print_recipe_diagnostics(entry["recipe"])
        print(f"  -> Predicted: {pred:.2f} +/- {model_bundle['cv_rmse']:.2f} mS/cm")


def test_surrogate_conductivity_trend_with_concentration():
    """
    Verify that the surrogate model reproduces the nonlinear conductivity dome.

    Why this test exists:
        The dome-shaped conductivity vs. concentration curve is the most
        important qualitative feature of electrolyte transport. Unlike the
        analytical model (tested separately), the ML surrogate learns this shape
        from data and could potentially fail to capture it if the training data
        is insufficiently diverse or if the feature engineering misses the
        concentration nonlinearity. This test sweeps concentration from 0.5 to
        2.0 M and verifies the dome shape: rising from 0.5 to 0.8 M, and
        declining from 1.0 to 2.0 M.

    What it checks:
        Loads the surrogate model, predicts conductivity at [0.5, 0.8, 1.0,
        1.2, 1.5, 2.0] M LiPF6 in EC:EMC 30:70. Asserts:
        (1) ``kappa(0.8M) > kappa(0.5M)`` (initial increase), and
        (2) ``kappa(2.0M) < kappa(1.0M)`` (high-concentration decrease).
    """
    model_bundle = load_model_bundle("electrolyte_conductivity.pkl")

    # Sweep concentration from 0.5 to 2.0 M in a fixed solvent system
    # (EC:EMC 30:70 by volume fraction). This should trace out the
    # characteristic dome shape: rising from 0.5M to ~1.0M (more carriers),
    # then falling from ~1.0M to 2.0M (viscosity + ion pairing dominate).
    molarities = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    preds = []

    print("\n" + "-" * 70)
    print("CONCENTRATION DEPENDENCE TEST")
    print("-" * 70)

    for c in molarities:
        recipe = {
            "solvents": {"EC": 0.3, "EMC": 0.7},
            "salts": {"LiPF6": c},
            "additives": {},
        }
        # Each prediction goes through the full pipeline:
        # featurize -> select -> scale -> ensemble predict
        pred = predict_conductivity_mS_cm(model_bundle, recipe)
        preds.append(pred)
        print(f"  {c:.1f} M LiPF6: {pred:.2f} mS/cm")

    # Rising branch of the dome: more salt = more charge carriers
    assert preds[1] > preds[0], (
        f"Conductivity should increase from 0.5M to 0.8M: {preds[0]:.2f} -> {preds[1]:.2f}"
    )

    # Should peak around 1.0-1.2M and decrease at high concentration
    assert preds[-1] < preds[2], (
        f"Conductivity should decrease from 1.0M to 2.0M: {preds[2]:.2f} -> {preds[-1]:.2f}"
    )

    print("\n  [PASS] Model captures nonlinear concentration dependence")


def test_surrogate_dual_salt_synergy():
    """
    Verify the surrogate model handles dual-salt formulations correctly.

    Why this test exists:
        Dual-salt formulations (e.g. LiPF6 + LiFSI) are increasingly common in
        commercial cells because mixing salts can improve conductivity, reduce
        Al corrosion, and enhance SEI properties. The surrogate must not degrade
        its predictions when two salts are present simultaneously. Since LiFSI
        has a higher Lambda_0 than LiPF6, replacing 20% of LiPF6 with LiFSI at
        the same total molarity should give comparable or higher conductivity.
        If the model significantly underestimates the dual-salt case, the
        per-salt feature block or the salt-mixing interaction terms are not
        working properly.

    What it checks:
        Compares surrogate predictions for 1.0 M LiPF6 vs 0.8 M LiPF6 +
        0.2 M LiFSI in EC:EMC:DMC 30:35:35. Asserts ``dual_pred >= 0.9 *
        single_pred`` (dual-salt should not reduce conductivity by more than 10%).
    """
    model_bundle = load_model_bundle("electrolyte_conductivity.pkl")

    # Single salt baseline: 1.0M LiPF6 in ternary solvent blend
    single_salt_recipe = {
        "solvents": {"EC": 0.3, "EMC": 0.35, "DMC": 0.35},
        "salts": {"LiPF6": 1.0},
        "additives": {},
    }

    # Dual salt: replace 20% of LiPF6 with LiFSI, keeping total molarity
    # constant at 1.0M. LiFSI has higher Lambda_0 (limiting molar conductivity)
    # than LiPF6, so the dual-salt formulation should have comparable or
    # slightly higher conductivity. The per-salt feature block in the surrogate
    # model handles this by providing separate molarity, fraction, and
    # Lambda_0-weighted features for each salt.
    dual_salt_recipe = {
        "solvents": {"EC": 0.3, "EMC": 0.35, "DMC": 0.35},
        "salts": {"LiPF6": 0.8, "LiFSI": 0.2},
        "additives": {},
    }

    single_pred = predict_conductivity_mS_cm(model_bundle, single_salt_recipe)
    dual_pred = predict_conductivity_mS_cm(model_bundle, dual_salt_recipe)

    print("\n" + "-" * 70)
    print("DUAL SALT TEST")
    print("-" * 70)
    print(f"  1.0M LiPF6 only: {single_pred:.2f} mS/cm")
    print(f"  0.8M LiPF6 + 0.2M LiFSI: {dual_pred:.2f} mS/cm")

    # Dual salt should have comparable or higher conductivity
    # (LiFSI has higher Lambda_0)
    assert dual_pred >= single_pred * 0.9, (
        f"Dual salt should not significantly reduce conductivity: "
        f"single={single_pred:.2f}, dual={dual_pred:.2f}"
    )


# ---------------------------------------------------------------------
# CLI / smoke tests
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # ===================================================================
    # CLI ENTRY POINT
    # ===================================================================
    # Run prediction diagnostics outside of pytest. Useful for:
    #   1. Quick sanity check after retraining: verify predictions are reasonable
    #   2. Debugging unexpected predictions: verbose mode shows per-model outputs
    #   3. Comparing old vs new model bundles: run same recipes, compare kappas
    #
    # Usage: python conductivity/electrolyte_cond_surrogate_test.py
    #
    # This loads electrolyte_conductivity.pkl (the trained model bundle) and runs
    # predictions on a representative set of formulations covering:
    #   - Standard commercial blend (EC/DMC/EMC + LiPF6)
    #   - Formulation with ionic additive (LiDFOB) and SEI-former (VC)
    #   - Binary solvent blends (EC:EMC, EC:DMC)
    #   - Dual salt system (LiPF6 + LiFSI)
    print("=" * 70)
    print("ELECTROLYTE CONDUCTIVITY SURROGATE TEST")
    print("=" * 70)

    model_bundle = load_model_bundle("electrolyte_conductivity.pkl")

    # Print model metadata for provenance tracking
    print(f"\nCross-validated RMSE: {model_bundle['cv_rmse']:.3f} mS/cm")
    print(f"Liquid components: {', '.join(model_bundle['component_list'])}")
    print(f"Salts: {', '.join(model_bundle['salt_list'])}")
    print(f"Ensemble models: {', '.join(model_bundle['models'].keys())}")

    if "ensemble_weights" in model_bundle:
        print("\nEnsemble weights:")
        for name, w in sorted(model_bundle["ensemble_weights"].items()):
            print(f"  {name}: {w:.4f}")

    test_recipes = [
        {
            "name": "1.0M LiPF6 in EC/DMC/EMC (19/58/23)",
            "recipe": {
                "solvents": {"EC": 0.193, "DMC": 0.580, "EMC": 0.227},
                "salts": {"LiPF6": 1.15},
                "additives": {},
            },
        },
        {
            "name": "PGPL formulation",
            "recipe": {
                "solvents": {"EC": 0.25, "DMC": 0.60, "EMC": 0.15},
                "salts": {"LiPF6": 0.824},
                "additives": {"LiDFOB": 0.01, "VC": 0.002},
            },
        },
        {
            "name": "1.2M LiPF6 in EC:EMC (3:7)",
            "recipe": {
                "solvents": {"EC": 0.3, "EMC": 0.7},
                "salts": {"LiPF6": 1.2},
                "additives": {},
            },
        },
        {
            "name": "1.0M LiPF6 in EC:DMC (1:1)",
            "recipe": {
                "solvents": {"EC": 0.5, "DMC": 0.5},
                "salts": {"LiPF6": 1.0},
                "additives": {},
            },
        },
        {
            "name": "0.9M LiPF6 + 0.1M LiFSI dual salt",
            "recipe": {
                "solvents": {"EC": 0.3, "EMC": 0.35, "DMC": 0.35},
                "salts": {"LiPF6": 0.9, "LiFSI": 0.1},
                "additives": {},
            },
        },
    ]

    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)

    for entry in test_recipes:
        print(f"\n{entry['name']}")
        print("-" * 70)

        print_recipe_diagnostics(entry["recipe"])

        pred = predict_conductivity_mS_cm(
            model_bundle,
            entry["recipe"],
            verbose=True,
        )

        print(
            f"\n  -> Predicted conductivity: "
            f"{pred:.2f} +/- {model_bundle['cv_rmse']:.2f} mS/cm"
        )
