"""
Electrolyte conductivity surrogate model package.

Why this package exists:
    Electrolyte ionic conductivity is the single most important transport property
    for lithium-ion battery performance, but computing it from first principles
    (molecular dynamics, DFT) is too expensive for the optimization loop. A
    single DFT calculation for one formulation takes hours; the optimizer needs
    to evaluate thousands of formulations per run. This package provides a
    machine-learned surrogate that predicts conductivity in microseconds from a
    recipe specification (solvents, salts, additives with their loadings),
    enabling rapid formulation screening and gradient-based optimization in the
    electrolyte design pipeline.

    DECISION RATIONALE: A surrogate model was chosen over (a) lookup tables
    (too sparse for the high-dimensional formulation space), (b) analytical
    models like Casteel-Amis alone (good for single-salt but cannot capture
    multi-salt, multi-solvent, and additive interactions accurately), and
    (c) neural network potentials (overkill for this prediction task and
    harder to train on small datasets). The ensemble of tree models + physics-
    motivated features achieves <0.8 mS/cm RMSE on ~155 unique recipes.

What it contains:
    - electrolyte_utils_features: Pure-function utilities for recipe parsing,
      mixture property aggregation, and physics-motivated featurization. This is
      the single source of truth for converting a recipe dict into a fixed-length
      numeric feature vector consumed by the surrogate models. All feature
      engineering decisions (polynomial Li terms, Nernst-Einstein proxies,
      salt quality features) are centralized here.

    - electrolyte_cond_surrogate_train: Training pipeline that reads experimental
      conductivity data from ``data.electrolyte_property_db``, featurizes all
      recipes, trains a weighted ensemble of diverse ML models (GBM, RF,
      ExtraTrees), performs GroupKFold cross-validation (by recipe) to prevent
      data leakage, and serializes the model bundle to a pickle file.

    - electrolyte_cond_surrogate_test: Prediction utilities (``load_model_bundle``,
      ``predict_conductivity_mS_cm``) and a comprehensive pytest test suite that
      validates surrogate predictions against physics monotonicity constraints
      (viscosity bracketing, concentration dome, Lambda_0 ordering) and
      experimental data points.

Data flow:
    recipe dict --> featurize_recipe() --> np.ndarray --> scaler.transform()
    --> ensemble.predict() --> weighted average --> conductivity [mS/cm]

    The feature vector includes ~100+ features organized in 12 blocks:
    core Li concentration terms, mixture physics, solvent structure, salt
    aggregates, per-salt features, component fractions, additive proxies,
    transport couplings, Nernst-Einstein proxies, and enhanced physics features.

Config dependencies:
    - Species properties are looked up via ``data.species_fns.get_species_property``
      (density_g_ml, epsilon_r, viscosity_cP, donor_number, Lambda_0, etc.).
    - No config JSON files are read directly by this package.
    - The serialized model bundle (.pkl) is the only artifact dependency at
      inference time.

Architecture role in the electrolyte design pipeline:
    This package is the primary evaluator of ionic conductivity, called by:
    1. ``electrolyte_model.ElectrolyteFormulation`` -- surrogate computes kappa_0
       (base conductivity at 25C). Replaces the analytical Casteel-Amis model
       because the surrogate captures multi-salt and additive interactions.
    2. ``mixture_optimizer.py`` -- DSPy ReAct agent conductivity surrogate sweep
       evaluates hundreds of solvent ratio candidates in <1s.
    3. ``electrolyte_design_pipeline.py`` -- Stage 5 forward selection and DE
       optimization (~10,000 evaluations per DE run).

    Separation of concerns:
    - ``electrolyte_utils_features.py``: ONLY featurization (pure functions)
    - ``electrolyte_cond_surrogate_train.py``: ONLY training (run once offline)
    - ``electrolyte_cond_surrogate_test.py``: Prediction API + validation tests

    Training flow (offline): DATA -> featurize -> GroupKFold CV -> fit -> .pkl
    Inference flow (online): .pkl -> featurize -> scale -> predict -> kappa

Physical background on ionic conductivity:
    kappa depends on ion concentration c [mol/L], solvent viscosity eta [cP],
    dielectric constant epsilon_r, salt quality Lambda_0 [S*cm^2/mol], ion
    pairing binding energy [kJ/mol], and temperature T [K]. These produce
    the characteristic dome shape: kappa rises with c at low concentrations
    (more charge carriers), peaks around 1.0-1.2 M for most Li salts in
    carbonates, then falls at high c due to viscosity increase and ion pairing.
    The surrogate captures this via polynomial Li features (Li, Li^2, Li^3,
    log1p(Li)) and physics-motivated interaction terms.
"""
