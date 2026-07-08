# electrolyte_utils_features.py
"""
Shared electrolyte utilities and featurization for the conductivity surrogate.

MOTIVATION:
    The conductivity surrogate model needs a stable, physics-motivated mapping
    from electrolyte recipes (dicts of species names and loadings) to fixed-length
    numeric feature vectors. This mapping encapsulates all of the domain knowledge
    about what drives ionic conductivity: Li+ concentration (and its nonlinear
    effects), solvent dielectric constant (dissociation), viscosity (mobility),
    salt quality (Lambda_0), ion pairing (binding energy), and structural
    descriptors (cyclic vs. linear carbonate balance). By encoding these physics
    priors as explicit features, we give the ML models a strong inductive bias
    that dramatically improves accuracy on the small training dataset (~155
    unique recipes).

    This module also provides composition utilities (get_component_fractions,
    get_total_li_molarity, compute_mixture_properties) used by both the
    surrogate and the broader electrolyte model. Centralizing them here ensures
    a single source of truth for basis conversions and mixture averaging.

DECISION RATIONALE:
    - Pure functions only (no model objects, no sklearn, no state): ensures
      featurization is deterministic and side-effect-free, making it safe to
      call from any context (training, inference, tests, pipeline stages).
    - All species properties are looked up via ``get_species_property()`` from
      the species database -- no hardcoded values in this module. This means
      adding a new solvent or salt to the database automatically makes it
      available to the surrogate without code changes.
    - The feature vector is organized in 12 blocks with stable ordering. This
      ordering is critical: the model bundle's scaler and feature selection
      indices are coupled to the exact feature positions. Any change to the
      block ordering or feature count requires retraining.

This module is the single source of truth for:
- electrolyte composition parsing (recipe dict -> normalized fractions)
- mixture property aggregation (weighted averages of dielectric, viscosity, etc.)
- component discovery (scan dataset for all species names)
- conductivity surrogate featurization (recipe -> fixed-length numeric vector)

RULES:
- No model objects (sklearn estimators, etc.)
- No sklearn / training logic (fitting, cross-validation, etc.)
- No PyBaMM (simulation logic)
- No DSPy (agent/tool logic)
- Pure functions only (no global state, no side effects)
"""

from typing import Dict, List, Tuple
import numpy as np

from constants import T_REF_K
from species_fns import get_species_property


# ---------------------------------------------------------------------
# Core composition utilities
# ---------------------------------------------------------------------


def discover_components(data) -> Tuple[List[str], List[str]]:
    """
    Discover all unique non-ionic liquid components and salts from a dataset.

    Why this exists:
        The conductivity surrogate model needs a fixed, reproducible ordering of
        all species that appear across the training dataset. This function scans
        every recipe row once to build canonical sorted lists of (a) liquid
        components (solvents + non-ionic additives) and (b) salts (ionic species).
        These lists define the feature vector dimensionality: each component and
        salt gets dedicated feature slots in ``featurize_recipe()``, so the
        ordering must be stable and deterministic (hence sorted output).

    What it does:
        Iterates over all rows in ``data``, inspecting the ``recipe`` sub-dict.
        Solvents are always classified as liquid components. Additives are
        classified as salts if ``get_species_property(name, "provides_ionic_conductivity")``
        is True, otherwise as liquid components. Explicit salts (under the
        ``"salts"`` key) are always classified as salts.

    Args:
        data: List of dataset rows, each a dict with at minimum a ``"recipe"``
            key containing ``{"solvents": {name: frac}, "salts": {name: mol},
            "additives": {name: frac}}``. Typically sourced from
            ``data.electrolyte_property_db.DATA``.

    Returns:
        Tuple of two sorted lists:
            - component_list: Sorted list of unique solvent and non-ionic additive
              names (e.g. ``["DMC", "EC", "EMC", "FEC", "VC"]``).
            - salt_list: Sorted list of unique salt / ionic additive names
              (e.g. ``["LiDFOB", "LiFSI", "LiPF6", "LiTFSI"]``).

    Side effects:
        None. Pure function with no mutations.
    """
    # Accumulate unique species names across all dataset rows.
    # Using sets ensures each species is counted only once regardless of how
    # many recipes contain it.
    components = set()  # Solvents + non-ionic additives (liquid phase species)
    salts = set()  # Ionic species that contribute Li+ charge carriers

    for row in data:
        recipe = row["recipe"]

        # All solvents are liquid-phase components by definition
        for name in recipe.get("solvents", {}).keys():
            components.add(name)

        # Additives must be classified: ionic additives (e.g., LiDFOB) are
        # functionally salts (they dissociate to provide Li+), while non-ionic
        # additives (e.g., FEC, VC, TPP) modify the liquid phase properties
        for name in recipe.get("additives", {}).keys():
            if get_species_property(name, "provides_ionic_conductivity"):
                salts.add(name)  # Ionic additive -> treated as salt for features
            else:
                components.add(name)  # Non-ionic -> liquid phase component

        # Explicit salts are always ionic species
        for name in recipe.get("salts", {}).keys():
            salts.add(name)

    # Sorting ensures deterministic, reproducible feature vector ordering.
    # This is critical: the model bundle's scaler and feature selection indices
    # are coupled to the exact feature positions.
    return sorted(components), sorted(salts)


def get_training_species() -> Tuple[List[str], List[str]]:
    """Return the species present in the XGB conductivity surrogate training data.

    This loads the merged training dataset (original + CALiSol at 25°C) and
    discovers all unique components and salts. These are the ONLY species for
    which the XGB model produces in-distribution predictions. Any species not
    in these lists will cause OOD extrapolation with unreliable gradients.

    Returns:
        Tuple of two sorted lists:
            - component_list: Sorted liquid-phase species (solvents + non-ionic additives).
            - salt_list: Sorted ionic species (salts + ionic additives).
    """
    from conductivity.electrolyte_cond_surrogate_train import DATA

    return discover_components(DATA)


def get_component_fractions(recipe: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute normalized weight/volume fractions of liquid components only.

    Why this exists:
        Many mixture-property calculations (dielectric constant, viscosity, donor
        number) require knowing the relative proportions of the liquid phase
        only, excluding dissolved salts. This function extracts solvents and
        non-ionic additives from a recipe, then normalizes their fractions to
        sum to 1.0 so they can be used as mixing-rule weights.

    What it does:
        Collects all solvents and non-ionic additives (those for which
        ``get_species_property(name, "provides_ionic_conductivity")`` is False)
        from the recipe. Normalizes their raw fractions so they sum to unity.
        Ionic additives (e.g. LiDFOB) are excluded because they are salts, not
        part of the liquid solvent blend.

    Args:
        recipe: Dict with keys ``"solvents"`` and optionally ``"additives"``,
            each mapping species names to their raw fraction values (weight or
            volume fraction depending on convention).

    Returns:
        Dict mapping each liquid component name to its normalized fraction
        (summing to 1.0). Returns an empty dict if no liquid components exist
        or their total fraction is <= 0.

    Side effects:
        None. Pure function.
    """
    # Collect all liquid-phase species: solvents and non-ionic additives.
    # Ionic additives (salts stored under "additives" key) are excluded because
    # they are dissolved species, not part of the liquid solvent blend.
    components: Dict[str, float] = {}

    # Solvents are always part of the liquid phase
    for name, frac in recipe.get("solvents", {}).items():
        components[name] = float(frac)

    # Non-ionic additives (FEC, VC, TPP, etc.) are part of the liquid phase.
    # Ionic additives (LiDFOB, etc.) are excluded -- they are salt species.
    for name, frac in recipe.get("additives", {}).items():
        if not get_species_property(name, "provides_ionic_conductivity"):
            components[name] = float(frac)

    # Normalize so fractions sum to 1.0 for use as mixing-rule weights.
    # This normalization is necessary because raw recipe fractions may not
    # sum to 1.0 (e.g., if additives are specified as wt% of total electrolyte
    # rather than wt% of solvent blend).
    total = sum(components.values())
    if total <= 0.0:
        return {}

    return {k: v / total for k, v in components.items()}


def get_total_li_molarity(
    recipe: Dict[str, Dict[str, float]],
    electrolyte_density_g_ml: float = 1.2,
) -> float:
    """
    Compute total lithium-ion molarity from all ionic sources in a recipe.

    Why this exists:
        The total Li+ concentration is the primary driver of ionic conductivity
        (via the Nernst-Einstein relation) and appears in multiple surrogate
        features. Recipes store salts as molarity (mol/L) but ionic additives
        as weight fractions (g/g electrolyte), so a unit-conversion step is
        needed to combine them into a single total Li+ molarity value.

    What it does:
        1. Sums explicit salt molarities directly from ``recipe["salts"]``.
        2. For ionic additives (``provides_ionic_conductivity=True``), converts
           their weight fractions to molarity via:
           ``molarity = (wt_frac * density [g/mL] * 1000) / molecular_weight``.
        3. Returns the combined total assuming 1 Li+ per formula unit.

    Args:
        recipe: Dict with optional ``"salts"`` (name -> molarity in mol/L) and
            ``"additives"`` (name -> weight fraction in g/g electrolyte).
        electrolyte_density_g_ml: Bulk electrolyte density used for wt-frac to
            molarity conversion of ionic additives. Defaults to 1.2 g/mL as a
            reasonable estimate for carbonate electrolytes.

    Returns:
        Total Li+ molarity in mol/L (float). This assumes monovalent Li salts
        (1 Li+ per formula unit).

    Raises:
        ValueError: If an ionic additive is missing its ``molecular_weight``
            in the species database.

    Side effects:
        None. Pure function.
    """

    total_li = 0.0

    # 1. Explicit salts (already molarity)
    for name, molarity in recipe.get("salts", {}).items():
        required_species_property(name, "molecular_weight")
        total_li += float(molarity)

    # 2. Ionic additives: wt_fraction -> molarity
    for name, wt_frac in recipe.get("additives", {}).items():
        if not get_species_property(name, "provides_ionic_conductivity"):
            continue

        mw = get_species_property(name, "molecular_weight")
        if mw is None:
            raise ValueError(f"Missing molecular_weight for ionic additive '{name}'")

        # wt_frac [g/g] × density [g/mL] × 1000 → g/L → mol/L
        molarity = (float(wt_frac) * electrolyte_density_g_ml * 1000.0) / mw
        total_li += molarity

    return float(total_li)


def required_species_property(species_name: str, property_name: str):
    property_value = get_species_property(species_name, property_name)
    if property_value is None:
        raise ValueError(f"missing species property {species_name}.{property_name}")
    return property_value


# ---------------------------------------------------------------------
# Mixture property aggregation
# ---------------------------------------------------------------------


def compute_mixture_properties(recipe: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute linear mixing-rule averaged properties of the liquid phase.

    Why this exists:
        The conductivity surrogate and the broader electrolyte model both need
        mixture-level dielectric constant, viscosity, and donor number as
        diagnostic inputs. These are the three most important liquid-phase
        properties governing ion dissociation (epsilon_r), ion mobility
        (viscosity), and Li+ solvation strength (donor_number). Computing them
        centrally here avoids duplication across the pipeline.

    What it does:
        Collects solvents and non-ionic additives from the recipe, then computes
        a simple linear (weight/volume-fraction) weighted average of each
        property using values from the species database. This is an ideal-mixing
        approximation; non-ideal corrections are handled downstream by the
        surrogate model's learned residuals.

    Args:
        recipe: Dict with ``"solvents"`` and optionally ``"additives"`` keys,
            each mapping species names to fraction values. Ionic additives are
            excluded from the liquid-phase average.

    Returns:
        Dict with three keys:
            - ``"epsilon_r"`` (float): Mixture dielectric constant (dimensionless).
            - ``"viscosity_cP"`` (float): Mixture viscosity in centipoise.
            - ``"donor_number"`` (float): Mixture Gutmann donor number (kcal/mol).

    Raises:
        ValueError: If no liquid components are found in the recipe.

    Side effects:
        None. Pure function.
    """
    # Collect liquid-phase components (solvents + non-ionic additives)
    components: Dict[str, float] = {}

    for name, frac in recipe.get("solvents", {}).items():
        components[name] = float(frac)

    for name, frac in recipe.get("additives", {}).items():
        if not get_species_property(name, "provides_ionic_conductivity"):
            components[name] = float(frac)

    if not components:
        raise ValueError("No liquid components in recipe")

    total = sum(components.values())

    # LINEAR MIXING RULE: property_mix = sum(x_i * property_i) / sum(x_i)
    # This is an ideal-mixing approximation. For dielectric constant, the
    # exact mixing rule depends on molecular geometry (Clausius-Mossotti),
    # but linear mixing is a reasonable first approximation for similar-polarity
    # solvents. Non-ideal corrections are learned by the ML surrogate.

    # Dielectric constant (epsilon_r): drives ion dissociation.
    # High epsilon (EC=89.8) promotes dissociation of Li+ from anions.
    # Low epsilon (DMC=3.1) reduces dissociation. The balance between
    # high-epsilon cyclic carbonates and low-epsilon linear carbonates is
    # the primary design lever for tuning conductivity.
    epsilon_r = (
        sum(
            frac * float(get_species_property(name, "epsilon_r"))
            for name, frac in components.items()
        )
        / total
    )

    # Viscosity (cP): inversely related to ion mobility.
    # Stokes-Einstein relation: D = kT / (6*pi*eta*r), so lower viscosity
    # means faster ion diffusion and higher conductivity. Cyclic carbonates
    # (EC: 1.9 cP) are more viscous than linear ones (DMC: 0.59 cP).
    viscosity = (
        sum(
            frac * float(get_species_property(name, "viscosity_cP"))
            for name, frac in components.items()
        )
        / total
    )

    # Donor number (kcal/mol): measures Lewis basicity / electron-pair
    # donating ability. Higher donor number means stronger Li+ solvation,
    # which promotes dissociation but can also slow Li+ transfer at interfaces.
    # Used as a diagnostic feature in the surrogate model.
    donor_number = (
        sum(
            frac * float(get_species_property(name, "donor_number"))
            for name, frac in components.items()
        )
        / total
    )

    return {
        "epsilon_r": float(epsilon_r),
        "viscosity_cP": float(viscosity),
        "donor_number": float(donor_number),
    }


# ---------------------------------------------------------------------
# Conductivity surrogate featurization
# ---------------------------------------------------------------------


def featurize_recipe(
    recipe: Dict[str, Dict[str, float]],
    component_list: List[str],
    salt_list: List[str],
) -> np.ndarray:
    """
    Convert an electrolyte recipe into a physics-motivated numeric feature vector.

    Why this exists:
        The conductivity surrogate is a machine-learned model that needs a fixed-
        length numeric input. Naive featurization (just species fractions) misses
        the physics that governs conductivity: ion mobility depends on viscosity
        and dielectric constant, dissociation depends on donor number and epsilon,
        and salt quality depends on Lambda_0 and ion-pair binding. This function
        encodes all of these physics priors as explicit features, giving the ML
        models a strong inductive bias and dramatically improving accuracy on
        small datasets (~155 unique recipes).

    What it does:
        Constructs a feature vector in 12 blocks (stable ordering):
        1. **Core Li features** (4): Li_total, Li_total^2, Li_total^3, log1p(Li_total)
           -- polynomial + log terms to capture the dome-shaped concentration response.
        2. **Mixture physics** (7): epsilon_mix, viscosity_mix, donor_mix,
           density_mix, dipole_mix, acceptor_mix, mobility (epsilon/viscosity)
           -- liquid-phase-weighted averages of species properties.
        3. **Solvent structure** (4): cyclic_frac, linear_frac, emc_minus_dmc,
           cyclic_over_linear -- structural descriptors capturing the cyclic vs
           linear carbonate balance that controls the epsilon-viscosity tradeoff.
        4. **Salt aggregates** (5): total_salt_molarity, ionic_additive_molarity_sum,
           avg_Lambda0, avg_binding, avg_anion_vol -- molarity-weighted salt properties.
        5. **Per-salt block** (5 * len(salt_list)): For each salt: molarity,
           molarity fraction, molarity*Lambda_0, molarity*binding, molarity*anion_vol
           -- allows the model to learn salt-specific nonlinearities.
        6. **Component fractions** (len(component_list)): Liquid-phase mass fraction
           of each solvent/non-ionic additive.
        7. **Component additive proxies** (4 * len(component_list)): For each
           component: frac*LiF_yield, frac*gas_yield, frac*E_red, frac*is_cei
           -- chemical effect proxies weighted by loading.
        8. **Transport couplings** (4): Li/viscosity, Li*mobility,
           mobility*donor, viscosity^2 -- physically motivated interaction terms.
        9. **Nernst-Einstein proxies** (2): kappa_nernst_proxy (c*Lambda_0/eta),
           kappa_with_dissociation (proxy * epsilon/40) -- physics baseline for
           the model to learn corrections from.
        10. **Enhanced physics features** (8): Lambda0_sq, Lambda0_ratio (vs
            LiPF6), Lambda0/eta, Lambda0*epsilon, conc_deviation_from_1M,
            eta*c, eta*c^2, kappa_composite -- address systematic
            underestimation of high-conductivity formulations by capturing
            nonlinear salt quality and salt-solvent interactions.

    Basis conversion details:
        - recipe["salts"] values are molarity (mol/L) per project convention.
        - recipe["solvents"] values are volume fractions within the solvent blend.
        - recipe["additives"] non-ionic values are weight fractions of total
          electrolyte; ionic additives are converted to mol/L via mixture density.
        - All species properties come from ``get_species_property()`` -- no
          hardcoded values.

    Args:
        recipe: Dict with keys ``"solvents"`` (name -> vol fraction),
            ``"salts"`` (name -> molarity mol/L), ``"additives"``
            (name -> wt fraction). Missing keys default to empty dicts.
        component_list: Canonical sorted list of liquid component names, as
            returned by ``discover_components()``. Defines feature vector slot
            ordering for components.
        salt_list: Canonical sorted list of salt names, as returned by
            ``discover_components()``. Defines feature vector slot ordering for
            per-salt features.

    Returns:
        1-D numpy array of dtype float with length:
            4 + 7 + 4 + 5 + 5*len(salt_list) + len(component_list)
            + 4*len(component_list) + 4 + 2 + 8
        Feature names in matching order are produced by ``get_feature_names()``.

    Raises:
        ValueError: If a solvent is missing ``density_g_ml``, if there are no
            solvents with positive volume fraction, if liquid fractions are
            non-positive (salt + additives exceed 100%), or if LiPF6 Lambda_0
            is missing from the species database.

    Side effects:
        None. Pure function (reads species database but does not mutate it).
    """

    # ========================
    # BLOCK 1: SALT MOLARITIES
    # ========================
    # Two sources of ionic species in a recipe:
    #   (a) Explicit salts under recipe["salts"] -- stored as molarity (mol/L)
    #   (b) Ionic additives under recipe["additives"] -- stored as wt fraction
    # Both contribute Li+ charge carriers and must be unified into mol/L.

    # (a) Explicit salts: extract molarity for each salt in the canonical salt_list.
    # Missing salts default to 0.0 mol/L.
    salt_m = {s: float(recipe.get("salts", {}).get(s, 0.0)) for s in salt_list}

    # (b) Ionic additives need wt% -> mol/L conversion, which requires the
    # mixture density (computed below in block 3). For now, just collect their
    # weight fractions. The actual conversion happens in block 4.
    ionic_additive_wt = {}
    for name, wt in recipe.get("additives", {}).items():
        if get_species_property(name, "provides_ionic_conductivity"):
            ionic_additive_wt[name] = float(wt)

    # ------------------------
    # 2) Liquid composition — basis conversion
    # ------------------------
    # Solvents are v/v of solvent blend.  Additives are w/w of total electrolyte.
    # Convert both to mass fraction of the liquid phase.

    # 2a) Solvent v/v → mass fractions within blend (via density)
    solvent_blend_mass = {}
    total_blend_mass = 0.0
    for s, v in recipe.get("solvents", {}).items():
        rho = get_species_property(s, "density_g_ml")
        if rho is None or float(rho) <= 0:
            raise ValueError(f"Solvent '{s}' missing density_g_ml for basis conversion")
        m = float(v) * float(rho)
        solvent_blend_mass[s] = m
        total_blend_mass += m

    if total_blend_mass <= 0.0:
        raise ValueError("No solvents with positive vol_fraction and density")

    solvent_blend_wt = {k: m / total_blend_mass for k, m in solvent_blend_mass.items()}

    # 2b) Salt mass fraction from molarity × MW
    rho_blend = total_blend_mass  # Σ(v_i × ρ_i)  [g/mL]
    salt_mass_per_L = 0.0
    for s in salt_list:
        mol = float(recipe.get("salts", {}).get(s, 0.0))
        if mol <= 0:
            continue
        mw = get_species_property(s, "molecular_weight")
        if mw is None or float(mw) <= 0:
            continue
        salt_mass_per_L += mol * float(mw)
    # Total electrolyte mass per liter: solvent mass + dissolved salt mass
    # rho_blend * 1000 converts g/mL to g/L
    total_mass_per_L = rho_blend * 1000.0 + salt_mass_per_L
    # w_salt: weight fraction of salt in total electrolyte. Typically ~10-15%
    # for 1M LiPF6 (MW=151.9, so 151.9 g/L in ~1200 g/L total = ~13%)
    w_salt = salt_mass_per_L / total_mass_per_L if total_mass_per_L > 0 else 0.0

    # 2c) Total additive weight fraction and non-ionic subset.
    # Separate ionic additives (already handled as salts above) from non-ionic
    # additives (which are part of the liquid phase).
    w_a = 0.0
    non_ionic_additives = {}
    for a, v in recipe.get("additives", {}).items():
        wf = float(v)
        w_a += wf
        if not get_species_property(a, "provides_ionic_conductivity"):
            non_ionic_additives[a] = wf

    # 2d) Convert to liquid-phase mass fractions.
    # liquid_frac = fraction of total electrolyte that is liquid (not salt)
    # solvent_blend_frac = fraction that is pure solvent (not salt, not additive)
    # The liquid phase = solvents + non-ionic additives
    liquid_frac = 1.0 - w_salt
    solvent_blend_frac = 1.0 - w_a - w_salt
    if liquid_frac <= 0 or solvent_blend_frac <= 0:
        raise ValueError(
            f"Non-positive fractions: liquid={liquid_frac:.4f}, "
            f"solvent_blend={solvent_blend_frac:.4f}"
        )

    # Final liquid-phase weight fractions: each solvent's contribution to the
    # liquid phase is its blend fraction * (solvent share / liquid share).
    # Non-ionic additives are simply their wt_frac normalized by liquid_frac.
    liquid_w = {}
    for name, blend_wt in solvent_blend_wt.items():
        liquid_w[name] = blend_wt * solvent_blend_frac / liquid_frac
    for name, wf in non_ionic_additives.items():
        liquid_w[name] = wf / liquid_frac

    # =============================
    # BLOCK 3: MIXTURE DENSITY
    # =============================
    # Compute liquid-phase density using the inverse-volume mixing rule:
    #   1/rho_mix = sum(w_i / rho_i)
    # This assumes ideal mixing (no excess volume on mixing), which is
    # reasonable for carbonate mixtures. The density is needed for converting
    # ionic additive weight fractions to molarities (block 4).
    inv_vol = 0.0
    for k, w in liquid_w.items():
        rho = get_species_property(k, "density_g_ml")
        if rho is None or rho == 0.0:
            raise ValueError(
                f"Species '{k}' missing density_g_ml for density calculation"
            )
        inv_vol += w / float(rho)  # Additive specific volumes
    # Invert to get density: rho_mix = 1 / sum(w_i/rho_i)
    density_mix_g_ml = 1.0 / inv_vol if inv_vol > 0 else 1.0

    # ================================================
    # BLOCK 4: IONIC ADDITIVE WT FRACTION -> MOLARITY
    # ================================================
    # Now that we have the mixture density, we can convert ionic additive
    # weight fractions to molarities. The conversion chain:
    #   wt_frac [g_salt/g_electrolyte] * rho_mix [g_electrolyte/mL] * 1000 [mL/L]
    #   = mass_concentration [g_salt/L_electrolyte]
    #   / MW [g_salt/mol_salt] = molarity [mol/L]
    ionic_additive_m = {}
    for name, wt_frac in ionic_additive_wt.items():
        mw = get_species_property(name, "molecular_weight")
        if mw is None or mw == 0.0:
            # skip if missing molecular weight -- cannot convert
            continue
        molar = (float(wt_frac) * density_mix_g_ml * 1000.0) / float(mw)
        ionic_additive_m[name] = molar

    # Merge explicit salts and ionic additives into a unified per-salt molarity
    # dict. This is the canonical salt concentration used for all downstream
    # feature calculations. A salt might appear in both sources (e.g., LiDFOB
    # could be an explicit salt in one recipe and an additive in another).
    per_salt_molarity = {}
    for s in salt_list:
        per_salt_molarity[s] = salt_m.get(s, 0.0) + ionic_additive_m.get(s, 0.0)

    # Total salt molarity: sum of all Li salt contributions [mol/L]
    total_salt_molarity = sum(per_salt_molarity.values())

    # ===================================
    # BLOCK 5: TOTAL Li+ CONCENTRATION
    # ===================================
    # Li_total [mol/L] is the single most important feature for conductivity.
    # The Nernst-Einstein relation: kappa ~ F^2 * c * D / (R * T)
    # shows that conductivity is proportional to ion concentration c at low c.
    # At high c, viscosity increase and ion pairing reduce D, creating the dome.
    #
    # ASSUMPTION: Each salt formula unit contributes exactly 1 Li+ ion.
    # This holds for all common Li salts (LiPF6, LiFSI, LiTFSI, LiBOB, LiDFOB).
    Li_total = float(total_salt_molarity)

    # ==========================================================
    # BLOCK 6: MIXTURE-AVERAGED PHYSICAL PROPERTIES
    # ==========================================================
    # These liquid-phase properties are the primary drivers of conductivity:
    # - epsilon_mix: dielectric constant, controls ion dissociation
    # - viscosity_mix: controls ion mobility (Stokes-Einstein)
    # - donor_mix: donor number, controls Li+ solvation strength
    # - dipole_mix: dipole moment, correlates with solvation shell geometry
    # - acceptor_mix: acceptor number, relates to anion solvation
    def safe_get(prop_name, k):
        """Retrieve a numeric species property, returning None if absent.

        Args:
            prop_name: Property key to look up in the species database.
            k: Species name to query.

        Returns:
            Float value of the property, or None if the species does not have
            the requested property.
        """
        val = get_species_property(k, prop_name)
        return None if val is None else float(val)

    # For averaging, ignore missing props
    def weighted_avg(prop):
        """Compute mass-fraction-weighted average of a species property.

        Skips species that lack the requested property, re-normalizing
        weights over the remaining species so the average is still valid.

        Args:
            prop: Property key string (e.g. ``"epsilon_r"``, ``"viscosity_cP"``).

        Returns:
            Weighted average (float). Returns 0.0 if no species provide the
            property.
        """
        num = 0.0
        denom = 0.0
        for k, w in liquid_w.items():
            v = safe_get(prop, k)
            if v is None:
                continue
            num += w * v
            denom += w
        return (num / denom) if denom > 0 else 0.0

    epsilon_mix = weighted_avg("epsilon_r")
    viscosity_mix = weighted_avg("viscosity_cP")
    donor_mix = weighted_avg("donor_number")
    dipole_mix = weighted_avg("dipole_moment_D")
    acceptor_mix = weighted_avg("acceptor_number")

    # ==========================================================
    # SALT AGGREGATE PROPERTIES (molarity-weighted averages)
    # ==========================================================
    # These aggregate features capture "average salt quality" without
    # per-species identity features. For single-salt systems, these equal
    # the salt's own properties. For dual-salt, they're molarity-weighted.
    weighted_Lambda0_num = 0.0
    weighted_binding_num = 0.0
    weighted_anionvol_num = 0.0

    for s in salt_list:
        mol = float(per_salt_molarity.get(s, 0.0))
        Lambda0 = float(get_species_property(s, "Lambda_0", 0.0) or 0.0)
        binding = float(get_species_property(s, "ion_pair_binding_kj_mol", 0.0) or 0.0)
        anion_vol = float(get_species_property(s, "anion_volume", 0.0) or 0.0)

        weighted_Lambda0_num += mol * Lambda0
        weighted_binding_num += mol * binding
        weighted_anionvol_num += mol * anion_vol

    avg_Lambda0 = (
        (weighted_Lambda0_num / total_salt_molarity) if total_salt_molarity > 0 else 0.0
    )
    avg_binding = (
        (weighted_binding_num / total_salt_molarity) if total_salt_molarity > 0 else 0.0
    )
    avg_anion_vol = (
        (weighted_anionvol_num / total_salt_molarity)
        if total_salt_molarity > 0
        else 0.0
    )

    # ==========================================================
    # BLOCK 9b: SALT HETEROGENEITY FEATURES (multi-salt physics)
    # ==========================================================
    # These features capture WHY dual/triple-salt formulations behave
    # differently from single-salt, without naming specific species.
    # When salts with different anion properties mix:
    #   - Lambda0 spread: high-mobility anion disrupts ion pairs of the other
    #   - Binding spread: weakly-bound salt suppresses pairing of strongly-bound
    #   - Anion volume spread: size mismatch disrupts correlated ion ordering
    #   - Entropy: well-mixed salts maximize disruption of long-range order
    #
    # For single-salt systems, all spread/entropy features are exactly 0,
    # letting the model learn a clean single-salt vs multi-salt split.

    # Collect per-salt property vectors for active salts
    _active_Lambda0s = []
    _active_bindings = []
    _active_anion_vols = []
    _active_molarities = []
    for s in salt_list:
        mol_s = float(per_salt_molarity.get(s, 0.0))
        if mol_s <= 0.01:  # Same threshold as n_salts_active
            continue
        _active_Lambda0s.append(float(get_species_property(s, "Lambda_0", 0.0) or 0.0))
        _active_bindings.append(
            float(get_species_property(s, "ion_pair_binding_kj_mol", 0.0) or 0.0)
        )
        _active_anion_vols.append(
            float(get_species_property(s, "anion_volume", 0.0) or 0.0)
        )
        _active_molarities.append(mol_s)

    _n_active = len(_active_molarities)

    # Property spreads: max - min across active salts (0 for single-salt)
    Lambda0_spread = (
        (max(_active_Lambda0s) - min(_active_Lambda0s)) if _n_active >= 2 else 0.0
    )
    binding_spread = (
        (max(_active_bindings) - min(_active_bindings)) if _n_active >= 2 else 0.0
    )
    anion_vol_spread = (
        (max(_active_anion_vols) - min(_active_anion_vols)) if _n_active >= 2 else 0.0
    )

    # Salt mixing entropy: -Σ(f_i × ln(f_i)) where f_i = mol_i / mol_total
    # Maximum at equal molar split, 0 for single-salt. Captures mixing uniformity.
    salt_entropy = 0.0
    if _n_active >= 2:
        _mol_total = sum(_active_molarities)
        for _m in _active_molarities:
            _f = _m / _mol_total
            if _f > 1e-12:
                salt_entropy -= _f * np.log(_f)

    # ==========================================================
    # BLOCK 9c: CONCENTRATION CLIFF FEATURES
    # ==========================================================
    # The conductivity dome has a steep cliff above the peak concentration.
    # Trees need very fine resolution to capture this. Pre-computing
    # signed deviation powers gives single-split access to the cliff shape.
    # |c - 1| already exists (conc_deviation_from_1M), but it's unsigned.
    # Signed powers capture the ASYMMETRY: conductivity drops faster
    # above the peak than it rises below it.
    conc_deviation_signed = Li_total - 1.0  # Positive = above 1M
    conc_deviation_sq = conc_deviation_signed ** 2  # Symmetric quadratic penalty
    conc_deviation_cubed = conc_deviation_signed ** 3  # Asymmetric: + above, - below

    # ==========================================================
    # BLOCK 10: TRANSPORT AND DISSOCIATION COUPLING FEATURES
    # ==========================================================
    # These are physically motivated interaction terms that capture how
    # different properties combine to affect conductivity. Individual
    # properties (epsilon, viscosity, Li concentration) are necessary but
    # not sufficient: conductivity depends on their INTERACTIONS.
    #
    # mobility proxy = epsilon / viscosity:
    #   HIGH epsilon means more free ions (dissociation), LOW viscosity means
    #   faster ion movement. Their ratio is proportional to the product of
    #   dissociation fraction and diffusion coefficient. This is the single
    #   most predictive simple feature for conductivity.
    mobility = epsilon_mix / (viscosity_mix + 1e-12)  # [dimensionless/cP]
    # Li/viscosity: concentration-limited mobility. At low Li, this is small
    # (few carriers); at high viscosity, this is small (slow carriers).
    Li_over_visc = Li_total / (viscosity_mix + 1e-12)  # [mol/(L*cP)]
    # Li * mobility: combines concentration with the epsilon/viscosity ratio.
    # This is a simplified Nernst-Einstein-like term.
    Li_times_mobility = Li_total * mobility
    # mobility * donor: captures the synergy between dissociation (epsilon/eta)
    # and solvation strength (donor number). Strong solvation + high mobility
    # = well-dissociated, fast-moving ions.
    mobility_times_donor = mobility * donor_mix

    # ------------------------
    # 10b) Physics-based conductivity estimate (Nernst-Einstein approximation)
    #      κ ≈ F² × c × Λ₀ × α / (R × T × η)
    #      Simplified: κ_est ∝ c × Λ₀ / η
    #      This gives the model a physics baseline to learn corrections from
    # ------------------------
    # Faraday constant squared / (R * T) at 25°C ≈ 3.76e6 S·mol/(cm·L)
    # Simplified dimensionless proxy (will be scaled by model):
    kappa_nernst_proxy = (Li_total * avg_Lambda0) / (viscosity_mix + 1e-12)

    # Also include dissociation correction term (high epsilon → better dissociation)
    dissociation_factor = epsilon_mix / 40.0  # Normalized to EC's epsilon
    kappa_with_dissociation = kappa_nernst_proxy * dissociation_factor

    # ------------------------
    # 10c) Enhanced physics features for better high-conductivity prediction
    #      These features address systematic underestimation of high-κ formulations
    #      by capturing non-linear salt quality and salt-solvent interactions.
    #      All features are dimensionless ratios or products - no arbitrary constants.
    # ------------------------

    # Lambda_0 squared: captures non-linear salt quality effect
    # High-Λ₀ salts (LiFSI=120) benefit disproportionately vs low-Λ₀ (LiBOB=70)
    Lambda0_sq = avg_Lambda0**2

    # Lambda_0 ratio: relative to LiPF6 baseline from species_db
    # Uses actual LiPF6 Lambda_0 from database, not hardcoded
    lipf6_lambda0 = get_species_property("LiPF6", "Lambda_0")
    if lipf6_lambda0 is None or lipf6_lambda0 == 0:
        raise ValueError(
            "LiPF6 Lambda_0 must be defined in species_db for normalization"
        )
    Lambda0_ratio = avg_Lambda0 / float(lipf6_lambda0)

    # Lambda_0 / viscosity: mobility-enhanced conductivity
    # Better captures how high-Λ₀ salts in low-η solvents achieve high κ
    Lambda0_over_eta = avg_Lambda0 / (viscosity_mix + 1e-12)

    # Lambda_0 × epsilon: dissociation-enhanced conductivity interaction
    # High ε promotes dissociation; high Λ₀ provides intrinsic mobility
    # This interaction term captures synergy (r=+0.513 for high-κ samples)
    Lambda0_x_epsilon = avg_Lambda0 * epsilon_mix

    # Concentration optimality: deviation from 1 M (dimensionless)
    # Let model learn the penalty - we just provide |c - 1|
    conc_deviation_from_1M = abs(Li_total - 1.0)

    # Concentration-corrected viscosity using Jones-Dole expansion
    # η_eff/η₀ = 1 + A√c + Bc (A~0, B~0.3 for typical Li salts)
    # Use dimensionless form: model learns coefficients
    eta_c_linear = viscosity_mix * Li_total  # η × c term
    eta_c_quadratic = viscosity_mix * Li_total**2  # η × c² term

    # Composite conductivity estimate: full physics proxy
    # κ ∝ c × Λ₀ × (ε/ε_ref) / η where we use ε directly (no arbitrary ref)
    # Dimensionless form: (c × Λ₀ × ε) / (η × scale)
    # Scale by max typical values to keep O(1): ε~90, η~2, Λ₀~100
    kappa_composite = (Li_total * avg_Lambda0 * epsilon_mix) / (
        (viscosity_mix + 1e-12)
        * (1.0 + Li_total)  # Include concentration viscosity effect
    )

    # ==========================================================
    # BLOCK 10d: ION-ION CORRELATION FEATURES
    # ==========================================================
    # These features encode the physical effects of ion-ion interactions
    # that make real electrolyte conductivity deviate from the independent-ion
    # Nernst-Einstein model. At real concentrations (0.5-1.5 M), ion-ion
    # correlations dominate: electrophoretic drag, ion atmosphere relaxation,
    # Bjerrum pairing, and activity coefficient suppression.
    #
    # These corrections involve specific functional forms (√c, 1/ε^1.5,
    # √(ε/I)) that require 2+ sequential tree splits to approximate.
    # Pre-computing them gives the tree single-split access.
    #
    # CODATA 2018 fundamental constants (exact, universal — not chemistry):
    _T_K = T_REF_K  # Training data temperature [K]
    _e_C = 1.602176634e-19  # Elementary charge [C]
    _eps0_Fm = 8.8541878128e-12  # Vacuum permittivity [F/m]
    _kB_JK = 1.380649e-23  # Boltzmann constant [J/K]
    _NA = 6.02214076e23  # Avogadro constant [/mol]

    # --- Ionic strength ---
    # I = ½Σcᵢzᵢ² [mol/L]. For monovalent 1:1 salts, I ≈ c.
    # For multivalent anions (BOB²⁻), I differs. Controls Debye-Hückel
    # screening: higher I = shorter screening = stronger correlations.
    ionic_strength = 0.0
    for s in salt_list:
        mol_s = float(per_salt_molarity.get(s, 0.0))
        if mol_s <= 0:
            continue
        z_anion = float(get_species_property(s, "anion_charge", -1.0) or -1.0)
        ionic_strength += 0.5 * mol_s * (1.0 + z_anion**2)

    # --- Effective dielectric after salt suppression ---
    # ε_eff = ε_mix × (1 - δ × c): dissolved ions suppress solvent ε.
    # This is the dielectric environment ions actually experience.
    avg_diel_decrement = 0.0
    weighted_diel_decrement_num = 0.0
    for s in salt_list:
        mol_s = float(per_salt_molarity.get(s, 0.0))
        if mol_s <= 0:
            continue
        dd = float(
            get_species_property(s, "dielectric_decrement_frac_per_M", 0.0) or 0.0
        )
        weighted_diel_decrement_num += mol_s * dd
    if total_salt_molarity > 0:
        avg_diel_decrement = weighted_diel_decrement_num / total_salt_molarity
    epsilon_effective = epsilon_mix * (1.0 - avg_diel_decrement * Li_total)
    epsilon_effective = max(epsilon_effective, 1.0)  # Physical floor: vacuum

    I_safe = max(ionic_strength, 1e-12)
    eps_eff = max(epsilon_effective, 1.0)

    # --- Bjerrum length [nm] ---
    # l_B = e² / (4π ε₀ ε k_B T): distance where Coulomb energy = k_B T.
    # When l_B > sum of ionic radii, ion pairing becomes favorable.
    # At ε=40: l_B ≈ 1.4 nm. At ε=7 (pure DMC): l_B ≈ 8 nm.
    bjerrum_length_m = _e_C**2 / (4.0 * np.pi * _eps0_Fm * eps_eff * _kB_JK * _T_K)
    bjerrum_length_nm = bjerrum_length_m * 1e9

    # --- Debye screening length [nm] ---
    # λ_D = √(ε₀ ε k_B T / (2 e² N_A I × 1000))
    # Screening distance for electrostatic interactions.
    # ×1000 converts mol/L → mol/m³.
    debye_length_m = np.sqrt(
        _eps0_Fm * eps_eff * _kB_JK * _T_K / (2.0 * _e_C**2 * _NA * I_safe * 1000.0)
    )
    debye_length_nm = debye_length_m * 1e9

    # --- Coupling parameter Γ = l_B / λ_D [dimensionless] ---
    # THE key dimensionless number for electrolyte correlations:
    #   Γ << 1: dilute, independent ions, Nernst-Einstein valid
    #   Γ ~ 1: moderate correlations, Onsager corrections needed
    #   Γ >> 1: strong correlations, concentrated-solution theory
    # At 1M LiPF6 in EC:DMC (ε~40): Γ ≈ 2 — firmly correlated.
    coupling_parameter = bjerrum_length_m / (debye_length_m + 1e-30)

    # --- Onsager limiting law corrections (1927) ---
    # First-principles correction from ion-ion correlations:
    # Λ = Λ₀ - (S₁ + S₂·Λ₀)·√c
    #
    # S₁ = electrophoretic coefficient [S·cm²/(mol·√(mol/L))]:
    #   Moving ion drags its counter-ion atmosphere in the opposite
    #   direction, creating retarding force. S₁ = 82.501/(η_P·√(ε·T))
    #   82.501 = Onsager's analytical result: F²√2/(12π√(ε₀RN_A))
    #   in CGS-practical units (Robinson & Stokes, Table 6.1)
    eta_poise = viscosity_mix / 100.0  # cP → Poise: 1 Poise = 100 cP
    onsager_S1 = 82.501 / (eta_poise * np.sqrt(eps_eff * _T_K) + 1e-30)

    # S₂ = relaxation coefficient [dimensionless, multiplied by Λ₀]:
    #   Ion atmosphere distorts asymmetrically under applied field.
    #   S₂ = q/(1+√q) × 8.2487e5/(ε·T)^(3/2)
    #   8.2487e5 = Onsager's analytical result: eF√2/(24πε₀kT·√(ε₀RN_A))
    #   in CGS-practical units (Robinson & Stokes, Table 6.1)
    #   q/(1+√q) = 0.2929 for q=0.5 (symmetric 1:1 electrolyte,
    #   equal transference). This is an exact algebraic result, not a
    #   tuning parameter: q = z₊|z₋|/(z₊+|z₋|) × (λ₊⁰+λ₋⁰)/
    #   ((|z₋|λ₊⁰+z₊λ₋⁰)/(z₊+|z₋|)) = 0.5 when t₊=t₋=0.5.
    onsager_S2 = 0.2929 * 8.2487e5 / (eps_eff * _T_K + 1e-30) ** 1.5

    sqrt_c = np.sqrt(Li_total + 1e-12)

    # Individual correction magnitudes (both reduce Λ):
    electrophoretic_correction = onsager_S1 * sqrt_c
    relaxation_correction = onsager_S2 * avg_Lambda0 * sqrt_c

    # Onsager-corrected conductivity [mS/cm]:
    # At high c (>~1.4M), limiting law over-corrects → λ→0.
    # The tree learns empirical corrections at high c from other features.
    lambda_onsager = max(
        avg_Lambda0 - electrophoretic_correction - relaxation_correction, 0.0
    )
    kappa_onsager = (
        Li_total * lambda_onsager / 1000.0
    )  # S·cm²/mol × mol/L / 1000 → S/cm → ×1000 → mS/cm

    # --- Walden product: Λ₀ × η [S·cm²·cP/mol] ---
    # Walden's rule: Λ·η ≈ const for ideal electrolytes.
    # Deviations = ion-ion correlations.
    walden_product = avg_Lambda0 * viscosity_mix

    # --- Debye-Hückel mean activity coefficient ---
    # Extended DH: log₁₀(γ±) = -A·√I / (1 + B·a·√I)
    # Uses avg_anion_radius as ion-size parameter 'a'.
    # A_DH = 1.8246e6 / (ε·T)^(3/2) [(mol/L)^(-1/2)]
    #   1.8246e6 = analytical from e³√(2N_A·1000)/(8π(ε₀kT)^(3/2)·ln(10))
    #   in CGS-practical units (Robinson & Stokes eq 4.23)
    # B_DH = 50.29e8 / √(ε·T) [cm⁻¹]
    #   50.29e8 = √(8πN_A·1000·e²/(ε₀kT)) in CGS (R&S eq 4.24)
    A_dh = 1.8246e6 / (eps_eff * _T_K) ** 1.5
    B_dh = 50.29e8 / np.sqrt(eps_eff * _T_K)  # [cm⁻¹]
    a_ion_cm = (
        avg_anion_vol ** (1.0 / 3.0) * 1e-8 if avg_anion_vol > 0 else 3.0e-8
    )  # Å³→cm: cube root for radius, ×1e-8
    sqrt_I = np.sqrt(I_safe)
    dh_log_gamma = -A_dh * sqrt_I / (1.0 + B_dh * a_ion_cm * sqrt_I + 1e-30)

    # ==========================================================
    # BLOCK 11: DERIVED AGGREGATE FEATURES
    # ==========================================================
    # Total molarity from ionic additives only (not explicit salts).
    # This allows the model to distinguish between primary salts (explicit)
    # and supplementary ionic additives, which may have different effects
    # on conductivity due to lower concentrations and different anion sizes.
    ionic_additive_molarity_sum = (
        sum(ionic_additive_m.values()) if ionic_additive_m else 0.0
    )

    # ==========================================================
    # BLOCK 12: ASSEMBLE FINAL FEATURE VECTOR
    # ==========================================================
    # The feature vector is assembled in a FIXED ORDER that must match
    # get_feature_names() exactly. This ordering is frozen once the model
    # is trained: the scaler, feature selection indices, and tree split
    # thresholds are all coupled to these exact positions.
    #
    # TOTAL FEATURE COUNT:
    #   4 (core Li) + 7 (mixture) + 4 (structure) + 5 (salt agg)
    #   + 5*N_salts (per-salt) + N_components (fracs)
    #   + 4*N_components (proxies) + 4 (transport) + 2 (Nernst-Einstein)
    #   + 8 (enhanced) = 34 + 5*N_salts + 5*N_components
    # For typical datasets: N_salts=4, N_components=8 -> 34+20+40 = 94 features
    features = []

    # Core Li concentration features (4 features):
    # Polynomial basis (Li, Li^2, Li^3) captures the dome shape nonlinearity.
    # log1p(Li) provides a soft, monotonic representation that is more
    # numerically stable than raw Li at low concentrations.
    features.extend(
        [
            Li_total,  # Linear concentration [mol/L]
            Li_total**2,  # Quadratic: captures dome curvature
            Li_total**3,  # Cubic: captures asymmetry of dome
            np.log1p(Li_total),  # Log transform: stable at low c, saturating at high c
        ]
    )

    # Mixture physics features (7 features):
    # These are the primary liquid-phase properties that govern ion transport.
    features.extend(
        [
            epsilon_mix,  # Dielectric constant [dimensionless] -- ion dissociation
            viscosity_mix,  # Viscosity [cP] -- ion mobility (inverse relationship)
            donor_mix,  # Donor number [kcal/mol] -- Li+ solvation strength
            density_mix_g_ml,  # Density [g/mL] -- volume/mass basis conversion
            dipole_mix,  # Dipole moment [Debye] -- solvation shell geometry
            acceptor_mix,  # Acceptor number -- anion solvation
            mobility,  # epsilon/viscosity -- combined dissociation-mobility proxy
        ]
    )

    # Salt aggregate features (5 features):
    # Molarity-weighted average salt properties for multi-salt formulations
    features.extend(
        [
            total_salt_molarity,  # Total Li+ from all salts [mol/L]
            ionic_additive_molarity_sum,  # Li+ from ionic additives only [mol/L]
            avg_Lambda0,  # Weighted-avg limiting molar conductivity
            avg_binding,  # Weighted-avg ion pair binding energy
            avg_anion_vol,  # Weighted-avg anion volume
        ]
    )

    # Transport coupling features (4 features):
    # Physically motivated interaction terms
    features.extend(
        [
            Li_over_visc,  # c/eta: concentration-limited mobility
            Li_times_mobility,  # c*epsilon/eta: Nernst-Einstein-like
            mobility_times_donor,  # (epsilon/eta)*DN: dissociation-solvation synergy
            viscosity_mix**2,  # eta^2: nonlinear viscosity penalty at high eta
        ]
    )

    # Nernst-Einstein conductivity proxy features (2 features):
    # Physics-based baselines for the model to learn corrections from
    features.extend(
        [
            kappa_nernst_proxy,  # c*Lambda_0/eta: simplified Nernst-Einstein
            kappa_with_dissociation,  # Above * (epsilon/40): with dissociation correction
        ]
    )

    # Enhanced physics features (8 features):
    # Added to correct systematic underestimation of high-kappa formulations
    features.extend(
        [
            Lambda0_sq,  # Nonlinear salt quality (Lambda_0^2)
            Lambda0_ratio,  # Salt quality relative to LiPF6 baseline
            Lambda0_over_eta,  # Salt quality / solvent viscosity
            Lambda0_x_epsilon,  # Salt quality * dissociation
            conc_deviation_from_1M,  # Distance from optimal concentration
            eta_c_linear,  # Jones-Dole linear term (eta*c)
            eta_c_quadratic,  # Jones-Dole quadratic term (eta*c^2)
            kappa_composite,  # Full physics proxy: c*Lambda_0*epsilon/(eta*(1+c))
        ]
    )

    # Salt heterogeneity features (5 features):
    # Generalizable multi-salt physics — not species-specific
    features.extend(
        [
            float(_n_active),  # Number of active salts (standalone, not just in cross-terms)
            Lambda0_spread,  # max(Λ₀) - min(Λ₀): dissociation heterogeneity
            binding_spread,  # max(binding) - min(binding): ion-pair disruption
            anion_vol_spread,  # max(V_anion) - min(V_anion): size mismatch
            salt_entropy,  # -Σ(f_i·ln(f_i)): salt mixing uniformity
        ]
    )

    # Concentration cliff features (3 features):
    # Signed deviation powers for asymmetric dome shape
    features.extend(
        [
            conc_deviation_signed,  # (c - 1.0): signed, positive above peak
            conc_deviation_sq,  # (c - 1.0)²: symmetric quadratic penalty
            conc_deviation_cubed,  # (c - 1.0)³: asymmetric cliff shape
        ]
    )

    # Ion-ion correlation features (10 features):
    # Physical ion-ion interaction corrections from Onsager/Debye-Hückel theory
    features.extend(
        [
            ionic_strength,  # I = ½Σcᵢzᵢ² [mol/L]
            epsilon_effective,  # ε after salt suppression
            bjerrum_length_nm,  # Coulomb distance scale [nm]
            debye_length_nm,  # Screening distance [nm]
            coupling_parameter,  # Γ = l_B/λ_D: correlation strength
            electrophoretic_correction,  # Onsager S₁·√c: counter-ion drag
            relaxation_correction,  # Onsager S₂·Λ₀·√c: atmosphere distortion
            kappa_onsager,  # Onsager-corrected κ [mS/cm]
            walden_product,  # Λ₀·η: ideal-deviation indicator
            dh_log_gamma,  # Debye-Hückel ln(γ±): activity correction
        ]
    )

    # Additive × salt interaction features (4 features):
    # These cross-features let the model distinguish salt-dependent additive effects:
    # FEC helps dual-salt systems (+0.85 mS/cm) but hurts single-salt LiPF6 (dilution).
    # Linear epsilon_mix cannot capture this — needs explicit additive×salt cross-terms.

    # Count active salts (molarity > 0.01 M threshold to exclude trace contamination)
    n_salts_active = sum(
        1
        for s in salt_list
        if s in per_salt_molarity and float(per_salt_molarity[s]) > 0.01
    )

    # Additive dielectric excess above solvent baseline
    epsilon_solvents_only = 0.0
    solvent_total_w = 0.0
    for name, wt in solvent_blend_wt.items():
        eps_s = safe_get("epsilon_r", name)
        if eps_s is not None:
            epsilon_solvents_only += wt * eps_s
            solvent_total_w += wt
    if solvent_total_w > 0:
        epsilon_solvents_only /= solvent_total_w

    additive_epsilon_excess = 0.0
    total_additive_wt = 0.0
    for name, wf in non_ionic_additives.items():
        eps_a = safe_get("epsilon_r", name)
        if eps_a is not None:
            additive_epsilon_excess += wf * max(0.0, eps_a - epsilon_solvents_only)
            total_additive_wt += wf

    # Cross-features: additive dielectric excess × salt system properties
    additive_eps_x_nsalts = additive_epsilon_excess * n_salts_active
    additive_eps_x_mobility = additive_epsilon_excess * mobility
    additive_eps_x_Lambda0 = additive_epsilon_excess * avg_Lambda0
    additive_loading_x_multisalt = total_additive_wt * max(0, n_salts_active - 1)

    features.extend(
        [
            additive_eps_x_nsalts,
            additive_eps_x_mobility,
            additive_eps_x_Lambda0,
            additive_loading_x_multisalt,
        ]
    )

    # ==========================================================
    # BLOCK 14: ADDITIVE × ANION INTERACTION FEATURES
    # ==========================================================
    # High-ε additives (FEC ε=107) disrupt ion pairing preferentially for
    # bulky, polarizable anions (FSI⁻ V=95 ų) vs compact anions (PF₆⁻ V=73 ų).
    # The existing additive cross-features encode additive × aggregate salt quality
    # but not additive × anion identity. These features give trees single-split
    # access to the anion-specific solvation restructuring effect.
    additive_eps_x_anion_vol = additive_epsilon_excess * avg_anion_vol
    additive_eps_x_binding = additive_epsilon_excess * avg_binding
    additive_eps_x_coupling = additive_epsilon_excess * coupling_parameter
    additive_loading_x_anion_vol = total_additive_wt * avg_anion_vol
    additive_eps_x_conc = additive_epsilon_excess * Li_total

    features.extend(
        [
            additive_eps_x_anion_vol,
            additive_eps_x_binding,
            additive_eps_x_coupling,
            additive_loading_x_anion_vol,
            additive_eps_x_conc,
        ]
    )

    # ==========================================================
    # BLOCK 15: SOLVENT × SALT CROSS-TERMS
    # ==========================================================
    # Solvent solvation properties interact with anion identity to determine
    # dissociation, mobility, and correlation behavior. These encode how
    # solvent donor/acceptor strength differentially affects different anions.
    donor_x_anion_vol = donor_mix * avg_anion_vol
    acceptor_x_anion_vol = acceptor_mix * avg_anion_vol
    epsilon_x_anion_vol = epsilon_mix * avg_anion_vol
    epsilon_x_binding = epsilon_mix * avg_binding
    viscosity_x_anion_vol = viscosity_mix * avg_anion_vol

    features.extend(
        [
            donor_x_anion_vol,
            acceptor_x_anion_vol,
            epsilon_x_anion_vol,
            epsilon_x_binding,
            viscosity_x_anion_vol,
        ]
    )

    # ==========================================================
    # BLOCK 16: CONCENTRATION × ANION CROSS-TERMS
    # ==========================================================
    # Ion pairing and correlation effects are concentration-dependent AND
    # anion-dependent. Bulky anions have weaker pairing at all concentrations
    # but the concentration dependence differs from compact anions.
    conc_x_anion_vol = Li_total * avg_anion_vol
    conc_x_binding = Li_total * avg_binding
    conc_sq_x_anion_vol = Li_total ** 2 * avg_anion_vol

    features.extend(
        [
            conc_x_anion_vol,
            conc_x_binding,
            conc_sq_x_anion_vol,
        ]
    )

    # ==========================================================
    # BLOCK 17: BJERRUM AND WALDEN × ANION FEATURES
    # ==========================================================
    # Bjerrum length relative to anion radius: when l_B < r_anion, pairing is weak.
    # 0.1 nm/ų^(1/3): unit conversion from ų^(1/3) to nm (1 Å = 0.1 nm, V^(1/3) ≈ r in Å)
    bjerrum_over_anion = bjerrum_length_nm / (avg_anion_vol ** (1.0 / 3.0) * 0.1 + 1e-12)  # 0.1 = Å-to-nm conversion
    walden_x_anion_vol = walden_product * avg_anion_vol

    features.extend(
        [
            bjerrum_over_anion,
            walden_x_anion_vol,
        ]
    )

    return np.asarray(features, dtype=float)


def get_feature_names(
    component_list: List[str],
    salt_list: List[str],
) -> List[str]:
    """
    Produce human-readable feature names matching the featurize_recipe vector order.

    Why this exists:
        Feature interpretability is critical for debugging the surrogate model,
        computing feature importances, and verifying that the feature vector
        structure has not drifted between training and inference. This function
        produces a list of descriptive string names whose i-th element corresponds
        exactly to the i-th element of the array returned by ``featurize_recipe()``.
        The names are used by the training pipeline for logging feature importances,
        by the feature selection step for reporting which features survive
        correlation filtering, and by the serialized model bundle so that
        downstream consumers can identify features by name.

    What it does:
        Constructs a list of string names in the same 12-block ordering used by
        ``featurize_recipe()``: core Li features, mixture physics, solvent
        structure, salt aggregates, per-salt block (5 names per salt), component
        fractions (1 per component), component additive proxies (4 per component),
        transport couplings, Nernst-Einstein proxies, and enhanced physics
        features. The list length is guaranteed to equal the feature vector
        length for the same ``component_list`` and ``salt_list``.

    Args:
        component_list: Canonical sorted list of liquid component names, as
            returned by ``discover_components()`` and used by ``featurize_recipe()``.
        salt_list: Canonical sorted list of salt names, as returned by
            ``discover_components()`` and used by ``featurize_recipe()``.

    Returns:
        List of strings with the same length as the feature vector produced by
        ``featurize_recipe(component_list, salt_list)``. Each string is a
        descriptive name for the corresponding feature dimension.

    Side effects:
        None. Pure function.
    """
    names = []

    # Core Li features
    names += ["Li_total", "Li_total_sq", "Li_total_cubed", "log1p_Li_total"]

    # Mixture physics
    names += [
        "epsilon_mix",
        "viscosity_mix",
        "donor_mix",
        "density_mix_g_ml",
        "dipole_mix",
        "acceptor_mix",
        "mobility_epsilon_over_viscosity",
    ]

    # Salt aggregates
    names += [
        "total_salt_molarity",
        "ionic_additive_molarity_sum",
        "avg_Lambda0_weighted",
        "avg_ion_pair_binding_kj_mol_weighted",
        "avg_anion_volume_weighted",
    ]

    # Transport coupling names
    names += [
        "Li_over_viscosity",
        "Li_times_mobility",
        "mobility_times_donor",
        "viscosity_sq",
    ]

    # Physics-based conductivity estimates
    names += [
        "kappa_nernst_proxy",
        "kappa_with_dissociation",
    ]

    # Enhanced physics features for high-conductivity prediction
    names += [
        "Lambda0_sq",
        "Lambda0_ratio_vs_LiPF6",
        "Lambda0_over_viscosity",
        "Lambda0_x_epsilon",
        "conc_deviation_from_1M",
        "viscosity_x_conc_linear",
        "viscosity_x_conc_quadratic",
        "kappa_composite",
    ]

    # Salt heterogeneity features (5 features)
    names += [
        "n_salts_active",
        "Lambda0_spread",
        "binding_energy_spread",
        "anion_vol_spread",
        "salt_molarity_entropy",
    ]

    # Concentration cliff features (3 features)
    names += [
        "conc_deviation_signed",
        "conc_deviation_sq",
        "conc_deviation_cubed",
    ]

    # Ion-ion correlation features
    names += [
        "ionic_strength",
        "epsilon_effective",
        "bjerrum_length_nm",
        "debye_length_nm",
        "coupling_parameter",
        "onsager_electrophoretic",
        "onsager_relaxation",
        "kappa_onsager",
        "walden_product",
        "dh_log_gamma",
    ]

    # Additive × salt interaction features (4 features)
    names += [
        "additive_eps_x_nsalts",
        "additive_eps_x_mobility",
        "additive_eps_x_Lambda0",
        "additive_loading_x_multisalt",
    ]

    # Additive × anion interaction features (5 features)
    names += [
        "additive_eps_x_anion_vol",
        "additive_eps_x_binding",
        "additive_eps_x_coupling",
        "additive_loading_x_anion_vol",
        "additive_eps_x_conc",
    ]

    # Solvent × salt cross-terms (5 features)
    names += [
        "donor_x_anion_vol",
        "acceptor_x_anion_vol",
        "epsilon_x_anion_vol",
        "epsilon_x_binding",
        "viscosity_x_anion_vol",
    ]

    # Concentration × anion cross-terms (3 features)
    names += [
        "conc_x_anion_vol",
        "conc_x_binding",
        "conc_sq_x_anion_vol",
    ]

    # Bjerrum and Walden × anion features (2 features)
    names += [
        "bjerrum_over_anion",
        "walden_x_anion_vol",
    ]

    return names
