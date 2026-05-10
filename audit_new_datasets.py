"""Audit 5 unused conductivity datasets for data quality before MolSet integration.

Datasets audited:
  1. data/electrolyte_electrolytomics_db.py  — 1849 entries (dict format, DATA export)
  2. data/lehnert2025_db.py                  — 150  entries (NamedTuple, load_lehnert2025())
  3. data/logan2018_db.py                    — 160  entries (NamedTuple, load_logan2018())
  4. data/nyman2008_db.py                    — 37   entries (NamedTuple, load_nyman2008())
  5. data/valoen2005_db.py                   — 315  entries (NamedTuple, load_valoen2005())

Checks performed:
  - Species coverage against data.species_data (SOLVENTS, SALTS, ADDITIVES)
  - Temperature, concentration, conductivity range validation
  - Per-recipe measurement consistency (within 5K window, std/mean > 30% flagged)
  - Single-solvent dominance (>90% of solvent fraction)
  - Cross-dataset overlap (new vs existing training data, and between new datasets)
  - Composition sanity (fraction sums, missing salts/solvents)

Entry point: python -m conductivity.audit_new_datasets
"""

import sys
import logging
import math
from collections import defaultdict
from typing import Any

sys.path.insert(0, "/Users/jreddy/electrolyte_formation_sim")

from constants import T_REF_K
from data.species_data import SOLVENTS, SALTS, ADDITIVES
from data.electrolyte_property_db import DATA as DATA_PROPERTY
from data.electrolyte_calisol_db import DATA as DATA_CALISOL
from data.electrolyte_electrolytomics_db import DATA as DATA_ELECTROLYTOMICS
from data.lehnert2025_db import load_lehnert2025
from data.logan2018_db import load_logan2018
from data.nyman2008_db import load_nyman2008
from data.valoen2005_db import load_valoen2005

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("audit_new_datasets")

# All known species names from species_data.py
KNOWN_SOLVENTS = set(SOLVENTS.keys())
KNOWN_SALTS = set(SALTS.keys())
KNOWN_ADDITIVES = set(ADDITIVES.keys())
KNOWN_ALL = KNOWN_SOLVENTS | KNOWN_SALTS | KNOWN_ADDITIVES

# =============================================================================
# AUDIT THRESHOLDS — explicit constants for visibility and auditing.
# These define the boundaries for flagging suspicious data, not physics constraints.
# =============================================================================

# Temperature: Li-ion electrolyte measurements below -20C or above 80C are rare
# and likely outside the MolSet training regime.
TEMP_LOW_K = 253.0   # Explicit audit threshold: -20C, below this is unusual for Li-ion electrolyte data
TEMP_HIGH_K = 353.0  # Explicit audit threshold: +80C, above this is unusual for Li-ion electrolyte data

# Molarity: typical Li-salt electrolytes are 0.5-1.5 M; below 0.1 M is extremely
# dilute (near-zero conductivity), above 3.0 M is highly concentrated (viscosity-dominated).
MOLARITY_LOW_M = 0.1   # Explicit audit threshold: below this is extremely dilute, near measurement floor
MOLARITY_HIGH_M = 3.0  # Explicit audit threshold: above this is highly concentrated regime

# Conductivity: Li-ion electrolytes at room temp are typically 5-15 mS/cm.
# Below 0.1 is near-zero (frozen, extreme dilution, or measurement error).
# Above 25 exceeds any known Li-salt organic electrolyte at any temperature.
SIGMA_LOW_MS_CM = 0.1   # Explicit audit threshold: below this is near measurement floor
SIGMA_HIGH_MS_CM = 25.0  # Explicit audit threshold: no known Li-salt organic electrolyte exceeds this

# Temperature binning for measurement consistency: measurements within 5K of each
# other should give consistent conductivity values for the same composition.
TEMP_BIN_WIDTH_K = 5.0  # Explicit audit threshold: bin width for grouping measurements at similar T

# CV threshold for flagging inconsistent measurements within a temperature bin.
# 30% coefficient of variation is generous — typical EIS reproducibility is <5%.
CV_INCONSISTENCY_THRESHOLD = 0.30  # Explicit audit threshold: std/mean above this flags inconsistency

# Single-solvent dominance: if one solvent is >90% of the mixture, it is
# effectively a pure-solvent measurement, which may not represent blended behavior.
SINGLE_SOLVENT_FRAC_THRESHOLD = 0.90  # Explicit audit threshold: above this is effectively pure solvent

# Solvent fraction sum tolerance: solvent weight/volume fractions should sum to
# ~1.0. More than 10% deviation indicates data entry error or unit confusion.
FRAC_SUM_TOLERANCE = 0.10  # Explicit audit threshold: |sum(solvent_fracs) - 1.0| above this is suspicious


# =============================================================================
# UNIFIED ENTRY EXTRACTION
# =============================================================================

def _normalize_entry(raw: Any, dataset_name: str, index: int) -> dict | None:
    """Convert any entry format to a uniform dict.

    Returns dict with keys:
        recipe: {solvents: {}, salts: {}, additives: {}}
        conductivity_mS_cm: float | None
        temperature_K: float | None
        dataset: str
        index: int
    """
    # Dict-format (electrolytomics, property_db, calisol_db)
    if isinstance(raw, dict):
        recipe = raw["recipe"]
        props = raw["properties"]
        sigma = props["conductivity_mS_cm"] if "conductivity_mS_cm" in props else None
        # Temperature: top-level for electrolytomics/calisol, absent for property_db
        if "temperature_K" in raw:
            temp = raw["temperature_K"]
        elif "T_K" in props:
            temp = props["T_K"]
        else:
            temp = None
    # NamedTuple-format (lehnert, logan, nyman, valoen)
    elif hasattr(raw, "recipe") and hasattr(raw, "properties"):
        recipe = raw.recipe
        props = raw.properties
        sigma = props["conductivity_mS_cm"] if "conductivity_mS_cm" in props else None
        temp = props["T_K"] if "T_K" in props else None
    else:
        logger.warning(
            "  [%s] Entry %d: unrecognized format type=%s, skipping",
            dataset_name, index, type(raw).__name__,
        )
        return None

    return {
        "recipe": recipe,
        "conductivity_mS_cm": sigma,
        "temperature_K": temp,
        "dataset": dataset_name,
        "index": index,
    }


def _recipe_key(recipe: dict) -> tuple:
    """Canonical hashable key for a recipe (salt, solvent, additive composition).

    Requires recipe to have "salts", "solvents", "additives" keys (guaranteed by
    all dataset formats: property_db, calisol, electrolytomics, and NamedTuple loaders).
    """
    salts = tuple(sorted(recipe["salts"].items()))
    solvents = tuple(sorted(recipe["solvents"].items()))
    additives = tuple(sorted(recipe["additives"].items()))
    return (salts, solvents, additives)


def _all_species(recipe: dict) -> set[str]:
    """All species names in a recipe.

    Requires recipe to have "salts", "solvents", "additives" keys.
    """
    species: set[str] = set()
    species.update(recipe["salts"].keys())
    species.update(recipe["solvents"].keys())
    species.update(recipe["additives"].keys())
    return species


# =============================================================================
# PER-DATASET AUDIT
# =============================================================================

class DatasetAudit:
    """Audit results for one dataset."""

    def __init__(self, name: str, entries: list[dict]):
        self.name = name
        self.entries = entries  # list of normalized dicts
        self.flags: list[dict] = []  # (index, reason, detail)
        self.rejected: list[dict] = []  # (index, reason, detail)
        self.usable_indices: set[int] = set()

    def add_flag(self, index: int, reason: str, detail: str = "") -> None:
        self.flags.append({"index": index, "reason": reason, "detail": detail})

    def add_rejection(self, index: int, reason: str, detail: str = "") -> None:
        self.rejected.append({"index": index, "reason": reason, "detail": detail})


def _check_species_coverage(audit: DatasetAudit) -> None:
    """Check which species in the dataset are known in species_data.py."""
    all_species_in_dataset: set[str] = set()
    unknown_species: dict[str, int] = defaultdict(int)
    species_counts: dict[str, int] = defaultdict(int)

    for e in audit.entries:
        recipe = e["recipe"]
        sp_set = _all_species(recipe)
        all_species_in_dataset.update(sp_set)
        for sp in sp_set:
            species_counts[sp] += 1
            if sp not in KNOWN_ALL:
                unknown_species[sp] += 1

    logger.info("  Species found: %s", sorted(all_species_in_dataset))
    logger.info("  Species counts: %s", dict(sorted(species_counts.items(), key=lambda x: -x[1])))

    if unknown_species:
        logger.warning("  UNKNOWN species (not in species_data.py):")
        for sp, count in sorted(unknown_species.items(), key=lambda x: -x[1]):
            logger.warning("    %s: %d entries (NO property vector available)", sp, count)
        # Reject entries with unknown species
        for e in audit.entries:
            sp_set = _all_species(e["recipe"])
            unknowns = sp_set - KNOWN_ALL
            if unknowns:
                audit.add_rejection(
                    e["index"],
                    "unknown_species",
                    f"species not in registry: {sorted(unknowns)}",
                )
    else:
        logger.info("  All species are in the registry.")


def _check_temperature_range(audit: DatasetAudit) -> None:
    """Flag entries with temperature outside TEMP_LOW_K..TEMP_HIGH_K."""
    temps = [e["temperature_K"] for e in audit.entries if e["temperature_K"] is not None]
    n_no_temp = sum(1 for e in audit.entries if e["temperature_K"] is None)
    if temps:
        logger.info("  Temperature range: [%.2f, %.2f] K", min(temps), max(temps))
        logger.info("  Entries without temperature: %d", n_no_temp)
        n_cold = sum(1 for t in temps if t < TEMP_LOW_K)
        n_hot = sum(1 for t in temps if t > TEMP_HIGH_K)
        if n_cold > 0:
            logger.warning("  Temperature < %.0f K (unusual): %d entries", TEMP_LOW_K, n_cold)
        if n_hot > 0:
            logger.warning("  Temperature > %.0f K (unusual): %d entries", TEMP_HIGH_K, n_hot)
        for e in audit.entries:
            t = e["temperature_K"]
            if t is not None:
                if t < TEMP_LOW_K:
                    audit.add_flag(e["index"], "low_temperature", f"T={t:.2f} K < {TEMP_LOW_K} K")
                elif t > TEMP_HIGH_K:
                    audit.add_flag(e["index"], "high_temperature", f"T={t:.2f} K > {TEMP_HIGH_K} K")
    else:
        logger.info("  No temperature data available in this dataset.")


def _check_molarity_range(audit: DatasetAudit) -> None:
    """Flag entries with salt molarity outside MOLARITY_LOW_M..MOLARITY_HIGH_M."""
    molarities: list[tuple[int, float]] = []
    for e in audit.entries:
        recipe = e["recipe"]
        for salt_name, molarity in recipe["salts"].items():
            molarities.append((e["index"], molarity))

    if molarities:
        all_m = [m for _, m in molarities]
        logger.info("  Salt molarity range: [%.4f, %.4f] M", min(all_m), max(all_m))
        n_low_m = sum(1 for m in all_m if m < MOLARITY_LOW_M)
        n_high_m = sum(1 for m in all_m if m > MOLARITY_HIGH_M)
        if n_low_m > 0:
            logger.warning("  Molarity < %.1f M (unusual): %d entries", MOLARITY_LOW_M, n_low_m)
        if n_high_m > 0:
            logger.warning("  Molarity > %.1f M (unusual): %d entries", MOLARITY_HIGH_M, n_high_m)
        for idx, m in molarities:
            if m < MOLARITY_LOW_M:
                audit.add_flag(idx, "low_molarity", f"salt molarity={m:.4f} M < {MOLARITY_LOW_M} M")
            elif m > MOLARITY_HIGH_M:
                audit.add_flag(idx, "high_molarity", f"salt molarity={m:.4f} M > {MOLARITY_HIGH_M} M")


def _check_conductivity_range(audit: DatasetAudit) -> None:
    """Flag entries with conductivity outside 0.1-25.0 mS/cm."""
    sigmas = [
        (e["index"], e["conductivity_mS_cm"])
        for e in audit.entries
        if e["conductivity_mS_cm"] is not None
    ]
    if sigmas:
        all_s = [s for _, s in sigmas]
        logger.info("  Conductivity range: [%.4f, %.4f] mS/cm", min(all_s), max(all_s))
        n_low_s = sum(1 for s in all_s if s < 0.1)
        n_high_s = sum(1 for s in all_s if s > 25.0)
        if n_low_s > 0:
            logger.warning("  Conductivity < 0.1 mS/cm (suspicious): %d entries", n_low_s)
        if n_high_s > 0:
            logger.warning("  Conductivity > 25.0 mS/cm (suspicious): %d entries", n_high_s)
        for idx, s in sigmas:
            if s < 0.1:
                audit.add_flag(idx, "low_conductivity", f"sigma={s:.4f} mS/cm < 0.1")
            elif s > 25.0:
                audit.add_flag(idx, "high_conductivity", f"sigma={s:.4f} mS/cm > 25.0")


def _check_measurement_consistency(audit: DatasetAudit) -> None:
    """For each unique composition, if multiple measurements at similar T (within 5K),
    flag if conductivity CV > 30%."""
    logger.info("  --- Per-recipe consistency (within 5K window) ---")
    recipe_groups: dict[tuple, list] = defaultdict(list)
    for e in audit.entries:
        if e["conductivity_mS_cm"] is None:
            continue
        key = _recipe_key(e["recipe"])
        recipe_groups[key].append(e)

    n_inconsistent = 0
    for key, group in recipe_groups.items():
        # Group by temperature within 5K bins
        temp_bins: dict[float, list] = defaultdict(list)
        for e in group:
            t = e["temperature_K"] if e["temperature_K"] is not None else 298.15
            # Bin to nearest 5K
            t_bin = round(t / 5.0) * 5.0
            temp_bins[t_bin].append(e)

        for t_bin, bin_entries in temp_bins.items():
            if len(bin_entries) < 2:
                continue
            sigma_vals = [e["conductivity_mS_cm"] for e in bin_entries]
            mean_s = sum(sigma_vals) / len(sigma_vals)
            if mean_s < 1e-10:
                continue
            std_s = (sum((s - mean_s) ** 2 for s in sigma_vals) / len(sigma_vals)) ** 0.5
            cv = std_s / mean_s
            if cv > 0.30:
                n_inconsistent += 1
                indices = [e["index"] for e in bin_entries]
                detail = (
                    f"T_bin={t_bin:.0f}K, n={len(bin_entries)}, "
                    f"sigma={sigma_vals}, mean={mean_s:.3f}, std={std_s:.3f}, "
                    f"CV={cv:.2%}"
                )
                logger.warning("    Inconsistent recipe: indices=%s %s", indices, detail)
                for e in bin_entries:
                    audit.add_flag(e["index"], "inconsistent_measurement", detail)

    logger.info("  Inconsistent recipe-temperature groups (CV > 30%%): %d", n_inconsistent)


def _check_single_solvent_dominance(audit: DatasetAudit) -> None:
    """Flag recipes where a single solvent is >90% of the solvent mixture."""
    logger.info("  --- Single-solvent dominance (>90%% of solvent mix) ---")
    n_single_dominant = 0
    for e in audit.entries:
        solvents = e["recipe"].get("solvents", {})
        if not solvents:
            continue
        total_solvent = sum(solvents.values())
        if total_solvent < 1e-10:
            continue
        for solv_name, frac in solvents.items():
            if frac / total_solvent > 0.90:
                n_single_dominant += 1
                detail = (
                    f"{solv_name} = {frac:.4f} / {total_solvent:.4f} "
                    f"= {frac / total_solvent:.1%} of solvent mix"
                )
                audit.add_flag(
                    e["index"],
                    "single_solvent_dominant",
                    detail,
                )
                logger.info("    Entry %d: %s", e["index"], detail)
                break  # one flag per entry
    logger.info("  Entries with single-solvent >90%%: %d / %d", n_single_dominant, len(audit.entries))


def _check_composition_sanity(audit: DatasetAudit) -> None:
    """Flag recipes with bad fraction sums, salts-but-no-solvents, solvents-but-no-salts."""
    logger.info("  --- Composition sanity ---")
    n_bad_frac_sum = 0
    n_salt_no_solvent = 0
    n_solvent_no_salt = 0

    for e in audit.entries:
        recipe = e["recipe"]
        solvents = recipe.get("solvents", {})
        salts = recipe.get("salts", {})

        # Check solvent fractions sum to ~1.0 (within 10%)
        solvent_sum = sum(solvents.values()) if solvents else 0.0
        if solvents and abs(solvent_sum - 1.0) > 0.10:
            n_bad_frac_sum += 1
            detail = f"solvent fractions sum={solvent_sum:.4f} (expected ~1.0)"
            audit.add_flag(e["index"], "bad_solvent_frac_sum", detail)
            logger.warning("    Entry %d: %s, solvents=%s", e["index"], detail, solvents)

        # Salts but no solvents
        if salts and not solvents:
            n_salt_no_solvent += 1
            audit.add_flag(e["index"], "salt_no_solvent", f"salts={salts}, no solvents")

        # Solvents but no salts
        if solvents and not salts:
            n_solvent_no_salt += 1
            audit.add_flag(e["index"], "solvent_no_salt", f"solvents={solvents}, no salts")

    logger.info("  Bad solvent fraction sums (|sum - 1.0| > 10%%): %d", n_bad_frac_sum)
    logger.info("  Salts but no solvents: %d", n_salt_no_solvent)
    logger.info("  Solvents but no salts: %d", n_solvent_no_salt)


def _determine_usable_entries(audit: DatasetAudit) -> None:
    """Mark entries as usable: has conductivity + all species known + finite positive sigma."""
    rejected_indices = set(r["index"] for r in audit.rejected)
    for e in audit.entries:
        idx = e["index"]
        if idx in rejected_indices:
            continue
        # Must have conductivity
        sigma = e["conductivity_mS_cm"]
        if sigma is None:
            continue
        # Must have finite conductivity
        if not math.isfinite(sigma) or sigma <= 0:
            audit.add_rejection(idx, "invalid_conductivity", f"sigma={sigma}")
            continue
        # All species must be known (entries with unknowns already rejected)
        sp_set = _all_species(e["recipe"])
        unknowns = sp_set - KNOWN_ALL
        if unknowns:
            continue  # already rejected above
        audit.usable_indices.add(idx)


def audit_dataset(name: str, raw_entries: list) -> DatasetAudit:
    """Run all quality checks on a single dataset."""
    logger.info("=" * 80)
    logger.info("AUDITING: %s (%d raw entries)", name, len(raw_entries))
    logger.info("=" * 80)

    # --- Normalize entries ---
    entries = []
    for i, raw in enumerate(raw_entries):
        norm = _normalize_entry(raw, name, i)
        if norm is not None:
            entries.append(norm)
        else:
            logger.warning("  Entry %d could not be normalized, skipping", i)

    audit = DatasetAudit(name, entries)

    # --- Basic counts ---
    n_total = len(entries)
    n_with_sigma = sum(1 for e in entries if e["conductivity_mS_cm"] is not None)
    n_without_sigma = n_total - n_with_sigma
    logger.info("  Total normalized entries: %d", n_total)
    logger.info("  With conductivity_mS_cm: %d", n_with_sigma)
    logger.info("  Without conductivity_mS_cm: %d", n_without_sigma)

    for e in entries:
        if e["conductivity_mS_cm"] is None:
            audit.add_rejection(e["index"], "no_conductivity", "conductivity_mS_cm is missing")

    # --- Run all checks ---
    _check_species_coverage(audit)
    _check_temperature_range(audit)
    _check_molarity_range(audit)
    _check_conductivity_range(audit)
    _check_measurement_consistency(audit)
    _check_single_solvent_dominance(audit)
    _check_composition_sanity(audit)
    _determine_usable_entries(audit)

    logger.info("  USABLE entries: %d / %d", len(audit.usable_indices), n_total)
    logger.info("  REJECTED entries: %d", len(set(r["index"] for r in audit.rejected)))
    logger.info("  FLAGGED entries (warnings, still usable): %d",
                len(set(f["index"] for f in audit.flags)))

    return audit


# =============================================================================
# CROSS-DATASET OVERLAP
# =============================================================================

def check_cross_overlap(
    new_audits: list[DatasetAudit],
    existing_entries: list[dict],
) -> dict:
    """Check recipe overlap between new datasets and existing training data,
    and between new datasets themselves.

    Returns dict with overlap counts and details.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("CROSS-DATASET OVERLAP ANALYSIS")
    logger.info("=" * 80)

    # Build existing recipe key set
    existing_keys: set[tuple] = set()
    for raw in existing_entries:
        norm = _normalize_entry(raw, "existing", 0)
        if norm is not None:
            existing_keys.add(_recipe_key(norm["recipe"]))
    logger.info("Existing training data: %d unique recipe keys", len(existing_keys))

    # Build per-dataset recipe key sets (usable entries only)
    dataset_keys: dict[str, set[tuple]] = {}

    for audit in new_audits:
        keys = set()
        for e in audit.entries:
            if e["index"] in audit.usable_indices:
                k = _recipe_key(e["recipe"])
                keys.add(k)
        dataset_keys[audit.name] = keys
        logger.info("  %s: %d usable unique recipe keys", audit.name, len(keys))

    # Overlap with existing
    logger.info("")
    logger.info("--- Overlap with existing training data (property_db + calisol_db) ---")
    overlap_with_existing: dict[str, set[tuple]] = {}
    for audit in new_audits:
        overlap = dataset_keys[audit.name] & existing_keys
        overlap_with_existing[audit.name] = overlap
        logger.info(
            "  %s: %d / %d usable recipes already in training data",
            audit.name, len(overlap), len(dataset_keys[audit.name]),
        )
        if overlap and len(overlap) <= 10:
            for k in sorted(overlap, key=str):
                logger.info("    overlap recipe key: %s", k)

    # Pairwise overlap between new datasets
    logger.info("")
    logger.info("--- Pairwise overlap between new datasets ---")
    names = [a.name for a in new_audits]
    pairwise_overlaps: dict[tuple[str, str], set[tuple]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = dataset_keys[names[i]] & dataset_keys[names[j]]
            pairwise_overlaps[(names[i], names[j])] = overlap
            logger.info(
                "  %s vs %s: %d shared recipes",
                names[i], names[j], len(overlap),
            )
            if overlap and len(overlap) <= 5:
                for k in sorted(overlap, key=str):
                    logger.info("    shared key: %s", k)

    # Combined new unique (not in existing, not duplicated)
    all_new_keys: set[tuple] = set()
    for keys in dataset_keys.values():
        all_new_keys.update(keys)
    truly_new = all_new_keys - existing_keys
    logger.info("")
    logger.info(
        "TOTAL new unique recipes across all 5 datasets: %d (of %d total unique)",
        len(truly_new), len(all_new_keys),
    )
    logger.info(
        "Recipes overlapping with existing training data: %d",
        len(all_new_keys) - len(truly_new),
    )

    return {
        "existing_key_count": len(existing_keys),
        "new_unique_total": len(all_new_keys),
        "truly_new": len(truly_new),
        "overlap_with_existing": {
            name: len(ov) for name, ov in overlap_with_existing.items()
        },
        "pairwise_overlaps": {
            f"{a}_vs_{b}": len(ov) for (a, b), ov in pairwise_overlaps.items()
        },
    }


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_summary(audits: list[DatasetAudit], overlap_info: dict) -> None:
    """Print a clear final summary."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL AUDIT SUMMARY")
    logger.info("=" * 80)

    total_entries = 0
    total_usable = 0
    total_flagged_entries = 0
    total_rejected = 0

    for audit in audits:
        n_total = len(audit.entries)
        n_usable = len(audit.usable_indices)
        # Count unique flagged entry indices (an entry can have multiple flags)
        flagged_indices = set(f["index"] for f in audit.flags)
        n_flagged = len(flagged_indices)
        n_rejected = len(set(r["index"] for r in audit.rejected))

        total_entries += n_total
        total_usable += n_usable
        total_flagged_entries += n_flagged
        total_rejected += n_rejected

        logger.info(
            "  %-30s  total=%4d  usable=%4d  flagged=%4d  rejected=%4d",
            audit.name, n_total, n_usable, n_flagged, n_rejected,
        )

    logger.info("  " + "-" * 75)
    logger.info(
        "  %-30s  total=%4d  usable=%4d  flagged=%4d  rejected=%4d",
        "COMBINED", total_entries, total_usable, total_flagged_entries, total_rejected,
    )

    logger.info("")
    logger.info("  Existing training data unique recipes: %d", overlap_info["existing_key_count"])
    logger.info("  New unique recipes (all 5 datasets): %d", overlap_info["new_unique_total"])
    logger.info("  Truly NEW recipes (not in existing): %d", overlap_info["truly_new"])
    logger.info(
        "  Overlap with existing: %d",
        overlap_info["new_unique_total"] - overlap_info["truly_new"],
    )

    # Rejection reasons breakdown
    logger.info("")
    logger.info("--- Rejection reasons (all datasets) ---")
    reason_counts: dict[str, int] = defaultdict(int)
    for audit in audits:
        for r in audit.rejected:
            reason_counts[r["reason"]] += 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-30s  %d entries", reason, count)

    # Flag reasons breakdown
    logger.info("")
    logger.info("--- Flag reasons (warnings, all datasets) ---")
    flag_counts: dict[str, int] = defaultdict(int)
    for audit in audits:
        for f in audit.flags:
            flag_counts[f["reason"]] += 1
    for reason, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-30s  %d flags", reason, count)

    # List all rejected entries with reasons
    logger.info("")
    logger.info("--- All rejected entries ---")
    for audit in audits:
        if not audit.rejected:
            continue
        # Deduplicate by index (an entry may have multiple rejection reasons)
        by_index: dict[int, list[str]] = defaultdict(list)
        for r in audit.rejected:
            by_index[r["index"]].append(f"{r['reason']}: {r['detail']}")
        logger.info("  [%s] %d rejected entries:", audit.name, len(by_index))
        for idx, reasons in sorted(by_index.items()):
            logger.info("    idx=%d: %s", idx, "; ".join(reasons))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("Audit of 5 unused conductivity datasets for MolSet integration")
    logger.info("Known species: %d solvents, %d salts, %d additives",
                len(KNOWN_SOLVENTS), len(KNOWN_SALTS), len(KNOWN_ADDITIVES))
    logger.info("Known solvents: %s", sorted(KNOWN_SOLVENTS))
    logger.info("Known salts: %s", sorted(KNOWN_SALTS))
    logger.info("Known additives: %s", sorted(KNOWN_ADDITIVES))
    logger.info("")

    # Load all 5 new datasets
    logger.info("Loading datasets...")
    raw_electrolytomics = DATA_ELECTROLYTOMICS
    raw_lehnert = load_lehnert2025()
    raw_logan = load_logan2018()
    raw_nyman = load_nyman2008()
    raw_valoen = load_valoen2005()
    logger.info(
        "  electrolytomics=%d, lehnert=%d, logan=%d, nyman=%d, valoen=%d",
        len(raw_electrolytomics), len(raw_lehnert), len(raw_logan),
        len(raw_nyman), len(raw_valoen),
    )
    logger.info("")

    # Audit each dataset
    audit_electrolytomics = audit_dataset("electrolytomics", raw_electrolytomics)
    audit_lehnert = audit_dataset("lehnert2025", raw_lehnert)
    audit_logan = audit_dataset("logan2018", raw_logan)
    audit_nyman = audit_dataset("nyman2008", raw_nyman)
    audit_valoen = audit_dataset("valoen2005", raw_valoen)

    all_audits = [
        audit_electrolytomics,
        audit_lehnert,
        audit_logan,
        audit_nyman,
        audit_valoen,
    ]

    # Cross-dataset overlap
    existing_entries = list(DATA_PROPERTY) + list(DATA_CALISOL)
    overlap_info = check_cross_overlap(all_audits, existing_entries)

    # Final summary
    print_summary(all_audits, overlap_info)

    logger.info("")
    logger.info("Audit complete.")


if __name__ == "__main__":
    main()
