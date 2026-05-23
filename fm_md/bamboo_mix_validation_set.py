"""Build the BAMBOO-Mix validation set for predict_sigma generalization tests.

BAMBOO-Mix (Bytedance, conductivity/fm_data/bamboo_mix/dataset/data.json) is a
regression dataset of 115,064 (composition, T, sigma) tuples. 10,407 of those
rows have `conductivity_mask=True` and carry real MD-computed sigma values;
the remaining 104,657 rows have a constant 4.4136 mS/cm placeholder (memory
[[loss3-f1-f2-f3-fixes]] and plan §1h).

This module extracts the mask=True subset whose species are ALL in our
SPECIES_CATALOGUE (Li+, FSI-, EC, EMC, DMC, DEC, PC, FEC, VC, DME, plus the
salts LiPF6 / LiFSI which decompose into Li+ + their anion). The result is a
list of dicts of (species_list, mole_fractions, T, sigma_md). This is the
eval harness for `predict_sigma` -- we run the propagator on each row and
compare to the BAMBOO sigma_MD; both use classical-FF MD, with the in-
distribution subset (FSI-similar compositions) expected to agree more
tightly than the truly-novel chemistries.

Entry: python -m conductivity.fm_md.bamboo_mix_validation_set
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from constants import CELSIUS_TO_KELVIN

logger = logging.getLogger(__name__)


BAMBOO_PATH = Path("conductivity/fm_data/bamboo_mix/dataset/data.json")
OUTPUT_PATH = Path("conductivity/fm_data/bamboo_mix_validation_set.npz")


# Mapping from BAMBOO-Mix salt names to (cation_name, anion_name) tuples in
# SPECIES_CATALOGUE. Only salts we can decompose into species we recognize are
# listed; rows with other salts (LiBOB, LiTFSI, LiBF4, ...) are excluded.
SALT_TO_IONS: dict[str, tuple[str, str]] = {
    "LiPF6":     ("Li+", "PF6-"),
    "LiFSI":     ("Li+", "FSI-"),
    "LiFSA":     ("Li+", "FSI-"),     # BAMBOO sometimes uses LiFSA as an alias for LiFSI
    "LiTFSI":    ("Li+", "TFSI-"),
    "LiClO4":    ("Li+", "ClO4-"),
    "LiCF3SO3":  ("Li+", "CF3SO3-"),
}

# Solvents we have in SPECIES_CATALOGUE + SPECIES_SMILES.
SUPPORTED_SOLVENTS = {"EC", "EMC", "DMC", "DEC", "PC", "FEC", "VC", "DME"}

# Required BAMBOO-Mix keys per row. Missing any of these means the row is
# malformed; we drop it rather than silently substituting defaults.
REQUIRED_ROW_KEYS = {"conductivity_mask", "salts", "solvents", "temperature", "conductivity"}
REQUIRED_SALT_KEYS = {"name", "molar_ratio"}
REQUIRED_SOLVENT_KEYS = {"name", "molar_ratio"}


def _row_has_required_keys(row: dict) -> bool:
    if not REQUIRED_ROW_KEYS.issubset(row.keys()):
        return False
    for s in row["salts"]:
        if not REQUIRED_SALT_KEYS.issubset(s.keys()):
            return False
    for sv in row["solvents"]:
        if not REQUIRED_SOLVENT_KEYS.issubset(sv.keys()):
            return False
    return True


def extract_validation_rows(data: list[dict]) -> list[dict]:
    """Filter mask=True rows whose species are all supported. Rows missing
    required keys are counted in skipped_malformed and dropped; we do not
    substitute defaults for missing wiring."""
    kept: list[dict] = []
    skipped_mask = 0
    skipped_salt = 0
    skipped_solvent = 0
    skipped_no_solvent = 0
    skipped_malformed = 0
    for d in data:
        if not _row_has_required_keys(d):
            skipped_malformed += 1
            continue
        if d["conductivity_mask"] is not True:
            skipped_mask += 1
            continue
        salts = d["salts"]
        if len(salts) != 1:
            skipped_salt += 1
            continue
        salt_name = salts[0]["name"]
        if salt_name not in SALT_TO_IONS:
            skipped_salt += 1
            continue
        solvents = d["solvents"]
        if not solvents:
            skipped_no_solvent += 1
            continue
        if not all(sv["name"] in SUPPORTED_SOLVENTS for sv in solvents):
            skipped_solvent += 1
            continue
        cation_name, anion_name = SALT_TO_IONS[salt_name]
        salt_n = float(salts[0]["molar_ratio"])
        species_list: list[str] = [cation_name, anion_name]
        molar_ratios: list[float] = [salt_n, salt_n]
        for sv in solvents:
            species_list.append(sv["name"])
            molar_ratios.append(float(sv["molar_ratio"]))
        total = sum(molar_ratios)
        if total <= 0:
            skipped_malformed += 1
            continue
        mole_fractions = [m / total for m in molar_ratios]
        kept.append({
            "species_list": species_list,
            "mole_fractions": mole_fractions,
            "T": float(d["temperature"]) + CELSIUS_TO_KELVIN,   # BAMBOO uses Celsius
            "sigma_md": float(d["conductivity"]),
            "salt": salt_name,
        })
    logger.info("kept %d rows; skipped: %d mask=False, %d unsupported salt, %d unsupported solvent, %d no-solvent, %d malformed",
                len(kept), skipped_mask, skipped_salt, skipped_solvent, skipped_no_solvent, skipped_malformed)
    return kept


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger.info("loading BAMBOO-Mix dataset: %s", BAMBOO_PATH)
    with open(BAMBOO_PATH) as f:
        data = json.load(f)
    logger.info("total rows: %d", len(data))

    rows = extract_validation_rows(data)
    if not rows:
        logger.error("no validation rows extracted!")
        return 1

    # Summary stats
    salts = [r["salt"] for r in rows]
    Ts = np.array([r["T"] for r in rows])
    sigmas = np.array([r["sigma_md"] for r in rows])
    n_solvents = [len(r["species_list"]) - 2 for r in rows]
    logger.info("summary:")
    logger.info("  salts: %s", {s: salts.count(s) for s in set(salts)})
    logger.info("  T range: %.1f - %.1f K (median %.1f)", Ts.min(), Ts.max(), np.median(Ts))
    logger.info("  sigma_md range: %.3f - %.3f mS/cm (median %.3f)",
                sigmas.min(), sigmas.max(), np.median(sigmas))
    logger.info("  n_solvents distribution: %s", {n: n_solvents.count(n) for n in set(n_solvents)})

    np.savez(
        OUTPUT_PATH,
        n_rows=len(rows),
        species_list_json=np.array([json.dumps(r["species_list"]) for r in rows], dtype=object),
        mole_fractions_json=np.array([json.dumps(r["mole_fractions"]) for r in rows], dtype=object),
        T=Ts,
        sigma_md=sigmas,
        salt=np.array(salts, dtype=object),
    )
    logger.info("wrote %s (%d rows)", OUTPUT_PATH, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
