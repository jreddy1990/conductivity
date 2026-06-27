"""Name-blind site-measure features for conductivity transport kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from constants import N_A
from data.species_data import ADDITIVES, CATION_PROPERTIES, SALTS
from utils.strict_validation import require_float, require_mapping, require_string


ANGSTROM3_TO_CM3_PER_MOL = 1.0e-24 * N_A
CATION_RADIUS_MATCH_TOLERANCE_A = 1.0e-9
DONOR_GROUP_NAMES = ("S=O", "B-O", "C=O", "N=O", "P=O")


@dataclass(frozen=True)
class CationSiteFeature:
    canonical_feature_id: str
    charge: int
    ion_symbol: str
    ionic_radius_A: float
    solvated_radius_A: float
    stokes_einstein_alpha: float
    molar_volume_cm3_mol: float


@dataclass(frozen=True)
class AnionSiteFeature:
    feature_id: str
    canonical_feature_id: str
    molarity_M: float
    charge: int
    carrier_label: str
    limiting_molar_conductivity_S_cm2_mol: float
    cation_radius_A: float
    anion_radius_A: float
    anion_volume_A3: float
    anion_molar_volume_cm3_mol: float
    bjerrum_association_reference_M_inv: float
    ion_pair_binding_kJ_mol: float
    stokes_einstein_alpha_anion: float
    preferred_coordination_number: float
    donor_site_count: float
    coordination_multiplicity: float
    ligand_field_asymmetry: float
    shape_friction_factor: float
    feature_descriptor: str


@dataclass(frozen=True)
class NeutralLigandSiteFeature:
    feature_id: str
    canonical_feature_id: str
    molarity_M: float
    coordination_affinity_M_inv: float
    molecular_volume_cm3_mol: float
    donor_site_count: float
    ligand_field_asymmetry: float
    feature_descriptor: str


@dataclass(frozen=True)
class TransportSiteMeasure:
    cation: CationSiteFeature
    anion_sites: tuple[AnionSiteFeature, ...]
    neutral_ligand_sites: tuple[NeutralLigandSiteFeature, ...]

    def anion_by_canonical_id(self) -> dict[str, AnionSiteFeature]:
        return {site.canonical_feature_id: site for site in self.anion_sites}

    def neutral_ligand_by_canonical_id(self) -> dict[str, NeutralLigandSiteFeature]:
        return {site.canonical_feature_id: site for site in self.neutral_ligand_sites}


def build_transport_site_measure(composition: Any) -> TransportSiteMeasure:
    """Parse registry-named composition into generic site features."""

    anion_sites: list[AnionSiteFeature] = []
    cation_names: set[str] = set()
    for source_index, source_name in enumerate(composition.ionic_source_molarities_M):
        molarity_M = float(composition.ionic_source_molarities_M[source_name])
        if molarity_M <= 0.0:
            continue
        source_props = _ionic_source_props(source_name)
        cation_name = _ionic_source_cation_name(source_name, source_props)
        cation_names.add(cation_name)
        canonical_feature_id = f"anion_site_{len(anion_sites)}"
        anion_sites.append(
            _anion_site_feature(
                source_name=source_name,
                source_props=source_props,
                molarity_M=molarity_M,
                feature_id=f"registry_anion_source_{source_index}",
                canonical_feature_id=canonical_feature_id,
            )
        )
    if not anion_sites:
        raise ValueError("transport site measure requires at least one positive ionic source")
    if len(cation_names) != 1:
        raise ValueError(f"transport site measure requires one cation family, found {sorted(cation_names)}")
    cation_name = next(iter(cation_names))
    cation_feature = _cation_site_feature(cation_name)

    neutral_ligand_sites: list[NeutralLigandSiteFeature] = []
    for ligand_index, additive_name in enumerate(composition.additive_molarities_M):
        molarity_M = float(composition.additive_molarities_M[additive_name])
        if molarity_M <= 0.0:
            continue
        additive_props = _require_species(ADDITIVES, additive_name, "additive")
        if _is_ionic_source_props(additive_props):
            continue
        if "f_donor" not in additive_props:
            raise ValueError(f"additive {additive_name} missing f_donor")
        if not bool(additive_props["f_donor"]):
            continue
        canonical_feature_id = f"neutral_ligand_site_{len(neutral_ligand_sites)}"
        neutral_ligand_sites.append(
            _neutral_ligand_site_feature(
                additive_name=additive_name,
                additive_props=additive_props,
                molarity_M=molarity_M,
                feature_id=f"registry_neutral_ligand_{ligand_index}",
                canonical_feature_id=canonical_feature_id,
            )
        )

    return TransportSiteMeasure(
        cation=cation_feature,
        anion_sites=tuple(anion_sites),
        neutral_ligand_sites=tuple(neutral_ligand_sites),
    )


def rename_transport_site_measure_feature_ids(
    site_measure: TransportSiteMeasure,
    anion_feature_prefix: str,
    neutral_ligand_feature_prefix: str,
) -> TransportSiteMeasure:
    """Rename opaque feature IDs without changing canonical physical features."""

    renamed_anions = tuple(
        replace(site, feature_id=f"{anion_feature_prefix}_{site_index}")
        for site_index, site in enumerate(site_measure.anion_sites)
    )
    renamed_ligands = tuple(
        replace(site, feature_id=f"{neutral_ligand_feature_prefix}_{site_index}")
        for site_index, site in enumerate(site_measure.neutral_ligand_sites)
    )
    return TransportSiteMeasure(
        cation=site_measure.cation,
        anion_sites=renamed_anions,
        neutral_ligand_sites=renamed_ligands,
    )


def anion_shape_friction_factor_from_feature(source_props: Mapping[str, Any], context: str) -> float:
    anion_radius_A = require_float(source_props, "anion_radius", context)
    anion_volume_A3 = require_float(source_props, "anion_volume", context)
    _assert_positive_float(anion_radius_A, f"{context}.anion_radius")
    _assert_positive_float(anion_volume_A3, f"{context}.anion_volume")
    equivalent_sphere_radius_A = (3.0 * anion_volume_A3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    _assert_positive_float(equivalent_sphere_radius_A, f"{context}.equivalent_sphere_radius")
    if anion_radius_A >= equivalent_sphere_radius_A:
        radial_asphericity = anion_radius_A / equivalent_sphere_radius_A
    else:
        radial_asphericity = equivalent_sphere_radius_A / anion_radius_A
    ligand_asymmetry = ligand_field_asymmetry_from_feature(source_props, context)
    factor = math.sqrt(radial_asphericity * (1.0 + ligand_asymmetry))
    _assert_positive_float(factor, f"{context}.shape_friction_factor")
    if factor < 1.0:
        raise ValueError(f"{context}.shape_friction_factor must be at least one, got {factor}")
    return factor


def donor_site_count_from_feature(props: Mapping[str, Any], context: str) -> float:
    if "functional_groups" not in props:
        return 0.0
    functional_groups = require_mapping(props, "functional_groups", context)
    donor_site_count = 0.0
    for donor_group_name in DONOR_GROUP_NAMES:
        if donor_group_name not in functional_groups:
            continue
        donor_site_count += require_float(functional_groups, donor_group_name, f"{context}.functional_groups")
    _assert_nonnegative_float(donor_site_count, f"{context}.donor_site_count")
    return donor_site_count


def coordination_multiplicity_from_feature(props: Mapping[str, Any], context: str) -> float:
    if "coordination_mode" not in props:
        return 1.0
    coordination_mode = require_string(props, "coordination_mode", context)
    if coordination_mode == "monodentate":
        return 1.0
    if coordination_mode == "bidentate":
        return 2.0
    raise ValueError(f"Unsupported coordination mode {coordination_mode} for {context}")


def ligand_field_asymmetry_from_feature(props: Mapping[str, Any], context: str) -> float:
    if "ligand_field_asymmetry" in props:
        asymmetry = require_float(props, "ligand_field_asymmetry", context)
        _assert_nonnegative_float(asymmetry, f"{context}.ligand_field_asymmetry")
        return asymmetry
    donor_site_count = donor_site_count_from_feature(props, context)
    if donor_site_count <= 0.0:
        return 0.0
    coordination_multiplicity = coordination_multiplicity_from_feature(props, context)
    return donor_site_count / (donor_site_count + coordination_multiplicity)


def _anion_site_feature(
    source_name: str,
    source_props: Mapping[str, Any],
    molarity_M: float,
    feature_id: str,
    canonical_feature_id: str,
) -> AnionSiteFeature:
    context = f"ionic source {source_name}"
    anion_volume_A3 = require_float(source_props, "anion_volume", context)
    donor_site_count = donor_site_count_from_feature(source_props, context)
    coordination_multiplicity = coordination_multiplicity_from_feature(source_props, context)
    ligand_asymmetry = ligand_field_asymmetry_from_feature(source_props, context)
    return AnionSiteFeature(
        feature_id=feature_id,
        canonical_feature_id=canonical_feature_id,
        molarity_M=molarity_M,
        charge=int(require_float(source_props, "anion_charge", context)),
        carrier_label=canonical_feature_id,
        limiting_molar_conductivity_S_cm2_mol=require_float(source_props, "Lambda_0", context),
        cation_radius_A=require_float(source_props, "cation_radius", context),
        anion_radius_A=require_float(source_props, "anion_radius", context),
        anion_volume_A3=anion_volume_A3,
        anion_molar_volume_cm3_mol=anion_volume_A3 * ANGSTROM3_TO_CM3_PER_MOL,
        bjerrum_association_reference_M_inv=require_float(source_props, "bjerrum_K_A_ref", context),
        ion_pair_binding_kJ_mol=require_float(source_props, "ion_pair_binding_kj_mol", context),
        stokes_einstein_alpha_anion=require_float(source_props, "stokes_einstein_alpha_anion", context),
        preferred_coordination_number=_ionic_source_preferred_coordination_number(source_name, source_props),
        donor_site_count=donor_site_count,
        coordination_multiplicity=coordination_multiplicity,
        ligand_field_asymmetry=ligand_asymmetry,
        shape_friction_factor=anion_shape_friction_factor_from_feature(source_props, context),
        feature_descriptor=_anion_feature_descriptor(
            source_props,
            donor_site_count,
            coordination_multiplicity,
            ligand_asymmetry,
        ),
    )


def _neutral_ligand_site_feature(
    additive_name: str,
    additive_props: Mapping[str, Any],
    molarity_M: float,
    feature_id: str,
    canonical_feature_id: str,
) -> NeutralLigandSiteFeature:
    context = f"neutral ligand {additive_name}"
    molecular_weight_g_mol = require_float(additive_props, "molecular_weight", context)
    density_g_ml = require_float(additive_props, "density_g_ml", context)
    _assert_positive_float(molecular_weight_g_mol, f"{context}.molecular_weight")
    _assert_positive_float(density_g_ml, f"{context}.density")
    donor_site_count = donor_site_count_from_feature(additive_props, context)
    ligand_asymmetry = ligand_field_asymmetry_from_feature(additive_props, context)
    return NeutralLigandSiteFeature(
        feature_id=feature_id,
        canonical_feature_id=canonical_feature_id,
        molarity_M=molarity_M,
        coordination_affinity_M_inv=require_float(additive_props, "coordination_affinity_M_inv", context),
        molecular_volume_cm3_mol=molecular_weight_g_mol / density_g_ml,
        donor_site_count=donor_site_count,
        ligand_field_asymmetry=ligand_asymmetry,
        feature_descriptor=_neutral_ligand_feature_descriptor(additive_props, donor_site_count, ligand_asymmetry),
    )


def _cation_site_feature(cation_name: str) -> CationSiteFeature:
    cation_props = _require_species(CATION_PROPERTIES, cation_name, "cation")
    ionic_radius_A = require_float(cation_props, "ionic_radius_A", f"cation {cation_name}")
    return CationSiteFeature(
        canonical_feature_id="cation_site_0",
        charge=int(require_float(cation_props, "charge", f"cation {cation_name}")),
        ion_symbol=require_string(cation_props, "ion_symbol", f"cation {cation_name}"),
        ionic_radius_A=ionic_radius_A,
        solvated_radius_A=require_float(cation_props, "solvated_radius_A", f"cation {cation_name}"),
        stokes_einstein_alpha=require_float(cation_props, "stokes_einstein_alpha", f"cation {cation_name}"),
        molar_volume_cm3_mol=_sphere_molar_volume_cm3_mol(ionic_radius_A),
    )


def _anion_feature_descriptor(
    source_props: Mapping[str, Any],
    donor_site_count: float,
    coordination_multiplicity: float,
    ligand_asymmetry: float,
) -> str:
    charge = int(require_float(source_props, "anion_charge", "anion feature"))
    volume = require_float(source_props, "anion_volume", "anion feature")
    radius = require_float(source_props, "anion_radius", "anion feature")
    return (
        f"anion_z{charge}_donor{donor_site_count:.3g}_dent{coordination_multiplicity:.3g}_"
        f"shape{ligand_asymmetry:.3g}_volume{volume:.3g}_radius{radius:.3g}"
    )


def _neutral_ligand_feature_descriptor(
    additive_props: Mapping[str, Any],
    donor_site_count: float,
    ligand_asymmetry: float,
) -> str:
    volume = require_float(additive_props, "molecular_weight", "neutral ligand feature") / require_float(
        additive_props,
        "density_g_ml",
        "neutral ligand feature",
    )
    affinity = require_float(additive_props, "coordination_affinity_M_inv", "neutral ligand feature")
    return (
        f"neutral_ligand_donor{donor_site_count:.3g}_shape{ligand_asymmetry:.3g}_"
        f"volume{volume:.3g}_affinity{affinity:.3g}"
    )


def _ionic_source_props(source_name: str) -> Mapping[str, Any]:
    if source_name in SALTS:
        return SALTS[source_name]
    if source_name in ADDITIVES:
        props = ADDITIVES[source_name]
        if _is_ionic_source_props(props):
            return props
    raise ValueError(f"Species {source_name} is not an ionic source")


def _ionic_source_cation_name(source_name: str, props: Mapping[str, Any]) -> str:
    if "cation" in props:
        return require_string(props, "cation", f"ionic source {source_name}")
    cation_radius_A = require_float(props, "cation_radius", f"ionic source {source_name}")
    matches: list[str] = []
    for cation_name, cation_props in CATION_PROPERTIES.items():
        reference_radius_A = require_float(cation_props, "ionic_radius_A", f"cation {cation_name}")
        if abs(reference_radius_A - cation_radius_A) <= CATION_RADIUS_MATCH_TOLERANCE_A:
            matches.append(cation_name)
    if len(matches) != 1:
        raise ValueError(
            f"ionic source {source_name} cation_radius {cation_radius_A} A matched cations {matches}"
        )
    return matches[0]


def _ionic_source_preferred_coordination_number(source_name: str, props: Mapping[str, Any]) -> float:
    if "preferred_coordination_number" in props:
        return require_float(props, "preferred_coordination_number", f"ionic source {source_name}")
    cation_name = _ionic_source_cation_name(source_name, props)
    reference_values: list[float] = []
    for salt_name, salt_props in SALTS.items():
        salt_cation_name = _ionic_source_cation_name(salt_name, salt_props)
        if salt_cation_name != cation_name:
            continue
        reference_values.append(
            require_float(
                salt_props,
                "preferred_coordination_number",
                f"salt reference {salt_name}",
            )
        )
    if not reference_values:
        raise ValueError(f"ionic source {source_name} has no cation-family preferred coordination reference")
    reference_min = min(reference_values)
    reference_max = max(reference_values)
    if reference_max - reference_min > CATION_RADIUS_MATCH_TOLERANCE_A:
        raise ValueError(
            f"ionic source {source_name} cation {cation_name} has non-unique coordination references "
            f"{reference_values}"
        )
    return reference_values[0]


def _is_ionic_source_props(props: Mapping[str, Any]) -> bool:
    has_cation_identity = "cation" in props or "cation_radius" in props
    return has_cation_identity and "anion" in props and "Lambda_0" in props


def _require_species(
    species_map: Mapping[str, Mapping[str, Any]],
    species_name: str,
    species_kind: str,
) -> Mapping[str, Any]:
    if species_name not in species_map:
        raise ValueError(f"Unknown {species_kind} species {species_name}")
    return species_map[species_name]


def _sphere_molar_volume_cm3_mol(radius_A: float) -> float:
    return 4.0 / 3.0 * math.pi * radius_A * radius_A * radius_A * ANGSTROM3_TO_CM3_PER_MOL


def _assert_positive_float(value: float, context: str) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{context} must be positive and finite, got {value}")


def _assert_nonnegative_float(value: float, context: str) -> None:
    if value < 0.0 or not math.isfinite(value):
        raise ValueError(f"{context} must be nonnegative and finite, got {value}")
