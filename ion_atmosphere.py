"""Ion-atmosphere friction state for conductivity transport kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from constants import EPS_0, F, K_B, N_A, R


SUPPORTED_ION_ATMOSPHERE_SOLVERS = ("off", "diagonal_pnp_stokes_l1_cell_experimental")
SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS = ("off", "finite_size_bulk_pnp_stokes_l1_cell")
STOKES_TRANSLATION_AXIS_COUNT = 3  # Cartesian translation axes in the spherical Stokes drag solution.
STOKES_NO_SLIP_BOUNDARY_FACTOR = 2  # No-slip sphere doubles the axis count in zeta = 6*pi*eta*a.
STOKES_SPHERE_DRAG_FACTOR = STOKES_NO_SLIP_BOUNDARY_FACTOR * STOKES_TRANSLATION_AXIS_COUNT
SPHERE_VOLUME_NUMERATOR = STOKES_SPHERE_DRAG_FACTOR - STOKES_NO_SLIP_BOUNDARY_FACTOR
SPHERE_VOLUME_DENOMINATOR = STOKES_TRANSLATION_AXIS_COUNT


@dataclass(frozen=True)
class IonAtmosphereInput:
    carrier_concentrations_mol_m3: Mapping[str, float]
    carrier_charges: Mapping[str, int]
    local_diffusivity_m2_s_by_carrier: Mapping[str, float]
    hydrodynamic_radius_m_by_carrier: Mapping[str, float]
    viscosity_Pa_s: float
    relative_dielectric: float
    temperature_K: float
    solver: str


@dataclass(frozen=True)
class BulkIonAtmosphereInput:
    carrier_labels: tuple[str, ...]
    carrier_concentrations_mol_m3: Mapping[str, float]
    carrier_charges: Mapping[str, int]
    local_diffusivity_m2_s_by_carrier: Mapping[str, float]
    hydrodynamic_radius_m_by_carrier: Mapping[str, float]
    viscosity_Pa_s: float
    relative_dielectric: float
    temperature_K: float
    solver: str


@dataclass(frozen=True)
class IonAtmosphereState:
    kappa_inv_m: float
    ionic_strength_mol_m3: float
    friction_ratio_by_carrier: dict[str, float]
    zeta0_by_carrier: dict[str, float]
    zeta_ep_by_carrier: dict[str, float]
    zeta_rel_by_carrier: dict[str, float]
    zeta_atm_by_carrier: dict[str, float]
    D_micro_by_carrier: dict[str, float]
    D_eff_by_carrier: dict[str, float]
    solver: str


@dataclass(frozen=True)
class BulkIonAtmosphereState:
    carrier_labels: tuple[str, ...]
    kappa_inv_m: float
    ionic_strength_mol_m3: float
    ambipolar_diffusivity_m2_s: float
    resistance_matrix_kg_s: np.ndarray
    resistance_ep_kg_s: np.ndarray
    resistance_rel_kg_s: np.ndarray
    steric_volume_fraction: float
    thermodynamic_factor_trace: float
    thermodynamic_factor_matrix: np.ndarray
    thermodynamic_factor_eigenvalues: tuple[float, ...]
    structure_response_matrix: np.ndarray
    structure_factor_charge_mode: float
    kappa_radius_by_carrier: dict[str, float]
    solver: str


def build_ion_atmosphere_state(ion_atmosphere_input: IonAtmosphereInput) -> IonAtmosphereState:
    """Build ion-atmosphere friction diagnostics for charged mobile carriers."""

    _assert_positive_finite(ion_atmosphere_input.viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(ion_atmosphere_input.relative_dielectric, "relative_dielectric")
    _assert_positive_finite(ion_atmosphere_input.temperature_K, "temperature_K")
    _validate_solver(ion_atmosphere_input.solver)

    carrier_names = tuple(ion_atmosphere_input.carrier_concentrations_mol_m3)
    if not carrier_names:
        raise ValueError("ion atmosphere requires at least one charged carrier")

    charge_weighted_concentration_mol_m3 = 0.0
    friction_ratio_by_carrier: dict[str, float] = {}
    zeta0_by_carrier: dict[str, float] = {}
    zeta_ep_by_carrier: dict[str, float] = {}
    zeta_rel_by_carrier: dict[str, float] = {}
    zeta_atm_by_carrier: dict[str, float] = {}
    D_micro_by_carrier: dict[str, float] = {}
    D_eff_by_carrier: dict[str, float] = {}
    local_diffusivity_by_carrier: dict[str, float] = {}
    hydrodynamic_radius_by_carrier: dict[str, float] = {}
    charge_by_carrier: dict[str, int] = {}

    for carrier_name in carrier_names:
        concentration_mol_m3 = _require_nonnegative_finite(
            ion_atmosphere_input.carrier_concentrations_mol_m3,
            carrier_name,
            "carrier_concentrations_mol_m3",
        )
        charge_number = _require_charge(ion_atmosphere_input.carrier_charges, carrier_name)
        local_diffusivity_m2_s = _require_positive_finite(
            ion_atmosphere_input.local_diffusivity_m2_s_by_carrier,
            carrier_name,
            "local_diffusivity_m2_s_by_carrier",
        )
        hydrodynamic_radius_m = _require_positive_finite(
            ion_atmosphere_input.hydrodynamic_radius_m_by_carrier,
            carrier_name,
            "hydrodynamic_radius_m_by_carrier",
        )
        charge_weighted_concentration_mol_m3 += charge_number * charge_number * concentration_mol_m3
        local_diffusivity_by_carrier[carrier_name] = local_diffusivity_m2_s
        hydrodynamic_radius_by_carrier[carrier_name] = hydrodynamic_radius_m
        charge_by_carrier[carrier_name] = charge_number

    kappa_inv_m = _debye_kappa_inv_m(
        charge_weighted_concentration_mol_m3,
        ion_atmosphere_input.relative_dielectric,
        ion_atmosphere_input.temperature_K,
    )
    if math.isinf(kappa_inv_m):
        kappa_m_inv = 0.0
    else:
        kappa_m_inv = 1.0 / kappa_inv_m

    for carrier_name in carrier_names:
        local_diffusivity_m2_s = local_diffusivity_by_carrier[carrier_name]
        hydrodynamic_radius_m = hydrodynamic_radius_by_carrier[carrier_name]
        charge_number = charge_by_carrier[carrier_name]
        zeta0_kg_s = K_B * ion_atmosphere_input.temperature_K / local_diffusivity_m2_s
        if ion_atmosphere_input.solver == "off":
            zeta_ep_kg_s = 0.0
            zeta_rel_kg_s = 0.0
        elif ion_atmosphere_input.solver == "diagonal_pnp_stokes_l1_cell_experimental":
            zeta_ep_kg_s = _electrophoretic_drag_kg_s(
                viscosity_Pa_s=ion_atmosphere_input.viscosity_Pa_s,
                hydrodynamic_radius_m=hydrodynamic_radius_m,
                kappa_m_inv=kappa_m_inv,
            )
            zeta_rel_kg_s = _relaxation_drag_kg_s(
                charge_number=charge_number,
                local_diffusivity_m2_s=local_diffusivity_m2_s,
                hydrodynamic_radius_m=hydrodynamic_radius_m,
                relative_dielectric=ion_atmosphere_input.relative_dielectric,
                kappa_m_inv=kappa_m_inv,
            )
        else:
            raise ValueError(f"Unsupported ion-atmosphere solver {ion_atmosphere_input.solver!r}")
        zeta_atm_kg_s = zeta_ep_kg_s + zeta_rel_kg_s
        _assert_nonnegative_finite(zeta_ep_kg_s, f"{carrier_name}.zeta_ep_kg_s")
        _assert_nonnegative_finite(zeta_rel_kg_s, f"{carrier_name}.zeta_rel_kg_s")
        _assert_nonnegative_finite(zeta_atm_kg_s, f"{carrier_name}.zeta_atm_kg_s")
        friction_ratio = zeta0_kg_s / (zeta0_kg_s + zeta_atm_kg_s)
        if friction_ratio <= 0.0 or friction_ratio > 1.0:
            raise ValueError(f"{carrier_name}.friction_ratio must be in (0, 1], got {friction_ratio}")

        zeta0_by_carrier[carrier_name] = zeta0_kg_s
        zeta_ep_by_carrier[carrier_name] = zeta_ep_kg_s
        zeta_rel_by_carrier[carrier_name] = zeta_rel_kg_s
        zeta_atm_by_carrier[carrier_name] = zeta_atm_kg_s
        D_micro_by_carrier[carrier_name] = local_diffusivity_m2_s
        D_eff_by_carrier[carrier_name] = local_diffusivity_m2_s * friction_ratio
        friction_ratio_by_carrier[carrier_name] = friction_ratio

    return IonAtmosphereState(
        kappa_inv_m=kappa_inv_m,
        ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
        friction_ratio_by_carrier=friction_ratio_by_carrier,
        zeta0_by_carrier=zeta0_by_carrier,
        zeta_ep_by_carrier=zeta_ep_by_carrier,
        zeta_rel_by_carrier=zeta_rel_by_carrier,
        zeta_atm_by_carrier=zeta_atm_by_carrier,
        D_micro_by_carrier=D_micro_by_carrier,
        D_eff_by_carrier=D_eff_by_carrier,
        solver=ion_atmosphere_input.solver,
    )


def build_bulk_ion_atmosphere_state(
    bulk_ion_atmosphere_input: BulkIonAtmosphereInput,
) -> BulkIonAtmosphereState:
    """Build a finite-size bulk carrier atmosphere resistance matrix."""

    _assert_positive_finite(bulk_ion_atmosphere_input.viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(bulk_ion_atmosphere_input.relative_dielectric, "relative_dielectric")
    _assert_positive_finite(bulk_ion_atmosphere_input.temperature_K, "temperature_K")
    _validate_bulk_solver(bulk_ion_atmosphere_input.solver)
    carrier_labels = tuple(bulk_ion_atmosphere_input.carrier_labels)
    if not carrier_labels:
        raise ValueError("bulk ion atmosphere requires at least one charged carrier")
    if len(set(carrier_labels)) != len(carrier_labels):
        raise ValueError("bulk ion atmosphere carrier_labels must be unique")

    charge_weighted_concentration_mol_m3 = 0.0
    steric_volume_fraction = 0.0
    concentration_by_carrier: dict[str, float] = {}
    charge_by_carrier: dict[str, int] = {}
    diffusivity_by_carrier: dict[str, float] = {}
    radius_by_carrier: dict[str, float] = {}
    for carrier_label in carrier_labels:
        concentration_mol_m3 = _require_nonnegative_finite(
            bulk_ion_atmosphere_input.carrier_concentrations_mol_m3,
            carrier_label,
            "carrier_concentrations_mol_m3",
        )
        charge_number = _require_charge(bulk_ion_atmosphere_input.carrier_charges, carrier_label)
        local_diffusivity_m2_s = _require_positive_finite(
            bulk_ion_atmosphere_input.local_diffusivity_m2_s_by_carrier,
            carrier_label,
            "local_diffusivity_m2_s_by_carrier",
        )
        hydrodynamic_radius_m = _require_positive_finite(
            bulk_ion_atmosphere_input.hydrodynamic_radius_m_by_carrier,
            carrier_label,
            "hydrodynamic_radius_m_by_carrier",
        )
        charge_weighted_concentration_mol_m3 += charge_number * charge_number * concentration_mol_m3
        steric_volume_fraction += (
            concentration_mol_m3
            * N_A
            * _sphere_volume_m3(hydrodynamic_radius_m)
        )
        concentration_by_carrier[carrier_label] = concentration_mol_m3
        charge_by_carrier[carrier_label] = charge_number
        diffusivity_by_carrier[carrier_label] = local_diffusivity_m2_s
        radius_by_carrier[carrier_label] = hydrodynamic_radius_m
    _assert_nonnegative_finite(steric_volume_fraction, "steric_volume_fraction")
    if steric_volume_fraction >= 1.0:
        raise ValueError(f"steric_volume_fraction must be below one, got {steric_volume_fraction}")
    ambipolar_diffusivity_m2_s = _ambipolar_diffusivity_m2_s(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        diffusivity_by_carrier=diffusivity_by_carrier,
    )

    kappa_inv_m = _debye_kappa_inv_m(
        charge_weighted_concentration_mol_m3,
        bulk_ion_atmosphere_input.relative_dielectric,
        bulk_ion_atmosphere_input.temperature_K,
    )
    if (
        bulk_ion_atmosphere_input.solver == "off"
        or charge_weighted_concentration_mol_m3 == 0.0
    ):
        matrix_shape = (len(carrier_labels), len(carrier_labels))
        zero_matrix = np.zeros(matrix_shape, dtype=float)
        thermodynamic_factor_matrix = _finite_size_thermodynamic_factor_matrix(
            carrier_labels=carrier_labels,
            concentration_by_carrier=concentration_by_carrier,
            radius_by_carrier=radius_by_carrier,
            steric_volume_fraction=steric_volume_fraction,
        )
        thermodynamic_factor_trace = float(np.trace(thermodynamic_factor_matrix))
        thermodynamic_factor_eigenvalues = _matrix_eigenvalue_tuple(
            thermodynamic_factor_matrix,
            "thermodynamic_factor_matrix",
        )
        structure_response_matrix = thermodynamic_factor_matrix.copy()
        return BulkIonAtmosphereState(
            carrier_labels=carrier_labels,
            kappa_inv_m=kappa_inv_m,
            ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
            ambipolar_diffusivity_m2_s=ambipolar_diffusivity_m2_s,
            resistance_matrix_kg_s=zero_matrix,
            resistance_ep_kg_s=zero_matrix.copy(),
            resistance_rel_kg_s=zero_matrix.copy(),
            steric_volume_fraction=steric_volume_fraction,
            thermodynamic_factor_trace=thermodynamic_factor_trace,
            thermodynamic_factor_matrix=thermodynamic_factor_matrix,
            thermodynamic_factor_eigenvalues=thermodynamic_factor_eigenvalues,
            structure_response_matrix=structure_response_matrix,
            structure_factor_charge_mode=0.0,
            kappa_radius_by_carrier={carrier_label: 0.0 for carrier_label in carrier_labels},
            solver=bulk_ion_atmosphere_input.solver,
        )
    if bulk_ion_atmosphere_input.solver != "finite_size_bulk_pnp_stokes_l1_cell":
        raise ValueError(f"Unsupported bulk ion-atmosphere solver {bulk_ion_atmosphere_input.solver!r}")
    if math.isinf(kappa_inv_m):
        kappa_m_inv = 0.0
    else:
        kappa_m_inv = 1.0 / kappa_inv_m
    finite_size_result = _finite_size_bulk_resistance_matrices_kg_s(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        diffusivity_by_carrier=diffusivity_by_carrier,
        radius_by_carrier=radius_by_carrier,
        viscosity_Pa_s=bulk_ion_atmosphere_input.viscosity_Pa_s,
        relative_dielectric=bulk_ion_atmosphere_input.relative_dielectric,
        kappa_m_inv=kappa_m_inv,
        steric_volume_fraction=steric_volume_fraction,
    )
    zeta_ep_values_kg_s = finite_size_result[0]
    zeta_rel_values_kg_s = finite_size_result[1]
    kappa_radius_by_carrier = finite_size_result[2]
    overlap_values = finite_size_result[3]
    relaxation_sign_values = finite_size_result[4]
    thermodynamic_factor_matrix = _finite_size_thermodynamic_factor_matrix(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        radius_by_carrier=radius_by_carrier,
        steric_volume_fraction=steric_volume_fraction,
    )
    thermodynamic_factor_trace = float(np.trace(thermodynamic_factor_matrix))
    thermodynamic_factor_eigenvalues = _matrix_eigenvalue_tuple(
        thermodynamic_factor_matrix,
        "thermodynamic_factor_matrix",
    )
    structure_response = _finite_size_structure_response_matrix(
        carrier_labels=carrier_labels,
        concentration_by_carrier=concentration_by_carrier,
        charge_by_carrier=charge_by_carrier,
        radius_by_carrier=radius_by_carrier,
        kappa_m_inv=kappa_m_inv,
        thermodynamic_factor_matrix=thermodynamic_factor_matrix,
    )
    structure_response_matrix = structure_response[0]
    structure_factor_charge_mode = structure_response[1]
    electrophoretic_sign_values = np.ones(len(carrier_labels), dtype=float)
    resistance_ep_kg_s = _matrix_coupled_psd_component_kg_s(
        zeta_values_kg_s=zeta_ep_values_kg_s,
        overlap_values=overlap_values,
        coupling_sign_values=electrophoretic_sign_values,
        structure_response_matrix=structure_response_matrix,
    )
    resistance_rel_kg_s = _matrix_coupled_psd_component_kg_s(
        zeta_values_kg_s=zeta_rel_values_kg_s,
        overlap_values=overlap_values,
        coupling_sign_values=relaxation_sign_values,
        structure_response_matrix=structure_response_matrix,
    )
    resistance_matrix_kg_s = resistance_ep_kg_s + resistance_rel_kg_s
    _validate_bulk_resistance_matrix(resistance_matrix_kg_s, "resistance_matrix_kg_s")
    return BulkIonAtmosphereState(
        carrier_labels=carrier_labels,
        kappa_inv_m=kappa_inv_m,
        ionic_strength_mol_m3=charge_weighted_concentration_mol_m3,
        ambipolar_diffusivity_m2_s=ambipolar_diffusivity_m2_s,
        resistance_matrix_kg_s=resistance_matrix_kg_s,
        resistance_ep_kg_s=resistance_ep_kg_s,
        resistance_rel_kg_s=resistance_rel_kg_s,
        steric_volume_fraction=steric_volume_fraction,
        thermodynamic_factor_trace=thermodynamic_factor_trace,
        thermodynamic_factor_matrix=thermodynamic_factor_matrix,
        thermodynamic_factor_eigenvalues=thermodynamic_factor_eigenvalues,
        structure_response_matrix=structure_response_matrix,
        structure_factor_charge_mode=structure_factor_charge_mode,
        kappa_radius_by_carrier=kappa_radius_by_carrier,
        solver=bulk_ion_atmosphere_input.solver,
    )


def _ambipolar_diffusivity_m2_s(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    diffusivity_by_carrier: Mapping[str, float],
) -> float:
    charge_weighted_diffusivity_sum = 0.0
    charge_weighted_concentration_sum = 0.0
    for carrier_label in carrier_labels:
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        charge_number = charge_by_carrier[carrier_label]
        diffusivity_m2_s = diffusivity_by_carrier[carrier_label]
        charge_weight = charge_number * charge_number * concentration_mol_m3
        charge_weighted_diffusivity_sum += charge_weight * diffusivity_m2_s
        charge_weighted_concentration_sum += charge_weight
    _assert_nonnegative_finite(
        charge_weighted_concentration_sum,
        "charge_weighted_concentration_sum",
    )
    if charge_weighted_concentration_sum == 0.0:
        return 0.0
    ambipolar_diffusivity_m2_s = (
        charge_weighted_diffusivity_sum / charge_weighted_concentration_sum
    )
    _assert_positive_finite(ambipolar_diffusivity_m2_s, "ambipolar_diffusivity_m2_s")
    return ambipolar_diffusivity_m2_s


def _finite_size_thermodynamic_factor_matrix(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    radius_by_carrier: Mapping[str, float],
    steric_volume_fraction: float,
) -> np.ndarray:
    free_volume_fraction = 1.0 - steric_volume_fraction
    _assert_positive_finite(free_volume_fraction, "free_volume_fraction")
    carrier_count = len(carrier_labels)
    finite_volume_vector = np.zeros(carrier_count, dtype=float)
    for carrier_index, carrier_label in enumerate(carrier_labels):
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        hydrodynamic_radius_m = radius_by_carrier[carrier_label]
        finite_volume_argument = (
            concentration_mol_m3
            * N_A
            * _sphere_volume_m3(hydrodynamic_radius_m)
        )
        _assert_nonnegative_finite(
            finite_volume_argument,
            f"{carrier_label}.finite_volume_argument",
        )
        finite_volume_vector[carrier_index] = math.sqrt(finite_volume_argument)
    thermodynamic_factor_matrix = (
        np.eye(carrier_count, dtype=float)
        + np.outer(finite_volume_vector, finite_volume_vector) / free_volume_fraction
    )
    _validate_bulk_resistance_matrix(
        thermodynamic_factor_matrix,
        "thermodynamic_factor_matrix",
    )
    return thermodynamic_factor_matrix


def _finite_size_structure_response_matrix(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    radius_by_carrier: Mapping[str, float],
    kappa_m_inv: float,
    thermodynamic_factor_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    charge_weighted_concentration = 0.0
    concentration_sum_mol_m3 = 0.0
    radius_weighted_concentration_m = 0.0
    charge_mode_vector = np.zeros(len(carrier_labels), dtype=float)
    for carrier_index, carrier_label in enumerate(carrier_labels):
        concentration_mol_m3 = concentration_by_carrier[carrier_label]
        charge_number = charge_by_carrier[carrier_label]
        radius_m = radius_by_carrier[carrier_label]
        charge_weighted_concentration += charge_number * charge_number * concentration_mol_m3
        concentration_sum_mol_m3 += concentration_mol_m3
        radius_weighted_concentration_m += concentration_mol_m3 * radius_m
        charge_mode_vector[carrier_index] = math.sqrt(concentration_mol_m3) * abs(charge_number)
    if charge_weighted_concentration == 0.0:
        return (thermodynamic_factor_matrix.copy(), 0.0)
    _assert_positive_finite(concentration_sum_mol_m3, "concentration_sum_mol_m3")
    average_hydrodynamic_radius_m = radius_weighted_concentration_m / concentration_sum_mol_m3
    _assert_positive_finite(average_hydrodynamic_radius_m, "average_hydrodynamic_radius_m")
    charge_mode_norm = float(np.linalg.norm(charge_mode_vector))
    _assert_positive_finite(charge_mode_norm, "charge_mode_norm")
    normalized_charge_mode = charge_mode_vector / charge_mode_norm
    structure_factor_charge_mode = kappa_m_inv * average_hydrodynamic_radius_m
    _assert_nonnegative_finite(structure_factor_charge_mode, "structure_factor_charge_mode")
    structure_response_matrix = (
        thermodynamic_factor_matrix
        + structure_factor_charge_mode * np.outer(normalized_charge_mode, normalized_charge_mode)
    )
    _validate_bulk_resistance_matrix(
        structure_response_matrix,
        "structure_response_matrix",
    )
    return (structure_response_matrix, structure_factor_charge_mode)


def _finite_size_bulk_resistance_matrices_kg_s(
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    diffusivity_by_carrier: Mapping[str, float],
    radius_by_carrier: Mapping[str, float],
    viscosity_Pa_s: float,
    relative_dielectric: float,
    kappa_m_inv: float,
    steric_volume_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    free_volume_fraction = 1.0 - steric_volume_fraction
    _assert_positive_finite(free_volume_fraction, "free_volume_fraction")
    carrier_count = len(carrier_labels)
    zeta_ep_values = np.zeros(carrier_count, dtype=float)
    zeta_rel_values = np.zeros(carrier_count, dtype=float)
    overlap_values = np.zeros(carrier_count, dtype=float)
    relaxation_sign_values = np.zeros(carrier_count, dtype=float)
    kappa_radius_by_carrier: dict[str, float] = {}
    for carrier_index, carrier_label in enumerate(carrier_labels):
        hydrodynamic_radius_m = radius_by_carrier[carrier_label]
        stern_radius_m = hydrodynamic_radius_m + _opposite_charge_weighted_radius_m(
            carrier_label=carrier_label,
            carrier_labels=carrier_labels,
            concentration_by_carrier=concentration_by_carrier,
            charge_by_carrier=charge_by_carrier,
            radius_by_carrier=radius_by_carrier,
        )
        effective_kappa_m_inv = kappa_m_inv * free_volume_fraction / (1.0 + kappa_m_inv * stern_radius_m)
        kappa_radius_by_carrier[carrier_label] = effective_kappa_m_inv * hydrodynamic_radius_m
        zeta_ep_values[carrier_index] = _electrophoretic_drag_kg_s(
            viscosity_Pa_s=viscosity_Pa_s,
            hydrodynamic_radius_m=hydrodynamic_radius_m,
            kappa_m_inv=effective_kappa_m_inv,
        )
        zeta_rel_values[carrier_index] = _relaxation_drag_kg_s(
            charge_number=charge_by_carrier[carrier_label],
            local_diffusivity_m2_s=diffusivity_by_carrier[carrier_label],
            hydrodynamic_radius_m=hydrodynamic_radius_m,
            relative_dielectric=relative_dielectric,
            kappa_m_inv=effective_kappa_m_inv,
        )
        overlap_values[carrier_index] = math.exp(-effective_kappa_m_inv * stern_radius_m)
        relaxation_sign_values[carrier_index] = math.copysign(1.0, charge_by_carrier[carrier_label])
    return (
        zeta_ep_values,
        zeta_rel_values,
        kappa_radius_by_carrier,
        overlap_values,
        relaxation_sign_values,
    )


def _matrix_coupled_psd_component_kg_s(
    zeta_values_kg_s: np.ndarray,
    overlap_values: np.ndarray,
    coupling_sign_values: np.ndarray,
    structure_response_matrix: np.ndarray,
) -> np.ndarray:
    weighted_zeta_values = zeta_values_kg_s * overlap_values
    sign_matrix = np.diag(coupling_sign_values)
    weighted_matrix = np.diag(np.sqrt(weighted_zeta_values))
    residual_diagonal = zeta_values_kg_s * (1.0 - overlap_values)
    coupled_matrix = (
        sign_matrix
        @ weighted_matrix
        @ structure_response_matrix
        @ weighted_matrix
        @ sign_matrix
    )
    resistance_matrix = coupled_matrix + np.diag(residual_diagonal)
    _validate_bulk_resistance_matrix(resistance_matrix, "matrix_coupled_component_kg_s")
    return resistance_matrix


def _matrix_eigenvalue_tuple(
    matrix: np.ndarray,
    context: str,
) -> tuple[float, ...]:
    _validate_bulk_resistance_matrix(matrix, context)
    return tuple(float(value) for value in np.linalg.eigvalsh(matrix))


def _opposite_charge_weighted_radius_m(
    carrier_label: str,
    carrier_labels: tuple[str, ...],
    concentration_by_carrier: Mapping[str, float],
    charge_by_carrier: Mapping[str, int],
    radius_by_carrier: Mapping[str, float],
) -> float:
    source_charge = charge_by_carrier[carrier_label]
    weighted_radius_m = 0.0
    concentration_sum_mol_m3 = 0.0
    for other_label in carrier_labels:
        if source_charge * charge_by_carrier[other_label] >= 0:
            continue
        concentration_mol_m3 = concentration_by_carrier[other_label]
        weighted_radius_m += concentration_mol_m3 * radius_by_carrier[other_label]
        concentration_sum_mol_m3 += concentration_mol_m3
    _assert_positive_finite(concentration_sum_mol_m3, f"{carrier_label}.opposite_charge_concentration")
    return weighted_radius_m / concentration_sum_mol_m3


def _electrophoretic_drag_kg_s(
    viscosity_Pa_s: float,
    hydrodynamic_radius_m: float,
    kappa_m_inv: float,
) -> float:
    _assert_positive_finite(viscosity_Pa_s, "viscosity_Pa_s")
    _assert_positive_finite(hydrodynamic_radius_m, "hydrodynamic_radius_m")
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    kappa_radius = kappa_m_inv * hydrodynamic_radius_m
    if kappa_radius == 0.0:
        return 0.0
    return (
        STOKES_SPHERE_DRAG_FACTOR
        * math.pi
        * viscosity_Pa_s
        * hydrodynamic_radius_m
        * kappa_radius
        / (1.0 + kappa_radius)
    )


def _relaxation_drag_kg_s(
    charge_number: int,
    local_diffusivity_m2_s: float,
    hydrodynamic_radius_m: float,
    relative_dielectric: float,
    kappa_m_inv: float,
) -> float:
    _assert_positive_finite(local_diffusivity_m2_s, "local_diffusivity_m2_s")
    _assert_positive_finite(hydrodynamic_radius_m, "hydrodynamic_radius_m")
    _assert_positive_finite(relative_dielectric, "relative_dielectric")
    _assert_nonnegative_finite(kappa_m_inv, "kappa_m_inv")
    kappa_radius = kappa_m_inv * hydrodynamic_radius_m
    if kappa_radius == 0.0:
        return 0.0
    elementary_charge_C = F / N_A
    return (
        charge_number
        * charge_number
        * elementary_charge_C
        * elementary_charge_C
        * kappa_m_inv
        / (
            STOKES_SPHERE_DRAG_FACTOR
            * math.pi
            * EPS_0
            * relative_dielectric
            * local_diffusivity_m2_s
            * (1.0 + kappa_radius)
        )
    )


def _debye_kappa_inv_m(
    charge_weighted_concentration_mol_m3: float,
    relative_dielectric: float,
    temperature_K: float,
) -> float:
    _assert_nonnegative_finite(
        charge_weighted_concentration_mol_m3,
        "charge_weighted_concentration_mol_m3",
    )
    if charge_weighted_concentration_mol_m3 == 0.0:
        return math.inf
    kappa_squared_m_inv2 = (
        F
        * F
        * charge_weighted_concentration_mol_m3
        / (EPS_0 * relative_dielectric * R * temperature_K)
    )
    _assert_positive_finite(kappa_squared_m_inv2, "kappa_squared_m_inv2")
    return 1.0 / math.sqrt(kappa_squared_m_inv2)


def _sphere_volume_m3(radius_m: float) -> float:
    _assert_positive_finite(radius_m, "sphere_radius_m")
    return (
        SPHERE_VOLUME_NUMERATOR
        / SPHERE_VOLUME_DENOMINATOR
        * math.pi
        * radius_m
        * radius_m
        * radius_m
    )


def _validate_solver(solver: str) -> None:
    if solver not in SUPPORTED_ION_ATMOSPHERE_SOLVERS:
        raise ValueError(
            "Unsupported ion-atmosphere solver "
            f"{solver!r}; supported solvers are {SUPPORTED_ION_ATMOSPHERE_SOLVERS}"
        )


def _validate_bulk_solver(solver: str) -> None:
    if solver not in SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS:
        raise ValueError(
            "Unsupported bulk ion-atmosphere solver "
            f"{solver!r}; supported solvers are {SUPPORTED_BULK_ION_ATMOSPHERE_SOLVERS}"
        )


def _validate_bulk_resistance_matrix(
    resistance_matrix_kg_s: np.ndarray,
    context: str,
) -> None:
    if resistance_matrix_kg_s.ndim != 2:
        raise ValueError(f"{context} must be a matrix")
    if resistance_matrix_kg_s.shape[0] != resistance_matrix_kg_s.shape[1]:
        raise ValueError(f"{context} must be square")
    if not np.all(np.isfinite(resistance_matrix_kg_s)):
        raise ValueError(f"{context} contains non-finite values")
    if not np.allclose(resistance_matrix_kg_s, resistance_matrix_kg_s.T):
        raise ValueError(f"{context} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(resistance_matrix_kg_s)
    if float(np.min(eigenvalues)) < 0.0:
        raise ValueError(f"{context} must be positive semidefinite")


def _require_charge(
    values: Mapping[str, int],
    key: str,
) -> int:
    if key not in values:
        raise KeyError(f"carrier_charges missing {key}")
    charge_number = int(values[key])
    if charge_number == 0:
        raise ValueError(f"carrier_charges.{key} must be nonzero")
    return charge_number


def _require_positive_finite(
    values: Mapping[str, float],
    key: str,
    context: str,
) -> float:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    value = float(values[key])
    _assert_positive_finite(value, f"{context}.{key}")
    return value


def _require_nonnegative_finite(
    values: Mapping[str, float],
    key: str,
    context: str,
) -> float:
    if key not in values:
        raise KeyError(f"{context} missing {key}")
    value = float(values[key])
    _assert_nonnegative_finite(value, f"{context}.{key}")
    return value


def _assert_positive_finite(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{context} must be a positive finite number, got {value}")


def _assert_nonnegative_finite(value: float, context: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{context} must be a non-negative finite number, got {value}")
