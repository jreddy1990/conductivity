"""Inertial phase-space conductivity from one transferable energy model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import K_B, S_M_TO_MS_CM
from electrolyte_model import ElectrolyteRecipeModel
from utils.strict_validation import read_json_object, write_json_object
from utils.time_series_statistics import linear_fit, select_stationary_suffix

Array = np.ndarray
CARTESIAN_DIMENSION = 3
HALF = 0.5  # Analytical half-step used by Verlet and BAOAB splitting.


@dataclass(frozen=True)
class MicroscopicConfiguration:
    positions_m: Array
    velocities_m_s: Array
    masses_kg: Array
    partial_charges_C: Array
    molecule_index: Array
    atom_type_index: Array
    box_vectors_m: Array


@dataclass(frozen=True)
class DynamicsSettings:
    timestep_s: float
    equilibration_steps: int
    production_steps: int
    sample_stride_steps: int
    langevin_friction_per_s: float
    force_difference_step_m: float
    force_consistency_relative_tolerance: float


@dataclass(frozen=True)
class NumericalSettings:
    minimum_stationary_effective_sample_size: float
    stationarity_standard_error_limit: float
    singular_value_relative_tolerance: float
    residual_tolerance: float
    conductivity_tolerance_S_m: float
    maximum_basis_size: int
    radial_basis_count: int
    radial_cutoff_m: float
    fourier_shell_count: int
    configuration_directional_derivative_step_s: float
    charge_consistency_tolerance_C: float
    gk_maximum_lag_fraction: float
    gk_plateau_fraction: float
    gk_plateau_relative_tolerance: float


@dataclass(frozen=True)
class ConductivityResult:
    conductivity_S_m: float
    direct_current_term_S_m: float
    projected_correction_S_m: float
    green_kubo_conductivity_S_m: float
    basis_size: int
    basis_conductivities_S_m: tuple[float, ...]
    residual_history: tuple[float, ...]
    maximum_residual_score: float
    sample_count: int
    effective_sample_size: float


@dataclass(frozen=True)
class PhaseSpaceTrajectory:
    positions_m: Array
    velocities_m_s: Array
    forces_N: Array
    potential_energies_J: Array
    partial_charges_C: Array
    box_vectors_m: Array
    masses_kg: Array
    molecule_index: Array
    sample_interval_s: float


@runtime_checkable
class InteratomicModel(Protocol):
    def initial_configuration(
        self,
        recipe: ElectrolyteRecipeModel,
        density_kg_m3: float,
        molecule_count: int,
        random_generator: np.random.Generator,
    ) -> MicroscopicConfiguration: ...

    def energy_J(self, positions_m: Array, box_vectors_m: Array) -> float: ...

    def forces_N(self, positions_m: Array, box_vectors_m: Array) -> Array: ...

    def virial_J(self, positions_m: Array, box_vectors_m: Array) -> Array: ...

    def partial_charges_C(
        self, positions_m: Array, box_vectors_m: Array
    ) -> Array: ...


class TorchScriptInteratomicModel:
    """TorchScript energy/charge model with an explicit atomistic topology."""

    def __init__(
        self,
        model_path: Path,
        configuration_path: Path,
        recipe: ElectrolyteRecipeModel,
        charge_consistency_tolerance_C: float,
    ) -> None:
        import torch

        self._torch = torch
        self._model = torch.jit.load(str(model_path), map_location="cpu")
        self._model.eval()
        configuration = np.load(configuration_path, allow_pickle=False)
        self._positions_m = np.asarray(configuration["positions_m"], dtype=float)
        self._masses_kg = np.asarray(configuration["masses_kg"], dtype=float)
        self._molecule_index = np.asarray(configuration["molecule_index"], dtype=int)
        self._atom_type_index = np.asarray(configuration["atom_type_index"], dtype=int)
        self._partial_charges_C = np.asarray(
            configuration["partial_charges_C"], dtype=float
        )
        self._box_vectors_m = np.asarray(configuration["box_vectors_m"], dtype=float)
        self._recipe = recipe
        self._charge_consistency_tolerance_C = charge_consistency_tolerance_C
        self._validate_topology()

    def initial_configuration(
        self,
        recipe: ElectrolyteRecipeModel,
        density_kg_m3: float,
        molecule_count: int,
        random_generator: np.random.Generator,
    ) -> MicroscopicConfiguration:
        del random_generator
        if recipe.model_dump(mode="python") != self._recipe.model_dump(mode="python"):
            raise ValueError("atomistic topology composition differs from recipe")
        volume_m3 = float(abs(np.linalg.det(self._box_vectors_m)))
        observed_density_kg_m3 = float(np.sum(self._masses_kg) / volume_m3)
        relative_error = abs(observed_density_kg_m3 - density_kg_m3) / density_kg_m3
        if relative_error > np.sqrt(np.finfo(float).eps):
            raise ValueError("requested density differs from topology density")
        if int(np.unique(self._molecule_index).size) != molecule_count:
            raise ValueError("requested molecule count differs from topology")
        positions_m = wrap_positions(self._positions_m, self._box_vectors_m)
        return MicroscopicConfiguration(
            positions_m=positions_m,
            velocities_m_s=np.zeros_like(positions_m),
            masses_kg=self._masses_kg.copy(),
            partial_charges_C=self.partial_charges_C(
                positions_m, self._box_vectors_m
            ),
            molecule_index=self._molecule_index.copy(),
            atom_type_index=self._atom_type_index.copy(),
            box_vectors_m=self._box_vectors_m.copy(),
        )

    def energy_J(self, positions_m: Array, box_vectors_m: Array) -> float:
        energy, _charges = self._energy_and_charges(positions_m, box_vectors_m)
        return float(energy.detach().cpu().numpy())

    def forces_N(self, positions_m: Array, box_vectors_m: Array) -> Array:
        positions = self._torch.tensor(
            np.asarray(positions_m), dtype=self._torch.float64, requires_grad=True
        )
        box = self._torch.tensor(np.asarray(box_vectors_m), dtype=self._torch.float64)
        atom_types = self._torch.tensor(self._atom_type_index, dtype=self._torch.int64)
        energy, _charges = self._model(positions, box, atom_types)
        gradient = self._torch.autograd.grad(energy, positions)[0]
        return -np.asarray(gradient.detach().cpu().numpy(), dtype=float)

    def virial_J(self, positions_m: Array, box_vectors_m: Array) -> Array:
        forces_N = self.forces_N(positions_m, box_vectors_m)
        centered_positions_m = positions_m - np.mean(positions_m, axis=0)
        return -np.einsum("ia,ib->ab", centered_positions_m, forces_N)

    def partial_charges_C(self, positions_m: Array, box_vectors_m: Array) -> Array:
        _energy, charges = self._energy_and_charges(positions_m, box_vectors_m)
        model_charges_C = np.asarray(charges.detach().cpu().numpy(), dtype=float)
        if not np.allclose(
            model_charges_C,
            self._partial_charges_C,
            rtol=0.0,
            atol=self._charge_consistency_tolerance_C,
        ):
            raise ValueError(
                "position-dependent charges require an explicit charge-flux current"
            )
        return self._partial_charges_C.copy()

    def _energy_and_charges(self, positions_m: Array, box_vectors_m: Array):
        positions = self._torch.tensor(np.asarray(positions_m), dtype=self._torch.float64)
        box = self._torch.tensor(np.asarray(box_vectors_m), dtype=self._torch.float64)
        atom_types = self._torch.tensor(self._atom_type_index, dtype=self._torch.int64)
        energy, charges = self._model(positions, box, atom_types)
        if energy.ndim != 0 or charges.shape != positions.shape[:1]:
            raise ValueError("model must return scalar energy and per-atom charges")
        return energy, charges

    def _validate_topology(self) -> None:
        atom_count = self._positions_m.shape[0]
        if self._positions_m.shape != (atom_count, CARTESIAN_DIMENSION):
            raise ValueError("positions_m must have shape (n_atoms, 3)")
        if self._masses_kg.shape != (atom_count,) or np.any(self._masses_kg <= 0.0):
            raise ValueError("masses_kg must be positive per atom")
        if self._molecule_index.shape != (atom_count,):
            raise ValueError("molecule_index must be defined per atom")
        unique_molecule_indices = np.unique(self._molecule_index)
        expected_molecule_indices = np.arange(unique_molecule_indices.size)
        if not np.array_equal(unique_molecule_indices, expected_molecule_indices):
            raise ValueError("molecule_index must be contiguous and start at zero")
        if self._atom_type_index.shape != (atom_count,):
            raise ValueError("atom_type_index must be defined per atom")
        if self._partial_charges_C.shape != (atom_count,):
            raise ValueError("partial_charges_C must be defined per atom")
        if self._box_vectors_m.shape != (CARTESIAN_DIMENSION, CARTESIAN_DIMENSION):
            raise ValueError("box_vectors_m must have shape (3, 3)")
        if abs(np.linalg.det(self._box_vectors_m)) <= 0.0:
            raise ValueError("periodic box volume must be positive")


def wrap_positions(positions_m: Array, box_vectors_m: Array) -> Array:
    original_shape = np.asarray(positions_m).shape
    vectors = np.asarray(positions_m).reshape(-1, CARTESIAN_DIMENSION)
    fractional = np.linalg.solve(box_vectors_m.T, vectors.T).T
    wrapped = (fractional - np.floor(fractional)) @ box_vectors_m
    return wrapped.reshape(original_shape)


def minimum_image_displacement(displacement_m: Array, box_vectors_m: Array) -> Array:
    original_shape = np.asarray(displacement_m).shape
    vectors = np.asarray(displacement_m).reshape(-1, CARTESIAN_DIMENSION)
    fractional = np.linalg.solve(box_vectors_m.T, vectors.T).T
    return ((fractional - np.rint(fractional)) @ box_vectors_m).reshape(original_shape)


def remove_center_of_mass_momentum(velocities_m_s: Array, masses_kg: Array) -> Array:
    center_velocity_m_s = np.einsum("i,ia->a", masses_kg, velocities_m_s) / np.sum(
        masses_kg
    )
    return velocities_m_s - center_velocity_m_s


def maxwell_boltzmann_velocities(
    masses_kg: Array,
    temperature_K: float,
    random_generator: np.random.Generator,
) -> Array:
    scale_m_s = np.sqrt(K_B * temperature_K / masses_kg)
    velocities = random_generator.normal(
        size=(masses_kg.size, CARTESIAN_DIMENSION)
    ) * scale_m_s[:, None]
    return remove_center_of_mass_momentum(velocities, masses_kg)


def velocity_verlet_step(
    configuration: MicroscopicConfiguration,
    model: InteratomicModel,
    timestep_s: float,
) -> MicroscopicConfiguration:
    forces_N = model.forces_N(configuration.positions_m, configuration.box_vectors_m)
    half_velocity = configuration.velocities_m_s + (
        HALF * timestep_s * forces_N / configuration.masses_kg[:, None]
    )
    positions_m = wrap_positions(
        configuration.positions_m + timestep_s * half_velocity,
        configuration.box_vectors_m,
    )
    new_forces_N = model.forces_N(positions_m, configuration.box_vectors_m)
    velocities_m_s = half_velocity + (
        HALF * timestep_s * new_forces_N / configuration.masses_kg[:, None]
    )
    return MicroscopicConfiguration(
        positions_m=positions_m,
        velocities_m_s=velocities_m_s,
        masses_kg=configuration.masses_kg,
        partial_charges_C=model.partial_charges_C(
            positions_m, configuration.box_vectors_m
        ),
        molecule_index=configuration.molecule_index,
        atom_type_index=configuration.atom_type_index,
        box_vectors_m=configuration.box_vectors_m,
    )


def langevin_baoab_step(
    configuration: MicroscopicConfiguration,
    model: InteratomicModel,
    temperature_K: float,
    timestep_s: float,
    friction_per_s: float,
    random_generator: np.random.Generator,
) -> MicroscopicConfiguration:
    forces_N = model.forces_N(configuration.positions_m, configuration.box_vectors_m)
    velocities = configuration.velocities_m_s + (
        HALF * timestep_s * forces_N / configuration.masses_kg[:, None]
    )
    positions = wrap_positions(
        configuration.positions_m + HALF * timestep_s * velocities,
        configuration.box_vectors_m,
    )
    damping = np.exp(-friction_per_s * timestep_s)
    thermal_scale = np.sqrt(
        (1.0 - damping * damping)
        * K_B
        * temperature_K
        / configuration.masses_kg
    )
    velocities = damping * velocities + thermal_scale[:, None] * random_generator.normal(
        size=velocities.shape
    )
    positions = wrap_positions(
        positions + HALF * timestep_s * velocities, configuration.box_vectors_m
    )
    new_forces_N = model.forces_N(positions, configuration.box_vectors_m)
    velocities += HALF * timestep_s * new_forces_N / configuration.masses_kg[:, None]
    return MicroscopicConfiguration(
        positions_m=positions,
        velocities_m_s=velocities,
        masses_kg=configuration.masses_kg,
        partial_charges_C=model.partial_charges_C(
            positions, configuration.box_vectors_m
        ),
        molecule_index=configuration.molecule_index,
        atom_type_index=configuration.atom_type_index,
        box_vectors_m=configuration.box_vectors_m,
    )


def equilibrate_configuration(
    configuration: MicroscopicConfiguration,
    model: InteratomicModel,
    temperature_K: float,
    settings: DynamicsSettings,
    random_generator: np.random.Generator,
) -> MicroscopicConfiguration:
    state = configuration
    for _step_index in range(settings.equilibration_steps):
        state = langevin_baoab_step(
            state,
            model,
            temperature_K,
            settings.timestep_s,
            settings.langevin_friction_per_s,
            random_generator,
        )
    return state


def sample_equilibrium_trajectory(
    configuration: MicroscopicConfiguration,
    model: InteratomicModel,
    temperature_K: float,
    settings: DynamicsSettings,
    random_generator: np.random.Generator,
) -> PhaseSpaceTrajectory:
    positions: list[Array] = []
    velocities: list[Array] = []
    forces: list[Array] = []
    energies: list[float] = []
    charges: list[Array] = []
    state = configuration
    for step_index in range(settings.production_steps):
        state = langevin_baoab_step(
            state,
            model,
            temperature_K,
            settings.timestep_s,
            settings.langevin_friction_per_s,
            random_generator,
        )
        if (step_index + 1) % settings.sample_stride_steps == 0:
            positions.append(state.positions_m.copy())
            velocities.append(state.velocities_m_s.copy())
            forces.append(model.forces_N(state.positions_m, state.box_vectors_m))
            energies.append(model.energy_J(state.positions_m, state.box_vectors_m))
            charges.append(state.partial_charges_C.copy())
    if len(positions) < CARTESIAN_DIMENSION:
        raise ValueError("production trajectory has too few sampled phase-space points")
    return PhaseSpaceTrajectory(
        positions_m=np.asarray(positions),
        velocities_m_s=np.asarray(velocities),
        forces_N=np.asarray(forces),
        potential_energies_J=np.asarray(energies),
        partial_charges_C=np.asarray(charges),
        box_vectors_m=state.box_vectors_m,
        masses_kg=state.masses_kg,
        molecule_index=state.molecule_index,
        sample_interval_s=settings.timestep_s * settings.sample_stride_steps,
    )


def molecular_center_of_mass(
    positions_m: Array, masses_kg: Array, molecule_index: Array
) -> Array:
    molecule_count = int(np.max(molecule_index)) + 1
    centers_m = np.zeros((molecule_count, CARTESIAN_DIMENSION))
    for molecule in range(molecule_count):
        atom_mask = molecule_index == molecule
        centers_m[molecule] = np.einsum(
            "i,ia->a", masses_kg[atom_mask], positions_m[atom_mask]
        ) / np.sum(masses_kg[atom_mask])
    return centers_m


def total_charge_current_density_A_m2(
    velocities_m_s: Array, partial_charges_C: Array, volume_m3: float
) -> Array:
    return np.einsum("i,ia->a", partial_charges_C, velocities_m_s) / volume_m3


def molecular_com_current_density_A_m2(
    velocities_m_s: Array,
    masses_kg: Array,
    partial_charges_C: Array,
    molecule_index: Array,
    volume_m3: float,
) -> Array:
    molecule_count = int(np.max(molecule_index)) + 1
    current_A_m2 = np.zeros(CARTESIAN_DIMENSION)
    for molecule in range(molecule_count):
        atom_mask = molecule_index == molecule
        velocity_m_s = np.einsum(
            "i,ia->a", masses_kg[atom_mask], velocities_m_s[atom_mask]
        ) / np.sum(masses_kg[atom_mask])
        current_A_m2 += np.sum(partial_charges_C[atom_mask]) * velocity_m_s
    return current_A_m2 / volume_m3


def internal_polarization_current_density_A_m2(
    velocities_m_s: Array,
    masses_kg: Array,
    partial_charges_C: Array,
    molecule_index: Array,
    volume_m3: float,
) -> Array:
    molecule_count = int(np.max(molecule_index)) + 1
    internal_current_A_m2 = np.zeros(CARTESIAN_DIMENSION)
    for molecule in range(molecule_count):
        atom_mask = molecule_index == molecule
        molecular_velocity_m_s = np.einsum(
            "i,ia->a", masses_kg[atom_mask], velocities_m_s[atom_mask]
        ) / np.sum(masses_kg[atom_mask])
        relative_velocities_m_s = (
            velocities_m_s[atom_mask] - molecular_velocity_m_s
        )
        internal_current_A_m2 += np.einsum(
            "i,ia->a", partial_charges_C[atom_mask], relative_velocities_m_s
        )
    return internal_current_A_m2 / volume_m3


def current_series_A_m2(trajectory: PhaseSpaceTrajectory) -> tuple[Array, Array]:
    volume_m3 = float(abs(np.linalg.det(trajectory.box_vectors_m)))
    total_currents: list[Array] = []
    molecular_currents: list[Array] = []
    for velocities, charges in zip(
        trajectory.velocities_m_s, trajectory.partial_charges_C, strict=True
    ):
        total = total_charge_current_density_A_m2(velocities, charges, volume_m3)
        molecular = molecular_com_current_density_A_m2(
            velocities,
            trajectory.masses_kg,
            charges,
            trajectory.molecule_index,
            volume_m3,
        )
        internal = internal_polarization_current_density_A_m2(
            velocities,
            trajectory.masses_kg,
            charges,
            trajectory.molecule_index,
            volume_m3,
        )
        if not np.allclose(total, molecular + internal):
            raise RuntimeError("molecular current decomposition failed")
        total_currents.append(total)
        molecular_currents.append(molecular)
    return np.asarray(total_currents), np.asarray(molecular_currents)


def charge_acceleration_series_A_m2_s(
    trajectory: PhaseSpaceTrajectory,
) -> tuple[Array, Array]:
    volume_m3 = float(abs(np.linalg.det(trajectory.box_vectors_m)))
    atomic_acceleration = np.einsum(
        "ti,tia,i->ta",
        trajectory.partial_charges_C,
        trajectory.forces_N,
        1.0 / trajectory.masses_kg,
    ) / volume_m3
    molecular_acceleration = np.zeros_like(atomic_acceleration)
    molecule_count = int(np.max(trajectory.molecule_index)) + 1
    for molecule in range(molecule_count):
        atom_mask = trajectory.molecule_index == molecule
        molecular_charge_C = trajectory.partial_charges_C[:, atom_mask].sum(axis=1)
        molecular_force_N = trajectory.forces_N[:, atom_mask].sum(axis=1)
        molecular_mass_kg = float(np.sum(trajectory.masses_kg[atom_mask]))
        molecular_acceleration += (
            molecular_charge_C[:, None]
            * molecular_force_N
            / (molecular_mass_kg * volume_m3)
        )
    return atomic_acceleration, molecular_acceleration


def validate_force_consistency(
    configuration: MicroscopicConfiguration,
    model: InteratomicModel,
    settings: DynamicsSettings,
    random_generator: np.random.Generator,
) -> None:
    direction = random_generator.normal(size=configuration.positions_m.shape)
    direction /= np.linalg.norm(direction)
    displacement = settings.force_difference_step_m * direction
    energy_plus = model.energy_J(
        wrap_positions(configuration.positions_m + displacement, configuration.box_vectors_m),
        configuration.box_vectors_m,
    )
    energy_minus = model.energy_J(
        wrap_positions(configuration.positions_m - displacement, configuration.box_vectors_m),
        configuration.box_vectors_m,
    )
    numerical_derivative = (energy_plus - energy_minus) / (
        2.0 * settings.force_difference_step_m
    )
    force_derivative = -float(
        np.sum(
            model.forces_N(configuration.positions_m, configuration.box_vectors_m)
            * direction
        )
    )
    comparison_scale = max(
        abs(numerical_derivative), abs(force_derivative), np.finfo(float).tiny
    )
    relative_error = abs(numerical_derivative - force_derivative) / comparison_scale
    if relative_error > settings.force_consistency_relative_tolerance:
        raise ValueError("interatomic forces disagree with the energy gradient")


def _reciprocal_shells(box_vectors_m: Array, shell_count: int) -> tuple[Array, ...]:
    reciprocal_basis = 2.0 * np.pi * np.linalg.inv(box_vectors_m).T
    candidate_vectors: list[Array] = []
    for first_index in range(-shell_count, shell_count + 1):
        for second_index in range(-shell_count, shell_count + 1):
            for third_index in range(-shell_count, shell_count + 1):
                integer_vector = np.asarray(
                    [first_index, second_index, third_index], dtype=float
                )
                if np.all(integer_vector == 0.0):
                    continue
                candidate_vectors.append(integer_vector @ reciprocal_basis)
    candidate_vectors.sort(key=np.linalg.norm)
    shells: list[list[Array]] = []
    shell_norms: list[float] = []
    norm_tolerance = np.sqrt(np.finfo(float).eps)
    for reciprocal_vector in candidate_vectors:
        vector_norm = float(np.linalg.norm(reciprocal_vector))
        matching_shell = None
        for shell_index, shell_norm in enumerate(shell_norms):
            if abs(vector_norm - shell_norm) <= norm_tolerance * shell_norm:
                matching_shell = shell_index
                break
        if matching_shell is None:
            if len(shells) == shell_count:
                break
            shell_norms.append(vector_norm)
            shells.append([reciprocal_vector])
        else:
            shells[matching_shell].append(reciprocal_vector)
    if len(shells) != shell_count:
        raise ValueError("periodic box does not resolve the requested Fourier shells")
    return tuple(np.asarray(shell) for shell in shells)


def _scalar_features_for_positions(
    positions_m: Array,
    partial_charges_C: Array,
    box_vectors_m: Array,
    settings: NumericalSettings,
) -> Array:
    frame_count, atom_count, _axis_count = positions_m.shape
    volume_m3 = float(abs(np.linalg.det(box_vectors_m)))
    columns: list[Array] = [np.ones(frame_count)]
    radial_spacing_m = settings.radial_cutoff_m / (settings.radial_basis_count + 1)
    radial_centers_m = radial_spacing_m * np.arange(
        1, settings.radial_basis_count + 1
    )
    pair_mask = np.triu(np.ones((atom_count, atom_count), dtype=bool), k=1)
    for radial_center_m in radial_centers_m:
        values = np.zeros(frame_count)
        for frame_index in range(frame_count):
            displacement = (
                positions_m[frame_index, :, None, :]
                - positions_m[frame_index, None, :, :]
            )
            distance = np.linalg.norm(
                minimum_image_displacement(displacement, box_vectors_m),
                axis=-1,
            )
            charge_product = (
                partial_charges_C[frame_index, :, None]
                * partial_charges_C[frame_index, None, :]
            )
            radial_weight = np.exp(
                -HALF * ((distance - radial_center_m) / radial_spacing_m) ** 2
            )
            values[frame_index] = np.sum(
                charge_product[pair_mask] * radial_weight[pair_mask]
            ) / volume_m3
        columns.append(values)
    for reciprocal_vectors in _reciprocal_shells(
        box_vectors_m, settings.fourier_shell_count
    ):
        shell_density = np.zeros(frame_count)
        for reciprocal_vector in reciprocal_vectors:
            phase = np.einsum("tia,a->ti", positions_m, reciprocal_vector)
            real_density = np.sum(
                partial_charges_C * np.cos(phase), axis=1
            )
            imaginary_density = np.sum(
                partial_charges_C * np.sin(phase), axis=1
            )
            shell_density += real_density**2 + imaginary_density**2
        columns.append(shell_density / (reciprocal_vectors.shape[0] * volume_m3))
    return np.column_stack(columns)


def _even_fourier_modes_and_generator(
    trajectory: PhaseSpaceTrajectory, settings: NumericalSettings
) -> tuple[Array, Array]:
    frame_count = trajectory.positions_m.shape[0]
    modes: list[Array] = []
    generator_modes: list[Array] = []
    for reciprocal_vectors in _reciprocal_shells(
        trajectory.box_vectors_m, settings.fourier_shell_count
    ):
        shell_mode = np.zeros((frame_count, CARTESIAN_DIMENSION))
        shell_generator = np.zeros_like(shell_mode)
        for reciprocal_vector in reciprocal_vectors:
            wavevector_norm = np.linalg.norm(reciprocal_vector)
            unit_wavevector = reciprocal_vector / wavevector_norm
            phase = np.einsum(
                "tia,a->ti", trajectory.positions_m, reciprocal_vector
            )
            charge_sine = np.sum(
                trajectory.partial_charges_C * np.sin(phase), axis=1
            )
            phase_velocity = np.einsum(
                "tia,a->ti", trajectory.velocities_m_s, reciprocal_vector
            )
            charge_cosine_velocity = np.sum(
                trajectory.partial_charges_C * np.cos(phase) * phase_velocity,
                axis=1,
            )
            shell_mode += charge_sine[:, None] * unit_wavevector
            shell_generator += charge_cosine_velocity[:, None] * unit_wavevector
        modes.append(shell_mode / reciprocal_vectors.shape[0])
        generator_modes.append(shell_generator / reciprocal_vectors.shape[0])
    return np.stack(modes, axis=1), np.stack(generator_modes, axis=1)


def _basis_and_generator(
    trajectory: PhaseSpaceTrajectory,
    current_A_m2: Array,
    charge_acceleration_A_m2_s: Array,
    settings: NumericalSettings,
    friction_per_s: float,
) -> tuple[Array, Array]:
    features = _scalar_features_for_positions(
        trajectory.positions_m,
        trajectory.partial_charges_C,
        trajectory.box_vectors_m,
        settings,
    )
    feature_scales = np.std(features, axis=0)
    feature_scales[feature_scales <= np.finfo(float).eps] = 1.0
    features = (features - np.mean(features, axis=0)) / feature_scales
    features[:, 0] = 1.0
    displaced_positions_plus = wrap_positions(
        trajectory.positions_m
        + settings.configuration_directional_derivative_step_s
        * trajectory.velocities_m_s,
        trajectory.box_vectors_m,
    )
    displaced_positions_minus = wrap_positions(
        trajectory.positions_m
        - settings.configuration_directional_derivative_step_s
        * trajectory.velocities_m_s,
        trajectory.box_vectors_m,
    )
    feature_derivative = (
        _scalar_features_for_positions(
            displaced_positions_plus,
            trajectory.partial_charges_C,
            trajectory.box_vectors_m,
            settings,
        )
        - _scalar_features_for_positions(
            displaced_positions_minus,
            trajectory.partial_charges_C,
            trajectory.box_vectors_m,
            settings,
        )
    ) / (
        2.0
        * settings.configuration_directional_derivative_step_s
        * feature_scales
    )
    odd_basis = features[:, :, None] * current_A_m2[:, None, :]
    odd_generator = (
        feature_derivative[:, :, None] * current_A_m2[:, None, :]
        + features[:, :, None]
        * (charge_acceleration_A_m2_s - friction_per_s * current_A_m2)[:, None, :]
    )
    even_basis, even_generator = _even_fourier_modes_and_generator(
        trajectory, settings
    )
    basis = np.concatenate((odd_basis, even_basis), axis=1)
    generator_basis = np.concatenate((odd_generator, even_generator), axis=1)
    basis_scale = np.sqrt(np.mean(basis**2, axis=(0, 2)))
    retained = basis_scale > np.finfo(float).eps
    if not np.any(retained):
        raise ValueError("phase-space basis has no fluctuating modes")
    return (
        basis[:, retained] / basis_scale[retained][None, :, None],
        generator_basis[:, retained] / basis_scale[retained][None, :, None],
    )


def evaluate_basis(
    trajectory: PhaseSpaceTrajectory,
    current_A_m2: Array,
    settings: NumericalSettings,
) -> Array:
    charge_acceleration, _molecular_acceleration = charge_acceleration_series_A_m2_s(
        trajectory
    )
    basis, _generator_basis = _basis_and_generator(
        trajectory,
        current_A_m2,
        charge_acceleration,
        settings,
        friction_per_s=0.0,
    )
    return basis


def apply_generator_to_basis(
    trajectory: PhaseSpaceTrajectory,
    current_A_m2: Array,
    charge_acceleration_A_m2_s: Array,
    settings: NumericalSettings,
    friction_per_s: float,
) -> Array:
    _basis, generator_basis = _basis_and_generator(
        trajectory,
        current_A_m2,
        charge_acceleration_A_m2_s,
        settings,
        friction_per_s,
    )
    return generator_basis


def dirichlet_matrix(basis: Array, generator_basis: Array) -> Array:
    return -np.einsum("tma,tna->amn", basis, generator_basis) / basis.shape[0]


def current_coupling_matrix(basis: Array, current_A_m2: Array) -> Array:
    return np.einsum("tma,ta->ma", basis, current_A_m2) / basis.shape[0]


def solve_projected_poisson(
    generator_matrix: Array,
    current_coupling: Array,
    relative_tolerance: float,
) -> Array:
    if generator_matrix.ndim == 3:
        coefficients = np.zeros_like(current_coupling)
        for axis_index in range(CARTESIAN_DIMENSION):
            coefficients[:, axis_index] = solve_projected_poisson(
                generator_matrix[axis_index],
                current_coupling[:, axis_index],
                relative_tolerance,
            )
        return coefficients
    if generator_matrix.ndim != 2:
        raise ValueError("generator matrix must have matrix or axis-matrix shape")
    left_vectors, singular_values, right_vectors_transpose = svd(
        generator_matrix, full_matrices=False
    )
    if singular_values[0] <= 0.0:
        raise ValueError("projected generator has no nonzero singular modes")
    retained = singular_values > relative_tolerance * singular_values[0]
    inverse_singular_values = np.zeros_like(singular_values)
    inverse_singular_values[retained] = 1.0 / singular_values[retained]
    pseudoinverse = (
        right_vectors_transpose.T * inverse_singular_values
    ) @ left_vectors.T
    return pseudoinverse @ current_coupling


def projected_conductivity(
    current_coupling: Array,
    coefficients: Array,
    volume_m3: float,
    temperature_K: float,
) -> float:
    return float(
        volume_m3
        * np.sum(current_coupling * coefficients)
        / (CARTESIAN_DIMENSION * K_B * temperature_K)
    )


def _candidate_residual_key(candidate: tuple[float, int, float]) -> float:
    return candidate[0]


def refine_projected_basis(
    basis: Array,
    generator_basis: Array,
    current_A_m2: Array,
    volume_m3: float,
    temperature_K: float,
    settings: NumericalSettings,
) -> tuple[float, float, tuple[float, ...], tuple[float, ...], int]:
    split_index = basis.shape[0] // 2
    if split_index < CARTESIAN_DIMENSION:
        raise ValueError("trajectory is too short for held-out basis refinement")
    candidate_count = min(basis.shape[1], settings.maximum_basis_size)
    selected: list[int] = []
    remaining = list(range(candidate_count))
    conductivities: list[float] = []
    residuals: list[float] = []
    heldout_current = current_A_m2[split_index:]
    current_scale = float(np.mean(heldout_current**2))
    if current_scale <= 0.0:
        raise ValueError("microscopic charge current has zero variance")
    while remaining:
        candidates: list[tuple[float, int, float]] = []
        for candidate_index in remaining:
            indices = (*selected, candidate_index)
            training_basis = basis[:split_index, indices]
            training_generator = generator_basis[:split_index, indices]
            coupling = current_coupling_matrix(
                training_basis, current_A_m2[:split_index]
            )
            coefficients = solve_projected_poisson(
                dirichlet_matrix(training_basis, training_generator),
                coupling,
                settings.singular_value_relative_tolerance,
            )
            heldout_residual = heldout_current + np.einsum(
                "tma,ma->ta", generator_basis[split_index:, indices], coefficients
            )
            residual_score = float(np.mean(heldout_residual**2) / current_scale)
            conductivity = projected_conductivity(
                coupling, coefficients, volume_m3, temperature_K
            )
            candidates.append((residual_score, candidate_index, conductivity))
        residual_score, selected_index, conductivity = min(
            candidates, key=_candidate_residual_key
        )
        selected.append(selected_index)
        remaining.remove(selected_index)
        residuals.append(residual_score)
        conductivities.append(conductivity)
        conductivity_converged = False
        if len(conductivities) > 1:
            conductivity_converged = (
                abs(conductivities[-1] - conductivities[-2])
                <= settings.conductivity_tolerance_S_m
            )
        if residual_score <= settings.residual_tolerance and conductivity_converged:
            break
    if not residuals or residuals[-1] > settings.residual_tolerance:
        raise ValueError("phase-space basis did not reach the residual tolerance")
    return (
        conductivities[-1],
        conductivities[0],
        tuple(conductivities),
        tuple(residuals),
        len(selected),
    )


def current_autocorrelation_fft(current_A_m2: Array) -> Array:
    centered = current_A_m2 - np.mean(current_A_m2, axis=0)
    sample_count = centered.shape[0]
    transform_length = 2 * sample_count
    spectrum = np.fft.rfft(centered, n=transform_length, axis=0)
    correlation = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=transform_length, axis=0
    )[:sample_count]
    unbiased_count = np.arange(sample_count, 0, -1)[:, None]
    return np.sum(correlation / unbiased_count, axis=1)


def integrated_green_kubo_conductivity(
    current_A_m2: Array,
    sample_interval_s: float,
    volume_m3: float,
    temperature_K: float,
    settings: NumericalSettings,
) -> float:
    autocorrelation = current_autocorrelation_fft(current_A_m2)
    maximum_lag = int(settings.gk_maximum_lag_fraction * autocorrelation.size)
    if maximum_lag < CARTESIAN_DIMENSION:
        raise ValueError("trajectory is too short for Green-Kubo integration")
    integral = cumulative_trapezoid(
        autocorrelation[:maximum_lag], dx=sample_interval_s, initial=0.0
    )
    plateau_count = max(
        CARTESIAN_DIMENSION, int(settings.gk_plateau_fraction * maximum_lag)
    )
    plateau = integral[-plateau_count:]
    plateau_times_s = sample_interval_s * np.arange(plateau_count)
    fit = linear_fit(plateau_times_s, plateau)
    plateau_scale = max(abs(float(np.mean(plateau))), np.finfo(float).tiny)
    relative_drift = abs(fit.slope) * plateau_times_s[-1] / plateau_scale
    if relative_drift > settings.gk_plateau_relative_tolerance:
        raise ValueError("Green-Kubo integral has no resolved plateau")
    return float(
        volume_m3
        * np.mean(plateau)
        / (CARTESIAN_DIMENSION * K_B * temperature_K)
    )


def _stationary_trajectory(
    trajectory: PhaseSpaceTrajectory,
    current_A_m2: Array,
    settings: NumericalSettings,
) -> tuple[PhaseSpaceTrajectory, Array, float]:
    stationary = select_stationary_suffix(
        values=np.linalg.norm(current_A_m2, axis=1),
        maximum_split_mean_difference_standard_errors=(
            settings.stationarity_standard_error_limit
        ),
        maximum_linear_drift_standard_errors=(
            settings.stationarity_standard_error_limit
        ),
        minimum_effective_sample_size=(
            settings.minimum_stationary_effective_sample_size
        ),
    )
    start_index = stationary.start_index
    stationary_trajectory = PhaseSpaceTrajectory(
        positions_m=trajectory.positions_m[start_index:],
        velocities_m_s=trajectory.velocities_m_s[start_index:],
        forces_N=trajectory.forces_N[start_index:],
        potential_energies_J=trajectory.potential_energies_J[start_index:],
        partial_charges_C=trajectory.partial_charges_C[start_index:],
        box_vectors_m=trajectory.box_vectors_m,
        masses_kg=trajectory.masses_kg,
        molecule_index=trajectory.molecule_index,
        sample_interval_s=trajectory.sample_interval_s,
    )
    return (
        stationary_trajectory,
        current_A_m2[start_index:],
        stationary.autocorrelation.effective_sample_size,
    )


def _validate_settings(
    dynamics: DynamicsSettings, numerics: NumericalSettings
) -> None:
    positive_dynamics = (
        dynamics.timestep_s,
        dynamics.langevin_friction_per_s,
        dynamics.force_difference_step_m,
        dynamics.force_consistency_relative_tolerance,
    )
    if any(value <= 0.0 for value in positive_dynamics):
        raise ValueError("all dimensional dynamics settings must be positive")
    positive_counts = (
        dynamics.equilibration_steps,
        dynamics.production_steps,
        dynamics.sample_stride_steps,
        numerics.maximum_basis_size,
        numerics.radial_basis_count,
        numerics.fourier_shell_count,
    )
    if any(value <= 0 for value in positive_counts):
        raise ValueError("all dynamics and basis counts must be positive")
    positive_numerics = (
        numerics.minimum_stationary_effective_sample_size,
        numerics.stationarity_standard_error_limit,
        numerics.singular_value_relative_tolerance,
        numerics.residual_tolerance,
        numerics.conductivity_tolerance_S_m,
        numerics.radial_cutoff_m,
        numerics.configuration_directional_derivative_step_s,
        numerics.charge_consistency_tolerance_C,
        numerics.gk_maximum_lag_fraction,
        numerics.gk_plateau_fraction,
        numerics.gk_plateau_relative_tolerance,
    )
    if any(value <= 0.0 for value in positive_numerics):
        raise ValueError("all numerical tolerances and scales must be positive")
    unit_interval_values = (
        numerics.gk_maximum_lag_fraction,
        numerics.gk_plateau_fraction,
    )
    if any(value >= 1.0 for value in unit_interval_values):
        raise ValueError("Green-Kubo lag and plateau fractions must be below one")


def _validate_microscopic_configuration(
    configuration: MicroscopicConfiguration,
    expected_molecule_count: int,
    charge_neutrality_tolerance_C: float,
) -> None:
    atom_count = configuration.positions_m.shape[0]
    if configuration.positions_m.shape != (atom_count, CARTESIAN_DIMENSION):
        raise ValueError("positions_m must have shape (n_atoms, 3)")
    if configuration.velocities_m_s.shape != configuration.positions_m.shape:
        raise ValueError("velocities_m_s must align with positions_m")
    atom_vectors = (
        configuration.masses_kg,
        configuration.partial_charges_C,
        configuration.molecule_index,
        configuration.atom_type_index,
    )
    if any(vector.shape != (atom_count,) for vector in atom_vectors):
        raise ValueError("all microscopic atom properties must align")
    if np.any(configuration.masses_kg <= 0.0):
        raise ValueError("all atomic masses must be positive")
    molecule_indices = np.unique(configuration.molecule_index)
    if not np.array_equal(molecule_indices, np.arange(molecule_indices.size)):
        raise ValueError("molecule indices must be contiguous and start at zero")
    if molecule_indices.size != expected_molecule_count:
        raise ValueError(
            "interatomic model returned "
            f"{molecule_indices.size} molecules; expected {expected_molecule_count}"
        )
    charge_scale_C = float(np.sum(np.abs(configuration.partial_charges_C)))
    charge_residual_C = abs(float(np.sum(configuration.partial_charges_C)))
    if charge_scale_C <= 0.0:
        raise ValueError("microscopic topology must contain charged atoms")
    if charge_residual_C > charge_neutrality_tolerance_C:
        raise ValueError(
            "microscopic topology charge residual "
            f"{charge_residual_C:.6e} C exceeds the declared tolerance "
            f"{charge_neutrality_tolerance_C:.6e} C"
        )
    if configuration.box_vectors_m.shape != (
        CARTESIAN_DIMENSION,
        CARTESIAN_DIMENSION,
    ) or abs(np.linalg.det(configuration.box_vectors_m)) <= 0.0:
        raise ValueError("box_vectors_m must define a positive periodic volume")


def compute_first_principles_conductivity(
    recipe: ElectrolyteRecipeModel,
    interatomic_model: InteratomicModel,
    temperature_K: float,
    density_kg_m3: float,
    molecule_count: int,
    dynamics: DynamicsSettings,
    numerics: NumericalSettings,
    random_seed: int,
) -> ConductivityResult:
    if temperature_K <= 0.0 or density_kg_m3 <= 0.0 or molecule_count <= 0:
        raise ValueError("temperature, density, and molecule count must be positive")
    _validate_settings(dynamics, numerics)
    random_generator = np.random.default_rng(random_seed)
    configuration = interatomic_model.initial_configuration(
        recipe,
        density_kg_m3,
        molecule_count,
        random_generator,
    )
    _validate_microscopic_configuration(
        configuration,
        molecule_count,
        numerics.charge_consistency_tolerance_C,
    )
    configuration = MicroscopicConfiguration(
        positions_m=configuration.positions_m,
        velocities_m_s=maxwell_boltzmann_velocities(
            configuration.masses_kg, temperature_K, random_generator
        ),
        masses_kg=configuration.masses_kg,
        partial_charges_C=configuration.partial_charges_C,
        molecule_index=configuration.molecule_index,
        atom_type_index=configuration.atom_type_index,
        box_vectors_m=configuration.box_vectors_m,
    )
    validate_force_consistency(
        configuration, interatomic_model, dynamics, random_generator
    )
    equilibrated = equilibrate_configuration(
        configuration,
        interatomic_model,
        temperature_K,
        dynamics,
        random_generator,
    )
    trajectory = sample_equilibrium_trajectory(
        equilibrated,
        interatomic_model,
        temperature_K,
        dynamics,
        random_generator,
    )
    current_A_m2, molecular_current_A_m2 = current_series_A_m2(trajectory)
    original_sample_count = current_A_m2.shape[0]
    trajectory, current_A_m2, effective_sample_size = _stationary_trajectory(
        trajectory, current_A_m2, numerics
    )
    stationary_start_index = original_sample_count - current_A_m2.shape[0]
    molecular_current_A_m2 = molecular_current_A_m2[stationary_start_index:]
    charge_acceleration_A_m2_s, molecular_acceleration_A_m2_s = (
        charge_acceleration_series_A_m2_s(trajectory)
    )
    basis, generator_basis = _basis_and_generator(
        trajectory,
        current_A_m2,
        charge_acceleration_A_m2_s,
        numerics,
        dynamics.langevin_friction_per_s,
    )
    volume_m3 = float(abs(np.linalg.det(trajectory.box_vectors_m)))
    conductivity, _first_basis, history, residuals, basis_size = refine_projected_basis(
        basis,
        generator_basis,
        current_A_m2,
        volume_m3,
        temperature_K,
        numerics,
    )
    molecular_basis, molecular_generator_basis = _basis_and_generator(
        trajectory,
        molecular_current_A_m2,
        molecular_acceleration_A_m2_s,
        numerics,
        dynamics.langevin_friction_per_s,
    )
    direct, _direct_first_basis, _direct_history, _direct_residuals, _direct_size = (
        refine_projected_basis(
            molecular_basis,
            molecular_generator_basis,
            molecular_current_A_m2,
            volume_m3,
            temperature_K,
            numerics,
        )
    )
    green_kubo = integrated_green_kubo_conductivity(
        current_A_m2,
        trajectory.sample_interval_s,
        volume_m3,
        temperature_K,
        numerics,
    )
    return ConductivityResult(
        conductivity_S_m=conductivity,
        direct_current_term_S_m=direct,
        projected_correction_S_m=conductivity - direct,
        green_kubo_conductivity_S_m=green_kubo,
        basis_size=basis_size,
        basis_conductivities_S_m=history,
        residual_history=residuals,
        maximum_residual_score=residuals[-1],
        sample_count=current_A_m2.shape[0],
        effective_sample_size=effective_sample_size,
    )


def _settings_from_record(
    record: dict,
) -> tuple[DynamicsSettings, NumericalSettings]:
    return (
        DynamicsSettings(**record["dynamics"]),
        NumericalSettings(**record["numerics"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-json", required=True, type=Path)
    parser.add_argument("--interatomic-model-json", required=True, type=Path)
    parser.add_argument("--numerics-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    arguments = parser.parse_args()
    recipe = ElectrolyteRecipeModel.model_validate(
        read_json_object(arguments.recipe_json, "electrolyte recipe")
    )
    model_record = read_json_object(
        arguments.interatomic_model_json, "interatomic model"
    )
    settings_record = read_json_object(
        arguments.numerics_json, "first-principles conductivity numerics"
    )
    dynamics, numerics = _settings_from_record(settings_record)
    model = TorchScriptInteratomicModel(
        model_path=Path(model_record["torchscript_model_path"]),
        configuration_path=Path(model_record["configuration_npz_path"]),
        recipe=ElectrolyteRecipeModel.model_validate(model_record["recipe"]),
        charge_consistency_tolerance_C=numerics.charge_consistency_tolerance_C,
    )
    result = compute_first_principles_conductivity(
        recipe=recipe,
        interatomic_model=model,
        temperature_K=float(settings_record["temperature_K"]),
        density_kg_m3=float(settings_record["density_kg_m3"]),
        molecule_count=int(settings_record["molecule_count"]),
        dynamics=dynamics,
        numerics=numerics,
        random_seed=int(settings_record["random_seed"]),
    )
    write_json_object(arguments.output_json, asdict(result), "conductivity result")
    print(f"conductivity = {result.conductivity_S_m:.8g} S/m")
    print(
        "conductivity = "
        f"{result.conductivity_S_m * S_M_TO_MS_CM:.8g} mS/cm"
    )
    print(f"Green-Kubo = {result.green_kubo_conductivity_S_m:.8g} S/m")
    print(f"molecular COM direct = {result.direct_current_term_S_m:.8g} S/m")
    print(f"projected correction = {result.projected_correction_S_m:.8g} S/m")
    print(f"basis conductivity sequence = {result.basis_conductivities_S_m}")
    print(f"held-out residual sequence = {result.residual_history}")
    print(f"effective sample size = {result.effective_sample_size:.8g}")
    print(f"basis size = {result.basis_size}; samples = {result.sample_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
