from constants import T_REF_K
import numpy as np
import pytest

from conductivity.physical_library.projected_primitives_io import (
    PrimitiveExternalScalarInput,
    PrimitiveScalarEstimateNotProvided,
    PrimitiveScalarEstimateValue,
    PrimitiveScalarGapNotComputed,
    PrimitiveScalarGapValue,
    audit_primitive_oracle_closure,
)
from conductivity.physical_library.projected_analytical_conductivity import (
    ProjectedPrimitiveInput,
)


def test_projection_closure_audit_separates_gk_eh_and_recipe_gaps() -> None:
    trajectory_primitives = _two_state_primitives(
        capacity_flux_mol_m3_s=2.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=1.0e-10,
    )
    projected_primitives = _two_state_primitives(
        capacity_flux_mol_m3_s=3.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=1.0e-10,
    )
    recipe_primitives = _two_state_primitives(
        capacity_flux_mol_m3_s=3.0e12,
        first_moment_m=2.0e-10,
        self_diffusion_m2_s=4.0e-10,
    )

    report = audit_primitive_oracle_closure(
        trajectory_primitives,
        projected_primitives,
        recipe_primitives,
        PrimitiveExternalScalarInput(
            green_kubo=PrimitiveScalarEstimateValue(sigma_mS_cm=4.0),
            einstein_helfand=PrimitiveScalarEstimateValue(sigma_mS_cm=4.1),
        ),
        PrimitiveExternalScalarInput(
            green_kubo=PrimitiveScalarEstimateValue(sigma_mS_cm=4.05),
            einstein_helfand=PrimitiveScalarEstimateValue(sigma_mS_cm=4.15),
        ),
        PrimitiveExternalScalarInput(
            green_kubo=PrimitiveScalarEstimateValue(sigma_mS_cm=6.0),
            einstein_helfand=PrimitiveScalarEstimateNotProvided(
                scalar_name="einstein_helfand"
            ),
        ),
    )

    assert report.projection_gap.K_gap_mol_m3_s > 0.0
    assert report.projection_gap.d_gap_m == pytest.approx(0.0)
    assert report.recipe_primitive_gap.K_gap_mol_m3_s == pytest.approx(0.0)
    assert report.recipe_primitive_gap.d_gap_m > 0.0
    assert report.recipe_primitive_gap.D_self_gap_m2_s > 0.0
    assert isinstance(report.scalar_gap.green_kubo, PrimitiveScalarGapValue)
    assert report.scalar_gap.green_kubo.gap_mS_cm == pytest.approx(2.0)
    assert isinstance(report.scalar_gap.einstein_helfand, PrimitiveScalarGapNotComputed)


def test_projection_closure_audit_fails_on_illegal_primitive_process() -> None:
    illegal_recipe_primitives = _two_state_primitives(
        capacity_flux_mol_m3_s=2.0e12,
        first_moment_m=1.0e-10,
        self_diffusion_m2_s=1.0e-10,
    )
    illegal_recipe_primitives = ProjectedPrimitiveInput(
        state_concentrations_mol_m3=illegal_recipe_primitives.state_concentrations_mol_m3,
        symmetric_capacity_fluxes_K_ij_mol_m3_s=(
            illegal_recipe_primitives.symmetric_capacity_fluxes_K_ij_mol_m3_s
        ),
        transition_first_moments_d_ij_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0e-10, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        transition_second_moments_M_ij_m2=(
            illegal_recipe_primitives.transition_second_moments_M_ij_m2
        ),
        self_current_tensors_D_self_i_m2_s=(
            illegal_recipe_primitives.self_current_tensors_D_self_i_m2_s
        ),
        mori_memory_matrix_A=illegal_recipe_primitives.mori_memory_matrix_A,
        mori_current_coupling_matrix_h=(
            illegal_recipe_primitives.mori_current_coupling_matrix_h
        ),
        temperature_K=illegal_recipe_primitives.temperature_K,
        volume_m3=illegal_recipe_primitives.volume_m3,
    )
    scalar_input = PrimitiveExternalScalarInput(
        green_kubo=PrimitiveScalarEstimateNotProvided(scalar_name="green_kubo"),
        einstein_helfand=PrimitiveScalarEstimateNotProvided(
            scalar_name="einstein_helfand"
        ),
    )

    with pytest.raises(ValueError, match="d_ji must equal -d_ij"):
        audit_primitive_oracle_closure(
            illegal_recipe_primitives,
            illegal_recipe_primitives,
            illegal_recipe_primitives,
            scalar_input,
            scalar_input,
            scalar_input,
        )


def _two_state_primitives(
    capacity_flux_mol_m3_s: float,
    first_moment_m: float,
    self_diffusion_m2_s: float,
) -> ProjectedPrimitiveInput:
    return ProjectedPrimitiveInput(
        state_concentrations_mol_m3=np.asarray([500.0, 500.0], dtype=float),
        symmetric_capacity_fluxes_K_ij_mol_m3_s=np.asarray(
            [[0.0, capacity_flux_mol_m3_s], [capacity_flux_mol_m3_s, 0.0]],
            dtype=float,
        ),
        transition_first_moments_d_ij_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [first_moment_m, 0.0, 0.0]],
                [[-first_moment_m, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        transition_second_moments_M_ij_m2=np.asarray(
            [
                [
                    np.zeros((3, 3), dtype=float),
                    np.diag([first_moment_m * first_moment_m, 0.0, 0.0]),
                ],
                [
                    np.diag([first_moment_m * first_moment_m, 0.0, 0.0]),
                    np.zeros((3, 3), dtype=float),
                ],
            ],
            dtype=float,
        ),
        self_current_tensors_D_self_i_m2_s=np.asarray(
            [np.eye(3) * self_diffusion_m2_s, np.eye(3) * self_diffusion_m2_s],
            dtype=float,
        ),
        mori_memory_matrix_A=np.zeros((0, 0), dtype=float),
        mori_current_coupling_matrix_h=np.zeros((0, 3), dtype=float),
        temperature_K=T_REF_K,
        volume_m3=1.0,
    )
