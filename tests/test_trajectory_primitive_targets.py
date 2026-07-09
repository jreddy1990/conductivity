import io
import tarfile

import numpy as np
import pytest

from constants import T_REF_K
from conductivity.physical_library.generator_construction import (
    charge_covariance_mobility_from_center_matrix,
)
from conductivity.old.trajectory_primitive_targets import (
    PF6AssociationCutoffs,
    PF6TrajectoryPrimitiveTargetInput,
    PF6ZenodoTrajectoryLayout,
    TrajectoryPrimitiveTargetProcessInput,
    TrajectoryMarkovAdditiveSampleInput,
    build_trajectory_primitive_target_markov_input,
    build_sampled_trajectory_markov_additive_input,
    compute_pf6_trajectory_primitive_targets,
    compute_trajectory_primitive_target_conductivity,
    project_sampled_trajectory_to_generator_primitives,
)


def test_charge_covariance_uses_center_mobility_matrix() -> None:
    charge_numbers = (1.0, -1.0)

    uncorrelated_charge_mobility = charge_covariance_mobility_from_center_matrix(
        charge_numbers,
        ((1.0e-10, 0.0), (0.0, 2.0e-10)),
    )
    perfect_comotion_charge_mobility = charge_covariance_mobility_from_center_matrix(
        charge_numbers,
        ((2.0e-10, 2.0e-10), (2.0e-10, 2.0e-10)),
    )
    anticorrelated_charge_mobility = charge_covariance_mobility_from_center_matrix(
        charge_numbers,
        ((1.0e-10, -0.4e-10), (-0.4e-10, 1.0e-10)),
    )

    assert uncorrelated_charge_mobility == pytest.approx(3.0e-10)
    assert perfect_comotion_charge_mobility == pytest.approx(0.0)
    assert anticorrelated_charge_mobility == pytest.approx(2.8e-10)


def _two_state_process_input() -> TrajectoryPrimitiveTargetProcessInput:
    return TrajectoryPrimitiveTargetProcessInput(
        state_labels=("A", "B"),
        state_index_by_frame=np.asarray((0, 1, 0, 1, 0), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            (
                (1.0e-10, 0.0, 0.0),
                (-1.0e-10, 0.0, 0.0),
                (1.0e-10, 0.0, 0.0),
                (-1.0e-10, 0.0, 0.0),
            ),
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )


def test_trajectory_target_process_builds_reversible_pair_events():
    markov_input, diagnostics, state_index_remap = (
        build_trajectory_primitive_target_markov_input(_two_state_process_input())
    )

    assert state_index_remap == {0: 0, 1: 1}
    assert diagnostics.transition_sample_count == 4
    assert diagnostics.generated_event_count == 8
    assert markov_input.state_labels == ("A", "B")
    assert np.all(markov_input.state_concentrations_mol_m3 > 0.0)

    for event in markov_input.events:
        reverse_matches = tuple(
            candidate
            for candidate in markov_input.events
            if candidate.from_state_index == event.to_state_index
            and candidate.to_state_index == event.from_state_index
            and np.allclose(
                np.asarray(candidate.charge_displacement_m),
                -np.asarray(event.charge_displacement_m),
            )
        )
        assert reverse_matches


def test_trajectory_target_process_computes_nonnegative_conductivity():
    result = compute_trajectory_primitive_target_conductivity(
        _two_state_process_input(),
    )

    assert result.conductivity_result.sigma_mS_cm >= 0.0
    assert result.conductivity_result.event_reversal_residual_mol_m3_s >= 0.0


def test_trajectory_target_process_keeps_self_displacement_as_direct_variance():
    process_input = TrajectoryPrimitiveTargetProcessInput(
        state_labels=("A",),
        state_index_by_frame=np.asarray((0, 0, 0), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            ((1.0e-10, 0.0, 0.0), (-1.0e-10, 0.0, 0.0)),
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )

    markov_input, diagnostics, state_index_remap = (
        build_trajectory_primitive_target_markov_input(process_input)
    )

    assert state_index_remap == {0: 0}
    assert diagnostics.self_displacement_sample_count == 2
    assert diagnostics.generated_event_count == 4
    assert all(
        event.from_state_index == event.to_state_index for event in markov_input.events
    )


def test_sampled_trajectory_target_builds_parallel_center_process():
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("Li:contact_pair", "Li:free"),
        occupancy_state_index_by_observation=np.asarray((0, 1, 0, 1), dtype=int),
        from_state_index_by_step=np.asarray((0, 1), dtype=int),
        to_state_index_by_step=np.asarray((1, 0), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            ((1.0e-10, 0.0, 0.0), (-1.0e-10, 0.0, 0.0)),
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1000.0,
        temperature_K=T_REF_K,
    )

    markov_input, diagnostics, state_index_remap = (
        build_sampled_trajectory_markov_additive_input(sample_input)
    )

    assert state_index_remap == {0: 0, 1: 1}
    assert diagnostics.transition_sample_count == 2
    assert diagnostics.generated_event_count == 4
    assert markov_input.state_labels == ("Li:contact_pair", "Li:free")


def test_projected_generator_primitives_include_c_k_m_and_self_current():
    sample_input = TrajectoryMarkovAdditiveSampleInput(
        state_labels=("free_ion_center:Li+", "contact_pair_center:Li+"),
        occupancy_state_index_by_observation=np.asarray(
            (0, 0, 1, 1, 0, 1),
            dtype=int,
        ),
        from_state_index_by_step=np.asarray((0, 1, 0, 1), dtype=int),
        to_state_index_by_step=np.asarray((1, 0, 0, 1), dtype=int),
        charge_displacement_by_step_m=np.asarray(
            (
                (2.0e-10, 0.0, 0.0),
                (-2.0e-10, 0.0, 0.0),
                (1.0e-10, 0.0, 0.0),
                (-1.0e-10, 0.0, 0.0),
            ),
            dtype=float,
        ),
        dt_s=1.0e-12,
        total_transport_concentration_mol_m3=1200.0,
        temperature_K=T_REF_K,
    )

    primitive_set = project_sampled_trajectory_to_generator_primitives(sample_input)

    assert primitive_set.state_concentrations_mol_m3["free_ion_center:Li+"] == 600.0
    assert primitive_set.state_occupancy_fractions["contact_pair_center:Li+"] == 0.5
    assert len(primitive_set.reactive_fluxes) == 1
    reactive_flux = primitive_set.reactive_fluxes[0]
    assert reactive_flux.from_state_label == "free_ion_center:Li+"
    assert reactive_flux.to_state_label == "contact_pair_center:Li+"
    assert reactive_flux.symmetric_flux_mol_m3_s == 3.0e14
    assert reactive_flux.forward_rate_s_inv == 5.0e11
    assert reactive_flux.reverse_rate_s_inv == 5.0e11
    assert len(primitive_set.conditional_displacement_moments) == 2
    first_moment = primitive_set.conditional_displacement_moments[0]
    covariance = np.asarray(first_moment.covariance_m2, dtype=float)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-30
    assert len(primitive_set.self_current_tensors) == 2
    self_tensor_by_label = {
        self_current.state_label: np.asarray(
            self_current.diffusion_tensor_m2_s,
            dtype=float,
        )
        for self_current in primitive_set.self_current_tensors
    }
    assert np.isclose(self_tensor_by_label["free_ion_center:Li+"][0, 0], 5.0e-9)
    assert np.isclose(
        self_tensor_by_label["contact_pair_center:Li+"][0, 0],
        5.0e-9,
    )
    assert primitive_set.markov_conductivity_result.sigma_mS_cm >= 0.0


def test_pf6_zenodo_parser_builds_primitive_targets_from_streamed_xyz(tmp_path):
    archive_path = tmp_path / "traj_PF6.tar.gz"
    member_name = "traj_PF6.xyz"
    _write_synthetic_pf6_archive(archive_path, member_name)

    target_input = PF6TrajectoryPrimitiveTargetInput(
        system_id="synthetic_ec_emc_lipf6",
        archive_path=archive_path,
        member_name=member_name,
        layout=PF6ZenodoTrajectoryLayout(
            expected_atom_count=33,
            ec_molecule_count=1,
            ec_atoms_per_molecule=10,
            emc_molecule_count=1,
            emc_atoms_per_molecule=15,
            pf6_molecule_count=1,
            pf6_atoms_per_molecule=7,
            li_atom_count=1,
        ),
        association_cutoffs=PF6AssociationCutoffs(
            contact_pair_max_distance_A=2.5,
            solvent_separated_pair_max_distance_A=5.0,
            aggregate_counterion_count=2,
        ),
        max_frames=3,
        frame_stride=1,
        block_count=1,
        temperature_K=333.0,
        expected_frame_interval_ps=1.5,
    )

    result = compute_pf6_trajectory_primitive_targets(target_input)

    assert result.diagnostics.frame_count == 3
    assert result.diagnostics.raw_frame_interval_ps == 1.5
    assert result.diagnostics.transition_sample_count == 4
    assert result.diagnostics.li_state_counts == {
        "free_ion_center:Li+": 1,
        "contact_pair_center:Li+": 1,
        "solvent_separated_pair_center:Li+": 1,
        "internal_polarization_center:Li+": 0,
    }
    assert result.diagnostics.pf6_state_counts == {
        "free_ion_center:PF6-": 1,
        "contact_pair_center:PF6-": 1,
        "solvent_separated_pair_center:PF6-": 1,
        "internal_polarization_center:PF6-": 0,
    }
    assert result.sample_input.charge_displacement_by_step_m.shape == (4, 3)
    assert result.process_result.conductivity_result.sigma_mS_cm >= 0.0
    artifact = result.primitive_target_artifact
    assert artifact.system_id == "synthetic_ec_emc_lipf6"
    assert artifact.block_count == 1
    assert artifact.state_occupancy_fractions["contact_pair_center:Li+"] > 0.0
    assert artifact.transition_rates_s_inv
    assert artifact.transition_rate_targets_validated is True
    assert artifact.displacement_moments_by_family
    assert artifact.displacement_moment_targets_validated is True
    assert artifact.markov_additive_sigma_mS_cm >= 0.0
    assert artifact.markov_additive_sigma_validated is True


def _write_synthetic_pf6_archive(archive_path, member_name):
    frame_text = "".join(
        (
            _synthetic_pf6_frame(0.0, 20.0, 0.0),
            _synthetic_pf6_frame(1.5, 20.0, 2.5),
            _synthetic_pf6_frame(3.0, 20.0, 6.5),
        ),
    )
    frame_bytes = frame_text.encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        tar_info = _tar_info_with_size(member_name, len(frame_bytes))
        archive.addfile(tar_info, io.BytesIO(frame_bytes))


def _tar_info_with_size(member_name, size_bytes):
    tar_info = tarfile.TarInfo(member_name)
    object.__setattr__(tar_info, "size", size_bytes)
    return tar_info


def _synthetic_pf6_frame(time_ps, box_length_A, li_x_offset_A):
    atoms = (
        *_ec_atoms(),
        *_emc_atoms(),
        *_pf6_atoms(),
        ("Li", 6.5 + li_x_offset_A, 5.0, 5.0),
    )
    atom_lines = tuple(
        f"{element} {x_position_A:.6f} {y_position_A:.6f} {z_position_A:.6f}"
        for element, x_position_A, y_position_A, z_position_A in atoms
    )
    lines = (f"{len(atoms)} {box_length_A:.1f} {time_ps:.1f}", "", *atom_lines)
    return "\n".join(lines) + "\n"


def _ec_atoms():
    return (
        ("C", 1.0, 1.0, 1.0),
        ("C", 1.5, 1.0, 1.0),
        ("O", 1.0, 1.5, 1.0),
        ("O", 1.5, 1.5, 1.0),
        ("C", 1.25, 1.25, 1.5),
        ("H", 0.5, 1.0, 1.0),
        ("H", 2.0, 1.0, 1.0),
        ("H", 1.0, 2.0, 1.0),
        ("H", 1.5, 2.0, 1.0),
        ("O", 1.25, 1.25, 2.0),
    )


def _emc_atoms():
    return (
        ("C", 10.0, 10.0, 10.0),
        ("O", 10.5, 10.0, 10.0),
        ("C", 11.0, 10.0, 10.0),
        ("O", 11.5, 10.0, 10.0),
        ("C", 12.0, 10.0, 10.0),
        ("C", 12.5, 10.0, 10.0),
        ("H", 9.5, 10.0, 10.0),
        ("H", 10.0, 9.5, 10.0),
        ("H", 10.0, 10.5, 10.0),
        ("O", 13.0, 10.0, 10.0),
        ("H", 11.0, 9.5, 10.0),
        ("H", 11.0, 10.5, 10.0),
        ("H", 12.5, 9.5, 10.0),
        ("H", 12.5, 10.5, 10.0),
        ("H", 13.0, 10.5, 10.0),
    )


def _pf6_atoms():
    return (
        ("P", 5.0, 5.0, 5.0),
        ("F", 4.5, 5.0, 5.0),
        ("F", 5.5, 5.0, 5.0),
        ("F", 5.0, 4.5, 5.0),
        ("F", 5.0, 5.5, 5.0),
        ("F", 5.0, 5.0, 4.5),
        ("F", 5.0, 5.0, 5.5),
    )
