from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None  # type: ignore[assignment]

from validation.oracles.analyze_kh_mcnally import (
    REFERENCE_TIME_MAX,
    ReferenceError,
    SnapshotError,
    load_public_references,
    maximum_vertical_kinetic_energy_density,
    mode_amplitude,
    official_initial_condition_errors,
    read_snapshot,
    reference_at_time,
    snapshot_diagnostics,
)
from validation.oracles.check_kh_mcnally_trajectory import (
    MAXIMUM_ABSOLUTE_MOMENTUM_DRIFT,
    MAXIMUM_RELATIVE_DIAGNOSTIC_ENERGY_DRIFT,
    MAXIMUM_RELATIVE_MASS_DRIFT,
    MINIMUM_TERMINAL_MODE_GROWTH,
    enforce_trajectory_rows,
    load_corrected_c_trajectory,
)
from validation.oracles.reduce_kh_mcnally_trajectory import FIELDS

ORACLE_DIR = Path(__file__).parents[1] / "oracles" / "kh_mcnally"
REPOSITORY_ROOT = Path(__file__).parents[2]


class KhMcnallyOracleTests(unittest.TestCase):
    def test_public_assets_configuration_and_run_horizon_are_pinned(self) -> None:
        manifest = json.loads((ORACLE_DIR / "assets.json").read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            {record["name"] for record in manifest["assets"]},
            {
                "kh_mcnally_2d_ics.hdf5",
                "public.params",
                "khmode_rev0.txt",
                "khener_rev1.txt",
            },
        )
        expected = {
            "kh_mcnally_2d_ics.hdf5": (
                2_948_736,
                "55d2a579f8e1710b65feacf1941a65f465b76742037e943e229c1c5ccb215777",
            ),
            "public.params": (
                2_186,
                "21780fc538046c3fa02b357128be698479afb428e8a55f7f73998f2af33a8441",
            ),
            "khmode_rev0.txt": (
                5_989,
                "ce9f36369d5f0178b976e12abb0eb99c95fcd37b01f41bbd77379d13d0fcb340",
            ),
            "khener_rev1.txt": (
                4_048,
                "7db4537eec7bcfd703662c014e6556b9105217574fc10fa02a1da4b785407113",
            ),
        }
        for record in manifest["assets"]:
            self.assertEqual(
                (record["size_bytes"], record["sha256"]), expected[record["name"]]
            )
            local = ORACLE_DIR / record["name"]
            self.assertEqual(local.stat().st_size, record["size_bytes"])
            self.assertEqual(
                hashlib.sha256(local.read_bytes()).hexdigest(), record["sha256"]
            )
        self.assertEqual(
            (ORACLE_DIR / "public.params").read_bytes(),
            (
                REPOSITORY_ROOT / "scripts/test_problems/kh_mcnally_2d.params"
            ).read_bytes(),
        )
        self.assertIn(
            "TimeMax                            10",
            (ORACLE_DIR / "public.params").read_text(),
        )
        self.assertEqual(REFERENCE_TIME_MAX, 1.5)
        self.assertEqual(
            (ORACLE_DIR / "public-config.sh").read_text().splitlines(),
            [
                "HYDRO_MESHLESS_FINITE_MASS",
                "BOX_PERIODIC",
                "BOX_SPATIAL_DIMENSION=2",
                "SELFGRAVITY_OFF",
                "EOS_GAMMA=(5.0/3.0)",
                "KERNEL_FUNCTION=5",
            ],
        )

    def test_reference_tables_are_strict_and_not_extrapolated(self) -> None:
        references = load_public_references(ORACLE_DIR)
        self.assertEqual(references["mode"].shape, (76, 3))
        self.assertEqual(references["energy"].shape, (76, 2))
        initial = reference_at_time(0.0, references)
        self.assertAlmostEqual(initial["mode_amplitude"], 0.01, places=13)
        self.assertAlmostEqual(
            initial["maximum_vertical_kinetic_energy_density"],
            9.999886500175594667e-5,
            places=18,
        )
        terminal = reference_at_time(1.5, references)
        self.assertAlmostEqual(
            terminal["mode_amplitude"], 0.1479463247047325, places=14
        )
        self.assertAlmostEqual(
            terminal["maximum_vertical_kinetic_energy_density"],
            0.08347445880021516,
            places=14,
        )
        with self.assertRaisesRegex(ReferenceError, "outside"):
            reference_at_time(1.500001, references)

    def test_corrected_c_trajectory_is_checksum_pinned_and_passes_the_gate(
        self,
    ) -> None:
        manifest_path = ORACLE_DIR / "corrected-c-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertIsNone(manifest["source_commit"])
        self.assertIn("without .git metadata", manifest["source_commit_note"])
        self.assertEqual(manifest["execution"]["mpi_ranks"], 4)
        self.assertEqual(
            manifest["execution"]["config_sha256"],
            "fabef9b80e4249979ae6e1f50035aa439492038b223660bd0187b2981f84f42e",
        )
        self.assertEqual(
            manifest["execution"]["parameters_sha256"],
            "318f02976a507ba143b9a4a41f56bd171b36db99e171379d14e2a7c27eb6bc3b",
        )
        for prefix in ("config", "parameters"):
            path = ORACLE_DIR / manifest["execution"][f"{prefix}_path"]
            self.assertEqual(
                path.stat().st_size,
                manifest["execution"][f"{prefix}_size_bytes"],
            )
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                manifest["execution"][f"{prefix}_sha256"],
            )
        record = manifest["trajectory"]
        trajectory_path = ORACLE_DIR / record["path"]
        self.assertEqual(trajectory_path.stat().st_size, record["size_bytes"])
        self.assertEqual(
            hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
            record["sha256"],
        )
        rows = load_corrected_c_trajectory(trajectory_path)
        for name in (
            "smoothing_length_minimum",
            "smoothing_length_maximum",
            "smoothing_length_mean",
            "effective_neighbors_minimum",
            "effective_neighbors_maximum",
            "effective_neighbors_mean",
        ):
            self.assertIn(name, FIELDS)
            self.assertTrue(all(float(row[name]) > 0.0 for row in rows))
        for row in rows:
            self.assertLessEqual(
                float(row["smoothing_length_minimum"]),
                float(row["smoothing_length_mean"]),
            )
            self.assertLessEqual(
                float(row["smoothing_length_mean"]),
                float(row["smoothing_length_maximum"]),
            )
            self.assertLessEqual(
                float(row["effective_neighbors_minimum"]),
                float(row["effective_neighbors_mean"]),
            )
            self.assertLessEqual(
                float(row["effective_neighbors_mean"]),
                float(row["effective_neighbors_maximum"]),
            )
        report = enforce_trajectory_rows(rows, load_public_references(ORACLE_DIR), rows)
        self.assertEqual(len(report["snapshots"]), 16)
        self.assertGreater(report["terminal_mode_growth"], MINIMUM_TERMINAL_MODE_GROWTH)
        self.assertLess(
            max(row["relative_mass_drift"] for row in report["snapshots"]),
            MAXIMUM_RELATIVE_MASS_DRIFT,
        )
        self.assertLess(
            max(
                row["maximum_absolute_component_momentum_drift"]
                for row in report["snapshots"]
            ),
            MAXIMUM_ABSOLUTE_MOMENTUM_DRIFT,
        )
        self.assertLess(
            max(row["relative_diagnostic_energy_drift"] for row in report["snapshots"]),
            MAXIMUM_RELATIVE_DIAGNOSTIC_ENERGY_DRIFT,
        )

    def test_trajectory_gate_rejects_frozen_explosive_and_nonconservative_runs(
        self,
    ) -> None:
        corrected = load_corrected_c_trajectory(
            ORACLE_DIR / "corrected-c-trajectory.csv"
        )
        references = load_public_references(ORACLE_DIR)

        frozen = copy.deepcopy(corrected)
        for row in frozen:
            row["mode_amplitude"] = frozen[0]["mode_amplitude"]
            row["maximum_vertical_kinetic_energy_density"] = frozen[0][
                "maximum_vertical_kinetic_energy_density"
            ]
        with self.assertRaisesRegex(SnapshotError, "anti-noop|published-curve"):
            enforce_trajectory_rows(frozen, references, corrected)

        explosive = copy.deepcopy(corrected)
        explosive[9]["maximum_vertical_kinetic_energy_density"] = 1.0
        with self.assertRaisesRegex(SnapshotError, "vertical kinetic-energy"):
            enforce_trajectory_rows(explosive, references, corrected)

        suppressed = copy.deepcopy(corrected)
        for row in suppressed:
            row["maximum_vertical_kinetic_energy_density"] = 1.0e-300
        with self.assertRaisesRegex(SnapshotError, "vertical kinetic-energy"):
            enforce_trajectory_rows(suppressed, references, corrected)

        wrong_neighbors = copy.deepcopy(corrected)
        wrong_neighbors[4]["effective_neighbors_maximum"] = 60.0
        with self.assertRaisesRegex(SnapshotError, "effective-neighbor"):
            enforce_trajectory_rows(wrong_neighbors, references, corrected)

        nonconservative = copy.deepcopy(corrected)
        nonconservative[8]["mass"] = float(nonconservative[0]["mass"]) * 1.001
        nonconservative[8]["momentum_x"] = (
            float(nonconservative[0]["momentum_x"]) + 1.0e-4
        )
        nonconservative[8]["diagnostic_total_energy"] = (
            float(nonconservative[0]["diagnostic_total_energy"]) * 1.01
        )
        with self.assertRaisesRegex(
            SnapshotError, "mass drift.*momentum drift.*energy drift"
        ):
            enforce_trajectory_rows(nonconservative, references, corrected)

    def test_trajectory_gate_never_extrapolates_the_published_curve(self) -> None:
        corrected = load_corrected_c_trajectory(
            ORACLE_DIR / "corrected-c-trajectory.csv"
        )
        outside = copy.deepcopy(corrected)
        outside[-1]["time"] = REFERENCE_TIME_MAX + 0.1
        with self.assertRaisesRegex(SnapshotError, "time 1.6"):
            enforce_trajectory_rows(
                outside, load_public_references(ORACLE_DIR), corrected
            )

    def test_equations_14_to_17_recover_amplitude_and_phase(self) -> None:
        side = 32
        centers = (np.arange(side) + 0.5) / side
        x, y = np.meshgrid(centers, centers, indexing="xy")
        phase = 0.37
        amplitude = 0.023
        velocity_y = amplitude * np.sin(4.0 * math.pi * x + phase)
        volume = np.full(x.size, 1.0 / x.size)
        measured = mode_amplitude(x.ravel(), y.ravel(), velocity_y.ravel(), volume)
        self.assertAlmostEqual(measured, amplitude, places=14)
        self.assertAlmostEqual(
            mode_amplitude(x.ravel(), y.ravel(), velocity_y.ravel(), 7.0 * volume),
            amplitude,
            places=14,
        )

    def test_diagnostics_reject_bad_quadrature_and_use_density_in_energy(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            mode_amplitude(
                np.array([0.0]),
                np.array([0.25]),
                np.array([0.01]),
                np.array([0.0]),
            )
        self.assertEqual(
            maximum_vertical_kinetic_energy_density(
                np.array([1.0, 3.0]), np.array([2.0, 1.5])
            ),
            3.375,
        )

    @unittest.skipIf(h5py is None, "h5py is required for fixture checks")
    def test_official_fixture_matches_translated_paper_state_and_tables(
        self,
    ) -> None:
        state = read_snapshot(ORACLE_DIR / "kh_mcnally_2d_ics.hdf5")
        errors = official_initial_condition_errors(state)
        self.assertLess(errors["density_max_abs_error"], 6.0e-8)
        self.assertLess(errors["velocity_x_max_abs_error"], 4.5e-8)
        self.assertLess(errors["velocity_y_max_abs_error"], 2.1e-9)
        self.assertLess(errors["internal_energy_max_abs_error"], 1.6e-5)
        self.assertLess(errors["pressure_max_abs_error"], 1.5e-5)

        diagnostics = snapshot_diagnostics(state)
        self.assertEqual(diagnostics["particle_count"], 66_868)
        self.assertAlmostEqual(
            diagnostics["quadrature_volume_sum"], 1.000278457980588, places=13
        )
        self.assertAlmostEqual(
            diagnostics["mode_amplitude"], 0.010000086446268255, places=14
        )
        self.assertAlmostEqual(
            diagnostics["maximum_vertical_kinetic_energy_density"],
            9.999667484264911e-5,
            places=17,
        )
        reference = reference_at_time(0.0, load_public_references(ORACLE_DIR))
        self.assertLess(
            abs(diagnostics["mode_amplitude"] - reference["mode_amplitude"]),
            9.0e-8,
        )
        smoothing_length = np.asarray(state["smoothing_length"])
        effective_neighbors = (
            math.pi
            * smoothing_length**2
            * np.asarray(state["density"])
            / np.asarray(state["masses"])
        )
        self.assertAlmostEqual(float(smoothing_length.min()), 0.008017426, places=9)
        self.assertAlmostEqual(float(smoothing_length.max()), 0.011322895, places=9)
        self.assertTrue(np.all(np.isfinite(effective_neighbors)))
        self.assertTrue(np.all(effective_neighbors > 0.0))

    @unittest.skipIf(h5py is None, "h5py is required for snapshot checks")
    def test_snapshot_reader_rejects_three_dimensional_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.hdf5"
            with h5py.File(path, "w") as handle:
                header = handle.create_group("Header")
                header.attrs["BoxSize"] = 1.0
                header.attrs["Time"] = 0.0
                gas = handle.create_group("PartType0")
                gas["ParticleIDs"] = np.array([1])
                gas["Coordinates"] = np.array([[0.5, 0.5, 0.0]])
                gas["Velocities"] = np.array([[0.0, 0.0, 1.0e-9]])
                gas["Masses"] = np.array([1.0])
                gas["Density"] = np.array([1.0])
                gas["InternalEnergy"] = np.array([1.0])
                gas["SmoothingLength"] = np.array([0.1])
            with self.assertRaisesRegex(SnapshotError, "strictly two-dimensional"):
                read_snapshot(path)

    @unittest.skipIf(h5py is None, "h5py is required for snapshot checks")
    def test_snapshot_reader_rejects_noninteger_particle_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-ids.hdf5"
            with h5py.File(path, "w") as handle:
                header = handle.create_group("Header")
                header.attrs["BoxSize"] = 1.0
                header.attrs["Time"] = 0.0
                gas = handle.create_group("PartType0")
                gas["ParticleIDs"] = np.array([1.0])
                gas["Coordinates"] = np.array([[0.5, 0.5, 0.0]])
                gas["Velocities"] = np.array([[0.0, 0.0, 0.0]])
                gas["Masses"] = np.array([1.0])
                gas["Density"] = np.array([1.0])
                gas["InternalEnergy"] = np.array([1.0])
            with self.assertRaisesRegex(SnapshotError, "integer"):
                read_snapshot(path)

    @unittest.skipIf(h5py is None, "h5py is required for snapshot checks")
    def test_snapshot_reader_requires_positive_smoothing_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-smoothing.hdf5"
            with h5py.File(path, "w") as handle:
                header = handle.create_group("Header")
                header.attrs["BoxSize"] = 1.0
                header.attrs["Time"] = 0.0
                gas = handle.create_group("PartType0")
                gas["ParticleIDs"] = np.array([1])
                gas["Coordinates"] = np.array([[0.5, 0.5, 0.0]])
                gas["Velocities"] = np.array([[0.0, 0.0, 0.0]])
                gas["Masses"] = np.array([1.0])
                gas["Density"] = np.array([1.0])
                gas["InternalEnergy"] = np.array([1.0])
            with self.assertRaisesRegex(SnapshotError, "SmoothingLength"):
                read_snapshot(path)

            with h5py.File(path, "a") as handle:
                handle["PartType0"]["SmoothingLength"] = np.array([0.0])
            with self.assertRaisesRegex(SnapshotError, "smoothing length"):
                read_snapshot(path)

            with h5py.File(path, "a") as handle:
                handle["PartType0"]["SmoothingLength"][...] = np.nan
            with self.assertRaisesRegex(SnapshotError, "non-finite"):
                read_snapshot(path)


if __name__ == "__main__":
    unittest.main()
