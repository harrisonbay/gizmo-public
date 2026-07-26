from __future__ import annotations

import csv
import gzip
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

from validation.oracles.check_gresho_trajectory import (
    MAXIMUM_TERMINAL_C_POSITION_L1,
    MAXIMUM_TERMINAL_C_VELOCITY_L1,
    corrected_c_differentials,
)
from validation.oracles.compare_gresho_analytic import (
    BOX_SIZE,
    GAMMA,
    SnapshotError,
    analytic_fields,
    check_official_initial_condition,
    comparison_metrics,
    enforce_limits,
    parse_limits,
    read_snapshot,
    reduced_rows,
    write_reduction,
)

ORACLE_DIR = Path(__file__).parents[1] / "oracles" / "gresho"


def write_synthetic_snapshot(
    path: Path,
    *,
    count: int = 64,
    velocity_sign: float = 1.0,
    radial_velocity: float = 0.0,
    pressure_scale: float = 1.0,
    z: float = 0.0,
    duplicate_id: bool = False,
) -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    radius = np.linspace(0.01, 0.49, count)
    x = 0.5 + radius * np.cos(angles)
    y = 0.5 + radius * np.sin(angles)
    tangential, pressure = analytic_fields(radius)
    vx = velocity_sign * -tangential * np.sin(angles) + radial_velocity * np.cos(angles)
    vy = velocity_sign * tangential * np.cos(angles) + radial_velocity * np.sin(angles)
    ids = np.arange(count, dtype=np.uint32)
    if duplicate_id:
        ids[-1] = ids[-2]
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["BoxSize"] = BOX_SIZE
        header.attrs["Time"] = 0.0
        gas = handle.create_group("PartType0")
        gas["Coordinates"] = np.column_stack((x, y, np.full(count, z)))
        gas["Velocities"] = np.column_stack((vx, vy, np.full(count, z)))
        gas["Masses"] = np.full(count, 1.0 / count)
        gas["Density"] = np.ones(count)
        gas["InternalEnergy"] = pressure_scale * pressure / (GAMMA - 1.0)
        gas["SmoothingLength"] = np.full(count, 0.05)
        gas["ParticleIDs"] = ids


class GreshoOracleTests(unittest.TestCase):
    def test_public_assets_and_configuration_are_exactly_pinned(self) -> None:
        manifest = json.loads((ORACLE_DIR / "assets.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            {asset["name"] for asset in manifest["assets"]},
            {"gresho_ics.hdf5", "public.params"},
        )
        records = {asset["name"]: asset for asset in manifest["assets"]}
        self.assertEqual(records["gresho_ics.hdf5"]["size_bytes"], 186_624)
        self.assertEqual(
            records["gresho_ics.hdf5"]["sha256"],
            "9109bbce2a58eef3cb13d665582fa636e667a1098ae491b80f8f39c2fae1a43c",
        )
        for asset in manifest["assets"]:
            self.assertTrue(asset["url"].startswith("http://www.tapir.caltech.edu/"))
            local = ORACLE_DIR / asset["name"]
            if local.exists():
                self.assertEqual(local.stat().st_size, asset["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(local.read_bytes()).hexdigest(), asset["sha256"]
                )
        self.assertEqual(
            (ORACLE_DIR / "public.params").read_bytes(),
            (
                Path(__file__).parents[2] / "scripts/test_problems/gresho.params"
            ).read_bytes(),
        )
        self.assertEqual(
            (ORACLE_DIR / "public-config.sh").read_text(encoding="utf-8").splitlines(),
            [
                "HYDRO_MESHLESS_FINITE_MASS",
                "BOX_PERIODIC",
                "BOX_SPATIAL_DIMENSION=2",
                "SELFGRAVITY_OFF",
                "EOS_GAMMA=(1.4)",
            ],
        )

    def test_corrected_c_manifest_tables_and_analytic_metrics_are_consistent(
        self,
    ) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "evolution-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["source_commit"], "f40611353d02f68d8714eb380e92fef1c1c34c9c"
        )
        self.assertEqual(
            manifest["source_repository"],
            "https://github.com/harrisonbay/gizmo-public",
        )
        self.assertEqual(
            manifest["upstream_base_commit"],
            "a828c4ba79d67093bd1a3d04f6f90b5d2d94d65f",
        )
        self.assertIn("7847189", manifest["c_correction_commits"])
        self.assertEqual(manifest["timeline"]["steps"], 8_192)
        self.assertEqual(manifest["timeline"]["shared_step_ticks"], 2**47)
        self.assertEqual(manifest["timeline"]["scheduled_outputs"], 7)
        self.assertEqual(
            manifest["timeline"]["output_ticks"],
            [
                0,
                192_153_584_101_141_152,
                384_307_168_202_282_304,
                576_460_752_303_423_488,
                768_614_336_404_564_608,
                960_767_920_505_705_856,
                2**60,
            ],
        )
        self.assertEqual(
            manifest["parameter_resolution"]["ignored_without_developer_mode"],
            ["ErrTolIntAccuracy", "CourantFac", "MaxRMSDisplacementFac"],
        )
        for record in manifest["inputs"].values():
            local = ORACLE_DIR / record["path"]
            self.assertEqual(
                hashlib.sha256(local.read_bytes()).hexdigest(), record["sha256"]
            )
        for record in manifest["tools"].values():
            local = ORACLE_DIR / record["path"]
            self.assertEqual(
                hashlib.sha256(local.read_bytes()).hexdigest(), record["sha256"]
            )

        expected_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.assertEqual(
            [record["time"] for record in manifest["snapshots"]], expected_times
        )
        expected_fields = manifest["table_format"]["columns"]
        previous_mass: np.ndarray | None = None
        for record in manifest["snapshots"]:
            table = ORACLE_DIR / record["table"]
            self.assertEqual(table.stat().st_size, record["table_size_bytes"])
            self.assertEqual(
                hashlib.sha256(table.read_bytes()).hexdigest(),
                record["table_sha256"],
            )
            with gzip.open(table, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), expected_fields)
            self.assertEqual(len(rows), 4_092)
            self.assertEqual(
                [int(row["particle_id"]) for row in rows], list(range(4_092))
            )

            def column(
                name: str, source_rows: list[dict[str, str]] = rows
            ) -> np.ndarray:
                values = np.array([float(row[name]) for row in source_rows])
                self.assertTrue(np.all(np.isfinite(values)))
                return values

            x = column("x")
            y = column("y")
            velocity_x = column("velocity_x")
            velocity_y = column("velocity_y")
            density = column("density")
            internal_energy = column("specific_internal_energy")
            smoothing_length = column("smoothing_length")
            mass = column("mass")
            self.assertTrue(np.all((x >= 0.0) & (x < 1.0)))
            self.assertTrue(np.all((y >= 0.0) & (y < 1.0)))
            self.assertTrue(np.all(density > 0.0))
            self.assertTrue(np.all(internal_energy > 0.0))
            self.assertTrue(np.all(smoothing_length > 0.0))
            self.assertTrue(np.all(mass > 0.0))
            if previous_mass is not None:
                self.assertTrue(np.array_equal(mass, previous_mass))
            previous_mass = mass

            dx = (x - 0.5 + 0.5) % 1.0 - 0.5
            dy = (y - 0.5 + 0.5) % 1.0 - 0.5
            radius = np.hypot(dx, dy)
            analytic_tangential, analytic_pressure = analytic_fields(radius)
            nonzero = radius > 0.0
            radial = np.zeros_like(radius)
            tangential = np.zeros_like(radius)
            analytic_velocity_x = np.zeros_like(radius)
            analytic_velocity_y = np.zeros_like(radius)
            radial[nonzero] = (
                velocity_x[nonzero] * dx[nonzero] + velocity_y[nonzero] * dy[nonzero]
            ) / radius[nonzero]
            tangential[nonzero] = (
                -velocity_x[nonzero] * dy[nonzero] + velocity_y[nonzero] * dx[nonzero]
            ) / radius[nonzero]
            analytic_velocity_x[nonzero] = (
                -analytic_tangential[nonzero] * dy[nonzero] / radius[nonzero]
            )
            analytic_velocity_y[nonzero] = (
                analytic_tangential[nonzero] * dx[nonzero] / radius[nonzero]
            )
            pressure = (GAMMA - 1.0) * density * internal_energy
            weights = mass / math.fsum(float(value) for value in mass)
            pressure_range = 4.0 * math.log(2.0) - 2.0

            def mean(values: np.ndarray, metric_weights: np.ndarray = weights) -> float:
                return float(np.sum(metric_weights * values))

            measured = {
                "density_l1": mean(np.abs(density - 1.0)),
                "pressure_l1": mean(np.abs(pressure - analytic_pressure)),
                "pressure_normalized_l1": mean(np.abs(pressure - analytic_pressure))
                / pressure_range,
                "specific_internal_energy_normalized_l1": mean(
                    np.abs(internal_energy - analytic_pressure / (GAMMA - 1.0))
                )
                / (pressure_range / (GAMMA - 1.0)),
                "radial_velocity_rms": math.sqrt(mean(radial**2)),
                "tangential_velocity_l1": mean(
                    np.abs(tangential - analytic_tangential)
                ),
                "velocity_vector_l1": mean(
                    np.hypot(
                        velocity_x - analytic_velocity_x,
                        velocity_y - analytic_velocity_y,
                    )
                ),
                "peak_tangential_velocity": float(np.max(tangential)),
            }
            self.assertEqual(set(measured), set(record["analytic"]))
            for name, value in measured.items():
                self.assertAlmostEqual(value, record["analytic"][name], delta=1.0e-15)

    def test_piecewise_analytic_solution_is_continuous(self) -> None:
        radii = np.array([0.0, 0.2 - 1.0e-12, 0.2, 0.4 - 1.0e-12, 0.4, 0.7])
        velocity, pressure = analytic_fields(radii)
        self.assertEqual(velocity[0], 0.0)
        self.assertEqual(velocity[-1], 0.0)
        self.assertAlmostEqual(velocity[1], velocity[2], delta=1.0e-10)
        self.assertAlmostEqual(velocity[3], velocity[4], delta=1.0e-10)
        self.assertAlmostEqual(pressure[1], pressure[2], delta=1.0e-10)
        self.assertAlmostEqual(pressure[3], pressure[4], delta=1.0e-10)
        self.assertAlmostEqual(pressure[0], 5.0)
        self.assertAlmostEqual(pressure[-1], 3.0 + 4.0 * math.log(2.0))

    @unittest.skipIf(h5py is None, "h5py is required for snapshot tests")
    def test_exact_synthetic_snapshot_has_roundoff_level_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact.hdf5"
            write_synthetic_snapshot(path)
            state = read_snapshot(path)
            metrics = comparison_metrics(state)
            self.assertTrue(all(value < 2.0e-15 for value in metrics.values()))
            rows = reduced_rows(state, bins=16)
            self.assertGreater(len(rows), 8)
            output = Path(directory) / "reduced.csv.gz"
            write_reduction(output, rows)
            first = output.read_bytes()
            write_reduction(output, rows)
            self.assertEqual(output.read_bytes(), first)

    @unittest.skipIf(h5py is None, "h5py is required for snapshot tests")
    def test_per_id_corrected_c_differential_is_zero_for_identical_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact.hdf5"
            write_synthetic_snapshot(path)
            state = read_snapshot(path)
            reference = {
                "x": state["coordinates"][:, 0].copy(),
                "y": state["coordinates"][:, 1].copy(),
                "velocity_x": state["velocities"][:, 0].copy(),
                "velocity_y": state["velocities"][:, 1].copy(),
                "density": state["density"].copy(),
                "specific_internal_energy": state["internal_energy"].copy(),
                "smoothing_length": state["smoothing_length"].copy(),
                "mass": state["masses"].copy(),
            }
            measured = corrected_c_differentials(state, reference)
            self.assertTrue(all(value == 0.0 for value in measured.values()))

    def test_terminal_differential_gates_reject_a_frozen_trajectory(self) -> None:
        def read_table(index: int) -> dict[str, np.ndarray]:
            with gzip.open(
                ORACLE_DIR / f"evolution_t{index}.csv.gz",
                "rt",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
            return {
                name: np.array([float(row[name]) for row in rows])
                for name in ("x", "y", "velocity_x", "velocity_y", "mass")
            }

        initial = read_table(0)
        terminal = read_table(6)
        weights = initial["mass"] / math.fsum(float(value) for value in initial["mass"])
        dx = (terminal["x"] - initial["x"] + 0.5) % 1.0 - 0.5
        dy = (terminal["y"] - initial["y"] + 0.5) % 1.0 - 0.5
        frozen_position_error = float(np.sum(weights * np.hypot(dx, dy)))
        frozen_velocity_error = float(
            np.sum(
                weights
                * np.hypot(
                    terminal["velocity_x"] - initial["velocity_x"],
                    terminal["velocity_y"] - initial["velocity_y"],
                )
            )
        )
        self.assertGreater(
            frozen_position_error, MAXIMUM_TERMINAL_C_POSITION_L1
        )
        self.assertGreater(
            frozen_velocity_error, MAXIMUM_TERMINAL_C_VELOCITY_L1
        )

    @unittest.skipIf(h5py is None, "h5py is required for snapshot tests")
    def test_wrong_rotation_and_radial_flow_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wrong_rotation = Path(directory) / "wrong-rotation.hdf5"
            write_synthetic_snapshot(wrong_rotation, velocity_sign=-1.0)
            rotation_metrics = comparison_metrics(read_snapshot(wrong_rotation))
            self.assertGreater(rotation_metrics["tangential_velocity_l1"], 0.2)
            self.assertGreater(rotation_metrics["velocity_vector_l1"], 0.2)
            with self.assertRaises(SnapshotError):
                enforce_limits(rotation_metrics, {"velocity_vector_l1": 1.0e-3})

            radial = Path(directory) / "radial.hdf5"
            write_synthetic_snapshot(radial, radial_velocity=0.1)
            radial_metrics = comparison_metrics(read_snapshot(radial))
            self.assertAlmostEqual(radial_metrics["radial_velocity_rms"], 0.1)
            with self.assertRaises(SnapshotError):
                enforce_limits(radial_metrics, {"radial_velocity_rms": 1.0e-3})

    @unittest.skipIf(h5py is None, "h5py is required for snapshot tests")
    def test_wrong_pressure_normalization_and_gamma_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pressure.hdf5"
            write_synthetic_snapshot(path, pressure_scale=1.1)
            metrics = comparison_metrics(read_snapshot(path))
            self.assertGreater(metrics["pressure_normalized_l1"], 0.5)
            wrong_gamma = comparison_metrics(read_snapshot(path, gamma=5.0 / 3.0))
            self.assertGreater(wrong_gamma["pressure_normalized_l1"], 1.0)

    @unittest.skipIf(h5py is None, "h5py is required for snapshot tests")
    def test_non_2d_state_and_duplicate_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nonplanar = Path(directory) / "nonplanar.hdf5"
            write_synthetic_snapshot(nonplanar, z=1.0e-8)
            with self.assertRaisesRegex(SnapshotError, "strictly two-dimensional"):
                read_snapshot(nonplanar)

            duplicate = Path(directory) / "duplicate.hdf5"
            write_synthetic_snapshot(duplicate, duplicate_id=True)
            with self.assertRaisesRegex(SnapshotError, "not unique"):
                read_snapshot(duplicate)

    def test_limit_parser_rejects_ambiguous_or_unsafe_gates(self) -> None:
        self.assertEqual(parse_limits(["density_l1=0"]), {"density_l1": 0.0})
        for invalid in (
            ["unknown=1"],
            ["density_l1=-1"],
            ["density_l1=nan"],
            ["density_l1=1", "density_l1=2"],
            ["density_l1"],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_limits(invalid)

    @unittest.skipIf(h5py is None, "h5py is required for snapshot tests")
    def test_official_ring_fixture_matches_gamma_1_4_equilibrium(self) -> None:
        path = ORACLE_DIR / "gresho_ics.hdf5"
        if not path.exists():
            self.skipTest("ignored official Gresho fixture has not been fetched")
        state = read_snapshot(path)
        check_official_initial_condition(state)
        self.assertEqual(len(state["ids"]), 4_092)
        with h5py.File(path, "r") as handle:
            header = handle["Header"].attrs
            self.assertEqual(int(header["Flag_DoublePrecision"]), 0)
            self.assertEqual(header["NumPart_Total"].tolist(), [4_092, 0, 0, 0, 0, 0])
            gas = handle["PartType0"]
            for name in (
                "Coordinates",
                "Velocities",
                "Masses",
                "Density",
                "InternalEnergy",
                "SmoothingLength",
            ):
                self.assertEqual(gas[name].dtype, np.dtype("float32"))
            self.assertEqual(gas["ParticleIDs"].dtype, np.dtype("uint32"))
        wrong_gamma = read_snapshot(path, gamma=5.0 / 3.0)
        with self.assertRaises(SnapshotError):
            check_official_initial_condition(wrong_gamma)


if __name__ == "__main__":
    unittest.main()
