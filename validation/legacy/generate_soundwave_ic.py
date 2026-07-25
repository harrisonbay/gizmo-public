#!/usr/bin/env python3
"""Generate a deterministic, linear 1-D sound wave in GIZMO HDF5 format."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np


GAMMA = 5.0 / 3.0
AMPLITUDE = 1.0e-2


def generate(path: Path, count: int, amplitude: float = AMPLITUDE) -> None:
    # Equal-mass Lagrangian samples. The displacement gives
    # rho = 1 + A sin(2 pi x) to first order in A.
    q = (np.arange(count, dtype=np.float64) + 0.5) / count
    x = np.mod(q + amplitude * np.cos(2.0 * math.pi * q) / (2.0 * math.pi), 1.0)
    order = np.argsort(x)
    x = x[order]

    phase = np.sin(2.0 * math.pi * x)
    density = 1.0 + amplitude * phase
    pressure = 1.0 / GAMMA + amplitude * phase
    velocity = amplitude * phase  # sound speed is unity
    internal_energy = pressure / ((GAMMA - 1.0) * density)

    coordinates = np.zeros((count, 3), dtype=np.float64)
    velocities = np.zeros((count, 3), dtype=np.float64)
    coordinates[:, 0] = x
    velocities[:, 0] = velocity

    particle_count = np.array([count, 0, 0, 0, 0, 0], dtype=np.uint32)
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["NumPart_ThisFile"] = particle_count
        header.attrs["NumPart_Total"] = particle_count
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        header.attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        header.attrs["Time"] = 0.0
        header.attrs["Redshift"] = 0.0
        header.attrs["BoxSize"] = 1.0
        header.attrs["NumFilesPerSnapshot"] = 1
        header.attrs["Omega0"] = 0.0
        header.attrs["OmegaLambda"] = 0.0
        header.attrs["HubbleParam"] = 1.0
        header.attrs["Flag_Sfr"] = 0
        header.attrs["Flag_Cooling"] = 0
        header.attrs["Flag_StellarAge"] = 0
        header.attrs["Flag_Metals"] = 0
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_DoublePrecision"] = 1
        header.attrs["ValidationAmplitude"] = amplitude

        gas = handle.create_group("PartType0")
        gas.create_dataset("Coordinates", data=coordinates)
        gas.create_dataset("Velocities", data=velocities)
        gas.create_dataset("ParticleIDs", data=np.arange(1, count + 1, dtype=np.uint64))
        gas.create_dataset("Masses", data=np.full(count, 1.0 / count))
        gas.create_dataset("InternalEnergy", data=internal_energy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    args = parser.parse_args()
    if args.count < 16:
        parser.error("--count must be at least 16")
    if not math.isfinite(args.amplitude) or args.amplitude <= 0.0:
        parser.error("--amplitude must be finite and positive")
    generate(args.output, args.count, args.amplitude)


if __name__ == "__main__":
    main()
