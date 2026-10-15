"""Apply Angle-NSI to a non-pickled NPZ image stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ._api import angle_nsi


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ with complex images and angles_deg.")
    parser.add_argument("--output", type=Path, help="Default: INPUT_angle_nsi.npz.")
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--angle-axis", type=int, default=-1)
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--angle-key", default="angles_deg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    target = (args.output or source.with_name(source.stem + "_angle_nsi.npz")).expanduser().resolve()
    if source == target:
        parser.error("Select an output path different from the input file.")
    if target.exists() and not args.overwrite:
        parser.error("Output already exists; choose another path or pass --overwrite.")
    try:
        with np.load(source, allow_pickle=False) as archive:
            images = archive[args.image_key]
            angles = archive[args.angle_key]
            envelope = angle_nsi(images, angles, args.c, angle_axis=args.angle_axis)
            payload = {
                "angle_nsi": envelope,
                "das": np.abs(images.sum(axis=args.angle_axis)),
                "angles_deg": angles,
                "c": np.asarray(args.c),
                "angle_axis": np.asarray(args.angle_axis),
                "format_version": np.asarray(1),
            }
            for key in ("x_mm", "z_mm", "time_s"):
                if key in archive:
                    payload[key] = archive[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            np.savez_compressed(stream, **payload)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Saved DAS and Angle-NSI linear envelopes {envelope.shape} to {target}")
