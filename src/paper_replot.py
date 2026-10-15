"""CPU rendering of the final paper's maps and saved simulation profiles.

The published Figure 1 uses peak-depth map rows for five methods and the
archived directly beamformed fine Receive-NSI profile. The Receive-NSI width
reported in the tables comes from the separate grid-convergence study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from method_names import METHODS, RECEIVE_NSI
from simulation_six_method_comparison import metric_row, plot_figure as plot_simulation
from picmus_full_phantom_comparison import plot_figure as plot_phantom
from conventional_baseline_comparison import plot_carotid_combined


def slug(method: str) -> str:
    return method.lower().replace("-", "_")


def simulation_from_cache(results: Path, output: Path | None = None) -> list[dict]:
    """Recreate Figure 1 without interpolating the coarse Receive-NSI map."""
    arrays = results / "simulation_six_method/simulation_six_method_arrays.npz"
    with np.load(arrays, allow_pickle=False) as archive:
        x_mm, z_mm = archive["map_x_mm"], archive["map_z_mm"]
        images = {m: archive["image_" + slug(m)] for m in METHODS}
        profiles = {}
        rows = []
        for method in METHODS:
            image = images[method]
            if not np.isfinite(image).all() or image.max() <= 0:
                raise ValueError(f"Invalid simulation map: {method}")
            _, peak_z = np.unravel_index(image.argmax(), image.shape)
            if method == RECEIVE_NSI:
                axis = archive["profile_axis_receive_nsi_mm"]
                profile = archive["profile_receive_nsi"]
                if np.max(np.diff(axis)) > 0.000391:
                    raise ValueError("The published Receive-NSI profile requires the direct fine grid.")
            else:
                axis, profile = x_mm, image[:, peak_z]
            profiles[method] = (axis, profile)
            rows.append(metric_row(method, image, x_mm, z_mm, axis, profile, float(z_mm[peak_z])))

        convergence = json.loads((results / "point_target/simulation_psf_fwhm_summary.json").read_text())
        receive = next(row for row in convergence["final_results"] if row["method"] == RECEIVE_NSI)
        if not receive.get("lateral_fwhm_converged"):
            raise ValueError("Receive-NSI width has not passed grid convergence.")
        next(row for row in rows if row["method"] == RECEIVE_NSI)["lateral_width_mm"] = float(receive["lateral_fwhm_mm"])
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            plot_simulation(output, images, x_mm, z_mm, profiles)
    return rows


def phantom_from_cache(results: Path, output: Path) -> None:
    """Render the six unannotated overview panels used in Figure 2."""
    path = results / "picmus_full_phantom/picmus_full_phantom_cache.npz"
    with np.load(path, allow_pickle=False) as archive:
        x_mm = archive["x_axis_m"].astype(float) * 1e3
        z_mm = archive["z_axis_m"].astype(float) * 1e3
        images = {method: archive["image_" + slug(method)] for method in METHODS}
        for method, image in images.items():
            if not np.isfinite(image).all() or image.max() <= 0:
                raise ValueError(f"Invalid phantom map: {method}")
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_phantom(output, images, x_mm, z_mm, np.empty((0, 3)), [])


def carotid_from_cache(results: Path, output: Path) -> None:
    cases = {}
    for view in ("CL", "CC"):
        path = results / f"conventional_baselines/conventional_baseline_carotid_{view}_cache.npz"
        with np.load(path, allow_pickle=False) as archive:
            images = {}
            for method in METHODS:
                key = "image_" + slug(method)
                if method == "CF-DAS" and key not in archive:
                    key = "image_receive_cf_das"
                if method == "DMAS" and key not in archive:
                    key = "image_f_dmas"
                images[method] = archive[key].copy()
            cases[view] = (images, archive["axis_x_m"].copy(), archive["axis_z_m"].copy())
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_carotid_combined(output, cases)
