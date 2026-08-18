"""
simulation_point_target.py
===========================
Single-scatterer comparison of conventional DAS, conventional (element-domain)
Null Subtraction Imaging (NSI), and angle-domain NSI.

Version 6 adds a dedicated lateral-grid convergence study to resolve the very
narrow conventional-NSI point-spread function:

* A 201 x 201 coarse grid is used only for the full-field overview. Its odd
  dimensions place the simulated target exactly on a grid node.
* A target-centred 2D grid with 0.0125 mm default spacing locates each method's
  PSF peak, measures the well-sampled axial response, and supplies the contour
  image.
* A separate narrow lateral reconstruction is evaluated at those axial peak
  positions. Its default finest spacing is 0.000390625 mm
  (0.390625 micrometres), and nested subsampling gives 0.00078125,
  0.0015625, 0.003125, and 0.00625 mm profiles.
* Lateral convergence requires at least eight samples across the final FWHM and
  changes no larger than 1% over the final two successive refinements.
* Connected main-lobe widths are reported at -6.0206, -10, and -20 dB.
* The profile figure uses the converged lateral line and the 2D-grid axial
  profile; a separate figure records width versus lateral grid spacing.

The optimized conventional NSI formulation beamforms only two independent
receive fields per plane wave. If U is the uniformly apodized field and Z is
the zero-mean receive field, the DC-offset fields are Z+cU and -Z+cU.

Requires: pymust, mach-beamform, cupy, numpy, matplotlib, and an NVIDIA GPU.
"""

import csv
import json
import os
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pymust
from mach import wavefront
from mach.io.must import linear_probe_positions, scan_grid
from mach.kernel import beamform


HALF_AMPLITUDE_DB = float(20.0 * np.log10(0.5))
WIDTH_LEVELS = (
    ("minus6_db", HALF_AMPLITUDE_DB),
    ("minus10_db", -10.0),
    ("minus20_db", -20.0),
)
MIN_RECOMMENDED_SAMPLES_PER_FWHM = 8.0
CONVERGENCE_TOLERANCE_PERCENT = 1.0
LATERAL_SUBSAMPLING_STRIDES = (16, 8, 4, 2, 1)
N_FINAL_CONVERGENCE_COMPARISONS = 2


def verify_saved_file(path):
    """Fail clearly if an expected output was not created or is empty."""

    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Expected output was not written correctly: {path}")
    print(f"Saved: {path}")
    return path


def centred_axis(centre_m, half_width_m, spacing_m):
    """Return an odd-length uniform axis that contains ``centre_m`` exactly."""

    if half_width_m <= 0.0 or spacing_m <= 0.0:
        raise ValueError("Axis half-width and spacing must both be positive.")
    n_each_side = int(round(half_width_m / spacing_m))
    if n_each_side < 2:
        raise ValueError("The requested centred axis contains too few samples.")
    axis = centre_m + np.arange(-n_each_side, n_each_side + 1) * spacing_m
    if not np.isclose(axis[n_each_side], centre_m, rtol=0.0, atol=1e-15):
        raise RuntimeError("The target was not placed on the centred grid.")
    return axis.astype(float)


def amplitude_to_db(envelope, reference=None):
    """Convert a non-negative amplitude envelope to dB."""

    envelope = np.maximum(np.asarray(envelope, dtype=float), 0.0)
    if reference is None:
        reference = float(np.max(envelope))
    else:
        reference = float(reference)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("The dB reference must be finite and strictly positive.")
    normalized = envelope / reference
    return 20.0 * np.log10(np.maximum(normalized, np.finfo(float).tiny))


def measure_connected_width(axis_m, profile, level_db, peak_index=None):
    """Measure the connected main-lobe width at an amplitude level in dB.

    Starting from the selected peak, the function walks left and right until
    the profile first falls below the requested threshold. Crossing locations
    are then estimated by linear interpolation between adjacent grid samples.
    """

    axis_m = np.asarray(axis_m, dtype=float).reshape(-1)
    profile = np.maximum(np.asarray(profile, dtype=float).reshape(-1), 0.0)
    if axis_m.size != profile.size or axis_m.size < 3:
        raise ValueError("axis_m and profile must have the same length >= 3.")
    if not np.all(np.isfinite(axis_m)) or not np.all(np.isfinite(profile)):
        raise ValueError("Width inputs must contain only finite values.")
    if not np.all(np.diff(axis_m) > 0.0):
        raise ValueError("The coordinate axis must be strictly increasing.")
    if not np.isfinite(level_db) or level_db >= 0.0:
        raise ValueError("The width level must be a finite negative dB value.")

    if peak_index is None:
        peak_index = int(np.argmax(profile))
    else:
        peak_index = int(peak_index)
    if peak_index < 0 or peak_index >= profile.size:
        raise IndexError("peak_index lies outside the supplied profile.")

    peak_value = float(profile[peak_index])
    if peak_value <= 0.0:
        raise ValueError("Cannot measure a width from a zero-valued profile.")
    threshold_ratio = float(10.0 ** (level_db / 20.0))
    threshold_value = peak_value * threshold_ratio

    left_outside = peak_index
    while left_outside > 0 and profile[left_outside] >= threshold_value:
        left_outside -= 1
    if left_outside == 0 and profile[left_outside] >= threshold_value:
        raise ValueError("No threshold crossing was found on the left side.")

    right_outside = peak_index
    while (
        right_outside < profile.size - 1
        and profile[right_outside] >= threshold_value
    ):
        right_outside += 1
    if (
        right_outside == profile.size - 1
        and profile[right_outside] >= threshold_value
    ):
        raise ValueError("No threshold crossing was found on the right side.")

    def interpolate_crossing(index_1, index_2):
        coordinate_1 = float(axis_m[index_1])
        coordinate_2 = float(axis_m[index_2])
        value_1 = float(profile[index_1])
        value_2 = float(profile[index_2])
        denominator = value_2 - value_1
        if denominator == 0.0:
            return 0.5 * (coordinate_1 + coordinate_2)
        fraction = (threshold_value - value_1) / denominator
        return coordinate_1 + fraction * (coordinate_2 - coordinate_1)

    left_crossing_m = interpolate_crossing(left_outside, left_outside + 1)
    right_crossing_m = interpolate_crossing(right_outside - 1, right_outside)
    return {
        "peak_index": peak_index,
        "peak_coordinate_m": float(axis_m[peak_index]),
        "peak_value": peak_value,
        "level_db": float(level_db),
        "threshold_value": float(threshold_value),
        "left_crossing_m": float(left_crossing_m),
        "right_crossing_m": float(right_crossing_m),
        "width_m": float(right_crossing_m - left_crossing_m),
    }


def find_central_psf_peak(
    envelope,
    x_axis,
    z_axis,
    target_x_m,
    target_z_m,
    x_half_width_m,
    z_half_width_m,
):
    """Locate a method's PSF peak only inside the target search window."""

    x_indices = np.flatnonzero(
        (x_axis >= target_x_m - x_half_width_m)
        & (x_axis <= target_x_m + x_half_width_m)
    )
    z_indices = np.flatnonzero(
        (z_axis >= target_z_m - z_half_width_m)
        & (z_axis <= target_z_m + z_half_width_m)
    )
    if x_indices.size == 0 or z_indices.size == 0:
        raise ValueError("The central peak-search window misses the image grid.")
    central_window = envelope[np.ix_(x_indices, z_indices)]
    local_x_index, local_z_index = np.unravel_index(
        int(np.argmax(central_window)), central_window.shape
    )
    return int(x_indices[local_x_index]), int(z_indices[local_z_index])


def measure_psf(
    method,
    envelope,
    x_axis,
    z_axis,
    target_x_m,
    target_z_m,
    search_x_half_width_m,
    search_z_half_width_m,
):
    """Measure lateral and axial connected widths through the central peak."""

    peak_x_index, peak_z_index = find_central_psf_peak(
        envelope,
        x_axis,
        z_axis,
        target_x_m,
        target_z_m,
        search_x_half_width_m,
        search_z_half_width_m,
    )
    lateral_profile = envelope[:, peak_z_index]
    axial_profile = envelope[peak_x_index, :]

    lateral_measurements = {}
    axial_measurements = {}
    for level_name, level_db in WIDTH_LEVELS:
        lateral_measurements[level_name] = measure_connected_width(
            x_axis,
            lateral_profile,
            level_db,
            peak_index=peak_x_index,
        )
        axial_measurements[level_name] = measure_connected_width(
            z_axis,
            axial_profile,
            level_db,
            peak_index=peak_z_index,
        )

    grid_spacing_x_mm = float(np.median(np.diff(x_axis)) * 1e3)
    grid_spacing_z_mm = float(np.median(np.diff(z_axis)) * 1e3)
    lateral_fwhm_mm = (
        lateral_measurements["minus6_db"]["width_m"] * 1e3
    )
    axial_fwhm_mm = axial_measurements["minus6_db"]["width_m"] * 1e3
    lateral_samples = lateral_fwhm_mm / grid_spacing_x_mm
    axial_samples = axial_fwhm_mm / grid_spacing_z_mm

    result = {
        "method": method,
        "grid_spacing_x_mm": grid_spacing_x_mm,
        "grid_spacing_z_mm": grid_spacing_z_mm,
        "peak_x_mm": float(x_axis[peak_x_index] * 1e3),
        "peak_z_mm": float(z_axis[peak_z_index] * 1e3),
        "peak_amplitude": float(envelope[peak_x_index, peak_z_index]),
        "lateral_fwhm_mm": float(lateral_fwhm_mm),
        "axial_fwhm_mm": float(axial_fwhm_mm),
        "lateral_width_minus10_db_mm": float(
            lateral_measurements["minus10_db"]["width_m"] * 1e3
        ),
        "axial_width_minus10_db_mm": float(
            axial_measurements["minus10_db"]["width_m"] * 1e3
        ),
        "lateral_width_minus20_db_mm": float(
            lateral_measurements["minus20_db"]["width_m"] * 1e3
        ),
        "axial_width_minus20_db_mm": float(
            axial_measurements["minus20_db"]["width_m"] * 1e3
        ),
        "lateral_minus6_left_crossing_mm": float(
            lateral_measurements["minus6_db"]["left_crossing_m"] * 1e3
        ),
        "lateral_minus6_right_crossing_mm": float(
            lateral_measurements["minus6_db"]["right_crossing_m"] * 1e3
        ),
        "axial_minus6_left_crossing_mm": float(
            axial_measurements["minus6_db"]["left_crossing_m"] * 1e3
        ),
        "axial_minus6_right_crossing_mm": float(
            axial_measurements["minus6_db"]["right_crossing_m"] * 1e3
        ),
        "lateral_samples_per_fwhm": float(lateral_samples),
        "axial_samples_per_fwhm": float(axial_samples),
        "lateral_fwhm_sampling_ok": bool(
            lateral_samples >= MIN_RECOMMENDED_SAMPLES_PER_FWHM
        ),
        "axial_fwhm_sampling_ok": bool(
            axial_samples >= MIN_RECOMMENDED_SAMPLES_PER_FWHM
        ),
    }
    profiles = {
        "lateral": lateral_profile,
        "axial": axial_profile,
        "peak_x_index": peak_x_index,
        "peak_z_index": peak_z_index,
        "lateral_measurements": lateral_measurements,
        "axial_measurements": axial_measurements,
    }
    return result, profiles


def measure_lateral_profile(
    method,
    envelope,
    x_axis,
    evaluation_z_m,
    z_index,
    target_x_m,
    search_x_half_width_m,
):
    """Measure connected lateral widths on one selected axial line."""

    envelope = np.asarray(envelope, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float).reshape(-1)
    if envelope.ndim != 2 or envelope.shape[0] != x_axis.size:
        raise ValueError("The lateral envelope must have shape (len(x_axis), nz).")
    z_index = int(z_index)
    if z_index < 0 or z_index >= envelope.shape[1]:
        raise IndexError("z_index lies outside the supplied lateral reconstruction.")

    profile = np.maximum(envelope[:, z_index], 0.0)
    search_indices = np.flatnonzero(
        (x_axis >= target_x_m - search_x_half_width_m)
        & (x_axis <= target_x_m + search_x_half_width_m)
    )
    if search_indices.size == 0:
        raise ValueError("The lateral peak-search window misses the x axis.")
    peak_index = int(
        search_indices[int(np.argmax(profile[search_indices]))]
    )

    measurements = {
        level_name: measure_connected_width(
            x_axis,
            profile,
            level_db,
            peak_index=peak_index,
        )
        for level_name, level_db in WIDTH_LEVELS
    }
    spacing_mm = float(np.median(np.diff(x_axis)) * 1e3)
    lateral_fwhm_mm = float(
        measurements["minus6_db"]["width_m"] * 1e3
    )
    samples_per_fwhm = lateral_fwhm_mm / spacing_mm
    result = {
        "method": method,
        "grid_spacing_x_mm": spacing_mm,
        "evaluation_z_mm": float(evaluation_z_m * 1e3),
        "peak_x_mm": float(x_axis[peak_index] * 1e3),
        "peak_amplitude": float(profile[peak_index]),
        "lateral_fwhm_mm": lateral_fwhm_mm,
        "lateral_width_minus10_db_mm": float(
            measurements["minus10_db"]["width_m"] * 1e3
        ),
        "lateral_width_minus20_db_mm": float(
            measurements["minus20_db"]["width_m"] * 1e3
        ),
        "lateral_minus6_left_crossing_mm": float(
            measurements["minus6_db"]["left_crossing_m"] * 1e3
        ),
        "lateral_minus6_right_crossing_mm": float(
            measurements["minus6_db"]["right_crossing_m"] * 1e3
        ),
        "lateral_samples_per_fwhm": float(samples_per_fwhm),
        "lateral_fwhm_sampling_ok": bool(
            samples_per_fwhm >= MIN_RECOMMENDED_SAMPLES_PER_FWHM
        ),
    }
    details = {
        "profile": profile,
        "peak_x_index": peak_index,
        "measurements": measurements,
    }
    return result, details


def relative_change_percent(finer_value, coarser_value):
    """Return absolute relative change, using the finer value as reference."""

    finer_value = float(finer_value)
    coarser_value = float(coarser_value)
    if not np.isfinite(finer_value) or finer_value <= 0.0:
        raise ValueError("The finer-grid value must be finite and positive.")
    return float(100.0 * abs(finer_value - coarser_value) / finer_value)


def build_lateral_convergence_checks(results_by_stride, strides):
    """Compare adjacent nested grids and assess the final two refinements."""

    strides = tuple(int(stride) for stride in strides)
    if len(strides) < N_FINAL_CONVERGENCE_COMPARISONS + 1:
        raise ValueError("Too few nested grids for the requested convergence test.")
    if any(stride <= 0 for stride in strides):
        raise ValueError("Subsampling strides must be positive integers.")

    methods = [row["method"] for row in results_by_stride[strides[-1]]]
    adjacent_checks = []
    checks_by_method = {method: [] for method in methods}
    for coarser_stride, finer_stride in zip(strides[:-1], strides[1:]):
        coarser_lookup = {
            row["method"]: row for row in results_by_stride[coarser_stride]
        }
        finer_lookup = {
            row["method"]: row for row in results_by_stride[finer_stride]
        }
        for method in methods:
            coarser = coarser_lookup[method]
            finer = finer_lookup[method]
            change_percent = relative_change_percent(
                finer["lateral_fwhm_mm"], coarser["lateral_fwhm_mm"]
            )
            check = {
                "method": method,
                "coarser_spacing_mm": float(coarser["grid_spacing_x_mm"]),
                "finer_spacing_mm": float(finer["grid_spacing_x_mm"]),
                "coarser_lateral_fwhm_mm": float(
                    coarser["lateral_fwhm_mm"]
                ),
                "finer_lateral_fwhm_mm": float(finer["lateral_fwhm_mm"]),
                "lateral_fwhm_relative_change_percent": change_percent,
                "finer_samples_per_fwhm": float(
                    finer["lateral_samples_per_fwhm"]
                ),
                "comparison_within_tolerance": bool(
                    change_percent <= CONVERGENCE_TOLERANCE_PERCENT
                ),
            }
            adjacent_checks.append(check)
            checks_by_method[method].append(check)

    final_lookup = {
        row["method"]: row for row in results_by_stride[strides[-1]]
    }
    final_checks = []
    for method in methods:
        decisive_checks = checks_by_method[method][
            -N_FINAL_CONVERGENCE_COMPARISONS:
        ]
        final_result = final_lookup[method]
        final_checks.append(
            {
                "method": method,
                "finest_spacing_mm": float(
                    final_result["grid_spacing_x_mm"]
                ),
                "finest_lateral_fwhm_mm": float(
                    final_result["lateral_fwhm_mm"]
                ),
                "finest_samples_per_fwhm": float(
                    final_result["lateral_samples_per_fwhm"]
                ),
                "final_two_relative_changes_percent": [
                    float(check["lateral_fwhm_relative_change_percent"])
                    for check in decisive_checks
                ],
                "sampling_requirement_met": bool(
                    final_result["lateral_fwhm_sampling_ok"]
                ),
                "final_two_refinements_within_tolerance": bool(
                    all(
                        check["comparison_within_tolerance"]
                        for check in decisive_checks
                    )
                ),
                "lateral_fwhm_converged": bool(
                    final_result["lateral_fwhm_sampling_ok"]
                    and all(
                        check["comparison_within_tolerance"]
                        for check in decisive_checks
                    )
                ),
            }
        )
    return adjacent_checks, final_checks


# ============================================================================
# 1) Simulation parameters and RF generation
# ============================================================================
xs = np.array([0.0]) * 1e-2
zs = np.array([2.0]) * 1e-2
RC = np.ones(xs.shape)

if xs.size != 1 or zs.size != 1:
    raise ValueError("The v6 PSF diagnostics require one point scatterer.")

target_x_m = float(xs[0])
target_z_m = float(zs[0])

param = pymust.getparam("L11-5v")
n_pw = 17
angles_deg = np.linspace(-4.0, 4.0, n_pw)
tilt = np.deg2rad(angles_deg)
if not np.allclose(angles_deg, -angles_deg[::-1]):
    raise ValueError("Angle-domain NSI requires symmetric steering angles.")

txdel = [pymust.txdelay(param, angle) for angle in tilt]
param.fs = 4 * param.fc
sound_speed = float(param.get("c", 1540.0))
t0 = float(param.get("t0", 0.0))

RF = []
for delay in txdel:
    rf_data, _ = pymust.simus(xs, zs, RC, delay, param)
    RF.append(rf_data)
IQ = [
    pymust.rf2iq(rf_data, param.fs, param.fc).astype(np.complex64)
    for rf_data in RF
]

element_positions = linear_probe_positions(param.Nelements, param.pitch)
rx_coords_gpu = cp.asarray(element_positions, dtype=np.float32)
f_number = 0.0
dc = 0.05

apo_null = np.ones(param.Nelements, dtype=np.float32)
apo_null[: param.Nelements // 2] = -1.0


def reconstruct_grid(x_axis, z_axis, label):
    """Beamform the three methods on one Cartesian grid and return CPU arrays."""

    y_axis = np.array([0.0])
    grid_points = scan_grid(x_axis, y_axis, z_axis)
    grid_shape = (len(x_axis), len(z_axis))
    scan_coords_gpu = cp.asarray(grid_points, dtype=np.float32)

    raw_angle_stack = cp.zeros(
        (*grid_shape, n_pw), dtype=cp.complex64
    )
    uniform_sum = cp.zeros(grid_shape, dtype=cp.complex64)
    receive_null_sum = cp.zeros(grid_shape, dtype=cp.complex64)

    cp.cuda.Stream.null.synchronize()
    start_time = time.perf_counter()
    for angle_index, angle_deg in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle_deg)
        direction = np.array(
            [np.sin(angle_rad), 0.0, np.cos(angle_rad)], dtype=np.float32
        )
        tx_arrivals_s = (
            wavefront.plane(
                origin_m=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                points_m=grid_points,
                direction=direction,
            )
            / sound_speed
        )
        tx_arrivals_gpu = cp.asarray(tx_arrivals_s, dtype=np.float32)

        iq_angle = IQ[angle_index]
        frames = np.stack(
            [iq_angle, iq_angle * apo_null], axis=-1
        ).astype(np.complex64)
        channel_data_gpu = cp.asarray(
            np.ascontiguousarray(frames.transpose(1, 0, 2))
        )
        result_gpu = beamform(
            channel_data=channel_data_gpu,
            rx_coords_m=rx_coords_gpu,
            scan_coords_m=scan_coords_gpu,
            tx_wave_arrivals_s=tx_arrivals_gpu,
            f_number=f_number,
            rx_start_s=t0,
            sampling_freq_hz=float(param.fs),
            sound_speed_m_s=sound_speed,
            modulation_freq_hz=float(param.fc),
            tukey_alpha=0.0,
        )
        raw_image = result_gpu[:, 0].reshape(grid_shape)
        null_image = result_gpu[:, 1].reshape(grid_shape)
        raw_angle_stack[:, :, angle_index] = raw_image
        uniform_sum += raw_image
        receive_null_sum += null_image

    angular_null_weights = cp.asarray(
        np.sign(angles_deg), dtype=cp.float32
    )
    if not np.isclose(float(cp.sum(angular_null_weights).get()), 0.0):
        raise ValueError("Angular zero-mean weights must sum to zero.")

    angular_null_sum = cp.sum(
        raw_angle_stack * angular_null_weights.reshape(1, 1, -1), axis=2
    )
    angular_dc_1 = angular_null_sum + dc * uniform_sum
    angular_dc_2 = -angular_null_sum + dc * uniform_sum
    angular_envelope = (
        (cp.abs(angular_dc_1) + cp.abs(angular_dc_2)) * 0.5
        - cp.abs(angular_null_sum)
    )

    conventional_dc_1 = receive_null_sum + dc * uniform_sum
    conventional_dc_2 = -receive_null_sum + dc * uniform_sum
    conventional_envelope = (
        (cp.abs(conventional_dc_1) + cp.abs(conventional_dc_2)) * 0.5
        - cp.abs(receive_null_sum)
    )

    cp.cuda.Stream.null.synchronize()
    elapsed_seconds = time.perf_counter() - start_time
    envelopes_cpu = {
        "DAS": cp.asnumpy(cp.abs(uniform_sum)),
        "Conventional NSI": cp.asnumpy(
            cp.maximum(conventional_envelope, 0.0)
        ),
        "Angular NSI": cp.asnumpy(cp.maximum(angular_envelope, 0.0)),
    }
    print(
        f"{label}: {len(x_axis)} x {len(z_axis)} points, "
        f"{elapsed_seconds:.3f} s"
    )

    del (
        scan_coords_gpu,
        raw_angle_stack,
        uniform_sum,
        receive_null_sum,
        angular_null_sum,
        angular_dc_1,
        angular_dc_2,
        angular_envelope,
        conventional_dc_1,
        conventional_dc_2,
        conventional_envelope,
    )
    cp.get_default_memory_pool().free_all_blocks()
    return envelopes_cpu, float(elapsed_seconds)


# ============================================================================
# 2) Coarse full-field overview (not used for quantitative PSF widths)
# ============================================================================
overview_x = np.linspace(-2e-2, 2e-2, 201)
overview_z = np.linspace(0.0, 4e-2, 201)
overview_envelopes, overview_time_seconds = reconstruct_grid(
    overview_x, overview_z, "Full-field overview reconstruction"
)

output_dir = Path(
    os.environ.get(
        "NSI_OUTPUT_DIR",
        Path(__file__).resolve().parents[1]
        / "results"
        / "generated"
        / "point_target",
    )
).expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

fig_overview, overview_axes = plt.subplots(
    1, 3, figsize=(12, 5), constrained_layout=True
)
overview_extent = [
    overview_x.min() * 1e2,
    overview_x.max() * 1e2,
    overview_z.max() * 1e2,
    overview_z.min() * 1e2,
]
for axis, (method, envelope) in zip(
    overview_axes, overview_envelopes.items()
):
    image = axis.imshow(
        amplitude_to_db(envelope).T,
        cmap="gray",
        vmin=-40.0,
        vmax=0.0,
        extent=overview_extent,
        aspect="equal",
        interpolation="nearest",
    )
    axis.set_xlabel("x [cm]")
    axis.set_ylabel("z [cm]")
    axis.set_title(method)
fig_overview.colorbar(
    image,
    ax=overview_axes,
    shrink=0.8,
    label="Normalized envelope [dB]",
)
overview_path = output_dir / "single_scatterer_comparison.png"
fig_overview.savefig(overview_path, dpi=300, bbox_inches="tight")
verify_saved_file(overview_path)


# ============================================================================
# 3) Target-centred 2D reconstruction for axial PSF and peak localization
# ============================================================================
fine_spacing_mm = float(
    os.environ.get("NSI_PSF_2D_GRID_SPACING_MM", "0.0125")
)
fine_spacing_m = fine_spacing_mm * 1e-3
lateral_plot_half_width_mm = 6.0
axial_plot_half_width_mm = 3.0
fine_x = centred_axis(
    target_x_m, lateral_plot_half_width_mm * 1e-3, fine_spacing_m
)
fine_z = centred_axis(
    target_z_m, axial_plot_half_width_mm * 1e-3, fine_spacing_m
)
fine_envelopes, fine_time_seconds = reconstruct_grid(
    fine_x, fine_z, "Target-centred 2D PSF reconstruction"
)

target_x_index = int(np.argmin(np.abs(fine_x - target_x_m)))
target_z_index = int(np.argmin(np.abs(fine_z - target_z_m)))
if fine_x[target_x_index] != target_x_m or fine_z[target_z_index] != target_z_m:
    raise RuntimeError("The fine grid does not contain the exact target location.")

search_x_half_width_m = 1.0e-3
search_z_half_width_m = 1.5e-3
two_d_subsampling_strides = (4, 2, 1)
two_d_convergence_rows = []
two_d_results_by_stride = {}
profiles_by_method_2d = {}

for stride in two_d_subsampling_strides:
    x_slice_start = target_x_index % stride
    z_slice_start = target_z_index % stride
    sampled_x = fine_x[x_slice_start::stride]
    sampled_z = fine_z[z_slice_start::stride]
    sampled_results = []
    for method, fine_envelope in fine_envelopes.items():
        sampled_envelope = fine_envelope[
            x_slice_start::stride, z_slice_start::stride
        ]
        result, profiles = measure_psf(
            method,
            sampled_envelope,
            sampled_x,
            sampled_z,
            target_x_m,
            target_z_m,
            search_x_half_width_m,
            search_z_half_width_m,
        )
        result["subsampling_stride"] = int(stride)
        sampled_results.append(result)
        two_d_convergence_rows.append(result.copy())
        if stride == 1:
            profiles_by_method_2d[method] = profiles
    two_d_results_by_stride[stride] = sampled_results

fine_2d_results = two_d_results_by_stride[1]
axial_adjacent_grid_checks = []
axial_checks_by_method = {
    result["method"]: [] for result in fine_2d_results
}
for coarser_stride, finer_stride in zip(
    two_d_subsampling_strides[:-1], two_d_subsampling_strides[1:]
):
    coarser_lookup = {
        row["method"]: row
        for row in two_d_results_by_stride[coarser_stride]
    }
    finer_lookup = {
        row["method"]: row
        for row in two_d_results_by_stride[finer_stride]
    }
    for method in axial_checks_by_method:
        coarser = coarser_lookup[method]
        finer = finer_lookup[method]
        change_percent = relative_change_percent(
            finer["axial_fwhm_mm"], coarser["axial_fwhm_mm"]
        )
        check = {
            "method": method,
            "coarser_spacing_mm": float(coarser["grid_spacing_z_mm"]),
            "finer_spacing_mm": float(finer["grid_spacing_z_mm"]),
            "coarser_axial_fwhm_mm": float(coarser["axial_fwhm_mm"]),
            "finer_axial_fwhm_mm": float(finer["axial_fwhm_mm"]),
            "axial_fwhm_relative_change_percent": change_percent,
            "comparison_within_tolerance": bool(
                change_percent <= CONVERGENCE_TOLERANCE_PERCENT
            ),
        }
        axial_adjacent_grid_checks.append(check)
        axial_checks_by_method[method].append(check)

axial_final_checks = []
fine_2d_lookup = {row["method"]: row for row in fine_2d_results}
for method, method_checks in axial_checks_by_method.items():
    final_2d_result = fine_2d_lookup[method]
    axial_final_checks.append(
        {
            "method": method,
            "finest_spacing_mm": float(final_2d_result["grid_spacing_z_mm"]),
            "finest_axial_fwhm_mm": float(final_2d_result["axial_fwhm_mm"]),
            "finest_samples_per_fwhm": float(
                final_2d_result["axial_samples_per_fwhm"]
            ),
            "final_two_relative_changes_percent": [
                float(check["axial_fwhm_relative_change_percent"])
                for check in method_checks[-N_FINAL_CONVERGENCE_COMPARISONS:]
            ],
            "axial_fwhm_converged": bool(
                final_2d_result["axial_fwhm_sampling_ok"]
                and all(
                    check["comparison_within_tolerance"]
                    for check in method_checks[
                        -N_FINAL_CONVERGENCE_COMPARISONS:
                    ]
                )
            ),
        }
    )


# ============================================================================
# 4) Dedicated micrometre-scale lateral-grid convergence study
# ============================================================================
lateral_finest_spacing_mm = float(
    os.environ.get("NSI_LATERAL_FINEST_SPACING_MM", "0.000390625")
)
lateral_half_width_mm = float(
    os.environ.get("NSI_LATERAL_HALF_WIDTH_MM", "0.75")
)
lateral_finest_spacing_m = lateral_finest_spacing_mm * 1e-3
lateral_x = centred_axis(
    target_x_m,
    lateral_half_width_mm * 1e-3,
    lateral_finest_spacing_m,
)
lateral_target_x_index = int(np.argmin(np.abs(lateral_x - target_x_m)))
if lateral_x[lateral_target_x_index] != target_x_m:
    raise RuntimeError("The lateral convergence grid misses the exact target.")

peak_z_indices = {
    method: int(profiles_by_method_2d[method]["peak_z_index"])
    for method in profiles_by_method_2d
}
lateral_z = np.unique(
    np.asarray(
        [fine_z[index] for index in peak_z_indices.values()], dtype=float
    )
)
lateral_envelopes, lateral_time_seconds = reconstruct_grid(
    lateral_x,
    lateral_z,
    "Micrometre-scale lateral PSF reconstruction",
)

lateral_convergence_rows = []
lateral_results_by_stride = {}
lateral_profiles_by_method = {}
for stride in LATERAL_SUBSAMPLING_STRIDES:
    x_slice_start = lateral_target_x_index % stride
    sampled_x = lateral_x[x_slice_start::stride]
    sampled_results = []
    for method, finest_envelope in lateral_envelopes.items():
        evaluation_z_m = float(fine_z[peak_z_indices[method]])
        z_index = int(np.argmin(np.abs(lateral_z - evaluation_z_m)))
        if not np.isclose(
            lateral_z[z_index], evaluation_z_m, rtol=0.0, atol=1e-15
        ):
            raise RuntimeError("The lateral reconstruction misses a PSF peak z.")
        sampled_envelope = finest_envelope[x_slice_start::stride, :]
        result, profile_details = measure_lateral_profile(
            method,
            sampled_envelope,
            sampled_x,
            evaluation_z_m,
            z_index,
            target_x_m,
            search_x_half_width_m,
        )
        result["subsampling_stride"] = int(stride)
        sampled_results.append(result)
        lateral_convergence_rows.append(result.copy())
        if stride == 1:
            lateral_profiles_by_method[method] = profile_details
    lateral_results_by_stride[stride] = sampled_results

lateral_adjacent_grid_checks, lateral_final_checks = (
    build_lateral_convergence_checks(
        lateral_results_by_stride, LATERAL_SUBSAMPLING_STRIDES
    )
)
lateral_final_lookup = {
    row["method"]: row for row in lateral_results_by_stride[1]
}
lateral_check_lookup = {
    row["method"]: row for row in lateral_final_checks
}
axial_check_lookup = {row["method"]: row for row in axial_final_checks}

final_results = []
profiles_by_method = {}
for two_d_result in fine_2d_results:
    method = two_d_result["method"]
    lateral_result = lateral_final_lookup[method]
    combined = two_d_result.copy()
    for key in (
        "grid_spacing_x_mm",
        "lateral_fwhm_mm",
        "lateral_width_minus10_db_mm",
        "lateral_width_minus20_db_mm",
        "lateral_minus6_left_crossing_mm",
        "lateral_minus6_right_crossing_mm",
        "lateral_samples_per_fwhm",
        "lateral_fwhm_sampling_ok",
    ):
        combined[key] = lateral_result[key]
    combined["lateral_peak_x_mm"] = lateral_result["peak_x_mm"]
    combined["lateral_profile_z_mm"] = lateral_result["evaluation_z_mm"]
    combined["lateral_fwhm_converged"] = lateral_check_lookup[method][
        "lateral_fwhm_converged"
    ]
    combined["lateral_final_two_relative_changes_percent"] = (
        lateral_check_lookup[method]["final_two_relative_changes_percent"]
    )
    combined["axial_fwhm_converged"] = axial_check_lookup[method][
        "axial_fwhm_converged"
    ]
    final_results.append(combined)

    lateral_profile_details = lateral_profiles_by_method[method]
    axial_profile_details = profiles_by_method_2d[method]
    profiles_by_method[method] = {
        "lateral": lateral_profile_details["profile"],
        "axial": axial_profile_details["axial"],
        "peak_x_index": lateral_profile_details["peak_x_index"],
        "peak_z_index": axial_profile_details["peak_z_index"],
        "lateral_measurements": lateral_profile_details["measurements"],
        "axial_measurements": axial_profile_details["axial_measurements"],
    }


# ============================================================================
# 5) Print and save quantitative results
# ============================================================================
print("\n" + "=" * 119)
print("Connected PSF widths with dedicated lateral-grid convergence")
print("=" * 119)
print(
    "Method             dx lateral   Lateral -6 dB   Axial -6 dB   "
    "Lat. samples   Lat. conv.   Ax. conv."
)
print("-" * 119)
for result in final_results:
    print(
        f"{result['method']:<18s} "
        f"{result['grid_spacing_x_mm']:10.7f} mm "
        f"{result['lateral_fwhm_mm']:13.6f} mm "
        f"{result['axial_fwhm_mm']:12.6f} mm "
        f"{result['lateral_samples_per_fwhm']:12.2f} "
        f"{str(result['lateral_fwhm_converged']):>11s} "
        f"{str(result['axial_fwhm_converged']):>10s}"
    )
print("=" * 119)

print("\nNested lateral FWHM values:")
for method in ("DAS", "Conventional NSI", "Angular NSI"):
    method_rows = [
        row for row in lateral_convergence_rows if row["method"] == method
    ]
    values = ", ".join(
        f"dx={row['grid_spacing_x_mm'] * 1e3:.4f} um: "
        f"{row['lateral_fwhm_mm'] * 1e3:.3f} um"
        for row in method_rows
    )
    print(f"  {method}: {values}")

for result in final_results:
    if not result["lateral_fwhm_sampling_ok"]:
        print(
            "WARNING: "
            f"{result['method']} lateral FWHM has only "
            f"{result['lateral_samples_per_fwhm']:.2f} fine-grid samples."
        )
    if not result["axial_fwhm_sampling_ok"]:
        print(
            "WARNING: "
            f"{result['method']} axial FWHM has only "
            f"{result['axial_samples_per_fwhm']:.2f} fine-grid samples."
        )
    if not result["lateral_fwhm_converged"]:
        print(
            "WARNING: lateral FWHM did not satisfy the sampling and final-two-"
            f"refinement criteria for {result['method']}."
        )
    if not result["axial_fwhm_converged"]:
        print(
            "WARNING: axial FWHM did not satisfy the 2D-grid convergence "
            f"criteria for {result['method']}."
        )

fwhm_csv_path = output_dir / "simulation_psf_fwhm.csv"
with fwhm_csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(final_results[0].keys()))
    writer.writeheader()
    writer.writerows(final_results)
verify_saved_file(fwhm_csv_path)

convergence_csv_path = output_dir / "simulation_psf_grid_convergence.csv"
with convergence_csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=list(lateral_convergence_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(lateral_convergence_rows)
verify_saved_file(convergence_csv_path)

convergence_checks_csv_path = (
    output_dir / "simulation_psf_grid_convergence_checks.csv"
)
with convergence_checks_csv_path.open(
    "w", newline="", encoding="utf-8"
) as stream:
    writer = csv.DictWriter(
        stream, fieldnames=list(lateral_adjacent_grid_checks[0].keys())
    )
    writer.writeheader()
    writer.writerows(lateral_adjacent_grid_checks)
verify_saved_file(convergence_checks_csv_path)

json_path = output_dir / "simulation_psf_fwhm_summary.json"
with json_path.open("w", encoding="utf-8") as stream:
    json.dump(
        {
            "definition": (
                "Connected main-lobe envelope width around each method's "
                "central target peak"
            ),
            "width_levels_db": {
                name: float(level_db) for name, level_db in WIDTH_LEVELS
            },
            "simulated_target_mm": {
                "x": target_x_m * 1e3,
                "z": target_z_m * 1e3,
            },
            "overview_grid": {
                "shape": [int(overview_x.size), int(overview_z.size)],
                "spacing_x_mm": float(np.median(np.diff(overview_x)) * 1e3),
                "spacing_z_mm": float(np.median(np.diff(overview_z)) * 1e3),
                "used_for_psf_measurement": False,
            },
            "psf_2d_grid": {
                "shape": [int(fine_x.size), int(fine_z.size)],
                "spacing_x_mm": float(np.median(np.diff(fine_x)) * 1e3),
                "spacing_z_mm": float(np.median(np.diff(fine_z)) * 1e3),
                "x_limits_mm": [float(fine_x[0] * 1e3), float(fine_x[-1] * 1e3)],
                "z_limits_mm": [float(fine_z[0] * 1e3), float(fine_z[-1] * 1e3)],
                "target_is_grid_node": True,
                "purpose": (
                    "Axial PSF, axial convergence, peak localization, and "
                    "contour visualization"
                ),
            },
            "lateral_convergence_grid": {
                "shape": [int(lateral_x.size), int(lateral_z.size)],
                "finest_spacing_x_mm": float(
                    np.median(np.diff(lateral_x)) * 1e3
                ),
                "nested_spacings_x_mm": [
                    float(lateral_finest_spacing_mm * stride)
                    for stride in LATERAL_SUBSAMPLING_STRIDES
                ],
                "x_limits_mm": [
                    float(lateral_x[0] * 1e3),
                    float(lateral_x[-1] * 1e3),
                ],
                "evaluation_z_mm": [float(value * 1e3) for value in lateral_z],
                "target_is_x_grid_node": True,
                "purpose": "Quantitative lateral PSF and grid convergence",
            },
            "convergence_method": (
                "Nested subsampling of one finest-grid reconstruction; the "
                "lateral result must meet the sampling requirement and the "
                "tolerance over the final two successive refinements"
            ),
            "convergence_tolerance_percent": CONVERGENCE_TOLERANCE_PERCENT,
            "number_of_final_comparisons_required": (
                N_FINAL_CONVERGENCE_COMPARISONS
            ),
            "minimum_recommended_samples_per_fwhm": (
                MIN_RECOMMENDED_SAMPLES_PER_FWHM
            ),
            "peak_search_window_mm": {
                "x_half_width": search_x_half_width_m * 1e3,
                "z_half_width": search_z_half_width_m * 1e3,
            },
            "beamforming_time_seconds": {
                "overview": overview_time_seconds,
                "psf_2d": fine_time_seconds,
                "lateral_convergence": lateral_time_seconds,
            },
            "final_results": final_results,
            "lateral_grid_results": lateral_convergence_rows,
            "lateral_adjacent_grid_checks": lateral_adjacent_grid_checks,
            "final_lateral_convergence_checks": lateral_final_checks,
            "axial_2d_grid_results": two_d_convergence_rows,
            "axial_adjacent_grid_checks": axial_adjacent_grid_checks,
            "final_axial_convergence_checks": axial_final_checks,
        },
        stream,
        indent=2,
        allow_nan=False,
    )
    stream.write("\n")
verify_saved_file(json_path)


# ============================================================================
# 6) Fine-grid images with -6 dB contours
# ============================================================================
result_lookup = {result["method"]: result for result in final_results}
fig_fine, fine_axes = plt.subplots(
    1, 3, figsize=(14, 5), constrained_layout=True
)
fine_extent = [
    fine_x.min() * 1e3,
    fine_x.max() * 1e3,
    fine_z.max() * 1e3,
    fine_z.min() * 1e3,
]
for axis, (method, envelope) in zip(fine_axes, fine_envelopes.items()):
    reference = result_lookup[method]["peak_amplitude"]
    image_db = amplitude_to_db(envelope, reference=reference)
    fine_image = axis.imshow(
        image_db.T,
        cmap="gray",
        vmin=-40.0,
        vmax=0.0,
        extent=fine_extent,
        aspect="equal",
        interpolation="nearest",
    )
    # axis.contour(
    #     fine_x * 1e3,
    #     fine_z * 1e3,
    #     image_db.T,
    #     levels=[HALF_AMPLITUDE_DB],
    #     colors=["tab:orange"],
    #     linewidths=1.0,
    # )
    # axis.plot(
    #     result_lookup[method]["peak_x_mm"],
    #     result_lookup[method]["peak_z_mm"],
    #     marker="+",
    #     color="tab:cyan",
    #     markersize=7,
    #     markeredgewidth=1.2,
    # )
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("z [mm]")
    axis.set_title(method)
fig_fine.colorbar(
    fine_image,
    ax=fine_axes,
    shrink=0.8,
    label="Intensity [dB]",
)
fine_path = output_dir / "single_scatterer_fine_psf_contours.png"
fig_fine.savefig(fine_path, dpi=300, bbox_inches="tight")
verify_saved_file(fine_path)


# ============================================================================
# 7) Lateral-grid convergence figure
# ============================================================================
fig_convergence, (width_axis, sampling_axis) = plt.subplots(
    1, 2, figsize=(11, 4.5), constrained_layout=True
)
for method in ("DAS", "Conventional NSI", "Angular NSI"):
    color, line_style = {
        "DAS": ("tab:purple", "-"),
        "Conventional NSI": ("tab:red", "--"),
        "Angular NSI": ("tab:blue", "-."),
    }[method]
    method_rows = [
        row for row in lateral_convergence_rows if row["method"] == method
    ]
    spacings_um = np.asarray(
        [row["grid_spacing_x_mm"] * 1e3 for row in method_rows]
    )
    widths_um = np.asarray(
        [row["lateral_fwhm_mm"] * 1e3 for row in method_rows]
    )
    samples = np.asarray(
        [row["lateral_samples_per_fwhm"] for row in method_rows]
    )
    width_axis.plot(
        spacings_um,
        widths_um,
        color=color,
        linestyle=line_style,
        marker="o",
        label=method,
    )
    sampling_axis.plot(
        spacings_um,
        samples,
        color=color,
        linestyle=line_style,
        marker="o",
        label=method,
    )

for axis in (width_axis, sampling_axis):
    axis.set_xscale("log", base=2)
    axis.invert_xaxis()
    axis.set_xlabel("Lateral grid spacing [micrometres]")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False, fontsize=8)

width_axis.set_ylabel("Lateral -6.02 dB width [micrometres]")
width_axis.set_title("FWHM convergence")
sampling_axis.axhline(
    MIN_RECOMMENDED_SAMPLES_PER_FWHM,
    color="0.35",
    linestyle=":",
    linewidth=1.0,
)
sampling_axis.set_ylabel("Samples per lateral FWHM")
sampling_axis.set_title("Sampling of the measured width")
convergence_figure_path = (
    output_dir / "single_scatterer_lateral_grid_convergence.png"
)
fig_convergence.savefig(
    convergence_figure_path, dpi=300, bbox_inches="tight"
)
verify_saved_file(convergence_figure_path)


# ============================================================================
# 8) Fine-grid profiles with explicit -6 dB crossings
# ============================================================================
fig_profiles, (lateral_axis, axial_axis) = plt.subplots(
    1, 2, figsize=(12, 5), constrained_layout=True
)
styles = {
    "DAS": ("tab:purple", "-"),
    "Conventional NSI": ("tab:red", "--"),
    "Angular NSI": ("tab:blue", "-."),
}

for result in final_results:
    method = result["method"]
    color, line_style = styles[method]
    profiles = profiles_by_method[method]
    lateral_profile = profiles["lateral"]
    axial_profile = profiles["axial"]
    lateral_reference = float(lateral_profile[profiles["peak_x_index"]])
    axial_reference = float(axial_profile[profiles["peak_z_index"]])
    lateral_db = amplitude_to_db(lateral_profile, lateral_reference)
    axial_db = amplitude_to_db(axial_profile, axial_reference)

    lateral_axis.plot(
        lateral_x * 1e3,
        lateral_db,
        color=color,
        linestyle=line_style,
        linewidth=1.6,
        label=(
            f"{method}"
            #f"{method} : {result['lateral_fwhm_mm']:.4f} mm"
            #+ ("" if result["lateral_fwhm_converged"] else " (not converged)")
        ),
    )
    axial_axis.plot(
        fine_z * 1e3,
        axial_db,
        color=color,
        linestyle=line_style,
        linewidth=1.6,
        label= f"{method}"
        #f"{method}: {result['axial_fwhm_mm']:.3f} mm",
    )

    lateral_minus6 = profiles["lateral_measurements"]["minus6_db"]
    axial_minus6 = profiles["axial_measurements"]["minus6_db"]
    lateral_axis.scatter(
        [
            lateral_minus6["left_crossing_m"] * 1e3,
            lateral_minus6["right_crossing_m"] * 1e3,
        ],
        [HALF_AMPLITUDE_DB, HALF_AMPLITUDE_DB],
        s=32,
        marker="o",
        facecolors="white",
        edgecolors=color,
        linewidths=1.2,
        zorder=5,
    )
    axial_axis.scatter(
        [
            axial_minus6["left_crossing_m"] * 1e3,
            axial_minus6["right_crossing_m"] * 1e3,
        ],
        [HALF_AMPLITUDE_DB, HALF_AMPLITUDE_DB],
        s=32,
        marker="o",
        facecolors="white",
        edgecolors=color,
        linewidths=1.2,
        zorder=5,
    )

for axis in (lateral_axis, axial_axis):
    axis.axhline(
        HALF_AMPLITUDE_DB,
        color="0.35",
        linestyle=":",
        linewidth=1.0,
    )
    axis.set_ylim(-60.0, 1.0)
    axis.set_yticks([-60.0, -40.0, -20.0, HALF_AMPLITUDE_DB, 0.0])
    axis.set_yticklabels(["-60", "-40", "-20", "-6", "0"])
    axis.set_ylabel("Normalized envelope [dB]")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)

lateral_axis.set_xlabel("x [mm]")
lateral_axis.set_title("Lateral PSF profile")
lateral_axis.set_xlim(
    target_x_m * 1e3 - lateral_half_width_mm,
    target_x_m * 1e3 + lateral_half_width_mm,
)
axial_axis.set_xlabel("z [mm]")
axial_axis.set_title("Axial PSF profile")
axial_axis.set_xlim(
    target_z_m * 1e3 - axial_plot_half_width_mm,
    target_z_m * 1e3 + axial_plot_half_width_mm,
)

profile_path = output_dir / "single_scatterer_fwhm_profiles.png"
fig_profiles.savefig(profile_path, dpi=300, bbox_inches="tight")
verify_saved_file(profile_path)

plt.show()
