"""
Ultrafast Doppler imaging pipeline using a common CuPy DAS implementation:
DAS versus conventional receive-domain NSI versus angular NSI.

Custom CuPy delay-and-sum. The uniform per-angle image and the receive zero-mean field share exactly the
same delays, interpolation and dynamic aperture.

Status:
  - DAS:              uniform field from the common custom beamformer.
  - Angular NSI:      zero-mean scalar weighting of the per-angle DAS stack.
  - Conventional NSI: fully ported below (Section 2b + 4b). This is a
                      custom delay-and-sum that reproduces das_localNSI.m's
                      per-pixel dynamic receive sub-aperture split (quantized,
                      symmetric, matching ApodGen.m), applied identically
                      across all TX angles then compounded. The two DC fields are derived
                      algebraically from the uniform and zero-mean fields.

"""

import csv
import json
import os
import time
from pathlib import Path
import scipy.io as sio

# Select the GPU before importing CuPy.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
from mach.io.must import linear_probe_positions, scan_grid
from mach import wavefront
from mach._vis import db_zero

import cupy as cp
import matplotlib.pyplot as plt
import imageio.v3 as iio

from trace_width_analysis import (
    analyse_three_profiles,
    format_analysis_report,
    match_to_reference,
    measure_peak_widths,
    save_analysis_outputs,
    save_concordance_outputs,
)
from nsi_core import (
    angular_sign_weights,
    coherence_factor_from_moments,
    nsi_envelope,
)

MM = 1e3  # meters -> mm, for display only

# ---------------------------------------------------------------
# 0. Define Paths & Configurations
# ---------------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
mat_path = Path(os.environ.get(
    "OPEN_NSI_MBTRACE_FILE",
    project_root / "data" / "Open-NSI" / "Basic" / "data" / "MBTrace.mat",
))
output_dir = Path(os.environ.get(
    "NSI_OUTPUT_DIR",
    project_root / "results" / "generated" / "doppler",
))

output_dir.mkdir(parents=True, exist_ok=True)

angle_min = -4.0
angle_max = 4.0

# NSI dc offset 
dc_offset = 0.1
c_values = tuple(sorted({
    dc_offset,
    *(
        float(value)
        for value in os.environ.get(
            "NSI_C_VALUES", "0.02,0.05,0.1,0.2"
        ).split(",")
        if value.strip()
    ),
}))
if any(not np.isfinite(value) or value <= 0.0 for value in c_values):
    raise ValueError("Every NSI c-sensitivity value must be finite and positive.")

# Open-NSI's MATLAB loop zeros components 1--10 and 350--400 inclusive for
# 400 frames.  The equivalent zero-based half-open slices are [:10] and
# [349:], so the high-order block contains 51 components despite the released
# parameter name ``EndCutOff=50``.
svd_low, svd_high = 10, 349

# ---------------------------------------------------------------
# 1. Fast Load Dataset & Parameters
# ---------------------------------------------------------------
t_pipeline_start = time.perf_counter()

npy_path = mat_path.with_suffix(".npy")

if not mat_path.is_file() and not npy_path.is_file():
    raise FileNotFoundError(
        "The MBTrace dataset was not found. Place MBTrace.mat under "
        f"{mat_path.parent} or set OPEN_NSI_MBTRACE_FILE to its location."
    )

if os.path.exists(npy_path):
    print(f"Fast-loading pre-converted dataset from {npy_path}...")
    rf_raw = np.load(npy_path)
else:
    print(f"Loading .mat dataset from {mat_path} (first time setup)...")
    mat_data = sio.loadmat(mat_path)
    if 'RF' not in mat_data:
        valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        raise KeyError(f"'RF' key not found in {mat_path}. Available keys: {valid_keys}")
    rf_raw = mat_data['RF']
    np.save(npy_path, rf_raw)

# Open-NSI Parameters
fc = 15.625e6
fs = 62.5e6             # BeamformPara.SamplingFreq
c = 1481.0              # BeamformPara.SoS
pitch = 0.1e-3          # TransPara.Pitch
Nelements = 128         # Standard L11-5v element count
f_number = 1.0          # BeamformPara.FNum
rx_start_s = 0.0        # BeamformPara.InitDepth / SoS  (= t0 in das_localNSI.m)
wc = 2.0 * np.pi * fc   # carrier angular frequency, matches das_localNSI.m's wc

# Normalize dimensions to 4D: (samples, elements, angles, Nt)
if rf_raw.ndim == 3:
    rf_raw = rf_raw[:, :, :, np.newaxis]

samples, elements, num_angles, Nt = rf_raw.shape
print(f"Parsed dimensions -> Samples: {samples}, Elements: {elements}, Angles: {num_angles}, Time frames (Nt): {Nt}")
nl, nc = samples, elements   # match das_localNSI.m's [nl,nc] = size(IQ)

angles_deg = np.linspace(angle_min, angle_max, num_angles)
angles_rad = np.deg2rad(angles_deg)

# ---------------------------------------------------------------
# 1b. Angular NSI apodization (mirrors apo_ZM / apo_DC1 / apo_DC2)
# ---------------------------------------------------------------
apo_zm, angular_weight_diagnostics = angular_sign_weights(angles_deg)

apo_zm_gpu = cp.asarray(apo_zm, dtype=cp.float32)

# ---------------------------------------------------------------
# 2. Build Imaging Geometry & Wavefronts
# ---------------------------------------------------------------
element_positions = linear_probe_positions(Nelements, pitch)
x = np.linspace(-6e-3, 6e-3, 120)
y = np.array([0.0])
z = np.linspace(1e-3, 18e-3, 170)
grid_points = scan_grid(x, y, z)
num_voxels = len(x) * len(z)

array_width = (Nelements - 1) * pitch
wavefront_arrivals_list = []          # dTX/c per angle, shape (num_voxels,), seconds
for ang_rad in angles_rad:
    direction = np.array([np.sin(ang_rad), 0.0, np.cos(ang_rad)], dtype=np.float32)
    wf = wavefront.plane(
        origin_m=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        points_m=grid_points,
        direction=direction,
    ) / c
    t0_offset = (array_width / (2.0 * c)) * np.sin(np.abs(ang_rad))
    wavefront_arrivals_list.append(cp.asarray(wf + t0_offset, dtype=np.float32))

# Conventional receive-domain NSI parameters
xe = ((np.arange(nc) - (nc - 1) / 2.0) * pitch).astype(np.float32)   
ze = np.zeros(nc, dtype=np.float32)

x_pix = np.repeat(x, len(z)).astype(np.float32)   
z_pix = np.tile(z, len(x)).astype(np.float32)     
 
k0 = np.clip(np.round((x_pix - xe[0]) / pitch), 0, nc - 1).astype(np.float64)
fullWidthEl = z_pix / (f_number * pitch)

aprLimit = 2.0 * np.minimum(k0, (nc - 1) - k0)
aprSize = np.minimum(np.maximum(2.0 * np.round(fullWidthEl / 2.0), 2.0), aprLimit)
half = aprSize / 2.0

leftHalf = k0 < (elements / 2.0)
lo0 = np.where(leftHalf, k0 - half, k0 - half + 1.0)
hi0 = np.where(leftHalf, k0 + half - 1.0, k0 + half)
lo0 = np.clip(lo0, 0, nc - 1)
hi0 = np.clip(hi0, 0, nc - 1)

elIdx = np.arange(nc, dtype=np.float64)[None, :]           # (1,nc)
apertureOK = (elIdx >= lo0[:, None]) & (elIdx <= hi0[:, None])   # (num_voxels,nc)

midIdx0 = lo0 + half
signMat = (2.0 * (elIdx >= midIdx0[:, None]) - 1.0).astype(np.float32)  # (num_voxels,nc)

apertureOK_gpu = cp.asarray(apertureOK)
signMat_gpu = cp.asarray(signMat)

dRX_over_c = (np.hypot(x_pix[:, None] - xe[None, :], z_pix[:, None] - ze[None, :]) / c).astype(np.float32)
dRX_over_c_gpu = cp.asarray(dRX_over_c)

col_idx_gpu = cp.arange(nc, dtype=cp.int32)[None, :]   # (1,nc), broadcasts against (num_voxels,nc)

conv_idxf_list = []   # 0-indexed floor sample index, clipped for safe gather, int32
conv_frac_list = []   # fractional part (float32)
conv_mask_list = []   # bool: aperture AND valid time index
conv_phase_list = []  # exp(1j*wc*tau), complex64

for i in range(num_angles):
    tau_i = wavefront_arrivals_list[i][:, None] + dRX_over_c_gpu           # (num_voxels,nc)
    idxt0_i = (tau_i - rx_start_s) * fs                                    # 0-indexed continuous sample position
    idxf0_i = cp.floor(idxt0_i)
    frac_i = (idxt0_i - idxf0_i).astype(cp.float32)
    timeOK_i = (idxt0_i >= 0) & (idxt0_i <= (nl - 2))
    mask_i = apertureOK_gpu & timeOK_i

    idxf0_i_clipped = cp.clip(idxf0_i, 0, nl - 2).astype(cp.int32)         # safe for gather regardless of mask
    phase_i = cp.exp(1j * wc * tau_i).astype(cp.complex64)

    conv_idxf_list.append(idxf0_i_clipped)
    conv_frac_list.append(frac_i)
    conv_mask_list.append(mask_i.astype(cp.float32))   # float32 mask, multiplies cleanly into complex
    conv_phase_list.append(phase_i)

print("Conventional receive-domain NSI aperture/delay tables precomputed.")

# ---------------------------------------------------------------
# 3. Pre-compute Single-Frame GPU Demodulation Operators
# ---------------------------------------------------------------
h_gpu = cp.zeros((samples, 1, 1), dtype=cp.float32)
if samples % 2 == 0:
    h_gpu[0] = 1.0
    h_gpu[samples // 2] = 1.0
    h_gpu[1:samples // 2] = 2.0
else:
    h_gpu[0] = 1.0
    h_gpu[1:(samples + 1) // 2] = 2.0

t_vec_gpu = (cp.arange(samples, dtype=cp.float32) / fs)[:, None, None]
carrier_gpu = cp.exp(-2j * cp.pi * fc * t_vec_gpu).astype(cp.complex64)

print(f"Beamforming {Nt} temporal frames (frame-by-frame GPU demodulation)...")
t_bf_start = time.perf_counter()

M_das = cp.zeros((num_voxels, Nt), dtype=cp.complex64)
M_angle_power = cp.zeros((num_voxels, Nt), dtype=cp.float32)
M_zm_ang = cp.zeros((num_voxels, Nt), dtype=cp.complex64)
M_u_conv = cp.zeros((num_voxels, Nt), dtype=cp.complex64)
M_zm_conv = cp.zeros((num_voxels, Nt), dtype=cp.complex64)

for t in range(Nt):
    rf_frame_gpu = cp.asarray(rf_raw[:, :, :, t], dtype=cp.float32)

    F_rf = cp.fft.fft(rf_frame_gpu, axis=0)
    analytic_frame = cp.fft.ifft(F_rf * h_gpu, axis=0)
    iq_frame_gpu = (analytic_frame * carrier_gpu).astype(cp.complex64)

    comp_das = cp.zeros(num_voxels, dtype=cp.complex64)
    comp_angle_power = cp.zeros(num_voxels, dtype=cp.float32)
    
    comp_zm_ang = cp.zeros(num_voxels, dtype=cp.complex64)
    
    comp_u_conv = cp.zeros(num_voxels, dtype=cp.complex64)
    comp_zm_conv = cp.zeros(num_voxels, dtype=cp.complex64)

    for i, ang_rad in enumerate(angles_rad):
        iq_slice = iq_frame_gpu[:, :, i]                       # (nl, nc)

        # Common DAS interpolation/aperture for all three methods. This keeps
        # beamformer implementation differences out of the domain comparison.
        idxf0_i = conv_idxf_list[i]
        frac_i = conv_frac_list[i]
        mask_i = conv_mask_list[i]
        phase_i = conv_phase_list[i]
        floor_samp = iq_slice[idxf0_i, col_idx_gpu]      
        ceil_samp = iq_slice[idxf0_i + 1, col_idx_gpu]   
        # Standard linear interpolation between adjacent RF/IQ samples.
        interp = floor_samp * (1.0 - frac_i) + ceil_samp * frac_i
        base = interp * phase_i * mask_i                
        angle_result = base.sum(axis=1)
        contrib_zm = base * signMat_gpu

        comp_das += angle_result
        comp_angle_power += cp.abs(angle_result) ** 2
        comp_zm_ang += angle_result * apo_zm_gpu[i]
        comp_u_conv += angle_result
        comp_zm_conv += contrib_zm.sum(axis=1)

    M_das[:, t] = comp_das
    M_angle_power[:, t] = comp_angle_power
    M_zm_ang[:, t] = comp_zm_ang
    M_u_conv[:, t] = comp_u_conv
    M_zm_conv[:, t] = comp_zm_conv

cp.cuda.Stream.null.synchronize()
print(f"Beamforming complete in {(time.perf_counter() - t_bf_start) * 1000:.2f} ms")

# Reconstruct the dependent DC-offset fields exactly by linearity. This avoids
# treating three conventional receive apodizations as three independent DAS
# passes and provides the fair optimized baseline discussed in the manuscript.
M_dc1_ang = M_zm_ang + dc_offset * M_das
M_dc2_ang = -M_zm_ang + dc_offset * M_das
M_dc1_conv = M_zm_conv + dc_offset * M_u_conv
M_dc2_conv = -M_zm_conv + dc_offset * M_u_conv

# ---------------------------------------------------------------
# 5. Spatiotemporal SVD Filtering
#    Both NSI variants use a JOINT Casorati matrix across their three
#    apodization channels (concatenated along the spatial axis).
# ---------------------------------------------------------------
print("Applying SVD clutter filtering...")


def svd_filter(M, low, high):
    """Casorati-matrix SVD clutter filter. M: (space, time)."""
    U, S, Vt = cp.linalg.svd(M, full_matrices=False)
    S_filtered = cp.copy(S)
    S_filtered[:low] = 0.0
    if high < len(S_filtered):
        S_filtered[high:] = 0.0
    return (U * S_filtered) @ Vt


def joint_filtered_nsi(uniform, null, c_value, low, high):
    """Apply the Open-NSI joint SVD convention and shared NSI functional."""

    plus = null + c_value * uniform
    minus = -null + c_value * uniform
    stacked = cp.concatenate([plus, minus, null], axis=0)
    filtered = svd_filter(stacked, low, high)
    plus_filt, minus_filt, null_filt = cp.split(filtered, 3, axis=0)
    uniform_filt = (plus_filt + minus_filt) / (2.0 * c_value)
    return nsi_envelope(uniform_filt, null_filt, c_value, xp=cp)


M_das_filt = svd_filter(M_das, svd_low, svd_high)

# Angular coherence-factor-weighted DAS comparator.  The ensemble is the same
# set of focused per-angle images used by Angle-NSI, rather than receive-channel
# contributions; this precise definition is recorded with the output.
M_cf = coherence_factor_from_moments(
    M_das, M_angle_power, num_angles, xp=cp
) * M_das
M_cf_filt = svd_filter(M_cf, svd_low, svd_high)

M_angular_nsi = joint_filtered_nsi(
    M_das, M_zm_ang, dc_offset, svd_low, svd_high
)
M_conv_nsi = joint_filtered_nsi(
    M_u_conv, M_zm_conv, dc_offset, svd_low, svd_high
)

# ---------------------------------------------------------------
# 6. Power Doppler: DAS vs Conventional NSI vs Angular NSI
# ---------------------------------------------------------------
pd_das = cp.sqrt(cp.sum(cp.abs(M_das_filt) ** 2, axis=1).reshape(len(x), len(z)))
pd_cf = cp.sqrt(cp.sum(cp.abs(M_cf_filt) ** 2, axis=1).reshape(len(x), len(z)))
pd_conv = cp.sqrt(cp.sum(M_conv_nsi ** 2, axis=1).reshape(len(x), len(z)))
pd_angular = cp.sqrt(cp.sum(M_angular_nsi ** 2, axis=1).reshape(len(x), len(z)))

pd_das_db = db_zero(pd_das).get()
pd_cf_db = db_zero(pd_cf).get()
pd_conv_db = db_zero(pd_conv).get()
pd_angular_db = db_zero(pd_angular).get()

extent = [x.min() * MM, x.max() * MM, z.max() * MM, z.min() * MM]
DR = 35.0

# ---------------------------------------------------------------
# 6b. Matched microbubble trace widths at one cross-section
# ---------------------------------------------------------------
z_cross_requested_mm = float(os.environ.get("NSI_CROSS_SECTION_DEPTH_MM", "11"))
z_mm = z * MM
z_cross_idx = int(np.argmin(np.abs(z_mm - z_cross_requested_mm)))
z_cross_actual_mm = float(z_mm[z_cross_idx])
x_mm = x * MM

profile_das = pd_das_db[:, z_cross_idx]
profile_conv = pd_conv_db[:, z_cross_idx]
profile_angular = pd_angular_db[:, z_cross_idx]

trace_analysis = analyse_three_profiles(
    x_mm,
    profile_das,
    profile_conv,
    profile_angular,
    prominence_db=6.0,
    min_distance_mm=0.2,
    min_height_db=-25.0,
    matching_tolerance_mm=0.15,
)
print(format_analysis_report(trace_analysis, z_cross_actual_mm))
trace_csv, trace_json = save_analysis_outputs(
    trace_analysis,
    output_dir,
    depth_mm=z_cross_actual_mm,
)
print(f"Saved matched trace measurements to {trace_csv}")
print(f"Saved trace-analysis audit metadata to {trace_json}")
concordance_paths = save_concordance_outputs(trace_analysis, output_dir)
for path in concordance_paths.values():
    print(f"Saved peak-concordance output to {path}")

das_peaks = trace_analysis["peaks"]["das"]
comparison_profiles = {
    "DAS": profile_das,
    "Angular CF-DAS": pd_cf_db[:, z_cross_idx],
    "Receive-NSI": profile_conv,
    "Angle-NSI": profile_angular,
}
comparison_rows = []
for method_key, profile in comparison_profiles.items():
    peaks = measure_peak_widths(
        x_mm,
        profile,
        prominence_db=6.0,
        min_distance_mm=0.2,
        min_height_db=-25.0,
    )
    pairs = (
        {index: index for index in range(len(das_peaks))}
        if method_key == "DAS"
        else match_to_reference(das_peaks, peaks, tolerance_mm=0.15)
    )
    matched_widths = [peaks[index].width_mm for index in pairs.values()]
    comparison_rows.append(
        {
            "method": method_key,
            "cross_section_depth_mm": z_cross_actual_mm,
            "detected_peak_count": len(peaks),
            "das_reference_peak_count": len(das_peaks),
            "das_concordant_count_at_0p15_mm": len(pairs),
            "das_concordant_fraction_at_0p15_mm": (
                len(pairs) / len(das_peaks) if das_peaks else None
            ),
            "mean_matched_width_mm": (
                float(np.mean(matched_widths)) if matched_widths else None
            ),
            "sample_sd_matched_width_mm": (
                float(np.std(matched_widths, ddof=1))
                if len(matched_widths) > 1
                else None
            ),
        }
    )
comparison_csv = output_dir / "mbtrace_four_method_comparison.csv"
with comparison_csv.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
    writer.writeheader()
    writer.writerows(comparison_rows)
print(f"Saved four-method trace comparison to {comparison_csv}")

# ---------------------------------------------------------------
# 6c. Sensitivity of positional concordance and widths to c
# ---------------------------------------------------------------
print(f"Evaluating NSI c sensitivity at {c_values}...")
c_rows = []
pd_das_peak = float(cp.max(pd_das).get())
for c_value in c_values:
    for method_key, uniform_field, null_field in (
        ("Receive-NSI", M_u_conv, M_zm_conv),
        ("Angle-NSI", M_das, M_zm_ang),
    ):
        if np.isclose(c_value, dc_offset, rtol=0.0, atol=1e-12):
            pd_method = pd_conv if method_key == "Receive-NSI" else pd_angular
            profile_method = (
                pd_conv_db[:, z_cross_idx]
                if method_key == "Receive-NSI"
                else pd_angular_db[:, z_cross_idx]
            )
        else:
            filtered = joint_filtered_nsi(
                uniform_field, null_field, c_value, svd_low, svd_high
            )
            pd_method = cp.sqrt(
                cp.sum(filtered ** 2, axis=1).reshape(len(x), len(z))
            )
            profile_method = db_zero(pd_method).get()[:, z_cross_idx]

        target_peaks = measure_peak_widths(
            x_mm,
            profile_method,
            prominence_db=6.0,
            min_distance_mm=0.2,
            min_height_db=-25.0,
        )
        strict_pairs = match_to_reference(
            das_peaks, target_peaks, tolerance_mm=0.15
        )
        tracking_pairs = match_to_reference(
            das_peaks, target_peaks, tolerance_mm=0.35
        )
        matched_widths = [
            target_peaks[target_index].width_mm
            for target_index in strict_pairs.values()
        ]
        signed_displacements = [
            target_peaks[target_index].x_mm - das_peaks[reference_index].x_mm
            for reference_index, target_index in tracking_pairs.items()
        ]
        c_rows.append(
            {
                "method": method_key,
                "c": float(c_value),
                "das_reference_peak_count": len(das_peaks),
                "detected_peak_count": len(target_peaks),
                "matched_count_at_0p15_mm": len(strict_pairs),
                "matched_fraction_at_0p15_mm": (
                    len(strict_pairs) / len(das_peaks) if das_peaks else None
                ),
                "matched_count_at_0p35_mm": len(tracking_pairs),
                "matched_fraction_at_0p35_mm": (
                    len(tracking_pairs) / len(das_peaks) if das_peaks else None
                ),
                "mean_strictly_matched_width_mm": (
                    float(np.mean(matched_widths)) if matched_widths else None
                ),
                "mean_signed_displacement_within_0p35_mm": (
                    float(np.mean(signed_displacements))
                    if signed_displacements
                    else None
                ),
                "mean_absolute_displacement_within_0p35_mm": (
                    float(np.mean(np.abs(signed_displacements)))
                    if signed_displacements
                    else None
                ),
                "peak_power_amplitude_ratio_to_c_times_das": (
                    float(cp.max(pd_method).get()) / (c_value * pd_das_peak)
                ),
            }
        )
        if not np.isclose(c_value, dc_offset, rtol=0.0, atol=1e-12):
            del filtered, pd_method

c_csv = output_dir / "mbtrace_c_sensitivity.csv"
with c_csv.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(c_rows[0]))
    writer.writeheader()
    writer.writerows(c_rows)
c_json = output_dir / "mbtrace_c_sensitivity.json"
with c_json.open("w", encoding="utf-8") as stream:
    json.dump(
        {
            "primary_c": dc_offset,
            "c_values": list(c_values),
            "cross_section_depth_mm": z_cross_actual_mm,
            "angular_weight_diagnostics": angular_weight_diagnostics.to_dict(),
            "svd_filter": {
                "frame_count": int(Nt),
                "zeroed_components_matlab_indexing": ["1--10", "350--400"],
                "zeroed_slices_python_indexing": ["[:10]", "[349:]"],
                "source_convention": (
                    "Open-NSI Basic/SVDFilt.m with InitCutOff=10 and "
                    "EndCutOff=50"
                ),
            },
            "matching_interpretation": (
                "DAS is a reconstruction reference, not independent ground "
                "truth; matched fractions quantify positional concordance, "
                "not sensitivity."
            ),
            "thresholding": {
                "prominence_db": 6.0,
                "minimum_peak_distance_mm": 0.2,
                "minimum_height_db": -25.0,
                "profiles_normalized_independently": True,
            },
            "rows": c_rows,
        },
        stream,
        indent=2,
        allow_nan=False,
    )
    stream.write("\n")

c_figure = output_dir / "mbtrace_c_sensitivity.png"
figure_c, axes_c = plt.subplots(1, 2, figsize=(8.2, 3.3), dpi=180)
method_styles = {
    "Receive-NSI": ("tab:red", "s", "--"),
    "Angle-NSI": ("tab:blue", "^", "-."),
}
for method_key, (color, marker, linestyle) in method_styles.items():
    selected = [row for row in c_rows if row["method"] == method_key]
    axes_c[0].plot(
        [row["c"] for row in selected],
        [row["matched_fraction_at_0p15_mm"] for row in selected],
        color=color, marker=marker, linestyle=linestyle, label=method_key,
    )
    axes_c[0].plot(
        [row["c"] for row in selected],
        [row["matched_fraction_at_0p35_mm"] for row in selected],
        color=color, marker=marker, linestyle=":", alpha=0.65,
    )
    axes_c[1].plot(
        [row["c"] for row in selected],
        [row["mean_strictly_matched_width_mm"] for row in selected],
        color=color, marker=marker, linestyle=linestyle, label=method_key,
    )
axes_c[0].set(
    xlabel="NSI offset c",
    ylabel="Fraction of DAS peaks matched",
    ylim=(-0.02, 1.02),
)
axes_c[0].text(
    0.98, 0.05, "solid/dashed: 0.15 mm\ndotted: 0.35 mm",
    transform=axes_c[0].transAxes, ha="right", va="bottom", fontsize=7,
)
axes_c[1].set(xlabel="NSI offset c", ylabel="Mean matched -6 dB width [mm]")
for axis in axes_c:
    axis.grid(True, alpha=0.25)
axes_c[1].legend(frameon=False, fontsize=8)
figure_c.tight_layout()
figure_c.savefig(c_figure, dpi=300, bbox_inches="tight")
plt.close(figure_c)
for path in (c_csv, c_json, c_figure):
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Expected c-sensitivity output was not written: {path}")
    print(f"Saved c-sensitivity output to {path}")

# ---------------------------------------------------------------
# 6d. Power-Doppler images and matched cross-section profile
# ---------------------------------------------------------------
fig = plt.figure(figsize=(22, 5.5), dpi=300)
gs = fig.add_gridspec(
    1, 6, width_ratios=[1, 1, 1, 1, 0.05, 1.15], wspace=0.35
)
ax0 = fig.add_subplot(gs[0, 0])
ax_cf = fig.add_subplot(gs[0, 1], sharey=ax0)
ax1 = fig.add_subplot(gs[0, 2], sharey=ax0)
ax2 = fig.add_subplot(gs[0, 3], sharey=ax0)
cax = fig.add_subplot(gs[0, 4])
ax_profile = fig.add_subplot(gs[0, 5])

im0 = ax0.imshow(
    pd_das_db.T, cmap="gray", vmin=-DR, vmax=0, extent=extent, aspect="equal"
)
ax0.set_xlabel("Lateral [mm]")
ax0.set_ylabel("Depth [mm]")
ax0.set_title("(a)")

ax_cf.imshow(
    pd_cf_db.T, cmap="gray", vmin=-DR, vmax=0, extent=extent, aspect="equal"
)
ax_cf.set_xlabel("Lateral [mm]")
ax_cf.set_title("(b)")
plt.setp(ax_cf.get_yticklabels(), visible=False)

ax1.imshow(
    pd_conv_db.T, cmap="gray", vmin=-DR, vmax=0, extent=extent, aspect="equal"
)
ax1.set_xlabel("Lateral [mm]")
ax1.set_title("(c)")
plt.setp(ax1.get_yticklabels(), visible=False)

ax2.imshow(
    pd_angular_db.T, cmap="gray", vmin=-DR, vmax=0, extent=extent, aspect="equal"
)
ax2.set_xlabel("Lateral [mm]")
ax2.set_title("(d)")
plt.setp(ax2.get_yticklabels(), visible=False)
fig.colorbar(im0, cax=cax, label="Normalized intensity [dB]")

for axis in (ax0, ax_cf, ax1, ax2):
    axis.axhline(z_cross_actual_mm, color="orange", linewidth=1.2)

ax_profile.plot(
    x_mm, profile_das, color="tab:purple", linestyle="-", linewidth=1.4, label="DAS"
)
ax_profile.plot(
    x_mm,
    pd_cf_db[:, z_cross_idx],
    color="tab:green",
    linestyle=":",
    linewidth=1.4,
    label="Angular CF-DAS",
)
ax_profile.plot(
    x_mm,
    profile_conv,
    color="tab:red",
    linestyle="--",
    linewidth=1.4,
    label="Receive-NSI",
)
ax_profile.plot(
    x_mm,
    profile_angular,
    color="tab:blue",
    linestyle="-.",
    linewidth=1.4,
    label="Angle-NSI",
)
for match_id, match in enumerate(trace_analysis["matches"], start=1):
    ax_profile.axvline(
        match.das.x_mm, color="0.55", linestyle=":", linewidth=0.7, alpha=0.7
    )
    ax_profile.text(
        match.das.x_mm,
        -4.8,
        str(match_id),
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.25",
    )
ax_profile.set_xlabel("Lateral [mm]")
ax_profile.set_ylabel("Normalized intensity [dB]")
ax_profile.set_ylim(-DR - 4.0, -4.0)
ax_profile.set_title("(e)")
ax_profile.legend(frameon=False, fontsize=8)
ax_profile.yaxis.set_label_position("right")
ax_profile.yaxis.tick_right()

png_out = os.path.join(output_dir, "power_doppler_four_method_comparison.png")
fig.savefig(png_out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved comparison figure with matched trace identifiers to {png_out}")

# Difference map
# fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=300)
# im_diff = ax2.imshow(cp.abs(pd_conv - pd_angular).get().T, extent=extent, aspect="equal")
# ax2.set_xlabel("Lateral [mm]")
# ax2.set_ylabel("Depth [mm]")
# ax2.set_title("|Conventional NSI - Angular NSI|")
# fig2.colorbar(im_diff, ax=ax2, fraction=0.046, pad=0.04)
# plt.tight_layout()
# diff_png_out = os.path.join(output_dir, "diff_convNSI_vs_angularNSI.png")
# fig2.savefig(diff_png_out, dpi=300, bbox_inches="tight")
# plt.close(fig2)

# ---------------------------------------------------------------
# 7. Fast 3-Panel Side-by-Side B-Mode Animation Export
# ---------------------------------------------------------------
print("Exporting 3-panel side-by-side B-mode MP4...")
t_anim_start = time.perf_counter()

# Compute B-mode envelope stacks across all 3 methods
bmode_das_stack = cp.abs(M_das).reshape(len(x), len(z), Nt)
bmode_conv_stack = (0.5 * (cp.abs(M_dc1_conv) + cp.abs(M_dc2_conv)) - cp.abs(M_zm_conv)).reshape(len(x), len(z), Nt)
bmode_ang_stack = (0.5 * (cp.abs(M_dc1_ang) + cp.abs(M_dc2_ang)) - cp.abs(M_zm_ang)).reshape(len(x), len(z), Nt)

bmode_das_movie = db_zero(bmode_das_stack).get()
bmode_conv_movie = db_zero(bmode_conv_stack).get()
bmode_ang_movie = db_zero(bmode_ang_stack).get()

# Dimensions tuned to 1536x640 (exact multiples of 16 for FFmpeg)
fig_anim, axes_anim = plt.subplots(1, 3, figsize=(15.36, 6.4), dpi=100)

im_das = axes_anim[0].imshow(
    bmode_das_movie[:, :, 0].T,
    cmap="gray",
    vmin=-40,
    vmax=0,
    extent=extent,
    aspect="equal"
)
axes_anim[0].set_xlabel("Lateral [mm]", fontsize=11)
axes_anim[0].set_ylabel("Depth [mm]", fontsize=11)
axes_anim[0].set_title("(a)", fontsize=12, fontweight="bold")
fig_anim.colorbar(im_das, ax=axes_anim[0], label="Intensity [dB]", fraction=0.046, pad=0.04)

im_conv = axes_anim[1].imshow(
    bmode_conv_movie[:, :, 0].T,
    cmap="gray",
    vmin=-40,
    vmax=0,
    extent=extent,
    aspect="equal"
)
axes_anim[1].set_xlabel("Lateral [mm]", fontsize=11)
axes_anim[1].set_title("(b)", fontsize=12, fontweight="bold")
fig_anim.colorbar(im_conv, ax=axes_anim[1], label="Intensity [dB]", fraction=0.046, pad=0.04)

im_ang = axes_anim[2].imshow(
    bmode_ang_movie[:, :, 0].T,
    cmap="gray",
    vmin=-40,
    vmax=0,
    extent=extent,
    aspect="equal"
)
axes_anim[2].set_xlabel("Lateral [mm]", fontsize=11)
axes_anim[2].set_title("(c)", fontsize=12, fontweight="bold")
fig_anim.colorbar(im_ang, ax=axes_anim[2], label="Intensity [dB]", fraction=0.046, pad=0.04)

title_anim = fig_anim.suptitle(f"B-Mode (Frame 1/{Nt})", fontsize=13, fontweight="bold")
plt.tight_layout()

# Reserve layout space and setup blitting canvas
title_anim.set_visible(False)
fig_anim.canvas.draw()
bg_anim = fig_anim.canvas.copy_from_bbox(fig_anim.bbox)
title_anim.set_visible(True)

frames_bmode = []
for t in range(Nt):
    fig_anim.canvas.restore_region(bg_anim)

    im_das.set_data(bmode_das_movie[:, :, t].T)
    im_conv.set_data(bmode_conv_movie[:, :, t].T)
    im_ang.set_data(bmode_ang_movie[:, :, t].T)
    title_anim.set_text(f"B-Mode (Frame {t + 1}/{Nt})")

    axes_anim[0].draw_artist(im_das)
    axes_anim[1].draw_artist(im_conv)
    axes_anim[2].draw_artist(im_ang)
    fig_anim.draw_artist(title_anim)

    rgba = np.asarray(fig_anim.canvas.buffer_rgba())
    frames_bmode.append(rgba[:, :, :3].copy())

plt.close(fig_anim)

mp4_bmode_out = os.path.join(output_dir, "bmode_animation_3way.mp4")
iio.imwrite(mp4_bmode_out, np.stack(frames_bmode), fps=20)
print(f"B-mode video export completed in {(time.perf_counter() - t_anim_start):.2f} seconds")

# ---------------------------------------------------------------
# 8. Fast 3-Panel Side-by-Side SVD Blood Signal Animation Export
# ---------------------------------------------------------------
print("Exporting 3-panel side-by-side SVD blood MP4...")
t_blood_start = time.perf_counter()

blood_das_stack = cp.abs(M_das_filt).reshape(len(x), len(z), Nt)
blood_conv_stack = cp.abs(M_conv_nsi).reshape(len(x), len(z), Nt)
blood_ang_stack = cp.abs(M_angular_nsi).reshape(len(x), len(z), Nt)

blood_das_movie = db_zero(blood_das_stack).get()
blood_conv_movie = db_zero(blood_conv_stack).get()
blood_ang_movie = db_zero(blood_ang_stack).get()

das_max = np.max(blood_das_movie)
conv_max = np.max(blood_conv_movie)
ang_max = np.max(blood_ang_movie)

fig_blood, axes_blood = plt.subplots(1, 3, figsize=(15.36, 6.4), dpi=100)

im_blood_das = axes_blood[0].imshow(
    blood_das_movie[:, :, 0].T,
    cmap="gray",
    vmin=das_max - 30.0,
    vmax=das_max,
    extent=extent,
    aspect="equal"
)
axes_blood[0].set_xlabel("Lateral [mm]", fontsize=11)
axes_blood[0].set_ylabel("Depth [mm]", fontsize=11)
axes_blood[0].set_title("(a)", fontsize=12, fontweight="bold")
fig_blood.colorbar(im_blood_das, ax=axes_blood[0], label="Magnitude [dB]", fraction=0.046, pad=0.04)

im_blood_conv = axes_blood[1].imshow(
    blood_conv_movie[:, :, 0].T,
    cmap="gray",
    vmin=conv_max - 30.0,
    vmax=conv_max,
    extent=extent,
    aspect="equal"
)
axes_blood[1].set_xlabel("Lateral [mm]", fontsize=11)
axes_blood[1].set_title("(b)", fontsize=12, fontweight="bold")
fig_blood.colorbar(im_blood_conv, ax=axes_blood[1], label="Magnitude [dB]", fraction=0.046, pad=0.04)

im_blood_ang = axes_blood[2].imshow(
    blood_ang_movie[:, :, 0].T,
    cmap="gray",
    vmin=ang_max - 30.0,
    vmax=ang_max,
    extent=extent,
    aspect="equal"
)
axes_blood[2].set_xlabel("Lateral [mm]", fontsize=11)
axes_blood[2].set_title("(c)", fontsize=12, fontweight="bold")
fig_blood.colorbar(im_blood_ang, ax=axes_blood[2], label="Magnitude [dB]", fraction=0.046, pad=0.04)

title_blood = fig_blood.suptitle(f"Filtered signal (Frame 1/{Nt})", fontsize=13, fontweight="bold")
plt.tight_layout()

# Reserve layout space and setup blitting canvas
title_blood.set_visible(False)
fig_blood.canvas.draw()
bg_blood = fig_blood.canvas.copy_from_bbox(fig_blood.bbox)
title_blood.set_visible(True)

frames_blood = []
for t in range(Nt):
    fig_blood.canvas.restore_region(bg_blood)

    im_blood_das.set_data(blood_das_movie[:, :, t].T)
    im_blood_conv.set_data(blood_conv_movie[:, :, t].T)
    im_blood_ang.set_data(blood_ang_movie[:, :, t].T)
    title_blood.set_text(f"Filtered signal (Frame {t + 1}/{Nt})")

    axes_blood[0].draw_artist(im_blood_das)
    axes_blood[1].draw_artist(im_blood_conv)
    axes_blood[2].draw_artist(im_blood_ang)
    fig_blood.draw_artist(title_blood)

    rgba = np.asarray(fig_blood.canvas.buffer_rgba())
    frames_blood.append(rgba[:, :, :3].copy())

plt.close(fig_blood)

mp4_blood_out = os.path.join(output_dir, "svd_filtered_animation_3way.mp4")
iio.imwrite(mp4_blood_out, np.stack(frames_blood), fps=20)
print(f"SVD blood video export completed in {(time.perf_counter() - t_blood_start):.2f} seconds")

# ---------------------------------------------------------------
# Pipeline Execution Summary
# ---------------------------------------------------------------
t_pipeline_total = time.perf_counter() - t_pipeline_start
print("\n" + "=" * 55)
print(f"TOTAL PIPELINE EXECUTION TIME: {t_pipeline_total:.2f} seconds")
print("=" * 55)
