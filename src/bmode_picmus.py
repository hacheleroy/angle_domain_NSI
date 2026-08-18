"""
bmode_picmus.py
-------------
GPU-accelerated B-mode imaging on PICMUS in vivo carotid datasets using:
1. Standard Delay-and-Sum (DAS)
2. Conventional receive-domain Null Subtraction Imaging (NSI)
3. Angular Null Subtraction Imaging (NSI)

Version 4 saves all CNR, CR and gCNR measurements, ROI definitions,
descriptive ROI statistics, acquisition metadata and timings to CSV and JSON.
All nested NumPy values are converted recursively before JSON serialization.
"""

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Select the GPU before importing CuPy.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import h5py
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

# Geometry and display helpers from mach; beamforming uses the common custom
# CuPy implementation below so all reconstructions share identical delays.
from mach.io.must import scan_grid
from mach import wavefront
from mach._vis import db_zero

# =============================================================================
# 1. ROBUST PICMUS HDF5 LOADER
# =============================================================================
def load_picmus_hdf5(file_path, dataset_group="/US/US_DATASET0000"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Specified PICMUS HDF5 file does not exist: {file_path}")

    with h5py.File(file_path, "r") as h5:
        if dataset_group in h5:
            grp = h5[dataset_group]
        else:
            available_keys = list(h5.keys())
            if len(available_keys) == 1 and isinstance(h5[available_keys[0]], h5py.Group):
                grp = h5[available_keys[0]]
            else:
                grp = h5

        def _unwrap_node(node):
            if isinstance(node, h5py.Dataset):
                if node.dtype.names is not None:
                    names = node.dtype.names
                    arr = node[()]
                    if "real" in names and "imag" in names:
                        return arr["real"] + 1j * arr["imag"]
                    elif "r" in names and "i" in names:
                        return arr["r"] + 1j * arr["i"]
                    elif "val" in names:
                        return arr["val"]

                if node.dtype.kind == "O":
                    ref = node[()]
                    if isinstance(ref, np.ndarray) and ref.size > 0:
                        ref = ref.flat[0]
                    if isinstance(ref, h5py.Reference):
                        return _unwrap_node(h5[ref])

                data = node[()]
                if isinstance(data, np.ndarray) and data.dtype.kind == "O" and data.size > 0:
                    if isinstance(data.flat[0], h5py.Reference):
                        return _unwrap_node(h5[data.flat[0]])
                return data

            elif isinstance(node, h5py.Group):
                keys = list(node.keys())
                if "real" in keys and "imag" in keys:
                    return _unwrap_node(node["real"]) + 1j * _unwrap_node(node["imag"])
                elif "r" in keys and "i" in keys:
                    return _unwrap_node(node["r"]) + 1j * _unwrap_node(node["i"])

                for sub_key in ["data", "value", "val", "raw", "rf", "array"]:
                    if sub_key in keys and sub_key != node.name.split("/")[-1]:
                        return _unwrap_node(node[sub_key])

                valid_keys = [k for k in keys if k not in ["#refs#", "#subsystem#", "_MATLAB_Attribute_"]]
                if len(valid_keys) == 1:
                    return _unwrap_node(node[valid_keys[0]])
                elif len(valid_keys) > 1:
                    for k in valid_keys:
                        if isinstance(node[k], h5py.Dataset):
                            return _unwrap_node(node[k])
                    return _unwrap_node(node[valid_keys[0]])

            raise TypeError(f"Unable to unwrap HDF5 node '{node.name}' (type: {type(node)})")

        # Read core arrays
        rf_raw = _unwrap_node(grp["data"])
        angles_rad = np.array(_unwrap_node(grp["angles"])).flatten()
        initial_time = float(np.squeeze(_unwrap_node(grp["initial_time"])))
        fs = float(np.squeeze(_unwrap_node(grp["sampling_frequency"])))
        c = float(np.squeeze(_unwrap_node(grp["sound_speed"])))
        probe_geom = np.array(_unwrap_node(grp["probe_geometry"]))
        pitch = round(abs(probe_geom[0,0]-probe_geom[0,1]),4)
        
    # Normalize probe_geometry shape to (Nelements, 3)
    if probe_geom.ndim == 2 and probe_geom.shape[0] in [2, 3]:
        probe_geom = probe_geom.T

    # Normalize RF array dimensions to (samples, elements, angles)
    n_angles = len(angles_rad)
    if rf_raw.ndim == 3:
        if rf_raw.shape[0] == n_angles:          # (Nangles, Nelements, Nt)
            rf_raw = np.transpose(rf_raw, (2, 1, 0))
        elif rf_raw.shape[1] == n_angles:        # (Nt, Nangles, Nelements)
            rf_raw = np.transpose(rf_raw, (0, 2, 1))

    return rf_raw, angles_rad, initial_time, fs, c, probe_geom, pitch


def compute_gcnr(region_signal, region_background, num_bins=100):
    """
    Generalized Contrast-to-Noise Ratio (gCNR).
    Rodriguez-Molares et al., "The Generalized Contrast-to-Noise Ratio:
    A Formal Definition for Lesion Detectability," IEEE TUFFC, 2020.
    DOI: 10.1109/TUFFC.2019.2956855

    gCNR = 1 - overlap between the normalized pixel-value histograms of a
    'signal' ROI (e.g. hypoechoic vessel lumen) and a 'background' ROI
    (e.g. surrounding tissue), using a common set of bins spanning the
    pooled value range of both.

    Ranges [0, 1]: 0 = distributions identical (no detectability),
    1 = distributions fully separated (perfect detectability).
    Invariant to monotonic transforms (log compression, gain, etc.), so
    it can be computed equally on linear envelope or dB-compressed data.

    Parameters
    ----------
    region_signal, region_background : array_like
        Pixel values sampled from each ROI (any shape; flattened internally).
    num_bins : int
        Number of histogram bins spanning the pooled value range.
        100 is a common default; results converge for finer binning,
        so this normally isn't a sensitive knob unless regions are tiny.

    Returns
    -------
    gcnr : float
    """
    s = np.asarray(region_signal).ravel()
    b = np.asarray(region_background).ravel()

    lo = min(s.min(), b.min())
    hi = max(s.max(), b.max())
    bin_edges = np.linspace(lo, hi, num_bins + 1)

    h_s, _ = np.histogram(s, bins=bin_edges)
    h_b, _ = np.histogram(b, bins=bin_edges)

    h_s = h_s / h_s.sum()
    h_b = h_b / h_b.sum()

    return 1.0 - np.sum(np.minimum(h_s, h_b))


def circular_roi_mask(x_m, z_m, center_cm, radius_cm):
    """
    Boolean mask, shape (nx, nz), selecting grid points inside a circle
    of `radius_cm` centered at `center_cm = (x0_cm, z0_cm)`.
    x_m, z_m: the same 1D x/z grid arrays (in meters) used to build the
    reconstruction grid (so this lines up exactly with bmode_*.reshape(nx,nz)).
    """
    X, Z = np.meshgrid(x_m * 100.0, z_m * 100.0, indexing="ij")  # -> cm, shape (nx,nz)
    x0, z0 = center_cm
    return (X - x0) ** 2 + (Z - z0) ** 2 <= radius_cm ** 2

def compute_contrast_ratio(region_signal, region_background, input_is_db=False):
    """
    Contrast Ratio (CR), in dB.

    CR = 20*log10(mu_background / mu_signal)   [amplitude/envelope data]
    or, if the ROIs are already in dB:
    CR = mu_background_dB - mu_signal_dB       [logs subtract instead]

    Positive CR = signal region darker than background (e.g. a hypoechoic
    vessel lumen), matching the usual convention in vascular imaging.

    Parameters
    ----------
    region_signal, region_background : array_like
        Pixel values sampled from each ROI (any shape; flattened internally).
    input_is_db : bool
        Set True if region_signal/region_background were sampled from a
        log-compressed (dB) image rather than linear amplitude/envelope data.
    """
    s = np.asarray(region_signal).ravel()
    b = np.asarray(region_background).ravel()
    mu_s, mu_b = s.mean(), b.mean()

    if input_is_db:
        return mu_b - mu_s
    return 20.0 * np.log10(mu_b / mu_s)


def compute_cnr(region_signal, region_background):
    """
    Contrast-to-Noise Ratio (CNR), dimensionless.

    CNR = |mu_background - mu_signal| / sqrt(sigma_background^2 + sigma_signal^2)

    Classically computed on linear (pre-log) amplitude/envelope data --
    unlike gCNR, CR and CNR are NOT invariant to log compression or gain,
    so which domain you compute them in changes the number and should be
    reported alongside the value.
    """
    s = np.asarray(region_signal).ravel()
    b = np.asarray(region_background).ravel()
    mu_s, mu_b = s.mean(), b.mean()
    sigma_s, sigma_b = s.std(), b.std()
    return np.abs(mu_b - mu_s) / np.sqrt(sigma_b ** 2 + sigma_s ** 2)


def summarize_region(values):
    """Return reproducible descriptive statistics for one image ROI."""

    array = np.asarray(values, dtype=float).ravel()
    return {
        "pixel_count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def finite_float_or_none(value):
    """Convert NumPy scalars to JSON-safe floats, preserving failed metrics."""

    value = float(value)
    return value if np.isfinite(value) else None


def format_optional_metric(value, format_spec):
    """Format a metric for the terminal without failing on undefined values."""

    return "undefined" if value is None else format(value, format_spec)


def to_json_safe(value):
    """Recursively convert NumPy/Path objects to strict JSON-compatible types."""

    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, complex):
        return {
            "real": to_json_safe(value.real),
            "imaginary": to_json_safe(value.imag),
        }
    return value


def save_metric_results(output_dir, csv_rows, json_summary):
    """Save the cumulative B-mode metrics as a flat CSV and structured JSON."""

    if not csv_rows:
        raise ValueError("No B-mode metric rows are available to save.")

    csv_path = output_dir / "bmode_nsi_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = output_dir / "bmode_nsi_results.json"
    json_ready = to_json_safe(json_summary)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(json_ready, stream, indent=2, allow_nan=False)
        stream.write("\n")

    for path in (csv_path, json_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Expected output was not written correctly: {path}")

    return csv_path.resolve(), json_path.resolve()

# =============================================================================
# 2. MAIN BEAMFORMING EXECUTABLE
# =============================================================================
def main():
    t_script_start = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).isoformat()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(os.environ.get(
        "NSI_OUTPUT_DIR", project_root / "results" / "generated" / "bmode"
    )).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    picmus_root = Path(os.environ.get(
        "PICMUS_DATA_DIR",
        project_root / "data" / "PICMUS" / "in_vivo",
    ))
    cc_file = Path(os.environ.get(
        "PICMUS_CC_FILE",
        picmus_root / "carotid_cross" / "carotid_cross_expe_dataset_rf.hdf5",
    ))
    cl_file = Path(os.environ.get(
        "PICMUS_CL_FILE",
        picmus_root / "carotid_long" / "carotid_long_expe_dataset_rf.hdf5",
    ))

    views = [
        {"name": "CL", "label": "Longitudinal", "file": str(cl_file)},
        {"name": "CC", "label": "Cross-section", "file": str(cc_file)},
    ]

    # Keep the two view definitions independent even when they initially use
    # the same coordinates. This makes later view-specific adjustment explicit
    # and records the exact masks used for every reported result.
    roi_by_view = {
        "CL": {
            "signal_center_cm": (-0.70, 1.54),
            "signal_radius_cm": 0.30,
            "background_center_cm": (0.20, 2.70),
            "background_radius_cm": 0.30,
        },
        "CC": {
            "signal_center_cm": (-0.13, 1.77),
            "signal_radius_cm": 0.30,
            "background_center_cm": (-0.10, 2.70),
            "background_radius_cm": 0.30,
        },
    }

    # Reconstruction & NSI parameters
    f_number = 1.0
    fc = 5.208333e6   # PICMUS L11-4v probe carrier frequency
    dc = 0.05     # NSI DC Offset for angular and conventional NSI (see Agarwal et al.)
    gcnr_num_bins = 100
    display_min_db = -60.0
    display_max_db = 0.0

    # Reconstruction Grid Setup (Carotid FOV)
    nx, nz = 128, 250
    x = np.linspace(-19.2e-3, 19.2e-3, nx, dtype=np.float32)
    y = np.array([0.0])
    z = np.linspace(0e-3, 40e-3, nz, dtype=np.float32)
    grid_points = scan_grid(x, y, z)
    extent_cm = [x.min() * 100.0, x.max() * 100.0, z.max() * 100.0, z.min() * 100.0]
    num_voxels = len(x) * len(z)

    csv_rows = []
    json_summary = {
        "script": Path(__file__).name,
        "schema_version": 1,
        "run_started_utc": run_started_utc,
        "metric_definitions": {
            "CR_dB": "20*log10(mean background / mean signal), linear envelope",
            "CNR": "abs(mean background - mean signal) / sqrt(var background + var signal), linear envelope",
            "gCNR": "1 minus overlap of normalized signal/background histograms, normalized dB image",
            "standard_deviation": "population standard deviation (ddof=0)",
        },
        "reconstruction_parameters": {
            "f_number": f_number,
            "carrier_frequency_hz": fc,
            "nsi_dc_offset": dc,
            "gcnr_histogram_bins": gcnr_num_bins,
            "grid_nx": nx,
            "grid_nz": nz,
            "grid_x_min_m": float(x.min()),
            "grid_x_max_m": float(x.max()),
            "grid_z_min_m": float(z.min()),
            "grid_z_max_m": float(z.max()),
            "display_min_db": display_min_db,
            "display_max_db": display_max_db,
            "angular_zero_mean_weights": "sign(angle), with broadside weight zero",
            "interpolation": "linear between adjacent IQ samples",
        },
        "views": [],
    }

    for v in views:
        t_view_start = time.perf_counter()
        print(f"\n==================================================")
        print(f" Processing View: {v['name']} ({v['label']})")
        print(f" File: {v['file']}")
        print(f"==================================================")

        # 1. Load Data & Acoustic Metadata
        rf_raw, angles_rad, initial_time, fs, c, probe_geom, pitch = load_picmus_hdf5(v["file"])
        samples, elements, num_angles = rf_raw.shape
        wc = 2.0 * np.pi * fc

        print(f"Loaded RF: {samples} samples x {elements} elements x {num_angles} angles")
        print(f"Acoustic Params -> fs: {fs/1e6:.2f} MHz, fc: {fc/1e6:.2f} MHz, c: {c:.1f} m/s")
        print(f"Extracted initial_time: {initial_time * 1e6:.3f} us")

        # 2. Transmit Wavefront Arrivals using PICMUS global initial_time
        wavefront_arrivals_list = []
        for ang in angles_rad:
            direction = np.array([np.sin(ang), 0.0, np.cos(ang)], dtype=np.float32)
            wf = wavefront.plane(
                origin_m=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                points_m=grid_points,
                direction=direction,
            ) / c
            wavefront_arrivals_list.append(cp.asarray(wf + initial_time, dtype=np.float32))

        # 3. Angular NSI Apodization Vectors
        angles_deg = np.rad2deg(angles_rad)
        if not np.allclose(angles_deg, -angles_deg[::-1], atol=1e-6):
            raise ValueError("The steering sequence must be symmetric about broadside.")
        apo_zm = np.sign(angles_deg).astype(np.float32)
        apo_zm[np.isclose(angles_deg, 0.0, atol=1e-6)] = 0.0
        if not np.isclose(apo_zm.sum(), 0.0):
            raise ValueError("The steering angles must be symmetric about broadside.")

        apo_zm_gpu = cp.asarray(apo_zm, dtype=cp.float32)

        # 4. Conventional receive-domain NSI geometry and precomputations
        xe = probe_geom[:, 0].astype(np.float32)
        ze = probe_geom[:, 2].astype(np.float32)
        
        x_pix = np.repeat(x, len(z)).astype(np.float32)
        z_pix = np.tile(z, len(x)).astype(np.float32)
        
        k0 = np.clip(np.round((x_pix - xe[0]) / pitch), 0, elements - 1).astype(np.float64)  
        fullWidthEl = z_pix / (f_number * pitch)
          
        aprLimit = 2.0 * np.minimum(k0, (elements - 1) - k0)
        aprSize = np.minimum(np.maximum(2.0 * np.round(fullWidthEl / 2.0), 2.0), aprLimit)
        half = aprSize / 2.0
        
        leftHalf = k0 < (elements / 2.0)
        lo0 = np.where(leftHalf, k0 - half, k0 - half + 1.0)
        hi0 = np.where(leftHalf, k0 + half - 1.0, k0 + half)
        lo0 = np.clip(lo0, 0, elements - 1)
        hi0 = np.clip(hi0, 0, elements - 1)

        elIdx = np.arange(elements, dtype=np.float64)[None, :]           # (1,elements)
        apertureOK = (elIdx >= lo0[:, None]) & (elIdx <= hi0[:, None])   # (num_voxels,elements)

        midIdx0 = lo0 + half
        signMat = (2.0 * (elIdx >= midIdx0[:, None]) - 1.0).astype(np.float32) 
        
        apertureOK_gpu = cp.asarray(apertureOK)
        signMat_gpu = cp.asarray(signMat)
        
        dRX_over_c = (np.hypot(x_pix[:, None] - xe[None, :], z_pix[:, None] - ze[None, :]) / c).astype(np.float32)
        dRX_over_c_gpu = cp.asarray(dRX_over_c)

        col_idx_gpu = cp.arange(elements, dtype=cp.int32)[None, :]

        conv_idxf_list = []
        conv_frac_list = []
        conv_mask_list = []
        conv_phase_list = []

        for i in range(num_angles):
            tau_i = wavefront_arrivals_list[i][:, None] + dRX_over_c_gpu
            idxt0_i = (tau_i - initial_time) * fs  
            idxf0_i = cp.floor(idxt0_i)
            frac_i = (idxt0_i - idxf0_i).astype(cp.float32)
            timeOK_i = (idxt0_i >= 0) & (idxt0_i <= (samples - 2))
            mask_i = apertureOK_gpu & timeOK_i

            idxf0_i_clipped = cp.clip(idxf0_i, 0, samples - 2).astype(cp.int32)
            phase_i = cp.exp(1j * wc * tau_i).astype(cp.complex64)

            conv_idxf_list.append(idxf0_i_clipped)
            conv_frac_list.append(frac_i)
            conv_mask_list.append(mask_i.astype(cp.float32))
            conv_phase_list.append(phase_i)

        # 5. Demodulate RF -> Baseband IQ on GPU
        rf_gpu = cp.asarray(rf_raw.real, dtype=cp.float32)
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

        F_rf = cp.fft.fft(rf_gpu, axis=0)
        analytic_rf = cp.fft.ifft(F_rf * h_gpu, axis=0)
        iq_gpu = (analytic_rf * carrier_gpu).astype(cp.complex64)

        # 6. Beamforming DAS + Conventional NSI + Angular NSI
        print(f"\nBeamforming DAS & NSI on GPU (f-number = {f_number}, dc_offset = {dc})...")
        t_bf = time.perf_counter()

        comp_das = cp.zeros(num_voxels, dtype=cp.complex64)
        
        comp_zm_ang = cp.zeros(num_voxels, dtype=cp.complex64)
        
        comp_u_conv = cp.zeros(num_voxels, dtype=cp.complex64)
        comp_zm_conv = cp.zeros(num_voxels, dtype=cp.complex64)

        for i in range(num_angles):
            iq_slice = iq_gpu[:, :, i]

            # Common DAS interpolation/aperture for all three methods. Using
            # the same per-angle field here avoids confounding the apodization
            # comparison with two different beamformer implementations.
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
            comp_zm_ang += angle_result * apo_zm_gpu[i]
            comp_u_conv += angle_result
            comp_zm_conv += contrib_zm.sum(axis=1)

        # 7. Form Final images
        das_lin = cp.abs(comp_das.reshape(nx, nz)).get()
        bmode_das = db_zero(cp.abs(comp_das.reshape(nx, nz))).get()

        # Only the uniform and zero-mean fields are independent:
        # I_DC1 = Z + cU and I_DC2 = -Z + cU.
        comp_dc1_ang = comp_zm_ang + dc * comp_das
        comp_dc2_ang = -comp_zm_ang + dc * comp_das
        ang_nsi_comp = 0.5 * (cp.abs(comp_dc1_ang) + cp.abs(comp_dc2_ang)) - cp.abs(comp_zm_ang)
        ang_nsi_lin = cp.maximum(ang_nsi_comp.reshape(nx, nz), 1e-12).get()
        bmode_ang_nsi = db_zero(cp.maximum(ang_nsi_comp.reshape(nx, nz), 1e-12)).get()

        comp_dc1_conv = comp_zm_conv + dc * comp_u_conv
        comp_dc2_conv = -comp_zm_conv + dc * comp_u_conv
        conv_nsi_comp = 0.5 * (cp.abs(comp_dc1_conv) + cp.abs(comp_dc2_conv)) - cp.abs(comp_zm_conv)
        conv_nsi_lin = cp.maximum(conv_nsi_comp.reshape(nx, nz), 1e-12).get()
        bmode_conv_nsi = db_zero(cp.maximum(conv_nsi_comp.reshape(nx, nz), 1e-12)).get()

        cp.cuda.Stream.null.synchronize()
        beamforming_seconds = time.perf_counter() - t_bf
        print(f"Beamforming completed in {beamforming_seconds:.2f} seconds.")

        # --- Image-quality metrics and complete audit record ---
        roi = roi_by_view[v["name"]]
        roi_signal_center_cm = roi["signal_center_cm"]
        roi_signal_radius_cm = roi["signal_radius_cm"]
        roi_bg_center_cm = roi["background_center_cm"]
        roi_bg_radius_cm = roi["background_radius_cm"]

        mask_signal = circular_roi_mask(
            x, z, roi_signal_center_cm, roi_signal_radius_cm
        )
        mask_bg = circular_roi_mask(x, z, roi_bg_center_cm, roi_bg_radius_cm)
        signal_pixel_count = int(np.count_nonzero(mask_signal))
        background_pixel_count = int(np.count_nonzero(mask_bg))
        if signal_pixel_count == 0 or background_pixel_count == 0:
            raise ValueError(
                f"The {v['name']} ROI configuration produced an empty mask."
            )

        output_png = output_dir / f"Fig_BMode_NSI_Comparison_{v['name']}.png"
        method_images = [
            ("DAS", bmode_das, das_lin),
            ("Conventional NSI", bmode_conv_nsi, conv_nsi_lin),
            ("Angular NSI", bmode_ang_nsi, ang_nsi_lin),
        ]
        view_metrics = []
        view_row_start = len(csv_rows)

        for name, img_db, img_lin in method_images:
            signal_linear = img_lin[mask_signal]
            background_linear = img_lin[mask_bg]
            signal_db = img_db[mask_signal]
            background_db = img_db[mask_bg]

            gcnr = finite_float_or_none(compute_gcnr(
                signal_db, background_db, num_bins=gcnr_num_bins
            ))
            cr = finite_float_or_none(compute_contrast_ratio(
                signal_linear, background_linear
            ))
            cnr = finite_float_or_none(compute_cnr(
                signal_linear, background_linear
            ))
            signal_linear_stats = summarize_region(signal_linear)
            background_linear_stats = summarize_region(background_linear)
            signal_db_stats = summarize_region(signal_db)
            background_db_stats = summarize_region(background_db)

            print(
                f"{name:20s}  "
                f"gCNR={format_optional_metric(gcnr, '.3f')}   "
                f"CR={format_optional_metric(cr, '6.2f')} dB   "
                f"CNR={format_optional_metric(cnr, '.3f')}"
            )

            csv_rows.append({
                "view": v["name"],
                "view_label": v["label"],
                "method": name,
                "dataset_file": v["file"],
                "figure_file": str(output_png),
                "gcnr": gcnr,
                "contrast_ratio_db": cr,
                "cnr": cnr,
                "signal_pixel_count": signal_pixel_count,
                "background_pixel_count": background_pixel_count,
                "signal_mean_linear": signal_linear_stats["mean"],
                "signal_sd_linear": signal_linear_stats["standard_deviation"],
                "signal_median_linear": signal_linear_stats["median"],
                "background_mean_linear": background_linear_stats["mean"],
                "background_sd_linear": background_linear_stats["standard_deviation"],
                "background_median_linear": background_linear_stats["median"],
                "signal_mean_db": signal_db_stats["mean"],
                "signal_sd_db": signal_db_stats["standard_deviation"],
                "background_mean_db": background_db_stats["mean"],
                "background_sd_db": background_db_stats["standard_deviation"],
                "signal_center_x_cm": roi_signal_center_cm[0],
                "signal_center_z_cm": roi_signal_center_cm[1],
                "signal_radius_cm": roi_signal_radius_cm,
                "background_center_x_cm": roi_bg_center_cm[0],
                "background_center_z_cm": roi_bg_center_cm[1],
                "background_radius_cm": roi_bg_radius_cm,
                "samples": samples,
                "elements": elements,
                "angles": num_angles,
                "angle_min_deg": float(np.min(angles_deg)),
                "angle_max_deg": float(np.max(angles_deg)),
                "sampling_frequency_hz": fs,
                "carrier_frequency_hz": fc,
                "sound_speed_m_per_s": c,
                "pitch_m": pitch,
                "initial_time_s": initial_time,
                "f_number": f_number,
                "nsi_dc_offset": dc,
                "gcnr_histogram_bins": gcnr_num_bins,
                "grid_nx": nx,
                "grid_nz": nz,
                "beamforming_seconds": beamforming_seconds,
                "view_processing_seconds": None,
            })
            view_metrics.append({
                "method": name,
                "gcnr": gcnr,
                "contrast_ratio_db": cr,
                "cnr": cnr,
                "signal_roi_linear": signal_linear_stats,
                "background_roi_linear": background_linear_stats,
                "signal_roi_db": signal_db_stats,
                "background_roi_db": background_db_stats,
            })

        view_json = {
            "view": v["name"],
            "view_label": v["label"],
            "dataset_file": v["file"],
            "acquisition_metadata": {
                "samples": samples,
                "elements": elements,
                "angles": num_angles,
                "angle_min_deg": float(np.min(angles_deg)),
                "angle_max_deg": float(np.max(angles_deg)),
                "sampling_frequency_hz": fs,
                "carrier_frequency_hz": fc,
                "sound_speed_m_per_s": c,
                "pitch_m": pitch,
                "initial_time_s": initial_time,
            },
            "roi": {
                "signal": {
                    "role": "vessel lumen",
                    "center_cm": list(roi_signal_center_cm),
                    "radius_cm": roi_signal_radius_cm,
                    "pixel_count": signal_pixel_count,
                },
                "background": {
                    "role": "surrounding tissue",
                    "center_cm": list(roi_bg_center_cm),
                    "radius_cm": roi_bg_radius_cm,
                    "pixel_count": background_pixel_count,
                },
            },
            "beamforming_seconds": beamforming_seconds,
            "metrics": view_metrics,
        }

        # 8. Plot & Export 3-Panel Side-by-Side Comparison Figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), dpi=300)

        im0 = axes[0].imshow(bmode_das.T, cmap="gray", vmin=display_min_db, vmax=display_max_db, extent=extent_cm, aspect="equal")
        axes[0].set_title("(a) DAS", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("x [cm]")
        axes[0].set_ylabel("z [cm]")
        fig.colorbar(im0, ax=axes[0], label="Intensity [dB]", fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(bmode_conv_nsi.T, cmap="gray", vmin=display_min_db, vmax=display_max_db, extent=extent_cm, aspect="equal")
        axes[1].set_title("(b) Conventional NSI", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("x [cm]")
        fig.colorbar(im1, ax=axes[1], label="Intensity [dB]", fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(bmode_ang_nsi.T, cmap="gray", vmin=display_min_db, vmax=display_max_db, extent=extent_cm, aspect="equal")
        axes[2].set_title("(c) Angular NSI", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("x [cm]")
        fig.colorbar(im2, ax=axes[2], label="Intensity [dB]", fraction=0.046, pad=0.04)

        #fig.suptitle(f"PICMUS {v['label']} View - Beamforming Comparison", fontsize=13, fontweight="bold")
        
        from matplotlib.patches import Circle
        for ax in axes:
            ax.add_patch(Circle(roi_signal_center_cm, roi_signal_radius_cm,
                                    edgecolor="lime", facecolor="none", linewidth=1.2))
            ax.add_patch(Circle(roi_bg_center_cm, roi_bg_radius_cm,
                                    edgecolor="cyan", facecolor="none", linewidth=1.2))
        
        plt.tight_layout()

        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 3-panel comparison figure: {output_png}")

        view_processing_seconds = time.perf_counter() - t_view_start
        for row in csv_rows[view_row_start:]:
            row["view_processing_seconds"] = view_processing_seconds
        view_json["view_processing_seconds"] = view_processing_seconds
        view_json["figure_file"] = str(output_png)
        json_summary["views"].append(view_json)

        # Save cumulatively after every view so results from a completed view
        # survive if a later dataset fails.
        json_summary["run_completed_utc"] = datetime.now(timezone.utc).isoformat()
        json_summary["total_execution_seconds"] = time.perf_counter() - t_script_start
        csv_path, json_path = save_metric_results(
            output_dir, csv_rows, json_summary
        )
        print(f"Saved cumulative metrics CSV: {csv_path}")
        print(f"Saved cumulative results JSON: {json_path}")

    total_execution_seconds = time.perf_counter() - t_script_start
    json_summary["run_completed_utc"] = datetime.now(timezone.utc).isoformat()
    json_summary["total_execution_seconds"] = total_execution_seconds
    csv_path, json_path = save_metric_results(output_dir, csv_rows, json_summary)

    print(f"\nTotal execution time: {total_execution_seconds:.2f} seconds.")
    print(f"Final metrics CSV: {csv_path}")
    print(f"Final structured results JSON: {json_path}")


if __name__ == "__main__":
    main()
