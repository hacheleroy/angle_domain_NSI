"""Small, CPU-only readers for the MATLAB-style PICMUS HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class PicmusDataset:
    data: np.ndarray
    angles_rad: np.ndarray
    initial_time_s: float
    sampling_frequency_hz: float
    sound_speed_m_s: float
    probe_geometry_m: np.ndarray
    modulation_frequency_hz: float | None


@dataclass(frozen=True)
class PicmusPhantom:
    positions_m: np.ndarray
    amplitudes: np.ndarray


def _unwrap(node: h5py.Group | h5py.Dataset, h5: h5py.File) -> Any:
    if isinstance(node, h5py.Dataset):
        if node.dtype.names:
            value = node[()]
            names = node.dtype.names
            if "real" in names and "imag" in names:
                return value["real"] + 1j * value["imag"]
        if node.dtype.kind == "O":
            value = node[()]
            reference = value.flat[0] if isinstance(value, np.ndarray) else value
            if isinstance(reference, h5py.Reference):
                return _unwrap(h5[reference], h5)
        return node[()]
    keys = list(node.keys())
    if "real" in keys and "imag" in keys:
        return _unwrap(node["real"], h5) + 1j * _unwrap(node["imag"], h5)
    if "r" in keys and "i" in keys:
        return _unwrap(node["r"], h5) + 1j * _unwrap(node["i"], h5)
    for candidate in ("data", "value", "val", "raw", "rf", "array"):
        if candidate in node:
            return _unwrap(node[candidate], h5)
    valid = [key for key in keys if not key.startswith("#")]
    if len(valid) == 1:
        return _unwrap(node[valid[0]], h5)
    raise TypeError(f"Cannot unwrap HDF5 group {node.name}; keys={keys}")


def _group(h5: h5py.File, group_path: str) -> h5py.Group:
    if group_path in h5:
        return h5[group_path]
    top = [value for value in h5.values() if isinstance(value, h5py.Group)]
    if len(top) == 1:
        return top[0]
    raise KeyError(f"Dataset group {group_path!r} was not found in {h5.filename}")


def _scalar(group: h5py.Group, h5: h5py.File, name: str) -> float:
    return float(np.asarray(_unwrap(group[name], h5)).squeeze())


def load_picmus_dataset(
    path: str | Path, group_path: str = "/US/US_DATASET0000"
) -> PicmusDataset:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with h5py.File(source, "r") as h5:
        group = _group(h5, group_path)
        angles = np.asarray(_unwrap(group["angles"], h5), dtype=float).reshape(-1)
        raw = np.asarray(_unwrap(group["data"], h5))
        geometry = np.asarray(_unwrap(group["probe_geometry"], h5), dtype=float)
        initial_time = _scalar(group, h5, "initial_time")
        fs = _scalar(group, h5, "sampling_frequency")
        sound_speed = _scalar(group, h5, "sound_speed")
        modulation = (
            _scalar(group, h5, "modulation_frequency")
            if "modulation_frequency" in group
            else None
        )

    if geometry.ndim != 2:
        raise ValueError("PICMUS probe_geometry must be two-dimensional.")
    if geometry.shape[0] in (2, 3):
        geometry = geometry.T
    if geometry.shape[1] == 2:
        geometry = np.column_stack([geometry[:, 0], np.zeros(len(geometry)), geometry[:, 1]])
    if geometry.shape[1] != 3:
        raise ValueError(f"Unexpected probe_geometry shape: {geometry.shape}")

    if raw.ndim != 3:
        raise ValueError(f"PICMUS channel data must be 3-D, got {raw.shape}")
    angle_count = angles.size
    element_count = geometry.shape[0]
    if raw.shape[0] == angle_count and raw.shape[1] == element_count:
        raw = raw.transpose(2, 1, 0)
    elif raw.shape[1] == angle_count and raw.shape[2] == element_count:
        raw = raw.transpose(0, 2, 1)
    elif raw.shape[2] == angle_count and raw.shape[1] == element_count:
        pass
    else:
        raise ValueError(
            "Cannot map PICMUS channel data to (samples, elements, angles): "
            f"data={raw.shape}, elements={element_count}, angles={angle_count}."
        )
    return PicmusDataset(
        data=np.ascontiguousarray(raw.astype(np.complex64, copy=False)),
        angles_rad=angles.astype(np.float64, copy=False),
        initial_time_s=initial_time,
        sampling_frequency_hz=fs,
        sound_speed_m_s=sound_speed,
        probe_geometry_m=geometry.astype(np.float32, copy=False),
        modulation_frequency_hz=modulation,
    )


def load_picmus_phantom(
    path: str | Path, group_path: str = "/US/US_DATASET0000"
) -> PicmusPhantom:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with h5py.File(source, "r") as h5:
        group = _group(h5, group_path)
        if "scatterers_positions" in group:
            positions = np.asarray(
                _unwrap(group["scatterers_positions"], h5), dtype=float
            )
            if positions.shape[0] in (2, 3):
                positions = positions.T
        else:
            x = np.asarray(_unwrap(group["phantom_xPts"], h5), dtype=float).reshape(-1)
            z = np.asarray(_unwrap(group["phantom_zPts"], h5), dtype=float).reshape(-1)
            positions = np.column_stack([x, np.zeros_like(x), z])
        if positions.shape[1] == 2:
            positions = np.column_stack(
                [positions[:, 0], np.zeros(len(positions)), positions[:, 1]]
            )
        amplitudes = np.asarray(
            _unwrap(group["scatterers_amplitude"], h5), dtype=float
        ).reshape(-1)
    if positions.shape != (amplitudes.size, 3):
        raise ValueError(
            f"Phantom positions/amplitudes disagree: {positions.shape}, {amplitudes.shape}"
        )
    return PicmusPhantom(
        positions_m=positions.astype(np.float64, copy=False),
        amplitudes=amplitudes.astype(np.float64, copy=False),
    )


def load_picmus_scan(
    path: str | Path, group_path: str = "/US/US_DATASET0000"
) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with h5py.File(source, "r") as h5:
        group = _group(h5, group_path)
        x_axis = np.asarray(_unwrap(group["x_axis"], h5), dtype=float).reshape(-1)
        z_axis = np.asarray(_unwrap(group["z_axis"], h5), dtype=float).reshape(-1)
    return x_axis, z_axis
