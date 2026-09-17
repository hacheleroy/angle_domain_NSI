"""Shared conventional/adaptive beamforming primitives for the PMB revision.

The functions in this module operate on *already delayed* receive-element
samples.  They deliberately accept either NumPy or CuPy arrays so the exact
algebra can be covered by CPU tests while production reconstructions remain
GPU accelerated.

Implemented comparators
-----------------------

``receive_cf_das``
    Conventional receive-aperture coherence-factor-weighted DAS,
    ``|sum(x)|^2 / (M sum(|x|^2))``.

``capon_minimum_variance``
    Spatially smoothed Capon/MV beamforming with overlapping subarrays,
    optional axial/temporal averaging, and trace-proportional diagonal
    loading.  The defaults reproduce the USTB/Synnevag convention
    ``L=floor(M/2)`` and ``epsilon=trace(R)/(100 L)``.

``signed_sqrt_pair_sum`` and ``fdmas_analytic_image``
    Matrone's signed-square-root DMAS pair sum followed by a linear-phase
    band-pass around ``2*f0`` and analytic-signal formation.  The pair sum is
    evaluated with an exact algebraic identity and is tested against the
    direct double sum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import firwin, kaiserord


USTB_REFERENCE_COMMIT = "a539fbea0e08a6acc754b89b111c1b1a699ee830"


def _array_namespace(*arrays: Any) -> Any:
    """Select CuPy when any argument is a CuPy array, otherwise NumPy."""

    for array in arrays:
        if type(array).__module__.split(".", 1)[0] == "cupy":
            import cupy as cp

            return cp
    return np


def _to_numpy(value: Any) -> np.ndarray:
    """Return a host array without importing CuPy in CPU-only environments."""

    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    return np.asarray(value)


@dataclass(frozen=True)
class MvConfiguration:
    """Fully declared minimum-variance implementation choices."""

    subarray_fraction: float = 0.5
    diagonal_loading_coefficient: float = 0.01
    temporal_half_window_samples: int = 0
    scale_to_das_amplitude: bool = True
    chunk_pixels: int = 32

    def validate(self) -> None:
        if not 0.0 < self.subarray_fraction <= 0.5:
            raise ValueError("MV subarray_fraction must lie in (0, 0.5].")
        if not np.isfinite(self.diagonal_loading_coefficient) or (
            self.diagonal_loading_coefficient <= 0.0
        ):
            raise ValueError("MV diagonal loading must be finite and positive.")
        if self.temporal_half_window_samples < 0:
            raise ValueError("MV temporal half-window cannot be negative.")
        if self.chunk_pixels < 1:
            raise ValueError("MV chunk_pixels must be positive.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value.update(
            {
                "subarray_rule": "L=floor(subarray_fraction*M_active)",
                "spatial_smoothing": "all overlapping contiguous subarrays",
                "diagonal_loading_rule": (
                    "epsilon=(diagonal_loading_coefficient/L)*trace(R)"
                ),
                "steering_vector": "ones after focusing delays",
                "linear_solver": "batched dense solve; no explicit inverse",
                "references": (
                    "Synnevag, Austeng and Holm, IEEE TUFFC 2007; "
                    "Synnevag, Austeng and Holm, IEEE TUFFC 2009"
                ),
                "ustb_reference_commit": USTB_REFERENCE_COMMIT,
            }
        )
        return value


@dataclass(frozen=True)
class FdmasFilterConfiguration:
    """Frequency-normalized F-DMAS filter specification."""

    lower_stop_ratio: float = 1.50
    lower_pass_ratio: float = 1.75
    upper_pass_ratio: float = 2.50
    upper_stop_ratio: float = 2.75
    attenuation_db: float = 60.0

    def validate(self) -> None:
        edges = (
            self.lower_stop_ratio,
            self.lower_pass_ratio,
            self.upper_pass_ratio,
            self.upper_stop_ratio,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in edges):
            raise ValueError("F-DMAS frequency ratios must be finite and positive.")
        if not all(left < right for left, right in zip(edges, edges[1:])):
            raise ValueError("F-DMAS frequency ratios must be strictly increasing.")
        if not np.isfinite(self.attenuation_db) or self.attenuation_db <= 0.0:
            raise ValueError("F-DMAS attenuation must be finite and positive.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value.update(
            {
                "pair_product": "sign(si*sj)*sqrt(abs(si*sj)), i<j",
                "filter_domain": "focused real-RF depth traces",
                "analytic_signal": "FFT Hilbert transform after band-pass",
                "reference": "Matrone et al., IEEE TMI 2015",
                "ustb_reference_commit": USTB_REFERENCE_COMMIT,
            }
        )
        return value


def receive_coherence_factor(
    delayed: Any,
    active_mask: Any | None = None,
    *,
    axis: int = -1,
    xp: Any | None = None,
) -> Any:
    """Return the conventional receive-aperture coherence factor.

    ``delayed`` contains focused receive-element contributions.  The active
    count is evaluated per pixel, which is required when a dynamic aperture is
    used.  Inactive entries may already be zero, but the explicit mask avoids
    counting them in the denominator.
    """

    namespace = xp or _array_namespace(delayed, active_mask)
    values = delayed
    if active_mask is None:
        count = values.shape[axis]
    else:
        values = values * active_mask
        count = namespace.sum(active_mask, axis=axis)
    coherent = namespace.sum(values, axis=axis)
    incoherent = namespace.sum(namespace.abs(values) ** 2, axis=axis)
    real_dtype = getattr(incoherent, "dtype", np.dtype(np.float32))
    try:
        tiny = float(np.finfo(real_dtype).tiny)
    except ValueError:
        tiny = float(np.finfo(np.float32).tiny)
    denominator = count * incoherent
    factor = namespace.abs(coherent) ** 2 / namespace.maximum(denominator, tiny)
    factor = namespace.where(count > 0, factor, 0.0)
    return namespace.clip(factor.real, 0.0, 1.0)


def receive_cf_das(
    delayed: Any,
    active_mask: Any | None = None,
    *,
    axis: int = -1,
    xp: Any | None = None,
    return_factor: bool = False,
) -> Any:
    """Apply conventional receive coherence-factor weighting to DAS."""

    namespace = xp or _array_namespace(delayed, active_mask)
    values = delayed if active_mask is None else delayed * active_mask
    coherent = namespace.sum(values, axis=axis)
    factor = receive_coherence_factor(
        delayed, active_mask, axis=axis, xp=namespace
    )
    output = factor * coherent
    return (output, factor) if return_factor else output


def signed_sqrt_pair_sum(
    delayed_rf: Any,
    active_mask: Any | None = None,
    *,
    axis: int = -1,
    xp: Any | None = None,
) -> Any:
    r"""Evaluate Matrone's signed-square-root DMAS pair sum exactly.

    With ``u_i = sign(s_i)*sqrt(abs(s_i))``, the original double sum obeys

    ``sum_{i<j} u_i*u_j = ((sum_i u_i)^2 - sum_i u_i^2)/2``.

    This removes an unnecessary explicit pair tensor without changing the
    mathematical beamformer.
    """

    namespace = xp or _array_namespace(delayed_rf, active_mask)
    if np.dtype(delayed_rf.dtype).kind == "c":
        raise TypeError("F-DMAS requires real delayed RF samples, not complex IQ.")
    values = delayed_rf
    if active_mask is not None:
        values = values * active_mask
    transformed = namespace.sign(values) * namespace.sqrt(namespace.abs(values))
    coherent = namespace.sum(transformed, axis=axis)
    square_sum = namespace.sum(transformed * transformed, axis=axis)
    return 0.5 * (coherent * coherent - square_sum)


def direct_signed_sqrt_pair_sum(delayed_rf: Any) -> Any:
    """Slow direct reference used only for validation and small tests."""

    namespace = _array_namespace(delayed_rf)
    if np.dtype(delayed_rf.dtype).kind == "c":
        raise TypeError("F-DMAS requires real delayed RF samples, not complex IQ.")
    result = namespace.zeros(delayed_rf.shape[:-1], dtype=delayed_rf.dtype)
    for left in range(delayed_rf.shape[-1] - 1):
        products = delayed_rf[..., left, None] * delayed_rf[..., left + 1 :]
        result += namespace.sum(
            namespace.sign(products) * namespace.sqrt(namespace.abs(products)),
            axis=-1,
        )
    return result


def design_fdmas_fir(
    center_frequency_hz: float,
    depth_sampling_frequency_hz: float,
    configuration: FdmasFilterConfiguration | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Design the fixed linear-phase band-pass used by F-DMAS.

    The normalized edges follow USTB's Matrone implementation.  They are
    determined only by ``f0`` and the depth sampling rate, never by the image
    result.  ``depth_sampling_frequency_hz`` is the sampling frequency of the
    focused depth trace, i.e. ``c/(2*dz)`` for a Cartesian pulse-echo image.
    """

    config = configuration or FdmasFilterConfiguration()
    config.validate()
    if not np.isfinite(center_frequency_hz) or center_frequency_hz <= 0.0:
        raise ValueError("center_frequency_hz must be finite and positive.")
    if (
        not np.isfinite(depth_sampling_frequency_hz)
        or depth_sampling_frequency_hz <= 0.0
    ):
        raise ValueError(
            "depth_sampling_frequency_hz must be finite and positive."
        )
    edges_hz = np.asarray(
        [
            config.lower_stop_ratio,
            config.lower_pass_ratio,
            config.upper_pass_ratio,
            config.upper_stop_ratio,
        ],
        dtype=float,
    ) * center_frequency_hz
    nyquist = 0.5 * depth_sampling_frequency_hz
    if edges_hz[-1] >= nyquist:
        raise ValueError(
            "The axial/depth sampling is too coarse for the F-DMAS 2*f0 "
            f"band: upper stop {edges_hz[-1] / 1e6:.3f} MHz is not below "
            f"Nyquist {nyquist / 1e6:.3f} MHz. Increase the focused depth "
            "sampling frequency (reduce the axial grid spacing)."
        )
    transition_hz = min(edges_hz[1] - edges_hz[0], edges_hz[3] - edges_hz[2])
    normalized_width = transition_hz / nyquist
    taps_estimate, beta = kaiserord(config.attenuation_db, normalized_width)
    num_taps = max(15, int(taps_estimate))
    if num_taps % 2 == 0:
        num_taps += 1
    # firwin defines its cutoff at the half-amplitude point.  USTB's
    # kaiserord/fir1 design supplies that cutoff inside each declared
    # stop-to-pass transition, so use the transition midpoints rather than
    # incorrectly treating the passband edges as firwin cutoffs.
    cutoff_hz = np.asarray(
        [0.5 * (edges_hz[0] + edges_hz[1]), 0.5 * (edges_hz[2] + edges_hz[3])]
    )
    coefficients = firwin(
        num_taps,
        cutoff_hz,
        pass_zero=False,
        window=("kaiser", beta),
        fs=depth_sampling_frequency_hz,
        scale=False,
    ).astype(np.float32)
    metadata = config.to_dict()
    metadata.update(
        {
            "center_frequency_hz": float(center_frequency_hz),
            "depth_sampling_frequency_hz": float(depth_sampling_frequency_hz),
            "edges_hz": edges_hz.tolist(),
            "fir_cutoff_hz": cutoff_hz.tolist(),
            "fir_num_taps": int(num_taps),
            "fir_group_delay_samples": int((num_taps - 1) // 2),
            "kaiser_beta": float(beta),
        }
    )
    return coefficients, metadata


def _move_axis_to_last(values: Any, axis: int, namespace: Any) -> tuple[Any, int]:
    normalized = axis if axis >= 0 else values.ndim + axis
    if normalized < 0 or normalized >= values.ndim:
        raise np.AxisError(axis, values.ndim)
    return namespace.moveaxis(values, normalized, -1), normalized


def fft_fir_same(
    values: Any,
    coefficients: Any,
    *,
    axis: int = -1,
    xp: Any | None = None,
) -> Any:
    """Linear FIR filtering with group-delay-corrected ``same`` output."""

    namespace = xp or _array_namespace(values, coefficients)
    moved, original_axis = _move_axis_to_last(values, axis, namespace)
    taps = namespace.asarray(coefficients, dtype=namespace.float32)
    sample_count = moved.shape[-1]
    convolution_count = sample_count + int(taps.size) - 1
    nfft = 1 << max(1, int(convolution_count - 1)).bit_length()
    spectrum = namespace.fft.rfft(moved, n=nfft, axis=-1)
    response = namespace.fft.rfft(taps, n=nfft)
    filtered = namespace.fft.irfft(
        spectrum * response, n=nfft, axis=-1
    )[..., :convolution_count]
    delay = (int(taps.size) - 1) // 2
    same = filtered[..., delay : delay + sample_count]
    return namespace.moveaxis(same, -1, original_axis)


def analytic_signal(
    values: Any,
    *,
    axis: int = -1,
    xp: Any | None = None,
) -> Any:
    """Return the discrete analytic signal using the standard FFT mask."""

    namespace = xp or _array_namespace(values)
    moved, original_axis = _move_axis_to_last(values, axis, namespace)
    count = moved.shape[-1]
    mask = namespace.zeros(count, dtype=namespace.float32)
    mask[0] = 1.0
    if count % 2 == 0:
        mask[count // 2] = 1.0
        mask[1 : count // 2] = 2.0
    else:
        mask[1 : (count + 1) // 2] = 2.0
    result = namespace.fft.ifft(
        namespace.fft.fft(moved, axis=-1) * mask, axis=-1
    )
    return namespace.moveaxis(result, -1, original_axis)


def fdmas_analytic_image(
    pair_sum_image: Any,
    coefficients: Any,
    *,
    depth_axis: int = -1,
    xp: Any | None = None,
) -> Any:
    """Filter a real DMAS image around ``2*f0`` and form its analytic signal."""

    namespace = xp or _array_namespace(pair_sum_image, coefficients)
    if np.dtype(pair_sum_image.dtype).kind == "c":
        raise TypeError("The unfiltered DMAS image must be real.")
    filtered = fft_fir_same(
        pair_sum_image, coefficients, axis=depth_axis, xp=namespace
    )
    return analytic_signal(filtered, axis=depth_axis, xp=namespace)


def temporal_half_window_from_wavelengths(
    wavelength_count: float,
    axial_spacing_m: float,
    center_frequency_hz: float,
    sound_speed_m_s: float,
) -> int:
    """Convert USTB's physical MV averaging half-width into grid samples."""

    values = (
        wavelength_count,
        axial_spacing_m,
        center_frequency_hz,
        sound_speed_m_s,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("MV wavelength/window inputs must be finite and positive.")
    wavelength_m = sound_speed_m_s / center_frequency_hz
    return max(0, int(round(wavelength_count * wavelength_m / axial_spacing_m)))


def _assert_contiguous_apertures(mask_host: np.ndarray) -> None:
    """Reject masks with holes because MV subarrays assume contiguous elements."""

    flat = np.asarray(mask_host, dtype=bool).reshape(-1, mask_host.shape[-1])
    for row in flat:
        indices = np.flatnonzero(row)
        if indices.size and indices[-1] - indices[0] + 1 != indices.size:
            raise ValueError("MV requires a contiguous active receive aperture.")


def capon_minimum_variance(
    delayed: Any,
    active_mask: Any,
    *,
    grid_shape: tuple[int, int] | None = None,
    configuration: MvConfiguration | None = None,
    xp: Any | None = None,
) -> Any:
    """Beamform delayed complex samples with spatially smoothed Capon MV.

    Parameters
    ----------
    delayed, active_mask
        Arrays with identical shape ``(..., receive_elements)``.  Inactive
        contributions are explicitly zeroed before covariance estimation.
    grid_shape
        Optional ``(nx, nz)`` organization of the flattened pixels.  It is
        required when ``temporal_half_window_samples > 0``; temporal averaging
        then occurs along ``z`` without crossing lateral scan lines.
    configuration
        Frozen MV choices.  ``diagonal_loading_coefficient=0.01`` implements
        USTB's ``regCoef=1/100``, because loading is
        ``(regCoef/L)*trace(R)``.
    """

    config = configuration or MvConfiguration()
    config.validate()
    namespace = xp or _array_namespace(delayed, active_mask)
    if np.dtype(delayed.dtype).kind != "c":
        raise TypeError("MV expects delayed analytic/IQ receive samples.")
    if delayed.shape != active_mask.shape:
        raise ValueError("delayed and active_mask must have identical shapes.")
    if delayed.ndim < 2:
        raise ValueError("MV input must have a receive-element axis.")

    element_count = delayed.shape[-1]
    leading_shape = delayed.shape[:-1]
    pixel_count = int(np.prod(leading_shape, dtype=np.int64))
    if grid_shape is None:
        if config.temporal_half_window_samples:
            raise ValueError("grid_shape is required for MV temporal averaging.")
        grid_shape = (pixel_count, 1)
    if tuple(grid_shape) != tuple(leading_shape):
        if int(np.prod(grid_shape, dtype=np.int64)) != pixel_count:
            raise ValueError("grid_shape does not match the MV pixel count.")

    mask_bool = active_mask.astype(bool, copy=False)
    mask_host = _to_numpy(mask_bool).reshape(pixel_count, element_count)
    _assert_contiguous_apertures(mask_host)
    counts_host = mask_host.sum(axis=1).astype(np.int32)
    starts_host = np.zeros(pixel_count, dtype=np.int32)
    nonempty = counts_host > 0
    starts_host[nonempty] = np.argmax(mask_host[nonempty], axis=1)

    flat_values = (delayed * active_mask).reshape(pixel_count, element_count)
    output = namespace.zeros(pixel_count, dtype=delayed.dtype)
    nx, nz = (int(grid_shape[0]), int(grid_shape[1]))
    offsets_host = np.arange(
        -config.temporal_half_window_samples,
        config.temporal_half_window_samples + 1,
        dtype=np.int64,
    )

    # Empty apertures remain zero; one-element apertures fall back to DAS.
    singleton = np.flatnonzero(counts_host == 1)
    if singleton.size:
        singleton_gpu = namespace.asarray(singleton, dtype=namespace.int64)
        output[singleton_gpu] = namespace.sum(flat_values[singleton_gpu], axis=1)

    for active_count in sorted(set(counts_host.tolist())):
        if active_count < 2:
            continue
        subarray_length = max(
            1, int(np.floor(config.subarray_fraction * active_count))
        )
        subarray_length = min(subarray_length, active_count - 1)
        subarray_count = active_count - subarray_length + 1
        members = np.flatnonzero(counts_host == active_count)
        window_indices = (
            namespace.arange(subarray_count, dtype=namespace.int64)[:, None]
            + namespace.arange(subarray_length, dtype=namespace.int64)[None, :]
        )
        identity = namespace.eye(subarray_length, dtype=delayed.dtype)
        steering = namespace.ones(subarray_length, dtype=delayed.dtype)

        for begin in range(0, members.size, config.chunk_pixels):
            chunk_host = members[begin : begin + config.chunk_pixels]
            chunk = namespace.asarray(chunk_host, dtype=namespace.int64)
            starts = namespace.asarray(starts_host[chunk_host], dtype=namespace.int64)
            active_indices = starts[:, None] + namespace.arange(
                active_count, dtype=namespace.int64
            )[None, :]
            current = flat_values[chunk[:, None], active_indices]

            if config.temporal_half_window_samples:
                x_host = chunk_host // nz
                z_host = chunk_host % nz
                neighbor_z = z_host[:, None] + offsets_host[None, :]
                valid_host = (neighbor_z >= 0) & (neighbor_z < nz)
                neighbor_z = np.clip(neighbor_z, 0, nz - 1)
                neighbor_flat_host = x_host[:, None] * nz + neighbor_z
                neighbor_flat = namespace.asarray(
                    neighbor_flat_host, dtype=namespace.int64
                )
                neighbors = flat_values[
                    neighbor_flat[:, :, None], active_indices[:, None, :]
                ]
                temporal_valid = namespace.asarray(
                    valid_host, dtype=namespace.float32
                )
                neighbors = neighbors * temporal_valid[:, :, None]
                valid_count = namespace.asarray(
                    valid_host.sum(axis=1), dtype=namespace.float32
                )
            else:
                neighbors = current[:, None, :]
                valid_count = namespace.ones(current.shape[0], dtype=namespace.float32)

            samples = neighbors[:, :, window_indices]
            covariance = namespace.einsum(
                "btnl,btnm->blm", samples, namespace.conj(samples), optimize=True
            )
            covariance /= (valid_count * subarray_count)[:, None, None]
            trace = namespace.real(namespace.trace(covariance, axis1=1, axis2=2))
            loading = (
                config.diagonal_loading_coefficient / subarray_length
            ) * trace
            floating_epsilon = float(np.finfo(np.float32).eps)
            loading = namespace.where(
                trace > 0.0,
                namespace.maximum(loading, floating_epsilon * trace),
                floating_epsilon,
            )
            loaded = covariance + loading[:, None, None] * identity[None, :, :]
            rhs = namespace.broadcast_to(
                steering, (current.shape[0], subarray_length)
            )[..., None]
            inverse_times_steering = namespace.linalg.solve(loaded, rhs)[..., 0]
            denominator = namespace.einsum(
                "l,bl->b", namespace.conj(steering), inverse_times_steering
            )
            weights = inverse_times_steering / denominator[:, None]

            current_subarrays = current[:, window_indices]
            subarray_outputs = namespace.einsum(
                "bl,bnl->bn",
                namespace.conj(weights),
                current_subarrays,
                optimize=True,
            )
            result = namespace.mean(subarray_outputs, axis=1)
            if config.scale_to_das_amplitude:
                result = result * active_count
            output[chunk] = result

    return output.reshape(leading_shape)


def method_complexities() -> dict[str, dict[str, str]]:
    """Return manuscript-ready leading-order operation counts per pixel/angle."""

    return {
        "DAS": {"order": "O(M)", "note": "receive-element sum"},
        "CF-DAS": {
            "order": "O(M)",
            "note": "DAS plus coherent and incoherent energy sums",
        },
        "MV": {
            "order": "O((2K+1)(M-L+1)L^2 + L^3)",
            "note": "covariance construction plus dense linear solve",
        },
        "DMAS": {
            "order": "O(M) exact / O(M^2) direct",
            "note": "exact pair-sum identity plus axial FIR/Hilbert filtering",
        },
        "Receive-NSI": {
            "order": "O(M)",
            "note": "simultaneous uniform and receive-null sums",
        },
        "Angle-NSI": {
            "order": "O(M)",
            "note": "DAS plus streaming angular-null accumulation",
        },
    }
