"""Backend-agnostic primitives shared by the NSI reconstruction scripts.

The functions in this module deliberately accept either NumPy or CuPy arrays.
Keeping the algebra here makes the reported reconstructions use the same
definitions and permits CPU-only unit tests of the cancellation identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class AngularWeightDiagnostics:
    """Audit information for an angular null-weight vector."""

    angle_count: int
    negative_count: int
    broadside_count: int
    positive_count: int
    weight_sum: float
    weight_l1_norm: float
    symmetric_angle_pairs: bool
    antisymmetric_weights: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _array_namespace(*arrays: Any) -> Any:
    """Select CuPy when any input is a CuPy array, otherwise NumPy."""

    for array in arrays:
        module = type(array).__module__.split(".", 1)[0]
        if module == "cupy":
            import cupy as cp  # Imported only in GPU environments.

            return cp
    return np


def angular_sign_weights(
    angles_deg: Iterable[float],
    *,
    require_symmetric: bool = True,
    atol_deg: float = 1e-6,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, AngularWeightDiagnostics]:
    """Return ``sign(theta)`` weights and verify the zero-sum assumptions.

    A broadside acquisition receives weight zero.  Even and odd angle counts
    are both valid provided the nonzero steering angles occur in symmetric
    pairs.  Uniform angular spacing is *not* required.  Incomplete or otherwise
    asymmetric angle sets are rejected because ``sign(theta)`` is then not a
    justified angular null without an explicitly redesigned weight vector.
    """

    angles = np.asarray(list(angles_deg), dtype=float).reshape(-1)
    if angles.size < 2 or not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg must contain at least two finite angles.")
    if np.any(np.diff(angles) <= 0.0):
        raise ValueError("angles_deg must be strictly increasing.")

    symmetric = bool(
        np.allclose(angles, -angles[::-1], rtol=0.0, atol=atol_deg)
    )
    if require_symmetric and not symmetric:
        raise ValueError(
            "Angle-NSI with sign(theta) requires symmetric positive/negative "
            "angle pairs; use a redesigned zero-sum vector for incomplete "
            "or asymmetric acquisitions."
        )

    weights = np.sign(angles)
    broadside = np.isclose(angles, 0.0, rtol=0.0, atol=atol_deg)
    weights[broadside] = 0.0
    antisymmetric = bool(
        np.allclose(weights, -weights[::-1], rtol=0.0, atol=1e-7)
    )
    if require_symmetric and not np.isclose(weights.sum(), 0.0, atol=1e-7):
        raise RuntimeError("The angular-null weights do not sum to zero.")

    diagnostics = AngularWeightDiagnostics(
        angle_count=int(angles.size),
        negative_count=int(np.count_nonzero(angles < -atol_deg)),
        broadside_count=int(np.count_nonzero(broadside)),
        positive_count=int(np.count_nonzero(angles > atol_deg)),
        weight_sum=float(weights.sum()),
        weight_l1_norm=float(np.abs(weights).sum()),
        symmetric_angle_pairs=symmetric,
        antisymmetric_weights=antisymmetric,
    )
    return weights.astype(dtype, copy=False), diagnostics


def nsi_envelope(
    uniform: Any,
    null: Any,
    dc_offset: float,
    *,
    xp: Any | None = None,
    clip_nonnegative: bool = True,
) -> Any:
    r"""Evaluate the two-field NSI envelope.

    .. math::

       E = \tfrac12\left(|Z+cU|+|-Z+cU|\right)-|Z|.

    ``U`` is the uniformly weighted reference *sum* and ``Z`` is the relevant
    receive- or angle-domain null sum.  The implementation convention uses raw
    :math:`\pm1` sign weights, so ``c`` is defined relative to that convention.
    """

    if not np.isfinite(dc_offset) or dc_offset <= 0.0:
        raise ValueError("dc_offset must be finite and positive.")
    namespace = xp or _array_namespace(uniform, null)
    plus = null + dc_offset * uniform
    minus = -null + dc_offset * uniform
    result = 0.5 * (namespace.abs(plus) + namespace.abs(minus)) - namespace.abs(
        null
    )
    return namespace.maximum(result, 0.0) if clip_nonnegative else result


def coherence_factor_from_moments(
    coherent_sum: Any,
    incoherent_power_sum: Any,
    contribution_count: int,
    *,
    xp: Any | None = None,
    epsilon: float | None = None,
) -> Any:
    r"""Compute the conventional coherence factor from sufficient moments.

    The contribution ensemble can be receive channels or transmit angles.  In
    this project it is explicitly the per-angle focused-image ensemble
    :math:`B_k`, giving

    .. math:: CF_\theta = |\sum_k B_k|^2/(K\sum_k |B_k|^2).
    """

    if contribution_count < 1:
        raise ValueError("contribution_count must be positive.")
    namespace = xp or _array_namespace(coherent_sum, incoherent_power_sum)
    if epsilon is None:
        # This scalar works for float32 and float64 and only guards an all-zero
        # denominator; it has no practical effect at measured signal levels.
        epsilon = float(np.finfo(np.float32).tiny)
    numerator = namespace.abs(coherent_sum) ** 2
    denominator = contribution_count * incoherent_power_sum
    factor = numerator / namespace.maximum(denominator, epsilon)
    return namespace.clip(factor.real, 0.0, 1.0)


def coherence_factor(
    contributions: Any,
    *,
    axis: int = -1,
    xp: Any | None = None,
    epsilon: float | None = None,
) -> Any:
    """Compute coherence factor along an explicitly selected ensemble axis."""

    namespace = xp or _array_namespace(contributions)
    count = int(contributions.shape[axis])
    coherent = namespace.sum(contributions, axis=axis)
    power = namespace.sum(namespace.abs(contributions) ** 2, axis=axis)
    return coherence_factor_from_moments(
        coherent,
        power,
        count,
        xp=namespace,
        epsilon=epsilon,
    )


def coherence_weighted_das(
    contributions: Any,
    *,
    axis: int = -1,
    xp: Any | None = None,
    epsilon: float | None = None,
) -> Any:
    """Return the linear envelope of angular-CF-weighted coherent DAS."""

    namespace = xp or _array_namespace(contributions)
    coherent = namespace.sum(contributions, axis=axis)
    factor = coherence_factor(
        contributions, axis=axis, xp=namespace, epsilon=epsilon
    )
    return factor * namespace.abs(coherent)


def nsi_c_sweep(
    uniform: Any,
    null: Any,
    c_values: Iterable[float],
    *,
    xp: Any | None = None,
) -> dict[float, Any]:
    """Evaluate identical fields over a declared set of positive ``c`` values."""

    namespace = xp or _array_namespace(uniform, null)
    output: dict[float, Any] = {}
    for value in c_values:
        c_value = float(value)
        if c_value in output:
            raise ValueError(f"Duplicate c value: {c_value}")
        output[c_value] = nsi_envelope(
            uniform, null, c_value, xp=namespace
        )
    if not output:
        raise ValueError("At least one c value is required.")
    return output
