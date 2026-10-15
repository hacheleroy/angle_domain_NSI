"""Public stack-to-envelope interface, with NumPy and optional CuPy support."""

from __future__ import annotations

import operator
from typing import Any, Iterable

import numpy as np

from ._core import _array_namespace, angular_sign_weights, nsi_envelope


def angle_nsi(
    images: Any,
    angles_deg: Iterable[float],
    c: float = 0.05,
    *,
    angle_axis: int = -1,
) -> Any:
    """Return the linear Angle-NSI envelope of a complex per-angle image stack.

    Parameters
    ----------
    images:
        Complex NumPy or CuPy array. Each angle must be receive-beamformed
        onto the same spatial grid with a consistent phase reference. Other
        axes, including a frame axis, are preserved. Real envelope images,
        log-compressed images and already compounded images are unsuitable.
    angles_deg:
        One-dimensional host sequence of steering angles in degrees, strictly
        increasing and symmetric about zero. Nonuniform symmetric pairs are
        allowed. An optional zero-degree transmission receives null weight 0.
    c:
        Finite positive DC offset, relative to raw sign(theta) null weights.
        No separate normalization of the uniform or null sum is applied.
    angle_axis:
        Axis whose length equals the number of angles; default: last axis.

    Returns
    -------
    array
        Nonnegative real envelope on the input backend, with the angle axis
        removed. For identical coherent per-angle images B, the response is
        c*K*abs(B); the matching DAS envelope is K*abs(B). There is no division
        by K or c. This is an envelope operation, not a complex IQ output.

    Notes
    -----
    U=sum_k B_k, Z=sum_k sign(theta_k)*B_k, and
    E=(abs(Z+c*U)+abs(-Z+c*U))/2-abs(Z), clipped at zero for roundoff.
    Validation rejects non-finite samples and checks the full input once.
    """
    try:
        if np.ndim(c) != 0 or np.iscomplexobj(c):
            raise ValueError
        offset = float(c)
    except (TypeError, ValueError) as error:
        raise ValueError("c must be a finite positive scalar.") from error
    if not np.isfinite(offset) or offset <= 0.0:
        raise ValueError("c must be a finite positive scalar.")
    namespace = _array_namespace(images)
    values = namespace.asarray(images)
    if values.ndim < 1 or values.size == 0:
        raise ValueError("images must be a nonempty complex array with an angle axis.")
    if values.dtype.kind != "c":
        raise ValueError("images must contain complex per-angle IQ, before envelope detection.")
    try:
        axis = operator.index(angle_axis)
    except TypeError as error:
        raise ValueError("angle_axis must be an integer.") from error
    if not -values.ndim <= axis < values.ndim:
        raise ValueError(f"angle_axis={axis} is invalid for a {values.ndim}-D array.")
    axis %= values.ndim
    angles = np.asarray(list(angles_deg), dtype=float)
    if angles.ndim != 1:
        raise ValueError("angles_deg must be one-dimensional.")
    if values.shape[axis] != angles.size:
        raise ValueError("The angle-axis length must equal len(angles_deg).")
    weights, _ = angular_sign_weights(angles, dtype=values.real.dtype)
    if not bool(namespace.isfinite(values).all()):
        raise ValueError("images must contain only finite complex samples.")
    shape = [1] * values.ndim
    shape[axis] = angles.size
    weights_on_backend = namespace.asarray(weights).reshape(shape)
    uniform = namespace.sum(values, axis=axis)
    null = namespace.sum(values * weights_on_backend, axis=axis)
    return nsi_envelope(uniform, null, offset, xp=namespace)
