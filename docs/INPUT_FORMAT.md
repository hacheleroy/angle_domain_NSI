# Using your own per-angle images

## Required processing stage

Supply the complex receive-beamformed image B_k for each transmit angle k.
All images must use the same spatial grid, coordinate convention and coherent
phase reference. Preserve their relative phase and amplitude; arbitrary
per-angle phase rotations or independent image normalization alter the result.

Apply Angle-NSI before coherent angular compounding, envelope detection and
log compression. Receive-channel RF access is unnecessary once these complex
images exist. A final compounded IQ image or magnitude/B-mode image does not
contain the angular ensemble. Casting an envelope image to complex dtype
cannot restore the lost phase.

For Doppler, the supplied stack may also include a frame axis. The stack API
processes each frame independently; it does not perform clutter filtering.
The MBTrace publication workflow additionally uses the joint-SVD convention
implemented in `src/doppler_mbtrace.py`. Its filtering settings belong to that
acquisition and are not universal preprocessing defaults.

## Array and angle specification

| Item | Requirement |
|---|---|
| `images` | Finite complex NumPy or CuPy array; normally complex64 or complex128 |
| Spatial dimensions | Any shape; all angles share the same grid |
| `angle_axis` | Integer axis indexing the transmit angles; default -1 |
| Example shape | `(depth, lateral, angle)` or `(frame, depth, lateral, angle)` |
| `angles_deg` | Host sequence with one value per angle, in degrees |
| Angle ordering | Strictly increasing; reorder the image angle axis together with the angle vector |
| Angle set | Symmetric positive/negative pairs, within 1e-6 degree tolerance |
| Angular spacing | May be nonuniform when pairs remain symmetric |
| Broadside | Optional; angles within 1e-6 degree of zero get null weight 0 |
| `c` | Finite positive scalar, relative to raw sign(theta) weights |
| Output | Real nonnegative envelope; angle axis removed, other axes preserved |

If your metadata use radians, convert with `np.rad2deg(angles_rad)`. An
incomplete/asymmetric sequence is rejected. The public interface does not
silently recenter or redesign its weights.

## Exact amplitude convention

For each pixel/frame, the implementation forms:

```text
U = sum_k B_k
Z = sum_k sign(theta_k) B_k
E = (abs(Z + c U) + abs(-Z + c U))/2 - abs(Z)
```

Roundoff below zero is clipped. U and Z are raw sums with the same scale;
there is no division by angle count, null-weight norm or c. In particular,
identical coherent per-angle images B give `E = c*K*abs(B)`, while the matching
DAS envelope is `K*abs(B)`. Preserve this convention when comparing c settings.
The return value is an envelope, not a complex signal suitable for recovering
phase or Doppler velocity by itself.

A smaller c can narrow the apparent lateral response while suppressing
signals. Independent display normalization hides absolute amplitude reduction.
Compare linear outputs, target/background separation and the relevant
application endpoint before choosing a setting. The article's detectability
results and parameter sweeps are retained in the numerical snapshot.

## Python interface

```python
import numpy as np
from angle_nsi import angle_nsi

with np.load("my_images.npz", allow_pickle=False) as data:
    envelope = angle_nsi(data["images"], data["angles_deg"],
                         c=0.05, angle_axis=-1)
```

The NumPy interface needs only NumPy. For optional GPU execution, supply a
CuPy array in a suitable CUDA environment; image data and the result remain
on the GPU. The angle vector stays on the host. Validation checks the full
input for finite samples once, including a synchronization for CuPy inputs.
The stack implementation allocates an intermediate weighted stack; use the
streaming accumulation patterns in the reconstruction scripts for ensembles
that cannot fit in memory.

## NPZ command-line format

Save `images` and `angles_deg` with `np.savez_compressed`. Optional `x_mm`,
`z_mm` and `time_s` arrays are copied to the output. Object/pickled arrays are
not supported. These coordinate arrays are metadata and do not change the
calculation.

```bash
angle-nsi my_images.npz --output envelopes.npz --c 0.05 --angle-axis -1
```

Output keys are `angle_nsi`, `das`, `angles_deg`, `c`, `angle_axis` and
`format_version` (currently 1), plus the supplied coordinate arrays. Use
`--image-key`/`--angle-key` for other input names. The input file is protected
from overwrite; an existing output requires explicit `--overwrite`.
