# Try Angle-NSI on a small image stack

After `python -m pip install ".[demo]"` from the repository root:

```bash
python examples/quickstart.py
angle-nsi results/generated/quickstart/per_angle_images.npz --output results/generated/quickstart/cli_envelopes.npz
```

Open `results/generated/quickstart/quickstart.png`. The model uses a Gaussian
image envelope with the lateral steering-angle phase term. It is deterministic,
small enough for CPU use, and illustrates the angular null; it is not the
PyMUST/RF experiment used for the publication's resolution measurements.

The generated `per_angle_images.npz` provides an input-format template. Its
`images` array has shape `(121, 321, 17)` and complex64 dtype. Outputs include
raw linear envelopes; displayed panels are normalized independently. Vary c
with `python examples/quickstart.py --c 0.1 --output-dir results/generated/demo_c01`.

To create an input file from your own beamformer:

```python
import numpy as np

# per_angle_iq is complex and has shape (depth, lateral, angle).
np.savez_compressed("my_images.npz", images=per_angle_iq,
                    angles_deg=angles_deg, x_mm=x_mm, z_mm=z_mm)
```

Then run `angle-nsi my_images.npz --output my_envelopes.npz`. No receive-channel
RF data or CUDA is needed for this operation. Read the
[input specification](../docs/INPUT_FORMAT.md) before using envelope-detected,
filtered or differently normalized data.

The installable wheel contains the API and CLI. Examples are supplied in the
Git checkout and source archive; publication replay requires the Git checkout
with its numerical and reference files.
