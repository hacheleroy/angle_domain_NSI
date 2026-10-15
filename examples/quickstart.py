"""Create a toy complex per-angle stack and compare DAS with Angle-NSI on CPU.

This paired-angle phase model illustrates the operation; it is not a full
ultrasound simulation or a replacement for the paper's RF experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from angle_nsi import angle_nsi


def synthetic_stack() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(-4.0, 4.0, 17)
    x_mm = np.linspace(-0.8, 0.8, 321)
    z_mm = np.linspace(19.4, 20.6, 121)
    envelope = np.exp(-0.5 * ((z_mm[:, None] - 20.0) / 0.13) ** 2)
    envelope = envelope * np.exp(-0.5 * (x_mm[None, :] / 0.20) ** 2)
    k = 2.0 * np.pi * 5.21e6 / 1540.0
    phase = np.exp(1j * k * x_mm[:, None] * 1e-3 * np.sin(np.deg2rad(angles)))
    images = (envelope[..., None] * phase[None, ...]).astype(np.complex64)
    return images, angles, x_mm, z_mm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated/quickstart"))
    parser.add_argument("--c", type=float, default=0.05)
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    images, angles, x_mm, z_mm = synthetic_stack()
    source = output / "per_angle_images.npz"
    np.savez_compressed(source, images=images, angles_deg=angles, x_mm=x_mm, z_mm=z_mm)
    # The saved stack is also the input template for the angle-nsi CLI.
    with np.load(source, allow_pickle=False) as data:
        nsi = angle_nsi(data["images"], data["angles_deg"], c=args.c)
        das = np.abs(data["images"].sum(axis=-1))
    np.savez_compressed(output / "envelopes.npz", das=das, angle_nsi=nsi, c=args.c, x_mm=x_mm, z_mm=z_mm)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for ax, image, title in zip(axes[:2], (das, nsi), ("DAS", f"Angle-NSI, c={args.c:g}")):
        db = 20 * np.log10(np.maximum(image / image.max(), 1e-3))
        ax.imshow(db, extent=(x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]), cmap="gray", vmin=-60, vmax=0, aspect="auto")
        ax.set(xlabel="Lateral position (mm)", ylabel="Depth (mm)", title=title)
    center = len(z_mm) // 2
    for image, label in ((das, "DAS"), (nsi, "Angle-NSI")):
        profile = image[center] / image[center].max()
        axes[2].plot(x_mm, 20 * np.log10(np.maximum(profile, 1e-3)), label=label)
    axes[2].set(xlabel="Lateral position (mm)", ylabel="Normalized amplitude (dB)", ylim=(-60, 1), title="Central profiles")
    axes[2].legend()
    fig.suptitle("Toy phase model; each image/profile normalized independently")
    fig.savefig(output / "quickstart.png", dpi=150)
    plt.close(fig)
    print(f"Saved input stack {images.shape}, linear envelopes and quickstart.png in {output}")
    print("Independent display normalization hides the c-dependent amplitude reduction; inspect envelopes.npz for raw values.")


if __name__ == "__main__":
    main()
