#!/usr/bin/env python3
"""Rebuild and verify the final paper assets, or rerun the raw-data pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/reference"
INPUTS = ROOT / "results/reported/paper_inputs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest() -> dict:
    return json.loads((REFERENCE / "asset_manifest.json").read_text())


def verify_sources(manifest: dict) -> None:
    for name, expected in manifest["numerical_inputs"].items():
        if sha256(INPUTS / name) != expected:
            raise ValueError(f"Frozen numerical input changed: {name}")
    for name, expected in manifest.get("shared_numerical_inputs", {}).items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Shared frozen numerical input changed: {name}")


def verify_reference_assets(manifest: dict) -> None:
    """Check archived PNGs; table references are hashes of regenerated outputs."""
    expected = {name: row for name, row in manifest["assets"].items() if name.endswith(".png")}
    observed = {p.relative_to(REFERENCE).as_posix() for p in REFERENCE.rglob("*.png")}
    if observed != set(expected):
        raise ValueError("Reference figure set differs from the manifest.")
    for name, row in expected.items():
        if sha256(REFERENCE / name) != row["sha256"]:
            raise ValueError(f"Frozen reference figure changed: {name}")


def verify_assets(output: Path, manifest: dict, *, restore_reference_bytes: bool) -> dict:
    """Check outputs, reporting metadata and explicitly bounded raster differences."""
    from PIL import Image

    metadata_only = []
    rendering_differences = []
    expected_names = set(manifest["assets"])
    observed_names = {
        str(p.relative_to(output))
        for p in output.rglob("*")
        if p.is_file() and (p.suffix == ".png" or p.name.startswith("table_") or p.name == "revision_results.tex")
    }
    if observed_names != expected_names:
        raise ValueError(f"Asset set differs: missing={sorted(expected_names - observed_names)}, extra={sorted(observed_names - expected_names)}")
    for name, expected in manifest["assets"].items():
        target = output / name
        if sha256(target) == expected["sha256"]:
            continue
        if target.suffix != ".png":
            raise ValueError(f"Generated table or macro differs from the paper: {name}")
        with Image.open(target) as image:
            rgba = image.convert("RGBA")
            digest = hashlib.sha256(rgba.tobytes()).hexdigest()
            dimensions = list(rgba.size)
        if digest != expected["rgba_sha256"] or dimensions != expected["dimensions"]:
            tolerance = expected.get("rendering_tolerance")
            if not tolerance or dimensions != expected["dimensions"]:
                raise ValueError(f"Rendered figure differs from the paper: {name}")
            import numpy as np

            with Image.open(REFERENCE / name) as reference:
                reference_rgba = np.asarray(reference.convert("RGBA"), dtype=np.int16)
            delta = np.asarray(rgba, dtype=np.int16) - reference_rgba
            changed_pixels = int(np.count_nonzero(np.any(delta, axis=-1)))
            max_difference = int(np.max(np.abs(delta)))
            # Only equal RGB shifts are permitted: colors and opacity stay exact.
            grayscale_only = (not np.any(delta[..., 3])
                              and np.array_equal(delta[..., 0], delta[..., 1])
                              and np.array_equal(delta[..., 0], delta[..., 2]))
            if (not grayscale_only
                    or changed_pixels > tolerance["max_changed_pixels"]
                    or max_difference > tolerance["max_channel_difference"]):
                raise ValueError(f"Rendered figure exceeds its raster tolerance: {name}")
            rendering_differences.append({
                "asset": name, "changed_pixels": changed_pixels,
                "max_channel_difference": max_difference,
            })
        else:
            metadata_only.append(name)
        if restore_reference_bytes:
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes((REFERENCE / name).read_bytes())
            temporary.replace(target)
    return {"metadata_only_png_differences": metadata_only,
            "rendering_tolerance_differences": rendering_differences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generated/paper_assets", help="Output directory for frozen CPU replay.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Verify frozen numerical inputs and reference assets without rebuilding.")
    mode.add_argument("--from-raw", action="store_true", help="Run all public-data analyses, including new seven-target and inclusion stages; requires CUDA.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.check_only:
        manifest = read_manifest()
        verify_sources(manifest)
        verify_reference_assets(manifest)
        print("Frozen numerical inputs and 15 reference figures verified; table/macro hashes retained for replay.")
        return
    if args.from_raw:
        import run_revision_gpu

        steps = list(run_revision_gpu.DEFAULT_STEPS)
        steps.insert(steps.index("manuscript-assets"), "robustness")
        subprocess.run([
            sys.executable, str(ROOT / "scripts/run_revision_gpu.py"),
            "--device", args.device, "--data-root", str(args.data_root),
            "--steps", ",".join(steps),
        ], cwd=ROOT, check=True)
        return

    manifest = read_manifest()
    verify_sources(manifest)
    verify_reference_assets(manifest)
    output = args.output_dir.expanduser().resolve()
    if any(output == frozen or frozen in output.parents or output in frozen.parents
           for frozen in (REFERENCE, ROOT / "results/reported")):
        parser.error("Select a generated output directory, not a frozen source directory.")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/materialize_revision_assets.py"),
        "--snapshot", "--results-root", str(INPUTS), "--output-dir", str(output),
    ], cwd=ROOT, check=True)

    # Independently reproduce the inclusion measurement and its ROI figure.
    with tempfile.TemporaryDirectory(prefix="nsi-paper-inclusion-") as temp:
        subprocess.run([
            sys.executable, str(ROOT / "src/picmus_inclusion_metrics.py"),
            "--cache", str(INPUTS / "picmus_full_phantom/picmus_full_phantom_cache.npz"),
            "--output-dir", temp,
        ], cwd=ROOT, check=True)
        observed = json.loads((Path(temp) / "inclusion_gcnr_summary.json").read_text())
        expected = json.loads((INPUTS / "phantom_inclusion/inclusion_gcnr_summary.json").read_text())
        reference = {r["method"]: r["gcnr"] for r in expected["primary"]}
        for row in observed["primary"]:
            if abs(row["gcnr"] - reference[row["method"]]) > 1e-12:
                raise ValueError("Inclusion gCNR did not reproduce the frozen measurement.")
        target = output / "supplementary/figureS13_picmus_inclusion_rois.png"
        target.write_bytes((Path(temp) / "figureS13_picmus_inclusion_rois.png").read_bytes())

    differences = verify_assets(output, manifest, restore_reference_bytes=True)
    for name, expected in manifest["assets"].items():
        if sha256(output / name) != expected["sha256"]:
            raise ValueError(f"Final artifact checksum mismatch: {name}")
    report = {
        "paper_snapshot": manifest["snapshot_date"],
        "all_25_assets_byte_identical": True,
        "redrawn_figures": manifest["cpu_redrawn_figures"],
        "archived_rendered_figures": manifest["archived_rendered_figures"],
        **differences,
        "note": "Archived PNG bytes are restored after verified redraws. Metadata-only differences and the explicitly bounded Figure 4 float32 grayscale variation are reported separately. Numerical-input and table/macro hashes remain exact. GPU-rendered panels without saved display arrays are copied from the published record; --from-raw regenerates them.",
    }
    (output / "paper_reproduction_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reproduced all 25 paper assets exactly in {output}")


if __name__ == "__main__":
    main()
