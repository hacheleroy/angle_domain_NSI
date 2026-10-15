"""Inspect the wheel and source archive built by ``python -m build``.

The installable package intentionally contains the reusable CPU method and
CLI. Full RF pipelines, frozen results and reference figures live in the
Git checkout, so the wheel does not redistribute external data or CUDA code.
"""

from pathlib import Path
import tarfile
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("angle_nsi-*.whl"))
    archives = list((root / "dist").glob("angle_nsi-*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise SystemExit("Build exactly one release wheel and source archive in dist/.")
    required = {"angle_nsi/__init__.py", "angle_nsi/_api.py", "angle_nsi/_core.py", "angle_nsi/cli.py", "angle_nsi/__main__.py"}
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        if not required <= names:
            raise SystemExit(f"Wheel is missing: {required - names}")
        if any(name.startswith(("src/", "results/", "paper/")) for name in names):
            raise SystemExit("Wheel unexpectedly contains manuscript/reproduction files.")
    with tarfile.open(archives[0]) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
        if not {"src/" + name for name in required} <= names:
            raise SystemExit("Source archive is missing package code.")
        for name in ("pyproject.toml", "examples/quickstart.py", "docs/INPUT_FORMAT.md"):
            if name not in names:
                raise SystemExit(f"Source archive is missing {name}")
    print("Wheel and source archive contain the public API, CLI and source examples.")


if __name__ == "__main__":
    main()
