# Publication repository handoff: v1.2.0

The final numerical record is dated 15 October 2026. The maintenance release
adds an installable NumPy/CuPy stack interface, an NPZ CLI, CPU example and
input specification. Manuscript projects remain separate.

## First use

```bash
python -m pip install ".[demo]"
python examples/quickstart.py
```

Use `angle_nsi(images, angles_deg, c=0.05, angle_axis=-1)` for your own complex
per-angle images. See `docs/INPUT_FORMAT.md`.

## Publication replay

Use the locked CPython 3.12 Linux/WSL environment:

```bash
python -m pip install -e . -r requirements-test.txt
python scripts/reproduce_paper.py --check-only
python scripts/reproduce_paper.py
python -m unittest discover -s tests -v
```

Figure references and all output checksums are under `results/reference/`.
Table/macro fragments are generated from frozen CSV/JSON without tracked
`.tex` reference files. Eight figures are
redrawn, seven restored, and tables/macros regenerated; final outputs are
byte-checked. The report distinguishes metadata-only PNG changes from the
manifest's narrowly bounded Figure 4 grayscale-rounding variation. No RF dataset or GPU is
needed for this replay.

## Fresh reconstruction

Install the separate optional environment from `requirements.txt`, follow
`docs/GPU_SETUP.md` and `data/README.md`, and use:

```bash
python scripts/fetch_datasets.py --dataset all
python scripts/reproduce_paper.py --from-raw --device 0
```

New results belong under `results/generated/`; frozen results/reference files
are retained for regression. The raw GPU path requires a separate end-to-end
hardware check. Historical validation notes describe their original runs;
current entry points and boundaries are in `README.md` and
`docs/REPRODUCIBILITY.md`.
