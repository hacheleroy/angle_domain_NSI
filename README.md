# Angle-domain null subtraction imaging

Code, frozen numerical inputs and final paper sources for Henri Leroy,
"Angle-domain null subtraction imaging from beamformed plane-wave data".
[Preprint: arXiv:2608.18252](https://arxiv.org/abs/2608.18252).

The authoritative paper snapshot is **15 October 2026**. The earlier
`v1.1.0` release is preserved as the 17 September revision snapshot; it does
not contain the final figure/table reorganization.

## Reproduce the final figures and tables on a CPU

Use CPython 3.12 on Linux or WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python scripts/reproduce_paper.py
```

The command writes `results/generated/paper_assets/` and verifies **25 used
assets: five main figures, ten supplementary figures, nine table files and
one numerical-macro file**. It redraws eight figures from the frozen arrays
and measurements and restores seven archived GPU-rendered figures. Every
output is checked against the accepted paper; metadata-only PNG differences
are replaced with the original PNG bytes after a pixel-identical comparison.
No external dataset or GPU is needed for this replay. The report explicitly
lists which figures were redrawn and which were restored.

```bash
python scripts/reproduce_paper.py --check-only
python -m unittest discover -s tests -v
```

Tests also remeasure all 42 method/target PSFs from saved maps and profiles,
verify the seven-target means and sample standard deviations, and run the
complete asset replay. Source and data checksums are recorded in
[`paper/asset_manifest.json`](paper/asset_manifest.json).

The clean manuscript and standalone supplement are in [`paper/`](paper/).
They contain only the local figure/table dependencies used by the final text.
The response letter and highlighted review copy are excluded from this public
scientific-source snapshot.

## Main figures

| Figure | Contents | Generation path |
|---|---|---|
| 1 | Simulated point target, six methods and lateral profiles | `src/simulation_six_method_comparison.py`, `src/paper_replot.py` |
| 2 | Complete experimental PICMUS phantom, six unannotated maps | `src/picmus_full_phantom_comparison.py` |
| 3 | MBTrace power Doppler | `src/doppler_mbtrace.py` |
| 4 | Longitudinal and cross-sectional PICMUS carotid acquisitions | `src/conventional_baseline_comparison.py` |
| 5 | End-to-end and post-delay timing | `src/benchmark_scaling.py`, `src/benchmark_conventional.py`, `scripts/materialize_revision_assets.py` |

Table 2 uses the mean and sample SD of seven local target measurements for
width and peak/background, plus one fixed inclusion/background ROI pair for
gCNR. Supplementary table S7 lists all 42 individual cases. The inclusion
measurement is independently reproduced during CPU replay.

The displayed method names are **DAS, CF-DAS, MV, DMAS, Receive-NSI and
Angle-NSI**. CF-DAS uses the ordinary receive-channel coherence factor per
transmit angle. DMAS uses delayed RF pair products and the fixed band-pass
around twice the carrier frequency. USTB is an implementation reference; all
reconstructions use the Python/NumPy/CuPy pipeline.

## Recompute from public RF data

Full reconstruction requires a CUDA-capable NVIDIA GPU and the additional
simulation/data dependencies in `requirements.txt`. The recorded GPU study
used Python 3.14.4, NumPy 2.2.6, CuPy 14.1.1, Matplotlib 3.10.9 and
MACH-beamform 0.1.3 on an RTX A2000. The CPU rendering environment above is
separate and pinned to the environment used for the final presentation edits.

```bash
python -m pip install -r requirements.txt
python scripts/fetch_datasets.py --dataset all
python scripts/reproduce_paper.py --from-raw --device 0
```

Data locations and checksums are documented in [`data/README.md`](data/README.md).
The Open-NSI Box share can require a manual `MBTrace.mat` download; put the
file at `data/Open-NSI/Basic/data/MBTrace.mat` before fetching/verifying inputs.
On the WSL setup that needs the Windows CUDA library first in the search path:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

The raw-data workflow includes theory, angular/perturbation robustness,
point-target simulation and grid convergence, six-method simulation, MBTrace,
carotid, the full PICMUS phantom, seven-target local measurements, inclusion
gCNR and both timing boundaries. The seven-target stage uses the validated
NumPy/SciPy CPU backend, with two workers by default. It checks the central
target against saved GPU profiles before reconstructing all seven targets.
The final materializer emits the current figure/table set without the removed
point markers, diagnostic panel, duplicate timing figure or old single-target
comparison table.

Large PICMUS IQ/MV stages retain isolated, validated lateral checkpoints for
WSL. Steps resume from completed outputs; `--force` is available through
`scripts/run_revision_gpu.py` for an intentional rerun. New run manifests
record installed package versions. Future MBTrace runs also save display
arrays for CPU figure regeneration.

A fresh raw-data run produces new floating-point reconstructions and timing
samples. Wall-clock times depend on the GPU, driver and software. The frozen
replay reproduces the exact published artifacts; the raw-data path reruns the
acquisition-specific calculations. The full CUDA workflow is not exercised
by CPU CI.

## Layout and provenance

```text
paper/                           Final clean LaTeX, used assets and checksums
results/reported/paper_inputs/   Frozen numerical inputs for the final paper
results/reported/                Earlier release records, retained for provenance
results/generated/               New runs and asset replays (ignored by Git)
src/                             Beamforming, measurement and plotting modules
scripts/                         Data acquisition and workflow entry points
tests/                           CPU algebra, measurement and paper regression tests
docs/                            Reproduction protocol and historical validation notes
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for measurement sources,
figure replay boundaries and the remaining software-maintenance work.
External RF datasets are not redistributed. Code uses the MIT license;
committed result records and generated scientific assets use CC BY 4.0.
The IOP class retains its LaTeX Project Public License notice.
Citation metadata are in `CITATION.cff`; please cite the paper and the relevant
PICMUS, Open-NSI, PyMUST/MUST, MACH and baseline-method references.
