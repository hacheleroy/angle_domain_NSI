# Angle-domain null subtraction imaging

Code and frozen numerical inputs for Henri Leroy,
"Angle-domain null subtraction imaging from beamformed plane-wave data".
[Preprint: arXiv:2608.18252](https://arxiv.org/abs/2608.18252).

The final numerical-results snapshot is **15 October 2026**. Manuscript
sources, compiled documents and rendered publication assets are maintained
separately; the `paper/` directory is no longer included in the current checkout.
The earlier `v1.1.0` release is preserved as the 17 September revision snapshot; it does
not contain the final figure/table reorganization.

## Use the frozen numerical inputs on a CPU

Use CPython 3.12 on Linux or WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

The arrays, profiles, numerical summaries and timing records remain in
[`results/reported/paper_inputs/`](results/reported/paper_inputs/).
They support numerical checks and plotting without the manuscript sources.
For example, recompute the inclusion gCNR and its ROI figure from the saved
six-method phantom maps, without external RF data or a GPU:

```bash
python src/picmus_inclusion_metrics.py \
  --cache results/reported/paper_inputs/picmus_full_phantom/picmus_full_phantom_cache.npz \
  --output-dir results/generated/phantom_inclusion
```

Run the NSI algebra and PSF measurement helper tests with:

```bash
python -m unittest discover -s tests -p "test_nsi_identities.py" -v
python -m unittest discover -s tests -p "test_psf_metrics.py" -v
```

Exact replay of all 25 publication assets depended on the removed
`paper/asset_manifest.json` and archived rendered figures. Consequently,
`scripts/reproduce_paper.py` (including `--check-only` and `--from-raw`) and
the materializer's `--snapshot` mode require that archive and cannot complete
from the current checkout alone. The paper-source and complete-replay tests
in `tests/test_final_paper.py`, plus the existing **Paper reproducibility**
GitHub Actions workflow, also retain that dependency and need a separate
maintenance update. Full test discovery therefore does not currently pass.
The original exact-replay setup is retained in the
[15 October checkpoint](https://github.com/hacheleroy/angle_domain_NSI/tree/44f90fc598969bdeaf78f858252477d3219b7cbc).

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
measurement can be reproduced with the CPU command above.

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
python scripts/run_revision_gpu.py --device 0 \
  --steps theory,robustness,point-target,simulation-six-method,mbtrace,carotid,experimental-psf,picmus-full-phantom,conventional-baselines,seven-target-metrics,phantom-inclusion,timing,conventional-timing,manuscript-assets
```

This invokes the analysis runner directly, without the removed paper
manifest. New analysis outputs go to `results/generated/revision/`; the
final materialization stage writes figures and LaTeX table fragments to
`results/generated/revision/manuscript_assets/`. These generated fragments
can be imported into a separately maintained manuscript project.

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
samples. Wall-clock times depend on the GPU, driver and software. This path
reruns the acquisition-specific calculations; it does not verify byte-identical
agreement with the separately maintained publication assets. The full CUDA
workflow is not exercised by CPU CI.

## Layout and provenance

```text
results/reported/paper_inputs/   Frozen numerical inputs for the final paper
results/reported/                Earlier release records, retained for provenance
results/generated/               New analyses and generated assets (ignored by Git)
src/                             Beamforming, measurement and plotting modules
scripts/                         Data acquisition and workflow entry points
tests/                           CPU helpers, measurements and legacy paper checks
docs/                            Reproduction protocol and historical validation notes
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for measurement sources
and analysis definitions. Its exact-replay instructions, along with
`results/README.md`, describe the earlier checkout containing `paper/`;
use this README for the current commands and limitations.
External RF datasets are not redistributed. Code uses the MIT license;
committed result records and generated scientific assets use CC BY 4.0.
Citation metadata are in `CITATION.cff`; please cite the paper and the relevant
PICMUS, Open-NSI, PyMUST/MUST, MACH and baseline-method references.
