# Reproduction and numerical provenance

Release v1.2.0 retains the final 15 October 2026 numerical snapshot. Manuscript
sources and compiled documents are maintained separately. The repository
keeps the accepted figures/tables in `results/reference/` and the frozen
numerical inputs in `results/reported/paper_inputs/`.

## Exact publication replay

Use CPython 3.12 on Linux/WSL and a dedicated environment:

```bash
python -m pip install -e . -r requirements-test.txt
python scripts/reproduce_paper.py --check-only
python scripts/reproduce_paper.py
python -m unittest discover -s tests -v
```

`requirements-cpu.txt` freezes direct and transitive rendering dependencies.
The default output is `results/generated/paper_assets/`. The reference manifest
schema 2 verifies every frozen input, shared robustness record and the 25
accepted output assets without requiring manuscript LaTeX.

Eight figures are redrawn: main Figures 1, 2, 4 and 5; angular, perturbation
and spatial robustness; and the inclusion ROI figure. Seven GPU-rendered
figures are restored from the accepted reference images. All tables/macros
are regenerated from records. The report lists both sets. Exact replay
therefore combines numerical regeneration with explicitly identified archival
restoration; it does not repeat RF reconstruction.

PNG byte differences are accepted only after decoded-pixel hashes and image
dimensions match. The original reference bytes are then restored to retain
original Matplotlib metadata. Every final asset must match its reference
SHA-256. This behavior is reported, not hidden as a new rendering result.

Reference assets were recovered byte-for-byte from commit
`44f90fc598969bdeaf78f858252477d3219b7cbc`. Numerical files were preserved.
The old manuscript-source checks are replaced by reference archive completeness
and integrity checks. Numerical regression coverage remains in place.

## Measurement sources

| Paper item | Frozen source and calculation |
|---|---|
| Figure 1 | `simulation_six_method/*arrays.npz`; peak-depth rows for five methods, direct fine Receive-NSI profile |
| Simulated Receive-NSI width | Dedicated grid-convergence study in `point_target/simulation_psf_fwhm_summary.json` |
| Figure 2 | Six envelope maps in `picmus_full_phantom/picmus_full_phantom_cache.npz` |
| Figure 3 | Accepted MBTrace reference image; fresh generation in `src/doppler_mbtrace.py` |
| Figure 4 | Both six-method carotid cache files in `conventional_baselines/` |
| Figure 5 | End-to-end and post-delay timing CSV/JSON records, including repetitions |
| Table 2 and supplementary S7 | Seven local array files and all 42 records in `seven_target_metrics/` |
| Inclusion gCNR/ROI figure | Shared fixed 2-mm-radius ROIs at (-10.5,27.5) and (10.5,27.5) mm; 100 log-envelope histogram bins, -120 dB floor |

The fine Receive-NSI curve is an archived direct reconstruction. Its reported
width comes from the separate convergence study, without interpolation of the
coarse overview map. Experimental local maps span 1.6 by 1.6 mm with 0.02-mm
spacing. IQ profiles use 0.002-mm lateral and 0.005-mm axial sampling; DMAS
uses 0.005-mm lateral and 0.02-mm axial sampling before profile resampling.

Widths use connected -6 dB crossings around a target-proximal peak.
Peak/background is 20 log10(local peak/median envelope outside a 0.35-mm disk).
Seven-target aggregates use sample SD (`ddof=1`); peak/background is averaged
in dB. Tests independently remeasure all 42 cases. The saved central-target
CPU/GPU comparison agrees within 0.002% of profile peak for every method.

`conventional_baselines/published_carotid_metrics.csv` preserves accepted
printed values. Replay compares them with full-precision GPU records within
0.6 units of the last printed digit, covering three rounding-boundary cases.
Fresh RF runs report their new measured values.

## Full RF reconstruction

Follow [GPU setup](GPU_SETUP.md) and [data placement](../data/README.md), then:

```bash
python scripts/reproduce_paper.py --from-raw --device 0
```

The raw branch invokes `scripts/run_revision_gpu.py` without requiring the
reference archive first. It includes robustness and all default analysis
stages, including the seven-target CPU stage and inclusion measurement before
materialization. Results and run manifests go to `results/generated/revision/`.
Installed package versions are recorded; historical versions absent from old
records are not inferred. Large WSL GPU stages retain isolated validated tiles.

The raw workflow needs external RF data, CUDA, CuPy, MACH and simulation
packages. It is outside CPU CI. New floating-point outputs and timing values
can differ with hardware/software. The maintenance release preserves the
published numerical record and validates CPU behavior, without asserting a
new complete GPU run for the newly frozen optional environment.

## Tests, packaging and release

CI runs the full numerical/replay suite in the locked Linux CPU environment,
executes the demo and NPZ CLI, and builds/inspects a wheel and source archive.
A separate Windows/Linux matrix checks the installed NumPy-only package.
The wheel contains the reusable API/CLI; the Git checkout contains the complete
publication reproduction materials. `CITATION.cff`, package version and the
v1.2.0 tag identify the maintenance release. The v1.1.0 tag retains the earlier
17 September snapshot unchanged.
