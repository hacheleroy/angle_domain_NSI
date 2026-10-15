# Final paper reproduction

The scientific sources in `paper/` match the journal manuscript finalized on
15 October 2026. `paper/asset_manifest.json` lists the 30 local source files,
25 used assets and frozen numerical inputs. It includes decoded-pixel hashes
so a Matplotlib metadata change can be distinguished from a changed figure.

## Exact publication replay

```bash
python -m pip install -r requirements-cpu.txt
python scripts/reproduce_paper.py
```

The validated renderer uses CPython 3.12, NumPy 2.3.5, SciPy 1.17.0,
Matplotlib 3.10.8, h5py 3.16.0 and Pillow 12.3.0. The command reconstructs
the final asset directory from frozen data and accepted rendered records.
Eight figures are redrawn: main Figures 1, 2, 4 and 5; the angular,
perturbation and spatial robustness figures; and the inclusion ROI figure.
The other seven rendered figures are restored from the published archive.
All 25 output files are checksum-verified against the paper. For pixel-equal
redraws, the original PNG bytes are retained to preserve their original
Matplotlib 3.10.8/3.10.9 metadata. The reproduction report records this step.

The replay does not repeat RF reconstruction. A new RF analysis uses
`python scripts/reproduce_paper.py --from-raw --device 0` and the external
datasets described in `data/README.md`.

## Measurement sources

| Paper item | Frozen source and calculation |
|---|---|
| Figure 1 | `simulation_six_method/*arrays.npz`; peak-depth map rows for DAS, CF-DAS, MV, DMAS and Angle-NSI; saved directly beamformed fine profile for Receive-NSI |
| Receive-NSI simulated width | Separate dedicated convergence record in `point_target/simulation_psf_fwhm_summary.json`; 0.008979 mm, printed as 0.0090 mm |
| Figure 2 | Six envelope maps in `picmus_full_phantom/picmus_full_phantom_cache.npz`; no markers or lower diagnostic panel |
| Figure 3 | Accepted MBTrace rendered artifact; raw generation is in `src/doppler_mbtrace.py`. New runs also save `mbtrace_three_method_arrays.npz` |
| Figure 4 | Both six-method carotid cache files in `conventional_baselines/` |
| Figure 5 | Synchronized end-to-end and post-delay timing CSV/JSON records, including all raw repetitions |
| Table 2 and supplementary table S7 | Seven local target array files and all 42 method/target records in `seven_target_metrics/` |
| Inclusion gCNR and ROI figure | Fixed 2-mm-radius ROIs at (-10.5,27.5) and (10.5,27.5) mm on the shared phantom maps; 100 common histogram bins in log-envelope space, -120 dB numerical floor |

The fine Receive-NSI curve is preserved as an archived direct reconstruction.
Its scalar reported width comes from the separate grid-convergence study.
It is never estimated by interpolating the 0.02-mm overview map.

Experimental local maps cover 1.6 by 1.6 mm at 0.02-mm spacing. IQ profiles
use 0.002-mm lateral and 0.005-mm axial sampling. DMAS uses 0.005-mm lateral
and 0.02-mm axial sampling before profile resampling. All target measurements
use the same beamforming functions through a NumPy/SciPy CPU adapter. A fresh
central-target check agreed with the saved GPU profiles within 0.002% of the
profile peak for every method. The archived comparison is included, and the
CPU raw-data stage repeats this check before processing all seven targets.

Widths are connected -6 dB crossings around a target-proximal peak.
Peak/background is 20 log10(local map peak / median envelope outside a
0.35-mm-radius disk). The main table averages seven target-wise scalar
measurements with sample SD (`ddof=1`); peak/background is averaged in dB.
Unit tests independently remeasure all 42 cases from the saved arrays.

The accepted carotid table's printed values are retained in
`conventional_baselines/published_carotid_metrics.csv`. Replay checks them
against the full-precision GPU record within 0.6 units of the last printed
digit, covering three values close to a rounding boundary. New raw runs
format their newly measured values directly. The original full-precision
records remain available.

## Full raw-data workflow

`scripts/run_revision_gpu.py` orchestrates the physical analyses and timing.
The final-paper entry point additionally includes the robustness suite. The
new `seven-target-metrics` and `phantom-inclusion` stages run before asset
materialization. The seven-target CPU stage defaults to two workers and
stores target-specific validated checkpoints. The existing isolated GPU
tiles for the large phantom/MV stages are retained for WSL stability.

The six display names are DAS, CF-DAS, MV, DMAS, Receive-NSI and Angle-NSI.
CF-DAS uses ordinary receive coherence per transmit angle. MV subarray,
loading and averaging settings and the fixed filtered-DMAS definition are
recorded in each analysis. No baseline parameter is fitted during plotting.

A complete new GPU run is not part of CPU CI. It requires the public RF
datasets, PyMUST, MACH, CuPy and CUDA. New manifests record package versions;
historical versions that were not captured are not invented. New hardware
and numerical libraries can change floating-point results and measured
latencies. Exact published artifacts are supplied by the frozen replay.

## Maintenance and versioning

The final-paper workflow needs a focused maintenance release, not a breaking
v2 rewrite. The existing beamforming algebra and published `v1.1.0` history
are preserved. The final source/asset snapshot, measured inputs and regression
checks now have an explicit authority.

A future reusable-library release could move the remaining top-level GPU
scripts into importable reconstruction functions, formalize a shared backend
interface, centralize acquisition settings and add GPU CI. Those changes
would help broader software reuse; they are not prerequisites for reproducing
this paper. They should be validated against the frozen numerical fixtures
before changing the public API.
