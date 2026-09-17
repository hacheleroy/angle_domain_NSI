# Angle-domain null subtraction imaging

Reproducibility code and reported numerical outputs for the manuscript
"Angle-domain null subtraction imaging from beamformed plane-wave data" by
Henri Leroy.

**Preprint:** [arXiv:2608.18252](https://arxiv.org/abs/2608.18252)

Release `v1.1.0` is the reproducibility snapshot for the PMB major
resubmission. It includes the reviewer-requested CF-DAS, MV and DMAS
implementations, the six-method publication workflow, corrected GPU records,
and line-by-line DMAS cache validation.

The revised publication workflow compares six reconstructions with the same
delays, apertures and sampling:

1. delay-and-sum (DAS);
2. coherence-factor-weighted DAS (CF-DAS);
3. minimum variance (MV);
4. delay-multiply-and-sum (DMAS);
5. Receive-NSI; and
6. Angle-NSI, in which zero-sum weights are applied across the per-angle
   complex image stack.

All six are implemented in Python/CuPy from the published formulations. The
corresponding USTB implementations at a pinned commit are recorded as an
independent reference. CF-DAS uses the ordinary coherence factor, not a
nonzero generalized-coherence-factor spectral band. DMAS includes the fixed
filtering around `2*f0` specified for the standard ultrasound implementation;
the publication label remains DMAS.

Angle-NSI is a post-receive-beamforming construction. It is not algebraically
identical to Receive-NSI and is not presented as a performance-equivalent
replacement. Its intended use is an environment that exposes phase-preserving
complex images for every transmit angle but does not expose receive-channel
data. It cannot be applied to a final coherently compounded, envelope-detected,
or log-compressed image.

## Repository layout

```text
src/                 Analysis and reconstruction scripts
tests/               CPU-only unit tests for algebra and measurement helpers
data/                Placement instructions for external data (not included)
results/reported/    CSV/JSON outputs used in the manuscript
results/exploratory/ Additional analyses not used for manuscript claims
docs/                Validation protocol and implementation notes
```

The stable script names replace the version suffixes used during manuscript
development. No beamforming or measurement formula was changed during this
packaging step.

## Software environment

The synchronized benchmark reported in the manuscript used Python 3.14.4,
NumPy 2.2.6, CuPy 14.1.1, Matplotlib 3.10.9 and MACH-beamform 0.1.3 on an
NVIDIA RTX A2000. Install the dependencies in a fresh environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` selects the CUDA 12 CuPy wheel used for the reported
benchmark. On another CUDA major version, install the corresponding CuPy
package instead. The CPU-only tests require NumPy, SciPy, Matplotlib, h5py and
pytest and can instead be installed with `requirements-test.txt`.

## Verify the installation

From the repository root:

```bash
python -m unittest discover -s tests -v
python src/benchmark_nsi.py --help
python src/benchmark_scaling.py --help
python src/picmus_experimental_psf.py --help
python src/picmus_full_phantom_comparison.py --help
python src/simulation_six_method_comparison.py --help
python src/conventional_baseline_comparison.py --help
python src/benchmark_conventional.py --help
python src/simulation_robustness.py --help
```

The full GPU scripts require a CUDA-capable NVIDIA GPU. Quick GPU checks are:

```bash
python src/benchmark_nsi.py --quick
python src/simulation_robustness.py --quick
```

Quick-mode values are engineering checks and must not be used in publications.

```bash
python src/benchmark_conventional.py --quick
```

## Reproduce the reported analyses

### Six-method point-target simulation

```bash
python src/simulation_six_method_comparison.py \
  --output-dir results/generated/simulation_six_method
```

This is the source of revised manuscript figure 1. It reconstructs DAS,
CF-DAS, MV, DMAS, Receive-NSI and Angle-NSI. Quantitative lateral profiles are
beamformed directly on the fine grid. The DMAS axial grid is fine enough to
keep its fixed `2*f0` band below Nyquist.

The separate NSI offset-sensitivity analysis remains available with
`python src/simulation_point_target.py`; it evaluates `c = 0.02, 0.05, 0.1,
0.2` and writes to the directory selected by `NSI_OUTPUT_DIR`.

### Angular-null small-angle model

```bash
python src/angular_null_theory.py
```

This CPU script checks the paired-angle first-order expansion and plots the
normalized null slope versus angle count and total angular span.

### Robustness sweeps

```bash
python src/simulation_robustness.py \
  --output-dir results/generated/robustness
```

The angle-count, angular-span, missing-angle, noise, phase-jitter and spatial
PSF outputs are reported in the manuscript or supplement. The two-target
analysis is retained as exploratory output and is not used for a manuscript
claim.

### Synchronized timing benchmark

```bash
python src/benchmark_nsi.py \
  --warmups 10 \
  --repetitions 50 \
  --scope reconstruction \
  --transfers both \
  --receive-input-mode device-weighted \
  --angle-storage-mode both \
  --output-dir results/generated/timing
```

The benchmark validates streaming and stored-stack angular NSI using
scale-aware numerical tolerances because complex64 summation order can cause
small round-off differences. See `docs/VALIDATION_PROTOCOL.md` for the timing
scope and optional commands.

The complete scaling and transfer-control suite is resumable:

```bash
python src/benchmark_scaling.py \
  --warmups 10 \
  --repetitions 50 \
  --output-dir results/generated/timing_scaling
```

It varies image-grid size, receive-element count, and transmit-angle count. It
also records a preloaded-kernel control and the older host-stacked two-field
Receive-NSI path. Exact logical H2D/D2H payload bytes and bootstrap intervals
for median latency are written to JSON and CSV.

### PICMUS carotid B-mode data

Place the two PICMUS in-vivo datasets as described in `data/README.md`, then:

```bash
PICMUS_DATA_DIR=/path/to/PICMUS/in_vivo \
NSI_OUTPUT_DIR=results/generated/bmode \
python src/bmode_picmus.py
```

This script provides the NSI `c`-sensitivity table for CNR, CR and gCNR. The
six-method main-figure carotid comparison is produced by
`conventional_baseline_comparison.py`.

### PICMUS experimental point-target PSFs

After running `python scripts/fetch_datasets.py --dataset picmus-resolution`:

```bash
python src/picmus_experimental_psf.py \
  --output-dir results/generated/picmus_experimental_psf
```

The five near-axis targets quantify depth dependence; the two off-axis targets
together with the central target near 37.5 mm quantify lateral variability.
Use `--metadata-only` to validate the HDF5 dataset, phantom, and scan files on a
CPU-only machine.

The full-phantom six-method reconstruction used for revised manuscript figure
2 is:

```bash
python src/picmus_full_phantom_comparison.py \
  --local-profile-summary results/generated/picmus_experimental_psf/picmus_experimental_psf_summary.json \
  --local-profiles results/generated/picmus_experimental_psf/picmus_experimental_psf_profiles.npz \
  --output-dir results/generated/picmus_full_phantom
```

It displays all seven nominal targets and combines the maps with a target-wise
central-response diagnostic derived from directly beamformed fine profiles.
DAS, CF-DAS, Receive-NSI and Angle-NSI use a 0.05 mm native lateral display
grid; MV and DMAS use 0.15 and 0.10 mm native lateral grids, respectively, and
are resampled only for the common overview. Quantitative profiles are measured
directly and are not extracted from the resampled images.
The full-phantom cache is staged, so a completed non-MV or MV reconstruction is
retained if the later DMAS stage is interrupted.

### CF-DAS, MV, and DMAS comparison

The dedicated comparison preserves the established four-method studies and
uses the same receive delays, linear interpolation, dynamic aperture, angle
set, and experimental inputs for all methods:

```bash
python src/conventional_baseline_comparison.py \
  --output-dir results/generated/conventional_baselines
```

It reconstructs the representative central PICMUS target near 37.5 mm and
both carotid views with DAS, CF-DAS, MV, DMAS, Receive-NSI and Angle-NSI. The
fixed baseline choices are:

- CF is evaluated over the active receive aperture for each transmit
  angle and applied to that complex DAS field before coherent compounding;
- MV uses all overlapping subarrays with
  `L=floor(M_active/2)`, diagonal loading `trace(R)/(100L)`, no temporal
  averaging for the point target, and a 1.5-wavelength axial half-window for
  the diffuse carotid data;
- DMAS uses delayed real RF, signed square-root pair products, a Kaiser FIR
  with stop/pass/pass/stop edges `(1.5, 1.75, 2.5, 2.75)*f0`, and Hilbert
  envelope detection. It is reconstructed at 0.02 mm axial spacing so the
  band near `2*f0` is below Nyquist.

Interrupted comparison runs reuse configuration- and input-validated caches
for the completed PSF and carotid cases. Use `--force` only to invalidate
those caches. A CPU-only input check is available with `--metadata-only`.

The common post-delay kernel benchmark is:

```bash
python src/benchmark_conventional.py \
  --warmups 10 \
  --repetitions 50 \
  --output-dir results/generated/conventional_timing
```

It records synchronized wall-clock measurements and leading-order complexity
for all six methods. Transfers, delay calculation, and interpolation are
excluded equally; MV covariance/solve and DMAS filtering/Hilbert formation
are included.

### Open-NSI MBTrace Doppler data

Place `MBTrace.mat` as described in `data/README.md`, then:

```bash
OPEN_NSI_MBTRACE_FILE=/path/to/MBTrace.mat \
NSI_OUTPUT_DIR=results/generated/doppler \
python src/doppler_mbtrace.py
```

The first run creates a neighboring `.npy` cache. The Doppler pipeline imports
the matched trace-width measurements from `src/trace_width_analysis.py`. It
reports one-to-one matching-tolerance curves and signed displacements as
positional concordance with DAS—not sensitivity, because DAS is not independent
ground truth. The original saved peak lists can be reanalysed without a GPU:

```bash
python src/reanalyse_mbtrace_peaks.py
```

### One-command revision run

Fetch and checksum-verify all public inputs, then run the complete revision
analysis on GPU 0:

```bash
python scripts/fetch_datasets.py --dataset all
python scripts/run_revision_gpu.py --device 0
```

The workflow writes a manifest after every step and reuses completed outputs on
restart. It includes the six-method simulation, full PICMUS phantom,
three-method MBTrace reconstruction, six-method carotid reconstruction and
both timing boundaries. Add `--force` to repeat every stage. Its final strict validation step writes
the publication tables, LaTeX result macros, and standardized figure files to
`results/generated/revision/manuscript_assets/`. It refuses metadata-only PSF
results, quick timing runs, incomplete benchmark cases, or fewer than 10
warmups and 50 repetitions. The existing long robustness suite is included
only when `--steps` explicitly contains `robustness`.

See [`docs/REVISION_HANDOFF.md`](docs/REVISION_HANDOFF.md) for the expected
outputs and the post-run validation checklist.

## Interpretation and provenance

- Reported widths are connected main-lobe widths at -6 dB in amplitude.
- The very narrow conventional-NSI value is explicitly described as an
  apparent nonlinear-output width, not a conventional linear-system PSF.
- Receive-NSI is implemented from two independent complex fields,
  `U` and `Z_e`; its two DC-offset fields are formed algebraically.
- Angle-NSI streaming and stored-stack reductions are numerically equivalent
  within the recorded complex64 tolerances.
- The raw `sign(theta)` convention uses weights `-1`, `0`, and `+1`; `c` is
  therefore defined relative to the unnormalized reference sum `U`.
- CF-DAS uses `|sum_m x_m|^2 / (M_active sum_m |x_m|^2)` over the active
  receive aperture for each transmit angle. It is ordinary CF, not a selected
  GCF spectral band.
- DMAS pair products are evaluated through an exact algebraic reduction
  that is unit-tested against the literal `i<j` double sum; it does not change
  the Matrone beamformer output. Independent lateral batches are validated
  line by line and recursively bisected if a CUDA/CuPy gather returns an empty
  batch; this changes only execution granularity, not the beamformer.
- External datasets are not redistributed. Their placement and provenance are
  documented under `data/`.

The committed CSV and JSON files under `results/reported/` are the values used
in the manuscript. New runs write to `results/generated/`, which is ignored by
Git so reported results cannot be overwritten accidentally.

## Citation and licenses

Source code is licensed under the MIT License ([`LICENSE`](LICENSE)).
Committed result records under `results/` are licensed under the Creative
Commons Attribution 4.0 International License
([`LICENSE-DATA`](LICENSE-DATA)). External datasets are not redistributed and
remain subject to their original terms.
Citation metadata are provided in `CITATION.cff`. 
If you find this code or method useful in your research, please cite the paper:

```bibtex
@article{leroy2026angledomain,
  title={Angle-domain null subtraction imaging from beamformed plane-wave data},
  author={Henri Leroy},
  journal={arXiv preprint arXiv:2608.18252},
  year={2026}
}
```

Please also cite the relevant SIMUS/MUST/pyMUST, MACH-beamform, Open-NSI and PICMUS publications and datasets when using these resources.
